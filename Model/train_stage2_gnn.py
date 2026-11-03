from __future__ import annotations

import argparse
import json
import math
import random
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Tuple
import os
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataclasses import dataclass, is_dataclass
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import polygonize, unary_union
from .config import SelfSupervisedConfig
from .data import GCGDataset, TextTokenizer
from .models import MultiModalSSLModel
from transformers import AutoTokenizer, BertModel


class FrozenHFTokenizer:
    """Wrapper around a HuggingFace tokenizer to keep it frozen for stage-2 usage."""

    def __init__(self, model_path: str, max_tokens: int):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True,
        )
        if self.hf_tokenizer.pad_token is None:
            if self.hf_tokenizer.unk_token is not None:
                self.hf_tokenizer.pad_token = self.hf_tokenizer.unk_token
            elif self.hf_tokenizer.eos_token is not None:
                self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
        self.vocab = self.hf_tokenizer.get_vocab()

    def encode(self, text: str | None):
        encoded = self.hf_tokenizer(
            text or "",
            max_length=self.max_tokens,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)

    def update(self, text: str) -> None:
        # The tokenizer is frozen; ignore update requests.
        return

# DEFAULTS = {}

FUNC_NAMES = ["build", "activity", "commercial_service", "industrial", "green_blue"]

OSM_MODE_LEGACY = "legacy"
OSM_MODE_OFF = "off"
OSM_MODE_NONROAD = "nonroad"
OSM_MODES = (OSM_MODE_LEGACY, OSM_MODE_OFF, OSM_MODE_NONROAD)
OSM_NONROAD_FEATURE_NAMES = [
    "osm_nonroad_shop_count",
    "osm_nonroad_amenity_count",
    "osm_nonroad_building_count",
    "osm_nonroad_building_area_m2",
    "osm_nonroad_landuse_residential_area_m2",
    "osm_nonroad_landuse_commercial_area_m2",
    "osm_nonroad_landuse_industrial_area_m2",
    "osm_nonroad_poi_mix_entropy",
]


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)

    if hasattr(torch, "use_deterministic_algorithms"):
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
        else:
            torch.use_deterministic_algorithms(False)


def collate_embeddings(batch):
    images = torch.stack([item["image"] for item in batch])
    grid_ids = torch.stack([item["grid_id"] for item in batch])
    caption_ids = torch.stack([item["caption_ids"] for item in batch])
    caption_mask = torch.stack([item["caption_mask"] for item in batch])
    mask_ratio = torch.stack([item["mask_ratio"] for item in batch])
    return {
        "images": images,
        "grid_ids": grid_ids,
        "caption_ids": caption_ids,
        "caption_mask": caption_mask,
        "gcg_feats": mask_ratio,
    }


def load_direction_mapping(path: str | None):
    """
    读取方向聚合映射：期望格式
    {
      "sat": {"build": [idx...], "activity": [...], ...},
      "street": {...}
    }
    """
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"direction mapping file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_category_order(cat_path: str) -> List[str]:
    """
    按 GCGDataset 的逻辑，类别顺序为大类名称的排序结果。
    这里直接从 sat_cat.json/street_cat.json 读取大类集合并排序，确保与 mask_ratio 对齐。
    """
    with open(cat_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bigs = set()
    for big, smalls in data.items():
        _ = smalls  # unused
        bigs.add(big)
    return sorted(bigs)


def _items_to_indices(items, cat_order: List[str]) -> List[int]:
    idxs = []
    for it in items:
        if isinstance(it, int):
            idxs.append(it)
        elif isinstance(it, str):
            if it in cat_order:
                idxs.append(cat_order.index(it))
            else:
                continue
    return idxs


def default_sign_prior(dir_mapping: dict | None):
    """
    基于相关性经验的符号先验：正值鼓励正权重，负值鼓励负权重，0 表示无约束。
    """
    if not dir_mapping:
        return None
    priors = []
    # 经验规则：sat: activity/绿蓝正，工业负；street: 活动/绿蓝负，工业略正，商业小正，建成度弱负
    sign_rules = {
        "sat": {"activity": 1.0, "绿蓝空间": 1.0, "green": 1.0, "industrial": -1.0, "工业度": -1.0, "build": 0.0, "商业服务": 0.0},
        "street": {"activity": -1.0, "绿蓝空间": -1.0, "green": -1.0, "industrial": 0.5, "工业度": 0.5, "商业服务": 0.3, "build": -0.2},
    }
    for modality in ("sat", "street"):
        for dir_name in dir_mapping.get(modality, {}):
            rule = sign_rules.get(modality, {})
            priors.append(torch.tensor(rule.get(dir_name, 0.0)))
    if not priors:
        return None
    return torch.stack(priors)


def load_func_prior(path: str | None, dim: int, func_dim: int = 5) -> torch.Tensor:
    """
    从 JSON 加载概念→功能先验矩阵。
    期望格式：{ "0": [w1,...,w5], "12": [...], "concept_name": [...] }，未列出的概念填 0。
    """
    mat = torch.zeros(dim, func_dim)
    if not path or not os.path.exists(path):
        return mat
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for k, v in data.items():
        try:
            idx = int(k)
        except ValueError:
            continue
        if 0 <= idx < dim and isinstance(v, (list, tuple)) and len(v) == func_dim:
            mat[idx] = torch.tensor(v, dtype=torch.float32)
    return mat


def load_ssl_model(ckpt_path: str, device: torch.device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg_obj = checkpoint["config"]
    if is_dataclass(cfg_obj):
        ssl_cfg = cfg_obj
    elif isinstance(cfg_obj, dict):
        ssl_cfg = SelfSupervisedConfig()
        for section in ("data", "model", "loss", "optim"):
            for key, value in cfg_obj[section].items():
                setattr(getattr(ssl_cfg, section), key, value)
        for key in ("log_interval", "ckpt_interval", "output_dir"):
            if key in cfg_obj:
                setattr(ssl_cfg, key, cfg_obj[key])
    else:
        raise TypeError("Unsupported config type in checkpoint.")
    state_dict = checkpoint["model"]
    if "text_encoder.embedding.weight" in state_dict:
        vocab_size = state_dict["text_encoder.embedding.weight"].shape[0]
    else:
        vocab_size = ssl_cfg.model.vocab_size
    model = MultiModalSSLModel(
        ssl_cfg.model,
        use_sat=ssl_cfg.data.include_satellite,
        use_street=ssl_cfg.data.include_street,
        vocab_size=vocab_size,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, ssl_cfg


def gather_embeddings(
    model: MultiModalSSLModel,
    dataset: GCGDataset,
    modality: str,
    device: torch.device,
    text_encoder: BertModel | None,
) -> Dict[int, List[dict]]:
    loader = DataLoader(
        dataset,
        batch_size=dataset.cfg.batch_size,
        shuffle=False,
        num_workers=dataset.cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_embeddings,
    )
    '''
    用于离线提取stage1自监督模型在某个模态上的所有图像、文本特征。
    并按照grid id进行聚合
    以供stage2使用
    '''
    results: Dict[int, List[dict]] = defaultdict(list)
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Encode {modality}", dynamic_ncols=True):
            images = batch["images"].to(device)
            caption_ids = batch["caption_ids"].to(device)
            caption_mask = batch["caption_mask"].to(device)
            gcg_feats = batch["gcg_feats"].to(device)
            img_out = model.encode_image(images, modality)
            ####
            # text_out = model.encode_text(caption_ids, caption_mask)
            # txt_feats = text_out["features"].cpu().numpy()
            ###
            if text_encoder is not None:
                text_outputs = text_encoder(input_ids=caption_ids, attention_mask=caption_mask)
                pooled = (
                    text_outputs.pooler_output
                    if text_outputs.pooler_output is not None
                    else text_outputs.last_hidden_state[:, 0]
                )
                txt_feats = pooled.cpu().numpy()
            else:
                txt_feats = np.zeros((caption_ids.size(0), 1), dtype=np.float32)
            img_feats = img_out["features"].cpu().numpy()
            gcg_np = gcg_feats.cpu().numpy()
            for idx, gid in enumerate(batch["grid_ids"].cpu().numpy().tolist()):
                results[int(gid)].append(
                    {
                        "image_feat": img_feats[idx],
                        "text_feat": txt_feats[idx],
                        "gcg_feat": gcg_np[idx],
                    }
                )
    return results


def compute_grid_bounds(merged_path: str, margin: float = 0.0005):
    with open(merged_path, "r", encoding="utf-8") as f:
        grids = json.load(f)
    bounds = {}
    for entry in grids:
        gid = int(entry["grid_id"])
        lats = []
        lons = []
        for group in ("sat_images", "street_images"):
            for img in entry.get(group, []):
                if img.get("lat") is None or img.get("lon") is None:
                    continue
                lats.append(float(img["lat"]))
                lons.append(float(img["lon"]))
        if not lats or not lons:
            continue
        lat_min = min(lats) - margin
        lat_max = max(lats) + margin
        lon_min = min(lons) - margin
        lon_max = max(lons) + margin
        bounds[gid] = {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        }
    print("一共数据有：：",len(bounds))
    return bounds


def build_bucket_index(bounds: Dict[int, Dict[str, float]], bucket_size: float = 0.02):
    bucket_map: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for gid, b in bounds.items():
        lat_start = int(math.floor(b["lat_min"] / bucket_size))
        lat_end = int(math.floor(b["lat_max"] / bucket_size))
        lon_start = int(math.floor(b["lon_min"] / bucket_size))
        lon_end = int(math.floor(b["lon_max"] / bucket_size))
        for la in range(lat_start, lat_end + 1):
            for lo in range(lon_start, lon_end + 1):
                bucket_map[(la, lo)].append(gid)
    return bucket_map, bucket_size


def _parse_osm_tags(elem: ET.Element) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    for tag in elem.findall("tag"):
        k = tag.get("k")
        v = tag.get("v")
        if k is None or v is None:
            continue
        tags[k] = v
    return tags


def _landuse_bucket(tags: Dict[str, str]) -> str | None:
    value = str(tags.get("landuse", "")).strip().lower()
    if value in {"residential"}:
        return "residential"
    if value in {"commercial", "retail", "mixed_use"}:
        return "commercial"
    if value in {"industrial"}:
        return "industrial"
    return None


def _is_nonroad_feature(tags: Dict[str, str]) -> bool:
    return ("shop" in tags) or ("amenity" in tags) or ("building" in tags) or (_landuse_bucket(tags) is not None)


def _assign_grid_by_point(
    lat: float,
    lon: float,
    bounds: Dict[int, Dict[str, float]],
    bucket_map: Dict[Tuple[int, int], List[int]],
    bucket_size: float,
) -> int | None:
    key = (int(math.floor(lat / bucket_size)), int(math.floor(lon / bucket_size)))
    for gid in bucket_map.get(key, []):
        b = bounds[gid]
        if b["lat_min"] <= lat <= b["lat_max"] and b["lon_min"] <= lon <= b["lon_max"]:
            return gid
    return None


def _candidate_grids_for_bbox(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    bucket_map: Dict[Tuple[int, int], List[int]],
    bucket_size: float,
) -> List[int]:
    lat_start = int(math.floor(min_lat / bucket_size))
    lat_end = int(math.floor(max_lat / bucket_size))
    lon_start = int(math.floor(min_lon / bucket_size))
    lon_end = int(math.floor(max_lon / bucket_size))
    out = set()
    for la in range(lat_start, lat_end + 1):
        for lo in range(lon_start, lon_end + 1):
            out.update(bucket_map.get((la, lo), []))
    return list(out)


def _geometry_area_m2(geom) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    area_deg2 = float(geom.area)
    if area_deg2 <= 0:
        return 0.0
    lat = float(geom.centroid.y)
    meters_per_deg_lat = 110540.0
    meters_per_deg_lon = 111320.0 * max(math.cos(math.radians(lat)), 1e-4)
    return area_deg2 * meters_per_deg_lat * meters_per_deg_lon


def _iter_geom_grid_intersections(
    geom,
    grid_polygons: Dict[int, Polygon],
    bucket_map: Dict[Tuple[int, int], List[int]],
    bucket_size: float,
):
    if geom is None or geom.is_empty:
        return
    min_lon, min_lat, max_lon, max_lat = geom.bounds
    candidates = _candidate_grids_for_bbox(
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        bucket_map=bucket_map,
        bucket_size=bucket_size,
    )
    for gid in candidates:
        grid_poly = grid_polygons[gid]
        if not geom.intersects(grid_poly):
            continue
        inter = geom.intersection(grid_poly)
        if inter is None or inter.is_empty:
            continue
        yield gid, inter


def _build_way_geometry(refs: List[int], node_coords: Dict[int, Tuple[float, float]]):
    coords = []
    for ref in refs:
        node = node_coords.get(ref)
        if node is None:
            continue
        lat, lon = node
        coords.append((lon, lat))
    if len(coords) < 2:
        return None
    if coords[0] == coords[-1] and len(coords) >= 4:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly is None or poly.is_empty:
            return None
        return poly
    line = LineString(coords)
    if line.is_empty:
        return None
    return line


def _build_relation_geometry(
    members: List[Tuple[str, int, str]],
    way_nodes: Dict[int, List[int]],
    node_coords: Dict[int, Tuple[float, float]],
):
    polygons = []
    line_parts = []
    for member_type, ref_id, _role in members:
        if member_type != "way":
            continue
        refs = way_nodes.get(ref_id)
        if not refs:
            continue
        geom = _build_way_geometry(refs, node_coords)
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, Polygon):
            polygons.append(geom)
        elif isinstance(geom, LineString):
            line_parts.append(geom)
    if line_parts:
        try:
            polygons.extend(list(polygonize(line_parts)))
        except Exception:
            pass
    if not polygons:
        return None
    geom = unary_union(polygons)
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom is None or geom.is_empty:
        return None
    return geom


def _align_osm_df(
    osm_df: pd.DataFrame,
    feature_names: List[str],
    bounds: Dict[int, Dict[str, float]],
) -> pd.DataFrame:
    full = pd.DataFrame({"grid_id": sorted(bounds)})
    if not feature_names:
        return full
    keep_cols = ["grid_id"] + [c for c in feature_names if c in osm_df.columns]
    merged = full.merge(osm_df[keep_cols], on="grid_id", how="left")
    for col in feature_names:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0).astype(np.float32)
    merged["grid_id"] = merged["grid_id"].astype(int)
    return merged[["grid_id"] + feature_names]


def extract_osm_features_legacy(osm_path: str, bounds: Dict[int, Dict[str, float]]):
    bucket_map, bucket_size = build_bucket_index(bounds)
    feature_names = ["osm_nodes", "osm_highway_nodes", "osm_amenity_nodes", "osm_building_nodes", "osm_shop_nodes"]
    features = {gid: np.zeros(len(feature_names), dtype=np.float32) for gid in bounds}

    for _, elem in ET.iterparse(osm_path, events=("end",)):
        if elem.tag != "node":
            elem.clear()
            continue
        lat = elem.get("lat")
        lon = elem.get("lon")
        if lat is None or lon is None:
            elem.clear()
            continue
        lat = float(lat)
        lon = float(lon)
        gid = _assign_grid_by_point(lat, lon, bounds, bucket_map, bucket_size)
        if gid is None:
            elem.clear()
            continue
        tags = _parse_osm_tags(elem)
        arr = features[gid]
        arr[0] += 1
        if "highway" in tags:
            arr[1] += 1
        if "amenity" in tags:
            arr[2] += 1
        if "building" in tags:
            arr[3] += 1
        if "shop" in tags:
            arr[4] += 1
        elem.clear()

    rows = []
    for gid in sorted(features):
        row = {"grid_id": gid}
        for name, value in zip(feature_names, features[gid]):
            row[name] = value
        rows.append(row)
    return pd.DataFrame(rows), feature_names


def extract_osm_features_nonroad(osm_path: str, bounds: Dict[int, Dict[str, float]]):
    bucket_map, bucket_size = build_bucket_index(bounds)
    grid_polygons = {
        gid: box(b["lon_min"], b["lat_min"], b["lon_max"], b["lat_max"])
        for gid, b in bounds.items()
    }
    feature_names = list(OSM_NONROAD_FEATURE_NAMES)
    features = {gid: np.zeros(len(feature_names), dtype=np.float32) for gid in bounds}
    entropy_bins = {gid: np.zeros(6, dtype=np.float32) for gid in bounds}
    node_coords: Dict[int, Tuple[float, float]] = {}
    way_nodes: Dict[int, List[int]] = {}

    def add_counts(gid: int, tags: Dict[str, str]) -> None:
        arr = features[gid]
        bins = entropy_bins[gid]
        if "shop" in tags:
            arr[0] += 1.0
            bins[0] += 1.0
        if "amenity" in tags:
            arr[1] += 1.0
            bins[1] += 1.0
        if "building" in tags:
            arr[2] += 1.0
            bins[2] += 1.0
        bucket = _landuse_bucket(tags)
        if bucket == "residential":
            bins[3] += 1.0
        elif bucket == "commercial":
            bins[4] += 1.0
        elif bucket == "industrial":
            bins[5] += 1.0

    def add_area(geom, tags: Dict[str, str]) -> None:
        if geom is None or geom.is_empty:
            return
        if float(getattr(geom, "area", 0.0)) <= 0:
            return
        lu_bucket = _landuse_bucket(tags)
        for gid, inter in _iter_geom_grid_intersections(geom, grid_polygons, bucket_map, bucket_size):
            area_m2 = _geometry_area_m2(inter)
            if area_m2 <= 0:
                continue
            if "building" in tags:
                features[gid][3] += area_m2
            if lu_bucket == "residential":
                features[gid][4] += area_m2
            elif lu_bucket == "commercial":
                features[gid][5] += area_m2
            elif lu_bucket == "industrial":
                features[gid][6] += area_m2

    def process_geom(tags: Dict[str, str], geom) -> None:
        if geom is None or geom.is_empty or not _is_nonroad_feature(tags):
            return
        rep_pt = geom if isinstance(geom, Point) else geom.representative_point()
        gid = _assign_grid_by_point(float(rep_pt.y), float(rep_pt.x), bounds, bucket_map, bucket_size)
        if gid is not None:
            add_counts(gid, tags)
        add_area(geom, tags)

    for _, elem in ET.iterparse(osm_path, events=("end",)):
        if elem.tag == "node":
            node_id = elem.get("id")
            lat = elem.get("lat")
            lon = elem.get("lon")
            lat_f = None
            lon_f = None
            if node_id is not None and lat is not None and lon is not None:
                try:
                    nid = int(node_id)
                    lat_f = float(lat)
                    lon_f = float(lon)
                    node_coords[nid] = (lat_f, lon_f)
                except ValueError:
                    pass
            tags = _parse_osm_tags(elem)
            if tags and lat_f is not None and lon_f is not None:
                process_geom(tags, Point(lon_f, lat_f))
            elem.clear()
            continue

        if elem.tag == "way":
            way_id = elem.get("id")
            refs: List[int] = []
            for nd in elem.findall("nd"):
                ref = nd.get("ref")
                if ref is None:
                    continue
                try:
                    refs.append(int(ref))
                except ValueError:
                    continue
            if way_id is not None:
                try:
                    way_nodes[int(way_id)] = refs
                except ValueError:
                    pass
            tags = _parse_osm_tags(elem)
            if tags and _is_nonroad_feature(tags):
                process_geom(tags, _build_way_geometry(refs, node_coords))
            elem.clear()
            continue

        if elem.tag == "relation":
            tags = _parse_osm_tags(elem)
            if tags and _is_nonroad_feature(tags):
                members = []
                for member in elem.findall("member"):
                    member_type = member.get("type")
                    ref = member.get("ref")
                    role = str(member.get("role") or "").strip().lower()
                    if member_type != "way" or ref is None:
                        continue
                    try:
                        members.append(("way", int(ref), role))
                    except ValueError:
                        continue
                process_geom(tags, _build_relation_geometry(members, way_nodes, node_coords))
            elem.clear()
            continue

        elem.clear()

    for gid in sorted(features):
        bins = entropy_bins[gid]
        total = float(bins.sum())
        if total <= 0:
            features[gid][7] = 0.0
            continue
        probs = bins / total
        probs = probs[probs > 0]
        features[gid][7] = float(-(probs * np.log(probs)).sum())

    rows = []
    for gid in sorted(features):
        row = {"grid_id": gid}
        for name, value in zip(feature_names, features[gid]):
            row[name] = value
        rows.append(row)
    return pd.DataFrame(rows), feature_names


def extract_osm_features(
    osm_path: str,
    bounds: Dict[int, Dict[str, float]],
    osm_mode: str = OSM_MODE_NONROAD,
    osm_cache_path: str | None = None,
):
    if osm_mode not in OSM_MODES:
        raise ValueError(f"Unsupported osm_mode={osm_mode}, expected one of {OSM_MODES}.")
    if osm_mode == OSM_MODE_OFF:
        return pd.DataFrame({"grid_id": sorted(bounds)}), []

    expected_cols = (
        ["osm_nodes", "osm_highway_nodes", "osm_amenity_nodes", "osm_building_nodes", "osm_shop_nodes"]
        if osm_mode == OSM_MODE_LEGACY
        else list(OSM_NONROAD_FEATURE_NAMES)
    )
    if osm_cache_path and os.path.exists(osm_cache_path):
        try:
            cached = pd.read_csv(osm_cache_path)
            if "grid_id" in cached.columns and all(col in cached.columns for col in expected_cols):
                print(f"[osm] load cached features from: {osm_cache_path}")
                return _align_osm_df(cached, expected_cols, bounds), expected_cols
            print(f"[osm] cache missing required columns for mode={osm_mode}, rebuild: {osm_cache_path}")
        except Exception as exc:
            print(f"[osm] failed to read cache {osm_cache_path}, rebuild. err={exc}")

    if osm_mode == OSM_MODE_LEGACY:
        osm_df, feature_names = extract_osm_features_legacy(osm_path, bounds)
    else:
        osm_df, feature_names = extract_osm_features_nonroad(osm_path, bounds)
    osm_df = _align_osm_df(osm_df, feature_names, bounds)

    if osm_cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(osm_cache_path)), exist_ok=True)
        osm_df.to_csv(osm_cache_path, index=False)
        print(f"[osm] saved cache to: {osm_cache_path}")
    return osm_df, feature_names


def load_targets(gpkg_path: str, target_cols: List[str]):
    if gpkg_path.lower().endswith(".csv"):
        gdf = pd.read_csv(gpkg_path)
    else:
        gdf = gpd.read_file(gpkg_path)
    if "grid_id" not in gdf.columns:
        raise ValueError("grid_id column missing in target gpkg.")
    # 自适应过滤不存在的指标列（例如长沙只有少数经济指标）
    existing_targets = [c for c in target_cols if c in gdf.columns]
    missing_targets = [c for c in target_cols if c not in gdf.columns]
    if missing_targets:
        print(f"[警告] 指标列缺失，将跳过: {missing_targets}")
    if not existing_targets:
        raise ValueError(f"No target columns found in {gpkg_path}. Available: {gdf.columns.tolist()}")
    extra_cols = []
    if "county" in gdf.columns:
        extra_cols.append("county")
    df = gdf[["grid_id"] + extra_cols + existing_targets].copy()
    df["grid_id"] = df["grid_id"].astype(int)
    return df


def transform_targets(values: np.ndarray):
    mins = values.min(axis=0, keepdims=True)
    offset = np.where(mins <= 0, 1.0 - mins, 0.0)
    shifted = values + offset
    log_vals = np.log1p(np.clip(shifted, a_min=0.0, a_max=None))
    mean = log_vals.mean(axis=0, keepdims=True)
    std = log_vals.std(axis=0, keepdims=True) + 1e-6
    normalized = (log_vals - mean) / std
    return normalized, offset, mean, std


def inverse_transform_targets(values: np.ndarray, offset: np.ndarray, mean: np.ndarray, std: np.ndarray):
    log_vals = values * std + mean
    shifted = np.expm1(log_vals)
    return shifted - offset


def sanitize_feature_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"\s+", "_", name)
    name = name.replace("/", "_").replace("-", "_")
    name = name.replace("(", "").replace(")", "")
    name = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", "_", name)
    return name.strip("_")


def resolve_category_names(cat_path: str | None, dim: int, prefix: str) -> List[str]:
    if cat_path and os.path.exists(cat_path):
        try:
            names = [sanitize_feature_name(x) for x in load_category_order(cat_path)]
            if len(names) == dim:
                return names
        except Exception as exc:
            print(f"[warn] failed to load category names from {cat_path}: {exc}")
    return [f"{prefix}_{i:02d}" for i in range(dim)]


def make_split_labels(indices: Dict[str, List[int]], num_nodes: int) -> List[str]:
    labels = ["unused"] * num_nodes
    for split in ("train", "val", "test"):
        for idx in indices.get(f"{split}_idx", []):
            if 0 <= idx < num_nodes:
                labels[idx] = split
    return labels


def append_matrix_columns(df: pd.DataFrame, matrix: np.ndarray | None, col_names: List[str]):
    if matrix is None:
        return
    if matrix.ndim != 2 or matrix.shape[1] != len(col_names):
        raise ValueError(f"matrix shape {matrix.shape} does not match column names {len(col_names)}")
    for idx, col in enumerate(col_names):
        df[col] = matrix[:, idx]


def export_explainability_inputs(
    export_path: str,
    metadata_path: str,
    valid_grids: List[int],
    grid_counties: List[str],
    split_labels: List[str],
    centers: np.ndarray,
    target_cols: List[str],
    target_raw: np.ndarray,
    target_norm: np.ndarray,
    pred_norm: np.ndarray,
    target_offset: np.ndarray,
    target_mean_arr: np.ndarray,
    target_std_arr: np.ndarray,
    osm_raw: np.ndarray,
    osm_norm: np.ndarray,
    osm_cols: List[str],
    sat_cat_names: List[str],
    street_cat_names: List[str],
    aux_data: Dict[str, torch.Tensor],
    modal_mode: str,
    osm_mode: str = OSM_MODE_NONROAD,
    train_seed: int | None = None,
    split_seed: int | None = None,
    deterministic: bool = False,
):
    os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
    pred_raw = inverse_transform_targets(pred_norm, target_offset, target_mean_arr, target_std_arr)

    df = pd.DataFrame(
        {
            "grid_id": valid_grids,
            "county": grid_counties,
            "split": split_labels,
            "center_lat": centers[:, 0],
            "center_lon": centers[:, 1],
            "modal_mode": modal_mode,
            "train_seed": int(train_seed) if train_seed is not None else -1,
            "split_seed": int(split_seed) if split_seed is not None else -1,
            "deterministic": int(bool(deterministic)),
            "osm_mode": osm_mode,
        }
    )

    for idx, col in enumerate(target_cols):
        raw_name = sanitize_feature_name(col)
        df[f"y_true_raw__{raw_name}"] = target_raw[:, idx]
        df[f"y_pred_raw__{raw_name}"] = pred_raw[:, idx]
        df[f"residual_raw__{raw_name}"] = pred_raw[:, idx] - target_raw[:, idx]
        df[f"y_true_norm__{raw_name}"] = target_norm[:, idx]
        df[f"y_pred_norm__{raw_name}"] = pred_norm[:, idx]
        df[f"residual_norm__{raw_name}"] = pred_norm[:, idx] - target_norm[:, idx]

    append_matrix_columns(df, osm_raw, [f"osm_raw__{sanitize_feature_name(c)}" for c in osm_cols])
    append_matrix_columns(df, osm_norm, [f"osm_norm__{sanitize_feature_name(c)}" for c in osm_cols])

    sat_gcg_cols = [f"sat_gcg__{name}" for name in sat_cat_names]
    street_gcg_cols = [f"street_gcg__{name}" for name in street_cat_names]
    append_matrix_columns(df, aux_data.get("sat_gcg"), sat_gcg_cols)
    append_matrix_columns(df, aux_data.get("street_gcg"), street_gcg_cols)

    append_matrix_columns(df, aux_data.get("sat_func"), [f"sat_func__{name}" for name in FUNC_NAMES])
    append_matrix_columns(df, aux_data.get("street_func"), [f"street_func__{name}" for name in FUNC_NAMES])
    append_matrix_columns(df, aux_data.get("fused_func"), [f"fused_func__{name}" for name in FUNC_NAMES])
    append_matrix_columns(
        df,
        aux_data.get("sat_func_evidence"),
        [f"sat_func_evidence__{name}" for name in FUNC_NAMES],
    )
    append_matrix_columns(
        df,
        aux_data.get("street_func_evidence"),
        [f"street_func_evidence__{name}" for name in FUNC_NAMES],
    )
    append_matrix_columns(
        df,
        aux_data.get("fused_func_evidence"),
        [f"fused_func_evidence__{name}" for name in FUNC_NAMES],
    )

    fused_visual = aux_data.get("fused_visual")
    hidden = aux_data.get("hidden")
    if fused_visual is not None:
        append_matrix_columns(
            df,
            fused_visual,
            [f"fused_visual_{i:03d}" for i in range(fused_visual.shape[1])],
        )
    if hidden is not None:
        append_matrix_columns(
            df,
            hidden,
            [f"hidden_{i:03d}" for i in range(hidden.shape[1])],
        )

    df.to_csv(export_path, index=False)

    metadata = {
        "modal_mode": modal_mode,
        "osm_mode": osm_mode,
        "train_seed": int(train_seed) if train_seed is not None else None,
        "split_seed": int(split_seed) if split_seed is not None else None,
        "deterministic": bool(deterministic),
        "num_rows": int(len(df)),
        "target_cols": target_cols,
        "func_names": FUNC_NAMES,
        "osm_cols": osm_cols,
        "sat_cat_names": sat_cat_names,
        "street_cat_names": street_cat_names,
        "column_groups": {
            "targets_raw": [c for c in df.columns if c.startswith("y_true_raw__") or c.startswith("y_pred_raw__") or c.startswith("residual_raw__")],
            "targets_norm": [c for c in df.columns if c.startswith("y_true_norm__") or c.startswith("y_pred_norm__") or c.startswith("residual_norm__")],
            "osm_raw": [c for c in df.columns if c.startswith("osm_raw__")],
            "osm_norm": [c for c in df.columns if c.startswith("osm_norm__")],
            "sat_gcg": sat_gcg_cols,
            "street_gcg": street_gcg_cols,
            "sat_func": [f"sat_func__{name}" for name in FUNC_NAMES],
            "street_func": [f"street_func__{name}" for name in FUNC_NAMES],
            "fused_func": [f"fused_func__{name}" for name in FUNC_NAMES],
            "hidden": [c for c in df.columns if c.startswith("hidden_")],
            "fused_visual": [c for c in df.columns if c.startswith("fused_visual_")],
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


@dataclass
class ModalSequence:
    image_feats: torch.Tensor
    text_feats: torch.Tensor
    gcg_feats: torch.Tensor

    def to(self, device: torch.device) -> "ModalSequence":
        return ModalSequence(
            image_feats=self.image_feats.to(device),
            text_feats=self.text_feats.to(device),
            gcg_feats=self.gcg_feats.to(device),
        )


class CrossModalInjector(nn.Module):
    def __init__(self, image_dim: int, text_dim: int):
        super().__init__()
        hidden = max(64, image_dim // 2)
        self.query = nn.Linear(image_dim, hidden)
        self.key = nn.Linear(text_dim, hidden)
        self.value = nn.Linear(text_dim, image_dim)
        self.scale = hidden ** -0.5
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, image_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        q = self.query(image_feats)
        k = self.key(text_feats)
        scores = torch.sum(q * k, dim=-1, keepdim=True) * self.scale
        attn = torch.softmax(scores, dim=0)
        infused = self.value(text_feats) * attn
        weight = torch.sigmoid(self.gate)
        return image_feats + weight * infused


class FiLMConditioner(nn.Module):
    """
    用概念级证据映射到功能方向，再生成 FiLM 调制参数。
    concept_dim: 概念数 C
    func_prior: 形状 [C, F] 的先验矩阵（仅部分锚点概念非零），可微调
    """

    def __init__(
        self,
        concept_dim: int,
        feature_dim: int,
        func_prior: torch.Tensor,
        hidden: int = 128,
        func_temp: float = 1.0,
    ):
        super().__init__()
        self.func_delta = nn.Parameter(torch.zeros_like(func_prior))
        self.register_buffer("func_prior", func_prior.clone())
        self.func_temp = max(float(func_temp), 1e-6)
        func_dim = func_prior.size(1)
        self.gamma_mlp = nn.Sequential(
            nn.Linear(func_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(func_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        )

    def forward(self, concept_feats: torch.Tensor):
        # 功能证据：概念证据与先验+可学习增量相乘
        # self.func_prior是一个固定的概念→功能矩阵（形状 C×5），锚点概念有先验权重，其他概念为 0
        # func_delta 是同形状的可学习增量，训练时在先验基础上微调。
        # gamma_mlp、beta_mlp 是两个小 MLP，把功能向量映射到与视觉特征同维度的 γ、β。
        func_weight = self.func_prior + self.func_delta
        func_evidence = torch.matmul(concept_feats, func_weight)
        func_vec = torch.softmax(func_evidence / self.func_temp, dim=-1)
        gamma = self.gamma_mlp(func_vec)
        beta = self.beta_mlp(func_vec)
        return gamma, beta, func_vec, func_evidence


class ModalAttentionAggregator(nn.Module):
    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        gcg_dim: int,
        hidden: int = 128,
        use_text: bool = False,
        conditioner: FiLMConditioner | None = None,
    ):
        super().__init__()
        self.use_text = use_text  # 是否使用文本对比学习
        self.conditioner = conditioner  # 是否有FiLM调制器
        if use_text:
            proj_dim = max(image_dim, text_dim)
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, image_dim),
            )
            self.injector = CrossModalInjector(image_dim, image_dim)
            self.image_contrast_proj = nn.Sequential(
                nn.Linear(image_dim, image_dim),
                nn.ReLU(),
                nn.Linear(image_dim, image_dim),
            )
            self.text_contrast_proj = nn.Sequential(
                nn.Linear(text_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, image_dim),
            )
        else:
            self.text_proj = None
            self.injector = None
            self.image_contrast_proj = None
            self.text_contrast_proj = None
        self.output_dim = image_dim * 2 if use_text else image_dim
        self.scorer = nn.Sequential( # 是一个小的MLP 用来给每张图打权重分数
            nn.Linear(self.output_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.gcg_dim = gcg_dim

    def forward(self, image_feats: torch.Tensor, text_feats: torch.Tensor, gcg_feats: torch.Tensor):
        # 输入一张图的视觉特征、文本特征、gcg特征
        if self.conditioner is not None:
            gamma, beta, func_vec, func_evidence = self.conditioner(gcg_feats)
            image_feats = gamma * image_feats + beta
        else:
            func_vec = None
            func_evidence = None
        if self.injector is not None:
            projected_text = self.text_proj(text_feats)
            injected = self.injector(image_feats, projected_text)
            fused = torch.cat([injected, image_feats], dim=-1)
            contrast_text = self.text_contrast_proj(text_feats)
        else:
            fused = image_feats
            contrast_text = None
        contrast_img = self.image_contrast_proj(image_feats) if self.image_contrast_proj is not None else None
        logits = self.scorer(fused).squeeze(-1)
        weights = torch.softmax(logits, dim=0)
        agg_feat = torch.sum(weights.unsqueeze(-1) * fused, dim=0)
        agg_gcg = torch.sum(weights.unsqueeze(-1) * gcg_feats, dim=0)
        if func_vec is not None:
            agg_func = torch.sum(weights.unsqueeze(-1) * func_vec, dim=0)
            agg_func_evidence = torch.sum(weights.unsqueeze(-1) * func_evidence, dim=0)
        else:
            agg_func = None
            agg_func_evidence = None
        return agg_feat, agg_gcg, agg_func, agg_func_evidence, weights, contrast_img, contrast_text


class GCGHead(nn.Module):
    """
    方向级 GCG 贡献：线性变换（可解释、可稀疏），并提供符号/稀疏正则。
    """

    def __init__(self, dir_dim: int, target_cols: List[str]):
        super().__init__()
        self.dir_dim = dir_dim
        self.targets = target_cols
        self.weight = nn.Parameter(torch.zeros(len(target_cols), dir_dim))
        self.bias = nn.Parameter(torch.zeros(len(target_cols)))

    def forward(self, dir_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {}
        for i, name in enumerate(self.targets):
            out[name] = dir_feats @ self.weight[i] + self.bias[i]
        return out

    def regularization(self, sign_prior: torch.Tensor | None, l1_lambda: float, sign_lambda: float):
        """
        sign_prior: shape [dir_dim]，>0 期望正，<0 期望负，0 表示无约束。
        返回 reg_loss 张量。
        """
        reg = torch.tensor(0.0, device=self.weight.device)
        if l1_lambda > 0:
            reg = reg + l1_lambda * torch.norm(self.weight, p=1)
        if sign_lambda > 0 and sign_prior is not None:
            # 对每个目标的权重施加符号软约束：若 prior>0，惩罚负权；prior<0，惩罚正权
            prior = sign_prior.view(1, -1).to(self.weight.device)
            violation = torch.where(prior > 0, torch.clamp(-self.weight, min=0.0), torch.clamp(self.weight, min=0.0))
            reg = reg + sign_lambda * violation.abs().mean()
        return reg


class ModalFusion(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, feature_dim)
        self.bias = nn.Parameter(torch.zeros(feature_dim))
        self.context = nn.Parameter(torch.randn(feature_dim))

    def forward(self, sat_feat: torch.Tensor, street_feat: torch.Tensor):
        def score(vec: torch.Tensor):
            hidden = torch.relu(self.linear(vec) + self.bias)
            return torch.matmul(self.context, hidden)

        alpha_sat = score(sat_feat)
        alpha_street = score(street_feat)
        betas = torch.softmax(torch.stack([alpha_sat, alpha_street], dim=0), dim=0)
        fused = betas[0] * sat_feat + betas[1] * street_feat
        return fused, betas


class EdgeAwareGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.edge_proj = nn.Linear(2, edge_dim)
        self.attn = nn.Linear(out_dim * 2 + edge_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_feats: torch.Tensor,
        neighbors: torch.Tensor,
        sat_weights: torch.Tensor,
        street_weights: torch.Tensor,
    ) -> torch.Tensor:
        h = self.linear(node_feats)
        neigh_feats = h[neighbors]
        self_feats = h.unsqueeze(1).expand_as(neigh_feats)
        edge_attr = torch.stack([sat_weights, street_weights], dim=-1)
        edge_embed = self.edge_proj(edge_attr)
        attn_input = torch.cat([self_feats, neigh_feats, edge_embed], dim=-1)
        e = self.leakyrelu(self.attn(attn_input)).squeeze(-1)
        alpha = torch.softmax(e, dim=1)
        alpha = self.dropout(alpha)
        updated = torch.sum(alpha.unsqueeze(-1) * neigh_feats, dim=1)
        return updated


class VisualEconomicGNN(nn.Module):
    def __init__(
        self,
        sat_dim: int,
        street_dim: int,
        sat_text_dim: int,
        street_text_dim: int,
        sat_gcg_dim: int,
        street_gcg_dim: int,
        osm_dim: int,
        hidden_dim: int,
        target_cols: List[str],
        dropout: float,
        k_neighbors: int,
        use_osm_branch: bool = True,
        modal_mode: str = "both",
        use_text: bool = True,
        sat_func_prior: torch.Tensor | None = None,
        street_func_prior: torch.Tensor | None = None,
        graph_log_interval: int = 1,
        func_topk: int = 5,
        func_temp: float = 1.0,
        func_div_target_var: float = 0.0,
        gnn_layers: int = 2,
    ):
        super().__init__()
        self.modal_mode = modal_mode
        attn_hidden = max(hidden_dim // 2, 64)
        sat_prior = sat_func_prior if sat_func_prior is not None else torch.zeros(sat_gcg_dim, 5)
        street_prior = street_func_prior if street_func_prior is not None else torch.zeros(street_gcg_dim, 5)
        self.sat_conditioner = FiLMConditioner(sat_gcg_dim, sat_dim, sat_prior, func_temp=func_temp)
        self.street_conditioner = FiLMConditioner(street_gcg_dim, street_dim, street_prior, func_temp=func_temp)
        self.sat_agg = ModalAttentionAggregator(
            sat_dim, sat_text_dim, sat_gcg_dim, attn_hidden, use_text=use_text, conditioner=self.sat_conditioner
        )
        self.street_agg = ModalAttentionAggregator(
            street_dim, street_text_dim, street_gcg_dim, attn_hidden, use_text=use_text, conditioner=self.street_conditioner
        )
        sat_feat_dim = self.sat_agg.output_dim
        street_feat_dim = self.street_agg.output_dim
        if sat_feat_dim != street_feat_dim:
            raise ValueError("Satellite and street feature dimensions must match after aggregation.")
        self.fusion = ModalFusion(sat_feat_dim)
        if modal_mode == "street":
            vision_dim = street_feat_dim
        else:
            vision_dim = sat_feat_dim
        self.use_osm_branch = bool(use_osm_branch)
        if self.use_osm_branch and osm_dim <= 0:
            raise ValueError("use_osm_branch=True but osm_dim<=0.")
        if (not self.use_osm_branch) and osm_dim > 0:
            print(f"[model] use_osm_branch=False, ignore provided osm_dim={osm_dim}")
        node_input_dim = vision_dim + (osm_dim if self.use_osm_branch else 0)  # 不再把方向特征拼接到节点输入
        self.k_neighbors = k_neighbors
        self.input_proj = nn.Linear(node_input_dim, hidden_dim)
        if gnn_layers not in (0, 1, 2, 3):
            raise ValueError(f"gnn_layers must be 0, 1, 2 or 3, got {gnn_layers}")
        self.gnn_layers = int(gnn_layers)
        self.layers = nn.ModuleList(
            [EdgeAwareGATLayer(hidden_dim, hidden_dim, edge_dim=32, dropout=dropout)
             for _ in range(self.gnn_layers)]
        )
        self.heads = nn.ModuleDict({col: nn.Linear(hidden_dim, 1) for col in target_cols})
        self.dropout = nn.Dropout(dropout)
        self.contrast_temp = 0.07
        self.graph_log_interval = max(0, int(graph_log_interval))
        self._graph_call_count = 0
        self.func_topk = max(0, int(func_topk))
        self.func_div_target_var = max(0.0, float(func_div_target_var))

    def _fuse_modal_sequences(self, sequences: List[ModalSequence], aggregator: ModalAttentionAggregator):
        '''
        这一段是在将同一网格内的多张图像（某一模态）汇总成网格级特征，并收集聚合过程中需要的附加信息
        对网格内的每张图，用 aggregator 算出：
        这张图的融合视觉特征 feat。
        这张图的 GCG 概念向量 gcg。
        由概念映射出的 5 维功能向量 func_vec，以及 softmax 前的 func_evidence。
        这张图在网格内的注意力权重 weights。
        对比学习用的投影 contrast_img/contrast_text。
        -----
        把上述结果逐张收集，最后堆成张量：
        所有图的视觉特征堆成一个张量；
        所有图的 GCG 向量堆成一个张量；
        所有图的功能向量和功能证据堆成张量（没算出的用 0 填充）；
        注意力权重列表；
        对比特征列表。
        '''
        
        modal_feats = []
        gcg_feats = []
        func_feats = []
        func_evidence_feats = []
        attn_weights = []
        contrast_pairs = []
        for seq in sequences:
            feat, gcg, func_vec, func_evidence, weights, contrast_img, contrast_text = aggregator(
                seq.image_feats, seq.text_feats, seq.gcg_feats
            )
            modal_feats.append(feat)
            gcg_feats.append(gcg)
            func_feats.append(func_vec if func_vec is not None else torch.zeros(5, device=feat.device))
            func_evidence_feats.append(
                func_evidence if func_evidence is not None else torch.zeros(5, device=feat.device)
            )
            attn_weights.append(weights)
            contrast_pairs.append((contrast_img, contrast_text))
        return (
            torch.stack(modal_feats),
            torch.stack(gcg_feats),
            torch.stack(func_feats),
            torch.stack(func_evidence_feats),
            attn_weights,
            contrast_pairs,
        )

    def _build_graph(self, graph_func_feats: torch.Tensor):
        """
        构建混合图：空间近邻 + 功能相似 top-k。
        返回 neighbors (N,k), spatial_flags (N,k), func_sims (N,k)
        """
        num_nodes = graph_func_feats.size(0)
        # 计算空间邻居（用中心点欧式距离，取8个最近邻）
        coords = self.grid_centers  # [N,2]
        dists = torch.cdist(coords, coords)
        spatial_neighbors = torch.topk(
            dists + torch.eye(num_nodes, device=dists.device) * 1e6,
            k=min(9, num_nodes),
            largest=False,
        ).indices[:, 1:]  # 排除自己

        # 功能相似矩阵（用于统计和 top-k 选边）
        norm = F.normalize(graph_func_feats, dim=1, p=2)
        sim = torch.matmul(norm, norm.t())
        sim.fill_diagonal_(0.0)

        func_topk = min(self.func_topk, max(0, num_nodes - 1))
        if func_topk > 0:
            sim_for_topk = sim.clone()
            sim_for_topk.fill_diagonal_(-1e9)
            top_vals, top_idx = torch.topk(sim_for_topk, k=func_topk, dim=1, largest=True, sorted=True)
            func_edges = [top_idx[i] for i in range(num_nodes)]
            func_weights = [top_vals[i] for i in range(num_nodes)]
            candidate_func_edges = int(num_nodes * func_topk)
        else:
            top_vals = torch.empty((num_nodes, 0), device=sim.device, dtype=sim.dtype)
            func_edges = [torch.empty(0, device=sim.device, dtype=torch.long) for _ in range(num_nodes)]
            func_weights = [torch.empty(0, device=sim.device, dtype=sim.dtype) for _ in range(num_nodes)]
            candidate_func_edges = 0

        # 合并空间与功能边，限制最多 k_neighbors
        k = min(self.k_neighbors, num_nodes - 1) if num_nodes > 1 else 1
        neigh_list = []
        flag_list = []
        func_list = []
        for i in range(num_nodes):
            neigh = spatial_neighbors[i].tolist()
            flag = [1.0] * len(neigh)  # 空间边标记为 1
            fsim = [0.0] * len(neigh)
            # 加功能边（避免与空间边重复）
            for idx, w in zip(func_edges[i].tolist(), func_weights[i].tolist()):
                if idx not in neigh:
                    neigh.append(idx)
                    flag.append(0.0)  # 功能边标记为 0
                    fsim.append(float(w))
            # 截断
            if len(neigh) > k:
                neigh = neigh[:k]
                flag = flag[:k]
                fsim = fsim[:k]
            # pad
            while len(neigh) < k:
                neigh.append(i)
                flag.append(0.0)
                fsim.append(0.0)
            neigh_list.append(torch.tensor(neigh, device=graph_func_feats.device, dtype=torch.long))
            flag_list.append(torch.tensor(flag, device=graph_func_feats.device, dtype=graph_func_feats.dtype))
            func_list.append(torch.tensor(fsim, device=graph_func_feats.device, dtype=graph_func_feats.dtype))

        neighbors = torch.stack(neigh_list)
        spatial_flags = torch.stack(flag_list)
        func_sims = torch.stack(func_list)
        selected_func_edges = int((func_sims > 0).sum().item())

        # 相似度分布统计（去掉对角线）
        sim_no_diag = sim[~torch.eye(num_nodes, dtype=torch.bool, device=sim.device)]
        if sim_no_diag.numel() > 0:
            sim_min = float(sim_no_diag.min().item())
            sim_mean = float(sim_no_diag.mean().item())
            sim_p95 = float(torch.quantile(sim_no_diag, 0.95).item())
        else:
            sim_min = sim_mean = sim_p95 = 0.0

        self._graph_call_count += 1
        if self.graph_log_interval > 0 and self._graph_call_count % self.graph_log_interval == 0:
            avg_candidate = candidate_func_edges / max(num_nodes, 1)
            avg_selected = selected_func_edges / max(num_nodes, 1)
            print(
                f"[graph] call={self._graph_call_count} mode=topk func_topk={func_topk} "
                f"func_edges(candidate)={candidate_func_edges} ({avg_candidate:.2f}/node), "
                f"func_edges(selected)={selected_func_edges} ({avg_selected:.2f}/node), "
                f"k_neighbors={self.k_neighbors}"
            )
            print(f"[graph] sim(min/mean/p95)=({sim_min:.6f}/{sim_mean:.6f}/{sim_p95:.6f})")
            preview_nodes = min(3, num_nodes)
            for n in range(preview_nodes):
                vec = ", ".join(f"{v:.4f}" for v in graph_func_feats[n].detach().cpu().tolist())
                sims = ", ".join(f"{v:.4f}" for v in top_vals[n].detach().cpu().tolist())
                print(f"[graph] node={n} func_feat=[{vec}] top{func_topk}_sim=[{sims}]")

        return neighbors, spatial_flags, func_sims

    def _contrastive_loss(self, contrast_pairs: List[Tuple[torch.Tensor | None, torch.Tensor | None]], device: torch.device):
        losses = []
        for img_emb, txt_emb in contrast_pairs:
            if img_emb is None or txt_emb is None:
                continue
            n = min(img_emb.size(0), txt_emb.size(0))
            if n < 2:
                continue
            img = F.normalize(img_emb[:n], dim=-1)
            txt = F.normalize(txt_emb[:n], dim=-1)
            logits = (img @ txt.T) / self.contrast_temp
            labels = torch.arange(n, device=device)
            loss_i = F.cross_entropy(logits, labels)
            loss_t = F.cross_entropy(logits.T, labels)
            losses.append(0.5 * (loss_i + loss_t))
        if not losses:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    def _func_diversity_loss(self, func_vecs: torch.Tensor):
        if func_vecs is None or func_vecs.numel() == 0 or func_vecs.size(0) < 2:
            return torch.tensor(0.0, device=self.input_proj.weight.device)
        var_per_dim = torch.var(func_vecs, dim=0, unbiased=False)
        if self.func_div_target_var <= 0:
            return -var_per_dim.mean()
        return torch.relu(self.func_div_target_var - var_per_dim).mean()

    def forward(
        self,
        sat_seqs: List[ModalSequence],
        street_seqs: List[ModalSequence],
        osm_feats: torch.Tensor,
        return_aux: bool = False,
    ):
        sat_feats, sat_gcg, sat_func, sat_func_evidence, _, sat_contrast = self._fuse_modal_sequences(sat_seqs, self.sat_agg) # 对每个网格内部的多张图做加权聚合，得到：sat_feats/street_feats：网格级视觉表示；sat_gcg/street_gcg：用图像权重加权后的网格级 GCG 大类占比向量；sat_contrast/street_contrast：对比学习用的特征对。
        street_feats, street_gcg, street_func, street_func_evidence, _, street_contrast = self._fuse_modal_sequences(street_seqs, self.street_agg)
        # 功能向量：按模态融合
        if self.modal_mode == "sat":
            fused_nodes = sat_feats
            func_vecs = sat_func
            graph_func_feats = sat_func_evidence
        elif self.modal_mode == "street":
            fused_nodes = street_feats
            func_vecs = street_func
            graph_func_feats = street_func_evidence
        else:
            fused_nodes = []
            for s_feat, t_feat in zip(sat_feats, street_feats):
                fused, _ = self.fusion(s_feat, t_feat)
                fused_nodes.append(fused)
            fused_nodes = torch.stack(fused_nodes)
            if sat_func is not None and street_func is not None:
                func_vecs = 0.5 * (sat_func + street_func)
                graph_func_feats = 0.5 * (sat_func_evidence + street_func_evidence)
            else:
                func_vecs = sat_func if sat_func is not None else street_func
                graph_func_feats = sat_func_evidence if sat_func_evidence is not None else street_func_evidence
        if self.use_osm_branch:
            node_inputs = torch.cat([fused_nodes, osm_feats], dim=-1)
        else:
            node_inputs = fused_nodes
        x = self.input_proj(node_inputs)
        neighbors, sat_weights, street_weights = self._build_graph(graph_func_feats)

        # 进入GNN循环==========
        for layer in self.layers:
            updated = layer(x, neighbors, sat_weights, street_weights)
            x = F.elu(updated + x)
            x = self.dropout(x)
        outputs = {name: head(x).squeeze(-1) for name, head in self.heads.items()}
        contrast_loss = self._contrastive_loss(sat_contrast + street_contrast, node_inputs.device)
        func_div_loss = self._func_diversity_loss(func_vecs)
        if not return_aux:
            return outputs, contrast_loss, func_div_loss
        sat_gcg_out = sat_gcg.detach()
        street_gcg_out = street_gcg.detach()
        sat_func_out = sat_func.detach() if sat_func is not None else None
        street_func_out = street_func.detach() if street_func is not None else None
        sat_func_evidence_out = sat_func_evidence.detach() if sat_func_evidence is not None else None
        street_func_evidence_out = street_func_evidence.detach() if street_func_evidence is not None else None
        # 单模态导出时只保留真正进入预测路径的那一侧模态特征，避免后续解释分析误把未使用模态当成有效输入。
        if self.modal_mode == "sat":
            street_gcg_out = None
            street_func_out = None
            street_func_evidence_out = None
        elif self.modal_mode == "street":
            sat_gcg_out = None
            sat_func_out = None
            sat_func_evidence_out = None
        aux = {
            "sat_gcg": sat_gcg_out,
            "street_gcg": street_gcg_out,
            "sat_func": sat_func_out,
            "street_func": street_func_out,
            "fused_func": func_vecs.detach() if func_vecs is not None else None,
            "sat_func_evidence": sat_func_evidence_out,
            "street_func_evidence": street_func_evidence_out,
            "fused_func_evidence": graph_func_feats.detach() if graph_func_feats is not None else None,
            "fused_visual": fused_nodes.detach(),
            "hidden": x.detach(),
        }
        return outputs, contrast_loss, func_div_loss, aux


def build_feature_matrix(
    grid_ids: List[int],
    sat_embs: Dict[int, List[dict]],
    street_embs: Dict[int, List[dict]],
    osm_df: pd.DataFrame,
) -> Tuple[List[ModalSequence], List[ModalSequence], torch.Tensor, List[str]]:
    osm_cols = [c for c in osm_df.columns if c != "grid_id"]
    osm_indexed = osm_df.set_index("grid_id") if "grid_id" in osm_df.columns else pd.DataFrame()
    osm_values = []
    sat_seqs: List[ModalSequence] = []
    street_seqs: List[ModalSequence] = []
    for gid in grid_ids:
        if gid not in osm_indexed.index:
            raise ValueError(f"OSM features missing for grid {gid}")
        if osm_cols:
            osm_values.append(osm_indexed.loc[gid, osm_cols].values.astype(np.float32))

        def build_sequence(items: List[dict]) -> ModalSequence:
            image_feats = torch.from_numpy(np.stack([entry["image_feat"] for entry in items]).astype(np.float32))
            text_feats = torch.from_numpy(np.stack([entry["text_feat"] for entry in items]).astype(np.float32))
            gcg_feats = torch.from_numpy(np.stack([entry["gcg_feat"] for entry in items]).astype(np.float32))
            return ModalSequence(image_feats=image_feats, text_feats=text_feats, gcg_feats=gcg_feats)

        sat_seqs.append(build_sequence(sat_embs[gid]))
        street_seqs.append(build_sequence(street_embs[gid]))

    if osm_cols:
        osm_tensor = torch.from_numpy(np.stack(osm_values))
    else:
        osm_tensor = torch.zeros((len(grid_ids), 0), dtype=torch.float32)
    return sat_seqs, street_seqs, osm_tensor, osm_cols


def normalize_osm_features(osm_tensor: torch.Tensor):
    if osm_tensor.ndim != 2:
        raise ValueError(f"osm_tensor must be rank-2, got shape={tuple(osm_tensor.shape)}")
    if osm_tensor.size(1) == 0:
        empty_stats = torch.zeros((1, 0), dtype=osm_tensor.dtype, device=osm_tensor.device)
        return osm_tensor.clone(), empty_stats, empty_stats
    vals = torch.log1p(osm_tensor)
    mean = vals.mean(dim=0, keepdim=True)
    std = vals.std(dim=0, keepdim=True) + 1e-6
    return (vals - mean) / std, mean, std


def split_by_ratio(num_nodes: int, train_ratio: float, val_ratio: float, seed: int):
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("train_ratio and val_ratio must be between 0 and 1 and sum to < 1.")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(num_nodes)
    train_end = int(num_nodes * train_ratio)
    val_end = int(num_nodes * (train_ratio + val_ratio))
    splits = {
        "train_idx": idx[:train_end].tolist(),
        "val_idx": idx[train_end:val_end].tolist(),
        "test_idx": idx[val_end:].tolist(),
    }
    splits["mode"] = "ratio"
    splits["seed"] = seed
    splits["train_ratio"] = train_ratio
    splits["val_ratio"] = val_ratio
    return splits


def split_by_county(grid_counties: List[str], seed: int):
    unique = sorted(set(grid_counties))
    if len(unique) < 3:
        raise ValueError("At least three counties are required to create train/val/test splits.")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(unique, size=2, replace=False)
    val_county, test_county = chosen[0], chosen[1]
    splits = {"train_idx": [], "val_idx": [], "test_idx": []}
    for idx, county in enumerate(grid_counties):
        if county == val_county:
            splits["val_idx"].append(idx)
        elif county == test_county:
            splits["test_idx"].append(idx)
        else:
            splits["train_idx"].append(idx)
    splits["val_county"] = val_county
    splits["test_county"] = test_county
    splits["mode"] = "county"
    splits["seed"] = seed
    return splits


def _kmeans(points: np.ndarray, k: int, seed: int, iters: int = 25) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centroids = points[rng.choice(len(points), size=k, replace=False)]
    for _ in range(iters):
        dists = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            mask = labels == i
            if mask.any():
                new_centroids[i] = points[mask].mean(axis=0)
            else:  # 如果某个簇为空，随机重置
                new_centroids[i] = points[rng.integers(0, len(points))]
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels


def split_by_region(grid_centers: np.ndarray, test_ratio: float, val_ratio: float, seed: int, region_count: int | None = None, viz_path: str | None = None):
    n = len(grid_centers)
    if n < 3:
        raise ValueError("Need at least 3 grids for region split.")
    if region_count is None:
        region_count = max(4, int(round(1 / min(test_ratio, val_ratio))))
    region_count = max(3, min(region_count, n))
    labels = _kmeans(grid_centers, region_count, seed)
    # 可视化
    if viz_path:
        plt.figure(figsize=(6, 6))
        scatter = plt.scatter(grid_centers[:, 1], grid_centers[:, 0], c=labels, cmap="tab20", s=6)
        plt.xlabel("lon")
        plt.ylabel("lat")
        plt.title("Region split by KMeans")
        plt.tight_layout()
        plt.savefig(viz_path, dpi=300)
        plt.close()
    # 按区域划分
    regions = {}
    for idx, lab in enumerate(labels):
        regions.setdefault(int(lab), []).append(idx)
    region_sizes = {r: len(idx_list) for r, idx_list in regions.items()}
    sorted_regions = sorted(regions.items(), key=lambda x: -len(x[1]))
    test_regions, val_regions, train_regions = [], [], []
    target_test = test_ratio * n
    target_val = val_ratio * n
    cum_test = 0
    cum_val = 0
    rng = np.random.default_rng(seed)
    for r, idx_list in sorted_regions:
        if cum_test < target_test:
            test_regions.append(r)
            cum_test += len(idx_list)
        elif cum_val < target_val:
            val_regions.append(r)
            cum_val += len(idx_list)
        else:
            train_regions.append(r)
    # 若某个 split 为空，做简单补救
    if not test_regions and sorted_regions:
        test_regions.append(sorted_regions[0][0])
    if not val_regions and len(sorted_regions) > 1:
        val_regions.append(sorted_regions[1][0])
    splits = {"train_idx": [], "val_idx": [], "test_idx": [], "mode": "region", "seed": seed, "region_count": region_count}
    for r, idx_list in regions.items():
        if r in test_regions:
            splits["test_idx"].extend(idx_list)
        elif r in val_regions:
            splits["val_idx"].extend(idx_list)
        else:
            splits["train_idx"].extend(idx_list)
    return splits, labels


def masks_from_indices(indices: Dict[str, List[int]], num_nodes: int, device: torch.device):
    masks = {}
    for name in ["train", "val", "test"]:
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        mask[indices[f"{name}_idx"]] = True
        masks[name] = mask
    return masks


def main():
    import os
    try:
        import debugpy
    except ImportError:
        debugpy = None
    def is_main_process():
            try:
                import torch.distributed as dist
                return not dist.is_initialized() or dist.get_rank() == 0
            except:
                return True  # 如果没有使用分布式，默认是主进程

    # ====== [插入debugpy监听器] ======
    if debugpy is not None and os.environ.get("DEBUGPY_WAIT_FOR_ATTACH") == "1" and is_main_process():
        PORT = int(os.environ.get("DEBUGPY_PORT", 5678))
        try:
            debugpy.listen(("0.0.0.0", PORT))
            print(f"[Debugpy] Waiting for debugger attach on port {PORT}...")
            debugpy.wait_for_client()
            print("[Debugpy] Debugger attached. Continue running...")
        except RuntimeError as e:
            print(f"[Debugpy] Could not listen on port {PORT}: {e}")


    print("=" * 50)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"PyTorch 可见 GPU 数量: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"当前 GPU: {torch.cuda.current_device()}")
    else:
        print("当前设备: CPU")
    print("=" * 50)
    # ====== [结束] ======

    parser = argparse.ArgumentParser(description="Stage 2: GNN for economic indicators.")
    parser.add_argument("--ssl_ckpt", type=str, required=True, help="Stage-1 checkpoint path.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--k_neighbors", type=int, default=12)
    parser.add_argument("--gnn_layers", type=int, choices=[0, 1, 2, 3], default=2,
                        help="Number of graph propagation layers for ablations.")
    parser.add_argument("--graph_log_interval", type=int, default=1, help="功能边统计日志频率（按构图调用次数），0 关闭。")
    parser.add_argument("--func_topk", type=int, default=4, help="每个节点按功能相似度选取的候选 top-k 功能边数。")
    parser.add_argument("--func_temp", type=float, default=0.15, help="func_vec softmax 温度，<1 更尖锐，>1 更平滑。")
    parser.add_argument("--func_div_weight", type=float, default=5.0, help="功能向量多样性正则权重，0 表示关闭。")
    parser.add_argument("--func_div_target_var", type=float, default=0.003, help="每个功能维度的最小方差目标，0 表示改用最大化方差。")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--split_path", type=str, required=True, help="region split json 输出/输入路径")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="区域划分验证集占比")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="区域划分测试集占比")
    parser.add_argument("--region_count", type=int, default=16, help="区域数量，缺省自动估计")
    parser.add_argument("--region_viz_path", type=str, required=True, help="region 可视化 png 路径")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--target_transform_fit_train",
        action="store_true",
        help="Fit target offset, log mean and std on training grids only (county holdout diagnostics).",
    )
    parser.add_argument("--seed", type=int, default=192, help="训练随机种子，控制参数初始化、dropout 等训练随机性")
    parser.add_argument("--deterministic", action="store_true", help="启用更强的确定性设置，便于多 seed 复现实验")
    parser.add_argument("--metrics_path", type=str, required=True, help="stage2 metrics csv 路径")
    parser.add_argument("--run_name", type=str, default="default_run", help="用于在stage2-checkpoint下创建子目录保存本次输出")
    parser.add_argument(
        "--modal_mode",
        type=str,
        default="both",
        choices=["both", "sat", "street"],
        help="选择只使用遥感(sat)、街景(street)或两者融合(both)进行消融实验。",
    )
    parser.add_argument(
        "--use_text_guidance",
        action="store_true",
        help="若设置则在模态聚合时使用 caption 文本作为指导；否则回退到仅图像信息。",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="GDP_2023_grid,Employment_2023_grid,Wage_2023_grid,Retail_2023_grid,Construction_area_grid,Finished_area_grid,Finished_value_grid,Total population",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=os.environ.get("TEXT_ENCODER_PATH", "bert-base-uncased"),
        help="若提供路径，则加载该预训练 tokenizer；留空可回退到简单分词器。",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=os.environ.get("TEXT_ENCODER_PATH", "bert-base-uncased"),
        help="用于提取 caption 表征的冻结 BERT 权重。",
    )
    parser.add_argument(
        "--text_contrast_weight",
        type=float,
        default=0.0,
        help="图像-文本对齐损失的权重，为 0 时不启用。",
    )
    parser.add_argument("--sat_func_prior", type=str, default=None, help="卫星概念→功能先验矩阵 JSON 路径。")
    parser.add_argument("--street_func_prior", type=str, default=None, help="街景概念→功能先验矩阵 JSON 路径。")
    parser.add_argument("--export_explain_inputs", action="store_true", help="若设置则在最佳 val checkpoint 更新时导出解释性输入。")
    parser.add_argument("--explain_inputs_path", type=str, default=None, help="解释性输入 CSV 输出路径，缺省为 metrics 同目录 explainability_inputs_best.csv。")
    parser.add_argument("--sat_cat_path", type=str, default=None, help="卫星大类定义 JSON，用于解释性导出列名。")
    parser.add_argument("--street_cat_path", type=str, default=None, help="街景大类定义 JSON，用于解释性导出列名。")
    parser.add_argument("--city", type=str, default="zhengzhou", help="切换城市数据")
    parser.add_argument("--merged_json", type=str, required=True, help="merged_by_grid 路径（外部脚本传入）")
    parser.add_argument("--osm_path", type=str, required=True, help="OSM 路径（外部脚本传入）")
    parser.add_argument(
        "--osm_mode",
        type=str,
        default=OSM_MODE_NONROAD,
        choices=list(OSM_MODES),
        help="OSM 特征模式：legacy(旧版节点计数) / off(关闭 OSM 分支) / nonroad(way+relation 非道路特征)。",
    )
    parser.add_argument(
        "--osm_cache_path",
        type=str,
        default=None,
        help="OSM 特征缓存 CSV 路径。设置后会优先读取缓存，不存在时构建并保存。",
    )
    parser.add_argument("--target_path", type=str, required=True, help="经济指标 GPKG/CSV 路径（外部脚本传入）")
    parser.add_argument("--sat_caption_path", type=str, default=None, help="卫星 caption json，可选，用于覆盖 stage1 配置")
    parser.add_argument("--street_caption_path", type=str, default=None, help="街景 caption json，可选，用于覆盖 stage1 配置")
    parser.add_argument("--sat_category_path", type=str, default=None, help="卫星概念类别映射 JSON")
    parser.add_argument("--street_category_path", type=str, default=None, help="街景概念类别映射 JSON")

    args = parser.parse_args()

    # 所有路径由外部脚本传入，避免代码中城市分支
    merged_json = args.merged_json
    osm_path = args.osm_path
    target_path = args.target_path
    split_path = args.split_path
    region_viz_path = args.region_viz_path
    metrics_path = args.metrics_path

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_global_seed(args.seed, deterministic=args.deterministic)
    print(f"[seed] train_seed={args.seed} split_seed={args.split_seed} deterministic={int(args.deterministic)}")
    stage1_model, ssl_cfg = load_ssl_model(args.ssl_ckpt, device)
    # 覆盖 stage1 数据配置，避免继续使用其他城市的数据
    data_cfg = getattr(ssl_cfg, "data", None)
    if data_cfg is not None:
        if hasattr(data_cfg, "merged_json_path"):
            data_cfg.merged_json_path = merged_json
        if args.sat_category_path:
            data_cfg.sat_category_path = args.sat_category_path
        if args.street_category_path:
            data_cfg.street_category_path = args.street_category_path
        if args.sat_caption_path and hasattr(data_cfg, "sat_caption_path"):
            data_cfg.sat_caption_path = args.sat_caption_path
        if args.street_caption_path and hasattr(data_cfg, "street_caption_path"):
            data_cfg.street_caption_path = args.street_caption_path
        if hasattr(data_cfg, "sat_rle_dir"):
            data_cfg.sat_rle_dir = None
        if hasattr(data_cfg, "street_rle_dir"):
            data_cfg.street_rle_dir = None

    print("[city]", args.city)
    print("[paths] merged_json:", merged_json)
    print("[paths] osm_path:", osm_path)
    print("[paths] target_path:", target_path)
    print("[paths] split_path:", split_path)
    print("[paths] region_viz_path:", region_viz_path)
    print("[paths] metrics_path:", metrics_path)
    print("[osm] mode:", args.osm_mode)
    print("[osm] cache:", args.osm_cache_path if args.osm_cache_path else "None")
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    text_encoder = BertModel.from_pretrained(args.text_encoder_path)
    text_encoder.to(device)
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad_(False)

    if args.tokenizer_path:
        tokenizer = FrozenHFTokenizer(args.tokenizer_path, ssl_cfg.data.max_tokens)
    else:
        tokenizer = TextTokenizer(max_tokens=ssl_cfg.data.max_tokens)
    sat_dataset = GCGDataset(ssl_cfg.data, modality="sat", tokenizer=tokenizer, build_vocab=False)
    street_dataset = GCGDataset(ssl_cfg.data, modality="street", tokenizer=tokenizer, build_vocab=False)


    sat_embeddings = gather_embeddings(stage1_model, sat_dataset, "sat", device, text_encoder) # len=2314
    street_embeddings = gather_embeddings(stage1_model, street_dataset, "street", device, text_encoder)# len=2314
    # 根据概念维度加载先验矩阵（未提供则为零）
    sample_gid = next(iter(sat_embeddings)) # 1224
    concept_dim_sat = len(sat_embeddings[sample_gid][0]["gcg_feat"])  # 20
    sample_gid_st = next(iter(street_embeddings))  # 1224
    concept_dim_st = len(street_embeddings[sample_gid_st][0]["gcg_feat"]) # 25
    sat_prior = load_func_prior(args.sat_func_prior, concept_dim_sat, func_dim=5) # (20,5)
    street_prior = load_func_prior(args.street_func_prior, concept_dim_st, func_dim=5) # (25,5)

    common_grids = sorted(set(sat_embeddings.keys()) & set(street_embeddings.keys()))  # len=2314
    if not common_grids:
        raise RuntimeError("No grids have both satellite and street embeddings.")

    bounds = compute_grid_bounds(merged_json)
    osm_df, osm_cols = extract_osm_features(
        osm_path=osm_path,
        bounds=bounds,
        osm_mode=args.osm_mode,
        osm_cache_path=args.osm_cache_path,
    )
    use_osm_branch = args.osm_mode != OSM_MODE_OFF
    print(f"[osm] extracted cols={len(osm_cols)} use_osm_branch={int(use_osm_branch)}")
    if osm_cols:
        print(f"[osm] columns: {osm_cols}")

    target_cols = [c.strip() for c in args.targets.split(",") if c.strip()]
    target_df = load_targets(target_path, target_cols)
    # 若部分指标缺失，load_targets 已过滤；此处同步更新 target_cols，避免后续索引缺失
    target_cols = [c for c in target_df.columns if c not in ("grid_id", "county", "geometry")]

    valid_grids = [gid for gid in common_grids if gid in osm_df["grid_id"].values and gid in target_df["grid_id"].values]
    sat_seqs, street_seqs, osm_tensor, osm_cols = build_feature_matrix(valid_grids, sat_embeddings, street_embeddings, osm_df)
    osm_raw_tensor = osm_tensor.clone()
    osm_tensor, osm_mean, osm_std = normalize_osm_features(osm_tensor)
    sat_seqs = [seq.to(device) for seq in sat_seqs]
    street_seqs = [seq.to(device) for seq in street_seqs]
    osm_tensor = osm_tensor.to(device)
    if use_osm_branch and osm_tensor.size(-1) == 0:
        raise ValueError("osm_mode requires OSM branch but no OSM columns were extracted.")

    target_indexed = target_df.set_index("grid_id")
    target_values = []
    grid_counties = []
    has_county = "county" in target_indexed.columns
    for gid in valid_grids:
        if gid not in target_indexed.index:
            raise ValueError(f"Missing target for grid {gid}")
        row = target_indexed.loc[gid]
        target_values.append(row[target_cols].values.astype(np.float32))
        grid_counties.append(str(row["county"]) if has_county else "unknown")
    target_values = np.stack(target_values)
    sat_cat_names = resolve_category_names(args.sat_cat_path, concept_dim_sat, "sat_cat")
    street_cat_names = resolve_category_names(args.street_cat_path, concept_dim_st, "street_cat")
    checkpoint_dir = os.path.dirname(os.path.abspath(metrics_path))
    explain_inputs_path = args.explain_inputs_path or os.path.join(checkpoint_dir, "explainability_inputs_best.csv")
    explain_metadata_path = os.path.splitext(explain_inputs_path)[0] + ".meta.json"
    # best 权重跟随 metrics_path，保存到 stage2_metrics.csv 同目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_r2 = -1e9
    best_ckpt_path = os.path.join(checkpoint_dir, "best.pt")

    gnn = VisualEconomicGNN(
        sat_dim=sat_seqs[0].image_feats.size(-1),
        street_dim=street_seqs[0].image_feats.size(-1),
        sat_text_dim=sat_seqs[0].text_feats.size(-1),
        street_text_dim=street_seqs[0].text_feats.size(-1),
        sat_gcg_dim=sat_seqs[0].gcg_feats.size(-1),
        street_gcg_dim=street_seqs[0].gcg_feats.size(-1),
        osm_dim=osm_tensor.size(-1),
        hidden_dim=args.hidden_dim,
        target_cols=target_cols,
        dropout=args.dropout,
        k_neighbors=args.k_neighbors,
        use_osm_branch=use_osm_branch,
        modal_mode=args.modal_mode,
        use_text=args.use_text_guidance,
        sat_func_prior=sat_prior,
        street_func_prior=street_prior,
        graph_log_interval=args.graph_log_interval,
        func_topk=args.func_topk,
        func_temp=args.func_temp,
        func_div_target_var=args.func_div_target_var,
        gnn_layers=args.gnn_layers,
    ).to(device)
    # 设置网格中心坐标用于区域划分/空间邻接（lat, lon），顺序与 valid_grids 对齐
    centers = []
    for gid in valid_grids:
        b = bounds.get(gid)
        if b:
            centers.append([(b["lat_min"] + b["lat_max"]) * 0.5, (b["lon_min"] + b["lon_max"]) * 0.5])
        else:
            centers.append([0.0, 0.0])
    gnn.grid_centers = torch.tensor(centers, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=args.lr)
    # 使用城市适配后的 split_path
    need_new_split = False
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            indices = json.load(f)
        total = sum(len(indices.get(f"{name}_idx", [])) for name in ["train", "val", "test"])
        mode_match = str(indices.get("mode", "")).startswith("region")
        if total == len(valid_grids) and mode_match:
            need_new_split = False
        if need_new_split:
            print("Existing split file incompatible with current dataset or mode. Regenerating splits.")
    if need_new_split:
        indices, labels = split_by_region(
            np.array(centers),
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            seed=args.split_seed,
            region_count=args.region_count,
            viz_path=region_viz_path,
        )
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(indices, f, ensure_ascii=False, indent=2)

    if args.target_transform_fit_train:
        train_values = target_values[indices["train_idx"]]
        _, target_offset, target_mean, target_std = transform_targets(train_values)
        target_norm = (
            np.log1p(np.clip(target_values + target_offset, a_min=0.0, a_max=None))
            - target_mean
        ) / target_std
    else:
        target_norm, target_offset, target_mean, target_std = transform_targets(target_values)
    target_tensor = torch.from_numpy(target_norm).to(device)
    masks = masks_from_indices(indices, len(valid_grids), device)

    metrics_history = []

    text_contrast_weight = args.text_contrast_weight
    func_div_weight = args.func_div_weight
    for epoch in range(1, args.epochs + 1):
        gnn.train()
        optimizer.zero_grad()
        outputs, contrast_loss, func_div_loss = gnn(sat_seqs, street_seqs, osm_tensor)
        loss = text_contrast_weight * contrast_loss + func_div_weight * func_div_loss
        for idx, col in enumerate(target_cols):
            pred = outputs[col]
            target = target_tensor[:, idx]
            loss = loss + F.mse_loss(pred[masks["train"]], target[masks["train"]])
        loss.backward()
        optimizer.step()

        if epoch % 1 == 0 or epoch == args.epochs:
            gnn.eval()
            with torch.no_grad():
                outputs, contrast_eval, func_div_eval = gnn(sat_seqs, street_seqs, osm_tensor)
                log_line = [
                    f"Epoch {epoch:04d}",
                    f"Train {loss.item():.4f}",
                    f"TxtContrast {contrast_eval.item():.4f}",
                    f"FuncDiv {func_div_eval.item():.4f}",
                ]
                epoch_records = []
                for split in ["val", "test"]:
                    split_metrics = []
                    for idx, col in enumerate(target_cols):
                        pred = outputs[col][masks[split]]
                        target = target_tensor[:, idx][masks[split]]
                        mse = F.mse_loss(pred, target).item()
                        rmse = math.sqrt(max(mse, 0.0))
                        target_mean_split = target.mean().item()
                        sst = torch.sum((target - target_mean_split) ** 2).item()
                        sse = torch.sum((pred - target) ** 2).item()
                        r2 = 1.0 - (sse / sst) if sst > 1e-6 else 1.0
                        metric_entry = {
                                "epoch": epoch,
                                "split": split,
                                "target": col,
                                "rmse": rmse,
                                "r2": r2,
                                "train_seed": args.seed,
                                "split_seed": args.split_seed,
                                "deterministic": bool(args.deterministic),
                            }
                        split_metrics.append(metric_entry)
                        epoch_records.append(
                            {
                                "epoch": epoch,
                                "split": split,
                                "target": col,
                                "rmse": rmse,
                                "r2": r2,
                                "train_seed": args.seed,
                                "split_seed": args.split_seed,
                                "deterministic": bool(args.deterministic),
                            }
                        )
                    best = max(split_metrics, key=lambda item: item["r2"]) if split_metrics else None
                    summary = " ".join(f"{col_dict['target']}:RMSE {col_dict['rmse']:.4f} R2 {col_dict['r2']:.4f}" for col_dict in split_metrics) if split_metrics else ""
                    log_line.append(f"{split.capitalize()} | {summary}")
                    if split == "val" and split_metrics:
                        val_r2_mean = sum(m["r2"] for m in split_metrics) / len(split_metrics)
                        if val_r2_mean > best_val_r2:
                            best_val_r2 = val_r2_mean
                            torch.save(
                                {
                                    "epoch": epoch,
                                    "model_state": gnn.state_dict(),
                                    "optimizer_state": optimizer.state_dict(),
                                    "val_metrics": split_metrics,
                                    "val_r2_mean": val_r2_mean,
                                    "target_offset": target_offset,
                                    "target_mean": target_mean,
                                    "target_std": target_std,
                                    "train_seed": args.seed,
                                    "split_seed": args.split_seed,
                                    "deterministic": bool(args.deterministic),
                                },
                                best_ckpt_path,
                            )
                            if args.export_explain_inputs:
                                best_out = gnn(sat_seqs, street_seqs, osm_tensor, return_aux=True)
                                best_outputs, _, _, best_aux = best_out
                                pred_norm = np.stack(
                                    [best_outputs[col].detach().cpu().numpy() for col in target_cols],
                                    axis=1,
                                )
                                aux_np = {}
                                for key, value in best_aux.items():
                                    aux_np[key] = value.cpu().numpy() if value is not None else None
                                export_explainability_inputs(
                                    export_path=explain_inputs_path,
                                    metadata_path=explain_metadata_path,
                                    valid_grids=valid_grids,
                                    grid_counties=grid_counties,
                                    split_labels=make_split_labels(indices, len(valid_grids)),
                                    centers=np.asarray(centers, dtype=np.float32),
                                    target_cols=target_cols,
                                    target_raw=target_values,
                                    target_norm=target_norm,
                                    pred_norm=pred_norm,
                                    target_offset=target_offset,
                                    target_mean_arr=target_mean,
                                    target_std_arr=target_std,
                                    osm_raw=osm_raw_tensor.cpu().numpy(),
                                    osm_norm=osm_tensor.detach().cpu().numpy(),
                                    osm_cols=osm_cols,
                                    sat_cat_names=sat_cat_names,
                                    street_cat_names=street_cat_names,
                                    aux_data=aux_np,
                                    modal_mode=args.modal_mode,
                                    osm_mode=args.osm_mode,
                                    train_seed=args.seed,
                                    split_seed=args.split_seed,
                                    deterministic=args.deterministic,
                                )
                                log_line.append(f"Saved explain inputs -> {explain_inputs_path}")
                            log_line.append(f"Saved best ckpt (val_r2_mean={val_r2_mean:.4f})")
                metrics_history.extend(epoch_records)
                # if gnn.gcg_head is not None:
                #     # 记录方向权重平均绝对值，便于跟踪可解释性
                #     weight_abs = gnn.gcg_head.weight.detach().abs().mean(dim=0).cpu().numpy()
                #     dir_stats = ", ".join(
                #         f"{name}:{val:.3f}" for name, val in zip(gnn.dir_names, weight_abs)
                #     )
                #     log_line.append(f"GCG|{dir_stats}")
                print(" | ".join(log_line))

    if metrics_history:
        pd.DataFrame(metrics_history).to_csv(metrics_path, index=False)


if __name__ == "__main__":
    main()
