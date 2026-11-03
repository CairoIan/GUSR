from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

try:
    from pycocotools import mask as mask_utils
except ImportError:  # pragma: no cover - optional dependency
    mask_utils = None

from ..config import DataConfig


def _load_rgb_image(path: str, retries: int, retry_delay: float) -> Image.Image:
    last_exc: OSError | None = None
    attempts = max(1, int(retries) + 1)
    delay = max(0.0, float(retry_delay))
    for attempt in range(attempts):
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except OSError as exc:
            last_exc = exc
            if attempt + 1 >= attempts:
                break
            time.sleep(delay)
    raise OSError(f"Failed to read image after {attempts} attempts: {path}; last_error={last_exc}") from last_exc


class CategoryMapper:
    """Maps fine-grained GCG labels/phases to coarse categories."""

    def __init__(self, mapping_path: str):
        with open(mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        small_to_big = {}
        for big, smalls in data.items():
            for name in smalls:
                small_to_big[name.lower()] = big
        self.small_to_big = small_to_big
        self.big_categories = sorted(set(small_to_big.values()))
        self.big_to_idx = {c: i for i, c in enumerate(self.big_categories)}

    def __len__(self) -> int:
        return len(self.big_categories)

    def lookup(self, label: str) -> Optional[str]:
        return self.small_to_big.get(label.lower().strip())


class TextTokenizer:
    """Lightweight tokenizer + vocab for the captions."""

    def __init__(self, max_tokens: int, vocab: Optional[Dict[str, int]] = None):
        self.max_tokens = max_tokens
        if vocab is None:
            vocab = {
                "<pad>": 0,
                "<unk>": 1,
                "<cls>": 2,
                "<sep>": 3,
            }
        self.vocab = vocab

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"[A-Za-z0-9]+|[.,!?;:%]", text.lower())

    def update(self, text: str) -> None:
        for token in self._tokenize(text):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self._tokenize(text)
        max_body = self.max_tokens - 2
        tokens = tokens[:max(0, max_body)]
        ids = [self.vocab["<cls>"]]
        for tok in tokens:
            ids.append(self.vocab.get(tok, self.vocab["<unk>"]))
        ids.append(self.vocab["<sep>"])
        pad_len = self.max_tokens - len(ids)
        if pad_len > 0:
            ids = ids + [self.vocab["<pad>"]] * pad_len
        attn = [1] * min(len(ids), self.max_tokens)
        if len(attn) < self.max_tokens:
            attn += [0] * (self.max_tokens - len(attn))
        return torch.tensor(ids[: self.max_tokens], dtype=torch.long), torch.tensor(attn[: self.max_tokens], dtype=torch.long)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _rle_area(mask_rle: dict) -> float:
    if mask_utils is None:
        raise ImportError("pycocotools is required to compute mask areas. Please install it before training.")
    rle = dict(mask_rle)
    counts = rle["counts"]
    if isinstance(counts, str):
        rle["counts"] = counts.encode("utf-8")
    return float(mask_utils.area(rle))


def _ratio_vector(length: int, values: Dict[int, float]) -> np.ndarray:
    vec = np.zeros(length, dtype=np.float32)
    for idx, val in values.items():
        vec[idx] = val
    return vec


def _default_transforms(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            # 禁用色相抖动，避免部分 PIL/numpy 版本在 adjust_hue 时出现 OverflowError
            T.ColorJitter(0.2, 0.2, 0.2, 0.0),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@dataclass
class SampleMeta:
    grid_id: int
    modality: Literal["sat", "street"]
    image_path: str
    ann_path: str


class GCGDataset(Dataset):
    """
    Torch dataset producing multimodal training samples for self-supervised learning.
    """

    def __init__(
        self,
        cfg: DataConfig,
        modality: Literal["sat", "street", "both"] = "both",
        tokenizer: Optional[TextTokenizer] = None,
        transforms: Optional[T.Compose] = None,
        cache_annotations: bool = True,
        build_vocab: bool = True,
    ):
        super().__init__()
        if transforms is None:
            transforms = _default_transforms(cfg.image_size)
        self.cfg = cfg
        self.transforms = transforms
        self.modality = modality
        self.cache_annotations = cache_annotations
        self.annotation_cache: Dict[str, dict] = {}
        self.samples: List[SampleMeta] = []

        self.cat_mappers = {
            "sat": CategoryMapper(cfg.sat_category_path),
            "street": CategoryMapper(cfg.street_category_path),
        }

        merged = _load_json(cfg.merged_json_path)
        manifest_dir = os.path.dirname(os.path.abspath(cfg.merged_json_path))

        def add_samples(images: List[dict], kind: Literal["sat", "street"], grid_id: int):
            for img in images:
                image_path = img.get("image") or img.get("img") or img.get("image_path")
                if image_path and not os.path.isabs(image_path):
                    image_path = os.path.join(manifest_dir, image_path)
                if not image_path or not os.path.exists(image_path):
                    continue
                ann_path = img.get("rle_path")
                if ann_path and not os.path.isabs(ann_path):
                    ann_path = os.path.join(manifest_dir, ann_path)
                if not ann_path or not os.path.exists(ann_path):
                    continue
                self.samples.append(SampleMeta(grid_id=grid_id, modality=kind, image_path=image_path, ann_path=ann_path))

        include_sat = cfg.include_satellite and modality in ("sat", "both")
        include_street = cfg.include_street and modality in ("street", "both")
        for grid in merged:
            grid_id = int(grid.get("grid_id"))
            if include_sat:
                add_samples(grid.get("sat_images", []), "sat", grid_id)
            if include_street:
                add_samples(grid.get("street_images", []), "street", grid_id)

        if not self.samples:
            raise RuntimeError("No samples found. Check dataset paths and modality flags.")

        build_now = build_vocab or tokenizer is None
        if tokenizer is None:
            tokenizer = TextTokenizer(max_tokens=cfg.max_tokens)
        if build_now:
            self._build_vocab(tokenizer)
        self.tokenizer = tokenizer

    def _build_vocab(self, tokenizer: TextTokenizer) -> None:
        for meta in self.samples:
            ann = self._get_annotation(meta.ann_path)
            text = ann.get("gcg_description") or ann.get("caption") or ""
            tokenizer.update(text)

    def _get_annotation(self, path: str) -> dict:
        if self.cache_annotations and path in self.annotation_cache:
            return self.annotation_cache[path]
        data = _load_json(path)
        if self.cache_annotations:
            self.annotation_cache[path] = data
        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta = self.samples[idx]
        ann = self._get_annotation(meta.ann_path)
        image = _load_rgb_image(
            meta.image_path,
            retries=getattr(self.cfg, "image_read_retries", 0),
            retry_delay=getattr(self.cfg, "image_read_retry_delay", 0.0),
        )  # 700*700
        image = self.transforms(image) # 3,224,224

        mask_vec, phrase_vec = self._extract_targets(ann, meta.modality)
        caption_text = ann.get("gcg_description") or ann.get("caption", "")
        caption_ids, caption_mask = self.tokenizer.encode(caption_text)

        return {
            "image": image,
            "caption_ids": caption_ids,
            "caption_mask": caption_mask,
            "mask_ratio": torch.from_numpy(mask_vec),
            "phrase_ratio": torch.from_numpy(phrase_vec),
            "grid_id": torch.tensor(meta.grid_id, dtype=torch.long),
            "modality": meta.modality,
            "caption_text": caption_text,
        }

    def _extract_targets(self, ann: dict, modality: Literal["sat", "street"]) -> Tuple[np.ndarray, np.ndarray]:
        '''
        从单张图像的标注ann中提取GCG目标项链，根据 modality（sat/street）选取对应的 CategoryMapper，把细粒度标签映射到大类，并记录大类总数。
        对每个 rle_masks 里的实例：用 mapper.lookup 找到大类，计算掩膜面积（优先 RLE，失败则回退 bbox 面积），累加到 area_accumulator。最后把各类别面积除以整张图像像素数 total_pixels，得到 mask_ratio——每个大类的像素占比。
        同样地，遍历 phrases（文本描述中的类别），统计每个大类出现次数并归一化为占比，得到 phrase_ratio。
        函数返回 (mask_ratio, phrase_ratio)，即该图像在不同大类上的像素分布和文本分布，用于后续模型训练。
        '''
        mapper = self.cat_mappers[modality]
        num_classes = len(mapper)
        width = ann.get("width")
        height = ann.get("height")
        total_pixels = max(1.0, float(width * height)) if width and height else 1.0

        area_accumulator: Dict[int, float] = {}
        for mask in ann.get("rle_masks", []):
            label = mask.get("label", "").lower()
            big = mapper.lookup(label)
            if not big:
                continue
            idx = mapper.big_to_idx[big]
            try:
                area = _rle_area(mask.get("mask_rle", {}))
            except Exception:
                # fall back to bbox approximation if decoding fails
                bbox = mask.get("bbox", [0, 0, 0, 0])
                area = max(0.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
            if area <= 0:
                continue
            area_accumulator[idx] = area_accumulator.get(idx, 0.0) + area

        mask_ratio = _ratio_vector(num_classes, {idx: val / total_pixels for idx, val in area_accumulator.items()})

        phrase_counter: Dict[int, float] = {}
        phrases = ann.get("phrases") or []
        for phrase in phrases:
            big = mapper.lookup(phrase)
            if not big:
                continue
            idx = mapper.big_to_idx[big]
            phrase_counter[idx] = phrase_counter.get(idx, 0.0) + 1.0
        total_phrases = sum(phrase_counter.values()) or 1.0
        phrase_ratio = _ratio_vector(num_classes, {idx: count / total_phrases for idx, count in phrase_counter.items()})

        return mask_ratio, phrase_ratio
