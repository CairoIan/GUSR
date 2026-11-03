from __future__ import annotations

import argparse  # 解析命令行参数
import json  # 读取/写入配置
import os  # 文件路径与目录操作
from dataclasses import is_dataclass, asdict  # 判断嵌套 dataclass 用于递归更新
from contextlib import nullcontext  # 在 CPU 下占位的空上下文
from typing import Dict, Iterator, Optional, List
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer

from .config import SelfSupervisedConfig  # 自监督阶段的统一配置对象
from .data import GCGDataset, TextTokenizer  # 数据集与分词器
from .models import MultiModalSSLModel  # 主模型
from .models.losses import LossOutput, contrastive_loss, gcg_intra_contrastive  # 自定义损失

def _update_dataclass(dst, src_dict):
    """递归地将字典内容写入 dataclass，用于载入 JSON 配置覆盖默认值。"""
    for key, value in src_dict.items():
        if not hasattr(dst, key):
            continue
        current = getattr(dst, key)
        if is_dataclass(current):
            _update_dataclass(current, value)
        else:
            setattr(dst, key, value)


def load_config(path: Optional[str]) -> SelfSupervisedConfig:
    """若提供路径则读取 JSON，生成 SelfSupervisedConfig 实例。"""
    cfg = SelfSupervisedConfig()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _update_dataclass(cfg, data)
    return cfg


POI_BASE_COLUMNS = {"grid_id", "kind", "osm_id", "lon", "lat"}
DEFAULT_POI_FUNCTIONAL_TAGS = [
    "amenity",
    "shop",
    "tourism",
    "leisure",
    "office",
    "healthcare",
    "public_transport",
    "railway",
    "highway",
    "aeroway",
    "landuse",
    "building",
    "industrial",
    "craft",
    "sport",
    "historic",
    "religion",
]


def collate_batch(batch):
    """自定义 collate，把同名字段堆叠成 batch tensor。"""
    tensors = {}
    for key in ("image", "caption_ids", "caption_mask", "mask_ratio", "phrase_ratio", "grid_id"):
        tensors[key] = torch.stack([item[key] for item in batch])
    # 保留原始 caption 文本供 BERT 编码
    if "caption_text" in batch[0]:
        tensors["caption_text"] = [item["caption_text"] for item in batch]
    return tensors


def build_loader(dataset: GCGDataset, cfg) -> DataLoader:
    """根据配置构建 DataLoader，必要时丢弃最后不完整 batch。"""
    drop_last = len(dataset) >= cfg.batch_size
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_batch,
    )


def endless_iterator(loader: DataLoader) -> Iterator[Dict[str, torch.Tensor]]:
    """把 DataLoader 包装成无限迭代器，方便在 while 循环中交替取样。"""
    while True:
        for batch in loader:
            yield batch


def _clean_str(val) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    text = str(val).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text.lower()


def _parse_poi_tag_whitelist(raw: Optional[str]) -> Optional[set]:
    if raw is None:
        return set(DEFAULT_POI_FUNCTIONAL_TAGS)
    text = raw.strip().lower()
    if text in {"", "default"}:
        return set(DEFAULT_POI_FUNCTIONAL_TAGS)
    if text in {"none", "all", "*"}:
        return None
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def load_poi_feature_bank(csv_path: str, min_freq: int = 5, tag_whitelist: Optional[set] = None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"POI CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    all_tag_cols = [c for c in df.columns if c not in POI_BASE_COLUMNS]
    if tag_whitelist is None:
        tag_cols = all_tag_cols
    else:
        tag_cols = [c for c in all_tag_cols if c.lower() in tag_whitelist]
    print(f"[poi] using {len(tag_cols)} columns (total={len(all_tag_cols)})")
    if not tag_cols:
        print("[poi] warning: no POI columns selected by whitelist.")
    token_counter = Counter()
    grid_tokens = defaultdict(Counter)
    for row in df.itertuples(index=False, name=None):
        row_dict = {col: row[idx] for idx, col in enumerate(df.columns)}
        gid = int(row_dict["grid_id"])
        for col in tag_cols:
            val = row_dict.get(col)
            cleaned = _clean_str(val)
            if not cleaned:
                continue
            token = f"{col}:{cleaned}"
            token_counter[token] += 1
            grid_tokens[gid][token] += 1
    vocab = sorted([tok for tok, cnt in token_counter.items() if cnt >= min_freq])
    idx_map = {tok: idx for idx, tok in enumerate(vocab)}
    dim = len(vocab)
    default_vec = torch.zeros(dim, dtype=torch.float32)
    feature_bank = {}
    if dim == 0:
        return feature_bank, default_vec
    for gid, counter in grid_tokens.items():
        vec = torch.zeros(dim, dtype=torch.float32)
        for tok, cnt in counter.items():
            idx = idx_map.get(tok)
            if idx is not None:
                vec[idx] = float(cnt)
        if vec.sum() > 0:
            vec = F.normalize(vec.unsqueeze(0), dim=-1).squeeze(0)
        feature_bank[gid] = vec
    return feature_bank, default_vec


def load_osm_feature_bank(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"OSM features CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    numeric_cols = [c for c in df.columns if c != "grid_id" and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric OSM features found.")
    values = df[numeric_cols].fillna(0.0).to_numpy(dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True) + 1e-6
    normalized = (values - mean) / std
    feature_bank = {}
    for gid, vec in zip(df["grid_id"].astype(int).tolist(), normalized):
        feature_bank[gid] = torch.from_numpy(vec.copy())
    default_vec = torch.zeros(normalized.shape[1], dtype=torch.float32)
    return feature_bank, default_vec


def same_grid_intra_contrastive(
    embeddings: torch.Tensor,
    grid_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Supervised contrastive loss: positives are samples from the same grid."""
    if embeddings.size(0) <= 1:
        return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)

    grid_ids = grid_ids.view(-1)
    positive_mask = (grid_ids.unsqueeze(0) == grid_ids.unsqueeze(1)).float()
    positive_mask.fill_diagonal_(0.0)
    if positive_mask.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)

    temp = max(float(temperature), 1e-6)
    logits = (embeddings @ embeddings.t()) / temp
    logits = logits - logits.max(dim=1, keepdim=True)[0]

    eye = torch.eye(logits.size(0), device=logits.device)
    exp_logits = torch.exp(logits) * (1.0 - eye)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos_counts = positive_mask.sum(dim=1)
    valid = pos_counts > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)

    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / pos_counts.clamp(min=1.0)
    return -(mean_log_prob_pos[valid]).mean()


def run_step(
    model: MultiModalSSLModel,
    batch: Dict[str, torch.Tensor],
    modality: str,
    device: torch.device,
    loss_cfg,
    poi_bank,
    poi_default,
    osm_bank,
    osm_default,
    text_encoder: Optional[AutoModel],
    text_proj: Optional[nn.Module],
    hf_tokenizer: Optional[AutoTokenizer],
    text_weight: float,
    same_grid_weight: float,
) -> LossOutput:
    # 将 batch 中的张量移动到目标设备
    images = batch["image"].to(device)
    grid_ids_tensor = batch["grid_id"].to(device)
    grid_ids = batch["grid_id"].cpu().tolist()

    # 前向计算图像与文本的特征
    img_out = model.encode_image(images, modality)

    components: Dict[str, torch.Tensor] = {}

    intra_loss = torch.zeros(1, device=device)
    if loss_cfg.intra_modal_weight > 0 and images.size(0) > 1 and poi_default.numel() > 0:
        poi_vecs = []
        for gid in grid_ids:
            vec = poi_bank.get(int(gid))
            if vec is None:
                vec = poi_default
            poi_vecs.append(vec)
        poi_tensor = torch.stack(poi_vecs).to(device)
        intra_loss = gcg_intra_contrastive(
            img_out["embeddings"],
            poi_tensor,
            threshold=loss_cfg.intra_modal_threshold,
            temperature=loss_cfg.intra_modal_temperature,
        )
        components[f"{modality}_poi_contrast"] = intra_loss.detach()

    same_grid_loss = torch.zeros(1, device=device)
    if same_grid_weight > 0 and images.size(0) > 1:
        same_grid_loss = same_grid_intra_contrastive(
            img_out["embeddings"],
            grid_ids_tensor,
            temperature=loss_cfg.intra_modal_temperature,
        )
        components[f"{modality}_same_grid_contrast"] = same_grid_loss.detach()

    # 图像-OSM嵌入做 InfoNCE 对比损失
    osm_vecs = []
    for gid in grid_ids:
        vec = osm_bank.get(int(gid))
        if vec is None:
            vec = osm_default
        osm_vecs.append(vec)
    osm_tensor = torch.stack(osm_vecs).to(device)
    metadata_out = model.encode_metadata(osm_tensor)
    cross = contrastive_loss(img_out["embeddings"], metadata_out["embeddings"], model.temperature)
    components[f"{modality}_osm_cross"] = cross.detach()

    # 图-文对比：使用冻结 BERT 编码 caption CLS，MLP 映射到图像空间
    text_loss = torch.zeros(1, device=device)
    if text_weight > 0 and text_encoder is not None and hf_tokenizer is not None and text_proj is not None:
        captions: List[str] = batch.get("caption_text") or []
        if captions:
            enc = hf_tokenizer(
                captions,
                padding=True,
                truncation=True,
                max_length=hf_tokenizer.model_max_length,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                txt_out = text_encoder(**enc).last_hidden_state[:, 0, :]  # CLS
            txt_feat = text_proj(txt_out)
            text_loss = contrastive_loss(img_out["embeddings"], txt_feat, model.temperature)
            components[f"{modality}_txt_cross"] = text_loss.detach()

    # 加权组合 GCG 对比和跨模态，得到总损失
    total = loss_cfg.cross_modal_weight * cross + loss_cfg.intra_modal_weight * intra_loss
    if same_grid_weight > 0:
        total = total + same_grid_weight * same_grid_loss
    # total = loss_cfg.intra_modal_weight * intra_loss
    if text_weight > 0:
        total = total + text_weight * text_loss
    components[f"{modality}_total"] = total.detach()
    return LossOutput(total=total, components=components)


def save_checkpoint(model, optimizer, scheduler, step: int, cfg: SelfSupervisedConfig) -> None:
    """保存模型、优化器等状态，文件名包含训练步数。"""
    os.makedirs(cfg.output_dir, exist_ok=True)
    path = os.path.join(cfg.output_dir, f"ssl_step_{step:06d}.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "step": step,
            "config": asdict(cfg),
        },
        path,
    )


def build_scheduler(optimizer, cfg):
    """构建线性 warmup + 余下线性衰减的 LambdaLR。"""
    warmup = cfg.optim.warmup_steps
    total = max(cfg.optim.max_steps, warmup + 1)

    def lr_lambda(step):
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        return max(0.0, float(total - step) / float(max(1, total - warmup)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    """训练入口：初始化配置、数据、模型并循环优化。"""
    parser = argparse.ArgumentParser(description="Train Zhengzhou multimodal SSL stage.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config overriding defaults.")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda).")
    parser.add_argument("--log_csv_prefix", type=str, default="stage1_loss_history", help="Prefix for loss CSV per modality.")
    parser.add_argument("--image_backbone", type=str, default=None, help="Override image backbone (e.g., vit_l_16, vit_h_14, clip_vit_l14).")
    parser.add_argument("--text_backbone", type=str, default=None, help="Override text backbone (gru or bert).")
    parser.add_argument("--image_embed_dim", type=int, default=None, help="Target feature dim after image MLP.")
    parser.add_argument("--text_embed_dim", type=int, default=None, help="Target feature dim after text MLP.")
    parser.add_argument("--image_mlp_layers", type=int, default=None, help="Number of layers in satellite/street feature MLPs.")
    parser.add_argument("--image_mlp_hidden", type=int, default=None, help="Hidden dim for image feature MLPs.")
    parser.add_argument("--text_mlp_layers", type=int, default=None, help="Number of layers in text feature MLP.")
    parser.add_argument("--text_mlp_hidden", type=int, default=None, help="Hidden dim for text feature MLP.")
    parser.add_argument("--poi_csv", type=str, default=None, help="Path to grid-level POI records.")
    parser.add_argument("--poi_min_freq", type=int, default=5, help="Minimum token frequency to keep in POI vocabulary.")
    parser.add_argument(
        "--poi_tag_whitelist",
        type=str,
        default=None,
        help='POI 功能类标签白名单，逗号分隔；默认(空值/default)使用内置功能类。设为 "all"/"none" 可不过滤。',
    )
    parser.add_argument("--osm_csv", type=str, default=None, help="Path to grid-level OSM features.")
    parser.add_argument("--text_encoder_path", type=str, default=os.environ.get("TEXT_ENCODER_PATH", "bert-base-uncased"), help="预训练 BERT 路径，用于图文对比。")
    parser.add_argument("--text_contrast_weight", type=float, default=0.2, help="图文对比损失权重，0 则关闭。")
    parser.add_argument("--same_grid_contrast_weight", type=float, default=0.5, help="同网格图像模态内对比损失权重，0 表示关闭。")
    parser.add_argument("--text_max_len", type=int, default=512, help="BERT 编码的最大长度。")
    parser.add_argument("--city", type=str, choices=["zhengzhou", "changsha", "beijing", "shanghai", "wuhan", "xian"], default="zhengzhou", help="数据城市切换")
    parser.add_argument("--merged_json", type=str, default=None, help="覆盖 merged_by_grid 路径")
    parser.add_argument("--sat_category_path", type=str, default=None, help="卫星概念类别映射 JSON")
    parser.add_argument("--street_category_path", type=str, default=None, help="街景概念类别映射 JSON")
    parser.add_argument("--output_dir", type=str, default=None, help="覆盖输出目录")
    args = parser.parse_args()

    # 城市路径选择与配置覆盖
    merged_json = args.merged_json
    poi_csv = args.poi_csv
    osm_csv = args.osm_csv
    output_dir = args.output_dir
    missing = [name for name, value in {
        "merged_json": merged_json,
        "poi_csv": poi_csv,
        "osm_csv": osm_csv,
        "output_dir": output_dir,
    }.items() if not value]
    if missing:
        raise ValueError("Pass explicit paths for: " + ", ".join(missing))
    print("[city]", args.city)
    print("[paths] merged_json:", merged_json)
    print("[paths] poi_csv:", poi_csv)
    print("[paths] osm_csv:", osm_csv)
    print("[paths] output_dir:", output_dir)

    # 加载配置 & 自动选择设备
    cfg = load_config(args.config)
    cfg.data.merged_json_path = merged_json
    if args.sat_category_path:
        cfg.data.sat_category_path = args.sat_category_path
    if args.street_category_path:
        cfg.data.street_category_path = args.street_category_path
    cfg.output_dir = output_dir
    if args.image_backbone:
        cfg.model.image_backbone = args.image_backbone
    if args.text_backbone:
        cfg.model.text_backbone = args.text_backbone
    if args.image_embed_dim is not None:
        cfg.model.image_embed_dim = args.image_embed_dim
    if args.text_embed_dim is not None:
        cfg.model.text_embed_dim = args.text_embed_dim
    if args.image_mlp_layers is not None:
        cfg.model.image_mlp_layers = args.image_mlp_layers
    if args.image_mlp_hidden is not None:
        cfg.model.image_mlp_hidden_dim = args.image_mlp_hidden
    if args.text_mlp_layers is not None:
        cfg.model.text_mlp_layers = args.text_mlp_layers
    if args.text_mlp_hidden is not None:
        cfg.model.text_mlp_hidden_dim = args.text_mlp_hidden
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # tokenizer 先置空，构建首个数据集时创建词表
    tokenizer: Optional[TextTokenizer] = None
    datasets: Dict[str, GCGDataset] = {}
    # 若启用遥感模态，则加载对应数据集，并共享 tokenizer
    if cfg.data.include_satellite:
        sat_dataset = GCGDataset(cfg.data, modality="sat", tokenizer=tokenizer, build_vocab=True)
        tokenizer = sat_dataset.tokenizer
        datasets["sat"] = sat_dataset
    # 若启用街景模态，同理
    if cfg.data.include_street:
        street_dataset = GCGDataset(cfg.data, modality="street", tokenizer=tokenizer, build_vocab=True)
        tokenizer = street_dataset.tokenizer
        datasets["street"] = street_dataset

    if not datasets:
        raise RuntimeError("No datasets initialized. Please enable at least one modality.")

    poi_whitelist = _parse_poi_tag_whitelist(args.poi_tag_whitelist)
    poi_bank, poi_default = load_poi_feature_bank(
        poi_csv,
        min_freq=args.poi_min_freq,
        tag_whitelist=poi_whitelist,
    )
    osm_bank, osm_default = load_osm_feature_bank(osm_csv)
    metadata_dim = osm_default.numel()
    if metadata_dim == 0:
        raise ValueError("OSM features have zero dimension; please check the CSV.")

    # 为每个模态建立 DataLoader 与无限迭代器
    loaders = {k: build_loader(ds, cfg.data) for k, ds in datasets.items()}
    iters = {k: endless_iterator(ld) for k, ld in loaders.items()}

    # tokenizer 若存在则用其词表大小，否则使用配置默认值
    vocab_size = len(tokenizer.vocab) if tokenizer else cfg.model.vocab_size
    use_sat = "sat" in datasets
    use_street = "street" in datasets
    model = MultiModalSSLModel(cfg.model, use_sat, use_street, vocab_size, metadata_dim=metadata_dim).to(device)

    # 预训练 BERT 文本编码 + MLP 映射到图像空间
    text_encoder = None
    text_proj = None
    hf_tokenizer = None
    if args.text_contrast_weight > 0:
        hf_tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_path, model_max_length=args.text_max_len)
        text_encoder = AutoModel.from_pretrained(args.text_encoder_path).to(device)
        text_encoder.eval()
        for p in text_encoder.parameters():
            p.requires_grad_(False)
        target_dim = cfg.model.projection_dim if hasattr(cfg.model, "projection_dim") else cfg.model.image_embed_dim
        text_proj = nn.Linear(text_encoder.config.hidden_size, target_dim).to(device)

    # 优化器、学习率调度器与混合精度缩放器
    params = list(model.parameters())
    if text_proj is not None:
        params += list(text_proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    model.train()
    step = 0
    progress = tqdm(total=cfg.optim.max_steps, desc="Training", dynamic_ncols=True)
    log_counters = {"sat": 0, "street": 0}
    history: Dict[str, List[dict]] = {"sat": [], "street": []}
    while step < cfg.optim.max_steps:
        for modality, iterator in iters.items():
            batch = next(iterator)
            # GPU 时启用 autocast，CPU 时用空上下文占位
            autocast_context = (
                torch.cuda.amp.autocast(enabled=scaler.is_enabled())
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                loss_out = run_step(
                    model,
                    batch,
                    modality,
                    device,
                    cfg.loss,
                    poi_bank,
                    poi_default,
                    osm_bank,
                    osm_default,
                    text_encoder,
                    text_proj,
                    hf_tokenizer,
                    args.text_contrast_weight,
                    args.same_grid_contrast_weight,
                )
            optimizer.zero_grad()
            scaler.scale(loss_out.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            progress.update(1)

            # 定期打印学习率与各项损失（按模态独立计数）
            log_counters[modality] += 1
            if log_counters[modality] % cfg.log_interval == 0 or step == cfg.optim.max_steps:
                lr = scheduler.get_last_lr()[0]
                comp = {k: float(v.item()) for k, v in loss_out.components.items()}
                print(f"Step {step:06d} | Mod {modality} | LR {lr:.2e} | Loss {float(loss_out.total.item()):.4f} | {comp}")
                record = {"step": step, "lr": lr, "total_loss": float(loss_out.total.item())}
                for key, value in comp.items():
                    record[key] = float(value)
                history[modality].append(record)
              
            # =========

            # # 每个step都保存loss
            # comp1 = {k: float(v.item()) for k, v in loss_out.components.items()}
            # lr = scheduler.get_last_lr()[0]
            # record1 = {"step": step, "lr": lr,}
            # for key, value in comp1.items():
            #     record1[key] = float(value)
            # history[modality].append(record1)
            # =========

            # 按设定间隔保存 checkpoint
            if step > 0 and step % cfg.ckpt_interval == 0:
                save_checkpoint(model, optimizer, scheduler, step, cfg)

            if step >= cfg.optim.max_steps:
                break

    progress.close()
    # 训练完成后保存最终模型
    save_checkpoint(model, optimizer, scheduler, step, cfg)
    import pandas as pd
    for modality, records in history.items():
        if not records:
            continue
        log_path = os.path.join(cfg.output_dir, f"{args.log_csv_prefix}_{modality}.csv")
        pd.DataFrame(records).to_csv(log_path, index=False)
    print("Training completed.")


if __name__ == "__main__":
    main()
