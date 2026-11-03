from __future__ import annotations

from typing import Dict, Optional
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torchvision.models import (
    ViT_B_16_Weights,
    ViT_L_16_Weights,
    ViT_L_32_Weights,
    ViT_H_14_Weights,
    vit_b_16,
    vit_l_16,
    vit_l_32,
    vit_h_14,
)

try:
    from transformers import BertModel, CLIPVisionModel
except ImportError:  # pragma: no cover - optional dependency
    BertModel = None
    CLIPVisionModel = None

from ..config import ModelConfig


class ResNetEncoder(nn.Module):
    """ResNet backbone returning global pooled features."""

    def __init__(self, name: str = "resnet50", pretrained: bool = False):
        super().__init__()
        # 根据名称加载 torchvision 的 ResNet，并可选择是否加载 ImageNet 预训练权重
        try:
            backbone = getattr(tvm, name)(weights="IMAGENET1K_V2" if pretrained else None)
        except TypeError:
            backbone = getattr(tvm, name)(pretrained=pretrained)
        # 记录输出维度（原始 fc 输入维度）
        self.out_dim = backbone.fc.in_features
        # 去掉最后的全连接层，只保留卷积特征提取部分
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向时得到 (B, C, 1, 1)，再展平成 (B, C)
        feats = self.encoder(x)
        return feats.flatten(1)


class ViTEncoder(nn.Module):
    """Vision Transformer encoder returning CLS token features."""

    def __init__(self, name: str = "vit_b_16", pretrained: bool = True):
        super().__init__()
        name = name.lower()
        if name == "vit_b_16":
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            model = vit_b_16(weights=weights)
        elif name == "vit_l_16":
            weights = ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
            model = vit_l_16(weights=weights)
        elif name == "vit_l_32":
            weights = ViT_L_32_Weights.IMAGENET1K_V1 if pretrained else None
            model = vit_l_32(weights=weights)
        elif name == "vit_h_14":
            weights = ViT_H_14_Weights.IMAGENET1K_V1 if pretrained else None
            model = vit_h_14(weights=weights)
        else:
            raise ValueError(f"Unsupported ViT backbone: {name}")
        self.out_dim = model.heads.head.in_features
        model.heads = nn.Identity()
        self.encoder = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CLIPVisionEncoder(nn.Module):
    """CLIP ViT encoder（L/14）用于产生全局视觉特征。"""

    def __init__(self, name: str = "clip_vit_l14", pretrained_path: Optional[str] = None):
        super().__init__()
        if CLIPVisionModel is None:
            raise ImportError("`transformers` 未安装，无法构建 CLIP 编码器。请先 `pip install transformers`。")
        name = name.lower()
        if name not in {"clip_vit_l14", "clip-vit-large-patch14"}:
            raise ValueError(f"Unsupported CLIP backbone: {name}")
        model_name = pretrained_path or os.environ.get("CLIP_MODEL_PATH", "openai/clip-vit-large-patch14")
        self.encoder = CLIPVisionModel.from_pretrained(model_name)
        self.out_dim = self.encoder.config.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(pixel_values=x)
        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return pooled


class ProjectionHead(nn.Module):
    """Small projection MLP used before contrastive losses."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        # 若未指定隐藏维度，则与输入维度一致
        hidden_dim = hidden_dim or in_dim
        layers = []
        dim = in_dim
        # 按层数堆叠线性层 + GELU + Dropout
        for _ in range(max(1, num_layers - 1)):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输出向量做 L2 归一化，方便 InfoNCE 对比学习
        return F.normalize(self.net(x), dim=-1)


def build_modal_mlp(in_dim: int, out_dim: int, num_layers: Optional[int], hidden_dim: Optional[int], dropout: float) -> nn.Module:
    """构建模态特定的 feature MLP，若层数为 0 则视情况简化为 Identity/Linear。"""
    if num_layers is None:
        num_layers = 0
    if num_layers <= 0:
        if in_dim == out_dim:
            return nn.Identity()
        return nn.Linear(in_dim, out_dim)
    hidden_dim = hidden_dim or out_dim
    layers = []
    dim = in_dim
    for idx in range(num_layers):
        next_dim = out_dim if idx == num_layers - 1 else hidden_dim
        layers.append(nn.Linear(dim, next_dim))
        if idx != num_layers - 1:
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        dim = next_dim
    return nn.Sequential(*layers)


def build_image_encoder(cfg: ModelConfig):
    backbone_name = cfg.image_backbone.lower()
    if backbone_name.startswith("clip"):
        return CLIPVisionEncoder(backbone_name, pretrained_path=cfg.image_pretrained_path)
    if backbone_name.startswith("vit"):
        return ViTEncoder(backbone_name, pretrained=cfg.pretrained_backbone)
    return ResNetEncoder(backbone_name, pretrained=cfg.pretrained_backbone)


class ImageBranch(nn.Module):
    """图像编码分支，仅输出特征和对比嵌入。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # 针对特定模态初始化 backbone
        self.encoder = build_image_encoder(cfg)
        if cfg.freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.feature_dim = self.encoder.out_dim
        # 图像 → 对比空间的投影
        self.projector = ProjectionHead(self.feature_dim, cfg.projection_dim, hidden_dim=self.feature_dim, num_layers=2, dropout=cfg.dropout)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.encoder(images)
        return {
            "features": feats,
            "embeddings": self.projector(feats),
        }


class GRUTextEncoder(nn.Module):
    """Bi-GRU encoder that pools caption tokens."""

    def __init__(self, vocab_size: int, cfg: ModelConfig):
        super().__init__()
        # 字符级别的嵌入层
        self.embedding = nn.Embedding(vocab_size, cfg.text_hidden_dim, padding_idx=0)
        self.gru = nn.GRU(
            cfg.text_hidden_dim,
            cfg.text_hidden_dim,
            num_layers=cfg.text_num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.text_num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.feature_dim = cfg.text_hidden_dim * 2
        # 将文本特征映射到对比空间
        self.proj = ProjectionHead(self.feature_dim, cfg.projection_dim, hidden_dim=self.feature_dim, num_layers=2, dropout=cfg.dropout)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 通过 attention_mask 统计每个样本的有效长度
        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
        # 嵌入并 pack 成变长序列，避免 PAD 对 GRU 的影响
        embedded = self.embedding(token_ids)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.gru(packed)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        if self.gru.bidirectional:
            forward = hidden[-2]
            backward = hidden[-1]
            pooled = torch.cat([forward, backward], dim=-1)
        else:
            pooled = hidden[-1]
        # 经过 projection 得到最终文本嵌入
        proj = self.proj(pooled)
        return {"features": pooled, "embeddings": proj}


class BERTTextEncoder(nn.Module):
    """BERT encoder，用 transformer CLS 向量作为文本表示。"""

    def __init__(self, cfg: ModelConfig, pretrained_path: Optional[str] = None):
        super().__init__()
        if BertModel is None:
            raise ImportError("`transformers` 未安装，无法构建 BERT 文本编码器。")
        model_name = pretrained_path or os.environ.get("TEXT_ENCODER_PATH", "bert-base-uncased")
        self.bert = BertModel.from_pretrained(model_name)
        self.feature_dim = self.bert.config.hidden_size
        self.proj = ProjectionHead(self.feature_dim, cfg.projection_dim, hidden_dim=self.feature_dim, num_layers=2, dropout=cfg.dropout)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.bert(input_ids=token_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
        proj = self.proj(pooled)
        return {"features": pooled, "embeddings": proj}


class MetadataEncoder(nn.Module):
    """Encoder for structured numeric features such as OSM attributes."""

    def __init__(self, input_dim: int, cfg: ModelConfig):
        super().__init__()
        self.feature_dim = input_dim
        self.proj = ProjectionHead(self.feature_dim, cfg.projection_dim, hidden_dim=self.feature_dim, num_layers=2, dropout=cfg.dropout)

    def forward(self, feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        proj = self.proj(feats)
        return {"features": feats, "embeddings": proj}


def build_text_encoder(cfg: ModelConfig, vocab_size: int):
    backbone = cfg.text_backbone.lower()
    if backbone == "bert":
        return BERTTextEncoder(cfg, pretrained_path=cfg.text_pretrained_path)
    if backbone != "gru":
        raise ValueError(f"Unsupported text backbone: {cfg.text_backbone}")
    return GRUTextEncoder(vocab_size, cfg)


class MultiModalSSLModel(nn.Module):
    """End-to-end module delivering image & text representations."""

    def __init__(self, cfg: ModelConfig, use_sat: bool, use_street: bool, vocab_size: int, metadata_dim: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        # 遥感与街景分别拥有各自的图像分支
        self.sat_branch = ImageBranch(cfg) if use_sat else None
        self.street_branch = ImageBranch(cfg) if use_street else None
        self.metadata_encoder = None
        if metadata_dim is not None:
            self.metadata_encoder = MetadataEncoder(metadata_dim, cfg)
            self.text_encoder = None
        else:
            self.text_encoder = build_text_encoder(cfg, vocab_size)
        # 对比学习中的温度参数，使用 log 参数化确保为正
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / max(1e-3, 0.07))))

    def encode_image(self, images: torch.Tensor, modality: str) -> Dict[str, torch.Tensor]:
        # 根据模态路由到对应分支
        if modality == "sat":
            if self.sat_branch is None:
                raise RuntimeError("Satellite branch not initialized.")
            return self.sat_branch(images)
        if modality == "street":
            if self.street_branch is None:
                raise RuntimeError("Street branch not initialized.")
            return self.street_branch(images)
        raise ValueError(f"Unsupported modality: {modality}")

    def encode_text(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.text_encoder is None:
            raise RuntimeError("Text encoder is disabled when metadata_dim is provided.")
        return self.text_encoder(token_ids, attention_mask)

    def encode_metadata(self, metadata_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.metadata_encoder is None:
            raise RuntimeError("Metadata encoder not initialized.")
        return self.metadata_encoder(metadata_feats)

    @property
    def temperature(self) -> torch.Tensor:
        # 限制温度范围，避免数值不稳定
        return torch.clamp(self.logit_scale.exp(), 1e-3, 100.0)
