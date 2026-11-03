from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    merged_json_path: str = ""
    sat_category_path: str = ""
    street_category_path: str = ""
    image_size: int = 224
    num_workers: int = 8
    batch_size: int = 16
    image_read_retries: int = 3
    image_read_retry_delay: float = 0.2
    max_tokens: int = 350
    min_area_ratio: float = 1e-4
    augment_prob: float = 0.5
    include_street: bool = True
    include_satellite: bool = True
    cache_dir: str | None = None


@dataclass
class ModelConfig:
    image_embed_dim: int = 768 # mlp输出的维度
    text_embed_dim: int = 512
    projection_dim: int = 256
    num_image_layers: int = 0  # legacy extra MLP layers after backbone
    image_mlp_layers: int | None = None
    image_mlp_hidden_dim: int | None = None
    freeze_backbone: bool = False
    image_backbone: str = "vit_b_16"  # options: resnet50, vit_b_16, clip_vit_l14, vit_l_16, vit_h_14, clip_vit_l14
    pretrained_backbone: bool = True
    image_pretrained_path: str | None = None
    vocab_size: int = 60000
    text_hidden_dim: int = 512
    text_num_layers: int = 2
    text_mlp_layers: int = 1
    text_mlp_hidden_dim: int | None = None
    dropout: float = 0.1
    use_shared_backbone: bool = False
    text_backbone: str = "bert"  # options: gru, bert
    text_pretrained_path: str | None = None


@dataclass
class LossConfig:
    single_modal_weight: float = 1.0
    cross_modal_weight: float = 1.0
    phrase_weight: float = 0.5
    temp_init: float = 0.07
    intra_modal_weight: float = 0.5
    intra_modal_threshold: float = 0.5
    intra_modal_temperature: float = 0.1
    intra_modal_use_mask: bool = True
    intra_modal_use_phrase: bool = True


@dataclass
class OptimConfig:
    lr: float = 5e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    max_steps: int = 20000
    gradient_clip: float = 1.0


@dataclass
class SelfSupervisedConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    log_interval: int = 125
    ckpt_interval: int = 2500
    output_dir: str = "outputs/stage1"
