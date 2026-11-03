from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn.functional as F


def _normalize_distribution(target: torch.Tensor) -> torch.Tensor:
    norm = target.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return target / norm


def distribution_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    targets = _normalize_distribution(targets)
    log_probs = F.log_softmax(logits, dim=-1)
    return F.kl_div(log_probs, targets, reduction="batchmean")


def contrastive_loss(image_embeds: torch.Tensor, text_embeds: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    logits = temperature * image_embeds @ text_embeds.t()
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i + loss_t)


def gcg_intra_contrastive(
    embeddings: torch.Tensor,
    gcg_vectors: torch.Tensor,
    threshold: float,
    temperature: float,
) -> torch.Tensor:
    """Supervised contrastive loss guided by GCG similarity mask."""
    if gcg_vectors is None or gcg_vectors.numel() == 0:
        raise ValueError("gcg_vectors must be provided for intra-modal contrast.")

    gcg_norm = F.normalize(gcg_vectors, dim=-1)
    sim_targets = gcg_norm @ gcg_norm.t()
    positive_mask = (sim_targets >= threshold).float()
    positive_mask.fill_diagonal_(0.0)
    if positive_mask.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)

    logits = (embeddings @ embeddings.t()) / temperature
    logits = logits - logits.max(dim=1, keepdim=True)[0]

    exp_logits = torch.exp(logits) * (1.0 - torch.eye(logits.size(0), device=logits.device))
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos_counts = positive_mask.sum(dim=1)
    valid = pos_counts > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)

    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / pos_counts.clamp(min=1.0)
    loss = -(mean_log_prob_pos[valid]).mean()
    return loss


@dataclass
class LossOutput:
    total: torch.Tensor
    components: Dict[str, torch.Tensor] = field(default_factory=dict)
