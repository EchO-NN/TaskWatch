from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vljepa_droid.distributed.gather import differentiable_all_gather


@dataclass
class InfoNCEResult:
    loss: torch.Tensor
    loss_pred_to_text: torch.Tensor
    loss_text_to_pred: torch.Tensor
    positive_cosine_mean: torch.Tensor
    negative_cosine_mean: torch.Tensor
    embedding_std_pred: torch.Tensor
    embedding_std_target: torch.Tensor
    global_batch: int


def bidirectional_infonce(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    *,
    temperature: float = 0.07,
    distributed_gather: bool = True,
) -> InfoNCEResult:
    if z_pred.ndim != 2 or z_pred.shape != z_target.shape:
        raise ValueError(
            f"embeddings must have equal [B,D] shape, got "
            f"{tuple(z_pred.shape)} and {tuple(z_target.shape)}"
        )
    if z_pred.shape[0] < 1 or temperature <= 0:
        raise ValueError("batch must be non-empty and temperature must be positive")

    z_pred = F.normalize(z_pred.float(), dim=-1)
    z_target = F.normalize(z_target.float(), dim=-1)
    if distributed_gather:
        z_pred = differentiable_all_gather(z_pred)
        z_target = differentiable_all_gather(z_target)

    cosine = z_pred @ z_target.T
    logits = cosine / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_pred_to_text = F.cross_entropy(logits, labels)
    loss_text_to_pred = F.cross_entropy(logits.T, labels)
    loss = 0.5 * (loss_pred_to_text + loss_text_to_pred)

    diagonal = cosine.diagonal()
    if cosine.numel() == diagonal.numel():
        negative_mean = cosine.new_zeros(())
    else:
        negative_mean = (cosine.sum() - diagonal.sum()) / (cosine.numel() - diagonal.numel())
    return InfoNCEResult(
        loss=loss,
        loss_pred_to_text=loss_pred_to_text,
        loss_text_to_pred=loss_text_to_pred,
        positive_cosine_mean=diagonal.mean(),
        negative_cosine_mean=negative_mean,
        embedding_std_pred=z_pred.std(dim=0, unbiased=False).mean(),
        embedding_std_target=z_target.std(dim=0, unbiased=False).mean(),
        global_batch=logits.shape[0],
    )
