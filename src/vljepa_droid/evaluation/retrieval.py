from __future__ import annotations

import torch
import torch.nn.functional as F


def _direction_metrics(similarity: torch.Tensor, prefix: str) -> dict[str, float]:
    order = similarity.argsort(dim=1, descending=True)
    labels = torch.arange(similarity.shape[0], device=similarity.device).unsqueeze(1)
    ranks = (order == labels).nonzero(as_tuple=False)[:, 1] + 1
    return {
        f"{prefix}_r1": float((ranks <= 1).float().mean()),
        f"{prefix}_r5": float((ranks <= 5).float().mean()),
        f"{prefix}_r10": float((ranks <= 10).float().mean()),
        f"{prefix}_median_rank": float(ranks.float().median()),
    }


def _relevance_metrics(
    similarity: torch.Tensor,
    relevance: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    if relevance.shape != similarity.shape or relevance.dtype != torch.bool:
        raise ValueError("relevance must be a boolean matrix matching similarity")
    if not bool(relevance.any(dim=1).all()):
        raise ValueError("every query needs at least one relevant target")
    order = similarity.argsort(dim=1, descending=True)
    ordered_relevance = relevance.gather(1, order)
    ranks = ordered_relevance.float().argmax(dim=1) + 1
    return {
        f"{prefix}_r1": float((ranks <= 1).float().mean()),
        f"{prefix}_r5": float((ranks <= 5).float().mean()),
        f"{prefix}_r10": float((ranks <= 10).float().mean()),
        f"{prefix}_median_rank": float(ranks.float().median()),
    }


def compute_retrieval_metrics(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    *,
    target_texts: list[str] | None = None,
) -> dict[str, float]:
    if z_pred.ndim != 2 or z_pred.shape != z_target.shape or z_pred.shape[0] < 1:
        raise ValueError("retrieval embeddings must have equal non-empty [N,D] shape")
    z_pred = F.normalize(z_pred.float(), dim=-1)
    z_target = F.normalize(z_target.float(), dim=-1)
    similarity = z_pred @ z_target.T
    metrics = _direction_metrics(similarity, "video_to_text")
    metrics.update(_direction_metrics(similarity.T, "text_to_video"))
    metrics["positive_cosine_mean"] = float(similarity.diagonal().mean())
    metrics["random_r1"] = 1.0 / similarity.shape[0]
    metrics["num_samples"] = int(similarity.shape[0])
    if target_texts is not None:
        if len(target_texts) != similarity.shape[0]:
            raise ValueError("target_texts must match the embedding batch length")
        normalized_texts = [" ".join(text.casefold().split()) for text in target_texts]
        relevance = torch.tensor(
            [[candidate == query for candidate in normalized_texts] for query in normalized_texts],
            dtype=torch.bool,
            device=similarity.device,
        )
        metrics.update(
            _relevance_metrics(
                similarity,
                relevance,
                "video_to_text_any_matching_caption",
            )
        )
        relevant_counts = relevance.sum(dim=1).float()
        metrics["video_to_text_any_matching_caption_random_r1"] = float(
            (relevant_counts / similarity.shape[0]).mean()
        )
        metrics["unique_target_texts"] = len(set(normalized_texts))
        metrics["duplicate_target_text_fraction"] = 1.0 - (
            len(set(normalized_texts)) / len(normalized_texts)
        )
    return metrics
