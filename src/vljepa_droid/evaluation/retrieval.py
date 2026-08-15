from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


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


def _topk_cosine_metrics(
    similarity: torch.Tensor,
    prefix: str,
    *,
    k: int = 10,
) -> dict[str, float]:
    effective_k = min(k, similarity.shape[1])
    values = similarity.topk(effective_k, dim=1, largest=True, sorted=True).values
    metrics = {
        f"{prefix}_top{effective_k}_cosine_mean": float(values.mean()),
        f"{prefix}_rank{effective_k}_cosine_mean": float(values[:, -1].mean()),
        f"{prefix}_rank{effective_k}_cosine_median": float(values[:, -1].median()),
        f"{prefix}_rank{effective_k}_cosine_min": float(values[:, -1].min()),
        f"{prefix}_rank{effective_k}_cosine_max": float(values[:, -1].max()),
    }
    for rank in range(effective_k):
        metrics[f"{prefix}_rank{rank + 1}_cosine_mean"] = float(values[:, rank].mean())
    return metrics


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
    metrics.update(_topk_cosine_metrics(similarity, "video_to_text"))
    metrics["positive_cosine_mean"] = float(similarity.diagonal().mean())
    metrics["random_r1"] = 1.0 / similarity.shape[0]
    metrics["num_samples"] = int(similarity.shape[0])
    if target_texts is not None:
        if len(target_texts) != similarity.shape[0]:
            raise ValueError("target_texts must match the embedding batch length")
        normalized_texts = [_normalize_text(text) for text in target_texts]
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


def ranked_text_examples(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    *,
    target_texts: list[str],
    episode_ids: list[str],
    count: int = 1,
    top_k: int = 10,
) -> list[dict[str, object]]:
    if z_pred.ndim != 2 or z_pred.shape != z_target.shape or z_pred.shape[0] < 1:
        raise ValueError("retrieval embeddings must have equal non-empty [N,D] shape")
    if len(target_texts) != z_pred.shape[0] or len(episode_ids) != z_pred.shape[0]:
        raise ValueError("texts and episode IDs must match the embedding batch length")
    if count < 0 or top_k < 1:
        raise ValueError("count must be non-negative and top_k must be positive")

    similarity = F.normalize(z_pred.float(), dim=-1) @ F.normalize(
        z_target.float(), dim=-1
    ).T
    effective_k = min(top_k, similarity.shape[1])
    normalized_texts = [_normalize_text(text) for text in target_texts]
    examples: list[dict[str, object]] = []
    for query_index in range(min(count, similarity.shape[0])):
        values, indices = similarity[query_index].topk(
            effective_k,
            largest=True,
            sorted=True,
        )
        query_text = target_texts[query_index]
        candidates = []
        for rank, (candidate_index, cosine) in enumerate(
            zip(indices.tolist(), values.tolist(), strict=True),
            start=1,
        ):
            candidates.append(
                {
                    "rank": rank,
                    "candidate_index": candidate_index,
                    "candidate_episode_id": episode_ids[candidate_index],
                    "text": target_texts[candidate_index],
                    "cosine": cosine,
                    "is_identical_target_text": (
                        normalized_texts[candidate_index] == normalized_texts[query_index]
                    ),
                }
            )
        examples.append(
            {
                "query_index": query_index,
                "query_episode_id": episode_ids[query_index],
                "query_target_text": query_text,
                "candidates": candidates,
            }
        )
    return examples
