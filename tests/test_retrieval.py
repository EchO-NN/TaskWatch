import torch
import pytest

from vljepa_droid.evaluation.retrieval import compute_retrieval_metrics


def test_perfect_retrieval() -> None:
    embeddings = torch.eye(12)
    metrics = compute_retrieval_metrics(embeddings, embeddings)
    assert metrics["video_to_text_r1"] == 1.0
    assert metrics["text_to_video_r1"] == 1.0
    assert metrics["video_to_text_median_rank"] == 1.0
    assert metrics["random_r1"] == 1 / 12


def test_duplicate_aware_video_to_text_retrieval() -> None:
    predictions = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    metrics = compute_retrieval_metrics(
        predictions,
        targets,
        target_texts=["Pick cup", "  pick   CUP ", "open drawer"],
    )
    assert metrics["video_to_text_any_matching_caption_r1"] == 1.0
    assert metrics["video_to_text_any_matching_caption_median_rank"] == 1.0
    assert metrics["unique_target_texts"] == 2
    assert metrics["duplicate_target_text_fraction"] == pytest.approx(1 / 3)
