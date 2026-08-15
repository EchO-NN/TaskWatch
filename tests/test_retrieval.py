import torch
import pytest

from vljepa_droid.evaluation.retrieval import compute_retrieval_metrics, ranked_text_examples


def test_perfect_retrieval() -> None:
    embeddings = torch.eye(12)
    metrics = compute_retrieval_metrics(embeddings, embeddings)
    assert metrics["video_to_text_r1"] == 1.0
    assert metrics["text_to_video_r1"] == 1.0
    assert metrics["video_to_text_median_rank"] == 1.0
    assert metrics["random_r1"] == 1 / 12
    assert metrics["video_to_text_rank1_cosine_mean"] == 1.0
    assert metrics["video_to_text_rank10_cosine_mean"] == 0.0
    assert metrics["video_to_text_rank10_cosine_median"] == 0.0
    assert metrics["video_to_text_top10_cosine_mean"] == pytest.approx(0.1)


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


def test_ranked_text_examples_include_scores_and_semantic_match() -> None:
    predictions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    targets = predictions.clone()
    examples = ranked_text_examples(
        predictions,
        targets,
        target_texts=["pick cup", "open drawer", "move left"],
        episode_ids=["episode-a", "episode-b", "episode-c"],
        count=1,
        top_k=3,
    )

    assert examples[0]["query_episode_id"] == "episode-a"
    candidates = examples[0]["candidates"]
    assert [candidate["text"] for candidate in candidates] == [
        "pick cup",
        "open drawer",
        "move left",
    ]
    assert candidates[0]["cosine"] == 1.0
    assert candidates[0]["is_identical_target_text"] is True
    assert candidates[1]["is_identical_target_text"] is False
