from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vljepa_droid.data.collate import collate_droid_samples
from vljepa_droid.data.dataset import PreparedDroidDataset
from vljepa_droid.evaluation.retrieval import compute_retrieval_metrics, ranked_text_examples
from vljepa_droid.models.vljepa import VLJEPAModel


@torch.no_grad()
def evaluate_model(
    model: VLJEPAModel,
    dataset: PreparedDroidDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    ranked_example_count: int = 0,
) -> dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_droid_samples,
    )
    was_training = model.training
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    target_texts: list[str] = []
    episode_ids: list[str] = []
    for batch in loader:
        inputs = {}
        if "visual_tokens" in batch:
            inputs["visual_tokens"] = batch["visual_tokens"].to(device, non_blocking=True)
        else:
            inputs["video"] = batch["video"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(target_texts=batch["target_text"], **inputs)
        predictions.append(output.z_pred.cpu())
        targets.append(output.z_target.cpu())
        target_texts.extend(batch["target_text"])
        episode_ids.extend(batch["episode_id"])
    if was_training:
        model.train()
    prediction_tensor = torch.cat(predictions)
    target_tensor = torch.cat(targets)
    metrics: dict[str, object] = compute_retrieval_metrics(
        prediction_tensor,
        target_tensor,
        target_texts=target_texts,
    )
    if ranked_example_count:
        metrics["video_to_text_examples"] = ranked_text_examples(
            prediction_tensor,
            target_tensor,
            target_texts=target_texts,
            episode_ids=episode_ids,
            count=ranked_example_count,
        )
    return metrics


def write_retrieval_metrics(path: str | Path, metrics: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
