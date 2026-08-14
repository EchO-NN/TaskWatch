from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vljepa_droid.data.collate import collate_droid_samples
from vljepa_droid.data.dataset import PreparedDroidDataset
from vljepa_droid.evaluation.retrieval import compute_retrieval_metrics
from vljepa_droid.models.vljepa import VLJEPAModel


@torch.no_grad()
def evaluate_model(
    model: VLJEPAModel,
    dataset: PreparedDroidDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, float]:
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
    if was_training:
        model.train()
    return compute_retrieval_metrics(
        torch.cat(predictions),
        torch.cat(targets),
        target_texts=target_texts,
    )


def write_retrieval_metrics(path: str | Path, metrics: dict[str, float]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
