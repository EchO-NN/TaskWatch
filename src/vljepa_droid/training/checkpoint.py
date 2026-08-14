from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from vljepa_droid.models.vljepa import VLJEPAModel


CHECKPOINT_METADATA = {
    "x_encoder": "V-JEPA-2.1-ViT-G-2B",
    "input_resolution": 256,
    "num_frames": 32,
    "patch_size": 16,
    "tubelet_size": 2,
    "visual_tokens": 4096,
    "predictor_base": "Qwen/Qwen3-4B",
    "predictor_layers": list(range(28, 36)),
    "shared_embedding_dim": 1536,
    "y_encoder": "google/embeddinggemma-300m",
    "loss": "bidirectional_infonce",
}


def _trainable_state(model: VLJEPAModel) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "predictor": {
            key: value
            for key, value in model.predictor.state_dict().items()
            if not key.startswith("embed_tokens.")
        },
        "y_encoder": model.y_encoder.state_dict(),
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: VLJEPAModel,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    config: dict[str, Any],
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "global_step": int(global_step),
        "model": _trainable_state(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": {"name": "constant", "global_step": int(global_step)},
        "config": config,
        "metadata": CHECKPOINT_METADATA,
    }
    torch.save(payload, checkpoint_path)
    metadata_path = checkpoint_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {"global_step": global_step, **CHECKPOINT_METADATA},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    *,
    model: VLJEPAModel,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    incompatible = model.predictor.load_state_dict(payload["model"]["predictor"], strict=False)
    if set(incompatible.missing_keys) != {"embed_tokens.weight"}:
        raise RuntimeError(f"unexpected missing Predictor checkpoint keys: {incompatible}")
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected Predictor checkpoint keys: {incompatible}")
    model.y_encoder.load_state_dict(payload["model"]["y_encoder"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload["global_step"])
