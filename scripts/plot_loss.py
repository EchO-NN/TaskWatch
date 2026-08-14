#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    result = np.convolve(values, np.ones(window) / window, mode="valid")
    return np.concatenate([np.full(window - 1, np.nan), result])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--smooth", type=int, default=5)
    args = parser.parse_args()
    log_path = Path(args.log)
    rows = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty training log: {log_path}")
    steps = np.array([row["step"] for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for key, label, color in (
        ("loss_total", "bidirectional total", "#1f77b4"),
        ("loss_pred_to_text", "video to text", "#ff7f0e"),
        ("loss_text_to_pred", "text to video", "#2ca02c"),
    ):
        values = np.array([row[key] for row in rows], dtype=float)
        axes[0].plot(steps, values, color=color, alpha=0.22, linewidth=1)
        axes[0].plot(
            steps,
            moving_average(values, args.smooth),
            color=color,
            linewidth=2,
            label=label,
        )
    axes[0].set_ylabel("InfoNCE loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for key, label, color in (
        ("positive_cosine_mean", "positive cosine", "#9467bd"),
        ("negative_cosine_mean", "negative cosine", "#8c564b"),
    ):
        values = np.array([row[key] for row in rows], dtype=float)
        axes[1].plot(
            steps,
            moving_average(values, args.smooth),
            color=color,
            linewidth=2,
            label=label,
        )
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("cosine similarity")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("VL-JEPA DROID Stage 1 training")
    fig.tight_layout()
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(prefix.with_suffix(f".{suffix}"), dpi=180, bbox_inches="tight")
    print(
        json.dumps({"png": str(prefix.with_suffix(".png")), "svg": str(prefix.with_suffix(".svg"))})
    )


if __name__ == "__main__":
    main()
