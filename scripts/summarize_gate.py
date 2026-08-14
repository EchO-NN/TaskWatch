#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean


TRAIN_METRICS = (
    "loss_total",
    "positive_cosine_mean",
    "negative_cosine_mean",
    "embedding_std_pred",
    "embedding_std_target",
)


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: fmean(float(row[key]) for row in rows) for key in TRAIN_METRICS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train_log = output_dir / "train.jsonl"
    rows = [
        json.loads(line)
        for line in train_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < args.window * 2:
        raise ValueError(f"need at least {args.window * 2} logged rows, found {len(rows)}")

    retrieval_paths = sorted(output_dir.glob("retrieval_step_*.json"))
    if not retrieval_paths:
        raise FileNotFoundError(f"no retrieval_step_*.json under {output_dir}")
    retrievals = [json.loads(path.read_text(encoding="utf-8")) for path in retrieval_paths]
    initial_retrieval = min(retrievals, key=lambda item: int(item["step"]))
    final_retrieval = max(retrievals, key=lambda item: int(item["step"]))

    first_window = mean_metrics(rows[: args.window])
    last_window = mean_metrics(rows[-args.window :])
    finite = all(math.isfinite(float(row[key])) for row in rows for key in TRAIN_METRICS)
    min_pred_std = min(float(row["embedding_std_pred"]) for row in rows)
    min_target_std = min(float(row["embedding_std_target"]) for row in rows)
    random_r1 = float(final_retrieval["random_r1"])
    v2t_gain = float(final_retrieval["video_to_text_r1"]) / random_r1
    t2v_gain = float(final_retrieval["text_to_video_r1"]) / random_r1

    # The spec gives qualitative gate requirements. These explicit numerical
    # thresholds are engineering acceptance criteria and are reported as such.
    checks = {
        "all_logged_metrics_finite": finite,
        "last_window_loss_below_first_window": (
            last_window["loss_total"] < first_window["loss_total"]
        ),
        "last_window_positive_cosine_above_first_window": (
            last_window["positive_cosine_mean"] > first_window["positive_cosine_mean"]
        ),
        "embedding_std_never_below_1e-4": (min_pred_std > 1.0e-4 and min_target_std > 1.0e-4),
        "both_retrieval_r1_at_least_5x_random": (v2t_gain >= 5.0 and t2v_gain >= 5.0),
    }
    summary = {
        "gate_passed": all(checks.values()),
        "engineering_acceptance_checks": checks,
        "train": {
            "logged_rows": len(rows),
            "first_step": int(rows[0]["step"]),
            "final_step": int(rows[-1]["step"]),
            "samples_seen": int(rows[-1]["samples_seen"]),
            "window_size": args.window,
            "first_window_mean": first_window,
            "last_window_mean": last_window,
            "loss_ratio_last_over_first": (last_window["loss_total"] / first_window["loss_total"]),
            "minimum_embedding_std_pred": min_pred_std,
            "minimum_embedding_std_target": min_target_std,
            "gpu_peak_memory_gib": max(float(row["gpu_peak_memory_gib"]) for row in rows),
        },
        "retrieval": {
            "initial": initial_retrieval,
            "final": final_retrieval,
            "video_to_text_r1_over_random": v2t_gain,
            "text_to_video_r1_over_random": t2v_gain,
        },
        "threshold_note": (
            "The loss/similarity direction, 1e-4 std floor, and 5x-random R@1 "
            "rules are explicit engineering acceptance criteria; the source "
            "spec states these requirements qualitatively."
        ),
    }
    output = Path(args.output) if args.output else output_dir / "gate_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "gate_passed": summary["gate_passed"]}))


if __name__ == "__main__":
    main()
