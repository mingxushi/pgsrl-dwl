"""Evaluate a validation-selected Phase-3 checkpoint on test after configuration freeze."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from train_stage2 import (
    ROOT, TemporalSequenceDataset, build_model, evaluate_full_loop, evaluate_one_step,
    load_normalization_stats, read_json, sha256, write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    source_metrics = read_json(args.source_run / "metrics.json")
    if source_metrics.get("test_accessed_during_selection") or source_metrics.get("test_evaluated_after_selection"):
        raise RuntimeError("Source Phase-3 run must be validation-only")
    resolved = read_json(args.source_run / "config_resolved.json")
    runtime = resolved["runtime"]
    dataset, seed = str(runtime["dataset"]), int(runtime["seed"])
    config = {key: value for key, value in resolved.items() if key != "runtime"}
    data_dir = ROOT / "data" / "ready" / dataset
    max_horizon = max(int(value) for value in config["stage2"]["rollout_curriculum"])
    train_dataset = TemporalSequenceDataset(str(data_dir / "train_sequence.npz"), int(config["data"]["input_steps"]),
                                             max_horizon, 1, expected_split="train")
    device = torch.device(args.device)
    stage1 = Path(runtime["stage1_checkpoint"])
    model = build_model(config, train_dataset, stage1, device)
    source_checkpoint = Path(source_metrics["best_checkpoint"])
    checkpoint = torch.load(source_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    stats = load_normalization_stats(str(data_dir / "normalization_stats.npz"))
    train_stability = runtime["train_stability_thresholds"]
    test_one = evaluate_one_step(model, data_dir / "test_sequence.npz", "test", stats, config, device)
    test_full, test_horizon, test_state, test_arrays = evaluate_full_loop(
        model, data_dir / "test_sequence.npz", "test", stats, config, device, train_stability
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = source_checkpoint.name
    final_checkpoint = args.output_dir / checkpoint_name
    shutil.copy2(source_checkpoint, final_checkpoint)
    for name in ("config_resolved.json", "training_curve.csv", "validation_horizon_mae.csv",
                 "validation_state_drift.csv", "validation_predictions.npz"):
        shutil.copy2(args.source_run / name, args.output_dir / name)
    write_csv(args.output_dir / "test_horizon_mae.csv", test_horizon)
    write_csv(args.output_dir / "test_state_drift.csv", test_state)
    np.savez_compressed(args.output_dir / "test_predictions.npz", **test_arrays)
    summary = dict(source_metrics)
    summary.update({
        "best_checkpoint": str(final_checkpoint), "best_checkpoint_sha256": sha256(final_checkpoint),
        "test_evaluated_after_selection": True, "test_one_step": test_one, "test_full_loop": test_full,
        "formal_reuse": "Phase-3 checkpoint selected without test access; evaluated only after final configuration freeze",
        "formal_source_run": str(args.source_run), "dataset": dataset, "seed": seed,
    })
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": dataset, "seed": seed, "refined_mae": test_full["refined_mae"],
                      "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
