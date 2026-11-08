"""Train closed-loop dynamics or guidance using validation-only selection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rollout import (  # noqa: E402
    ClosedLoopConfig,
    LATENT_KEYS,
    compute_delta_thresholds,
    curriculum_horizon,
    rollout,
    selection_value,
    teacher_forcing_probability,
)
from model import (  # noqa: E402
    TemporalSequenceDataset,
    _load_stage1_weights_into_temporal_model,
    build_temporal_model_from_dataset,
    load_normalization_stats,
    set_experiment_seed,
)
from utils import denormalize_wind  # noqa: E402
from refinement import RefinementConfig  # noqa: E402


DATASET_NAMES = {name: name for name in ("NR-I", "NR-II", "FR")}


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Config must be an object: {path}")
    return value


def read_sequence(path: Path, expected_split: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        split = str(data["split_name"])
        if split != expected_split:
            raise ValueError(f"Split mismatch for {path}: expected {expected_split}, got {split}")
        return {
            "spectra": data["spectra"].astype(np.float32),
            "wind": data["wind_speed"].astype(np.float32),
            "step_indices": data["step_indices"].astype(np.int64),
            "split": split,
        }


def refinement_from_config(section: Mapping[str, Any], enabled_override: Optional[bool] = None) -> RefinementConfig:
    values = dict(section)
    if "variable" in values and "optimize_target" not in values:
        variable = values.pop("variable")
        values["optimize_target"] = "peak_pos_only" if variable == "peak_pos" else str(variable)
    aliases = {
        "max_shift_bins": "max_peak_pos_shift_bins",
        "anchor_weight": "lambda_anchor",
        "max_steps": "num_steps",
    }
    for old, new in aliases.items():
        if old in values and new not in values:
            values[new] = values.pop(old)
    if enabled_override is not None:
        values["enabled"] = enabled_override
    return RefinementConfig.from_mapping(values)


def make_rollout_config(
    config: Mapping[str, Any],
    horizon: int,
    gt_probability: float,
    *,
    train: bool,
    delta_thresholds: Mapping[str, float],
) -> ClosedLoopConfig:
    stage2 = config["stage2"]
    direct = config.get("direct_branch", {})
    loss = config.get("loss", {})
    refinement_section = config.get("refinement", {})
    refinement = refinement_from_config(
        refinement_section,
        enabled_override=bool(refinement_section.get("enabled", False)) and (
            not train
            or bool(refinement_section.get("enable_during_training", False))
            or bool(refinement_section.get("propagate_refined_latent", False))
        ),
    )
    return ClosedLoopConfig(
        rollout_horizon=int(horizon),
        feedback_strategy=str(stage2["feedback_strategy"]) if train else "free_running",
        teacher_forcing_probability=float(gt_probability) if train else 0.0,
        rollout_loss_discount=float(stage2.get("rollout_loss_discount", 0.98)),
        dual_wind_history=bool(stage2.get("dual_wind_history", True)),
        shared_feedback_branch=str(stage2.get("shared_feedback_branch", "decode")),
        latent_input_mode=str(direct.get("latent_input_mode", "predicted")),
        direct_only=bool(direct.get("direct_only", False)),
        physics_only=bool(direct.get("physics_only", False)),
        full_path_mask_latents=tuple(config.get("ablation", {}).get("full_path_mask_latents", ())),
        direct_output_mode=str(direct.get("output_mode", "absolute")),
        direct_residual_scale=float(direct.get("residual_scale", 1.0)),
        direct_loss_type=str(direct.get("loss_type", "mse")),
        decode_loss_type=str(config.get("decode_branch", {}).get("loss_type", "mse")),
        physics_decode_weight=float(config.get("decode_branch", {}).get("physics_weight", 1.0)),
        guidance_enabled=bool(config.get("guidance", {}).get("enabled", False)),
        guidance_max_horizon=int(config.get("guidance", {}).get("max_horizon", 56)),
        guidance_gate_loss_type=str(config.get("guidance", {}).get("gate_loss_type", "bce")),
        propagate_refined_latent=bool(refinement_section.get("propagate_refined_latent", False)),
        lambda_direct=float(loss.get("lambda_direct", 1.0)),
        lambda_decode=float(loss.get("lambda_decode", 1.0)),
        lambda_latent_rollout=float(loss.get("lambda_latent_rollout", 1.0)),
        lambda_consistency=float(loss.get("lambda_consistency", 0.1)),
        lambda_delta=float(loss.get("lambda_delta", 0.0)),
        lambda_raw_decode=float(loss.get("lambda_raw_decode", 0.25)),
        lambda_guidance_gate=float(loss.get("lambda_guidance_gate", 0.5)),
        lambda_guidance_anchor=float(loss.get("lambda_guidance_anchor", 0.01)),
        delta_thresholds=dict(delta_thresholds),
        refinement=refinement,
    )


def build_model(config: Mapping[str, Any], dataset: TemporalSequenceDataset, stage1: Path, device: torch.device):
    model_cfg = config["model"]
    model = build_temporal_model_from_dataset(
        sample=dataset[0],
        d_model=int(model_cfg.get("d_model", 64)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        temporal_hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        decoder_mode=str(model_cfg.get("decoder_mode", "physics")),
        latent_layout=str(model_cfg.get("latent_layout", "structured")),
        lambda_cons=float(config.get("loss", {}).get("lambda_cons", 0.5)),
        mi_weight=float(config.get("loss", {}).get("mi_weight", 0.1)),
        direct_transition_type=str(model_cfg.get("direct_transition_type", "gru")),
        direct_input_steps=int(config["data"].get("input_steps", 12)),
        direct_dlinear_kernel=int(model_cfg.get("direct_dlinear_kernel", 3)),
        guidance_enabled=bool(config.get("guidance", {}).get("enabled", False)),
        guidance_hidden_dim=int(config.get("guidance", {}).get("hidden_dim", 128)),
        guidance_residual_scale=float(config.get("guidance", {}).get("residual_scale", 1.0)),
        guidance_hard_gate=bool(config.get("guidance", {}).get("hard_gate", False)),
        guidance_gate_temperature=float(config.get("guidance", {}).get("gate_temperature", 1.0)),
        guidance_output_mode=str(config.get("guidance", {}).get("output_mode", "expert_mix")),
        guidance_mask_latents=tuple(config.get("guidance", {}).get("mask_latents", ())),
    ).to(device)
    _load_stage1_weights_into_temporal_model(model, stage1, device)
    for parameter in model.single_step_model.parameters():
        parameter.requires_grad_(False)
    if bool(config.get("stage2", {}).get("finetune_decoder", False)):
        for parameter in model.decoder.parameters():
            parameter.requires_grad_(True)
    model.single_step_model.eval()
    if bool(config.get("direct_branch", {}).get("direct_only", False)):
        for parameter in model.latent_transition.parameters():
            parameter.requires_grad_(False)
    if bool(config.get("direct_branch", {}).get("physics_only", False)):
        for parameter in model.direct_transition.parameters():
            parameter.requires_grad_(False)
    return model


def encode_future(model, future_spectrum: torch.Tensor) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        model.single_step_model.eval()
        return model.encode_history(future_spectrum)


def train_epoch(
    model,
    loader: DataLoader,
    optimizers: Sequence[torch.optim.Optimizer],
    rollout_config: ClosedLoopConfig,
    device: torch.device,
    gradient_clip: float,
    gradient_clip_mode: str,
    generator: torch.Generator,
    max_batches: Optional[int],
) -> Dict[str, float]:
    model.direct_transition.train()
    model.latent_transition.train()
    if getattr(model, "guided_decoder", None) is not None:
        model.guided_decoder.train()
    model.single_step_model.eval()
    decoder_trainable = any(parameter.requires_grad for parameter in model.decoder.parameters())
    if decoder_trainable:
        model.decoder.train()
    totals: Dict[str, float] = {}
    count = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        spectrum_history = batch["spectrum_history"].to(device)
        past_wind = batch["past_wind"].to(device)
        future_spectrum = batch["future_spectrum"][:, : rollout_config.rollout_horizon].to(device)
        future_wind = batch["future_wind"][:, : rollout_config.rollout_horizon].to(device)
        with torch.no_grad():
            history_latent = model.encode_history(spectrum_history)
            teacher_latent = encode_future(model, future_spectrum)
        valid = torch.ones(future_wind.shape[:2], dtype=torch.bool, device=device)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        output = rollout(
            model,
            history_latent,
            past_wind,
            rollout_config,
            future_wind=future_wind,
            teacher_latent=teacher_latent,
            valid_mask=valid,
            generator=generator,
        )
        loss = output["losses"]["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite Stage-II loss at batch {batch_idx}")
        loss.backward()
        if gradient_clip > 0.0:
            if gradient_clip_mode == "separate":
                torch.nn.utils.clip_grad_norm_(model.direct_transition.parameters(), gradient_clip)
                torch.nn.utils.clip_grad_norm_(model.latent_transition.parameters(), gradient_clip)
                if decoder_trainable:
                    torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), gradient_clip)
                if getattr(model, "guided_decoder", None) is not None:
                    torch.nn.utils.clip_grad_norm_(model.guided_decoder.parameters(), gradient_clip)
            elif gradient_clip_mode == "joint":
                trainable_decoder = [parameter for parameter in model.decoder.parameters() if parameter.requires_grad]
                guidance_parameters = list(model.guided_decoder.parameters()) if getattr(model, "guided_decoder", None) is not None else []
                torch.nn.utils.clip_grad_norm_(
                    list(model.direct_transition.parameters()) + list(model.latent_transition.parameters()) + trainable_decoder + guidance_parameters,
                    gradient_clip,
                )
            else:
                raise ValueError("train.gradient_clip_mode must be 'joint' or 'separate'")
        for optimizer in optimizers:
            optimizer.step()
        for key, value in output["losses"].items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().item())
        totals["teacher_fraction"] = totals.get("teacher_fraction", 0.0) + float(output["teacher_mask"].float().mean().item())
        totals["history_separation"] = totals.get("history_separation", 0.0) + float(
            (output["feedback_direct"] - output["feedback_decode"]).abs().mean().detach().item()
        )
        count += 1
    if count == 0:
        raise RuntimeError("No training batches were executed")
    return {key: value / count for key, value in totals.items()}


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    residual = prediction - target
    ss_res = float(np.square(residual).sum())
    ss_tot = float(np.square(target - target.mean()).sum())
    return {
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 0.0,
        "max_abs_prediction": float(np.abs(prediction).max()),
    }


@torch.no_grad()
def evaluate_one_step(
    model,
    sequence_path: Path,
    split: str,
    stats: Mapping[str, float],
    config: Mapping[str, Any],
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    input_steps = int(config["data"].get("input_steps", 12))
    dataset = TemporalSequenceDataset(str(sequence_path), input_steps, 1, 1, expected_split=split)
    loader = DataLoader(dataset, batch_size=int(config["train"].get("batch_size", 16)), shuffle=False)
    branch_rows: Dict[str, List[np.ndarray]] = {key: [] for key in ("direct", "raw_decode", "decode", "refined")}
    targets: List[np.ndarray] = []
    model.eval()
    rollout_cfg = make_rollout_config(config, 1, 0.0, train=False, delta_thresholds={})
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        spectra = batch["spectrum_history"].to(device)
        past_wind = batch["past_wind"].to(device)
        history = model.encode_history(spectra)
        output = rollout(model, history, past_wind, rollout_cfg)
        for branch in branch_rows:
            branch_rows[branch].append(output[branch][:, 0].cpu().numpy())
        targets.append(batch["future_wind"][:, 0].numpy())
    target = denormalize_wind(np.concatenate(targets), dict(stats))
    result: Dict[str, float] = {}
    for branch, rows in branch_rows.items():
        prediction = denormalize_wind(np.concatenate(rows), dict(stats))
        metrics = regression_metrics(target, prediction)
        for key, value in metrics.items():
            result[f"{branch}_{key}"] = value
        per_window_mae = np.abs(prediction - target).mean(axis=1)
        result[f"{branch}_window_mae_std"] = float(per_window_mae.std(ddof=1)) if len(per_window_mae) > 1 else 0.0
    return result


@torch.no_grad()
def evaluate_full_loop(
    model,
    sequence_path: Path,
    split: str,
    stats: Mapping[str, float],
    config: Mapping[str, Any],
    device: torch.device,
    train_stability: Mapping[str, float],
) -> Tuple[Dict[str, float], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """Only the initial observed window enters the model; labels are read after rollout."""
    sequence = read_sequence(sequence_path, split)
    input_steps = int(config["data"].get("input_steps", 12))
    horizon = int(len(sequence["wind"]) - input_steps)
    spectra0 = torch.from_numpy(sequence["spectra"][:input_steps]).unsqueeze(0).to(device)
    wind0 = torch.from_numpy(sequence["wind"][:input_steps]).unsqueeze(0).to(device)
    model.eval()
    history = model.encode_history(spectra0)
    rollout_cfg = make_rollout_config(config, horizon, 0.0, train=False, delta_thresholds={})
    output = rollout(model, history, wind0, rollout_cfg)
    target_norm = sequence["wind"][input_steps:]
    target = denormalize_wind(target_norm, dict(stats))
    arrays: Dict[str, np.ndarray] = {"target": target}
    horizon_rows: List[Dict[str, Any]] = []
    state_rows: List[Dict[str, Any]] = []
    metrics: Dict[str, float] = {"horizon": float(horizon)}
    for branch in ("direct", "raw_decode", "decode", "refined"):
        pred_norm = output[branch][0].cpu().numpy()
        pred = denormalize_wind(pred_norm, dict(stats))
        arrays[branch] = pred
        branch_metrics = regression_metrics(target, pred)
        for key, value in branch_metrics.items():
            metrics[f"{branch}_{key}"] = value
        step_mae = np.abs(pred - target).mean(axis=1)
        metrics[f"{branch}_step1_mae"] = float(step_mae[0])
        metrics[f"{branch}_first5_mae"] = float(step_mae[: min(5, horizon)].mean())
        metrics[f"{branch}_first10_mae"] = float(step_mae[: min(10, horizon)].mean())
        middle = horizon // 2
        metrics[f"{branch}_middle_mae"] = float(step_mae[middle])
        metrics[f"{branch}_final_mae"] = float(step_mae[-1])
        metrics[f"{branch}_first_half_mae"] = float(step_mae[: max(middle, 1)].mean())
        metrics[f"{branch}_second_half_mae"] = float(step_mae[middle:].mean())
        for step, value in enumerate(step_mae, 1):
            horizon_rows.append({"split": split, "horizon": step, "branch": branch, "mae": float(value)})

    latent_norm = output["latent_norm"][0].cpu().numpy()
    disagreement = output["direct_decode_disagreement"][0].cpu().numpy()
    shifts = output["refined_shift"][0].cpu().numpy()
    peak_pos_shifts = output["peak_pos_shift"][0].cpu().numpy()
    direct_confidence = output["direct_confidence"][0].cpu().numpy()
    guidance_gate = output["guidance_gate"][0].cpu().numpy()
    guidance_residual = output["guidance_residual"][0].cpu().numpy()
    direct_decode_gap = (output["feedback_direct"] - output["feedback_decode"]).abs().mean(dim=2)[0].cpu().numpy()
    for step in range(horizon):
        state_rows.append({
            "split": split,
            "horizon": step + 1,
            "latent_norm": float(latent_norm[step]),
            "direct_decode_disagreement": float(disagreement[step]),
            "refined_shift": float(shifts[step]),
            "peak_position_shift": float(peak_pos_shifts[step]),
            "direct_confidence": float(direct_confidence[step]),
            "wind_history_separation": float(direct_decode_gap[step]),
            "guidance_gate_mean": float(guidance_gate[step].mean()),
            "guidance_residual_abs_mean": float(np.abs(guidance_residual[step]).mean()),
        })
    latent_limit = float(train_stability.get("latent_norm_q99", np.inf))
    delta_limit = float(train_stability.get("wind_delta_q99_physical", np.inf))
    refined = arrays["refined"]
    adjacent = np.abs(np.diff(refined, axis=0)) if horizon > 1 else np.zeros_like(refined)
    finite = np.isfinite(refined)
    train_min = float(train_stability.get("wind_min_physical", -np.inf))
    train_max = float(train_stability.get("wind_max_physical", np.inf))
    explosion = (refined < train_min) | (refined > train_max)
    growth = metrics["refined_second_half_mae"] / max(metrics["refined_first_half_mae"], 1e-12) - 1.0
    metrics.update({
        "nan_inf_ratio": float(1.0 - finite.mean()),
        "explosion_ratio": float(explosion.mean()),
        "large_adjacent_change_ratio": float((adjacent > delta_limit).mean()),
        "late_early_mae_growth": float(growth),
        "latent_norm_outlier_ratio": float((latent_norm > latent_limit).mean()),
        "window_mae_std": float(np.abs(refined - target).mean(axis=1).std(ddof=1)) if horizon > 1 else 0.0,
        "mean_peak_position_shift": float(peak_pos_shifts.mean()),
        "mean_joint_refined_latent_shift": float(shifts.mean()),
        "mean_direct_confidence": float(direct_confidence.mean()),
        "mean_direct_decode_disagreement": float(disagreement.mean()),
        "mean_guidance_gate": float(guidance_gate.mean()),
        "mean_guidance_residual_abs": float(np.abs(guidance_residual).mean()),
    })
    if "raw_decode" in arrays:
        direct_better = np.abs(arrays["direct"] - target) < np.abs(arrays["raw_decode"] - target)
        metrics["direct_advantage_ratio"] = float(direct_better.mean())
        metrics["guidance_gate_accuracy"] = float(((guidance_gate >= 0.5) == direct_better).mean())
    metrics["instability"] = (
        metrics["explosion_ratio"]
        + metrics["large_adjacent_change_ratio"]
        + max(metrics["late_early_mae_growth"], 0.0)
        + metrics["latent_norm_outlier_ratio"]
    )
    return metrics, horizon_rows, state_rows, arrays


@torch.no_grad()
def compute_train_stability(model, train_sequence: Mapping[str, Any], stats: Mapping[str, float], device: torch.device) -> Dict[str, float]:
    spectra = train_sequence["spectra"]
    wind_physical = denormalize_wind(train_sequence["wind"], dict(stats))
    norms: List[np.ndarray] = []
    model.eval()
    for start in range(0, len(spectra), 32):
        encoded = model.encoder(torch.from_numpy(spectra[start : start + 32]).to(device))
        norm = torch.sqrt(sum(value.reshape(value.shape[0], -1).pow(2).sum(dim=1) for value in encoded.values()))
        norms.append(norm.cpu().numpy())
    delta = np.abs(np.diff(wind_physical, axis=0)).reshape(-1)
    margin = 3.0 * float(np.std(wind_physical))
    return {
        "latent_norm_q99": float(np.quantile(np.concatenate(norms), 0.99)),
        "wind_delta_q99_physical": float(np.quantile(delta, 0.99)),
        "wind_min_physical": float(wind_physical.min() - margin),
        "wind_max_physical": float(wind_physical.max() + margin),
    }


def snapshot_info() -> Dict[str, Any]:
    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        git_head = ""
    return {
        "git_head": git_head if len(git_head) == 40 else "unavailable-uncommitted-parent-repository",
        "source_sha256": sha256(ROOT / "rollout.py"),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASET_NAMES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-init-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    config = read_json(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / "data" / "ready" / args.dataset
    input_steps = int(config["data"].get("input_steps", 12))
    curriculum = [int(value) for value in config["stage2"].get("rollout_curriculum", [config["stage2"]["rollout_horizon"]])]
    curriculum_epochs = [int(value) for value in config["stage2"].get("curriculum_epochs", [config["train"]["stage2_epochs"]])]
    total_epochs = int(config["train"].get("stage2_epochs", sum(curriculum_epochs)))
    if sum(curriculum_epochs) != total_epochs:
        raise ValueError("sum(curriculum_epochs) must equal train.stage2_epochs")
    max_horizon = max(curriculum)
    if int(config["stage2"].get("rollout_horizon", max_horizon)) != max_horizon:
        raise ValueError("stage2.rollout_horizon must equal max(stage2.rollout_curriculum)")
    train_dataset = TemporalSequenceDataset(str(data_dir / "train_sequence.npz"), input_steps, max_horizon, 1, expected_split="train")
    train_loader = DataLoader(train_dataset, batch_size=int(config["train"].get("batch_size", 16)), shuffle=False)
    device = torch.device(args.device)
    set_experiment_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(config, train_dataset, args.stage1_checkpoint, device)
    stage2_init_audit: Dict[str, Any] = {"enabled": False}
    if args.stage2_init_checkpoint is not None:
        init_payload = torch.load(args.stage2_init_checkpoint, map_location=device, weights_only=False)
        incompatible = model.load_state_dict(init_payload["model_state_dict"], strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = list(incompatible.missing_keys)
        if unexpected or any(not key.startswith("guided_decoder.") for key in missing):
            raise RuntimeError(f"Incompatible Stage-II init: missing={missing}, unexpected={unexpected}")
        stage2_init_audit = {
            "enabled": True,
            "path": str(args.stage2_init_checkpoint),
            "sha256": sha256(args.stage2_init_checkpoint),
            "source_epoch": int(init_payload.get("epoch", -1)),
            "missing_new_guidance_keys": missing,
        }
    train_sequence = read_sequence(data_dir / "train_sequence.npz", "train")
    stats = load_normalization_stats(str(data_dir / "normalization_stats.npz"))
    delta_thresholds = compute_delta_thresholds(
        model,
        train_sequence["spectra"],
        device,
        quantile=float(config.get("stability", {}).get("delta_quantile", 0.99)),
    )
    train_stability = compute_train_stability(model, train_sequence, stats, device)
    direct_optimizer = torch.optim.Adam(model.direct_transition.parameters(), lr=float(config["train"].get("direct_lr", config["train"].get("lr", 1e-3))))
    latent_optimizer = torch.optim.Adam(model.latent_transition.parameters(), lr=float(config["train"].get("latent_lr", config["train"].get("lr", 1e-3))))
    optimizers: List[torch.optim.Optimizer] = [direct_optimizer, latent_optimizer]
    guidance_optimizer: Optional[torch.optim.Optimizer] = None
    if getattr(model, "guided_decoder", None) is not None:
        guidance_optimizer = torch.optim.Adam(
            model.guided_decoder.parameters(),
            lr=float(config["train"].get("guidance_lr", config["train"].get("lr", 1e-3))),
        )
        optimizers.append(guidance_optimizer)
    decoder_optimizer: Optional[torch.optim.Optimizer] = None
    trainable_decoder = [parameter for parameter in model.decoder.parameters() if parameter.requires_grad]
    if trainable_decoder:
        decoder_optimizer = torch.optim.Adam(
            trainable_decoder,
            lr=float(config["train"].get("decoder_lr", 1e-4)),
        )
        optimizers.append(decoder_optimizer)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(args.seed + 100003)
    best_value = float("inf")
    best_epoch = 0
    epoch_rows: List[Dict[str, Any]] = []
    start_time = time.time()
    selection = config.get("selection", {"criterion": "val_long_loop_refined_mae"})
    selection_name = str(selection.get("criterion", "val_long_loop_refined_mae")).replace("/", "-")
    checkpoint_stem = f"{args.dataset}_seed{args.seed}_h{max_horizon}_sel-{selection_name}"
    best_checkpoint_path = output_dir / f"{checkpoint_stem}.best.pt"
    last_checkpoint_path = output_dir / f"{checkpoint_stem}.last.pt"
    resolved = {
        **config,
        "runtime": {
            "dataset": args.dataset,
            "dataset_name": DATASET_NAMES[args.dataset],
            "seed": args.seed,
            "stage1_checkpoint": str(args.stage1_checkpoint),
            "stage1_sha256": sha256(args.stage1_checkpoint),
            "delta_thresholds_train_only": delta_thresholds,
            "train_stability_thresholds": train_stability,
            "scheduled_sampling_probability_semantics": "probability of writing ground truth feedback",
            "test_accessed_during_selection": False,
            "stage2_initialization": stage2_init_audit,
            **snapshot_info(),
        },
    }
    (output_dir / "config_resolved.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")

    for epoch in range(1, total_epochs + 1):
        horizon = min(curriculum_horizon(curriculum, curriculum_epochs, epoch), max_horizon)
        gt_probability = teacher_forcing_probability(
            float(config["stage2"].get("scheduled_sampling_start", 1.0)),
            float(config["stage2"].get("scheduled_sampling_end", 0.1)),
            epoch,
            total_epochs,
            str(config["stage2"].get("scheduled_sampling_schedule", "linear")),
        )
        rollout_cfg = make_rollout_config(config, horizon, gt_probability, train=True, delta_thresholds=delta_thresholds)
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizers,
            rollout_cfg,
            device,
            float(config["train"].get("gradient_clip", 1.0)),
            str(config["train"].get("gradient_clip_mode", "joint")),
            generator,
            args.max_train_batches,
        )
        one = evaluate_one_step(model, data_dir / "val_sequence.npz", "val", stats, config, device, args.max_val_batches)
        full, horizon_rows, state_rows, _ = evaluate_full_loop(
            model, data_dir / "val_sequence.npz", "val", stats, config, device, train_stability
        )
        selection_metrics = {
            "val_one_step_direct_mae": one["direct_mae"],
            "val_one_step_raw_decode_mae": one["raw_decode_mae"],
            "val_one_step_decode_mae": one["decode_mae"],
            "val_one_step_refined_mae": one["refined_mae"],
            "val_long_loop_direct_mae": full["direct_mae"],
            "val_long_loop_raw_decode_mae": full["raw_decode_mae"],
            "val_long_loop_decode_mae": full["decode_mae"],
            "val_long_loop_refined_mae": full["refined_mae"],
            "val_instability": full["instability"],
        }
        current = selection_value(selection_metrics, selection)
        row = {
            "epoch": epoch,
            "rollout_horizon": horizon,
            "gt_feedback_probability": gt_probability,
            "selection_value": current,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **selection_metrics,
            "val_one_step_window_mae_std": one["refined_window_mae_std"],
            "val_full_loop_horizon_mae_std": full["window_mae_std"],
            "val_max_abs_prediction": full["refined_max_abs_prediction"],
            "val_explosion_ratio": full["explosion_ratio"],
            "val_late_early_growth": full["late_early_mae_growth"],
            "val_latent_norm_outlier_ratio": full["latent_norm_outlier_ratio"],
        }
        epoch_rows.append(row)
        payload = {
            "phase": "rollout_stage2",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "direct_optimizer_state_dict": direct_optimizer.state_dict(),
            "latent_optimizer_state_dict": latent_optimizer.state_dict(),
            "decoder_optimizer_state_dict": decoder_optimizer.state_dict() if decoder_optimizer is not None else None,
            "guidance_optimizer_state_dict": guidance_optimizer.state_dict() if guidance_optimizer is not None else None,
            "config": resolved,
            "selection_metrics": selection_metrics,
            "selection_value": current,
        }
        torch.save(payload, last_checkpoint_path)
        if current < best_value:
            best_value = current
            best_epoch = epoch
            torch.save(payload, best_checkpoint_path)
            write_csv(output_dir / "val_horizon_mae_best.csv", horizon_rows)
            write_csv(output_dir / "val_state_drift_best.csv", state_rows)
        print(
            f"epoch={epoch:03d} K={horizon} train={train_metrics['total']:.6f} "
            f"val1={one['refined_mae']:.4f} val_loop={full['refined_mae']:.4f} select={current:.4f}",
            flush=True,
        )

    write_csv(output_dir / "training_curve.csv", epoch_rows)
    # The file was created by this process; PyTorch >=2.6 otherwise defaults
    # to weights_only=True and rejects optimizer/config metadata.
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_one = evaluate_one_step(model, data_dir / "val_sequence.npz", "val", stats, config, device)
    val_full, val_horizon, val_state, val_arrays = evaluate_full_loop(
        model, data_dir / "val_sequence.npz", "val", stats, config, device, train_stability
    )
    write_csv(output_dir / "validation_horizon_mae.csv", val_horizon)
    write_csv(output_dir / "validation_state_drift.csv", val_state)
    np.savez_compressed(output_dir / "validation_predictions.npz", **val_arrays)
    summary: Dict[str, Any] = {
        "status": "completed",
        "selection_split": "validation",
        "test_accessed_during_selection": False,
        "best_epoch": best_epoch,
        "best_selection_value": best_value,
        "selection": selection,
        "validation_one_step": val_one,
        "validation_full_loop": val_full,
        "training_seconds": time.time() - start_time,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "best_checkpoint": str(best_checkpoint_path),
        "best_checkpoint_sha256": sha256(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
    }
    if args.evaluate_test:
        test_one = evaluate_one_step(model, data_dir / "test_sequence.npz", "test", stats, config, device)
        test_full, test_horizon, test_state, test_arrays = evaluate_full_loop(
            model, data_dir / "test_sequence.npz", "test", stats, config, device, train_stability
        )
        write_csv(output_dir / "test_horizon_mae.csv", test_horizon)
        write_csv(output_dir / "test_state_drift.csv", test_state)
        np.savez_compressed(output_dir / "test_predictions.npz", **test_arrays)
        summary["test_evaluated_after_selection"] = True
        summary["test_one_step"] = test_one
        summary["test_full_loop"] = test_full
    else:
        summary["test_evaluated_after_selection"] = False
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"completed best_epoch={best_epoch} best_validation_selection={best_value:.6f}", flush=True)


if __name__ == "__main__":
    main()
