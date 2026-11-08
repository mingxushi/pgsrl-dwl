"""Closed-loop Stage-II training and evaluation utilities.

One rollout implementation is shared by training, validation and testing,
with separate direct and guided wind histories.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from model import (
    DisentangledTemporalWindForecaster,
    _update_history_latent,
    _update_wind_history,
)
from refinement import RefinementConfig, refine_latent_state


LATENT_KEYS = ("z_peak", "z_width", "z_noise", "peak_pos")
FULL_PATH_MASKABLE_LATENTS = LATENT_KEYS
LATENT_MODES = ("predicted", "detached_predicted", "initial_static", "wind_only")
FEEDBACK_STRATEGIES = ("full_teacher_forcing", "scheduled_sampling", "free_running")
DIRECT_OUTPUT_MODES = ("absolute", "residual_tanh")
DIRECT_LOSS_TYPES = ("mse", "mae", "smooth_l1")


@dataclass
class ClosedLoopConfig:
    rollout_horizon: int = 1
    feedback_strategy: str = "full_teacher_forcing"
    teacher_forcing_probability: float = 1.0
    rollout_loss_discount: float = 0.98
    dual_wind_history: bool = True
    shared_feedback_branch: str = "decode"
    latent_input_mode: str = "predicted"
    direct_only: bool = False
    physics_only: bool = False
    full_path_mask_latents: tuple[str, ...] = ()
    direct_output_mode: str = "absolute"
    direct_residual_scale: float = 1.0
    direct_loss_type: str = "mse"
    decode_loss_type: str = "mse"
    physics_decode_weight: float = 1.0
    guidance_enabled: bool = False
    guidance_max_horizon: int = 56
    guidance_gate_loss_type: str = "bce"
    propagate_refined_latent: bool = False
    lambda_direct: float = 1.0
    lambda_decode: float = 1.0
    lambda_latent_rollout: float = 1.0
    lambda_consistency: float = 0.1
    lambda_delta: float = 0.0
    lambda_raw_decode: float = 0.25
    lambda_guidance_gate: float = 0.5
    lambda_guidance_anchor: float = 0.01
    delta_thresholds: Dict[str, float] = field(default_factory=dict)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)

    def validate(self) -> None:
        if self.rollout_horizon <= 0:
            raise ValueError("rollout_horizon must be positive")
        if self.feedback_strategy not in FEEDBACK_STRATEGIES:
            raise ValueError(f"Unsupported feedback strategy: {self.feedback_strategy}")
        if not 0.0 <= float(self.teacher_forcing_probability) <= 1.0:
            raise ValueError("teacher_forcing_probability is the GT probability and must be in [0, 1]")
        if not 0.0 < float(self.rollout_loss_discount) <= 1.0:
            raise ValueError("rollout_loss_discount must be in (0, 1]")
        if self.latent_input_mode not in LATENT_MODES:
            raise ValueError(f"Unsupported latent_input_mode: {self.latent_input_mode}")
        if self.direct_only and self.physics_only:
            raise ValueError("direct_only and physics_only are mutually exclusive")
        if len(set(self.full_path_mask_latents)) != len(self.full_path_mask_latents):
            raise ValueError("full_path_mask_latents must not contain duplicates")
        unsupported_masks = sorted(set(self.full_path_mask_latents) - set(FULL_PATH_MASKABLE_LATENTS))
        if unsupported_masks:
            raise ValueError(
                f"Unsupported full-path latent masks: {unsupported_masks}; "
                f"expected a subset of {FULL_PATH_MASKABLE_LATENTS}"
            )
        if self.full_path_mask_latents and self.refinement.enabled:
            raise ValueError("full-path latent ablations require refinement.enabled=false")
        if self.direct_output_mode not in DIRECT_OUTPUT_MODES:
            raise ValueError(f"Unsupported direct_output_mode: {self.direct_output_mode}")
        if float(self.direct_residual_scale) <= 0.0:
            raise ValueError("direct_residual_scale must be positive")
        if self.direct_loss_type not in DIRECT_LOSS_TYPES:
            raise ValueError(f"Unsupported direct_loss_type: {self.direct_loss_type}")
        if self.decode_loss_type not in DIRECT_LOSS_TYPES:
            raise ValueError(f"Unsupported decode_loss_type: {self.decode_loss_type}")
        if not 0.0 <= float(self.physics_decode_weight) <= 1.0:
            raise ValueError("physics_decode_weight must be in [0, 1]")
        if self.guidance_max_horizon <= 1:
            raise ValueError("guidance_max_horizon must exceed 1")
        if self.guidance_gate_loss_type not in {"bce", "balanced_bce"}:
            raise ValueError("guidance_gate_loss_type must be bce or balanced_bce")
        if self.guidance_enabled and self.refinement.enabled:
            raise ValueError("guidance and test-time refinement cannot be enabled together")
        if self.shared_feedback_branch not in {"direct", "decode", "refined"}:
            raise ValueError("shared_feedback_branch must be direct, decode or refined")
        self.refinement.validate()


def detach_latent(latent: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: latent[key].detach() for key in LATENT_KEYS}


def _direct_latent_input(history: Mapping[str, torch.Tensor], mode: str) -> Dict[str, torch.Tensor]:
    if mode in {"predicted", "initial_static"}:
        return {key: history[key] for key in LATENT_KEYS}
    if mode == "detached_predicted":
        return detach_latent(history)
    if mode == "wind_only":
        return {key: torch.zeros_like(history[key]) for key in LATENT_KEYS}
    raise ValueError(f"Unsupported latent_input_mode: {mode}")


def _teacher_mask(
    strategy: str,
    probability: float,
    reference: torch.Tensor,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Return [B, G] mask; True means ground-truth feedback is used."""
    if strategy == "full_teacher_forcing":
        return torch.ones_like(reference, dtype=torch.bool)
    if strategy == "free_running":
        return torch.zeros_like(reference, dtype=torch.bool)
    if strategy != "scheduled_sampling":
        raise ValueError(f"Unsupported feedback strategy: {strategy}")
    return torch.rand(reference.shape, device=reference.device, generator=generator) < float(probability)


def _mix_latent(
    predicted: Mapping[str, torch.Tensor],
    teacher: Mapping[str, torch.Tensor],
    gt_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    return {
        "z_peak": torch.where(gt_mask.unsqueeze(-1), teacher["z_peak"], predicted["z_peak"]),
        "z_width": torch.where(gt_mask.unsqueeze(-1), teacher["z_width"], predicted["z_width"]),
        "z_noise": torch.where(gt_mask.unsqueeze(-1), teacher["z_noise"], predicted["z_noise"]),
        "peak_pos": torch.where(gt_mask, teacher["peak_pos"], predicted["peak_pos"]),
    }


def _apply_full_path_latent_mask(
    latent: Mapping[str, torch.Tensor],
    masked_keys: tuple[str, ...],
) -> Dict[str, torch.Tensor]:
    """Functionally remove selected latent blocks without changing dimensions."""
    return {
        key: torch.zeros_like(latent[key]) if key in masked_keys else latent[key]
        for key in LATENT_KEYS
    }


def _masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean over all non-time axes and valid [B] samples."""
    if valid.ndim != 1 or value.shape[0] != valid.shape[0]:
        raise ValueError(f"Mask/value mismatch: value={tuple(value.shape)} valid={tuple(valid.shape)}")
    per_sample = value.reshape(value.shape[0], -1).mean(dim=1)
    weight = valid.to(dtype=per_sample.dtype)
    return (per_sample * weight).sum() / weight.sum().clamp_min(1.0)


def _balanced_gate_bce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Equal positive/negative contribution over valid [B,G] gate targets."""
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    valid_gate = valid.unsqueeze(1).expand_as(target)
    positive = valid_gate & (target >= 0.5)
    negative = valid_gate & (target < 0.5)
    if bool(positive.any()) and bool(negative.any()):
        return 0.5 * loss[positive].mean() + 0.5 * loss[negative].mean()
    return loss[valid_gate].mean() if bool(valid_gate.any()) else loss.mean() * 0.0


def _latent_l1(
    predicted: Mapping[str, torch.Tensor],
    teacher: Mapping[str, torch.Tensor],
    valid: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    pieces = {
        "peak": _masked_mean((predicted["z_peak"] - teacher["z_peak"]).abs(), valid),
        "width": _masked_mean((predicted["z_width"] - teacher["z_width"]).abs(), valid),
        "noise": _masked_mean((predicted["z_noise"] - teacher["z_noise"]).abs(), valid),
        "pos": _masked_mean((predicted["peak_pos"] - teacher["peak_pos"]).abs(), valid),
    }
    pieces["total"] = sum(pieces.values())
    return pieces


def _delta_penalty(
    predicted: Mapping[str, torch.Tensor],
    previous: Mapping[str, torch.Tensor],
    thresholds: Mapping[str, float],
    valid: torch.Tensor,
) -> torch.Tensor:
    zero = torch.zeros((), device=predicted["peak_pos"].device)
    if not thresholds:
        return zero
    penalties: List[torch.Tensor] = []
    for key in LATENT_KEYS:
        if key not in thresholds:
            continue
        delta = (predicted[key] - previous[key][:, -1]).abs()
        penalties.append(_masked_mean(F.relu(delta - float(thresholds[key])), valid))
    return sum(penalties, zero) / max(len(penalties), 1)


def rollout(
    model: DisentangledTemporalWindForecaster,
    history_latent: Mapping[str, torch.Tensor],
    past_wind: torch.Tensor,
    config: ClosedLoopConfig,
    *,
    future_wind: Optional[torch.Tensor] = None,
    teacher_latent: Optional[Mapping[str, torch.Tensor]] = None,
    valid_mask: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """Run a differentiable closed loop.

    Shapes: wind tensors are [B,T,G], z_* are [B,T,G,D], peak_pos is
    [B,T,G], and ``valid_mask`` is [B,K]. Future tensors are supervision only;
    they are never read in free-running evaluation.
    """
    config.validate()
    masked_keys = tuple(config.full_path_mask_latents)
    latent_history = _apply_full_path_latent_mask(history_latent, masked_keys)
    initial_latent_history = _apply_full_path_latent_mask(history_latent, masked_keys)
    if past_wind.ndim != 3:
        raise ValueError(f"past_wind must be [B,T,G], got {tuple(past_wind.shape)}")
    batch_size, _, n_gate = past_wind.shape
    horizon = int(config.rollout_horizon)
    if future_wind is not None and future_wind.shape[1] < horizon:
        raise ValueError(f"future_wind length {future_wind.shape[1]} < horizon {horizon}")
    if teacher_latent is not None and teacher_latent["peak_pos"].shape[1] < horizon:
        raise ValueError("teacher_latent is shorter than rollout horizon")
    if valid_mask is None:
        valid_mask = torch.ones(batch_size, horizon, dtype=torch.bool, device=past_wind.device)
    if tuple(valid_mask.shape) != (batch_size, horizon):
        raise ValueError(f"valid_mask must be {(batch_size, horizon)}, got {tuple(valid_mask.shape)}")

    wind_history_direct = past_wind
    wind_history_decode = past_wind.clone()
    predictions: Dict[str, List[torch.Tensor]] = {
        key: [] for key in (
            "direct", "raw_decode", "decode", "refined", "z_peak", "z_width", "z_noise", "peak_pos",
            "feedback_direct", "feedback_decode", "teacher_mask", "latent_norm",
            "direct_decode_disagreement", "refined_shift", "peak_pos_shift", "direct_confidence",
            "guidance_gate", "guidance_residual", "direct_advantage_target",
        )
    }
    loss_rows: Dict[str, List[torch.Tensor]] = {
        key: [] for key in ("direct", "raw_decode", "decode", "consistency", "latent", "peak", "width", "noise", "pos", "delta", "guidance_gate", "guidance_anchor", "total")
    }

    for step in range(horizon):
        valid = valid_mask[:, step]
        direct_source = initial_latent_history if config.latent_input_mode == "initial_static" else latent_history
        direct_history_input = _direct_latent_input(direct_source, config.latent_input_mode)
        if config.physics_only:
            # No-direct-branch path: condition latent dynamics only on the
            # latest observed/decoded wind state.
            direct_raw = wind_history_direct[:, -1]
        else:
            direct_raw = model.direct_transition(direct_history_input, wind_history_direct)
        if config.direct_output_mode == "residual_tanh" and not config.physics_only:
            # Stable normalized-wind state update. Each recursive correction is
            # bounded by direct_residual_scale, while the network remains free
            # to choose its sign and gate-wise magnitude.
            direct = wind_history_direct[:, -1] + float(config.direct_residual_scale) * torch.tanh(direct_raw)
        else:
            direct = direct_raw
        assert direct.shape == (batch_size, n_gate)
        # Keep the direct branch a stable pseudo-teacher: decode/state losses
        # may train latent dynamics but cannot update v_direct through the
        # current-step conditioning edge.
        if config.direct_only:
            # API-compatible fast path: no latent transition or decoder is
            # called. The retained state tensors are diagnostics only and are
            # never consumed when latent_input_mode=wind_only.
            latent_pred = {key: latent_history[key][:, -1] for key in LATENT_KEYS}
            physics_decode = direct
            decode = direct
            guidance_gate = torch.ones_like(direct)
            guidance_residual = torch.zeros_like(direct)
            guidance_logits = torch.zeros_like(direct)
        else:
            latent_pred = _apply_full_path_latent_mask(
                model.latent_transition(latent_history, direct.detach()),
                masked_keys,
            )
            decoded = model.decoder(
                latent_pred["z_peak"], latent_pred["z_width"], latent_pred["z_noise"], latent_pred["peak_pos"]
            )
            physics_decode = decoded["v_estimate"]
            if config.guidance_enabled:
                if getattr(model, "guided_decoder", None) is None:
                    raise RuntimeError("guidance_enabled requires model.guided_decoder")
                progress_value = min(float(step) / float(config.guidance_max_horizon - 1), 1.0)
                progress = torch.full_like(direct, progress_value)
                guided = model.guided_decoder(
                    latent_pred,
                    direct,
                    physics_decode,
                    wind_history_direct[:, -1],
                    wind_history_decode[:, -1],
                    progress,
                )
                decode = guided["v_guided"]
                guidance_gate = guided["gate"]
                guidance_residual = guided["residual"]
                guidance_logits = guided["gate_logits"]
            else:
                alpha = float(config.physics_decode_weight)
                if alpha == 1.0:
                    decode = physics_decode  # exact legacy path
                elif alpha == 0.0:
                    decode = direct
                else:
                    decode = direct + alpha * (physics_decode - direct)
                guidance_gate = torch.zeros_like(direct)
                guidance_residual = torch.zeros_like(direct)
                guidance_logits = torch.zeros_like(direct)
        final = decode
        propagated_latent = latent_pred
        refined_shift = torch.zeros(batch_size, device=direct.device, dtype=direct.dtype)
        peak_pos_shift = torch.zeros(batch_size, device=direct.device, dtype=direct.dtype)
        # Direct-only confidence proxy: a large jump relative to the direct
        # branch's own latest history lowers confidence. No decoder output is
        # used, so gating cannot feed decode errors back into the direct chain.
        direct_confidence = torch.exp(-(direct - wind_history_direct[:, -1]).abs().mean(dim=1))
        if config.refinement.enabled and not config.direct_only:
            refined = refine_latent_state(
                model.decoder,
                latent_pred,
                direct,
                config.refinement,
                direct_confidence=direct_confidence,
            )
            final = refined["v_decode_refined"]
            refined_shift = refined["latent_shift"]
            peak_pos_shift = refined["peak_pos_shift"]
            if config.propagate_refined_latent:
                # Straight-through state propagation: forward values are the
                # exact refined state; backward follows the raw transition so
                # later rollout losses retain multi-step credit assignment.
                propagated_latent = {
                    key: latent_pred[key] + (refined["z_refined"][key] - latent_pred[key]).detach()
                    for key in LATENT_KEYS
                }

        zero = torch.zeros((), device=direct.device)
        direct_loss = zero
        raw_decode_loss = zero
        decode_loss = zero
        gate_loss = zero
        guide_anchor = zero
        advantage_target = torch.zeros_like(direct)
        latent_losses = {key: zero for key in ("total", "peak", "width", "noise", "pos")}
        if future_wind is not None:
            target = future_wind[:, step]
            if config.physics_only:
                direct_loss = zero
            elif config.direct_loss_type == "mae":
                direct_loss = _masked_mean((direct - target).abs(), valid)
            elif config.direct_loss_type == "smooth_l1":
                direct_loss = _masked_mean(F.smooth_l1_loss(direct, target, reduction="none"), valid)
            else:
                direct_loss = _masked_mean((direct - target).pow(2), valid)
            if config.decode_loss_type == "mae":
                decode_loss = _masked_mean((decode - target).abs(), valid)
            elif config.decode_loss_type == "smooth_l1":
                decode_loss = _masked_mean(F.smooth_l1_loss(decode, target, reduction="none"), valid)
            else:
                decode_loss = _masked_mean((decode - target).pow(2), valid)
            if not config.direct_only:
                raw_decode_loss = _masked_mean((physics_decode - target).pow(2), valid)
            if config.guidance_enabled:
                advantage_target = ((direct.detach() - target).abs() < (physics_decode.detach() - target).abs()).to(direct.dtype)
                if config.guidance_gate_loss_type == "balanced_bce":
                    gate_loss = _balanced_gate_bce(guidance_logits, advantage_target, valid)
                else:
                    gate_loss = _masked_mean(
                        F.binary_cross_entropy_with_logits(guidance_logits, advantage_target, reduction="none"),
                        valid,
                    )
                guide_anchor = _masked_mean(guidance_residual.pow(2), valid)
        if teacher_latent is not None and not config.direct_only:
            teacher_step = _apply_full_path_latent_mask(
                {key: teacher_latent[key][:, step] for key in LATENT_KEYS},
                masked_keys,
            )
            latent_losses = _latent_l1(latent_pred, teacher_step, valid)
        consistency = zero if config.direct_only else _masked_mean((decode - direct.detach()).pow(2), valid)
        delta = zero if config.direct_only else _delta_penalty(latent_pred, latent_history, config.delta_thresholds, valid)
        total = (
            float(config.lambda_direct) * direct_loss
            + float(config.lambda_decode) * decode_loss
            + float(config.lambda_raw_decode) * raw_decode_loss
            + float(config.lambda_latent_rollout) * latent_losses["total"]
            + float(config.lambda_consistency) * consistency
            + float(config.lambda_delta) * delta
            + float(config.lambda_guidance_gate) * gate_loss
            + float(config.lambda_guidance_anchor) * guide_anchor
        )
        loss_rows["direct"].append(direct_loss)
        loss_rows["raw_decode"].append(raw_decode_loss)
        loss_rows["decode"].append(decode_loss)
        loss_rows["consistency"].append(consistency)
        loss_rows["latent"].append(latent_losses["total"])
        for key in ("peak", "width", "noise", "pos"):
            loss_rows[key].append(latent_losses[key])
        loss_rows["delta"].append(delta)
        loss_rows["guidance_gate"].append(gate_loss)
        loss_rows["guidance_anchor"].append(guide_anchor)
        loss_rows["total"].append(total)

        if teacher_latent is not None and future_wind is not None:
            gt_mask = _teacher_mask(config.feedback_strategy, config.teacher_forcing_probability, direct, generator)
            gt_mask &= valid.unsqueeze(1)
            teacher_step = _apply_full_path_latent_mask(
                {key: teacher_latent[key][:, step].detach() for key in LATENT_KEYS},
                masked_keys,
            )
            next_latent = (
                {key: latent_history[key][:, -1] for key in LATENT_KEYS}
                if config.direct_only
                else _mix_latent(propagated_latent, teacher_step, gt_mask)
            )
            next_direct_candidate = decode if config.physics_only else direct
            next_direct_wind = torch.where(gt_mask, future_wind[:, step], next_direct_candidate)
            next_decode_wind = torch.where(gt_mask, future_wind[:, step], decode)
            next_refined_wind = torch.where(gt_mask, future_wind[:, step], final)
        else:
            gt_mask = torch.zeros_like(direct, dtype=torch.bool)
            next_latent = propagated_latent
            next_direct_wind = decode if config.physics_only else direct
            next_decode_wind = decode
            next_refined_wind = final

        if config.dual_wind_history:
            wind_history_direct = _update_wind_history(wind_history_direct, next_direct_wind)
            decode_feedback = next_refined_wind if config.refinement.enabled else next_decode_wind
            wind_history_decode = _update_wind_history(wind_history_decode, decode_feedback)
        else:
            if config.shared_feedback_branch == "direct":
                shared = next_direct_wind
            elif config.shared_feedback_branch == "decode":
                shared = next_decode_wind
            else:
                # Mode A: refinement is output-only and must not enter a shared
                # state that affects the next direct/latent transition.
                shared = next_refined_wind if config.propagate_refined_latent else next_decode_wind
            wind_history_direct = _update_wind_history(wind_history_direct, shared)
            wind_history_decode = wind_history_direct
        latent_history = _update_history_latent(
            latent_history,
            _apply_full_path_latent_mask(next_latent, masked_keys),
        )

        predictions["direct"].append(direct)
        predictions["raw_decode"].append(physics_decode)
        predictions["decode"].append(decode)
        predictions["refined"].append(final)
        for key in LATENT_KEYS:
            predictions[key].append(latent_pred[key])
        predictions["feedback_direct"].append(wind_history_direct[:, -1])
        predictions["feedback_decode"].append(wind_history_decode[:, -1])
        predictions["teacher_mask"].append(gt_mask)
        latent_norm = torch.sqrt(sum(value.reshape(batch_size, -1).pow(2).sum(dim=1) for value in latent_pred.values()))
        predictions["latent_norm"].append(latent_norm)
        predictions["direct_decode_disagreement"].append((direct - decode).abs().mean(dim=1))
        predictions["refined_shift"].append(refined_shift)
        predictions["peak_pos_shift"].append(peak_pos_shift)
        predictions["direct_confidence"].append(direct_confidence)
        predictions["guidance_gate"].append(guidance_gate)
        predictions["guidance_residual"].append(guidance_residual)
        predictions["direct_advantage_target"].append(advantage_target)

    discount = torch.tensor(
        [float(config.rollout_loss_discount) ** step for step in range(horizon)],
        device=past_wind.device,
        dtype=past_wind.dtype,
    )
    valid_counts = valid_mask.sum(dim=0).to(discount.dtype)
    normalized_weights = discount * valid_counts
    normalized_weights = normalized_weights / normalized_weights.sum().clamp_min(1e-12)
    losses = {
        key: sum(weight * value for weight, value in zip(normalized_weights, values))
        for key, values in loss_rows.items()
    }
    output: Dict[str, Any] = {key: torch.stack(values, dim=1) for key, values in predictions.items()}
    output["losses"] = losses
    output["valid_mask"] = valid_mask
    output["final_history_latent"] = latent_history
    output["final_wind_history_direct"] = wind_history_direct
    output["final_wind_history_decode"] = wind_history_decode
    return output


def teacher_forcing_probability(start: float, end: float, epoch: int, total_epochs: int, schedule: str) -> float:
    if schedule == "constant":
        return float(start)
    progress = min(max((int(epoch) - 1) / max(int(total_epochs) - 1, 1), 0.0), 1.0)
    if schedule == "linear":
        return float(start) + (float(end) - float(start)) * progress
    if schedule == "cosine":
        mix = 0.5 * (1.0 - math.cos(math.pi * progress))
        return float(start) + (float(end) - float(start)) * mix
    raise ValueError(f"Unsupported scheduled-sampling schedule: {schedule}")


def curriculum_horizon(curriculum: Sequence[int], epoch_counts: Sequence[int], epoch: int) -> int:
    if len(curriculum) != len(epoch_counts) or not curriculum:
        raise ValueError("rollout_curriculum and curriculum_epochs must be non-empty and equal length")
    cursor = 0
    for horizon, count in zip(curriculum, epoch_counts):
        cursor += int(count)
        if int(epoch) <= cursor:
            return int(horizon)
    return int(curriculum[-1])


def selection_value(metrics: Mapping[str, float], selection: Mapping[str, Any]) -> float:
    criterion = str(selection.get("criterion", "val_long_loop_refined_mae"))
    if criterion == "val_long_loop_refined_mae":
        return float(metrics["val_long_loop_refined_mae"])
    if criterion == "val_long_loop_decode_mae":
        return float(metrics["val_long_loop_decode_mae"])
    if criterion == "val_long_loop_direct_mae":
        return float(metrics["val_long_loop_direct_mae"])
    if criterion == "one_step_decode_mae":
        return float(metrics["val_one_step_decode_mae"])
    if criterion == "composite":
        one = float(metrics["val_one_step_refined_mae"])
        long = float(metrics["val_long_loop_refined_mae"])
        instability = float(metrics.get("val_instability", 0.0))
        return (
            float(selection.get("one_step_weight", 0.2)) * one
            + float(selection.get("long_loop_weight", 0.8)) * long
            + float(selection.get("instability_weight", 0.1)) * instability
        )
    raise ValueError(f"Unsupported selection criterion: {criterion}")


def compute_delta_thresholds(
    model: DisentangledTemporalWindForecaster,
    spectra: np.ndarray,
    device: torch.device,
    quantile: float = 0.99,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Estimate per-element GT latent delta thresholds from the train split only."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("delta quantile must be in (0,1)")
    rows: Dict[str, List[np.ndarray]] = {key: [] for key in LATENT_KEYS}
    model.eval()
    with torch.no_grad():
        encoded_parts: Dict[str, List[torch.Tensor]] = {key: [] for key in LATENT_KEYS}
        for start in range(0, len(spectra), batch_size):
            batch = torch.from_numpy(spectra[start : start + batch_size]).to(device)
            encoded = model.encoder(batch)
            for key in LATENT_KEYS:
                encoded_parts[key].append(encoded[key].cpu())
        for key in LATENT_KEYS:
            sequence = torch.cat(encoded_parts[key], dim=0)
            rows[key].append((sequence[1:] - sequence[:-1]).abs().reshape(-1).numpy())
    return {key: float(np.quantile(np.concatenate(value), quantile)) for key, value in rows.items()}
