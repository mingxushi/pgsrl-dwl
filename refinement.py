from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn.functional as F


LATENT_KEYS = ("z_peak", "z_width", "z_noise", "peak_pos")


@dataclass(frozen=True)
class RefinementConfig:
    enabled: bool = False
    final_output: str = "v_decode_refined"
    optimize_target: str = "peak_only"
    num_steps: int = 1
    learning_rate: float = 1e-3
    lambda_pseudo: float = 1.0
    lambda_anchor: float = 5.0
    lambda_temporal: float = 0.0
    lambda_bound: float = 0.0
    grad_clip_norm: Optional[float] = 1.0
    max_latent_shift: Optional[float] = 0.1
    safe_fallback: bool = True
    propagate_refined_latent: bool = False
    save_diagnostics: bool = True
    log_per_sample: bool = False
    online_loss_increase_tolerance: float = 1e-6
    max_peak_pos_shift_bins: Optional[float] = None
    n_frequency_bins: int = 100
    max_step_shift: Optional[float] = None
    early_stop_tolerance: float = 0.0
    disagreement_gate: Optional[float] = None
    direct_confidence_threshold: Optional[float] = None

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, object]]) -> "RefinementConfig":
        if not values:
            return cls()
        known = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in values.items() if key in known})

    def validate(self) -> None:
        if self.final_output != "v_decode_refined":
            raise ValueError("Refinement final_output must be v_decode_refined; branch fusion is not supported.")
        if self.optimize_target not in {"peak_only", "peak_pos_only", "all_latent"}:
            raise ValueError(f"Unsupported refinement optimize_target: {self.optimize_target}")
        if self.num_steps < 0:
            raise ValueError("refinement num_steps must be non-negative")
        if self.learning_rate < 0.0:
            raise ValueError("refinement learning_rate must be non-negative")
        for name in ("lambda_pseudo", "lambda_anchor", "lambda_temporal", "lambda_bound"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.grad_clip_norm is not None and float(self.grad_clip_norm) <= 0.0:
            raise ValueError("grad_clip_norm must be positive or null")
        if self.max_latent_shift is not None and float(self.max_latent_shift) < 0.0:
            raise ValueError("max_latent_shift must be non-negative or null")
        if self.max_peak_pos_shift_bins is not None and float(self.max_peak_pos_shift_bins) < 0.0:
            raise ValueError("max_peak_pos_shift_bins must be non-negative or null")
        if int(self.n_frequency_bins) < 2:
            raise ValueError("n_frequency_bins must be at least 2")
        if self.max_step_shift is not None and float(self.max_step_shift) < 0.0:
            raise ValueError("max_step_shift must be non-negative or null")
        if float(self.early_stop_tolerance) < 0.0:
            raise ValueError("early_stop_tolerance must be non-negative")
        if self.disagreement_gate is not None and float(self.disagreement_gate) < 0.0:
            raise ValueError("disagreement_gate must be non-negative or null")
        if self.direct_confidence_threshold is not None and not 0.0 <= float(self.direct_confidence_threshold) <= 1.0:
            raise ValueError("direct_confidence_threshold must be in [0,1] or null")


def _per_sample_mean(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim <= 1:
        return tensor
    return tensor.reshape(tensor.shape[0], -1).mean(dim=1)


def _per_sample_mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return _per_sample_mean((left - right).pow(2))


def _clone_latent(latent: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    missing = [key for key in LATENT_KEYS if key not in latent]
    if missing:
        raise KeyError(f"Missing structured latent keys: {missing}")
    return {key: latent[key].detach().clone() for key in LATENT_KEYS}


def _decode(decoder: torch.nn.Module, latent: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return decoder(latent["z_peak"], latent["z_width"], latent["z_noise"], latent["peak_pos"])


def _optimized_keys(config: RefinementConfig) -> tuple[str, ...]:
    if config.optimize_target == "peak_only":
        return ("z_peak",)
    if config.optimize_target == "peak_pos_only":
        return ("peak_pos",)
    return LATENT_KEYS


def _anchor_vector(current: Mapping[str, torch.Tensor], initial: Mapping[str, torch.Tensor], keys: tuple[str, ...]) -> torch.Tensor:
    pieces = [_per_sample_mse(current[key], initial[key]) for key in keys]
    return torch.stack(pieces, dim=0).mean(dim=0)


def _bound_vector(current: Mapping[str, torch.Tensor]) -> torch.Tensor:
    peak_pos = current["peak_pos"]
    violation = F.relu(-peak_pos).pow(2) + F.relu(peak_pos - 1.0).pow(2)
    return _per_sample_mean(violation)


def _shift_vector(current: Mapping[str, torch.Tensor], initial: Mapping[str, torch.Tensor], keys: tuple[str, ...]) -> torch.Tensor:
    squared = []
    for key in keys:
        delta = (current[key] - initial[key]).reshape(current[key].shape[0], -1)
        squared.append(delta.pow(2).sum(dim=1))
    return torch.stack(squared, dim=0).sum(dim=0).sqrt()


def _tensor_norm(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1).pow(2).sum(dim=1).sqrt()


def _clip_gradients_per_sample(gradients: List[torch.Tensor], max_norm: Optional[float]) -> List[torch.Tensor]:
    if max_norm is None:
        return gradients
    squared = []
    for gradient in gradients:
        squared.append(gradient.reshape(gradient.shape[0], -1).pow(2).sum(dim=1))
    norms = torch.stack(squared, dim=0).sum(dim=0).sqrt()
    scale = torch.clamp(float(max_norm) / norms.clamp_min(1e-12), max=1.0)
    return [gradient * scale.reshape((-1,) + (1,) * (gradient.ndim - 1)) for gradient in gradients]


def _loss_vectors(
    decoder: torch.nn.Module,
    current: Mapping[str, torch.Tensor],
    initial: Mapping[str, torch.Tensor],
    direct_reference: torch.Tensor,
    config: RefinementConfig,
    temporal_reference: Optional[Mapping[str, torch.Tensor]],
    keys: tuple[str, ...],
) -> Dict[str, torch.Tensor]:
    prediction = _decode(decoder, current)["v_estimate"]
    pseudo = _per_sample_mse(prediction, direct_reference)
    anchor = _anchor_vector(current, initial, keys)
    if temporal_reference is None or config.lambda_temporal == 0.0:
        temporal = torch.zeros_like(anchor)
    else:
        temporal = _anchor_vector(current, temporal_reference, keys)
    bound = _bound_vector(current)
    total = (
        float(config.lambda_pseudo) * pseudo
        + float(config.lambda_anchor) * anchor
        + float(config.lambda_temporal) * temporal
        + float(config.lambda_bound) * bound
    )
    return {"prediction": prediction, "pseudo": pseudo, "anchor": anchor, "temporal": temporal, "bound": bound, "total": total}


def refine_latent_state(
    decoder: torch.nn.Module,
    latent_state: Mapping[str, torch.Tensor],
    v_direct: torch.Tensor,
    config: RefinementConfig,
    temporal_reference: Optional[Mapping[str, torch.Tensor]] = None,
    direct_confidence: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """Refine only detached per-sample latent tensors; no labels or model optimizer are accepted."""
    config.validate()
    initial = _clone_latent(latent_state)
    current = _clone_latent(latent_state)
    direct_reference = v_direct.detach().clone()
    temporal_detached = _clone_latent(temporal_reference) if temporal_reference is not None else None
    keys = _optimized_keys(config)
    batch_size = int(direct_reference.shape[0])
    device = direct_reference.device
    failure_reasons: List[set[str]] = [set() for _ in range(batch_size)]

    with torch.no_grad():
        decoded_before = _decode(decoder, initial)
        prediction_before = decoded_before["v_estimate"].detach().clone()
        pseudo_before = _per_sample_mse(prediction_before, direct_reference)
        initial_total = _loss_vectors(decoder, initial, initial, direct_reference, config, temporal_detached, keys)["total"]
        disagreement = _per_sample_mean((prediction_before - direct_reference).abs())
        if config.disagreement_gate is not None:
            gated = disagreement > float(config.disagreement_gate)
            for sample_idx in gated.nonzero(as_tuple=False).flatten().tolist():
                failure_reasons[sample_idx].add("disagreement_gate")
        if config.direct_confidence_threshold is not None:
            if direct_confidence is None:
                raise ValueError("direct_confidence is required when direct_confidence_threshold is configured")
            confidence = direct_confidence.detach().reshape(batch_size)
            gated = confidence < float(config.direct_confidence_threshold)
            for sample_idx in gated.nonzero(as_tuple=False).flatten().tolist():
                failure_reasons[sample_idx].add("low_direct_confidence")

    if config.enabled and config.num_steps > 0 and config.learning_rate > 0.0:
        with torch.enable_grad():
            previous_total = initial_total
            for _ in range(int(config.num_steps)):
                variables: Dict[str, torch.Tensor] = {}
                graph_current: Dict[str, torch.Tensor] = {}
                for key in LATENT_KEYS:
                    value = current[key].detach()
                    if key in keys:
                        value = value.requires_grad_(True)
                        variables[key] = value
                    graph_current[key] = value
                losses = _loss_vectors(decoder, graph_current, initial, direct_reference, config, temporal_detached, keys)
                gradients = list(torch.autograd.grad(losses["total"].sum(), tuple(variables.values()), create_graph=False, retain_graph=False))
                gradients = _clip_gradients_per_sample(gradients, config.grad_clip_norm)
                candidate = {key: graph_current[key].detach().clone() for key in LATENT_KEYS}
                for key, gradient in zip(variables.keys(), gradients):
                    finite_by_sample = torch.isfinite(gradient.reshape(batch_size, -1)).all(dim=1)
                    for sample_idx in (~finite_by_sample).nonzero(as_tuple=False).flatten().tolist():
                        failure_reasons[sample_idx].add("nonfinite_gradient")
                    safe_gradient = torch.where(
                        finite_by_sample.reshape((-1,) + (1,) * (gradient.ndim - 1)),
                        gradient,
                        torch.zeros_like(gradient),
                    )
                    candidate[key] = graph_current[key].detach() - float(config.learning_rate) * safe_gradient.detach()
                if config.max_step_shift is not None:
                    step_shift = _shift_vector(candidate, current, keys)
                    step_scale = torch.clamp(float(config.max_step_shift) / step_shift.clamp_min(1e-12), max=1.0)
                    for key in keys:
                        reshape = step_scale.reshape((-1,) + (1,) * (candidate[key].ndim - 1))
                        candidate[key] = current[key] + (candidate[key] - current[key]) * reshape
                if "peak_pos" in keys:
                    candidate["peak_pos"] = candidate["peak_pos"].clamp(0.0, 1.0)
                    if config.max_peak_pos_shift_bins is not None:
                        normalized_bound = float(config.max_peak_pos_shift_bins) / float(config.n_frequency_bins - 1)
                        candidate["peak_pos"] = torch.maximum(
                            torch.minimum(candidate["peak_pos"], initial["peak_pos"] + normalized_bound),
                            initial["peak_pos"] - normalized_bound,
                        ).clamp(0.0, 1.0)
                proposed_shift = _shift_vector(candidate, initial, keys)
                if config.max_latent_shift is not None:
                    excessive = proposed_shift > float(config.max_latent_shift) + 1e-12
                    for sample_idx in excessive.nonzero(as_tuple=False).flatten().tolist():
                        failure_reasons[sample_idx].add("excessive_latent_shift")
                    scale = torch.clamp(float(config.max_latent_shift) / proposed_shift.clamp_min(1e-12), max=1.0)
                    for key in keys:
                        reshape = scale.reshape((-1,) + (1,) * (candidate[key].ndim - 1))
                        candidate[key] = initial[key] + (candidate[key] - initial[key]) * reshape
                current = {key: value.detach() for key, value in candidate.items()}
                if config.early_stop_tolerance > 0.0:
                    with torch.no_grad():
                        candidate_total = _loss_vectors(decoder, current, initial, direct_reference, config, temporal_detached, keys)["total"]
                    if bool(((previous_total - candidate_total).abs() <= float(config.early_stop_tolerance)).all()):
                        break
                    previous_total = candidate_total

    with torch.no_grad():
        final_losses = _loss_vectors(decoder, current, initial, direct_reference, config, temporal_detached, keys)
        z_finite = torch.ones(batch_size, dtype=torch.bool, device=device)
        for key in LATENT_KEYS:
            z_finite &= torch.isfinite(current[key].reshape(batch_size, -1)).all(dim=1)
        prediction_finite = torch.isfinite(final_losses["prediction"].reshape(batch_size, -1)).all(dim=1)
        physical_ok = ((current["peak_pos"] >= 0.0) & (current["peak_pos"] <= 1.0)).reshape(batch_size, -1).all(dim=1)
        shift = _shift_vector(current, initial, keys)
        online_increase = final_losses["total"] > initial_total + float(config.online_loss_increase_tolerance)
        for sample_idx in (~z_finite).nonzero(as_tuple=False).flatten().tolist():
            failure_reasons[sample_idx].add("nonfinite_latent")
        for sample_idx in (~prediction_finite).nonzero(as_tuple=False).flatten().tolist():
            failure_reasons[sample_idx].add("nonfinite_prediction")
        for sample_idx in (~physical_ok).nonzero(as_tuple=False).flatten().tolist():
            failure_reasons[sample_idx].add("physical_bound_violation")
        for sample_idx in online_increase.nonzero(as_tuple=False).flatten().tolist():
            failure_reasons[sample_idx].add("online_loss_increase")

        accepted = torch.tensor(
            [not reasons or not config.safe_fallback for reasons in failure_reasons],
            dtype=torch.bool,
            device=device,
        )
        mask_for_latent = accepted
        refined: Dict[str, torch.Tensor] = {}
        for key in LATENT_KEYS:
            mask = mask_for_latent.reshape((-1,) + (1,) * (current[key].ndim - 1))
            refined[key] = torch.where(mask, current[key], initial[key]).detach()
        decoded_refined = _decode(decoder, refined)
        prediction_refined = decoded_refined["v_estimate"].detach()
        final_shift = _shift_vector(refined, initial, keys)
        pseudo_after = _per_sample_mse(prediction_refined, direct_reference)
        anchor_after = _anchor_vector(refined, initial, keys)
        component_shifts = {f"{key}_shift": _tensor_norm(refined[key] - initial[key]) for key in LATENT_KEYS}
        latent_norm_before = _shift_vector(initial, {key: torch.zeros_like(value) for key, value in initial.items()}, LATENT_KEYS)
        latent_norm_after = _shift_vector(refined, {key: torch.zeros_like(value) for key, value in refined.items()}, LATENT_KEYS)
        correction_before = decoded_before.get("correction")
        correction_after = decoded_refined.get("correction")
        if correction_before is None or correction_after is None:
            correction_change = torch.zeros(batch_size, device=device, dtype=prediction_refined.dtype)
        else:
            correction_change = _per_sample_mean((correction_after - correction_before).abs())

    return {
        "v_direct": direct_reference,
        "z_before": initial,
        "v_decode_before": prediction_before,
        "z_refined": refined,
        "v_decode_refined": prediction_refined,
        "refinement_accepted": accepted,
        "fallback_reason": ["accepted" if not reasons else ";".join(sorted(reasons)) for reasons in failure_reasons],
        "latent_shift": final_shift,
        "loss_pseudo_before": pseudo_before,
        "loss_pseudo_after": pseudo_after,
        "loss_anchor_after": anchor_after,
        "latent_norm_before": latent_norm_before,
        "latent_norm_after": latent_norm_after,
        "correction_before": correction_before,
        "correction_after": correction_after,
        "correction_change": correction_change,
        **component_shifts,
    }
