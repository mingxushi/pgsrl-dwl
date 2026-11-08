"""
Physics-Guided Disentangled Representation Learning for Doppler Wind Estimation

在保留原有“单时刻物理引导解耦风速估计器”的基础上，
增量实现“单步自回归的风速-隐状态耦合预测模型”。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ABLATION_CODE_VERSION = "reviewer-controlled-v1"


def set_experiment_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PeakAttentionModule(nn.Module):
    """峰值相关表示提取模块。"""

    def __init__(
        self,
        n_freq: int = 100,
        d_model: int = 64,
        n_heads: int = 4,
        n_gate: int = 20,
    ):
        super().__init__()

        self.n_freq = n_freq
        self.d_model = d_model
        self.n_gate = n_gate

        freq_pos = torch.arange(n_freq).float() / max(n_freq - 1, 1)
        self.register_buffer("freq_pos", freq_pos)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(n_freq, d_model)
        pe[:, 0::2] = torch.sin(freq_pos.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(freq_pos.unsqueeze(1) * div_term)
        self.register_buffer("pos_encoding", pe)

        self.multi_scale_conv = nn.ModuleList(
            [nn.Conv1d(1, 16, kernel_size=k, padding=k // 2) for k in [3, 5, 7, 11]]
        )
        self.scale_fusion = nn.Linear(16 * 4, d_model)

        self.intensity_encoder = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

        self.gate_embed = nn.Parameter(torch.randn(n_gate, d_model) * 0.02)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.peak_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.softmax_temperature = nn.Parameter(torch.ones(1) * 10.0)

        self.pos_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.intensity_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.representation = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def soft_argmax(self, spectrum: torch.Tensor) -> torch.Tensor:
        """可微分峰值位置估计。"""
        original_shape = spectrum.shape
        if len(original_shape) == 3:
            batch_size, n_gate, n_freq = spectrum.shape
            spectrum = spectrum.reshape(batch_size * n_gate, n_freq)

        weights = F.softmax(spectrum * self.softmax_temperature, dim=-1)
        peak_pos = (weights * self.freq_pos.unsqueeze(0)).sum(dim=-1)

        if len(original_shape) == 3:
            peak_pos = peak_pos.reshape(batch_size, n_gate)

        return peak_pos

    def forward(self, spectrum: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, n_gate, n_freq = spectrum.shape

        peak_pos_soft_argmax = self.soft_argmax(spectrum)

        multi_scale_features = []
        spectrum_flat = spectrum.reshape(batch_size * n_gate, n_freq).unsqueeze(1)
        for conv in self.multi_scale_conv:
            multi_scale_features.append(conv(spectrum_flat))

        multi_scale = torch.cat(multi_scale_features, dim=1)
        multi_scale = multi_scale.permute(0, 2, 1)
        multi_scale_feat = self.scale_fusion(multi_scale)
        multi_scale_feat = multi_scale_feat.reshape(
            batch_size, n_gate, n_freq, self.d_model
        )

        intensity_embed = self.intensity_encoder(spectrum.unsqueeze(-1))
        pos_embed = self.pos_encoding.unsqueeze(0).unsqueeze(0)
        spectrum_embed = multi_scale_feat + intensity_embed + pos_embed

        peak_pos_list = []
        peak_intensity_list = []
        z_peak_list = []
        attn_weights_list = []

        for gate_idx in range(n_gate):
            spectrum_gate = spectrum_embed[:, gate_idx, :, :]
            spectrum_gate = self.self_attn(spectrum_gate, spectrum_gate, spectrum_gate)[0]

            query = self.peak_query.expand(batch_size, -1, -1)
            attended, attn_weights = self.self_attn(query, spectrum_gate, spectrum_gate)

            attended = attended.squeeze(1)
            attn_weights = attn_weights.squeeze(1)

            peak_pos_attn = self.pos_head(attended).squeeze(-1)
            peak_intensity = self.intensity_head(attended).squeeze(-1)
            z_peak = self.representation(attended)

            peak_pos_list.append(peak_pos_attn)
            peak_intensity_list.append(peak_intensity)
            z_peak_list.append(z_peak)
            attn_weights_list.append(attn_weights)

        peak_pos_attn = torch.stack(peak_pos_list, dim=1)
        peak_intensity = torch.stack(peak_intensity_list, dim=1)
        z_peak = torch.stack(z_peak_list, dim=1)
        attn_weights = torch.stack(attn_weights_list, dim=1)
        peak_pos = 0.5 * peak_pos_attn + 0.5 * peak_pos_soft_argmax

        return {
            "peak_pos": peak_pos,
            "peak_pos_attn": peak_pos_attn,
            "peak_pos_soft_argmax": peak_pos_soft_argmax,
            "peak_intensity": peak_intensity,
            "z_peak": z_peak,
            "attn_weights": attn_weights,
        }


class WidthEstimationModule(nn.Module):
    """谱宽相关表示提取模块。"""

    def __init__(
        self,
        n_freq: int = 100,
        d_model: int = 64,
        n_gate: int = 20,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        self.width_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

        self.representation = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, spectrum: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, n_gate, _ = spectrum.shape
        width_list = []
        z_width_list = []

        for gate_idx in range(n_gate):
            x = spectrum[:, gate_idx, :].unsqueeze(1)
            x = self.encoder(x)
            width_list.append(self.width_head(x).squeeze(-1))
            z_width_list.append(self.representation(x))

        return {
            "width": torch.stack(width_list, dim=1),
            "z_width": torch.stack(z_width_list, dim=1),
        }


class NoiseFloorModule(nn.Module):
    """噪声底相关表示提取模块。"""

    def __init__(
        self,
        n_freq: int = 100,
        d_model: int = 64,
        n_gate: int = 20,
    ):
        super().__init__()

        self.edge_encoder = nn.Sequential(
            nn.Linear(20, 32),
            nn.GELU(),
            nn.Linear(32, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.global_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.noise_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        self.snr_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, spectrum: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, n_gate, _ = spectrum.shape
        noise_level_list = []
        snr_list = []
        z_noise_list = []

        for gate_idx in range(n_gate):
            spec_gate = spectrum[:, gate_idx, :]
            edge_features = torch.cat([spec_gate[:, :10], spec_gate[:, -10:]], dim=-1)
            edge_embed = self.edge_encoder(edge_features)
            global_embed = self.global_encoder(spec_gate.unsqueeze(1))
            z_noise = self.fusion(torch.cat([edge_embed, global_embed], dim=-1))

            noise_level_list.append(self.noise_head(z_noise).squeeze(-1))
            snr_list.append(self.snr_head(z_noise).squeeze(-1))
            z_noise_list.append(z_noise)

        return {
            "noise_level": torch.stack(noise_level_list, dim=1),
            "snr": torch.stack(snr_list, dim=1),
            "z_noise": torch.stack(z_noise_list, dim=1),
        }


class DisentangledSpectrumEncoder(nn.Module):
    """单时刻解耦编码器。"""

    def __init__(
        self,
        n_freq: int = 100,
        d_model: int = 64,
        n_gate: int = 20,
        n_heads: int = 4,
        latent_layout: str = "structured",
    ):
        super().__init__()
        if latent_layout not in {"structured", "shared", "entangled_matched", "beta_vae", "mean_pool", "concat_only"}:
            raise ValueError(f"Unsupported latent_layout: {latent_layout}")
        self.latent_layout = latent_layout
        self.n_freq = n_freq
        self.n_gate = n_gate
        self.d_model = d_model
        if self.latent_layout == "structured":
            self.peak_module = PeakAttentionModule(n_freq, d_model, n_heads, n_gate)
            self.width_module = WidthEstimationModule(n_freq, d_model, n_gate)
            self.noise_module = NoiseFloorModule(n_freq, d_model, n_gate)
        elif self.latent_layout == "shared":
            self.shared_encoder = nn.Sequential(
                nn.Linear(n_freq, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.shared_peak_head = nn.Linear(d_model, 1)
            self.shared_intensity_head = nn.Linear(d_model, 1)
            self.shared_width_head = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
            self.shared_noise_head = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
            self.shared_snr_head = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
        elif self.latent_layout == "entangled_matched":
            entangled_dim = d_model * 3
            self.entangled_encoder = nn.Sequential(
                nn.Linear(n_freq, entangled_dim),
                nn.LayerNorm(entangled_dim),
                nn.GELU(),
                nn.Linear(entangled_dim, entangled_dim),
                nn.LayerNorm(entangled_dim),
                nn.GELU(),
            )
            # A single adapter keeps the full-model parameter difference below 5%.
            self.entangled_decoder_projection = nn.Linear(entangled_dim, entangled_dim)
            self.entangled_capacity_matcher = nn.Sequential(
                nn.Linear(entangled_dim, 128),
                nn.GELU(),
                nn.Linear(128, entangled_dim),
            )
            self.entangled_peak_head = nn.Linear(entangled_dim, 1)
            self.entangled_intensity_head = nn.Linear(entangled_dim, 1)
            self.entangled_width_head = nn.Sequential(nn.Linear(entangled_dim, 1), nn.Softplus())
            self.entangled_noise_head = nn.Sequential(nn.Linear(entangled_dim, 1), nn.Softplus())
            self.entangled_snr_head = nn.Sequential(nn.Linear(entangled_dim, 1), nn.Softplus())
        elif self.latent_layout == "beta_vae":
            # Capacity-matched variational alternative.  The stochastic code is
            # a single shared 3*d_model vector; the chunks below exist only to
            # preserve the unchanged temporal/decoder interface.
            latent_dim = d_model * 3
            self.beta_vae_backbone = nn.Sequential(
                nn.Linear(n_freq, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.beta_vae_mu = nn.Linear(d_model, latent_dim)
            self.beta_vae_logvar = nn.Linear(d_model, latent_dim)
            self.beta_vae_capacity_adapter = nn.Linear(latent_dim, latent_dim)
            self.beta_vae_capacity_matcher = nn.Sequential(
                nn.Linear(latent_dim, 218),
                nn.GELU(),
                nn.Linear(218, latent_dim),
            )
            self.beta_vae_peak_head = nn.Linear(latent_dim, 1)
            self.beta_vae_intensity_head = nn.Linear(latent_dim, 1)
            self.beta_vae_width_head = nn.Sequential(nn.Linear(latent_dim, 1), nn.Softplus())
            self.beta_vae_noise_head = nn.Sequential(nn.Linear(latent_dim, 1), nn.Softplus())
            self.beta_vae_snr_head = nn.Sequential(nn.Linear(latent_dim, 1), nn.Softplus())
        elif self.latent_layout == "mean_pool":
            self.summary_encoder = nn.Sequential(
                nn.Linear(4, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.summary_peak_head = nn.Linear(d_model, 1)
            self.summary_intensity_head = nn.Linear(d_model, 1)
            self.summary_width_head = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
            self.summary_noise_head = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
            self.summary_snr_head = nn.Linear(d_model, 1)
        else:
            self.raw_peak_encoder = nn.Sequential(
                nn.Linear(n_freq, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.raw_width_encoder = nn.Sequential(
                nn.Linear(n_freq, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.raw_noise_encoder = nn.Sequential(
                nn.Linear(n_freq, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.concat_peak_head = nn.Linear(d_model, 1)
            self.concat_intensity_head = nn.Linear(d_model, 1)
            self.concat_width_head = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
            self.concat_noise_head = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
            self.concat_snr_head = nn.Linear(d_model, 1)

    def _uniform_attention(self, spectrum: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (*spectrum.shape[:2], self.n_freq),
            1.0 / float(self.n_freq),
            dtype=spectrum.dtype,
            device=spectrum.device,
        )

    def _summary_features(self, spectrum: torch.Tensor) -> torch.Tensor:
        peak_pos = spectrum.argmax(dim=-1, keepdim=True).float() / max(self.n_freq - 1, 1)
        return torch.cat(
            [
                spectrum.mean(dim=-1, keepdim=True),
                spectrum.std(dim=-1, keepdim=True, unbiased=False),
                spectrum.amax(dim=-1, keepdim=True),
                peak_pos,
            ],
            dim=-1,
        )

    def forward(self, spectrum: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.latent_layout == "shared":
            z_shared = self.shared_encoder(spectrum)
            peak_pos = torch.sigmoid(self.shared_peak_head(z_shared).squeeze(-1))
            peak_intensity = self.shared_intensity_head(z_shared).squeeze(-1)
            width = self.shared_width_head(z_shared).squeeze(-1)
            noise_level = self.shared_noise_head(z_shared).squeeze(-1)
            snr = self.shared_snr_head(z_shared).squeeze(-1)
            attn_weights = self._uniform_attention(spectrum)
            return {
                "peak_pos": peak_pos,
                "peak_pos_attn": peak_pos,
                "peak_pos_soft_argmax": peak_pos,
                "peak_intensity": peak_intensity,
                "z_peak": z_shared,
                "attn_weights": attn_weights,
                "width": width,
                "z_width": z_shared,
                "noise_level": noise_level,
                "snr": snr,
                "z_noise": z_shared,
            }
        if self.latent_layout == "entangled_matched":
            z_entangled = self.entangled_encoder(spectrum)
            z_entangled = z_entangled + self.entangled_capacity_matcher(z_entangled)
            decoder_latent = self.entangled_decoder_projection(z_entangled)
            z_peak, z_width, z_noise = torch.chunk(decoder_latent, 3, dim=-1)
            peak_pos = torch.sigmoid(self.entangled_peak_head(z_entangled).squeeze(-1))
            peak_intensity = self.entangled_intensity_head(z_entangled).squeeze(-1)
            width = self.entangled_width_head(z_entangled).squeeze(-1)
            noise_level = self.entangled_noise_head(z_entangled).squeeze(-1)
            snr = self.entangled_snr_head(z_entangled).squeeze(-1)
            attn_weights = self._uniform_attention(spectrum)
            return {
                "peak_pos": peak_pos,
                "peak_pos_attn": peak_pos,
                "peak_pos_soft_argmax": peak_pos,
                "peak_intensity": peak_intensity,
                "z_peak": z_peak,
                "attn_weights": attn_weights,
                "width": width,
                "z_width": z_width,
                "noise_level": noise_level,
                "snr": snr,
                "z_noise": z_noise,
                "z_entangled": z_entangled,
            }
        if self.latent_layout == "beta_vae":
            hidden = self.beta_vae_backbone(spectrum)
            mu = self.beta_vae_mu(hidden)
            logvar = self.beta_vae_logvar(hidden).clamp(min=-12.0, max=12.0)
            if self.training:
                z_shared = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            else:
                z_shared = mu
            z_shared = z_shared + self.beta_vae_capacity_matcher(z_shared)
            decoder_latent = self.beta_vae_capacity_adapter(z_shared)
            z_peak, z_width, z_noise = torch.chunk(decoder_latent, 3, dim=-1)
            peak_pos = torch.sigmoid(self.beta_vae_peak_head(z_shared).squeeze(-1))
            peak_intensity = self.beta_vae_intensity_head(z_shared).squeeze(-1)
            width = self.beta_vae_width_head(z_shared).squeeze(-1)
            noise_level = self.beta_vae_noise_head(z_shared).squeeze(-1)
            snr = self.beta_vae_snr_head(z_shared).squeeze(-1)
            attn_weights = self._uniform_attention(spectrum)
            return {
                "peak_pos": peak_pos,
                "peak_pos_attn": peak_pos,
                "peak_pos_soft_argmax": peak_pos,
                "peak_intensity": peak_intensity,
                "z_peak": z_peak,
                "attn_weights": attn_weights,
                "width": width,
                "z_width": z_width,
                "noise_level": noise_level,
                "snr": snr,
                "z_noise": z_noise,
                "z_shared": z_shared,
                "mu": mu,
                "logvar": logvar,
            }
        if self.latent_layout == "mean_pool":
            z_summary = self.summary_encoder(self._summary_features(spectrum))
            peak_pos = torch.sigmoid(self.summary_peak_head(z_summary).squeeze(-1))
            peak_intensity = self.summary_intensity_head(z_summary).squeeze(-1)
            width = self.summary_width_head(z_summary).squeeze(-1)
            noise_level = self.summary_noise_head(z_summary).squeeze(-1)
            snr = self.summary_snr_head(z_summary).squeeze(-1)
            attn_weights = self._uniform_attention(spectrum)
            return {
                "peak_pos": peak_pos,
                "peak_pos_attn": peak_pos,
                "peak_pos_soft_argmax": peak_pos,
                "peak_intensity": peak_intensity,
                "z_peak": z_summary,
                "attn_weights": attn_weights,
                "width": width,
                "z_width": z_summary,
                "noise_level": noise_level,
                "snr": snr,
                "z_noise": z_summary,
            }
        if self.latent_layout == "concat_only":
            z_peak = self.raw_peak_encoder(spectrum)
            z_width = self.raw_width_encoder(spectrum)
            z_noise = self.raw_noise_encoder(spectrum)
            peak_pos = torch.sigmoid(self.concat_peak_head(z_peak).squeeze(-1))
            peak_intensity = self.concat_intensity_head(z_peak).squeeze(-1)
            width = self.concat_width_head(z_width).squeeze(-1)
            noise_level = self.concat_noise_head(z_noise).squeeze(-1)
            snr = self.concat_snr_head(z_noise).squeeze(-1)
            attn_weights = self._uniform_attention(spectrum)
            return {
                "peak_pos": peak_pos,
                "peak_pos_attn": peak_pos,
                "peak_pos_soft_argmax": peak_pos,
                "peak_intensity": peak_intensity,
                "z_peak": z_peak,
                "attn_weights": attn_weights,
                "width": width,
                "z_width": z_width,
                "noise_level": noise_level,
                "snr": snr,
                "z_noise": z_noise,
            }

        peak_output = self.peak_module(spectrum)
        width_output = self.width_module(spectrum)
        noise_output = self.noise_module(spectrum)

        return {
            "peak_pos": peak_output["peak_pos"],
            "peak_pos_attn": peak_output["peak_pos_attn"],
            "peak_pos_soft_argmax": peak_output["peak_pos_soft_argmax"],
            "peak_intensity": peak_output["peak_intensity"],
            "z_peak": peak_output["z_peak"],
            "attn_weights": peak_output["attn_weights"],
            "width": width_output["width"],
            "z_width": width_output["z_width"],
            "noise_level": noise_output["noise_level"],
            "snr": noise_output["snr"],
            "z_noise": noise_output["z_noise"],
        }


class PhysicsGuidedDecoder(nn.Module):
    """保留原物理语义的解码器。"""

    def __init__(
        self,
        d_model: int = 64,
        n_gate: int = 20,
        wavelength: float = 1.55,
        f_res: float = 1000 / 1024,
        n_freq: int = 100,
        f_offset: float = 79.29,
        f_center_ideal: float = 82.0,
        decoder_mode: str = "physics",
    ):
        super().__init__()
        if decoder_mode not in {"physics", "pure_doppler", "mlp", "matched_mlp", "no_physics_mlp", "concat_only"}:
            raise ValueError(f"Unsupported decoder_mode: {decoder_mode}")
        self.decoder_mode = decoder_mode

        self.wavelength = wavelength
        self.f_res = f_res
        self.n_freq = n_freq
        self.n_gate = n_gate
        self.f_center = n_freq / 2.0 + (f_offset / f_res - f_center_ideal)

        self.register_buffer("freq_axis", torch.arange(n_freq) * f_res)

        self.correction_net = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.turbulence_decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        self.uncertainty_decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        self.wind_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.concat_decoder = nn.Linear(d_model * 3, 1)

    def _doppler_velocity(self, peak_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bin_pos = peak_pos * (self.n_freq - 1) + 1
        f_d = (bin_pos - self.f_center) * self.f_res
        return f_d, self.wavelength * f_d / 2.0

    def forward(
        self,
        z_peak: torch.Tensor,
        z_width: torch.Tensor,
        z_noise: torch.Tensor,
        peak_pos: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.decoder_mode in {"matched_mlp", "no_physics_mlp"}:
            f_d = torch.zeros_like(peak_pos)
            v_physics = torch.zeros_like(peak_pos)
        else:
            f_d, v_physics = self._doppler_velocity(peak_pos)
        turbulence = self.turbulence_decoder(z_width).squeeze(-1)
        uncertainty = self.uncertainty_decoder(z_noise).squeeze(-1)

        if self.decoder_mode == "pure_doppler":
            correction = torch.zeros_like(v_physics)
            v_estimate = v_physics
            v_fused = torch.zeros_like(v_physics)
        elif self.decoder_mode in {"matched_mlp", "no_physics_mlp"}:
            # Reuse the exact learned correction head from the physics decoder.
            # The controlled change is the removal of the fixed Doppler term.
            v_estimate = self.correction_net(z_peak).squeeze(-1)
            v_fused = v_estimate
            correction = torch.zeros_like(v_estimate)
        elif self.decoder_mode == "mlp":
            z_fused = torch.cat([z_peak, z_width, z_noise], dim=-1)
            v_fused = self.wind_fusion(z_fused).squeeze(-1)
            v_estimate = v_fused
            correction = v_estimate - v_physics
        elif self.decoder_mode == "concat_only":
            z_fused = torch.cat([z_peak, z_width, z_noise], dim=-1)
            v_fused = self.concat_decoder(z_fused).squeeze(-1)
            v_estimate = v_fused
            correction = v_estimate - v_physics
        else:
            raw_correction = self.correction_net(z_peak).squeeze(-1)
            z_fused = torch.cat([z_peak, z_width, z_noise], dim=-1)
            v_fused = self.wind_fusion(z_fused).squeeze(-1)
            correction = raw_correction
            v_estimate = v_physics + correction

        return {
            "v_estimate": v_estimate,
            "v_physics": v_physics,
            "v_fused": v_fused,
            "turbulence": turbulence,
            "uncertainty": uncertainty,
            "correction": correction,
            "f_d": f_d,
        }


class MIEstimator(nn.Module):
    """MINE 风格互信息估计器。"""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch_size = z1.size(0)
        joint = torch.cat([z1, z2], dim=-1)
        t_joint = self.net(joint)

        perm = torch.randperm(batch_size, device=z1.device)
        z2_shuffled = z2[perm]
        marginal = torch.cat([z1, z2_shuffled], dim=-1)
        t_marginal = self.net(marginal)

        mi = t_joint.mean() - torch.log(t_marginal.exp().mean() + 1e-8)
        return mi.abs()


class DisentanglementLoss(nn.Module):
    """解耦约束。"""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.mi_estimator_peak_width = MIEstimator(d_model)
        self.mi_estimator_peak_noise = MIEstimator(d_model)
        self.mi_estimator_width_noise = MIEstimator(d_model)

    def forward(
        self,
        z_peak: torch.Tensor,
        z_width: torch.Tensor,
        z_noise: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, n_gate, d_model = z_peak.shape
        z_peak_flat = z_peak.reshape(batch_size * n_gate, d_model)
        z_width_flat = z_width.reshape(batch_size * n_gate, d_model)
        z_noise_flat = z_noise.reshape(batch_size * n_gate, d_model)

        mi_peak_width = self.mi_estimator_peak_width(z_peak_flat, z_width_flat)
        mi_peak_noise = self.mi_estimator_peak_noise(z_peak_flat, z_noise_flat)
        mi_width_noise = self.mi_estimator_width_noise(z_width_flat, z_noise_flat)
        mi_loss = mi_peak_width + mi_peak_noise + mi_width_noise

        return {
            "mi_loss": mi_loss,
            "mi_peak_width": mi_peak_width,
            "mi_peak_noise": mi_peak_noise,
            "mi_width_noise": mi_width_noise,
        }


class DisentangledWindEstimator(nn.Module):
    """原单时刻物理引导解耦风速估计器。"""

    def __init__(
        self,
        n_freq: int = 100,
        n_gate: int = 20,
        d_model: int = 64,
        n_heads: int = 4,
        wavelength: float = 1.55,
        f_res: float = 1000 / 1024,
        decoder_mode: str = "physics",
        latent_layout: str = "structured",
        lambda_cons: float = 0.5,
        mi_weight: float = 0.1,
        beta_vae_beta: float = 4.0,
    ):
        super().__init__()

        self.n_freq = n_freq
        self.n_gate = n_gate
        self.d_model = d_model
        self.decoder_mode = decoder_mode
        self.latent_layout = latent_layout
        self.lambda_cons = lambda_cons
        self.mi_weight = float(mi_weight)
        self.beta_vae_beta = float(beta_vae_beta)

        self.encoder = DisentangledSpectrumEncoder(
            n_freq=n_freq,
            d_model=d_model,
            n_gate=n_gate,
            n_heads=n_heads,
            latent_layout=latent_layout,
        )
        self.decoder = PhysicsGuidedDecoder(
            d_model=d_model,
            n_gate=n_gate,
            wavelength=wavelength,
            f_res=f_res,
            n_freq=n_freq,
            decoder_mode=decoder_mode,
        )
        if self.mi_weight > 0.0:
            self.disentangle_loss = DisentanglementLoss(d_model)
        else:
            _ = DisentanglementLoss(d_model)
            self.disentangle_loss = None
        self.spectrum_reconstructor = nn.Sequential(
            nn.Linear(d_model * 3, 128),
            nn.GELU(),
            nn.Linear(128, n_freq),
        )

    def forward(
        self,
        spectrum: torch.Tensor,
        wind_speed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(spectrum)
        decoded = self.decoder(
            encoded["z_peak"],
            encoded["z_width"],
            encoded["z_noise"],
            encoded["peak_pos"],
        )
        zero = torch.zeros((), device=spectrum.device)
        if self.disentangle_loss is not None and self.mi_weight > 0.0:
            disentangle = self.disentangle_loss(
                encoded["z_peak"], encoded["z_width"], encoded["z_noise"]
            )
        else:
            disentangle = {
                "mi_loss": zero,
                "mi_peak_width": zero,
                "mi_peak_noise": zero,
                "mi_width_noise": zero,
            }

        z_fused = torch.cat(
            [encoded["z_peak"], encoded["z_width"], encoded["z_noise"]], dim=-1
        )
        spectrum_recon = self.spectrum_reconstructor(z_fused)

        losses: Dict[str, torch.Tensor] = {}
        if wind_speed is not None:
            losses["wind_loss"] = F.mse_loss(decoded["v_estimate"], wind_speed)
            losses["consistency_loss"] = F.mse_loss(
                decoded["v_estimate"], decoded["v_physics"]
            )
            residuals = (decoded["v_estimate"] - wind_speed).abs()
            losses["calibration_loss"] = F.mse_loss(
                decoded["uncertainty"], residuals.detach()
            )

        losses["peak_consistency_loss"] = F.mse_loss(
            encoded["peak_pos_attn"], encoded["peak_pos_soft_argmax"].detach()
        )
        losses["recon_loss"] = F.mse_loss(spectrum_recon, spectrum)
        losses["mi_loss"] = disentangle["mi_loss"]
        if self.latent_layout == "beta_vae":
            mu = encoded["mu"]
            logvar = encoded["logvar"]
            losses["kl_loss"] = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        else:
            losses["kl_loss"] = zero
        losses["beta_kl_loss"] = self.beta_vae_beta * losses["kl_loss"]
        losses["total_loss"] = (
            1.0 * losses.get("wind_loss", torch.zeros((), device=spectrum.device))
            + self.lambda_cons
            * losses.get("consistency_loss", torch.zeros((), device=spectrum.device))
            + 0.3
            * losses.get("calibration_loss", torch.zeros((), device=spectrum.device))
            + 0.5 * losses["peak_consistency_loss"]
            + 0.2 * losses["recon_loss"]
            + self.mi_weight * losses["mi_loss"]
            + losses["beta_kl_loss"]
        )

        return {
            "v_estimate": decoded["v_estimate"],
            "v_physics": decoded["v_physics"],
            "turbulence": decoded["turbulence"],
            "uncertainty": decoded["uncertainty"],
            "peak_pos": encoded["peak_pos"],
            "peak_pos_attn": encoded["peak_pos_attn"],
            "peak_pos_soft_argmax": encoded["peak_pos_soft_argmax"],
            "peak_intensity": encoded["peak_intensity"],
            "width": encoded["width"],
            "snr": encoded["snr"],
            "noise_level": encoded["noise_level"],
            "z_peak": encoded["z_peak"],
            "z_width": encoded["z_width"],
            "z_noise": encoded["z_noise"],
            "attn_weights": encoded["attn_weights"],
            "correction": decoded["correction"],
            "spectrum_recon": spectrum_recon,
            "losses": losses,
            "mi_metrics": {
                "mi_peak_width": disentangle["mi_peak_width"],
                "mi_peak_noise": disentangle["mi_peak_noise"],
                "mi_width_noise": disentangle["mi_width_noise"],
            },
            "vae_metrics": {
                "kl_loss": losses["kl_loss"],
                "beta_kl_loss": losses["beta_kl_loss"],
                "beta": torch.as_tensor(self.beta_vae_beta, device=spectrum.device),
            },
        }

    def estimate(self, spectrum: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            output = self.forward(spectrum)
        return {
            "v_estimate": output["v_estimate"],
            "v_physics": output["v_physics"],
            "turbulence": output["turbulence"],
            "uncertainty": output["uncertainty"],
            "peak_pos": output["peak_pos"],
        }


def _merge_latent_features(latent_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """保留解耦结构可见性的历史特征组织函数。"""
    return torch.cat(
        [
            latent_dict["z_peak"],
            latent_dict["z_width"],
            latent_dict["z_noise"],
            latent_dict["peak_pos"].unsqueeze(-1),
        ],
        dim=-1,
    )


def _slice_next_step(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """兼容 [B, 1, ...] 和 [B, ...] 两种未来步输入。"""
    if x is None:
        return None
    if x.dim() >= 2 and x.size(1) == 1:
        return x[:, 0]
    return x


class DirectWindTransition(nn.Module):
    """历史 latent + 历史风速 -> 下一时刻直接风速预测。"""

    def __init__(
        self,
        d_model: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        input_dim = d_model * 3 + 2
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.wind_head = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        history_latent: Dict[str, torch.Tensor],
        past_wind: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, history_len, n_gate, _ = history_latent["z_peak"].shape
        latent_features = _merge_latent_features(history_latent)
        structured_input = torch.cat([latent_features, past_wind.unsqueeze(-1)], dim=-1)

        per_gate_sequence = structured_input.permute(0, 2, 1, 3).reshape(
            batch_size * n_gate, history_len, -1
        )
        encoded_input = self.input_proj(per_gate_sequence)
        _, temporal_hidden = self.temporal_gru(encoded_input)

        last_hidden = temporal_hidden[-1]
        last_feature = per_gate_sequence[:, -1, :]
        next_wind = self.wind_head(torch.cat([last_hidden, last_feature], dim=-1))
        return next_wind.reshape(batch_size, n_gate)


class DLinearDirectWindTransition(nn.Module):
    """DLinear-style direct branch over causal latent and wind histories.

    Each gate is processed independently.  The moving-average decomposition
    and temporal projections operate only on the observed ``input_steps``;
    no future spectrum or wind enters this module.
    """

    def __init__(self, d_model: int = 64, n_gate: int = 20, input_steps: int = 12, moving_average: int = 3, latent_correction_scale: float = 0.25):
        super().__init__()
        if input_steps <= 0:
            raise ValueError("input_steps must be positive")
        if moving_average <= 0 or moving_average % 2 == 0:
            raise ValueError("moving_average must be a positive odd integer")
        self.input_steps = int(input_steps)
        self.moving_average = int(moving_average)
        self.n_gate = int(n_gate)
        self.latent_correction_scale = float(latent_correction_scale)
        self.wind_seasonal = nn.ModuleList([nn.Linear(self.input_steps, 1) for _ in range(self.n_gate)])
        self.wind_trend = nn.ModuleList([nn.Linear(self.input_steps, 1) for _ in range(self.n_gate)])
        self.latent_temporal = nn.Linear(self.input_steps, 1)
        self.latent_head = nn.Linear(d_model * 3 + 1, 1)
        # Standard DLinear averaging initialization gives a stable forecast at
        # epoch zero.  The physics-latent correction starts as an exact zero,
        # so it cannot destroy that anchor before validation demonstrates gain.
        for layer in list(self.wind_seasonal) + list(self.wind_trend) + [self.latent_temporal]:
            nn.init.constant_(layer.weight, 1.0 / self.input_steps)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.latent_head.weight)
        nn.init.zeros_(self.latent_head.bias)

    def _moving_average(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence: [B*G,F,T]. Replication padding preserves the history length.
        padding = (self.moving_average - 1) // 2
        padded = F.pad(sequence, (padding, padding), mode="replicate")
        return F.avg_pool1d(padded, kernel_size=self.moving_average, stride=1)

    def forward(
        self,
        history_latent: Dict[str, torch.Tensor],
        past_wind: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, history_len, n_gate, _ = history_latent["z_peak"].shape
        if history_len != self.input_steps or past_wind.shape[:2] != (batch_size, history_len):
            raise ValueError(
                f"DLinear direct branch expects history length {self.input_steps}; "
                f"got latent={history_len}, wind={tuple(past_wind.shape)}"
            )
        if n_gate != self.n_gate:
            raise ValueError(f"DLinear configured for {self.n_gate} gates, got {n_gate}")
        wind_sequence = past_wind.permute(0, 2, 1)  # [B,G,T]
        wind_trend = self._moving_average(wind_sequence)
        wind_seasonal = wind_sequence - wind_trend
        wind_base = torch.cat(
            [self.wind_seasonal[gate](wind_seasonal[:, gate]) + self.wind_trend[gate](wind_trend[:, gate])
             for gate in range(n_gate)],
            dim=1,
        )

        latent_features = _merge_latent_features(history_latent)  # [B,T,G,3D+1]
        latent_sequence = latent_features.permute(0, 2, 3, 1).reshape(batch_size * n_gate, -1, history_len)
        latent_context = self.latent_temporal(latent_sequence).squeeze(-1)
        latent_residual = self.latent_correction_scale * torch.tanh(self.latent_head(latent_context))
        return wind_base + latent_residual.reshape(batch_size, n_gate)


class LatentTransitionConditionedOnWind(nn.Module):
    """历史 latent + 预测风速 -> 下一时刻 latent 预测。"""

    def __init__(
        self,
        d_model: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        input_dim = d_model * 3 + 1
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        conditioned_dim = hidden_dim + d_model * 3 + 2
        self.state_proj = nn.Sequential(
            nn.Linear(conditioned_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.z_peak_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.z_width_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.z_noise_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.peak_pos_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        history_latent: Dict[str, torch.Tensor],
        predicted_next_wind: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, history_len, n_gate, d_model = history_latent["z_peak"].shape
        latent_features = _merge_latent_features(history_latent)
        per_gate_sequence = latent_features.permute(0, 2, 1, 3).reshape(
            batch_size * n_gate, history_len, -1
        )

        encoded_input = self.input_proj(per_gate_sequence)
        _, temporal_hidden = self.temporal_gru(encoded_input)

        last_z_peak = history_latent["z_peak"][:, -1].reshape(batch_size * n_gate, d_model)
        last_z_width = history_latent["z_width"][:, -1].reshape(batch_size * n_gate, d_model)
        last_z_noise = history_latent["z_noise"][:, -1].reshape(batch_size * n_gate, d_model)
        last_peak_pos = history_latent["peak_pos"][:, -1].reshape(batch_size * n_gate, 1)
        predicted_wind = predicted_next_wind.reshape(batch_size * n_gate, 1)

        conditioned_feature = torch.cat(
            [
                temporal_hidden[-1],
                last_z_peak,
                last_z_width,
                last_z_noise,
                last_peak_pos,
                predicted_wind,
            ],
            dim=-1,
        )
        conditioned_feature = self.state_proj(conditioned_feature)

        z_peak_hat = last_z_peak + self.z_peak_head(conditioned_feature)
        z_width_hat = last_z_width + self.z_width_head(conditioned_feature)
        z_noise_hat = last_z_noise + self.z_noise_head(conditioned_feature)
        peak_pos_hat = torch.sigmoid(last_peak_pos + self.peak_pos_head(conditioned_feature))

        return {
            "z_peak": z_peak_hat.reshape(batch_size, n_gate, d_model),
            "z_width": z_width_hat.reshape(batch_size, n_gate, d_model),
            "z_noise": z_noise_hat.reshape(batch_size, n_gate, d_model),
            "peak_pos": peak_pos_hat.reshape(batch_size, n_gate),
        }


class CausalDirectGuidedDecoder(nn.Module):
    """Causal mixture-of-experts decoder with a physics-latent residual.

    The gate sees only predicted latent state, current expert outputs, causal
    histories and rollout progress. Ground-truth expert advantage is used only
    as a training loss target and is never an inference input.
    """

    MASKABLE_LATENTS = ("z_peak", "z_width", "z_noise")

    def __init__(self, d_model: int = 64, hidden_dim: int = 128, residual_scale: float = 1.0, hard_gate: bool = False, gate_temperature: float = 1.0, output_mode: str = "expert_mix", mask_latents: Sequence[str] = ()):
        super().__init__()
        input_dim = d_model * 3 + 8
        self.residual_scale = float(residual_scale)
        self.hard_gate = bool(hard_gate)
        self.gate_temperature = float(gate_temperature)
        self.output_mode = str(output_mode)
        self.mask_latents = tuple(str(key) for key in mask_latents)
        if self.gate_temperature <= 0.0:
            raise ValueError("gate_temperature must be positive")
        if self.output_mode not in {"expert_mix", "direct_anchor_correction"}:
            raise ValueError("output_mode must be expert_mix or direct_anchor_correction")
        if len(set(self.mask_latents)) != len(self.mask_latents):
            raise ValueError("mask_latents must not contain duplicates")
        unsupported_masks = sorted(set(self.mask_latents) - set(self.MASKABLE_LATENTS))
        if unsupported_masks:
            raise ValueError(
                f"Unsupported guidance latent masks: {unsupported_masks}; "
                f"expected a subset of {self.MASKABLE_LATENTS}"
            )
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.gate_head = nn.Linear(hidden_dim, 1)
        self.residual_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self,
        latent_state: Dict[str, torch.Tensor],
        direct: torch.Tensor,
        physics_decode: torch.Tensor,
        last_direct_wind: torch.Tensor,
        last_decode_wind: torch.Tensor,
        progress: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, n_gate = direct.shape
        guidance_latent = {
            key: torch.zeros_like(latent_state[key]) if key in self.mask_latents else latent_state[key]
            for key in self.MASKABLE_LATENTS
        }
        latent = torch.cat(
            [guidance_latent["z_peak"], guidance_latent["z_width"], guidance_latent["z_noise"], latent_state["peak_pos"].unsqueeze(-1)],
            dim=-1,
        ).detach()
        if progress.shape != (batch_size, n_gate):
            raise ValueError(f"progress must be {(batch_size, n_gate)}, got {tuple(progress.shape)}")
        features = torch.cat(
            [
                latent,
                direct.detach().unsqueeze(-1),
                physics_decode.detach().unsqueeze(-1),
                last_direct_wind.detach().unsqueeze(-1),
                last_decode_wind.detach().unsqueeze(-1),
                (direct.detach() - physics_decode.detach()).abs().unsqueeze(-1),
                (direct.detach() - last_direct_wind.detach()).abs().unsqueeze(-1),
                progress.unsqueeze(-1),
            ],
            dim=-1,
        )
        hidden = self.feature_net(features.reshape(batch_size * n_gate, -1))
        gate_logits = self.gate_head(hidden).reshape(batch_size, n_gate)
        gate_soft = torch.sigmoid(gate_logits / self.gate_temperature)
        if self.hard_gate:
            gate_hard = (gate_soft >= 0.5).to(gate_soft.dtype)
            gate = gate_hard + gate_soft - gate_soft.detach() if self.training else gate_hard
        else:
            gate = gate_soft
        residual = self.residual_scale * torch.tanh(self.residual_head(hidden)).reshape(batch_size, n_gate)
        if self.output_mode == "direct_anchor_correction":
            v_guided = direct.detach() + (1.0 - gate) * residual
        else:
            mixture = physics_decode + gate * (direct.detach() - physics_decode)
            v_guided = mixture + residual
        return {
            "v_guided": v_guided,
            "gate": gate,
            "gate_soft": gate_soft,
            "gate_logits": gate_logits,
            "residual": residual,
        }


class DisentangledTemporalWindForecaster(nn.Module):
    """单步自回归风速-隐状态耦合预测模型。"""

    def __init__(
        self,
        n_freq: int = 100,
        n_gate: int = 20,
        d_model: int = 64,
        n_heads: int = 4,
        wavelength: float = 1.55,
        f_res: float = 1000 / 1024,
        temporal_hidden_dim: int = 128,
        temporal_num_layers: int = 1,
        temporal_dropout: float = 0.1,
        lambda_direct: float = 1.0,
        lambda_decode: float = 1.0,
        lambda_align: float = 0.5,
        lambda_phys: float = 0.5,
        lambda_state: float = 1.0,
        lambda_uncertainty: float = 0.2,
        lambda_state_peak: float = 1.0,
        lambda_state_width: float = 1.0,
        lambda_state_noise: float = 1.0,
        lambda_state_pos: float = 1.0,
        lambda_consistency: float = 0.0,
        decoder_mode: str = "physics",
        latent_layout: str = "structured",
        lambda_cons: float = 0.5,
        mi_weight: float = 0.1,
        direct_transition_type: str = "gru",
        direct_input_steps: int = 12,
        direct_dlinear_kernel: int = 3,
        guidance_enabled: bool = False,
        guidance_hidden_dim: int = 128,
        guidance_residual_scale: float = 1.0,
        guidance_hard_gate: bool = False,
        guidance_gate_temperature: float = 1.0,
        guidance_output_mode: str = "expert_mix",
        guidance_mask_latents: Sequence[str] = (),
    ):
        super().__init__()

        self.n_freq = n_freq
        self.n_gate = n_gate
        self.d_model = d_model

        self.single_step_model = DisentangledWindEstimator(
            n_freq=n_freq,
            n_gate=n_gate,
            d_model=d_model,
            n_heads=n_heads,
            wavelength=wavelength,
            f_res=f_res,
            decoder_mode=decoder_mode,
            latent_layout=latent_layout,
            lambda_cons=lambda_cons,
            mi_weight=mi_weight,
        )
        self.encoder = self.single_step_model.encoder
        self.decoder = self.single_step_model.decoder

        if direct_transition_type == "gru":
            self.direct_transition = DirectWindTransition(
                d_model=d_model,
                hidden_dim=temporal_hidden_dim,
                num_layers=temporal_num_layers,
                dropout=temporal_dropout,
            )
        elif direct_transition_type == "dlinear":
            self.direct_transition = DLinearDirectWindTransition(
                d_model=d_model,
                n_gate=n_gate,
                input_steps=direct_input_steps,
                moving_average=direct_dlinear_kernel,
            )
        else:
            raise ValueError(f"Unsupported direct_transition_type: {direct_transition_type}")
        self.direct_transition_type = direct_transition_type
        self.latent_transition = LatentTransitionConditionedOnWind(
            d_model=d_model,
            hidden_dim=temporal_hidden_dim,
            num_layers=temporal_num_layers,
            dropout=temporal_dropout,
        )
        self.guided_decoder = (
            CausalDirectGuidedDecoder(
                d_model=d_model,
                hidden_dim=guidance_hidden_dim,
                residual_scale=guidance_residual_scale,
                hard_gate=guidance_hard_gate,
                gate_temperature=guidance_gate_temperature,
                output_mode=guidance_output_mode,
                mask_latents=guidance_mask_latents,
            )
            if guidance_enabled
            else None
        )

        self.loss_weights = {
            "lambda_direct": lambda_direct,
            "lambda_decode": lambda_decode,
            "lambda_align": lambda_align,
            "lambda_phys": lambda_phys,
            "lambda_state": lambda_state,
            "lambda_uncertainty": lambda_uncertainty,
            "lambda_state_peak": lambda_state_peak,
            "lambda_state_width": lambda_state_width,
            "lambda_state_noise": lambda_state_noise,
            "lambda_state_pos": lambda_state_pos,
            "lambda_consistency": lambda_consistency,
        }

    def encode_history(self, spectrum_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """逐时刻编码历史谱序列。"""
        batch_size, history_len, n_gate, n_freq = spectrum_history.shape
        flat_spectrum = spectrum_history.reshape(batch_size * history_len, n_gate, n_freq)
        encoded = self.encoder(flat_spectrum)

        history_encoded: Dict[str, torch.Tensor] = {}
        for key, value in encoded.items():
            if value.dim() == 2:
                history_encoded[key] = value.reshape(batch_size, history_len, n_gate)
            elif value.dim() == 3:
                history_encoded[key] = value.reshape(
                    batch_size, history_len, value.size(1), value.size(2)
                )
            else:
                raise ValueError(f"Unsupported encoded tensor shape for key={key}: {value.shape}")

        return history_encoded

    def _compute_temporal_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        future_wind: Optional[torch.Tensor],
        future_spectrum: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = outputs["v_direct_pred"].device
        zero = torch.zeros((), device=device)

        losses: Dict[str, torch.Tensor] = {
            "L_direct": zero,
            "L_decode": zero,
            "L_align": F.mse_loss(outputs["v_direct_pred"], outputs["v_decode_pred"]),
            "L_phys": F.mse_loss(outputs["v_decode_pred"], outputs["v_physics_pred"]),
            "L_state_peak": zero,
            "L_state_width": zero,
            "L_state_noise": zero,
            "L_state_pos": zero,
            "L_state": zero,
            "L_uncertainty": zero,
        }

        if future_wind is not None:
            losses["L_direct"] = F.mse_loss(outputs["v_direct_pred"], future_wind)
            losses["L_decode"] = F.mse_loss(outputs["v_decode_pred"], future_wind)
            decode_error = (outputs["v_decode_pred"] - future_wind).abs()
            losses["L_uncertainty"] = F.mse_loss(
                outputs["uncertainty_pred"], decode_error.detach()
            )

        if future_spectrum is not None:
            teacher_latent = self.encoder(future_spectrum)
            outputs["teacher_latent"] = teacher_latent
            losses["L_state_peak"] = F.mse_loss(
                outputs["z_peak_pred"], teacher_latent["z_peak"]
            )
            losses["L_state_width"] = F.mse_loss(
                outputs["z_width_pred"], teacher_latent["z_width"]
            )
            losses["L_state_noise"] = F.mse_loss(
                outputs["z_noise_pred"], teacher_latent["z_noise"]
            )
            losses["L_state_pos"] = F.mse_loss(
                outputs["peak_pos_pred"], teacher_latent["peak_pos"]
            )
            losses["L_state"] = (
                self.loss_weights["lambda_state_peak"] * losses["L_state_peak"]
                + self.loss_weights["lambda_state_width"] * losses["L_state_width"]
                + self.loss_weights["lambda_state_noise"] * losses["L_state_noise"]
                + self.loss_weights["lambda_state_pos"] * losses["L_state_pos"]
            )

        losses["total_loss"] = (
            self.loss_weights["lambda_direct"] * losses["L_direct"]
            + self.loss_weights["lambda_decode"] * losses["L_decode"]
            + self.loss_weights["lambda_align"] * losses["L_align"]
            + self.loss_weights["lambda_phys"] * losses["L_phys"]
            + self.loss_weights["lambda_state"] * losses["L_state"]
            + self.loss_weights["lambda_uncertainty"] * losses["L_uncertainty"]
        )
        return losses

    def forward(
        self,
        spectrum_history: torch.Tensor,
        past_wind: torch.Tensor,
        future_spectrum: Optional[torch.Tensor] = None,
        future_wind: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        future_spectrum_step = _slice_next_step(future_spectrum)
        future_wind_step = _slice_next_step(future_wind)

        history_latent = self.encode_history(spectrum_history)
        v_direct_pred = self.direct_transition(history_latent, past_wind)
        z_hat_next = self.latent_transition(history_latent, v_direct_pred)
        decoded = self.decoder(
            z_hat_next["z_peak"],
            z_hat_next["z_width"],
            z_hat_next["z_noise"],
            z_hat_next["peak_pos"],
        )

        outputs: Dict[str, Any] = {
            "history_latent": history_latent,
            "v_direct_pred": v_direct_pred,
            "v_decode_pred": decoded["v_estimate"],
            "v_physics_pred": decoded["v_physics"],
            "turbulence_pred": decoded["turbulence"],
            "uncertainty_pred": decoded["uncertainty"],
            "z_peak_pred": z_hat_next["z_peak"],
            "z_width_pred": z_hat_next["z_width"],
            "z_noise_pred": z_hat_next["z_noise"],
            "peak_pos_pred": z_hat_next["peak_pos"],
            "correction_pred": decoded["correction"],
        }

        outputs["losses"] = self._compute_temporal_losses(
            outputs=outputs,
            future_wind=future_wind_step,
            future_spectrum=future_spectrum_step,
        )
        return outputs

    @torch.no_grad()
    def predict(
        self,
        spectrum_history: torch.Tensor,
        past_wind: torch.Tensor,
        horizon: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """通过单步自回归循环实现多步预测。"""
        self.eval()

        history_latent = self.encode_history(spectrum_history)
        wind_history = past_wind

        v_direct_preds = []
        v_decode_preds = []
        v_physics_preds = []
        peak_pos_preds = []
        z_peak_preds = []
        z_width_preds = []
        z_noise_preds = []
        turbulence_preds = []
        uncertainty_preds = []

        for _ in range(horizon):
            v_direct_pred = self.direct_transition(history_latent, wind_history)
            z_hat_next = self.latent_transition(history_latent, v_direct_pred)
            decoded = self.decoder(
                z_hat_next["z_peak"],
                z_hat_next["z_width"],
                z_hat_next["z_noise"],
                z_hat_next["peak_pos"],
            )

            v_direct_preds.append(v_direct_pred)
            v_decode_preds.append(decoded["v_estimate"])
            v_physics_preds.append(decoded["v_physics"])
            peak_pos_preds.append(z_hat_next["peak_pos"])
            z_peak_preds.append(z_hat_next["z_peak"])
            z_width_preds.append(z_hat_next["z_width"])
            z_noise_preds.append(z_hat_next["z_noise"])
            turbulence_preds.append(decoded["turbulence"])
            uncertainty_preds.append(decoded["uncertainty"])

            history_latent = {
                "z_peak": torch.cat(
                    [history_latent["z_peak"][:, 1:], z_hat_next["z_peak"].unsqueeze(1)],
                    dim=1,
                ),
                "z_width": torch.cat(
                    [history_latent["z_width"][:, 1:], z_hat_next["z_width"].unsqueeze(1)],
                    dim=1,
                ),
                "z_noise": torch.cat(
                    [history_latent["z_noise"][:, 1:], z_hat_next["z_noise"].unsqueeze(1)],
                    dim=1,
                ),
                "peak_pos": torch.cat(
                    [history_latent["peak_pos"][:, 1:], z_hat_next["peak_pos"].unsqueeze(1)],
                    dim=1,
                ),
            }
            wind_history = torch.cat([wind_history[:, 1:], v_direct_pred.unsqueeze(1)], dim=1)

        return {
            "v_direct_pred": torch.stack(v_direct_preds, dim=1),
            "v_decode_pred": torch.stack(v_decode_preds, dim=1),
            "v_physics_pred": torch.stack(v_physics_preds, dim=1),
            "peak_pos_pred": torch.stack(peak_pos_preds, dim=1),
            "z_peak_pred": torch.stack(z_peak_preds, dim=1),
            "z_width_pred": torch.stack(z_width_preds, dim=1),
            "z_noise_pred": torch.stack(z_noise_preds, dim=1),
            "turbulence_pred": torch.stack(turbulence_preds, dim=1),
            "uncertainty_pred": torch.stack(uncertainty_preds, dim=1),
        }

    def predict_with_refinement(
        self,
        spectrum_history: torch.Tensor,
        past_wind: torch.Tensor,
        horizon: int,
        refinement_config: Any,
        history_latent_override: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Autoregressive prediction whose enabled final/feedback output is refined decode."""
        from refinement import refine_latent_state

        self.eval()
        with torch.no_grad():
            history_latent = self.encode_history(spectrum_history)
            if history_latent_override is not None:
                missing = [key for key in ("z_peak", "z_width", "z_noise", "peak_pos") if key not in history_latent_override]
                if missing:
                    raise KeyError(f"history_latent_override is missing: {missing}")
                history_latent = {
                    key: torch.cat([history_latent[key][:, :-1], history_latent_override[key].detach().unsqueeze(1)], dim=1)
                    for key in ("z_peak", "z_width", "z_noise", "peak_pos")
                }
        wind_history = past_wind.detach()
        tensor_rows: Dict[str, List[torch.Tensor]] = {
            "v_direct_pred": [],
            "v_decode_pred": [],
            "v_decode_refined": [],
            "v_physics_pred": [],
            "z_peak_pred": [],
            "z_width_pred": [],
            "z_noise_pred": [],
            "peak_pos_pred": [],
            "z_peak_refined": [],
            "z_width_refined": [],
            "z_noise_refined": [],
            "peak_pos_refined": [],
            "refinement_accepted": [],
            "latent_shift": [],
            "loss_pseudo_before": [],
            "loss_pseudo_after": [],
            "loss_anchor_after": [],
            "z_peak_shift": [],
            "z_width_shift": [],
            "z_noise_shift": [],
            "peak_pos_shift": [],
            "latent_norm_before": [],
            "latent_norm_after": [],
            "correction_change": [],
        }
        fallback_reasons: List[List[str]] = []
        for _ in range(int(horizon)):
            with torch.no_grad():
                direct_pred = self.direct_transition(history_latent, wind_history)
                latent_pred = self.latent_transition(history_latent, direct_pred)
                decoded_before = self.decoder(latent_pred["z_peak"], latent_pred["z_width"], latent_pred["z_noise"], latent_pred["peak_pos"])
            refinement = refine_latent_state(self.decoder, latent_pred, direct_pred, refinement_config)
            refined_latent = refinement["z_refined"]
            final_wind = refinement["v_decode_refined"]
            tensor_rows["v_direct_pred"].append(direct_pred.detach())
            tensor_rows["v_decode_pred"].append(decoded_before["v_estimate"].detach())
            tensor_rows["v_decode_refined"].append(final_wind.detach())
            tensor_rows["v_physics_pred"].append(decoded_before["v_physics"].detach())
            tensor_rows["z_peak_pred"].append(latent_pred["z_peak"].detach())
            tensor_rows["z_width_pred"].append(latent_pred["z_width"].detach())
            tensor_rows["z_noise_pred"].append(latent_pred["z_noise"].detach())
            tensor_rows["peak_pos_pred"].append(latent_pred["peak_pos"].detach())
            tensor_rows["z_peak_refined"].append(refined_latent["z_peak"].detach())
            tensor_rows["z_width_refined"].append(refined_latent["z_width"].detach())
            tensor_rows["z_noise_refined"].append(refined_latent["z_noise"].detach())
            tensor_rows["peak_pos_refined"].append(refined_latent["peak_pos"].detach())
            for key in ("refinement_accepted", "latent_shift", "loss_pseudo_before", "loss_pseudo_after", "loss_anchor_after", "z_peak_shift", "z_width_shift", "z_noise_shift", "peak_pos_shift", "latent_norm_before", "latent_norm_after", "correction_change"):
                tensor_rows[key].append(refinement[key].detach())
            fallback_reasons.append(list(refinement["fallback_reason"]))
            roll_latent = refined_latent if refinement_config.propagate_refined_latent else latent_pred
            with torch.no_grad():
                history_latent = _update_history_latent(history_latent, roll_latent)
                wind_history = _update_wind_history(wind_history, final_wind)
        result: Dict[str, Any] = {key: torch.stack(values, dim=1) for key, values in tensor_rows.items()}
        result["fallback_reason"] = fallback_reasons
        return result


class IdentifiabilityVerifier:
    """单时刻解耦表示可辨识性检验工具。"""

    def __init__(self, model: DisentangledWindEstimator):
        self.model = model

    def intervention_test(
        self,
        base_spectrum: torch.Tensor,
        intervention_type: str = "width",
        num_variations: int = 10,
    ) -> Dict[str, float]:
        self.model.eval()

        with torch.no_grad():
            base_output = self.model.encoder(base_spectrum)
            results = {
                "z_peak_variance": [],
                "z_width_variance": [],
                "z_noise_variance": [],
            }

            for _ in range(num_variations):
                if intervention_type == "width":
                    varied_spectrum = self._vary_width(base_spectrum)
                elif intervention_type == "peak":
                    varied_spectrum = self._vary_peak(base_spectrum)
                else:
                    varied_spectrum = self._vary_noise(base_spectrum)

                output = self.model.encoder(varied_spectrum)
                results["z_peak_variance"].append(
                    (output["z_peak"] - base_output["z_peak"]).var().item()
                )
                results["z_width_variance"].append(
                    (output["z_width"] - base_output["z_width"]).var().item()
                )
                results["z_noise_variance"].append(
                    (output["z_noise"] - base_output["z_noise"]).var().item()
                )

        return {key: float(np.mean(values)) for key, values in results.items()}

    def _vary_width(self, spectrum: torch.Tensor) -> torch.Tensor:
        kernel_size = int(np.random.choice([3, 5, 7, 9]))
        kernel = torch.ones(1, 1, kernel_size, device=spectrum.device) / kernel_size
        batch_size, n_gate, n_freq = spectrum.shape
        return F.conv1d(
            spectrum.reshape(batch_size * n_gate, 1, n_freq),
            kernel,
            padding=kernel_size // 2,
        ).reshape(batch_size, n_gate, n_freq)

    def _vary_peak(self, spectrum: torch.Tensor) -> torch.Tensor:
        shift = int(np.random.randint(-10, 10))
        return torch.roll(spectrum, shifts=shift, dims=-1)

    def _vary_noise(self, spectrum: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(spectrum) * 0.1 * spectrum.std().clamp_min(1e-6)
        return spectrum + noise


class TemporalWindDataset(Dataset):
    """读取 preprocess_data.py 生成的 one-step npz 数据。"""

    def __init__(self, npz_path: str):
        super().__init__()
        data = np.load(npz_path)
        self.x_spec = data["X_spec"].astype(np.float32)
        self.x_wind = data["X_wind"].astype(np.float32)
        self.y_spec = data["Y_spec"].astype(np.float32)
        self.y_wind = data["Y_wind"].astype(np.float32)

    def __len__(self) -> int:
        return len(self.x_spec)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "spectrum_history": torch.from_numpy(self.x_spec[idx]),
            "past_wind": torch.from_numpy(self.x_wind[idx]),
            "future_spectrum": torch.from_numpy(self.y_spec[idx]),
            "future_wind": torch.from_numpy(self.y_wind[idx]),
        }


def build_temporal_model_from_dataset(sample: Dict[str, torch.Tensor], d_model: int = 64, n_heads: int = 4, temporal_hidden_dim: int = 128, decoder_mode: str = "physics", latent_layout: str = "structured", lambda_cons: float = 0.5, mi_weight: float = 0.1, direct_transition_type: str = "gru", direct_input_steps: int = 12, direct_dlinear_kernel: int = 3, guidance_enabled: bool = False, guidance_hidden_dim: int = 128, guidance_residual_scale: float = 1.0, guidance_hard_gate: bool = False, guidance_gate_temperature: float = 1.0, guidance_output_mode: str = "expert_mix", guidance_mask_latents: Sequence[str] = ()) -> DisentangledTemporalWindForecaster:
    _, n_gate, n_freq = sample["spectrum_history"].shape
    return DisentangledTemporalWindForecaster(n_freq=n_freq, n_gate=n_gate, d_model=d_model, n_heads=n_heads, temporal_hidden_dim=temporal_hidden_dim, decoder_mode=decoder_mode, latent_layout=latent_layout, lambda_cons=lambda_cons, mi_weight=mi_weight, direct_transition_type=direct_transition_type, direct_input_steps=direct_input_steps, direct_dlinear_kernel=direct_dlinear_kernel, guidance_enabled=guidance_enabled, guidance_hidden_dim=guidance_hidden_dim, guidance_residual_scale=guidance_residual_scale, guidance_hard_gate=guidance_hard_gate, guidance_gate_temperature=guidance_gate_temperature, guidance_output_mode=guidance_output_mode, guidance_mask_latents=guidance_mask_latents)


def load_normalization_stats(stats_path: Optional[str]) -> Dict[str, float]:
    if not stats_path:
        return {}
    stats_data = np.load(stats_path)
    return {key: float(stats_data[key]) for key in stats_data.files}


def build_window_start_indices(num_steps: int, input_steps: int, output_steps: int, stride: int) -> np.ndarray:
    if input_steps <= 0 or output_steps <= 0:
        raise ValueError("input_steps and output_steps must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    max_start = num_steps - input_steps - output_steps
    if max_start < 0:
        return np.empty((0,), dtype=np.int64)
    return np.arange(0, max_start + 1, stride, dtype=np.int64)


class TemporalSequenceDataset(Dataset):
    def __init__(self, npz_path: str, input_steps: int, output_steps: int, stride: int, expected_split: Optional[str] = None):
        super().__init__()
        data = np.load(npz_path, allow_pickle=False)
        self.spectra = data["spectra"].astype(np.float32)
        self.wind = data["wind_speed"].astype(np.float32)
        self.step_indices = data["step_indices"].astype(np.int64)
        self.split_name = str(data["split_name"]) if "split_name" in data.files else "unknown"
        self.step_records = int(data["step_records"]) if "step_records" in data.files else -1
        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)
        self.stride = int(stride)
        if expected_split and self.split_name != expected_split:
            raise ValueError(f"Split mismatch for {npz_path}: expected {expected_split}, got {self.split_name}.")
        if self.step_records <= 0:
            raise ValueError(f"{npz_path} has invalid step_records={self.step_records}.")
        if self.spectra.shape[0] != self.wind.shape[0]:
            raise ValueError(f"Temporal length mismatch in {npz_path}: spectra={self.spectra.shape[0]}, wind={self.wind.shape[0]}.")
        self.window_starts = build_window_start_indices(int(self.spectra.shape[0]), self.input_steps, self.output_steps, self.stride)
        if len(self.window_starts) == 0:
            raise ValueError(f"No windows can be formed from {npz_path}.")

    def __len__(self) -> int:
        return len(self.window_starts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = int(self.window_starts[idx])
        history_end = start + self.input_steps
        future_end = history_end + self.output_steps
        return {
            "spectrum_history": torch.from_numpy(self.spectra[start:history_end]),
            "past_wind": torch.from_numpy(self.wind[start:history_end]),
            "future_spectrum": torch.from_numpy(self.spectra[history_end:future_end]),
            "future_wind": torch.from_numpy(self.wind[history_end:future_end]),
            "start_step": torch.tensor(int(self.step_indices[start]), dtype=torch.long),
        }


class SingleStepWindDataset(Dataset):
    def __init__(self, npz_path: str, expected_split: Optional[str] = None):
        super().__init__()
        data = np.load(npz_path, allow_pickle=False)
        self.spectra = data["spectra"].astype(np.float32)
        self.wind = data["wind_speed"].astype(np.float32)
        self.step_indices = data["step_indices"].astype(np.int64)
        self.split_name = str(data["split_name"]) if "split_name" in data.files else "unknown"
        if expected_split and self.split_name != expected_split:
            raise ValueError(f"Split mismatch for {npz_path}: expected {expected_split}, got {self.split_name}.")

    def __len__(self) -> int:
        return int(self.spectra.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"spectrum": torch.from_numpy(self.spectra[idx]), "wind": torch.from_numpy(self.wind[idx]), "step_index": torch.tensor(int(self.step_indices[idx]), dtype=torch.long)}


def _module_parameter_counts(module: nn.Module) -> Dict[str, int]:
    return {"trainable": int(sum(p.numel() for p in module.parameters() if p.requires_grad)), "frozen": int(sum(p.numel() for p in module.parameters() if not p.requires_grad))}


def _print_module_freeze_report(title: str, module_map: Dict[str, nn.Module]) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)
    total_trainable = 0
    total_frozen = 0
    for module_name, module in module_map.items():
        counts = _module_parameter_counts(module)
        total_trainable += counts["trainable"]
        total_frozen += counts["frozen"]
        print(f"{module_name:<24} trainable={counts['trainable']:<10d} frozen={counts['frozen']:<10d}")
    print(f"TOTAL                    trainable={total_trainable:<10d} frozen={total_frozen:<10d}")


def _set_module_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def _fingerprint_module(module: nn.Module) -> str:
    hasher = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def _build_optimizer(parameters: List[nn.Parameter], lr: float, name: str) -> torch.optim.Optimizer:
    if not parameters:
        raise ValueError(f"{name} received no parameters.")
    if len({id(param) for param in parameters}) != len(parameters):
        raise ValueError(f"{name} received duplicate parameters.")
    for param in parameters:
        if not param.requires_grad:
            raise ValueError(f"{name} received frozen parameters.")
    return torch.optim.Adam(parameters, lr=lr)


def _assert_optimizer_matches_parameters(optimizer: torch.optim.Optimizer, expected_parameters: List[nn.Parameter], optimizer_name: str) -> None:
    expected_ids = {id(param) for param in expected_parameters}
    optimizer_ids: Set[int] = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            optimizer_ids.add(id(param))
            if not param.requires_grad:
                raise ValueError(f"{optimizer_name} contains frozen parameters.")
    if optimizer_ids != expected_ids:
        raise ValueError(f"{optimizer_name} parameter mismatch.")


def _assert_optimizers_disjoint(optimizers: Dict[str, Optional[torch.optim.Optimizer]]) -> None:
    """Reject parameters registered in more than one optimizer."""
    owners: Dict[int, str] = {}
    for optimizer_name, optimizer in optimizers.items():
        if optimizer is None:
            continue
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter_id = id(parameter)
                if parameter_id in owners:
                    raise ValueError(
                        f"Optimizer parameter overlap: {optimizer_name} and "
                        f"{owners[parameter_id]} contain the same parameter."
                    )
                owners[parameter_id] = optimizer_name


def _consistency_weight(
    epoch: int,
    total_epochs: int,
    schedule_type: str,
    warmup_epochs: int,
    target_weight: float,
) -> float:
    if target_weight < 0.0:
        raise ValueError("Consistency target weight must be non-negative.")
    if schedule_type == "constant":
        return float(target_weight)
    if schedule_type != "delayed_linear":
        raise ValueError(f"Unsupported consistency schedule: {schedule_type}")
    if warmup_epochs < 0:
        raise ValueError("Consistency warmup epochs must be non-negative.")
    if epoch <= warmup_epochs:
        return 0.0
    ramp_epochs = max(int(total_epochs) - int(warmup_epochs), 1)
    progress = min(max((int(epoch) - int(warmup_epochs)) / ramp_epochs, 0.0), 1.0)
    return float(target_weight) * progress


def _structured_state_losses(
    model: DisentangledTemporalWindForecaster,
    latent_pred: Dict[str, torch.Tensor],
    latent_ref: Optional[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    zero = torch.zeros((), device=device)
    losses = {"peak": zero, "width": zero, "noise": zero, "pos": zero, "total": zero}
    if latent_ref is None:
        return losses
    losses["peak"] = F.mse_loss(latent_pred["z_peak"], latent_ref["z_peak"].detach())
    losses["width"] = F.mse_loss(latent_pred["z_width"], latent_ref["z_width"].detach())
    losses["noise"] = F.mse_loss(latent_pred["z_noise"], latent_ref["z_noise"].detach())
    losses["pos"] = F.mse_loss(latent_pred["peak_pos"], latent_ref["peak_pos"].detach())
    losses["total"] = (
        model.loss_weights["lambda_state_peak"] * losses["peak"]
        + model.loss_weights["lambda_state_width"] * losses["width"]
        + model.loss_weights["lambda_state_noise"] * losses["noise"]
        + model.loss_weights["lambda_state_pos"] * losses["pos"]
    )
    return losses


def _write_csv_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _resolve_checkpoint_paths(save_path: Optional[str]) -> Dict[str, Path]:
    base = Path(save_path or "checkpoints/temporal_forecaster.pt")
    directory = base.parent if base.suffix else base
    stem = base.stem if base.suffix else "temporal_forecaster"
    suffix = base.suffix if base.suffix else ".pt"
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "stage1_best": directory / f"{stem}.stage1.best{suffix}",
        "stage1_last": directory / f"{stem}.stage1.last{suffix}",
        "stage2_best": directory / f"{stem}.stage2.best{suffix}",
        "stage2_last": directory / f"{stem}.stage2.last{suffix}",
        "stage1_log": directory / f"{stem}.stage1.metrics.csv",
        "stage2_epoch_log": directory / f"{stem}.stage2.epoch_metrics.csv",
        "stage2_window_log": directory / f"{stem}.stage2.window_metrics.csv",
        "training_summary": directory / f"{stem}.training_summary.json",
    }


def _save_checkpoint(path: Path, payload: Dict[str, Any], announce: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    if announce:
        print(f"saved checkpoint to: {path}")


def _extract_stage1_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "stage1_model_state_dict" in checkpoint:
        return checkpoint["stage1_model_state_dict"]
    state = checkpoint.get("model_state_dict", checkpoint)
    if any(key.startswith("single_step_model.") for key in state.keys()):
        return {key.replace("single_step_model.", "", 1): value for key, value in state.items() if key.startswith("single_step_model.")}
    return state


def _load_stage1_weights_into_temporal_model(model: DisentangledTemporalWindForecaster, checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Stage1 checkpoint not found: {checkpoint_path}. Stage2 is not allowed to start without it.")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    stage1_state = _extract_stage1_state_dict(checkpoint)
    missing, unexpected = model.single_step_model.load_state_dict(stage1_state, strict=False)
    allowed_inactive_missing: Tuple[str, ...] = ()
    if model.decoder.decoder_mode != "concat_only":
        allowed_inactive_missing += ("decoder.concat_decoder.",)
    critical_missing = [
        key for key in missing
        if (key.startswith("encoder.") or key.startswith("decoder."))
        and not key.startswith(allowed_inactive_missing)
    ]
    if critical_missing:
        raise RuntimeError(f"Missing critical stage1 weights: {critical_missing}")
    if unexpected:
        print(f"stage1 load note: ignored unexpected keys: {unexpected}")
    print(f"loaded stage1 weights from: {checkpoint_path}")
    return checkpoint


def _validate_no_grads(module: nn.Module, module_name: str) -> None:
    for param_name, param in module.named_parameters():
        if param.grad is not None and float(param.grad.detach().abs().sum().item()) > 0.0:
            raise RuntimeError(f"Gradient leakage detected in {module_name}.{param_name}")


def _update_history_latent(history_latent: Dict[str, torch.Tensor], next_latent: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "z_peak": torch.cat([history_latent["z_peak"][:, 1:], next_latent["z_peak"].unsqueeze(1)], dim=1),
        "z_width": torch.cat([history_latent["z_width"][:, 1:], next_latent["z_width"].unsqueeze(1)], dim=1),
        "z_noise": torch.cat([history_latent["z_noise"][:, 1:], next_latent["z_noise"].unsqueeze(1)], dim=1),
        "peak_pos": torch.cat([history_latent["peak_pos"][:, 1:], next_latent["peak_pos"].unsqueeze(1)], dim=1),
    }


def _update_wind_history(wind_history: torch.Tensor, next_wind: torch.Tensor) -> torch.Tensor:
    return torch.cat([wind_history[:, 1:], next_wind.unsqueeze(1)], dim=1)


def _select_next_inputs_for_roll(strategy: str, step_idx: int, predicted_latent: Dict[str, torch.Tensor], predicted_wind: torch.Tensor, teacher_latent: Optional[Dict[str, torch.Tensor]], future_wind: Optional[torch.Tensor], scheduled_sampling_prob: float) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    if teacher_latent is None or future_wind is None or strategy == "free_running":
        return predicted_latent, predicted_wind
    teacher_step_latent = {"z_peak": teacher_latent["z_peak"][:, step_idx], "z_width": teacher_latent["z_width"][:, step_idx], "z_noise": teacher_latent["z_noise"][:, step_idx], "peak_pos": teacher_latent["peak_pos"][:, step_idx]}
    teacher_step_wind = future_wind[:, step_idx]
    if strategy == "full_teacher_forcing":
        return teacher_step_latent, teacher_step_wind
    if strategy != "scheduled_sampling":
        raise ValueError(f"Unsupported training input strategy: {strategy}")
    gate_mask = torch.rand_like(predicted_wind) < scheduled_sampling_prob
    mixed_latent = {
        "z_peak": torch.where(gate_mask.unsqueeze(-1), teacher_step_latent["z_peak"], predicted_latent["z_peak"]),
        "z_width": torch.where(gate_mask.unsqueeze(-1), teacher_step_latent["z_width"], predicted_latent["z_width"]),
        "z_noise": torch.where(gate_mask.unsqueeze(-1), teacher_step_latent["z_noise"], predicted_latent["z_noise"]),
        "peak_pos": torch.where(gate_mask, teacher_step_latent["peak_pos"], predicted_latent["peak_pos"]),
    }
    mixed_wind = torch.where(gate_mask, teacher_step_wind, predicted_wind)
    return mixed_latent, mixed_wind


def _encode_history_for_stage2(model: DisentangledTemporalWindForecaster, spectrum_history: torch.Tensor, freeze_stage1: bool) -> Dict[str, torch.Tensor]:
    if freeze_stage1:
        with torch.no_grad():
            model.single_step_model.eval()
            return model.encode_history(spectrum_history)
    model.single_step_model.train(model.training)
    return model.encode_history(spectrum_history)


def _encode_future_sequence_frozen(model: DisentangledTemporalWindForecaster, future_spectrum: Optional[torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
    if future_spectrum is None:
        return None
    with torch.no_grad():
        model.single_step_model.eval()
        return model.encode_history(future_spectrum)


def _slice_latent_step(latent_sequence: Dict[str, torch.Tensor], step_idx: int) -> Dict[str, torch.Tensor]:
    """Select one future latent step from tensors shaped [B, T, ...]."""
    return {key: value[:, step_idx] for key, value in latent_sequence.items()}


def _configure_stage2_modules(model: DisentangledTemporalWindForecaster, phase: str, train_mode: bool, freeze_stage1: bool = True) -> None:
    if freeze_stage1:
        model.single_step_model.eval()
        _set_module_requires_grad(model.single_step_model, False)
    else:
        model.single_step_model.train(train_mode)
        _set_module_requires_grad(model.single_step_model, train_mode)
    if phase == "front":
        model.direct_transition.train(train_mode)
        model.latent_transition.eval()
        _set_module_requires_grad(model.direct_transition, train_mode)
        _set_module_requires_grad(model.latent_transition, False)
    elif phase == "back":
        model.direct_transition.eval()
        model.latent_transition.train(train_mode)
        _set_module_requires_grad(model.direct_transition, False)
        _set_module_requires_grad(model.latent_transition, train_mode)
        if freeze_stage1:
            _set_module_requires_grad(model.decoder, False)
    elif phase == "joint":
        model.direct_transition.train(train_mode)
        model.latent_transition.train(train_mode)
        _set_module_requires_grad(model.direct_transition, train_mode)
        _set_module_requires_grad(model.latent_transition, train_mode)
    else:
        raise ValueError(f"Unsupported stage2 phase: {phase}")


def _build_stage1_model_from_dataset(dataset: SingleStepWindDataset, d_model: int, n_heads: int, decoder_mode: str = "physics", latent_layout: str = "structured", lambda_cons: float = 0.5, mi_weight: float = 0.1) -> DisentangledWindEstimator:
    sample = dataset[0]
    n_gate, n_freq = sample["spectrum"].shape
    return DisentangledWindEstimator(n_freq=n_freq, n_gate=n_gate, d_model=d_model, n_heads=n_heads, decoder_mode=decoder_mode, latent_layout=latent_layout, lambda_cons=lambda_cons, mi_weight=mi_weight)


def run_stage1_epoch(model: DisentangledWindEstimator, loader: DataLoader, optimizer: Optional[torch.optim.Optimizer], device: torch.device, max_batches: Optional[int] = None, mi_weight: Optional[float] = None) -> Dict[str, float]:
    is_train = optimizer is not None
    if mi_weight is not None and abs(float(mi_weight) - float(model.mi_weight)) > 1e-12:
        raise ValueError("run_stage1_epoch mi_weight must match model.mi_weight; build a separate ablation model instead.")
    model.train(is_train)
    total_loss = 0.0
    total_abs = 0.0
    total_sq = 0.0
    total_elements = 0
    total_batches = 0
    total_mi = 0.0
    for batch_idx, batch in enumerate(tqdm(loader, desc="stage1-train" if is_train else "stage1-val", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        spectrum = batch["spectrum"].to(device)
        wind = batch["wind"].to(device)
        with torch.set_grad_enabled(is_train):
            output = model(spectrum, wind_speed=wind)
            loss = output["losses"]["total_loss"]
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        error = output["v_estimate"] - wind
        total_loss += float(loss.item())
        total_abs += float(error.abs().sum().item())
        total_sq += float((error ** 2).sum().item())
        total_elements += int(wind.numel())
        total_mi += float(output["losses"]["mi_loss"].item())
        total_batches += 1
    if total_batches == 0:
        return {}
    return {"L_stage1": total_loss / total_batches, "mae": total_abs / total_elements, "rmse": (total_sq / total_elements) ** 0.5, "mi_loss": total_mi / total_batches}


def train_stage1(train_npz: str, val_npz: str, device: torch.device, d_model: int, n_heads: int, epochs: int, batch_size: int, lr: float, checkpoint_paths: Dict[str, Path], max_train_batches: Optional[int], max_val_batches: Optional[int], mi_weight: Optional[float] = None, decoder_mode: str = "physics", latent_layout: str = "structured", lambda_cons: float = 0.5, experiment_name: str = "full_model", seed: Optional[int] = None) -> Tuple[DisentangledWindEstimator, Dict[str, float]]:
    train_dataset = SingleStepWindDataset(train_npz, expected_split="train")
    val_dataset = SingleStepWindDataset(val_npz, expected_split="val")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    effective_mi_weight = 0.1 if mi_weight is None else float(mi_weight)
    model = _build_stage1_model_from_dataset(train_dataset, d_model=d_model, n_heads=n_heads, decoder_mode=decoder_mode, latent_layout=latent_layout, lambda_cons=lambda_cons, mi_weight=effective_mi_weight).to(device)
    stage1_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = _build_optimizer(stage1_params, lr=lr, name="stage1_optimizer")
    _assert_optimizer_matches_parameters(optimizer, stage1_params, "stage1_optimizer")
    report_modules = {"stage1_model": model, "encoder": model.encoder, "decoder": model.decoder, "spectrum_reconstructor": model.spectrum_reconstructor}
    if model.disentangle_loss is not None:
        report_modules["disentangle_loss"] = model.disentangle_loss
    _print_module_freeze_report("Stage1 Parameter Freeze Report", report_modules)
    best_val = float("inf")
    best_metrics: Dict[str, float] = {}
    stage1_log_rows: List[Dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = run_stage1_epoch(model, train_loader, optimizer, device, max_batches=max_train_batches, mi_weight=mi_weight)
        val_metrics = run_stage1_epoch(model, val_loader, None, device, max_batches=max_val_batches, mi_weight=mi_weight)
        stage1_log_rows.append({"epoch": epoch, "train_L_stage1": train_metrics.get("L_stage1", 0.0), "val_L_stage1": val_metrics.get("L_stage1", 0.0), "train_mae": train_metrics.get("mae", 0.0), "val_mae": val_metrics.get("mae", 0.0), "train_rmse": train_metrics.get("rmse", 0.0), "val_rmse": val_metrics.get("rmse", 0.0)})
        print(f"stage1 epoch {epoch:03d} | train L_stage1={train_metrics.get('L_stage1', 0.0):.6f} val L_stage1={val_metrics.get('L_stage1', 0.0):.6f}")
        last_payload = {"phase": "stage1", "model_state_dict": model.state_dict(), "stage1_model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "best_val_L_stage1": best_val, "config": {"experiment_name": experiment_name, "seed": seed, "stage1_epochs": epochs, "optimizer": "Adam", "lr_scheduler": "constant", "d_model": d_model, "n_heads": n_heads, "decoder_mode": decoder_mode, "latent_layout": latent_layout, "lambda_cons": lambda_cons, "mi_weight": effective_mi_weight, "ablation_code_version": ABLATION_CODE_VERSION}}
        _save_checkpoint(checkpoint_paths["stage1_last"], last_payload, announce=False)
        current_val = val_metrics.get("L_stage1", float("inf"))
        if current_val < best_val:
            best_val = current_val
            best_metrics = dict(val_metrics)
            best_payload = dict(last_payload)
            best_payload["best_val_L_stage1"] = best_val
            best_payload["best_metrics"] = best_metrics
            _save_checkpoint(checkpoint_paths["stage1_best"], best_payload, announce=True)
    _write_csv_rows(checkpoint_paths["stage1_log"], stage1_log_rows)
    best_checkpoint = torch.load(checkpoint_paths["stage1_best"], map_location=device)
    model.load_state_dict(best_checkpoint["stage1_model_state_dict"])
    return model, best_metrics

def _empty_stage2_metrics() -> Dict[str, float]:
    return {
        "L_stage2_front": 0.0,
        "L_stage2_back": 0.0,
        "L_direct": 0.0,
        "L_decode_gt": 0.0,
        "L_consistency": 0.0,
        "L_state": 0.0,
        "L_peak": 0.0,
        "L_width": 0.0,
        "L_noise": 0.0,
        "L_pos": 0.0,
        "front_mae": 0.0,
        "front_rmse": 0.0,
        "decode_gt_mae": 0.0,
        "decode_gt_rmse": 0.0,
        "back_mae": 0.0,
        "back_rmse": 0.0,
        "window_count": 0.0,
    }


def _accumulate_stage2_metrics(
    metrics: Dict[str, float],
    front_loss: float,
    back_loss: float,
    future_wind: torch.Tensor,
    direct_pred: torch.Tensor,
    decode_pred: torch.Tensor,
    component_losses: Optional[Dict[str, float]] = None,
) -> None:
    front_error = direct_pred - future_wind
    back_error = decode_pred - direct_pred.detach()
    decode_gt_error = decode_pred - future_wind
    metrics["L_stage2_front"] += float(front_loss)
    metrics["L_stage2_back"] += float(back_loss)
    metrics["front_mae"] += float(front_error.abs().mean().item())
    metrics["front_rmse"] += float(torch.sqrt((front_error ** 2).mean()).item())
    metrics["back_mae"] += float(back_error.abs().mean().item())
    metrics["back_rmse"] += float(torch.sqrt((back_error ** 2).mean()).item())
    metrics["decode_gt_mae"] += float(decode_gt_error.abs().mean().item())
    metrics["decode_gt_rmse"] += float(torch.sqrt((decode_gt_error ** 2).mean()).item())
    if component_losses:
        for key, value in component_losses.items():
            if key in metrics:
                metrics[key] += float(value)
    metrics["window_count"] += 1.0


def _finalize_stage2_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    if metrics["window_count"] == 0:
        return {}
    count = metrics["window_count"]
    return {key: (value if key == "window_count" else value / count) for key, value in metrics.items()}


def _save_stage2_resume_checkpoint(checkpoint_path: Path, model: DisentangledTemporalWindForecaster, stage2_front_optimizer: torch.optim.Optimizer, stage2_back_optimizer: torch.optim.Optimizer, epoch: int, next_batch_idx: int, next_window_idx: int, next_phase: str, config: Dict[str, Any], stage1_optimizer: Optional[torch.optim.Optimizer] = None) -> None:
    payload = {"phase": "stage2", "resume_phase": next_phase, "epoch": epoch, "next_batch_idx": next_batch_idx, "next_window_idx": next_window_idx, "model_state_dict": model.state_dict(), "stage2_front_optimizer_state_dict": stage2_front_optimizer.state_dict(), "stage2_back_optimizer_state_dict": stage2_back_optimizer.state_dict(), "config": config}
    if stage1_optimizer is not None:
        payload["stage1_optimizer_state_dict"] = stage1_optimizer.state_dict()
    _save_checkpoint(checkpoint_path, payload, announce=False)


def _default_resume_state() -> Dict[str, int | str]:
    return {"epoch": 1, "next_batch_idx": 0, "next_window_idx": 0, "resume_phase": "front"}


def _load_stage2_resume(resume_path: Optional[str], model: DisentangledTemporalWindForecaster, stage2_front_optimizer: torch.optim.Optimizer, stage2_back_optimizer: torch.optim.Optimizer, device: torch.device, stage1_optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, int | str]:
    state = _default_resume_state()
    if not resume_path:
        return state
    checkpoint = torch.load(resume_path, map_location=device)
    if checkpoint.get("phase") != "stage2":
        raise ValueError(f"resume checkpoint must be stage2: {resume_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    if "stage2_front_optimizer_state_dict" in checkpoint:
        stage2_front_optimizer.load_state_dict(checkpoint["stage2_front_optimizer_state_dict"])
    if "stage2_back_optimizer_state_dict" in checkpoint:
        stage2_back_optimizer.load_state_dict(checkpoint["stage2_back_optimizer_state_dict"])
    if stage1_optimizer is not None and "stage1_optimizer_state_dict" in checkpoint:
        stage1_optimizer.load_state_dict(checkpoint["stage1_optimizer_state_dict"])
    state["epoch"] = int(checkpoint.get("epoch", 1))
    state["next_batch_idx"] = int(checkpoint.get("next_batch_idx", 0))
    state["next_window_idx"] = int(checkpoint.get("next_window_idx", 0))
    state["resume_phase"] = str(checkpoint.get("resume_phase", "front"))
    print(f"loaded stage2 resume state from: {resume_path}")
    return state


def run_stage2_epoch(model: DisentangledTemporalWindForecaster, loader: DataLoader, device: torch.device, epoch: int, stage_name: str, checkpoint_paths: Dict[str, Path], checkpoint_config: Dict[str, Any], stage2_front_optimizer: Optional[torch.optim.Optimizer], stage2_back_optimizer: Optional[torch.optim.Optimizer], training_input_strategy: str, scheduled_sampling_prob: float, window_log_rows: List[Dict[str, object]], resume_state: Optional[Dict[str, int | str]] = None, max_batches: Optional[int] = None, verify_checks: bool = False, stage2_training_mode: str = "front_back", freeze_stage1: bool = True, stage1_optimizer: Optional[torch.optim.Optimizer] = None, consistency_weight: float = 0.0) -> Dict[str, float]:
    is_train = stage2_front_optimizer is not None and stage2_back_optimizer is not None
    supported_modes = {"front_back", "original_dual_substep", "joint", "joint_two_branch", "joint_two_substeps"}
    if stage2_training_mode not in supported_modes:
        raise ValueError(f"Unsupported Stage-2 training mode: {stage2_training_mode}")
    if stage2_training_mode == "joint_two_substeps" and not freeze_stage1 and is_train:
        raise ValueError(
            "joint_two_substeps requires freeze_stage1=true: otherwise shared Stage-1 "
            "parameters would be stepped once in each sub-step. Use joint_two_branch "
            "for a single controlled update of shared parameters."
        )
    metrics = _empty_stage2_metrics()
    start_batch_idx = int(resume_state.get("next_batch_idx", 0)) if resume_state else 0
    resume_window_idx = int(resume_state.get("next_window_idx", 0)) if resume_state else 0
    resume_phase = str(resume_state.get("resume_phase", "front")) if resume_state else "front"
    sequence_offset = 0
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"{stage_name}-stage2-epoch-{epoch}", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if batch_idx < start_batch_idx:
            continue
        spectrum_history = batch["spectrum_history"].to(device)
        past_wind = batch["past_wind"].to(device)
        future_spectrum = batch["future_spectrum"].to(device)
        future_wind = batch["future_wind"].to(device)
        history_latent = _encode_history_for_stage2(model, spectrum_history, freeze_stage1=freeze_stage1)
        teacher_latent = _encode_future_sequence_frozen(model, future_spectrum)
        wind_history = past_wind.detach()
        horizon = int(future_wind.size(1))
        window_start = resume_window_idx if batch_idx == start_batch_idx else 0
        phase_to_start = resume_phase if batch_idx == start_batch_idx else "front"
        for window_idx in range(window_start, horizon):
            global_window_idx = sequence_offset + window_idx
            target_future_wind = future_wind[:, window_idx]
            if stage2_training_mode == "joint_two_branch":
                _configure_stage2_modules(model, "joint", train_mode=is_train, freeze_stage1=freeze_stage1)
                direct_pred = model.direct_transition(history_latent, wind_history)
                direct_reference = direct_pred.detach()
                latent_new = model.latent_transition(history_latent, direct_reference)
                decoded = model.decoder(latent_new["z_peak"], latent_new["z_width"], latent_new["z_noise"], latent_new["peak_pos"])
                decode_pred = decoded["v_estimate"]
                direct_loss_tensor = F.mse_loss(direct_pred, target_future_wind)
                decode_gt_loss_tensor = F.mse_loss(decode_pred, target_future_wind)
                consistency_loss_tensor = F.mse_loss(decode_pred, direct_reference)
                teacher_step_latent = _slice_latent_step(teacher_latent, window_idx) if teacher_latent is not None else None
                state_losses = _structured_state_losses(model, latent_new, teacher_step_latent, device)
                back_loss_tensor = (
                    model.loss_weights["lambda_decode"] * decode_gt_loss_tensor
                    + model.loss_weights["lambda_state"] * state_losses["total"]
                    + float(consistency_weight) * consistency_loss_tensor
                )
                joint_loss_tensor = model.loss_weights["lambda_direct"] * direct_loss_tensor + back_loss_tensor
                if is_train:
                    stage2_front_optimizer.zero_grad(set_to_none=True)
                    stage2_back_optimizer.zero_grad(set_to_none=True)
                    if stage1_optimizer is not None:
                        stage1_optimizer.zero_grad(set_to_none=True)
                    joint_loss_tensor.backward()
                    stage2_front_optimizer.step()
                    stage2_back_optimizer.step()
                    if stage1_optimizer is not None:
                        stage1_optimizer.step()
                    _save_stage2_resume_checkpoint(checkpoint_paths["stage2_last"], model, stage2_front_optimizer, stage2_back_optimizer, epoch=epoch, next_batch_idx=batch_idx, next_window_idx=window_idx + 1, next_phase="front", config=checkpoint_config, stage1_optimizer=stage1_optimizer)
                components = {
                    "L_direct": float(direct_loss_tensor.detach().item()),
                    "L_decode_gt": float(decode_gt_loss_tensor.detach().item()),
                    "L_consistency": float(consistency_loss_tensor.detach().item()),
                    "L_state": float(state_losses["total"].detach().item()),
                    "L_peak": float(state_losses["peak"].detach().item()),
                    "L_width": float(state_losses["width"].detach().item()),
                    "L_noise": float(state_losses["noise"].detach().item()),
                    "L_pos": float(state_losses["pos"].detach().item()),
                }
                window_log_rows.append({"epoch": epoch, "split": stage_name, "batch_idx": batch_idx, "window_idx": window_idx, "global_window_idx": global_window_idx, "substep": "joint_two_branch", "loss": float(joint_loss_tensor.detach().item()), "consistency_weight": float(consistency_weight), **components})
                predicted_roll_latent = {"z_peak": latent_new["z_peak"].detach(), "z_width": latent_new["z_width"].detach(), "z_noise": latent_new["z_noise"].detach(), "peak_pos": latent_new["peak_pos"].detach()}
                next_latent, next_wind = _select_next_inputs_for_roll(strategy=training_input_strategy if is_train else "free_running", step_idx=window_idx, predicted_latent=predicted_roll_latent, predicted_wind=direct_reference, teacher_latent=teacher_latent, future_wind=future_wind, scheduled_sampling_prob=scheduled_sampling_prob if is_train else 0.0)
                history_latent = _update_history_latent(history_latent, next_latent)
                wind_history = _update_wind_history(wind_history, next_wind.detach())
                _accumulate_stage2_metrics(metrics, float(direct_loss_tensor.detach().item()), float(back_loss_tensor.detach().item()), target_future_wind, direct_reference, decode_pred.detach(), components)
                phase_to_start = "front"
                continue

            if stage2_training_mode == "joint_two_substeps":
                _configure_stage2_modules(model, "front", train_mode=is_train, freeze_stage1=freeze_stage1)
                direct_pred = model.direct_transition(history_latent, wind_history)
                direct_loss_tensor = F.mse_loss(direct_pred, target_future_wind)
                if is_train:
                    stage2_front_optimizer.zero_grad(set_to_none=True)
                    stage2_back_optimizer.zero_grad(set_to_none=True)
                    if stage1_optimizer is not None:
                        stage1_optimizer.zero_grad(set_to_none=True)
                    (model.loss_weights["lambda_direct"] * direct_loss_tensor).backward()
                    stage2_front_optimizer.step()
                    if stage1_optimizer is not None:
                        stage1_optimizer.step()
                direct_reference = direct_pred.detach()
                if is_train and not freeze_stage1:
                    history_latent = _encode_history_for_stage2(model, spectrum_history, freeze_stage1=False)
                _configure_stage2_modules(model, "back", train_mode=is_train, freeze_stage1=freeze_stage1)
                latent_new = model.latent_transition(history_latent, direct_reference)
                decoded = model.decoder(latent_new["z_peak"], latent_new["z_width"], latent_new["z_noise"], latent_new["peak_pos"])
                decode_pred = decoded["v_estimate"]
                decode_gt_loss_tensor = F.mse_loss(decode_pred, target_future_wind)
                consistency_loss_tensor = F.mse_loss(decode_pred, direct_reference)
                teacher_step_latent = _slice_latent_step(teacher_latent, window_idx) if teacher_latent is not None else None
                state_losses = _structured_state_losses(model, latent_new, teacher_step_latent, device)
                back_loss_tensor = (
                    model.loss_weights["lambda_decode"] * decode_gt_loss_tensor
                    + model.loss_weights["lambda_state"] * state_losses["total"]
                    + float(consistency_weight) * consistency_loss_tensor
                )
                if is_train:
                    stage2_front_optimizer.zero_grad(set_to_none=True)
                    stage2_back_optimizer.zero_grad(set_to_none=True)
                    if stage1_optimizer is not None:
                        stage1_optimizer.zero_grad(set_to_none=True)
                    back_loss_tensor.backward()
                    _validate_no_grads(model.direct_transition, "direct_transition")
                    stage2_back_optimizer.step()
                    if stage1_optimizer is not None:
                        stage1_optimizer.step()
                    _save_stage2_resume_checkpoint(checkpoint_paths["stage2_last"], model, stage2_front_optimizer, stage2_back_optimizer, epoch=epoch, next_batch_idx=batch_idx, next_window_idx=window_idx + 1, next_phase="front", config=checkpoint_config, stage1_optimizer=stage1_optimizer)
                components = {
                    "L_direct": float(direct_loss_tensor.detach().item()),
                    "L_decode_gt": float(decode_gt_loss_tensor.detach().item()),
                    "L_consistency": float(consistency_loss_tensor.detach().item()),
                    "L_state": float(state_losses["total"].detach().item()),
                    "L_peak": float(state_losses["peak"].detach().item()),
                    "L_width": float(state_losses["width"].detach().item()),
                    "L_noise": float(state_losses["noise"].detach().item()),
                    "L_pos": float(state_losses["pos"].detach().item()),
                }
                window_log_rows.append({"epoch": epoch, "split": stage_name, "batch_idx": batch_idx, "window_idx": window_idx, "global_window_idx": global_window_idx, "substep": "joint_two_substeps", "loss": float((model.loss_weights["lambda_direct"] * direct_loss_tensor.detach() + back_loss_tensor.detach()).item()), "consistency_weight": float(consistency_weight), **components})
                predicted_roll_latent = {"z_peak": latent_new["z_peak"].detach(), "z_width": latent_new["z_width"].detach(), "z_noise": latent_new["z_noise"].detach(), "peak_pos": latent_new["peak_pos"].detach()}
                next_latent, next_wind = _select_next_inputs_for_roll(strategy=training_input_strategy if is_train else "free_running", step_idx=window_idx, predicted_latent=predicted_roll_latent, predicted_wind=direct_reference, teacher_latent=teacher_latent, future_wind=future_wind, scheduled_sampling_prob=scheduled_sampling_prob if is_train else 0.0)
                history_latent = _update_history_latent(history_latent, next_latent)
                wind_history = _update_wind_history(wind_history, next_wind.detach())
                _accumulate_stage2_metrics(metrics, float(direct_loss_tensor.detach().item()), float(back_loss_tensor.detach().item()), target_future_wind, direct_reference, decode_pred.detach(), components)
                phase_to_start = "front"
                continue

            if stage2_training_mode == "joint":
                _configure_stage2_modules(model, "joint", train_mode=is_train, freeze_stage1=freeze_stage1)
                direct_pred = model.direct_transition(history_latent, wind_history)
                latent_new = model.latent_transition(history_latent, direct_pred)
                decoded = model.decoder(latent_new["z_peak"], latent_new["z_width"], latent_new["z_noise"], latent_new["peak_pos"])
                decode_pred = decoded["v_estimate"]
                front_loss_tensor = F.mse_loss(direct_pred, target_future_wind)
                decode_loss_tensor = F.mse_loss(decode_pred, target_future_wind)
                align_loss_tensor = F.mse_loss(decode_pred, direct_pred)
                phys_loss_tensor = F.mse_loss(decode_pred, decoded["v_physics"])
                state_loss_tensor = torch.zeros((), device=device)
                if teacher_latent is not None:
                    teacher_step_latent = _slice_latent_step(teacher_latent, window_idx)
                    state_loss_tensor = (
                        model.loss_weights["lambda_state_peak"] * F.mse_loss(latent_new["z_peak"], teacher_step_latent["z_peak"])
                        + model.loss_weights["lambda_state_width"] * F.mse_loss(latent_new["z_width"], teacher_step_latent["z_width"])
                        + model.loss_weights["lambda_state_noise"] * F.mse_loss(latent_new["z_noise"], teacher_step_latent["z_noise"])
                        + model.loss_weights["lambda_state_pos"] * F.mse_loss(latent_new["peak_pos"], teacher_step_latent["peak_pos"])
                    )
                back_loss_tensor = (
                    model.loss_weights["lambda_decode"] * decode_loss_tensor
                    + model.loss_weights["lambda_align"] * align_loss_tensor
                    + model.loss_weights["lambda_phys"] * phys_loss_tensor
                    + model.loss_weights["lambda_state"] * state_loss_tensor
                )
                joint_loss_tensor = front_loss_tensor + back_loss_tensor
                if is_train:
                    stage2_front_optimizer.zero_grad(set_to_none=True)
                    stage2_back_optimizer.zero_grad(set_to_none=True)
                    if stage1_optimizer is not None:
                        stage1_optimizer.zero_grad(set_to_none=True)
                    joint_loss_tensor.backward()
                    stage2_front_optimizer.step()
                    stage2_back_optimizer.step()
                    if stage1_optimizer is not None:
                        stage1_optimizer.step()
                    _save_stage2_resume_checkpoint(checkpoint_paths["stage2_last"], model, stage2_front_optimizer, stage2_back_optimizer, epoch=epoch, next_batch_idx=batch_idx, next_window_idx=window_idx + 1, next_phase="front", config=checkpoint_config, stage1_optimizer=stage1_optimizer)
                window_log_rows.append({"epoch": epoch, "split": stage_name, "batch_idx": batch_idx, "window_idx": window_idx, "global_window_idx": global_window_idx, "substep": "joint", "loss": float(joint_loss_tensor.item())})
                predicted_roll_latent = {"z_peak": latent_new["z_peak"].detach(), "z_width": latent_new["z_width"].detach(), "z_noise": latent_new["z_noise"].detach(), "peak_pos": latent_new["peak_pos"].detach()}
                next_latent, next_wind = _select_next_inputs_for_roll(strategy=training_input_strategy if is_train else "free_running", step_idx=window_idx, predicted_latent=predicted_roll_latent, predicted_wind=direct_pred.detach(), teacher_latent=teacher_latent, future_wind=future_wind, scheduled_sampling_prob=scheduled_sampling_prob if is_train else 0.0)
                history_latent = _update_history_latent(history_latent, next_latent)
                wind_history = _update_wind_history(wind_history, next_wind.detach())
                _accumulate_stage2_metrics(metrics=metrics, front_loss=float(front_loss_tensor.item()), back_loss=float(back_loss_tensor.item()), future_wind=target_future_wind, direct_pred=direct_pred.detach(), decode_pred=decode_pred.detach())
                phase_to_start = "front"
                continue
            if phase_to_start == "front":
                _configure_stage2_modules(model, "front", train_mode=is_train, freeze_stage1=freeze_stage1)
                direct_pred = model.direct_transition(history_latent, wind_history)
                front_loss_tensor = F.mse_loss(direct_pred, target_future_wind)
                if is_train:
                    stage2_back_optimizer.zero_grad(set_to_none=True)
                    stage2_front_optimizer.zero_grad(set_to_none=True)
                    if stage1_optimizer is not None:
                        stage1_optimizer.zero_grad(set_to_none=True)
                    front_loss_tensor.backward()
                    stage2_front_optimizer.step()
                    if stage1_optimizer is not None:
                        stage1_optimizer.step()
                    _save_stage2_resume_checkpoint(checkpoint_paths["stage2_last"], model, stage2_front_optimizer, stage2_back_optimizer, epoch=epoch, next_batch_idx=batch_idx, next_window_idx=window_idx, next_phase="back", config=checkpoint_config, stage1_optimizer=stage1_optimizer)
                window_log_rows.append({"epoch": epoch, "split": stage_name, "batch_idx": batch_idx, "window_idx": window_idx, "global_window_idx": global_window_idx, "substep": "front", "loss": float(front_loss_tensor.item())})
                if is_train and not freeze_stage1:
                    history_latent = _encode_history_for_stage2(model, spectrum_history, freeze_stage1=False)
                    direct_pred = model.direct_transition(history_latent, wind_history)
                    front_loss_tensor = F.mse_loss(direct_pred, target_future_wind)
                phase_to_start = "back"
            else:
                direct_pred = model.direct_transition(history_latent, wind_history)
                front_loss_tensor = F.mse_loss(direct_pred, target_future_wind)
            direct_supervision = direct_pred.detach()
            decoder_hash_before = _fingerprint_module(model.decoder) if verify_checks else ""
            direct_hash_before = _fingerprint_module(model.direct_transition) if verify_checks else ""
            _configure_stage2_modules(model, "back", train_mode=is_train, freeze_stage1=freeze_stage1)
            latent_new = model.latent_transition(history_latent, direct_supervision)
            decoded = model.decoder(latent_new["z_peak"], latent_new["z_width"], latent_new["z_noise"], latent_new["peak_pos"])
            decode_pred = decoded["v_estimate"]
            decode_loss_tensor = F.mse_loss(decode_pred, direct_supervision)
            phys_loss_tensor = F.mse_loss(decode_pred, decoded["v_physics"])
            state_loss_tensor = torch.zeros((), device=device)
            if teacher_latent is not None:
                teacher_step_latent = _slice_latent_step(teacher_latent, window_idx)
                state_loss_tensor = (
                    model.loss_weights["lambda_state_peak"] * F.mse_loss(latent_new["z_peak"], teacher_step_latent["z_peak"])
                    + model.loss_weights["lambda_state_width"] * F.mse_loss(latent_new["z_width"], teacher_step_latent["z_width"])
                    + model.loss_weights["lambda_state_noise"] * F.mse_loss(latent_new["z_noise"], teacher_step_latent["z_noise"])
                    + model.loss_weights["lambda_state_pos"] * F.mse_loss(latent_new["peak_pos"], teacher_step_latent["peak_pos"])
                )
            back_loss_tensor = (
                model.loss_weights["lambda_decode"] * decode_loss_tensor
                + model.loss_weights["lambda_phys"] * phys_loss_tensor
                + model.loss_weights["lambda_state"] * state_loss_tensor
            )
            if is_train:
                stage2_front_optimizer.zero_grad(set_to_none=True)
                stage2_back_optimizer.zero_grad(set_to_none=True)
                if stage1_optimizer is not None:
                    stage1_optimizer.zero_grad(set_to_none=True)
                back_loss_tensor.backward()
                _validate_no_grads(model.direct_transition, "direct_transition")
                if freeze_stage1:
                    _validate_no_grads(model.decoder, "decoder")
                    _validate_no_grads(model.single_step_model, "single_step_model")
                stage2_back_optimizer.step()
                if stage1_optimizer is not None:
                    stage1_optimizer.step()
                _save_stage2_resume_checkpoint(checkpoint_paths["stage2_last"], model, stage2_front_optimizer, stage2_back_optimizer, epoch=epoch, next_batch_idx=batch_idx, next_window_idx=window_idx + 1, next_phase="front", config=checkpoint_config, stage1_optimizer=stage1_optimizer)
            if verify_checks and freeze_stage1:
                if _fingerprint_module(model.decoder) != decoder_hash_before:
                    raise RuntimeError("decoder changed during stage2 back training")
                if _fingerprint_module(model.direct_transition) != direct_hash_before:
                    raise RuntimeError("direct_transition changed during stage2 back training")
            window_log_rows.append({"epoch": epoch, "split": stage_name, "batch_idx": batch_idx, "window_idx": window_idx, "global_window_idx": global_window_idx, "substep": "back", "loss": float(back_loss_tensor.item()), "decode_loss": float(decode_loss_tensor.detach().item()), "physics_loss": float(phys_loss_tensor.detach().item()), "state_loss": float(state_loss_tensor.detach().item())})
            predicted_roll_latent = {"z_peak": latent_new["z_peak"].detach(), "z_width": latent_new["z_width"].detach(), "z_noise": latent_new["z_noise"].detach(), "peak_pos": latent_new["peak_pos"].detach()}
            next_latent, next_wind = _select_next_inputs_for_roll(strategy=training_input_strategy if is_train else "free_running", step_idx=window_idx, predicted_latent=predicted_roll_latent, predicted_wind=direct_supervision, teacher_latent=teacher_latent, future_wind=future_wind, scheduled_sampling_prob=scheduled_sampling_prob if is_train else 0.0)
            history_latent = _update_history_latent(history_latent, next_latent)
            wind_history = _update_wind_history(wind_history, next_wind.detach())
            window_log_rows.append({"epoch": epoch, "split": stage_name, "batch_idx": batch_idx, "window_idx": window_idx, "global_window_idx": global_window_idx, "substep": "roll", "loss": ""})
            _accumulate_stage2_metrics(metrics=metrics, front_loss=float(front_loss_tensor.item()), back_loss=float(back_loss_tensor.item()), future_wind=target_future_wind, direct_pred=direct_supervision, decode_pred=decode_pred.detach())
            phase_to_start = "front"
        sequence_offset += horizon
    return _finalize_stage2_metrics(metrics)

def _stage2_selection_value(metrics: Dict[str, float], criterion: str) -> float:
    front = metrics.get("L_stage2_front", float("inf"))
    back = metrics.get("L_stage2_back", float("inf"))
    if criterion == "direct_front":
        return front
    if criterion == "decode_gt":
        return metrics.get("L_decode_gt", float("inf"))
    if criterion == "back":
        return back
    if criterion == "combined":
        return front + back
    raise ValueError(f"Unsupported Stage 2 selection criterion: {criterion}")


def train_stage2(train_npz: str, val_npz: str, input_steps: int, output_steps: int, stride: int, device: torch.device, d_model: int, n_heads: int, temporal_hidden_dim: int, epochs: int, batch_size: int, lr: float, checkpoint_paths: Dict[str, Path], training_input_strategy: str, scheduled_sampling_prob: float, stage1_checkpoint_path: Path, resume_path: Optional[str], max_train_batches: Optional[int], max_val_batches: Optional[int], verify_checks: bool, lambda_direct: Optional[float] = None, lambda_decode: Optional[float] = None, lambda_state: Optional[float] = None, lambda_phys: Optional[float] = None, lambda_consistency: float = 0.0, lambda_state_peak: Optional[float] = None, lambda_state_width: Optional[float] = None, lambda_state_noise: Optional[float] = None, lambda_state_pos: Optional[float] = None, consistency_schedule_type: str = "constant", consistency_warmup_epochs: int = 0, decoder_mode: str = "physics", latent_layout: str = "structured", lambda_cons: float = 0.5, mi_weight: float = 0.1, stage2_training_mode: str = "front_back", freeze_stage1: bool = True, selection_criterion: str = "combined", experiment_name: str = "full_model", seed: Optional[int] = None) -> Tuple[DisentangledTemporalWindForecaster, Dict[str, float], Dict[str, Any]]:
    train_dataset = TemporalSequenceDataset(train_npz, input_steps, output_steps, stride, expected_split="train")
    val_dataset = TemporalSequenceDataset(val_npz, input_steps, output_steps, stride, expected_split="val")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = build_temporal_model_from_dataset(sample=train_dataset[0], d_model=d_model, n_heads=n_heads, temporal_hidden_dim=temporal_hidden_dim, decoder_mode=decoder_mode, latent_layout=latent_layout, lambda_cons=lambda_cons, mi_weight=mi_weight).to(device)
    if lambda_direct is not None:
        model.loss_weights["lambda_direct"] = lambda_direct
    if lambda_decode is not None:
        model.loss_weights["lambda_decode"] = lambda_decode
    if lambda_state is not None:
        model.loss_weights["lambda_state"] = lambda_state
    if lambda_phys is not None:
        model.loss_weights["lambda_phys"] = lambda_phys
    model.loss_weights["lambda_consistency"] = float(lambda_consistency)
    for key, value in {
        "lambda_state_peak": lambda_state_peak,
        "lambda_state_width": lambda_state_width,
        "lambda_state_noise": lambda_state_noise,
        "lambda_state_pos": lambda_state_pos,
    }.items():
        if value is not None:
            model.loss_weights[key] = float(value)
    if stage2_training_mode in {"joint_two_branch", "joint_two_substeps"} and float(lambda_consistency) > float(model.loss_weights["lambda_decode"]):
        raise ValueError("lambda_consistency must not exceed lambda_decode for the controlled joint modes.")
    stage1_checkpoint = _load_stage1_weights_into_temporal_model(model, stage1_checkpoint_path, device)
    _set_module_requires_grad(model.single_step_model, not freeze_stage1)
    model.single_step_model.train(not freeze_stage1)
    stage1_params = [param for param in model.single_step_model.parameters() if param.requires_grad]
    stage2_front_params = [param for param in model.direct_transition.parameters() if param.requires_grad]
    stage2_back_params = [param for param in model.latent_transition.parameters() if param.requires_grad]
    stage2_front_optimizer = _build_optimizer(stage2_front_params, lr=lr, name="stage2_front_optimizer")
    stage2_back_optimizer = _build_optimizer(stage2_back_params, lr=lr, name="stage2_back_optimizer")
    stage1_optimizer = _build_optimizer(stage1_params, lr=lr, name="stage1_optimizer") if stage1_params else None
    _assert_optimizer_matches_parameters(stage2_front_optimizer, stage2_front_params, "stage2_front_optimizer")
    _assert_optimizer_matches_parameters(stage2_back_optimizer, stage2_back_params, "stage2_back_optimizer")
    if stage1_optimizer is not None:
        _assert_optimizer_matches_parameters(stage1_optimizer, stage1_params, "stage1_optimizer")
    _assert_optimizers_disjoint({
        "stage2_front_optimizer": stage2_front_optimizer,
        "stage2_back_optimizer": stage2_back_optimizer,
        "stage1_optimizer": stage1_optimizer,
    })
    resume_state = _load_stage2_resume(resume_path, model, stage2_front_optimizer, stage2_back_optimizer, device, stage1_optimizer=stage1_optimizer)
    _print_module_freeze_report("Stage2 Parameter Freeze Report", {"single_step_model": model.single_step_model, "encoder": model.encoder, "decoder": model.decoder, "direct_transition": model.direct_transition, "latent_transition": model.latent_transition})
    checkpoint_config = {"experiment_name": experiment_name, "seed": seed, "d_model": d_model, "n_heads": n_heads, "temporal_hidden_dim": temporal_hidden_dim, "input_steps": input_steps, "output_steps": output_steps, "stride": stride, "stage2_epochs": epochs, "batch_size": batch_size, "optimizer": "Adam", "initial_lr": lr, "lr_scheduler": "constant", "early_stopping_patience": 0, "gradient_clip": None, "selection_criterion": selection_criterion, "training_input_strategy": training_input_strategy, "scheduled_sampling_prob": scheduled_sampling_prob, "stage1_checkpoint_path": str(stage1_checkpoint_path), "lambda_direct": model.loss_weights["lambda_direct"], "lambda_decode": model.loss_weights["lambda_decode"], "lambda_state": model.loss_weights["lambda_state"], "lambda_phys": model.loss_weights["lambda_phys"], "lambda_consistency": model.loss_weights["lambda_consistency"], "lambda_state_peak": model.loss_weights["lambda_state_peak"], "lambda_state_width": model.loss_weights["lambda_state_width"], "lambda_state_noise": model.loss_weights["lambda_state_noise"], "lambda_state_pos": model.loss_weights["lambda_state_pos"], "consistency_schedule": {"type": consistency_schedule_type, "warmup_epochs": consistency_warmup_epochs, "target_weight": model.loss_weights["lambda_consistency"]}, "decoder_mode": decoder_mode, "latent_layout": latent_layout, "lambda_cons": lambda_cons, "mi_weight": mi_weight, "stage2_training_mode": stage2_training_mode, "freeze_stage1": freeze_stage1, "ablation_code_version": ABLATION_CODE_VERSION}
    window_log_rows: List[Dict[str, object]] = []
    epoch_log_rows: List[Dict[str, object]] = []
    best_val = float("inf")
    best_metrics: Dict[str, float] = {}
    stage1_fingerprint_before = _fingerprint_module(model.single_step_model)
    for epoch in range(int(resume_state["epoch"]), epochs + 1):
        actual_consistency_weight = _consistency_weight(epoch, epochs, consistency_schedule_type, consistency_warmup_epochs, model.loss_weights["lambda_consistency"])
        train_resume = resume_state if epoch == int(resume_state["epoch"]) else _default_resume_state()
        train_metrics = run_stage2_epoch(model, train_loader, device, epoch, "train", checkpoint_paths, checkpoint_config, stage2_front_optimizer, stage2_back_optimizer, training_input_strategy, scheduled_sampling_prob, window_log_rows, train_resume, max_train_batches, verify_checks, stage2_training_mode=stage2_training_mode, freeze_stage1=freeze_stage1, stage1_optimizer=stage1_optimizer, consistency_weight=actual_consistency_weight)
        val_metrics = run_stage2_epoch(model, val_loader, device, epoch, "val", checkpoint_paths, checkpoint_config, None, None, "free_running", 0.0, window_log_rows, None, max_val_batches, False, stage2_training_mode=stage2_training_mode, freeze_stage1=freeze_stage1, consistency_weight=actual_consistency_weight)
        epoch_log_rows.append({"epoch": epoch, "consistency_weight": actual_consistency_weight, "train_L_stage2_front": train_metrics.get("L_stage2_front", 0.0), "train_L_stage2_back": train_metrics.get("L_stage2_back", 0.0), "train_L_direct": train_metrics.get("L_direct", 0.0), "train_L_decode_gt": train_metrics.get("L_decode_gt", 0.0), "train_L_state": train_metrics.get("L_state", 0.0), "train_L_consistency": train_metrics.get("L_consistency", 0.0), "val_L_stage2_front": val_metrics.get("L_stage2_front", 0.0), "val_L_stage2_back": val_metrics.get("L_stage2_back", 0.0), "val_L_direct": val_metrics.get("L_direct", 0.0), "val_L_decode_gt": val_metrics.get("L_decode_gt", 0.0), "val_L_state": val_metrics.get("L_state", 0.0), "val_L_consistency": val_metrics.get("L_consistency", 0.0)})
        print(f"stage2 epoch {epoch:03d} | train front={train_metrics.get('L_stage2_front', 0.0):.6f} train back={train_metrics.get('L_stage2_back', 0.0):.6f} | val front={val_metrics.get('L_stage2_front', 0.0):.6f} val back={val_metrics.get('L_stage2_back', 0.0):.6f}")
        last_payload = {"phase": "stage2", "resume_phase": "front", "epoch": epoch + 1, "next_batch_idx": 0, "next_window_idx": 0, "model_state_dict": model.state_dict(), "stage2_front_optimizer_state_dict": stage2_front_optimizer.state_dict(), "stage2_back_optimizer_state_dict": stage2_back_optimizer.state_dict(), "config": checkpoint_config, "best_val_loss": best_val, "best_val_metrics": best_metrics, "stage1_checkpoint_meta": {"path": str(stage1_checkpoint_path), "best_val_L_stage1": stage1_checkpoint.get("best_val_L_stage1")}}
        if stage1_optimizer is not None:
            last_payload["stage1_optimizer_state_dict"] = stage1_optimizer.state_dict()
        _save_checkpoint(checkpoint_paths["stage2_last"], last_payload, announce=False)
        current_val = _stage2_selection_value(val_metrics, selection_criterion)
        if current_val < best_val:
            best_val = current_val
            best_metrics = dict(val_metrics)
            best_payload = dict(last_payload)
            best_payload["best_val_loss"] = best_val
            best_payload["best_val_metrics"] = best_metrics
            _save_checkpoint(checkpoint_paths["stage2_best"], best_payload, announce=True)
    _write_csv_rows(checkpoint_paths["stage2_epoch_log"], epoch_log_rows)
    _write_csv_rows(checkpoint_paths["stage2_window_log"], window_log_rows)
    stage1_fingerprint_after = _fingerprint_module(model.single_step_model)
    verification = {"stage1_frozen_unchanged": (stage1_fingerprint_before == stage1_fingerprint_after) if freeze_stage1 else None, "stage1_fingerprint_before": stage1_fingerprint_before, "stage1_fingerprint_after": stage1_fingerprint_after, "window_log_path": str(checkpoint_paths["stage2_window_log"]), "freeze_stage1": freeze_stage1, "stage2_training_mode": stage2_training_mode, "selection_criterion": selection_criterion}
    best_checkpoint = torch.load(checkpoint_paths["stage2_best"], map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    return model, best_metrics, verification


def train_temporal_model(train_npz: str, val_npz: str, stats_npz: Optional[str] = None, input_steps: int = 12, output_steps: int = 1, stride: int = 1, epochs: int = 10, stage1_epochs: int = 10, batch_size: int = 16, lr: float = 1e-3, d_model: int = 64, n_heads: int = 4, temporal_hidden_dim: int = 128, training_input_strategy: str = "full_teacher_forcing", scheduled_sampling_prob: float = 0.5, device: Optional[str] = None, save_path: Optional[str] = None, resume_path: Optional[str] = None, run_mode: str = "all", stage1_checkpoint: Optional[str] = None, max_train_batches: Optional[int] = None, max_val_batches: Optional[int] = None, verify_checks: bool = False, mi_weight: Optional[float] = None, lambda_direct: Optional[float] = None, lambda_decode: Optional[float] = None, lambda_state: Optional[float] = None, lambda_phys: Optional[float] = None, lambda_consistency: float = 0.0, lambda_state_peak: Optional[float] = None, lambda_state_width: Optional[float] = None, lambda_state_noise: Optional[float] = None, lambda_state_pos: Optional[float] = None, consistency_schedule_type: str = "constant", consistency_warmup_epochs: int = 0, decoder_mode: str = "physics", latent_layout: str = "structured", lambda_cons: float = 0.5, stage2_training_mode: str = "front_back", freeze_stage1: bool = True, selection_criterion: str = "combined", experiment_name: str = "full_model", seed: Optional[int] = None) -> Tuple[Optional[nn.Module], Dict[str, Any]]:
    del stats_npz
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_paths = _resolve_checkpoint_paths(save_path)
    results: Dict[str, Any] = {"checkpoint_paths": {key: str(value) for key, value in checkpoint_paths.items()}, "run_mode": run_mode, "device": str(device_obj), "experiment_name": experiment_name, "seed": seed, "stage1_epochs": stage1_epochs, "stage2_epochs": epochs, "selection_criterion": selection_criterion, "timing_procedure": {"clock": "time.perf_counter", "gpu_synchronization": device_obj.type == "cuda", "includes_data_loading": True}}
    if run_mode not in {"stage1", "stage2", "all"}:
        raise ValueError(f"Unsupported run_mode: {run_mode}")
    stage1_model: Optional[DisentangledWindEstimator] = None
    if run_mode in {"stage1", "all"}:
        set_experiment_seed(seed)
        if device_obj.type == "cuda":
            torch.cuda.synchronize(device_obj)
            torch.cuda.reset_peak_memory_stats(device_obj)
        stage1_start = time.perf_counter()
        stage1_model, stage1_metrics = train_stage1(train_npz, val_npz, device_obj, d_model, n_heads, stage1_epochs, batch_size, lr, checkpoint_paths, max_train_batches, max_val_batches, mi_weight=mi_weight, decoder_mode=decoder_mode, latent_layout=latent_layout, lambda_cons=lambda_cons, experiment_name=experiment_name, seed=seed)
        if device_obj.type == "cuda":
            torch.cuda.synchronize(device_obj)
        results["stage1_training_seconds"] = time.perf_counter() - stage1_start
        results["stage1_peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device_obj)) if device_obj.type == "cuda" else 0
        results["stage1_reused"] = False
        results["stage1_metrics"] = stage1_metrics
    else:
        results["stage1_training_seconds"] = 0.0
        results["stage1_peak_gpu_memory_bytes"] = 0
        results["stage1_reused"] = True
    if run_mode == "stage1":
        results["total_stage_training_seconds"] = results["stage1_training_seconds"]
        results["peak_gpu_memory_bytes"] = results["stage1_peak_gpu_memory_bytes"]
        checkpoint_paths["training_summary"].write_text(json.dumps(results, indent=2), encoding="utf-8")
        return stage1_model, results
    stage1_checkpoint_path = Path(stage1_checkpoint) if stage1_checkpoint else checkpoint_paths["stage1_best"]
    effective_mi_weight = 0.1 if mi_weight is None else float(mi_weight)
    set_experiment_seed(seed)
    if device_obj.type == "cuda":
        torch.cuda.synchronize(device_obj)
        torch.cuda.reset_peak_memory_stats(device_obj)
    stage2_start = time.perf_counter()
    stage2_model, stage2_metrics, verification = train_stage2(train_npz, val_npz, input_steps, output_steps, stride, device_obj, d_model, n_heads, temporal_hidden_dim, epochs, batch_size, lr, checkpoint_paths, training_input_strategy, scheduled_sampling_prob, stage1_checkpoint_path, resume_path, max_train_batches, max_val_batches, verify_checks, lambda_direct=lambda_direct, lambda_decode=lambda_decode, lambda_state=lambda_state, lambda_phys=lambda_phys, lambda_consistency=lambda_consistency, lambda_state_peak=lambda_state_peak, lambda_state_width=lambda_state_width, lambda_state_noise=lambda_state_noise, lambda_state_pos=lambda_state_pos, consistency_schedule_type=consistency_schedule_type, consistency_warmup_epochs=consistency_warmup_epochs, decoder_mode=decoder_mode, latent_layout=latent_layout, lambda_cons=lambda_cons, mi_weight=effective_mi_weight, stage2_training_mode=stage2_training_mode, freeze_stage1=freeze_stage1, selection_criterion=selection_criterion, experiment_name=experiment_name, seed=seed)
    if device_obj.type == "cuda":
        torch.cuda.synchronize(device_obj)
    results["stage2_training_seconds"] = time.perf_counter() - stage2_start
    results["stage2_peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device_obj)) if device_obj.type == "cuda" else 0
    results["total_stage_training_seconds"] = results["stage1_training_seconds"] + results["stage2_training_seconds"]
    results["peak_gpu_memory_bytes"] = max(results["stage1_peak_gpu_memory_bytes"], results["stage2_peak_gpu_memory_bytes"])
    results["stage2_metrics"] = stage2_metrics
    results["verification"] = verification
    checkpoint_paths["training_summary"].write_text(json.dumps(results, indent=2), encoding="utf-8")
    return stage2_model, results


def cli_main():
    parser = argparse.ArgumentParser(description="Disentangled temporal wind forecaster")
    parser.add_argument("--mode", type=str, default="all", choices=["stage1", "stage2", "all"], help="Run stage1 only, stage2 only, or both in order")
    parser.add_argument("--train-npz", type=str, default="data/ready/NR-I/train_sequence.npz", help="Normalized training sequence from preprocess.py")
    parser.add_argument("--val-npz", type=str, default="data/ready/NR-I/val_sequence.npz", help="Normalized validation sequence from preprocess.py")
    parser.add_argument("--stats-npz", type=str, default="data/ready/NR-I/normalization_stats.npz", help="Training-set normalization statistics")
    parser.add_argument("--input-steps", type=int, default=12, help="History length in step units")
    parser.add_argument("--output-steps", type=int, default=1, help="Forecast horizon in step units")
    parser.add_argument("--stride", type=int, default=1, help="Window stride in step units")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per active stage")
    parser.add_argument("--stage1-epochs", type=int, default=10, help="Stage 1 epoch limit")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=64, help="Latent dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--temporal-hidden-dim", type=int, default=128, help="Temporal transition hidden dimension")
    parser.add_argument("--training-input-strategy", type=str, default="full_teacher_forcing", choices=["full_teacher_forcing", "free_running", "scheduled_sampling"], help="How the next rollout step is fed during stage2 rolling updates")
    parser.add_argument("--scheduled-sampling-prob", type=float, default=0.5, help="Teacher-forcing probability for scheduled sampling")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--save-path", type=str, default="checkpoints/temporal_forecaster.pt", help="Checkpoint base path; stage1/stage2 best/last paths are derived from it")
    parser.add_argument("--resume-path", type=str, default=None, help="Stage2 resume checkpoint for front/back recovery")
    parser.add_argument("--stage1-checkpoint", type=str, default=None, help="Explicit stage1 checkpoint path for stage2 or all mode")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional debug cap for train batches per epoch")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Optional debug cap for val batches per epoch")
    parser.add_argument("--verify-checks", action="store_true", help="Enable strict runtime freeze/hash verification during stage2")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--mi-weight", type=float, default=None, help="Override MI loss weight in stage1 (default 0.1)")
    parser.add_argument("--lambda-decode", type=float, default=None, help="Override lambda_decode in stage2")
    parser.add_argument("--lambda-direct", type=float, default=None, help="Override lambda_direct in controlled joint Stage2 modes")
    parser.add_argument("--lambda-state", type=float, default=None, help="Override lambda_state in stage2")
    parser.add_argument("--lambda-phys", type=float, default=None, help="Override physics consistency weight in stage2")
    parser.add_argument("--lambda-consistency", type=float, default=0.0, help="Target direct-to-decode consistency weight in controlled joint modes")
    parser.add_argument("--lambda-state-peak", type=float, default=None)
    parser.add_argument("--lambda-state-width", type=float, default=None)
    parser.add_argument("--lambda-state-noise", type=float, default=None)
    parser.add_argument("--lambda-state-pos", type=float, default=None)
    parser.add_argument("--consistency-schedule-type", choices=["constant", "delayed_linear"], default="constant")
    parser.add_argument("--consistency-warmup-epochs", type=int, default=0)
    parser.add_argument("--lambda-cons", type=float, default=0.5, help="Stage1 physics consistency loss weight")
    parser.add_argument("--decoder-mode", type=str, default="physics", choices=["physics", "pure_doppler", "mlp", "matched_mlp", "no_physics_mlp", "concat_only"], help="Ablation switch for the Stage1/decoder wind estimate")
    parser.add_argument("--latent-layout", type=str, default="structured", choices=["structured", "shared", "entangled_matched", "beta_vae", "mean_pool", "concat_only"], help="Use structured subspaces or a replacement latent ablation")
    parser.add_argument("--stage2-training-mode", type=str, default="front_back", choices=["front_back", "original_dual_substep", "joint", "joint_two_branch", "joint_two_substeps"], help="Stage2 temporal optimization mode")
    parser.add_argument("--freeze-stage1", type=str, default="true", choices=["true", "false"], help="Whether Stage1 stays frozen during Stage2")
    parser.add_argument("--selection-criterion", type=str, default="combined", choices=["direct_front", "decode_gt", "back", "combined"], help="Validation criterion used to select the Stage 2 checkpoint")
    parser.add_argument("--experiment-name", type=str, default="full_model", help="Experiment identifier stored in checkpoints")
    args = parser.parse_args()
    train_temporal_model(train_npz=args.train_npz, val_npz=args.val_npz, stats_npz=args.stats_npz, input_steps=args.input_steps, output_steps=args.output_steps, stride=args.stride, epochs=args.epochs, stage1_epochs=args.stage1_epochs, batch_size=args.batch_size, lr=args.lr, d_model=args.d_model, n_heads=args.n_heads, temporal_hidden_dim=args.temporal_hidden_dim, training_input_strategy=args.training_input_strategy, scheduled_sampling_prob=args.scheduled_sampling_prob, device=args.device, save_path=args.save_path, resume_path=args.resume_path, run_mode=args.mode, stage1_checkpoint=args.stage1_checkpoint, max_train_batches=args.max_train_batches, max_val_batches=args.max_val_batches, verify_checks=args.verify_checks, mi_weight=args.mi_weight, lambda_direct=args.lambda_direct, lambda_decode=args.lambda_decode, lambda_state=args.lambda_state, lambda_phys=args.lambda_phys, lambda_consistency=args.lambda_consistency, lambda_state_peak=args.lambda_state_peak, lambda_state_width=args.lambda_state_width, lambda_state_noise=args.lambda_state_noise, lambda_state_pos=args.lambda_state_pos, consistency_schedule_type=args.consistency_schedule_type, consistency_warmup_epochs=args.consistency_warmup_epochs, decoder_mode=args.decoder_mode, latent_layout=args.latent_layout, lambda_cons=args.lambda_cons, stage2_training_mode=args.stage2_training_mode, freeze_stage1=(args.freeze_stage1 == "true"), selection_criterion=args.selection_criterion, experiment_name=args.experiment_name, seed=args.seed)


if __name__ == "__main__":
    cli_main()
