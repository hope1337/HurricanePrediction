"""Forecasting strategies: direct multi-horizon vs recursive one-step.

Representation contract (never mixed silently)
---------------------------------------------
* ``X_norm`` — normalized input features ``[B, 5, D]`` (Task 2 scaler).
* ``disp_km`` — physical east/north displacements in kilometres, ``[B, S, 2]``
  (direct, origin-anchored) or ``[B, 2]`` (one step, step-origin frame).
* ``positions`` — geographic ``(lat, lon)`` with lon in the file convention
  0–360°E, matching the Task 2 ``Y`` ground truth.

Recursive rollout
-----------------
Repeats: predict one-step ``(e, n)`` -> Geod-forward to the next position ->
rebuild that row's causal motion features from the predicted displacement
only -> renormalize with the training-fitted scaler -> slide the window.
Inputs are re-encoded each step (stateless across steps).  The rollout
accepts **no ground-truth argument**, so future observations cannot leak by
construction; geometry/feature ops run in NumPy (inference-only, gradients
stop at the one-step head — train recursive models with teacher forcing).

Per-config next-row reconstruction from predicted ``(e, n)`` over ``dt`` h:

* A ``[LAT, LON]``: new lat/lon only.
* B ``[LAT, LON, SPEED, sinDIR, cosDIR]``: speed=hypot(e,n)/dt, dir=atan2(e,n).
* C ``[LAT, LON, v_E, v_N]``: v=(e/dt, n/dt).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.data import geometry as G
from src.data.features import CONFIG_COLUMNS

from .heads import DirectDisplacementHead, OneStepDisplacementHead
from .recurrent import RecurrentBackbone


def decode_direct(disp_km: np.ndarray | torch.Tensor,
                  origin_latlon: np.ndarray | torch.Tensor) -> np.ndarray:
    """Decode origin-anchored ENU displacements (km) to ``(lat, lon)``.

    Returns ``[B, S, 2]`` with lon wrapped to 0–360°E.  Exact inverse of the
    Task 2 ``Y_direct`` target construction.
    """
    e = np.asarray(disp_km.detach().cpu() if isinstance(disp_km, torch.Tensor) else disp_km,
                   dtype=float)[..., 0]
    n = np.asarray(disp_km.detach().cpu() if isinstance(disp_km, torch.Tensor) else disp_km,
                   dtype=float)[..., 1]
    o = np.asarray(origin_latlon.detach().cpu() if isinstance(origin_latlon, torch.Tensor)
                   else origin_latlon, dtype=float)
    lat, lon = G.geodetic_from_enu(o[..., 0:1], o[..., 1:2], e, n)
    return np.stack([np.asarray(lat, float), G.normalize_lon_360(np.asarray(lon, float))], axis=-1)


class DirectForecaster(nn.Module):
    """Single-forward-pass prediction of all horizons (no feedback)."""

    def __init__(self, backbone: RecurrentBackbone, head: DirectDisplacementHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """``[B, 5, D]`` normalized inputs -> ``[B, S, 2]`` origin-anchored km."""
        return self.head(self.backbone(x_norm))


class RecursiveForecaster(nn.Module):
    """Autoregressive rollout producing decoded geographic positions."""

    def __init__(self, backbone: RecurrentBackbone, head: OneStepDisplacementHead,
                 scaler: dict, config: str = "C", forecast_steps: int = 8,
                 interval_hours: float = 6.0):
        super().__init__()
        if config not in CONFIG_COLUMNS:
            raise ValueError(f"config must be one of {tuple(CONFIG_COLUMNS)}, got {config!r}")
        self.backbone = backbone
        self.head = head
        self.cols = list(CONFIG_COLUMNS[config])
        self.config = config
        self.forecast_steps = int(forecast_steps)
        self.interval_hours = float(interval_hours)
        mean = np.asarray(scaler["mean"], dtype=np.float64)
        scale = np.asarray(scaler["scale"], dtype=np.float64)
        self.register_buffer("mean", torch.as_tensor(mean).float())
        self.register_buffer("scale", torch.as_tensor(scale).float())

    def forward_onestep(self, x_norm: torch.Tensor) -> torch.Tensor:
        """``[B, 5, D]`` normalized inputs -> ``[B, 2]`` one-step km (grad-safe)."""
        return self.head(self.backbone(x_norm))

    def _normalize(self, phys_cfg: np.ndarray, device) -> torch.Tensor:
        import torch as T
        return T.as_tensor((phys_cfg - self.mean.cpu().numpy()[self.cols])
                           / self.scale.cpu().numpy()[self.cols],
                           dtype=torch.float32, device=device)

    @torch.no_grad()
    def rollout(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Roll out 8 steps -> decoded ``[B, 8, 2]`` positions (lon 0–360).

        Inference-only: gradients are detached at each geometry boundary.
        Uses only ``x_norm`` + the fitted scaler — no ground truth involved.
        """
        device = x_norm.device
        cols = self.cols
        mean = self.mean.cpu().numpy()[cols]
        scale = self.scale.cpu().numpy()[cols]
        window = (x_norm.detach().cpu().numpy().astype(np.float64) * scale + mean)  # [B,5,D] phys
        lat = window[..., 0].copy()
        lon_u = window[..., 1].copy()  # unwrapped; Geod accepts any convention
        positions = []
        cur = torch.as_tensor((window - mean) / scale, dtype=torch.float32, device=device)
        for _ in range(self.forecast_steps):
            disp = self.head(self.backbone(cur)).detach().cpu().numpy().astype(np.float64)  # [B,2]
            e, n = disp[:, 0], disp[:, 1]
            dist_m = np.hypot(e, n) * 1000.0
            az = (np.rad2deg(np.arctan2(e, n)) + 360.0) % 360.0
            la2, lo2 = G.forward(lat[:, -1], lon_u[:, -1], az, dist_m)
            la2 = np.asarray(la2, float).reshape(-1)
            lo2 = np.asarray(lo2, float).reshape(-1)
            speed = np.hypot(e, n) / self.interval_hours
            s_az = np.deg2rad(az)
            if self.config == "A":
                new_row = np.stack([la2, lo2], axis=-1)
            elif self.config == "B":
                new_row = np.stack([la2, lo2, speed, np.sin(s_az), np.cos(s_az)], axis=-1)
            else:
                new_row = np.stack([la2, lo2, e / self.interval_hours, n / self.interval_hours],
                                   axis=-1)
            phys = np.concatenate([window[:, 1:, :], new_row[:, None, :]], axis=1)
            window = phys
            lat = phys[..., 0]
            lon_u = phys[..., 1]
            positions.append(np.stack([la2, G.normalize_lon_360(lo2)], axis=-1))
            cur = torch.as_tensor((phys - mean) / scale, dtype=torch.float32, device=device)
        return torch.as_tensor(np.stack(positions, axis=1), dtype=torch.float32, device=device)
