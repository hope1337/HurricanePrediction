"""Normalization (train-only fitting) and the PyTorch dataset interface.

Scaler: per-column mean/std fitted on the train split's 7-column input
features over all input timesteps (train rows only — validation/test rows
never influence the statistics).  The same fitted parameters transform
validation and test.  Fitted values are persisted with joblib and the
transform is exactly invertible (:func:`invert`); targets are NOT
normalized — geographic ground truth (LAT/LON) is retained as-is for final
Haversine evaluation.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .features import CONFIG_COLUMNS

SCALER_EPS = 1e-8


def fit_scaler(X7_train: np.ndarray) -> dict:
    """Fit mean/std per feature column on train inputs shaped [N, 5, 7]."""
    flat = np.asarray(X7_train, dtype=np.float64).reshape(-1, 7)
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale = np.where(scale < SCALER_EPS, 1.0, scale)
    return {"mean": mean.astype(np.float64), "scale": scale.astype(np.float64),
            "columns": ["LAT", "LON_U", "VE_KMH", "VN_KMH", "SPEED_KMH", "SIN_DIR", "COS_DIR"]}


def apply_scaler(X7: np.ndarray, scaler: dict) -> np.ndarray:
    """Standardize with fitted parameters (any [N, 5, 7] array)."""
    return (np.asarray(X7, dtype=np.float64) - scaler["mean"]) / scaler["scale"]


def invert(X_scaled_cfg: np.ndarray, scaler: dict, config: str = "C") -> np.ndarray:
    """Inverse-transform a scaled config slice back to physical units."""
    cols = CONFIG_COLUMNS[config]
    return (np.asarray(X_scaled_cfg, dtype=np.float64) * scaler["scale"][cols]
            + scaler["mean"][cols])


def save_scaler(scaler: dict, path: str | Path) -> None:
    joblib.dump(scaler, path)


def load_scaler(path: str | Path) -> dict:
    scaler = joblib.load(path)
    scaler["mean"] = np.asarray(scaler["mean"], dtype=np.float64)
    scaler["scale"] = np.asarray(scaler["scale"], dtype=np.float64)
    return scaler
