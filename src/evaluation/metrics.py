"""Geographical trajectory error metrics (all in kilometres).

Definitions (H = 8 forecast steps, N = valid windows, d(i,k) = Haversine km):

* ``MTE(k) = mean_i d(i,k)`` — mean track error at horizon step k.
* ``ADE = mean over all N·H (i,k)`` — average displacement error (all steps).
* ``FDE = mean_i d(i,H)`` — final displacement error; equals MTE at +48h.
* Median / P90 by horizon — robust counterparts of MTE.
* Per-storm averages — mean over a storm's windows (storm-level aggregation).

Primary convention is per-window averaging; storm-level aggregation is
provided alongside so storms with many overlapping windows can be
down-weighted.  Invalid entries are handled only through an explicit
``valid_mask`` — NaNs outside the mask raise instead of being silently
dropped, and every aggregate reports its contributing sample count.
"""

from __future__ import annotations

import numpy as np

from src.data.geometry import haversine_distance_km

HORIZON_HOURS = (6, 12, 18, 24, 30, 36, 42, 48)
REPORT_HORIZONS = {6: 0, 12: 1, 24: 3, 48: 7}


def _as_float(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def pairwise_errors_km(pred_latlon, true_latlon) -> np.ndarray:
    """Haversine error matrix ``[N, H]`` from ``[N, H, 2]`` (lat, lon) pairs."""
    pred, true = _as_float(pred_latlon), _as_float(true_latlon)
    if pred.shape != true.shape or pred.ndim != 3 or pred.shape[2] != 2:
        raise ValueError(f"expected two [N, H, 2] arrays, got {pred.shape} and {true.shape}")
    for name, arr in (("predictions", pred), ("targets", true)):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contain NaN/inf — pass an explicit valid_mask or clean inputs")
        if (np.abs(arr[..., 0]) > 90).any():
            raise ValueError(f"{name} latitude outside [-90, 90]")
    return haversine_distance_km(pred[..., 0], pred[..., 1], true[..., 0], true[..., 1])


def _check_errors(errors, valid_mask=None) -> tuple[np.ndarray, np.ndarray]:
    err = _as_float(errors)
    if err.ndim != 2:
        raise ValueError(f"errors must be [N, H], got {err.shape}")
    mask = np.ones_like(err, dtype=bool) if valid_mask is None else np.asarray(valid_mask, bool)
    if mask.shape != err.shape:
        raise ValueError(f"mask shape {mask.shape} != errors shape {err.shape}")
    masked = np.where(mask, err, np.nan)
    if np.isnan(err[mask]).any():
        raise ValueError("NaN errors inside valid_mask — handle invalid values explicitly")
    if not (masked[mask] >= 0).all():
        raise ValueError("negative distances are impossible")
    return masked, mask


def mte_by_horizon(errors, valid_mask=None) -> tuple[np.ndarray, np.ndarray]:
    """Mean + contributing counts per horizon step; returns ``(mte[H], n[H])``."""
    masked, mask = _check_errors(errors, valid_mask)
    with np.errstate(invalid="ignore"):
        mte = np.nanmean(np.where(mask, masked, np.nan), axis=0)
    return mte, mask.sum(axis=0).astype(np.int64)


def median_by_horizon(errors, valid_mask=None) -> np.ndarray:
    masked, mask = _check_errors(errors, valid_mask)
    return np.array([np.median(masked[mask[:, k], k]) if mask[:, k].any() else np.nan
                     for k in range(masked.shape[1])])


def p90_by_horizon(errors, valid_mask=None) -> np.ndarray:
    masked, mask = _check_errors(errors, valid_mask)
    return np.array([np.percentile(masked[mask[:, k], k], 90) if mask[:, k].any() else np.nan
                     for k in range(masked.shape[1])])


def ade(errors, valid_mask=None) -> tuple[float, int]:
    """Average displacement error over all valid (i, k); returns ``(value, n)``."""
    masked, mask = _check_errors(errors, valid_mask)
    vals = masked[mask]
    return float(vals.mean()), int(vals.size)


def fde(errors, valid_mask=None) -> tuple[float, int]:
    """Final displacement error (last horizon step); returns ``(value, n)``."""
    masked, mask = _check_errors(errors, valid_mask)
    col = masked[:, -1]
    valid = mask[:, -1] & np.isfinite(col)
    if not valid.any():
        raise ValueError("no valid samples at the final horizon")
    return float(col[valid].mean()), int(valid.sum())


def per_trajectory_ade(errors, valid_mask=None) -> np.ndarray:
    """Mean error of each window over its valid steps; ``[N]``."""
    masked, mask = _check_errors(errors, valid_mask)
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.where(mask, masked, np.nan), axis=1)


def per_storm_average(errors, storm_ids, valid_mask=None) -> dict:
    """Mean window-ADE per storm: ``{storm_id: (mean_km, n_windows)}``."""
    traj = per_trajectory_ade(errors, valid_mask)
    sids = np.asarray(storm_ids)
    if sids.shape[0] != traj.shape[0]:
        raise ValueError("storm_ids length must match number of windows")
    out: dict[str, tuple[float, int]] = {}
    for sid in np.unique(sids):
        vals = traj[sids == sid]
        out[str(sid)] = (float(np.mean(vals)), int(vals.size))
    return out
