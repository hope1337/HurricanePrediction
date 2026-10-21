"""Error-analysis support: per-group evaluation without imposed thresholds.

Movement categories (e.g. straight-moving vs turning storms) are defined
downstream, during the experiment stage, from historical motion available at
forecast initialization.  This module only *applies* a caller-supplied
per-sample grouping: it never classifies tracks or imposes scientific
thresholds itself.  The single exception is :func:`classify_movement`, the
documented experiment-stage rule kept here so it can be unit-tested.
"""

from __future__ import annotations

import numpy as np

from src.data import geometry as G

from . import metrics as M


def per_sample_frame(result: dict) -> dict:
    """Flatten an ``evaluate_trajectories`` result to per-sample arrays."""
    n = result["n_windows"]
    return {
        "sample_id": np.asarray(result["sample_ids"], dtype=str),
        "storm_id": np.asarray(result["storm_ids"], dtype=str),
        "init_time": (None if result["init_times"] is None
                      else np.asarray(result["init_times"])),
        "errors_km": np.asarray(result["per_sample_errors_km"], dtype=float),  # [N, H]
        "ade_km": np.asarray(result["per_trajectory_ade_km"], dtype=float),     # [N]
        "n": n,
    }


def group_evaluate(errors_km, groups) -> dict:
    """Evaluate metric set per group label.

    ``errors_km`` is ``[N, H]``; ``groups`` is length-N (any hashable labels).
    Returns ``{label: {'n_windows', 'ade_km', 'fde_km', 'mte_full_km'}}``.
    """
    err = np.asarray(errors_km, dtype=float)
    grp = np.asarray(groups)
    if grp.shape[0] != err.shape[0]:
        raise ValueError("groups length must match number of windows")
    out: dict = {}
    for label in np.unique(grp):
        sub = err[grp == label]
        ade_v, _ = M.ade(sub)
        fde_v, _ = M.fde(sub)
        mte, n_h = M.mte_by_horizon(sub)
        out[str(label)] = {"n_windows": int(sub.shape[0]), "ade_km": ade_v,
                           "fde_km": fde_v,
                           "mte_full_km": [float(v) for v in mte],
                           "n_by_horizon": [int(v) for v in n_h]}
    return out


def classify_movement(hist_latlon, turn_deg: float = 30.0,
                      low_speed_kmh: float = 5.0) -> tuple[np.ndarray, dict]:
    """Classify test windows by *historical* motion only (fully causal).

    ``hist_latlon`` is ``[N, 6, 2]`` (t-30..t, degrees).  For each window:

    1. Compute the 5 consecutive 6-hourly bearings and their speeds (km/h).
    2. Take circular (smallest-angle) changes between consecutive bearings;
       ``max_change`` is the largest of the 4 values.
    3. ``ambiguous`` if mean history speed < ``low_speed_kmh`` (bearings from
       near-stationary fixes are unreliable).
    4. Else ``turning`` if ``max_change`` > ``turn_deg``, else ``straight``.

    Never touches future positions.  Returns (labels [N], details dict with
    per-window max_change_deg and mean_speed_kmh).
    """
    lat = np.asarray(hist_latlon, dtype=float)[:, :, 0]
    lon = np.asarray(hist_latlon, dtype=float)[:, :, 1]
    dist_m, az = G.inverse(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
    bearings = np.asarray(az, float)  # [N, 5]
    speeds = np.asarray(dist_m, float) / 1000.0 / 6.0  # [N, 5] km/h
    delta = np.abs((bearings[:, 1:] - bearings[:, :-1] + 180.0) % 360.0 - 180.0)
    max_change = delta.max(axis=1)
    mean_speed = speeds.mean(axis=1)
    labels = np.where(mean_speed < low_speed_kmh, "ambiguous",
                      np.where(max_change > turn_deg, "turning", "straight"))
    return labels, {"max_change_deg": max_change, "mean_speed_kmh": mean_speed}
