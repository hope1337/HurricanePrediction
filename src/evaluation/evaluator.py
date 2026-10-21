"""Model-agnostic evaluation interface for trajectory forecasts.

``evaluate_trajectories`` accepts decoded geographic positions ``[N, 8, 2]``
from any of the six architecture × strategy baselines and returns a
structured result dict.  ``assert_consistent_samples`` enforces identical
forecast populations when comparing models: same sample IDs in the same
order, otherwise a clear error is raised instead of silently comparing
different populations.
"""

from __future__ import annotations

import numpy as np

from . import metrics as M


def assert_consistent_samples(sample_ids_a, sample_ids_b, label_a="A", label_b="B") -> None:
    """Raise unless two result sets share identical sample IDs in order."""
    a = np.asarray(sample_ids_a, dtype=str)
    b = np.asarray(sample_ids_b, dtype=str)
    if a.shape != b.shape or not (a == b).all():
        raise ValueError(
            f"inconsistent forecast samples between {label_a} (n={len(a)}) and "
            f"{label_b} (n={len(b)}): identical sample_ids in identical order are required")
    if len(np.unique(a)) != len(a):
        raise ValueError("duplicate sample_ids detected")


def evaluate_trajectories(predictions, targets, storm_ids, sample_ids,
                          init_times=None, valid_mask=None) -> dict:
    """Evaluate one model's decoded forecasts.

    Parameters
    ----------
    predictions, targets : [N, H, 2] (lat, lon) in matching lon conventions.
    storm_ids, sample_ids : length-N identifiers.
    init_times : optional length-N forecast-initialization timestamps.
    valid_mask : optional [N, H] bool; invalid entries are excluded and
        reported — never silently dropped.

    Returns a dict with MTE/median/P90 by horizon (plus horizon hours),
    ADE, FDE, per-sample ``[N, H]`` errors, per-storm summaries, and counts.
    """
    errors = M.pairwise_errors_km(predictions, targets)
    if errors.shape[1] != 8:
        raise ValueError(f"expected 8 forecast steps, got {errors.shape[1]}")
    sids = np.asarray(storm_ids, dtype=str)
    aids = np.asarray(sample_ids, dtype=str)
    if not (len(sids) == len(aids) == errors.shape[0]):
        raise ValueError("storm_ids/sample_ids length must match number of windows")
    if len(np.unique(aids)) != len(aids):
        raise ValueError("duplicate sample_ids detected")
    mte, n_h = M.mte_by_horizon(errors, valid_mask)
    med = M.median_by_horizon(errors, valid_mask)
    p90 = M.p90_by_horizon(errors, valid_mask)
    ade_v, ade_n = M.ade(errors, valid_mask)
    fde_v, fde_n = M.fde(errors, valid_mask)
    per_traj = M.per_trajectory_ade(errors, valid_mask)
    per_storm = M.per_storm_average(errors, sids, valid_mask)
    return {
        "n_windows": int(errors.shape[0]),
        "n_storms": int(len(np.unique(sids))),
        "horizon_hours": list(M.HORIZON_HOURS[:errors.shape[1]]),
        "mte_km": {h: float(mte[k]) for h, k in M.REPORT_HORIZONS.items()},
        "mte_full_km": [float(v) for v in mte],
        "median_km": {h: float(med[k]) for h, k in M.REPORT_HORIZONS.items()},
        "p90_km": {h: float(p90[k]) for h, k in M.REPORT_HORIZONS.items()},
        "ade_km": ade_v, "ade_n": ade_n,
        "fde_km": fde_v, "fde_n": fde_n,
        "n_by_horizon": [int(v) for v in n_h],
        "per_sample_errors_km": errors if valid_mask is None else np.where(valid_mask, errors, np.nan),
        "per_trajectory_ade_km": per_traj,
        "per_storm": {s: {"mean_ade_km": v[0], "n_windows": v[1]} for s, v in per_storm.items()},
        "storm_ids": sids, "sample_ids": aids,
        "init_times": None if init_times is None else np.asarray(init_times),
    }
