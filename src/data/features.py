"""Causal motion features and ablation feature configurations.

The file variables STORM_SPEED/STORM_DIR are NEVER used (EDA Task 1 proved
they cannot be pure backward differences: they are populated at each storm's
first observation, where no past exists).  All motion below is computed from
past-only observed positions inside the same continuous segment.

For consecutive positions p(t-1) -> p(t) within one segment:

* distance ``d`` and forward bearing ``theta`` via :mod:`geometry`
  (WGS84 Geod, degrees east of north in [0, 360)),
* translation speed ``v = d / dt`` in **km/h**,
* eastward ``v_E = v·sin(theta)`` and northward ``v_N = v·cos(theta)`` in km/h.

Causality: the feature row at time t uses only positions at times <= t.
The first row of each segment has undefined (NaN) motion by construction.

History-boundary rule: the 5 input rows (t-24 … t) need the extra position
t-30 to give the first row a causal velocity, so windows require 6
consecutive history positions but expose only 5 input feature steps.

Feature matrices (per input row, 7 columns — scalers fit on these):
``[LAT, LON_U, VE_KMH, VN_KMH, SPEED_KMH, SIN_DIR, COS_DIR]``
where LON_U is the per-segment unwrapped longitude (continuous; equals the
stored 0–360 value except across antimeridian crossings).

Ablation configurations (identical sample IDs / origins / ground truth):
* A (position only):            [LAT, LON]
* B (position + speed + dir):   [LAT, LON, SPEED, sin(DIR), cos(DIR)]
* C (main: position + vector):  [LAT, LON, v_E, v_N]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import geometry as G

FEATURE_COLUMNS_7 = ["LAT", "LON_U", "VE_KMH", "VN_KMH", "SPEED_KMH", "SIN_DIR", "COS_DIR"]
CONFIG_COLUMNS = {
    "A": [0, 1],
    "B": [0, 1, 4, 5, 6],
    "C": [0, 1, 2, 3],
}
CONFIG_NAMES = {"A": "position_only", "B": "position_speed_dir", "C": "position_velocity"}
CONFIG_DIMS = {k: len(v) for k, v in CONFIG_COLUMNS.items()}


def add_segment_motion(seg: pd.DataFrame) -> pd.DataFrame:
    """Add causal motion columns to ONE time-sorted continuous segment.

    New columns: LON_U (unwrapped), DT_H, DIST_KM, DIR_DEG, SPEED_KMH,
    VE_KMH, VN_KMH.  Row 0 keeps NaN motion (no past).  Only the previous
    row of the same segment is ever used.
    """
    s = seg.copy()
    lon_u = G.unwrap_lon(pd.to_numeric(s["LON"], errors="coerce").to_numpy(dtype=float))
    lat = pd.to_numeric(s["LAT"], errors="coerce").to_numpy(dtype=float)
    s["LON_U"] = lon_u
    prev_lat = pd.Series(lat).shift(1).to_numpy(dtype=float)
    prev_lon = pd.Series(lon_u).shift(1).to_numpy(dtype=float)
    dt_h = s["TIME"].diff().dt.total_seconds().to_numpy(dtype=float) / 3600.0
    ok = np.isfinite(lat) & np.isfinite(lon_u) & np.isfinite(prev_lat) & np.isfinite(prev_lon) & (dt_h > 0)
    dist_km = np.full(len(s), np.nan)
    direction = np.full(len(s), np.nan)
    if ok.any():
        idx = np.where(ok)[0]
        dist_m, az = G.inverse(prev_lat[idx], prev_lon[idx], lat[idx], lon_u[idx])
        dist_km[idx] = np.asarray(dist_m, dtype=float) / 1000.0
        direction[idx] = np.asarray(az, dtype=float)
    speed = dist_km / dt_h
    rad = np.deg2rad(direction)
    s["DT_H"] = dt_h
    s["DIST_KM"] = dist_km
    s["DIR_DEG"] = direction
    s["SPEED_KMH"] = speed
    s["VE_KMH"] = speed * np.sin(rad)
    s["VN_KMH"] = speed * np.cos(rad)
    return s


def feature_matrix_7(rows: pd.DataFrame) -> np.ndarray:
    """Stack the 7-column feature matrix for already-motion-labelled rows."""
    dirs = np.deg2rad(pd.to_numeric(rows["DIR_DEG"], errors="coerce").to_numpy(dtype=float))
    mat = np.column_stack([
        pd.to_numeric(rows["LAT"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(rows["LON_U"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(rows["VE_KMH"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(rows["VN_KMH"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(rows["SPEED_KMH"], errors="coerce").to_numpy(dtype=float),
        np.sin(dirs),
        np.cos(dirs),
    ])
    return mat.astype(np.float64)
