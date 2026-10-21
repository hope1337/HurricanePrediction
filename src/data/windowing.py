"""Sliding-window construction over continuous 6-hourly segments.

One forecast window consumes 14 consecutive observations of a single
segment (indices k … k+13):

* history positions ``H`` = rows k … k+5  (t-30 … t, original LAT/LON),
* input rows ``X`` = rows k+1 … k+5 (t-24 … t) with causal motion computed
  from H (the t-30 position gives the first input row its velocity),
* targets ``Y`` = rows k+6 … k+13 (t+6 … t+48, original LAT/LON).

Expected shapes: X7 [N, 5, 7] (configs slice columns: A [N,5,2],
B [N,5,5], C/main [N,5,4]); Y [N, 8, 2].

Displacement targets (km, geodesic, exactly reversible — see tests):
* ``direct``: future east/north offset of each of the 8 targets relative to
  the forecast origin (last input position at t), in the origin's local
  tangent frame.  Reference frame: origin-anchored ENU.
* ``step`` (recursive): one-step east/north offsets between successive
  positions (t→t+6, …, t+42→t+48), each expressed in *its own step-origin*
  local frame.  Chaining Geod-forward steps from the forecast origin
  reconstructs the full trajectory.

No window is generated across storms, missing intervals, track
discontinuities, or split boundaries: generation runs per segment, and
segments belong to exactly one storm and one season-block split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import geometry as G
from .features import add_segment_motion, feature_matrix_7

WINDOW_OBS = 14  # 6 history + 8 future
HISTORY_POSITIONS = 6
INPUT_STEPS = 5
FORECAST_STEPS = 8


def generate_windows(df_segments: pd.DataFrame) -> dict:
    """Build all windows; return arrays + per-window metadata lists.

    ``df_segments`` must be sorted by (SID, TIME) and carry SEG_ID/SEG_LEN.
    Segments shorter than 14 observations yield no windows.
    """
    X7, Y, Ydir, Ystep = [], [], [], []
    meta: list[dict] = []
    for seg_id, seg in df_segments.groupby("SEG_ID", sort=False):
        seg = seg.sort_values("TIME", kind="mergesort").reset_index(drop=True)
        if len(seg) < WINDOW_OBS:
            continue
        seg = add_segment_motion(seg)
        lat = pd.to_numeric(seg["LAT"], errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(seg["LON"], errors="coerce").to_numpy(dtype=float)  # orig convention
        times = pd.to_datetime(seg["TIME"], utc=True)
        sid = str(seg["SID"].iloc[0])
        season = int(seg["SEASON"].iloc[0])
        for k in range(len(seg) - WINDOW_OBS + 1):
            h = seg.iloc[k:k + HISTORY_POSITIONS]          # 6 positions t-30..t
            rows = seg.iloc[k + 1:k + HISTORY_POSITIONS]  # 5 input rows t-24..t
            fut = seg.iloc[k + HISTORY_POSITIONS:k + WINDOW_OBS]  # 8 targets
            if feature_matrix_7(rows).shape != (INPUT_STEPS, 7):
                raise RuntimeError("Unexpected feature matrix shape")
            if not np.isfinite(feature_matrix_7(rows)).all():
                continue  # never emit NaN features (paranoia; filtered data are complete)
            X7.append(feature_matrix_7(rows))
            y = np.column_stack([lat[k + 6:k + 14], lon[k + 6:k + 14]])
            Y.append(y)
            o_lat, o_lon = float(lat[k + 5]), float(lon[k + 5])  # forecast origin (t)
            e_d, n_d = G.enu_from_geodetic(o_lat, o_lon, y[:, 0], y[:, 1])
            Ydir.append(np.column_stack([e_d, n_d]))
            seq_lat = np.concatenate([[o_lat], y[:, 0]])
            seq_lon = np.concatenate([[o_lon], y[:, 1]])
            e_s, n_s = G.enu_from_geodetic(seq_lat[:-1], seq_lon[:-1], seq_lat[1:], seq_lon[1:])
            Ystep.append(np.column_stack([e_s, n_s]))
            t0 = times.iloc[k + 5]
            meta.append({
                "sample_id": f"{sid}__{t0.strftime('%Y%m%dT%H%M')}__k{k}",
                "sid": sid, "season": season, "seg_id": str(seg_id),
                "t0_iso": t0.isoformat(),
                "target_iso": [t.isoformat() for t in times.iloc[k + 6:k + 14]],
                "last_lat": o_lat, "last_lon": o_lon,
                "hist_lat": lat[k:k + 6].tolist(), "hist_lon": lon[k:k + 6].tolist(),
            })
    if not X7:
        raise RuntimeError("No forecast windows generated (check filters/segments).")
    return {"X7": np.asarray(X7), "Y": np.asarray(Y),
            "Y_direct": np.asarray(Ydir), "Y_step": np.asarray(Ystep), "meta": meta}
