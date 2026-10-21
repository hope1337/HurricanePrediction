"""Geodesic and trajectory metrics (past-only, leakage-safe).

All motion estimates here use *past* observations only (backward differences),
so they are valid references for what a causal model input could contain.
IBTrACS ``STORM_SPEED``/``STORM_DIR`` are investigated against these; the
comparison must not assume the file variables are causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EXTREME_SPEED_KTS, JUMP_KM_THRESHOLD
from .io import unwrap_longitude_per_storm

EARTH_RADIUS_KM = 6371.0
KM_PER_NM = 1.852


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km (longitudes in degrees, antimeridian-safe)."""
    lat1 = np.deg2rad(np.asarray(lat1, dtype=float))
    lat2 = np.deg2rad(np.asarray(lat2, dtype=float))
    dlat = lat2 - lat1
    dlon = np.deg2rad((np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float) + 540.0) % 360.0 - 180.0)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Initial bearing in degrees east of north (same convention as STORM_DIR)."""
    lat1 = np.deg2rad(np.asarray(lat1, dtype=float))
    lat2 = np.deg2rad(np.asarray(lat2, dtype=float))
    dlon = np.deg2rad((np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float) + 540.0) % 360.0 - 180.0)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.rad2deg(np.arctan2(x, y)) + 360.0) % 360.0


def add_past_only_motion(df: pd.DataFrame) -> pd.DataFrame:
    """Add backward-difference displacement/speed/direction columns.

    ``RECOMPUTED_SPEED`` uses only the previous valid position of the same
    storm (NaN for each storm's first valid observation by construction).
    """
    out = df.copy()
    out["LON_UNWRAPPED"] = unwrap_longitude_per_storm(out, "LON")
    out["PREV_LAT"] = out.groupby("SID")["LAT"].shift(1)
    out["PREV_LON"] = out.groupby("SID")["LON_UNWRAPPED"].shift(1)
    out["PREV_TIME"] = out.groupby("SID")["TIME"].shift(1)
    dt_h = (out["TIME"] - out["PREV_TIME"]).dt.total_seconds() / 3600.0
    valid = (
        out["LAT"].notna() & out["LON_UNWRAPPED"].notna()
        & out["PREV_LAT"].notna() & out["PREV_LON"].notna()
        & out["PREV_TIME"].notna() & (dt_h > 0)
    )
    dist = np.full(len(out), np.nan)
    brg = np.full(len(out), np.nan)
    dist[valid] = haversine_km(
        out.loc[valid, "PREV_LAT"], out.loc[valid, "PREV_LON"],
        out.loc[valid, "LAT"], out.loc[valid, "LON_UNWRAPPED"],
    )
    brg[valid] = bearing_deg(
        out.loc[valid, "PREV_LAT"], out.loc[valid, "PREV_LON"],
        out.loc[valid, "LAT"], out.loc[valid, "LON_UNWRAPPED"],
    )
    out["DISPLACEMENT_KM"] = dist
    out["DT_HOURS"] = np.where(valid, dt_h, np.nan)
    out["RECOMPUTED_SPEED"] = dist / KM_PER_NM / np.where(valid, dt_h, np.nan)  # knots
    out["RECOMPUTED_DIR"] = brg
    out["SUSPICIOUS_JUMP"] = (out["DISPLACEMENT_KM"] > JUMP_KM_THRESHOLD).fillna(False)
    out["EXTREME_RECOMPUTED_SPEED"] = (out["RECOMPUTED_SPEED"] > EXTREME_SPEED_KTS).fillna(False)
    return out


def gap_table(df: pd.DataFrame) -> pd.DataFrame:
    """Consecutive-observation time gaps per storm (hours)."""
    d = df.sort_values(["SID", "TIME"], kind="mergesort").copy()
    d["GAP_H"] = d.groupby("SID")["TIME"].diff().dt.total_seconds() / 3600.0
    gaps = d.loc[d["GAP_H"].notna(), "GAP_H"].to_numpy(dtype=float)
    return d[["SID", "TIME", "GAP_H"]], gaps


def runs_of_exact_6h(df: pd.DataFrame, tol_seconds: float = 1.0) -> pd.DataFrame:
    """Label maximal runs of exactly-6-hourly observations per storm.

    Parameters
    ----------
    df : DataFrame with SID/TIME, sorted or unsorted.
    tol_seconds : tolerance around exactly 6h (timestamps are minute-exact so
        the default 1s demands a true 6h cadence).

    Returns a copy with ``RUN_ID`` (``<SID>#<k>``) and ``RUN_LEN`` columns;
    isolated observations (no exact-6h neighbour) get ``RUN_LEN == 1``.
    """
    d = df.sort_values(["SID", "TIME"], kind="mergesort").copy()
    gap = d.groupby("SID")["TIME"].diff().dt.total_seconds()
    is_6h = (np.abs(gap - 6 * 3600.0) <= tol_seconds).fillna(False)
    run = np.zeros(len(d), dtype=int)
    current = 0
    prev_sid = None
    for pos, (sid, cont) in enumerate(zip(d["SID"].to_numpy(), is_6h.to_numpy())):
        if sid != prev_sid:
            current = 0
            prev_sid = sid
        elif not cont:
            current += 1
        run[pos] = current
    d["RUN_ID"] = d["SID"].astype(str) + "#" + pd.Series(run, index=d.index).astype(str)
    d["RUN_LEN"] = d.groupby("RUN_ID")["RUN_ID"].transform("size")
    return d


def count_windows(run_lens: pd.Series, window: int = 13) -> int:
    """Count sliding windows of ``window`` consecutive obs inside each run."""
    return int(sum(max(0, int(n) - window + 1) for n in run_lens))


def subset_filters(df: pd.DataFrame, start: int = 1980, end: int = 2024) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the experimental-subset filters in a fixed, documented order.

    Order: F1 season -> F2 per-observation WP basin -> F3 main track ->
    F4 synoptic 00/06/12/18 UTC -> F5 valid lat/lon.
    (F6 continuity is analysed separately on the F5 output.)
    """
    from .config import SYNOPTIC_HOURS_6H

    steps: list[dict] = []

    def record(step: str, frame: pd.DataFrame, desc: str) -> None:
        steps.append({"step": step, "description": desc,
                      "n_obs": int(len(frame)),
                      "n_storms": int(frame["SID"].nunique()) if len(frame) else 0})

    record("F0_raw", df, "All raw records (numobs cells, incl. padding-excluded only)")
    f1 = df[(df["SEASON"] >= start) & (df["SEASON"] <= end)].copy()
    record(f"F1_season_{start}_{end}", f1, f"Season in [{start}, {end}]")
    f2 = f1[f1["BASIN"] == "WP"].copy()
    record("F2_basin_WP", f2, "Per-observation BASIN == 'WP'")
    f3 = f2[f2["TRACK_TYPE"] == "main"].copy()
    record("F3_track_main", f3, "TRACK_TYPE == 'main' (spur-merge/split/other, PROVISIONAL, US-PROVISIONAL excluded)")
    hh = f3["TIME"].dt.hour
    mm = f3["TIME"].dt.minute
    f4 = f3[mm.eq(0) & hh.isin(list(SYNOPTIC_HOURS_6H))].copy()
    record("F4_synoptic_6h", f4, "UTC synoptic HH in {0,6,12,18}, minute == 0")
    f5 = f4[f4["LAT"].between(-90, 90) & f4["LON"].between(0, 360)
            & f4["LAT"].notna() & f4["LON"].notna() & f4["TIME"].notna()].copy()
    record("F5_valid_position", f5, "Finite LAT in [-90,90], LON in [0,360], non-missing TIME")
    return f5, pd.DataFrame(steps)
