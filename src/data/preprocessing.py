"""Filtering and continuous-segment construction for IBTrACS WP tracks.

Filter order (fixed, documented; mirrors EDA F0–F5 so counts stay comparable):

1. ``season`` — SEASON in [start, end].
2. ``basin`` — per-observation BASIN == 'WP'.  Observations outside WP are
   dropped; because segmentation runs *after* filtering, windows can never
   span a basin transition.  Storms crossing boundaries therefore contribute
   separate WP-only segments.
3. ``track`` — TRACK_TYPE == 'main' only (verified EDA semantics: excludes
   spur-merge/split/other and PROVISIONAL/US-PROVISIONAL rows).
4. ``synoptic`` — timestamps with minute == 0 and hour in {0, 6, 12, 18} UTC.
5. ``valid`` — finite LAT in [-90, 90], LON in [0, 360], non-missing TIME.
6. ``sorted`` — sort by (SID, TIME, OBS_IDX).
7. ``dedup`` — deterministic duplicate policy: after sorting, keep the first
   row of any duplicated (SID, TIME) pair (EDA confirmed zero such pairs in
   the raw file; the policy exists so behaviour stays deterministic).
8. ``segments`` — split each storm into maximal runs with exact 6 h spacing
   (tolerance configurable, default 1 s).  No interpolation across gaps; no
   window may span different storms, missing intervals, or discontinuities.

Splits are assigned by storm SEASON into fixed season blocks, so a given
SID can only ever belong to one split (SEASON is a per-storm attribute).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def apply_filters(df: pd.DataFrame, start: int, end: int,
                  basin: str = "WP", track: str = "main",
                  synoptic_hours: tuple = (0, 6, 12, 18)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply filters 1–5 in fixed order; return (filtered, step_log)."""
    steps: list[dict] = []

    def record(name: str, frame: pd.DataFrame, desc: str) -> None:
        steps.append({"step": name, "description": desc,
                      "n_obs": int(len(frame)),
                      "n_storms": int(frame["SID"].nunique()) if len(frame) else 0})

    record("raw", df, "All raw records (numobs cells)")
    f = df[(df["SEASON"] >= start) & (df["SEASON"] <= end)].copy()
    record("season", f, f"SEASON in [{start}, {end}]")
    f = f[f["BASIN"] == basin].copy()
    record("basin", f, f"Per-observation BASIN == {basin!r}")
    f = f[f["TRACK_TYPE"] == track].copy()
    record("track", f, f"TRACK_TYPE == {track!r}")
    hh, mm = f["TIME"].dt.hour, f["TIME"].dt.minute
    f = f[mm.eq(0) & hh.isin(list(synoptic_hours))].copy()
    record("synoptic", f, f"UTC synoptic hours {sorted(synoptic_hours)}, minute == 0")
    f = f[f["LAT"].between(-90, 90) & f["LON"].between(0, 360)
          & f["LAT"].notna() & f["LON"].notna() & f["TIME"].notna()].copy()
    record("valid", f, "Finite LAT in [-90,90], LON in [0,360], non-missing TIME")
    return f, pd.DataFrame(steps)


def sort_and_deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Sort by (SID, TIME, OBS_IDX); keep first of any duplicated (SID, TIME)."""
    ordered = df.sort_values(["SID", "TIME", "OBS_IDX"], kind="mergesort").reset_index(drop=True)
    dup = ordered.duplicated(subset=["SID", "TIME"], keep="first")
    n_dropped = int(dup.sum())
    if n_dropped:
        log.warning("Dropping %d duplicated (SID, TIME) rows (keep first).", n_dropped)
    return ordered.loc[~dup].reset_index(drop=True), n_dropped


def split_segments(df: pd.DataFrame, interval_hours: float = 6.0,
                   tol_seconds: float = 1.0) -> pd.DataFrame:
    """Label maximal exactly-``interval_hours`` runs per storm.

    Adds ``SEG_ID`` (``<SID>#<k>``), ``SEG_LEN`` and ``SEG_POS`` (0-based index
    of the row inside its segment).  Isolated observations get ``SEG_LEN == 1``.
    """
    d = df.copy()
    gap_s = d.groupby("SID")["TIME"].diff().dt.total_seconds()
    is_step = (np.abs(gap_s - interval_hours * 3600.0) <= tol_seconds).fillna(False).to_numpy()
    sids = d["SID"].to_numpy()
    run = np.zeros(len(d), dtype=np.int64)
    cur = 0
    prev = None
    for pos, (sid, cont) in enumerate(zip(sids, is_step)):
        if sid != prev:
            cur, prev = 0, sid
        elif not cont:
            cur += 1
        run[pos] = cur
    d["SEG_ID"] = d["SID"].astype(str) + "#" + pd.Series(run, index=d.index).astype(str)
    d["SEG_LEN"] = d.groupby("SEG_ID")["SEG_ID"].transform("size")
    d["SEG_POS"] = d.groupby("SEG_ID").cumcount()
    return d


def assign_split(season: int, splits: dict[str, list]) -> str:
    """Map a storm season to its split name (train/validation/test)."""
    for name, (a, b) in splits.items():
        if a <= season <= b:
            return name
    raise ValueError(f"Season {season} is outside all configured splits: {splits}")
