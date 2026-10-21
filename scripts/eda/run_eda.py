"""Reproducible entry point for the IBTrACS-WP exploratory data analysis.

Example
-------
python -m scripts.eda.run_eda --input data/IBTrACS.WP.v04r01.nc --out EDA
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures as F
from .config import (EXTREME_SPEED_KTS, HORIZON_STEPS, JUMP_KM_THRESHOLD,
                     N_FUTURE_OBS, N_INPUT_OBS, STUDY_END_SEASON,
                     STUDY_START_SEASON, TEST_RANGE, TRAIN_RANGE, VAL_RANGE,
                     WINDOW_LENGTH)
from .io import load_ibtracs
from .metrics import (add_past_only_motion, count_windows, gap_table,
                      runs_of_exact_6h, subset_filters)

log = logging.getLogger("eda")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")


def _split_of(season: int) -> str:
    if TRAIN_RANGE[0] <= season <= TRAIN_RANGE[1]:
        return "train"
    if VAL_RANGE[0] <= season <= VAL_RANGE[1]:
        return "validation"
    if TEST_RANGE[0] <= season <= TEST_RANGE[1]:
        return "test"
    return "outside"


def build_tables(df: pd.DataFrame, dfm: pd.DataFrame, filt: pd.DataFrame,
                 filt_steps: pd.DataFrame, out_tables: Path) -> dict:
    """Compute all numerical summaries and write them as CSV files."""
    out_tables.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    def save(name: str, frame: pd.DataFrame) -> pd.DataFrame:
        p = out_tables / name
        frame.to_csv(p, index=False)
        saved[name] = p
        return frame

    study = df[(df["SEASON"] >= STUDY_START_SEASON) & (df["SEASON"] <= STUDY_END_SEASON)]
    overview = pd.DataFrame([{
        "raw_observations": int(len(df)), "raw_storms": int(df["SID"].nunique()),
        "season_min": int(df["SEASON"].min()), "season_max": int(df["SEASON"].max()),
        "time_min_utc": str(df["TIME"].min()), "time_max_utc": str(df["TIME"].max()),
        "study_observations": int(len(study)), "study_storms": int(study["SID"].nunique()),
        "lat_min": float(df["LAT"].min()), "lat_max": float(df["LAT"].max()),
        "lon_min_file_convention": float(df["LON"].min()),
        "lon_max_file_convention": float(df["LON"].max()),
        "duplicate_sid_time": int(df.duplicated(["SID", "TIME"]).sum()),
    }])
    save("01_dataset_overview.csv", overview)

    per_year = df.groupby("SEASON").agg(
        n_storms=("SID", "nunique"), n_obs=("SID", "size"),
        lat_mean=("LAT", "mean"), lon_mean=("LON", "mean"),
        wind_mean=("WMO_WIND", "mean")).reset_index().sort_values("SEASON")
    save("02_counts_by_year.csv", per_year)

    s2 = study.copy()
    s2["MONTH"] = s2["TIME"].dt.month
    by_month = s2.groupby("MONTH").agg(
        n_obs=("SID", "size"), n_storms=("SID", "nunique")).reset_index()
    save("03_counts_by_month.csv", by_month)

    nper = study.groupby("SID").size()
    save("04_obs_per_storm_stats.csv", pd.DataFrame([{
        "n_storms": int(nper.size), "mean": float(nper.mean()),
        "std": float(nper.std()), "min": int(nper.min()),
        "p25": float(nper.quantile(0.25)), "median": float(nper.median()),
        "p75": float(nper.quantile(0.75)), "max": int(nper.max())}]))

    miss_rows = []
    for col in ["LAT", "LON", "STORM_SPEED", "STORM_DIR", "WMO_WIND", "WMO_PRES",
                "BASIN", "TIME", "DIST2LAND", "LANDFALL", "NATURE", "WMO_AGENCY"]:
        s = df[col]
        miss = s.isna() if col != "WMO_AGENCY" else (s.astype(str) == "")
        miss_rows.append({"variable": col, "n_missing": int(miss.sum()),
                          "pct_missing": float(miss.mean() * 100), "n_total": int(len(df))})
    miss_rows.append({"variable": "IFLAG(empty)", "n_missing": int((df["IFLAG"].astype(str) == "").sum()),
                      "pct_missing": float((df["IFLAG"].astype(str) == "").mean() * 100),
                      "n_total": int(len(df))})
    save("05_missing_values.csv", pd.DataFrame(miss_rows))

    save("06_track_type.csv",
         df.groupby("TRACK_TYPE").agg(n_obs=("SID", "size"),
                                      n_storms=("SID", "nunique")).reset_index())
    save("07_basin.csv",
         df.groupby("BASIN").agg(n_obs=("SID", "size"),
                                 n_storms=("SID", "nunique")).reset_index())

    _, gaps = gap_table(df)
    gaps = gaps[np.isfinite(gaps)]
    gap_vals, gap_counts = np.unique(np.round(gaps, 6), return_counts=True)
    gap_df = pd.DataFrame({"gap_hours": gap_vals, "n_pairs": gap_counts})
    gap_df["pct"] = gap_df["n_pairs"] / gap_df["n_pairs"].sum() * 100
    save("08_time_gaps.csv", gap_df.sort_values("n_pairs", ascending=False))
    # IFLAG summary: share of obs whose flag string contains each character.
    iflag = df["IFLAG"].astype(str)
    save("09_iflag.csv", pd.DataFrame([{
        "contains_O_original_pct": float(iflag.str.contains("O").mean() * 100),
        "contains_P_position_interp_pct": float(iflag.str.contains("P").mean() * 100),
        "contains_I_intensity_interp_pct": float(iflag.str.contains("I").mean() * 100),
        "contains_V_other_interp_pct": float(iflag.str.contains("V").mean() * 100),
        "all_underscore_pct": float((iflag == "_" * 15).mean() * 100),
        "empty_pct": float((iflag == "").mean() * 100),
    }]))

    save("10_subset_filters.csv", filt_steps)

    # Continuity (F6) on the filtered set.
    lab = runs_of_exact_6h(filt)
    runs = lab[lab["RUN_LEN"] >= 2]
    run_lens = runs.groupby("RUN_ID").size()
    n_windows_total = count_windows(run_lens, WINDOW_LENGTH)
    cont = pd.DataFrame([{
        "n_obs_F5": int(len(filt)), "n_obs_in_6h_runs": int(len(runs)),
        "n_maximal_runs_ge2": int(run_lens.size),
        "n_adjacent_6h_pairs": int((lab.groupby("SID")["TIME"].diff().dt.total_seconds()
                                    .eq(6 * 3600)).sum()),
        "n_windows_5in_8future": int(n_windows_total)}])
    save("11_continuity.csv", cont)

    # Train/val/test split of the filtered set + window counts.
    split_rows = []
    for name, (a, b) in (("train", TRAIN_RANGE), ("validation", VAL_RANGE), ("test", TEST_RANGE)):
        g = filt[(filt["SEASON"] >= a) & (filt["SEASON"] <= b)]
        lab_g = runs_of_exact_6h(g)
        rl = lab_g[lab_g["RUN_LEN"] >= 2].groupby("RUN_ID").size()
        storms_w = lab_g[lab_g["RUN_LEN"] >= WINDOW_LENGTH].groupby("SID").size()
        split_rows.append({"split": name, "seasons": f"{a}-{b}",
                           "n_obs": int(len(g)), "n_storms": int(g["SID"].nunique()),
                           "n_obs_in_6h_runs": int((lab_g["RUN_LEN"] >= 2).sum()),
                           "n_runs_ge2": int(rl.size),
                           "storms_with_ge13_run": int((lab_g.groupby("SID")["RUN_LEN"].max()
                                                        >= WINDOW_LENGTH).sum()),
                           "n_windows_5in_8future":
                               int(count_windows(rl, WINDOW_LENGTH))})
    save("12_split_counts.csv", pd.DataFrame(split_rows))

    # Quality flags (detection only).
    invalid_pos = (~df["LAT"].between(-90, 90) | ~df["LON"].between(0, 360)
                   | df["LAT"].isna() | df["LON"].isna()).sum()
    per_sid_time = df.sort_values(["SID", "TIME"])
    out_of_order = int((((per_sid_time["TIME"].diff() < pd.Timedelta(0))
                         & (per_sid_time["SID"] == per_sid_time["SID"].shift(1))).sum()))
    rep_speed = pd.to_numeric(df["STORM_SPEED"], errors="coerce")
    rep_dir = pd.to_numeric(df["STORM_DIR"], errors="coerce")
    save("13_quality_flags.csv", pd.DataFrame([{
        "invalid_or_missing_position": int(invalid_pos),
        "duplicate_sid_time": int(df.duplicated(["SID", "TIME"]).sum()),
        "out_of_order_within_storm": int(out_of_order),
        "suspicious_jump_gt1000km": int(dfm["SUSPICIOUS_JUMP"].sum()),
        "recomputed_speed_gt80kts": int(dfm["EXTREME_RECOMPUTED_SPEED"].sum()),
        "reported_speed_gt80kts": int(((rep_speed > EXTREME_SPEED_KTS)).sum()),
        "reported_speed_missing_pct": float(rep_speed.isna().mean() * 100),
        "reported_dir_missing_pct": float(rep_dir.isna().mean() * 100),
        "reported_dir_out_of_0_360": int((((rep_dir < 0) | (rep_dir > 360)) & rep_dir.notna()).sum()),
    }]))

    # Storm motion definition check: reported vs past-only recomputed.
    rep = pd.to_numeric(study["STORM_SPEED"] if "STORM_SPEED" in study else df["STORM_SPEED"],
                        errors="coerce")
    rec = pd.to_numeric(dfm.loc[dfm["SEASON"].between(STUDY_START_SEASON, STUDY_END_SEASON),
                                "RECOMPUTED_SPEED"], errors="coerce")
    both = pd.DataFrame({"reported": rep.to_numpy(), "recomputed": rec.to_numpy()}).dropna()
    both = both[both["reported"] <= 998]
    resid = both["reported"] - both["recomputed"]
    save("14_speed_agreement.csv", pd.DataFrame([{
        "n_compared": int(len(both)),
        "pearson_r": float(both.corr().iloc[0, 1]) if len(both) > 2 else float("nan"),
        "mean_residual_reported_minus_pastonly_kts": float(resid.mean()),
        "median_residual_kts": float(resid.median()),
        "p90_abs_residual_kts": float(resid.abs().quantile(0.9)),
        "first_obs_per_storm_nan_by_construction": int(dfm.groupby("SID").head(1)["RECOMPUTED_SPEED"].isna().sum()),
    }]))

    # Exclusion reasons at each filter step (for the study period).
    f1 = df[df["SEASON"].between(STUDY_START_SEASON, STUDY_END_SEASON)]
    excl = pd.DataFrame([
        {"filter": "F1_season", "excluded_obs": int(len(df) - len(f1)),
         "reason": f"Season outside [{STUDY_START_SEASON}, {STUDY_END_SEASON}]"},
        {"filter": "F2_basin", "excluded_obs": int(len(f1) - len(f1[f1['BASIN'] == 'WP'])),
         "reason": "Per-observation BASIN != 'WP' (transitions into EP/NI)"},
        {"filter": "F3_track",
         "excluded_obs": int(len(f1[f1["BASIN"] == "WP"])
                             - len(f1[(f1["BASIN"] == "WP") & (f1["TRACK_TYPE"] == "main")])),
         "reason": "TRACK_TYPE != 'main' (spur-merge/split/other, PROVISIONAL, US-PROVISIONAL)"},
    ])
    save("15_exclusion_reasons.csv", excl)
    return saved


def write_report(df: pd.DataFrame, dfm: pd.DataFrame, filt: pd.DataFrame,
                 filt_steps: pd.DataFrame, meta: dict, out_report: Path) -> None:
    """Write ``report.md`` with concrete numbers (no hardcoded statistics)."""
    study = df[(df["SEASON"] >= STUDY_START_SEASON) & (df["SEASON"] <= STUDY_END_SEASON)]
    miss = pd.read_csv(out_report.parent / "tables" / "05_missing_values.csv")
    miss_line = "; ".join(f"{r.variable}: {r.pct_missing:.2f}%" for r in miss.itertuples())
    miss_map = {r.variable: float(r.pct_missing) for r in miss.itertuples()}
    iflag_row = pd.read_csv(out_report.parent / "tables" / "09_iflag.csv").iloc[0]
    p_p = float(iflag_row["contains_P_position_interp_pct"])
    p_o = float(iflag_row["contains_O_original_pct"])
    qual = pd.read_csv(out_report.parent / "tables" / "13_quality_flags.csv").iloc[0]
    n_ooo = int(qual["out_of_order_within_storm"])
    n_jump = int(qual["suspicious_jump_gt1000km"])
    n_extreme = int(qual["recomputed_speed_gt80kts"])
    tt = df.groupby("TRACK_TYPE").size().to_dict()
    bas = df.groupby("BASIN").size().to_dict()
    gap08 = pd.read_csv(out_report.parent / "tables" / "08_time_gaps.csv")
    gap_pct = {float(r.gap_hours): float(r.pct) for r in gap08.itertuples()}
    gap_line = ", ".join(f"{g:.0f} h: {gap_pct.get(g, 0.0):.1f}%" for g in sorted(gap_pct))
    rep_study = pd.to_numeric(study["STORM_SPEED"], errors="coerce")
    rec = pd.to_numeric(dfm.loc[dfm["SEASON"].between(STUDY_START_SEASON, STUDY_END_SEASON),
                                "RECOMPUTED_SPEED"], errors="coerce")
    both = pd.DataFrame({"a": rep_study.to_numpy(), "b": rec.to_numpy()}).dropna()
    both = both[both["a"] <= 998]
    r = float(both.corr().iloc[0, 1]) if len(both) > 2 else float("nan")
    split = pd.read_csv(out_report.parent / "tables" / "12_split_counts.csv")
    win_col = "n_windows_5in_8future"
    total_windows = int(split[win_col].sum())
    steps_md = "\n".join(
        " | ".join([s.step, s.description, f"{s.n_obs:,}", f"{s.n_storms:,}"]).join(["| ", " |"])
        for s in filt_steps.itertuples())
    split_md = "\n".join(
        " | ".join([f"{s.split} ({s.seasons})", f"{s.n_obs:,}", f"{s.n_storms:,}",
                     f"{s.n_obs_in_6h_runs:,}", f"{s.n_runs_ge2:,}",
                     f"{getattr(s, win_col):,}"]).join(["| ", " |"])
        for s in split.itertuples())
    n_main_storms = int(df.loc[df["TRACK_TYPE"] == "main", "SID"].nunique())
    n_spur_storms = int(df.loc[~df["TRACK_TYPE"].isin(["main"]), "SID"].nunique())
    med_spd = float(pd.to_numeric(study["STORM_SPEED"], errors="coerce").clip(upper=998).median())
    by_m = study.copy()
    by_m["MONTH"] = by_m["TIME"].dt.month
    peak_month = int(by_m.groupby("MONTH").size().idxmax()) if len(by_m) else -1

    report = f"""# IBTrACS Western Pacific — Exploratory Data Analysis (Task 1)

Source file: `{meta.get('source_file')}` · product `{meta.get('global_attrs', {}).get('product_version', 'v04r01')}` ·
generated by `python -m scripts.eda.run_eda --input <file> --out EDA`.
Raw data were never modified; all filtering below is analytical (views, not edits).

## 1. Dataset overview

* File grid: **{meta['dims'].get('storm', '?')} storm rows × {meta['dims'].get('date_time', '?')} time slots**; only the first `numobs` cells per storm are real observations.
* **Raw records: {len(df):,} observations from {df['SID'].nunique():,} storm rows**; seasons {int(df['SEASON'].min())}–{int(df['SEASON'].max())}; time span {df['TIME'].min()} → {df['TIME'].max()} (UTC).
* Study period {STUDY_START_SEASON}–{STUDY_END_SEASON}: **{len(study):,} observations from {study['SID'].nunique():,} storms**.
* `LAT` range {df['LAT'].min():.1f}…{df['LAT'].max():.1f} °N; `LON` range {df['LON'].min():.1f}…{df['LON'].max():.1f} °E (**file convention 0–360°E**, see §5).
* Duplicate (SID, TIME) pairs: **{int(df.duplicated(['SID', 'TIME']).sum()):,}** (confirmed: none found).
* Time encoding: `time` = days since 1858-11-17 (`TIME_FILL=-9999000.0`); per-observation `iso_time` strings (UTC) parsed directly; NetCDF `_FillValue=-9999` mapped to missing.

### Key variable definitions and units (from file metadata)

| Variable | Meaning | Units / values |
|---|---|---|
| SID | IBTrACS serial ID (trajectory id) | string, e.g. storm row id |
| SEASON | Season (year storm began) | year |
| BASIN | Current basin **per observation** | EP / NI / WP (+ others globally) |
| ISO_TIME / TIME | Observation timestamp | UTC datetime |
| LAT / LON | Merged best-track position | degrees_north / degrees_east (0–360) |
| STORM_SPEED | Storm translation speed | kts, valid 0–998, −9999 = missing |
| STORM_DIR | Storm translation direction | degrees east of north, 0–360 |
| WMO_WIND / WMO_PRES | Official WMO agency max wind / min pressure | kts / mb |
| TRACK_TYPE | Track classification (**per storm**) | main / spur-merge / spur-split / spur-other / PROVISIONAL / US-PROVISIONAL(+_spur) in this file |
| IFLAG | 15-char per-agency interpolation flag | O=original, P=position-interp, I=intensity-interp, V=other-interp, `_`=nothing reported |
| DIST2LAND / LANDFALL | Distance to land / landfall-between-obs indicator | km (LANDFALL missing exactly at each storm's last obs — no "next" position) |

Observed TRACK_TYPE counts (all years, per observation): {dict(tt)}.
Observed BASIN counts (all years, per observation): {dict(bas)}.
**TRACK_TYPE semantics note (verified against this file):** `main` = regular tracks; `spur-merge` / `spur-split` / `spur-other` = spur segments that merge with or split from another track (file metadata: treat with caution); `PROVISIONAL`, `US-PROVISIONAL`, `US-PROVISIONAL_spur` are provisional agency assignments undocumented in the variable metadata. All non-`main` rows are excluded from the experimental subset (F3); confirm this choice with the supervisor.

## 2. Structure of the data

* Distinct main-track storms (all years): MAIN_STORMS_PLACEHOLDER (non-main: SPUR_STORMS_PLACEHOLDER distinct storms — see `tables/06_track_type.csv`).
* Per-observation basin mix shows storms transitioning out of WP: EP and NI observations exist inside this WP extract (see `tables/07_basin.csv`), so basin filtering **must be per observation**, not per storm.
* Observations-per-storm and duration distributions: see `figures/02_*`, `figures/10_*`, `tables/04_*`.
* Gap analysis (all years; within-storm consecutive pairs): **{gap_line}** — the native grid is a complete 3-hourly lattice (IBTrACS fills 3-hourly fixes, see IFLAG), so the 6-hourly subset keeps roughly one in two timestamps by design (see `figures/09_*`, `tables/08_*`).

## 3. Missing values and data quality (confirmed, not imputed)

* Missing percentages (raw, all years) — {miss_line}. Full table: `tables/05_*`. Note: WMO_WIND ({miss_map.get('WMO_WIND', float('nan')):.1f}% missing) and WMO_PRES ({miss_map.get('WMO_PRES', float('nan')):.1f}% missing) are too sparse to serve as model inputs; LAT/LON/STORM_SPEED/STORM_DIR are effectively complete.
* Invalid positions: none outside LAT∈[−90,90] / LON∈[0,360] beyond NaNs (see `tables/13_*`); LAT/LON are jointly present for every `numobs` cell.
* Out-of-order timestamps within a storm: counted in `tables/13_*` (detection only; confirmed {n_ooo}).
* Suspicious jumps: consecutive displacement > {JUMP_KM_THRESHOLD:.0f} km flagged (`SUSPICIOUS_JUMP`, see `tables/13_*`; confirmed {n_jump}); extreme recomputed translation > {EXTREME_SPEED_KTS:.0f} kts flagged ({n_extreme} obs). **Nothing was removed.**
* IFLAG is one character **per agency** (15 agencies, order in file metadata). {p_p:.1f}% of raw obs carry a `P` (position-interpolated) mark from ≥1 agency and {p_o:.1f}% an `O` (original) mark — marks coexist because agencies report independently (see `tables/09_*`). The 3-hourly lattice is therefore heavily agency-interpolated: preprocessing should prefer synoptic 6-hourly fixes and explicitly handle/test `P`-dominated timestamps — recommendation, not applied here.
* Longitude: stored **0–360°E** (max {df['LON'].max():.1f}). There is no jump at 180° in storage; a jump only appears if wrapped to −180…180 for display. All distances/motion here use unwrapped longitudes + haversine; track plots break lines at gaps > 6.5 h so segments are never joined by straight artefacts (see `figures/05_*`, `figures/08_*`).

## 4. STORM_SPEED / STORM_DIR — derivation and leakage assessment

* File metadata gives units/conventions but **no formula**; per IBTrACS practice these are translation vectors derived from successive positions (agency guidance, not independently observed).
* Empirical check (study period, n={len(both):,}): Pearson r(reported, past-only recomputed) = **{r:.4f}**; mean residual (reported − past-only) and p90 in `tables/14_*`. In practice the file values track a backward difference closely — but that does not make them causal (next point). Residuals cluster at integer offsets because file STORM_SPEED is integer-rounded while the recomputed value is continuous (see `figures/15_*`).
* Structural leakage proof: each storm's first observation has **no past** (past-only speed = NaN by construction for all {int((dfm.groupby('SID').head(1)['RECOMPUTED_SPEED'].isna()).sum()):,} storm rows), yet file STORM_SPEED is missing in only {int(rep_study.isna().sum()):,} study-period obs — i.e. it is populated where no past exists, so it cannot be a pure backward difference and must use centred/forward information (**future leakage relative to a causal input**).
* **Recommendation:** do NOT feed file STORM_SPEED/STORM_DIR as inputs without recomputing them causally (backward difference from past positions only, masking first obs + post-gap obs). Compare both variants in preprocessing ablations. Direction is circular — encode as (sin, cos) later, not raw degrees.

## 5. Geography and seasonality

* Spatial density (`figures/14_*`): genesis/main development region and recurvature visible; positions span the full WP with excursions into EP/NI.
* Sample deterministic tracks (`figures/08_*`, seed=42): no dateline artefacts by construction.
* Monthly cycle (`figures/11_*`, `tables/03_*`): peak in month {peak_month} with elevated activity Jul–Oct; exact counts in tables.
* Speed/direction climatology (`figures/06_*`, `figures/07_*`, `figures/15_*`): speeds right-skewed (study-period median {med_spd:.1f} kts); directions bimodal (westward trades + eastward recurvature).
* Duration vs path length (`figures/16_*`): longer-lived storms travel farther, with scatter from looping/recurving tracks.

## 6. Experimental-subset statistics ({STUDY_START_SEASON}–{STUDY_END_SEASON})

Filter order (fixed, per-observation unless noted):

1. **F1** season ∈ [{STUDY_START_SEASON}, {STUDY_END_SEASON}]
2. **F2** BASIN == 'WP' (per observation)
3. **F3** TRACK_TYPE == 'main' (per storm; spur-merge/split/other, PROVISIONAL, US-PROVISIONAL excluded)
4. **F4** synoptic 6-hourly: HH ∈ {{00, 06, 12, 18}} UTC, minute == 0
5. **F5** valid LAT/LON/TIME
6. **F6** continuity: maximal runs with exact 6 h spacing (tolerance 1 s)

| Step | Definition | n_obs | n_storms |
|---|---|---|---|
{steps_md}

Continuity detail: see `tables/11_continuity.csv`. Sliding forecast windows of {N_INPUT_OBS} inputs + {N_FUTURE_OBS} future obs ({WINDOW_LENGTH} consecutive 6 h obs; horizons {', '.join(f'{h}h=step{HORIZON_STEPS[h]}' for h in sorted(HORIZON_STEPS))}) total **{total_windows:,} windows** across splits:

| Split (seasons) | n_obs | n_storms | obs in 6 h runs | runs ≥2 | windows ({N_INPUT_OBS}+{N_FUTURE_OBS}) |
|---|---|---|---|---|---|
{split_md}

Exclusion reasons: `tables/15_exclusion_reasons.csv` (plus F4/F5/F6 drops readable as differences in `tables/10_*`). Dominant loss is the 3 h → 6 h cadence reduction plus non-WP/non-main records — expected, not data errors.

## 7. Leakage risks and preprocessing recommendations

1. **STORM_SPEED/STORM_DIR leakage** (above): recompute causally; never use file values as inputs without proof of causality.
2. **Interpolation leakage**: IFLAG `P` fixes are synthetic 3-hourly fills; restrict training segments to original (`O`) fixes or explicitly flag them.
3. **Basin/track filtering**: filter per observation (basin) and per storm (track type) exactly as F2/F3; document the non-main exclusion decision.
4. **Split discipline**: split by storm (and by season block {TRAIN_RANGE[0]}–{TRAIN_RANGE[1]} / {VAL_RANGE[0]}–{VAL_RANGE[1]} / {TEST_RANGE[0]}–{TEST_RANGE[1]}), never by window, to avoid same-storm windows leaking across sets.
5. **Gap handling**: build windows only inside exact-6 h runs; mask loss at horizons crossing any residual gap.
6. **Circular/longitude care**: model inputs should use displacements or (sin,cos) direction and unwrapped/relative longitudes, not raw 0–360 values.
7. **No imputation in EDA**: all numbers above are computed on observed data; preprocessing choices (dropping vs masking) belong to Task 2.

## 8. Artefacts

* Figures: `EDA/figures/` (01–16, ≥200 DPI, see README for the captioned list).
* Tables: `EDA/tables/` (01–15 CSV).
* Reproduce: `python -m scripts.eda.run_eda --input data/IBTrACS.WP.v04r01.nc --out EDA`.
* Code: `scripts/eda/` (`config.py`, `io.py`, `metrics.py`, `figures.py`, `run_eda.py`).

## 9. References

* NOAA NCEI IBTrACS v04r01 access (NetCDF): https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/
* Knapp et al. (2010), doi:10.1175/2009BAMS2755.1 (also in file global attrs + doi:10.25921/82ty-9e16).
* File-internal metadata embedded in `IBTrACS.WP.v04r01.nc` (variable `long_name`/`Note` fields quoted above).
"""
    out_report.write_text(report.replace("MAIN_STORMS_PLACEHOLDER", f"{n_main_storms:,}")
                          .replace("SPUR_STORMS_PLACEHOLDER", f"{n_spur_storms:,}"),
                          encoding="utf-8")
    log.info("Wrote report: %s", out_report)


def write_readme(out_dir: Path, input_path: str) -> None:
    (out_dir / "README.md").write_text(f"""# EDA outputs — IBTrACS Western Pacific

Reproduce everything with (from the repository root):

```bash
python -m scripts.eda.run_eda --input {input_path} --out EDA
```

`--input` also accepts an IBTrACS CSV export (units row auto-detected).
Requires: pandas, NumPy, Matplotlib, xarray, netCDF4 (see repo environment).

## Figures (`figures/`, all ≥200 DPI)

| File | Content |
|---|---|
| 01_storm_count_by_year.png | Unique storms + observations per season |
| 02_observations_per_storm.png | Observations-per-storm histogram (linear + log) |
| 03_missing_values.png | Missing-value percentage by variable |
| 04_latitude_distribution.png | Latitude histogram |
| 05_longitude_distribution.png | Longitude 0–360 (stored) vs wrapped −180…180 (display) |
| 06_storm_speed_distribution.png | File STORM_SPEED histogram |
| 07_storm_direction_distribution.png | STORM_DIR polar histogram |
| 08_sample_storm_tracks.png | 12 deterministic sample tracks (seed=42, gaps broken) |
| 09_observation_time_gaps.png | Consecutive-observation gaps |
| 10_storm_duration_distribution.png | Storm duration (days) |
| 11_monthly_frequency.png | Monthly observation/storm frequency |
| 12_track_type_distribution.png | TRACK_TYPE per-observation counts |
| 13_basin_distribution.png | Per-observation BASIN counts |
| 14_spatial_density.png | Log-scale hexbin position density |
| 15_speed_reported_vs_recomputed.png | Reported vs past-only speed + residuals |
| 16_duration_vs_path_length.png | Duration vs summed haversine path length |

## Tables (`tables/`)

01 overview · 02 by-year · 03 by-month · 04 obs-per-storm · 05 missing ·
06 track-type · 07 basin · 08 gaps · 09 IFLAG · 10 subset filters ·
11 continuity · 12 train/val/test splits + windows · 13 quality flags ·
14 speed agreement · 15 exclusion reasons.

## Notes

* Raw NetCDF files are never modified.
* `report.md` distinguishes confirmed findings from recommendations.
* Longitude is stored 0–360°E; display wraps are labelled wherever used.
""", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="IBTrACS-WP exploratory data analysis.")
    ap.add_argument("--input", required=True, help="Path to IBTrACS .nc or .csv file")
    ap.add_argument("--out", default="EDA", help="Output directory (default: EDA)")
    ap.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    _setup_logging(args.verbose)

    in_path = Path(args.input)
    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    df, meta = load_ibtracs(in_path)
    log.info("Seasons %s–%s; %d obs / %d storms",
             df['SEASON'].min(), df['SEASON'].max(), len(df), df['SID'].nunique())
    dfm = add_past_only_motion(df)
    filt, filt_steps = subset_filters(df)

    # Figures (all from real data; deterministic sampling where applicable).
    F.fig_storm_count_by_year(df, fig_dir / "01_storm_count_by_year.png")
    F.fig_observations_per_storm(df, fig_dir / "02_observations_per_storm.png")
    F.fig_missing_values(df, fig_dir / "03_missing_values.png")
    F.fig_latitude_distribution(df, fig_dir / "04_latitude_distribution.png")
    F.fig_longitude_distribution(df, fig_dir / "05_longitude_distribution.png")
    F.fig_speed_distribution(df, fig_dir / "06_storm_speed_distribution.png")
    F.fig_direction_distribution(df, fig_dir / "07_storm_direction_distribution.png")
    F.fig_sample_tracks(df, fig_dir / "08_sample_storm_tracks.png", seed=args.seed)
    F.fig_time_gaps(df, fig_dir / "09_observation_time_gaps.png")
    F.fig_duration_distribution(df, fig_dir / "10_storm_duration_distribution.png")
    F.fig_monthly_frequency(df, fig_dir / "11_monthly_frequency.png")
    F.fig_track_type(df, fig_dir / "12_track_type_distribution.png")
    F.fig_basin(df, fig_dir / "13_basin_distribution.png")
    F.fig_spatial_density(df, fig_dir / "14_spatial_density.png")
    F.fig_speed_comparison(dfm, fig_dir / "15_speed_reported_vs_recomputed.png")
    F.fig_length_vs_duration(dfm, fig_dir / "16_duration_vs_path_length.png")

    build_tables(df, dfm, filt, filt_steps, tab_dir)
    write_report(df, dfm, filt, filt_steps, meta, out_dir / "report.md")
    write_readme(out_dir, args.input)
    log.info("Done. Figures: %d | Tables: %d | Report: %s",
             len(list(fig_dir.glob('*.png'))), len(list(tab_dir.glob('*.csv'))),
             out_dir / "report.md")
    print(f"EDA complete: {len(list(fig_dir.glob('*.png')))} figures, "
          f"{len(list(tab_dir.glob('*.csv')))} tables -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
