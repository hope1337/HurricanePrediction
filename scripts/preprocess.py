"""Reproducible preprocessing entry point (Task 2).

Example:
    python scripts/preprocess.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eda.io import load_ibtracs  # noqa: E402  (reuse Task 1 loader)
from src.data import geometry as G  # noqa: E402
from src.data.features import CONFIG_COLUMNS, CONFIG_NAMES  # noqa: E402
from src.data.preprocessing import (apply_filters, assign_split, sort_and_deduplicate,  # noqa: E402
                                    split_segments)
from src.data.scaling import apply_scaler, fit_scaler  # noqa: E402
from src.data.windowing import FORECAST_STEPS, INPUT_STEPS, WINDOW_OBS, generate_windows  # noqa: E402

log = logging.getLogger("preprocess")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")


def _load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["config_path"] = str(path)
    return cfg


def _save_split_npz(path: Path, X7_scaled: np.ndarray, X7_raw: np.ndarray,
                    out: dict) -> None:
    n = len(out["X7"])
    sid = np.array([m["sid"] for m in out["meta"]], dtype="<U16")
    sample_id = np.array([m["sample_id"] for m in out["meta"]], dtype="<U64")
    season = np.array([m["season"] for m in out["meta"]], dtype=np.int16)
    t0_ns = np.array([pd.Timestamp(m["t0_iso"]).value for m in out["meta"]], dtype=np.int64)
    tgt_ns = np.array([[pd.Timestamp(t).value for t in m["target_iso"]] for m in out["meta"]],
                      dtype=np.int64)
    last = np.array([[m["last_lat"], m["last_lon"]] for m in out["meta"]], dtype=np.float32)
    hist = np.array([np.column_stack([m["hist_lat"], m["hist_lon"]]) for m in out["meta"]],
                    dtype=np.float32)
    np.savez_compressed(
        path,
        X_main=X7_scaled[:, :, CONFIG_COLUMNS["C"]].astype(np.float32),
        X_pos=X7_scaled[:, :, CONFIG_COLUMNS["A"]].astype(np.float32),
        X_spd=X7_scaled[:, :, CONFIG_COLUMNS["B"]].astype(np.float32),
        X_raw7=X7_raw.astype(np.float32),
        Y=np.asarray(out["Y"], dtype=np.float32),
        Y_direct=np.asarray(out["Y_direct"], dtype=np.float32),
        Y_step=np.asarray(out["Y_step"], dtype=np.float32),
        sample_id=sample_id, sid=sid, season=season,
        t0_ns=t0_ns, target_ns=tgt_ns, last_latlon=last, hist_latlon=hist,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="IBTrACS-WP preprocessing (Task 2).")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    _setup_logging(args.verbose)

    cfg = _load_config(args.config)
    out_dir = Path(cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    interval = float(cfg["interval_hours"])
    tol = float(cfg["interval_tolerance_seconds"])
    splits = {k: list(v) for k, v in cfg["splits"].items()}

    # 1. Load (reuse EDA NetCDF/CSV loader; raw file never written to).
    df, meta = load_ibtracs(cfg["raw_path"])
    log.info("Loaded %d obs / %d storms from %s", len(df), df["SID"].nunique(), cfg["raw_path"])

    # 2–5. Filter in fixed documented order.
    start, end = cfg["seasons"]
    filt, step_log = apply_filters(df, start, end, cfg["basin"], cfg["track_type"],
                                   tuple(cfg["synoptic_hours"]))
    for _, r in step_log.iterrows():
        log.info("filter %-10s n_obs=%7d n_storms=%5d (%s)", r["step"], r["n_obs"], r["n_storms"], r["description"])

    # 6–7. Sort + deterministic dedup.
    filt, n_dup = sort_and_deduplicate(filt)
    log.info("dedup: dropped %d duplicated (SID, TIME) rows", n_dup)

    # 8. Continuous segments (no interpolation across gaps).
    seg = split_segments(filt, interval, tol)
    n_seg = int(seg["SEG_ID"].nunique())
    seg_len = seg.groupby("SEG_ID").size()
    log.info("segments: %d (len>=14: %d)", n_seg, int((seg_len >= WINDOW_OBS).sum()))

    # 9. Split by storm SEASON; assert SID disjointness across splits.
    seg["split"] = seg["SEASON"].map(lambda s: assign_split(int(s), splits))
    sid_sets = {k: set(seg.loc[seg["split"] == k, "SID"].unique()) for k in splits}
    overlap = (sid_sets["train"] & sid_sets["validation"]) | (sid_sets["train"] & sid_sets["test"]) | (
        sid_sets["validation"] & sid_sets["test"])
    if overlap:
        raise RuntimeError(f"SID leakage across splits: {sorted(overlap)[:5]}")
    log.info("split storms: %s", {k: len(v) for k, v in sid_sets.items()})

    # 10. Windows per split (identical eligible-window definition everywhere).
    results, manifest_rows = {}, []
    for name in ("train", "validation", "test"):
        out = generate_windows(seg[seg["split"] == name].drop(columns=["split"]))
        results[name] = out
        log.info("%-10s windows=%6d storms=%5d", name, len(out["X7"]),
                 len({m["sid"] for m in out["meta"]}))
        for m in out["meta"]:
            manifest_rows.append({"split": name, "sample_id": m["sample_id"], "sid": m["sid"],
                                  "season": m["season"], "seg_id": m["seg_id"],
                                  "t0_iso": m["t0_iso"],
                                  **{f"target_tplus{(i + 1) * 6}h": t
                                     for i, t in enumerate(m["target_iso"])},
                                  "last_lat": m["last_lat"], "last_lon": m["last_lon"]})

    # 11. Scaler: fit on TRAIN inputs only, apply to all splits.
    scaler = fit_scaler(results["train"]["X7"])
    joblib.dump(scaler, out_dir / "scalers.joblib")
    log.info("scaler fitted on train (n=%d); mean=%s", len(results["train"]["X7"]),
             np.array2string(scaler["mean"], precision=3))

    for name in ("train", "validation", "test"):
        _save_split_npz(out_dir / f"{name}.npz",
                        apply_scaler(results[name]["X7"], scaler),
                        results[name]["X7"], results[name])
    pd.DataFrame(manifest_rows).to_csv(out_dir / "sample_manifest.csv", index=False)

    # 12. Config + metadata artefacts.
    cfg_echo = {k: v for k, v in cfg.items()}
    (out_dir / "preprocessing_config.json").write_text(json.dumps(cfg_echo, indent=2), encoding="utf-8")
    eda_cmp = {}
    eda_path = Path(cfg.get("eda_tables_dir", "EDA/tables")) / "12_split_counts.csv"
    if eda_path.exists():
        eda = pd.read_csv(eda_path).set_index("split")
        for name in ("train", "validation", "test"):
            new = len(results[name]["X7"])
            old = int(eda.loc[name, "n_windows_5in_8future"])
            eda_cmp[name] = {"eda_13obs_windows": old, "new_14obs_windows": new,
                             "delta": new - old,
                             "reason": "Task 2 requires 6 history positions + 8 targets (14 obs) "
                                       "vs EDA's 5 + 8 (13 obs); every EDA-eligible segment "
                                       "(>=13 obs) yields exactly one fewer window."}
    metadata = {
        "raw_path": str(cfg["raw_path"]), "geometry_backend": G.BACKEND,
        "filter_steps": step_log.to_dict(orient="records"),
        "n_duplicates_dropped": n_dup,
        "n_segments": n_seg,
        "segments_len_ge14": int((seg_len >= WINDOW_OBS).sum()),
        "splits": {n: {"n_windows": len(results[n]["X7"]),
                       "n_storms": len({m["sid"] for m in results[n]["meta"]}),
                       "seasons": splits[n],
                       "X_main_shape": [len(results[n]["X7"]), INPUT_STEPS, 4],
                       "X_pos_shape": [len(results[n]["X7"]), INPUT_STEPS, 2],
                       "X_spd_shape": [len(results[n]["X7"]), INPUT_STEPS, 5],
                       "Y_shape": [len(results[n]["X7"]), FORECAST_STEPS, 2]}
                   for n in ("train", "validation", "test")},
        "feature_configs": {k: {"name": CONFIG_NAMES[k], "columns": v,
                                "shape": [INPUT_STEPS, len(v)]}
                            for k, v in CONFIG_COLUMNS.items()},
        "velocity_unit": cfg["velocity_unit"], "lon_convention": cfg["lon_convention"],
        "scaler": {"fitted_on": "train", "columns": scaler["columns"],
                   "mean": [float(x) for x in scaler["mean"]],
                   "scale": [float(x) for x in scaler["scale"]]},
        "eda_comparison": eda_cmp,
        "config_sha256": hashlib.sha256(json.dumps(cfg_echo, sort_keys=True).encode()).hexdigest()[:16],
    }
    (out_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(cfg, step_log, n_dup, n_seg, results, splits, scaler, eda_cmp, G.BACKEND)
    log.info("Wrote %s (windows train/val/test = %d/%d/%d)",
             out_dir, *(len(results[n]["X7"]) for n in ("train", "validation", "test")))
    print(f"Preprocessing complete: train={len(results['train']['X7'])} "
          f"val={len(results['validation']['X7'])} test={len(results['test']['X7'])} -> {out_dir}")
    return 0


def write_report(cfg, step_log, n_dup, n_seg, results, splits, scaler, eda_cmp, backend) -> None:
    """Write docs/preprocessing_report.md with measured (never hardcoded) numbers."""
    rows = "\n".join(
        f"| {r['step']} | {r['description']} | {r['n_obs']:,} | {r['n_storms']:,} |"
        for _, r in step_log.iterrows())
    t = results["train"]["X7"]
    per_cfg = {k: f"[{len(t)}, 5, {len(v)}]" for k, v in CONFIG_COLUMNS.items()}
    eda_rows = "\n".join(
        f"| {k} | {v['eda_13obs_windows']:,} | {v['new_14obs_windows']:,} | {v['delta']:+,} |" 
        for k, v in eda_cmp.items()) if eda_cmp else "_EDA table not found; comparison skipped._"
    split_rows = "\n".join(
        f"| {n} ({splits[n][0]}–{splits[n][1]}) | {len(results[n]['X7']):,} | "
        f"{len({m['sid'] for m in results[n]['meta']}):,} |" for n in ("train", "validation", "test"))
    Path("docs").mkdir(exist_ok=True)
    Path("docs/preprocessing_report.md").write_text(f"""# Preprocessing Report — Tropical Cyclone Trajectory Prediction (Task 2)

Config: `configs/data.yaml` · run: `python scripts/preprocess.py --config configs/data.yaml`.
Raw file (`{cfg['raw_path']}`) was only read, never modified. No models trained.

## 1. Filtering (fixed order)

| Step | Definition | n_obs | n_storms |
|---|---|---|---|
{rows}

* Duplicates dropped by deterministic policy (sort by SID/TIME/OBS_IDX, keep first): **{n_dup}**.
* Basin rule: per-observation `BASIN == 'WP'`; obs outside WP are dropped, so storms crossing
  boundaries contribute separate WP-only segments and no window ever spans a transition.
* Continuous segments (exact 6 h, tolerance {cfg['interval_tolerance_seconds']} s): **{n_seg:,}** total;
  short segments (< 14 obs) yield no windows and are excluded from sample counts below.

## 2. Splits and samples

| Split (seasons) | Windows | Storms |
|---|---|---|
{split_rows}

* Split strictly by storm SEASON into {splits['train'][0]}–{splits['train'][1]} /
  {splits['validation'][0]}–{splits['validation'][1]} / {splits['test'][0]}–{splits['test'][1]};
  SID sets verified disjoint across splits (pipeline asserts, tests re-check).

## 3. Features (causal; file STORM_SPEED/STORM_DIR never used)

* Per input row from past-only positions: geodesic distance + forward bearing (WGS84 via
  `{backend}`), speed = distance/dt in **km/h**, `v_E = v·sinθ`, `v_N = v·cosθ` (θ east of north).
* Feature row at time t uses only positions ≤ t; first segment row has NaN motion by construction.
* 6 history positions (t−30…t) produce 5 input rows (t−24…t); targets are the next 8 positions
  (t+6…t+48). All ablation configs share identical sample IDs, origins and ground truth.
* Configs — A position-only {per_cfg['A']}: [LAT, LON]; B {per_cfg['B']}: [LAT, LON, SPEED, sinDIR, cosDIR];
  **C (main) {per_cfg['C']}: [LAT, LON, v_E, v_N]** with SPEED in km/h, DIR circular-encoded.
* X longitude is per-segment unwrapped (continuous; equals stored 0–360°E except across the
  antimeridian); Y and metadata keep original 0–360°E ground truth for Haversine evaluation.

## 4. Targets and shapes

* Per split: `X_main [N, 5, 4]`, `X_pos [N, 5, 2]`, `X_spd [N, 5, 5]`, `Y [N, 8, 2]`
  (train N={len(results['train']['X7']):,}, val N={len(results['validation']['X7']):,},
  test N={len(results['test']['X7']):,}).
* `Y_direct [N, 8, 2]`: east/north km of each target relative to the forecast origin (t),
  in the origin-anchored local frame — for direct multi-horizon forecasting.
* `Y_step [N, 8, 2]`: one-step east/north km between successive positions, each in its own
  step-origin frame — for recursive forecasting (chain Geod-forward steps from t to rebuild).
* Degree differences are never used as distances; all conversions via `{backend}` and
  round-trip tested (synthetic float64 tracks: position ↔ displacement error < 1e−9 deg;
  stored float32 arrays: < 1e−5 deg).

## 5. Normalization (train-only)

* Standard mean/std per feature column fitted on train inputs only
  (mean={np.array2string(scaler['mean'], precision=4, separator=', ')},
  scale={np.array2string(scaler['scale'], precision=4, separator=', ')}); identical transform
  applied to validation/test; persisted in `data/processed/scalers.joblib`; inverse implemented
  in `src/data/scaling.py` and tested. Targets are not normalized.

## 6. Window counts vs EDA

| Split | EDA 13-obs windows | Task-2 14-obs windows | Δ |
|---|---|---|---|
{eda_rows}

Task 2 needs one extra history position (t−30) for causal first-row velocity, so each segment
contributes one fewer window than the EDA estimate — expected, not a data problem.

## 7. Artefacts

* `data/processed/{{train,val,test}}.npz` (arrays + metadata), `sample_manifest.csv`,
  `preprocessing_config.json`, `scalers.joblib`, `dataset_metadata.json`.
* Code: `src/data/{{geometry,preprocessing,features,windowing,scaling,dataset}}.py`,
  `scripts/preprocess.py`, `configs/data.yaml`; tests: `tests/test_preprocessing.py`.
* Interface: `HurricaneTrackDataset(npz_path, config='A'|'B'|'C', target='position'|'direct'|'step')`.

## 8. Remaining limitations

* IFLAG-interpolated 3-hourly fills propagate into synoptic fixes (see EDA §3); positions are
  used as-is, no provenance flag in v1 outputs.
* WMO intensity variables excluded (87.8%/71.2% missing — EDA §3).
* Translation speeds assume great-ellipse/Geod path between fixes; sub-6h wobble unresolved.
* Provisional/spur exclusion follows EDA semantics — confirm with supervisor if scope changes.
""", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
