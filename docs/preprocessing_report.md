# Preprocessing Report — Tropical Cyclone Trajectory Prediction (Task 2)

Config: `configs/data.yaml` · run: `python scripts/preprocess.py --config configs/data.yaml`.
Raw file (`data/IBTrACS.WP.v04r01.nc`) was only read, never modified. No models trained.

## 1. Filtering (fixed order)

| Step | Definition | n_obs | n_storms |
|---|---|---|---|
| raw | All raw records (numobs cells) | 249,084 | 4,255 |
| season | SEASON in [1980, 2024] | 104,927 | 1,471 |
| basin | Per-observation BASIN == 'WP' | 100,970 | 1,471 |
| track | TRACK_TYPE == 'main' | 100,386 | 1,446 |
| synoptic | UTC synoptic hours [0, 6, 12, 18], minute == 0 | 50,674 | 1,446 |
| valid | Finite LAT in [-90,90], LON in [0,360], non-missing TIME | 50,674 | 1,446 |

* Duplicates dropped by deterministic policy (sort by SID/TIME/OBS_IDX, keep first): **0**.
* Basin rule: per-observation `BASIN == 'WP'`; obs outside WP are dropped, so storms crossing
  boundaries contribute separate WP-only segments and no window ever spans a transition.
* Continuous segments (exact 6 h, tolerance 1.0 s): **1,451** total;
  short segments (< 14 obs) yield no windows and are excluded from sample counts below.

## 2. Splits and samples

| Split (seasons) | Windows | Storms |
|---|---|---|
| train (1980–2016) | 27,719 | 1,078 |
| validation (2017–2020) | 2,499 | 118 |
| test (2021–2024) | 2,207 | 98 |

* Split strictly by storm SEASON into 1980–2016 /
  2017–2020 / 2021–2024;
  SID sets verified disjoint across splits (pipeline asserts, tests re-check).

## 3. Features (causal; file STORM_SPEED/STORM_DIR never used)

* Per input row from past-only positions: geodesic distance + forward bearing (WGS84 via
  `pyproj.Geod(WGS84)`), speed = distance/dt in **km/h**, `v_E = v·sinθ`, `v_N = v·cosθ` (θ east of north).
* Feature row at time t uses only positions ≤ t; first segment row has NaN motion by construction.
* 6 history positions (t−30…t) produce 5 input rows (t−24…t); targets are the next 8 positions
  (t+6…t+48). All ablation configs share identical sample IDs, origins and ground truth.
* Configs — A position-only [27719, 5, 2]: [LAT, LON]; B [27719, 5, 5]: [LAT, LON, SPEED, sinDIR, cosDIR];
  **C (main) [27719, 5, 4]: [LAT, LON, v_E, v_N]** with SPEED in km/h, DIR circular-encoded.
* X longitude is per-segment unwrapped (continuous; equals stored 0–360°E except across the
  antimeridian); Y and metadata keep original 0–360°E ground truth for Haversine evaluation.

## 4. Targets and shapes

* Per split: `X_main [N, 5, 4]`, `X_pos [N, 5, 2]`, `X_spd [N, 5, 5]`, `Y [N, 8, 2]`
  (train N=27,719, val N=2,499,
  test N=2,207).
* `Y_direct [N, 8, 2]`: east/north km of each target relative to the forecast origin (t),
  in the origin-anchored local frame — for direct multi-horizon forecasting.
* `Y_step [N, 8, 2]`: one-step east/north km between successive positions, each in its own
  step-origin frame — for recursive forecasting (chain Geod-forward steps from t to rebuild).
* Degree differences are never used as distances; all conversions via `pyproj.Geod(WGS84)` and
  round-trip tested (synthetic float64 tracks: position ↔ displacement error < 1e−9 deg;
  stored float32 arrays: < 1e−5 deg).

## 5. Normalization (train-only)

* Standard mean/std per feature column fitted on train inputs only
  (mean=[ 18.3563, 137.5026,  -8.039 ,   8.0593,  18.3796,  -0.4579,   0.415 ],
  scale=[ 8.4342, 15.1296, 14.8642,  9.5207, 10.1663,  0.6457,  0.4485]); identical transform
  applied to validation/test; persisted in `data/processed/scalers.joblib`; inverse implemented
  in `src/data/scaling.py` and tested. Targets are not normalized.

## 6. Window counts vs EDA

| Split | EDA 13-obs windows | Task-2 14-obs windows | Δ |
|---|---|---|---|
| train | 28,823 | 27,719 | -1,104 |
| validation | 2,620 | 2,499 | -121 |
| test | 2,307 | 2,207 | -100 |

Task 2 needs one extra history position (t−30) for causal first-row velocity, so each segment
contributes one fewer window than the EDA estimate — expected, not a data problem.

## 7. Artefacts

* `data/processed/{train,val,test}.npz` (arrays + metadata), `sample_manifest.csv`,
  `preprocessing_config.json`, `scalers.joblib`, `dataset_metadata.json`.
* Code: `src/data/{geometry,preprocessing,features,windowing,scaling,dataset}.py`,
  `scripts/preprocess.py`, `configs/data.yaml`; tests: `tests/test_preprocessing.py`.
* Interface: `HurricaneTrackDataset(npz_path, config='A'|'B'|'C', target='position'|'direct'|'step')`.

## 8. Remaining limitations

* IFLAG-interpolated 3-hourly fills propagate into synoptic fixes (see EDA §3); positions are
  used as-is, no provenance flag in v1 outputs.
* WMO intensity variables excluded (87.8%/71.2% missing — EDA §3).
* Translation speeds assume great-ellipse/Geod path between fixes; sub-6h wobble unresolved.
* Provisional/spur exclusion follows EDA semantics — confirm with supervisor if scope changes.
