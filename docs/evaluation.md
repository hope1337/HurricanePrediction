# Evaluation Metrics — Trajectory Forecasting (Task 4)

Code: `src/evaluation/{metrics,evaluator,analysis}.py` · geometry:
`src/data/geometry.py::haversine_distance_km` · tests: `tests/test_metrics.py`
(16 passed, synthetic data only). No training loop, no model fitting.

## 1. Why Haversine

Storm positions are points on a sphere, so Euclidean MSE over latitude/longitude
degrees is physically meaningless (a degree of longitude shrinks to zero at the
poles, and the ±180° meridian wraps). The Haversine great-circle distance gives
the shortest surface path in kilometres, is symmetric, handles the antimeridian
by wrapping Δlon to [−180°, 180°), and is numerically stabilized by clipping the
`a` term to [0, 1]. We use the explicit spherical mean radius **R = 6371.0088 km**
(documented constant, shared with the preprocessing fallback) so every reported
error is exactly reproducible from the formula below — no ellipsoid dependence.

```text
a = sin²(Δφ/2) + cos φ1 · cos φ2 · sin²(Δλ/2);  d = 2R · asin(√clip(a, 0, 1))
```

`src/data/geometry.py` already contained this verified core (`_haversine_inverse`,
also cross-checked against `pyproj.Geod` in Task 2 tests); Task 4 exposes it as
`haversine_distance_km` rather than duplicating the implementation.

## 2. Metric definitions (H = 8 steps, d(i,k) = Haversine km)

| Metric | Formula | Reported as |
|---|---|---|
| MTE(k) | `mean_i d(i,k)` | +6h (k=0), +12h (k=1), +24h (k=3), +48h (k=7); full 8-step curve kept |
| ADE | `mean over all N·H (i,k)` | single value over **all 8 steps** |
| FDE | `mean_i d(i,H)` | single value; **equals MTE(+48h)** on identical samples (tested) |
| Median / P90 by horizon | robust counterparts of MTE | same four horizons |

MTE answers "typical error *at* a lead time" (bias/variance per horizon); ADE
summarizes whole-trajectory quality in one number (uses all 8 steps, not just the
four report horizons, so mid-range skill counts); FDE isolates endpoint skill at
+48h, the hardest horizon.

## 3. Aggregation: per-window (primary) vs per-storm

Primary convention averages over **forecast windows** (each sliding window is one
sample; storms with long tracks contribute more windows). Because overlapping
windows from one storm are correlated, `per_storm` summaries (mean window-ADE per
storm + window counts) are returned alongside, so downstream analysis can
re-weight or stratify by storm instead of letting a few long-lived storms
dominate. Every aggregate reports its contributing sample count (`n`, `n_by_horizon`).

## 4. Invalid predictions

Invalid entries are handled only via an explicit boolean `valid_mask [N, H]`:
masked entries are excluded and the reduced counts reported. NaN/inf inputs
*without* a mask, NaNs *inside* the mask, out-of-range latitudes, and shape
mismatches all raise `ValueError` — nothing is silently ignored and the
evaluation population can never change unnoticed between models.

## 5. Consistent model comparison

`evaluate_trajectories(predictions[N,8,2], targets[N,8,2], storm_ids, sample_ids,
init_times?)` works identically for all six RNN/GRU/LSTM × recursive/direct
baselines (it is decoupled from architectures). `assert_consistent_samples`
requires identical sample IDs in identical order before any comparison and
rejects duplicates — different populations or shuffled rows raise instead of
producing misleading deltas.

## 6. Error analysis support (`analysis.py`)

`per_sample_frame` flattens a result to per-sample `[N, H]` errors + ADE with
identifiers for downstream joins; `group_evaluate(errors, groups)` recomputes
ADE/FDE/MTE per caller-supplied label (e.g. straight vs turning tracks, defined
in the experiment stage from history available at initialization). The module
imposes **no thresholds** — grouping logic lives with the experiments.

## 7. Verification

`tests/test_metrics.py` covers: zero self-distance; 1° latitude ≈ 111.195 km;
antimeridian equivalence across lon conventions; symmetry; km (not degree) scale;
ADE = 50 / FDE = 95 on the manual `[20, 35, 50, 95]` example; FDE ≡ last-step MTE;
horizon indexing (error injected only at step 8 surfaces solely at +48h); batch
sizes 1/7/64; six mismatch/duplicate/order rejections; tiny-distance and
near-antipodal (≈πR) stability. Sanity on real validation data (naive persistence
baseline, N=2499): MTE 121 → 952 km across horizons, ADE 538 km — monotone
degradation as expected.
