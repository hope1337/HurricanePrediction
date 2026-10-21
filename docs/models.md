# Models — RNN / GRU / LSTM Trajectory Forecasting (Task 3)

Code: `src/models/{recurrent,heads,forecasting,factory}.py` · config: `configs/models.yaml` ·
tests: `tests/test_models.py` (65 passed, synthetic tensors only) ·
demo: `python scripts/demo_models.py`. No training loop, no real experiments yet.

## 1. What was built

| Component | File | Role |
|---|---|---|
| `RecurrentBackbone` | `recurrent.py` | Configurable RNN/GRU/LSTM, batch-first, **unidirectional**, returns last-layer final hidden state `[B, H]`. Vanilla RNN uses `tanh` (documented default). |
| `OneStepDisplacementHead` | `heads.py` | Linear `H → 2`: one 6-hour ENU displacement (km), for teacher-forced training + recursive rollout. |
| `DirectDisplacementHead` | `heads.py` | Linear `H → 8·2`, reshaped `[B, 8, 2]`: all horizons at once, origin-anchored. |
| `DirectForecaster` | `forecasting.py` | Single forward pass, no feedback. |
| `RecursiveForecaster` | `forecasting.py` | `forward_onestep` (grad-safe, for training) + `rollout` (inference-only, decoded positions). |
| `build_model` / `count_parameters` | `factory.py` | `build_model(architecture, strategy, input_size, hidden_size, num_layers, dropout, forecast_steps, ...)` for all six combos. |

Heads are single linear maps, identical across backbones, so comparisons stay fair.
No attention, transformers, CNNs, or pretrained weights.

## 2. Representation contract (never mixed silently)

| Name | Shape | Units / convention |
|---|---|---|
| `X_norm` (model input) | `[B, 5, D]`, D ∈ {2, 4, 5} for configs A/B/C | Standardized with the **train-fitted** Task 2 scaler |
| `disp_km` (model output) | direct `[B, 8, 2]`, one-step `[B, 2]` | Physical km, east/north; direct = origin-anchored, one-step = step-origin frame |
| `positions` (decoded) | `[B, 8, 2]` | `(lat, lon)`, lon in file convention **0–360°E**, matching Task 2 `Y` |
| Ground truth `Y` | `[B, 8, 2]` | Same geographic convention, unnormalized |

Recursive rollout rebuilds each step's causal motion features from the **predicted**
displacement only (`A`: lat/lon; `B`: speed/sin/cos; `C`: v_E/v_N), renormalizes with the
train scaler, and slides the window. `rollout()` takes **no ground-truth argument** —
leakage is impossible by construction (verified by test). Geometry/feature ops run in
NumPy, so rollout is inference-only; recursive models are trained via `forward_onestep`
with teacher forcing (Task 4).

## 3. Baseline hyperparameters (`configs/models.yaml`)

`hidden_size=64`, `num_layers=2`, `dropout=0.1`, `forecast_steps=8`,
`interval_hours=6.0`, unidirectional, RNN nonlinearity `tanh`. These are starting
values, not tuned optima; nothing is hardcoded inside model classes.

## 4. Trainable parameter counts (measured, baseline, `input_size=4`)

| Architecture | direct | recursive | Δ (head size only) |
|---|---|---|---|
| rnn | 13,840 | 12,930 | 910 |
| gru | 39,440 | 38,530 | 910 |
| lstm | 52,240 | 51,330 | 910 |

Regenerate: `python scripts/demo_models.py`. Ordering `lstm > gru > rnn` reflects gate
count (4/3/1) at identical hidden sizes; strategy Δ is exactly the direct-head
(`64·16+16`) minus one-step-head (`64·2+2`) difference. Exact equality is not required
and not claimed — backbones, inputs, horizons, and heads are otherwise identical.

## 5. Decoding

- Direct: `decode_direct(disp_km[B,8,2], origin_latlon[B,2]) → [B,8,2]` via WGS84
  Geod-forward; origins come from the npz `last_latlon` (see
  `HurricaneTrackDataset(..., return_origin=True)`).
- Recursive: `rollout` returns already-decoded positions; intermediate displacements
  stay in the step-origin frame (chain Geod-forward from the origin to rebuild).

## 6. Assumptions and limits

- Units fixed: km for displacements/speed-derived features, km/h for v_E/v_N,
  degrees east-of-north for bearings, 0–360°E stored lon, unwrapped lon in X.
- Rollout re-encodes the sliding window each step (stateless); error compounds over
  8 steps — expected, to be measured in Task 4.
- No NaN/inf for representative scaled inputs (tested); absurd displacements still
  decode (Geod is global) but will score poorly — training must regularize.
- Task 2 artefacts unchanged except one additive, backward-compatible Dataset flag
  (`return_origin`); all 15 preprocessing tests still pass.
