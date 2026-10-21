"""Minimal runnable example for Task 3 models (synthetic batch, no training).

Demonstrates: constructing each model, forward passes, reading predictions,
recursive inference, and decoding back to latitude/longitude.

Run:  python scripts/demo_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.factory import build_model, count_parameters
from src.models.forecasting import decode_direct

BASE = {"hidden_size": 64, "num_layers": 2, "dropout": 0.1, "forecast_steps": 8}
SCALER = {"mean": np.array([18.36, 137.50, -8.04, 8.06, 18.38, -0.46, 0.42]),
          "scale": np.array([8.43, 15.13, 14.86, 9.52, 10.17, 0.65, 0.45])}


def main() -> None:
    torch.manual_seed(42)
    print("== 1. construct + forward (direct, lstm, config C) ==")
    direct = build_model("lstm", "direct", input_size=4, **BASE)
    x = torch.randn(4, 5, 4)
    y_km = direct(x)
    print("in :", tuple(x.shape), "-> out:", tuple(y_km.shape), "(B, steps, [e,n] km)")

    print("\n== 2. all six architecture x strategy combos ==")
    for arch in ("rnn", "gru", "lstm"):
        for strat in ("direct", "recursive"):
            kw = {"scaler": SCALER, "feature_config": "C"} if strat == "recursive" else {}
            m = build_model(arch, strat, input_size=4, **BASE, **kw)
            total, _ = count_parameters(m)
            m.eval()
            with torch.no_grad():
                out = m.rollout(x) if strat == "recursive" else m(x)
            print(f"{arch:5s}/{strat:9s} params={total:7d} out={tuple(out.shape)}")

    print("\n== 3. recursive rollout -> decoded positions ==")
    rec = build_model("gru", "recursive", input_size=4, feature_config="C",
                      scaler=SCALER, **BASE)
    rec.eval()
    with torch.no_grad():
        pos = rec.rollout(x)
    print("positions:", tuple(pos.shape), "lon range:",
          (float(pos[..., 1].min()), float(pos[..., 1].max())), "(0-360E)")

    print("\n== 4. decode direct displacements with forecast origins ==")
    origins = np.array([[20.0, 135.0]] * 4)  # (lat, lon) of last observed position
    geo = decode_direct(y_km, origins)
    print("decoded:", geo.shape, "sample track-0 step-0:", geo[0, 0])


if __name__ == "__main__":
    main()
