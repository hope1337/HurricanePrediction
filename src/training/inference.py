"""Checkpoint-based inference shared by evaluate.py and run_experiments.

``predict_split`` loads a checkpoint, runs the matching inference path
(direct forward + decode, or genuine recursive rollout), and returns decoded
``[N, 8, 2]`` positions with the split's identifiers — the same forecast
cases every model is evaluated on.
"""

from __future__ import annotations

import numpy as np
import torch

from src.models.forecasting import decode_direct

from .data import load_split_arrays
from .trainer import load_checkpoint


def predict_split(checkpoint_path: str, npz_path: str, device="cpu",
                  batch_size: int = 512) -> dict:
    """Run inference for one checkpoint on one processed split."""
    device = torch.device(device)
    model, ckpt = load_checkpoint(checkpoint_path, device)
    cfg = ckpt["config"]
    split = load_split_arrays(npz_path, cfg.get("feature_config", "C"))
    X = torch.as_tensor(split["X"]).to(device)
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            chunk = X[i:i + batch_size]
            if cfg["strategy"] == "direct":
                outs.append(decode_direct(model(chunk).cpu().numpy(),
                                          split["last_latlon"][i:i + len(chunk)]))
            else:
                outs.append(model.rollout(chunk).cpu().numpy())
    return {"positions": np.concatenate(outs, axis=0).astype(np.float64),
            "targets": split["Y"].astype(np.float64),
            "sample_id": split["sample_id"], "sid": split["sid"],
            "season": split["season"], "hist_latlon": split["hist_latlon"],
            "config": cfg, "n_params": ckpt.get("n_params")}
