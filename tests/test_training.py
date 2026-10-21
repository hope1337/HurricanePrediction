"""Tests for Task 5 training infrastructure (tiny synthetic runs only)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.training.data import derive_onestep_arrays, load_split_arrays
from src.training.seed import select_device, set_seed
from src.training.trainer import build_optimizer, load_checkpoint, train_run


def _tiny_arrays(n_train=64, n_val=32):
    rng = np.random.default_rng(0)
    mk = lambda n: {
        "X": rng.normal(0, 1, (n, 5, 4)).astype(np.float32),
        "X_raw7": rng.normal(0, 1, (n, 5, 7)).astype(np.float32),
        "Y": np.stack([rng.uniform(0, 30, (n, 8)),
                       rng.uniform(100, 160, (n, 8))], -1).astype(np.float32),
        "Y_direct": rng.normal(0, 50, (n, 8, 2)).astype(np.float32),
        "Y_step": rng.normal(0, 20, (n, 8, 2)).astype(np.float32),
        "sample_id": np.array([f"id{i}" for i in range(n)]),
        "sid": np.array([f"S{i % 4}" for i in range(n)]),
        "season": np.full(n, 2000),
        "last_latlon": np.stack([rng.uniform(0, 30, n),
                                 rng.uniform(100, 160, n)], -1).astype(np.float32),
        "hist_latlon": np.stack([rng.uniform(0, 30, (n, 6)),
                                 rng.uniform(100, 160, (n, 6))], -1).astype(np.float32),
    }
    return {"train": mk(n_train), "val": mk(n_val)}


def _scaler():
    return {"mean": np.zeros(7), "scale": np.ones(7),
            "columns": ["LAT", "LON_U", "VE_KMH", "VN_KMH", "SPEED_KMH", "SIN_DIR", "COS_DIR"]}


def _cfg(**kw):
    base = {"architecture": "gru", "strategy": "direct", "feature_config": "C",
            "input_size": 4, "hidden_size": 8, "num_layers": 1, "dropout": 0.0,
            "forecast_steps": 8, "batch_size": 16, "lr": 0.01, "optimizer": "adam",
            "weight_decay": 0.0, "huber_delta": 50.0, "max_epochs": 2,
            "early_stopping_patience": 10, "gradient_clip_norm": 1.0,
            "rollout_every": 1, "scheduler": {"enabled": False}, "seed": 0}
    base.update(kw)
    return base


def test_direct_smoke_train(tmp_path):
    from src.training.trainer import train_run
    out = train_run(_cfg(), _tiny_arrays(), _scaler(), tmp_path / "run",
                    torch.device("cpu"))
    assert (tmp_path / "run" / "best.pt").exists()
    assert (tmp_path / "run" / "last.pt").exists()
    assert len(out["history"]["train_loss"]) == 2
    assert out["best_val_ade"] > 0 and out["n_params"] > 0


def test_recursive_smoke_train_and_counts(tmp_path):
    from src.training.data import SUB_WINDOWS_PER_WINDOW, derive_onestep_arrays
    from src.training.trainer import train_run
    arrays = _tiny_arrays()
    Xtr, Ttr = derive_onestep_arrays(arrays["train"], _scaler(), "C")
    assert len(Xtr) == len(arrays["train"]["X"]) * SUB_WINDOWS_PER_WINDOW
    assert Xtr.shape[1:] == (5, 4) and Ttr.shape[1:] == (2,)
    assert np.isfinite(Xtr).all() and np.isfinite(Ttr).all()
    out = train_run(_cfg(strategy="recursive"), arrays, _scaler(),
                    tmp_path / "run", torch.device("cpu"))
    assert (tmp_path / "run" / "best.pt").exists()
    m, ckpt = load_checkpoint(tmp_path / "run" / "best.pt", "cpu")
    assert ckpt["config"]["strategy"] == "recursive"
    with torch.no_grad():
        pos = m.rollout(torch.as_tensor(arrays["val"]["X"][:4]))
    assert pos.shape == (4, 8, 2) and torch.isfinite(pos).all()


def test_optimizer_choices_and_seed():
    import torch.nn as nn
    m = nn.Linear(4, 2)
    assert build_optimizer(m, "adam", 1e-3, 0.0).__class__.__name__ == "Adam"
    assert build_optimizer(m, "adamw", 1e-3, 0.0).__class__.__name__ == "AdamW"
    with pytest.raises(ValueError):
        build_optimizer(m, "sgd", 1e-3, 0.0)
    set_seed(123)
    a = torch.randn(3).tolist()
    set_seed(123)
    assert torch.randn(3).tolist() == a
    assert select_device("cpu").type == "cpu"
    with pytest.raises(RuntimeError):
        select_device("tpu")


def test_checkpoint_roundtrip_preserves_weights(tmp_path):
    from src.training.trainer import train_run
    d = tmp_path / "run"
    train_run(_cfg(max_epochs=1), _tiny_arrays(), _scaler(), d, torch.device("cpu"))
    m1, _ = load_checkpoint(d / "best.pt", "cpu")
    m2, _ = load_checkpoint(d / "best.pt", "cpu")
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)
