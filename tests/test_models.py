"""Unit tests for Task 3 models — synthetic tensors only.

Covers: forward shapes for all six architecture × strategy combos, variable
batch sizes, feature dims (2/4/5), gradient flow, device placement, rollout
shape, no-leakage construction, decoding consistency, and finite outputs.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from src.data import geometry as G
from src.models.factory import BASELINE, build_model, count_parameters
from src.models.forecasting import decode_direct

ARCHS = ("rnn", "gru", "lstm")
DIMS = {"A": 2, "B": 5, "C": 4}
Bsz = BASELINE


def _scaler(dim7: int = 7):
    return {"mean": np.zeros(dim7), "scale": np.ones(dim7)}


def _batch(b: int, d: int, steps: int = 5):
    torch.manual_seed(0)
    return torch.randn(b, steps, d)


# ------------------------------------------------------------ direct ----
@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("batch", (1, 4, 16))
@pytest.mark.parametrize("cfg,dim", (("A", 2), ("B", 5), ("C", 4)))
def test_direct_shapes(arch, batch, cfg, dim):
    m = build_model(architecture=arch, strategy="direct", input_size=dim,
                    **{k: Bsz[k] for k in ("hidden_size", "num_layers", "dropout",
                                            "forecast_steps")})
    m.eval()
    with torch.no_grad():
        y = m(_batch(batch, dim))
    assert y.shape == (batch, 8, 2)


@pytest.mark.parametrize("arch", ARCHS)
def test_direct_gradients(arch):
    m = build_model(architecture=arch, strategy="direct", input_size=4,
                    **{k: Bsz[k] for k in ("hidden_size", "num_layers", "dropout",
                                            "forecast_steps")})
    m.train()
    y = m(_batch(4, 4))
    loss = (y ** 2).mean()
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


# ---------------------------------------------------------- recursive ---
@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("batch", (1, 4, 16))
@pytest.mark.parametrize("cfg,dim", (("A", 2), ("B", 5), ("C", 4)))
def test_recursive_shapes(arch, batch, cfg, dim):
    m = build_model(architecture=arch, strategy="recursive", input_size=dim,
                    feature_config=cfg, scaler=_scaler(),
                    **{k: Bsz[k] for k in ("hidden_size", "num_layers", "dropout",
                                            "forecast_steps")})
    m.eval()
    pos = m.rollout(_batch(batch, dim))
    assert pos.shape == (batch, 8, 2)
    assert torch.isfinite(pos).all()


def test_recursive_onestep_gradients():
    m = build_model(architecture="lstm", strategy="recursive", input_size=4,
                    feature_config="C", scaler=_scaler(),
                    **{k: Bsz[k] for k in ("hidden_size", "num_layers", "dropout",
                                            "forecast_steps")})
    m.train()
    d = m.forward_onestep(_batch(4, 4))
    assert d.shape == (4, 2)
    d.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.parameters() if p.requires_grad)


def test_recursive_no_ground_truth_leakage():
    """Rollout must not accept ground truth: signature + invariance checks."""
    from src.models.forecasting import RecursiveForecaster
    assert "y" not in inspect.signature(RecursiveForecaster.rollout).parameters
    assert "target" not in inspect.signature(RecursiveForecaster.rollout).parameters
    m = build_model(architecture="gru", strategy="recursive", input_size=4,
                    feature_config="C", scaler=_scaler(),
                    **{k: Bsz[k] for k in ("hidden_size", "num_layers", "dropout",
                                            "forecast_steps")})
    m.eval()
    x = _batch(3, 4)
    a, b = m.rollout(x), m.rollout(x.clone())
    torch.testing.assert_close(a, b)  # deterministic, Y-independent
    c = m.rollout(x + 0.5)  # history change must change output
    assert not torch.allclose(a, c)


def test_recursive_uses_only_history():
    """Perturbing the earliest input row still flows through (causal use)."""
    m = build_model(architecture="rnn", strategy="recursive", input_size=2,
                    feature_config="A", scaler=_scaler(),
                    **{k: Bsz[k] for k in ("hidden_size", "num_layers", "dropout",
                                            "forecast_steps")})
    m.eval()
    torch.manual_seed(1)
    x = torch.randn(2, 5, 2) * 0.1
    base = m.rollout(x)
    x2 = x.clone()
    x2[:, 0, :] += 2.0
    assert not torch.allclose(base, m.rollout(x2))


# ------------------------------------------------------------ misc ----
def test_device_placement():
    for arch in ARCHS:
        m = build_model(architecture=arch, strategy="direct", input_size=4,
                        **{k: Bsz[k] for k in ("hidden_size", "num_layers", "dropout",
                                                "forecast_steps")})
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m.to(dev)
        y = m(_batch(2, 4).to(dev))
        assert y.device.type == dev.type


def test_decode_consistency():
    rng = np.random.default_rng(0)
    o = np.column_stack([rng.uniform(-10, 30, 6), rng.uniform(100, 200, 6)])
    d = rng.normal(0, 60, (6, 8, 2))
    pos = decode_direct(d, o)
    assert pos.shape == (6, 8, 2)
    assert ((pos[..., 1] >= 0) & (pos[..., 1] < 360)).all()
    back_e, back_n = G.enu_from_geodetic(o[:, 0:1], o[:, 1:2], pos[..., 0], pos[..., 1])
    np.testing.assert_allclose(np.stack([back_e, back_n], -1), d, rtol=1e-9, atol=1e-9)


def test_no_nan_inf_representative_inputs():
    torch.manual_seed(7)
    for arch in ARCHS:
        md = build_model(architecture=arch, strategy="direct", input_size=4,
                         **{k: Bsz[k] for k in ("hidden_size", "num_layers", "dropout",
                                                 "forecast_steps")})
        mr = build_model(architecture=arch, strategy="recursive", input_size=4,
                         feature_config="C", scaler=_scaler(),
                         **{k: Bsz[k] for k in ("hidden_size", "num_layers", "dropout",
                                                 "forecast_steps")})
        md.eval()
        mr.eval()
        x = torch.randn(8, 5, 4).clamp(-4, 4)  # scaled-feature range
        with torch.no_grad():
            assert torch.isfinite(md(x)).all()
            assert torch.isfinite(mr.rollout(x)).all()
            assert torch.isfinite(mr.forward_onestep(x)).all()


def test_invalid_configs_rejected():
    with pytest.raises(ValueError):
        build_model(architecture="transformer", strategy="direct", input_size=4)
    with pytest.raises(ValueError):
        build_model(architecture="lstm", strategy="beam", input_size=4)
    with pytest.raises(ValueError):
        build_model(architecture="lstm", strategy="recursive", input_size=4)  # no scaler


def test_parameter_counts_recorded():
    counts = {}
    for arch in ARCHS:
        m = build_model(architecture=arch, strategy="direct", input_size=4,
                        **{k: Bsz[k] for k in ("hidden_size", "num_layers", "dropout",
                                                "forecast_steps")})
        total, _ = count_parameters(m)
        counts[arch] = total
        assert total > 0
    # LSTM(4 gates) > GRU(3 gates) > RNN(1 gate) with identical sizes.
    assert counts["lstm"] > counts["gru"] > counts["rnn"]
