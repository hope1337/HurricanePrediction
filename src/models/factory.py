"""Model factory with baseline defaults and parameter accounting."""

from __future__ import annotations

from torch import nn

from .forecasting import DirectForecaster, RecursiveForecaster
from .heads import DirectDisplacementHead, OneStepDisplacementHead
from .recurrent import ARCHITECTURES, RecurrentBackbone

STRATEGIES = ("direct", "recursive")

BASELINE = {"hidden_size": 64, "num_layers": 2, "dropout": 0.1, "forecast_steps": 8}


def count_parameters(model: nn.Module) -> tuple[int, dict]:
    """Return (total_trainable, {module_path: count}) for a model."""
    counts: dict[str, int] = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            counts[name.rsplit(".", 1)[0]] = counts.get(name.rsplit(".", 1)[0], 0) + p.numel()
    return sum(counts.values()), counts


def build_model(architecture: str = "lstm", strategy: str = "direct",
                input_size: int = 4, hidden_size: int = 64,
                num_layers: int = 2, dropout: float = 0.1,
                forecast_steps: int = 8, scaler: dict | None = None,
                feature_config: str = "C",
                interval_hours: float = 6.0) -> nn.Module:
    """Build one of the six architecture × strategy baselines.

    ``strategy='direct'`` needs no scaler; ``strategy='recursive'`` requires
    the Task 2 training-fitted ``scaler`` dict (features are renormalized
    from physical units at every rollout step).
    """
    arch = architecture.lower()
    strat = strategy.lower()
    if arch not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}, got {architecture!r}")
    if strat not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    backbone = RecurrentBackbone(architecture=arch, input_size=input_size,
                                 hidden_size=hidden_size, num_layers=num_layers,
                                 dropout=dropout)
    if strat == "direct":
        return DirectForecaster(backbone, DirectDisplacementHead(hidden_size, forecast_steps))
    if scaler is None:
        raise ValueError("strategy='recursive' requires the training-fitted scaler dict")
    return RecursiveForecaster(backbone, OneStepDisplacementHead(hidden_size),
                               scaler=scaler, config=feature_config,
                               forecast_steps=forecast_steps, interval_hours=interval_hours)
