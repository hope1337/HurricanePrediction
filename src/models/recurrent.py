"""Configurable unidirectional recurrent backbone (RNN / GRU / LSTM).

Conventions
-----------
* Batch-first tensors throughout: ``[batch, seq_len, input_size]``.
* Unidirectional only — bidirectional layers are deliberately excluded for
  this causal forecasting baseline (a backward pass would read the future).
* Vanilla RNN uses ``tanh`` nonlinearity (documented choice; ``relu`` is
  available via the ``nonlinearity`` argument but is not the default).
* The backbone returns the last layer's final hidden state ``[B, H]`` as the
  sequence representation consumed by the forecasting heads.  Dropout is
  applied between recurrent layers only (PyTorch applies it when
  ``num_layers > 1``); the constructor silences the single-layer case.
"""

from __future__ import annotations

import torch
from torch import nn

ARCHITECTURES = ("rnn", "gru", "lstm")


class RecurrentBackbone(nn.Module):
    """Shared backbone for the three baseline architectures."""

    def __init__(self, architecture: str = "lstm", input_size: int = 4,
                 hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.1, nonlinearity: str = "tanh"):
        super().__init__()
        arch = architecture.lower()
        if arch not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}, got {arch!r}")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.architecture = arch
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        layer_kwargs: dict = dict(input_size=input_size, hidden_size=hidden_size,
                                  num_layers=num_layers, batch_first=True,
                                  bidirectional=False,
                                  dropout=dropout if num_layers > 1 else 0.0)
        if arch == "rnn":
            if nonlinearity not in ("tanh", "relu"):
                raise ValueError("RNN nonlinearity must be 'tanh' or 'relu'")
            self.rnn = nn.RNN(nonlinearity=nonlinearity, **layer_kwargs)
        elif arch == "gru":
            self.rnn = nn.GRU(**layer_kwargs)
        else:
            self.rnn = nn.LSTM(**layer_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, T, D]`` -> final hidden state ``[B, H]``."""
        if x.dim() != 3 or x.size(-1) != self.input_size:
            raise ValueError(f"expected [B, T, {self.input_size}], got {tuple(x.shape)}")
        _, hidden = self.rnn(x)
        h = hidden[0] if self.architecture == "lstm" else hidden
        return h[-1]
