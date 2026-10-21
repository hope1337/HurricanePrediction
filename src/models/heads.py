"""Forecasting heads: linear maps from backbone state to displacements.

Both heads predict geographical displacements in **physical kilometres**
(not normalized units, not degrees):

* :class:`OneStepDisplacementHead` — ``[B, H] -> [B, 2]``: a single 6-hour
  east/north step ``(e, n)`` in the current position's local tangent frame.
  Used for training (teacher-forced one-step loss) and recursive rollout.
* :class:`DirectDisplacementHead` — ``[B, H] -> [B, S, 2]``: all ``S`` future
  east/north offsets at once, each relative to the forecast origin (last
  observed position) in the origin-anchored frame.  Matches the Task 2
  ``Y_direct`` target definition.  No recursive feedback.

Head design is intentionally identical across backbones (single linear map)
so architecture comparisons stay fair.
"""

from __future__ import annotations

import torch
from torch import nn


class OneStepDisplacementHead(nn.Module):
    """Linear head: backbone state -> one-step ENU displacement (km)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(int(hidden_size), 2)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() != 2:
            raise ValueError(f"expected [B, H], got {tuple(h.shape)}")
        return self.proj(h)


class DirectDisplacementHead(nn.Module):
    """Linear head: backbone state -> all-horizon ENU displacements (km)."""

    def __init__(self, hidden_size: int, forecast_steps: int = 8):
        super().__init__()
        self.forecast_steps = int(forecast_steps)
        self.proj = nn.Linear(int(hidden_size), 2 * self.forecast_steps)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() != 2:
            raise ValueError(f"expected [B, H], got {tuple(h.shape)}")
        return self.proj(h).view(h.size(0), self.forecast_steps, 2)
