"""PyTorch dataset interface over processed split ``.npz`` files.

``config`` selects the ablation feature set ('A', 'B' or 'C') without
regenerating samples: all configs share sample IDs, origins and ground
truth by construction.  ``target`` selects 'position' (default), 'direct'
or 'step' displacement targets.  Set ``return_raw=True`` for the
unnormalized 7-column features and ``return_origin=True`` for the
forecast-origin ``last_latlon`` (needed to decode displacement outputs).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import CONFIG_COLUMNS, CONFIG_DIMS
from .scaling import invert, load_scaler


class HurricaneTrackDataset:
    """Torch-compatible dataset; ``__getitem__`` returns a metadata dict."""

    CONFIGS = tuple(CONFIG_COLUMNS)
    TARGETS = ("position", "direct", "step")
    _X_KEYS = {"A": "X_pos", "B": "X_spd", "C": "X_main"}

    def __init__(self, npz_path: str | Path, config: str = "C",
                 target: str = "position", return_raw: bool = False,
                 return_origin: bool = False):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required for HurricaneTrackDataset.") from exc
        if config not in self.CONFIGS:
            raise ValueError(f"config must be one of {self.CONFIGS}, got {config!r}")
        if target not in self.TARGETS:
            raise ValueError(f"target must be one of {self.TARGETS}, got {target!r}")
        self._torch = torch
        data = np.load(npz_path, allow_pickle=False)
        self.X = torch.as_tensor(data[self._X_KEYS[config]]).float()
        y_key = {"position": "Y", "direct": "Y_direct", "step": "Y_step"}[target]
        self.y = torch.as_tensor(data[y_key]).float()
        self.sample_id = data["sample_id"].astype(str)
        self.sid = data["sid"].astype(str)
        self.season = data["season"].astype(int)
        self.t0_ns = data["t0_ns"].astype(np.int64)
        self.config, self.target_kind = config, target
        self.return_raw, self.return_origin = return_raw, return_origin
        self.raw7 = torch.as_tensor(data["X_raw7"]).float() if return_raw else None
        self.origin = (torch.as_tensor(data["last_latlon"]).float()
                       if (return_origin and "last_latlon" in data) else None)
        self.n_features = CONFIG_DIMS[config]

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> dict:
        item = {"X": self.X[idx], "y": self.y[idx],
                "sample_id": str(self.sample_id[idx]),
                "sid": str(self.sid[idx]),
                "season": int(self.season[idx])}
        if self.return_raw:
            item["X_raw7"] = self.raw7[idx]
        if self.return_origin:
            if self.origin is None:
                raise KeyError("npz has no 'last_latlon'; regenerate with Task 2 pipeline.")
            item["origin_latlon"] = self.origin[idx]  # [lat, lon], orig 0-360 convention
        return item

    def inverse_features(self, X_batch, scaler_path: str | Path):
        """Map a scaled batch of this dataset's config back to physical units."""
        scaler = load_scaler(scaler_path)
        arr = X_batch.detach().cpu().numpy() if hasattr(X_batch, "detach") else np.asarray(X_batch)
        return invert(arr, scaler, self.config)
