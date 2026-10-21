"""Training data access: split loaders and causal one-step derivation.

Direct training uses the Task 2 14-observation windows as-is
(``X_* [N,5,D]`` inputs, ``Y_direct [N,8,2]`` km targets).

Recursive training needs one-step samples (5 input rows + next 6-hour
displacement).  These are derived inside this module from each stored
14-position sequence ``concat(hist_latlon[6], Y[8])``: for start offsets
j = 0..7, input rows come from positions j+1..j+5 with motion recomputed
causally (past-only, via :mod:`src.data.geometry`) from positions j..j+5,
and the target is the step displacement j+5 -> j+6.  The j = 0 sub-window
reuses the staged features exactly; j > 0 extends supervision to later
origins.  Everything is renormalized with the SAME train-fitted scaler so
rollout renormalization stays consistent.

Result: recursive one-step training examples (8 per direct window) vs the
direct count are reported by the pipeline and documented in the report.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data import geometry as G
from src.data.features import CONFIG_COLUMNS

SUB_WINDOWS_PER_WINDOW = 8  # start offsets j = 0..7 inside each 14-obs window


def load_split_arrays(npz_path: str, feature_config: str = "C") -> dict:
    """Load one processed split; return normalized X, raw targets, metadata."""
    data = np.load(npz_path, allow_pickle=False)
    key = {"A": "X_pos", "B": "X_spd", "C": "X_main"}[feature_config]
    return {
        "X": data[key].astype(np.float32),
        "X_raw7": data["X_raw7"].astype(np.float32),
        "Y": data["Y"].astype(np.float32),
        "Y_direct": data["Y_direct"].astype(np.float32),
        "Y_step": data["Y_step"].astype(np.float32),
        "sample_id": data["sample_id"].astype(str),
        "sid": data["sid"].astype(str),
        "season": data["season"].astype(int),
        "last_latlon": data["last_latlon"].astype(np.float32),
        "hist_latlon": data["hist_latlon"].astype(np.float32),
    }


def derive_onestep_arrays(split: dict, scaler: dict, feature_config: str = "C") -> tuple[np.ndarray, np.ndarray]:
    """Build normalized one-step (input [M,5,D], target [M,2] km) training arrays.

    ``split`` is a :func:`load_split_arrays` dict.  Motion is recomputed
    causally from positions only — never from file STORM_SPEED/STORM_DIR.
    """
    cols = CONFIG_COLUMNS[feature_config]
    mean = np.asarray(scaler["mean"], dtype=np.float64)[cols]
    scale = np.asarray(scaler["scale"], dtype=np.float64)[cols]
    pos = np.concatenate([split["hist_latlon"].astype(np.float64),
                          split["Y"].astype(np.float64)], axis=1)  # [N,14,2]
    n = pos.shape[0]
    lat, lon = pos[:, :, 0], pos[:, :, 1]
    lon_u = np.stack([G.unwrap_lon(lon[i]) for i in range(n)])
    Xs, Ts = [], []
    for j in range(SUB_WINDOWS_PER_WINDOW):
        seg_lat, seg_lon = lat[:, j:j + 6], lon_u[:, j:j + 6]
        dist_m, az = G.inverse(seg_lat[:, :-1], seg_lon[:, :-1], seg_lat[:, 1:], seg_lon[:, 1:])
        v = np.asarray(dist_m, float) / 1000.0 / 6.0
        r = np.deg2rad(np.asarray(az, float))
        rows = seg_lat[:, 1:], seg_lon[:, 1:]
        full7 = np.stack([rows[0], rows[1], v * np.sin(r), v * np.cos(r),
                          v, np.sin(r), np.cos(r)], axis=-1)  # [N,5,7]
        Xs.append((full7[:, :, cols] - mean) / scale)
        e_s, n_s = G.enu_from_geodetic(seg_lat[:, -1], seg_lon[:, -1],
                                        lat[:, j + 6], lon[:, j + 6])
        Ts.append(np.stack([np.asarray(e_s, float), np.asarray(n_s, float)], axis=-1))
    return (np.concatenate(Xs, axis=0).astype(np.float32),
            np.concatenate(Ts, axis=0).astype(np.float32))


def make_loader(X: np.ndarray, T: np.ndarray, batch_size: int,
                shuffle: bool, seed: int = 0) -> DataLoader:
    """Deterministic DataLoader (single-process workers for reproducibility)."""
    ds = TensorDataset(torch.as_tensor(X), torch.as_tensor(T))
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=gen)
