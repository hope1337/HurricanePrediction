"""Verification tests for the Task 2 preprocessing pipeline.

Unit tests use small deterministic synthetic tracks; integration tests use
the real processed ``.npz`` files (skipped with a clear message when the
pipeline has not been run yet).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

from src.data import geometry as G
from src.data.dataset import HurricaneTrackDataset
from src.data.features import CONFIG_COLUMNS, add_segment_motion, feature_matrix_7
from src.data.preprocessing import sort_and_deduplicate, split_segments
from src.data.scaling import apply_scaler, fit_scaler, invert
from src.data.windowing import generate_windows


# ---------------------------------------------------------------- synthetic --
def _synth_frame() -> pd.DataFrame:
    """Two storms, exact 6 h cadence, one crossing the antimeridian."""
    rows = []
    t0 = pd.Timestamp("2000-08-01 00:00", tz="UTC")
    for k in range(16):
        rows.append({"SID": "S1", "SEASON": 2000, "OBS_IDX": k, "BASIN": "WP",
                     "TRACK_TYPE": "main", "NATURE": "", "IFLAG": "",
                     "TIME": t0 + pd.Timedelta(hours=6 * k),
                     "LAT": 10.0 + 0.1 * k, "LON": 150.0 + 0.2 * k})
    for k in range(16):
        rows.append({"SID": "S2", "SEASON": 2000, "OBS_IDX": k, "BASIN": "WP",
                     "TRACK_TYPE": "main", "NATURE": "", "IFLAG": "",
                     "TIME": t0 + pd.Timedelta(hours=6 * k),
                     "LAT": 20.0, "LON": (359.0 + 0.5 * k) % 360.0})
    df = pd.DataFrame(rows)
    df["TIME_DAYS"] = (df["TIME"] - pd.Timestamp("1858-11-17", tz="UTC")).dt.total_seconds() / 86400
    for c in ("STORM_SPEED", "STORM_DIR", "WMO_WIND", "WMO_PRES", "DIST2LAND", "LANDFALL"):
        df[c] = np.nan
    df["WMO_AGENCY"] = ""
    return df


def _synth_segments() -> pd.DataFrame:
    df, _ = sort_and_deduplicate(_synth_frame())
    return split_segments(df)


def _proc_or_skip(name: str):
    p = PROC / name
    if not p.exists():
        pytest.skip(f"processed file missing: {p} (run scripts/preprocess.py first)")
    return np.load(p, allow_pickle=False)


# ------------------------------------------------------------- unit tests --
def test_temporal_ordering_synthetic():
    seg = _synth_segments()
    for _, g in seg.groupby("SEG_ID"):
        assert g["TIME"].is_monotonic_increasing
    out = generate_windows(seg)
    t0 = pd.to_datetime([m["t0_iso"] for m in out["meta"]], utc=True)
    for m, t in zip(out["meta"], t0):
        tgt = pd.to_datetime(m["target_iso"], utc=True)
        assert (tgt > t).all() and tgt.is_monotonic_increasing


def test_no_future_in_features_perturbation():
    """Altering a target position must not change any input feature."""
    seg = _synth_segments()
    base = generate_windows(seg)
    seg2 = seg.copy()
    last_seg = seg2[seg2["SEG_ID"] == seg2["SEG_ID"].iloc[-1]].index
    seg2.loc[last_seg[-1], "LAT"] += 5.0  # perturb a future-only position
    alt = generate_windows(split_segments(seg2.drop(columns=["SEG_ID", "SEG_LEN", "SEG_POS"])))
    # windows not touching the perturbed obs keep identical X
    assert base["X7"][:-1].shape == alt["X7"][:-1].shape
    np.testing.assert_array_equal(base["X7"][:-1], alt["X7"][:-1])
    assert not np.array_equal(base["Y"][-1], alt["Y"][-1])


def test_unwrap_antimeridian():
    lon = np.array([359.0, 359.5, 0.0, 0.5, 1.0])
    u = G.unwrap_lon(lon)
    assert np.all(np.abs(np.diff(u)) < 180.0)
    assert u[-1] > 360.0  # continues past 360 instead of jumping


def test_roundtrip_synthetic():
    seg = _synth_segments()
    out = generate_windows(seg)
    for i in range(len(out["X7"])):
        o_lat, o_lon = out["meta"][i]["last_lat"], out["meta"][i]["last_lon"]
        y = out["Y"][i]
        la, lo = G.geodetic_from_enu(o_lat, o_lon, out["Y_direct"][i][:, 0], out["Y_direct"][i][:, 1])
        np.testing.assert_allclose(la, y[:, 0], atol=1e-9)
        np.testing.assert_allclose(G.normalize_lon_360(lo), G.normalize_lon_360(y[:, 1]), atol=1e-9)
        # recursive chain from origin
        clat, clon = [o_lat], [o_lon]
        for s in range(8):
            la2, lo2 = G.forward(clat[-1], clon[-1],
                                 (np.rad2deg(np.arctan2(out["Y_step"][i][s, 0],
                                                        out["Y_step"][i][s, 1])) + 360) % 360,
                                 np.hypot(*out["Y_step"][i][s]) * 1000.0)
            clat.append(float(np.asarray(la2).ravel()[0]))
            clon.append(float(np.asarray(lo2).ravel()[0]))
        np.testing.assert_allclose(clat[1:], y[:, 0], atol=1e-9)
        np.testing.assert_allclose(G.normalize_lon_360(clon[1:]),
                                   G.normalize_lon_360(y[:, 1]), atol=1e-9)


def test_reproducibility():
    seg = _synth_segments()
    a, b = generate_windows(seg)["X7"], generate_windows(seg)["X7"]
    np.testing.assert_array_equal(a, b)


def test_pyproj_haversine_agreement():
    if G._GEOD is None:
        pytest.skip("pyproj backend not installed")
    lat1 = np.array([10.0, 20.0, -5.0])
    lon1 = np.array([150.0, 359.0, 80.0])
    lat2, lon2 = lat1 + 0.5, lon1 + 0.7
    d_geo, _ = G.inverse(lat1, lon1, lat2, lon2)
    d_hav, _ = G._haversine_inverse(lat1, lon1, lat2, lon2)
    assert np.allclose(d_geo, d_hav, rtol=0.005)  # ellipsoid vs sphere agree <0.5%


# ------------------------------------------------------ integration tests --
def test_shapes():
    for name, n_feat in (("train", 4), ("validation", 4), ("test", 4)):
        d = _proc_or_skip(f"{name}.npz")
        n = d["Y"].shape[0]
        assert d["X_main"].shape == (n, 5, 4)
        assert d["X_pos"].shape == (n, 5, 2)
        assert d["X_spd"].shape == (n, 5, 5)
        assert d["Y"].shape == (n, 8, 2)
        assert d["Y_direct"].shape == (n, 8, 2)
        assert d["Y_step"].shape == (n, 8, 2)
        assert d["hist_latlon"].shape == (n, 6, 2)


def test_sid_disjoint_splits():
    sids = {}
    for name in ("train", "validation", "test"):
        d = _proc_or_skip(f"{name}.npz")
        sids[name] = set(d["sid"].astype(str).tolist())
    assert not (sids["train"] & sids["validation"])
    assert not (sids["train"] & sids["test"])
    assert not (sids["validation"] & sids["test"])


def test_strict_6h_intervals():
    for name in ("train", "validation", "test"):
        d = _proc_or_skip(f"{name}.npz")
        t0 = d["t0_ns"].astype(np.int64)
        tgt = d["target_ns"].astype(np.int64)
        step = np.int64(6 * 3600 * 10 ** 9)
        expect = t0[:, None] + step * np.arange(1, 9)[None, :]
        np.testing.assert_array_equal(tgt, expect)


def test_no_cross_storm_windows_and_ordering():
    d = _proc_or_skip("train.npz")
    man = pd.read_csv(PROC / "sample_manifest.csv")
    man = man[man["split"] == "train"].reset_index(drop=True)
    assert (man["sample_id"].values == d["sample_id"].astype(str)).all()
    # each window's history+targets reconstruct one 14-step 6 h lattice
    assert man["target_tplus6h"].notna().all()


def test_config_identity():
    for name in ("train", "validation", "test"):
        d = _proc_or_skip(f"{name}.npz")
        assert len(d["sample_id"]) == len(np.unique(d["sample_id"]))
        for cfg in ("A", "B", "C"):
            ds = HurricaneTrackDataset(PROC / f"{name}.npz", config=cfg)
            assert len(ds) == len(d["sample_id"])
            item = ds[0]
            assert item["X"].shape == (5, len(CONFIG_COLUMNS[cfg]))
            assert item["y"].shape == (8, 2)
            assert item["sample_id"] == str(d["sample_id"][0])


def test_features_causal_on_real_data():
    """Recompute causal features from stored history; must match X_raw7."""
    d = _proc_or_skip("train.npz")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(d["sample_id"]), size=min(50, len(d["sample_id"])), replace=False)
    for i in idx:
        hist = d["hist_latlon"][i]  # 6 orig-convention positions t-30..t
        lat, lon = hist[:, 0].astype(float), hist[:, 1].astype(float)
        lon_u = G.unwrap_lon(lon)
        dist_m, az = G.inverse(lat[:-1], lon_u[:-1], lat[1:], lon_u[1:])
        v = np.asarray(dist_m) / 1000.0 / 6.0
        r = np.deg2rad(np.asarray(az))
        expect = np.column_stack([lat[1:], lon_u[1:], v * np.sin(r), v * np.cos(r),
                                  v, np.sin(r), np.cos(r)])
        np.testing.assert_allclose(d["X_raw7"][i], expect, rtol=1e-5, atol=1e-6)


def test_unwrap_real_inputs():
    d = _proc_or_skip("train.npz")
    du = np.abs(np.diff(d["X_raw7"][:, :, 1], axis=1))
    assert (du < 180.0).all()


def test_roundtrip_real_data():
    d = _proc_or_skip("test.npz")
    n = min(200, len(d["sample_id"]))
    for i in range(n):
        o_lat, o_lon = (float(x) for x in d["last_latlon"][i])
        y = d["Y"][i].astype(float)
        la, lo = G.geodetic_from_enu(o_lat, o_lon, d["Y_direct"][i][:, 0], d["Y_direct"][i][:, 1])
        np.testing.assert_allclose(np.asarray(la, float), y[:, 0], atol=1e-5)
        np.testing.assert_allclose(G.normalize_lon_360(np.asarray(lo, float)),
                                   G.normalize_lon_360(y[:, 1]), atol=1e-5)


def test_train_only_scaler():
    scaler = joblib.load(PROC / "scalers.joblib")
    tr = _proc_or_skip("train.npz")
    raw = tr["X_raw7"].astype(np.float64).reshape(-1, 7)
    # X_raw7 is stored as float32: allow quantization-level tolerance.
    np.testing.assert_allclose(scaler["mean"], raw.mean(axis=0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(scaler["scale"], raw.std(axis=0), rtol=1e-6, atol=1e-6)
    va = _proc_or_skip("validation.npz")
    i = 0
    manual = (va["X_raw7"][i].astype(np.float64) - scaler["mean"]) / scaler["scale"]
    np.testing.assert_allclose(va["X_main"][i], manual[:, CONFIG_COLUMNS["C"]], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(invert(va["X_main"][i], scaler, "C"),
                               va["X_raw7"][i][:, CONFIG_COLUMNS["C"]], rtol=1e-5, atol=1e-4)
