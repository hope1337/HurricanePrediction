"""Unit tests for Task 4 evaluation metrics (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.geometry import R_KM, haversine_distance_km
from src.evaluation import analysis as A
from src.evaluation import metrics as M
from src.evaluation.evaluator import assert_consistent_samples, evaluate_trajectories


def _latlon(lat, lon):
    return np.asarray(lat, float), np.asarray(lon, float)


# 1. identical coordinates -> zero ------------------------------------------------
def test_identical_zero():
    lat = np.array([[10.0, 20.0], [30.0, -5.0]])
    lon = np.array([[150.0, 151.0], [200.0, 100.0]])
    np.testing.assert_array_equal(haversine_distance_km(lat, lon, lat, lon),
                                  np.zeros((2, 2)))


# 2. one degree of latitude at equator ~111 km ------------------------------------
def test_one_degree_latitude_km():
    d = haversine_distance_km(0.0, 0.0, 1.0, 0.0)
    assert float(np.asarray(d).ravel()[0]) == pytest.approx(R_KM * np.pi / 180.0, rel=1e-9)
    assert 110.0 < float(np.asarray(d).ravel()[0]) < 112.0  # km, not degrees


# 3. antimeridian crossing ---------------------------------------------------------
def test_antimeridian_crossing():
    d1 = float(np.asarray(haversine_distance_km(0.0, 179.9, 0.0, -179.9)).ravel()[0])
    d2 = float(np.asarray(haversine_distance_km(0.0, 0.1, 0.0, -0.1)).ravel()[0])
    assert d1 == pytest.approx(d2, rel=1e-9)
    assert d1 == pytest.approx(0.2 * R_KM * np.pi / 180.0, rel=1e-6)
    # 0-360 convention gives the same answer
    d3 = float(np.asarray(haversine_distance_km(0.0, 359.9, 0.0, 0.1)).ravel()[0])
    assert d3 == pytest.approx(d1, rel=1e-9)


# 4. symmetry -----------------------------------------------------------------------
def test_symmetry():
    rng = np.random.default_rng(0)
    a_lat, a_lon = rng.uniform(-60, 60, 25), rng.uniform(0, 360, 25)
    b_lat, b_lon = rng.uniform(-60, 60, 25), rng.uniform(0, 360, 25)
    d_ab = haversine_distance_km(a_lat, a_lon, b_lat, b_lon)
    d_ba = haversine_distance_km(b_lat, b_lon, a_lat, a_lon)
    np.testing.assert_allclose(d_ab, d_ba, rtol=1e-12)


# 5. units are kilometres -------------------------------------------------------------
def test_units_km():
    # quarter meridian equator->pole must be ~10,000 km, not ~90 (degrees)
    d = float(np.asarray(haversine_distance_km(0.0, 0.0, 90.0, 0.0)).ravel()[0])
    assert d == pytest.approx(np.pi * R_KM / 2.0, rel=1e-9)
    assert d > 9000.0


# 6. ADE manual example ---------------------------------------------------------------
def test_ade_manual_example():
    err = np.array([[20.0, 35.0, 50.0, 95.0]])
    val, n = M.ade(err)
    assert val == pytest.approx(50.0)
    assert n == 4


# 7. FDE equals last-step MTE ----------------------------------------------------------
def test_fde_equals_last_mte():
    rng = np.random.default_rng(1)
    err = np.abs(rng.normal(40, 10, (30, 8)))
    fde_v, fde_n = M.fde(err)
    mte, n_h = M.mte_by_horizon(err)
    assert fde_v == pytest.approx(mte[-1])
    assert fde_n == n_h[-1]


# 8. horizon indexing --------------------------------------------------------------------
def test_horizon_indexing():
    assert M.HORIZON_HOURS == (6, 12, 18, 24, 30, 36, 42, 48)
    assert M.REPORT_HORIZONS == {6: 0, 12: 1, 24: 3, 48: 7}
    # error injected only at the last step surfaces only at +48h
    n = 10
    pred = np.zeros((n, 8, 2))
    true = np.zeros((n, 8, 2))
    true[:, -1, 0] = 1.0  # 1 deg lat offset at step 8 only
    res = evaluate_trajectories(pred, true,
                                storm_ids=[f"S{i}" for i in range(n)],
                                sample_ids=[f"id{i}" for i in range(n)])
    assert res["mte_km"][48] == pytest.approx(R_KM * np.pi / 180.0, rel=1e-6)
    assert res["mte_km"][6] == 0.0 and res["mte_km"][12] == 0.0 and res["mte_km"][24] == 0.0
    assert res["fde_km"] == pytest.approx(res["mte_km"][48])


# 9. batch sizes --------------------------------------------------------------------------
@pytest.mark.parametrize("n", (1, 7, 64))
def test_batch_sizes(n):
    rng = np.random.default_rng(2)
    lat = rng.uniform(-20, 40, (n, 8))
    lon = rng.uniform(100, 200, (n, 8))
    pred = np.stack([lat, lon], axis=-1)
    true = pred + rng.normal(0, 0.5, (n, 8, 2))
    res = evaluate_trajectories(pred, true, [f"S{i}" for i in range(n)], [f"id{i}" for i in range(n)])
    assert res["n_windows"] == n
    assert len(res["mte_full_km"]) == 8
    assert res["per_sample_errors_km"].shape == (n, 8)


# 10. shape mismatch detection ----------------------------------------------------------------
def test_shape_mismatch():
    pred = np.zeros((5, 8, 2))
    bad = np.zeros((5, 7, 2))
    with pytest.raises(ValueError):
        evaluate_trajectories(pred, bad, ["S"] * 5, [f"id{i}" for i in range(5)])
    with pytest.raises(ValueError):
        evaluate_trajectories(pred, pred, ["S"] * 4, [f"id{i}" for i in range(5)])
    with pytest.raises(ValueError):
        M.mte_by_horizon(np.zeros((4, 8)), valid_mask=np.ones((4, 7), bool))
    with pytest.raises(ValueError):
        evaluate_trajectories(pred, pred, ["S"] * 5, ["dup"] * 5)  # duplicate ids
    with pytest.raises(ValueError):
        assert_consistent_samples(["a", "b"], ["a", "c"], "M1", "M2")
    with pytest.raises(ValueError):
        assert_consistent_samples(["b", "a"], ["a", "b"], "M1", "M2")  # order matters


# 11. numerical stability -----------------------------------------------------------------------
def test_numerical_stability():
    tiny = float(np.asarray(haversine_distance_km(0.0, 0.0, 1e-9, 0.0)).ravel()[0])
    assert np.isfinite(tiny) and tiny >= 0.0
    anti = float(np.asarray(haversine_distance_km(0.0, 0.0, 0.0, 179.9999)).ravel()[0])
    assert np.isfinite(anti) and anti == pytest.approx(np.pi * R_KM, rel=1e-4)


# evaluator + analysis behaviour ---------------------------------------------------------------
def test_evaluator_full_result():
    rng = np.random.default_rng(3)
    n = 20
    lat = rng.uniform(0, 30, (n, 8))
    lon = rng.uniform(120, 180, (n, 8))
    pred = np.stack([lat, lon], axis=-1)
    true = pred + rng.normal(0, 1.0, (n, 8, 2))
    sids = [f"S{i % 4}" for i in range(n)]
    res = evaluate_trajectories(pred, true, sids, [f"id{i}" for i in range(n)],
                                init_times=[f"2000-01-{(i % 28) + 1:02d}" for i in range(n)])
    assert res["n_windows"] == 20 and res["n_storms"] == 4
    assert set(res["mte_km"]) == {6, 12, 24, 48}
    assert res["ade_n"] == 160 and res["fde_n"] == 20
    assert len(res["per_storm"]) == 4
    assert sum(v["n_windows"] for v in res["per_storm"].values()) == 20
    # storm-level mean consistency with per-trajectory values
    frame = A.per_sample_frame(res)
    assert frame["errors_km"].shape == (20, 8)
    for s, info in res["per_storm"].items():
        sub = frame["ade_km"][frame["storm_id"] == s]
        assert info["mean_ade_km"] == pytest.approx(sub.mean())
    # grouping without imposed thresholds
    groups = A.group_evaluate(frame["errors_km"], ["g1"] * 10 + ["g2"] * 10)
    assert set(groups) == {"g1", "g2"} and groups["g1"]["n_windows"] == 10
    assert groups["g1"]["ade_km"] == pytest.approx(frame["errors_km"][:10].mean())


def test_invalid_handling_explicit():
    err = np.array([[10.0, np.nan], [20.0, 30.0]])
    with pytest.raises(ValueError):  # NaN without mask must raise, not drop silently
        M.ade(err)
    val, n = M.ade(err, valid_mask=[[True, False], [True, True]])
    assert val == pytest.approx(20.0) and n == 3
    with pytest.raises(ValueError):
        M.pairwise_errors_km(np.full((2, 8, 2), np.nan), np.zeros((2, 8, 2)))
    with pytest.raises(ValueError):
        M.pairwise_errors_km(np.zeros((2, 8, 2)), np.zeros((2, 8, 2)).reshape(2, 2, 8))


def test_consistent_samples_ok():
    assert_consistent_samples(["a", "b"], ["a", "b"], "M1", "M2")


def test_classify_movement():
    # straight eastward track at ~10 km/h
    lat = np.full(6, 20.0)
    lon = np.linspace(135.0, 138.0, 6)
    labels, det = A.classify_movement(np.stack([lat, lon], -1)[None])
    assert labels.tolist() == ["straight"]
    # sharp >30 deg turn in the middle
    lat2 = np.array([20.0, 20.0, 20.0, 20.5, 21.0, 21.5])
    lon2 = np.array([135.0, 135.5, 136.0, 136.0, 136.0, 136.0])
    labels2, _ = A.classify_movement(np.stack([lat2, lon2], -1)[None])
    assert labels2.tolist() == ["turning"]
    # near-stationary -> ambiguous regardless of bearing noise
    labels3, det3 = A.classify_movement(np.stack(
        [np.full(6, 20.0), np.full(6, 135.0)], -1)[None])
    assert labels3.tolist() == ["ambiguous"]
    assert det3["mean_speed_kmh"][0] < 5.0
    # causality: labels identical with/without knowledge of any future
    assert labels.shape == (1,)
