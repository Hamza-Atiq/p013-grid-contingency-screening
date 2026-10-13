"""Voltage calibration and validation-only cutoff selection (research round 1, task A)."""
from pathlib import Path

import numpy as np
import pytest

from p013 import calibration as cal
from p013.danger import risk_score

P_, C_, B_ = 12, 4, 6  # points, contingencies, buses: small synthetic cache


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    truth = 1.0 + 0.04 * rng.standard_normal((P_, C_ + 1, B_))
    load = rng.uniform(800, 1400, P_)
    pinned = np.zeros((P_, C_), dtype=bool)
    pinned[3, 1] = True
    return truth, load, pinned


def test_risk_scores_match_danger_risk_score_case_by_case():
    rng = np.random.default_rng(1)
    vm = 0.9 + 0.2 * rng.random((P_, C_ + 1, B_))
    thermal = rng.random((P_, C_)) * 1.5
    pinned = np.zeros((P_, C_), dtype=bool)
    pinned[0, 2] = True
    got = cal.risk_scores(vm, thermal, pinned)
    for p in range(P_):
        for c in range(C_):
            if pinned[p, c]:
                assert np.isnan(got[p, c])
                continue
            # a single branch at thermal*100 % reproduces the thermal term exactly
            want = risk_score(vm[p, c + 1], np.array([thermal[p, c] * 100.0]), vm[p, 0])
            assert got[p, c] == pytest.approx(want, abs=1e-12)


def test_bias_recovers_a_planted_offset_and_apply_removes_it():
    truth, load, pinned = _synthetic()
    offset = np.linspace(-0.03, 0.03, B_)[None, :] * np.ones((C_ + 1, 1))
    model = truth + offset
    fitted = cal.fit("bias", model, truth, pinned, load, np.arange(P_))
    assert fitted["bias"] == pytest.approx(offset, abs=1e-12)
    assert cal.apply(fitted, model, load) == pytest.approx(truth, abs=1e-12)


def test_affine_recovers_slope_and_intercept_and_leaves_flat_cells_alone():
    truth, load, pinned = _synthetic()
    model = (truth - 0.02) / 0.9  # truth = 0.02 + 0.9 * model
    model[:, :, 0] = 1.02         # a generator bus: model never moves
    fitted = cal.fit("affine", model, truth, pinned, load, np.arange(P_))
    assert fitted["slope"][:, 1:] == pytest.approx(0.9, abs=1e-9)
    assert fitted["intercept"][:, 1:] == pytest.approx(0.02, abs=1e-9)
    assert np.all(fitted["slope"][:, 0] == 1.0)


def test_load_bias_recovers_an_error_that_grows_with_load():
    truth, load, pinned = _synthetic()
    model = truth + (0.001 + 2e-5 * (load - 1000))[:, None, None]
    fitted = cal.fit("load_bias", model, truth, pinned, load, np.arange(P_))
    assert cal.apply(fitted, model, load) == pytest.approx(truth, abs=1e-9)


def test_pinned_rows_are_not_learned_from():
    truth, load, pinned = _synthetic()
    model = truth + 0.01
    poisoned = truth.copy()
    poisoned[3, 2] = np.nan  # contingency 1 at point 3 is pinned: row 2
    fitted = cal.fit("bias", model, poisoned, pinned, load, np.arange(P_))
    assert np.isfinite(fitted["bias"]).all()
    assert fitted["bias"] == pytest.approx(0.01, abs=1e-12)


@pytest.mark.parametrize("kind", cal.KINDS)
def test_no_leakage_fit_never_reads_rows_it_was_not_given(kind):
    """Rows outside the fit set can hold anything - NaN, nonsense - and change nothing."""
    truth, load, pinned = _synthetic()
    rng = np.random.default_rng(7)
    model = truth + 0.01 * rng.standard_normal(truth.shape) + 0.005
    val = np.array([0, 2, 4, 5, 7, 9, 11])
    test = np.setdiff1d(np.arange(P_), val)

    clean = cal.fit(kind, model, truth, pinned, load, val)

    truth_bad, model_bad, load_bad, pinned_bad = truth.copy(), model.copy(), load.copy(), pinned.copy()
    truth_bad[test] = np.nan
    model_bad[test] = 1e6
    load_bad[test] = -1e9
    pinned_bad[test] = ~pinned_bad[test]
    dirty = cal.fit(kind, model_bad, truth_bad, pinned_bad, load_bad, val)

    for key, value in clean.items():
        if key != "kind":
            assert np.array_equal(value, dirty[key]), key


def test_cross_fitting_never_scores_a_fold_with_a_model_that_saw_it():
    folds = cal.make_folds(np.arange(10), n_folds=3, seed=0)
    seen = []

    def fitter(rows):
        seen.append(set(rows.tolist()))
        return {"rows": set(rows.tolist())}

    def scorer(fitted):
        # score each point 1.0 if the calibration was fitted on it, else 0.0
        return np.array([[1.0 if p in fitted["rows"] else 0.0] for p in range(12)])

    out = cal.cross_fitted_scores(fitter, scorer, folds, (12, 1))
    assert np.all(out[:10] == 0.0)          # nobody was scored by a fit that saw them
    assert np.isnan(out[10:]).all()         # rows outside validation are left empty
    assert all(not (s & {10, 11}) for s in seen)


def test_make_folds_is_a_partition():
    rows = np.array([3, 5, 8, 9, 13, 21, 22])
    folds = cal.make_folds(rows, n_folds=3, seed=4)
    joined = np.sort(np.concatenate(folds))
    assert joined.tolist() == rows.tolist()
    assert len(folds) == 3


def test_zero_miss_threshold_is_the_lowest_dangerous_unpinned_score():
    score = np.array([[0.9, 0.3, np.nan], [1.4, 0.7, 0.2]])
    dangerous = np.array([[True, False, True], [True, True, False]])
    pinned = np.array([[False, False, True], [False, False, False]])
    assert cal.zero_miss_threshold(score, dangerous, pinned) == 0.7
    assert cal.zero_miss_threshold(score, dangerous, pinned, rows=[0]) == 0.9
    none = np.zeros_like(dangerous)
    assert cal.zero_miss_threshold(score, none, pinned) == float("inf")


def test_cv_shortfall_measures_how_far_a_held_out_fold_undercuts_the_rest():
    score = np.array([[0.8], [0.9], [0.5], [1.0]])
    dangerous = np.ones((4, 1), dtype=bool)
    pinned = np.zeros((4, 1), dtype=bool)
    folds = [np.array([0, 1]), np.array([2, 3])]
    out = cal.cv_shortfall(score, dangerous, pinned, folds)
    # fold 0 alone: 0.8, rest: 0.5 -> no shortfall. fold 1 alone: 0.5, rest: 0.8 -> 0.3
    assert out["per_fold"] == pytest.approx([0.0, 0.3])
    assert out["margin"] == pytest.approx(0.3)


def test_unknown_kind_is_an_error():
    truth, load, pinned = _synthetic()
    with pytest.raises(ValueError):
        cal.fit("isotonic", truth, truth, pinned, load, np.arange(P_))


CACHE = Path("eval/cache/predictions.npz")


@pytest.mark.skipif(not CACHE.exists(), reason="cache not built; experiments/build_cache.py")
def test_uncorrected_voltages_reproduce_the_cached_model_score():
    """The recomputed score is the evaluation's score when nothing is corrected."""
    from p013 import prediction_cache

    c = prediction_cache.load(CACHE)
    identity = {"kind": "bias", "bias": np.zeros(c["model_vm"].shape[1:])}
    got = cal.calibrated_scores(identity, c)
    want = c["model_score"].astype(float)
    assert np.array_equal(np.isnan(got), np.isnan(want))
    live = ~np.isnan(want)
    assert np.abs(got[live] - want[live]).max() < 1e-4
