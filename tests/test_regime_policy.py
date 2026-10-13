"""Policies P1/P2/P3 and the round-2 harness (research round 2)."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from p013 import calibration as cal
from p013 import regime_policy as rp

P_, C_, B_, LOADS_ = 60, 8, 6, 3


def _synthetic(seed=0):
    """A small cache. Heavy days have lower voltages, so danger lives mostly there."""
    rng = np.random.default_rng(seed)
    load_p = rng.uniform(250, 500, (P_, LOADS_))
    total = load_p.sum(axis=1)
    heaviness = (total - total.min()) / (total.max() - total.min())
    truth_vm = 1.0 - 0.09 * heaviness[:, None, None] * rng.random((P_, C_ + 1, B_)) \
        + 0.01 * rng.standard_normal((P_, C_ + 1, B_))
    truth_vm[:, 0, :] = 1.0  # intact state inside the band
    model_vm = truth_vm + 0.01 + 0.005 * rng.standard_normal(truth_vm.shape)
    model_vm[:, 0, :] = 1.0
    thermal = rng.uniform(0.2, 0.9, (P_, C_))
    pinned = np.zeros((P_, C_), dtype=bool)
    pinned[5, 2] = pinned[40, 7] = True
    truth_score = cal.risk_scores(truth_vm, thermal, pinned)
    dangerous = (np.nan_to_num(truth_score, nan=0.0) >= 0.5) | pinned
    return {
        "model_vm": model_vm, "truth_vm": truth_vm, "model_thermal": thermal,
        "pinned": pinned, "dangerous": dangerous, "load_p_mw": load_p,
        "is_test": np.arange(P_) >= 40,
    }


PRACTICE = np.arange(40)
EXAM = np.arange(40, P_)


def _fingerprint(fitted):
    """Every fitted parameter and the selection, as plain comparable values."""
    out = {"primary": fitted["primary"], "bias": fitted["scores"].calibration["bias"].tolist(),
           "oof_practice": np.nan_to_num(fitted["scores"].oof[PRACTICE], nan=-9).tolist()}
    for name, pol in fitted["policies"].items():
        s = pol.summary()
        out[name] = json.dumps(s, sort_keys=True, default=float)
    return out


def test_synthetic_cache_has_danger_in_both_regimes():
    d = _synthetic()
    need = d["dangerous"] & ~d["pinned"]
    assert need[PRACTICE].sum() > 5 and need[EXAM].sum() > 0


# ---------------------------------------------------------------- regimes
def test_heavy_days_is_load_at_or_above_L():
    assert rp.heavy_days(np.array([99.0, 100.0, 101.0]), 100.0).tolist() == [False, True, True]


def test_load_grid_uses_practice_days_only():
    d = _synthetic()
    load = cal.total_load(d)
    grid = rp.load_grid(load, PRACTICE)
    assert list(grid) == [50, 60, 70, 80]
    assert grid[50] == pytest.approx(np.percentile(load[PRACTICE], 50))
    changed = load.copy()
    changed[EXAM] = 1e9
    assert rp.load_grid(changed, PRACTICE) == grid


def test_decide_splits_regimes_and_always_verifies_pinned():
    score = np.array([[0.1, 0.9, np.nan], [0.1, 0.9, 0.5]])
    pinned = np.array([[False, False, True], [False, False, False]])
    load = np.array([10.0, 20.0])
    # day 0 normal, day 1 heavy
    assert rp.decide("P1", score, load, pinned, 0.5).tolist() == [[False, True, True], [False, True, True]]
    assert rp.decide("P2", score, load, pinned, 0.5, None, 15.0).tolist() == [[False, True, True], [True, True, True]]
    assert rp.decide("P3", score, load, pinned, 0.95, 0.05, 15.0).tolist() == [[False, False, True], [True, True, True]]


def test_p2_verifies_all_80_on_heavy_days():
    d = _synthetic()
    fitted = rp.fit_all(d, PRACTICE)
    p2 = fitted["policies"]["P2"]
    assert p2.cutoff_heavy is None and p2.L is not None
    verify = p2.verify(d, EXAM)
    heavy = rp.heavy_days(cal.total_load(d)[EXAM], p2.L)
    assert heavy.any() and (~heavy).any()
    assert verify[heavy].all()
    assert verify.shape == (EXAM.size, C_)


def test_p2_with_80_contingencies_verifies_80_on_a_heavy_day():
    score = np.zeros((2, 80))
    v = rp.decide("P2", score, np.array([1.0, 2.0]), np.zeros((2, 80), bool), 0.5, None, 1.5)
    assert v.sum(axis=1).tolist() == [0, 80]


# ---------------------------------------------------------------- cutoffs
def test_regime_cutoffs_equal_task_a_recipe_on_the_regime_rows():
    d = _synthetic()
    fitted = rp.fit_all(d, PRACTICE)
    ps, D, P = fitted["scores"], d["dangerous"], d["pinned"]
    p3 = fitted["policies"]["P3"]
    heavy = rp.heavy_days(cal.total_load(d)[PRACTICE], p3.L)
    for rows, got in ((PRACTICE[~heavy], p3.cutoff_normal), (PRACTICE[heavy], p3.cutoff_heavy)):
        folds = [np.intersect1d(f, rows) for f in ps.folds]
        want = cal.zero_miss_threshold(ps.oof, D, P, rows) - cal.cv_shortfall(ps.oof, D, P, folds)["margin"]
        assert got == want


def test_p1_is_task_a_recipe_unchanged():
    d = _synthetic()
    folds = cal.make_folds(PRACTICE, 5, 0)
    fitter = lambda rows: cal.fit_on("bias", d, rows)  # noqa: E731
    scorer = lambda f: cal.calibrated_scores(f, d)  # noqa: E731
    oof = cal.cross_fitted_scores(fitter, scorer, folds, d["pinned"].shape)
    D, P = d["dangerous"], d["pinned"]
    thr = cal.zero_miss_threshold(oof, D, P, PRACTICE) - cal.cv_shortfall(oof, D, P, folds)["margin"]
    p1 = rp.fit_all(d, PRACTICE)["policies"]["P1"]
    assert p1.cutoff_normal == pytest.approx(thr, abs=1e-12)
    from p013.screening_metrics import cutoff
    want = cutoff(scorer(fitter(PRACTICE))[EXAM], thr) | P[EXAM]
    assert np.array_equal(p1.verify(d, EXAM), want)


def test_normal_cutoff_ignores_heavy_practice_rows_truth():
    """C_n is chosen on normal practice days only: heavy days' danger labels cannot move it."""
    d = _synthetic()
    ps = rp.practice_scores(d, PRACTICE)
    L = rp.load_grid(cal.total_load(d), PRACTICE)[50]
    heavy_rows = PRACTICE[rp.heavy_days(cal.total_load(d)[PRACTICE], L)]
    before = rp._fit_regimes_at("P3", d, ps, L)
    d2 = copy.deepcopy(d)
    d2["dangerous"][heavy_rows] = ~d2["pinned"][heavy_rows]  # every heavy case now dangerous
    after = rp._fit_regimes_at("P3", d2, ps, L)
    assert after.cutoff_normal == before.cutoff_normal
    assert after.cutoff_heavy != before.cutoff_heavy


def test_scores_at_matches_full_scoring_on_its_rows():
    d = _synthetic()
    f = cal.fit_on("bias", d, PRACTICE)
    full = cal.calibrated_scores(f, d)
    part = rp.scores_at(f, d, EXAM)
    assert np.array_equal(np.isnan(part[PRACTICE]), np.ones_like(part[PRACTICE], bool))
    np.testing.assert_array_equal(part[EXAM], full[EXAM])


# ---------------------------------------------------------------- selection
def test_l_is_chosen_by_lowest_oof_mean_verified():
    d = _synthetic()
    for name in ("P2", "P3"):
        pol = rp.fit_all(d, PRACTICE)["policies"][name]
        grid = pol.details["L_grid"]
        best = min(g["oof_mean_verified"] for g in grid)
        first = next(g for g in grid if g["oof_mean_verified"] == best)
        assert pol.percentile == first["percentile"] and pol.L == first["L"]


def test_primary_is_lowest_oof_mean_verified_ties_in_order():
    mk = lambda n, v: rp.FittedPolicy(n, {}, 0.0, oof_mean_verified=v)  # noqa: E731
    assert rp.select_primary({"P1": mk("P1", 30), "P2": mk("P2", 20), "P3": mk("P3", 25)}) == "P2"
    assert rp.select_primary({"P1": mk("P1", 20), "P2": mk("P2", 20), "P3": mk("P3", 20)}) == "P1"


def test_p3_never_costs_more_than_p2_on_practice_at_the_same_L():
    d = _synthetic()
    ps = rp.practice_scores(d, PRACTICE)
    for L in rp.load_grid(cal.total_load(d), PRACTICE).values():
        assert rp._fit_regimes_at("P3", d, ps, L).oof_mean_verified <= rp._fit_regimes_at("P2", d, ps, L).oof_mean_verified


def test_oof_mean_verified_is_measured_on_practice_oof_scores():
    d = _synthetic()
    fitted = rp.fit_all(d, PRACTICE)
    ps, p1 = fitted["scores"], fitted["policies"]["P1"]
    v = rp.decide("P1", ps.oof[PRACTICE], None, d["pinned"][PRACTICE], p1.cutoff_normal)
    assert p1.oof_mean_verified == rp.mean_verified(v, d["pinned"][PRACTICE])
    assert np.isnan(ps.oof[EXAM]).all()


# ---------------------------------------------------------------- no leakage
def test_no_leakage_exam_truth_changes_nothing_fitted_or_selected():
    d = _synthetic()
    before = _fingerprint(rp.fit_all(d, PRACTICE))
    d2 = copy.deepcopy(d)
    rng = np.random.default_rng(99)
    d2["truth_vm"][EXAM] = rng.uniform(0.5, 1.5, d2["truth_vm"][EXAM].shape)
    d2["truth_vm"][EXAM[0]] = np.nan
    d2["dangerous"][EXAM] = ~d2["dangerous"][EXAM]
    assert _fingerprint(rp.fit_all(d2, PRACTICE)) == before


def test_no_leakage_exam_inputs_do_not_move_l_or_cutoffs():
    """Exam model outputs and loads are runtime inputs; they still must not shape the fit."""
    d = _synthetic()
    before = _fingerprint(rp.fit_all(d, PRACTICE))
    d2 = copy.deepcopy(d)
    d2["model_vm"][EXAM] = 0.7
    d2["load_p_mw"][EXAM] = 1e6
    d2["model_thermal"][EXAM] = 5.0
    assert _fingerprint(rp.fit_all(d2, PRACTICE)) == before


def test_no_leakage_practice_truth_does_change_the_fit():
    """The fingerprint is sensitive: the same edit on practice rows moves it."""
    d = _synthetic()
    before = _fingerprint(rp.fit_all(d, PRACTICE))
    d2 = copy.deepcopy(d)
    d2["truth_vm"][PRACTICE[:10]] += 0.02
    assert _fingerprint(rp.fit_all(d2, PRACTICE)) != before


# ---------------------------------------------------------------- +inf regime
def test_regime_without_danger_gets_infinite_cutoff_and_verifies_only_pinned():
    d = _synthetic()
    ps = rp.practice_scores(d, PRACTICE)
    L = rp.load_grid(cal.total_load(d), PRACTICE)[50]
    normal_rows = PRACTICE[~rp.heavy_days(cal.total_load(d)[PRACTICE], L)]
    d2 = copy.deepcopy(d)
    d2["dangerous"][normal_rows] = d2["pinned"][normal_rows]  # no non-pinned danger on normal days
    pol = rp._fit_regimes_at("P3", d2, ps, L)
    assert np.isposinf(pol.cutoff_normal)
    assert pol.details["normal"]["margin"] == 0.0
    assert pol.infinite_cutoffs == 1
    verify = pol.verify(d2, EXAM)
    normal_exam = ~rp.heavy_days(cal.total_load(d2)[EXAM], L)
    assert normal_exam.any()
    assert np.array_equal(verify[normal_exam], d2["pinned"][EXAM][normal_exam])


def test_empty_regime_is_infinite_not_an_error():
    d = _synthetic()
    ps = rp.practice_scores(d, PRACTICE)
    r = rp.regime_cutoff(ps.oof, d["dangerous"], d["pinned"], np.array([], dtype=int), ps.folds)
    assert np.isposinf(r["cutoff"]) and r["n_days"] == 0


# ---------------------------------------------------------------- harness
def _harness():
    path = Path(__file__).resolve().parents[1] / "experiments" / "round2_robustness.py"
    spec = importlib.util.spec_from_file_location("round2_robustness", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _save(path, d, names):
    arrays = {k: np.asarray(v) for k, v in d.items()}
    np.savez(path, names=names, **arrays)
    path.with_suffix(".json").write_text(json.dumps({"weights_sha256": "w", "split": "x"}), encoding="utf-8")


def test_load_pool_concatenates_and_keeps_source_and_is_test(tmp_path):
    h = _harness()
    d = _synthetic()
    names = np.array([f"L{i}" for i in range(C_)])
    a, b = copy.deepcopy(d), copy.deepcopy(d)
    b["is_test"] = np.zeros(P_, bool)
    _save(tmp_path / "a.npz", a, names)
    _save(tmp_path / "b.npz", b, names)
    pool = h.load_pool([tmp_path / "a.npz", tmp_path / "b.npz"])
    assert pool["model_vm"].shape[0] == 2 * P_
    assert pool["source"].tolist() == [0] * P_ + [1] * P_
    assert pool["index_in_source"][P_] == 0
    assert pool["is_test"].sum() == d["is_test"].sum()
    np.testing.assert_array_equal(pool["truth_vm"][P_:], b["truth_vm"])


def test_load_pool_refuses_different_contingency_order(tmp_path):
    h = _harness()
    d = _synthetic()
    names = np.array([f"L{i}" for i in range(C_)])
    _save(tmp_path / "a.npz", d, names)
    _save(tmp_path / "b.npz", d, names[::-1])
    with pytest.raises(ValueError):
        h.load_pool([tmp_path / "a.npz", tmp_path / "b.npz"])
