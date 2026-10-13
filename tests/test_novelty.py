"""Task D's detector pieces: labels, runtime signals, and fitting that cannot leak."""
import numpy as np
import pytest

from p013.novelty import (
    TRAIN_KEYS,
    TrainingReference,
    auc,
    bad_points,
    fit_flag_policy,
    flag_threshold,
    flagged_policy,
    k_star,
    load_training_inputs,
    min_dangerous_score,
    runtime_signals,
    watchlist,
)
from p013.screening_metrics import Timing

TIMING = Timing(solve_s=0.05, overhead_s=0.5, baseline_s=4.0, n_contingencies=80)


def _world(n_points=40, n_cont=20, seed=0):
    """A small random screening world with a rank per point that puts pinned first."""
    rng = np.random.default_rng(seed)
    pinned = rng.random((n_points, n_cont)) < 0.05
    dangerous = rng.random((n_points, n_cont)) < 0.2
    score = rng.random((n_points, n_cont)) * 2
    score[pinned] = np.nan
    key = np.where(pinned, np.inf, np.nan_to_num(score, nan=0.0))
    order = np.argsort(-key, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(n_points)[:, None]
    rank[rows, order] = np.arange(1, n_cont + 1)
    signal = rng.normal(size=n_points)
    fit_mask = np.zeros(n_points, dtype=bool)
    fit_mask[: n_points // 2] = True
    return dict(pinned=pinned, dangerous=dangerous, score=score, rank=rank, signal=signal, fit=fit_mask)


# ---- labels -----------------------------------------------------------------


def test_k_star_is_the_deepest_dangerous_rank_with_pinned_counted():
    rank = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    dangerous = np.array([[False, False, True, False], [False] * 4, [False] * 4])
    pinned = np.array([[False] * 4, [True, True, False, False], [False] * 4])
    assert k_star(rank, dangerous, pinned).tolist() == [3, 2, 0]


def test_bad_points_means_top_k_would_miss():
    assert bad_points(np.array([5, 12, 13]), 12).tolist() == [False, False, True]


def test_auc_perfect_reversed_and_ties():
    y = np.array([False, False, True, True])
    assert auc(np.array([0, 1, 2, 3]), y) == 1.0
    assert auc(np.array([3, 2, 1, 0]), y) == 0.0
    assert auc(np.ones(4), y) == 0.5
    assert np.isnan(auc(np.ones(2), np.array([True, True])))


# ---- runtime signals ------------------------------------------------------------


def _training(n=300, seed=1):
    rng = np.random.default_rng(seed)
    return rng.random((n, 42)) * 30, rng.random((n, 42)) * 10, rng.random((n, 6)) * 200


def test_training_reference_distances_behave():
    lp, lq, gp = _training()
    ref = TrainingReference.fit(lp, lq, gp)
    # A training point is its own nearest neighbour.
    assert ref.knn_distance(lp[:3], lq[:3], gp[:3], k=1) == pytest.approx(0.0, abs=1e-6)
    # The training mean is the least unusual input there is.
    centre = ref.mahalanobis(lp.mean(0, keepdims=True), lq.mean(0, keepdims=True), gp.mean(0, keepdims=True))
    far = ref.mahalanobis(lp[:1] * 3, lq[:1] * 3, gp[:1] * 3)
    assert centre[0] == pytest.approx(0.0, abs=1e-6) and far[0] > centre[0]
    pct = ref.load_percentile(np.vstack([lp.min(0) * 0, lp * 0 + 1e6]))
    assert pct[0] == 0.0 and pct[-1] == 1.0


def test_load_training_inputs_reads_only_the_input_keys(tmp_path):
    lp, lq, gp = _training(10)
    path = tmp_path / "train.npz"
    np.savez(path, load_p_mw=lp, load_q_mvar=lq, gen_p_mw=gp,
             in_service=np.ones((10, 80), bool), vm_pu=np.ones((10, 57)))
    assert set(load_training_inputs(path)) == set(TRAIN_KEYS)


def test_runtime_signals_never_read_truth():
    """Built from a cache with every truth key removed: it must still work."""
    rng = np.random.default_rng(2)
    lp, lq, gp = _training()
    ref = TrainingReference.fit(lp, lq, gp)
    P, C = 5, 80
    cache = {
        "load_p_mw": lp[:P], "load_q_mvar": lq[:P], "gen_p_mw": gp[:P],
        "model_score": rng.random((P, C)) * 2,
        "model_vm": 0.9 + 0.1 * rng.random((P, 81, 57)),
        "model_loading": 100 * rng.random((P, 81, C)),
        "pinned": rng.random((P, C)) < 0.05,
    }
    sig = runtime_signals(cache, ref)
    assert all(v.shape == (P,) for v in sig.values())
    assert sig["n_scored_dangerous_pred"].tolist() == (cache["model_score"] > 1).sum(1).tolist()


# ---- fitting ------------------------------------------------------------------


def test_flag_threshold_catches_every_bad_fit_row_and_margin_lowers_it():
    s = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    bad = np.array([False, True, False, True, True])
    fit = np.array([True, True, True, True, False])
    t = flag_threshold(s, bad, fit)
    assert t == 2.0  # the test-side bad row at 10 is not consulted
    assert flag_threshold(s, bad, fit, margin_sd=1.0) < t
    assert flag_threshold(s, np.zeros(5, bool), fit) == float("inf")


def test_flagged_policy_modes():
    rank = np.array([[1, 2, 3], [1, 2, 3]])
    flagged = np.array([False, True])
    assert flagged_policy(rank, flagged, 1, "all").tolist() == [[True, False, False], [True, True, True]]
    W = np.array([False, False, True])
    assert flagged_policy(rank, flagged, 1, "extra", W).tolist() == [[True, False, False], [True, False, True]]
    with pytest.raises(ValueError):
        flagged_policy(rank, flagged, 1, "nope")


def test_fitted_flag_policy_has_zero_misses_on_its_fit_rows():
    w = _world()
    fit = fit_flag_policy(w["signal"], w["rank"], w["dangerous"], w["pinned"], w["fit"], TIMING, range(1, 21))
    flagged = w["signal"] >= fit["threshold"]
    verify = flagged_policy(w["rank"], flagged, fit["k_normal"], "all") | w["pinned"]
    missed = (w["dangerous"] & ~verify)[w["fit"]]
    assert missed.sum() == 0


def test_fitting_uses_only_fit_rows():
    """Scramble every truth-bearing array and the signal on non-fit rows: nothing fitted may change."""
    w = _world()
    rng = np.random.default_rng(99)
    other = ~w["fit"]
    v = {k: (x.copy() if isinstance(x, np.ndarray) else x) for k, x in w.items()}
    v["dangerous"][other] = rng.random(v["dangerous"][other].shape) < 0.9
    v["pinned"][other] = False
    v["signal"][other] = rng.normal(size=other.sum()) * 100
    v["rank"][other] = np.argsort(rng.random(v["rank"][other].shape), axis=1) + 1
    v["score"][other] = rng.random(v["score"][other].shape) * 0.01

    def fitted(x):
        f = fit_flag_policy(x["signal"], x["rank"], x["dangerous"], x["pinned"], x["fit"], TIMING, range(1, 21), 0.25)
        mk = k_star(x["rank"], x["dangerous"], x["pinned"])
        return (
            f,
            flag_threshold(x["signal"], bad_points(mk, 5), x["fit"], 0.25),
            watchlist(x["rank"], x["dangerous"], x["pinned"], x["fit"], np.ones(len(mk), bool), 5).tolist(),
            min_dangerous_score(x["score"], x["dangerous"], x["pinned"], x["fit"]),
        )

    assert fitted(w) == fitted(v)
