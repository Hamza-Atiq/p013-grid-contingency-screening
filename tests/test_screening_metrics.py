"""The shared definitions every research-round task reports through."""
import json
from pathlib import Path

import numpy as np
import pytest

from p013.screening_metrics import Timing, cutoff, evaluate_policy, top_k

TIMING = Timing(solve_s=0.05, overhead_s=0.5, baseline_s=4.0, n_contingencies=80)


def _arrays(dangerous, pinned=None):
    dangerous = np.array(dangerous, dtype=bool)
    pinned = np.zeros_like(dangerous) if pinned is None else np.array(pinned, dtype=bool)
    return dangerous, pinned


def test_verifying_everything_catches_everything_at_full_cost_plus_overhead():
    d, p = _arrays([[True, False, False, True]])
    r = evaluate_policy(np.ones_like(d), d, p, TIMING)
    assert r["recall"] == 1.0
    assert r["points_with_miss"] == 0
    assert r["mean_seconds"] == pytest.approx(0.5 + 4 * 0.05)


def test_a_missed_case_is_counted_once_per_point_and_in_recall():
    d, p = _arrays([[True, True, False], [True, False, False]])
    verify = np.array([[True, False, False], [True, False, False]])
    r = evaluate_policy(verify, d, p, TIMING)
    assert r["missed_cases"] == 1
    assert r["points_with_miss"] == 1
    assert r["recall"] == pytest.approx(2 / 3)


def test_pinned_cases_are_always_solved_and_always_dangerous():
    """A policy cannot skip a pinned case to look cheaper, nor dodge it as a miss."""
    d, p = _arrays([[False, False, False]], pinned=[[True, False, False]])
    r = evaluate_policy(np.zeros_like(d), d, p, TIMING)
    assert r["mean_verified"] == 1
    assert r["missed_cases"] == 0
    assert r["recall"] == 1.0


def test_speedup_uses_the_mean_cost_across_points():
    d, p = _arrays([[False] * 4, [False] * 4])
    verify = np.array([[True, False, False, False], [True, True, True, True]])
    r = evaluate_policy(verify, d, p, TIMING)
    mean = ((0.5 + 0.05) + (0.5 + 0.2)) / 2
    assert r["speedup"] == pytest.approx(4.0 / mean)


def test_overhead_can_be_overridden_for_a_cheaper_screen():
    d, p = _arrays([[False, False]])
    r = evaluate_policy(np.zeros_like(d), d, p, TIMING, overhead_s=0.1)
    assert r["mean_seconds"] == pytest.approx(0.1)


def test_mismatched_shapes_are_an_error():
    with pytest.raises(ValueError):
        evaluate_policy(np.zeros((2, 3)), np.zeros((2, 4)), np.zeros((2, 3)), TIMING)


def test_top_k_and_cutoff():
    rank = np.array([[1, 3, 2]])
    assert top_k(rank, 2).tolist() == [[True, False, True]]
    score = np.array([[np.nan, 0.4, 1.2]])
    assert cutoff(score, 0.5).tolist() == [[False, False, True]]


def test_timing_is_derived_the_way_eval_notes_derives_it():
    results = {
        "n_contingencies": 80,
        "timing": {
            "baseline_full_ac_sweep_seconds_median": 4.0056,
            "ours_seconds_median": 4.2307,
            "k_deploy_solved": 72,
        },
    }
    t = Timing.from_results(results)
    assert t.solve_s == pytest.approx(0.0501, abs=1e-4)
    assert t.overhead_s == pytest.approx(0.626, abs=1e-3)
    assert t.overhead_in_solves == pytest.approx(12.5, abs=0.1)


CACHE = Path("eval/cache/predictions.npz")


@pytest.mark.skipif(not CACHE.exists(), reason="cache not built; experiments/build_cache.py")
def test_cache_reproduces_eval_notes_section_4():
    from p013 import prediction_cache

    cache = prediction_cache.load(CACHE)
    timing = Timing.from_results(json.loads(Path("eval/results.json").read_text(encoding="utf-8")))
    t = cache["is_test"]
    for k, (recall, missed, speedup) in {12: (0.804, 82, 3.27), 26: (0.961, 53, 2.08), 72: (1.0, 0, 0.95)}.items():
        r = evaluate_policy(top_k(cache["model_rank"][t], k), cache["dangerous"][t], cache["pinned"][t], timing)
        assert round(r["recall"], 3) == recall
        assert r["points_with_miss"] == missed
        assert round(r["speedup"], 2) == speedup
