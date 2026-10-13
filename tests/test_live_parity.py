"""The live Space is the same model and policy as the recorded results (spec 5, 2a)."""
import json
from pathlib import Path

import numpy as np
import pytest

from p013 import calibration as cal
from p013 import live
from p013.forward_numpy import load_weights

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "eval" / "cache" / "predictions.npz"
SWEEP = ROOT / "data" / "test_sweep.npz"
POLICY = ROOT / "model" / "live_policy.npz"
WEIGHTS = ROOT / "model" / "weights.npz"

pytestmark = pytest.mark.skipif(
    not (CACHE.exists() and SWEEP.exists() and POLICY.exists()),
    reason="needs the downloaded data and model/live_policy.npz",
)

EXAM_ROWS = 5


def _day(z, sweep, row):
    base = int(z["point"][row]) * 81
    return live.Day(sweep["load_p_mw"][base], sweep["load_q_mvar"][base], sweep["gen_p_mw"][base])


def test_policy_matches_fixed_exam_record():
    rec = json.loads((ROOT / "experiments/results/round2_main_original.json").read_text())
    p2 = rec["test3_fixed_exam"]["policies"]["P2"]
    pol = live.load_policy(POLICY)
    assert abs(pol.L - p2["L"]) < 1e-9 and abs(pol.cutoff - p2["cutoff_normal"]) < 1e-9
    assert [n for n, a in zip(live.NAMES, pol.always) if a] == [
        "L37", "L40", "L41", "L42", "L44", "T05", "T06"]


def test_raw_model_output_matches_cache_for_all_81_rows():
    """Before any mask: voltages for the intact grid and all 80 outages."""
    w = load_weights(WEIGHTS)
    z, sweep = dict(np.load(CACHE)), np.load(SWEEP)
    for row in np.flatnonzero(z["is_test"])[:EXAM_ROWS]:
        vm, _, thermal = live.model_voltages(_day(z, sweep, row), w)
        np.testing.assert_allclose(vm, z["model_vm"][row], atol=1e-5)
        cached = z["model_thermal"][row]
        assert np.array_equal(np.isnan(cached), z["pinned"][row])   # NaN exactly where pinned
        keep = ~z["pinned"][row]
        np.testing.assert_allclose(thermal[keep], cached[keep], atol=1e-5)


def test_live_scores_and_decisions_match_recorded_run():
    pol, w = live.load_policy(POLICY), load_weights(WEIGHTS)
    z, sweep = dict(np.load(CACHE)), np.load(SWEEP)
    exam = np.flatnonzero(z["is_test"])[:EXAM_ROWS]
    want = cal.calibrated_scores(
        {"kind": "bias", "bias": pol.bias},
        {k: z[k][exam] for k in ("model_vm", "model_thermal", "pinned", "load_p_mw")},
    )
    for j, row in enumerate(exam):
        got = live.score_day(_day(z, sweep, row), w, pol)
        # NaN exactly on the 7 always-send cases; recorded pins are a subset of them.
        assert np.array_equal(np.isnan(got.score), pol.always)
        assert not (z["pinned"][row] & ~pol.always).any()
        rest = ~pol.always
        np.testing.assert_allclose(got.score[rest], want[j][rest], atol=1e-5)
        # Same cases sent as the recorded run (spec 3.2: measured identical).
        by_score = np.ones(80, bool) if got.heavy else np.nan_to_num(want[j], nan=-1.0) >= pol.cutoff
        assert np.array_equal(got.sent, pol.always | z["pinned"][row] | by_score)
