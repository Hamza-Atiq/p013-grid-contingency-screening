"""The live screen the Hugging Face Space runs (spec 3.2)."""
import subprocess
import sys

import numpy as np
import pytest

from p013 import live


def test_build_day_hits_target_load_at_both_ends():
    for target in live.LOAD_RANGE:
        day = live.build_day(seed=1, target_mw=target)
        assert abs(day.total_mw - target) < 0.1
        assert abs(day.load_p.sum() - target) < 0.1


def test_build_day_refuses_outside_range():
    with pytest.raises(ValueError):
        live.build_day(seed=1, target_mw=700.0)
    with pytest.raises(ValueError):
        live.build_day(seed=1, target_mw=1600.0)


def test_build_day_is_deterministic():
    a, b = live.build_day(7, 1000.0), live.build_day(7, 1000.0)
    assert np.array_equal(a.load_p, b.load_p) and np.array_equal(a.gen_p, b.gen_p)


def test_live_does_not_import_torch():
    code = "import sys; import p013.live; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_nonconverged_sent_case_counts_as_dangerous(monkeypatch):
    from p013.solving import SolveResult

    monkeypatch.setattr(live, "solve", lambda net: SolveResult.failed("no_solution"))
    day = live.build_day(3, 1000.0)
    res = live.solve_cases(day, ["L42"], vm_before=np.ones(57))
    assert res["L42"].status == "no_solution" and res["L42"].dangerous is True


def test_overlapping_requests_give_the_same_answers_as_one_at_a_time():
    from concurrent.futures import ThreadPoolExecutor

    def job(seed):
        day = live.build_day(seed, 1100.0)
        intact = live.solve_intact(day)
        r = live.solve_cases(day, ["L01", "L10", "L20", "T03"], vm_before=intact.vm_pu)
        return day.load_p.tolist(), {k: (v.status, v.dangerous) for k, v in r.items()}

    alone = [job(s) for s in (1, 2, 3, 4)]
    with ThreadPoolExecutor(4) as pool:
        together = list(pool.map(job, (1, 2, 3, 4)))
    assert together == alone
