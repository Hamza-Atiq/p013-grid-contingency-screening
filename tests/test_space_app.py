"""The Hugging Face Space's two buttons (spec 3.2)."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "space" / "app.py"
pytestmark = pytest.mark.skipif(not (ROOT / "model" / "live_policy.npz").exists(),
                                reason="run experiments/build_live_policy.py first")


def load_app():
    spec = importlib.util.spec_from_file_location("space_app", APP)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_screen_runs_on_a_fixed_seed():
    summary, table = load_app().screen(1000.0, 1)
    assert "sent to the solver" in summary and len(table) == 80
    solved = [r for r in table if r[3]]
    assert all(r[2] == "yes" for r in solved)          # only sent cases have a solver answer


def test_check_all_solves_every_case_and_reports_misses():
    summary, table = load_app().check_all(1000.0, 1)
    assert "Missed by the AI:" in summary
    assert all(r[3] for r in table)                    # every case has a solver answer


def test_unsolvable_intact_day_gives_a_plain_message(monkeypatch):
    app = load_app()
    from p013.solving import SolveResult

    monkeypatch.setattr(app.live, "solve_intact", lambda day: SolveResult.failed("no_solution"))
    summary, table = app.screen(1000.0, 1)
    assert "try another seed" in summary and table == []


def test_out_of_range_load_gives_a_plain_message():
    summary, table = load_app().screen(2000.0, 1)
    assert "735" in summary and table == []


def test_bad_seed_gives_a_seed_message_not_a_load_message():
    app = load_app()
    summary, table = app.screen(1000.0, -1)
    assert "seed" in summary.lower() and "735" not in summary and table == []


def test_blank_seed_is_treated_as_zero():
    app = load_app()
    blank, zero = app.screen(1000.0, None), app.screen(1000.0, 0)
    assert blank[1] == zero[1] and len(blank[1]) == 80    # same day, same table (timings differ)
