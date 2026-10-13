"""site/data.js, the only source of numbers for the static page (spec 3.1, 4)."""
import json
import re
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "site" / "data.js"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="run site/make_site.py first")


def load():
    text = DATA.read_text(encoding="utf-8").strip()
    return json.loads(re.sub(r"^window\.P013 = |;$", "", text))


def test_fixed_exam_has_175_days_and_no_misses():
    d = load()
    fixed = [x for x in d["days"] if x["group"] == "fixed"]
    assert len(fixed) == 175 and all(not x["missed"] for x in fixed)
    assert d["honest"]["fixedMissDays"] == 0 and d["honest"]["fixedExamDays"] == 175


def test_miss_group_is_the_4_distinct_days():
    d = load()
    missed = [x for x in d["days"] if x["group"] == "missed"]
    assert len(missed) == 4 == d["honest"]["distinctMissDays"]
    assert all(x["missed"] and x["examNo"] for x in missed)
    assert d["honest"]["missDays"] == 14 and d["honest"]["zeroMiss"] == 39
    assert d["honest"]["registeredZeroMiss"] == 31 and d["honest"]["passNeeded"] == 48


def test_sent_counts_and_rows_are_consistent():
    for x in load()["days"]:
        assert len(x["rows"]) == 80
        assert x["nSent"] == sum(r["sent"] for r in x["rows"])
        assert x["nDangerous"] == sum(r["dangerous"] for r in x["rows"])
        assert x["missed"] == [r["name"] for r in x["rows"] if r["dangerous"] and not r["sent"]]
        if x["heavy"]:
            assert x["nSent"] == 80


def test_timing_arithmetic():
    d = load()
    t = d["timing"]
    for x in d["days"]:
        screen = t["overhead"] + x["nSent"] * t["solve"]
        assert abs(x["allSeconds"] - t["baseline"]) < 1e-3
        assert abs(x["screenSeconds"] - screen) < 1e-3
        assert abs(x["speedup"] - t["baseline"] / screen) < 0.01


def test_miss_days_explain_score_vs_cutoff():
    for x in load()["days"]:
        if x["group"] == "missed":
            assert x["missNote"]
            assert all(m["score"] < m["cutoff"] and m["examNo"] for m in x["missNote"])


def test_layout_covers_the_grid():
    d = load()
    assert len(d["buses"]) == 57 and len(d["branches"]) == 80
    assert all(0 <= b["x"] <= 1 and 0 <= b["y"] <= 1 for b in d["buses"])


def test_miss_days_carry_their_own_heavy_threshold():
    for x in load()["days"]:
        if x["group"] == "missed":
            assert all(m["L"] > x["loadMw"] for m in x["missNote"])   # a normal day for that exam


def test_page_does_not_call_every_miss_close():
    page = (DATA.parent / "index.html").read_text(encoding="utf-8")
    assert "just below" not in page
