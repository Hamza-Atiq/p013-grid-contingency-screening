# P-013 Live Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put P-013 online as (1) a static GitHub Pages site that plays back the recorded model + solver results for every exam day, and (2) a Hugging Face Space that screens a fresh day live with the real model and the real solver.

**Architecture:** One new module `src/p013/live.py` holds everything the Space needs (build a day, score it with the NumPy forward pass and the P2 policy, solve cases). A frozen file `model/live_policy.npz`, built and checked by a script, carries the fitted policy. `site/make_site.py` reuses `demo/make_demo.py`'s proven refit-and-check functions to write `site/data.js`; `site/index.html` only displays it. `space/` is a thin Gradio wrapper over `p013.live`.

**Tech Stack:** Python 3.11, NumPy, pandapower 3.5.4, networkx 3.6.1 (layout), Gradio 5.9.1 (Space only), plain HTML/CSS/JS (no libraries), GitHub Actions for Pages.

**Spec:** `docs/2026-10-13-live-page-spec.md`

## Global Constraints

- Design and test every page at **1024 x 768**.
- The static page computes nothing about results; every number comes from `site/data.js`, written by `site/make_site.py`.
- `make_site.py` stops with an error unless the fixed exam and all 50 re-split exams reproduce `experiments/results/round2_main_original.json` to **1e-9**.
- Space load slider: **735 to 1,550 MW**.
- Space policy: P2 with the fixed-exam fit (L = 1,274.2 MW, cutoff 0.784) plus "always send the 7 cases that failed to solve on any practice day" (L37, L40, L41, L42, L44, T05, T06).
- Space parity: for 5 exam days, live scores equal the calibrated cached scores to **1e-5**.
- Space must not import torch (keeps it light); it uses `p013.forward_numpy`.
- No em or en dashes in any page or README text (house style).
- Hamza's rule: **no git add, commit or push until he says "commit"**. Steps below that publish are marked GATED.
- Plain English on every page; Hamza must be able to explain each line.

## Review Focus

1. **A seed whose intact day does not solve** in the Space: expect a plain "this day has no solution, try another seed" message, no traceback. Test in Task 1.
2. **Slider at the exact ends (735, 1,550 MW)**: expect a valid day whose total load equals the slider to 0.1 MW. Test in Task 1.
3. **A sent case that does not converge**: expect "no solution, counted as dangerous" and it counts as caught, not missed. Test in Task 1.
4. **Opening `site/index.html` from `file://`** (double-click) as well as from Pages: expect it to work with no fetch. Checked in Task 4 by loading data with a `<script>` tag, not `fetch`.
5. **Clicking days quickly / switching days mid-animation**: expect the newest day to win, never a mix of two days. Handled in Task 4 by a run token.

---

### Task 1: `p013.live`, the live screening core

**Files:**
- Create: `src/p013/live.py`
- Test: `tests/test_live.py`

**Interfaces:**
- Produces:
  - `Day` dataclass: `load_p: np.ndarray[42]`, `load_q: np.ndarray[42]`, `gen_p: np.ndarray[6]`, `total_mw: float`
  - `build_day(seed: int, target_mw: float) -> Day` (raises `ValueError` outside `LOAD_RANGE`)
  - `LOAD_RANGE = (735.0, 1550.0)`
  - `Policy` dataclass: `bias: np.ndarray[81,57]`, `L: float`, `cutoff: float`, `always: np.ndarray[80] bool`
  - `load_policy(path) -> Policy`, `save_policy(path, policy)`
  - `score_day(day, weights, policy) -> Screen` with `Screen.score: np.ndarray[80]` (NaN where `always`), `Screen.sent: np.ndarray[80] bool`, `Screen.heavy: bool`, `Screen.seconds: float`
  - `solve_cases(day, names: list[str]) -> dict[str, CaseResult]`, `CaseResult(status: str, dangerous: bool, seconds: float)`; `solve_intact(day) -> SolveResult`
  - `NAMES: tuple[str, ...]` (80 canonical names)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_live.py
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


def test_build_day_is_deterministic():
    a, b = live.build_day(7, 1000.0), live.build_day(7, 1000.0)
    assert np.array_equal(a.load_p, b.load_p) and np.array_equal(a.gen_p, b.gen_p)


def test_live_does_not_import_torch():
    import subprocess, sys
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
```

- [ ] **Step 2: Run them, expect FAIL** (`ModuleNotFoundError: p013.live`)

Run: `.venv\Scripts\python.exe -m pytest tests/test_live.py -v`

- [ ] **Step 3: Implement `src/p013/live.py`**

```python
"""Screen one fresh operating point, live. Used by the Hugging Face Space.

Mirrors the recorded pipeline exactly: ``evaluate.model_ranking`` for the
model's voltages, ``calibration.calibrated_scores`` for the bias-corrected
score, P2 for the decision, ``evaluate.truth_ranking`` for the solver's
answer. ``tests/test_live_parity.py`` proves the scores match the caches.

NumPy only. Importing torch here would make the Space several hundred MB
heavier for nothing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from p013.admittance import loading_percent
from p013.case_loader import CaseTemplate, parse_matpower
from p013.danger import RATING_PERCENT, excess, violations
from p013.dataset import to_bus_order
from p013.features import branch_features, bus_features, edge_index
from p013.forward_numpy import forward
from p013.naming import branch_names, set_in_service
from p013.sampling import sample_operating_point
from p013.solving import SolveResult, solve
import p013
import threading

#: CaseTemplate.fresh() hands back ONE shared mutable network. Every use of
#: it (sampling, solving) runs under this lock, so two Space visitors can
#: never reset each other's loads or outages mid-solve.
LOCK = threading.RLock()

#: Total load seen in the 750 generated days was 732.3 to 1,553.2 MW.
LOAD_RANGE = (735.0, 1550.0)

_TEMPLATE = CaseTemplate()
_PPC = parse_matpower(p013.CASE_PATH)
NAMES = tuple(branch_names(_TEMPLATE.fresh()))
_LOAD_BUS = [int(b[0]) for b in _PPC["bus"] if b[2] != 0 or b[3] != 0]
_GEN_BUS = [int(b) for b in _TEMPLATE.fresh().gen.bus]
_EDGES = edge_index(_PPC)


@dataclass
class Day:
    load_p: np.ndarray
    load_q: np.ndarray
    gen_p: np.ndarray

    @property
    def total_mw(self) -> float:
        return float(self.load_p.sum())


def build_day(seed: int, target_mw: float) -> Day:
    """The training sampler, then one common factor so total load = target."""
    lo, hi = LOAD_RANGE
    if not lo <= target_mw <= hi:
        raise ValueError(f"total load must be {lo:.0f} to {hi:.0f} MW, got {target_mw}")
    with LOCK:
        net = _TEMPLATE.fresh()
        sample_operating_point(net, np.random.default_rng(seed))
        k = target_mw / float(net.load.p_mw.sum())
        return Day(
            load_p=net.load.p_mw.to_numpy() * k,
            load_q=net.load.q_mvar.to_numpy() * k,
            gen_p=net.gen.p_mw.to_numpy() * k,
        )


@dataclass
class Policy:
    bias: np.ndarray      # [81, 57] voltage bias, P2's calibration
    L: float              # heavy-day threshold, MW
    cutoff: float         # normal-day score cutoff
    always: np.ndarray    # [80] bool, cases that failed to solve on a practice day


def save_policy(path: Path, p: Policy) -> None:
    np.savez(path, bias=p.bias, L=p.L, cutoff=p.cutoff, always=p.always)


def load_policy(path: Path) -> Policy:
    with np.load(path) as z:
        return Policy(z["bias"], float(z["L"]), float(z["cutoff"]), z["always"].astype(bool))


@dataclass
class Screen:
    score: np.ndarray
    sent: np.ndarray
    heavy: bool
    seconds: float


def _inputs(day: Day):
    from p013.evaluate import _with_slack  # pure NumPy helper, no torch
    load_p = to_bus_order(_PPC, _LOAD_BUS, day.load_p[None])[0]
    load_q = to_bus_order(_PPC, _LOAD_BUS, day.load_q[None])[0]
    gen_p = _with_slack(_PPC, _GEN_BUS, to_bus_order(_PPC, _GEN_BUS, day.gen_p[None])[0])
    return bus_features(_PPC, load_p, load_q, gen_p)


def model_voltages(day: Day, weights: dict):
    """``(vm, va, thermal)``: [81,57], [81,57], [80]. Row 0 is the intact grid."""
    bus_one = _inputs(day)
    n = len(NAMES) + 1
    in_service = np.ones((n, len(NAMES)), dtype=bool)
    for c in range(len(NAMES)):
        in_service[c + 1, c] = False
    bus_x = np.repeat(bus_one[None].astype(np.float32), n, axis=0)
    branch_x = np.stack([branch_features(_PPC, s) for s in in_service]).astype(np.float32)
    vm, va = forward(weights, bus_x, branch_x, _EDGES, slack_mask=bus_one[:, 4].astype(np.float32))
    thermal = np.array([
        loading_percent(_PPC, vm[o] * np.exp(1j * va[o]), in_service[o]).max() / RATING_PERCENT
        for o in range(1, n)
    ])
    return vm, va, thermal


def score_day(day: Day, weights: dict, policy: Policy) -> Screen:
    t0 = time.perf_counter()
    vm, _, thermal = model_voltages(day, weights)
    v = vm - policy.bias                                   # calibration "bias"
    voltage = (excess(v[1:]) - excess(v[:1])).max(axis=1)
    score = np.where(policy.always, np.nan, np.maximum(thermal, voltage))
    heavy = day.total_mw >= policy.L
    sent = policy.always | (np.ones(len(NAMES), bool) if heavy else (score >= policy.cutoff))
    return Screen(score, sent, heavy, time.perf_counter() - t0)


@dataclass
class CaseResult:
    status: str
    dangerous: bool
    seconds: float


def _net(day: Day, outage: str | None):
    net = _TEMPLATE.fresh()
    net.load["p_mw"] = day.load_p.copy()
    net.load["q_mvar"] = day.load_q.copy()
    net.gen["p_mw"] = day.gen_p.copy()
    if outage:
        set_in_service(net, outage, False)
    return net


def solve_intact(day: Day) -> SolveResult:
    with LOCK:
        return solve(_net(day, None))


def solve_cases(day: Day, names, vm_before: np.ndarray) -> dict[str, CaseResult]:
    """The solver's answer, judged exactly as ``evaluate.truth_ranking``."""
    out = {}
    for name in names:
        t0 = time.perf_counter()
        with LOCK:
            r = solve(_net(day, name))
        if not r.ok:   # islanded or no solution: counted as dangerous
            out[name] = CaseResult(r.status, True, time.perf_counter() - t0)
            continue
        in_service = np.ones(len(NAMES), bool)
        in_service[NAMES.index(name)] = False
        v = r.vm_pu * np.exp(1j * np.deg2rad(r.va_degree))
        loading = loading_percent(_PPC, v, in_service)
        out[name] = CaseResult("converged", violations(r.vm_pu, loading, vm_before).is_dangerous,
                               time.perf_counter() - t0)
    return out
```

Note: `CaseTemplate.fresh()` resets in-service flags, loads and generation (see `case_loader.py`). If `p013.evaluate` or `p013.dataset` imports torch at module level, move `_with_slack` and `to_bus_order` imports to a torch-free path (copy the two small functions into `live.py` with a comment pointing at the originals) until `test_live_does_not_import_torch` passes.

- [ ] **Step 4: Run, expect PASS**

Run: `.venv\Scripts\python.exe -m pytest tests/test_live.py -v`

---

### Task 2: freeze the live policy, and prove parity with the recorded results

**Files:**
- Create: `experiments/build_live_policy.py`, output `model/live_policy.npz`
- Test: `tests/test_live_parity.py`

**Interfaces:**
- Consumes: `live.Policy`, `live.save_policy`, `live.load_policy`, `live.model_voltages`, `live.score_day`, `live.Day`
- Produces: `model/live_policy.npz` (bias [81,57], L, cutoff, always[80])

- [ ] **Step 1: Write the failing parity test**

```python
# tests/test_live_parity.py
from pathlib import Path
import numpy as np
import pytest

from p013 import calibration as cal, live
from p013.forward_numpy import load_weights

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "eval" / "cache" / "predictions.npz"
SWEEP = ROOT / "data" / "test_sweep.npz"
POLICY = ROOT / "model" / "live_policy.npz"

pytestmark = pytest.mark.skipif(not (CACHE.exists() and SWEEP.exists() and POLICY.exists()),
                                reason="needs the downloaded data and model/live_policy.npz")


def test_policy_matches_fixed_exam_record():
    import json
    rec = json.loads((ROOT / "experiments/results/round2_main_original.json").read_text())
    p2 = rec["test3_fixed_exam"]["policies"]["P2"]
    pol = live.load_policy(POLICY)
    assert abs(pol.L - p2["L"]) < 1e-9 and abs(pol.cutoff - p2["cutoff_normal"]) < 1e-9
    assert [n for n, a in zip(live.NAMES, pol.always) if a] == ["L37", "L40", "L41", "L42", "L44", "T05", "T06"]


EXAM_ROWS = 5


def _day(z, sweep, row):
    base = int(z["point"][row]) * 81
    return live.Day(sweep["load_p_mw"][base], sweep["load_q_mvar"][base], sweep["gen_p_mw"][base])


def test_raw_model_output_matches_cache_for_all_81_rows():
    """Before any mask: voltages for the intact grid and all 80 outages."""
    w = load_weights(ROOT / "model" / "weights.npz")
    z, sweep = dict(np.load(CACHE)), np.load(SWEEP)
    for row in np.flatnonzero(z["is_test"])[:EXAM_ROWS]:
        vm, _, thermal = live.model_voltages(_day(z, sweep, row), w)
        np.testing.assert_allclose(vm, z["model_vm"][row], atol=1e-5)
        cached = z["model_thermal"][row]
        assert np.array_equal(np.isnan(cached), z["pinned"][row])   # NaN exactly where pinned
        keep = ~z["pinned"][row]
        np.testing.assert_allclose(thermal[keep], cached[keep], atol=1e-5)


def test_live_scores_and_decisions_match_recorded_run():
    pol, w = live.load_policy(POLICY), load_weights(ROOT / "model" / "weights.npz")
    z, sweep = dict(np.load(CACHE)), np.load(SWEEP)
    exam = np.flatnonzero(z["is_test"])[:EXAM_ROWS]
    want = cal.calibrated_scores({"kind": "bias", "bias": pol.bias},
                                 {k: z[k][exam] for k in ("model_vm", "model_thermal", "pinned", "load_p_mw")})
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
```

(If `calibration.apply` names the bias key differently, read `calibration.fit` for kind `"bias"` and use the same key in both `save_policy` and this test.)

- [ ] **Step 2: Run, expect SKIP or FAIL** (policy file missing)

- [ ] **Step 3: Implement `experiments/build_live_policy.py`**

```python
"""Freeze the policy the Space uses: P2 fitted on the fixed practice days.

Refits with the same code as round 2 and refuses to write unless the fit
equals the recorded one. Adds the 'always send' set: every case that failed
to solve on any practice day (spec 3.2; measured to change nothing).

    .venv/Scripts/python.exe experiments/build_live_policy.py
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
from round2_robustness import load_pool  # noqa: E402
from p013 import live, regime_policy as rp  # noqa: E402

rec = json.loads((ROOT / "experiments/results/round2_main_original.json").read_text())["test3_fixed_exam"]["policies"]["P2"]
pool = load_pool([ROOT / "eval/cache/predictions.npz", ROOT / "eval/cache/practice_predictions.npz"])
practice = np.flatnonzero(~pool["is_test"])
pol = rp.fit_policy("P2", pool, rp.practice_scores(pool, practice))
for k, v in (("L", rec["L"]), ("cutoff_normal", rec["cutoff_normal"])):
    if abs(getattr(pol, k) - v) > 1e-9:
        raise SystemExit(f"{k} {getattr(pol, k)} differs from the recorded {v}")
always = pool["pinned"][practice].any(axis=0)
live.save_policy(ROOT / "model/live_policy.npz",
                 live.Policy(np.asarray(pol.calibration["bias"]), float(pol.L), float(pol.cutoff_normal), always))
print("written model/live_policy.npz; always send:", [n for n, a in zip(live.NAMES, always) if a])
```

- [ ] **Step 4: Run the script, then the parity tests, expect PASS**

Run: `.venv\Scripts\python.exe experiments\build_live_policy.py` then `.venv\Scripts\python.exe -m pytest tests/test_live_parity.py tests/test_live.py -v`

---

### Task 3: `site/make_site.py`, the checked data for the static page

**Files:**
- Create: `site/make_site.py`, output `site/data.js`
- Modify: `demo/make_demo.py:52` (`OUT = ROOT / "site" / "results.html"`), delete root `demo.html` after the rebuild
- Test: `tests/test_site_data.py`

**Interfaces:**
- Consumes: from `demo/make_demo.py`: `fit_p2`, `check_exam`, `day_payload`, `check_against_model`, constants `CACHES`, `RESULTS`, `POLICY`; from `round2_robustness`: `load_pool`, `score_exam`, `N_PRACTICE`, `N_EXAM`; `verify_resplit.clean_timings`
- Produces: `site/data.js` defining `window.P013 = {...}` with keys:
  `case, n, timing{solve,overhead,baseline}, policy{L,cutoff}, buses[{id,x,y,gen}], branches[{name,a,b}], days[{label,group,loadMw,heavy,nSent,nDangerous,missed[],missNote[{name,score,cutoff,examNo}],examNo,cutoff,allSeconds,screenSeconds,speedup,rows[{name,score,pinned,dangerous,sent}]}], honest{fixedMissDays, fixedExamDays, zeroMiss, nExams, passNeeded, missDays, distinctMissDays, registeredZeroMiss, speedupMedian, fixedSpeedup}, links{space,github,results}`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_site_data.py
import json, re
from pathlib import Path
import pytest

DATA = Path(__file__).resolve().parents[1] / "site" / "data.js"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="run site/make_site.py first")


def load():
    text = DATA.read_text(encoding="utf-8")
    return json.loads(re.sub(r"^window\.P013 = |;\s*$", "", text.strip()))


def test_fixed_exam_has_175_days_and_no_misses():
    d = load()
    fixed = [x for x in d["days"] if x["group"] != "missed"]
    assert len(fixed) == 175 and all(not x["missed"] for x in fixed)


def test_miss_group_is_the_4_distinct_days():
    d = load()
    missed = [x for x in d["days"] if x["group"] == "missed"]
    assert len(missed) == 4 == d["honest"]["distinctMissDays"]
    assert all(x["missed"] and x["examNo"] for x in missed)
    assert d["honest"]["missDays"] == 14 and d["honest"]["zeroMiss"] == 39


def test_sent_counts_and_rows_are_consistent():
    for x in load()["days"]:
        assert len(x["rows"]) == 80
        assert x["nSent"] == sum(r["sent"] for r in x["rows"])
        assert x["missed"] == [r["name"] for r in x["rows"] if r["dangerous"] and not r["sent"]]
        if x["heavy"]:
            assert x["nSent"] == 80


def test_timing_arithmetic():
    d = load(); t = d["timing"]
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
```

- [ ] **Step 2: Run, expect SKIP** (no data.js yet)

- [ ] **Step 3: Implement `site/make_site.py`**

`make_demo.close()` uses a relative tolerance (1e-9 times the value, so about 1.3e-6 at 1,274 MW). The spec promises an absolute 1e-9, so `make_site.py` replaces it right after the import, before any check runs:

```python
md.close = lambda a, b: (a is b) if (a is None or b is None) else abs(float(a) - float(b)) <= 1e-9
```

Structure (reuse, do not copy, `make_demo`'s checks):

```python
"""Write site/data.js: every fixed-exam day plus the 4 days P2 ever missed.

Every number is refitted and checked against
experiments/results/round2_main_original.json before anything is written
(spec section 4). The page only displays this file.

    .venv/Scripts/python.exe site/make_site.py
"""
import json, sys
from pathlib import Path
import numpy as np
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))
import make_demo as md  # noqa: E402  (chdirs to ROOT, sets up imports, holds the checks)
from round2_robustness import N_EXAM, N_PRACTICE, load_pool, score_exam  # noqa: E402
from verify_resplit import clean_timings  # noqa: E402
from p013 import calibration as cal  # noqa: E402
from p013.case_loader import load_case  # noqa: E402
from p013.naming import branch_table  # noqa: E402
from p013.forward_numpy import load_weights  # noqa: E402

OUT = ROOT / "site" / "data.js"
LINKS = {
    "space": "https://huggingface.co/spaces/ABDHAM/p013-grid-screening",
    "github": "https://github.com/Hamza-Atiq/p013-grid-contingency-screening",
    "results": "results.html",
}


def layout():
    net = load_case()
    refs = branch_table(net)
    g = nx.Graph((r.from_bus, r.to_bus) for r in refs)
    pos = nx.kamada_kawai_layout(g)          # deterministic, no seed needed
    xs = np.array([p[0] for p in pos.values()]); ys = np.array([p[1] for p in pos.values()])
    scale = lambda v, lo, hi: (v - lo) / (hi - lo)
    gen = set(int(b) for b in net.gen.bus) | {int(net.ext_grid.bus.iloc[0])}
    buses = [{"id": int(b), "x": round(float(scale(pos[b][0], xs.min(), xs.max())), 4),
              "y": round(float(scale(pos[b][1], ys.min(), ys.max())), 4), "gen": int(b) in gen}
             for b in sorted(pos)]
    return buses, [{"name": r.name, "a": r.from_bus, "b": r.to_bus} for r in refs]


def label(d, near):
    if d["heavy"]:
        return "heavy day, all 80 checked"
    return "near miss" if near else ("busy day" if d["busy"] else "quiet day")


def main():
    results = json.loads(md.RESULTS.read_text(encoding="utf-8"))
    pool = load_pool(md.CACHES); names = pool["names"]; n = pool["pinned"].shape[0]
    timings = clean_timings()
    test1 = results["test1_robustness"]

    # 50 re-split exams: refit and reproduce every run, collect miss days.
    rng = np.random.default_rng(results["seed"]); miss = {}
    for s, run in enumerate(results["runs"]):
        perm = rng.permutation(n)
        practice, exam = np.sort(perm[:N_PRACTICE]), np.sort(perm[N_PRACTICE:N_PRACTICE + N_EXAM])
        rec = run["policies"][md.POLICY]
        pol = md.fit_p2(pool, practice, rec, f"exam {s + 1}")
        verify = pol.verify(pool, exam)
        md.check_exam(score_exam(verify, pool, exam, timings), rec["exam"], f"exam {s + 1}")
        for j, row in enumerate(exam):
            if (pool["dangerous"][row] & ~(verify[j] | pool["pinned"][row])).any() and row not in miss:
                miss[row] = md.day_payload(pool, pol, row, names, verify[j],
                                           f"missed one (exam {s + 1} of 50)", "missed", s + 1)
        print(f"  exam {s + 1:2d} reproduced", flush=True)
    total = sum(r["policies"][md.POLICY]["exam"]["fast_screen"]["points_with_miss"] for r in results["runs"])
    if total != 14 or len(miss) != 4:
        raise SystemExit(f"expected 14 miss-days over 4 distinct days, got {total} over {len(miss)}")

    # Fixed exam: one policy, all 175 days.
    fixed = results["test3_fixed_exam"]["policies"][md.POLICY]
    practice, exam = np.flatnonzero(~pool["is_test"]), np.flatnonzero(pool["is_test"])
    pol = md.fit_p2(pool, practice, fixed, "fixed exam")
    verify = pol.verify(pool, exam)
    md.check_exam(score_exam(verify, pool, exam, timings), fixed["exam"], "fixed exam")
    days = []
    sent_normal = [int((verify[j] | pool["pinned"][r]).sum()) for j, r in enumerate(exam)
                   if cal.total_load(pool)[r] < pol.L]
    busy_at = float(np.percentile(sent_normal, 75))
    for j, row in enumerate(exam):
        d = md.day_payload(pool, pol, row, names, verify[j], "", "fixed")
        d["busy"] = d["nSent"] >= busy_at
        dang = [r["score"] for r in d["rows"] if r["dangerous"] and not r["pinned"]]
        near = (not d["heavy"]) and bool(dang) and min(dang) - pol.cutoff_normal < 0.05
        d["group"] = "fixed"; d["label"] = f"Day {j + 1}: {d['loadMw']:,.0f} MW, {label(d, near)}"
        days.append(d)
    for d in miss.values():
        d["group"] = "missed"; d["label"] = f"{d['loadMw']:,.0f} MW, {d['label']}"
        by_name = {r["name"]: r for r in d["rows"]}
        d["missNote"] = [{"name": b, "score": by_name[b]["score"], "cutoff": d["cutoff"],
                          "examNo": d["examNo"]} for b in d["missed"]]
        days.append(d)
    t = timings["fast_screen"]
    for d in days:   # estimates from laptop medians (spec 3.1); the page only displays them
        d["allSeconds"] = round(t.baseline_s, 3)
        d["screenSeconds"] = round(t.overhead_s + d["nSent"] * t.solve_s, 3)
        d["speedup"] = round(t.baseline_s / (t.overhead_s + d["nSent"] * t.solve_s), 2)

    md.check_against_model(days[:5] + list(miss.values()), load_weights(md.WEIGHTS),
                           [np.load(p)["point"] for p in md.CACHES])

    summary = test1["summary"]
    buses, branches = layout()
    data = {
        "case": "IEEE 57-bus (PGLib-OPF)", "n": 80,
        "timing": {"solve": t.solve_s, "overhead": t.overhead_s, "baseline": t.baseline_s},
        "policy": {"L": round(pol.L, 1), "cutoff": round(pol.cutoff_normal, 4)},
        "buses": buses, "branches": branches,
        "days": [{k: d.get(k) for k in ("label", "group", "loadMw", "heavy", "nSent", "nDangerous",
                                        "missed", "missNote", "examNo", "cutoff", "allSeconds",
                                        "screenSeconds", "speedup", "rows")} for d in days],
        "honest": {
            "fixedMissDays": fixed["exam"]["fast_screen"]["points_with_miss"], "fixedExamDays": int(exam.size),
            "fixedSpeedup": fixed["exam"]["fast_screen"]["speedup"],
            "zeroMiss": summary[md.POLICY]["exams_with_zero_misses"], "nExams": len(results["runs"]),
            "passNeeded": test1["pass_mark"]["needed"], "missDays": total, "distinctMissDays": len(miss),
            "registeredZeroMiss": summary["primary"]["exams_with_zero_misses"],
            "speedupMedian": summary[md.POLICY]["fast_screen"]["speedup_median"],
        },
        "links": LINKS,
    }
    OUT.write_text("window.P013 = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"written {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run it, then the test, expect PASS**

Run: `.venv\Scripts\python.exe site\make_site.py` then `.venv\Scripts\python.exe -m pytest tests/test_site_data.py -v`

- [ ] **Step 5: Move the old report**

Change `OUT` in `demo/make_demo.py` to `ROOT / "site" / "results.html"`, run `.venv\Scripts\python.exe demo\make_demo.py`, delete the root `demo.html`, and update the README's "Open the demo" paragraph to point at `site/index.html` and `site/results.html`.

---

### Task 4: `site/index.html`, the page a professor sees

**Files:**
- Create: `site/index.html` (loads `data.js` with `<script src="data.js">`, no fetch, no external scripts or fonts)

**Interfaces:**
- Consumes: `window.P013` from Task 3.

- [ ] **Step 1: Build the page**, in this order top to bottom, max width 980 px:

1. `<h1>` "If one power line fails, does anything overload?" and one line under it: "An AI ranks all 80 possible line failures on the IEEE 57-bus grid in about 0.15 s. The exact solver then checks only the risky ones. The AI sorts, the solver decides."
2. Two-column block (grid 60/40 at >= 900 px, stacked below): left an inline `<svg viewBox="0 0 600 420">` with branches as `<line>` (from `buses` x,y scaled into a 20 px margin) and buses as `<circle>` (generators as squares); right the controls: `<select id="day">` with `<optgroup label="Test days (175, one fixed policy)">` and `<optgroup label="Days it missed (other tests)">`, a "Run screening" `<button>`, and four stat tiles (Total load, Sent to solver x of 80, Dangerous, Missed).
3. Result table: 80 rows ordered as in `rows`: name, model score (or "always checked"), solver says (DANGEROUS / safe), sent (yes / no), and a tick or cross column for dangerous rows. A dashed rule after the last sent row on normal days. Show 20 rows with a "Show all 80" button.
3a. **Miss callout**, beside the controls, shown on every "Days it missed" day and never hidden by the 20-row limit: "Missed: L48. The AI scored it 0.805, just below the cutoff of 0.823 that exam's policy had learned (exam 2 of 50)." Built from `missNote`. Missed rows are always among the visible table rows.
4. Time line: "All 80 solved: X s. Screen + sent cases: Y s. About Z times faster." taken directly from `allSeconds`, `screenSeconds` and `speedup` (the page does no arithmetic), with the sentence "Estimated from laptop medians (0.051 s per solve, 0.149 s per screen), not timed live for this day."
5. Honest box, always visible, text built from `honest`: "These are recorded results from the real model and solver. On this fixed test the AI missed nothing: 0 of 175 days. But over 50 reshuffled tests my pass mark was not reached: zero misses on 39 of 50 tests, and I needed 48. The 14 miss-days are only 4 different days, listed above under 'Days it missed'. I show policy P2 because it did best; the selection rule I registered in advance picked another policy and reached 31 of 50. Next I would study why the misses are all one kind (a voltage drop the model under-predicts) and fix it with supervision."
6. Links row: "Run it live (Hugging Face, may take a minute to wake up)", "Code and full write-up (GitHub)", "Full results".

Run behaviour (JS, about 120 lines):

```js
let runToken = 0;
function run() {
  const token = ++runToken;            // newest click wins (Review Focus 5)
  const d = P013.days[+daySel.value];
  resetGrid(); fillStats(d, {phase: "model"});
  colourByScore(d);                    // step 1: model ranking, instant
  setTimeout(() => {                   // step 2: solver answers, 600 ms later
    if (token !== runToken) return;
    colourBySolver(d); fillTable(d); fillStats(d, {phase: "solver"}); fillTime(d);
  }, 600);
}
```

Colours (defined as CSS variables, with a dark-mode set): sent and dangerous = red, sent and safe = amber, not sent = grey, missed = red dashed and thick. Branch tooltips via `<title>`.

- [ ] **Step 2: Check it in Chrome at 1024 x 768** (double-click the file, i.e. `file://`): every day group, the 4 miss days, "Show all 80", fast repeated clicking, the three links, no horizontal scroll, no console errors. Fix anything found and repeat.

---

### Task 5: the Hugging Face Space

**Files:**
- Create: `space/app.py`, `space/README.md` (Space card), `space/requirements.txt`, `space/build_space.py` (copies `src/p013`, `cases/`, `model/weights.npz`, `model/live_policy.npz` into `space/_build/` with the same relative layout so `p013.CASE_PATH` resolves)
- Test: `tests/test_space_app.py`

**Interfaces:**
- Consumes: `p013.live` (Task 1), `model/live_policy.npz` (Task 2)
- Produces: `space/app.py` functions `screen(load_mw: float, seed: int) -> tuple[str, list[list]]` and `check_all(load_mw: float, seed: int) -> tuple[str, list[list]]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_space_app.py
import importlib.util
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "space" / "app.py"


def load_app():
    spec = importlib.util.spec_from_file_location("space_app", APP)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_screen_runs_on_a_fixed_seed():
    summary, table = load_app().screen(1000.0, 1)
    assert "sent to the solver" in summary and len(table) == 80


def test_unsolvable_intact_day_gives_a_plain_message(monkeypatch):
    app = load_app()
    from p013.solving import SolveResult
    monkeypatch.setattr(app.live, "solve_intact", lambda day: SolveResult.failed("no_solution"))
    summary, table = app.screen(1000.0, 1)
    assert "try another seed" in summary and table == []
```

- [ ] **Step 2: Run, expect FAIL** (no `space/app.py`)

- [ ] **Step 3: Implement `space/app.py`**

```python
"""P-013 live screening. Builds a fresh day, the AI ranks all 80 line
failures, the real solver checks the ones the AI sends."""
import sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))        # build layout (space/_build)
sys.path.insert(0, str(HERE.parent / "src"))  # repo layout, for tests

import numpy as np
from p013 import live
from p013.forward_numpy import load_weights

MODEL = HERE / "model" if (HERE / "model").exists() else HERE.parent / "model"
WEIGHTS = load_weights(MODEL / "weights.npz")
POLICY = live.load_policy(MODEL / "live_policy.npz")
SOLVE_S = 0.051  # laptop median, only for the 'all 80' estimate line


def _rows(scr, truth):
    order = sorted(range(80), key=lambda c: (not POLICY.always[c], -np.nan_to_num(scr.score[c], nan=9)))
    rows = []
    for c in order:
        n = live.NAMES[c]; t = truth.get(n)
        rows.append([n, "always checked" if POLICY.always[c] else round(float(scr.score[c]), 3),
                     "yes" if scr.sent[c] else "no",
                     "" if t is None else ("DANGEROUS" if t.dangerous else "safe") +
                     ("" if t.status == "converged" else " (no solution)")])
    return rows


def _run(load_mw, seed, everything):
    day = live.build_day(int(seed), float(load_mw))
    intact = live.solve_intact(day)
    if not intact.ok:
        return "This day has no power flow solution even with every line in. Please try another seed.", []
    scr = live.score_day(day, WEIGHTS, POLICY)
    sent = [n for n, s in zip(live.NAMES, scr.sent) if s]
    todo = list(live.NAMES) if everything else sent
    t0 = time.perf_counter()
    truth = live.solve_cases(day, todo, vm_before=intact.vm_pu)
    solve_s = time.perf_counter() - t0
    lines = [f"Total load {day.total_mw:,.0f} MW ({'heavy day: all 80 sent' if scr.heavy else 'normal day'}).",
             f"AI ranked all 80 in {scr.seconds:.2f} s; {len(sent)} of 80 sent to the solver.",
             f"Solver time on this Space: {solve_s:.1f} s for {len(todo)} cases."]
    if everything:
        missed = [n for n in live.NAMES if truth[n].dangerous and n not in sent]
        lines.append("Missed by the AI: " + (", ".join(missed) if missed else "none") + ".")
    return "\n\n".join(lines), _rows(scr, truth)


def screen(load_mw, seed):
    return _run(load_mw, seed, everything=False)


def check_all(load_mw, seed):
    return _run(load_mw, seed, everything=True)


if __name__ == "__main__":
    import gradio as gr
    with gr.Blocks(title="P-013 grid screening") as demo:
        gr.Markdown("# If one power line fails, does anything overload?\n"
                    "The AI ranks all 80 possible line failures on the IEEE 57-bus grid, and the real "
                    "solver checks only the risky ones. *This free Space sleeps after 48 hours unused; "
                    "the first visit can take a minute or more.*")
        load = gr.Slider(735, 1550, value=1000, step=5, label="Total load (MW)")
        seed = gr.Number(value=1, precision=0, label="Day seed")
        with gr.Row():
            b1 = gr.Button("Run screening", variant="primary")
            b2 = gr.Button("Check all 80 with the solver")
        out = gr.Markdown()
        table = gr.Dataframe(headers=["Branch", "AI score", "Sent to solver", "Solver says"], wrap=True)
        b1.click(screen, [load, seed], [out, table], concurrency_id="grid")
        b2.click(check_all, [load, seed], [out, table], concurrency_id="grid")
    # One queue for both buttons, one job at a time. live.LOCK is the second guard.
    demo.queue(default_concurrency_limit=1).launch()
```

`space/requirements.txt`: `pandapower==3.5.4`, `numpy==2.4.6`, `numba==0.67.0`, `networkx==3.6.1`, `pandas==2.3.3`, `scipy==1.16.3`.
`space/README.md` front matter: `title: P-013 grid screening`, `sdk: gradio`, `sdk_version: 5.9.1`, `python_version: "3.11"`, `app_file: app.py`, `license: mit`, then 5 plain lines linking the GitHub repo and the static site.

- [ ] **Step 4: Run tests, expect PASS; then run `python space/app.py` locally and click both buttons at 1024 x 768**

- [ ] **Step 5 (needs the HF login already on this machine): publish the Space**

```
.venv\Scripts\python.exe space\build_space.py
hf repo create ABDHAM/p013-grid-screening --repo-type space --space_sdk gradio
hf upload ABDHAM/p013-grid-screening space\_build . --repo-type space
```

Then open the Space in Chrome, wait for the build, click both buttons, and compare one day's "Missed" line with a local run of the same load and seed.

---

### Task 6: GitHub repo, Pages, README, final checks (GATED on "commit")

**Files:**
- Create: `.github/workflows/pages.yml`
- Modify: `README.md` (two links at the top; one sentence on the always-send rule; "Open the demo" paragraph)

- [ ] **Step 1: Pages workflow**

```yaml
name: pages
on:
  push:
    branches: [main]
  workflow_dispatch:
permissions:
  contents: read
  pages: write
  id-token: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: site
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: README top**, above "The problem":

```
**Try it:** [live page](https://hamza-atiq.github.io/p013-grid-contingency-screening/) (instant, recorded results) · [run it live](https://huggingface.co/spaces/ABDHAM/p013-grid-screening) (real model and solver; may take a minute to wake up)
```

and under "What I built": "Cases the solver could not solve on any practice day (7 of 80) always go to the solver. Measured: this sends exactly the same cases as the recorded run on all 50 exams."

- [ ] **Step 3: Full test run**: `.venv\Scripts\python.exe -m pytest` (expect 304 + new tests, 0 failed, 0 skipped).

- [ ] **Step 4: Codex check** (read-only, effort low) of `src/p013/live.py`, `site/make_site.py`, `site/index.html`, `space/app.py` against the spec, 6-item checklist, max 250 words.

- [ ] **Step 5 (GATED, only after Hamza says "commit"):**

```
git init -b main
git add .
git commit -m "P-013: live page (GitHub Pages) and live Space"
gh repo create Hamza-Atiq/p013-grid-contingency-screening --public --source . --push
gh api -X POST repos/Hamza-Atiq/p013-grid-contingency-screening/pages -f build_type=workflow
```

Then wait for the workflow, open the Pages URL in Chrome at 1024 x 768, repeat Task 4 Step 2 on the live URL, and click through to the Space.

---

## Review history

- 13 Oct 2026, independent Codex review (gpt-6-astra, read-only, open prompt). Five findings, all fixed in this version:
  1. High: one shared mutable network could be corrupted by two Space visitors at once. Fixed with `live.LOCK` around every sample and solve, one Gradio queue for both buttons, and an overlapping-requests test.
  2. High: the parity test could hide missing predictions. Now compares raw voltages for all 81 rows before any mask, asserts the NaN masks exactly, then the calibrated scores, then the sent decisions.
  3. Medium: the miss days did not explain themselves. Added `missNote` (branch, score, cutoff, exam) and an always-visible miss callout.
  4. Medium: `make_demo.close()` is relative (about 1.3e-6 at 1,274 MW). `make_site.py` replaces it with an absolute 1e-9 check.
  5. Medium: timing was computed in JavaScript. Now computed per day in `make_site.py`, with a timing-arithmetic test; the page does no arithmetic.
