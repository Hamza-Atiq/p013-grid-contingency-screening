"""Screen one fresh operating point, live. Used by the Hugging Face Space.

Mirrors the recorded pipeline exactly: ``fast_screen.ScreenCase`` (bit-identical
to ``evaluate.model_ranking``) for the model's voltages,
``calibration.calibrated_scores`` for the bias-corrected score, P2 for the
decision, ``evaluate.truth_ranking`` for the solver's answer.
``tests/test_live_parity.py`` proves the scores match the caches.

NumPy only. Importing torch here would make the Space several hundred MB
heavier for nothing.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import p013
from p013.admittance import loading_percent
from p013.case_loader import CaseTemplate, parse_matpower
from p013.danger import RATING_PERCENT, excess, violations
from p013.fast_screen import ScreenCase
from p013.naming import branch_names, set_in_service
from p013.sampling import sample_operating_point
from p013.solving import SolveResult, solve

#: Total load seen in the 750 generated days was 732.3 to 1,553.2 MW.
LOAD_RANGE = (735.0, 1550.0)

#: CaseTemplate.fresh() hands back ONE shared mutable network. Every use of
#: it (sampling, solving) runs under this lock, so two Space visitors can
#: never reset each other's loads or outages mid-solve.
LOCK = threading.RLock()

_TEMPLATE = CaseTemplate()
_PPC = parse_matpower(p013.CASE_PATH)
NAMES = tuple(branch_names(_TEMPLATE.fresh()))
# The same bus lists evaluate.Sweep uses to place loads and generation.
_LOAD_BUS = [int(b[0]) for b in _PPC["bus"] if b[2] != 0 or b[3] != 0]
_GEN_BUS = [int(b) for b in _TEMPLATE.fresh().gen.bus]


@dataclass
class Day:
    """One operating point: loads in ``net.load`` order, generation in ``net.gen`` order."""

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
    score: np.ndarray     # [80], NaN on the always-send cases (the model is not asked)
    sent: np.ndarray      # [80] bool
    heavy: bool
    seconds: float


_IN_SERVICE = np.ones((len(NAMES) + 1, len(NAMES)), dtype=bool)
for _c in range(len(NAMES)):
    _IN_SERVICE[_c + 1, _c] = False   # row 0 intact, row c+1 has branch c out
_CASES: dict[int, ScreenCase] = {}


def _screen_case(weights: dict) -> ScreenCase:
    """The optimized screen (``p013.fast_screen``), built once per weights.

    Bit-identical to the plain forward pass (its module docstring and the
    parity tests say why), and it is the path the recorded 0.149 s timing
    measured, so the Space times the same thing the README quotes.
    """
    key = id(weights)
    if key not in _CASES:
        _CASES[key] = ScreenCase(_PPC, NAMES, _LOAD_BUS, _GEN_BUS, weights)
    return _CASES[key]


def model_voltages(day: Day, weights: dict):
    """``(vm, va, thermal)``: [81,57], [81,57], [80]. Row 0 is the intact grid."""
    case = _screen_case(weights)
    bus_one = case.bus_features_at(day.load_p, day.load_q, day.gen_p)
    bus_x = np.repeat(bus_one[None].astype(np.float32), len(_IN_SERVICE), axis=0)
    vm, va = case.forward(bus_x, case.branch_x(_IN_SERVICE), bus_one[:, 4].astype(np.float32))
    loading = case.loading_percent(vm[1:], va[1:], _IN_SERVICE[1:])
    thermal = np.asarray(loading).max(axis=1) / RATING_PERCENT
    return vm, va, thermal


def score_day(day: Day, weights: dict, policy: Policy) -> Screen:
    """The model's danger score for all 80 cases, and P2's decision."""
    t0 = time.perf_counter()
    vm, _, thermal = model_voltages(day, weights)
    v = vm - policy.bias                                   # calibration "bias"
    voltage = (excess(v[1:]) - excess(v[:1])).max(axis=1)
    score = np.where(policy.always, np.nan, np.maximum(thermal, voltage))
    heavy = day.total_mw >= policy.L
    by_score = np.ones(len(NAMES), bool) if heavy else np.nan_to_num(score, nan=-np.inf) >= policy.cutoff
    return Screen(score, policy.always | by_score, bool(heavy), time.perf_counter() - t0)


@dataclass
class CaseResult:
    status: str
    dangerous: bool
    seconds: float


def _net(day: Day, outage: str | None):
    """The shared network set to this day. Call only while holding LOCK."""
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
    """The solver's answer, judged exactly as ``evaluate.truth_ranking``.

    A case the solver cannot solve (islanded or no solution) counts as
    dangerous, the same rule as the recorded results.
    """
    out = {}
    for name in names:
        t0 = time.perf_counter()
        with LOCK:
            r = solve(_net(day, name))
        if not r.ok:
            out[name] = CaseResult(r.status, True, time.perf_counter() - t0)
            continue
        in_service = np.ones(len(NAMES), bool)
        in_service[NAMES.index(name)] = False
        v = r.vm_pu * np.exp(1j * np.deg2rad(r.va_degree))
        loading = loading_percent(_PPC, v, in_service)
        out[name] = CaseResult(
            "converged",
            violations(r.vm_pu, loading, vm_before).is_dangerous,
            time.perf_counter() - t0,
        )
    return out
