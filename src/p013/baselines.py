"""The three references the model is measured against. SPEC 3.6.

| Reference | What it establishes |
| --- | --- |
| Full AC sweep | the cost ceiling |
| DC power flow screen | the real competitor |
| Naive: rank by pre-outage loading | the sanity floor |

**The DC screen is the one that matters**, because "DC power flow already
screens contingencies fast, so why do I need a neural network" is the strongest
objection to this whole project. It answers itself once SPEC 3.1 is settled:
DC linearises around flat voltages, every bus comes back at exactly 1.0 pu, and
so ``excess(V) - excess(V_before)`` is identically zero. The screen is
**structurally** blind to the voltage half of the danger test - not inaccurate
about it, incapable of seeing it.

Measured on the shipped case at its own operating point: **12 contingencies are
dangerous, 11 of them on voltage alone.** The one exception, L08, is over its
rating at 102.2%, and the DC screen finds exactly that one and none of the
other eleven. Following the DC ranking you must verify **74 of 80** before you
have caught every danger; following the truth, **14**. A screen that saves six
checks out of eighty is not a screening tool, and that sentence is the answer
to the objection.

**The naive floor is not a straw man.** If sorting by pre-outage loading
already achieves zero misses then the model has earned nothing, and SPEC 6
commits us to saying that first and loudly. It gets exactly one power flow -
the intact system - and a test enforces that, because a "pre-outage" baseline
that peeks at post-outage state is not a floor at all.

**Every ranker gets the islanding pre-filter** (SPEC 2.5). It is exact graph
code rather than a model, so withholding it from the baselines would flatter
us. Pinned cases are ranked above everything without a score and, per SPEC 3.3,
they consume shortlist slots rather than getting free ones.

Scoring is never done here. All three call :mod:`p013.danger`, so a change to
what "dangerous" means cannot reach one ranker and miss another.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import numpy as np
import pandapower as pp

from p013.case_loader import CaseTemplate
from p013.danger import DangerReport, risk_score, violations
from p013.naming import branch_index, branch_names
from p013.solving import solve
from p013.topology import islanding_contingencies


@dataclass(frozen=True)
class OperatingPoint:
    """Load and dispatch, in pandapower's own row order.

    Generation is carried explicitly for the reason recorded in
    :mod:`p013.generate`: the sampler scales it to follow load, and a state
    that stores only the load cannot be rebuilt.
    """

    load_p_mw: np.ndarray
    load_q_mvar: np.ndarray
    gen_p_mw: np.ndarray

    @classmethod
    def base(cls, net) -> "OperatingPoint":
        """The case's own operating point, untouched."""
        return cls(
            load_p_mw=net.load.p_mw.to_numpy(dtype=float).copy(),
            load_q_mvar=net.load.q_mvar.to_numpy(dtype=float).copy(),
            gen_p_mw=net.gen.p_mw.to_numpy(dtype=float).copy(),
        )


class Ranking(NamedTuple):
    """One ranker's answer for one operating point.

    Attributes:
        order: all 80 branch names, most dangerous first. Pinned cases lead.
        score: name -> risk score. Pinned cases are absent, because they are
            never scored.
        report: name -> :class:`DangerReport`. Absent for pinned cases.
    """

    order: tuple[str, ...]
    score: dict[str, float]
    report: dict[str, DangerReport]
    pinned: tuple[str, ...]


def apply_operating_point(net, point: OperatingPoint) -> None:
    """Put an operating point onto a fresh network. Mutates ``net``."""
    net.load["p_mw"] = point.load_p_mw.copy()
    net.load["q_mvar"] = point.load_q_mvar.copy()
    net.gen["p_mw"] = point.gen_p_mw.copy()


#: Islanding depends on the TOPOLOGY and nothing else, so the answer is the
#: same at every operating point. Recomputing it per point cost 2.06 s a time -
#: eight and a half minutes across a 250-point sweep - to re-derive a constant.
#: Keyed on the in-service pattern rather than cached outright, because a split
#: busbar or an already-open branch genuinely changes the answer (SPEC 2.5: the
#: test is always relative to the CURRENT topology, never a fixed number).
_ISLANDING: dict[tuple, frozenset] = {}


def _topology_key(net) -> tuple:
    return (
        tuple(net.line.in_service.to_numpy().tolist()),
        tuple(net.trafo.in_service.to_numpy().tolist()),
        len(net.bus),
        tuple(net.switch.closed.to_numpy().tolist()) if len(net.switch) else (),
    )


def pinned_contingencies(net) -> set[str]:
    """Outages settled by exact code rather than by any ranking. SPEC 2.5."""
    key = _topology_key(net)
    if key not in _ISLANDING:
        _ISLANDING[key] = frozenset(islanding_contingencies(net))
    return set(_ISLANDING[key])


def rank(score: dict[str, float], pinned) -> tuple[str, ...]:
    """Pinned cases first, then by descending score.

    **Ties break in canonical branch order**, never by whatever order a dict
    happened to be built in. ``k*`` is a reported number; a ranking that
    reshuffles its ties between runs makes it unreproducible.
    """
    position, _ = _lookups()
    pinned = sorted(pinned, key=position.__getitem__)
    rest = sorted(score, key=lambda n: (-score[n], position[n]))
    return tuple(pinned) + tuple(rest)


_TEMPLATE: CaseTemplate | None = None
_POSITION: dict[str, int] | None = None
_INDEX: dict[str, tuple[str, int]] | None = None


def _reference():
    """One shared network for name lookups. Building one costs 1.2 s."""
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = CaseTemplate()
    return _TEMPLATE.fresh()


def _lookups() -> tuple[dict[str, int], dict[str, tuple[str, int]]]:
    """Name -> rank position, and name -> pandapower frame and row.

    Built once. ``naming.set_in_service`` resolves the name by walking both
    pandas frames on every call, which measured **11.8 ms** - against 89 ms for
    the power flow it accompanies, so it was an eighth of the run spent
    re-deriving a constant. ``generate.py`` already hoists exactly this for
    exactly this reason; this is the same trap one module along.
    """
    global _POSITION, _INDEX
    if _POSITION is None:
        net = _reference()
        names = branch_names(net)
        _POSITION = {name: i for i, name in enumerate(names)}
        _INDEX = {name: branch_index(net, name) for name in names}
    return _POSITION, _INDEX


def take_out_of_service(net, name: str) -> None:
    """``set_in_service(net, name, False)`` with the lookup hoisted."""
    _, index = _lookups()
    kind, idx = index[name]
    frame = net.line if kind == "line" else net.trafo
    frame.at[idx, "in_service"] = False


def _sweep(point: OperatingPoint, prepare: Callable, evaluate: Callable) -> Ranking:
    """Walk every contingency at one operating point, however it is solved.

    The baselines differ only in ``prepare`` and ``evaluate``; sharing the walk
    keeps the pinning, the ordering and the tie-break identical across all of
    them, which is what makes the rows of SPEC 3.6 comparable.

    **The pre-outage state is taken as an ARRAY before the loop starts, never
    as a network held across it.** ``CaseTemplate.fresh()`` returns the *same
    object* every call - its own docstring says callers must not hold a
    previous return value - so keeping an ``intact`` net and re-calling
    ``fresh()`` aliases the two. The first version of this function did exactly
    that, and every contingency was scored against a "pre-outage" state that
    already had branch L01 out of service. It passed the tests, because L01 out
    barely moves any voltage and no danger label flipped. That is the same
    shape as the warm-start contamination SPEC 2.6 warns about, and it is why
    the deterioration baseline is now a plain NumPy array that nothing can
    write into.
    """
    global _TEMPLATE
    if _TEMPLATE is None:
        # Constructing a CaseTemplate costs 1.2 s, which over a 250-point sweep
        # is five minutes spent rebuilding an identical network.
        _TEMPLATE = CaseTemplate()
    template = _TEMPLATE

    intact = template.fresh()
    apply_operating_point(intact, point)
    pinned = pinned_contingencies(intact)
    names = branch_names(intact)
    before = prepare(intact)

    scores: dict[str, float] = {}
    reports: dict[str, DangerReport] = {}

    for name in names:
        if name in pinned:
            continue
        net = template.fresh()
        apply_operating_point(net, point)
        take_out_of_service(net, name)

        outcome = evaluate(net, before)
        if outcome is None:
            # No answer from the exact solver is not a safe answer. SPEC 2.4
            # pins these beside the islanding cases and says which applies.
            pinned = pinned | {name}
            continue
        scores[name], reports[name] = outcome

    return Ranking(
        order=rank(scores, pinned),
        score=scores,
        report=reports,
        pinned=tuple(rank({}, pinned)),
    )


def full_ac_sweep(point: OperatingPoint) -> Ranking:
    """Solve every contingency exactly. The cost ceiling, and the truth.

    This is what the model exists to avoid paying for, so its ranking is also
    the ceiling no ranker can beat: it scores each case on the same numbers the
    danger label is read from.
    """
    def prepare(intact) -> np.ndarray:
        base = solve(intact)
        if not base.ok:
            raise ValueError(
                f"the intact system does not solve ({base.status}); there is no "
                "pre-outage state to measure deterioration against"
            )
        return base.vm_pu.copy()

    def evaluate(net, before):
        result = solve(net)
        if not result.ok:
            return None
        return (
            risk_score(result.vm_pu, result.loading_percent, vm_before=before),
            violations(result.vm_pu, result.loading_percent, vm_before=before),
        )

    return _sweep(point, prepare, evaluate)


def dc_screen(point: OperatingPoint) -> Ranking:
    """DC power flow on every contingency. The real competitor.

    DC returns every bus at exactly 1.0 pu, so it is handed flat voltages for
    both the post-outage and the pre-outage state. The voltage term of SPEC 3.2
    is then identically zero by construction, which is the point: this is the
    same scoring code as the AC sweep, given everything DC can actually
    provide. The blindness is DC's, not the comparison's.
    """
    def prepare(intact) -> np.ndarray:
        # DC's own answer for the intact system: flat, like every DC answer.
        return np.ones(len(intact.bus))

    def evaluate(net, flat):
        pp.rundcpp(net)
        loading = np.concatenate(
            [
                net.res_line.loading_percent.to_numpy(dtype=float),
                net.res_trafo.loading_percent.to_numpy(dtype=float),
            ]
        )
        return (
            risk_score(flat, loading, vm_before=flat),
            violations(flat, loading, vm_before=flat),
        )

    return _sweep(point, prepare, evaluate)


def naive_by_pre_outage_loading(point: OperatingPoint) -> Ranking:
    """Rank by how loaded each branch was BEFORE anything tripped.

    The sanity floor. One power flow of the intact system, then sort - it never
    looks at a post-outage state, which is what makes it a floor rather than a
    cheap approximation of the answer.

    It carries no danger reports, because it never computes a post-outage
    state and therefore has no opinion about what is dangerous. It ranks, and
    that is all ``k*`` needs from it.
    """
    net = _reference()
    apply_operating_point(net, point)
    intact = solve(net)
    if not intact.ok:
        raise ValueError(f"the intact system does not solve ({intact.status})")

    names = branch_names(net)
    pinned = pinned_contingencies(net)
    scores = {
        name: float(intact.loading_percent[i]) / 100.0
        for i, name in enumerate(names)
        if name not in pinned
    }
    return Ranking(
        order=rank(scores, pinned),
        score=scores,
        report={},
        pinned=tuple(rank({}, pinned)),
    )
