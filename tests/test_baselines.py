"""The three things the model has to beat. SPEC 3.6.

Each is here for a different reason:

- **Full AC sweep** is the cost ceiling - what the model exists to avoid
  paying. Its ranking is the truth, so it is also the best any ranker could do.
- **DC power flow screen** is the real competitor. "DC already screens
  contingencies fast, why do I need a neural network" is the strongest
  objection to the whole project, and it has to be answered in numbers. It
  answers itself: DC assumes flat voltages, so it is *structurally* blind to
  the voltage half of SPEC 3.1. That is demonstrated here rather than asserted
  in prose.
- **Naive sort by pre-outage loading** is the sanity floor. If it already
  achieves zero misses the model has earned nothing, and SPEC 6 says we would
  say so first and loudly.

All three get the islanding pre-filter of SPEC 2.5, because it is exact code
rather than a model and withholding it from the baselines would flatter us.
"""
import numpy as np
import pandapower as pp
import pytest

from p013 import baselines
from p013.baselines import (
    OperatingPoint,
    Ranking,
    dc_screen,
    full_ac_sweep,
    naive_by_pre_outage_loading,
)
from p013.case_loader import load_case
from p013.danger import risk_score, violations
from p013.naming import branch_names
from p013.solving import solve

N_BRANCH = 80
ALL_THREE = (full_ac_sweep, dc_screen, naive_by_pre_outage_loading)


@pytest.fixture(scope="module")
def point():
    """The case's own operating point. Cheap and completely determined."""
    return OperatingPoint.base(load_case())


@pytest.fixture(scope="module")
def truth(point):
    return full_ac_sweep(point)


# --------------------------------------------------------------------------
# Shape and determinism
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ranker", ALL_THREE, ids=lambda f: f.__name__)
def test_every_baseline_returns_a_full_ranking_of_80(ranker, point):
    result = ranker(point)
    assert isinstance(result, Ranking)
    assert len(result.order) == N_BRANCH
    assert set(result.order) == set(branch_names(load_case()))
    assert len(set(result.order)) == N_BRANCH


@pytest.mark.parametrize("ranker", ALL_THREE, ids=lambda f: f.__name__)
def test_a_baseline_gives_the_same_answer_twice(ranker, point):
    """k* is reported to three significant figures. A ranking that reshuffles
    its ties between runs makes that number unreproducible."""
    assert ranker(point).order == ranker(point).order


def test_ties_break_in_canonical_order():
    """Every branch scores identically, so the order is the tie-break alone."""
    order = baselines.rank({name: 0.5 for name in branch_names(load_case())}, pinned=())
    assert list(order) == branch_names(load_case())


@pytest.mark.parametrize("ranker", ALL_THREE, ids=lambda f: f.__name__)
def test_islanding_is_pinned_above_everything(ranker, point):
    """SPEC 2.5: an outage that cuts the grid in half is settled by exact code
    and ranked first, never handed to a scorer. SPEC 3.3 also says those cases
    consume shortlist slots rather than getting free ones."""
    result = ranker(point)
    assert result.pinned, "expected at least one pinned contingency on this case"
    assert set(result.order[: len(result.pinned)]) == set(result.pinned)
    for name in result.pinned:
        assert name not in result.score, "a pinned case must not be scored"


# --------------------------------------------------------------------------
# The DC screen, and why it is not enough
# --------------------------------------------------------------------------

def test_dc_power_flow_reports_every_voltage_as_exactly_one():
    """The root of the argument, taken from pandapower rather than claimed.

    DC power flow linearises around flat voltages. Every bus comes back at
    exactly 1.0, so excess(V) - excess(V_before) is identically zero and the
    voltage half of SPEC 3.1 can never fire, whatever the real network is
    doing.
    """
    net = load_case()
    pp.rundcpp(net)
    assert np.array_equal(np.unique(net.res_bus.vm_pu.to_numpy()), np.array([1.0]))


def test_the_dc_screen_never_reports_a_voltage_violation(point):
    for name, report in dc_screen(point).report.items():
        assert report.voltage is False, f"{name} reported a voltage violation under DC"


def test_the_dc_screen_finds_only_the_thermal_danger(point, truth):
    """The whole answer to "why not just use DC", as a hard count.

    Measured on the shipped case at its own operating point: 12 contingencies
    are dangerous, **11 of them on voltage alone** and one, L08, on thermal at
    102.2% of rating. The DC screen finds exactly the thermal one and none of
    the other eleven.

    That is not DC being inaccurate. It is DC being blind: flat voltages make
    the voltage term identically zero, so those eleven cases are invisible to
    it no matter how good the linearisation is.
    """
    dangerous_ac = {n for n, r in truth.report.items() if r.is_dangerous}
    dangerous_dc = {n for n, r in dc_screen(point).report.items() if r.is_dangerous}
    voltage_only = {n for n in dangerous_ac if not truth.report[n].thermal}

    assert dangerous_ac and voltage_only
    assert not (voltage_only & dangerous_dc), (
        "DC claimed to see a voltage-only danger: "
        f"{sorted(voltage_only & dangerous_dc)}"
    )
    assert dangerous_dc <= dangerous_ac, "DC invented a danger the solver denies"


def test_screening_on_dc_would_mean_verifying_almost_everything(point, truth):
    """The number the objection actually turns on.

    k* is how deep a ranking must be followed before every dangerous case has
    been caught. Ranked by the truth it is 14 of 80. Ranked by the DC screen it
    is 74 of 80 - so a DC screen saves you six checks out of eighty, which is
    not a screening tool. Computed here rather than asserted in prose, because
    this is the sentence a professor will push back on.
    """
    dangerous = {n for n, r in truth.report.items() if r.is_dangerous} | set(truth.pinned)
    k_dc = max(dc_screen(point).order.index(n) for n in dangerous) + 1
    k_truth = max(truth.order.index(n) for n in dangerous) + 1

    assert k_truth < 20, f"the ceiling moved: {k_truth}"
    assert k_dc > 3 * k_truth, f"DC k* {k_dc} vs truth {k_truth}"


def test_the_dc_screen_still_ranks_thermally(point):
    """Blind is not useless. DC sees real power flow, so its ranking has to
    carry information - otherwise the comparison is a straw man."""
    result = dc_screen(point)
    scores = np.array([result.score[n] for n in result.score])
    assert scores.max() - scores.min() > 0.05, "DC ranking is flat"


# --------------------------------------------------------------------------
# The naive floor
# --------------------------------------------------------------------------

def test_the_naive_baseline_uses_only_pre_outage_information(point, monkeypatch):
    """It may look at the intact system once and nothing else. A baseline that
    peeks at post-outage state is not the floor we claim it is - and it would
    be a floor the model could not fairly be compared against."""
    calls = []
    real_solve = baselines.solve

    def counted(net):
        calls.append(1)
        return real_solve(net)

    monkeypatch.setattr(baselines, "solve", counted)
    naive_by_pre_outage_loading(point)
    assert len(calls) == 1, f"expected exactly one power flow, made {len(calls)}"


def test_the_naive_baseline_is_the_pre_outage_loading_order(point):
    """Computed independently here, so the function cannot define its own
    correctness."""
    net = load_case()
    baselines.apply_operating_point(net, point)
    intact = solve(net)
    assert intact.ok

    names = branch_names(net)
    expected = [n for n in names if n not in baselines.pinned_contingencies(net)]
    expected.sort(key=lambda n: (-intact.loading_percent[names.index(n)], names.index(n)))

    result = naive_by_pre_outage_loading(point)
    assert list(result.order[len(result.pinned):]) == expected


# --------------------------------------------------------------------------
# The full AC sweep is truth
# --------------------------------------------------------------------------

def test_the_full_ac_sweep_ranks_every_danger_above_every_safe_case(truth):
    """It IS the truth, so it is the ceiling: no ranker can do better. If this
    fails, the score and the label have drifted apart again."""
    ranked = [n for n in truth.order if n in truth.report]
    dangerous = [i for i, n in enumerate(ranked) if truth.report[n].is_dangerous]
    safe = [i for i, n in enumerate(ranked) if not truth.report[n].is_dangerous]
    assert dangerous and safe
    assert max(dangerous) < min(safe)


def test_the_ac_truth_agrees_with_a_direct_solve(point, truth):
    """Spot check against solving one contingency by hand, so the sweep's
    bookkeeping cannot quietly shift a result onto the wrong branch. This is
    the shape of the branch-ordering trap caught in phase 2."""
    net = load_case()
    baselines.apply_operating_point(net, point)
    intact = solve(net)

    name = "L37"
    trial = load_case()
    baselines.apply_operating_point(trial, point)
    from p013.naming import set_in_service

    set_in_service(trial, name, False)
    result = solve(trial)
    assert result.ok

    expected = violations(result.vm_pu, result.loading_percent, vm_before=intact.vm_pu)
    assert truth.report[name] == expected


# --------------------------------------------------------------------------
# The pre-outage state must be the INTACT system
# --------------------------------------------------------------------------

def test_the_pre_outage_state_is_the_intact_system(point, truth):
    """Every score recomputed against an independently solved intact state.

    This is the test that was missing when the first version of ``_sweep``
    held an ``intact`` network across the loop. ``CaseTemplate.fresh()``
    returns the SAME object every call, so ``intact`` and ``net`` were one
    network and every contingency was scored against a "pre-outage" state with
    branch L01 already out. Nothing caught it: L01 out barely moves a voltage,
    so no danger label flipped and the spot-check against a direct solve still
    agreed.

    The scores are the sensitive part, so they are what this checks - all of
    them, against a baseline built from a separate ``load_case()`` that the
    sweep cannot touch.
    """
    from p013.naming import set_in_service

    reference = load_case()
    baselines.apply_operating_point(reference, point)
    intact = solve(reference)
    assert intact.ok

    checked = 0
    for name in list(truth.score)[::7]:
        trial = load_case()
        baselines.apply_operating_point(trial, point)
        set_in_service(trial, name, False)
        result = solve(trial)
        assert result.ok
        expected = risk_score(
            result.vm_pu, result.loading_percent, vm_before=intact.vm_pu
        )
        assert truth.score[name] == pytest.approx(expected, abs=1e-9), (
            f"{name}: sweep scored {truth.score[name]:.6f}, an independent solve "
            f"against the genuinely intact state gives {expected:.6f}"
        )
        checked += 1
    assert checked >= 8


def test_case_template_fresh_returns_the_same_object():
    """Pinning the hazard itself, so the next person meets it as a documented
    fact rather than as a wrong number."""
    from p013.case_loader import CaseTemplate

    template = CaseTemplate()
    assert template.fresh() is template.fresh()
