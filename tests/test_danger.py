"""What counts as dangerous, and how a case is scored. SPEC 3.1, 3.2.

Both were amended 2026-09-10 and the amendment is the whole point of this file.
The voltage check used to be a plain absolute band. On the shipped PGLib case
**5 of 57 buses already sit outside 0.95-1.05 pu with nothing wrong at all**,
the worst at 0.937, so an absolute band marks every contingency dangerous, k*
degenerates to the whole sweep, and 3.3's headline means nothing.

So a bus is violated only when the outage **creates a violation or deepens an
existing one**. Thermal stays absolute, because no branch is over its rating in
the intact case - measured, and pinned below rather than trusted.

One more thing this file defends: the same two checks score the solver's answer
and the model's. Scoring the model against its own predicted labels would
measure nothing (SPEC 3.1's "truth and score" table).
"""
import numpy as np
import pytest

from p013.case_loader import load_case
from p013.danger import DangerReport, excess, risk_score, violations
from p013.naming import branch_names, set_in_service
from p013.solving import solve

N_BUS = 57
N_BRANCH = 80


def flat(value: float, n: int = N_BUS) -> np.ndarray:
    return np.full(n, value, dtype=float)


def quiet_loading(percent: float = 50.0) -> np.ndarray:
    """Branch loadings well inside their ratings, so only voltage can fire."""
    return np.full(N_BRANCH, percent, dtype=float)


# --------------------------------------------------------------------------
# The amendment itself
# --------------------------------------------------------------------------

def test_a_bus_already_outside_the_band_is_not_a_violation_on_its_own():
    """The load-bearing test. Bus sits at 0.937 before AND after the outage.

    Under the old absolute band this was a violation in every contingency,
    which is exactly what flattened the metric. The outage did nothing to it,
    so it must not be flagged.
    """
    vm = flat(1.0)
    vm[7] = 0.937
    report = violations(vm, quiet_loading(), vm_before=vm.copy())
    assert report.voltage is False
    assert report.is_dangerous is False


def test_a_bus_pushed_a_full_band_width_further_out_is_a_violation():
    """0.937 -> 0.880 is 1.14 band-widths of deterioration. Over the line."""
    before = flat(1.0)
    before[7] = 0.937
    after = before.copy()
    after[7] = 0.880
    assert violations(after, quiet_loading(), vm_before=before).voltage is True


def test_a_bus_pushed_less_than_a_band_width_is_not_a_violation():
    """0.937 -> 0.910 is 0.54 band-widths. Real, ranked, but not DANGEROUS.

    This is the 2026-09-11 amendment. Under the previous form - any
    deterioration at all - 78 of 80 contingencies on the shipped case were
    dangerous and k* was the entire list, so the headline was unreachable
    before the model was even consulted.
    """
    before = flat(1.0)
    before[7] = 0.937
    after = before.copy()
    after[7] = 0.910
    assert violations(after, quiet_loading(), vm_before=before).voltage is False


def test_a_bus_pushed_far_out_of_a_healthy_band_is_a_violation():
    before = flat(1.0)
    after = before.copy()
    after[3] = 0.88
    assert violations(after, quiet_loading(), vm_before=before).voltage is True


def test_a_bus_that_recovers_toward_the_band_is_not_a_violation():
    """Still outside 0.95-1.05, but the outage improved it. Not a violation."""
    before = flat(1.0)
    before[7] = 0.930
    after = before.copy()
    after[7] = 0.945
    assert violations(after, quiet_loading(), vm_before=before).voltage is False


def test_a_bus_inside_the_band_is_never_a_violation_however_much_it_moved():
    """1.00 -> 1.04 is a big move and still perfectly healthy. SPEC 3.1 needs
    BOTH conditions: outside the band AND worse than before."""
    before = flat(1.0)
    after = before.copy()
    after[3] = 1.04
    assert violations(after, quiet_loading(), vm_before=before).voltage is False


def test_the_band_edge_is_not_a_violation():
    before = flat(1.0)
    after = before.copy()
    after[3] = 0.95
    assert violations(after, quiet_loading(), vm_before=before).voltage is False


def test_overvoltage_counts_the_same_as_undervoltage():
    before = flat(1.0)
    after = before.copy()
    after[3] = 1.12
    assert violations(after, quiet_loading(), vm_before=before).voltage is True


# --------------------------------------------------------------------------
# Thermal
# --------------------------------------------------------------------------

def test_thermal_is_absolute():
    """No deterioration term on the thermal side: over rating is over rating,
    whatever the pre-outage flow was."""
    loading = quiet_loading()
    loading[11] = 120.0
    report = violations(flat(1.0), loading, vm_before=flat(1.0))
    assert report.thermal is True
    assert report.is_dangerous is True


def test_exactly_at_rating_is_not_over_it():
    loading = quiet_loading()
    loading[11] = 100.0
    assert violations(flat(1.0), loading, vm_before=flat(1.0)).thermal is False


def test_no_branch_is_over_its_rating_in_the_intact_case():
    """This is WHY thermal may stay absolute (SPEC 3.1, measured 51.6%). If a
    future case starts with a branch over rating, absolute thermal would flag
    every contingency and this test is the early warning."""
    result = solve(load_case())
    assert result.ok
    assert result.loading_percent.max() < 100.0


# --------------------------------------------------------------------------
# The score
# --------------------------------------------------------------------------

def test_excess_is_zero_inside_the_band_and_positive_outside():
    assert excess(1.00) == 0.0
    assert excess(0.95) == pytest.approx(0.0)
    assert excess(1.05) == pytest.approx(0.0)
    assert excess(0.937) == pytest.approx(0.26, abs=1e-9)
    assert excess(0.90) == pytest.approx(1.0)


def test_the_score_is_the_worse_of_the_two_terms():
    """SPEC 3.2: risk = max(thermal, voltage deterioration). A branch at 130%
    of rating outranks a voltage term that is only just positive."""
    loading = quiet_loading()
    loading[11] = 130.0
    score = risk_score(flat(1.0), loading, vm_before=flat(1.0))
    assert score == pytest.approx(1.3)


def test_the_voltage_term_is_a_deterioration_not_an_absolute():
    """The bus sits at 0.937 in both states, so it contributes nothing. Under
    the old form it would have put a floor of 1.26 under every score, which is
    the saturation that destroys a ranking."""
    vm = flat(1.0)
    vm[7] = 0.937
    score = risk_score(vm, quiet_loading(50.0), vm_before=vm.copy())
    assert score == pytest.approx(0.5)


def test_a_deepened_violation_shows_up_in_the_score():
    before = flat(1.0)
    before[7] = 0.937
    after = before.copy()
    after[7] = 0.900
    score = risk_score(after, quiet_loading(10.0), vm_before=before)
    assert score == pytest.approx(1.0 - 0.26, abs=1e-9)


def test_score_does_not_saturate_across_real_contingencies():
    """The failure the console mockup already taught us: a constant floor makes
    every case score the same and the ranking carries no information.

    Real solves, not synthetic arrays - 25 real single-branch outages against
    the real intact state.
    """
    net = load_case()
    base = solve(net)
    assert base.ok

    scores = []
    for name in branch_names(net)[:25]:
        trial = load_case()
        set_in_service(trial, name, False)
        result = solve(trial)
        if not result.ok:
            continue
        scores.append(
            risk_score(result.vm_pu, result.loading_percent, vm_before=base.vm_pu)
        )

    assert len(scores) >= 20, "too few converged outages to judge spread"
    assert len(set(np.round(scores, 6))) > len(scores) // 2, "scores are clustered"
    assert max(scores) - min(scores) > 0.05, f"no spread in scores: {scores}"


# --------------------------------------------------------------------------
# One code path for truth and for prediction
# --------------------------------------------------------------------------

def test_the_same_function_scores_truth_and_prediction():
    """SPEC 3.1 splits WHERE the numbers come from, never WHICH checks run.
    The function must be pure: same arrays in, same answer out, with nothing
    remembered about who asked.
    """
    vm = flat(1.0)
    vm[7] = 0.92
    loading = quiet_loading()
    loading[3] = 88.0
    before = flat(1.0)

    from_solver = risk_score(vm, loading, vm_before=before)
    from_model = risk_score(vm.copy(), loading.copy(), vm_before=before.copy())
    assert from_solver == from_model

    again = risk_score(vm, loading, vm_before=before)
    assert again == from_solver, "risk_score is carrying state between calls"


def test_the_inputs_are_not_modified():
    """evaluate.py scores thousands of rows out of one array. A function that
    writes back into its arguments would corrupt the sweep silently."""
    vm = flat(1.0)
    vm[7] = 0.92
    loading = quiet_loading()
    before = flat(1.0)
    vm_copy, loading_copy, before_copy = vm.copy(), loading.copy(), before.copy()

    violations(vm, loading, vm_before=before)
    risk_score(vm, loading, vm_before=before)

    assert np.array_equal(vm, vm_copy)
    assert np.array_equal(loading, loading_copy)
    assert np.array_equal(before, before_copy)


def test_wrong_sized_arrays_are_an_error_not_a_silent_broadcast():
    """A 57-bus array against a 56-bus one must not quietly broadcast. This is
    how a whole evaluation comes out plausible and wrong."""
    with pytest.raises(ValueError):
        violations(flat(1.0), quiet_loading(), vm_before=flat(1.0, N_BUS - 1))


def test_report_fields_are_the_three_the_plan_names():
    report = violations(flat(1.0), quiet_loading(), vm_before=flat(1.0))
    assert isinstance(report, DangerReport)
    assert report.thermal is False
    assert report.voltage is False
    assert report.is_dangerous is False


# --------------------------------------------------------------------------
# The label IS the score crossing 1.0. Amended 2026-09-11.
# --------------------------------------------------------------------------

def test_the_label_is_exactly_the_score_crossing_one():
    """The load-bearing invariant of the amendment, over random states.

    If the danger threshold sits anywhere other than 1.0, the label and the
    ranking measure different things and a SAFE case can outrank a DANGEROUS
    one. Measured at a half band-width: 25 dangerous cases but k* = 42, a gap
    no model however perfect can close, because a safe case scoring 0.9 on
    thermal outranks a dangerous one scoring 0.6 on voltage.

    At 1.0 they cannot disagree, and this test is what keeps them equal.
    """
    rng = np.random.default_rng(20260911)
    # The peak loading is drawn ONCE per case and straddles 100%. Drawing all
    # 80 branches uniformly would put the maximum over 100 almost every time,
    # the thermal axis would decide every case, and the voltage threshold this
    # test exists to check would never be exercised. (It was written that way
    # first, and passed against the OLD code, which is how the hole showed.)
    both_axes_seen = {"thermal": 0, "voltage": 0, "safe": 0}
    for _ in range(2000):
        before = 1.0 + rng.normal(0, 0.02, N_BUS)
        after = before + rng.normal(0, rng.uniform(0.004, 0.045), N_BUS)
        loading = rng.uniform(0, 60, N_BRANCH)
        loading[rng.integers(N_BRANCH)] = rng.uniform(0, 130)

        report = violations(after, loading, vm_before=before)
        assert report.is_dangerous == (risk_score(after, loading, vm_before=before) > 1.0)

        key = "thermal" if report.thermal else "voltage" if report.voltage else "safe"
        both_axes_seen[key] += 1

    assert min(both_axes_seen.values()) > 20, f"one path never exercised: {both_axes_seen}"


def test_thermal_still_means_exactly_over_rating():
    """The amendment must not move the thermal line. A term over 1.0 IS over
    rating - that is where the 1.0 threshold comes from in the first place."""
    for percent, expected in ((99.9, False), (100.0, False), (100.1, True)):
        loading = quiet_loading()
        loading[11] = percent
        assert violations(flat(1.0), loading, vm_before=flat(1.0)).thermal is expected


def test_a_band_width_is_five_hundredths_of_a_per_unit():
    """What the threshold means in the unit a power engineer will ask in."""
    before = flat(1.0)
    before[7] = 0.95           # exactly at the limit, excess 0
    just_under = before.copy()
    just_under[7] = 0.95 - 0.0499
    just_over = before.copy()
    just_over[7] = 0.95 - 0.0501
    assert violations(just_under, quiet_loading(), vm_before=before).voltage is False
    assert violations(just_over, quiet_loading(), vm_before=before).voltage is True
