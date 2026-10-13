"""The most important function in phase 1. SPEC 2.4, 2.6.

Warm-starting Newton-Raphson from the base-case solution makes it diverge on
outages that a flat start solves without trouble. It is the fastest setting per
solve, it raises no error, and using it would label a large block of healthy
contingencies "no solution" throughout the training set - teaching the model
that a whole region of the system is unsolvable.

Measured 2026-09-09 on pandapower's built-in case57, sweeping all 63 lines:

    warm start from base-case results   36 converged, 27 failed, 22.2 s
    flat start                          59 converged,  4 failed,  9.2 s

Deliberately NOT asserted as an absolute count here: those numbers were never
measured on PGLib, and asserting an unverified constant is the exact mistake
this project keeps having to undo. The invariant is asserted instead.
"""
import pandapower as pp
import pytest

from p013.case_loader import load_case
from p013.naming import branch_table, set_in_service
from p013.solving import SolveResult, solve
from p013.topology import islanding_contingencies


def _out(net, name):
    set_in_service(net, name, False)
    return net


def test_base_case_converges_and_matches_the_measured_state():
    r = solve(load_case())
    assert r.status == "converged"
    assert r.vm_pu.min() == pytest.approx(0.9372, abs=1e-3)
    assert r.loading_percent.max() == pytest.approx(51.6, abs=0.5)


def test_base_case_has_five_buses_outside_the_band():
    """SPEC 3.1's amendment rests on this. If it ever became zero, the absolute
    voltage band would be usable again and 3.1 could be simplified."""
    r = solve(load_case())
    outside = int(((r.vm_pu < 0.95) | (r.vm_pu > 1.05)).sum())
    assert outside == 5, f"{outside} buses outside 0.95-1.05 in the intact case"


def test_islanding_is_reported_as_islanded_not_as_failure():
    """SPEC 2.4: islanding and non-convergence are different things and used to
    be run together. An islanded topology is a topology fact settled before the
    solver runs, never a solver failure."""
    net = load_case()
    name = next(iter(islanding_contingencies(net)))
    assert solve(_out(net, name)).status == "islanded"


def test_flat_start_is_never_worse_than_warm_start():
    """The regression guard for SPEC 2.6's flat-start decision.

    MEASURED ON PGLIB 2026-09-11, and it does not reproduce the spec's numbers:

        flat start   79 of 80 converged, fails only on L42
        warm start   79 of 80 converged, fails only on L42

    They are identical here. The 27-of-63 warm-start catastrophe in SPEC 2.6
    was measured on pandapower's built-in case57, whose base case sits at
    Vmin 0.72 pu with 40 of 57 buses out of band - a degenerate state, and
    therefore a terrible initial guess. PGLib starts from a healthy 0.937 pu,
    so warm-starting from it is not harmful on this case.

    Flat start still stands as the decision: it is never worse, it does not
    depend on the base case being healthy, and it costs nothing here. But the
    spec's *justification* does not hold for the case we ship, which matters
    because that figure was about to be quoted in public artifacts.

    So this asserts the invariant, which is true on any case, rather than a
    count that turned out to be case-specific.
    """

    def sweep(init: str) -> int:
        ok = 0
        for ref in branch_table(load_case()):
            net = _out(load_case(), ref.name)
            try:
                pp.runpp(net, init=init, numba=True)
                ok += 1
            except Exception:
                pass
        return ok

    flat = sweep("flat")
    warm = sweep("results")
    assert flat >= warm, f"flat {flat}/80, warm {warm}/80 - flat must never be worse"
    assert flat / 80 > 0.8, f"only {flat}/80 converged from a flat start; investigate"


def test_the_wrapper_accounts_for_every_branch():
    """solve() splits the sweep three ways and the three must add up to 80.

    Note the raw sweep above counts the islanding outage as "converged" -
    pandapower will happily solve a network in pieces by NaN-ing the isolated
    part. solve() classifies it as islanded BEFORE solving, which is the whole
    point of SPEC 2.5, so its converged count is legitimately one lower.
    """
    counts = {"converged": 0, "islanded": 0, "no_solution": 0}
    for ref in branch_table(load_case()):
        counts[solve(_out(load_case(), ref.name)).status] += 1

    assert sum(counts.values()) == 80
    assert counts["islanded"] == 1, "exactly one single-branch outage islands this case"
    assert counts["no_solution"] == 1, "L42 is the one genuine non-convergence"
    assert counts["converged"] == 78


def test_result_shape_is_what_the_dataset_expects():
    r = solve(load_case())
    assert isinstance(r, SolveResult)
    assert len(r.vm_pu) == 57
    assert len(r.va_degree) == 57
    assert len(r.loading_percent) == 80


def test_failure_carries_no_numbers():
    """A non-converged solve must not hand back stale or zero arrays that
    downstream code could mistake for an answer."""
    r = SolveResult.failed("no_solution")
    assert r.status == "no_solution"
    assert r.vm_pu.size == 0


def test_solving_does_not_leak_state_between_calls():
    a = solve(load_case())
    solve(_out(load_case(), "L11"))
    b = solve(load_case())
    assert a.vm_pu.tolist() == b.vm_pu.tolist()
