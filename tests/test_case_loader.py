"""The loader is also a test.

Every constant asserted here was measured against the real file on 2026-09-10
(see ../measurements/check_pglib.py). If the vendored case ever changes, the
build fails here rather than shipping a diagram that disagrees with the model.
"""
import pytest

import p013
from p013.case_loader import CaseInvariants, load_case, parse_matpower


def test_parses_the_three_matrices():
    ppc = parse_matpower(p013.CASE_PATH)
    assert ppc["baseMVA"] == 100.0
    assert ppc["bus"].shape[0] == 57
    assert ppc["gen"].shape[0] == 7
    assert ppc["branch"].shape[0] == 80


def test_every_branch_carries_a_rating():
    """SPEC 1.4. The whole project is denominated in percent of rating; the
    built-in case57 has rateA = 0 on all 80, which is why we use PGLib."""
    ppc = parse_matpower(p013.CASE_PATH)
    rate_a = ppc["branch"][:, 5]
    assert (rate_a > 0).all(), "a zero rating would make every thermal number meaningless"
    assert rate_a.min() == pytest.approx(22.0)
    assert rate_a.max() == pytest.approx(1617.0)


def test_network_shape_matches_the_measured_invariants():
    net = load_case()
    assert len(net.bus) == CaseInvariants.N_BUS == 57
    assert len(net.line) == CaseInvariants.N_LINE == 65
    assert len(net.trafo) == CaseInvariants.N_TRAFO == 15
    assert len(net.line) + len(net.trafo) == CaseInvariants.N_BRANCH == 80


def test_bus_indices_are_one_based_and_contiguous():
    """SPEC 5.5.1: the index IS the display number, so there is no mapping
    layer between the diagram, the ranked list, the API and the model."""
    net = load_case()
    assert list(net.bus.index) == list(range(1, 58))


def test_slack_bus_is_where_the_invariants_say():
    net = load_case()
    assert int(net.ext_grid.bus.iloc[0]) == CaseInvariants.SLACK_BUS == 1


def test_loader_rejects_a_case_that_does_not_match():
    with pytest.raises(ValueError, match="expected 80 branches"):
        load_case(_force_branch_count=79)


def test_each_load_returns_an_independent_network():
    """Callers mutate the net (taking branches out, splitting buses). If two
    calls shared state, every downstream sweep would be contaminated."""
    a, b = load_case(), load_case()
    a.line.at[a.line.index[0], "in_service"] = False
    assert bool(b.line.at[b.line.index[0], "in_service"]) is True


def test_template_reset_matches_a_fresh_load():
    """CaseTemplate exists purely for speed. If it ever diverged from
    load_case(), every generated sample would be from a different network than
    the one the tests verify."""
    from p013.case_loader import CaseTemplate
    from p013.naming import set_in_service
    from p013.solving import solve

    template = CaseTemplate()

    baseline = solve(load_case())
    first = solve(template.fresh())
    assert first.vm_pu.tolist() == baseline.vm_pu.tolist()

    # dirty it the way generation will, then check the reset really resets
    dirty = template.fresh()
    set_in_service(dirty, "L11", False)
    dirty.load["p_mw"] = dirty.load.p_mw.to_numpy() * 1.3
    solve(dirty)

    after = solve(template.fresh())
    assert after.vm_pu.tolist() == baseline.vm_pu.tolist()
    assert after.loading_percent.tolist() == baseline.loading_percent.tolist()


def test_template_is_much_faster_than_reloading():
    """The whole reason it exists. Measured 442 ms against microseconds."""
    import time
    from p013.case_loader import CaseTemplate

    template = CaseTemplate()
    template.fresh()

    t0 = time.perf_counter()
    for _ in range(20):
        template.fresh()
    reset_ms = (time.perf_counter() - t0) / 20 * 1000

    t0 = time.perf_counter()
    for _ in range(3):
        load_case()
    load_ms = (time.perf_counter() - t0) / 3 * 1000

    assert reset_ms < load_ms / 20, f"reset {reset_ms:.2f} ms vs load {load_ms:.1f} ms"
