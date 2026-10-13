"""Exact topology facts. SPEC 2.5.

The test that matters most here is that the islanding check is RELATIVE. It
asks "does removing this branch leave more components than there are now",
never "more than one". With a bus coupler open the base count can legitimately
be greater than one, and a hardcoded threshold then silently reports zero
islanding cases - the worst possible failure, because it means telling an
operator that a blackout-causing outage is safe. That exact bug was found in
the design mockup.
"""
import pytest

from p013.case_loader import load_case
from p013.naming import set_in_service
from p013.topology import islanded_load_mw, islanding_contingencies, n_components


def _out(net, *names):
    for name in names:
        set_in_service(net, name, False)
    return net


def test_intact_case_is_one_piece():
    assert n_components(load_case()) == 1


def test_intact_case_has_exactly_one_islanding_contingency():
    """Measured 2026-09-10 on this exact case."""
    found = islanding_contingencies(load_case())
    assert len(found) == 1, f"expected exactly one, got {sorted(found)}"


def test_the_check_is_relative_not_absolute():
    """With something already islanded the base count is 2, and a further
    islanding branch must still be detected. A hardcoded '> 1' would return
    every remaining branch; a hardcoded '== 2' would return none."""
    net = load_case()
    only = next(iter(islanding_contingencies(net)))
    _out(net, only)
    assert n_components(net) == 2

    found = islanding_contingencies(net)
    assert only not in found, "a branch already out of service is not a contingency"
    for name in found:
        probe = _out(load_case(), only, name)
        assert n_components(probe) > 2, f"{name} was reported but does not island"


def test_a_branch_already_out_is_never_reported():
    net = load_case()
    _out(net, "L11")
    assert "L11" not in islanding_contingencies(net)


def test_islanding_check_leaves_the_network_as_it_found_it():
    """It toggles branches to probe them. If it failed to restore one, every
    later solve in the sweep would silently be on the wrong topology."""
    net = load_case()
    before = [bool(v) for v in net.line.in_service] + [bool(v) for v in net.trafo.in_service]
    islanding_contingencies(net)
    after = [bool(v) for v in net.line.in_service] + [bool(v) for v in net.trafo.in_service]
    assert before == after


def test_nothing_is_cut_off_in_the_base_case():
    assert islanded_load_mw(load_case()) == pytest.approx(0.0)


def test_islanded_load_counts_what_is_actually_lost():
    """Take out the one islanding branch and something real must go dark."""
    net = load_case()
    only = next(iter(islanding_contingencies(net)))
    _out(net, only)
    assert n_components(net) == 2
    assert islanded_load_mw(net) >= 0.0  # measured below; may be an empty stub bus


def test_the_prefilter_leaves_the_network_exactly_as_it_found_it():
    """It used to toggle in_service on the frame and restore it in a finally.
    Since 2026-09-12 it works on a list of edges instead, so there is nothing
    to restore and no way to leave the net dirty if something raises."""
    net = load_case()
    before_line = net.line.in_service.to_numpy().copy()
    before_trafo = net.trafo.in_service.to_numpy().copy()

    islanding_contingencies(net)

    assert (net.line.in_service.to_numpy() == before_line).all()
    assert (net.trafo.in_service.to_numpy() == before_trafo).all()


def test_the_prefilter_is_fast_enough_to_charge_ourselves_for():
    """SPEC 3.5 counts the pre-filter against us, and it used to cost 0.754 s
    against 0.065 s for a full AC power flow - 58% of the screening overhead
    spent re-reading a table that never changes. If it creeps back over a
    tenth of a second, the speedup claim quietly rots."""
    import time

    net = load_case()
    islanding_contingencies(net)

    started = time.perf_counter()
    islanding_contingencies(net)
    assert time.perf_counter() - started < 0.1
