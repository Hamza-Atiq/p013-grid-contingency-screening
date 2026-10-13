"""The foundation of the physics term. SPEC 2.3.

The load-bearing test is the first one. If the mismatch at the solver's own
answer is not essentially zero, then the physics term is measuring our bug
rather than the model's error - and the conclusion "the physics term did not
help" would be meaningless. SPEC 6 lists that ablation as a real possible
outcome, so it has to be a real measurement.
"""
import numpy as np
import pandapower as pp
import pytest

import p013
from p013.admittance import (
    BR_RATE_A,
    BUS_TYPE_PQ,
    BUS_TYPE_PV,
    branch_flows,
    build_ybus,
    canonical_branch_rows,
    power_injection,
    specified_injection,
    to_canonical_order,
    to_matpower_order,
)
from p013.case_loader import load_case, parse_matpower
from p013.naming import branch_names, set_in_service


@pytest.fixture(scope="module")
def ppc():
    return parse_matpower(p013.CASE_PATH)


@pytest.fixture(scope="module")
def solved():
    net = load_case()
    pp.runpp(net, init="flat", numba=True)
    vm = net.res_bus.vm_pu.to_numpy()
    va = np.deg2rad(net.res_bus.va_degree.to_numpy())
    return net, vm * np.exp(1j * va)


def test_mismatch_at_the_solver_solution_is_zero(ppc, solved):
    _, v = solved
    actual = power_injection(build_ybus(ppc), v)
    spec = specified_injection(ppc)
    pq = ppc["bus"][:, 1].astype(int) == BUS_TYPE_PQ

    assert np.abs(actual.real - spec.real)[pq].max() < 1e-9
    assert np.abs(actual.imag - spec.imag)[pq].max() < 1e-9


def test_real_power_matches_at_pv_buses_too(ppc, solved):
    """At a PV bus the solver chooses Q but P is specified, so only P must
    match. Checking Q here would fail for a correct implementation."""
    _, v = solved
    actual = power_injection(build_ybus(ppc), v)
    spec = specified_injection(ppc)
    pv = ppc["bus"][:, 1].astype(int) == BUS_TYPE_PV
    assert np.abs(actual.real - spec.real)[pv].max() < 1e-9


def test_ybus_shape_and_symmetry(ppc):
    ybus = build_ybus(ppc)
    assert ybus.shape == (57, 57)
    # This case has no phase shifters (all BR_ANGLE are zero), so Ybus is
    # symmetric. A phase shifter would legitimately break this.
    assert np.abs(ppc["branch"][:, 9]).max() == 0.0
    assert np.allclose(ybus, ybus.T)


def test_taking_a_branch_out_changes_the_matrix(ppc):
    full = build_ybus(ppc)
    flags = np.ones(80, dtype=bool)
    flags[0] = False
    reduced = build_ybus(ppc, in_service=flags)
    assert not np.allclose(full, reduced)


def test_out_of_service_equals_deleting_the_branch(ppc):
    """The architectural claim in miniature: 'out of service' contributes
    nothing, which is why one model covers every outage."""
    flags = np.ones(80, dtype=bool)
    flags[7] = False
    via_flag = build_ybus(ppc, in_service=flags)

    matpower_row = int(canonical_branch_rows(ppc)[7])
    trimmed = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in ppc.items()}
    trimmed["branch"] = np.delete(ppc["branch"], matpower_row, axis=0)
    via_deletion = build_ybus(trimmed)

    assert np.allclose(via_flag, via_deletion)


def test_shunts_are_in_the_diagonal(ppc):
    """Buses 18, 25 and 53 carry shunts. Dropping them would be a real bug
    (SPEC 2.1), so prove they reach the matrix."""
    shunt_rows = np.flatnonzero(
        (ppc["bus"][:, 4] != 0) | (ppc["bus"][:, 5] != 0)
    )
    assert len(shunt_rows) == 3

    with_shunts = build_ybus(ppc)
    stripped = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in ppc.items()}
    stripped["bus"] = ppc["bus"].copy()
    stripped["bus"][:, 4] = 0.0
    stripped["bus"][:, 5] = 0.0
    without = build_ybus(stripped)

    changed = np.flatnonzero(~np.isclose(np.diag(with_shunts), np.diag(without)))
    assert changed.tolist() == shunt_rows.tolist()


def test_a_tap_ratio_of_zero_is_treated_as_one(ppc):
    """MATPOWER's convention. 63 branches in this case carry tap = 0, and
    reading that literally would divide by zero."""
    taps = ppc["branch"][:, 8]
    assert (taps == 0).sum() == 63
    assert np.isfinite(build_ybus(ppc)).all()


def test_derived_flows_match_the_solver(ppc, solved):
    """SPEC 2.1: flows are derived from voltages, never predicted separately.
    They must agree with what pandapower reports, branch for branch."""
    net, v = solved
    derived = branch_flows(ppc, v) * ppc["baseMVA"]

    # canonical order is 65 lines then 15 transformers, which is exactly how
    # pandapower lays out res_line and res_trafo
    reported = np.concatenate(
        [net.res_line.p_from_mw.to_numpy(), net.res_trafo.p_hv_mw.to_numpy()]
    )
    assert len(derived) == len(reported) == 80
    assert np.allclose(derived.real, reported, atol=1e-6)


def test_canonical_order_is_not_matpower_order(ppc):
    """The trap. pandapower splits branches into lines and transformers by tap
    ratio, which interleaves them against the case file. If this ever became
    the identity, someone has changed the ordering and every stored array in
    the project would silently mean something else."""
    rows = canonical_branch_rows(ppc)
    assert len(rows) == 80
    assert sorted(rows.tolist()) == list(range(80)), "must be a permutation"
    assert rows.tolist() != list(range(80)), "identity would mean the split changed"
    assert rows[24] == 26, "L25 is MATPOWER row 26"
    assert rows[65] == 18, "T01 is MATPOWER row 18"


def test_canonical_order_matches_the_branch_table(ppc):
    """The mapping must agree with the names, or L11 in the dataset is not L11
    in the physics."""
    from p013.case_loader import load_case
    from p013.naming import branch_table

    rows = canonical_branch_rows(ppc)
    for position, ref in enumerate(branch_table(load_case())):
        row = ppc["branch"][rows[position]]
        assert {ref.from_bus, ref.to_bus} == {int(row[0]), int(row[1])}, ref.name


def test_order_translation_round_trips(ppc):
    canonical = np.arange(80)
    assert np.array_equal(
        to_canonical_order(ppc, to_matpower_order(ppc, canonical)), canonical
    )


def test_taking_out_L25_removes_the_right_branch(ppc):
    """Named because SPEC 5.5.3's story turns on L25. Under the old implicit
    ordering this would have removed MATPOWER row 24, a different branch."""
    from p013.case_loader import load_case
    from p013.naming import branch_table

    ref = {b.name: b for b in branch_table(load_case())}["L25"]
    flags = np.ones(80, dtype=bool)
    flags[24] = False

    full = build_ybus(ppc)
    reduced = build_ybus(ppc, in_service=flags)
    changed = np.flatnonzero(~np.isclose(np.diag(full), np.diag(reduced)))
    positions = [int(x) + 1 for x in changed]  # matrix row -> bus number
    assert set(positions) == {ref.from_bus, ref.to_bus}


def test_mismatch_is_zero_on_an_outage_topology(ppc):
    """The property that matters at screening time: it must hold for any
    topology, not just the intact one."""
    net = load_case()
    name = branch_names(net)[3]
    set_in_service(net, name, False)
    pp.runpp(net, init="flat", numba=True)

    flags = np.ones(80, dtype=bool)
    flags[3] = False
    v = net.res_bus.vm_pu.to_numpy() * np.exp(
        1j * np.deg2rad(net.res_bus.va_degree.to_numpy())
    )
    actual = power_injection(build_ybus(ppc, in_service=flags), v)
    spec = specified_injection(ppc)
    pq = ppc["bus"][:, 1].astype(int) == BUS_TYPE_PQ

    assert np.abs(actual.real - spec.real)[pq].max() < 1e-9


# --------------------------------------------------------------------------
# Loading as a percentage of the published MVA rating
# --------------------------------------------------------------------------

def test_derived_apparent_power_matches_pandapower_exactly():
    """The model predicts voltages and we derive flows. If that derivation
    disagrees with the solver's own, the model is being judged against a
    quantity it cannot produce - so this pins them together."""
    import pandapower as pp

    from p013.case_loader import load_case

    net = load_case()
    pp.runpp(net, init="flat", numba=True)
    v = net.res_bus.vm_pu.to_numpy() * np.exp(
        1j * np.deg2rad(net.res_bus.va_degree.to_numpy())
    )

    ppc = parse_matpower(p013.CASE_PATH)
    derived = np.abs(branch_flows(ppc, v)) * ppc["baseMVA"]

    from_end = np.hypot(
        net.res_line.p_from_mw.to_numpy(), net.res_line.q_from_mvar.to_numpy()
    )
    assert np.abs(derived[:65] - from_end).max() < 1e-6


def test_loading_percent_is_apparent_power_over_the_published_rating():
    """SPEC 3.1 states the limit in MVA and PGLib publishes rateA in MVA.

    pandapower's own ``loading_percent`` is a CURRENT against ``max_i_ka``,
    which differs from this by a factor of 1/V - up to 28.7 percentage points
    on a collapsed contingency. Truth and model must not use different ones,
    and this is the one the spec and the data both state.
    """
    from p013.admittance import loading_percent

    ppc = parse_matpower(p013.CASE_PATH)
    rate = to_canonical_order(ppc, ppc["branch"][:, BR_RATE_A])
    v = np.ones(len(ppc["bus"]), dtype=complex)

    got = loading_percent(ppc, v)
    expected = np.abs(branch_flows(ppc, v)) * ppc["baseMVA"] / rate * 100.0
    assert np.allclose(got, expected)


def test_an_out_of_service_branch_carries_no_loading():
    from p013.admittance import loading_percent

    ppc = parse_matpower(p013.CASE_PATH)
    in_service = np.ones(80, dtype=bool)
    in_service[24] = False
    assert loading_percent(ppc, np.ones(len(ppc["bus"]), dtype=complex), in_service)[24] == 0.0


def test_every_branch_has_a_rating_so_loading_is_always_defined():
    """PGLib carries rateA on all 80. A zero would make loading infinite and
    the thermal check meaningless - SPEC 3.1 says this is exactly why the case
    was swapped for PGLib in the first place."""
    ppc = parse_matpower(p013.CASE_PATH)
    rate = to_canonical_order(ppc, ppc["branch"][:, BR_RATE_A])
    assert (rate > 0).all()
    assert rate.min() == pytest.approx(22.0)
    assert rate.max() == pytest.approx(1617.0)
