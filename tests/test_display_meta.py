"""PGLib normalises the voltage levels away; we recover them, having checked.

SPEC 5.5.6 draws voltage level as line weight and names bus 41 as the 230 kV
substation. Measured 2026-09-10: PGLib sets baseKV = 1.0 on all 57 buses, so
neither claim is true of the shipped case. The real IEEE levels live in
pandapower's built-in case57, on the same topology.

Borrowing them is only legitimate if the two cases really are the same network,
so that is verified rather than assumed - the +1 offset was originally noticed
from a degree sequence, which is nowhere near proof.
"""
import pytest

from p013.case_loader import load_case
from p013.display_meta import bus_kv, verify_topology_match


def test_pglib_really_has_no_voltage_levels():
    """The premise of this whole module. If PGLib ever ships real base
    voltages, delete this module and read them directly."""
    net = load_case()
    assert set(net.bus.vn_kv.unique()) == {1.0}


def test_the_two_cases_describe_the_same_network():
    verify_topology_match()  # raises with a diff if they disagree


def test_real_voltage_levels_are_recovered():
    """Six levels, not three.

    Measured 2026-09-10. SPEC 5.5.6 says "115/138/230 kV", which came from
    reading only the fifteen highest-degree buses in bus_ranking.json. The real
    IEEE 57-bus system has six levels, and the highest three carry 13 buses
    that the spec never mentions.
    """
    kv = bus_kv()
    assert set(kv) == set(range(1, 58))
    assert sorted(set(kv.values())) == [115.0, 138.0, 161.0, 230.0, 345.0, 500.0]


def test_level_populations_are_what_the_diagram_must_cope_with():
    """SPEC 5.5.6 draws voltage level as line weight. Six levels is more than
    stroke width can separate legibly at 1024x768; this pins the real spread so
    the decision is taken against facts."""
    kv = bus_kv()
    counts = {level: sum(1 for v in kv.values() if v == level) for level in set(kv.values())}
    assert counts == {115.0: 17, 138.0: 19, 161.0: 8, 230.0: 5, 345.0: 5, 500.0: 3}


def test_the_five_detailed_substations_span_voltage_levels():
    """SPEC 5.5.6 picks the five partly for voltage spread. Measured: they
    cover 115, 138 and 230 kV, and miss 161, 345 and 500 entirely."""
    kv = bus_kv()
    levels = {kv[b] for b in (12, 9, 38, 41, 1)}
    assert levels == {115.0, 138.0, 230.0}
    assert len(levels) >= 2, f"the five substations sit at only {levels}"


def test_bus_41_is_the_230_kv_one():
    """Named explicitly in SPEC 5.5.6. If the offset were wrong this fails."""
    assert bus_kv()[41] == pytest.approx(230.0)


def test_mismatch_is_reported_not_swallowed():
    with pytest.raises(ValueError, match="not the same network"):
        verify_topology_match(_offset=0)
