"""One index space, one display name. SPEC 5.5.1, 5.5.6.

The naming question was open in the spec: pandapower's converter builds 65
lines + 15 transformers, while the 2026-09-09 review counted 17 tap-ratio
branches. Resolved 2026-09-10 by reading the case: 63 branches have tap = 0,
15 have a ratio that is not 1, and 2 have a ratio of exactly 1.0. The review
counted the 17 with a non-zero ratio field; pandapower builds the 15 that are
electrically transformers, treating a 1:1 ratio as a line. Both were right
about different things. Shipped naming is L01-L65 and T01-T15.
"""
import pytest

from p013.case_loader import load_case
from p013.naming import branch_index, branch_names, branch_table


def test_eighty_unique_names_in_a_stable_order():
    net = load_case()
    names = branch_names(net)
    assert len(names) == 80
    assert len(set(names)) == 80
    assert names[0] == "L01" and names[64] == "L65"
    assert names[65] == "T01" and names[79] == "T15"


def test_name_round_trips_to_a_pandapower_index():
    net = load_case()
    for name in branch_names(net):
        kind, idx = branch_index(net, name)
        assert kind in ("line", "trafo")
        frame = net.line if kind == "line" else net.trafo
        assert idx in frame.index


def test_the_story_branches_exist_and_touch_substation_12():
    """SPEC 5.5.3 names L11, L25, L21, L23, L24 at bus 12. If naming ever
    shifts, the guided story silently points at the wrong branches and the
    console narrates something false."""
    net = load_case()
    table = {b.name: b for b in branch_table(net)}
    for name in ("L11", "L25", "L21", "L23", "L24"):
        assert name in table, f"{name} does not exist"
        assert 12 in (table[name].from_bus, table[name].to_bus), f"{name} should touch bus 12"


def test_substation_12_has_exactly_the_five_branches_the_story_assumes():
    """Busbar A = L11 + L25, busbar B = L21 + L23 + L24. If bus 12 gained or
    lost a branch the split would be wrong and the 377 MW claim with it."""
    net = load_case()
    touching = {b.name for b in branch_table(net) if 12 in (b.from_bus, b.to_bus)}
    assert touching == {"L11", "L21", "L23", "L24", "L25"}


def test_unknown_name_is_an_error_not_a_guess():
    net = load_case()
    with pytest.raises(KeyError):
        branch_index(net, "L99")


def test_names_are_stable_across_loads():
    a, b = branch_names(load_case()), branch_names(load_case())
    assert a == b
