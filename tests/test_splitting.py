"""The guided story is a measured claim, so it is a test. SPEC 5.5.3.

Substation 12 is the largest load on the system, 377 MW, with a generator.

    busbar A   L11, L25, and the load
    busbar B   L21, L23, L24, and the generator

Load and generation on opposite bars is what a double busbar is for.

Open the coupler, trip L11, and L25 becomes an islanding contingency worth
377 MW - 30.1% of system load - which it is in NEITHER the base case, NOR with
the coupler open alone, NOR with L11 out alone. It takes both steps, and that
is what makes the console's "that was not true a minute ago" true rather than
rhetorical.

If any assertion here starts failing, the console is narrating something false
and must not ship until it is re-derived.
"""
import pandapower as pp
import pytest

from p013.case_loader import load_case
from p013.naming import set_in_service
from p013.topology import islanded_load_mw, islanding_contingencies, n_components, split_bus

STORY_BUS = 12
BUSBAR_B = ["L21", "L23", "L24"]  # busbar A keeps L11, L25 and the load


def story_net(coupler_closed: bool = True, out: tuple[str, ...] = ()):
    net = load_case()
    split_bus(net, STORY_BUS, BUSBAR_B, coupler_closed=coupler_closed)
    for name in out:
        set_in_service(net, name, False)
    return net


def test_a_closed_coupler_is_electrically_invisible():
    """A zero-impedance bus-bus switch fuses the bars back into one node.

    Measured at 0.00e+00 on 2026-09-10. Anything above 1e-9 means the split is
    changing the electrical network, and every result downstream would be from
    a different system than the one we claim to model.
    """
    intact = load_case()
    pp.runpp(intact, init="flat")

    split = story_net(coupler_closed=True)
    pp.runpp(split, init="flat")

    diff = (intact.res_bus.vm_pu - split.res_bus.vm_pu.iloc[:57]).abs().max()
    assert diff < 1e-9, f"closing the coupler changed the network by {diff:.2e}"


def test_the_split_creates_exactly_one_extra_bus():
    net = story_net(coupler_closed=False)
    assert len(net.bus) == 58


def test_every_step_of_the_guided_story_converges():
    """A step that will not solve cannot show solver truth, and the console
    would fall through to the degraded path on the first thing a professor
    sees. Substation 9 was rejected for exactly this reason."""
    for label, closed, out in [
        ("step 1  base, coupler closed", True, ()),
        ("step 2  coupler OPEN", False, ()),
        ("step 3  coupler OPEN + L11 out", False, ("L11",)),
    ]:
        net = story_net(closed, out)
        pp.runpp(net, init="flat")  # must not raise
        assert n_components(net) == 1, f"{label}: network split unexpectedly"
        assert net.res_bus.vm_pu.notna().all(), f"{label}: unsolved buses"


def test_L25_islands_only_after_both_steps():
    """The whole point of the story. Three negatives and one positive."""
    assert "L25" not in islanding_contingencies(story_net(True, ())), "base case"
    assert "L25" not in islanding_contingencies(story_net(False, ())), "coupler open alone"
    assert "L25" not in islanding_contingencies(story_net(True, ("L11",))), "L11 out alone"
    assert "L25" in islanding_contingencies(story_net(False, ("L11",))), "both steps"


def test_the_payoff_is_377_MW():
    """The number the console narrates out loud."""
    net = story_net(False, ("L11", "L25"))
    assert n_components(net) == 2
    assert islanded_load_mw(net) == pytest.approx(377.0, abs=0.1)


def test_the_payoff_is_30_percent_of_the_system():
    net = story_net(False, ("L11", "L25"))
    total = float(load_case().load.p_mw.sum())
    assert 100.0 * islanded_load_mw(net) / total == pytest.approx(30.1, abs=0.2)


def test_moving_a_circuit_to_the_other_bar_changes_the_answer():
    """SPEC 5.5.4: which bar a circuit sits on is an operator choice. If the
    load's feeds are split across both bars instead of sharing one, opening
    the coupler no longer strands it."""
    net = load_case()
    split_bus(net, STORY_BUS, ["L21", "L23", "L24", "L25"], coupler_closed=False)
    set_in_service(net, "L11", False)
    assert "L25" not in islanding_contingencies(net)


def test_splitting_rejects_a_branch_that_does_not_touch_the_bus():
    net = load_case()
    with pytest.raises(ValueError, match="does not touch bus"):
        split_bus(net, STORY_BUS, ["L01"], coupler_closed=False)
