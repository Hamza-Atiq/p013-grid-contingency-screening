"""What the model is given. SPEC 2.1."""
import numpy as np
import pytest

import p013
from p013.admittance import canonical_branch_rows
from p013.case_loader import load_case, parse_matpower
from p013.features import (
    BRANCH_FEATURE_NAMES,
    BUS_FEATURE_NAMES,
    F_BRANCH,
    F_BUS,
    branch_features,
    bus_features,
    edge_index,
)
from p013.naming import branch_table


@pytest.fixture(scope="module")
def ppc():
    return parse_matpower(p013.CASE_PATH)


def test_shapes(ppc):
    assert bus_features(ppc).shape == (57, F_BUS)
    assert branch_features(ppc).shape == (80, F_BRANCH)
    assert edge_index(ppc).shape == (2, 80)
    assert len(BUS_FEATURE_NAMES) == F_BUS
    assert len(BRANCH_FEATURE_NAMES) == F_BRANCH


def test_shunts_reach_the_features(ppc):
    """SPEC 2.1 calls their absence a real bug. Three buses carry one."""
    x = bus_features(ppc)
    gs = x[:, BUS_FEATURE_NAMES.index("gs")]
    bs = x[:, BUS_FEATURE_NAMES.index("bs")]
    nonzero = np.flatnonzero((gs != 0) | (bs != 0))
    assert len(nonzero) == 3
    # buses 18, 25 and 53 -> zero-based positions 17, 24, 52
    assert nonzero.tolist() == [17, 24, 52]

    # Sign convention, pinned because it is exactly where this kind of bug
    # lives. In MATPOWER, Bs POSITIVE is capacitive (a shunt that injects
    # reactive power). pandapower reports the same three shunts as
    # q_mvar = -10.0, -5.9, -6.3, i.e. with the opposite sign. Both describe
    # the same capacitors. Features follow the MATPOWER convention because
    # that is the array the admittance matrix is built from.
    assert (bs[nonzero] > 0).all(), "these three shunts are capacitive"
    assert np.allclose(bs[nonzero] * 100.0, [10.0, 5.9, 6.3])
    assert (gs[nonzero] == 0).all(), "all three are purely reactive"


def test_bus_type_one_hot_sums_to_one(ppc):
    x = bus_features(ppc)
    one_hot = x[:, 2:5]
    assert np.allclose(one_hot.sum(axis=1), 1.0)
    assert one_hot.sum(axis=0).tolist() == [50.0, 6.0, 1.0]


def test_slack_is_bus_one(ppc):
    x = bus_features(ppc)
    assert x[0, BUS_FEATURE_NAMES.index("is_slack")] == 1.0


def test_injection_follows_the_supplied_operating_point(ppc):
    base = bus_features(ppc)
    doubled = bus_features(ppc, load_p=ppc["bus"][:, 2] * 2, load_q=ppc["bus"][:, 3] * 2)
    load_buses = np.flatnonzero(ppc["bus"][:, 2] != 0)
    # doubling demand makes net injection more negative at load buses
    assert (doubled[load_buses, 0] < base[load_buses, 0]).all()


def test_generator_q_is_not_an_input(ppc):
    """Q at a generator is chosen by the solver. Feeding it in would hand the
    model part of the answer it is supposed to predict."""
    x = bus_features(ppc)
    pv = x[:, BUS_FEATURE_NAMES.index("is_pv")] == 1
    assert (x[pv, 1] <= 0).all(), "q_injection at PV buses should be demand only"


def test_out_of_service_branch_flag_is_zero_and_impedance_survives(ppc):
    """The model must learn 'this line exists and carries nothing', not 'this
    line has no impedance'. SPEC 2.2."""
    flags = np.ones(80, dtype=bool)
    flags[24] = False  # L25
    intact = branch_features(ppc)
    outaged = branch_features(ppc, in_service=flags)

    assert outaged[24, 5] == 0.0
    assert intact[24, 5] == 1.0
    assert np.allclose(outaged[24, :5], intact[24, :5])
    assert np.allclose(np.delete(outaged, 24, axis=0), np.delete(intact, 24, axis=0))


def test_tap_ratio_of_zero_becomes_one(ppc):
    x = branch_features(ppc)
    assert (x[:, 3] > 0).all(), "a ratio of zero would be a degenerate transformer"
    assert (x[:65, 3] == 1.0).all(), "the 65 lines all have unit ratio"


def test_branch_features_are_in_canonical_order(ppc):
    """Row i must be the branch named i in branch_table, or every stored array
    in the project means something else."""
    rows = canonical_branch_rows(ppc)
    x = branch_features(ppc)
    for position, ref in enumerate(branch_table(load_case())):
        assert x[position, 0] == ppc["branch"][rows[position], 2], ref.name


def test_edge_index_matches_the_branch_table(ppc):
    ei = edge_index(ppc)
    for position, ref in enumerate(branch_table(load_case())):
        pair = {int(ei[0, position]) + 1, int(ei[1, position]) + 1}
        assert pair == {ref.from_bus, ref.to_bus}, ref.name


def test_everything_is_finite(ppc):
    assert np.isfinite(bus_features(ppc)).all()
    assert np.isfinite(branch_features(ppc)).all()


def test_wrong_flag_length_is_rejected(ppc):
    with pytest.raises(ValueError, match="expected 80 flags"):
        branch_features(ppc, in_service=np.ones(79, dtype=bool))
