"""Operating points and outage sets. SPEC 2.4.

Outage depth is zero to three, amended 2026-09-10. With one outage per sample
the model only ever learns 0- and 1-out topologies, and the console asks it
about 2- and 3-out states as soon as a visitor clicks twice - which is the
console's entire selling point.
"""
import numpy as np
import pytest

from p013.case_loader import load_case
from p013.naming import branch_names
from p013.sampling import MAX_OUTAGES, sample_operating_point, sample_outages


def test_load_scaling_stays_in_the_stated_range():
    base = float(load_case().load.p_mw.sum())
    for seed in range(20):
        net = load_case()
        sample_operating_point(net, np.random.default_rng(seed))
        assert 0.5 * base < float(net.load.p_mw.sum()) < 1.4 * base


def test_sampling_is_reproducible_from_a_seed():
    a, b = load_case(), load_case()
    sample_operating_point(a, np.random.default_rng(7))
    sample_operating_point(b, np.random.default_rng(7))
    assert np.allclose(a.load.p_mw.to_numpy(), b.load.p_mw.to_numpy())
    assert np.allclose(a.load.q_mvar.to_numpy(), b.load.q_mvar.to_numpy())


def test_different_seeds_give_different_operating_points():
    a, b = load_case(), load_case()
    sample_operating_point(a, np.random.default_rng(1))
    sample_operating_point(b, np.random.default_rng(2))
    assert not np.allclose(a.load.p_mw.to_numpy(), b.load.p_mw.to_numpy())


def test_power_factor_is_preserved():
    """P and Q scale together. Scaling only P would silently walk the system
    to a different power factor than any real operating point."""
    base = load_case()
    net = load_case()
    sample_operating_point(net, np.random.default_rng(3))
    before = base.load.q_mvar.to_numpy() / np.where(base.load.p_mw.to_numpy() == 0, np.nan, base.load.p_mw.to_numpy())
    after = net.load.q_mvar.to_numpy() / np.where(net.load.p_mw.to_numpy() == 0, np.nan, net.load.p_mw.to_numpy())
    assert np.allclose(before, after, equal_nan=True)


def test_generation_follows_load():
    """If generation did not move with load, the slack bus would absorb the
    entire change and every sample would be a study of one generator."""
    base = float(load_case().gen.p_mw.sum())
    net = load_case()
    sample_operating_point(net, np.random.default_rng(4))
    assert float(net.gen.p_mw.sum()) != pytest.approx(base)


def test_outage_depth_covers_zero_to_three():
    net = load_case()
    depths = {len(sample_outages(net, np.random.default_rng(s))) for s in range(200)}
    assert depths == {0, 1, 2, 3}, f"got {sorted(depths)}; SPEC 2.4 requires 0-{MAX_OUTAGES}"


def test_outages_are_distinct_and_real():
    net = load_case()
    valid = set(branch_names(net))
    for seed in range(100):
        picked = sample_outages(net, np.random.default_rng(seed))
        assert len(picked) == len(set(picked)), "the same branch twice is not two outages"
        assert set(picked) <= valid


def test_outage_depths_are_roughly_balanced():
    """Uniform over depth, so the model sees each depth often enough to learn
    it. A depth that appears twice in 400 samples is not trained."""
    net = load_case()
    rng = np.random.default_rng(0)
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for _ in range(400):
        counts[len(sample_outages(net, rng))] += 1
    assert min(counts.values()) > 400 / 4 * 0.6, counts
