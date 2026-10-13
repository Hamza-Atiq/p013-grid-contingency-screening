"""The fast screen must give the original screen's answer, bit for bit.

Research round 1, task B. ``p013.fast_screen`` is only worth having if nothing
downstream can tell it apart from ``p013.evaluate.model_ranking`` and
``p013.topology.islanding_contingencies``. These are the fast checks; the
full 250-point parity run is ``experiments/exp_b_profile.py --parity``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from p013.evaluate import DEFAULT_SWEEP, DEFAULT_WEIGHTS, Sweep, model_ranking
from p013.fast_screen import (
    ScreenCase,
    _cmul,
    fast_islanding_contingencies,
    fast_model_ranking,
)
from p013.topology import islanding_contingencies, split_bus

needs_data = pytest.mark.skipif(
    not (Path(DEFAULT_SWEEP).exists() and Path(DEFAULT_WEIGHTS).exists()),
    reason="generated sweep and weights are gitignored; run p013.generate and p013.train",
)

#: Spread across the sweep, including the first and last point.
POINTS = (0, 61, 124, 187, 249)


@pytest.fixture(scope="module")
def sweep():
    return Sweep(Path(DEFAULT_SWEEP))


@pytest.fixture(scope="module")
def weights():
    from p013.forward_numpy import load_weights

    return load_weights(Path(DEFAULT_WEIGHTS))


@pytest.fixture(scope="module")
def case(sweep, weights):
    return ScreenCase.from_sweep(sweep, weights)


def assert_same_ranking(a, b):
    assert a.order == b.order
    assert a.pinned == b.pinned
    assert a.score.keys() == b.score.keys()
    # Exact is what is achieved; 1e-9 is the release bar.
    worst = max((abs(a.score[n] - b.score[n]) for n in a.score), default=0.0)
    assert worst <= 1e-9
    assert a.report == b.report


@needs_data
@pytest.mark.parametrize("point", POINTS)
def test_the_fast_ranking_is_the_original_ranking(sweep, weights, case, point):
    assert_same_ranking(model_ranking(sweep, point, weights), fast_model_ranking(case, sweep, point))


@needs_data
def test_the_scores_are_bit_identical_not_just_close(sweep, weights, case):
    a = model_ranking(sweep, 61, weights)
    b = fast_model_ranking(case, sweep, 61)
    assert all(a.score[n] == b.score[n] for n in a.score)


@needs_data
def test_the_measured_pre_outage_variant_matches_too(sweep, weights, case):
    assert_same_ranking(
        model_ranking(sweep, 124, weights, pre_outage="measured"),
        fast_model_ranking(case, sweep, 124, pre_outage="measured"),
    )


@needs_data
def test_the_forward_pass_is_bit_identical_on_arbitrary_topologies(sweep, weights, case):
    """The delta evaluation must not depend on the screen's one-outage layout.

    Rows here take out several random branches each, and row 0 is itself not
    intact, so the "what differs from row 0" tracking is exercised hard.
    """
    from p013.forward_numpy import forward

    rng = np.random.default_rng(20260917)
    raw = sweep.raw
    base = sweep.row(7, 0)
    bus_one = case.bus_features_at(
        raw["load_p_mw"][base], raw["load_q_mvar"][base], raw["gen_p_mw"][base]
    )
    rows = 12
    in_service = np.ones((rows, case.n_branch), dtype=bool)
    for r in range(rows):
        in_service[r, rng.choice(case.n_branch, size=r % 4, replace=False)] = False
    bus_x = np.repeat(bus_one[None].astype(np.float32), rows, axis=0)
    bus_x[5, 3, 0] += np.float32(0.25)  # one row with a different injection
    branch_x = case.branch_x(in_service)
    slack = bus_one[:, 4].astype(np.float32)

    vm_a, va_a = forward(weights, bus_x, branch_x, case.edge_index, slack_mask=slack)
    vm_b, va_b = case.forward(bus_x, branch_x, slack)
    assert np.array_equal(vm_a, vm_b)
    assert np.array_equal(va_a, va_b)


def test_the_complex_product_matches_numpy_scalar_arithmetic():
    rng = np.random.default_rng(3)
    a = rng.normal(size=500) + 1j * rng.normal(size=500)
    b = (rng.normal(size=500) + 1j * rng.normal(size=500)).astype(np.complex64)
    scalar = np.array([a[i] * b[i] for i in range(500)])
    assert np.array_equal(_cmul(a, b.astype(np.complex128)), scalar)


# ---------------------------------------------------------------------------
# Islanding
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def template():
    from p013.case_loader import CaseTemplate

    return CaseTemplate()


def test_islanding_matches_on_the_intact_case(template):
    net = template.fresh()
    assert fast_islanding_contingencies(net) == islanding_contingencies(net)


def test_islanding_matches_on_random_topologies(template):
    """Random outages of 1 to 12 branches, which create real islands and
    parallel-branch situations the intact case does not have."""
    rng = np.random.default_rng(20260917)
    for trial in range(40):
        net = template.fresh()
        n_out = 1 + trial % 12
        lines = rng.choice(len(net.line), size=min(n_out, len(net.line)), replace=False)
        net.line.loc[net.line.index[lines], "in_service"] = False
        if trial % 3 == 0:
            t = rng.integers(len(net.trafo))
            net.trafo.loc[net.trafo.index[t], "in_service"] = False
        assert fast_islanding_contingencies(net) == islanding_contingencies(net), trial


@pytest.mark.parametrize("moved, coupler_closed", [(1, False), (2, True), (2, False)])
def test_islanding_matches_on_a_split_busbar(moved, coupler_closed):
    """With the coupler open and one circuit on bar B, that circuit becomes a
    bridge - a case the intact network never produces."""
    from p013.case_loader import load_case
    from p013.naming import branch_table

    net = load_case()
    touching = [r.name for r in branch_table(net) if 9 in (r.from_bus, r.to_bus)]
    split_bus(net, 9, touching[:moved], coupler_closed=coupler_closed)
    expected = islanding_contingencies(net)
    if moved == 1 and not coupler_closed:
        assert touching[0] in expected  # the test is really testing something
    assert fast_islanding_contingencies(net) == expected
