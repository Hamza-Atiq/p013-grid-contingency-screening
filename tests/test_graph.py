"""The model. SPEC 2.2.

The load-bearing test is ``test_an_out_of_service_branch_carries_no_message``.
If it fails, the central architectural claim of the whole project is false -
that one trained model covers every outage because a branch that is out simply
carries nothing.
"""
import numpy as np
import pytest
import torch

import p013
from p013.case_loader import parse_matpower
from p013.features import branch_features, bus_features, edge_index
from p013.graph import PowerFlowGNN, count_parameters


@pytest.fixture(scope="module")
def ppc():
    return parse_matpower(p013.CASE_PATH)


@pytest.fixture(scope="module")
def inputs(ppc):
    bus_x = torch.tensor(bus_features(ppc), dtype=torch.float32).unsqueeze(0)
    branch_x = torch.tensor(branch_features(ppc), dtype=torch.float32).unsqueeze(0)
    ei = torch.tensor(edge_index(ppc), dtype=torch.long)
    return bus_x, branch_x, ei


def _model(seed: int = 0) -> PowerFlowGNN:
    torch.manual_seed(seed)
    return PowerFlowGNN(hidden=16, rounds=3)


def test_output_shapes(inputs):
    bus_x, branch_x, ei = inputs
    vm, va = _model()(bus_x, branch_x, ei)
    assert vm.shape == (1, 57)
    assert va.shape == (1, 57)


def test_an_out_of_service_branch_carries_no_message(ppc):
    """THE architectural claim, in one test.

    Setting a branch's flag to zero must give exactly the same answer as
    deleting that branch from the graph. If it does not, then 'one model covers
    every outage' is not true and the project's premise fails.
    """
    model = _model().eval()

    flags = np.ones(80, dtype=bool)
    flags[24] = False  # L25
    bus_x = torch.tensor(bus_features(ppc), dtype=torch.float32).unsqueeze(0)

    via_flag_branch = torch.tensor(
        branch_features(ppc, in_service=flags), dtype=torch.float32
    ).unsqueeze(0)
    ei_full = torch.tensor(edge_index(ppc), dtype=torch.long)

    keep = [i for i in range(80) if i != 24]
    via_deletion_branch = via_flag_branch[:, keep, :]
    ei_deleted = ei_full[:, keep]

    with torch.no_grad():
        a = model(bus_x, via_flag_branch, ei_full)
        b = model(bus_x, via_deletion_branch, ei_deleted)

    assert torch.allclose(a[0], b[0], atol=1e-6)
    assert torch.allclose(a[1], b[1], atol=1e-6)


def test_taking_a_branch_out_changes_the_prediction(ppc, inputs):
    """The flag must actually be used. A model that ignored it would pass the
    test above trivially."""
    bus_x, branch_x, ei = inputs
    model = _model().eval()

    flags = np.ones(80, dtype=bool)
    flags[24] = False
    outaged = torch.tensor(branch_features(ppc, in_service=flags), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        intact_vm, _ = model(bus_x, branch_x, ei)
        outaged_vm, _ = model(bus_x, outaged, ei)

    assert not torch.allclose(intact_vm, outaged_vm, atol=1e-6)


def test_slack_angle_is_pinned_at_zero(inputs):
    """SPEC 2.1: angles are relative to the slack, which is fixed at 0. An
    absolute angle is meaningless in a power flow solution."""
    bus_x, branch_x, ei = inputs
    slack = torch.zeros(57)
    slack[0] = 1.0

    _, va = _model()(bus_x, branch_x, ei, slack_mask=slack)
    assert abs(float(va[0, 0].detach())) < 1e-6


def test_handles_a_different_bus_count():
    """A coupler opening splits one bus into two, so the graph grows to 58.
    Message passing has no fixed bus count baked in - the property that makes
    every line outage free makes busbar splitting free too (SPEC 5.1)."""
    model = _model().eval()
    torch.manual_seed(1)

    bus_x = torch.randn(1, 58, 8)
    branch_x = torch.randn(1, 81, 6)
    branch_x[..., -1] = 1.0
    ei = torch.randint(0, 58, (2, 81))

    with torch.no_grad():
        vm, va = model(bus_x, branch_x, ei)
    assert vm.shape == (1, 58)


def test_gradients_reach_every_parameter(inputs):
    bus_x, branch_x, ei = inputs
    model = _model()
    vm, va = model(bus_x, branch_x, ei)
    (vm.sum() + va.sum()).backward()

    missing = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert missing == [], f"no gradient reaches {missing} - the body trains on nothing"


def test_batching_matches_one_at_a_time(inputs):
    """Rows in a batch must not leak into each other. index_add_ over a shared
    bus axis is exactly where that would happen."""
    bus_x, branch_x, ei = inputs
    model = _model().eval()

    two_bus = torch.cat([bus_x, bus_x * 1.1], dim=0)
    two_branch = torch.cat([branch_x, branch_x], dim=0)

    with torch.no_grad():
        batched_vm, _ = model(two_bus, two_branch, ei)
        single_vm, _ = model(bus_x * 1.1, branch_x, ei)

    assert torch.allclose(batched_vm[1], single_vm[0], atol=1e-6)


def test_starts_near_a_flat_voltage_profile(inputs):
    """Untrained, the model should say roughly 1.0 pu rather than a random
    number, so early training is not spent discovering where voltages live.

    NEAR one, not exactly one. An exact constant means the last layer's weight
    is zero, which stops all gradient reaching the body - see the note in
    graph.py. The tolerance here is what distinguishes 'sensibly initialised'
    from 'silently dead'.
    """
    bus_x, branch_x, ei = inputs
    with torch.no_grad():
        vm, va = PowerFlowGNN(hidden=16, rounds=3)(bus_x, branch_x, ei)
    assert torch.allclose(vm, torch.ones_like(vm), atol=0.1)
    assert torch.allclose(va, torch.zeros_like(va), atol=0.1)
    assert not torch.allclose(vm, torch.ones_like(vm), atol=1e-9), "exactly constant = dead body"


def test_parameter_count_is_small_enough_for_a_cpu(inputs):
    model = PowerFlowGNN(hidden=64, rounds=6)
    n = count_parameters(model)
    assert n < 200_000, f"{n} parameters is more than a dual-core CPU should train"


def test_is_deterministic(inputs):
    bus_x, branch_x, ei = inputs
    a = _model(3).eval()
    b = _model(3).eval()
    with torch.no_grad():
        assert torch.allclose(a(bus_x, branch_x, ei)[0], b(bus_x, branch_x, ei)[0])
