"""The physics term. SPEC 2.3.

The load-bearing test is that the loss is essentially zero at the solver's own
answer. If it is not, the term is measuring our bug rather than the model's
error, and SPEC 6's ablation - "we tried the physics term and it did not help"
- would be reporting a bug as a research finding.
"""
import numpy as np
import pandapower as pp
import pytest
import torch

import p013
from p013.admittance import (
    BUS_BS,
    BUS_GS,
    build_ybus,
    power_injection as np_power_injection,
    specified_injection,
)
from p013.case_loader import load_case, parse_matpower
from p013.features import branch_features, bus_features, edge_index
from p013.naming import set_in_service
from p013.physics import bus_masks, power_injection, power_mismatch_loss


@pytest.fixture(scope="module")
def ppc():
    return parse_matpower(p013.CASE_PATH)


@pytest.fixture(scope="module")
def shunt(ppc):
    admittance = (ppc["bus"][:, BUS_GS] + 1j * ppc["bus"][:, BUS_BS]) / ppc["baseMVA"]
    return torch.tensor(admittance, dtype=torch.complex64)


def _solved(names=()):
    net = load_case()
    for name in names:
        set_in_service(net, name, False)
    pp.runpp(net, init="flat", numba=True)
    return net


def _tensors(ppc, flags=None):
    bus_x = torch.tensor(bus_features(ppc), dtype=torch.float32).unsqueeze(0)
    branch_x = torch.tensor(
        branch_features(ppc, in_service=flags), dtype=torch.float32
    ).unsqueeze(0)
    ei = torch.tensor(edge_index(ppc), dtype=torch.long)
    return bus_x, branch_x, ei


def test_matches_the_numpy_implementation(ppc, shunt):
    """Two implementations of the same equations. If they disagree, one of them
    is wrong and neither can be trusted."""
    net = _solved()
    vm = torch.tensor(net.res_bus.vm_pu.to_numpy(), dtype=torch.float32).unsqueeze(0)
    va = torch.tensor(
        np.deg2rad(net.res_bus.va_degree.to_numpy()), dtype=torch.float32
    ).unsqueeze(0)

    _, branch_x, ei = _tensors(ppc)
    torch_s = power_injection(vm, va, branch_x, ei, shunt)[0].numpy()

    v = net.res_bus.vm_pu.to_numpy() * np.exp(1j * np.deg2rad(net.res_bus.va_degree.to_numpy()))
    numpy_s = np_power_injection(build_ybus(ppc), v)

    assert np.abs(torch_s - numpy_s).max() < 1e-4


def test_loss_is_essentially_zero_at_the_true_solution(ppc, shunt):
    net = _solved()
    vm = torch.tensor(net.res_bus.vm_pu.to_numpy(), dtype=torch.float32).unsqueeze(0)
    va = torch.tensor(
        np.deg2rad(net.res_bus.va_degree.to_numpy()), dtype=torch.float32
    ).unsqueeze(0)

    bus_x, branch_x, ei = _tensors(ppc)
    p_mask, q_mask = bus_masks(bus_x)
    s_spec = torch.tensor(specified_injection(ppc), dtype=torch.complex64).unsqueeze(0)

    loss = power_mismatch_loss(vm, va, branch_x, ei, shunt, s_spec, p_mask, q_mask)
    assert float(loss) < 1e-6, f"mismatch at the solver's own answer is {float(loss):.2e}"


def test_loss_is_large_at_a_wrong_answer(ppc, shunt):
    """The term must actually discriminate. A loss that is small everywhere
    teaches nothing."""
    bus_x, branch_x, ei = _tensors(ppc)
    p_mask, q_mask = bus_masks(bus_x)
    s_spec = torch.tensor(specified_injection(ppc), dtype=torch.complex64).unsqueeze(0)

    flat_vm = torch.ones(1, 57)
    flat_va = torch.zeros(1, 57)
    loss = power_mismatch_loss(flat_vm, flat_va, branch_x, ei, shunt, s_spec, p_mask, q_mask)
    assert float(loss) > 1e-4


def test_zero_at_the_solution_holds_on_an_outage_topology(ppc, shunt):
    """It must hold for any topology, which is the whole point of the flag."""
    net = _solved(names=("L11",))
    flags = np.ones(80, dtype=bool)
    flags[10] = False  # L11

    vm = torch.tensor(net.res_bus.vm_pu.to_numpy(), dtype=torch.float32).unsqueeze(0)
    va = torch.tensor(
        np.deg2rad(net.res_bus.va_degree.to_numpy()), dtype=torch.float32
    ).unsqueeze(0)

    bus_x, branch_x, ei = _tensors(ppc, flags=flags)
    p_mask, q_mask = bus_masks(bus_x)
    s_spec = torch.tensor(specified_injection(ppc), dtype=torch.complex64).unsqueeze(0)

    loss = power_mismatch_loss(vm, va, branch_x, ei, shunt, s_spec, p_mask, q_mask)
    assert float(loss) < 1e-6


def test_q_is_not_enforced_at_pv_buses(ppc):
    """A PV bus's reactive output is chosen by the solver. Enforcing Q there
    would punish a correct answer."""
    bus_x, _, _ = _tensors(ppc)
    p_mask, q_mask = bus_masks(bus_x)
    assert int(p_mask.sum()) == 56, "P is specified at the 50 PQ and 6 PV buses"
    assert int(q_mask.sum()) == 50, "Q is specified only at the 50 PQ buses"
    assert float(p_mask[0]) == 0.0, "nothing is specified at the slack"
    assert float(q_mask[0]) == 0.0


def test_gradients_are_finite(ppc, shunt):
    bus_x, branch_x, ei = _tensors(ppc)
    p_mask, q_mask = bus_masks(bus_x)
    s_spec = torch.tensor(specified_injection(ppc), dtype=torch.complex64).unsqueeze(0)

    vm = torch.ones(1, 57, requires_grad=True)
    va = torch.zeros(1, 57, requires_grad=True)
    power_mismatch_loss(vm, va, branch_x, ei, shunt, s_spec, p_mask, q_mask).backward()

    assert torch.isfinite(vm.grad).all()
    assert torch.isfinite(va.grad).all()
    assert vm.grad.abs().sum() > 0


def test_row_mask_excludes_unsolved_rows(ppc, shunt):
    """Rows with no valid target must not drag the loss around."""
    bus_x, branch_x, ei = _tensors(ppc)
    p_mask, q_mask = bus_masks(bus_x)
    s_spec = torch.tensor(specified_injection(ppc), dtype=torch.complex64).unsqueeze(0)

    good_vm, good_va = torch.ones(1, 57), torch.zeros(1, 57)
    two_vm = torch.cat([good_vm, good_vm * 5.0])
    two_va = torch.cat([good_va, good_va])
    two_branch = torch.cat([branch_x, branch_x])
    two_spec = torch.cat([s_spec, s_spec])

    masked = power_mismatch_loss(
        two_vm, two_va, two_branch, ei, shunt, two_spec, p_mask, q_mask,
        row_mask=torch.tensor([1.0, 0.0]),
    )
    alone = power_mismatch_loss(good_vm, good_va, branch_x, ei, shunt, s_spec, p_mask, q_mask)
    assert float(masked) == pytest.approx(float(alone), rel=1e-5)


def test_an_out_of_service_branch_contributes_nothing(ppc, shunt):
    """Same claim as the model's, now in the physics."""
    flags = np.ones(80, dtype=bool)
    flags[24] = False

    _, branch_x, ei = _tensors(ppc, flags=flags)
    vm, va = torch.ones(1, 57), torch.zeros(1, 57)
    with_flag = power_injection(vm, va, branch_x, ei, shunt)

    keep = [i for i in range(80) if i != 24]
    deleted = power_injection(vm, va, branch_x[:, keep], ei[:, keep], shunt)

    assert torch.allclose(with_flag, deleted, atol=1e-6)
