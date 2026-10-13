"""The torch score used for training. Research round 1, task C.

The load-bearing test is parity: the thermal term here must be
:func:`p013.admittance.loading_percent`, on real solved rows with their own
outages, and the whole score must be :func:`p013.danger.risk_score` when the
smoothing is off. If they differ, the model is trained toward a score the
ranking never computes, and a failed experiment would be reported as "training
on the score does not help" when it was our port that was wrong.

Real rows come from ``data_smoke/train.npz``. Only their voltages and in-service
flags are used, so the file predating the ``gen_p_mw`` fix does not matter here.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import p013
from p013 import admittance, danger
from p013.case_loader import parse_matpower
from p013.dataset import Dataset, to_bus_order
from p013.features import branch_features, bus_features, edge_index
from p013.graph import PowerFlowGNN
from p013.score_loss import (
    Ratings,
    excess_loss,
    loading_percent,
    risk_score,
    score_loss,
    score_terms,
    smooth_max,
    soft_excess,
)
from p013.train import TrainConfig, _Batches, _losses

torch.set_num_threads(1)

BUILD = Path(__file__).resolve().parents[1]
SMOKE = BUILD / "data_smoke" / "train.npz"


def _train_score():
    spec = importlib.util.spec_from_file_location(
        "train_score", BUILD / "experiments" / "train_score.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Registered first: dataclasses looks the module up by name while building
    # ScoreTrainConfig, and an unregistered module is not found.
    sys.modules["train_score"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ppc():
    return parse_matpower(p013.CASE_PATH)


@pytest.fixture(scope="module")
def rows(ppc):
    """Converged rows, with outages of every size: (vm, va_rad, in_service)."""
    raw = np.load(SMOKE, allow_pickle=False)
    keep = np.flatnonzero(raw["status"] == "converged")
    outages = (~raw["in_service"][keep]).sum(axis=1)
    # Up to 10 rows of each outage count, so every topology size is exercised.
    chosen = np.concatenate([keep[outages == k][:10] for k in range(4)])
    return (
        raw["vm_pu"][chosen].astype(float),
        np.deg2rad(raw["va_degree"][chosen]).astype(float),
        raw["in_service"][chosen].astype(bool),
    )


def _branch_x(ppc, in_service, dtype):
    return torch.tensor(
        np.stack([branch_features(ppc, flags) for flags in in_service]), dtype=dtype
    )


def _numpy_loading(ppc, vm, va, in_service):
    return np.stack(
        [
            admittance.loading_percent(ppc, vm[i] * np.exp(1j * va[i]), in_service[i])
            for i in range(len(vm))
        ]
    )


# ---- thermal term parity ------------------------------------------------------


def test_loading_matches_numpy_on_real_rows(ppc, rows):
    vm, va, in_service = rows
    assert set((~in_service).sum(axis=1)) == {0, 1, 2, 3}
    expected = _numpy_loading(ppc, vm, va, in_service)

    got = loading_percent(
        torch.tensor(vm), torch.tensor(va), _branch_x(ppc, in_service, torch.float64),
        torch.tensor(edge_index(ppc)), Ratings.from_ppc(ppc, torch.float64),
    ).numpy()

    loaded = expected > 1.0  # percent; "non-trivial" loading
    rel = np.abs(got[loaded] - expected[loaded]) / expected[loaded]
    assert rel.max() < 1e-5, f"worst relative error {rel.max():.3e}"
    assert np.abs(got - expected).max() < 1e-6  # percentage points, everywhere
    # Out of service reads exactly zero on both sides.
    assert np.all(got[~in_service] == 0.0) and np.all(expected[~in_service] == 0.0)


def test_loading_in_float32_as_trained_stays_close(ppc, rows):
    """Training runs in float32. Flows are a difference of large currents, so
    some digits go; this bounds how many."""
    vm, va, in_service = rows
    expected = _numpy_loading(ppc, vm, va, in_service)
    got = loading_percent(
        torch.tensor(vm, dtype=torch.float32), torch.tensor(va, dtype=torch.float32),
        _branch_x(ppc, in_service, torch.float32), torch.tensor(edge_index(ppc)),
        Ratings.from_ppc(ppc),
    ).numpy()
    # 0.05 percentage points is 0.0005 in score units: invisible to the ranking.
    assert np.abs(got - expected).max() < 0.05


def test_loading_gradients_flow_and_are_finite(ppc, rows):
    """Including rows with branches out, whose flow is exactly zero - the one
    place ``abs`` of a complex number could produce a NaN gradient."""
    vm, va, in_service = rows
    vm_t = torch.tensor(vm, dtype=torch.float32, requires_grad=True)
    va_t = torch.tensor(va, dtype=torch.float32, requires_grad=True)
    loading = loading_percent(
        vm_t, va_t, _branch_x(ppc, in_service, torch.float32),
        torch.tensor(edge_index(ppc)), Ratings.from_ppc(ppc),
    )
    loading.sum().backward()
    for grad in (vm_t.grad, va_t.grad):
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


# ---- the score, exact mode ----------------------------------------------------


def test_thermal_term_from_true_voltage_matches_risk_score(ppc, rows):
    vm, va, in_service = rows
    loading = _numpy_loading(ppc, vm, va, in_service)
    thermal, _ = score_terms(
        torch.tensor(vm), torch.tensor(va), torch.tensor(vm),
        _branch_x(ppc, in_service, torch.float64), torch.tensor(edge_index(ppc)),
        Ratings.from_ppc(ppc, torch.float64),
    )
    expected = loading.max(axis=1) / danger.RATING_PERCENT
    np.testing.assert_allclose(thermal.numpy(), expected, rtol=1e-7)


def test_whole_score_matches_danger_risk_score(ppc, rows):
    """V_before is taken from another row, so the voltage term is non-trivial
    (vm_before = vm would make it identically zero and prove nothing)."""
    vm, va, in_service = rows
    vm_before = np.roll(vm, 1, axis=0)
    loading = _numpy_loading(ppc, vm, va, in_service)

    got = risk_score(
        torch.tensor(vm), torch.tensor(va), torch.tensor(vm_before),
        _branch_x(ppc, in_service, torch.float64), torch.tensor(edge_index(ppc)),
        Ratings.from_ppc(ppc, torch.float64),
    ).numpy()
    expected = np.array(
        [danger.risk_score(vm[i], loading[i], vm_before[i]) for i in range(len(vm))]
    )
    np.testing.assert_allclose(got, expected, rtol=1e-7, atol=1e-9)
    # Both terms must actually be exercised, or the test is weaker than it reads.
    _, voltage = score_terms(
        torch.tensor(vm), torch.tensor(va), torch.tensor(vm_before),
        _branch_x(ppc, in_service, torch.float64), torch.tensor(edge_index(ppc)),
        Ratings.from_ppc(ppc, torch.float64),
    )
    assert (voltage.numpy() != 0).any()


def test_exact_excess_matches_danger():
    vm = np.linspace(0.8, 1.2, 401)
    np.testing.assert_allclose(soft_excess(torch.tensor(vm)).numpy(), danger.excess(vm), atol=1e-12)


# ---- smoothing ----------------------------------------------------------------


def test_smooth_max_approaches_max_from_above():
    torch.manual_seed(0)
    x = torch.randn(16, 57, dtype=torch.float64)
    hard = x.amax(dim=-1)
    previous = None
    for tau in (0.5, 0.1, 0.02, 0.001):
        soft = smooth_max(x, tau)
        gap = soft - hard
        assert (gap >= -1e-12).all()
        assert (gap <= tau * np.log(57) + 1e-12).all()
        if previous is not None:
            assert gap.max() < previous
        previous = gap.max()
    assert previous < 1e-3
    assert torch.equal(smooth_max(x, 0.0), hard)


def test_smooth_max_spreads_the_gradient():
    x = torch.tensor([[1.0, 0.99, 0.0]], requires_grad=True)
    smooth_max(x, 0.02).sum().backward()
    assert x.grad[0, 0] > 0 and x.grad[0, 1] > 0  # the near-winner hears it too
    assert x.grad.sum().item() == pytest.approx(1.0, rel=1e-5)


def test_soft_excess_is_close_and_gives_signal_inside_the_band():
    vm = torch.linspace(0.85, 1.15, 301, dtype=torch.float64)
    tau = 0.05
    gap = soft_excess(vm, tau) - soft_excess(vm)
    assert (gap >= -1e-12).all() and (gap <= tau * np.log(2) + 1e-12).all()

    inside = torch.tensor([0.955], requires_grad=True)
    soft_excess(inside, tau).sum().backward()
    assert inside.grad.abs().item() > 0


# ---- the loss -----------------------------------------------------------------


def test_under_scoring_a_dangerous_case_costs_most():
    f64 = lambda v: torch.tensor([v], dtype=torch.float64)  # noqa: E731
    target = f64(1.2)
    under = score_loss(f64(1.0), target)
    over = score_loss(f64(1.4), target)
    assert under == pytest.approx(2.0 * over)

    safe = score_loss(f64(0.1), f64(0.3))  # same error, safe row
    assert under == pytest.approx(5.0 * safe)


def test_cap_stops_the_far_tail_dominating():
    far = score_loss(torch.tensor([5.0]), torch.tensor([6.0]))
    assert far.item() == 0.0  # both plainly dangerous: nothing to learn for ranking
    near = score_loss(torch.tensor([0.6]), torch.tensor([6.0]))
    assert near.item() == pytest.approx(5.0 * 2.0 * 1.4**2)
    assert score_loss(torch.tensor([5.0]), torch.tensor([6.0]), cap=0).item() > 0


def test_excess_loss_is_silent_inside_the_band():
    assert excess_loss(torch.tensor([[1.01, 0.97]]), torch.tensor([[0.99, 1.04]])).item() == 0.0
    assert excess_loss(torch.tensor([[0.93]]), torch.tensor([[0.89]])).item() > 0


# ---- the training entry -------------------------------------------------------


@pytest.fixture(scope="module")
def batch_setup(ppc):
    """A small Dataset built from smoke rows, and a tiny model."""
    raw = np.load(SMOKE, allow_pickle=False)
    keep = np.flatnonzero(raw["status"] == "converged")[:48]
    load_buses = [int(b[0]) for b in ppc["bus"] if b[2] != 0 or b[3] != 0]
    load_p = to_bus_order(ppc, load_buses, raw["load_p_mw"][keep])
    load_q = to_bus_order(ppc, load_buses, raw["load_q_mvar"][keep])
    bus_x = np.stack([bus_features(ppc, load_p[i], load_q[i]) for i in range(len(keep))])
    va = np.deg2rad(raw["va_degree"][keep])
    data = Dataset(
        bus_x=bus_x.astype(np.float32),
        branch_x=np.stack([branch_features(ppc, raw["in_service"][i]) for i in keep]).astype(np.float32),
        edge_index=edge_index(ppc),
        vm=raw["vm_pu"][keep].astype(np.float32),
        va=(va - va[:, [0]]).astype(np.float32),
        s_specified=(bus_x[:, :, 0] + 1j * bus_x[:, :, 1]).astype(np.complex64),
        shunt=((ppc["bus"][:, 4] + 1j * ppc["bus"][:, 5]) / ppc["baseMVA"]).astype(np.complex64),
        operating_point=raw["operating_point_hash"][keep],
        excluded={},
    )
    batches = _Batches(data, 16, seed=1)
    torch.manual_seed(7)
    model = PowerFlowGNN(hidden=8, rounds=2)
    slack = batches.bus_x[0, :, 4].clone()
    return batches, model, slack, Ratings.from_ppc(ppc)


def test_zero_new_weights_is_exactly_the_main_loss(batch_setup):
    ts = _train_score()
    batches, model, slack, ratings = batch_setup
    batch = next(batches.epoch(shuffle=False))
    base = TrainConfig()
    config = ts.ScoreTrainConfig(score_weight=0.0, excess_weight=0.0)
    main_total, main_v, main_p = _losses(model, batch, batches, base, slack)
    total, voltage, physics, score, excess = ts.score_losses(
        model, batch, batches, config, slack, ratings
    )
    assert total.item() == main_total.item()
    assert voltage.item() == main_v.item() and physics.item() == main_p.item()
    assert score.item() == 0.0 and excess.item() == 0.0


def test_score_term_reaches_the_weights(batch_setup):
    ts = _train_score()
    batches, model, slack, ratings = batch_setup
    batch = next(batches.epoch(shuffle=False))
    config = ts.ScoreTrainConfig(score_weight=1.0, excess_weight=1.0)
    _, _, _, score, excess = ts.score_losses(model, batch, batches, config, slack, ratings)
    assert score.item() > 0

    model.zero_grad()
    score.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    # Reaches the body of the network, not only the head.
    assert model.encode[0].weight.grad.abs().sum() > 0
    assert model.message[0].weight.grad.abs().sum() > 0


def test_intact_reference_is_all_in_service_and_detached(batch_setup):
    ts = _train_score()
    batches, model, slack, _ = batch_setup
    bus_x, branch_x, *_ = next(batches.epoch(shuffle=False))
    assert (branch_x[..., -1] == 0).any()  # the batch really has outages
    vm_before = ts.intact_reference(model, bus_x, branch_x, batches.edge_index, slack)
    assert not vm_before.requires_grad
    assert (branch_x[..., -1] == 0).any()  # the input was not modified in place

    intact = branch_x.clone()
    intact[..., -1] = 1.0
    with torch.no_grad():
        expected, _ = model(bus_x, intact, batches.edge_index, slack_mask=slack)
    assert torch.equal(vm_before, expected)
