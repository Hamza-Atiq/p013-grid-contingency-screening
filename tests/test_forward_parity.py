"""THE RELEASE GATE. SPEC 5.1.

The console asserts things about the model. If the served NumPy forward pass
differs from the trained PyTorch one, the console is lying about what the model
said. This is not a nice-to-have test.
"""
import subprocess
import sys

import numpy as np
import pytest
import torch

import p013
from p013.case_loader import parse_matpower
from p013.export_weights import export
from p013.features import branch_features, bus_features, edge_index
from p013.forward_numpy import forward, load_weights
from p013.graph import PowerFlowGNN

TOLERANCE = 1e-5


@pytest.fixture(scope="module")
def ppc():
    return parse_matpower(p013.CASE_PATH)


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    torch.manual_seed(7)
    model = PowerFlowGNN(hidden=64, rounds=6).eval()
    path = tmp_path_factory.mktemp("w") / "weights.npz"
    export(model, path)
    return model, load_weights(path)


def _inputs(ppc, flags=None):
    bus = bus_features(ppc)[None]
    branch = branch_features(ppc, in_service=flags)[None]
    return bus.astype(np.float32), branch.astype(np.float32), edge_index(ppc)


def test_numpy_matches_torch_on_the_intact_case(ppc, pair):
    model, weights = pair
    bus, branch, ei = _inputs(ppc)

    with torch.no_grad():
        t_vm, t_va = model(torch.from_numpy(bus), torch.from_numpy(branch), torch.from_numpy(ei))
    n_vm, n_va = forward(weights, bus, branch, ei)

    assert np.abs(t_vm.numpy() - n_vm).max() < TOLERANCE
    assert np.abs(t_va.numpy() - n_va).max() < TOLERANCE


def test_numpy_matches_torch_over_fifty_random_topologies(ppc, pair):
    """Including multi-branch outages, which is what a console visitor builds."""
    model, weights = pair
    rng = np.random.default_rng(0)
    worst = 0.0

    for _ in range(50):
        flags = np.ones(80, dtype=bool)
        for k in rng.choice(80, size=int(rng.integers(0, 4)), replace=False):
            flags[k] = False
        bus, branch, ei = _inputs(ppc, flags)

        with torch.no_grad():
            t_vm, t_va = model(
                torch.from_numpy(bus), torch.from_numpy(branch), torch.from_numpy(ei)
            )
        n_vm, n_va = forward(weights, bus, branch, ei)
        worst = max(worst, float(np.abs(t_vm.numpy() - n_vm).max()))
        worst = max(worst, float(np.abs(t_va.numpy() - n_va).max()))

    assert worst < TOLERANCE, f"worst disagreement {worst:.2e}"


def test_parity_holds_on_a_split_busbar(ppc, pair):
    """58 buses, 81 branches - what the console produces when a coupler opens."""
    model, weights = pair
    rng = np.random.default_rng(3)
    bus = rng.standard_normal((1, 58, 8)).astype(np.float32)
    branch = rng.standard_normal((1, 81, 6)).astype(np.float32)
    branch[..., -1] = 1.0
    ei = rng.integers(0, 58, size=(2, 81))

    with torch.no_grad():
        t_vm, _ = model(torch.from_numpy(bus), torch.from_numpy(branch), torch.from_numpy(ei))
    n_vm, _ = forward(weights, bus, branch, ei)

    assert np.abs(t_vm.numpy() - n_vm).max() < TOLERANCE


def test_parity_holds_with_the_slack_pinned(ppc, pair):
    model, weights = pair
    bus, branch, ei = _inputs(ppc)
    slack = bus[0, :, 4].copy()

    with torch.no_grad():
        _, t_va = model(
            torch.from_numpy(bus), torch.from_numpy(branch), torch.from_numpy(ei),
            slack_mask=torch.from_numpy(slack),
        )
    _, n_va = forward(weights, bus, branch, ei, slack_mask=slack)

    assert np.abs(t_va.numpy() - n_va).max() < TOLERANCE
    assert abs(float(n_va[0, 0])) < 1e-6


def test_batching_is_consistent(ppc, pair):
    _, weights = pair
    bus, branch, ei = _inputs(ppc)
    two_bus = np.concatenate([bus, bus * 1.05])
    two_branch = np.concatenate([branch, branch])

    batched_vm, _ = forward(weights, two_bus, two_branch, ei)
    single_vm, _ = forward(weights, bus * 1.05, branch, ei)
    assert np.abs(batched_vm[1] - single_vm[0]).max() < 1e-6


def test_rounds_travel_with_the_weights(pair):
    """The server must not have to be told the architecture separately, since
    that is one more thing that can disagree with the trained model."""
    model, weights = pair
    assert int(weights["meta_rounds"]) == model.rounds
    assert int(weights["meta_hidden"]) == model.hidden


def test_the_served_forward_pass_does_not_need_torch(tmp_path):
    """The serverless bundle will not contain torch. Importing forward_numpy
    must not pull it in - checked by running in a subprocess that blocks it."""
    script = (
        "import sys\n"
        "class Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name == 'torch': raise ImportError('torch is not in the bundle')\n"
        "sys.meta_path.insert(0, Block())\n"
        "import p013.forward_numpy as f\n"
        "assert 'torch' not in sys.modules, 'forward_numpy imported torch'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
