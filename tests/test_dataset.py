"""Turning generated data into model inputs. SPEC 2.1, 2.4."""
import numpy as np
import pytest

import p013
from p013.admittance import BUS_TYPE, BUS_TYPE_PQ, BUS_TYPE_PV, build_ybus, power_injection
from p013.case_loader import parse_matpower
from p013.dataset import load_dataset, split_by_operating_point
from p013.features import F_BRANCH, F_BUS
from p013.generate import generate_training_set


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    path = tmp_path_factory.mktemp("ds") / "train.npz"
    generate_training_set(n=200, seed=21, out_path=path)
    return load_dataset(path)


def test_shapes(data):
    n = len(data)
    assert data.bus_x.shape == (n, 57, F_BUS)
    assert data.branch_x.shape == (n, 80, F_BRANCH)
    assert data.edge_index.shape == (2, 80)
    assert data.vm.shape == (n, 57)


def test_only_converged_rows_are_kept_and_the_rest_are_counted(data):
    """SPEC 2.4: failures are excluded from training but never hidden."""
    e = data.excluded
    assert e["kept"] == len(data)
    assert e["kept"] + e["islanded"] + e["no_solution"] == e["total"]
    assert e["total"] == 200
    assert e["islanded"] + e["no_solution"] > 0, "expected some failures in 200 samples"


def test_targets_are_all_finite(data):
    assert np.isfinite(data.vm).all()
    assert np.isfinite(data.va).all()
    assert (data.vm > 0.3).all()


def test_angles_are_relative_to_the_slack(data):
    """SPEC 2.1. An absolute angle is meaningless in a power flow solution."""
    assert np.abs(data.va[:, 0]).max() < 1e-6


def test_specified_injection_varies_with_the_operating_point(data):
    assert not np.allclose(data.s_specified[0], data.s_specified[1])


def test_shunts_survive_into_the_dataset(data):
    """Three buses carry one. SPEC 2.1 calls their absence a real bug."""
    nonzero = np.flatnonzero(data.shunt != 0)
    assert nonzero.tolist() == [17, 24, 52]


def test_branch_flags_reach_the_features(data):
    """Column 5 of branch_x is the in-service flag."""
    flags = data.branch_x[:, :, 5]
    assert set(np.unique(flags).tolist()) <= {0.0, 1.0}
    out_per_row = (flags == 0).sum(axis=1)
    assert out_per_row.max() <= 3
    assert out_per_row.min() == 0


def test_split_never_shares_an_operating_point(data):
    """SPEC 2.4: split by operating condition, not randomly. A random split
    leaks the condition across the boundary and flatters the score."""
    train, val = split_by_operating_point(data, val_fraction=0.25, seed=1)
    assert len(train) + len(val) == len(data)
    assert len(val) > 0
    assert set(train.operating_point.tolist()).isdisjoint(set(val.operating_point.tolist()))


def test_split_is_reproducible(data):
    a, _ = split_by_operating_point(data, seed=5)
    b, _ = split_by_operating_point(data, seed=5)
    assert (a.operating_point == b.operating_point).all()


def test_split_keeps_the_shared_fields(data):
    train, val = split_by_operating_point(data, seed=2)
    assert (train.edge_index == data.edge_index).all()
    assert (val.shunt == data.shunt).all()


def test_dataset_module_does_not_import_torch():
    """forward_numpy shares this path and runs in a serverless bundle that has
    no torch (SPEC 5.1). A torch import here would leak into deployment."""
    import ast
    import pathlib

    source = pathlib.Path("src/p013/dataset.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "torch" not in imported


def test_the_dataset_can_rebuild_the_state_that_was_solved(data):
    """The features must describe the SAME operating point the solver answered.

    This is the project's second "dataset cannot rebuild its own input" bug.
    The first was load Q. This one is generation: ``sample_operating_point()``
    scales ``net.gen.p_mw`` by the common load factor, but the generator never
    stored that, so ``bus_features`` fell back to the case file's base
    generation. Measured on the 20,000-sample set before the fix, the real
    power mismatch at the six PV buses ran to **2.3 pu** - 230 MW of
    generation that the features claimed and the solve never had - and it
    tracked |factor - 1| exactly.

    The consequences were both silent: the model was fed the wrong injection at
    every PV bus, and the physics term was scored against a state that was
    never solved, so it pushed toward the wrong answer.

    PQ buses were always clean, which is why nothing else caught it: they carry
    no generation. The existing physics test uses the base case, where the
    factor is 1 and the bug disappears.
    """
    ppc = parse_matpower(p013.CASE_PATH)
    types = ppc["bus"][:, BUS_TYPE].astype(int)
    pv = np.flatnonzero(types == BUS_TYPE_PV)
    pq = np.flatnonzero(types == BUS_TYPE_PQ)
    assert len(pv) and len(pq)

    worst_pv = worst_pq = 0.0
    for i in range(min(len(data), 40)):
        v = (data.vm[i] * np.exp(1j * data.va[i])).astype(complex)
        ybus = build_ybus(ppc, data.branch_x[i][:, 5] > 0.5)
        gap = (power_injection(ybus, v) - data.s_specified[i]).real
        worst_pv = max(worst_pv, float(np.abs(gap[pv]).max()))
        worst_pq = max(worst_pq, float(np.abs(gap[pq]).max()))

    assert worst_pq < 1e-4, f"PQ buses do not reconcile: {worst_pq:.3e} pu"
    assert worst_pv < 1e-4, (
        f"PV buses do not reconcile: {worst_pv:.3e} pu. The features describe a "
        "different operating point than the one that was solved."
    )
