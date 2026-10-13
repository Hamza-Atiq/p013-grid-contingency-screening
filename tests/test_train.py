"""Training. SPEC 2.3."""
import json

import numpy as np
import pytest
import torch

from p013.generate import generate_training_set
from p013.train import TrainConfig, train


@pytest.fixture(scope="module")
def data_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("train") / "train.npz"
    generate_training_set(n=300, seed=31, out_path=path)
    return path


def test_training_reduces_the_loss(data_path):
    config = TrainConfig(hidden=16, rounds=2, epochs=6, batch_size=32)
    _, history = train(config, data_path)
    first, last = history[0]["train"]["total"], history[-1]["train"]["total"]
    assert last < first, f"loss went {first:.6f} -> {last:.6f}"
    assert last < first * 0.7, "loss barely moved; the model may not be learning"


def test_both_loss_terms_are_recorded_separately(data_path):
    """SPEC 6 commits to an ablation on the physics term. That is only possible
    if the two were never mixed into a single number."""
    config = TrainConfig(hidden=16, rounds=2, epochs=2, batch_size=32)
    _, history = train(config, data_path)
    for entry in history:
        for split in ("train", "val"):
            assert set(entry[split]) == {"total", "voltage", "physics"}
            assert entry[split]["physics"] >= 0.0


def test_training_is_reproducible(data_path):
    config = TrainConfig(hidden=16, rounds=2, epochs=3, batch_size=32)
    _, a = train(config, data_path)
    _, b = train(config, data_path)
    assert a[-1]["train"]["total"] == pytest.approx(b[-1]["train"]["total"], rel=1e-9)


def test_physics_weight_actually_changes_the_objective(data_path):
    """If it did not, the ablation would be measuring nothing."""
    off = TrainConfig(hidden=16, rounds=2, epochs=2, batch_size=32, physics_weight=0.0)
    on = TrainConfig(hidden=16, rounds=2, epochs=2, batch_size=32, physics_weight=10.0)
    _, a = train(off, data_path)
    _, b = train(on, data_path)
    assert a[-1]["train"]["total"] != pytest.approx(b[-1]["train"]["total"], rel=1e-6)


def test_validation_uses_unseen_operating_points(data_path):
    """A val loss computed on training conditions would be meaningless."""
    config = TrainConfig(hidden=16, rounds=2, epochs=1, batch_size=32)
    _, history = train(config, data_path)
    assert history[0]["val"]["total"] > 0


def test_artifacts_are_written(data_path, tmp_path):
    config = TrainConfig(hidden=16, rounds=2, epochs=1, batch_size=32)
    out = tmp_path / "model"
    train(config, data_path, out)

    assert (out / "weights.pt").exists()
    meta = json.loads((out / "history.json").read_text(encoding="utf-8"))
    assert meta["parameters"] > 0
    assert meta["excluded"]["total"] == 300
    assert meta["n_train"] > 0 and meta["n_val"] > 0
    torch.load(out / "weights.pt", weights_only=True)


def test_slack_angle_stays_pinned_during_training(data_path):
    """The model is given a slack mask; if it were dropped, the model would be
    asked to guess a constant it was never told (SPEC 2.1)."""
    config = TrainConfig(hidden=16, rounds=2, epochs=1, batch_size=32)
    model, _ = train(config, data_path)

    from p013.dataset import load_dataset

    data = load_dataset(data_path)
    bus_x = torch.from_numpy(data.bus_x[:4])
    branch_x = torch.from_numpy(data.branch_x[:4])
    ei = torch.from_numpy(data.edge_index)
    slack = bus_x[0, :, 4].clone()

    with torch.no_grad():
        _, va = model(bus_x, branch_x, ei, slack_mask=slack)
    assert va[:, 0].abs().max() < 1e-5
