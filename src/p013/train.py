"""Train the model. SPEC 2.3.

``loss = voltage error + lambda * power mismatch``

Both terms are recorded separately in the history, because SPEC 6 commits to
running an ablation on the physics term and reporting the result either way -
"we tried it and it did not help on this system" is a stronger sentence than a
term left in for decoration. That is only possible if the two were never mixed
into one number.

No GPU. Measured on the target machine: numpy reaches 72 GFLOP/s, and one epoch
over 20,000 samples at hidden 64 / 6 rounds is about 835 GFLOP, so a full run is
hours rather than days on a six-year-old dual-core laptop.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from p013.dataset import Dataset, load_dataset, split_by_operating_point
from p013.graph import PowerFlowGNN, count_parameters
from p013.physics import bus_masks, power_mismatch_loss


@dataclass
class TrainConfig:
    hidden: int = 64
    rounds: int = 6
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    physics_weight: float = 0.1
    angle_weight: float = 1.0
    val_fraction: float = 0.2
    seed: int = 20260911
    limit: int | None = None


class _Batches:
    """Tensors on the CPU, sliced into shuffled minibatches."""

    def __init__(self, data: Dataset, batch_size: int, seed: int) -> None:
        self.bus_x = torch.from_numpy(data.bus_x)
        self.branch_x = torch.from_numpy(data.branch_x)
        self.edge_index = torch.from_numpy(data.edge_index)
        self.vm = torch.from_numpy(data.vm)
        self.va = torch.from_numpy(data.va)
        self.s_specified = torch.from_numpy(data.s_specified)
        self.shunt = torch.from_numpy(data.shunt)
        self.p_mask, self.q_mask = bus_masks(self.bus_x)
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.vm)

    def epoch(self, shuffle: bool = True):
        order = np.arange(len(self))
        if shuffle:
            self.rng.shuffle(order)
        for start in range(0, len(order), self.batch_size):
            idx = torch.from_numpy(order[start : start + self.batch_size])
            yield (
                self.bus_x[idx],
                self.branch_x[idx],
                self.vm[idx],
                self.va[idx],
                self.s_specified[idx],
            )


def _scalar(t: Tensor) -> float:
    """float() on a grad-tracking tensor warns. Detach explicitly."""
    return float(t.detach())


def _losses(
    model: PowerFlowGNN,
    batch,
    batches: _Batches,
    config: TrainConfig,
    slack_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    bus_x, branch_x, vm_true, va_true, s_spec = batch
    vm, va = model(bus_x, branch_x, batches.edge_index, slack_mask=slack_mask)

    voltage = torch.nn.functional.mse_loss(vm, vm_true) + config.angle_weight * (
        torch.nn.functional.mse_loss(va, va_true)
    )
    physics = power_mismatch_loss(
        vm, va, branch_x, batches.edge_index, batches.shunt,
        s_spec, batches.p_mask, batches.q_mask,
    )
    return voltage + config.physics_weight * physics, voltage, physics


@torch.no_grad()
def evaluate(model, batches: _Batches, config: TrainConfig, slack_mask) -> dict:
    model.eval()
    totals = np.zeros(3)
    n = 0
    for batch in batches.epoch(shuffle=False):
        total, voltage, physics = _losses(model, batch, batches, config, slack_mask)
        size = len(batch[0])
        totals += np.array([_scalar(total), _scalar(voltage), _scalar(physics)]) * size
        n += size
    model.train()
    return dict(zip(("total", "voltage", "physics"), (totals / max(n, 1)).tolist()))


def train(config: TrainConfig, data_path: Path, out_dir: Path | None = None):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    data = load_dataset(data_path)
    train_data, val_data = split_by_operating_point(data, config.val_fraction, config.seed)

    if config.limit:
        train_data = _head(train_data, config.limit)

    train_batches = _Batches(train_data, config.batch_size, config.seed)
    val_batches = _Batches(val_data, config.batch_size, config.seed)

    slack_mask = train_batches.bus_x[0, :, 4].clone()

    model = PowerFlowGNN(hidden=config.hidden, rounds=config.rounds)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: list[dict] = []
    started = time.perf_counter()

    for epoch in range(config.epochs):
        running = np.zeros(3)
        seen = 0
        for batch in train_batches.epoch():
            optimiser.zero_grad()
            total, voltage, physics = _losses(model, batch, train_batches, config, slack_mask)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()

            size = len(batch[0])
            running += np.array([_scalar(total), _scalar(voltage), _scalar(physics)]) * size
            seen += size

        entry = {
            "epoch": epoch,
            "train": dict(zip(("total", "voltage", "physics"), (running / seen).tolist())),
            "val": evaluate(model, val_batches, config, slack_mask),
            "seconds": time.perf_counter() - started,
        }
        history.append(entry)
        print(
            f"epoch {epoch:>3}  train {entry['train']['total']:.6f}"
            f"  (v {entry['train']['voltage']:.6f}  p {entry['train']['physics']:.6f})"
            f"  val {entry['val']['total']:.6f}  {entry['seconds'] / 60:.1f} min",
            flush=True,
        )

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "weights.pt")
        (out_dir / "history.json").write_text(
            json.dumps(
                {
                    "config": asdict(config),
                    "parameters": count_parameters(model),
                    "excluded": data.excluded,
                    "n_train": len(train_data),
                    "n_val": len(val_data),
                    "history": history,
                },
                indent=1,
            ),
            encoding="utf-8",
        )

    return model, history


def _head(data: Dataset, n: int) -> Dataset:
    mask = np.zeros(len(data), dtype=bool)
    mask[:n] = True
    from p013.dataset import _subset

    return _subset(data, mask)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the P-013 model")
    ap.add_argument("--data", type=Path, default=Path("data/train.npz"))
    ap.add_argument("--out", type=Path, default=Path("model"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--physics-weight", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None)
    # The seed was reachable in TrainConfig but not from the command line, so
    # every run was the same draw. The physics ablation is a claim about the
    # model rather than about one initialisation, and it cannot be repeated at
    # another seed without this. It seeds the split as well as the weights, so
    # two runs compared against each other must pass the same value.
    ap.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = ap.parse_args()

    config = TrainConfig(
        hidden=args.hidden,
        rounds=args.rounds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        physics_weight=args.physics_weight,
        limit=args.limit,
        seed=args.seed,
    )
    print(f"config: {asdict(config)}", flush=True)
    train(config, args.data, args.out)


if __name__ == "__main__":
    main()
