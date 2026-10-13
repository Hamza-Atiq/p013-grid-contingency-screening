"""Train the same model on the score the ranking uses. Research round 1, task C.

``loss = voltage error + 0.1 * power mismatch            (exactly p013.train)
       + score_weight  * score_loss(score(V_hat), score(V_true))
       + excess_weight * excess_loss(V_hat, V_true)``

The first line is unchanged, term for term, so this is the main model plus two
terms and nothing else. See ``EXP_C_NOTES.md`` for why these two and not a
ranking loss or new data.

**Same architecture, same everything downstream.** The model is
:class:`p013.graph.PowerFlowGNN` at the same width and depth, saved as the same
``weights.pt`` state dict, so ``p013.export_weights`` -> ``forward_numpy`` ->
``prediction_cache.build`` run on it with no edits.

**Same draw as the main model.** The seed, the split, the batch order and the
initial weights all follow :func:`p013.train.train` step for step: seeds first,
then the data, the split, the batches, then the model. Nothing added here
consumes torch's or NumPy's random state. So with both new weights at zero this
run is the main model, and a comparison of the two measures the new terms, not
a different initialisation. ``tests/test_score_loss.py`` checks the loss half
of that claim against ``p013.train._losses`` directly.

**Where V_before comes from.** Each batch gets a second forward pass with every
branch set in service, under ``no_grad``: the model's own intact prediction for
that operating point. That is what the evaluation uses too. It costs one
forward pass without a backward, and it is skipped when ``score_weight`` is 0.

Run from ``build/``::

    .venv/Scripts/python.exe experiments/train_score.py --out model_score
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

import p013
from p013.case_loader import parse_matpower
from p013.dataset import load_dataset, split_by_operating_point
from p013.graph import PowerFlowGNN, count_parameters
from p013.physics import power_mismatch_loss
from p013.score_loss import Ratings, excess_loss, risk_score, score_loss
from p013.train import TrainConfig, _Batches, _head, _scalar

#: Loss terms, in the order they are recorded. The first three mean what they
#: mean in p013.train's history, so the two histories can be read side by side.
TERMS = ("total", "voltage", "physics", "score", "excess")


@dataclass
class ScoreTrainConfig(TrainConfig):
    """:class:`p013.train.TrainConfig` plus the new terms. Defaults: EXP_C_NOTES §4."""

    score_weight: float = 0.002
    excess_weight: float = 0.01
    focus: float = 0.7
    boost: float = 4.0
    under_weight: float = 2.0
    cap: float = 2.0
    tau: float = 0.02
    excess_tau: float = 0.05


def intact_reference(model, bus_x, branch_x, edge_index, slack_mask) -> Tensor:
    """The model's own V_before: same inputs, every branch in service, no gradient.

    No gradient on purpose. The intact rows in the training set get their own
    voltage loss; letting the score term also drag the intact prediction around
    would let the model lower a score by *raising* excess before the outage
    instead of by getting the post-outage voltage right.
    """
    intact = branch_x.clone()
    intact[..., -1] = 1.0
    with torch.no_grad():
        vm_before, _ = model(bus_x, intact, edge_index, slack_mask=slack_mask)
    return vm_before


def score_losses(
    model: PowerFlowGNN,
    batch,
    batches: _Batches,
    config: ScoreTrainConfig,
    slack_mask: Tensor,
    ratings: Ratings,
) -> tuple[Tensor, ...]:
    """``(total, voltage, physics, score, excess)`` for one batch.

    The voltage and physics lines are :func:`p013.train._losses` verbatim. They
    are repeated rather than called because that function hides the predicted
    voltages the new terms need; a test holds the two equal.
    """
    bus_x, branch_x, vm_true, va_true, s_spec = batch
    vm, va = model(bus_x, branch_x, batches.edge_index, slack_mask=slack_mask)

    voltage = torch.nn.functional.mse_loss(vm, vm_true) + config.angle_weight * (
        torch.nn.functional.mse_loss(va, va_true)
    )
    physics = power_mismatch_loss(
        vm, va, branch_x, batches.edge_index, batches.shunt,
        s_spec, batches.p_mask, batches.q_mask,
    )
    total = voltage + config.physics_weight * physics

    zero = vm.new_zeros(())
    score = zero
    if config.score_weight > 0:
        vm_before = intact_reference(model, bus_x, branch_x, batches.edge_index, slack_mask)
        args = (vm_before, branch_x, batches.edge_index, ratings, config.tau, config.excess_tau)
        predicted = risk_score(vm, va, *args)
        with torch.no_grad():
            target = risk_score(vm_true, va_true, *args)
        score = score_loss(
            predicted, target, config.focus, config.boost, config.under_weight, config.cap
        )
        total = total + config.score_weight * score

    excess = zero
    if config.excess_weight > 0:
        excess = excess_loss(vm, vm_true, config.excess_tau)
        total = total + config.excess_weight * excess

    return total, voltage, physics, score, excess


def _row(values: Tensor | tuple) -> np.ndarray:
    return np.array([_scalar(v) for v in values])


@torch.no_grad()
def evaluate(model, batches: _Batches, config, slack_mask, ratings) -> dict:
    model.eval()
    totals = np.zeros(len(TERMS))
    n = 0
    for batch in batches.epoch(shuffle=False):
        size = len(batch[0])
        totals += _row(score_losses(model, batch, batches, config, slack_mask, ratings)) * size
        n += size
    model.train()
    return dict(zip(TERMS, (totals / max(n, 1)).tolist()))


def train(config: ScoreTrainConfig, data_path: Path, out_dir: Path | None = None):
    """:func:`p013.train.train`, step for step, with the two extra terms."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    data = load_dataset(data_path)
    train_data, val_data = split_by_operating_point(data, config.val_fraction, config.seed)

    if config.limit:
        train_data = _head(train_data, config.limit)

    train_batches = _Batches(train_data, config.batch_size, config.seed)
    val_batches = _Batches(val_data, config.batch_size, config.seed)

    slack_mask = train_batches.bus_x[0, :, 4].clone()
    ratings = Ratings.from_ppc(parse_matpower(p013.CASE_PATH))

    model = PowerFlowGNN(hidden=config.hidden, rounds=config.rounds)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: list[dict] = []
    started = time.perf_counter()

    for epoch in range(config.epochs):
        running = np.zeros(len(TERMS))
        seen = 0
        epoch_started = time.perf_counter()
        for batch in train_batches.epoch():
            optimiser.zero_grad()
            terms = score_losses(model, batch, train_batches, config, slack_mask, ratings)
            terms[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()

            size = len(batch[0])
            running += _row(terms) * size
            seen += size
        train_seconds = time.perf_counter() - epoch_started

        entry = {
            "epoch": epoch,
            "train": dict(zip(TERMS, (running / seen).tolist())),
            "val": evaluate(model, val_batches, config, slack_mask, ratings),
            "seconds": time.perf_counter() - started,
            # Training pass only, without validation, so the cost of the new
            # terms can be compared per training row against the main model.
            "train_seconds": train_seconds,
        }
        history.append(entry)
        t, v = entry["train"], entry["val"]
        print(
            f"epoch {epoch:>3}  train {t['total']:.6f}"
            f"  (v {t['voltage']:.6f}  p {t['physics']:.6f}"
            f"  s {t['score']:.5f}  e {t['excess']:.5f})"
            f"  val {v['total']:.6f} (s {v['score']:.5f})"
            f"  {entry['seconds'] / 60:.1f} min",
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the P-013 model on the ranking score")
    ap.add_argument("--data", type=Path, default=Path("data/train.npz"))
    ap.add_argument("--out", type=Path, default=Path("model_score"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--physics-weight", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=TrainConfig.seed)
    d = ScoreTrainConfig
    ap.add_argument("--score-weight", type=float, default=d.score_weight)
    ap.add_argument("--excess-weight", type=float, default=d.excess_weight)
    ap.add_argument("--focus", type=float, default=d.focus,
                    help="true score from which a row counts 1 + boost times")
    ap.add_argument("--boost", type=float, default=d.boost)
    ap.add_argument("--under-weight", type=float, default=d.under_weight,
                    help="extra weight on rows the model under-scores")
    ap.add_argument("--cap", type=float, default=d.cap, help="scores are capped here; 0 = no cap")
    ap.add_argument("--tau", type=float, default=d.tau, help="smooth-max temperature; 0 = hard max")
    ap.add_argument("--excess-tau", type=float, default=d.excess_tau,
                    help="softplus width at the band edge; 0 = exact excess")
    args = ap.parse_args()

    config = ScoreTrainConfig(
        hidden=args.hidden,
        rounds=args.rounds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        physics_weight=args.physics_weight,
        limit=args.limit,
        seed=args.seed,
        score_weight=args.score_weight,
        excess_weight=args.excess_weight,
        focus=args.focus,
        boost=args.boost,
        under_weight=args.under_weight,
        cap=args.cap,
        tau=args.tau,
        excess_tau=args.excess_tau,
    )
    print(f"config: {asdict(config)}", flush=True)
    train(config, args.data, args.out)


if __name__ == "__main__":
    main()
