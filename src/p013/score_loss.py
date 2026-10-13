"""The ranking score, written in torch so it can be trained on. Research round 1, task C.

**Why this exists.** The model is trained on per-bus voltage error. The ranking
is derived afterwards, from :func:`p013.danger.risk_score`, which is a *max*
over 57 buses and 80 branches. Nothing ever told the model which bus matters.
Measured on the validation points of ``eval/cache/predictions.npz``: the 89
dangerous contingencies the model scored below 1.0 are **all** voltage-driven,
and at the bus where the true voltage term peaks the model is off by
**0.031 pu on average, always upward** (true voltage 0.85-0.90 pu) - four times
its average error of 0.0075 pu. A good average and a bad max, exactly as the
research plan (O4) suspected.

So this module gives training the score itself:

``score = max( max loading/100 , max over buses (excess(V) - excess(V_before)) )``

with each ``max`` optionally softened, and a loss that compares the score the
model's voltages imply against the score the true voltages imply.

**Both sides go through the same functions.** Prediction and target are both
computed here, with the same smoothing. A smoothed max is biased upward by at
most ``tau * log(n)``; applying it to both sides makes the comparison
like-with-like instead of charging the model for our own approximation. With
``tau = 0`` every function here is the exact NumPy definition, and
``tests/test_score_loss.py`` holds them to that.

**V_before.** The training set stores no intact state for its outage rows (every
sample is its own operating point). The score is therefore built against a
*reference* intact voltage the caller supplies; ``experiments/train_score.py``
uses the model's own intact prediction, which is also exactly what the
evaluation uses (``prediction_cache``: "model's own predicted intact state").
Measured on the cache's validation points, swapping the true intact state for
the model's changes **1 of 744** danger labels. See ``EXP_C_NOTES.md``.

**Thermal flows are a port, not a new definition.** :func:`loading_percent` is
:func:`p013.admittance.loading_percent` - apparent power at the *from* end,
against PGLib's MVA rating - rebuilt branch by branch on top of
:func:`p013.physics.branch_admittances`, so the physics term and this share one
admittance. The dataset's own ``loading_percent`` is pandapower's current-based
figure and is deliberately not used.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from p013.admittance import BR_RATE_A, to_canonical_order
from p013.danger import BAND_PU, DANGER_THRESHOLD, NOMINAL_PU, RATING_PERCENT
from p013.physics import branch_admittances


@dataclass(frozen=True)
class Ratings:
    """What the thermal term needs from the case, in canonical branch order."""

    rate_mva: Tensor  # (n_branch,)
    base_mva: float

    @classmethod
    def from_ppc(cls, ppc, dtype: torch.dtype = torch.float32) -> "Ratings":
        rate = to_canonical_order(ppc, ppc["branch"][:, BR_RATE_A]).astype(float)
        if not (rate > 0).all():
            # Same guard, same reason, as admittance.loading_percent.
            raise ValueError("a branch has no rating, so its loading is undefined")
        return cls(torch.tensor(rate, dtype=dtype), float(ppc["baseMVA"]))


def loading_percent(
    vm: Tensor, va: Tensor, branch_x: Tensor, edge_index: Tensor, ratings: Ratings
) -> Tensor:
    """Apparent power at the from end, percent of rating. ``(B, n_branch)``.

    Args:
        vm, va: ``(B, n_bus)``, magnitude in pu and angle in radians. Any common
            angle offset cancels, so slack-relative angles are fine.
        branch_x: ``(B, n_branch, 6)`` model features; the in-service flag in the
            last column zeroes a branch that is out, as in NumPy.
        edge_index: ``(2, n_branch)`` zero-based bus positions, canonical order.
    """
    src, dst = edge_index[0], edge_index[1]
    v = torch.polar(vm, va)
    y_ff, y_ft, _, _ = branch_admittances(branch_x)
    v_f = v.index_select(-1, src)
    v_t = v.index_select(-1, dst)
    s_from = v_f * (y_ff * v_f + y_ft * v_t).conj()
    rate = ratings.rate_mva.to(vm.dtype)
    return s_from.abs() * ratings.base_mva / rate * 100.0


def soft_excess(vm: Tensor, tau: float = 0.0) -> Tensor:
    """``excess(V) = max(0, |V - 1| / 0.05 - 1)``, band-widths outside the band.

    ``tau = 0`` is the exact definition from :mod:`p013.danger`. ``tau > 0``
    replaces the kink at the band edge with ``tau * softplus(x / tau)``, which
    lies above the exact value by at most ``tau * ln 2`` and gives a small,
    non-zero gradient to a prediction sitting just inside the band when the
    truth is outside it - the exact form gives such a prediction no signal.
    """
    x = (vm - NOMINAL_PU).abs() / BAND_PU - 1.0
    if tau <= 0:
        return torch.relu(x)
    return tau * torch.nn.functional.softplus(x / tau)


def smooth_max(x: Tensor, tau: float = 0.0, dim: int = -1) -> Tensor:
    """``tau * logsumexp(x / tau)``: a max whose gradient reaches the near-winners.

    Lies in ``[max, max + tau * ln n]`` and tends to ``max`` as ``tau -> 0``;
    ``tau = 0`` is the hard max. A hard max sends the whole gradient to one bus,
    and early in training that bus is often the wrong one.
    """
    if tau <= 0:
        return x.amax(dim=dim)
    return tau * torch.logsumexp(x / tau, dim=dim)


def score_terms(
    vm: Tensor,
    va: Tensor,
    vm_before: Tensor,
    branch_x: Tensor,
    edge_index: Tensor,
    ratings: Ratings,
    tau: float = 0.0,
    excess_tau: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """``(thermal, voltage)`` per row, each ``(B,)``. The two terms of ``risk_score``.

    With both taus zero, ``thermal`` is ``loading.max() / 100`` and ``voltage``
    is ``(excess(vm) - excess(vm_before)).max()``, exactly as in
    :func:`p013.danger.risk_score` - not clipped at zero, for the same reason.
    """
    loading = loading_percent(vm, va, branch_x, edge_index, ratings)
    thermal = smooth_max(loading / RATING_PERCENT, tau)
    deterioration = soft_excess(vm, excess_tau) - soft_excess(vm_before, excess_tau)
    voltage = smooth_max(deterioration, tau)
    return thermal, voltage


def risk_score(
    vm: Tensor,
    va: Tensor,
    vm_before: Tensor,
    branch_x: Tensor,
    edge_index: Tensor,
    ratings: Ratings,
    tau: float = 0.0,
    excess_tau: float = 0.0,
) -> Tensor:
    """``max(thermal, voltage)`` per row, ``(B,)``. Exact at ``tau = 0``."""
    thermal, voltage = score_terms(
        vm, va, vm_before, branch_x, edge_index, ratings, tau, excess_tau
    )
    return smooth_max(torch.stack([thermal, voltage], dim=-1), tau)


def score_loss(
    predicted: Tensor,
    target: Tensor,
    focus: float = 0.7,
    boost: float = 4.0,
    under_weight: float = 2.0,
    cap: float = 2.0,
) -> Tensor:
    """Squared score error, weighted toward the cases the ranking can lose.

    - Both scores are **capped** at ``cap`` first. Scores are unbounded - a
      three-branch outage can reach 7 - and a squared error on those would let
      a case at 6 predicted 5 outweigh a case at 1.05 predicted 0.6, although
      only the second changes the ranking. Measured on the trained model's
      training rows, dangerous rows are off by 0.70 on average; uncapped, the
      loss is spent on the far tail. A prediction above the cap with a target
      above it gets no gradient, which is correct: both are plainly dangerous.
    - A row whose **true** score is at least ``focus`` counts ``1 + boost``
      times. Those are the dangerous and nearly-dangerous cases; a safe case at
      0.3 scored 0.35 costs the ranking nothing.
    - A row the model **under-scores** counts ``under_weight`` times more. The
      measured failure is one-sided (voltages predicted too healthy, scores too
      low), and under-scoring a dangerous case is a miss while over-scoring a
      safe one is only a wasted AC solve.

    The weights depend on the target and on the sign of the error only, never
    on a gradient-carrying quantity, so they shape the loss without adding
    terms to its derivative.
    """
    target = target.detach()
    if cap > 0:
        predicted = predicted.clamp(max=cap)
        target = target.clamp(max=cap)
    error = predicted - target
    weight = 1.0 + boost * (target >= focus).to(error.dtype)
    weight = weight * torch.where(error < 0, under_weight, 1.0).to(error.dtype)
    return (weight * error.pow(2)).mean()


def excess_loss(vm_pred: Tensor, vm_true: Tensor, tau: float = 0.0) -> Tensor:
    """Mean squared error in ``excess``, over every bus of every row.

    Zero wherever both voltages are inside the band, so it only speaks where
    the voltage term of the score can. It needs no ``V_before`` at all:
    matching ``excess`` bus by bus matches the voltage term for *any* intact
    reference, which makes it the safety net under :func:`score_loss`.
    """
    return (soft_excess(vm_pred, tau) - soft_excess(vm_true.detach(), tau)).pow(2).mean()


__all__ = [
    "DANGER_THRESHOLD",
    "Ratings",
    "excess_loss",
    "loading_percent",
    "risk_score",
    "score_loss",
    "score_terms",
    "smooth_max",
    "soft_excess",
]

