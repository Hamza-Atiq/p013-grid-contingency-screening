"""Correct the model's voltages before scoring, and pick a score cutoff on validation.

Research round 1, task A (plans/2026-09-17-research-round-system-design.md).

**Why the correction is on voltages, not on the score.** Measured on the
validation split of ``eval/cache/predictions.npz`` (all numbers in
``experiments/results/exp_a.json``):

- Every dangerous case the model under-scores is a *voltage* case. None is
  thermal.
- At the bus where the truth's voltage term peaks, the model's post-outage
  voltage is too **high**, and on the under-scored cases the error has the same
  sign every time. So the model's deterioration comes out too small.
- The model's own intact state (``vm_before``) is a small part of it. Swapping in
  the true intact state barely moves the term; swapping in the true post-outage
  voltage restores almost all of it.
- The error is concentrated in a handful of contingencies (L35, L48, L46, L57,
  L27, ...), and for those the model's term still tracks the truth closely
  (correlation 0.98 for L35). A *systematic, per-contingency, per-bus* offset,
  in other words, not noise.

A correction on the **score** cannot fix that well. The score has a dead zone:
``excess`` is exactly zero inside 0.95-1.05 pu, so a bus the model puts at
0.951 pu contributes nothing, however far below it the truth is. Measured: the
model gives L27 a voltage term of 0.00 where the truth is 1.05. Nothing applied
after the max can recover information the dead zone has already thrown away.
Correcting the voltages first and then re-running the same ``excess`` and max
keeps the danger definition untouched.

**Why isotonic regression of the score is not here.** It is monotone, so a
cutoff on the isotonic score selects exactly the same cases as some cutoff on
the raw score. For a cutoff policy it changes nothing.

**Three corrections, few parameters each, all fitted per (row, bus) cell** from
validation rows only:

- ``bias``: ``V_corrected = V_model - mean(V_model - V_true)``.
- ``affine``: ``V_corrected = a + b * V_model``, least squares, with a guard for
  cells whose model voltage never moves (generator buses).
- ``load_bias``: the bias as a straight line in total system load, because the
  size of the error was measured to grow or shrink with load for the worst
  contingencies.

Each cell is fitted from about 70 validation points, so each parameter is an
average of many numbers. That is different from fitting 80 per-contingency
*cutoffs*, which would be one number chosen from a handful of dangerous cases,
and none at all for contingencies never dangerous on validation.

**Choosing the cutoff honestly.** A threshold picked on scores the calibration
was fitted on is optimistic. So validation is split into folds: each fold is
scored by a calibration fitted on the other folds (``cross_fitted_scores``),
and the cutoff is chosen on those out-of-fold scores. The final calibration for
test is then fitted on all of validation.

**The margin is also measured on validation**, not picked. ``cv_shortfall``
asks, fold by fold: if the cutoff had been chosen without this fold, how far
below it would this fold's lowest dangerous score have fallen? The largest such
shortfall is the margin. It is an estimate of how much a cutoff chosen on about
60 points gets undercut by about 15 new ones, and it is still an estimate from
75 points.

Only ``model_*``, ``pinned`` and the inputs are used at run time. Truth arrays
are read only inside the fit functions, and only at the rows passed in.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from p013.danger import excess

KINDS = ("bias", "affine", "load_bias")

#: A cell whose model voltage varies less than this (sum of squares, pu^2) over
#: the fitted points gets slope 1: a fixed-voltage generator bus has nothing to
#: regress on.
_FLAT_SXX = 1e-9


def usable_rows(pinned: np.ndarray) -> np.ndarray:
    """``[P, 81]`` True where a row has a real solution to learn from.

    Row 0 (the intact state) is always usable. Row ``c + 1`` is usable unless
    contingency ``c`` is pinned: an islanded or non-converged case has no true
    voltage to compare against.
    """
    pinned = np.asarray(pinned, dtype=bool)
    return np.concatenate([np.ones((pinned.shape[0], 1), dtype=bool), ~pinned], axis=1)


def _weighted_moments(x, y, w):
    """Per-cell weighted means and centred sums. ``w`` is 0/1, NaNs under w=0 ignored."""
    x = np.where(w, x, 0.0)
    y = np.where(w, y, 0.0)
    n = w.sum(axis=0)
    safe = np.maximum(n, 1)
    mx = x.sum(axis=0) / safe
    my = y.sum(axis=0) / safe
    sxx = (np.where(w, (x - mx) ** 2, 0.0)).sum(axis=0)
    sxy = (np.where(w, (x - mx) * (y - my), 0.0)).sum(axis=0)
    return n, mx, my, sxx, sxy


def fit(
    kind: str,
    model_vm: np.ndarray,
    truth_vm: np.ndarray,
    pinned: np.ndarray,
    total_load: np.ndarray,
    rows: np.ndarray,
) -> dict:
    """Fit one voltage correction on the operating points ``rows`` only.

    ``model_vm``/``truth_vm`` are ``[P, 81, 57]``, ``pinned`` is ``[P, 80]``,
    ``total_load`` is ``[P]``. Nothing outside ``rows`` is read.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown calibration {kind!r}; expected one of {KINDS}")
    rows = np.asarray(rows)
    if rows.dtype == bool:
        rows = np.flatnonzero(rows)
    if rows.size == 0:
        raise ValueError("cannot fit a calibration on zero points")

    mv = np.asarray(model_vm, dtype=float)[rows]
    tv = np.asarray(truth_vm, dtype=float)[rows]
    w = np.broadcast_to(usable_rows(np.asarray(pinned)[rows])[..., None], mv.shape)
    w = w & np.isfinite(mv) & np.isfinite(tv)
    n = w.sum(axis=0)

    if kind == "bias":
        err = np.where(w, mv - tv, 0.0)
        bias = np.where(n > 0, err.sum(axis=0) / np.maximum(n, 1), 0.0)
        return {"kind": kind, "bias": bias}

    if kind == "affine":
        n, mx, my, sxx, sxy = _weighted_moments(mv, tv, w)
        flat = sxx < _FLAT_SXX
        slope = np.where(flat, 1.0, sxy / np.where(flat, 1.0, sxx))
        intercept = np.where(n > 0, my - slope * mx, 0.0)
        slope = np.where(n > 0, slope, 1.0)
        return {"kind": kind, "slope": slope, "intercept": intercept}

    # load_bias: error = a + b * load, per cell
    load = np.asarray(total_load, dtype=float)[rows]
    x = np.broadcast_to(load[:, None, None], mv.shape)
    n, mx, my, sxx, sxy = _weighted_moments(x, mv - tv, w)
    flat = sxx < _FLAT_SXX
    b = np.where(flat, 0.0, sxy / np.where(flat, 1.0, sxx))
    a = np.where(n > 0, my - b * mx, 0.0)
    return {"kind": kind, "a": a, "b": np.where(n > 0, b, 0.0)}


def apply(cal: dict, model_vm: np.ndarray, total_load: np.ndarray) -> np.ndarray:
    """Corrected voltages, ``[P, 81, 57]``. Uses runtime-available inputs only."""
    mv = np.asarray(model_vm, dtype=float)
    kind = cal["kind"]
    if kind == "bias":
        return mv - cal["bias"]
    if kind == "affine":
        return cal["intercept"] + cal["slope"] * mv
    if kind == "load_bias":
        load = np.asarray(total_load, dtype=float)[:, None, None]
        return mv - (cal["a"] + cal["b"] * load)
    raise ValueError(f"unknown calibration {kind!r}")


def risk_scores(vm: np.ndarray, thermal: np.ndarray, pinned: np.ndarray) -> np.ndarray:
    """``risk_score`` for every case at once, from ``[P, 81, 57]`` voltages.

    The same formula as ``p013.danger.risk_score``: the voltage term is the max
    over buses of ``excess(V) - excess(V_before)`` with row 0 as ``V_before``,
    and the score is its max with the thermal term. Thermal is passed in
    unchanged: the diagnosis found no thermal under-scoring to correct.
    Pinned cases are NaN, as in the cache.
    """
    vm = np.asarray(vm, dtype=float)
    voltage = (excess(vm[:, 1:, :]) - excess(vm[:, :1, :])).max(axis=2)
    score = np.maximum(np.asarray(thermal, dtype=float), voltage)
    return np.where(np.asarray(pinned, dtype=bool), np.nan, score)


def calibrated_scores(cal: dict, data: dict) -> np.ndarray:
    """Scores ``[P, 80]`` for every point, from a fitted calibration."""
    vm = apply(cal, data["model_vm"], total_load(data))
    return risk_scores(vm, data["model_thermal"], data["pinned"])


def total_load(data: dict) -> np.ndarray:
    """Total active load per point, MW. An input, so runtime-available."""
    return np.asarray(data["load_p_mw"], dtype=float).sum(axis=1)


def fit_on(kind: str, data: dict, rows) -> dict:
    """``fit`` with the arrays taken from a prediction cache dict."""
    return fit(
        kind, data["model_vm"], data["truth_vm"], data["pinned"], total_load(data), rows
    )


def make_folds(rows, n_folds: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Split operating-point indices into shuffled, near-equal folds."""
    rows = np.asarray(rows)
    if rows.dtype == bool:
        rows = np.flatnonzero(rows)
    perm = np.random.default_rng(seed).permutation(rows)
    return [np.sort(f) for f in np.array_split(perm, n_folds)]


def cross_fitted_scores(
    fitter: Callable[[np.ndarray], dict],
    scorer: Callable[[dict], np.ndarray],
    folds: list[np.ndarray],
    shape: tuple[int, int],
) -> np.ndarray:
    """Out-of-fold scores: each fold scored by a calibration fitted without it.

    Only the fold rows are filled; everything else is NaN.
    """
    out = np.full(shape, np.nan)
    for i, fold in enumerate(folds):
        train = np.concatenate([f for j, f in enumerate(folds) if j != i])
        out[fold] = scorer(fitter(train))[fold]
    return out


def zero_miss_threshold(score, dangerous, pinned, rows=None) -> float:
    """The loosest cutoff that verifies every dangerous, non-pinned case in ``rows``.

    That is the lowest score given to any such case. ``+inf`` when there is none,
    meaning no cutoff is forced by these rows.
    """
    score = np.asarray(score, dtype=float)
    need = np.asarray(dangerous, dtype=bool) & ~np.asarray(pinned, dtype=bool)
    if rows is not None:
        rows = np.asarray(rows)
        score, need = score[rows], need[rows]
    vals = score[need]
    if vals.size == 0:
        return float("inf")
    if np.isnan(vals).any():
        raise ValueError("a dangerous non-pinned case has no score")
    return float(vals.min())


def cv_shortfall(score, dangerous, pinned, folds) -> dict:
    """How far each fold's lowest dangerous score falls below the other folds' cutoff.

    Returns every per-fold shortfall and their max, which is the margin.
    """
    shortfalls = []
    for i, fold in enumerate(folds):
        rest = np.concatenate([f for j, f in enumerate(folds) if j != i])
        c_rest = zero_miss_threshold(score, dangerous, pinned, rest)
        c_fold = zero_miss_threshold(score, dangerous, pinned, fold)
        shortfalls.append(max(0.0, c_rest - c_fold) if np.isfinite(c_rest) else 0.0)
    return {"per_fold": [float(s) for s in shortfalls], "margin": float(max(shortfalls))}
