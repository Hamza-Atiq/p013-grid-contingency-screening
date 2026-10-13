"""One definition of what a screening policy catches and what it costs.

Research round 1 (plans/2026-09-17-research-round-system-design.md). Every task
in that round reports through this module, so that "speedup" means the same
thing in all of them.

**A policy is a boolean array** ``verify[point, contingency]``: True means the
contingency is AC-solved. Anything not verified is trusted to be safe. That one
shape covers a fixed top-k, a score cutoff, and a rule that verifies a whole
operating point because it looks unusual.

The definitions are the ones ``p013.evaluate`` already uses, and
``tests/test_screening_metrics.py`` holds this module to EVAL_NOTES §4:

- **Pinned cases** (islanded or no solution) are always solved and always count
  as dangerous. They consume solver time; hiding that would flatter the policy.
  This is the one place this module departs from ``measure_k_star``: a top-k
  with ``k`` smaller than a point's pinned count there solved only ``k`` of
  them. Test points carry up to 6 pinned cases, so recall differs for
  ``k <= 5`` and is identical from ``k = 6`` upward - verified 2026-09-17
  against every value of ``results.json``'s recall curve. No real policy runs
  below ``k = 6``, and a real screen cannot skip an islanded case.
- **Recall** is micro-averaged: dangerous cases caught over dangerous cases in
  total, across all points. Identical to ``measure_k_star``'s recall curve.
- **Cost of one point** is ``overhead + verified x one AC solve``.
- **Speedup** is the full sweep's time over the policy's MEAN time per point.
  A fixed top-k costs the same at every point, so mean and median agree with
  the original measurement. An adaptive policy does not, and the mean is what
  an operator pays over a day.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Timing:
    """Measured costs, in seconds, from ``eval/results.json``."""

    solve_s: float
    overhead_s: float
    baseline_s: float
    n_contingencies: int

    @classmethod
    def from_results(cls, results: dict) -> "Timing":
        """Derive the per-solve and overhead costs the way EVAL_NOTES §4 does.

        The results file stores two medians: the full sweep, and the screen at
        ``k_deploy``. One solve is the sweep over N; the overhead is whatever
        the screen cost beyond its ``k_deploy`` solves.
        """
        t = results["timing"]
        n = int(results["n_contingencies"])
        baseline = float(t["baseline_full_ac_sweep_seconds_median"])
        solve = baseline / n
        overhead = float(t["ours_seconds_median"]) - int(t["k_deploy_solved"]) * solve
        return cls(solve_s=solve, overhead_s=overhead, baseline_s=baseline, n_contingencies=n)

    @property
    def overhead_in_solves(self) -> float:
        return self.overhead_s / self.solve_s


def evaluate_policy(
    verify: np.ndarray,
    dangerous: np.ndarray,
    pinned: np.ndarray,
    timing: Timing,
    overhead_s: float | None = None,
) -> dict:
    """Score a policy. All three arrays are ``[points, contingencies]`` booleans.

    ``overhead_s`` overrides the measured overhead, for a policy that is cheaper
    or dearer to run than the original screen (task B, or a detector that adds
    work of its own). Leaving it out uses the measured 0.626 s.
    """
    verify = np.asarray(verify, dtype=bool)
    dangerous = np.asarray(dangerous, dtype=bool)
    pinned = np.asarray(pinned, dtype=bool)
    if not (verify.shape == dangerous.shape == pinned.shape) or verify.ndim != 2:
        raise ValueError(
            f"verify {verify.shape}, dangerous {dangerous.shape} and pinned "
            f"{pinned.shape} must be the same 2-D shape"
        )

    # Pinned cases go to the solver whatever the policy says, and they are
    # dangerous by definition. Both facts are enforced here rather than trusted
    # to every caller.
    verify = verify | pinned
    dangerous = dangerous | pinned

    missed = dangerous & ~verify
    missed_per_point = missed.sum(axis=1)
    n_verified = verify.sum(axis=1)

    overhead = timing.overhead_s if overhead_s is None else float(overhead_s)
    seconds = overhead + n_verified * timing.solve_s
    total_dangerous = int(dangerous.sum())

    return {
        "n_points": int(verify.shape[0]),
        "recall": (
            1.0 - float(missed.sum()) / total_dangerous if total_dangerous else 1.0
        ),
        "missed_cases": int(missed.sum()),
        "points_with_miss": int((missed_per_point > 0).sum()),
        "mean_verified": float(n_verified.mean()),
        "median_verified": float(np.median(n_verified)),
        "max_verified": int(n_verified.max()),
        "overhead_s": overhead,
        "mean_seconds": float(seconds.mean()),
        "speedup": timing.baseline_s / float(seconds.mean()),
    }


def top_k(rank: np.ndarray, k: int) -> np.ndarray:
    """Verify the ``k`` highest-ranked cases at every point.

    ``rank`` is 1-based and puts pinned cases first, as ``p013.evaluate`` does.
    """
    return np.asarray(rank) <= k


def cutoff(score: np.ndarray, threshold: float) -> np.ndarray:
    """Verify every case scoring at or above ``threshold``.

    NaN scores are pinned cases; ``evaluate_policy`` verifies those regardless,
    so they are simply not selected here.
    """
    score = np.asarray(score, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(score, nan=-np.inf) >= threshold
