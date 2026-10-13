"""Can we tell, before any AC solve, which operating points the model will get wrong?

Research round 1, task D (plans/2026-09-17-research-round-system-design.md).
The owner's rule is "trust the robot when things are normal, check its homework
when something is unusual". This module holds the pieces that rule needs:

- **What "wrong" means** (``k_star``, ``bad_points``). Labels use the solver's
  truth. They are for measuring a detector and for fitting its threshold on the
  validation split, never for deciding what to verify on a new point.
- **Runtime signals** (``TrainingReference``, ``runtime_signals``). Every one of
  them is known before the first AC solve: the point's own inputs, the training
  set's inputs, and the model's own predictions.
- **Policies and how they are fitted** (``fit_flag_policy``, ``watchlist``).
  Every fitting function takes an explicit ``fit_mask`` and reads only those
  rows of anything that carries truth. ``tests/test_novelty.py`` holds them to
  that by corrupting the other rows and checking nothing changes.

**Why "larger is more unusual" everywhere.** A signal where low values are the
worrying end (lowest predicted voltage) is negated once, here, so every
threshold in the experiment means the same thing: flag when ``signal >= t``.

**Why the training reference loads only four keys.** ``data/train.npz`` also
holds the solver's voltages and flows for 20,000 rows. None of that is needed to
say how usual a point's *inputs* are, and loading it would cost memory on a
shared 16 GB laptop for nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from p013.screening_metrics import Timing, evaluate_policy, top_k

TRAIN_KEYS = ("load_p_mw", "load_q_mvar", "gen_p_mw", "in_service")


# ---------------------------------------------------------------------------
# Labels: what "the model got it wrong" means
# ---------------------------------------------------------------------------


def k_star(rank: np.ndarray, dangerous: np.ndarray, pinned: np.ndarray) -> np.ndarray:
    """Per point, the shortest top-k that catches every dangerous case.

    That is the deepest rank held by any dangerous case. Pinned cases count as
    dangerous, and the rankings already put them first, so a point whose only
    danger is pinned has ``k*`` equal to its pinned count. A point with nothing
    dangerous at all has ``k* = 0``.
    """
    rank = np.asarray(rank)
    hit = np.asarray(dangerous, dtype=bool) | np.asarray(pinned, dtype=bool)
    return np.where(hit, rank, 0).max(axis=1).astype(int)


def bad_points(model_k: np.ndarray, k_normal: int) -> np.ndarray:
    """A point is bad when trusting the model's top ``k_normal`` would miss something.

    This is the definition that matters to the screen: the policy verifies a
    fixed top-k on normal points, and exactly the points with ``model k* > k``
    are the ones that k would fail. It is deliberately about the model's list,
    not about the gap to a perfect ranker - a point that is hard for any ranker
    still has to be flagged.
    """
    return np.asarray(model_k) > int(k_normal)


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Probability a random bad point scores higher than a random good one. Ties count half."""
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------
# Runtime signals
# ---------------------------------------------------------------------------


def load_training_inputs(path: Path) -> dict:
    """Read only the input arrays of the training set. See the module docstring."""
    with np.load(path, allow_pickle=False) as raw:
        return {key: raw[key] for key in TRAIN_KEYS}


def _inputs(load_p, load_q, gen_p) -> np.ndarray:
    return np.hstack([np.asarray(a, dtype=float) for a in (load_p, load_q, gen_p)])


@dataclass
class TrainingReference:
    """How usual a point's inputs are, measured against the training set only.

    Features are the 90 raw inputs (42 load P, 42 load Q, 6 generator P),
    standardised by the training mean and standard deviation. Nothing from the
    sweep is used to fit this.
    """

    mean: np.ndarray
    std: np.ndarray
    inv_cov: np.ndarray
    train_z: np.ndarray
    train_total_load: np.ndarray

    @classmethod
    def fit(cls, load_p, load_q, gen_p) -> "TrainingReference":
        x = _inputs(load_p, load_q, gen_p)
        mean, std = x.mean(axis=0), x.std(axis=0)
        std[std == 0] = 1.0
        z = (x - mean) / std
        inv_cov = np.linalg.pinv(np.cov(z, rowvar=False))
        return cls(mean, std, inv_cov, z, np.asarray(load_p, dtype=float).sum(axis=1))

    def _z(self, load_p, load_q, gen_p) -> np.ndarray:
        return (_inputs(load_p, load_q, gen_p) - self.mean) / self.std

    def mahalanobis(self, load_p, load_q, gen_p) -> np.ndarray:
        z = self._z(load_p, load_q, gen_p)
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", z, self.inv_cov, z), 0.0))

    def knn_distance(self, load_p, load_q, gen_p, k: int = 10) -> np.ndarray:
        """Mean distance to the ``k`` nearest training points, in standardised units."""
        z = self._z(load_p, load_q, gen_p)
        d2 = (z**2).sum(1)[:, None] + (self.train_z**2).sum(1)[None, :] - 2.0 * z @ self.train_z.T
        nearest = np.partition(np.maximum(d2, 0.0), k - 1, axis=1)[:, :k]
        return np.sqrt(nearest).mean(axis=1)

    def load_percentile(self, load_p) -> np.ndarray:
        """Share of training points with less total load. Same order as total load."""
        total = np.asarray(load_p, dtype=float).sum(axis=1)
        return np.searchsorted(np.sort(self.train_total_load), total, side="left") / len(
            self.train_total_load
        )


# Band for "the model is unsure": scores in [low, 1.0). The danger threshold is
# 1.0, and the lowest score the model gave a truly dangerous case on validation
# is about 0.46, so 0.5 is where "near the threshold" starts to include the
# cases the model buries. Fixed here, not tuned.
NEAR_BAND_LOW = 0.5


def runtime_signals(cache: dict, reference: TrainingReference) -> dict[str, np.ndarray]:
    """Per-point signals, every one known before any AC solve. Larger = more unusual.

    Reads only ``load_*``, ``gen_p_mw``, ``model_*`` and ``pinned`` from the cache.
    """
    lp, lq, gp = cache["load_p_mw"], cache["load_q_mvar"], cache["gen_p_mw"]
    score = np.nan_to_num(np.asarray(cache["model_score"], dtype=float), nan=-np.inf)
    return {
        "total_load_mw": np.asarray(lp, dtype=float).sum(axis=1),
        "load_percentile_vs_train": reference.load_percentile(lp),
        "mahalanobis_vs_train": reference.mahalanobis(lp, lq, gp),
        "knn10_distance_vs_train": reference.knn_distance(lp, lq, gp, k=10),
        "intact_max_loading_pred": np.nanmax(cache["model_loading"][:, 0], axis=1).astype(float),
        "intact_min_vm_pred_negated": -np.nanmin(cache["model_vm"][:, 0], axis=1).astype(float),
        "n_scored_dangerous_pred": (score > 1.0).sum(axis=1).astype(float),
        "n_near_threshold_pred": ((score >= NEAR_BAND_LOW) & (score <= 1.0)).sum(axis=1).astype(float),
        "n_pinned": np.asarray(cache["pinned"]).sum(axis=1).astype(float),
    }


# ---------------------------------------------------------------------------
# Policies and their fitting
# ---------------------------------------------------------------------------


def flag_threshold(signal, bad, fit_mask, margin_sd: float = 0.0) -> float:
    """Lowest threshold that flags every bad fit row, lowered by a safety margin.

    The margin is in standard deviations of the signal over the fit rows, so it
    means the same thing for megawatts and for counts. With no bad fit rows the
    threshold is +inf: nothing is flagged.
    """
    signal = np.asarray(signal, dtype=float)
    fit = np.asarray(fit_mask, dtype=bool)
    bad_fit = np.asarray(bad, dtype=bool) & fit
    if not bad_fit.any():
        return float("inf")
    return float(signal[bad_fit].min() - margin_sd * signal[fit].std())


def watchlist(rank, dangerous, pinned, fit_mask, rows_mask, k_list: int) -> np.ndarray:
    """Contingencies that were ever dangerous and ranked below ``k_list`` on fit rows.

    ``rows_mask`` narrows the fit rows further (e.g. only flagged points). The
    result is one boolean per contingency, applied to every flagged point.
    """
    use = np.asarray(fit_mask, dtype=bool) & np.asarray(rows_mask, dtype=bool)
    danger = np.asarray(dangerous, dtype=bool) & ~np.asarray(pinned, dtype=bool)
    buried = danger & (np.asarray(rank) > int(k_list))
    return buried[use].any(axis=0)


def flagged_policy(rank, flagged, k_normal: int, mode: str = "all", extra=None) -> np.ndarray:
    """Build ``verify[point, contingency]``.

    Every point verifies its top ``k_normal``. A flagged point also verifies:
    ``all`` - every contingency; ``extra`` - the boolean ``extra`` array, either
    ``[contingencies]`` (a watchlist) or ``[points, contingencies]`` (a cutoff).
    """
    rank = np.asarray(rank)
    flagged = np.asarray(flagged, dtype=bool)
    verify = top_k(rank, k_normal)
    if mode == "all":
        verify = verify | flagged[:, None]
    elif mode == "extra":
        extra = np.asarray(extra, dtype=bool)
        if extra.ndim == 1:
            extra = np.broadcast_to(extra, rank.shape)
        verify = verify | (flagged[:, None] & extra)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return verify


def fit_flag_policy(
    signal,
    rank,
    dangerous,
    pinned,
    fit_mask,
    timing: Timing,
    k_grid=range(6, 41),
    margin_sd: float = 0.0,
    overhead_s: float | None = None,
) -> dict:
    """Pick ``k_normal`` and the flag threshold on fit rows only: flagged points verify all 80.

    For each ``k`` the threshold is the lowest that flags every fit point with
    ``model k* > k``, so the fit rows have zero misses by construction. The
    ``k`` with the best fit-row speedup wins.
    """
    fit = np.asarray(fit_mask, dtype=bool)
    rank, dangerous, pinned = np.asarray(rank), np.asarray(dangerous), np.asarray(pinned)
    mk = k_star(rank, dangerous, pinned)
    best = None
    for k in k_grid:
        t = flag_threshold(signal, bad_points(mk, k), fit, margin_sd)
        flagged = np.asarray(signal, dtype=float) >= t
        verify = flagged_policy(rank, flagged, k, "all")
        r = evaluate_policy(verify[fit], dangerous[fit], pinned[fit], timing, overhead_s)
        if r["points_with_miss"] == 0 and (best is None or r["speedup"] > best["fit_speedup"]):
            best = {
                "k_normal": int(k),
                "threshold": t,
                "fit_speedup": r["speedup"],
                "fit_flagged_share": float(flagged[fit].mean()),
            }
    return best


def min_dangerous_score(score, dangerous, pinned, rows_mask) -> float:
    """Lowest model score given to any truly dangerous, non-pinned case on the given rows."""
    score = np.asarray(score, dtype=float)
    hit = np.asarray(dangerous, dtype=bool) & ~np.asarray(pinned, dtype=bool)
    hit = hit & np.asarray(rows_mask, dtype=bool)[:, None]
    return float(score[hit].min()) if hit.any() else float("inf")
