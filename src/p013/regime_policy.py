"""Screening policies P1, P2 and P3: one cutoff, or a split by total load.

Research round 2 (plans/2026-09-17-research-round-2-design.md, section 5).
Built on task A's functions in ``p013.calibration``; nothing there is
re-implemented.

**The three policies.** A day is *heavy* when its total load is at least ``L``.

| Policy | Heavy day | Normal day |
| --- | --- | --- |
| P1 | cutoff C (task A's recipe, unchanged) | same cutoff C |
| P2 | verify all 80 | cutoff C_n, chosen on normal practice days only |
| P3 | cutoff C_h, chosen on heavy practice days only | cutoff C_n, chosen on normal practice days only |

**What is fitted, and on what.** Everything below uses the *practice* rows of
one split and nothing else:

1. **Bias calibration** (``calibration.fit_on("bias", ...)``), fitted once on
   all practice rows. It is shared by P1, P2, P3 and every ``L``.
2. **Out-of-fold scores.** Practice is cut into 5 folds
   (``calibration.make_folds``, seed 0). Each fold is scored by a bias
   calibration fitted on the other 4 folds (``calibration.cross_fitted_scores``).
   These scores are out-of-fold *for the calibration*.
3. **A cutoff for a set of practice rows** (all of them for P1; one regime for
   P2/P3) is ``zero_miss_threshold(out-of-fold scores, rows)`` minus
   ``cv_shortfall(out-of-fold scores, folds cut down to those rows).margin``.
   The folds are the same 5 folds, restricted to the regime's rows, so every
   score a fold sees still comes from a calibration that did not see it.
4. **Out-of-fold mean verified on practice**: the policy's cutoffs applied to
   the out-of-fold scores of every practice row, and the mean number of cases
   verified per day (pinned cases always count). This is how task A computed
   ``validation_out_of_fold``. Note what is *not* out-of-fold here: the cutoff
   itself was chosen on those same out-of-fold scores, so this number is
   in-sample with respect to the cutoff (it is optimistic about speed, and it
   knows nothing about misses - a cutoff chosen on practice never misses on
   practice).
5. **Choosing L** (P2 and P3, separately): the 50th, 60th, 70th and 80th
   percentile of total load over the practice days. The ``L`` kept is the one
   with the lowest out-of-fold mean verified. A tie keeps the lower percentile.
6. **The primary pipeline** is the one of P1/P2/P3 with the lowest out-of-fold
   mean verified. A tie keeps the earlier name in the order P1, P2, P3.

The final decision on a new day uses the calibration fitted on all practice
rows (step 1), the day's total load, ``model_*`` and ``pinned`` only. Truth
arrays (``truth_vm``, ``dangerous``) are read only at practice rows, inside the
fit.

**A regime with no dangerous case on practice.** ``zero_miss_threshold``
returns ``+inf`` when the rows hold no dangerous, non-pinned case, and
``cv_shortfall`` then gives margin 0, so the cutoff is ``+inf``. A cutoff of
``+inf`` selects **no case at all**: on that regime the policy verifies only the
pinned cases and trusts the model on everything else. That is what the recipe
literally implies - practice showed nothing to catch - but it is a bet on an
empty sample. Every fitted policy reports ``infinite_cutoffs`` so the harness
can count how often this happens. It should never happen with hundreds of
practice days; if it does, it is a risk to report, not a result.

**A structural fact about the selection rule.** For the same ``L``, P3's
out-of-fold mean verified is never above P2's: normal days get the identical
cutoff, and a heavy-day cutoff verifies at most the 80 cases P2 verifies. So
the rule "lowest out-of-fold mean verified" can pick P2 only on an exact tie.
The rule measures cost on practice, not the risk of a miss on new days.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from p013 import calibration as cal
from p013.screening_metrics import cutoff

N_FOLDS = 5
FOLD_SEED = 0
PERCENTILES = (50, 60, 70, 80)
POLICIES = ("P1", "P2", "P3")
CALIBRATION = "bias"


# --------------------------------------------------------------------------
# scores
# --------------------------------------------------------------------------
def _subset(data: dict, rows: np.ndarray) -> dict:
    """Only the runtime-available arrays ``calibrated_scores`` reads, at ``rows``."""
    return {k: data[k][rows] for k in ("model_vm", "model_thermal", "pinned", "load_p_mw")}


def scores_at(fitted: dict, data: dict, rows) -> np.ndarray:
    """Calibrated scores ``[P, C]``: filled at ``rows``, NaN elsewhere.

    Reads ``model_vm``, ``model_thermal``, ``pinned``, ``load_p_mw`` only.
    Scoring only the rows needed keeps a 50-split run short on 750 points.
    """
    rows = np.asarray(rows)
    out = np.full(data["pinned"].shape, np.nan)
    out[rows] = cal.calibrated_scores(fitted, _subset(data, rows))
    return out


@dataclass
class PracticeScores:
    """Everything fitted on the practice rows before any policy is chosen."""

    practice: np.ndarray            # sorted practice indices
    folds: list[np.ndarray]         # 5 folds of ``practice``
    calibration: dict               # bias, fitted on all practice rows
    oof: np.ndarray                 # [P, C] out-of-fold scores, NaN outside practice


def practice_scores(data: dict, practice, n_folds: int = N_FOLDS, seed: int = FOLD_SEED) -> PracticeScores:
    """Fit the shared bias calibration and the out-of-fold scores on ``practice`` only.

    ``truth_vm`` is read at practice rows only (inside ``calibration.fit``).
    """
    practice = np.sort(np.asarray(practice))
    folds = cal.make_folds(practice, n_folds, seed)
    fitter = lambda rows: cal.fit_on(CALIBRATION, data, rows)  # noqa: E731
    scorer = lambda fitted: scores_at(fitted, data, practice)  # noqa: E731
    oof = cal.cross_fitted_scores(fitter, scorer, folds, data["pinned"].shape)
    return PracticeScores(practice, folds, fitter(practice), oof)


# --------------------------------------------------------------------------
# regimes and cutoffs
# --------------------------------------------------------------------------
def heavy_days(total_load: np.ndarray, L: float) -> np.ndarray:
    """True where a day is heavy: total load at least ``L``. Runtime-available."""
    return np.asarray(total_load, dtype=float) >= L


def load_grid(total_load: np.ndarray, practice) -> dict[int, float]:
    """``{percentile: L}`` over the practice days' total load only."""
    load = np.asarray(total_load, dtype=float)[np.asarray(practice)]
    return {q: float(np.percentile(load, q)) for q in PERCENTILES}


def regime_cutoff(oof, dangerous, pinned, rows, folds) -> dict:
    """Task A's cutoff on a subset of practice rows.

    ``zero_miss_threshold`` on ``rows`` minus the ``cv_shortfall`` margin over
    the same folds restricted to ``rows``. ``+inf`` when ``rows`` hold no
    dangerous non-pinned case (see the module docstring).
    """
    rows = np.asarray(rows)
    zero_miss = cal.zero_miss_threshold(oof, dangerous, pinned, rows)
    sub_folds = [np.intersect1d(f, rows) for f in folds]
    margin = cal.cv_shortfall(oof, dangerous, pinned, sub_folds)["margin"] if rows.size else 0.0
    return {
        "zero_miss": zero_miss,
        "margin": margin,
        "cutoff": zero_miss - margin,
        "n_days": int(rows.size),
        "n_dangerous_cases": int((np.asarray(dangerous, bool) & ~np.asarray(pinned, bool))[rows].sum()),
    }


@dataclass
class FittedPolicy:
    """A policy ready to decide on new days.

    ``L`` is None for P1 (no regimes). For P2 ``cutoff_heavy`` is None, meaning
    verify all 80 on heavy days.
    """

    name: str
    calibration: dict
    cutoff_normal: float
    cutoff_heavy: float | None = None
    L: float | None = None
    percentile: int | None = None
    oof_mean_verified: float = float("nan")
    details: dict = field(default_factory=dict)

    @property
    def infinite_cutoffs(self) -> int:
        cuts = [self.cutoff_normal] + ([] if self.cutoff_heavy is None else [self.cutoff_heavy])
        return sum(1 for c in cuts if np.isposinf(c))

    def decide(self, score, total_load, pinned) -> np.ndarray:
        """``verify[day, case]`` from runtime-available inputs only."""
        return decide(self.name, score, total_load, pinned, self.cutoff_normal, self.cutoff_heavy, self.L)

    def verify(self, data: dict, rows) -> np.ndarray:
        """Decide for ``rows`` of a cache: final calibration, model outputs, total load."""
        rows = np.asarray(rows)
        score = cal.calibrated_scores(self.calibration, _subset(data, rows))
        return self.decide(score, cal.total_load(_subset(data, rows)), data["pinned"][rows])

    def summary(self) -> dict:
        return {
            "name": self.name,
            "L": self.L,
            "percentile": self.percentile,
            "cutoff_normal": self.cutoff_normal,
            "cutoff_heavy": self.cutoff_heavy,
            "heavy_rule": "verify all 80" if self.name == "P2" else ("same cutoff" if self.name == "P1" else "cutoff"),
            "oof_mean_verified": self.oof_mean_verified,
            "infinite_cutoffs": self.infinite_cutoffs,
            "details": self.details,
        }


def decide(name, score, total_load, pinned, cutoff_normal, cutoff_heavy=None, L=None) -> np.ndarray:
    """The decision rule of P1/P2/P3. Pinned cases are always verified."""
    score = np.asarray(score, dtype=float)
    pinned = np.asarray(pinned, dtype=bool)
    if name == "P1":
        v = cal_cutoff(score, cutoff_normal)
    else:
        heavy = heavy_days(total_load, L)[:, None]
        normal_v = cal_cutoff(score, cutoff_normal)
        if name == "P2":
            heavy_v = np.ones_like(pinned)
        elif name == "P3":
            heavy_v = cal_cutoff(score, cutoff_heavy)
        else:
            raise ValueError(f"unknown policy {name!r}")
        v = np.where(heavy, heavy_v, normal_v)
    return v | pinned


def cal_cutoff(score, threshold) -> np.ndarray:
    """``screening_metrics.cutoff``: verify score >= threshold; NaN never selected."""
    return cutoff(score, threshold)


def mean_verified(verify, pinned) -> float:
    """Mean cases verified per day, pinned always counted (as ``evaluate_policy``)."""
    return float((np.asarray(verify, bool) | np.asarray(pinned, bool)).sum(axis=1).mean())


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
def fit_p1(data: dict, ps: PracticeScores) -> FittedPolicy:
    """P1: task A's recipe on all practice rows."""
    D, P = data["dangerous"], data["pinned"]
    r = regime_cutoff(ps.oof, D, P, ps.practice, ps.folds)
    pr = ps.practice
    v = decide("P1", ps.oof[pr], None, P[pr], r["cutoff"])
    return FittedPolicy("P1", ps.calibration, r["cutoff"], oof_mean_verified=mean_verified(v, P[pr]),
                        details={"all": r})


def _fit_regimes_at(name: str, data: dict, ps: PracticeScores, L: float) -> FittedPolicy:
    D, P = data["dangerous"], data["pinned"]
    pr = ps.practice
    load = cal.total_load(data)
    heavy = heavy_days(load[pr], L)
    normal = regime_cutoff(ps.oof, D, P, pr[~heavy], ps.folds)
    details = {"normal": normal}
    c_heavy = None
    if name == "P3":
        h = regime_cutoff(ps.oof, D, P, pr[heavy], ps.folds)
        details["heavy"] = h
        c_heavy = h["cutoff"]
    else:
        details["heavy"] = {"n_days": int(heavy.sum()), "rule": "verify all 80"}
    v = decide(name, ps.oof[pr], load[pr], P[pr], normal["cutoff"], c_heavy, L)
    return FittedPolicy(name, ps.calibration, normal["cutoff"], c_heavy, L,
                        oof_mean_verified=mean_verified(v, P[pr]), details=details)


def fit_regime_policy(name: str, data: dict, ps: PracticeScores) -> FittedPolicy:
    """P2 or P3, with ``L`` chosen from the practice percentile grid."""
    if name not in ("P2", "P3"):
        raise ValueError(f"fit_regime_policy fits P2 or P3, not {name!r}")
    grid = load_grid(cal.total_load(data), ps.practice)
    best, tried = None, []
    for q, L in grid.items():  # ascending percentile; strict < keeps the lower one on ties
        pol = _fit_regimes_at(name, data, ps, L)
        pol.percentile = q
        tried.append({"percentile": q, "L": L, "oof_mean_verified": pol.oof_mean_verified,
                      "cutoff_normal": pol.cutoff_normal, "cutoff_heavy": pol.cutoff_heavy,
                      "infinite_cutoffs": pol.infinite_cutoffs})
        if best is None or pol.oof_mean_verified < best.oof_mean_verified:
            best = pol
    best.details["L_grid"] = tried
    return best


def fit_policy(name: str, data: dict, ps: PracticeScores) -> FittedPolicy:
    return fit_p1(data, ps) if name == "P1" else fit_regime_policy(name, data, ps)


def fit_all(data: dict, practice, n_folds: int = N_FOLDS, seed: int = FOLD_SEED) -> dict:
    """Fit P1, P2, P3 and select the primary, all on ``practice`` only.

    Returns ``{"policies": {name: FittedPolicy}, "primary": name, "scores": PracticeScores}``.
    """
    ps = practice_scores(data, practice, n_folds, seed)
    policies = {name: fit_policy(name, data, ps) for name in POLICIES}
    primary = select_primary(policies)
    return {"policies": policies, "primary": primary, "scores": ps}


def select_primary(policies: dict[str, FittedPolicy]) -> str:
    """Lowest out-of-fold practice mean verified; ties keep the order P1, P2, P3."""
    names = [n for n in POLICIES if n in policies]
    return min(names, key=lambda n: (policies[n].oof_mean_verified, names.index(n)))
