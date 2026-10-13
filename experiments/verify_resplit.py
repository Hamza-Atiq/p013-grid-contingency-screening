"""Is a gate pass robust, or the luck of one validation/test split?

Research round 1, orchestrator's verification of task A. The official split has
75 validation points; a cutoff chosen as "the lowest dangerous score seen on 75
points" is expected to be undercut by some of 175 new ones. Task A said so
itself. This measures it: task A's exact recipe - calibration fitted on
validation, cutoff at the out-of-fold zero-miss score minus the cross-validated
shortfall margin - re-run on N random 75/175 re-splits of the same 250 points,
using task A's functions unchanged.

A policy you would promise "never misses" should have zero test misses on
nearly every split, not most of them.

    .venv/Scripts/python.exe experiments/verify_resplit.py
    .venv/Scripts/python.exe experiments/verify_resplit.py --cache eval/cache/score_model.npz --out experiments/results/verify_resplit_score_model.json

Costs are the clean-timing ones (experiments/results/exp_b_profile.json),
reported for both the original and the fast screen overhead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from p013 import calibration as cal
from p013 import prediction_cache
from p013.screening_metrics import Timing, cutoff, evaluate_policy

GATE_SPEEDUP = 1.5


def clean_timings() -> dict[str, Timing]:
    """Directly measured costs from the idle-machine run, one per screen."""
    b = json.loads(Path("experiments/results/exp_b_profile.json").read_text(encoding="utf-8"))
    ct = b["clean_timing"]
    solve = ct["cross_check_solve_s_measured_now"]["median_s"]
    return {
        name: Timing(solve_s=solve, overhead_s=ct[key]["median_s"], baseline_s=80 * solve, n_contingencies=80)
        for name, key in (("original_screen", "original_overhead_s"), ("fast_screen", "fast_overhead_s"))
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=prediction_cache.DEFAULT_CACHE)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", type=Path, default=Path("experiments/results/verify_resplit.json"))
    args = ap.parse_args()

    c = prediction_cache.load(args.cache)
    timings = clean_timings()
    D, P = c["dangerous"], c["pinned"]
    shape = D.shape
    n = shape[0]
    rng = np.random.default_rng(args.seed)
    raw = c["model_score"].astype(float)

    runs = {k: [] for k in ("raw", "bias")}
    for _ in range(args.splits):
        perm = rng.permutation(n)
        val_idx, test_idx = np.sort(perm[:75]), np.sort(perm[75:])
        folds = cal.make_folds(val_idx, 5, 0)
        fitter = lambda rows: cal.fit_on("bias", c, rows)  # noqa: E731
        scorer = lambda fitted: cal.calibrated_scores(fitted, c)  # noqa: E731
        candidates = {
            "raw": (raw, raw),
            "bias": (cal.cross_fitted_scores(fitter, scorer, folds, shape), scorer(fitter(val_idx))),
        }
        for name, (oof, final) in candidates.items():
            thr = cal.zero_miss_threshold(oof, D, P, val_idx) - cal.cv_shortfall(oof, D, P, folds)["margin"]
            verify = cutoff(final[test_idx], thr)
            row = {"threshold": thr}
            for tname, t in timings.items():
                r = evaluate_policy(verify, D[test_idx], P[test_idx], t)
                row[tname] = {k: r[k] for k in ("points_with_miss", "missed_cases", "recall", "mean_verified", "speedup")}
            runs[name].append(row)

    summary = {}
    for name, rows in runs.items():
        misses = np.array([r["fast_screen"]["points_with_miss"] for r in rows])
        cases = np.array([r["fast_screen"]["missed_cases"] for r in rows])
        entry = {
            "splits": args.splits,
            "splits_with_zero_test_misses": int((misses == 0).sum()),
            "points_with_miss_when_missing": {
                "median": float(np.median(misses[misses > 0])) if (misses > 0).any() else 0.0,
                "max": int(misses.max()),
            },
            "missed_cases_when_missing_median": float(np.median(cases[cases > 0])) if (cases > 0).any() else 0.0,
            "mean_verified_median": float(np.median([r["fast_screen"]["mean_verified"] for r in rows])),
        }
        for tname in timings:
            sp = np.array([r[tname]["speedup"] for r in rows])
            entry[tname] = {
                "speedup_median": round(float(np.median(sp)), 3),
                "speedup_p10_p90": [round(float(np.quantile(sp, q)), 3) for q in (0.1, 0.9)],
                "splits_passing_gate": int(((misses == 0) & (sp > GATE_SPEEDUP)).sum()),
            }
        summary[name] = entry

    result = {
        "cache": str(args.cache),
        "cache_weights": c["meta"]["weights"],
        "recipe": "task A: bias calibration on validation; cutoff = out-of-fold zero-miss minus CV shortfall",
        "seed": args.seed,
        "timings": {k: vars(v) for k, v in timings.items()},
        "summary": summary,
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for name, e in summary.items():
        print(f"{name:5}: zero test misses on {e['splits_with_zero_test_misses']}/{e['splits']} splits; "
              f"when missing, median {e['points_with_miss_when_missing']['median']} points "
              f"(max {e['points_with_miss_when_missing']['max']}); mean verified {e['mean_verified_median']:.1f}; "
              f"speedup original {e['original_screen']['speedup_median']}x / fast {e['fast_screen']['speedup_median']}x; "
              f"gate passes {e['fast_screen']['splits_passing_gate']}/{e['splits']} (fast)")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
