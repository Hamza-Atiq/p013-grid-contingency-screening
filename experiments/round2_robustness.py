"""Research round 2: can P1, P2, P3 promise "never misses" on new days?

Implements the three tests of plans/2026-09-17-research-round-2-design.md,
section 6, with the policies of ``p013.regime_policy``:

1. **Robustness (the pass mark).** ``--splits`` random splits of the pool into
   575 practice / 175 exam. Every policy fitted on practice, scored on exam.
   If the pool has fewer than 750 points, the exam stays 175 and practice is
   whatever is left; the output says so.
2. **Control.** Same splits; P1 fitted on a random 75 of each split's practice.
3. **Fixed exam.** Practice = every pool point with ``is_test == False``;
   exam = every point with ``is_test == True``.

Pass mark (primary pipeline, test 1): zero misses on at least 48 of 50 exams
AND median fast-screen speedup at least 2.0.

    OMP_NUM_THREADS=1 .venv/Scripts/python.exe experiments/round2_robustness.py \\
        --pool eval/cache/predictions.npz eval/cache/practice_predictions.npz

Costs are the clean timings (``verify_resplit.clean_timings``), both screens.
A heavy day under P2 is priced as ``overhead + 80 solves``: the screen's
overhead is still charged, which is conservative.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_resplit import clean_timings  # noqa: E402

from p013 import regime_policy as rp  # noqa: E402
from p013.screening_metrics import evaluate_policy  # noqa: E402

N_PRACTICE = 575
N_EXAM = 175
N_CONTROL = 75
SPLIT_SEED = 12345      # same as round 1's verify_resplit
CONTROL_SEED = 20260917
PASS_ZERO_MISS = 48
PASS_SPEEDUP = 2.0

#: Only these arrays are loaded. truth_vm and dangerous are needed to FIT on
#: practice rows and to SCORE exam rows; policies never read them at exam rows.
KEYS = ("model_vm", "truth_vm", "model_thermal", "pinned", "dangerous", "load_p_mw", "is_test")
METRIC_KEYS = ("points_with_miss", "missed_cases", "recall", "mean_verified", "speedup")


# --------------------------------------------------------------------------
# pool
# --------------------------------------------------------------------------
def load_pool(paths: list[Path]) -> dict:
    """Concatenate caches along the points axis, remembering where each point came from."""
    parts, names, sources = [], None, []
    for i, path in enumerate(paths):
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            part = {k: z[k] for k in KEYS}
            n = z["names"]
        if names is None:
            names = n
        elif not np.array_equal(names, n):
            raise ValueError(f"{path}: contingency names differ from {paths[0]}")
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        sources.append({
            "path": str(path),
            "n_points": int(part["pinned"].shape[0]),
            "n_is_test": int(part["is_test"].sum()),
            "weights": meta.get("weights"),
            "weights_sha256": meta.get("weights_sha256"),
            "split": meta.get("split"),
        })
        part["source"] = np.full(part["pinned"].shape[0], i, dtype=np.int16)
        part["index_in_source"] = np.arange(part["pinned"].shape[0], dtype=np.int32)
        parts.append(part)
    weights = {s["weights_sha256"] for s in sources}
    if len(weights) > 1:
        raise ValueError(f"pool mixes caches from different models: {sources}")
    pool = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    pool["names"] = names
    pool["sources"] = sources
    return pool


# --------------------------------------------------------------------------
# scoring one policy on one exam
# --------------------------------------------------------------------------
def score_exam(verify, data, exam, timings) -> dict:
    D, P = data["dangerous"][exam], data["pinned"][exam]
    out = {}
    for tname, t in timings.items():
        r = evaluate_policy(verify, D, P, t)
        out[tname] = {k: r[k] for k in METRIC_KEYS}
    return out


def run_split(data, practice, exam, timings, control_rows=None) -> dict:
    """Fit on ``practice`` only, then score on ``exam``."""
    fitted = rp.fit_all(data, practice)
    row = {"primary": fitted["primary"], "policies": {}}
    for name, pol in fitted["policies"].items():
        row["policies"][name] = {**pol.summary(), "exam": score_exam(pol.verify(data, exam), data, exam, timings)}
    row["policies"]["primary"] = {
        "selected": fitted["primary"],
        "exam": row["policies"][fitted["primary"]]["exam"],
    }
    if control_rows is not None:
        ps = rp.practice_scores(data, control_rows)
        pol = rp.fit_p1(data, ps)
        row["policies"]["control_P1_75"] = {
            **pol.summary(), "exam": score_exam(pol.verify(data, exam), data, exam, timings)}
    return row


def summarise(rows: list[dict], name: str, timings) -> dict:
    ex = [r["policies"][name]["exam"] for r in rows]
    misses = np.array([e["fast_screen"]["points_with_miss"] for e in ex])
    entry = {
        "exams": len(ex),
        "exams_with_zero_misses": int((misses == 0).sum()),
        "points_with_miss_when_missing": {
            "median": float(np.median(misses[misses > 0])) if (misses > 0).any() else 0.0,
            "max": int(misses.max()),
        },
        "worst_recall": float(min(e["fast_screen"]["recall"] for e in ex)),
        "mean_verified_median": float(np.median([e["fast_screen"]["mean_verified"] for e in ex])),
    }
    for tname in timings:
        sp = np.array([e[tname]["speedup"] for e in ex])
        entry[tname] = {
            "speedup_median": round(float(np.median(sp)), 3),
            "speedup_p10_p90": [round(float(np.quantile(sp, q)), 3) for q in (0.1, 0.9)],
        }
    if name != "primary":
        entry["splits_with_an_infinite_cutoff"] = int(
            sum(1 for r in rows if r["policies"][name]["infinite_cutoffs"] > 0))
        if name in ("P2", "P3"):
            qs = [r["policies"][name]["percentile"] for r in rows]
            entry["chosen_L_percentile_counts"] = {str(q): qs.count(q) for q in rp.PERCENTILES}
    return entry


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", type=Path, nargs="+", default=[Path("eval/cache/predictions.npz")])
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--out", type=Path, default=Path("experiments/results/round2_robustness.json"))
    ap.add_argument("--skip-fixed", action="store_true", help="skip test 3")
    args = ap.parse_args(argv)

    started = time.perf_counter()
    data = load_pool(args.pool)
    timings = clean_timings()
    n = data["pinned"].shape[0]
    if n < N_EXAM + 1:
        raise SystemExit(f"pool has {n} points; need more than {N_EXAM}")
    n_practice = min(N_PRACTICE, n - N_EXAM)
    notes = []
    if n_practice < N_PRACTICE:
        notes.append(f"pool has {n} points (< {N_PRACTICE + N_EXAM}): test 1 uses {n_practice} practice / "
                     f"{N_EXAM} exam instead of {N_PRACTICE}/{N_EXAM}")
    if n_practice <= N_CONTROL:
        notes.append(f"practice has only {n_practice} days, so test 2's random {N_CONTROL} is all of it: "
                     "the control equals P1 and tells nothing")
    for msg in notes:
        print("NOTE:", msg)

    # ---- tests 1 and 2 ------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    rng_control = np.random.default_rng(CONTROL_SEED)
    rows = []
    for s in range(args.splits):
        perm = rng.permutation(n)
        practice = np.sort(perm[:n_practice])
        exam = np.sort(perm[n_practice:n_practice + N_EXAM])
        control = np.sort(rng_control.choice(practice, size=min(N_CONTROL, practice.size), replace=False))
        row = run_split(data, practice, exam, timings, control)
        row["split"] = s
        row["exam_sources"] = np.bincount(data["source"][exam], minlength=len(data["sources"])).tolist()
        row["exam_is_test"] = int(data["is_test"][exam].sum())
        rows.append(row)
        p = row["policies"]
        print(f"  split {s + 1:2d}/{args.splits}: primary {row['primary']}; misses "
              + " ".join(f"{k}={p[k]['exam']['fast_screen']['points_with_miss']}"
                         for k in ("P1", "P2", "P3", "control_P1_75"))
              + f"; L% P2={p['P2']['percentile']} P3={p['P3']['percentile']}"
              + f"; {time.perf_counter() - started:.0f} s", flush=True)

    names = ("P1", "P2", "P3", "primary", "control_P1_75")
    test1 = {k: summarise(rows, k, timings) for k in names}
    sel = [r["primary"] for r in rows]
    prim = test1["primary"]
    test1["primary"]["selected_counts"] = {k: sel.count(k) for k in rp.POLICIES}
    verdict = {
        "exams_with_zero_misses": prim["exams_with_zero_misses"],
        "needed": PASS_ZERO_MISS,
        "fast_speedup_median": prim["fast_screen"]["speedup_median"],
        "needed_speedup": PASS_SPEEDUP,
        "valid_for_pass_mark": args.splits == 50 and n_practice == N_PRACTICE,
        "pass": bool(prim["exams_with_zero_misses"] >= PASS_ZERO_MISS
                     and prim["fast_screen"]["speedup_median"] >= PASS_SPEEDUP),
    }

    # ---- test 3 -------------------------------------------------------------
    fixed = None
    if not args.skip_fixed:
        practice = np.flatnonzero(~data["is_test"])
        exam = np.flatnonzero(data["is_test"])
        if exam.size and practice.size:
            r = run_split(data, practice, exam, timings)
            fixed = {"practice_points": int(practice.size), "exam_points": int(exam.size),
                     "primary": r["primary"],
                     "policies": {k: v for k, v in r["policies"].items()}}

    result = {
        "design": "plans/2026-09-17-research-round-2-design.md",
        "pool": data["sources"],
        "pool_points": n,
        "notes": notes,
        "seed": args.seed,
        "control_seed": CONTROL_SEED,
        "folds": {"n": rp.N_FOLDS, "seed": rp.FOLD_SEED},
        "L_percentiles": list(rp.PERCENTILES),
        "timings": {k: vars(v) for k, v in timings.items()},
        "test1_robustness": {"practice": n_practice, "exam": N_EXAM, "splits": args.splits,
                             "summary": test1, "pass_mark": verdict},
        "test2_control": {"practice": min(N_CONTROL, n_practice),
                          "summary": {"control_P1_75": test1["control_P1_75"], "P1_full_practice": test1["P1"]}},
        "test3_fixed_exam": fixed,
        "chosen_L_per_split": [{"split": r["split"], "primary": r["primary"],
                                **{k: {"percentile": r["policies"][k]["percentile"], "L": r["policies"][k]["L"]}
                                   for k in ("P2", "P3")}} for r in rows],
        "runs": rows,
        "seconds": round(time.perf_counter() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=_json), encoding="utf-8")

    print(f"\nTest 1: {args.splits} splits, {n_practice} practice / {N_EXAM} exam")
    _print_table(test1, names)
    print(f"primary selected: {test1['primary']['selected_counts']}")
    print(f"PASS MARK (>= {PASS_ZERO_MISS}/50 zero-miss AND fast median >= {PASS_SPEEDUP}x): "
          f"{'PASS' if verdict['pass'] else 'FAIL'}"
          + ("" if verdict["valid_for_pass_mark"] else "  [NOT a valid pass-mark run: wrong split size or count]"))
    if fixed:
        print(f"\nTest 3: fixed exam, {fixed['practice_points']} practice / {fixed['exam_points']} exam; "
              f"primary = {fixed['primary']}")
        for k, v in fixed["policies"].items():
            f = v["exam"]["fast_screen"]
            print(f"  {k:8} misses {f['points_with_miss']:3d} pts  recall {f['recall']:.4f}  "
                  f"mean verified {f['mean_verified']:5.1f}  fast {f['speedup']:.3f}x  "
                  f"orig {v['exam']['original_screen']['speedup']:.3f}x")
    print(f"\n{time.perf_counter() - started:.0f} s; written to {args.out}")
    return 0


def _print_table(summary, names):
    print(f"  {'policy':14} {'zero-miss':>9} {'miss pts med/max':>16} {'worst recall':>12} "
          f"{'mean ver':>8} {'fast x (p10-p90)':>22} {'orig x':>7}")
    for k in names:
        e = summary[k]
        f, o = e["fast_screen"], e["original_screen"]
        mm = e["points_with_miss_when_missing"]
        print(f"  {k:14} {e['exams_with_zero_misses']:>4}/{e['exams']:<4} {mm['median']:>8.1f}/{mm['max']:<7} "
              f"{e['worst_recall']:>12.4f} {e['mean_verified_median']:>8.1f} "
              f"{f['speedup_median']:>7.3f} ({f['speedup_p10_p90'][0]:.2f}-{f['speedup_p10_p90'][1]:.2f}) "
              f"{o['speedup_median']:>7.3f}")


def _json(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


if __name__ == "__main__":
    sys.exit(main())
