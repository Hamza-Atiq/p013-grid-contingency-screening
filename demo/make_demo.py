"""Build ``demo.html`` from the real model output and the round 2 results.

Nothing on the page is made up. The script

1. reads the recorded result of the 50-exam robustness test
   (``experiments/results/round2_main_original.json``),
2. refits the screening policy P2 with the same code that produced that file
   (``p013.regime_policy`` and ``experiments/round2_robustness.py``) on the
   same prediction caches, and stops with an error if any refitted exam does
   not reproduce the recorded numbers,
3. reruns the trained model (``model/weights.npz``) on every day shown in
   detail and checks that its scores match the cache,
4. writes everything into one self-contained HTML file, because a page opened
   from ``file://`` cannot fetch a sibling JSON file.

Run from the repository root, with the data downloaded (see README)::

    python demo/make_demo.py

It writes ``site/results.html`` (the "Full results" page of the site).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)  # the experiment helpers read paths relative to the repo root
sys.path.insert(0, str(ROOT / "experiments"))

from round2_robustness import N_EXAM, N_PRACTICE, load_pool, score_exam  # noqa: E402
from verify_resplit import clean_timings  # noqa: E402

from p013 import calibration as cal  # noqa: E402
from p013 import regime_policy as rp  # noqa: E402
from p013.evaluate import Sweep, model_ranking  # noqa: E402
from p013.forward_numpy import load_weights  # noqa: E402

RESULTS = ROOT / "experiments" / "results" / "round2_main_original.json"
ACCURACY = ROOT / "eval" / "results.json"
WEIGHTS = ROOT / "model" / "weights.npz"
CACHES = [ROOT / "eval" / "cache" / "predictions.npz", ROOT / "eval" / "cache" / "practice_predictions.npz"]
SWEEPS = [ROOT / "data" / "test_sweep.npz", ROOT / "data" / "practice" / "test_sweep.npz"]
TEMPLATE = Path(__file__).resolve().parent / "template.html"
OUT = ROOT / "site" / "results.html"

POLICY = "P2"
TOL = 1e-9  # refitted numbers must equal the recorded ones


def close(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= TOL * max(1.0, abs(float(b)))


def check_exam(got: dict, want: dict, where: str) -> None:
    """The refitted exam must reproduce the recorded one, metric by metric."""
    for screen, metrics in want.items():
        for k, v in metrics.items():
            if not close(got[screen][k], v):
                raise SystemExit(f"{where}: {screen}.{k} is {got[screen][k]}, recorded {v}")


def fit_p2(pool: dict, practice: np.ndarray, recorded: dict, where: str):
    """Fit P2 on ``practice`` only and check it against the recorded fit."""
    ps = rp.practice_scores(pool, practice)
    pol = rp.fit_policy(POLICY, pool, ps)
    for k in ("L", "percentile", "cutoff_normal"):
        if not close(getattr(pol, k), recorded[k]):
            raise SystemExit(f"{where}: {k} is {getattr(pol, k)}, recorded {recorded[k]}")
    return pol


def day_rows(pool, pol, row, names, verify_row) -> list[dict]:
    """All 80 branches of one day, in the order the model puts them."""
    score = cal.calibrated_scores(pol.calibration, rp._subset(pool, np.array([row])))[0]
    out = []
    for c, name in enumerate(names):
        pinned = bool(pool["pinned"][row, c])
        out.append({
            "name": str(name),
            "score": None if pinned else round(float(score[c]), 4),
            "pinned": pinned,
            "dangerous": bool(pool["dangerous"][row, c]),
            "sent": bool(verify_row[c] or pinned),
        })
    # Pinned first (always sent), then by model score, highest first.
    out.sort(key=lambda r: (not r["pinned"], -(r["score"] if r["score"] is not None else 0.0)))
    return out


def day_payload(pool, pol, row, names, verify_row, label, kind, exam_no=None) -> dict:
    rows = day_rows(pool, pol, row, names, verify_row)
    load = float(cal.total_load({"load_p_mw": pool["load_p_mw"][row:row + 1]})[0])
    missed = [r["name"] for r in rows if r["dangerous"] and not r["sent"]]
    return {
        "label": label,
        "kind": kind,
        "examNo": exam_no,
        "source": int(pool["source"][row]),
        "indexInSource": int(pool["index_in_source"][row]),
        "loadMw": round(load, 1),
        "L": round(float(pol.L), 1),
        "heavy": bool(load >= pol.L),
        "cutoff": round(float(pol.cutoff_normal), 4),
        "nSent": sum(r["sent"] for r in rows),
        "nPinned": sum(r["pinned"] for r in rows),
        "nDangerous": sum(r["dangerous"] for r in rows),
        "missed": missed,
        "rows": rows,
    }


def check_against_model(days: list[dict], weights: dict, points: list[np.ndarray]) -> list[float]:
    """Rerun the trained model on every day shown and compare with the cache.

    The policy reads the cache. This proves the cache is what the model says.
    """
    sweeps = {}
    seconds = []
    for d in days:
        src = d["source"]
        sweeps.setdefault(src, Sweep(SWEEPS[src]))
        point = int(points[src][d["indexInSource"]])
        t0 = time.perf_counter()
        ranking = model_ranking(sweeps[src], point, weights)
        seconds.append(time.perf_counter() - t0)
        with np.load(CACHES[src], allow_pickle=False) as z:
            cached = z["model_score"][d["indexInSource"]]
            names = [str(n) for n in z["names"]]
        for c, name in enumerate(names):
            live = ranking.score.get(name)
            if live is None:
                if not np.isnan(cached[c]):
                    raise SystemExit(f"{d['label']}: {name} pinned live but not in the cache")
            elif abs(live - float(cached[c])) > 1e-4:
                raise SystemExit(f"{d['label']}: {name} live score {live} against cache {cached[c]}")
        d["modelChecked"] = True
    return seconds


def main() -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    accuracy = json.loads(ACCURACY.read_text(encoding="utf-8"))
    test1 = results["test1_robustness"]

    pool = load_pool(CACHES)
    names = pool["names"]
    n = pool["pinned"].shape[0]
    points = []
    for path in CACHES:
        with np.load(path, allow_pickle=False) as z:
            points.append(z["point"])

    timings = clean_timings()
    for k, t in timings.items():
        if vars(t) != results["timings"][k]:
            raise SystemExit(f"timings differ from the recorded run: {k}")

    # ---- the 50 exams: refit P2 on every split and reproduce the record ----
    if n != results["pool_points"] or test1["practice"] != N_PRACTICE or test1["exam"] != N_EXAM:
        raise SystemExit("pool or split sizes differ from the recorded run")
    rng = np.random.default_rng(results["seed"])
    exams, miss_days = [], []
    for s, run in enumerate(results["runs"]):
        perm = rng.permutation(n)
        practice = np.sort(perm[:N_PRACTICE])
        exam = np.sort(perm[N_PRACTICE:N_PRACTICE + N_EXAM])
        rec = run["policies"][POLICY]
        pol = fit_p2(pool, practice, rec, f"exam {s + 1}")
        verify = pol.verify(pool, exam)
        check_exam(score_exam(verify, pool, exam, timings), rec["exam"], f"exam {s + 1}")
        fast = rec["exam"]["fast_screen"]
        missed_here = []
        for j, row in enumerate(exam):
            miss = pool["dangerous"][row] & ~(verify[j] | pool["pinned"][row])
            if miss.any():
                d = day_payload(pool, pol, row, names, verify[j],
                                f"A day it missed (exam {s + 1} of 50)", "miss", s + 1)
                miss_days.append(d)
                missed_here.append({"day": [d["source"], d["indexInSource"]],
                                    "loadMw": d["loadMw"], "heavy": d["heavy"], "branches": d["missed"],
                                    "scores": [r["score"] for r in d["rows"] if r["name"] in d["missed"]]})
        print(f"  exam {s + 1:2d}: reproduced; {fast['points_with_miss']} miss day(s)", flush=True)
        exams.append({
            "no": s + 1,
            "missDays": fast["points_with_miss"],
            "missedCases": fast["missed_cases"],
            "recall": fast["recall"],
            "meanSent": fast["mean_verified"],
            "speedup": fast["speedup"],
            "L": round(rec["L"], 1),
            "percentile": rec["percentile"],
            "cutoff": round(rec["cutoff_normal"], 4),
            "heavyPractice": rec["details"]["heavy"]["n_days"],
            "misses": missed_here,
        })
    total_miss_days = sum(e["missDays"] for e in exams)
    # The exams are re-splits of one pool of days, so one hard day can sit in
    # several exams. Count the different days behind the miss days too.
    distinct_miss_days = len({(d["source"], d["indexInSource"]) for d in miss_days})
    if total_miss_days != len(miss_days):
        raise SystemExit(f"{len(miss_days)} miss days found, the record says {total_miss_days}")
    if any(d["heavy"] for d in miss_days):
        raise SystemExit("a miss on a heavy day: P2 sends all 80 then, so this cannot happen")

    # ---- the fixed exam: example days ---------------------------------------
    fixed_rec = results["test3_fixed_exam"]
    practice = np.flatnonzero(~pool["is_test"])
    exam = np.flatnonzero(pool["is_test"])
    pol = fit_p2(pool, practice, fixed_rec["policies"][POLICY], "fixed exam")
    verify = pol.verify(pool, exam)
    check_exam(score_exam(verify, pool, exam, timings), fixed_rec["policies"][POLICY]["exam"], "fixed exam")
    load = cal.total_load(pool)[exam]
    sent = (verify | pool["pinned"][exam]).sum(axis=1)
    heavy = load >= pol.L
    normal_idx = np.flatnonzero(~heavy)
    heavy_idx = np.flatnonzero(heavy)

    def margin(j):
        """How far the lowest-scored dangerous case sat above the cutoff."""
        row = exam[j]
        score = cal.calibrated_scores(pol.calibration, rp._subset(pool, np.array([row])))[0]
        d = pool["dangerous"][row] & ~pool["pinned"][row]
        return float(score[d].min() - pol.cutoff_normal) if d.any() else np.inf

    by_sent = normal_idx[np.argsort(sent[normal_idx], kind="stable")]
    picks = [
        ("A typical normal day", "normal", by_sent[len(by_sent) // 2]),
        ("The busiest normal day", "normal", by_sent[-1]),
        ("The closest call", "close", min(normal_idx, key=margin)),
        ("A heavy day", "heavy", heavy_idx[np.argsort(load[heavy_idx])[len(heavy_idx) // 2]]),
    ]
    days = [day_payload(pool, pol, exam[j], names, verify[j], label, kind) for label, kind, j in picks]
    days.append(miss_days[0])

    print("  rerunning the model on the days shown ...", flush=True)
    model_seconds = check_against_model(days, load_weights(WEIGHTS), points)
    print(f"  model rerun on {len(model_seconds)} days: every score matches the cache", flush=True)

    summary = test1["summary"]
    policies = []
    for key, label in (("P1", "P1: one cutoff for every day"),
                       ("P2", "P2: heavy days send all 80, normal days use a cutoff"),
                       ("P3", "P3: a separate cutoff for heavy and for normal days"),
                       ("primary", "The selection rule as first registered (picked P3 in 50 of 50)"),
                       ("control_P1_75", "Control: P1 fitted on only 75 practice days")):
        e = summary[key]
        policies.append({
            "key": key, "label": label,
            "zeroMiss": e["exams_with_zero_misses"],
            "missDays": sum(r["policies"][key]["exam"]["fast_screen"]["points_with_miss"]
                            for r in results["runs"]),
            "worstRecall": e["worst_recall"],
            "meanSent": e["mean_verified_median"],
            "speedup": e["fast_screen"]["speedup_median"],
        })

    t = timings["fast_screen"]
    data = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "case": accuracy["case"],
        "n": int(len(names)),
        "nExams": len(exams),
        "nExamDays": N_EXAM,
        "nPractice": N_PRACTICE,
        "poolDays": n,
        "passNeeded": test1["pass_mark"]["needed"],
        "passSpeed": test1["pass_mark"]["needed_speedup"],
        "headline": {
            "zeroMiss": summary[POLICY]["exams_with_zero_misses"],
            "missDays": total_miss_days,
            "worstRecall": summary[POLICY]["worst_recall"],
            "meanSent": summary[POLICY]["mean_verified_median"],
            "speedup": summary[POLICY]["fast_screen"]["speedup_median"],
            "speedupP10P90": summary[POLICY]["fast_screen"]["speedup_p10_p90"],
            "registeredZeroMiss": summary["primary"]["exams_with_zero_misses"],
        },
        "timing": {"solve": t.solve_s, "overhead": t.overhead_s, "baseline": t.baseline_s},
        "exams": exams,
        "distinctMissDays": distinct_miss_days,
        "policies": policies,
        "days": days,
        "fixedExam": {"practice": int(practice.size), "exam": int(exam.size),
                      "missDays": fixed_rec["policies"][POLICY]["exam"]["fast_screen"]["points_with_miss"]},
        "accuracy": accuracy["model_accuracy"],
        "accuracyDays": accuracy["split"]["test_points"],
    }

    html = TEMPLATE.read_text(encoding="utf-8")
    marker = "/*DATA*/"
    if marker not in html:
        raise SystemExit("template lost its data marker")
    OUT.write_text(html.replace(marker, json.dumps(data, separators=(",", ":"))), encoding="utf-8")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
