"""Write site/data.js: every fixed-exam day plus the 4 days P2 ever missed.

Every number is refitted and checked against
experiments/results/round2_main_original.json before anything is written
(spec section 4). The page only displays this file.

    .venv/Scripts/python.exe site/make_site.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))

import make_demo as md  # noqa: E402  (chdirs to ROOT, sets up imports, holds the checks)
from round2_robustness import N_EXAM, N_PRACTICE, load_pool, score_exam  # noqa: E402
from verify_resplit import clean_timings  # noqa: E402

from p013 import calibration as cal  # noqa: E402
from p013.case_loader import load_case  # noqa: E402
from p013.forward_numpy import load_weights  # noqa: E402
from p013.naming import branch_table  # noqa: E402

# make_demo.close() is relative (1e-9 x value); the spec promises an absolute 1e-9.
md.close = lambda a, b: (a is b) if (a is None or b is None) else abs(float(a) - float(b)) <= 1e-9

OUT = ROOT / "site" / "data.js"
LINKS = {
    "live": "https://colab.research.google.com/github/Hamza-Atiq/p013-grid-contingency-screening/blob/main/notebooks/p013_live.ipynb",
    "github": "https://github.com/Hamza-Atiq/p013-grid-contingency-screening",
    "results": "results.html",
}
NEAR = 0.05  # a normal day is a "near miss" if a dangerous case scored within this of the cutoff


def layout():
    """Bus positions for the drawing, from the real case. Kamada-Kawai is deterministic."""
    net = load_case()
    refs = branch_table(net)
    pos = nx.kamada_kawai_layout(nx.Graph((r.from_bus, r.to_bus) for r in refs))
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    gen = {int(b) for b in net.gen.bus} | {int(net.ext_grid.bus.iloc[0])}
    buses = [{
        "id": int(b),
        "x": round(float((pos[b][0] - xs.min()) / (xs.max() - xs.min())), 4),
        "y": round(float((pos[b][1] - ys.min()) / (ys.max() - ys.min())), 4),
        "gen": int(b) in gen,
    } for b in sorted(pos)]
    return buses, [{"name": r.name, "a": r.from_bus, "b": r.to_bus} for r in refs]


def kind_of(d: dict, near: bool) -> str:
    if d["heavy"]:
        return "heavy day, all 80 checked"
    if near:
        return "near miss"
    return "busy day" if d["busy"] else "quiet day"


def main() -> None:
    results = json.loads(md.RESULTS.read_text(encoding="utf-8"))
    test1 = results["test1_robustness"]
    pool = load_pool(md.CACHES)
    names = pool["names"]
    n = pool["pinned"].shape[0]
    timings = clean_timings()
    for k, t in timings.items():
        if vars(t) != results["timings"][k]:
            raise SystemExit(f"timings differ from the recorded run: {k}")

    # ---- 50 re-split exams: reproduce every run, then find the miss days ----
    rng = np.random.default_rng(results["seed"])
    miss: dict[int, dict] = {}
    for s, run in enumerate(results["runs"]):
        perm = rng.permutation(n)
        practice = np.sort(perm[:N_PRACTICE])
        exam = np.sort(perm[N_PRACTICE:N_PRACTICE + N_EXAM])
        rec = run["policies"][md.POLICY]
        pol = md.fit_p2(pool, practice, rec, f"exam {s + 1}")
        verify = pol.verify(pool, exam)
        md.check_exam(score_exam(verify, pool, exam, timings), rec["exam"], f"exam {s + 1}")
        for j, row in enumerate(exam):
            missed_here = pool["dangerous"][row] & ~(verify[j] | pool["pinned"][row])
            if missed_here.any() and int(row) not in miss:   # first (lowest-numbered) exam wins
                miss[int(row)] = md.day_payload(pool, pol, row, names, verify[j],
                                                f"missed one (exam {s + 1} of 50)", "missed", s + 1)
        print(f"  exam {s + 1:2d}: reproduced", flush=True)
    total = sum(r["policies"][md.POLICY]["exam"]["fast_screen"]["points_with_miss"]
                for r in results["runs"])
    if total != 14 or len(miss) != 4:
        raise SystemExit(f"expected 14 miss-days over 4 distinct days, got {total} over {len(miss)}")

    # ---- the fixed exam: one policy, all 175 days ---------------------------
    fixed = results["test3_fixed_exam"]["policies"][md.POLICY]
    practice = np.flatnonzero(~pool["is_test"])
    exam = np.flatnonzero(pool["is_test"])
    pol = md.fit_p2(pool, practice, fixed, "fixed exam")
    verify = pol.verify(pool, exam)
    md.check_exam(score_exam(verify, pool, exam, timings), fixed["exam"], "fixed exam")
    load = cal.total_load(pool)
    sent_normal = [int((verify[j] | pool["pinned"][r]).sum())
                   for j, r in enumerate(exam) if load[r] < pol.L]
    busy_at = float(np.percentile(sent_normal, 75))

    days = []
    for j, row in enumerate(exam):
        d = md.day_payload(pool, pol, row, names, verify[j], "", "fixed")
        d["busy"] = d["nSent"] >= busy_at
        dang = [r["score"] for r in d["rows"] if r["dangerous"] and not r["pinned"]]
        near = (not d["heavy"]) and bool(dang) and min(dang) - pol.cutoff_normal < NEAR
        d["group"] = "fixed"
        d["label"] = f"Day {j + 1}: {d['loadMw']:,.0f} MW, {kind_of(d, near)}"
        days.append(d)
    if any(d["missed"] for d in days):
        raise SystemExit("a miss on the fixed exam, but the record says 0")
    for d in miss.values():
        d["group"] = "missed"
        d["label"] = f"{d['loadMw']:,.0f} MW, {d['label']}"
        by_name = {r["name"]: r for r in d["rows"]}
        d["missNote"] = [{"name": b, "score": by_name[b]["score"], "cutoff": d["cutoff"],
                          "L": d["L"], "examNo": d["examNo"]} for b in d["missed"]]
        days.append(d)

    t = timings["fast_screen"]
    for d in days:   # estimates from laptop medians (spec 3.1); the page only displays them
        screen = t.overhead_s + d["nSent"] * t.solve_s
        d["allSeconds"] = round(t.baseline_s, 3)
        d["screenSeconds"] = round(screen, 3)
        d["speedup"] = round(t.baseline_s / screen, 2)

    print("  rerunning the model on a sample of days ...", flush=True)
    points = []
    for path in md.CACHES:
        with np.load(path, allow_pickle=False) as z:
            points.append(z["point"])
    md.check_against_model(days[:5] + list(miss.values()), load_weights(md.WEIGHTS), points)

    summary = test1["summary"]
    buses, branches = layout()
    keys = ("label", "group", "loadMw", "heavy", "nSent", "nDangerous", "missed", "missNote",
            "examNo", "cutoff", "allSeconds", "screenSeconds", "speedup", "rows")
    data = {
        "case": "IEEE 57-bus (PGLib-OPF)",
        "n": int(len(names)),
        "timing": {"solve": t.solve_s, "overhead": t.overhead_s, "baseline": t.baseline_s},
        "policy": {"L": round(pol.L, 1), "cutoff": round(pol.cutoff_normal, 4)},
        "buses": buses,
        "branches": branches,
        "days": [{k: d.get(k) for k in keys} for d in days],
        "honest": {
            "fixedMissDays": fixed["exam"]["fast_screen"]["points_with_miss"],
            "fixedExamDays": int(exam.size),
            "fixedSpeedup": fixed["exam"]["fast_screen"]["speedup"],
            "zeroMiss": summary[md.POLICY]["exams_with_zero_misses"],
            "nExams": len(results["runs"]),
            "passNeeded": test1["pass_mark"]["needed"],
            "missDays": total,
            "distinctMissDays": len(miss),
            "registeredZeroMiss": summary["primary"]["exams_with_zero_misses"],
            "speedupMedian": summary[md.POLICY]["fast_screen"]["speedup_median"],
        },
        "links": LINKS,
    }
    OUT.write_text("window.P013 = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"written {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
