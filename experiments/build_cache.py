"""Build eval/cache/predictions.npz, then prove it reproduces EVAL_NOTES §4.

Research round 1, phase 0. Run from ``build/``, ALONE - nothing else heavy on
the machine while it runs::

    .venv/Scripts/python.exe experiments/build_cache.py
    .venv/Scripts/python.exe experiments/build_cache.py --weights model_x/weights.npz --out eval/cache/model_x.npz

The anchor at the end is the point of the script. If the cache and the shared
metrics module cannot reproduce the trade-off table that was measured on
2026-09-12, nothing built on them can be trusted, so a mismatch exits non-zero.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from p013 import prediction_cache
from p013.screening_metrics import Timing, evaluate_policy, top_k

#: EVAL_NOTES §4, measured 2026-09-12: k -> (recall, points with a miss, speedup).
#: Recall at k=12 is 0.804487 in results.json; the write-up rounded it to 0.805,
#: which was corrected on 2026-09-17 when this anchor caught it.
ANCHOR = {12: (0.804, 82, 3.27), 26: (0.961, 53, 2.08), 30: (0.972, 36, 1.88), 72: (1.000, 0, 0.95)}


def anchor(cache: dict, timing: Timing) -> bool:
    t = cache["is_test"]
    args = (cache["dangerous"][t], cache["pinned"][t], timing)
    ok = True
    print("\nanchor against EVAL_NOTES §4 (test split, model ranking):")
    for k, (recall, missed, speedup) in ANCHOR.items():
        r = evaluate_policy(top_k(cache["model_rank"][t], k), *args)
        match = (
            round(r["recall"], 3) == recall
            and r["points_with_miss"] == missed
            and round(r["speedup"], 2) == speedup
        )
        ok &= match
        print(
            f"  k={k:>2}: recall {r['recall']:.3f} (want {recall}), "
            f"misses {r['points_with_miss']} (want {missed}), "
            f"speedup {r['speedup']:.2f}x (want {speedup})  {'OK' if match else 'MISMATCH'}"
        )
    truth = evaluate_policy(top_k(cache["truth_rank"][t], 26), *args)
    perfect = truth["points_with_miss"] == 0 and round(truth["speedup"], 2) == 2.08
    ok &= perfect
    print(f"  perfect ranking at k=26: misses {truth['points_with_miss']}, "
          f"{truth['speedup']:.2f}x (want 0, 2.08x)  {'OK' if perfect else 'MISMATCH'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, default=Path("data/test_sweep.npz"))
    ap.add_argument("--weights", type=Path, default=Path("model/weights.npz"))
    ap.add_argument("--out", type=Path, default=prediction_cache.DEFAULT_CACHE)
    ap.add_argument("--split", choices=("evaluate", "practice"), default="evaluate",
                    help="practice: every point is practice-only (round 2 extra sweep); implies --skip-anchor")
    ap.add_argument("--anchor-only", action="store_true",
                    help="check an existing cache without rebuilding it")
    ap.add_argument("--skip-anchor", action="store_true",
                    help="for a DIFFERENT model, whose numbers should not match §4")
    args = ap.parse_args()

    if not args.anchor_only:
        print(f"building {args.out} from {args.weights}", flush=True)
        prediction_cache.build(args.sweep, args.weights, args.out, split=args.split)
    cache = prediction_cache.load(args.out)
    print(f"built in {cache['meta']['seconds']} s, "
          f"{cache['is_test'].sum()} test + {(~cache['is_test']).sum()} validation points")

    if args.skip_anchor or args.split == "practice":
        return
    timing = Timing.from_results(json.loads(Path("eval/results.json").read_text(encoding="utf-8")))
    print(f"timing: one solve {timing.solve_s:.4f} s, overhead {timing.overhead_s:.3f} s "
          f"= {timing.overhead_in_solves:.1f} solves")
    if not anchor(cache, timing):
        print("\nANCHOR FAILED - do not use this cache")
        sys.exit(1)
    print("\nanchor passed - cache and metrics reproduce EVAL_NOTES §4")


if __name__ == "__main__":
    main()
