"""Freeze the policy the Space uses: P2 fitted on the fixed practice days.

Refits with the same code as round 2 and refuses to write unless the fit
equals the recorded one. Adds the 'always send' set: every case that failed
to solve on any practice day (spec 3.2; measured to change nothing).

    .venv/Scripts/python.exe experiments/build_live_policy.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

from round2_robustness import load_pool  # noqa: E402

from p013 import live  # noqa: E402
from p013 import regime_policy as rp  # noqa: E402

RESULTS = ROOT / "experiments" / "results" / "round2_main_original.json"
CACHES = [ROOT / "eval" / "cache" / "predictions.npz", ROOT / "eval" / "cache" / "practice_predictions.npz"]
OUT = ROOT / "model" / "live_policy.npz"


def main() -> None:
    rec = json.loads(RESULTS.read_text(encoding="utf-8"))["test3_fixed_exam"]["policies"]["P2"]
    pool = load_pool(CACHES)
    practice = np.flatnonzero(~pool["is_test"])
    pol = rp.fit_policy("P2", pool, rp.practice_scores(pool, practice))
    for k, v in (("L", rec["L"]), ("cutoff_normal", rec["cutoff_normal"])):
        if abs(getattr(pol, k) - v) > 1e-9:
            raise SystemExit(f"{k} {getattr(pol, k)} differs from the recorded {v}")
    if pol.calibration["kind"] != "bias":
        raise SystemExit(f"P2 calibration is {pol.calibration['kind']!r}, live.py assumes 'bias'")
    always = pool["pinned"][practice].any(axis=0)
    live.save_policy(OUT, live.Policy(np.asarray(pol.calibration["bias"]), float(pol.L),
                                      float(pol.cutoff_normal), always))
    print(f"written {OUT.relative_to(ROOT)}; always send:",
          [n for n, a in zip(live.NAMES, always) if a])


if __name__ == "__main__":
    main()
