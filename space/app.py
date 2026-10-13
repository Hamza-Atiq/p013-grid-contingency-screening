"""P-013 live screening app (Gradio).

Runs inline in Google Colab through ``notebooks/p013_live.ipynb`` (the "Run it
live" link), locally with ``python space/app.py``, or as a Hugging Face Space
(``space/build_space.py`` assembles one; hosting it needs HF PRO).

Builds a fresh operating day, the AI ranks all 80 single line failures, and
the real Newton-Raphson solver checks the ones the AI sends. "Check all 80"
solves every case so a visitor can see for themselves whether anything was
missed. All the physics and the model live in ``p013.live``.
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))         # layout on the Space (space/_build)
sys.path.insert(0, str(HERE.parent / "src"))  # layout in the repository, for tests

import numpy as np  # noqa: E402

from p013 import live  # noqa: E402
from p013.forward_numpy import load_weights  # noqa: E402

MODEL = HERE / "model" if (HERE / "model").exists() else HERE.parent / "model"
WEIGHTS = load_weights(MODEL / "weights.npz")
POLICY = live.load_policy(MODEL / "live_policy.npz")

SITE = "https://hamza-atiq.github.io/p013-grid-contingency-screening/"
GITHUB = "https://github.com/Hamza-Atiq/p013-grid-contingency-screening"
HEADERS = ["Line", "AI danger score", "Sent to solver", "Solver says", "Result"]


def _rows(scr, truth) -> list[list]:
    """Always-checked lines first, then by AI score, highest first."""
    order = sorted(range(len(live.NAMES)),
                   key=lambda c: (not POLICY.always[c], -np.nan_to_num(scr.score[c], nan=0.0), c))
    rows = []
    for c in order:
        name = live.NAMES[c]
        t = truth.get(name)
        if t is None:
            says, result = "", ""
        else:
            says = ("DANGEROUS" if t.dangerous else "safe") + ("" if t.status == "converged" else " (no solution)")
            result = ("caught" if scr.sent[c] else "MISSED") if t.dangerous else ""
        score = "always checked" if POLICY.always[c] else f"{scr.score[c]:.3f}"
        rows.append([name, score, "yes" if scr.sent[c] else "no", says, result])
    return rows


def _run(load_mw, seed, everything: bool):
    lo, hi = live.LOAD_RANGE
    seed = 0 if seed is None else int(seed)          # a cleared seed box means seed 0
    if seed < 0:
        return "The day seed must be 0 or a positive whole number.", []
    if load_mw is None or not lo <= float(load_mw) <= hi:
        return f"Total load must be between {lo:.0f} and {hi:.0f} MW, the range the AI was trained on.", []
    day = live.build_day(seed, float(load_mw))
    intact = live.solve_intact(day)
    if not intact.ok:
        return ("The solver could not solve this day even with every line in service. "
                "Please try another seed."), []

    scr = live.score_day(day, WEIGHTS, POLICY)
    sent = [n for n, s in zip(live.NAMES, scr.sent) if s]
    todo = list(live.NAMES) if everything else sent
    t0 = time.perf_counter()
    truth = live.solve_cases(day, todo, vm_before=intact.vm_pu)
    solve_s = time.perf_counter() - t0

    kind = "heavy day, so all 80 are sent" if scr.heavy else "normal day"
    lines = [
        f"**Total load {day.total_mw:,.0f} MW** ({kind}).",
        f"The AI ranked all 80 line failures in **{scr.seconds:.2f} s** and **{len(sent)} of 80 were "
        f"sent to the solver** (7 lines that sometimes cannot be solved are always sent).",
        f"The solver took **{solve_s:.1f} s** for {len(todo)} cases on this machine.",
    ]
    if everything:
        missed = [n for n in live.NAMES if truth[n].dangerous and n not in sent]
        dangerous = sum(truth[n].dangerous for n in live.NAMES)
        lines.append(f"The solver found **{dangerous} dangerous** line failures. "
                     f"**Missed by the AI: {', '.join(missed) if missed else 'none'}.**")
    else:
        lines.append("Press **Check all 80 with the solver** to see whether the AI missed anything.")
    return "\n\n".join(lines), _rows(scr, truth)


def screen(load_mw, seed):
    return _run(load_mw, seed, everything=False)


def check_all(load_mw, seed):
    return _run(load_mw, seed, everything=True)


INTRO = f"""# If one power line fails, does anything overload?

An AI ranks all 80 possible single line failures on the IEEE 57-bus grid, and the real
Newton-Raphson solver checks only the risky ones. **The AI sorts, the solver decides.**

Pick how busy the grid is and a seed (each seed is a different day), then press **Run screening**.
Above {POLICY.L:,.0f} MW the day counts as heavy and every line is sent to the solver.

[Recorded results for 175 test days]({SITE}) | [Code and full write-up]({GITHUB})
"""


def build():
    import gradio as gr

    lo, hi = live.LOAD_RANGE
    with gr.Blocks(title="P-013 grid screening") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            load = gr.Slider(lo, hi, value=1000, step=5, label="Total load (MW)")
            seed = gr.Number(value=1, precision=0, minimum=0, label="Day seed")
        with gr.Row():
            b1 = gr.Button("Run screening", variant="primary")
            b2 = gr.Button("Check all 80 with the solver")
        out = gr.Markdown()
        table = gr.Dataframe(headers=HEADERS, wrap=True, interactive=False)
        # One queue for both buttons, one job at a time; live.LOCK is the second guard.
        b1.click(screen, [load, seed], [out, table], concurrency_id="grid")
        b2.click(check_all, [load, seed], [out, table], concurrency_id="grid")
    return demo


def warm_up() -> None:
    """Compile numba and build the screen once at start-up, so the first
    visitor does not wait for it (measured: about 13 s on the first click)."""
    day = live.build_day(1, 1000.0)
    intact = live.solve_intact(day)
    live.score_day(day, WEIGHTS, POLICY)
    live.solve_cases(day, ["L01"], vm_before=intact.vm_pu)


if __name__ == "__main__":
    warm_up()
    build().queue(default_concurrency_limit=1).launch()
