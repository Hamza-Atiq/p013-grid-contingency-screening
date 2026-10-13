# Grid contingency screening with a learned power flow model

A small project on the IEEE 57-bus grid. A graph neural network ranks every
single-branch outage by how dangerous it looks, and the exact power flow solver
then checks only the cases that need checking. The model sorts, the solver
decides.

**Try it:** [the live page](https://hamza-atiq.github.io/p013-grid-contingency-screening/)
(opens instantly; recorded results for 175 test days and the 4 days it ever
missed) or [run it live in Google Colab](https://colab.research.google.com/github/Hamza-Atiq/p013-grid-contingency-screening/blob/main/notebooks/p013_live.ipynb)
(the real model and solver on a new day; press Run all, about a minute).

## The problem

Grid operators run N-1 contingency analysis: for every line or transformer,
they ask "what if this one trips?", solve the AC power flow again, and check
for overloaded branches or bus voltages out of range. On IEEE 57-bus that is
80 branches, so 80 full Newton-Raphson solves for every operating point. That
is slow when conditions change quickly. The usual ML answer is to replace the
solver with a fast model, but a model that gets one dangerous case wrong is
not safe. So here the model never gets the final word.

## What I built

- A message passing network over the bus graph, written by hand in PyTorch
  (no PyTorch Geometric). Each branch has an in-service flag, and a branch
  that is out carries no message, so one trained model covers all 80 outages.
  It predicts bus voltages; branch flows are computed from those voltages.
- An exact islanding check in code, so the model is never asked about an
  outage that splits the grid. Those cases always go to the solver.
- In the recorded results, a few other outages were also sent without asking
  the model, because the solver could not solve them that day. A live screen
  cannot know that before solving, so the live app uses a rule fitted on
  practice days only: the 7 cases that failed to solve on any practice day
  (L37, L40, L41, L42, L44, T05, T06) always go to the solver. Measured: this
  sends exactly the same cases as the recorded run on the fixed exam and on
  all 50 re-split exams, so the numbers below hold for a screen that could
  really be deployed.
- A screening policy that decides which cases the solver must verify, fitted
  only on "practice" days and graded on unseen "exam" days.
- A data generator using pandapower on the PGLib-OPF version of IEEE 57-bus.

## How it works

For one operating point, the model predicts the grid state after each of the
80 outages and gives each case a danger score. The policy looks at total
system load. On heavy days it sends all 80 cases to the solver. On normal
days it sends every case whose score is at or above a cutoff learned from
practice days. Both the heavy-day load threshold and the cutoff are fitted on
practice days only. The solver's answer is the one that counts.

## Results (honest version)

Tested on 50 random re-splits of 750 generated days (575 practice, 175 exam
each), so 8,750 exam days in total:

| | Result |
| --- | --- |
| Speed vs. solving all 80 cases | **2.25x** faster (median over the 50 exams) |
| Cases sent to the solver | 32.7 of 80 per day on average (median over the 50 exams) |
| Days with at least one missed dangerous case | **14 of 8,750, about 1 day in 625** |
| Worst recall in any of the 50 exams | **99.91%** |
| Exams with zero misses | **39 of 50** |

**The pass mark I set in advance was not reached.** I wanted zero misses on at
least 48 of 50 exams and got 39. With the selection rule exactly as I first
registered it (which picked a different policy) the result was worse, 31 of
50. So this tool cannot promise "never misses". What it can state is the
measured miss rate above.

For context: one AC solve takes about 0.051 s on my laptop, so solving all 80
cases is about 4.1 s. The model's voltage error is about 0.0076 pu on average.

## Limitations

- One grid (IEEE 57-bus), all synthetic operating points. Not tested on a
  bigger system or on real operating data.
- Only branch outages. Generator, load and busbar outages are out of scope.
- The misses happen on normal-load days, where the learned cutoff is set from
  a limited number of practice days.
- The 50 exams are re-splits of the same 750 days, so they overlap. The 14
  miss days are only 4 different days, each counted once for every exam it
  was missed in.
- Timings are from one dual-core laptop with no GPU.

## How to run

Needs Python 3.11. Use a virtual environment.

```
python -m venv .venv
.venv\Scripts\activate          (Windows; on Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

**Get the data** (about 270 MB, not in the repo: 121 MB of generated power
flow data and 146 MB of saved model predictions). It is on Hugging Face at
[ABDHAM/p013-grid-contingency-screening-data](https://huggingface.co/datasets/ABDHAM/p013-grid-contingency-screening-data).
From the repository root:

```
hf download ABDHAM/p013-grid-contingency-screening-data --repo-type dataset --include "data/*" --include "eval/*" --local-dir .
```

This puts `data/` and `eval/cache/` exactly where the code expects them. (`hf`
comes with `pip install huggingface_hub`.) The dataset card on Hugging Face
lists the SHA-256 of every file.

**Run the tests** (from the repository root):

```
pytest
```

With the data downloaded, all tests run. Without it, the tests that need the
full generated data or the prediction cache skip themselves. `data_smoke/` is
a small (about 1 MB) dataset kept in the repo for the unit tests.

**Or regenerate the data from scratch** instead of downloading it. Fixed
seeds, so it comes out the same. Each step is slow on a laptop; run them one
at a time.

```
python -m p013.generate
python -m p013.generate --only sweep --test-points 500 --seed 20260916 --out data/practice
```

The first command writes `data/train.npz` (20,000 samples) and
`data/test_sweep.npz` (250 days). The second writes the 500 extra practice
days. Then rebuild the prediction caches and the robustness test:

```
python experiments/build_cache.py
python experiments/build_cache.py --sweep data/practice/test_sweep.npz --split practice --out eval/cache/practice_predictions.npz
python experiments/round2_robustness.py --pool eval/cache/predictions.npz eval/cache/practice_predictions.npz --out experiments/results/round2_rerun.json
```

The trained weights are already in `model/`, so you do not need to retrain.
(`python -m p013.train --out model_new` retrains from `data/train.npz` if you
want to; without `--out` it overwrites `model/`.)

**Open the site:** double-click `site/index.html` (the same page as the live
link at the top). Pick any of the 175 exam days, or one of the 4 days the
screen ever missed, and press Run. `site/results.html` is the full results
page: the 50 exams one by one and every missed case. To rebuild both (needs
the data):

```
python site/make_site.py
python demo/make_demo.py
```

Both refit the policy on all 50 exams and stop with an error unless every
exam matches `experiments/results/round2_main_original.json`, and both rerun
the trained model on days they show.

**Run the live app locally:** `pip install gradio==6.29.1`, then
`python space/app.py`. It uses `model/live_policy.npz`, the screening policy
frozen by `python experiments/build_live_policy.py`, which refuses to write
unless the fit equals the recorded one. `tests/test_live_parity.py` checks
that the app gives the recorded model outputs, scores and decisions.
`notebooks/p013_live.ipynb` is the same app for Google Colab.

## Prior work

The model is not new, and I do not claim it is. What I built is the pipeline
around it: the exact islanding check, the solver-has-the-final-word setup,
and an honest measurement of how often it misses.

- PowerFlowNet: *Power Flow Approximation Using Message Passing Graph Neural
  Networks*, arXiv:2311.03415. The closest model to mine.
- *Graph Neural Networks for Fast Contingency Analysis of Power Systems*,
  arXiv:2310.04213. GNN contingency screening, also tested on IEEE 57 and 118.
- DeepOPF: X. Pan, T. Zhao, M. Chen, S. Zhang, *DeepOPF: A Deep Neural Network
  Approach for Security-Constrained DC Optimal Power Flow*, IEEE Transactions
  on Power Systems, arXiv:1910.14448. A dense network for a related problem
  (OPF, not screening).
- pandapower: L. Thurner et al., *pandapower: An Open-Source Python Tool for
  Convenient Modeling, Analysis, and Optimization of Electric Power Systems*,
  IEEE Transactions on Power Systems, doi:10.1109/TPWRS.2018.2829021. Used for
  every power flow solve in this project.
- The grid data is `pglib_opf_case57_ieee.m` from the IEEE PES Power Grid
  Library (PGLib-OPF, v23.07), vendored unmodified in `cases/` under its
  CC BY 4.0 license.

## License

MIT, see `LICENSE`. The vendored case file in `cases/` keeps its own CC BY 4.0
license.
