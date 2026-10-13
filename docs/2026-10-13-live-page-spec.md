# Spec: P-013 live page (GitHub Pages + Hugging Face Space)

Date: 13 October 2026. Design approved by Hamza in chat on 13 October 2026.

## 1. Why

The current `demo.html` is a report card: exam scores, cutoffs, policies.
A professor who clicks a link in an email wants to know, in 30 seconds,
"what does this do, and can I try it?". The page never lets them try
anything, and it is not online. This spec fixes both.

## 2. Who it is for, and what counts as success

- A professor with a power-systems or ML background, opening a link from
  an email, probably on a laptop. Design and test at **1024 x 768**.
- Success: within 30 seconds they can say what the tool does, they have
  clicked Run at least once, and they have seen both the speed-up and the
  honest miss.
- Hamza must be able to explain every word on the page in a meeting.

## 3. What gets built

### 3.1 GitHub Pages site (the main link in emails)

One static page, `site/index.html`, plus one data file. No server, no
external scripts, works from `file://` as well.

1. **Header**, one sentence: if one power line fails, does anything
   overload? An AI ranks all 80 cases in about 0.15 s, and the exact
   solver checks only the risky ones.
2. **Grid picture**: the real IEEE 57-bus grid, 57 buses and 80 branches,
   laid out once in Python with a fixed seed. Generator buses marked.
3. **Pick a day**: a list of the 175 exam days. Each label is plain:
   "quiet day", "busy day", "heavy day (all 80 sent)", "near miss",
   "missed one". Default is a typical normal day.
4. **Run screening** button. It plays back the recorded result for that
   day, in order:
   - the model's danger ranking, coloured on the grid;
   - which cases were sent to the solver, and the solver's answer for each;
   - every truly dangerous case with a tick if it was sent, a cross if
     missed;
   - time: all 80 solved vs screen plus the sent cases. These are
     **estimates from laptop medians** (0.051 s per solve, 0.149 s per
     screen), not stopwatch times for that day, and the page says so;
   - "Missed: none", or the missed branch and its score vs the cutoff.
5. **Honest box**, always visible (not behind a click). It says:
   - these are recorded results from the real model and solver;
   - on this fixed exam, 0 of 175 days had a miss;
   - but over 50 re-shuffled exams the pass mark was **not** reached:
     zero misses on 39 of 50 exams against the 48 required, and 14
     miss-days that are only 4 distinct days;
   - P2 is the policy shown because it did best, but it was **not** the
     policy the pre-registered selection rule picked; that rule gave
     31 of 50;
   - what I would study next.
   Every number in the box is computed by `make_site.py`, not typed.
6. **Links**: "Run it live" (Hugging Face Space), "Code and write-up"
   (GitHub), "Full results" (the existing `demo.html`, moved to
   `site/results.html`).

The policy shown is P2 fitted on the 575 fixed practice days (the
"fixed exam" split, test 3 of round 2: heavy-day threshold 1,274.2 MW,
normal-day cutoff 0.784), so every one of the 175 days is a real unseen
exam day for that one fitted policy. On that split P2 missed **nothing**
(0 of 175 days, recorded).

Showing only that split would look cleaner than the truth. So the day
list also has a separate group, **"Days it missed (other exams)"**: the 4
distinct days missed somewhere in the 50 re-split exams, each shown with
the policy of one exam where it was missed, and labelled with that exam's
number and cutoff. The honest box states both facts: 0 misses on the
fixed exam, and 14 miss-days across 50 re-split exams that are only 4
distinct days.

### 3.2 Hugging Face Space (the live version)

A Gradio app, Python, CPU only, using the package in this repo.

- Inputs: target total load in MW (slider) and a seed.
- **Building the day.** Start from the case, apply
  `sampling.sample_operating_point` with the seed (same per-bus noise as
  training, P and Q scaled together, generation follows load), then scale
  all loads and generators by one common factor so total load equals the
  slider value. No outages beyond the one being screened.
- **Allowed range.** 735 to 1,550 MW, the range of total load seen in the
  750 generated days (measured: 732.3 to 1,553.2 MW). Outside it, the
  slider cannot go.
- **Scoring.** Run the model from `model/weights.npz` through the same
  scoring path as the round 2 caches, including any voltage correction the
  P2 pipeline applies, with the model's own predicted intact state as
  `vm_before`. It must not read any saved solver answer. Save whatever the
  Space needs (weights, fitted correction, P2 thresholds) in one small
  file built by a script, not by hand.
- **Cases that always go to the solver.** In the recorded results a case
  was "pinned" (sent without asking the model) when the solver could not
  solve it: L40 splits the grid on every day, and L37, L41, L42, L44,
  T05, T06 fail to solve on some days. A live screen cannot know that
  before solving, so the Space uses a rule fitted on practice days only:
  **every case that failed to solve on any practice day always goes to
  the solver** (these 7). Measured 13 Oct 2026 with
  `round2_main_original.json`'s seeds: on the fixed exam and on all 50
  re-split exams this rule sends exactly the same cases as the recorded
  run (0 of 50 splits differ, 14 miss-days either way), so the recorded
  numbers stand for a screen that could really be deployed. The README
  gets one sentence saying so.
- **Policy.** P2 with the fixed-exam thresholds (1,274.2 MW, 0.784).
- **Run.** Solve the sent cases with the real Newton-Raphson solver. Show
  the ranked table, sent cases, solver answers, and real stopwatch times
  on the Space's own CPU (labelled as such).
- **"Check all 80 with the solver"** button: solves every case so the
  visitor can see for themselves whether anything was missed.
- **Errors.** A sent case where the solver does not converge is shown as
  "no solution, counted as dangerous", the same rule as the paper version.
  If the intact day itself does not solve, show a plain message and ask
  for another seed.
- A note at the top: the free Space sleeps after 48 hours unused, so the
  first visit can take a minute or more to wake up.

### 3.3 GitHub repo

This folder becomes the public repo `Hamza-Atiq/p013-grid-contingency-screening`,
with GitHub Pages serving `site/` through a small GitHub Actions
workflow (`.github/workflows/pages.yml`), because branch publishing can
only serve the repo root or `/docs`. The README gets the two links at the
very top. Data stays on Hugging Face as now (`.gitignore` unchanged).

## 4. How the static page gets its numbers

`site/make_site.py` reads `eval/cache/predictions.npz` and the practice
cache, fits P2 on the practice days with `p013.regime_policy`, and writes
`site/data.js`. It **stops with an error** unless:

- **Fixed exam:** the refitted P2 thresholds equal test 3's recorded `L`
  and `cutoff_normal` to 1e-9, and the recomputed totals equal test 3's
  recorded `points_with_miss` (0), `missed_cases`, `mean_verified` and
  `speedup` to 1e-9.
- **Re-split exams:** all 50 runs are refitted with the same seeds, and
  each run's recorded P2 `points_with_miss`, `missed_cases`,
  `mean_verified` and `speedup` (in `runs`) are reproduced to 1e-9. Only
  then are the missed days identified, from the refit itself. Their
  count must be 14 miss-days over 4 distinct days. The JSON stores totals,
  not day identities, so this per-run match is the evidence.
- **Shown miss days:** each of the 4 is shown under the policy of the
  lowest-numbered exam that missed it.
- the model, rerun from `model/weights.npz` on a sample of days, gives
  the cached scores.

The page itself computes nothing about the result. It only displays
`data.js`.

## 5. Checks before calling it done

1. `pytest`: all existing tests still pass, plus tests for
   `make_site.py` (miss count, sent count, timing sum) and for the
   Space's run function on one fixed seed.
2a. **Space parity test:** for 5 exam days, the Space's scoring path, fed
   that day's loads and generation, reproduces the cached `model_score`
   for all 80 outages to 1e-5. This proves the live app is the same model
   as the recorded results.
2. I click through both pages in Chrome at 1024 x 768: every day type,
   the miss day, the links, the Space's both buttons.
3. Codex read-only check of `make_site.py` and the Space app against this
   spec.
4. Hamza clicks through and can explain the page.

## 6. Out of scope

- Generator, load or busbar outages; other grids; retraining the model.
- Accounts, analytics, custom domain.

## 6a. Amendment, 13 Oct 2026: the live app runs in Google Colab

Publishing the Space failed: Hugging Face now requires a PRO subscription to
host a Gradio or Docker Space on free CPU. Hamza chose Google Colab instead.
The "Run it live" link opens `notebooks/p013_live.ipynb` in Colab from the
public GitHub repo. It clones the repo, installs pandapower, numba and
Gradio, and launches the same `space/app.py` inline, so the app, the model,
the policy file and the solver are unchanged. Everything in 3.2 still holds
except hosting: the visitor's own free Colab runs it, so it never sleeps and
nobody pays. Colab's NumPy build may differ from the laptop's, so scores can
differ in the last digits from the recorded ones; the notebook says so.
`space/` stays in the repo so the Space can be published later unchanged.

## 7. Review history

- 13 Oct 2026: independent Codex review (gpt-6-astra, read-only). Numbers
  PASS. Concerns fixed in this version: the failed pass mark and the
  pre-registered selection rule are now in the always-visible honest box;
  times are labelled as laptop estimates; the live Space now defines how a
  day is built, the load range, scoring without saved answers, islanding
  and solver failures; GitHub Pages publishes `site/` through Actions; and
  section 4 proves the miss days by reproducing all 50 per-run records,
  because the JSON stores totals, not day identities.
