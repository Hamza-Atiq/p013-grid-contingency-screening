"""k_deploy, k*, and the results file. SPEC 3.3.

**k\\* is a measurement. The shortlist the tool actually verifies is a choice.**
They were one quantity until 2026-09-09, and separating them is what stops the
design being quietly circular:

- ``k_deploy`` is fixed **in advance** from a **validation** split - the worst
  ``k*`` seen there plus a stated margin. It is a constant baked into the
  shipped tool, and the console shows it.
- ``k*`` is then measured on a **test** split that ``k_deploy`` was never
  chosen from, and that is the headline.

Choosing the shortlist on the same data you then report zero misses on is a
result that means nothing. A reviewer enjoys finding that, so it is designed
out rather than argued about.

**The headline is the MAXIMUM k\\* across the test set, never the mean.** A
screening tool is judged on its worst day. Leading with the maximum is the
difference between a claim that is accepted and one that gets poked at.

**A miss is reported as a miss.** If a test point produces ``k* > k_deploy``
then a dangerous case fell outside the shortlist, it is named in the report,
and ``k_deploy`` is *not* enlarged afterwards to make it disappear (SPEC 6).
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import numpy as np

import p013
from p013.admittance import loading_percent
from p013.case_loader import load_case, parse_matpower
from p013.danger import DangerReport, risk_score, violations
from p013.dataset import to_bus_order
from p013.features import branch_features, bus_features, edge_index
from p013.forward_numpy import forward
from p013.naming import branch_names

#: Generated artefacts, both gitignored. Regenerate with p013.generate and
#: p013.train rather than expecting them in a clone.
DEFAULT_SWEEP = Path("data/test_sweep.npz")
DEFAULT_WEIGHTS = Path("model/weights.npz")

#: A point's contribution: its ranking, most dangerous first, and the set of
#: names the solver says are genuinely dangerous.
PointResult = tuple[Sequence[str], set]


class KStarReport(NamedTuple):
    """Everything SPEC 3.3 says to report, and the misses SPEC 6 insists on.

    Attributes:
        max: the headline. The deepest any dangerous case hid, anywhere.
        median: the typical operating point.
        per_point: k* for every point, in order, so nothing is hidden behind
            a summary statistic.
        recall_curve: ``recall_curve[k - 1]`` is the fraction of all dangerous
            cases caught by verifying the top ``k`` at every point.
        misses: one entry per point where ``k* > k_deploy``. Empty when no
            ``k_deploy`` was supplied, because k* alone cannot say whether
            anything was missed.
    """

    max: int
    median: float
    per_point: tuple[int, ...]
    recall_curve: tuple[float, ...]
    misses: list[dict]


def k_star(order: Sequence[str], dangerous: Iterable[str]) -> int:
    """How deep the ranking must be followed to catch every dangerous case.

    Zero when nothing is dangerous - which is not the same as "we did not
    look", and must never quietly become the whole list.
    """
    dangerous = set(dangerous)
    position = {name: i for i, name in enumerate(order)}

    unknown = dangerous - position.keys()
    if unknown:
        # A dangerous case missing from the ranking would silently lower k*,
        # which is the most flattering possible bug in this file.
        raise ValueError(
            f"dangerous contingencies absent from the ranking: {sorted(unknown)}"
        )

    return max((position[name] + 1 for name in dangerous), default=0)


def measure_k_star(
    per_point: Sequence[PointResult], k_deploy: int | None = None
) -> KStarReport:
    """Measure k* across a set of operating points. SPEC 3.3."""
    if not per_point:
        raise ValueError("no operating points to measure")

    n_contingencies = len(per_point[0][0])
    values = tuple(k_star(order, dangerous) for order, dangerous in per_point)

    total_dangerous = sum(len(dangerous) for _, dangerous in per_point)
    recall = []
    for k in range(1, n_contingencies + 1):
        caught = sum(
            len(dangerous & set(order[:k])) for order, dangerous in per_point
        )
        recall.append(caught / total_dangerous if total_dangerous else 1.0)

    misses: list[dict] = []
    if k_deploy is not None:
        for i, (order, dangerous) in enumerate(per_point):
            if values[i] <= k_deploy:
                continue
            misses.append(
                {
                    "point": i,
                    "k_star": values[i],
                    "k_deploy": k_deploy,
                    "missed": sorted(dangerous - set(order[:k_deploy])),
                }
            )

    return KStarReport(
        max=max(values),
        median=statistics.median(values),
        per_point=values,
        recall_curve=tuple(recall),
        misses=misses,
    )


def choose_k_deploy(validation: Sequence[PointResult], margin: int) -> int:
    """The shortlist the shipped tool verifies. A CHOICE, not a measurement.

    The worst ``k*`` on the validation split plus a stated safety margin,
    capped at the number of contingencies - a margin that runs off the end
    would claim to verify "the top 95 of 80", which is verifying everything and
    should say so.

    It must never be computed from the test split. The test in
    ``test_evaluate.py`` hands this function a validation set and then checks
    the answer does not move when a far worse test set exists.
    """
    if margin < 0:
        raise ValueError(f"margin must not be negative, got {margin}")
    if not validation:
        raise ValueError("k_deploy needs a validation split to be chosen from")

    n_contingencies = len(validation[0][0])
    worst = max(k_star(order, dangerous) for order, dangerous in validation)
    return min(worst + margin, n_contingencies)


def split_points(
    n_points: int, validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split operating points into validation and test. SPEC 2.4 and 3.3.

    **Whole operating points, never rows.** Each point contributes 81 rows that
    share a pre-outage state; splitting inside one would leak a test point's
    intact solve into validation, and the score would come out flattering.

    Seeded, because ``k_deploy`` is a constant shipped in the console and a
    split that moved between runs would move it.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}")

    order = np.random.default_rng(seed).permutation(n_points)
    cut = int(n_points * validation_fraction)
    return sorted(int(i) for i in order[:cut]), sorted(int(i) for i in order[cut:])


class PointRanking(NamedTuple):
    """One ranker's answer for one operating point of the sweep."""

    order: tuple[str, ...]
    score: dict[str, float]
    report: dict[str, DangerReport]
    pinned: tuple[str, ...]

    @property
    def dangerous(self) -> set[str]:
        """Pinned cases count as dangerous: they consume shortlist slots and go
        to the solver first (SPEC 3.3). Giving them free slots outside the
        budget would flatter the timing by hiding real solver work."""
        return {n for n, r in self.report.items() if r.is_dangerous} | set(self.pinned)


class Sweep:
    """The held-out sweep, read as operating points rather than as rows.

    Each point contributes 81 rows: **offset 0 is the intact N-0 state**,
    offsets 1 to 80 are the contingencies in canonical order. That layout is
    asserted here rather than assumed, because everything downstream measures
    deterioration against offset 0 and a silent shift would move every number
    in the project.
    """

    def __init__(self, path: Path | str = DEFAULT_SWEEP, case_path: Path | None = None):
        # Materialised ONCE, deliberately. np.load on an .npz returns a lazy
        # NpzFile: every `raw["vm_pu"]` re-reads and re-inflates the whole 9 MB
        # array from the zip. truth_ranking touches four arrays per
        # contingency, so one operating point was inflating roughly 320 arrays
        # and taking 28.6 s instead of 0.2 s - and a test run was killed for
        # running the machine out of memory before this was found.
        with np.load(path, allow_pickle=False) as handle:
            self.raw = {name: handle[name] for name in handle.files}
        self.ppc = parse_matpower(case_path or p013.CASE_PATH)
        self.names = tuple(branch_names(load_case()))

        self.n_branch = len(self.names)
        self.rows_per_point = self.n_branch + 1
        total = len(self.raw["status"])
        if total % self.rows_per_point:
            raise ValueError(
                f"{total} rows is not a whole number of {self.rows_per_point}-row "
                "operating points; this sweep predates the N-0 row"
            )
        self.n_points = total // self.rows_per_point

        self._position = {name: i for i, name in enumerate(self.names)}
        self._gen_bus = [int(b) for b in self.raw["gen_bus"]]
        self._load_bus = [int(b[0]) for b in self.ppc["bus"] if b[2] != 0 or b[3] != 0]

    def row(self, point: int, offset: int) -> int:
        return point * self.rows_per_point + offset

    def in_service_at(self, point: int, offset: int) -> np.ndarray:
        return self.raw["in_service"][self.row(point, offset)]

    def voltage_at(self, point: int, offset: int) -> np.ndarray:
        i = self.row(point, offset)
        return self.raw["vm_pu"][i] * np.exp(1j * np.deg2rad(self.raw["va_degree"][i]))

    def pre_outage(self, point: int) -> np.ndarray:
        """Solved voltage magnitudes of the intact system at this point."""
        i = self.row(point, 0)
        if self.raw["status"][i] != "converged":
            raise ValueError(
                f"point {point} has no solved intact state; there is nothing to "
                "measure deterioration against"
            )
        if not self.raw["in_service"][i].all():
            raise ValueError(f"offset 0 of point {point} is not the intact topology")
        return self.raw["vm_pu"][i]


def truth_ranking(sweep: Sweep, point: int) -> PointRanking:
    """Ground truth from the solver. SPEC 3.1's left-hand column.

    Loading is re-derived from the stored voltages rather than read from the
    stored ``loading_percent``: see :func:`p013.admittance.loading_percent` for
    why the two are different quantities and why both sides must use this one.
    """
    vm_before = sweep.pre_outage(point)
    scores: dict[str, float] = {}
    reports: dict[str, DangerReport] = {}
    pinned: list[str] = []

    for offset in range(1, sweep.rows_per_point):
        name = sweep.names[offset - 1]
        i = sweep.row(point, offset)
        if sweep.raw["status"][i] != "converged":
            pinned.append(name)
            continue
        vm = sweep.raw["vm_pu"][i]
        loading = loading_percent(
            sweep.ppc,
            sweep.voltage_at(point, offset),
            sweep.in_service_at(point, offset),
        )
        scores[name] = risk_score(vm, loading, vm_before=vm_before)
        reports[name] = violations(vm, loading, vm_before=vm_before)

    return _ranked(sweep, scores, reports, pinned)


def model_ranking(
    sweep: Sweep, point: int, weights: dict, pre_outage: str = "predicted"
) -> PointRanking:
    """The model's ranking. SPEC 3.1's right-hand column, same two checks.

    One batched forward pass covers the intact state and all 80 contingencies,
    which is the path SPEC 3.5 prices.

    ``pre_outage`` selects what the deterioration is measured against:

    - ``"predicted"`` (default) - the model's own prediction of the intact
      state. Self-contained: no solver is consulted anywhere, which is the
      claim SPEC 3.5's timing describes.
    - ``"measured"`` - the solved intact state. A real control room does know
      its present condition from state estimation, so this is a fair second
      claim. It is reported alongside rather than instead, because the two say
      different things and silently picking the flattering one is exactly the
      sort of choice this project exists to avoid.
    """
    if pre_outage not in ("predicted", "measured"):
        raise ValueError(
            f"pre_outage must be 'predicted' or 'measured', got {pre_outage!r}"
        )

    base = sweep.row(point, 0)
    one = slice(base, base + 1)
    load_p = to_bus_order(sweep.ppc, sweep._load_bus, sweep.raw["load_p_mw"][one])[0]
    load_q = to_bus_order(sweep.ppc, sweep._load_bus, sweep.raw["load_q_mvar"][one])[0]
    gen_p = to_bus_order(sweep.ppc, sweep._gen_bus, sweep.raw["gen_p_mw"][one])[0]
    gen_p = _with_slack(sweep.ppc, sweep._gen_bus, gen_p)

    bus_one = bus_features(sweep.ppc, load_p, load_q, gen_p)
    rows = sweep.rows_per_point
    bus_x = np.repeat(bus_one[None].astype(np.float32), rows, axis=0)
    branch_x = np.stack(
        [branch_features(sweep.ppc, sweep.in_service_at(point, o)) for o in range(rows)]
    ).astype(np.float32)

    slack_mask = bus_one[:, 4].astype(np.float32)
    vm, va = forward(
        weights, bus_x, branch_x, edge_index(sweep.ppc), slack_mask=slack_mask
    )

    vm_before = vm[0] if pre_outage == "predicted" else sweep.pre_outage(point)

    scores: dict[str, float] = {}
    reports: dict[str, DangerReport] = {}
    pinned: list[str] = []

    for offset in range(1, rows):
        name = sweep.names[offset - 1]
        if sweep.raw["status"][sweep.row(point, offset)] != "converged":
            # Pinned by exact code, identically for every ranker. The model is
            # never asked, so it can never be credited for these (SPEC 3.9).
            pinned.append(name)
            continue
        in_service = sweep.in_service_at(point, offset)
        v = vm[offset] * np.exp(1j * va[offset])
        loading = loading_percent(sweep.ppc, v, in_service)
        scores[name] = risk_score(vm[offset], loading, vm_before=vm_before)
        reports[name] = violations(vm[offset], loading, vm_before=vm_before)

    return _ranked(sweep, scores, reports, pinned)


def _with_slack(ppc, sampled_buses, gen_p: np.ndarray) -> np.ndarray:
    """Add the slack's dispatch, which the sweep does not store.

    Its output is decided by the solve, never fed in, so it keeps the case
    file's value - exactly what :func:`p013.dataset.load_dataset` does. The two
    must agree, or the model sees different inputs at evaluation than it saw in
    training.
    """
    from p013.admittance import GEN_BUS, GEN_PG, GEN_STATUS, bus_position

    pos = bus_position(ppc)
    sampled = set(sampled_buses)
    out = gen_p.copy()
    for row in ppc["gen"]:
        if row[GEN_STATUS] <= 0 or int(row[GEN_BUS]) in sampled:
            continue
        out[pos[int(row[GEN_BUS])]] += row[GEN_PG]
    return out


def _ranked(sweep: Sweep, scores, reports, pinned) -> PointRanking:
    """Pinned first, then descending score, ties in canonical branch order."""
    position = sweep._position
    pinned = sorted(pinned, key=position.__getitem__)
    rest = sorted(scores, key=lambda n: (-scores[n], position[n]))
    return PointRanking(
        order=tuple(pinned) + tuple(rest),
        score=scores,
        report=reports,
        pinned=tuple(pinned),
    )


# --------------------------------------------------------------------------
# The results file
# --------------------------------------------------------------------------

#: Fraction of operating points reserved for choosing k_deploy.
VALIDATION_FRACTION = 0.3

#: Safety margin added to the worst validation k*. A stated choice rather than
#: a tuned one: it is the only free parameter in the headline, so it is named
#: in the results file and can be argued with.
DEFAULT_MARGIN = 2

SPLIT_SEED = 20260911


def false_alarms(order: Sequence[str], dangerous: Iterable[str], k: int) -> int:
    """How many of the top ``k`` turned out to be safe. SPEC 3.4.

    Zero misses is bought with false alarms, and stating the price first is
    stronger than being asked for it.

    **Judged against the TRUTH's danger set, never the ranker's own opinion.**
    The first version read the set off the ranking being scored, which lets a
    ranker grade its own homework: one that called everything dangerous would
    have reported zero false alarms. It also happened to crash on the baseline
    rankings, which is the only reason it was caught.
    """
    dangerous = set(dangerous)
    return sum(1 for name in order[:k] if name not in dangerous)


def _score_orders(orders, truth_pairs, k_deploy) -> dict:
    """One row of SPEC 3.6's table: k*, recall, misses, false alarms.

    **Danger always comes from the truth**, never from the ranker being scored.
    Scoring a ranker against its own opinion measures nothing (SPEC 3.1).
    """
    per_point = [
        (order, dangerous) for order, (_, dangerous) in zip(orders, truth_pairs)
    ]
    report = measure_k_star(per_point, k_deploy=k_deploy)
    alarms = [
        false_alarms(order, dangerous, k_deploy)
        for order, (_, dangerous) in zip(orders, truth_pairs)
    ]
    return {
        "k_star_max": report.max,
        "k_star_median": report.median,
        "k_star_per_point": list(report.per_point),
        "recall_curve": [round(v, 6) for v in report.recall_curve],
        "misses": report.misses,
        "false_alarms": {
            "at_k_deploy": alarms,
            "mean_at_k_deploy": round(float(np.mean(alarms)), 3),
        },
    }


def _operating_point(sweep: Sweep, point: int):
    from p013.baselines import OperatingPoint

    base = sweep.row(point, 0)
    return OperatingPoint(
        load_p_mw=sweep.raw["load_p_mw"][base].copy(),
        load_q_mvar=sweep.raw["load_q_mvar"][base].copy(),
        gen_p_mw=sweep.raw["gen_p_mw"][base].copy(),
    )


def _predict_point(sweep: Sweep, point: int, weights: dict):
    """One batched forward pass over the intact state and all 80 outages."""
    base = sweep.row(point, 0)
    one = slice(base, base + 1)
    load_p = to_bus_order(sweep.ppc, sweep._load_bus, sweep.raw["load_p_mw"][one])[0]
    load_q = to_bus_order(sweep.ppc, sweep._load_bus, sweep.raw["load_q_mvar"][one])[0]
    gen_p = _with_slack(
        sweep.ppc,
        sweep._gen_bus,
        to_bus_order(sweep.ppc, sweep._gen_bus, sweep.raw["gen_p_mw"][one])[0],
    )
    bus_one = bus_features(sweep.ppc, load_p, load_q, gen_p)
    rows = sweep.rows_per_point
    bus_x = np.repeat(bus_one[None].astype(np.float32), rows, axis=0)
    branch_x = np.stack(
        [branch_features(sweep.ppc, sweep.in_service_at(point, o)) for o in range(rows)]
    ).astype(np.float32)
    return forward(
        weights,
        bus_x,
        branch_x,
        edge_index(sweep.ppc),
        slack_mask=bus_one[:, 4].astype(np.float32),
    )


def _accuracy(sweep: Sweep, weights, points, truth_pairs) -> dict:
    """SPEC 3.8, reported second by design rather than by omission.

    Reported twice - over all cases and over the dangerous subset alone.
    Accuracy where it matters is the number worth having.
    """
    everywhere: dict[str, list] = {"vm": [], "va": [], "flow": []}
    dangerous: dict[str, list] = {"vm": [], "va": [], "flow": []}

    for point, (_, dangerous_here) in zip(points, truth_pairs):
        vm, va = _predict_point(sweep, point, weights)
        for offset in range(1, sweep.rows_per_point):
            i = sweep.row(point, offset)
            if sweep.raw["status"][i] != "converged":
                continue
            in_service = sweep.in_service_at(point, offset)
            true_vm = sweep.raw["vm_pu"][i]
            true_va = np.deg2rad(sweep.raw["va_degree"][i])
            true_load = loading_percent(
                sweep.ppc, sweep.voltage_at(point, offset), in_service
            )
            pred_load = loading_percent(
                sweep.ppc, vm[offset] * np.exp(1j * va[offset]), in_service
            )

            buckets = [everywhere]
            if sweep.names[offset - 1] in dangerous_here:
                buckets.append(dangerous)
            for b in buckets:
                b["vm"].append(float(np.abs(vm[offset] - true_vm).mean()))
                b["va"].append(float(np.abs(np.rad2deg(va[offset] - true_va)).mean()))
                b["flow"].append(
                    float(np.abs(pred_load - true_load)[in_service].mean())
                )

    def summarise(b: dict) -> dict:
        if not b["vm"]:
            return {"n_cases": 0}
        return {
            "mae_vm_pu": round(float(np.mean(b["vm"])), 6),
            "mae_va_degrees": round(float(np.mean(b["va"])), 6),
            "mae_flow_percent_of_rating": round(float(np.mean(b["flow"])), 6),
            "n_cases": len(b["vm"]),
        }

    return {"all_cases": summarise(everywhere), "dangerous_only": summarise(dangerous)}


def _orders_for(points, make, every: int = 25):
    """Rankings for a list of points, keeping only what scoring needs.

    Returns the per-point ranking ORDER and nothing else. A full
    :class:`PointRanking` carries an 80-entry score dict and an 80-entry report
    dict, and holding six rankers x 175 points of those at once ran the machine
    out of memory - twice, with the OS killing the run. The scoring functions
    only ever read the order, so only the order is kept.
    """
    import gc

    orders = []
    for i, point in enumerate(points, 1):
        ranking = make(point)
        orders.append(ranking.order)
        del ranking
        if i % every == 0:
            gc.collect()
    gc.collect()
    return orders


def build_results(
    sweep: Sweep,
    weights: dict,
    *,
    validation: Sequence[int],
    test: Sequence[int],
    margin: int = DEFAULT_MARGIN,
    ablation_weights: dict | None = None,
    run_dc: bool = True,
    accuracy: bool = True,
    progress: bool = False,
) -> dict:
    """Every number the console and the results page read. SPEC 4.2.

    Built in one place so the two surfaces cannot disagree, and so no number
    can be quoted that was not measured here.
    """
    import datetime
    import gc

    from p013.baselines import dc_screen, naive_by_pre_outage_loading

    def say(message: str) -> None:
        if progress:
            print(f"  {message}", flush=True)

    # Truth first, and only the two things scoring needs from it: the order it
    # would rank in, and which contingencies are genuinely dangerous.
    say(f"truth over {len(validation)} validation + {len(test)} test points")
    validation_truth = []
    for point in validation:
        t = truth_ranking(sweep, point)
        validation_truth.append((t.order, t.dangerous))
        del t

    test_truth: list[tuple] = []
    pinned_per_point: list[int] = []
    dangerous_per_point: list[int] = []
    for point in test:
        t = truth_ranking(sweep, point)
        test_truth.append((t.order, t.dangerous))
        pinned_per_point.append(len(t.pinned))
        dangerous_per_point.append(len(t.dangerous))
        del t
    gc.collect()

    # k_deploy is chosen on VALIDATION ONLY, and from the MODEL's ranking of
    # those points - choosing it from the truth's ranking would pick a depth
    # the shipped tool cannot achieve, since at run time it has no truth.
    say("k_deploy from the validation split")
    validation_model = _orders_for(validation, lambda p: model_ranking(sweep, p, weights))
    k_deploy = choose_k_deploy(
        [(order, dangerous) for order, (_, dangerous) in zip(validation_model, validation_truth)],
        margin=margin,
    )
    del validation_model
    say(f"k_deploy = {k_deploy}")

    makers = {
        "model_predicted": lambda p: model_ranking(sweep, p, weights),
        "model_measured": lambda p: model_ranking(sweep, p, weights, pre_outage="measured"),
        "naive_pre_outage_loading": lambda p: naive_by_pre_outage_loading(
            _operating_point(sweep, p)
        ),
    }
    if run_dc:
        makers["dc_screen"] = lambda p: dc_screen(_operating_point(sweep, p))
    if ablation_weights is not None:
        makers["model_no_physics"] = lambda p: model_ranking(sweep, p, ablation_weights)

    # The full AC sweep IS the truth, so its ranking is already in hand and
    # re-solving it would only re-derive numbers we have.
    scored = {
        "full_ac_sweep": _score_orders([o for o, _ in test_truth], test_truth, k_deploy)
    }
    for name, make in makers.items():
        say(f"{name} over {len(test)} test points")
        scored[name] = _score_orders(_orders_for(test, make), test_truth, k_deploy)
        gc.collect()

    status = sweep.raw["status"]
    results = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "case": "pglib_opf_case57_ieee",
        "n_contingencies": sweep.n_branch,
        "margin": margin,
        "k_deploy": k_deploy,
        "split": {
            "validation_points": len(validation),
            "test_points": len(test),
            "seed": SPLIT_SEED,
            "disjoint": not (set(validation) & set(test)),
        },
        "danger_criterion": "risk_score > 1.0, SPEC 3.1 amended 2026-09-11",
        "pinned": {
            "islanded_rows": int((status == "islanded").sum()),
            "no_solution_rows": int((status == "no_solution").sum()),
            "model_was_consulted": False,
            "mean_pinned_per_test_point": round(float(np.mean(pinned_per_point)), 3),
        },
        "dangerous_per_test_point": {
            "mean": round(float(np.mean(dangerous_per_point)), 3),
            "max": int(max(dangerous_per_point)),
        },
        "rankers": scored,
    }
    if accuracy:
        say("model accuracy")
        results["model_accuracy"] = _accuracy(sweep, weights, test, test_truth)
    return results


def measure_timing(sweep: Sweep, weights: dict, k_deploy: int, points: Sequence[int]) -> dict:
    """SPEC 3.5. Same CPU, same process, one-off costs excluded.

    | Path | Timed work |
    | Ours | pre-filter over all N + one batched forward pass + AC solve of top k_deploy |
    | Baseline | AC solve of all N |

    ``k_deploy``, never ``k*`` - timing against ``k*`` would price the tool
    using an answer it does not have at run time and would understate its cost.

    **The pre-filter and the forward pass are counted against us**, not waved
    away as negligible, and the islanding filter is deliberately called through
    a path that does not use its cache: a real screen at a new topology pays
    for it.
    """
    import time

    from p013.baselines import apply_operating_point
    from p013.case_loader import CaseTemplate
    from p013.solving import solve
    from p013.topology import islanding_contingencies

    template = CaseTemplate()
    # Warm up numba before timing anything: its one-off JIT compile is a
    # start-up cost, and SPEC 3.5 excludes one-off costs from both paths.
    solve(template.fresh())
    _predict_point(sweep, points[0], weights)

    ours, baseline = [], []
    for point in points:
        op = _operating_point(sweep, point)

        started = time.perf_counter()
        net = template.fresh()
        apply_operating_point(net, op)
        # The pre-filter is called through islanding_contingencies directly
        # rather than the cached pinned_contingencies: a real screen at a new
        # topology pays for it, and SPEC 3.5 counts it against us.
        islanding_contingencies(net)
        # ONE forward pass. The first version called _predict_point here and
        # then model_ranking, which predicts again - the screening path was
        # billed for two forward passes and the speedup came out too low.
        ranked = model_ranking(sweep, point, weights)
        for name in ranked.order[:k_deploy]:
            trial = template.fresh()
            apply_operating_point(trial, op)
            from p013.baselines import take_out_of_service

            take_out_of_service(trial, name)
            solve(trial)
        ours.append(time.perf_counter() - started)

        started = time.perf_counter()
        for name in sweep.names:
            trial = template.fresh()
            apply_operating_point(trial, op)
            from p013.baselines import take_out_of_service

            take_out_of_service(trial, name)
            solve(trial)
        baseline.append(time.perf_counter() - started)

    ours_s = float(np.median(ours))
    base_s = float(np.median(baseline))
    return {
        "n_points_timed": len(points),
        "ours_seconds_median": round(ours_s, 4),
        "baseline_full_ac_sweep_seconds_median": round(base_s, 4),
        "speedup": round(base_s / ours_s, 2) if ours_s else None,
        "k_deploy_solved": k_deploy,
        "note": (
            "Same CPU, same process. Model loading and network construction "
            "excluded as one-off; the connectivity pre-filter and the forward "
            "pass are counted against us. pandapower is Python and a production "
            "EMS runs compiled code, so the RATIO is the claim, not the "
            "milliseconds."
        ),
    }


def main() -> None:
    import argparse
    import json

    from p013.forward_numpy import load_weights

    ap = argparse.ArgumentParser(description="Measure k*, k_deploy and the baselines")
    ap.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--ablation-weights", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("eval/results.json"))
    ap.add_argument("--margin", type=int, default=DEFAULT_MARGIN)
    ap.add_argument("--points", type=int, default=None, help="use only the first N points")
    ap.add_argument("--skip-dc", action="store_true")
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--timing-points", type=int, default=5)
    args = ap.parse_args()

    sweep = Sweep(args.sweep)
    weights = load_weights(args.weights)
    ablation = load_weights(args.ablation_weights) if args.ablation_weights else None

    n_points = min(args.points, sweep.n_points) if args.points else sweep.n_points
    validation, test = split_points(n_points, VALIDATION_FRACTION, SPLIT_SEED)
    print(
        f"{n_points} operating points: {len(validation)} validation, {len(test)} test",
        flush=True,
    )

    results = build_results(
        sweep,
        weights,
        validation=validation,
        test=test,
        margin=args.margin,
        ablation_weights=ablation,
        run_dc=not args.skip_dc,
        progress=True,
    )

    if not args.skip_timing:
        print("timing...", flush=True)
        results["timing"] = measure_timing(
            sweep, weights, results["k_deploy"], test[: args.timing_points]
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"\nk_deploy = {results['k_deploy']} of {results['n_contingencies']}")
    print(f"{'ranker':<28} {'k* max':>7} {'k* med':>7} {'misses':>7} {'false alarms':>13}")
    for name, r in results["rankers"].items():
        print(
            f"{name:<28} {r['k_star_max']:>7} {r['k_star_median']:>7} "
            f"{len(r['misses']):>7} {r['false_alarms']['mean_at_k_deploy']:>13}"
        )
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
