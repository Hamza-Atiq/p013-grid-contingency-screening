"""k_deploy, k*, and the results file. SPEC 3.3.

Two quantities that were one until 2026-09-09, and the separation is the whole
point:

- **k_deploy is a CHOICE**, fixed in advance from a validation split and baked
  into the shipped tool.
- **k\\* is a MEASUREMENT**, taken on a test split that k_deploy was never
  chosen from.

Choosing the shortlist on the same data you then report zero misses on proves
nothing, and it is the kind of circularity a reviewer enjoys finding.

The headline is the **maximum** k* across the test set, never the mean: a
screening tool is judged on its worst day.
"""
from pathlib import Path

import numpy as np
import pytest

from p013.evaluate import (
    DEFAULT_SWEEP,
    DEFAULT_WEIGHTS,
    KStarReport,
    Sweep,
    choose_k_deploy,
    k_star,
    measure_k_star,
    model_ranking,
    split_points,
    truth_ranking,
)

#: The sweep and the weights are generated artefacts and gitignored, so a fresh
#: clone has neither. Skipping is honest; inventing a stand-in would test the
#: stand-in.
needs_data = pytest.mark.skipif(
    not (Path(DEFAULT_SWEEP).exists() and Path(DEFAULT_WEIGHTS).exists()),
    reason="run p013.generate and p013.train first",
)

NAMES = tuple(f"B{i:02d}" for i in range(10))


def ranking(*order: str) -> tuple[str, ...]:
    """A ranking of the ten toy branches, most dangerous first."""
    rest = [n for n in NAMES if n not in order]
    return tuple(order) + tuple(rest)


# --------------------------------------------------------------------------
# k* itself
# --------------------------------------------------------------------------

def test_k_star_is_the_deepest_a_dangerous_case_hid():
    """Dangerous cases at positions 1 and 4 -> you must verify 4 to catch both.
    Not 2, which is how many there are, and not 10."""
    order = ranking("B00", "B05", "B06", "B01")
    assert k_star(order, dangerous={"B00", "B01"}) == 4


def test_k_star_is_one_when_the_only_danger_is_ranked_first():
    assert k_star(ranking("B03"), dangerous={"B03"}) == 1


def test_k_star_is_zero_when_nothing_is_dangerous():
    """No dangerous case means no shortlist is needed to catch them all. It is
    not the same as 'we did not look', and it must not be silently the whole
    list."""
    assert k_star(ranking(), dangerous=set()) == 0


def test_k_star_is_the_whole_list_when_the_worst_case_ranks_last():
    order = ranking()
    assert k_star(order, dangerous={order[-1]}) == len(NAMES)


def test_k_star_rejects_a_danger_that_is_not_in_the_ranking():
    """A dangerous case missing from the ranking would silently lower k*, which
    is the most flattering possible bug."""
    with pytest.raises(ValueError):
        k_star(ranking(), dangerous={"NOT_A_BRANCH"})


# --------------------------------------------------------------------------
# The headline is the maximum
# --------------------------------------------------------------------------

def test_the_headline_is_the_maximum_not_the_mean():
    """SPEC 3.3. Nine easy points and one hard one: the report leads with the
    hard one."""
    per_point = [
        (ranking("B00"), {"B00"}),           # k* = 1, nine times
    ] * 9 + [
        (ranking("B00", "B01", "B02", "B03", "B04", "B05", "B06"), {"B06"}),  # k* = 7
    ]
    report = measure_k_star(per_point)
    assert isinstance(report, KStarReport)
    assert report.max == 7
    assert report.median == 1
    assert report.max != int(np.mean([1] * 9 + [7]))


def test_the_recall_curve_is_the_full_picture():
    """Recall at k = the fraction of dangerous cases inside the top k, across
    every point. It must reach 1.0 exactly at the maximum k*."""
    per_point = [
        (ranking("B00", "B01"), {"B00", "B01"}),
        (ranking("B00", "B01", "B02"), {"B02"}),
    ]
    report = measure_k_star(per_point)
    assert report.recall_curve[0] == pytest.approx(1 / 3)
    assert report.recall_curve[report.max - 1] == pytest.approx(1.0)
    assert len(report.recall_curve) == len(NAMES)


# --------------------------------------------------------------------------
# k_deploy is a choice, made on data k* is never measured on
# --------------------------------------------------------------------------

def test_k_deploy_is_chosen_on_validation_only():
    """Hand it a test split it has never seen; the answer must not move."""
    validation = [(ranking("B00", "B01"), {"B01"})]          # k* = 2
    chosen = choose_k_deploy(validation, margin=1)

    catastrophic = [(ranking(), {ranking()[-1]})]            # k* = 10
    assert choose_k_deploy(validation, margin=1) == chosen
    assert chosen < k_star(*catastrophic[0][:1], dangerous=catastrophic[0][1])


def test_k_deploy_is_the_worst_validation_case_plus_the_margin():
    validation = [
        (ranking("B00"), {"B00"}),                            # 1
        (ranking("B00", "B01", "B02"), {"B02"}),              # 3
    ]
    assert choose_k_deploy(validation, margin=2) == 5


def test_k_deploy_never_exceeds_the_number_of_contingencies():
    """A margin that runs off the end is meaningless: verifying 'the top 95 of
    80' is verifying everything, and it should say so."""
    validation = [(ranking(), {ranking()[-1]})]               # k* = 10 of 10
    assert choose_k_deploy(validation, margin=15) == len(NAMES)


# --------------------------------------------------------------------------
# A miss is reported as a miss
# --------------------------------------------------------------------------

def test_a_miss_is_reported_as_a_miss():
    """SPEC 6: k_deploy is NOT enlarged after the fact to make a miss vanish.
    The report names the point and how deep the case actually was."""
    per_point = [
        (ranking("B00"), {"B00"}),                            # k* = 1, inside
        (ranking("B00", "B01", "B02", "B03"), {"B03"}),       # k* = 4, outside
    ]
    report = measure_k_star(per_point, k_deploy=2)
    assert len(report.misses) == 1
    miss = report.misses[0]
    assert miss["point"] == 1
    assert miss["k_star"] == 4
    assert miss["k_deploy"] == 2
    assert "B03" in miss["missed"]


def test_no_misses_when_every_point_fits_inside_k_deploy():
    per_point = [(ranking("B00"), {"B00"})]
    assert measure_k_star(per_point, k_deploy=3).misses == []


def test_measuring_without_a_k_deploy_reports_no_misses_rather_than_guessing():
    """k* alone cannot say whether anything was missed - that needs the choice
    k_deploy, and inventing one here would be the circularity SPEC 3.3 forbids."""
    assert measure_k_star([(ranking("B00"), {"B00"})]).misses == []


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

def test_validation_and_test_points_are_disjoint():
    validation, test = split_points(250, validation_fraction=0.3, seed=1)
    assert not (set(validation) & set(test))
    assert len(validation) + len(test) == 250
    assert len(validation) == 75


def test_the_split_is_reproducible():
    assert split_points(250, 0.3, seed=1) == split_points(250, 0.3, seed=1)


def test_the_split_is_by_whole_operating_point():
    """SPEC 2.4: a point's 81 rows must never straddle the boundary, or the
    pre-outage state of a test point leaks into validation."""
    validation, test = split_points(10, 0.5, seed=3)
    assert sorted(validation + test) == list(range(10))


# --------------------------------------------------------------------------
# Truth, read from the solved sweep
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sweep():
    return Sweep(Path(DEFAULT_SWEEP))


@needs_data
def test_the_sweep_has_the_shape_the_evaluation_assumes(sweep):
    assert sweep.rows_per_point == 81
    assert sweep.n_branch == 80
    assert sweep.n_points == 250
    assert len(sweep.names) == 80


@needs_data
def test_the_pre_outage_row_is_the_intact_state(sweep):
    """Offset 0 of every point. If this drifts, every deterioration in the
    project is measured against the wrong baseline."""
    for i in (0, 7, 249):
        assert sweep.in_service_at(i, 0).all()
        for offset in range(1, 81):
            out = ~sweep.in_service_at(i, offset)
            assert out.sum() == 1
            assert sweep.names[offset - 1] == sweep.names[int(np.flatnonzero(out)[0])]


@needs_data
def test_a_failed_contingency_is_pinned_and_never_scored(sweep):
    """SPEC 2.4 and 2.5: no answer from the exact solver is not a safe answer.
    L40 islands at every point and L42 fails at most of them."""
    truth = truth_ranking(sweep, 0)
    assert truth.pinned
    for name in truth.pinned:
        assert name not in truth.score
        assert name not in truth.report
    assert set(truth.order[: len(truth.pinned)]) == set(truth.pinned)


@needs_data
def test_truth_matches_an_independent_recomputation(sweep):
    """Recompute one point from the raw arrays without going through the
    module, so the module cannot define its own correctness."""
    import p013
    from p013.admittance import loading_percent
    from p013.case_loader import parse_matpower
    from p013.danger import risk_score

    ppc = parse_matpower(p013.CASE_PATH)
    raw = np.load(DEFAULT_SWEEP, allow_pickle=False)
    truth = truth_ranking(sweep, 3)

    base = 3 * 81
    vm_before = raw["vm_pu"][base]
    checked = 0
    for offset in range(1, 81):
        row = base + offset
        if raw["status"][row] != "converged":
            continue
        v = raw["vm_pu"][row] * np.exp(1j * np.deg2rad(raw["va_degree"][row]))
        expected = risk_score(
            raw["vm_pu"][row],
            loading_percent(ppc, v, raw["in_service"][row]),
            vm_before=vm_before,
        )
        assert truth.score[sweep.names[offset - 1]] == pytest.approx(expected, abs=1e-12)
        checked += 1
    assert checked > 70


# --------------------------------------------------------------------------
# The model's ranking
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def weights():
    from p013.forward_numpy import load_weights

    return load_weights(Path(DEFAULT_WEIGHTS))


@needs_data
def test_the_model_ranking_covers_every_contingency(sweep, weights):
    ranked = model_ranking(sweep, 0, weights)
    assert len(ranked.order) == 80
    assert set(ranked.order) == set(sweep.names)


@needs_data
def test_the_model_is_never_consulted_about_a_pinned_case(sweep, weights):
    """SPEC 2.5. Islanding is settled by exact graph code; the model would
    happily emit voltages for a network that has fallen in half."""
    truth = truth_ranking(sweep, 0)
    ranked = model_ranking(sweep, 0, weights)
    assert ranked.pinned == truth.pinned
    for name in ranked.pinned:
        assert name not in ranked.score


@needs_data
def test_the_model_uses_its_own_pre_outage_prediction_by_default(sweep, weights):
    """At screening time there is no solver, so there is no solved pre-outage
    state either. The default must not quietly borrow one - SPEC 3.5's timed
    path is the pre-filter plus one forward pass, with no solve in it.

    The 'measured' variant is reported alongside, because a real control room
    does know its present state from state estimation. They are different
    claims and the results file carries both.
    """
    predicted = model_ranking(sweep, 0, weights)
    measured = model_ranking(sweep, 0, weights, pre_outage="measured")
    assert predicted.score != measured.score


@needs_data
def test_the_model_scores_are_finite_and_spread(sweep, weights):
    """A constant score is the saturation failure the mockup taught us; a NaN
    would sort arbitrarily and silently corrupt k*."""
    values = np.array(list(model_ranking(sweep, 0, weights).score.values()))
    assert np.isfinite(values).all()
    assert values.max() - values.min() > 0.01


# --------------------------------------------------------------------------
# The results file, the one place every public number is authored
# --------------------------------------------------------------------------

@needs_data
def test_results_carry_every_field_the_two_surfaces_read(sweep, weights):
    """SPEC 4.2: the console and the results page both READ this file and
    neither computes its own numbers. A missing field is a blank on a page."""
    from p013.evaluate import build_results

    results = build_results(
        sweep, weights, validation=[0, 1], test=[2, 3, 4], margin=2, run_dc=False
    )

    for field in ("k_deploy", "margin", "n_contingencies", "split", "rankers", "generated"):
        assert field in results, f"missing top-level field {field!r}"

    ours = results["rankers"]["model_predicted"]
    for field in ("k_star_max", "k_star_median", "recall_curve", "misses", "false_alarms"):
        assert field in ours, f"missing ranker field {field!r}"

    assert results["split"]["validation_points"] == 2
    assert results["split"]["test_points"] == 3
    assert results["n_contingencies"] == 80


@needs_data
def test_k_deploy_in_the_results_comes_from_the_validation_split(sweep, weights):
    """The circularity SPEC 3.3 forbids: k_deploy must not have seen the test
    points it is later judged on."""
    from p013.evaluate import build_results

    a = build_results(sweep, weights, validation=[0, 1], test=[2, 3], margin=1, run_dc=False)
    b = build_results(sweep, weights, validation=[0, 1], test=[7, 8], margin=1, run_dc=False)
    assert a["k_deploy"] == b["k_deploy"]


@needs_data
def test_the_results_report_truth_and_the_model_over_the_same_points(sweep, weights):
    """Comparing rankers scored on different points would prove nothing."""
    from p013.evaluate import build_results

    results = build_results(
        sweep, weights, validation=[0], test=[1, 2, 3], margin=1, run_dc=False
    )
    lengths = {
        name: len(r["k_star_per_point"]) for name, r in results["rankers"].items()
    }
    assert len(set(lengths.values())) == 1, lengths


@needs_data
def test_the_full_ac_sweep_is_the_ceiling(sweep, weights):
    """It ranks by the truth, so no other ranker can beat its k*. If one does,
    the ranking and the label have drifted apart."""
    from p013.evaluate import build_results

    results = build_results(
        sweep, weights, validation=[0], test=[1, 2, 3, 4, 5], margin=1, run_dc=False
    )
    ceiling = results["rankers"]["full_ac_sweep"]["k_star_max"]
    for name, r in results["rankers"].items():
        assert r["k_star_max"] >= ceiling, f"{name} beat the truth: {r['k_star_max']} < {ceiling}"


@needs_data
def test_excluded_rows_are_counted_not_hidden(sweep, weights):
    """SPEC 3.9: every pinned case and every non-convergence is counted."""
    from p013.evaluate import build_results

    results = build_results(
        sweep, weights, validation=[0], test=[1, 2], margin=1, run_dc=False
    )
    pinned = results["pinned"]
    assert pinned["islanded_rows"] + pinned["no_solution_rows"] > 0
    assert pinned["model_was_consulted"] is False
