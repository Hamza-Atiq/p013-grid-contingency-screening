"""The two datasets. SPEC 2.4 and 3.3."""
import numpy as np

from p013.case_loader import CaseInvariants
from p013.generate import ROWS_PER_POINT, generate_test_sweep, generate_training_set


def test_training_set_is_reproducible(tmp_path):
    """SPEC 4.3: generate.py plus a fixed seed regenerates the data, which is a
    stronger reproducibility claim than shipping a tarball. It only holds if
    the seed really determines everything."""
    a = generate_training_set(n=40, seed=11, out_path=tmp_path / "a.npz")
    b = generate_training_set(n=40, seed=11, out_path=tmp_path / "b.npz")
    assert np.allclose(a["vm_pu"], b["vm_pu"], equal_nan=True)
    assert (a["status"] == b["status"]).all()
    assert (a["in_service"] == b["in_service"]).all()


def test_a_different_seed_gives_different_data(tmp_path):
    a = generate_training_set(n=40, seed=11, out_path=tmp_path / "a.npz")
    b = generate_training_set(n=40, seed=12, out_path=tmp_path / "b.npz")
    assert not np.allclose(a["load_p_mw"], b["load_p_mw"], equal_nan=True)


def test_failed_solves_are_kept_and_labelled_not_deleted(tmp_path):
    """SPEC 2.4: deleting them would make our results look better than they are."""
    d = generate_training_set(n=150, seed=3, out_path=tmp_path / "t.npz")
    assert set(np.unique(d["status"]).tolist()) <= {"converged", "islanded", "no_solution"}
    assert (d["status"] == "converged").sum() > 0
    assert len(d["status"]) == 150


def test_unsolved_rows_carry_nan_not_zero(tmp_path):
    """A zero voltage is a plausible-looking number. Training on it would
    teach the model that a failed case has a flat profile."""
    d = generate_training_set(n=150, seed=3, out_path=tmp_path / "t.npz")
    failed = d["status"] != "converged"
    if failed.any():
        assert np.isnan(d["vm_pu"][failed]).all()


def test_converged_rows_carry_real_voltages(tmp_path):
    d = generate_training_set(n=60, seed=5, out_path=tmp_path / "t.npz")
    ok = d["status"] == "converged"
    assert np.isfinite(d["vm_pu"][ok]).all()
    assert (d["vm_pu"][ok] > 0.3).all(), "a voltage below 0.3 pu is not a solved state"


def test_in_service_flags_match_the_outage_depth(tmp_path):
    """The flags are the model's input for which branches are out. If they
    disagreed with what was actually solved, every sample would be mislabelled."""
    d = generate_training_set(n=100, seed=9, out_path=tmp_path / "t.npz")
    out_counts = (~d["in_service"]).sum(axis=1)
    assert out_counts.min() == 0
    assert out_counts.max() <= 3


def test_test_sweep_solves_every_contingency_at_each_point(tmp_path):
    from p013.generate import ROWS_PER_POINT

    d = generate_test_sweep(n_points=3, seed=5, out_path=tmp_path / "s.npz")
    assert d["vm_pu"].shape[0] == 3 * ROWS_PER_POINT
    assert int(d["operating_point_id"].max()) == 2
    for point in range(3):
        rows = np.flatnonzero(d["operating_point_id"] == point)
        assert len(rows) == ROWS_PER_POINT
        out_counts = (~d["in_service"][rows]).sum(axis=1)
        # offset 0 is the pre-outage state, the rest are one contingency each
        assert out_counts[0] == 0
        assert (out_counts[1:] == 1).all()
        assert (~d["in_service"][rows[1:]]).sum(axis=0).max() == 1


def test_sweep_carries_the_pre_outage_state(tmp_path):
    """SPEC 3.1, amended: a voltage violation is a bus outside the band AND
    worse than before the outage, because 5 of 57 buses start outside it.
    Scoring deterioration needs the N-0 voltages at the same operating point,
    and nothing else in the dataset carries them."""
    from p013.generate import ROWS_PER_POINT

    d = generate_test_sweep(n_points=2, seed=7, out_path=tmp_path / "s.npz")
    for point in range(2):
        rows = np.flatnonzero(d["operating_point_id"] == point)
        base = rows[0]
        assert d["in_service"][base].all(), "the N-0 row must have nothing out"
        assert d["status"][base] == "converged"
        assert np.isfinite(d["vm_pu"][base]).all()
        # and it shares the operating point with its 80 contingencies
        assert len(set(d["operating_point_hash"][rows].tolist())) == 1


def test_pre_outage_row_is_the_intact_answer(tmp_path):
    """It must actually be the N-0 solve, not a copy of a contingency."""
    from p013.generate import ROWS_PER_POINT

    d = generate_test_sweep(n_points=1, seed=9, out_path=tmp_path / "s.npz")
    base_vm = d["vm_pu"][0]
    differing = sum(
        1 for r in range(1, ROWS_PER_POINT)
        if d["status"][r] == "converged" and not np.allclose(d["vm_pu"][r], base_vm)
    )
    assert differing > 50, "taking a branch out should move the voltages"


def test_every_row_of_one_sweep_point_shares_one_operating_point(tmp_path):
    """The reason the sweep exists (SPEC 3.3): k* needs the true label for
    every contingency at the SAME condition."""
    d = generate_test_sweep(n_points=2, seed=6, out_path=tmp_path / "s.npz")
    for point in range(2):
        rows = d["operating_point_id"] == point
        assert len(set(d["operating_point_hash"][rows].tolist())) == 1
        loads = d["load_p_mw"][rows]
        assert np.allclose(loads, loads[0])


def test_splits_never_share_an_operating_point(tmp_path):
    """SPEC 2.4: split by operating condition, not randomly."""
    train = generate_training_set(n=60, seed=1, out_path=tmp_path / "tr.npz")
    sweep = generate_test_sweep(n_points=3, seed=2, out_path=tmp_path / "te.npz")
    assert set(train["operating_point_hash"].tolist()).isdisjoint(
        set(sweep["operating_point_hash"].tolist())
    )


def test_saved_file_round_trips(tmp_path):
    path = tmp_path / "t.npz"
    d = generate_training_set(n=20, seed=2, out_path=path)
    loaded = np.load(path, allow_pickle=False)
    assert np.allclose(loaded["vm_pu"], d["vm_pu"], equal_nan=True)
    assert (loaded["status"] == d["status"]).all()


def test_dataset_carries_everything_needed_to_rebuild_the_model_input(tmp_path):
    """The dataset must be self-sufficient. It originally stored load P but not
    Q, which the model needs (SPEC 2.1). Q was recoverable only because the
    sampler happens to scale P and Q by the same factor and no load has zero P
    - an implementation detail of sampling.py, not a contract."""
    d = generate_training_set(n=10, seed=4, out_path=tmp_path / "t.npz")
    for key in ("load_p_mw", "load_q_mvar", "in_service", "vm_pu", "va_degree", "status"):
        assert key in d, f"{key} missing from the dataset"
    assert d["load_p_mw"].shape == d["load_q_mvar"].shape
    assert d["load_q_mvar"].shape[1] == 42


def test_load_q_varies_with_the_operating_point(tmp_path):
    d = generate_training_set(n=12, seed=8, out_path=tmp_path / "t.npz")
    assert not np.allclose(d["load_q_mvar"][0], d["load_q_mvar"][1])


def test_power_factor_is_preserved_in_the_stored_data(tmp_path):
    """P and Q scale together, so the ratio is constant across samples."""
    d = generate_training_set(n=8, seed=13, out_path=tmp_path / "t.npz")
    ratios = d["load_q_mvar"] / d["load_p_mw"]
    assert np.allclose(ratios, ratios[0])


def test_the_sampled_generation_is_stored(tmp_path):
    """``sample_operating_point`` scales generation as well as load. If only
    the load is stored the dataset cannot rebuild the state it solved.

    The same argument ``_blank`` already makes for load Q: recoverability
    through an implementation detail of sampling.py is not a contract. For
    generation it is not even recoverable - the common factor can only be
    ESTIMATED back out of the per-bus-jittered loads.
    """
    out = tmp_path / "train.npz"
    generate_training_set(n=12, seed=5, out_path=out)
    raw = np.load(out, allow_pickle=False)

    assert "gen_p_mw" in raw, "generation is not stored"
    assert raw["gen_p_mw"].shape[0] == 12
    assert np.isfinite(raw["gen_p_mw"]).all()
    assert raw["gen_p_mw"].std(axis=0).max() > 0, "generation never varies between samples"


def test_the_sweep_stores_generation_too(tmp_path):
    out = tmp_path / "sweep.npz"
    generate_test_sweep(n_points=2, seed=5, out_path=out)
    raw = np.load(out, allow_pickle=False)

    assert "gen_p_mw" in raw
    assert raw["gen_p_mw"].shape[0] == 2 * ROWS_PER_POINT
    # every row of one operating point shares that point's generation
    per_point = raw["gen_p_mw"][:ROWS_PER_POINT]
    assert np.allclose(per_point, per_point[0])
