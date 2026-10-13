"""Build the datasets. SPEC 2.4 and 3.3.

Two datasets, built differently, and that is not optional.

**Training** solves one sampled topology per operating point. Cheap, and right
for learning the mapping.

**Test** is a separate full sweep: every one of the 80 contingencies at the
*same* operating point. ``k*`` cannot be measured any other way - it needs the
true danger label for every contingency at one operating point, and one random
outage per point cannot produce that label set (SPEC 3.3).

**Splits never share an operating point.** SPEC 2.4: split by operating
condition, not randomly. If samples from one condition land in both train and
test, the test score comes out flattering and means nothing.

Budget, measured on this machine 2026-09-11: **76 ms per sample** end to end,
so 20,000 training samples take about 25 minutes and the test sweep about the
same. Note this is not the spec's 43 ms - that figure is the power flow alone,
and a sample also costs a network reset, an islanding check and the outage
bookkeeping.
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

from p013.case_loader import CaseInvariants, CaseTemplate, load_case
from p013.naming import branch_names, set_in_service
from p013.sampling import sample_operating_point, sample_outages
from p013.solving import solve

N_BUS = CaseInvariants.N_BUS
N_BRANCH = CaseInvariants.N_BRANCH

#: Branch order is fixed by the case, so position is computed once rather than
#: rebuilt per row. Rebuilding it walks two pandas frames with ``iterrows`` and
#: would cost more than the power flow it accompanies.
_NAMES = branch_names(load_case())
_POSITION = {name: pos for pos, name in enumerate(_NAMES)}


def _blank(n: int) -> dict:
    return {
        "vm_pu": np.full((n, N_BUS), np.nan),
        "va_degree": np.full((n, N_BUS), np.nan),
        "loading_percent": np.full((n, N_BRANCH), np.nan),
        "in_service": np.ones((n, N_BRANCH), dtype=bool),
        # Both P and Q are stored explicitly. Q is technically recoverable from
        # P today, because the sampler scales them by the same factor and no
        # load has zero P - but that is an implementation detail of
        # sampling.py, not a contract, and this data feeds every later phase.
        "load_p_mw": np.full((n, 0), np.nan),  # replaced on first write
        "load_q_mvar": np.full((n, 0), np.nan),
        # Generation, for the same reason as load Q and a stronger one: the
        # sampler scales net.gen.p_mw by the common load factor, and that
        # factor can only be ESTIMATED back out of the per-bus-jittered loads,
        # never recovered. Added 2026-09-11 after measuring a 2.3 pu power
        # mismatch at the PV buses of the first 20,000-sample set - the
        # features described the base dispatch while the solve used the scaled
        # one, and both the model inputs and the physics term were wrong.
        "gen_p_mw": np.full((n, 0), np.nan),
        "status": np.empty(n, dtype="<U12"),
        "operating_point_id": np.zeros(n, dtype=np.int32),
        "operating_point_hash": np.empty(n, dtype="<U16"),
    }


def _operating_point_hash(net) -> str:
    """Identifies the operating condition, so splits can be checked disjoint."""
    payload = net.load.p_mw.to_numpy().tobytes() + net.load.q_mvar.to_numpy().tobytes()
    return hashlib.sha1(payload).hexdigest()[:16]


def _record(data, row, result, outages, point_id, point_hash) -> None:
    data["status"][row] = result.status
    data["operating_point_id"][row] = point_id
    data["operating_point_hash"][row] = point_hash

    flags = np.ones(N_BRANCH, dtype=bool)
    for name in outages:
        flags[_POSITION[name]] = False
    data["in_service"][row] = flags

    if result.status == "converged":
        data["vm_pu"][row] = result.vm_pu[:N_BUS]
        data["va_degree"][row] = result.va_degree[:N_BUS]
        data["loading_percent"][row] = result.loading_percent[:N_BRANCH]


def _finish(data: dict, out_path: Path, seconds: float) -> dict:
    data = {k: v for k, v in data.items() if v.size or v.ndim == 1}
    data["seconds"] = np.array([seconds])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)
    return data


def generate_training_set(n: int, seed: int, out_path: Path, progress: int = 0) -> dict:
    """One sampled topology per operating point."""
    rng = np.random.default_rng(seed)
    template = CaseTemplate()
    data = _blank(n)
    reference = template.fresh()
    n_load = len(reference.load)
    n_gen = len(reference.gen)
    gen_bus = reference.gen.bus.to_numpy().astype(np.int32)
    loads_p = np.full((n, n_load), np.nan)
    loads_q = np.full((n, n_load), np.nan)
    gens_p = np.full((n, n_gen), np.nan)

    started = time.perf_counter()
    for row in range(n):
        net = template.fresh()
        sample_operating_point(net, rng)
        loads_p[row] = net.load.p_mw.to_numpy()
        loads_q[row] = net.load.q_mvar.to_numpy()
        gens_p[row] = net.gen.p_mw.to_numpy()
        point_hash = _operating_point_hash(net)

        outages = sample_outages(net, rng)
        for name in outages:
            set_in_service(net, name, False)

        _record(data, row, solve(net), outages, row, point_hash)

        if progress and row and row % progress == 0:
            rate = (time.perf_counter() - started) / row * 1000
            print(f"  train {row:>6}/{n}  {rate:5.1f} ms/sample", flush=True)

    data["load_p_mw"] = loads_p
    data["load_q_mvar"] = loads_q
    data["gen_p_mw"] = gens_p
    data["gen_bus"] = gen_bus
    return _finish(data, out_path, time.perf_counter() - started)


#: Each operating point contributes the 80 contingencies plus the N-0 state.
ROWS_PER_POINT = N_BRANCH + 1


def generate_test_sweep(n_points: int, seed: int, out_path: Path, progress: int = 0) -> dict:
    """Every contingency at each operating point, plus the pre-outage state.

    The N-0 row is not decoration. SPEC 3.1, amended 2026-09-10, defines a
    voltage violation as a bus outside 0.95-1.05 pu **and worse than it was
    before the outage** - because 5 of 57 buses already sit outside the band in
    the intact case, so an absolute test marks every contingency dangerous and
    the headline metric degenerates. Scoring deterioration requires the
    pre-outage voltages at the *same* operating point, which nothing else in
    the dataset carries.

    It is marked by an all-true ``in_service`` row, and costs 250 extra solves.
    """
    rng = np.random.default_rng(seed)
    template = CaseTemplate()
    data = _blank(n_points * ROWS_PER_POINT)
    reference = template.fresh()
    n_load = len(reference.load)
    n_gen = len(reference.gen)
    gen_bus = reference.gen.bus.to_numpy().astype(np.int32)
    loads_p = np.full((n_points * ROWS_PER_POINT, n_load), np.nan)
    loads_q = np.full((n_points * ROWS_PER_POINT, n_load), np.nan)
    gens_p = np.full((n_points * ROWS_PER_POINT, n_gen), np.nan)

    started = time.perf_counter()
    row = 0
    for point_id in range(n_points):
        # Draw the operating point once, then re-apply it for every contingency
        # so all 80 share exactly one condition. That is what makes k*
        # measurable at all.
        scratch = template.fresh()
        sample_operating_point(scratch, rng)
        point_p = scratch.load.p_mw.to_numpy().copy()
        point_q = scratch.load.q_mvar.to_numpy().copy()
        point_gen = scratch.gen.p_mw.to_numpy().copy()
        point_hash = _operating_point_hash(scratch)

        # The pre-outage state first, so row index within a point is stable:
        # offset 0 is N-0, offsets 1..80 are the contingencies in canonical
        # order.
        for outage in [[]] + [[name] for name in _NAMES]:
            net = template.fresh()
            net.load["p_mw"] = point_p.copy()
            net.load["q_mvar"] = point_q.copy()
            net.gen["p_mw"] = point_gen.copy()
            for name in outage:
                set_in_service(net, name, False)

            loads_p[row] = point_p
            loads_q[row] = point_q
            gens_p[row] = point_gen
            _record(data, row, solve(net), outage, point_id, point_hash)
            row += 1

        if progress and point_id and point_id % progress == 0:
            rate = (time.perf_counter() - started) / row * 1000
            print(f"  sweep {point_id:>5}/{n_points} points  {rate:5.1f} ms/solve", flush=True)

    data["load_p_mw"] = loads_p
    data["load_q_mvar"] = loads_q
    data["gen_p_mw"] = gens_p
    data["gen_bus"] = gen_bus
    return _finish(data, out_path, time.perf_counter() - started)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the P-013 datasets")
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--test-points", type=int, default=250)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--only", choices=("train", "sweep"), default=None,
                    help="regenerate just one of the two datasets")
    args = ap.parse_args()

    train = None
    if args.only != "sweep":
        print(f"training set: {args.train} samples", flush=True)
        train = generate_training_set(
            args.train, args.seed, args.out / "train.npz", progress=max(1, args.train // 20)
        )
        print(f"  done in {float(train['seconds'][0]) / 60:.1f} min", flush=True)

    if args.only == "train":
        return

    print(
        f"test sweep: {args.test_points} points x ({N_BRANCH} contingencies + N-0)",
        flush=True,
    )
    sweep = generate_test_sweep(
        args.test_points,
        args.seed + 1,
        args.out / "test_sweep.npz",
        progress=max(1, args.test_points // 20),
    )
    print(f"  done in {float(sweep['seconds'][0]) / 60:.1f} min", flush=True)

    for label, data in (("train", train), ("sweep", sweep)):
        if data is None:
            continue
        unique, counts = np.unique(data["status"], return_counts=True)
        print(f"{label}: {dict(zip(unique.tolist(), counts.tolist()))}", flush=True)


if __name__ == "__main__":
    main()
