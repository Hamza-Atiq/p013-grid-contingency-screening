"""Turn a generated ``.npz`` into model inputs and targets. NumPy only.

Deliberately no torch here. ``forward_numpy`` has to run in a serverless bundle
that does not contain torch (SPEC 5.1), and it shares this feature-building
path, so a torch import at this level would leak into the deployment.

Two things this module refuses to do quietly:

**It does not hide failed rows.** Rows that did not converge are excluded from
training - there is no target to learn - but the count is reported, because
SPEC 2.4 says deleting them would make our results look better than they are.

**It does not split randomly.** Splits are by operating point, so samples from
one operating condition cannot land in both halves. A random split leaks the
condition across the boundary and the score comes out flattering (SPEC 2.4).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import p013
from p013.admittance import (
    BUS_BS,
    BUS_GS,
    BUS_I,
    GEN_BUS,
    GEN_PG,
    GEN_STATUS,
    bus_position,
)
from p013.case_loader import parse_matpower
from p013.features import branch_features, bus_features, edge_index


@dataclass(frozen=True)
class Dataset:
    """Model inputs and targets, all NumPy.

    Attributes:
        bus_x: ``(n, n_bus, F_BUS)``
        branch_x: ``(n, n_branch, F_BRANCH)``
        edge_index: ``(2, n_branch)``
        vm: ``(n, n_bus)`` target voltage magnitude
        va: ``(n, n_bus)`` target angle in radians, relative to the slack
        s_specified: ``(n, n_bus)`` complex injection actually fed in
        shunt: ``(n_bus,)`` complex bus shunt admittance
        operating_point: ``(n,)`` hash identifying the operating condition
        excluded: how many rows were dropped, and why
    """

    bus_x: np.ndarray
    branch_x: np.ndarray
    edge_index: np.ndarray
    vm: np.ndarray
    va: np.ndarray
    s_specified: np.ndarray
    shunt: np.ndarray
    operating_point: np.ndarray
    excluded: dict[str, int]

    def __len__(self) -> int:
        return len(self.vm)


def to_bus_order(ppc, buses, values) -> np.ndarray:
    """Scatter per-element values onto the 57 bus positions.

    Used for loads and for generation. Public because the evaluation has to
    rebuild exactly these inputs from the sweep file, and rebuilding them a
    second way is how the two drift apart.
    """
    pos = bus_position(ppc)
    out = np.zeros((values.shape[0], len(ppc["bus"])))
    for column, bus in enumerate(buses):
        out[:, pos[int(bus)]] += values[:, column]
    return out


def _sampled_generation(ppc, raw, keep) -> np.ndarray:
    """``(n, n_bus)`` real generation in MW, as the solver actually saw it.

    The generated file stores ``gen_p_mw`` for pandapower's ``net.gen`` rows
    and ``gen_bus`` to say which bus each belongs to, so nothing here has to
    guess at row order. The **slack** is not in that table - its dispatch is an
    output of the solve, never an input - so it keeps the case file's value,
    which is what it always had and which the physics term ignores anyway.

    A file without ``gen_p_mw`` predates 2026-09-11 and cannot rebuild the
    state it solved. That is an error rather than a fallback: falling back to
    the base dispatch is precisely the bug this replaced.
    """
    if "gen_p_mw" not in raw.files:
        raise ValueError(
            f"this dataset has no 'gen_p_mw', so it cannot rebuild the operating "
            "point it solved - generation was scaled by the sampler and never "
            "stored. Regenerate it with p013.generate (fixed 2026-09-11)."
        )

    gen_bus = [int(b) for b in raw["gen_bus"]]
    gen_p = to_bus_order(ppc, gen_bus, raw["gen_p_mw"][keep])

    pos = bus_position(ppc)
    sampled = set(gen_bus)
    for row in ppc["gen"]:
        if row[GEN_STATUS] <= 0 or int(row[GEN_BUS]) in sampled:
            continue
        gen_p[:, pos[int(row[GEN_BUS])]] += row[GEN_PG]
    return gen_p


def load_dataset(path: Path, case_path: Path | None = None) -> Dataset:
    """Read a generated ``.npz`` and build everything the model needs."""
    ppc = parse_matpower(case_path or p013.CASE_PATH)
    raw = np.load(path, allow_pickle=False)

    status = raw["status"]
    keep = status == "converged"
    excluded = {
        "islanded": int((status == "islanded").sum()),
        "no_solution": int((status == "no_solution").sum()),
        "kept": int(keep.sum()),
        "total": int(len(status)),
    }

    # Load buses, in the order generate.py stored them. pandapower's load table
    # is built from the case in bus order, one load per bus in this case.
    load_buses = [int(b[BUS_I]) for b in ppc["bus"] if b[2] != 0 or b[3] != 0]
    if len(load_buses) != raw["load_p_mw"].shape[1]:
        raise ValueError(
            f"dataset has {raw['load_p_mw'].shape[1]} loads, case has {len(load_buses)}"
        )

    load_p = to_bus_order(ppc, load_buses, raw["load_p_mw"][keep])
    load_q = to_bus_order(ppc, load_buses, raw["load_q_mvar"][keep])
    gen_p = _sampled_generation(ppc, raw, keep)
    in_service = raw["in_service"][keep]

    n = int(keep.sum())
    bus_x = np.stack(
        [bus_features(ppc, load_p[i], load_q[i], gen_p[i]) for i in range(n)]
    )
    branch_x = np.stack([branch_features(ppc, in_service[i]) for i in range(n)])

    vm = raw["vm_pu"][keep]
    va = np.deg2rad(raw["va_degree"][keep])
    # SPEC 2.1: angles are relative to the slack, which is fixed at 0. The
    # solver already reports them that way; subtracting is belt and braces and
    # makes the invariant explicit rather than inherited.
    va = va - va[:, [0]]

    s_specified = bus_x[:, :, 0] + 1j * bus_x[:, :, 1]
    shunt = (ppc["bus"][:, BUS_GS] + 1j * ppc["bus"][:, BUS_BS]) / ppc["baseMVA"]

    return Dataset(
        bus_x=bus_x.astype(np.float32),
        branch_x=branch_x.astype(np.float32),
        edge_index=edge_index(ppc),
        vm=vm.astype(np.float32),
        va=va.astype(np.float32),
        s_specified=s_specified.astype(np.complex64),
        shunt=shunt.astype(np.complex64),
        operating_point=raw["operating_point_hash"][keep],
        excluded=excluded,
    )


def split_by_operating_point(
    data: Dataset, val_fraction: float = 0.2, seed: int = 0
) -> tuple[Dataset, Dataset]:
    """Split so no operating condition appears in both halves. SPEC 2.4."""
    points = np.unique(data.operating_point)
    rng = np.random.default_rng(seed)
    rng.shuffle(points)

    n_val = max(1, int(len(points) * val_fraction))
    val_points = set(points[:n_val].tolist())

    is_val = np.array([p in val_points for p in data.operating_point])
    return _subset(data, ~is_val), _subset(data, is_val)


def _subset(data: Dataset, mask: np.ndarray) -> Dataset:
    return Dataset(
        bus_x=data.bus_x[mask],
        branch_x=data.branch_x[mask],
        edge_index=data.edge_index,
        vm=data.vm[mask],
        va=data.va[mask],
        s_specified=data.s_specified[mask],
        shunt=data.shunt,
        operating_point=data.operating_point[mask],
        excluded=data.excluded,
    )
