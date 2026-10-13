"""What the model is given. SPEC 2.1.

Per bus: real and reactive injection, one-hot bus type, voltage setpoint, and
the shunt conductance and susceptance.

Per branch: resistance, reactance, line charging, tap ratio, phase shift, and
the in-service flag.

**The shunts nearly did not make it into this list.** SPEC 2.1 records that
they were missing until 2026-09-09 and that their absence "would have been a
real bug": three buses in this case carry a shunt, a shunt injects reactive
power, and a model that never sees them is asked to explain voltages it has no
cause for - while the physics term would be computing mismatch against the
wrong injection and punishing correct answers.

**The in-service flag is a branch feature, and the impedance stays.** A branch
that is out keeps its R, X and B in the feature vector and has its flag set to
zero. The model learns "this line exists and carries nothing", not "this line
has no impedance". That flag is the entire reason one trained model covers
every outage and every busbar split (SPEC 2.2).

Everything here is per unit, taken from the MATPOWER arrays rather than the
pandapower network - see :mod:`p013.admittance` for why.
"""
from __future__ import annotations

import numpy as np

from p013.admittance import (
    BR_ANGLE,
    BR_B,
    BR_R,
    BR_RATIO,
    BR_X,
    BUS_BS,
    BUS_GS,
    BUS_PD,
    BUS_QD,
    BUS_TYPE,
    BUS_TYPE_PQ,
    BUS_TYPE_PV,
    BUS_TYPE_SLACK,
    GEN_BUS,
    GEN_PG,
    GEN_STATUS,
    GEN_VG,
    bus_position,
    canonical_branch_rows,
)

#: p_injection, q_injection, is_pq, is_pv, is_slack, v_setpoint, gs, bs
F_BUS = 8

#: r, x, b, tap_ratio, shift_rad, in_service
F_BRANCH = 6

BUS_FEATURE_NAMES = (
    "p_injection",
    "q_injection",
    "is_pq",
    "is_pv",
    "is_slack",
    "v_setpoint",
    "gs",
    "bs",
)

BRANCH_FEATURE_NAMES = ("r", "x", "b", "tap_ratio", "shift_rad", "in_service")


def bus_features(ppc, load_p=None, load_q=None, gen_p=None) -> np.ndarray:
    """``(n_bus, F_BUS)`` in per unit.

    ``load_p`` / ``load_q`` are in MW / MVAr in bus-matrix order and override
    the case's own demand, so a sampled operating point can be encoded.

    ``gen_p`` does the same for real generation, in MW in bus-matrix order, and
    **it is not optional for sampled data**. The sampler scales generation to
    follow load (SPEC 2.4); without this override the features carry the case
    file's base dispatch while the labels come from a solve that used the
    scaled one. Measured on the first 20,000-sample set, that was a **2.3 pu**
    real-power mismatch at the PV buses - the model fed a wrong injection, and
    the physics term scored against a state nobody ever solved. Added
    2026-09-11. Voltage setpoints are NOT scaled and still come from the case.
    """
    bus, gen = ppc["bus"], ppc["gen"]
    base = ppc["baseMVA"]
    n = len(bus)
    pos = bus_position(ppc)

    pd = (bus[:, BUS_PD] if load_p is None else np.asarray(load_p, float)) / base
    qd = (bus[:, BUS_QD] if load_q is None else np.asarray(load_q, float)) / base

    pg = np.zeros(n)
    v_set = np.zeros(n)
    for row in gen:
        if row[GEN_STATUS] <= 0:
            continue
        i = pos[int(row[GEN_BUS])]
        pg[i] += row[GEN_PG] / base
        v_set[i] = row[GEN_VG]

    if gen_p is not None:
        pg = np.asarray(gen_p, dtype=float) / base

    types = bus[:, BUS_TYPE].astype(int)

    out = np.zeros((n, F_BUS))
    out[:, 0] = pg - pd
    out[:, 1] = -qd  # generator Q is an output of the solve, never an input
    out[:, 2] = types == BUS_TYPE_PQ
    out[:, 3] = types == BUS_TYPE_PV
    out[:, 4] = types == BUS_TYPE_SLACK
    out[:, 5] = v_set
    out[:, 6] = bus[:, BUS_GS] / base
    out[:, 7] = bus[:, BUS_BS] / base
    return out


def branch_features(ppc, in_service: np.ndarray | None = None) -> np.ndarray:
    """``(n_branch, F_BRANCH)`` in per unit, in canonical branch order.

    ``in_service`` is a boolean array in canonical order, or ``None`` for the
    intact network.
    """
    rows = canonical_branch_rows(ppc)
    branch = ppc["branch"][rows]

    flags = (
        np.ones(len(branch), dtype=float)
        if in_service is None
        else np.asarray(in_service, dtype=float)
    )
    if len(flags) != len(branch):
        raise ValueError(f"expected {len(branch)} flags, got {len(flags)}")

    ratio = branch[:, BR_RATIO].copy()
    ratio[ratio == 0.0] = 1.0  # MATPOWER: 0 means "this is a line"

    out = np.zeros((len(branch), F_BRANCH))
    out[:, 0] = branch[:, BR_R]
    out[:, 1] = branch[:, BR_X]
    out[:, 2] = branch[:, BR_B]
    out[:, 3] = ratio
    out[:, 4] = np.deg2rad(branch[:, BR_ANGLE])
    out[:, 5] = flags
    return out


def edge_index(ppc) -> np.ndarray:
    """``(2, n_branch)`` of zero-based bus positions, in canonical order.

    Zero-based *positions*, not bus numbers: the model indexes arrays with
    these. Bus number 1 is position 0.
    """
    rows = canonical_branch_rows(ppc)
    branch = ppc["branch"][rows]
    pos = bus_position(ppc)
    return np.array(
        [
            [pos[int(r[0])] for r in branch],
            [pos[int(r[1])] for r in branch],
        ],
        dtype=np.int64,
    )
