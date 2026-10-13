"""The admittance matrix, and complex power from voltage. NumPy, shared.

One implementation, used by three things: the physics term in training, the
derivation of branch flows from predicted voltages (SPEC 2.1 - the model never
outputs flows, so they can never disagree with its voltages), and the baselines
in phase 3. One place to be wrong.

**Why the MATPOWER arrays and not the pandapower network.** PGLib publishes
every bus at ``baseKV = 1.0``, so pandapower's engineering units (ohms per km,
nanofarads) are a degenerate round trip. The ``.m`` file is already per-unit on
a 100 MVA base, which is exactly what the power flow equations want.

**Verified 2026-09-11.** Building ``Ybus`` this way and evaluating
``S = V . conj(Y V)`` at pandapower's own solution gives a mismatch of
**1.8e-12** at PQ buses and **4.4e-14** at PV buses. That check is the reason
the physics term can be trusted to measure the model rather than our own bug:
a sign error here would have surfaced as "the physics term does not help" and
been written off as a modelling result.
"""
from __future__ import annotations

import numpy as np

# MATPOWER bus matrix columns
BUS_I, BUS_TYPE, BUS_PD, BUS_QD, BUS_GS, BUS_BS = 0, 1, 2, 3, 4, 5
BUS_VM, BUS_VA, BUS_BASEKV = 7, 8, 9

# MATPOWER branch matrix columns
BR_F, BR_T, BR_R, BR_X, BR_B, BR_RATE_A = 0, 1, 2, 3, 4, 5
BR_RATIO, BR_ANGLE, BR_STATUS = 8, 9, 10

# MATPOWER gen matrix columns
GEN_BUS, GEN_PG, GEN_QG, GEN_VG, GEN_STATUS = 0, 1, 2, 5, 7

#: MATPOWER bus types.
BUS_TYPE_PQ, BUS_TYPE_PV, BUS_TYPE_SLACK = 1, 2, 3


def bus_position(ppc) -> dict[int, int]:
    """External bus number -> row position in the matrices."""
    return {int(row[BUS_I]): i for i, row in enumerate(ppc["bus"])}


def canonical_branch_rows(ppc) -> np.ndarray:
    """Canonical branch position -> MATPOWER branch row.

    **The project has one canonical branch order**, the one
    :func:`p013.naming.branch_table` produces: the 65 lines ``L01``-``L65``
    followed by the 15 transformers ``T01``-``T15``. Every array the project
    stores - the ``in_service`` flags in the datasets, the ranked list, the
    console's data files - is in that order.

    The MATPOWER branch matrix is **not** in that order. pandapower's
    ``from_ppc`` splits branches into lines and transformers by tap ratio
    (a ratio of 0 or exactly 1 is a line), which interleaves them: ``L25`` is
    MATPOWER row 26 and ``T01`` is row 18.

    Discovered 2026-09-11 when derived branch flows matched the solver for the
    first few branches and then diverged. Left implicit, it would have fed the
    physics term a different topology than the one that was solved - a silent
    error that produces plausible numbers.
    """
    ratio = ppc["branch"][:, BR_RATIO]
    is_line = (ratio == 0.0) | (ratio == 1.0)
    lines = np.flatnonzero(is_line)
    trafos = np.flatnonzero(~is_line)
    return np.concatenate([lines, trafos])


def to_matpower_order(ppc, canonical: np.ndarray) -> np.ndarray:
    """Reorder an array from canonical branch order to MATPOWER row order."""
    out = np.empty_like(np.asarray(canonical))
    out[canonical_branch_rows(ppc)] = np.asarray(canonical)
    return out


def to_canonical_order(ppc, matpower: np.ndarray) -> np.ndarray:
    """Reorder an array from MATPOWER row order to canonical branch order."""
    return np.asarray(matpower)[canonical_branch_rows(ppc)]


def build_ybus(ppc, in_service: np.ndarray | None = None) -> np.ndarray:
    """The bus admittance matrix, in per unit.

    ``in_service`` is a boolean array over the 80 branches in **canonical
    order** (``L01``-``L65`` then ``T01``-``T15``; see
    :func:`canonical_branch_rows`), which is the order every array in this
    project uses. It is translated to MATPOWER row order here so callers never
    have to think about it.

    A branch that is out contributes nothing at all - which is exactly what
    "out of service" means, and is why one trained model covers every outage.
    """
    bus, branch = ppc["bus"], ppc["branch"]
    n = len(bus)
    pos = bus_position(ppc)
    ybus = np.zeros((n, n), dtype=complex)

    if in_service is not None:
        in_service = to_matpower_order(ppc, np.asarray(in_service, dtype=bool))

    for k, row in enumerate(branch):
        if in_service is not None and not in_service[k]:
            continue
        if row[BR_STATUS] == 0:
            continue

        f, t = pos[int(row[BR_F])], pos[int(row[BR_T])]
        series = 1.0 / complex(row[BR_R], row[BR_X])
        charging = row[BR_B]

        # MATPOWER: a ratio of 0 means "this is a line", i.e. a ratio of 1.
        ratio = row[BR_RATIO] if row[BR_RATIO] != 0 else 1.0
        tap = ratio * np.exp(1j * np.deg2rad(row[BR_ANGLE]))

        y_tt = series + 1j * charging / 2.0
        ybus[f, f] += y_tt / (tap * np.conj(tap))
        ybus[t, t] += y_tt
        ybus[f, t] += -series / np.conj(tap)
        ybus[t, f] += -series / tap

    # Bus shunts. Three of them in this case, on buses 18, 25 and 53, all
    # capacitive. SPEC 2.1: leaving them out would ask the model to explain
    # voltages it has no cause for.
    for i, row in enumerate(bus):
        ybus[i, i] += complex(row[BUS_GS], row[BUS_BS]) / ppc["baseMVA"]

    return ybus


def power_injection(ybus: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Complex power injected at each bus, per unit: ``S = V . conj(Y V)``."""
    return v * np.conj(ybus @ v)


def specified_injection(ppc, load_p=None, load_q=None) -> np.ndarray:
    """What we told the network to inject, per unit.

    ``load_p`` and ``load_q`` override the case's own demand, in MW/MVAr and in
    bus-matrix order, so a sampled operating point can be evaluated.

    **Generation is NOT overridable here, and this function is therefore only
    correct for the base dispatch.** Every caller today passes no load override
    at all, so it only ever sees the base case and the gap cannot bite. It is
    flagged because the identical gap in :func:`p013.features.bus_features` was
    a live bug until 2026-09-11: the sampler scales generation to follow load
    (SPEC 2.4), and reading ``GEN_PG`` from the case file while the labels came
    from a scaled solve put a **2.3 pu** real-power mismatch at the PV buses.
    If you ever call this with a sampled ``load_p``, add a ``gen_p`` override
    first - and write the test that fails without it.
    """
    bus, gen = ppc["bus"], ppc["gen"]
    base = ppc["baseMVA"]
    n = len(bus)
    pos = bus_position(ppc)

    pd = (bus[:, BUS_PD] if load_p is None else np.asarray(load_p, dtype=float)) / base
    qd = (bus[:, BUS_QD] if load_q is None else np.asarray(load_q, dtype=float)) / base

    pg = np.zeros(n)
    qg = np.zeros(n)
    for row in gen:
        if row[GEN_STATUS] <= 0:
            continue
        i = pos[int(row[GEN_BUS])]
        pg[i] += row[GEN_PG] / base
        qg[i] += row[GEN_QG] / base

    return (pg - pd) + 1j * (qg - qd)


def loading_percent(ppc, v: np.ndarray, in_service: np.ndarray | None = None) -> np.ndarray:
    """Apparent power as a percentage of each branch's published MVA rating.

    In canonical order, zero for anything out of service.

    **This is deliberately NOT pandapower's ``loading_percent``.** Added
    2026-09-11 after measuring the difference. pandapower reports current
    against ``max_i_ka``; PGLib publishes ``rateA`` in MVA; the two differ by a
    factor of ``1/V`` and by up to **28.7 percentage points** on a contingency
    that collapses a voltage to 0.78 pu. On a healthy system they agree to
    about one percentage point, which is why nothing noticed sooner.

    Truth and model must run the identical checks (SPEC 3.1), and the model
    predicts voltages from which apparent power is derived. If ground truth
    used pandapower's current-based figure while the model used this one, the
    model would be judged against a quantity it never computes. So both sides
    use this, and this matches what SPEC 3.1 and the PGLib file actually say -
    a rating in MVA.

    Measured on 1,500 real sweep rows, the two definitions disagree on **zero**
    danger labels, so adopting it changes no result; it only makes the two
    sides comparable by construction.
    """
    rate = to_canonical_order(ppc, ppc["branch"][:, BR_RATE_A])
    if not (rate > 0).all():
        raise ValueError(
            "a branch has no rating, so its loading is undefined. SPEC 3.1: this "
            "is why the case is loaded from PGLib rather than MATPOWER's case57, "
            "where rateA is zero on every branch."
        )
    mva = np.abs(branch_flows(ppc, v, in_service)) * ppc["baseMVA"]
    return mva / rate * 100.0


def branch_flows(ppc, v: np.ndarray, in_service: np.ndarray | None = None) -> np.ndarray:
    """Apparent power at the *from* end of each branch, per unit.

    SPEC 2.1: the model does not predict flows. They are derived here from the
    predicted voltages through the same equations the solver uses, so the two
    can never disagree with each other - which a professor would catch.

    ``in_service`` is in canonical order and the result is returned in
    canonical order, so it lines up with the ranked list and the datasets.
    """
    branch = ppc["branch"]
    pos = bus_position(ppc)
    out = np.zeros(len(branch), dtype=complex)

    if in_service is not None:
        in_service = to_matpower_order(ppc, np.asarray(in_service, dtype=bool))

    for k, row in enumerate(branch):
        if (in_service is not None and not in_service[k]) or row[BR_STATUS] == 0:
            continue

        f, t = pos[int(row[BR_F])], pos[int(row[BR_T])]
        series = 1.0 / complex(row[BR_R], row[BR_X])
        ratio = row[BR_RATIO] if row[BR_RATIO] != 0 else 1.0
        tap = ratio * np.exp(1j * np.deg2rad(row[BR_ANGLE]))

        y_ff = (series + 1j * row[BR_B] / 2.0) / (tap * np.conj(tap))
        y_ft = -series / np.conj(tap)
        i_f = y_ff * v[f] + y_ft * v[t]
        out[k] = v[f] * np.conj(i_f)

    return to_canonical_order(ppc, out)
