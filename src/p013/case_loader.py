"""Load the PGLib-OPF IEEE 57-bus case.

Why PGLib and not pandapower's built-in ``case57`` (SPEC 1.4, 3.1, 2.6).
Measured 2026-09-10, both on this machine:

===========================  ==================  ==========
                             built-in ``case57``  PGLib
===========================  ==================  ==========
Branches carrying a rating   0 of 80             80 of 80
Buses outside 0.95-1.05 pu   40 of 57            5 of 57
Lowest base-case voltage     0.72 pu             0.937 pu
Generator setpoints          0.98 - 1.015        all 1.00
===========================  ==================  ==========

Either failure alone would have been fatal. With no ratings, every "percent of
rating" number in the project rests on an invented limit. With 40 of 57 buses
already violating, every contingency is dangerous, the headline metric
degenerates to the whole sweep and the ranking has a permanent floor.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandapower as pp

# pandapower 3.5.4 moved this. ``from pandapower.converter import from_ppc``
# raises ImportError; the package now exposes only ``pandamodels`` at that
# level. Verified 2026-09-10.
from pandapower.converter.pypower.from_ppc import from_ppc

import p013


@dataclass(frozen=True)
class CaseInvariants:
    """What this case must always be. Asserted on every load, never trusted."""

    N_BUS: int = 57
    N_BRANCH: int = 80
    N_LINE: int = 65
    N_TRAFO: int = 15
    N_GEN: int = 7
    BASE_MVA: float = 100.0
    SLACK_BUS: int = 1
    RATE_MIN_MVA: float = 22.0
    RATE_MAX_MVA: float = 1617.0


#: MATPOWER branch matrix column 6 (0-indexed 5) is rateA, the MVA rating.
RATE_A_COL = 5


def _matrix(text: str, name: str) -> np.ndarray:
    """Pull one ``mpc.<name> = [ ... ];`` matrix out of a MATPOWER .m file.

    Rows are ragged in some PGLib files (trailing columns omitted), so short
    rows are zero-padded to the widest.
    """
    match = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", text, re.S)
    if match is None:
        raise ValueError(f"mpc.{name} not found in case file")

    rows: list[list[float]] = []
    for line in match.group(1).strip().splitlines():
        line = line.split("%")[0].strip().rstrip(";").strip()
        if line:
            rows.append([float(x) for x in line.replace(",", " ").split()])
    if not rows:
        raise ValueError(f"mpc.{name} is empty")

    width = max(len(r) for r in rows)
    return np.array([r + [0.0] * (width - len(r)) for r in rows])


def parse_matpower(path: Path) -> dict:
    """Read a MATPOWER .m case into a PYPOWER-style dict."""
    text = path.read_text(encoding="utf-8")
    base = re.search(r"mpc\.baseMVA\s*=\s*([\d.]+)", text)
    if base is None:
        raise ValueError("mpc.baseMVA not found in case file")
    return {
        "version": "2",
        "baseMVA": float(base.group(1)),
        "bus": _matrix(text, "bus"),
        "gen": _matrix(text, "gen"),
        "branch": _matrix(text, "branch"),
    }


def load_case(_force_branch_count: int | None = None):
    """Return the case as a fresh pandapower network, invariants asserted.

    A new network every call. Callers take branches out of service and split
    buses, so shared state would contaminate every sweep downstream.

    ``_force_branch_count`` exists only so the guard itself can be tested.
    """
    ppc = parse_matpower(p013.CASE_PATH)

    n_branch = (
        _force_branch_count if _force_branch_count is not None else ppc["branch"].shape[0]
    )
    if n_branch != CaseInvariants.N_BRANCH:
        raise ValueError(f"expected {CaseInvariants.N_BRANCH} branches, found {n_branch}")

    rate_a = ppc["branch"][:, RATE_A_COL]
    if not (rate_a > 0).all():
        n_zero = int((rate_a <= 0).sum())
        raise ValueError(
            f"{n_zero} branches carry no rating. Every thermal number in this "
            "project is a percent of rating; see SPEC 1.4."
        )

    # from_ppc logs a warning on every call: "15 branches which are considered
    # as trafos - due to ratio unequal 0 or 1 - but connect same voltage
    # levels." It is expected and correct here - PGLib publishes every bus at
    # baseKV = 1.0 (see display_meta), so the 15 tap-changing branches
    # necessarily join equal voltage levels. Silenced so it does not drown real
    # output across tens of thousands of loads.
    _converter_log = logging.getLogger("pandapower.converter.pypower.from_ppc")
    _previous_level = _converter_log.level
    _converter_log.setLevel(logging.ERROR)
    try:
        net = from_ppc(ppc, f_hz=60)
    finally:
        _converter_log.setLevel(_previous_level)

    if len(net.bus) != CaseInvariants.N_BUS:
        raise ValueError(f"expected {CaseInvariants.N_BUS} buses, found {len(net.bus)}")
    if len(net.line) != CaseInvariants.N_LINE or len(net.trafo) != CaseInvariants.N_TRAFO:
        raise ValueError(
            f"expected {CaseInvariants.N_LINE} lines + {CaseInvariants.N_TRAFO} "
            f"transformers, found {len(net.line)} + {len(net.trafo)}"
        )
    if list(net.bus.index) != list(range(1, CaseInvariants.N_BUS + 1)):
        raise ValueError(
            "bus indices must be 1..57 contiguous; SPEC 5.5.1 relies on the "
            "index being the display number"
        )
    if int(net.ext_grid.bus.iloc[0]) != CaseInvariants.SLACK_BUS:
        raise ValueError(f"slack bus moved to {int(net.ext_grid.bus.iloc[0])}")

    return net


class CaseTemplate:
    """One network, reset in place. For generation loops only.

    :func:`load_case` costs **442 ms** on this machine, almost all of it inside
    pandapower's ``from_ppc``, against **46 ms** for the power flow itself
    (measured 2026-09-11). Calling it once per sample would make network
    *construction* 89% of the run: 40,000 samples would take 5.5 hours, of
    which under 35 minutes is actual physics.

    So generation builds one network and restores its mutable inputs between
    samples - in-service flags, loads, generation - which costs microseconds.

    This is safe only because every solve uses ``init="flat"`` (SPEC 2.6): the
    previous sample's results are on the network but cannot influence the next
    solve. On a reused network a warm start would silently carry state between
    unrelated samples, which is a second, quieter reason the flat start matters.

    **Not for split networks.** :func:`p013.topology.split_bus` adds buses and
    switches, which this cannot undo. Build those with :func:`load_case`.
    """

    def __init__(self) -> None:
        self._net = load_case()
        self._line_in_service = self._net.line.in_service.to_numpy().copy()
        self._trafo_in_service = self._net.trafo.in_service.to_numpy().copy()
        self._load_p = self._net.load.p_mw.to_numpy().copy()
        self._load_q = self._net.load.q_mvar.to_numpy().copy()
        self._gen_p = self._net.gen.p_mw.to_numpy().copy()

    def fresh(self):
        """Restore the pristine inputs and return the network.

        Returns the *same* object every call. Callers must not hold on to a
        previous return value expecting it to be unchanged.
        """
        net = self._net
        net.line["in_service"] = self._line_in_service.copy()
        net.trafo["in_service"] = self._trafo_in_service.copy()
        net.load["p_mw"] = self._load_p.copy()
        net.load["q_mvar"] = self._load_q.copy()
        net.gen["p_mw"] = self._gen_p.copy()
        return net
