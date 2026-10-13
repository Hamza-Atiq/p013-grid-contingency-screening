"""Recover the base voltages PGLib normalises away. Display only.

PGLib-OPF publishes every bus at ``baseKV = 1.0`` because it is an OPF
benchmark stated in per-unit. pandapower's built-in ``case57`` carries the real
levels on the same topology.

**There are six of them, not three** (measured 2026-09-10)::

    115 kV  17 buses      230 kV   5 buses
    138 kV  19 buses      345 kV   5 buses
    161 kV   8 buses      500 kV   3 buses

SPEC 5.5.6 says "115/138/230 kV", which came from reading only the fifteen
highest-degree buses. Two consequences for the console, both open design
questions rather than anything this module decides:

1. Six levels is more than stroke width can separate legibly at 1024x768.
2. The five chosen substations cover 115, 138 and 230 only. The three 500 kV
   buses - the top of the system - are not represented at all.

SPEC 5.5.6 needs them for two things, both cosmetic: drawing voltage level as
line weight, and justifying the choice of the five detailed substations by
voltage spread. **Nothing electrical uses these numbers.** The power flow runs
entirely in per-unit on the PGLib case.

Borrowing across cases is only sound if they are the same network, so
:func:`verify_topology_match` compares the full undirected edge multisets
before any value is handed out. The ``+1`` offset was first noticed from a
degree sequence, which is not proof - two different graphs can share one.
"""
from __future__ import annotations

from collections import Counter

import pandapower.networks as pn

from p013.case_loader import load_case

#: built-in ``case57`` bus *i* corresponds to PGLib bus *i + 1*.
#: pandapower's built-in network is 0-based; PGLib loads 1-based.
BUILTIN_OFFSET = 1


def _edge_multiset(net, offset: int = 0) -> Counter:
    """Undirected {from, to} pairs for all 80 branches, as a multiset.

    A multiset rather than a set because parallel branches between the same
    pair of buses are real and must not collapse.
    """
    edges = [
        frozenset((int(r.from_bus) + offset, int(r.to_bus) + offset))
        for _, r in net.line.iterrows()
    ]
    edges += [
        frozenset((int(r.hv_bus) + offset, int(r.lv_bus) + offset))
        for _, r in net.trafo.iterrows()
    ]
    return Counter(edges)


def verify_topology_match(_offset: int | None = None) -> None:
    """Raise unless PGLib and the built-in case are the same network.

    ``_offset`` exists so the guard itself can be tested.
    """
    offset = BUILTIN_OFFSET if _offset is None else _offset

    pglib = _edge_multiset(load_case())
    builtin = _edge_multiset(pn.case57(), offset=offset)

    if pglib != builtin:
        only_pglib = sorted(tuple(sorted(e)) for e in (pglib - builtin).elements())
        only_builtin = sorted(tuple(sorted(e)) for e in (builtin - pglib).elements())
        raise ValueError(
            "PGLib and the built-in case57 are not the same network, so voltage "
            "levels cannot be borrowed from one for the other.\n"
            f"  edges only in PGLib   ({len(only_pglib)}): {only_pglib[:8]}\n"
            f"  edges only in built-in ({len(only_builtin)}): {only_builtin[:8]}\n"
            "If this fires, do not widen the check - drop SPEC 5.5.6's "
            "voltage-level rule instead and report it."
        )


def bus_kv() -> dict[int, float]:
    """PGLib bus index -> real base voltage in kV. Verified before returning."""
    verify_topology_match()
    builtin = pn.case57()
    return {
        int(i) + BUILTIN_OFFSET: float(builtin.bus.at[i, "vn_kv"])
        for i in builtin.bus.index
    }
