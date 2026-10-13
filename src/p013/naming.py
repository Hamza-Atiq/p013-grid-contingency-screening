"""One index space, one display name. SPEC 5.5.1 and 5.5.6.

Lines are ``L01``..``L65`` and transformers ``T01``..``T15``, both 1-based like
the bus indices, so the diagram, the ranked list, the API and the model all
speak the same names with no translation table for anyone to get wrong.

**Why 65 + 15 and not 63 + 17.** The spec left this open. Reading the case
settles it: 63 branches carry ``tap = 0`` (MATPOWER's "this is a line"), 15
carry a ratio that is not 1, and 2 carry a ratio of exactly 1.0. The
2026-09-09 review counted the 17 with a non-zero ratio *field*; pandapower
builds transformers only for the 15 that are electrically transformers,
because a 1:1 ratio is a line. Both counts were right about different things.
"""
from __future__ import annotations

from typing import NamedTuple


class BranchRef(NamedTuple):
    """A branch, addressable every way the project needs."""

    name: str
    kind: str  #: "line" or "trafo"
    index: int  #: the pandapower index within that frame
    from_bus: int
    to_bus: int


def branch_table(net) -> list[BranchRef]:
    """All 80 branches: lines first, then transformers, each in case order.

    The order is the contract. Position in this list is the branch's index in
    every array the project stores, so it must not depend on anything that can
    vary between loads.
    """
    refs: list[BranchRef] = []
    for pos, (idx, row) in enumerate(net.line.iterrows(), start=1):
        refs.append(
            BranchRef(f"L{pos:02d}", "line", int(idx), int(row.from_bus), int(row.to_bus))
        )
    for pos, (idx, row) in enumerate(net.trafo.iterrows(), start=1):
        refs.append(
            BranchRef(f"T{pos:02d}", "trafo", int(idx), int(row.hv_bus), int(row.lv_bus))
        )
    return refs


def branch_names(net) -> list[str]:
    """Display names in the canonical order."""
    return [b.name for b in branch_table(net)]


def branch_index(net, name: str) -> tuple[str, int]:
    """``"L11"`` -> ``("line", 10)``. Raises rather than guessing."""
    for ref in branch_table(net):
        if ref.name == name:
            return ref.kind, ref.index
    raise KeyError(f"no branch named {name!r}")


def set_in_service(net, name: str, in_service: bool) -> None:
    """Take a branch out of service, or put it back, by display name."""
    kind, idx = branch_index(net, name)
    frame = net.line if kind == "line" else net.trafo
    frame.at[idx, "in_service"] = in_service
