"""Exact topology facts. No power flow, no model. SPEC 2.5.

Islanding is a graph question, and it is answered exactly, in microseconds, by
a breadth-first search. It is settled here and never left to the model, because
the worst error a screening tool can make is calling a blackout-causing outage
safe: the model will happily emit voltages for a network that has fallen in
half, and it may rank that case as harmless.

So the model is never asked a question it cannot answer. Any outage that
increases the number of connected components is pinned to the top of the
ranking without consulting it.

**The test is always relative.** "More components than the current topology
has", never a fixed number. With a bus coupler open the base count can
legitimately be two, and a hardcoded threshold then reports nothing at all.
That bug was found in the design mockup and must not come back.
"""
from __future__ import annotations

from collections import defaultdict, deque

import pandapower as pp

from p013.case_loader import CaseInvariants
from p013.naming import branch_index, branch_table


def _coupler_pairs(net) -> list[tuple[int, int]]:
    """Closed bus-bus switches fuse two busbars into one electrical node."""
    if not hasattr(net, "switch") or len(net.switch) == 0:
        return []
    et = net.switch.et.to_numpy()
    closed = net.switch.closed.to_numpy()
    bus = net.switch.bus.to_numpy()
    element = net.switch.element.to_numpy()
    return [
        (int(bus[i]), int(element[i]))
        for i in range(len(et))
        if et[i] == "b" and bool(closed[i])
    ]


def _graph(net):
    """Read the whole topology out of pandas ONCE.

    ``branch_table`` walks two pandas frames with ``iterrows``, and reading
    each branch's in-service flag took a scalar pandas ``.at`` lookup. Doing
    that 81 times - once for the base case and once per contingency - made the
    islanding pre-filter cost **0.754 s**, against 0.065 s for a full AC power
    flow. SPEC 3.5 counts the pre-filter against us, so it was 58% of the
    screening overhead, spent re-reading a table that never changes.

    The algorithm below is unchanged: still "remove the branch, count the
    components, compare". Only the bookkeeping moved out of pandas.

    Returns ``(buses, branches, couplers)`` where each branch is
    ``(name, from_bus, to_bus, in_service)``.
    """
    buses = [int(b) for b in net.bus.index]

    line_at = {int(idx): i for i, idx in enumerate(net.line.index)}
    trafo_at = {int(idx): i for i, idx in enumerate(net.trafo.index)}
    line_live = net.line.in_service.to_numpy()
    trafo_live = net.trafo.in_service.to_numpy()

    branches = []
    for ref in branch_table(net):
        live = (
            line_live[line_at[ref.index]]
            if ref.kind == "line"
            else trafo_live[trafo_at[ref.index]]
        )
        branches.append((ref.name, ref.from_bus, ref.to_bus, bool(live)))

    return buses, branches, _coupler_pairs(net)


def _groups(buses, edges, couplers) -> list[list[int]]:
    """Connected components by breadth-first search over plain lists."""
    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    for a, b in couplers:
        adjacency[a].append(b)
        adjacency[b].append(a)

    seen: set[int] = set()
    out: list[list[int]] = []
    for start in buses:
        if start in seen:
            continue
        seen.add(start)
        queue, group = deque([start]), [start]
        while queue:
            node = queue.popleft()
            for nxt in adjacency[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
                    group.append(nxt)
        out.append(group)
    return out


def components(net) -> list[list[int]]:
    """The connected components of the in-service network, as bus lists."""
    buses, branches, couplers = _graph(net)
    edges = [(f, t) for _, f, t, live in branches if live]
    return _groups(buses, edges, couplers)


def n_components(net) -> int:
    return len(components(net))


def islanding_contingencies(net) -> set[str]:
    """Branches whose loss would leave MORE components than there are now.

    **The test is always relative** - "more components than the current
    topology has", never a fixed number. With a bus coupler open the base count
    can legitimately be two, and a hardcoded threshold then reports nothing at
    all. That bug was found in the design mockup and must not come back.

    The network is not modified at all: the branch is dropped from a list of
    edges rather than toggled in the pandas frame, so there is nothing to
    restore and no way to leave the net dirty if something raises.
    """
    buses, branches, couplers = _graph(net)
    live = [(i, (f, t)) for i, (_, f, t, ok) in enumerate(branches) if ok]

    base = len(_groups(buses, [edge for _, edge in live], couplers))

    found: set[str] = set()
    for dropped, _ in live:
        edges = [edge for i, edge in live if i != dropped]
        if len(_groups(buses, edges, couplers)) > base:
            found.add(branches[dropped][0])
    return found


def islanded_load_mw(net) -> float:
    """Load sitting in any component that does not contain the slack bus.

    This is what "blacked out" means, and it is why the guided story is scored
    in megawatts: an islanding contingency that cuts off an empty busbar is
    technically an island and demonstrates nothing.
    """
    total = 0.0
    for group in components(net):
        if CaseInvariants.SLACK_BUS in group:
            continue
        total += float(net.load[net.load.bus.isin(group)].p_mw.sum())
    return total


def split_bus(net, bus: int, to_busbar_b: list[str], coupler_closed: bool = True) -> int:
    """Model a double busbar: add a second bar and move the named circuits.

    A real double busbar exists so that circuits can be transferred between
    bars with isolators - which bar a circuit sits on is an operator choice,
    not a fixed property of the station (SPEC 5.5.4, added on Tarbela field
    evidence). So the caller names the circuits; nothing is hardcoded here.

    pandapower models the coupler as a bus-bus switch: closed fuses the bars
    into one electrical node at zero impedance, open leaves them separate.

    Returns the index of the new busbar.
    """
    if bus not in net.bus.index:
        raise KeyError(f"no bus {bus}")

    new_bus = pp.create_bus(
        net, vn_kv=float(net.bus.at[bus, "vn_kv"]), name=f"{bus}B"
    )

    for name in to_busbar_b:
        kind, idx = branch_index(net, name)
        if kind == "line":
            if int(net.line.at[idx, "from_bus"]) == bus:
                net.line.at[idx, "from_bus"] = new_bus
            elif int(net.line.at[idx, "to_bus"]) == bus:
                net.line.at[idx, "to_bus"] = new_bus
            else:
                raise ValueError(f"{name} does not touch bus {bus}")
        else:
            if int(net.trafo.at[idx, "hv_bus"]) == bus:
                net.trafo.at[idx, "hv_bus"] = new_bus
            elif int(net.trafo.at[idx, "lv_bus"]) == bus:
                net.trafo.at[idx, "lv_bus"] = new_bus
            else:
                raise ValueError(f"{name} does not touch bus {bus}")

    pp.create_switch(net, bus=bus, element=new_bus, et="b", closed=coupler_closed)
    return new_bus
