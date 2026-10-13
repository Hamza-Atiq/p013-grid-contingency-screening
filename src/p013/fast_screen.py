"""A faster screening overhead path, with output identical to the original.

Research round 1, task B (plans/2026-09-17-research-round-system-design.md).

**Why this exists.** Speedup is ``80 x solve / (overhead + k x solve)``. The
screen's overhead - everything it does before its first AC solve - was
measured at **0.626 s, 12.5 AC solves** (EVAL_NOTES §4), and nobody had ever
profiled it. At the ``k = 26`` a perfect ranking needs, that overhead alone
holds the ceiling at 2.08x instead of 3.08x.

**Where the time went, profiled 2026-09-17** (shared machine; the numbers are
in ``experiments/EXP_B_NOTES.md``): about three quarters in the forward pass -
and inside it ``np.add.at``, numpy's slow unbuffered scatter, run 12 times -
then ``loading_percent``, a Python loop over 80 branches called 80 times, then
the islanding pre-filter and 81 calls to ``branch_features``.

**The release gate is parity, not speed.** A faster screen that ranks
differently is a different model, and every number in EVAL_NOTES would have
to be re-measured. Nothing here approximates anything:

- **The forward pass is bit-identical, and that constrains it hard.**
  Measured: the batched matmul ``(81, 80, 134) @ W.T`` and the same numbers
  reshaped into one ``(6480, 134)`` matmul differ by up to **1.2e-7** in the
  output voltages, because OpenBLAS picks a different kernel. That moves risk
  scores by ~1e-6, far above the 1e-9 parity bar. So every matmul keeps its
  original item shape, and the saving comes from computing *fewer entries*:
  row 0 once, plus only what an outage actually changes (see
  :meth:`ScreenCase.forward`). The scatter keeps ``np.add.at``'s exact
  sequence of float32 additions.
- **Branch flows use the same scalar arithmetic.** ``y_ff`` and ``y_ft`` are
  built once per case by the *same* Python scalar expressions
  :func:`p013.admittance.branch_flows` uses, and complex products are written
  out so they match numpy's scalar path bit for bit (see :func:`_cmul`).
- **The islanding check finds bridges** (one depth-first search) instead of
  running 80 breadth-first searches. It still reads the live network on every
  call, so a new topology is never answered from a cache.

**What is one-off, and why that is fair.** :class:`ScreenCase` is built once
per *case and model*: case matrices, edge index, admittance terms, base branch
features, scatter tables. That is the same class of cost SPEC 3.5 already
excludes - loading weights, building ``CaseTemplate``. Everything that depends
on the operating point or the topology - bus features, in-service flags, the
forward pass, flows, scores, the islanding search - is paid on every screen.

**Parity is a property of this machine's numpy and OpenBLAS**, exactly as the
original's outputs are. It is verified on all 250 sweep points by
``experiments/exp_b_profile.py --parity``; re-run that after any library
upgrade before trusting a timing.

**Not for a split busbar.** A :func:`p013.topology.split_bus` network has a
different bus count, so it needs its own ``ScreenCase``, just as it needs its
own model inputs. The islanding function has no such limit: it reads the net.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from p013.admittance import (
    BR_ANGLE,
    BR_B,
    BR_F,
    BR_R,
    BR_RATE_A,
    BR_RATIO,
    BR_STATUS,
    BR_T,
    BR_X,
    GEN_BUS,
    GEN_PG,
    GEN_STATUS,
    bus_position,
    canonical_branch_rows,
    to_canonical_order,
)
from p013.danger import BAND_PU, DANGER_THRESHOLD, NOMINAL_PU, RATING_PERCENT, DangerReport
from p013.features import branch_features, bus_features, edge_index


def _cmul(a, b) -> np.ndarray:
    """Complex multiply with numpy's SCALAR arithmetic, over whole arrays.

    Measured 2026-09-17 on numpy 2.4.6: ``a * b`` on complex128 arrays and the
    same product taken one numpy scalar at a time differ in the last bit (the
    array loop is vectorised differently). ``branch_flows`` multiplies scalars,
    so vectorising it naively moved loading by up to 6.4e-14 %. The textbook
    formula written out as float ufuncs reproduces the scalar result exactly,
    verified on 1,000 random pairs and on every sweep point.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    ar, ai, br, bi = a.real, a.imag, b.real, b.imag
    out = np.empty(np.broadcast_shapes(a.shape, b.shape), dtype=np.complex128)
    out.real = ar * br - ai * bi
    out.imag = ar * bi + ai * br
    return out


def _differs_from_row0(x: np.ndarray) -> np.ndarray:
    """``[b, n]`` True where ``x[b, n]`` is not bitwise equal to ``x[0, n]``.

    Bitwise, not ``!=``: 0.0 and -0.0 compare equal but are different inputs.
    """
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return (bits != bits[0:1]).any(axis=-1)


class _Changed:
    """Which entries of a ``(rows, P)`` quantity may differ from row 0.

    ``row``/``pos`` list them in C order. ``slot`` is each entry's rank among
    the changed entries at the same position - the extra matmul item it is
    packed into. ``index[b, p]`` is ``p`` for an unchanged entry and ``P + k``
    for the k-th changed one, so ``concat([row0, changed])[index]`` reads the
    full array without ever building it.
    """

    def __init__(self, mask: np.ndarray):
        mask = mask.copy()
        mask[0] = False  # row 0 is the reference by definition
        self.mask = mask
        self.row, self.pos = np.nonzero(mask)
        self.count = len(self.row)
        width = mask.shape[1]
        if self.count:
            self.slot = (np.cumsum(mask, axis=0) - 1)[self.row, self.pos]
            self.depth = int(self.slot.max()) + 1
        else:
            self.slot = self.row
            self.depth = 0
        self.index = np.broadcast_to(np.arange(width), mask.shape).copy()
        self.index[self.row, self.pos] = width + np.arange(self.count)


class ScreenCase:
    """Everything about one case and one model that no operating point changes.

    Built once, like ``CaseTemplate``. ``load_bus`` and ``gen_bus`` are the
    element-to-bus maps the sweep stores an operating point in.
    """

    def __init__(self, ppc, names: Sequence[str], load_bus, gen_bus, weights: dict):
        self.ppc = ppc
        self.names = tuple(names)
        self.n_branch = len(self.names)
        self.position = {name: i for i, name in enumerate(self.names)}
        self.weights = weights
        self.rounds = int(weights["meta_rounds"])
        self.base_mva = ppc["baseMVA"]
        pos = bus_position(ppc)
        self.n_bus = len(ppc["bus"])

        # --- bus features: the columns no operating point touches ----------
        self.load_pos = np.array([pos[int(b)] for b in load_bus], dtype=np.intp)
        self.gen_pos = np.array([pos[int(b)] for b in gen_bus], dtype=np.intp)
        sampled = {int(b) for b in gen_bus}
        # The slack's dispatch, added exactly as evaluate._with_slack adds it:
        # one scalar add per unsampled in-service generator, in case order.
        self.slack_adds = [
            (pos[int(row[GEN_BUS])], row[GEN_PG])
            for row in ppc["gen"]
            if not (row[GEN_STATUS] <= 0 or int(row[GEN_BUS]) in sampled)
        ]
        self.bus_static = bus_features(ppc)[:, 2:].copy()

        # --- branch features and graph -------------------------------------
        self.branch_base32 = branch_features(ppc).astype(np.float32)
        self.edge_index = edge_index(ppc)
        src, dst = self.edge_index
        self.src, self.dst = src, dst

        # np.add.at(agg, dst, forward) then np.add.at(agg, src, backward) adds,
        # for each bus, its incoming messages one at a time in that order.
        # Slot s holds every bus's s-th incoming message, so applying the slots
        # in order performs the identical additions in the identical order,
        # with no bus repeated inside one fancy-index add.
        incoming: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for j in range(len(dst)):
            incoming[int(dst[j])].append((0, j))
        for j in range(len(src)):
            incoming[int(src[j])].append((1, j))
        depth = max(len(v) for v in incoming.values())
        self.slots = []
        # The same slots looked up by bus: slot_kind[s][n] is 0 (forward),
        # 1 (backward) or -1 (bus n has fewer than s + 1 incoming messages).
        self.slot_kind, self.slot_edge = [], []
        for s in range(depth):
            fwd_b, fwd_j, bwd_b, bwd_j = [], [], [], []
            kind = np.full(self.n_bus, -1, dtype=np.int8)
            edge = np.zeros(self.n_bus, dtype=np.intp)
            for bus in sorted(incoming):
                if len(incoming[bus]) > s:
                    direction, j = incoming[bus][s]
                    (fwd_b if direction == 0 else bwd_b).append(bus)
                    (fwd_j if direction == 0 else bwd_j).append(j)
                    kind[bus], edge[bus] = direction, j
            self.slots.append(
                tuple(np.array(a, dtype=np.intp) for a in (fwd_b, fwd_j, bwd_b, bwd_j))
            )
            self.slot_kind.append(kind)
            self.slot_edge.append(edge)

        # incidence[j, n] = 1 when branch j carries a message into bus n, in
        # either direction. Used only to track which aggregates can differ.
        self.incidence = np.zeros((len(src), self.n_bus), dtype=np.uint8)
        self.incidence[np.arange(len(src)), dst] = 1
        self.incidence[np.arange(len(src)), src] = 1

        # --- branch flows ----------------------------------------------------
        branch = ppc["branch"]
        self.canon_rows = canonical_branch_rows(ppc)
        n = len(branch)
        self.f = np.zeros(n, dtype=np.intp)
        self.t = np.zeros(n, dtype=np.intp)
        self.y_ff = np.zeros(n, dtype=complex)
        self.y_ft = np.zeros(n, dtype=complex)
        self.status_ok = branch[:, BR_STATUS] != 0
        for k, row in enumerate(branch):
            # The same scalar expressions as admittance.branch_flows, on
            # purpose: Python's complex division and numpy's differ in the
            # last bit, and parity is judged at 1e-9.
            f, t = pos[int(row[BR_F])], pos[int(row[BR_T])]
            series = 1.0 / complex(row[BR_R], row[BR_X])
            ratio = row[BR_RATIO] if row[BR_RATIO] != 0 else 1.0
            tap = ratio * np.exp(1j * np.deg2rad(row[BR_ANGLE]))
            self.f[k], self.t[k] = f, t
            self.y_ff[k] = (series + 1j * row[BR_B] / 2.0) / (tap * np.conj(tap))
            self.y_ft[k] = -series / np.conj(tap)
        self.rate = to_canonical_order(ppc, branch[:, BR_RATE_A])
        if not (self.rate > 0).all():
            raise ValueError("a branch has no rating, so its loading is undefined")

    @classmethod
    def from_sweep(cls, sweep, weights: dict) -> "ScreenCase":
        return cls(sweep.ppc, sweep.names, sweep._load_bus, sweep._gen_bus, weights)

    # ------------------------------------------------------------------ inputs

    def bus_features_at(self, load_p_mw, load_q_mvar, gen_p_mw) -> np.ndarray:
        """``(n_bus, 8)`` float64, identical to what model_ranking builds."""
        n = self.n_bus
        load_p = np.zeros(n)
        load_q = np.zeros(n)
        gen_p = np.zeros(n)
        np.add.at(load_p, self.load_pos, np.asarray(load_p_mw))
        np.add.at(load_q, self.load_pos, np.asarray(load_q_mvar))
        np.add.at(gen_p, self.gen_pos, np.asarray(gen_p_mw))
        for i, pg in self.slack_adds:
            gen_p[i] += pg
        base = self.base_mva
        out = np.empty((n, 8))
        out[:, 0] = gen_p / base - load_p / base
        out[:, 1] = -(load_q / base)
        out[:, 2:] = self.bus_static
        return out

    def branch_x(self, in_service: np.ndarray) -> np.ndarray:
        """``(rows, n_branch, 6)`` float32 for a stack of in-service patterns."""
        out = np.empty((len(in_service), self.n_branch, 6), dtype=np.float32)
        out[:] = self.branch_base32
        out[..., 5] = in_service
        return out

    # ------------------------------------------------------------ forward pass

    def _mlp(self, prefix: str, x: np.ndarray) -> np.ndarray:
        """Two layers, SiLU between. The same ufuncs as forward_numpy._mlp,
        without the temporaries."""
        w = self.weights
        y = np.matmul(x, w[f"{prefix}.0.weight"].T)
        y += w[f"{prefix}.0.bias"]
        s = np.negative(y)
        np.exp(s, out=s)
        s += 1.0
        np.divide(y, s, out=y)
        z = np.matmul(y, w[f"{prefix}.2.weight"].T)
        z += w[f"{prefix}.2.bias"]
        return z

    def _run(self, prefix: str, x0: np.ndarray, xc: np.ndarray, where: _Changed):
        """The MLP on the row-0 values ``x0`` ``(P, F)`` and on the changed
        entries ``xc`` ``(C, F)``. Returns ``(y0, yc)``.

        **Why each changed entry keeps its position.** Measured 2026-09-17:
        the OpenBLAS kernel gives bit-different results for the same input row
        at different positions inside one ``(P, F)`` matmul item - positions
        fall into three classes, blocks of six plus a tail. It does NOT depend
        on the other rows' values, nor on how many items are in the batch. So
        changed entries are packed into extra items, each at its own position.
        Unused places are zeros and their output is thrown away. The work is
        set by the busiest position, not by the total.
        """
        y0 = self._mlp(prefix, x0[None])[0]
        if not where.count:
            return y0, y0[:0]
        packed = np.zeros((where.depth,) + x0.shape, dtype=x0.dtype)
        packed[where.slot, where.pos] = xc
        return y0, self._mlp(prefix, packed)[where.slot, where.pos]

    def forward(self, bus_x, branch_x, slack_mask) -> tuple[np.ndarray, np.ndarray]:
        """Bit-identical to :func:`p013.forward_numpy.forward`.

        **Only what differs from row 0 is computed.** In a screen, row 0 is
        the intact network and each other row removes one branch. A removal
        changes the state at two buses, and the change spreads one hop per
        message-passing round. Measured on sweep point 3, the share of message
        inputs that differ from row 0 is 1%, 7%, 17%, 32%, 50% and 67% over
        the six rounds.

        So every quantity is held as its row-0 value plus a list of the
        entries that differ, and each MLP runs on row 0 once plus those
        entries. Which entries *can* differ is tracked from the inputs, never
        guessed: a message input differs if either end bus's state or the
        branch's features differ; a bus's aggregate differs if any incoming
        message does; a bus's new state differs if its state or its aggregate
        does. Tracking a superset is always safe - it only costs work.

        Aggregation still adds each bus's incoming messages one at a time,
        forward direction first, in edge order: the exact float32 additions
        ``np.add.at`` performs.
        """
        rows, n_bus, _ = bus_x.shape
        n_edge = branch_x.shape[1]
        src, dst = self.src, self.dst
        in_service = branch_x[..., -1:]
        branch_changed = _differs_from_row0(branch_x)

        where = _Changed(_differs_from_row0(bus_x))
        h0, hc = self._run("encode", bus_x[0], bus_x[where.row, where.pos], where)
        h_where = where

        for _ in range(self.rounds):
            h_table = np.concatenate([h0, hc])
            h_index = h_where.index

            # --- messages, both directions ----------------------------------
            m = _Changed(h_where.mask[:, src] | h_where.mask[:, dst] | branch_changed)
            hs = h_table[h_index[m.row, src[m.pos]]]
            hd = h_table[h_index[m.row, dst[m.pos]]]
            br = branch_x[m.row, m.pos]
            hs0, hd0 = h0[src], h0[dst]
            f0, fc = self._run(
                "message",
                np.concatenate([hs0, hd0, branch_x[0]], axis=-1),
                np.concatenate([hs, hd, br], axis=-1),
                m,
            )
            b0, bc = self._run(
                "message",
                np.concatenate([hd0, hs0, branch_x[0]], axis=-1),
                np.concatenate([hd, hs, br], axis=-1),
                m,
            )
            ins_c = in_service[m.row, m.pos]
            f0 *= in_service[0]
            fc *= ins_c
            b0 *= in_service[0]
            bc *= ins_c
            msg_table = np.concatenate([f0, fc, b0, bc])
            bwd_offset = n_edge + m.count

            # --- aggregate ----------------------------------------------------
            a = _Changed((m.mask.astype(np.uint8) @ self.incidence) > 0)
            agg0 = np.zeros_like(h0)
            aggc = np.zeros((a.count, h0.shape[1]), dtype=h0.dtype)
            for s, (fwd_b, fwd_j, bwd_b, bwd_j) in enumerate(self.slots):
                if len(fwd_b):
                    agg0[fwd_b] += msg_table[fwd_j]
                if len(bwd_b):
                    agg0[bwd_b] += msg_table[bwd_offset + bwd_j]
                if a.count:
                    kind = self.slot_kind[s][a.pos]
                    edge = self.slot_edge[s][a.pos]
                    k = np.flatnonzero(kind == 0)
                    if k.size:
                        aggc[k] += msg_table[m.index[a.row[k], edge[k]]]
                    k = np.flatnonzero(kind == 1)
                    if k.size:
                        aggc[k] += msg_table[bwd_offset + m.index[a.row[k], edge[k]]]
            agg_table = np.concatenate([agg0, aggc])

            # --- update -------------------------------------------------------
            u = _Changed(h_where.mask | a.mask)
            h_old = h_table[h_index[u.row, u.pos]]
            u0, uc = self._run(
                "update",
                np.concatenate([h0, agg0], axis=-1),
                np.concatenate([h_old, agg_table[a.index[u.row, u.pos]]], axis=-1),
                u,
            )
            h0 = h0 + u0
            hc = h_old + uc
            h_where = u

        y0, yc = self._run("head", h0, hc, h_where)
        out = np.empty((rows, n_bus, y0.shape[1]), dtype=y0.dtype)
        out[:] = y0
        out[h_where.row, h_where.pos] = yc
        vm, va = out[..., 0], out[..., 1]
        va = va - (va * slack_mask).sum(axis=-1, keepdims=True)
        return vm, va

    # ------------------------------------------------------- flows and scores

    def loading_percent(self, vm, va, in_service) -> np.ndarray:
        """``(rows, n_branch)``, identical to admittance.loading_percent per row."""
        v = vm * np.exp(1j * va)  # complex64, exactly as model_ranking builds it
        v_from = v[:, self.f].astype(np.complex128)
        v_to = v[:, self.t].astype(np.complex128)
        i_f = _cmul(self.y_ff, v_from) + _cmul(self.y_ft, v_to)
        flow = _cmul(v_from, np.conj(i_f))

        live = np.empty(in_service.shape, dtype=bool)
        live[:, self.canon_rows] = in_service  # canonical -> MATPOWER order
        live &= self.status_ok
        flow = np.where(live, flow, 0.0)

        mva = np.abs(flow[:, self.canon_rows]) * self.base_mva
        return mva / self.rate * 100.0

    @staticmethod
    def _excess(vm) -> np.ndarray:
        return np.maximum(0.0, np.abs(vm - NOMINAL_PU) / BAND_PU - 1.0)

    def scores(self, vm, loading, vm_before):
        """Risk score and both danger checks for every row at once."""
        vm = np.asarray(vm, dtype=float)
        vm_before = np.asarray(vm_before, dtype=float)
        thermal_term = loading.max(axis=1) / RATING_PERCENT
        voltage_term = (self._excess(vm) - self._excess(vm_before)[None]).max(axis=1)
        # Python's max(thermal, voltage) keeps `thermal` unless voltage is
        # strictly larger; np.maximum would differ on NaN.
        risk = np.where(voltage_term > thermal_term, voltage_term, thermal_term)
        thermal = (loading > RATING_PERCENT).any(axis=1)
        voltage = voltage_term > DANGER_THRESHOLD
        return risk, thermal, voltage


def fast_model_ranking(case: ScreenCase, sweep, point: int, pre_outage: str = "predicted"):
    """Drop-in for :func:`p013.evaluate.model_ranking`. Same PointRanking."""
    from p013.evaluate import PointRanking

    if pre_outage not in ("predicted", "measured"):
        raise ValueError(
            f"pre_outage must be 'predicted' or 'measured', got {pre_outage!r}"
        )
    rows = sweep.rows_per_point
    base = sweep.row(point, 0)
    raw = sweep.raw

    bus_one = case.bus_features_at(
        raw["load_p_mw"][base], raw["load_q_mvar"][base], raw["gen_p_mw"][base]
    )
    bus_x = np.repeat(bus_one[None].astype(np.float32), rows, axis=0)
    in_service = raw["in_service"][base:base + rows]
    branch_x = case.branch_x(in_service)
    slack_mask = bus_one[:, 4].astype(np.float32)

    vm, va = case.forward(bus_x, branch_x, slack_mask)
    vm_before = vm[0] if pre_outage == "predicted" else sweep.pre_outage(point)

    loading = case.loading_percent(vm[1:], va[1:], in_service[1:])
    risk, thermal, voltage = case.scores(vm[1:], loading, vm_before)

    converged = raw["status"][base + 1:base + rows] == "converged"
    scores: dict[str, float] = {}
    reports: dict[str, DangerReport] = {}
    pinned: list[str] = []
    for i, name in enumerate(case.names):
        if not converged[i]:
            # Pinned exactly as model_ranking pins them; never scored.
            pinned.append(name)
            continue
        scores[name] = float(risk[i])
        reports[name] = DangerReport(thermal=bool(thermal[i]), voltage=bool(voltage[i]))

    position = case.position
    rest = sorted(scores, key=lambda n: (-scores[n], position[n]))
    return PointRanking(
        order=tuple(pinned) + tuple(rest),
        score=scores,
        report=reports,
        pinned=tuple(pinned),
    )


# ---------------------------------------------------------------------------
# Islanding
# ---------------------------------------------------------------------------


def fast_islanding_contingencies(net) -> set[str]:
    """Identical output to :func:`p013.topology.islanding_contingencies`.

    That function removes each live branch in turn and counts components: 80
    breadth-first searches. A branch whose removal raises the count is, by
    definition, a *bridge*, and all bridges come out of one depth-first search
    (Tarjan's low-link). Two details keep the answers equal:

    - **parallel branches are never bridges**, so the search skips only the
      edge it arrived by (by id), not every edge back to the parent bus;
    - **closed bus couplers join the graph but are never candidates**, because
      the original never removes them either.

    Names follow :func:`p013.naming.branch_table`: lines ``L01..`` in frame
    order, then transformers ``T01..``. The recursion is an explicit stack, so
    a large case cannot hit Python's recursion limit.
    """
    line_from = net.line.from_bus.to_numpy()
    line_to = net.line.to_bus.to_numpy()
    line_live = net.line.in_service.to_numpy()
    trafo_from = net.trafo.hv_bus.to_numpy()
    trafo_to = net.trafo.lv_bus.to_numpy()
    trafo_live = net.trafo.in_service.to_numpy()

    names: list[str] = []
    ends: list[tuple[int, int]] = []
    for i in range(len(line_from)):
        if bool(line_live[i]):
            names.append(f"L{i + 1:02d}")
            ends.append((int(line_from[i]), int(line_to[i])))
    for i in range(len(trafo_from)):
        if bool(trafo_live[i]):
            names.append(f"T{i + 1:02d}")
            ends.append((int(trafo_from[i]), int(trafo_to[i])))
    n_candidates = len(ends)

    if hasattr(net, "switch") and len(net.switch):
        et = net.switch.et.to_numpy()
        closed = net.switch.closed.to_numpy()
        bus = net.switch.bus.to_numpy()
        element = net.switch.element.to_numpy()
        for i in range(len(et)):
            if et[i] == "b" and bool(closed[i]):
                ends.append((int(bus[i]), int(element[i])))

    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for e, (a, b) in enumerate(ends):
        if a == b:
            continue  # a self-loop never disconnects anything
        adjacency[a].append((b, e))
        adjacency[b].append((a, e))

    disc: dict[int, int] = {}
    low: dict[int, int] = {}
    found: set[str] = set()
    counter = 0
    for root in [int(b) for b in net.bus.index]:
        if root in disc:
            continue
        disc[root] = low[root] = counter
        counter += 1
        stack = [(root, -1, 0)]  # (bus, edge arrived by, next neighbour)
        while stack:
            node, via, k = stack[-1]
            neighbours = adjacency[node]
            if k < len(neighbours):
                stack[-1] = (node, via, k + 1)
                nxt, e = neighbours[k]
                if e == via:
                    continue
                if nxt in disc:
                    if disc[nxt] < low[node]:
                        low[node] = disc[nxt]
                else:
                    disc[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append((nxt, e, 0))
            else:
                stack.pop()
                if stack:
                    parent = stack[-1][0]
                    if low[node] < low[parent]:
                        low[parent] = low[node]
                    if low[node] > disc[parent] and via < n_candidates:
                        found.add(names[via])
    return found
