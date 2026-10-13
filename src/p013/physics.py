"""The physics term. SPEC 2.3.

From the predicted voltages we work out what power injection the network
equations say each bus should have, and compare that against the injections we
actually fed in. The gap is the **power mismatch**, and it goes in the loss.

This term needs no labels at all. It pushes the model toward answers that obey
the physics rather than answers that merely copy the training examples, and it
is the honest reply when a professor asks "is this just curve fitting?" -
partly yes, and this is what stops it drifting.

``loss = voltage error + lambda * power mismatch``

**What is enforced where.** A power flow does not specify everything
everywhere, and pretending otherwise would punish correct answers:

============  =================================  ====================
Bus type      P specified?                       Q specified?
============  =================================  ====================
PQ (50 here)  yes                                yes
PV (6 here)   yes                                no - solver chooses it
slack (1)     no - it absorbs the imbalance      no
============  =================================  ====================

**Computed branch by branch, not through a dense matrix.** The topology differs
from row to row, so a dense ``Ybus`` per sample would be 57x57 complex numbers
built 64 times a batch. Summing branch currents costs O(branches) instead, and
it mirrors how the model itself is organised. :mod:`p013.admittance` holds the
dense NumPy version, and a test asserts the two agree.
"""
from __future__ import annotations

import torch
from torch import Tensor


def branch_admittances(branch_x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The four per-branch admittance terms, with the in-service flag applied.

    ``branch_x`` columns are ``r, x, b, tap_ratio, shift_rad, in_service``.
    A branch that is out contributes exactly nothing.
    """
    r, x, b = branch_x[..., 0], branch_x[..., 1], branch_x[..., 2]
    ratio, shift, flag = branch_x[..., 3], branch_x[..., 4], branch_x[..., 5]

    series = 1.0 / torch.complex(r, x)
    charging = torch.complex(torch.zeros_like(b), b / 2.0)
    tap = torch.polar(ratio, shift)

    y_tt = series + charging
    y_ff = y_tt / (tap * tap.conj())
    y_ft = -series / tap.conj()
    y_tf = -series / tap

    mask = torch.complex(flag, torch.zeros_like(flag))
    return y_ff * mask, y_ft * mask, y_tf * mask, y_tt * mask


def power_injection(
    vm: Tensor,
    va: Tensor,
    branch_x: Tensor,
    edge_index: Tensor,
    shunt: Tensor,
) -> Tensor:
    """Complex power injected at each bus, per unit. ``(B, n_bus)`` complex.

    Args:
        vm: ``(B, n_bus)`` voltage magnitude
        va: ``(B, n_bus)`` voltage angle in radians
        branch_x: ``(B, n_branch, 6)``
        edge_index: ``(2, n_branch)`` zero-based bus positions
        shunt: ``(n_bus,)`` complex bus shunt admittance in per unit
    """
    src, dst = edge_index[0], edge_index[1]
    v = torch.polar(vm, va)

    y_ff, y_ft, y_tf, y_tt = branch_admittances(branch_x)
    v_f = v.index_select(-1, src)
    v_t = v.index_select(-1, dst)

    i_f = y_ff * v_f + y_ft * v_t
    i_t = y_tf * v_f + y_tt * v_t

    i_bus = torch.zeros_like(v)
    i_bus.index_add_(-1, src, i_f)
    i_bus.index_add_(-1, dst, i_t)
    i_bus = i_bus + shunt * v

    return v * i_bus.conj()


def power_mismatch_loss(
    vm: Tensor,
    va: Tensor,
    branch_x: Tensor,
    edge_index: Tensor,
    shunt: Tensor,
    s_specified: Tensor,
    p_mask: Tensor,
    q_mask: Tensor,
    row_mask: Tensor | None = None,
) -> Tensor:
    """Mean squared power mismatch where the injection is actually specified.

    Args:
        s_specified: ``(B, n_bus)`` complex, the injection we fed in
        p_mask: ``(n_bus,)`` 1 where real power is specified (PQ and PV)
        q_mask: ``(n_bus,)`` 1 where reactive power is specified (PQ only)
        row_mask: ``(B,)`` 1 for rows that have a valid target at all
    """
    s = power_injection(vm, va, branch_x, edge_index, shunt)

    d_p = (s.real - s_specified.real) * p_mask
    d_q = (s.imag - s_specified.imag) * q_mask

    per_row = (d_p.pow(2).sum(dim=-1) + d_q.pow(2).sum(dim=-1)) / (
        p_mask.sum() + q_mask.sum()
    )

    if row_mask is None:
        return per_row.mean()
    weight = row_mask.to(per_row.dtype)
    return (per_row * weight).sum() / weight.sum().clamp(min=1.0)


def bus_masks(bus_x: Tensor) -> tuple[Tensor, Tensor]:
    """``(p_mask, q_mask)`` from the one-hot bus type in the bus features.

    Columns 2, 3, 4 are is_pq, is_pv, is_slack (see :mod:`p013.features`).
    """
    is_pq, is_pv = bus_x[..., 2], bus_x[..., 3]
    if bus_x.dim() == 3:
        is_pq, is_pv = is_pq[0], is_pv[0]
    return is_pq + is_pv, is_pq
