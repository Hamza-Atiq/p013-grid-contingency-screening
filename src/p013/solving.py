"""The one place a power flow is solved. SPEC 2.4, 2.6.

FLAT START IS MANDATORY.

Measured 2026-09-09 on a full N-1 sweep of pandapower's built-in case57::

    warm start from base-case results   36 of 63 converged   22.2 s
    flat start                          59 of 63 converged    9.2 s

Warm-starting from the base case is the fastest configuration per solve (32 ms
against 43 ms) and it produces no error of any kind. It would have silently
labelled 23 solvable contingencies "no solution" across the whole training set,
teaching the model that a large slice of the system cannot be solved. It is
also *slower* over a whole sweep, because diverging costs full iterations
before the solver gives up.

The base-case voltages are simply a bad initial guess once a branch is gone.

Three outcomes, deliberately distinct (SPEC 2.4, amended 2026-09-10):

``islanded``
    A topology fact, settled exactly by :mod:`p013.topology` before the solver
    is asked anything. The network is in pieces.
``no_solution``
    The network is in one piece and Newton-Raphson still failed. Usually means
    the post-outage state is at or past what the system can serve.
``converged``
    An answer.

These used to be run together. They are different things, they have different
causes, and the console tells a visitor which one applies.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandapower as pp

from p013.topology import n_components

Status = Literal["converged", "islanded", "no_solution"]

_EMPTY = np.array([], dtype=float)


@dataclass(frozen=True)
class SolveResult:
    """A solved state, or a precise reason there is not one."""

    status: Status
    vm_pu: np.ndarray
    va_degree: np.ndarray
    loading_percent: np.ndarray

    @classmethod
    def failed(cls, status: Status) -> "SolveResult":
        """A failure carries no numbers, so nothing downstream can mistake a
        stale or zero-filled array for an answer."""
        return cls(status, _EMPTY, _EMPTY, _EMPTY)

    @property
    def ok(self) -> bool:
        return self.status == "converged"


def solve(net) -> SolveResult:
    """Solve the network, or say precisely why not.

    Mutates ``net`` (pandapower writes results onto it), so callers pass a
    fresh network from :func:`p013.case_loader.load_case`.
    """
    if n_components(net) > 1:
        return SolveResult.failed("islanded")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pp.runpp(net, init="flat", numba=True)
    except Exception:
        return SolveResult.failed("no_solution")

    if not net.converged:
        return SolveResult.failed("no_solution")

    loading = np.concatenate(
        [
            net.res_line.loading_percent.to_numpy(dtype=float),
            net.res_trafo.loading_percent.to_numpy(dtype=float),
        ]
    )
    return SolveResult(
        "converged",
        net.res_bus.vm_pu.to_numpy(dtype=float),
        net.res_bus.va_degree.to_numpy(dtype=float),
        loading,
    )
