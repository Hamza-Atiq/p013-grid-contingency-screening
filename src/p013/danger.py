"""What counts as dangerous, and how a case is scored. SPEC 3.1 and 3.2.

Two functions, and **both are called twice** - once on the solver's answer to
get ground truth, once on the model's predicted answer to get the ranking. That
is the whole design. SPEC 3.1's table splits *where the numbers come from*, never
*which checks run*; scoring the model against its own predicted labels would
measure nothing at all.

**A contingency is dangerous when either check fails:**

- any branch carries more than 100% of its rating, or
- any bus is outside 0.95-1.05 pu **and worse than before the outage**.

**How much deterioration counts, amended 2026-09-11.** A full band-width. The
2026-09-10 amendment below fixed the constant floor but left the trigger
infinitely sensitive: *any* deterioration counted, so a bus easing from 0.9370
to 0.9369 was a violation. Measured on the shipped case, that made **78 of 80
contingencies dangerous and k\* the entire list** - the headline was
unreachable before the model was ever consulted. The deteriorations span five
orders of magnitude, from 1e-5 pu to 0.36 pu, and a yes/no test at zero cannot
tell those apart. The *score* always could: its voltage term runs from 6.88
down to 0.0002.

At a full band-width the shipped case gives 14 dangerous cases at base load and
23 at 1.2x, with the perfect ranking reaching exactly those numbers.

An absolute floor was measured as the alternative - dangerous if a bus falls
outside 0.90-1.10 whatever its history - and **it collapses**. At a 1.2 load
factor the intact case already has buses under 0.90, so it flags all 80, which
is the pre-amendment failure one level down. It is not in the code for that
reason.

**Why the voltage check is relative.** Amended 2026-09-10, after the shipped
case was measured rather than assumed. On PGLib, **5 of 57 buses already sit
outside 0.95-1.05 with nothing wrong at all**, the worst at 0.937 pu. Under a
plain absolute band those five are violations in every contingency, so every
contingency is dangerous, ``k*`` degenerates to the whole sweep and the headline
means nothing. The score suffers worse than the label does: the old form
``|V - 1| / 0.05`` would have contributed a constant **1.26** from that one bus
to every case, putting a floor under the entire ranking - and a ranking with a
floor under it carries no information. So a bus is violated when the outage
**creates a violation or deepens an existing one**.

It is also what an operator actually asks after a switching operation: what got
*worse*, not what was already marginal.

**Thermal stays absolute**, because no branch is over its rating in the intact
case - measured at 51.6% heaviest, and pinned by a test rather than trusted, so
that a future case which starts overloaded cannot slip through.

The two terms are each divided by their own limit, so they share a scale with no
weighting to defend. A blended severity index with chosen weights would invite
an argument about the weights, and that argument buys nothing.

Islanding is not handled here. Those cases are pinned above everything without a
score at all (SPEC 2.5) and the model is never consulted for them.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

#: Voltages are healthy within 1.0 +/- BAND pu.
NOMINAL_PU = 1.0
BAND_PU = 0.05

#: ``loading_percent`` is already a percentage of the branch's own rating, so
#: the thermal limit is 100 and the normalised term is that over 100.
RATING_PERCENT = 100.0

#: A case is dangerous when its worst normalised violation exceeds this.
#:
#: **It is 1.0 on both axes, and that is not a tuned number.** On the thermal
#: side 1.0 already means "exactly at the rating" - the threshold is simply
#: what the units already say. Using the same 1.0 on the voltage side means a
#: bus must be pushed a full band-width, 0.05 pu, further outside the limits.
#:
#: **Any other value breaks the metric**, and this is structural rather than a
#: matter of taste. The label and the ranking are the same quantity here, so
#: they cannot disagree. Put the threshold at, say, half a band-width and they
#: do: a SAFE case scoring 0.9 on thermal then outranks a DANGEROUS one
#: scoring 0.6 on voltage. Measured on the shipped case at tau = 0.5, that is
#: 25 dangerous cases against a k* of 42 - a 17-case gap no model however
#: perfect can close, because it is in the definitions, not the predictions.
DANGER_THRESHOLD = 1.0


class DangerReport(NamedTuple):
    """Which of the two checks fired. Fields are plain ``bool``, not
    ``np.bool_``, because callers and tests compare them with ``is``."""

    thermal: bool
    voltage: bool

    @property
    def is_dangerous(self) -> bool:
        return self.thermal or self.voltage


def excess(vm) -> np.ndarray:
    """How far outside the band a voltage sits, in band-widths. Zero inside.

    ``excess(V) = max(0, |V - 1.0| / 0.05 - 1)``

    So 0.95 and 1.05 are exactly 0, and the 0.937 pu bus of the intact case is
    0.26 - a real number rather than the 1.26 the pre-amendment form gave it.
    """
    return np.maximum(0.0, np.abs(np.asarray(vm, dtype=float) - NOMINAL_PU) / BAND_PU - 1.0)


def _checked(vm, loading_percent, vm_before):
    """Arrays as floats, with the shapes actually verified.

    A 57-bus array silently broadcasting against a 56-bus one is how a whole
    evaluation comes out plausible and wrong, so it is an error here.
    """
    vm = np.asarray(vm, dtype=float)
    vm_before = np.asarray(vm_before, dtype=float)
    loading = np.asarray(loading_percent, dtype=float)
    if vm.shape != vm_before.shape:
        raise ValueError(
            f"vm {vm.shape} and vm_before {vm_before.shape} must be the same shape; "
            "the pre-outage state has to be the SAME operating point"
        )
    if vm.ndim != 1 or loading.ndim != 1:
        raise ValueError(f"expected 1-D arrays, got vm {vm.shape} and loading {loading.shape}")
    return vm, loading, vm_before


def violations(vm, loading_percent, vm_before) -> DangerReport:
    """Ground truth from the solver, or the same two checks on a prediction."""
    vm, loading, vm_before = _checked(vm, loading_percent, vm_before)

    thermal = bool((loading > RATING_PERCENT).any())

    now, before = excess(vm), excess(vm_before)
    voltage = bool(float((now - before).max()) > DANGER_THRESHOLD)

    return DangerReport(thermal=thermal, voltage=voltage)


def risk_score(vm, loading_percent, vm_before) -> float:
    """Worst normalised violation, for ranking. SPEC 3.2.

    ``risk = max( max(flow / rating), max(excess(V) - excess(V_before)) )``

    The voltage term is deliberately **not** clipped at zero: a contingency that
    improves every voltage should rank below one that leaves them alone, and
    clipping would tie them.
    """
    vm, loading, vm_before = _checked(vm, loading_percent, vm_before)

    thermal = float(loading.max()) / RATING_PERCENT if loading.size else 0.0
    voltage = float((excess(vm) - excess(vm_before)).max()) if vm.size else 0.0

    return max(thermal, voltage)
