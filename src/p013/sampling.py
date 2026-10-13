"""Draw operating points and outage sets. SPEC 2.4.

**Outage depth is zero to three, not one.** Amended 2026-09-10 while designing
the console. With one outage per training sample the model learns topologies
with zero or one branch out. Screening from the base case then asks about
one-out states, which matches - but the console lets a visitor trip branches,
so the moment they click once we are asking about two-out states, and after two
clicks, three. Arbitrary topology is the console's entire selling point, and
the model was out of its depth on the second click.

The fix costs nothing: the same number of solves, the same wall clock.
"""
from __future__ import annotations

import numpy as np

from p013.naming import branch_names

#: One common factor across all loads, drawn uniformly. Gives lightly and
#: heavily loaded systems rather than variations on one condition.
LOAD_SCALE_RANGE = (0.6, 1.2)

#: Per-bus multiplicative jitter on top of the common factor, so the shape of
#: the load changes too and not merely its size.
PER_BUS_NOISE = 0.05

#: SPEC 2.4, amended 2026-09-10.
MAX_OUTAGES = 3


def sample_operating_point(net, rng: np.random.Generator) -> None:
    """Scale the loads and move generation to match. Mutates ``net``.

    P and Q scale by the same factor so the power factor is preserved - a
    system scaled only in P walks to a power factor no real operating point
    has, and the model would learn to explain voltages from an implausible
    reactive picture.

    Generation follows load. Without that the slack bus absorbs the entire
    change and every sample becomes a study of one generator.
    """
    factor = float(rng.uniform(*LOAD_SCALE_RANGE))
    noise = 1.0 + rng.normal(0.0, PER_BUS_NOISE, size=len(net.load))
    scale = factor * noise

    net.load["p_mw"] = net.load.p_mw.to_numpy() * scale
    net.load["q_mvar"] = net.load.q_mvar.to_numpy() * scale

    if len(net.gen):
        net.gen["p_mw"] = net.gen.p_mw.to_numpy() * factor


def sample_outages(net, rng: np.random.Generator) -> list[str]:
    """Zero to three distinct branches, uniform over depth then over branches.

    Uniform over *depth* rather than over subsets, so each depth appears about
    a quarter of the time. Sampling uniformly over subsets would make depth 3
    overwhelmingly the most common and leave the model barely trained on the
    intact system.
    """
    depth = int(rng.integers(0, MAX_OUTAGES + 1))
    if depth == 0:
        return []
    names = branch_names(net)
    chosen = rng.choice(len(names), size=depth, replace=False)
    return [names[int(i)] for i in chosen]
