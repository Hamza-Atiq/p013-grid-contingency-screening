"""Message passing over the bus graph, written by hand. SPEC 2.2.

Not PyTorch Geometric, for two reasons that both matter (SPEC 1.5): you can
defend code you wrote yourself in a meeting, and it keeps a large dependency
out of the served bundle.

How it works:

1. Each bus starts with a vector encoded from its own numbers.
2. Then several rounds. In each round every branch carries a message between
   the two buses it connects, built from the sending bus vector, the receiving
   bus vector, and that branch's own numbers. Each bus sums the messages it
   received and updates its vector.
3. After the last round a small head reads each bus vector and gives voltage
   magnitude and angle.

**The in-service flag is a branch input, and a branch that is out carries no
message.** That single property is the whole reason one trained model handles
every outage and every busbar split with no retraining - and because message
passing has no fixed bus count baked in, a 57-bus graph becoming 58 buses when
a coupler opens needs no architecture change at all. That is what made option C
affordable (SPEC 5.1).

Angles are predicted relative to the slack, which is pinned at zero. An
absolute angle is meaningless in a power flow solution and a model asked for
one is being asked to guess a constant (SPEC 2.1).

The model does **not** output branch flows. Those are derived from the
predicted voltages by :mod:`p013.admittance`, so the two can never disagree.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from p013.features import F_BRANCH, F_BUS


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class PowerFlowGNN(nn.Module):
    """Predict per-bus voltage magnitude and angle for any topology.

    Args:
        hidden: width of the bus vector. SPEC 2.2 plans 64 to 128.
        rounds: message-passing rounds. SPEC 2.2 plans 4 to 6.
    """

    def __init__(self, hidden: int = 64, rounds: int = 6) -> None:
        super().__init__()
        self.hidden = hidden
        self.rounds = rounds

        self.encode = _mlp([F_BUS, hidden, hidden])

        # One set of weights reused every round. Sharing keeps the parameter
        # count small enough to train on a dual-core CPU and makes the number
        # of rounds a runtime choice rather than an architectural one.
        self.message = _mlp([2 * hidden + F_BRANCH, hidden, hidden])
        self.update = _mlp([2 * hidden, hidden, hidden])

        self.head = _mlp([hidden, hidden, 2])

        # Start near a flat voltage profile: 1.0 pu, 0 radians. Without this
        # the first epochs are spent discovering that voltages live near 1.
        #
        # SMALL weights, not zero. Zeroing the last layer makes the output an
        # exact constant, which sets d(output)/d(h) to zero and so stops any
        # gradient reaching the encoder, the message function or the update -
        # the entire body of the network trains on nothing. It also makes every
        # behavioural test pass trivially, including the out-of-service test
        # that carries the project's central claim. Caught 2026-09-11 by a test
        # asserting that taking a branch out changes the prediction.
        with torch.no_grad():
            self.head[-1].weight.mul_(0.01)
            self.head[-1].bias.copy_(torch.tensor([1.0, 0.0]))

    def forward(
        self,
        bus_x: Tensor,
        branch_x: Tensor,
        edge_index: Tensor,
        slack_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            bus_x: ``(B, n_bus, F_BUS)``
            branch_x: ``(B, n_branch, F_BRANCH)``, last column the in-service flag
            edge_index: ``(2, n_branch)`` of zero-based bus positions
            slack_mask: ``(n_bus,)`` boolean, true at the slack bus

        Returns:
            ``(vm, va)``, each ``(B, n_bus)``. ``va`` is in radians relative to
            the slack.
        """
        src, dst = edge_index[0], edge_index[1]
        in_service = branch_x[..., -1:]  # (B, n_branch, 1)

        h = self.encode(bus_x)  # (B, n_bus, hidden)

        for _ in range(self.rounds):
            h_src = h.index_select(-2, src)
            h_dst = h.index_select(-2, dst)
            edge_in = torch.cat([h_src, h_dst, branch_x], dim=-1)

            # Both directions, because power flows either way along a branch.
            msg_forward = self.message(edge_in) * in_service
            msg_backward = self.message(
                torch.cat([h_dst, h_src, branch_x], dim=-1)
            ) * in_service

            aggregated = torch.zeros_like(h)
            aggregated.index_add_(-2, dst, msg_forward)
            aggregated.index_add_(-2, src, msg_backward)

            h = h + self.update(torch.cat([h, aggregated], dim=-1))

        out = self.head(h)
        vm, va = out[..., 0], out[..., 1]

        if slack_mask is not None:
            # Pin the slack angle at exactly zero rather than hoping the model
            # learns a constant it was never given a reason to learn.
            va = va - (va * slack_mask).sum(dim=-1, keepdim=True)

        return vm, va


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
