"""The served forward pass. Hand-written NumPy, no torch. SPEC 5.1.

This is the model that actually answers a visitor. It exists because of a
measurement: the in-browser JavaScript forward pass originally approved in 5.1
was benchmarked at **4.5 seconds** per screen of all 80 contingencies on the
target machine, against the 22 ms the mockup advertised, and IEEE 118-bus would
have been 23 seconds. Inference moved to a Python function on the server.

**NumPy and not torch**, for two reasons that both matter. torch is far too
large for a serverless bundle, while NumPy is small and starts fast. And
hand-writing the forward pass preserves the property that motivated the
original decision - he can point at every line - in the language he wants to
defend.

**Parity is a release gate, not a nice-to-have.** ``tests/test_forward_parity``
asserts this matches PyTorch to 1e-5. The console asserts things about the
model; if the served model differs from the trained one, the console is lying.

Importing this module must never import torch. There is a test for that.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_weights(path: Path) -> dict[str, np.ndarray]:
    """Read the exported ``weights.npz``."""
    raw = np.load(path, allow_pickle=False)
    return {key: raw[key] for key in raw.files}


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _mlp(weights: dict, prefix: str, x: np.ndarray, n_layers: int) -> np.ndarray:
    """Apply a stored MLP. SiLU between layers, nothing after the last."""
    for layer in range(n_layers):
        w = weights[f"{prefix}.{layer * 2}.weight"]
        b = weights[f"{prefix}.{layer * 2}.bias"]
        x = x @ w.T + b
        if layer < n_layers - 1:
            x = _silu(x)
    return x


def _index_add(target: np.ndarray, index: np.ndarray, source: np.ndarray) -> None:
    """``target[..., index, :] += source``, accumulating repeats.

    ``np.add.at`` is the unbuffered version. Plain fancy-index assignment
    silently keeps only the last write when an index repeats, and bus indices
    repeat constantly - every bus with more than one branch.
    """
    np.add.at(target, (slice(None), index), source)


def forward(
    weights: dict[str, np.ndarray],
    bus_x: np.ndarray,
    branch_x: np.ndarray,
    edge_index: np.ndarray,
    slack_mask: np.ndarray | None = None,
    rounds: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict ``(vm, va)`` for a batch of topologies.

    Mirrors :class:`p013.graph.PowerFlowGNN.forward` exactly.
    """
    if bus_x.ndim == 2:
        bus_x = bus_x[None]
        branch_x = branch_x[None]

    if rounds is None:
        rounds = int(weights["meta_rounds"])

    src, dst = edge_index[0], edge_index[1]
    in_service = branch_x[..., -1:]

    h = _mlp(weights, "encode", bus_x, n_layers=2)

    for _ in range(rounds):
        h_src = h[:, src, :]
        h_dst = h[:, dst, :]

        msg_forward = _mlp(
            weights, "message", np.concatenate([h_src, h_dst, branch_x], axis=-1), 2
        ) * in_service
        msg_backward = _mlp(
            weights, "message", np.concatenate([h_dst, h_src, branch_x], axis=-1), 2
        ) * in_service

        aggregated = np.zeros_like(h)
        _index_add(aggregated, dst, msg_forward)
        _index_add(aggregated, src, msg_backward)

        h = h + _mlp(weights, "update", np.concatenate([h, aggregated], axis=-1), 2)

    out = _mlp(weights, "head", h, n_layers=2)
    vm, va = out[..., 0], out[..., 1]

    if slack_mask is not None:
        va = va - (va * slack_mask).sum(axis=-1, keepdims=True)

    return vm, va
