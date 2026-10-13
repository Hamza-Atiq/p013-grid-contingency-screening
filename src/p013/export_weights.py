"""PyTorch state dict -> ``weights.npz`` for the served forward pass. SPEC 5.1.

The served bundle contains NumPy and this file, never torch. Everything the
NumPy forward pass needs to reconstruct the computation - including the number
of message-passing rounds - travels in the archive, so the server never has to
be told the architecture separately and cannot be told it wrongly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def export(model, path: Path) -> dict[str, np.ndarray]:
    """Write ``weights.npz``. ``model`` is a :class:`p013.graph.PowerFlowGNN`."""
    arrays: dict[str, np.ndarray] = {
        name: tensor.detach().cpu().numpy().astype(np.float32)
        for name, tensor in model.state_dict().items()
    }

    # Architecture travels with the weights. Passing the round count separately
    # is one more thing that can disagree with the trained model.
    arrays["meta_rounds"] = np.array(model.rounds, dtype=np.int64)
    arrays["meta_hidden"] = np.array(model.hidden, dtype=np.int64)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return arrays


def main() -> None:
    import torch

    from p013.graph import PowerFlowGNN

    ap = argparse.ArgumentParser(description="Export weights for the NumPy forward pass")
    ap.add_argument("--weights", type=Path, default=Path("model/weights.pt"))
    ap.add_argument("--out", type=Path, default=Path("model/weights.npz"))
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=6)
    args = ap.parse_args()

    model = PowerFlowGNN(hidden=args.hidden, rounds=args.rounds)
    model.load_state_dict(torch.load(args.weights, weights_only=True))
    arrays = export(model, args.out)
    size_kb = args.out.stat().st_size / 1024
    print(f"exported {len(arrays)} arrays to {args.out} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
