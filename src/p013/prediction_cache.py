"""Every model output and every truth value, computed once and saved.

Research round 1 (plans/2026-09-17-research-round-system-design.md). The tasks
in that round all need the model's predictions on the held-out sweep. Running
the model inside each of them would mean several heavy processes on a 16 GB
laptop at once, which is how the previous round ran the machine out of memory.
So the predictions are computed here, once, and every task reads the file.

**The cache is checked against the original code, not trusted.** While it is
built, every model score and every model ordering is compared with
``p013.evaluate.model_ranking`` on every point, and every truth ordering with
``truth_ranking``. A cache that disagreed with the evaluation would make every
downstream number quietly wrong, so a disagreement is an error.

Arrays, with ``P = 250`` operating points, ``C = 80`` contingencies, 57 buses:

| Key | Shape | Meaning |
| --- | --- | --- |
| ``point`` | P | index into the sweep |
| ``is_test`` | P | True for the test split, False for validation (same split as evaluate); all False for a practice-only sweep |
| ``names`` | C | contingency names, canonical order; row offset ``c + 1`` in the sweep |
| ``pinned`` | P,C | islanded or no solution: always solved, never scored |
| ``dangerous`` | P,C | solver truth, pinned included |
| ``truth_score`` / ``model_score`` | P,C | ``risk_score``; NaN when pinned |
| ``truth_thermal`` / ``model_thermal`` | P,C | the thermal term, max flow / rating |
| ``truth_voltage`` / ``model_voltage`` | P,C | the voltage term, max deterioration in band-widths |
| ``truth_vbus`` / ``model_vbus`` | P,C | bus index where the voltage term peaks; -1 when pinned |
| ``truth_branch`` / ``model_branch`` | P,C | branch index where the thermal term peaks; -1 when pinned |
| ``truth_rank`` / ``model_rank`` | P,C | 1-based position in each ordering, pinned first |
| ``model_flags_danger`` | P,C | the model's own verdict from ``violations`` |
| ``truth_vm`` / ``model_vm`` | P,81,57 | row 0 is the intact state |
| ``truth_va_deg`` / ``model_va_deg`` | P,81,57 | angles, degrees on both sides |
| ``truth_loading`` / ``model_loading`` | P,81,80 | percent of rating |
| ``load_p_mw`` / ``load_q_mvar`` / ``gen_p_mw`` | P,42 / P,42 / P,6 | the point's inputs, as stored |

**Runtime-available versus not.** ``model_*``, ``pinned``, the inputs and the
names are all known before any AC solve. ``truth_*`` and ``dangerous`` are not.
A policy that decides what to verify may only use the first group.

Model predictions use the model's own predicted intact state as ``vm_before``,
which is the self-contained path the headline is measured on.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

DEFAULT_CACHE = Path("eval/cache/predictions.npz")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _terms(vm, loading, vm_before):
    """The two terms of ``risk_score`` and where each peaks."""
    from p013.danger import RATING_PERCENT, excess

    diff = excess(vm) - excess(vm_before)
    return (
        float(loading.max()) / RATING_PERCENT,
        float(diff.max()),
        int(diff.argmax()),
        int(loading.argmax()),
    )


def build(
    sweep_path: Path,
    weights_path: Path,
    out_path: Path = DEFAULT_CACHE,
    progress: bool = True,
    split: str = "evaluate",
) -> Path:
    """Build a cache. ``split`` is ``"evaluate"`` for the original sweep's
    validation/test split, or ``"practice"`` for an extra sweep whose every
    point is practice-only (research round 2): those points must never be
    marked as test, or they would silently join an exam.
    """
    if split not in ("evaluate", "practice"):
        raise ValueError(f"split must be 'evaluate' or 'practice', got {split!r}")
    from p013.admittance import loading_percent
    from p013.danger import risk_score, violations
    from p013.evaluate import (
        SPLIT_SEED,
        VALIDATION_FRACTION,
        Sweep,
        _predict_point,
        _ranked,
        model_ranking,
        split_points,
        truth_ranking,
    )
    from p013.forward_numpy import load_weights

    started = time.perf_counter()
    sweep = Sweep(sweep_path)
    weights = load_weights(weights_path)
    P, C, R = sweep.n_points, sweep.n_branch, sweep.rows_per_point
    n_bus = sweep.raw["vm_pu"].shape[1]

    if split == "evaluate":
        _, test = split_points(P, VALIDATION_FRACTION, SPLIT_SEED)
    else:
        test = []
    test_set = set(test)

    f32 = lambda *shape: np.full(shape, np.nan, dtype=np.float32)  # noqa: E731
    i16 = lambda *shape: np.full(shape, -1, dtype=np.int16)  # noqa: E731
    out = {
        "point": np.arange(P, dtype=np.int32),
        "is_test": np.array([p in test_set for p in range(P)]),
        "names": np.array(sweep.names),
        "pinned": np.zeros((P, C), dtype=bool),
        "dangerous": np.zeros((P, C), dtype=bool),
        "model_flags_danger": np.zeros((P, C), dtype=bool),
        "truth_rank": i16(P, C),
        "model_rank": i16(P, C),
        "truth_vm": f32(P, R, n_bus),
        "model_vm": f32(P, R, n_bus),
        "truth_va_deg": f32(P, R, n_bus),
        "model_va_deg": f32(P, R, n_bus),
        "truth_loading": f32(P, R, C),
        "model_loading": f32(P, R, C),
    }
    for side in ("truth", "model"):
        for field in ("score", "thermal", "voltage"):
            out[f"{side}_{field}"] = f32(P, C)
        out[f"{side}_vbus"] = i16(P, C)
        out[f"{side}_branch"] = i16(P, C)

    base_rows = [sweep.row(p, 0) for p in range(P)]
    out["load_p_mw"] = sweep.raw["load_p_mw"][base_rows].astype(np.float32)
    out["load_q_mvar"] = sweep.raw["load_q_mvar"][base_rows].astype(np.float32)
    out["gen_p_mw"] = sweep.raw["gen_p_mw"][base_rows].astype(np.float32)

    for p in range(P):
        # ---- truth -----------------------------------------------------
        truth = truth_ranking(sweep, p)
        vm_before_true = sweep.pre_outage(p)
        # Kept in float64 for the term split; the stored copy is float32.
        truth_loading = {}
        for o in range(R):
            i = sweep.row(p, o)
            out["truth_vm"][p, o] = sweep.raw["vm_pu"][i]
            out["truth_va_deg"][p, o] = sweep.raw["va_degree"][i]
            if sweep.raw["status"][i] == "converged":
                truth_loading[o] = loading_percent(
                    sweep.ppc, sweep.voltage_at(p, o), sweep.in_service_at(p, o)
                )
                out["truth_loading"][p, o] = truth_loading[o]

        # ---- model: one batched forward pass, as the screen does ---------
        vm, va = _predict_point(sweep, p, weights)
        out["model_vm"][p] = vm
        out["model_va_deg"][p] = np.rad2deg(va)
        vm_before = vm[0]

        scores, reports, pinned = {}, {}, []
        for o in range(R):
            loading = loading_percent(
                sweep.ppc, vm[o] * np.exp(1j * va[o]), sweep.in_service_at(p, o)
            )
            out["model_loading"][p, o] = loading
            if o == 0:
                continue
            c, name = o - 1, sweep.names[o - 1]
            if sweep.raw["status"][sweep.row(p, o)] != "converged":
                pinned.append(name)
                out["pinned"][p, c] = True
                continue

            score = risk_score(vm[o], loading, vm_before=vm_before)
            report = violations(vm[o], loading, vm_before=vm_before)
            scores[name], reports[name] = score, report
            th, vo, vb, br = _terms(vm[o], loading, vm_before)
            assert max(th, vo) == score, f"term split disagrees at point {p} {name}"
            out["model_score"][p, c] = score
            out["model_thermal"][p, c], out["model_voltage"][p, c] = th, vo
            out["model_vbus"][p, c], out["model_branch"][p, c] = vb, br
            out["model_flags_danger"][p, c] = report.is_dangerous

            ts = truth.score[name]
            th, vo, vb, br = _terms(
                sweep.raw["vm_pu"][sweep.row(p, o)], truth_loading[o], vm_before_true
            )
            assert max(th, vo) == ts, f"truth term split disagrees at point {p} {name}"
            out["truth_score"][p, c] = ts
            out["truth_thermal"][p, c], out["truth_voltage"][p, c] = th, vo
            out["truth_vbus"][p, c], out["truth_branch"][p, c] = vb, br

        mine = _ranked(sweep, scores, reports, pinned)

        # ---- the cache must agree with the evaluation, exactly ------------
        reference = model_ranking(sweep, p, weights)
        if mine.order != reference.order or mine.score != reference.score:
            raise AssertionError(f"cache disagrees with model_ranking at point {p}")
        if set(pinned) != set(truth.pinned):
            raise AssertionError(f"pinned sets disagree at point {p}")

        for c, name in enumerate(sweep.names):
            out["truth_rank"][p, c] = truth.order.index(name) + 1
            out["model_rank"][p, c] = mine.order.index(name) + 1
            out["dangerous"][p, c] = name in truth.dangerous

        if progress and (p + 1) % 25 == 0:
            print(f"  {p + 1}/{P} points, {time.perf_counter() - started:.0f} s", flush=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)

    meta = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.perf_counter() - started, 1),
        "sweep": str(sweep_path),
        "sweep_sha256": _sha256(Path(sweep_path)),
        "weights": str(weights_path),
        "weights_sha256": _sha256(Path(weights_path)),
        "split": split,
        "split_seed": SPLIT_SEED if split == "evaluate" else None,
        "operating_point_hashes_sha256": hashlib.sha256(
            "".join(sweep.raw["operating_point_hash"][[sweep.row(p, 0) for p in range(P)]]).encode()
        ).hexdigest(),
        "validation_fraction": VALIDATION_FRACTION,
        "n_points": P,
        "n_test": len(test),
        "vm_before": "model's own predicted intact state",
        "verified_against": "p013.evaluate.model_ranking and truth_ranking, every point",
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out_path


def load(path: Path = DEFAULT_CACHE) -> dict:
    """Read the cache into plain arrays, plus its build record under ``meta``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Build it with: "
            ".venv/Scripts/python.exe experiments/build_cache.py"
        )
    with np.load(path, allow_pickle=False) as raw:
        data = {key: raw[key] for key in raw.files}
    data["meta"] = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return data
