"""notebooks/p013_live.ipynb, the 'Run it live' link (spec 6a).

Runs every cell in a fresh kernel from the repository root, the way Colab
runs it after cloning, with the inline UI launch switched off.
"""
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "p013_live.ipynb"
pytestmark = pytest.mark.skipif(not (ROOT / "model" / "live_policy.npz").exists(),
                                reason="run experiments/build_live_policy.py first")


def test_notebook_clones_the_public_repo_and_uses_the_space_app():
    import nbformat

    nb = nbformat.read(NB, as_version=4)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "github.com/Hamza-Atiq/p013-grid-contingency-screening" in code
    assert "space/app.py" in code and "P013_NO_LAUNCH" in code


def test_notebook_runs_end_to_end(monkeypatch):
    import nbformat
    from nbclient import NotebookClient

    monkeypatch.setenv("P013_NO_LAUNCH", "1")
    nb = nbformat.read(NB, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    text = "".join(
        o.get("text", "") + "".join(o.get("data", {}).get("text/markdown", ""))
        for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
    )
    assert "Missed by the AI" in text
    assert not any(o.get("output_type") == "error" for c in nb.cells for o in c.get("outputs", []))


def test_notebook_does_not_overstate_score_differences():
    import nbformat

    text = "\n".join(c.source for c in nbformat.read(NB, as_version=4).cells)
    assert "does not depend on that" not in text


def test_inline_app_is_tall_enough_to_read():
    import nbformat

    code = "\n".join(c.source for c in nbformat.read(NB, as_version=4).cells if c.cell_type == "code")
    assert "launch(height=" in code
