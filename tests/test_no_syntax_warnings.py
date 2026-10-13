"""Every source file compiles with warnings as errors.

A stray backslash in a docstring printed a SyntaxWarning in the Colab
notebook, in front of every visitor (found on Colab, 13 Oct 2026).
"""
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(p for d in ("src", "space", "site", "experiments", "notebooks", "demo")
               for p in (ROOT / d).rglob("*.py") if "_build" not in p.parts)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_compiles_without_warnings(path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
