"""P-013: fast N-1 contingency screening with a learned power flow model.

See README.md at the repository root for what this is and how to run it.
Docstrings cite "SPEC" sections and plan files from the original project
notes, which are not part of this repository.
"""
from pathlib import Path

#: The vendored PGLib-OPF IEEE 57-bus case.
#:
#: Vendored rather than downloaded at run time so the build is reproducible
#: from a clean checkout with no network. SPEC 1.4 explains why this case and
#: not pandapower's built-in ``case57``: the built-in has ``rateA = 0`` on all
#: 80 branches, and the whole project is denominated in percent of rating.
CASE_PATH = Path(__file__).resolve().parents[2] / "cases" / "pglib_opf_case57_ieee.m"

__all__ = ["CASE_PATH"]
