"""The environment is part of the deliverable.

SPEC 1.5: the project lives in its own virtualenv with a pinned
requirements.txt, never in base conda - which on this machine is already broken
(pandas compiled against numpy 1.x, numpy 2.5.2 installed).
"""
import p013


def test_case_file_is_vendored_and_readable():
    assert p013.CASE_PATH.exists(), "the PGLib case must be vendored, not downloaded at runtime"
    text = p013.CASE_PATH.read_text(encoding="utf-8")
    assert "Power Grid Library" in text
    assert "case57" in text


def test_vendored_case_records_its_provenance():
    """A case file with no source is a number with no source. SPEC: every number
    that reaches a public surface is measured by us or quoted with its source."""
    head = p013.CASE_PATH.read_text(encoding="utf-8")[:600]
    assert "power-grid-lib" in head, "the vendored case must name where it came from"


def test_pinned_versions_match_what_the_spec_measured():
    import numpy
    import pandapower

    assert pandapower.__version__.startswith("3.5."), pandapower.__version__
    assert numpy.__version__.startswith("2."), numpy.__version__
