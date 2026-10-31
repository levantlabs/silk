"""Tests for the fast-suite skip-reason guard added for TASK-103.

Before the fix, a fresh worktree reported 8 skips against the main
checkout's 6: two tests skipped on ``testSpice/ldo_example.cir`` and
``tests/browser/node_modules`` — the first because an earlier example run
had left a generated deck behind in the main tree only, the second a
legitimate machine-gate (npm install). The guard in ``conftest.py``
(``_skip_reason_names_generated_artifact`` / ``pytest_sessionfinish``) makes
the first kind of skip fail the whole session instead of silently drifting
between checkouts. These tests pin the predicate directly rather than
re-running the full suite as a subprocess (too slow to repeat per test).
"""

from tests.conftest import _skip_reason_names_generated_artifact


class TestSkipReasonNamesGeneratedArtifact:
    def test_flags_testspice_path(self):
        """The exact TASK-103 offender: ``testSpice/ldo_example.cir not
        found`` used to be a normal (unflagged) skip reason — it is the
        thing this guard exists to catch."""
        assert _skip_reason_names_generated_artifact(
            'testSpice/ldo_example.cir not found')

    def test_flags_examples_output_path(self):
        """A hypothetical sibling offender under examples/output/ — the
        other artifact directory the task calls out — must also be
        flagged, not just the one concrete case that was found."""
        assert _skip_reason_names_generated_artifact(
            'examples/output/some_report.json not found')

    def test_does_not_flag_node_modules_skip(self):
        """The browser e2e test's real, current skip reason: a legitimate
        build-step gate (npm install), not a leftover artifact. The guard
        must not flag it or the fast suite would fail on every checkout
        that has not run ``npm install`` in tests/browser."""
        assert not _skip_reason_names_generated_artifact(
            'tests/browser/node_modules missing (run `npm install` in '
            'tests/browser)')

    def test_does_not_flag_sg13g2_osdi_skip(self):
        """The machine-level SG13G2 OSDI skip reason, present in every
        checkout regardless of history — must never be flagged."""
        assert not _skip_reason_names_generated_artifact(
            'needs the sg13g2 PDK installed, its OSDI plugins compiled, '
            'and an ngspice that loads them ($SILK_NGSPICE_BIN; see '
            'docs/installing-the-pdk.md)')
