"""A stored SpecReport's MetricWarnings round-trip through the database
(TASK-337): store_report() attaches each output's warnings under the
reserved ``conditions['silk_metric_warnings']`` key on that output's own
``simulation_results`` row -- no new table, no schema version bump.
"""
from __future__ import annotations

import math
import time

import pytest

from silk import Module
from silk.circuit.spec_report import SpecReport
from silk.db.schema import DATABASE_SCHEMA_VERSION
from silk.exceptions import EvidenceMismatchError
from silk.metrics.diagnostics import MetricWarningRecord


def _module(pdk, tmp_path, name, *, nets=('out',)):
    mod = Module(
        'sky130', str(tmp_path / name), f'{name}.cir', pdk_details=pdk,
        module_name=name, db_path=':memory:',
    )
    for net in nets:
        mod.add_net(net)
    mod.create_circuit_version()
    return mod


def _warning(output, metric='rise_time', code='zero_swing'):
    return MetricWarningRecord(
        metric=metric, code=code, message='msg', details={'a': 1},
        suggestion='sugg', output=output,
    )


def test_warnings_need_no_column_beyond_current_schema():
    """No schema bump for warnings: they reuse the existing free-form
    conditions JSON column (the same reserved-key convention as
    conditions['silk_operating_conditions']). Version 4 is TASK-306's
    provenance change, version 5 TASK-409's condition states, version 6
    aggregate evidence, and version 7 supply sweeps; none was introduced
    for warnings."""
    assert DATABASE_SCHEMA_VERSION == 7


class TestRoundTrip:

    def test_stored_report_read_back_carries_its_warnings(self, pdk, tmp_path):
        mod = _module(pdk, tmp_path, 'rt1')
        try:
            rec = _warning('rt')
            report = SpecReport(
                't', 'rt1', {'rt': float('nan')}, corner='tt',
                metric_warnings={'rt': (rec,)},
            )
            mod.store_report(report)

            result, = mod.query_results(parameter='rt')
            assert math.isnan(result.value)
            assert result.conditions['silk_metric_warnings'] == [rec.to_dict()]
        finally:
            mod.close()

    def test_two_outputs_sharing_one_metric_type_keep_separate_warnings(
        self, pdk, tmp_path,
    ):
        mod = _module(pdk, tmp_path, 'rt2', nets=('out', 'a', 'b'))
        try:
            report = SpecReport(
                't', 'rt2', {'rt_a': float('nan'), 'rt_b': float('nan')},
                corner='tt',
                metric_warnings={
                    'rt_a': (_warning('rt_a'),),
                    'rt_b': (_warning('rt_b'),),
                },
            )
            mod.store_report(report)

            a, = mod.query_results(parameter='rt_a')
            b, = mod.query_results(parameter='rt_b')
            assert a.conditions['silk_metric_warnings'][0]['output'] == 'rt_a'
            assert b.conditions['silk_metric_warnings'][0]['output'] == 'rt_b'
        finally:
            mod.close()

    def test_metric_with_no_warning_stores_no_warnings_key(self, pdk, tmp_path):
        mod = _module(pdk, tmp_path, 'rt3')
        try:
            report = SpecReport('t', 'rt3', {'clean': 1.0}, corner='tt')
            mod.store_report(report)
            result, = mod.query_results(parameter='clean')
            assert result.conditions is None
        finally:
            mod.close()

    def test_conditions_argument_still_reaches_metrics_with_no_warnings(
        self, pdk, tmp_path,
    ):
        """A metric with no warnings still writes the caller's own
        conditions, byte-identical to before this task."""
        mod = _module(pdk, tmp_path, 'rt4')
        try:
            report = SpecReport(
                't', 'rt4', {'clean': 1.0, 'rt': float('nan')}, corner='tt',
                metric_warnings={'rt': (_warning('rt'),)},
            )
            mod.store_report(report, conditions={'note': 'hello'})
            clean, = mod.query_results(parameter='clean')
            rt, = mod.query_results(parameter='rt')
            assert clean.conditions == {'note': 'hello'}
            assert rt.conditions == {
                'note': 'hello',
                'silk_metric_warnings': [_warning('rt').to_dict()],
            }
        finally:
            mod.close()


class TestReservedKeyGuard:

    def test_explicit_reserved_key_conflicts(self, pdk, tmp_path):
        mod = _module(pdk, tmp_path, 'rt5')
        try:
            report = SpecReport('t', 'rt5', {'clean': 1.0}, corner='tt')
            with pytest.raises(EvidenceMismatchError):
                mod.store_report(
                    report, conditions={'silk_metric_warnings': ['bogus']}
                )
        finally:
            mod.close()


class TestOldDatabaseCompatibility:

    def test_pre_existing_row_with_no_key_reads_back_as_no_warnings(
        self, pdk, tmp_path,
    ):
        """A row written before this task (or by any caller that never set
        the key) has no silk_metric_warnings entry; reading it back must
        not raise and must not fabricate a warning."""
        mod = _module(pdk, tmp_path, 'rt6')
        try:
            report = SpecReport('t', 'rt6', {'old': 1.0}, corner='tt')
            mod.store_report(report)
            result, = mod.query_results(parameter='old')
            assert result.conditions is None
        finally:
            mod.close()

    def test_pre_existing_conditions_without_the_key_are_unaffected(
        self, pdk, tmp_path,
    ):
        mod = _module(pdk, tmp_path, 'rt7')
        try:
            report = SpecReport('t', 'rt7', {'old': 1.0}, corner='tt')
            mod.store_report(report, conditions={'legacy': True})
            result, = mod.query_results(parameter='old')
            assert result.conditions == {'legacy': True}
        finally:
            mod.close()


class TestLatestOnlyIdentityExcludesWarnings:
    """(D): simulation_results.conditions feeds the ``latest_only=True``
    identity/dedup key in Database.get_simulation_results(); a metric that
    warned once and then didn't (or vice versa) must still collapse to its
    single newest reading, not survive as two "latest" rows."""

    def test_rerun_without_a_warning_supersedes_the_warning_bearing_run(
        self, pdk, tmp_path,
    ):
        mod = _module(pdk, tmp_path, 'rt8')
        try:
            first = SpecReport(
                't', 'rt8', {'rt': float('nan')}, corner='tt',
                metric_warnings={'rt': (_warning('rt'),)},
            )
            mod.store_report(first)
            time.sleep(0.01)
            second = SpecReport('t', 'rt8', {'rt': 1.23}, corner='tt')
            mod.store_report(second)

            everything = mod.query_results(parameter='rt')
            assert len(everything) == 2

            latest = mod.query_results(parameter='rt', latest_only=True)
            assert len(latest) == 1
            assert latest[0].value == 1.23
            assert latest[0].conditions is None
        finally:
            mod.close()

    def test_rerun_that_newly_warns_supersedes_the_clean_run(
        self, pdk, tmp_path,
    ):
        mod = _module(pdk, tmp_path, 'rt9')
        try:
            first = SpecReport('t', 'rt9', {'rt': 1.23}, corner='tt')
            mod.store_report(first)
            time.sleep(0.01)
            second = SpecReport(
                't', 'rt9', {'rt': float('nan')}, corner='tt',
                metric_warnings={'rt': (_warning('rt'),)},
            )
            mod.store_report(second)

            latest = mod.query_results(parameter='rt', latest_only=True)
            assert len(latest) == 1
            assert math.isnan(latest[0].value)
            assert latest[0].conditions['silk_metric_warnings'][0]['output'] == 'rt'
        finally:
            mod.close()
