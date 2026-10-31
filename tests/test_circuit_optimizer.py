"""
Tests for Optimizer — evaluation harness for circuit optimization.

All tests use mocked ModuleTestbench.run() so no ngspice is needed.
The mock lets each test control exactly what the simulation returns
(success with metrics, SimulationError, CircuitError, bad metric, etc.)
"""

from copy import deepcopy
from unittest.mock import Mock
import time
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from silk.circuit.variable import Variable
from silk.circuit.spec_report import SpecReport
from silk.optimize.parameter_space import VariableSpace
from silk.optimize.optimizer import Optimizer
from silk.exceptions import CircuitError, SimulationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def space():
    """Two-variable bounded space."""
    w = Variable('w', value=1.0, bounds=(0.5, 20.0))
    l = Variable('l', value=0.15, bounds=(0.15, 1.0))
    return VariableSpace([w, l])


def _make_mock_test(return_value=None, side_effect=None):
    """Return a stub whose .run() either returns a value or raises."""
    class MockTest:
        def run(self, corner='tt'):
            if side_effect is not None:
                raise side_effect
            return return_value
    return MockTest()


def _success_test(gain=45.0, power=300.0):
    return _make_mock_test({'metrics': {'gain_dB': gain, 'power_uW': power}})


# ---------------------------------------------------------------------------
# Constructor and configuration
# ---------------------------------------------------------------------------

class TestCircuitOptimizerConfig:

    def test_construction(self, space):
        test = _success_test()
        opt = Optimizer(test, space)
        assert opt.space is space
        assert opt.test is test
        assert opt.history == []

    def test_add_objective_returns_self(self, space):
        opt = Optimizer(_success_test(), space)
        result = opt.add_objective('gain_dB', minimize=False)
        assert result is opt

    def test_add_constraint_returns_self(self, space):
        opt = Optimizer(_success_test(), space)
        result = opt.add_constraint('power_uW', max=500.0)
        assert result is opt

    def test_objectives_stored(self, space):
        opt = Optimizer(_success_test(), space)
        opt.add_objective('gain_dB', minimize=False)
        opt.add_objective('power_uW', minimize=True)
        assert 'gain_dB' in opt._objectives
        assert opt._objectives['gain_dB']['minimize'] is False
        assert opt._objectives['power_uW']['minimize'] is True

    def test_constraints_stored(self, space):
        opt = Optimizer(_success_test(), space)
        opt.add_constraint('gain_dB', min=40.0)
        opt.add_constraint('power_uW', max=500.0)
        assert opt._constraints['gain_dB']['min'] == 40.0
        assert opt._constraints['power_uW']['max'] == 500.0

    def test_chaining(self, space):
        opt = (Optimizer(_success_test(), space)
               .add_objective('gain_dB', minimize=False)
               .add_constraint('power_uW', max=500.0))
        assert 'gain_dB' in opt._objectives
        assert 'power_uW' in opt._constraints


# ---------------------------------------------------------------------------
# evaluate() — success path
# ---------------------------------------------------------------------------

class TestEvaluateSuccess:

    def test_returns_dict_with_required_keys(self, space):
        opt = Optimizer(_success_test(), space)
        rec = opt.evaluate({'w': 2.0, 'l': 0.3})
        for key in ('id', 'params', 'corner', 'status', 'metrics',
                    'feasible', 'error', 'error_type', 'timestamp'):
            assert key in rec, f"Missing key '{key}'"

    def test_success_status(self, space):
        opt = Optimizer(_success_test(), space)
        rec = opt.evaluate({'w': 2.0, 'l': 0.3})
        assert rec['status'] == 'success'

    def test_metrics_populated(self, space):
        opt = Optimizer(_success_test(gain=48.0, power=250.0), space)
        rec = opt.evaluate({'w': 2.0, 'l': 0.3})
        assert rec['metrics']['gain_dB'] == 48.0
        assert rec['metrics']['power_uW'] == 250.0

    def test_error_fields_none_on_success(self, space):
        opt = Optimizer(_success_test(), space)
        rec = opt.evaluate({'w': 2.0, 'l': 0.3})
        assert rec['error'] is None
        assert rec['error_type'] is None

    def test_id_increments(self, space):
        opt = Optimizer(_success_test(), space)
        r0 = opt.evaluate({'w': 1.0, 'l': 0.15})
        r1 = opt.evaluate({'w': 2.0, 'l': 0.3})
        assert r0['id'] == 0
        assert r1['id'] == 1

    def test_params_copied(self, space):
        opt = Optimizer(_success_test(), space)
        pt = {'w': 3.0, 'l': 0.5}
        rec = opt.evaluate(pt)
        assert rec['params'] == pt
        # Mutating the original should not affect the record
        pt['w'] = 999.0
        assert rec['params']['w'] == 3.0

    def test_added_to_history(self, space):
        opt = Optimizer(_success_test(), space)
        opt.evaluate({'w': 2.0, 'l': 0.3})
        assert len(opt.history) == 1

    def test_timestamp_is_recent(self, space):
        opt = Optimizer(_success_test(), space)
        before = time.time()
        rec = opt.evaluate({'w': 2.0, 'l': 0.3})
        after = time.time()
        assert before <= rec['timestamp'] <= after

    def test_corner_forwarded(self, space):
        received = {}
        class MockTest:
            def run(self, corner='tt'):
                received['corner'] = corner
                return {'metrics': {'gain_dB': 45.0}}
        opt = Optimizer(MockTest(), space)
        opt.evaluate({'w': 2.0, 'l': 0.3}, corner='ss')
        assert received['corner'] == 'ss'


# ---------------------------------------------------------------------------
# evaluate() — feasibility
# ---------------------------------------------------------------------------

class TestEvaluateFeasibility:

    def test_feasible_when_no_constraints(self, space):
        opt = Optimizer(_success_test(), space)
        rec = opt.evaluate({'w': 2.0, 'l': 0.3})
        assert rec['feasible'] is True

    def test_feasible_true_when_constraint_met(self, space):
        opt = Optimizer(_success_test(gain=48.0, power=250.0), space)
        opt.add_constraint('gain_dB', min=40.0)
        opt.add_constraint('power_uW', max=500.0)
        rec = opt.evaluate({'w': 2.0, 'l': 0.3})
        assert rec['feasible'] is True

    def test_feasible_false_when_constraint_violated(self, space):
        opt = Optimizer(_success_test(gain=35.0, power=250.0), space)
        opt.add_constraint('gain_dB', min=40.0)
        rec = opt.evaluate({'w': 2.0, 'l': 0.3})
        assert rec['feasible'] is False

    def test_feasible_false_when_max_exceeded(self, space):
        opt = Optimizer(_success_test(power=800.0), space)
        opt.add_constraint('power_uW', max=500.0)
        rec = opt.evaluate({'w': 2.0, 'l': 0.3})
        assert rec['feasible'] is False

    def test_feasible_none_on_failure(self, space):
        opt = Optimizer(
            _make_mock_test(side_effect=SimulationError('ngspice failed')),
            space
        )
        rec = opt.evaluate({'w': 2.0, 'l': 0.3})
        assert rec['feasible'] is None


# ---------------------------------------------------------------------------
# evaluate() — failure paths
# ---------------------------------------------------------------------------

class TestEvaluateFailures:

    def test_simulation_failure_status(self, space):
        opt = Optimizer(
            _make_mock_test(side_effect=SimulationError('ngspice exit 1')),
            space
        )
        rec = opt.evaluate({'w': 2.0, 'l': 0.3})
        assert rec['status'] == 'simulation_failure'
        assert rec['error_type'] == 'SimulationError'
        assert 'ngspice exit 1' in rec['error']

    def test_netlist_error_status(self, space):
        opt = Optimizer(
            _make_mock_test(side_effect=CircuitError('bad net')),
            space
        )
        rec = opt.evaluate({'w': 2.0, 'l': 0.3})
        assert rec['status'] == 'netlist_error'
        assert rec['error_type'] == 'CircuitError'

    def test_parameter_error_unknown_variable(self, space):
        opt = Optimizer(_success_test(), space)
        rec = opt.evaluate({'unknown_var': 1.0})
        assert rec['status'] == 'parameter_error'
        assert rec['error_type'] in ('KeyError', 'ValueError')

    def test_metric_error_status(self, space):
        """ModuleTestbench stores 'Error: ...' strings for failed extractors."""
        bad_metrics = {'metrics': {'gain_dB': 'Error: signal not found', 'power_uW': 200.0}}
        opt = Optimizer(_make_mock_test(bad_metrics), space)
        rec = opt.evaluate({'w': 2.0, 'l': 0.3})
        assert rec['status'] == 'metric_error'
        assert rec['error_type'] == 'MetricExtractionError'
        assert 'gain_dB' in rec['error']

    def test_unexpected_error_status(self, space):
        opt = Optimizer(
            _make_mock_test(side_effect=RuntimeError('unexpected')),
            space
        )
        rec = opt.evaluate({'w': 2.0, 'l': 0.3})
        assert rec['status'] == 'unexpected_error'
        assert rec['error_type'] == 'RuntimeError'

    def test_failure_still_logged_to_history(self, space):
        opt = Optimizer(
            _make_mock_test(side_effect=SimulationError('fail')),
            space
        )
        opt.evaluate({'w': 2.0, 'l': 0.3})
        assert len(opt.history) == 1
        assert opt.history[0]['status'] == 'simulation_failure'

    def test_different_failures_have_distinct_status_tags(self, space):
        """Different error types produce different status values."""
        statuses = set()
        for exc in [SimulationError('sim'), CircuitError('netlist'),
                    RuntimeError('other')]:
            opt = Optimizer(_make_mock_test(side_effect=exc), space)
            rec = opt.evaluate({'w': 2.0, 'l': 0.3})
            statuses.add(rec['status'])
        assert len(statuses) == 3, f"Expected 3 distinct statuses, got: {statuses}"


# ---------------------------------------------------------------------------
# Real SpecReport results and the legacy dictionary contract
# ---------------------------------------------------------------------------

class TestReportResults:

    @pytest.mark.parametrize(
        ('gain', 'minimum', 'spec_target', 'feasible'),
        [(45.0, None, 50.0, True), (45.0, 40.0, 50.0, True),
         (35.0, 40.0, 30.0, False)],
    )
    def test_report_matches_legacy_dict(
        self, space, gain, minimum, spec_target, feasible
    ):
        report = SpecReport('ac', 'amp', {'gain_dB': gain})
        report.check('design gain', 'gain_dB', '>', spec_target)
        assert report.passed == (gain > spec_target)
        records = []
        for result in (report, {'metrics': dict(report.metrics)}):
            opt = Optimizer(_make_mock_test(result), space)
            opt.add_objective('gain_dB', maximize=True)
            if minimum is not None:
                opt.add_constraint('gain_dB', min=minimum)
            rec = opt.evaluate({'w': 2.0, 'l': 0.3})
            assert rec['status'] == 'success'
            assert rec['metrics'] == {'gain_dB': gain}
            assert rec['feasible'] is feasible
            assert opt.history == [rec]
            assert opt.successes == [rec]
            assert opt.failures == []
            assert opt.best is (rec if feasible else None)
            records.append({k: v for k, v in rec.items() if k != 'timestamp'})
        assert records[0] == records[1]

    def test_applies_and_runs_once_without_mutating_report(self, monkeypatch):
        from silk.db.context import build_characterization_context

        width = Variable('w', value=1.0, bounds=(0.5, 20.0))
        space = VariableSpace([width])
        apply = Mock(wraps=space.apply)
        monkeypatch.setattr(space, 'apply', apply)
        context = build_characterization_context(
            owner_path='$/amp', test_name='ac', parameters={'load': 1e-12},
            corner='ss', temperature=85.0, voltage=1.62,
        )
        report = SpecReport(
            'ac', 'amp', {'gain_dB': 45.0}, owner_path='$/amp',
            parameters={'load': 1e-12}, characterization_context=context,
            corner='ss',
        )
        report.check('gain', 'gain_dB', '>', 40.0,
                     spec_key='gain', owner_path='$/amp')
        report.storage = {'metrics_stored': 1, 'cached': False}
        original_state = deepcopy(vars(report))
        original_metrics = report.metrics

        def run(*, corner):
            assert width.value == 2.0
            return report

        test = Mock(run=Mock(side_effect=run))
        opt = Optimizer(test, space)
        rec = opt.evaluate({'w': 2.0}, corner='ss')

        assert rec['status'] == 'success'
        assert rec['corner'] == 'ss'
        apply.assert_called_once_with({'w': 2.0})
        test.run.assert_called_once_with(corner='ss')
        assert opt.history == [rec]
        assert vars(report) == original_state
        assert report.metrics is original_metrics
        assert report.characterization_context is context

    @pytest.mark.parametrize('report_result', [True, False])
    def test_extractor_error_is_logged_once(self, space, report_result):
        metrics = {'gain_dB': 'Error: signal not found', 'power_uW': 200.0}
        result = (SpecReport('ac', 'amp', metrics) if report_result
                  else {'metrics': metrics})
        test = Mock(run=Mock(return_value=result))
        opt = Optimizer(test, space)
        opt.add_objective('power_uW', minimize=True)
        rec = opt.evaluate({'w': 2.0, 'l': 0.3}, corner='ff')

        assert rec['status'] == 'metric_error'
        assert rec['metrics'] == metrics
        assert rec['feasible'] is None
        assert rec['error_type'] == 'MetricExtractionError'
        assert rec['error'] == 'gain_dB: Error: signal not found'
        assert opt.history == opt.failures == [rec]
        assert opt.best is None
        test.run.assert_called_once_with(corner='ff')

    @pytest.mark.parametrize('result', [None, object(), [], 'unsupported', {}])
    def test_unsupported_result_is_logged_as_unexpected_error(self, space, result):
        test = Mock(run=Mock(return_value=result))
        opt = Optimizer(test, space)
        rec = opt.evaluate({'w': 2.0, 'l': 0.3})

        assert rec['status'] == 'unexpected_error'
        assert rec['metrics'] is None
        assert rec['feasible'] is None
        assert rec['error_type'] in ('TypeError', 'KeyError')
        assert rec['error']
        assert opt.history == opt.failures == [rec]
        assert opt.best is None
        test.run.assert_called_once_with(corner='tt')


# ---------------------------------------------------------------------------
# sweep()
# ---------------------------------------------------------------------------

class TestSweep:

    def test_sweep_returns_n_records(self, space):
        opt = Optimizer(_success_test(), space)
        records = opt.sweep(n=5, method='random')
        assert len(records) == 5

    def test_sweep_appends_to_history(self, space):
        opt = Optimizer(_success_test(), space)
        opt.sweep(n=4, method='random')
        assert len(opt.history) == 4

    def test_sweep_continues_through_failures(self, space):
        """All 5 points are evaluated even when all fail."""
        opt = Optimizer(
            _make_mock_test(side_effect=SimulationError('fail')),
            space
        )
        records = opt.sweep(n=5, method='random')
        assert len(records) == 5
        assert all(r['status'] == 'simulation_failure' for r in records)

    def test_sweep_max_failures_stops_early(self, space):
        """sweep stops after max_failures failures."""
        opt = Optimizer(
            _make_mock_test(side_effect=SimulationError('fail')),
            space
        )
        records = opt.sweep(n=10, method='random', max_failures=3)
        assert len(records) == 3
        assert all(r['status'] == 'simulation_failure' for r in records)

    def test_sweep_max_failures_none_runs_all(self, space):
        """max_failures=None (default) runs all points regardless of failures."""
        opt = Optimizer(
            _make_mock_test(side_effect=SimulationError('fail')),
            space
        )
        records = opt.sweep(n=8, method='random', max_failures=None)
        assert len(records) == 8

    def test_sweep_mixed_results_counted_correctly(self, space):
        """max_failures only counts failures, not successes."""
        call_count = {'n': 0}
        class AlternatingTest:
            def run(self, corner='tt'):
                call_count['n'] += 1
                if call_count['n'] % 2 == 0:
                    raise SimulationError('fail')
                return {'metrics': {'gain_dB': 45.0}}

        opt = Optimizer(AlternatingTest(), space)
        # max_failures=2: should stop after 2 failures (at calls 2 and 4)
        records = opt.sweep(n=10, method='random', max_failures=2)
        failures = [r for r in records if r['status'] != 'success']
        assert len(failures) == 2

    def test_sweep_returns_subset_of_history(self, space):
        """Records returned by sweep are in history."""
        opt = Optimizer(_success_test(), space)
        opt.evaluate({'w': 1.0, 'l': 0.15})  # pre-existing
        sweep_records = opt.sweep(n=3, method='random')
        assert len(opt.history) == 4
        for r in sweep_records:
            assert r in opt.history


# ---------------------------------------------------------------------------
# History accessors and summary
# ---------------------------------------------------------------------------

class TestHistoryAccessors:

    def _populated_opt(self, space):
        call_count = {'n': 0}
        class MixedTest:
            def run(self, corner='tt'):
                call_count['n'] += 1
                if call_count['n'] == 2:
                    raise SimulationError('fail')
                return {'metrics': {'gain_dB': 45.0, 'power_uW': 200.0}}
        opt = Optimizer(MixedTest(), space)
        opt.add_constraint('gain_dB', min=40.0)
        opt.sweep(n=4, method='random')
        return opt

    def test_successes_property(self, space):
        opt = self._populated_opt(space)
        assert all(r['status'] == 'success' for r in opt.successes)

    def test_failures_property(self, space):
        opt = self._populated_opt(space)
        assert all(r['status'] != 'success' for r in opt.failures)

    def test_successes_and_failures_partition_history(self, space):
        opt = self._populated_opt(space)
        assert len(opt.successes) + len(opt.failures) == len(opt.history)

    def test_summary_keys(self, space):
        opt = self._populated_opt(space)
        s = opt.summary()
        assert 'total_evaluations' in s
        assert 'by_status' in s
        assert 'feasible_count' in s
        assert 'best' in s

    def test_summary_total_correct(self, space):
        opt = self._populated_opt(space)
        assert opt.summary()['total_evaluations'] == 4

    def test_summary_by_status_counts(self, space):
        opt = self._populated_opt(space)
        s = opt.summary()
        assert s['by_status'].get('success', 0) == 3
        assert s['by_status'].get('simulation_failure', 0) == 1


# ---------------------------------------------------------------------------
# best property
# ---------------------------------------------------------------------------

class TestBestProperty:

    def test_best_none_with_no_objectives(self, space):
        opt = Optimizer(_success_test(), space)
        opt.sweep(n=3, method='random')
        assert opt.best is None

    def test_best_none_with_no_successes(self, space):
        opt = Optimizer(
            _make_mock_test(side_effect=SimulationError('fail')), space
        )
        opt.add_objective('gain_dB', minimize=False)
        opt.sweep(n=3, method='random')
        assert opt.best is None

    def test_best_maximize_selects_highest(self, space):
        results = iter([
            {'metrics': {'gain_dB': 40.0}},
            {'metrics': {'gain_dB': 55.0}},
            {'metrics': {'gain_dB': 48.0}},
        ])
        class SeqTest:
            def run(self, corner='tt'):
                return next(results)
        opt = Optimizer(SeqTest(), space)
        opt.add_objective('gain_dB', minimize=False)
        opt.sweep(n=3, method='random')
        assert opt.best['metrics']['gain_dB'] == 55.0

    def test_best_minimize_selects_lowest(self, space):
        results = iter([
            {'metrics': {'power_uW': 300.0}},
            {'metrics': {'power_uW': 150.0}},
            {'metrics': {'power_uW': 250.0}},
        ])
        class SeqTest:
            def run(self, corner='tt'):
                return next(results)
        opt = Optimizer(SeqTest(), space)
        opt.add_objective('power_uW', minimize=True)
        opt.sweep(n=3, method='random')
        assert opt.best['metrics']['power_uW'] == 150.0

    def test_best_only_from_feasible(self, space):
        """best ignores infeasible successes."""
        results = iter([
            {'metrics': {'gain_dB': 60.0, 'power_uW': 800.0}},  # infeasible
            {'metrics': {'gain_dB': 48.0, 'power_uW': 200.0}},  # feasible
        ])
        class SeqTest:
            def run(self, corner='tt'):
                return next(results)
        opt = Optimizer(SeqTest(), space)
        opt.add_objective('gain_dB', minimize=False)
        opt.add_constraint('power_uW', max=500.0)
        opt.sweep(n=2, method='random')
        assert opt.best['metrics']['gain_dB'] == 48.0


# ---------------------------------------------------------------------------
# summary()['objectives'], history and VariableSpace.variables()
# ---------------------------------------------------------------------------

class TestObservation:

    def test_summary_objectives_in_declaration_order(self, space):
        opt = Optimizer(_success_test(), space)
        opt.add_objective('power_uW', minimize=True)
        opt.add_objective('gain_dB', minimize=False)
        assert opt.summary()['objectives'] == [
            {'name': 'power_uW', 'goal': 'minimize', 'value_source': 'metric'},
            {'name': 'gain_dB', 'goal': 'maximize', 'value_source': 'metric'},
        ]

    def test_summary_objectives_empty_by_default(self, space):
        opt = Optimizer(_success_test(), space)
        assert opt.summary()['objectives'] == []

    def test_history_records_have_core_keys(self, space):
        opt = Optimizer(_success_test(), space)
        opt.sweep(n=2, method='random')
        assert len(opt.history) == 2
        for rec in opt.history:
            for k in ('id', 'params', 'corner', 'status', 'metrics', 'feasible'):
                assert k in rec

    def test_space_variables_expose_bounds(self, space):
        got = space.variables()
        assert list(got) == space.names
        assert got['w'].bounds == (0.5, 20.0)

    def test_space_variables_is_new_dict_with_same_objects(self, space):
        got = space.variables()
        got.clear()
        again = space.variables()
        assert list(again) == space.names
        assert all(again[n] is space._vars[n] for n in space.names)

    def test_space_variables_accepted_by_add_objective_and_constraint(self, space):
        opt = Optimizer(_success_test(), space)
        var = space.variables()['w']
        opt.add_objective(var).add_constraint(var, max=10.0)
        assert opt.summary()['objectives'][0]['value_source'] == 'variable'

    def test_space_key_may_differ_from_variable_name(self):
        from silk.circuit.variable import Variable
        var = Variable('real_name', bounds=(1, 2))
        sp = VariableSpace({'key': var})
        assert sp.variables() == {'key': var}
        assert sp.variables()['key'] is var

    def test_successes_exclude_failures(self, space):
        call_count = {'n': 0}
        class MixedTest:
            def run(self, corner='tt'):
                call_count['n'] += 1
                if call_count['n'] == 2:
                    raise SimulationError('fail')
                return {'metrics': {'gain_dB': 45.0}}
        opt = Optimizer(MixedTest(), space)
        opt.add_objective('gain_dB', minimize=False)
        opt.sweep(n=4, method='random')
        assert len(opt.successes) == 3  # 4 total - 1 failure


# ---------------------------------------------------------------------------
# Public export
# ---------------------------------------------------------------------------

class TestPublicExport:

    def test_importable_from_package(self):
        from silk.optimize import Optimizer as CO
        assert CO is Optimizer

    def test_importable_from_core(self):
        from silk.optimize.optimizer import Optimizer as CO
        assert CO is Optimizer
