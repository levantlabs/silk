"""MetricWarnings raised while an output's extractor runs are attributed to
that output and kept on the SpecReport (TASK-337).

Uses the same fast pattern as ``tests/circuit/test_testbench_probe_rebinding.py``:
the committed TASK-74 rawfile fixture stands in for the simulator (``Module.
simulate`` is stubbed), so ``run()`` exercises its real ``evaluate()`` tail
against real Signal objects sourced from a real rawfile. The fixture's
``v(out)`` vector is flattened to a constant in place before extraction --
still the real rawfile-reading/Signal-construction pipeline, just with a
genuinely degenerate (zero-swing) voltage, which is what
``silk.metrics.timing.rise_time`` needs to raise its ``zero_swing``
``MetricWarning``.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

from silk import Module, sources
from silk.circuit.ports import PortDir
from silk.elements import elements
from silk.exceptions import AmbiguousBranchCurrentError, MetricBindingError
from silk.metrics.diagnostics import MetricWarning
from silk.metrics.timing import rise_time
from silk.simulation import Transient
from silk.signals import schema as S
from tests._env import needs_pdk
from tests._temperature import mark_simulated
from tests.test_multi_output_metrics import Pair


FIXTURE = (Path(__file__).resolve().parents[1]
           / 'fixtures' / 'rawfiles' / 'e_ng_tran.raw')


@pytest.mark.parametrize('decoder', ['as_int', 'codes_from_bits'])
def test_binary_precision_warning_captured_in_extractor(pdk, tmp_path, decoder):
    import numpy as np
    from silk.signals import Signal, codes_from_bits
    x = np.array([0., 1.])
    words = [1 << 53, (1 << 53) + 1]
    bits = tuple(Signal(f'd[{i}]', 'voltage', 'time', 'V', x,
                        np.array([(word >> i) & 1 for word in words], dtype=float))
                 for i in range(54))
    bus = Signal._make_bus_handle('d', bits)
    tb = _bench(pdk, tmp_path, 'binary_capture', declarative=False, function_form=False)

    def extract(ckt):
        decoded = (bus.as_int() if decoder == 'as_int'
                   else codes_from_bits(bus, msb_first=False))
        return decoded.y[-1]

    tb.add_metric('word', extract)
    with _stub(tb), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', MetricWarning)
        report = tb.run(corner='tt')
    assert report.metrics['word'] == float(words[-1])
    record, = report.metric_warnings['word']
    assert (record.metric, record.code, record.output) == (
        'as_int', 'binary_code_precision_loss', 'word')
    assert record.details['rounded_sample_count'] == 1
    assert len(caught) == 1


def _flatten_out(netlist):
    """Force v(out) to a constant (zero swing) on the loaded real rawfile."""
    for vec in netlist._raw.native.current_plot.vectors:
        if vec.name == 'v(out)':
            vec.data[:] = 1.8


def _load_fixture(netlist, corner='tt'):
    netlist.load_file(str(FIXTURE))
    netlist.read_raw_file()
    netlist.results_file = str(FIXTURE)
    mark_simulated(netlist, corner, voltage=1.8, supply_source='vsup')
    _flatten_out(netlist)
    return netlist


def _stub(bench):
    def fake_simulate(corner=None, **_):
        return _load_fixture(bench.netlist, corner)
    return patch.object(bench.netlist, 'simulate', side_effect=fake_simulate)


def _bench(pdk, tmp_path, name, *, declarative=True, function_form=True,
          extra_metric=None):
    """A one-resistor bench with rise_time bound both ways on net 'out'."""
    block = Module('sky130', str(tmp_path), f'{name}.cir', pdk,
                   module_name=name)
    block.define_module_ports({'out': PortDir.OUT, 'vss': PortDir.GROUND})
    block.add(elements.resistor('rload', p='out', n='vss', value=1000))
    tb = block.create_test(name)
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_source(sources.VoltageSource('vsup', positive='out', negative='0',
                                        dc=1.8))
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    if declarative:
        tb.add_metric('rt_declarative', type='rise_time', signal='out')
    if function_form:
        tb.add_metric('rt_function', rise_time, sig=tb.probe('out'))
    if extra_metric is not None:
        extra_metric(tb)
    return tb


@needs_pdk
class TestAttributionBothForms:
    """AC#2: declarative and @metric forms, two outputs sharing one type."""

    def test_declarative_and_function_form_each_attributed(self, pdk, tmp_path):
        tb = _bench(pdk, tmp_path, 'both')
        with _stub(tb):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                report = tb.run(corner='tt')

        import math
        assert math.isnan(report.metrics['rt_declarative'])
        assert math.isnan(report.metrics['rt_function'])

        assert set(report.metric_warnings) == {'rt_declarative', 'rt_function'}
        for name in ('rt_declarative', 'rt_function'):
            warnings_for = report.metric_warnings[name]
            assert len(warnings_for) == 1
            record = warnings_for[0]
            assert record.metric == 'rise_time'
            assert record.code == 'zero_swing'
            assert record.output == name

        # Exactly one MetricWarning surfaces per occurrence (AC#5): two
        # occurrences here (one per output), not four (original + re-emit).
        assert len(caught) == 2
        outputs_seen = {w.message.output for w in caught}
        assert outputs_seen == {'rt_declarative', 'rt_function'}
        for w in caught:
            assert isinstance(w.message, MetricWarning)
            assert f"output {w.message.output!r} (rise_time):" in str(w.message)

    def test_default_outer_filter_still_records_exactly_one_per_output(
        self, pdk, tmp_path,
    ):
        """AC#5 repeated under the outer 'default' filter action."""
        tb = _bench(pdk, tmp_path, 'default_filter')
        with _stub(tb):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('default')
                report = tb.run(corner='tt')
        assert len(caught) == 2
        assert {w.message.output for w in caught} == {
            'rt_declarative', 'rt_function'
        }
        assert set(report.metric_warnings) == {'rt_declarative', 'rt_function'}


@needs_pdk
class TestConsoleWording:

    def test_console_message_is_the_reemitted_warning_naming_the_output(
        self, pdk, tmp_path, capsys,
    ):
        """The console text (the re-emitted MetricWarning's ``str()``, which
        is what Python's default ``showwarning`` prints) names the output,
        and there is no separate 'Warning: error extracting metric' print
        alongside it (ruling B: no separate print, nothing shows twice)."""
        tb = _bench(pdk, tmp_path, 'console', function_form=False)
        with _stub(tb):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                tb.run(corner='tt')
        assert len(caught) == 1
        text = str(caught[0].message)
        assert text.startswith("output 'rt_declarative' (rise_time): ")
        assert 'zero_swing' in text
        assert 'error extracting metric' not in capsys.readouterr().out


@needs_pdk
class TestFilterPolicies:

    def test_filterwarnings_error_becomes_error_string_like_today(
        self, pdk, tmp_path,
    ):
        """The metric value falls back to 'Error: ...' exactly as before
        this task, but the record is still kept on the report -- the
        re-emit that raises must not skip storing metric_warnings first."""
        tb = _bench(pdk, tmp_path, 'errpolicy', function_form=False)
        with _stub(tb):
            with warnings.catch_warnings():
                warnings.simplefilter('error', MetricWarning)
                report = tb.run(corner='tt')
        assert isinstance(report.metrics['rt_declarative'], str)
        assert report.metrics['rt_declarative'].startswith('Error: ')
        assert len(report.metric_warnings['rt_declarative']) == 1
        record = report.metric_warnings['rt_declarative'][0]
        assert record.metric == 'rise_time'
        assert record.code == 'zero_swing'
        assert record.output == 'rt_declarative'
        assert report.metric_errors['rt_declarative']['exception_type'] == 'MetricWarning'
        assert 'zero_swing' in report.metric_errors['rt_declarative']['message']

    def test_filterwarnings_ignore_stays_silent_but_report_keeps_the_record(
        self, pdk, tmp_path,
    ):
        tb = _bench(pdk, tmp_path, 'ignorepolicy', function_form=False)
        with _stub(tb):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('ignore', MetricWarning)
                report = tb.run(corner='tt')
        assert len(caught) == 0
        import math
        assert math.isnan(report.metrics['rt_declarative'])
        assert len(report.metric_warnings['rt_declarative']) == 1
        assert report.metric_warnings['rt_declarative'][0].code == 'zero_swing'

    def test_default_policy_leaves_metric_value_unchanged(self, pdk, tmp_path):
        tb = _bench(pdk, tmp_path, 'defaultpolicy', function_form=False)
        with _stub(tb):
            with warnings.catch_warnings():
                warnings.simplefilter('default')
                report = tb.run(corner='tt')
        import math
        assert math.isnan(report.metrics['rt_declarative'])
        assert len(report.metric_warnings['rt_declarative']) == 1


@needs_pdk
class TestNonMetricWarningsAndExceptions:

    def test_non_metric_warning_reaches_pytest_warns_untouched(
        self, pdk, tmp_path,
    ):
        def deprecated_extractor(ckt):
            warnings.warn('deprecated thing', DeprecationWarning, stacklevel=2)
            return 1.0

        tb = _bench(pdk, tmp_path, 'nonmetric', declarative=False,
                   function_form=False,
                   extra_metric=lambda t: t.add_metric('dep', deprecated_extractor))
        with _stub(tb):
            with pytest.warns(DeprecationWarning, match='deprecated thing'):
                report = tb.run(corner='tt')
        assert report.metrics['dep'] == 1.0
        assert report.metric_warnings == {}

    def test_metric_binding_error_still_raises_and_propagates(
        self, pdk, tmp_path,
    ):
        tb = _bench(pdk, tmp_path, 'binding', declarative=False,
                   function_form=False)

        def raises_binding_error(ckt):
            raise MetricBindingError('no such slot')

        tb.add_metric('bad', raises_binding_error)
        with _stub(tb):
            with pytest.raises(MetricBindingError):
                tb.run(corner='tt')

    def test_ordinary_exception_still_becomes_error_string(self, pdk, tmp_path):
        tb = _bench(pdk, tmp_path, 'plainerror', declarative=False,
                   function_form=False)

        def raises_value_error(ckt):
            raise ValueError('boom')

        tb.add_metric('bad', raises_value_error)
        with _stub(tb):
            report = tb.run(corner='tt')
        assert report.metrics['bad'] == 'Error: boom'
        assert report.metric_warnings == {}


@pytest.mark.parametrize('contaminated', [False, True])
def test_spectral_interpolation_warning_survives_report_and_storage(pdk, tmp_path, contaminated):
    import math
    import numpy as np
    from silk.metrics import sndr
    from silk.circuit.spec_report import SpecReport
    from tests.db.test_metric_warning_persistence import _module

    def add_metric(tb):
        tb.add_metric('spectrum', sndr, sig=tb.probe('out'),
                      tclk=1e-7, tstart=1.23456789e-8, n_samples=16)

    tb = _bench(pdk, tmp_path, 'spectral_capture', declarative=False,
                function_form=False, extra_metric=add_metric)
    with _stub(tb):
        if contaminated:
            # Keep the real loading path; contaminate its interpolation support.
            original = tb.netlist.simulate.side_effect
            def missing(corner=None, **kw):
                netlist = original(corner=corner, **kw)
                for vec in netlist._raw.native.current_plot.vectors:
                    if vec.name == 'v(out)':
                        vec.data[:] = np.nan
                return netlist
            tb.netlist.simulate.side_effect = missing
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', MetricWarning)
            report = tb.run(corner='tt')
    records = report.metric_warnings['spectrum']
    codes = [r.code for r in records]
    assert codes.count('interpolated_samples') == 1
    assert ('nan_samples' in codes) == contaminated
    assert math.isnan(report.metrics['spectrum']) == contaminated
    assert all(r.output == 'spectrum' and r.metric == 'sndr' for r in records)
    assert len(caught) == len(records)
    restored = SpecReport.from_dict(report.to_dict())
    assert [r.to_dict() for r in restored.metric_warnings['spectrum']] == [r.to_dict() for r in records]
    mod = _module(pdk, tmp_path, 'stored_spectrum')
    try:
        mod.store_report(report)
        result, = mod.query_results(parameter='spectrum')
        assert result.conditions['silk_metric_warnings'] == [r.to_dict() for r in records]
    finally:
        mod.close()


@pytest.mark.parametrize('policy', ['always', 'default', 'error', 'ignore'])
def test_warnings_survive_extractor_failure(pdk, tmp_path, policy):
    tb = _bench(pdk, tmp_path, 'failed_warning', declarative=False,
                function_form=False)

    def extract(ckt):
        warnings.warn(MetricWarning(metric='measure', code='first', message='limited',
                                    details={'count': 2}, suggestion='inspect input'))
        warnings.warn_explicit('plain evidence', UserWarning, 'extractor.py', 123)
        warnings.warn(MetricWarning(metric='measure', code='second', message='missing',
                                    details={}))
        raise ValueError('extractor failed')

    tb.add_metric('bad', extract)
    with _stub(tb), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        warnings.simplefilter(policy, MetricWarning)
        if policy == 'ignore':
            warnings.simplefilter('ignore', UserWarning)
        report = tb.run(corner='tt')
    assert report.metrics['bad'] == 'Error: extractor failed'
    assert report.metric_errors['bad']['exception_type'] == 'ValueError'
    assert report.metric_errors['bad']['message'] == 'extractor failed'
    first, second = report.metric_warnings['bad']
    assert (first.metric, first.code, first.output, first.message,
            first.details, first.suggestion) == (
                'measure', 'first', 'bad', 'limited', {'count': 2}, 'inspect input')
    assert (second.code, second.output) == ('second', 'bad')
    assert len(caught) == {'always': 3, 'default': 3, 'error': 1, 'ignore': 0}[policy]
    if policy != 'ignore':
        plain, = [w for w in caught if type(w.message) is UserWarning]
        assert (str(plain.message), plain.category, plain.filename, plain.lineno) == (
            'plain evidence', UserWarning, 'extractor.py', 123)


def test_plain_warning_before_failure_is_delivered(pdk, tmp_path):
    tb = _bench(pdk, tmp_path, 'plain_failure', declarative=False, function_form=False)

    def extract(ckt):
        warnings.warn_explicit('plain evidence', UserWarning, 'extractor.py', 42)
        raise ValueError('failed')

    tb.add_metric('bad', extract)
    with _stub(tb), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        report = tb.run(corner='tt')
    warning, = caught
    assert (warning.category, str(warning.message), warning.filename, warning.lineno) == (
        UserWarning, 'plain evidence', 'extractor.py', 42)
    assert report.metric_warnings == {}
    assert report.metric_errors['bad']['exception_type'] == 'ValueError'
    assert report.metric_errors['bad']['message'] == 'failed'


def test_plain_warning_promoted_inside_extractor_stays_error(pdk, tmp_path):
    tb = _bench(pdk, tmp_path, 'plain_promotion', declarative=False, function_form=False)

    def extract(ckt):
        warnings.warn('plain evidence', UserWarning)
        raise AssertionError('must not reach later raise')

    tb.add_metric('bad', extract)
    with _stub(tb), warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        report = tb.run(corner='tt')
    assert report.metric_errors['bad']['exception_type'] == 'UserWarning'
    assert report.metric_errors['bad']['message'] == 'plain evidence'
    assert report.metric_warnings == {}


@pytest.mark.parametrize('policy', ['always', 'error'])
@pytest.mark.parametrize('error_type', [MetricBindingError, AmbiguousBranchCurrentError, KeyboardInterrupt, SystemExit])
def test_warning_before_fatal_error_preserves_exception(pdk, tmp_path, policy, error_type):
    tb = _bench(pdk, tmp_path, 'fatal_warning', declarative=False, function_form=False)
    error = error_type('fatal extractor error')

    def extract(ckt):
        warnings.warn(MetricWarning(metric='measure', code='limited',
                                    message='evidence', details={}))
        raise error

    tb.add_metric('bad', extract)
    with _stub(tb), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(policy, MetricWarning)
        with pytest.raises(error_type) as raised:
            tb.run(corner='tt')
    assert raised.value is error
    assert len(caught) == (1 if policy == 'always' else 0)
    if caught:
        assert caught[0].message.output == 'bad'


def test_warned_result_with_invalid_field_keeps_warning(pdk, tmp_path, monkeypatch):
    tb = _bench(pdk, tmp_path, 'invalid_field', declarative=False, function_form=False)

    monkeypatch.setattr(S, '_REGISTRY', S._REGISTRY.copy())

    @S.metric(domain='tran', outputs={'height': 'V'})
    def extract(sig: S.Transient) -> Pair:
        warnings.warn(MetricWarning(metric='measure', code='limited',
                                    message='evidence', details={}))
        return object()

    tb.add_metric('bad', extract, sig=tb.probe('out'))
    with _stub(tb), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        report = tb.run(corner='tt')
    assert len(caught) == 1
    record, = report.metric_warnings['bad']
    assert (record.code, record.output) == ('limited', 'bad')
    assert set(report.metric_warnings) == {'bad'}
    assert report.metrics['bad.height'] == report.metrics['bad']
    assert report.metric_errors['bad']['exception_type'] == 'AttributeError'
    assert report.metric_errors['bad.height'] == report.metric_errors['bad']
