"""``add_metric(name, fn, **probes_and_params)`` with a PLAIN function (TASK-323).

A designer's one-off metric takes probes without ``@metric`` and without
registration; registration stays for catalog metrics. Reviewer ruling
2026-09-28 on the proposal in the task notes:

* slot names come from ``fn``'s signature, checked at ``add_metric``;
* a keyword whose VALUE is a ``Signal`` is a slot, every other value a scalar
  -- a ``str`` included, never a signal path in this form;
* a slot has a kind only if ``fn`` annotates it ``Annotated[Signal, KIND]``;
  a kind of ``None`` narrows nothing, so an unpinned probe in a
  multi-analysis test is refused (op+tran device-param probes included);
* nothing reaches the metric registry or ``export_metric_schema()``;
* a lambda / nested-def metric pickles with an error naming the metric.
"""
from __future__ import annotations

import copy
import os
import pickle
from typing import Annotated

import numpy as np
import pytest

from silk import sources
from silk.circuit.circuit_elements import PortDir
from silk.circuit.module_test import (
    _function_form_slot_analyses, _is_function_form_metric,
)
from silk.circuit.modules import Module
from silk.exceptions import MetricBindingError
from silk.signals.base import Signal
from silk.signals.kinds import FREQUENCY_RESPONSE, SignalKindError
from silk.signals.schema import export_metric_schema, get_metric
from silk.simulation import analysis

from tests._env import needs_ngspice, needs_pdk
from tests._env import scratch_dir


# Module-level so they pickle by reference.
def peak(sig, scale=1.0):
    return float(np.max(sig.y)) * scale


def first_ratio(num, den):
    return float(num.y[0] / den.y[0])


def last_value(sig):
    return float(sig.y[-1])


def ac_peak(sig: Annotated[Signal, FREQUENCY_RESPONSE]):
    return float(np.max(np.abs(sig.y)))


def _bound(name, y, x=None, xlabel='time'):
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float) if x is None else x
    return Signal(name, 'V', xlabel, 'V', x, y)


@pytest.fixture
def tb(pdk, tmp_dir):
    """A wired testbench, unsimulated: enough for probe() and add_metric()."""
    dut = Module('PDK_skywater130', tmp_dir, 'dut.cir', pdk_details=pdk,
                 module_name='dut')
    dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT,
                             'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
    with dut:
        pdk.nmos1p8('m1', d='b', g='a', s='vss', b='vss', w=1.0, l=0.15)
    test = dut.create_test('t')
    test.add_module_under_test({'a': 'vin', 'b': 'vout', 'vdd': 'vdd',
                                'vss': '0'})
    test.add_source(sources.VoltageSource('vs', positive='vin', negative='0',
                                          dc=0.9))
    test.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0',
                                          dc=1.8))
    test.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
    return test


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------

class TestBinding:

    def test_signal_valued_keyword_is_a_slot_other_values_are_params(self, tb):
        vout = tb.probe('vout')
        tb.add_metric('pk', peak, sig=vout, scale=2.0)
        entry = tb._metrics[-1]
        assert _is_function_form_metric(entry)
        assert entry['fn'] is peak
        assert entry['slots'] == {'sig': vout}
        assert entry['params'] == {'scale': 2.0}
        assert entry['slot_kinds'] == {'sig': None}

    def test_default_param_may_be_omitted(self, tb):
        tb.add_metric('pk', peak, sig=tb.probe('vout'))
        assert tb._metrics[-1]['params'] == {}

    def test_lambda_binds(self, tb):
        tb.add_metric('r', lambda a, b: 0.0,
                      a=tb.probe('vout'), b=tb.probe('vin'))
        assert set(tb._metrics[-1]['slots']) == {'a', 'b'}

    def test_derived_expression_is_a_slot(self, tb):
        diff = tb.probe('vout') - tb.probe('vin')
        tb.add_metric('pk', peak, sig=diff)
        assert tb._metrics[-1]['slots'] == {'sig': diff}

    def test_str_is_always_a_scalar(self, tb):
        def crossing(sig, edge):
            return 0.0

        tb.add_metric('c', crossing, sig=tb.probe('vout'), edge='rising')
        entry = tb._metrics[-1]
        assert entry['params'] == {'edge': 'rising'}
        assert set(entry['slots']) == {'sig'}

    def test_annotated_kind_is_recorded(self, tb):
        tb.add_metric('a', ac_peak, sig=tb.probe('vout'))
        assert tb._metrics[-1]['slot_kinds'] == {'sig': 'frequency_response'}

    def test_var_keyword_function_accepts_any_name(self, tb):
        def anything(**sigs):
            return 0.0

        tb.add_metric('any', anything, x=tb.probe('vout'), k=3)
        entry = tb._metrics[-1]
        assert set(entry['slots']) == {'x'}
        assert entry['params'] == {'k': 3}

    def test_no_keywords_is_still_the_extractor_form(self, tb):
        def extractor(ckt):
            return 1.0

        tb.add_metric('e', extractor)
        entry = tb._metrics[-1]
        assert entry['extractor'] is extractor
        assert 'fn' not in entry


# ---------------------------------------------------------------------------
# Refusals at declaration
# ---------------------------------------------------------------------------

class TestRefusals:

    def test_unknown_keyword_names_the_parameters(self, tb):
        with pytest.raises(TypeError) as exc:
            tb.add_metric('pk', peak, sgi=tb.probe('vout'))
        msg = str(exc.value)
        assert "add_metric('pk', peak)" in msg
        assert "unknown keyword argument 'sgi'" in msg
        assert "'sig', 'scale'" in msg

    def test_missing_required_parameter_is_named(self, tb):
        with pytest.raises(TypeError, match=r"missing parameter 'den'"):
            tb.add_metric('r', first_ratio, num=tb.probe('vout'))

    def test_no_signal_bound_is_refused(self, tb):
        def scaled(sig, k):
            return 0.0

        with pytest.raises(TypeError, match='none of the keyword arguments is a Signal'):
            tb.add_metric('s', scaled, sig='vout', k=2)

    def test_str_for_a_signal_annotated_parameter_is_refused(self, tb):
        with pytest.raises(TypeError, match=r"sig=tb.probe\('vout'\)"):
            tb.add_metric('a', ac_peak, sig='vout')

    def test_positional_extra_is_refused(self, tb):
        with pytest.raises(TypeError, match='keyword name, not position'):
            tb.add_metric('pk', peak, tb.probe('vout'))

    def test_probe_of_another_bench_is_refused(self, tb, pdk, tmp_dir):
        other = Module('PDK_skywater130', tmp_dir, 'o.cir', pdk_details=pdk,
                       module_name='other').create_test('o')
        other.add_source(sources.VoltageSource('vx', positive='x',
                                               negative='0', dc=1.0))
        with pytest.raises(TypeError, match='belongs to bench'):
            tb.add_metric('pk', peak, sig=other.probe('x'))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestEvaluation:

    def test_bound_signals_and_params_reach_the_function(self, tb):
        tb.add_metric('pk', peak, sig=_bound('s', [1.0, 3.0, 2.0]), scale=2.0)
        assert tb._metrics[-1]['extractor'](None) == 6.0

    def test_kind_error_inside_the_body_is_a_binding_error(self, tb):
        def strict(sig):
            raise SignalKindError('wanted an AC trace')

        tb.add_metric('k', strict, sig=_bound('s', [1.0]))
        with pytest.raises(MetricBindingError, match='wrong kind'):
            tb._metrics[-1]['extractor'](None)


# ---------------------------------------------------------------------------
# Choosing the analysis without schema metadata
# ---------------------------------------------------------------------------

class TestAnalysis:

    def test_single_analysis_needs_nothing(self, tb):
        tb.add_metric('pk', peak, sig=tb.probe('vout'))
        tb._require_unambiguous_analyses()          # no raise
        assert _function_form_slot_analyses(tb._metrics[-1], ['tran']) == {'sig': None}

    def test_unpinned_probe_in_tran_ac_is_refused(self, tb):
        tb.add_sim_type(analysis.AC('dec', 5, 1, 1e6))
        tb.add_metric('pk', peak, sig=tb.probe('vout'))
        with pytest.raises(MetricBindingError) as exc:
            tb._require_unambiguous_analyses()
        msg = str(exc.value)
        assert "'ac', 'tran'" in msg
        assert "tb.probe(..., analysis='ac')" in msg

    def test_pinned_probe_in_tran_ac_is_accepted(self, tb):
        tb.add_sim_type(analysis.AC('dec', 5, 1, 1e6))
        tb.add_metric('pk', peak, sig=tb.probe('vout', analysis='tran'))
        tb._require_unambiguous_analyses()
        assert _function_form_slot_analyses(
            tb._metrics[-1], ['tran', 'ac']) == {'sig': 'tran'}

    def test_annotated_kind_settles_the_analysis(self, tb):
        tb.add_sim_type(analysis.AC('dec', 5, 1, 1e6))
        tb.add_metric('a', ac_peak, sig=tb.probe('vout'))
        assert _function_form_slot_analyses(
            tb._metrics[-1], ['tran', 'ac']) == {'sig': 'ac'}

    def test_device_param_probe_unpinned_in_op_tran_is_refused(self, tb):
        tb.add_sim_type(analysis.Op())
        tb.add_metric('g', last_value, sig=tb.probe('m1:gm'))
        with pytest.raises(MetricBindingError, match="'op', 'tran'"):
            tb._require_unambiguous_analyses()

    def test_device_param_probe_pinned_op_in_op_tran_is_accepted(self, tb):
        tb.add_sim_type(analysis.Op())
        tb.add_metric('g', last_value, sig=tb.probe('m1:gm', analysis='op'))
        tb._require_unambiguous_analyses()
        assert _function_form_slot_analyses(
            tb._metrics[-1], ['tran', 'op']) == {'sig': 'op'}


# ---------------------------------------------------------------------------
# Schema untouched
# ---------------------------------------------------------------------------

class TestSchemaUntouched:

    def test_export_is_identical_and_nothing_is_registered(self, tb):
        before = export_metric_schema()

        def my_one_off_metric_323(sig):
            return 0.0

        tb.add_metric('m', my_one_off_metric_323, sig=tb.probe('vout'))
        tb.add_metric('p', peak, sig=tb.probe('vout'))
        assert export_metric_schema() == before
        assert not hasattr(my_one_off_metric_323, '_silk_metric')
        assert not hasattr(peak, '_silk_metric')
        with pytest.raises(KeyError):
            get_metric('my_one_off_metric_323')


# ---------------------------------------------------------------------------
# Copy and pickle
# ---------------------------------------------------------------------------

class TestCopyAndPickle:

    def test_deepcopy_rebinds_the_probe(self, tb):
        tb.add_metric('pk', peak, sig=tb.probe('vout'), scale=3.0)
        new = copy.deepcopy(tb)
        entry = new._metrics[-1]
        assert entry['fn'] is peak
        assert entry['params'] == {'scale': 3.0}
        assert entry['slot_kinds'] == {'sig': None}
        assert entry['slots']['sig']._bench_or_none() is new

    def test_module_level_function_round_trips(self, tb):
        tb.add_metric('pk', peak, sig=_bound('s', [1.0, 4.0]), scale=0.5)
        restored = pickle.loads(pickle.dumps(tb))
        entry = restored._metrics[-1]
        assert entry['fn'] is peak
        assert entry['extractor'](None) == 2.0
        # The original bench's dict is untouched by pickling.
        assert tb._metrics[-1]['fn'] is peak

    def test_lambda_metric_pickle_error_names_the_metric(self, tb):
        tb.add_metric('duty_up', lambda sig: 0.0, sig=tb.probe('vout'))
        with pytest.raises(pickle.PicklingError) as exc:
            pickle.dumps(tb)
        msg = str(exc.value)
        assert "metric 'duty_up'" in msg
        assert 'module level' in msg

    def test_nested_def_metric_pickle_error_names_the_metric(self, tb):
        def nested(sig):
            return 0.0

        tb.add_metric('n', nested, sig=tb.probe('vout'))
        with pytest.raises(pickle.PicklingError, match="metric 'n'"):
            pickle.dumps(tb.module)

    def test_copy_copy_keeps_the_function_not_the_stand_in(self, tb):
        fn = lambda sig: 0.0  # noqa: E731
        tb.add_metric('l', fn, sig=tb.probe('vout'))
        shallow = copy.copy(tb)
        assert shallow._metrics[-1]['fn'] is fn
        assert tb._metrics[-1]['fn'] is fn


# ---------------------------------------------------------------------------
# Real ngspice: op-only bench, net probe + derived expression + device param
# ---------------------------------------------------------------------------

@pytest.mark.slow
@needs_pdk
@needs_ngspice
class TestRealNgspice:

    def test_op_bench_plain_metrics(self):
        from silk.pdk import load_pdk

        pdk = load_pdk('sky130')
        dut = Module('sky130', scratch_dir(), 'dut.cir', pdk_details=pdk,
                     module_name='dut')
        dut.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN,
                                 's': PortDir.GROUND, 'b': PortDir.GROUND})
        with dut:
            pdk.nmos1p8('M1', d='d', g='g', s='s', b='b', w=1.0, l=0.15)
        t = dut.create_test('plain')
        t.add_module_under_test({'d': 'vd', 'g': 'vg', 's': '0', 'b': '0'})
        t.add_source(sources.VoltageSource('vd', positive='vd', negative='0',
                                           dc=1.0))
        t.add_source(sources.VoltageSource('vg', positive='vg', negative='0',
                                           dc=0.9))
        t.add_sim_type(analysis.Op())

        vg = t.probe('vg')                           # net probe
        gm, gds = t.probe('M1:gm'), t.probe('M1:gds')
        t.add_metric('vg_x2', peak, sig=vg, scale=2.0)
        t.add_metric('gm', last_value, sig=gm)       # device-param probe
        t.add_metric('intrinsic_gain', last_value, sig=gm / gds)  # derived
        t.add_metric('gm_over_gds', first_ratio, num=gm, den=gds)
        t.add_spec('gain > 1', 'intrinsic_gain', '>', 1.0)

        report = t.run('tt')
        m = report.metrics
        assert m['vg_x2'] == pytest.approx(1.8, rel=1e-9)
        assert m['gm'] > 0.0
        assert m['gm'] == pytest.approx(float(gm.y[0]), rel=1e-12)
        assert m['intrinsic_gain'] == pytest.approx(
            float(gm.y[0]) / float(gds.y[0]), rel=1e-12)
        assert m['gm_over_gds'] == pytest.approx(m['intrinsic_gain'], rel=1e-12)
        assert m['intrinsic_gain'] > 1.0
        assert report.results[0].passed
