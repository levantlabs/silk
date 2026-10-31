"""``add_metric(name, fn, **slots_and_params)`` — the function form (TASK-215.6).

Alongside the existing
callable (``extractor_func(ckt)``) and declarative (``type=``/``signal=``/
``signals=``) forms, ``add_metric`` accepts an ``@metric``-registered
function directly, with its Signal-input slots bound **by keyword name**:

    tb.add_metric('vcm_avg', average, sig=vcm)
    tb.add_metric('p_avg', average_power, sig_v=vout, sig_i=iout)

``fn`` is recognised as a registered metric only through the ``@metric``
decoration marker (``fn._silk_metric``) — never by name, so an undecorated
function called ``'average'`` is treated as today's plain
``extractor(ckt)`` form. Four refusals fire at declaration, never at
simulation time: a positional signal after ``fn``; an undecorated ``fn``
given keywords its signature does not take (TASK-323 made an undecorated
``fn`` with keywords the plain-function form, covered in
``test_add_metric_plain_function.py``); a decorated ``fn`` missing a required slot; a decorated
``fn`` with no signal-input slots at all. Slot values are stored in the
metric dict (``'fn'``, ``'slots'``, ``'params'``) rather than closed over, so
they are inspectable.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tests._env import needs_ngspice

from silk import sources
from silk.circuit.circuit_elements import PortDir
from silk.circuit.module_test import _is_function_form_metric
from silk.circuit.modules import Module
from silk.elements import elements
from silk.metrics.statistics import average
from silk.metrics.power import average_power
from silk.signals.base import Signal
from silk.signals.schema import export_metric_schema, metric
from silk.simulation import analysis

_SKY130_LIB = os.path.expanduser(
    '~/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice')
_have_pdk = os.path.exists(_SKY130_LIB)


# ---------------------------------------------------------------------------
# Fast fixtures — a real testbench, never simulated
# ---------------------------------------------------------------------------

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


@pytest.fixture
def no_slot_metric():
    """A ``@metric``-decorated function with zero signal-input slots.

    ``@metric`` registers into the module-level catalog (``_REGISTRY``) as a
    side effect, which other tests count (``test_metric_schema.py``,
    ``test_metric_scalar_return_guard.py``); this fixture pops its own entry
    back out on teardown so a throwaway test metric never leaks into the
    real catalog.
    """
    from silk.signals import schema as _schema_module

    @metric(domain='dc', unit='')
    def _add_metric_function_form_no_slot_metric() -> float:
        return 1.0

    yield _add_metric_function_form_no_slot_metric
    _schema_module._REGISTRY.pop('_add_metric_function_form_no_slot_metric', None)


# ---------------------------------------------------------------------------
# The four refusals
# ---------------------------------------------------------------------------

class TestFourRefusals:

    def test_positional_signal_after_fn_names_the_slot(self, tb):
        vout = tb.probe('vout')
        with pytest.raises(TypeError, match=r'\bsig\b'):
            tb.add_metric('m', average, vout)

    def test_positional_signal_names_all_slots_of_a_multi_slot_metric(self, tb):
        vout = tb.probe('vout')
        ivs = tb.probe('i(vs)')
        with pytest.raises(TypeError, match=r'sig_v.*sig_i|sig_i.*sig_v'):
            tb.add_metric('m', average_power, vout, ivs)

    def test_undecorated_extractor_signature_given_keyword_slots_is_refused(self, tb):
        # TASK-323: an undecorated fn given keywords is the plain-function
        # form, bound against ITS OWN signature -- an extractor(ckt) has no
        # 'sig' parameter, so the keyword is refused by name.
        def average(ckt):  # shadows the registered one; deliberately undecorated
            return 0.0

        vout = tb.probe('vout')
        with pytest.raises(TypeError, match=r"unknown keyword argument 'sig'"):
            tb.add_metric('m', average, sig=vout)

    def test_undecorated_fn_with_positional_extra_is_refused(self, tb):
        def scale(ckt, factor):
            return factor

        with pytest.raises(TypeError, match='keyword name, not position'):
            tb.add_metric('m', scale, 2.0)

    def test_decorated_fn_missing_required_slot_names_it(self, tb):
        with pytest.raises(TypeError, match=r"\bsig\b"):
            tb.add_metric('m', average)

    def test_decorated_multi_slot_fn_missing_one_slot_names_it(self, tb):
        vout = tb.probe('vout')
        with pytest.raises(TypeError, match=r'sig_i'):
            tb.add_metric('m', average_power, sig_v=vout)

    def test_decorated_fn_with_no_slots_at_all_names_the_slots(self, tb, no_slot_metric):
        with pytest.raises(TypeError, match='no signal-input slots'):
            tb.add_metric('m', no_slot_metric)

    def test_decorated_fn_with_no_slots_refused_even_with_no_kwargs(self, tb, no_slot_metric):
        """The no-slots refusal is unconditional -- not only on a mismatch."""
        with pytest.raises(TypeError):
            tb.add_metric('m', no_slot_metric)


# ---------------------------------------------------------------------------
# The unchanged callable form: an undecorated function is an extractor
# ---------------------------------------------------------------------------

class TestUndecoratedIsExtractor:

    def test_undecorated_function_named_average_is_treated_as_extractor(self, tb):
        def average(ckt):  # not the registered metric -- no @metric marker
            return 42.0

        tb.add_metric('m', average)
        entry = tb._metrics[-1]
        assert entry['name'] == 'm'
        assert entry['extractor'] is average
        # No function-form bookkeeping for the plain callable form.
        assert 'fn' not in entry
        assert 'slots' not in entry
        assert 'params' not in entry

    def test_undecorated_extractor_runs_unchanged(self, tb):
        def constant(ckt):
            return 7.0

        tb.add_metric('m', constant)
        entry = tb._metrics[-1]
        assert entry['extractor'](None) == 7.0


# ---------------------------------------------------------------------------
# Slot values stored and inspectable
# ---------------------------------------------------------------------------

class TestSlotsInspectable:

    def test_single_slot_stored(self, tb):
        vout = tb.probe('vout')
        tb.add_metric('vavg', average, sig=vout)
        entry = tb._metrics[-1]
        assert entry['fn'] is average
        assert entry['slots'] == {'sig': vout}
        assert entry['params'] == {}

    def test_multi_slot_stored_by_name(self, tb):
        vout = tb.probe('vout')
        ivs = tb.probe('i(vs)')
        tb.add_metric('p', average_power, sig_v=vout, sig_i=ivs)
        entry = tb._metrics[-1]
        assert entry['slots'] == {'sig_v': vout, 'sig_i': ivs}

    def test_scalar_params_forwarded_and_stored_separately_from_slots(self, tb):
        vout = tb.probe('vout')
        tb.add_metric('vavg', average, sig=vout, start=1e-9, end=5e-9)
        entry = tb._metrics[-1]
        assert entry['slots'] == {'sig': vout}
        assert entry['params'] == {'start': 1e-9, 'end': 5e-9}

    def test_string_slot_value_is_a_signal_path_like_the_declarative_form(self, tb):
        tb.add_metric('vavg', average, sig='vout')
        entry = tb._metrics[-1]
        assert entry['slots'] == {'sig': 'vout'}

    def test_unknown_keyword_refused_naming_both_slots_and_params(self, tb):
        vout = tb.probe('vout')
        with pytest.raises(TypeError, match='sig'):
            tb.add_metric('vavg', average, sig=vout, bogus=1)

    def test_unknown_keyword_message_names_available_params(self, tb):
        vout = tb.probe('vout')
        with pytest.raises(TypeError) as excinfo:
            tb.add_metric('vavg', average, sig=vout, bogus=1)
        msg = str(excinfo.value)
        assert 'bogus' in msg
        assert 'start' in msg or 'end' in msg


# ---------------------------------------------------------------------------
# Declarative (type=) and callable forms stay byte-identical
# ---------------------------------------------------------------------------

class TestOtherFormsUnchanged:

    def test_declarative_form_still_works(self, tb):
        vout = tb.probe('vout')
        tb.add_metric('vavg', type='average', signal='vout')
        entry = tb._metrics[-1]
        assert entry['name'] == 'vavg'
        assert 'extractor' in entry
        # No FUNCTION-form bookkeeping leaks into the declarative dict: 'fn'
        # and 'params' are what _is_function_form_metric discriminates on,
        # and the copy/pickle hooks act on nothing else.
        assert 'fn' not in entry
        assert 'params' not in entry
        assert not _is_function_form_metric(entry)
        # TASK-212's own bookkeeping IS here — the keys the pre-extraction
        # analysis check reads, with nothing selected for a one-analysis test.
        assert entry['type'] == 'average'
        assert entry['analysis'] is None
        assert entry['signals'] == ['vout']
        assert set(entry.keys()) == {
            'name', 'extractor', 'type', 'slots', 'analysis', 'signals',
            'where'}

    def test_type_and_extractor_both_given_refused(self, tb):
        with pytest.raises(TypeError, match='exactly one'):
            tb.add_metric('m', average, type='average')

    def test_neither_type_nor_extractor_refused(self, tb):
        with pytest.raises(TypeError, match='exactly one'):
            tb.add_metric('m')


def test_export_metric_schema_is_stable_and_unaffected():
    """add_metric's function form touches no schema machinery: silk/signals/
    schema.py is untouched by TASK-215.6 (verified separately against main
    by the implementer with ``git diff --stat``), so exporting is stable
    across repeated calls in one process."""
    schema_now = json.dumps(export_metric_schema(), sort_keys=True)
    schema_again = json.dumps(export_metric_schema(), sort_keys=True)
    assert schema_now == schema_again


# ---------------------------------------------------------------------------
# Real ngspice: the function form matches the lambda form to 1e-12
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not _have_pdk, reason='sky130 ngspice models not installed')
class TestAgainstRealNgspice:
    """One transient run on a 1 kOhm / 1 nF RC low-pass driven by a vpulse.

    ``add_metric(name, average, sig=probe_or_expr)`` must return the same
    float as the lambda form, to 1e-12; a multi-slot metric
    (``average_power``) binds two probes by name.
    """

    @pytest.fixture(scope='class')
    def ran(self, tmp_path_factory):
        from silk.pdk.basepdk import BasePDK
        pdk_yaml = os.path.join(
            os.path.dirname(__file__), '..', 'silk', 'pdk', 'data',
            'skywater_130_device_details.yaml')
        pdk = BasePDK(pdk_yaml, library_location=_SKY130_LIB)
        tmp_dir = str(tmp_path_factory.mktemp('rc_add_metric'))
        dut = Module('PDK_skywater130', tmp_dir, 'rc.cir', pdk_details=pdk,
                     module_name='rc')
        dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT})
        with dut:
            elements.resistor('r1', p='a', n='b', value=1e3)
            elements.capacitor('c1', p='b', n='0', value=1e-9)
        test = dut.create_test('rc')
        test.add_module_under_test({'a': 'vin', 'b': 'vout'})
        test.add_source(sources.VoltageSource(
            'vs', positive='vin', negative='0', dc=0.0,
            pulse=sources.Pulse(v1=0.0, v2=1.0, td=0.0, tr=1e-9, tf=1e-9,
                                pw=5e-6, per=10e-6)))
        test.add_sim_type(analysis.Transient(step=1e-8, stop=1e-5))

        vout = test.probe('vout')
        vin = test.probe('vin')
        iout = test.probe('i(vs)')
        vsum = vin + vout

        test.add_metric('vavg', average, sig=vout)
        test.add_metric('vsum', average, sig=vsum)
        test.add_metric('p_avg', average_power, sig_v=vout, sig_i=iout)
        test.add_metric('vavg_lambda',
                        lambda ckt: average(ckt.load_signal('vout')))
        report = test.run()
        return test, report

    def test_function_form_matches_lambda_form(self, ran):
        _, report = ran
        assert report.metrics['vavg'] == pytest.approx(
            report.metrics['vavg_lambda'], abs=1e-12)

    def test_function_form_matches_eager_computation_directly(self, ran):
        test, report = ran
        expected = average(test.netlist.load_signal('vout'))
        assert report.metrics['vavg'] == pytest.approx(expected, abs=1e-12)

    def test_function_form_on_a_probe_expression_matches_eager(self, ran):
        test, report = ran
        eager_vin = test.netlist.load_signal('vin')
        eager_vout = test.netlist.load_signal('vout')
        expected = average(eager_vin + eager_vout)
        assert report.metrics['vsum'] == pytest.approx(expected, abs=1e-12)

    def test_multi_slot_metric_binds_by_name(self, ran):
        test, report = ran
        expected = average_power(
            test.netlist.load_signal('vout'),
            test.netlist.load_signal('vvs__silk_wave', 'I'),
        )
        assert report.metrics['p_avg'] == pytest.approx(expected, abs=1e-12)


@needs_ngspice
def test_raw_ac_refusal_is_per_metric_and_projected_probe_is_measured(tmp_path):
    """A complex scalar survives the report while its spec fails visibly."""
    from silk import load_pdk
    dut = Module('PDK_skywater130', str(tmp_path), 'divider.cir',
                 pdk_details=load_pdk('sky130'), module_name='divider')
    dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT})
    with dut:
        elements.resistor('r1', p='a', n='b', value=1e3)
        elements.resistor('r2', p='b', n='0', value=1e3)
    tb = dut.create_test('ac')
    tb.add_module_under_test({'a': 'vin', 'b': 'vout'})
    tb.add_source(sources.VoltageSource('vs', positive='vin', negative='0', ac=1))
    tb.add_sim_type(analysis.AC('dec', 3, 1, 100))
    vout, vin = tb.probe('vout'), tb.probe('vin')
    tb._cache_enabled = True
    raw = tb.add_metric('raw', average, sig=vout)
    magnitude = tb.add_metric('magnitude', abs(raw))
    tb.add_metric('projected', average, sig=(vout / vin).real())
    refused_spec = tb.add_spec('raw limit', 'raw', '<', 1.0)
    valid_spec = tb.add_spec('projected limit', 'projected', '<', 1.0)
    report = tb.run()
    measured = report.metrics['raw']
    assert type(measured) is complex
    assert measured == pytest.approx(0.5+0j)
    assert raw.value == measured
    assert '.real' in report.summary() and 'abs' in report.summary()
    assert report.metrics['projected'] == pytest.approx(0.5)
    assert magnitude.value == pytest.approx(0.5)
    assert tb.run() is report
    assert raw.value == measured
    assert not refused_spec.measurements
    assert len(valid_spec.measurements) == 1
    assert next(iter(valid_spec.measurements.values())).value == pytest.approx(0.5)
