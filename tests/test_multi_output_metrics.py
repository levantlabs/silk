"""Declared result outputs share one computation and remain spec-able."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from silk import Module
from silk.circuit.ports import PortDir
from silk.elements import elements
from silk.signals.base import Signal
from silk.signals import schema as S
from silk.simulation import Transient
from tests._env import needs_ngspice


@dataclass
class Pair:
    height: float
    width: float


@pytest.fixture
def registry_guard():
    before = S._REGISTRY.copy()
    yield
    S._REGISTRY.clear()
    S._REGISTRY.update(before)


@pytest.fixture
def bench(pdk, tmp_path):
    dut = Module('sky130', str(tmp_path), 'multi.cir', pdk,
                 module_name='multi')
    dut.define_module_ports({'out': PortDir.OUT})
    dut.add(elements.resistor('r', p='out', n='0', value=1000))
    tb = dut.create_test('multi')
    tb.add_module_under_test({'out': 'out'})
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    raw = Path(__file__).parent / 'fixtures/rawfiles/e_ng_tran.raw'
    tb.netlist.load_file(str(raw))
    tb.netlist.read_raw_file()
    return tb


@pytest.mark.parametrize('declarative', [False, True])
def test_result_outputs_are_measured_once(bench, registry_guard, declarative):
    calls = []

    @S.metric(domain='tran', unit='', outputs={'height': 'V', 'width': 's'})
    def pair(sig: S.Transient, *, scale: float = 1.0) -> Pair:
        calls.append(sig)
        return Pair(float(np.max(sig.y)) * scale, 2.0)

    if declarative:
        bench.add_metric('eye', type='pair', signal='out', scale=2.0)
    else:
        bench.add_metric('eye', pair, sig=bench.probe('out'), scale=2.0)
    bench.add_spec('wide', 'eye.width', '>', 1.0)
    report = bench.evaluate()
    assert len(calls) == 1
    assert report.metrics['eye.height'] == report.metrics['eye'].height
    assert report.metrics['eye.width'] == 2.0
    assert report.passed
    assert 'pair' not in S.export_metric_schema()


def test_failure_fans_out_to_all_outputs(bench, registry_guard):
    @S.metric(domain='tran', unit='', outputs={'height': 'V', 'width': 's'})
    def broken(sig: S.Transient) -> Pair:
        raise ValueError('broken measurement')

    bench.add_metric('eye', broken, sig=bench.probe('out'))
    report = bench.evaluate()
    assert report.metrics == dict.fromkeys(
        ['eye', 'eye.height', 'eye.width'], 'Error: broken measurement')


@pytest.mark.parametrize('outputs', [{}, {'missing': 's'}, {'height': 2},
                                     {'bad.name': 's'}])
def test_invalid_output_declaration_rejected(outputs, registry_guard):
    def pair(sig: S.Transient) -> Pair:
        return Pair(1.0, 2.0)

    with pytest.raises((ValueError, TypeError)):
        S.metric(domain='tran', unit='', outputs=outputs)(pair)
    assert 'pair' not in S._REGISTRY


def test_outputs_do_not_admit_bool_returns(registry_guard):
    def truth(sig: S.Transient) -> bool:
        return True

    with pytest.raises(S.NonScalarMetricError):
        S.metric(domain='tran', unit='', outputs={'real': ''})(truth)


def test_bool_and_complex_points_schema(registry_guard):
    # Kept as resolved annotations: ComplexPoints is a public schema alias.
    def inputs(sig, *, ideal_points, reference=None, auto_center=False):
        return 1.0

    inputs.__annotations__ = dict(sig=S.Transient, ideal_points=S.ComplexPoints,
                                  reference=S.ComplexPoints | None,
                                  auto_center=bool, return_=float)
    inputs.__annotations__['return'] = inputs.__annotations__.pop('return_')
    S.metric(domain='tran', unit='%')(inputs)
    params = {p['name']: p for p in S.metric_schema(inputs)['parameters']}
    assert params['ideal_points']['type'] == 'complex_points'
    assert params['ideal_points']['required'] is True
    assert params['reference']['type'] == 'complex_points'
    assert params['reference']['optional'] is True
    assert params['reference']['default'] is None
    assert params['auto_center']['type'] == 'bool'
    assert params['auto_center']['default'] is False
    assert S._normalise_param_type(int)[0] == 'int'
    assert S._normalise_param_type(type('bool', (), {})) is None


WRAPPERS = {'sfdr_db', 'eye_height', 'eye_width', 'inl_max_lsb', 'dnl_max_lsb',
            'missing_code_count', 'harmonic_level_dbc',
            'non_monotonic_steps', 'offset_error_lsb', 'gain_error_lsb',
            'full_scale_error_lsb'}


def wave(y, x=None, xlabel='time'):
    y = np.asarray(y, dtype=float)
    return Signal('test', 'voltage', xlabel, 'voltage',
                  np.arange(len(y), dtype=float) if x is None else x, y)


def test_eight_wrappers_exported():
    schema = S.export_metric_schema()
    assert WRAPPERS <= schema.keys()
    assert schema['harmonic_level_dbc']['unit'] == 'dBc'
    assert schema['eye_height']['parameters'][-1]['default'] is True
    assert not {'sfdr', 'eye_diagram', 'inl', 'dnl', 'evm_constellation'} & schema.keys()


@pytest.mark.parametrize('wrapper,parent,field,params', [
    ('sfdr_db', 'sfdr', 'sfdr_db', {'tclk': 1.0, 'n_samples': 64}),
    ('eye_height', 'eye_diagram', 'height', {'period': 4.0, 'auto_center': True, 'threshold': 0.}),
    ('eye_width', 'eye_diagram', 'width', {'period': 4.0, 'auto_center': True, 'threshold': 0.}),
    ('inl_max_lsb', 'inl', 'max_abs_lsb', {}),
    ('dnl_max_lsb', 'dnl', 'max_abs_lsb', {}),
])
def test_wrapper_matches_expanded_result(bench, wrapper, parent, field, params):
    import silk.metrics as M
    sig = (wave([0, 1, 2.2, 3], xlabel='code') if parent in ('inl', 'dnl')
           else wave(np.sin(2*np.pi*np.arange(128)/8)))
    bench.add_metric('result', getattr(M, parent), sig=sig, **params)
    bench.add_metric('scalar', getattr(M, wrapper), sig=sig, **params)
    report = bench.evaluate()
    assert isinstance(report.metrics['scalar'], float)
    assert report.metrics['scalar'] == pytest.approx(report.metrics[f'result.{field}'])


def test_evm_reference_reaches_dotted_result(bench):
    import silk.metrics as M
    i, q = wave(np.ones(20)), wave(np.zeros(20))
    params = dict(symbol_period=2.0, ideal_points=[-1+0j, 1+0j],
                  reference=[-1+0j]*4, n_symbols=4)
    bench.add_metric('result', M.evm_constellation, i_sig=i, q_sig=q, **params)
    report = bench.evaluate()
    assert report.metrics['result.evm_percent'] == 200.0
    params.pop('reference')
    assert M.evm_constellation(i, q, **params).evm_percent == 0.0


def test_evm_wrapper_matches_expanded_result(bench):
    # TASK-369 replaced the deferral: evm_percent takes a named modulation.
    import silk.metrics as M
    sym = M.modulation_grid('qpsk')[np.arange(32) % 4] * 0.5j * 1.05
    t = np.arange(sym.size * 4) / 4.0
    i, q = wave(np.repeat(sym.real, 4), t), wave(np.repeat(sym.imag, 4), t)
    params = dict(symbol_period=1.0, modulation='qpsk')
    bench.add_metric('result', M.evm_constellation, i_sig=i, q_sig=q, **params)
    bench.add_metric('scalar', M.evm_percent, i_sig=i, q_sig=q, **params)
    report = bench.evaluate()
    assert isinstance(report.metrics['scalar'], float)
    assert report.metrics['scalar'] == report.metrics['result.evm_percent']
    assert report.metrics['scalar'] == pytest.approx(0.0, abs=1e-9)


def test_array_reductions_and_monotonic_count():
    import silk.metrics as M
    sig = wave([0, 1, 1, 0.5, 4], xlabel='code')
    assert M.missing_code_count(sig) == float(len(M.missing_codes(sig))) == 2.0
    assert M.non_monotonic_steps(sig) == 2.0
    assert not M.is_monotonic(sig)
    assert M.non_monotonic_steps(wave([0, 1, 2], xlabel='code')) == 0.0
    sig = wave(np.sin(2*np.pi*np.arange(128)/16) + .1*np.sin(4*np.pi*np.arange(128)/16))
    assert M.harmonic_level_dbc(sig, n=2, tclk=1.0, n_samples=128) == pytest.approx(-20.0)
    assert M.harmonic_level_dbc(sig, n=3, tclk=1.0, n_samples=128) == M.harmonic_levels_db(
        sig, tclk=1.0, n_samples=128, n_harmonics=3)[1]


@pytest.mark.parametrize('wrapper', ['inl_max_lsb', 'dnl_max_lsb',
                                    'missing_code_count', 'non_monotonic_steps'])
def test_nan_codes_never_become_a_passing_count(wrapper):
    import silk.metrics as M
    assert np.isnan(getattr(M, wrapper)(wave([0, np.nan, 2], xlabel='code')))


@pytest.mark.parametrize('n', [1, 0, -2, 2.5, True])
def test_harmonic_index_is_an_integer_at_least_two(n):
    import silk.metrics as M
    with pytest.raises(ValueError, match='n.*integer.*2'):
        M.harmonic_level_dbc(wave([0, 1, 0, -1]), tclk=1.0, n=n)


def test_eye_wrappers_honor_explicitly_disabled_centering():
    import silk.metrics as M
    sig = wave(np.sin(2*np.pi*np.arange(128)/8))
    assert np.isnan(M.eye_height(sig, period=8.0, threshold=0., auto_center=False))
    assert np.isnan(M.eye_width(sig, period=8.0, threshold=0., auto_center=False))


def test_preexisting_schema_entries_are_byte_identical():
    import hashlib
    import json
    from silk._json_codec import encode_json

    schema = S.export_metric_schema()
    # TASK-323.3 added final_value and made settling_time.target optional,
    # so the 61 pre-TASK-233 entries were re-pinned deliberately then;
    # TASK-364 replaced x_when with crossing and re-pinned them again;
    # TASK-365 added rise_time/fall_time/slew_rate edge params, re-pinned.
    # TASK-363 widened propagation_delay and removed comparator_prop_delay
    # (61 -> 60 pre-TASK-233 entries) and re-pinned them again.
    # TASK-366 replaced period_zero_cross with average_period (and added
    # average_frequency, dropped here) and re-pinned them again.
    # TASK-226.20 added crossing_jitter_rms; dropped here like the others.
    # TASK-367 replaced duty_cycle with average_duty and widened
    # pulse_width, on_time and off_time; re-pinned again.
    # TASK-385 added error_norm mean_abs + start/end; re-pinned. TASK-378 added interpolate on the windowed primitives; re-pinned.
    # TASK-377 renamed minimum/maximum and start/end on the windowed primitives; re-pinned.
    # TASK-386 added relative_error_norm; dropped here like the others.
    # TASK-396 added fundamental_hz/guard_bins/window/full_scale to the
    # spectral ratio metrics; re-pinned.
    # TASK-380 added refine to argmin_x/argmax_x; re-pinned.
    # TASK-379 added variance; dropped here like the others.
    # Hashed with sort_keys since stack13 (order-independent); re-pinned.
    # TASK-413 added interpolate to gain_at_freq/phase_at_freq/noise_density_at_freq; re-pinned.
    # TASK-412 removed input_referred_offset, kickback and hold_droop (60 -> 57); re-pinned.
    # TASK-418 added long_term_jitter units/reference/phase/threshold/
    # edge/start/end; re-pinned.
    # TASK-417 widened period_jitter / cycle_to_cycle_jitter (start, end,
    # reference, cycles; threshold unit [y]); re-pinned. Their _pp forms
    # are dropped here like the others.
    # TASK-414 added noise_figure_at_freq and band_noise_figure; dropped here like the others.
    for name in [*WRAPPERS, 'final_value', 'evm_percent', 'average_frequency',
                 'crossing_jitter_rms', 'relative_error_norm', 'variance',
                 'oip3_dbm', 'iip3_dbm', 'period_jitter_pp',
                 'cycle_to_cycle_jitter_pp', 'noise_figure_at_freq',
                 'band_noise_figure']:
        del schema[name]
    schema['_meta']['metric_count'] = 57
    # sort_keys: the pin is on content, not on registration order, which
    # shifts with import order (found by TASK-403: a lazy import moved it).
    original = json.dumps(encode_json(schema), allow_nan=False, indent=2,
                          sort_keys=True)
    assert hashlib.sha256(original.encode()).hexdigest() == (
        '727738524a28ef263782b22daaa484e03b58beb2e7685b11c67b7d59959552be')


@pytest.mark.parametrize('value', [True, [1.0], 1+2j])
def test_malformed_result_fails_every_output(bench, registry_guard, value):
    @S.metric(domain='tran', outputs={'height': 'V', 'width': 's'})
    def bad_pair(sig: S.Transient) -> Pair:
        return Pair(1.0, value)

    bench.add_metric('bad', bad_pair, sig=bench.probe('out'))
    report = bench.evaluate()
    assert report.metrics['bad'].startswith('Error: ')
    assert report.metrics['bad.height'] == report.metrics['bad.width'] == report.metrics['bad']


@pytest.mark.parametrize('first', [True, False])
def test_dotted_name_collision_raises_before_extraction(bench, registry_guard, first):
    @S.metric(domain='tran', outputs={'height': 'V', 'width': 's'})
    def pair(sig: S.Transient) -> Pair:
        raise AssertionError('must not extract on a name collision')

    if first:
        bench.add_metric('eye.width', lambda ckt: 1.0)
    bench.add_metric('eye', pair, sig=bench.probe('out'))
    if not first:
        bench.add_metric('eye.width', lambda ckt: 1.0)
    with pytest.raises(ValueError, match='eye.width.*collision'):
        bench.evaluate()


@pytest.mark.slow
@needs_ngspice
def test_real_ngspice_eye_specs_and_database_roundtrip(pdk, tmp_path):
    from silk import sources
    from silk.metrics import eye_diagram
    dut = Module('sky130', str(tmp_path), 'eye.cir', pdk,
                 module_name='eye_load', db_path=str(tmp_path / 'eye.db'))
    dut.define_module_ports({'out': PortDir.OUT})
    dut.add(elements.resistor('rload', p='out', n='0', value=1000))
    tb = dut.create_test('eye')
    tb.add_module_under_test({'out': 'out'})
    tb.add_source(sources.VoltageSource(
        'stim', positive='out', negative='0', dc=0.0,
        pulse=sources.Pulse(v1=0, v2=1, td=0, tr=1e-9, tf=1e-9,
                            pw=99e-9, per=200e-9)))
    tb.add_sim_type(Transient(step=1e-9, stop=4e-6))
    tb.add_metric('eye', eye_diagram, sig=tb.probe('out'), period=100e-9,
                  auto_center=True, threshold=.5)
    tb.add_spec('eye height', 'eye.height', '>', .4)
    tb.add_spec('eye width', 'eye.width', '>', 80e-9)
    try:
        dut.create_circuit_version()
        report = tb.run()
        assert report.passed
        assert report.metrics['eye.height'] > .4
        assert report.metrics['eye.width'] > 80e-9
        receipt = dut.store_report(report)
        assert receipt['metrics_stored'] == 3
        assert receipt['specs_stored'] == 2
        assert 'eye' in receipt['skipped']
        from silk.exceptions import EvidenceNotFoundError
        with pytest.raises(EvidenceNotFoundError):
            dut.query_results(parameter='eye')
        for name in ('eye.height', 'eye.width', 'eye.sampling_instant_s'):
            records = dut.query_results(parameter=name)
            assert len(records) == 1
            assert records[0].value == report.metrics[name]
    finally:
        dut.close()
