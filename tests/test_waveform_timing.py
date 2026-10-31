"""Timing regressions measured with ngspice 42 and Xyce 7.10 (TASK-472)."""
import shutil
import subprocess

import numpy as np
import pytest

from silk import settings
from silk.circuit.spice_netlister import PySpiceNetlister
from silk.circuit.xyce_netlister import XyceNetlister
from silk.circuit.variable import Variable
from silk.exceptions import NotSupportedError
from silk.sources import CurrentSource, VoltageSource, PWL, Pulse


POINTS = [(0, 2), (1, 4), (2, 1), (3, 3)]


def test_pwl_delay_first_and_last_holds():
    wave = PWL([(1, 2), (2, 4), (3, 1)], td='1')
    np.testing.assert_allclose(wave.to_numpy([-1, 0, 1, 2, 2.5, 3, 4, 5]),
                               [2, 2, 2, 2, 3, 4, 1, 1])


def test_pwl_repeat_selected_segment_and_endpoint():
    wave = PWL(POINTS, r=1, td=1)
    np.testing.assert_allclose(
        wave.to_numpy([0, 1, 1.5, 2, 3, 4, 4.01, 4.5, 5, 5.5, 6, 6.01]),
        [2, 2, 3, 4, 1, 3, 3.97, 2.5, 1, 2, 4, 3.97])


def test_pwl_zero_repeat_and_unchanged_plain_interpolation():
    plain = PWL(POINTS)
    t = np.array([-1, 0, .5, 1, 2, 3, 4])
    np.testing.assert_allclose(plain.to_numpy(t), [2, 2, 3, 4, 1, 3, 3])
    np.testing.assert_allclose(PWL(POINTS, r=0).to_numpy([3, 3.5, 4, 6, 6.5]),
                               [3, 3, 4, 2, 3])


@pytest.mark.parametrize('kwargs', [dict(td=-1), dict(td=float('nan')),
    dict(r=.5), dict(r=3), dict(r=-1), dict(r=float('inf'))])
def test_pwl_invalid_timing_is_not_silently_ignored(kwargs):
    with pytest.raises(ValueError, match='PWL'):
        PWL(POINTS, **kwargs).to_numpy([0, 5])


@pytest.mark.parametrize('points', [[(0, 0), (0, 1)], [(-1, 0), (1, 1)],
                                    [(0, 0), (1, float('nan'))]])
def test_pwl_malformed_points_fail_instead_of_interpolating(points):
    with pytest.raises(ValueError, match='PWL'):
        PWL(points, td=1).to_numpy([0, 5])


def pulse(count=None):
    return Pulse(2, 4, td=1, tr=.2, tf=.3, pw=.5, per=2, np=count)


def test_pulse_finite_count_exhaustion_and_edges():
    np.testing.assert_allclose(pulse(2).to_numpy(
        [0, 1, 1.1, 1.2, 1.7, 1.85, 2, 3, 3.1, 3.7, 4, 5, 5.1, 9]),
        [2, 2, 3, 4, 4, 3, 2, 2, 3, 4, 2, 2, 2, 2], atol=1e-14)


@pytest.mark.parametrize('count', [None, 0])
def test_pulse_unlimited_count(count):
    np.testing.assert_allclose(pulse(count).to_numpy([5.1, 7.5, 9.85]), [3, 4, 3])


@pytest.mark.parametrize('count', [-1, 1.5, True, float('nan'), float('inf')])
def test_pulse_invalid_count_fails(count):
    with pytest.raises(ValueError, match='Pulse np'):
        pulse(count).to_numpy([0, 9])


def test_pulse_finite_count_requires_explicit_period_for_numpy():
    wave = Pulse(0, 1, tr=.2, tf=.3, pw=.5, np=2)
    with pytest.raises(ValueError, match='per'):
        wave.to_numpy([0, 9])


def test_timing_variables_resolve_each_evaluation():
    delay = Variable('delay', value=1)
    repeat = Variable('repeat', value=1)
    count = Variable('count', value=1)
    wave = PWL(POINTS, td=delay, r=repeat)
    np.testing.assert_allclose(wave.to_numpy([1.5, 4.5]), [3, 2.5])
    delay.value = 0
    repeat.value = 0
    np.testing.assert_allclose(wave.to_numpy([1.5, 4.5]), [2.5, 2.5])
    p = pulse(count)
    assert p.to_numpy([3.5])[0] == 2
    count.value = 2
    assert p.to_numpy([3.5])[0] == 4
    count.value = -1
    with pytest.raises(ValueError, match='Pulse np'):
        p.to_numpy([3.5])
    p = pulse()
    edge = Variable('edge', value=.2)
    p.tr = edge
    edge.value = 0
    with pytest.raises(ValueError, match='Pulse tr'):
        p.to_numpy([1.1])


@pytest.mark.parametrize('netlister,cls,wave', [
    (PySpiceNetlister, CurrentSource, PWL(POINTS, td=1)),
    (PySpiceNetlister, CurrentSource, PWL(POINTS, r=1)),
    *[(XyceNetlister, cls, wave) for cls in (VoltageSource, CurrentSource)
      for wave in (PWL(POINTS, td=1), PWL(POINTS, r=1), pulse(2))],
])
@pytest.mark.parametrize('generic', [False, True])
def test_unsupported_backend_timing_rejected_before_simulation(monkeypatch, netlister, cls, wave, generic):
    from silk.simulation import _ngspice_version
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: '42')
    slot = 'waveform' if generic else ('pwl' if isinstance(wave, PWL) else 'pulse')
    source = cls('stim', positive='out', negative='0', **{slot: wave})
    with pytest.raises(NotSupportedError, match='PWL|Pulse'):
        netlister(None).generate_sources({'stim': source})


def run_raw(tmp_path, backend, source_text):
    exe = settings.get('SILK_NGSPICE_BIN' if backend == 'ngspice' else 'SILK_XYCE_BIN')
    if not exe or not shutil.which(exe):
        pytest.skip(f'{backend} executable unavailable through silk settings')
    tail = ('.control\nrun\nwrdata values v(out)\n.endc\n' if backend == 'ngspice'
            else '.print tran format=csv file=values.csv V(out)\n')
    deck = 'waveform timing\n' + source_text + '\nRload out 0 1\n.tran .002 10 0 .002\n' + tail + '.end\n'
    (tmp_path / 'deck.cir').write_text(deck)
    result = subprocess.run([exe, *(['-b'] if backend == 'ngspice' else []), 'deck.cir'],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    log = result.stdout + result.stderr
    (tmp_path / 'simulator.log').write_text(log)
    path = tmp_path / ('values' if backend == 'ngspice' else 'values.csv')
    return path, log


BACKENDS = ['ngspice', pytest.param('xyce', marks=pytest.mark.requires_xyce)]


@pytest.mark.slow
@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('cls', [VoltageSource, CurrentSource])
@pytest.mark.parametrize('case', ['plain', 'delay', 'repeat', 'repeat_only', 'finite', 'unlimited', 'zero_count'])
def test_measured_backend_support_and_parity(tmp_path, backend, cls, case):
    # Continuous repeat join permits sampling the exact join on an adaptive grid.
    wave = (PWL([(0, 2), (1, 4), (2, 1), (3, 4)],
                td=1 if case in ('delay', 'repeat') else None,
                r=1 if case in ('repeat', 'repeat_only') else None)
            if case in ('plain', 'delay', 'repeat', 'repeat_only')
            else pulse(2 if case == 'finite' else 0 if case == 'zero_count' else None))
    slot = 'pwl' if isinstance(wave, PWL) else 'pulse'
    source = cls('stim', positive='out', negative='0', **{slot: wave})
    path, log = run_raw(tmp_path, backend, source.to_spice())
    timed_pwl = case in ('delay', 'repeat', 'repeat_only')
    from silk.simulation._ngspice_version import ngspice_version
    version = ngspice_version() if backend == 'ngspice' else None
    if (backend == 'ngspice' and cls is CurrentSource and timed_pwl
            and version is not None and int(version.split('.')[0].rstrip('+')) < 47):
        assert not path.exists() and 'unknown parameter' in log
        return
    assert path.exists(), log
    data = np.loadtxt(path, delimiter=None if backend == 'ngspice' else ',',
                      skiprows=0 if backend == 'ngspice' else 1)
    samples = np.array([0, .5, 1, 1.1, 1.2, 1.5, 1.7, 1.85, 2,
                        3, 3.1, 3.7, 4, 4.01, 4.5, 5, 5.1, 5.5, 6, 6.01, 9.85])
    measured = np.interp(samples, data[:, 0], data[:, 1]) * (1 if cls is VoltageSource else -1)
    if backend == 'xyce' and case in ('delay', 'repeat'):
        # Existing delay suffix syntax is silently accepted as a zero waveform.
        np.testing.assert_allclose(measured, 0, atol=1e-10)
    elif backend == 'xyce' and case == 'finite':
        # The eighth PULSE token does not impose a pulse count.
        np.testing.assert_allclose(measured, pulse().to_numpy(samples), atol=2e-5, rtol=0)
        assert measured[list(samples).index(5.5)] == pytest.approx(4)
    else:
        np.testing.assert_allclose(measured, source.to_numpy(samples), atol=2e-5, rtol=0)


@pytest.mark.parametrize('field,bad', [('tr', float('nan')), ('tf', float('inf')),
                                      ('pw', 0), ('pw', -.1), ('td', -1),
                                      ('per', .5)])
def test_finite_pulse_refuses_ambiguous_or_nonfinite_timing(field, bad):
    wave = pulse(2)
    setattr(wave, field, bad)
    with pytest.raises(ValueError, match='Pulse'):
        wave.to_numpy([0, 5])


@pytest.mark.parametrize('cls', [VoltageSource, CurrentSource])
def test_multiple_timed_waveforms_refuse_before_backend_selection(cls):
    from silk.exceptions import CircuitError
    from silk.sources import Sine
    with pytest.raises(CircuitError, match='stim.*pwl.*sine'):
        cls('stim', positive='out', negative='0',
            sine=Sine(0, 1, 1), pwl=PWL(POINTS, td=1))


@pytest.mark.slow
def test_ngspice_discontinuous_repeat_join(tmp_path):
    from silk.circuit.modules import Module
    mod = Module(None, str(tmp_path), 'timing.cir', None)
    source = VoltageSource('stim', positive='out', negative='0',
                           pwl=PWL(POINTS, r=1, td=1))
    mod.add_source(source)
    path, log = run_raw(tmp_path, 'ngspice', source.to_spice())
    assert path.exists(), log
    data = np.loadtxt(path)
    sample = [3.99, 4, 4.01, 5.99, 6, 6.01]
    np.testing.assert_allclose(np.interp(sample, data[:, 0], data[:, 1]),
                               [2.98, 3, 3.97, 2.98, 4, 3.97], atol=2e-5, rtol=0)
    assert np.interp(1.5, data[:, 0], data[:, 1]) == pytest.approx(3, abs=2e-5)
    mod.update_source('stim', ['pwl_td'], [2])
    revised = tmp_path / 'revised'
    revised.mkdir()
    path, log = run_raw(revised, 'ngspice', source.to_spice())
    assert path.exists(), log
    data = np.loadtxt(path)
    samples = [1.5, 2.5, 3.5, 4.5, 5.5]
    np.testing.assert_allclose(np.interp(samples, data[:, 0], data[:, 1]),
                               [2, 3, 2.5, 2, 2.5], atol=2e-5, rtol=0)
    np.testing.assert_allclose(source.to_numpy(samples), [2, 3, 2.5, 2, 2.5])


@pytest.mark.parametrize('cls', [VoltageSource, CurrentSource])
@pytest.mark.parametrize('case', ['pwl', 'pulse'])
def test_registered_behavioral_bench_uses_corrected_timing(tmp_path, pdk, cls, case):
    from silk.circuit.modules import Module
    from silk.simulation import Transient
    dut = Module('sky130', dir_folder=str(tmp_path), netlistname='timing.cir', pdk_details=pdk,
                 module_name='dut')
    from silk.circuit.circuit_elements import PortDir
    dut.define_module_ports({'result': PortDir.OUT})
    def model_1(ports, t, params):
        ports.result.v = 0 * t
    dut.set_behavioral_model(model_1, execution='vector')
    tb = dut.create_test('timing')
    tb.add_module_under_test({'result': 'result'})
    wave = PWL(POINTS, td=1, r=1) if case == 'pwl' else pulse(2)
    tb.add_source(cls('stim', positive='out', negative='gnd', **{case: wave}))
    tb.add_sim_type(Transient(step=.1, stop=7))
    tb.netlist.simulator = 'behavioral'
    sig = tb.probe('out') if cls is VoltageSource else tb.probe('istim', 'I')
    tb.run()
    samples = [.5, 1.5, 3.5, 4.5, 5.5, 6.5]
    want = [2, 3, 2, 2.5, 2, 2.5] if case == 'pwl' else [2, 4, 4, 2, 2, 2]
    np.testing.assert_allclose(np.interp(samples, sig.x, sig.y), want, atol=1e-12)
    if case == 'pwl':
        tb.netlist.update_source('stim', ['pwl_td'], [2])
        tb.run()
        np.testing.assert_allclose(np.interp(samples, sig.x, sig.y),
                                   [2, 2, 2.5, 2, 2.5, 2], atol=1e-12)
        tb.netlist.update_source('stim', ['pwl_r'], [None])
        tb.run()
        np.testing.assert_allclose(np.interp(samples, sig.x, sig.y),
                                   [2, 2, 2.5, 2, 3, 3], atol=1e-12)


@pytest.mark.parametrize('wave', [pulse(2), PWL(POINTS, td=1), PWL(POINTS, r=1)])
def test_parent_backend_controls_nested_source_validation_without_stale_state(tmp_path, pdk, wave):
    from silk.circuit.modules import Module
    child = Module('sky130', str(tmp_path / 'child'), 'child.cir',
                   pdk_details=pdk, module_name='child')
    slot = 'pulse' if isinstance(wave, Pulse) else 'pwl'
    child.add_source(VoltageSource('stim', positive='out', negative='0', **{slot: wave}))
    middle = Module('sky130', str(tmp_path / 'middle'), 'middle.cir',
                    pdk_details=pdk, module_name='middle')
    child.define_module_ports({})
    middle.define_module_ports({})
    middle.add_module(child, 'child')
    modules = {'middle': {'device': middle}}
    before = PySpiceNetlister(pdk).generate_module_definitions(modules)
    with pytest.raises(NotSupportedError, match='PWL|Pulse'):
        XyceNetlister(pdk).generate_module_definitions(modules)
    # An unsuccessful Xyce emission must not poison the reusable child.
    assert PySpiceNetlister(pdk).generate_module_definitions(modules) == before


def test_finite_pulse_period_equal_to_edges_and_width_at_si_scale():
    wave = Pulse(0, 1, tr='1n', tf='1n', pw='1n', per=3e-9, np=2)
    np.testing.assert_allclose(wave.to_numpy([.5e-9, 1.5e-9, 2.5e-9, 4.5e-9, 6.5e-9]),
                               [.5, 1, .5, 1, 0], atol=1e-14)


@pytest.mark.parametrize('cls', [VoltageSource, CurrentSource])
def test_xyce_accepts_repeat_only_with_continuous_join(cls):
    wave = PWL([(0, 2), (1, 4), (2, 1), (3, 4)], r=1)
    source = cls('stim', positive='out', negative='0', pwl=wave)
    text = XyceNetlister(None).generate_sources({'stim': source})
    if cls is VoltageSource:
        assert 'DC 2.0 PWL(' in text
        assert 'Vstim__silk_dc __silk_src_Vstim 0 DC 0' in text
    else:
        assert 'Istim__silk_pwl out 0 DC 2.0 PWL(' in text
        assert 'Istim__silk_dc out 0 DC 0' in text


@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.parametrize('cls', [VoltageSource, CurrentSource])
def test_xyce_discontinuous_repeat_uses_final_value_at_later_exact_join(tmp_path, cls):
    source = cls('stim', positive='out', negative='0', pwl=PWL(POINTS, r=1))
    path, log = run_raw(tmp_path, 'xyce', source.to_spice())
    assert path.exists(), log
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    samples = [2.99, 3, 3.01, 4.99, 5, 5.01, 7]
    actual = np.interp(samples, data[:, 0], data[:, 1]) * (1 if cls is VoltageSource else -1)
    np.testing.assert_allclose(actual, [2.98, 3, 3.97, 2.98, 3, 3.97, 3], atol=2e-5, rtol=0)
    with pytest.raises(NotSupportedError, match='join'):
        XyceNetlister(None).generate_sources({'stim': source})


@pytest.mark.parametrize('version', ['42', '44.2', '45.2', '46', None, '47', '47+', '47.1', '48'])
@pytest.mark.parametrize('timing', [{'r': 0}, {'td': 0}, {'r': 0, 'td': 1e-9}])
@pytest.mark.parametrize('generic', [False, True])
def test_current_pwl_selected_ngspice_version(monkeypatch, version, timing, generic):
    from silk.simulation import _ngspice_version
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: version)
    source = CurrentSource('stim', positive='out', negative='0',
                           **{'waveform' if generic else 'pwl':
                              PWL([(0, 0), (1e-9, 1e-3)], **timing)})
    netlister = PySpiceNetlister(None)
    if version is None or int(version.split('.')[0].rstrip('+')) < 47:
        with pytest.raises(NotSupportedError) as caught:
            netlister.generate_sources({'stim': source})
        message = str(caught.value)
        assert (version or 'unknown') in message
        assert '47' in message and 'SILK_NGSPICE_BIN' in message
    else:
        assert netlister.generate_sources({'stim': source}) == (
            '*** CurrentSource: stim\n' + source.to_spice())


def test_ngspice_version_probe_cached_per_resolved_binary(monkeypatch, tmp_path):
    from silk import doctor
    from silk.simulation import _ngspice_version, runner
    first, second = (tmp_path / name for name in ('old', 'new'))
    for path in (first, second):
        path.write_text('placeholder')
        path.chmod(0o755)
    alias = tmp_path / 'alias'
    alias.symlink_to(first)
    calls = []
    def probe(argv, pattern):
        calls.append(argv)
        return 'ngspice-42' if argv[0] == str(first) else 'ngspice-47'
    monkeypatch.setattr(doctor, '_version_line', probe)
    _ngspice_version._binary_version.cache_clear()
    try:
        for path, expected in [(first, '42'), (alias, '42'), (second, '47'), (first, '42')]:
            monkeypatch.setenv('SILK_NGSPICE_BIN', str(path))
            assert runner.ngspice_executable() == str(path)
            assert _ngspice_version.ngspice_version() == expected
            assert _ngspice_version.ngspice_version() == expected
            source = CurrentSource('stim', positive='out', negative='0',
                                   pwl=PWL(POINTS, r=0))
            for _ in range(2):
                if expected == '42':
                    with pytest.raises(NotSupportedError):
                        PySpiceNetlister(None).generate_sources({'stim': source})
                else:
                    assert source.to_spice() in PySpiceNetlister(None).generate_sources({'stim': source})
        assert calls == [[str(first), '--version'], [str(second), '--version']]
    finally:
        _ngspice_version._binary_version.cache_clear()


@pytest.mark.parametrize('output,expected', [
    ('ngspice-44.2', '44.2'), ('ngspice-45.2', '45.2'),
    ('ngspice-47', '47'), ('ngspice-47+', '47+'),
    (None, None), ('unrecognized output', None), ('ngspice-47broken', None),
])
def test_ngspice_version_parsing(monkeypatch, output, expected):
    from silk import doctor
    from silk.simulation import _ngspice_version
    monkeypatch.setattr(doctor, '_version_line', lambda *args: output)
    _ngspice_version._binary_version.cache_clear()
    try:
        assert _ngspice_version._binary_version('/example/ngspice') == expected
    finally:
        _ngspice_version._binary_version.cache_clear()


def test_missing_selected_ngspice_version_is_unknown(monkeypatch, tmp_path):
    from silk.simulation import _ngspice_version
    monkeypatch.setenv('SILK_NGSPICE_BIN', str(tmp_path / 'missing'))
    assert _ngspice_version.ngspice_version() is None


def test_current_pwl_schema_retains_timing_controls():
    from silk.sources.schema import export_source_schema
    params = export_source_schema()['CurrentSource']['waveforms']['pwl']['parameters']
    assert {'r', 'td'} <= {param['name'] for param in params}


@pytest.mark.parametrize('cls,timing', [
    (VoltageSource, {'r': 0, 'td': 1e-9}), (CurrentSource, {}),
])
def test_ngspice_version_not_probed_for_unaffected_sources(monkeypatch, cls, timing):
    from silk.simulation import _ngspice_version
    def unexpected_probe():
        pytest.fail('unaffected source must not probe ngspice')
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', unexpected_probe)
    source = cls('stim', positive='out', negative='0',
                 pwl=PWL([(0, 0), (1e-9, 1e-3)], **timing))
    assert source.to_spice() in PySpiceNetlister(None).generate_sources({'stim': source})


@pytest.mark.slow
@pytest.mark.parametrize('timing', [{'td': 1e-9}, {'r': 0}])
def test_selected_ngspice_current_pwl_real(tmp_path, timing):
    from silk.simulation._ngspice_version import ngspice_version
    from silk.simulation.runner import ngspice_executable
    executable = shutil.which(ngspice_executable())
    if executable is None:
        pytest.skip('selected ngspice unavailable')
    version = ngspice_version()
    source = CurrentSource('stim', positive='0', negative='out',
                           pwl=PWL([(0, 0), (1e-9, 1e-3)], **timing))
    supported = version is not None and int(version.split('.')[0].rstrip('+')) >= 47
    if supported:
        emitted = PySpiceNetlister(None).generate_sources({'stim': source})
    else:
        with pytest.raises(NotSupportedError, match='47.*SILK_NGSPICE_BIN') as caught:
            PySpiceNetlister(None).generate_sources({'stim': source})
        assert (version or 'unknown') in str(caught.value)
        print(str(caught.value))
        emitted = source.to_spice()  # Also confirm the old binary's raw rejection.
    deck = ('current PWL release boundary\n' + emitted +
            '\nRload out 0 1k\n.tran 10p 5n\n.control\n'
            'set numdgt=15\nrun\nwrdata values v(out)\n'
            'meas tran v15 find v(out) at=1.5n\n'
            'meas tran v3 find v(out) at=3n\n.endc\n.end\n')
    (tmp_path / 'deck.cir').write_text(deck)
    result = subprocess.run([executable, '-b', 'deck.cir'], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    log = result.stdout + result.stderr
    path = tmp_path / 'values'
    if not supported:
        assert not path.exists() and 'unknown parameter' in log, log
        return
    assert path.exists(), log
    data = np.loadtxt(path)
    measured = np.interp([1.5e-9, 2.5e-9, 3e-9], data[:, 0], data[:, 1])
    print(f'ngspice {version}, {timing}: V(out) at 1.5 / 2.5 / 3 ns = {measured}')
    if 'r' in timing:
        # Verify repetition away from the discontinuity. Floating-point
        # placement of the exact 3 ns join can select either endpoint.
        np.testing.assert_allclose(measured[:2], [.5, .5], atol=2e-5, rtol=0)
        assert min(abs(measured[2]), abs(measured[2] - 1)) < 2e-5
    else:
        np.testing.assert_allclose(measured, [.5, 1, 1], atol=2e-5, rtol=0)
