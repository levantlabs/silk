"""Waveform voltage-source operating-point parity (TASK-473)."""
import pytest

from silk.circuit.spice_netlister import PySpiceNetlister
from silk.circuit.xyce_netlister import XyceNetlister
from silk.circuit.variable import Variable
from silk.exceptions import NotSupportedError
from silk.sources import (VoltageSource, CurrentSource, PWL, Pulse, Sine,
                          Exponential, SFFM, AM, TransientNoise, RandomSource)


def emit(source, backend=XyceNetlister):
    return backend(None).generate_sources({'stim': source})


@pytest.mark.parametrize('dc', [0, .7])
def test_pwl_series_element_receives_own_dc_without_changing_offset(dc):
    source = VoltageSource('stim', 'out', '0', dc=dc,
                           pwl=PWL([(1e-6, 1.2), (2e-6, 0)]))
    expected = ('*** VoltageSource: stim\n'
                'Vstim__silk_pwl out __silk_src_Vstim DC 1.2 PWL(1e-06 1.2 2e-06 0.0)\n'
                f'Vstim__silk_dc __silk_src_Vstim 0 DC {dc}\n')
    assert emit(source) == expected
    assert emit(source, PySpiceNetlister) == '*** VoltageSource: stim\n' + source.to_spice()


@pytest.mark.parametrize('dc', [0, .7])
def test_pulse_dc_override_and_transient_are_preserved(dc):
    source = VoltageSource('stim', 'out', '0', dc=dc, ac=1,
                           pulse=Pulse(1.2, 0, tr='1n', tf='1n', pw='1u', per='2u'))
    assert 'Vstim__silk_wave out __silk_src_Vstim DC 1.2 PULSE(' in emit(source)
    assert f'Vstim__silk_dc __silk_src_Vstim 0 DC {dc} AC 1' in emit(source)
    assert emit(source, PySpiceNetlister) == '*** VoltageSource: stim\n' + source.to_spice()


def test_generic_waveform_initial_value_preserves_expression_token():
    class CustomWave:
        def to_spice(self):
            return 'PWL(0 {level} 1u 0)'

        def initial_value(self):
            return '{level}'

    source = VoltageSource('stim', 'out', '0', waveform=CustomWave())
    assert 'DC {level} PWL(0 {level} 1u 0)' in emit(source)


def test_unknown_generic_initial_value_refuses_instead_of_silent_zero():
    class CustomWave:
        def to_spice(self):
            return 'PWL(0 1.2 1u 0)'

    source = VoltageSource('stim', 'out', '0', waveform=CustomWave())
    with pytest.raises(NotSupportedError, match='initial_value'):
        emit(source)
    source.dc = .7
    with pytest.raises(NotSupportedError, match='initial_value'):
        emit(source)


def test_current_waveform_initial_and_constant_source_emission():
    source = CurrentSource('stim', 'out', '0', sine=Sine(1.2, .3, 1e3))
    assert 'Istim__silk_wave out 0 DC 1.2 SIN(' in emit(source)
    assert 'Istim__silk_wave out 0 SIN(' in emit(source, PySpiceNetlister)
    constant = VoltageSource('stim', 'out', '0', ac=1)
    assert emit(constant) == emit(constant, PySpiceNetlister)


def test_initial_values_resolve_variables_at_each_emission():
    value = Variable('level', value=1.2)
    source = VoltageSource('stim', 'out', '0', pwl=PWL([(0, value), (1, 0)]))
    assert 'DC 1.2 PWL(' in emit(source)
    value.value = 1.7
    assert 'DC 1.7 PWL(' in emit(source)


def test_voltage_pwl_and_pulse_refused():
    from silk.exceptions import CircuitError
    with pytest.raises(CircuitError, match='stim.*pwl.*pulse'):
        VoltageSource('stim', 'out', '0', pwl=PWL([(0, 9), (1, 0)]),
                      pulse=Pulse(1.2, 0, tr='1n', tf='1n'))


def test_pwl_parameter_expression_remains_a_token():
    value = Variable('level', value='{level}')
    source = VoltageSource('stim', 'out', '0', pwl=PWL([(0, value), (1, 0)]))
    assert 'DC {level} PWL(0.0 {level} 1.0 0.0)' in emit(source)


def test_nested_sources_follow_parent_backend_without_mutating_child(tmp_path, pdk):
    from silk.circuit.modules import Module
    child = Module('sky130', str(tmp_path), 'child.cir', pdk_details=pdk,
                   module_name='child')
    child.define_module_ports({})
    child.add_source(VoltageSource('stim', 'out', '0', pwl=PWL([(0, 1.2), (1, 0)])))
    modules = {'child': {'device': child}}
    original = PySpiceNetlister(pdk).generate_module_definitions(modules)
    assert 'DC 1.2 PWL(' in XyceNetlister(pdk).generate_module_definitions(modules)
    assert PySpiceNetlister(pdk).generate_module_definitions(modules) == original


def run_analysis(tmp_path, backend, source_text, analysis):
    """Measure a source's DC bias, or the AC slope of its squared voltage."""
    import shutil
    import subprocess
    import numpy as np
    from silk import settings
    exe = settings.get('SILK_NGSPICE_BIN' if backend == 'ngspice' else 'SILK_XYCE_BIN')
    if not exe or not shutil.which(exe):
        pytest.skip(f'{backend} executable unavailable through silk settings')
    commands = {'op': 'op', 'dc': 'dc Vdummy 0 1 1',
                'ac': 'ac lin 1 1000 1000', 'tran': 'tran 10n 2u'}
    observed = 'sense' if analysis == 'ac' else 'out'
    if backend == 'ngspice':
        tail = (f'.control\n{commands[analysis]}\n'
                f'wrdata values v({observed})\nquit\n.endc\n')
    else:
        print_type = 'dc' if analysis == 'op' else analysis
        tail = (f'.{commands[analysis]}\n.print {print_type} format=csv '
                f'file=values.csv V({observed})\n')
    deck = ('waveform operating point\n' + source_text +
            '\nRload out 0 1k\nVdummy dummy 0 0\nRdummy dummy 0 1k\n'
            'Bsense sense 0 V={V(out)*V(out)}\n' + tail + '.end\n')
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / 'deck.cir').write_text(deck)
    result = subprocess.run([exe, *(['-b'] if backend == 'ngspice' else []), 'deck.cir'],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    log = result.stdout + result.stderr
    (tmp_path / 'simulator.log').write_text(log)
    assert result.returncode == 0, log
    data_file = tmp_path / ('values' if backend == 'ngspice' else 'values.csv')
    assert data_file.exists(), log
    return np.loadtxt(data_file, delimiter=None if backend == 'ngspice' else ',',
                      skiprows=0 if backend == 'ngspice' else 1, ndmin=2)


# Literal expectations are checked against native ngspice in the slow cases.
CASES = [
    pytest.param('pulse', Pulse(1.2, 0, tr='1n', tf='1n', pw='1u', per='2u'), 1.2, id='pulse'),
    pytest.param('pwl', PWL([(1e-6, 1.2), (2e-6, 0)]), 1.2, id='pwl'),
    pytest.param('sine', Sine(1.2, .3, 1e3), 1.2, id='sine'),
    pytest.param('sine', Sine(1.2, .3, 1e3, phase=30), 1.35, id='sine-phase'),
    pytest.param('sine', Sine(1.2, .3, 1e3, td=1e-6, phase=30), 1.35, id='sine-delay-phase'),
    pytest.param('exponential', Exponential(1.2, 0, 1e-6, 1e-6, 2e-6, 1e-6), 1.2, id='exponential'),
    pytest.param('sffm', SFFM(1.2, .3, 1e3, 2, 10), 1.2, id='sffm'),
    pytest.param('sffm', SFFM(1.2, .3, 1e3, 2, 10, 30, 40), 1.4915166986870765, id='sffm-phases'),
]

OP_ONLY_CASES = [
    pytest.param('am', AM(.3, 1.2, 10, 1e3, phases=30), 0, id='am'),
    pytest.param('am', AM(.3, 1.2, 10, 1e3, td=1e-6, phases=30), 0, id='am-delay'),
    pytest.param('trnoise', TransientNoise(.1, 1e-6), 0, id='noise'),
    pytest.param('trnoise', TransientNoise(0, 1e-6, nalpha=1, namp=.2), 0, id='noise-flicker'),
    pytest.param('trnoise', TransientNoise(0, 1e-6, rtsam=.3, rtscapt=1e-6, rtsemt=1e-6), 0, id='noise-rts'),
    *[pytest.param('trrandom', RandomSource(kind, 1e-6, param1=.3, param2=1.2), 1.2,
                   id=f'random-{kind}') for kind in range(1, 5)],
]


@pytest.mark.parametrize('slot,wave,want', CASES + OP_ONLY_CASES)
def test_defined_initial_values(slot, wave, want):
    assert float(wave.initial_value()) == pytest.approx(want)


@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.parametrize('slot,wave,want', CASES)
@pytest.mark.parametrize('dc', [0, .7])
@pytest.mark.parametrize('analysis', ['op', 'dc', 'ac'])
def test_real_operating_point_and_ac_bias_match(tmp_path, slot, wave, want, dc, analysis):
    import numpy as np
    source = VoltageSource('stim', 'out', '0', dc=dc, ac=1, **{slot: wave})
    bias = want + dc
    expected = 2 * bias if analysis == 'ac' else bias
    from silk.simulation._ngspice_version import ngspice_version
    for backend, netlister in [('ngspice', PySpiceNetlister), ('xyce', XyceNetlister)]:
        if backend == 'ngspice' and slot == 'sffm' and ngspice_version() != '42':
            with pytest.raises(NotSupportedError, match='ngspice.*SFFM.*stim'):
                emit(source, netlister)
            continue
        if backend == 'xyce' and slot == 'sffm' and (wave.phasec or wave.phases):
            with pytest.raises(NotSupportedError, match='phase'):
                emit(source, netlister)
            continue
        data = run_analysis(tmp_path / backend, backend, emit(source, netlister), analysis)
        # ngspice wrdata includes a scale; Xyce OP/DC prints only V(out).
        col = 0 if backend == 'xyce' and analysis in ('op', 'dc') else 1
        np.testing.assert_allclose(data[:, col], expected, rtol=1e-7, atol=1e-10)
        if analysis == 'ac':
            np.testing.assert_allclose(data[:, col + 1], 0, atol=1e-10)


@pytest.mark.slow
@pytest.mark.requires_xyce
def test_pwl_offset_transient_unchanged_by_added_dc(tmp_path):
    import numpy as np
    source = VoltageSource('stim', 'out', '0', dc=.7,
                           pwl=PWL([(0, 1.2), (1e-6, 0)]))
    for backend, netlister in [('ngspice', PySpiceNetlister), ('xyce', XyceNetlister)]:
        data = run_analysis(tmp_path / backend, backend, emit(source, netlister), 'tran')
        np.testing.assert_allclose(np.interp([0, 1e-6, 2e-6], data[:, 0], data[:, 1]),
                                   [1.9, .7, .7], atol=1e-8)


@pytest.mark.parametrize('slot,wave,want', OP_ONLY_CASES)
@pytest.mark.parametrize('cls', [VoltageSource, CurrentSource])
@pytest.mark.parametrize('generic', [False, True])
@pytest.mark.parametrize('dc,ac', [(0, 0), (.7, 0), (0, 1)])
def test_xyce_unsupported_waveforms_refused_before_emission(monkeypatch, slot, wave, want, cls, generic, dc, ac):
    from silk.simulation import _ngspice_version
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: '42')
    source = cls('stim', 'out', '0', dc=dc, ac=ac,
                 **{'waveform' if generic else slot: wave})
    with pytest.raises(NotSupportedError) as error:
        emit(source)
    message = str(error.value)
    assert type(wave).__name__ in message
    assert cls.__name__ in message and 'stim' in message
    assert 'ngspice' in message
    if slot == 'am':
        with pytest.raises(NotSupportedError, match='42.*stim.*phase'):
            emit(source, PySpiceNetlister)
    else:
        assert emit(source, PySpiceNetlister) == f'*** {cls.__name__}: stim\n' + source.to_spice()


@pytest.mark.parametrize('cls', [VoltageSource, CurrentSource])
def test_multiple_waveforms_refused_before_backend_selection(cls):
    from silk.exceptions import CircuitError
    with pytest.raises(CircuitError, match='stim.*pwl.*am'):
        cls('stim', 'out', '0', am=AM(1, 0, 1e5, 1e6),
            pwl=PWL([(0, 1.2), (1e-6, 0)]))
    with pytest.raises(CircuitError, match='stim.*sine.*am'):
        cls('stim', 'out', '0', sine=Sine(1.2, .3, 1e3), am=AM(1, 0, 1e5, 1e6))


@pytest.mark.parametrize('cls', [VoltageSource, CurrentSource])
def test_nested_unsupported_waveform_refused_by_parent_xyce(tmp_path, pdk, cls):
    from silk.circuit.modules import Module
    child = Module('sky130', str(tmp_path), 'child.cir', pdk_details=pdk,
                   module_name='child')
    child.define_module_ports({})
    child.add_source(cls('stim', 'out', '0', trnoise=TransientNoise(.1, 1e-6)))
    modules = {'child': {'device': child}}
    original = PySpiceNetlister(pdk).generate_module_definitions(modules)
    with pytest.raises(NotSupportedError, match='TransientNoise'):
        XyceNetlister(pdk).generate_module_definitions(modules)
    assert PySpiceNetlister(pdk).generate_module_definitions(modules) == original


@pytest.mark.parametrize('wave', [Pulse(1.2, 0, td=-1e-6, tr='1n', tf='1n'),
                                   Sine(1.2, .3, 1e3, td=-1e-6),
                                   Exponential(1.2, 0, td1=-1e-6),
                                   AM(.3, 1.2, 10, 1e3, td=-1e-6)])
def test_negative_delays_refuse_unverified_initial_values(wave):
    with pytest.raises(NotSupportedError, match='delay'):
        wave.initial_value()


@pytest.mark.slow
@pytest.mark.parametrize('slot,wave,want', OP_ONLY_CASES)
@pytest.mark.parametrize('dc', [0, .7])
@pytest.mark.parametrize('analysis', ['op', 'dc', 'ac'])
def test_ngspice_only_waveform_defaults(tmp_path, slot, wave, want, dc, analysis):
    import numpy as np
    source = VoltageSource('stim', 'out', '0', dc=dc, ac=1, **{slot: wave})
    if slot == 'am':
        # These fixtures author a nonzero phase, ignored by 42 and now refused.
        with pytest.raises(NotSupportedError, match='ngspice.*stim'):
            emit(source, PySpiceNetlister)
        return
    data = run_analysis(tmp_path, 'ngspice', emit(source, PySpiceNetlister), analysis)
    np.testing.assert_allclose(data[:, 1], (dc + want) * (2 if analysis == 'ac' else 1),
                               rtol=1e-7, atol=1e-10)




@pytest.mark.parametrize('wave', [Sine(1.2, .3, 1e3, phase='{phase}'),
                                   SFFM(1.2, .3, 1e3, 2, 10, phasec=float('nan'))])
def test_unresolved_phase_arithmetic_has_actionable_typed_error(wave):
    for dc in (0, .7):
        with pytest.raises(NotSupportedError, match='phase'):
            emit(VoltageSource('stim', 'out', '0', dc=dc, waveform=wave))


def test_sine_phase_variable_and_si_levels_resolve_each_emission():
    phase = Variable('phase', value=30)
    wave = Sine('1200m', '300m', '1k', phase=phase)
    source = VoltageSource('stim', 'out', '0', waveform=wave)
    assert float(emit(source).split('DC ', 1)[1].split()[0]) == pytest.approx(1.35)
    phase.value = 90
    assert 'DC 1.5 ' in emit(source)


@pytest.mark.parametrize('dc', ['0', Variable('explicit_zero', value=0)])
def test_zero_dc_is_separate_from_waveform_initial_value(dc):
    source = VoltageSource('stim', 'out', '0', dc=dc,
                           pulse=Pulse(1.2, 0, tr='1n', tf='1n'))
    assert 'Vstim__silk_wave out __silk_src_Vstim PULSE(' in source.to_spice()
    assert 'Vstim__silk_dc __silk_src_Vstim 0 DC 0' in source.to_spice()
    assert 'Vstim__silk_wave out __silk_src_Vstim DC 1.2 PULSE(' in emit(source)
