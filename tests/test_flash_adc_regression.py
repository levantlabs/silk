"""Behavioral regression for silk/examples/flash_adc_example.py.

Runs the example as a subprocess in ``--no-spice`` mode and checks the
printed ADC transfer table. The example hits a large fraction of silk
(sources, behavioral engine, spec reconciliation, report generation,
ladder + T/H + comparator wiring) so a successful run is a good
integration smoke test.

The full SPICE example is skipped here. Opt-in slow tests run a short
conversion bench to check saved-signal readback and typed omission errors.
"""

import os
import re
import subprocess
import sys
import tempfile

import pytest
from tests._env import needs_pdk, needs_ngspice

# Every test here builds against the real installed PDK.
pytestmark = needs_pdk


_EXAMPLE = os.path.join(
    os.path.dirname(__file__), '..', 'silk', 'examples', 'flash_adc_example.py'
)


def _run_example(*extra_args):
    """Run the example in --no-spice mode. Returns stdout as a string."""
    env = dict(os.environ)
    env['MPLBACKEND'] = 'Agg'
    env['PYTHONUNBUFFERED'] = '1'
    with tempfile.TemporaryDirectory(prefix='flash_adc_test_') as tmp:
        # Keep the example's netlist/report writes out of examples/output —
        # suite runs should not touch checked-out artifact folders.
        env['SILK_EXAMPLES_OUT'] = tmp
        result = subprocess.run(
            [sys.executable, _EXAMPLE, '--no-spice', *extra_args],
            capture_output=True, text=True, env=env, timeout=300,
        )
    if result.returncode != 0:
        pytest.fail(
            f"Example exited non-zero ({result.returncode}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def _extract_codes(stdout, n_codes):
    """Parse the ADC transfer table rows and return the Code column as ints."""
    lines = stdout.splitlines()
    try:
        header_idx = next(
            i for i, line in enumerate(lines)
            if 'ADC Transfer Function' in line
        )
    except StopIteration:
        pytest.fail(f'No ADC Transfer Function header in output:\n{stdout}')
    # Skip header + column row + divider
    data_start = header_idx + 3
    row_pattern = re.compile(
        r'^\s*\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+(\d+)\s+[01]+\s*$'
    )
    codes = []
    for line in lines[data_start:]:
        m = row_pattern.match(line)
        if not m:
            break
        codes.append(int(m.group(1)))
    assert len(codes) >= n_codes, (
        f'Expected at least {n_codes} transfer rows, got {len(codes)}.\n'
        f'stdout (last 40 lines):\n' + '\n'.join(lines[-40:])
    )
    return codes[:n_codes]


class TestFlashAdcBehavioral:

    def test_ota_comparator_produces_monotonic_staircase(self):
        """4-bit OTA: behavioral codes should march 0 → 15 monotonically."""
        stdout = _run_example()
        assert 'Calibrations applied:' not in stdout
        assert 'source=spice' not in stdout
        assert '[UNMEASURED] comp1_offset (instance comp1): (no measurement)' in stdout
        codes = _extract_codes(stdout, n_codes=16)
        assert codes[0] == 0, f'first code should be 0, got {codes[0]}'
        assert codes[-1] == 15, f'last code should be 15, got {codes[-1]}'
        for i in range(1, len(codes)):
            assert codes[i] >= codes[i - 1], (
                f'Non-monotonic at cycle {i + 1}: '
                f'{codes[i - 1]} → {codes[i]}'
            )

    def test_strongarm_comparator_produces_monotonic_staircase(self):
        """StrongArm behavioral path must also give clean 0 → 15."""
        stdout = _run_example('--strongarm')
        assert 'Calibrations applied:' not in stdout
        assert '[UNMEASURED] comp15_offset (instance comp15): (no measurement)' in stdout
        codes = _extract_codes(stdout, n_codes=16)
        assert codes[0] == 0
        assert codes[-1] == 15
        for i in range(1, len(codes)):
            assert codes[i] >= codes[i - 1]

    def test_bits_flag_extends_range(self):
        """--bits 5 expands to 31 comparators and 32-cycle staircase."""
        stdout = _run_example('--bits', '5')
        codes = _extract_codes(stdout, n_codes=32)
        assert codes[0] == 0
        assert codes[-1] == 31
        assert max(codes) == 31

    def test_sine_mode_records_sndr_metrics(self):
        """--sine declares th_sndr / adc_sndr specs and reconciles them."""
        stdout = _run_example('--sine')
        # The behavioral path records Measurements and the reconcile report
        # shows both SNDR specs in the Metrics section of print_summary.
        assert 'th_sndr' in stdout
        assert 'adc_sndr' in stdout
        # A pure tone reaches silk's finite floating-point ceiling; summing
        # the residual bins must not cancel to an infinite passing result.
        match = re.search(r'th_sndr\s+>=\s+20 dB\s+measured (inf|[\d.]+) dB', stdout)
        assert match, 'th_sndr metric row not found'
        assert float(match.group(1)) == pytest.approx(166.619, abs=0.001)


@pytest.mark.parametrize('strongarm', [False, True])
def test_characterization_emits_spice_without_a_ground_port(tmp_path, monkeypatch,
                                                           strongarm):
    from silk.examples import flash_adc_example as example

    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    monkeypatch.setattr(example, 'USE_STRONGARM', strongarm)
    dut, _ = example.make_flash_adc(with_specs=False)
    test = example._make_characterization(dut)
    # A gnd subcircuit port is rejected at SPICE emission, even though the
    # behavioral engine can simulate it. Exercise both comparator interfaces.
    assert os.path.isfile(test.netlist.generate_netlist())


@pytest.mark.parametrize('strongarm', [False, True])
@pytest.mark.parametrize('sine', [False, True])
@pytest.mark.parametrize('characterized', [False, True])
def test_comparator_offsets_have_no_measurement_or_calibration_path(
        tmp_path, monkeypatch, strongarm, sine, characterized):
    from silk.examples import flash_adc_example as example

    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    monkeypatch.setattr(example, 'USE_STRONGARM', strongarm)
    monkeypatch.setattr(example, 'USE_SINE', sine)
    dut, _ = example.make_flash_adc(with_specs=not characterized)
    if characterized:
        example._make_characterization(dut)
        assert dut.list_tests() == ['conversion']
    report = dut.reconcile_specs(prefer_source='spice')
    offsets = [outcome for outcome in report.outcomes
               if outcome.spec.name.startswith('comp')]
    assert len(offsets) == example.N_COMP
    assert not report.calibrations_applied
    for k, outcome in enumerate(offsets, 1):
        assert outcome.spec.name == f'comp{k}_offset'
        assert outcome.status == 'UNMEASURED'
        assert outcome.measurement is None
        assert outcome.spec.calibration is None
        assert outcome.spec.parent.name == 'adc_code_span'
        assert 'offset' not in dut.elem_dict[f'comp{k}']['device'].list_tests()


@pytest.mark.parametrize('strongarm', [False, True])
@pytest.mark.parametrize('sine', [False, True])
def test_conversion_saves_observations_without_all_device_nodes(
        tmp_path, monkeypatch, strongarm, sine):
    """Dropping the save list would retain hundreds of unused device nodes."""
    from pathlib import Path
    from silk.examples import flash_adc_example as example

    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    monkeypatch.setattr(example, 'USE_STRONGARM', strongarm)
    monkeypatch.setattr(example, 'USE_SINE', sine)
    dut, _ = example.make_flash_adc(with_specs=False)
    bench = example._make_characterization(dut).netlist
    bench_deck_path = bench.generate_netlist()
    deck = Path(bench_deck_path).read_text()
    saves = [line for line in deck.splitlines() if line.startswith('save ')]
    assert saves and 'save all' not in saves
    saved = set(re.findall(r'v\(([^)]+)\)', '\n'.join(saves)))
    required = {'vin', 'vh', 'vh_buf', 'clk', 'clkb'}
    required.update(f'vref{k}' for k in range(1, 16))
    required.update(f'vcmp{k}' for k in range(1, 16))
    if strongarm:
        required.add('clkb_d')
        required.update(f'xdut.xcomp{k}.{node}'
                        for k in (3, 7, 11)
                        for node in ('vtail', 'di_p', 'di_n'))
    assert saved == required


@pytest.mark.slow
@needs_ngspice
@pytest.mark.parametrize('strongarm', [False, True])
def test_conversion_saved_signals_read_back_and_omissions_raise(
        tmp_path, monkeypatch, strongarm):
    """An omitted observation must raise, never become an empty waveform."""
    from silk.examples import flash_adc_example as example
    from silk.exceptions import SimulationError

    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    monkeypatch.setattr(example, 'USE_STRONGARM', strongarm)
    monkeypatch.setattr(example, 'T_STOP', 1e-9)
    dut, _ = example.make_flash_adc(with_specs=False)
    bench = example._make_characterization(dut).netlist
    bench.simulate()
    for name in ('vin', 'vh', 'vref3', 'vcmp3'):
        assert bench.load_signal(name).y.size > 0
    if strongarm:
        for node in ('vtail', 'di_p', 'di_n'):
            assert bench.load_signal(f'xdut.xcomp3.{node}').y.size > 0
    # Remove a signal that the conversion metric actually reads.
    bench.set_save_list(['vin'])
    bench.simulate()
    with pytest.raises(SimulationError, match='vcmp1'):
        example._code_trace(bench)


@pytest.mark.parametrize('clocked', [False, True])
@pytest.mark.parametrize('pattern, expected', [
    ('low', 0.1), ('high', -0.1), ('single', 0.0),
    ('multiple', float('nan')),
])
def test_offset_decision_limits(clocked, pattern, expected):
    import numpy as np
    from silk.examples import flash_adc_example as f
    from silk.signals.base import Signal

    t = np.linspace(0, 100 * f.CLK_PER, 10001)
    vp = 0.9 + 0.2 * t / t[-1]
    high = {'low': np.zeros(t.size, dtype=bool),
            'high': np.ones(t.size, dtype=bool),
            'single': vp >= 1.0,
            'multiple': ((vp >= .96) & (vp < 1.0)) | (vp >= 1.04)}[pattern]
    if clocked:
        high &= np.mod(t, f.CLK_PER) > .5 * f.CLK_PER
    out = Signal('vout', 'voltage', 'time', 'voltage', t, high * f.VDD)
    inp = Signal('vp', 'voltage', 'time', 'voltage', t, vp)
    got = f._offset_from_decisions(out, inp, clocked=clocked)
    if np.isnan(expected):
        assert np.isnan(got)
    else:
        assert got == pytest.approx(expected, abs=.002)


def test_characterize_offset_prints_cap_counts(monkeypatch, capsys):
    """Report censored observations and refuse to calibrate from them."""
    from types import SimpleNamespace
    from silk.circuit.module_test import ModuleTestbench
    from silk.circuit.spec_report import SpecReport
    from silk.examples import flash_adc_example as f
    from silk.exceptions import SimulationError

    reports = {i: SpecReport('offset', 'comp',
                            {'offset_v': value, 'offset_cap_side': side})
               for i, (value, side) in enumerate([(-.1, -1), (.1, 1), (.1, 1), (0., 0)])}
    monkeypatch.setattr(ModuleTestbench, 'run_monte_carlo',
                        lambda *a, **kw: SimpleNamespace(reports=reports, failed={}))
    with pytest.raises(SimulationError, match='uncensored'):
        f.characterize_offset(samples=4)
    output = capsys.readouterr().out
    assert 'capped=3/4: -100 mV=1, +100 mV=2' in output
    assert 'mean=' not in output


@pytest.mark.parametrize('upper', [False, True])
def test_offset_one_step_from_end_is_capped(upper):
    import numpy as np
    from silk.examples import flash_adc_example as f
    from silk.signals.base import Signal

    vp = np.linspace(.9, 1.1, 4001)
    out = np.zeros(vp.size)
    out[-1 if upper else 1:] = f.VDD
    got = f._offset_from_decisions(
        Signal('vout', 'voltage', 'voltage', 'voltage', vp, out),
        Signal('vp', 'voltage', 'voltage', 'voltage', vp, vp), clocked=False)
    assert got == pytest.approx(.1 if upper else -.1, abs=1e-12)
