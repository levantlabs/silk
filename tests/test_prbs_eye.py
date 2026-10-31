"""Exercise the PRBS example's measurement path without a simulator or PDK."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from silk.examples.metrics import prbs_eye as example
from silk.signals.base import Signal
from tests._env import needs_ngspice, needs_pdk


def _lane(symbols, *, low, swing, model=False):
    # Unequal sample counts and distinct model times prevent a shared-grid
    # centre-column assumption from passing this runtime regression.
    x, y = [], []
    for k, bit in enumerate(symbols):
        phases = [0.0, 0.16, 0.34, 0.86, 0.98] if model else [
            0.0, 0.13, 0.38, 0.79, 0.97,
        ]
        if k % 2:
            phases = sorted([*phases, 0.21])
        for phase in phases:
            x.append((k + phase) * example.UI)
            y.append(low + swing * bit + 0.1 * (phase - 0.5))
    x.append(len(symbols) * example.UI)
    y.append(y[-1])
    return Signal('lane', 'V', 'time', 'V', np.asarray(x), np.asarray(y))


@pytest.mark.parametrize('phase', [0.0, 0.07])
def test_fold_margin_and_evm_on_original_sample_windows(phase, capsys):
    sym_i, sym_q, _ = example.prbs_lanes()
    rx_i = _lane(sym_i, low=0.1, swing=0.8)
    rx_q = _lane(sym_q, low=0.1, swing=0.8)
    model = _lane(sym_i, low=0.15, swing=0.7, model=True)
    fold = example.fold_at(
        rx_i, model, phase * example.UI, len(sym_i) * example.UI,
        'test-phase', 'analytic ramp within each symbol', sym_i,
    )

    assert fold.n_traces >= example.MIN_TRACES
    assert len({len(window.x) for window in fold.eye.traces}) > 1
    first_bit = 2 * example.N_LEAD
    bits_i = np.asarray(sym_i[first_bit:first_bit + fold.n_traces])
    bits_q = np.asarray(sym_q[first_bit:first_bit + fold.n_traces])
    actual_phase = (fold.t_start - example.T_DATA) / example.UI
    offset = 0.1 * actual_phase
    np.testing.assert_allclose(
        fold.centre_samples, 0.1 + 0.8 * bits_i + offset, atol=1e-13,
        rtol=0,
    )
    assert fold.bit_aware == pytest.approx(0.8)
    assert fold.model_bit_aware == pytest.approx(0.7)

    evm = example.measure_evm(rx_i, rx_q, 0.5, 1.0, fold, sym_i, sym_q)
    err_i = 0.1 - 0.2 * bits_i + offset
    err_q = 0.1 - 0.2 * bits_q + offset
    expected_evm = 100 * np.sqrt(np.mean(err_i**2 + err_q**2) / 0.5)
    assert evm.evm_percent == pytest.approx(expected_evm)
    assert evm.evm_true == pytest.approx(expected_evm)
    assert evm.n_wrong == 0
    assert evm.identity_dv < 1e-13

    example.report_folds([fold])
    example.report_evm([evm])
    report = capsys.readouterr().out
    assert '800.00' in report
    assert '700.00' in report
    assert 'nan' not in report.lower()


def test_report_openings_on_auto_centered_irregular_traces(capsys):
    # Catches a report that omits current geometric measurements or invents
    # a shared trace grid. The two analytic eyes have known 0.8/0.7 V heights.
    sym_i, _, _ = example.prbs_lanes()
    rx = _lane(sym_i, low=0.1, swing=0.8)
    model = _lane(sym_i, low=0.15, swing=0.7, model=True)
    stop = len(sym_i) * example.UI
    eye = example.eye_diagram(example.window(rx, example.T_DATA, stop),
                              period=example.UI, auto_center=True, threshold=.5)
    model_eye = example.eye_diagram(example.window(model, example.T_DATA, stop),
                                    period=example.UI, auto_center=True, threshold=.5)
    example.report_openings(eye, model_eye)
    report = capsys.readouterr().out
    assert 'height (mV)' in report
    assert 'width (ps)' in report
    assert 'level (mV)' in report
    assert '800.00' in report
    assert '700.00' in report
    assert 'nan' not in report.lower()


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_prbs_example_runs_and_reports_current_eye_fields(tmp_path):
    # Import coverage cannot catch a removed keyword or obsolete result read
    # inside main(); run the shipped entry point with real sky130/ngspice.
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, 'silk/examples/metrics/prbs_eye.py'], cwd=root,
        env={**os.environ, 'MPLBACKEND': 'Agg', 'SILK_EXAMPLES_OUT': str(tmp_path),
             'PYTHONPATH': str(root) + os.pathsep + os.environ.get('PYTHONPATH', '')},
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'nan' not in result.stdout.lower()
    assert result.stdout.count('Auto-centred geometric openings') == 3
    assert result.stdout.count('level (mV)') == 6
    summary = result.stdout.split('--- auto-centred geometric minima')[1]
    assert 'height (mV)' in summary and 'width (ps)' in summary
    rows = [line.split() for line in summary.splitlines()
            if line.strip().startswith(('0.10 ', '0.60 ', '1.00 '))]
    assert len(rows) == 3
    for row in rows:
        height, model_height, width, model_width = map(float, row[2:6])
        assert 0 < height < 1800 and 0 < width <= 200
        assert height == pytest.approx(model_height, abs=1.0)
        assert width == pytest.approx(model_width, abs=1.0)
    assert len(list((tmp_path / 'metrics' / 'prbs_eye').glob('r_*'))) == 3
    assert list(tmp_path.rglob('*.raw'))
