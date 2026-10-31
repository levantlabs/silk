"""Example metrics must retain polarity, normalize excitation and check flatness."""
import pytest


def _rawfile(path, axis, rows):
    complex_data = axis == 'frequency'
    variables = [(axis, axis), ('i(vvout)', 'current'), ('v(out)', 'voltage')]
    lines = ['Title: mirror measurement fixture', 'Date: test',
             'Plotname: AC Analysis' if complex_data else 'Plotname: Transient Analysis',
             'Flags: complex' if complex_data else 'Flags: real',
             'No. Variables: 3', f'No. Points: {len(rows)}', 'Variables:']
    lines.extend(f'\t{i}\t{name}\t{kind}' for i, (name, kind) in enumerate(variables))
    lines.append('Values:')
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            value = f'{complex(val).real},{complex(val).imag}' if complex_data else str(val)
            lines.append(f'{i if j == 0 else ""}\t{value}')
    path.write_text('\n'.join(lines) + '\n')
    return path


@pytest.fixture
def mirror(tmp_path, monkeypatch):
    from silk.examples import current_mirror_example as ex
    monkeypatch.setattr(ex, 'example_output_dir', lambda *args: str(tmp_path))
    return ex.build_current_mirror()


@pytest.mark.parametrize('sink, passed', [(10e-6, True), (-10e-6, False)])
def test_copy_preserves_sink_polarity(mirror, tmp_path, sink, passed):
    raw = _rawfile(tmp_path / 'copy.raw', 'time',
                   [(t, -sink, 0.9) for t in [0, 0.5e-6, 0.75e-6, 1e-6]])
    report = mirror._tests['match'].evaluate(rawfile=raw)
    assert report.metrics['i_out'] == pytest.approx(sink)
    assert report.metrics['gain_error_pct'] == pytest.approx(0 if passed else 200)
    assert report.passed is passed


def test_copy_ignores_early_transient_but_rejects_late_ripple(mirror, tmp_path):
    tb = mirror._tests['match']
    raw = _rawfile(tmp_path / 'settled.raw', 'time',
                   [(0, 0, 0.9), (0.5e-6, -10e-6, 0.9),
                    (0.6e-6, -10.004e-6, 0.9), (1e-6, -10.008e-6, 0.9)])
    report = tb.evaluate(rawfile=raw)
    # Trapezoids span 0.1 and 0.4 us: mean 10.0052 uA, not final 10.008.
    assert report.metrics['i_out'] == pytest.approx(10.0052e-6)
    assert report.passed
    raw = _rawfile(tmp_path / 'ripple.raw', 'time',
                   [(0, -10e-6, 0.9), (0.5e-6, -10.2e-6, 0.9),
                    (0.75e-6, -9.8e-6, 0.9), (1e-6, -10.2e-6, 0.9)])
    report = tb.evaluate(rawfile=raw)
    assert report.metrics['gain_error_pct'] < 5
    assert not report.passed


@pytest.mark.parametrize('voltage', [1.0, 0.25 + 0.5j])
def test_rout_normalizes_actual_complex_excitation(mirror, tmp_path, voltage):
    # G=1 uS -> Rout=1 Mohm even with reactive current and a phased stimulus.
    raw = _rawfile(tmp_path / 'ac.raw', 'frequency',
                   [(f, -(1e-6 + 2e-6j) * voltage, voltage) for f in [0.1, 1, 10]])
    report = mirror._tests['zout'].evaluate(rawfile=raw)
    assert report.metrics['rout_ohm'] == pytest.approx(1e6)
    assert report.passed


def test_rout_rejects_nonflat_low_band(mirror, tmp_path):
    raw = _rawfile(tmp_path / 'nonflat.raw', 'frequency',
                   [(0.1, -1e-6, 1), (1, -1.1e-6, 1), (10, -2e-6, 1)])
    report = mirror._tests['zout'].evaluate(rawfile=raw)
    assert report.metrics['rout_ohm'] == pytest.approx(1e6)
    assert not report.passed
