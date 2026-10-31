"""Operating-point payload shape, selection, and row roles (TASK-533)."""
import base64
import numpy as np
import pytest
from silk.signals.view import emit_view, emit_view_data
from tests.test_view_op_e2e import raw_plot, op_raw, ac_raw


def decoded(level, key):
    return np.frombuffer(base64.b64decode(level[key]), dtype='<f8' if key == 'x' or level['dt'] == 'f64' else '<f4')


def test_op_keeps_every_vector_and_default(tmp_path):
    source = tmp_path / 'op.raw'
    source.write_text(ac_raw() + op_raw())
    p = emit_view_data(source)
    assert p['format'] == 2 and p['default_plot'] == 1
    op = p['plots'][1]
    assert op['kind'] == 'op'
    assert op['x']['name'] == 'sample' and op['x']['unit'] == ''
    np.testing.assert_array_equal(decoded(op['x']['levels'][0], 'x'), [0])
    assert [s['name'] for s in op['signals']] == ['v(in)', 'v(out)', 'i(v1)']
    np.testing.assert_allclose([decoded(s['levels'][0], 'y')[0] for s in op['signals']], [1, 2, -.003])
    assert not any(s.get('role') == 'coordinate' for s in op['signals'])


def test_vector_zero_selection_counts_and_designer_address(tmp_path):
    source = tmp_path / 'op.raw'
    source.write_text(raw_plot('Operating Point', [('v(xx1<0>.in)', 'voltage'), ('v(out)', 'voltage')], [[3, 4]]))
    p = emit_view_data(source, signals=['v(xx1[0].in)'], labels='designer')
    s, = p['plots'][0]['signals']
    assert s['name'] == 'v(xx1<0>.in)'
    assert s['label'] == s['address'] == 'v(xx1[0].in)'
    assert decoded(s['levels'][0], 'y')[0] == 3
    out = emit_view(source, tmp_path / 'op.html', exclude=['v(out)'])
    assert out.n_signals_total == 2
    assert len(emit_view_data(source, exclude=['v(out)'])['plots'][0]['signals']) == 1
    assert not emit_view_data(source, max_depth=0)['plots'][0]['signals'][0]['name'].startswith('v(xx1')


@pytest.mark.parametrize('name,axis,type_,kind', [('Transient Analysis', 'time', 'time', 'tran'),
    ('DC transfer characteristic', 'sweep', 'voltage', 'dc'),
    ('DC transfer characteristic', 'v(v-sweep)', 'voltage', 'dc')])
def test_one_point_analysis_keeps_its_axis(tmp_path, name, axis, type_, kind):
    """TASK-591 (Q24): only a true operating point has no x axis."""
    source = tmp_path / 'point.raw'
    source.write_text(raw_plot(name, [(axis, type_), ('v(time)', 'voltage'), ('v(sweep)', 'voltage')], [[1, 2, 3]]))
    p = emit_view_data(source)['plots'][0]
    assert p['kind'] == kind
    assert p['x']['name'] == axis
    np.testing.assert_array_equal(decoded(p['x']['levels'][0], 'x'), [1])
    assert [s['name'] for s in p['signals']] == ['v(time)', 'v(sweep)']
    assert all('role' not in s for s in p['signals'])


def test_true_op_named_like_an_axis_keeps_every_vector(tmp_path):
    source = tmp_path / 'op.raw'
    source.write_text(raw_plot('Operating Point', [('time', 'time'), ('v(sweep)', 'voltage')], [[1, 2]]))
    p = emit_view_data(source)['plots'][0]
    assert p['kind'] == 'op'
    assert [s['name'] for s in p['signals']] == ['time', 'v(sweep)']
    assert all('role' not in s for s in p['signals'])


def test_one_point_complex_keeps_its_frequency_axis(tmp_path):
    source = tmp_path / 'ac.raw'
    source.write_text(raw_plot('AC Analysis', [('frequency', 'frequency'), ('v(out)', 'voltage')], [[10, 2]], complex=True))
    p = emit_view_data(source)['plots'][0]
    assert p['kind'] == 'ac' and p['x']['name'] == 'frequency'
    assert [s['name'] for s in p['signals']] == ['v(out) (mag)', 'v(out) (phase)']


def test_complex_first_and_constants_exception(tmp_path):
    source = tmp_path / 'point.raw'
    source.write_text(raw_plot('Operating Point', [('frequency', 'frequency'), ('v(out)', 'voltage')], [[1, 2]], complex=True))
    p = emit_view_data(source)['plots'][0]
    assert p['kind'] == 'ac'
    assert [s['name'] for s in p['signals']] == ['v(out) (mag)', 'v(out) (phase)']
    source.write_text(raw_plot('constants', [('yes', 'notype'), ('pi', 'notype')], [[1, 3.14]]))
    assert emit_view_data(source)['plots'][0]['kind'] == 'constants'
