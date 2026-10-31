"""TASK-530: [k] on every Python-facing surface (inputs, display, reports).

Fixture: top > xx1<0> (X, from add_module(X, 'x1[0]')) > xxy<1> (Y, 'xy[1]')
> net g[0]; top-level net d[0] driven by source bias[0]. Ideal elements only.
"""
import os

import pytest

from silk import sources
from silk.circuit.modules import Module
from silk.circuit.ports import PortDir
from silk.elements import elements
from silk.exceptions import NotSupportedError, CircuitError
from silk.simulation import Transient
from tests._env import needs_ngspice, HAVE_XYCE


def _build(tmp_path, simulator='ngspice', copies=('x1[0]',)):
    out = str(tmp_path)
    Y = Module(None, out, 'y.cir', None, module_name='ym')
    Y.define_module_ports({'a': PortDir.IN})
    Y.add(elements.resistor('R1', p='a', n='g[0]', value=1000))
    Y.add(elements.resistor('R2', p='g[0]', n='0', value=1000))
    X = Module(None, out, 'x.cir', None, module_name='xm')
    X.define_module_ports({'a': PortDir.IN})
    X.add_module(Y, 'xy[1]', nets={'a': 'a'})
    top = Module(None, out, 't.cir', None)
    top.add_source(sources.VoltageSource('bias[0]', positive='d[0]', negative='0', dc=1.0))
    handles = [top.add_module(X, c, nets={'a': 'd[0]'}) for c in copies]
    top.add_sim_type(Transient(step=1e-9, stop=2e-9))
    top.simulator = simulator
    return top, handles


def _save_lines(top):
    deck_path = top.generate_netlist()
    return open(deck_path).read()


def _resolved(top):
    from silk.circuit.modules_netlist import _resolve_save_list
    return _resolve_save_list(top)


def test_save_list_bracket_and_angle_spelling_select_the_same_nodes(tmp_path, capsys):
    top, _ = _build(tmp_path)
    top.set_output_options(mode='overwrite')
    top.set_save_list(['xx1[0].xxy[1].g[0]'])
    a = _resolved(top)
    top.set_save_list(['xx1<0>.xxy<1>.g[0]'])
    assert a == _resolved(top) == ['xx1<0>.xxy<1>.g[0]']


def test_save_list_block_selector_and_index_wildcard(tmp_path, capsys):
    top, _ = _build(tmp_path, copies=('x1[0]', 'x1[1]'))
    top.set_save_list(['xx1[0]'])
    assert _resolved(top) == ['xx1<0>.xxy<1>.g[0]']
    top.set_save_list(['xx1[*].xxy[1].g[0]'])
    assert sorted(_resolved(top)) == ['xx1<0>.xxy<1>.g[0]', 'xx1<1>.xxy<1>.g[0]']


def test_save_list_exact_bracketed_leaves_and_no_character_class(tmp_path, capsys):
    top, _ = _build(tmp_path, copies=('x1[0]', 'x1[1]'))
    top.set_save_list(['d[0]'])
    assert _resolved(top) == ['d[0]']
    top.set_save_list(['xx1[01]'])               # a class would select copies 0 and 1
    with pytest.raises(CircuitError):
        _resolved(top)


def test_save_list_refusal_suggests_bracket_spelling(tmp_path):
    top, _ = _build(tmp_path)
    top.set_save_list(['xx1[0].zz[1].g[0]'])
    with pytest.raises(CircuitError) as e:
        _resolved(top)
    tail = str(e.value).split('Closest', 1)[1]
    assert 'xx1[0]' in tail and '<' not in tail


@pytest.mark.slow
@pytest.mark.requires_xyce
def test_save_list_xyce_bracket_and_angle_render_the_same_print(tmp_path):
    top, _ = _build(tmp_path, simulator='xyce')
    top.set_output_options(mode='overwrite')
    top.set_save_list(['xx1[0].xxy[1].g[0]'])
    a = [l for l in _save_lines(top).splitlines() if l.startswith('.print')]
    top.set_save_list(['xx1<0>.xxy<1>.g[0]'])
    b = [l for l in _save_lines(top).splitlines() if l.startswith('.print')]
    assert a == b and 'v(xx1<0>:xxy<1>:g[0])' in a[0]


@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.skipif(not HAVE_XYCE, reason='requires Xyce (SILK_XYCE_BIN setting)')
def test_view_selector_on_xyce_full_wrapped_colon_name(tmp_path):
    from silk.signals.view import emit_view_data
    top, _ = _build(tmp_path, simulator='xyce')
    top.set_save_list(['xx1[0].xxy[1].g[0]', 'd[0]'])
    top.simulate()
    sig = lambda s: [x['name'] for x in emit_view_data(top.results_file, signals=[s])['plots'][0]['signals']]
    assert sig('v(xx1[0]:xxy[1]:g[0])') == sig('v(xx1<0>:xxy<1>:g[0])') == ['V(XX1<0>:XXY<1>:G[0])']
    assert sig('d[0]') == ['V(D[0])']


# ---- get_signal_names / path / repr -----------------------------------------

def test_collision_safe_designer_names_pure():
    from silk.spice_names import designer_signal_names
    names = ['time', 'v(xx1<0>.mid)', 'i(vbias<0>)', 'v(out)']
    assert designer_signal_names(names) == ['time', 'v(xx1[0].mid)', 'i(vbias[0])', 'v(out)']
    for order in (['time', 'v(xx1[0].mid)', 'v(xx1<0>.mid)'], ['time', 'v(xx1<0>.mid)', 'v(xx1[0].mid)']):
        assert designer_signal_names(order) == order          # whole collision group stays raw
    assert designer_signal_names(['time', 'V(XX1<0>:Y)', 'v(xx1[0]:y)'])[1] == 'V(XX1<0>:Y)'   # case-insensitive group
    assert designer_signal_names(['time']) == ['time']        # axis never aliased


@needs_ngspice
def test_get_signal_names_default_brackets_and_raw_option(tmp_path):
    top, _ = _build(tmp_path)
    top.simulate()
    raw = top.get_signal_names(raw=True)
    assert 'i(vbias<0>)' in raw and 'v(xx1<0>.xxy<1>.g[0])' in raw
    names = top.get_signal_names()
    assert len(names) == len(raw) and names[0] == raw[0]
    assert 'i(vbias[0])' in names and 'v(xx1[0].xxy[1].g[0])' in names
    assert all('<' not in n for n in names)
    import numpy as np
    for a, r in (('i(vbias[0])', 'i(vbias<0>)'), ('v(xx1[0].xxy[1].g[0])', 'v(xx1<0>.xxy<1>.g[0])')):
        np.testing.assert_array_equal(top.load_signal(a[2:-1], 'I' if a[0] == 'i' else 'V').y,
                                      top.load_signal(r[2:-1], 'I' if r[0] == 'i' else 'V').y)


def test_get_signal_names_collision_keeps_raw_and_request_still_refuses(tmp_path):
    """Historical raw names remain readable even though new names are stricter."""
    import numpy as np
    from silk.exceptions import AmbiguousSignalError
    from silk.signals.rawfile import RawFile, RawPlot, RawVector, write_rawfile
    top = Module(None, str(tmp_path), 't.cir', None)
    vectors = [RawVector('time', 'time', data=np.array([0., 1.]))]
    vectors.extend(RawVector(name, 'voltage', data=np.array([.5, .5]))
                   for name in ('v(xx1[0].mid)', 'v(xx1<0>.mid)'))
    path = tmp_path / 'legacy.raw'
    write_rawfile(path, RawFile([RawPlot({'plotname': 'Transient Analysis'},
                                       vectors, npoints=2, real=True, padded=True)]))
    top.load_file(str(path))
    top.read_raw_file()
    raw = top.get_signal_names(raw=True)
    assert 'v(xx1[0].mid)' in raw and 'v(xx1<0>.mid)' in raw
    assert top.get_signal_names() == raw
    with pytest.raises(AmbiguousSignalError):
        top.load_signal('xx1[0].mid')


def test_subcircuit_path_default_brackets_raw_option_instance_name_token(tmp_path):
    top, (h,) = _build(tmp_path)
    assert h.instance_name == 'xx1<0>'
    assert h.path('zz[1]') == 'xx1[0].zz[1]'
    assert h.path('zz[1]', raw=True) == 'xx1<0>.zz[1]'
    assert repr(h) == "SubcircuitInstance('xx1[0]' of 'xm')"


@needs_ngspice
def test_annotate_currents_report_net_paths_use_brackets(tmp_path):
    top, _ = _build(tmp_path)
    top.simulate()
    rep = top.annotate_currents()
    shown = [n for n, *_ in rep.skips] + list(rep.nets_annotated) + list(rep.partial_nets) \
        + list(rep.worst_kcl_residual)
    assert any('xx1[0]' in s for s in shown)
    assert not any('<' in s for s in shown)
    assert all('<' in k or '[' not in k for k in rep.instance_modules)   # keys stay raw


def test_xyce_gnd_message_shows_brackets(tmp_path):
    from silk.exceptions import CircuitError
    out = str(tmp_path)
    Y = Module(None, out, 'y.cir', None, module_name='ym')
    Y.define_module_ports({'a': PortDir.IN})
    Y.add(elements.resistor('R1', p='a', n='gnd', value=1000))
    top = Module(None, out, 't.cir', None)
    top.add_source(sources.VoltageSource('v1', positive='d', negative='0', dc=1.0))
    top.add_module(Y, 'y[0]', nets={'a': 'd'})
    top.add_sim_type(Transient(step=1e-9, stop=2e-9))
    top.simulator = 'xyce'
    with pytest.raises(CircuitError) as e:
        top.generate_netlist()
    assert 'xy[0]' in str(e.value) and 'xy<0>' not in str(e.value)


def test_behavioral_refusal_shows_brackets(tmp_path):
    from silk.exceptions import SimulationError
    top, _ = _build(tmp_path, simulator='behavioral')
    with pytest.raises((NotSupportedError, SimulationError)) as e:
        top.simulate()
    msg = str(e.value)
    assert 'xx1[0].xxy[1].R1' in msg and 'xx1<0>' not in msg


# ---- viewer defaults ----------------------------------------------------------

def _data_text(html_path):
    base = os.path.splitext(html_path)[0]
    text = open(html_path).read()
    side = base + '.data.js'
    return text + (open(side).read() if os.path.exists(side) else '')


@needs_ngspice
def test_tb_view_labels_are_brackets_and_report_view_too(tmp_path):
    top, _ = _build(tmp_path)
    top.simulate()
    path = top.view(open=False)
    text = _data_text(path)
    assert '"label":"i(vbias[0])"' in text.replace(' ', '') or "i(vbias[0])" in text


@needs_ngspice
def test_cli_list_blocks_and_signals_use_brackets(tmp_path, capsys):
    import silk.signals.view.__main__ as M
    top, _ = _build(tmp_path)
    top.simulate()
    assert M.main([top.results_file, '--list-blocks']) == 0
    out = capsys.readouterr().out
    assert 'xx1[0]' in out and 'xx1<0>' not in out
    assert M.main([top.results_file, '-o', str(tmp_path / 'v.html'), '--signals', 'xx1[0]']) == 0
