"""TASK-530 implementation round 2: collision-safe aliases everywhere, the
alias is never an executable address unless proven, silk.view defaults."""
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from silk import sources
from silk.circuit.modules import Module
from silk.circuit.ports import PortDir
from silk.elements import elements
from silk.signals.rawfile import RawFile
from silk.signals.view import emit_view_data
from silk.simulation import Transient
from tests.test_view_bracket_selectors import _plot, _labels, _names
from tests.test_view_trace_identity import _dump

ROOT = Path(__file__).resolve().parents[1]


# ---- P2-4/5: designer_signal_names and designer labels ---------------------------

def test_protected_entry_is_an_occupied_raw_spelling():
    from silk.spice_names import designer_signal_names
    names = ['v(xx1[0].mid)', 'v(xx1<0>.mid)']
    assert designer_signal_names(names) == names          # skip=1: first protected, second must not alias onto it
    assert designer_signal_names(names, skip=0) == names
    dup = ['time', 'v(xx1<0>.mid)', 'v(xx1<0>.mid)']
    assert designer_signal_names(dup) == dup               # duplicate raw rows stay duplicates, none re-aliased


@pytest.mark.parametrize('order', [0, 1])
def test_designer_labels_keep_collision_groups_case_insensitively(order):
    pair = ['v(XX1<0>.mid)', 'v(xx1[0].mid)']
    names = pair if order == 0 else pair[::-1]
    raw = RawFile([_plot(names + ['v(other<1>.a)'])])
    lab = _labels(emit_view_data(raw, labels='designer'))
    assert lab['v(XX1<0>.mid)'] is None and lab['v(xx1[0].mid)'] is None
    assert lab['v(other<1>.a)'] == 'v(other[1].a)'          # unrelated labels survive


def test_designer_labels_see_the_whole_rawfile_not_the_selection():
    raw = RawFile([_plot(['v(xx1<0>.mid)', 'v(xx1[0].mid)', 'v(out)'])])
    p = emit_view_data(raw, signals=['v(xx1<0>.mid)'], labels='designer')
    assert _labels(p) == {'v(xx1<0>.mid)': None}           # the hidden literal row still collides
    p = emit_view_data(raw, exclude=['v(xx1[0].mid)'], labels='designer')
    assert all(v is None for v in _labels(p).values())
    ac = RawFile([_plot(['v(xx1<0>.a)'], real=False), _plot(['v(xx1[0].a)'])])
    assert all(v is None for pl in (0, 1) for v in _labels(emit_view_data(ac, labels='designer'), pl).values())


def test_explicit_caller_labels_keep_current_behaviour():
    raw = RawFile([_plot(['v(XX1<0>.mid)', 'v(xx1[0].mid)'])])
    p = emit_view_data(raw, labels={'v(XX1<0>.mid)': 'v(XX1[0].mid)'})
    assert _labels(p)['v(XX1<0>.mid)'] == 'v(XX1[0].mid)'


# ---- P2-1: Copy-as-code never turns a display label into an address ----------------

def _sig(name, **extra):
    return {'name': name, 'unit': 'V', 'type': 'voltage', 'part': None, 'group': None, **extra}


def _code(payload):
    return {r['name']: r['code'] for r in _dump(payload)}


def _payload(signals):
    return {'run': {'id': 'r'}, 'plots': [{'name': 'Transient Analysis', 'npoints': 4, 'x': {'name': 'time'},
                                           'signals': signals}]}


def test_arbitrary_labels_are_display_only():
    out = _code(_payload([_sig('v(in)'), _sig('v(out)', label='in')]))
    assert out['v(out)']['code'] == "tb.probe('out')"
    out = _code(_payload([_sig('v(out)', label='v(missing)')]))
    assert out['v(out)']['code'] == "tb.probe('out')"


def test_labels_never_become_addresses_only_the_address_field_does():
    out = _code(_payload([_sig('v(xx1<0>.mid)', label='v(xx1[0].mid)'),
                          _sig('i(vbias<0>)', type='current', label='i(vbias[0])'),
                          _sig('v(xx1<1>.mid)', label='z', address='v(xx1[1].mid)')]))
    assert out['v(xx1<0>.mid)']['code'] == "tb.probe('xx1<0>.mid')"
    assert out['i(vbias<0>)']['code'] == "tb.probe('vbias<0>', 'I')"
    assert out['v(xx1<1>.mid)']['code'] == "tb.probe('xx1[1].mid')"


def test_filtered_out_conflicting_row_never_yields_an_ambiguous_alias():
    raw = RawFile([_plot(['v(xx1<0>.mid)', 'v(xx1[0].mid)'])])
    p = emit_view_data(raw, signals=['v(xx1<0>.mid)'], labels='designer')
    out = _code(p)
    assert out['v(xx1<0>.mid)']['code'] == "tb.probe('xx1<0>.mid')"


def test_ac_and_derived_controls_keep_working():
    raw = RawFile([_plot(['v(xx1<0>.a)'], real=False)])
    p = emit_view_data(raw, labels='designer')
    out = _code(p)
    assert out['v(xx1<0>.a) (mag)']['code'] == "tb.probe('xx1[0].a').mag()"
    derived = [{'plot': 0, 'name': 'd', 'operation': 'abs', 'operands': ['v(xx1<0>.a) (mag)']}]
    rows = {r['name']: r['code'] for r in _dump(p, derived)}
    assert 'xx1[0].a' in rows['d']['code']


# ---- P2-2: annotation report names ------------------------------------------------

def test_report_names_are_collision_safe_and_order_independent():
    from silk.circuit.modules_results import plan_report_net_names
    a = plan_report_net_names(['xx1[0].mid', 'xx1<0>.mid', 'xx1<0>.other', 'd[0]'])
    b = plan_report_net_names(['d[0]', 'xx1<0>.other', 'xx1<0>.mid', 'xx1[0].mid'])
    assert a == b
    assert a['xx1[0].mid'] == 'xx1[0].mid' and a['xx1<0>.mid'] == 'xx1<0>.mid'   # whole group raw
    assert a['xx1<0>.other'] == 'xx1[0].other' and a['d[0]'] == 'd[0]'
    assert len(set(a.values())) == len(a)


def test_report_with_a_collision_keeps_both_rows():
    from silk.circuit.modules_results import CurrentAnnotationReport, finalize_report_names
    r = CurrentAnnotationReport(nets_annotated=['xx1[0].mid', 'xx1<0>.mid', 'xx1<0>.o'],
                                partial_nets=['xx1<0>.mid'],
                                worst_kcl_residual={'xx1[0].mid': -2.0, 'xx1<0>.mid': -3.0, 'xx1<0>.o': 0.5},
                                skips=[('xx1<0>.mid', 'R1.p', 'no-vector')])
    finalize_report_names(r)
    assert r.worst_kcl_residual == {'xx1[0].mid': -2.0, 'xx1<0>.mid': -3.0, 'xx1[0].o': 0.5}
    assert r.nets_annotated == ['xx1[0].mid', 'xx1<0>.mid', 'xx1[0].o']
    assert r.partial_nets == ['xx1<0>.mid'] and r.skips[0][0] == 'xx1<0>.mid'


# ---- P2-3: collision-safe path() ---------------------------------------------------

def _path_design(tmp_path, literal):
    out = str(tmp_path)
    X = Module(None, out, 'x.cir', None, module_name='xm')
    X.define_module_ports({'a': PortDir.IN})
    X.add(elements.resistor('R1', p='a', n='mid', value=1000))
    X.add(elements.resistor('R2', p='mid', n='0', value=1000))
    top = Module(None, out, 't.cir', None)
    top.add_source(sources.VoltageSource('vs', positive='a', negative='0', dc=1.0))
    if literal:
        from tests._legacy_names import add_legacy_net
        add_legacy_net(top, 'xx1[0].mid')
    h = top.add_module(X, 'x1[0]', nets={'a': 'a'})
    top.add_sim_type(Transient(step=1e-9, stop=2e-9))
    return top, h


def test_path_falls_back_to_raw_when_the_design_has_a_colliding_net(tmp_path):
    top, h = _path_design(tmp_path, literal=True)
    assert h.path('mid') == 'xx1<0>.mid' == h.path('mid', raw=True)
    assert h.path('a') == 'a'                                     # mapped port: parent net


def test_path_uses_brackets_when_unique(tmp_path):
    top, h = _path_design(tmp_path, literal=False)
    assert h.path('mid') == 'xx1[0].mid' and h.path('mid', raw=True) == 'xx1<0>.mid'
    assert h.path('a') == 'a'


# ---- P2-5: silk.view default labels ---------------------------------------------------

def test_silk_view_raw_source_defaults_to_designer_labels(tmp_path, monkeypatch):
    import silk
    import silk.signals.view as view
    seen = []

    class R:
        html_path = str(tmp_path / 'v.html')

    monkeypatch.setattr(view, 'emit_view', lambda source, out, **kw: seen.append(kw) or R())
    raw = RawFile([_plot(['v(xx1<0>.a)'])])
    silk.view(raw, out_path=str(tmp_path / 'v.html'), open=False)
    silk.view(str(tmp_path / 'x.raw'), out_path=str(tmp_path / 'v.html'), open=False)
    silk.view(raw, out_path=str(tmp_path / 'v.html'), open=False, labels=None)
    silk.view(raw, out_path=str(tmp_path / 'v.html'), open=False, labels={'v(xx1<0>.a)': 'z'})
    assert [kw.get('labels', 'MISSING') for kw in seen] == ['designer', 'designer', None, {'v(xx1<0>.a)': 'z'}]


def test_silk_view_module_delegation_passes_designer_labels(tmp_path, monkeypatch):
    import silk
    import silk.signals.view as view
    top, _ = _path_design(tmp_path, literal=False)
    top.results_file = str(tmp_path / 'x.raw')
    seen = []

    class R:
        html_path = str(tmp_path / 'v.html')

    monkeypatch.setattr(view, 'emit_view', lambda source, out, **kw: seen.append(kw) or R())
    silk.view(top, out_path=str(tmp_path / 'v.html'), open=False)
    assert seen[0]['labels'] == 'designer'


# ---- P3: claims and docstrings ---------------------------------------------------------

def test_no_false_unique_names_claim_and_no_beside_wording():
    for rel in ('silk/spice_names.py', 'silk/circuit/modules_results.py', 'docs/api-reference.md'):
        text = (ROOT / rel).read_text()
        assert 'never holds two identical names' not in text, rel
    assert 'beside the raw' not in (ROOT / 'silk/signals/view/emit.py').read_text()


# ---- P2-6: selector contract fixture ------------------------------------------

def test_selector_cases_fixture_matches_the_python_matcher():
    from silk.signals.view.select import select_for_payload
    cases = json.loads((ROOT / 'tests/fixtures/selector_cases.json').read_text())
    assert cases['version'] == 1 and cases['cases']
    covered = {c['kind'] for c in cases['cases']}
    assert {'role-block', 'role-branch', 'wrapped', 'bare', 'union', 'exclude', 'literal-leaf',
            'unknown-form', 'xyce-wrapped-colon', 'char-class'} <= covered
    for c in cases['cases']:
        names = c.get('names') or cases['names']
        try:
            mask = select_for_payload([names], signals=c.get('signals'), exclude=c.get('exclude'))[0]
            got = [n for n, k in zip(names, mask) if k]
        except Exception as exc:                     # a refusal is a recorded outcome
            # Record the concrete class: a selection failure is its typed
            # subclass (TASK-615), so the fixture pins the subclass, not ViewError.
            got = {'refuses': type(exc).__name__}
        assert got == c['expect'], c['id']
