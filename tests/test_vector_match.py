"""TASK-530 round 3: one shared stdlib matcher (silk.vector_match) behind the
reader and the view layer's addresses, with a Python/JS shared case file."""
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from silk.signals.rawfile import RawFile, RawPlot, RawVector
from silk.signals.view import emit_view_data

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / 'tests/fixtures/vector_match_cases.json').read_text())


def _stub(name):
    p = CASES['plots'][name]
    return SimpleNamespace(plotname=p['plotname'], real=p['real'], npoints=p['npoints'],
                           vectors=[SimpleNamespace(name=n) for n in p['vectors']],
                           nvars=len(p['vectors']))


def test_module_is_a_stdlib_plus_spice_names_leaf():
    from silk import vector_match
    tree = ast.parse(Path(vector_match.__file__).read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, 'no relative imports'
            mods.add(node.module)
    assert all(m.split('.')[0] in sys.stdlib_module_names or m == 'silk.spice_names' for m in mods), mods


def test_reader_and_view_share_the_one_matcher():
    from silk import vector_match
    from silk.signals import spice
    assert spice.collect_vector_candidates is vector_match.collect_vector_candidates


@pytest.mark.parametrize('case', CASES['first_row'], ids=lambda c: c['plot'])
def test_first_row_matches_fixed_expectation_and_the_readers_rule(case):
    from silk.signals.analysis_kind import ANALYSIS_OP, rawfile_analysis_kind
    from silk.vector_match import first_row
    plot = _stub(case['plot'])
    assert first_row(plot) == case['expect']
    assert (first_row(plot) == 0) == (rawfile_analysis_kind(plot) == ANALYSIS_OP)


@pytest.mark.parametrize('case', CASES['candidates'], ids=lambda c: f"{c['plot']}:{c['name']}:{c['type']}")
def test_candidates_are_the_fixed_row_indices(case):
    from silk.vector_match import collect_vector_candidates, first_row
    plot = _stub(case['plot'])
    assert collect_vector_candidates(plot, case['name'], case['type'], first=first_row(plot)) == case['expect']


@pytest.mark.parametrize('case', CASES['address'], ids=lambda c: f"{c['plot']}:{c['row']}")
def test_address_is_attached_only_when_the_reader_resolves_exactly_that_row(case):
    from silk.vector_match import vector_address
    assert vector_address(_stub(case['plot']), case['row']) == case['expect']


def _rawplot(name, names, real=True, plotname='Transient Analysis', npoints=4):
    cols = [RawVector(n, 'voltage', data=np.arange(npoints, dtype=float) * (1 if real else 1 + 1j) + i)
            for i, n in enumerate(names)]
    return RawPlot({'plotname': plotname, 'title': 't'}, cols, npoints=npoints, real=real, padded=True)


def _addr(payload, plot=0):
    return {s['name']: s.get('address') for s in payload['plots'][plot]['signals']}


def _raw(*plots):
    return RawFile(list(plots))


def test_designer_emit_attaches_addresses_from_the_whole_source_plot():
    raw = _raw(_rawplot(0, ['time', 'v(xx1<0>.mid)', 'xx1[0].mid']))
    sel = emit_view_data(raw, signals=['v(xx1<0>.mid)'], labels='designer')
    assert _addr(sel) == {'v(xx1<0>.mid)': None}                 # hidden bare row still collides
    raw = _raw(_rawplot(0, ['time', 'v(xx1<0>.mid)', 'v(out)']))
    assert _addr(emit_view_data(raw, labels='designer')) == {'v(xx1<0>.mid)': 'v(xx1[0].mid)', 'v(out)': None}
    assert _addr(emit_view_data(raw, signals=['v(xx1<0>.mid)'], labels='designer')) == {'v(xx1<0>.mid)': 'v(xx1[0].mid)'}


def test_duplicate_rows_get_no_address_and_plots_are_proven_separately():
    dup = _raw(_rawplot(0, ['time', 'v(a<0>.x)', 'v(a<0>.x)']))
    assert set(_addr(emit_view_data(dup, labels='designer')).values()) <= {None}
    two = _raw(_rawplot(0, ['time', 'v(xx1<0>.mid)'], plotname='Transient Analysis'),
               _rawplot(1, ['time', 'v(xx1<0>.mid)', 'xx1[0].mid'], plotname='DC transfer characteristic', npoints=5))
    p = emit_view_data(two, labels='designer')
    assert _addr(p, 0) == {'v(xx1<0>.mid)': 'v(xx1[0].mid)'}
    assert _addr(p, 1)['v(xx1<0>.mid)'] is None


def test_ac_parts_share_the_vector_address():
    raw = _raw(_rawplot(0, ['frequency', 'v(xx1<0>.a)'], real=False, plotname='AC Analysis'))
    a = _addr(emit_view_data(raw, labels='designer'))
    assert a == {'v(xx1<0>.a) (mag)': 'v(xx1[0].a)', 'v(xx1<0>.a) (phase)': 'v(xx1[0].a)'}


def test_op_plot_counts_row_zero():
    raw = _raw(_rawplot(0, ['v(xx1<0>.a)', 'xx1[0].a'], plotname='Operating Point', npoints=1))
    assert set(_addr(emit_view_data(raw, labels='designer')).values()) <= {None}      # row 1 reads as a second candidate


def test_explicit_dicts_and_no_labels_never_carry_addresses_and_hashes_hold():
    raw = _raw(_rawplot(0, ['time', 'v(xx1<0>.mid)', 'v(out)']))
    base = emit_view_data(raw)
    assert all('address' not in s for s in base['plots'][0]['signals'])
    explicit = emit_view_data(raw, labels={'v(xx1<0>.mid)': 'v(xx1[0].mid)'})
    assert all('address' not in s for s in explicit['plots'][0]['signals'])
    auto = emit_view_data(raw, labels='designer')
    assert auto['run']['id'] != explicit['run']['id']             # an address exists only in auto mode
    assert auto['run']['state_key'] == explicit['run']['state_key'] == base['run']['state_key']


def test_auto_equals_identical_explicit_when_no_address_can_be_proved():
    names = ['v(xx1<0>.mid)', 'v(xx1[0]:mid)']          # distinct alias strings, two reader candidates
    raw = _raw(_rawplot(0, ['time'] + names))
    auto = emit_view_data(raw, labels='designer')
    sigs = auto['plots'][0]['signals']
    shown = {s['name']: s['label'] for s in sigs if 'label' in s}
    assert shown == {'v(xx1<0>.mid)': 'v(xx1[0].mid)'}     # the label IS shown ...
    assert all('address' not in s for s in sigs)            # ... but no address can be proved
    explicit = emit_view_data(raw, labels=shown)
    assert auto['run']['id'] == explicit['run']['id']       # no mode-only hash difference
    assert auto['plots'] == explicit['plots']
    # trivial controls: nothing shown, nothing proved
    only = _raw(_rawplot(0, ['time', 'v(xx1<0>.mid)', 'v(xx1[0].mid)']))
    assert emit_view_data(only, labels='designer')['run']['id'] == emit_view_data(only, labels={})['run']['id']
    # a proved address does change the id against the explicit dict of the same labels
    ok = _raw(_rawplot(0, ['time', 'v(xx1<0>.mid)']))
    a = emit_view_data(ok, labels='designer')
    assert a['run']['id'] != emit_view_data(ok, labels={'v(xx1<0>.mid)': 'v(xx1[0].mid)'})['run']['id']


def test_old_payload_without_address_keeps_raw_copy_as_code():
    from tests.test_view_trace_identity import _dump
    payload = emit_view_data(_raw(_rawplot(0, ['time', 'v(xx1<0>.mid)'])), labels={'v(xx1<0>.mid)': 'v(xx1[0].mid)'})
    codes = {r['name']: r['code']['code'] for r in _dump(payload)}
    assert codes['v(xx1<0>.mid)'] == "tb.probe('xx1<0>.mid')"


def test_copy_as_code_uses_the_address_through_the_probe_formatter():
    from tests.test_view_trace_identity import _dump
    raw = _raw(_rawplot(0, ['time', 'v(xx1<0>.mid)', 'i(vbias<0>)']),)
    payload = emit_view_data(raw, labels='designer')
    codes = {r['name']: r['code']['code'] for r in _dump(payload)}
    assert codes['v(xx1<0>.mid)'] == "tb.probe('xx1[0].mid')"
    assert codes['i(vbias<0>)'] == "tb.probe('vbias[0]', 'I')"
    ac = emit_view_data(_raw(_rawplot(0, ['frequency', 'v(xx1<0>.a)'], real=False, plotname='AC Analysis')), labels='designer')
    codes = {r['name']: r['code']['code'] for r in _dump(ac)}
    assert codes['v(xx1<0>.a) (mag)'] == "tb.probe('xx1[0].a').mag()"
    # hidden collision: the filtered payload keeps the raw copy
    hidden = emit_view_data(_raw(_rawplot(0, ['time', 'v(xx1<0>.mid)', 'xx1[0].mid'])),
                            signals=['v(xx1<0>.mid)'], labels='designer')
    codes = {r['name']: r['code']['code'] for r in _dump(hidden)}
    assert codes['v(xx1<0>.mid)'] == "tb.probe('xx1<0>.mid')"


def test_node_vector_match_contract():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node >=20 is required for the JavaScript vector-match contract')
    result = subprocess.run(
        [node, '--test', str(ROOT / 'tests/js/vector_match.test.mjs')],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
