"""TASK-530: in every Python-facing selector, square brackets are an array
index (never a glob character class); [k] selects the same vector as <k>."""
import numpy as np
import pytest

from silk.signals.rawfile import RawFile, RawPlot, RawVector
from silk.signals.view import emit_view_data
from silk.signals.view.emit import ViewError
from silk.signals.view.select import matches_selector, select_for_payload

NAMES = [
    'v(d[0])', 'v(d<0>)', 'i(vbias<0>)', 'i(vbias<1>)', 'v(out)', 'v(xdut.out)',
    'i(vb)', 'i(xdut.vb)',
    'v(xx1<0>.xxy<1>.g[0])', 'v(xx1<0>.xxy<1>.g<0>)', 'v(xx1<1>.xxy<1>.g[0])',
    'v(xx1<0>.xxy<1>.g[1])',
]


def _sel(sel, names=NAMES):
    return [n for n in names if matches_selector(n, sel)]


def test_bare_block_selects_every_vector_under_the_copy():
    assert set(_sel('xx1[0]')) == {'v(xx1<0>.xxy<1>.g[0])', 'v(xx1<0>.xxy<1>.g<0>)',
                                   'v(xx1<0>.xxy<1>.g[1])'}
    assert _sel('xx1[0]') == _sel('xx1<0>')


def test_two_component_block():
    assert set(_sel('xx1[0].xxy[1]')) == set(_sel('xx1<0>.xxy<1>'))
    assert len(_sel('xx1[0].xxy[1]')) == 3


def test_bare_branch_leaf_is_an_instance_position():
    assert _sel('vbias[0]') == ['i(vbias<0>)']
    assert _sel('vbias[0]') == _sel('vbias<0>')


def test_i_wrapper_is_full_name_only():
    assert _sel('i(vbias[0])') == ['i(vbias<0>)']
    assert _sel('i(vb)') == ['i(vb)']                 # not i(xdut.vb), as today


def test_v_wrapper_full_name_only_positive_negative_and_exclude():
    assert _sel('v(out)') == ['v(out)']
    assert 'v(xdut.out)' in _sel('out') and 'v(out)' in _sel('out')   # bare keeps nesting
    masks = select_for_payload([NAMES], exclude=['v(out)'])[0]
    kept = [n for n, k in zip(NAMES, masks) if k]
    assert 'v(out)' not in kept and 'v(xdut.out)' in kept
    masks = select_for_payload([NAMES], exclude=['i(vb)'])[0]
    kept = [n for n, k in zip(NAMES, masks) if k]
    assert 'i(vb)' not in kept and 'i(xdut.vb)' in kept


def test_nested_leaf_g0_is_literal_and_never_reaches_g_angle():
    assert 'v(xx1<0>.xxy<1>.g[0])' in _sel('g[0]')
    assert 'v(xx1<0>.xxy<1>.g<0>)' not in _sel('g[0]')
    assert _sel('g<0>') == ['v(xx1<0>.xxy<1>.g<0>)']
    assert 'v(d<0>)' not in _sel('d[0]') and 'v(d[0])' in _sel('d[0]')
    assert _sel('v(d[0])') == ['v(d[0])']


def test_full_path_bracket_equals_angle_spelling():
    assert _sel('xx1[0].xxy[1].g[0]') == ['v(xx1<0>.xxy<1>.g[0])']
    assert _sel('xx1<0>.xxy<1>.g[0]') == ['v(xx1<0>.xxy<1>.g[0])']
    assert _sel('v(xx1[0].xxy[1].g[0])') == ['v(xx1<0>.xxy<1>.g[0])']


def test_index_wildcards():
    assert set(_sel('xx1[*].xxy[1].g[0]')) == {'v(xx1<0>.xxy<1>.g[0])', 'v(xx1<1>.xxy<1>.g[0])'}
    assert set(_sel('vbias[*]')) == {'i(vbias<0>)', 'i(vbias<1>)'}
    assert set(_sel('d[*]')) == {'v(d[0])'}              # literal leaf, never d<0>
    assert set(_sel('g[*]')) >= {'v(xx1<0>.xxy<1>.g[1])'}
    assert 'v(xx1<0>.xxy<1>.g<0>)' not in _sel('g[*]')


def test_brackets_are_never_a_character_class():
    names = ['v(a0)', 'v(a1)', 'v(a[01])']
    assert _sel('v(a[01])', names) == ['v(a[01])']
    assert _sel('v(a[0-9])', names) == []
    assert _sel('v(a[!0])', names) == []


def test_unrecognised_selector_has_no_instance_reading():
    names = ['v(xx1<0>&foo)', 'v(xx1<0>.x)', 'v(a\nb)']
    assert _sel('xx1[0]&foo', names) == []
    assert _sel('xx1[0]\nfoo', names) == []


def test_whole_name_glob_crosses_separators():
    assert 'v(xx1<0>.xxy<1>.g[0])' in _sel('v(xx1*g[0])')


def test_union_two_distinct_rows_and_exclude_removes_both():
    names = ['v(a[0])', 'i(va<0>)', 'v(zz)']
    got = [n for n in names if matches_selector(n, 'a[0]') or matches_selector(n, 'va[0]')]
    assert got == ['v(a[0])', 'i(va<0>)']
    masks = select_for_payload([names], signals=['a[0]', 'va[0]'])[0]
    assert masks == [True, True, False]
    masks = select_for_payload([names], exclude=['a[0]', 'va[0]'])[0]
    assert masks == [False, False, True]


def test_every_selector_credited_and_exclude_checks_whole_inventory():
    select_for_payload([NAMES], signals=['*', 'v(out)'])
    with pytest.raises(ViewError):
        select_for_payload([NAMES], signals=['v(out)'], exclude=['xx1[7]'])
    select_for_payload([NAMES], signals=['v(out)'], exclude=['xx1[0]'])   # dropped rows still count


def test_refusal_suggests_bracket_spelling():
    with pytest.raises(ViewError) as e:
        select_for_payload([NAMES], signals=['xx1[0].zz[1].g[0]'])
    msg = str(e.value)
    assert 'xx1[0]' in msg and 'xx1<0>' not in msg


def test_block_label_helper_and_save_inventory_suggestion():
    from silk.signals.view.select import block_label
    assert block_label('xx1<0>.xxy<1>') == 'xx1[0].xxy[1]'
    with pytest.raises(ViewError) as e:
        select_for_payload([['xx1<0>.xxy<1>.g[0]']], signals=['xx1[0].xxy[1].h[0]'],
                           where="in this design's save inventory")
    assert '<' not in str(e.value).split('Closest', 1)[1]


def _plot(names, real=True):
    n = 4
    x = RawVector('time', 'time', data=np.arange(1, n + 1, dtype=float))
    vecs = [x] + [RawVector(nm, 'voltage', data=np.arange(n, dtype=float) * (1 if real else 1 + 1j) + i)
                  for i, nm in enumerate(names)]
    return RawPlot({'plotname': 'Transient Analysis', 'title': 't'}, vecs, npoints=n, real=real, padded=True)


def _names(payload, plot=0):
    return [s['name'] for s in payload['plots'][plot]['signals']]


def test_emit_view_data_signals_and_exclude_accept_brackets():
    raw = RawFile([_plot(['v(d[0])', 'i(vbias<0>)', 'v(xx1<0>.mid)'])])
    assert _names(emit_view_data(raw, signals=['xx1[0].mid'])) == ['v(xx1<0>.mid)']
    assert _names(emit_view_data(raw, exclude=['xx1[0].mid', 'vbias[0]'])) == ['v(d[0])']


# ---- labels='designer' ------------------------------------------------------

def _labels(p, plot=0):
    return {s['name']: s.get('label') for s in p['plots'][plot]['signals']}


def test_labels_designer_builds_bracket_labels_only_where_they_differ():
    raw = RawFile([_plot(['v(d[0])', 'i(vbias<0>)', 'v(xx1<0>.mid)', 'v(out)'])])
    lab = _labels(emit_view_data(raw, labels='designer'))
    assert lab == {'v(d[0])': None, 'i(vbias<0>)': 'i(vbias[0])',
                   'v(xx1<0>.mid)': 'v(xx1[0].mid)', 'v(out)': None}


def test_labels_designer_without_arrays_is_byte_identical():
    raw = RawFile([_plot(['v(a)', 'v(b)'])])
    a, b = emit_view_data(raw), emit_view_data(raw, labels='designer')
    assert a['run']['id'] == b['run']['id']
    assert a == b


def test_labels_designer_uses_selected_plan():
    raw = RawFile([_plot(['v(xx1<0>.mid)', 'v(out)'])])
    p = emit_view_data(raw, signals=['v(out)'], labels='designer')       # filtered-out array: no error
    assert _names(p) == ['v(out)']
    p = emit_view_data(raw, exclude=['xx1[0]'], labels='designer')
    assert _labels(p) == {'v(out)': None}
    p = emit_view_data(raw, max_depth=0, labels='designer')
    assert _labels(p) == {'v(out)': None}
    with pytest.raises(ViewError):                                      # explicit dicts keep validation
        emit_view_data(raw, signals=['v(out)'], labels={'v(xx1<0>.mid)': 'v(xx1[0].mid)'})


def test_labels_designer_multi_plot_and_ac():
    raw = RawFile([_plot(['v(xx1<0>.a)']), _plot(['v(xx1<0>.a)', 'v(b)'], real=False)])
    p = emit_view_data(raw, labels='designer')
    assert _labels(p, 0) == {'v(xx1<0>.a)': 'v(xx1[0].a)'}
    ac = _labels(p, 1)
    assert ac['v(xx1<0>.a) (mag)'] == 'v(xx1[0].a) (mag)'
    assert ac['v(xx1<0>.a) (phase)'] == 'v(xx1[0].a) (phase)'


def test_labels_designer_collision_falls_back_and_keeps_run_id():
    raw = RawFile([_plot(['v(xx1<0>.a)', 'v(xx1[0].a)'])])
    p = emit_view_data(raw, labels='designer')
    lab = _labels(p)
    assert lab['v(xx1[0].a)'] is None and lab['v(xx1<0>.a)'] is None
    assert p['run']['id'] == emit_view_data(raw)['run']['id']


def test_labels_designer_changes_run_id_not_state_key():
    raw = RawFile([_plot(['v(xx1<0>.a)'])])
    a, b = emit_view_data(raw), emit_view_data(raw, labels='designer')
    assert a['run']['id'] != b['run']['id']
    assert a['run']['state_key'] == b['run']['state_key']


def test_labels_rejects_unknown_string():
    raw = RawFile([_plot(['v(a)'])])
    with pytest.raises(ViewError):
        emit_view_data(raw, labels='nope')


@pytest.mark.parametrize('selector', [
    'xx1[0].xxy[1].g[0]', 'xx1<0>.xxy<1>.g[0]',
    'xx1[0]', 'xx1<0>', 'xxy[1]', 'g[0]', 'g<0>',
    'xx1[*].xxy[1].g[0]', 'g[*]',
])
def test_xyce_hierarchy_selection_matches_ngspice(selector):
    xyce = [n.replace('.', ':').upper() for n in NAMES]
    expected = [matches_selector(n, selector) for n in NAMES]
    assert any(expected)
    assert [matches_selector(n, selector) for n in xyce] == expected
    for keyword in ('signals', 'exclude'):
        options = {keyword: [selector]}
        assert select_for_payload([xyce], **options) == select_for_payload([NAMES], **options)
        payload = emit_view_data(RawFile([_plot(xyce)]), **options)
        keep = expected if keyword == 'signals' else [not x for x in expected]
        assert _names(payload) == [n for n, yes in zip(xyce, keep) if yes]


def test_xyce_wrapped_selectors_remain_full_name_only():
    name = 'V(XX1<0>:XXY<1>:G[0])'
    assert matches_selector(name, 'v(xx1[0]:xxy[1]:g[0])')
    assert not matches_selector(name, 'v(g[0])')
    assert not matches_selector(name, 'v(xx1[0].xxy[1].g[0])')


@pytest.mark.slow
@pytest.mark.requires_xyce
def test_real_ngspice_and_xyce_hierarchy_selectors(tmp_path):
    import shutil
    import subprocess
    from silk import settings
    from silk.signals.rawfile import read_rawfile

    binaries = [settings.get('SILK_NGSPICE_BIN'), settings.get('SILK_XYCE_BIN')]
    if not all(binary and shutil.which(binary) for binary in binaries):
        pytest.skip('needs configured ngspice and Xyce')
    body = """selector parity
Vsrc in 0 1
XX1<0> in outer
XX1<1> in outer
.subckt outer in
XXY<1> in inner
.ends outer
.subckt inner in
R1 in g[0] 1k
R2 g[0] 0 1k
.ends inner
.tran 1n 4n
"""
    payloads = []
    for backend, binary in zip(('ngspice', 'xyce'), binaries):
        folder = tmp_path / backend
        folder.mkdir()
        raw = folder / 'result.raw'
        output = ('.control\nrun\nwrite result.raw all\n.endc\n' if backend == 'ngspice'
                  else '.print tran format=raw file=result.raw V(*) I(Vsrc)\n')
        deck = folder / 'test.cir'
        deck.write_text(body + output + '.end\n')
        result = subprocess.run([binary, *(['-b'] if backend == 'ngspice' else []), str(deck)],
                                cwd=folder, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        payloads.append(read_rawfile(raw))
    for selector, count in [('xx1[0].xxy[1].g[0]', 1),
                            ('xx1<0>.xxy<1>.g[0]', 1),
                            ('xx1[0]', 1), ('xx1<0>', 1), ('g[0]', 2)]:
        for keyword in ('signals', 'exclude'):
            selected = [_names(emit_view_data(raw, **{keyword: [selector]})) for raw in payloads]
            normalized = [[n.lower().replace(':', '.') for n in names] for names in selected]
            assert set(normalized[0]) == set(normalized[1])
            if keyword == 'signals':
                assert len(selected[0]) == count
