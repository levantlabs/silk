"""TASK-527: labels= on emit_view_data (raw name stays the key)."""
import json

import numpy as np
import pytest

from silk.signals.rawfile import RawFile, RawPlot, RawVector
from silk.signals.view import emit_view_data
from silk.signals.view.emit import ViewError


def _plot(names, real=True, plotname='Transient Analysis'):
    n = 4
    x = RawVector('time' if real else 'frequency', 'time' if real else 'frequency',
                  data=np.arange(1, n + 1, dtype=float))
    vecs = [x]
    for i, name in enumerate(names):
        data = np.arange(n, dtype=float) + i
        if not real:
            data = data * (1 + 1j)
        vecs.append(RawVector(name, 'voltage', data=data))
    return RawPlot({'plotname': plotname, 'title': 't'}, vecs, npoints=n, real=real, padded=True)


def _signals(payload, plot=0):
    return {s['name']: s for s in payload['plots'][plot]['signals']}


def _two_plots():
    return RawFile([_plot(['v(a)', 'v(b)']), _plot(['v(a)', 'v(c)'], plotname='AC')])


def test_labels_field_and_raw_keys():
    raw = RawFile([_plot(['v(xx1<0>.mid)', 'v(out)'])])
    p = emit_view_data(raw, labels={'v(xx1<0>.mid)': 'v(xx1[0].mid)'})
    sigs = _signals(p)
    assert sigs['v(xx1<0>.mid)']['label'] == 'v(xx1[0].mid)'
    assert 'label' not in sigs['v(out)']
    assert 'v(xx1[0].mid)' not in sigs


def test_annotation_by_raw_name_still_validates():
    raw = RawFile([_plot(['v(xx1<0>.mid)'])])
    emit_view_data(raw, labels={'v(xx1<0>.mid)': 'v(xx1[0].mid)'},
                   annotations={'markers': [{'signal': 'v(xx1<0>.mid)', 'axis': 'x', 'value': 2.0}]})


def test_no_labels_payload_has_no_label_keys_and_hash_unchanged():
    raw = RawFile([_plot(['v(a)'])])
    base = emit_view_data(raw)
    for labels in (None, {}):
        p = emit_view_data(raw, labels=labels)
        assert p['run']['id'] == base['run']['id']
        assert 'label' not in json.dumps(p['plots'][0]['signals'][0].keys().__repr__())
    assert 'label' not in base['plots'][0]['signals'][0]


def test_run_id_changes_state_key_stable():
    raw = RawFile([_plot(['v(xx1<0>.mid)'])])
    a = emit_view_data(raw)
    b = emit_view_data(raw, labels={'v(xx1<0>.mid)': 'v(xx1[0].mid)'})
    c = emit_view_data(raw, labels={'v(xx1<0>.mid)': 'v(xx1[0].MID)'})
    assert len({a['run']['id'], b['run']['id'], c['run']['id']}) == 3
    assert a['run']['state_key'] == b['run']['state_key'] == c['run']['state_key']


def test_bad_label_key_refused_with_close_matches():
    raw = RawFile([_plot(['v(xx1<0>.mid)'])])
    with pytest.raises(ViewError, match=r'v\(xx1<0>\.mid\)'):
        emit_view_data(raw, labels={'v(xx1<0>.mdi)': 'x'})
    with pytest.raises(ViewError):
        emit_view_data(raw, labels={'v(xx1<0>.mid)': 3})


def test_label_collides_with_label_dropped_for_all():
    raw = RawFile([_plot(['v(a)', 'v(b)'])])
    p = emit_view_data(raw, labels={'v(a)': 'L', 'v(b)': 'L'})
    assert all('label' not in s for s in p['plots'][0]['signals'])


def test_label_collides_with_unlabelled_raw_name():
    raw = RawFile([_plot(['v(a)', 'v(b)'])])
    p = emit_view_data(raw, labels={'v(a)': 'v(b)'})
    assert all('label' not in s for s in p['plots'][0]['signals'])


def test_fallback_cascade_until_stable():
    # a->b collides with unlabelled b? no: b is labelled to c, c unlabelled.
    # a:'v(b)' and b:'v(c)'. a's label equals raw b's name but b has its own label,
    # so a is fine; b's label equals raw c -> dropped, which frees nothing for a.
    # Dropping b's label makes b display 'v(b)', which now collides with a's 'v(b)'.
    raw = RawFile([_plot(['v(a)', 'v(b)', 'v(c)'])])
    p = emit_view_data(raw, labels={'v(a)': 'v(b)', 'v(b)': 'v(c)'})
    assert all('label' not in s for s in p['plots'][0]['signals'])


def test_mag_phase_expansion_collision_and_suffix():
    raw = RawFile([_plot(['v(a)', 'v(b)'], real=False, plotname='AC')])
    p = emit_view_data(raw, labels={'v(a)': 'Q'})
    sigs = _signals(p)
    assert sigs['v(a) (mag)']['label'] == 'Q (mag)'
    assert sigs['v(a) (phase)']['label'] == 'Q (phase)'
    assert sigs['v(a) (mag)']['group'] == 'v(a)'
    p2 = emit_view_data(raw, labels={'v(a)': 'Q', 'v(b)': 'Q'})
    assert all('label' not in s for s in p2['plots'][0]['signals'])


def test_labels_per_plot_collision_is_independent():
    p = emit_view_data(_two_plots(), labels={'v(a)': 'L', 'v(b)': 'L'})
    assert all('label' not in s for s in p['plots'][0]['signals'])
    assert _signals(p, 1)['v(a)']['label'] == 'L'


def test_never_changes_signal_name_or_group():
    raw = RawFile([_plot(['v(a)'], real=False, plotname='AC')])
    base = emit_view_data(raw)
    p = emit_view_data(raw, labels={'v(a)': 'Q'})
    for x, y in zip(base['plots'][0]['signals'], p['plots'][0]['signals']):
        assert (x['name'], x['group']) == (y['name'], y['group'])


def test_emit_view_accepts_labels(tmp_path):
    from silk.signals.view import emit_view
    raw = RawFile([_plot(['v(xx1<0>.mid)'])])
    out = emit_view(raw, tmp_path / 'v.html', inline=True,
                    labels={'v(xx1<0>.mid)': 'v(xx1[0].mid)'})
    assert 'v(xx1[0].mid)' in (tmp_path / 'v.html').read_text()


def test_view_imports_only_the_codec_from_silk():
    from pathlib import Path
    import silk.signals.view.emit as emit
    text = Path(emit.__file__).read_text()
    assert 'silk.circuit' not in text and 'silk.simulation' not in text
