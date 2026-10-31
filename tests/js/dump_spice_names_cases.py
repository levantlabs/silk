"""Dump Python codec and label-plan results for the JS parity test.

.venv/bin/python tests/js/dump_spice_names_cases.py --output /tmp/cases.json
node tests/js/spice_names_parity.mjs --cases /tmp/cases.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from silk.signals.rawfile import RawFile, RawPlot, RawVector, write_rawfile
from silk.signals.view import emit_view_data
from silk.spice_names import to_designer_name, to_spice_name

NAMES = ['xx1[0].xy[1].g[0]', 'xx1<0>.xy<1>.g[0]', 'g<0>', 'g[0]', 'v(xdut.xx1<0>.mid)',
         'V(vvs[0])', 'I(vvs[0])', 'i(v.xdut.xx1[0].vint)', '@m.xdut.xx1<0>.xmm1.m0[gm]',
         'i(@m.x[0].m0[id])', 'V(XDUT:XX1<0>:MID)', 'N(XDUT:XX1<0>:XMM1:GM)',
         'ID(XDUT:XX1<0>:XMM1)', 'Xdut.XX1<0>.Mid', 'v(a<0>)-v(b<1>)', 'a[0] + b[1]', '',
         'onoise_spectrum', 'vvs[0]', 'vvs<0>',
         'I(XA<0>:VB<1>)', 'xdut.xx1<0>.mid', 'XDUT:XX1<0>:MID', 'xx1<0>.mid?xx2<0>.a:b',
         'xx1<0>.m1#body', 'v(xdut.xx1<0>.m1#body)', 'xx1<0>.mid:xx2<0>.a', 'N(XA<0>.XM1.GM)',
         'ID(XA<0>:XM1)', '@m.xa<0>:xm1.m0[gm]', 'x[-1].a', 'x<-1>.a', 'temp-sweep',
         'i(v.xa<0>.vint#branch)', 'xa<0>.mid',
         'xé<0>.mid', 'xé[0].mid', 'x中<0>.mid', 'x中[0].mid', 'x²<0>.mid', 'x²[0].mid',
         'i(vé<0>)', 'i(vé[0])', 'xe\u0301<0>.mid', 'x\u00a0<0>.mid', 'x\u0085<0>.a',
         'xx1<0>.mid&xx2<0>.mid', 'xx1<0>.mid%2', 'xx1<0>.mid^2', '!xx1<0>.mid', 'xx1<0>.mid\u0080',
         'xx1<0>.m\u008did', 'xx1<0>.mid\u009f', 'a$b-c_d<0>.e9']


LISTED_SPACES = ['\u00a0', '\u1680', *map(chr, range(0x2000, 0x200b)), '\u2028', '\u2029',
                 '\u202f', '\u205f', '\u3000', '\ufeff']
BOUNDARY_CHARS = [chr(c) for c in range(0x100)] + LISTED_SPACES
ENDINGS = ['\n', '\r', '\r\n', '\u2003', '\u3000']
WRAPPED = ['xx1<0>.mid', 'v(xx1<0>.mid)', 'i(vvs<0>)', 'I(XA<0>:VB<1>)', 'N(XA<0>:XM1:GM)',
           'ID(XA<0>:XM1)', '@m.xx1<0>.m0[gm]', 'i(@m.xx1<0>.m0[id])', 'XDUT:XX1<0>:MID',
           'vvs<0>']
EXTRA = [f'{w}{e}' for w in WRAPPED for e in ENDINGS] + [
    'xx1<-1>.mid', 'a$b<0>.c', 'a_b<0>.c', 'xx1[0].mid', 'xé<0>.mid', 'x中<0>.mid', 'x²<0>.mid',
    'xe\u0301<0>.mid', 'xx1<0>.mid`', 'xx1<0>.mid~', 'xx1<0>.mid&xx2<0>.mid']
for _c in BOUNDARY_CHARS:
    EXTRA += [f'{_c}xx1<0>.mid', f'xx1<0>{_c}.mid', f'xx1<0>.mid{_c}']

GLOB_NAMES = ['xx1[*].g[0]', 'xx1[?].xxy[1].g[*]', 'd[*]', 'v(xx1[*].g[0])', 'i(vbias[*])',
              'vbias[*]', 'xx1[*]&foo', 'xx1[*]\n.g', 'xx1[*].g\n', '*', 'v(*)', 'xé[*].mid',
              'xx1[*]\u00a0.g', 'V(XX1[*]:G[0])', '@m.xx1[*].m0[gm]']


def _plot(names, real=True, name='Transient Analysis'):
    x = RawVector('time' if real else 'frequency', 'time' if real else 'frequency',
                  data=np.arange(1, 5, dtype=float))
    vecs = [x]
    for i, n in enumerate(names):
        d = np.arange(4, dtype=float) + i
        vecs.append(RawVector(n, 'voltage', data=d if real else d * (1 + 1j)))
    return RawPlot({'plotname': name, 'title': 't'}, vecs, npoints=4, real=real, padded=True)


LABEL_CASES = [
    ([(['v(a)', 'v(b)'], True)], {'v(a)': 'L', 'v(b)': 'L'}),
    ([(['v(a)', 'v(b)'], True), (['v(a)', 'v(c)'], True)], {'v(a)': 'L', 'v(c)': 'L'}),
    ([(['v(a)', 'v(b)', 'v(c)'], True)], {'v(a)': 'v(b)', 'v(b)': 'v(c)'}),
    ([(['v(xx1<0>.m)', 'v(o)'], True)], {'v(xx1<0>.m)': 'v(xx1[0].m)'}),
    ([(['v(a)', 'v(b)'], False)], {'v(a)': 'Q'}),
    ([(['v(a)', 'v(a)'], True)], {'v(a)': 'Q'}),
]

# designer mode: (plots, ) -> labels AND proven addresses must agree
DESIGNER_CASES = [
    [(['v(xx1<0>.mid)', 'v(out)'], True)],
    [(['v(xx1<0>.mid)', 'xx1[0].mid'], True)],
    [(['v(xx1<0>.mid)', 'v(xx1[0]:mid)'], True)],
    [(['V(XX1<0>:MID)', 'v(xx1[0].mid)'], True)],
    [(['v(XX1<0>.MID)'], True)],
    [(['v(a<0>.x)', 'v(a<0>.x)'], True)],
    [(['i(vbias<0>)', 'v(vbias<0>)'], True)],
    [(['v(xx1<0>.a)'], False)],
    [(['v(xx1<0>.mid)'], True), (['v(xx1<0>.mid)', 'xx1[0].mid'], True)],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    out = Path(ap.parse_args().output)
    out.parent.mkdir(parents=True, exist_ok=True)
    codec = []
    for n in NAMES + EXTRA:
        for kind in ('node', 'branch'):
            for style in ('auto', 'ngspice', 'xyce', 'behavioral'):
                codec.append({'name': n, 'kind': kind, 'style': style,
                              'spice': to_spice_name(n, style=style, kind=kind),
                              'designer': to_designer_name(n, style=style, kind=kind)})
    for n in GLOB_NAMES:
        for kind in ('node', 'branch'):
            codec.append({'name': n, 'kind': kind, 'style': 'auto', 'allow_glob': True,
                          'spice': to_spice_name(n, kind=kind, allow_glob=True),
                          'designer': to_designer_name(n, kind=kind)})
    labels = []
    for k, (plots, mapping) in enumerate(LABEL_CASES):
        path = out.parent / f'labels_{k}.raw'
        write_rawfile(path, RawFile([_plot(n, r, f'P{i}') for i, (n, r) in enumerate(plots)]), binary=False)
        payload = emit_view_data(str(path), labels=mapping)
        labels.append({'raw': path.name, 'labels': mapping,
                       'shown': [[[s['name'], s.get('label')] for s in p['signals']] for p in payload['plots']]})
    designer = []
    for k, plots in enumerate(DESIGNER_CASES):
        path = out.parent / f'designer_{k}.raw'
        write_rawfile(path, RawFile([_plot(n, r, f'P{i}') for i, (n, r) in enumerate(plots)]), binary=False)
        payload = emit_view_data(str(path), labels='designer')
        designer.append({'raw': path.name,
                         'shown': [[[s['name'], s.get('label'), s.get('address')] for s in p['signals']]
                                   for p in payload['plots']]})
    out.write_text(json.dumps({'codec': codec, 'labels': labels, 'designer': designer}))


if __name__ == '__main__':
    main()
