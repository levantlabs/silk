"""Successful simulator doubles derive their result grids from the launched deck."""
import math
from decimal import Decimal

import numpy as np

from silk.signals.rawfile import RawPlot, RawVector, read_rawfile, write_rawfile
from silk.utils._utils import _parse_si


def analysis_grids(deck, simulator='ngspice'):
    """Return built-in top-level analysis grids, independently of the gate."""
    depth = 0
    grids = []
    for line in deck.splitlines()[1:]:
        fields = line.lower().split()
        if not fields:
            continue
        card = fields[0]
        if card == '.subckt':
            depth += 1
        elif card == '.ends':
            depth -= 1
        if depth:
            continue
        if card == '.end':
            break
        if card == '.tran':
            grids.append(('tran', np.array([0., _parse_si(fields[2])])))
        elif card == '.dc':
            start, stop, step = map(_parse_si, fields[2:5])
            quotient = (Decimal(str(stop)) - Decimal(str(start))) / Decimal(str(step))
            count = int(quotient) + 1
            grids.append(('dc', start + np.arange(count) * step))
        elif card in ('.ac', '.noise'):
            offset = 1 if card == '.ac' else 3
            mode = fields[offset]
            points = int(fields[offset + 1])
            start, stop = map(_parse_si, fields[offset + 2:offset + 4])
            if mode == 'lin':
                grid = np.linspace(start, stop, points)
            else:
                base = 10 if mode == 'dec' else 2
                steps = math.floor(points * math.log(stop / start, base) + 1e-9)
                if card == '.ac' and mode == 'dec' and simulator == 'ngspice':
                    grid = np.geomspace(start, stop, steps + 1)
                else:
                    grid = start * base ** (np.arange(steps + 1) / points)
            grids.append((card[1:], grid))
        elif card == '.op':
            grids.append(('op', np.array([0.])))
    # Bare external-deck tests exercise data presence, without an analysis request.
    return grids or [('tran', np.array([0., 1e-9]))]


def write_complete_rawfile(path, deck, *, simulator='ngspice', template=None,
                           values=None):
    """Write complete mock plots, retaining template signal samples if supplied."""
    source = read_rawfile(template).plots[0] if template else None
    plots = []
    names = {'tran': 'Transient Analysis', 'dc': 'DC transfer characteristic',
             'ac': 'AC Analysis', 'noise': 'Noise Spectral Density',
             'op': 'Operating Point'}
    for kind, grid in analysis_grids(deck, simulator):
        if source is not None and kind == 'tran':
            old = source.vectors[0].data.real
            grid = (old - old[0]) / (old[-1] - old[0]) * grid[-1]
        axis = 'time' if kind == 'tran' else 'frequency' if kind in ('ac', 'noise') else 'v-sweep'
        vectors = [RawVector(axis, axis, (), grid.astype(complex) if kind == 'ac' else grid)]
        signals = values
        if signals is None and source is not None:
            signals = {v.name: v.data for v in source.vectors[1:]}
        if signals is None:
            signals = ({'onoise_spectrum': 1e-10, 'inoise_spectrum': 2.5e-13}
                       if kind == 'noise' else {'v(out)': 0.})
        for name, samples in signals.items():
            samples = np.asarray(samples)
            if samples.ndim == 0:
                samples = np.full(len(grid), samples)
            elif kind == 'op':
                samples = np.array([samples[-1]])
            elif len(samples) != len(grid):
                samples = np.interp(np.linspace(0, 1, len(grid)),
                                    np.linspace(0, 1, len(samples)), samples)
            if kind == 'ac':
                samples = samples.astype(complex)
            vectors.append(RawVector(name, 'current' if name.lower().startswith('i(') else 'voltage', (), samples))
        real = kind != 'ac'
        plots.append(RawPlot({'title': 'mock', 'date': 'mock', 'plotname': names[kind]},
                             vectors, len(grid), real, None))
    write_rawfile(path, plots, binary=False)


def complete_noise_csv(deck, simulator='xyce'):
    grid = next((g for kind, g in analysis_grids(deck, simulator) if kind == 'noise'),
                np.array([1., 10.]))
    position = np.linspace(0, 1, len(grid))
    onoise = np.interp(position, [0, 1], [4e-10, 1e-10])
    inoise = np.interp(position, [0, 1], [1e-12, 2.5e-13])
    return '"FREQ","ONOISE","INOISE"\n' + ''.join(
        f'{freq:.17g},{out:.17g},{inp:.17g}\n'
        for freq, out, inp in zip(grid, onoise, inoise))
