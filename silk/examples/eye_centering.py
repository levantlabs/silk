"""Waveform-only NRZ/PAM4 eye alignment; no PDK or simulator required.

Run with ``MPLBACKEND=Agg .venv/bin/python -m silk.examples.eye_centering``.
Saves before/after plots under ``silk-output/eye_centering`` (or the directory
selected by SILK_EXAMPLES_OUT). This demonstrates display alignment only.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from silk.examples import example_output_dir
from silk.metrics import eye_diagram
from silk.signals import Signal


UI = 1e-9


def rc_waveform(levels: tuple[float, ...]) -> Signal:
    """Generate a seeded 300-symbol stimulus with 0.1-UI edges and RC tau."""
    symbols = np.random.default_rng(11).choice(levels, 300)
    samples_per_ui = 100
    phase = np.arange(len(symbols) * samples_per_ui + 1) / samples_per_ui
    symbol_index = np.minimum(phase.astype(int), len(symbols) - 1)
    previous = symbols[np.maximum(symbol_index - 1, 0)]
    ramp = np.clip((phase - symbol_index) / .1, 0., 1.)
    source = previous + ramp * (symbols[symbol_index] - previous)
    t = phase * UI
    tau = .1 * UI
    y = np.empty_like(source)
    y[0] = source[0]

    # Integrate each linear source segment exactly at its endpoint.
    for i, dt in enumerate(np.diff(t), start=1):
        decay = np.exp(-dt / tau)
        slope = (source[i] - source[i - 1]) / dt
        y[i] = (decay * y[i - 1] + (1 - decay) * source[i - 1]
                + slope * (dt - tau * (1 - decay)))
    return Signal('rx', 'voltage', 'time', 'V', t, y)


def main() -> None:
    """Save two reproducible before/after figures using only the public fold."""
    output = Path(example_output_dir('eye_centering'))
    for name, levels in [('NRZ', (0., 1.8)), ('PAM4', (0., .6, 1.2, 1.8))]:
        sig = rc_waveform(levels)
        before = eye_diagram(sig, period=UI)
        after = eye_diagram(sig, period=UI, auto_center=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True,
                                 constrained_layout=True)
        for ax, result, title in zip(axes, [before, after], ['Before', 'After']):
            for trace in result.traces:
                Signal('eye', 'voltage', 'Time within window (UI)', ax.get_ylabel(),
                       trace.x / UI, trace.y).plot(ax=ax, color='steelblue', lw=.6, alpha=.14)
            ax.axvline(.5, color='0.4', lw=.8, ls='--')
            ax.set(xlim=(0, 1), xlabel='Time within window (UI)', title=title)
            ax.grid(alpha=.15)
        axes[0].set_ylabel('Voltage (V)')
        fig.suptitle(f'{name}: average-voltage crossing alignment')
        path = output / f'{name.lower()}.png'
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f'{name}: centre = {after.sampling_instant_s / UI:.6f} UI; {path}')


if __name__ == '__main__':
    main()
