"""Selector callbacks whose shared state exposes unsafe coverage checks."""
from types import MethodType

from silk.sources.waveforms import Sine


def install_selector(source, override, callback):
    class CountingSine(Sine):
        def to_spice(self):
            self.renderer_calls += 1
            return super().to_spice()

    wave = CountingSine(0, 1, 1e3) if callback == 'rendering' else Sine(0, 1, 1e3)
    wave.selector_calls = wave.renderer_calls = 0
    source.sine = wave

    def select(self):
        self.sine.selector_calls += 1
        if callback == 'raising':
            raise RuntimeError('custom selector must not be called')
        if callback == 'rendering':
            self.sine.to_spice()
        return self.sine

    if override == 'class':
        source.__class__ = type('CustomSource', (type(source),), {'selected_waveform': select})
    elif override == 'instance':
        source.selected_waveform = MethodType(select, source)
    else:
        source.__class__ = type('InheritedSource', (type(source),), {})
    return wave
