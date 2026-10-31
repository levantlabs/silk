"""Regressions for truncated serializer transients (TASK-540, TASK-557)."""
import pytest

from silk.examples import serializer_example as serializer
from tests._env import needs_ngspice, needs_pdk


@pytest.mark.slow
@needs_ngspice
@needs_pdk
@pytest.mark.parametrize('variant,words,signal,stop', [
    ('cmos', 5, 'sout', 15e-9),
    ('cml', 4, 'soutp', 12.5e-9),
])
def test_4gbps_reaches_requested_stop(tmp_path, monkeypatch, variant, words, signal, stop):
    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    simulated = []
    simulate = serializer.Module.simulate

    def capture_simulation(module, *args, **kwargs):
        result = simulate(module, *args, **kwargs)
        simulated.append(module)
        return result

    monkeypatch.setattr(serializer.Module, 'simulate', capture_simulation)
    bits = serializer.sources.prbs_bits(9, 80)
    if variant == 'cmos':
        serializer._run_cmos(bits, tbit=250e-12, n_words=words)
    else:
        serializer._run_cml(bits, tbit=250e-12, n_words=words, probe='dp')

    # Two trailing clocks follow the finite word sequence (no extra LOAD).
    # A partial rawfile previously ended at 1.9125 ns, before any clock edge.
    time = simulated[0].load_signal(signal).x
    assert float(time[-1]) == pytest.approx(stop, rel=1e-9, abs=1e-15)
    load = 'load' if variant == 'cmos' else 'loadp'
    low = 0 if variant == 'cmos' else 0.6
    trailing_load = serializer.sample(simulated[0].load_signal(load),
                                      x_new=[stop - 500e-12])
    assert float(trailing_load.y[0]) == pytest.approx(low)
