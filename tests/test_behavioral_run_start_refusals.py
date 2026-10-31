"""TASK-448.4: transient grid mapping and run-start refusals of the
'behavioral' backend. Each refused item has one test; the message names it."""
import numpy as np
import pytest

from silk.exceptions import NotSupportedError
from silk.simulation import AC, Noise, Op, Transient
from tests.test_behavioral_backend_seam import _flat_top


@pytest.mark.parametrize('analysis, name', [
    (AC(variation='dec', points=10, fstart=1, fstop=1e6), 'AC'),
    (Noise('vout', 'gnd', 'vs', 'dec', 10, 1, 1e6), 'Noise'),
    (Op(), 'Op'),
])
def test_refused_analysis_names_itself(tmp_path, analysis, name):
    top = _flat_top(tmp_path, analysis=analysis)
    with pytest.raises(NotSupportedError, match=f'declares: {name}'):
        top.simulate()


def test_ambient_temperature_refused(tmp_path):
    top = _flat_top(tmp_path)
    top.set_temperature(85)
    with pytest.raises(NotSupportedError, match='ambient TEMP'):
        top.simulate()


def test_custom_spice_text_refused(tmp_path):
    top = _flat_top(tmp_path)
    top.add_custom_line('.param foo=1')
    with pytest.raises(NotSupportedError, match='add_custom_line'):
        top.simulate()


@pytest.mark.parametrize('flag', ['save_device_currents', 'save_device_params'])
def test_device_switches_refused(tmp_path, flag):
    top = _flat_top(tmp_path)
    setattr(top, flag, True)
    with pytest.raises(NotSupportedError, match=flag):
        top.simulate()


def test_grid_includes_stop(tmp_path):
    top = _flat_top(tmp_path, analysis=Transient(step=1e-9, stop=10e-9))
    top.simulate()
    t = top.load_signal('vout').x
    assert len(t) == 11
    assert t[0] == 0 and t[-1] == pytest.approx(10e-9)


def test_grid_with_stop_not_a_multiple_still_ends_at_stop(tmp_path):
    top = _flat_top(tmp_path, analysis=Transient(step=3e-9, stop=10e-9))
    top.simulate()
    t = top.load_signal('vout').x
    assert t[-1] == pytest.approx(10e-9)
    assert len(t) == 5


def test_start_drops_earlier_samples(tmp_path):
    top = _flat_top(tmp_path, analysis=Transient(step=1e-9, stop=10e-9, start=4e-9))
    top.simulate()
    sig = top.load_signal('vout')
    assert sig.x[0] == pytest.approx(4e-9)
    assert sig.x[-1] == pytest.approx(10e-9)
    assert len(sig.x) == len(sig.y) == 7


def test_maxstep_caps_the_step(tmp_path):
    top = _flat_top(tmp_path, analysis=Transient(step=2e-9, stop=10e-9, maxstep=1e-9))
    top.simulate()
    assert len(top.load_signal('vout').x) == 11
