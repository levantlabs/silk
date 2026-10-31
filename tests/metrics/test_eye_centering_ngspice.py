"""Center real adaptive-timestep RC transients, including ISI and phase wrap."""

import math
import warnings

import numpy as np
import pytest

from silk import Module, load_pdk, sources
from silk.circuit.ports import PortDir
from silk.elements import elements
from silk.metrics import crossings, eye_diagram
from silk.signals.base import Signal
from silk.simulation import Transient
from tests._env import needs_ngspice


@needs_ngspice
@pytest.mark.parametrize('levels', [(0., 1.8), (0., .6, 1.2, 1.8)],
                         ids=['NRZ', 'PAM4'])
@pytest.mark.parametrize('tau_ui', [.1, .3])
def test_real_rc_crossing_alignment_at_every_origin(tmp_path, levels, tau_ui, record_property):
    """Regenerate the approved stimulus; no cached /tmp waveforms or band oracle.

    Cropping changes the time average: check each full supplied record against
    independent public crossing extraction. Ideal R/C need no PDK models.
    """
    ui, count = 1e-9, 300
    symbols = np.random.default_rng(11).choice(levels, count)
    points = [(0., symbols[0])]
    for i in range(1, count):
        points.extend([(i * ui - 1e-15, symbols[i - 1]),
                       ((i + .1) * ui, symbols[i])])
    points.append((count * ui, symbols[-1]))
    dut = Module('sky130', str(tmp_path), 'rc.cir', load_pdk('sky130'),
                 module_name='rc')
    dut.define_module_ports({'in': PortDir.IN, 'out': PortDir.OUT,
                             'vss': PortDir.GROUND})
    dut.add(elements.resistor('r1', p='in', n='out', value=3e3))
    dut.add(elements.capacitor('c1', p='out', n='vss',
                              value=tau_ui * ui / 3e3))
    tb = dut.create_test('eye')
    tb.add_module_under_test({'in': 'vin', 'out': 'vout', 'vss': '0'})
    tb.add_source(sources.VoltageSource('vs', 'vin', '0', dc=0, pwl=points))
    tb.add_sim_type(Transient(step=ui / 50, stop=count * ui))
    observed = {}

    def capture(ckt):
        observed['output'] = ckt.load_signal('vout')
        return 0.

    tb.add_metric('capture', capture)
    tb.run()
    output = observed['output']
    assert np.ptp(np.diff(output.x)) > ui / 100  # actually adaptive SPICE steps
    centres = []
    for origin in (0., .23, .50, .77):
        keep = output.x >= output.x[0] + origin * ui
        sig = Signal('rx', 'voltage', 'time', 'V', output.x[keep], output.y[keep])
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = eye_diagram(sig, period=ui, auto_center=True)
        # Scalar compensated integration and the existing public event extractor
        # are independent of the private eye helper. No exact-threshold vertices
        # occur in these simulator fixtures (analytic tests cover those rules).
        average = math.fsum(float(dt) * (float(a) + float(b)) / 2
                            for dt, a, b in zip(np.diff(sig.x), sig.y[:-1], sig.y[1:]))
        average /= float(sig.x[-1] - sig.x[0])
        assert not np.any(sig.y == average)
        events = crossings(sig, threshold=average, edge='any')
        angles = 2 * math.pi * ((events - sig.x[0]) / ui % 1)
        expected_angle = math.atan2(math.fsum(map(math.sin, angles)),
                                   math.fsum(map(math.cos, angles)))
        expected = (expected_angle / (2 * math.pi) + .5) % 1
        phase = result.sampling_instant_s
        assert abs((phase / ui - expected + .5) % 1 - .5) < 2e-12
        assert np.isfinite(phase), (levels, tau_ui, origin)
        centres.append(((sig.x[0] + phase) / ui) % 1.)
        shift = (phase - ui / 2) % ui
        for k, trace in enumerate(result.traces):
            start = float(sig.x[0]) + shift + k * ui
            end = float(sig.x[0]) + shift + (k + 1) * ui
            take = (sig.x >= start) & (sig.x < end)
            assert trace.y.tobytes() == sig.y[take].tobytes()
            np.testing.assert_array_equal(trace.x, sig.x[take] - start)
    record_property('absolute_centres_ui', centres)
    if tau_ui == .1:
        # Owner-reviewed recorded-waveform observations, with tolerance for a
        # fresh simulator run rather than requiring the temporary artifact.
        reference = .623878283641 if len(levels) == 2 else .628795868380
        assert abs((centres[0] - reference + .5) % 1 - .5) < 5e-4
