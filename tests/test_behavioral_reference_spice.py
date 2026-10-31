"""Small real-SPICE comparisons of supported voltage-reference equations."""
import numpy as np
import pytest

from silk.circuit.modules import Module
from silk.circuit.circuit_elements import PortDir
from silk.simulation import DC, Transient
from silk.sources import VoltageSource
from tests._env import needs_ngspice
from tests.test_behavioral_rawresult import _pdk

pytestmark = [pytest.mark.slow, needs_ngspice]


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('hierarchy', [False, True])
@pytest.mark.parametrize('analysis', ['tran', 'dc'])
def test_reference_equation_matches_ngspice(tmp_path, reverse, hierarchy, analysis):
    top = Module('sky130', str(tmp_path), 'references.cir', pdk_details=_pdk())
    target = top
    if hierarchy:
        target = Module('sky130', str(tmp_path / 'offset'), 'offset.cir',
                        pdk_details=_pdk(), module_name='offset')
        target.define_module_ports({'out': PortDir.OUT, 'ref': PortDir.IN})
    target.add_source(VoltageSource(
        'offset', positive='ref' if reverse else 'out',
        negative='out' if reverse else 'ref', dc=-.2 if reverse else .2,
        pwl=[(0, -.3 if reverse else .3), (1, -.8 if reverse else .8)]))
    if hierarchy:
        top.add_module(target, 'shift', nets={'out': 'out', 'ref': 'ref'})
    top.add_source(VoltageSource('reference', positive='ref', negative='0', dc=1))
    top.add_sim_type(Transient(step=.25, stop=1) if analysis == 'tran'
                     else DC('reference', 0, 1, .5))
    top.simulator = 'behavioral'
    top.simulate()
    behavioral = top.load_signal('out')
    x, y = np.array(behavioral.x), np.array(behavioral.y)
    np.testing.assert_allclose(y, [1.5, 1.625, 1.75, 1.875, 2]
                               if analysis == 'tran' else [.5, 1, 1.5])
    top.simulator = 'ngspice'
    top.simulate()
    spice = top.load_signal('out')
    np.testing.assert_allclose(np.interp(x, spice.x, spice.y), y,
                               rtol=1e-8, atol=1e-9)
