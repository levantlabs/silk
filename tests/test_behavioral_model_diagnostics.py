"""Actionable failures for dictionary access on ports-form attachments."""
import pytest

from silk import PortDir
from silk.circuit.modules import Module
from silk.exceptions import SimulationError
from silk.simulation import DC
from silk.simulation.behavioral import BehavioralEngine
from silk.sources import VoltageSource


@pytest.mark.parametrize('execution', ['default', 'vector', 'sample'])
@pytest.mark.parametrize('dc', [False, True])
@pytest.mark.parametrize('operation', ['read', 'write', 'return_dict', 'return_scalar'])
def test_dictionary_model_errors_teach_electrical_port_access(tmp_path, execution, dc, operation):
    def model(ports, t, params):
        if operation == 'read':
            ports['VIN']
        elif operation == 'write':
            ports['VOUT'] = 1
        elif operation == 'return_dict':
            return {'VOUT': ports.VIN.v}
        else:
            return 1

    dut = Module(None, str(tmp_path / 'dut'), 'dut.cir', pdk_details=None, module_name='gain')
    dut.define_module_ports({'VIN': PortDir.IN, 'VOUT': PortDir.OUT})
    kwargs = {} if execution == 'default' else {'execution': execution}
    dut.set_behavioral_model(model, params={}, **kwargs)
    top = Module(None, str(tmp_path), 'top.cir', pdk_details=None)
    top.add_module(dut, 'gain', nets={'VIN': 'in', 'VOUT': 'out'})
    top.add_source(VoltageSource('drive', positive='in', negative='gnd', dc=1))
    engine = BehavioralEngine()
    with pytest.raises(SimulationError) as caught:
        if dc:
            engine.run_dc(top, DC('drive', 0, 1, 1))
        else:
            engine.run(top, 1, 2)
    message = str(caught.value)
    assert 'gain' in message
    if operation in ('read', 'write'):
        assert ('ports.VIN.v' if operation == 'read' else 'ports.VOUT.v') in message
    else:
        assert 'return None' in message
        assert 'assign' in message
        assert 'ports.' in message and '.v' in message


def test_two_argument_callable_has_ports_contract_diagnostic(tmp_path):
    def model(inputs, t):
        return {'VOUT': inputs['VIN']}

    dut = Module(None, str(tmp_path / 'dut'), 'dut.cir', None, module_name='gain')
    dut.define_module_ports({'VIN': PortDir.IN, 'VOUT': PortDir.OUT})
    dut.set_behavioral_model(model)
    top = Module(None, str(tmp_path), 'top.cir', None)
    top.add_module(dut, 'gain', nets={'VIN': 'in', 'VOUT': 'out'})
    with pytest.raises(SimulationError, match=r'gain.*fn\(ports, t, params\).*ports.VIN.v'):
        BehavioralEngine().run(top, 1, 2)


def test_model_body_type_error_keeps_its_type_and_message(tmp_path):
    def model(ports, t, params):
        raise TypeError('invalid calibration payload')

    dut = Module(None, str(tmp_path / 'dut'), 'dut.cir', None, module_name='gain')
    dut.define_module_ports({'VIN': PortDir.IN, 'VOUT': PortDir.OUT})
    dut.set_behavioral_model(model)
    top = Module(None, str(tmp_path), 'top.cir', None)
    top.add_module(dut, 'gain', nets={'VIN': 'in', 'VOUT': 'out'})
    with pytest.raises(TypeError, match='^invalid calibration payload$'):
        BehavioralEngine().run(top, 1, 2)
