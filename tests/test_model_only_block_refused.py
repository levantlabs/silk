"""TASK-448.8: a SPICE deck never instantiates a block that has a behavioural
model and no SPICE content. A twin (model plus SPICE content) is unaffected."""
import pytest

from silk.exceptions import NotSupportedError
from silk.examples import behavioral_example as be


def test_model_only_block_refused_at_generate_netlist():
    top = be.build_top()
    top.module_name = None   # a top-level deck, not a .subckt
    with pytest.raises(NotSupportedError, match=r"comp1.*simulator = 'behavioral'"):
        top.generate_netlist()


def test_twin_block_with_spice_content_is_not_refused():
    from silk import sources
    comp = be.build_comparator()
    comp.add_source(sources.VoltageSource(
        'vbias', positive='vref', negative='gnd', dc=0.9))
    top = be._make_bare_module('top')
    top.module_name = None
    top.add_module(comp, 'comp1',
                   nets={'vin': 'vin', 'vref': 'vref', 'vout': 'vcomp'})
    top.generate_netlist()   # must not raise


def _model_only_bench(tmp_path):
    import numpy as np
    from silk import PortDir, sources
    from silk.circuit.modules import Module
    from silk.pdk.basepdk import BasePDK
    from silk.simulation import analysis
    import os

    stub_lib = tmp_path / 'stub.lib.spice'
    stub_lib.write_text('* stub\n')
    import silk
    yaml = os.path.join(os.path.dirname(silk.__file__), 'pdk', 'data',
                        'skywater_130_device_details.yaml')
    pdk = BasePDK(yaml, library_location=str(stub_lib))
    dut = Module('PDK_skywater130', str(tmp_path), 'g.cir', pdk_details=pdk,
                 module_name='gain')
    dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT})
    def model_1(ports, t, params):
        ports.b.v = 2 * ports.a.v
    dut.set_behavioral_model(model_1, execution='vector')
    tb = dut.create_test('g')
    tb.add_module_under_test({'a': 'vin', 'b': 'vout'})
    tb.add_source(sources.VoltageSource('vs', positive='vin', negative='0', dc=1.0))
    tb.add_sim_type(analysis.Transient(step=1e-8, stop=1e-6))
    return tb


def test_testbench_run_refuses_before_launch(tmp_path, monkeypatch):
    marker = tmp_path / 'launched'
    stub = tmp_path / 'ngspice_stub.sh'
    stub.write_text(f'#!/bin/sh\ntouch {marker}\nexit 1\n')
    stub.chmod(0o755)
    monkeypatch.setenv('SILK_NGSPICE_BIN', str(stub))
    tb = _model_only_bench(tmp_path)
    with pytest.raises(NotSupportedError, match="simulator = 'behavioral'"):
        tb.run()
    assert not marker.exists()
