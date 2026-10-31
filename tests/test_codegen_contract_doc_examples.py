"""Runs the JSON examples in docs/codegen-contract.md's sources[]/
ideal_elements[]/ctrl_vname sections (TASK-282.3) through Generator.generate,
so the doc cannot silently drift from the generator's actual behavior.

Only the sections this task touches are exercised here; every other section
of the contract (voltage_domains, ports, bus_nets, submodules, devices,
waveforms, pwl, simulation, testbench specs) already has dedicated coverage
in tests/test_codegen_domains_buses.py, tests/test_codegen_ports.py,
tests/test_codegen_submodules.py, tests/test_codegen_arrays.py,
tests/test_codegen_sources.py and tests/test_testbench_generator*.py.
"""

import os
from pathlib import Path
import subprocess

import pytest

from silk.codegen.generator import Generator
from silk.signals.rawfile import read_rawfile
from silk.simulation.runner import ngspice_executable
from tests._env import needs_ngspice, needs_pdk


def _spec(sources=None, ideal_elements=None, circuit_name='tb'):
    return {
        'circuit_name': circuit_name, 'pdk': 'sky130',
        'ideal_elements': ideal_elements or [],
        'sources': sources or [],
    }


class TestIdealElementsExamples:
    def test_resistor_example(self):
        # docs/codegen-contract.md "ideal_elements" worked example.
        spec = _spec(ideal_elements=[
            {"type": "resistor", "instance": "r_load",
             "nets": {"p": "vout", "n": "gnd"}, "params": {"value": 10000}},
        ])
        code = Generator().generate(spec)
        assert "ckt.add(elements.resistor('r_load', p='vout', n='gnd', value=10000))" in code


class TestControlledSourceExamples:
    def test_vcvs_example(self):
        # docs/codegen-contract.md dependent-sources VCVS example.
        spec = _spec(sources=[
            {"type": "VCVS", "instance": "e1", "out_pos": "vout", "out_neg": "gnd",
             "ctrl_pos": "vin", "ctrl_neg": "gnd", "gain": 10},
        ])
        code = Generator().generate(spec)
        assert ("ckt.add(sources.VoltageControlledVoltageSource('e1', "
                "out_pos='vout', out_neg='gnd', ctrl_pos='vin', "
                "ctrl_neg='gnd', gain=10))") in code

    def test_cccs_example_with_its_ctrl_vname_target(self):
        # docs/codegen-contract.md dependent-sources CCCS example,
        # paired with the "VoltageSource placed purely to carry the sensed
        # current" ctrl_vname target the prose describes.
        spec = _spec(sources=[
            {"type": "VoltageSource", "instance": "vmeas",
             "positive": "a", "negative": "m", "dc": 0},
            {"type": "CCCS", "instance": "f1", "out_pos": "icopy", "out_neg": "gnd",
             "gain": 2, "ctrl_vname": "vmeas"},
        ])
        code = Generator().generate(spec)
        assert ("ckt.add(sources.CurrentControlledCurrentSource('f1', "
                "out_pos='icopy', out_neg='gnd', ctrl_vname='vmeas', gain=2))") in code

    def test_ctrl_vname_ineligible_target_error_text(self):
        # docs/codegen-contract.md ctrl_vname eligibility table's worked
        # ValueError example: a CCCS sensing a CurrentSource (no branch
        # current). The exact message text is pinned in the doc.
        spec = _spec(sources=[
            {"type": "CurrentSource", "instance": "ibias",
             "positive": "vdd", "negative": "gnd", "dc": 0.001},
            {"type": "CCCS", "instance": "f1", "out_pos": "icopy", "out_neg": "gnd",
             "gain": 2, "ctrl_vname": "ibias"},
        ])
        with pytest.raises(ValueError) as exc:
            Generator().generate(spec)
        assert str(exc.value) == (
            "sources entry 'f1' (CurrentControlledCurrentSource): "
            "ctrl_vname='ibias' targets source 'ibias' (CurrentSource), "
            "which has no voltage-source branch current to sense. Choose a "
            "voltage source (V, E, H or B V=), or insert a zero-volt "
            "ammeter source in the current path."
        )

    def test_ctrl_vname_eligible_behavioral_voltage_target_generates(self):
        # docs/codegen-contract.md ctrl_vname worked example: a CCVS
        # sensing a BehavioralVoltage ammeter.
        spec = _spec(sources=[
            {"type": "BehavioralVoltage", "instance": "iprobe",
             "positive": "isense", "negative": "gnd", "expression": "0"},
            {"type": "CurrentSource", "instance": "idrive",
             "positive": "vdd", "negative": "isense", "dc": 0.001},
            {"type": "CCVS", "instance": "h1", "out_pos": "vout", "out_neg": "gnd",
             "transresistance": 1000, "ctrl_vname": "iprobe"},
        ])
        code = Generator().generate(spec)
        assert "ckt.add(sources.BehavioralVoltage('iprobe'" in code
        assert "ckt.add(sources.CurrentControlledVoltageSource('h1'" in code

    @staticmethod
    def _ctrl_vname_worked_example_spec(output_dir):
        # docs/codegen-contract.md's ctrl_vname worked example (CCVS
        # sensing a BehavioralVoltage ammeter), shared by the deck-text and
        # ngspice-execution checks below.
        spec = _spec(sources=[
            {"type": "BehavioralVoltage", "instance": "iprobe",
             "positive": "isense", "negative": "gnd", "expression": "0"},
            {"type": "CurrentSource", "instance": "idrive",
             "positive": "vdd", "negative": "isense", "dc": 0.001},
            {"type": "CCVS", "instance": "h1", "out_pos": "vout", "out_neg": "gnd",
             "transresistance": 1000, "ctrl_vname": "iprobe"},
        ])
        spec['output_dir'] = str(output_dir)
        return spec

    @needs_pdk
    def test_ctrl_vname_resolves_to_the_documented_h_card(self, tmp_path):
        # TASK-282.4, reviewer-requested addition: the doc's "ctrl_vname at
        # deck emission" subsection asserts the exact emitted H card
        # (Hh1 vout gnd Biprobe 1000) — ctrl_vname='iprobe' (the instance
        # name) resolves to iprobe's own SPICE element name (Biprobe), not
        # emitted verbatim.
        spec = self._ctrl_vname_worked_example_spec(tmp_path)
        code = Generator().generate(spec)
        code = code.replace("ckt.simulate(corner='tt')",
                            "deck_path = ckt.generate_netlist(corner='tt')")
        ns: dict = {}
        exec(code, ns)
        ckt = ns['ckt']
        deck_path = ns['deck_path']
        with open(deck_path) as fh:
            deck_text = fh.read()
        assert 'Hh1 vout gnd Biprobe 1000' in deck_text

    @pytest.mark.slow
    @needs_pdk
    @needs_ngspice
    def test_ctrl_vname_eligible_behavioral_voltage_target_simulates_1v(self, tmp_path):
        # Same spec as above, proving the doc's numeric claim (transresistance
        # 1000 * 1 mA = 1.0 V) is literally true, not just asserted in prose.
        spec = self._ctrl_vname_worked_example_spec(tmp_path)
        code = Generator().generate(spec)
        code = code.replace("ckt.simulate(corner='tt')",
                            "deck_path = ckt.generate_netlist(corner='tt')")
        ns: dict = {}
        exec(code, ns)
        ckt = ns['ckt']
        deck_path = ns['deck_path']
        with open(deck_path) as fh:
            deck_text = fh.read()
        assert 'Hh1 vout gnd Biprobe 1000' in deck_text
        with open(deck_path, 'w') as fh:
            fh.write(deck_text.replace('.control', '.op\n.control', 1))
        raw_path = Path(deck_path).with_name('rawfile.raw')
        subprocess.run([ngspice_executable(), '-b', deck_path],
                       check=True, capture_output=True, timeout=30)
        vectors = {v.name.lower(): v.data for v in read_rawfile(raw_path).vectors}
        assert vectors['v(vout)'] == pytest.approx([1.0], abs=1e-9)


class TestBehavioralSourceExample:
    def test_behavioral_voltage_field_table_example(self):
        # docs/codegen-contract.md Behavioral sources worked example.
        spec = _spec(sources=[
            {"type": "BehavioralVoltage", "instance": "bv1",
             "positive": "vout", "negative": "gnd", "expression": "v(vin) * 2"},
        ])
        code = Generator().generate(spec)
        assert ("ckt.add(sources.BehavioralVoltage('bv1', positive='vout', "
                "negative='gnd', expression='v(vin) * 2'))") in code
