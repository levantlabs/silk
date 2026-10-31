"""Current controls must expose a voltage-source branch current (TASK-281)."""

import hashlib
import os
from pathlib import Path
import subprocess

import pytest

from silk import sources
from silk.circuit.modules import Module
from silk.codegen.generator import CircuitSpec, Generator
from silk.signals.rawfile import read_rawfile
from silk.simulation.runner import ngspice_executable
from silk.sources.schema import export_source_schema_json
from tests._env import needs_ngspice, needs_pdk


# Independent expectations for every exported source class. The source under
# test sits between sense and ground; a separate source provides its control.
_TARGETS = [
    (sources.VoltageSource, True,
     dict(positive='sense', negative='0', dc=0)),
    (sources.CurrentSource, False,
     dict(positive='sense', negative='0', dc=0.001)),
    (sources.VoltageControlledVoltageSource, True,
     dict(out_pos='sense', out_neg='0', ctrl_pos='seed', ctrl_neg='0', gain=1)),
    (sources.VoltageControlledCurrentSource, False,
     dict(out_pos='sense', out_neg='0', ctrl_pos='seed', ctrl_neg='0', gm=0.001)),
    (sources.CurrentControlledVoltageSource, True,
     dict(out_pos='sense', out_neg='0', ctrl_vname='seed', transresistance=1000)),
    (sources.CurrentControlledCurrentSource, False,
     dict(out_pos='sense', out_neg='0', ctrl_vname='seed', gain=1)),
    (sources.BehavioralVoltage, True,
     dict(positive='sense', negative='0', expression='0')),
    (sources.BehavioralCurrent, False,
     dict(positive='sense', negative='0', expression='0.001')),
]


def _spec(target_type, target_params, consumer_type='CCVS'):
    scalar = ({'transresistance': 1000} if consumer_type in
              ('CCVS', 'CurrentControlledVoltageSource') else {'gain': 1})
    return {
        'circuit_name': 'branch_current',
        'sources': [
            # Forward references must work as well as earlier declarations.
            dict(type=consumer_type, instance='readout', out_pos='out',
                 out_neg='0', ctrl_vname='target', **scalar),
            dict(type=target_type, instance='target', **target_params),
            dict(type='VoltageSource', instance='seed', positive='seed',
                 negative='0', dc=0),
        ],
    }


def test_target_cases_cover_every_exported_source_class():
    exported = {getattr(sources, name) for name in sources.__all__
                if isinstance(getattr(sources, name), type)
                and issubclass(getattr(sources, name), sources.SourceBase)
                and getattr(sources, name) is not sources.SourceBase}
    assert {cls for cls, _, _ in _TARGETS} == exported


@pytest.mark.parametrize('consumer_type', [
    'CCVS', 'CCCS', 'CurrentControlledVoltageSource',
    'CurrentControlledCurrentSource',
])
@pytest.mark.parametrize('target_cls,eligible,params', _TARGETS,
                         ids=[cls.__name__ for cls, _, _ in _TARGETS])
def test_control_target_requires_branch_current(target_cls, eligible, params,
                                                consumer_type):
    spec = _spec(target_cls.__name__, params, consumer_type)
    if eligible:
        parsed = CircuitSpec.from_dict(spec)
        assert parsed.sources[0].ctrl_vname == 'target'
    else:
        with pytest.raises(ValueError) as exc:
            CircuitSpec.from_dict(spec)
        message = str(exc.value)
        for detail in ('readout', 'ctrl_vname', 'target', target_cls.__name__,
                       'branch current'):
            assert detail in message


@pytest.mark.parametrize('alias,eligible,params', [
    ('VCVS', True, _TARGETS[2][2]),
    ('VCCS', False, _TARGETS[3][2]),
    ('CCVS', True, _TARGETS[4][2]),
    ('CCCS', False, _TARGETS[5][2]),
])
def test_control_target_aliases(alias, eligible, params):
    spec = _spec(alias, params)
    if eligible:
        CircuitSpec.from_dict(spec)
    else:
        with pytest.raises(ValueError, match=getattr(sources, alias).__name__):
            CircuitSpec.from_dict(spec)


def test_control_eligibility_follows_source_class_not_type_spelling(monkeypatch):
    class Ammeter(sources.VoltageSource):
        pass

    monkeypatch.setattr(sources, 'Ammeter', Ammeter, raising=False)
    spec = _spec('Ammeter', _TARGETS[0][2])
    # A class alone is not a codegen source type: emission also needs its
    # catalog alias. Once registered, control eligibility follows the class.
    with pytest.raises(ValueError, match="unknown type 'Ammeter'"):
        CircuitSpec.from_dict(spec)
    from silk.codegen.generator import _source_meta
    monkeypatch.setitem(_source_meta()['sources'], 'Ammeter', {'alias': 'v'})
    CircuitSpec.from_dict(spec)
    assert "sources.vdc('target'" in Generator().generate(spec)


@pytest.mark.slow
@needs_pdk
@needs_ngspice
def test_ccvs_sensing_behavioral_voltage_generates_and_simulates_via_full_codegen(tmp_path):
    """TASK-282.2 AC #3, JSON path: a CCVS ctrl_vname naming a
    BehavioralVoltage in sources[] must generate AND simulate with the
    physically correct value through the full Generator (not just
    CircuitSpec.from_dict) — now that TASK-282.4 resolves ctrl_vname to the
    control's own SPICE element name at deck emission, this is no longer a
    generation-only check.

    A zero-volt BehavioralVoltage (expression='0') is the ammeter — same
    role a VoltageSource(dc=0) plays elsewhere — carrying whatever current
    an explicit drive source forces through it; 1 mA times transresistance
    1000 must give v(out) = 1.0, exactly as TASK-281/TASK-282.4's own
    sources.*-API regression (test_eligible_control_senses_one_milliamp_in_ngspice
    above) already proves for the same target class outside codegen.
    """
    spec = {
        'circuit_name': 'branch_current',
        'pdk': 'sky130',
        'output_dir': str(tmp_path),
        'sources': [
            dict(type='CCVS', instance='readout', out_pos='out', out_neg='0',
                 ctrl_vname='target', transresistance=1000),
            dict(type='BehavioralVoltage', instance='target',
                 positive='sense', negative='0', expression='0'),
            dict(type='CurrentSource', instance='drive',
                 positive='0', negative='sense', dc=0.001),
        ],
    }
    code = Generator().generate(spec)
    assert "ckt.add(sources.BehavioralVoltage('target'" in code
    assert "ckt.add(sources.CurrentControlledVoltageSource('readout'" in code
    code = code.replace("ckt.simulate(corner='tt')",
                        "deck_path = ckt.generate_netlist(corner='tt')")
    ns: dict = {}
    exec(code, ns)
    ckt = ns['ckt']
    deck_path = ns['deck_path']
    with open(deck_path) as fh:
        deck_text = fh.read()
    assert 'Hreadout out 0 Btarget 1000' in deck_text, (
        "ctrl_vname did not resolve to the BehavioralVoltage's own SPICE "
        "element name (TASK-282.4)"
    )
    # No analysis is emitted for a spec with no 'simulation' key — insert an
    # operating-point analysis before the existing .control block so the
    # deck actually simulates, same as the deck text produced above.
    with open(deck_path, 'w') as fh:
        fh.write(deck_text.replace('.control', '.op\n.control', 1))
    raw_path = Path(deck_path).with_name('rawfile.raw')
    subprocess.run([ngspice_executable(), '-b', deck_path],
                   check=True, capture_output=True, timeout=30)
    vectors = {v.name.lower(): v.data for v in read_rawfile(raw_path).vectors}
    assert vectors['v(out)'] == pytest.approx([1.0], abs=1e-9)


def test_ccvs_sensing_behavioral_current_raises_via_full_codegen():
    """Same JSON path as above, but BehavioralCurrent has no voltage-source
    branch current — the TASK-281 ValueError must still fire through the
    full Generator, not just CircuitSpec.from_dict."""
    spec = _spec('BehavioralCurrent',
                  dict(positive='sense', negative='0', expression='0.001'))
    with pytest.raises(ValueError, match='branch current'):
        Generator().generate(spec)


def test_source_schema_export_stays_byte_identical():
    # TASK-535: Sine.voffset now defaults to zero; positional order is unchanged.
    # Re-pin deliberately on the next intentional schema change; the six
    # pre-existing entries' own byte-identity is separately asserted in
    # tests/test_source_schema.py::test_six_pre_existing_entries_byte_identical.
    assert hashlib.sha256(export_source_schema_json().encode('utf-8')).hexdigest() == (
        '3ba8bde9b845b5ad26417bba2f30556975f1fc6e3076c0368f9fd631e5434273'
    )


@pytest.mark.slow
@needs_ngspice
@pytest.mark.parametrize('target_cls,eligible,params',
                         [case for case in _TARGETS if case[1]],
                         ids=[cls.__name__ for cls, ok, _ in _TARGETS if ok])
def test_eligible_control_senses_one_milliamp_in_ngspice(tmp_path, target_cls,
                                                       eligible, params):
    """1000 ohms times +1 mA must produce +1 V for V, E, H and B V=.

    Validate at the spec boundary, then exercise the source classes' existing
    cards directly: behavioral sources have no codegen schema yet. Every
    ``ctrl_vname`` here — ``target``'s own (for the CCVS case) and
    ``readout``'s — is the referenced source's PYTHON INSTANCE NAME, exactly
    as documented and as a caller would write it; silk translates each to its
    SPICE element name at deck emission (owner ruling, TASK-282.4), so the
    test never touches ``to_spice()`` to build a reference.
    """
    CircuitSpec.from_dict(_spec(target_cls.__name__, params))
    with Module('branch_current', str(tmp_path), 'branch_current.cir',
                pdk_details=None):
        seed = sources.VoltageSource('seed', positive='seed', negative='0', dc=0)
        target = target_cls('target', **params)
        readout = sources.CCVS('readout', out_pos='out', out_neg='0',
                               ctrl_vname='target', transresistance=1000)
        drive = sources.CurrentSource('drive', positive='0', negative='sense',
                                      dc=0.001)
    deck = tmp_path / 'branch_current.cir'
    deck.write_text(
        '* Branch-current control regression\n'
        + ''.join(src.to_spice() for src in (seed, target, drive, readout))
        + 'Rload out 0 1meg\n.op\n.end\n'
    )
    raw_path = tmp_path / 'branch_current.raw'
    subprocess.run([ngspice_executable(), '-b', '-r', str(raw_path), str(deck)],
                   check=True, capture_output=True, timeout=30)
    vectors = {v.name.lower(): v.data for v in read_rawfile(raw_path).vectors}
    assert vectors['v(out)'] == pytest.approx([1.0], abs=1e-9)


@needs_ngspice
def test_control_resolves_inside_subckt_module_and_at_top_level(tmp_path):
    """The same instance-name resolution works nested in a ``.subckt`` as at
    testbench top level (AC #5): a child module has its own control chain
    (``target``/``readout``, named the same as the top level's), the top
    level has its own, and each ``ctrl_vname`` must resolve against its OWN
    module's sources — not the other one's — through the real
    ``generate_netlist()`` emission path (not hand-authored deck text)."""
    from silk.elements import elements
    from silk.simulation import Op

    tmp_path = str(tmp_path)

    # Only ideal sources/passives are involved: do not parse a transistor PDK
    # to verify control-name scoping, or require models on a bare checkout.
    child = Module('branch_current', tmp_path, 'child.cir', pdk_details=None,
                   module_name='sense_child')
    child.define_module_ports({'sense': 'inout', 'out': 'inout'})
    with child:
        sources.VoltageSource('target', positive='sense', negative='0', dc=0)
        sources.CCVS('readout', out_pos='out', out_neg='0',
                     ctrl_vname='target', transresistance=1000)

    tb = Module('branch_current', tmp_path, 'nested.cir', pdk_details=None)
    with tb:
        sources.VoltageSource('target', positive='top_sense', negative='0', dc=0)
        sources.CCVS('readout', out_pos='top_out', out_neg='0',
                     ctrl_vname='target', transresistance=1000)
        sources.CurrentSource('drive', positive='0', negative='top_sense', dc=0.001)
        sources.CurrentSource('drive2', positive='0', negative='sense_in', dc=0.001)
    tb.add_module(child, 'u1', sense='sense_in', out='child_out')
    tb.add(elements.resistor('rload_top', p='top_out', n='0', value=1e6))
    tb.add(elements.resistor('rload_child', p='child_out', n='0', value=1e6))
    tb.add_sim_type(Op())

    tb.generate_netlist()
    subprocess.run([ngspice_executable(), '-b', tb._run_files.netlist_path],
                   check=True, capture_output=True, timeout=30)
    vectors = {v.name.lower(): v.data
               for v in read_rawfile(tb._run_files.rawfile_path).vectors}
    assert vectors['v(top_out)'] == pytest.approx([1.0], abs=1e-9)
    assert vectors['v(child_out)'] == pytest.approx([1.0], abs=1e-9)
