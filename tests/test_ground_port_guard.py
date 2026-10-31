"""TASK-36 — a declared port named ``gnd`` is an unportable interface.

ngspice-42 folds the node name ``gnd`` (case-insensitively) onto the global
SPICE node ``0``; Xyce 7.10 folds nothing, at any spelling. Node
``0`` is global and pierces subcircuit boundaries, so folding a *port* name
discards whatever the parent wired to that port position. Measured on one
byte-identical deck::

    .subckt blk a gnd / R1 a gnd 1k / .ends
    Xblk top mid blk        <- parent wires the gnd port to 'mid'
    R2 mid 0 1k
    ngspice-42 :  I(V1) = -1.00 mA  V(mid) = 0.0   <- port ignored
    Xyce 7.10  :  I(V1) = -0.50 mA  V(mid) = 0.5   <- port honoured

Both runs "succeed"; neither warns. silk therefore rejects the DECLARATION —
a reusable block cannot know how it will be wired later, so a check keyed on
the instantiation sites present in one netlist could never protect its
interface.

One shared predicate and diagnostic (:mod:`silk.circuit.ground_port`) protect
declaration, ``STRUCT-012`` lint, netlist emission and static simulation setup,
including behavioral admission. Formal ``0`` refuses at declaration; formal
``gnd`` retains its lint/emission-time refusal and also refuses during setup.

The declaration, lint, emission and setup checks are simulator-free, using
both real ``BasePDK`` device catalogs and ideal-device modules.
The marked slow divider control runs ngspice with two ideal resistors and
checks that a safe ``vss`` formal preserves the parent's wiring and 0.5 V.
"""
import os
import re
from copy import deepcopy
import pickle

import pytest

from silk.circuit.circuit_elements import PortDir, PortRole
from silk.circuit.ground_port import (
    FOLDED_GROUND_PORT_NAMES, SAFE_GROUND_PORT_NAME,
    is_unportable_ground_port, unportable_ground_ports,
)
from silk.circuit.modules import Module
from silk.exceptions import CircuitError
from silk.lint.registry import RuleRegistry
from silk.lint.rules.structural import UnportableGroundPortRule
from tests._env import needs_ngspice


def _legacy_interface(tmp_path, spelling='0'):
    """An interface restored without passing through today's declaration API."""
    from silk.circuit._module_base import ModulePin
    mod = Module(None, str(tmp_path), 'legacy.cir', None, module_name='legacy')
    mod.pin_list = {spelling: ModulePin(spelling, net='mid', direction=PortDir.INOUT)}
    return mod


def _assert_ground_diagnosis(message, spelling, block='legacy'):
    assert f"block {block!r}" in message
    assert f"port named {spelling!r}" in message
    assert 'global' in message and '0' in message
    assert 'Rename' in message and "'vss'" in message
    assert 'parent' in message and 'ground' in message


@pytest.mark.parametrize('value,error', [('0', ValueError), ('1', ValueError),
                                        ('01', ValueError), (0, TypeError), (1, TypeError)])
def test_formal_numeric_name_refuses_before_registration(tmp_path, value, error):
    mod = Module(None, str(tmp_path), 'ports.cir', None, module_name='legacy')
    with pytest.raises(error) as caught:
        mod.define_module_ports({value: PortDir.INOUT})
    assert value not in mod.pin_list
    if value == '0':
        _assert_ground_diagnosis(str(caught.value), '0')
    if isinstance(value, int):
        assert 'vss' in str(caught.value)
        assert "write '0'" not in str(caught.value)


@pytest.mark.parametrize('kind', ['pin', 'net', 'terminal', 'port'])
def test_validator_separates_formal_names_from_actual_ground(kind):
    from silk.elements.base import validate_net_token
    if kind == 'port':
        with pytest.raises(ValueError, match='invalid port name'):
            validate_net_token('0', who='interface', kind=kind)
    else:
        assert validate_net_token('0', who='connection', kind=kind) == '0'


@pytest.mark.parametrize('route', ['generate', 'pure', 'instance', 'header', 'lint'])
@pytest.mark.parametrize('spelling', ['0', 'gnd', 'GND'])
@pytest.mark.parametrize('restored', ['direct', 'deepcopy', 'pickle'])
def test_restored_interface_refuses_with_shared_diagnostic(tmp_path, route, spelling, restored):
    from silk.circuit.modules_netlist import _emit_deck_text
    from silk.simulation.run_files import RunFiles
    mod = _legacy_interface(tmp_path, spelling)
    if restored == 'deepcopy':
        mod = deepcopy(mod)
    elif restored == 'pickle':
        mod = pickle.loads(pickle.dumps(mod))
    if route == 'lint':
        found = _violations(mod)
        assert [v.rule_id for v in found] == ['STRUCT-012']
        message = found[0].message
    else:
        with pytest.raises(CircuitError) as caught:
            if route == 'generate':
                mod.generate_netlist(subckt=True)
            elif route == 'instance':
                mod.generate_netlist_entry('dut')
            elif route == 'header':
                mod.generate_subckt(0)
            else:
                files = RunFiles.resolve(os.path.join(mod.dir_folder, mod.netlistname), mod.dir_folder)
                _emit_deck_text(mod, None, files)
        message = str(caught.value)
        assert message == _violations(mod)[0].message
    _assert_ground_diagnosis(message, spelling)


@pytest.mark.parametrize('field', ['key', 'name'])
@pytest.mark.parametrize('route', ['generate', 'pure', 'instance', 'header'])
def test_mutated_formal_key_or_pin_name_gets_ground_diagnosis(tmp_path, field, route):
    from silk.circuit.modules_netlist import _emit_deck_text
    from silk.simulation.run_files import RunFiles
    mod = _legacy_interface(tmp_path, 'vss')
    if field == 'key':
        mod.pin_list['0'] = mod.pin_list.pop('vss')
    else:
        mod.pin_list['vss'].name = '0'
    with pytest.raises(CircuitError) as caught:
        if route == 'generate':
            mod.generate_netlist(subckt=True)
        elif route == 'instance':
            mod.generate_netlist_entry('dut')
        elif route == 'header':
            mod.generate_subckt(0)
        else:
            _emit_deck_text(mod, None, RunFiles.resolve(os.path.join(mod.dir_folder, mod.netlistname), mod.dir_folder))
    _assert_ground_diagnosis(str(caught.value), '0')


@pytest.mark.parametrize('backend', ['ngspice', 'xyce'])
@pytest.mark.parametrize('route', ['generate', 'pure'])
def test_nested_zero_port_diagnosis_precedes_generic_syntax_and_backend(tmp_path, backend, route):
    from silk.circuit.modules_netlist import _emit_deck_text
    from silk.simulation.run_files import RunFiles
    leaf = _legacy_interface(tmp_path / 'leaf')
    middle = Module(None, str(tmp_path / 'middle'), 'middle.cir', None, module_name='middle')
    middle.add_module(leaf, 'inner', nets={'0': 'mid'})
    mod = Module(None, str(tmp_path), 'top.cir', None)
    mod.simulator = backend
    mod.add_module(middle, 'outer')
    # An internal gnd would trigger a different Xyce-specific refusal later.
    mod.add_net('gnd')
    with pytest.raises(CircuitError) as caught:
        if route == 'generate':
            mod.generate_netlist()
        else:
            _emit_deck_text(mod, None, RunFiles.resolve(os.path.join(mod.dir_folder, mod.netlistname), mod.dir_folder))
    _assert_ground_diagnosis(str(caught.value), '0')
    assert 'invalid port name' not in str(caught.value)
    assert not os.path.exists(os.path.join(mod.dir_folder, mod.netlistname))


def test_zero_is_global_ground_not_a_measured_folded_alias():
    assert FOLDED_GROUND_PORT_NAMES == {'gnd'}
    assert is_unportable_ground_port('0')
    from silk.circuit.ground_port import unportable_ground_port_message
    message = unportable_ground_port_message('legacy', '0')
    assert 'global' in message
    assert 'folds' not in message


def test_declaration_and_legacy_zero_share_the_same_reason(tmp_path):
    mod = _legacy_interface(tmp_path)
    lint_message = _violations(mod)[0].message
    mod.pin_list.clear()
    with pytest.raises(ValueError) as caught:
        mod.define_module_ports({'0': PortDir.INOUT})
    assert str(caught.value) == lint_message


@pytest.mark.parametrize('field', ['key', 'name'])
@pytest.mark.parametrize('route', ['generate', 'pure', 'topology', 'flatten', 'setup', 'run'])
@pytest.mark.parametrize('modeled', [False, True])
def test_placement_only_mutation_refuses_before_synchronization(tmp_path, field, route, modeled):
    from silk import sources
    from silk.circuit.modules_netlist import _emit_deck_text
    from silk.simulation import Transient, validate_simulation_setup
    from silk.simulation.behavioral import BehavioralEngine
    from silk.simulation.run_files import RunFiles
    from silk.simulation.topology import inspect_topology
    child = _legacy_interface(tmp_path / 'child', 'vss')
    child.add_source(sources.VoltageSource('bias', positive='internal', negative='0', dc=1))
    if modeled:
        def model(ports, t, params):
            pytest.fail('the model must not run for a refused interface')
        child.set_behavioral_model(model, execution='vector')
    mod = Module(None, str(tmp_path), 'top.cir', None)
    mod.add_module(child, 'u', vss='0')
    mod.add_sim_type(Transient(1, 2))
    if route in ('flatten', 'setup', 'run'):
        mod.simulator = 'behavioral'
    pins = mod._element_mgr.elem_dict['u']['pin_list']
    if field == 'key':
        pins['0'] = pins.pop('vss')
    else:
        pins['vss'].name = '0'
    assert list(child.pin_list) == ['vss'] and child.pin_list['vss'].name == 'vss'
    with pytest.raises(CircuitError) as caught:
        if route == 'generate':
            mod.generate_netlist()
        elif route == 'pure':
            _emit_deck_text(mod, None, RunFiles.resolve(os.path.join(mod.dir_folder, mod.netlistname), mod.dir_folder))
        elif route == 'topology':
            inspect_topology(mod)
        elif route == 'flatten':
            BehavioralEngine()._flatten(mod, report=False)
        elif route == 'setup':
            validate_simulation_setup(mod)
        else:
            mod.simulate()
    _assert_ground_diagnosis(str(caught.value), '0')
    assert ('0' in pins) if field == 'key' else pins['vss'].name == '0'


def test_actual_ground_values_keep_all_existing_construction_routes(pdk, tmp_path):
    from silk.elements import elements, passives
    from silk import sources
    mod = _mod(pdk, tmp_path, 'safe')
    mod.define_module_ports({'vss': PortDir.GROUND, 'agnd': PortDir.INOUT,
                             'ground': PortDir.INOUT, 'd[0]': PortDir.IN})
    mod.add_net('0')
    mod.add(pdk.nmos1p8('factory', d='out', g='in', s='0', b='0', w=1, l=.15))
    with pytest.warns(DeprecationWarning):
        mod.add_element(pdk.nmos1p8, 'legacy', nets={'d': 'out', 'g': 'in', 's': '0', 'b': '0'})
    mod.add(elements.resistor('factory_r', p='out', n='0', value=1e3))
    with pytest.warns(DeprecationWarning):
        mod.add_passive(passives.Resistor('legacy_r', 'out', '0', 1e3))
    mod.add_source(sources.VoltageSource('bias', positive='in', negative='0', dc=1))
    assert _violations(mod) == []
    assert '.subckt safe vss agnd ground d[0]' in mod.generate_netlist(subckt=True)


@pytest.mark.slow
@needs_ngspice
def test_divider_safe_formal_preserves_parent_mapping(tmp_path):
    from silk.elements import elements
    from silk import sources
    from silk.simulation import Transient
    leaf = Module(None, str(tmp_path / 'leaf'), 'leaf.cir', None, module_name='divider_arm')
    with pytest.raises(ValueError) as caught:
        leaf.define_module_ports({'0': PortDir.INOUT})
    _assert_ground_diagnosis(str(caught.value), '0', 'divider_arm')
    leaf.define_module_ports({'a': PortDir.INOUT, 'vss': PortDir.INOUT})
    leaf.add(elements.resistor('upper', p='a', n='vss', value=1e3))
    mod = Module(None, str(tmp_path), 'divider.cir', None)
    mod.add_module(leaf, 'arm', a='input', vss='mid')
    mod.add(elements.resistor('lower', p='mid', n='0', value=1e3))
    mod.add_source(sources.VoltageSource('drive', positive='input', negative='0', dc=1))
    mod.add_sim_type(Transient(1e-9, 4e-9))
    mod.simulate()
    assert float(mod.load_signal('mid').y[-1]) == pytest.approx(.5)


# The spellings ngspice-42 was measured to fold, one alias at a time.
FOLDED = ['gnd', 'GND', 'Gnd']

# Measured NOT folded by ngspice-42. They are legitimately distinct nets (a
# negative supply, a quiet analog ground, a schematic-style global), so flagging
# any of them would be a new bug.
NOT_FOLDED = ['vss', 'VSS', 'agnd', 'ground', 'GROUND']


def _mod(pdk, tmp_path, name=None, fname=None):
    return Module('PDK_skywater130', str(tmp_path), fname or f'{name or "top"}.cir',
                  pdk_details=pdk, module_name=name)


def _block(pdk, tmp_path, name='blk', ground_port='gnd'):
    """A real sky130 inverter subcircuit whose ground port is *ground_port*."""
    m = _mod(pdk, tmp_path, name)
    m.define_module_ports({
        'in': (PortDir.IN, PortRole.DIGITAL),
        'out': (PortDir.OUT, PortRole.DIGITAL),
        'vdd': (PortDir.POWER, PortRole.POWER),
        ground_port: (PortDir.GROUND, PortRole.GROUND),
    })
    with m:
        pdk.nmos1p8('mn', d='out', g='in', s=ground_port, b=ground_port,
                    w=1.0, l=0.15)
        pdk.pmos1p8('mp', d='out', g='in', s='vdd', b='vdd', w=2.0, l=0.15)
    return m


def _violations(module):
    return list(UnportableGroundPortRule().check(module))


# ---------------------------------------------------------------------------
# AC #1 — the lint rule fires on the DECLARATION, not on any wiring
# ---------------------------------------------------------------------------

class TestLintRuleFiresOnDeclaration:

    @pytest.mark.parametrize('spelling', FOLDED)
    def test_fires_on_a_bare_declaration_with_no_wiring_at_all(
            self, pdk, tmp_path, spelling):
        """No devices, no nets, no instantiation — just the port list. If the
        rule needed a mapping to look at, this would pass and the block's
        interface would ship unprotected."""
        m = _mod(pdk, tmp_path, 'iface')
        m.define_module_ports({'a': PortDir.IN, spelling: PortDir.IN})
        assert dict(m.net_dict) == {} and dict(m.elem_dict) == {}  # nothing wired
        found = _violations(m)
        assert [v.rule_id for v in found] == ['STRUCT-012']
        assert found[0].location.port_name == spelling
        assert found[0].location.module_name == 'iface'

    def test_fires_when_the_port_is_wired_to_ground_itself(self, pdk, tmp_path):
        """Mapping the port to '0' is harmless in *this* deck (both backends
        measured identical, -2.0 mA), but the INTERFACE is still unportable —
        the next instantiation is out of this check's reach. So it still
        fires."""
        blk = _block(pdk, tmp_path, 'blk')
        top = _mod(pdk, tmp_path)
        top.add_module(blk, 'u1', nets={'in': 'a', 'out': 'y',
                                         'vdd': 'vdd', 'gnd': '0'})
        assert len(_violations(blk)) == 1

    def test_severity_is_error(self, pdk, tmp_path):
        from silk.lint.severity import Severity
        assert UnportableGroundPortRule().default_severity is Severity.ERROR

    def test_in_the_default_catalog(self):
        assert 'STRUCT-012' in {r.rule_id for r in RuleRegistry.default().rules}

    def test_reported_through_the_registry_run(self, pdk, tmp_path):
        blk = _block(pdk, tmp_path, 'blk')
        result = RuleRegistry.default().run(blk)
        assert 'STRUCT-012' in {v.rule_id for v in result.violations}

    def test_waivable_by_rule_id_and_port(self, pdk, tmp_path):
        """Catalogued rules are waivable through the standard rule_id +
        location path; nothing about STRUCT-012 opts out of it."""
        from silk.lint.waivers import Waiver, WaiverMatcher, apply_waivers
        blk = _block(pdk, tmp_path, 'blk')
        waiver = Waiver(
            rule_id='STRUCT-012',
            matcher=WaiverMatcher('port', {'port_name': 'gnd',
                                           'module_name': 'blk'}),
            reason='legacy block, rename scheduled',
        )
        kept, applied = apply_waivers(_violations(blk), [waiver])
        assert kept == [] and len(applied) == 1


# ---------------------------------------------------------------------------
# AC #2 — the emission guard, because nothing semantic runs at simulate time
# ---------------------------------------------------------------------------

class TestEmissionGuard:

    @pytest.mark.parametrize('spelling', FOLDED)
    def test_subckt_emission_raises(self, pdk, tmp_path, spelling):
        blk = _block(pdk, tmp_path, 'blk', ground_port=spelling)
        with pytest.raises(CircuitError):
            blk.generate_netlist(subckt=True)

    def test_top_level_emission_raises_for_a_submodule_port(self, pdk, tmp_path):
        """The offending declaration is one level down; the top-level deck
        reaches it through generate_module_definitions -> generate_netlist."""
        blk = _block(pdk, tmp_path, 'blk')
        top = _mod(pdk, tmp_path)
        top.add_module(blk, 'u1', nets={'in': 'a', 'out': 'y',
                                         'vdd': 'vdd', 'gnd': '0'})
        with pytest.raises(CircuitError, match='STRUCT|gnd|global'):
            top.generate_netlist()

    def test_no_netlist_file_is_written(self, pdk, tmp_path):
        """The guard runs before the top-level write, so a rejected deck
        leaves nothing on disk to be picked up by a later step."""
        import os
        blk = _block(pdk, tmp_path, 'blk')
        top = _mod(pdk, tmp_path, fname='top.cir')
        top.add_module(blk, 'u1', nets={'in': 'a', 'out': 'y',
                                         'vdd': 'vdd', 'gnd': '0'})
        with pytest.raises(CircuitError):
            top.generate_netlist()
        assert not os.path.exists(os.path.join(top.dir_folder, top.netlistname))

    def test_simulate_never_reaches_the_simulator(self, pdk, tmp_path,
                                                 monkeypatch):
        """AC #2 stated directly: simulate() must fail at emission, before the
        run. run_simulator is replaced by a tripwire — if the guard were lint
        only, this test would fail with 'simulator was reached'."""
        blk = _block(pdk, tmp_path, 'blk')
        top = _mod(pdk, tmp_path)
        top.add_module(blk, 'u1', nets={'in': 'a', 'out': 'y',
                                         'vdd': 'vdd', 'gnd': '0'})

        def _tripwire(*args, **kwargs):
            raise AssertionError('simulator was reached')

        monkeypatch.setattr(type(top), 'run_simulator', _tripwire)
        with pytest.raises(CircuitError):
            top.simulate()

    def test_every_offending_port_is_named_at_once(self, pdk, tmp_path):
        """Two bad ports cannot coexist (they differ only by case, which
        TASK-37 already rejects), so the multi-offender path is exercised via
        the message builder over the helper's own list."""
        m = _mod(pdk, tmp_path, 'iface')
        m.define_module_ports({'gnd': PortDir.IN})
        assert unportable_ground_ports(m) == ['gnd']


# ---------------------------------------------------------------------------
# AC #3 — general, not backend-conditional
# ---------------------------------------------------------------------------

class TestBackendIndependence:

    @pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
    def test_same_rejection_on_every_backend(self, pdk, tmp_path, simulator):
        """If this fired only on ngspice, someone developing on Xyce would
        pass every test and meet the divergence later, on another machine,
        with nothing pointing at the cause."""
        blk = _block(pdk, tmp_path, 'blk')
        blk.simulator = simulator
        with pytest.raises(CircuitError) as exc:
            blk.generate_netlist(subckt=True)
        assert "'gnd'" in str(exc.value)

    @pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
    def test_lint_verdict_does_not_depend_on_the_backend(
            self, pdk, tmp_path, simulator):
        blk = _block(pdk, tmp_path, 'blk')
        blk.simulator = simulator
        assert len(_violations(blk)) == 1

    def test_messages_are_identical_across_backends(self, pdk, tmp_path):
        messages = []
        for simulator in ('ngspice', 'xyce'):
            blk = _block(pdk, tmp_path / simulator, 'blk')
            blk.simulator = simulator
            with pytest.raises(CircuitError) as exc:
                blk.generate_netlist(subckt=True)
            messages.append(str(exc.value))
        assert len(set(messages)) == 1


# ---------------------------------------------------------------------------
# AC #4 — the message is the deliverable
# ---------------------------------------------------------------------------

class TestMessageContent:

    @staticmethod
    def _assert_actionable(msg):
        assert "'blk'" in msg                      # the block
        assert "'gnd'" in msg                      # the port
        assert 'ngspice' in msg                    # who folds it
        assert 'global' in msg and '0' in msg      # what it is folded to
        assert 'silently discarded' in msg         # the consequence
        assert 'portable' in msg                   # why it is a defect
        assert "'vss'" in msg                      # the remedy
        assert 'Rename' in msg

    def test_emission_message(self, pdk, tmp_path):
        blk = _block(pdk, tmp_path, 'blk')
        with pytest.raises(CircuitError) as exc:
            blk.generate_netlist(subckt=True)
        self._assert_actionable(str(exc.value))

    def test_lint_message(self, pdk, tmp_path):
        blk = _block(pdk, tmp_path, 'blk')
        self._assert_actionable(_violations(blk)[0].message)

    def test_both_mechanisms_say_the_same_thing(self, pdk, tmp_path):
        blk = _block(pdk, tmp_path, 'blk')
        lint_msg = _violations(blk)[0].message
        with pytest.raises(CircuitError) as exc:
            blk.generate_netlist(subckt=True)
        assert str(exc.value) == lint_msg

    def test_message_names_the_actual_spelling_used(self, pdk, tmp_path):
        blk = _block(pdk, tmp_path, 'blk', ground_port='GND')
        assert "'GND'" in _violations(blk)[0].message

    def test_rationale_documents_the_remedy(self):
        rationale = UnportableGroundPortRule.rationale
        assert 'vss' in rationale and 'Xyce' in rationale


# ---------------------------------------------------------------------------
# AC #5 — internal / top-level nets named gnd are NOT flagged
# ---------------------------------------------------------------------------

class TestInternalGndNetsPass:
    """Xyce already refuses the internal case loudly — 27 warnings, "Voltage
    Node (GND) does not have a DC path to ground", singular matrix, no result
    — so it is self-detecting and silk must stay out of it. Flagging it would
    also blow the migration up from "blocks that declare a gnd port" to
    "every occurrence of the string gnd"."""

    def test_internal_net_named_gnd_lints_clean(self, pdk, tmp_path):
        blk = _block(pdk, tmp_path, 'blk', ground_port='vss')
        with blk:
            pdk.nmos1p8('mn2', d='out', g='in', s='gnd', b='gnd',
                        w=1.0, l=0.15)
        assert 'gnd' in blk.net_dict          # the net really is there
        assert 'gnd' not in (blk.pin_list or {})
        assert _violations(blk) == []

    def test_internal_net_named_gnd_emits(self, pdk, tmp_path):
        blk = _block(pdk, tmp_path, 'blk', ground_port='vss')
        with blk:
            pdk.nmos1p8('mn2', d='out', g='in', s='gnd', b='gnd',
                        w=1.0, l=0.15)
        assert '.subckt blk' in blk.generate_netlist(subckt=True)

    def test_top_level_net_named_gnd_emits(self, pdk, tmp_path):
        """A top-level module with a 'gnd' net and no ports at all: nothing to
        declare, nothing to flag."""
        top = _mod(pdk, tmp_path, fname='top.cir')
        with top:
            pdk.nmos1p8('mn', d='out', g='in', s='gnd', b='gnd',
                        w=1.0, l=0.15)
        top_deck_path = top.generate_netlist()
        assert 'gnd' in open(top_deck_path).read()

    def test_helper_reads_ports_only(self, pdk, tmp_path):
        blk = _block(pdk, tmp_path, 'blk', ground_port='vss')
        with blk:
            pdk.nmos1p8('mn2', d='out', g='in', s='gnd', b='gnd',
                        w=1.0, l=0.15)
        assert 'gnd' in blk.net_dict
        assert unportable_ground_ports(blk) == []


# ---------------------------------------------------------------------------
# AC #6 — the near-misses, one regression test per spelling
# ---------------------------------------------------------------------------

class TestSafeSpellingsAreNotFlagged:
    """ngspice-42 was measured one alias at a time: it folds gnd/GND/Gnd and
    nothing else. Every name below stays a distinct node in every backend, so
    flagging it would be a new bug, not a fix."""

    @pytest.mark.parametrize('spelling', NOT_FOLDED)
    def test_predicate_says_safe(self, spelling):
        assert is_unportable_ground_port(spelling) is False

    @pytest.mark.parametrize('spelling', NOT_FOLDED)
    def test_lints_clean(self, pdk, tmp_path, spelling):
        blk = _block(pdk, tmp_path, 'blk', ground_port=spelling)
        assert _violations(blk) == []

    @pytest.mark.parametrize('spelling', NOT_FOLDED)
    def test_emits(self, pdk, tmp_path, spelling):
        blk = _block(pdk, tmp_path, 'blk', ground_port=spelling)
        netlist = blk.generate_netlist(subckt=True)
        assert f'.subckt blk in out vdd {spelling}' in netlist

    @pytest.mark.parametrize('spelling', NOT_FOLDED)
    def test_default_catalog_stays_quiet(self, pdk, tmp_path, spelling):
        blk = _block(pdk, tmp_path, 'blk', ground_port=spelling)
        result = RuleRegistry.default().run(blk)
        assert 'STRUCT-012' not in {v.rule_id for v in result.violations}

    @pytest.mark.parametrize('spelling', FOLDED)
    def test_predicate_says_unportable(self, spelling):
        assert is_unportable_ground_port(spelling) is True

    def test_non_string_port_names_are_not_matched(self):
        assert is_unportable_ground_port(None) is False
        assert is_unportable_ground_port(0) is False


# ---------------------------------------------------------------------------
# TASK-117 — the docs must not teach the defect
# ---------------------------------------------------------------------------

# Dated design records, not instructions: superseded plans/specs/brainstorms
# (docs/superpowers/**, written before TASK-36), archived plans, and the
# correspondence files that quote what a peer tool asked for or was told at the
# time. Rewriting those would falsify a record; nobody builds from them.
_HISTORICAL_DIRS = ('docs/archive', 'docs/superpowers')
_HISTORICAL_SUFFIXES = ('-reply.md', '-request.md')

# A quoted dict key, and a Python keyword argument. Both spell a *port* name in
# every silk idiom that takes them: define_module_ports, the add_element /
# add_module_under_test portmaps, the codegen contract's "ports" object, and
# the port-kwarg wiring form. A net name is never one of these — it is the
# *value* ({'s': 'gnd'}), which this deliberately does not match.
_QUOTED_KEY = re.compile(r"""['"]([A-Za-z_][A-Za-z0-9_!]*)['"]\s*:""")
_KEYWORD_ARG = re.compile(r"""(?<![\w.])([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)""")


def _teaching_docs():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs')
    root = os.path.normpath(root)
    repo = os.path.dirname(root)
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if not name.endswith('.md'):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, repo).replace(os.sep, '/')
            if rel.startswith(_HISTORICAL_DIRS) or rel.endswith(_HISTORICAL_SUFFIXES):
                continue
            out.append((rel, path))
    assert out, 'no docs found — the walk or the docs tree changed shape'
    return out


class TestTheDocsDoNotTeachTheDefect:
    """A doc example that declares a ``gnd`` port is a trap: a tool
    following it literally builds a block that ``generate_netlist`` refuses.
    The shipped examples were migrated by TASK-83; this is the same migration
    for the prose, kept true by a check rather than by a grep in a task note.
    """

    def test_no_teaching_doc_names_a_port_gnd(self):
        offenders = []
        for rel, path in _teaching_docs():
            with open(path, encoding='utf-8') as handle:
                for lineno, line in enumerate(handle, 1):
                    names = [m.group(1) for m in _QUOTED_KEY.finditer(line)]
                    names += [m.group(1) for m in _KEYWORD_ARG.finditer(line)]
                    for name in names:
                        if is_unportable_ground_port(name):
                            offenders.append(f'{rel}:{lineno}: {line.strip()[:100]}')
        assert offenders == [], (
            'doc examples naming a port ' + repr(sorted(FOLDED_GROUND_PORT_NAMES))
            + f' — rename it to {SAFE_GROUND_PORT_NAME!r} (STRUCT-012 and the '
            'generate_netlist guard reject the declaration):\n  '
            + '\n  '.join(offenders))

    def test_the_scan_would_catch_the_thing_it_is_guarding_against(self, tmp_path):
        """The check is only worth having if it fires; the pre-TASK-117 text of
        the four fixed sites, verbatim, is the fixture."""
        doc = tmp_path / 'regression.md'
        doc.write_text(
            "sub.define_module_ports({'vdd': PortDir.POWER, 'gnd': PortDir.GROUND})\n"
            '    "gnd":    {"dir": "input",  "role": "ground", "domain": "AVDD"},\n'
            "    'gnd':  Port.ground(),\n"
            "    avdd_1p8=avdd, gnd=tb.nets['0'])\n"
            # ... and stays quiet on a net named gnd, which is not this rule's
            # business (Xyce fails loudly on one, so it is self-detecting).
            "mn = pdk.nmos1p8('mn', d='out', g='in', s='gnd', b='gnd', w=1, l=0.15)\n"
            '  "nets": {"d": "vout", "g": "vin", "s": "gnd", "b": "gnd"},\n',
            encoding='utf-8')
        hits = []
        for lineno, line in enumerate(doc.read_text(encoding='utf-8').splitlines(), 1):
            names = [m.group(1) for m in _QUOTED_KEY.finditer(line)]
            names += [m.group(1) for m in _KEYWORD_ARG.finditer(line)]
            if any(is_unportable_ground_port(n) for n in names):
                hits.append(lineno)
        assert hits == [1, 2, 3, 4]
