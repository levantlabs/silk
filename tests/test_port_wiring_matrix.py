"""TASK-33 REVIEW-PORTS — the wiring-mistake × construction-style matrix.

One test per cell. Every cell either asserts the loud, typed behaviour silk
has today, or documents-and-asserts a silent behaviour together with the
verdict on it. The docstring of each test carries the verdict tag:

* ``LOUD``               — a typed exception at construction or netlist time.
* ``SILENT-BY-DESIGN``   — silk stays quiet and that is acceptable (SPICE
                           semantics, documented opt-in mode, or lint is the
                           declared home for the check and lint fires).
* ``SILENT-GAP (TASK-n)`` — silk emits a deck that simulates wrong (or is
                           mis-parsed) with no report anywhere; filed as its
                           own task with the repro in this cell.

Construction styles (columns):

* ``with mod:`` factory devices             → ``TestFactoryWithBlock``
* ``add_element`` on a primitive device     → ``TestAddElementDevice``
* ``add_element`` on a Module (submodule)   → ``TestAddElementModule``
* testbench ``add_module_under_test``       → ``TestTestbenchPortmap``
* explicit ``connect_module``/``connect_source`` rewires → in the classes
  above plus ``TestConnectSource``
* port declaration itself                   → ``TestPortDeclaration``
* port-role inference (AC #3)               → ``TestRoleInference``

Framing fact behind many verdicts: **nothing semantic runs at simulate time.**
``Module.simulate`` / ``ModuleTestbench.run`` never invoke the lint registry
(grep: no ``RuleRegistry`` reference outside ``silk/lint``, ``interface_check``
and ``report.py``). The only hard gates are: net existence under
``strict_nets``, pin/port-name existence at wiring time, and ``pin.net is
None`` at netlist time. Every "two outputs shorted / supply on ground / no
driver" check lives in opt-in lint. Cells where lint DOES fire are marked
SILENT-BY-DESIGN with that caveat; cells where even lint is blind are GAPs.

Simulator evidence quoted in verdicts was measured 2026-09-02 with the real
sky130A models and ngspice-42. These tests use the fake-lib
``pdk`` fixture and never run a simulator.
"""
import re

import pytest

from silk import sources
from silk.circuit.circuit_elements import PortDir, PortRole, Port
from silk.circuit.modules import Module
from silk.exceptions import DuplicateNameError, CircuitError, UndeclaredNetError
from silk.lint.registry import RuleRegistry


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _module(pdk, tmp_path, name=None, strict=False):
    tag = name or 'top'
    return Module('PDK_skywater130', str(tmp_path / tag), f'{tag}.cir',
                  pdk_details=pdk, strict_nets=strict, module_name=name)


# The ground PORT is 'vss', not 'gnd': since TASK-36 a declared port named
# 'gnd' is rejected at emission (ngspice folds the port name onto global node
# 0 and silently discards the parent's wiring), and almost every cell below
# emits a deck. The ground NET the parent wires it to is still 'gnd', so the
# gnd-vs-'0' cells keep the exact split they were written to record.
_INV_PORTS = {
    'vin': Port.analog_input(),
    'vout': Port.analog_output(),
    'vdd': Port.supply(),
    'vss': Port.ground(),
}


def _inverter(pdk, tmp_path, name='inv', ports=None):
    """Factory/with-block inverter with fully annotated ports."""
    m = _module(pdk, tmp_path, name)
    m.define_module_ports(dict(ports or _INV_PORTS))
    with m:
        pdk.nmos1p8('mn', d='vout', g='vin', s='vss', b='vss', w=1, l=0.15)
        pdk.pmos1p8('mp', d='vout', g='vin', s='vdd', b='vdd', w=2, l=0.15)
    return m


def _lint(module, *, include_optional=False):
    """rule_id -> [messages] from the default registry.

    ``include_optional=True`` adds the opt-in STRUCT-009 (TASK-49). S9's
    defect — a body net that is not a port — is reported by the DEFAULT
    registry since TASK-93 promoted STRUCT-013 HIDDEN_SUPPLY_NET; the cell
    asserts both that and the opt-in STRUCT-009 view of the same block (see
    ``tests/lint/test_struct013_hidden_supply.py``). P1's STRUCT-010 report
    comes from the DEFAULT registry since TASK-94 promoted the rule with a
    regulated-output exemption (the TASK-38 gate had found it warning on an
    LDO's regulated POWER-role OUT port; see
    ``tests/lint/test_struct009_010_registry.py``).
    """
    out = {}
    for v in RuleRegistry.default(include_optional=include_optional).run(module).violations:
        out.setdefault(v.rule_id, []).append(v.message)
    return out


def _wiring_rules(lint):
    """Every rule's findings. STRUCT-006 used to be dropped here because it
    false-positived on every factory MOS (it looked for uppercase 'W'/'L';
    factory devices store 'w'/'l'); TASK-44 fixed the rule, so nothing is
    subtracted any more and these assertions now cover it too."""
    return dict(lint)


def _single_conn_nets(lint):
    """Net names STRUCT-011 SINGLE_CONNECTION_NET reported, in report order.

    The rule (TASK-35) fires on every net carrying exactly one binding, so
    it is the loud half of many cells below whose *point* is something else.
    Cells that exist to pin STRUCT-011 itself assert the whole message.
    """
    return [re.search(r"Net '([^']+)'", m).group(1)
            for m in lint.get('STRUCT-011', [])]


def _deck(module):
    module_deck_path = module.generate_netlist()
    with open(module_deck_path) as f:
        return f.read()


def _device_line(deck, inst):
    return next(l for l in deck.splitlines() if l.startswith(f'XM{inst} '))


_CASE_WARNING = "differs only by case from existing net"


# ---------------------------------------------------------------------------
# Column 1 — factory devices inside ``with mod:``
# ---------------------------------------------------------------------------

class TestFactoryWithBlock:

    def test_pin_omitted_raises_typeerror(self, pdk, tmp_path):
        """LOUD. The factory constructor requires every pin kwarg."""
        m = _module(pdk, tmp_path)
        with m, pytest.raises(TypeError, match="missing required net for pin 'g'"):
            pdk.nmos1p8('mn', d='out', s='gnd', b='gnd', w=1, l=0.15)

    def test_unknown_pin_kwarg_raises_typeerror(self, pdk, tmp_path):
        """LOUD. A misspelled pin name is an unexpected keyword."""
        m = _module(pdk, tmp_path)
        with m, pytest.raises(TypeError, match=r"unexpected keyword arguments: \['gate'\]"):
            pdk.nmos1p8('mn', d='out', g='in', s='gnd', b='gnd', w=1, l=0.15, gate='x')

    def test_undeclared_net_strict_raises(self, pdk, tmp_path):
        """LOUD. strict_nets turns a net typo into UndeclaredNetError naming the fix."""
        m = _module(pdk, tmp_path, strict=True)
        m.add_net('out'); m.add_net('gnd')
        with m, pytest.raises(UndeclaredNetError, match='net "inn" is not declared. Call add_net\\("inn"\\)'):
            pdk.nmos1p8('mn', d='out', g='inn', s='gnd', b='gnd', w=1, l=0.15)

    def test_undeclared_net_nonstrict_autocreates(self, pdk, tmp_path):
        """SILENT-BY-DESIGN at construction, LOUD in lint (TASK-35). Default
        mode auto-creates nets on first use — that is the declared contract
        of strict_nets=False and what every example relies on. The cost is
        that a typo'd net becomes a single-pin net; STRUCT-011 now names it
        (along with the equally one-ended 'out')."""
        m = _module(pdk, tmp_path)
        with m:
            pdk.nmos1p8('mn', d='out', g='inn', s='gnd', b='gnd', w=1, l=0.15)
        assert sorted(m.net_dict) == ['gnd', 'inn', 'out']
        assert _single_conn_nets(_lint(m)) == ['out', 'inn']

    def test_gate_on_single_pin_net_reported_by_lint(self, pdk, tmp_path):
        """SILENT-BY-DESIGN at netlist time, LOUD in lint (was the S5
        SILENT-GAP, fixed by TASK-35). A gate on a net nothing else touches
        still passes construction and netlist, and ngspice-42 still
        simulates it (v(nowhere) = 0.50 V, no warning) — but STRUCT-011 now
        names the net. STRUCT-004 fires only when the gate has NO net,
        STRUCT-003 only on ZERO pins, and INT-004 still skips nets with
        fewer than two submodule/source bindings; STRUCT-011 is what covers
        the one-binding shape."""
        m = _module(pdk, tmp_path)
        with m:
            pdk.nmos1p8('mn', d='y', g='nowhere', s='gnd', b='gnd', w=1, l=0.15)
            sources.VoltageSource('vdd', positive='y', negative='gnd', dc=1.8)
        deck = _device_line(_deck(m), 'mn')
        assert ' nowhere ' in deck
        assert len(m.net_dict['nowhere'].pins) == 1
        assert _wiring_rules(_lint(m)) == {'STRUCT-011': [
            "Net 'nowhere' has only one connection: mn.g — "
            "nothing else is wired to it"]}

    def test_empty_string_pin_rejected_at_construction(self, pdk, tmp_path):
        """LOUD (was SILENT-GAP, fixed by TASK-41). g='' used to be accepted,
        the chokepoint skipped the binding (falsy), and to_spice emitted a
        line one node short: ngspice-42 accepted that deck through
        silk.simulate() and invented a node named after the model
        ('sky130_fd_pr__nfet_01v8' = 0.167 V in the .op table) — a mis-wired
        transistor with no error. It is now a ValueError naming the instance
        and the pin, raised before the device is registered."""
        m = _module(pdk, tmp_path)
        with m:
            with pytest.raises(ValueError, match=r"nmos1p8 'mn' pin 'g'"):
                pdk.nmos1p8('mn', d='y', g='', s='gnd', b='gnd', w=1, l=0.15)
        assert 'mn' not in m.elem_dict
        assert '' not in m.net_dict

    def test_none_pin_rejected_at_construction(self, pdk, tmp_path):
        """LOUD (was SILENT-GAP, fixed by TASK-41). g=None used to be accepted
        at construction and only die inside ' '.join at netlist time with a
        bare 'expected str instance, NoneType found' that named neither the
        instance nor the pin. It is now a TypeError at the call that made the
        mistake, naming both."""
        m = _module(pdk, tmp_path)
        with m:
            with pytest.raises(TypeError, match=r"nmos1p8 'mn' pin 'g'"):
                pdk.nmos1p8('mn', d='y', g=None, s='gnd', b='gnd', w=1, l=0.15)
        assert 'mn' not in m.elem_dict

    def test_case_variant_net_merges_with_warning(self, pdk, tmp_path, capsys):
        """SILENT-BY-DESIGN (G16, warns). 'vdd' after 'VDD' is ONE silk net,
        the first spelling is canonical, a WARNING is printed once, and the
        deck emits the author's spelling (simulators fold case)."""
        m = _module(pdk, tmp_path)
        m.add_net('VDD')
        with m:
            pdk.nmos1p8('mn', d='vdd', g='in', s='gnd', b='gnd', w=1, l=0.15)
        assert sorted(m.net_dict) == ['VDD', 'gnd', 'in']
        assert _CASE_WARNING in capsys.readouterr().err
        assert m.net_dict['VDD'].pins[0].instance == 'mn'

    def test_gnd_and_zero_are_distinct_silk_nets(self, pdk, tmp_path):
        """SILENT-BY-DESIGN at device level, with a caveat. ngspice-42 and
        Xyce alias 'gnd' to '0' (verified: V1 a gnd / R1 a gnd / R2 a 0 —
        both resistors conduct), so a deck mixing them simulates correctly.
        silk's model keeps two nets; the port-level consequence (a POWER port
        on '0' invisible to INT-002) is the SILENT-GAP in TASK-36.

        STRUCT-011 (TASK-35) is not that check, and since TASK-88 it no
        longer leaves a trail on '0' either: the net named '0' is the
        simulator's global ground node, so the simulator is always its
        other party and a lone binding there is never one-ended (before
        TASK-88, '0' carrying only the source's return showed up here as a
        one-connection net — a false positive). silk still keeps 'gnd' and
        '0' as two distinct nets at the model level, which is precisely
        TASK-36's complaint."""
        m = _module(pdk, tmp_path)
        with m:
            pdk.nmos1p8('mn', d='out', g='in', s='gnd', b='gnd', w=1, l=0.15)
            sources.VoltageSource('v1', positive='out', negative='0', dc=1)
        assert {'0', 'gnd'} <= set(m.net_dict)
        assert _single_conn_nets(_lint(m)) == ['in']

    def test_two_sources_on_one_net_flagged_only_by_lint(self, pdk, tmp_path):
        """SILENT-BY-DESIGN (lint is the declared home; opt-in). Construction
        and netlist accept two voltage sources on one net; INT-001 names both
        drivers when lint is run."""
        m = _module(pdk, tmp_path)
        with m:
            pdk.nmos1p8('mn', d='out', g='in', s='gnd', b='gnd', w=1, l=0.15)
            sources.VoltageSource('v1', positive='in', negative='gnd', dc=1)
            sources.VoltageSource('v2', positive='in', negative='gnd', dc=0.5)
        _deck(m)
        msgs = _lint(m)['INT-001']
        assert msgs == ["Net 'in' has multiple drivers: v1.positive, v2.positive"]

    def test_source_with_both_terminals_on_one_net_is_silent_until_simulate(self, pdk, tmp_path):
        """SILENT-BY-DESIGN (simulator refuses). A source shorted across one
        net passes construction, netlist and lint; ngspice aborts with
        'instance vvdd is a shorted VSRC' and silk surfaces it as
        SimulationAborted (verified 2026-09-02). No rule targets the short
        itself; the STRUCT-011 findings below are the unrelated one-ended
        'out'/'in' nets of this deliberately minimal fragment."""
        m = _module(pdk, tmp_path)
        with m:
            pdk.nmos1p8('mn', d='out', g='in', s='gnd', b='gnd', w=1, l=0.15)
            sources.VoltageSource('vdd', positive='gnd', negative='gnd', dc=1.8)
        assert 'Vvdd gnd gnd' in _deck(m)
        assert _single_conn_nets(_lint(m)) == ['out', 'in']


# ---------------------------------------------------------------------------
# Column 2 — add_element on a primitive device
# ---------------------------------------------------------------------------

_MN_PARAMS = {'w': 1, 'l': 0.15}


class TestAddElementDevice:

    def test_empty_string_net_rejected_before_registration(self, pdk, tmp_path):
        """LOUD (was SILENT-GAP, fixed by TASK-75). The factory cell above is
        this same mistake on the other construction style; here it used to be
        bound as a falsy no-op and emitted as
        `XMmn out  gnd gnd sky130_fd_pr__nfet_01v8 ...` — one node short, which
        ngspice-42 accepted, reading the model name as the missing node. Both
        styles now route through the one validator, so the exception type and
        message are identical (full parity matrix in
        tests/test_net_value_validation.py)."""
        m = _module(pdk, tmp_path)
        with pytest.raises(ValueError, match=r"nmos1p8 'mn' pin 'g'"):
            m.add_element(pdk.nmos1p8, 'mn',
                          nets={'d': 'out', 'g': '', 's': 'gnd', 'b': 'gnd'},
                          parameters=_MN_PARAMS)
        assert 'mn' not in m.elem_dict
        assert '' not in m.net_dict

    def test_none_net_rejected_naming_instance_and_pin(self, pdk, tmp_path):
        """LOUD (was LOUD-but-cryptic, fixed by TASK-75). g=None used to die
        with a bare `AttributeError: 'NoneType' object has no attribute
        'casefold'` from inside net-name canonicalization, naming neither the
        instance nor the pin."""
        m = _module(pdk, tmp_path)
        with pytest.raises(TypeError, match=r"nmos1p8 'mn' pin 'g'"):
            m.add_element(pdk.nmos1p8, 'mn',
                          nets={'d': 'out', 'g': None, 's': 'gnd', 'b': 'gnd'},
                          parameters=_MN_PARAMS)
        assert 'mn' not in m.elem_dict

    def test_whitespace_bearing_net_rejected(self, pdk, tmp_path):
        """LOUD (was SILENT-GAP, fixed by TASK-75). `g='a b'` emitted FIVE
        node tokens for a four-pin device; add_net('a b') registered the net
        without complaint and is now rejected as well."""
        m = _module(pdk, tmp_path)
        with pytest.raises(ValueError, match=r"nmos1p8 'mn' pin 'g'"):
            m.add_element(pdk.nmos1p8, 'mn',
                          nets={'d': 'out', 'g': 'a b', 's': 'gnd', 'b': 'gnd'},
                          parameters=_MN_PARAMS)
        with pytest.raises(ValueError, match='add_net'):
            m.add_net('a b')
        assert m.net_dict == {}

    def test_nonexistent_pin_raises_naming_pin_and_element(self, pdk, tmp_path):
        """LOUD."""
        m = _module(pdk, tmp_path)
        with pytest.raises(CircuitError, match='Pin gate is not in defined pins for element mn'):
            m.add_element(pdk.nmos1p8, 'mn',
                          nets={'d': 'out', 'gate': 'in', 's': 'gnd', 'b': 'gnd'},
                          parameters=_MN_PARAMS)

    def test_pin_name_case_mismatch_raises(self, pdk, tmp_path):
        """LOUD. Pin names are exact; 'D' is not 'd' (unlike net names)."""
        m = _module(pdk, tmp_path)
        with pytest.raises(CircuitError, match='Pin D is not in defined pins'):
            m.add_element(pdk.nmos1p8, 'mn',
                          nets={'D': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
                          parameters=_MN_PARAMS)

    def test_pin_omitted_raises_at_netlist_naming_the_pin(self, pdk, tmp_path):
        """LOUD, now actionable (TASK-43 AC #1). Construction accepts a
        partial dict; generate_netlist raises. Pre-fix the message was
        'ERROR: pin in element mn is still undefined.' — it omitted WHICH
        pin, forcing the caller to diff the dict by hand; it now names 'g'."""
        m = _module(pdk, tmp_path)
        m.add_element(pdk.nmos1p8, 'mn', nets={'d': 'out', 's': 'gnd', 'b': 'gnd'},
                      parameters=_MN_PARAMS)
        with pytest.raises(CircuitError, match=r"pin g in element mn is still undefined"):
            m.generate_netlist()

    def test_positional_net_list_via_module_rejected_pointing_at_dict_form(self, pdk, tmp_path):
        """LOUD, now actionable (TASK-43 AC #2). Module.add_element never
        documented a positional form; pre-fix it fed the list straight into
        dict(nets), which raised an opaque 'ValueError: dictionary update
        sequence element #0 has length 3; 2 is required' for 3-char pin
        values, and would have silently MISPAIRED 2-char values instead
        (dict(['ab', 'cd']) == {'a': 'b', 'c': 'd'}) with no error at all.
        It now rejects the shape outright, pointing at the dict/kwargs
        forms, before anything is registered."""
        m = _module(pdk, tmp_path)
        with pytest.raises(CircuitError,
                           match=r"add_element\('mn'\): positional net lists are not supported"):
            m.add_element(pdk.nmos1p8, 'mn', nets=['out', 'in', 'gnd', 'gnd'],
                          parameters=_MN_PARAMS)
        assert 'mn' not in m.elem_dict

    def test_positional_net_list_via_graph_raises_on_length_mismatch(self, pdk, tmp_path):
        """LOUD in both directions (TASK-43 AC #3, fixed). Through
        CircuitGraph.add_element the positional form pairs by index against
        the device's own pin_list; pre-fix, too few nets raised a bare
        `IndexError` (message names nothing) and too many silently dropped
        the extras with no error and no created net. connect_element's new
        length check now fires for both, naming pins-count and nets-count,
        and (AC #4) leaves no partial instance behind either way."""
        m = _module(pdk, tmp_path)
        with pytest.raises(CircuitError, match=r'pins has 4 entries but nets has 3'):
            m._element_mgr.add_element(pdk.nmos1p8, 'mn', ['out', 'in', 'gnd'], _MN_PARAMS)
        assert 'mn' not in m.elem_dict
        m2 = _module(pdk, tmp_path, name=None)
        with pytest.raises(CircuitError, match=r'pins has 4 entries but nets has 5'):
            m2._element_mgr.add_element(pdk.nmos1p8, 'mn', ['out', 'in', 'gnd', 'gnd', 'extra'], _MN_PARAMS)
        assert 'mn' not in m2.elem_dict
        assert 'extra' not in m2.net_dict

    def test_tuple_list_form_works(self, pdk, tmp_path):
        """Sanity: the (pin, net) tuple-list form survives dict() and wires."""
        m = _module(pdk, tmp_path)
        m.add_element(pdk.nmos1p8, 'mn',
                      nets=[('d', 'out'), ('g', 'in'), ('s', 'gnd'), ('b', 'gnd')],
                      parameters=_MN_PARAMS)
        assert _device_line(_deck(m), 'mn').split()[1:5] == ['out', 'in', 'gnd', 'gnd']

    def test_strict_undeclared_net_raises(self, pdk, tmp_path):
        """LOUD, and now leaves no partial instance (TASK-43 AC #4). 'd' (a
        declared net) binds before the loop reaches 'g' (undeclared) and
        fails; pre-fix 'mn' stayed registered in elem_dict with pin 'd'
        already bound to 'out' and net 'out' carrying that binding. State
        before this call must equal state after the rejected call."""
        m = _module(pdk, tmp_path, strict=True)
        m.add_net('out'); m.add_net('gnd')
        elem_before = dict(m.elem_dict)
        out_pins_before = list(m.net_dict['out'].pins)
        with pytest.raises(UndeclaredNetError, match='net "in" is not declared'):
            m.add_element(pdk.nmos1p8, 'mn', nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
                          parameters=_MN_PARAMS)
        assert 'mn' not in m.elem_dict
        assert dict(m.elem_dict) == elem_before
        assert m.net_dict['out'].pins == out_pins_before
        assert 'in' not in m.net_dict

    def test_connect_module_unknown_port_raises(self, pdk, tmp_path):
        from silk.exceptions import CircuitError
        m = _module(pdk, tmp_path)
        m.add_module(_inverter(pdk, tmp_path), 'stage', vin='in', vout='out')
        with pytest.raises(CircuitError, match='Unknown port.*gate'):
            m.connect_module('stage', gate='x')


# ---------------------------------------------------------------------------
# Column 3 — add_element on a Module (submodule instance)
# ---------------------------------------------------------------------------

_INV_MAP = {'vin': 'a', 'vout': 'y', 'vdd': 'vdd', 'vss': 'gnd'}


class TestAddElementModule:

    def test_declared_port_omitted_raises_at_netlist_and_lint_flags_early(self, pdk, tmp_path):
        """LOUD at netlist (names instance and pin); STRUCT-001/002 flag it
        earlier if lint is run. Construction itself accepts the partial map."""
        top = _module(pdk, tmp_path)
        top.add_module(_inverter(pdk, tmp_path), 'u1', nets={'vin': 'a', 'vout': 'y', 'vdd': 'vdd'})
        lint = _lint(top)
        assert lint['STRUCT-001'] == ["Submodule instance 'u1' port 'vss' is not connected"]
        assert lint['STRUCT-002'] == ["Instance 'u1' port 'vss' (role=ground) is not connected"]
        with pytest.raises(CircuitError, match='In module instantiation u1, net for pin vss is undefined'):
            top.generate_netlist()

    @pytest.mark.parametrize('style', ['dict', 'kwargs'])
    def test_nonexistent_port_raises(self, pdk, tmp_path, style):
        """LOUD in both the dict and kwargs forms."""
        top = _module(pdk, tmp_path)
        inv = _inverter(pdk, tmp_path)
        with pytest.raises(CircuitError, match='Pin enable is not in defined pins for element u1'):
            if style == 'dict':
                top.add_module(inv, 'u1', nets={**_INV_MAP, 'enable': 'en'})
            else:
                top.add_module(inv, 'u1', enable='en', **_INV_MAP)

    def test_port_name_case_mismatch_raises(self, pdk, tmp_path):
        """LOUD. Port names are exact-case (only NET names are case-folded)."""
        top = _module(pdk, tmp_path)
        with pytest.raises(CircuitError, match='Pin VIN is not in defined pins'):
            top.add_module(_inverter(pdk, tmp_path), 'u1',
                            nets={'VIN': 'a', 'vout': 'y', 'vdd': 'vdd', 'vss': 'gnd'})

    def test_empty_port_value_rejected_before_registration(self, pdk, tmp_path):
        """LOUD (was SILENT-GAP, fixed by TASK-75). An empty port value on a
        submodule emitted a subcircuit line one node short; it is now a
        ValueError naming the submodule, the instance and the port, raised
        before the instance is registered and before its .subckt definition is
        recorded."""
        top = _module(pdk, tmp_path)
        with pytest.raises(ValueError, match=r"inv 'u1' pin 'vdd'"):
            top.add_module(_inverter(pdk, tmp_path), 'u1',
                            nets={**_INV_MAP, 'vdd': ''})
        assert 'u1' not in top.elem_dict
        assert top.net_dict == {}

    def test_two_out_ports_shorted_silent_until_lint(self, pdk, tmp_path):
        """SILENT-BY-DESIGN (lint is the home; opt-in). The deck emits both
        instances on 'y'; INT-001 names both drivers under lint."""
        top = _module(pdk, tmp_path)
        inv = _inverter(pdk, tmp_path)
        top.add_module(inv, 'u1', nets=_INV_MAP)
        top.add_module(inv, 'u2', nets={**_INV_MAP, 'vin': 'b'})
        deck = _deck(top)
        assert 'xu1 a y vdd gnd inv' in deck and 'xu2 b y vdd gnd inv' in deck
        assert _lint(top)['INT-001'] == ["Net 'y' has multiple drivers: u1.vout, u2.vout"]

    def test_power_port_on_ground_net_silent_until_lint(self, pdk, tmp_path):
        """SILENT-BY-DESIGN (lint is the home; opt-in). INT-002 catches a
        POWER port and a GROUND port sharing one silk net."""
        top = _module(pdk, tmp_path)
        top.add_module(_inverter(pdk, tmp_path), 'u1', nets={**_INV_MAP, 'vdd': 'gnd'})
        assert 'xu1 a y gnd gnd inv' in _deck(top)
        assert _lint(top)['INT-002'] == [
            "Net 'gnd' connects POWER and GROUND ports: u1.vdd(power), u1.vss(ground)"]

    def test_power_port_on_node_zero_is_reported_as_grounded_supply(self, pdk, tmp_path):
        """LOUD in lint (was the S4b SILENT-GAP, fixed by TASK-85). vdd port
        → '0', DUT gnd → 'gnd', source on 'vdd'. silk sees three unrelated
        nets and the simulator (where '0' is the global ground node) runs
        an unpowered DUT: v(y)=0.0 with vin=0 on the sky130 inverter, both
        backends, no error. Before the fix no rule knew '0' is ground, so
        INT-002 could not fire and the only report was STRUCT-011's four
        one-ended nets (TASK-35), none of which says "supply grounded".
        INT-002 now names the grounded supply outright; the deck is
        unchanged."""
        top = _module(pdk, tmp_path)
        top.add_module(_inverter(pdk, tmp_path), 'u1', nets={**_INV_MAP, 'vdd': '0'})
        with top:
            sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8)
        assert 'xu1 a y 0 gnd inv' in _deck(top)
        lint = _lint(top)
        assert lint['INT-002'] == [
            "Net '0' is the simulator ground node — supply is grounded: u1.vdd(power)"]
        assert _single_conn_nets(lint) == ['a', 'y', 'gnd', 'vdd']

    def test_single_in_port_with_no_driver_is_reported_by_lint(self, pdk, tmp_path):
        """LOUD in lint (was the T3 SILENT-GAP, fixed by TASK-35). The DUT
        input 'a' has one binding and no source. INT-004 still drops nets
        with < 2 relevant bindings — STRUCT-011 is what covers the
        single-binding shape, and it names the unread output 'y' too.
        ngspice-42 still simulates the deck (v(a)=1.02 V, v(y)=0.04 V);
        the deck is unchanged, only lint is louder."""
        top = _module(pdk, tmp_path)
        top.add_module(_inverter(pdk, tmp_path), 'u1', nets=_INV_MAP)
        with top:
            sources.VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8)
        _deck(top)
        assert len(top.net_dict['a'].pins) == 1
        assert 'INT-004' not in _lint(top)
        assert _wiring_rules(_lint(top)) == {'STRUCT-011': [
            "Net 'a' has only one connection: u1.vin(input) — "
            "nothing else is wired to it",
            "Net 'y' has only one connection: u1.vout(output) — "
            "nothing else is wired to it"]}

    def test_single_binding_net_suppressed_by_add_no_connect(self, pdk, tmp_path):
        """The documented STRUCT-011 escape hatch (TASK-35 AC #1): a net
        that is one-ended on purpose is silenced per net by
        Module.add_no_connect, which also records the reason in the
        report's 'Intentionally Unconnected' section."""
        top = _module(pdk, tmp_path)
        top.add_module(_inverter(pdk, tmp_path), 'u1', nets=_INV_MAP)
        with top:
            sources.VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8)
            sources.VoltageSource('vin', positive='a', negative='gnd', dc=0.9)
        assert _single_conn_nets(_lint(top)) == ['y']
        top.add_no_connect('y', reason='probe-only output, scoped externally')
        assert _wiring_rules(_lint(top)) == {}

    def test_strict_undeclared_net_raises_leaving_no_partial_instance(self, pdk, tmp_path):
        """LOUD (typed, actionable), partial-state wart fixed (TASK-43 AC #4).
        Pre-fix the instance stayed registered in elem_dict with whichever
        pins had bound before the failing one (measured: 'u1' present,
        vin.net == 'a', gnd.net is None) — state before != state after a
        rejected call. Now the failure purges the instance and every pin
        binding it acquired, so elem_dict/net_dict come back exactly as
        they were before the call."""
        top = _module(pdk, tmp_path, strict=True)
        for n in ('a', 'y', 'vdd'):
            top.add_net(n)
        elem_before = dict(top.elem_dict)
        nets_before = {name: list(net.pins) for name, net in top.net_dict.items()}
        with pytest.raises(UndeclaredNetError, match='net "gnd" is not declared'):
            top.add_module(_inverter(pdk, tmp_path), 'u1', nets=_INV_MAP)
        assert 'u1' not in top.elem_dict
        assert dict(top.elem_dict) == elem_before
        assert {name: list(net.pins) for name, net in top.net_dict.items()} == nets_before

    def test_rejected_first_module_instantiation_leaves_module_dict_without_entry(
            self, pdk, tmp_path):
        """A rejected placement leaves no orphan subcircuit registration."""
        top = _module(pdk, tmp_path, strict=True)
        for n in ('a', 'y', 'vdd'):
            top.add_net(n)
        inv = _inverter(pdk, tmp_path)
        with pytest.raises(UndeclaredNetError, match='net "gnd" is not declared'):
            top.add_module(inv, 'u1', nets=_INV_MAP)
        assert inv.module_name not in top.module_dict

    def test_rejected_second_module_instantiation_leaves_first_entry_intact(
            self, pdk, tmp_path):
        """A rejected second placement leaves the first registration intact.

        The count and source identities belong to the parent registry; the
        shared source Module itself carries no placement count.
        """
        top = _module(pdk, tmp_path, strict=True)
        for n in ('a', 'y', 'vdd', 'gnd', 'b'):
            top.add_net(n)
        inv = _inverter(pdk, tmp_path)
        top.add_module(inv, 'u1', nets=_INV_MAP)
        entry = top.module_dict[inv.module_name]
        number_before = entry['numberInstantiated']
        source_ids_before = set(entry['_source_ids'])

        with pytest.raises(UndeclaredNetError, match='net "ny2" is not declared'):
            top.add_module(inv, 'u2', nets={**_INV_MAP, 'vout': 'ny2'})

        assert 'u1' in top.elem_dict and 'u2' not in top.elem_dict
        entry_after = top.module_dict[inv.module_name]
        assert entry_after['numberInstantiated'] == number_before
        assert entry_after['_source_ids'] == source_ids_before

    def test_net_case_mismatch_merges_with_warning(self, pdk, tmp_path, capsys):
        """SILENT-BY-DESIGN (G16, warns). Port wired to 'VDD', source to
        'vdd': one net, one warning, deck keeps each author spelling."""
        top = _module(pdk, tmp_path)
        top.add_module(_inverter(pdk, tmp_path), 'u1', nets={**_INV_MAP, 'vdd': 'VDD'})
        with top:
            sources.VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8)
        assert 'vdd' not in top.net_dict and 'VDD' in top.net_dict
        assert _CASE_WARNING in capsys.readouterr().err
        assert {b.instance for b in top.net_dict['VDD'].pins} == {'u1', 'vdd'}

    def test_gnd_vs_zero_split_no_false_positive_on_ground_node(self, pdk, tmp_path):
        """SILENT-BY-DESIGN (SPICE semantics), model caveat. The DUT's
        ground port is on net 'gnd', the source returns on '0'. ngspice
        folds the NET name gnd≡0 and the deck simulates correctly
        (v(y)=1.8 with vin=0); Xyce does not fold. The related PORT hazard
        was TASK-36 and is now LOUD — see
        ``TestPortDeclaration.test_ground_port_name_rejected_at_emission``
        and ``tests/test_ground_port_guard.py``; the net-level split
        recorded here is unchanged and still silent. silk keeps two nets, so
        Net.pins / annotate_currents see half-nets and STRUCT-011 names the
        one-ended 'gnd'. Before TASK-85 lint also reported a FALSE POSITIVE
        on the ground node itself — INT-004 "Net '0' has no driver:
        vdd.negative(input), vin.negative(input)" — because nothing told it
        '0' is ground. INT-004 is now silent on '0'."""
        top = _module(pdk, tmp_path)
        top.add_module(_inverter(pdk, tmp_path), 'u1', nets=_INV_MAP)
        with top:
            sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8)
            sources.VoltageSource('vin', positive='a', negative='0', dc=0)
        _deck(top)
        assert {b.instance for b in top.net_dict['gnd'].pins} == {'u1'}
        assert {b.instance for b in top.net_dict['0'].pins} == {'vdd', 'vin'}
        assert 'INT-004' not in _lint(top)
        assert _wiring_rules(_lint(top)) == {
            'STRUCT-011': [
                "Net 'y' has only one connection: u1.vout(output) — "
                "nothing else is wired to it",
                "Net 'gnd' has only one connection: u1.vss(input) — "
                "nothing else is wired to it"]}

    def test_body_net_not_declared_as_port_reported_by_struct013(self, pdk, tmp_path):
        """REPORTED (TASK-93, default lint). The body wires the PMOS to 'vdd'
        but the port list has no vdd: '.subckt leaky vin vout vss' with an
        internal floating supply node. ngspice-42: v(y)=0.0, no error. This
        cell tracked the fix through three stages — SILENT-GAP before
        TASK-38, then the opt-in STRUCT-009 report, and since the TASK-93
        owner ruling the DEFAULT registry names the rail and the terminals on
        it (STRUCT-013 HIDDEN_SUPPLY_NET, which keys on the wiring). The
        opt-in STRUCT-009 still reports the same block from the other side —
        the missing POWER port declaration — and both are asserted here so a
        change to either contract surfaces in the matrix."""
        leaky = _inverter(pdk, tmp_path, name='leaky',
                          ports={k: v for k, v in _INV_PORTS.items() if k != 'vdd'})
        top = _module(pdk, tmp_path)
        top.add_module(leaky, 'u1', nets={'vin': 'a', 'vout': 'y', 'vss': 'gnd'})
        deck = _deck(top)
        assert '.subckt leaky vin vout vss' in deck
        assert 'XMmp vout vin vdd vdd' in deck
        assert 'vdd' not in top.net_dict
        assert _lint(leaky)['STRUCT-013'] == [
            "Subcircuit 'leaky' wires mp.s, mp.b to net 'vdd', which is not "
            "a declared port — hidden local supply"]
        assert _lint(leaky, include_optional=True)['STRUCT-009'] == [
            "Subcircuit 'leaky' contains devices but declares no port with "
            "role POWER (missing POWER supply port)"]
        # The parent is a top-level module (no module_name): not a subcircuit,
        # so neither rule has anything to say there — both findings live on
        # 'leaky', whose interface is what a parent would have to trust.
        assert 'STRUCT-013' not in _lint(top)
        assert 'STRUCT-009' not in _lint(top, include_optional=True)
        assert 'STRUCT-009' not in _lint(leaky)

    def test_declared_port_unused_by_body_is_silent(self, pdk, tmp_path):
        """SILENT-BY-DESIGN (SPICE-legal). A declared 'en' port no device
        touches stays in the .subckt header; the parent must still wire it.
        No rule looks inside a definition for unused ports; acceptable —
        it is harmless electrically (the parent net simply gains a no-op
        connection) and is a lint-quality item, not a wiring gap.

        The definition stays clean under STRUCT-011 too: 'en' is a declared
        port of `withen`, and a port net's second connection comes from
        whoever instantiates the block, so the rule exempts it. The parent
        has no such excuse — nothing there wires 'en' (or any other net) to
        a second pin, and all five show up."""
        ports = {**_INV_PORTS, 'en': Port.digital_input()}
        m = _inverter(pdk, tmp_path, name='withen', ports=ports)
        top = _module(pdk, tmp_path)
        top.add_module(m, 'u1', nets={**_INV_MAP, 'en': 'en'})
        deck = _deck(top)
        assert '.subckt withen vin vout vdd vss en' in deck
        assert _wiring_rules(_lint(m)) == {}
        assert _single_conn_nets(_lint(top)) == ['a', 'y', 'vdd', 'gnd', 'en']

    def test_connect_module_rewire_keeps_net_pins_consistent(self, pdk, tmp_path):
        """Sanity for the chokepoint: rewiring an instance port via
        connect_module moves the PortBinding, so the orphaned net is
        reported by STRUCT-003. (connect_source matches since TASK-39.)"""
        top = _module(pdk, tmp_path)
        top.add_module(_inverter(pdk, tmp_path), 'u1', nets=_INV_MAP)
        top.connect_module('u1', vout='a')
        assert top.net_dict['y'].pins == []
        assert {b.port for b in top.net_dict['a'].pins} == {'vin', 'vout'}
        assert _lint(top)['STRUCT-003'] == ["Net 'y' has no connections"]


# ---------------------------------------------------------------------------
# Column 4 — testbench portmap (add_module_under_test)
# ---------------------------------------------------------------------------

class TestTestbenchPortmap:

    def test_declared_port_omitted_raises_at_netlist(self, pdk, tmp_path):
        """LOUD at netlist time, naming instance 'dut' and the pin."""
        tb = _inverter(pdk, tmp_path).create_test('t')
        tb.add_module_under_test({'vin': 'a', 'vout': 'y', 'vdd': 'vdd'})
        tb.add(sources.VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8))
        with pytest.raises(CircuitError, match='In module instantiation dut, net for pin vss is undefined'):
            tb.netlist.generate_netlist()

    def test_nonexistent_port_raises(self, pdk, tmp_path):
        """LOUD."""
        tb = _inverter(pdk, tmp_path).create_test('t')
        with pytest.raises(CircuitError, match='Pin en is not in defined pins for element dut'):
            tb.add_module_under_test({**_INV_MAP, 'en': 'en'})

    def test_undriven_input_is_reported_by_lint(self, pdk, tmp_path):
        """LOUD in lint (was the D1 SILENT-GAP, fixed by TASK-35) — the
        testbench column of the floating-input row. Lint runs on the
        testbench's own netlist Module, so the DUT portmap gets the same
        STRUCT-011 treatment as a hand-built parent."""
        tb = _inverter(pdk, tmp_path).create_test('t')
        tb.add_module_under_test(_INV_MAP)
        tb.add(sources.VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8))
        tb_netlist_deck_path = tb.netlist.generate_netlist()
        assert 'xdut a y vdd gnd inv' in open(tb_netlist_deck_path).read()
        assert _wiring_rules(_lint(tb.netlist)) == {'STRUCT-011': [
            "Net 'a' has only one connection: dut.vin(input) — "
            "nothing else is wired to it",
            "Net 'y' has only one connection: dut.vout(output) — "
            "nothing else is wired to it"]}

    def test_power_port_on_gnd_flagged_by_lint_only(self, pdk, tmp_path):
        """SILENT-BY-DESIGN (lint is the home; opt-in). run() never lints."""
        tb = _inverter(pdk, tmp_path).create_test('t')
        tb.add_module_under_test({**_INV_MAP, 'vdd': 'gnd'})
        tb.netlist.generate_netlist()
        assert _lint(tb.netlist)['INT-002'] == [
            "Net 'gnd' connects POWER and GROUND ports: dut.vdd(power), dut.vss(ground)"]

    def test_strict_nets_inherited_from_dut_raises(self, pdk, tmp_path):
        """LOUD (FIXED TASK-40). Before the fix, the testbench Module built by
        _create_netlist was always strict_nets=False regardless of the DUT,
        so tb.netlist.strict_nets was False and 'typo_net' silently landed in
        tb.netlist.net_dict with no error. Now the testbench inherits the
        DUT's strict_nets by default, so the same portmap typo raises
        UndeclaredNetError naming the net — same type/message shape a strict
        standalone Module raises from _ensure_net."""
        dut = _inverter(pdk, tmp_path)
        dut.strict_nets = True
        dut._element_mgr.strict_nets = True
        tb = dut.create_test('t')
        assert tb.strict_nets is True
        with pytest.raises(UndeclaredNetError, match='net "typo_net" is not declared'):
            tb.add_module_under_test({**_INV_MAP, 'vin': 'typo_net'})
        assert tb.netlist.strict_nets is True

    def test_strict_nets_explicit_override_false_on_strict_dut(self, pdk, tmp_path):
        """create_test(strict_nets=False) overrides a strict DUT: the
        testbench stays permissive even though the DUT itself is strict."""
        dut = _inverter(pdk, tmp_path)
        dut.strict_nets = True
        dut._element_mgr.strict_nets = True
        tb = dut.create_test('t', strict_nets=False)
        assert tb.strict_nets is False
        tb.add_module_under_test({**_INV_MAP, 'vin': 'typo_net'})   # no raise
        assert tb.netlist.strict_nets is False
        assert 'typo_net' in tb.netlist.net_dict

    def test_strict_nets_explicit_override_true_on_nonstrict_dut(self, pdk, tmp_path):
        """create_test(strict_nets=True) overrides a non-strict DUT: the
        testbench can be made strict even when the DUT is not."""
        tb = _inverter(pdk, tmp_path).create_test('t', strict_nets=True)
        assert tb.strict_nets is True
        with pytest.raises(UndeclaredNetError, match='net "typo_net" is not declared'):
            tb.add_module_under_test({**_INV_MAP, 'vin': 'typo_net'})

    def test_strict_nets_default_inheritance_stays_permissive(self, pdk, tmp_path):
        """No regression: a non-strict DUT still gets a non-strict testbench
        by default (same as before TASK-40 for the common, non-strict case)."""
        tb = _inverter(pdk, tmp_path).create_test('t')
        assert tb.strict_nets is False
        tb.add_module_under_test({**_INV_MAP, 'vin': 'typo_net'})   # no raise
        assert tb.netlist.strict_nets is False
        assert 'typo_net' in tb.netlist.net_dict

    def test_net_case_mismatch_merges_with_warning(self, pdk, tmp_path, capsys):
        """SILENT-BY-DESIGN (G16, warns) — same canonicalizer as the parent."""
        tb = _inverter(pdk, tmp_path).create_test('t')
        tb.add_module_under_test({**_INV_MAP, 'vdd': 'VDD'})
        tb.add(sources.VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8))
        assert sorted(tb.netlist.net_dict) == ['VDD', 'a', 'gnd', 'y']
        assert _CASE_WARNING in capsys.readouterr().err

    def test_source_on_dut_output_port_flagged_by_lint_only(self, pdk, tmp_path):
        """SILENT-BY-DESIGN (lint is the home). A stimulus on an OUTPUT port
        is INT-001 (source terminals bind with direction='output')."""
        tb = _inverter(pdk, tmp_path).create_test('t')
        tb.add_module_under_test(_INV_MAP)
        tb.add(sources.VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8))
        tb.add(sources.VoltageSource('vin', positive='a', negative='gnd', dc=0.9))
        tb.add(sources.VoltageSource('vy', positive='y', negative='gnd', dc=0.9))
        tb.netlist.generate_netlist()
        assert _lint(tb.netlist)['INT-001'] == ["Net 'y' has multiple drivers: dut.vout, vy.positive"]


# ---------------------------------------------------------------------------
# connect_source rewires
# ---------------------------------------------------------------------------

class TestConnectSource:

    def _with_source(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        with m:
            sources.VoltageSource('v1', positive='a', negative='gnd', dc=1)
        return m

    def test_undeclared_net_raises_even_when_not_strict(self, pdk, tmp_path):
        """LOUD. Stricter than connect_module (which auto-creates); keep."""
        m = self._with_source(pdk, tmp_path)
        with pytest.raises(UndeclaredNetError, match='Net "nonet" is undefined'):
            m.connect_source('v1', ['positive'], ['nonet'])

    def test_length_mismatch_raises_naming_both_counts(self, pdk, tmp_path):
        """LOUD, previously SILENT (TASK-43, flagged in TASK-39's handoff).
        Pre-fix the write loop was a bare ``zip(pins, nets)``, which
        TRUNCATES to the shorter sequence with no error at all: measured,
        ``connect_source('v1', ['positive'], ['a', 'b'])`` rewired
        'positive' and silently dropped 'b'. Now both directions raise
        before anything is written, naming both lengths — matching
        connect_element's new length check."""
        m = self._with_source(pdk, tmp_path)
        m.add_net('b')
        with pytest.raises(CircuitError, match=r'pins has 1 entries but nets has 2'):
            m.connect_source('v1', ['positive'], ['a', 'b'])
        assert m.sig_sources_dict['v1'].positive == 'a'
        with pytest.raises(CircuitError, match=r'pins has 2 entries but nets has 1'):
            m.connect_source('v1', ['positive', 'negative'], ['b'])
        assert m.sig_sources_dict['v1'].positive == 'a'
        assert m.sig_sources_dict['v1'].negative == 'gnd'

    def test_unknown_pin_raises_naming_the_terminals(self, pdk, tmp_path):
        """LOUD (was SILENT-GAP, fixed in TASK-39). Pre-fix a misspelled
        terminal passed a ``hasattr`` guard and did nothing: ``positive``
        stayed ``'a'`` and no error was raised, unlike ``connect_element``.
        Now it raises before any write, naming the valid terminals."""
        m = self._with_source(pdk, tmp_path)
        m.add_net('b')
        with pytest.raises(CircuitError,
                           match=r'Pin "plus" is not a terminal of source "v1"; '
                                 r'valid terminals: positive, negative\.'):
            m.connect_source('v1', ['plus'], ['b'])
        assert m.sig_sources_dict['v1'].positive == 'a'
        assert m.net_dict['b'].pins == []

    def test_unknown_pin_rejected_before_any_terminal_is_written(self, pdk, tmp_path):
        """A call mixing one good and one bad pin is rejected whole: the
        good rewire is not applied either (source and Net.pins untouched)."""
        m = self._with_source(pdk, tmp_path)
        m.add_net('b')
        with pytest.raises(CircuitError, match='Pin "minus"'):
            m.connect_source('v1', ['positive', 'minus'], ['b', 'b'])
        assert m.sig_sources_dict['v1'].positive == 'a'
        assert [(p.instance, p.port) for p in m.net_dict['a'].pins] == [('v1', 'positive')]
        assert m.net_dict['b'].pins == []

    def test_rewire_moves_net_pins_with_the_deck(self, pdk, tmp_path):
        """LOUD (was SILENT-GAP, fixed in TASK-39). Pre-fix the deck followed
        the rewire (``Vv1 b gnd``) but ``Net.pins`` did not: net ``a`` still
        listed ``v1.positive``, net ``b`` had no pins, and STRUCT-003 reported
        ``"Net 'b' has no connections"`` — the wrong net. Now the binding
        moves through ``_bind_pin_to_net`` and the orphaned net ``a`` is the
        one reported, matching ``connect_element``."""
        m = self._with_source(pdk, tmp_path)
        m.add_net('b')
        m.connect_source('v1', ['positive'], ['b'])
        assert m.sig_sources_dict['v1'].positive == 'b'
        assert 'Vv1 b gnd' in _deck(m)
        assert m.net_dict['a'].pins == []
        assert [(p.instance, p.port, p.direction, p.kind) for p in m.net_dict['b'].pins] \
            == [('v1', 'positive', 'output', 'source')]
        assert _lint(m)['STRUCT-003'] == ["Net 'a' has no connections"]

    def test_rewire_to_same_net_does_not_duplicate_binding(self, pdk, tmp_path):
        """Idempotent: re-connecting a terminal to the net it already sits
        on leaves exactly one PortBinding (the chokepoint's same-net no-op)."""
        m = self._with_source(pdk, tmp_path)
        m.connect_source('v1', ['positive', 'negative'], ['a', 'gnd'])
        assert [(p.instance, p.port) for p in m.net_dict['a'].pins] == [('v1', 'positive')]
        assert [(p.instance, p.port) for p in m.net_dict['gnd'].pins] == [('v1', 'negative')]

    def test_controlled_source_rewire_and_terminal_names(self, pdk, tmp_path):
        """A VCVS rewires its four terminals through the same path; the
        unknown-pin error lists the controlled-source terminal names, not
        positive/negative."""
        m = _module(pdk, tmp_path)
        for n in ('a', 'b', 'c', 'd', 'e'):
            m.add_net(n)
        m.add_source(sources.VoltageControlledVoltageSource(
            'e1', out_pos='a', out_neg='gnd', ctrl_pos='b', ctrl_neg='c', gain=10))
        with pytest.raises(CircuitError,
                           match='valid terminals: out_pos, out_neg, ctrl_pos, ctrl_neg'):
            m.connect_source('e1', ['positive'], ['d'])
        m.connect_source('e1', ['out_pos', 'ctrl_neg'], ['d', 'e'])
        assert m.net_dict['a'].pins == [] and m.net_dict['c'].pins == []
        assert [(p.port, p.direction) for p in m.net_dict['d'].pins] == [('out_pos', 'output')]
        assert [(p.port, p.direction) for p in m.net_dict['e'].pins] == [('ctrl_neg', 'input')]

    def test_legacy_dict_source_unknown_pin_raises(self, pdk, tmp_path):
        """The legacy dict-format source path gets the same error (pre-fix
        it too was silently skipped)."""
        m = _module(pdk, tmp_path)
        m.add_net('vdd')
        m.sig_sources_dict['vleg'] = {'type': 'VoltageSource', 'pinNets': ['vdd', 'gnd'],
                                      'params': [1.8, None, None, None, None]}
        with pytest.raises(CircuitError, match='valid terminals: positive, negative'):
            m.connect_source('vleg', ['plus'], ['vdd'])
        assert m.sig_sources_dict['vleg']['pinNets'] == ['vdd', 'gnd']

    @pytest.mark.parametrize('skip', [False, True], ids=['default', '_skipCheck'])
    def test_unknown_source_raises_pynetlisterror_regardless_of_skipcheck(
            self, pdk, tmp_path, skip):
        """LOUD, now typed on both paths (TASK-100). ``_skipCheck=True`` was
        dead for an unknown name: it skipped the guard only for the
        ``sig_sources_dict[sourceName]`` lookup on the next line to raise a
        bare ``KeyError('nosuch')`` — measured pre-fix. The flag stays in the
        signature (the refactor golden pins it, as for ``connect_element``)
        but an unknown source now raises the same ``CircuitError`` either
        way; nothing is written."""
        m = self._with_source(pdk, tmp_path)
        with pytest.raises(CircuitError, match='Source "nosuch" does not exist'):
            m.connect_source('nosuch', ['positive'], ['a'], _skipCheck=skip)
        assert _pins(m, 'a') == [('v1', 'positive', 'output', 'source')]


class TestUpdateSource:

    def _with_source(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        with m:
            sources.VoltageSource('v1', positive='a', negative='gnd', dc=1)
        return m

    @pytest.mark.parametrize('skip', [False, True], ids=['default', '_skipCheck'])
    def test_unknown_source_raises_pynetlisterror_regardless_of_skipcheck(
            self, pdk, tmp_path, skip):
        """LOUD, now typed on both paths (TASK-98's scope addition, filed by
        the TASK-100 implementer as the same dead-flag bug on
        ``update_source``). ``_skipCheck=True`` was dead for an unknown
        name: it skipped the guard only for the
        ``sig_sources_dict[sourceName]`` lookup on the next line to raise a
        bare ``KeyError('nosuch')`` — measured pre-fix. The flag stays in
        the signature (the refactor golden pins it) but an unknown source
        now raises the same ``CircuitError`` either way; nothing is
        written."""
        m = self._with_source(pdk, tmp_path)
        with pytest.raises(CircuitError, match='Source "nosuch" does not exist'):
            m.update_source('nosuch', ['dc'], [3.3], _skipCheck=skip)
        assert m.sig_sources_dict['v1'].dc == 1


# ---------------------------------------------------------------------------
# Direct attribute rewire on an attached source (TASK-90)
# ---------------------------------------------------------------------------

def _pins(module, net):
    return [(p.instance, p.port, p.direction, p.kind) for p in module.net_dict[net].pins]


def _binding_counts(module):
    """(instance, port) -> number of PortBinding entries across every net."""
    counts = {}
    for net in module.net_dict.values():
        for p in net.pins:
            counts[(p.instance, p.port)] = counts.get((p.instance, p.port), 0) + 1
    return counts


class TestSourceAttributeRewire:
    """``m.sig_sources_dict['v1'].positive = 'b'`` — the other rewire entry
    point, found while closing TASK-39. Devices had this since TASK-41
    (``mn1.d = 'b'`` re-binds through ``AnalogElement.__setattr__``); a source
    had no module back-reference, so only ``connect_source`` moved
    ``Net.pins``."""

    def _with_source(self, pdk, tmp_path, strict=False):
        m = _module(pdk, tmp_path, strict=strict)
        m.add_net('b')
        with m:
            sources.VoltageSource('v1', positive='a', negative='gnd', dc=1)
        return m

    def test_attribute_rewire_moves_net_pins_with_the_deck(self, pdk, tmp_path):
        """LOUD (was SILENT-GAP, fixed in TASK-90). Pre-fix, probed after
        TASK-39 on an attached source: ``src.positive = 'b'`` changed the
        attribute (the deck emitted ``Vv1 b gnd``) but ``a.pins`` still held
        ``('v1', 'positive')`` and ``b.pins == []`` — STRUCT-003 reported
        ``"Net 'b' has no connections"``, the wrong net, exactly the stale
        state ``connect_source`` had before TASK-39. Now the binding moves
        through ``_bind_source_terminal`` with the driver direction
        ``add_source`` assigns."""
        m = self._with_source(pdk, tmp_path)
        src = m.sig_sources_dict['v1']
        src.positive = 'b'
        assert src.positive == 'b'
        assert 'Vv1 b gnd' in _deck(m)
        assert m.net_dict['a'].pins == []
        assert _pins(m, 'b') == [('v1', 'positive', 'output', 'source')]
        assert _lint(m)['STRUCT-003'] == ["Net 'a' has no connections"]

    def test_return_terminal_rewire_binds_as_input(self, pdk, tmp_path):
        """The direction rule is add_source's: ``negative`` is the return
        path and binds as ``'input'`` (else every source's ground return
        would look like a driver to INT-001)."""
        m = self._with_source(pdk, tmp_path)
        m.sig_sources_dict['v1'].negative = 'b'
        assert m.net_dict['gnd'].pins == []
        assert _pins(m, 'b') == [('v1', 'negative', 'input', 'source')]

    def test_controlled_source_terminals_follow_the_same_direction_rule(self, pdk, tmp_path):
        """VCVS: ``out_pos`` drives (output); ``out_neg``/``ctrl_pos``/
        ``ctrl_neg`` bind as inputs — the same table ``add_source`` and
        ``connect_source`` use (``_SOURCE_OUTPUT_TERMINALS``)."""
        m = _module(pdk, tmp_path)
        for n in ('a', 'b', 'c', 'd', 'e', 'f', 'g'):
            m.add_net(n)
        e1 = sources.VoltageControlledVoltageSource(
            'e1', out_pos='a', out_neg='gnd', ctrl_pos='b', ctrl_neg='c', gain=10)
        m.add_source(e1)
        e1.out_pos, e1.out_neg, e1.ctrl_pos, e1.ctrl_neg = 'd', 'e', 'f', 'g'
        for old in ('a', 'gnd', 'b', 'c'):
            assert m.net_dict[old].pins == []
        assert _pins(m, 'd') == [('e1', 'out_pos', 'output', 'source')]
        assert _pins(m, 'e') == [('e1', 'out_neg', 'input', 'source')]
        assert _pins(m, 'f') == [('e1', 'ctrl_pos', 'input', 'source')]
        assert _pins(m, 'g') == [('e1', 'ctrl_neg', 'input', 'source')]

    @pytest.mark.parametrize('bad,exc', [('', ValueError), ('a b', ValueError),
                                         (None, TypeError), (3, TypeError)],
                             ids=['empty', 'whitespace', 'None', 'number'])
    def test_rejected_value_changes_nothing(self, pdk, tmp_path, bad, exc):
        """Order per terminal is validate, then write, then bind: a value
        TASK-76 rejects raises before either the attribute or ``Net.pins``
        is touched, so the old binding survives intact."""
        m = self._with_source(pdk, tmp_path)
        src = m.sig_sources_dict['v1']
        with pytest.raises(exc, match=r"VoltageSource 'v1' terminal 'positive'"):
            src.positive = bad
        assert src.positive == 'a'
        assert _pins(m, 'a') == [('v1', 'positive', 'output', 'source')]
        assert m.net_dict['b'].pins == []
        assert 'Vv1 a gnd' in _deck(m)

    def test_rewire_to_same_net_does_not_duplicate_binding(self, pdk, tmp_path):
        """Idempotent, like the connect_source cell: re-assigning the net a
        terminal already sits on leaves exactly one PortBinding."""
        m = self._with_source(pdk, tmp_path)
        src = m.sig_sources_dict['v1']
        src.positive = 'a'
        src.negative = 'gnd'
        assert _binding_counts(m) == {('v1', 'positive'): 1, ('v1', 'negative'): 1}

    def test_rewire_with_a_net_object_stores_its_name_and_binds(self, pdk, tmp_path):
        """A ``Net`` is accepted where a name is (TASK-76 stores its name);
        the re-bind sees the name too."""
        m = self._with_source(pdk, tmp_path)
        m.sig_sources_dict['v1'].positive = m.net_dict['b']
        assert m.sig_sources_dict['v1'].positive == 'b'
        assert m.net_dict['a'].pins == []
        assert _pins(m, 'b') == [('v1', 'positive', 'output', 'source')]

    def test_unattached_source_keeps_plain_attribute_semantics(self, pdk, tmp_path):
        """AC #2. A source never passed to ``add_source`` has no module to
        bind on: terminal writes are validated (TASK-76) and stored, nothing
        else. Attaching it afterwards binds every terminal exactly once, on
        the nets it holds THEN — the pre-attach net ``a`` never sees it."""
        m = _module(pdk, tmp_path)
        for n in ('a', 'gnd', 'b'):
            m.add_net(n)
        src = sources.VoltageSource('v1', positive='a', negative='gnd', dc=1)
        assert src._module is None
        src.positive = 'b'
        assert src.positive == 'b' and src._module is None
        assert m.net_dict['a'].pins == [] and m.net_dict['b'].pins == []
        m.add_source(src)
        assert src._module is m
        assert m.net_dict['a'].pins == []
        assert _pins(m, 'b') == [('v1', 'positive', 'output', 'source')]
        assert _pins(m, 'gnd') == [('v1', 'negative', 'input', 'source')]
        assert _binding_counts(m) == {('v1', 'positive'): 1, ('v1', 'negative'): 1}

    def test_add_source_is_the_attach_point_for_every_construction_style(self, pdk, tmp_path):
        """``with m:`` auto-registration, explicit ``add_source`` and the
        unified ``m.add(src)`` all end in ``add_source``, so all three leave
        the back-reference set."""
        m = _module(pdk, tmp_path)
        with m:
            v_with = sources.VoltageSource('v_with', positive='a', negative='gnd', dc=1)
        v_add = sources.VoltageSource('v_add', positive='a', negative='gnd', dc=1)
        m.add_source(v_add)
        v_uni = sources.VoltageSource('v_uni', positive='a', negative='gnd', dc=1)
        m.add(v_uni)
        assert all(s._module is m for s in (v_with, v_add, v_uni))

    def test_connect_source_still_binds_each_terminal_exactly_once(self, pdk, tmp_path):
        """``connect_source`` now writes the attribute (which re-binds) and
        then calls ``_bind_source_terminal`` itself; the chokepoint's
        same-net no-op makes the second call inert, so no terminal is
        double-bound and no net carries a duplicate entry."""
        m = self._with_source(pdk, tmp_path)
        m.connect_source('v1', ['positive', 'negative'], ['b', 'a'])
        assert _binding_counts(m) == {('v1', 'positive'): 1, ('v1', 'negative'): 1}
        assert _pins(m, 'b') == [('v1', 'positive', 'output', 'source')]
        assert _pins(m, 'a') == [('v1', 'negative', 'input', 'source')]
        assert m.net_dict['gnd'].pins == []
        # And back again via the attribute path: still one entry per terminal.
        m.sig_sources_dict['v1'].positive = 'a'
        assert _binding_counts(m) == {('v1', 'positive'): 1, ('v1', 'negative'): 1}
        assert [p.port for p in m.net_dict['a'].pins] == ['negative', 'positive']

    def test_deepcopy_and_pickle_rebind_the_copy_to_itself(self, pdk, tmp_path):
        """The back-reference is a plain strong reference, as on a device:
        ``Module.__deepcopy__``/``__reduce_ex__`` copy the source with the
        module as the memo root, so the copy's source points at the copy
        and a rewire on the copy leaves the original's ``Net.pins`` alone
        (and vice versa)."""
        import copy
        import pickle
        m = self._with_source(pdk, tmp_path)
        for restored in (copy.deepcopy(m), pickle.loads(pickle.dumps(m))):
            assert restored.sig_sources_dict['v1']._module is restored
            restored.sig_sources_dict['v1'].positive = 'b'
            assert restored.net_dict['a'].pins == []
            assert _pins(restored, 'b') == [('v1', 'positive', 'output', 'source')]
            assert _pins(m, 'a') == [('v1', 'positive', 'output', 'source')]
            assert m.net_dict['b'].pins == []
        assert m.sig_sources_dict['v1']._module is m

    def test_module_pickle_does_not_drag_a_second_module_along(self, pdk, tmp_path):
        """The reference points at the OWNING module only: pickling a source
        that sits in one module produces one Module in the stream, not a
        chain (the same guarantee a device carries)."""
        import pickle
        m = self._with_source(pdk, tmp_path)
        n_before = pickle.dumps(m).count(b'_reconstruct_module')
        assert n_before == 1
        m.sig_sources_dict['v1'].positive = 'b'
        assert pickle.dumps(m).count(b'_reconstruct_module') == 1


# ---------------------------------------------------------------------------
# strict_nets rejection of an attribute rewire is atomic (TASK-100)
# ---------------------------------------------------------------------------

def _binding_state(module, inst, port):
    """Everything a rejected rewire must leave alone on the Net side: every
    net's pins plus the reverse-index entry for the rewired pin."""
    return ({n: [(p.instance, p.port) for p in net.pins]
             for n, net in module.net_dict.items()},
            module._element_mgr._pin_to_net.get((inst, port)))


class TestStrictRewireAtomicity:
    """``mn1.d = 'undeclared'`` / ``src.positive = 'undeclared'`` on a
    ``strict_nets`` module. The ``__setattr__`` twin of the partial-state
    class TASK-43 closed for ``add_element`` / ``connect_*``: pre-fix the
    chokepoint dropped the old ``Net.pins`` entry BEFORE ``_ensure_net``
    raised, and both owners had already written the attribute. Fixed once in
    ``_bind_pin_to_net`` (gate first) with the owner write moved after the
    bind in both families."""

    def test_device_pin_rewire_rejected_under_strict_changes_nothing(self, pdk, tmp_path):
        """LOUD, now atomic (TASK-100 AC #1, device). Measured pre-fix with
        nets out/in/gnd declared and ``mn1`` on ``d='out'``: after the raise,
        ``mn1.nets['d'] == 'undeclared'`` (the deck would name a net the
        model does not have), ``net_dict['out'].pins == []`` (was
        ``[('mn1', 'd')]``) and ``_pin_to_net[('mn1', 'd')] == 'out'`` —
        three views of one pin that disagreed. Worse, rewiring back with
        ``mn1.d = 'out'`` was then the chokepoint's same-net no-op (the stale
        reverse entry still said 'out'), so ``out.pins`` never recovered.
        Now before == after, and the rewire-back is a plain rewire."""
        m = _module(pdk, tmp_path, strict=True)
        for n in ('out', 'in', 'gnd'):
            m.add_net(n)
        with m:
            pdk.nmos1p8('mn1', d='out', g='in', s='gnd', b='gnd', w=1, l=0.15)
        mn1 = m.elem_dict['mn1']['device']
        before = _binding_state(m, 'mn1', 'd')
        assert before == ({'out': [('mn1', 'd')], 'in': [('mn1', 'g')],
                           'gnd': [('mn1', 's'), ('mn1', 'b')]}, 'out')
        with pytest.raises(UndeclaredNetError, match='net "undeclared" is not declared'):
            mn1.d = 'undeclared'
        assert mn1.nets['d'] == 'out'
        assert _binding_state(m, 'mn1', 'd') == before
        assert 'undeclared' not in m.net_dict
        # A later legitimate rewire still works, and 'out' keeps exactly one
        # binding for the pin — no duplicate, no orphan.
        mn1.d = 'in'
        assert _binding_state(m, 'mn1', 'd')[0]['out'] == []
        mn1.d = 'out'
        assert _binding_state(m, 'mn1', 'd') == before

    def test_source_terminal_rewire_rejected_under_strict_changes_nothing(self, pdk, tmp_path):
        """LOUD, now atomic (TASK-100 AC #1, source). Measured pre-fix with
        nets a/gnd declared and ``v1`` on ``positive='a'``: after the raise,
        ``src.positive == 'undeclared'``, ``net_dict['a'].pins == []`` (was
        ``[('v1', 'positive')]``) and ``_pin_to_net[('v1', 'positive')] ==
        'a'`` — the same residue as the device cell, because TASK-90 made
        ``SourceBase.__setattr__`` mirror the device order (write, then
        bind). Both now bind first and write only when the bind returns."""
        m = _module(pdk, tmp_path, strict=True)
        m.add_net('a'); m.add_net('gnd')
        with m:
            sources.VoltageSource('v1', positive='a', negative='gnd', dc=1)
        src = m.sig_sources_dict['v1']
        before = _binding_state(m, 'v1', 'positive')
        assert before == ({'a': [('v1', 'positive')], 'gnd': [('v1', 'negative')]}, 'a')
        with pytest.raises(UndeclaredNetError, match='net "undeclared" is not declared'):
            src.positive = 'undeclared'
        assert src.positive == 'a'
        assert _binding_state(m, 'v1', 'positive') == before
        assert 'undeclared' not in m.net_dict
        assert 'Vv1 a gnd' in _deck(m)

    def test_non_strict_rewire_order_is_unchanged(self, pdk, tmp_path):
        """Guard on the reorder: with ``strict_nets=False`` the chokepoint
        auto-creates the net (now before the stale binding is dropped) and
        the owner write follows — the observable result is the TASK-41 /
        TASK-90 behaviour exactly."""
        m = _module(pdk, tmp_path)
        with m:
            pdk.nmos1p8('mn1', d='out', g='in', s='gnd', b='gnd', w=1, l=0.15)
            sources.VoltageSource('v1', positive='a', negative='gnd', dc=1)
        m.elem_dict['mn1']['device'].d = 'fresh_d'
        m.sig_sources_dict['v1'].positive = 'fresh_s'
        assert m.elem_dict['mn1']['device'].nets['d'] == 'fresh_d'
        assert m.sig_sources_dict['v1'].positive == 'fresh_s'
        assert m.net_dict['out'].pins == [] and m.net_dict['a'].pins == []
        assert _binding_state(m, 'mn1', 'd')[1] == 'fresh_d'
        assert _pins(m, 'fresh_d') == [('mn1', 'd', None, 'device')]
        assert _pins(m, 'fresh_s') == [('v1', 'positive', 'output', 'source')]


# ---------------------------------------------------------------------------
# Port declaration
# ---------------------------------------------------------------------------

class TestPortDeclaration:

    def test_one_tuple_raises(self, pdk, tmp_path):
        """LOUD."""
        m = _module(pdk, tmp_path, 'x')
        with pytest.raises(ValueError, match='expected \\(PortDir, PortRole\\) 2-tuple, got 1-element tuple'):
            m.define_module_ports({'a': (PortDir.IN,)})

    def test_case_duplicate_port_names_raise(self, pdk, tmp_path):
        """LOUD (was SILENT-GAP, fixed in TASK-37). 'Vin' and 'vin' would be
        two silk ports and ONE SPICE node; the second parent connection is
        dropped by the simulator (ngspice-42: i(Vb)=0 on a resistor subckt;
        sky130 inverter deck ran with no warning). The G16 canonicalizer folds
        NETS onto a canonical spelling; ports cannot be folded (that would
        change the subckt signature), so the declaration raises and names both
        spellings."""
        cd = _module(pdk, tmp_path, 'casedup')
        with pytest.raises(DuplicateNameError) as exc:
            cd.define_module_ports({'Vin': Port.analog_input(), 'vin': Port.analog_input(),
                                    'vout': Port.analog_output(), 'vdd': Port.supply(),
                                    'vss': Port.ground()})
        assert "'vin'" in str(exc.value) and "'Vin'" in str(exc.value)
        assert 'vin' not in cd.pin_list          # nothing half-declared under the clash

    def test_ground_port_name_rejected_at_emission(self, pdk, tmp_path):
        """LOUD (was SILENT-GAP, fixed in TASK-36). A port named 'gnd' is an
        unportable interface: ngspice-42 folds the port NAME onto the global
        node 0, Xyce 7.10 folds nothing. Node 0 pierces subcircuit
        boundaries, so the fold discards whatever the parent wired to that
        port position — measured on one byte-identical deck, ngspice
        I(V1) = -1.0 mA with V(mid) = 0 against Xyce -0.5 mA with
        V(mid) = 0.5, and ZERO warnings from either.

        The trigger is the DECLARATION, not the mapping: a reusable block
        cannot know how it will be wired later. Two mechanisms — STRUCT-012
        at lint time, and this guard at emission, because nothing semantic
        runs at simulate time. Focused coverage is in
        ``tests/test_ground_port_guard.py``; this cell is the matrix entry.
        """
        m = _module(pdk, tmp_path, 'gndport')
        m.define_module_ports({**{k: v for k, v in _INV_PORTS.items()
                                  if k != 'vss'},
                               'gnd': Port.ground()})
        with m:
            pdk.nmos1p8('mn', d='vout', g='vin', s='gnd', b='gnd', w=1, l=0.15)
        # Declaring it is allowed; emitting it is not.
        assert 'gnd' in m.pin_list
        with pytest.raises(CircuitError, match="declares a port named 'gnd'"):
            m.generate_netlist(subckt=True)
        assert _lint(m)['STRUCT-012'] == [
            "block 'gndport' declares a port named 'gnd': ngspice folds this "
            "port name to the global SPICE node 0, so the block's interface "
            "is not portable — a parent's wiring to that port is silently "
            "discarded on ngspice, while Xyce honours it, and no "
            "simulator warns. The declaration is the defect: a reusable "
            "block cannot know how it will be wired later. Rename the port "
            "to 'vss' and wire it to the parent's ground net when ground is "
            "intended; keep actual ground connections named '0'."]

    def test_raw_direction_string_normalized_or_rejected(self, pdk, tmp_path):
        """LOUD (was SILENT, folded into TASK-37). Direction strings now go
        through Port._parse_direction: short and long spellings (any case)
        normalize to PortDir, and an unknown spelling raises instead of being
        stored verbatim and matching no direction rule."""
        m = _module(pdk, tmp_path, 'x')
        m.define_module_ports({'a': 'input', 'c': 'INPUT', 'd': 'out'})
        assert m.pin_list['a'].direction is PortDir.IN
        assert m.pin_list['c'].direction is PortDir.IN
        assert m.pin_list['d'].direction is PortDir.OUT
        with pytest.raises(ValueError, match="port 'b': unknown direction 'inp'"):
            m.define_module_ports({'b': 'inp'})
        assert 'b' not in m.pin_list

    def test_redeclare_raises_instead_of_overwriting(self, pdk, tmp_path):
        """LOUD (was SILENT, folded into TASK-37). A redeclaration that would
        change the port raises and shows old vs new; an identical
        redeclaration is a no-op that keeps the existing pin (so a wired
        pin.net is not reset)."""
        m = _module(pdk, tmp_path, 'x')
        m.define_module_ports({'a': PortDir.IN})
        m.pin_list['a'].net = 'n1'
        with pytest.raises(DuplicateNameError, match="already declared"):
            m.define_module_ports({'a': PortDir.OUT})
        assert m.pin_list['a'].direction is PortDir.IN
        m.define_module_ports({'a': PortDir.IN})                 # identical: no-op
        assert m.pin_list['a'].net == 'n1'

    def test_supply_port_direction_out_warned_by_struct010(self, pdk, tmp_path):
        """LOUD (was SILENT-GAP; TASK-38 made the opt-in lint fire, TASK-94
        promoted it). Port.supply(direction='out') is accepted at
        declaration. The DEFAULT registry now warns, naming port, role and
        direction, and says why: this block declares no POWER-role input,
        so nothing feeds the rail it claims to emit. The one legitimate
        shape — a regulator's POWER OUT next to a POWER IN, which is what
        the promotion gate found on the LDO example — is exempt (asserted
        in ``tests/lint/test_struct009_010_registry.py``). STRUCT-009 stays
        opt-in (pinned)."""
        m = _module(pdk, tmp_path, 'x')
        m.define_module_ports({'vdd': Port.supply(direction='out')})
        assert m.pin_list['vdd'].direction == PortDir.OUT
        assert _lint(m)['STRUCT-010'] == [
            "Port 'vdd' (role=POWER) has direction OUT and the block declares "
            "no POWER-role input; supply ports should be IN unless the block "
            "derives the rail from a same-role supply it receives"]
        registered = {r.rule_id for r in RuleRegistry.default().rules}
        assert 'STRUCT-010' in registered and 'STRUCT-009' not in registered
        # A regulator: the same OUT port next to a POWER IN is not reported.
        reg = _module(pdk, tmp_path, 'reg')
        reg.define_module_ports({'vin': Port.supply(), 'vdd': Port.supply(direction='out')})
        assert 'STRUCT-010' not in _lint(reg)

    def test_define_module_nets_makes_every_port_inout(self, pdk, tmp_path):
        """SILENT-BY-DESIGN (legacy API). define_module_nets declares INOUT,
        role None — every direction/role rule (INT-001..005, STRUCT-002/007)
        is inert for such modules. Documented: prefer define_module_ports."""
        m = _module(pdk, tmp_path, 'legacy')
        m.define_module_nets(['vin', 'vout', 'vdd', 'vss'])
        assert {p.direction for p in m.pin_list.values()} == {PortDir.INOUT}
        assert {p.role for p in m.pin_list.values()} == {None}
        top = _module(pdk, tmp_path)
        top.add_module(m, 'u1', nets={**_INV_MAP, 'vdd': 'gnd'})     # supply on ground
        top.add_module(m, 'u2', nets={**_INV_MAP, 'vdd': 'gnd'})
        assert 'INT-002' not in _lint(top) and 'INT-001' not in _lint(top)


# ---------------------------------------------------------------------------
# AC #3 — port-role inference against the matrix
# ---------------------------------------------------------------------------

class TestRoleInference:

    def _sense_module(self, pdk, tmp_path):
        m = _module(pdk, tmp_path, 'sense')
        m.define_module_ports({'vdd_sense': PortDir.IN, 'vout': PortDir.OUT,
                               'vdd': PortDir.IN, 'gnd': PortDir.IN})
        return m

    def _wire_two(self, pdk, tmp_path, m):
        top = _module(pdk, tmp_path)
        top.add_module(m, 'u1', nets={'vdd_sense': 'sense', 'vout': 'y', 'vdd': 'vdd', 'gnd': 'gnd'})
        top.add_module(m, 'u2', nets={'vdd_sense': 'sense', 'vout': 'y2', 'vdd': 'vdd', 'gnd': 'gnd'})
        with top:
            sources.VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8)
        return top

    def test_inferred_power_role_no_longer_masks_no_driver(self, pdk, tmp_path):
        """R1 — FIXED (TASK-42). Pre-fix: without roles INT-004 reported the
        undriven 'sense' net; after the default ruleset marked 'vdd_sense'
        POWER ('*vdd*'), NoDriverRule's all-supply skip silenced it
        (INT-004 absent) — inference MASKED a wiring error. Now the
        inferred POWER role is a guess and buys no silence: 'sense' is
        reported exactly as without inference, while the inferred-GROUND
        'gnd' net (u1.gnd, u2.gnd, vdd.negative — never OUT-driven by
        construction) stays silent so ordinary inference users get no
        ground false positive."""
        from silk.circuit.port_roles import infer_port_roles, RoleRuleset
        plain = self._wire_two(pdk, tmp_path, self._sense_module(pdk, tmp_path))
        assert "Net 'sense' has no driver: u1.vdd_sense(input), u2.vdd_sense(input)" in _lint(plain)['INT-004']

        inferred = self._sense_module(pdk, tmp_path)
        rep = infer_port_roles(inferred, RoleRuleset.default())
        assert ('vdd_sense', PortRole.POWER) in rep.filled
        top = self._wire_two(pdk, tmp_path, inferred)
        assert {b.role_inferred for b in top.net_dict['sense'].pins} == {True}
        assert _lint(top)['INT-004'] == [
            "Net 'sense' has no driver: u1.vdd_sense(input), u2.vdd_sense(input)"]

    def test_infer_after_wiring_updates_live_parent_metadata(self, pdk, tmp_path):
        """Late inference reaches both the shared definition and parent bindings."""
        from silk.circuit.port_roles import infer_port_roles, RoleRuleset
        m = _module(pdk, tmp_path, 'clkmod')
        m.define_module_ports({'clk_out': PortDir.OUT, 'vin': Port.analog_input(),
                               'vdd': PortDir.IN, 'gnd': PortDir.IN})
        top = _module(pdk, tmp_path)
        top.add_module(m, 'u1', nets={'clk_out': 'n1', 'vin': 'n1', 'vdd': 'vdd', 'gnd': 'gnd'})
        assert 'INT-005' not in _lint(top)
        rep = infer_port_roles(m, RoleRuleset.default())
        assert ('clk_out', PortRole.CLOCK) in rep.filled
        assert m.pin_list['clk_out'].role == PortRole.CLOCK
        assert top.elem_dict['u1']['device'].pin_list['clk_out'].role == PortRole.CLOCK
        assert {b.role for b in top.net_dict['n1'].pins} == {'clock', 'analog'}
        assert 'INT-005' in _lint(top)

    def test_infer_before_wiring_reaches_bindings_and_never_clobbers(self, pdk, tmp_path):
        """Sanity (the supported order, no warning). Inferred roles reach the
        instance copy and Net.pins; an explicit role is never overwritten."""
        import warnings
        from silk.circuit.port_roles import infer_port_roles, RoleRuleset
        m = _module(pdk, tmp_path, 'clkmod')
        m.define_module_ports({'clk_out': PortDir.OUT, 'vin': Port.analog_input(),
                               'vdd_x': Port.ground(), 'gnd': PortDir.IN})
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            rep = infer_port_roles(m, RoleRuleset.default())
        assert dict(rep.filled) == {'clk_out': PortRole.CLOCK, 'gnd': PortRole.GROUND}
        assert m.pin_list['vdd_x'].role == PortRole.GROUND        # explicit wins over '*vdd*'
        top = _module(pdk, tmp_path)
        top.add_module(m, 'u1', nets={'clk_out': 'n1', 'vin': 'n1', 'vdd_x': 'g', 'gnd': 'g'})
        assert {b.role for b in top.net_dict['n1'].pins} == {'clock', 'analog'}
        assert _lint(top)['INT-005'] == ["Net 'n1' connects CLOCK and ANALOG ports: u1.clk_out(clock), u1.vin(analog)"]
