"""TASK-76 — a source terminal (and a passive terminal) is ONE SPICE node
token, so it must be a non-empty, whitespace-free str or a Net.

TASK-41 established the rule for factory device pins and TASK-75 extended it
to the legacy ``add_element`` styles. Sources (``silk/sources``) had no
terminal validation at all, and passives let whitespace through. Measured
2026-09-03 on the sky130 fake-lib PDK before this fix:

* ``VoltageSource('vs', positive='', negative='gnd', dc=1.8)`` was accepted
  and emitted ``Vvs  gnd DC 1.8`` — one node short. ngspice-42 read ``gnd``
  as the positive node and the literal ``DC`` as the negative node, inventing
  a node ``dc`` at -1.8 V with no error or warning: polarity inverted relative
  to intent. silk's own model registered a net named ``''``.
* ``positive=None`` / ``positive=0`` died with a bare
  ``AttributeError: 'NoneType' object has no attribute 'casefold'`` from inside
  net-name canonicalization — naming neither the source nor the terminal.
* ``positive='a b'`` emitted ``Vvs a b gnd DC 1.8``: three node tokens for a
  two-terminal source.
* ``Resistor('r1', pos='a b', neg='gnd', resistance=1e3)`` emitted
  ``Rr1 a b gnd 1k``; ngspice-42 warned "can not find model gnd" and left
  node ``a`` floating (singular matrix) — a warning silk never surfaced.

Per decision-1 these are CALLER ERRORS (malformed for every corner), so they
raise, through the ONE validator ``silk.elements.base.validate_net_token``
that devices already use — a second validator is how the two would drift.
"""
import pytest

from silk import sources
from silk.circuit.modules import Module
from silk.elements.passives import Capacitor, Inductor, Resistor


# (bad value, exception type) — the same table TASK-75 uses for device pins.
_BAD = [
    ('', ValueError),
    ('   ', ValueError),
    ('\t', ValueError),
    ('a b', ValueError),
    ('a\tb', ValueError),
    (' a', ValueError),
    ('a\n', ValueError),
    (None, TypeError),
    (0, TypeError),
    (1.5, TypeError),
    (object(), TypeError),
]
_BAD_IDS = ['empty', 'blank', 'tab', 'space-inside', 'tab-inside',
            'space-leading', 'newline-trailing',
            'none', 'int', 'float', 'object']


def _module(pdk, tmp_path, name='top'):
    return Module('PDK_skywater130', str(tmp_path / name), f'{name}.cir',
                  pdk_details=pdk, module_name=name)


# Every SourceBase subclass: (factory, good kwargs, node terminals, expected
# node-token count on the emitted line). ``ctrl_vname`` on CCVS/CCCS is a
# SOURCE name, not a node, so it is not a terminal.
def _vs(**kw):
    return sources.VoltageSource('vs', **{'positive': 'a', 'negative': 'gnd', 'dc': 1.8, **kw})

def _is(**kw):
    return sources.CurrentSource('is', **{'positive': 'a', 'negative': 'gnd', 'dc': 1e-6, **kw})

def _bv(**kw):
    return sources.BehavioralVoltage('bv', **{'positive': 'a', 'negative': 'gnd',
                                              'expression': 'v(x)', **kw})

def _bi(**kw):
    return sources.BehavioralCurrent('bi', **{'positive': 'a', 'negative': 'gnd',
                                              'expression': 'v(x)', **kw})

def _e(**kw):
    return sources.VCVS('e', **{'out_pos': 'o', 'out_neg': 'gnd',
                                'ctrl_pos': 'a', 'ctrl_neg': 'b', 'gain': 2, **kw})

def _g(**kw):
    return sources.VCCS('g', **{'out_pos': 'o', 'out_neg': 'gnd',
                                'ctrl_pos': 'a', 'ctrl_neg': 'b', 'gm': 1e-3, **kw})

def _h(**kw):
    return sources.CCVS('h', **{'out_pos': 'o', 'out_neg': 'gnd',
                                'transresistance': 1e3, 'ctrl_vname': 'vmeas', **kw})

def _f(**kw):
    return sources.CCCS('f', **{'out_pos': 'o', 'out_neg': 'gnd',
                                'gain': 2, 'ctrl_vname': 'vmeas', **kw})

_TWO = ('positive', 'negative')
_FOUR = ('out_pos', 'out_neg', 'ctrl_pos', 'ctrl_neg')
_OUT = ('out_pos', 'out_neg')

_SOURCES = [
    ('VoltageSource', _vs, _TWO),
    ('CurrentSource', _is, _TWO),
    ('BehavioralVoltage', _bv, _TWO),
    ('BehavioralCurrent', _bi, _TWO),
    ('VoltageControlledVoltageSource', _e, _FOUR),
    ('VoltageControlledCurrentSource', _g, _FOUR),
    ('CurrentControlledVoltageSource', _h, _OUT),
    ('CurrentControlledCurrentSource', _f, _OUT),
]
_SOURCE_IDS = [s[0] for s in _SOURCES]

# Every (source, terminal) pair, so each node slot on each line is covered.
_CELLS = [(cls, make, term) for cls, make, terms in _SOURCES for term in terms]
_CELL_IDS = [f'{cls}.{term}' for cls, _, term in _CELLS]


# ---------------------------------------------------------------------------
# AC #1 / AC #2 — every terminal of every source class, at construction
# ---------------------------------------------------------------------------

class TestSourceTerminalValues:

    @pytest.mark.parametrize('cls,make,term', _CELLS, ids=_CELL_IDS)
    @pytest.mark.parametrize('bad,exc', _BAD, ids=_BAD_IDS)
    def test_bad_terminal_is_typed_and_names_source_and_terminal(
            self, cls, make, term, bad, exc):
        with pytest.raises(exc) as e:
            make(**{term: bad})
        src_name = make.__name__.lstrip('_')
        assert f"{cls} '{src_name}' terminal '{term}'" in str(e.value)

    def test_none_is_a_typeerror_not_a_bare_attributeerror(self):
        """AC #2. The old failure was ``AttributeError: 'NoneType' object has
        no attribute 'casefold'`` from inside net-name canonicalization."""
        with pytest.raises(TypeError) as e:
            _vs(positive=None)
        assert 'casefold' not in str(e.value)
        assert 'never assigned' in str(e.value)

    def test_int_message_points_at_the_missing_quotes(self):
        """``negative=0`` for the SPICE ground node is the plausible slip."""
        with pytest.raises(TypeError, match=r"write '0', not 0"):
            _vs(negative=0)

    def test_message_states_the_source_rule_and_the_consequence(self):
        with pytest.raises(ValueError) as e:
            _vs(positive='')
        msg = str(e.value)
        assert 'source terminal must be a non-empty net name' in msg
        assert 'one node short' in msg

    def test_ctrl_vname_is_a_source_name_not_a_node(self):
        """CCVS/CCCS reference a controlling source by name; that slot is not
        a terminal and is emitted verbatim."""
        assert _h(ctrl_vname='vmeas').to_spice() == 'Hh o gnd vmeas 1000.0\n'
        assert 'ctrl_vname' not in sources.CCVS._node_terminals
        assert 'ctrl_vname' not in sources.CCCS._node_terminals

    def test_library_constructors_go_through_the_same_check(self):
        """``sources.vdc(name, p, n, value)`` builds a VoltageSource, so it
        inherits the rule without its own copy."""
        with pytest.raises(ValueError, match=r"VoltageSource 'vdd' terminal 'positive'"):
            sources.vdc('vdd', '', '0', 1.8)
        with pytest.raises(TypeError, match=r"CurrentSource 'ib' terminal 'negative'"):
            sources.idc('ib', 'vb', 0, 1e-6)


# ---------------------------------------------------------------------------
# AC #1 / AC #3 — rejected BEFORE registration; never a net named ''
# ---------------------------------------------------------------------------

class TestNothingIsRegistered:

    @pytest.mark.parametrize('bad,exc', _BAD, ids=_BAD_IDS)
    def test_with_module_registers_neither_source_nor_junk_net(
            self, pdk, tmp_path, bad, exc):
        m = _module(pdk, tmp_path)
        with m:
            with pytest.raises(exc):
                sources.VoltageSource('vs', positive=bad, negative='gnd', dc=1.8)
        assert 'vs' not in m.sig_sources_dict
        assert m.net_dict == {}

    def test_add_source_never_sees_an_empty_terminal(self, pdk, tmp_path):
        """The two placeholder cells this task rewrote used to leave a net
        named ``''`` in ``Module.net_dict``."""
        m = _module(pdk, tmp_path)
        with pytest.raises(ValueError):
            m.add_source(sources.VoltageSource('vs', positive='', negative=''))
        assert '' not in m.net_dict
        assert m.sig_sources_dict == {}

    @pytest.mark.parametrize('cls,make,terms', _SOURCES, ids=_SOURCE_IDS)
    def test_emitted_line_has_exactly_one_token_per_terminal(self, cls, make, terms):
        """AC #3 positive half: node-token count == terminal count for every
        source class, with the ctrl_vname/value tail accounted for."""
        src = make()
        tokens = src.to_spice().split()
        nodes = tokens[1:1 + len(terms)]
        assert nodes == [getattr(src, t) for t in terms]
        # nothing that follows the nodes looks like a leaked node slot
        assert '' not in tokens


# ---------------------------------------------------------------------------
# Net objects normalise to their name
# ---------------------------------------------------------------------------

class TestNetObjectsAccepted:

    def test_net_is_stored_as_its_name_and_emits_it(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        m.add_net('vdd')
        m.add_net('gnd')
        vdd, gnd = m.net_dict['vdd'], m.net_dict['gnd']
        src = sources.VoltageSource('vs', positive=vdd, negative=gnd, dc=1.8)
        assert src.positive == 'vdd' and src.negative == 'gnd'
        assert src.to_spice() == 'Vvs vdd gnd DC 1.8\n'
        m.add_source(src)
        assert [b.instance for b in m.net_dict['vdd'].pins] == ['vs']

    def test_four_terminal_source_accepts_nets(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        for n in ('o', 'gnd', 'a', 'b'):
            m.add_net(n)
        nd = m.net_dict
        e = sources.VCVS('e', out_pos=nd['o'], out_neg=nd['gnd'],
                         ctrl_pos=nd['a'], ctrl_neg=nd['b'], gain=2)
        assert e.to_spice() == 'Ee o gnd a b 2\n'


# ---------------------------------------------------------------------------
# Rewires and the emitter guard
# ---------------------------------------------------------------------------

class TestRewireAndEmitter:

    @pytest.mark.parametrize('bad,exc', _BAD, ids=_BAD_IDS)
    def test_rewire_is_rejected_and_keeps_the_old_terminal(self, bad, exc):
        """``src.positive = ''`` is the construction slip one rewire later —
        the path ``Module.connect_source`` takes via ``setattr``."""
        src = _vs()
        with pytest.raises(exc, match=r"VoltageSource 'vs' terminal 'positive'"):
            src.positive = bad
        assert src.positive == 'a'
        assert src.to_spice() == 'Vvs a gnd DC 1.8\n'

    def test_connect_source_cannot_reach_a_bad_name(self, pdk, tmp_path):
        """Belt: ``connect_source`` requires the net to be declared, and
        ``add_net`` (TASK-75) will not declare a bad name, so the setattr
        route is closed on both ends."""
        m = _module(pdk, tmp_path)
        with pytest.raises(ValueError):
            m.add_net('a b')
        m.add_source(_vs())
        with pytest.raises(Exception):
            m.connect_source('vs', ['positive'], ['a b'])
        assert m.sig_sources_dict['vs'].positive == 'a'

    @pytest.mark.parametrize('cls,make,terms', _SOURCES, ids=_SOURCE_IDS)
    def test_to_spice_refuses_to_emit_a_line_short_of_a_node(self, cls, make, terms):
        """AC #3, the emitter side: even with the attribute forced past
        ``__setattr__``, the line is never written one token short."""
        src = make()
        object.__setattr__(src, terms[-1], '')
        with pytest.raises(ValueError, match=rf"{cls} .* terminal '{terms[-1]}'"):
            src.to_spice()

    def test_deleted_terminal_reports_unconnected(self):
        src = _vs()
        object.__delattr__(src, 'negative')
        with pytest.raises(TypeError, match=r"terminal 'negative': no net connected"):
            src.to_spice()


# ---------------------------------------------------------------------------
# AC #4 — passives reject whitespace in a net name, same message shape
# ---------------------------------------------------------------------------

class TestPassiveWhitespace:

    _MAKE = [
        ('Resistor', lambda **kw: Resistor('r1', **{'pos': 'a', 'neg': 'gnd', 'resistance': 1e3, **kw})),
        ('Capacitor', lambda **kw: Capacitor('c1', **{'pos': 'a', 'neg': 'gnd', 'capacitance': 1e-12, **kw})),
        ('Inductor', lambda **kw: Inductor('l1', **{'pos': 'a', 'neg': 'gnd', 'inductance': 1e-9, **kw})),
    ]

    @pytest.mark.parametrize('make', [m for _, m in _MAKE], ids=[n for n, _ in _MAKE])
    @pytest.mark.parametrize('term', ['pos', 'neg'])
    @pytest.mark.parametrize('bad', ['a b', 'a\tb', ' a', 'a\n'], ids=['space', 'tab', 'lead', 'trail'])
    def test_whitespace_in_a_terminal_is_a_valueerror(self, make, term, bad):
        with pytest.raises(ValueError) as e:
            make(**{term: bad})
        msg = str(e.value)
        # Same shape as the existing non-empty check, plus the whitespace rule.
        assert msg.startswith(f'{term} must be a non-empty string')
        assert 'no whitespace' in msg
        assert repr(bad) in msg

    def test_existing_empty_message_is_unchanged(self):
        with pytest.raises(ValueError, match=r"^pos must be a non-empty string, got ''$"):
            Resistor('r1', pos='', neg='gnd', resistance=1e3)

    def test_nothing_is_registered_in_a_with_block(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        with m:
            with pytest.raises(ValueError):
                Resistor('r1', pos='a b', neg='gnd', resistance=1e3)
        assert m.passives_dict == {}
        assert m.net_dict == {}

    def test_good_passive_still_emits_two_node_tokens(self):
        assert Resistor('r1', pos='a', neg='gnd', resistance=1e3).to_spice() == 'Rr1 a gnd 1k\n'
