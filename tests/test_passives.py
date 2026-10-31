"""
Tests for passive element classes in pynetlist.core.passives:
  _fmt(), PassiveBase, Resistor, Capacitor, Inductor

Also tests add_passive() on PyNetlist and PySpiceNetlister.generate_passives().
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from silk.utils._utils import _fmt, _parse_si
from silk.elements.passives import PassiveBase, Resistor, Capacitor, Inductor
from silk.exceptions import CircuitError


# ---------------------------------------------------------------------------
# _fmt() tests
# ---------------------------------------------------------------------------

class TestFmt:

    def test_string_passthrough(self):
        """String values pass through unchanged."""
        assert _fmt('100p') == '100p'

    def test_zero(self):
        """Zero formats to '0'."""
        assert _fmt(0) == '0'

    def test_one_kilo(self):
        """1000 formats to '1k'."""
        assert _fmt(1000) == '1k'

    def test_one_point_five_kilo(self):
        """1500 formats to '1.5k'."""
        assert _fmt(1500) == '1.5k'

    def test_one_meg(self):
        """1e6 formats to '1Meg'."""
        assert _fmt(1e6) == '1Meg'

    def test_one_nano(self):
        """1e-9 formats to '1n'."""
        assert _fmt(1e-9) == '1n'

    def test_hundred_pico(self):
        """100e-12 formats to '100p'."""
        assert _fmt(100e-12) == '100p'

    def test_negative_plain_band(self):
        """The milli band is PLAIN, not suffixed: -1e-3 -> '-0.001', not '-1m'.

        An SI prefix only earns its place when it removes exponent noise. In the
        plain band (1e-3 <= |v| < 1e3) it does not, and 'm' is the one suffix
        that is genuinely dangerous — some SPICE tools read it as mega, and a
        UNIT_PDK length of 0.5 um once emitted as the nonsensical '500m'.
        """
        assert _fmt(-1e-3) == '-0.001'
        assert _fmt(0.5) == '0.5'        # not '500m'
        assert _fmt(0.15) == '0.15'      # not '150m'
        assert _fmt(1e-6) == '1u'        # suffixes below the band still apply
        assert _fmt(10e3) == '10k'       # ...and above it

    def test_femto(self):
        """1e-15 maps to the femto suffix."""
        assert _fmt(1e-15) == '1f'


# ---------------------------------------------------------------------------
# _parse_si() tests — the inverse of _fmt()
# ---------------------------------------------------------------------------

class TestParseSI:

    @pytest.mark.parametrize('text,expected', [
        ('0', 0.0), ('2.5', 2.5), ('1.5e-6', 1.5e-6), ('-3', -3.0),
        ('1n', 1e-9), ('100p', 1e-10), ('.5n', 5e-10), ('1f', 1e-15),
        ('1u', 1e-6), ('1k', 1e3), ('1G', 1e9), ('1T', 1e12),
    ])
    def test_plain_and_suffixed(self, text, expected):
        assert _parse_si(text) == pytest.approx(expected)

    def test_meg_beats_milli(self):
        """'MEG' must be matched before 'M', or 1Meg reads as 1 milli."""
        assert _parse_si('1Meg') == pytest.approx(1e6)
        assert _parse_si('1m') == pytest.approx(1e-3)

    def test_case_insensitive(self):
        assert _parse_si('1N') == _parse_si('1n') == pytest.approx(1e-9)

    def test_letters_after_the_suffix_are_ignored(self):
        """SPICE convention — '2.56us' is 2.56u, '1ns' is 1n."""
        assert _parse_si('2.56us') == pytest.approx(2.56e-6)
        assert _parse_si('1ns') == pytest.approx(1e-9)

    def test_unrecognized_trailing_letter_does_not_rescale(self):
        """'1s' is 1 second, not 1 milli. Deviating from what the simulator
        computes would make silk disagree with the netlist it emits."""
        assert _parse_si('1s') == pytest.approx(1.0)

    def test_round_trips_with_fmt(self):
        for value in (1e-9, 100e-12, 1e-6, 1e3, 1e6, 0.5, 1.8):
            assert _parse_si(_fmt(value)) == pytest.approx(value)

    def test_numbers_pass_through(self):
        assert _parse_si(5) == 5.0
        assert _parse_si(1e-9) == 1e-9

    def test_numpy_scalars_pass_through(self):
        """np.int64 is not an int subclass — parse by float-ability, not type."""
        np = pytest.importorskip('numpy')
        assert _parse_si(np.int64(5)) == 5.0
        assert _parse_si(np.float64(1e-9)) == pytest.approx(1e-9)

    @pytest.mark.parametrize('bad', ['', 'abc', 'n', None, [1]])
    def test_unparseable_raises(self, bad):
        with pytest.raises(ValueError):
            _parse_si(bad)


# ---------------------------------------------------------------------------
# to_spice() tests for Resistor, Capacitor, Inductor
# ---------------------------------------------------------------------------

class TestResistorToSpice:

    def test_basic_resistor(self):
        """Resistor with numeric value formats correctly."""
        assert Resistor('load', pos='out', neg='gnd', resistance=1000).to_spice() == 'Rload out gnd 1k\n'

    def test_string_value_passthrough(self):
        """Resistor with string value passes through unchanged."""
        assert Resistor('r2', pos='a', neg='b', resistance='10k').to_spice() == 'Rr2 a b 10k\n'


class TestCapacitorToSpice:

    def test_basic_capacitor(self):
        """Capacitor with numeric value formats correctly."""
        assert Capacitor('c1', pos='vdd', neg='gnd', capacitance=100e-12).to_spice() == 'Cc1 vdd gnd 100p\n'

    def test_string_value_passthrough(self):
        """Capacitor with string value passes through unchanged."""
        assert Capacitor('cx', pos='a', neg='b', capacitance='10p').to_spice() == 'Ccx a b 10p\n'


class TestInductorToSpice:

    def test_basic_inductor(self):
        """Inductor with numeric value formats correctly."""
        assert Inductor('l1', pos='a', neg='b', inductance=1e-6).to_spice() == 'Ll1 a b 1u\n'

    def test_string_value_passthrough(self):
        """Inductor with string value passes through unchanged."""
        assert Inductor('lx', pos='in', neg='out', inductance='100n').to_spice() == 'Llx in out 100n\n'


class TestPassiveBaseToSpice:

    def test_raises_not_implemented(self):
        """PassiveBase.to_spice() raises NotImplementedError."""
        base = PassiveBase('x', 'a', 'b', 1)
        with pytest.raises(NotImplementedError):
            base.to_spice()


# ---------------------------------------------------------------------------
# Aliases: res= / cap= / ind=
# ---------------------------------------------------------------------------

class TestAliases:

    def test_resistor_res_alias(self):
        assert Resistor('r1', pos='a', neg='b', res=1e3).to_spice() == 'Rr1 a b 1k\n'

    def test_capacitor_cap_alias(self):
        assert Capacitor('c1', pos='a', neg='b', cap=100e-12).to_spice() == 'Cc1 a b 100p\n'

    def test_inductor_ind_alias(self):
        assert Inductor('l1', pos='a', neg='b', ind=1e-6).to_spice() == 'Ll1 a b 1u\n'

    def test_resistor_full_name_takes_priority(self):
        """resistance= wins when both are passed (resistance is positional)."""
        r = Resistor('r1', pos='a', neg='b', resistance=2e3)
        assert r.resistance == 2e3


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match='name'):
            Resistor('', pos='a', neg='b', resistance=1e3)

    def test_non_string_name_raises(self):
        with pytest.raises(ValueError, match='name'):
            Resistor(42, pos='a', neg='b', resistance=1e3)

    def test_empty_pos_raises(self):
        with pytest.raises(ValueError, match='pos'):
            Resistor('r1', pos='', neg='b', resistance=1e3)

    def test_empty_neg_raises(self):
        with pytest.raises(ValueError, match='neg'):
            Capacitor('c1', pos='a', neg='', capacitance=1e-9)

    def test_negative_resistance_raises(self):
        with pytest.raises(ValueError, match='resistance'):
            Resistor('r1', pos='a', neg='b', resistance=-1)

    def test_negative_capacitance_raises(self):
        with pytest.raises(ValueError, match='capacitance'):
            Capacitor('c1', pos='a', neg='b', capacitance=-1e-12)

    def test_negative_inductance_raises(self):
        with pytest.raises(ValueError, match='inductance'):
            Inductor('l1', pos='a', neg='b', inductance=-1e-9)

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            Resistor('r1', pos='a', neg='b', resistance=[1, 2])

    def test_missing_value_raises(self):
        with pytest.raises(ValueError, match='resistance'):
            Resistor('r1', pos='a', neg='b')

    def test_zero_resistance_allowed(self):
        """Zero is valid (short circuit)."""
        r = Resistor('short', pos='a', neg='b', resistance=0)
        assert '0' in r.to_spice()

    def test_empty_string_resistance_raises(self):
        """An empty-string resistance raises ValueError (line 20)."""
        with pytest.raises(ValueError, match='resistance'):
            Resistor('r1', pos='a', neg='b', resistance='   ')

    def test_empty_string_capacitance_raises(self):
        """An empty-string capacitance raises ValueError (line 64)."""
        with pytest.raises(ValueError, match='capacitance'):
            Capacitor('c1', pos='a', neg='b', capacitance='')

    def test_empty_string_inductance_raises(self):
        """An empty-string inductance raises ValueError (line 79)."""
        with pytest.raises(ValueError, match='inductance'):
            Inductor('l1', pos='a', neg='b', inductance='')

    def test_missing_capacitance_raises(self):
        """Capacitor with no value raises ValueError (line 64)."""
        with pytest.raises(ValueError, match='capacitance'):
            Capacitor('c1', pos='a', neg='b')

    def test_missing_inductance_raises(self):
        """Inductor with no value raises ValueError (line 79)."""
        with pytest.raises(ValueError, match='inductance'):
            Inductor('l1', pos='a', neg='b')


# ---------------------------------------------------------------------------
# add_passive() on PyNetlist
# ---------------------------------------------------------------------------

class TestAddPassiveOnNetlist:

    def test_capacitor_added_to_passives_dict(self, pdk, tmp_dir):
        """Adding a Capacitor stores it in passives_dict."""
        from silk.circuit.modules import Module

        ckt = Module('PDK_skywater130', tmp_dir, 'p_test.cir', pdk_details=pdk)
        cap = Capacitor('c1', pos='vdd', neg='gnd', capacitance=100e-12)
        ckt.add_passive(cap)
        assert 'c1' in ckt.passives_dict
        assert ckt.passives_dict['c1'] is cap

    def test_duplicate_name_raises_system_exit(self, pdk, tmp_dir):
        """Adding a duplicate passive name raises CircuitError."""
        from silk.circuit.modules import Module

        ckt = Module('PDK_skywater130', tmp_dir, 'p_dup.cir', pdk_details=pdk)
        ckt.add_passive(Capacitor('c1', pos='vdd', neg='gnd', capacitance=100e-12))
        with pytest.raises(CircuitError):
            ckt.add_passive(Resistor('c1', pos='a', neg='b', resistance=1000))

    def test_non_passive_raises_system_exit(self, pdk, tmp_dir):
        """Passing a non-passive object raises CircuitError."""
        from silk.circuit.modules import Module

        ckt = Module('PDK_skywater130', tmp_dir, 'p_bad.cir', pdk_details=pdk)
        with pytest.raises(CircuitError):
            ckt.add_passive('not_a_passive')

    def test_nets_auto_registered(self, pdk, tmp_dir):
        """Passive nets are auto-registered in net_dict by default."""
        from silk.circuit.modules import Module

        ckt = Module('PDK_skywater130', tmp_dir, 'p_nets.cir', pdk_details=pdk)
        ckt.add_passive(Resistor('r1', pos='node_a', neg='node_b', resistance=1000))
        assert 'node_a' in ckt.net_dict
        assert 'node_b' in ckt.net_dict

    def test_strict_nets_raises_for_undeclared_net(self, pdk, tmp_dir):
        """strict_nets=True raises CircuitError when nets are not pre-declared."""
        from silk.circuit.modules import Module

        ckt = Module('PDK_skywater130', tmp_dir, 'p_strict.cir',
                           pdk_details=pdk, strict_nets=True)
        with pytest.raises(CircuitError):
            ckt.add_passive(Capacitor('c1', pos='undeclared_vdd', neg='undeclared_gnd', capacitance=1e-12))

    def test_strict_nets_succeeds_with_predeclared_nets(self, pdk, tmp_dir):
        """strict_nets=True succeeds when both nets are pre-declared."""
        from silk.circuit.modules import Module

        ckt = Module('PDK_skywater130', tmp_dir, 'p_strict_ok.cir',
                           pdk_details=pdk, strict_nets=True)
        ckt.add_net('vdd')
        ckt.add_net('gnd')
        ckt.add_passive(Capacitor('c1', pos='vdd', neg='gnd', capacitance=100e-12))
        assert 'c1' in ckt.passives_dict


# ---------------------------------------------------------------------------
# generate_passives() on PySpiceNetlister
# ---------------------------------------------------------------------------

class TestGeneratePassives:

    def test_empty_dict_returns_empty_string(self, netlister):
        """generate_passives({}) returns ''."""
        assert netlister.generate_passives({}) == ''

    def test_none_returns_empty_string(self, netlister):
        """generate_passives(None) returns ''."""
        assert netlister.generate_passives(None) == ''

    def test_single_capacitor_in_output(self, netlister):
        """generate_passives with one capacitor contains the SPICE line."""
        passives = {'c1': Capacitor('c1', pos='vdd', neg='gnd', capacitance='100p')}
        result = netlister.generate_passives(passives)
        assert 'Cc1 vdd gnd 100p' in result

    def test_multiple_passives_all_in_output(self, netlister):
        """generate_passives with multiple elements includes all of them."""
        passives = {
            'r1': Resistor('r1', pos='out', neg='gnd', resistance=1000),
            'c1': Capacitor('c1', pos='vdd', neg='gnd', capacitance=100e-12),
            'l1': Inductor('l1', pos='a', neg='b', inductance=1e-6),
        }
        result = netlister.generate_passives(passives)
        assert 'Rr1 out gnd 1k' in result
        assert 'Cc1 vdd gnd 100p' in result
        assert 'Ll1 a b 1u' in result
