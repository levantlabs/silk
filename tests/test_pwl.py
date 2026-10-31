"""Tests for PWL waveform class and related fixes."""

import pytest
import warnings
from silk.sources import PWL, VoltageSource, CurrentSource


class TestPWLConstruction:

    def test_empty(self):
        """PWL with no points."""
        pwl = PWL()
        assert len(pwl) == 0

    def test_from_tuples(self):
        """PWL from list of (time, value) tuples."""
        pwl = PWL([(0, 0), (1e-9, 1.8)])
        assert len(pwl) == 2
        assert pwl[0] == (0.0, 0.0)
        assert pwl[1] == (1e-9, 1.8)

    def test_from_flat_list(self):
        """PWL from flat [t0, v0, t1, v1] list."""
        pwl = PWL([0, 0, 1e-9, 1.8, 10e-9, 1.8])
        assert len(pwl) == 3

    def test_flat_list_odd_length_raises(self):
        """Flat list with odd length raises ValueError."""
        with pytest.raises(ValueError, match="even length"):
            PWL([0, 0, 1e-9])

    def test_bad_tuple_length_raises(self):
        """Tuple with wrong number of elements raises ValueError."""
        with pytest.raises(ValueError, match="must be.*time.*value"):
            PWL([(0, 0, 0)])

    def test_sorts_by_time(self):
        """Points are sorted by time."""
        pwl = PWL([(10e-9, 1.8), (0, 0), (5e-9, 0.9)])
        assert pwl[0][0] == 0.0
        assert pwl[1][0] == 5e-9
        assert pwl[2][0] == 10e-9

    def test_with_r_and_td(self):
        """PWL stores r and td parameters."""
        pwl = PWL([(0, 0), (1e-9, 1.8)], r=5e-9, td=1e-9)
        assert pwl.r == 5e-9
        assert pwl.td == 1e-9


class TestPWLAdd:

    def test_add_returns_self(self):
        """add() returns self for chaining."""
        pwl = PWL()
        result = pwl.add(0, 0)
        assert result is pwl

    def test_chaining(self):
        """Fluent chaining builds correctly."""
        pwl = PWL().add(0, 0).add(1e-9, 0).add(2e-9, 1.8)
        assert len(pwl) == 3

    def test_add_sorts(self):
        """add() maintains sort order."""
        pwl = PWL().add(10e-9, 1.8).add(0, 0)
        assert pwl[0][0] == 0.0


class TestPWLProtocol:

    def test_len(self):
        assert len(PWL([(0, 0), (1, 1)])) == 2

    def test_iter(self):
        pwl = PWL([(0, 0), (1e-9, 1.8)])
        times = [t for t, v in pwl]
        assert times == [0.0, 1e-9]

    def test_index(self):
        pwl = PWL([(0, 0), (1e-9, 1.8)])
        assert pwl[1] == (1e-9, 1.8)

    def test_repr(self):
        pwl = PWL([(0, 0.0)])
        assert 'PWL' in repr(pwl)

    def test_points_property(self):
        """points returns a copy."""
        pwl = PWL([(0, 0)])
        pts = pwl.points
        pts.append((1, 1))
        assert len(pwl) == 1  # original unchanged


class TestPWLToSpice:

    def test_basic(self):
        pwl = PWL([(0, 0), (1e-9, 1.8)])
        result = pwl.to_spice()
        assert result.startswith('PWL(')
        assert '1e-09 1.8' in result

    def test_with_r_and_td(self):
        pwl = PWL([(0, 0), (1e-9, 1.8)], r=5e-9, td=1e-9)
        result = pwl.to_spice()
        assert 'r=' in result
        assert 'td=' in result

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="no points"):
            PWL().to_spice()


class TestPWLInSources:

    def test_voltage_source_with_pwl_object(self):
        pwl = PWL([(0, 0), (1e-9, 1.8)])
        vs = VoltageSource('vin', positive='in', negative='gnd', pwl=pwl)
        result = vs.to_spice()
        assert 'PWL(' in result

    def test_voltage_source_flat_list_autoconvert(self):
        """Flat list is auto-converted to PWL."""
        vs = VoltageSource('vin', positive='in', negative='gnd',
                           pwl=[0, 0, 1e-9, 1.8])
        assert isinstance(vs.pwl, PWL)
        result = vs.to_spice()
        assert 'PWL(' in result

    def test_voltage_source_tuple_list_autoconvert(self):
        """Tuple list is auto-converted to PWL."""
        vs = VoltageSource('vin', positive='in', negative='gnd',
                           pwl=[(0, 0), (1e-9, 1.8)])
        assert isinstance(vs.pwl, PWL)

    def test_voltage_source_string_raises(self):
        """String pwl raises TypeError."""
        with pytest.raises(TypeError, match="pwl must be"):
            VoltageSource('vin', positive='in', negative='gnd',
                          pwl='file=/etc/passwd')

    def test_voltage_source_pwl_r_td_passthrough(self):
        """pwl_r and pwl_td are passed to PWL object."""
        vs = VoltageSource('vin', positive='in', negative='gnd',
                           pwl=[0, 0, 1e-9, 1.8], pwl_r=5e-9, pwl_td=1e-9)
        assert vs.pwl.r == 5e-9
        assert vs.pwl.td == 1e-9

    def test_current_source_with_pwl(self):
        pwl = PWL([(0, 0), (1e-9, 1e-3)])
        cs = CurrentSource('ibias', positive='vdd', negative='gnd', pwl=pwl)
        result = cs.to_spice()
        assert 'PWL(' in result

    def test_current_source_pwl_r_td_passthrough(self):
        """CurrentSource accepts pwl_r/pwl_td like VoltageSource (symmetry)."""
        cs = CurrentSource('ipwl', positive='vdd', negative='gnd',
                           pwl=[0, 0, 1e-9, 1e-3], pwl_r=5e-9, pwl_td=1e-9)
        assert cs.pwl.r == 5e-9
        assert cs.pwl.td == 1e-9
        assert cs.pwl_r == 5e-9 and cs.pwl_td == 1e-9
        # ...and they emit into the SPICE PWL line.
        spice = cs.to_spice()
        assert 'r=5e-09' in spice and 'td=1e-09' in spice

    def test_current_source_flat_list(self):
        cs = CurrentSource('ibias', positive='vdd', negative='gnd',
                           pwl=[0, 0, 1e-9, 1e-3])
        assert isinstance(cs.pwl, PWL)


@pytest.mark.parametrize('cls', [VoltageSource, CurrentSource])
@pytest.mark.parametrize('field,values,sample,expected', [
    ('td', [2, 0, None], 1.5, [0, 1.5, 1.5]),
    ('r', [1, 0, None], 2.5, [1.5, .5, 2]),
])
@pytest.mark.parametrize('batch', [False, True])
def test_timing_alias_updates_effective_pwl(tmp_path, cls, field, values, sample, expected, batch):
    from silk.circuit.modules import Module
    mod = Module(None, str(tmp_path), 'timing.cir', None)
    src = cls('stim', 'out', '0', pwl=PWL([(0, 0), (1, 1), (2, 2)],
                                       **{field: 1 if field == 'td' else 0}))
    mod.add_source(src)
    direct = cls('stim', 'out', '0', pwl=PWL(src.pwl.points, **{field: getattr(src.pwl, field)}))
    before = src.to_numpy([sample])[0]
    for value, want in zip(values, expected):
        if batch:
            mod.update_source('stim', ['pwl_' + field], [value])
        else:
            setattr(src, 'pwl_' + field, value)
        setattr(direct.pwl, field, value)
        assert getattr(src.pwl, field) == value
        assert getattr(src, 'pwl_' + field) == value
        assert src.to_spice() == direct.to_spice()
        assert src.to_numpy([sample])[0] == pytest.approx(want)
    assert before != expected[0]


@pytest.mark.parametrize('cls', [VoltageSource, CurrentSource])
def test_timing_alias_constructor_precedence_and_shared_readback(cls):
    wave = PWL([(0, 0), (1, 1)], r=0, td=2)
    first = cls('first', 'out', '0', pwl=wave, pwl_r=1, pwl_td=9)
    second = cls('second', 'out', '0', pwl=wave)
    assert (wave.r, wave.td) == (0, 2)
    assert (first.pwl_r, first.pwl_td, second.pwl_td) == (0, 2, 2)
    first.pwl_td = 3
    assert first.pwl is second.pwl is wave
    assert wave.td == second.pwl_td == 3
    wave.td = 4
    assert first.pwl_td == second.pwl_td == 4
    assert second.to_numpy([4.5])[0] == pytest.approx(.5)


@pytest.mark.parametrize('cls', [VoltageSource, CurrentSource])
def test_timing_alias_without_pwl_and_symbolic_values(cls):
    from silk.circuit.variable import Variable
    loose = cls('stim', 'out', '0', dc=.2, pwl_td=1)
    before = loose.to_spice()
    loose.pwl_td = 2
    loose.pwl_r = None
    assert loose.pwl_td == 2 and loose.pwl_r is None
    assert loose.pwl is None and loose.to_spice() == before
    delay = Variable('delay', 2)
    wave = PWL([(0, 0), (1, 1)], td=delay)
    src = cls('stim', 'out', '0', pwl=wave)
    assert src.pwl_td is delay
    src.pwl_td = delay + 1
    assert src.pwl.td is src.pwl_td
    assert src.to_numpy([3.5])[0] == pytest.approx(.5)
    src.pwl_td = '{delay}'
    assert 'td={delay}' in src.to_spice()


@pytest.mark.parametrize('cls', [VoltageSource, CurrentSource])
@pytest.mark.parametrize('field', ['pwl_r', 'pwl_td'])
@pytest.mark.parametrize('explicit_pwl', [False, True])
def test_constructor_timing_alias_refuses_line_break_even_when_unused(cls, field, explicit_pwl):
    from silk.exceptions import DeckTextError
    wave = PWL([(0, 0), (1, 1)], r=0, td=2) if explicit_pwl else None
    with pytest.raises(DeckTextError, match=field):
        cls('stim', 'out', '0', pwl=wave, **{field: '1\n'})
    if wave is not None:
        assert (wave.r, wave.td) == (0, 2)
