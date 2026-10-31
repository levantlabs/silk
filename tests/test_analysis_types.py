"""
Tests for typed analysis classes in pynetlist.core.analysis:
  _fmt(), AnalysisBase, Transient, AC, DC, Noise

Also tests the object-based path in PyNetlist.add_sim_type().
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from silk.simulation.analysis import (
    _fmt, AnalysisBase, Transient, AC, DC, Noise,
)


# ---------------------------------------------------------------------------
# _fmt() tests  (same function, defined independently in analysis.py)
# ---------------------------------------------------------------------------

class TestFmt:

    def test_string_passthrough(self):
        assert _fmt('100p') == '100p'

    def test_zero(self):
        assert _fmt(0) == '0'

    def test_one_kilo(self):
        assert _fmt(1000) == '1k'

    def test_one_point_five_kilo(self):
        assert _fmt(1500) == '1.5k'

    def test_one_meg(self):
        assert _fmt(1e6) == '1Meg'

    def test_one_nano(self):
        assert _fmt(1e-9) == '1n'

    def test_hundred_pico(self):
        assert _fmt(100e-12) == '100p'

    def test_negative_plain_band(self):
        """The milli band is PLAIN, not suffixed (see test_passives for why)."""
        assert _fmt(-1e-3) == '-0.001'
        assert _fmt(0.5) == '0.5'        # not '500m'
        assert _fmt(1e-6) == '1u'        # suffixes below the band still apply

    def test_femto(self):
        """1e-15 maps to the femto suffix."""
        assert _fmt(1e-15) == '1f'


# ---------------------------------------------------------------------------
# AnalysisBase
# ---------------------------------------------------------------------------

class TestAnalysisBase:

    def test_raises_not_implemented(self):
        """AnalysisBase.to_spice() raises NotImplementedError."""
        base = AnalysisBase()
        with pytest.raises(NotImplementedError):
            base.to_spice()


# ---------------------------------------------------------------------------
# Transient
# ---------------------------------------------------------------------------

class TestTransient:

    def test_basic_transient(self):
        """Transient(1e-9, 5e-6) produces correct .tran line."""
        assert Transient(1e-9, 5e-6).to_spice() == '.tran 1n 5u\n'

    def test_transient_with_nonzero_start(self):
        """Transient with start != 0 includes the start value."""
        result = Transient(1e-9, 5e-6, start=1e-7).to_spice()
        assert '.tran' in result
        assert '1n' in result
        assert '5u' in result
        assert '100n' in result  # 1e-7 -> 100n

    def test_transient_with_zero_start_omits_start(self):
        """Transient with start=0 does NOT include start in output."""
        result = Transient(1e-9, 5e-6, start=0).to_spice()
        parts = result.strip().split()
        # .tran step stop  — exactly 3 tokens
        assert len(parts) == 3

    def test_transient_default_start_omits_start(self):
        """Transient default start=0 gives exactly 3 tokens."""
        result = Transient(1e-9, 5e-6).to_spice()
        parts = result.strip().split()
        assert len(parts) == 3

    # ── maxstep / uic ───────────────

    def test_transient_maxstep_appends_fourth_field(self):
        """maxstep shows up as the 4th field on the .tran line."""
        result = Transient(1e-9, 5e-6, maxstep=20e-12).to_spice()
        parts = result.strip().split()
        # .tran <step> <stop> <start> <maxstep>
        assert len(parts) == 5
        assert parts[-1] == '20p'

    def test_transient_maxstep_forces_start_field(self):
        """Emitting maxstep requires start to be in the line (else ngspice
        would interpret the 3rd field as tstart, not maxstep)."""
        # Even with default start=0, the field should be present when maxstep is set.
        result = Transient(1e-9, 5e-6, maxstep=10e-12).to_spice()
        assert ' 0 ' in result

    def test_transient_uic_appended(self):
        """uic=True appends the literal 'uic' token."""
        result = Transient(1e-9, 5e-6, uic=True).to_spice()
        assert result.rstrip().endswith('uic')

    def test_transient_uic_default_false(self):
        """uic is not appended when left default."""
        assert 'uic' not in Transient(1e-9, 5e-6).to_spice()

    def test_transient_maxstep_and_uic_together(self):
        """Both maxstep and uic on the same line — order: maxstep then uic."""
        result = Transient(1e-9, 5e-6, maxstep=20e-12, uic=True).to_spice()
        parts = result.strip().split()
        # .tran <step> <stop> <start> <maxstep> uic  → 6 tokens
        assert len(parts) == 6
        assert parts[-2] == '20p'
        assert parts[-1] == 'uic'

    def test_transient_start_plus_maxstep_plus_uic(self):
        """All three optional fields together."""
        result = Transient(1e-9, 5e-6, start=1e-7,
                           maxstep=20e-12, uic=True).to_spice()
        parts = result.strip().split()
        assert parts[0] == '.tran'
        assert parts[1] == '1n'
        assert parts[2] == '5u'
        assert parts[3] == '100n'
        assert parts[4] == '20p'
        assert parts[5] == 'uic'

    @pytest.mark.parametrize('step', [0, -1e-9, True, 'not-a-number'])
    def test_rejects_invalid_step(self, step):
        with pytest.raises(ValueError, match=r'Transient\.step'):
            Transient(step=step, stop=1e-6)

    @pytest.mark.parametrize('field,value', [
        ('step', float('nan')),
        ('stop', float('inf')),
        ('start', float('-inf')),
        ('maxstep', float('nan')),
    ])
    def test_rejects_nonfinite_timing(self, field, value):
        kwargs = {'step': 1e-9, 'stop': 1e-6}
        kwargs[field] = value
        with pytest.raises(ValueError, match=rf'Transient\.{field}'):
            Transient(**kwargs)

    def test_rejects_negative_start(self):
        with pytest.raises(ValueError, match=r'Transient\.start'):
            Transient(step=1e-9, stop=1e-6, start=-1e-9)

    @pytest.mark.parametrize('start,stop', [(1e-6, 1e-6), (2e-6, 1e-6)])
    def test_rejects_stop_not_after_start(self, start, stop):
        with pytest.raises(ValueError, match=r'Transient\.stop'):
            Transient(step=1e-9, stop=stop, start=start)

    @pytest.mark.parametrize('maxstep', [0, -1e-12])
    def test_rejects_nonpositive_maxstep(self, maxstep):
        with pytest.raises(ValueError, match=r'Transient\.maxstep'):
            Transient(step=1e-9, stop=1e-6, maxstep=maxstep)

    def test_spice_timing_strings_remain_supported(self):
        assert Transient('1n', '10n').to_spice() == '.tran 1n 10n\n'


# ---------------------------------------------------------------------------
# AC
# ---------------------------------------------------------------------------

class TestAC:

    def test_basic_ac_dec(self):
        """AC('dec', 10, 1e3, 1e9) produces correct .ac line."""
        assert AC('dec', 10, 1e3, 1e9).to_spice() == '.ac dec 10 1k 1G\n'

    def test_ac_oct_variation(self):
        """AC with 'oct' variation is accepted."""
        result = AC('oct', 5, 100, 1e6).to_spice()
        assert '.ac oct' in result

    def test_ac_lin_variation(self):
        """AC with 'lin' variation is accepted."""
        result = AC('lin', 100, 1e3, 1e6).to_spice()
        assert '.ac lin' in result

    def test_ac_invalid_variation_raises_value_error(self):
        """AC raises ValueError for an invalid variation string."""
        with pytest.raises(ValueError):
            AC('invalid', 10, 1e3, 1e9)

    @pytest.mark.parametrize('points,expected', [
        (10, 10),
        (10.0, 10),
        ('10', 10),
        ('10.0', 10),
        ('1k', 1000),
    ])
    def test_accepts_and_normalizes_positive_whole_point_counts(
            self, points, expected):
        analysis = AC('dec', points, 1, 1e6)

        assert analysis.points == expected
        assert type(analysis.points) is int
        assert analysis.to_spice().split()[2] == str(expected)

    @pytest.mark.parametrize(
        'points',
        [0, -1, 1.5, '10.5', True, float('nan'), float('inf')],
    )
    def test_rejects_points_that_are_not_positive_whole_numbers(self, points):
        with pytest.raises(
                ValueError,
                match=r'AC\.points must resolve to a positive whole number'):
            AC('dec', points, 1, 1e6)

    @pytest.mark.parametrize('field,value', [
        ('fstart', 0),
        ('fstart', -1),
        ('fstart', float('nan')),
        ('fstart', True),
        ('fstop', 0),
        ('fstop', float('inf')),
    ])
    def test_rejects_invalid_frequencies(self, field, value):
        kwargs = {'variation': 'dec', 'points': 10,
                  'fstart': 1, 'fstop': 1e6}
        kwargs[field] = value
        with pytest.raises(ValueError, match=rf'AC\.{field}'):
            AC(**kwargs)

    def test_rejects_descending_frequency_range(self):
        with pytest.raises(ValueError, match=r'AC\.fstop'):
            AC('dec', 10, 1e6, 1)

    def test_single_frequency_and_spice_strings_remain_supported(self):
        assert AC('lin', 1, '1k', '1k').to_spice() == '.ac lin 1 1k 1k\n'


# ---------------------------------------------------------------------------
# DC
# ---------------------------------------------------------------------------

class TestDC:

    def test_basic_dc_sweep(self):
        """DC('vin', -1.8, 1.8, 0.01) produces correct .dc line."""
        assert DC('vin', -1.8, 1.8, 0.01).to_spice() == '.dc vin -1.8 1.8 0.01\n'

    def test_dc_source_name_in_output(self):
        """DC to_spice() includes the source name."""
        result = DC('vsupply', 0, 3.3, 0.1).to_spice()
        assert 'vsupply' in result

    def test_dc_contains_start_stop_step(self):
        """DC to_spice() contains start, stop, step values."""
        result = DC('vin', 0, 1.8, 0.01).to_spice()
        assert '0' in result
        assert '1.8' in result
        assert '0.01' in result      # plain decimal, not '10m'

    def test_rejects_boolean_sweep_value(self):
        with pytest.raises(ValueError, match=r'DC\.step'):
            DC('vin', 0, 1.8, True)

    def test_rejects_zero_step(self):
        with pytest.raises(ValueError, match=r'DC\.step'):
            DC('vin', 0, 1.8, 0)

    @pytest.mark.parametrize('field,value', [
        ('start', float('nan')),
        ('stop', float('inf')),
        ('step', float('-inf')),
    ])
    def test_rejects_nonfinite_sweep_values(self, field, value):
        kwargs = {'source': 'vin', 'start': 0, 'stop': 1.8, 'step': 0.01}
        kwargs[field] = value
        with pytest.raises(ValueError, match=rf'DC\.{field}'):
            DC(**kwargs)

    @pytest.mark.parametrize('start,stop,step', [
        (0, 1.8, -0.01),
        (1.8, 0, 0.01),
    ])
    def test_rejects_step_that_moves_away_from_stop(self, start, stop, step):
        with pytest.raises(ValueError, match=r'DC\.step'):
            DC('vin', start, stop, step)

    def test_reverse_sweep_and_spice_strings_remain_supported(self):
        assert DC('vin', '1.8', '0', '-0.1').to_spice() == \
            '.dc vin 1.8 0 -0.1\n'

    def test_single_point_sweep_remains_supported(self):
        assert DC('vin', 0.9, 0.9, 0.1).to_spice() == '.dc vin 0.9 0.9 0.1\n'


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

class TestNoise:

    def test_basic_noise(self):
        """Noise produces correct .noise line."""
        result = Noise('out', '0', 'vin', 'dec', 100, 1, 1e9).to_spice()
        assert result.startswith('.noise v(out,0) vin dec 100 ')
        assert '1G' in result
        assert result.endswith('\n')

    def test_noise_contains_output_ref(self):
        """Noise to_spice() contains output and reference nodes."""
        result = Noise('vout', 'gnd', 'vsrc', 'dec', 10, 1e3, 1e9).to_spice()
        assert 'v(vout,gnd)' in result

    def test_noise_contains_source(self):
        """Noise to_spice() contains the source name."""
        result = Noise('out', '0', 'vsrc', 'dec', 10, 1e3, 1e9).to_spice()
        assert 'vsrc' in result

    def test_noise_invalid_variation_raises_value_error(self):
        """Noise raises ValueError for an invalid variation string."""
        with pytest.raises(ValueError):
            Noise('out', '0', 'vin', 'bad', 100, 1, 1e9)

    @pytest.mark.parametrize('points,expected', [
        (10, 10),
        (10.0, 10),
        ('10', 10),
        ('10.0', 10),
        ('1k', 1000),
    ])
    def test_accepts_and_normalizes_positive_whole_point_counts(
            self, points, expected):
        analysis = Noise(
            'out', '0', 'vin', 'dec', points, 1, 1e6)

        assert analysis.points == expected
        assert type(analysis.points) is int
        assert analysis.to_spice().split()[4] == str(expected)

    @pytest.mark.parametrize(
        'points',
        [0, -1, 1.5, '10.5', True, float('-inf'), float('nan')],
    )
    def test_rejects_points_that_are_not_positive_whole_numbers(self, points):
        with pytest.raises(
                ValueError,
                match=(
                    r'Noise\.points must resolve to a positive whole number')):
            Noise('out', '0', 'vin', 'dec', points, 1, 1e6)

    @pytest.mark.parametrize('field,value', [
        ('fstart', 0),
        ('fstart', float('-inf')),
        ('fstop', -1),
        ('fstop', float('nan')),
        ('fstop', True),
    ])
    def test_rejects_invalid_frequencies(self, field, value):
        kwargs = {'output': 'out', 'ref': '0', 'source': 'vin',
                  'variation': 'dec', 'points': 10,
                  'fstart': 1, 'fstop': 1e6}
        kwargs[field] = value
        with pytest.raises(ValueError, match=rf'Noise\.{field}'):
            Noise(**kwargs)

    def test_rejects_descending_frequency_range(self):
        with pytest.raises(ValueError, match=r'Noise\.fstop'):
            Noise('out', '0', 'vin', 'dec', 10, 1e6, 1)


# ---------------------------------------------------------------------------
# add_sim_type() object-based path on PyNetlist
# ---------------------------------------------------------------------------

class TestAddSimTypeObjectPath:

    def test_transient_object_stored_under_transient_key(self, pdk, tmp_dir):
        """add_sim_type(Transient(...)) stores under key 'transient'."""
        from silk.circuit.modules import Module

        ckt = Module('PDK_skywater130', tmp_dir, 'sim_tran.cir', pdk_details=pdk)
        t = Transient(1e-9, 5e-6)
        ckt.add_sim_type(t)
        assert 'transient' in ckt.sim_analysis_dict
        assert ckt.sim_analysis_dict['transient'] is t

    def test_ac_object_stored_under_ac_key(self, pdk, tmp_dir):
        """add_sim_type(AC(...)) stores under key 'ac'."""
        from silk.circuit.modules import Module

        ckt = Module('PDK_skywater130', tmp_dir, 'sim_ac.cir', pdk_details=pdk)
        a = AC('dec', 10, 1e3, 1e9)
        ckt.add_sim_type(a)
        assert 'ac' in ckt.sim_analysis_dict
        assert ckt.sim_analysis_dict['ac'] is a

    def test_dc_object_stored_under_dc_key(self, pdk, tmp_dir):
        """add_sim_type(DC(...)) stores under key 'dc'."""
        from silk.circuit.modules import Module

        ckt = Module('PDK_skywater130', tmp_dir, 'sim_dc.cir', pdk_details=pdk)
        d = DC('vin', 0, 1.8, 0.01)
        ckt.add_sim_type(d)
        assert 'dc' in ckt.sim_analysis_dict
        assert ckt.sim_analysis_dict['dc'] is d


@pytest.mark.parametrize('kind', ['AC', 'Noise'])
@pytest.mark.parametrize('points,start,stop,minimum', [
    (1, '1k', '9k', 2), (3, 1000, 2000, 4), (1, 1000, 2000, 4),
])
def test_dec_requires_one_whole_step(kind, points, start, stop, minimum):
    factory = AC if kind == 'AC' else lambda *a: Noise('out', '0', 'vin', *a)
    with pytest.raises(ValueError, match=rf'{kind}.*dec.*at least {minimum} points per decade') as exc:
        factory('dec', points, start, stop)
    assert 'wider' in str(exc.value)
    assert 'lin' in str(exc.value)


@pytest.mark.parametrize('kind', ['AC', 'Noise'])
def test_dec_zero_span_requires_wider_range(kind):
    factory = AC if kind == 'AC' else lambda *a: Noise('out', '0', 'vin', *a)
    with pytest.raises(ValueError, match='zero span.*wider.*lin'):
        factory('dec', 10, '1k', '1k')


@pytest.mark.parametrize('variation,points,stop', [('dec', 1, 10000), ('dec', 4, 2000), ('oct', 1, 1500), ('lin', 1, 1000)])
def test_frequency_step_boundary_retains_emission(variation, points, stop):
    assert AC(variation, points, 1000, stop).to_spice().startswith(f'.ac {variation} {points} 1k ')


def test_dec_refuses_substep_created_by_spice_formatting():
    # Three points/decade reaches this exact mathematical step, but the
    # emitted 2.15443k is below it and would run away on ngspice 42.
    with pytest.raises(ValueError, match='SPICE formatting.*at least 4'):
        AC('dec', 3, 1000, 1000 * 10 ** (1 / 3))


def test_dec_does_not_round_a_substep_up_to_one():
    with pytest.raises(ValueError, match='at least 2'):
        AC('dec', 1, '1k', '9.999999999999998k')


def test_dec_uncapped_integer_points_and_zero_span():
    count = 10 ** 400
    assert AC('dec', count, 1, 2).points == count
    with pytest.raises(ValueError, match='zero span'):
        AC('dec', count, 1, 1)
