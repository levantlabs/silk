"""
Tests for the source classes in pynetlist.core.sources.

Covers VoltageSource.to_spice(), CurrentSource.to_spice(),
Pulse.to_spice(), Sine.to_spice(), and related waveform classes.
"""

import pytest
from silk import sources


# ---------------------------------------------------------------------------
# Pulse.to_spice()
# ---------------------------------------------------------------------------

class TestPulseToSpice:

    def test_pulse_keyword(self):
        """Pulse.to_spice() output contains 'PULSE'."""
        p = sources.Pulse(v1=0, v2=1.8, td=0, tr='1n', tf='1n', pw='50n', per='100n')
        assert 'PULSE(' in p.to_spice()

    def test_pulse_contains_v1_v2(self):
        """Pulse.to_spice() contains v1 and v2 values."""
        p = sources.Pulse(v1=0, v2=3.3, td=0, tr='1n', tf='1n', pw='10n', per='20n')
        result = p.to_spice()
        assert '3.3' in result

    def test_pulse_contains_period(self):
        """Pulse.to_spice() contains the period value."""
        p = sources.Pulse(v1=0, v2=1.8, td=0, tr='1n', tf='1n', pw='50n', per='100n')
        assert '100n' in p.to_spice()

    def test_pulse_np_omitted_when_none(self):
        """Pulse.to_spice() omits np when np=None."""
        p = sources.Pulse(v1=0, v2=1.8, td=0, tr='1n', tf='1n', pw=0, per=0)
        result = p.to_spice()
        # 7 values: v1 v2 td tr tf pw per
        parts = result[len('PULSE('):-1].split()
        assert len(parts) == 7

    def test_pulse_np_included_when_set(self):
        """Pulse.to_spice() includes np when np is not None."""
        p = sources.Pulse(v1=0, v2=1.8, td=0, tr='1n', tf='1n', pw=0, per=0, np=3)
        result = p.to_spice()
        parts = result[len('PULSE('):-1].split()
        assert len(parts) == 8
        assert parts[-1] == '3'


# ---------------------------------------------------------------------------
# Sine.to_spice()
# ---------------------------------------------------------------------------

class TestSineToSpice:

    def test_sine_keyword(self):
        """Sine.to_spice() output contains 'SIN('."""
        s = sources.Sine(voffset=0, vampl=0.5, freq='1meg', td=0, theta=0)
        assert 'SIN(' in s.to_spice()

    def test_sine_contains_amplitude(self):
        """Sine.to_spice() contains the amplitude value."""
        s = sources.Sine(voffset=0, vampl=1.0, freq='100k', td=0, theta=0)
        assert '1.0' in s.to_spice()

    def test_sine_contains_frequency(self):
        """Sine.to_spice() contains the frequency value."""
        s = sources.Sine(voffset=0, vampl=0.5, freq='100k', td=0, theta=0)
        assert '100k' in s.to_spice()

    def test_sine_phase_omitted_when_zero(self):
        """Sine.to_spice() omits phase when phase=0."""
        s = sources.Sine(voffset=0, vampl=0.5, freq='1meg', td=0, theta=0)
        result = s.to_spice()
        # 5 values: voffset vampl freq td theta
        parts = result[len('SIN('):-1].split()
        assert len(parts) == 5

    def test_sine_phase_included_when_nonzero(self):
        """Sine.to_spice() includes phase when phase != 0."""
        s = sources.Sine(voffset=0, vampl=0.5, freq='1meg', td=0, theta=0, phase=45)
        result = s.to_spice()
        parts = result[len('SIN('):-1].split()
        assert len(parts) == 6
        assert parts[-1] == '45'


# ---------------------------------------------------------------------------
# VoltageSource.to_spice() — DC only
# ---------------------------------------------------------------------------

class TestVoltageSourceDC:

    def test_dc_contains_dc_keyword(self):
        """DC voltage source output contains 'DC'."""
        vs = sources.VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8)
        result = vs.to_spice()
        assert 'DC' in result

    def test_dc_contains_value(self):
        """DC voltage source output contains the dc value."""
        vs = sources.VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8)
        assert '1.8' in vs.to_spice()

    def test_dc_prefix(self):
        """Voltage source name is prefixed with 'V'."""
        vs = sources.VoltageSource('supply', positive='vdd', negative='gnd', dc=3.3)
        assert 'Vsupply' in vs.to_spice()

    def test_dc_no_ampere_suffix(self):
        """Voltage DC must not end with a bare letter 'A' (ngspice would
        parse it as the SI prefix atto, 1e-18). SI prefixes like 'm', 'u'
        produced by _render_param are fine."""
        vs = sources.VoltageSource('vin', positive='in', negative='gnd', dc=0.9)
        line = vs.to_spice()
        # Must contain either a bare-number DC or a value with SI prefix
        assert 'DC ' in line
        # Never the atto trap
        assert '0.9A' not in line and '900mA' not in line

    def test_dc_contains_net_names(self):
        """DC voltage source output contains positive and negative net names."""
        vs = sources.VoltageSource('vsig', positive='signal_in', negative='signal_gnd', dc=0.9)
        result = vs.to_spice()
        assert 'signal_in' in result
        assert 'signal_gnd' in result


# ---------------------------------------------------------------------------
# VoltageSource.to_spice() — AC
# ---------------------------------------------------------------------------

class TestVoltageSourceAC:

    def test_ac_keyword_present(self):
        """AC voltage source output contains 'AC' keyword and amplitude."""
        vs = sources.VoltageSource('vac', positive='in', negative='gnd', dc=0.5, ac=1)
        result = vs.to_spice()
        assert 'AC' in result

    def test_ac_zero_not_included(self):
        """When AC amplitude is 0, 'AC' should not appear in output."""
        vs = sources.VoltageSource('vdc_only', positive='in', negative='gnd', dc=1.0, ac=0)
        assert 'AC' not in vs.to_spice()


# ---------------------------------------------------------------------------
# VoltageSource.to_spice() — PWL
# ---------------------------------------------------------------------------

class TestVoltageSourcePWL:

    def test_pwl_keyword(self):
        """PWL voltage source output contains 'PWL'."""
        pwl_data = [0, 0, 1e-9, 1.8, 10e-9, 1.8]
        vs = sources.VoltageSource('vpwl', positive='vdd', negative='gnd', pwl=pwl_data)
        assert 'PWL(' in vs.to_spice()

    def test_pwl_contains_time_values(self):
        """PWL output contains the provided time-value data."""
        pwl_data = [0, 0, 5e-9, 1.8]
        vs = sources.VoltageSource('vramp', positive='vdd', negative='gnd', pwl=pwl_data)
        result = vs.to_spice()
        # 5e-9 may be rendered as 5e-09 or 5e-9
        assert '5e-0' in result or '5e-9' in result

    def test_pwl_object(self):
        """PWL voltage source from a PWL object."""
        pwl = sources.PWL([(0, 0), (1e-9, 1.8), (10e-9, 1.8)])
        vs = sources.VoltageSource('vpwl', positive='vdd', negative='gnd', pwl=pwl)
        result = vs.to_spice()
        assert 'PWL(' in result
        assert '1.8' in result

    def test_pwl_tuple_list(self):
        """PWL from list of tuples."""
        vs = sources.VoltageSource('vpwl', positive='vdd', negative='gnd',
                                   pwl=[(0, 0), (1e-9, 1.8)])
        result = vs.to_spice()
        assert 'PWL(' in result

    def test_pwl_string_raises(self):
        """Passing a string as pwl raises TypeError (file= removed)."""
        import pytest
        with pytest.raises(TypeError, match="pwl must be"):
            sources.VoltageSource('vpwl', positive='vdd', negative='gnd',
                                 pwl='file=/etc/passwd')


# ---------------------------------------------------------------------------
# VoltageSource.to_spice() — Pulse waveform
# ---------------------------------------------------------------------------

class TestVoltageSourcePulse:

    def test_pulse_keyword_in_output(self):
        """Voltage source with pulse waveform contains 'PULSE'."""
        pulse = sources.Pulse(v1=0, v2=1.8, td=0, tr='1n', tf='1n', pw='50n', per='100n')
        vs = sources.VoltageSource('vclk', positive='clk', negative='gnd', pulse=pulse)
        assert 'PULSE(' in vs.to_spice()

    def test_pulse_contains_parameters(self):
        """Pulse waveform output contains v2 and period values."""
        pulse = sources.Pulse(v1=0, v2=3.3, td='1n', tr='1n', tf='1n', pw='10n', per='20n')
        vs = sources.VoltageSource('vpulse', positive='out', negative='gnd', pulse=pulse)
        result = vs.to_spice()
        assert '3.3' in result
        assert '20n' in result

    def test_pulse_emitted_on_separate_line(self):
        """Pulse source is emitted on a separate line from the DC component."""
        pulse = sources.Pulse(v1=0, v2=1.8, td=0, tr='1n', tf='1n', pw='50n', per='100n')
        vs = sources.VoltageSource('vp', positive='a', negative='b', dc=0, pulse=pulse)
        result = vs.to_spice()
        assert '\n' in result


# ---------------------------------------------------------------------------
# VoltageSource.to_spice() — Sine waveform
# ---------------------------------------------------------------------------

class TestVoltageSourceSine:

    def test_sine_keyword_in_output(self):
        """Voltage source with sine waveform contains 'SIN'."""
        sine = sources.Sine(voffset=0, vampl=0.5, freq='1meg', td=0, theta=0)
        vs = sources.VoltageSource('vsin', positive='in', negative='gnd', sine=sine)
        assert 'SIN(' in vs.to_spice()

    def test_sine_contains_amplitude_and_freq(self):
        """Sine waveform output contains amplitude and frequency values."""
        sine = sources.Sine(voffset=0, vampl=1.0, freq='100k', td=0, theta=0)
        vs = sources.VoltageSource('vac', positive='inp', negative='gnd', sine=sine)
        result = vs.to_spice()
        assert '1.0' in result
        assert '100k' in result


# ---------------------------------------------------------------------------
# CurrentSource.to_spice()
# ---------------------------------------------------------------------------

class TestCurrentSourceDC:

    def test_dc_current_prefix(self):
        """DC current source name is prefixed with 'I'."""
        cs = sources.CurrentSource('ibias', positive='vdd', negative='gnd', dc=100e-6)
        assert 'Iibias' in cs.to_spice()

    def test_dc_current_no_ampere_suffix(self):
        """DC current value must NOT end with a bare 'A' — ngspice parses
        it as the SI prefix atto (1e-18), making the current effectively
        zero. SI prefixes like 'm', 'u' produced by _render_param are fine
        (they are valid SPICE magnitudes)."""
        cs = sources.CurrentSource('ibias', positive='vdd', negative='gnd', dc=0.001)
        line = cs.to_spice()
        assert 'DC ' in line
        # The atto trap: '0.001A' or '1mA' would both be wrong
        assert '0.001A' not in line and '1mA' not in line

    def test_dc_current_contains_value(self):
        """DC current source output contains the dc value (either as a
        bare number or formatted with an SI magnitude prefix by _render_param)."""
        cs = sources.CurrentSource('ibias', positive='vdd', negative='gnd', dc=100e-6)
        line = cs.to_spice()
        # Accept any representation of 100 uA
        assert ('0.0001' in line or '1e-04' in line or '1e-4' in line
                or '100u' in line or '0.1m' in line)

    def test_dc_current_contains_nets(self):
        """DC current source output contains positive and negative net names."""
        cs = sources.CurrentSource('ibias', positive='vdd', negative='gnd', dc=1e-3)
        result = cs.to_spice()
        assert 'vdd' in result
        assert 'gnd' in result


# ---------------------------------------------------------------------------
# CurrentSource.to_spice() — PWL
# ---------------------------------------------------------------------------

class TestCurrentSourcePWL:

    def test_pwl_current_contains_pwl_keyword(self):
        """PWL current source output contains 'PWL'."""
        pwl_data = [0, 0, 1e-9, 0.001, 10e-9, 0.001]
        cs = sources.CurrentSource('ipwl', positive='vdd', negative='gnd', pwl=pwl_data)
        result = cs.to_spice()
        assert 'PWL(' in result
        assert 'Iipwl' in result
        assert 'vdd' in result
        assert 'gnd' in result


# ---------------------------------------------------------------------------
# CurrentSource.to_spice() — PWL from file
# ---------------------------------------------------------------------------

class TestCurrentSourcePWLFromFile:

    def test_pwl_current_object(self):
        """PWL current source from a PWL object."""
        pwl = sources.PWL([(0, 0), (1e-9, 0.001), (10e-9, 0.001)])
        cs = sources.CurrentSource('ipwl', positive='vdd', negative='gnd', pwl=pwl)
        result = cs.to_spice()
        assert 'PWL(' in result
        assert '0.001' in result


# ---------------------------------------------------------------------------
# Exponential.to_spice()
# ---------------------------------------------------------------------------

class TestExponentialToSpice:

    def test_exponential_keyword(self):
        """Exponential.to_spice() output contains 'EXP('."""
        e = sources.Exponential(v1=0, v2=1.8)
        assert 'EXP(' in e.to_spice()

    def test_exponential_contains_v1_v2(self):
        """Exponential.to_spice() output contains v1 and v2 values."""
        e = sources.Exponential(v1=0, v2=3.3)
        result = e.to_spice()
        assert '3.3' in result

    def test_exponential_contains_all_six_params(self):
        """Exponential.to_spice() always outputs all six parameters."""
        e = sources.Exponential(v1=0, v2=1.8, td1='1n', tau1='5n', td2='10n', tau2='20n')
        result = e.to_spice()
        parts = result[len('EXP('):-1].split()
        assert len(parts) == 6

    def test_exponential_contains_tau_values(self):
        """Exponential.to_spice() contains tau1 and tau2 values."""
        e = sources.Exponential(v1=0, v2=1.8, td1='1n', tau1='5n', td2='10n', tau2='20n')
        result = e.to_spice()
        assert '5n' in result
        assert '20n' in result


# ---------------------------------------------------------------------------
# SFFM.to_spice()
# ---------------------------------------------------------------------------

class TestSFFMToSpice:

    def test_sffm_keyword(self):
        """SFFM.to_spice() output contains 'SFFM('."""
        s = sources.SFFM(vo=0, va=1, fc='1meg', mdi=1, fs='1k')
        assert 'SFFM(' in s.to_spice()

    def test_sffm_contains_key_params(self):
        """SFFM.to_spice() contains vo, va, fc, mdi, fs values."""
        s = sources.SFFM(vo=0, va=2.5, fc='2meg', mdi=0.5, fs='10k')
        result = s.to_spice()
        assert '2.5' in result
        assert '2meg' in result
        assert '10k' in result

    def test_sffm_phase_omitted_when_both_zero(self):
        """SFFM.to_spice() omits phasec/phases when both are 0."""
        s = sources.SFFM(vo=0, va=1, fc='1meg', mdi=1, fs='1k', phasec=0, phases=0)
        result = s.to_spice()
        parts = result[len('SFFM('):-1].split()
        assert len(parts) == 5

    def test_sffm_phase_included_when_phasec_nonzero(self):
        """SFFM.to_spice() includes phasec and phases when phasec != 0."""
        s = sources.SFFM(vo=0, va=1, fc='1meg', mdi=1, fs='1k', phasec=45, phases=0)
        result = s.to_spice()
        parts = result[len('SFFM('):-1].split()
        assert len(parts) == 7
        assert '45' in result

    def test_sffm_phase_included_when_phases_nonzero(self):
        """SFFM.to_spice() includes phasec and phases when phases != 0."""
        s = sources.SFFM(vo=0, va=1, fc='1meg', mdi=1, fs='1k', phasec=0, phases=30)
        result = s.to_spice()
        parts = result[len('SFFM('):-1].split()
        assert len(parts) == 7
        assert '30' in result

    def test_sffm_alias(self):
        """SingleFrequencyFM is the same class as SFFM."""
        assert sources.SingleFrequencyFM is sources.SFFM


# ---------------------------------------------------------------------------
# AM.to_spice() (AmplitudeModulated)
# ---------------------------------------------------------------------------

class TestAMToSpice:

    def test_am_alias(self):
        """AmplitudeModulated is the same class as AM."""
        assert sources.AmplitudeModulated is sources.AM

    def test_am_keyword(self):
        """AM.to_spice() output contains 'AM('."""
        a = sources.AM(va=1, vo=0, mf='1k', fc='1meg')
        assert 'AM(' in a.to_spice()

    def test_am_phases_omitted_when_zero(self):
        """AM.to_spice() omits phases when phases=0."""
        a = sources.AM(va=1, vo=0, mf='1k', fc='1meg', td=0, phases=0)
        parts = a.to_spice()[len('AM('):-1].split()
        assert len(parts) == 5

    def test_am_phases_included_when_nonzero(self):
        """AM.to_spice() includes phases when phases != 0."""
        a = sources.AM(va=1, vo=0, mf='1k', fc='1meg', td=0, phases=90)
        result = a.to_spice()
        parts = result[len('AM('):-1].split()
        assert len(parts) == 6
        assert '90' in result


# ---------------------------------------------------------------------------
# TransientNoise.to_spice()
# ---------------------------------------------------------------------------

class TestTransientNoiseToSpice:

    def test_trnoise_keyword(self):
        """TransientNoise.to_spice() output contains 'TRNOISE('."""
        n = sources.TransientNoise(na='20n', nt='0.5n')
        assert 'TRNOISE(' in n.to_spice()

    def test_trnoise_contains_na_nt(self):
        """TransientNoise.to_spice() contains na and nt values."""
        n = sources.TransientNoise(na='20n', nt='0.5n')
        result = n.to_spice()
        assert '20n' in result
        assert '0.5n' in result

    def test_trnoise_rts_omitted_when_all_zero(self):
        """TransientNoise.to_spice() omits RTS params when all three are 0."""
        n = sources.TransientNoise(na='20n', nt='0.5n', nalpha=1.1, namp='12p',
                                   rtsam=0, rtscapt=0, rtsemt=0)
        parts = n.to_spice()[len('TRNOISE('):-1].split()
        assert len(parts) == 4

    def test_trnoise_rts_included_when_rtsam_nonzero(self):
        """TransientNoise.to_spice() includes RTS params when rtsam != 0."""
        n = sources.TransientNoise(na=0, nt='10p', nalpha=1.1, namp='12p',
                                   rtsam='15m', rtscapt='22u', rtsemt='50u')
        result = n.to_spice()
        parts = result[len('TRNOISE('):-1].split()
        assert len(parts) == 7
        assert '15m' in result

    def test_trnoise_rts_included_when_rtscapt_nonzero(self):
        """TransientNoise.to_spice() includes RTS params when rtscapt != 0."""
        n = sources.TransientNoise(na='1n', nt='1n', rtsam=0, rtscapt='5u', rtsemt=0)
        parts = n.to_spice()[len('TRNOISE('):-1].split()
        assert len(parts) == 7

    def test_trnoise_rts_included_when_rtsemt_nonzero(self):
        """TransientNoise.to_spice() includes RTS params when rtsemt != 0."""
        n = sources.TransientNoise(na='1n', nt='1n', rtsam=0, rtscapt=0, rtsemt='10u')
        parts = n.to_spice()[len('TRNOISE('):-1].split()
        assert len(parts) == 7


# ---------------------------------------------------------------------------
# RandomSource.to_spice()
# ---------------------------------------------------------------------------

class TestRandomSourceToSpice:

    def test_trrandom_keyword(self):
        """RandomSource.to_spice() output contains 'TRRANDOM('."""
        r = sources.RandomSource(type=1, ts='1n')
        assert 'TRRANDOM(' in r.to_spice()

    def test_trrandom_contains_type_and_ts(self):
        """RandomSource.to_spice() contains type and ts values."""
        r = sources.RandomSource(type=2, ts='5n')
        result = r.to_spice()
        assert '2' in result
        assert '5n' in result

    def test_trrandom_contains_all_five_params(self):
        """RandomSource.to_spice() always outputs all five parameters."""
        r = sources.RandomSource(type=3, ts='1n', td='2n', param1=0.5, param2=1.0)
        parts = r.to_spice()[len('TRRANDOM('):-1].split()
        assert len(parts) == 5

    def test_trrandom_contains_param_values(self):
        """RandomSource.to_spice() contains param1 and param2 values."""
        r = sources.RandomSource(type=4, ts='1n', td=0, param1=2.5, param2=0.3)
        result = r.to_spice()
        assert '2.5' in result
        assert '0.3' in result


# ---------------------------------------------------------------------------
# SourceBase.to_spice() raises NotImplementedError
# ---------------------------------------------------------------------------

class TestSourceBaseNotImplemented:

    def test_to_spice_raises(self):
        """SourceBase.to_spice() raises NotImplementedError."""
        sb = sources.SourceBase('base')
        with pytest.raises(NotImplementedError):
            sb.to_spice()


# ---------------------------------------------------------------------------
# VoltageSource.to_spice() — additional waveform types
# ---------------------------------------------------------------------------

class TestVoltageSourceWaveforms:

    def test_vs_with_exponential(self):
        """VoltageSource with exponential waveform: two cards, contains EXP."""
        exp = sources.Exponential(v1=0, v2=1.8, td1='1n', tau1='5n', td2='10n', tau2='20n')
        vs = sources.VoltageSource('vexp', positive='in', negative='gnd', exponential=exp)
        result = vs.to_spice()
        assert 'EXP(' in result
        assert 'Vvexp' in result
        assert len(result.splitlines()) == 2
        assert ' DC 0' in result.splitlines()[1]

    def test_vs_with_sffm(self):
        """VoltageSource with SFFM waveform: two cards, contains SFFM."""
        sffm = sources.SFFM(vo=0, va=1, fc='1meg', mdi=1, fs='1k')
        vs = sources.VoltageSource('vfm', positive='in', negative='gnd', sffm=sffm)
        result = vs.to_spice()
        assert 'SFFM(' in result
        assert 'Vvfm' in result
        assert len(result.splitlines()) == 2
        assert ' DC 0' in result.splitlines()[1]

    def test_vs_with_am(self):
        """VoltageSource with AM waveform: two cards, contains AM."""
        am = sources.AM(va=1, vo=0, mf='1k', fc='1meg')
        vs = sources.VoltageSource('vam', positive='in', negative='gnd', am=am)
        result = vs.to_spice()
        assert 'AM(' in result
        assert 'Vvam' in result
        assert len(result.splitlines()) == 2
        assert ' DC 0' in result.splitlines()[1]

    def test_vs_with_trnoise(self):
        """VoltageSource with trnoise waveform: two cards, contains TRNOISE."""
        noise = sources.TransientNoise(na='20n', nt='0.5n')
        vs = sources.VoltageSource('vnoise', positive='in', negative='gnd', trnoise=noise)
        result = vs.to_spice()
        assert 'TRNOISE(' in result
        assert 'Vvnoise' in result
        assert len(result.splitlines()) == 2
        assert ' DC 0' in result.splitlines()[1]

    def test_vs_with_trrandom(self):
        """VoltageSource with trrandom waveform: two cards, contains TRRANDOM."""
        rand = sources.RandomSource(type=2, ts='1n')
        vs = sources.VoltageSource('vrand', positive='in', negative='gnd', trrandom=rand)
        result = vs.to_spice()
        assert 'TRRANDOM(' in result
        assert 'Vvrand' in result
        assert len(result.splitlines()) == 2
        assert ' DC 0' in result.splitlines()[1]

    def test_vs_dc_nonzero_with_waveform(self):
        """VoltageSource with dc != 0 and waveform adds a separate DC card after the waveform."""
        sine = sources.Sine(voffset=0, vampl=0.5, freq='1meg')
        vs = sources.VoltageSource('vbias', positive='in', negative='gnd',
                                   dc=1.0, sine=sine)
        result = vs.to_spice()
        # dc=1.0 may render as '1.0', '1', or SI-prefixed form
        assert 'DC 1' in result
        assert 'SIN(' in result
        # The waveform card is followed by the additive DC card.
        assert result.index('SIN(') < result.index('DC')

    def test_vs_pwl_r_in_output(self):
        """VoltageSource with pwl_r includes 'r=val' in PWL line."""
        pwl_data = [0, 0, 1e-9, 1.8]
        vs = sources.VoltageSource('vpwl', positive='in', negative='gnd',
                                   pwl=pwl_data, pwl_r='5n')
        result = vs.to_spice()
        assert 'r=5n' in result

    def test_vs_pwl_td_in_output(self):
        """VoltageSource with pwl_td includes 'td=val' in PWL line."""
        pwl_data = [0, 0, 1e-9, 1.8]
        vs = sources.VoltageSource('vpwl', positive='in', negative='gnd',
                                   pwl=pwl_data, pwl_td='10n')
        result = vs.to_spice()
        assert 'td=10n' in result

    def test_vs_pwl_with_ac(self):
        """VoltageSource with PWL and ac != 0 includes 'AC' in the dc line."""
        pwl_data = [0, 0, 1e-9, 1.8]
        vs = sources.VoltageSource('vpwl', positive='in', negative='gnd',
                                   pwl=pwl_data, ac=1)
        result = vs.to_spice()
        assert 'AC' in result
        # The result has two lines
        lines = result.strip().split('\n')
        assert len(lines) == 2
        dc_line = lines[1]
        assert 'AC' in dc_line

    def test_vs_pwl_with_r_and_td(self):
        """VoltageSource with PWL repeat and delay parameters."""
        pwl = sources.PWL([(0, 0), (2e-9, 1.8), (20e-9, 1.8)], r=5e-9, td=1e-9)
        vs = sources.VoltageSource('vpwl', positive='in', negative='gnd', pwl=pwl)
        result = vs.to_spice()
        assert 'PWL(' in result
        assert 'r=' in result
        assert 'td=' in result


# ---------------------------------------------------------------------------
# CurrentSource.to_spice() — additional waveform types
# ---------------------------------------------------------------------------

class TestCurrentSourceWaveforms:

    def test_cs_with_pulse(self):
        """CurrentSource with pulse waveform: two cards, contains PULSE, prefix I."""
        pulse = sources.Pulse(v1=0, v2=1e-3, td=0, tr='1n', tf='1n', pw='50n', per='100n')
        cs = sources.CurrentSource('ipulse', positive='out', negative='gnd', pulse=pulse)
        result = cs.to_spice()
        assert 'PULSE(' in result
        assert 'Iipulse' in result

    def test_cs_with_sine(self):
        """CurrentSource with sine waveform: two cards, contains SIN, prefix I."""
        sine = sources.Sine(voffset=0, vampl=1e-3, freq='1meg')
        cs = sources.CurrentSource('isine', positive='out', negative='gnd', sine=sine)
        result = cs.to_spice()
        assert 'SIN(' in result
        assert 'Iisine' in result

    def test_cs_with_exponential(self):
        """CurrentSource with exponential waveform: contains EXP, prefix I."""
        exp = sources.Exponential(v1=0, v2=1e-3, td1='1n', tau1='5n', td2='10n', tau2='20n')
        cs = sources.CurrentSource('iexp', positive='out', negative='gnd', exponential=exp)
        result = cs.to_spice()
        assert 'EXP(' in result
        assert 'Iiexp' in result

    def test_cs_with_sffm(self):
        """CurrentSource with SFFM waveform: contains SFFM, prefix I."""
        sffm = sources.SFFM(vo=0, va=1e-3, fc='1meg', mdi=1, fs='1k')
        cs = sources.CurrentSource('ifm', positive='out', negative='gnd', sffm=sffm)
        result = cs.to_spice()
        assert 'SFFM(' in result
        assert 'Iifm' in result

    def test_cs_with_am(self):
        """CurrentSource with AM waveform: contains AM, prefix I."""
        am = sources.AM(va=1e-3, vo=0, mf='1k', fc='1meg')
        cs = sources.CurrentSource('iam', positive='out', negative='gnd', am=am)
        result = cs.to_spice()
        assert 'AM(' in result
        assert 'Iiam' in result

    def test_cs_with_trnoise(self):
        """CurrentSource with trnoise waveform: contains TRNOISE, prefix I."""
        noise = sources.TransientNoise(na='1p', nt='0.5n')
        cs = sources.CurrentSource('inoise', positive='out', negative='gnd', trnoise=noise)
        result = cs.to_spice()
        assert 'TRNOISE(' in result
        assert 'Iinoise' in result

    def test_cs_with_trrandom(self):
        """CurrentSource with trrandom waveform: contains TRRANDOM, prefix I."""
        rand = sources.RandomSource(type=1, ts='1n')
        cs = sources.CurrentSource('irand', positive='out', negative='gnd', trrandom=rand)
        result = cs.to_spice()
        assert 'TRRANDOM(' in result
        assert 'Iirand' in result

    def test_cs_pwl_flat_list(self):
        """CurrentSource with pwl as flat list (backward compat)."""
        cs = sources.CurrentSource('ipwlf', positive='out', negative='gnd',
                                   pwl=[0, 0, 3e-9, 0.5e-3])
        result = cs.to_spice()
        assert 'PWL(' in result
        assert '0.0005' in result


# ---------------------------------------------------------------------------
# Dependent sources: VCVS / VoltageControlledVoltageSource
# ---------------------------------------------------------------------------

class TestVCVS:

    def test_vcvs_prefix_e(self):
        """VCVS to_spice() starts with 'E' prefix."""
        e = sources.VoltageControlledVoltageSource(
            'amp', out_pos='vout', out_neg='gnd',
            ctrl_pos='vin', ctrl_neg='gnd', gain=10)
        result = e.to_spice()
        assert result.startswith('Eamp')

    def test_vcvs_contains_all_nodes(self):
        """VCVS to_spice() contains all four node names."""
        e = sources.VoltageControlledVoltageSource(
            'e1', out_pos='vo', out_neg='gnd',
            ctrl_pos='vi', ctrl_neg='ref', gain=5)
        result = e.to_spice()
        assert 'vo' in result
        assert 'gnd' in result
        assert 'vi' in result
        assert 'ref' in result

    def test_vcvs_contains_gain(self):
        """VCVS to_spice() contains the gain value."""
        e = sources.VoltageControlledVoltageSource(
            'e1', out_pos='vo', out_neg='gnd',
            ctrl_pos='vi', ctrl_neg='gnd', gain=100)
        assert '100' in e.to_spice()

    def test_vcvs_alias(self):
        """VCVS is the same class as VoltageControlledVoltageSource."""
        assert sources.VCVS is sources.VoltageControlledVoltageSource


# ---------------------------------------------------------------------------
# Dependent sources: VCCS / VoltageControlledCurrentSource
# ---------------------------------------------------------------------------

class TestVCCS:

    def test_vccs_prefix_g(self):
        """VCCS to_spice() starts with 'G' prefix."""
        g = sources.VoltageControlledCurrentSource(
            'gm1', out_pos='io', out_neg='gnd',
            ctrl_pos='vin', ctrl_neg='gnd', gm=0.01)
        assert g.to_spice().startswith('Ggm1')

    def test_vccs_contains_all_nodes(self):
        """VCCS to_spice() contains all four node names."""
        g = sources.VoltageControlledCurrentSource(
            'g1', out_pos='iout', out_neg='gnd',
            ctrl_pos='vctrl', ctrl_neg='ref', gm=0.005)
        result = g.to_spice()
        assert 'iout' in result
        assert 'vctrl' in result
        assert 'ref' in result

    def test_vccs_contains_gm(self):
        """VCCS to_spice() contains the gm value."""
        g = sources.VoltageControlledCurrentSource(
            'g1', out_pos='io', out_neg='gnd',
            ctrl_pos='vin', ctrl_neg='gnd', gm=0.025)
        assert '0.025' in g.to_spice()

    def test_vccs_alias(self):
        """VCCS is the same class as VoltageControlledCurrentSource."""
        assert sources.VCCS is sources.VoltageControlledCurrentSource


# ---------------------------------------------------------------------------
# Dependent sources: CCVS / CurrentControlledVoltageSource
# ---------------------------------------------------------------------------

class TestCCVS:

    def test_ccvs_prefix_h(self):
        """CCVS to_spice() starts with 'H' prefix."""
        h = sources.CurrentControlledVoltageSource(
            'htrans', out_pos='vout', out_neg='gnd',
            transresistance=1000, ctrl_vname='vmeas')
        assert h.to_spice().startswith('Hhtrans')

    def test_ccvs_contains_all_nodes(self):
        """CCVS to_spice() contains out_pos, out_neg, and ctrl_vname."""
        h = sources.CurrentControlledVoltageSource(
            'h1', out_pos='vo', out_neg='gnd',
            transresistance=500, ctrl_vname='vsense')
        result = h.to_spice()
        assert 'vo' in result
        assert 'gnd' in result
        assert 'vsense' in result

    def test_ccvs_contains_transresistance(self):
        """CCVS to_spice() contains the transresistance value."""
        h = sources.CurrentControlledVoltageSource(
            'h1', out_pos='vo', out_neg='gnd',
            transresistance=2200, ctrl_vname='vm')
        assert '2200' in h.to_spice()

    def test_ccvs_alias(self):
        """CCVS is the same class as CurrentControlledVoltageSource."""
        assert sources.CCVS is sources.CurrentControlledVoltageSource


# ---------------------------------------------------------------------------
# Dependent sources: CCCS / CurrentControlledCurrentSource
# ---------------------------------------------------------------------------

class TestCCCS:

    def test_cccs_prefix_f(self):
        """CCCS to_spice() starts with 'F' prefix."""
        f = sources.CurrentControlledCurrentSource(
            'fmir', out_pos='icopy', out_neg='gnd',
            gain=2, ctrl_vname='vmeas')
        assert f.to_spice().startswith('Ffmir')

    def test_cccs_contains_all_nodes(self):
        """CCCS to_spice() contains out_pos, out_neg, and ctrl_vname."""
        f = sources.CurrentControlledCurrentSource(
            'f1', out_pos='iout', out_neg='gnd',
            gain=4, ctrl_vname='vref')
        result = f.to_spice()
        assert 'iout' in result
        assert 'gnd' in result
        assert 'vref' in result

    def test_cccs_contains_gain(self):
        """CCCS to_spice() contains the gain value."""
        f = sources.CurrentControlledCurrentSource(
            'f1', out_pos='io', out_neg='gnd',
            gain=8, ctrl_vname='vm')
        assert '8' in f.to_spice()

    def test_cccs_alias(self):
        """CCCS is the same class as CurrentControlledCurrentSource."""
        assert sources.CCCS is sources.CurrentControlledCurrentSource


# ---------------------------------------------------------------------------
# BehavioralVoltage.to_spice()
# ---------------------------------------------------------------------------

class TestBehavioralVoltage:

    def test_bvoltage_prefix_b(self):
        """BehavioralVoltage to_spice() starts with 'B'."""
        b = sources.BehavioralVoltage('bdiff', positive='vd', negative='gnd',
                                      expression='v(inp)-v(inn)')
        assert b.to_spice().startswith('B')

    def test_bvoltage_contains_v_eq(self):
        """BehavioralVoltage to_spice() contains 'V={'."""
        b = sources.BehavioralVoltage('bdiff', positive='vd', negative='gnd',
                                      expression='v(inp)-v(inn)')
        assert 'V={' in b.to_spice()

    def test_bvoltage_contains_expression(self):
        """BehavioralVoltage to_spice() contains the expression string."""
        expr = 'v(inp)-v(inn)'
        b = sources.BehavioralVoltage('bdiff', positive='vd', negative='gnd',
                                      expression=expr)
        assert expr in b.to_spice()

    def test_bvoltage_contains_nodes(self):
        """BehavioralVoltage to_spice() contains positive and negative node names."""
        b = sources.BehavioralVoltage('b1', positive='mypos', negative='myneg',
                                      expression='1')
        result = b.to_spice()
        assert 'mypos' in result
        assert 'myneg' in result


# ---------------------------------------------------------------------------
# BehavioralCurrent.to_spice()
# ---------------------------------------------------------------------------

class TestBehavioralCurrent:

    def test_bcurrent_prefix_b(self):
        """BehavioralCurrent to_spice() starts with 'B'."""
        b = sources.BehavioralCurrent('bgm', positive='io', negative='gnd',
                                      expression='0.01*v(vin,gnd)')
        assert b.to_spice().startswith('B')

    def test_bcurrent_contains_i_eq(self):
        """BehavioralCurrent to_spice() contains 'I={'."""
        b = sources.BehavioralCurrent('bgm', positive='io', negative='gnd',
                                      expression='0.01*v(vin,gnd)')
        assert 'I={' in b.to_spice()

    def test_bcurrent_contains_expression(self):
        """BehavioralCurrent to_spice() contains the expression string."""
        expr = '0.01*v(vin,gnd)'
        b = sources.BehavioralCurrent('bgm', positive='io', negative='gnd',
                                      expression=expr)
        assert expr in b.to_spice()

    def test_bcurrent_contains_nodes(self):
        """BehavioralCurrent to_spice() contains positive and negative node names."""
        b = sources.BehavioralCurrent('b1', positive='ipos', negative='ineg',
                                      expression='1')
        result = b.to_spice()
        assert 'ipos' in result
        assert 'ineg' in result


class TestToNumpyUnmodeledWaveforms:
    """to_numpy() must fail loudly on waveforms with no numpy model —
    silently returning the DC value corrupted behavioral results."""

    def test_vsource_exponential_raises(self):
        import numpy as np
        v = sources.VoltageSource(
            'vexp', positive='a', negative='0', dc=0.5,
            exponential=sources.Exponential(v1=0, v2=1, td1=0, tau1=1e-9,
                                            td2=5e-9, tau2=1e-9))
        with pytest.raises(NotImplementedError, match='exponential'):
            v.to_numpy(np.linspace(0, 1e-8, 10))

    def test_isource_trnoise_raises(self):
        import numpy as np
        i = sources.CurrentSource(
            'inoise', positive='a', negative='0', dc=1e-6,
            trnoise=sources.TransientNoise(na=1e-9, nt=1e-10))
        with pytest.raises(NotImplementedError, match='trnoise'):
            i.to_numpy(np.linspace(0, 1e-8, 10))

    def test_vsource_supported_waveforms_still_work(self):
        import numpy as np
        t = np.linspace(0, 1e-8, 11)
        v_dc = sources.VoltageSource('vd', positive='a', negative='0', dc=1.8)
        assert np.allclose(v_dc.to_numpy(t), 1.8)
        v_sin = sources.VoltageSource(
            'vs', positive='a', negative='0',
            sine=sources.Sine(voffset=0, vampl=1, freq=1e8))
        assert v_sin.to_numpy(t).shape == t.shape
