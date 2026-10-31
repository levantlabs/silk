"""Pulse rise/fall times are mandatory and nonzero — and that makes silk's
behavioral twin agree with ngspice.

A zero ``tr``/``tf`` is not a well-defined waveform. ngspice inherits SPICE3's
rule of substituting the transient step ``TSTEP`` for a zero edge and ramping
over it; silk's behavioral ``to_numpy`` stepped instantaneously. The two
therefore disagreed *at t=0 by the full supply* and skewed by TSTEP at every
later edge, which is exactly the failure mode ``test_twin_matches_ngspice``
below pins shut.
"""
import shutil
import subprocess

import numpy as np
import pytest

from silk.signals.rawfile import read_rawfile
from silk.simulation.runner import ngspice_executable
from silk.sources.classes import CurrentSource, VoltageSource
from silk.sources.waveforms import Pulse


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

class TestEdgesAreMandatory:
    def test_omitting_tr_and_tf_raises(self):
        with pytest.raises(TypeError, match='tr'):
            Pulse(v1=0, v2=1.8, pw=5e-9, per=10e-9)

    def test_omitting_only_tf_raises(self):
        with pytest.raises(TypeError, match='tf'):
            Pulse(v1=0, v2=1.8, tr=1e-9, pw=5e-9, per=10e-9)

    @pytest.mark.parametrize('bad', [0, 0.0, '0', '0n', '0.0', -1e-12, '-1n'])
    @pytest.mark.parametrize('edge', ['tr', 'tf'])
    def test_zero_or_negative_edge_rejected(self, edge, bad):
        kwargs = {'tr': 1e-9, 'tf': 1e-9, 'pw': 5e-9, 'per': 10e-9}
        kwargs[edge] = bad
        with pytest.raises(ValueError, match=f'Pulse {edge}='):
            Pulse(v1=0, v2=1.8, **kwargs)

    def test_error_says_why_and_what_to_do(self):
        """Fail-loud message must name the mechanism, not just the rule."""
        with pytest.raises(ValueError) as exc:
            Pulse(v1=0, v2=1.8, tr=0, tf=1e-9)
        msg = str(exc.value)
        assert 'TSTEP' in msg and 'to_numpy' in msg
        assert '100e-12' in msg          # a concrete edge time to copy

    @pytest.mark.parametrize('good', [1e-9, 100e-12, '1n', '100p', '0.1n',
                                      '1ns', 1, '1.5e-10'])
    @pytest.mark.parametrize('edge', ['tr', 'tf'])
    def test_positive_edges_accepted(self, edge, good):
        kwargs = {'tr': 1e-9, 'tf': 1e-9, 'pw': 5e-9, 'per': 10e-9}
        kwargs[edge] = good
        assert Pulse(v1=0, v2=1.8, **kwargs) is not None

    def test_authored_value_survives_into_the_netlist(self):
        """Validation parses SI strings but must not rewrite them — the
        emitted deck stays byte-identical to what the caller wrote."""
        p = Pulse(v1=0, v2=1.8, td=0, tr='1n', tf='1n', pw='5n', per='10n')
        assert p.to_spice() == 'PULSE(0 1.8 0 1n 1n 5n 10n)'

    def test_rejection_reaches_the_declarative_factory(self):
        """sources.vpulse(...) delegates to Pulse, so it inherits the rule."""
        import silk.sources as sources
        with pytest.raises(ValueError, match='Pulse tr='):
            sources.vpulse('clk', 'clk', '0', v1=0, v2=1.8, tr=0, tf=0)


class TestToNumpyParsesSIStrings:
    """A pulse written with SI strings must model identically to the same
    pulse written with floats. Before edge validation forced these values
    through the SPICE parser, ``to_numpy`` raised on a string ``tr``."""

    def test_string_and_float_pulses_model_identically(self):
        t = np.linspace(0, 20e-9, 401)
        as_str = Pulse(v1=0, v2=1.8, td='1n', tr='1n', tf='1n',
                       pw='5n', per='10n').to_numpy(t)
        as_num = Pulse(v1=0, v2=1.8, td=1e-9, tr=1e-9, tf=1e-9,
                       pw=5e-9, per=10e-9).to_numpy(t)
        assert np.allclose(as_str, as_num)

    def test_current_source_pulse_models_with_strings(self):
        src = CurrentSource('iload', positive='vout', negative='gnd',
                            pulse=Pulse(v1=0, v2='1m', td=0, tr='1n', tf='1n',
                                        pw='5n', per='10n'))
        y = src.to_numpy(np.array([0.0, 0.5e-9, 3e-9]))
        assert y == pytest.approx([0.0, 0.5e-3, 1e-3])


# --------------------------------------------------------------------------
# The bug this change exists to kill: twin vs simulator
# --------------------------------------------------------------------------

# One pulse, exercised by both engines below.
_TR = _TF = 1e-9
_PW, _PER, _V2 = 5e-9, 20e-9, 1.8
# Segment layout: rise [0,1n), high [1n,6n), fall [6n,7n), low [7n,20n).
# Probe t=0 and the midpoint of each edge — t=0 is where the zero-edge twin
# and ngspice used to differ by the full 1.8 V supply.
_PROBE_TIMES = np.array([0.0, 0.5e-9, 1e-9, 3.5e-9, 6e-9,
                         6.5e-9, 7e-9, 12e-9, 20e-9])
_EXPECTED = np.array([0.0, 0.9, 1.8, 1.8, 1.8, 0.9, 0.0, 0.0, 0.0])


def _pulse():
    return Pulse(v1=0, v2=_V2, td=0, tr=_TR, tf=_TF, pw=_PW, per=_PER)


def test_twin_matches_hand_computed_waveform():
    """The behavioral twin ramps over tr — no instantaneous step at t=0."""
    assert _pulse().to_numpy(_PROBE_TIMES) == pytest.approx(_EXPECTED, abs=1e-12)


@pytest.mark.slow
@pytest.mark.skipif(shutil.which(ngspice_executable()) is None,
                    reason='ngspice not installed')
def test_twin_matches_ngspice(tmp_path):
    """Same Pulse object, two engines, one answer.

    The source line is emitted by ``Pulse.to_spice()`` and the reference
    values come from ``Pulse.to_numpy()``, so this compares the two silk code
    paths that used to resolve a zero edge differently — against a real
    ngspice run, at t=0 and at the midpoint of both edges.
    """
    pulse = _pulse()
    src = VoltageSource('in', positive='out', negative='0', pulse=pulse)

    deck = tmp_path / 'pulse_edges.cir'
    deck.write_text(
        '* twin-vs-ngspice pulse edge check (title line)\n'
        + src.to_spice()
        + 'Rload out 0 1k\n'
        '.tran 10p 20n\n'
        '.end\n'
    )
    raw_path = tmp_path / 'pulse_edges.raw'
    subprocess.run([ngspice_executable(), '-b', '-r', str(raw_path), str(deck)],
                   check=True, capture_output=True, timeout=120)

    raw = read_rawfile(raw_path)
    by_name = {v.name.lower(): v.data for v in raw.vectors}
    time = by_name['time']
    spice = by_name['v(out)']

    # ngspice places a timepoint on every pulse breakpoint, so linear
    # interpolation of a piecewise-linear waveform is exact between them.
    spice_at_probe = np.interp(_PROBE_TIMES, time, spice)
    twin_at_probe = pulse.to_numpy(_PROBE_TIMES)

    assert spice_at_probe == pytest.approx(_EXPECTED, abs=1e-3)
    assert twin_at_probe == pytest.approx(spice_at_probe, abs=1e-3)
