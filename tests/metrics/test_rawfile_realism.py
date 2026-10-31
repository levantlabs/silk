"""TASK-74 — metric realism: drive the metrics library from REAL rawfiles.

Every other test under ``tests/metrics`` hand-builds its ``Signal``. That
is how F12 (``gain_bw_product`` reading the −3 dB point off ``Re(H)``,
37 % low) stayed green through five test touchpoints: the tests built
a REAL magnitude array, while ``from_rawfile`` hands every ngspice and
Xyce AC sweep a ``complex128`` array untouched. Coverage was complete;
the INPUT was not what a simulator produces.

This module closes the general form of that gap. For each analysis
family — transient, AC, DC (sweep + code transitions), noise — it loads
a committed fixture produced by a real simulator run
(``tests/fixtures/rawfiles/gen_e.py``; ngspice AND Xyce) through the
REAL load path (``read_rawfile`` → ``NgspiceSignal.from_rawfile``, or
``SimulatorBackend.load_signal`` / ``load_noise``) and runs that
family's metrics on it. No simulator is needed at test time.

What the assertions check, and why:

* **Contract realism** — dtype (AC ``y`` is complex128, transient ``y``
  float64), axis label as the simulator wrote it, shape (row count ==
  ``No. Points``, the transient grid is the simulator's own NON-uniform
  adaptive one), monotone x.
* **Closed-form oracle** — every circuit in ``gen_e.py`` has an analytic
  answer derived from its component values (``gen_e.CIRCUIT``, imported
  by path so deck and oracle cannot drift). A metric is compared against
  that, within the metric's DOCUMENTED resolution (grid snap, sample
  rate, simulator ``reltol``) — a silent wrong number fails, not just a
  crash.
* **Complex-vs-magnitude cross-check (the F12 check)** — every AC metric
  whose contract is "reads |y|" must give the SAME answer on the
  complex Signal the simulator produced and on its magnitude, with
  ``ComplexWarning`` promoted to an error. A completeness guard
  enumerates the public surface of ``silk.metrics.ac`` so a new AC
  metric cannot skip this.
* **Cross-backend agreement** — ngspice and Xyce ran the SAME deck, so
  the scalar metrics must agree between them.

FOUND here and FIXED (TASK-87, 2026-09-05): Xyce's ``.print ...
format=raw`` upper-cases the axis name (``TIME`` / ``FREQUENCY``) and
``from_rawfile`` used to pass it through verbatim as ``xlabel``, while every
axis check in the library compares with exact lowercase equality — so NO
transient or AC metric accepted a Xyce Signal loaded through the real path.
``from_rawfile`` now case-folds the label it assigns (``RawVector.name``
stays verbatim); ``TestXyceAxisNameContract`` pins both halves of that.
Every fixture below is the untouched real-path Signal on both backends.

FOUND here and FIXED (TASK-99, 2026-09-05): ``group_delay`` cast the complex
``y`` to float and differentiated Re(H) instead of the phase — the F12 defect
surviving in the one function TASK-63's guard excluded. The old test in this
module hand-unwrapped the phase into a REAL Signal, so it never handed the
function what ``from_rawfile`` produces; it now runs on the untouched complex
fixture on both backends against the closed-form 1- and 3-pole delay.
"""
from __future__ import annotations

import importlib.util
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from silk.signals.base import Signal
from silk.utils._compat import trapezoid
from silk.signals.rawfile import read_rawfile
from silk.signals.spice import NgspiceSignal
from silk.simulation.backend import SimulatorBackend, RawResult
import silk.metrics as M
import silk.metrics.ac as AC

from tests._env import needs_ngspice


# ---------------------------------------------------------------------------
# Fixture access
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).resolve().parent.parent / 'fixtures' / 'rawfiles'


def _load_gen_e():
    spec = importlib.util.spec_from_file_location('gen_e', FIXTURES / 'gen_e.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN_E = _load_gen_e()
CIRCUIT = GEN_E.CIRCUIT

PREFIX = {'ngspice': 'e_ng', 'xyce': 'e_xy'}
BACKENDS = ('ngspice', 'xyce')

def rawfile(backend: str, family: str):
    return read_rawfile(FIXTURES / f'{PREFIX[backend]}_{family}.raw')


def load(backend: str, family: str, name: str, type: str = 'V') -> Signal:
    """The REAL load path: reader → NgspiceSignal.from_rawfile, verbatim."""
    return NgspiceSignal.from_rawfile(rawfile(backend, family), name, type)


def load_noise(backend: str, port: str) -> Signal:
    """The REAL noise path: SimulatorBackend.load_signal(raw, 'noise_out')."""
    be = SimulatorBackend.get(backend)
    if backend == 'ngspice':
        raw = be.load_raw(FIXTURES / 'e_ng_noise.raw')
    else:
        # Xyce noise is CSV-only; XyceBackend.load_noise resolves
        # <dir of rawfile>/noise.csv from the RawResult's source_path.
        raw = RawResult(native=None,
                        source_path=str(FIXTURES / 'e_xy_noise' / 'rawfile.raw'))
    return be.load_signal(raw, f'noise_{port}')


def _no_complex_warning(fn):
    with warnings.catch_warnings():
        warnings.simplefilter('error', np.exceptions.ComplexWarning)
        return fn()


def _magnitude(sig: Signal) -> Signal:
    return Signal(sig.name, sig.type, sig.xlabel, sig.ylabel,
                  np.asarray(sig.x, dtype=float), np.abs(sig.y))


def _bisect(fn, lo, hi, iters=200):
    """Root of fn on [lo, hi] (sign change assumed); log-spaced search."""
    flo = fn(lo)
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        fm = fn(mid)
        if (fm < 0) == (flo < 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return math.sqrt(lo * hi)


# ===========================================================================
# AC family — 3-pole behavioural amplifier
# ===========================================================================

_A = CIRCUIT['ac']
A0 = _A['A0']
FP = tuple(_A['fp'])
GRID_RATIO = 10.0 ** (1.0 / _A['pts_per_dec'])        # 1.0233 at 100/dec
N_AC = int(round(math.log10(_A['fstop'] / _A['fstart']) * _A['pts_per_dec'])) + 1


def H3(f):
    """Closed-form 3-pole transfer function of v(out)."""
    f = np.asarray(f, dtype=float)
    h = A0 * np.ones_like(f, dtype=complex)
    for fp in FP:
        h = h / (1 + 1j * f / fp)
    return h


def H1(f):
    """Closed-form single-pole transfer of v(p1) (after stage 1)."""
    return A0 / (1 + 1j * np.asarray(f, dtype=float) / FP[0])


def _f_3db_true():
    return _bisect(lambda f: abs(H3(f)) - A0 / math.sqrt(2), 1e3, 1e5)


def _f_ugf_true():
    return _bisect(lambda f: abs(H3(f)) - 1.0, 1e5, 1e8)


def _f_180_true():
    # unwrapped phase = -sum(atan(f/fp)); crosses -180 deg once
    return _bisect(lambda f: -sum(math.atan(f / fp) for fp in FP) + math.pi,
                   1e5, 1e8)


@pytest.fixture(params=BACKENDS)
def ac(request):
    b = request.param
    return {
        'backend': b,
        'in': load(b, 'ac', 'in'),
        'p1': load(b, 'ac', 'p1'),
        'out': load(b, 'ac', 'out'),
    }


class TestAcContract:
    """What a real .ac rawfile looks like once loaded."""

    def test_axis_label_is_canonical_on_both_simulators(self):
        # Xyce writes the vector as 'FREQUENCY'; the label is folded (TASK-87).
        assert load('ngspice', 'ac', 'out').xlabel == 'frequency'
        assert load('xyce', 'ac', 'out').xlabel == 'frequency'

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_dtype_shape_and_grid(self, backend):
        out = load(backend, 'ac', 'out')
        raw = rawfile(backend, 'ac')
        assert out.y.dtype == np.complex128, 'AC y is complex — F12 territory'
        assert out.x.dtype == np.float64
        assert len(out.x) == len(out.y) == raw.npoints == N_AC
        assert np.all(np.diff(out.x) > 0)
        assert out.x[0] == pytest.approx(_A['fstart'], rel=1e-9)
        assert out.x[-1] == pytest.approx(_A['fstop'], rel=1e-6)
        ratios = out.x[1:] / out.x[:-1]
        assert np.allclose(ratios, GRID_RATIO, rtol=1e-6)

    def test_vector_names_upper_cased_by_xyce(self):
        names = [v.name for v in rawfile('xyce', 'ac').vectors]
        assert names == ['FREQUENCY', 'V(IN)', 'V(P1)', 'V(OUT)']
        names = [v.name for v in rawfile('ngspice', 'ac').vectors]
        assert names[0] == 'frequency' and 'v(out)' in names

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_simulator_matches_closed_form_transfer(self, backend):
        out = load(backend, 'ac', 'out')
        p1 = load(backend, 'ac', 'p1')
        vin = load(backend, 'ac', 'in')
        assert np.allclose(vin.y, 1.0 + 0j, atol=1e-12)
        assert np.max(np.abs(out.y - H3(out.x)) / np.abs(H3(out.x))) < 1e-6
        assert np.max(np.abs(p1.y - H1(p1.x)) / np.abs(H1(p1.x))) < 1e-6


# Coverage registry for the AC completeness guard --------------------------
COVERED_AC: set = set()


def covers(*names):
    def deco(fn):
        COVERED_AC.update(names)
        return fn
    return deco


class TestAcMetricsAgainstClosedForm:

    @covers('dc_gain_dB', 'peak_gain_dB')
    def test_dc_and_peak_gain(self, ac):
        out = ac['out']
        expect = 20 * math.log10(abs(H3(out.x[0])))
        assert M.dc_gain_dB(out) == pytest.approx(expect, abs=1e-6)
        assert M.dc_gain_dB(out) == pytest.approx(20 * math.log10(A0), abs=1e-4)
        # monotonic roll-off: the peak IS the DC gain
        assert M.peak_gain_dB(out) == M.dc_gain_dB(out)

    @covers('bandwidth_3dB')
    def test_bandwidth_3db_is_last_grid_point_within_3db(self, ac):
        out = ac['out']
        f3 = _f_3db_true()
        got = M.bandwidth_3dB(out)
        # documented resolution: the last GRID point still within 3 dB —
        # at most one grid step below the true -3 dB frequency
        assert f3 / GRID_RATIO < got <= f3 * (1 + 1e-9)
        # and exactly the grid point the closed form predicts
        mag = np.abs(H3(out.x))
        expect = out.x[np.where(mag >= mag.max() / math.sqrt(2))[0][-1]]
        assert got == pytest.approx(expect, rel=1e-12)

    @covers('gain_bw_product')
    def test_gbw_single_pole_node_is_a0_times_fp(self, ac):
        p1 = ac['p1']
        got = M.gain_bw_product(p1)
        assert A0 * FP[0] / GRID_RATIO < got <= A0 * FP[0] * (1 + 1e-6)

    def test_gbw_three_pole_consistent_with_bandwidth_and_dc_gain(self, ac):
        out = ac['out']
        gbw = M.gain_bw_product(out)
        dc_lin = 10 ** (M.dc_gain_dB(out) / 20)
        assert gbw == pytest.approx(dc_lin * M.bandwidth_3dB(out), rel=1e-9)
        f3 = _f_3db_true()
        assert A0 * f3 / GRID_RATIO < gbw <= A0 * f3 * (1 + 1e-6)

    @covers('unity_gain_freq')
    def test_unity_gain_freq(self, ac):
        got = M.unity_gain_freq(ac['out'])
        # interpolated between grid points 2.3 % apart — well inside 1 %
        assert got == pytest.approx(_f_ugf_true(), rel=1e-2)

    @covers('phase_margin')
    def test_phase_margin(self, ac):
        f_u = _f_ugf_true()
        expect = 180.0 + math.degrees(np.angle(H3(f_u)))
        got = M.phase_margin(ac['out'])
        # phase read at the grid sample nearest the UGF — a degree or so
        assert got == pytest.approx(expect, abs=1.5)
        assert 0 < got < 90

    @covers('gain_margin', 'gain_margins')
    def test_gain_margin_single_phase_crossover(self, ac):
        f180 = _f_180_true()
        expect = -20 * math.log10(abs(H3(f180)))
        crossings = M.gain_margins(ac['out'])
        assert len(crossings) == 1
        fc, gm = crossings[0]
        assert fc == pytest.approx(f180, rel=1e-2)
        assert gm == pytest.approx(expect, abs=0.1)
        assert M.gain_margin(ac['out']) == pytest.approx(expect, abs=0.1)
        assert gm > 0   # stable loop

    @covers('gain_at_freq', 'phase_at_freq')
    def test_gain_and_phase_at_freq_read_nearest_grid_sample(self, ac):
        out = ac['out']
        for f in (1e3, 1e6, 2.5e7):
            i = int(np.argmin(np.abs(out.x - f)))
            fg = out.x[i]
            assert M.gain_at_freq(out, f, interpolate=False) == pytest.approx(
                abs(H3(fg)), rel=1e-6)
            assert M.phase_at_freq(out, f, interpolate=False) == pytest.approx(
                math.degrees(np.angle(H3(fg))), abs=1e-4)
            # TASK-413: the default interpolates, landing on the closed form
            # at f itself rather than at the neighbouring grid point.
            assert M.gain_at_freq(out, f) == pytest.approx(abs(H3(f)), rel=2e-3)
            assert M.phase_at_freq(out, f) == pytest.approx(
                math.degrees(np.angle(H3(f))), abs=0.5)

    @covers('passband_ripple_db')
    def test_passband_ripple(self, ac):
        out = ac['out']
        f_pass = 1e3
        band = out.x[out.x <= f_pass]
        mag = np.abs(H3(band))
        expect = 20 * math.log10(mag.max() / mag.min())
        assert M.passband_ripple_db(out, f_pass) == pytest.approx(expect, abs=1e-6)
        assert 0 < expect < 0.1   # genuinely flat passband

    @covers('stopband_attenuation_db')
    def test_stopband_attenuation(self, ac):
        out = ac['out']
        f_stop = 1e6
        pass_peak = np.abs(H3(out.x[out.x < f_stop])).max()
        stop_peak = np.abs(H3(out.x[out.x >= f_stop])).max()
        expect = 20 * math.log10(pass_peak / stop_peak)
        assert M.stopband_attenuation_db(out, f_stop) == pytest.approx(expect, abs=1e-6)
        # ~40 dB: 60 dB DC gain minus |H(1 MHz)| ≈ 19.5 dB (grid placement
        # of the first stop point moves it by up to one 2.3 % step)
        assert 40.0 < expect < 41.0

    @covers('q_from_peak')
    def test_q_from_peak_on_monotonic_response_is_unavailable(self, ac):
        # The lower half-power boundary is outside the supplied sweep.
        q = M.q_from_peak(ac['out'])
        assert isinstance(q, float) and np.isnan(q)

    @covers('group_delay')
    def test_group_delay_on_the_complex_signal_at_every_sweep_point(self, ac):
        """F14 / FIXED (TASK-99). The old test here hand-unwrapped the phase
        into a REAL Signal — the same blind spot as F12 — so the function
        never saw what ``from_rawfile`` produces. On the untouched complex
        fixture the pre-fix ``dtype=float`` cast differentiated Re(H):
        ``v(p1)`` read 32.2 µs at 10 Hz (true 15.9 µs) and 7.96 ms at the
        10 kHz pole (true 7.96 µs — 1000x, the 60 dB gain leaking into a
        time); ``v(out)`` 32.3 µs / 7.99 ms against 16.0 µs / 8.02 µs.
        Now every one of the 701 grid points is within the docstring's
        finite-difference bound on both nodes."""
        r = GRID_RATIO
        for node, poles in (('p1', FP[:1]), ('out', FP)):
            sig = ac[node]
            gd = _no_complex_warning(lambda: M.group_delay(sig))
            assert isinstance(gd, Signal) and gd.xlabel == 'frequency'
            np.testing.assert_array_equal(gd.x, sig.x)
            tau = sum((1 / (2 * math.pi * fp)) / (1 + (sig.x / fp) ** 2) for fp in poles)
            rel = np.abs(gd.y / tau - 1)
            assert rel[1:-1].max() <= (r - 1) ** 2 / r * 1.1, node   # 0.05 % at 100/dec
            assert max(rel[0], rel[-1]) <= r - 1, node                # 2.3 % at 100/dec
            assert rel[0] < 1e-6, node     # flat region: docstring says exact to round-off

    def test_group_delay_backends_agree(self):
        for node in ('p1', 'out'):
            a = M.group_delay(load('ngspice', 'ac', node))
            b = M.group_delay(load('xyce', 'ac', node))
            np.testing.assert_allclose(a.y, b.y, rtol=1e-7)   # measured 2.7e-9

    @covers('psrr_db')
    def test_psrr_is_minus_gain_in_db_on_the_grid(self, ac):
        r = M.psrr_db(ac['in'], ac['out'])
        assert isinstance(r, Signal)
        assert r.xlabel == 'frequency'
        np.testing.assert_allclose(r.y, -20 * np.log10(np.abs(H3(r.x))), atol=1e-6)

    @covers('cmrr_dB')
    def test_cmrr_out_vs_in(self, ac):
        assert M.cmrr_dB(ac['out'], ac['in']) == pytest.approx(
            20 * math.log10(A0), abs=1e-4)
        assert M.cmrr_dB(ac['out'], ac['in'], freq=1e6) == pytest.approx(
            20 * math.log10(abs(H3(ac['out'].x[np.argmin(np.abs(ac['out'].x - 1e6))]))),
            abs=1e-6)

    @covers('integrated_noise')
    def test_integrated_noise_on_ac_magnitude_is_band_rms(self, ac):
        # integrated_noise's contract: sqrt(∫|y|² df) over the band. On the
        # AC sweep that is simply the band RMS of |H| — trapezoid on the grid.
        #
        # CHANGED by TASK-197: the COMPLEX fixture is now refused (the
        # kind check below), because integrated_noise declares a
        # noise_density input and complex y rules that kind out. The
        # numeric oracle is unchanged and still runs — on the MAGNITUDE
        # Signal, which is real y on the frequency axis and therefore
        # UNDECIDABLE, i.e. accepted.
        out = _magnitude(ac['out'])
        f1, f2 = 1e3, 1e5
        mask = (out.x >= f1) & (out.x <= f2)
        expect = math.sqrt(trapezoid(np.abs(H3(out.x[mask])) ** 2, out.x[mask]))
        assert M.integrated_noise(out, f1, f2) == pytest.approx(expect, rel=1e-6)

    def test_integrated_noise_refuses_the_complex_ac_signal(self, ac):
        # TASK-197 (design §3.4): this call used to return 1.25e+04 and
        # call it volts. The fixture is the untouched real-path AC
        # Signal, so this pins the refusal against what a simulator
        # actually produces, not against a hand-built array.
        assert ac['out'].y.dtype == np.complex128
        with pytest.raises(ValueError, match='must be a noise_density Signal'):
            M.integrated_noise(ac['out'], 1e3, 1e5)


# --- The F12 check: complex vs magnitude ---------------------------------

MAGNITUDE_METRICS = [
    # (name, callable taking one Signal) — contract: reads |y| only
    ('dc_gain_dB', lambda s: M.dc_gain_dB(s)),
    ('peak_gain_dB', lambda s: M.peak_gain_dB(s)),
    ('gain_at_freq', lambda s: M.gain_at_freq(s, 1e6)),
    ('bandwidth_3dB', lambda s: M.bandwidth_3dB(s)),
    ('unity_gain_freq', lambda s: M.unity_gain_freq(s)),
    ('gain_bw_product', lambda s: M.gain_bw_product(s)),
    ('q_from_peak', lambda s: M.q_from_peak(s)),
    ('passband_ripple_db', lambda s: M.passband_ripple_db(s, 1e3)),
    ('stopband_attenuation_db', lambda s: M.stopband_attenuation_db(s, 1e6)),
    # integrated_noise was here until TASK-197. It is not an AC metric —
    # it declares a noise_density input — and it now REFUSES a complex y
    # outright, which closes the F12 class for it more firmly than an
    # agreement check could: see
    # TestAcMetricsAgainstClosedForm.test_integrated_noise_refuses_the_complex_ac_signal
    # for the refusal and ...on_ac_magnitude_is_band_rms for the oracle.
]


class TestComplexVsMagnitudeCrossCheck:
    """A magnitude-contract metric must not care whether it is handed the
    complex Signal the simulator produced or its magnitude — and must not
    warn while doing it. This is the assertion that would have caught F12
    on day one."""

    @pytest.mark.parametrize('name,fn', MAGNITUDE_METRICS, ids=[m[0] for m in MAGNITUDE_METRICS])
    @pytest.mark.parametrize('node', ['p1', 'out'])
    def test_same_answer_on_complex_and_on_abs(self, ac, node, name, fn):
        sig = ac[node]
        assert sig.y.dtype == np.complex128
        on_complex = _no_complex_warning(lambda: fn(sig))
        on_mag = _no_complex_warning(lambda: fn(_magnitude(sig)))
        if name == 'q_from_peak':
            # Neither monotonic trace supplies a bounded half-power band.
            assert np.isnan(on_complex) and np.isnan(on_mag)
            return
        assert on_complex == pytest.approx(on_mag, rel=1e-12, abs=0.0), (
            f'{name}: {on_complex!r} on complex y vs {on_mag!r} on |y| — '
            f'a magnitude metric read something other than |y| (F12 class)')

    def test_two_input_magnitude_metrics(self, ac):
        vin, out = ac['in'], ac['out']
        c = _no_complex_warning(lambda: M.cmrr_dB(out, vin))
        m = _no_complex_warning(lambda: M.cmrr_dB(_magnitude(out), _magnitude(vin)))
        assert c == pytest.approx(m, rel=1e-12)
        pc = _no_complex_warning(lambda: M.psrr_db(vin, out))
        pm = _no_complex_warning(lambda: M.psrr_db(_magnitude(vin), _magnitude(out)))
        np.testing.assert_allclose(pc.y, pm.y, rtol=1e-12)

    def test_phase_metrics_need_the_complex_signal(self, ac):
        # Sanity for the classification above: these read np.angle, so
        # the magnitude Signal (phase identically 0) gives a DIFFERENT
        # answer. If one of these ever agreed on |y| it would mean it
        # stopped reading phase.
        out = ac['out']
        assert M.phase_at_freq(out, 1e6) != pytest.approx(
            M.phase_at_freq(_magnitude(out), 1e6), abs=1.0)
        assert M.phase_margin(out) != pytest.approx(
            M.phase_margin(_magnitude(out)), abs=1.0)
        # group_delay refuses the magnitude outright (TASK-99): a real y
        # carries no phase to differentiate
        with pytest.raises(ValueError, match='real-valued y'):
            M.group_delay(_magnitude(out))


# --- Completeness guard ----------------------------------------------------

EXCLUDED_AC: dict = {}     # name -> reason; empty since TASK-99 made group_delay a phase metric


def _public_names(module):
    return {n for n, o in vars(module).items()
            if callable(o) and not n.startswith('_')
            and getattr(o, '__module__', '') == module.__name__}


def test_every_public_ac_metric_is_driven_from_a_real_rawfile():
    names = _public_names(AC)
    missing = sorted(names - COVERED_AC)
    assert not missing, (
        f'AC metrics never run on a real simulator rawfile: {missing} — '
        f'add a @covers(...) test in TestAcMetricsAgainstClosedForm')


def test_every_magnitude_metric_is_in_the_f12_cross_check():
    # Every covered metric that is not phase-based and not excluded must be
    # in MAGNITUDE_METRICS (single-input) or the two-input test.
    phase_based = {'phase_at_freq', 'phase_margin', 'gain_margin', 'gain_margins',
                   'group_delay'}
    two_input = {'cmrr_dB', 'psrr_db'}
    checked = {m[0] for m in MAGNITUDE_METRICS} | two_input | phase_based | set(EXCLUDED_AC)
    names = _public_names(AC)
    missing = sorted(names - checked)
    assert not missing, f'AC metrics without a complex-vs-|y| classification: {missing}'


# ===========================================================================
# Transient family — series RLC step, clock, coherent tone, source current
# ===========================================================================

_T = CIRCUIT['tran']
WN = 1.0 / math.sqrt(_T['L'] * _T['C'])
ZETA = _T['R'] / 2.0 * math.sqrt(_T['C'] / _T['L'])
WD = WN * math.sqrt(1 - ZETA ** 2)
T0 = _T['t_step'] + _T['edge'] / 2          # effective step instant
TCLK = 100e-9                               # 10 MHz sampling of the tone
N_FFT = 256
TONE_BIN = 7


def step_response(t):
    """Unit step response of the series RLC, closed form, t absolute."""
    tt = np.asarray(t, dtype=float) - T0
    y = 1 - np.exp(-ZETA * WN * tt) * (
        np.cos(WD * tt) + ZETA / math.sqrt(1 - ZETA ** 2) * np.sin(WD * tt))
    return np.where(tt > 0, y, 0.0) * _T['V_step']


_FINE_T = np.arange(0.0, _T['tstop'], 1e-10)
_FINE_Y = step_response(_FINE_T)


def _fine_first_crossing(level, rising=True):
    y = _FINE_Y
    if rising:
        idx = np.where((y[:-1] < level) & (y[1:] >= level))[0]
    else:
        idx = np.where((y[:-1] > level) & (y[1:] <= level))[0]
    i = idx[0]
    t0, t1, y0, y1 = _FINE_T[i], _FINE_T[i + 1], y[i], y[i + 1]
    return float(t0 + (level - y0) / (y1 - y0) * (t1 - t0))


OVERSHOOT_TRUE = math.exp(-math.pi * ZETA / math.sqrt(1 - ZETA ** 2))
PEAK_TRUE = _T['V_step'] * (1 + OVERSHOOT_TRUE)
T_PEAK_TRUE = T0 + math.pi / WD
# post-transition undershoot: the next reversal one half-cycle later decays
# by another factor of OVERSHOOT_TRUE (same derivation as rlc_step.py).
UNDERSHOOT_TRUE = OVERSHOOT_TRUE ** 2


@pytest.fixture(params=BACKENDS)
def tran(request):
    b = request.param
    return {
        'backend': b,
        'step': load(b, 'tran', 'step'),
        'out': load(b, 'tran', 'out'),
        'clk': load(b, 'tran', 'clk'),
        'tone': load(b, 'tran', 'tone'),
        'i1': load(b, 'tran', 'v1', 'I'),
    }


class TestTranContract:

    def test_axis_label_is_canonical_on_both_simulators(self):
        # Xyce writes the vector as 'TIME'; the label is folded (TASK-87).
        assert load('ngspice', 'tran', 'out').xlabel == 'time'
        assert load('xyce', 'tran', 'out').xlabel == 'time'

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_dtype_shape_and_adaptive_grid(self, backend):
        out = load(backend, 'tran', 'out')
        raw = rawfile(backend, 'tran')
        assert out.y.dtype == np.float64 and out.x.dtype == np.float64
        assert len(out.x) == len(out.y) == raw.npoints
        assert raw.npoints > 2000
        dt = np.diff(out.x)
        assert np.all(dt > 0), 'time must be strictly increasing'
        assert dt.max() <= _T['tstep'] * (1 + 1e-9), 'tmax honoured'
        assert dt.min() < dt.max() / 10, (
            'the grid must be the simulator\'s own adaptive one, not uniform')
        assert out.x[0] == 0.0
        assert out.x[-1] == pytest.approx(_T['tstop'], rel=1e-9)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_current_vector_loads_as_type_I(self, backend):
        i1 = load(backend, 'tran', 'v1', 'I')
        assert i1.type == 'current'
        assert i1.y.dtype == np.float64

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_simulator_matches_closed_form_step_response(self, backend):
        out = load(backend, 'tran', 'out')
        mask = out.x > T0 + 50e-9      # skip the finite-edge region
        err = np.abs(out.y[mask] - step_response(out.x[mask]))
        assert err.max() < 5e-3


class TestTranEdgeMetrics:

    def test_overshoot(self, tran):
        assert M.overshoot(tran['out'], initial=0.0, target=_T['V_step']) == pytest.approx(
            OVERSHOOT_TRUE, rel=1e-3)
        assert M.undershoot(tran['out'], initial=0.0, target=_T['V_step']) == pytest.approx(
            UNDERSHOOT_TRUE, rel=1e-3)

    def test_peak_and_time_to_peak(self, tran):
        out = tran['out']
        assert M.peak(out) == pytest.approx(PEAK_TRUE, rel=1e-3)
        assert M.peak_to_peak(out) == pytest.approx(PEAK_TRUE, rel=1e-3)
        assert M.argmax_x(out) == pytest.approx(T_PEAK_TRUE, abs=_T['tstep'])
        assert M.maximum(out, start=20e-6) < 1.02
        assert M.minimum(out, start=20e-6) > 0.98

    def test_rise_time_against_fine_grid_closed_form(self, tran):
        out = tran['out']
        swing = PEAK_TRUE          # observed min is 0
        t10 = _fine_first_crossing(0.1 * swing)
        t90 = _fine_first_crossing(0.9 * swing)
        assert M.rise_time(out) == pytest.approx(t90 - t10, abs=_T['tstep'])
        # slew_rate uses the OBSERVED swing (its documented convention)
        assert M.slew_rate(out) == pytest.approx(
            0.8 * M.peak_to_peak(out) / M.rise_time(out), rel=1e-9)

    def test_settling_time_1pct(self, tran):
        out = tran['out']
        tol = 0.01
        band = abs(_T['V_step']) * tol
        last_out = _FINE_T[np.where(np.abs(_FINE_Y - _T['V_step']) > band)[0][-1]]
        got = M.settling_time(out, initial=0.0, target=_T['V_step'], tol=tol, start=_T['t_step'])
        # grid-snapped: first sample at/after the true instant
        assert last_out - _T['t_step'] <= got <= last_out - _T['t_step'] + _T['tstep'] * 2
        # physically: a few time constants of zeta*wn
        assert 3 / (ZETA * WN) < got < 6 / (ZETA * WN)

    def test_propagation_delay_and_crossing(self, tran):
        step, out = tran['step'], tran['out']
        # step's mid-swing crossing is at T0 (10 ns edge, mid = 0.5)
        t_mid_out = _fine_first_crossing(0.5 * PEAK_TRUE)
        assert M.propagation_delay(step, out) == pytest.approx(
            t_mid_out - T0, abs=_T['tstep'])
        assert M.crossing(out, 0.5, edge='any') == pytest.approx(
            _fine_first_crossing(0.5), abs=_T['tstep'])
        cr = M.crossings(out, threshold=_T['V_step'], edge='rising')
        assert len(cr) >= 3    # ringing crosses the final value repeatedly
        assert cr[0] == pytest.approx(_fine_first_crossing(_T['V_step']),
                                       abs=_T['tstep'])

    def test_value_at_droop_and_sh_error(self, tran):
        out = tran['out']
        for t in (5e-6, 20e-6, 40e-6):
            assert M.value_at(out, t) == pytest.approx(float(step_response(t)), abs=2e-3)
        y30, y45 = float(step_response(30e-6)), float(step_response(45e-6))
        # droop over a hold interval, composed by the caller from value_at
        # (TASK-412 removed the dedicated metric)
        v30, v45 = M.value_at(out, 30e-6), M.value_at(out, 45e-6)
        assert (v30 - v45) / v30 == pytest.approx((y30 - y45) / y30, abs=1e-3)
        # sample-and-hold settling error is |value_at(t) - target| / |target|,
        # computed by the caller (TASK-287 removed the dedicated metric)
        sh_error = abs(M.value_at(out, 40e-6) - _T['V_step']) / abs(_T['V_step'])
        assert sh_error == pytest.approx(
            abs(float(step_response(40e-6)) - 1.0), abs=1e-4)
        assert sh_error < 1e-3

    def test_integral_average_and_derivative(self, tran):
        out, step = tran['out'], tran['step']
        expect = float(trapezoid(_FINE_Y, _FINE_T))
        assert M.integral(out) == pytest.approx(expect, rel=2e-3)
        assert M.average(step, start=2e-6) == pytest.approx(_T['V_step'], abs=1e-9)
        assert M.average(out, start=25e-6) == pytest.approx(_T['V_step'], abs=2e-3)
        d = M.derivative(out)
        assert d.xlabel == out.xlabel and len(d.y) == len(out.y)
        # peak slope of the closed-form response (away from the 10 ns edge)
        dy = float(np.gradient(_FINE_Y, _FINE_T)[_FINE_T > T0 + 50e-9].max())
        assert d.y.max() == pytest.approx(dy, rel=5e-2)
        ri = M.running_integral(out)
        assert ri.y[-1] == pytest.approx(M.integral(out), rel=1e-9)

    def test_energy_from_source_is_c_v_squared(self, tran):
        # ∫ v·i dt over the run: the source delivers exactly C·V² (½CV²
        # stored, ½CV² dissipated in R) — SPICE current convention makes
        # i(v1) negative when the source delivers, hence the sign.
        p = M.average_power(tran['step'], tran['i1'])
        expect = _T['C'] * _T['V_step'] ** 2 / _T['tstop']
        assert p < 0
        assert -p == pytest.approx(expect, rel=1e-2)


class TestTranClockMetrics:

    def test_period_duty_and_pulse_width(self, tran):
        clk = tran['clk']
        assert M.average_period(clk) == pytest.approx(_T['clk_period'], rel=1e-6)
        assert M.average_duty(clk) == pytest.approx(_T['clk_duty'], abs=1e-3)
        assert M.pulse_width(clk) == pytest.approx(
            _T['clk_period'] * _T['clk_duty'], rel=1e-3)
        cr = M.crossings(clk)
        n_expect = int((_T['tstop'] - _T['clk_delay']) // _T['clk_period']) + 1
        assert len(cr) == n_expect
        assert cr[0] == pytest.approx(_T['clk_delay'] + _T['clk_edge'] / 2, abs=1e-9)

    def test_ideal_clock_has_no_jitter(self, tran):
        clk = tran['clk']
        assert abs(M.period_jitter(clk)) < 1e-12
        assert abs(M.cycle_to_cycle_jitter(clk)) < 1e-12
        assert abs(M.long_term_jitter(clk)) < 1e-12
        t = M.tie(clk)
        assert t.xlabel == 'time' and np.all(np.abs(t.y) < 1e-12)
        f = M.cycle_frequency(clk)
        assert np.allclose(f.y, 1 / _T['clk_period'], rtol=1e-6)
        ad = M.allan_deviation(clk)
        assert np.all(np.isfinite(ad.y)) and np.all(ad.y < 1e-9)

    def test_on_off_time(self, tran):
        clk = tran['clk']
        level = _T['V_clk'] / 2
        # rising edges at delay + k*period; high for duty*period each,
        # the last one truncated by tstop
        n_full = int((_T['tstop'] - _T['clk_delay']) // _T['clk_period'])
        t_last = _T['clk_delay'] + n_full * _T['clk_period']
        on_expect = (n_full * _T['clk_period'] * _T['clk_duty']
                     + min(_T['clk_period'] * _T['clk_duty'], _T['tstop'] - t_last))
        assert M.on_time(clk, level) == pytest.approx(on_expect, abs=2 * _T['clk_edge'])
        assert M.off_time(clk, level) == pytest.approx(
            _T['tstop'] - on_expect, abs=2 * _T['clk_edge'])

    # --- BEGIN commented-out (TASK-226.11): eye_diagram realism test,
    # superseded by TASK-226.12+. Kept for reference until the replacement
    # is accepted; do not delete. ---
    # def test_eye_diagram_of_a_clean_clock_is_wide_open(self, tran):
    #     # one UI of a clock is HALF its period (1010... pattern); folding at
    #     # the full period puts every trace at the same level and, by the
    #     # documented algorithm, yields a single population (height 0).
    #     ui = _T['clk_period'] / 2
    #     eye = M.eye_diagram(tran['clk'], period=ui)
    #     assert eye.height == pytest.approx(_T['V_clk'], rel=1e-2)
    #     assert eye.width == pytest.approx(ui, rel=1e-2)
    #     assert eye.jitter_rms < 1e-12
    #     assert eye.traces.shape[0] == int((_T['tstop']) // ui)
    # --- END commented-out (TASK-226.11) ---


class TestTranSpectralMetrics:
    """The tone net, sampled coherently (bin 7 of 256 at 10 MHz)."""

    kw = dict(tclk=TCLK, tstart=0.0, n_samples=N_FFT)

    def test_rms_of_the_tone(self, tran):
        assert M.rms(tran['tone']) == pytest.approx(
            _T['A_tone'] / math.sqrt(2), rel=1e-2)
        assert M.stddev(tran['tone']) == pytest.approx(
            _T['A_tone'] / math.sqrt(2), rel=1e-2)

    def test_fundamental_lands_in_the_designed_bin(self, tran):
        spec = M.fft(tran['tone'], **self.kw)
        assert spec.xlabel == 'frequency' and len(spec.x) == N_FFT // 2 + 1
        assert int(np.argmax(spec.y[1:]) + 1) == TONE_BIN
        assert spec.x[TONE_BIN] == pytest.approx(_T['f_tone'], rel=1e-12)
        hf = M.harmonic_freqs(tran['tone'], **self.kw)
        np.testing.assert_allclose(hf[:3], _T['f_tone'] * np.array([1, 2, 3]), rtol=1e-12)
        s = M.sfdr(tran['tone'], **self.kw)
        assert s.fundamental_hz == pytest.approx(_T['f_tone'], rel=1e-12)

    def test_purity_metrics_are_consistent_and_physically_bounded(self, tran):
        tone = tran['tone']
        sndr = M.sndr(tone, **self.kw)
        snr = M.snr(tone, **self.kw)
        thd = M.thd(tone, **self.kw)
        enob = M.enob(tone, **self.kw)
        sfdr = M.sfdr(tone, **self.kw).sfdr_db
        nf = M.noise_floor_dbfs(tone, **self.kw)
        # a pure source resampled off an adaptive grid: clean but finite
        assert 40 < sndr < 120
        assert snr >= sndr - 1e-9
        assert thd < -40
        assert sfdr > sndr - 1e-9 or sfdr > 40
        assert nf < -40
        assert enob == pytest.approx((sndr - 1.76) / 6.02, abs=1e-9)
        assert M.sinad(tone, **self.kw) == pytest.approx(sndr, abs=1e-9)
        assert M.thd_percent(tone, **self.kw) == pytest.approx(
            100 * 10 ** (thd / 20), rel=1e-9)
        # 1/SNDR = 1/SNR + THD in power — exact when both use the SAME
        # harmonic set (defaults differ: snr excludes 10, thd sums 5)
        nh = 10
        lhs = 10 ** (-M.sndr(tone, **self.kw) / 10)
        rhs = (10 ** (-M.snr(tone, n_harmonics=nh, **self.kw) / 10)
               + 10 ** (M.thd(tone, n_harmonics=nh, **self.kw) / 10))
        assert lhs == pytest.approx(rhs, rel=1e-9)

    def test_harmonic_levels_and_band_power(self, tran):
        tone = tran['tone']
        lv = M.harmonic_levels_db(tone, **self.kw)
        assert len(lv) >= 2 and np.all(np.asarray(lv) < -40)
        # harmonic() returns the COMPLEX bin: |X[k]| of an unwindowed rfft
        # over N samples of amplitude A is A·N/2 for the fundamental
        h1 = M.harmonic(tone, 1, **self.kw)
        h2 = M.harmonic(tone, 2, **self.kw)
        assert isinstance(h1, complex)
        assert abs(h1) == pytest.approx(_T['A_tone'] * N_FFT / 2, rel=2e-2)
        assert 20 * math.log10(abs(h2) / abs(h1)) < -40
        # spectral power in a band around the fundamental ≈ A²/2 (Hann)
        bw = 2 / (N_FFT * TCLK)
        p = M.spectral_power(tone, _T['f_tone'] - bw, _T['f_tone'] + bw, **self.kw)
        assert p == pytest.approx(_T['A_tone'] ** 2 / 2, rel=0.2)
        psd = M.psd(tone, **self.kw)
        assert psd.xlabel == 'frequency' and np.all(psd.y >= 0)
        # welch: 8 half-overlapping segments of 56 samples → 178 kHz bins;
        # the tone must peak within one Welch bin of its frequency
        wp = M.welch_psd(tone, **self.kw)
        assert wp.xlabel == 'frequency'
        f_peak = wp.x[int(np.argmax(wp.y[1:]) + 1)]
        assert abs(f_peak - _T['f_tone']) <= wp.x[1]


# ===========================================================================
# DC family — tanh transfer sweep + quantizer ramp → code transitions
# ===========================================================================

_D = CIRCUIT['dc']
N_DC = int(round((_D['vstop'] - _D['vstart']) / _D['vstep'])) + 1
# ngspice's B-source converges to its default reltol (~1.5e-4 observed);
# Xyce is exact to 1e-16. One tolerance for both.
DC_TOL = 1e-3


def tanh_transfer(v):
    return np.tanh(_D['k'] * np.asarray(v, dtype=float))


@pytest.fixture(params=BACKENDS)
def dc(request):
    b = request.param
    # Nothing in silk.metrics has a DC-sweep axis contract, so the axis
    # label is simply whatever each simulator named the sweep vector.
    return {'backend': b, 'in': load(b, 'dc', 'in'), 'out': load(b, 'dc', 'out')}


class TestDcSweepContract:

    def test_axis_label_is_the_sweep_vector_each_simulator_names(self):
        assert load('ngspice', 'dc', 'out').xlabel == 'v(v-sweep)'
        assert load('xyce', 'dc', 'out').xlabel == 'sweep'

    def test_dtype_shape_and_grid(self, dc):
        out = dc['out']
        assert out.y.dtype == np.float64
        assert len(out.x) == N_DC
        np.testing.assert_allclose(
            out.x, np.linspace(_D['vstart'], _D['vstop'], N_DC), atol=1e-9)
        np.testing.assert_allclose(dc['in'].y, out.x, atol=1e-12)
        assert np.max(np.abs(out.y - tanh_transfer(out.x))) < DC_TOL


class TestDcSweepPrimitives:
    """The axis-agnostic primitives are the metrics that legitimately
    consume a .dc sweep."""

    def test_point_queries(self, dc):
        out = dc['out']
        for v in (-1.0, 0.0, 0.5, 1.5):
            assert M.value_at(out, v) == pytest.approx(float(tanh_transfer(v)), abs=DC_TOL)
        assert M.crossing(out, 0.0, edge='any') == pytest.approx(0.0, abs=DC_TOL)
        assert M.crossing(out, 0.5, edge='any') == pytest.approx(math.atanh(0.5) / _D['k'], abs=DC_TOL)

    def test_extrema(self, dc):
        out = dc['out']
        assert M.peak(out) == pytest.approx(float(tanh_transfer(_D['vstop'])), abs=DC_TOL)
        assert M.peak_to_peak(out) == pytest.approx(
            2 * float(tanh_transfer(_D['vstop'])), abs=2 * DC_TOL)
        assert M.argmax_x(out) == pytest.approx(_D['vstop'], abs=1e-9)
        assert M.argmin_x(out) == pytest.approx(_D['vstart'], abs=1e-9)
        # tstop nudged past the grid point (float placement of 1.0)
        assert M.maximum(out, start=0.0, end=1.005) == pytest.approx(
            float(tanh_transfer(1.0)), abs=DC_TOL)

    def test_small_signal_gain_from_derivative(self, dc):
        d = M.derivative(dc['out'])
        assert d.y.max() == pytest.approx(_D['k'], rel=1e-2)
        i0 = int(np.argmin(np.abs(d.x)))
        assert d.y[i0] == pytest.approx(_D['k'], rel=1e-2)

    def test_integrals_of_an_odd_function(self, dc):
        out = dc['out']
        assert abs(M.integral(out)) < 1e-4
        assert abs(M.average(out)) < 1e-4
        span = _D['vstop'] - _D['vstart']
        # ∫tanh²(kv)dv = v − tanh(kv)/k
        ms = (span - 2 * math.tanh(_D['k'] * _D['vstop']) / _D['k']) / span
        assert M.rms(out) == pytest.approx(math.sqrt(ms), rel=2e-3)

    def test_p1db_from_the_real_compression_curve(self, dc):
        out = dc['out']
        pos = out.x > 0
        p_in = 20 * np.log10(out.x[pos])
        p_out = 20 * np.log10(np.abs(out.y[pos]))
        # p1db takes a power-sweep Signal (TASK-180): x = p_in, y = p_out,
        # both in dBm.
        got = M.p1db(Signal('psweep', 'power', 'power', 'dBm', p_in, p_out))
        # gain drops 1 dB where tanh(kv)/(kv) = 10^(-1/20)
        u = _bisect(lambda u: math.tanh(u) / u - 10 ** (-1 / 20), 1e-3, 5.0)
        assert got == pytest.approx(20 * math.log10(u / _D['k']), abs=0.1)


# --- code transitions --------------------------------------------------------

_R = CIRCUIT['ramp']
N_CODES = 2 ** _R['bits']


def transition_level_true(k):
    """V at which 2^bits·(V + nl·V²) == k."""
    nl = _R['nl']
    return (-1 + math.sqrt(1 + 4 * nl * k / N_CODES)) / (2 * nl)


TRANSITIONS_TRUE = np.array([transition_level_true(k) for k in range(1, N_CODES + 1)])
LSB_TRUE = (TRANSITIONS_TRUE[-1] - TRANSITIONS_TRUE[0]) / (N_CODES - 1)
INL_TRUE = (TRANSITIONS_TRUE - (TRANSITIONS_TRUE[0] + np.arange(N_CODES) * LSB_TRUE)) / LSB_TRUE
DNL_TRUE = np.diff(TRANSITIONS_TRUE) / LSB_TRUE - 1
# ramp slope × sample step + quantizer softness (width_lsb × LSB)
RAMP_TOL = (_R['v_end'] / _R['t_end']) * _R['tstep'] + _R['width_lsb'] / N_CODES * 4


@pytest.fixture(params=BACKENDS)
def ramp(request):
    b = request.param
    return {
        'backend': b,
        'in': load(b, 'ramp', 'in'),
        'code': load(b, 'ramp', 'code'),
    }


class TestCodeTransitionsFromRamp:

    def test_transition_levels_match_closed_form(self, ramp):
        ct = M.extract_code_transitions(ramp['in'], ramp['code'])
        assert ct.xlabel == 'code'
        np.testing.assert_array_equal(ct.x, np.arange(1, N_CODES + 1))
        assert np.max(np.abs(ct.y - TRANSITIONS_TRUE)) < RAMP_TOL

    def test_inl_dnl_monotonic_missing(self, ramp):
        ct = M.extract_code_transitions(ramp['in'], ramp['code'])
        inl = M.inl(ct)
        dnl = M.dnl(ct)
        assert inl.method == 'endpoint'
        np.testing.assert_allclose(inl.inl_lsb, INL_TRUE, atol=RAMP_TOL / LSB_TRUE)
        np.testing.assert_allclose(dnl.dnl_lsb, DNL_TRUE, atol=2 * RAMP_TOL / LSB_TRUE)
        assert inl.max_abs_lsb == pytest.approx(np.max(np.abs(INL_TRUE)), abs=RAMP_TOL / LSB_TRUE)
        assert 0.02 < inl.max_abs_lsb < 0.2       # the designed 5 % bow, not 0
        assert M.is_monotonic(ct)
        assert M.missing_codes(ct).size == 0
        bf = M.inl(ct, method='best_fit')
        assert bf.max_abs_lsb < inl.max_abs_lsb    # best-fit never worse than endpoint


# ===========================================================================
# Noise family — diode shot + 1/f in parallel with a thermal resistor
# ===========================================================================

_N = CIRCUIT['noise']
K_B = 1.380649e-23
Q_E = 1.602176634e-19
VT = K_B * _N['T_kelvin'] / Q_E
RD = VT / _N['I_bias']                              # diode small-signal r
RP = RD * _N['R1'] / (RD + _N['R1'])                # node impedance
WHITE_I2 = 4 * K_B * _N['T_kelvin'] / _N['R1'] + 2 * Q_E * _N['I_bias']   # A²/Hz
FLICKER_I2 = _N['KF'] * _N['I_bias'] ** _N['AF']                          # A²·Hz/Hz at 1 Hz
F_CORNER_TRUE = FLICKER_I2 / WHITE_I2
GAIN_IN_TO_OUT = _N['G'] * RD / (RD + _N['R1'])
N_NOISE = int(round(math.log10(_N['fstop'] / _N['fstart']) * _N['pts_per_dec'])) + 1


def onoise_true(f):
    f = np.asarray(f, dtype=float)
    return _N['G'] * RP * np.sqrt(WHITE_I2 + FLICKER_I2 / f)


def integrated_true(f1, f2):
    return _N['G'] * RP * math.sqrt(WHITE_I2 * (f2 - f1) + FLICKER_I2 * math.log(f2 / f1))


@pytest.fixture(params=BACKENDS)
def noise(request):
    b = request.param
    return {'backend': b, 'out': load_noise(b, 'out'), 'in': load_noise(b, 'in')}


class TestNoiseContract:

    def test_canonical_signal_from_each_backend(self, noise):
        on = noise['out']
        assert on.name == 'noise_out' and on.xlabel == 'frequency'
        assert on.y.dtype == np.float64, 'noise density is real, never complex'
        assert len(on.x) == N_NOISE
        assert np.all(np.diff(on.x) > 0) and np.all(on.y > 0)
        assert on.x[0] == pytest.approx(_N['fstart']) and on.x[-1] == pytest.approx(_N['fstop'], rel=1e-9)

    def test_ngspice_picks_the_spectrum_out_of_the_breakdown_clutter(self):
        raw = read_rawfile(FIXTURES / 'e_ng_noise.raw')
        names = [v.name for v in raw.vectors]
        assert 'onoise_spectrum' in names and 'inoise_spectrum' in names
        assert any(n.startswith('onoise_d1') for n in names), (
            'ngspice writes per-device breakdown vectors; the fixture keeps them')
        on = load_noise('ngspice', 'out')
        np.testing.assert_array_equal(
            on.y, raw.vectors[names.index('onoise_spectrum')].data)

    def test_xyce_csv_is_v2_per_hz_converted_to_amplitude(self):
        from silk.simulation.xyce_runner import read_xyce_noise_csv
        cols = read_xyce_noise_csv(str(FIXTURES / 'e_xy_noise' / 'noise.csv'))
        on = load_noise('xyce', 'out')
        np.testing.assert_allclose(on.y ** 2, cols['ONOISE'], rtol=1e-12)

    def test_simulator_matches_closed_form_spectrum(self, noise):
        on = noise['out']
        rel = np.abs(on.y - onoise_true(on.x)) / onoise_true(on.x)
        assert rel.max() < 2e-4

    def test_backends_agree(self):
        a, b = load_noise('ngspice', 'out'), load_noise('xyce', 'out')
        np.testing.assert_allclose(a.x, b.x, rtol=1e-8)   # Xyce CSV: 9 digits
        np.testing.assert_allclose(a.y, b.y, rtol=2e-4)


class TestNoiseMetrics:

    def test_integrated_noise(self, noise):
        on = noise['out']
        for f1, f2 in ((1.0, 1e6), (10.0, 1e3), (1e4, 1e6)):
            got = M.integrated_noise(on, f1, f2)
            # trapezoid on a 20/dec log grid over-reads a 1/f PSD by ~0.45 %
            assert got == pytest.approx(integrated_true(f1, f2), rel=1e-2)
        assert M.integrated_noise(on, 1.0, 1e6) > M.integrated_noise(on, 1e4, 1e6)

    def test_one_over_f_corner(self, noise):
        got = M.one_over_f_corner(noise['out'])
        # documented: first grid point below √2·floor (≤ one 12 % grid step
        # up) and the floor estimate still carries some 1/f (reads low)
        assert F_CORNER_TRUE / 1.3 < got < F_CORNER_TRUE * 1.15

    def test_input_referred_matches_the_loaded_inoise(self, noise):
        on, inn = noise['out'], noise['in']
        for i in (0, 40, 60, 80, len(on.x) - 1):
            expect = M.input_referred_noise(float(on.y[i]), GAIN_IN_TO_OUT)
            assert inn.y[i] == pytest.approx(expect, rel=1e-3)
        ratio = inn.y / on.y
        assert np.std(ratio) / np.mean(ratio) < 1e-6   # frequency-independent

    def test_axis_agnostic_primitives_on_a_log_grid(self, noise):
        on = noise['out']
        assert M.argmax_x(on) == pytest.approx(_N['fstart'])
        assert M.peak(on) == pytest.approx(float(onoise_true(_N['fstart'])), rel=2e-4)
        assert M.value_at(on, 1e3) == pytest.approx(float(onoise_true(1e3)), rel=2e-4)


# ===========================================================================
# Cross-backend agreement on scalar metrics (same deck, two simulators)
# ===========================================================================

class TestBackendsAgree:

    def test_ac_scalars(self):
        a = load('xyce', 'ac', 'out')
        b = load('ngspice', 'ac', 'out')
        for fn in (M.dc_gain_dB, M.bandwidth_3dB, M.unity_gain_freq,
                   M.phase_margin, M.gain_margin, M.gain_bw_product):
            assert fn(a) == pytest.approx(fn(b), rel=1e-5), fn.__name__

    def test_tran_scalars(self):
        a = load('xyce', 'tran', 'out')
        b = load('ngspice', 'tran', 'out')
        for fn, kw in ((M.overshoot, dict(initial=0.0, target=1.0)), (M.rise_time, {}),
                       (M.peak, {}), (M.argmax_x, {})):
            assert fn(a, **kw) == pytest.approx(fn(b, **kw), rel=2e-3), fn.__name__
        ca = load('xyce', 'tran', 'clk')
        cb = load('ngspice', 'tran', 'clk')
        assert M.average_period(ca) == pytest.approx(M.average_period(cb), rel=1e-9)

    def test_dc_and_ramp_scalars(self):
        a, b = load('xyce', 'dc', 'out'), load('ngspice', 'dc', 'out')
        assert M.value_at(a, 0.5) == pytest.approx(M.value_at(b, 0.5), abs=DC_TOL)
        ra = M.extract_code_transitions(load('xyce', 'ramp', 'in'),
                                        load('xyce', 'ramp', 'code'))
        rb = M.extract_code_transitions(load('ngspice', 'ramp', 'in'),
                                        load('ngspice', 'ramp', 'code'))
        assert M.inl(ra).max_abs_lsb == pytest.approx(M.inl(rb).max_abs_lsb, abs=RAMP_TOL / LSB_TRUE)


# ===========================================================================
# TASK-87 — Xyce axis label accepted by every axis check
# ===========================================================================

class TestXyceAxisNameContract:
    """Pre-fix, each of the four ``*_accepts_xyce_signal`` tests raised
    ``ValueError("sig must be a transient Signal (xlabel='time'), got
    xlabel='TIME'")`` (or the 'frequency'/'FREQUENCY' twin) straight out of
    the axis check: ``from_rawfile`` assigned vector 0's verbatim name as
    ``xlabel``. They were strict xfails until TASK-87 folded the label at
    the load boundary; now they are the plain real-path assertions."""

    def test_transient_metric_accepts_xyce_signal(self):
        out = load('xyce', 'tran', 'out')
        assert M.overshoot(out, initial=0.0, target=1.0) == pytest.approx(OVERSHOOT_TRUE, rel=1e-3)

    def test_ac_metric_accepts_xyce_signal(self):
        out = load('xyce', 'ac', 'out')
        assert M.dc_gain_dB(out) == pytest.approx(20 * math.log10(A0), abs=1e-4)

    def test_silk_metrics_ac_accepts_xyce_signal(self):
        assert np.isfinite(M.gain_bw_product(load('xyce', 'ac', 'p1')))

    def test_code_transitions_accept_xyce_signal(self):
        ct = M.extract_code_transitions(load('xyce', 'ramp', 'in'),
                                        load('xyce', 'ramp', 'code'))
        assert len(ct.x) == N_CODES

    def test_the_ngspice_leg_of_each_repro_passes(self):
        # Same four calls on the ngspice fixture: the defect was Xyce-only.
        assert M.overshoot(load('ngspice', 'tran', 'out'), initial=0.0, target=1.0) == pytest.approx(OVERSHOOT_TRUE, rel=1e-3)
        assert M.dc_gain_dB(load('ngspice', 'ac', 'out')) == pytest.approx(20 * math.log10(A0), abs=1e-4)
        assert np.isfinite(M.gain_bw_product(load('ngspice', 'ac', 'p1')))
        assert len(M.extract_code_transitions(load('ngspice', 'ramp', 'in'),
                                              load('ngspice', 'ramp', 'code')).x) == N_CODES

    @pytest.mark.parametrize('family, verbatim', [('tran', 'TIME'), ('ac', 'FREQUENCY'),
                                                  ('ramp', 'TIME'), ('dc', 'sweep')])
    def test_rawvector_name_stays_verbatim_only_the_label_is_folded(self, family, verbatim):
        """AC#3: the reader contract is untouched — ``RawVector.name`` is the
        exact string Xyce wrote (the rawfile viewer shows it as such); only the
        ``Signal.xlabel`` assigned by ``from_rawfile`` is case-folded. Xyce's
        DC sweep axis is already lowercase ('sweep'), so it is unchanged."""
        raw = rawfile('xyce', family)
        assert raw.vectors[0].name == verbatim
        name = {'ramp': 'code'}.get(family, 'out')
        assert load('xyce', family, name).xlabel == verbatim.lower()

    @pytest.mark.parametrize('family', ['tran', 'ac', 'ramp', 'dc', 'noise'])
    def test_ngspice_axis_label_is_byte_unchanged(self, family):
        """AC#3: ngspice lower-cases every name it writes, so folding the
        label is the identity on its output — the label IS the verbatim
        vector-0 name for every ngspice fixture ('time', 'frequency',
        'v(v-sweep)'), byte for byte."""
        raw = rawfile('ngspice', family)
        name = {'noise': 'onoise_spectrum', 'ramp': 'code'}.get(family, 'out')
        sig = NgspiceSignal.from_rawfile(raw, name, 'V')
        assert sig.xlabel == raw.vectors[0].name
        assert raw.vectors[0].name == raw.vectors[0].name.lower()


# ===========================================================================
# TASK-303.2 — ngspice `.measure` differential tests (second oracle)
# ===========================================================================
#
# TASK-303 (see docs/metrics-ngspice-measure-catalog.md, TASK-303.1's
# catalog): ngspice's own `.measure` engine is an INDEPENDENT oracle on the
# SAME rawfile the closed-form checks above already use — it can catch a
# defect the closed-form derivation shares a blind spot with, which a
# from-scratch analytic answer cannot.
#
# The reference numbers below come from `gen_e_meas.py`, which runs real
# `.measure` statements against the identical deck bodies
# (`gen_e.tran_core()`/`gen_e.ac_core()`) that already produced
# `e_ng_tran.raw`/`e_ng_ac.raw` — same simulated waveform, independent
# post-processing — and are committed in `e_ng_tran_meas.json` /
# `e_ng_ac_meas.json`. REPORT, DO NOT FIX (owner 2026-09-26): every
# assertion below either confirms agreement within the metric's documented
# resolution, or encodes a convention difference already named in the
# catalog with the numbers ngspice actually produced. No silk/metrics code
# was changed to make any of this pass.
#
# Resolution stated per metric, before claiming agreement (reviewer
# constraint): rise/fall time and delay are read on the SAME finite-edge
# waveform silk parses (resolution = the interpolated-crossing precision
# both engines share, not a sample-rate floor — the edges are linear PWL
# ramps); FIND AT and the windowed AVG/RMS/MIN/MAX/PP/INTEG forms use a
# FROM/TO window read off `e_ng_tran.raw`'s own adaptive grid
# (`gen_e_meas._tran_window_bounds`), landing exactly on samples both
# engines snap to identically — see the catalog's "window snap vs.
# interpolate" note, which is why an OFF-grid window is avoided here; the
# AC crossings inherit the interpolation-domain differences the catalog
# already documents for `bandwidth_3dB`/`unity_gain_freq`/`phase_margin`,
# so their tolerances are wider and explained inline.

MEAS_TRAN = json.loads((FIXTURES / 'e_ng_tran_meas.json').read_text())
MEAS_AC = json.loads((FIXTURES / 'e_ng_ac_meas.json').read_text())


class TestNgspiceMeasDifferentialTran:
    """Second-oracle checks on the tran family, against `e_ng_tran_meas.json`."""

    def test_rise_and_fall_time_of_the_clock(self):
        # TRIG/TARG on v(clk)'s linear PWL edges vs rise_time/fall_time:
        # both read the same two interpolated threshold crossings, so
        # agreement is to interpolation precision, not a grid-step floor.
        clk = load('ngspice', 'tran', 'clk')
        assert M.rise_time(clk) == pytest.approx(
            MEAS_TRAN['values']['rise_time_clk'], rel=1e-6)
        assert M.fall_time(clk) == pytest.approx(
            MEAS_TRAN['values']['fall_time_clk'], rel=1e-6)

    def test_propagation_delay_at_the_ngspice_trig_targ_threshold(self):
        # threshold=0.5 explicit (not propagation_delay's auto mid-swing) to
        # match TRIG/TARG's VAL=0.5 exactly — the same convention TASK-303.1
        # verified live on a standalone deck; this repeats it on OUR fixture.
        step, out = load('ngspice', 'tran', 'step'), load('ngspice', 'tran', 'out')
        got = M.propagation_delay(step, out, in_threshold=0.5, out_threshold=0.5,
                                  in_edge='rising', out_edge='rising')
        assert got == pytest.approx(
            MEAS_TRAN['values']['propagation_delay_step_out'], rel=1e-5)

    def test_when_single_crossing(self):
        out = load('ngspice', 'tran', 'out')
        assert M.crossing(out, 0.5, edge='rising', n=1) == pytest.approx(
            MEAS_TRAN['values']['x_when_out_050'], rel=1e-4)

    def test_find_at_a_fixed_instant(self):
        out = load('ngspice', 'tran', 'out')
        got = M.value_at(out, MEAS_TRAN['value_at_t_s'])
        assert got == pytest.approx(MEAS_TRAN['values']['value_at_out'], rel=1e-5)

    def test_find_when_composition(self):
        # ngspice's bundled "FIND v1 WHEN v2=val" vs silk's two separate
        # calls (catalog: "compose") — same numeric answer once composed.
        step, out = load('ngspice', 'tran', 'step'), load('ngspice', 'tran', 'out')
        t = M.crossing(step, 0.5, edge='rising', n=1)
        got = M.value_at(out, t)
        assert got == pytest.approx(
            MEAS_TRAN['values']['find_when_out_at_step_050'], rel=1e-5)

    def test_windowed_avg_rms_min_max_pp_integral_on_an_on_grid_window(self):
        # FROM/TO land exactly on samples of e_ng_tran.raw's own adaptive
        # grid (gen_e_meas._tran_window_bounds), so ngspice's snap-only
        # AVG/MIN/MAX/PP convention and its interpolating RMS/INTEG
        # convention both reduce to the SAME sample set silk's _window()
        # snaps to — the catalog's ~2.6% off-grid RMS/INTEG gap does not
        # apply here by construction.
        out = load('ngspice', 'tran', 'out')
        tfrom, tto = MEAS_TRAN['window_from_s'], MEAS_TRAN['window_to_s']
        v = MEAS_TRAN['values']
        assert M.average(out, start=tfrom, end=tto) == pytest.approx(
            v['window_avg_out'], rel=1e-5)
        assert M.rms(out, start=tfrom, end=tto) == pytest.approx(
            v['window_rms_out'], rel=1e-4)
        assert M.minimum(out, start=tfrom, end=tto) == pytest.approx(
            v['window_min_out'], rel=1e-5)
        assert M.maximum(out, start=tfrom, end=tto) == pytest.approx(
            v['window_max_out'], rel=1e-5)
        assert M.peak_to_peak(out, start=tfrom, end=tto) == pytest.approx(
            v['window_pp_out'], rel=1e-5)
        # INTEG uses a composite Simpson's/trapezoidal rule (catalog);
        # silk's integral is plain trapezoidal. On this fine adaptive grid
        # with matched endpoints the two rules agree well inside 1e-4.
        assert M.integral(out, start=tfrom, end=tto) == pytest.approx(
            v['window_integ_out'], rel=1e-4)

    def test_deriv_has_no_ngspice_oracle(self):
        # TASK-303.1: ngspice parses DERIV but its .measure dispatch refuses
        # it in every version on this box. Confirmed again here, on THIS
        # exact deck, by gen_e_meas.py. Not a silk gap to report — there is
        # simply nothing on the ngspice side to assert equality against;
        # this only pins that silk's own derivative() still works where
        # ngspice's DERIV cannot even attempt the measurement.
        assert MEAS_TRAN['deriv']['status'] == 'unsupported'
        out = load('ngspice', 'tran', 'out')
        d = M.derivative(out)
        assert np.isfinite(M.value_at(d, MEAS_TRAN['value_at_t_s']))


class TestNgspiceMeasDifferentialAc:
    """Second-oracle checks on the AC family, against `e_ng_ac_meas.json`."""

    def test_bandwidth_3db_is_one_grid_step_below_the_interpolated_ngspice_when(self):
        # silk's bandwidth_3dB is deliberately grid-snapped, no
        # interpolation (ac.py docstring): it reads the last SWEEP GRID
        # POINT still >= -3 dB, up to one grid step (GRID_RATIO) low of the
        # true/interpolated point. ngspice's WHEN vdb(out)=<peak-3dB>
        # interpolates linearly in dB and linear frequency, landing at the
        # true crossing. This is the catalog's documented "silk convention"
        # row, not a bug: the two must differ by AT MOST one grid step.
        out = load('ngspice', 'ac', 'out')
        got = M.bandwidth_3dB(out)
        ng = MEAS_AC['values']['bandwidth_3db_out']
        assert got <= ng * (1 + 1e-9)
        assert got >= ng / GRID_RATIO * (1 - 1e-6)

    def test_unity_gain_freq_close_to_ngspices_linear_frequency_interpolation(self):
        # silk interpolates linear-magnitude/log-frequency; ngspice's WHEN
        # vm(out)=1 interpolates linear-magnitude/linear-frequency
        # (catalog: "silk convention", converges at high sweep density).
        # At 100 pts/decade the two domains agree to ~5e-5 relative
        # (measured) -- well inside a single grid step (2.3%).
        out = load('ngspice', 'ac', 'out')
        got = M.unity_gain_freq(out)
        ng = MEAS_AC['values']['unity_gain_freq_out']
        assert got == pytest.approx(ng, rel=5e-4)

    def test_phase_margin_after_converting_ngspices_radians(self):
        # ngspice's raw .measure result for vp() is in RADIANS (verified
        # while building gen_e_meas.py: the printed value here is -2.769414,
        # which is math.degrees(-2.769414) =~ -158.7 DEGREES, matching the
        # closed-form phase at the unity-gain crossing computed elsewhere
        # in this module -- not -2.77 degrees, which would put the loop
        # implausibly close to 0 dB AND -180 deg phase simultaneously).
        # FIND vp(out) WHEN vdb(out)=0 also finds a slightly different
        # crossing frequency than unity_gain_freq's own search (dB- vs
        # magnitude-interpolation), so the two phase margins differ by a
        # fraction of a degree, not to printed precision.
        assert MEAS_AC['phase_margin_find_out_unit'] == 'rad'
        out = load('ngspice', 'ac', 'out')
        got = M.phase_margin(out)
        pm_rad = MEAS_AC['values']['phase_margin_find_out']
        ng_pm_deg = 180.0 + math.degrees(pm_rad)
        assert got == pytest.approx(ng_pm_deg, abs=1.0)

    def test_gain_margin_has_no_literal_ngspice_when_recipe(self):
        # Extends the catalog's gain_margin row: ngspice's vp() is wrapped
        # to (-pi, pi] PER SAMPLE, not unwrapped across the sweep (verified
        # live while building gen_e_meas.py -- the printed vp(out) vector
        # jumps straight from -3.13108 to +3.13883 rad between two adjacent
        # samples straddling the true -180 deg crossover; it never equals
        # -pi or +pi at any sample). This circuit's phase genuinely passes
        # -180 deg on its way to a -270 deg asymptote, so a literal
        # `WHEN vp(out)=-3.14159...` / `FIND vdb(out) WHEN vp(out)=-3.14159...`
        # recipe fails with "out of interval" on ngspice itself -- there is
        # NO ngspice .measure oracle for this crossing at all (not merely
        # "compose", as the original catalog pass assumed), so nothing was
        # captured to assert equality against. silk's own gain_margin
        # sidesteps this by unwrapping the phase before searching; this
        # only pins that it still returns a real number here.
        assert 'gain_margin_out' not in MEAS_AC['values']
        out = load('ngspice', 'ac', 'out')
        assert np.isfinite(M.gain_margin(out))


@pytest.mark.slow
@needs_ngspice
def test_meas_sidecar_regenerates_from_a_real_ngspice_run():
    """AC#1: the slow profile regenerates the sidecar reference values.
    ngspice's `.measure` is deterministic for a fixed deck, so re-running
    `gen_e_meas.py`'s builders (without writing anything — they only run
    ngspice and parse stdout) must reproduce the committed numbers; a real
    disagreement here means the committed sidecar is stale (a different
    ngspice version, or the deck drifted from `gen_e.tran_core()`/
    `gen_e.ac_core()`), which is exactly the staleness this test exists to
    catch."""
    spec = importlib.util.spec_from_file_location(
        'gen_e_meas', FIXTURES / 'gen_e_meas.py')
    gen_e_meas = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_e_meas)

    fresh_tran = gen_e_meas.build_tran_sidecar()
    fresh_ac = gen_e_meas.build_ac_sidecar()

    for name, expect in MEAS_TRAN['values'].items():
        assert fresh_tran['values'][name] == pytest.approx(expect, rel=1e-6), name
    assert fresh_tran['deriv']['status'] == MEAS_TRAN['deriv']['status']
    for name, expect in MEAS_AC['values'].items():
        assert fresh_ac['values'][name] == pytest.approx(expect, rel=1e-6), name
