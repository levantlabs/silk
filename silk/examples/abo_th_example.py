"""Bootstrapped NMOS track-and-hold with acquisition and SFDR tests.

During hold, M4 precharges n1 to VDD and M5 pulls n2 to ground, charging
C3 between those nodes. M12/M13 discharge the sampling gate vg. During
track, M9 (gate=vg) connects n2 to vin, lifting n1 toward vin+VDD. M8
connects n1 to vg, driving sampling switch M11 with approximately constant
VGS. M3 holds n1 high during hold and releases it during track.

The acquisition bench measures settling and hold droop with a fixed input.
The sine bench measures SFDR near the end of track and hold using one
sample per clock. The behavioral model is an ideal track-and-hold; SPICE
includes transistor nonlinearity and charge injection.

Boosted-node voltage is not capacitor terminal voltage: n1 and n2 rise
together while C3 nominally holds VDD. Device stress remains: during track,
M8 has its gate at clkb=0 and its source near vin+VDD, so its VGS magnitude
can approach 3.4 V for the default input. Stacking the vg pulldown does not
guarantee safe terminal voltages throughout this circuit.

Run from the repository root with MPLBACKEND=Agg:
    .venv/bin/python -m silk.examples.abo_th_example
Use --no-spice for the ideal behavioral run only. The default run prints
acquisition, droop, and SFDR results and saves spectra, a circuit report,
and a SPICE waveform viewer under example_output_dir().
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless-safe; no blocking GUI window
import matplotlib.pyplot as plt


from silk import PortDir, PortRole, Port, VoltageDomain, generate_report
from silk import load_pdk
from silk.circuit.modules import Module
from silk.simulation import Transient
from silk.sources.classes import VoltageSource
from silk.sources.waveforms import Pulse, Sine
from silk.elements import elements as el
from silk.metrics.spectral import sfdr as metric_sfdr, sfdr_db, fft as metric_fft
from silk.metrics.timing import settling_time
from silk.metrics.waveform import value_at
from silk.examples import example_output_dir


# ── Global constants ───────────────────────────────────────────────────────────
VDD      = 1.8          # supply voltage (V)
AVDD     = VoltageDomain('avdd', nominal=VDD, tolerance=0.1)

# Clock
CLK_PER  = 20e-9        # clock period = 20 ns  → Fs = 50 MHz
CLK_PW   = CLK_PER / 2  # 50 % duty cycle (10 ns HIGH / 10 ns LOW)
# Nonzero edges resolve the transition and leave a guard before each sample.
CLK_TR   = CLK_PER / 200

# Coherent-sampling sine parameters
# f_sig = M_COH / N_FFT × Fs  with M_COH coprime to N_FFT → no spectral leakage
N_FFT    = 128          # FFT length (= number of clock cycles simulated)
M_COH    = 11           # input bin index; 11 is prime, coprime to 128
FS       = 1.0 / CLK_PER
F_SIG    = M_COH / N_FFT * FS          # ≈ 4.297 MHz
V_OFFSET = VDD / 2                     # 0.9 V DC offset
V_AMPL   = 0.7                         # 0.7 V peak → 0.2 V … 1.6 V swing

# Simulation window
T_STEP   = 0.1e-9        # 100 ps time step
T_STOP   = N_FFT * CLK_PER   # 2.56 µs (128 clock cycles)

# SFDR sample-anchor times (within a single period, from period start)
# tstart is the absolute time of the FIRST sample in the FFT sequence.
# Subsequent samples are taken every tclk = CLK_PER.
TSTART_TRACK_END = CLK_PW * 0.95       # ≈ 9.5 ns  (just before CLK falls)
TSTART_HOLD_END  = CLK_PER * 0.975     # ≈ 19.5 ns (just before CLK rises)

# Bootstrap / hold capacitor values
C_BOOT   = 500e-15      # 500 fF bootstrap capacitor
C_HOLD   = 250e-15      # 250 fF hold capacitor (balances track settling vs. hold)

# A smaller switch reduces charge injection but increases track resistance.
W_SW     = 2.0          # width (um) of the boost pass M_9 and sampling switch M_11

# Single-ended acquisition/droop test: a fixed DC input, normally clocked.
ACQ_VIN  = 1.2          # held sample value for the settling/droop test
ACQ_SETTLE_TARGET_NS = 2.0   # acquisition settling target
ACQ_DROOP_TARGET_PCT = 0.01  # hold-droop target


# ── PDK / module helpers ───────────────────────────────────────────────────────
def _pdk():
    return load_pdk('sky130')


def _module(name, pdk, out_dir):
    return Module('PDK_skywater130', out_dir, f'{name}.cir',
                  pdk_details=pdk, module_name=name)


# ══════════════════════════════════════════════════════════════════════════════
#  ABO-Bootstrapped Track-and-Hold
# ══════════════════════════════════════════════════════════════════════════════
def make_abo_th(pdk, out_dir, w_sw=W_SW, c_hold=C_HOLD, with_tests=True):
    """Build the bootstrapped sampler and optionally register its acquisition test.

    clk high selects track; clkb high selects hold. M9 is driven by the
    boosted vg node. M11 has its body tied to vss. The behavioral model
    tracks vin while clk exceeds 0.9 V and holds the last value otherwise.
    """
    th = _module('abo_th', pdk, out_dir)

    # ── PMOS latch ────────────────────────────────────────────────────────────
    # The latch holds n1 high in hold; its body follows the boosted drain.
    m_3 = pdk.pmos1p8('m_3', d='n1', g='vg',   s='vdd', b='n1',
                       w=4.0, l=0.18)
    # M11 uses a fixed body connection to vss.

    # ── Precharge / discharge of C_3 ──────────────────────────────────────────
    m_4 = pdk.pmos1p8('m_4', d='n1', g='clk',  s='vdd', b='n1',
                       w=4.0, l=0.18)
    m_5 = pdk.nmos1p8('m_5', d='n2', g='clkb', s='vss', b='vss',
                       w=2.0, l=0.18)
    c_3 = el.capacitor('c_3', p='n1', n='n2', value=C_BOOT)

    # ── Track network ─────────────────────────────────────────────────────────
    # The boosted gate lets M9 track high inputs without a clock-gate clamp.
    m_9 = pdk.nmos1p8('m_9', d='n2', g='vg',   s='vin', b='vss',
                       w=w_sw, l=0.18)
    # Tie M8 body to its boosted source to avoid a source-body diode clamp.
    m_8 = pdk.pmos1p8('m_8', d='vg', g='clkb', s='n1',  b='n1',
                       w=4.0, l=0.18)
    # M_11 — sampling switch driven by bootstrapped vg.
    m_11 = pdk.nmos1p8('m_11', d='vh', g='vg', s='vin', b='vss',
                        w=w_sw, l=0.18)

    # ── Stacked vg pulldown ───────────────────────────────────────────────────
    # Two series devices discharge the boosted gate during hold.
    m_12 = pdk.nmos1p8('m_12', d='vgm', g='clkb', s='vss', b='vss',
                        w=2.0, l=0.18)
    m_13 = pdk.nmos1p8('m_13', d='vg',  g='clkb', s='vgm', b='vss',
                        w=2.0, l=0.18)

    # ── Hold capacitor ────────────────────────────────────────────────────────
    c_s = el.capacitor('c_s', p='vh', n='vss', value=c_hold)

    th.add(m_3, m_4, m_5, c_3, m_9, m_8, m_11, m_12, m_13, c_s)

    th.define_module_ports({
        'vin':  Port(dir=PortDir.IN,  role=PortRole.ANALOG,  domain=AVDD),
        'clk':  (PortDir.IN,  PortRole.CLOCK),
        'clkb': (PortDir.IN,  PortRole.CLOCK),
        'vdd':  Port(dir=PortDir.IN,  role=PortRole.POWER,   domain=AVDD),
        'vss':  Port(dir=PortDir.IN,  role=PortRole.GROUND,  domain=AVDD),
        'vh':   Port(dir=PortDir.OUT, role=PortRole.ANALOG,  domain=AVDD),
    })

    def _abo_th_model(ports, t, params):
        vin  = np.asarray(ports.vin.v)
        clk  = np.asarray(ports.clk.v)
        vh   = np.empty_like(t)
        held = 0.0
        for i in range(len(t)):
            if clk[i] > 0.9:
                vh[i] = held = vin[i]
            else:
                vh[i] = held
        ports.vh.v = vh

    th.set_behavioral_model(_abo_th_model, execution='vector')
    if with_tests:
        _attach_acq_test(th)
    return th


def _hold_droop_pct(ckt):
    """Droop of ``vh`` across the first hold phase, in % of the held level.

    Computed at the call site from ``value_at``: the fraction is meaningful
    here because the held level is ``ACQ_VIN`` (1.2 V), well away from 0 V.
    """
    sig = ckt.load_signal('vh')
    v_start = value_at(sig, CLK_PW + 1e-9)
    v_end = value_at(sig, CLK_PER * 0.975)
    return abs((v_start - v_end) / v_start) * 100.0


def _acq_settle_ns(ckt):
    """Measure 1%-of-step acquisition settling time in ns.

    Use the trace's initial held voltage and ACQ_VIN to define the step.
    """
    sig = ckt.load_signal('vh')
    initial = float(sig.y[0])
    return settling_time(sig, initial=initial, target=ACQ_VIN, tol=0.01,
                         start=0.0) * 1e9


def _attach_acq_test(th):
    """Measure settling and hold droop with a fixed input and running clock.

    The separate sine test measures single-ended SFDR.
    """
    tb = th.create_test('acq')
    tb.add_module_under_test({
        'vin': 'vin', 'clk': 'clk', 'clkb': 'clkb',
        'vdd': 'vdd', 'vss': '0', 'vh': 'vh',
    })
    tb.add_source(VoltageSource('vdd_src', positive='vdd', negative='0', dc=VDD))
    tb.add_source(VoltageSource('vin_src', positive='vin', negative='0', dc=ACQ_VIN))
    tb.add_source(VoltageSource(
        'clk_src', positive='clk', negative='0',
        pulse=Pulse(v1=0, v2=VDD, td=0, tr=CLK_TR, tf=CLK_TR, pw=CLK_PW, per=CLK_PER)))
    tb.add_source(VoltageSource(
        'clkb_src', positive='clkb', negative='0',
        pulse=Pulse(v1=VDD, v2=0, td=0, tr=CLK_TR, tf=CLK_TR, pw=CLK_PW, per=CLK_PER)))
    tb.add_sim_type(Transient(step=T_STEP, stop=4 * CLK_PER))
    tb.add_metric('acq_settle_ns', _acq_settle_ns)
    tb.add_metric('hold_droop_pct', _hold_droop_pct)
    tb.add_spec('acquisition settling < 2 ns', 'acq_settle_ns', '<', ACQ_SETTLE_TARGET_NS)
    tb.add_spec('hold droop < 0.01%', 'hold_droop_pct', '<', ACQ_DROOP_TARGET_PCT)
    return tb


# ══════════════════════════════════════════════════════════════════════════════
#  Top-level test bench
# ══════════════════════════════════════════════════════════════════════════════
def make_th_testbench(netlist_name='abo_th_tb.cir', *, return_test=False):
    """Build the SFDR bench; optionally return its registered test object."""
    out_dir = example_output_dir()
    pdk     = _pdk()
    th = make_abo_th(pdk, out_dir, with_tests=False)
    sfdr_test = th.create_test('sfdr')
    sfdr_test.add_module_under_test({
        'vin': 'vin', 'clk': 'clk', 'clkb': 'clkb',
        'vdd': 'vdd', 'vss': 'gnd', 'vh': 'vh',
    })
    tb = sfdr_test.netlist
    tb.netlistname = netlist_name
    sfdr_test.add_sim_type(Transient(step=T_STEP, stop=T_STOP))

    # ── Stimulus sources ───────────────────────────────────────────────────────
    tb.add_source(VoltageSource(
        'vin_src', positive='vin', negative='gnd',
        sine=Sine(voffset=V_OFFSET, vampl=V_AMPL, freq=F_SIG)))
    tb.add_source(VoltageSource(
        'vdd_src', positive='vdd', negative='gnd', dc=VDD))
    tb.add_source(VoltageSource(
        'clk_src', positive='clk', negative='gnd',
        pulse=Pulse(v1=0, v2=VDD, td=0, tr=CLK_TR, tf=CLK_TR,
                    pw=CLK_PW, per=CLK_PER)))
    tb.add_source(VoltageSource(
        'clkb_src', positive='clkb', negative='gnd',
        pulse=Pulse(v1=VDD, v2=0, td=0, tr=CLK_TR, tf=CLK_TR,
                    pw=CLK_PW, per=CLK_PER)))

    tb.assign_net_domain('vdd', AVDD)
    tb.assign_net_domain('gnd', AVDD)

    # ── Specs ──────────────────────────────────────────────────────────────────
    # Evaluate track-end and hold-end SFDR against the single-ended targets.
    vh = sfdr_test.probe('vh')
    for name, tstart, description in (
        ('sfdr_track_end', TSTART_TRACK_END,
         'SFDR measured just before the end of the track phase'),
        ('sfdr_hold_end', TSTART_HOLD_END,
         'SFDR measured just before the end of the hold phase'),
    ):
        sfdr_test.add_metric(name, sfdr_db, sig=vh, tclk=CLK_PER,
                             tstart=tstart, n_samples=N_FFT)
        spec = sfdr_test.add_spec(name, name, '>=', 50.0)
        spec.units = 'dB'
        spec.description = description

    return (sfdr_test if return_test else tb), out_dir


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    _parser = argparse.ArgumentParser(
        description='ABO-bootstrap T/H — behavioral + sky130 SPICE + SFDR.')
    _parser.add_argument(
        '--no-spice', action='store_true',
        help='Skip the ngspice run (behavioral only).')
    _args = _parser.parse_args()

    print(f'Building ABO-bootstrap T/H testbench...')
    print(f'  Sine:  {F_SIG/1e6:.4f} MHz,  ±{V_AMPL:.2f} V  '
          f'(@  {V_OFFSET:.2f} V DC offset)')
    print(f'  Clock: {1/CLK_PER/1e6:.0f} MHz  ({CLK_PW*1e9:.0f} ns HIGH / '
          f'{CLK_PW*1e9:.0f} ns LOW)')
    print(f'  FFT:   {N_FFT} points  (fundamental bin {M_COH})')
    print(f'  Sim:   t_stop = {T_STOP*1e9:.0f} ns,  '
          f't_step = {T_STEP*1e12:.0f} ps\n')

    # ── Single-ended block: embedded acquisition/droop characterization ────────
    # Run acquisition and droop separately from the sine-wave SFDR test.
    if not _args.no_spice:
        print('── Single-ended T/H block — embedded acquisition/droop test ────────')
        _th = make_abo_th(_pdk(), example_output_dir())
        _th.run_all_tests(corner='tt')
        print(_th.reconcile_specs().summary())
        print()

    sfdr_test, out_dir = make_th_testbench(return_test=True)
    tb = sfdr_test.netlist

    # ── Behavioral simulation ──────────────────────────────────────────────────
    print('Running behavioral simulation...')
    tb.simulate_behavioral(t_step=T_STEP, t_stop=T_STOP)

    vin_sig = tb.load_signal('vin')
    t = vin_sig.x
    t_ns = t * 1e9
    vh_sig  = tb.load_signal('vh')
    vin     = vin_sig.y
    vh      = vh_sig.y

    # ── SFDR — just before end of TRACK phase ─────────────────────────────────
    sfdr_track = metric_sfdr(
        vh_sig,
        tclk=CLK_PER,
        tstart=TSTART_TRACK_END,
        n_samples=N_FFT,
    )
    fft_track = metric_fft(
        vh_sig,
        tclk=CLK_PER,
        tstart=TSTART_TRACK_END,
        n_samples=N_FFT,
    )

    # ── SFDR — just before end of HOLD phase ─────────────────────────────────
    sfdr_hold = metric_sfdr(
        vh_sig,
        tclk=CLK_PER,
        tstart=TSTART_HOLD_END,
        n_samples=N_FFT,
    )
    fft_hold = metric_fft(
        vh_sig,
        tclk=CLK_PER,
        tstart=TSTART_HOLD_END,
        n_samples=N_FFT,
    )

    sfdr_test.evaluate()

    reconcile_report = sfdr_test.module.reconcile_specs()

    print('\n── SFDR Results (behavioral) ──────────────────────────────────')
    print(f'  Track-end sample  (tstart = {TSTART_TRACK_END*1e9:.1f} ns/cycle):')
    print(f'    SFDR           = {sfdr_track.sfdr_db:.2f} dB')
    print(f'    Fundamental    = {sfdr_track.fundamental_hz/1e6:.4f} MHz')
    print(f'    Largest spur   = {sfdr_track.spur_hz/1e6:.4f} MHz')
    print()
    print(f'  Hold-end sample   (tstart = {TSTART_HOLD_END*1e9:.1f} ns/cycle):')
    print(f'    SFDR           = {sfdr_hold.sfdr_db:.2f} dB')
    print(f'    Fundamental    = {sfdr_hold.fundamental_hz/1e6:.4f} MHz')
    print(f'    Largest spur   = {sfdr_hold.spur_hz/1e6:.4f} MHz')

    print('\n── Spec Reconciliation ───────────────────────────────────────')
    print(reconcile_report.summary())

    # ── Circuit report ────────────────────────────────────────────────────────
    report = generate_report(tb)
    report.spec_reconciliation = reconcile_report
    report_path = os.path.join(out_dir, 'abo_th_report.md')
    report.save(report_path)
    print(f'\nReport (MD) saved → {report_path}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    fig.suptitle(
        'ABO-Bootstrapped Track-and-Hold — Behavioral Simulation\n'
        f'sky130 topology  ·  f_in = {F_SIG/1e6:.3f} MHz  ·  '
        f'Fs = {FS/1e6:.0f} MHz  ·  {N_FFT}-point FFT',
        fontsize=12, fontweight='bold',
    )

    # Panel 1: time-domain Vin and Vh
    ax = axes[0]
    ax.plot(t_ns, vin, lw=1.5, color='steelblue',  label='$V_{in}$ (sine)')
    ax.plot(t_ns, vh,  lw=1.0, color='darkorange', label='$V_h$  (T/H output)', alpha=0.85)
    # Mark the two sample grids for the first 4 cycles
    for k in range(4):
        t_trk = (k * CLK_PER + TSTART_TRACK_END) * 1e9
        t_hld = (k * CLK_PER + TSTART_HOLD_END) * 1e9
        ax.axvline(t_trk, color='seagreen',  lw=0.8, ls=':', alpha=0.7,
                   label='track-end sample' if k == 0 else None)
        ax.axvline(t_hld, color='crimson',   lw=0.8, ls=':', alpha=0.7,
                   label='hold-end sample'  if k == 0 else None)
    ax.set_ylabel('Voltage (V)')
    ax.set_ylim(-0.15, 2.0)
    ax.set_xlim(0, min(200, t_ns[-1]))
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('1. Time domain — first 200 ns  (vertical markers = sample instants)',
                 fontsize=9)
    ax.grid(True, alpha=0.25)

    # Panel 2: FFT spectrum — track-end samples (dBc)
    freqs_mhz = fft_track.x / 1e6
    P_track    = fft_track.y
    P_track_db = 10.0 * np.log10(np.maximum(P_track / np.max(P_track[1:]), 1e-14))
    ax = axes[1]
    ax.stem(freqs_mhz, P_track_db, linefmt='seagreen', markerfmt=' ', basefmt=' ')
    ax.axvline(sfdr_track.spur_hz / 1e6, color='seagreen', lw=0.8, ls='--', alpha=0.7)
    ax.set_ylabel('Power (dBc)')
    ax.set_ylim(-120, 5)
    ax.set_xlim(0, FS / 2 / 1e6)
    ax.set_title(
        f'2. Spectrum — track-end samples  '
        f'(SFDR = {sfdr_track.sfdr_db:.1f} dB'
        + (f',  spur @ {sfdr_track.spur_hz/1e6:.3f} MHz'
           if sfdr_track.sfdr_db < 200 else ',  no measurable spur')
        + ')',
        fontsize=9,
    )
    ax.grid(True, alpha=0.25)

    # Panel 3: FFT spectrum — hold-end samples (dBc)
    freqs_mhz_h = fft_hold.x / 1e6
    P_hold      = fft_hold.y
    P_hold_db   = 10.0 * np.log10(np.maximum(P_hold / np.max(P_hold[1:]), 1e-14))
    ax = axes[2]
    ax.stem(freqs_mhz_h, P_hold_db, linefmt='crimson', markerfmt=' ', basefmt=' ')
    ax.axvline(sfdr_hold.spur_hz / 1e6, color='crimson', lw=0.8, ls='--', alpha=0.7)
    ax.set_ylabel('Power (dBc)')
    ax.set_ylim(-120, 5)
    ax.set_xlim(0, FS / 2 / 1e6)
    ax.set_title(
        f'3. Spectrum — hold-end samples  '
        f'(SFDR = {sfdr_hold.sfdr_db:.1f} dB'
        + (f',  spur @ {sfdr_hold.spur_hz/1e6:.3f} MHz'
           if sfdr_hold.sfdr_db < 200 else ',  no measurable spur')
        + ')',
        fontsize=9,
    )
    ax.set_xlabel('Frequency (MHz)')
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    png_path = os.path.join(out_dir, 'abo_th_sfdr.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved → {png_path}')

    report.add_figure(fig, title='ABO T/H — Behavioral: time domain + SFDR spectra')
    html_path = os.path.join(out_dir, 'abo_th_report.html')
    report.save_html(html_path)
    print(f'Report (HTML) saved → {html_path}')


    # ── SPICE ────────────────────────────────────────────────────────────────
    if _args.no_spice:
        print('\n── SPICE skipped (--no-spice) ─────────────────────────────────')
        netlist_path = os.path.join(tb.dir_folder, tb.netlistname)
        print(f'Netlist: {netlist_path}')
        sys.exit(0)

    print('\n── Running SPICE simulation (ngspice, sky130 tt) ───────────────────')
    print(f'  t_stop = {T_STOP*1e9:.0f} ns,  t_step = {T_STEP*1e12:.0f} ps')
    tb.simulator = 'ngspice'
    tb.simulate(corner='tt')
    print('  SPICE run complete.')

    vh_sp    = tb.load_signal('vh')

    sfdr_track_sp = metric_sfdr(
        vh_sp, tclk=CLK_PER, tstart=TSTART_TRACK_END, n_samples=N_FFT)
    sfdr_hold_sp  = metric_sfdr(
        vh_sp, tclk=CLK_PER, tstart=TSTART_HOLD_END,  n_samples=N_FFT)
    fft_track_sp  = metric_fft(
        vh_sp, tclk=CLK_PER, tstart=TSTART_TRACK_END, n_samples=N_FFT)
    fft_hold_sp   = metric_fft(
        vh_sp, tclk=CLK_PER, tstart=TSTART_HOLD_END,  n_samples=N_FFT)

    sfdr_test.evaluate()

    reconcile_report = sfdr_test.module.reconcile_specs()

    print('\n── SFDR Results (SPICE) ───────────────────────────────────────')
    print(f'  Track-end:  SFDR = {sfdr_track_sp.sfdr_db:.2f} dB'
          f'   (spur @ {sfdr_track_sp.spur_hz/1e6:.3f} MHz)')
    print(f'  Hold-end:   SFDR = {sfdr_hold_sp.sfdr_db:.2f} dB'
          f'   (spur @ {sfdr_hold_sp.spur_hz/1e6:.3f} MHz)')
    print()
    print('  Difference (track − hold):')
    print(f'    ΔSFDR = {sfdr_track_sp.sfdr_db - sfdr_hold_sp.sfdr_db:+.2f} dB')
    print('  (positive = higher track-end SFDR; negative = higher hold-end SFDR)')
    print('\n── Spec Reconciliation (SPICE) ───────────────────────────────')
    print(reconcile_report.summary())

    # Comparison plot: behavioral vs SPICE, both sample points
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    fig2.suptitle(
        'ABO T/H — SFDR Spectra: Behavioral vs SPICE\n'
        f'f_in = {F_SIG/1e6:.3f} MHz,  Fs = {FS/1e6:.0f} MHz,  '
        f'{N_FFT}-pt FFT,  sky130 tt',
        fontsize=12, fontweight='bold',
    )

    def _plot_spec(ax, P, spur_hz, sfdr_db, title, color):
        freqs = np.fft.rfftfreq(N_FFT, d=CLK_PER)
        db = 10.0 * np.log10(np.maximum(P / np.max(P[1:]), 1e-14))
        ax.stem(freqs / 1e6, db, linefmt=color, markerfmt=' ', basefmt=' ')
        if sfdr_db < 200:
            ax.axvline(spur_hz / 1e6, color=color, lw=0.8, ls='--', alpha=0.6)
        ax.set_title(f'{title}\nSFDR = {sfdr_db:.1f} dB', fontsize=9)
        ax.set_ylim(-100, 5)
        ax.set_xlim(0, FS / 2 / 1e6)
        ax.grid(True, alpha=0.25)

    _plot_spec(axes2[0, 0], P_track, sfdr_track.spur_hz,
               sfdr_track.sfdr_db, 'Track-end — Behavioral', 'seagreen')
    _plot_spec(axes2[0, 1], P_hold,  sfdr_hold.spur_hz,
               sfdr_hold.sfdr_db,  'Hold-end — Behavioral',  'crimson')
    _plot_spec(axes2[1, 0], fft_track_sp.y, sfdr_track_sp.spur_hz, sfdr_track_sp.sfdr_db,
               'Track-end — SPICE (sky130 tt)', 'steelblue')
    _plot_spec(axes2[1, 1], fft_hold_sp.y,  sfdr_hold_sp.spur_hz,  sfdr_hold_sp.sfdr_db,
               'Hold-end — SPICE (sky130 tt)',  'darkorange')

    for ax in axes2[1]:
        ax.set_xlabel('Frequency (MHz)')
    for ax in axes2[:, 0]:
        ax.set_ylabel('Power (dBc)')

    plt.tight_layout()
    png2 = os.path.join(out_dir, 'abo_th_sfdr_spice.png')
    plt.savefig(png2, dpi=150, bbox_inches='tight')
    print(f'\nComparison plot saved → {png2}')
    report.add_figure(fig2, title='ABO T/H — SFDR: Behavioral vs SPICE')

    # Add an interactive SPICE waveform viewer and link it from the report.
    report.add_waveform_view(
        tb,
        out_path=os.path.join(out_dir, 'abo_th_view.html'),
        title='ABO T/H — acquisition & droop (sky130 tt)')
    view_path = report.waveform_views[-1][0]
    report.spec_reconciliation = reconcile_report
    report.save_html(html_path)

    print(f'\nNetlist:  {os.path.join(tb.dir_folder, tb.netlistname)}')
    print(f'Rawfile:  {tb.results_file}')
    print(f'Waveform: {view_path}')
