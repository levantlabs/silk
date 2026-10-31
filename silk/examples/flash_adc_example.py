"""Flash ADC with a track-and-hold, reference ladder and comparator bank.

The default is four bits with fifteen continuous-time OTA comparators.
The clocked StrongArm option uses the shared comparator builder. Output codes
are reconstructed by counting high comparator outputs; the encoder builder is
an optional behavioral block and is not placed in the SPICE circuit.

Run .venv/bin/python -m silk.examples.flash_adc_example from the repository root.
When run as a script, --bits N selects N >= 2, --strongarm selects clocked
comparators, --sine selects a coherent sine, and --no-spice runs only the
behavioral model. The default stimulus is a 0.8 to 1.4 V ramp with a 20 MHz clock.
SPICE requires ngspice and sky130 models. The script prints checks, specification
results and sampled transfer tables, and saves plots and MD, JSON and HTML reports
in the configured example output directory.

--offset-mc 16 --chips 16 --seed 468 characterizes comparator mismatch in SPICE
and runs an explicit behavioral chip loop with fixed seeded offsets. Bounded
or unavailable offset observations prevent calibration of that chip variation.
"""

import argparse
import gc
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt


from silk import load_pdk, PortDir, PortRole, Port, generate_report
from silk import check_voltage_domains, check_junction_overrange
from silk.circuit.interface_check import check_interfaces, annotation_coverage
from silk.circuit.modules import Module
from silk.simulation import DC, Transient
from silk.exceptions import SimulationError
from silk.sources.classes import VoltageSource
from silk.sources.waveforms import PWL, Pulse, Sine
from silk.elements import elements as el
from silk.examples.strongarm_tb import (
                                        AVDD,
                                        make_comparator_strongarm, make_clock_delay,
                                        )
from silk.metrics import sample, sndr, fft, peak_to_peak
from silk.signals import codes_from_bits
from silk.examples import example_output_dir


# Shared specification limits and descriptions for both characterization paths.
ADC_SPECS = {
    'code_span': ('>=', None, 'Full-scale ADC code span. The ramp covers the ladder '
                  'rails; the sine leaves 20 mV margin and may not exercise end codes.'),
    'th_sndr': ('>=', 20.0, 'Track-and-hold SNDR from late-hold samples and a coherent FFT.'),
    'adc_sndr': ('>=', 18.0, 'ADC SNDR from reconstructed code converted to voltage.'),
    'offset': ('<=', 5e-3, 'Comparator input-referred offset magnitude.'),
}

# ── Global constants ──────────────────────────────────────────────────────────
VDD     = 1.8           # supply / full-scale reference  (V)
OFFSET_HALF_RANGE = 100e-3  # comparator offset bench: ±100 mV around vm=1 V

# ── Voltage domains ────────────────────────────────────────────────────────
# Defined once, shared across all subcircuits
N_BITS  = 4             # ADC resolution
N_COMP  = 2**N_BITS - 1 # 15 comparators for 4-bit flash

# Reference range: ladder spans VREFL → VREFH, narrower than the supply
# to keep comparator inputs above Vth (VREFL) and give the PMOS loads
# headroom below VDD (VREFH). Input ramp / sine must live in this range.
VREFL   = 0.8           # bottom of ladder (V)
VREFH   = 1.4           # top    of ladder (V)
VSTEP   = (VREFH - VREFL) / (N_COMP + 1)  # ≈ 37.5 mV per tap at 4 bits

# Plot headroom around the ladder range — panels showing vin / vh / taps use
# this instead of the full 0–VDD supply span, so the thresholds fill the axes.
V_MARGIN = 0.1 * (VREFH - VREFL)

CLK_PER = 50e-9         # clock period  (20 MHz). Slow enough to give the
                        # StrongArm plenty of regeneration time per cycle.
CLK_PW  = CLK_PER / 2   # 50 % duty cycle (25 ns high / 25 ns low)
# Clock edge — must be nonzero, or the behavioral twin and the SPICE deck
# disagree on the clock phase (see silk.sources.Pulse). CLK_PER/500 = 100 ps:
# one T_STEP, so the ramp is resolved, and 5x inside the 0.5 ns that
# SAMPLE_S_IN_CYCLE leaves before the rising edge it samples against.
CLK_TR  = CLK_PER / 500
T_STEP  = 0.1e-9        # time step (100 ps)
T_STOP  = 16 * CLK_PER  # 16 clock cycles (800 ns at 20 MHz)

# Comparator topology switch:
#   False → the continuous-time OTA-style comparator (make_comparator)
#   True  → the clocked StrongArm-latch comparator (make_comparator_strongarm)
# The StrongArm evaluates on the T/H hold phase (gate = clkb).
# Default here; overridable via --strongarm on the CLI.
USE_STRONGARM = False

# Input-stimulus switch:
#   False → VREFL → VREFH ramp (0.8 → 1.4 V), for the static transfer function.
#   True  → coherent-sampling sine wave, used for SNDR measurement.
# Default here; overridable via --sine on the CLI.
USE_SINE = False

# Sine / FFT settings (only used when USE_SINE=True).
# Coherent sampling: f_sig / Fs = M / N with M coprime to N → no leakage,
# no window needed. Fs = 1 / CLK_PER = 20 MHz.
SINE_N_FFT   = 128                       # samples per FFT
SINE_M_COH   = 11                        # bin index; 11 is prime, coprime to 128
SINE_FREQ    = SINE_M_COH / SINE_N_FFT / CLK_PER   # ≈ 1.71875 MHz
SINE_T_STOP  = SINE_N_FFT * CLK_PER      # 6.4 µs (128 samples × 50 ns)
SINE_VOFFSET = (VREFL + VREFH) / 2       # 1.1 V, mid of ladder range
SINE_VAMPL   = (VREFH - VREFL) / 2 - 0.02  # ~0.28 V peak → swings ~0.82 → 1.38 V
                                         # (20 mV margin so samples stay in-range)


# ══════════════════════════════════════════════════════════════════════════════
#  1.  CMOS Track-and-Hold
# ══════════════════════════════════════════════════════════════════════════════
def make_track_hold(pdk, out_dir):
    """
    SPICE topology
    ──────────────
      NMOS switch (mn_sw):  vin → vh,  gate = clk
      PMOS switch (mp_sw):  vin → vh,  gate = clkb  (inverted clock)
      Hold cap (c_hold):    vh → vss,  100 fF

    The complementary CMOS gate equalises the on-resistance across the
    entire common-mode range, reducing signal-dependent charge injection.

    Behavioral model
    ────────────────
      Tracks vin while CLK is high; holds the last sampled value while CLK is low.
    """
    th = Module('PDK_skywater130', out_dir, 'track_hold.cir',
                pdk_details=pdk, module_name='track_hold')

    # SPICE structure
    mn_sw = pdk.nmos1p8('mn_sw', d='vh', g='clk',  s='vin', b='vss',
                         w=4.0, l=0.18)
    mp_sw = pdk.pmos1p8('mp_sw', d='vh', g='clkb', s='vin', b='vdd',
                         w=8.0, l=0.18)   # 2× width for equal drive strength
    c_hold = el.capacitor('c_hold', p='vh', n='vss', value=100e-15)
    th.add(mn_sw, mp_sw, c_hold)

    th.define_module_ports({
        'vin':  PortDir.IN,
        'clk':  (PortDir.IN,  PortRole.CLOCK),
        'clkb': (PortDir.IN,  PortRole.CLOCK),
        'vdd':  Port(dir=PortDir.IN,  role=PortRole.POWER,  domain=AVDD),
        'vss':  Port(dir=PortDir.IN,  role=PortRole.GROUND, domain=AVDD),
        'vh':   Port(dir=PortDir.OUT, role=PortRole.ANALOG,  domain=AVDD),
    })

    # Behavioral model — pure Python, no SPICE required
    def _th_model(ports, t, params):
        vin, clk = ports.vin.v, ports.clk.v
        vh   = np.empty_like(t)
        held = 0.0
        for i in range(len(t)):
            if clk[i] > 0.9:       # tracking phase (CLK high)
                vh[i] = held = vin[i]
            else:                   # hold phase (CLK low)
                vh[i] = held
        ports.vh.v = vh

    th.set_behavioral_model(_th_model, execution='vector')
    return th


# ══════════════════════════════════════════════════════════════════════════════
#  2.  Resistor Reference Ladder
# ══════════════════════════════════════════════════════════════════════════════
def make_ref_ladder(pdk, out_dir, *, n_bits=None):
    """
    SPICE topology
    ──────────────
      2**n_bits equal 10 kΩ resistors in series, vref → vrefl. Each tap gets a
      decoupling cap to vss so StrongArm kickback (charge injected back
      through the comparator's input Cgd) is absorbed locally and vref_k
      settles before the latch samples.

        vref ─ R1 ─ vref15 ─ R2 ─ vref14 ─ … ─ vref1 ─ R16 ─ vrefl
                    │           │               │
                   C15         C14             C1    (to vss)

      vrefl is externally driven (allows non-ground pedestal so the
      comparator diff pair stays in saturation).

    Behavioral model
    ────────────────
      vrefk = vrefl + (vref - vrefl) × k / (2**n_bits)
    """
    n_bits = N_BITS if n_bits is None else n_bits
    n_comp = 2 ** n_bits - 1
    CAP_PER_TAP = 500e-15   # 500 fF decoupling cap per vref_k tap

    ladder = Module('PDK_skywater130', out_dir, 'ref_ladder.cir',
                    pdk_details=pdk, module_name='ref_ladder')

    # SPICE structure — n_comp+1 resistors, n_comp taps + one decoupling cap each
    nodes = ['vref'] + [f'vref{k}' for k in range(n_comp, 0, -1)] + ['vrefl']
    for i in range(n_comp + 1):
        ladder.add(el.resistor(f'r{i + 1}', p=nodes[i], n=nodes[i + 1],
                               value=10e3))
    for k in range(1, n_comp + 1):
        ladder.add(el.capacitor(f'c{k}', p=f'vref{k}', n='vss',
                                value=CAP_PER_TAP))

    ladder.define_module_ports({
        'vref':  PortDir.IN,
        'vrefl': PortDir.IN,
        'vss':   Port(dir=PortDir.IN, role=PortRole.GROUND, domain=AVDD),
        **{f'vref{k}': PortDir.OUT for k in range(1, n_comp + 1)},
    })

    # Behavioral model
    def _ladder_model(ports, t, params):
        vref  = ports.vref.v
        vrefl = ports.vrefl.v
        span  = vref - vrefl
        for k in range(1, n_comp + 1):
            getattr(ports, f'vref{k}').v = vrefl + span * k / (n_comp + 1)

    ladder.set_behavioral_model(_ladder_model, execution='vector')
    return ladder


# ══════════════════════════════════════════════════════════════════════════════
#  3.  Differential Comparator
# ══════════════════════════════════════════════════════════════════════════════
def make_comparator(pdk, out_dir):
    """
    SPICE topology
    ──────────────
      NMOS differential pair (MN1, MN2) with PMOS current-mirror load (MP3, MP4).
      Tail current source (MN5) biased by vbias.
      Two cascaded inverter stages (MP6/MN7, MP8/MN9): the first stage sharpens
      vout_p into a rail-to-rail digital signal ``vout_b``; the second inverts
      again so vout is HIGH when vp > vm (matches the behavioral polarity).

          vdd ─ MP3(diode-connected) ─ MP4
                       │                 │
                    vout_n            vout_p ─┬─ MP6 ─ vdd     MP8 ─ vdd
                       │                 │   └─ MN7 ─ vss   ┌── MN9 ─ vss
                  MN1(vp)          MN2(vm)         │         │
                       │                 │      vout_b ──────┘     vout
                       └─────── vtail ───┘
                                  │
                              MN5(vbias)
                                  │
                                 vss

    Behavioral model
    ────────────────
      vout = VDD  when vp > vm
      vout = 0 V  otherwise
    """
    comp = Module('PDK_skywater130', out_dir, 'comparator.cir',
                  pdk_details=pdk, module_name='comparator')

    mn1 = pdk.nmos1p8('mn1', d='vout_n', g='vp',     s='vtail', b='vss',
                       w=2.0, l=0.18)
    mn2 = pdk.nmos1p8('mn2', d='vout_p', g='vm',     s='vtail', b='vss',
                       w=2.0, l=0.18)
    mp3 = pdk.pmos1p8('mp3', d='vout_n', g='vout_n', s='vdd',   b='vdd',
                       w=4.0, l=0.18)
    mp4 = pdk.pmos1p8('mp4', d='vout_p', g='vout_n', s='vdd',   b='vdd',
                       w=4.0, l=0.18)
    mn5 = pdk.nmos1p8('mn5', d='vtail',  g='vbias',  s='vss',   b='vss',
                       w=2.0, l=0.50)
    # First output-stage inverter: vout_p → vout_b (inverted)
    mp6 = pdk.pmos1p8('mp6', d='vout_b', g='vout_p', s='vdd',   b='vdd',
                       w=2.0, l=0.18)
    mn7 = pdk.nmos1p8('mn7', d='vout_b', g='vout_p', s='vss',   b='vss',
                       w=1.0, l=0.18)
    # Second inverter: vout_b → vout (re-inverted → matches vp>vm polarity)
    mp8 = pdk.pmos1p8('mp8', d='vout',   g='vout_b', s='vdd',   b='vdd',
                       w=4.0, l=0.18)
    mn9 = pdk.nmos1p8('mn9', d='vout',   g='vout_b', s='vss',   b='vss',
                       w=2.0, l=0.18)
    comp.add(mn1, mn2, mp3, mp4, mn5, mp6, mn7, mp8, mn9)

    comp.define_module_ports({
        'vp':    Port(dir=PortDir.IN,  role=PortRole.ANALOG,  domain=AVDD),
        'vm':    Port(dir=PortDir.IN,  role=PortRole.ANALOG,  domain=AVDD),
        'vbias': Port(dir=PortDir.IN,  role=PortRole.ANALOG,  domain=AVDD),
        'vdd':   Port(dir=PortDir.IN,  role=PortRole.POWER,   domain=AVDD),
        'vss':   Port(dir=PortDir.IN,  role=PortRole.GROUND,  domain=AVDD),
        'vout':  Port(dir=PortDir.OUT, role=PortRole.DIGITAL, domain=AVDD),
    })

    # Parameterized behavioral comparator model.
    def _comparator_model(ports, t, params):
        offset = params['offset_voltage']
        ports.vout.v = np.where(ports.vp.v + offset > ports.vm.v, VDD, 0.0)

    comp.set_behavioral_model(_comparator_model, params={'offset_voltage': 0.0},
                              execution='vector')
    return comp


# ══════════════════════════════════════════════════════════════════════════════
#  3d.  Ideal T/H output buffer (VCVS)
# ══════════════════════════════════════════════════════════════════════════════
def make_ideal_buffer(pdk, out_dir):
    """Unity-gain buffer that isolates the T/H hold cap from the comparator
    inputs (all 2**N_BITS−1 of them). Prevents comparator kickback (Cgd
    transients from di_p/di_n at each evaluate edge) from pumping charge off
    the hold node.

    SPICE: a VCVS (E element), gain=1. Infinite input impedance, zero
    output impedance — ideal behavior, good enough for a design-stage
    model that isolates the kickback problem from the rest of the ADC.

    Behavioral: identity pass-through.
    """
    from silk.sources.classes import VCVS

    buf = Module('PDK_skywater130', out_dir, 'ideal_buf.cir',
                 pdk_details=pdk, module_name='ideal_buf')

    # SPICE: E<name> out_pos out_neg ctrl_pos ctrl_neg gain
    buf.add_source(VCVS('ebuf',
                        out_pos='vout', out_neg='vss',
                        ctrl_pos='vin',  ctrl_neg='vss',
                        gain=1.0))

    buf.define_module_ports({
        'vin':  Port(dir=PortDir.IN,  role=PortRole.ANALOG, domain=AVDD),
        'vss':  Port(dir=PortDir.IN,  role=PortRole.GROUND, domain=AVDD),
        'vout': Port(dir=PortDir.OUT, role=PortRole.ANALOG, domain=AVDD),
    })

    def _buffer_model(ports, t, params):
        ports.vout.v = np.asarray(ports.vin.v, dtype=float)

    buf.set_behavioral_model(_buffer_model, execution='vector')
    return buf


# ══════════════════════════════════════════════════════════════════════════════
#  4.  Thermometer-to-Binary Encoder  (behavioral only)
# ══════════════════════════════════════════════════════════════════════════════
def make_encoder(pdk, out_dir, *, n_bits=None):
    """
    Counts the number of high comparator outputs (thermometer code → integer),
    then extracts the n_bits binary bits.

    No SPICE transistor implementation — this block is digital and would
    typically be synthesised from standard cells.  The .subckt in the
    generated netlist is intentionally empty.
    """
    n_bits = N_BITS if n_bits is None else n_bits
    n_comp = 2 ** n_bits - 1
    enc = Module('PDK_skywater130', out_dir, 'therm_to_bin.cir',
                 pdk_details=pdk, module_name='therm_to_bin')

    enc.define_module_ports({
        **{f'vcmp{k}': (PortDir.IN,  PortRole.DIGITAL) for k in range(1, n_comp + 1)},
        **{f'd{b}':    (PortDir.OUT, PortRole.DIGITAL) for b in range(n_bits)},
    })

    def _encoder_model(ports, t, params):
        # Stack all n_comp comparator outputs → shape (n_comp, len(t))
        therm = np.stack([getattr(ports, f'vcmp{k}').v for k in range(1, n_comp + 1)])
        code  = np.sum(therm > 0.9, axis=0).astype(int)   # thermometer count
        for b in range(n_bits):
            getattr(ports, f'd{b}').v = ((code >> b) & 1).astype(float) * VDD

    enc.set_behavioral_model(_encoder_model, execution='vector')
    return enc


# ══════════════════════════════════════════════════════════════════════════════
#  4b.  SNDR analysis
# ══════════════════════════════════════════════════════════════════════════════
# Sample point within each clock cycle, in seconds.
#
# T/H holds during the second half of each cycle (clk LOW), and the
# StrongArm evaluates during that same half (its clock is clkb). We
# sample 0.5 ns before the next reset edge so the latch has the full
# evaluate window to regenerate, but we don't land on the reset edge
# itself. Scales automatically with CLK_PER.
SAMPLE_S_IN_CYCLE = CLK_PER - 0.5e-9


def _sample_indices_at_phase(t, num_cycles):
    """Indices into time array ``t`` for ``num_cycles`` one-per-cycle samples,
    each taken late in the evaluate / hold phase.

    Uses ``side='right' - 1`` so ``t[idx]`` is the last SPICE timepoint ≤ the
    requested sample time. ngspice places timepoints adaptively, so we don't
    assume uniform spacing.
    """
    sample_times = np.arange(num_cycles) * CLK_PER + SAMPLE_S_IN_CYCLE
    idx = np.searchsorted(t, sample_times, side='right') - 1
    return np.clip(idx, 0, len(t) - 1)


def _sine_sample_indices(t):
    """SNDR convenience: ``SINE_N_FFT`` samples, one per cycle."""
    return _sample_indices_at_phase(t, SINE_N_FFT)


# ══════════════════════════════════════════════════════════════════════════════
#  5.  Top-Level Flash ADC
# ══════════════════════════════════════════════════════════════════════════════
def make_flash_adc(netlist_name='flash_adc_4bit.cir', *, with_specs=True,
                   comparator_offsets=None, n_bits=None, strongarm=None, sine=None, t_stop=None):
    """Assemble the flat ADC deck and return (adc_module, out_dir).

    The demo passes ``with_specs=False`` and declares the same specifications
    through its characterization testbenches instead. Optional comparator_offsets
    are fixed physical input offsets in volts, one per comparator in tap order.
    """
    n_bits = N_BITS if n_bits is None else n_bits
    strongarm = USE_STRONGARM if strongarm is None else strongarm
    sine = USE_SINE if sine is None else sine
    n_comp = 2 ** n_bits - 1
    if t_stop is None:
        t_stop = SINE_T_STOP if sine else T_STOP
    if n_bits < 2:
        raise ValueError('Flash ADC needs at least two bits.')
    if comparator_offsets is not None:
        comparator_offsets = tuple(float(v) for v in comparator_offsets)
        if len(comparator_offsets) != n_comp or not np.isfinite(comparator_offsets).all():
            raise ValueError(f'Provide {n_comp} finite physical comparator offsets.')
    # Use the configured example output directory.
    out_dir = example_output_dir()
    pdk     = load_pdk('sky130')
    # Top-level module: no module_name → generates a flat netlist, not a .subckt
    adc     = Module('PDK_skywater130', out_dir, netlist_name,
                     pdk_details=pdk)
    adc.comparator_offsets = comparator_offsets

    # ── Stimulus sources ──────────────────────────────────────────────────────
    if sine:
        # Coherent-sampling sine for SNDR measurement.
        adc.add_source(VoltageSource(
            'vin_src', positive='vin', negative='gnd',
            sine=Sine(voffset=SINE_VOFFSET, vampl=SINE_VAMPL, freq=SINE_FREQ)))
    else:
        # Input ramp spans VREFL to VREFH.
        ramp = PWL().add(0, VREFL).add(t_stop, VREFH)
        adc.add_source(VoltageSource('vin_src', positive='vin', negative='gnd', pwl=ramp))
    adc.add_source(VoltageSource('vref_src',  positive='vref',  negative='gnd', dc=VREFH))
    adc.add_source(VoltageSource('vrefl_src', positive='vrefl', negative='gnd', dc=VREFL))
    adc.add_source(VoltageSource('vdd_src',   positive='vdd',   negative='gnd', dc=VDD))
    adc.add_source(VoltageSource('vbias_src', positive='vbias', negative='gnd', dc=0.7))
    adc.add_source(VoltageSource('clk_src',   positive='clk',   negative='gnd',
                                pulse=Pulse(v1=0, v2=VDD, td=0, tr=CLK_TR, tf=CLK_TR,
                                            pw=CLK_PW, per=CLK_PER)))
    # Inverted clock for the PMOS switch gate in the CMOS TG, and for the
    # StrongArm latch evaluate trigger when strongarm is True.
    adc.add_source(VoltageSource('clkb_src',  positive='clkb',  negative='gnd',
                                pulse=Pulse(v1=VDD, v2=0, td=0, tr=CLK_TR, tf=CLK_TR,
                                            pw=CLK_PW, per=CLK_PER)))

    # ── Track-and-Hold ────────────────────────────────────────────────────────
    adc.add_module(make_track_hold(pdk, out_dir), 'th1', nets={
        'vin':  'vin',
        'clk':  'clk',
        'clkb': 'clkb',
        'vdd':  'vdd',
        'vss':  'gnd',
        'vh':   'vh',
    })

    # ── Reference ladder ──────────────────────────────────────────────────────
    ladder_nets = {'vref': 'vref', 'vrefl': 'vrefl', 'vss': 'gnd'}
    ladder_nets.update({f'vref{k}': f'vref{k}' for k in range(1, n_comp + 1)})
    adc.add_module(make_ref_ladder(pdk, out_dir, n_bits=n_bits), 'ladder1',
                   nets=ladder_nets)

    # ── Clock delay (StrongArm only) ──────────────────────────────────────────
    # Two-inverter chain between the T/H clock and the StrongArm strobe.
    # Gives Vh time to settle after the T/H switches open, before the latch
    # starts regenerating. ~120 ps delay.
    if strongarm:
        adc.add_module(make_clock_delay(pdk, out_dir), 'clkdel1',
                       nets={
                           'clkb':   'clkb',
                           'vdd':    'vdd',
                           'vss':    'gnd',
                           'clkb_d': 'clkb_d',
                       })

    # ── Ideal buffer between T/H output and comparators ──────────────────────
    # Isolates the hold cap from comparator kickback. VCVS in SPICE,
    # pass-through in behavioral.
    adc.add_module(make_ideal_buffer(pdk, out_dir), 'vh_buf1',
                   nets={
                       'vin':  'vh',
                       'vss':  'gnd',
                       'vout': 'vh_buf',
                   })

    # ── n_comp comparators, one per reference tap ─────────────────────────────
    # The buffer isolates comparator inputs from the hold capacitor.
    _comp_factory = (make_comparator_strongarm if strongarm
                     else make_comparator)
    for k in range(1, n_comp + 1):
        if strongarm:
            comp_nets = {
                'vp':   'vh_buf',
                'vm':   f'vref{k}',
                'clkb': 'clkb_d',   # <-- delayed clock (post 2-inv chain)
                'vdd':  'vdd',
                'vss':  'gnd',
                'vout': f'vcmp{k}',
            }
        else:
            comp_nets = {
                'vp':    'vh_buf',
                'vm':    f'vref{k}',
                'vbias': 'vbias',
                'vdd':   'vdd',
                'vss':   'gnd',
                'vout':  f'vcmp{k}',
            }
        comp = _comp_factory(pdk, out_dir)
        if comparator_offsets is not None:
            comp.variants = True
            comp.calibrate_behavioral('offset_voltage', -comparator_offsets[k - 1])
        adc.add_module(comp, f'comp{k}', nets=comp_nets)

    # ── Thermometer-to-binary encoder ─────────────────────────────────────────
    # Encoder is behavioral-only (empty .subckt); excluded so ngspice can run
    # the analog front-end. Code is reconstructed in Python from vcmp* below.
    # Assign voltage domains to top-level supply nets
    adc.assign_net_domain('vdd', AVDD)
    adc.assign_net_domain('gnd', AVDD)

    # Annotation coverage — how many ports have PortRole declared?
    cov = annotation_coverage(adc)
    print(f"\n{cov}")
    if cov.unannotated_ports:
        print("  Unannotated ports:")
        for u in cov.unannotated_ports:
            print(f"    {u.instance}.{u.port}  ({u.direction})  [{u.module_name}]")

    # Interface check — verify port roles are compatible across hierarchy
    violations = check_interfaces(adc)
    if violations:
        print("\nInterface violations:")
        for v in violations:
            print(f"  [{v.severity}] {v.kind}: {v.message}")
    else:
        print("Interface check: PASS (no port conflicts)")

    # Domain check — verify voltage domain consistency across port connections
    domain_violations = check_voltage_domains(adc)
    if domain_violations:
        print("\nDomain violations:")
        for v in domain_violations:
            print(f"  [{v.severity}] {v.kind}: {v.message}")
    else:
        print("Domain check: PASS (no domain conflicts)")

    adc.adc_n_bits = n_bits
    adc.adc_strongarm = strongarm
    adc.adc_sine = sine
    adc.adc_t_stop = t_stop
    adc.adc_spectral = dict(tclk=CLK_PER, tstart=SAMPLE_S_IN_CYCLE,
                            n_samples=SINE_N_FFT)
    if not with_specs:
        return adc, out_dir

    # ── Spec declarations ─────────────────────────────────────────────────────

    # Top-level system spec: ADC code must span the full range [0, n_comp]
    system_code_span = adc.declare_spec(
        name='adc_code_span',
        metric='code_span',
        op=ADC_SPECS['code_span'][0],
        target=float(n_comp),
        units='LSB',
        description=ADC_SPECS['code_span'][2],
    )

    # SNDR specs (only meaningful when sine=True). Ideal SQNR is
    # 6.02·N + 1.76 dB; targets give margin for non-idealities.
    if sine:
        adc.declare_spec(
            name='th_sndr',
            metric='sndr_db',
            op=ADC_SPECS['th_sndr'][0],
            target=ADC_SPECS['th_sndr'][1],
            units='dB',
            description=ADC_SPECS['th_sndr'][2],
        )
        adc.declare_spec(
            name='adc_sndr',
            metric='sndr_db',
            op=ADC_SPECS['adc_sndr'][0],
            target=ADC_SPECS['adc_sndr'][1],
            units='dB',
            description=ADC_SPECS['adc_sndr'][2],
            parent=system_code_span,
        )

    # Declare offset specs on each placed comparator.
    for k in range(1, n_comp + 1):
        comp_instance = adc.elem_dict[f'comp{k}']['device']
        comp_instance.declare_spec(
            name=f'comp{k}_offset',
            metric='abs_offset',
            op=ADC_SPECS['offset'][0],
            target=ADC_SPECS['offset'][1],      # 5 mV
            units='V',
            description=ADC_SPECS['offset'][2],
            parent=system_code_span,
        )

    return adc, out_dir


def _code_trace(ckt, *, n_comp=None):
    """Count high comparator outputs on the shared simulation grid."""
    n_comp = N_COMP if n_comp is None else n_comp
    bits = [ckt.load_signal(f'vcmp{k}') for k in range(1, n_comp + 1)]
    code = codes_from_bits(bits, weighting='thermometer', threshold=VDD / 2,
                           name='code')
    if not np.array_equal(code.x, ckt.load_signal('vin').x):
        raise ValueError('flash ADC code trace: vin and comparator bits must '
                         'share the same simulation time grid')
    return code


def _make_characterization(adc):
    """Use one DUT testbench for both simulation backends."""

    n_bits = adc.adc_n_bits
    n_comp = 2 ** n_bits - 1
    strongarm = adc.adc_strongarm
    sine = adc.adc_sine
    vstep = (VREFH - VREFL) / (n_comp + 1)
    adc.define_module_name('flash_adc')
    # Expose the existing observation nodes so probes keep their familiar names.
    # This deck includes its stimuli and global ground; ground is not a port.
    adc.define_module_ports({name: PortDir.INOUT for name in adc.net_dict
                             if name != 'gnd'})
    test = adc.create_test('conversion')
    test.add_module_under_test({name: name for name in adc.pin_list})
    test.netlist.assign_net_domain('gnd', AVDD)
    # Keep the observations used by the conversion metrics and plots, plus
    # the buffered input and clocks. Code is reconstructed from vcmp*, not
    # emitted by a SPICE encoder. Avoid saving unused device-internal nodes.
    saved = ['vin', 'vh', 'vh_buf', 'clk', 'clkb']
    saved.extend(f'vref{k}' for k in range(1, n_comp + 1))
    saved.extend(f'vcmp{k}' for k in range(1, n_comp + 1))
    if strongarm:
        saved.append('clkb_d')
        picks = (max(1, n_comp // 4), max(1, n_comp // 2),
                 min(n_comp, (3 * n_comp) // 4))
        saved.extend(f'xdut.xcomp{k}.{node}' for k in picks
                     for node in ('vtail', 'di_p', 'di_n'))
    test.netlist.set_save_list(saved)
    test.add_sim_type(Transient(step=T_STEP, stop=adc.adc_t_stop))
    test.add_metric('code_span', lambda ckt: peak_to_peak(_code_trace(ckt, n_comp=n_comp)))
    system = test.add_spec('adc_code_span', 'code_span', ADC_SPECS['code_span'][0], float(n_comp))
    system.units = 'LSB'
    system.description = ADC_SPECS['code_span'][2]
    if sine:
        spectral = adc.adc_spectral.copy()
        test.add_metric('th_sndr', sndr, sig=test.probe('vh'), **spectral)
        test.add_metric('adc_sndr', lambda ckt: sndr(
            VREFL + (_code_trace(ckt, n_comp=n_comp) + 0.5) * vstep, **spectral))
        th_spec = test.add_spec('th_sndr', 'th_sndr', ADC_SPECS['th_sndr'][0], ADC_SPECS['th_sndr'][1])
        th_spec.units = 'dB'
        th_spec.description = ADC_SPECS['th_sndr'][2]
        adc_spec = test.add_spec('adc_sndr', 'adc_sndr', ADC_SPECS['adc_sndr'][0], ADC_SPECS['adc_sndr'][1])
        adc_spec.units = 'dB'
        adc_spec.description = ADC_SPECS['adc_sndr'][2]
        adc_spec.parent = system
        system.children.append(adc_spec)

    # Fixed physical draws are model inputs, not measurements from the ADC.
    for k in range(1, n_comp + 1):
        if adc.comparator_offsets is not None:
            metric = f'comp{k}_abs_offset'
            value = abs(adc.comparator_offsets[k - 1])
            test.add_metric(metric, lambda ckt, value=value: value)
            spec = test.add_spec(f'comp{k}_offset', metric, ADC_SPECS['offset'][0], ADC_SPECS['offset'][1])
            spec.description = 'Magnitude of the fixed, sampled physical offset'
            spec.parent = system
            system.children.append(spec)
        else:
            comp = adc.elem_dict[f'comp{k}']['device']
            comp.declare_spec(
                name=f'comp{k}_offset', metric='abs_offset', op=ADC_SPECS['offset'][0],
                target=ADC_SPECS['offset'][1], units='V',
                description=ADC_SPECS['offset'][2],
                parent=system,
            )
    return test


def _offset_from_decisions(sig_vout, sig_vp, *, clocked):
    """Bracket a low-to-high decision; cap constant outputs at +/-100 mV.

    Constant-low/high outputs return +100/-100 mV bounds, not measurements.
    Thresholds within one decision step of a ramp end are capped too.
    Multiple decision reversals remain nan (unmeasurable). Clocked decisions
    are read late in evaluate, after startup;
    their input is the value at the start of evaluate. Resolution is one
    decision's input increment, not the simulator's tiny clock-edge timestep.
    """
    vp = np.asarray(sig_vp.y)
    out = np.asarray(sig_vout.y)
    if clocked:
        t = np.asarray(sig_vout.x)
        cycles = np.arange(1, int(t[-1] / CLK_PER))
        if len(cycles):
            vp = sample(sig_vp, x_new=(cycles + 0.5) * CLK_PER).y
            out = sample(sig_vout, x_new=(cycles + 0.9) * CLK_PER).y
        else:
            vp = out = np.empty(0)
    high = out > VDD / 2
    flips = np.flatnonzero(np.diff(high.astype(int)))
    if (len(high) < 2 or not np.all(np.isfinite(out))
            or not np.all(np.isfinite(vp))):
        return float('nan')
    if not high.any():
        return OFFSET_HALF_RANGE
    if high.all():
        return -OFFSET_HALF_RANGE
    if high[0] or not high[-1] or len(flips) != 1:
        return float('nan')
    i = flips[0]
    offset = float((vp[i] + vp[i + 1]) / 2 - 1.0)
    step = float(vp[i + 1] - vp[i])
    if offset <= -OFFSET_HALF_RANGE + step:
        return -OFFSET_HALF_RANGE
    if offset >= OFFSET_HALF_RANGE - step:
        return OFFSET_HALF_RANGE
    return offset


def make_comparator_offset_tb(comp, *, strongarm=False):
    """Register one comparator's offset bench at vm=1 V, with +/-100 mV range.

    Continuous comparators use a 50 uV DC grid. StrongARM uses a 40 us ramp
    and 50 ns clock: 0.25 mV per decision. No system-ADC rawfile is involved.
    The physical offset is vp_at_threshold - vm; the behavioral model's
    additive input parameter therefore receives its negative.
    """
    tb = comp.create_test('offset')
    tb.add_module_under_test({name: name for name in comp.pin_list})
    for name, value in [('vdd', VDD), ('vss', 0.0), ('vm', 1.0)]:
        tb.add_source(VoltageSource(name, positive=name, negative='gnd', dc=value))
    if strongarm:
        tb.add_source(VoltageSource(
            'clkb', positive='clkb', negative='gnd',
            pulse=Pulse(v1=VDD, v2=0, td=0, tr=CLK_TR, tf=CLK_TR,
                        pw=CLK_PW, per=CLK_PER)))
        tb.add_source(VoltageSource('vp', positive='vp', negative='gnd',
                                    pwl=PWL().add(0, 1.0 - OFFSET_HALF_RANGE)
                                             .add(40e-6, 1.0 + OFFSET_HALF_RANGE)))
        tb.add_sim_type(Transient(step=0.5e-9, stop=40e-6,
                                  maxstep=0.5e-9, uic=True))
    else:
        tb.add_source(VoltageSource('vbias', positive='vbias', negative='gnd', dc=0.7))
        tb.add_source(VoltageSource('vp', positive='vp', negative='gnd',
                                    dc=1.0 - OFFSET_HALF_RANGE))
        tb.add_sim_type(DC(source='vp', start=1.0 - OFFSET_HALF_RANGE,
                            stop=1.0 + OFFSET_HALF_RANGE, step=50e-6))
    tb.netlist.set_save_list(['vp', 'vout'])
    offset = tb.add_metric('offset_v', _offset_from_decisions,
                           sig_vout=tb.probe('vout'), sig_vp=tb.probe('vp'),
                           clocked=strongarm)
    from silk.examples.strongarm_tb import _offset_cap_side
    tb.add_metric('offset_cap_side',
                  lambda sig_vout, sig_vp: _offset_cap_side(
                      _offset_from_decisions(sig_vout, sig_vp, clocked=strongarm),
                      limit=OFFSET_HALF_RANGE),
                  sig_vout=tb.probe('vout'), sig_vp=tb.probe('vp'))
    tb.add_metric('abs_offset_v', abs(offset))
    tb.add_spec('offset magnitude', 'abs_offset_v', '<=', 5e-3)
    return tb


def characterize_offset(*, samples=16, seed=468, strongarm=False):
    """SPICE mismatch MC of one comparator; retain only scalar statistics."""
    if samples < 2:
        raise ValueError('Offset sigma needs at least two samples.')
    factory = make_comparator_strongarm if strongarm else make_comparator
    comp = factory(load_pdk('sky130'), os.path.join(example_output_dir(), 'offset_mc'))
    tb = make_comparator_offset_tb(comp, strongarm=strongarm)
    started = time.perf_counter()
    # silk evaluates each sample from its already-loaded results once. The
    # two probes share that load; derived abs(offset) reuses the scalar metric.
    mc = tb.run_monte_carlo(samples, seed=seed, variation='mismatch')
    elapsed = time.perf_counter() - started
    values = [r.metrics['offset_v'] for r in mc.reports.values()]
    usable = np.asarray([v for v in values if isinstance(v, (float, int))
                         and np.isfinite(v)])
    sides = [r.metrics.get('offset_cap_side', 0) for r in mc.reports.values()]
    lower, upper = sides.count(-1), sides.count(1)
    failed = samples - len(usable)
    print(f'Offset SPICE mismatch MC: N={samples}, seed={seed}, failed={failed} '
          f'(simulator={len(mc.failed)}, unavailable={failed - len(mc.failed)})')
    print(f'  capped={lower + upper}/{samples}: '
          f'{-OFFSET_HALF_RANGE * 1e3:+g} mV={lower}, '
          f'{OFFSET_HALF_RANGE * 1e3:+g} mV={upper}')
    if lower or upper or failed:
        raise SimulationError('Offset calibration needs uncensored, measurable samples; '
                              'widen or refine the offset bench before using chip variation.')
    mean = float(usable.mean())
    sigma = float(usable.std(ddof=1))
    # Approximate standard error for independent normal draws.
    uncertainty = sigma / np.sqrt(2 * (len(usable) - 1))
    resolution = 0.25e-3 if strongarm else 50e-6
    print(f'  mean={mean * 1e3:+.4f} mV, sigma={sigma * 1e3:.4f} mV, '
          f'sigma uncertainty (approx 1 SE, normal)={uncertainty * 1e3:.4f} mV')
    print(f'  per-sample offset resolution={resolution * 1e3:.4f} mV; '
          f'sigma step wall time={elapsed:.3f} s')
    if not np.isfinite(sigma) or sigma < 10 * resolution:
        raise SimulationError('Offset sigma is unavailable or less than ten '
                              'resolution steps. Refine the offset bench before '
                              'using this estimate for behavioral chips.')
    return mean, sigma


def run_offset_chips(*, mean, sigma, chips=16, seed=468):
    """User-level seeded chip loop, not built-in behavioral Monte Carlo."""
    if chips < 1 or seed < 0 or not np.isfinite([mean, sigma]).all() or sigma < 0:
        raise ValueError('Use chips >= 1, seed >= 0, finite mean and sigma >= 0.')
    print('This example uses an explicit chip loop.')
    print(f'K={chips}, first chip seed={seed}; each chip holds its offsets '
          'through every registered test.')
    metrics, checks = {}, {}
    for chip_seed in range(seed, seed + chips):
        # Draw each chip's offsets once so they stay fixed throughout its tests.
        offsets = np.random.default_rng(chip_seed).normal(mean, sigma, N_COMP)
        print(f'Chip seed={chip_seed}; physical offsets (mV)='
              f'{np.array2string(offsets * 1e3, precision=3)}')
        dut, _ = make_flash_adc(
            netlist_name=f'flash_chip_{chip_seed}.cir', with_specs=False,
            comparator_offsets=offsets)
        _make_characterization(dut)
        dut.simulator = 'behavioral'
        reports = dut.run_all_tests(n_workers=1)
        for test_name, report in reports.items():
            if not hasattr(report, 'metrics'):
                raise SimulationError(f'Chip seed={chip_seed}, {test_name}: {report}')
            for name, value in report.metrics.items():
                metrics.setdefault((test_name, name), []).append((chip_seed, float(value)))
            for result in report.results:
                checks.setdefault((test_name, result.name), []).append(result)
        del reports, dut
        gc.collect()
    summary = {}
    for (test_name, name), rows in metrics.items():
        values = np.asarray([v for _, v in rows])
        specs = [r for (test, _), rs in checks.items() if test == test_name
                 for r in rs if r.metric_name == name]
        # All metrics registered by this example have one-sided specs.
        lower = specs and specs[0].op in ('>', '>=')
        worst = int(np.argmin(values) if lower else np.argmax(values))
        stats = dict(mean=float(values.mean()),
                     sigma=float(values.std(ddof=1)) if chips > 1 else float('nan'),
                     worst=float(values[worst]), worst_seed=rows[worst][0])
        summary[(test_name, name)] = stats
        print(f'{test_name}.{name}: mean={stats["mean"]:.6g}, '
              f'sigma={stats["sigma"]:.6g}, worst={stats["worst"]:.6g}, '
              f'worst chip seed={stats["worst_seed"]}')
    for (test_name, name), rows in checks.items():
        print(f'{test_name}.{name}: spec pass fraction='
              f'{sum(r.passed for r in rows)}/{chips}')
    print('Re-run a worst chip preserving --offset-mc N, --mc-seed, --bits, '
          '--strongarm and --sine settings, with '
          '--chips 1 --seed <worst chip seed>.')
    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  Main — behavioral simulation + plot + SPICE netlist preview
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    _parser = argparse.ArgumentParser(
        description='4-bit flash ADC — behavioral + sky130 SPICE.')
    _parser.add_argument(
        '--strongarm', action='store_true',
        help='Use the clocked StrongArm-latch comparator instead of the '
             'default continuous-time OTA comparator.')
    _parser.add_argument(
        '--no-spice', action='store_true',
        help='Skip the ngspice run (behavioral only).')
    _parser.add_argument(
        '--sine', action='store_true',
        help='Drive Vin with a coherent-sampling sine instead of a ramp, '
             'and compute T/H and ADC SNDR. Extends T_STOP to '
             f'{SINE_T_STOP*1e6:.2f} µs for {SINE_N_FFT} samples.')
    _parser.add_argument(
        '--bits', type=int, default=N_BITS, metavar='N',
        help=f'ADC resolution in bits (default {N_BITS}). Flash has 2**N-1 '
             'comparators. Higher resolution increases circuit size and, '
             'for ramp runs, simulation duration.')
    _parser.add_argument('--offset-mc', type=int, default=0, metavar='N',
                         help='Characterize one comparator with N SPICE mismatch '
                              'samples, then run the user-level behavioral chip loop.')
    _parser.add_argument('--chips', type=int, default=16, metavar='K')
    _parser.add_argument('--seed', type=int, default=468, help='First chip seed.')
    _parser.add_argument('--mc-seed', type=int, default=468, help='SPICE MC seed.')
    _args = _parser.parse_args()
    if _args.bits < 2:
        _parser.error('--bits must be at least 2.')
    if _args.offset_mc and (_args.offset_mc < 2 or _args.no_spice):
        _parser.error('--offset-mc needs N >= 2 and SPICE; omit --no-spice.')
    if _args.chips < 1 or min(_args.seed, _args.mc_seed) < 0:
        _parser.error('--chips must be >= 1 and seeds must be nonnegative.')
    USE_STRONGARM = _args.strongarm
    USE_SINE = _args.sine
    if _args.bits != N_BITS:
        N_BITS = _args.bits
        N_COMP = 2 ** N_BITS - 1
        VSTEP = (VREFH - VREFL) / (N_COMP + 1)
    # Sine mode needs a long-enough window for a clean FFT; override T_STOP.
    if USE_SINE:
        T_STOP = SINE_T_STOP
    # For ramp mode, give at least one cycle per ideal code plus a bit of
    # margin so we see the full staircase regardless of resolution.
    elif _args.bits != 4:
        T_STOP = (N_COMP + 1) * CLK_PER

    if _args.offset_mc:
        mean, sigma = characterize_offset(samples=_args.offset_mc,
                                          seed=_args.mc_seed,
                                          strongarm=USE_STRONGARM)
        run_offset_chips(mean=mean, sigma=sigma, chips=_args.chips, seed=_args.seed)
        sys.exit(0)

    _stimulus = (f'sine @ {SINE_FREQ/1e6:.3f} MHz, {SINE_N_FFT} samples'
                 if USE_SINE else
                 f'{VREFL:.2f}→{VREFH:.2f} V ramp')
    _comp = 'StrongArm latch' if USE_STRONGARM else 'OTA'
    print(f'Building {N_BITS}-bit flash ADC '
          f'({N_COMP} comparators, {_comp}, {_stimulus})...')
    _comp_tag = 'strongarm' if USE_STRONGARM else 'ota'
    _stim_tag = 'sine' if USE_SINE else 'ramp'
    _netlist_name = f'flash_adc_{N_BITS}bit_{_comp_tag}_{_stim_tag}.cir'
    dut, out_dir = make_flash_adc(netlist_name=_netlist_name, with_specs=False)
    adc_test = _make_characterization(dut)
    adc = adc_test.netlist
    adc.netlistname = _netlist_name

    # ── Behavioral simulation ─────────────────────────────────────────────────
    print(f"Running behavioral simulation  "
          f"(t_stop={T_STOP*1e9:.0f} ns, t_step={T_STEP*1e12:.0f} ps, "
          f"requested grid)...")
    adc.simulate_behavioral(t_step=T_STEP, t_stop=T_STOP)
    print(f"  Loaded {len(adc.load_signal('vin').x)} behavioral points.")

    # Junction overrange check — verify no port voltages exceed PDK limits
    junction_violations = check_junction_overrange(adc)
    if junction_violations:
        print(f"\nJunction overrange: {len(junction_violations)} violation(s)")
        for v in junction_violations:
            print(f"  {v.message}")
    else:
        print("Junction check: PASS (no overrange detected)")

    t    = adc.load_signal('vin').x
    t_ns = t * 1e9

    # Load analog signals
    vin = adc.load_signal('vin').y
    vh  = adc.load_signal('vh').y

    # Keep the individual thermometer waveforms for plotting.
    therm = np.stack([adc.load_signal(f'vcmp{k}').y
                      for k in range(1, N_COMP + 1)])
    # Decode the comparator thermometer traces on the shared simulation grid.
    code = _code_trace(adc).y.astype(int)

    # Load a few comparator outputs for the waterfall panel. Pick three
    # indices that straddle the middle of the ladder so the panels stay
    # useful at any bit count.
    _vcmp_picks = (max(1, N_COMP // 4),
                   max(1, N_COMP // 2),
                   min(N_COMP, (3 * N_COMP) // 4))
    vcmp_mid = {k: adc.load_signal(f'vcmp{k}').y for k in _vcmp_picks}

    # Reference levels (constant; just read t=0 sample)
    ref_levels = [adc.load_signal(f'vref{k}').y[0] for k in range(1, N_COMP + 1)]

    # ── Registered measurements from the behavioral evidence ─────────────────
    behavioral_report = adc_test.evaluate()
    if USE_SINE:
        spectral = dict(tclk=CLK_PER, tstart=SAMPLE_S_IN_CYCLE, n_samples=SINE_N_FFT)
        th_sndr_beh = behavioral_report.metrics['th_sndr']
        adc_sndr_beh = behavioral_report.metrics['adc_sndr']
        th_spectrum_beh = fft(adc.load_signal('vh'), **spectral).y
        vadc = VREFL + (_code_trace(adc) + 0.5) * VSTEP
        adc_spectrum_beh = fft(vadc, **spectral).y
        code_sampled = sample(_code_trace(adc), x_new=(
            np.arange(SINE_N_FFT) * CLK_PER + SAMPLE_S_IN_CYCLE)).y

    reconcile_report = adc_test.module.reconcile_specs()
    print('\n── Spec Reconciliation ───────────────────────────────────────')
    print(reconcile_report.summary())

    # Show the ADC's circuit structure while retaining simulation and junction
    # evidence from the executing testbench (including its hierarchy paths).
    circuit_report = generate_report(dut)
    report = generate_report(adc, spec_reports=[behavioral_report])
    report.topology = circuit_report.topology
    report.annotation = circuit_report.annotation
    report.spec_reconciliation = reconcile_report
    report_path = os.path.join(out_dir, 'flash_adc_report.md')
    report.save(report_path)
    json_path = os.path.join(out_dir, 'flash_adc_report.json')
    report.save_json(json_path)
    print(f"\n── Circuit Report ──────────────────────────────────────────")
    report.print_summary()
    print(f"Report (MD)   saved → {report_path}")
    print(f"Report (JSON) saved → {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    fig.suptitle(
        f'{N_BITS}-Bit Flash ADC — Behavioral Simulation\n'
        'sky130 NMOS/PMOS topology + Python behavioral models',
        fontsize=13, fontweight='bold'
    )

    # Panel 1: analog input and track-and-hold output
    ax = axes[0]
    ax.plot(t_ns, vin, lw=1.5, color='steelblue',
            label=f'$V_{{in}}$  ({_stimulus})')
    ax.plot(t_ns, vh,  lw=1.5, color='darkorange',
            label='$V_h$  (track & hold output)')
    for lvl in ref_levels:
        ax.axhline(lvl, color='#cccccc', lw=0.5, ls='--', zorder=0)
    ax.set_ylabel('Voltage (V)')
    ax.set_ylim(VREFL - V_MARGIN, VREFH + V_MARGIN)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title(f'1. Track-and-Hold — samples the {_stim_tag} at each clock edge',
                 fontsize=9)
    ax.grid(True, alpha=0.25)

    # Panel 2: reference ladder taps + track-and-hold overlay
    ax = axes[1]
    cmap = plt.cm.RdYlGn
    for k, lvl in enumerate(ref_levels, 1):
        ax.axhline(lvl, color=cmap((k - 1) / (N_COMP - 1)), lw=0.9, alpha=0.8)
    ax.plot(t_ns, vh, lw=2, color='darkorange', label='$V_h$', zorder=5)
    # Annotate a few taps (top, bottom, and three interior points)
    _tap_annotate = sorted({1, max(1, N_COMP // 4),
                            max(1, N_COMP // 2),
                            min(N_COMP, (3 * N_COMP) // 4),
                            N_COMP})
    for k in _tap_annotate:
        ax.text(t_ns[-1] * 1.005, ref_levels[k - 1],
                f'vref{k}={ref_levels[k-1]:.3f} V',
                va='center', ha='left', fontsize=7, color='dimgray')
    ax.set_ylabel('Voltage (V)')
    ax.set_ylim(VREFL - V_MARGIN, VREFH + V_MARGIN)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title(f'2. Reference Ladder — {N_COMP} equally-spaced thresholds '
                 f'({VSTEP*1e3:.1f} mV each)', fontsize=9)
    ax.grid(True, alpha=0.25)

    # Panel 3: comparator outputs (waterfall — select 3 representative ones)
    ax = axes[2]
    offset_scale = 0.25
    _waterfall_colors = ('#d62728', '#2ca02c', '#1f77b4')
    for idx, (k, arr) in enumerate(vcmp_mid.items()):
        offset = idx * offset_scale
        ax.step(t_ns, arr / VDD + offset, where='post',
                color=_waterfall_colors[idx % len(_waterfall_colors)],
                label=f'vcmp{k}  (ref={ref_levels[k-1]:.3f} V)')
    ax.set_ylabel('Normalised output\n(+ offset for clarity)')
    ax.set_ylim(-0.5, 1.8)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title(
        f'3. Comparator Outputs — '
        f'{" / ".join(f"vcmp{k}" for k in vcmp_mid)} '
        f'(sample of {N_COMP})', fontsize=9)
    ax.grid(True, alpha=0.25)

    # Panel 4: 4-bit integer ADC code
    ax = axes[3]
    ax.step(t_ns, code, where='post', color='black', lw=2)
    ax.fill_between(t_ns, code, step='post', alpha=0.12, color='black')
    # Annotate ideal staircase alignment
    for step_idx in range(N_COMP + 2):
        ax.axhline(step_idx, color='#dddddd', lw=0.4, ls=':')
    ax.set_ylabel(f'ADC Code  (0 – {N_COMP})')
    ax.set_xlabel('Time (ns)')
    ax.set_ylim(-0.5, N_COMP + 1.5)
    # Tick every 1 at low bit counts, every 2**(N_BITS-4) at higher to stay readable
    _tick_step = max(1, 2 ** (N_BITS - 4))
    ax.set_yticks(range(0, N_COMP + 2, _tick_step))
    ax.set_title(f'4. {N_BITS}-Bit Output Code — staircase confirms correct flash conversion',
                 fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    # Save figures in the configured example output directory.
    png_path = os.path.join(out_dir, f'flash_adc_{N_BITS}bit.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved → {png_path}")
    # Embed plot in HTML report
    report.add_figure(fig, title=f'{N_BITS}-Bit Flash ADC — Behavioral Simulation')
    html_path = os.path.join(out_dir, 'flash_adc_report.html')
    report.save_html(html_path)
    print(f"Report (HTML) saved → {html_path}")
    plt.show()

    # ── Print ADC transfer table ───────────────────────────────────────────────
    print('\n── ADC Transfer Function (sampled late in evaluate) ─────────────────')
    print(f'{"Cycle":>5}  {"Time (ns)":>10}  {"Vin (V)":>8}  '
          f'{"Vh (V)":>8}  {"Code":>5}  {"Binary":>6}')
    print('─' * 54)
    _n_cycles = min(N_COMP + 1, int((t[-1] - SAMPLE_S_IN_CYCLE) / CLK_PER) + 1)
    beh_sample_idx = _sample_indices_at_phase(t, _n_cycles)
    for cycle in range(_n_cycles):
        idx = int(beh_sample_idx[cycle])
        t_mid_ns = t[idx] * 1e9
        vin_v = vin[idx]
        vh_v  = vh[idx]
        code_v = code[idx]
        bits_str = format(int(code_v), f'0{N_BITS}b')
        print(f'{cycle + 1:>5}  {t_mid_ns:>10.1f}  {vin_v:>8.4f}  '
              f'{vh_v:>8.4f}  {int(code_v):>5}  {bits_str:>6}')

    # ── SPICE simulation ──────────────────────────────────────────────────────
    if _args.no_spice:
        print('\n── SPICE skipped (--no-spice) ─────────────────────────────────')
        netlist_path = os.path.join(adc.dir_folder, adc.netlistname)
        print(f'Full netlist: {netlist_path}')
        sys.exit(0)

    print('\n── Running SPICE simulation (ngspice, sky130 tt) ───────────────────')
    print(f'  {N_COMP} comparators, t_stop={T_STOP*1e9:.0f} ns — '
          f'this may take a few minutes.')
    # StrongArm + regenerative latch: tighter tolerances + max time step
    # so the simulator doesn't stride past the reset→eval edge or settle
    # into a metastable equilibrium with symmetric inputs.
    if USE_STRONGARM:
        adc.set_simulator_options(preset='tight')
        adc.add_sim_type(Transient(step=T_STEP, stop=T_STOP,
                                   maxstep=20e-12, uic=True))
        # Pre-charge the reference-ladder taps to their DC values so the
        # 500 fF decoupling caps don't start at 0 V and drag the latch
        # decision on the first few cycles. (uic=True skips DC op.)
        adc.add_custom_line(f'.ic v(vref)={VREFH} v(vrefl)={VREFL} '
                            f'v(vdd)={VDD}')
        for k in range(1, N_COMP + 1):
            tap = VREFL + (VREFH - VREFL) * k / (N_COMP + 1)
            adc.add_custom_line(f'.ic v(vref{k})={tap:.5f}')
    else:
        adc.add_sim_type(Transient(step=T_STEP, stop=T_STOP))
    adc.simulator = 'ngspice'
    adc.simulate(corner='tt')
    spice_report = adc_test.evaluate()
    print('  SPICE run complete.')

    t_sp    = adc.load_signal('vin').x
    t_sp_ns = t_sp * 1e9
    vin_sp  = adc.load_signal('vin').y
    vh_sp   = adc.load_signal('vh').y
    therm_sp = np.stack([adc.load_signal(f'vcmp{k}').y
                         for k in range(1, N_COMP + 1)])
    code_sp = _code_trace(adc).y.astype(int)

    # Refresh every path with the real SPICE measurements.
    reconcile_report = adc_test.module.reconcile_specs()

    # ── SPICE SNDR ────────────────────────────────────────────────────────────
    if USE_SINE:
        code_samples_sp = sample(_code_trace(adc), x_new=(
            np.arange(SINE_N_FFT) * CLK_PER + SAMPLE_S_IN_CYCLE)).y
        th_sndr_sp = spice_report.metrics['th_sndr']
        adc_sndr_sp = spice_report.metrics['adc_sndr']
        th_spectrum_sp = fft(adc.load_signal('vh'), **spectral).y
        vadc_sp = VREFL + (_code_trace(adc) + 0.5) * VSTEP
        adc_spectrum_sp = fft(vadc_sp, **spectral).y

        print('\n── SNDR Analysis ───────────────────────────────────────────')
        print(f'  Fundamental bin (coherent):  {SINE_M_COH}/{SINE_N_FFT} '
              f'→ {SINE_FREQ/1e6:.3f} MHz')
        print(f'  T/H SNDR:  behavioral = {th_sndr_beh:6.2f} dB   '
              f'SPICE = {th_sndr_sp:6.2f} dB')
        print(f'  ADC SNDR:  behavioral = {adc_sndr_beh:6.2f} dB   '
              f'SPICE = {adc_sndr_sp:6.2f} dB   '
              f'(ideal {N_BITS}-bit ≈ {6.02 * N_BITS + 1.76:.2f} dB)')

    # ── Comparison plot: behavioral vs SPICE ──────────────────────────────────
    fig2, axes2 = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
    fig2.suptitle(f'{N_BITS}-Bit Flash ADC — Behavioral vs SPICE',
                  fontsize=13, fontweight='bold')

    ax = axes2[0]
    ax.plot(t_ns,    vh,    color='darkorange', lw=1.5, label='behavioral')
    ax.plot(t_sp_ns, vh_sp, color='steelblue',  lw=1.0, ls='--', label='SPICE')
    ax.set_ylabel('$V_h$ (V)')
    ax.set_title('Track-and-Hold output', fontsize=9)
    ax.legend(loc='upper left', fontsize=9); ax.grid(True, alpha=0.25)

    for idx, k in enumerate(_vcmp_picks, start=1):
        ax = axes2[idx]
        ax.step(t_ns,    vcmp_mid[k],     where='post',
                color='darkorange', lw=1.2, label='vcmp (behavioral)')
        ax.step(t_sp_ns, therm_sp[k - 1], where='post',
                color='steelblue',  lw=1.0, ls='--', label='vcmp (SPICE)')
        ax.set_ylabel(f'vcmp{k}\n(V)')
        ax.set_title(f'Comparator {k} (ref={ref_levels[k-1]:.3f} V)',
                     fontsize=9)
        ax.grid(True, alpha=0.25)

        # These internal nodes are explicitly saved by the conversion bench.
        # A missing observation must propagate its typed readback error.
        if USE_STRONGARM:
            vtail_sig = adc.load_signal(f'xdut.xcomp{k}.vtail')
            di_p_sig  = adc.load_signal(f'xdut.xcomp{k}.di_p')
            di_n_sig  = adc.load_signal(f'xdut.xcomp{k}.di_n')
            ax2 = ax.twinx()
            ax2.plot(vtail_sig.x * 1e9, vtail_sig.y,
                     color='seagreen', lw=0.8, alpha=0.7,
                     label='vtail')
            ax2.plot(di_p_sig.x * 1e9, di_p_sig.y,
                     color='purple',   lw=0.8, alpha=0.7, ls=':',
                     label='di_p (drain of input pair p-side)')
            ax2.plot(di_n_sig.x * 1e9, di_n_sig.y,
                     color='crimson',  lw=0.8, alpha=0.7, ls='-.',
                     label='di_n (drain of input pair n-side)')
            # Probe the reference tap itself — is the ladder impedance
            # letting the StrongArm kickback disturb vref_k during eval?
            vref_k_sig = adc.load_signal(f'vref{k}')
            ax2.plot(vref_k_sig.x * 1e9, vref_k_sig.y,
                     color='darkblue', lw=0.8, alpha=0.9,
                     label=f'vref{k} (ladder tap)')
            ax2.set_ylabel('internal (V)', fontsize=8)
            ax2.tick_params(axis='y', labelsize=7)
            # Combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2,
                      loc='upper left', fontsize=7, ncol=2)
        else:
            ax.legend(loc='upper left', fontsize=9)

    # Compare reconstructed ADC codes.
    # In --sine mode, plot the one-sample-per-cycle discrete series that
    # actually feeds the FFT (rather than the continuous raster). In ramp
    # mode, keep the continuous step plot.
    ax = axes2[4]
    if USE_SINE:
        sample_times_ns = (np.arange(SINE_N_FFT) * CLK_PER
                           + SAMPLE_S_IN_CYCLE) * 1e9
        ax.plot(sample_times_ns, code_sampled, 'o-',
                color='darkorange', lw=1.0, ms=4, label='behavioral (sampled)')
        ax.plot(sample_times_ns, code_samples_sp, 's--',
                color='steelblue',  lw=1.0, ms=4, label='SPICE (sampled)')
        ax.set_title(f'Sampled {N_BITS}-bit code — one sample per cycle, '
                     f'the same series fed to the FFT', fontsize=9)
    else:
        ax.step(t_ns,    code,    where='post',
                color='darkorange', lw=1.5, label='behavioral')
        ax.step(t_sp_ns, code_sp, where='post',
                color='steelblue',  lw=1.0, ls='--', label='SPICE')
        ax.set_title(f'Reconstructed {N_BITS}-bit code (thermometer → integer)',
                     fontsize=9)
    ax.set_ylabel(f'ADC code\n(0–{N_COMP})')
    ax.set_ylim(-0.5, N_COMP + 1)
    ax.set_yticks(range(0, N_COMP + 2, max(1, 2 ** (N_BITS - 4))))
    ax.legend(loc='upper left', fontsize=9); ax.grid(True, alpha=0.25)

    axes2[-1].set_xlabel('Time (ns)')
    plt.tight_layout()
    png2 = os.path.join(out_dir, f'flash_adc_{N_BITS}bit_spice.png')
    plt.savefig(png2, dpi=150, bbox_inches='tight')
    print(f'SPICE comparison plot saved → {png2}')
    report.add_figure(fig2, title='Behavioral vs SPICE — Track/Hold + Comparators')

    # ── All N_COMP comparator outputs in one window (debug) ───────────────────
    fig_all, axes_all = plt.subplots(N_COMP, 1, figsize=(13, 1.0 * N_COMP),
                                     sharex=True)
    fig_all.suptitle(f'All {N_COMP} comparator outputs — behavioral vs SPICE',
                     fontsize=13, fontweight='bold')
    for k in range(1, N_COMP + 1):
        ax = axes_all[N_COMP - k]   # comp15 on top, comp1 on bottom
        ax.step(t_ns,    therm[k - 1],    where='post',
                color='darkorange', lw=1.0, alpha=0.9,
                label='behavioral' if k == N_COMP else None)
        ax.step(t_sp_ns, therm_sp[k - 1], where='post',
                color='steelblue',  lw=0.8, ls='--',
                label='SPICE' if k == N_COMP else None)
        ax.axhline(VDD / 2, color='lightgray', lw=0.3, ls=':', zorder=0)
        ax.set_ylabel(f'vcmp{k}\nref={ref_levels[k-1]:.3f}V',
                      fontsize=7, rotation=0, ha='right', va='center')
        ax.set_ylim(-0.15, VDD + 0.15)
        ax.set_yticks([])
        ax.grid(True, alpha=0.15, axis='x')
        if k == N_COMP:
            ax.legend(loc='upper right', fontsize=8, ncol=2)
    axes_all[-1].set_xlabel('Time (ns)')
    plt.tight_layout()
    png_all = os.path.join(out_dir, f'flash_adc_{N_BITS}bit_all_comps.png')
    plt.savefig(png_all, dpi=150, bbox_inches='tight')
    print(f'All-comparators plot saved → {png_all}')
    report.add_figure(fig_all, title=f'All {N_COMP} comparator outputs — '
                      f'behavioral vs SPICE')

    # FFT spectra for --sine mode
    if USE_SINE:
        fig3, axes3 = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        fig3.suptitle(
            f'SNDR Spectra — sine {SINE_FREQ/1e6:.3f} MHz, '
            f'{SINE_N_FFT} samples @ {1/CLK_PER/1e6:.0f} MHz',
            fontsize=12, fontweight='bold')
        bins = np.arange(len(th_spectrum_beh))
        freq_mhz = bins / SINE_N_FFT / CLK_PER / 1e6

        def _plot_spectrum(ax, P_beh, P_sp, title, sndr_beh, sndr_sp):
            # Normalize to the measured fundamental.
            fund = int(np.argmax(P_beh[1:]) + 1)
            P_beh_db = 10 * np.log10(np.maximum(P_beh / P_beh[fund], 1e-12))
            P_sp_db  = 10 * np.log10(np.maximum(P_sp  / P_sp[fund],  1e-12))
            ax.stem(freq_mhz, P_beh_db, linefmt='darkorange', markerfmt=' ',
                    basefmt=' ', label=f'behavioral  SNDR={sndr_beh:.2f} dB')
            ax.stem(freq_mhz, P_sp_db,  linefmt='steelblue',  markerfmt=' ',
                    basefmt=' ', label=f'SPICE       SNDR={sndr_sp:.2f} dB')
            ax.set_ylabel('Power (dBc)')
            ax.set_title(title, fontsize=10)
            ax.set_ylim(-100, 5)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)

        _plot_spectrum(axes3[0], th_spectrum_beh, th_spectrum_sp,
                       'T/H output spectrum', th_sndr_beh, th_sndr_sp)
        _plot_spectrum(axes3[1], adc_spectrum_beh, adc_spectrum_sp,
                       'ADC-reconstructed spectrum', adc_sndr_beh, adc_sndr_sp)
        axes3[-1].set_xlabel('Frequency (MHz)')
        plt.tight_layout()
        png3 = os.path.join(out_dir, f'flash_adc_{N_BITS}bit_sndr.png')
        plt.savefig(png3, dpi=150, bbox_inches='tight')
        print(f'SNDR spectra plot saved → {png3}')
        report.add_figure(fig3, title=f'SNDR Spectra — '
                          f'T/H vs ADC (fund {SINE_FREQ/1e6:.2f} MHz)')

    figures = report.figures
    report = generate_report(adc, spec_reports=[spice_report])
    report.topology = circuit_report.topology
    report.annotation = circuit_report.annotation
    report.figures = figures
    report.spec_reconciliation = reconcile_report
    report.save(report_path)
    report.save_json(json_path)
    report.save_html(html_path)
    plt.show()

    # ── SPICE transfer table ──────────────────────────────────────────────────
    print('\n── SPICE ADC Transfer Function (sampled late in evaluate) ───────────────')
    print(f'{"Cycle":>5}  {"Time (ns)":>10}  {"Vin (V)":>8}  '
          f'{"Vh (V)":>8}  {"Code":>5}  {"Binary":>6}')
    print('─' * 54)
    _n_cycles = min(N_COMP + 1, int((t_sp[-1] - SAMPLE_S_IN_CYCLE) / CLK_PER) + 1)
    spice_sample_idx = _sample_indices_at_phase(t_sp, _n_cycles)
    for cycle in range(_n_cycles):
        idx = int(spice_sample_idx[cycle])
        t_mid_ns = t_sp_ns[idx]
        if idx >= len(t_sp_ns):
            break
        print(f'{cycle + 1:>5}  {t_mid_ns:>10.1f}  {vin_sp[idx]:>8.4f}  '
              f'{vh_sp[idx]:>8.4f}  {int(code_sp[idx]):>5}  '
              f'{format(int(code_sp[idx]), f"0{N_BITS}b"):>6}')

    netlist_path = os.path.join(adc.dir_folder, adc.netlistname)
    print(f'\nFull netlist: {netlist_path}')
    print(f'Rawfile:      {adc.results_file}')
