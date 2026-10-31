"""StrongArm latch comparator with ten registered characterization testbenches.

Four benches sweep the input threshold at different common-mode voltages.
Four measure decision time at different input overdrives. One measures offset;
one sweeps five process corners at fixed supply and temperature.
The behavioral model is a memoryless, zero-delay comparison during evaluation;
it does not model regeneration or hold a decision when the input changes.

Run .venv/bin/python -m silk.examples.strongarm_tb from the repository root.
Requires ngspice and sky130 models. Prints a pass/fail table and saves four
strongarm_*.png plots in the configured example output directory
(./silk-output by default). flash_adc_example imports the shared builders.
"""

import os

import numpy as np
import matplotlib.pyplot as plt


from silk import PortDir, PortRole, Port, VoltageDomain
from silk import load_pdk
from silk.circuit.modules import Module
from silk.metrics import crossing, propagation_delay
from silk.simulation import Transient
from silk.sources.classes import VoltageSource
from silk.sources.waveforms import PWL, Pulse
from silk.examples import example_output_dir


# ── Global constants ────────────────────────────────────────────────────────
VDD     = 1.8           # supply (V)
AVDD    = VoltageDomain('avdd', nominal=VDD, tolerance=0.1)
OUT_DIR = example_output_dir(create=False)

CLK_PER = 50e-9         # 20 MHz clock (same as flash ADC default)
T_STEP  = 0.1e-9        # 100 ps default sim step


# ── Testbench-specific constants ────────────────────────────────────────────
VM_VALUES = (0.85, 1.00, 1.15, 1.30)              # threshold sweep
OVERDRIVE_VALUES = (1e-3, 5e-3, 20e-3, 100e-3)    # decision-time sweep (V)

VP_HALF_RANGE   = 50e-3                           # ±50 mV around vm
CYCLES_TOTAL    = int(round(2 * VP_HALF_RANGE / 1e-3))  # 100 cycles at 1 mV/cycle
T_STOP          = CYCLES_TOTAL * CLK_PER          # 5 µs

# Slow ramp for the offset TB: 1 mV / µs through vm, ±100 mV.
OFFSET_RAMP_V_PER_S = 1e-3 / 1e-6                 # 1 mV/µs
OFFSET_HALF_RANGE   = 100e-3                      # ±100 mV around vm
T_STOP_OFFSET       = 2 * OFFSET_HALF_RANGE / OFFSET_RAMP_V_PER_S   # 200 µs


# ── PDK / module factory ────────────────────────────────────────────────────
def _pdk():
    return load_pdk('sky130')


def _module(name, pdk, out_dir):
    """Return a blank Module configured as a named subcircuit."""
    return Module('PDK_skywater130', out_dir, f'{name}.cir',
                  pdk_details=pdk, module_name=name)


# ══════════════════════════════════════════════════════════════════════════════
#  Comparator and clock-delay builders
# ══════════════════════════════════════════════════════════════════════════════
# Shared comparator and clock-delay builders used by flash_adc_example.
# ══════════════════════════════════════════════════════════════════════════════

def make_comparator_strongarm(pdk, out_dir):
    """
    Clocked regenerative comparator with balanced latch loads.
    The circuit has no static bias branch. Device mismatch is not included
    in its nominal behavioral model.

    SPICE topology (clk = clkb internally so we evaluate on T/H hold)
    ────────────────────────────────────────────────────────────────
      Reset (clkb LOW): PMOS reset switches pull outp/outn to VDD,
      tail is off, no static current.
      Evaluate (clkb HIGH): tail turns on, diff pair pulls di_p/di_n
      toward vtail at rates set by vp vs vm, cross-coupled latch
      regenerates to rails.

        vdd ── MP_rst_p(g=clkb) ── outp     vdd ── MP_rst_n(g=clkb) ── outn
        vdd ── MP_load_p(g=outn) ─ outp     vdd ── MP_load_n(g=outp) ─ outn
                                  outp                              outn
                                  │                                   │
                        MN_inv_p(g=outn)                   MN_inv_n(g=outp)
                                  │                                   │
                                 di_p                               di_n
                                  │                                   │
                           MN_in_p(g=vp)                     MN_in_n(g=vm)
                                  │                                   │
                                  └─────── vtail ────────────────────┘
                                                  │
                                          MN_tail(g=clkb)
                                                  │
                                                 vss

      Each internal latch output (outp, outn) drives a matching output
      inverter. One inverter produces the external ``vout`` (real load);
      the other drives a FLOATING node (``outn_float``) and exists only
      to balance the capacitive load on the latch so regeneration is
      symmetric. vout = NOT(outp), so vout is HIGH when vp > vm (matches
      behavioral polarity). During reset outp=outn=VDD so both inverters
      pull their outputs to 0 — i.e. vout = 0 during reset, not VDD.

    Behavioral model
    ────────────────
      During evaluate (clkb > 0.5·VDD): vout = VDD if (vp + offset) > vm else 0.
      During reset    (clkb <= 0.5·VDD): vout = 0   (output inverter sees
      outp=VDD, so external vout is low during reset).
    """
    comp = _module('comparator_sa', pdk, out_dir)

    # Input diff pair (NMOS)
    mn_in_p = pdk.nmos1p8('mn_in_p', d='di_p',  g='vp',    s='vtail', b='vss',
                          w=2.0, l=0.18, m=4)
    mn_in_n = pdk.nmos1p8('mn_in_n', d='di_n',  g='vm',    s='vtail', b='vss',
                          w=2.0, l=0.18, m=4)
    # Tail NMOS — clocked, not biased. Dynamic.
    mn_tail = pdk.nmos1p8('mn_tail', d='vtail', g='clkb',  s='vss',   b='vss',
                          w=4.0, l=0.18, m=2)
    # Cross-coupled regenerative latch — both outputs are internal nodes
    # (``outp`` and ``outn``). External ``vout`` is driven by an output-
    # buffer inverter on the ``outp`` side (below); a matching dummy
    # inverter on the ``outn`` side balances the capacitive load on the
    # latch so regeneration is symmetric.
    mn_inv_p = pdk.nmos1p8('mn_inv_p', d='outp', g='outn', s='di_p', b='vss',
                           w=2.0, l=0.18)
    mn_inv_n = pdk.nmos1p8('mn_inv_n', d='outn', g='outp', s='di_n', b='vss',
                           w=2.0, l=0.18)
    mp_load_p = pdk.pmos1p8('mp_load_p', d='outp', g='outn', s='vdd', b='vdd',
                            w=4.0, l=0.18)
    mp_load_n = pdk.pmos1p8('mp_load_n', d='outn', g='outp', s='vdd', b='vdd',
                            w=4.0, l=0.18)
    # PMOS reset switches (gate = clkb, ON when clkb low = T/H track phase)
    mp_rst_p = pdk.pmos1p8('mp_rst_p', d='outp', g='clkb', s='vdd', b='vdd',
                           w=1.0, l=0.18)
    mp_rst_n = pdk.pmos1p8('mp_rst_n', d='outn', g='clkb', s='vdd', b='vdd',
                           w=1.0, l=0.18)

    # Output stage: one real output inverter plus a dummy inverter that
    # matches the capacitive load on the opposite latch output.
    #   real:  outp → vout          (external port)
    #   dummy: outn → outn_float    (floats, unused; load-matching only)
    mp_buf_real = pdk.pmos1p8('mp_buf_real', d='vout', g='outp',
                              s='vdd', b='vdd', w=2.0, l=0.18)
    mn_buf_real = pdk.nmos1p8('mn_buf_real', d='vout', g='outp',
                              s='vss', b='vss', w=1.0, l=0.18)
    mp_buf_dmy  = pdk.pmos1p8('mp_buf_dmy',  d='outn_float', g='outn',
                              s='vdd', b='vdd', w=2.0, l=0.18)
    mn_buf_dmy  = pdk.nmos1p8('mn_buf_dmy',  d='outn_float', g='outn',
                              s='vss', b='vss', w=1.0, l=0.18)

    comp.add(mn_in_p, mn_in_n, mn_tail, mn_inv_p, mn_inv_n,
             mp_load_p, mp_load_n, mp_rst_p, mp_rst_n,
             mp_buf_real, mn_buf_real, mp_buf_dmy, mn_buf_dmy)

    comp.define_module_ports({
        'vp':   Port(dir=PortDir.IN,  role=PortRole.ANALOG,  domain=AVDD),
        'vm':   Port(dir=PortDir.IN,  role=PortRole.ANALOG,  domain=AVDD),
        'clkb': Port(dir=PortDir.IN,  role=PortRole.CLOCK,   domain=AVDD),
        'vdd':  Port(dir=PortDir.IN,  role=PortRole.POWER,   domain=AVDD),
        'vss':  Port(dir=PortDir.IN,  role=PortRole.GROUND,  domain=AVDD),
        'vout': Port(dir=PortDir.OUT, role=PortRole.DIGITAL, domain=AVDD),
    })

    def _strongarm_model(ports, t, params):
        offset = params['offset_voltage']
        clkb = ports.clkb.v
        vp = ports.vp.v
        vm = ports.vm.v
        evaluate = clkb > 0.5 * VDD
        decided = np.where(vp + offset > vm, VDD, 0.0)
        # Output-inverter topology: during reset outp=VDD → vout=0;
        # during evaluate vout follows the instantaneous comparison.
        ports.vout.v = np.where(evaluate, decided, 0.0)

    comp.set_behavioral_model(_strongarm_model, params={'offset_voltage': 0.0},
                              execution='vector')
    return comp


def make_clock_delay(pdk, out_dir):
    """2-inverter chain. Gives the T/H time to settle before the
    StrongArm strobes — without this, the latch regenerates on
    kick-back transients.

    SPICE topology
    ──────────────
      clkb ── INV1 ── clkb_n ── INV2 ── clkb_d
      Each inverter uses W_p=2 µm, W_n=1 µm and L=0.18 µm.
      The first uses m=1 and the second uses m=6 to drive the clock load.

    Behavioral model
    ────────────────
      Output is clkb shifted by CLK_DELAY_S (approximates the total
      two-inverter delay).
    """
    CLK_DELAY_S = 120e-12  # Behavioral approximation for the two-inverter delay.

    delay = _module('clk_delay', pdk, out_dir)

    # INV1: clkb → clkb_n. Small — only drives INV2's input.
    mp_inv1 = pdk.pmos1p8('mp_inv1', d='clkb_n', g='clkb',   s='vdd', b='vdd',
                          w=2.0, l=0.18, m=1)
    mn_inv1 = pdk.nmos1p8('mn_inv1', d='clkb_n', g='clkb',   s='vss', b='vss',
                          w=1.0, l=0.18, m=1)
    # The larger second inverter drives the comparator clock loads.
    # Slow reset release can hold latch nodes high and external vout low.
    mp_inv2 = pdk.pmos1p8('mp_inv2', d='clkb_d', g='clkb_n', s='vdd', b='vdd',
                          w=2.0, l=0.18, m=6)
    mn_inv2 = pdk.nmos1p8('mn_inv2', d='clkb_d', g='clkb_n', s='vss', b='vss',
                          w=1.0, l=0.18, m=6)
    delay.add(mp_inv1, mn_inv1, mp_inv2, mn_inv2)

    delay.define_module_ports({
        'clkb':   Port(dir=PortDir.IN,  role=PortRole.CLOCK,  domain=AVDD),
        'vdd':    Port(dir=PortDir.IN,  role=PortRole.POWER,  domain=AVDD),
        'vss':    Port(dir=PortDir.IN,  role=PortRole.GROUND, domain=AVDD),
        'clkb_d': Port(dir=PortDir.OUT, role=PortRole.CLOCK,  domain=AVDD),
    })

    def _delay_model(ports, t, params):
        # Shift clkb right in time by CLK_DELAY_S. Assumes uniform t_step.
        clkb_in = np.asarray(ports.clkb.v)
        if len(t) < 2:
            ports.clkb_d.v = clkb_in
            return
        t_step = float(t[1] - t[0])
        shift = max(1, int(round(CLK_DELAY_S / t_step)))
        shift = min(shift, len(clkb_in))
        pad = np.full(shift, clkb_in[0])
        ports.clkb_d.v = np.concatenate([pad, clkb_in[:-shift]])

    delay.set_behavioral_model(_delay_model, execution='vector')
    return delay


# ══════════════════════════════════════════════════════════════════════════════
#  Shared rig (pytest-fixture-style)
# ══════════════════════════════════════════════════════════════════════════════
def add_strongarm_rig(tb, *, vdd=VDD, clk_per=CLK_PER, out_dir=OUT_DIR):
    """Apply the shared rig (supply + clock pair + clock-delay cell) to a TB.

    Every testbench factory below calls this once after wiring the DUT
    via ``tb.add_module_under_test(...)`` and before adding the TB's
    specific stimulus. The rig is shared so all four testbench categories
    see identical supply and clock generation.

    Parameters
    ----------
    tb : ModuleTestbench
        Mutated in place. The TB's internal netlist gains:
          - vdd_src    : DC supply at ``vdd``
          - clk_src    : 0 → vdd pulse, 50 % duty, ``clk_per`` period
          - clkb_src   : vdd → 0 pulse, complementary
          - clkdel1    : clock-delay submodule (clkb → clkb_d, ~120 ps)
    vdd : float
        Supply voltage. Default :data:`VDD`.
    clk_per : float
        Clock period in seconds. Default :data:`CLK_PER`.
    out_dir : str
        Output directory for the clock-delay submodule's .cir file.
    """
    clk_pw = clk_per / 2
    # Nonzero clock edge, derived from the caller's period so it scales with it
    # (100 ps at the 50 ns default = one T_STEP). A zero edge is
    # simulator-dependent — see silk.sources.Pulse.
    clk_tr = clk_per / 500
    tb.add_source(VoltageSource('vdd_src',  positive='vdd',  negative='gnd',
                                dc=vdd))
    tb.add_source(VoltageSource('clk_src',  positive='clk',  negative='gnd',
                                pulse=Pulse(v1=0, v2=vdd, td=0, tr=clk_tr, tf=clk_tr,
                                            pw=clk_pw, per=clk_per)))
    tb.add_source(VoltageSource('clkb_src', positive='clkb', negative='gnd',
                                pulse=Pulse(v1=vdd, v2=0, td=0, tr=clk_tr, tf=clk_tr,
                                            pw=clk_pw, per=clk_per)))
    tb.netlist.add_module(make_clock_delay(tb.module.pdk_details, out_dir),
                           'clkdel1',
                           nets={'clkb': 'clkb', 'vdd': 'vdd', 'vss': 'gnd',
                                 'clkb_d': 'clkb_d'})


# ══════════════════════════════════════════════════════════════════════════════
#  Metric extractors — pure functions on the simulated netlist
# ══════════════════════════════════════════════════════════════════════════════
#: Samples the ``vout`` high state must persist for before a rising edge counts
#: as a comparator decision rather than a glitch.
_THRESHOLD_PERSIST_SAMPLES = 3


def _extract_threshold(sig_vout, sig_vp, vm_dc):
    """Extract a rising decision, rejecting startup and short glitches.

    Two guards distinguish decisions from reset artifacts:

    1. Rising edges only: ``uic=True`` starts ``vout`` high; reset pulls it
       low, so that falling edge is not a decision.
    2. One clock of settling, then persistence: ignore edges in the first
       ``CLK_PER`` and require ``_THRESHOLD_PERSIST_SAMPLES`` high samples
       to reject glitches and partial regeneration.

    One settled decision per clock distinguishes reset pulses from decision
    reversals. Resolution is one clock of input ramp (1 mV for threshold
    benches, 0.05 mV for the slow offset bench).

    Parameters
    ----------
    sig_vout : Signal
        Completed simulation's comparator output voltage versus time.
    sig_vp : Signal
        Input ramp voltage on the same time grid as ``sig_vout``.
    vm_dc : float
        Unused; retained for the registered metric signature.

    Returns
    -------
    float
        Input voltage at the first persistent rising decision. Constant-low
        or constant-high decisions return the upper or lower ramp limit,
        respectively; thresholds within one decision step of an end are
        capped there too. Caps are bounds, not measured thresholds.
        Returns nan for multiple reversals, a falling-only decision change,
        nonfinite data, insufficient settled decisions, or no persistent edge.
    """
    t = np.asarray(sig_vout.x)
    vout = np.asarray(sig_vout.y)
    vp = np.asarray(sig_vp.y)
    if (len(t) < 2 or not np.all(np.isfinite(vout))
            or not np.all(np.isfinite(vp))):
        return float('nan')
    cycles = np.arange(1, int((t[-1] - t[0]) / CLK_PER))
    if len(cycles) < 2:
        return float('nan')
    high = vout > VDD / 2
    indices = np.searchsorted(t, t[0] + (cycles + 0.9) * CLK_PER)
    n = _THRESHOLD_PERSIST_SAMPLES
    decisions = np.array([high[max(0, i - n + 1):i + 1].all()
                          for i in indices])
    flips = np.flatnonzero(np.diff(decisions.astype(int)))
    if len(flips) > 1 or (len(flips) == 1 and decisions[0]):
        return float('nan')
    if not decisions.any():
        return float(vp[-1])
    if decisions.all():
        return float(vp[0])

    # Keep the original persistent rising-edge measurement for in-range data.
    step = (vp[-1] - vp[0]) * CLK_PER / (t[-1] - t[0])
    for i in np.where(np.diff(high.astype(int)) > 0)[0]:
        if t[i] < t[0] + CLK_PER:
            continue
        if i + 1 + n <= len(high) and high[i + 1:i + 1 + n].all():
            value = float(vp[i])
            if value <= vp[0] + step:
                return float(vp[0])
            if value >= vp[-1] - step:
                return float(vp[-1])
            return value
    return float('nan')


def _offset_cap_side(offset, limit):
    """Signed cap flag: -1 lower limit, +1 upper limit, 0 measured/unavailable."""
    if np.isfinite(offset) and np.isclose(abs(offset), limit, rtol=0, atol=1e-12):
        return float(np.sign(offset))
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  Testbench factories — each calls comp.create_test(name) to register on comp
# ══════════════════════════════════════════════════════════════════════════════
def make_threshold_tb(comp, *, vm_dc):
    """Threshold characterization at a fixed vm.

    Sweeps vp linearly from vm−50 mV to vm+50 mV over T_STOP (5 µs)
    while vm is held at vm_dc. The threshold is the vp value at the
    first persistent rising decision — ideally equal to vm.

    Parameters
    ----------
    comp : Module
        The comparator under test.
    vm_dc : float
        Common-mode level (V).

    Returns
    -------
    ModuleTestbench
        Registered on comp under the name ``threshold_vm{NNNN}mV``
        (4-digit millivolt formatting).
    """
    name = f'threshold_vm{int(vm_dc * 1000):04d}mV'
    tb = comp.create_test(name)
    tb.add_module_under_test({
        'vp':  'vp',  'vm':   'vm',
        'clkb': 'clkb_d',
        'vdd': 'vdd', 'vss': 'gnd', 'vout': 'vout',
    })
    add_strongarm_rig(tb)

    vp_start = vm_dc - VP_HALF_RANGE
    vp_end   = vm_dc + VP_HALF_RANGE
    tb.add_source(VoltageSource('vp_src', positive='vp', negative='gnd',
                                pwl=PWL().add(0, vp_start).add(T_STOP, vp_end)))
    tb.add_source(VoltageSource('vm_src', positive='vm', negative='gnd',
                                dc=vm_dc))
    tb.add_sim_type(Transient(step=T_STEP, stop=T_STOP,
                              maxstep=20e-12, uic=True))

    vout_probe = tb.probe('vout')
    vp_probe = tb.probe('vp')
    tb.add_metric('threshold_v', _extract_threshold, sig_vout=vout_probe,
                  sig_vp=vp_probe, vm_dc=vm_dc)
    tb.add_spec(f'threshold < vm + 10 mV',
                'threshold_v', '<', vm_dc + 0.010,
                key=f'{name}.threshold_upper')
    tb.add_spec(f'threshold > vm − 10 mV',
                'threshold_v', '>', vm_dc - 0.010,
                key=f'{name}.threshold_lower')
    return tb


def make_decision_time_tb(comp, *, overdrive):
    """Decision-time characterization at fixed vm = 1.0 V and a single overdrive.

    Stimulus: vp = vm + overdrive applied as a step at t = CLK_PER/4
    (before the first clk evaluate edge). vm held DC at 1.0 V.

    Spec ``t_decide < 1 ns`` is attached only to the 100 mV overdrive TB;
    others are recorded as informational metrics with no spec.

    Parameters
    ----------
    comp : Module
        The comparator under test.
    overdrive : float
        Differential overdrive (vp − vm) in volts.

    Returns
    -------
    ModuleTestbench
        Registered as ``decision_time_od{NNNN}mV``.
    """
    vm_dc = 1.0
    name = f'decision_time_od{int(overdrive * 1000):04d}mV'
    tb = comp.create_test(name)
    tb.add_module_under_test({
        'vp':  'vp',  'vm':   'vm',
        'clkb': 'clkb_d',
        'vdd': 'vdd', 'vss': 'gnd', 'vout': 'vout',
    })
    add_strongarm_rig(tb)

    # vp step at CLK_PER/4: starts at vm, jumps to vm+overdrive
    vp_pwl = (PWL()
              .add(0, vm_dc)
              .add(CLK_PER / 4 - 10e-12, vm_dc)
              .add(CLK_PER / 4, vm_dc + overdrive)
              .add(4 * CLK_PER, vm_dc + overdrive))
    tb.add_source(VoltageSource('vp_src', positive='vp', negative='gnd',
                                pwl=vp_pwl))
    tb.add_source(VoltageSource('vm_src', positive='vm', negative='gnd',
                                dc=vm_dc))
    tb.add_sim_type(Transient(step=T_STEP, stop=4 * CLK_PER,
                              maxstep=10e-12, uic=True))

    clkb_d_probe = tb.probe('clkb_d')
    vout_probe = tb.probe('vout')
    # From the clkb_d evaluate edge (VDD/2) to vout reaching 0.9 VDD;
    # start at the input step to skip the startup clock crossing.
    tb.add_metric('t_decide', propagation_delay, sig_in=clkb_d_probe,
                  sig_out=vout_probe, in_threshold=VDD / 2,
                  out_threshold=0.9 * VDD, start=CLK_PER / 4)
    # Spec only on the 100 mV TB.
    if abs(overdrive - 100e-3) < 1e-9:
        tb.add_spec('t_decide < 1 ns', 't_decide', '<', 1e-9)
    return tb


def make_offset_tb(comp):
    """Input-referred offset / kickback at fixed vm = 1.0 V.

    Stimulus: vp is a slow monotone PWL ramp through vm at 1 mV/µs over
    T_STOP_OFFSET (200 µs). Slow enough that kickback settles between
    cycles, exposing the static offset.

    Metric: offset_v = (vp at first decision flip) − vm.
    Spec:   |offset_v| < 5 mV.

    Returns
    -------
    ModuleTestbench
        Registered as ``offset``.
    """
    vm_dc = 1.0
    tb = comp.create_test('offset')
    tb.add_module_under_test({
        'vp':  'vp',  'vm':   'vm',
        'clkb': 'clkb_d',
        'vdd': 'vdd', 'vss': 'gnd', 'vout': 'vout',
    })
    add_strongarm_rig(tb)

    vp_start = vm_dc - OFFSET_HALF_RANGE
    vp_end   = vm_dc + OFFSET_HALF_RANGE
    tb.add_source(VoltageSource('vp_src', positive='vp', negative='gnd',
                                pwl=PWL().add(0, vp_start)
                                          .add(T_STOP_OFFSET, vp_end)))
    tb.add_source(VoltageSource('vm_src', positive='vm', negative='gnd',
                                dc=vm_dc))
    tb.add_sim_type(Transient(step=T_STEP, stop=T_STOP_OFFSET,
                              maxstep=50e-12, uic=True))

    # Save only observed voltages to keep the long offset ramp within memory.
    tb.netlist.set_save_list(['vout', 'vp', 'vm'])

    vout_probe = tb.probe('vout')
    vp_probe = tb.probe('vp')
    threshold = tb.add_metric('threshold_v', _extract_threshold,
                              sig_vout=vout_probe, sig_vp=vp_probe, vm_dc=vm_dc)
    tb.add_metric('offset_v', threshold - vm_dc)
    tb.add_metric('offset_cap_side',
                  lambda sig_vout, sig_vp: _offset_cap_side(
                      _extract_threshold(sig_vout, sig_vp, vm_dc) - vm_dc,
                      limit=OFFSET_HALF_RANGE),
                  sig_vout=vout_probe, sig_vp=vp_probe)
    tb.add_spec('offset < +5 mV', 'offset_v', '<',  0.005)
    tb.add_spec('offset > −5 mV', 'offset_v', '>', -0.005)
    return tb


def make_pvt_threshold_tb(comp, *, vm_dc=1.0):
    """Process-corner sweep of the threshold at vm = 1.0 V.

    Same stimulus + metric shape as make_threshold_tb at the single
    vm = 1.0 V point, but with ``set_corners`` declaring the corner-set on
    the TB. comp.run_all_tests() will invoke
    this across ('tt','ff','ss','sf','fs') uniformly with the
    single-corner TBs.

    Spec is looser than make_threshold_tb (15 mV vs 10 mV) because
    process extremes admit more deviation.

    Returns
    -------
    ModuleTestbench
        Registered on comp under the name ``pvt_threshold``.
    """
    tb = comp.create_test('pvt_threshold')
    tb.set_corners(['tt', 'ff', 'ss', 'sf', 'fs'])

    tb.add_module_under_test({
        'vp':  'vp',  'vm':   'vm',
        'clkb': 'clkb_d',
        'vdd': 'vdd', 'vss': 'gnd', 'vout': 'vout',
    })
    add_strongarm_rig(tb)

    vp_start = vm_dc - VP_HALF_RANGE
    vp_end   = vm_dc + VP_HALF_RANGE
    tb.add_source(VoltageSource('vp_src', positive='vp', negative='gnd',
                                pwl=PWL().add(0, vp_start).add(T_STOP, vp_end)))
    tb.add_source(VoltageSource('vm_src', positive='vm', negative='gnd',
                                dc=vm_dc))
    tb.add_sim_type(Transient(step=T_STEP, stop=T_STOP,
                              maxstep=20e-12, uic=True))

    vout_probe = tb.probe('vout')
    vp_probe = tb.probe('vp')
    tb.add_metric('threshold_v', _extract_threshold, sig_vout=vout_probe,
                  sig_vp=vp_probe, vm_dc=vm_dc)
    tb.add_spec('threshold < vm + 15 mV (process corners)',
                'threshold_v', '<', vm_dc + 0.015)
    tb.add_spec('threshold > vm − 15 mV (process corners)',
                'threshold_v', '>', vm_dc - 0.015)
    return tb


# ══════════════════════════════════════════════════════════════════════════════
#  Summary printer — handles SpecReport and CornerResults uniformly
# ══════════════════════════════════════════════════════════════════════════════
def _print_summary_table(results):
    """Print a pass/fail table for ``comp.run_all_tests()`` output.

    ``results`` is a dict keyed by ``{module_name}.{test_name}``. Each value
    is either a :class:`~silk.circuit.spec_report.SpecReport` (single-corner)
    or a :class:`~silk.circuit.spec_report.CornerResults` (process corners). Renders one
    row per spec per corner.
    """
    from silk.circuit.spec_report import SpecReport, CornerResults

    rows = []   # list of (test_label, corner, spec_label, status)
    total = passed = 0

    for full_name, result in results.items():
        # full_name = 'comparator_sa.threshold_vm0850mV'
        test_label = full_name.split('.', 1)[1] if '.' in full_name else full_name

        if isinstance(result, CornerResults):
            for corner, report in result.items():
                _row_per_spec(rows, test_label, corner, report)
        elif isinstance(result, SpecReport):
            _row_per_spec(rows, test_label, 'tt', result)
        else:
            rows.append((test_label, '?', '(unknown result type)', 'ERROR'))

    for _, _, _, status in rows:
        if status == 'PASS':
            passed += 1
        if status in ('PASS', 'FAIL'):
            total += 1

    print()
    print('StrongArm Comparator — Test Results')
    for name, report in results.items():
        if isinstance(report, SpecReport):
            side = report.metrics.get('offset_cap_side', 0)
            if side in (-1, 1):
                limit_mv = side * OFFSET_HALF_RANGE * 1e3
                print(f'  {name}: offset capped at {limit_mv:+.0f} mV sweep limit')
    print('═' * 79)
    print(f'  {"Test":<32}{"Corner":<10}{"Spec":<27}{"Status":<8}')
    print('─' * 79)
    for test_label, corner, spec_label, status in rows:
        print(f'  {test_label:<32}{corner:<10}{spec_label:<27}{status:<8}')
    print('─' * 79)
    print(f'  Total: {passed}/{total} passed')
    print('═' * 79)


def _row_per_spec(rows, test_label, corner, report):
    """Append one row per spec on the report. If no specs, append a single
    info-only row.

    ``report.results`` is a list of :class:`~silk.circuit.spec_report.SpecResult`
    dataclass instances, accessed via attribute (not dict).
    """
    spec_results = getattr(report, 'results', None) or []
    if not spec_results:
        rows.append((test_label, corner, '(info only)', '—'))
        return
    for s in spec_results:
        status = 'PASS' if s.passed else 'FAIL'
        rows.append((test_label, corner, s.name, status))


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting helpers — one per TB shape
# ══════════════════════════════════════════════════════════════════════════════
def _plot_threshold_sweep(reports, png_path):
    """4-panel figure, one panel per vm_dc value in VM_VALUES.

    Each panel overlays vp (orange), vm (blue), vout (green) over time
    in µs. Annotates the extracted threshold with a red horizontal line.

    Parameters
    ----------
    reports : dict[str, SpecReport]
        Keyed by TB name (``threshold_vm0850mV``, ...). Each report
        carries ``sim_results`` with the testbench's completed netlist.
    png_path : str
        Output PNG file path.
    """
    fig, axes = plt.subplots(len(reports), 1,
                             figsize=(13, 2.2 * len(reports)),
                             sharex=True)
    if len(reports) == 1:
        axes = [axes]

    fig.suptitle('StrongArm — Threshold sweep across common-mode levels',
                 fontsize=12, fontweight='bold')

    for ax, (tb_name, report) in zip(axes, sorted(reports.items())):
        ckt = report.sim_results
        t  = np.asarray(ckt.load_signal('vp').x) * 1e6   # µs
        vp = np.asarray(ckt.load_signal('vp').y)
        vm = np.asarray(ckt.load_signal('vm').y)
        vout = np.asarray(ckt.load_signal('vout').y)
        threshold = report.metrics.get('threshold_v', None)

        ax.plot(t, vp, color='darkorange', lw=1.5, label='vp')
        ax.plot(t, vm, color='steelblue',  lw=1.0, alpha=0.7, label='vm')
        ax.plot(t, vout, color='seagreen', lw=1.0, alpha=0.6, label='vout')
        if threshold is not None and not isinstance(threshold, str):
            ax.axhline(float(threshold), color='crimson', lw=0.8, ls='--',
                       alpha=0.7, label=f'threshold = {float(threshold):.4f} V')
        ax.set_ylabel(f'{tb_name}\n(V)', fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper right', fontsize=7)

    axes[-1].set_xlabel('Time (µs)')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot saved → {png_path}')


def _plot_decision_time(reports, png_path):
    """4-panel figure, one panel per overdrive, zoomed around the first clk edge.

    Parameters
    ----------
    reports : dict[str, SpecReport]
        Keyed by TB name (``decision_time_od0001mV``, ...).
    png_path : str
    """
    fig, axes = plt.subplots(len(reports), 1,
                             figsize=(13, 2.2 * len(reports)),
                             sharex=True)
    if len(reports) == 1:
        axes = [axes]

    fig.suptitle('StrongArm — Decision time vs overdrive',
                 fontsize=12, fontweight='bold')

    for ax, (tb_name, report) in zip(axes, sorted(reports.items())):
        ckt = report.sim_results
        t       = np.asarray(ckt.load_signal('vp').x) * 1e9   # ns
        vout    = np.asarray(ckt.load_signal('vout').y)
        clkb_d  = np.asarray(ckt.load_signal('clkb_d').y)
        t_decide = report.metrics.get('t_decide', None)

        ax.plot(t, vout, color='seagreen', lw=1.5, label='vout')
        ax.plot(t, clkb_d, color='steelblue', lw=0.8, alpha=0.5,
                label='clkb_d')
        if t_decide is not None and not isinstance(t_decide, str):
            # Skip startup, matching the evaluation edge t_decide counts from.
            t0 = crossing(ckt.load_signal('clkb_d'), VDD / 2,
                          edge='rising', start=CLK_PER / 4) * 1e9
            ax.axvline(t0, color='gray', lw=0.6, ls='--', alpha=0.5)
            ax.axvline(t0 + float(t_decide) * 1e9,
                       color='crimson', lw=0.8, ls='--', alpha=0.7,
                       label=f't_decide = {float(t_decide) * 1e12:.0f} ps')
        ax.set_ylabel(f'{tb_name}\n(V)', fontsize=8)
        ax.set_xlim(0, 2 * CLK_PER * 1e9)
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper right', fontsize=7)

    axes[-1].set_xlabel('Time (ns)')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot saved → {png_path}')


def _plot_offset_kickback(report, png_path):
    """Single-panel figure: vp − vm (mV) on x-axis, vout on y-axis.

    Annotates the first decision flip and the extracted offset value.
    """
    ckt = report.sim_results
    vp = np.asarray(ckt.load_signal('vp').y)
    vm = np.asarray(ckt.load_signal('vm').y)
    vout = np.asarray(ckt.load_signal('vout').y)
    offset_v = report.metrics.get('offset_v', None)

    fig, ax = plt.subplots(figsize=(13, 4.0))
    ax.plot((vp - vm) * 1e3, vout, color='seagreen', lw=1.2, label='vout')

    if offset_v is not None and not isinstance(offset_v, str):
        off_mv = float(offset_v) * 1e3
        ax.axvline(off_mv, color='crimson', lw=0.8, ls='--', alpha=0.7,
                   label=f'offset = {off_mv:+.2f} mV')
    ax.axvline(0.0, color='steelblue', lw=0.6, ls='--', alpha=0.4,
               label='vp = vm')
    ax.set_xlabel('vp − vm (mV)')
    ax.set_ylabel('vout (V)')
    ax.set_title('StrongArm — Input-referred offset (slow ramp)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot saved → {png_path}')


def _plot_pvt_threshold(corner_results, png_path):
    """Single-panel bar plot: (threshold − vm) per corner.

    Parameters
    ----------
    corner_results : CornerResults
        Dict-like {corner_name → SpecReport} from a ``set_corners`` testbench.
    png_path : str
    """
    vm_dc = 1.0
    corners = []
    deltas = []
    for corner, report in corner_results.items():
        threshold = report.metrics.get('threshold_v', None)
        if threshold is None or isinstance(threshold, str):
            continue
        corners.append(str(corner))
        deltas.append((float(threshold) - vm_dc) * 1e3)   # mV

    fig, ax = plt.subplots(figsize=(13, 4.0))
    colors = ['steelblue' if abs(d) < 15 else 'crimson' for d in deltas]
    ax.bar(corners, deltas, color=colors)
    ax.axhline( 15, color='crimson', lw=0.6, ls='--', alpha=0.5,
               label='spec ±15 mV')
    ax.axhline(-15, color='crimson', lw=0.6, ls='--', alpha=0.5)
    ax.axhline(0,   color='gray',    lw=0.6, ls='-',  alpha=0.4)
    ax.set_ylabel('threshold − vm (mV)')
    ax.set_xlabel('Corner')
    ax.set_title('StrongArm — Threshold deviation across process corners '
                 '(vm = 1.0 V)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.25, axis='y')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot saved → {png_path}')


# ══════════════════════════════════════════════════════════════════════════════
#  Runner — registers all 10 testbenches and drives the suite
# ══════════════════════════════════════════════════════════════════════════════
def _build_and_register_all(comp):
    """Construct and register all 10 testbenches on ``comp``. Returns nothing.

    Exposed as a function so the test suite can exercise registration
    without running the full simulation pipeline.
    """
    for vm in VM_VALUES:
        make_threshold_tb(comp, vm_dc=vm)
    for od in OVERDRIVE_VALUES:
        make_decision_time_tb(comp, overdrive=od)
    make_offset_tb(comp)
    make_pvt_threshold_tb(comp)


if __name__ == '__main__':
    example_output_dir()

    pdk  = _pdk()
    comp = make_comparator_strongarm(pdk, OUT_DIR)
    _build_and_register_all(comp)
    print(f'Registered {len(comp.list_tests())} testbenches '
          f'on {comp.module_name!r}: {sorted(comp.list_tests())}')

    results = comp.run_all_tests()
    _print_summary_table(results)

    # Threshold plot — collect the 4 vm-sweep reports
    threshold_reports = {
        name: results[f'{comp.module_name}.{name}']
        for name in (f'threshold_vm{int(v*1000):04d}mV' for v in VM_VALUES)
    }
    _plot_threshold_sweep(threshold_reports,
                          os.path.join(OUT_DIR, 'strongarm_threshold.png'))

    decision_reports = {
        name: results[f'{comp.module_name}.{name}']
        for name in (f'decision_time_od{int(o*1000):04d}mV'
                     for o in OVERDRIVE_VALUES)
    }
    _plot_decision_time(decision_reports,
                        os.path.join(OUT_DIR, 'strongarm_decision_time.png'))

    _plot_offset_kickback(
        results[f'{comp.module_name}.offset'],
        os.path.join(OUT_DIR, 'strongarm_offset.png'))

    _plot_pvt_threshold(
        results[f'{comp.module_name}.pvt_threshold'],
        os.path.join(OUT_DIR, 'strongarm_pvt.png'))

    print(f'\nAll testbenches complete. Plots saved to {OUT_DIR}.')
