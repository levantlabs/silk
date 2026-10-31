"""PRBS generators built from standard-cell flip-flops and XNOR feedback.

A pseudo-random bit sequence (PRBS) is a repeatable stream used to test serial
links. An n-stage maximal-length linear-feedback shift register (LFSR) repeats
after ``2**n - 1`` bits. ``build_prbs(order=7, form="fibonacci")`` supports
orders 7, 9, 15, 23 and 31 using the shared ``stdcells`` library.

The flip-flops reset to zero. XOR feedback would leave them stuck there;
XNOR feedback lets the sequence start from zero and excludes the all-ones
state instead. Each period contains ``2**(n-1) - 1`` ones, and the longest
run is n zeros.

The two forms wire feedback differently:

* Fibonacci: XNOR of the two taps drives stage 1; the other stages shift.
* Galois: the last stage wraps to stage 1 and feeds an XNOR between the two
  tap stages; the other stages shift.

Each form is compared with its own reference recurrence. The order-7 streams
are also checked for alignment by a cyclic shift; no such shift aligns them.

Verification at 600 MHz depends on the order:

* Orders 7 and 9: SPICE captures 260 and 1040 bits, respectively, covering two
  periods. Checks compare every captured bit with the reference and measure
  period, ones per period and longest run. The main program runs both forms
  for order 7 and Fibonacci for order 9.
* Order 15: SPICE checks 320 bits against the reference. A separate behavioral
  run captures 320 bits and compares its first 300 with the reference. The
  reference then supplies two full periods plus 15 bits for period, balance
  and run-length checks; the full period is 32767 bits.
* Orders 23 and 31: the table lists calculated periods and their duration at
  600 MHz. The main program does not simulate these orders.

The behavioral run uses one model for the whole LFSR so it can update all
stages together at each clock edge and apply the shared DFF clock-to-Q delay.
Simply wiring the existing whole-array cell models into a loop does not make
that update happen: the toggle-flop probe runs with an undelayed-feedback
warning, but does not verify toggling. A composed sequential model would need
state and explicit delays between updates.

Run ``MPLBACKEND=Agg .venv/bin/python -m silk.examples.prbs_generator_example``
from the repository root. It prints the spec results, a family summary and the
order-7 sequence comparison, then saves ``prbs_generator.png`` in the example
output directory.
"""
import os

import numpy as np

from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk.exceptions import SimulationError
from silk import load_pdk
from silk import sources
from silk.sources import prbs_bits
from silk.simulation import Transient
from silk.metrics import sample
from silk.signals import Signal

from silk.examples.stdcells import (build_inverter, build_dff, build_xor2, _delay_signal,
                                    _CLKQ_NOM_PS)
from silk.examples import example_output_dir

pdk = load_pdk('sky130')

_VDD = 1.8
F_CLK = 600e6                     # the PLL's output — clock the PRBS at the real rate
_T_CLK = 1.0 / F_CLK

# ITU-T O.150 two-tap PRBS polynomials, x^hi + x^lo + 1. All primitive, so both
# Fibonacci and Galois forms are maximal length (period 2^order - 1).
_PRBS_TAPS = {7: (7, 6), 9: (9, 5), 15: (15, 14), 23: (23, 18), 31: (31, 28)}

# Verification mode per order (see the module docstring's teaching point).
_VERIFY = {
    7:  ("spice_full", 260),      # 2 periods -> exact period detection
    9:  ("spice_full", 1040),     # 2 periods (0.85 us each)
    15: ("spice_spot", 320),      # spot-check; full-period statistics from the reference
    23: ("reference_only", 0),
    31: ("reference_only", 0),
}

# Reset / capture schedule. RSTB held low through the first couple of clock edges
# (proving async clear holds against a running clock), then released; the first
# rising edge after release is reference bit 0.
_T_RESET = 3.5e-9
_T_RELEASE = 3.6e-9

# Metric / spec names.
MATCH_METRIC = "bit_mismatches"
PERIOD_METRIC = "period_bits"
ONES_METRIC = "ones_per_period"
RUN_METRIC = "longest_run"


# --- the LFSR reference ------------------------------------

def prbs_reference(n_bits, *, order=7, form="fibonacci"):
    """The exact stream ``build_prbs(order, form)`` must produce.

    Models the hardware bit-for-bit: ``order`` stages cleared to 0, XNOR feedback,
    output = last stage sampled AFTER each clock edge. ``form`` selects Fibonacci
    (XNOR of the two taps into stage 1) or Galois (wrap ``Q_n`` to stage 1, XNOR
    it in between the tap stages). Raises ``ValueError`` on an unknown order/form.
    """
    if order not in _PRBS_TAPS:
        raise ValueError(f"unknown PRBS order {order}; have {sorted(_PRBS_TAPS)}")
    if form not in ("fibonacci", "galois"):
        raise ValueError(f"unknown PRBS form {form!r}; use 'fibonacci' or 'galois'")
    n_bits = range(n_bits).stop
    if form == "fibonacci":
        # Preserve the reference's empty result for nonpositive lengths.
        return list(prbs_bits(order, n_bits, seed=0)) if n_bits > 0 else []
    _, lo = _PRBS_TAPS[order]
    q = [0] * order
    bits = []
    for _ in range(n_bits):
        fb = q[order - 1]                         # Q_n, the shifted-out bit
        nq = [0] * order
        nq[0] = fb                                # wrap to stage 1
        for k in range(1, order):
            nq[k] = (1 - (q[k - 1] ^ fb)) if k == lo else q[k - 1]
        q = nq
        bits.append(q[order - 1])
    return bits


def prbs7_reference(n_bits, *, variant="fibonacci"):
    """Order-7 reference alias; ``variant`` selects the feedback form."""
    return prbs_reference(n_bits, order=7, form=variant)


# --- behavioral twin (a single model for the whole registered loop) -------------

def _prbs_behavioral_model(ports, t, params):
    """Update the whole LFSR on each rising CLK edge, with asynchronous reset.

    Uses the reference recurrence and delays OUT by ``params['tcq_s']``, the
    shared DFF clock-to-Q delay.
    """
    vdd = float(params["vdd"])
    order = int(params["order"])
    hi, lo = params["taps"]
    form = params["form"]
    clk = np.asarray(ports.CLK.v, dtype=float) > vdd / 2
    rstb = np.asarray(ports.RSTB.v, dtype=float) > vdd / 2
    q = [0] * order
    out_bool = np.zeros(len(t), dtype=bool)
    for i in range(len(t)):
        if not rstb[i]:
            q = [0] * order
        elif clk[i] and i > 0 and not clk[i - 1]:      # rising edge
            if form == "fibonacci":
                fb = 1 - (q[hi - 1] ^ q[lo - 1])
                q = [fb] + q[:order - 1]
            else:
                fb = q[order - 1]
                nq = [0] * order
                nq[0] = fb
                for k in range(1, order):
                    nq[k] = (1 - (q[k - 1] ^ fb)) if k == lo else q[k - 1]
                q = nq
        out_bool[i] = q[order - 1]
    out = _delay_signal(np.where(out_bool, vdd, 0.0), t, float(params["tcq_s"]))
    ports.OUT.v = out


# --- the generator --------------------------------------------------------------

def build_prbs(*, order=7, form="fibonacci", with_tests=True) -> Module:
    """``order``-stage PRBS generator, ``form`` in {fibonacci, galois}.

    The DFFs share CLK and RSTB (active-low clear). XNOR feedback lets the
    sequence start from the cleared state. Ports are CLK, RSTB, OUT, VDD and
    VSS. A single behavioral model represents the whole LFSR. ``with_tests``
    attaches a SPICE test for orders 7, 9 and 15.
    """
    if order not in _PRBS_TAPS:
        raise ValueError(f"unknown PRBS order {order}; have {sorted(_PRBS_TAPS)}")
    if form not in ("fibonacci", "galois"):
        raise ValueError(f"unknown PRBS form {form!r}; use 'fibonacci' or 'galois'")
    hi, lo = _PRBS_TAPS[order]

    mod = Module("PDK_skywater130", example_output_dir(), "prbs.cir", pdk_details=pdk,
                 module_name=f"prbs{order}_{form}")
    mod.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "CLK": Port.analog_input(), "RSTB": Port.analog_input(),
        "OUT": Port.analog_output(),
    })

    dff = build_dff(with_tests=False)
    xor2 = build_xor2(with_tests=False)         # default: 12T static, full swing
    inv = build_inverter(with_tests=False)

    def _q(i):
        return "OUT" if i == order else f"q{i}"

    if form == "fibonacci":
        # Shift chain; stage 1's D is the feedback, stage N's Q is OUT.
        prev = "fb"
        for i in range(1, order + 1):
            mod.add_module(dff, f"ff{i}", nets={
                "VDD": "VDD", "VSS": "VSS", "D": prev, "CLK": "CLK",
                "CLRB": "RSTB", "Q": _q(i), "QB": f"qb{i}"})
            prev = _q(i)
        # fb = XNOR(Q_hi, Q_lo) = NOT(XOR(Q_hi, Q_lo)).
        mod.add_module(xor2, "g_fb_xor", nets={
            "VDD": "VDD", "VSS": "VSS", "A": _q(hi), "B": _q(lo), "Y": "xnor_a"})
        mod.add_module(inv, "g_fb_inv", nets={
            "VDD": "VDD", "VSS": "VSS", "A": "xnor_a", "Y": "fb"})
    else:   # galois
        # The injected XNOR: D_{lo+1} = XNOR(Q_lo, Q_n).  Q_n = OUT (wraps to Q1).
        mod.add_module(xor2, "g_inj_xor", nets={
            "VDD": "VDD", "VSS": "VSS", "A": _q(lo), "B": "OUT", "Y": "xnor_a"})
        mod.add_module(inv, "g_inj_inv", nets={
            "VDD": "VDD", "VSS": "VSS", "A": "xnor_a", "Y": "fb"})
        for i in range(1, order + 1):
            if i == 1:
                d = "OUT"          # wrap the shifted-out bit to stage 1
            elif i == lo + 1:
                d = "fb"           # XNOR injection between the tap stages
            else:
                d = _q(i - 1)      # plain shift
            mod.add_module(dff, f"ff{i}", nets={
                "VDD": "VDD", "VSS": "VSS", "D": d, "CLK": "CLK",
                "CLRB": "RSTB", "Q": _q(i), "QB": f"qb{i}"})

    mod.set_behavioral_model(_prbs_behavioral_model, params={
        "vdd": _VDD, "order": order, "taps": (hi, lo), "form": form,
        "tcq_s": _CLKQ_NOM_PS * 1e-12}, execution='vector')

    if with_tests:
        mode, ncap = _VERIFY[order]
        if mode in ("spice_full", "spice_spot"):
            attach_prbs_test(mod, order=order, form=form, mode=mode, n_capture=ncap)
    return mod


def build_prbs7(*, form="fibonacci", with_tests=True) -> Module:
    """Order-7 generator alias."""
    return build_prbs(order=7, form=form, with_tests=with_tests)


# --- bit capture / sequence analysis --------------------------------------------

def _sample_bits(t, vout, n, *, t_release=_T_RELEASE, tclk=_T_CLK, vdd=_VDD):
    """OUT sampled at CLK falling edges after reset: the measured bit stream.

    The clock's rising edges are at ``k*tclk`` (k>=1); the first past ``t_release``
    is reference bit 0's clock. Sample OUT half a period later (the following
    falling edge), where OUT — which only changes on rising edges — has settled.
    """
    t = np.asarray(t, dtype=float)
    vout = np.asarray(vout, dtype=float)
    ks = [k for k in range(1, n + int(t_release / tclk) + 6) if k * tclk > t_release]
    samp = np.array([k * tclk + tclk / 2 for k in ks[:n]])
    if not len(samp):
        return []
    sig = Signal("out", "voltage", "time", "V", t, vout)
    return (sample(sig, x_new=samp).y > vdd / 2).astype(int).tolist()


def _capture_bits(sig, *, n):
    return _sample_bits(sig.x, sig.y, n)


def _detect_period(bits):
    """Smallest p (<= len//2) with bits[i]==bits[i+p] over the whole overlap."""
    n = len(bits)
    for p in range(1, n // 2 + 1):
        if all(bits[i] == bits[i + p] for i in range(n - p)):
            return p
    return n


def _is_periodic(bits, p):
    """True if bits repeat with exactly period p (needs >= 2p bits)."""
    return len(bits) >= 2 * p and all(bits[i] == bits[i + p] for i in range(p))


def _longest_run(bits):
    from itertools import groupby
    return max(len(list(g)) for _, g in groupby(bits))


# --- SPICE testbench (per order) ------------------------------------------------

def _prbs_rig(tb, *, n_capture):
    """Clock the PRBS at 600 MHz; hold RSTB low through the first edges, release."""
    stop = _T_RELEASE + (n_capture + 5) * _T_CLK
    tb.add_module_under_test({"VDD": "VDD", "VSS": "0", "CLK": "clk",
                              "RSTB": "rstb", "OUT": "out"})
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0",
                                        dc=_VDD))
    tb.add_source(sources.VoltageSource(
        "clk", positive="clk", negative="0", dc=0,
        pulse=sources.Pulse(v1=0, v2=_VDD, td=_T_CLK, tr=50e-12, tf=50e-12,
                            pw=_T_CLK / 2, per=_T_CLK)))
    tb.add_source(sources.VoltageSource(
        "rstb", positive="rstb", negative="0", dc=0,
        pwl=sources.PWL().add(0, 0).add(_T_RESET, 0).add(_T_RELEASE, _VDD)))
    tb.add_sim_type(Transient(step=10e-12, stop=stop, maxstep=20e-12))


def attach_prbs_test(mod, *, order, form, mode, n_capture):
    """One transient at 600 MHz; specs depend on the verification mode.

    ``spice_full`` (orders 7, 9): capture >= 2 periods and demand bit-for-bit vs
    the reference, exact period, XNOR balance (``2**(n-1)-1`` ones) and
    longest run n. ``spice_spot`` (order 15): a first-``n_capture``-bit spot check
    only — full-period statistics come from the reference.
    """
    period = 2 ** order - 1
    tb = mod.create_test("verify")
    _prbs_rig(tb, n_capture=n_capture)
    out = tb.probe("out")
    tb.add_metric(MATCH_METRIC, _bit_mismatches, sig=out, order=order,
                  form=form, n_capture=n_capture)
    tb.add_spec(f"PRBS{order} {form}: bit-for-bit vs reference (0 mismatches)",
                MATCH_METRIC, "==", 0.0)
    if mode == "spice_full":
        tb.add_metric(PERIOD_METRIC, _bit_period, sig=out, n_capture=n_capture)
        tb.add_metric(ONES_METRIC, _bit_ones, sig=out, period=period)
        tb.add_metric(RUN_METRIC, _bit_longest_run, sig=out, n_capture=n_capture)
        tb.add_spec(f"period == {period}", PERIOD_METRIC, "==", float(period))
        tb.add_spec(f"balance: {2**(order-1)-1} ones per {period} (XNOR)",
                    ONES_METRIC, "==", float(2 ** (order - 1) - 1))
        tb.add_spec(f"longest run == {order}", RUN_METRIC, "==", float(order))
    return tb


def _bit_mismatches(sig, order, form, n_capture):
    meas = _capture_bits(sig, n=n_capture)
    ref = prbs_reference(len(meas), order=order, form=form)
    return float(sum(1 for a, b in zip(meas, ref) if a != b))


def _bit_period(sig, n_capture):
    return float(_detect_period(_capture_bits(sig, n=n_capture)))


def _bit_ones(sig, period):
    return float(sum(_capture_bits(sig, n=period + 8)[:period]))


def _bit_longest_run(sig, n_capture):
    return float(_longest_run(_capture_bits(sig, n=n_capture)))


# --- behavioral segment and reference full-period verification ------------------

def demonstrate_composed_feedback():
    """Run a composed toggle-flop probe using the whole-array cell twins.

    Returns None if the simulation runs, or the first error line if it fails.
    The loop runs with an undelayed-feedback warning. This checks whether the
    engine accepts the connection; it does not check that the output toggles.
    A sequential feedback model needs state and explicit delays between updates.
    """
    top = Module("PDK_skywater130", example_output_dir(), "toggle_loop.cir", pdk_details=pdk,
                 module_name="toggle_loop_top")
    top.add_module(build_dff(with_tests=False), "ff", nets={
        "VDD": "vdd", "VSS": "0", "D": "d", "CLK": "clk", "CLRB": "clrb",
        "Q": "q", "QB": "qb"})
    top.add_module(build_inverter(with_tests=False), "inv", nets={
        "VDD": "vdd", "VSS": "0", "A": "q", "Y": "d"})     # D = NOT Q: the cycle
    top.add_source(sources.VoltageSource("vdd", positive="vdd", negative="0", dc=_VDD))
    top.add_source(sources.VoltageSource("clk", positive="clk", negative="0",
                   pulse=sources.Pulse(v1=0, v2=_VDD, td=_T_CLK, tr=50e-12,
                                       tf=50e-12, pw=_T_CLK / 2, per=_T_CLK)))
    top.add_source(sources.VoltageSource("clrb", positive="clrb", negative="0",
                   pwl=sources.PWL().add(0, 0).add(_T_RESET, 0).add(_T_RELEASE, _VDD)))
    try:
        top.simulate_behavioral(t_step=_T_CLK / 8, t_stop=20e-9)
        return None
    except SimulationError as e:
        return str(e).splitlines()[0]


def run_behavioral_segment(order, n_bits, *, form="fibonacci"):
    """Run the whole-LFSR behavioral model and return its sampled bit stream."""
    prbs = build_prbs(order=order, form=form, with_tests=False)
    top = Module("PDK_skywater130", example_output_dir(), "prbs_twin.cir", pdk_details=pdk,
                 module_name="prbs_twin_top")
    top.add_module(prbs, "u", nets={"VDD": "vdd", "VSS": "0", "CLK": "clk",
                                     "RSTB": "rstb", "OUT": "out"})
    top.add_source(sources.VoltageSource("vdd", positive="vdd", negative="0", dc=_VDD))
    top.add_source(sources.VoltageSource("clk", positive="clk", negative="0",
                   pulse=sources.Pulse(v1=0, v2=_VDD, td=_T_CLK, tr=50e-12,
                                       tf=50e-12, pw=_T_CLK / 2, per=_T_CLK)))
    top.add_source(sources.VoltageSource("rstb", positive="rstb", negative="0",
                   pwl=sources.PWL().add(0, 0).add(_T_RESET, 0).add(_T_RELEASE, _VDD)))
    stop = _T_RELEASE + (n_bits + 5) * _T_CLK
    top.simulate_behavioral(t_step=_T_CLK / 10, t_stop=stop)
    out = top.load_signal("out")
    return _sample_bits(out.x, out.y, n_bits)


def verify_full_period(order, *, form="fibonacci"):
    """Full-period statistics for an order SPICE cannot reach.

    Compare the first 300 bits of a 320-bit behavioral run with the reference.
    Check period, balance and longest run on two reference periods plus
    ``order`` bits. Returns a dict of results.
    """
    period = 2 ** order - 1
    seg_engine = run_behavioral_segment(order, 320, form=form)
    seg_ref = prbs_reference(320, order=order, form=form)
    anchor_ok = seg_engine[:300] == seg_ref[:300]
    full = prbs_reference(2 * period + order, order=order, form=form)
    return {
        "period": period if _is_periodic(full, period) else _detect_period(full),
        "ones": sum(full[:period]),
        "longest_run": _longest_run(full),
        "anchor_ok": anchor_ok,
    }


def run_characterization(mod, corner="tt"):
    results = mod.run_all_tests(corner=corner)
    return results, mod.reconcile_specs()


# --- plot -----------------------------------------------------------------------

def save_plot(prbs_res, path):
    """Order-7 bitstream + measured-vs-reference raster, and a family panel
    showing period vs verification mode across the orders."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig = plt.figure(figsize=(11, 9), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.75, 1.0])
    ax_wave = fig.add_subplot(gs[0])
    ax_bits = fig.add_subplot(gs[1])
    ax_fam = fig.add_subplot(gs[2])

    ck = prbs_res["prbs7_fibonacci.verify"].sim_results
    tc = np.asarray(ck.load_signal("clk").x, dtype=float)
    vc = np.asarray(ck.load_signal("clk").y, dtype=float)
    to = np.asarray(ck.load_signal("out").x, dtype=float)
    vo = np.asarray(ck.load_signal("out").y, dtype=float)
    lo, hi = _T_RELEASE, _T_RELEASE + 24 * _T_CLK
    mc = (tc >= lo) & (tc <= hi)
    mo = (to >= lo) & (to <= hi)
    ax_wave.plot((tc[mc] - lo) * 1e9, vc[mc] / _VDD * 0.8 + 2.2,
                 color="#9fb4c7", lw=1.0, label="CLK")
    ax_wave.plot((to[mo] - lo) * 1e9, vo[mo], color="#0072B2", lw=1.6, label="OUT")
    ax_wave.axhline(_VDD / 2, color="#5a5a6e", ls="--", lw=0.8)
    ax_wave.set_yticks([0, _VDD], ["0", f"{_VDD:.1f}"])
    ax_wave.set_xlabel("time after RSTB release (ns)  —  600 MHz clock")
    ax_wave.set_ylabel("OUT (V)")
    ax_wave.set_title("PRBS7 (fibonacci) out of the shift register — all-zeros "
                      "clear is a valid XNOR seed")
    ax_wave.legend(frameon=False, fontsize=8, loc="upper right")

    meas = _sample_bits(to, vo, 127)
    ref = prbs_reference(127, order=7)
    grid = np.array([meas, ref])
    ax_bits.imshow(grid, aspect="auto", cmap=ListedColormap(["#eef1f5", "#0072B2"]),
                   interpolation="nearest", extent=[0, 127, 0, 2])
    mism = sum(1 for a, b in zip(meas, ref) if a != b)
    ax_bits.set_yticks([0.5, 1.5], ["reference", "measured\n(SPICE)"])
    ax_bits.set_xlabel("bit index (one period = 127)")
    ax_bits.set_title(f"PRBS7 bit-for-bit: {mism} mismatches, "
                      f"{sum(meas)} ones / {127 - sum(meas)} zeros (dark = 1)")

    # Family panel: period (log) per order, coloured by verification mode.
    modecolor = {"spice_full": "#0072B2", "spice_spot": "#009E73",
                 "reference_only": "#D55E00"}
    orders = [7, 9, 15, 23, 31]
    periods = [2 ** o - 1 for o in orders]
    modes = [_VERIFY[o][0] for o in orders]
    ax_fam.bar([str(o) for o in orders], periods,
               color=[modecolor[m] for m in modes])
    ax_fam.set_yscale("log")
    ax_fam.set_ylabel("period = 2^order - 1 (bits, log)")
    ax_fam.set_xlabel("PRBS order")
    ax_fam.set_title("Verification scales with the period: SPICE bit-for-bit "
                     "(blue) / SPICE spot + reference full (green) / "
                     "reference only (orange)")
    for o, p, m in zip(orders, periods, modes):
        us = p / F_CLK * 1e6
        lab = f"{us:.2g} us" if us < 1e3 else f"{us/1e3:.2g} ms"
        ax_fam.text(str(o), p * 1.4, lab, ha="center", fontsize=8, color="#5a5a6e")
    ax_fam.set_ylim(50, periods[-1] * 6)

    fig.suptitle("PRBS family — generalized LFSR (x^n + x^m + 1, XNOR feedback)",
                 fontsize=13)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# family members characterized in __main__ (order, form)
_FAMILY = [(7, "fibonacci"), (7, "galois"), (9, "fibonacci"), (15, "fibonacci")]


if __name__ == "__main__":
    OUT = example_output_dir()

    print("=" * 72)
    print("PRBS family — generalized LFSR, cells from stdcells (static XOR), "
          "600 MHz")
    print("=" * 72)

    # Admission of this vector loop does not establish causal toggle dynamics.
    msg = demonstrate_composed_feedback()
    print(f"\ncomposed behavioral loop (toggle flop) via engine -> {msg!r}")
    print("  Full-period statistics use the reference; a short behavioral run checks the model.\n")

    family_rows = []          # (order, form, flops, period, mode, result)
    all_res = {}

    for order, form in _FAMILY:
        mode = _VERIFY[order][0]
        print("-" * 72)
        print(f"PRBS{order} {form}  ({mode})")
        print("-" * 72)
        prbs = build_prbs(order=order, form=form)
        res, recon = run_characterization(prbs)
        all_res[f"prbs{order}_{form}"] = res
        print(recon.summary())
        period = 2 ** order - 1
        v = res[f"prbs{order}_{form}.verify"]
        nmiss = int(v.metrics[MATCH_METRIC])
        if mode == "spice_full":
            result = (f"{nmiss} miss, period {int(v.metrics[PERIOD_METRIC])}, "
                      f"{int(v.metrics[ONES_METRIC])} ones, run "
                      f"{int(v.metrics[RUN_METRIC])}")
            family_rows.append((order, form, order, period, "SPICE full", result))
        else:   # spice_spot -> behavioral segment and reference full period
            fp = verify_full_period(order, form=form)
            result = (f"SPICE {nmiss}/{_VERIFY[order][1]} miss; reference period {fp['period']}, "
                      f"{fp['ones']} ones, run {fp['longest_run']}, "
                      f"anchor {'ok' if fp['anchor_ok'] else 'FAIL'}")
            family_rows.append((order, form, order, period,
                                "SPICE spot + ref", result))
            print(f"  reference full period: {fp['period']} bits, {fp['ones']} "
                  f"ones ({2**(order-1)-1} expected), longest run {fp['longest_run']}")
            print(f"  behavioral<->reference first 300 bits: "
                  f"{'match' if fp['anchor_ok'] else 'MISMATCH'}")
        print(f"  result: {result}")

    # Reference-only orders: table entries, no simulation.
    for order in (23, 31):
        period = 2 ** order - 1
        family_rows.append((order, "fibonacci", order, period, "reference only",
                            f"period {period} = {period/F_CLK*1e3:.1f} ms @600MHz "
                            f"(no sim)"))

    print("\n" + "=" * 72)
    print("PRBS family table")
    print("=" * 72)
    print(f"  {'order':>5} {'form':<10}{'flops':>6} {'period':>12}  "
          f"{'verify':<18}result")
    for order, form, flops, period, mode, result in family_rows:
        print(f"  {order:>5} {form:<10}{flops:>6} {period:>12}  {mode:<18}{result}")

    # Check whether a cyclic shift aligns the two order-7 references.
    bf = prbs_reference(127, order=7, form="fibonacci")
    bg = prbs_reference(127, order=7, form="galois")
    shift = next((s for s in range(127)
                  if all(bg[i] == bf[(i + s) % 127] for i in range(127))), None)
    print("\n  galois vs fibonacci (order 7): "
          + ("aligned by cyclic shift " + str(shift) if shift is not None
             else "no cyclic shift aligns the reference streams"))

    print(f"\n  saved {save_plot(all_res['prbs7_fibonacci'], os.path.join(OUT, 'prbs_generator.png'))}")
