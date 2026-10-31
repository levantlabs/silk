"""10:1 parallel-load serializers in CMOS and current-mode logic (CML).

Each stage uses a mux and a D flip-flop. LOAD captures the parallel lanes;
otherwise the word shifts toward Q0, the serial output. Reference PRBS9 bits
supply the lanes. Ideal sources drive the clock and LOAD.

The tests compare sampled serial bits with a shift-register reference and
measure supply current at 600 Mb/s. Rate sweeps report the highest passing
sample over a finite bit window, not a continuous operating limit. CML data
and clock legs use 1.4..1.8 V; LOAD legs use 0.6..1.8 V to select the mux path.
The CML stages use an 800 uA reference and 500 ohm loads. A CMOS converter
provides an additional CML readout.

The CMOS timing estimate adds isolated cell delays and an assumed setup time.
Internal probes measure clock-to-Q and mux delay in the composed register.
Subtracting them from the highest passing sampled period gives a remaining
period budget. It does not measure setup time. The clock stimulus has a fixed
rise time; this experiment does not measure clock-distribution loading.

Run with ``MPLBACKEND=Agg .venv/bin/python -m silk.examples.serializer_example``.
Requires ngspice and sky130 models. Prints mismatches, timing estimates,
sampled-rate results, power and energy per bit. Saves ``serializer.png``
in the configured example output directory. Tests use one corner and ideal
resistor loads; serializer and clock stimulus are separate circuits.
"""
import os

import numpy as np

from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk import load_pdk
from silk import sources
from silk.simulation import Transient

from silk.examples import stdcells as sc
from silk.examples import cml_cells as cml
from silk.examples import example_output_dir
from silk.metrics import average, crossing, sample

pdk = load_pdk('sky130')

_VDD = 1.8
F_BIT = 600e6
_T_BIT = 1.0 / F_BIT
RATIO = 10
_PRBS_ORDER = 9
_T_START = 2e-9


_CML_HI, _CML_LO = 1.8, 1.4
_SEL_HI, _SEL_LO = 1.8, 0.6    # Wider select swing steers the mux into one data pair.


_CML_IMUL = 4.0
_CML_IREF = 200e-6 * _CML_IMUL
_CML_RL = 2000.0 / _CML_IMUL


def build_serializer(*, variant="cmos", ratio=RATIO) -> Module:
    """Build a ratio:1 parallel-load shift register in CMOS or CML.

    Each stage selects its parallel lane on LOAD and its neighbour otherwise.
    A flip-flop captures the selected bit. The first LOAD initializes the
    register; Q0 is the serial output."""
    if variant == "cmos":
        return _build_cmos_serializer(ratio)
    if variant == "cml":
        return _build_cml_serializer(ratio)
    raise ValueError(f"unknown serializer variant {variant!r}; use 'cmos' or 'cml'")


def _build_cmos_serializer(ratio) -> Module:
    mod = Module("PDK_skywater130", example_output_dir(), "ser_cmos.cir", pdk_details=pdk,
                 module_name="ser_cmos")
    ports = {"VDD": Port.supply(), "VSS": Port.ground(),
             "CLK": Port.analog_input(), "LOAD": Port.analog_input(),
             "SOUT": Port.analog_output()}
    for i in range(ratio):
        ports[f"P{i}"] = Port.analog_input()
    mod.define_module_ports(ports)

    mux = sc.build_mux2(with_tests=False)
    dff = sc.build_dff(with_tests=False)
    for i in range(ratio):
        q = "SOUT" if i == 0 else f"q{i}"
        shift_in = f"q{i+1}" if i < ratio - 1 else "VSS"
        mod.add_module(mux, f"mux{i}", nets={
            "VDD": "VDD", "VSS": "VSS", "A": shift_in, "B": f"P{i}",
            "S": "LOAD", "Y": f"d{i}"})
        mod.add_module(dff, f"ff{i}", nets={
            "VDD": "VDD", "VSS": "VSS", "D": f"d{i}", "CLK": "CLK",
            "CLRB": "VDD", "Q": q, "QB": f"qb{i}"})
    return mod


def _build_cml_serializer(ratio, *, rl=_CML_RL, imul=_CML_IMUL) -> Module:
    """Build differential mux/DFF stages and a CMOS serial readout.

    rl sets each load; imul scales tail current and device sizing."""
    mod = Module("PDK_skywater130", example_output_dir(), "ser_cml.cir", pdk_details=pdk,
                 module_name="ser_cml")
    ports = {"VDD": Port.supply(), "VSS": Port.ground(), "IBIAS": Port.analog_input(),
             "CLKP": Port.analog_input(), "CLKN": Port.analog_input(),
             "LOADP": Port.analog_input(), "LOADN": Port.analog_input(),


             "SOUT": Port.analog_output(),
             "SOUTP": Port.analog_output(), "SOUTN": Port.analog_output()}
    for i in range(ratio):
        ports[f"P{i}P"] = Port.analog_input()
        ports[f"P{i}N"] = Port.analog_input()
    mod.define_module_ports(ports)
    muxc = cml.build_cml_mux2(rl=rl, imul=imul)
    dffc = cml.build_cml_dff(rl=rl, imul=imul)
    for i in range(ratio):
        qp = "SOUTP" if i == 0 else f"qp{i}"
        qn = "SOUTN" if i == 0 else f"qn{i}"

        sip, sin = (f"qp{i+1}", f"qn{i+1}") if i < ratio - 1 else ("VSS", "VDD")

        mod.add_module(muxc, f"mux{i}", nets={
            "VDD": "VDD", "VSS": "VSS", "IBIAS": "IBIAS",
            "AP": sip, "AN": sin, "BP": f"P{i}P", "BN": f"P{i}N",
            "SELP": "LOADP", "SELN": "LOADN", "OUTP": f"mp{i}", "OUTN": f"mn{i}"})
        mod.add_module(dffc, f"ff{i}", nets={
            "VDD": "VDD", "VSS": "VSS", "IBIAS": "IBIAS",
            "CLKP": "CLKP", "CLKN": "CLKN", "DP": f"mp{i}", "DN": f"mn{i}",
            "OUTP": qp, "OUTN": qn})

    mod.add_module(cml.build_cml2cmos(imul=imul), "obs", nets={
        "VDD": "VDD", "VSS": "VSS", "IBIAS": "IBIAS",
        "INP": "SOUTP", "INM": "SOUTN", "Y": "SOUT"})
    return mod


def _lane_bits(prbs_bits, ratio, n_words):
    """Demux prbs into ``ratio`` lanes: lane i of word w = prbs[ratio*w + i]."""
    return [[prbs_bits[ratio * w + i] for w in range(n_words)] for i in range(ratio)]


def serialize_golden(prbs_bits, n_clocks, *, ratio=RATIO):
    """Clock-by-clock RTL golden: mux(LOAD?lane:shift)->DFF, SOUT=Q0 after edge.

    LOAD high when ``clock % ratio == 0``. Returns SOUT bit after each edge.
    """
    q = [0] * ratio
    out = []
    for c in range(n_clocks):
        load = (c % ratio == 0)
        w = c // ratio
        d = []
        for i in range(ratio):
            if load:
                idx = ratio * w + i
                d.append(prbs_bits[idx] if idx < len(prbs_bits) else 0)
            else:
                d.append(q[i + 1] if i < ratio - 1 else 0)
        q = d
        out.append(q[0])
    return out


def _sample_sout(ckt, n_clocks, *, tbit=_T_BIT, t0=_T_START, vdd=_VDD, sig="sout"):
    """Sample SOUT half a period after each nominal clock rise at t0 + c*tbit."""
    s = ckt.load_signal(sig)
    samp = np.array([t0 + c * tbit + tbit / 2 for c in range(n_clocks)])
    return (sample(s, x_new=samp).y > vdd / 2).astype(int).tolist()


def _best_lag_mismatch(meas, golden, *, skip=RATIO, compare=30, max_lag=6):
    """Min mismatches between measured and golden over ``compare`` bits after a
    ``skip``-bit startup, trying small pipeline lags. Returns (mismatch, lag)."""
    best = (compare + 1, 0)
    for lag in range(max_lag + 1):
        m = meas[skip + lag: skip + lag + compare]
        g = golden[skip: skip + compare]
        if len(m) < compare:
            continue
        mis = sum(1 for a, b in zip(m, g) if a != b)
        if mis < best[0]:
            best = (mis, lag)
    return best


def _run_cmos(prbs_bits, *, tbit, n_words, measure_power=False):
    """Build the CMOS serializer top, drive CLK/LOAD/lanes, simulate, and return
    (measured SOUT bits, mean supply current A or None)."""
    n_clocks = RATIO * n_words
    lanes = _lane_bits(prbs_bits, RATIO, n_words)
    mod = _build_cmos_serializer(RATIO)
    top = Module("PDK_skywater130", example_output_dir(), "ser_cmos_tb.cir", pdk_details=pdk)
    nets = {"VDD": "VDD", "VSS": "0", "CLK": "clk", "LOAD": "load", "SOUT": "sout"}
    for i in range(RATIO):
        nets[f"P{i}"] = f"p{i}"
    top.add_module(mod, "dut", nets=nets)
    with top:
        sources.VoltageSource("vdd", positive="VDD", negative="0", dc=_VDD)
        _tedge = min(50e-12, tbit / 10)
        sources.VoltageSource("clk", positive="clk", negative="0", dc=0,
            pulse=sources.Pulse(v1=0, v2=_VDD, td=_T_START, tr=_tedge, tf=_tedge,
                                pw=tbit / 2, per=tbit))
        sources.VoltageSource("load", positive="load", negative="0", dc=0,
            pulse=sources.Pulse(v1=0, v2=_VDD, td=_T_START - 0.3 * tbit,
                                tr=0.05 * tbit, tf=0.05 * tbit,
                                pw=0.95 * tbit, per=RATIO * tbit, np=n_words))
        for i in range(RATIO):


            sources.VoltageSource(f"vp{i}", positive=f"p{i}", negative="0",
                waveform=sources.BitPattern(lanes[i], v1=0, v2=_VDD,
                    ui=RATIO * tbit, td=_T_START - 0.4 * tbit,
                    tr=0.05 * tbit, tf=0.05 * tbit))
    stop = _T_START + (n_clocks + 2) * tbit
    top.add_sim_type(Transient(step=tbit / 60, stop=stop, maxstep=tbit / 30))
    top.simulate(corner="tt")
    meas = _sample_sout(top, n_clocks, tbit=tbit)
    cur = None
    if measure_power:

        # Weight current by time because SPICE uses an adaptive grid.
        cur = float(average(top.load_signal("vvdd", "I").mag().window(
            _T_START + 5 * tbit, stop - tbit)))
    return meas, cur


def cmos_functional(prbs_bits, *, tbit=_T_BIT, n_words=6):
    """Return the mismatch count against the reference over three words."""
    meas, _ = _run_cmos(prbs_bits, tbit=tbit, n_words=n_words)
    golden = serialize_golden(prbs_bits, RATIO * n_words)
    return _best_lag_mismatch(meas, golden, skip=RATIO, compare=3 * RATIO)[0]


def cmos_power(prbs_bits, *, tbit=_T_BIT, n_words=6):
    """(avg supply current A, power W, energy/bit J) at the given bit rate."""
    _, cur = _run_cmos(prbs_bits, tbit=tbit, n_words=n_words, measure_power=True)
    p = _VDD * cur
    return cur, p, p * tbit


def predicted_cmos_ceiling():
    """Estimate a bit period from isolated clock-to-Q, mux delay and setup.

    Return (period_s, rate_Hz); setup is an assumed 100 ps."""
    clkq = sc._CLKQ_NOM_PS * 1e-12
    mux_tp = sc._TP_NOM_PS["mux2"] * 1e-12
    setup = 100e-12
    period = clkq + mux_tp + setup
    return period, 1.0 / period


def cmos_rate_ceiling(prbs_bits, *, rates_ghz=(0.6, 1.0, 1.5, 2.0, 3.0, 4.0),
                      n_words=5):
    """Return the highest passing sampled CMOS rate and (rate, mismatch) rows.

    Finite bit windows and discrete rates do not establish an operating limit."""
    rows = []
    ceiling = 0.0
    for f in rates_ghz:
        tb = 1.0 / (f * 1e9)
        meas, _ = _run_cmos(prbs_bits, tbit=tb, n_words=n_words)
        golden = serialize_golden(prbs_bits, RATIO * n_words)
        mis = _best_lag_mismatch(meas, golden, skip=RATIO, compare=2 * RATIO)[0]
        rows.append((f * 1e9, mis))
        if mis == 0:
            ceiling = f * 1e9
    return ceiling, rows


def _run_cml(prbs_bits, *, tbit, n_words, measure_power=False, probe="sout"):
    """Simulate differential clock, LOAD and lanes through the CML serializer.

    Return measured bits and optional mean supply current. probe="dp" reads
    differential SOUTP-SOUTN; probe="sout" reads the CMOS converter output."""
    n_clocks = RATIO * n_words
    lanes = _lane_bits(prbs_bits, RATIO, n_words)
    mod = _build_cml_serializer(RATIO, rl=_CML_RL, imul=_CML_IMUL)
    top = Module("PDK_skywater130", example_output_dir(), "ser_cml_tb.cir", pdk_details=pdk)
    top.add_module(cml.build_cml_bias(imul=_CML_IMUL), "bias", nets={"VSS": "0", "IBIAS": "ibias"})
    nets = {"VDD": "VDD", "VSS": "0", "IBIAS": "ibias",
            "CLKP": "clkp", "CLKN": "clkn", "LOADP": "loadp", "LOADN": "loadn",
            "SOUT": "sout", "SOUTP": "soutp", "SOUTN": "soutn"}
    for i in range(RATIO):
        nets[f"P{i}P"] = f"p{i}p"
        nets[f"P{i}N"] = f"p{i}n"
    top.add_module(mod, "dut", nets=nets)

    with top:
        sources.VoltageSource("vdd", positive="VDD", negative="0", dc=_VDD)
        sources.CurrentSource("iref", positive="VDD", negative="ibias", dc=_CML_IREF)
        _tedge = min(50e-12, tbit / 10)
        sources.VoltageSource("clkp", positive="clkp", negative="0",
            pulse=sources.Pulse(v1=_CML_LO, v2=_CML_HI, td=_T_START, tr=_tedge,
                                tf=_tedge, pw=tbit / 2, per=tbit))
        sources.VoltageSource("clkn", positive="clkn", negative="0",
            pulse=sources.Pulse(v1=_CML_HI, v2=_CML_LO, td=_T_START, tr=_tedge,
                                tf=_tedge, pw=tbit / 2, per=tbit))
        sources.VoltageSource("loadp", positive="loadp", negative="0",
            pulse=sources.Pulse(v1=_SEL_LO, v2=_SEL_HI,
                td=_T_START - 0.3 * tbit, tr=0.05 * tbit, tf=0.05 * tbit,
                pw=0.95 * tbit, per=RATIO * tbit, np=n_words))
        sources.VoltageSource("loadn", positive="loadn", negative="0",
            pulse=sources.Pulse(v1=_SEL_HI, v2=_SEL_LO,
                td=_T_START - 0.3 * tbit, tr=0.05 * tbit, tf=0.05 * tbit,
                pw=0.95 * tbit, per=RATIO * tbit, np=n_words))
        for i in range(RATIO):
            sources.VoltageSource(f"vp{i}p", positive=f"p{i}p", negative="0",
                waveform=sources.BitPattern(lanes[i], v1=_CML_LO, v2=_CML_HI,
                    ui=RATIO * tbit, td=_T_START - 0.4 * tbit,
                    tr=0.05 * tbit, tf=0.05 * tbit))
            sources.VoltageSource(f"vp{i}n", positive=f"p{i}n", negative="0",
                waveform=sources.BitPattern(lanes[i], v1=_CML_HI, v2=_CML_LO,
                    ui=RATIO * tbit, td=_T_START - 0.4 * tbit,
                    tr=0.05 * tbit, tf=0.05 * tbit))
    stop = _T_START + (n_clocks + 2) * tbit
    top.add_sim_type(Transient(step=tbit / 60, stop=stop, maxstep=tbit / 30))
    top.simulate(corner="tt")
    if probe == "dp":

        sp, sn = top.load_signal("soutp"), top.load_signal("soutn")
        samp = np.array([_T_START + c * tbit + tbit / 2 for c in range(n_clocks)])
        meas = (sample(sp - sn, x_new=samp).y > 0).astype(int).tolist()
    else:
        meas = _sample_sout(top, n_clocks, tbit=tbit)
    cur = None
    if measure_power:

        # Weight current by time because SPICE uses an adaptive grid.
        cur = float(average(top.load_signal("vvdd", "I").mag().window(
            _T_START + 5 * tbit, stop - tbit)))
    return meas, cur


def cml_functional(prbs_bits, *, tbit=_T_BIT, n_words=5, probe="sout"):
    meas, _ = _run_cml(prbs_bits, tbit=tbit, n_words=n_words, probe=probe)
    golden = serialize_golden(prbs_bits, RATIO * n_words)
    return _best_lag_mismatch(meas, golden, skip=RATIO, compare=3 * RATIO)[0]


def cml_power(prbs_bits, *, tbit=_T_BIT, n_words=5):
    _, cur = _run_cml(prbs_bits, tbit=tbit, n_words=n_words, measure_power=True)
    p = _VDD * cur
    return cur, p, p * tbit


def cml_rate_ceiling(prbs_bits, *, rates_ghz=(2.0, 3.0, 4.0, 5.0, 6.0),
                     n_words=4, probe="dp"):
    """Return the highest passing sampled CML rate and (rate, mismatch) rows.

    probe="dp" reads the differential datapath; probe="sout" reads its CMOS
    converter. Finite bit windows do not establish an operating limit."""
    rows = []
    ceiling = 0.0
    for f in rates_ghz:
        tb = 1.0 / (f * 1e9)
        meas, _ = _run_cml(prbs_bits, tbit=tb, n_words=n_words, probe=probe)
        golden = serialize_golden(prbs_bits, RATIO * n_words)
        mis = _best_lag_mismatch(meas, golden, skip=RATIO, compare=2 * RATIO)[0]
        rows.append((f * 1e9, mis))
        if mis == 0:
            ceiling = f * 1e9
    return ceiling, rows


def explain_cmos_gap(prbs_bits, *, f_ghz=1.0):
    """Measure clock-to-Q and mux delays in the composed CMOS serializer.

    Return median delays in ps from internal probes. These do not measure
    setup time or the loading of the ideal clock source."""
    tbit = 1.0 / (f_ghz * 1e9)
    n_clocks = RATIO * 4
    lanes = _lane_bits(prbs_bits, RATIO, 4)
    mod = _build_cmos_serializer(RATIO)
    top = Module("PDK_skywater130", example_output_dir(), "ser_gap.cir", pdk_details=pdk)
    nets = {"VDD": "VDD", "VSS": "0", "CLK": "clk", "LOAD": "load", "SOUT": "sout"}
    for i in range(RATIO):
        nets[f"P{i}"] = f"p{i}"
    h = top.add_module(mod, "dut", nets=nets)
    with top:
        sources.VoltageSource("vdd", positive="VDD", negative="0", dc=_VDD)
        _tedge = min(50e-12, tbit / 10)
        sources.VoltageSource("clk", positive="clk", negative="0", dc=0,
            pulse=sources.Pulse(v1=0, v2=_VDD, td=_T_START, tr=_tedge, tf=_tedge,
                                pw=tbit / 2, per=tbit))
        sources.VoltageSource("load", positive="load", negative="0", dc=0,
            pulse=sources.Pulse(v1=0, v2=_VDD, td=_T_START - 0.3 * tbit,
                                tr=0.05 * tbit, tf=0.05 * tbit,
                                pw=0.95 * tbit, per=RATIO * tbit, np=4))
        for i in range(RATIO):
            sources.VoltageSource(f"vp{i}", positive=f"p{i}", negative="0",
                waveform=sources.BitPattern(lanes[i], v1=0, v2=_VDD,
                    ui=RATIO * tbit, td=_T_START - 0.4 * tbit,
                    tr=0.05 * tbit, tf=0.05 * tbit))
    stop = _T_START + (n_clocks + 2) * tbit
    top.add_sim_type(Transient(step=tbit / 80, stop=stop, maxstep=tbit / 40))
    top.simulate(corner="tt")

    clk = top.load_signal("clk")

    def cross(sig, t_from):
        t_x = crossing(sig, _VDD / 2, edge='any', start=t_from)
        return None if np.isnan(t_x) else float(t_x)


    q1 = getattr(h, "q1")
    d0 = getattr(h, "d0")
    clkq, muxd = [], []
    for c in range(RATIO + 2, n_clocks - 2):
        te = _T_START + c * tbit
        tclk = cross(clk, te - 0.2 * tbit)
        if tclk is None:
            continue
        tq = cross(q1, tclk + 5e-12)
        if tq is None or tq - tclk > 0.9 * tbit:
            continue
        td = cross(d0, tq + 1e-12)
        clkq.append((tq - tclk) * 1e12)
        if td is not None and td - tq < 0.9 * tbit:
            muxd.append((td - tq) * 1e12)
    return {"clk_q_ctx_ps": float(np.median(clkq)) if clkq else float("nan"),
            "mux_d_ctx_ps": float(np.median(muxd)) if muxd else float("nan")}


def save_plot(prbs_bits, cmos_rows, cml_rows, path):
    """Plot serial-output comparisons and sampled rate-sweep results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig = plt.figure(figsize=(12, 8), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.0])
    ax_c = fig.add_subplot(gs[0, 0])
    ax_l = fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[1, :])
    cmap = ListedColormap(["#eef1f5", "#0072B2"])

    for ax, runner, title in ((ax_c, _run_cmos, "CMOS"), (ax_l, _run_cml, "CML")):
        nb = 3 * RATIO
        meas, _ = runner(prbs_bits, tbit=_T_BIT, n_words=5)
        golden = serialize_golden(prbs_bits, RATIO * 5)
        lag = _best_lag_mismatch(meas, golden, skip=RATIO, compare=3 * RATIO)[1]
        m = meas[RATIO + lag: RATIO + lag + nb]
        g = golden[RATIO: RATIO + nb]
        mis = sum(1 for a, b in zip(m, g) if a != b)
        ax.imshow(np.array([m, g]), aspect="auto", cmap=cmap,
                  interpolation="nearest", extent=[0, nb, 0, 2])
        ax.set_yticks([0.5, 1.5], ["golden", "SOUT"])
        ax.set_xlabel("serial bit index (3 words)")
        ax.set_title(f"{title} 10:1 @ 600 Mb/s — {mis} mismatches")


    all_rates = sorted({f for f, _ in cmos_rows} | {f for f, _ in cml_rows})
    cmos_ok = {f: m == 0 for f, m in cmos_rows}
    cml_ok = {f: m == 0 for f, m in cml_rows}
    x = np.arange(len(all_rates))
    for offset, measured, color, label in (
            (-0.2, cmos_ok, "#0072B2", "CMOS"),
            (0.2, cml_ok, "#D55E00", "CML")):
        ax_r.bar(x + offset, [np.nan if f not in measured else
                              1 if measured[f] else 0.02 for f in all_rates],
                 width=0.4, color=color, label=label)
        for xi, f in zip(x, all_rates):
            if f not in measured:
                ax_r.text(xi + offset, 0.05, "n/a", ha="center", fontsize=7,
                          rotation=90, color=color)
    ax_r.set_xticks(x, [f"{f/1e9:g}" for f in all_rates])
    ax_r.set_yticks([])
    ax_r.set_xlabel("bit rate (Gb/s) — full: pass; short: fail; n/a: unmeasured")
    ax_r.set_title("Sampled rates: full bar = bit-exact over the finite PRBS window")
    ax_r.legend(frameon=False, fontsize=9, loc="upper right")

    fig.suptitle("10:1 serializer — CMOS vs CML: same architecture, the speed/"
                 "power trade", fontsize=13)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    OUT = example_output_dir()
    prbs = sources.prbs_bits(_PRBS_ORDER, RATIO * 8)

    print("=" * 72)
    print("10:1 serializer — CMOS (stdcells) vs CML (cml_cells), 600 Mb/s")
    print("=" * 72)


    mis_c = cmos_functional(prbs)
    cur_c, p_c, e_c = cmos_power(prbs)
    print(f"\nCMOS @ 600 Mb/s : {mis_c} mismatches, {cur_c*1e6:.0f} uA, "
          f"{p_c*1e6:.0f} uW, {e_c*1e15:.0f} fJ/bit")
    pred_p, pred_f = predicted_cmos_ceiling()
    ceil_c, rows_c = cmos_rate_ceiling(prbs)
    print(f"  timing estimate: {pred_f/1e9:.1f} Gb/s "
          f"(clk-Q {sc._CLKQ_NOM_PS:.0f}+mux {sc._TP_NOM_PS['mux2']:.0f}+setup 100 "
          f"= {pred_p*1e12:.0f} ps)  -> highest passing sampled CMOS rate: {ceil_c/1e9:.1f} Gb/s")
    for f, m in rows_c:
        print(f"    {f/1e9:>4.1f} Gb/s : {'PASS' if m == 0 else f'FAIL({m})'}")


    gap = explain_cmos_gap(prbs)
    real_period = 1.0 / ceil_c
    print("\n  Internal timing probes at 1 Gb/s:")
    print(f"    clk-to-Q: {gap['clk_q_ctx_ps']:.0f} ps "
          f"(isolated cell {sc._CLKQ_NOM_PS:.0f} ps)")
    print(f"    mux delay: {gap['mux_d_ctx_ps']:.0f} ps "
          f"(isolated cell {sc._TP_NOM_PS['mux2']:.0f} ps)")
    remaining_budget = real_period * 1e12 - gap['clk_q_ctx_ps'] - gap['mux_d_ctx_ps']
    print(f"    highest passing sampled period: {real_period*1e12:.0f} ps; "
          f"remaining period budget: {remaining_budget:.0f} ps")
    print("    This budget subtracts delays measured at 1 Gb/s; it is not a setup "
          "measurement. The ideal clock has a fixed rise time and does not "
          "measure clock-distribution loading.")


    mis_l = cml_functional(prbs)
    cur_l, p_l, e_l = cml_power(prbs)

    ceil_l, rows_l = cml_rate_ceiling(prbs, probe="dp")
    ceil_sout, rows_sout = cml_rate_ceiling(prbs, probe="sout")
    print(f"\nCML @ 600 Mb/s  : {mis_l} mismatches, {cur_l*1e3:.1f} mA, "
          f"{p_l*1e3:.1f} mW, {e_l*1e15:.0f} fJ/bit  "
          f"(speed point: imul {_CML_IMUL:.0f}, {_CML_IREF*1e6:.0f} uA, "
          f"{_CML_RL:.0f} ohm)")
    print(f"  DATAPATH highest passing sampled rate (differential q0): {ceil_l/1e9:.0f} Gb/s")
    for f, m in rows_l:
        print(f"    {f/1e9:>4.0f} Gb/s : {'PASS' if m == 0 else f'FAIL({m})'}")
    print(f"  SOUT highest passing sampled rate through cml2cmos: {ceil_sout/1e9:.0f} Gb/s.")
    print("  A highest passing sample in this finite PRBS sweep does not "
          "establish a continuous operating ceiling.")


    print("\n" + "=" * 72)
    print("comparison — same architecture, the speed/power trade")
    print("=" * 72)
    print(f"  {'metric':<22}{'CMOS':>16}{'CML':>16}")
    print(f"  {'functional':<22}{f'{mis_c} miss':>16}{f'{mis_l} miss':>16}")
    print(f"  {'highest sampled pass':<22}{'%.1f Gb/s' % (ceil_c/1e9):>16}"
          f"{'%.0f Gb/s' % (ceil_l/1e9):>16}")
    print(f"  {'power @600Mb/s':<22}{'%.0f uW' % (p_c*1e6):>16}"
          f"{'%.1f mW' % (p_l*1e3):>16}")
    print(f"  {'energy/bit @600Mb/s':<22}{'%.0f fJ' % (e_c*1e15):>16}"
          f"{'%.0f fJ' % (e_l*1e15):>16}")
    print(f"  {'output swing':<22}{'1.8 V rail':>16}{'~0.4 V diff':>16}")
    print(f"  {'devices/stage':<22}{'6+32=38':>16}{'7+14=21':>16}")
    print(f"\n  Highest passing samples: CML {ceil_l/1e9:g} Gb/s, CMOS {ceil_c/1e9:g} Gb/s. "
          f"At 600 Mb/s, measured power is {p_l*1e3:.1f} mW for CML and "
          f"{p_c*1e6:.0f} uW for CMOS.")

    png = save_plot(prbs, rows_c, rows_l, os.path.join(OUT, "serializer.png"))
    print(f"\n  saved {png}")
