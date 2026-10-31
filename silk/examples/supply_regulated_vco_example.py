"""Supply-regulated ring VCO composed from an LDO and a ring oscillator.

VCTRL drives the LDO reference. Its output VREG supplies the ring, so the
ring's current loads the regulator. The tuning tests measure frequency at
five control voltages after the LDO settles and fit a tuning slope.
A separate transient plots the rail, VREG and ring output.

The rejection test applies a 200 mV, 1 MHz supply perturbation, where the
LDO retains measurable rejection. It compares period jitter with and
without the regulator. This ratio is not an AC PSRR measurement.
A fine time step is needed to resolve small period changes.

Run from the repository root with ngspice and sky130 models installed:
    MPLBACKEND=Agg .venv/bin/python -m silk.examples.supply_regulated_vco_example

The script prints the tuning curve, fitted slope, both jitter values,
their rejection ratio and spec results, and saves a transient plot.
"""
import os

import numpy as np

from silk.metrics import average_frequency, period_jitter, peak_to_peak
from silk.signals.spice import NgspiceSignal

from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk import load_pdk
from silk import sources
from silk.simulation import Transient

from silk.examples.ldo_regulator_example import build_ldo, _VDD as _RAIL, _I_AMP
from silk.examples.ring_oscillator_example import build_ring_oscillator
from silk.examples import example_output_dir

pdk = load_pdk('sky130')


_VCTRL_LO, _VCTRL_HI = 1.0, 1.8
_TUNE_VCTRLS = (1.0, 1.2, 1.4, 1.6, 1.8)

_T_STEP = 2e-12
_T_EN = 1e-9


_TUNE_STOP = 1.5e-6
_TUNE_SETTLE = 1.0e-6
_TUNE_MAXSTEP = 10e-12


F_METRIC = "f_osc_mhz"
KVCO_METRIC = "kvco_mhz_per_v"
REJ_METRIC = "jitter_rejection_x"

KVCO_MIN = 800.0

REJ_MIN = 20.0


def build_supply_regulated_vco(*, with_tests=True) -> Module:
    """Connect the LDO output to the ring supply on the shared VREG net."""
    mod = Module("PDK_skywater130", example_output_dir(), "supply_regulated_vco.cir",
                 pdk_details=pdk, module_name="supply_regulated_vco")
    mod.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "VCTRL": Port.analog_input(), "EN": Port.analog_input(),
        "IBIAS_AMP": Port.analog_input(), "OUT": Port.analog_output(),
    })

    ldo = build_ldo(with_tests=False)
    mod.add_module(ldo, "ldo", nets={
        "VDD": "VDD", "VSS": "VSS", "VCTRL": "VCTRL", "VREG": "VREG",
        "IBIAS_AMP": "IBIAS_AMP"})

    ring = build_ring_oscillator(with_tests=False)
    mod.add_module(ring, "ring", nets={
        "VDD": "VREG", "VSS": "VSS", "EN": "EN", "OUT": "OUT"})

    if with_tests:
        attach_tune_test(mod)
    return mod


def _rig(tb, *, rail=None, bias=True):
    """Add a selectable rail and enable step; omit amplifier bias for a bare ring."""
    if rail is None:
        rail = sources.VoltageSource("vdd", positive="VDD", negative="0", dc=_RAIL)
    tb.add_source(rail)
    if bias:
        tb.add_source(sources.CurrentSource("iamp", positive="ibias_amp", negative="0",
                                            dc=_I_AMP))

    # Hold EN high after its initial step throughout the run.
    tb.add_source(sources.VoltageSource(
        "en", positive="en", negative="0", dc=0,
        pwl=sources.PWL().add(0, 0).add(_T_EN, 0).add(_T_EN + 100e-12, 1.8)))


def _mut(tb):
    tb.add_module_under_test({"VDD": "VDD", "VSS": "0", "VCTRL": "vctrl",
                              "EN": "en", "IBIAS_AMP": "ibias_amp", "OUT": "out"})


def attach_tune_test(mod, *, vctrls=_TUNE_VCTRLS):
    """Register one settled-frequency transient per control voltage."""
    # Measure frequency after the LDO settles to exclude startup supply sag.
    for v in vctrls:
        tb = mod.create_test(f"tune_{v}")
        _mut(tb)
        _rig(tb)
        tb.add_source(sources.VoltageSource("vctrl", positive="vctrl", negative="0",
                                            dc=v))

        tb.add_sim_type(Transient(step=_T_STEP, stop=_TUNE_STOP,
                                  maxstep=_TUNE_MAXSTEP))
        frequency = tb.add_metric("frequency_hz", average_frequency,
                                  sig=tb.probe("out"), threshold=0.5,
                                  start=_TUNE_SETTLE)
        tb.add_metric(F_METRIC, frequency / 1e6)
    return mod


def run_tune(mod, *, vctrls=_TUNE_VCTRLS, corner="tt"):
    """Run the per-VCTRL tuning testbenches; return (vctrls, freqs, kvco_MHz/V)."""
    import contextlib
    import io
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        results = mod.run_all_tests(corner=corner)
    freqs = []
    for v in vctrls:
        r = results.get(f"{mod.module_name}.tune_{v}")
        if r is None or not hasattr(r, "metrics"):
            raise RuntimeError(f"tune at VCTRL={v} failed:\n{log.getvalue()}")
        freqs.append(r.metrics[F_METRIC])
    kvco = float(np.polyfit(vctrls, freqs, 1)[0])
    return list(vctrls), freqs, kvco


_REJ_FM = 1e6

_REJ_AMPL = 0.2
_REJ_STOP = 3e-6
_REJ_SETTLE = 1.5e-6
_REJ_MAXSTEP = 2e-12


def _period_rms_ps(sig):
    """RMS spread of the ring's period over the settled window, in ps."""
    return period_jitter(sig, threshold=0.5, start=_REJ_SETTLE) * 1e12


def _regulated_jitter_ps(mod, *, vctrl, fm, ampl, corner="tt"):
    """Measure period jitter with a sine perturbation on the LDO supply."""
    tb = mod.create_test(f"rej_reg_{fm:.0e}")
    _mut(tb)

    _rig(tb, rail=sources.VoltageSource(
        "vdd", positive="VDD", negative="0",
        sine=sources.Sine(voffset=_RAIL, vampl=ampl, freq=fm)))
    tb.add_source(sources.VoltageSource("vctrl", positive="vctrl", negative="0",
                                        dc=vctrl))
    tb.add_sim_type(Transient(step=_T_STEP, stop=_REJ_STOP, maxstep=_REJ_MAXSTEP))
    out = tb.probe("out")
    tb.run(corner=corner)
    return _period_rms_ps(out)


def _unregulated_jitter_ps(*, vctrl, fm, ampl, corner="tt"):
    """Measure period jitter with the same perturbation on the ring supply."""
    ring = build_ring_oscillator(with_tests=False)
    tb = ring.create_test(f"rej_raw_{fm:.0e}")
    tb.add_module_under_test({"VDD": "VDD", "VSS": "0", "EN": "en", "OUT": "out"})
    _rig(tb, bias=False, rail=sources.VoltageSource(
        "vdd", positive="VDD", negative="0",
        sine=sources.Sine(voffset=vctrl, vampl=ampl, freq=fm)))
    tb.add_sim_type(Transient(step=_T_STEP, stop=_REJ_STOP, maxstep=_REJ_MAXSTEP))
    out = tb.probe("out")
    tb.run(corner=corner)
    return _period_rms_ps(out)


def measure_rejection(mod, *, vctrl=1.4, fm=_REJ_FM, ampl=_REJ_AMPL):
    """Return regulated jitter, unregulated jitter and their ratio.

    Jitter values are in ps. An unavailable jitter value produces NaN
    rejection, so the lower-bound spec fails.
    """
    j_reg = _regulated_jitter_ps(mod, vctrl=vctrl, fm=fm, ampl=ampl)
    j_raw = _unregulated_jitter_ps(vctrl=vctrl, fm=fm, ampl=ampl)
    rejection = (float("nan") if np.isnan(j_reg) or np.isnan(j_raw)
                 else j_raw / j_reg if j_reg > 0 else float("inf"))
    return j_reg, j_raw, rejection


def transient_waveforms(mod, *, vctrl=1.4, stop=2e-6, corner="tt"):
    """Return snapshots as arrays: time, bench rail, DUT VREG and bench OUT."""
    import contextlib
    import io
    tb = mod.create_test("wave")
    _mut(tb)
    _rig(tb)
    tb.add_source(sources.VoltageSource("vctrl", positive="vctrl", negative="0",
                                        dc=vctrl))
    tb.add_sim_type(Transient(step=_T_STEP, stop=stop, maxstep=2 * _T_STEP))
    with contextlib.redirect_stdout(io.StringIO()):
        tb.run(corner=corner)

    out = tb.probe("out").snapshot()
    t = np.asarray(out.x, dtype=float)
    return (t,
            np.asarray(tb.probe("VDD").snapshot().y, dtype=float),
            np.asarray(tb.probe("dut.VREG").snapshot().y, dtype=float),
            np.asarray(out.y, dtype=float))


def save_transient_plot(mod, path, *, vctrl=1.4, stop=2e-6):
    """Plot the supply rail, regulated voltage and ring output."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, rail, vreg, out = transient_waveforms(mod, vctrl=vctrl, stop=stop)
    tus = t * 1e6
    vreg_sig = NgspiceSignal(name="vreg", type="voltage", xlabel="time", ylabel="V",
                             xvalues=t, yvalues=vreg)
    ripple = peak_to_peak(vreg_sig, start=1e-6) * 1e3

    fig, ax = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    ax[0].plot(tus, rail, color="#5a5a6e", lw=1)
    ax[0].set_ylabel("VDD rail (V)")
    ax[0].set_title(f"Supply-regulated ring VCO — clean 3.3 V rail, VCTRL={vctrl} V "
                    f"(tail VREG ripple {ripple:.0f} mV)")
    ax[1].plot(tus, vreg, color="#0072B2", lw=1.2)
    ax[1].axhline(vctrl, color="#D55E00", ls="--", lw=1, label=f"VCTRL target = {vctrl} V")
    ax[1].set_ylabel("VREG = LDO out (V)")
    ax[1].legend(frameon=False, fontsize=8)
    ax[2].plot(tus, out, color="#009E73", lw=0.5)
    ax[2].set_ylabel("OUT = ring out (V)")
    ax[2].set_xlabel("time (µs)")
    for a in ax:
        a.grid(True, color="#d5d5dd", lw=0.6, alpha=0.6)
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    OUT = example_output_dir()

    mod = build_supply_regulated_vco()
    print("=" * 72)
    print("Supply-regulated ring VCO — LDO + ring composed")
    print("=" * 72)

    vctrls, freqs, kvco = run_tune(mod)
    print(f"\n  tuning curve (VCTRL -> VREG -> f_osc):")
    print(f"  {'VCTRL (V)':>10} {'f_osc (MHz)':>13}")
    for v, f in zip(vctrls, freqs):
        print(f"  {v:10.2f} {f:13.1f}")
    print(f"\n  K_vco = {kvco:.0f} MHz/V over VCTRL {_VCTRL_LO}-{_VCTRL_HI} V")

    png = save_transient_plot(mod, os.path.join(OUT, "vco_regulated_transient.png"))
    print(f"\n  saved {png}")

    print(f"\n  supply-noise rejection ({_REJ_AMPL*1e3:.0f} mV @ "
          f"{_REJ_FM/1e6:.0f} MHz on the {_RAIL} V rail):")
    j_reg, j_raw, rej = measure_rejection(mod)
    print(f"    ring on the perturbed supply directly : {j_raw:7.2f} ps rms")
    print(f"    ring fed through the LDO              : {j_reg:7.2f} ps rms")
    print(f"    -> LDO rejects supply-induced jitter {rej:.0f}x "
          f"({20*np.log10(rej):.0f} dB)")

    import time as _time
    from silk.specs import Measurement
    kvco_spec = mod.declare_spec(f"K_vco > {KVCO_MIN:.0f} MHz/V", KVCO_METRIC, ">", KVCO_MIN,
                     units="MHz/V")
    rejection_spec = mod.declare_spec(f"jitter rejection > {REJ_MIN:.0f}x @ {_REJ_FM/1e6:.0f} MHz",
                     REJ_METRIC, ">", REJ_MIN)
    for spec, value in ((kvco_spec, kvco), (rejection_spec, rej)):
        spec.record_measurement(Measurement(source="spice", value=value,
                                            corner="tt", timestamp=_time.time()))
    print()
    print(mod.reconcile_specs().summary())
