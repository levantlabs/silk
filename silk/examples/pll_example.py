"""PLL built from a behavioral controller, LDO and ring VCO.

The controller combines the PFD, charge pump, loop filter and /24 divider.
The LDO and ring use their stored calibration parameters. The loop-filter
design prints its component values and subtracts LDO and sampling delay
from the phase margin.

The lock test measures cold-start settling, tail frequency, control voltage,
phase offset and ripple. The script repeats the run at half the time step
and plots the control voltages and frequency averaged over 50 cycles.
The spur estimate uses the reference-frequency component at the VCO supply.

The controller, LDO and ring each run once per time point.
The model has no device noise. Feedback is delayed by one time point.
The controller detects REF and OUT crossings at 0.9 V. Resolve both clock levels and
keep the step below the PFD reset width. The half-step frequency comparison
does not establish convergence of phase error or ripple.

Run from the repository root:
    MPLBACKEND=Agg .venv/bin/python -m silk.examples.pll_example

No external simulator or installed PDK models are needed.
"""
import os

import numpy as np

from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk import load_pdk
from silk import sources
from silk import analysis
from silk.exceptions import NotSupportedError, SimulationError
from silk.simulation import Transient
from silk.metrics import (average_frequency, average, peak_to_peak, settling_time,
                          crossings, harmonic)
from silk.signals.spice import NgspiceSignal

from silk.examples.ldo_regulator_example import build_ldo
from silk.examples.ring_oscillator_example import build_ring_oscillator
from silk.examples.pfd_example import T_RESET_NOM_PS
from silk.examples.charge_pump_example import (I_UP_NOM_UA, I_DN_NOM_UA, LEAK_NOM_NA,
                                               VCOMP_LO_NOM_V, VCOMP_HI_NOM_V)
from silk.examples import example_output_dir

pdk = load_pdk('sky130')


F_REF = 25e6
N_DIV = 24
F_TARGET = F_REF * N_DIV
F_BW = 500e3
PM_TARGET = 45.0
I_CP = 10e-6
_GAMMA = 4.0

_VDD_CP = 1.8


_LDO_P = dict(build_ldo(with_tests=False)._behavioral_params)
_RING_P = dict(build_ring_oscillator(with_tests=False)._behavioral_params)


_PFD_CP_P = dict(
    t_reset_s=T_RESET_NOM_PS * 1e-12,
    i_up_a=I_UP_NOM_UA * 1e-6,
    i_dn_a=I_DN_NOM_UA * 1e-6,
    i_leak_a=LEAK_NOM_NA * 1e-9,
    v_comp_lo=VCOMP_LO_NOM_V,
    v_comp_hi=VCOMP_HI_NOM_V,
)


LOCKTIME_METRIC = "lock_time_us"
FERR_METRIC = "f_err_pct"
VCTRL_METRIC = "locked_vctrl_v"
PHERR_METRIC = "static_phase_err_ps"
LOCKTIME_MAX_US = 40.0
FERR_MAX_PCT = 0.2
PHERR_MAX_PS = 50.0


def design_loop_filter(*, f_bw=F_BW, i_cp=I_CP, kvco_hz_per_v=None, n_div=N_DIV,
                       gamma=_GAMMA, verbose=True):
    """Compute a type-II R/C filter and its estimated phase margin.

    Place the zero at wc/gamma and the pole at gamma*wc, then scale
    impedance for unity loop gain at wc. Subtract LDO phase lag and
    half a reference period of sampling delay. Return component values,
    bandwidth, phase margin and acquisition time constant in a dict.
    Raise ValueError if the margin is below PM_TARGET.
    """
    kv = float(kvco_hz_per_v if kvco_hz_per_v is not None
               else _RING_P["kvco_mhz_per_v"] * 1e6)
    wc = 2 * np.pi * f_bw

    # Place the zero and pole, then scale impedance for unity gain at wc.
    rc1 = gamma / wc
    def z_of(c1):
        c2 = c1 / (gamma ** 2 - 1.0)
        r = rc1 / c1
        cp = c1 * c2 / (c1 + c2)
        s = 1j * wc
        return (1 + s * r * c1) / (s * (c1 + c2) * (1 + s * r * cp)), r, c2
    z_unit, _, _ = z_of(1.0)
    z_needed = n_div * wc / (i_cp * kv)
    c1 = abs(z_unit) / z_needed
    z, r, c2 = z_of(c1)

    # Include the VCO integrator before subtracting LDO and sampling lag.
    pm_filter = 180.0 + np.degrees(np.angle(z * 1.0 / (1j * wc)))
    fn = _LDO_P["bw_hz"]
    zeta = _LDO_P["zeta"]
    ratio = f_bw / fn
    ldo_lag = np.degrees(np.arctan2(2 * zeta * ratio, 1 - ratio ** 2))
    pfd_lag = np.degrees(wc * (0.5 / F_REF))
    pm = pm_filter - ldo_lag - pfd_lag

    if verbose:
        print(f"  loop filter design (f_bw = {f_bw/1e3:.0f} kHz, "
              f"I_cp = {i_cp*1e6:.0f} uA, N = {n_div}, "
              f"K_vco = {kv/1e6:.0f} MHz/V):")
        print(f"    R  = {r:8.0f} ohm")
        print(f"    C1 = {c1*1e12:8.1f} pF   (zero at {f_bw/gamma/1e3:.0f} kHz)")
        print(f"    C2 = {c2*1e12:8.1f} pF   (pole at {f_bw*gamma/1e3:.0f} kHz)")
        print(f"    phase margin: filter {pm_filter:5.1f} deg"
              f"  - LDO pole {ldo_lag:4.1f} deg (f_n {fn/1e6:.1f} MHz, measured)"
              f"  - PFD delay {pfd_lag:4.1f} deg")
        print(f"                = {pm:5.1f} deg")

    # Cycle-slipping pull-in estimate: tau = N*f_ref*(C1+C2)/(I_cp*Kvco).
    tau_acq = n_div * F_REF * (c1 + c2) / (i_cp * kv)
    if verbose:
        print(f"    pull-in tau ~ {tau_acq*1e6:.1f} us  ->  expect cold-start "
              f"lock ~{3*tau_acq*1e6:.0f} us")
    if pm < PM_TARGET:
        raise ValueError(f"loop design yields PM {pm:.1f} deg < {PM_TARGET} — "
                         f"reduce f_bw to increase phase margin")
    return {"r_ohm": float(r), "c1_f": float(c1), "c2_f": float(c2),
            "i_cp_a": float(i_cp), "f_bw_hz": float(f_bw), "pm_deg": float(pm),
            "tau_acq_s": float(tau_acq)}


class _PLLController:
    """Sampled PFD, charge pump, filter and clock-edge divider.

    Read OUT one time point back and interpolate its 0.9 V crossings.
    Reset state at transient time zero; refuse other analyses and steps
    as long as the reset pulse.
    """

    def __call__(self, ports, t, params):

        if analysis() != 'transient':
            raise NotSupportedError('The PLL feedback controller requires transient simulation.')
        clock = float(ports.CLK.v.previous_value)
        ref = float(ports.REF.v)
        if t == 0:
            self.up_on = self.dn_on = self.off_t = float('inf')
            self.pherr = 0.0
            self.v_ctrl = self.v_c1 = float(params.v_ctrl0)
            self.count = 0
            self.ref = ref
            self.clock_sample = None
        else:
            t0 = self.time
            dt = float(t - t0)
            if dt >= float(params.pfd_t_reset_s):
                raise SimulationError('PLL time step must be smaller than the PFD reset pulse width.')
            up_new = dn_new = False
            inf = float('inf')
            if ref > 0.9 >= self.ref and self.up_on == inf and self.off_t == inf:
                self.up_on = t0 + (0.9 - self.ref) / (ref - self.ref) * dt
                up_new = True
            self.ref = ref

            # The delayed clock sample belongs to t0, the previous time point.
            if self.clock_sample is not None:
                prev_time, prev_value = self.clock_sample
                if clock > 0.9 >= prev_value:
                    self.count += 1
                    if self.count == int(params.n_div):
                        self.count = 0
                        cross = prev_time + (0.9 - prev_value) / (clock - prev_value) * (t0 - prev_time)
                        if self.dn_on == inf and self.off_t == inf:
                            self.dn_on = cross
                            dn_new = True
            self.clock_sample = (t0, clock)
            if self.up_on < inf and self.dn_on < inf and self.off_t == inf:
                self.off_t = max(self.up_on, self.dn_on) + float(params.pfd_t_reset_s)
                self.pherr = self.dn_on - self.up_on
            q = -float(params.cp_i_leak_a) * dt
            if self.up_on < inf or self.dn_on < inf:
                b = min(self.off_t, float(t))
                if self.up_on < inf and self.v_ctrl < float(params.cp_v_comp_hi):
                    a = self.up_on if up_new else t0
                    if b > a:
                        q += float(params.cp_i_up_a) * (b - a)
                if self.dn_on < inf and self.v_ctrl > float(params.cp_v_comp_lo):
                    a = self.dn_on if dn_new else t0
                    if b > a:
                        q -= float(params.cp_i_dn_a) * (b - a)
                if self.off_t <= t:
                    self.up_on = self.dn_on = self.off_t = inf
            i_r = (self.v_ctrl - self.v_c1) / float(params.r_ohm)
            self.v_c1 += i_r / float(params.c1_f) * dt
            self.v_ctrl += (q - i_r * dt) / float(params.c2_f)
            self.v_ctrl = min(max(self.v_ctrl, 0.0), float(params.vdd_cp))
        self.time = float(t)
        ports.VCTRL.v = self.v_ctrl
        ports.PHERR.v = self.pherr * 1e12


def build_pll(*, filt=None, n_div=N_DIV, vdd_cp=_VDD_CP, pfd_cp=None,
              v_init=None) -> Module:
    """Compose the controller, calibrated LDO and ring VCO.

    pfd_cp overrides pump and reset constants. v_init sets the filter
    capacitors and initial LDO output; None starts them at zero.
    """
    filt = filt or design_loop_filter(verbose=False)
    nid = dict(_PFD_CP_P if pfd_cp is None else pfd_cp)
    mod = Module('PDK_skywater130', example_output_dir(), 'pll.cir', pdk_details=pdk,
                 module_name='pll')
    mod.define_module_ports({
        'REF': Port.analog_input(), 'OUT': Port.analog_output(),
        'VCTRL': Port.analog_output(), 'VREG': Port.analog_output(),
        'PHERR': Port.analog_output(),
    })
    controller = Module('PDK_skywater130', example_output_dir(), 'pll_controller.cir',
                        pdk_details=pdk, module_name='pll_controller')
    controller.define_module_ports({
        'REF': Port.analog_input(), 'CLK': Port.analog_input(),
        'VCTRL': Port.analog_output(), 'PHERR': Port.analog_output(),
    })
    controller.set_behavioral_model(_PLLController(), execution='sample', params={
        'r_ohm': filt['r_ohm'], 'c1_f': filt['c1_f'], 'c2_f': filt['c2_f'],
        'n_div': n_div, 'vdd_cp': vdd_cp, 'v_ctrl0': 0.0 if v_init is None else v_init,
        'pfd_t_reset_s': nid['t_reset_s'], 'cp_i_up_a': nid['i_up_a'],
        'cp_i_dn_a': nid['i_dn_a'], 'cp_i_leak_a': nid['i_leak_a'],
        'cp_v_comp_lo': nid['v_comp_lo'], 'cp_v_comp_hi': nid['v_comp_hi'],
    })
    mod.add_module(controller, 'controller', nets={
        'REF': 'REF', 'CLK': 'OUT', 'VCTRL': 'VCTRL', 'PHERR': 'PHERR'})
    ldo = build_ldo(with_tests=False, initial_output=0.0 if v_init is None else v_init)
    for name, value in _LDO_P.items():
        ldo.calibrate_behavioral(name, value)
    ldo.calibrate_behavioral('initial_output', 0.0 if v_init is None else v_init)
    mod.add_module(ldo, 'ldo', nets={
        'VDD': 'ldo_vdd', 'VSS': '0', 'VCTRL': 'VCTRL', 'VREG': 'VREG',
        'IBIAS_AMP': 'ldo_bias'})
    ring = build_ring_oscillator(with_tests=False)
    for name, value in _RING_P.items():
        ring.calibrate_behavioral(name, value)
    mod.add_module(ring, 'ring', nets={
        'VDD': 'VREG', 'VSS': '0', 'EN': 'ring_en', 'OUT': 'OUT'})
    mod.add_source(sources.VoltageSource('ldo_supply', positive='ldo_vdd', negative='0',
                                        dc=float(_LDO_P['vdd_nom'])))
    mod.add_source(sources.CurrentSource('ldo_bias', positive='ldo_bias', negative='0', dc=20e-6))
    mod.add_source(sources.VoltageSource('ring_enable', positive='ring_en', negative='0', dc=1.8))
    return mod


def _locked_frequency(sig, *, start):
    """Measure OUT frequency using its observed tail mid-swing threshold."""
    return average_frequency(sig, start=start)


def _lock_time(sig, *, start):
    """Return settling time within 5 mV of the tail mean, in seconds."""
    target = average(sig, start=start)
    initial = float(sig.y[0])

    # Convert the absolute 5 mV tolerance to a fraction of the total swing.
    return settling_time(sig, initial=initial, target=target,
                         tol=5e-3 / abs(target - initial))


def attach_lock_test(mod, *, t_stop=80e-6, t_step=0.2e-9):
    """Register the behavioral transient and its lock metrics."""
    tb = mod.create_test(f"lock_{len(mod.list_tests())}")
    tb.add_module_under_test({
        "REF": "ref", "OUT": "out", "VCTRL": "vctrl", "VREG": "vreg",
        "PHERR": "pherr"})
    tb.netlist.simulator = "behavioral"

    tb.add_source(sources.VoltageSource(
        "ref", positive="ref", negative="0", dc=0,
        pulse=sources.Pulse(v1=0, v2=1.8, td=0, tr=1e-9, tf=1e-9,
                            pw=0.5 / F_REF, per=1.0 / F_REF)))
    tb.add_sim_type(Transient(step=t_step, stop=t_stop))
    start = 0.8 * t_stop
    frequency = tb.add_metric("f_locked", _locked_frequency,
                              sig=tb.probe("out"), start=start)
    error = tb.add_metric("frequency_error_pct", (frequency - F_TARGET) / F_TARGET * 100)
    tb.add_metric(FERR_METRIC, abs(error))
    lock = tb.add_metric("lock_time_s", _lock_time,
                         sig=tb.probe("vctrl"), start=start)
    tb.add_metric(LOCKTIME_METRIC, lock * 1e6)
    tb.add_metric(VCTRL_METRIC, average, sig=tb.probe("vctrl"), start=start)
    phase = tb.add_metric("pherr_ps", average, sig=tb.probe("pherr"), start=start)
    tb.add_metric(PHERR_METRIC, abs(phase))
    ripple = tb.add_metric("ripple_vpp", peak_to_peak,
                           sig=tb.probe("vctrl"), start=start)
    tb.add_metric("ripple_uvpp", ripple * 1e6)
    return tb


def add_lock_specs(tb):
    """Keep the spec handles returned by the testbench declarations."""
    specs = (
        tb.add_spec(f"lock time < {LOCKTIME_MAX_US:.0f} us", LOCKTIME_METRIC,
                    "<", LOCKTIME_MAX_US),
        tb.add_spec(f"|f error| < {FERR_MAX_PCT}%", FERR_METRIC, "<", FERR_MAX_PCT),
        tb.add_spec("locked VCTRL > ring floor", VCTRL_METRIC, ">", 1.0),
        tb.add_spec(f"|static phase err| < {PHERR_MAX_PS:.0f} ps", PHERR_METRIC,
                    "<", PHERR_MAX_PS),
    )
    for spec, units in zip(specs, ("us", "%", "V", "ps")):
        spec.units = units
    return {spec.metric: spec for spec in specs}


def run_lock(mod, *, t_stop=80e-6, t_step=0.2e-9, testbench=None):
    """Run a behavioral lock test; return waveforms and registered metrics."""
    tb = testbench if testbench is not None else attach_lock_test(
        mod, t_stop=t_stop, t_step=t_step)
    report = tb.run()
    signals = {name: tb.netlist.load_signal(name)
               for name in ("vctrl", "vreg", "out")}
    metrics = report.metrics
    return {"t": signals["out"].x,
            **{name: sig.y for name, sig in signals.items()},
            "f_locked": metrics["f_locked"], "f_err_pct": metrics["frequency_error_pct"],
            "lock_time_us": metrics[LOCKTIME_METRIC],
            "v_ctrl_locked": metrics[VCTRL_METRIC],
            "pherr_ps": metrics["pherr_ps"], "ripple_uvpp": metrics["ripple_uvpp"]}


def save_lock_plot(res, path, *, filt):
    """Plot VCTRL, VREG and frequency averaged over 50-cycle windows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, out = res["t"], res["out"]
    th = 0.5 * float(out.max())
    waveform = NgspiceSignal(name="out", type="voltage", xlabel="time", ylabel="V",
                              xvalues=t, yvalues=out)
    te = crossings(waveform, threshold=th, edge="rising")
    # Frequency averaged over nonoverlapping 50-cycle windows.

    w = 50
    f_inst_t, f_inst = [], []
    for k in range(0, len(te) - w, w):
        span = te[k + w] - te[k]
        f_inst_t.append(te[k + w // 2] * 1e6)
        f_inst.append(w / span / 1e6)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    ax[0].plot(t * 1e6, res["vctrl"], color="#0072B2", lw=1.4, label="VCTRL")
    ax[0].plot(t * 1e6, res["vreg"], color="#D55E00", lw=1.0, label="VREG (LDO out)")
    ax[0].axhline(res["v_ctrl_locked"], color="#5a5a6e", ls="--", lw=0.8)
    ax[0].axhline(1.0, color="#CC79A7", ls=":", lw=1,
                  label="behavioral ring cutoff: 1.0 V")
    ax[0].axvline(res["lock_time_us"], color="#009E73", ls=":", lw=1.2)
    ax[0].text(res["lock_time_us"], 0.15, f" lock {res['lock_time_us']:.1f} µs",
               color="#009E73", fontsize=9)
    ax[0].set_ylabel("V")
    ax[0].set_title(f"PLL cold-start lock — f_ref {F_REF/1e6:.0f} MHz × N={N_DIV} "
                    f"→ {res['f_locked']/1e6:.1f} MHz "
                    f"({res['f_err_pct']:+.3f}%), loop BW {filt['f_bw_hz']/1e3:.0f} kHz, "
                    f"PM {filt['pm_deg']:.0f}°")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")

    ax[1].plot(f_inst_t, f_inst, color="#009E73", lw=1.4)
    ax[1].axhline(F_TARGET / 1e6, color="#5a5a6e", ls="--", lw=0.8,
                  label=f"target {F_TARGET/1e6:.0f} MHz")
    ax[1].set_ylabel("f_VCO (MHz)")
    ax[1].set_xlabel("time (µs)")
    ax[1].legend(frameon=False, fontsize=8, loc="lower right")
    for a in ax:
        a.grid(True, color="#d5d5dd", lw=0.6, alpha=0.6)
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    OUT = example_output_dir()

    print("=" * 72)
    print(f"PLL over the calibrated twins — {F_REF/1e6:.0f} MHz × {N_DIV} "
          f"= {F_TARGET/1e6:.0f} MHz")
    print("=" * 72)

    filt = design_loop_filter()
    mod = build_pll(filt=filt)

    print("\n  cold-start lock transient (behavioral, dt = 0.2 ns, 80 us)...")
    lock_test = attach_lock_test(mod)
    add_lock_specs(lock_test)
    res = run_lock(mod, testbench=lock_test)
    print(f"    locked f   = {res['f_locked']/1e6:9.2f} MHz "
          f"({res['f_err_pct']:+.3f}% of {F_TARGET/1e6:.0f} MHz)")
    print(f"    lock time  = {res['lock_time_us']:9.1f} us "
          f"(pull-in math predicted ~{3*filt['tau_acq_s']*1e6:.0f} us)")
    print(f"    VCTRL      = {res['v_ctrl_locked']:9.3f} V "
          f"(tuning curve predicts "
          f"{(F_TARGET/1e6 - _RING_P['f0_mhz'])/_RING_P['kvco_mhz_per_v']:.3f} V)")

    # Charge balance: i_up*(phase_offset+t_reset) = i_dn*t_reset+i_leak/F_REF.
    pred_ps = ((_PFD_CP_P["i_dn_a"] - _PFD_CP_P["i_up_a"]) * _PFD_CP_P["t_reset_s"]
               + _PFD_CP_P["i_leak_a"] / F_REF) / _PFD_CP_P["i_up_a"] * 1e12
    print(f"    phase err  = {res['pherr_ps']:+9.1f} ps static "
          f"(charge balance over the measured mismatch predicts {pred_ps:+.1f} ps)")
    kv_hz = _RING_P["kvco_mhz_per_v"] * 1e6
    vreg = NgspiceSignal(name="vreg", type="voltage", xlabel="time", ylabel="V",
                         xvalues=res["t"], yvalues=res["vreg"])
    # Sample 400 complete reference periods to isolate the 25 MHz component.
    n_spur = 400 * 200
    vreg_ref_peak = 2 * abs(harmonic(
        vreg, 1, tclk=1 / (200 * F_REF), tstart=res["t"][-1] - 400 / F_REF,
        n_samples=n_spur, fundamental_hz=F_REF)) / n_spur
    spur_dbc = 20 * np.log10(kv_hz * vreg_ref_peak / (2 * F_REF))
    print(f"    VCTRL ripple = {res['ripple_uvpp']:7.1f} uVpp")
    print(f"    VREG reference component = {vreg_ref_peak*1e6:.2f} uV peak "
          f"-> reference spur ~ {spur_dbc:.0f} dBc (narrowband-FM estimate)")

    # Compare frequency at dt/2 without asserting phase or ripple convergence.
    res2 = run_lock(mod, t_step=0.1e-9)
    print(f"    dt/2 check : f {res2['f_locked']/1e6:.2f} MHz "
          f"({res2['f_err_pct']:+.3f}%), lock {res2['lock_time_us']:.1f} us, "
          f"phase err {res2['pherr_ps']:+.1f} ps "
          f"-> {'Frequency agrees at dt/2; phase/ripple convergence not established' if abs(res2['f_err_pct'] - res['f_err_pct']) < 0.05 else 'Frequency changed — refine dt'}")

    png = save_lock_plot(res, os.path.join(OUT, "pll_lock.png"), filt=filt)
    print(f"\n  saved {png}")

    print()
    print(mod.reconcile_specs().summary())
