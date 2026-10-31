"""Measure the PFD and charge-pump constants used by the behavioral PLL.

The PFD tests measure phase response, frequency detection and reset-pulse
width. Charge-pump tests measure current mismatch, compliance and leakage.
The script prints stored and measured constants, percentage drift and
replacement assignments. It then runs the behavioral lock test with each
set and prints frequency, lock time and static phase error.

This script prints measured constants for manual updates; it does not
persist calibration.

Run from the repository root with ngspice and sky130 models installed:
    MPLBACKEND=Agg .venv/bin/python -m silk.examples.pll_calibrate
"""
from silk.examples import pfd_example as pfd_x
from silk.examples import charge_pump_example as cp_x
from silk.examples.pfd_example import build_pfd, PULSE_METRIC
from silk.examples.charge_pump_example import (build_charge_pump, IUP_METRIC, IDN_METRIC,
                                               LEAK_METRIC, DNLO_METRIC, UPHI_METRIC)
from silk.examples.pll_example import build_pll, run_lock, design_loop_filter, F_REF


def measure():
    """Run PFD and charge-pump tests and return measured calibration constants."""
    print("  re-measuring PFD (the dead-zone pulse is the imported one)...")
    pfd = build_pfd()
    pfd_res = pfd.run_all_tests(corner="tt")
    pulse_ps = pfd_res["pfd.deadzone"].metrics[PULSE_METRIC]

    print("  re-measuring charge pump (match / compliance / leakage)...")
    cp = build_charge_pump()
    cp_res = cp.run_all_tests(corner="tt")
    r = lambda t, m: cp_res[f"{cp.module_name}.{t}"].metrics[m]

    return {
        "t_reset_ps": pulse_ps,
        "i_up_ua": r("match", IUP_METRIC),
        "i_dn_ua": r("match", IDN_METRIC),
        "leak_na": r("leakage", LEAK_METRIC),
        "v_comp_lo": r("compl_dn", DNLO_METRIC),
        "v_comp_hi": r("compl_up", UPHI_METRIC),
    }


if __name__ == "__main__":
    print("=" * 72)
    print("PLL non-ideality recalibration — PFD reset pulse + charge pump")
    print("=" * 72)

    meas = measure()

    rows = [

        ("T_RESET_NOM_PS", "pfd_example.py", pfd_x.T_RESET_NOM_PS,
         meas["t_reset_ps"]),
        ("I_UP_NOM_UA", "charge_pump_example.py", cp_x.I_UP_NOM_UA,
         meas["i_up_ua"]),
        ("I_DN_NOM_UA", "charge_pump_example.py", cp_x.I_DN_NOM_UA,
         meas["i_dn_ua"]),
        ("LEAK_NOM_NA", "charge_pump_example.py", cp_x.LEAK_NOM_NA,
         meas["leak_na"]),
        ("VCOMP_LO_NOM_V", "charge_pump_example.py", cp_x.VCOMP_LO_NOM_V,
         meas["v_comp_lo"]),
        ("VCOMP_HI_NOM_V", "charge_pump_example.py", cp_x.VCOMP_HI_NOM_V,
         meas["v_comp_hi"]),
    ]

    print()
    print(f"  {'constant':<16s} {'in-code':>10s} {'measured':>10s} {'drift':>8s}")
    print("  " + "-" * 48)
    for name, _file, nom, m in rows:
        print(f"  {name:<16s} {nom:10.3f} {m:10.3f} "
              f"{(m - nom) / nom * 100:+7.1f}%")

    print()
    print("  Replacement assignments for manual updates:")
    for fname in ("pfd_example.py", "charge_pump_example.py"):
        print(f"    # --- {fname} ---")
        for name, file_, _nom, m in rows:
            if file_ == fname:
                print(f"    {name} = {m:.3f}")

    print()
    print("  validating: PLL lock on in-code vs freshly measured values...")
    filt = design_loop_filter(verbose=False)
    res_nom = run_lock(build_pll(filt=filt))
    fresh = dict(
        t_reset_s=meas["t_reset_ps"] * 1e-12,
        i_up_a=meas["i_up_ua"] * 1e-6,
        i_dn_a=meas["i_dn_ua"] * 1e-6,
        i_leak_a=meas["leak_na"] * 1e-9,
        v_comp_lo=meas["v_comp_lo"],
        v_comp_hi=meas["v_comp_hi"],
    )
    res_new = run_lock(build_pll(filt=filt, pfd_cp=fresh))
    pred_new = ((fresh["i_dn_a"] - fresh["i_up_a"]) * fresh["t_reset_s"]
                + fresh["i_leak_a"] / F_REF) / fresh["i_up_a"] * 1e12

    print(f"    {'':<22s} {'in-code':>10s} {'measured':>10s}")
    for label, key, unit in (("locked f (MHz)", "f_locked", 1e-6),
                             ("lock time (us)", "lock_time_us", 1.0),
                             ("static phase err (ps)", "pherr_ps", 1.0)):
        print(f"    {label:<22s} {res_nom[key] * unit:10.3f} "
              f"{res_new[key] * unit:10.3f}")
    print(f"    (charge balance on the fresh values predicts "
          f"{pred_new:+.2f} ps)")
