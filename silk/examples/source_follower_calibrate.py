"""Calibrate a source-follower behavioral model from SPICE.

CalibrationBinding maps the AC gain and bandwidth specs to the model's
gain and bw_hz parameters during reconciliation. A transient measures
the DC output level; calibrate_behavioral() applies the derived offset.
A step-response comparison checks the calibrated model against SPICE.

Run ``MPLBACKEND=Agg .venv/bin/python -m silk.examples.source_follower_calibrate``.
Without arguments, it runs calibration and comparison in one process,
prints parameters, spec results and RMS error, and saves an overlay plot
under example_output_dir(). This phase needs ngspice and sky130 models.

For separate processes, use the characterize and restore subcommands with
``--database follower.db``. Characterize stores observations; restore
rebuilds the factory, selects stored observations and reapplies bindings
without running SPICE. Both default to tt, 27 C and 1.8 V. Restore needs
only the catalog and that database. Optional --validation-output writes
comparison evidence; neither phase reads that file.
See docs/database.md#source-follower-calibration-across-python-processes.
"""
import argparse
import json
import os
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from silk import sources
from silk._json_codec import encode_json
from silk.simulation import Transient
from silk.specs import CalibrationBinding
from silk.metrics import average, error_norm
from silk.signals.base import Signal

from silk.examples.source_follower_example import (
                                                   build_source_follower, GAIN_METRIC, BW_METRIC,
                                                   )
from silk.examples import example_output_dir


_STEP_LO, _STEP_HI = 1.0, 1.4
_STEP_TD = 20e-9
_T_STEP, _T_STOP = 0.2e-9, 120e-9
_I_REF = 20e-6


def _attach_calibration_bindings(mod, specs):
    """Bind the authored AC specs to behavioral gain and bandwidth parameters."""
    for spec in specs:
        if spec.metric == GAIN_METRIC:
            spec.calibration = CalibrationBinding(target_module=mod, param_name="gain")
        elif spec.metric == BW_METRIC:
            spec.calibration = CalibrationBinding(target_module=mod, param_name="bw_hz")


def _step_test(sfmod, *, parameters=None):
    """Register one VIN step rig to run on both SPICE and behavioral backends."""
    tb = sfmod.create_test("beh_check", parameters=parameters)
    tb.add_module_under_test({
        "VDD": "VDD", "VSS": "0", "VIN": "vin", "VOUT": "vout", "IBIAS": "ibias"})
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0", dc=1.8))
    tb.add_source(sources.CurrentSource("iref", positive="VDD", negative="ibias", dc=_I_REF))
    tb.add_source(sources.VoltageSource(
        "vin", positive="vin", negative="0",
        pulse=sources.Pulse(v1=_STEP_LO, v2=_STEP_HI, td=_STEP_TD,
                            tr=1e-9, tf=1e-9, pw=60e-9, per=200e-9)))
    tb.add_sim_type(Transient(step=_T_STEP, stop=_T_STOP))
    return tb


def calibrate(corner="tt"):
    params = {}
    specs = []
    mod = build_source_follower(i_ref=_I_REF, model_params=params, specs=specs)
    before = dict(params)

    _attach_calibration_bindings(mod, specs)
    mod.run_all_tests(corner=corner)
    recon = mod.reconcile_specs()

    tb = _step_test(mod)
    tb.netlist.simulator = "ngspice"
    with redirect_stdout(StringIO()):
        tb.run(corner=corner)
    spice = tb.netlist.load_signal("vout")
    t_sp, vout_sp = np.array(spice.x), np.array(spice.y)

    # Measure the low output level before the input step.
    vout_dc = average(spice, start=_STEP_TD * 0.5, end=_STEP_TD * 0.95)
    offset = params["gain"] * _STEP_LO - vout_dc
    mod.calibrate_behavioral("offset", offset)

    after = dict(params)

    tb.netlist.simulator = "behavioral"

    # Suppress console output for the metric-free behavioral run.
    with redirect_stdout(StringIO()):
        tb.run()
    behavioral = tb.netlist.load_signal("vout")
    t_b, vout_b = np.array(behavioral.x), np.array(behavioral.y)

    return mod, recon, before, after, (t_sp, vout_sp), (t_b, vout_b)


_TEMPERATURE, _SUPPLY = 27.0, 1.8
_SETTLED_METRIC = "settled_vout_V"


def _persistent_recipe(db_path, *, model_params=None):
    """Rebuild the AC and settled-output declarations and calibration bindings."""
    specs = []
    mod = build_source_follower(i_ref=_I_REF, db_path=str(db_path),
                                model_params=model_params, specs=specs)
    _attach_calibration_bindings(mod, specs)
    tb = _step_test(mod, parameters={
        "input_low_V": _STEP_LO, "settled_start_s": _STEP_TD * 0.5,
        "settled_end_s": _STEP_TD * 0.95})
    tb.add_metric(_SETTLED_METRIC, average, sig=tb.probe("vout"),
                  start=_STEP_TD * 0.5, end=_STEP_TD * 0.95)
    for test in (mod.get_test("ac"), tb):
        test.set_temperature(_TEMPERATURE)
        test.set_supply_voltage(_SUPPLY, source="vdd")
    return mod


def _validate_behavioral(mod):
    """Run the step bench with the calibrated gain, bandwidth and offset."""
    tb = mod.get_test("beh_check")

    # Ambient temperature is a SPICE condition, not a behavioral solve.
    tb.set_temperature(None)
    tb.netlist.simulator = "behavioral"
    # Suppress console output for the behavioral validation run.
    with redirect_stdout(StringIO()):
        tb.run()
    signal = tb.netlist.load_signal("vout")
    return np.array(signal.x), np.array(signal.y)


def characterize(db_path, corner="tt", *, model_params=None):
    """Run and store AC and settled transient evidence, then validate the twin.

    Only observations are stored: offset is derived as gain * 1 V - settled
    VOUT. The transient window is [10 ns, 19 ns], before the input step.
    No fitted Python object or parameter sidecar is stored.
    """
    params = {} if model_params is None else model_params
    mod = _persistent_recipe(db_path, model_params=params)
    try:
        mod.create_circuit_version()
        mod.run_test("ac", corner=corner, store=True, store_options={
            "sim_type": "ac", "units": {GAIN_METRIC: "V/V", BW_METRIC: "Hz"}})
        mod.reconcile_specs()
        report = mod.run_test("beh_check", corner=corner, store=True, store_options={
            "sim_type": "tran", "units": {_SETTLED_METRIC: "V"},
            "conditions": {"input_low_V": _STEP_LO,
                           "settled_window_s": [_STEP_TD * 0.5, _STEP_TD * 0.95],
                           "offset_derivation": "gain * input_low_V - settled_vout_V"}})
        mod.calibrate_behavioral("offset", params["gain"] * _STEP_LO
                                 - report.metrics[_SETTLED_METRIC])
        behavioral = _validate_behavioral(mod)
        return mod, behavioral
    finally:
        mod.close()


def restore(db_path, corner="tt", *, model_params=None):
    """Restore selected observations and apply bindings without running SPICE.

    Public evidence APIs check structure and producing-deck compatibility.
    Unchecked compatibility conditions remain visible as warnings.
    """
    from silk.db.context import CharacterizationContext
    from silk.exceptions import EvidenceAmbiguityError, EvidenceMismatchError

    if not Path(db_path).is_file():
        raise FileNotFoundError(f"No evidence database at {db_path}; run characterize first")
    params = {} if model_params is None else model_params
    mod = _persistent_recipe(db_path, model_params=params)
    try:
        mod.create_circuit_version()
        for name in ("ac", "beh_check"):
            if name not in mod.list_tests():
                raise EvidenceMismatchError(f"Producing test {name!r} is missing; rebuild the recipe")
        profiles = {}
        for name in ("ac", "beh_check"):
            context = CharacterizationContext(
                owner_path="$", test_name=name, parameters=mod.get_test(name).parameters, corner=corner,
                temperature=_TEMPERATURE, voltage=_SUPPLY)
            profiles[name] = mod.resolve_characterization_profile(context=context).profile_id
        observations = {}
        for metric, test, unit in ((GAIN_METRIC, "ac", "V/V"),
                                   (BW_METRIC, "ac", "Hz"),
                                   (_SETTLED_METRIC, "beh_check", "V")):
            rows = mod.query_results(
                parameter=metric, profile_id=profiles[test], source="spice",
                corner=corner, temperature=_TEMPERATURE, voltage=_SUPPLY,
                latest_only=True)
            if len(rows) != 1:
                raise EvidenceAmbiguityError(f"Expected one {metric} observation; found {len(rows)}")
            row = rows[0]
            if row.unit != unit or row.value is None or not np.isfinite(row.value):
                raise EvidenceMismatchError(f"Invalid {metric} evidence; characterize again")
            observations[metric] = row.value
            print(f"{test}/{metric}: profile={row.profile_id}, "
                  f"deck={row.deck_check.state} ({row.deck_check.reason})")
        count = mod.restore_spec_measurements(
            profile_id=profiles["ac"], source="spice", corner=corner,
            temperature=_TEMPERATURE, voltage=_SUPPLY)
        if count != 2:
            raise EvidenceMismatchError(f"Expected two AC spec observations; restored {count}")
        mod.reconcile_specs()
        for metric, param in ((GAIN_METRIC, "gain"), (BW_METRIC, "bw_hz")):
            if params[param] != observations[metric]:
                raise EvidenceMismatchError(f"Binding for {param} did not apply selected {metric}")
        mod.calibrate_behavioral("offset", params["gain"] * _STEP_LO
                                 - observations[_SETTLED_METRIC])
        return mod, _validate_behavioral(mod)
    finally:
        mod.close()


def _persistence_cli(args):
    """Run one persistence phase and optionally write comparison evidence."""
    operation = characterize if args.phase == "characterize" else restore
    params = {}
    mod, (t, vout) = operation(args.database, corner=args.corner, model_params=params)
    print(f"{args.phase}: {args.database}, {args.corner}, 27 C, 1.8 V")
    print(json.dumps(encode_json(params), allow_nan=False, sort_keys=True))
    if args.validation_output:
        Path(args.validation_output).write_text(json.dumps(encode_json({
            "params": params, "time_s": t.tolist(), "vout_V": vout.tolist()}), allow_nan=False))


def _plot_overlay(spice, behavioral, png_path):
    (t_sp, v_sp), (t_b, v_b) = spice, behavioral
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t_sp * 1e9, v_sp, color="steelblue", lw=1.8, label="SPICE (sky130 tt)")
    ax.plot(t_b * 1e9, v_b, color="crimson", lw=1.2, ls="--",
            label="behavioral (calibrated)")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("VOUT (V)")
    ax.set_title("Source follower — calibrated behavioral vs SPICE step response")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", nargs="?", choices=("characterize", "restore"))
    parser.add_argument("--database", help="Explicit SQLite evidence location (required for phases)")
    parser.add_argument("--corner", help="Process corner for characterize/restore (default: tt)")
    parser.add_argument("--validation-output", help="Optional JSON comparison output; never read on restore")
    args = parser.parse_args()
    if args.phase is None and any(value is not None for value in (
            args.database, args.corner, args.validation_output)):
        parser.error("--database, --corner and --validation-output require characterize or restore")
    if args.corner is None:
        args.corner = "tt"
    if args.phase:
        if not args.database:
            parser.error("characterize/restore require --database")
        _persistence_cli(args)
        raise SystemExit(0)
    mod, recon, before, after, spice, behavioral = calibrate()
    print("Behavioral params — before → after SPICE calibration:")
    for k in ("gain", "offset", "bw_hz"):
        print(f"  {k:8s}: {before[k]:.5g}  →  {after[k]:.5g}")
    print()
    print(recon.summary())

    t_sp, v_sp = spice
    t_b, v_b = behavioral
    rms = error_norm(
        Signal("vout", "voltage", "time", "V", t_sp, v_sp),
        Signal("vout", "voltage", "time", "V", t_b, v_b), kind="rms")
    print(f"\nbehavioral-vs-SPICE RMS error: {rms*1e3:.2f} mV")

    out_dir = example_output_dir()
    png = os.path.join(out_dir, "source_follower_calibration.png")
    fig = _plot_overlay(spice, behavioral, png)
    plt.close(fig)
    print(f"Overlay plot saved -> {png}")
