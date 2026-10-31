"""Layout estimation, matching, parasitic loading and report comparison.

Build a differential-pair input stage, annotate peak currents, and compare
routing estimates with and without matching constraints. Print estimated
wire dimensions, resistance, capacitance and device areas. Save reports and
compare two device sizings.

The final section runs the 5T OTA's attached AC testbench before and after
routing-load injection. This measures the effect of added loading, not
inter-terminal wire resistance, routed delay or extracted layout accuracy.

Run: python -m silk.examples.layout_intelligence_example
Reports are written under example_output_dir(). Estimation uses the packaged
sky130 catalog. The AC section needs ngspice and installed sky130 models;
it prints a skip message when either dependency is missing.
"""
from __future__ import annotations

import os


from silk import load_pdk
from silk.examples import example_output_dir
from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk.layout import (
    RoutingModel,
    estimate_layout,
    capture_layout,
    inject_parasitics,
)
from silk.circuit.report import generate_report
from silk.circuit.report_diff import diff_reports

# ---------------------------------------------------------------------------
# PDK setup
# ---------------------------------------------------------------------------


def main():
    pdk = load_pdk('sky130')


    # ---------------------------------------------------------------------------
    # 1. Build a differential-pair amplifier
    # ---------------------------------------------------------------------------

    print("=" * 64)
    print("1. Build a differential-pair amplifier (SKY130 NMOS)")
    print("=" * 64)

    _out = example_output_dir()

    amp = Module('PDK_skywater130', _out, 'diff_amp.cir',
                 pdk_details=pdk, module_name='diff_amp')

    # This NMOS input stage exposes its drains and uses only a ground rail.
    _DIFF_AMP_PORTS = {
        'vss': Port.ground(),
        'vin_p': Port.analog_input(), 'vin_n': Port.analog_input(),
        'vout_p': Port.analog_output(), 'vout_n': Port.analog_output(),
    }
    amp.define_module_ports(_DIFF_AMP_PORTS)

    # Left half — drain is the positive output
    amp.add(pdk.nmos1p8('mn1', d='vout_p', g='vin_p', s='tail', b='vss', w=4, l=0.18, m=2))

    # Right half — drain is the negative output
    amp.add(pdk.nmos1p8('mn2', d='vout_n', g='vin_n', s='tail', b='vss', w=4, l=0.18, m=2))

    # Tail current source (NMOS diode-connected)
    amp.add(pdk.nmos1p8('mn_tail', d='tail', g='tail', s='vss', b='vss', w=8, l=0.18, m=2))

    print(f"  Module: {amp.module_name}")
    print(f"  Devices: {list(amp.elem_dict.keys())}")
    print(f"  Nets: {sorted(amp.net_dict.keys())}")


    # ---------------------------------------------------------------------------
    # 2. Configure a RoutingModel
    # ---------------------------------------------------------------------------

    print()
    print("=" * 64)
    print("2. Configure a RoutingModel (sky130 M1 metal layer)")
    print("=" * 64)

    # The PDK manifest declares the metal stack's routing values, so the
    # model comes straight from it. To use your own numbers, construct
    # RoutingModel(sheet_resistance=..., cap_per_um2=..., j_max=...,
    # min_width=..., max_width=...) directly (step 10 does).
    rm = RoutingModel.from_pdk(pdk)

    print(f"  RoutingModel id: {rm.id}")
    print(f"  j_max = {rm.j_max*1e3:.1f} mA/μm  "
          f"  sheet_R = {rm.sheet_resistance} Ω/sq")


    # ---------------------------------------------------------------------------
    # 3. Annotate nets with peak_current
    #    (in a real flow this comes from SPICE or behavioral sim)
    # ---------------------------------------------------------------------------

    print()
    print("=" * 64)
    print("3. Annotate peak currents on signal nets")
    print("=" * 64)

    amp.net_dict['vout_p'].peak_current = 2.1e-3   # 2.1 mA (slightly asymmetric)
    amp.net_dict['vout_n'].peak_current = 1.8e-3   # 1.8 mA
    amp.net_dict['tail'].peak_current   = 4.0e-3   # tail draws both halves

    print("  vout_p.peak_current = 2.1 mA")
    print("  vout_n.peak_current = 1.8 mA")
    print("  tail.peak_current   = 4.0 mA")
    print("  (vin_p, vin_n, vss — no current annotation → will use min_width)")


    # ---------------------------------------------------------------------------
    # 4. First estimate — no symmetry constraints
    # ---------------------------------------------------------------------------

    print()
    print("=" * 64)
    print("4. estimate_layout — no HalfCircuit declared yet")
    print("=" * 64)

    est_asymmetric = estimate_layout(amp, rm)

    print(f"  Mode:            {est_asymmetric.computation_mode}")
    print(f"  Routing model:   {est_asymmetric.routing_model_id}")
    print(f"  Module bbox:     {amp.bounding_box[0]:.2f} × {amp.bounding_box[1]:.2f} μm")
    print()
    print(f"  {'Net':<10}  {'Length':>10}  {'Width':>7}  {'R':>8}  {'C':>8}")
    print(f"  {'':-<10}  {'':-<10}  {'':-<7}  {'':-<8}  {'':-<8}")
    for name, nl in sorted(est_asymmetric.nets.items()):
        print(f"  {name:<10}  {nl.length:>9.2f}μ  {nl.width:>6.3f}μ  "
              f"{nl.r:>7.2f}Ω  {nl.c:>7.2f}f")

    # Note the asymmetry: vout_p width > vout_n width (2.1 mA vs 1.8 mA)
    w_p = est_asymmetric.nets['vout_p'].width
    w_n = est_asymmetric.nets['vout_n'].width
    print()
    print(f"  vout_p width = {w_p:.3f} μm  (driven by 2.1 mA)")
    print(f"  vout_n width = {w_n:.3f} μm  (driven by 1.8 mA)")
    print(f"  → Asymmetric by {abs(w_p - w_n)*1000:.1f} nm "
          "(unequal current annotations give unequal estimated widths)")

    if est_asymmetric.warnings:
        print()
        print("  Warnings:")
        for w in est_asymmetric.warnings:
            net_tag = f" [{w.net_name}]" if w.net_name else ""
            print(f"    {w.severity.upper()} {w.reason}{net_tag}")


    # ---------------------------------------------------------------------------
    # 5. Declare a HalfCircuit and re-estimate with symmetry enforcement
    # ---------------------------------------------------------------------------

    print()
    print("=" * 64)
    print("5. Declare a HalfCircuit — enforce symmetry on the matched pair")
    print("=" * 64)

    # declare_match_pair is shorthand for a single-pair HalfCircuit declaration
    amp.declare_match_pair('diff_pair', 'mn1', 'mn2')
    print(f"  Declared: {amp.half_circuits['diff_pair']}")

    est = estimate_layout(amp, rm)

    w_p2 = est.nets['vout_p'].width
    w_n2 = est.nets['vout_n'].width
    r_p2 = est.nets['vout_p'].r
    r_n2 = est.nets['vout_n'].r
    c_p2 = est.nets['vout_p'].c
    c_n2 = est.nets['vout_n'].c

    print()
    print(f"  After symmetrization (max-across-members canonical inputs):")
    print(f"    vout_p: width={w_p2:.3f}μm  R={r_p2:.3f}Ω  C={c_p2:.3f}fF")
    print(f"    vout_n: width={w_n2:.3f}μm  R={r_n2:.3f}Ω  C={c_n2:.3f}fF")
    print()
    print(f"  Width difference: {abs(w_p2-w_n2)*1000:.1f} nm  "
          f"(should be 0 — both scaled to the max 2.1 mA)")

    eq_classes = est.equivalence_classes
    print()
    print(f"  Equivalence classes (matched net pairs):")
    for cls_id, members in eq_classes.items():
        print(f"    {cls_id!r:>20s}  →  {members}")


    # ---------------------------------------------------------------------------
    # 6. Inspect per-device areas and bounding box
    # ---------------------------------------------------------------------------

    print()
    print("=" * 64)
    print("6. Per-device area estimates and module bounding box")
    print("=" * 64)

    total_device_area = 0.0
    for name, dl in sorted(est.devices.items()):
        entry = amp.elem_dict[name]
        dev   = entry['device']
        print(f"  {name:<10s}  area = {dl.area:6.2f} μm²  "
              f"(estimated_area on device = {dev.estimated_area:.2f})")
        total_device_area += dl.area

    bbox = est.bounding_box
    bbox_area = bbox[0] * bbox[1]
    print()
    print(f"  Total device area : {total_device_area:.2f} μm²")
    print(f"  Module bbox       : {bbox[0]:.2f} × {bbox[1]:.2f} μm  "
          f"(area = {bbox_area:.2f} μm²,  routing overhead = 2×)")


    # ---------------------------------------------------------------------------
    # 7. inject_parasitics — two modes
    # ---------------------------------------------------------------------------

    print()
    print("=" * 64)
    print("7. inject_parasitics — rc (lumped-π loading) mode")
    print("=" * 64)

    inject_parasitics(amp, est, mode='rc')

    par_nets     = [k for k in amp.net_dict     if k.startswith('__par_')]
    par_resistors = [k for k in amp.passives_dict if k.startswith('__par_R_')]
    par_caps     = [k for k in amp.passives_dict if k.startswith('__par_C')]

    print(f"  Midpoint nets added : {len(par_nets)}")
    print(f"  Loading-branch R    : {len(par_resistors)}")
    print(f"  Shunt capacitors    : {len(par_caps)}  (C1 + C2 per net)")

    # Show the vout_p π-section
    r_vout_p  = amp.passives_dict.get('__par_R_vout_p')
    c1_vout_p = amp.passives_dict.get('__par_C1_vout_p')
    c2_vout_p = amp.passives_dict.get('__par_C2_vout_p')
    if r_vout_p and c1_vout_p and c2_vout_p:
        print()
        print(f"  vout_p loading branch (original terminals stay on vout_p):")
        print(f"    {r_vout_p.name:<20s}  {r_vout_p.nets['p']} → {r_vout_p.nets['n']}   "
              f"R = {r_vout_p.value:.4f} Ω")
        print(f"    {c1_vout_p.name:<20s}  {c1_vout_p.nets['p']} → {c1_vout_p.nets['n']}  "
              f"C = {c1_vout_p.value*1e15:.4f} fF")
        print(f"    {c2_vout_p.name:<20s}  {c2_vout_p.nets['p']} → {c2_vout_p.nets['n']}  "
              f"C = {c2_vout_p.value*1e15:.4f} fF")

    print()
    print("  Switch to c_only mode (removes rc elements first):")
    inject_parasitics(amp, est, mode='c_only')

    par_nets_after   = [k for k in amp.net_dict     if k.startswith('__par_')]
    par_rs_after     = [k for k in amp.passives_dict if k.startswith('__par_R_')]
    par_caps_after   = [k for k in amp.passives_dict if k.startswith('__par_C_')]

    print(f"  Midpoint nets remaining  : {len(par_nets_after)}  (rc nets cleaned up)")
    print(f"  Loading-branch R         : {len(par_rs_after)}")
    print(f"  Shunt capacitors         : {len(par_caps_after)}  (one per net)")


    # ---------------------------------------------------------------------------
    # 8. generate_report with layout_estimate
    # ---------------------------------------------------------------------------

    print()
    print("=" * 64)
    print("8. generate_report with layout_estimate — layout section in Markdown")
    print("=" * 64)

    report = generate_report(amp, layout_estimate=est, auto_discover_config=False)

    md_path = os.path.join(_out, 'diff_amp_report.md')
    report.save(md_path)
    with open(md_path, encoding='utf-8') as f:
        md = f.read()
    # Find and print just the layout section
    layout_start = md.find('## Layout Estimate')
    layout_end   = md.find('\n---\n', layout_start)
    if layout_start != -1:
        layout_md = md[layout_start:layout_end if layout_end != -1 else layout_start+800]
        print()
        for line in layout_md.splitlines()[:28]:
            print(f"  {line}")
        print("  ...")

    # Save a JSON report
    json_path = os.path.join(_out, 'diff_amp_report.json')
    report.save_json(json_path)
    print()
    print(f"  JSON report saved to: {json_path}")

    import json as _json
    with open(json_path) as f:
        d = _json.load(f)
    layout_json = d.get('layout', {})
    print(f"  JSON layout keys: {sorted(layout_json.keys())}")


    # ---------------------------------------------------------------------------
    # 9. capture_layout + diff_reports between two sizing runs
    # ---------------------------------------------------------------------------

    print()
    print("=" * 64)
    print("9. capture_layout + diff_reports — compare sizing iterations")
    print("=" * 64)

    # Snapshot the current (nominal) estimate
    snap_nominal = capture_layout(amp)
    print(f"  Nominal snapshot:  computation_mode='{snap_nominal.computation_mode}'")
    print(f"  Nets captured: {sorted(snap_nominal.nets.keys())}")

    # Double W and M, giving four times the estimated device area.
    amp2 = Module('PDK_skywater130', _out, 'diff_amp_v2.cir',
                  pdk_details=pdk, module_name='diff_amp_v2')
    amp2.define_module_ports(_DIFF_AMP_PORTS)   # same interface as section 1
    amp2.add(pdk.nmos1p8('mn1', d='vout_p', g='vin_p', s='tail', b='vss', w=8, l=0.18, m=4))
    amp2.add(pdk.nmos1p8('mn2', d='vout_n', g='vin_n', s='tail', b='vss', w=8, l=0.18, m=4))
    amp2.add(pdk.nmos1p8('mn_tail', d='tail', g='tail', s='vss', b='vss', w=16, l=0.18, m=4))
    amp2.declare_match_pair('diff_pair', 'mn1', 'mn2')
    amp2.net_dict['vout_p'].peak_current = 4.2e-3
    amp2.net_dict['vout_n'].peak_current = 3.6e-3
    amp2.net_dict['tail'].peak_current   = 8.0e-3

    est2 = estimate_layout(amp2, rm)

    # Wrap both estimates in reports for diff_reports
    r1 = generate_report(amp,  layout_estimate=est,  auto_discover_config=False)
    r2 = generate_report(amp2, layout_estimate=est2, auto_discover_config=False)

    diff = diff_reports(r1, r2, label_a='nominal (W=4,M=2)', label_b='scaled (W=8,M=4)')

    print()
    print(f"  {diff}")

    ld = diff.layout_diff
    if ld:
        print()
        w1, h1 = ld.bbox_a
        w2, h2 = ld.bbox_b
        print(f"  Bounding box:  {w1:.2f}×{h1:.2f} μm → {w2:.2f}×{h2:.2f} μm  "
              f"(area Δ {ld.bbox_area_delta:+.1f} μm²)")
        print()
        if ld.changed:
            print(f"  Changed nets ({len(ld.changed)}):")
            print(f"  {'Net':<10}  {'ΔR (Ω)':>10}  {'ΔC (fF)':>10}  {'ΔWidth (μm)':>12}")
            print(f"  {'':-<10}  {'':-<10}  {'':-<10}  {'':-<12}")
            for delta in sorted(ld.changed, key=lambda d: d.name):
                print(f"  {delta.name:<10}  {delta.delta_r:>+10.3f}  "
                      f"{delta.delta_c:>+10.3f}  {delta.delta_width:>+12.3f}")

    # Save the Markdown diff
    md_diff_path = os.path.join(_out, 'sizing_diff.md')
    diff.save(md_diff_path)
    print()
    print(f"  Markdown diff saved to: {md_diff_path}")


    # ---------------------------------------------------------------------------
    # 10. Close the loop — simulate before/after inject_parasitics
    # ---------------------------------------------------------------------------
    # The testbench keeps the live DUT, so injection affects its next run.

    print()
    print("=" * 64)
    print("10. Close the loop — simulate the 5T OTA before/after injection")
    print("=" * 64)

    import shutil
    from silk.simulation.runner import ngspice_executable
    from silk.exceptions import SimulationError

    if shutil.which(ngspice_executable()) is None:
        print("  Skipped: ngspice not found; install it or set SILK_NGSPICE_BIN.")
    else:
        from silk.examples.ota_5t_guide import build_5t_ota

        ota = build_5t_ota()
        ota.set_output_options(mode="new")
        ac = ota.get_test("ac")

        try:
            before = ac.run(corner="tt")
        except SimulationError as exc:
            print(f"  Skipped: {exc}")
        else:
            bw_before = before.metrics["bw_hz"]
            print(f"  BW before parasitics : {bw_before/1e6:8.3f} MHz")

            # Use illustrative effective capacitance 1 fF/µm², equivalent to
            # 0.14 fF/µm at minimum width, including an approximate fringe allowance.
            ota_rm = RoutingModel(sheet_resistance=0.08, cap_per_um2=1.0,
                                  j_max=1e-3, min_width=0.14, max_width=10.0)
            ota_est = estimate_layout(ota, ota_rm)
            inject_parasitics(ota, ota_est, mode='rc')
            print(f"  VOUT estimate        : R={ota_est.nets['VOUT'].r:.2f} Ω  "
                  f"C={ota_est.nets['VOUT'].c:.2f} fF  (length={ota_est.nets['VOUT'].length:.1f} μm, "
                  f"width={ota_est.nets['VOUT'].width:.2f} μm)")

            try:
                after = ac.run(corner="tt")
            except SimulationError as exc:
                print(f"  Skipped: {exc}")
            else:
                bw_after = after.metrics["bw_hz"]
                print(f"  BW after parasitics  : {bw_after/1e6:8.3f} MHz")

                pct = 100.0 * (bw_before - bw_after) / bw_before
                print()
                print(f"  Injected routing loading changed bandwidth by {-pct:+.1f}%")
                print("  Sweep resolution: 20 points/decade, about 12% between samples.")


    print()
    print(f"All artifacts written to: {_out}")
    print()
    print("Done.")


if __name__ == '__main__':
    main()
