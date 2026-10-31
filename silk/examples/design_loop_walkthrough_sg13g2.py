"""Run the sampling-buffer design loop with the preview SG13G2 integration.

Use 1.2 V devices. Transient tests measure settling at nominal and hot
conditions; AC tests measure open-loop gain and bandwidth. Store metrics and
spec measurements, rebuild the circuit, restore spec measurements, then estimate
parasitics and measure bandwidth again. Print geometry checks, measurements,
spec checks and artifact paths.

The author writes dimensions in microns; the models use metres. Step 0 checks
netlist dimensions, the geometry accessor and the layout estimate.

OSDI loading requires a ``.control`` block, which device-current batch mode
omits. silk refuses ``save_device_currents`` for this PDK. The example checks
that refusal and the missing-current restore error, then restores spec
measurements with ``restore_spec_measurements``. Routing uses minimum widths
because no net currents were stored. See ``docs/known_issues.md``.

Needs OSDI-capable ngspice and SG13G2 models with compiled plugins under
``PDK_ROOT``; see ``docs/installing-the-pdk.md``. Run::

    python -m silk.examples.design_loop_walkthrough_sg13g2 [OUT_DIR]

Artifacts default to ``silk-output/design_loop_sg13g2/``, subject to
``SILK_EXAMPLES_OUT``.
"""
import os
import re
import sys

from silk.circuit.modules import Module
from silk.circuit.ports import Port
from silk.circuit.report import generate_report
from silk.db import Database, EvidenceNotFoundError
from silk.examples import design_loop_walkthrough as dl
from silk.exceptions import SimulationError
from silk.layout import estimate_layout
from silk.utils import parse_spice_value
from silk.examples import example_output_dir


#: Device names and corner spelling from the catalog; assume a 5 nm width grid.
SG13G2 = dl.Technology(pdk='sg13g2', label='PDK_sg13g2', nmos='nmos_lv',
                       pmos='pmos_lv', corner='typical', width_grid_um=0.005)

#: Supply, gain and bandwidth targets for the sampling buffer.
INTENT = dict(
    dl.INTENT,
    vdd_v=1.2,
    hot_vdd_v=1.08,              # -10 % supply
    gain_db_min=18.0,            # open-loop DC gain, light load
    bw_hz_min=200e6,             # open-loop -3 dB bandwidth, light load
)

_NETLIST_W = re.compile(r'\bw=([0-9.eE+-]+)([a-zA-Z]*)')

#: Expected refusal when device-current capture is requested with OSDI models.
CURRENT_BOUNDARY_MESSAGE = (
    'save_device_currents is not supported on a PDK with OSDI models: '
    'batch-write mode emits no .control block, which is where the pre_osdi '
    'load lines live. Run without save_device_currents.'
)


def build_add_element_twin(knobs, *, work_dir):
    """Build the six-device buffer using explicit Module.add calls; dimensions are microns."""
    pdk = dl.get_pdk(SG13G2)
    nmos, pmos = getattr(pdk, SG13G2.nmos), getattr(pdk, SG13G2.pmos)
    mod = Module(SG13G2.label, work_dir, 'sampling_buffer_twin.cir',
                pdk_details=pdk, module_name='sampling_buffer')
    mod.define_module_ports({
        'VDD': Port.supply(), 'VSS': Port.ground(),
        'VINP': Port.analog_input(), 'VINM': Port.analog_input(),
        'VOUT': Port.analog_output(), 'IBIAS': Port.analog_input(),
    })
    for net in ('vint', 'tail'):
        mod.add_net(net)
    devices = (
        ('M1', nmos, dict(d='vint', g='VINP', s='tail', b='VSS'),
         dict(w=knobs['w_in_um'], l=dl.L_INPUT_UM)),
        ('M2', nmos, dict(d='VOUT', g='VINM', s='tail', b='VSS'),
         dict(w=knobs['w_in_um'], l=dl.L_INPUT_UM)),
        ('M3', pmos, dict(d='vint', g='vint', s='VDD', b='VDD'),
         dict(w=knobs['w_load_um'], l=dl.L_LOAD_UM)),
        ('M4', pmos, dict(d='VOUT', g='vint', s='VDD', b='VDD'),
         dict(w=knobs['w_load_um'], l=dl.L_LOAD_UM)),
        ('M5', nmos, dict(d='tail', g='IBIAS', s='VSS', b='VSS'),
         dict(w=knobs['w_tail_um'], l=dl.L_TAIL_UM)),
        ('M6', nmos, dict(d='IBIAS', g='IBIAS', s='VSS', b='VSS'),
         dict(w=knobs['w_ref_um'], l=dl.L_TAIL_UM)),
    )
    for name, cls, nets, params in devices:
        mod.add(cls(name, **nets, **params))
    return mod


def prove_geometry_units(knobs, work_dir):
    """Compare netlist dimensions, geometry accessors and layout areas for both construction styles."""
    pdk = dl.get_pdk(SG13G2)
    factory, _tests = dl.build_sampling_buffer(
        knobs, tech=SG13G2, intent=INTENT,
        work_dir=os.path.join(work_dir, 'factory'))
    twin = build_add_element_twin(knobs, work_dir=os.path.join(work_dir, 'twin'))
    est_f = estimate_layout(factory, dl.MET1)
    est_t = estimate_layout(twin, dl.MET1)
    packing = est_f.packing_factor

    lines = {line.split()[0]: line for line in factory.generate_netlist().splitlines()
             if line.startswith('XMM')}
    expected_w = {'M1': 'w_in_um', 'M2': 'w_in_um', 'M3': 'w_load_um',
                  'M4': 'w_load_um', 'M5': 'w_tail_um', 'M6': 'w_ref_um'}
    expected_l = {'M1': dl.L_INPUT_UM, 'M2': dl.L_INPUT_UM, 'M3': dl.L_LOAD_UM,
                  'M4': dl.L_LOAD_UM, 'M5': dl.L_TAIL_UM, 'M6': dl.L_TAIL_UM}

    print(f'PDK {pdk.pdk_name}: author writes um (manifest user_units), models '
          f'take metres (catalog pdk_unit: base); packing factor {packing}')
    print(f'{"dev":<4} {"netlist w=":>12} {"= metres":>10} | {"accessor W×L×M (um)":>22} '
          f'| {"est. area":>9} {"hand W·L·M·pack":>15} | Module.add twin')
    out = {}
    for name in sorted(expected_w):
        number, suffix = _NETLIST_W.search(lines['XM' + name]).groups()
        w_m = parse_spice_value(number + suffix)
        dev_f = factory.elem_dict[name]['device']
        dev_t = twin.elem_dict[name]['device']
        geom_f = pdk.device_geometry_um(dev_f)
        geom_t = pdk.device_geometry_um(dev_t)
        W_um, L_um, M = geom_f
        hand = knobs[expected_w[name]] * expected_l[name] * 1 * packing
        area_f, area_t = est_f.devices[name].area, est_t.devices[name].area
        agree = (abs(area_f - area_t) < 1e-9 and geom_f == geom_t)
        out[name] = dict(netlist_w_m=w_m, accessor_um=geom_f, twin_um=geom_t,
                         area_um2=area_f, twin_area_um2=area_t, hand_um2=hand)
        print(f'{name:<4} {number + suffix:>12} {w_m:>10.3e} | '
              f'{W_um:>7.3f}×{L_um:<5.3f}×{M:<4.0f}     | {area_f:>9.3f} '
              f'{hand:>15.3f} | {"agrees" if agree else "DISAGREES"} ({area_t:.3f})')
        assert abs(w_m - W_um * 1e-6) < 1e-15, (name, w_m, W_um)
        assert abs(W_um - knobs[expected_w[name]]) < 1e-9, (name, geom_f)
        assert abs(L_um - expected_l[name]) < 1e-9, (name, geom_f)
        assert abs(area_f - hand) < 1e-9 * max(1.0, hand), (name, area_f, hand)
        assert agree, (name, geom_f, geom_t, area_f, area_t)
    assert est_f.bounding_box == est_t.bounding_box, (est_f.bounding_box,
                                                      est_t.bounding_box)
    print(f'module bounding box {est_f.bounding_box[0]:.2f} x '
          f'{est_f.bounding_box[1]:.2f} um, both styles')
    return out


def prove_current_annotation_boundary(knobs, work_dir):
    """Check the refusal of device-current capture with OSDI models; return the error message."""
    mod, tests = dl.build_sampling_buffer(
        knobs, tech=SG13G2, intent=INTENT,
        work_dir=os.path.join(work_dir, 'boundary'))
    dl.wire_pre_layout_tests(tests, knobs, INTENT)
    test = tests[dl.T_STEP_LIGHT]
    try:
        dl.run_test(test, corner=SG13G2.corner,
                    temperature=INTENT['nominal_temperature_c'],
                    voltage=INTENT['vdd_v'], currents=True)
        raise AssertionError(
            'expected SimulationError for device-current capture with OSDI models')
    except SimulationError as err:
        message = str(err)
        assert message == CURRENT_BOUNDARY_MESSAGE, message
        print(f'save_device_currents=True on SG13G2 (OSDI) -> {type(err).__name__}:')
        print(f'  {message}')
        print('  (metrics and spec measurements are stored; only spec measurements '
              'are attached to the rebuilt circuit; no per-net currents are stored)')
        return message
    finally:
        test.module.save_device_currents = False


# --------------------------------------------------------------------------- #
# Store and restore measurements without net-current evidence
# --------------------------------------------------------------------------- #
def characterize_and_store(mod, tests, knobs, db_path, out_dir, *,
                           tech=SG13G2, intent=INTENT):
    """Seal the circuit, run three tests and store metrics and specs without net currents."""
    corner = tech.corner
    version_id = mod.create_circuit_version('sampling buffer, pre-layout')
    print(f'sealed circuit version {dl._short(version_id)}')

    contexts, metrics = {}, {}
    for name, temperature, voltage in (
            (dl.T_STEP_LIGHT, intent['nominal_temperature_c'], intent['vdd_v']),
            (dl.T_STEP_HEAVY, intent['hot_temperature_c'], intent['hot_vdd_v'])):
        test = tests[name]
        print(f'\n[{name}] transient at ({corner}, {temperature} C, {voltage} V), '
              f'params={test.parameters} (currents=False -- OSDI boundary above)')
        report = dl.run_test(test, corner=corner, temperature=temperature,
                             voltage=voltage, currents=False)
        contexts[name] = test.netlist.characterization_context
        metrics.update(report.metrics)
        stored = mod.store_report(report, tool_name='ngspice')
        print(f'  {report.summary()}')
        print(f"  stored: {stored['metrics_stored']} metric(s), "
              f"{stored['specs_stored']} spec measurement(s); no per-net "
              "currents (OSDI models require a .control block)")

    test = tests[dl.T_AC_PRE]
    print(f"\n[{dl.T_AC_PRE}] AC at ({corner}, {intent['nominal_temperature_c']} C, "
          f"{intent['vdd_v']} V), params={test.parameters}")
    report = dl.run_test(test, corner=corner, temperature=intent['nominal_temperature_c'],
                         voltage=intent['vdd_v'], currents=False)
    contexts[dl.T_AC_PRE] = test.netlist.characterization_context
    metrics.update(report.metrics)
    stored = mod.store_report(report, tool_name='ngspice')
    print(f'  {report.summary()}')
    print(f"  stored: {stored['metrics_stored']} metric(s), "
          f"{stored['specs_stored']} spec measurement(s)")

    recon = mod.reconcile_specs()
    print('\nSpec reconciliation (pre-layout):')
    print(recon.summary())
    report_pre = generate_report(mod, auto_discover_config=False)
    report_pre.save_json(os.path.join(out_dir, 'report_pre_layout.json'))
    report_pre.save_html(os.path.join(out_dir, 'report_pre_layout.html'))
    print(f"report -> {os.path.join(out_dir, 'report_pre_layout.html')}")

    for name in (dl.T_STEP_LIGHT, dl.T_STEP_HEAVY, dl.T_AC_PRE):
        tests[name].netlist.close()
    mod.close()
    return version_id, contexts, metrics, report_pre


def reopen_and_query(db_path, version_id, *, tech=SG13G2, intent=INTENT):
    """Query stored profiles and pre-layout bandwidth from the database."""
    corner = tech.corner
    with Database(db_path) as history:
        profiles = history.list_characterization_profiles(version_id)
        print(f'{len(profiles)} characterization profile(s) on version '
              f'{dl._short(version_id)}:')
        for p in profiles:
            print(f'  profile {dl._short(p.profile_id)} test={p.test_name!r} '
                  f'params={dict(p.parameters)} origin={dl._short(p.origin_version_id)}')
        bw_rows = history.get_simulation_results(version_id, parameter='bw_hz',
                                                 corner=corner)
        bw_pre = bw_rows[0].value
        print(f"  stored pre-layout bw_hz = {bw_pre / 1e6:.2f} MHz "
              f"(source={bw_rows[0].source!r}, tool={bw_rows[0].tool_name!r})")
    return bw_pre


def rebuild_and_restore(knobs, db_path, version_id, contexts, work_dir, *,
                        tech=SG13G2, intent=INTENT):
    """Rebuild, check that current evidence is absent and restore spec measurements.

    Return (module, tests, restored_count)."""
    specs = []
    fresh, tests = dl.build_sampling_buffer(knobs, tech=tech, intent=intent,
                                            db_path=db_path, work_dir=work_dir, specs=specs)
    rebuilt_id = fresh.create_circuit_version()
    assert rebuilt_id == version_id, 'the rebuild resolves to the stored snapshot'
    print(f'rebuilt design resolves to the same version {dl._short(rebuilt_id)} — '
          'no simulation has run in this session')
    assert fresh.net_dict['tail'].peak_current is None

    # No net-current evidence was stored, so the combined restore must refuse.
    try:
        fresh.restore_characterization_evidence(
            context=contexts[dl.T_STEP_LIGHT], corner=tech.corner,
            temperature=intent['nominal_temperature_c'], voltage=intent['vdd_v'])
        raise AssertionError(
            'expected EvidenceNotFoundError because no net currents were stored')
    except EvidenceNotFoundError as err:
        print(f'restore_characterization_evidence(...) -> {type(err).__name__}: {err}')
        print('  (expected: no current evidence exists for this OSDI-PDK design)')
    assert fresh.net_dict['tail'].peak_current is None, 'the failed restore applied nothing'

    # Restore compatible spec measurements without requesting net currents.
    restored_count = fresh.restore_spec_measurements(corner=tech.corner)
    print(f"restore_spec_measurements(corner={tech.corner!r}) restored "
          f'{restored_count} spec measurement(s) on an OSDI PDK; per-net current '
          'evidence is absent')
    for s in specs:
        latest = s.latest()
        if latest is not None:
            print(f'  {s.key:<16} {latest.value:.4g} {"PASS" if latest.passes else "FAIL"}')
    return fresh, tests, restored_count


def main(out_dir=None):
    """Run the SG13G2 design loop and return its results."""
    out_dir = out_dir or example_output_dir('design_loop_sg13g2')
    work_dir = os.path.join(out_dir, 'work')
    os.makedirs(work_dir, exist_ok=True)
    db_path = os.path.join(out_dir, 'design_loop.db')
    if os.path.exists(db_path):
        os.remove(db_path)                       # deterministic demo

    dl._h('0a. Units: metre-native models, micron author, one geometry accessor')
    knobs = dl.size_from_intent(INTENT, SG13G2)
    geometry = prove_geometry_units(knobs, os.path.join(out_dir, 'work', 'units'))

    dl._h('0b. The OSDI/current-annotation boundary, demonstrated')
    boundary_message = prove_current_annotation_boundary(
        knobs, os.path.join(out_dir, 'work', 'boundary'))

    dl._h('1. System intent -> sizing')
    print(f"technology {SG13G2.pdk} (corner {SG13G2.corner!r}, VDD {INTENT['vdd_v']} V)")
    print(f"widths: input {knobs['w_in_um']} um, load {knobs['w_load_um']} um, "
          f"tail {knobs['w_tail_um']} um, reference {knobs['w_ref_um']} um")

    dl._h('2. Build: parameterized design with embedded characterization suite')
    mod, tests = dl.build_sampling_buffer(knobs, tech=SG13G2, intent=INTENT,
                                          db_path=db_path, work_dir=work_dir)
    ports = {name: f'{pin.direction.value}/{pin.role.value}'
             for name, pin in mod.pin_list.items()}
    print(f'ports  : {ports}')
    print(f'tests  : {mod.list_tests()}')
    dl.wire_pre_layout_tests(tests, knobs, INTENT)

    dl._h('3+4. Characterize under two contexts, report, persist (no currents)')
    version_id, contexts, metrics, report_pre = characterize_and_store(
        mod, tests, knobs, db_path, out_dir)

    dl._h('5. Reopen the database file — provenance with no live circuit')
    bw_pre = reopen_and_query(db_path, version_id)

    dl._h('6. Rebuild and restore spec measurements')
    fresh, fresh_tests, restored_count = rebuild_and_restore(
        knobs, db_path, version_id, contexts, os.path.join(work_dir, 'rebuild'))

    dl._h('7. Layout feedback: min-width routing (no restored currents) -> parasitics -> re-measure')
    print('Routing uses MET1.min_width because no net carries a restored current.')
    est, em, post_version_id, bw_post, report_post = dl.layout_feedback(
        fresh, fresh_tests, knobs, version_id, report_pre, bw_pre, out_dir,
        tech=SG13G2, intent=INTENT)

    dl._h('8. A mismatch fails loudly, with nothing half-applied')
    print('Checking the rebuilt circuit against an incompatible context.')
    mismatch = dl.demonstrate_mismatch(fresh, knobs, db_path, contexts, work_dir,
                                       tech=SG13G2, intent=INTENT)

    dl._h('9. The netlist comes from the same design object')
    netlist = fresh.generate_netlist()
    netlist_path = os.path.join(out_dir, 'sampling_buffer_post_layout.cir')
    with open(netlist_path, 'w') as f:
        f.write(netlist)
    lines = netlist.splitlines()
    print('\n'.join(lines[:3] + ['  ...']
                    + [l for l in lines if l.startswith(('XMM2', 'XMM4'))]
                    + [l for l in lines if '__par_' in l and 'VOUT' in l]
                    + ['  ...'] + lines[-1:]))
    print(f'({len(lines)} lines) -> {netlist_path}')
    fresh_tests[dl.T_AC_POST].netlist.close()
    fresh.close()

    print('\nDone. Per-net current evidence is unavailable with OSDI models. '
          'Guides: docs/agent-guide.md (loop), docs/database.md (evidence '
          'contract), docs/installing-the-pdk.md (SG13G2 section).')
    return dict(
        tech=SG13G2, intent=INTENT,
        knobs=knobs, version_id=version_id, post_version_id=post_version_id,
        contexts=contexts, metrics=metrics,
        restored_count=restored_count,
        layout=est, em=em, bw_pre_hz=bw_pre, bw_post_hz=bw_post,
        mismatch=mismatch, netlist_path=netlist_path, out_dir=out_dir,
        report_post=report_post, geometry=geometry,
        boundary_message=boundary_message,
    )


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
