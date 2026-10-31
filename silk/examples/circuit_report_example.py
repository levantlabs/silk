"""Generate, save, reload and compare reports for a two-transistor amplifier.

Print topology, annotation coverage and lint findings. Save HTML and JSON
reports under ``silk-output/`` (or ``SILK_EXAMPLES_OUT``), then reload the JSON.
Lint and coverage are omitted from JSON; regenerate the report to obtain them.
Compare the first report with an incomplete structural edit that adds one device
and two nets, ``vcas`` and ``vmid``. The diff shows two additional nets with
violations unchanged. The edited circuit remains incomplete.

Uses the packaged sky130 catalog; no installed PDK models or simulator required.
Run: ``python -m silk.examples.circuit_report_example``.
See ``docs/reports.md`` for the report format.
"""
import os

from silk import load_pdk
from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk.circuit.report import generate_report, CircuitReport
from silk.circuit.report_diff import diff_reports
from silk.elements import elements
from silk.examples import example_output_dir

OUT_DIR = example_output_dir(create=False)


def build_amp(pdk, w_in=4.0):
    m = Module('PDK_skywater130', OUT_DIR, 'report_amp.cir',
               pdk_details=pdk, module_name='report_amp')
    # The block's interface is declared, rails included: both supplies are
    # ports, so nothing inside is wired to a hidden local rail (the lint
    # catalog's STRUCT-013 reports that shape by default).
    m.define_module_ports({
        'vdd': Port.supply(), 'vss': Port.ground(),
        'vin': Port.analog_input(), 'vbias': Port.analog_input(),
        'vout': Port.analog_output(),
    })
    with m:
        pdk.nmos1p8('mn_in', d='vout', g='vin', s='vss', b='vss',
                    w=w_in, l=0.5)
        pdk.pmos1p8('mp_load', d='vout', g='vbias', s='vdd', b='vdd',
                    w=2.0 * w_in, l=0.5)
    m.add(elements.resistor('r_load', p='vout', n='vss', value=20e3))
    return m


def main():
    example_output_dir()
    pdk = load_pdk('sky130')

    # -- 1. One call harvests everything ---------------------------------
    amp = build_amp(pdk)
    report = generate_report(amp, auto_discover_config=False)
    print('--- print_summary() ------------------------------------------')
    report.print_summary()

    # -- 2. What's inside ------------------------------------------------
    print('\n--- selected contents ----------------------------------------')
    # topology counts SUBCIRCUIT instances (module_dict) — this flat design
    # has none; its devices show up in the lint/coverage half instead.
    print(f'subckt instances: {report.topology.total_instances}   '
          f'nets: {report.topology.total_nets}')
    print(f'annotation coverage: {report.annotation.coverage_pct:.0f}% '
          f'({report.annotation.annotated}/{report.annotation.total} ports)')
    if report.lint is not None:
        print(f'lint: {report.lint.summary()}')
    print(f'interface check passed: {report.interface_check_passed}')

    # -- 3. Save the shareable artifacts ---------------------------------
    html_path = os.path.join(OUT_DIR, 'report_amp.html')
    json_path = os.path.join(OUT_DIR, 'report_amp.json')
    report.save_html(html_path)
    report.save_json(json_path)
    print(f'\nsaved -> {os.path.relpath(html_path)}, '
          f'{os.path.relpath(json_path)}')

    # -- 4. Reload -------------------------------------------------------
    # The round-trip restores topology/violations/simulation/metadata/spec
    # reconciliation/layout estimate. lint and coverage are
    # not serialized (the file's omitted_sections key says so) — regenerate
    # when you need those two.
    loaded = CircuitReport.load_json(json_path)
    print('\n--- load_json() round-trip -----------------------------------')
    print(f'topology survives: instances={loaded.topology.total_instances}')
    print(f'metadata survives: silk {loaded.metadata.silk_version}' if loaded.metadata
          else 'metadata: None')
    print(f'lint after reload: {loaded.lint}   coverage: {loaded.coverage}'
          '   (not serialized — regenerate for these)')

    # -- 5. Diff two design iterations ------------------------------------
    # A width change alone is absent from the structural diff; the added device
    # introduces vcas and vmid, increasing the net count with violations unchanged.
    # vcas is undriven and vmid has no other connection: this is incomplete.
    v2 = build_amp(pdk, w_in=8.0)
    with v2:
        pdk.nmos1p8('mn_cas', d='vout', g='vcas', s='vmid', b='vss',
                    w=8.0, l=0.5)
    report2 = generate_report(v2, auto_discover_config=False)
    print('\n--- diff_reports(v1, v2) -------------------------------------')
    diff = diff_reports(report, report2, label_a='flat w=4', label_b='incomplete structural edit w=8')
    diff.print_summary()


if __name__ == '__main__':
    main()
