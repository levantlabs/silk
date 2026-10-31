"""Store and restore current measurements for a two-leg NMOS current mirror.

Run two characterizations with distinct test parameters and temperature/supply
conditions at the tt corner. Store metrics, spec measurements and net currents.
Rebuild the circuit, restore selected evidence and size wires without simulating
again. Re-store one run's currents to show that superseded sets remain readable
by ID. An edited circuit demonstrates an evidence mismatch.

A characterization profile identifies the registered test, its owner and its
parameters. Process corner, temperature and supply voltage select evidence
within that profile. A final section stores and restores unscoped net currents
without a registered test.

Print stored evidence IDs, restored measurement counts and estimated wire
widths. Artifacts go under ``silk-output/`` (or ``SILK_EXAMPLES_OUT``).
Needs ngspice and installed sky130 models. Run::

    python -m silk.examples.net_current_persistence_example
"""
import os

import numpy as np

from silk import load_pdk, sources
from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk.db import Database, EvidenceMismatchError
from silk.elements import elements
from silk.simulation import Transient
from silk.metrics import average
from silk.layout.estimator import estimate_layout
from silk.layout.routing_model import RoutingModel
from silk.examples import example_output_dir

pdk = load_pdk('sky130')

# met1 constants for wire sizing (j_max in A/µm, widths in µm), from the
# PDK manifest's routing section.
MET1 = RoutingModel.from_pdk(pdk)

# Shared spec declarations let a rebuild match the stored circuit version.
SPEC_NOMINAL = dict(
    name='Output stays below VDD (nominal)', metric_name='vout_headroom_v',
    op='<=', target=1.8, key='vout_headroom_nominal',
)
SPEC_HOT = dict(
    name='Output stays below VDD (hot corner)', metric_name='vout_headroom_v',
    op='<=', target=1.62, key='vout_headroom_hot',
)


# --------------------------------------------------------------------------- #
# The circuit: a 2-leg NMOS current mirror sinking current through a load
# --------------------------------------------------------------------------- #
def build_current_sink(load_ohm=560.0, db_path=None):
    """Build a two-leg NMOS mirror sinking current through Rload.

    The internal out net has measurable pin currents for wire sizing.
    Specs are registered separately by each characterization test.
    """
    blk = Module('PDK_skywater130', example_output_dir(), 'sink.cir',
                 pdk_details=pdk, module_name='current_sink', db_path=db_path)
    blk.define_module_ports({'VDD': Port.supply(), 'VSS': Port.ground()})

    blk.add_source(sources.CurrentSource('iref', positive='VDD',
                                         negative='vbias', dc=300e-6))
    # Diode-connected reference.
    blk.add(pdk.nmos1p8('Mref', d='vbias', g='vbias', s='VSS', b='VSS', w=2.0, l=0.5, m=1))
    # Mirror leg.
    blk.add(pdk.nmos1p8('Mout', d='out', g='vbias', s='VSS', b='VSS', w=24.0, l=0.5, m=1))
    blk.add(elements.resistor(name='Rload', p='VDD', n='out', value=load_ohm))
    return blk


def _vout_headroom(ckt):
    """Tail-average voltage on the internal net ``out`` inside the ``dut``
    submodule instance. ``x<instance>.<net>`` is silk's hierarchical signal
    path into a submodule's internal node (see
    ``ModuleTestbench.add_metric``'s docstring)."""
    sig = ckt.load_signal('xdut.out')
    t = np.asarray(sig.x)
    return average(sig, start=0.5 * (t[0] + t[-1]))


# --------------------------------------------------------------------------- #
# Step 1 — characterize twice, at two distinct contexts, one corner
# --------------------------------------------------------------------------- #
def declare_test(dut, *, test_name, spec, parameters):
    """Register a test and its spec before wiring its netlist.

    Declare the complete spec set before sealing the circuit version.
    The testbench retains the live DUT.
    """
    test = dut.create_test(test_name, parameters=parameters)
    test.add_metric(spec['metric_name'], _vout_headroom)
    test.add_spec(spec['name'], spec['metric_name'], spec['op'], spec['target'],
                  key=spec['key'])
    return test


def wire_test(test, *, voltage):
    """Wire a declared test with a supply and transient analysis.

    run_and_store configures the supply and operating temperature before running.
    """
    test.add_module_under_test({'VDD': 'vdd', 'VSS': '0'})
    test.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0',
                                          dc=voltage))
    test.add_sim_type(Transient(step=5e-9, stop=2e-6))


def run_and_store(dut, test, db_path, *, test_name, parameters, temperature,
                  voltage, tool_name='ngspice'):
    """Run one already-built characterization and store all three evidence
    families under the same declared context (same profile, same corner
    ('tt'), the context's own declared temperature/supply): ``store_report()``
    atomically persists this run's metrics and its declared spec measurement
    together in one transaction, and ``store_net_currents()`` is a separate
    write — its own transaction, run right after — that persists the
    per-net currents under that same profile/PVT context.

    Configure the operating point explicitly with ``set_temperature()`` and
    ``set_supply_voltage()`` before running. Repeated calls replace the
    previous operating point; do not add a custom ``.temp`` line that would
    override the configured temperature.

    Returns ``(version_id, context)``.
    """
    test.set_temperature(temperature)
    test.set_supply_voltage(voltage, source='vdd')
    report = test.run(corner='tt')
    context = test.netlist.characterization_context
    stored = dut.store_report(report, tool_name=tool_name)
    print(f"  [{test_name}] metrics_stored={stored['metrics_stored']} "
          f"specs_stored={stored['specs_stored']}")

    current_report = test.netlist.annotate_currents(tstart=parameters['tstart_s'])
    versions = test.netlist.store_net_currents(
        current_report, corner='tt', db_path=db_path, temperature=temperature,
        voltage=voltage, tool_name=tool_name,
    )
    version_id = versions['current_sink']
    print(f"  [{test_name}] currents stored -> {version_id[:8]}… "
          f"@ (tt, {temperature}°C, {voltage} V), "
          f"params={dict(context.parameters)}")
    return version_id, context


# --------------------------------------------------------------------------- #
# Step 2 — a later sizing session: no simulation
# --------------------------------------------------------------------------- #
def size_wires(module):
    """Hand the module to the layout estimator and read the widths it computed
    from each net's current.

    Returns (sized, unsized). The estimator sizes a net from its
    ``peak_current``; a **port net** (VDD/VSS) has none — a port's
    boundary current is not one of its pins — so it stays at min-width instead
    of being sized from a partial sum that would under-size the rail. Rails need
    a KCL-boundary calculation from their annotated pin currents.
    """
    estimate_layout(module, MET1)
    sized, unsized = {}, {}
    for name, net in sorted(module.net_dict.items()):
        if net.estimated_width is None:
            continue
        (sized if net.peak_current is not None else unsized)[name] = net.estimated_width
    return sized, unsized


def main():
    run_dir = example_output_dir('net_current_persistence')
    db_path = os.path.join(run_dir, 'em_currents.db')
    if os.path.exists(db_path):
        os.remove(db_path)                          # deterministic demo

    # ----------------------------------------------------------------- #
    # Profile-scoped path: one DUT, two characterizations, one corner
    # ----------------------------------------------------------------- #
    print('== Profile-scoped characterization ==\n')
    dut = build_current_sink(db_path=db_path)
    dut.save_device_currents = True                  # before any run()
    dut.terminal_current_method = 'ammeter_wrapper'   # read all 4 MOS terminals

    params_a = {'tstart_s': 1e-6}
    params_b = {'tstart_s': 1.5e-6}
    # Declare both tests before sealing the circuit version.
    test_a = declare_test(dut, test_name='characterize_nominal', spec=SPEC_NOMINAL,
                          parameters=params_a)
    test_b = declare_test(dut, test_name='characterize_hot_corner', spec=SPEC_HOT,
                          parameters=params_b)
    wire_test(test_a, voltage=1.8)
    wire_test(test_b, voltage=1.62)
    # Structure and both declared specs are sealed together before
    # either test runs.
    dut.create_circuit_version('current sink, both characterizations declared')

    print('Characterizing at nominal conditions (declared params + PVT A)...')
    version_id, context_a = run_and_store(
        dut, test_a, db_path, test_name='characterize_nominal',
        parameters=params_a, temperature=27.0, voltage=1.8,
    )

    print('Characterizing at a hot/low-supply corner (declared params + PVT B)...')
    version_id_b, _context_b = run_and_store(
        dut, test_b, db_path, test_name='characterize_hot_corner',
        parameters=params_b, temperature=85.0, voltage=1.62,
    )
    assert version_id_b == version_id, 'same DUT snapshot both times'
    print()

    # Re-storing the same waveform selects a new current set and retains the old one.
    print('Selected current set before re-storing:')
    evidence_a = dut.select_characterization_evidence(
        context=context_a, corner='tt', temperature=27.0, voltage=1.8,
    )
    old_current_set_id = evidence_a.current_set_id
    print(f'  current_set_id = {old_current_set_id[:8]}…')

    print("Re-storing the same run's current measurements...")
    current_report_a = test_a.netlist.annotate_currents(tstart=1e-6)
    test_a.netlist.store_net_currents(
        current_report_a, corner='tt', db_path=db_path,
        temperature=27.0, voltage=1.8, tool_name='ngspice',
    )
    evidence_a = dut.select_characterization_evidence(
        context=context_a, corner='tt', temperature=27.0, voltage=1.8,
    )
    new_current_set_id = evidence_a.current_set_id
    assert new_current_set_id != old_current_set_id
    print(f'  new current_set_id = {new_current_set_id[:8]}… '
          f'(selection now returns this one)')

    dut.close()
    test_a.netlist.close()
    test_b.netlist.close()

    # ----------------------------------------------------------------- #
    # Close/reopen: provenance query with just the SQLite file
    # ----------------------------------------------------------------- #
    print('\n== Reopen: provenance query with no live Python circuit ==\n')
    with Database(db_path) as history:
        profile_a = history.get_characterization_profile(
            version_id, context=context_a
        )
        print(f'Profile A: {profile_a.profile_id[:8]}… '
              f'owner={profile_a.owner_path!r} test={profile_a.test_name!r} '
              f'params={dict(profile_a.parameters)}')

        metrics_a = history.get_simulation_results(
            version_id, profile_id=profile_a.profile_id, corner='tt',
        )
        print(f'  metrics at this profile/corner: '
              f'{sorted(m.parameter for m in metrics_a)}')

        # Superseded current sets remain readable by ID.
        old_origin, old_rows = history.get_net_current_set(
            version_id, current_set_id=old_current_set_id
        )
        print(f'  old current_set_id {old_current_set_id[:8]}… still reads '
              f'back: origin={old_origin[:8]}…, {len(old_rows)} rows')

    # ----------------------------------------------------------------- #
    # Later sizing session — rebuild fresh, never simulated, restore, size
    # ----------------------------------------------------------------- #
    print('\n== Later sizing session — rebuild, restore, size wires ==\n')
    fresh = build_current_sink(db_path=db_path)      # never simulated
    # Match the stored spec declarations before resolving the rebuilt version.
    fresh.declare_spec(SPEC_NOMINAL['name'], SPEC_NOMINAL['metric_name'],
                       SPEC_NOMINAL['op'], SPEC_NOMINAL['target'],
                       key=SPEC_NOMINAL['key'])
    fresh.declare_spec(SPEC_HOT['name'], SPEC_HOT['metric_name'],
                       SPEC_HOT['op'], SPEC_HOT['target'], key=SPEC_HOT['key'])
    rebuilt_id = fresh.create_circuit_version()
    assert rebuilt_id == version_id, 'rebuild matches the stored snapshot'

    restored = fresh.restore_characterization_evidence(
        context=context_a, corner='tt', temperature=27.0, voltage=1.8,
    )
    print(f'Restored profile {restored.profile_id[:8]}… current_set '
          f'{restored.current_set_id[:8]}… '
          f'({restored.measurement_count} measurement(s), '
          f'{restored.current_count} current row(s)) — no simulation ran.')
    assert restored.current_set_id == new_current_set_id, (
        'restore selects the CURRENT set, never a superseded one'
    )

    sized, unsized = size_wires(fresh)
    for name, width in sized.items():
        print(f'  {name:<6} carries {fresh.net_dict[name].peak_current * 1e3:6.3f} mA'
              f'  ->  needs {width:.3f} µm on met1')
    if unsized:
        print(f'  Port nets held at min-width: {", ".join(unsized)} '
              '(a port has no net-level current by design).')
    fresh.close()

    # ----------------------------------------------------------------- #
    # Reject evidence for an edited circuit
    # ----------------------------------------------------------------- #
    print('\n== Editing the circuit invalidates its stored evidence ==\n')
    edited = build_current_sink(load_ohm=1200.0, db_path=db_path)  # was 560 Ω
    edited.create_circuit_version()          # a brand-new, unrelated snapshot
    try:
        edited.select_characterization_evidence(
            context=context_a, corner='tt', temperature=27.0, voltage=1.8,
        )
    except EvidenceMismatchError as err:
        print(f'  {err}')
        print('  -> re-run the characterization for the new circuit.')
    finally:
        edited.close()

    # ----------------------------------------------------------------- #
    # The unscoped path — no ModuleTestbench, no declared test identity.
    # store_net_currents()/load_net_currents() use no profile here.
    # ----------------------------------------------------------------- #
    print('\n== Unscoped path: a plain Module, no test identity ==\n')
    unscoped_run_dir = os.path.join(run_dir, 'unscoped')
    os.makedirs(unscoped_run_dir, exist_ok=True)
    tb = Module('PDK_skywater130', unscoped_run_dir, 'sink_tb.cir',
               pdk_details=pdk)
    tb.add_module(build_current_sink(), 'dut', nets={'VDD': 'vdd', 'VSS': '0'})
    tb.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
    # Configure the primary supply so the run records verified operating
    # conditions, without registering a test or characterization profile.
    tb.set_supply_voltage(1.8, source='vdd')
    tb.add_sim_type(Transient(step=5e-9, stop=2e-6))
    tb.save_device_currents = True
    tb.terminal_current_method = 'ammeter_wrapper'
    tb.simulate(corner='tt')
    unscoped_report = tb.annotate_currents(tstart=1e-6)
    unscoped_versions = tb.store_net_currents(
        unscoped_report, corner=tb.operating_conditions.scalar('corner'), db_path=db_path,
        temperature=27.0, voltage=1.8, tool_name='ngspice',
    )
    unscoped_version_id = unscoped_versions['current_sink']
    tb.close()

    with Database(db_path) as history:
        unscoped_origin, unscoped_rows = history.get_net_current_set(
            unscoped_version_id, 'tt'
        )
    print(f'  unscoped currents stored -> origin={unscoped_origin[:8]}…, '
          f'{len(unscoped_rows)} rows, no profile involved')

    unscoped_block = build_current_sink()            # never simulated
    unscoped_block.load_net_currents('tt', db_path=db_path)
    print(f"  reloaded via load_net_currents(): "
          f"out carries {unscoped_block.net_dict['out'].peak_current * 1e3:.3f} mA")
    unscoped_block.close()

    print('\nDone. See docs/database.md for the full profile-scoped contract.')


if __name__ == '__main__':
    main()
