"""Unscoped circuit tracking: seal, store, close, reopen, restore, and promote.

The SQLite file is silk's persistent tracking record. Python remains the
authoritative circuit and spec definition: rebuild the same ``Module``, then
silk verifies its structure/spec fingerprints before returning or restoring
evidence. Uses the packaged sky130 catalog; no installed PDK models or
simulator required.

Run::

    python -m silk.examples.circuit_tracking_example

The JSON written near the end is deliberately a lossy interoperability
projection, not a database backup. See ``docs/database.md`` for the complete
tracking contract and failure modes.
"""

import os

from silk import load_pdk
from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk.circuit.spec_report import SpecReport
from silk.db import CircuitImporter, Database, DesignStage
from silk.elements import elements
from silk.exceptions import LockedVersionError
from silk.examples import example_output_dir


OUT_DIR = example_output_dir(create=False)
DB_PATH = os.path.join(OUT_DIR, 'circuit_tracking.db')


def _fresh_demo_db(path):
    """Remove a stale demo database so the walkthrough always starts clean.

    The file this example writes is a disposable demo artifact (like the
    plots in ``output/``), so an old copy — including one from a pre-v2
    silk, which today's open would rightly refuse — is deleted, not
    migrated. A real database you care about is the opposite case: never
    delete it; migrate it explicitly with ``silk.db.migrate_v1_to_v2`` (see
    docs/database.md, "Migrating a schema-v1 database").
    """
    if os.path.exists(path):
        os.remove(path)


def build_amp(pdk, dir_folder):
    """Build the authoritative Python definition, including stable spec keys."""
    module = Module(
        'PDK_skywater130',
        dir_folder,
        'tracked_amp.cir',
        pdk_details=pdk,
        module_name='tracked_amp',
        db_path=DB_PATH,
    )
    # The interface — rails included — is part of the sealed identity: a
    # port-less block would wire its devices to hidden local rails (the
    # default lint catalog's STRUCT-013 reports that shape).
    module.define_module_ports({
        'vdd': Port.supply(), 'vss': Port.ground(),
        'vin': Port.analog_input(), 'vbias': Port.analog_input(),
        'vout': Port.analog_output(),
    })
    with module:
        pdk.nmos1p8(
            'mn_in', d='vout', g='vin', s='vss', b='vss', w=4.0, l=0.5
        )
        pdk.pmos1p8(
            'mp_load', d='vout', g='vbias', s='vdd', b='vdd',
            w=8.0, l=0.5
        )
    module.add(
        elements.resistor('r_deg', p='vout', n='vss', value=20e3)
    )
    gain_spec = module.declare_spec(
        'DC gain', 'gain_dB', '>=', 30.0, units='dB', key='gain'
    )
    module.declare_spec(
        'Unity-gain frequency', 'ugf_MHz', '>=', 100.0,
        units='MHz', key='ugf'
    )
    return module, gain_spec


def main():
    _fresh_demo_db(DB_PATH)
    example_output_dir()    # create it now, with a clear error if we cannot
    pdk = load_pdk('sky130')
    amp, _ = build_amp(pdk, OUT_DIR)

    # Seal one immutable snapshot. An identical rerun resolves this UUID;
    # changing structure or a spec definition produces a different snapshot.
    source_id = amp.create_circuit_version('resistor-loaded common-source stage')
    print(f'sealed dev snapshot: {source_id}')

    context = {'load_pf': 2.0, 'method': 'hand-entry'}
    amp.store_results(
        {'gain_dB': (31.7, 'dB'), 'ugf_MHz': (118.0, 'MHz')},
        source='user',
        sim_type='ac',
        corner='tt',
        temperature=27.0,
        voltage=1.8,
        tool_name='hand-entry',
        conditions=context,
    )
    amp.store_results(
        {'gain_dB': (28.4, 'dB'), 'ugf_MHz': (87.0, 'MHz')},
        source='user',
        sim_type='ac',
        corner='ss',
        temperature=85.0,
        voltage=1.62,
        tool_name='hand-entry',
        conditions=context,
    )

    # ModuleTestbench.run() normally creates this report. Constructing it here
    # keeps the example simulator-free; stable owner/key identity is explicit.
    ff_report = SpecReport(
        'corner_bench', 'tracked_amp',
        {'gain_dB': 33.9, 'ugf_MHz': 145.0},
    )
    ff_report.check(
        'DC gain', 'gain_dB', '>=', 30.0,
        spec_key='gain', owner_path='$'
    )
    ff_report.check(
        'Unity-gain frequency', 'ugf_MHz', '>=', 100.0,
        spec_key='ugf', owner_path='$'
    )
    stored = amp.store_report(
        ff_report,
        source='user',
        sim_type='ac',
        corner='ff',
        temperature=-40.0,
        voltage=1.98,
        tool_name='hand-entry',
        conditions=context,
    )
    print(
        f'store_report: {stored["metrics_stored"]} metrics, '
        f'{stored["specs_stored"]} spec measurements'
    )

    projection_path = os.path.join(
        OUT_DIR, 'circuit_tracking_projection.json'
    )
    amp.export_to_json(projection_path)
    amp.close()
    print('closed the first Python session')

    # Historical access needs only the database and the UUID. Returned records
    # say where each observation originated.
    with Database(DB_PATH) as database:
        history = database.get_simulation_results(
            source_id, parameter='gain_dB', include_ancestors=False
        )
        declarations = database.get_stored_specs(source_id)
    print('\nhistorical gain observations (no Module rebuilt):')
    for record in history:
        print(
            f'  {record.corner}: {record.value} {record.unit}; '
            f'origin={record.origin_version_id}'
        )
    print(
        'stored spec keys:',
        ', '.join(record.spec_key for record in declarations),
    )

    # A later process rebuilds authoritative Python, resolves the same sealed
    # snapshot, and asks for typed evidence/restore explicitly.
    rebuilt, gain_spec = build_amp(pdk, OUT_DIR)
    assert rebuilt.create_circuit_version() == source_id
    latest_tt = rebuilt.query_results(
        parameter='gain_dB',
        source='user',
        corner='tt',
        latest_only=True,
    )[0]
    restored = rebuilt.restore_spec_measurements(
        source='user', tool_name='hand-entry'
    )
    print(
        f'\nrebuilt query: {latest_tt.value} {latest_tt.unit}; '
        f'restored {restored} spec measurements; '
        f'gain origin={gain_spec.latest().origin_version_id}'
    )

    # Promotion creates a new immutable child identity but does not copy
    # evidence. Compatible queries walk the parent lineage and expose origin.
    stable_id = rebuilt.promote(
        'stable', description='reviewed release candidate'
    )
    inherited = rebuilt.query_results(
        parameter='gain_dB', source='user', corner='tt', latest_only=True
    )[0]
    print(
        f'promoted snapshot: {stable_id}; inherited metric '
        f'origin={inherited.origin_version_id}'
    )
    try:
        rebuilt.store_results(
            {'gain_dB': (99.0, 'dB')}, source='user', corner='tt'
        )
        raise AssertionError('locked version accepted a write')
    except LockedVersionError as error:
        print(f'locked write rejected: {error}')

    rebuilt.close()
    print(f'\ndatabase: {os.path.relpath(DB_PATH)}')
    print(
        'lossy JSON projection:',
        os.path.relpath(projection_path),
    )

    # Import the projection into a second, empty database. Import rebuilds the
    # snapshot identity from the content and allocates new ids, so this is a
    # new record, not a copy. Only what the projection carries is printed.
    imported_path = os.path.join(OUT_DIR, 'circuit_tracking_imported.db')
    _fresh_demo_db(imported_path)
    with Database(imported_path) as imported_db:
        imported_ids = CircuitImporter(imported_db).import_from_json(
            projection_path
        )
        print(f'\nimported projection into {os.path.relpath(imported_path)}:')
        for version_id in imported_ids:
            circuit = imported_db.get_circuit_by_id(version_id)
            summary = imported_db.get_circuit_summary(
                circuit['circuit_name'], circuit['version']
            )
            print(
                f'  {circuit["circuit_name"]} {circuit["version"]} '
                f'stage={circuit["stage"]}: '
                f'{len(summary["devices"])} devices, '
                f'{len(summary["nets"])} nets, '
                f'{len(summary["simulation_results"])} results'
            )

    # Listing by stage: latest_at_stage returns the newest sealed snapshot of
    # each circuit at a stage. A circuit with only building snapshots is left
    # out there; list_circuits(stage) shows that work in progress instead.
    with Database(DB_PATH) as database:
        print('\nlatest sealed snapshot by design stage:')
        for stage in DesignStage:
            latest = database.latest_at_stage(stage)
            labels = [
                f'{record["circuit_name"]} {record["version"]}'
                for record in latest
            ] or ['(none)']
            print(f'  {stage.value}: {", ".join(labels)}')


if __name__ == '__main__':
    main()
