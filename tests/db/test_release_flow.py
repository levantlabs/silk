"""Executable public contract for the schema-v1 database workflow."""

from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk.circuit.modules_results import CurrentAnnotationReport
from silk.circuit.spec_report import SpecReport
from silk.db import Database
from silk.elements import elements


def _build_release_amp(pdk, tmp_dir, *, db_path=None):
    module = Module(
        'PDK_skywater130',
        tmp_dir,
        'release_amp.cir',
        pdk_details=pdk,
        module_name='release_amp',
        db_path=db_path,
    )
    module.define_module_ports({
        'IN': Port.analog_input(),
        'OUT': Port.analog_output(),
    })
    module.add_net('mid')
    module.add(
        elements.resistor('r1', p='IN', n='mid', value=1000.0)
    )
    module.add(
        elements.resistor('r2', p='mid', n='OUT', value=2000.0)
    )
    specs = {
        'gain': module.declare_spec(
            'DC gain', 'gain_dB', '>=', 30.0, units='dB', key='gain'
        ),
        'bandwidth': module.declare_spec(
            'Bandwidth', 'bandwidth_Hz', '>=', 1e6,
            units='Hz', key='bandwidth'
        ),
    }
    return module, specs


def _fake_annotated_testbench(pdk, tmp_dir, db_path):
    definition, _specs = _build_release_amp(pdk, tmp_dir)
    testbench = Module(
        'PDK_skywater130',
        tmp_dir,
        'release_amp_tb.cir',
        pdk_details=pdk,
        db_path=db_path,
    )
    testbench.add_net('input')
    testbench.add_module(
        definition,
        'dut',
        nets={'IN': 'input', 'OUT': '0'},
    )
    stored_definition = testbench.module_dict['release_amp']['device']
    mid = stored_definition.net_dict['mid']
    for index, binding in enumerate(mid.pins):
        sign = 1.0 if index == 0 else -1.0
        binding.avg_current = sign * 1e-3
        binding.rms_current = 1e-3
        binding.peak_current = 1.2e-3
        binding.current_vector = f'i(v.fixture.{binding.instance})'
        binding._current_source = 'spice'
    mid.avg_current = 1e-3
    mid.rms_current = 1e-3
    mid.peak_current = 1.2e-3
    mid._current_source = 'spice'
    return testbench


def test_schema_v1_release_flow_without_simulator(pdk, tmp_dir, tmp_path):
    db_path = str(tmp_path / 'release-flow.db')
    source, _source_specs = _build_release_amp(
        pdk, tmp_dir, db_path=db_path
    )
    source_id = source.create_circuit_version(
        'schema-v1 release-flow fixture'
    )
    source.store_results(
        {
            'slew_V_per_us': (4.2, 'V/us'),
            'convergence': ('converged', None),
        },
        source='user',
        sim_type='ac',
        corner='tt',
        temperature=27.0,
        voltage=1.8,
        tool_name='release-fixture',
        conditions={'load_pf': 2.0},
    )
    report = SpecReport(
        'release_characterization',
        'release_amp',
        {'gain_dB': 32.5, 'bandwidth_Hz': 2.4e6},
    )
    report.check(
        'DC gain', 'gain_dB', '>=', 30.0,
        spec_key='gain', owner_path='$'
    )
    report.check(
        'Bandwidth', 'bandwidth_Hz', '>=', 1e6,
        spec_key='bandwidth', owner_path='$'
    )
    stored_report = source.store_report(
        report,
        source='user',
        sim_type='ac',
        corner='tt',
        temperature=27.0,
        voltage=1.8,
        tool_name='release-fixture',
        conditions={'load_pf': 2.0},
    )
    assert stored_report == {
        'metrics_stored': 2,
        'specs_stored': 2,
        'skipped': {},
    }
    source.close()

    testbench = _fake_annotated_testbench(pdk, tmp_dir, db_path)
    current_versions = testbench.store_net_currents(
        CurrentAnnotationReport(
            results_file='/tmp/release-fixture.raw',
            analysis='transient',
            window=(1e-6,),
        ),
        corner='tt',
        temperature=27.0,
        voltage=1.8,
        tool_name='fixture-ngspice',
    )
    assert current_versions == {'release_amp': source_id}
    testbench.close()

    # Historical queries need only the SQLite file and a version UUID; no
    # circuit reconstruction or live Python Module is required.
    with Database(db_path) as history:
        historical_metrics = history.get_simulation_results(
            source_id, include_ancestors=False
        )
        historical_specs = history.get_stored_specs(source_id)
        current_origin, historical_currents = history.get_net_current_set(
            source_id, 'tt'
        )
    assert {record.parameter for record in historical_metrics} >= {
        'slew_V_per_us', 'convergence', 'gain_dB', 'bandwidth_Hz'
    }
    assert all(
        record.origin_version_id == source_id
        for record in historical_metrics
    )
    assert {record.spec_key for record in historical_specs} == {
        'gain', 'bandwidth'
    }
    assert all(
        measurement.origin_version_id == source_id
        for declaration in historical_specs
        for measurement in declaration.measurements
    )
    assert current_origin == source_id
    assert historical_currents

    rebuilt, rebuilt_specs = _build_release_amp(
        pdk, tmp_dir, db_path=db_path
    )
    assert rebuilt.create_circuit_version() == source_id
    assert rebuilt.query_results(
        parameter='convergence',
        source='user',
        corner='tt',
        conditions={'load_pf': 2.0},
    )[0].value_text == 'converged'
    assert rebuilt.restore_spec_measurements(
        source='user', corner='tt', tool_name='release-fixture'
    ) == 2
    assert rebuilt_specs['gain'].latest().origin_version_id == source_id
    assert rebuilt.load_net_currents('tt') > 0
    assert rebuilt.net_dict['mid'].peak_current == 1.2e-3

    promoted_id = rebuilt.promote(
        'stable', description='release-flow promotion'
    )
    inherited_metrics = rebuilt.query_results(
        parameter='gain_dB', source='user', corner='tt'
    )
    assert inherited_metrics
    assert {
        record.origin_version_id for record in inherited_metrics
    } == {source_id}
    inherited_specs = rebuilt.get_stored_specs()
    assert {
        measurement.origin_version_id
        for declaration in inherited_specs
        for measurement in declaration.measurements
    } == {source_id}
    current_origin, rows = rebuilt._ensure_tracker().db.get_net_current_set(
        promoted_id, 'tt'
    )
    assert current_origin == source_id
    assert rows
    rebuilt.close()
