"""Execute examples on the manifest surface without simulator discovery.

Collection and execution share the API audit's language classification and
fresh-process runner. Literal formula blocks are reported as non-Python;
declared Python and console sessions must compile and import successfully.
"""
from collections import Counter

import pytest

from scripts import docstring_audit as audit

ENTRY_POINTS = (
    'silk.Module', 'silk.ModuleTestbench', 'silk.Signal', 'silk.Spec',
    'silk.Measurement', 'silk.CalibrationBinding', 'silk.pdk.load_pdk',
    'silk.elements.ElementLibrary', 'silk.sources.VoltageSource',
    'silk.sources.CurrentSource', 'silk.Transient', 'silk.AC', 'silk.DC',
    'silk.Noise', 'silk.Corner', 'silk.VoltageDomain', 'silk.annotation_coverage',
    'silk.export_schema', 'silk.export_source_schema', 'silk.export_metric_schema',
    'silk.VariableSpace', 'silk.Optimizer', 'silk.Database', 'silk.Generator',
    'silk.faults.FaultSet', 'silk.faults.FaultCampaign', 'silk.RuleRegistry',
    'silk.estimate_layout', 'silk.Variable', 'silk.simulation.Op',
    'silk.sources.Sine', 'silk.sources.Pulse', 'silk.sources.PWL',
)


@pytest.fixture(scope='module')
def rows():
    return [audit.score(entry) for entry in audit.collect(audit.load_manifest())]


def test_python_blocks_compile_and_silk_imports_resolve(rows):
    failures = [f'{row.name}: {block["status"]}'
                for row in rows for block in row.examples
                if block['status'].startswith(('compile:', 'import:'))]
    assert not failures, '\n'.join(failures)


def test_console_sessions_run_in_fresh_processes(rows, tmp_path, monkeypatch):
    # Exercise the bare profile in the child processes, even on a configured
    # machine. These temporary test overrides never change the settings file.
    monkeypatch.setenv('PDK_ROOT', str(tmp_path / 'absent-pdk'))
    monkeypatch.setenv('SILK_NGSPICE_BIN', str(tmp_path / 'absent-ngspice'))
    monkeypatch.setenv('SILK_XYCE_BIN', str(tmp_path / 'absent-xyce'))
    executed = audit.run_examples(rows)
    assert executed, 'No console sessions were collected from the manifest'
    report = audit.report(rows, base='pytest', worst=0, executed=executed)
    report = '## Doctest execution' + report.split('## Doctest execution', 1)[1]
    counts = Counter(status.split(' ')[0] for _, status, _ in executed)
    print('Docstring sessions: ' + ', '.join(
        f'{label}={counts[key]}' for label, key in
        [('passed', 'passed'), ('failed', 'failed'), ('skipped', 'skipped'),
         ('not-run', 'not')]))
    # Skips are visible in the report; an unexecuted session is a gate failure.
    assert all(status in ('passed', 'skipped') for _, status, _ in executed), report
    required = {audit._resolve(name) for name in ENTRY_POINTS}
    required_targets = {(obj.__module__, obj.__qualname__) for obj in required}
    targets = {row.name: row.target for row in rows}
    assert all(status == 'passed' for name, status, _ in executed
               if targets[name] in required_targets), report


@pytest.mark.parametrize('name', ENTRY_POINTS)
def test_entry_point_has_a_collected_console_session(rows, name):
    obj = audit._resolve(name)
    target = (obj.__module__, obj.__qualname__)
    assert any(row.target == target
               and any(block['kind'] == 'doctest' for block in row.examples)
               for row in rows), f'{name} needs a runnable console example'
