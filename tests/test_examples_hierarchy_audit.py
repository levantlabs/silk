"""Realism audit: structure extraction over every module every shipped
example builds (TASK-126 guard 2).

TASK-125 found that the structure fingerprint had omitted every child
instance's port bindings since the tracker shipped — and the symptom was in
plain sight the whole time: ``extract_structure(...)['hierarchy']`` came back
empty for every hierarchical example. Unit fixtures never noticed because
they are built to exercise the extractor, not to look like a design. This
test looks like the designs.

Method (the TASK-38/93 promotion-gate method): each ``silk/examples/*.py``
runs in its own interpreter via :mod:`tests._examples_hierarchy_driver`,
with SPICE stubbed at ``Module.run_simulator``, ``SimulatorBackend.run`` and
both runners (plus a fake ``ngspice``/``Xyce`` first on ``PATH``), every
``Module.__init__`` recorded, in two passes — the file executed as
``__main__`` (``runpy``), then every module-level ``build_*``/``make_*``
builder called directly with ``pdk``/``out_dir``/``n`` synthesized. For every
recorded module the driver walks the tree itself and requires that
``extract_structure`` yields exactly one hierarchy record per child instance
at every depth, with the instance path and a non-empty port map equal to the
parent instance record's ``pin_list`` bindings, and that
``_extract_module_structure`` lists exactly the children at every level.

Nothing is swallowed. The stub raises a ``BaseException`` subclass so a pass
stops exactly at its first simulation and says so by type; any other
exception in a pass, a builder or the audit fails the example here with its
traceback. Builders that need a live argument (a comparator, a knob dict)
are reported as skipped by name, never silently.

Slow-marked: ~70 interpreters, each loading the PDK, so it stays out of the
fast profile but in the ``-m slow`` gate.
"""
import glob
import json
import os
import subprocess
import sys

import pytest

from tests._env import needs_pdk

_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
_EXAMPLES = os.path.join(_REPO, 'silk', 'examples')
_DRIVER = os.path.join(_REPO, 'tests', '_examples_hierarchy_driver.py')

pytestmark = [pytest.mark.slow, needs_pdk]


def _example_paths():
    paths = [p for p in sorted(glob.glob(os.path.join(_EXAMPLES, '*.py')))
             if os.path.basename(p) != '__init__.py']
    assert paths, f'no examples found under {_EXAMPLES!r} — glob is broken'
    return paths


def _run_driver(example, tmp_path):
    shim = tmp_path / 'bin'
    shim.mkdir()
    marker = tmp_path / 'invoked.txt'
    for name in ('ngspice', 'Xyce'):
        exe = shim / name
        exe.write_text(f'#!/bin/sh\necho {name} >> {marker}\nexit 1\n')
        exe.chmod(0o755)
    env = dict(os.environ,
               PATH=f'{shim}{os.pathsep}' + os.environ.get('PATH', ''),
               MPLBACKEND='Agg')
    report = tmp_path / 'report.json'
    cp = subprocess.run([sys.executable, _DRIVER, example, str(report)],
                        cwd=str(tmp_path), env=env, capture_output=True,
                        text=True, timeout=900)
    assert report.exists(), (
        f'driver produced no report (rc={cp.returncode}):\n{cp.stderr[-3000:]}')
    data = json.loads(report.read_text())
    assert not marker.exists(), (
        'a stub was bypassed — a simulator binary was invoked: '
        + marker.read_text().strip().replace('\n', ', '))
    return cp, data


def _describe(record):
    if record['outcome'] == 'error':
        return f"{record['type']}: {record['detail']}\n{record['traceback']}"
    return record.get('detail', '')


@pytest.mark.parametrize('example', _example_paths(),
                         ids=[os.path.basename(p)[:-3] for p in _example_paths()])
def test_every_child_instance_yields_one_complete_hierarchy_record(example, tmp_path):
    cp, data = _run_driver(example, tmp_path)
    name = data['example']
    assert data['driver_error'] is None, (
        f'{name}: the driver itself failed (rc={cp.returncode}):\n'
        f'{data["driver_error"]}\n--- log tail ---\n{data["log_tail"]}')

    # Every exception that is not the simulation stub is a failure, in
    # pass 1 and in every builder of pass 2. A ``sys.exit(0)`` from a
    # ``main()`` is a completed run; any other exit code is a failure.
    errors = []
    records = [('__main__ run', data['pass1_main'])] + [
        (f'builder {r["builder"]}()', r) for r in data['pass2_builders']]
    for label, record in records:
        if record['outcome'] == 'error':
            errors.append(f'{label}: {_describe(record)}')
        elif record['outcome'] == 'exit':
            accepted = (0, None)
            if record['code'] not in accepted:
                errors.append(f'{label}: SystemExit({record["code"]!r})\n'
                              f'{record["traceback"]}')
    assert not errors, (
        f'{name}: {len(errors)} non-stub exception(s):\n' + '\n'.join(errors)
        + f'\n--- log tail ---\n{data["log_tail"]}')

    # The audit proper: print every offender the driver found, never a sample.
    modules = data['modules']
    offenders = [m for m in modules if not m['ok']]
    hierarchical = [m for m in modules if m['instances_all_depths']]
    main_outcome = data['pass1_main']['outcome']
    if main_outcome == 'exit':
        main_outcome = f'exit({data["pass1_main"]["code"]!r})'
    summary = (
        f'{name}: __main__ {main_outcome}; builders '
        + ', '.join(f'{b["builder"]}={b["outcome"]}' for b in data['pass2_builders'])
        + f'; {len(modules)} module(s) audited, {len(hierarchical)} hierarchical, '
        + f'{sum(m["instances_all_depths"] for m in hierarchical)} child instance(s) '
        + f'over all depths, {len(offenders)} offender(s)')
    print(summary)
    for record in data['pass2_builders']:
        if record['outcome'] == 'skipped':
            print(f'  skipped builder {record["builder"]}(): {record["detail"]}')
    for m in modules:
        if m.get('unbound_by_example'):
            print(f'  {m["class"]} {m["name"]!r} leaves {m["unbound_by_example"]} '
                  f'unbound on purpose; both extractors raised naming it')
    assert not offenders, (
        summary + '\n' + '\n'.join(
            f'  {m["class"]} {m["name"]!r}: {m["instances_all_depths"]} child '
            f'instance(s), {m["records"]} record(s): ' + '; '.join(m['problems'])
            for m in offenders))
    for m in hierarchical:
        if not m.get('unbound_by_example'):
            assert m['records'] == m['instances_all_depths'], m
