"""Every Python snippet in the user-facing docs must at least be real Python
against the current silk, and the complete workflow examples must run.

Extends the README-fence guard (``tests/test_readme_lead_example.py``) to
``docs/quickstart.md``, ``docs/api-reference.md`` and ``docs/agent-guide.md``.
Most fences there are deliberately fragments (they lean on a ``pdk`` or ``ckt``
built earlier on the page, or use ``...`` placeholders), so they cannot all be
executed; what *can* be checked mechanically for every one of them is:

* it **compiles** — a fence that is not valid Python is a typo the reader
  will copy;
* every ``from silk... import name`` / ``import silk...`` it contains
  **resolves** against the installed package — the cheapest possible detector
  of a renamed or removed public name surviving in the docs.

The quickstart's §1 behavioral-inverter block is self-contained and PDK-free
(it loads the packaged sky130 *catalog* with a placeholder library path), so
it is executed verbatim and its documented printed result asserted — the same
"the published text is the fixture" discipline as the README guard. The agent
guide's two-engine workflow is also executed verbatim: transient/DC without
external dependencies, plus its SPICE continuation when ngspice is available. The
PDK/ngspice-dependent snippets on those pages are hand-verified by execution
and the results recorded on the backlog task that last audited them.
"""
import ast
import importlib
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests._env import needs_ngspice, needs_pdk

_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DOCS = ('docs/quickstart.md', 'docs/api-reference.md', 'docs/agent-guide.md')


def _fences(doc):
    """(line_number, dedented_code) for every ```python fence in *doc*."""
    text = open(os.path.join(_REPO, doc), encoding='utf-8').read()
    out = []
    for m in re.finditer(r'^[ \t]*```python[ \t]*\n(.*?)^[ \t]*```', text, re.S | re.M):
        lineno = text[:m.start()].count('\n') + 2
        out.append((lineno, textwrap.dedent(m.group(1))))
    assert out, f'{doc}: no ```python fences found — the regex or the doc changed shape'
    return out


@pytest.mark.parametrize('doc', DOCS)
def test_every_python_fence_compiles(doc):
    bad = []
    for lineno, code in _fences(doc):
        try:
            compile(code, f'{doc}:{lineno}', 'exec')
        except SyntaxError as err:
            bad.append(f'{doc}:{lineno + (err.lineno or 1) - 1}: {err.msg}: {err.text!r}')
    assert bad == [], 'python fences that are not valid Python:\n  ' + '\n  '.join(bad)


def test_api_reference_empty_report_transcripts_run_as_written(tmp_path, monkeypatch, capsys):
    """The reference's report and diff transcripts come from its printed code."""
    monkeypatch.chdir(tmp_path)
    text = Path(_REPO, 'docs/api-reference.md').read_text(encoding='utf-8')
    namespace = {}
    from silk import diff_reports
    namespace['diff_reports'] = diff_reports
    for marker in ("from silk import Module, generate_report\n",
                   "diff_reports(report, report, label_a='schematic'"):
        block = next(code for _, code in _fences('docs/api-reference.md')
                     if code.startswith(marker))
        expected = text.split(block, 1)[1].split('```text\n', 1)[1].split('```', 1)[0]
        exec(compile(block, 'docs/api-reference.md', 'exec'), namespace)
        assert capsys.readouterr().out == expected


def _resolves(module, name):
    """``from module import name`` — an attribute, or an importable submodule."""
    try:
        mod = importlib.import_module(module)
    except Exception as exc:                       # noqa: BLE001 — report, don't crash
        return f'cannot import {module}: {exc}'
    if hasattr(mod, name):
        return None
    try:
        importlib.import_module(f'{module}.{name}')
    except Exception:
        return f'{module} has no {name!r}'
    return None


def test_agent_guide_transient_step_input_levels():
    """The published follower stimulus steps from 0.9 V to 0.95 V."""
    import numpy as np
    from silk import sources

    blocks = [code for _, code in _fences('docs/agent-guide.md')
              if 'def _attach_transient_test(' in code]
    assert len(blocks) == 1
    calls = [node for node in ast.walk(ast.parse(blocks[0]))
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == 'VoltageSource'
             and node.args and isinstance(node.args[0], ast.Constant)
             and node.args[0].value == 'vinp']
    assert len(calls) == 1
    source = eval(compile(ast.Expression(calls[0]), 'docs/agent-guide.md', 'eval'),
                  {'sources': sources})
    assert source.to_numpy(np.array([0.5e-6, 1.5e-6])) == pytest.approx([0.9, 0.95])


@pytest.mark.parametrize('doc', DOCS)
def test_every_silk_import_in_a_fence_resolves(doc):
    bad = []
    for lineno, code in _fences(doc):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue                                # reported by the compile test
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split('.')[0] == 'silk':
                for alias in node.names:
                    problem = _resolves(node.module, alias.name)
                    if problem:
                        bad.append(f'{doc}:{lineno + node.lineno - 1}: {problem}')
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] == 'silk':
                        try:
                            importlib.import_module(alias.name)
                        except Exception as exc:    # noqa: BLE001
                            bad.append(f'{doc}:{lineno + node.lineno - 1}: cannot import {alias.name}: {exc}')
    assert bad == [], 'silk imports in doc fences that do not resolve:\n  ' + '\n  '.join(bad)


def _quickstart_behavioral_inverter_block():
    for _lineno, code in _fences('docs/quickstart.md'):
        if "inv.set_behavioral_model(" in code and 'simulate_behavioral(' in code:
            return code
    raise AssertionError(
        'docs/quickstart.md no longer has the self-contained behavioral inverter '
        'fence (inv.set_behavioral_model(...) + simulate_behavioral) — '
        'update this test alongside the quickstart.')


def test_quickstart_behavioral_inverter_block_runs_and_prints_its_documented_result(tmp_path):
    """The §1 'runnable as-is on a fresh install' block, run exactly as printed,
    in a foreign cwd, with no PDK and no simulator reachable."""
    code = _quickstart_behavioral_inverter_block()
    env = dict(os.environ, PDK_ROOT=str(tmp_path / 'no-pdk'),
               SILK_NGSPICE_BIN=str(tmp_path / 'no-ngspice'), MPLBACKEND='Agg',
               PYTHONPATH=_REPO)
    cp = subprocess.run([sys.executable, '-c', code], cwd=str(tmp_path), env=env,
                        capture_output=True, text=True, timeout=120)
    assert cp.returncode == 0, cp.stderr[-3000:]
    # The comment under the block promises exactly this line.
    assert 'vout: min=0.00 V  max=1.80 V  (81 samples)' in cp.stdout, cp.stdout


def test_quickstart_documents_the_behavioral_example_output(tmp_path):
    """§1 quotes what `python -m silk.examples.behavioral_example` prints;
    those four result lines must still be what it prints."""
    env = dict(os.environ, PDK_ROOT=str(tmp_path / 'no-pdk'),
               SILK_NGSPICE_BIN=str(tmp_path / 'no-ngspice'), MPLBACKEND='Agg',
               PYTHONPATH=_REPO)
    cp = subprocess.run([sys.executable, '-m', 'silk.examples.behavioral_example'],
                        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=120)
    assert cp.returncode == 0, cp.stderr[-3000:]
    quickstart = open(os.path.join(_REPO, 'docs/quickstart.md'), encoding='utf-8').read()
    for line in ("Available signals: ['time', 'v(gnd)', 'v(vcomp)', 'v(vfilt)', 'v(vin)', 'v(vref)']",
                 'vin     min=0.000 V  max=2.000 V  mean=1.041 V  (121 samples)',
                 'vcomp   min=0.000 V  max=1.800 V  mean=0.937 V  (121 samples)',
                 'vfilt   min=0.000 V  max=0.721 V  mean=0.419 V  (121 samples)',
                 'Behavioral simulation completed successfully.'):
        assert line in quickstart, f'quickstart no longer quotes: {line!r}'
        assert line in cp.stdout, f'behavioral_example no longer prints: {line!r}\n{cp.stdout}'


def _two_engines_blocks():
    """Execute the published workflow, including its continuation blocks."""
    text = open(os.path.join(_REPO, 'docs/agent-guide.md'), encoding='utf-8').read()
    section = text.split('### One workflow, two engines\n', 1)[1].split('\n## 2.', 1)[0]
    blocks = re.findall(r'^```python\n(.*?)^```', section, re.S | re.M)
    assert len(blocks) == 3, 'update the runnable workflow guard alongside the guide'
    return blocks


def _run_two_engines_blocks(tmp_path, *, spice):
    blocks = _two_engines_blocks()
    code = '\n'.join(blocks if spice else [blocks[0], blocks[2]])
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONPATH=_REPO)
    if not spice:
        env.update(PDK_ROOT=str(tmp_path / 'no-pdk'),
                   SILK_NGSPICE_BIN=str(tmp_path / 'no-ngspice'))
    cp = subprocess.run([sys.executable, '-c', code], cwd=str(tmp_path), env=env,
                        capture_output=True, text=True, timeout=120)
    assert cp.returncode == 0, cp.stdout[-1000:] + cp.stderr[-3000:]
    assert (tmp_path / 'out/two-engines/transfer.png').is_file()
    assert list((tmp_path / 'out/two-engines').rglob('view.html'))


def test_agent_guide_behavioral_transient_and_dc_blocks_run(tmp_path):
    _run_two_engines_blocks(tmp_path, spice=False)


@needs_ngspice
def test_agent_guide_same_test_switches_to_spice(tmp_path):
    _run_two_engines_blocks(tmp_path, spice=True)


@needs_ngspice
def test_quickstart_characterization_recipe_configures_conditions(tmp_path, monkeypatch):
    """Run the published recipe on an ideal amplifier; inspect actual evidence."""
    from silk import Module, PortDir, load_pdk, sources
    import matplotlib.pyplot as plt

    monkeypatch.chdir(tmp_path)
    mod = Module('PDK_skywater130', str(tmp_path), 'amplifier.cir',
                 pdk_details=load_pdk('sky130'), module_name='amplifier')
    mod.define_module_ports({name: PortDir.OUT if name == 'VOUT' else PortDir.IN
                             for name in ('VDD', 'VSS', 'VINP', 'VINM', 'VOUT', 'IBIAS')})
    mod.add_source(sources.VoltageControlledVoltageSource(
        'gain', out_pos='VOUT', out_neg='VSS',
        ctrl_pos='VINP', ctrl_neg='VINM', gain=100))
    blocks = [code for _, code in _fences('docs/quickstart.md')
              if "tb = mod.create_test('ac')" in code]
    assert len(blocks) == 1
    ns = {'mod': mod}
    try:
        exec(compile(blocks[0], 'docs/quickstart.md', 'exec'), ns)
        report = ns['report']
        assert report.passed
        assert report.metrics['gain_dB'] == pytest.approx(40.0)
        conditions = report.operating_conditions
        assert conditions.temperature_c == 27.0
        assert conditions.temperature_basis == 'configured-and-observed'
        assert conditions.supply_voltage_v == 1.8
        assert conditions.supply_source == 'vdd'
        assert conditions.supply_basis == 'configured-and-emitted'
    finally:
        plt.close('all')


def _database_run_store_block():
    blocks = [code for _, code in _fences('docs/database.md')
              if 'run_store_demo' in code]
    assert len(blocks) == 1
    return blocks[0]


def test_database_run_store_requires_explicit_primary_supply(tmp_path, monkeypatch):
    """The two grounded rails cannot establish a primary supply by inference."""
    from silk.exceptions import SimulationConfigurationError

    monkeypatch.setenv('SILK_DOC_OUT', str(tmp_path))
    setup = _database_run_store_block().split('# 3.', 1)[0]
    ns = {}
    try:
        exec(compile(setup, 'docs/database.md', 'exec'), ns)
        ns['tb'].set_temperature(85)
        with pytest.raises(SimulationConfigurationError, match='2 sources are equally supply-shaped'):
            ns['amp'].run_test('ac', corner='ff', voltage=1.8, store=True)
        with ns['Database'](ns['DB_PATH']) as db:
            assert db.get_simulation_results(ns['version_id'], parameter='gain_dB') == []
    finally:
        if 'amp' in ns:
            ns['amp'].close()
        if 'tb' in ns:
            ns['tb'].netlist.close()


@needs_ngspice
@needs_pdk
def test_database_run_store_block_retains_verified_conditions(tmp_path, monkeypatch, capsys):
    """Execute the exact recipe, then compare both readback paths to execution."""
    monkeypatch.setenv('SILK_DOC_OUT', str(tmp_path))
    ns = {}
    try:
        exec(compile(_database_run_store_block(), 'docs/database.md', 'exec'), ns)
        reports = [ns['report'], ns['default_report']]
        assert [r.corner for r in reports] == ['ff', 'tt']
        for report in reports:
            assert report.passed
            conditions = report.operating_conditions
            assert conditions.temperature_c == 85
            assert conditions.temperature_basis == 'configured-and-observed'
            assert conditions.supply_source == 'vsup'
            assert conditions.supply_voltage_v == 1.8
            assert conditions.supply_basis == 'configured-and-emitted'
            assert report.storage == {'metrics_stored': 1, 'specs_stored': 1,
                                      'skipped': {}, 'cached': False}
        rows = ns['rows']
        measurements = ns['stored_gain'].measurements
        assert len(rows) == len(measurements) == 2
        assert {row.corner for row in rows} == {'ff', 'tt'}
        assert {m.corner for m in measurements} == {'ff', 'tt'}
        for report in reports:
            row, = [r for r in rows if r.corner == report.corner]
            measurement, = [m for m in measurements if m.corner == report.corner]
            assert row.temperature == 85
            assert row.voltage == 1.8
            assert row.profile_id is not None
            assert measurement.profile_id == row.profile_id
            assert measurement.passes
            assert row.value == pytest.approx(report.metrics['gain_dB'])
            assert measurement.value == pytest.approx(row.value)
        assert ns['scoped'].value == pytest.approx(ns['report'].metrics['gain_dB'])
        assert ns['default_scoped'].value == pytest.approx(ns['default_report'].metrics['gain_dB'])
        assert 'default corner: tt' in capsys.readouterr().out
    finally:
        if 'amp' in ns:
            ns['amp'].close()
        if 'tb' in ns:
            ns['tb'].netlist.close()


def test_agent_guide_run_accounting_workflow(tmp_path, monkeypatch, capsys):
    """Run the published inventory/inspection code on a real behavioral bench."""
    monkeypatch.chdir(tmp_path)
    ns = {}
    exec(compile(_two_engines_blocks()[0], 'docs/agent-guide.md', 'exec'), ns)
    blocks = [code for _, code in _fences('docs/agent-guide.md')
              if 'entries = mod.list_test_entries()' in code]
    assert len(blocks) == 1
    ns['mod'] = ns['dut']
    exec(compile(blocks[0], 'docs/agent-guide.md', 'exec'), ns)
    assert [e.report_key for e in ns['entries']] == list(ns['results'])
    assert all(r.passed and r.run and r.launch_id for r in ns['results'].values())
    assert 'metric extraction errors: {}' in capsys.readouterr().out

    # Exercise the same published inspection branches with public result objects.
    from silk import CornerResults
    from silk.circuit import TestFailure
    report = next(iter(ns['results'].values()))
    key = ns['entries'][0].report_key
    for outcome, expected in [
        (CornerResults({'tt': report}, failed={'ss': 'simulation failed'}),
         'condition/sample failed: ss simulation failed'),
        (CornerResults({('tt', 0): report}, failed={('ss', 0): 'sample failed'}),
         "condition/sample failed: ('ss', 0) sample failed"),
        (TestFailure('no data', error_type='SimulationAborted'),
         'test error: SimulationAborted no data'),
    ]:
        monkeypatch.setattr(ns['mod'], 'run_all_tests', lambda: {key: outcome})
        exec(compile(blocks[0], 'docs/agent-guide.md', 'exec'), ns)
        assert expected in capsys.readouterr().out


def test_calibration_guide_comparator_snippet_runs_with_ports_params(tmp_path):
    import numpy as np
    from silk import Module, PortDir, sources
    from silk.simulation.behavioral import BehavioralEngine

    snippets = [code for _, code in _fences('docs/design_stage_bridge_user_guide.md')
                if 'def _comparator_model(' in code]
    assert len(snippets) == 1
    comp = Module(None, str(tmp_path / 'comp'), 'comp.cir', None, module_name='comp')
    comp.define_module_ports({'vp': PortDir.IN, 'vm': PortDir.IN, 'vout': PortDir.OUT})
    exec(snippets[0], {'np': np, 'VDD': 1.8, 'comp': comp})
    top = Module(None, str(tmp_path), 'top.cir', None)
    top.add_module(comp, 'comp', nets={'vp': 'vp', 'vm': 'vm', 'vout': 'out'})
    top.add_source(sources.VoltageSource('vp', positive='vp', negative='0', dc=0.4))
    top.add_source(sources.VoltageSource('vm', positive='vm', negative='0', dc=0.5))
    before, _ = BehavioralEngine().run(top, 1, 2)
    np.testing.assert_array_equal(before['out'], [0, 0, 0])
    comp.calibrate_behavioral('offset_voltage', 0.2)
    after, _ = BehavioralEngine().run(top, 1, 2)
    np.testing.assert_array_equal(after['out'], [1.8, 1.8, 1.8])
