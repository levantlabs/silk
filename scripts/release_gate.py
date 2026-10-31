"""Run the local release gate.

Run from the candidate checkout root with .venv/bin/python. Children run
sequentially in the foreground; every command has a retained log.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

try:  # run as a script (scripts/ on sys.path) or imported as scripts.release_gate
    from fast_chunks import chunk_paths, test_files
except ImportError:
    from scripts.fast_chunks import chunk_paths, test_files


AUDIT = 'tests/test_examples_hierarchy_audit.py'
WHEEL_TESTS = (
    'tests/test_packaging.py::test_wheel_is_complete_and_importable',
    'tests/test_design_loop_walkthrough.py::test_walkthrough_runs_from_installed_wheel',
)


class GateError(RuntimeError):
    """A required gate could not be verified."""


def git(root, *args):
    return subprocess.check_output(
        ['git', '-C', str(root), *args], text=True).strip()


def require_candidate(root, candidate):
    if Path(git(root, 'rev-parse', '--show-toplevel')).resolve() != root:
        raise GateError('Run from the checkout root.')
    if git(root, 'rev-parse', 'HEAD') != candidate:
        raise GateError('HEAD changed or does not match --candidate.')
    dirty = git(root, 'status', '--porcelain', '--untracked-files=normal')
    if dirty:
        raise GateError('Commit or remove working-tree changes first:\n' + dirty)


def fast_chunks(root):
    """The shared chunk table, rejecting empty chunks and unplaced test files."""
    chunks = chunk_paths(root)
    for number, paths in enumerate(chunks, 1):
        if not paths:
            raise GateError(f'Empty fast chunk {number}.')
    covered = []
    for name in (n for paths in chunks for n in paths):
        path = root / name
        if path.is_dir():
            covered.extend(p.relative_to(root).as_posix()
                           for p in [*path.rglob('test_*.py'), *path.rglob('*_test.py')])
        else:
            covered.append(name)
    if sorted(covered) != sorted(test_files(root)):
        raise GateError('Fast chunk partition is incomplete or overlaps; '
                        'update CHUNKS in scripts/fast_chunks.py.')
    return chunks


def pytest_counts(report, output):
    """JUnit is authoritative for outcomes; pytest's summary adds deselection."""
    cases = ET.parse(report).getroot().findall('.//testcase')
    counts = dict(passed=0, failed=0, errors=0, skipped=0, xfailed=0,
                  deselected=0)
    for case in cases:
        if case.find('failure') is not None:
            counts['failed'] += 1
        elif case.find('error') is not None:
            counts['errors'] += 1
        elif case.find('skipped') is not None:
            key = ('xfailed' if case.find('skipped').get('type') == 'pytest.xfail'
                   else 'skipped')
            counts[key] += 1
        else:
            counts['passed'] += 1
    matches = re.findall(r'(\d+) deselected', output)
    if matches:
        counts['deselected'] = int(matches[-1])
    return counts


class Gate:
    def __init__(self, root, out, candidate):
        self.root, self.out = root, out
        self.env = dict(os.environ, MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
        # Inherited import paths or pytest selectors must not change the gate.
        for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTEST_ADDOPTS',
                    'PYTEST_PLUGINS', 'SILK_ALLOW_NO_NODE'):
            self.env.pop(key, None)
        self.record = {
            'candidate': candidate,
            'started_utc': datetime.now(timezone.utc).isoformat(),
            'checkout': str(root), 'python': sys.version,
            'interpreter': sys.executable, 'platform': platform.platform(),
            'environment': {key: self.env.get(key) for key in (
                'PDK_ROOT', 'SILK_NGSPICE_BIN', 'SILK_XYCE_BIN',
                'SILK_FIREFOX', 'SILK_CHROME', 'MPLBACKEND')},
            'stages': [], 'artifacts': [], 'status': 'RUNNING',
        }

    def save(self):
        (self.out / 'summary.json').write_text(
            json.dumps(self.record, indent=2) + '\n')

    def run(self, name, command, *, cwd=None, pytest=False, require_all=False):
        cwd = cwd or self.root
        command = [str(arg) for arg in command]
        report = self.out / f'{name}.xml'
        if pytest:
            command += ['-q', '-ra', '--color=no', f'--junitxml={report}']
        log = self.out / f'{name}.log'
        stage = dict(name=name, command=command, cwd=str(cwd),
                     log=str(log), status='RUNNING')
        self.record['stages'].append(stage)
        self.save()
        print(f'\n[{name}] ({cwd}) {shlex.join(command)}', flush=True)
        started = time.monotonic()
        try:
            with log.open('w') as stream:
                with subprocess.Popen(command, cwd=cwd, env=self.env,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True) as child:
                    try:
                        for line in child.stdout:
                            stream.write(line)
                            stream.flush()
                            print(line, end='', flush=True)
                        stage['returncode'] = child.wait()
                    except BaseException:
                        child.terminate()
                        child.wait()
                        raise
            if pytest:
                stage['counts'] = pytest_counts(report, log.read_text())
            if stage['returncode']:
                raise GateError(f'{name} exited {stage["returncode"]}; see {log}')
            if pytest:
                counts = stage['counts']
                if not counts['passed'] or counts['failed'] or counts['errors']:
                    raise GateError(f'{name} did not pass tests: {counts}')
                if require_all and (counts['skipped'] or counts['xfailed']):
                    raise GateError(f'{name} skipped required tests: {counts}')
            stage['status'] = 'PASS'
        except BaseException:
            stage['status'] = 'FAIL'
            raise
        finally:
            stage['seconds'] = round(time.monotonic() - started, 2)
            self.save()

    def summary(self):
        lines = [f'Local release gate: {self.record["status"]}',
                 f'Candidate: {self.record["candidate"]}',
                 f'Environment: {self.record["platform"]}; Python {platform.python_version()}']
        for stage in self.record['stages']:
            counts = ', '.join(f'{n} {key}' for key, n in stage.get('counts', {}).items())
            lines.append(f'{stage["status"]}: {stage["name"]} {counts}'.rstrip())
        if 'error' in self.record:
            lines.append('Error: ' + self.record['error'])
        for artifact in self.record['artifacts']:
            lines.append(f'SHA256 {artifact["name"]}: {artifact["sha256"]}')
        if self.record['status'] != 'PASS':
            lines.append('Only completed stages above have evidence; later stages did not run.')
        lines += ['Coverage: one local Python/OS/dependency set.',
                  f'Evidence: {self.out}']
        summary = '\n'.join(lines) + '\n'
        (self.out / 'summary.txt').write_text(summary)
        print('\n' + summary, flush=True)


def installed_smoke(gate, python, name, foreign):
    # -I and a foreign cwd make a source checkout unable to satisfy this probe.
    probe = """
from pathlib import Path
import sys
import silk
import silk.sources, silk.lint, silk.metrics, silk.faults, silk.specs
import silk.codegen, silk.db, silk.layout, silk.simulators
from silk.pdk.registry import available
from silk.circuit.port_roles import RoleRuleset
from silk.simulators._loader import load_simulator_config
assert Path(silk.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), silk.__file__
assert 'sky130' in available(), available()
RoleRuleset.default()
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('error')
    for backend in ('ngspice', 'xyce', 'behavioral'):
        load_simulator_config(backend)
print('INSTALLED-OK', silk.__version__, silk.__file__)
"""
    gate.run(f'{name}-import', [python, '-I', '-c', probe], cwd=foreign)
    gate.run(f'{name}-behavioral',
             [python, '-I', '-m', 'silk.examples.behavioral_example'], cwd=foreign)


def execute(gate):
    root, out = gate.root, gate.out
    python = root / '.venv/bin/python'
    candidate = gate.record['candidate']
    require_candidate(root, candidate)
    chunks = fast_chunks(root)
    gate.run('environment', [python, '-m', 'pip', 'freeze'])
    gate.run('doctor', [python, '-m', 'silk.doctor', '--json'])
    checks = json.loads((out / 'doctor.log').read_text())
    if not any(c['name'] == 'pdk sky130' and c['status'] == 'ok' for c in checks):
        raise GateError('The release gate requires installed sky130 models; see docs/installing-the-pdk.md.')
    gate.run('node', ['node', '--version'])
    node_version = (out / 'node.log').read_text().strip()
    if not re.fullmatch(r'v[0-9]+\.[0-9]+\.[0-9]+', node_version) or int(node_version[1:].split('.')[0]) < 20:
        raise GateError(f'Node >=20 is required; found {node_version!r}.')
    # No working-tree files (including ignored example output) enter this clone.
    clone = out / 'source'
    gate.run('clone', ['git', 'clone', '--no-local', '--no-checkout', str(root), str(clone)])
    gate.run('checkout', ['git', 'checkout', '--detach', candidate], cwd=clone)
    require_candidate(clone, candidate)
    # Build before tests can generate files in the otherwise pristine clone.
    gate.run('build', [python, '-m', 'build', '--outdir', out / 'dist'], cwd=clone)
    wheels, sdists = list((out / 'dist').glob('*.whl')), list((out / 'dist').glob('*.tar.gz'))
    if len(wheels) != 1 or len(sdists) != 1:
        raise GateError('Expected exactly one wheel and one sdist.')
    for artifact in wheels + sdists:
        gate.record['artifacts'].append(dict(
            name=artifact.name, sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()))
    gate.save()
    gate.run('metadata', [python, '-m', 'twine', 'check', '--strict', *wheels, *sdists])
    for number, paths in enumerate(chunks, 1):
        gate.run(f'fast-{number}', [python, '-m', 'pytest', *paths], pytest=True)
    gate.run('slow-examples-audit', [python, '-m', 'pytest', '-m', 'slow', AUDIT],
             pytest=True, require_all=True)
    foreign = out / 'installed-run'
    foreign.mkdir()
    for name, artifact in (('wheel', wheels[0]), ('sdist', sdists[0])):
        venv = out / f'{name}-venv'
        gate.run(f'{name}-venv', [python, '-m', 'venv', venv])
        installed_python = venv / 'bin/python'
        requirement = f'{artifact}[dev]' if name == 'wheel' else str(artifact)
        gate.run(f'{name}-install', [installed_python, '-m', 'pip', 'install', requirement])
        gate.run(f'{name}-dependencies', [installed_python, '-m', 'pip', 'freeze'])
        installed_smoke(gate, installed_python, name, foreign)
        if name == 'wheel':
            gate.run('wheel-walkthrough', [installed_python, '-I', '-m',
                     'silk.examples.design_loop_walkthrough', out / 'walkthrough'], cwd=foreign)
            gate.run('wheel-regressions', [installed_python, '-m', 'pytest', '-m', 'slow',
                                         *WHEEL_TESTS], cwd=clone, pytest=True, require_all=True)
    require_candidate(root, candidate)
    # The regression tests may create ignored build files, but never edit tracked files.
    require_candidate(clone, candidate)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, help='Exact commit or ref; must resolve to HEAD')
    parser.add_argument('--output', type=Path, required=True,
                        help='New evidence directory outside the checkout (retained on failure)')
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    out = args.output.expanduser().resolve()
    if out.is_relative_to(root):
        parser.error('--output must be outside the checkout')
    try:
        candidate = git(root, 'rev-parse', '--verify', args.candidate + '^{commit}')
        out.mkdir(parents=True, exist_ok=False)
    except (OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    gate = Gate(root, out, candidate)
    try:
        execute(gate)
        gate.record['status'] = 'PASS'
    except (Exception, KeyboardInterrupt) as exc:
        gate.record['status'] = 'FAIL'
        gate.record['error'] = str(exc) or type(exc).__name__
    finally:
        gate.record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        gate.save()
        gate.summary()
    return 0 if gate.record['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
