"""`python -m silk.doctor` — the environment checker must be right in *both*
directions: green on a working box, and red-with-a-fix on a broken one. The
broken case is synthesized by scrubbing the environment (no PDK_ROOT, empty
PATH, HOME pointing at an empty dir so the ~/pdk fallback can't rescue it),
so it needs neither ngspice nor a PDK to run.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

from silk import doctor
from tests._env import needs_ngspice

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SKY130_LIB = os.path.join(os.environ.get('PDK_ROOT', os.path.expanduser('~/pdk')),
                           'sky130A', 'libs.tech', 'ngspice', 'sky130.lib.spice')
_have_pdk = os.path.exists(_SKY130_LIB)
_have_ngspice = shutil.which('ngspice') is not None


def _run(env, *args):
    return subprocess.run([sys.executable, '-m', 'silk.doctor', *args],
                          capture_output=True, text=True, env=env, cwd=REPO_ROOT)


def _broken_env(tmp_path):
    """A deliberately unusable environment: no PDK_ROOT, no ngspice on PATH."""
    env = {k: v for k, v in os.environ.items()
           if k not in ('PDK_ROOT', 'SILK_XYCE_BIN', 'SILK_NGSPICE_BIN',
                        'SILK_REGION_THRESHOLDS')}
    empty_bin = tmp_path / 'bin'
    empty_bin.mkdir()
    env['PATH'] = str(empty_bin)
    env['SILK_SETTINGS'] = ''
    env['XDG_CONFIG_HOME'] = str(tmp_path / 'config')
    env['HOME'] = str(tmp_path / 'nohome')     # kills the ~/pdk fallback
    return env


def _real_env():
    env = dict(os.environ)
    env.setdefault('PDK_ROOT', os.path.expanduser('~/pdk'))
    return env


class TestBrokenEnvironment:
    @pytest.fixture(scope='class')
    def result(self, tmp_path_factory):
        return _run(_broken_env(tmp_path_factory.mktemp('broken')))

    def test_exits_nonzero(self, result):
        assert result.returncode == 1, result.stdout

    def test_reports_missing_ngspice_and_pdk(self, result):
        assert '[FAIL] ngspice' in result.stdout
        assert '[FAIL] PDK_ROOT' in result.stdout
        assert 'lib not found' in result.stdout

    def test_every_failure_carries_a_fix(self, result):
        """A FAIL or warn line without an actionable next line is a dead end."""
        lines = result.stdout.splitlines()
        for i, line in enumerate(lines):
            if line.startswith(('[FAIL]', '[warn]')):
                assert 'fix:' in lines[i + 1], line

    def test_fixes_are_the_real_commands(self, result):
        assert 'apt install ngspice' in result.stdout
        assert 'export PDK_ROOT=' in result.stdout

    def test_optional_checks_never_fail(self, result):
        """Xyce/matplotlib-backend absence must not gate the exit code."""
        assert '[FAIL] xyce' not in result.stdout
        assert '[FAIL] matplotlib backend' not in result.stdout


@pytest.mark.skipif(not _have_ngspice, reason='ngspice not installed')
@pytest.mark.skipif(not _have_pdk, reason='sky130 ngspice models not installed')
class TestRealEnvironment:
    @pytest.fixture(scope='class')
    def result(self):
        return _run(_real_env())

    @needs_ngspice

    def test_exits_zero(self, result):
        assert result.returncode == 0, result.stdout

    @needs_ngspice

    def test_summary_says_ready(self, result):
        assert 'silk is ready' in result.stdout
        assert '[FAIL]' not in result.stdout

    def test_reports_the_registered_pdks(self, result):
        """Every registered PDK gets a line; only sky130 must be `ok`.

        Registered means silk ships the catalog, not that this machine needs
        the install: a missing or OSDI-incomplete PDK is `[warn]` — honest and
        actionable, but non-gating. FAIL here used to make doctor (and the CI
        job running it) red on every machine missing any one of the three.
        """
        from silk.pdk import registry
        assert '[ ok ] pdk sky130' in result.stdout  # the class's skipif gate
        for name in registry.available():
            assert (f'[ ok ] pdk {name}' in result.stdout
                    or f'[warn] pdk {name}' in result.stdout), name


class TestJsonOutput:
    @pytest.fixture(scope='class')
    def payload(self, tmp_path_factory):
        result = _run(_broken_env(tmp_path_factory.mktemp('json')), '--json')
        assert result.returncode == 1, result.stdout
        return json.loads(result.stdout)

    def test_is_a_list_of_check_records(self, payload):
        assert isinstance(payload, list) and payload
        for rec in payload:
            assert set(rec) == {'name', 'status', 'detail', 'fix'}

    def test_statuses_are_from_the_known_set(self, payload):
        assert {r['status'] for r in payload} <= {'ok', 'FAIL', 'skip', 'warn'}

    def test_failures_carry_a_fix_string(self, payload):
        failures = [r for r in payload if r['status'] == 'FAIL']
        assert failures
        assert all(r['fix'] for r in failures)

    def test_pdk_root_fix_explains_the_installation_root(self, payload):
        row = next(r for r in payload if r['name'] == 'PDK_ROOT')
        assert 'directory that holds your installed PDKs' in row['fix']
        assert '~/pdk, which holds sky130A/' in row['fix']

    def test_stdout_is_json_only(self, tmp_path):
        """Agents pipe this into a parser — no banner may precede the JSON."""
        result = _run(_broken_env(tmp_path), '--json')
        assert result.stdout.lstrip().startswith('[')


class TestChecksInProcess:
    """The check functions are importable and honest without a subprocess."""

    def test_python_check_passes_on_this_interpreter(self):
        assert doctor.check_python()[0].status == doctor.OK

    def test_import_checks_cover_every_hard_dependency(self):
        names = [c.name for c in doctor.check_imports()]
        assert names == [f'import {m}' for m, _dist in doctor.REQUIRED_IMPORTS]

    def test_missing_default_pdk_root_fails_with_a_fix(self, monkeypatch, tmp_path):
        monkeypatch.setenv('HOME', str(tmp_path))
        monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
        monkeypatch.setenv('SILK_SETTINGS', '')
        monkeypatch.delenv('PDK_ROOT', raising=False)
        check = doctor.check_pdk_root()[0]
        assert check.status == doctor.FAIL
        assert 'export PDK_ROOT=' in check.fix

    @pytest.mark.parametrize('kind', ['missing', 'file'])
    def test_configured_pdk_root_must_be_a_directory(self, monkeypatch, tmp_path, kind):
        root = tmp_path / 'pdk'
        if kind == 'file':
            root.touch()
        monkeypatch.setenv('PDK_ROOT', str(root))
        check = doctor.check_pdk_root()[0]
        assert check.status == doctor.FAIL
        assert str(root) in check.detail
        assert 'directory that holds your installed PDKs' in check.fix

    def test_pdk_check_is_registry_driven_not_hardcoded(self, monkeypatch, tmp_path):
        """Every registered manifest gets its own line — no hardcoded sky130."""
        from silk.pdk import registry
        monkeypatch.setenv('PDK_ROOT', str(tmp_path))
        names = [c.name for c in doctor.check_pdks()]
        assert names == [f'pdk {n}' for n in registry.available()]

    def test_missing_lib_reports_the_resolved_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv('PDK_ROOT', str(tmp_path))
        for check in doctor.check_pdks():
            # warn, not FAIL: registered means silk ships the catalog, not
            # that this machine must have the install.
            assert check.status == doctor.WARN
            assert str(tmp_path) in check.detail

    def test_optional_checks_return_ok_or_skip_only(self):
        for fn in doctor.OPTIONAL_CHECKS:
            for check in fn():
                assert check.status in (doctor.OK, doctor.SKIP)


class TestNgspiceOverride:
    """doctor must report the binary the *runner* will invoke. When
    SILK_NGSPICE_BIN is set, "ngspice is on PATH" is the wrong answer twice
    over: it is green while every simulation fails, and it hides which of two
    installed ngspice versions is actually in play (v42 vs v47 — the save-class
    directives need >= 45).
    """

    def test_bogus_override_fails_naming_the_variable(self, monkeypatch, tmp_path):
        monkeypatch.setenv('SILK_NGSPICE_BIN', str(tmp_path / 'not-ngspice'))
        check = doctor.check_ngspice()[0]
        assert check.status == doctor.FAIL
        assert 'SILK_NGSPICE_BIN' in check.detail
        assert str(tmp_path / 'not-ngspice') in check.detail
        assert 'not executable' in check.detail
        assert check.fix and 'SILK_NGSPICE_BIN' in check.fix

    def test_non_executable_file_is_not_accepted(self, monkeypatch, tmp_path):
        """A readable but non-executable file is as unusable as a missing one —
        shutil.which() rejects it, and doctor must too."""
        fake = tmp_path / 'ngspice'
        fake.write_text('#!/bin/sh\n')       # no +x
        monkeypatch.setenv('SILK_NGSPICE_BIN', str(fake))
        assert doctor.check_ngspice()[0].status == doctor.FAIL

    @pytest.mark.skipif(not _have_ngspice, reason='ngspice not installed')
    def test_real_override_is_reported_as_the_source(self, monkeypatch):
        monkeypatch.setenv('SILK_NGSPICE_BIN', shutil.which('ngspice'))
        check = doctor.check_ngspice()[0]
        assert check.status == doctor.OK
        assert '(via SILK_NGSPICE_BIN)' in check.detail

    @pytest.mark.skipif(not _have_ngspice, reason='ngspice not installed')
    def test_empty_override_behaves_as_unset(self, monkeypatch):
        """An empty SILK_NGSPICE_BIN (an unset var expanded by a Makefile/.env)
        must fall back to PATH silently, with no 'via' attribution."""
        monkeypatch.setenv('SILK_NGSPICE_BIN', '')
        check = doctor.check_ngspice()[0]
        assert check.status == doctor.OK
        assert 'via SILK_NGSPICE_BIN' not in check.detail


@pytest.mark.parametrize('json_output', [False, True])
@pytest.mark.parametrize('with_simulator', [False, pytest.param(True, marks=needs_ngspice)])
def test_default_pdk_install_is_recognized(tmp_path, monkeypatch, json_output, with_simulator):
    monkeypatch.setenv('SILK_REGION_THRESHOLDS', str(tmp_path / 'missing-policy.yaml'))
    env = _broken_env(tmp_path)
    if with_simulator:
        from silk import settings
        env['SILK_NGSPICE_BIN'] = shutil.which(settings.get('SILK_NGSPICE_BIN'))
    root = tmp_path / 'nohome' / 'pdk'
    library = root / 'sky130A/libs.tech/ngspice/sky130.lib.spice'
    library.parent.mkdir(parents=True)
    library.touch()
    result = _run(env, *(['--json'] if json_output else []))
    assert result.returncode == (0 if with_simulator else 1), result.stdout
    if json_output:
        rows = {row['name']: row for row in json.loads(result.stdout)}
        assert rows['PDK_ROOT']['status'] == 'ok'
        assert rows['PDK_ROOT']['detail'] == f'{root} (built-in default)'
        assert rows['pdk sky130']['status'] == 'ok'
    else:
        assert '[FAIL] PDK_ROOT' not in result.stdout
        assert f'{root} (built-in default)' in result.stdout
        assert '[ ok ] pdk sky130' in result.stdout
