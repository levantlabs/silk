"""The Publish to PyPI workflow publishes through PyPI Trusted Publishing.

The repository holds no upload credential: each publish job exchanges a
short-lived GitHub OIDC token for an upload token, and only those two jobs may
request one. Both publishers upload the files the build job made once, after
the tag check, the full tests and the strict metadata check; PyPI receives
final versions only, after TestPyPI.
"""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / '.github' / 'workflows' / 'publish.yml'
TEXT = WORKFLOW.read_text()
JOBS = yaml.safe_load(TEXT)['jobs']
PUBLISHERS = {'publish-testpypi': 'testpypi', 'publish-pypi': 'pypi'}


def _needs(name):
    needs = JOBS[name].get('needs', [])
    return [needs] if isinstance(needs, str) else needs


def _upstream(name):
    seen = set()
    todo = list(_needs(name))
    while todo:
        job = todo.pop()
        if job not in seen:
            seen.add(job)
            todo.extend(_needs(job))
    return seen


def test_no_long_lived_upload_credential():
    for marker in ('secrets.', 'TWINE_PASSWORD', 'PYPI_TOKEN', 'password:'):
        assert marker not in TEXT, f'publish.yml mentions {marker!r}'


def test_workflow_token_is_read_only():
    assert yaml.safe_load(TEXT)['permissions'] == {'contents': 'read'}


def test_only_the_publishers_request_an_oidc_token():
    oidc = {name for name, job in JOBS.items()
            if (job.get('permissions') or {}).get('id-token') == 'write'}
    assert oidc == set(PUBLISHERS)
    for name, environment in PUBLISHERS.items():
        assert JOBS[name]['permissions'] == {'id-token': 'write'}
        assert JOBS[name]['environment']['name'] == environment


def test_publishers_run_after_the_tag_check_tests_and_build():
    for name in PUBLISHERS:
        assert {'verify-tag', 'tests', 'build'} <= _upstream(name), name
    assert 'publish-testpypi' in _upstream('publish-pypi')
    assert JOBS['tests']['with'] == {'run_slow': True}


def test_build_checks_and_keeps_the_distributions():
    steps = JOBS['build']['steps']
    runs = [step.get('run', '') for step in steps]
    check = runs.index('twine check --strict dist/*')
    upload = next(i for i, step in enumerate(steps)
                  if step.get('uses') == 'actions/upload-artifact@v4')
    assert check < upload
    assert steps[upload]['with'] == {
        'name': 'dist', 'path': 'dist/', 'if-no-files-found': 'error'}


@pytest.mark.parametrize('name', PUBLISHERS)
def test_publishers_upload_the_built_files_without_rebuilding(name):
    steps = JOBS[name]['steps']
    assert [step.get('uses') for step in steps] == [
        'actions/download-artifact@v4', 'pypa/gh-action-pypi-publish@release/v1']
    assert steps[0]['with'] == {'name': 'dist', 'path': 'dist/'}
    assert not any('run' in step for step in steps)


def test_testpypi_first_and_pypi_for_final_versions_only():
    testpypi = JOBS['publish-testpypi']
    assert 'if' not in testpypi
    assert testpypi['steps'][-1]['with'] == {
        'repository-url': 'https://test.pypi.org/legacy/'}
    pypi = JOBS['publish-pypi']
    assert pypi['if'] == "needs.verify-tag.outputs.prerelease == 'false'"
    assert 'verify-tag' in _needs('publish-pypi')
    assert JOBS['verify-tag']['outputs'] == {
        'prerelease': '${{ steps.version.outputs.prerelease }}'}
    assert 'with' not in pypi['steps'][-1]


def _tag_check(tmp_path, tag, version):
    """Run the workflow's own tag check against a pyproject with *version*."""
    step = next(step for step in JOBS['verify-tag']['steps']
                if step.get('id') == 'version')
    script = step['run'].split("<<'EOF'\n", 1)[1].rsplit('EOF', 1)[0]
    (tmp_path / 'pyproject.toml').write_text(
        f'[project]\nname = "example"\nversion = "{version}"\n')
    output = tmp_path / 'github_output'
    env = {**os.environ, 'GITHUB_REF_NAME': tag, 'GITHUB_OUTPUT': str(output)}
    proc = subprocess.run([sys.executable, '-c', script], cwd=tmp_path, env=env,
                          capture_output=True, text=True)
    return proc, output.read_text() if output.exists() else ''


needs_tomllib = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason='the check runs on the runner\'s Python 3, which has tomllib (3.11+)')


@needs_tomllib
@pytest.mark.parametrize('version, prerelease', [
    ('0.2.1', 'false'), ('1.0', 'false'), ('0.2.1.post1', 'false'),
    ('0.3.0rc1', 'true'), ('0.3.0a1', 'true'), ('0.3.0b2', 'true'),
    ('0.3.0.dev1', 'true'), ('0.3.0-rc1', 'true'),
])
def test_tag_check_marks_everything_but_a_final_version_as_prerelease(
        tmp_path, version, prerelease):
    proc, output = _tag_check(tmp_path, f'v{version}', version)
    assert proc.returncode == 0, proc.stderr
    assert output == f'prerelease={prerelease}\n'


@needs_tomllib
def test_tag_check_refuses_a_tag_that_differs_from_the_version(tmp_path):
    proc, output = _tag_check(tmp_path, 'v0.2.1', '0.2.0')
    assert proc.returncode == 1
    assert 'tag v0.2.1 != pyproject version 0.2.0' in proc.stderr
    assert output == ''
