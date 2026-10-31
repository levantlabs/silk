"""The local gate must not turn missing coverage or failed commands green."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.release_gate import Gate, GateError, fast_chunks, main, require_candidate


def test_fast_chunks_cover_the_checkout_once():
    root = Path(__file__).resolve().parents[1]
    chunks = fast_chunks(root)
    assert len(chunks) == 4
    assert 'tests/metrics/test_ac.py' in chunks[0]
    assert 'tests/test_packaging.py' in chunks[2]
    assert 'tests/refactor' in chunks[2]
    assert 'tests/test_view_js_differential.py' in chunks[3]


def test_new_unhandled_test_path_fails_closed(tmp_path):
    tests = tmp_path / 'tests'
    tests.mkdir()
    for name in ('a', 'n', 'q', 't', '9_new'):
        (tests / f'test_{name}.py').touch()
    with pytest.raises(GateError, match='incomplete|Empty'):
        fast_chunks(tmp_path)


def test_candidate_rejects_wrong_head_and_dirty_files(tmp_path):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], text=True).strip()

    git('init', '-q')
    git('-c', 'user.name=Gate test', '-c', 'user.email=gate@example.invalid',
        'commit', '--allow-empty', '-qm', 'fixture')
    sha = git('rev-parse', 'HEAD')
    require_candidate(tmp_path, sha)
    with pytest.raises(GateError, match='HEAD'):
        require_candidate(tmp_path, '0' * 40)
    (tmp_path / 'untracked.py').touch()
    with pytest.raises(GateError, match='working-tree'):
        require_candidate(tmp_path, sha)


@pytest.mark.parametrize('body, required, expected', [
    ('assert True', True, 'PASS'),
    ('assert False', False, 'FAIL'),
    ('pytest.skip("missing dependency")', True, 'FAIL'),
    ('pytest.skip("optional capability")', False, 'PASS'),
])
def test_pytest_outcomes_and_skip_policy(tmp_path, body, required, expected):
    source = tmp_path / 'test_fixture.py'
    source.write_text('import pytest\ndef test_ok():\n    assert True\n'
                      f'def test_case():\n    {body}\n')
    out = tmp_path / 'evidence'
    out.mkdir()
    gate = Gate(tmp_path, out, 'test-candidate')
    command = [sys.executable, '-m', 'pytest', '-c', '/dev/null', str(source)]
    if expected == 'FAIL':
        with pytest.raises(GateError):
            gate.run('check', command, pytest=True, require_all=required)
    else:
        gate.run('check', command, pytest=True, require_all=required)
    record = json.loads((out / 'summary.json').read_text())['stages'][0]
    assert record['status'] == expected
    assert record['counts']['passed'] >= 1
    assert record['counts']['skipped'] == int('skip' in body)


def test_cli_prints_and_retains_failure_summary(tmp_path, monkeypatch, capsys):
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    out = tmp_path / 'evidence'
    monkeypatch.chdir(checkout)
    monkeypatch.setattr('scripts.release_gate.git', lambda *args: 'abc123')

    def fail(gate):
        gate.run('broken', [sys.executable, '-c', 'raise SystemExit(7)'])

    monkeypatch.setattr('scripts.release_gate.execute', fail)
    assert main(['--candidate', 'HEAD', '--output', str(out)]) == 1
    record = json.loads((out / 'summary.json').read_text())
    assert record['status'] == 'FAIL'
    assert record['stages'][0]['returncode'] == 7
    assert 'Local release gate: FAIL' in capsys.readouterr().out
    assert 'Candidate: abc123' in (out / 'summary.txt').read_text()
