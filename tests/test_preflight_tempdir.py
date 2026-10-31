"""TASK-453: the ngspice preflight leaves nothing behind on success and keeps
the log its message names on failure. A stub script stands in for ngspice, so
no simulator or PDK is needed."""
import os
import stat
import tempfile

import pytest

from silk.circuit.spice_netlister import PySpiceNetlister

_STUB = """#!/bin/sh
# usage: stub -b -o <log> <deck>
log="$3"
{body}
"""


def _stub(tmp_path, body):
    path = tmp_path / 'fake_ngspice'
    path.write_text(_STUB.format(body=body))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def _run(tmp_path, monkeypatch, body):
    scratch = tmp_path / 'tmpdir'
    scratch.mkdir()
    monkeypatch.setattr(tempfile, 'tempdir', str(scratch))
    nl = PySpiceNetlister.__new__(PySpiceNetlister)
    nl.generate_pdk_include = lambda corner: ''
    nl._osdi_lines = lambda: ''
    result = nl._ngspice_param_preflight_uncached(
        _stub(tmp_path, body), 'tt', 'm', 'gm', 'n', 'm')
    return result, scratch


def test_success_leaves_no_directory(tmp_path, monkeypatch):
    result, scratch = _run(tmp_path, monkeypatch,
                           'echo "@m.xm1.m[gm] = 1.0e-3" > "$log"')
    assert result.available is True
    assert result.reason == ''
    assert os.listdir(scratch) == []


def test_unavailable_param_keeps_the_named_log(tmp_path, monkeypatch):
    result, scratch = _run(tmp_path, monkeypatch, 'echo "no such" > "$log"')
    assert result.available is False
    assert result.log_path in result.reason
    assert os.path.isfile(result.log_path)
    assert os.path.dirname(result.log_path) in [str(p) for p in scratch.iterdir()]


def test_nonzero_exit_keeps_the_named_log(tmp_path, monkeypatch):
    result, _ = _run(tmp_path, monkeypatch, 'echo "boom" > "$log"; exit 3')
    assert result.available is False
    assert 'exited 3' in result.reason
    assert os.path.isfile(result.log_path)
