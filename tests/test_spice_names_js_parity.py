"""TASK-527: Python and JS agree on codec outputs and effective labels."""
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_view_js_differential import _require_node

ROOT = Path(__file__).resolve().parents[1]


def test_codec_and_labels_parity(tmp_path):
    node = _require_node()
    cases = tmp_path / 'cases.json'
    dump = subprocess.run([sys.executable, str(ROOT / 'tests/js/dump_spice_names_cases.py'),
                           '--output', str(cases)], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert dump.returncode == 0, dump.stdout + dump.stderr
    res = subprocess.run([node, str(ROOT / 'tests/js/spice_names_parity.mjs'), '--cases', str(cases)],
                         cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr


def test_js_label_identity_behaviour():
    node = _require_node()
    res = subprocess.run([node, '--test', str(ROOT / 'tests/js/spice_names.test.mjs')],
                         cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr
