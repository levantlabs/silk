"""silk.db.circuit_db carries no __main__ demo (TASK-585)."""
import os
import subprocess
import sys
import tempfile

import silk.db.circuit_db as circuit_db


def test_module_has_no_main_block():
    with open(circuit_db.__file__, encoding='utf-8') as f:
        assert '__name__ == "__main__"' not in f.read()


def test_running_module_exits_cleanly_and_writes_nothing():
    root = os.path.dirname(os.path.dirname(circuit_db.__file__))
    env = dict(os.environ, PYTHONPATH=os.path.dirname(root))
    with tempfile.TemporaryDirectory() as cwd:
        r = subprocess.run([sys.executable, '-m', 'silk.db.circuit_db'],
                           cwd=cwd, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert os.listdir(cwd) == []
