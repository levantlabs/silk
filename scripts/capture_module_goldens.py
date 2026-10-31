#!/usr/bin/env python
"""Capture pre-split goldens for the Module file-decomposition safety net.

Run this **on the pre-split commit** (the current god-file HEAD). It builds the
battery from ``tests/refactor/battery.py`` and writes deterministic fixtures
into ``tests/refactor/goldens/``:

- ``<name>.netlist``     — SPICE netlist, byte-for-byte (netlist circuits only)
- ``<name>.schema.json`` — ``export_schema`` output, canonical JSON
- ``<name>.hash``        — structure hash
- ``<name>.faults``      — sorted device-path fault enumeration
- ``module_api.json``    — frozen ``dir(Module)`` names + signatures

``tests/refactor/test_module_golden.py`` replays these. The capture must be
committed so the test has something to diff against on the reorganized branch.

Usage::

    .venv/bin/python scripts/capture_module_goldens.py
"""
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tests.refactor import battery as B  # noqa: E402

GOLDEN_DIR = os.path.join(_REPO, "tests", "refactor", "goldens")


def main():
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    tb_dir = tempfile.mkdtemp(prefix="silk_golden_")
    circuits = B.build_battery(tb_dir)

    written = []
    for name, mod in circuits.items():
        # schema + hash for every circuit
        _write(f"{name}.schema.json", B.schema_of(mod))
        _write(f"{name}.hash", B.hash_of(mod) + "\n")
        written += [f"{name}.schema.json", f"{name}.hash"]

        if name in B.NETLIST_CIRCUITS:
            _write(f"{name}.netlist", B.netlist_of(name, mod, tb_dir))
            written.append(f"{name}.netlist")
        if name in B.FAULT_CIRCUITS:
            _write(f"{name}.faults", B.faults_of(mod) + "\n")
            written.append(f"{name}.faults")

    _write("module_api.json",
           json.dumps(B.module_api_snapshot(), indent=2, sort_keys=True) + "\n")
    written.append("module_api.json")

    print(f"Captured {len(written)} golden files into {GOLDEN_DIR}:")
    for f in written:
        print(f"  {f}")


def _write(fname, content):
    with open(os.path.join(GOLDEN_DIR, fname), "w") as fh:
        fh.write(content)


if __name__ == "__main__":
    main()
