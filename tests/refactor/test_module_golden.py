"""Safety-net golden replay for the Module file-decomposition refactor.

Replays the fixtures captured by ``scripts/capture_module_goldens.py`` against
a freshly-built battery. Guards the reorganization against any drift:

- **byte-identical** SPICE netlists, ``export_schema``, structure hash, and
  device-path fault enumeration;
- **frozen** ``dir(Module)`` name list + per-callable signatures;
- **round-trip equality** (observable, not byte-compare) through both
  ``pickle`` and ``copy.deepcopy`` — proving the pickle/deepcopy machinery
  still round-trips every circuit.

This file **must pass on the pre-split HEAD** (it is committed alongside the
goldens), then continue passing after every part is extracted.
"""
import copy
import json
import os
import pickle

import pytest

import silk.db.identity as db_identity
from silk.db.models import Fingerprint
from silk.db.structure_tracker import CircuitStructureTracker
from tests.refactor import battery as B

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")


def _golden(fname):
    with open(os.path.join(GOLDEN_DIR, fname)) as fh:
        return fh.read()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Build the battery once; ``tb_dir`` receives the top-level tb netlist."""
    tb_dir = str(tmp_path_factory.mktemp("golden_tb"))
    return tb_dir, B.build_battery(tb_dir)


# ---------------------------------------------------------------------------
# Per-circuit byte-identical observables
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", B.BATTERY_NAMES)
def test_schema_byte_identical(built, name):
    _tb_dir, circuits = built
    assert B.schema_of(circuits[name]) == _golden(f"{name}.schema.json")


@pytest.mark.parametrize("name", B.BATTERY_NAMES)
def test_structure_hash_byte_identical(built, name):
    _tb_dir, circuits = built
    expected = _golden(f"{name}.hash").strip()
    fingerprint_structure = getattr(db_identity, "_fingerprint_structure", None)
    assert callable(fingerprint_structure), "missing scheme-tagged structure helper"
    tracker = CircuitStructureTracker(None, None)
    structure = tracker.extract_structure(circuits[name])
    fingerprint, _module_fingerprints = fingerprint_structure(structure)

    assert fingerprint == Fingerprint("silk.structure/v4", expected)
    assert B.hash_of(circuits[name]) == expected


@pytest.mark.parametrize("name", B.NETLIST_CIRCUITS)
def test_netlist_byte_identical(built, name):
    tb_dir, circuits = built
    assert B.netlist_of(name, circuits[name], tb_dir) == _golden(f"{name}.netlist")


@pytest.mark.parametrize("name", B.FAULT_CIRCUITS)
def test_fault_paths_byte_identical(built, name):
    _tb_dir, circuits = built
    assert B.faults_of(circuits[name]) + "\n" == _golden(f"{name}.faults")


# ---------------------------------------------------------------------------
# Frozen Module API surface (backstops any method dropped/renamed by a part)
# ---------------------------------------------------------------------------
def test_module_api_names_frozen():
    golden = json.loads(_golden("module_api.json"))
    current = B.module_api_snapshot()
    missing = sorted(set(golden["names"]) - set(current["names"]))
    added = sorted(set(current["names"]) - set(golden["names"]))
    assert not missing, f"Module lost attributes: {missing}"
    assert not added, f"Module gained unexpected attributes: {added}"


def test_module_api_signatures_frozen():
    golden = json.loads(_golden("module_api.json"))
    current = B.module_api_snapshot()
    changed = {
        n: (golden["signatures"][n], current["signatures"].get(n))
        for n in golden["signatures"]
        if n in current["signatures"]
        and golden["signatures"][n] != current["signatures"][n]
    }
    assert not changed, f"Signatures changed: {changed}"


# ---------------------------------------------------------------------------
# Round-trip equality — pickle and deepcopy preserve every observable
# ---------------------------------------------------------------------------
def _observables(name, mod, tb_dir):
    obs = {"schema": B.schema_of(mod), "hash": B.hash_of(mod)}
    if name in B.NETLIST_CIRCUITS:
        obs["netlist"] = B.netlist_of(name, mod, tb_dir)
    if name in B.FAULT_CIRCUITS:
        obs["faults"] = B.faults_of(mod)
    return obs


@pytest.mark.parametrize("name", B.BATTERY_NAMES)
def test_pickle_round_trip_equal(built, name):
    tb_dir, circuits = built
    mod = circuits[name]
    before = _observables(name, mod, tb_dir)
    restored = pickle.loads(pickle.dumps(mod))
    assert _observables(name, restored, tb_dir) == before


@pytest.mark.parametrize("name", B.BATTERY_NAMES)
def test_deepcopy_round_trip_equal(built, name):
    tb_dir, circuits = built
    mod = circuits[name]
    before = _observables(name, mod, tb_dir)
    restored = copy.deepcopy(mod)
    assert _observables(name, restored, tb_dir) == before
