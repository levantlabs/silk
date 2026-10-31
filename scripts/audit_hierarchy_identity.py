"""Report hierarchical identity omissions without changing runtime behavior.

Run from the repository root::

    PYTHONPATH=. .venv/bin/python scripts/audit_hierarchy_identity.py

All circuits and the illustrative tracking database live in a temporary
directory. No simulator or PDK is used. JSON reports observations, including
known defects; exit 0 means the probe completed, not that identity is correct.
"""

import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from silk import Module, Port
from silk.db.identity import STRUCTURE_FINGERPRINT_SCHEME
from silk.db.structure_tracker import CircuitStructureTracker
from silk.elements import elements
from silk.elements.passives import Resistor
from silk.exceptions import CircuitError, EvidenceMismatchError


def _module(out, name, *, b="b", db_path=None):
    module = Module(
        "none", out, f"{name}.cir", pdk_details=None,
        module_name=name, db_path=db_path,
    )
    module.define_module_ports({
        "a": Port.analog_input(), b: Port.analog_output(),
        "vss": Port.ground(),
    })
    return module


def _flat(out, *, swap=False, b="b", device=False):
    module = _module(out, "leaf", b=b)
    endpoints = [("r1", "a", "a" if swap else b, 1000),
                 ("r2", b, "vss", 2000)]
    for name, pos, neg, value in endpoints:
        if device:
            module.add(elements.resistor(name, p=pos, n=neg, value=value))
        else:
            module.add_passive(Resistor(
                name, pos=pos, neg=neg, resistance=value,
            ))
    return module


def _parent(out, *, swap=False, db_path=None):
    module = _module(out, "root", db_path=db_path)
    module.add_element(_flat(out), "u", nets={
        "a": "b" if swap else "a", "b": "a" if swap else "b",
        "vss": "vss",
    })
    return module


def _wrap(out, child, name):
    module = _module(out, name)
    module.add_element(child, "stage", nets={
        "a": "a", "b": "b", "vss": "vss",
    })
    return module


def _bindings(parent, instance):
    pins = parent.elem_dict[instance]["pin_list"]
    return {name: pin.net for name, pin in pins.items()}


def _compare(tracker, original, changed):
    structures = [tracker.extract_structure(m) for m in (original, changed)]
    return {
        "net_names_equal": set(original.net_dict) == set(changed.net_dict),
        "extracted_structures_equal": structures[0] == structures[1],
        "hashes": [tracker.compute_structure_hash(s)[0] for s in structures],
        "hierarchy": [s["hierarchy"] for s in structures],
    }


def _registration(out, first, second):
    top = Module(
        "none", out, "top.cir", pdk_details=None, module_name="top",
    )
    top.define_module_ports({
        "vin": Port.analog_input(), "out1": Port.analog_output(),
        "out2": Port.analog_output(), "vss": Port.ground(),
    })
    top.add_element(first, "original", nets={
        "a": "vin", "b": "out1", "vss": "vss",
    })
    error = None
    deck = ''
    try:
        top.add_element(second, "changed", nets={
            "a": "vin", "b": "out2", "vss": "vss",
        })
        deck = top.generate_netlist()
    except CircuitError as exc:
        error = str(exc)
    return {
        "circuit_error": error,
        "definition_count": sum(
            line.split()[:2] == [".subckt", first.module_name]
            for line in deck.splitlines()
        ),
        "instance_lines": [line for line in deck.splitlines()
                           if line.startswith("x")],
    }


def main() -> None:
    """Print live observations for the audit and its positive controls."""
    tracker = CircuitStructureTracker(None, None)
    with TemporaryDirectory(prefix="silk-audit-hierarchy-") as out:
        original, rewired = _parent(out), _parent(out, swap=True)
        report = {
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True,
                cwd=Path(__file__).resolve().parents[1],
            ).strip(),
            "structure_scheme": STRUCTURE_FINGERPRINT_SCHEME,
            "parent_rewire": _compare(tracker, original, rewired),
            "identical_parent_rebuild": _compare(
                tracker, original, _parent(out),
            ),
            "unchanged_leaf_definition": _compare(
                tracker, original.module_dict["leaf"]["device"],
                rewired.module_dict["leaf"]["device"],
            ),
        }
        report["parent_rewire"]["actual_bindings"] = [
            _bindings(m, "u") for m in (original, rewired)
        ]
        report["parent_rewire"]["separate_child_lines"] = [
            [line for line in m.generate_netlist(subckt=True).splitlines()
             if line.startswith("xu ")]
            for m in (original, rewired)
        ]
        try:
            dict(original.elem_dict["u"]["device"].nets)
        except (TypeError, ValueError) as exc:
            report["nets_conversion_error"] = f"{type(exc).__name__}: {exc}"
        else:
            report["nets_conversion_error"] = None

        for device in (False, True):
            kind = "device" if device else "passive"
            flat = _flat(out, device=device)
            report[f"flat_{kind}_terminal_change"] = _compare(
                tracker, flat, _flat(out, swap=True, device=device),
            )
            report[f"flat_{kind}_net_rename"] = _compare(
                tracker, flat, _flat(out, b="renamed", device=device),
            )

        nested_original, nested_rewired = original, rewired
        for depth in (2, 3):
            nested_original = _wrap(out, nested_original, f"level{depth}")
            nested_rewired = _wrap(out, nested_rewired, f"level{depth}")
            observation = _compare(tracker, nested_original, nested_rewired)
            observation["outer_bindings"] = [
                _bindings(m, "stage")
                for m in (nested_original, nested_rewired)
            ]
            observation["module_definition_payloads_equal"] = (
                tracker._extract_module_structure(nested_original)
                == tracker._extract_module_structure(nested_rewired)
            )
            report[f"nested_depth_{depth}"] = observation

        db_path = str(Path(out) / "audit.db")
        db_original = _parent(out, db_path=db_path)
        try:
            original_id = db_original.create_circuit_version()
            # This sentinel is explicitly stored, never simulated.
            db_original.store_result("gain", 0.666, "V/V")
        finally:
            db_original.close()
        db_rewired = _parent(out, swap=True, db_path=db_path)
        try:
            rewired_id = db_rewired.create_circuit_version()
            try:
                values = [r.value for r in db_rewired.query_results(parameter="gain")]
                evidence_error = None
            except EvidenceMismatchError as exc:
                values = []
                evidence_error = str(exc)
            report["tracking"] = {
                "same_version": original_id == rewired_id,
                "rewired_reads_original_sentinel": values,
                "evidence_mismatch_error": evidence_error,
            }
        finally:
            db_rewired.close()
        db_rebuild = _parent(out, db_path=db_path)
        try:
            report["tracking"]["identical_rebuild_reuses_version"] = (
                db_rebuild.create_circuit_version() == original_id
            )
        finally:
            db_rebuild.close()

        report["rewired_parent_registration"] = _registration(
            out, _parent(out), _parent(out, swap=True),
        )
        report["identical_parent_registration"] = _registration(
            out, _parent(out), _parent(out),
        )
        leaf = _flat(out)
        report["shared_leaf_registration"] = _registration(out, leaf, leaf)
        report["identical_leaf_registration"] = _registration(
            out, _flat(out), _flat(out),
        )
        report["flat_changed_body_registration"] = _registration(
            out, _flat(out), _flat(out, swap=True),
        )
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
