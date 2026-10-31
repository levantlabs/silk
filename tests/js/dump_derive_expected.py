"""Dump silk Signal oracles for real and complex explorer derive operations.

.venv/bin/python tests/js/dump_derive_expected.py --output /tmp/silk-derive
node --test tests/js/explorer_derive.test.mjs   # with SILK_DERIVE_EXPECTED=/tmp/silk-derive

Each case names a payload, a signal and an operation. The expected samples are
silk's own Python computation (Signal.mag, Signal.derivative, Signal.integral
and gain * a + offset) on exactly the level-0 values the explorer decodes from
that payload, so the Node side compares like with like.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from silk.signals.rawfile import read_rawfile
from silk.signals.base import Signal
from silk.signals import export_signal_operations
from silk.signals.view import emit_view_data

# The shared operation vocabulary: TASK-373's canonical export, reshaped for
# the JS test (inputs is the exact count, or "N+" when there is no upper
# limit; params maps each parameter to its default). The explorer is tested
# against this list, so the viewer and the testbench spec cannot drift apart.
def _inputs(op):
    if op["max_inputs"] == op["min_inputs"]:
        return op["min_inputs"]
    if op["max_inputs"] is None:
        return f"{op['min_inputs']}+"
    raise ValueError(f"unexpected input range for {op['name']!r}: {op}")


OPERATIONS = [
    {"name": op["name"], "inputs": _inputs(op),
     "params": {p["name"]: p["default"] for p in op["parameters"]}}
    for op in export_signal_operations()
]


SINGLE = [
    ("abs", None),
    ("scale", {"gain": -1.0}),
    ("scale", {"gain": 1e6, "offset": 0.25}),
    ("scale", {"offset": -0.9}),
    ("derivative", None),
    ("integral", None),
]


def _b64(arr):
    return base64.b64encode(np.asarray(arr, dtype="<f8").tobytes()).decode("ascii")


def _decode(text, dt):
    dtype = "<f8" if dt == "f64" else "<f4"
    return np.frombuffer(base64.b64decode(text), dtype=dtype).astype(np.float64)


def _synthetic(name, x, signals):
    """A one-plot payload with float64 level-0 arrays, as the encoder emits."""
    n = len(x)
    return {
        "format": 1, "endian": "le", "run": {"id": name, "title": name},
        "default_plot": 0, "annotations": {"markers": [], "bands": [], "notes": []},
        "plots": [{
            "name": "Transient", "kind": "tran", "npoints": n, "complex": False,
            "x_log_default": False,
            "x": {"name": "time", "unit": "s", "levels": [{"stride": 1, "n": n, "dt": "f64", "x": _b64(x)}]},
            "signals": [{
                "name": sig_name, "unit": unit, "type": "voltage" if unit == "V" else "current",
                "part": None, "group": None,
                "nonfinite": int(np.count_nonzero(~np.isfinite(y))),
                "levels": [{"stride": 1, "n": n, "dt": "f64", "y": _b64(y)}],
            } for sig_name, unit, y in signals],
        }],
    }


def payloads():
    yield "e_ng_tran", emit_view_data(ROOT / "tests/fixtures/rawfiles/e_ng_tran.raw")
    x = np.arange(41) * 0.25e-9  # exactly uniform: numpy.gradient's scalar branch
    yield "uniform", _synthetic("uniform", x, [
        ("v(sin)", "V", np.sin(2 * np.pi * x / 4e-9)),
        ("i(ramp)", "A", 1e-3 * np.arange(41) - 0.02),
    ])
    rng = np.random.default_rng(374)
    x = np.cumsum(rng.uniform(1e-12, 5e-10, 64))
    y = np.cos(x * 3e9) * 1.7
    y[17] = np.nan
    yield "nonuniform_nan", _synthetic("nonuniform_nan", x, [("v(n)", "V", y)])
    yield "two_samples", _synthetic("two_samples", np.array([1e-9, 4e-9]), [("v(two)", "V", np.array([0.5, -1.25]))])
    yield "one_sample", _synthetic("one_sample", np.array([1e-9]), [("v(one)", "V", np.array([0.5]))])


def reference(sig, operation, params):
    if operation == "abs":
        return sig.mag()
    if operation == "derivative":
        return sig.derivative()
    if operation == "integral":
        return sig.integral()
    merged = {"gain": 1.0, "offset": 0.0, **(params or {})}
    return merged["gain"] * sig + merged["offset"]


def build():
    out = {"operations": OPERATIONS, "payloads": {}, "cases": [], "refusals": []}
    for key, payload in payloads():
        out["payloads"][key] = payload
        plot_idx = payload["default_plot"]
        plot = payload["plots"][plot_idx]
        x = _decode(plot["x"]["levels"][0]["x"], "f64")
        signals = plot["signals"][:3] if key == "e_ng_tran" else plot["signals"]
        for meta in signals:
            level = meta["levels"][0]
            y = _decode(level["y"], level["dt"])
            sig = Signal(meta["name"], meta["type"], "time", meta["unit"], x, y)
            for operation, params in SINGLE:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = reference(sig, operation, params)
                except ValueError as error:
                    out["refusals"].append({"payload": key, "plot": plot_idx, "signal": meta["name"],
                                            "operation": operation, "params": params, "python": str(error)})
                    continue
                out["cases"].append({
                    "payload": key, "plot": plot_idx, "signal": meta["name"],
                    "operation": operation, "params": params,
                    "expected": _b64(result.y),
                })
    # Actual ngspice RC AC vectors, not the already projected magnitudes.
    raw = read_rawfile(ROOT / "tests/fixtures/rawfiles/e_ng_ac.raw")
    out["payloads"]["ac"] = emit_view_data(raw)
    plot = raw.plots[0]
    x = plot.vectors[0].data.real
    signals = {v.name: Signal(v.name, v.type, "frequency", "V", x, v.data)
               for v in plot.vectors[1:]}
    # A second oracle isolates arithmetic from the existing float32 payload
    # quantization, while raw vectors still check end-to-end numerical error.
    reconstructed = {}
    for name, sig in signals.items():
        parts = {m["part"]: m["levels"][0] for m in out["payloads"]["ac"]["plots"][0]["signals"]
                 if m["group"] == name}
        mag = _decode(parts["mag"]["y"], parts["mag"]["dt"])
        phase = _decode(parts["phase"]["y"], parts["phase"]["dt"]) * (np.pi / 180)
        reconstructed[name] = Signal(name, "voltage", "frequency", "V", x,
                                      mag * np.cos(phase) + 1j * mag * np.sin(phase))
    a, b = signals["v(in)"], signals["v(out)"]
    out["complex_cases"] = []
    cases = [(op, ["v(in)", "v(out)"], None, result) for op, result in [
        ("sum", a + b), ("difference", a - b), ("product", a * b),
        ("quotient", a / b), ("mean", (a + b) / 2),
    ]]
    cases += [(op, ["v(out)"], params, reference(b, op, params)) for op, params in SINGLE]
    for operation, operands, params, result in cases:
        aa, bb = reconstructed["v(in)"], reconstructed["v(out)"]
        exact = ({"sum": lambda: aa + bb, "difference": lambda: aa - bb,
                  "product": lambda: aa * bb, "quotient": lambda: aa / bb,
                  "mean": lambda: (aa + bb) / 2}[operation]() if len(operands) == 2
                 else reference(bb, operation, params))
        out["complex_cases"].append({
            "operation": operation, "operands": operands, "params": params,
            "re": _b64(result.y.real), "im": _b64(result.y.imag),
            "mag": _b64(result.mag().y), "phase": _b64(result.phase_deg_unwrapped().y),
            "encoded_mag": _b64(exact.mag().y), "encoded_phase": _b64(exact.phase_deg_unwrapped().y),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "derive_expected.json").write_text(json.dumps(build()), encoding="utf-8")


if __name__ == "__main__":
    main()
