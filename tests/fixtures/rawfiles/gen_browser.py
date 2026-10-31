"""Generate browser-reader fixtures without a simulator.

Run from the repo root: .venv/bin/python tests/fixtures/rawfiles/gen_browser.py
The checked-in fixture pins byte offsets with C2 B5 and a lone E9 in Title:.
Additional adversarial cases are generated into the differential dump directory
so corpus tests with older float32-only assumptions need not consume them.
"""
from __future__ import annotations

import struct
from pathlib import Path


def raw_bytes(
    vectors: list[tuple[str, str]], rows: list[list], *, complex_: bool = False,
    binary: bool = True, title: bytes = b"browser differential fixture",
    plotname: str = "Transient Analysis", extra: bytes = b"",
    declared_points: int | None = None,
) -> bytes:
    """Build a deterministic point-major rawfile for reader boundary checks."""
    count = len(rows) if declared_points is None else declared_points
    header = (
        b"Title: " + title + b"\nDate: fixed fixture date\nPlotname: "
        + plotname.encode("latin-1") + b"\nFlags: "
        + (b"complex" if complex_ else b"real")
        + f"\nNo. Variables: {len(vectors)}\nNo. Points: {count}\n".encode()
        + extra + b"Variables:\n"
    )
    for i, (name, type_) in enumerate(vectors):
        header += f"\t{i}\t{name}\t{type_}\n".encode("latin-1")
    if binary:
        flat = [part for row in rows for value in row
                for part in (value if complex_ else [value])]
        return header + b"Binary:\n" + struct.pack(f"<{len(flat)}d", *flat)
    lines = []
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            token = f"{value[0]!r}, {value[1]!r}" if complex_ else repr(value)
            lines.append((f"{i}\t" if j == 0 else "\t") + token)
    return header + b"Values:\n" + ("\n".join(lines) + "\n").encode()


def nonascii_fixture() -> bytes:
    """Title bytes remain a one-to-one latin-1 map before unaligned binary data."""
    return raw_bytes(
        [("time", "time"), ("v(out)", "voltage")],
        [[-0.0, -0.0], [1.0, 0.25], [2.0, -0.5]],
        title=b"UTF-8 micro \xc2\xb5; lone high byte \xe9; latin-1 \x80\x9f",
    )


def adversarial_fixtures() -> dict[str, bytes]:
    """Return small, generated cases exercising planner and numeric boundaries."""
    nan, inf = float("nan"), float("inf")
    types = ["voltage", "current", "power", "charge", "capacitance", "notype",
             "", "time", "frequency", "temp-sweep", "res-sweep", "impedance",
             "admittance", "unknown", "VOLTAGE", "__proto__"]
    names = ["v(a)", "v(a)#2", "v(a)", "v(a)", "v(a)#3", "__proto__",
             "__proto__", "constructor", "name with spaces"]
    vectors = [("time", "time")] + [
        (names[i] if i < len(names) else f"v({i})", type_)
        for i, type_ in enumerate(types)
    ]
    samples = [-0.0, 0.0, nan, nan, inf, nan, -inf, nan, 3.0, 3.0,
               1e-38, -1e39, 3e38, 1e-37, 1.0, -2.0, 0.0]
    rows = [[float(i)] + [samples[(i + j) % len(samples)]
                         if j % 3 == 0 else (-0.0 if i % 2 else float(i % 4))
                         for j in range(len(types))] for i in range(len(samples))]
    result = {}
    for binary in (False, True):
        result[f"numeric_{'bin' if binary else 'ascii'}.raw"] = raw_bytes(
            vectors, rows, binary=binary)
    z = [(1.0, -0.0), (-1.0, 0.1), (-1.0, -0.1), (nan, 1.0),
         (-1.0, 0.1), (0.0, -1.0), (inf, inf), (1.0, 0.0)]
    complex_rows = [[(float(i + 1), 1e-310), value, value]
                    for i, value in enumerate(z)]
    for binary in (False, True):
        result[f"complex_{'bin' if binary else 'ascii'}.raw"] = raw_bytes(
            [("FREQUENCY", "frequency grid=3"), ("v(a)", "voltage"), ("v(a)", "voltage")],
            complex_rows, complex_=True, binary=binary, plotname="AC Analysis")
    result["scale_tiny.raw"] = raw_bytes(
        [("frequency", "frequency"), ("v(out)", "voltage")],
        [[(0.0, 1e-310), (1.0, 0.0)]], complex_=True)
    result["scale_complex.raw"] = raw_bytes(
        [("frequency", "frequency"), ("v(out)", "voltage")],
        [[(1.0, 1e-4), (1.0, 0.0)]], complex_=True, extra=b"Offset: 2\n")
    for binary in (False, True):
        result[f"truncated_{binary}.raw"] = raw_bytes(
            [("time", "time"), ("v(out)", "voltage")],
            [[0.0, -0.0], [1.0, 1.0]], binary=binary, declared_points=8)
    result["no_vectors.raw"] = raw_bytes([], [], plotname="Constants")
    result["axis_only.raw"] = raw_bytes([("sweep", "voltage")], [[1.0], [2.0]])
    result["empty_complex.raw"] = raw_bytes(
        [("frequency", "frequency"), ("v(out)", "voltage")], [], complex_=True)
    result["default_plot.raw"] = nonascii_fixture() + result["axis_only.raw"]
    result["nonpositive_ac.raw"] = raw_bytes(
        [("frequency", "frequency"), ("v(out)", "voltage")], [[0.0, 1.0]])
    return result


if __name__ == "__main__":
    path = Path(__file__).with_name("f_nonascii_title.raw")
    path.write_bytes(nonascii_fixture())
    print(f"{path.name}: {path.stat().st_size} bytes")
