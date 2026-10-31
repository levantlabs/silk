"""Dump Python waveform oracles for the standalone Node differential harness.

.venv/bin/python tests/js/dump_waveform_expected.py --output /tmp/silk-oracle
node tests/js/differential.mjs --module /path/to/waveform-core.mjs --expected /tmp/silk-oracle
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from silk.signals.rawfile import read_rawfile
from silk.signals.view import emit_view_data

CONFIGS = [{}, {"min_points": 0, "base": 2, "min_bucket": 1},
           {"min_points": 0, "base": 8, "min_bucket": 1}]


def _array(arr):
    return {"$f64": base64.b64encode(np.asarray(arr, dtype="<f8").tobytes()).decode("ascii")}


def _reader_oracle(raw):
    plots = []
    for plot in raw.plots:
        vectors = []
        for vector in plot.vectors:
            data = vector.data
            encoded = ({"re": _array(data.real), "im": _array(data.imag)}
                       if np.iscomplexobj(data) else _array(data))
            vectors.append({"name": vector.name, "type": vector.type,
                            "attrs": list(vector.attrs), "data": encoded})
        plots.append({"header": plot.header, "vectors": vectors, "npoints": plot.npoints,
                      "nvars": plot.nvars, "real": plot.real, "flags": list(plot.flags),
                      "padded": plot.padded, "title": plot.title, "date": plot.date,
                      "name": plot.plotname, "plotname": plot.plotname, "offset": plot.offset})
    return {"plots": plots, "file_name": raw.source_path}


def dump_expected(output: Path, fixtures: Path) -> int:
    """Write every corpus file and its full, unfiltered Python payloads."""
    output.mkdir(parents=True, exist_ok=True)
    source_files = sorted(fixtures.glob("*.raw"))
    if not source_files:
        raise ValueError(f"no .raw fixtures found in {fixtures}")
    raw_dir = output / "rawfiles"
    raw_dir.mkdir(exist_ok=True)
    for path in source_files:
        shutil.copyfile(path, raw_dir / path.name)
    generator_path = ROOT / "tests/fixtures/rawfiles/gen_browser.py"
    spec = importlib.util.spec_from_file_location("gen_browser", generator_path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    generated = generator.adversarial_fixtures()
    from tests.test_view_op_e2e import raw_plot, op_raw, ac_raw
    generated.update({
        'op.raw': op_raw().encode(), 'ac_op.raw': (ac_raw() + op_raw()).encode(),
        'scalar_dc.raw': raw_plot('DC transfer characteristic', [('sweep', 'voltage'),
            ('v(out)', 'voltage')], [[0, 30]]).encode(),
        'scalar_tran.raw': raw_plot('Transient Analysis', [('time', 'time'),
            ('v(out)', 'voltage')], [[1, 40]]).encode(),
        'scalar_ac.raw': raw_plot('AC Analysis', [('frequency', 'frequency'),
            ('v(out)', 'voltage')], [[10, 2]], complex=True).encode(),
    })
    for name, data in generated.items():
        (raw_dir / ("generated_" + name)).write_bytes(data)
    names = [path.name for path in source_files] + ["generated_" + name for name in generated]
    entries = []
    for name in names:
        raw = read_rawfile(raw_dir / name)
        raw.source_path = name  # Browser identity is seeded from File.name.
        cases = []
        for config in CONFIGS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
                payload = emit_view_data(raw, **{"decimate_" + k: v for k, v in config.items()})
            cases.append({"options": config, "payload": payload})
        entry_name = name + ".json"
        (output / entry_name).write_text(
            json.dumps({"reader": _reader_oracle(raw), "cases": cases}, allow_nan=False),
            encoding="utf-8",
        )
        entries.append({"file_name": name, "raw": "rawfiles/" + name, "expected": entry_name})
    (output / "manifest.json").write_text(
        json.dumps({"format": 1, "corpus_count": len(source_files), "entries": entries}),
        encoding="utf-8",
    )
    return len(entries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=ROOT / "tests/fixtures/rawfiles")
    args = parser.parse_args()
    count = dump_expected(args.output, args.fixtures)
    print(f"Dumped {count} rawfiles with {len(CONFIGS)} parameter sets each to {args.output}")
