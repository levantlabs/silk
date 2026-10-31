#!/usr/bin/env python3
"""TASK-330.8 AC#1 spike: confirm the main analysis's shared vectors are
byte-identical (same numeric values) between a deck with extra per-probe
'@' device-param saves and the same deck without them. Read-only use of
silk's own rawfile reader; no production code touched."""
import sys
import silk
print("silk.__file__ =", silk.__file__)
from silk.signals.rawfile import read_rawfile

baseline_path = sys.argv[1]
withp_path = sys.argv[2]

base = read_rawfile(baseline_path)
withp = read_rawfile(withp_path)

base_plot = base.current_plot
withp_plot = withp.current_plot

base_names = {v.name for v in base_plot.vectors}
withp_names = {v.name for v in withp_plot.vectors}

extra = withp_names - base_names
missing = base_names - withp_names
shared = base_names & withp_names

print(f"baseline vectors: {len(base_names)}")
print(f"with_params vectors: {len(withp_names)}")
print(f"shared: {len(shared)}  extra-in-with_params: {len(extra)}  missing-from-with_params: {len(missing)}")

mismatches = []
for name in sorted(shared):
    bv = base_plot.get_vector(name) if hasattr(base_plot, 'get_vector') else None
    # fall back to direct lookup
    bvec = next(v for v in base_plot.vectors if v.name == name)
    wvec = next(v for v in withp_plot.vectors if v.name == name)
    if len(bvec.data) != len(wvec.data):
        mismatches.append((name, 'length', len(bvec.data), len(wvec.data)))
        continue
    for i, (a, b) in enumerate(zip(bvec.data, wvec.data)):
        if a != b:
            mismatches.append((name, i, a, b))
            break

print(f"mismatches among shared vectors: {len(mismatches)}")
for m in mismatches[:20]:
    print("  MISMATCH:", m)

if missing:
    print("MISSING from with_params (unexpected):", sorted(missing)[:10])

print("RESULT:", "BYTE-IDENTICAL" if not mismatches and not missing else "DIFFERS")
