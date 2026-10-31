"""Generator for the ``a_*`` rawfile fixtures (slice A: ngspice binary rawfiles).

Every fixture in this file is REAL: produced by actually invoking ``ngspice``
(ngspice-42, ``/usr/bin/ngspice``) on the netlists embedded below, not
hand-crafted bytes. Re-run this script to regenerate the fixtures bit-for-bit
(ngspice's output is deterministic for a fixed netlist/version):

    python tests/fixtures/rawfiles/gen_a.py

Fixtures produced (all written into this directory):

* ``a_tran_bin.raw``   — transient analysis, binary, real vectors (``Flags: real``).
* ``a_tran_ascii.raw`` — the SAME transient run, written ASCII instead of binary
  (via ngspice's ``set filetype=ascii``), for the binary/ASCII round-trip test.
* ``a_ac_bin.raw``     — AC analysis, binary, complex vectors (``Flags: complex``).
* ``a_ac_ascii.raw``   — the SAME AC run, written ASCII, for the round-trip test.

Both circuits are simple RC networks (no PDK required). Circuit values and
sweep points were chosen so ngspice's adaptive transient control (and a
linear AC sweep) produce a SMALL, deterministic number of points:

* The transient run sets an explicit Tmax (``.tran 1u 5u 0 1u``) to stop
  ngspice from silently using its default internal step ceiling of
  Tstop/50 (confirmed empirically: without an explicit Tmax, even a DC-only
  circuit over 0..5u produced ~58 internal points). With Tmax pinned to the
  print step, and a slow (1u) rise time on the sole PULSE edge (no falling
  edge inside the simulated window), ngspice still ramps up its internal
  step geometrically from t=0 (doubling until it reaches Tmax), giving a
  fixed set of 22 rows for this exact netlist.
* The AC run uses a *linear* sweep (``.ac lin 5 1k 5k``) so the frequency
  points are round numbers (1k, 2k, 3k, 4k, 5k Hz) instead of the irrational
  points a ``dec``/``oct`` sweep would give.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).parent

NGSPICE = "ngspice"

TRAN_NETLIST = """\
* RC charging transient - real vectors, binary rawfile
* single slow (1u) rising edge, no falling edge inside the sim window
V1 in 0 PULSE(0 1 0 1u 1u 8u 20u)
R1 in out 1k
C1 out 0 1n
.tran 1u 5u 0 1u
.control
run
write a_tran_bin.raw
set filetype=ascii
write a_tran_ascii.raw
quit
.endc
.end
"""

AC_NETLIST = """\
* RC low-pass AC sweep - complex vectors, binary rawfile
V1 in 0 AC 1
R1 in out 1k
C1 out 0 1n
.ac lin 5 1k 5k
.control
run
write a_ac_bin.raw
set filetype=ascii
write a_ac_ascii.raw
quit
.endc
.end
"""


def _run(netlist: str) -> None:
    """Write ``netlist`` to a temp .cir and run ngspice from this directory.

    ngspice's ``write`` command (inside the netlist's own ``.control`` block)
    writes its output relative to the current working directory, so we run
    with cwd=HERE and let the netlist name the fixture files directly.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        f.write(netlist)
        cir_path = Path(f.name)
    try:
        result = subprocess.run(
            [NGSPICE, "-b", str(cir_path)],
            cwd=HERE,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ngspice failed (rc={result.returncode}):\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
    finally:
        cir_path.unlink(missing_ok=True)


def main() -> None:
    _run(TRAN_NETLIST)
    _run(AC_NETLIST)
    for name in (
        "a_tran_bin.raw",
        "a_tran_ascii.raw",
        "a_ac_bin.raw",
        "a_ac_ascii.raw",
    ):
        path = HERE / name
        assert path.exists(), f"expected {path} to be produced by ngspice"
        print(f"{name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
