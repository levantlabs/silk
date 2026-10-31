"""Generator for the ``b_*`` rawfile fixtures (slice B: ASCII ``Values:`` mode
depth, the ``Offset:`` keyword, and per-variable trailing attributes).

Each fixture below is labeled REAL, REAL+DERIVED, or HAND-CRAFTED:

* REAL — produced by actually invoking ``ngspice`` (ngspice-42,
  ``/usr/bin/ngspice``) on the netlists embedded below. Re-run this script to
  regenerate them bit-for-bit:

      python tests/fixtures/rawfiles/gen_b.py

* REAL+DERIVED — starts from a REAL ngspice rawfile (byte-for-byte, including
  its binary payload) with exactly one text-header line spliced in
  programmatically. Used for the ``Offset:`` keyword, which no installed
  version of ngspice actually *writes* — every ngspice run we generated here
  omits the keyword entirely, so a real fixture for it cannot be produced
  locally. ``Offset:`` is a real nutmeg-family field nonetheless (documented in
  the public nutmeg/rawfile format description as a start-time correction added
  to the scale) — so the only way to pin down what a nutmeg *reader* must do
  with it is to splice it into an otherwise-real file and observe.

* HAND-CRAFTED — small synthetic ASCII text, used where the feature being
  tested (a header keyword variant, a flag combination, an edge case) is not
  naturally produced by any ngspice invocation we could construct. Every
  hand-crafted byte layout below (whitespace, field order, attribute
  placement) was copied from what real ngspice ASCII output actually looks
  like (see ``b_tran_real.raw`` / ``b_ac_complex.raw``), not invented. The
  *semantics* asserted on hand-crafted files (Offset addition, attribute
  transparency, case-insensitive keywords, the padded/unpadded tokens) come
  from the published nutmeg format description.

Fixtures produced (all written into this directory):

* ``b_tran_real.raw``      REAL     — transient, ASCII, real vectors (Flags: real),
                                       no Offset/padded keyword present (establishes
                                       the ``padded`` default-when-absent = True).
* ``b_ac_complex.raw``     REAL     — AC analysis, ASCII, complex vectors
                                       (Flags: complex), decade sweep so the
                                       frequency scale carries a real per-variable
                                       trailing attribute (``frequency grid=3``).
* ``b_tran_bin_real.raw``  REAL     — small binary transient (baseline for the
                                       binary Offset check).
* ``b_tran_bin_offset.raw`` REAL+DERIVED — ``b_tran_bin_real.raw`` with an
                                       ``Offset:`` line spliced into the text
                                       header; binary payload untouched.
* ``b_offset_ascii.raw`` / ``b_offset_ascii_baseline.raw``  HAND-CRAFTED pair —
                                       identical except for one ``Offset:`` line;
                                       proves the exact additive semantics on the
                                       scale (vector 0) and non-effect elsewhere.
* ``b_offset_nontime_scale.raw``  HAND-CRAFTED — ``Offset:`` on a vector 0 named
                                       "frequency" (not "time"): proves the effect
                                       is purely positional (vectors[0]), not
                                       gated on the scale's type/name.
* ``b_offset_zero.raw``    HAND-CRAFTED — ``Offset: 0`` edge case.
* ``b_attrs_trailing.raw`` HAND-CRAFTED — ``Variables:`` lines with multiple
                                       trailing attributes (``dims=``, ``min=``,
                                       ``max=``, ``grid=``).
* ``b_zero_points.raw``    HAND-CRAFTED — ``No. Points: 0`` with an empty
                                       ``Values:`` block.
* ``b_lowercase_keywords.raw``  HAND-CRAFTED — every header keyword cased
                                       differently (upper/lower/mixed) than
                                       ngspice's own convention.
* ``b_flags_real_padded.raw`` / ``b_flags_real_unpadded.raw`` /
  ``b_flags_complex_padded.raw`` / ``b_flags_complex_unpadded.raw``
                              HAND-CRAFTED — the four ``padded``/``unpadded``
                                       x ``real``/``complex`` combinations.

Both REAL circuits are simple RC networks (no PDK required).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).parent

NGSPICE = "ngspice"

# ---------------------------------------------------------------------------
# REAL fixtures: actual ngspice invocations
# ---------------------------------------------------------------------------

# Real values, ASCII. No Tmax pinning here (unlike the small binary fixture
# below) - the PULSE has both a rising and falling edge inside the sim
# window, which is what "No. Points: 0"'s sibling tests want to avoid, but
# for the general "does ASCII real Values: parsing work" fixture, a few dozen
# points from ngspice's default adaptive control is fine; nothing here
# depends on the exact point count.
TRAN_NETLIST = """\
b_tran_real fixture - RC transient, ASCII real Values:
V1 1 0 DC 0 PULSE(0 1 0 1n 1n 10n 20n)
R1 1 2 1k
C1 2 0 1n
.control
tran 1n 4n
set filetype=ascii
write b_tran_real.raw
.endc
.end
"""

# Complex values, ASCII, decade AC sweep. A *decade* (or octave) sweep is
# what makes ngspice attach a "grid=3" trailing attribute to the frequency
# scale's Variables: line (confirmed empirically: the same circuit swept
# with `ac lin` instead of `ac dec` produces a bare "frequency" line, no
# attribute at all).
AC_NETLIST = """\
b_ac_complex fixture - RC lowpass AC sweep, ASCII complex Values:
V1 1 0 DC 0 AC 1
R1 1 2 1k
C1 2 0 1n
.control
ac dec 3 1k 10k
set filetype=ascii
write b_ac_complex.raw
.endc
.end
"""

# Small binary transient - baseline for the Offset-in-binary derived fixture.
# Same "pin Tmax, single slow rising edge, no falling edge in-window" trick
# slice A's gen_a.py documents: without it ngspice's adaptive transient
# control produces dozens of points; with it, a fixed, small 22.
TRAN_BIN_NETLIST = """\
b_tran_bin_real fixture - small RC transient, binary, real Values
V1 in 0 PULSE(0 1 0 1u 1u 8u 20u)
R1 in out 1k
C1 out 0 1n
.tran 1u 5u 0 1u
.control
run
write b_tran_bin_real.raw
.endc
.end
"""


def _run_ngspice(netlist: str) -> None:
    """Write ``netlist`` to a temp .cir and run ngspice with cwd=HERE.

    ngspice's ``write`` command (inside the netlist's own ``.control``
    block) writes relative to the current working directory, so the
    netlists above name the fixture files directly and we just need to run
    from this directory.
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


def _splice_offset_into_binary() -> None:
    """Derive ``b_tran_bin_offset.raw`` from the REAL ``b_tran_bin_real.raw``.

    Copies the file byte-for-byte and inserts a single
    ``Offset: 5.000000000000000e-09\\n`` text line right before the
    ``Variables:\\n`` marker in the header. The binary payload after
    ``Binary:\\n`` is untouched, so any difference in the parsed vector 0
    values is attributable only to the spliced Offset line.
    """
    src = HERE / "b_tran_bin_real.raw"
    data = src.read_bytes()
    marker = b"Variables:\n"
    idx = data.index(marker)
    assert idx > 0, f"expected {marker!r} in {src}"
    new_data = data[:idx] + b"Offset: 5.000000000000000e-09\n" + data[idx:]
    (HERE / "b_tran_bin_offset.raw").write_bytes(new_data)


# ---------------------------------------------------------------------------
# HAND-CRAFTED fixtures
# ---------------------------------------------------------------------------

# A matched pair, identical but for one Offset: line, so a test can diff
# them directly instead of trusting a remembered constant.
OFFSET_ASCII_BASELINE = """\
Title: b offset ascii baseline (time scale, no Offset)
Date: Mon Jul 20 00:00:00 2026
Plotname: Transient Analysis
Flags: real
No. Variables: 2
No. Points: 3
Variables:
\t0\ttime\ttime
\t1\tv(1)\tvoltage
Values:
 0\t0.000000000000000e+00
\t1.000000000000000e+00
 1\t1.000000000000000e-09
\t2.000000000000000e+00
 2\t2.000000000000000e-09
\t3.000000000000000e+00
"""

OFFSET_ASCII = """\
Title: b offset ascii (time scale, with Offset)
Date: Mon Jul 20 00:00:00 2026
Plotname: Transient Analysis
Flags: real
No. Variables: 2
No. Points: 3
Offset: 5.000000000000000e-09
Variables:
\t0\ttime\ttime
\t1\tv(1)\tvoltage
Values:
 0\t0.000000000000000e+00
\t1.000000000000000e+00
 1\t1.000000000000000e-09
\t2.000000000000000e+00
 2\t2.000000000000000e-09
\t3.000000000000000e+00
"""

# Offset on a vector 0 that is NOT named/typed "time" - proves the effect is
# purely positional (whatever sits at vectors[0]), not conditioned on the
# scale being a time axis.
OFFSET_NONTIME_SCALE = """\
Title: b offset on a non-time scale (positional check)
Date: Mon Jul 20 00:00:00 2026
Plotname: AC Analysis
Flags: real
No. Variables: 2
No. Points: 3
Offset: 2.500000000000000e+02
Variables:
\t0\tfrequency\tfrequency
\t1\tv(1)\tvoltage
Values:
 0\t1.000000000000000e+03
\t1.000000000000000e+00
 1\t2.000000000000000e+03
\t2.000000000000000e+00
 2\t3.000000000000000e+03
\t3.000000000000000e+00
"""

OFFSET_ZERO = """\
Title: b offset zero
Date: Mon Jul 20 00:00:00 2026
Plotname: Transient Analysis
Flags: real
No. Variables: 2
No. Points: 2
Offset: 0.000000000000000e+00
Variables:
\t0\ttime\ttime
\t1\tv(1)\tvoltage
Values:
 0\t1.000000000000000e-09
\t1.000000000000000e+00
 1\t2.000000000000000e-09
\t2.000000000000000e+00
"""

# Multiple trailing attributes per Variables: line, space-separated after the
# type field - matching the exact separator real ngspice uses for its own
# single-attribute case ("frequency grid=3" in b_ac_complex.raw).
ATTRS_TRAILING = """\
Title: b attrs trailing
Date: Mon Jul 20 00:00:00 2026
Plotname: Transient Analysis
Flags: real
No. Variables: 2
No. Points: 2
Variables:
\t0\ttime\ttime dims=2 min=0 max=1
\t1\tv(1)\tvoltage grid=3
Values:
 0\t0.000000000000000e+00
\t1.000000000000000e+00
 1\t1.000000000000000e-09
\t2.000000000000000e+00
"""

ZERO_POINTS = """\
Title: b zero points
Date: Mon Jul 20 00:00:00 2026
Plotname: Transient Analysis
Flags: real
No. Variables: 2
No. Points: 0
Variables:
\t0\ttime\ttime
\t1\tv(1)\tvoltage
Values:
"""

# Every header keyword cased differently than ngspice's own
# Title:/Date:/Plotname:/Flags:/No. Variables:/No. Points:/Variables:/Values:
# convention, to exercise the documented case-insensitivity of nutmeg
# keywords.
LOWERCASE_KEYWORDS = """\
TITLE: b lowercase/mixed-case keywords
date: Mon Jul 20 00:00:00 2026
PlotName: Transient Analysis
FLAGS: real
No. VARIABLES: 2
no. points: 2
VARIABLES:
\t0\ttime\ttime
\t1\tv(1)\tvoltage
values:
 0\t0.000000000000000e+00
\t1.000000000000000e+00
 1\t1.000000000000000e-09
\t2.000000000000000e+00
"""

FLAGS_REAL_PADDED = """\
Title: b flags real padded
Date: Mon Jul 20 00:00:00 2026
Plotname: Transient Analysis
Flags: real padded
No. Variables: 2
No. Points: 2
Variables:
\t0\ttime\ttime
\t1\tv(1)\tvoltage
Values:
 0\t0.000000000000000e+00
\t1.000000000000000e+00
 1\t1.000000000000000e-09
\t2.000000000000000e+00
"""

FLAGS_REAL_UNPADDED = """\
Title: b flags real unpadded
Date: Mon Jul 20 00:00:00 2026
Plotname: Transient Analysis
Flags: real unpadded
No. Variables: 2
No. Points: 2
Variables:
\t0\ttime\ttime
\t1\tv(1)\tvoltage
Values:
 0\t0.000000000000000e+00
\t1.000000000000000e+00
 1\t1.000000000000000e-09
\t2.000000000000000e+00
"""

FLAGS_COMPLEX_PADDED = """\
Title: b flags complex padded
Date: Mon Jul 20 00:00:00 2026
Plotname: AC Analysis
Flags: complex padded
No. Variables: 2
No. Points: 2
Variables:
\t0\tfrequency\tfrequency
\t1\tv(1)\tvoltage
Values:
 0\t1.000000000000000e+03,0.000000000000000e+00
\t1.000000000000000e+00,5.000000000000000e-01
 1\t2.000000000000000e+03,0.000000000000000e+00
\t2.000000000000000e+00,1.000000000000000e+00
"""

FLAGS_COMPLEX_UNPADDED = """\
Title: b flags complex unpadded
Date: Mon Jul 20 00:00:00 2026
Plotname: AC Analysis
Flags: complex unpadded
No. Variables: 2
No. Points: 2
Variables:
\t0\tfrequency\tfrequency
\t1\tv(1)\tvoltage
Values:
 0\t1.000000000000000e+03,0.000000000000000e+00
\t1.000000000000000e+00,5.000000000000000e-01
 1\t2.000000000000000e+03,0.000000000000000e+00
\t2.000000000000000e+00,1.000000000000000e+00
"""

HAND_CRAFTED = {
    "b_offset_ascii_baseline.raw": OFFSET_ASCII_BASELINE,
    "b_offset_ascii.raw": OFFSET_ASCII,
    "b_offset_nontime_scale.raw": OFFSET_NONTIME_SCALE,
    "b_offset_zero.raw": OFFSET_ZERO,
    "b_attrs_trailing.raw": ATTRS_TRAILING,
    "b_zero_points.raw": ZERO_POINTS,
    "b_lowercase_keywords.raw": LOWERCASE_KEYWORDS,
    "b_flags_real_padded.raw": FLAGS_REAL_PADDED,
    "b_flags_real_unpadded.raw": FLAGS_REAL_UNPADDED,
    "b_flags_complex_padded.raw": FLAGS_COMPLEX_PADDED,
    "b_flags_complex_unpadded.raw": FLAGS_COMPLEX_UNPADDED,
}


def main() -> None:
    _run_ngspice(TRAN_NETLIST)
    _run_ngspice(AC_NETLIST)
    _run_ngspice(TRAN_BIN_NETLIST)
    _splice_offset_into_binary()

    for name, content in HAND_CRAFTED.items():
        (HERE / name).write_text(content)

    all_names = [
        "b_tran_real.raw",
        "b_ac_complex.raw",
        "b_tran_bin_real.raw",
        "b_tran_bin_offset.raw",
        *HAND_CRAFTED.keys(),
    ]
    for name in all_names:
        path = HERE / name
        assert path.exists(), f"expected {path} to be produced"
        print(f"{name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
