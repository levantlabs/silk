"""Generator for the ``d_*`` rawfile fixtures (slice D: multi-plot rawfiles
and the constants-only abort case).

Each fixture below is labeled REAL or HAND-CRAFTED (real-block concatenation):

* REAL — produced by actually invoking ``ngspice`` (ngspice-42,
  ``/usr/bin/ngspice``) on the netlists embedded below. Re-run this script to
  regenerate them bit-for-bit:

      python tests/fixtures/rawfiles/gen_d.py

* HAND-CRAFTED (real-block concatenation) — built by literally slicing and
  concatenating the bytes of REAL single-plot ngspice rawfiles produced in
  this same script run. Confirmed empirically (see "ngspice multi-plot
  mechanics" below) that a genuine multi-plot ngspice rawfile IS just N
  single-plot blocks (full text header + ``Binary:``/``Values:`` + payload)
  placed back-to-back with zero separator/padding between them — so
  concatenating independently-real blocks produces bytes indistinguishable
  from what ngspice itself writes for that block ordering. Used for
  orderings ngspice cannot be made to produce directly (constants-then-real,
  a reversed real+real order, and two real plots with deliberately differing
  ``nvars``).

No expected value anywhere in this file or in the test suite that reads
these fixtures derives from running or reading any third-party rawfile
reader. Every REAL fixture's expected values are read back from the
fixture's own ASCII sibling (self-describing text) or from the circuit's
known physics; every HAND-CRAFTED fixture's layout is derived from the
published nutmeg format description.

## ngspice multi-plot mechanics (empirical findings, ngspice-42)

* A ``.control`` block that issues ``write file`` twice (once per analysis)
  does NOT append — the second ``write`` OVERWRITES the first. ``write file
  all`` does not help either: ``all`` selects all VECTORS of the CURRENT
  plot, not all plots. So explicit ``write`` inside ``.control`` can only
  ever produce a single-plot file, confirmed by inspecting the resulting
  byte counts and grepping for repeated ``Plotname:`` occurrences (there was
  only ever one).
* The genuine multi-plot mechanism is ngspice's **batch ``-r`` flag** with
  **multiple top-level analysis directives** (``.ac`` / ``.tran`` / ...) and
  NO ``.control`` block: ``ngspice -b -r out.raw deck.cir``. Each analysis
  ngspice actually executes gets its own plot, appended in EXECUTION order.
* Execution order is **NOT** the textual order of the directives in the
  deck: a deck with ``.tran`` written before ``.ac`` in the file still
  produced ``[AC Analysis, Transient Analysis]`` in the rawfile (verified:
  swap the two lines, rerun, ``grep -a -o 'Plotname: [A-Za-z ]+'`` is
  unchanged). ngspice-42 appears to run analysis types in a fixed internal
  order (AC before Transient) regardless of deck order. This means a real
  ngspice-42 rawfile with Transient preceding AC could not be produced in
  this environment — ``d_multiplot_order_reversed.raw`` below exists to
  cover that ordering anyway (constants-then-real has the identical
  problem, for the same underlying reason: the "current plot" is always the
  last thing genuinely computed, not a hand-picked file position).
* ``.options filetype=ascii`` (top-level, no ``.control`` needed) makes
  ``-r`` batch mode write the multi-plot file in ASCII instead of binary;
  the two plots are still just concatenated text blocks (verified: the
  second ``Title:`` line starts immediately after the first plot's last
  ``Values:`` data line, no blank line in between).
* The synthetic ``constants`` plot (``pi``/``boltz``/``echarge``/… — see
  ``silk/simulation/backend.py:_has_data_rows``) is reproduced for real by a
  deck with **no analysis directive at all** plus an explicit ``.control
  write file`` — ngspice has nothing to write, so it falls back to a plot of
  its built-in constants, ``No. Points: 1``, ``Flags: complex`` (each
  "constant" is stored as a complex pair; ``i`` is the one genuinely complex
  entry, ``0+1j``, the rest have a zero imaginary part). This reproduces the
  513-byte real fixture (12 vars: yes, false,
  true, boltz, c, e, echarge, i, kelvin, no, pi, planck) byte-for-byte.

## Fixture size

The two multi-directive circuits below are deliberately tiny: the transient
run pins ngspice's internal step ceiling explicitly (``.tran 1u 5u 0 1u``,
the same technique gen_a.py uses) to a fixed 22 points instead of the
thousand-plus points ngspice's adaptive control would otherwise take over a
longer window, and the AC sweep is a 3-point-per-decade, 3-decade sweep (10
points) instead of a fine one. Every ``d_*.raw`` fixture is well under 8 KB.
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

# Two top-level analysis directives, no .control block, run via `-r`. This is
# the real multi-plot mechanism (see docstring). AC is listed second here but
# still comes out FIRST in the file — ngspice's fixed execution-order
# behavior, not textual order (confirmed by also trying the opposite
# textual order; the output was identical). V1 combines a transient PULSE
# (for .tran) with an AC magnitude (for .ac) on one source line, same as any
# ngspice V-source: `Vxxx n+ n- [DC val] [AC mag] [transient-function]`.
# The PULSE's slow (1u) rise/no fall inside the 0..5u window plus the pinned
# Tmax is gen_a.py's technique for a small, deterministic point count (22).
MULTIPLOT_NETLIST = """\
* multi-analysis directives at top level, single -r rawfile (small)
V1 in 0 PULSE(0 1 0 1u 1u 8u 20u) AC 1
R1 in out 1k
C1 out 0 1n

.tran 1u 5u 0 1u
.ac dec 3 1 1k
.end
"""

MULTIPLOT_ASCII_NETLIST = """\
* multi-analysis directives, ascii via .options, batch -r (small)
.options filetype=ascii
V1 in 0 PULSE(0 1 0 1u 1u 8u 20u) AC 1
R1 in out 1k
C1 out 0 1n

.tran 1u 5u 0 1u
.ac dec 3 1 1k
.end
"""

# No analysis directive at all -> the synthetic 'constants' plot is the only
# thing `write` has to write.
CONSTANTS_ONLY_NETLIST = """\
* write with no analysis run -> ngspice falls back to a synthetic constants plot
V1 in 0 DC 1
R1 in 0 1k

.control
write d_constants_only.raw
.endc
.end
"""

CONSTANTS_ONLY_ASCII_NETLIST = """\
* write with no analysis run, ascii -> synthetic constants plot in ascii
V1 in 0 DC 1
R1 in 0 1k

.control
set filetype=ascii
write d_constants_only_ascii.raw
.endc
.end
"""

# A second, independent circuit whose AC plot is restricted (via `.save`) to
# 2 variables (frequency, v(out)) instead of the 4 that MULTIPLOT_NETLIST's
# circuit produces. Used only to build d_multiplot_nvars_differ.raw: two REAL
# plots that genuinely disagree on nvars (a single ngspice circuit's tran vs.
# ac plots always share the same variable set, so getting a real nvars
# mismatch needs two different circuits/analyses spliced together).
AC_SMALL_NETLIST = """\
* small AC-only circuit with restricted save set (2 vars: frequency, v(out))
V1 in 0 AC 1
R1 in out 1k
R2 out 0 1k

.save v(out)
.ac dec 3 1 1k
.end
"""


def _run_ngspice(netlist: str, *, rawfile: str | None = None, allow_nonzero: bool = False) -> None:
    """Write *netlist* to a temp .cir and invoke ngspice from this directory.

    If *rawfile* is given, pass ``-r rawfile`` (batch mode writes the rawfile
    itself — used for the multi-directive decks with no .control block).
    Otherwise the deck's own ``.control ... write ...`` block names its
    output file directly, and ngspice's cwd (=HERE) is where it lands.

    ``allow_nonzero`` — the constants-only decks deliberately have no
    analysis directive, and ngspice exits 1 for that ("no simulations run")
    even though it still writes the fallback constants-plot rawfile; this is
    itself part of the phenomenon `_has_data_rows` guards against (a rawfile
    can exist and look plausible with an unreliable exit code).
    """
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        f.write(netlist)
        cir_path = Path(f.name)
    try:
        cmd = [NGSPICE, "-b"]
        if rawfile is not None:
            cmd += ["-r", rawfile]
        cmd.append(str(cir_path))
        result = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=60)
        if result.returncode != 0 and not allow_nonzero:
            raise RuntimeError(
                f"ngspice failed (rc={result.returncode}):\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
    finally:
        cir_path.unlink(missing_ok=True)


def _split_first_plot(data: bytes) -> tuple[bytes, bytes]:
    """Split a two-plot nutmeg file into (first_plot_bytes, second_plot_bytes).

    Multi-plot ngspice files are just single-plot blocks concatenated with no
    separator (verified empirically — see module docstring), so the second
    plot starts exactly at the second occurrence of ``Title:``.
    """
    first_title = data.index(b"Title:")
    second_title = data.index(b"Title:", first_title + 1)
    return data[first_title:second_title], data[second_title:]


def main() -> None:
    # --- REAL: multi-plot, binary and ascii ---
    _run_ngspice(MULTIPLOT_NETLIST, rawfile="d_multiplot_ac_tran.raw")
    _run_ngspice(MULTIPLOT_ASCII_NETLIST, rawfile="d_multiplot_ac_tran_ascii.raw")

    # --- REAL: constants-only, binary and ascii ---
    _run_ngspice(CONSTANTS_ONLY_NETLIST, allow_nonzero=True)
    _run_ngspice(CONSTANTS_ONLY_ASCII_NETLIST, allow_nonzero=True)

    # --- REAL: small 2-var AC plot (source material for the nvars-differ fixture) ---
    _run_ngspice(AC_SMALL_NETLIST, rawfile="_d_ac_small_intermediate.raw")

    # --- HAND-CRAFTED (real-block concatenation) ---
    multiplot_bytes = (HERE / "d_multiplot_ac_tran.raw").read_bytes()
    ac_block, tran_block = _split_first_plot(multiplot_bytes)
    constants_block = (HERE / "d_constants_only.raw").read_bytes()
    ac_small_block = (HERE / "_d_ac_small_intermediate.raw").read_bytes()

    (HERE / "d_healthy_transient.raw").write_bytes(tran_block)
    (HERE / "d_multiplot_constants_first.raw").write_bytes(constants_block + tran_block)
    (HERE / "d_multiplot_constants_last.raw").write_bytes(tran_block + constants_block)
    (HERE / "d_multiplot_nvars_differ.raw").write_bytes(ac_small_block + tran_block)
    (HERE / "d_multiplot_order_reversed.raw").write_bytes(tran_block + ac_block)

    (HERE / "_d_ac_small_intermediate.raw").unlink()

    for name in (
        "d_multiplot_ac_tran.raw",
        "d_multiplot_ac_tran_ascii.raw",
        "d_constants_only.raw",
        "d_constants_only_ascii.raw",
        "d_healthy_transient.raw",
        "d_multiplot_constants_first.raw",
        "d_multiplot_constants_last.raw",
        "d_multiplot_nvars_differ.raw",
        "d_multiplot_order_reversed.raw",
    ):
        path = HERE / name
        assert path.exists(), f"expected {path} to be produced"
        print(f"{name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
