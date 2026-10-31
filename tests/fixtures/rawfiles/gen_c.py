"""Generator for the ``c_*`` rawfile fixtures (slice C: Xyce rawfiles).

Fixtures and expected values come from (1) the published nutmeg/rawfile format
description, (2) real Xyce 7.10 and ngspice-42 output generated locally (the two
simulator binaries, run directly -- see below), and (3) physics/arithmetic
derived from the decks' own component values (a 1k/1k divider must output
Vin/2; an RC low-pass has a known transfer function; a log sweep's frequency
points follow a closed-form formula).

Every fixture in this file is REAL: produced by actually invoking Xyce
(``Xyce Release 7.10.0-opensource``) on the decks embedded below, not
hand-crafted bytes. Re-run this script to regenerate the fixtures (the *data*
is bit-for-bit reproducible; the header's ``Title``/``Date`` lines will of
course differ, since ``Date`` embeds real wall-clock time). The Xyce binary is
resolved from ``$SILK_XYCE_BIN``, falling back to ``Xyce`` on ``$PATH``:

    SILK_XYCE_BIN=/path/to/Xyce python tests/fixtures/rawfiles/gen_c.py

Fixtures produced (all written into this directory):

* ``c_tran_bin.raw``        — ``.tran``, binary, real vectors (``Flags: real``).
  Produced via silk's OWN Xyce invocation convention:
  ``.print tran format=raw file=... v(in) v(out)`` (see
  ``silk/circuit/xyce_netlister.py``). Names come back **upper-cased and
  wrapped**: ``TIME``, ``V(IN)``, ``V(OUT)``.
* ``c_dc_sweep_bin.raw``    — ``.dc`` sweep, binary, real vectors. Same
  ``.print dc format=raw file=...`` convention. The independent-axis vector
  (position 0) is named **``sweep``** (all lowercase — NOT ``SWEEP``, NOT
  ``V(V1)``) with type ``voltage``. This is exactly why silk keys the x-axis
  off vector *position*, never a name whitelist.
* ``c_ac_bin.raw``          — ``.ac`` sweep, binary, **complex** vectors
  (``Flags: complex``). Same ``.print ac format=raw file=...`` convention.
* ``c_tran_ascii_cli.raw``  — the ONLY way to get an ASCII nutmeg rawfile out
  of Xyce: the CLI flags ``-r <file> -a``, NOT ``.print ... format=raw``.
  See "Xyce ASCII rawfiles — findings" below; this fixture is real but is
  NOT produced via the invocation path silk's ``XyceRunner`` actually uses.

All three circuits are simple RC networks or a resistive divider (no PDK
required, no ``.options parser scale=...`` needed).

---------------------------------------------------------------------------
Xyce vs. ngspice header/format findings (for the reader implementer)
---------------------------------------------------------------------------

1. **Xyce's ``.print <analysis> format=raw`` is ALWAYS binary.** There is no
   ASCII variant reachable from the deck. Tried ``.options output
   ascii=true`` — Xyce accepts the netlist but emits
   ``Netlist warning ... No options parameter ascii found, parameter will be
   ignored`` and writes binary anyway. The ONLY ASCII nutmeg output Xyce can
   produce is via the top-level CLI flags ``-r <file> -a`` (``-a`` "use with
   -r <file> to output in ascii format" per ``Xyce -h``) — and that path is
   categorically different, not a format toggle on the same output:
     * it IGNORES any ``.print ... format=raw file=...`` line in the deck
       entirely (confirmed: a deck with both ``-r`` on the CLI and a
       ``.print`` line only ever produces the CLI-named file; the
       ``.print``-named file is never written);
     * it dumps **every** circuit variable automatically (node voltages +
       branch currents), not the ``.print``-selected subset — e.g. our
       2-resistor RC deck's binary ``.print`` fixture has 3 vars
       (``TIME``, ``V(IN)``, ``V(OUT)``) but the ASCII CLI dump of the SAME
       circuit has 4 (``time``, ``IN``, ``OUT``, ``V1#branch``);
     * its **naming convention differs from Xyce's own ``.print`` path**:
       the independent axis is lowercase (``time``, not ``TIME``), node
       names are bare (``IN``/``OUT``, not ``V(IN)``/``V(OUT)``), and branch
       currents appear as ``<source>#branch`` (type ``current``). So even
       within Xyce itself, "upper-case, ``V(...)``-wrapped" is NOT a
       universal Xyce rule — it is specifically what ``.print format=raw``
       does. This fixture pins the CLI dump's own (different) convention so
       the reader isn't tacitly assuming one Xyce naming scheme everywhere.
   Since silk's ``XyceRunner``/``XyceNetlister`` only ever use ``.print ...
   format=raw`` (see ``silk/circuit/xyce_netlister.py``), the binary path is
   what matters for silk; the ASCII fixture is included for reader
   robustness/completeness and is clearly labeled.

2. **Header field padding — ``No. Points:`` is fixed-width in Xyce, NOT in
   ngspice.** Every Xyce rawfile observed (all 4 fixtures here, tran/dc/ac,
   bin/ascii) pads the ``No. Points: <n>`` line with trailing spaces out to
   a fixed 30-byte line width, e.g. ``b'No. Points: 70                '``
   (30 bytes) or ``b'No. Points: 6                 '`` (30 bytes) — same
   width regardless of digit count. A same-circuit ngspice-42 run
   (``ngspice -b``, plain ``write``, no PDK) of the identical RC deck gives
   ``b'No. Points: 22'`` — **14 bytes, no padding at all.** A reader that
   does ``int(line.split(':')[1])`` is fine either way, but a reader that
   does an exact-length/format check on the header line will break on Xyce
   output. (Likely a legacy spice3f5 "leave room to patch this in place if
   the run is streaming" convention that ngspice has dropped and Xyce has
   kept — not investigated further; it's a fact about the bytes, not a
   theory about why.)

3. **No footer/banner in the .raw file itself.** Verified byte-exact for all
   4 fixtures: ``header_bytes + npoints * nvars * itemsize == file size``
   with ``itemsize`` 8 (real) or 16 (complex, stored as two float64 per
   value) and nothing left over. Xyce's "End of Xyce(TM) Simulation" banner
   goes to stdout/log only, never into the rawfile.

4. **Xyce refuses multiple analysis types printed to the same run.** A deck
   with both ``.tran``+``.print tran format=raw file=X`` AND
   ``.dc``+``.print dc format=raw file=X`` (same or different target files)
   is rejected outright: ``Netlist error: Analysis type TRAN and print type
   DC are inconsistent.`` Two ``.print`` lines for the SAME analysis type
   targeting the same file do NOT append a second plot either — they merge
   into a single ``Plotname``/``Variables``/``Binary`` block with the union
   of requested vectors, in the order the ``.print`` lines appear (confirmed
   by experiment, not shipped as a fixture). **Net: every rawfile Xyce
   produces via ``.print ... format=raw`` — which is the only mechanism
   silk's ``XyceNetlister`` uses — contains exactly one plot.** Multi-plot
   accumulation (relevant for ngspice's sequential ``.control`` ``write``
   calls) does not arise for Xyce output; ``result.plots`` should have
   length 1 for every real Xyce fixture here.

5. Both simulators use the SAME ``Plotname:`` strings for the analyses
   tested here (``"Transient Analysis"``, ``"AC Analysis"`` — DC transfer
   characteristic not cross-checked against ngspice for this slice) — no
   difference found there.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).parent

# Not on $PATH — must be invoked by absolute path.
XYCE = os.environ.get("SILK_XYCE_BIN", "Xyce")

TRAN_NETLIST = """\
RC charging transient - real vectors, binary rawfile (Xyce)
V1 in 0 PULSE(0 1 0 1u 1u 8u 20u)
R1 in out 1k
C1 out 0 1n
.tran 1u 5u 0 1u
.print tran format=raw file=c_tran_bin.raw v(in) v(out)
.end
"""

DC_NETLIST = """\
DC sweep - resistor divider, binary rawfile (Xyce)
V1 in 0 1
R1 in out 1k
R2 out 0 1k
.dc V1 0 5 1
.print dc format=raw file=c_dc_sweep_bin.raw v(out)
.end
"""

AC_NETLIST = """\
AC sweep - RC low-pass, complex vectors, binary rawfile (Xyce)
V1 in 0 AC 1
R1 in out 1k
C1 out 0 1n
.ac dec 10 1 1e6
.print ac format=raw file=c_ac_bin.raw v(out)
.end
"""

# No .print line: the CLI -r/-a flags below both select the rawfile AND
# override/ignore any .print format=raw the deck might otherwise carry (see
# finding #1 above), so there is no point writing one.
TRAN_ASCII_CLI_NETLIST = """\
RC charging transient - ASCII rawfile via Xyce CLI -r/-a (Xyce)
V1 in 0 PULSE(0 1 0 1u 1u 8u 20u)
R1 in out 1k
C1 out 0 1n
.tran 1u 5u 0 1u
.end
"""


def _run(netlist: str, extra_args: list[str] | None = None) -> None:
    """Write *netlist* to a temp .cir and run Xyce from this directory.

    Xyce's ``.print ... format=raw file=...`` writes relative to the
    process cwd, so we run with cwd=HERE and let the deck name the fixture
    file directly (matching gen_a.py's approach for ngspice).
    """
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        f.write(netlist)
        cir_path = Path(f.name)
    try:
        cmd = [XYCE, *(extra_args or []), str(cir_path)]
        result = subprocess.run(
            cmd,
            cwd=HERE,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Xyce failed (rc={result.returncode}): {cmd}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
    finally:
        cir_path.unlink(missing_ok=True)


def main() -> None:
    _run(TRAN_NETLIST)
    _run(DC_NETLIST)
    _run(AC_NETLIST)
    _run(TRAN_ASCII_CLI_NETLIST, extra_args=["-r", "c_tran_ascii_cli.raw", "-a"])
    for name in (
        "c_tran_bin.raw",
        "c_dc_sweep_bin.raw",
        "c_ac_bin.raw",
        "c_tran_ascii_cli.raw",
    ):
        path = HERE / name
        assert path.exists(), f"expected {path} to be produced by Xyce"
        print(f"{name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
