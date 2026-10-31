"""Generator for the ``e_*`` rawfile fixtures (slice E: metric realism).

Every fixture here is REAL simulator output: ngspice (``$SILK_NGSPICE_BIN``,
default ``ngspice`` on ``$PATH`` — ngspice-42 when these were committed) and
Xyce 7.10 (``$SILK_XYCE_BIN``, default ``Xyce`` on ``$PATH``) run on the
decks embedded below. Re-run to regenerate::

    SILK_XYCE_BIN=/path/to/Xyce python tests/fixtures/rawfiles/gen_e.py

Why this slice exists (TASK-74)
-------------------------------
The ``a_``/``b_``/``c_``/``d_`` fixtures were built for READER conformance
(flags, padding, ``Offset:``, multiplot) and are tiny — a 5-point AC sweep
cannot exercise ``bandwidth_3dB``. No test drove ``silk.metrics`` or
``silk.metrics`` from a real simulator rawfile; every metric test
hand-built its Signal, and that is exactly how F12 (``gain_bw_product``
reading the −3 dB point off ``Re(H)``, 37 % low) stayed green through five
test touchpoints. These fixtures are sized and shaped so each analysis
family's metrics are genuinely exercised, and they are produced by BOTH
simulators because Xyce's output contract differs from ngspice's in the ways
past defects hid behind (upper-cased vector names, ``sweep`` as the DC axis,
noise as a CSV instead of a rawfile).

Every circuit is PDK-free (behavioural E/B sources, ideal R/L/C, one
default-model diode) and has a CLOSED-FORM answer derived from the component
values in :data:`CIRCUIT`, which ``tests/metrics/test_rawfile_realism.py``
imports by path so the oracle and the deck can never drift apart.

Fixtures (``e_ng_*`` = ngspice, ``e_xy_*`` = Xyce; same circuit in each pair):

* ``e_ng_ac.raw`` / ``e_xy_ac.raw`` — ``.ac dec 100 10 100Meg`` (701 points,
  complex vectors) of a 3-pole behavioural amplifier: A0 = 1000 (60 dB),
  poles at 10 kHz, 3 MHz, 30 MHz. ``v(p1)`` is the pure single-pole node
  after stage 1 (GBW = A0·fp1 = 1e7 exactly), ``v(out)`` the 3-pole output
  with a real unity-gain crossing (~5.0 MHz, PM ~21.5°) and a real phase
  crossover (~9.5 MHz, GM ~10.4 dB) inside the sweep.
* ``e_ng_tran.raw`` / ``e_xy_tran.raw`` — ``.tran 20n 50u 0 20n`` of three
  independent nets in one deck: ``v(out)`` = series-RLC step response
  (ζ = 0.3, ωn = 1e6 rad/s: 37.2 % overshoot, ringing, 1 % settling ~15 µs),
  ``v(clk)`` = 8 µs / 50 % clock (1.8 V), ``v(tone)`` = 0.5 V sine at
  273 437.5 Hz (coherent for tclk = 100 ns, 256 samples, bin 7), plus the
  step source's branch current ``i(v1)`` (energy from source = C·V² exactly).
  The time grid is the simulator's own adaptive one (non-uniform, ~2.6 k
  rows) — the shape ``silk.metrics.waveform`` really sees.
* ``e_ng_dc.raw`` / ``e_xy_dc.raw`` — ``.dc V1 -2 2 0.01`` (401 points) of a
  ``tanh(2·V)`` B-source transfer. The x-axis vector is ``v(v-sweep)`` on
  ngspice and ``sweep`` on Xyce — the axis names that broke DC readback
  before silk keyed the axis off vector position.
* ``e_ng_ramp.raw`` / ``e_xy_ramp.raw`` — slow-ramp transient (0→1 V over
  100 µs) through a 3-bit behavioural quantizer with a deliberate 5 %
  quadratic non-linearity, for ``extract_code_transitions`` → INL/DNL. The
  quantizer is a sum of steep ``tanh`` steps (0.002 LSB wide), not
  ``floor()``: Xyce cannot step across a hard B-source discontinuity
  (time-step-too-small at the first transition), ngspice can; a smooth
  quantizer lets both simulators run the SAME deck.
* ``e_ng_op.raw`` / ``e_xy_op.raw`` — ``.op`` of a resistive divider (1.5 V
  into 1 kΩ + 2 kΩ: ``v(mid)`` = 1.0 V, ``i(v1)`` = −0.5 mA). One point and
  NO independent variable. ngspice writes plotname ``Operating Point`` with
  the first node, ``v(in)``, as vector 0; Xyce has no ``.print op`` ("Analysis
  type OP and print type OP are inconsistent") — its ``.op`` is written by
  ``.print dc format=raw`` under plotname ``DC transfer characteristic`` with
  a placeholder ``sweep`` vector 0 holding 0.0 (NOT the source value). The
  shape ``rawfile_analysis_kind`` has to tell from a DC sweep (TASK-102).
* ``e_ng_noise.raw`` / ``e_xy_noise/noise.csv`` — ``.noise v(out) V1 dec 20 1
  1Meg`` of a diode (shot + 1/f via KF) in parallel with a 1 kΩ thermal
  source, buffered by a gain-100 stage; 1/f corner designed at 1 kHz. Xyce
  cannot write noise to a rawfile, so its fixture is the CSV
  ``.print noise format=csv`` produces, placed in a sub-directory under the
  name ``XyceBackend.load_noise`` resolves (``<dir of rawfile>/noise.csv``).
  The ngspice rawfile carries the per-device breakdown vectors ngspice
  writes alongside ``onoise_spectrum``/``inoise_spectrum`` — realistic
  clutter the bare-name match in ``from_rawfile`` has to pick through.

Simulator facts recorded while building this slice:

* ngspice-42's B-source ``tanh`` reaches the closed form only to ~1.5e-4
  (its default ``reltol``); Xyce is exact to 1e-16. The DC oracle uses a
  1e-3 tolerance for that reason, not because of anything in silk.
* Both simulators agree with the closed-form noise spectrum to < 6e-5
  relative at every one of the 121 points, and with the 3-pole AC transfer
  function to 2e-8.
* Xyce's ``.print tran format=raw`` upper-cases the axis name (``TIME``,
  ``FREQUENCY``); ngspice writes lowercase. ``from_rawfile`` passes the
  vector-0 name through verbatim as ``xlabel``.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).parent

NGSPICE = os.environ.get("SILK_NGSPICE_BIN", "ngspice")
XYCE = os.environ.get("SILK_XYCE_BIN", "Xyce")

# ---------------------------------------------------------------------------
# Circuit constants — the single source of truth for the decks below AND the
# closed-form oracle in tests/metrics/test_rawfile_realism.py.
# ---------------------------------------------------------------------------

CIRCUIT = {
    # --- AC: 3-pole behavioural amplifier ---
    'ac': {
        'A0': 1000.0,          # DC gain, V/V (60 dB)
        'R': 1e3,              # every stage uses the same series R
        'fp': (1e4, 3e6, 3e7), # pole frequencies, Hz (C = 1/(2*pi*R*fp))
        'fstart': 10.0, 'fstop': 100e6, 'pts_per_dec': 100,
    },
    # --- TRAN: series RLC step + clock + coherent tone ---
    'tran': {
        'V_step': 1.0, 't_step': 1e-6, 'edge': 10e-9,
        'R': 600.0, 'L': 1e-3, 'C': 1e-9,   # wn = 1/sqrt(LC) = 1e6, zeta = R/2*sqrt(C/L) = 0.3
        'V_clk': 1.8, 'clk_delay': 0.5e-6, 'clk_edge': 5e-9,
        'clk_period': 8e-6, 'clk_duty': 0.5,
        'A_tone': 0.5, 'f_tone': 273437.5,  # = 7/256 * 10 MHz (tclk = 100 ns)
        'tstep': 20e-9, 'tstop': 50e-6,
    },
    # --- DC: tanh(k*V) transfer ---
    'dc': {'k': 2.0, 'vstart': -2.0, 'vstop': 2.0, 'vstep': 0.01},
    # --- RAMP: 3-bit smooth quantizer with quadratic INL ---
    'ramp': {
        'bits': 3, 'nl': 0.05, 'width_lsb': 0.002,
        'v_end': 1.0, 't_end': 100e-6, 'tstep': 50e-9,
    },
    # --- OP: resistive divider, one operating point ---
    'op': {'V': 1.5, 'R1': 1e3, 'R2': 2e3},   # v(mid) = V*R2/(R1+R2), i(v1) = -V/(R1+R2)
    # --- NOISE: diode (shot + 1/f) || R thermal, gain stage ---
    'noise': {
        'R1': 1e3, 'I_bias': 100e-6, 'G': 100.0, 'C_couple': 1.0,
        'IS': 1e-14, 'KF': 4.86e-16, 'AF': 1.0,
        'fstart': 1.0, 'fstop': 1e6, 'pts_per_dec': 20,
        'T_kelvin': 300.15,   # SPICE default 27 C — sets kT and Vt = kT/q
    },
}


def _cap(fp: float, r: float) -> float:
    return 1.0 / (2.0 * math.pi * r * fp)


# ---------------------------------------------------------------------------
# Deck bodies (simulator-neutral unless noted)
# ---------------------------------------------------------------------------

def ac_core() -> str:
    a = CIRCUIT['ac']
    r = a['R']
    c1, c2, c3 = (_cap(fp, r) for fp in a['fp'])
    return (
        "V1 in 0 DC 0 AC 1\n"
        f"E1 p1a 0 in 0 {a['A0']:g}\n"
        f"R1 p1a p1 {r:g}\n"
        f"C1 p1 0 {c1:.10g}\n"
        "E2 p2a 0 p1 0 1\n"
        f"R2 p2a p2 {r:g}\n"
        f"C2 p2 0 {c2:.10g}\n"
        "E3 p3a 0 p2 0 1\n"
        f"R3 p3a out {r:g}\n"
        f"C3 out 0 {c3:.10g}\n"
        f".ac dec {a['pts_per_dec']} {a['fstart']:g} {a['fstop']:g}\n"
    )


def tran_core() -> str:
    t = CIRCUIT['tran']
    pw = t['clk_period'] * t['clk_duty'] - t['clk_edge']
    return (
        f"V1 step 0 PULSE(0 {t['V_step']:g} {t['t_step']:g} {t['edge']:g} "
        f"{t['edge']:g} 500u 1m)\n"
        f"R1 step n1 {t['R']:g}\n"
        f"L1 n1 out {t['L']:g}\n"
        f"C1 out 0 {t['C']:g}\n"
        f"V2 clk 0 PULSE(0 {t['V_clk']:g} {t['clk_delay']:g} {t['clk_edge']:g} "
        f"{t['clk_edge']:g} {pw:.10g} {t['clk_period']:g})\n"
        f"V3 tone 0 SIN(0 {t['A_tone']:g} {t['f_tone']:.10g})\n"
        f".tran {t['tstep']:g} {t['tstop']:g} 0 {t['tstep']:g}\n"
    )


def dc_core(sim: str) -> str:
    d = CIRCUIT['dc']
    expr = f"tanh({d['k']:g}*V(in))"
    if sim == 'xyce':
        expr = '{' + expr + '}'
    return (
        "V1 in 0 0\n"
        f"B1 out 0 V={expr}\n"
        f".dc V1 {d['vstart']:g} {d['vstop']:g} {d['vstep']:g}\n"
    )


def ramp_code_expr() -> str:
    """Smooth 3-bit quantizer of u = 2^bits * (V + nl*V^2): sum of tanh steps."""
    r = CIRCUIT['ramp']
    n = 2 ** r['bits']
    u = f"{n}*(V(in)+{r['nl']:g}*V(in)*V(in))"
    return "+".join(
        f"0.5*(1+tanh(({u}-{k})/{r['width_lsb']:g}))" for k in range(1, n + 1)
    )


def ramp_core(sim: str) -> str:
    r = CIRCUIT['ramp']
    expr = ramp_code_expr()
    if sim == 'xyce':
        expr = '{' + expr + '}'
    return (
        f"V1 in 0 PWL(0 0 {r['t_end']:g} {r['v_end']:g})\n"
        f"B1 code 0 V={expr}\n"
        f".tran {r['tstep']:g} {r['t_end']:g} 0 {r['tstep']:g}\n"
    )


def op_core() -> str:
    o = CIRCUIT['op']
    return (
        f"V1 in 0 {o['V']:g}\n"
        f"R1 in mid {o['R1']:g}\n"
        f"R2 mid 0 {o['R2']:g}\n"
        ".op\n"
    )


def noise_core() -> str:
    n = CIRCUIT['noise']
    return (
        "V1 in 0 DC 0 AC 1\n"
        f"C0 in inb {n['C_couple']:g}\n"
        f"R1 inb mid {n['R1']:g}\n"
        f"I1 0 mid {n['I_bias']:g}\n"
        "D1 mid 0 dmod\n"
        f".model dmod D(IS={n['IS']:g} KF={n['KF']:g} AF={n['AF']:g})\n"
        f"E1 out 0 mid 0 {n['G']:g}\n"
        f".noise v(out) V1 dec {n['pts_per_dec']} {n['fstart']:g} {n['fstop']:g} 1\n"
    )


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def _run(cmd: list[str], netlist: str, what: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False,
                                     dir=HERE) as f:
        f.write(netlist)
        cir_path = Path(f.name)
    try:
        result = subprocess.run(
            [*cmd, str(cir_path)], cwd=HERE, capture_output=True, text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{what} failed (rc={result.returncode}): {cmd}\n"
                f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
            )
    finally:
        cir_path.unlink(missing_ok=True)
        # Xyce drops a '<deck>_noise.dat' side file next to a noise deck.
        for side in HERE.glob(cir_path.name + '*'):
            side.unlink()


def run_ngspice(title: str, core: str, outfile: str, save: str,
                noise: bool = False) -> None:
    """ngspice batch run; ``save`` restricts the rawfile to the named vectors
    (empty string = ngspice's default, everything)."""
    deck = f"* {title}\n{core}.control\n"
    if save:
        deck += f"save {save}\n"
    deck += "run\n"
    if noise:
        # ngspice leaves the integrated-noise plot current after .noise;
        # select the spectral-density plot before writing (same as silk's
        # own control block in silk/circuit/spice_netlister.py).
        deck += "setplot noise1\n"
    deck += f"write {outfile}\nquit\n.endc\n.end\n"
    _run([NGSPICE, "-b"], deck, "ngspice")


def run_xyce(title: str, core: str, kind: str, outfile: str, vars_: str) -> None:
    fmt = "csv" if kind == "noise" else "raw"
    deck = (f"* {title}\n{core}"
            f".print {kind} format={fmt} file={outfile} {vars_}\n.end\n")
    _run([XYCE], deck, "Xyce")


FIXTURES = (
    "e_ng_ac.raw", "e_ng_tran.raw", "e_ng_dc.raw", "e_ng_ramp.raw", "e_ng_noise.raw",
    "e_ng_op.raw",
    "e_xy_ac.raw", "e_xy_tran.raw", "e_xy_dc.raw", "e_xy_ramp.raw",
    "e_xy_noise/noise.csv", "e_xy_op.raw",
)


def gen_op() -> None:
    """The two ``.op`` fixtures alone (TASK-102 added them after the rest
    were committed; regenerating everything would churn the other eight)."""
    run_ngspice("resistive divider operating point (metric realism)", op_core(),
                "e_ng_op.raw", "v(in) v(mid) i(v1)")
    # Xyce prints an .op result through the DC print type (see module doc).
    run_xyce("resistive divider operating point (metric realism)", op_core(),
             "dc", "e_xy_op.raw", "v(in) v(mid) i(v1)")


def main() -> None:
    (HERE / "e_xy_noise").mkdir(exist_ok=True)

    run_ngspice("3-pole amplifier AC sweep (metric realism)", ac_core(),
                "e_ng_ac.raw", "v(in) v(p1) v(out)")
    run_ngspice("RLC step + clock + tone transient (metric realism)", tran_core(),
                "e_ng_tran.raw", "v(step) v(out) v(clk) v(tone) i(v1)")
    run_ngspice("tanh transfer DC sweep (metric realism)", dc_core('ngspice'),
                "e_ng_dc.raw", "v(in) v(out)")
    run_ngspice("3-bit quantizer ramp (metric realism)", ramp_core('ngspice'),
                "e_ng_ramp.raw", "v(in) v(code)")
    run_ngspice("diode 1/f + thermal noise (metric realism)", noise_core(),
                "e_ng_noise.raw", "", noise=True)

    run_xyce("3-pole amplifier AC sweep (metric realism)", ac_core(),
             "ac", "e_xy_ac.raw", "v(in) v(p1) v(out)")
    run_xyce("RLC step + clock + tone transient (metric realism)", tran_core(),
             "tran", "e_xy_tran.raw", "v(step) v(out) v(clk) v(tone) i(v1)")
    run_xyce("tanh transfer DC sweep (metric realism)", dc_core('xyce'),
             "dc", "e_xy_dc.raw", "v(in) v(out)")
    run_xyce("3-bit quantizer ramp (metric realism)", ramp_core('xyce'),
             "tran", "e_xy_ramp.raw", "v(in) v(code)")
    run_xyce("diode 1/f + thermal noise (metric realism)", noise_core(),
             "noise", "e_xy_noise/noise.csv", "ONOISE INOISE")

    gen_op()

    for name in FIXTURES:
        path = HERE / name
        assert path.exists(), f"expected {path} to be produced"
        print(f"{name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
