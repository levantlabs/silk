"""Generator for the ngspice `.measure` sidecar reference values (TASK-303.2).

Half 2 of TASK-303 (see TASK-303's description and
``docs/metrics-ngspice-measure-catalog.md``, TASK-303.1's catalog). Runs REAL
``.measure``/``meas`` forms against the identical decks that produce the
already-committed ``e_ng_tran.raw`` / ``e_ng_ac.raw`` fixtures (reusing
:func:`gen_e.tran_core` / :func:`gen_e.ac_core` byte-for-byte, so the
simulated waveform is the SAME one those fixtures load) and writes the
printed reference numbers into a sidecar JSON next to them. ``tests/metrics/
test_rawfile_realism.py`` asserts silk agrees with these numbers within each
metric's documented resolution.

This script never calls ngspice's ``write`` on these decks and never touches
``e_ng_tran.raw``/``e_ng_ac.raw`` — it only captures ``.measure`` results
printed to stdout in batch mode (interactive/control-block ``meas`` is
unusable together with ``-b -r``, so ``-b`` alone with top-level
``.measure`` statements is the form that works — see TASK-303's
description). Every reference number below comes from that stdout, parsed
verbatim; none is hand-typed.

Regenerate with the project's ngspice on ``$PATH`` (or ``$SILK_NGSPICE_BIN``)
and this worktree on ``PYTHONPATH`` (the tran sidecar reads the committed
``e_ng_tran.raw`` through the real ``silk.signals.rawfile`` path to choose
on-grid ``FROM``/``TO`` bounds — see :func:`_tran_window_bounds`)::

    PYTHONPATH=<worktree root> <venv>/bin/python tests/fixtures/rawfiles/gen_e_meas.py

Why on-grid FROM/TO (TASK-303.1 finding). ngspice's ``AVG``/``MIN``/``MAX``/
``PP`` snap ``FROM``/``TO`` to the nearest INSIDE samples; ``RMS``/``INTEG``
instead INTERPOLATE the exact boundary. silk's shared ``_window()`` always
snaps (matches the first group, not the second). The two conventions only
provably agree when the requested boundary lands exactly on a sample time,
so this generator reads the real boundary times off the committed fixture's
own adaptive grid instead of picking round numbers.

Why phase is read in RADIANS below. ngspice's ``.measure`` results for
``vp()`` come back in RADIANS even though the ``vp()`` vector the OS prints
interactively is in degrees (verified live below, and again while building
this generator: a naive ``FIND vp(out) WHEN vdb(out)=0`` printed
``-2.769414`` for a value the closed-form oracle in
``test_rawfile_realism.py`` puts at roughly -158.7 DEGREES — `math.degrees`
of the printed number, not the number itself). The sidecar records the raw
ngspice value and tags its unit; the differential test converts before
comparing to silk's degrees-valued ``phase_margin``.

Why there is no ``gain_margin``/phase-crossover entry here. ngspice's own
``vp()`` is wrapped to ``(-pi, pi]`` per sample, not continuously unwrapped
across the sweep (verified live while building this generator: the printed
``vp(out)`` vector jumps directly from -3.13108 to +3.13883 radians between
two adjacent samples straddling the true -180 degree crossover — it never
actually EQUALS -pi or +pi at any sample or in between, since the
discontinuity is an artifact of the wrapped representation, not a real
signal transition). A literal ``WHEN vp(out)=-3.14159265358979`` /
``FIND vdb(out) WHEN vp(out)=-3.14159265358979`` recipe therefore fails
with "out of interval" on this circuit (whose phase genuinely passes
-180 degrees on the way to its -270 degree asymptote) even though
``docs/metrics-ngspice-measure-catalog.md``'s original TASK-303.1 pass
assumed such a composition was directly buildable. silk's own
``gain_margins`` avoids exactly this trap by unwrapping the phase with
``numpy.unwrap`` before searching — this is a genuine, now-verified
convention gap (ngspice's raw ``.measure`` primitives cannot express a
phase-crossover search past the first +-180 wrap without the caller doing
the unwrapping ngspice itself never performs), not a silk defect, and is
recorded in the catalog rather than exercised as a differential assertion
here (there is nothing on the ngspice side to assert equality against).
"""
from __future__ import annotations

import importlib.util
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).parent


def _load_gen_e():
    spec = importlib.util.spec_from_file_location('gen_e', HERE / 'gen_e.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN_E = _load_gen_e()
CIRCUIT = GEN_E.CIRCUIT
NGSPICE = GEN_E.NGSPICE

_MEAS_LINE = re.compile(r'^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*'
                        r'(?P<value>[-+0-9.eE]+)')


def _run_capture(deck: str) -> str:
    """Run *deck* under ``ngspice -b`` and return its combined stdout+stderr.

    Unlike :func:`gen_e._run`, this does not check the return code: a
    ``.measure`` that ngspice refuses (``DERIV``) still exits 0 and prints
    its own ``Error:`` line, which is itself a reference fact this module
    records.
    """
    with tempfile.NamedTemporaryFile('w', suffix='.cir', delete=False,
                                     dir=HERE) as f:
        f.write(deck)
        cir_path = Path(f.name)
    try:
        result = subprocess.run([NGSPICE, '-b', str(cir_path)], cwd=HERE,
                                capture_output=True, text=True, timeout=120)
        return result.stdout + '\n' + result.stderr
    finally:
        cir_path.unlink(missing_ok=True)


def _parse_measures(output: str) -> dict:
    """Every ``name = value ...`` line ngspice printed, as ``{name: float}``.

    ngspice prints extra ``from=``/``to=``/``at=``/``trig=``/``targ=``
    fields after some results (e.g. ``vpp = 1.9e-01 from=... to=...``); only
    the FIRST ``name = value`` pair on the line is captured, matching what a
    caller reading ``.measure`` output would use as "the answer".
    """
    values = {}
    for line in output.splitlines():
        m = _MEAS_LINE.match(line.strip())
        if m:
            try:
                values[m.group('name')] = float(m.group('value'))
            except ValueError:
                continue
    return values


def _ngspice_version() -> str:
    result = subprocess.run([NGSPICE, '-v'], capture_output=True, text=True,
                            timeout=30)
    lines = (result.stdout or result.stderr).splitlines()
    return next((l.strip(' *') for l in lines if 'ngspice-' in l), 'unknown')


def _tran_window_bounds() -> tuple[float, float]:
    """On-grid ``(FROM, TO)`` for the windowed AVG/RMS/MIN/MAX/PP/INTEG
    forms — real sample times read off the already-committed
    ``e_ng_tran.raw`` fixture (never hand-picked round numbers), so both
    ngspice's snap-to-sample convention and silk's snap-to-sample
    ``_window()`` land on the identical sample set (see module docstring).
    """
    from silk.signals.rawfile import read_rawfile
    from silk.signals.spice import NgspiceSignal
    raw = read_rawfile(HERE / 'e_ng_tran.raw')
    out = NgspiceSignal.from_rawfile(raw, 'out', 'V')
    n = len(out.x)
    return float(out.x[int(0.15 * n)]), float(out.x[int(0.65 * n)])


# ---------------------------------------------------------------------------
# Decks — reuse gen_e's core builders byte-for-byte so the simulated
# waveform is the SAME one e_ng_tran.raw / e_ng_ac.raw already commit.
# ---------------------------------------------------------------------------

VALUE_AT_T = 30e-6   # inside the settled region; matches the closed-form
                     # value_at() check already in test_rawfile_realism.py


def tran_deck(t_from: float, t_to: float) -> str:
    t = CIRCUIT['tran']
    v_low_clk = 0.1 * t['V_clk']
    v_high_clk = 0.9 * t['V_clk']
    return (
        "* TASK-303.2 ngspice .measure sidecar -- tran\n"
        f"{GEN_E.tran_core()}"
        f".measure tran rise_time_clk "
        f"TRIG v(clk) VAL={v_low_clk:.17g} RISE=1 "
        f"TARG v(clk) VAL={v_high_clk:.17g} RISE=1\n"
        f".measure tran fall_time_clk "
        f"TRIG v(clk) VAL={v_high_clk:.17g} FALL=1 "
        f"TARG v(clk) VAL={v_low_clk:.17g} FALL=1\n"
        ".measure tran propagation_delay_step_out "
        "TRIG v(step) VAL=0.5 RISE=1 TARG v(out) VAL=0.5 RISE=1\n"
        ".measure tran x_when_out_050 WHEN v(out)=0.5 RISE=1\n"
        f".measure tran value_at_out FIND v(out) AT={VALUE_AT_T:.17g}\n"
        ".measure tran find_when_out_at_step_050 "
        "FIND v(out) WHEN v(step)=0.5\n"
        f".measure tran window_avg_out AVG v(out) FROM={t_from:.17g} TO={t_to:.17g}\n"
        f".measure tran window_rms_out RMS v(out) FROM={t_from:.17g} TO={t_to:.17g}\n"
        f".measure tran window_min_out MIN v(out) FROM={t_from:.17g} TO={t_to:.17g}\n"
        f".measure tran window_max_out MAX v(out) FROM={t_from:.17g} TO={t_to:.17g}\n"
        f".measure tran window_pp_out PP v(out) FROM={t_from:.17g} TO={t_to:.17g}\n"
        f".measure tran window_integ_out INTEG v(out) FROM={t_from:.17g} TO={t_to:.17g}\n"
        f".measure tran deriv_out DERIV v(out) AT={VALUE_AT_T:.17g}\n"
        ".control\n"
        "run\n"
        "quit\n"
        ".endc\n"
        ".end\n"
    )


def ac_deck() -> str:
    a = CIRCUIT['ac']
    th_3db = 20.0 * math.log10(a['A0'] / math.sqrt(2.0))
    return (
        "* TASK-303.2 ngspice .measure sidecar -- ac\n"
        f"{GEN_E.ac_core()}"
        f".measure ac bandwidth_3db_out WHEN vdb(out)={th_3db:.17g}\n"
        ".measure ac unity_gain_freq_out WHEN vm(out)=1\n"
        ".measure ac phase_margin_find_out FIND vp(out) WHEN vdb(out)=0\n"
        ".control\n"
        "run\n"
        "quit\n"
        ".endc\n"
        ".end\n"
    )


# ---------------------------------------------------------------------------
# Sidecar assembly
# ---------------------------------------------------------------------------

def build_tran_sidecar() -> dict:
    t_from, t_to = _tran_window_bounds()
    output = _run_capture(tran_deck(t_from, t_to))
    values = _parse_measures(output)

    deriv_unsupported = "function 'deriv' currently not supported" in output.lower()

    required = ('rise_time_clk', 'fall_time_clk', 'propagation_delay_step_out',
                'x_when_out_050', 'value_at_out', 'find_when_out_at_step_050',
                'window_avg_out', 'window_rms_out', 'window_min_out',
                'window_max_out', 'window_pp_out', 'window_integ_out')
    missing = [n for n in required if n not in values]
    if missing:
        raise RuntimeError(
            f'gen_e_meas: ngspice did not report {missing} -- raw output:\n{output}')

    return {
        'ngspice_form': {
            'rise_time_clk': 'TRIG v(clk) VAL=0.18 RISE=1 TARG v(clk) VAL=1.62 RISE=1',
            'fall_time_clk': 'TRIG v(clk) VAL=1.62 FALL=1 TARG v(clk) VAL=0.18 FALL=1',
            'propagation_delay_step_out':
                'TRIG v(step) VAL=0.5 RISE=1 TARG v(out) VAL=0.5 RISE=1',
            'x_when_out_050': 'WHEN v(out)=0.5 RISE=1',
            'value_at_out': f'FIND v(out) AT={VALUE_AT_T:g}',
            'find_when_out_at_step_050': 'FIND v(out) WHEN v(step)=0.5',
            'window_avg_out': f'AVG v(out) FROM={t_from!r} TO={t_to!r}',
            'window_rms_out': f'RMS v(out) FROM={t_from!r} TO={t_to!r}',
            'window_min_out': f'MIN v(out) FROM={t_from!r} TO={t_to!r}',
            'window_max_out': f'MAX v(out) FROM={t_from!r} TO={t_to!r}',
            'window_pp_out': f'PP v(out) FROM={t_from!r} TO={t_to!r}',
            'window_integ_out': f'INTEG v(out) FROM={t_from!r} TO={t_to!r}',
            'deriv_out': f'DERIV v(out) AT={VALUE_AT_T:g}',
        },
        'values': values,
        'window_from_s': t_from,
        'window_to_s': t_to,
        'value_at_t_s': VALUE_AT_T,
        'deriv': {
            'status': 'unsupported' if deriv_unsupported else 'unexpected',
            'note': ("ngspice parses DERIV but its .measure dispatch refuses "
                     "it ('function 'deriv' currently not supported') in "
                     "every version checked by TASK-303.1 -- confirmed again "
                     "here on this exact deck; no ngspice oracle exists for "
                     "silk's derivative()."),
        },
    }


def build_ac_sidecar() -> dict:
    output = _run_capture(ac_deck())
    values = _parse_measures(output)
    required = ('bandwidth_3db_out', 'unity_gain_freq_out', 'phase_margin_find_out')
    missing = [n for n in required if n not in values]
    if missing:
        raise RuntimeError(
            f'gen_e_meas: ngspice did not report {missing} -- raw output:\n{output}')

    a = CIRCUIT['ac']
    th_3db = 20.0 * math.log10(a['A0'] / math.sqrt(2.0))
    return {
        'ngspice_form': {
            'bandwidth_3db_out': f'WHEN vdb(out)={th_3db:.17g}',
            'unity_gain_freq_out': 'WHEN vm(out)=1',
            'phase_margin_find_out': 'FIND vp(out) WHEN vdb(out)=0',
        },
        'values': values,
        'phase_margin_find_out_unit': 'rad',
        'note': ('phase_margin_find_out is ngspice\'s raw .measure result for '
                 'vp(), which comes back in RADIANS -- see module docstring. '
                 'No gain_margin/phase-crossover entry: ngspice\'s vp() is '
                 'wrapped to (-pi, pi] per sample, not unwrapped across the '
                 'sweep, so a literal WHEN/FIND vp(out)=-180 recipe cannot '
                 'find a crossover that passes through the wrap boundary -- '
                 'verified live while building this generator, see module '
                 'docstring and docs/metrics-ngspice-measure-catalog.md.'),
    }


def main() -> None:
    tran_sidecar = build_tran_sidecar()
    ac_sidecar = build_ac_sidecar()
    version = _ngspice_version()
    for sidecar in (tran_sidecar, ac_sidecar):
        sidecar['generated_by'] = 'gen_e_meas.py (TASK-303.2)'
        sidecar['ngspice_version'] = version

    tran_path = HERE / 'e_ng_tran_meas.json'
    ac_path = HERE / 'e_ng_ac_meas.json'
    tran_path.write_text(json.dumps(tran_sidecar, indent=2, sort_keys=True) + '\n')
    ac_path.write_text(json.dumps(ac_sidecar, indent=2, sort_keys=True) + '\n')
    print(f'{tran_path.name}: {tran_path.stat().st_size} bytes')
    print(f'{ac_path.name}: {ac_path.stat().st_size} bytes')


if __name__ == '__main__':
    main()
