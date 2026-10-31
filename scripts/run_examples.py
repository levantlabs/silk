"""Run every example in ``silk/examples/``, sequentially, and report.

Two jobs:

* **Verification.** The examples are silk's real integration test — they build
  modules, emit decks, run a simulator and check specs. The unit suite does not
  cover deck *emission* against a real hierarchy, so a netlister change can be
  green in pytest and broken here.
* **Deck generation.** Emitted decks land in ``silk-output/`` at the repo root
  (gitignored; the examples run with ``cwd`` there, and ``$SILK_EXAMPLES_OUT``
  moves it).

Examples run under whichever simulator each one selects — normally ngspice.

Usage
-----
::

    python scripts/run_examples.py                 # all of them
    python scripts/run_examples.py --filter ldo    # just matching names
    python scripts/run_examples.py --timeout 3600  # per-example cap

An example that fails gets its stdout and stderr written to the results
directory, so a failure is diagnosable without re-running the whole set.
Results stream to ``results.json`` as they complete, so progress survives an
interrupted run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / 'silk' / 'examples'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--filter', default='',
                    help='only run examples whose filename contains this')
    ap.add_argument('--timeout', type=float, default=2400.0,
                    help='per-example wall-clock cap in seconds')
    ap.add_argument('--out', default=str(Path.cwd() / 'examples-run'),
                    help='where to write results.json and failure logs')
    ap.add_argument('--python', default=sys.executable,
                    help='interpreter to run each example with')
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # The PDK location is the user's setting: $PDK_ROOT, else load_pdk's
    # ~/pdk fallback. Never a path baked into this script.
    env = dict(os.environ)
    env['MPLBACKEND'] = 'Agg'                 # headless: no GUI backend

    # Topic subpackages (silk/examples/metrics/) are examples too: they run
    # the same way and are the same kind of regression evidence.
    candidates = list(EXAMPLES.glob('*.py')) + [
        p for p in EXAMPLES.glob('*/*.py')
        if (p.parent / '__init__.py').exists()]     # skip output/, a run dir
    examples = sorted(
        (p for p in candidates
         if not p.name.startswith('__') and args.filter in p.name),
        key=lambda p: p.relative_to(EXAMPLES).as_posix())
    if not examples:
        print(f'no examples matching {args.filter!r} in {EXAMPLES}')
        return 1
    print(f'{len(examples)} examples, timeout {args.timeout:.0f}s each, '
          f'PDK_ROOT={env.get("PDK_ROOT", "unset (load_pdk uses ~/pdk)")}',
          flush=True)

    results, t_all = {}, time.time()
    for i, ex in enumerate(examples, 1):
        t0 = time.time()
        try:
            cp = subprocess.run([args.python, str(ex)], cwd=str(REPO), env=env,
                                capture_output=True, text=True,
                                timeout=args.timeout)
            rc, so, se = cp.returncode, cp.stdout, cp.stderr
        except subprocess.TimeoutExpired as e:
            rc = -9
            so = (e.stdout or b'').decode(errors='ignore') if e.stdout else ''
            se = f'TIMEOUT after {args.timeout}s'
        el = time.time() - t0

        # Last non-empty stderr line is nearly always the exception itself.
        lines = [x for x in (se or '').strip().splitlines() if x.strip()]
        detail = lines[-1][:400] if lines else ''
        ok = rc == 0
        results[ex.name] = {'rc': rc, 'seconds': round(el, 1),
                            'ok': ok, 'error': '' if ok else detail}
        (out / 'results.json').write_text(json.dumps(results, indent=1))
        if not ok:
            (out / f'{ex.stem}.stderr').write_text(se or '')
            (out / f'{ex.stem}.stdout').write_text(so or '')
        print(f'[{i}/{len(examples)}] {ex.name:<46} '
              f'{"ok  " if ok else "FAIL"} {el:7.1f}s  {detail[:110]}',
              flush=True)

    ok = sum(1 for r in results.values() if r['ok'])
    print(f'\n══ {ok}/{len(results)} passed in '
          f'{(time.time() - t_all) / 60:.1f} min ══')
    for name, r in sorted(results.items()):
        if not r['ok']:
            print(f'  FAIL {name} ({r["seconds"]}s): {r["error"][:200]}')
    print(f'\nresults: {out / "results.json"}')
    decks = os.environ.get('SILK_EXAMPLES_OUT') or str(REPO / 'silk-output')
    print(f'decks:   {decks}')
    return 0 if ok == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
