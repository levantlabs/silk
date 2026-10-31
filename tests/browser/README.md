# Browser end-to-end tests (waveform explorer)

These drive a **real emitted `view.html`** in headless Chrome with real mouse
and keyboard events, asserting on screenshot diffs. They catch interaction
bugs — zoom, the `f` fit key, the help dialog, event wiring, async state — that
the Python-side tests (which only check the emitted HTML/payload) cannot, and
that reading the source misses. See the module docstring in `view_e2e.mjs`.

They are **opt-in and auto-skip**: `tests/test_view_e2e.py` skips cleanly
unless all of Chrome, Node, and this directory's `node_modules` are present, so
a normal `pytest` run and CI without a browser are unaffected.

## One-time setup

```
cd tests/browser
npm install            # installs puppeteer-core, pixelmatch, pngjs (no bundled Chrome)
```

You also need a Chrome/Chromium binary on PATH (`google-chrome`,
`google-chrome-stable`, `chromium`, or `chromium-browser`), or set
`SILK_CHROME=/path/to/chrome`.

The source-current grouping check in `tests/test_view_source_currents.py` uses
the optional Python Playwright driver with `SILK_FIREFOX` or `SILK_CHROME`.
Select a browser build that matches the installed driver; an older cached
Firefox can reject newer driver protocol fields. The check skips when its
browser or driver is absent.

## Running

Through pytest (emits the page for you, then drives it):

```
.venv/bin/python -m pytest tests/test_view_e2e.py tests/test_view_open_raw_e2e.py -v
```

Or directly against any emitted page:

```
node tests/browser/view_e2e.mjs /path/to/view.html "$(command -v google-chrome)"
```

`node_modules/`, the lockfile, and screenshot scratch are gitignored.

`open_raw_e2e.mjs` additionally drives both producers through the real page
under `file://`, with networking disabled. It compares all decimation leaves
and uPlot data before/after zoom, exercises file input and drop, checks
same-file no-op reloads and filename-based saved layouts, and verifies that
malformed or aborted input reaches the fatal notice. The Python wrapper
builds a 65,536-point fixture (three levels, both Y dtypes, NaN/Inf gaps) and
also uses checked-in multi-plot/AC and non-ASCII fixtures. The original
`view_e2e.mjs` interaction/screenshot suite remains unchanged.


`tests/test_view_explorer.py` copies the documented assets into a bare host
page with a foreign stylesheet. Its browser checks cover two independent
mounts, both payload producers, root theming, keyboard and drop isolation,
container resizing, saved layouts, and cleanup during pending work. The picker
context menu is checked alongside outside-drop isolation: a foreign right-click
reaches the host unchanged while a selection menu is open. It also
exercises the standalone sidecar polling and scope transport. Run it with:

```bash
.venv/bin/python -m pytest -q tests/test_view_explorer.py
```

The same host smoke runs against byte-checked wheel assets in the slow profile:

```bash
.venv/bin/python -m pytest -q -m slow tests/test_view_explorer.py
```

The host test writes `explorer-host.png` in its pytest temporary directory.
A captured example is checked in at `docs/images/explorer-host.png`.

`explorer_units_e2e.mjs`, also run by `tests/test_view_explorer.py`, uses the
AC rawfile fixture to check Bode trace height in linear and dB modes,
independent axis ranges and controls, Y zoom gestures, cursor/statistics units,
annotation positions, exported CSV samples and PNG gutters, and saved-layout
compatibility. It also covers three/four axes, empty-unit groups, deterministic
placement, uniform-panel tick/gutter equivalence, and per-axis dB eligibility.
The test writes `bode.png` in its pytest temporary directory.

`explorer_derived_e2e.mjs`, run by `test_explorer_derived_signals`, checks the
standalone command and a mounted host with controlled level-0 arrays. It covers
arithmetic, unit rules and chained dimensions, deduplication, name collisions,
per-plot ownership, literal picker labels/counts, cursor/statistics and exports,
and session reset. Picker right-click regressions cover count-dependent entries,
both arithmetic directions, visible unit disabling, plotted values, focused-panel
placement, composition, deduplication, keyboard access and dismissal. Throwing
getters guard every decimated operand property;
an independent extrema reference checks the derived chart at a decimated stride.
It writes `derived-signals.png` and `derive-menu-units.png` in its pytest
temporary directory.
`explorer_derived_state_e2e.mjs`, run by `test_explorer_derived_state`, checks
fresh-page localStorage and exported view-file round trips against the same
rawfile, child-before-parent definitions, saved names, picker-only and inactive
plot definitions, reloads, stale operands, malformed/cyclic/unsupported
entries, legacy views, reset, and filename-isolated local rawfile state.
