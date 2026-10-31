# Union-x semantics fixture

`union-x-semantics.json` is copied byte-for-byte from the MIT-licensed source fixture,
`docs/fixtures/union-x-semantics.json` at commit `6e60b6a2`. Its MIT license,
Levant Labs copyright, and hand-computed provenance are retained in the JSON.
SHA-256: `3df239bb0bbb127d96c79176cbdb9ea5dca3f929462a6dcbf76240e7e68e870f`.

The fixture records the merge semantics and independent expectations used by
the browser waveform core.

The expected arrays were computed by hand from those semantics, never from
silk output. Do not regenerate or adjust them to match the implementation.
If a test disagrees, report the failing case in the agreement as a finding
against silk's merge before changing code or expectations.

`tests/js/explorer_union.test.mjs` runs the current `explorer.mjs` Store,
`unionX`, `buildData`, and `makeSeries` numerical path in Node. Only unrelated
axis/color presentation dependencies are stubbed. The fixture already contains
decimated levels, so the test does not rerun decimation or stride selection.
The full expected arrays pin sample ownership, repeated x multiplicity,
clean-signal null holes with `spanGaps`, and interpolated merge holes with
genuine NaN/Infinity gaps left unspanned.

Run directly with `node --test tests/js/explorer_union.test.mjs`, or through
`.venv/bin/python -m pytest -q tests/test_view_js_differential.py`. The latter
is part of the fast suite and uses its existing required-Node policy (only
`SILK_ALLOW_NO_NODE=1` explicitly opts out when Node is absent).
