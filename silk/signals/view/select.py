"""silk.signals.view.select — hierarchical signal selection & block discovery.

Implements the ``hierParse`` grammar (v1.3 spec §2.2, normative) in Python so
selection can be applied at emit time, plus the selector-matching rules and
block discovery used by ``emit_view(signals=..., exclude=..., max_depth=...)``
and ``emit_views_by_block`` (see docs/waveform-explorer.md for waveform
selection).

**This module is the pin that keeps the Python parser and the frontend's JS
``hierParse`` from drifting** — ``tests/fixtures/hierparse_cases.json`` is
asserted against both sides; see ``tests/test_view_select.py``.

Import boundary (spec §3.2): stdlib, numpy, :mod:`silk.signals.rawfile`, and
sibling modules under ``silk.signals.view`` only. ``emit_views_by_block``
needs :func:`silk.signals.view.emit.emit_view`, which would create a
module-level import cycle (``emit`` imports this module to filter payloads);
that one import is therefore deferred to call time, mirroring the lazy-import
pattern already used by ``silk/circuit/report.py``'s ``add_waveform_view``.
"""
import difflib
import os
import re

from silk.signals.rawfile import RawFile, read_rawfile
from silk.signals.view.template import (
    AmbiguousSelectorError, EmptySelectionError, SelectorNotFoundError, ViewError,
)

__all__ = [
    'hier_parse', 'signal_path_segments', 'matches_selector',
    'select_for_payload', 'block_label', 'list_blocks', 'derive_stems', 'emit_views_by_block',
]

# ---------------------------------------------------------------------------
# hierParse (v1.3 spec §2.2, normative — see the worked table there)
# ---------------------------------------------------------------------------

# Step 2: unwrap a wrapper(...) call. Greedy inner group so the FINAL ")" is
# the closer (a leaf like "...#dbody)" has no interior parens in the corpus,
# but greediness is the documented, correct behaviour regardless).
_WRAPPER_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$')


def _split_at_depth0(inner):
    """Step 3: split on '.' or ':' at bracket depth 0, tracking '[' / ']'.

    Empty segments (leading or repeated separators) are dropped.
    """
    segments = []
    current = []
    depth = 0
    for ch in inner:
        if ch == '[':
            depth += 1
            current.append(ch)
        elif ch == ']':
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch in '.:' and depth == 0:
            segments.append(''.join(current))
            current = []
        else:
            current.append(ch)
    segments.append(''.join(current))
    return [s for s in segments if s != '']


def hier_parse(name, part=None):
    """Parse *name* per the v1.3 §2.2 grammar.

    Hierarchy separators are ngspice's ``.`` and Xyce's ``:`` outside
    square brackets; bracket contents and raw component spellings stay literal.

    Returns ``{'wrapper': str|None, 'segments': [str, ...], 'leaf': str}``.

    *part* mirrors a Signal's ``part`` field (``'mag'``, ``'phase'``, or
    ``None``) — step 1 of the grammar strips exactly one trailing
    ``" (mag)"``/``" (phase)"`` suffix when it applies, so this function can
    be handed either a raw (pre-expansion) rawfile vector name (``part=None``,
    the case ``select_for_payload`` always uses) or a post-expansion payload
    Signal name (matching the frontend's ``hierParse``, which is what the
    grammar-drift fixture exercises).
    """
    base = name
    if part == 'mag' and base.endswith(' (mag)'):
        base = base[:-len(' (mag)')]
    elif part == 'phase' and base.endswith(' (phase)'):
        base = base[:-len(' (phase)')]

    m = _WRAPPER_RE.match(base)
    if m:
        wrapper, inner = m.group(1), m.group(2)
    else:
        wrapper, inner = None, base

    parts = _split_at_depth0(inner)
    if not parts:
        # Degenerate (empty-string) name — no leaf, no segments.
        return {'wrapper': wrapper, 'segments': [], 'leaf': ''}
    return {'wrapper': wrapper, 'segments': parts[:-1], 'leaf': parts[-1]}


def signal_path_segments(name, part=None):
    """The full hierarchy decomposition of *name* — ``segments + [leaf]`` from
    :func:`hier_parse`, excluding the wrapper (which is a leaf badge, not a
    tree level — spec §2.2). This is what selector form 3 (hierarchy
    contiguous-run) matches against.
    """
    parsed = hier_parse(name, part=part)
    segs = list(parsed['segments'])
    if parsed['leaf']:
        segs.append(parsed['leaf'])
    return segs


# ---------------------------------------------------------------------------
# Selector matching — three forms, all case-insensitive (team-lead spec:
# ngspice lowercases, Xyce upper-cases; a user must not have to care).
# ---------------------------------------------------------------------------

_WRAPPED_HEADS = frozenset({'v', 'i', 'n', 'id'})    # the codec's wrapper grammar


def _glob_regex(pattern):
    """A selector glob: ``*`` any run, ``?`` any one character, everything else
    literal. Square brackets are NEVER a character class here: in a
    selector they mean an array index. DOTALL, so a whole-name ``*`` crosses
    separators and newlines as ``fnmatch`` did."""
    out = []
    for ch in pattern:
        out.append('.*' if ch == '*' else '.' if ch == '?' else re.escape(ch))
    return re.compile(''.join(out), re.S)


_REGEX_CACHE = {}


def _glob_match(pattern, text):
    rx = _REGEX_CACHE.get(pattern)
    if rx is None:
        if len(_REGEX_CACHE) > 4096:
            _REGEX_CACHE.clear()
        rx = _REGEX_CACHE[pattern] = _glob_regex(pattern)
    return rx.fullmatch(text) is not None


def _is_wrapped_selector(selector):
    """A RECOGNISED wrapped selector (``v(...)``, ``i(...)``, ...): it takes
    part in full-name matching only, never in hierarchy-run matching."""
    m = _WRAPPER_RE.match(selector)
    return bool(m) and m.group(1).lower() in _WRAPPED_HEADS


def _instance_reading(selector, n_components):
    """Per-component instance readings of a BARE selector, from the codec.

    The codec recognises the selector as a whole (whole-string, ASCII-or-
    non-control segments, one separator kind); an unrecognised form comes
    back unchanged and has NO instance reading, so ``xx1[0]&foo`` stays
    literal. Never calls the component primitive on arbitrary text.
    """
    from silk.spice_names import to_spice_name
    conv = to_spice_name(selector, 'auto', kind='branch', allow_glob=True)
    if conv == selector:
        return [None] * n_components
    parts = conv.split('.')
    if len(parts) != n_components or any(p == '' for p in parts):
        return [None] * n_components
    return parts


def _name_roles(name):
    """``(components, instance_flags)`` of a vector name: every segment but the
    leaf is an instance position; the leaf is one only for a branch current
    ``i(...)`` or an ``@`` address (the codec's own branch/instance cases)."""
    parsed = hier_parse(name)
    comps = list(parsed['segments'])
    flags = [True] * len(comps)
    if parsed['leaf']:
        comps.append(parsed['leaf'])
        wrapper = (parsed['wrapper'] or '').lower()
        flags.append(wrapper == 'i' or comps[-1].startswith('@')
                     or (parsed['wrapper'] is None and name.startswith('@')))
    return [c.lower() for c in comps], flags


def matches_selector(name, selector):
    """True if *selector* matches *name* by any of the three forms, all
    case-insensitive:

    1. equals the full signal name;
    2. a glob against the full name (``*`` and ``?`` are wildcards; square
       brackets are literal, never a character class);
    3. a hierarchy match -- the selector's dot-separated components appear as a
       contiguous run in the name's path components (BARE selectors only; a
       recognised wrapped selector such as ``v(out)`` matches by full name).

    An array index may be written ``[k]`` or ``<k>``: each selector is read as
    written and through the codec's instance reading (``xx1[0]`` -> ``xx1<0>``),
    the latter only where the matched component is an instance (a non-leaf
    hierarchy component, or the leaf of an ``i(...)`` name), never at a node
    leaf, so ``g[0]`` stays the net ``g[0]`` and never reaches ``g<0>``.
    """
    from silk.spice_names import to_spice_name
    name_l = name.lower()
    sel_l = selector.lower()
    wrapped_head = _WRAPPER_RE.match(name)
    role = 'branch' if wrapped_head and wrapped_head.group(1).lower() in ('i', 'id') else 'node'
    for cand in {sel_l, to_spice_name(sel_l, 'auto', kind=role, allow_glob=True)}:
        if name_l == cand or _glob_match(cand, name_l):
            return True
    if _is_wrapped_selector(selector):
        return False
    sel_segs = [s for s in sel_l.split('.') if s != '']
    if not sel_segs:
        return False
    readings = _instance_reading(sel_l, len(sel_l.split('.')))
    if len(readings) != len(sel_l.split('.')):
        readings = [None] * len(sel_segs)
    pairs = [(s, r) for s, r in zip(sel_l.split('.'), readings) if s != '']
    comps, flags = _name_roles(name)
    n = len(pairs)
    for i in range(len(comps) - n + 1):
        for k, (lit, inst) in enumerate(pairs):
            comp = comps[i + k]
            if _glob_match(lit, comp) or (flags[i + k] and inst is not None
                                          and _glob_match(inst, comp)):
                continue
            break
        else:
            return True
    return False


def block_label(path):
    """Display spelling of a block key (``xx1<0>.xxy<1>`` -> ``xx1[0].xxy[1]``);
    the key itself stays the raw spelling."""
    from silk.spice_names import designer_component
    return '.'.join(designer_component(seg) for seg in path.split('.'))


# ---------------------------------------------------------------------------
# Near-miss suggestions for the fail-loud selector-not-found error. Suggests
# against both full signal names and the individual hierarchy tokens seen
# across the file (a selector is usually a short token like "xdut", not a
# full name, so token-level suggestions are the useful ones in practice).
# ---------------------------------------------------------------------------

def _vocabulary(all_names):
    """Display vocabulary for suggestions: full names through the codec and
    instance segments in the bracket spelling (leaves untouched)."""
    from silk.spice_names import designer_component, to_designer_name
    vocab = set()
    for name in all_names:
        vocab.add(to_designer_name(name))
        parsed = hier_parse(name)
        vocab.update(designer_component(seg) for seg in parsed['segments'])
        if parsed['leaf']:
            vocab.add(designer_component(parsed['leaf'])
                      if (parsed['wrapper'] or '').lower() == 'i' else parsed['leaf'])
    return vocab


def _selector_not_found_error(selector, all_names, kwarg,
                              where='in this rawfile'):
    vocab = sorted(_vocabulary(all_names))
    matches = difflib.get_close_matches(selector, vocab, n=5)
    hint = f' Closest names/hierarchy segments: {matches}.' if matches else ''
    return SelectorNotFoundError(
        f'{kwarg}= selector {selector!r} matched no signal {where}.{hint}',
        selector=selector, kwarg=kwarg,
    )


# ---------------------------------------------------------------------------
# select_for_payload — the emit-time filter. Operates on rawfile vector
# names (pre-complex-expansion, team-lead spec: "so selecting v(out) on an
# AC file keeps BOTH its (mag) and (phase) signals"). Returns per-plot
# boolean masks (not filtered name lists) so duplicate vector names within a
# plot resolve unambiguously by position.
# ---------------------------------------------------------------------------

def select_for_payload(plot_name_lists, *, signals=None, exclude=None, max_depth=None,
                       where='in this rawfile'):
    """Compute per-plot inclusion masks for ``emit_view``'s payload filter.

    Parameters
    ----------
    plot_name_lists : list of list of str
        One list per plot, of that plot's data-vector names (vector 0, the
        independent axis, excluded — it is never filtered) in file order.
    signals, exclude : list of str, optional
        Selector lists; see :func:`matches_selector`.
    max_depth : int, optional
        Keep only signals whose ``len(hier_parse(name)['segments'])`` is
        ``<= max_depth``.

    Returns
    -------
    list of list of bool
        Same shape as *plot_name_lists* — ``True`` where the vector at that
        position is included in the final selection.

    Notes
    -----
    *where* only rewords the fail-loud errors — callers that select over an
    inventory that is not a rawfile (the simulator save list) pass their own
    phrase so a typo error does not claim a rawfile that does not exist.

    Raises
    ------
    ViewError
        A *signals*/*exclude* selector that matches no vector anywhere in
        the file (across all plots), or a final selection (after
        signals=/exclude=/max_depth=, across all plots) that is empty.
    """
    all_names = [n for names in plot_name_lists for n in names]

    if signals is None:
        stage1 = [[True] * len(names) for names in plot_name_lists]
    else:
        hit = {s: False for s in signals}
        stage1 = []
        for names in plot_name_lists:
            mask = []
            for name in names:
                ok = False
                # No early exit: EVERY selector matching this name must be
                # credited in `hit`, or a selector whose match set is a
                # subset of an earlier one (e.g. ['*', 'v(out)']) looks
                # unmatched below and trips the fail-loud check.
                for sel in signals:
                    if matches_selector(name, sel):
                        hit[sel] = True
                        ok = True
                mask.append(ok)
            stage1.append(mask)
        for sel, ok in hit.items():
            if not ok:
                raise _selector_not_found_error(sel, all_names, 'signals', where)

    if exclude:
        exhit = {s: False for s in exclude}
        stage2 = []
        for names, mask in zip(plot_name_lists, stage1):
            new_mask = []
            for name, keep in zip(names, mask):
                # Same "credit every match" rule as the signals= stage above
                # (the loop continues after keep goes False), and matching is
                # tried on every name rather than only the kept ones: the
                # fail-loud check below is about a selector matching nothing
                # *in the file*, and a name that signals= already dropped is
                # still in the file.
                for sel in exclude:
                    if matches_selector(name, sel):
                        exhit[sel] = True
                        keep = False
                new_mask.append(keep)
            stage2.append(new_mask)
        for sel, ok in exhit.items():
            if not ok:
                raise _selector_not_found_error(sel, all_names, 'exclude', where)
    else:
        stage2 = stage1

    if max_depth is not None:
        stage3 = []
        for names, mask in zip(plot_name_lists, stage2):
            new_mask = [
                keep and len(hier_parse(name)['segments']) <= max_depth
                for name, keep in zip(names, mask)
            ]
            stage3.append(new_mask)
    else:
        stage3 = stage2

    if not any(any(mask) for mask in stage3):
        raise EmptySelectionError(
            f'signal selection (signals=/exclude=/max_depth=) matched zero '
            f'signals {where} — nothing would be emitted'
        )
    return stage3


# ---------------------------------------------------------------------------
# Discover hierarchy blocks in a rawfile for emitting one page per block.
# ---------------------------------------------------------------------------

def _read(source):
    if source is None:
        # Same contract as emit._resolve_source: tb.results_file is None
        # until the testbench has been simulated.
        raise ViewError(
            'rawfile source is None — if this came from tb.results_file / '
            'module.results_file, the testbench has not been simulated yet; '
            'run it first (tb.run() / module.simulate()).'
        )
    if isinstance(source, RawFile):
        return source
    return read_rawfile(os.fspath(source))


def list_blocks(source, *, depth=1):
    """Discover hierarchy blocks across the whole rawfile at *depth*.

    Returns ``(blocks, n_top_level)``:

    - *blocks* — an ordered list of ``(path, n_signals)``, *path* the
      dot-joined block key, in first-appearance order across the file
      (every plot's data vectors, never the independent axis).
    - *n_top_level* — the count of depth-0 (bare, no dotted hierarchy)
      signals; these never appear in any block, regardless of *depth*.

    A signal whose hierarchy is shallower than *depth* (e.g. ``v(xdut.out)``,
    one segment, with ``depth=2`` requested) contributes the key it has
    (``segments[:depth]``, which is just ``segments`` when shorter) rather
    than being dropped. ``depth <= 0`` discovers nothing (every non-bare
    signal is simply absent from *blocks*); the everyday case is the default
    ``depth=1``.

    **Counts use the same semantics as the page each block would produce.**
    Keys are *discovered* from leading segments, but counted with
    :func:`matches_selector` — exactly what :func:`emit_views_by_block` passes
    as ``signals=[path]`` — so a reported count is what that block's
    ``view.html`` will actually contain. The consequence, which the caller
    must present honestly: **blocks overlap and their counts do not sum to
    the signal total.** ngspice prefixes device internals with a device class,
    so ``m.xdut.xmm1.…`` lies inside both the ``m`` bucket and the ``xdut``
    block, and is counted in both. Counting a clean partition instead would
    report a number (e.g. ``xdut 6``) that the corresponding page (45 signals)
    visibly contradicts.

    Counts are of rawfile *vectors* across every plot. On a complex (AC) plot
    each vector becomes two signals on the page (magnitude + phase), and a
    page only renders its default plot, so for those files the count is a
    lower bound rather than an exact page total.
    """
    raw = _read(source)
    order = []
    seen = set()
    names = []
    n_top = 0
    for plot in raw.plots:
        for vec in plot.vectors[1:]:
            names.append(vec.name)
            segs = hier_parse(vec.name)['segments']
            if not segs:
                n_top += 1
                continue
            if depth < 1:
                continue
            key = '.'.join(segs[:depth])
            # Block matching (matches_selector) is case-insensitive, so
            # discovery must dedupe the same way — otherwise xdut.* and
            # XDUT.* list as two blocks with identical membership. First
            # appearance in the file wins the display spelling.
            key_fold = key.casefold()
            if key_fold not in seen:
                seen.add(key_fold)
                order.append(key)
    blocks = [(k, sum(1 for n in names if matches_selector(n, k))) for k in order]
    return blocks, n_top


_SANITIZE_RE = re.compile(r'[^A-Za-z0-9_-]+')

# The stem ``top.html`` (the depth-0 page) claims. Reserved unconditionally,
# not just when that page is actually emitted, so a given block path maps to
# the same stem whether or not the rawfile happens to have bare signals.
_RESERVED_STEMS = frozenset({'top'})


def _sanitize(path):
    return _SANITIZE_RE.sub('_', path) or 'block'


def derive_stems(paths):
    """Map each block path in *paths* to a **unique** filename stem.

    :func:`_sanitize` is many-to-one — every run of non-``[A-Za-z0-9_-]``
    collapses to one ``_``, so ``x.y_z`` and ``x_y.z`` both reduce to
    ``x_y_z`` — which makes the bare sanitized name unusable as an identity.
    Two blocks sharing one stem broke both emission modes: the split emit
    died with ``ViewError: scopes[1] has duplicate key 'x_y_z'`` (blaming the
    manifest for a filename problem), and ``inline=True``, which builds no
    manifest, silently wrote the second page *over* the first while printing
    two "wrote" lines.

    Colliding paths are ordered by path — not by rawfile discovery order, so
    the assignment does not shift when a re-run's vector order changes — and
    every one after the first takes ``-2``, ``-3``, … , bumped past any stem
    already claimed (a literal ``x_y_z-2`` block elsewhere in the file, or
    the reserved ``top``).
    """
    groups = {}
    for path in paths:
        groups.setdefault(_sanitize(path), []).append(path)
    stems = {}
    # Casefolded so 'Xdut' and 'XDUT' — distinct as Python strings but the
    # same file on a case-insensitive filesystem (macOS/Windows) — cannot
    # both be handed out as stems.
    taken = {s.casefold() for s in _RESERVED_STEMS}
    for base in sorted(groups):
        for i, path in enumerate(sorted(groups[base])):
            n = i + 1
            candidate = base if n == 1 else f'{base}-{n}'
            while candidate.casefold() in taken:
                n += 1
                candidate = f'{base}-{n}'
            taken.add(candidate.casefold())
            stems[path] = candidate
    return stems


def emit_views_by_block(source, out_dir, *, depth=1, title_prefix=None, **emit_kwargs):
    """Emit one page per hierarchy block at *depth*, plus ``top.html`` for
    the depth-0 (bare) signals, all into *out_dir*.

    Each block page is emitted with ``signals=[path]`` — the hierarchy-match
    form (selector form 3) selects every signal whose path contains that
    block's segments as a contiguous run, i.e. every descendant. Because
    sidecar filenames are now derived from the HTML stem (feature 1), all
    pages coexist in *out_dir* without clobbering each other's ``data.js``.
    Stems come from :func:`derive_stems`, so two block paths that sanitize
    alike (``x.y_z`` and ``x_y.z``) still get a page each.

    ``top.html`` is only emitted when there is at least one depth-0 signal —
    emitting it unconditionally would, on a file with none, hand
    ``emit_view`` a ``max_depth=0`` selection that is guaranteed empty and
    therefore guaranteed to raise (the same fail-loud rule
    :func:`select_for_payload` applies everywhere else).

    Every emitted page also carries the ``SILK_VIEW_SCOPES`` manifest
    (view-context-and-scopes design §3.2) for the whole set, so the
    frontend's ``Scope ▾`` control can switch between them without a page
    reload. Built in a single pass from :func:`list_blocks`'s own counts
    (rather than a second emit pass to discover exact post-encoding
    ``n_signals``) — those counts already use
    :func:`~silk.signals.view.select.matches_selector`, i.e. exactly what
    each block's ``signals=[path]`` selection below will match, so on the
    common single-plot rawfile they are exactly what the page will contain.
    Two documented divergences, inherited from :func:`list_blocks` (not
    introduced here): a *lower bound* on an AC file, since a complex vector
    expands to two signals post-encoding; and, on a genuinely **multi-plot**
    rawfile, an *over-count*, since :func:`list_blocks` sums a block's
    vectors across every plot while a page only ever renders its one
    default plot (see :func:`~silk.signals.view.pick_default_plot`).
    Both are the same "counts are what the CLI table already shows, not a
    second, silently-more-precise number" tradeoff `--list-blocks` already
    makes. Skipped entirely under ``inline=True``, where a manifest would be
    meaningless (no sibling ``data.js`` to switch to — ``emit_view`` rejects
    the combination).

    ``**emit_kwargs`` are forwarded verbatim to every
    :func:`~silk.signals.view.emit.emit_view` call, so ``signals=`` and
    ``max_depth=`` cannot be passed here — each page already carries the
    selection its own block implies (``signals=[path]``, or ``max_depth=0``
    for ``top.html``) and a second one would collide with it. The CLI rejects
    ``--split-by-block`` combined with ``--signals``/``--exclude``/
    ``--max-depth`` up front for the same reason.

    Returns a list of ``ViewOutput``, one per emitted page, blocks first (in
    :func:`list_blocks` order) then ``top.html`` last.
    """
    # Lazy import: emit.py imports this module (to filter payloads via
    # select_for_payload), so importing emit_view at module level here would
    # be a cycle. Deferred to call time, mirroring silk/circuit/report.py's
    # add_waveform_view.
    from silk.signals.view.emit import emit_view

    blocks, n_top = list_blocks(source, depth=depth)
    os.makedirs(out_dir, exist_ok=True)

    # One stem per block path, unique by construction (see derive_stems) —
    # manifest keys, HTML filenames and sidecar names all come from this map,
    # so they cannot disagree with each other.
    stems = derive_stems([path for path, _n in blocks])

    scope_entries = []
    for path, n in blocks:
        stem = stems[path]
        scope_entries.append({
            'key': stem, 'label': block_label(path), 'html': f'{stem}.html',
            'data_src': f'{stem}.data.js', 'stamp_src': f'{stem}.stamp.js',
            'n_signals': n,
        })
    if n_top:
        scope_entries.append({
            'key': 'top', 'label': 'top-level', 'html': 'top.html',
            'data_src': 'top.data.js', 'stamp_src': 'top.stamp.js',
            'n_signals': n_top,
        })
    scopes = scope_entries if scope_entries and not emit_kwargs.get('inline') else None

    outputs = []
    for path, _n in blocks:
        out_path = os.path.join(out_dir, f'{stems[path]}.html')
        shown = block_label(path)
        title = f'{title_prefix} — {shown}' if title_prefix else shown
        outputs.append(emit_view(source, out_path, title=title, signals=[path],
                                 scopes=scopes, **emit_kwargs))
    if n_top:
        out_path = os.path.join(out_dir, 'top.html')
        title = f'{title_prefix} — top-level' if title_prefix else 'top-level'
        outputs.append(emit_view(source, out_path, title=title, max_depth=0,
                                 scopes=scopes, **emit_kwargs))
    return outputs
