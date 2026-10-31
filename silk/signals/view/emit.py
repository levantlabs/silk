"""silk.signals.view.emit — payload assembly, run identity, annotations.

The Python-side heart of the waveform explorer: turns a nutmeg rawfile into
the ``data.js`` payload (spec §4) and, via :func:`emit_view`, writes the
three-file (or single-file, ``inline=True``) output described in spec §5.1.

Import boundary (spec §3.2): this module may import the standard library,
numpy, :mod:`silk.signals.rawfile`, the shared stdlib-only
:mod:`silk._json_codec`, and sibling ``silk.signals.view`` modules. In particular, the silk version string
is read via :mod:`importlib.metadata` (stdlib), not ``from silk import
__version__``, precisely to keep that promise.
"""
import difflib
import hashlib
import html as _html
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from silk._json_codec import encode_json
from silk.signals.rawfile import RawFile, RawfileError, read_rawfile
from silk.signals.view.encode import (
    DECIMATE_BASE, DECIMATE_MIN_BUCKET, DECIMATE_MIN_POINTS,
    build_data_signal, build_x_signal, decide_y_dtype, decimation_strides,
)
from silk.signals.view.select import select_for_payload
from silk.signals.view.serialize import build_data_js, build_stamp_js, to_json
from silk.signals.view.template import ViewError, render_view_html

__all__ = [
    'ViewError', 'ViewOutput', 'DATA_FORMAT_VERSION',
    'emit_view', 'emit_view_data', 'pick_default_plot',
]

DATA_FORMAT_VERSION = 2

_COLOR_RE = re.compile(r'^#[0-9a-fA-F]{6}$')

# ``ViewError`` lives in :mod:`silk.signals.view.template` (v1.3 spec §3.6) so
# that module can raise it too without an import cycle back to this one; it
# is re-exported here (imported above) so ``from
# silk.signals.view.emit import ViewError`` — the pre-v1.3 import path —
# keeps working unchanged.


@dataclass(frozen=True)
class ViewOutput:
    """Return value of :func:`emit_view`.

    ``n_signals``/``n_points`` describe the **default plot** (the one the
    explorer opens showing) — the number that matters to someone reading the
    one-line CLI summary. ``n_plots`` is the total plot count in the file.

    ``view_state_path`` (v1.3 spec §3.6) is the absolute path of the
    view-state sidecar that was embedded, or ``None`` when no state was
    embedded (including when ``view_state=`` was passed as a ``dict``, which
    has no path). Additive field; frozen dataclass.

    ``n_signals_total`` is the **default plot**'s post-expansion signal
    count *before* ``signals=``/``exclude=``/``max_depth=`` filtering —
    equal to ``n_signals`` when no filter was given. Additive field.
    """
    html_path: str
    data_path: Optional[str]
    stamp_path: Optional[str]
    run_id: str
    state_key: str
    n_plots: int
    n_signals: int
    n_points: int
    view_state_path: Optional[str] = None
    n_signals_total: int = 0


# ---------------------------------------------------------------------------
# silk version (stdlib-only — see module docstring)
# ---------------------------------------------------------------------------

def _silk_version():
    try:
        import importlib.metadata
        return importlib.metadata.version('circuitsilk')
    except Exception:
        return 'unknown'


def _now_iso():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------

def _resolve_source(source):
    """Return ``(RawFile, source_label)``. *source_label* is the path as the
    caller gave it (or the RawFile's own recorded path), for display only."""
    if source is None:
        # The documented habit is emit_view(tb.results_file, ...) /
        # emit_view(module.results_file, ...); results_file is None by
        # contract before the testbench/module has been simulated. Without
        # this check os.fspath(None) raises a raw, unhelpful TypeError.
        raise ViewError(
            'rawfile source is None — if this came from tb.results_file / '
            'module.results_file, the testbench has not been simulated yet; '
            'run it first (tb.run() / module.simulate()).'
        )
    if isinstance(source, RawFile):
        label = str(source.source_path) if getattr(source, 'source_path', None) else ''
        return source, label
    label = os.fspath(source)
    raw = read_rawfile(label)
    return raw, label


# ---------------------------------------------------------------------------
# Plot kind inference (spec §4.4)
# ---------------------------------------------------------------------------

def _infer_kind(plotname, x_name, x_type):
    pn = (plotname or '').lower()
    xn = (x_name or '').lower()
    xt = (x_type or '').lower()
    if 'constants' in pn:
        return 'constants'
    if 'noise' in pn:
        return 'noise'
    if 'ac analysis' in pn or xn == 'frequency' or xt == 'frequency':
        return 'ac'
    if 'transient' in pn or xn == 'time' or xt == 'time':
        return 'tran'
    if ('dc transfer' in pn or 'dc analysis' in pn
            or xn in ('sweep', 'v-sweep', 'temp-sweep') or xn.startswith('v(')):
        return 'dc'
    return 'other'


def pick_default_plot(plots):
    """Return the index of the plot the explorer opens showing.

    ``plots`` is a :class:`~silk.signals.rawfile.RawFile` or an ordered
    sequence of plot mappings with ``kind`` and ``signals`` fields (such as
    ``emit_view_data(source)['plots']``). A RawFile uses the viewer's
    unfiltered plot planning, including operating-point signals. To reflect
    a signal selection, pass the filtered payload's plot mappings.
    The input order is preserved. Empty input returns 0.

    Preference order, first hit wins, each scanned last-first:

    1. a non-constants plot that still has at least one signal;
    2. any plot that still has at least one signal (only ``constants``
       survived the filter);
    3. a non-constants plot (nothing anywhere has signals);
    4. the last plot.

    Steps 1-2 are what keeps a *selection* from opening a blank page: a
    ``signals=``/``exclude=``/``max_depth=`` filter is applied per plot, so
    on a multi-plot file the last non-constants plot can legitimately end up
    with zero signals while an earlier plot holds the whole selection
    (``_plan_plot`` runs the mask). Picking it anyway would have emitted
    a valid, exit-0 page that opens showing nothing.
    """
    plots_plan = ([_plan_plot(p) for p in plots.plots]
                  if isinstance(plots, RawFile) else plots)

    def _last_matching(pred):
        for i in range(len(plots_plan) - 1, -1, -1):
            if pred(plots_plan[i]):
                return i
        return None

    for pred in (lambda p: p['kind'] != 'constants' and p['signals'],
                 lambda p: bool(p['signals']),
                 lambda p: p['kind'] != 'constants'):
        idx = _last_matching(pred)
        if idx is not None:
            return idx
    return len(plots_plan) - 1 if plots_plan else 0


def _unwrap_phase_degrees(z):
    """Unwrapped phase of complex *z*, in degrees, NaN-tolerant.

    ``np.unwrap`` works on differences, so a single non-finite sample (one
    frequency point the simulator did not converge at) propagates NaN
    through every *later* sample — one bad point would blank the rest of the
    trace and make the page's nonfinite badge over-report by an arbitrary
    amount. Unwrapping only the finite subsequence and scattering it back
    leaves exactly the bad samples NaN while keeping the unwrap continuous
    across the gap.
    """
    angle = np.angle(z).astype(np.float64)
    finite = np.isfinite(angle)
    if finite.all():
        return np.degrees(np.unwrap(angle))
    out = np.full(angle.shape, np.nan, dtype=np.float64)
    if finite.any():
        out[finite] = np.degrees(np.unwrap(angle[finite]))
    return out


def _dedup_name(name, seen):
    """First occurrence keeps *name* unchanged; later ones get '#2', '#3', ...

    *seen* maps every name handed out so far to its occurrence count, and a
    key being present is what "this name is taken" means. The suffix keeps
    bumping until the candidate is free, because a '#'-bearing vector name is
    not hypothetical — ngspice emits ``v(m.xm1.…#dbody)`` — so the generated
    ``v(a)#2`` can land on a vector the file already calls ``v(a)#2``. A
    duplicate that survived here would silently mis-address everything keyed
    on the name: annotation targeting, ``addSignal`` commands, view-state
    pruning.

    Returns ``(final_name, renamed)``.
    """
    count = seen.get(name, 0) + 1
    if count == 1:
        seen[name] = 1
        return name, False
    candidate = f'{name}#{count}'
    while candidate in seen:
        count += 1
        candidate = f'{name}#{count}'
    seen[name] = count
    seen[candidate] = 1
    return candidate, True


def _plot_kind(plot):
    """Display shape; public analysis selection remains header-first."""
    # Mirror the reader's shape rule without widening the standalone view
    # package's import boundary; the JS producer uses the same rule.
    name = (plot.plotname or '').lower()
    if 'constants' in name:
        return 'constants'
    if not plot.real:
        return 'ac'
    if plot.vectors and name == 'operating point':
        return 'op'
    first = plot.vectors[0] if plot.vectors else None
    return _infer_kind(plot.plotname, first.name if first else '',
                       first.type if first else '')


def _signal_start(plot):
    return 0 if _plot_kind(plot) == 'op' else 1


def _plan_plot(plot, keep_mask=None):
    """Pre-encoding plan for one RawPlot: kind, x array, and the ordered list
    of (post complex-expansion, post dedup) signal specs.

    *keep_mask*, when given, is a list of bool the same length as
    the data rows (all rows for op, rows 1 onward otherwise; applied per
    the design note on ``emit_view(signals=...)``): a vector at index *i* is
    skipped entirely — never complex-expanded, never dedup-counted — when
    ``keep_mask[i]`` is False. An original independent axis is never filtered;
    op has only a synthetic sample axis, so every source row is selectable.
    """
    vectors = plot.vectors
    start = _signal_start(plot)
    kind = _plot_kind(plot)
    if kind == 'op':
        x_arr = np.arange(plot.npoints, dtype=np.float64)
        x_name, x_type = 'sample', ''
    elif vectors:
        x_vec = vectors[0]
        x_arr = np.asarray(x_vec.data, dtype=np.float64)
        x_name, x_type = x_vec.name, x_vec.type
    else:
        x_arr = np.empty(0, dtype=np.float64)
        x_name, x_type = '', ''

    complex_flag = not plot.real

    signals = []
    seen = {}
    for i, vec in enumerate(vectors[start:]):
        if keep_mask is not None and not keep_mask[i]:
            continue
        if complex_flag:
            z = np.asarray(vec.data)
            mag = np.abs(z).astype(np.float64)
            phase = _unwrap_phase_degrees(z)
            for part, arr in (('mag', mag), ('phase', phase)):
                base_name = f'{vec.name} ({part})'
                name, renamed = _dedup_name(base_name, seen)
                signals.append({
                    'name': name, 'type': vec.type, 'part': part,
                    'group': vec.name, 'y_arr': arr, 'renamed': renamed,
                    'raw': vec.name, 'row': i + start,
                })
        else:
            arr = np.asarray(vec.data, dtype=np.float64)
            name, renamed = _dedup_name(vec.name, seen)
            signals.append({
                'name': name, 'type': vec.type, 'part': None,
                'group': None, 'y_arr': arr, 'renamed': renamed,
                'raw': vec.name, 'row': i + start,
            })

    if kind == 'ac' and x_arr.size > 0:
        x_log_default = bool(np.all(x_arr > 0))
    else:
        x_log_default = False

    return {
        'plotname': plot.plotname, 'kind': kind, 'complex': complex_flag,
        'npoints': plot.npoints, 'x_arr': x_arr, 'x_name': x_name,
        'x_type': x_type, 'signals': signals,
        'x_log_default': x_log_default, 'date': plot.date,
    }


def _plan_all(raw, masks=None):
    if masks is None:
        plots_plan = [_plan_plot(p) for p in raw.plots]
    else:
        plots_plan = [_plan_plot(p, keep_mask=m) for p, m in zip(raw.plots, masks)]
    return plots_plan, pick_default_plot(plots_plan)


def _total_signals_for_plot(plot):
    """Post-expansion signal count for *plot* with no filtering applied —
    what ``n_signals_total`` reports. Cheap: counts vectors, does not touch
    the data (complex vectors expand to 2 signals each, spec §4.6)."""
    n_vectors = max(len(plot.vectors) - _signal_start(plot), 0)
    return n_vectors * 2 if not plot.real else n_vectors


def _resolve_selection(raw, signals, exclude, max_depth):
    """Returns ``(masks_or_None, n_signals_total_by_plot)``.

    *masks* is ``None`` — meaning "no filtering, use every vector" — when
    *signals*, *exclude* and *max_depth* are all unset (the common,
    zero-overhead default path per v1.4 §2: "signals=None -> every signal,
    today's behaviour unchanged"). Otherwise it is
    :func:`~silk.signals.view.select.select_for_payload`'s per-plot boolean
    masks, computed over raw (pre-complex-expansion) vector names.
    """
    n_signals_total_by_plot = [_total_signals_for_plot(p) for p in raw.plots]
    if signals is None and exclude is None and max_depth is None:
        return None, n_signals_total_by_plot
    plot_name_lists = [[v.name for v in p.vectors[_signal_start(p):]] for p in raw.plots]
    masks = select_for_payload(
        plot_name_lists, signals=signals, exclude=exclude, max_depth=max_depth,
    )
    return masks, n_signals_total_by_plot


# ---------------------------------------------------------------------------
# Annotations (spec §6)
# ---------------------------------------------------------------------------

_ANNOTATION_TOP_KEYS = frozenset({'markers', 'bands', 'notes'})
_SEVERITIES = frozenset({'info', 'warn', 'error'})


def _closest_matches(name, valid_names, n=3):
    """Return ``(matches, is_fallback)``.

    ``valid_names`` is a *set* (hash-ordered, not insertion-ordered), so
    everything here works off ``sorted(valid_names)`` — never the set's own
    iteration order — to stay deterministic across runs/hash seeds.

    Casefolded comparison so a pure-case mismatch (the normal post-G16
    case — silk-side 'V(OUT)' vs ngspice's 'v(out)') still scores as a
    match instead of falling below difflib's cutoff; original spellings are
    what get shown back to the caller. If difflib finds nothing at all, a
    hint is still owed — ``is_fallback=True`` marks the fallback list as
    *not* close matches, so a caller must not label them as such.
    """
    sorted_names = sorted(valid_names)
    folded = {}
    for v in sorted_names:
        folded.setdefault(v.casefold(), v)
    hits = difflib.get_close_matches(name.casefold(), sorted(folded), n=n)
    if hits:
        return [folded[h] for h in hits], False
    return sorted_names[:n], True


def _require_finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ViewError(f'{label} must be a finite number, got {value!r}')
    value = float(value)
    if not math.isfinite(value):
        raise ViewError(f'{label} must be a finite number, got {value!r}')
    return value


def _require_str(value, label, *, allow_none=False, default=None):
    if value is None:
        if allow_none:
            return default
        raise ViewError(f'{label} is required')
    if not isinstance(value, str):
        raise ViewError(f'{label} must be a string, got {type(value).__name__}')
    return value


def _check_signal(name, valid_names, label):
    if name is None:
        return None
    if name in valid_names:
        return name
    # Every selection path is case-insensitive by design (select.py's
    # matches_selector); annotations must be too. Resolve to the payload's
    # actual spelling so the frontend (which matches on exact string) finds
    # it — the annotation's own casing is not what gets embedded. Grouped via
    # sorted(valid_names) (a set — hash-ordered) rather than a first-write-
    # wins dict built by iterating it: iteration order is not deterministic,
    # so that would silently resolve to a different signal across runs
    # whenever a rawfile has two names differing only by case.
    candidates = sorted(v for v in valid_names if v.casefold() == name.casefold())
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # A rawfile with e.g. both 'v(out)' and 'V(OUT)' makes
        # case-insensitive resolution genuinely ambiguous — guessing would
        # be a silent-wrong-data path. Fail loud and ask for the exact name.
        raise ViewError(
            f'{label} {name!r} matches more than one signal in this rawfile '
            f'by case-insensitive comparison: {candidates}. Spell it exactly.'
        )
    matches, is_fallback = _closest_matches(name, valid_names)
    if not matches:
        hint = ''
    elif is_fallback:
        hint = f' No close match; available signals include {matches}.'
    else:
        hint = f' Closest matches: {matches}.'
    raise ViewError(
        f'{label} {name!r} does not match any signal in this rawfile.{hint}'
    )


def _normalize_marker(m, valid_names):
    if not isinstance(m, dict):
        raise ViewError(f'marker must be a dict, got {type(m).__name__}')
    axis = m.get('axis')
    if axis not in ('x', 'y'):
        raise ViewError(f"marker 'axis' must be 'x' or 'y', got {axis!r}")
    if 'value' not in m:
        raise ViewError("marker missing required 'value'")
    value = _require_finite(m['value'], "marker 'value'")
    label = _require_str(m.get('label'), "marker 'label'", allow_none=True, default='')
    signal = m.get('signal')
    if signal is not None:
        signal = _require_str(signal, "marker 'signal'")
        signal = _check_signal(signal, valid_names, "marker 'signal'")
    color = m.get('color')
    if color is not None and not (isinstance(color, str) and _COLOR_RE.match(color)):
        raise ViewError(
            f"marker 'color' must be a '#rrggbb' string or None, got {color!r}"
        )
    return {'axis': axis, 'value': value, 'label': label, 'signal': signal, 'color': color}


def _normalize_band(b, valid_names):
    if not isinstance(b, dict):
        raise ViewError(f'band must be a dict, got {type(b).__name__}')
    axis = b.get('axis')
    if axis not in ('x', 'y'):
        raise ViewError(f"band 'axis' must be 'x' or 'y', got {axis!r}")
    if 'lo' not in b or 'hi' not in b:
        raise ViewError("band missing required 'lo'/'hi'")
    lo = _require_finite(b['lo'], "band 'lo'")
    hi = _require_finite(b['hi'], "band 'hi'")
    if not lo < hi:
        raise ViewError(f"band 'lo' must be < 'hi', got lo={lo!r} hi={hi!r}")
    label = _require_str(b.get('label'), "band 'label'", allow_none=True, default='')
    signal = b.get('signal')
    if signal is not None:
        signal = _require_str(signal, "band 'signal'")
        signal = _check_signal(signal, valid_names, "band 'signal'")
    return {'axis': axis, 'lo': lo, 'hi': hi, 'label': label, 'signal': signal}


def _normalize_note(n, valid_names):
    if not isinstance(n, dict):
        raise ViewError(f'note must be a dict, got {type(n).__name__}')
    signal = _require_str(n.get('signal'), "note 'signal'")
    signal = _check_signal(signal, valid_names, "note 'signal'")
    text = _require_str(n.get('text'), "note 'text'")
    severity = n.get('severity', 'info')
    if severity not in _SEVERITIES:
        raise ViewError(
            f"note 'severity' must be one of {sorted(_SEVERITIES)}, got {severity!r}"
        )
    return {'signal': signal, 'text': text, 'severity': severity}


def _validate_annotations(annotations, valid_names):
    if not annotations:
        return {'markers': [], 'bands': [], 'notes': []}
    if not isinstance(annotations, dict):
        raise ViewError(f'annotations must be a dict, got {type(annotations).__name__}')
    unknown = set(annotations) - _ANNOTATION_TOP_KEYS
    if unknown:
        raise ViewError(
            f'unknown annotations key(s) {sorted(unknown)}; '
            f'accepted keys are {sorted(_ANNOTATION_TOP_KEYS)}'
        )
    return {
        'markers': [_normalize_marker(m, valid_names) for m in annotations.get('markers', [])],
        'bands': [_normalize_band(b, valid_names) for b in annotations.get('bands', [])],
        'notes': [_normalize_note(n, valid_names) for n in annotations.get('notes', [])],
    }


# ---------------------------------------------------------------------------
# Provenance context (view-context-and-scopes design §2.1) — shallow,
# fail-loud validation exactly like annotations above. Unlike annotations,
# an absent/empty context produces NO 'context' key in the payload at all
# (§2.1: "an optional top-level context key") rather than an always-present
# normalized-empty shape — a falsy `context` (None or {}) is the "no context"
# case, tested explicitly in test_view_context.py.
# ---------------------------------------------------------------------------

_CONTEXT_TOP_KEYS = frozenset({'title', 'subtitle', 'fields'})
_CONTEXT_FIELD_KEYS = frozenset({'label', 'value', 'severity'})
MAX_CONTEXT_FIELDS = 64
MAX_CONTEXT_BYTES = 4096


def _normalize_context_field(f, i):
    if not isinstance(f, dict):
        raise ViewError(f'context fields[{i}] must be a dict, got {type(f).__name__}')
    unknown = set(f) - _CONTEXT_FIELD_KEYS
    if unknown:
        raise ViewError(
            f'context fields[{i}] has unknown key(s) {sorted(unknown)}; '
            f'accepted keys are {sorted(_CONTEXT_FIELD_KEYS)}'
        )
    if 'label' not in f:
        raise ViewError(f"context fields[{i}] is missing required 'label'")
    label = f['label']
    if not isinstance(label, str):
        raise ViewError(
            f"context fields[{i}] 'label' must be a string, got {type(label).__name__}"
        )
    if 'value' not in f:
        raise ViewError(f"context fields[{i}] is missing required 'value'")
    value = f['value']
    if not isinstance(value, str):
        value = str(value)  # numbers etc. are common caller values (spec §2.1)
    severity = f.get('severity', 'info')
    if severity not in _SEVERITIES:
        raise ViewError(
            f"context fields[{i}] 'severity' must be one of {sorted(_SEVERITIES)}, "
            f'got {severity!r}'
        )
    return {'label': label, 'value': value, 'severity': severity}


def _validate_context(context):
    """Validate+normalize a §2.1 ``context`` dict.

    Returns ``None`` for a falsy *context* (``None`` or ``{}``) — the "no
    context key in the payload" case — otherwise a fully-normalized dict
    with ``title``/``subtitle`` (``None`` when absent) and ``fields`` (a
    list, ``[]`` when absent) always present, so the frontend parses exactly
    one shape.
    """
    if not context:
        return None
    if not isinstance(context, dict):
        raise ViewError(f'context must be a dict, got {type(context).__name__}')
    unknown = set(context) - _CONTEXT_TOP_KEYS
    if unknown:
        raise ViewError(
            f'unknown context key(s) {sorted(unknown)}; '
            f'accepted keys are {sorted(_CONTEXT_TOP_KEYS)}'
        )
    title = context.get('title')
    if title is not None and not isinstance(title, str):
        raise ViewError(f"context 'title' must be a string, got {type(title).__name__}")
    subtitle = context.get('subtitle')
    if subtitle is not None and not isinstance(subtitle, str):
        raise ViewError(f"context 'subtitle' must be a string, got {type(subtitle).__name__}")
    raw_fields = context.get('fields', [])
    if not isinstance(raw_fields, list):
        raise ViewError(f"context 'fields' must be a list, got {type(raw_fields).__name__}")
    if len(raw_fields) > MAX_CONTEXT_FIELDS:
        raise ViewError(
            f'context has {len(raw_fields)} fields, exceeding the cap of '
            f'{MAX_CONTEXT_FIELDS} fields (design spec §6 R4)'
        )
    fields = [_normalize_context_field(f, i) for i, f in enumerate(raw_fields)]
    normalized = {'title': title, 'subtitle': subtitle, 'fields': fields}
    size = len(json.dumps(encode_json(normalized, finite_only=True), allow_nan=False).encode('utf-8'))
    if size > MAX_CONTEXT_BYTES:
        raise ViewError(
            f'context serializes to {size:,} bytes, exceeding the cap of '
            f'{MAX_CONTEXT_BYTES:,} bytes (design spec §6 R4) — trim fields '
            f'or shorten values'
        )
    return normalized


# ---------------------------------------------------------------------------
# Scope manifest (view-context-and-scopes design §3.2) — structural
# validation of a hand-built ``scopes=`` list on ``emit_view``. Paths are
# relative basenames only (a file:// page must not carry absolute paths that
# break when the directory moves); 'current' is resolved by matching each
# scope's 'html' against this emission's own out_path basename, so the
# caller never has to say which entry is "this page".
# ---------------------------------------------------------------------------

_SCOPE_KEYS = frozenset({'key', 'label', 'html', 'data_src', 'stamp_src', 'n_signals'})
_SCOPE_REQUIRED_KEYS = frozenset({'key', 'html', 'data_src', 'stamp_src', 'n_signals'})
_SCOPE_PATH_FIELDS = ('html', 'data_src', 'stamp_src')


def _validate_scope_entry(s, i):
    if not isinstance(s, dict):
        raise ViewError(f'scopes[{i}] must be a dict, got {type(s).__name__}')
    unknown = set(s) - _SCOPE_KEYS
    if unknown:
        raise ViewError(
            f'scopes[{i}] has unknown key(s) {sorted(unknown)}; '
            f'accepted keys are {sorted(_SCOPE_KEYS)}'
        )
    missing = _SCOPE_REQUIRED_KEYS - set(s)
    if missing:
        raise ViewError(f'scopes[{i}] is missing required key(s) {sorted(missing)}')
    key = s['key']
    if not isinstance(key, str) or not key:
        raise ViewError(f"scopes[{i}] 'key' must be a non-empty string, got {key!r}")
    label = s.get('label', key)
    if not isinstance(label, str):
        raise ViewError(f"scopes[{i}] 'label' must be a string, got {type(label).__name__}")
    out = {'key': key, 'label': label}
    for field_name in _SCOPE_PATH_FIELDS:
        val = s[field_name]
        if not isinstance(val, str) or not val:
            raise ViewError(
                f"scopes[{i}] {field_name!r} must be a non-empty string, got {val!r}"
            )
        if os.path.isabs(val) or '/' in val or '\\' in val:
            raise ViewError(
                f"scopes[{i}] {field_name!r} must be a relative basename "
                f'(no path separators), got {val!r}'
            )
        out[field_name] = val
    n_signals = s['n_signals']
    if isinstance(n_signals, bool) or not isinstance(n_signals, int) or n_signals < 0:
        raise ViewError(
            f"scopes[{i}] 'n_signals' must be a non-negative int, got {n_signals!r}"
        )
    out['n_signals'] = n_signals
    return out


def _validate_scopes(scopes, out_path):
    """Validate+normalize a §3.2 ``scopes=`` list into the manifest dict
    ``{'current': key|None, 'scopes': [...]}`` — always this shape, never a
    bare list, per the "always present, one shape" rule."""
    if not scopes:
        return {'current': None, 'scopes': []}
    if not isinstance(scopes, list):
        raise ViewError(f'scopes must be a list, got {type(scopes).__name__}')
    normalized = [_validate_scope_entry(s, i) for i, s in enumerate(scopes)]
    seen_keys = set()
    for i, s in enumerate(normalized):
        if s['key'] in seen_keys:
            raise ViewError(f"scopes[{i}] has duplicate key {s['key']!r}")
        seen_keys.add(s['key'])
    out_basename = os.path.basename(out_path)
    current = next((s['key'] for s in normalized if s['html'] == out_basename), None)
    return {'current': current, 'scopes': normalized}


# ---------------------------------------------------------------------------
# View-state sidecar (v1.3 spec §3.2/§3.6) — envelope validation only. The
# 'view' block's interior is embedded verbatim, unvalidated: the frontend
# already prunes unknown signal names, clamps values, and falls back on
# corrupt state, and re-implementing that tolerance here would be a second,
# drifting copy of a UI contract (spec §3.6).
# ---------------------------------------------------------------------------

_VIEW_STATE_ENVELOPE_KEYS = frozenset({'silkview', 'generator', 'created', 'run', 'view'})


def _validate_view_state(obj):
    """Validate a §3.2 envelope (or a bare ``view`` block) and return just
    the ``view`` block — the part embedded as ``window.SILK_VIEW_STATE``.

    A bare block (no ``silkview`` key) is detected and wrapped, per §3.6's
    ``view_state=<dict>`` behaviour; a full envelope is validated as given.
    """
    if not isinstance(obj, dict):
        raise ViewError(f'view_state must be a dict, got {type(obj).__name__}')
    if 'silkview' not in obj:
        obj = {'silkview': 1, 'view': obj}
    silkview = obj.get('silkview')
    if silkview != 1:
        raise ViewError(
            f"view_state 'silkview' must be 1, got {silkview!r} "
            f'(this build understands schema version 1)'
        )
    unknown = set(obj) - _VIEW_STATE_ENVELOPE_KEYS
    if unknown:
        raise ViewError(
            f'unknown view_state key(s) {sorted(unknown)}; '
            f'accepted keys are {sorted(_VIEW_STATE_ENVELOPE_KEYS)}'
        )
    view = obj.get('view')
    if not isinstance(view, dict):
        raise ViewError(f"view_state 'view' must be a dict, got {type(view).__name__}")
    if 'v' in view and view['v'] != 1:
        raise ViewError(f"view_state view['v'] must be 1, got {view['v']!r}")
    return view


def _load_view_state_file(path):
    path = os.fspath(path)
    if not os.path.exists(path):
        raise ViewError(f'view_state file {path!r} does not exist')
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ViewError(f'view_state file {path!r} is not valid JSON: {exc}') from exc
    return _validate_view_state(obj)


def _resolve_view_state(view_state, autoload_view_state, out_path):
    """Returns ``(view_block_or_None, absolute_path_or_None)`` per §3.6's
    ``view_state=``/``autoload_view_state=`` kwarg table."""
    if view_state is not None:
        if isinstance(view_state, dict):
            return _validate_view_state(view_state), None
        path = os.fspath(view_state)
        return _load_view_state_file(path), os.path.abspath(path)
    if not autoload_view_state:
        return None, None
    sidecar_path = os.path.splitext(out_path)[0] + '.silkview.json'
    if not os.path.exists(sidecar_path):
        return None, None
    return _load_view_state_file(sidecar_path), os.path.abspath(sidecar_path)


# ---------------------------------------------------------------------------
# Command bus (v1.3 spec §4.5) — structural validation only. Python does not
# know the command registry (§4.3 is the frontend's) and must not grow a
# copy of it that drifts; an unknown command name is reported by the
# frontend at load time, not here.
# ---------------------------------------------------------------------------

_COMMAND_KEYS = frozenset({'cmd', 'args'})


def _validate_commands(commands):
    if commands is None:
        return []
    if not isinstance(commands, list):
        raise ViewError(f'commands must be a list, got {type(commands).__name__}')
    validated = []
    for i, cmd in enumerate(commands):
        if isinstance(cmd, str):
            validated.append(cmd)
            continue
        if not isinstance(cmd, dict):
            raise ViewError(
                f'commands[{i}] must be a str or dict, got {type(cmd).__name__}'
            )
        unknown = set(cmd) - _COMMAND_KEYS
        if unknown:
            raise ViewError(
                f'commands[{i}] has unknown key(s) {sorted(unknown)}; '
                f'accepted keys are {sorted(_COMMAND_KEYS)}'
            )
        if 'cmd' not in cmd:
            raise ViewError(f"commands[{i}] is missing required 'cmd'")
        if not isinstance(cmd['cmd'], str):
            raise ViewError(
                f"commands[{i}]['cmd'] must be a str, got {type(cmd['cmd']).__name__}"
            )
        if 'args' in cmd and not isinstance(cmd['args'], dict):
            raise ViewError(
                f"commands[{i}]['args'] must be a dict, got {type(cmd['args']).__name__}"
            )
        validated.append(cmd)
    return validated


# ---------------------------------------------------------------------------
# Display labels (TASK-527)
# ---------------------------------------------------------------------------

DESIGNER_LABELS = 'designer'


def _designer_labels(plots_plan, raw_file):
    """``labels='designer'``: the bracket alias of every SELECTED
    vector, built from the plan's ``raw`` names (after ``_plan_all``'s complex
    expansion, which is neither moved nor repeated) and recomputed on every
    emit. Collisions are decided over EVERY vector of the rawfile (all plots,
    case-insensitively, the independent axes counting as occupied), not over
    the selection, so a filtered-out literal ``xx1[0].mid`` still keeps
    ``xx1<0>.mid`` unlabelled. This display-collision check is not reader
    uniqueness: a shown label can still be ambiguous to the reader (a bare
    ``xx1[0].mid`` row, or the colon twin ``v(xx1[0]:mid)``), and only the
    ``address`` field (:func:`silk.vector_match.vector_address`) carries that
    proof. Only names whose alias differs are labelled, so a selection without
    arrays carries no label and its payload and ``run.id`` are byte-identical
    to an unlabelled emit."""
    from silk.spice_names import designer_alias_map
    axes = [p.vectors[0].name for p in raw_file.plots if p.vectors and _signal_start(p)]
    every = [v.name for p in raw_file.plots for v in p.vectors[_signal_start(p):]]
    aliases = designer_alias_map(every, protected=axes)
    out = {}
    for pp in plots_plan:
        for sig in pp['signals']:
            name = sig['raw']
            alias = aliases.get(name, name)
            if alias != name:
                out[name] = alias
    return out


def _apply_addresses(plots_plan, raw_file, labels):
    """Designer mode only: attach to each selected signal the
    bracket ADDRESS of its vector, proven from the COMPLETE source plot (before
    selection and exclude) with the reader's own matcher
    (:func:`silk.vector_match.vector_address`), per plot. A caller's labels dict
    or no labels never produce an address; complex magnitude and phase share
    their raw vector's address."""
    if not (isinstance(labels, str) and labels == DESIGNER_LABELS):
        return
    from silk.vector_match import vector_address
    for pp, plot in zip(plots_plan, raw_file.plots):
        proven = {}
        for sig in pp['signals']:
            row = sig['row']
            if row not in proven:
                proven[row] = vector_address(plot, row)
            if proven[row]:
                sig['address'] = proven[row]


def _validate_labels(labels, plots_plan, raw_file=None):
    """Check ``labels=`` (raw vector name -> display label, or the string
    ``'designer'``) and return a plain dict. Keys must be rawfile vector names,
    values non-empty strings; the raw name stays every trace's identity, a
    label is display only."""
    if isinstance(labels, str) and labels == DESIGNER_LABELS:
        return _designer_labels(plots_plan, raw_file)
    if not labels:
        return {}
    if not isinstance(labels, dict):
        raise ViewError(f'labels must be a dict of raw name -> label, got '
                        f'{type(labels).__name__}')
    valid = {sig['raw'] for pp in plots_plan for sig in pp['signals']}
    out = {}
    for key, value in labels.items():
        if not isinstance(key, str):
            raise ViewError(f'labels key must be a string, got {key!r}')
        if not isinstance(value, str) or not value:
            raise ViewError(f'labels[{key!r}] must be a non-empty string, got {value!r}')
        if key not in valid:
            matches, is_fallback = _closest_matches(key, valid) if valid else ([], True)
            hint = (f' No close match; available signals include {matches}.'
                    if is_fallback else f' Closest matches: {matches}.')
            raise ViewError(f'labels key {key!r} does not match any signal in '
                            f'this rawfile.{hint}')
        out[key] = value
    return out


def _apply_labels(plots_plan, labels):
    """The deterministic label plan, shared in spirit with the JS core.

    Per plot, after complex expansion and name dedup, a trace's effective
    label is its proposed label (with the ``(mag)``/``(phase)`` part) or its
    final name. Where two or more traces share an effective label, every
    LABELLED trace of that set loses its label (the raw name is shown), and
    the pass repeats until nothing drops, so a fallback cannot create a new
    collision. ``name``/``group`` are never touched.
    """
    if not labels:
        return
    for pp in plots_plan:
        proposed = {}
        for i, sig in enumerate(pp['signals']):
            base = labels.get(sig['raw'])
            if base:
                proposed[i] = f"{base} ({sig['part']})" if sig['part'] else base
        while proposed:
            shown = {}
            for i, sig in enumerate(pp['signals']):
                shown.setdefault(proposed.get(i, sig['name']), []).append(i)
            drop = {i for members in shown.values() if len(members) > 1
                    for i in members if i in proposed}
            if not drop:
                break
            for i in drop:
                del proposed[i]
        for i, label in proposed.items():
            pp['signals'][i]['label'] = label


# ---------------------------------------------------------------------------
# Size projection (must run before any base64 encoding — see emit_view)
# ---------------------------------------------------------------------------

def _project_bytes(plots_plan, decimate_kwargs):
    """Upper-bound estimate of the final payload size, in bytes.

    Pure numpy (dtype decision, stride count) — never calls the base64
    encoder. Used to fail loud on an oversized payload *before* paying for
    the (much more expensive) encoding pass.
    """
    total = 0.0
    for pp in plots_plan:
        total += len(pp['x_arr']) * 8
        for sig in pp['signals']:
            n = len(sig['y_arr'])
            dt_bytes = 4 if decide_y_dtype(sig['y_arr']) == 'f32' else 8
            total += n * dt_bytes
            strides = decimation_strides(n, **decimate_kwargs)
            for stride in strides[1:]:
                nbuckets = math.ceil(n / stride)
                total += 2 * nbuckets * (8 + dt_bytes)
    return int(total * 4 / 3)  # base64 inflation


# ---------------------------------------------------------------------------
# Encoding pass
# ---------------------------------------------------------------------------

def _encode_all(plots_plan, decimate_kwargs):
    encoded_plots = []
    n_signals_by_plot = []
    n_points_by_plot = []
    for pp in plots_plan:
        x_signal = build_x_signal(pp['x_arr'], type_=pp['x_type'])
        x_signal['name'] = pp['x_name']
        enc_signals = []
        for sig in pp['signals']:
            enc = build_data_signal(
                sig['name'], sig['type'], sig['y_arr'], pp['x_arr'],
                part=sig['part'], group=sig['group'], **decimate_kwargs,
            )
            if sig.get('role'):
                enc['role'] = sig['role']
            if sig['renamed']:
                enc['renamed'] = True
            if sig.get('label'):
                enc['label'] = sig['label']
            if sig.get('address'):
                enc['address'] = sig['address']
            for key in ('source_row', 'node_probe'):
                if key in sig:
                    enc[key] = sig[key]
            enc_signals.append(enc)
        encoded_plots.append({
            'name': pp['plotname'], 'kind': pp['kind'], 'complex': pp['complex'],
            'npoints': pp['npoints'], 'x': x_signal, 'signals': enc_signals,
            'x_log_default': pp['x_log_default'], 'date': pp['date'],
        })
        if 'node_paths' in pp:
            encoded_plots[-1]['node_paths'] = pp['node_paths']
        n_signals_by_plot.append(len(enc_signals))
        n_points_by_plot.append(pp['npoints'])
    return encoded_plots, n_signals_by_plot, n_points_by_plot


# ---------------------------------------------------------------------------
# Identity fields (spec §4.5)
# ---------------------------------------------------------------------------

def _compute_run_id(payload_wo_run):
    blob = json.dumps(encode_json(payload_wo_run, finite_only=True), sort_keys=True, allow_nan=False).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()[:16]


def _compute_state_key(seed):
    return hashlib.sha256(seed.encode('utf-8')).hexdigest()[:8]


def _resolve_title(title, raw, default_plot_idx, source_label, context_title=None):
    """Precedence per the view-context-and-scopes design §2.1:
    ``context["title"]`` -> caller's ``title=`` -> rawfile's own title ->
    basename. This is what stops every page being titled the netlist's
    ``*** automated netlist generator`` header comment."""
    if context_title:
        return context_title
    if title:
        return title
    if raw.plots:
        plot_title = raw.plots[default_plot_idx].title
        if plot_title:
            return plot_title
    if source_label:
        return os.path.basename(source_label)
    return 'rawfile'


def _assemble_payload(plots_plan, default_plot_idx, norm_annotations, decimate_kwargs,
                      raw, source_label, title, state_key, norm_context=None):
    encoded_plots, n_signals_by_plot, n_points_by_plot = _encode_all(plots_plan, decimate_kwargs)
    context_title = norm_context.get('title') if norm_context else None
    title_final = _resolve_title(title, raw, default_plot_idx, source_label, context_title)
    payload_wo_run = {
        'format': DATA_FORMAT_VERSION,
        'endian': 'le',
        'generator': 'silk.signals.view',
        'generator_version': _silk_version(),
        'plots': encoded_plots,
        'default_plot': default_plot_idx,
        'annotations': norm_annotations,
    }
    if norm_context is not None:
        # Additive, optional key (spec §2.1) — included in the run.id hash
        # alongside the resolved displayed title and other non-'run' fields,
        # so a context-only change (e.g. a new spec pass/fail summary) is a
        # new run for reload purposes.
        payload_wo_run['context'] = norm_context
    run_id = _compute_run_id({**payload_wo_run, 'title': title_final})
    payload = {
        'format': payload_wo_run['format'],
        'endian': payload_wo_run['endian'],
        'generator': payload_wo_run['generator'],
        'generator_version': payload_wo_run['generator_version'],
        'run': {
            'id': run_id,
            'state_key': state_key,
            'emitted_at': _now_iso(),
            'source': source_label,
            'title': title_final,
        },
        'plots': payload_wo_run['plots'],
        'default_plot': payload_wo_run['default_plot'],
        'annotations': payload_wo_run['annotations'],
    }
    if norm_context is not None:
        payload['context'] = norm_context
    return payload, n_signals_by_plot, n_points_by_plot


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def _sidecar_names(out_path):
    """``(data_name, stamp_name)`` for *out_path*.

    Derived from the HTML stem (not hardcoded ``data.js``/``stamp.js``) so two
    pages emitted into ONE output directory — e.g. by
    :func:`~silk.signals.view.select.emit_views_by_block` — do not silently
    clobber each other's data. Shared by the sidecar writer and the
    inline-mode cleanup below so the two cannot disagree about which files
    belong to a page.
    """
    stem = os.path.splitext(os.path.basename(out_path))[0]
    return f'{stem}.data.js', f'{stem}.stamp.js'


def _remove_stale_sidecars(out_dir, out_path):
    """Delete sidecars an earlier NON-inline emit to *out_path* left behind.

    An inline page carries its payload inside itself and never loads a
    sidecar, so a ``<stem>.data.js`` sitting next to one is an orphan holding
    a *previous* run's data while looking current. Called only after the
    inline HTML is safely written, so a failed emit can never strip a working
    page's data.
    """
    for name in _sidecar_names(out_path):
        path = os.path.join(out_dir, name)
        if not os.path.exists(path):
            continue
        try:
            os.remove(path)
        except OSError as exc:
            raise ViewError(
                f'wrote the inline page but could not remove {path!r}, a stale '
                f'sidecar from an earlier non-inline emit: {exc}. Delete it by '
                f'hand — an inline page never reads it, but it holds old data.'
            ) from exc


def _atomic_write(path, text, *, only_if_changed=False):
    """Write *text* to *path* via a temp file + ``os.replace`` in the same
    directory, so a poll never observes a half-written sidecar."""
    if only_if_changed and os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if f.read() == text:
                    return False
        except OSError:
            pass
    tmp_path = path + '.tmp'
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write(text)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

NODE_METADATA_MAX_BYTES = 8 * 1024 * 1024


def _bounded_metadata_bytes(value, limit):
    """Count the exact data.js bytes, stopping before any oversized output exists.

    These private facts contain only JSON strings, ints, bools and nulls, so
    the general numeric codec is unnecessary here. iterencode never builds
    the full string; each string token contains its entire script-close text.
    """
    from silk.signals.view.serialize import _escape_script_close
    size = len(build_data_js(None).encode('utf-8')) - len('null')
    if size > limit:
        return size
    for chunk in json.JSONEncoder(allow_nan=False).iterencode(value):
        size += len(_escape_script_close(chunk).encode('utf-8'))
        if size > limit:
            break
    return size


def _node_plan(raw, node_context, signals, exclude, max_depth, labels,
               decimate_kwargs, max_bytes=None):
    from silk.signals.view.node_context import ViewNodeContext
    from silk.signals.view.node_select import select_node_paths
    import warnings
    metadata = None
    if node_context is not None:
        if not isinstance(node_context, ViewNodeContext):
            raise TypeError('node_context must be a checked ViewNodeContext or None')
        metadata = node_context._for_source(raw)
    if metadata is None:
        masks, totals = _resolve_selection(raw, signals, exclude, max_depth)
    else:
        # Selection sees logical names as ordinary current vectors, while the
        # published node inventory and source-row indices remain physical.
        inventories = []
        for i, meta in enumerate(metadata):
            sources = (node_context._source_currents[i]
                       if node_context._source_currents else ())
            vectors = list(meta['vectors']) + [
                {'name': probe, 'selectable': True} for probe, _, _ in sources]
            inventories.append({'vectors': vectors, 'node_paths': dict(meta, vectors=vectors)})
        selection = select_node_paths(inventories, signals=signals, exclude=exclude,
                                      max_depth=max_depth)
        for diagnostic in selection['diagnostics']:
            warnings.warn(f"{diagnostic['keyword']} selector {diagnostic['selector']!r}: "
                          'emitted reading takes precedence over designer nodes.', UserWarning,
                          stacklevel=3)
        masks = [m[_signal_start(p):len(p.vectors)] for m,p in zip(selection['masks'], raw.plots)]
        totals = [_total_signals_for_plot(p) for p in raw.plots]
    plans, default = _plan_all(raw, masks=masks)
    _apply_labels(plans, _validate_labels(labels, plans, raw))
    _apply_addresses(plans, raw, labels)
    if metadata is None:
        return plans, default, totals
    # Keep the chosen rows if the metadata budget forces legacy display.
    legacy = [dict(p, signals=[dict(sig) for sig in p['signals']]) for p in plans]
    for pp, meta, proof in zip(plans, metadata, node_context._proofs):
        pp['node_paths'] = meta
        rows = {row: (label, probe) for row,label,probe in proof}
        graph_labels = {}
        for sig in pp['signals']:
            sig['source_row'] = sig['row']
            if sig['row'] in rows:
                label, probe = rows[sig['row']]
                sig['node_probe'] = {'name': probe, 'type': 'I' if probe.startswith('i(') else 'V'} if probe else None
                if labels == DESIGNER_LABELS:
                    sig.pop('label', None)
                    if label:
                        graph_labels[sig['raw']] = label
        _apply_labels([pp], graph_labels)
    for i, (pp, plot, sources) in enumerate(zip(plans, raw.plots, node_context._source_currents)):
        logical_mask = selection['masks'][i][len(plot.vectors):]
        for (probe, terms, missing), selected in zip(sources, logical_mask):
            if not selected:
                continue
            if missing:
                # Default views offer only complete logical currents; saved
                # physical cards remain independently available above.
                if signals is None:
                    continue
                raise ViewError(f'Logical source current {probe} is missing saved card(s): '
                                f'{", ".join(missing)}. Save these source-current cards '
                                'and rerun the simulation before plotting this source.')
            # Sum complex values before magnitude/phase expansion or decimation.
            values = np.asarray(plot.vectors[terms[0][0]].data)
            if terms[0][1] != 1:
                values = values * terms[0][1]
            for row, sign in terms[1:]:
                values = values + np.asarray(plot.vectors[row].data) * sign
            from silk.signals.rawfile import RawPlot, RawVector
            source_plot = RawPlot(plot.header, [plot.vectors[0], RawVector(probe, 'current', data=values)],
                                  plot.npoints, plot.real, None)
            logical = _plan_plot(source_plot)['signals']
            if _signal_start(plot) == 0:
                logical = logical[1:]
            for sig in logical:
                sig.pop('row', None)
                if labels == DESIGNER_LABELS:
                    sig['label'] = sig['name']
                sig['node_probe'] = {'name': probe, 'type': 'I',
                                     'terms': [{'row': row, 'sign': sign} for row, sign in terms]}
            pp['signals'].extend(logical)
    if ((signals is not None or exclude is not None or max_depth is not None)
            and not any(pp['signals'] for pp in plans)):
        from silk.signals.view.template import EmptySelectionError
        raise EmptySelectionError(
            'signal selection (signals=/exclude=/max_depth=) matched zero signals '
            'in this rawfile — nothing would be emitted')
    extra = [{'node_paths': p['node_paths'], 'signals': [
        {k: sig[k] for k in ('source_row', 'node_probe', 'label') if k in sig}
        for sig in p['signals']]} for p in plans]
    remaining = (None if max_bytes is None
                 else max_bytes - _project_bytes(legacy, decimate_kwargs))
    limit = NODE_METADATA_MAX_BYTES if remaining is None else min(NODE_METADATA_MAX_BYTES, remaining)
    metadata_bytes = _bounded_metadata_bytes(extra, limit)
    if metadata_bytes > limit:
        budget = '' if remaining is None else f', remaining payload budget {remaining:,} bytes'
        logical_sources = sorted({sig['node_probe']['name'] for pp in plans
                                  for sig in pp['signals']
                                  if (sig.get('node_probe') or {}).get('terms')})
        if logical_sources:
            remedy = ('Increase max_bytes or reduce exported waveforms/metadata.'
                      if remaining is not None and remaining < NODE_METADATA_MAX_BYTES
                      else 'Reduce the exported graph metadata.')
            payload_limit = f' (max_bytes={max_bytes:,} bytes)' if remaining is not None else ''
            raise ViewError(f'Cannot retain selected logical source current(s) '
                            f'{", ".join(logical_sources)}: metadata cap '
                            f'{NODE_METADATA_MAX_BYTES:,} bytes{budget}'
                            f'{payload_limit} '
                            'would force a physical-only fallback and discard these traces. '
                            f'{remedy}')
        warnings.warn(f'Estimated node metadata size exceeds cap {NODE_METADATA_MAX_BYTES:,} bytes'
                      f'{budget}; omitting graph metadata and using legacy labels. '
                      'Browser graph selectors and checked copy are unavailable.',
                      UserWarning, stacklevel=3)
        plans = legacy
    else:
        # Only make public mutable JSON containers after the bounded count fits.
        for i, pp in enumerate(plans):
            pp['node_paths'] = encode_json(pp['node_paths'], finite_only=True)
            totals[i] += len(pp['signals']) - len(legacy[i]['signals'])
    return plans, default, totals


def emit_view_data(source, *, title=None, annotations=None, context=None,
                   labels=None, signals=None, exclude=None, max_depth=None, node_context=None,
                   decimate_min_points=DECIMATE_MIN_POINTS,
                   decimate_base=DECIMATE_BASE,
                   decimate_min_bucket=DECIMATE_MIN_BUCKET):
    """Build the waveform-explorer payload for *source* without writing
    anything.

    Parameters mirror :func:`emit_view`, minus everything about where or how
    to write (there is no ``scopes=`` — the manifest is per-output-file and
    has nothing to attach to here). This is what the corpus round-trip tests
    assert on, and what a future caller would call to get the
    payload dict directly.

    Returns
    -------
    dict
        The payload (spec §4.3), ``json.dumps``-round-trippable. Carries an
        optional top-level ``context`` key (view-context-and-scopes design
        §2.1) when *context* is given.
    """
    raw, source_label = _resolve_source(source)
    if not raw.plots:
        raise ViewError(f'rawfile {source_label!r} contains no plots')

    decimate_kwargs = dict(min_points=decimate_min_points, base=decimate_base,
                           min_bucket=decimate_min_bucket)
    plots_plan, default_plot_idx, _ = _node_plan(
        raw, node_context, signals, exclude, max_depth, labels, decimate_kwargs)
    all_names = {sig['name'] for pp in plots_plan for sig in pp['signals']}
    norm_annotations = _validate_annotations(annotations, all_names)
    norm_context = _validate_context(context)

    # No output path in this entry point — state_key is keyed on the source
    # label instead of an output path (there is none to key on).
    state_key = _compute_state_key(source_label or '<in-memory>')

    payload, _, _ = _assemble_payload(
        plots_plan, default_plot_idx, norm_annotations, decimate_kwargs,
        raw, source_label, title, state_key, norm_context=norm_context,
    )
    return payload


def emit_view(source, out_path, *, title=None, annotations=None, context=None,
              inline=False, poll_ms=2000, max_bytes=512_000_000,
              view_state=None, autoload_view_state=True, commands=None,
              scopes=None, labels=None, node_context=None,
              signals=None, exclude=None, max_depth=None,
              decimate_min_points=DECIMATE_MIN_POINTS,
              decimate_base=DECIMATE_BASE,
              decimate_min_bucket=DECIMATE_MIN_BUCKET,
              group_signals_by_kind=True):
    """Write a self-contained waveform explorer for *source*.

    Parameters
    ----------
    source : str or os.PathLike or silk.signals.rawfile.RawFile
        A nutmeg rawfile path, or an already-parsed
        :class:`~silk.signals.rawfile.RawFile`. A path is read with
        ``silk.signals.rawfile.read_rawfile``.
    out_path : str or os.PathLike
        Destination ``.html`` path. Its parent directory is created if
        missing. ``<stem>.data.js`` and ``<stem>.stamp.js`` — *stem* being
        ``out_path`` with its extension stripped — are written alongside it
        (unless *inline*), so multiple pages can share one output directory
        without clobbering each other's data.
    title : str, optional
        Page title. Defaults to the rawfile's plot title, else its basename.
    annotations : dict, optional
        Design-aware overlay; schema in spec §6. Validated strictly — an
        unknown key or an unknown signal name raises :class:`~silk.signals.view.template.ViewError`.
    node_context : ViewNodeContext, optional
        Checked context from ``view_node_context(owner)``. Its source is a
        borrowed read-only snapshot. Enables designer selection and exact
        copy; with ``labels='designer'`` also enables checked display labels.
        Metadata counts toward max_bytes and has an 8 MiB cap. Oversized
        metadata is omitted with a warning, retaining selected waveform rows.
        Exclusion filters data, not names retained in the metadata inventory.
    labels : dict or ``'designer'``, optional
        Raw vector name -> display label, built by the caller, for
        example with :func:`silk.spice_names.to_designer_name`; or the string
        ``'designer'``, which labels every selected vector with its
        bracket alias where that differs and is unique among the vectors of the
        whole rawfile (rebuilt on every emit from the selected signals; none
        when the selection has no arrays). A label is carried as a per-signal
        ``label`` field and is the PRIMARY text of the trace in the viewer
        (the raw spelling is secondary); the raw name stays the signal's ``name``, ``group`` and
        every annotation/state/measurement key. Labels take part in
        ``run.id``; ``state_key`` is unchanged; with none, the payload and
        hash are byte-identical to before. Within one plot, labels that
        collide (with each other or with an unlabelled name) are all
        dropped, repeatedly, so every shown label is unique. A key naming no
        vector raises :class:`~silk.signals.view.template.ViewError`.
    context : dict, optional
        Run provenance (view-context-and-scopes design §2.1) — module,
        test, analysis, corner, PDK, spec outcomes, whatever the caller
        wants shown in the page's info popover. Shape:
        ``{'title': str, 'subtitle': str, 'fields': [{'label': str,
        'value': Any, 'severity': 'info'|'warn'|'error'}, ...]}``, all keys
        optional. Validated shallowly and fail-loud, exactly like
        *annotations*: an unknown top-level key, a field missing
        ``label``/``value``, or a bad ``severity`` raises
        :class:`~silk.signals.view.template.ViewError`; a non-str ``value`` is ``str()``-coerced.
        Capped at 64 fields / 4 KB serialized (spec §6 R4) — also a
        :class:`~silk.signals.view.template.ViewError`. A falsy *context* (``None`` or ``{}``, the
        default) embeds no ``context`` key in the payload at all.
        ``context['title']`` takes precedence over *title* for the page's
        displayed run title (spec §2.1).
    inline : bool, default False
        Write ONE file with the data embedded, and no sidecars. Auto-reload
        is unavailable in this mode. Use it to email or commit the artifact.
        Sidecars an earlier non-inline emit to the same *out_path* left
        behind are removed, so the directory does not keep a stale
        ``<stem>.data.js`` that no page reads.
    poll_ms : int, default 2000
        Sidecar poll interval baked into the page. 0 disables auto-reload.
    max_bytes : int, default 512_000_000
        Refuse to emit a payload projected larger than this.
    signals : list of str, optional
        Selection filter. ``None`` (default) includes every signal. Otherwise a vector is included iff it matches ANY
        selector — see ``silk.signals.view.select.matches_selector``
        for the three match forms (exact name, glob, hierarchy
        contiguous-run — all case-insensitive). Matching happens on the
        rawfile vector name *before* complex (AC) expansion, so selecting
        ``'v(out)'`` on an AC file keeps both its ``(mag)`` and ``(phase)``
        signals. A selector matching zero vectors anywhere in the file, or a
        final selection that is empty, raises :class:`~silk.signals.view.template.ViewError` naming it
        (with near-miss suggestions).
    exclude : list of str, optional
        Same selector forms as *signals*, applied after it; exclusion wins.
    max_depth : int, optional
        Keep only signals whose hierarchy depth (the number of dotted path
        segments before the leaf, per the v1.3 §2.2 grammar) is
        ``<= max_depth``. ANDed with *signals*/*exclude*. ``max_depth=0``
        keeps only bare/top-level names (``v(vout)``, ``inoise_spectrum``).
    view_state : str, os.PathLike, or dict, optional
        v1.3 spec §3.6. ``None`` (default): if *autoload_view_state*, read
        an adjacent ``<stem>.silkview.json`` when present; otherwise embed
        nothing. A path: read, validate, and embed that file — a missing
        file raises :class:`~silk.signals.view.template.ViewError` (an explicit path that does not
        exist is a typo, not a "no state" signal). A dict: validate and
        embed directly; accepts either the full §3.2 envelope or a bare
        ``view`` block (detected by the presence of a ``silkview`` key).
    autoload_view_state : bool, default True
        When ``view_state`` is ``None``, whether to look for the adjacent
        sidecar at all. Ignored when *view_state* is not ``None``.
    commands : list, optional
        v1.3 spec §4.5. A list of command dicts (``{'cmd': str, 'args':
        dict}``) or bare strings, embedded as ``window.SILK_VIEW_COMMANDS``
        and replayed by the frontend on load. Validated structurally only
        (never enumerates command names) — an element that is neither str
        nor dict, a dict missing ``cmd``, a non-str ``cmd``, a non-dict
        ``args``, or an unknown key each raise :class:`~silk.signals.view.template.ViewError` naming the
        offending index.
    scopes : list of dict, optional
        The ``SILK_VIEW_SCOPES`` manifest (view-context-and-scopes design
        §3.2) for a hand-built page set — the whole set's entries, this
        emission's own included. Each entry:
        ``{'key': str, 'label': str (optional, defaults to key), 'html':
        str, 'data_src': str, 'stamp_src': str, 'n_signals': int}``.
        ``html``/``data_src``/``stamp_src`` must be relative basenames (no
        path separators) — the set is one directory by construction, and a
        ``file://`` page must not carry paths that break when the directory
        moves. This emission's own entry (the one whose ``html`` matches
        this call's ``out_path`` basename) becomes the manifest's
        ``current``; no entry matching is *not* an error (``current`` is
        just ``None``). ``None`` (default) embeds the always-present empty
        shape ``{"current": null, "scopes": []}``. Incompatible with
        ``inline=True`` (there is no sibling ``data.js`` to switch to) —
        raises :class:`~silk.signals.view.template.ViewError`. ``silk.signals.view.select.emit_views_by_block``
        builds this automatically for the whole set it emits.
    decimate_min_points, decimate_base, decimate_min_bucket : int
        Decimation-pyramid tuning; see spec §4.8.
    group_signals_by_kind : bool, default True
        The signal list groups voltages and currents into their own
        collapsible sections; ``False`` restores the single,
        unsectioned tree exactly as before that feature. Baked into the page
        the same way as *poll_ms* (``window.SILK_VIEW_CONFIG``); the
        embedded API's equivalent is ``createExplorer``'s
        ``groupSignalsByKind`` option — see docs/explorer-integration.md.
        Display only: never changes the payload.

    Returns
    -------
    silk.signals.view.emit.ViewOutput

    Raises
    ------
    silk.signals.view.template.ViewError
        Unparseable annotations or context, an unknown signal name in
        annotations, an unwritable destination, a rawfile with no plots, a
        payload projected larger than *max_bytes*, a malformed/missing
        ``view_state``, a structurally invalid ``commands`` list, a
        malformed ``scopes`` manifest, ``scopes`` combined with
        ``inline=True``, a *signals*/*exclude* selector matching nothing in
        the file, or a selection (*signals*/*exclude*/*max_depth* combined)
        that is empty.
    silk.signals.rawfile.RawfileError
        Propagated from the reader for a non-rawfile input.
    """
    raw, source_label = _resolve_source(source)
    if not raw.plots:
        raise ViewError(f'rawfile {source_label!r} contains no plots')

    out_path = os.fspath(out_path)
    out_dir = os.path.dirname(os.path.abspath(out_path)) or '.'
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        raise ViewError(f'cannot create output directory {out_dir!r}: {exc}') from exc

    if inline and scopes:
        raise ViewError(
            'scopes= is not supported with inline=True — an inline page has '
            'no sibling data.js for a scope switch to inject'
        )

    decimate_kwargs = dict(min_points=decimate_min_points, base=decimate_base,
                           min_bucket=decimate_min_bucket)
    plots_plan, default_plot_idx, n_signals_total_by_plot = _node_plan(
        raw, node_context, signals, exclude, max_depth, labels, decimate_kwargs, max_bytes)
    all_names = {sig['name'] for pp in plots_plan for sig in pp['signals']}
    norm_annotations = _validate_annotations(annotations, all_names)
    norm_context = _validate_context(context)
    scopes_manifest = _validate_scopes(scopes, out_path)
    view_state_block, view_state_path = _resolve_view_state(
        view_state, autoload_view_state, out_path,
    )
    commands_list = _validate_commands(commands)

    # Fail loud on an oversized payload BEFORE any base64 encoding — a
    # browser tab that hangs on a 1.6 GB payload is worse than a loud error
    # (spec R2). _project_bytes never touches the encoder.
    projected = _project_bytes(plots_plan, decimate_kwargs)
    if projected > max_bytes:
        raise ViewError(
            f'projected payload size ({projected:,} bytes) exceeds '
            f'max_bytes={max_bytes:,}. Escapes: reduce the number of saved '
            f'nodes in the testbench, or pass max_bytes= deliberately if '
            f'this size is expected.'
        )

    state_key = _compute_state_key(os.path.abspath(out_path))
    payload, n_signals_by_plot, n_points_by_plot = _assemble_payload(
        plots_plan, default_plot_idx, norm_annotations, decimate_kwargs,
        raw, source_label, title, state_key, norm_context=norm_context,
    )
    run_id = payload['run']['id']
    title_final = payload['run']['title']
    n_signals_total = n_signals_total_by_plot[default_plot_idx]

    if inline:
        data_script_html = '<script>' + build_data_js(payload) + '</script>'
        html_text = render_view_html(
            title=title_final, data_script_html=data_script_html,
            poll_ms=poll_ms, inline=True,
            view_state=view_state_block, commands=commands_list,
            scopes=scopes_manifest, group_signals_by_kind=group_signals_by_kind,
        )
        _atomic_write(out_path, html_text, only_if_changed=True)
        _remove_stale_sidecars(out_dir, out_path)
        return ViewOutput(
            html_path=os.path.abspath(out_path), data_path=None, stamp_path=None,
            run_id=run_id, state_key=state_key,
            n_plots=len(payload['plots']),
            n_signals=n_signals_by_plot[default_plot_idx],
            n_points=n_points_by_plot[default_plot_idx],
            view_state_path=view_state_path,
            n_signals_total=n_signals_total,
        )

    data_name, stamp_name = _sidecar_names(out_path)
    data_path = os.path.join(out_dir, data_name)
    stamp_path = os.path.join(out_dir, stamp_name)
    # The stem comes from a caller-chosen out_path, so it can contain '"' or
    # '&' and break out of the attribute it is written into; escape it as the
    # page title already is. stamp_name needs no equivalent — it reaches the
    # page only through the JSON config block, and the frontend assigns it to
    # script.src as a DOM property rather than parsing it as markup.
    data_script_html = f'<script src="{_html.escape(data_name, quote=True)}"></script>'
    html_text = render_view_html(
        title=title_final, data_script_html=data_script_html,
        poll_ms=poll_ms, inline=False,
        view_state=view_state_block, commands=commands_list,
        scopes=scopes_manifest,
        data_src=data_name, stamp_src=stamp_name,
        group_signals_by_kind=group_signals_by_kind,
    )
    _atomic_write(out_path, html_text, only_if_changed=True)
    _atomic_write(data_path, build_data_js(payload), only_if_changed=False)
    _atomic_write(stamp_path, build_stamp_js(run_id, time.time()), only_if_changed=False)

    return ViewOutput(
        html_path=os.path.abspath(out_path),
        data_path=os.path.abspath(data_path),
        stamp_path=os.path.abspath(stamp_path),
        run_id=run_id, state_key=state_key,
        n_plots=len(payload['plots']),
        n_signals=n_signals_by_plot[default_plot_idx],
        n_points=n_points_by_plot[default_plot_idx],
        view_state_path=view_state_path,
        n_signals_total=n_signals_total,
    )
