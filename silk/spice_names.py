"""The one SPICE-name codec: instance components use ``[k]`` in silk and
``<k>`` in the simulator.

ngspice cannot save an ``@`` address whose instance path contains ``[``, so silk's netlist and rawfile spell an arrayed copy's instance
component with angle brackets while designer names keep square brackets. This
module is the single conversion between the two spellings of a *vector name*.

Only instance positions of recognised vector syntax are converted:

- a dotted (ngspice, behavioral) or colon (Xyce) path: every segment but the
  last is an instance position; the last is a node leaf and is never touched;
- ``v(...)`` / ``i(...)`` wrappers (case-insensitive): ``i`` makes the last
  segment an instance too (a branch current names a source); ``v`` does not;
- ``@`` addresses (optionally wrapped), whose final ``[param]`` selector is
  kept: every path segment is an instance position;
- Xyce raw vector grammar ``N(path:PARAM)`` and ``ID(path)``.

Node leaves, parameter selectors, expressions and unknown internals come back
unchanged, so a literal-angle net ``g<0>`` stays distinct from ``g[0]``.
Case and separators are preserved, and the engine is never inferred.

``to_designer_name`` returns a *bracketed SPICE alias* of a vector name: not
designer case, not designer prefixes (``x``, ``XM``, ``V``) and not a verified
entity. A bare input has no role of its own, so ``kind`` says whether it is a
``'node'`` (default; the last segment is a leaf) or a ``'branch'`` (the last
segment is an instance); a wrapper in the string overrides ``kind``.

This is a stdlib-only leaf (the waveform viewer imports it).
"""
from __future__ import annotations

import re

__all__ = ['to_spice_name', 'to_designer_name', 'spice_component',
           'designer_component', 'designer_signal_names', 'designer_alias_map']

_STYLES = ('auto', 'ngspice', 'xyce', 'behavioral')
_KINDS = ('node', 'branch')

# wrapper head (lower case) -> (role of the wrapped vector, separators it may use)
_WRAPPER_ROLE = {'v': 'node', 'i': 'branch', 'n': 'node', 'id': 'branch'}
_RAW_XYCE = ('n', 'id')            # Xyce raw vector grammar: colon paths only
_STYLE_WRAPPERS = {
    'auto': ('v', 'i', 'n', 'id'), 'ngspice': ('v', 'i'),
    'xyce': ('v', 'i', 'n', 'id'), 'behavioral': (),
}
_STYLE_SEPS = {'auto': '.:', 'ngspice': '.', 'xyce': ':', 'behavioral': '.'}
_STYLE_AT = {'auto': True, 'ngspice': True, 'xyce': False, 'behavioral': False}

_WRAPPER = re.compile(r'([A-Za-z]+)\((.*)\)', re.S)
_PARAM_TAIL = re.compile(r'\[[^\[\]]*\]\Z')
# One path segment, spelled out (the same text as waveform-core.mjs) so that
# recognition never depends on a runtime's Unicode tables. A character is
# EITHER in the ASCII set exactly (A-Z a-z 0-9 _ $ - [ ] < >) OR non-ASCII and
# neither a C1 control (U+0080-U+009F) nor a listed Unicode space/BOM (U+00A0,
# U+1680, U+2000-U+200A, U+2028, U+2029, U+202F, U+205F, U+3000, U+FEFF). So
# letters of any script, combining marks and digits pass; every other ASCII
# character (the separators, parentheses, operators, '#', '?', '|', '&', '%',
# '^', '!', '~', quotes, whitespace, controls) marks an expression or a
# simulator-internal form, which is left alone. Recognition is always of the
# WHOLE string (fullmatch / \Z), never "up to a final newline".
_SEGMENT = re.compile(
    '(?:[A-Za-z0-9_$\\-\\[\\]<>]'
    '|[^\\x00-\\x9f\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000\\ufeff])+')


# Selector recognition (TASK-530, ``allow_glob=True``): the same segment text
# plus the wildcards '*' and '?'. Nothing else about recognition changes.
_SEGMENT_GLOB = re.compile(
    '(?:[A-Za-z0-9_$\\-\\[\\]<>*?]'
    '|[^\\x00-\\x9f\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000\\ufeff])+')


_AUTHORED_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*(\[[0-9]+\])?')


def validate_authored_name(name, *, kind, who):
    """Check one authored component; selectors and raw paths use other grammar.

    Only actual net names (``kind='net'``) admit global ground ``0``.
    Formal port names (``kind='port'``) use the shared identifier grammar
    without that exception.
    """
    if isinstance(name, str) and (
            (kind == 'net' and name == '0') or _AUTHORED_NAME.fullmatch(name)):
        return
    message = (
        f"{who}: invalid {kind} name {name!r}. Use ASCII letters, digits and _, "
        "starting with a letter or _, optionally followed by one numeric "
        "index [k]. ':' separates hierarchy levels; '.' also separates "
        "hierarchy levels, and neither belongs in a name."
    )
    if kind == 'net':
        message += " The ground net '0' is the only exception."
    if isinstance(name, str) and name.lower() == 'gnd!':
        message += (" gnd! is an ordinary node in SPICE, not ground; silk's ground "
                    "is '0'. Inside a block, declare Port.ground() (conventionally "
                    "vss) and wire it to '0' at the bench.")
    raise ValueError(message)


def spice_component(part: str) -> str:
    """ONE instance component, designer to simulator: ``X1[0]`` -> ``X1<0>``."""
    return part.replace('[', '<').replace(']', '>')


def designer_component(part: str) -> str:
    """ONE instance component, simulator to bracket form: ``X1<0>`` -> ``X1[0]``."""
    return part.replace('<', '[').replace('>', ']')


def _check(style, kind):
    if style not in _STYLES:
        raise ValueError(f"style must be one of {_STYLES}, got {style!r}")
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")


def _convert_path(path, conv, style, last_is_instance, *, seps=None, glob=False):
    """Convert a recognised plain path, or return None.

    Recognised: segments of :data:`_SEGMENT` joined by ONE separator kind that
    the style's grammar uses (a path mixing ``.`` and ``:`` is not a vector
    path of any engine).
    """
    seps = _STYLE_SEPS[style] if seps is None else seps
    if not path:
        return None
    present = [c for c in '.:' if c in path]
    if len(present) > 1 or (present and present[0] not in seps):
        return None
    segments = path.split(present[0]) if present else [path]
    segment = _SEGMENT_GLOB if glob else _SEGMENT
    if not all(segment.fullmatch(seg) for seg in segments):
        return None
    last = len(segments) - 1
    converted = [conv(seg) if (i < last or last_is_instance) else seg
                 for i, seg in enumerate(segments)]
    return (present[0] if present else '').join(converted)


def _convert_body(body, conv, style, role, wrapper=None, glob=False):
    """Convert the inside of a wrapper (or an unwrapped name), else None."""
    if body.startswith('@'):
        if not _STYLE_AT[style]:
            return None
        tail = _PARAM_TAIL.search(body)
        param = tail.group(0) if tail else ''
        core = body[1:len(body) - len(param)]
        done = _convert_path(core, conv, style, True, seps='.', glob=glob)
        return None if done is None else '@' + done + param
    seps = ':' if wrapper in _RAW_XYCE else None
    if seps is not None and seps not in _STYLE_SEPS[style]:
        return None
    return _convert_path(body, conv, style, role == 'branch', seps=seps, glob=glob)


def _convert(name, conv, style, kind, glob=False):
    _check(style, kind)
    if not isinstance(name, str):
        raise TypeError(f'name must be a str, got {type(name).__name__}')
    match = _WRAPPER.fullmatch(name)
    if match:
        head = match.group(1).lower()
        if head not in _STYLE_WRAPPERS[style]:
            return name                        # not this style's grammar
        done = _convert_body(match.group(2), conv, style,
                             _WRAPPER_ROLE[head], head, glob)
        return name if done is None else f'{match.group(1)}({done})'
    done = _convert_body(name, conv, style, kind, None, glob)
    return name if done is None else done


def to_spice_name(name: str, style: str = 'auto', *, kind: str = 'node',
                  allow_glob: bool = False) -> str:
    """Bracket spelling to the simulator's: ``xx1[0].mid`` -> ``xx1<0>.mid``.

    ``allow_glob=True`` (selectors) additionally recognises ``*`` and
    ``?`` inside a segment, so ``xx1[*].g[0]`` -> ``xx1<*>.g[0]``; every other
    recognition rule and refusal is unchanged, and the default is off.
    """
    return _convert(name, spice_component, style, kind, allow_glob)


def to_designer_name(name: str, style: str = 'auto', *, kind: str = 'node') -> str:
    """Simulator spelling to the bracket form: ``xx1<0>.mid`` -> ``xx1[0].mid``.

    A bracketed SPICE alias of the vector name only: designer case, prefixes
    and entity existence are not recovered or checked.
    """
    return _convert(name, designer_component, style, kind)


def designer_signal_names(names, *, skip: int = 1):
    """Bracket aliases of a rawfile's vector names, only where collision-safe.

    The first *skip* entries (the independent axis) stay as they are but still
    OCCUPY their raw spelling. A name's alias is its :func:`to_designer_name`;
    entries whose spellings coincide (case-insensitively: an alias with another
    alias, or with a protected or unchanged raw name) form a collision group
    and EVERY unprotected member keeps its raw spelling, so this function
    introduces no new alias collision. Order and count are preserved; nothing
    is dropped or deduplicated (duplicate raw rows already in the input stay
    duplicates).
    """
    names = list(names)
    alias = [n if i < skip else to_designer_name(n) for i, n in enumerate(names)]
    counts = {}
    for a in alias:
        counts[a.lower()] = counts.get(a.lower(), 0) + 1
    clash = {i for i, a in enumerate(alias) if i >= skip and counts[a.lower()] > 1}
    return [names[i] if (i < skip or i in clash) else alias[i] for i in range(len(names))]


def designer_alias_map(names, *, protected=()):
    """``{raw name: bracket alias}`` over the distinct *names*, collision-safe.

    The same rule as :func:`designer_signal_names`, applied to a whole
    inventory (every vector of a rawfile, every net path of a report): the
    *protected* spellings occupy their raw name, identical raw names are one
    entry, and a collision group keeps its raw spelling. The result does not
    depend on the order of *names*.
    """
    protected = list(dict.fromkeys(protected))
    rest = [n for n in dict.fromkeys(names) if n not in set(protected)]
    both = protected + rest
    shown = designer_signal_names(both, skip=len(protected))
    return dict(zip(both, shown))
