# Shared formatting utilities for silk

import re

# An SI prefix earns its place only when it removes exponent noise. Below this
# band we emit u/n/p/f, above it k/Meg/G/T; inside it, a bare decimal.
_PLAIN_LO, _PLAIN_HI = 1e-3, 1e3

_SUFFIXES = [
    (1e12, 'T'), (1e9, 'G'), (1e6, 'Meg'), (1e3, 'k'),
    (1e-6, 'u'), (1e-9, 'n'), (1e-12, 'p'), (1e-15, 'f'),
]


def _fmt(value) -> str:
    """Format a value with an SI suffix for SPICE output.

    Strings pass through unchanged. Large and small magnitudes take an SI suffix,
    because there it is what makes the number readable::

        10e3  -> '10k'      1e-12 -> '1p'
        1e6   -> '1Meg'     55e-9 -> '55n'

    Values in the **plain-decimal band** (``1e-3 <= |v| < 1e3``) are emitted as
    bare decimals, with no suffix::

        0.5   -> '0.5'      4.0   -> '4'
        0.15  -> '0.15'     1.8   -> '1.8'

    A milli prefix there shortens nothing and actively misleads. Transistor W/L
    live in this band (UNIT_PDK puts them in um), so ``l=0.5`` used to emit as
    ``l=500m`` — which reads as "500 milli-um", and leans on the ``m``-vs-``M``
    (milli vs mega) suffix ambiguity that some SPICE tools get wrong. ``0.5`` is
    shorter, unambiguous and portable.
    """
    from silk.simulation.variation import is_declaration
    from silk.exceptions import NotSupportedError
    if is_declaration(value):
        raise NotSupportedError('SPICE value: statistical declarations are behavioural-only in v1')
    if isinstance(value, str):
        return value
    if value == 0:
        return '0'

    negative = value < 0
    abs_val = abs(value)

    if _PLAIN_LO <= abs_val < _PLAIN_HI:
        formatted = f'{abs_val:g}'
        return f'-{formatted}' if negative else formatted

    for threshold, suffix in _SUFFIXES:
        if abs_val >= threshold:
            formatted = f'{abs_val / threshold:g}{suffix}'
            return f'-{formatted}' if negative else formatted

    formatted = f'{abs_val:g}'
    return f'-{formatted}' if negative else formatted


# THE ONE SPICE-number parse table in silk (architecture.md invariant I16;
# tests/test_spice_number_contract.py fails if another file grows a copy).
# Longest-first: 'MEG' and 'MIL' must beat 'M', or 1Meg reads as 1 milli.
# Rows follow ngspice's scale-factor table (manual §2.1.3): T G Meg K mil m u
# n p f a. `_fmt` never emits 'mil' or 'a'; they are here so a hand-written
# string parses to the number the simulator will use.
_PARSE_SCALES = (
    ('MEG', 1e6), ('MIL', 25.4e-6), ('T', 1e12), ('G', 1e9), ('K', 1e3),
    ('M', 1e-3), ('U', 1e-6), ('N', 1e-9), ('P', 1e-12), ('F', 1e-15),
    ('A', 1e-18),
)

# ASCII letters only, on purpose: see "not accepted" in _parse_si's contract.
_NUMBER_RE = re.compile(
    r'^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z]*)$')


def _parse_si(value) -> float:
    """Parse a SPICE-style value to a float — the inverse of :func:`_fmt`.

    Public entry point: ``silk.utils.parse_spice_value``, the parse
    counterpart of ``silk.utils.format_spice_value`` (which aliases
    :func:`_fmt`).

    This is silk's one SPICE-number parser. ``silk.db.circuit_import``
    (reading foreign netlists), ``silk.simulation.analysis`` and ``silk.sources.waveforms`` all
    call it; none keeps a suffix table of its own.

    Numbers pass through; strings may carry an SI suffix::

        _parse_si(1e-9)    -> 1e-9        _parse_si('1n')    -> 1e-9
        _parse_si('100p')  -> 1e-10       _parse_si('1Meg')  -> 1e6
        _parse_si('2.5')   -> 2.5         _parse_si('1.5e-6')-> 1.5e-6

    **The contract** is agreement with the simulator: a magnitude computed here
    matches what ngspice computes from the same string, because silk emits
    user-supplied strings into the deck verbatim (``Pulse(tr='1n')`` is written
    as ``1n``). Concretely:

    * The result is ``float(mantissa) * scale`` in double precision, the same
      product the simulator forms, so it is *not* always the shortest double
      for the typed decimal: ``'10u'`` is ``9.999999999999999e-06`` and
      ``'7n'`` is ``7.000000000000001e-09``. The rejected alternative is
      shortest-repr construction (``float('10e-6')`` == ``1e-05``); it would
      make silk differ from the simulator by one ULP on a value the user typed
      and cannot cover ``mil`` (25.4e-6, not a power of ten). Display is
      unaffected: ``_fmt(_parse_si('10u')) == '10u'``. A caller that shows a
      parsed float to a person must format it (``_fmt``), not print it raw.
      ``tests/test_spice_number_contract.py`` verifies this contract.
    * Suffixes are ngspice's, case-insensitive: ``T G Meg K mil m u n p f a``
      (``mil`` = 25.4e-6, ``a`` = atto). ``M``/``m`` is *always* milli; mega
      is spelled ``Meg``.
    * Letters after the suffix are ignored (``'2.56us' == '2.56u'``,
      ``'1ns' == '1n'``, ``'10kHz' == '10k'``).
    * A trailing letter that is not a suffix is likewise ignored, not
      rescaled (``'1s'`` is 1, ``'5V'`` is 5).
    * Whitespace around the number, and between number and suffix, is
      allowed (``'  4  '``, ``'1 n'``).
    * **Not accepted:** the Unicode micro sign (``'1µ'``). The simulator would
      not read it either, so accepting it here would make silk's behavioral
      twin disagree with the netlist by 1e6. Spell micro ``u``.
    * **Failure mode:** anything that holds no parseable number — ``''``,
      ``'abc'``, ``'n'``, ``None``, ``'{2*w}'`` — raises ``ValueError``.
      Callers that must stay lenient (the foreign-netlist importer) catch it
      and say so; nothing returns ``None`` silently.

    **Display is the other direction.** Plot-axis labels
    (``silk/circuit/modules_results.py`` ``_PREFIXES``) and the waveform
    explorer's readouts (``view.js`` ``Fmt``) write SI for humans: ``M`` for
    mega, ``µ`` for micro. Those strings are display-only and must never be
    fed back into this parser — ``'2.5 MHz'`` would read as 2.5 milli. The
    explorer's own ``parseSI`` (typed cursor input, never netlist text)
    refuses an uppercase ``M``/``P`` for that reason, and — unlike this
    function — refuses unknown trailing letters, so a typo does not become a
    cursor position. ``tests/test_spice_number_contract.py`` holds all of
    the above.
    """
    if not isinstance(value, str):
        # Anything float-able passes through: int, float, numpy scalars
        # (np.int64 is not an int subclass), Decimal, ...
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f'cannot parse {value!r} ({type(value).__name__}) as a number'
            ) from None

    s = value.strip()
    try:
        return float(s)                       # '0', '1.5e-9', '-3'
    except ValueError:
        pass

    match = _NUMBER_RE.match(s)
    if match is None:
        raise ValueError(f'cannot parse {value!r} as a SPICE value')
    mantissa, tail = match.group(1), match.group(2).upper()

    scale = 1.0
    for suffix, factor in _PARSE_SCALES:
        if tail.startswith(suffix):
            scale = factor
            break
    return float(mantissa) * scale


def _resolve_value(val, who=None):
    """Resolve a variable-like value to the number it currently stands for.

    THE ONE variable-resolution point in silk. A
    :class:`~silk.circuit.variable.Variable` — and any ``_VarExpr`` built
    from one, e.g. ``vampl * 2`` — is duck-typed by its ``.value`` property and
    replaced with that value; **anything else is returned unchanged**, so plain
    numbers and SPICE value strings pass through untouched.

    Both emission paths call it, which is what keeps them from drifting apart:

    * device parameters and a source's ``dc=`` / ``ac=``, through
      :func:`_render_param`;
    * every numeric parameter of every waveform class in
      :mod:`silk.sources.waveforms`, through their own ``to_spice`` /
      ``to_numpy``.

    So a ``Variable`` on a stimulus amplitude tracks
    ``silk.optimize.VariableSpace.apply()`` exactly as one on a device width does —
    Previously, the waveform classes stringified the object itself and
    wrote ``SIN(0.75 Variable(name='vampl', value=0.02) 2.7e7 0 0)`` into the
    deck, which ngspice rejects as an undefined parameter.

    Formatting is deliberately *not* part of this function. The device path
    formats through :func:`_fmt` (SI suffixes) and the waveform classes format
    through ``str()``; sharing the resolution but not the rendering is what
    lets waveform decks stay byte-identical to the ones silk wrote before
    variables were resolvable there.

    Parameters
    ----------
    val : object
        A ``Variable``/``_VarExpr``, a number, or a string.
    who : str, optional
        What to name in the error message when the variable has no value —
        e.g. ``"Sine vampl"``. Omitted for the generic device-parameter case.

    Raises
    ------
    ValueError
        If a variable-like value has no value assigned yet (``.value`` is
        ``None``).
    """
    from silk.simulation.variation import is_declaration
    from silk.exceptions import NotSupportedError
    if is_declaration(val):
        raise NotSupportedError(
            f"{who or 'SPICE parameter'}: statistical declarations are behavioural-only in v1")
    if hasattr(val, 'value'):  # Variable / _VarExpr duck-type
        v = val.value
        if v is None:
            subject = f'{who} has' if who else 'Parameter has'
            raise ValueError(
                f'{subject} no value assigned (Variable.value is None)'
            )
        return v
    return val


def _actual_values(obj, who=None):
    """*obj* with every Variable and Variable expression replaced by its value.

    Stored and reported records hold the number a run used, never the
    Variable: a sweep changes a Variable run by run, and a
    record that kept the object (or its expression text, e.g. ``2*r_load``)
    could not tell two design points apart. This is the record-side
    companion of :func:`_resolve_value`, which emission uses; each Variable
    found is resolved through it.

    Walks dicts, lists and tuples (a new container is returned; the
    caller's is never changed; a container that holds itself is returned
    as it is). Anything else that is not a Variable or an
    expression is returned unchanged. Unlike :func:`_resolve_value` it
    recognises Variables by type, not by a ``.value`` attribute, because a
    record can legitimately hold other objects that have one.

    Raises
    ------
    ValueError
        If a Variable has no value assigned.
    """
    # Imported here: silk.circuit imports this module.
    from silk.circuit.variable import Variable, _VarExpr

    def walk(o, active):
        if isinstance(o, (Variable, _VarExpr)):
            return _resolve_value(o, who)
        if not isinstance(o, (dict, list, tuple)):
            return o
        if id(o) in active:
            # A container holding itself is left as it is, so the
            # record's own validation reports the cycle.
            return o
        active.add(id(o))
        try:
            if isinstance(o, dict):
                return {k: walk(v, active) for k, v in o.items()}
            if isinstance(o, list):
                return [walk(v, active) for v in o]
            return tuple(walk(v, active) for v in o)
        finally:
            active.discard(id(o))

    return walk(obj, set())


def _render_param(val, scale=1.0, *, who='parameter value') -> str:
    """Format one device parameter value for SPICE output.

    THE ONE device-parameter renderer in silk. Both build paths call
    it — the factory API from ``AnalogElement.to_spice`` (re-exported as
    ``silk.elements.factory._render_param``) and the legacy ``add_element``
    path from :meth:`BasePDK.generate_netlist_entry` — so the same device built
    either way emits byte-identical parameter text. They diverged once: the
    legacy path formatted only the values it had unit-scaled and fell back to
    ``str()`` for the rest, so ``w=4.0`` there was ``w=4`` on the factory path.

    Handles plain numbers, strings, and ``Variable`` objects — the
    latter through :func:`_resolve_value`, the one resolution point shared with
    the waveform classes.

    *scale* takes the caller's value to the PDK's own unit before formatting —
    1.0 unless the PDK was loaded with a ``user_units`` the parameter's prefix
    does not match. A string value passes through unscaled: it is already a
    SPICE literal (``'1u'``, or a parameter reference), not a quantity.

    A string is refused if it holds a line break: it is written
    into one card, and a break would start a new deck line. Checked after
    resolution, so a Variable's current value and a value set after
    construction are both covered. *who* names the instance and field in the
    message.

    Raises
    ------
    ValueError
        If a ``Variable`` has no value assigned yet.
    DeckTextError
        If the value is a string holding a line break.
    """
    from silk.utils.deck_text import card_text
    val = _resolve_value(val)   # Variable / _VarExpr -> its value
    if isinstance(val, (int, float)):
        return _fmt(val * scale)
    # The text actually written, after str(): an object whose string form
    # holds a break is refused too.
    return card_text(val, who=who)


def render_pdk_device_line(prefix, name, pins, pdk_device, params) -> str:
    """The ONE PDK device-line writer.

    ``<prefix><name> <pin nets…> <pdk_device> <p>=<v> …`` with a trailing
    newline. Both device representations end here —
    ``AnalogElement.to_spice`` (the factory API) and
    ``BasePDK.generate_netlist_entry`` (``add_element(pdk.cls, ...)`` and the
    old-style ``PDKDevice``) — so the line shape cannot drift between them.
    Each caller keeps its own validation; this function only spells the line.

    *pins* is the ordered node tokens, already validated. *params* is an
    ordered iterable of ``(name, value, scale)`` rendered with
    :func:`_render_param`.
    """
    from silk.circuit.emitted_names import emitted_component
    from silk.utils.deck_text import card_text
    # The catalog's own strings too: a model_list handed to BasePDK is
    # caller data (TASK-593). Each is checked as the text written.
    tokens = [card_text(prefix, who=f'{name!r} SPICE prefix')
              + emitted_component(name), *pins,
              card_text(pdk_device, who=f'{name!r} PDK device name')]
    tokens.extend(
        f'{card_text(p, who=f"{name!r} parameter name")}='
        f'{_render_param(v, sc, who=f"{name!r} parameter {p!r}")}'
        for p, v, sc in params)
    return ' '.join(tokens) + '\n'
