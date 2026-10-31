"""Which source may a DC sweep or a noise analysis name.

A sweep names ONE source element. An arrayed source is several elements
(``vin[0]``, ``vin[1]``), so the bare array name ``vin`` is not a source and is
refused; the element name ``vin[1]`` is. Anyone who wants every element to move
together uses one source instead.

One matcher serves the testbench code generator (which checks the spec) and the
run path (which checks the built bench), so both refuse the same names in the
same words.
"""

# One leading element letter is stripped to match, so 'Vvs' and 'Vvin[1]' match
# the designer names 'vs' and 'vin[1]'. An array element's emitted SPICE
# spelling ('Vvin<1>') does not match: Python names use [k], and <k> is SPICE's.
_SOURCE_PREFIXES = 'viefghb'


def _candidates(name):
    low = str(name).lower()
    out = [low]
    if len(low) > 1 and low[0] in _SOURCE_PREFIXES:
        out.append(low[1:])
    return out


def sweep_source_problem(name, names, arrays=None, allow=()):
    """Return why ``name`` cannot be swept, or ``None`` when it can.

    ``names`` are the designer names of every source (arrayed sources already
    expanded to their elements); ``arrays`` maps an arrayed source's bare name
    to its element names. When ``arrays`` is omitted, a bare name is treated as
    an array if some element is spelled ``<name>[<k>]``. ``allow`` lists extra
    accepted words (``'temp'`` for DC).
    """
    lowered = {str(n).lower() for n in names}
    cands = _candidates(name)
    if cands[0] in {str(a).lower() for a in allow}:
        return None
    if any(c in lowered for c in cands):
        return None
    if arrays is None:
        arrays = {}
        for n in names:
            if str(n).endswith(']') and '[' in str(n):
                arrays.setdefault(str(n)[:str(n).index('[')], []).append(n)
    for base, elems in arrays.items():
        if str(base).lower() in cands:
            return (f"{name!r} is an arrayed source, and a sweep names one "
                    f"element. Valid element names: "
                    f"{', '.join(repr(e) for e in elems)}. To move all "
                    f"elements together, use one source.")
    valid = ', '.join(repr(n) for n in names) or 'none'
    extra = ''.join(f" ({a!r} is also accepted.)" for a in allow)
    return (f"{name!r} is not a source of this test. Valid source names: "
            f"{valid}.{extra}")


def missing_source_problem(names, allow=()):
    """Why an analysis that names no source is refused. A sweep or
    a noise input is never defaulted: the analysis must name its source."""
    valid = ', '.join(repr(n) for n in names) or 'none'
    extra = ''.join(f" ({a!r} is also accepted.)" for a in allow)
    return f"names no source. Valid source names: {valid}.{extra}"
