"""Shared strict-JSON value encoding, independent of silk's runtime imports.

NaN and signed infinities retain their numeric meaning in Python; only the
wire representation changes. Use ``encode_json`` before every JSON dump and
``decode_json`` after reading a payload written with it. See docs/reports.md.
"""

import math


def _tag_kind(value, *, complex_values=False):
    """Recognize only the exact reserved shapes, never surrounding records."""
    if set(value) == {'value', 'status'}:
        if value['value'] is None and value['status'] == 'unavailable':
            return 'unavailable'
        if isinstance(value['value'], dict) and value['status'] == 'literal':
            return 'literal'
    if (set(value) == {'value', 'status', 'sign'}
            and value['value'] is None and value['status'] == 'infinite'
            and value['sign'] in ('+', '-')):
        return 'infinite'
    if (complex_values and set(value) == {'status', 're', 'im'}
            and value['status'] == 'complex'):
        return 'complex'
    return None


def encode_json(value, *, finite_only=False, complex_values=False):
    """Copy JSON containers, replacing non-finite float values with tags.

    NaN becomes ``{'value': None, 'status': 'unavailable'}``; infinities use
    ``status='infinite'`` and ``sign='+'`` or ``'-'``. Literal dictionaries
    matching a tag are escaped with ``status='literal'`` so they round-trip.
    Finite values, nulls and strings are unchanged; tuples become JSON lists.
    Unsupported values remain subject to the dump site's existing ``default``
    policy. Non-finite mapping keys and circular containers raise ValueError.

    ``finite_only=True`` retains an existing caller-input validation boundary
    (such as waveform-view annotations or simulation conditions). It is not
    for measurement results, where NaN and infinity are legitimate values.
    That mode writes no tags or escapes: finite JSON shapes stay unchanged,
    and its readers use ordinary JSON decoding.
    ``complex_values=True`` opts into complex tags (real and imaginary parts
    use the same numeric codec), including literal-tag collision escaping.
    Always also pass ``allow_nan=False`` to the actual JSON dump.
    """
    active = set()

    def encode(item):
        if complex_values and isinstance(item, complex):
            return {'status': 'complex', 're': encode(item.real),
                    'im': encode(item.imag)}
        if isinstance(item, float) and not math.isfinite(item):
            if finite_only:
                raise ValueError('Out of range float values are not JSON compliant')
            if math.isnan(item):
                return {'value': None, 'status': 'unavailable'}
            return {'value': None, 'status': 'infinite',
                    'sign': '+' if item > 0 else '-'}
        if not isinstance(item, (dict, list, tuple)):
            return item
        identity = id(item)
        if identity in active:
            raise ValueError('Circular reference detected')
        active.add(identity)
        try:
            if isinstance(item, dict):
                result = {}
                for key, child in item.items():
                    if isinstance(key, float) and not math.isfinite(key):
                        raise ValueError('Non-finite JSON mapping keys are not supported')
                    result[key] = encode(child)
                if not finite_only and _tag_kind(item, complex_values=complex_values):
                    return {'value': result, 'status': 'literal'}
                return result
            return [encode(child) for child in item]
        finally:
            active.remove(identity)

    return encode(value)


def decode_json(value, *, complex_values=False):
    """Restore numeric tags recursively after JSON loading.

    ``complex_values=True`` additionally restores complex tags and validates
    their components. The default retains the original real-only contract.
    Literal wrappers decode children without reinterpreting the enclosed map.
    """
    def decode(item, path):
        if isinstance(item, list):
            return [decode(child, f'{path}[{i}]') for i, child in enumerate(item)]
        if isinstance(item, dict):
            kind = _tag_kind(item, complex_values=complex_values)
            if kind == 'unavailable':
                return float('nan')
            if kind == 'infinite':
                return float('inf') if item['sign'] == '+' else -float('inf')
            if kind == 'literal':
                return {key: decode(child, f'{path}.{key}')
                        for key, child in item['value'].items()}
            if kind == 'complex':
                parts = [decode(item[key], f'{path}.{key}') for key in ('re', 'im')]
                if any(isinstance(part, bool) or not isinstance(part, (int, float))
                       for part in parts):
                    raise ValueError(f'{path}: complex components must be real numbers')
                return complex(*parts)
            return {key: decode(child, f'{path}.{key}') for key, child in item.items()}
        return item

    return decode(value, '$')
