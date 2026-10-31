"""Source facts for persistent structure identity; no circuit reconstruction.

Values use the existing resolver and each field's emitter precision. Opaque
emitters are refused here; executable consumers select a naming-only fallback.
"""
from enum import Enum
import math
from numbers import Real

from silk.exceptions import CircuitError
from silk.utils._utils import _resolve_value, _render_param


class _UnprovableSource(CircuitError):
    """A source projection failure, distinct from unrelated structure errors."""


def _builtin_type(obj, candidates):
    """Require effective built-in behavior, including instance overrides."""
    base = next((cls for cls in type(obj).__mro__ if cls in candidates), None)
    if base is None:
        raise ValueError(f'unprovable custom emitter {type(obj).__qualname__}')
    seen = set()
    for cls in base.__mro__:
        for name, expected in vars(cls).items():
            if name.startswith('__') or name in seen:
                continue
            seen.add(name)
            if callable(expected) or isinstance(expected, (property, staticmethod, classmethod)):
                actual = next((vars(c)[name] for c in type(obj).__mro__ if name in vars(c)), None)
                if actual is not expected or name in vars(obj):
                    raise ValueError(f'unprovable custom emitter method {name!r}')
    for name in ('_node_terminals', '_spice_prefix'):
        if hasattr(base, name) and getattr(type(obj), name) != getattr(base, name):
            raise ValueError(f'unprovable custom emitter attribute {name!r}')
    return base


def _value(value, path, active=None):
    """Full field tokens, after the same Variable resolution as devices."""
    value = _resolve_value(value, path)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Enum):
        return _value(value.value, path)
    if isinstance(value, str):
        return value
    if isinstance(value, Real):
        if not isinstance(value, int) and not math.isfinite(value):
            raise ValueError(f'{path} must be finite')
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        active = set() if active is None else active
        if id(value) in active:
            raise ValueError(f'{path} contains a cycle')
        active.add(id(value))
        try:
            if isinstance(value, dict):
                if any(not isinstance(k, str) for k in value):
                    raise ValueError(f'{path} needs string field names')
                return {k: _value(v, f'{path}.{k}', active) for k, v in value.items()}
            return [_value(v, f'{path}[{i}]', active) for i, v in enumerate(value)]
        finally:
            active.remove(id(value))
    raise ValueError(f'{path} has unsupported value {type(value).__qualname__}')


def _waveform(wave, path):
    from silk.sources import waveforms as w
    candidates = (w.PWL, w.Pulse, w.Sine, w.Exponential, w.SFFM, w.AM,
                  w.TransientNoise, w.RandomSource, w.BitPattern, w.PRBS)
    base = _builtin_type(wave, candidates)
    if base is w.PWL:
        state = {'points': wave._resolved_points(), 'r': wave.r, 'td': wave.td}
    elif base in (w.BitPattern, w.PRBS):
        # Compact exact seed/order/length/bits, without expanding a bit stream.
        state = wave.as_parameters()
        wave._electrical()
    else:
        state = vars(wave)
    return {'type': type(wave).__qualname__, 'fields': _value(state, path)}


def source_entry(name, source):
    """Capture one provable source without following its owning module."""
    from silk.sources import classes as s
    from silk.sources._layout import selected_waveform, source_layout
    candidates = (s.VoltageSource, s.CurrentSource,
                  s.VoltageControlledVoltageSource, s.VoltageControlledCurrentSource,
                  s.CurrentControlledVoltageSource, s.CurrentControlledCurrentSource,
                  s.BehavioralVoltage, s.BehavioralCurrent)
    try:
        base = _builtin_type(source, candidates)
        entry = {'name': name, 'type': type(source).__qualname__,
                 'nets': {pin: source._node(pin) for pin in source._node_terminals},
                 'parameters': {}}
        if base in (s.VoltageSource, s.CurrentSource):
            for key in ('dc', 'ac'):
                value = _resolve_value(getattr(source, key), f'{name}.{key}')
                _value(value, f'{name}.{key}')  # validate before formatter's fallback
                entry['parameters'][key] = _render_param(value)
            slot, wave = selected_waveform(source.name, vars(source))
            if wave is not None:
                entry['waveform'] = _waveform(wave, f'{name}.{slot}')
                layout = source_layout(source)
                entry['layout'] = {'names': list(layout.names), 'dc_name': layout.dc_name,
                                   'internal_node': layout.internal_node,
                                   'current_terms': [list(term) for term in layout.current_terms]}
        else:
            fields = {s.VoltageControlledVoltageSource: 'gain',
                      s.VoltageControlledCurrentSource: 'gm',
                      s.CurrentControlledVoltageSource: 'transresistance',
                      s.CurrentControlledCurrentSource: 'gain',
                      s.BehavioralVoltage: 'expression', s.BehavioralCurrent: 'expression'}
            key = fields[base]
            value = getattr(source, key)
            if key == 'expression':
                # B emitters stringify literal SPICE text, not Variable values.
                # Resolving a symbolic _VarExpr here would equate different decks.
                if not isinstance(value, (str, Real)):
                    raise ValueError('expression must be literal SPICE text or a number')
                _value(value, f'{name}.{key}')
                entry['parameters'][key] = str(value)
            else:
                entry['parameters'][key] = _value(value, f'{name}.{key}')
            if hasattr(source, 'ctrl_vname'):
                reference = source._resolve_ctrl_vname(source.ctrl_vname)
                if not isinstance(reference, str):
                    raise ValueError('control-source reference must be a literal source name')
                entry['control_source'] = reference
        return entry
    except (CircuitError, ArithmeticError, ValueError, TypeError,
            AttributeError, KeyError, RecursionError) as exc:
        raise _UnprovableSource(
            f'Cannot establish persistent identity for source {name!r}: {exc}. '
            'Assign resolved supported values and use built-in source/waveform '
            'semantics before storing or reusing circuit evidence.'
        ) from exc
