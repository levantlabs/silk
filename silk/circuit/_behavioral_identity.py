"""Deterministic behavioral identity for definition naming, never DB hashing."""

from functools import partial
import hashlib
import json
from types import CodeType

from silk._json_codec import encode_json
from silk.exceptions import CircuitError


def _value_key(value):
    """Encode values without process-local addresses or randomized hashes."""
    import numpy as np

    from silk.simulation.variation import Normal, Uniform
    from silk.circuit.variable import Variable, _VarExpr

    if isinstance(value, (Normal, Uniform)):
        fields = ((value.nominal, value.sigma) if isinstance(value, Normal)
                  else (value.low, value.high))
        return [type(value).__name__, _value_key(fields), value.variation, value.name]
    if isinstance(value, Variable):
        return ['Variable', value.name, _value_key(value.value)]
    if isinstance(value, _VarExpr):
        return ['_VarExpr', value._fn.__name__, _value_key(value._ops)]
    if value is None or value is Ellipsis:
        return [type(value).__name__]
    if isinstance(value, np.ndarray):
        dtype = value.dtype.descr if value.dtype.fields else value.dtype.str
        return ['array', _value_key(dtype), list(value.shape), _value_key(value.tolist())]
    if isinstance(value, np.generic):
        return _value_key(value.item())
    if isinstance(value, (bool, int, str)):
        return [type(value).__name__, value]
    if isinstance(value, float):
        return ['float', value.hex()]
    if isinstance(value, complex):
        return ['complex', _value_key(value.real), _value_key(value.imag)]
    if isinstance(value, bytes):
        return ['bytes', value.hex()]
    if isinstance(value, CodeType):
        # Encode fields explicitly: marshal's reference/interning flags depend
        # on object sharing. Filenames and line tables are source locations,
        # not model code.
        fields = ('co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount',
                  'co_nlocals', 'co_stacksize', 'co_flags', 'co_code',
                  'co_consts', 'co_names', 'co_varnames', 'co_freevars',
                  'co_cellvars', 'co_name',
                  'co_exceptiontable')
        return ['code', [[name, _value_key(getattr(value, name))]
                         for name in fields if hasattr(value, name)]]
    if isinstance(value, dict):
        pairs = [[_value_key(k), _value_key(v)] for k, v in value.items()]
        return ['dict', sorted(pairs, key=_dump)]
    if isinstance(value, (tuple, list, set, frozenset)):
        items = [_value_key(item) for item in value]
        if isinstance(value, (set, frozenset)):
            items.sort(key=_dump)
        return [type(value).__name__, items]
    raise TypeError(f'unsupported identity value {type(value).__name__}')


def _dump(value):
    return json.dumps(encode_json(value), ensure_ascii=True, separators=(',', ':'),
                      allow_nan=False)


def _own_model_key(module):
    """Code, explicit params and the ports form's contract; excludes captured closure values."""
    fn = getattr(module, '_behavioral_model', None)
    if fn is None:
        return None
    bound_args = ()
    bound_keywords = {}
    while isinstance(fn, partial):
        # Inner positional arguments precede outer arguments; outer keyword
        # bindings override inner ones, just as they do when the model runs.
        bound_args = fn.args + bound_args
        bound_keywords = {**fn.keywords, **bound_keywords}
        fn = fn.func
    code = getattr(fn, '__code__', None)
    if code is None:
        code = getattr(getattr(fn, '__call__', None), '__code__', None)
    try:
        if code is None:
            raise TypeError('callable has no Python code object')
        key = [_value_key(code),
               _value_key(getattr(module, '_behavioral_params', None))]
        if bound_args or bound_keywords:
            key.append([_value_key(bound_args), _value_key(bound_keywords)])
        # The calling contract distinguishes otherwise identical models.
        key.append(['ports', module._behavioral_execution])
        return key
    except (TypeError, ValueError, RecursionError) as exc:
        raise CircuitError(
            f'module_name {module.module_name!r}: cannot compare behavioral '
            f'model identity: {exc}. Use a Python model function and explicit '
            'params containing numeric values, arrays or built-in containers.'
        ) from exc


def behavioral_key(module):
    """Include descendant models so wrappers cannot merge different definitions."""
    own = _own_model_key(module)
    children = []
    for name, entry in sorted(module.elem_dict.items()):
        device = entry['device']
        if hasattr(device, 'module_dict'):
            key = behavioral_key(device)
            if key is not None:
                children.append([name, key])
    if own is None and not children:
        return None
    return [own, children]


def custom_key(module):
    """Authored ``add_custom_line`` text of *module* and its descendants.

    Raw text is opaque, so it is folded into the NAMING digest only (never the
    structure hash). ``None`` when no level carries any, so definitions without
    custom lines keep their legacy suffixes.
    """
    own = list(getattr(module, 'custom_lines', None) or [])
    children = []
    for name, entry in sorted(module.elem_dict.items()):
        device = entry['device']
        if hasattr(device, 'module_dict'):
            key = custom_key(device)
            if key is not None:
                children.append([name, key])
    if not own and not children:
        return None
    return [own, children]


def definition_hash(structure_hash, model_key, custom=None):
    """Preserve legacy suffixes exactly for definitions without a model or
    custom lines."""
    if model_key is None and custom is None:
        return structure_hash
    if custom is None:
        return hashlib.sha256(_dump([structure_hash, model_key]).encode()).hexdigest()
    return hashlib.sha256(
        _dump([structure_hash, model_key, custom]).encode()).hexdigest()
