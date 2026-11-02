"""Small declarative extensions to PDK instance emission.

Presence is separate from unit provenance: optional values absent from a
factory's params are defaults; legacy constraints carry an explicit presence
bit for overrides already expressed in PDK units.
"""
import math
from numbers import Real

from silk.exceptions import CircuitError, NotSupportedError
from silk.utils import require_python_identifier


def refuse_derived_inputs(names, spec, instance):
    """Refuse caller values for catalog-derived destinations."""
    for name in names:
        if name in spec.get('derivedParams', {}):
            raise TypeError(f'{instance}: {name!r} is derived from '
                            f'{spec["derivedParams"][name]["from"]!r}; '
                            'it cannot be assigned by the caller.')


def validate_required_parameters(parameters, spec, instance):
    """Require every catalog parameter except explicitly optional entries."""
    for name, rule in spec.get('paramList', {}).items():
        if name not in parameters and not rule.get('emit_if_supplied'):
            raise CircuitError(f'{instance}: {name!r} is required; '
                               'restore its value before emission.')


class ParameterValues(dict):
    """Parameter mapping that reserves catalog-derived names."""

    def __init__(self, values=(), derived=()):
        self._derived = frozenset(derived)
        self.update(values)

    def _check(self, names):
        for name in names:
            if name in self._derived:
                raise TypeError(f'{name!r} is derived; it cannot be assigned by the caller.')

    def __setitem__(self, key, value):
        self._check((key,))
        super().__setitem__(key, value)

    def update(self, values=(), **kwargs):
        incoming = dict(values, **kwargs)
        self._check(incoming)
        super().update(incoming)

    def setdefault(self, key, default=None):
        self._check((key,))
        return super().setdefault(key, default)

    def __ior__(self, values):
        self.update(values)
        return self

    def __reduce__(self):
        return (type(self), ((), self._derived), None, None, iter(self.items()))


def parameter_values(values, spec):
    """Protect mappings only where the catalog declares derived names."""
    derived = spec.get('derivedParams', {})
    return ParameterValues(values, derived) if derived else values


def validate_emission_spec(name, spec):
    """Reject invalid derived destinations, references and metadata at load time."""
    params = spec.get('paramList') or {}
    derived = spec.get('derivedParams') or {}
    if not isinstance(derived, dict):
        raise ValueError(f'{name}: derivedParams must be a mapping')
    for dest, rule in derived.items():
        require_python_identifier(dest, f'{name}: derivedParams destination')
        if dest in params or dest in (spec.get('paramAliases') or {}):
            raise ValueError(f'{name}: derivedParams destination {dest!r} collides with a caller parameter')
        if (not isinstance(rule, dict) or set(rule) != {'from', 'omit_when'}
                or rule['from'] not in params):
            raise ValueError(f'{name}: derivedParams {dest!r} needs a declared source and omit_when')
        if params[rule['from']].get('emit_if_supplied'):
            raise ValueError(f'{name}: derivedParams {dest!r} source must always be emitted')
        if not isinstance(rule['omit_when'], Real) or isinstance(rule['omit_when'], bool) or not math.isfinite(rule['omit_when']):
            raise ValueError(f'{name}: derivedParams {dest!r} omit_when must be finite numeric')
    for param, rule in params.items():
        for flag in ('emit_if_supplied', 'strict'):
            if flag in rule and not isinstance(rule[flag], bool):
                raise ValueError(f'{name}: {param} {flag} must be boolean')
        for field in ('simulator_options', 'backend_max'):
            if field in rule and not isinstance(rule[field], dict):
                raise ValueError(f'{name}: {param} {field} must be a mapping')
        for backend, maximum in (rule.get('backend_max') or {}).items():
            if not isinstance(maximum, Real) or isinstance(maximum, bool) or not math.isfinite(maximum):
                raise ValueError(f'{name}: {param} backend_max must be finite numeric')
        for backend, options in (rule.get('simulator_options') or {}).items():
            from silk.simulators import load_simulator_config
            load_simulator_config(backend).emit_options_line(options)
    return {'paramList': params, 'derivedParams': derived}


def resolved(value):
    return value.value if hasattr(value, 'value') else value


def validate_strict(value, param, rule, instance):
    """Opt-in finite integer validation; SPICE literals retain their semantics."""
    if not rule.get('strict'):
        return
    value = resolved(value)
    if isinstance(value, str):
        return
    minimum, maximum = float(rule['minval']), float(rule['maxval'])
    if (not isinstance(value, Real) or isinstance(value, bool)
            or not math.isfinite(value)
            or (minimum != -1 and value < minimum)
            or (maximum != -1 and value > maximum)
            or not float(value).is_integer()):
        raise CircuitError(f'{instance}: invalid {param}={value!r}; expected a finite integer '
                           f'in catalog range [{rule["minval"]}, {rule["maxval"]}] (-1 means unbounded).')


def parameter_state(device):
    """Return effective parameter values and explicitly present names."""
    if hasattr(device, 'params'):
        defaults = getattr(type(device), '_param_defaults', {})
        return {**defaults, **device.params}, set(device.params)
    values, present = {}, set()
    for name, c in device.param_list.items():
        values[name] = c.default if c.value is None else c.value
        if c.value is not None or c.emission_set:
            present.add(name)
    return values, present


def emission_spec(device):
    return getattr(device, '_emission_spec', {})


def collect_requirements(device, backend, options):
    """Collect default deck options and enforce parameter backend limits."""
    rules = emission_spec(device).get('paramList', {})
    if not rules:
        return
    values, present = parameter_state(device)
    for param, rule in rules.items():
        if param not in present:
            continue
        value = resolved(values[param])
        limits = rule.get('backend_max') or {}
        if backend in limits and (isinstance(value, bool) or not isinstance(value, Real)
                                  or not math.isfinite(value) or value > limits[backend]):
            raise NotSupportedError(f'{getattr(device, "name", "device")}: {param}={value!r} '
                                    f'is unsupported on {backend}; requires a proven value <= '
                                    f'{limits[backend]}. Use ngspice for this parameter.')
        options.update((rule.get('simulator_options') or {}).get(backend, {}))


def wrapper_defaults(device):
    """Neutral optional/derived defaults, independent of instance order."""
    spec = emission_spec(device)
    result = {p: rule.get('default', rule.get('minval'))
              for p, rule in spec.get('paramList', {}).items()
              if rule.get('emit_if_supplied')}
    result.update({p: rule['omit_when'] for p, rule in spec.get('derivedParams', {}).items()})
    return result
