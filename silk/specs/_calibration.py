"""Stage a complete calibration transaction before changing model parameters."""
from copy import deepcopy
from dataclasses import replace

from silk.circuit.variable import Variable, _VarExpr
from silk.exceptions import ConfigurationError, NotSupportedError, VariationError
from silk.simulation.variation import Normal, is_declaration


def apply_calibrations(candidates, log, *, root=None):
    groups = {}
    variables = {}
    memo = {}
    for owner, spec, measurement in candidates:
        binding = spec.calibration
        params = binding.target_module._behavioral_params
        original = params[binding.param_name]
        fields = ('nominal', 'sigma') if isinstance(original, Normal) else ('low', 'high')
        if is_declaration(original):
            if binding.field not in fields:
                raise VariationError(
                    f'{binding.param_name}: CalibrationBinding(field=...) requires one of {fields}')
            target = getattr(original, binding.field)
        else:
            if binding.field is not None:
                raise VariationError(f'{binding.param_name}: field needs a statistical declaration')
            target = original
        if is_declaration(original) and isinstance(target, _VarExpr):
            raise NotSupportedError(f'{binding.param_name}.{binding.field}: cannot calibrate a Variable expression')
        if is_declaration(original) and isinstance(target, Variable):
            if id(target) in variables:
                raise ConfigurationError(f'Two calibration bindings target Variable {target.name!r}; no parameters changed')
            # Variable deliberately preserves identity through copy/deepcopy.
            # A private validation object needs independent assignment state.
            prospective = object.__new__(type(target))
            prospective.__dict__ = target.__dict__.copy()
            prospective.value = measurement.value
            variables[id(target)] = (target, prospective.value)
            memo[id(target)] = prospective
        key = (id(binding.target_module), binding.param_name)
        group = groups.setdefault(key, (params, binding.param_name, original, {}, binding.target_module))
        group[3][binding.field] = (measurement.value, target)

    replacements = []
    # Compose all fields together. Deepcopy substitutes prospective Variables
    # (also inside expressions) without ever writing to a live design knob.
    for params, name, original, updates, module in groups.values():
        if is_declaration(original):
            numbers = {f: value for f, (value, target) in updates.items()
                       if not isinstance(target, Variable)}
            prospective = deepcopy(original, memo)
            replace(prospective, **numbers).resolve()
            if numbers:
                # Construct after Variable writes: current live limits may be
                # invalid in combination with the new numeric fields.
                replacements.append((params, name, original, numbers))
        else:
            value, target = updates[None]
            replacements.append((params, name, value, None))
    # Validate only declarations affected by staged Variables, including
    # unbound siblings reachable from the reconciliation root. Unrelated,
    # unresolved design knobs must not block a plain scalar calibration.
    if variables:
        roots = [g[4] for g in groups.values()]
        if root is not None:
            roots.append(root)
        modules = {}
        from silk.circuit.module_placement import walk_instances
        for module in roots:
            if hasattr(module, '_element_mgr'):
                modules.update((id(item.module), item.module)
                               for item in walk_instances(module, include_self=True))
            else:
                modules[id(module)] = module
        for module in modules.values():
            for name, declaration in (getattr(module, '_behavioral_params', None) or {}).items():
                if not is_declaration(declaration) or (id(module), name) in groups:
                    continue
                fields = ('nominal', 'sigma') if isinstance(declaration, Normal) else ('low', 'high')
                if any(_depends_on(getattr(declaration, field), variables) for field in fields):
                    deepcopy(declaration, memo).resolve()

    for variable, value in variables.values():
        variable.value = value
    for params, name, original, numbers in replacements:
        params[name] = replace(original, **numbers) if numbers is not None else original
    for _, spec, measurement in candidates:
        binding = spec.calibration
        suffix = '' if binding.field is None else f'.{binding.field}'
        log.append(f'{binding.target_module.module_name}.{binding.param_name}{suffix} '
                   f'= {measurement.value} (from {spec.name}, source={measurement.source})')


def _depends_on(value, variables):
    if isinstance(value, Variable):
        return id(value) in variables
    if isinstance(value, _VarExpr):
        return any(_depends_on(operand, variables) for operand in value._ops)
    return False
