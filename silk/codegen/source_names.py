"""Saved-source identity using the same names and layout as built elements."""
from silk.circuit.emitted_names import (
    ElementNames, _source_current_probe, _source_instance_name,
    validate_instance_name,
)
from silk.exceptions import CircuitError


def _source_element_names(entry: dict) -> tuple[ElementNames, ...]:
    # Local imports: circuit/source packages load before codegen during silk's
    # package initialization; importing the generator earlier would cycle.
    from silk import sources
    from silk.codegen.generator import (
        _source_entry_to_spec, _source_value_slots, _waveform_kind,
        _TEMPLATE_RE, _CONTROLLED_SOURCE_PINS,
        _BEHAVIORAL_SOURCE_TYPES,
    )
    from silk.codegen.testbench_generator import (
        TestbenchSpecError, _entry_names, _reject_variable_refs,
    )
    from silk.elements.base import validate_net_token
    from silk.sources._layout import WAVEFORM_SLOTS, source_card_layout
    from silk.utils import require_python_identifier

    if not isinstance(entry, dict):
        raise TestbenchSpecError(f'sources entry must be an object, got {entry!r}')
    label = f"sources entry {entry.get('instance', '?')!r}"
    try:
        spec = _source_entry_to_spec(entry, label)
        name = spec.instance
        if not isinstance(name, str) or not name or any(c.isspace() for c in name):
            raise ValueError(f'{label}: instance must be a nonempty name without whitespace')
        validate_instance_name(name)
        cls = getattr(sources, spec.type)
        if (spec.array is not None or spec.type in _CONTROLLED_SOURCE_PINS
                or spec.type in _BEHAVIORAL_SOURCE_TYPES):
            require_python_identifier(name, 'source instance name')
        for pin in cls._node_terminals:
            value = getattr(spec, pin)
            if spec.array is not None and isinstance(value, str):
                # Names do not need the pin index itself. Replace only tokens
                # accepted by the generator, including their internal spaces,
                # to validate a physical net token without evaluating Python.
                value = _TEMPLATE_RE.sub('0', value)
            validate_net_token(value, who=f'{label} {pin}', kind='terminal')
        _reject_variable_refs(label, _source_value_slots(spec))
        state = {}
        if spec.pwl is not None:
            state['pwl'] = spec.pwl
        elif spec.waveform is not None:
            kind = _waveform_kind(spec.waveform['type'])
            state[kind if kind in WAVEFORM_SLOTS else 'waveform'] = spec.waveform
        names = _entry_names(entry, label)
        if spec.array is not None:
            names = names[1:]
        return tuple(
            ElementNames(n, _source_current_probe(cls, n),
                         source_card_layout(n, _source_instance_name(cls, n), state).names)
            for n in names)
    except (KeyError, TypeError, ValueError, CircuitError) as exc:
        raise TestbenchSpecError(f'invalid {label}: {exc}') from None
