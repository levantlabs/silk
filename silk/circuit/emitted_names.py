"""Single source of truth for the SPICE instance name silk emits per element.

Current-annotation extraction matches rawfile ``@``-current
vectors by the emitted leaf name, and must not re-derive the naming rules the
netlister/factory already own. Designer array brackets become angle brackets
in instance components only; ports, nets and designer identity are unchanged.
Literal designer angles are reserved, so the mapping cannot collide. Kinds:

- PDK / ideal device (``AnalogElement``): ``_spice_prefix`` + name (``'XM'+name``
  for PDK subckt devices, ``'D'+name`` for a catalog ``spice_prefix: D`` diode).
- legacy ``add_element`` PDK device (``PDKDevice``): ``.spice_prefix`` + name —
  the same catalog key, carried on the dataclass.
- source (``SourceBase``): ``.instance_name`` (``'V'``/``'I'`` + name).
- passive (R/C/L): the type letter + name.
- submodule (``Module``): ``'x'`` + name (``modules_netlist.py`` emission).
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from silk.spice_names import spice_component


_generated_instance_tokens = ContextVar('_generated_instance_tokens', default=frozenset())
_generated_net_tokens = ContextVar('_generated_net_tokens', default=frozenset())


@contextmanager
def _generated_token_scope(*, instances=(), nets=()):
    """Allow exact generated tokens during one fault object's construction/emission.

    The two immutable sets replace any outer permission until this scope
    exits. All name grammar checks still apply; net declarations and formal
    port names never receive an exception to the reserved-prefix rule.
    """
    instance_token = _generated_instance_tokens.set(frozenset(instances))
    try:
        net_token = _generated_net_tokens.set(frozenset(nets))
        try:
            yield
        finally:
            _generated_net_tokens.reset(net_token)
    finally:
        _generated_instance_tokens.reset(instance_token)


@dataclass(frozen=True)
class ElementNames:
    """Designer identity, readable current probe, and emitted card tokens.

    ``card_names`` preserves emission case; ngspice rawfile names are lower
    case. ``current_probe`` is None when no supported branch-current readback
    exists, and does not promise availability for every analysis/backend.

    Parameters
    ----------
    designer_name : str
        The name the designer gave the element.
    current_probe : str or None
        Rawfile probe such as ``'i(vin)'``, or ``None``.
    card_names : tuple of str
        Emitted SPICE card tokens, one per physical card.

    Examples
    --------
    >>> from silk.codegen import source_element_names
    >>> entry = {'type': 'VoltageSource', 'instance': 'vin',
    ...          'positive': 'a', 'negative': 'b', 'dc': 1.0}
    >>> names, = source_element_names(entry)
    >>> names.designer_name, names.current_probe, names.card_names
    ('vin', 'i(vin)', ('Vvin',))
    """
    #: The name the designer gave the element.
    designer_name: str
    #: Branch-current probe such as ``'i(vin)'``, or ``None`` when unsupported.
    current_probe: str | None
    #: Emitted SPICE card tokens, in emission case.
    card_names: tuple[str, ...]


def _source_instance_name(source_type, name: str) -> str:
    """Class-level naming without constructing a source or resolving values."""
    return source_type._spice_prefix + emitted_component(name)


def _source_current_probe(source_type, name: str) -> str | None:
    # The existing logical lookup treats ':' as a hierarchy separator,
    # so it cannot select a source whose local designer name contains one.
    if ':' in name:
        return None
    # Local import: source classes use the shared naming functions here.
    from silk.sources.classes import (
        VoltageSource, CurrentSource, VoltageControlledVoltageSource,
        CurrentControlledVoltageSource, BehavioralVoltage,
    )
    if issubclass(source_type, (VoltageSource, CurrentSource,
                               VoltageControlledVoltageSource,
                               CurrentControlledVoltageSource, BehavioralVoltage)):
        return f'i({name})'
    return None


def element_names(element) -> ElementNames:
    """Report one built element's designer name and every emitted SPICE card.

    Source waveforms include both physical cards; their probe reads the logical
    source current. Other devices without a readable branch have a None probe.
    Custom PassiveBase subclasses supply physical_instance_names() shared with
    their emitter; subclasses without that declaration are refused.
    This call neither emits a deck nor evaluates device/waveform values.

    Parameters
    ----------
    element : object
        A built element.

    Returns
    -------
    ElementNames
        Designer name, current probe (``'i(<card>)'`` or ``None``) and the
        emitted card tokens.

    Raises
    ------
    TypeError
        If ``element`` is none of the supported kinds.
    CircuitError
        If a passive's ``physical_instance_names()`` is not a nonempty tuple
        of unique, whitespace-free strings.
    """
    from silk.circuit.device_view import _inspection_device
    element = _inspection_device(element)
    # Local imports avoid cycles with source/device initialization.
    from silk.sources.classes import SourceBase
    from silk.elements.base import AnalogElement
    if isinstance(element, SourceBase):
        return ElementNames(element.name, _source_current_probe(type(element), element.name),
                            element.physical_instance_names())
    if isinstance(element, AnalogElement):
        card = emitted_instance_name(element, element.name)
        # Devices are not in the source-name lookup; use the existing raw
        # branch path for SPICE voltage-defined devices and inductors.
        prefix = getattr(element, '_spice_prefix', None)
        probe = f'i({card})' if prefix in ('V', 'E', 'H', 'L') else None
        return ElementNames(element.name, probe, (card,))
    from silk.elements.passives import PassiveBase
    if isinstance(element, PassiveBase):
        from silk.exceptions import CircuitError
        cards = element.physical_instance_names()
        if (not isinstance(cards, tuple) or not cards or
                any(not isinstance(c, str) or not c or any(ch.isspace() for ch in c) for c in cards) or
                len({c.lower() for c in cards}) != len(cards)):
            raise CircuitError(f"Passive {element.name!r}: physical_instance_names() must "
                               "declare a nonempty tuple of unique SPICE card tokens without whitespace.")
        return ElementNames(element.name, None, cards)
    raise TypeError('element_names requires a built source, AnalogElement or PassiveBase; '
                    f'got {type(element).__name__}.')


def source_element_names(entry: dict) -> tuple[ElementNames, ...]:
    """Expand one saved ``sources[]`` entry into element-name records.

    Computes the names without constructing silk elements or emitting a deck.
    A scalar entry gives one record; an array gives one per created element,
    excluding the bare list name.

    Parameters
    ----------
    entry : dict
        A ``sources[]`` entry in the format of
        ``docs/codegen-contract.md`` (independent, controlled or
        behavioral source).

    Returns
    -------
    tuple of ElementNames
        One record per element the entry creates, in index order.

    Raises
    ------
    ValueError
        The entry is not a valid source entry: not a dict, unknown type,
        invalid instance name or terminal, or a variable reference (testbench
        specs do not support them). The exception raised is
        ``TestbenchSpecError``, a ``ValueError`` subclass defined in
        ``silk.codegen.testbench_generator``.

    Examples
    --------
    >>> from silk.codegen import source_element_names
    >>> entry = {'type': 'VoltageSource', 'instance': 'v', 'array': 2,
    ...          'positive': 'a', 'negative': 'b', 'dc': 1.0}
    >>> [n.card_names for n in source_element_names(entry)]
    [('Vv<0>',), ('Vv<1>',)]
    """
    # Local import keeps the circuit layer loadable before codegen.
    from silk.codegen.source_names import _source_element_names
    return _source_element_names(entry)


def validate_instance_name(name: str) -> None:
    """Validate an authored instance name before construction or registration.

    Parameters
    ----------
    name : str
        Designer instance component; square brackets denote array indices.

    Returns
    -------
    None
        The name passes the instance reservation checks.

    Raises
    ------
    CircuitError
        The name contains the case-insensitive reserved ``__silk_`` prefix
        anywhere, or violates the ASCII component and numeric index rule.

    Notes
    -----
    This is the same check used by silk's source, device and placement paths.
    It does not validate net names; use :func:`validate_net_name` for those.
    """
    if (isinstance(name, str) and '__silk_' in name.lower()
            and name not in _generated_instance_tokens.get()):
        from silk.exceptions import CircuitError
        raise CircuitError(
            f"Instance name {name!r} contains the reserved __silk_ prefix. "
            "The __silk_ prefix is reserved for silk-generated names; "
            "rename this instance.")
    from silk.spice_names import validate_authored_name
    try:
        validate_authored_name(name, kind='instance', who='Instance name')
    except ValueError as exc:
        from silk.exceptions import CircuitError
        raise CircuitError(str(exc)) from None


def emitted_component(name: str) -> str:
    """Translate ONE designer instance component, never a net or full path."""
    validate_instance_name(name)
    return spice_component(name)


def emitted_submodule_name(name: str) -> str:
    """The X token shared by placement emission and hierarchical readback."""
    return 'x' + emitted_component(name)


def emitted_instance_name(element, name: str) -> str:
    from silk.circuit.device_view import _inspection_device
    element = _inspection_device(element)
    from silk.sources.classes import SourceBase
    from silk.elements.passives import Resistor, Capacitor, Inductor
    from silk.elements.base import AnalogElement
    from silk.pdk.basepdk import PDKDevice

    if isinstance(element, SourceBase):
        return element.instance_name                 # 'V'/'I' + name
    name = emitted_component(name)
    if isinstance(element, Resistor):
        return 'R' + name
    if isinstance(element, Capacitor):
        return 'C' + name
    if isinstance(element, Inductor):
        return 'L' + name
    if isinstance(element, PDKDevice):
        # add_element(factory, name, nets=…) stores a PDKDevice; the PDK emits
        # it as spice_prefix+name (basepdk.PDKDevice.generate_netlist_entry),
        # where spice_prefix is the catalog's declaration (default 'XM').
        return element.spice_prefix + name
    if isinstance(element, AnalogElement):
        return getattr(element, '_spice_prefix', 'X') + name
    # Submodule Module instance (or any subckt-emitted element).
    return 'x' + name


def validate_net_name(name: str) -> None:
    """Validate a designer net name using the same rule as ``Module.add_net``.

    Parameters
    ----------
    name : str
        An ASCII identifier with an optional trailing numeric index, or ground
        ``0``. It must not begin with the case-insensitive ``__silk_`` prefix.

    Returns
    -------
    None
        The name passes the net token checks.

    Raises
    ------
    ValueError
        The name violates the ASCII identifier rule (including empty names,
        whitespace, punctuation, invalid indices or leading digits except
        ground ``0``), or starts with the reserved ``__silk_`` prefix.
    TypeError
        The value is not a string or a Net-like object with a string name.

    Notes
    -----
    Delegates to the shared net-token validator with the ``add_net`` context,
    preserving its messages. Pin validation uses that same rule with its own
    caller context. This does not apply the instance-name occurrence rule.
    """
    from silk.elements.base import validate_net_token
    validate_net_token(name, who='add_net', kind='net')
