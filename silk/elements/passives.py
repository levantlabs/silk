"""Deprecated ``Resistor``, ``Capacitor`` and ``Inductor`` constructors.

These three classes keep the older positional constructor
(``Resistor('r1', 'a', 'b', 1e3)``, with ``pos`` and ``neg`` as terminal
names). Each is a subclass of the matching generated class of
:data:`silk.elements.elements` (``resistor``, ``capacitor``, ``inductor``) and
emits a ``DeprecationWarning`` when instantiated; new code should call
``elements.resistor('r1', p='a', n='b', value=1e3)`` instead. The value is
also reachable as ``resistance``, ``capacitance`` or ``inductance``, and the
terminals as ``pos`` and ``neg``.

Inside a ``with module:`` block these classes register with the module as
passives, not as ordinary elements.
"""

import warnings

from silk.elements import elements as _ideal


def _validate_net(name, field):
    # One terminal is ONE SPICE node token. ``'a b'`` used to pass and emit
    # ``Rr1 a b gnd 1k``: ngspice-42 warned "can not find model gnd", then
    # left node ``a`` floating (singular matrix) — a warning silk never
    # surfaced. Same shape as the non-empty check, so callers matching on
    # ``'pos must be a non-empty string'`` keep matching (TASK-76).
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f'{field} must be a non-empty string, got {name!r}')
    if any(ch.isspace() for ch in name):
        raise ValueError(
            f'{field} must be a non-empty string with no whitespace, got {name!r}'
        )

    from silk.elements.base import validate_net_token
    validate_net_token(name, who=field, kind='net')


def _validate_name(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f'name must be a non-empty string, got {name!r}')

    from silk.circuit.emitted_names import validate_instance_name
    validate_instance_name(name)


def _validate_value(value, field, *, owner=None, name=None):
    # The line-break refusal first, so a value such as '\n' raises
    # DeckTextError naming the instance, not the blank-string ValueError
    # (TASK-593).
    from silk.elements.base import pin_context
    from silk.utils.deck_text import refuse_line_breaks
    refuse_line_breaks(value, who=pin_context(owner or 'passive', name, field,
                                              noun='parameter'))
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f'{field} must be a non-empty string value, got {value!r}')
    elif isinstance(value, (int, float)):
        if value < 0:
            raise ValueError(f'{field} must be >= 0, got {value}')
    else:
        raise TypeError(
            f'{field} must be a number or a SPICE value string (e.g. 1e3 or "1k"), '
            f'got {type(value).__name__!r}'
        )


class _AutoRegisterPassive(type):
    """Metaclass that auto-registers a fully-constructed passive with the
    active module (set by using a Module as a context manager, ``with mod:``),
    if any."""
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        from silk.circuit.modules import _active_module
        mod = _active_module()
        if mod is not None:
            mod.add_passive(instance)
        return instance


def _value_alias(kind: str) -> property:
    """A typed read/write alias for ``.value`` (``Resistor.resistance`` etc.).

    It must be an ALIAS, not a second copy: ``to_spice()`` reads the typed name
    while the structure hash reads ``.value``, so two independent attributes let
    an in-place edit change the simulated circuit without changing the hash —
    silently defeating net-current staleness detection (currents measured on the
    old circuit would load onto the edited one).

    Also used by the shim classes below, whose ``.value`` is a
    ``ParamDescriptor`` (an ``AnalogElement`` class attribute): reading it
    returns ``self.params['value']`` and — because
    ``AnalogElement.__setattr__`` routes any name in ``_param_names`` into
    ``self.params`` — assigning it writes ``self.params['value']`` too, so
    this really is the same one alias for both families, not a second
    implementation.
    """
    def _get(self):
        return self.value

    def _set(self, new):
        _validate_value(new, kind, owner=type(self).__name__,
                        name=getattr(self, 'name', None))
        self.value = new

    return property(_get, _set, doc=f'{kind.capitalize()} value (alias of ``value``).')


def _terminal_alias(pin: str) -> property:
    """Typed read/write alias for a terminal net ('pos' -> pin 'p', 'neg' ->
    pin 'n') on the shim classes below — parity for existing code
    that reads/writes ``r.pos``/``r.neg`` directly (``PassiveBase`` stored
    these as plain instance attributes; the ideal ``AnalogElement`` these
    classes now build stores terminals in ``self.nets``, with no bare
    ``r.p``/``r.n`` getter of its own).

    The setter goes through ``setattr(self, pin, new)`` — ``AnalogElement.
    __setattr__``'s own pin branch — so a post-construction ``r.pos =
    'new_net'`` gets exactly the same validate-then-rebind chokepoint a
    direct ``r.p = 'new_net'`` would (net revalidated, ``Net.pins``
    rewired if attached to a module).
    """
    def _get(self):
        return self.nets.get(pin)

    def _set(self, new):
        setattr(self, pin, new)

    return property(_get, _set, doc=f"Terminal net (alias of pin '{pin}').")


class PassiveBase(metaclass=_AutoRegisterPassive):
    def __init__(self, name: str, pos: str, neg: str, value):
        _validate_name(name)
        _validate_net(pos, 'pos')
        _validate_net(neg, 'neg')
        self.name  = name
        self.pos   = pos
        self.neg   = neg
        self.value = value

    def __setattr__(self, attr, value):
        if attr == 'name' and getattr(self, '_name_locked', False):
            from silk.exceptions import CircuitError
            raise CircuitError(f"Instance {self.name!r} is placed: its name is read-only. "
                               "Construct a new instance with the desired name.")
        object.__setattr__(self, attr, value)

    def physical_instance_names(self):
        """Value-independent card tokens, also used by a subclass's emitter.

        Custom subclasses must implement this protocol before placement and
        keep the declared identity stable while placed.
        """
        from silk.exceptions import NotSupportedError
        raise NotSupportedError(
            f"Passive {self.name!r} cannot be placed: its class must implement "
            "physical_instance_names() and use those same card names in to_spice(). "
            "Declare value-independent SPICE card names before placement.")

    def to_spice(self) -> str:
        raise NotImplementedError


def _deprecation_message(cls_name: str, target: str) -> str:
    return (
        f'silk.elements.passives.{cls_name} is deprecated; use '
        f'silk.elements.elements.{target} instead (same values; canonical pins '
        f"are 'p'/'n', canonical parameter is 'value')."
    )


def _init_ideal_shim(self, parent_cls, name, pos, neg, value):
    """Build the YAML element (via *parent_cls*.__init__, bypassing the
    Resistor/Capacitor/Inductor override that called this) and register it
    exactly the way these three classes always have.

    *parent_cls*'s own ``__init__`` (the factory-generated one) already
    auto-registers with any ``with mod:``-active module — but into
    ``elem_dict`` under its real pin names ('p'/'n'), since that is what a
    directly constructed ``elements.resistor(...)`` does. Auto-registration
    for THESE classes must keep landing in ``passives_dict`` under 'pos'/
    'neg' (legacy behavior; existing callers and tests key on it), so
    the active module is suspended for the duration of construction and the
    finished instance is registered with :meth:`Module.add_passive` instead
    — the same two-step ``construct, then register`` shape the old
    ``_AutoRegisterPassive`` metaclass used.

    An explicit ``Module.add(passives.Resistor(...))`` call (made outside any ``with mod:`` block) is unaffected: with no active
    module there is nothing to suspend, and the caller's own ``add()`` call
    registers it into ``elem_dict`` afterward, same as for any other
    AnalogElement.
    """
    from silk.circuit._module_base import _active_module, _push_active, _pop_active
    mod = _active_module()
    if mod is not None:
        _pop_active()
    try:
        parent_cls.__init__(self, name, pos=pos, neg=neg, value=value)
    finally:
        if mod is not None:
            _push_active(mod)
    if mod is not None:
        mod.add_passive(self)


class Resistor(_ideal.resistor):
    """Deprecated shim: builds ``silk.elements.elements.resistor``.

    Kept for existing call sites — same constructor signature as before
    (``name, pos, neg, resistance=None, *, res=None``) — but the object this
    returns IS a YAML-built ideal resistor (an ``AnalogElement``), not a
    ``PassiveBase`` instance: it still passes ``_is_ideal_passive`` (its
    ``_spice_prefix`` is ``'R'``, inherited), which is what
    :mod:`~silk.circuit.modules_results` and :meth:`Module.add_passive`
    recognise it by. Auto-registration inside ``with mod:`` still lands it in
    ``passives_dict`` via :meth:`Module.add_passive` (legacy behavior,
    see :func:`_init_ideal_shim`); an explicit ``Module.add(...)`` call lands
    it in ``elem_dict``, exactly like a directly constructed
    ``elements.resistor(...)``.
    """

    def __init__(self, name: str, pos: str, neg: str, resistance=None, *, res=None):
        warnings.warn(_deprecation_message('Resistor', 'resistor'),
                      DeprecationWarning, stacklevel=2)
        value = resistance if resistance is not None else res
        if value is None:
            raise ValueError(
                "Resistor requires a value: use resistance=<value> or res=<value>"
            )
        _validate_value(value, 'resistance', owner='Resistor', name=name)
        _validate_name(name)
        _validate_net(pos, 'pos')
        _validate_net(neg, 'neg')
        _init_ideal_shim(self, _ideal.resistor, name, pos, neg, value)

    resistance = _value_alias('resistance')
    pos = _terminal_alias('p')
    neg = _terminal_alias('n')


class Capacitor(_ideal.capacitor):
    """Deprecated shim: builds ``silk.elements.elements.capacitor``.
    See :class:`Resistor` for the full rationale — identical in shape."""

    def __init__(self, name: str, pos: str, neg: str, capacitance=None, *, cap=None):
        warnings.warn(_deprecation_message('Capacitor', 'capacitor'),
                      DeprecationWarning, stacklevel=2)
        value = capacitance if capacitance is not None else cap
        if value is None:
            raise ValueError(
                "Capacitor requires a value: use capacitance=<value> or cap=<value>"
            )
        _validate_value(value, 'capacitance', owner='Capacitor', name=name)
        _validate_name(name)
        _validate_net(pos, 'pos')
        _validate_net(neg, 'neg')
        _init_ideal_shim(self, _ideal.capacitor, name, pos, neg, value)

    capacitance = _value_alias('capacitance')
    pos = _terminal_alias('p')
    neg = _terminal_alias('n')


class Inductor(_ideal.inductor):
    """Deprecated shim: builds ``silk.elements.elements.inductor``.
    See :class:`Resistor` for the full rationale — identical in shape."""

    def __init__(self, name: str, pos: str, neg: str, inductance=None, *, ind=None):
        warnings.warn(_deprecation_message('Inductor', 'inductor'),
                      DeprecationWarning, stacklevel=2)
        value = inductance if inductance is not None else ind
        if value is None:
            raise ValueError(
                "Inductor requires a value: use inductance=<value> or ind=<value>"
            )
        _validate_value(value, 'inductance', owner='Inductor', name=name)
        _validate_name(name)
        _validate_net(pos, 'pos')
        _validate_net(neg, 'neg')
        _init_ideal_shim(self, _ideal.inductor, name, pos, neg, value)

    inductance = _value_alias('inductance')
    pos = _terminal_alias('p')
    neg = _terminal_alias('n')
