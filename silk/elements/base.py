"""AnalogElement — base class for all dynamically-generated device instances."""


_PIN_UNSET = object()

# What a net token is being asked for, and the rule each ask states.
#
# ``kind='pin'``  — the value a device pin (or a submodule port) is wired WITH.
# ``kind='net'``  — a net NAME being declared.
#
# Both have to survive into the deck as exactly ONE SPICE node token, so they
# share one validator. TASK-75: the factory style (``pdk.nmos1p8(...)``) and
# the legacy style (``add_element(..., nets={...})``) had drifted apart
# precisely because only the first one validated — a second, subtly different
# validator is how they would drift again.
#
# Each entry is (rule, empty-case tail, whitespace-case tail).
_TOKEN_RULES = {
    'pin': (
        'a pin must be a non-empty net name (str, no whitespace) or a Net',
        'An empty pin would emit a device line one node short, which SPICE '
        'mis-parses instead of rejecting.',
        'One pin is one node token on the device line.',
    ),
    'net': (
        'a net name must be a non-empty str with no whitespace (or a Net)',
        'An empty net name cannot be a SPICE node.',
        'A net name is one node token; whitespace splits it into two.',
    ),
    'port': (
        'a formal port name must be a non-empty str with no whitespace (or a Net)',
        'An empty formal name cannot identify a subcircuit port.',
        'One formal port name is one node token.',
    ),
    # TASK-76: a source terminal (VoltageSource.positive, VCVS.ctrl_neg, ...)
    # is the same thing as a device pin — one node token on the element line —
    # but the failure it guards against reads differently: ngspice-42 parsed
    # ``Vvs  gnd DC 1.8`` (positive='') as positive=gnd, negative=DC, and
    # invented a node called ``dc`` at -1.8 V — polarity inverted, no warning.
    'terminal': (
        'a source terminal must be a non-empty net name (str, no whitespace) '
        'or a Net',
        'An empty terminal would emit a source line one node short, which '
        'SPICE mis-parses into a phantom node instead of rejecting.',
        'One terminal is one node token on the source line.',
    ),
}


def pin_context(type_name, instance_name, pin, *, noun='pin'):
    """The ``<device type> '<instance>' pin '<pin>'`` prefix every pin-value
    error carries, so the message names WHERE the bad value was written.

    Built here rather than at each call site so the factory path
    (:func:`_validate_pin_value`) and the legacy ``add_element`` /
    ``connect_element`` path produce the same string for the same mistake. Sources pass ``noun='terminal'`` — same
    shape, the word the source API itself uses.
    """
    if instance_name is None:
        return f'{type_name} {noun} {pin!r}'
    return f'{type_name} {instance_name!r} {noun} {pin!r}'


def param_context(element, param):
    """``<device type> '<instance>' parameter '<param>'``: where a
    parameter value is written, for the messages that refuse it.
    The same shape as :func:`pin_context`."""
    return pin_context(type(element).__name__, getattr(element, 'name', None),
                       param, noun='parameter')


def validate_net_token(value, *, who, kind='pin'):
    """Return the net NAME *value* denotes, or raise naming *who*.

    A device pin contributes exactly ONE node token to its SPICE line, so the
    only things that can legally sit on one are an ASCII identifier with an
    optional trailing numeric index (or ground ``0``), or a :class:`~silk.circuit.net.Net` (duck-typed on a
    string ``.name``), which normalises to that name.  Anything else produces
    a deck that is wrong rather than rejected:

    * ``g=''`` used to be accepted, skipped by the binding chokepoint (falsy)
      and emitted as a line one node short — ngspice-42 parsed the model name
      as the missing node and invented a net called
      ``sky130_fd_pr__nfet_01v8``, silently mis-wiring the transistor.
    * ``g=None`` used to die deep inside ``' '.join`` at netlist time with a
      bare ``sequence item 1: expected str instance, NoneType found`` on the
      factory path, and with a bare ``'NoneType' object has no attribute
      'casefold'`` from inside net-name canonicalization on the legacy one —
      loud, but naming neither the instance nor the pin.
    * ``g='a b'`` (or any embedded whitespace) is the same defect with the
      opposite sign: one pin, two node tokens.

    The first two come from a variable that was never assigned, which is a
    realistic mistake in generated or parametric code. All three are CALLER
    ERRORS in the sense of the metric failure convention — malformed for every corner, not a
    data-dependent degenerate result — so they raise, with the diagnosis in
    the message rather than in silk's source.

    Parameters
    ----------
    value : object
        The candidate net.  ``_PIN_UNSET`` reports the pin as unconnected.
    who : str
        Where the value was written, e.g. ``"nmos1p8 'mn' pin 'g'"`` — build
        it with :func:`pin_context` for a pin.
    kind : {'pin', 'net', 'terminal', 'port'}
        Which rule the message states: the value a pin is wired with, a
        net name being declared, the node a source terminal sits on, or a
        formal port name. Formal names use the same identifier grammar but
        have no ground ``0`` exception; actual connected values retain it.

    Returns
    -------
    str
        The net name to store.

    Raises
    ------
    TypeError
        *value* is not a str and is not a Net-like object (``None``, an int,
        an arbitrary object) — the wrong KIND of thing for a pin.
    ValueError
        *value* is a str but violates the ASCII identifier rule, including
        empty names, whitespace, punctuation, invalid indices or leading
        digits (except actual ground ``0``), or uses the reserved ``__silk_`` prefix.
    """
    must, empty_tail, space_tail = _TOKEN_RULES[kind]

    if value is _PIN_UNSET:
        raise TypeError(f'{who}: no net connected — {must}.')

    # Net duck-type: anything carrying a usable string `.name` (Net, and bus
    # handles, which are Nets too). Checked before the str test so a str
    # subclass that also has .name still reads as a name.
    if not isinstance(value, str):
        name = getattr(value, 'name', None)
        if isinstance(name, str):
            return validate_net_token(name, who=who, kind=kind)
        if value is None:
            hint = ' A None value usually means a net variable was never assigned.'
        elif isinstance(value, (int, float)):
            if kind == 'port':
                hint = " Use a formal port name string such as 'vss'."
            elif value == 0:
                hint = f" A net name is a string: write '0', not {value}."
            else:
                hint = (" Use a legal net name such as 'node_42'; only the ground "
                        "net '0' may start with a digit.")
        else:
            hint = ''
        raise TypeError(
            f'{who}: {value!r} is not a net name (got '
            f'{type(value).__name__}) — {must}.{hint}'
        )

    if not value.strip():
        raise ValueError(f'{who}: {value!r} is empty — {must}. {empty_tail}')
    if any(ch.isspace() for ch in value):
        raise ValueError(
            f'{who}: {value!r} contains whitespace — {must}. {space_tail}'
        )
    if value.lower().startswith('__silk_'):
        from silk.circuit.emitted_names import _generated_net_tokens
        if kind not in ('pin', 'terminal') or value not in _generated_net_tokens.get():
            raise ValueError(f'{who}: net {value!r} uses the reserved __silk_ namespace.')
    from silk.spice_names import validate_authored_name
    validate_authored_name(value, kind='port' if kind == 'port' else 'net', who=who)
    return value


def _validate_pin_value(owner, pin, value):
    """:func:`validate_net_token` for a pin of *owner* (the factory path)."""
    who = pin_context(owner.__class__.__name__,
                      getattr(owner, 'name', None), pin)
    return validate_net_token(value, who=who, kind='pin')

class AnalogElement:
    """Base class for all analog device instances.

    Subclasses are created at runtime by ``make_device_class`` from YAML
    device specs, for both PDK devices and the ideal elements; the base class
    itself cannot be emitted (``to_spice`` raises ``NotImplementedError``).
    Each subclass has named constructor arguments matching its pin and
    parameter names from the YAML: the instance name is the only positional
    argument, pins and parameters are keyword-only. Pins and parameters can
    also be assigned afterwards (``mn1.w = 2.0``, ``mn1.d = 'vout'``); a
    pin assignment is validated, and rewires the net when the element belongs
    to a module.

    The factory sets these class-level attributes on each subclass:
    ``_pin_names`` (ordered pin names), ``_param_names`` (ordered parameter
    names), ``_param_defaults`` (default per parameter, ``None`` for required),
    ``_param_constraints`` (a constraint object per parameter),
    ``_spice_prefix`` (``'XM'``, ``'R'``, ``'C'``, ``'V'``, ...),
    ``_pdk_device`` (the PDK model name, ``''`` for ideal elements),
    ``_device_type_name`` (the catalog key, such as ``'nmos1p8'``),
    ``_is_pdk_device``, and the optional per-model declarations
    ``_op_params``, ``_op_internal_name``, ``_polarity`` and ``_op_backends``
    (``None`` when the catalog declares nothing; see
    :class:`~silk.pdk.PDKDevice`).

    Parameters
    ----------
    name : str
        Instance name, such as ``'mn1'``.
    **kwargs
        Accepted and ignored by the base class; generated subclasses consume
        pins and parameters here.

    Attributes
    ----------
    name : str
        Instance name. Read-only once the element is placed in a module.
    nets : dict
        Maps pin name to the connected net name.
    params : dict
        Maps parameter name to its value (a number, a string or a
        ``Variable``).
    layout : dict
        Layout hint annotations; empty initially.
    estimated_area : float or None
        Filled by ``estimate_layout``; ``None`` initially.

    Raises
    ------
    CircuitError
        *name* is not a valid instance name (ASCII letters, digits and
        underscores, not starting with a digit, with an optional numeric
        index) or uses the reserved ``__silk_`` prefix.
    """

    # Class-level defaults — overridden by factory-created subclasses
    _pin_names: tuple = ()
    _param_names: tuple = ()
    _param_defaults: dict = {}
    _param_constraints: dict = {}
    _spice_prefix: str = 'X'
    _pdk_device: str = ''
    _device_type_name: str = ''
    _is_pdk_device: bool = True
    #: (save_device_params): this model's verified '@' op-point
    #: parameter names, and the literal internal instance name ngspice needs
    #: before '[<param>]' in that address. Both None on a bare/undeclared
    #: subclass -- see PDKDevice.op_params/op_internal_name for the full
    #: rationale (declared per-model, never guessed).
    _op_params: tuple = None
    _op_internal_name: str = None
    #: (op_table): this model's declared channel polarity
    #: ('n'/'p'), same self-describing-PDK pattern as the two above. None
    #: on a bare/undeclared subclass -- see PDKDevice.polarity.
    _polarity: str = None
    #: (Xyce op_table support): which simulator backends this
    #: model's op-point addressing has been verified on by a real run, same
    #: self-describing-PDK pattern as the three above. None on a
    #: bare/undeclared subclass -- see PDKDevice.op_backends.
    _op_backends: tuple = None
    # Backward-compat attribute used by circuit_elements.py
    device_name: str = ''

    def __init__(self, name: str, **kwargs):
        from silk.circuit.emitted_names import validate_instance_name
        validate_instance_name(name)
        # Use object.__setattr__ to bypass our custom __setattr__ during init
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'nets', {})
        from silk.elements._pdk_emission import parameter_values, emission_spec
        object.__setattr__(self, 'params', parameter_values({}, emission_spec(self)))
        # Parameter names the CALLER has set — at construction or by later
        # attribute assignment. Emission unit-converts exactly these; a
        # catalog default is already in the PDK's own unit and never scales.
        # Owned here, not by the factory: bare AnalogElement subclasses hit
        # __setattr__ too.
        object.__setattr__(self, '_caller_supplied', set())
        object.__setattr__(self, 'layout', {})
        # Supported factory devices follow their weak wiring holders.
        # Custom emitters retain the legacy single-owner back-reference.
        object.__setattr__(self, '_module', None)
        # Phase 1 layout field — populated by estimate_layout()
        object.__setattr__(self, 'estimated_area', None)

    @property
    def _module(self):
        from silk.circuit.object_holders import supported
        if not supported(self):
            return vars(self).get('_module')
        holders = list(vars(self).get('_wiring_holders', {}))
        return holders[-1] if holders else None

    @_module.setter
    def _module(self, value):
        from silk.circuit.object_holders import supported
        if not supported(self):
            vars(self)['_module'] = value
        # Supported wiring holders are registered after placement succeeds.

    def __setattr__(self, name, value):
        """Allow direct pin/param assignment: ``mn1.w = 2.0``, ``mn1.d = 'vout'``.

        Pin assignments propagate through each supported wiring holder's
        binding chokepoint. Custom emitters retain their single-owner path.

        Order for a pin write is validate the value, then bind (which may
        raise — ``UndeclaredNetError`` under ``strict_nets``), then the
        owner-side write to ``self.nets``; so a rejected rewire leaves the
        attribute, both Nets' pins and the reverse index exactly as they
        were. ``SourceBase.__setattr__`` uses the same order.
        """
        from silk.elements._pdk_emission import (
            emission_spec, parameter_values, refuse_derived_inputs)
        spec = emission_spec(self)
        refuse_derived_inputs((name,), spec, getattr(self, 'name', type(self).__name__))
        if name == 'params':
            value = parameter_values(value, spec)
        if name == 'name' and getattr(self, '_name_locked', False):
            from silk.exceptions import CircuitError
            raise CircuitError(f"Instance {self.name!r} is placed: its name is read-only. "
                               "Construct a new instance with the desired name.")
        if name in ('name', 'nets', 'params', 'layout', '_module'):
            object.__setattr__(self, name, value)
            return
        if name in self.__class__._param_names:
            # One card per element: a line break in a string value would
            # start a new deck line (TASK-593). Rendering checks again.
            from silk.utils.deck_text import refuse_line_breaks
            self.params[name] = refuse_line_breaks(
                value, who=param_context(self, name))
            # An assigned parameter is caller-supplied: ``mn.w = 5`` is the
            # author writing 5 in THEIR unit, exactly like ``w=5`` at
            # construction, so emission must unit-convert it the same way.
            # Without this, assigning a parameter that was DEFAULTED at
            # construction skipped conversion — on a metres-PDK, ``mn.w = 5``
            # emitted five metres. A resistor's length is a continuous model
            # formula, so that one was a silently wrong answer, not an error.
            self._caller_supplied.add(name)
            return
        if name in self.__class__._pin_names:
            # Validate BEFORE the owner-side write, so a rejected rewire
            # leaves both device.nets and Net.pins exactly as they were.
            value = _validate_pin_value(self, name, value)
            from silk.circuit.object_holders import change_pins, supported
            if supported(self):
                change_pins(self, {name: value}, lambda: self.nets.__setitem__(name, value))
            else:
                module = self._module
                if module is not None:
                    from silk.circuit.net import PortBinding
                    module._element_mgr._bind_pin_to_net(value, PortBinding(
                        net=value, instance=self.name, port=name,
                        direction=None, role=None, kind='device'))
                self.nets[name] = value
            return
        object.__setattr__(self, name, value)

    def validate(self):
        """Do nothing; present as a hook for subclasses.

        Neither the base class nor the generated classes validate here.
        Caller-supplied parameter values of a PDK device are checked against the
        catalog when the netlist is emitted (``to_spice``).
        """
        pass

    @staticmethod
    def _constraint_bounds(constraint):
        """(lo, hi) bounds derived from a PDK parameter constraint, or None.

        The catalog vocabulary uses ``'-1'`` (and numeric ``-1``) to mean
        *unbounded* on either end — the same convention
        :meth:`ParameterConstraint.validate_value` reads — so each end maps
        to ``None`` when absent/unbounded/non-numeric. Returns None when
        neither end is bounded. This is the ONE derivation both variable
        factories use; it used to hardcode ``(minval, None)`` and never read
        ``maxval``, and the ``None`` upper end then crashed
        ``Variable.validate()`` on any value that passed the lower
        check.
        """
        if constraint is None:
            return None

        def _end(v):
            if v in (None, '-1', -1):
                return None
            try:
                f = float(v)
            except (ValueError, TypeError):
                return None
            return None if f == -1 else f

        lo = _end(getattr(constraint, 'minval', None))
        hi = _end(getattr(constraint, 'maxval', None))
        return None if lo is None and hi is None else (lo, hi)

    def to_spice(self) -> str:
        """Return the SPICE netlist line for this element.

        Returns
        -------
        str
            The instance line, ending in a newline. Generated subclasses
            implement this; a PDK device raises ``CircuitError`` if a
            caller-supplied value is outside the catalog's range.

        Raises
        ------
        NotImplementedError
            Called on the base class or on a subclass that does not override
            it.
        """
        raise NotImplementedError(
            f'{self.__class__.__name__}.to_spice() not implemented. '
            'Use a factory-created subclass.'
        )

    @classmethod
    def variable(cls, param, *, name, value=None, **kwargs):
        """Create a Variable pre-populated from PDK constraints for *param*.

        Bounds come from the catalog's ``minval`` and ``maxval`` for the
        parameter (``-1`` means that end is unbounded; both unbounded gives
        no bounds), and the resolution from its ``resolution`` entry when the
        catalog has one. For example the ``w`` of ``sky130``'s ``nmos1p8``
        gives bounds ``(0.42, None)`` in the PDK's own unit. Ideal elements
        carry no bounds.

        Parameters
        ----------
        param : str
            Parameter name (must be in ``cls._param_names``).
        name : str
            Variable name, required to avoid silent clashes in a
            ``VariableSpace``. Keyword-only.
        value : float, optional
            Initial / nominal value.
        **kwargs
            Passed through to ``Variable``.  ``bounds`` and
            ``resolution`` override the auto-derived PDK values when supplied.

        Returns
        -------
        Variable
            The new variable.

        Raises
        ------
        ValueError
            *param* is not a parameter of this device; the message lists the
            valid names.
        """
        if param not in cls._param_names:
            valid = ', '.join(cls._param_names) if cls._param_names else '(none)'
            raise ValueError(
                f'{cls.__name__}.variable(): unknown param {param!r}. '
                f'Valid params: {valid}'
            )

        if name is None:
            name = param

        constraint = cls._param_constraints.get(param)

        # Auto-derive bounds from the PDK constraint ('-1' = unbounded end).
        if 'bounds' not in kwargs:
            kwargs['bounds'] = cls._constraint_bounds(constraint)

        # Auto-derive resolution from PDK constraint
        if 'resolution' not in kwargs:
            res = None
            if constraint is not None and constraint.resolution is not None:
                try:
                    res = float(constraint.resolution)
                except (ValueError, TypeError):
                    pass
            if res is not None:
                kwargs['resolution'] = res

        from silk.circuit.variable import Variable
        return Variable(name, value=value, **kwargs)

    def scale(self, factor: float):
        """Do nothing; a hook for subclasses that scale device width.

        Parameters
        ----------
        factor : float
            Width scale factor. Ignored by the base class and by the
            generated classes.
        """
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name!r})'
