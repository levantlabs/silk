"""silk source element classes (VoltageSource, CurrentSource, controlled sources)."""

import functools
from silk.circuit.emitted_names import _source_instance_name, validate_instance_name
from silk.elements.base import _PIN_UNSET, pin_context, validate_net_token
from silk.elements.factory import ParamDescriptor, _render_param
from silk.sources.waveforms import PWL, Pulse, Sine, Exponential, SFFM, AM, TransientNoise, RandomSource
from silk.utils._utils import _resolve_value, _parse_si
from silk.utils.deck_text import card_text, refuse_line_breaks


@functools.lru_cache(maxsize=None)
def _builtin_card_fields(cls):
    """The fields *cls*'s card emitter writes, when that emitter is one of
    silk's own (defined in this module); otherwise none.

    A subclass that overrides ``to_spice()`` owns its own text, opaque to
    silk, so its attributes are never refused here."""
    owner = next((c for c in cls.__mro__ if 'to_spice' in vars(c)), None)
    if owner is None or owner.__module__ != __name__:
        return ()
    return getattr(cls, '_card_fields', ())

# Base class
# ---------------------------------------------------------------------------

class _AutoRegisterSource(type):
    """Metaclass that auto-registers a fully-constructed source with the
    active module (set by using a Module as a context manager, ``with mod:``),
    if any."""
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        from silk.circuit.modules import _active_module
        mod = _active_module()
        if mod is not None:
            mod.add_source(instance)
        return instance


class SourceBase(metaclass=_AutoRegisterSource):
    """Base class for all SPICE sources.

    Subclasses define their own terminals, parameters and ``to_spice``.
    Every attribute named in the subclass's terminal list (``positive`` and
    ``negative`` for the independent and behavioral sources, for example) is
    one SPICE node token on the source line, so it must be a non-empty,
    whitespace-free net name or a :class:`~silk.circuit.Net` (stored as its
    name). Anything else raises at assignment, naming the source and the
    terminal: ``ValueError`` for an empty, blank or whitespace-split name and
    ``TypeError`` for a value that is not a net name at all. Construction, a
    later ``src.positive = ...`` rewire and
    :meth:`Module.connect_source <silk.circuit.Module.connect_source>` all go
    through this check. Once the source is attached to a
    :class:`~silk.circuit.Module`, the same assignment also moves the
    terminal's ``Net.pins`` binding, and a rejected rewire leaves the
    terminal and both nets' pins unchanged.

    Without the check, ``VoltageSource('vs', positive='', negative='gnd',
    dc=1.8)`` would emit ``Vvs  gnd DC 1.8``, which a SPICE parser reads with
    the polarity inverted.

    A source is also registered automatically with the active module when it
    is created inside a ``with module:`` block.

    Parameters
    ----------
    name : str
        Instance name, stored as ``name``. The emitted SPICE name is this
        with the source type's prefix (``V``, ``I``, ``E``, ``G``, ``H``,
        ``F`` or ``B``) in front.

    Raises
    ------
    silk.exceptions.CircuitError
        If *name* is not a valid instance name (ASCII letters, digits and
        underscores, starting with a letter or underscore, optionally
        followed by one numeric index ``[k]``; ``:`` and ``.`` are not
        allowed) or contains the reserved ``__silk_`` prefix. Reassigning
        ``name`` after the source is placed in a module also raises it.
    """

    # Attribute names that are node tokens on this source's line, in emission
    # order. ``ctrl_vname`` (CCVS/CCCS) names a controlling SOURCE, not a
    # node, so it is deliberately absent. Subclasses override.
    _node_terminals = ()

    @property
    def instance_name(self):
        """The logical SPICE instance name, as a ``str``.

        The source type's prefix followed by the escaped ``name`` (for example
        ``'Vvdd'`` for a ``VoltageSource`` named ``'vdd'``). An independent
        source with a waveform writes its cards under this name plus a
        ``__silk_...`` suffix; see :meth:`physical_instance_names`.
        """
        return _source_instance_name(type(self), self.name)

    def physical_instance_names(self):
        """The instance names this source's card(s) carry in the deck.

        Constant and controlled sources have one instance. Independent sources
        with a waveform override this with their value-independent two-card
        layout; no renderer, Variable or waveform callback is evaluated.

        Returns
        -------
        tuple of str
            The emitted instance name of each card, in emission order.
        """
        return (self.instance_name,)

    def __init__(self, name):
        validate_instance_name(name)
        self.name = name
        # Strong emission context, selected by the last successful add.
        # Supported sources rewire through separate weak holder records;
        # custom emitters retain this context as their single wiring owner.
        object.__setattr__(self, '_module', None)

    def __setattr__(self, attr, value):
        """Terminal writes are validate, then bind, then the owner write.

        The bind (``_bind_source_terminal`` -> ``_bind_pin_to_net``) may
        raise ``UndeclaredNetError`` under ``strict_nets``; because the
        attribute is written only after it returns, a rejected rewire
        leaves the terminal, both Nets' pins and the reverse index exactly
        as they were. ``AnalogElement.__setattr__`` uses the same order.
        """
        if attr == 'name' and getattr(self, '_name_locked', False):
            from silk.exceptions import CircuitError
            raise CircuitError(f"Instance {self.name!r} is placed: its name is read-only. "
                               "Construct a new instance with the desired name.")
        if getattr(self, '_independent', False):
            if attr == 'pwl':
                value = VoltageSource._normalize_pwl(value)
            self._validate_update({attr: value})
        if attr in self._node_terminals:
            # Validate BEFORE anything is written (TASK-76).
            value = self._validate_terminal(attr, value)
            from silk.circuit.object_holders import change_pins, supported
            if supported(self):
                change_pins(self, {attr: value}, lambda: object.__setattr__(self, attr, value))
            else:
                module = getattr(self, '_module', None)
                if module is not None:
                    from silk.circuit.modules_structure import _bind_source_terminal
                    _bind_source_terminal(module, self.name, attr, value)
                object.__setattr__(self, attr, value)
            return
        # A field silk's own emitter writes into a card (dc, ac, gain,
        # expression, ctrl_vname, ...): a line break would start a new deck
        # line (TASK-593). Rendering checks again. Other attributes, and
        # every attribute of a subclass with its own to_spice(), are not
        # silk's card text and are left alone.
        if attr in _builtin_card_fields(type(self)):
            refuse_line_breaks(value, who=self._field_context(attr))
        object.__setattr__(self, attr, value)

    def __copy__(self):
        from silk.circuit.object_holders import detached_state
        result = object.__new__(type(self))
        result.__dict__.update(detached_state(self))
        result.__dict__.pop('_restore_context', None)
        object.__setattr__(result, '_module', getattr(self, '_module', None))
        return result

    def __getstate__(self):
        from silk.circuit.object_holders import detached_state, supported
        if not supported(self):
            return dict(vars(self))
        state = detached_state(self)
        module = getattr(self, '_module', None)
        if module is not None:
            token = vars(module).get('_source_context')
            if token is not None:
                state['_restore_context'] = token
            else:
                state['_module'] = module
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        from silk.circuit.object_holders import supported
        if not supported(self):
            return
        object.__setattr__(self, '_module', state.get('_module'))
        token = state.get('_restore_context')
        if token is not None:
            token.subscribe(self)

    def _apply_update(self, changes):
        proposed = dict(changes)
        if getattr(self, '_independent', False):
            if 'pwl' in proposed:
                proposed['pwl'] = VoltageSource._normalize_pwl(proposed['pwl'])
            self._validate_update(proposed)
        for key, value in list(proposed.items()):
            if key == 'name' and getattr(self, '_name_locked', False):
                from silk.exceptions import CircuitError
                raise CircuitError(f"Instance {self.name!r} is placed: its name is read-only.")
            if key in self._node_terminals:
                proposed[key] = self._validate_terminal(key, value)
            elif key in _builtin_card_fields(type(self)):
                refuse_line_breaks(value, who=self._field_context(key))
        from silk.circuit.object_holders import change_pins, supported
        if not supported(self):
            module = getattr(self, '_module', None)
            if module is not None:
                for key in self._node_terminals:
                    if key in proposed:
                        module._element_mgr._accept_net(proposed[key])
            from silk.sources._layout import WAVEFORM_SLOTS
            value_fields = (*WAVEFORM_SLOTS, 'dc', 'ac', 'pwl_r', 'pwl_td') if getattr(self, '_independent', False) else ()
            # Commit the validated waveform/value state together, after
            # ordinary legacy setters, without validating intermediate states.
            for key, value in proposed.items():
                if key not in value_fields:
                    setattr(self, key, value)
            for key, value in proposed.items():
                if key in value_fields:
                    object.__setattr__(self, key, value)
            return
        before = dict(vars(self))
        def commit():
            try:
                for key, value in proposed.items():
                    object.__setattr__(self, key, value)
            except BaseException:
                self.__dict__.clear()
                self.__dict__.update(before)
                raise
        change_pins(self, {k: v for k, v in proposed.items() if k in self._node_terminals}, commit)

    def _field_context(self, attr):
        """``<class> '<name>' parameter '<attr>'`` for a refusal message."""
        return pin_context(type(self).__name__, getattr(self, 'name', None),
                           attr, noun='parameter')

    def _card_value(self, value, attr):
        """The text written for *value* (already resolved), refused if it
        holds a line break."""
        return card_text(value, who=self._field_context(attr))

    def _validate_terminal(self, terminal, value):
        """Return the net name *value* denotes for *terminal*, or raise
        (``TypeError`` for the wrong kind of thing, ``ValueError`` for an
        empty/blank/split name) naming this source and the terminal."""
        who = pin_context(type(self).__name__, getattr(self, 'name', None),
                          terminal, noun='terminal')
        return validate_net_token(value, who=who, kind='terminal')

    def _node(self, terminal):
        """The node token *terminal* contributes to the SPICE line.

        Re-validated here as well as at assignment, so the emitter itself can
        never write a line with fewer (or more) node tokens than the source
        has terminals, whatever happened to the attribute in between (the
        same belt-and-braces as ``AnalogElement.to_spice``).
        """
        return self._validate_terminal(terminal,
                                       getattr(self, terminal, _PIN_UNSET))

    def to_spice(self):
        """The SPICE card(s) for this source.

        Implemented by each concrete source class.

        Returns
        -------
        str
            One or more newline-terminated cards.

        Raises
        ------
        NotImplementedError
            Always, on the base class itself.
        """
        raise NotImplementedError

    # Waveform kinds that emit to SPICE but have no numpy model. to_numpy()
    # calls this before its DC fallback so a behavioral run fails loudly
    # instead of silently flattening the stimulus to its DC value.
    _UNMODELED_WAVEFORMS = ('exponential', 'sffm', 'am', 'trnoise',
                            'trrandom', 'waveform')

    def _reject_unmodeled_waveform(self):
        for attr in self._UNMODELED_WAVEFORMS:
            wave = getattr(self, attr, None)
            if attr == 'waveform' and callable(getattr(wave, 'to_numpy', None)):
                continue
            if wave is not None:
                raise NotImplementedError(
                    f"Source {self.name!r} uses the {attr!r} waveform, which "
                    f"has no numpy model — behavioral simulation would "
                    f"silently reduce it to its DC value. Supported in "
                    f"to_numpy(): pulse, sine, pwl, dc. Use SPICE simulation "
                    f"for this source, or model the stimulus as a PWL."
                )

    def _resolve_ctrl_vname(self, ctrl_vname):
        """Map a CCVS/CCCS ``ctrl_vname`` — documented as the controlling
        source's PYTHON INSTANCE NAME (e.g. ``'vmeas'`` for
        ``sources.vdc('vmeas', ...)``) — to the SPICE ELEMENT name ngspice's
        H/F card actually needs (``'Vvmeas'``), via this source's owning
        module.

        Looked up first in the module's declarative sources (keyed by
        instance name, same as ``sig_sources_dict``), then in its
        ``add_element`` devices (keyed by ``elem_dict``). A ``ctrl_vname`` that matches neither — including one already
        given as the SPICE element name itself, e.g. ``'Vvmeas'`` — is
        returned unchanged, so pre-existing callers keep working.
        """
        module = getattr(self, '_module', None)
        if module is None:
            return ctrl_vname
        target = module.sig_sources_dict.get(ctrl_vname)
        if target is not None:
            if isinstance(target, VoltageSource):
                return target.physical_layout().current_terms[0][0]
            if not isinstance(target, (VoltageControlledVoltageSource,
                                       CurrentControlledVoltageSource, BehavioralVoltage)):
                from silk.exceptions import CircuitError
                raise CircuitError(f'Source {self.name!r}: control {ctrl_vname!r} '
                                   'must name a voltage-source branch.')
            return target.instance_name
        entry = module.elem_dict.get(ctrl_vname)
        if entry is not None:
            from silk.circuit.emitted_names import emitted_instance_name
            return emitted_instance_name(entry['device'], ctrl_vname)
        return ctrl_vname


# ---------------------------------------------------------------------------
# Independent sources
# ---------------------------------------------------------------------------


class _IndependentSource:
    """Common additive DC semantics, physical identity and update validation."""
    _independent = True
    _card_fields = ('dc', 'ac', 'pwl_r', 'pwl_td')

    def _validate_update(self, changes):
        from silk.sources._layout import selected_waveform, source_layout, WAVEFORM_SLOTS
        from silk.exceptions import CircuitError
        if not (set(changes) & (set(WAVEFORM_SLOTS) | {'name'})):
            return
        proposed = dict(vars(self), **changes)
        selected_waveform(proposed.get('name', '<unnamed>'), proposed)
        if getattr(self, '_module', None) is not None or getattr(self, '_name_locked', False):
            if (proposed.get('name') != self.name or
                    source_layout(self, proposed) != source_layout(self)):
                raise CircuitError(
                    f'Source {self.name!r} is placed: changing its name or waveform '
                    'topology is refused. Construct the desired source before placement.')

    @property
    def pwl_r(self):
        """Repeat setting of the current PWL, or the stored convenience value."""
        wave = self.__dict__.get('pwl')
        return wave.r if wave is not None else self.__dict__.get('_pwl_r')

    @pwl_r.setter
    def pwl_r(self, value):
        wave = self.__dict__.get('pwl')
        if wave is not None:
            wave.r = value
        else:
            self.__dict__['_pwl_r'] = value

    @property
    def pwl_td(self):
        """Delay setting of the current PWL, or the stored convenience value."""
        wave = self.__dict__.get('pwl')
        return wave.td if wave is not None else self.__dict__.get('_pwl_td')

    @pwl_td.setter
    def pwl_td(self, value):
        wave = self.__dict__.get('pwl')
        if wave is not None:
            wave.td = value
        else:
            self.__dict__['_pwl_td'] = value

    def _apply_update(self, changes):
        # Explicit timing edits apply to the final PWL, regardless of the
        # caller's parameter order. SourceBase validates before committing.
        proposed = {k: v for k, v in changes.items() if k not in ('pwl_r', 'pwl_td')}
        proposed.update((k, changes[k]) for k in ('pwl_r', 'pwl_td') if k in changes)
        return SourceBase._apply_update(self, proposed)

    def physical_layout(self):
        """The value-independent card layout this source emits.

        A source with no waveform is one card named ``instance_name``. With a
        waveform it is two cards: the waveform card (``..__silk_pwl`` for
        ``pwl``, ``..__silk_wave`` for any other kind) and the ``dc`` card
        (``..__silk_dc``). A voltage source places them in series through an
        internal node ``__silk_src_V<name>``; a current source places them in
        parallel between its terminals. No value is evaluated.

        Returns
        -------
        object
            A frozen ``SourceLayout`` dataclass with ``names`` (tuple of the emitted card
            names, in emission order), ``dc_name`` (the card carrying
            ``dc``/``ac``), ``internal_node`` (the series node name, or
            ``None``) and ``current_terms`` (pairs of card name and ``1`` for
            the cards whose current is the source current).

        Raises
        ------
        silk.exceptions.CircuitError
            If more than one waveform slot is set.
        """
        from silk.sources._layout import source_layout
        return source_layout(self)

    def physical_instance_names(self):
        """The instance names of the card(s) this source emits.

        Returns
        -------
        tuple of str
            ``(instance_name,)`` without a waveform; with one, the waveform
            card name followed by the ``dc`` card name. The result does not
            depend on any value or Variable.

        Raises
        ------
        silk.exceptions.CircuitError
            If more than one waveform slot is set.
        """
        return self.physical_layout().names

    def selected_waveform(self):
        """The waveform object this source carries, if any.

        Returns
        -------
        object or None
            The single object among the ``pwl``, ``pulse``, ``sine``,
            ``exponential``, ``sffm``, ``am``, ``trnoise``, ``trrandom`` and
            ``waveform`` attributes that is not ``None``, or ``None`` when all
            are unset.

        Raises
        ------
        silk.exceptions.CircuitError
            If more than one of those attributes is set.
        """
        from silk.sources._layout import selected_waveform
        return selected_waveform(self.name, vars(self))[1]

    def waveform_initial_value(self):
        """The operating-point level of the waveform alone, without ``dc``.

        Returns
        -------
        float
            ``0`` when the source has no waveform; otherwise the value of the
            waveform's own ``initial_value()``, in volts or amps.

        Raises
        ------
        silk.exceptions.NotSupportedError
            If the waveform object has no callable ``initial_value`` (a
            generic ``waveform=`` object must provide one), or the waveform
            cannot define one for its current parameters.
        silk.exceptions.CircuitError
            If more than one waveform slot is set.
        """
        from silk.exceptions import NotSupportedError
        wave = self.selected_waveform()
        if wave is None:
            return 0
        initial = getattr(wave, 'initial_value', None)
        if not callable(initial):
            raise NotSupportedError(
                f'Source {self.name!r}: waveform {type(wave).__name__} must '
                'provide initial_value() for its operating point.')
        return initial()

    def initial_value(self):
        """The operating-point value of the source.

        Returns
        -------
        float
            The ``dc`` value plus the waveform's initial value, in volts or
            amps. It agrees with the first transient sample from
            ``to_numpy`` for the waveforms that have a numpy model.

        Raises
        ------
        ValueError
            If ``dc`` is a Variable with no value or not a valid SPICE
            number.
        silk.exceptions.NotSupportedError
            As for :meth:`waveform_initial_value`.
        silk.exceptions.CircuitError
            If more than one waveform slot is set.
        """
        return (_parse_si(_resolve_value(self.dc, f'Source {self.name!r} dc'))
                + _parse_si(self.waveform_initial_value()))

    def to_spice(self):
        """The SPICE card(s) for this source.

        Without a waveform the result is one ``DC`` card. With a waveform it
        is two: the waveform card, then a ``DC`` card holding ``dc`` (a
        voltage source puts them in series through an internal node; a
        current source puts them in parallel). ``AC <ac>`` is added to the
        ``DC`` card only when ``ac`` is not zero. ``dc`` and ``ac`` are
        rendered like device parameters: numbers with SI suffixes, strings
        verbatim.

        Returns
        -------
        str
            The newline-terminated card(s), in the layout of
            :meth:`physical_layout`.

        Raises
        ------
        ValueError
            If a terminal is empty or contains whitespace, or a Variable has
            no value assigned.
        TypeError
            If a terminal is not a net name.
        silk.exceptions.CircuitError
            If more than one waveform slot is set.
        """
        layout = self.physical_layout()
        pos, neg = self._node('positive'), self._node('negative')
        wave = self.selected_waveform()
        dc = f"DC {_render_param(self.dc, who=self._field_context('dc'))}"
        if self.ac != 0:
            dc += f" AC {_render_param(self.ac, who=self._field_context('ac'))}"
        if wave is None:
            return f'{layout.dc_name} {pos} {neg} {dc}\n'
        if layout.internal_node is None:
            wave_neg, dc_pos = neg, pos
        else:
            wave_neg = dc_pos = layout.internal_node
        return (f'{layout.names[0]} {pos} {wave_neg} {wave.to_spice()}\n'
                f'{layout.dc_name} {dc_pos} {neg} {dc}\n')

    def to_numpy(self, t):
        """Evaluate the source at the given times, for behavioral simulation.

        Parameters
        ----------
        t : array-like
            Sample times in seconds. Must have a length; without a waveform
            only that length is used.

        Returns
        -------
        np.ndarray
            ``dc`` plus the waveform's samples (or ``dc`` alone without a
            waveform), in volts or amps, one per time.

        Raises
        ------
        NotImplementedError
            If the source carries an ``exponential``, ``sffm``, ``am``,
            ``trnoise`` or ``trrandom`` waveform, or a generic ``waveform=``
            object without a ``to_numpy`` method. These have no numpy model
            and are not silently reduced to ``dc``; use SPICE simulation or
            model the stimulus as a PWL.
        ValueError
            If ``dc`` is a Variable with no value or the waveform rejects
            its parameters.
        silk.exceptions.CircuitError
            If more than one waveform slot is set.
        """
        import numpy as np
        wave = self.selected_waveform()
        dc = _parse_si(_resolve_value(self.dc, f'Source {self.name!r} dc'))
        if wave is None:
            return np.full(len(t), dc)
        if not callable(getattr(wave, 'to_numpy', None)):
            raise NotImplementedError(
                f'Source {self.name!r} uses {type(wave).__name__} (waveform '
                f'{next(k for k in self._UNMODELED_WAVEFORMS if getattr(self, k, None) is wave)!r}), which has no numpy model; '
                'use SPICE simulation or model the stimulus as PWL.')
        return dc + wave.to_numpy(t)


class VoltageSource(_IndependentSource, SourceBase):
    """Independent voltage source.

    Example::

        sources.VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8)
        sources.VoltageSource('vin', positive='in', negative='gnd', dc=0,
                               pulse=sources.Pulse(v1=0, v2=1.8, tr='1n', tf='1n',
                                                   pw='50n', per='100n'))

    Class-level param descriptors enable the attribute-access variable API::

        vdd = sources.VoltageSource.dc.variable(name='vdd', value=1.8, bounds=(1.6, 2.0))

    Parameters
    ----------
    name : str
        Instance name; see :class:`silk.sources.SourceBase` for the rules.
    positive : str
        Positive terminal: a non-empty net name without whitespace, or a
        :class:`~silk.circuit.Net`.
    negative : str
        Negative terminal; same forms as *positive*.
    dc : float, optional
        DC value in volts: a number, SPICE value string or
        :class:`~silk.circuit.Variable`. Default ``0``. With a waveform it is
        a separate offset that adds to the waveform at every time.
    ac : float, optional
        Small-signal AC magnitude in volts. Default ``0``; written to the
        card only when nonzero.
    pwl : PWL or list, optional
        Piecewise-linear waveform: a :class:`silk.sources.PWL`, or a list or tuple of
        points that is wrapped in one. Anything else raises ``TypeError``.
    pwl_r : float, optional
        Repeat-start time in seconds, applied to *pwl* when it does not set
        its own ``r`` (a :class:`silk.sources.PWL` object is updated in place).
    pwl_td : float, optional
        Delay in seconds, applied to *pwl* when it does not set its own
        ``td``.
    pulse : Pulse, optional
        Pulse waveform.
    sine : Sine, optional
        Sine waveform.
    exponential : Exponential, optional
        Exponential waveform (SPICE only; no numpy model).
    sffm : SFFM, optional
        Single-frequency FM waveform (SPICE only).
    am : AM, optional
        Amplitude-modulated waveform (SPICE only).
    trnoise : TransientNoise, optional
        Transient-noise waveform (SPICE only).
    trrandom : RandomSource, optional
        Random waveform (SPICE only).
    waveform : object, optional
        Any other waveform object with a ``to_spice()`` method. It also needs
        an ``initial_value()`` method for the operating point and a
        ``to_numpy(t)`` method for behavioral simulation.

    Raises
    ------
    silk.exceptions.CircuitError
        If more than one waveform argument is given, or *name* is invalid.
    ValueError
        If a terminal is empty or contains whitespace.
    TypeError
        If a terminal is not a net name, or *pwl* is of the wrong type.

    Notes
    -----
    Each parameter is stored as an attribute of the same name, and the
    terminals and waveform slots may be reassigned until the source is placed
    in a module; afterwards changing its name or waveform topology raises
    ``CircuitError``.

    Examples
    --------
    A constant source emits one card. With a waveform, ``dc=`` is a
    separate series offset source, added to the waveform at every time.
    The operating-point value includes the waveform's initial value,
    so it agrees with the transient start.

    >>> import numpy as np
    >>> from silk.sources import VoltageSource, Sine
    >>> supply = VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8)
    >>> print(supply.to_spice(), end='')
    Vvdd vdd gnd DC 1.8
    >>> vin = VoltageSource('in', positive='in', negative='gnd', dc=1.2,
    ...                     sine=Sine(vampl=0.2, freq=1000))
    >>> print(vin.to_spice(), end='')
    Vin__silk_wave in __silk_src_Vin SIN(0 0.2 1000 0 0)
    Vin__silk_dc __silk_src_Vin gnd DC 1.2
    >>> vin.to_numpy(np.array([0, 0.00025])).round(2).tolist()
    [1.2, 1.4]
    >>> bool(np.isclose(vin.initial_value(), vin.to_numpy(np.array([0]))[0]))
    True
    """
    _param_names = ('dc', 'ac')
    _spice_prefix = 'V'
    _node_terminals = ('positive', 'negative')
    # Non-data descriptors: class access returns ParamAccessor; instance access
    # is shadowed by instance.__dict__ entries set in __init__ (self.dc = dc).
    dc = ParamDescriptor('dc')
    ac = ParamDescriptor('ac')

    def __init__(self, name, positive, negative,
                 dc=0, ac=0,
                 pwl=None, pwl_r=None, pwl_td=None,
                 pulse=None, sine=None, exponential=None,
                 sffm=None, am=None, trnoise=None, trrandom=None,
                 waveform=None):
        super().__init__(name)
        self.positive    = positive
        self.negative    = negative
        self.dc          = dc
        self.ac          = ac
        self.pwl         = self._normalize_pwl(pwl, pwl_r, pwl_td)
        # Normalization already applied constructor defaults with PWL-field
        # precedence. Store them without reapplying them through the aliases.
        for attr, value in (('pwl_r', pwl_r), ('pwl_td', pwl_td)):
            if attr in _builtin_card_fields(type(self)):
                refuse_line_breaks(value, who=self._field_context(attr))
        self.__dict__['_pwl_r'] = pwl_r
        self.__dict__['_pwl_td'] = pwl_td
        self.pulse       = pulse     # Pulse object or None
        self.sine        = sine      # Sine object or None
        self.exponential = exponential  # Exponential object or None
        self.sffm        = sffm     # SFFM object or None
        self.am          = am       # AM object or None
        self.trnoise     = trnoise  # TransientNoise object or None
        self.trrandom    = trrandom # RandomSource object or None
        # Generic waveform slot (spec 2026-07-07 §5.3): any object with a
        # to_spice(); the last-resort branch, so a NEW waveform kind emits
        # without editing this class. Per-kind kwargs above still win.
        self.waveform    = waveform

    @staticmethod
    def _normalize_pwl(pwl, pwl_r=None, pwl_td=None):
        """Convert raw pwl input to a PWL object, or return None."""
        if pwl is None:
            return None
        if isinstance(pwl, PWL):
            if pwl_r is not None and pwl.r is None:
                pwl.r = pwl_r
            if pwl_td is not None and pwl.td is None:
                pwl.td = pwl_td
            return pwl
        if isinstance(pwl, (list, tuple)):
            return PWL(pwl, r=pwl_r, td=pwl_td)
        raise TypeError(f"pwl must be a PWL object, list, or None — got {type(pwl).__name__}")


class CurrentSource(_IndependentSource, SourceBase):
    """Independent current source.

    Positive current flows from positive node, through the source, into negative node.

    Example::

        sources.CurrentSource('ibias', positive='vb', negative='gnd', dc=100e-6)

    Class-level param descriptors enable the attribute-access variable API::

        ibias = sources.CurrentSource.dc.variable(name='ibias', value=100e-6)

    Parameters
    ----------
    name : str
        Instance name; see :class:`silk.sources.SourceBase` for the rules.
    positive : str
        Positive terminal: a non-empty net name without whitespace, or a
        :class:`~silk.circuit.Net`.
    negative : str
        Negative terminal; same forms as *positive*.
    dc : float, optional
        DC value in amps: a number, SPICE value string or
        :class:`~silk.circuit.Variable`. Default ``0``. With a waveform it is
        a separate offset that adds to the waveform at every time.
    ac : float, optional
        Small-signal AC magnitude in amps. Default ``0``; written to the
        card only when nonzero.
    pwl : PWL or list, optional
        Piecewise-linear waveform: a :class:`silk.sources.PWL`, or a list or tuple of
        points that is wrapped in one. Anything else raises ``TypeError``.
    pwl_r : float, optional
        Repeat-start time in seconds, applied to *pwl* when it does not set
        its own ``r`` (a :class:`silk.sources.PWL` object is updated in place).
    pwl_td : float, optional
        Delay in seconds, applied to *pwl* when it does not set its own
        ``td``.
    pulse : Pulse, optional
        Pulse waveform.
    sine : Sine, optional
        Sine waveform.
    exponential : Exponential, optional
        Exponential waveform (SPICE only; no numpy model).
    sffm : SFFM, optional
        Single-frequency FM waveform (SPICE only).
    am : AM, optional
        Amplitude-modulated waveform (SPICE only).
    trnoise : TransientNoise, optional
        Transient-noise waveform (SPICE only).
    trrandom : RandomSource, optional
        Random waveform (SPICE only).
    waveform : object, optional
        Any other waveform object with a ``to_spice()`` method. It also needs
        an ``initial_value()`` method for the operating point and a
        ``to_numpy(t)`` method for behavioral simulation.

    Raises
    ------
    silk.exceptions.CircuitError
        If more than one waveform argument is given, or *name* is invalid.
    ValueError
        If a terminal is empty or contains whitespace.
    TypeError
        If a terminal is not a net name, or *pwl* is of the wrong type.

    Notes
    -----
    Each parameter is stored as an attribute of the same name, and the
    terminals and waveform slots may be reassigned until the source is placed
    in a module; afterwards changing its name or waveform topology raises
    ``CircuitError``.

    Examples
    --------
    A waveform current and its ``dc=`` offset flow through separate
    parallel cards. Their sum is the logical source current, including
    at the operating point and the transient start. Without a waveform,
    a constant current source emits one card.

    >>> import numpy as np
    >>> from silk.sources import CurrentSource, Pulse
    >>> bias = CurrentSource('bias', positive='out', negative='gnd', dc='100u')
    >>> print(bias.to_spice(), end='')
    Ibias out gnd DC 100u
    >>> drive = CurrentSource('drive', positive='out', negative='gnd', dc='100u',
    ...     pulse=Pulse(v1='20u', v2='80u', tr='1u', tf='1u', pw='4u', per='10u'))
    >>> print(drive.to_spice(), end='')
    Idrive__silk_wave out gnd PULSE(20u 80u 0 1u 1u 4u 10u)
    Idrive__silk_dc out gnd DC 100u
    >>> (drive.to_numpy(np.array([0, 2e-6])) * 1e6).round(1).tolist()
    [120.0, 180.0]
    >>> bool(np.isclose(drive.initial_value(), drive.to_numpy(np.array([0]))[0]))
    True
    """
    _param_names = ('dc', 'ac')
    _spice_prefix = 'I'
    _node_terminals = ('positive', 'negative')
    dc = ParamDescriptor('dc')
    ac = ParamDescriptor('ac')

    def __init__(self, name, positive, negative,
                 dc=0, ac=0,
                 pwl=None, pwl_r=None, pwl_td=None,
                 pulse=None, sine=None, exponential=None,
                 sffm=None, am=None, trnoise=None, trrandom=None,
                 waveform=None):
        super().__init__(name)
        self.positive    = positive
        self.negative    = negative
        self.dc          = dc
        self.ac          = ac
        # Mirror VoltageSource: accept the pwl_r (repeat) / pwl_td (time-delay)
        # convenience kwargs and expose them as attributes.
        self.pwl         = VoltageSource._normalize_pwl(pwl, pwl_r, pwl_td)
        for attr, value in (('pwl_r', pwl_r), ('pwl_td', pwl_td)):
            if attr in _builtin_card_fields(type(self)):
                refuse_line_breaks(value, who=self._field_context(attr))
        self.__dict__['_pwl_r'] = pwl_r
        self.__dict__['_pwl_td'] = pwl_td
        self.pulse       = pulse
        self.sine        = sine
        self.exponential = exponential
        self.sffm        = sffm
        self.am          = am
        self.trnoise     = trnoise
        self.trrandom    = trrandom
        self.waveform    = waveform  # generic last-resort slot (spec §5.3)


# ---------------------------------------------------------------------------
# Dependent (controlled) sources
# ---------------------------------------------------------------------------

class VoltageControlledVoltageSource(SourceBase):
    """Linear voltage-controlled voltage source (SPICE E element).

    The output voltage ``V(out_pos, out_neg)`` equals ``gain`` times the
    control voltage ``V(ctrl_pos, ctrl_neg)``. Also available as
    ``silk.sources.VCVS``.

    Parameters
    ----------
    name : str
        Instance name; see :class:`silk.sources.SourceBase` for the rules.
    out_pos : str
        Positive output terminal: a non-empty net name without whitespace, or
        a :class:`~silk.circuit.Net`.
    out_neg : str
        Negative output terminal; same forms.
    ctrl_pos : str
        Positive control node; same forms as *out_pos*.
    ctrl_neg : str
        Negative control node; same forms as *out_pos*.
    gain : float
        Voltage gain in V/V (dimensionless): a number, SPICE value string or
        :class:`~silk.circuit.Variable`.

    Raises
    ------
    silk.exceptions.CircuitError
        If *name* is not a valid instance name.
    ValueError
        If a terminal is empty or contains whitespace.
    TypeError
        If a terminal is not a net name.

    Notes
    -----
    Each parameter is stored as an attribute of the same name.

    Examples
    --------
    >>> from silk.sources import VoltageControlledVoltageSource
    >>> amp = VoltageControlledVoltageSource('eamp', out_pos='vout', out_neg='gnd',
    ...                                      ctrl_pos='vin', ctrl_neg='gnd', gain=10)
    >>> print(amp.to_spice(), end='')
    Eeamp vout gnd vin gnd 10
    """
    _spice_prefix = 'E'
    _node_terminals = ('out_pos', 'out_neg', 'ctrl_pos', 'ctrl_neg')
    _card_fields = ('gain',)
    _param_names = ('gain',)

    def __init__(self, name, out_pos, out_neg, ctrl_pos, ctrl_neg, gain):
        super().__init__(name)
        self.out_pos  = out_pos
        self.out_neg  = out_neg
        self.ctrl_pos = ctrl_pos
        self.ctrl_neg = ctrl_neg
        self.gain     = gain

    def to_spice(self):
        """The SPICE E card.

        Returns
        -------
        str
            ``E<name> out_pos out_neg ctrl_pos ctrl_neg <gain>`` followed by
            a newline; the gain is written with ``str()`` (strings verbatim).

        Raises
        ------
        ValueError
            If a terminal is empty or contains whitespace, or a Variable has
            no value assigned.
        TypeError
            If a terminal is not a net name.
        silk.exceptions.DeckTextError
            If its ``gain`` contains a line break (also refused when the
            attribute is assigned).
        """
        gain = self._card_value(_resolve_value(self.gain, 'VCVS gain'), 'gain')
        return (f"{self.instance_name} {self._node('out_pos')} {self._node('out_neg')} "
                f"{self._node('ctrl_pos')} {self._node('ctrl_neg')} {gain}\n")

VCVS = VoltageControlledVoltageSource


class VoltageControlledCurrentSource(SourceBase):
    """Linear voltage-controlled current source (SPICE G element).

    The output current equals ``gm`` times the control voltage
    ``V(ctrl_pos, ctrl_neg)``. Also available as ``silk.sources.VCCS``.

    Parameters
    ----------
    name : str
        Instance name; see :class:`silk.sources.SourceBase` for the rules.
    out_pos : str
        Positive output terminal: a non-empty net name without whitespace, or
        a :class:`~silk.circuit.Net`.
    out_neg : str
        Negative output terminal; same forms.
    ctrl_pos : str
        Positive control node; same forms as *out_pos*.
    ctrl_neg : str
        Negative control node; same forms as *out_pos*.
    gm : float
        Transconductance in siemens (A/V): a number, SPICE value string or
        :class:`~silk.circuit.Variable`.

    Raises
    ------
    silk.exceptions.CircuitError
        If *name* is not a valid instance name.
    ValueError
        If a terminal is empty or contains whitespace.
    TypeError
        If a terminal is not a net name.

    Notes
    -----
    Each parameter is stored as an attribute of the same name.

    Examples
    --------
    >>> from silk.sources import VoltageControlledCurrentSource
    >>> g = VoltageControlledCurrentSource('gm1', out_pos='iout', out_neg='gnd',
    ...                                    ctrl_pos='vin', ctrl_neg='gnd', gm=0.01)
    >>> print(g.to_spice(), end='')
    Ggm1 iout gnd vin gnd 0.01
    """
    _spice_prefix = 'G'
    _node_terminals = ('out_pos', 'out_neg', 'ctrl_pos', 'ctrl_neg')
    _card_fields = ('gm',)
    _param_names = ('gm',)

    def __init__(self, name, out_pos, out_neg, ctrl_pos, ctrl_neg, gm):
        super().__init__(name)
        self.out_pos  = out_pos
        self.out_neg  = out_neg
        self.ctrl_pos = ctrl_pos
        self.ctrl_neg = ctrl_neg
        self.gm       = gm

    def to_spice(self):
        """The SPICE G card.

        Returns
        -------
        str
            ``G<name> out_pos out_neg ctrl_pos ctrl_neg <gm>`` followed by a
            newline; ``gm`` is written with ``str()`` (strings verbatim).

        Raises
        ------
        ValueError
            If a terminal is empty or contains whitespace, or a Variable has
            no value assigned.
        TypeError
            If a terminal is not a net name.
        silk.exceptions.DeckTextError
            If its ``gm`` contains a line break (also refused when the
            attribute is assigned).
        """
        gm = self._card_value(_resolve_value(self.gm, 'VCCS gm'), 'gm')
        return (f"{self.instance_name} {self._node('out_pos')} {self._node('out_neg')} "
                f"{self._node('ctrl_pos')} {self._node('ctrl_neg')} {gm}\n")

VCCS = VoltageControlledCurrentSource


class CurrentControlledVoltageSource(SourceBase):
    """Linear current-controlled voltage source (SPICE H element).

    The output voltage equals ``transresistance`` times the current through
    the controlling voltage source. Also available as ``silk.sources.CCVS``.

    ``ctrl_vname`` is the PYTHON INSTANCE NAME of the zero-valued ammeter
    voltage source that the controlling current flows through (for example
    ``'vmeas'`` for ``sources.vdc('vmeas', ...)``), the same instance-name
    reference every other silk API uses. silk resolves it to that source's
    SPICE element name (``'Vvmeas'``) at deck emission through the source's
    owning module. A ``ctrl_vname`` that names no instance there (it is
    already a SPICE element name, or the source is not yet attached to a
    module) is emitted verbatim. The caller is responsible for placing that
    source in the circuit.

    Parameters
    ----------
    name : str
        Instance name; see :class:`silk.sources.SourceBase` for the rules.
    out_pos : str
        Positive output terminal: a non-empty net name without whitespace, or
        a :class:`~silk.circuit.Net`.
    out_neg : str
        Negative output terminal; same forms.
    transresistance : float
        Transresistance in ohms (V/A): a number, SPICE value string or
        :class:`~silk.circuit.Variable`.
    ctrl_vname : str
        Instance name of the controlling voltage source (see above). It is not
        a node and is not validated as a terminal.

    Raises
    ------
    silk.exceptions.CircuitError
        If *name* is not a valid instance name.
    ValueError
        If a terminal is empty or contains whitespace.
    TypeError
        If a terminal is not a net name.

    Notes
    -----
    Each parameter is stored as an attribute of the same name.

    Examples
    --------
    >>> from silk.sources import CurrentControlledVoltageSource
    >>> h = CurrentControlledVoltageSource('htrans', out_pos='vout', out_neg='gnd',
    ...                                    transresistance=1000, ctrl_vname='vmeas')
    >>> print(h.to_spice(), end='')
    Hhtrans vout gnd vmeas 1000
    """
    _spice_prefix = 'H'
    _node_terminals = ('out_pos', 'out_neg')
    _card_fields = ('transresistance', 'ctrl_vname')
    _param_names = ('transresistance',)

    def __init__(self, name, out_pos, out_neg, transresistance, ctrl_vname):
        super().__init__(name)
        self.out_pos         = out_pos
        self.out_neg         = out_neg
        self.transresistance = transresistance
        self.ctrl_vname      = ctrl_vname

    def to_spice(self):
        """The SPICE H card.

        Returns
        -------
        str
            ``H<name> out_pos out_neg <ctrl> <transresistance>`` followed by
            a newline, where ``<ctrl>`` is the resolved SPICE element name of
            the controlling source (or ``ctrl_vname`` verbatim).

        Raises
        ------
        silk.exceptions.CircuitError
            If ``ctrl_vname`` names a source in the owning module that is not
            a voltage-source branch.
        ValueError
            If a terminal is empty or contains whitespace, or a Variable has
            no value assigned.
        TypeError
            If a terminal is not a net name.
        silk.exceptions.DeckTextError
            If its ``ctrl_vname`` or ``transresistance`` contains a line break (also refused when the
            attribute is assigned).
        """
        ctrl = self._card_value(self._resolve_ctrl_vname(self.ctrl_vname),
                                'ctrl_vname')
        transresistance = self._card_value(
            _resolve_value(self.transresistance, 'CCVS transresistance'),
            'transresistance')
        return (f"{self.instance_name} {self._node('out_pos')} {self._node('out_neg')} "
                f"{ctrl} {transresistance}\n")

CCVS = CurrentControlledVoltageSource


class CurrentControlledCurrentSource(SourceBase):
    """Linear current-controlled current source (SPICE F element).

    The output current equals ``gain`` times the current through the
    controlling voltage source. Also available as ``silk.sources.CCCS``.

    ``ctrl_vname`` is the PYTHON INSTANCE NAME of the zero-valued ammeter
    voltage source that the controlling current flows through; see
    :class:`silk.sources.CurrentControlledVoltageSource` for the
    instance-name-to-SPICE-element-name resolution silk applies at deck
    emission.

    Parameters
    ----------
    name : str
        Instance name; see :class:`silk.sources.SourceBase` for the rules.
    out_pos : str
        Positive output terminal: a non-empty net name without whitespace, or
        a :class:`~silk.circuit.Net`.
    out_neg : str
        Negative output terminal; same forms.
    gain : float
        Current gain in A/A (dimensionless): a number, SPICE value string or
        :class:`~silk.circuit.Variable`.
    ctrl_vname : str
        Instance name of the controlling voltage source. It is not a node and
        is not validated as a terminal.

    Raises
    ------
    silk.exceptions.CircuitError
        If *name* is not a valid instance name.
    ValueError
        If a terminal is empty or contains whitespace.
    TypeError
        If a terminal is not a net name.

    Notes
    -----
    Each parameter is stored as an attribute of the same name.

    Examples
    --------
    >>> from silk.sources import CurrentControlledCurrentSource
    >>> f = CurrentControlledCurrentSource('fmir', out_pos='icopy', out_neg='gnd',
    ...                                    gain=2, ctrl_vname='vmeas')
    >>> print(f.to_spice(), end='')
    Ffmir icopy gnd vmeas 2
    """
    _spice_prefix = 'F'
    _node_terminals = ('out_pos', 'out_neg')
    _card_fields = ('gain', 'ctrl_vname')
    _param_names = ('gain',)

    def __init__(self, name, out_pos, out_neg, gain, ctrl_vname):
        super().__init__(name)
        self.out_pos    = out_pos
        self.out_neg    = out_neg
        self.gain       = gain
        self.ctrl_vname = ctrl_vname

    def to_spice(self):
        """The SPICE F card.

        Returns
        -------
        str
            ``F<name> out_pos out_neg <ctrl> <gain>`` followed by a newline,
            where ``<ctrl>`` is the resolved SPICE element name of the
            controlling source (or ``ctrl_vname`` verbatim).

        Raises
        ------
        silk.exceptions.CircuitError
            If ``ctrl_vname`` names a source in the owning module that is not
            a voltage-source branch.
        ValueError
            If a terminal is empty or contains whitespace, or a Variable has
            no value assigned.
        TypeError
            If a terminal is not a net name.
        silk.exceptions.DeckTextError
            If its ``ctrl_vname`` or ``gain`` contains a line break (also refused when the
            attribute is assigned).
        """
        ctrl = self._card_value(self._resolve_ctrl_vname(self.ctrl_vname),
                                'ctrl_vname')
        gain = self._card_value(_resolve_value(self.gain, 'CCCS gain'), 'gain')
        return (f"{self.instance_name} {self._node('out_pos')} {self._node('out_neg')} "
                f"{ctrl} {gain}\n")

CCCS = CurrentControlledCurrentSource


# ---------------------------------------------------------------------------
# Behavioral sources
# ---------------------------------------------------------------------------

class BehavioralVoltage(SourceBase):
    """Behavioral voltage source (SPICE B element with ``V=expression``).

    The expression may reference node voltages ``v(n)`` and ``v(n1,n2)``,
    branch currents ``i(vname)``, math functions, and the special variables
    ``time``, ``temper`` and ``hertz``, as the simulator's B-source syntax
    allows. Parameters are referenced as ``{param_name}``.

    Parameters
    ----------
    name : str
        Instance name; see :class:`silk.sources.SourceBase` for the rules.
    positive : str
        Positive terminal: a non-empty net name without whitespace, or a
        :class:`~silk.circuit.Net`.
    negative : str
        Negative terminal; same forms.
    expression : str
        The simulator-syntax voltage expression written verbatim between braces on the card; it is
        not parsed or validated by silk.

    Raises
    ------
    silk.exceptions.CircuitError
        If *name* is not a valid instance name.
    ValueError
        If a terminal is empty or contains whitespace.
    TypeError
        If a terminal is not a net name.

    Examples
    --------
    >>> from silk.sources import BehavioralVoltage
    >>> b = BehavioralVoltage('bdiff', positive='vd', negative='gnd',
    ...                       expression='v(inp)-v(inn)')
    >>> print(b.to_spice(), end='')
    Bbdiff vd gnd V={v(inp)-v(inn)}
    """
    _spice_prefix = 'B'
    _node_terminals = ('positive', 'negative')
    _card_fields = ('expression',)

    def __init__(self, name, positive, negative, expression):
        super().__init__(name)
        self.positive   = positive
        self.negative   = negative
        self.expression = expression

    def to_spice(self):
        """The SPICE B card with ``V=``.

        Returns
        -------
        str
            ``B<name> positive negative V={expression}`` followed by a newline.

        Raises
        ------
        ValueError
            If a terminal is empty or contains whitespace, or a Variable has
            no value assigned.
        TypeError
            If a terminal is not a net name.
        """
        return (f"{self.instance_name} {self._node('positive')} {self._node('negative')} "
                f"V={{{self._card_value(self.expression, 'expression')}}}\n")


class BehavioralCurrent(SourceBase):
    """Behavioral current source (SPICE B element with ``I=expression``).

    The expression follows the same rules as for
    :class:`silk.sources.BehavioralVoltage`.

    Parameters
    ----------
    name : str
        Instance name; see :class:`silk.sources.SourceBase` for the rules.
    positive : str
        Positive terminal: a non-empty net name without whitespace, or a
        :class:`~silk.circuit.Net`.
    negative : str
        Negative terminal; same forms.
    expression : str
        The simulator-syntax current expression written verbatim between braces on the card; it is
        not parsed or validated by silk.

    Raises
    ------
    silk.exceptions.CircuitError
        If *name* is not a valid instance name.
    ValueError
        If a terminal is empty or contains whitespace.
    TypeError
        If a terminal is not a net name.

    Examples
    --------
    >>> from silk.sources import BehavioralCurrent
    >>> b = BehavioralCurrent('bgm', positive='io', negative='gnd',
    ...                       expression='0.01*v(vin,gnd)')
    >>> print(b.to_spice(), end='')
    Bbgm io gnd I={0.01*v(vin,gnd)}
    """
    _spice_prefix = 'B'
    _node_terminals = ('positive', 'negative')
    _card_fields = ('expression',)

    def __init__(self, name, positive, negative, expression):
        super().__init__(name)
        self.positive   = positive
        self.negative   = negative
        self.expression = expression

    def to_spice(self):
        """The SPICE B card with ``I=``.

        Returns
        -------
        str
            ``B<name> positive negative I={expression}`` followed by a
            newline.

        Raises
        ------
        ValueError
            If a terminal is empty or contains whitespace, or a Variable has
            no value assigned.
        TypeError
            If a terminal is not a net name.
        """
        return (f"{self.instance_name} {self._node('positive')} {self._node('negative')} "
                f"I={{{self._card_value(self.expression, 'expression')}}}\n")
