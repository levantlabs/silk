"""silk waveform shape classes for SPICE source definitions."""

# Source classes for silk
# Each class represents one SPICE source element and owns its SPICE generation.
#
# Import via:
#   from silk import sources
#   ckt.add_source(sources.VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8))

import functools
from enum import Enum, IntEnum

from silk.elements.factory import ParamDescriptor
from silk.utils._utils import _parse_si, _resolve_value
from silk.utils.deck_text import card_text, refuse_line_breaks


# ---------------------------------------------------------------------------
# Waveform classes (used as arguments to VoltageSource / CurrentSource)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _builtin_card_fields(cls):
    """The constructor fields of the silk waveform whose ``to_spice()`` *cls*
    uses, or none when that emitter is not silk's own."""
    import inspect
    owner = next((c for c in cls.__mro__ if 'to_spice' in vars(c)), None)
    if owner is None or owner.__module__ != __name__:
        return ()
    try:
        params = inspect.signature(owner.__init__).parameters
    except (TypeError, ValueError):
        return ()
    return tuple(p for p, spec in params.items()
                 if p != 'self' and spec.kind in (spec.POSITIONAL_OR_KEYWORD,
                                                  spec.KEYWORD_ONLY))


class _Required(Enum):
    ARGUMENT = 'required'


REQUIRED = _Required.ARGUMENT


class _Waveform:
    """Shared parameter handling for every waveform class.

    A waveform parameter accepts the same things a device parameter and a
    source's ``dc=`` accept: a number, a SPICE value string, or a
    :class:`~silk.circuit.Variable` (or an expression over one,
    ``vampl * 2``). The variable is resolved to its *current* value at
    emission time, through :func:`silk.utils._utils._resolve_value` — the one
    resolver ``dc=`` uses — so an amplitude in a
    :class:`silk.optimize.VariableSpace` tracks ``space.apply()`` deck by deck. It is
    never resolved at construction: that would freeze the value the sweep
    exists to change.

    Before this, a waveform stringified the object itself and wrote
    ``SIN(0.75 Variable(name='vampl', value=0.02) 2.7e7 0 0)`` into the deck,
    which ngspice rejects with "Undefined parameter".

    Only the *resolution* is shared with the device path, not the formatting:
    these classes render with ``str()`` while devices render through ``_fmt``,
    which would rewrite ``27343750.0`` as ``27.3438Meg``. A plain number or
    string therefore reaches the deck byte-identically to before.
    """

    def __setattr__(self, name, value):
        # A field silk's own emitter writes into the source's card: a line
        # break in a string would start a new deck line (TASK-593). Checked
        # on the RAW string, before any SPICE number parsing, which strips
        # its input. _resolve checks again at emission. A subclass with its
        # own to_spice() owns its text and is left alone.
        if name in _builtin_card_fields(type(self)):
            refuse_line_breaks(value, who=f'{type(self).__name__} {name}')
        object.__setattr__(self, name, value)

    def _resolve(self, name):
        """The current value of parameter *name* — a number for a variable,
        the value as authored for anything else."""
        who = f'{type(self).__name__} {name}'
        return refuse_line_breaks(_resolve_value(getattr(self, name), who),
                                  who=who)

    def _initial_number(self, name):
        """Resolve a finite numeric parameter needed by an initial-value rule."""
        import math
        from silk.exceptions import NotSupportedError
        from silk.exceptions import DeckTextError
        try:
            value = _parse_si(self._resolve(name))
            if math.isfinite(value):
                return value
        except DeckTextError:
            raise
        except (TypeError, ValueError):
            pass
        raise NotSupportedError(
            f'{type(self).__name__}.initial_value() needs a finite numeric '
            f'{name}; supply numeric waveform parameters.')

    def _check_initial_delays(self, *names):
        """Negative delays move t=0 inside a transition; do not guess its bias."""
        from silk.exceptions import NotSupportedError
        if any(self._initial_number(name) < 0 for name in names):
            raise NotSupportedError(
                f'{type(self).__name__}.initial_value() does not support negative '
                'delay; use nonnegative delays.')

    def _tokens(self, *names):
        """The SPICE tokens for *names*, in order, each resolved first."""
        return [card_text(self._resolve(n), who=f'{type(self).__name__} {n}')
                for n in names]


def _pwl_number(val):
    """Coerce one PWL coordinate at construction time.

    Plain numbers become floats exactly as they always did, so ``pwl[0]`` and
    the emitted ``PWL(...)`` text are unchanged. A Variable (or an
    expression over one) is stored **as authored** and resolved at emission,
    so a swept time or value tracks ``space.apply()``.
    """
    if hasattr(val, 'value'):      # Variable / _VarExpr duck-type
        return val
    # The raw string, before float() strips a break away (TASK-593).
    refuse_line_breaks(val, who='PWL point')
    return float(val)


class PWL(_Waveform):
    """Piecewise-linear waveform — a sequence of (time, value) points.

    The value between two points is linearly interpolated. Times are in
    seconds; values are in volts or amps, according to the source the
    waveform is given to. Points are kept in the order added but are always
    read (``points``, indexing, iteration, ``to_spice``, ``to_numpy``) sorted
    by their resolved time.

    Parameters
    ----------
    points : list, optional
        ``(time, value)`` pairs, or a flat list ``[t0, v0, t1, v1, ...]``.
        Each coordinate is a number, a string that ``float()`` accepts
        (``'1e-9'``; SI suffixes such as ``'1n'`` are not parsed and raise
        ``ValueError``) or a :class:`~silk.circuit.Variable`. Numbers are
        stored as ``float``; a variable is kept as authored and resolved to its current
        value when read. ``None`` (the default) or an empty list starts the
        waveform with no points.
    r : float, optional
        Repeat-start time in seconds: the waveform repeats from the point at
        this time. Emitted as ``r=<value>``; ``None`` omits it. SPICE support
        depends on backend and source quantity; see the source API support
        matrix. ``to_numpy`` requires it to equal the time of a point other
        than the last.
    td : float, optional
        Global delay in seconds. Emitted as ``td=<value>``; ``None`` omits it.
        ``to_numpy`` requires a finite, nonnegative delay.

    Raises
    ------
    ValueError
        If a pair in *points* does not have exactly two items, or a flat list
        has an odd length. A coordinate that ``float()`` cannot convert raises
        that conversion's own error.
    silk.exceptions.DeckTextError
        A subclass of ``ValueError``: a string coordinate, or a string ``r``
        or ``td``, contains a line break. String ``r`` and ``td`` are checked
        on assignment, including during construction
        (``_Waveform.__setattr__``), and resolved values and emitted text are
        checked again when the waveform is rendered.

    Notes
    -----
    Each parameter is stored as an attribute of the same name; ``r`` and
    ``td`` may be changed after construction.

    Build incrementally::

        pwl = PWL()
        pwl.add(0, 0)
        pwl.add(1e-9, 0)
        pwl.add(2e-9, 1.8)

    Or from a list of tuples::

        pwl = PWL([(0, 0), (1e-9, 0), (2e-9, 1.8)])

    Or chain calls::

        pwl = PWL().add(0, 0).add(1e-9, 0).add(2e-9, 1.8)

    Supports len(), iteration, and indexing::

        len(pwl)      # 3
        pwl[1]        # (1e-9, 0)
        for t, v in pwl: ...

    Examples
    --------
    The waveform interpolates its own values. A source's ``dc=`` is added
    separately; it does not replace the waveform's first point.

    >>> import numpy as np
    >>> from silk.sources import PWL, VoltageSource
    >>> ramp = PWL().add(0, 0.2).add(1e-6, 0.8)
    >>> ramp.points
    [(0.0, 0.2), (1e-06, 0.8)]
    >>> src = VoltageSource('ramp', positive='out', negative='gnd', dc=0.5, pwl=ramp)
    >>> src.physical_instance_names()
    ('Vramp__silk_pwl', 'Vramp__silk_dc')
    >>> src.to_numpy(np.array([0, 0.5e-6, 1e-6])).round(2).tolist()
    [0.7, 1.0, 1.3]
    >>> round(src.initial_value(), 2)
    0.7
    >>> bool(np.isclose(src.initial_value(), src.to_numpy(np.array([0]))[0]))
    True
    """

    def __init__(self, points=None, r=None, td=None):
        self._points = []
        self.r = r
        self.td = td
        if points is not None:
            self._init_from(points)

    def _init_from(self, points):
        """Parse points from flat list or list of tuples."""
        if not points:
            return
        # Detect format: flat list vs list of tuples
        if isinstance(points[0], (tuple, list)):
            for p in points:
                if len(p) != 2:
                    raise ValueError(f"PWL point must be (time, value), got {p!r}")
                self._points.append((_pwl_number(p[0]), _pwl_number(p[1])))
        else:
            # Flat list: [t0, v0, t1, v1, ...]
            flat = list(points)
            if len(flat) % 2 != 0:
                raise ValueError(
                    f"Flat PWL list must have even length (time-value pairs), got {len(flat)}"
                )
            for i in range(0, len(flat), 2):
                self._points.append(
                    (_pwl_number(flat[i]), _pwl_number(flat[i + 1])))

    def add(self, time, value):
        """Add a ``(time, value)`` point.

        Parameters
        ----------
        time : float
            Point time in seconds: a number, a ``float()``-convertible string
            (no SI suffixes) or a :class:`~silk.circuit.Variable`.
        value : float
            Level at that time, in volts or amps according to the source the
            waveform is given to; same accepted forms as *time*.

        Returns
        -------
        PWL
            This waveform, so calls can be chained. Numbers are converted with
            ``float()``; a variable is stored as authored. A value that
            ``float()`` cannot convert raises that conversion's own error.
        """
        self._points.append((_pwl_number(time), _pwl_number(value)))
        return self

    def _resolved_points(self):
        """The points as the deck will see them: every coordinate resolved to
        its current number, ordered by resolved time.

        Ordering is deferred to read time rather than done in ``add`` because a
        time can be a Variable, and the order the deck needs is the order
        of the values the variables hold *now*. For plain numbers this is the
        same list eager sorting produced, element for element.
        """
        resolved = [(refuse_line_breaks(_resolve_value(t, 'PWL time'),
                                        who='PWL time'),
                     refuse_line_breaks(_resolve_value(v, 'PWL value'),
                                        who='PWL value'))
                    for t, v in self._points]
        return sorted(resolved, key=lambda p: float(p[0]))

    @property
    def points(self):
        """The ``(time, value)`` points as a list, sorted by time.

        Times are in seconds and values in volts or amps. Coordinates given
        as Variables are returned resolved to their current values, which is
        what a caller reading points back wants. A variable with no value
        assigned raises ``ValueError``.
        """
        return self._resolved_points()

    def initial_value(self):
        """DC default: hold the first point's value, including before its time.

        Returns
        -------
        float
            The value, in volts or amps, of the point with the smallest
            resolved time. It is returned as resolved, without further
            coercion, so it is the same token ``to_spice`` writes.

        Raises
        ------
        ValueError
            If the waveform has no points, or a variable coordinate has no
            value assigned.
        """
        if not self._points:
            raise ValueError("PWL has no points")
        return self._resolved_points()[0][1]

    def to_spice(self):
        """The SPICE ``PWL(...)`` text for this waveform.

        Returns
        -------
        str
            ``PWL(t0 v0 t1 v1 ...)`` with the points sorted by resolved time,
            followed by `` r=<value>`` and `` td=<value>`` when those are set.
            Numbers are written with ``str()``, strings verbatim.

        Raises
        ------
        ValueError
            If the waveform has no points, or a variable has no value
            assigned.
        """
        if not self._points:
            raise ValueError("PWL has no points")
        inner = ' '.join(f"{card_text(t, who='PWL time')} "
                         f"{card_text(v, who='PWL value')}"
                         for t, v in self._resolved_points())
        result = f'PWL({inner})'
        if self.r is not None:
            result += f' r={card_text(self._resolve("r"), who="PWL r")}'
        if self.td is not None:
            result += f' td={card_text(self._resolve("td"), who="PWL td")}'
        return result

    def __len__(self):
        return len(self._points)

    def __getitem__(self, index):
        return self._resolved_points()[index]

    def __iter__(self):
        return iter(self._resolved_points())

    def __repr__(self):
        # Deliberately the points AS AUTHORED: a repr must not raise, and an
        # unresolved variable is exactly what the reader wants to see here.
        return f'PWL({self._points!r})'

    def to_numpy(self, t):
        """Interpolate after td, holding endpoints and repeating the tail at r.

        The first traversal includes its final endpoint; later exact repeat
        joins select the repeat-start point, matching ngspice voltage PWL.
        Before the first point (after the delay) the first value is held, and
        after the last point the last value is held unless ``r`` is set.

        Parameters
        ----------
        t : array-like
            Sample times in seconds.

        Returns
        -------
        np.ndarray
            Levels in volts or amps, as floats, with the shape of *t*. The
            delay ``td`` shifts the whole waveform; only the tail from ``r``
            to the last point repeats, not the delay.

        Raises
        ------
        ValueError
            If the waveform has no points; if a time or value is not finite,
            the first time is negative or the times are not strictly
            increasing; if ``td`` is not finite and nonnegative; or if ``r``
            is not finite or does not equal the time of a point before the
            last one.
        """
        import numpy as np
        if not self._points:
            raise ValueError("PWL has no points")
        pts = self._resolved_points()
        times = np.array([float(p[0]) for p in pts])
        values = np.array([float(p[1]) for p in pts])
        if (not np.all(np.isfinite(times)) or not np.all(np.isfinite(values))
                or times[0] < 0 or np.any(np.diff(times) <= 0)):
            raise ValueError('PWL requires finite values and strictly increasing, '
                             'nonnegative point times')
        delay = 0 if self.td is None else _parse_si(self._resolve('td'))
        if not np.isfinite(delay) or delay < 0:
            raise ValueError('PWL td must be finite and nonnegative')
        local = np.asarray(t, dtype=float) - delay
        if self.r is not None:
            repeat = _parse_si(self._resolve('r'))
            if (not np.isfinite(repeat) or repeat not in times[:-1]):
                raise ValueError('PWL r must equal a point time before the last point')
            # ngspice includes the final point on the first traversal. Later
            # joins start at r; only the tail [r, last] repeats (not the delay).
            local = np.where(local > times[-1],
                             repeat + np.mod(local - times[-1], times[-1] - repeat),
                             local)
        return np.interp(local, times, values)


class Pulse(_Waveform):
    """PULSE(V1 V2 TD TR TF PW PER NP)

    ``tr`` and ``tf`` are **mandatory and must be > 0**, and are keyword-only
    so an edge time can never be supplied by accident of position::

        Pulse(v1=0, v2=1.8, td=0, tr=100e-12, tf=100e-12, pw='25n', per='50n')

    A zero edge is not a well-defined waveform — it is resolved differently by
    every consumer. ngspice substitutes the transient step ``TSTEP`` for a zero
    ``TR``/``TF`` (SPICE3 inheritance) and ramps over it, while silk's
    behavioral :meth:`to_numpy` steps instantaneously. A deck written with
    ``tr=0`` therefore runs its behavioral twin and its SPICE netlist in
    opposite clock phases at ``t=0``, and skewed by ``TSTEP`` at every edge
    after that. Zero-width edges are also a Newton-convergence hazard. Passing
    a real edge time removes the ambiguity by construction: both sides ramp
    over ``tr``, and both agree.

    Times accept plain numbers or SI-suffixed strings (``100e-12`` or
    ``'100p'``); magnitudes are parsed the way SPICE parses them.

    Parameters
    ----------
    v1 : float
        Initial (low) level, in volts or amps according to the source the
        waveform is given to. A number, SPICE value string or
        :class:`~silk.circuit.Variable`; the same forms are accepted
        by every other parameter.
    v2 : float
        Pulsed (high) level, in volts or amps.
    td : float, optional
        Delay before the first rising edge, in seconds. Default ``0``.
    tr : float
        Rise time in seconds; keyword-only, must be greater than zero.
    tf : float
        Fall time in seconds; keyword-only, must be greater than zero.
    pw : float, optional
        Time held at ``v2`` between the edges, in seconds. Default ``0``,
        written to the card as ``0``.
    per : float, optional
        Period in seconds. Default ``0``, written to the card as ``0``;
        ``to_numpy`` treats a period that is not positive as a single,
        non-repeating pulse.
    np : int, optional
        Number of pulses, keyword-only. ``None`` (the default) omits the
        count from the card; ``0`` or ``None`` mean unlimited. Otherwise a
        finite nonnegative integer.

    Raises
    ------
    ValueError
        If ``tr`` or ``tf`` is not a valid time or is not greater than zero.
        The same check runs again on every ``to_spice`` and ``to_numpy``
        call, so a variable moved to zero after construction is still
        caught.

    Notes
    -----
    Each parameter is stored as an attribute of the same name, as authored
    (strings and variables are not converted).

    Examples
    --------
    Choose positive edge times explicitly. The waveform starts at ``v1``;
    a source's separate ``dc=`` offset shifts both pulse levels.

    >>> import numpy as np
    >>> from silk.sources import Pulse, VoltageSource
    >>> clock = Pulse(v1=0, v2=0.6, tr='1n', tf='1n', pw='5n', per='10n')
    >>> clock.to_spice()
    'PULSE(0 0.6 0 1n 1n 5n 10n)'
    >>> src = VoltageSource('clock', positive='out', negative='gnd', dc=0.2, pulse=clock)
    >>> src.to_numpy(np.array([0, 1e-9, 3e-9])).round(2).tolist()
    [0.2, 0.8, 0.8]
    >>> src.initial_value()
    0.2
    >>> bool(np.isclose(src.initial_value(), src.to_numpy(np.array([0]))[0]))
    True
    """

    def __init__(self, v1, v2, td=0, *, tr, tf, pw=0, per=0, np=None):
        self.v1  = v1
        self.v2  = v2
        self.td  = td
        self.tr  = self._check_edge('tr', tr)
        self.tf  = self._check_edge('tf', tf)
        self.pw  = pw
        self.per = per
        self.np  = np   # None/0 = unlimited; None omits the SPICE count token

    @staticmethod
    def _check_edge(name, value):
        """Reject a zero/negative edge time. Returns *value* unchanged (as
        authored, so SI strings survive into the netlist verbatim, and a
        Variable stays a variable so later ``space.apply()`` calls still
        reach it). A variable is checked at its value of the moment, and
        re-checked from ``to_spice`` — so a sweep cannot slip a zero edge into
        a deck between construction and emission."""
        # Raw and resolved text, before _parse_si strips it, and outside the
        # catch below so DeckTextError survives as itself (TASK-593).
        refuse_line_breaks(value, who=f'Pulse {name}')
        resolved = refuse_line_breaks(_resolve_value(value, f'Pulse {name}'),
                                      who=f'Pulse {name}')
        try:
            seconds = _parse_si(resolved)
        except ValueError as exc:
            raise ValueError(
                f'Pulse {name}={value!r} is not a valid time: {exc}'
            ) from None
        if seconds <= 0:
            raise ValueError(
                f'Pulse {name}={value!r} must be > 0. A zero rise/fall time is '
                f'simulator-dependent: ngspice substitutes the transient step '
                f'TSTEP for it and ramps, while silk\'s behavioral to_numpy() '
                f'steps instantaneously — so the twin and the SPICE deck '
                f'disagree at t=0 and skew by TSTEP at every later edge. '
                f'Zero-width edges are also a Newton-convergence hazard. Pass '
                f'a real edge time — a small fraction of this circuit\'s own '
                f'period or pulse width, e.g. {name}=100e-12 or {name}=\'100p\'.'
            )
        return value

    def _pulse_count(self):
        """Resolve the count; zero and None mean unlimited in ngspice."""
        import math
        if self.np is None:
            return 0
        value = self._resolve('np')
        count = _parse_si(value)
        if (isinstance(value, bool) or not math.isfinite(count)
                or count < 0 or not count.is_integer()):
            raise ValueError('Pulse np must be a finite nonnegative integer '
                             '(0 or None means unlimited)')
        return count

    def initial_value(self):
        """ngspice's operating-point default is V1 for a nonnegative delay.

        Returns
        -------
        float
            ``v1`` as resolved (volts or amps); a string stays a string.

        Raises
        ------
        silk.exceptions.NotSupportedError
            If ``td`` is negative, not finite or not numeric.
        """
        self._check_initial_delays('td')
        return self._resolve('v1')

    def to_spice(self):
        """The SPICE ``PULSE(...)`` text for this waveform.

        Returns
        -------
        str
            ``PULSE(v1 v2 td tr tf pw per)``, with a trailing ``np`` token
            only when ``np`` is not ``None``. Values are written as authored
            (numbers with ``str()``, strings verbatim), so a ``0`` for ``pw``
            or ``per`` is emitted as ``0``.

        Raises
        ------
        ValueError
            If ``tr`` or ``tf`` is no longer greater than zero, or ``np`` is
            not a finite nonnegative integer, or a variable has no value.
        """
        # Re-check the edges against their CURRENT values: tr/tf may be
        # Variables that a sweep has moved since construction.
        self._check_edge('tr', self.tr)
        self._check_edge('tf', self.tf)
        self._pulse_count()
        parts = self._tokens('v1', 'v2', 'td', 'tr', 'tf', 'pw', 'per')
        if self.np is not None:
            parts += self._tokens('np')
        return 'PULSE(' + ' '.join(parts) + ')'

    def to_numpy(self, t):
        """Generate numpy array of pulse waveform values at times t.

        Every parameter is parsed through the SPICE value parser, so a pulse
        written with SI strings (``tr='1n'``) models identically to the same
        pulse written with floats — and to the netlist it emits.

        Parameters
        ----------
        t : array-like
            Sample times in seconds.

        Returns
        -------
        np.ndarray
            Levels in volts or amps, as floats, with the shape of *t*. The
            value is ``v1`` before ``td``, ramps linearly to ``v2`` over
            ``tr``, holds for ``pw``, ramps back over ``tf`` and returns to
            ``v1``; with a positive ``per`` the cycle repeats, and with a
            positive ``np`` it stops after ``np`` periods.

        Raises
        ------
        ValueError
            If ``tr`` or ``tf`` is not greater than zero. With a positive
            ``np`` it also requires an explicit positive ``per``, finite
            levels and times, ``td >= 0``, ``pw > 0`` and
            ``tr + pw + tf <= per``.
        """
        import numpy as np
        self._check_edge('tr', self.tr)
        self._check_edge('tf', self.tf)
        count = self._pulse_count()
        td, tr, tf = (_parse_si(self._resolve('td')),
                      _parse_si(self._resolve('tr')),
                      _parse_si(self._resolve('tf')))
        pw, per = _parse_si(self._resolve('pw')), _parse_si(self._resolve('per'))
        v1, v2 = _parse_si(self._resolve('v1')), _parse_si(self._resolve('v2'))

        if count and (not np.isfinite(per) or per <= 0):
            raise ValueError('Pulse finite np requires an explicit positive per '
                             'for to_numpy; simulator default periods depend on the analysis')
        duration = tr + pw + tf
        overlaps = duration > per and not np.isclose(duration, per, rtol=1e-14, atol=0)
        if count and (not np.all(np.isfinite([td, tr, tf, pw, v1, v2]))
                      or td < 0 or pw <= 0 or overlaps):
            raise ValueError('Pulse finite np requires finite levels, td >= 0, '
                             'pw > 0 and tr + pw + tf <= per for to_numpy')
        t = np.asarray(t, dtype=float)
        result = np.full_like(t, v1)

        active = t >= td
        if count:
            active &= t < td + count * per
        if not np.any(active):
            return result

        ta = t[active] - td

        if per > 0:
            ta = np.mod(ta, per)

        seg = np.full(int(np.sum(active)), v1, dtype=float)

        # tr > 0 is guaranteed by the constructor, so the ramp is always the
        # first segment — this is what keeps the twin in phase with SPICE.
        mask_rise = (ta >= 0) & (ta < tr)
        seg[mask_rise] = v1 + (v2 - v1) * ta[mask_rise] / tr

        t_high_end = tr + pw
        mask_high = (ta >= tr) & (ta < t_high_end)
        seg[mask_high] = v2

        t_fall_end = t_high_end + tf
        mask_fall = (ta >= t_high_end) & (ta < t_fall_end)
        seg[mask_fall] = v2 - (v2 - v1) * (ta[mask_fall] - t_high_end) / tf

        result[active] = seg
        return result


class Sine(_Waveform):
    """SIN(VO VA FREQ TD THETA PHASE).

    ``voffset`` defaults to zero; ``vampl`` and ``freq`` are required.
    The waveform's offset and a source's separate ``dc=`` offset add.

    Parameters
    ----------
    voffset : float, optional
        Offset, in volts or amps according to the source the waveform is
        given to. Default ``0``. A number, SPICE value string or
        :class:`~silk.circuit.Variable`; the same forms are accepted
        by every other parameter.
    vampl : float
        Peak amplitude, in volts or amps. Required.
    freq : float
        Frequency in hertz. Required.
    td : float, optional
        Delay in seconds. Default ``0``.
    theta : float, optional
        Damping factor in 1/s: the amplitude is multiplied by
        ``exp(-theta * (t - td))`` after the delay. Default ``0``.
    phase : float, optional
        Phase in degrees. Default ``0``; omitted from the SPICE card when
        zero.

    Raises
    ------
    TypeError
        If ``vampl`` or ``freq`` is not supplied (or is ``None``).

    Notes
    -----
    Each parameter is stored as an attribute of the same name, as authored.

    Examples
    --------
    >>> import numpy as np
    >>> from silk.sources import Sine, VoltageSource
    >>> tone = Sine(vampl=0.2, freq=1000)
    >>> tone.voffset
    0
    >>> tone.to_spice()
    'SIN(0 0.2 1000 0 0)'
    >>> src = VoltageSource('tone', positive='out', negative='gnd', dc=0.9, sine=tone)
    >>> src.to_numpy(np.array([0, 0.00025])).round(2).tolist()
    [0.9, 1.1]
    >>> src.initial_value()
    0.9
    >>> bool(np.isclose(src.initial_value(), src.to_numpy(np.array([0]))[0]))
    True
    """
    def __init__(self, voffset=0, vampl=REQUIRED, freq=REQUIRED, td=0, theta=0, phase=0):
        for name, value in (('vampl', vampl), ('freq', freq)):
            if value is REQUIRED or value is None:
                raise TypeError(f'Sine requires {name}; supply it explicitly.')
        self.voffset = voffset
        self.vampl   = vampl
        self.freq    = freq
        self.td      = td
        self.theta   = theta
        self.phase   = phase   # degrees; omitted from SPICE line if 0

    def initial_value(self):
        """ngspice DC default: offset + amplitude*sin(phase), even with delay.

        Phase is in degrees. This is the operating-point rule, independent of
        the transient delay and damping; it is not a sample of to_numpy().

        Returns
        -------
        float
            ``voffset + vampl * sin(phase)`` in volts or amps; ``voffset`` as
            resolved when ``phase`` is zero.

        Raises
        ------
        silk.exceptions.NotSupportedError
            If ``td`` is negative or ``td``, ``phase`` (or, for a nonzero
            phase, ``voffset`` and ``vampl``) is not a finite number.
        """
        import math
        self._check_initial_delays('td')
        phase = self._initial_number('phase')
        if phase == 0:
            return self._resolve('voffset')
        return (self._initial_number('voffset')
                + self._initial_number('vampl') * math.sin(math.radians(phase)))

    def to_spice(self):
        """The SPICE ``SIN(...)`` text for this waveform.

        Returns
        -------
        str
            ``SIN(voffset vampl freq td theta)``, plus ``phase`` only when it
            is not zero. Values are written as authored.

        Raises
        ------
        ValueError
            If a variable parameter has no value assigned.
        """
        parts = self._tokens('voffset', 'vampl', 'freq', 'td', 'theta')
        if self._resolve('phase') != 0:
            parts += self._tokens('phase')
        return 'SIN(' + ' '.join(parts) + ')'

    def to_numpy(self, t):
        """Generate numpy array of sine waveform values at times t.

        Parameters
        ----------
        t : array-like
            Sample times in seconds.

        Returns
        -------
        np.ndarray
            Levels in volts or amps, with the shape of *t*. Before ``td`` the
            value is held at ``voffset + vampl * sin(phase)``; afterwards it
            is ``voffset + vampl * exp(-theta * (t - td)) *
            sin(2 * pi * freq * (t - td) + phase)``.

        Raises
        ------
        ValueError
            If a variable parameter has no value assigned or a string is not
            a valid SPICE number.
        """
        import numpy as np
        # Resolve first: a Variable multiplied by a numpy array would
        # build a lazy _VarExpr instead of a waveform.
        voffset = _parse_si(self._resolve('voffset'))
        vampl = _parse_si(self._resolve('vampl'))
        freq = _parse_si(self._resolve('freq'))
        td = _parse_si(self._resolve('td'))
        theta = _parse_si(self._resolve('theta'))
        phase = _parse_si(self._resolve('phase'))

        t = np.asarray(t, dtype=float)
        result = np.full_like(t, voffset + vampl * np.sin(np.deg2rad(phase)))

        active = t >= td
        if not np.any(active):
            return result

        ta = t[active] - td
        phase_rad = np.deg2rad(phase)

        if theta != 0:
            envelope = np.exp(-theta * ta)
        else:
            envelope = 1.0

        result[active] = (
            voffset
            + vampl * envelope * np.sin(2 * np.pi * freq * ta + phase_rad)
        )
        return result


class Exponential(_Waveform):
    """Exponential rise and fall: ``EXP(V1 V2 TD1 TAU1 TD2 TAU2)``.

    ``td1``/``tau1`` are the rise delay and time constant and ``td2``/``tau2``
    the fall delay and time constant. It is emitted to SPICE only: it has no numpy
    model, so a source carrying it raises ``NotImplementedError`` from
    ``to_numpy`` (behavioral simulation).

    Parameters
    ----------
    v1 : float
        Initial level, in volts or amps according to the source the waveform
        is given to. A number, SPICE value string or
        :class:`~silk.circuit.Variable`; the same forms are accepted
        by every other parameter.
    v2 : float
        Pulsed level, in volts or amps.
    td1 : float, optional
        Rise delay in seconds. Default ``0``.
    tau1 : float, optional
        Rise time constant in seconds. Default ``0``.
    td2 : float, optional
        Fall delay in seconds. Default ``0``.
    tau2 : float, optional
        Fall time constant in seconds. Default ``0``.

    Notes
    -----
    Each parameter is stored as an attribute of the same name, as authored.
    Defaults of ``0`` are written to the card as ``0``.

    Examples
    --------
    >>> from silk.sources import Exponential
    >>> Exponential(v1=0, v2=1, td1='1n', tau1='2n', td2='10n', tau2='3n').to_spice()
    'EXP(0 1 1n 2n 10n 3n)'
    """
    def __init__(self, v1, v2, td1=0, tau1=0, td2=0, tau2=0):
        self.v1   = v1
        self.v2   = v2
        self.td1  = td1
        self.tau1 = tau1
        self.td2  = td2
        self.tau2 = tau2

    def initial_value(self):
        """ngspice DC default: V1, for nonnegative transition delays.

        Returns
        -------
        float
            ``v1`` as resolved (volts or amps).

        Raises
        ------
        silk.exceptions.NotSupportedError
            If ``td1`` or ``td2`` is negative, not finite or not numeric.
        """
        self._check_initial_delays('td1', 'td2')
        return self._resolve('v1')

    def to_spice(self):
        """The SPICE ``EXP(...)`` text for this waveform.

        Returns
        -------
        str
            ``EXP(v1 v2 td1 tau1 td2 tau2)`` with the values as authored.

        Raises
        ------
        ValueError
            If a variable parameter has no value assigned.
        """
        parts = self._tokens('v1', 'v2', 'td1', 'tau1', 'td2', 'tau2')
        return 'EXP(' + ' '.join(parts) + ')'


class SFFM(_Waveform):
    """Single-frequency FM: ``SFFM(VO VA FC MDI FS PHASEC PHASES)``.

    Also available as ``silk.sources.SingleFrequencyFM``. Emitted to SPICE
    only: it has no numpy model, so a source carrying it raises
    ``NotImplementedError`` from ``to_numpy`` (behavioral simulation).

    Parameters
    ----------
    vo : float
        Offset, in volts or amps according to the source the waveform is
        given to. A number, SPICE value string or
        :class:`~silk.circuit.Variable`; the same forms are accepted
        by every other parameter.
    va : float
        Amplitude, in volts or amps.
    fc : float
        Carrier frequency in hertz.
    mdi : float
        Modulation index; the operating-point rule treats it as radians.
    fs : float
        Signal (modulating) frequency in hertz.
    phasec : float, optional
        Carrier phase in degrees. Default ``0``.
    phases : float, optional
        Signal phase in degrees. Default ``0``. The two phases are written
        to the card only if either is nonzero.

    Notes
    -----
    Each parameter is stored as an attribute of the same name, as authored.

    Examples
    --------
    >>> from silk.sources import SFFM
    >>> SFFM(vo=0, va=1, fc='1Meg', mdi=5, fs='1k').to_spice()
    'SFFM(0 1 1Meg 5 1k)'
    """
    def __init__(self, vo, va, fc, mdi, fs, phasec=0, phases=0):
        self.vo     = vo
        self.va     = va
        self.fc     = fc
        self.mdi    = mdi
        self.fs     = fs
        self.phasec = phasec   # carrier phase in degrees; omitted if 0
        self.phases = phases   # signal phase in degrees; omitted if 0

    def initial_value(self):
        """ngspice DC default: VO + VA*sin(PHASEC + MDI*sin(PHASES)).

        Both authored phases are in degrees; modulation depth is in radians.

        Returns
        -------
        float
            ``vo`` as resolved when both phases are zero; otherwise the
            computed level in volts or amps.

        Raises
        ------
        silk.exceptions.NotSupportedError
            If a parameter the rule needs is not a finite number.
        """
        import math
        phasec = self._initial_number('phasec')
        phases = self._initial_number('phases')
        if phasec == 0 and phases == 0:
            return self._resolve('vo')
        angle = (math.radians(phasec) + self._initial_number('mdi')
                 * math.sin(math.radians(phases)))
        return self._initial_number('vo') + self._initial_number('va') * math.sin(angle)

    def to_spice(self):
        """The SPICE ``SFFM(...)`` text for this waveform.

        Returns
        -------
        str
            ``SFFM(vo va fc mdi fs)``, followed by ``phasec phases`` when
            either is nonzero. Values are written as authored.

        Raises
        ------
        ValueError
            If a variable parameter has no value assigned.
        """
        parts = self._tokens('vo', 'va', 'fc', 'mdi', 'fs')
        if self._resolve('phasec') != 0 or self._resolve('phases') != 0:
            parts += self._tokens('phasec', 'phases')
        return 'SFFM(' + ' '.join(parts) + ')'

SingleFrequencyFM = SFFM


class AM(_Waveform):
    """Amplitude-modulated waveform: ``AM(VA VO MF FC TD PHASES)``.

    Also available as ``silk.sources.AmplitudeModulated``. Emitted to SPICE
    only: it has no numpy model, so a source carrying it raises
    ``NotImplementedError`` from ``to_numpy`` (behavioral simulation).

    Parameters
    ----------
    va : float
        Amplitude, in volts or amps according to the source the waveform is
        given to. A number, SPICE value string or
        :class:`~silk.circuit.Variable`; the same forms are accepted
        by every other parameter.
    vo : float
        Offset, in volts or amps.
    mf : float
        Modulating frequency in hertz.
    fc : float
        Carrier frequency in hertz.
    td : float, optional
        Delay in seconds. Default ``0``.
    phases : float, optional
        Signal phase in degrees. Default ``0``; written to the card only
        when nonzero.

    Notes
    -----
    Each parameter is stored as an attribute of the same name, as authored.
    The positional order is ``va, vo, mf, fc, td, phases``, the order of the
    SPICE card.

    Examples
    --------
    >>> from silk.sources import AM
    >>> AM(va=1, vo=0, mf='1k', fc='1Meg').to_spice()
    'AM(1 0 1k 1Meg 0)'
    """
    def __init__(self, va, vo, mf, fc, td=0, phases=0):
        self.va     = va
        self.vo     = vo
        self.mf     = mf
        self.fc     = fc
        self.td     = td
        self.phases = phases   # degrees; omitted if 0

    def initial_value(self):
        """ngspice DC default: zero (the carrier sine starts at zero).

        Xyce treats this unsupported waveform as constant DC when a DC
        clause is present; this does not add transient AM support.

        Returns
        -------
        int
            ``0``.

        Raises
        ------
        silk.exceptions.NotSupportedError
            If ``td`` is negative, not finite or not numeric.
        """
        self._check_initial_delays('td')
        return 0

    def to_spice(self):
        """The SPICE ``AM(...)`` text for this waveform.

        Returns
        -------
        str
            ``AM(va vo mf fc td)``, followed by ``phases`` when it is
            nonzero. Values are written as authored.

        Raises
        ------
        ValueError
            If a variable parameter has no value assigned.
        """
        parts = self._tokens('va', 'vo', 'mf', 'fc', 'td')
        if self._resolve('phases') != 0:
            parts += self._tokens('phases')
        return 'AM(' + ' '.join(parts) + ')'

AmplitudeModulated = AM


class TransientNoise(_Waveform):
    """Transient noise: ``TRNOISE(NA NT NALPHA NAMP RTSAM RTSCAPT RTSEMT)``.

    Emitted to SPICE only: it has no numpy model, so a source carrying it
    raises ``NotImplementedError`` from ``to_numpy`` (behavioral simulation).

    ::

        White noise only:   TransientNoise(na='20n', nt='0.5n')
        White + 1/f:        TransientNoise(na='20n', nt='0.5n', nalpha=1.1, namp='12p')
        Full (+ RTS):       TransientNoise(na=0, nt='10p', nalpha=1.1, namp='12p',
                                           rtsam='15m', rtscapt='22u', rtsemt='50u')

    Parameters
    ----------
    na : float
        White-noise amplitude, in volts or amps according to the source the
        waveform is given to. A number, SPICE value string or
        :class:`~silk.circuit.Variable`; the same forms are accepted
        by every other parameter.
    nt : float
        Time step of the noise, in seconds.
    nalpha : float, optional
        Exponent of the 1/f noise component (dimensionless). Default ``0``.
    namp : float, optional
        Amplitude of the 1/f component, in volts or amps. Default ``0``.
    rtsam : float, optional
        Random-telegraph-signal (RTS) amplitude, in volts or amps. Default
        ``0``.
    rtscapt : float, optional
        RTS trap capture time in seconds. Default ``0``.
    rtsemt : float, optional
        RTS trap emission time in seconds. Default ``0``. The three RTS values
        are written to the card only if any of them is nonzero.

    Notes
    -----
    Each parameter is stored as an attribute of the same name, as authored.

    Examples
    --------
    >>> from silk.sources import TransientNoise
    >>> TransientNoise(na='20n', nt='0.5n').to_spice()
    'TRNOISE(20n 0.5n 0 0)'
    """
    def __init__(self, na, nt, nalpha=0, namp=0, rtsam=0, rtscapt=0, rtsemt=0):
        self.na      = na
        self.nt      = nt
        self.nalpha  = nalpha
        self.namp    = namp
        self.rtsam   = rtsam    # RTS amplitude; omit RTS params if all zero
        self.rtscapt = rtscapt  # trap capture time
        self.rtsemt  = rtsemt   # trap emission time

    def initial_value(self):
        """ngspice DC default: zero, including white, flicker and RTS noise.

        No random draw is made for the operating point. Xyce treats this
        unsupported waveform as constant DC when a DC clause is present.

        Returns
        -------
        int
            ``0``.
        """
        return 0

    def to_spice(self):
        """The SPICE ``TRNOISE(...)`` text for this waveform.

        Returns
        -------
        str
            ``TRNOISE(na nt nalpha namp)``, followed by
            ``rtsam rtscapt rtsemt`` when any of them is nonzero. Values are
            written as authored.

        Raises
        ------
        ValueError
            If a variable parameter has no value assigned.
        """
        parts = self._tokens('na', 'nt', 'nalpha', 'namp')
        if (self._resolve('rtsam') != 0 or self._resolve('rtscapt') != 0
                or self._resolve('rtsemt') != 0):
            parts += self._tokens('rtsam', 'rtscapt', 'rtsemt')
        return 'TRNOISE(' + ' '.join(parts) + ')'


class RandomType(IntEnum):
    """TRRANDOM TYPE codes accepted by :class:`silk.sources.RandomSource`.

    The members are ``uniform = 1``, ``gaussian = 2``, ``exponential = 3``
    and ``poisson = 4``. Being an :class:`enum.IntEnum`, a member compares
    equal to, and is emitted as, its integer code.

    This enum is the single home for the code -> label mapping:
    ``RandomSource.__init__``'s ``type`` parameter is annotated with it,
    which is how ``silk.sources.schema.export_source_schema()`` reflects
    ``choices: [{"value": 1, "label": "uniform"}, ...]`` onto the
    ``trrandom`` waveform's ``type`` parameter — no hand-authored table to
    drift out of sync with either the emitted SPICE code or the docs (see
    docs/codegen-contract.md's RandomSource subsection).
    """
    uniform     = 1
    gaussian    = 2
    exponential = 3
    poisson     = 4


class RandomSource(_Waveform):
    """Random transient stimulus: ``TRRANDOM(TYPE TS TD PARAM1 PARAM2)``.

    ``type`` selects the distribution — see :class:`silk.sources.RandomType` for the
    codes and labels. Accepts either a plain int (``1``-``4``) or a
    ``RandomType`` member; either way it is stored and emitted as the bare
    int, so ``to_spice()`` is unchanged (``TRRANDOM(1 ...)``, not
    ``TRRANDOM(RandomType.uniform ...)``). Emitted to SPICE only: it has no
    numpy model, so a source carrying it raises ``NotImplementedError`` from
    ``to_numpy`` (behavioral simulation).

    Parameters
    ----------
    type : RandomType
        Distribution code, ``1`` to ``4``, or a :class:`silk.sources.RandomType` member.
    ts : float
        Time between random draws, in seconds. A number, SPICE value string or
        :class:`~silk.circuit.Variable`; the same forms are accepted
        by every other parameter.
    td : float, optional
        Delay in seconds. Default ``0``.
    param1 : float, optional
        First distribution parameter. Default ``1``. Its meaning depends on
        ``type`` and is defined by the simulator's ``TRRANDOM`` card.
    param2 : float, optional
        Second distribution parameter. Default ``0``. Its meaning depends on
        ``type``; ``initial_value`` returns it.

    Raises
    ------
    ValueError
        If ``type`` is not one of the :class:`silk.sources.RandomType` codes; the message
        lists the valid codes and names.

    Notes
    -----
    Each parameter is stored as an attribute of the same name (``type`` as a
    plain ``int``).

    Examples
    --------
    >>> from silk.sources import RandomSource, RandomType
    >>> RandomSource(RandomType.gaussian, ts='1n', param1='1m').to_spice()
    'TRRANDOM(2 1n 0 1m 0)'
    """
    def __init__(self, type: RandomType, ts, td=0, param1=1, param2=0):
        try:
            self.type = int(RandomType(type))
        except ValueError:
            valid = ', '.join(f'{m.value}={m.name}' for m in RandomType)
            raise ValueError(
                f"RandomSource type must be one of {valid}; got {type!r}"
            ) from None
        self.ts     = ts
        self.td     = td
        self.param1 = param1
        self.param2 = param2

    def initial_value(self):
        """ngspice DC default: PARAM2, without drawing a random sample.

        This applies to all four distributions, with or without delay.
        Xyce treats this unsupported waveform as constant DC when a DC
        clause is present.

        Returns
        -------
        float
            ``param2`` as resolved.
        """
        return self._resolve('param2')

    def to_spice(self):
        """The SPICE ``TRRANDOM(...)`` text for this waveform.

        Returns
        -------
        str
            ``TRRANDOM(type ts td param1 param2)`` with the values as
            authored and ``type`` as a bare integer.

        Raises
        ------
        ValueError
            If a variable parameter has no value assigned.
        """
        parts = self._tokens('type', 'ts', 'td', 'param1', 'param2')
        return 'TRRANDOM(' + ' '.join(parts) + ')'


# ---------------------------------------------------------------------------


class _FiniteBits(_Waveform):
    """Shared finite points for electrical bit waveforms, resolved on demand."""

    def initial_value(self):
        """Operating-point default: the level of the first bit.

        Returns
        -------
        float
            ``v2`` if the first bit is 1, otherwise ``v1``, in volts or
            amps, as resolved (a string stays a string).
        """
        return self._resolve('v2' if self.stream[0] else 'v1')

    def _set_electrical(self, v1, v2, ui, tr, tf, td):
        self.v1, self.v2, self.ui = v1, v2, ui
        self.tr, self.tf, self.td = tr, tf, td
        self._electrical(allow_unassigned=True)

    def _electrical(self, *, allow_unassigned=False):
        import math
        from silk.circuit.variable import Variable, _VarExpr
        values = {}
        for name in ('v1', 'v2', 'ui', 'tr', 'tf', 'td'):
            authored = getattr(self, name)
            # Unassigned Variables are valid declarations. Literal values are
            # checked immediately; every value is checked again before use.
            if allow_unassigned and isinstance(authored, (Variable, _VarExpr)):
                continue
            value = _parse_si(self._resolve(name))
            if not math.isfinite(value):
                raise ValueError(f'{type(self).__name__} {name} must be finite')
            if name in ('ui', 'tr', 'tf') and value <= 0:
                raise ValueError(f'{type(self).__name__} {name} must be > 0')
            if name == 'td' and value < 0:
                raise ValueError(f'{type(self).__name__} td must be >= 0')
            values[name] = value
        for edge in ('tr', 'tf'):
            if edge in values and 'ui' in values and values[edge] > values['ui']:
                raise ValueError(f'{type(self).__name__} {edge} must be <= ui')
        return values

    def _points(self):
        import math
        values = self._electrical()
        stream = self.stream
        v1, v2 = values['v1'], values['v2']
        ui, td = values['ui'], values['td']
        end = td + len(stream) * ui
        if not math.isfinite(end):
            raise ValueError(f'{type(self).__name__} ui/td produce nonfinite completion time')
        level = v2 if stream[0] else v1
        points = [(0.0, level)]

        def append(time, value):
            if time == points[-1][0] and value == points[-1][1]:
                return
            if time <= points[-1][0]:
                raise ValueError('bit timing is not representable as increasing finite points; '
                                 'reduce td or increase ui/tr/tf')
            points.append((time, value))

        for k in range(1, len(stream)):
            if stream[k] == stream[k - 1]:
                continue
            start = td + k * ui
            append(start, level)
            level = v2 if stream[k] else v1
            edge = values['tr'] if stream[k] else values['tf']
            # With edge <= ui, roundoff must not put this ramp into the
            # next bit (e.g. SI '10u' is one ULP below numeric 1e-5).
            finish = min(start + edge, td + (k + 1) * ui)
            if finish <= start:
                raise ValueError('tr/tf is too small to represent at this bit time')
            append(finish, level)
        append(end, level)
        return points

    def to_spice(self):
        """The waveform as an explicit SPICE ``PWL(...)`` point list.

        The same points are used for SPICE and for ``to_numpy``. The list
        holds one ``time level`` pair per line, joined with SPICE
        continuation lines (a newline followed by ``+``), which keeps each line short for long
        streams.

        Returns
        -------
        str
            ``PWL(0.0 <first level>\\n+ <t> <level> ...)``, times in seconds
            from the start of the waveform and levels in volts or amps.

        Raises
        ------
        ValueError
            If an electrical value is not finite or violates its limits
            (``ui``, ``tr``, ``tf`` greater than zero, ``td`` nonnegative,
            ``tr``/``tf`` no greater than ``ui``), or the timing cannot be
            represented as strictly increasing points.
        """
        return 'PWL(' + '\n+ '.join(f'{t} {v}' for t, v in self._points()) + ')'

    def to_numpy(self, t):
        """Interpolate the same finite points used by SPICE; hold both endpoints.

        Parameters
        ----------
        t : array-like
            Sample times in seconds.

        Returns
        -------
        np.ndarray
            Levels in volts or amps, as floats, with the shape of *t*. Before
            the first point and after the last the end levels are held.

        Raises
        ------
        ValueError
            As for ``to_spice``.
        """
        import numpy as np
        times, levels = zip(*self._points())
        return np.interp(np.asarray(t, dtype=float), times, levels)

    def _electrical_parameters(self):
        return {name: getattr(self, name) for name in ('v1', 'v2', 'ui', 'tr', 'tf', 'td')}


class BitPattern(_FiniteBits):
    """Finite literal bits, repeated ``repeat`` total times.

    Bit k starts at td+k*ui; hold the first level before td and the last
    level afterwards. tr/tf refer to logical 0→1/1→0 even for inverted
    v1/v2 levels. Edges must be positive and no greater than ui.

    The waveform is emitted as an explicit ``PWL(...)`` point list on both
    SPICE and the behavioral engine, so it needs no simulator-specific bit
    source. Logical 0 is level ``v1`` and logical 1 is level ``v2``.

    Parameters
    ----------
    bits : str or sequence of int
        Nonempty bit sequence: a string of ``'0'`` and ``'1'`` characters, or
        a list or tuple of the integers ``0`` and ``1``. Stored as the
        normalized string.
    v1 : float
        Voltage or current level for logical 0 (volts or amps according to
        the source it feeds); keyword-only. A number, SPICE value string or
        :class:`~silk.circuit.Variable`; the same forms are accepted
        by ``v2``, ``ui``, ``tr``, ``tf`` and ``td``.
    v2 : float
        Level for logical 1; keyword-only.
    ui : float
        Unit interval (bit duration) in seconds; keyword-only, must be
        greater than zero.
    tr : float
        Rising-edge (logical 0 to 1) transition time in seconds;
        keyword-only, greater than zero and no greater than ``ui``.
    tf : float
        Falling-edge transition time in seconds; keyword-only, same limits
        as ``tr``.
    td : float, optional
        Delay in seconds before the first bit boundary. Default ``0``; must
        be nonnegative.
    repeat : int, optional
        Total number of times the whole bit string is played. Default ``1``;
        a literal integer of at least 1.

    Raises
    ------
    ValueError
        If *bits* is empty or not made of 0/1 bits, *repeat* is below 1, or
        a literal electrical value is not finite, ``ui``, ``tr`` or ``tf``
        is not greater than zero, ``td`` is negative, or ``tr``/``tf``
        exceeds ``ui``.
    TypeError
        If *repeat* is not a literal integer (a ``bool`` is rejected).

    Notes
    -----
    The electrical values are validated at construction when they are
    literals and again whenever the points are produced; a
    :class:`~silk.circuit.Variable` left without a value is accepted
    at construction and checked when used.

    Examples
    --------
    >>> from silk.sources import BitPattern
    >>> bits = BitPattern('0110', v1=0, v2=1.8, ui='1n', tr='100p', tf='100p')
    >>> bits.stream
    (0, 1, 1, 0)
    >>> bits.initial_value()
    0
    """

    def __init__(self, bits, *, v1, v2, ui, tr, tf, td=0, repeat=1):
        from silk.sources._bits import _integer, _normalize_bits
        self.bits = _normalize_bits(bits)
        self.repeat = _integer('repeat', repeat, 1)
        self._set_electrical(v1, v2, ui, tr, tf, td)

    @property
    def stream(self):
        """The bit stream as a tuple of ``0``/``1`` integers.

        The normalized bits repeated ``repeat`` times, so its length is
        ``len(bits) * repeat``. A new tuple is built on each access.
        """
        from silk.sources._bits import _integer, _normalize_bits
        return tuple(map(int, _normalize_bits(self.bits))) * _integer('repeat', self.repeat, 1)

    def as_parameters(self):
        """A fresh declaration snapshot of this waveform.

        Returns
        -------
        dict
            ``kind='bitpattern'``, the normalized ``bits`` string, ``repeat``,
            and the authored ``v1``, ``v2``, ``ui``, ``tr``, ``tf`` and ``td``
            (volts or amps for the levels, seconds for the times). Variables
            are kept as variables, not resolved.

        Raises
        ------
        ValueError
            If ``bits`` or ``repeat`` has been set to an invalid value since
            construction.
        TypeError
            If ``repeat`` has been set to a non-integer since construction.
        """
        from silk.sources._bits import _integer, _normalize_bits
        return dict(kind='bitpattern', bits=_normalize_bits(self.bits),
                    repeat=_integer('repeat', self.repeat, 1), **self._electrical_parameters())


class PRBS(_FiniteBits):
    """Finite Fibonacci XNOR PRBS, sampled at Qn after each shift.

    ``length`` is required to bound memory and explicit-point deck size.
    Use :func:`silk.sources.prbs_bits` for the same reference stream alone.
    The waveform is emitted as an explicit ``PWL(...)`` point list, like
    :class:`silk.sources.BitPattern`; logical 0 is level ``v1`` and logical 1 is ``v2``.

    Parameters
    ----------
    order : int
        Register length; one of 7, 9, 15, 23 or 31.
    length : int
        Number of bits to generate, at least 1.
    seed : int, optional
        Initial register state, keyword-only, ``0`` to ``2**order - 2``.
        Default ``0``. The all-ones state is refused.
    v1 : float
        Level for logical 0, in volts or amps according to the source it
        feeds; keyword-only. A number, SPICE value string or
        :class:`~silk.circuit.Variable`; the same forms are accepted
        by ``v2``, ``ui``, ``tr``, ``tf`` and ``td``.
    v2 : float
        Level for logical 1; keyword-only.
    ui : float
        Unit interval (bit duration) in seconds; keyword-only, greater than
        zero.
    tr : float
        Rising-edge transition time in seconds; keyword-only, greater than
        zero and no greater than ``ui``.
    tf : float
        Falling-edge transition time in seconds; keyword-only, same limits
        as ``tr``.
    td : float, optional
        Delay in seconds before the first bit boundary. Default ``0``; must
        be nonnegative.

    Raises
    ------
    TypeError
        If *order*, *length* or *seed* is not a literal integer.
    ValueError
        If *order* is not a supported length, *length* is below 1, *seed* is
        negative or not below ``2**order - 1``, or a literal electrical value
        violates the limits above.

    Examples
    --------
    >>> from silk.sources import PRBS
    >>> prbs = PRBS(7, 8, v1=0, v2=1.8, ui='1n', tr='100p', tf='100p')
    >>> prbs.stream
    (0, 0, 0, 0, 0, 0, 1, 1)
    """

    def __init__(self, order, length, *, seed=0, v1, v2, ui, tr, tf, td=0):
        from silk.sources._bits import _prbs_parameters
        self.order, self.length, self.seed = _prbs_parameters(order, length, seed)
        self._set_electrical(v1, v2, ui, tr, tf, td)

    @property
    def stream(self):
        """The generated bits as a tuple of ``length`` ``0``/``1`` integers.

        Equal to :func:`silk.sources.prbs_bits` for this ``order``,
        ``length`` and ``seed``; regenerated on each access.
        """
        from silk.sources._bits import prbs_bits
        return prbs_bits(self.order, self.length, seed=self.seed)

    def as_parameters(self):
        """A compact declaration snapshot, without the expanded stream.

        Returns
        -------
        dict
            ``kind='prbs'``, ``convention='xnor-post-shift'``, ``order``,
            ``length``, ``seed`` and the authored ``v1``, ``v2``, ``ui``,
            ``tr``, ``tf`` and ``td`` (volts or amps for the levels, seconds
            for the times). Variables are kept as variables.

        Raises
        ------
        TypeError
            If ``order``, ``length`` or ``seed`` has been set to a
            non-integer since construction.
        ValueError
            If one of them has been set to an out-of-range value.
        """
        from silk.sources._bits import _prbs_parameters
        order, length, seed = _prbs_parameters(self.order, self.length, self.seed)
        return dict(kind='prbs', convention='xnor-post-shift', order=order,
                    length=length, seed=seed, **self._electrical_parameters())
