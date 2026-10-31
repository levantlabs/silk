# Simulation setup types (Transient, AC, DC, Noise, …).
# Signal classes live in silk.signals.

import math
import operator

from silk.utils._utils import _fmt, _parse_si
from silk.utils.deck_text import refuse_line_breaks


def _finite_number(owner, field, value):
    """Parse an analysis field without changing the value stored for emission."""
    label = f'{owner}.{field}'
    if isinstance(value, bool):
        raise ValueError(
            f'{label} must be a finite SPICE number; got {value!r}')
    # The raw string, before _parse_si strips it (TASK-593).
    refuse_line_breaks(value, who=label)
    try:
        parsed = _parse_si(value)
    except (ValueError, OverflowError):
        raise ValueError(
            f'{label} must be a finite SPICE number; got {value!r}') from None
    if not math.isfinite(parsed):
        raise ValueError(f'{label} must be finite; got {value!r}')
    return parsed


def _positive_number(owner, field, value):
    """Validate a finite analysis field that must be greater than zero."""
    parsed = _finite_number(owner, field, value)
    if parsed <= 0:
        raise ValueError(
            f'{owner}.{field} must be greater than zero; got {value!r}')
    return parsed


def _positive_integer(owner, field, value):
    """Normalize a positive whole-number count without a resource-policy cap."""
    label = f'{owner}.{field}'
    message = (
        f'{label} must resolve to a positive whole number; got {value!r}')
    # The raw string, before _parse_si/int normalise a break away (TASK-593).
    refuse_line_breaks(value, who=label)
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        parsed = operator.index(value)
    except TypeError:
        try:
            parsed_number = _parse_si(value)
        except (ValueError, OverflowError):
            raise ValueError(message) from None
        if not math.isfinite(parsed_number) or not parsed_number.is_integer():
            raise ValueError(message)
        parsed = int(parsed_number)
    if parsed <= 0:
        raise ValueError(message)
    return int(parsed)


def _validate_frequency_sweep(owner, variation, points, fstart, fstop):
    """Validate the shared AC/noise frequency-grid contract."""
    normalized_points = _positive_integer(owner, 'points', points)
    start = _positive_number(owner, 'fstart', fstart)
    stop = _positive_number(owner, 'fstop', fstop)
    if stop < start:
        raise ValueError(
            f'{owner}.fstop must be greater than or equal to '
            f'{owner}.fstart; got fstart={fstart!r}, fstop={fstop!r}')
    if variation == 'dec':
        from silk.simulation._frequency_grid import logarithmic_span, logarithmic_steps
        # Numeric emission has finite precision. A valid mathematical span
        # must not turn into a zero-step deck when formatted for SPICE.
        grids = [(start, stop, ''),
                 (_parse_si(_fmt(fstart)), _parse_si(_fmt(fstop)),
                  ' after SPICE formatting')]
        for grid_start, grid_stop, detail in grids:
            span = logarithmic_span(grid_start, grid_stop, 10)
            # A whole decade always holds a step, even at huge point counts.
            if span >= 1:
                continue
            sweep = f'{owner} dec {normalized_points} {fstart} {fstop}{detail}'
            if span == 0:
                raise ValueError(
                    f'{sweep} has zero span; use a wider frequency range or lin')
            minimum = math.ceil(1 / span)
            # Compare integers first so accepted arbitrary-size counts do
            # not overflow a float merely while checking the lower bound.
            if normalized_points > minimum or logarithmic_steps(
                    normalized_points, grid_start, grid_stop, 10) >= 1:
                continue
            if minimum > 1 and logarithmic_steps(
                    minimum - 1, grid_start, grid_stop, 10) >= 1:
                minimum -= 1
            raise ValueError(
                f'{sweep} is shorter than one whole step; use at least {minimum} '
                'points per decade, a wider frequency range, or lin')
    return normalized_points


class AnalysisBase:
    """Base class for the typed analysis commands.

    :class:`Transient`, :class:`AC`, :class:`DC`, :class:`Noise` and
    :class:`Op` derive from it. An analysis object only records its request;
    the simulator-specific command text is produced by the selected
    simulator backend. Pass instances to ``Module.add_sim_type()``; the
    legacy string form is rejected.

    Examples
    --------
    >>> from silk.simulation import AnalysisBase, Op
    >>> isinstance(Op(), AnalysisBase)
    True
    """
    def to_spice(self) -> str:
        """Render this analysis as SPICE command text.

        Returns
        -------
        str
            The command line(s), newline-terminated.

        Raises
        ------
        NotImplementedError
            On the base class itself; each concrete analysis overrides it.
        """
        raise NotImplementedError


class Transient(AnalysisBase):
    """Transient (time-domain) analysis command.

    Parameters
    ----------
    step : float or str
        Nominal time step (``TSTEP`` in SPICE), in seconds. Must be finite
        and greater than zero. A SPICE-suffixed string such as ``'1n'`` is
        accepted.
    stop : float or str
        End time (``TSTOP``), in seconds. Must be finite, greater than zero
        and greater than ``start``.
    start : float or str, optional
        Start time for output (``TSTART``), in seconds. Defaults to 0. Must
        be finite and non-negative.
    maxstep : float or str, optional
        Maximum allowed time step, in seconds, emitted as the 4th field of
        ``.tran``. Must be greater than zero when given. Useful for
        regenerative circuits (StrongArm latches, SR flip-flops) where the
        simulator can otherwise stride past a fast transition. When given,
        ``start`` is also written, because the field is positional.
    uic : bool, optional
        Use Initial Conditions: skip the DC operating-point analysis and
        start the transient from the ``.ic``/device-initial values. Needed
        for regenerative circuits (StrongArm latches, cross-coupled
        inverters) whose DC operating point is ambiguous or non-convergent.
        Emitted as the trailing ``uic`` keyword.

    Raises
    ------
    ValueError
        If ``step``, ``stop``, ``start`` or ``maxstep`` is not a finite
        SPICE number (booleans are refused), if ``step``, ``stop`` or
        ``maxstep`` is not greater than zero, if ``start`` is negative, or
        if ``stop`` is not greater than ``start``.
    silk.exceptions.DeckTextError
        A subclass of ``ValueError``: a numeric field given as a string holds
        a line break.

    Attributes
    ----------
    step, stop, start, maxstep, uic
        The constructor arguments, stored exactly as supplied (strings are
        not converted), so emission reproduces what was written.

    Examples
    --------
    >>> from silk import Transient
    >>> analysis = Transient(step=1e-9, stop=10e-9)
    >>> analysis.to_spice().strip()
    '.tran 1n 10n'
    """
    def __init__(self, step, stop, start=0, maxstep=None, uic=False):
        _positive_number('Transient', 'step', step)
        stop_value = _positive_number('Transient', 'stop', stop)
        start_value = _finite_number('Transient', 'start', start)
        if start_value < 0:
            raise ValueError(
                f'Transient.start must be non-negative; got {start!r}')
        if stop_value <= start_value:
            raise ValueError(
                f'Transient.stop must be greater than Transient.start; '
                f'got start={start!r}, stop={stop!r}')
        if maxstep is not None:
            _positive_number('Transient', 'maxstep', maxstep)
        self.step = step
        self.stop = stop
        self.start = start
        self.maxstep = maxstep
        self.uic = uic

    def to_spice(self):
        """Render this analysis as a ``.tran`` card using the ngspice renderer.

        Returns
        -------
        str
            ``.tran <step> <stop>`` followed by ``<start>`` (when ``start``
            is non-zero or ``maxstep`` is set), ``<maxstep>`` (when set) and
            ``uic`` (when set), newline-terminated. Numbers use silk's
            SPICE formatter.
        """
        from silk.simulation.backend import SimulatorBackend, EmitContext
        return SimulatorBackend.get('ngspice').render_analysis(self, EmitContext())


class AC(AnalysisBase):
    """AC small-signal frequency-sweep analysis command.

    Parameters
    ----------
    variation : {'dec', 'oct', 'lin'}
        Sweep spacing: logarithmic per decade, logarithmic per octave, or
        linear.
    points : int or str
        Points per decade or octave (``'dec'``/``'oct'``), or the total
        number of points (``'lin'``). Must resolve to a positive whole
        number; it is stored as an ``int``.
    fstart : float or str
        Start frequency, in hertz. Must be finite and greater than zero.
    fstop : float or str
        Stop frequency, in hertz. Must be finite, greater than zero and not
        less than ``fstart``.

    Raises
    ------
    ValueError
        If ``variation`` is not one of :attr:`VARIATIONS`, if ``points`` is
        not a positive whole number, if a frequency is not a finite number
        greater than zero, if ``fstop`` is below ``fstart``, or if a
        ``'dec'`` sweep is shorter than one whole step (including a zero
        span), checked on the values as given and after SPICE formatting.
    silk.exceptions.DeckTextError
        A subclass of ``ValueError``: a numeric field given as a string holds
        a line break.

    Attributes
    ----------
    VARIATIONS : tuple of str
        The accepted ``variation`` words, ``('dec', 'oct', 'lin')``.
    variation, fstart, fstop
        The constructor arguments, stored as supplied.
    points : int
        ``points`` after normalization to a whole number.

    Examples
    --------
    >>> from silk import AC
    >>> AC('dec', 10, 1, 1e6).to_spice().strip()
    '.ac dec 10 1 1Meg'
    """
    VARIATIONS = ('dec', 'oct', 'lin')

    def __init__(self, variation, points, fstart, fstop):
        if variation not in self.VARIATIONS:
            raise ValueError(f'variation must be one of {self.VARIATIONS}')
        normalized_points = _validate_frequency_sweep(
            'AC', variation, points, fstart, fstop)
        self.variation = variation
        self.points = normalized_points
        self.fstart = fstart
        self.fstop = fstop

    def to_spice(self):
        """Render this analysis as an ``.ac`` card using the ngspice renderer.

        Returns
        -------
        str
            ``.ac <variation> <points> <fstart> <fstop>``,
            newline-terminated.
        """
        from silk.simulation.backend import SimulatorBackend, EmitContext
        return SimulatorBackend.get('ngspice').render_analysis(self, EmitContext())


class DC(AnalysisBase):
    """DC sweep analysis command.

    Parameters
    ----------
    source : str
        Name of the independent source to sweep, or ``'temp'`` (any case) to
        sweep temperature. A source name is the logical silk name and is
        mapped to the emitted instance name when the deck is written; the
        ``'temp'`` sweep is emitted unchanged.
    start : float or str
        First sweep value (volts, amps or degrees Celsius, per ``source``).
        Must be finite.
    stop : float or str
        Last sweep value. Must be finite.
    step : float or str
        Sweep increment. Must be finite, non-zero, and point from ``start``
        toward ``stop`` (positive when ``stop > start``, negative when
        ``stop < start``).

    Raises
    ------
    ValueError
        If ``start``, ``stop`` or ``step`` is not a finite SPICE number, if
        ``step`` is zero, or if ``step`` has the wrong sign for the
        direction from ``start`` to ``stop``.
    silk.exceptions.DeckTextError
        A subclass of ``ValueError``: ``start``, ``stop`` or ``step`` given as
        a string holds a line break. ``source`` is not checked here; it is
        checked when the card is rendered.

    Attributes
    ----------
    source, start, stop, step
        The constructor arguments, stored as supplied.

    Examples
    --------
    >>> from silk import DC
    >>> DC('vin', 0, 1, 0.1).to_spice().strip()
    '.dc vin 0 1 0.1'
    """
    def __init__(self, source, start, stop, step):
        start_value = _finite_number('DC', 'start', start)
        stop_value = _finite_number('DC', 'stop', stop)
        step_value = _finite_number('DC', 'step', step)
        if step_value == 0:
            raise ValueError(f'DC.step must be non-zero; got {step!r}')
        wrong_direction = (
            (stop_value > start_value and step_value < 0)
            or (stop_value < start_value and step_value > 0)
        )
        if wrong_direction:
            raise ValueError(
                f'DC.step must move from start toward stop; '
                f'got start={start!r}, stop={stop!r}, step={step!r}')
        self.source = source
        self.start = start
        self.stop = stop
        self.step = step

    def to_spice(self):
        """Render this analysis as a ``.dc`` card using the ngspice renderer.

        With no circuit to resolve against, the ``source`` name is emitted
        as given.

        Returns
        -------
        str
            ``.dc <source> <start> <stop> <step>``, newline-terminated.
        """
        from silk.simulation.backend import SimulatorBackend, EmitContext
        return SimulatorBackend.get('ngspice').render_analysis(self, EmitContext())


class Noise(AnalysisBase):
    """Small-signal noise analysis command.

    Parameters
    ----------
    output : str
        Output node name.
    ref : str
        Reference node for the output voltage, usually ground.
    source : str
        Name of the input source to which input-referred noise is
        referred (the logical silk source name, mapped to the emitted
        instance name when the deck is written).
    variation : {'dec', 'oct', 'lin'}
        Sweep spacing, as for :class:`AC`.
    points : int or str
        Points per decade or octave, or the total for ``'lin'``. Must
        resolve to a positive whole number; stored as an ``int``.
    fstart : float or str
        Start frequency, in hertz. Must be finite and greater than zero.
    fstop : float or str
        Stop frequency, in hertz. Must be finite, greater than zero and not
        less than ``fstart``.

    Raises
    ------
    ValueError
        Under the same conditions as :class:`AC`: unknown ``variation``,
        ``points`` not a positive whole number, an invalid or reversed
        frequency range, or a ``'dec'`` sweep shorter than one whole step.
    silk.exceptions.DeckTextError
        A subclass of ``ValueError``: a numeric field given as a string holds
        a line break (also ``output``, ``ref`` or ``source``).

    Attributes
    ----------
    VARIATIONS : tuple of str
        The accepted ``variation`` words, ``('dec', 'oct', 'lin')``.
    output, ref, source, variation, fstart, fstop
        The constructor arguments, stored as supplied.
    points : int
        ``points`` after normalization to a whole number.

    Examples
    --------
    >>> from silk import Noise
    >>> Noise('out', 'gnd', 'vin', 'dec', 10, 1, 1e6).to_spice().strip()
    '.noise v(out,gnd) vin dec 10 1 1Meg 1'
    """
    VARIATIONS = ('dec', 'oct', 'lin')

    def __init__(self, output, ref, source, variation, points, fstart, fstop):
        if variation not in self.VARIATIONS:
            raise ValueError(f'variation must be one of {self.VARIATIONS}')
        normalized_points = _validate_frequency_sweep(
            'Noise', variation, points, fstart, fstop)
        for field, value in (('output', output), ('ref', ref),
                             ('source', source)):
            refuse_line_breaks(value, who=f'Noise.{field}')
        self.output = output
        self.ref = ref
        self.source = source
        self.variation = variation
        self.points = normalized_points
        self.fstart = fstart
        self.fstop = fstop

    def to_spice(self, resolve_source=None):
        """Render this analysis as a ``.noise`` card using the ngspice renderer.

        Parameters
        ----------
        resolve_source : callable, optional
            Maps the logical ``source`` name to the emitted instance name.
            When omitted the name is used unchanged.

        Returns
        -------
        str
            ``.noise v(<output>,<ref>) <source> <variation> <points>
            <fstart> <fstop> 1``, newline-terminated; the trailing ``1`` is
            the summary interval.
        """
        from silk.simulation.backend import SimulatorBackend, EmitContext
        ctx = EmitContext(resolve_source=resolve_source) if resolve_source \
            else EmitContext()
        return SimulatorBackend.get('ngspice').render_analysis(self, ctx)


class Op(AnalysisBase):
    """Operating-point analysis command.

    Takes no parameters: a bare request for the circuit's DC operating
    point, the same one every other analysis implicitly solves first.
    ngspice emits it as a literal ``.op`` card, whose rawfile plot is named
    ``Operating Point``; silk classifies it as the ``'op'`` analysis kind,
    selectable through ``analysis='op'`` like any other.

    Xyce has no ``.print op`` type, so its backend instead adds a private
    zero-volt source on a reserved node and sweeps it over one point with
    ``.dc``; user sources are left untouched.

    Examples
    --------
    >>> from silk.simulation import Op
    >>> Op().to_spice().strip()
    '.op'
    """
    def to_spice(self):
        """Render this analysis as an ``.op`` card using the ngspice renderer.

        Returns
        -------
        str
            The card ``.op``, newline-terminated.
        """
        from silk.simulation.backend import SimulatorBackend, EmitContext
        return SimulatorBackend.get('ngspice').render_analysis(self, EmitContext())


