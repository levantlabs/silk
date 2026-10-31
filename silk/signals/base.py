"""Signal — simulator-agnostic base class for waveform signals."""

import copy as _copy
import functools
import numbers
import operator
import weakref

import numpy as np

from silk.exceptions import BusPairingError, UnboundSignalError

# dBm reference level: 1 mW, in watts (AUDIT-02 — named so the conversion is inspectable).
_DBM_REF_W = 1e-3

#: Declared physical type for each loader selector. ``load_signal``
#: takes the selector ``'V'``/``'I'``; the *declared type* is what the
#: arithmetic compatibility check reads. Both are kept on an unbound Signal:
#: the selector is what the loader wants, the type is what the checks read.
_SELECTOR_TYPE = {'V': 'voltage', 'I': 'current'}

#: The four attributes an unbound Signal resolves on demand. They
#: are exactly the ones a bound Signal stores, so this fallback never fires on
#: a Signal that carries data.
_RESOLVED_ATTRS = ('x', 'y', 'xlabel', 'ylabel')

#: Sentinel meaning "this transform's deferred node carries the RECEIVER's
#: own declared type" — the default for a method not listed in
#: ``_TRANSFORM_FIXED_TYPE`` below. Needed because ``None`` is itself a
#: real, distinct declared type (fft/dbm) and cannot double as "absent".
_RECEIVER_TYPE = object()

#: Declared type each transform's deferred (unbound) node carries
#: for the methods whose BOUND
#: implementation fixes a result type independent of the receiver's own
#: type. Verified against each method's ``_new()``/return call below:
#: ``phase_rad``/``phase_deg``/``phase_rad_unwrapped``/
#: ``phase_deg_unwrapped`` always build type ``'phase'``; ``db10``/``db20``
#: always build ``'dB'``; ``as_int`` always builds ``'count'`` (NOT the
#: receiver's type, unlike the other bus/scalar transforms). ``fft`` and
#: ``dbm`` map to ``None`` even
#: though their bound implementations currently fix a type of their own
#: (``'power'``): the declared type is only a hint for the
#: declaration-time compatibility check, and ``None`` simply defers that
#: hint to resolution rather than asserting one. A method absent from this
#: table (``mag``, ``real``, ``imag``, ``shift``, ``window``, ``flip``,
#: ``interpolate``, ``derivative``, ``integral``) keeps the receiver's own
#: declared type, matching their bound ``_new()`` calls, which all pass
#: ``self.type`` straight through.
_TRANSFORM_FIXED_TYPE = {
    'phase_rad':           'phase',
    'phase_deg':           'phase',
    'phase_rad_unwrapped': 'phase',
    'phase_deg_unwrapped': 'phase',
    'db10':                'dB',
    'db20':                'dB',
    'fft':                 None,
    'dbm':                 None,
    'as_int':               'count',
}


def _deferrable(func):
    """Decorator for every Signal-to-Signal transform method.

    Applied to ``mag``, ``real``, ``imag``, the four ``phase_*`` methods,
    ``db10``, ``db20``, ``shift``, ``window``, ``flip``, ``interpolate``,
    ``derivative``, ``integral``, ``fft``, ``dbm`` and ``as_int`` — every
    method that returns a Signal derived from ``self`` (design section 4).

    If the receiver (``self``) is unbound, returns a NEW unbound Signal
    recording ``(method_name, self, args, kwargs)`` instead of running the
    transform now — the receiver **expression** is retained (``slew =
    vcm.derivative()`` keeps ``vcm``, not vcm's current value), so
    :meth:`Signal.resolve` can resolve it fresh each time. Resolution
    resolves the receiver first, then calls the method by name on the
    result, so the actual math and argument validation (``window``'s
    ``x2 <= x1``, ``interpolate``'s complex check) stay defined exactly
    once, in the wrapped implementation below — this decorator only ever
    adds the deferred branch, never changes the bound one.

    If the receiver already carries data, ``func`` runs immediately and
    this decorator is a no-op.
    """
    method_name = func.__name__
    fixed_type = _TRANSFORM_FIXED_TYPE.get(method_name, _RECEIVER_TYPE)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        from silk.circuit.placement_reference import normalize_reference
        args = tuple(normalize_reference(arg) for arg in args)
        kwargs = {key: normalize_reference(arg) for key, arg in kwargs.items()}
        if method_name != 'as_int' and self.__dict__.get('_underlying_signals') is not None:
            return self._bus_transform(method_name, args, kwargs)
        if self._is_unbound():
            declared_type = (
                self.type if fixed_type is _RECEIVER_TYPE else fixed_type
            )
            return self._transform_recipe_signal(
                method_name, declared_type, args, kwargs,
            )
        return func(self, *args, **kwargs)

    return wrapper


class _DetachedBenchRef:
    """Stands in for a probe's weak bench reference across a pickle.

    A ``weakref.ref`` cannot be pickled, and a bench whose metrics are named
    functions pickles today (``FaultCampaign`` pickles whole Modules,
    testbenches included) — so a probe must not be the thing that stops it.
    :meth:`Signal.__getstate__` therefore drops the weak reference and leaves
    this marker in its place, and the unpickled probe is **detached**: it
    reads like a dead reference (calling it returns ``None``, so ``bound`` is
    ``False``), but it is distinguishable from one, so the error message can
    say *why* there is no bench (design section 3.0).

    Re-attaching a restored bench's probes to itself is
    ``ModuleTestbench.__setstate__``'s job, the same walk its
    ``__deepcopy__`` uses; it simply overwrites this marker.
    """

    __slots__ = ()

    def __call__(self):
        return None

    def __repr__(self):
        return '<detached bench reference>'

    def __reduce__(self):
        # Pickle by reference: one marker, always identical on the way back.
        return (_detached_bench_ref, ())

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def _detached_bench_ref():
    """Return the single detached-bench marker (pickle hook)."""
    return _DETACHED_BENCH


#: The one marker instance. ``is`` comparisons against it are what tell a
#: detached probe from one that never had a bench or whose bench has died.
_DETACHED_BENCH = _DetachedBenchRef()


class _DetachedSelectedBenchRef(_DetachedBenchRef):
    """A selected probe cannot be promoted to an authored bench on restore."""
    __slots__ = ()

    def __reduce__(self):
        return (_detached_selected_bench_ref, ())


def _detached_selected_bench_ref():
    return _DETACHED_SELECTED_BENCH


_DETACHED_SELECTED_BENCH = _DetachedSelectedBenchRef()


def _is_real_scalar(x):
    """True for a real Python/numpy scalar (int, float, numpy real dtype).

    Excludes complex scalars and :class:`Signal` — those go through the
    Signal-with-Signal or TypeError paths instead.
    """
    return isinstance(x, numbers.Real)


def _mirror_x(x):
    """Mirror an x-grid in place for a time reversal: keep the same span.

    ``x_new = x[0] + x[-1] - x[::-1]``. On a non-uniform grid (e.g.
    SPICE's adaptive transient steps) this correctly relocates each
    sample to its mirrored position instead of leaving the grid in
    place, which silently misattributes samples on a reversed trace. On a uniform grid the mirror is mathematically the
    identity, but evaluating the formula in floating point is NOT
    bit-identical to ``x`` (subtraction in reversed order rounds
    differently per sample — measured up to 1 ULP on ~90% of samples
    for realistic grids), so a uniform grid is detected explicitly and
    returned unchanged rather than recomputed. Shared by
    :meth:`Signal.flip` and :func:`silk.metrics.waveform.flip` so the
    arithmetic lives in exactly one place.
    """
    x = np.asarray(x, dtype=float)
    if len(x) >= 2:
        dx = np.diff(x)
        if np.allclose(dx, dx[0], rtol=1e-9, atol=0.0):
            return x.copy()
    return x[0] + x[-1] - x[::-1]


class Signal:
    """Base class for simulation waveform signals.

    The simulator-agnostic return type for loaded waveforms. Subclasses load
    data from a simulator's native format via ``from_rawfile()``; the active
    :class:`~silk.simulation.backend.SimulatorBackend` calls that mechanism
    from its ``load_signal`` verb (there is no per-simulator Signal registry —
    backend selection owns dispatch).

    This class provides the shared interface: arithmetic operators, ``plot()``,
    and attribute access (``x``, ``y``, ``name``, ``xlabel``, ``ylabel``).

    A Signal returned by :meth:`probe <silk.circuit.ModuleTestbench.probe>`
    before its testbench has run is **unbound**: it holds a name and a weak
    reference to the bench but no samples, and every arithmetic operator and
    every Signal-to-Signal transform (``+``, ``-``, ``*``, ``/``, ``**``,
    ``mag``, ``derivative``, ``window``, ``fft``, ...) called on one *defers*
    instead of computing — ``vcm = (vp + vm) / 2`` and ``slew =
    vcm.derivative()`` build a small recipe (an operator or method name plus
    its operands) rather than any data, so a metric or a plot can be declared
    on a derived signal before a single simulation has run; reading ``x``/
    ``y``/``xlabel``/``ylabel``, or calling ``resolve()`` directly, replays
    that recipe against the bench's current results on access, storing
    nothing back, and raises :class:`~silk.exceptions.UnboundSignalError`
    (never a bare ``AttributeError``) if the bench has not run yet. A Signal
    with data (everything ``load_signal`` returns, and everything above once
    resolved) is unaffected by any of this — ``bound`` is always ``True`` and
    every attribute is a plain stored value.

    Pointwise ``+``, ``-``, ``*`` and ``/`` between Signals require exactly
    equal coordinate shapes and values, including when recipes are evaluated.
    Equal copied axes and equal values of different dtypes work. Mismatches
    raise ``ValueError`` without interpolation or snapping. Use matching
    recorded coordinates or explicitly align with
    ``silk.metrics.sample(other, x_new=reference.x)`` when appropriate.

    Parameters
    ----------
    name : str
        Signal name, e.g. ``'v(out)'``.
    type : str
        Physical type, e.g. ``'voltage'``, ``'current'``.
    xlabel : str
        Label for the x-axis (``'time'``, ``'frequency'``, …).
    ylabel : str
        Label for the y-axis.
    xvalues : array-like
        Independent-axis values.
    yvalues : array-like
        Dependent-axis values (same length as *xvalues*).

    Attributes
    ----------
    x : numpy.ndarray
        Independent-axis values (``xvalues`` as an array): seconds for a
        transient, hertz for AC and noise, the swept value for DC. On an
        unbound signal, reading it resolves the signal against the bench
        (see above).
    y : numpy.ndarray
        Dependent-axis values (``yvalues`` as an array); complex for AC node
        and branch results, real for device-parameter probes. Its unit is
        that of the quantity read, which ``ylabel`` only names as a display
        label (it is not reliable unit metadata). Same resolve-on-read rule
        as ``x``.
    name : str
        Signal name, e.g. ``'v(out)'``; derived signals carry the
        expression, e.g. ``'mag(v(out))'``.
    type : str
        Physical type: ``'voltage'``, ``'current'``, ``'phase'``,
        ``'dB'``, ``'power'``, ``'count'``, and so on. Operators between
        two signals require equal types.
    xlabel : str
        Independent-axis label (``'time'``, ``'frequency'``). Operators
        between two signals require equal ``xlabel``.
    ylabel : str
        Dependent-axis display label (``'V'``, ``'dB'``, ``'rad'``); a
        label, not a guaranteed unit.

    Examples
    --------
    >>> from silk import Signal
    >>> sig = Signal('v(out)', 'voltage', 'time', 'V', [0, 1], [1, 2])
    >>> (sig * 2).y.tolist()
    [2, 4]
    """

    def __init__(self, name, type, xlabel, ylabel, xvalues, yvalues):  # noqa: A002
        self.name   = name
        self.type   = type
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.x      = np.asarray(xvalues)
        self.y      = np.asarray(yvalues)
        # Bus-handle support (spec §3.3). Scalar Signals get default values;
        # bus handles are constructed via _make_bus_handle.
        self.width: int = 1
        self._underlying_signals = None  # tuple[Signal, ...] | None

    # ------------------------------------------------------------------
    # Unbound Signals — a name before there is data (TASK-215)
    #
    # A probe is an ordinary Signal in an UNBOUND state, and that state is
    # represented by ABSENCE: an unbound Signal simply has no ``x``, ``y``,
    # ``xlabel`` or ``ylabel`` stored on it — no flag, no parallel class.
    # Reading one of those four goes through :meth:`__getattr__`, which
    # resolves the signal against its testbench's *current* results and
    # stores nothing back (no cache can therefore ever be stale).
    #
    # Signals that carry data are untouched by all of this: __init__ above,
    # :meth:`_new` and :meth:`_make_bus_handle` keep assigning x/y directly,
    # so no descriptor hop lands on the metrics library.
    # ------------------------------------------------------------------

    @classmethod
    def _unbound(cls, name, selector='V', bench=None, *, analysis=None):
        """Construct an unbound Signal — a name, not data. **Internal.**

        The public door is ``ModuleTestbench.probe()``, which
        validates the name and supplies the bench; this classmethod is the
        constructor it calls.

        Parameters
        ----------
        name : str
            The rawfile signal name, e.g. ``'vout'``, ``'xdut.mid'``,
            ``'vvs'`` — exactly what ``Module.load_signal`` takes.
        selector : {'V', 'I'}, optional
            The loader selector. Kept as its own field because a rawfile
            holds ``v(x)`` and ``i(x)`` under one name space; the declared
            :attr:`type` (``'voltage'``/``'current'``) is mapped from it.
        bench : ModuleTestbench, optional
            The testbench this probe reads from. Held by **weak** reference,
            so a bench's metric list drags no bench graph into a copy. The reference is to the bench, not to its netlist
            Module: the netlist object is replaced whenever the test is
            rebuilt, so a netlist reference would go stale.
        analysis : {'tran', 'ac', 'dc', 'noise'}, optional
            Which analysis's plot this probe reads, for a test declaring
            more than one. ``None`` means "settled elsewhere" —
            by the metric's slot kind when there is only one producer among
            the declared analyses, and otherwise refused rather than
            guessed.

        Returns
        -------
        Signal
            An unbound Signal: ``name``, ``type``, ``_selector``,
            ``_analysis``, ``_bench``, ``_recipe`` (``None`` for a leaf) and
            the two bus fields. ``width = 1`` and
            ``_underlying_signals = None`` so ``len()``, iteration and
            indexing fail with the scalar-Signal message they give today
            rather than reading a missing attribute.
        """
        sel = str(selector).upper()
        if sel not in _SELECTOR_TYPE:
            raise ValueError(
                f"signal selector must be 'V' (voltage) or 'I' (current), "
                f"got {selector!r}"
            )
        obj = object.__new__(cls)
        obj.name = name
        obj.type = _SELECTOR_TYPE[sel]
        obj._selector = sel
        obj._analysis = analysis
        obj._bench = None if bench is None else weakref.ref(bench)
        obj._recipe = None          # a leaf; operators/transforms record one
        obj.width = 1
        obj._underlying_signals = None
        return obj

    def __getattr__(self, name):
        """Resolve ``x``/``y``/``xlabel``/``ylabel`` on an unbound Signal.

        Only reached when normal lookup fails, so a Signal that carries data
        never gets here for those four names.

        Names starting with ``'_'`` raise a plain :class:`AttributeError`
        with no bench lookup: that is what keeps :mod:`copy` and
        :mod:`pickle` from recursing when they probe for ``__deepcopy__``,
        ``__getstate__``, ``__reduce_ex__`` and friends (the same guard
        ``SubcircuitInstance.__getattr__`` uses). Any other missing
        attribute raises the message Python itself would have raised.
        """
        if name.startswith('_'):
            raise AttributeError(name)
        if name in _RESOLVED_ATTRS:
            children = self.__dict__.get('_underlying_signals')
            if children is not None:
                if name == 'y':
                    return None
                return getattr(children[0], name)
            resolved = self.resolve()
            if resolved is self:
                # Defensive: a Signal that claims to be bound yet lacks one
                # of the four. Never happens through the constructors; this
                # keeps it an AttributeError instead of infinite recursion.
                raise AttributeError(
                    f'{type(self).__name__!r} object has no attribute {name!r}'
                )
            return getattr(resolved, name)
        raise AttributeError(
            f'{type(self).__name__!r} object has no attribute {name!r}'
        )

    # -- copying and pickling ------------------------------------------
    #
    # Copying KEEPS the bench reference (a weak reference is atomic, so the
    # copy points at the original bench until the copied *bench* rebinds it —
    # TASK-215.8); pickling DROPS it, because a weak reference cannot be
    # pickled and a bench with named-function metrics pickles today. The two
    # therefore cannot share one hook: ``copy`` would otherwise go through
    # ``__getstate__`` and silently detach every copied probe.

    def __deepcopy__(self, memo):
        """Deep-copy, sharing the weak bench reference (design section 3.0).

        Equivalent to the default protocol for a Signal that carries data
        (every ``__dict__`` value is deep-copied); the one special case is
        ``_bench``, where a weak reference is shared rather than copied, which
        is what :mod:`copy` does with one anyway.
        """
        new = object.__new__(type(self))
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            if key == '_bench':
                from silk.circuit.test_placement import SelectedTestbench, _SelectedBenchRef
                target = value() if value is not None else None
                if isinstance(target, SelectedTestbench) and id(target._root()) in memo:
                    new.__dict__[key] = _SelectedBenchRef(_copy.deepcopy(target, memo))
                else:
                    new.__dict__[key] = value
            else:
                new.__dict__[key] = _copy.deepcopy(value, memo)
        return new

    def __copy__(self):
        """Shallow-copy, sharing the weak bench reference."""
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        return new

    def __getstate__(self):
        """Pickle state: everything, minus the weak reference to the bench.

        An unbound Signal's state is its name, the selector, the declared
        type, the analysis and the recipe. The bench reference is replaced by
        the detached marker, so the restored probe explains itself instead of
        failing to pickle. Selected probes use a distinct detached marker:
        only their owning Module graph may restore that selection, never the
        authored bench alone. A Signal that carries data pickles as before.
        """
        state = dict(self.__dict__)
        if self._is_unbound() and state.get('_bench') is not None:
            from silk.circuit.test_placement import SelectedTestbench, _SelectedBenchRef
            ref = state['_bench']
            target = ref()
            state['_bench'] = (_DETACHED_SELECTED_BENCH
                               if ref is _DETACHED_SELECTED_BENCH or isinstance(ref, _SelectedBenchRef)
                               or isinstance(target, SelectedTestbench)
                               else _DETACHED_BENCH)
        return state

    def __setstate__(self, state):
        """Restore from :meth:`__getstate__` — a plain state update."""
        self.__dict__.update(state)

    @property
    def bound(self):
        """``True`` when input results are available for reading.

        Never raises. A Signal that carries data is always bound. A derived
        Signal is bound when all its input Signals are bound; a transform may
        still refuse its arguments on read. An unbound
        one asks its bench: no bench reference, a bench detached by pickling,
        a bench that has been garbage-collected, a bench with no netlist (an
        orphan replica) and a bench that has not run all answer ``False``.
        """
        children = self.__dict__.get('_underlying_signals')
        if children is not None:
            return all(child.bound for child in children)
        if not self._is_unbound():
            return True
        if self.__dict__.get('_recipe') is not None:
            return all(child.bound for child in self._signal_children())
        bench = self._bench_or_none()
        if bench is None:
            return False
        result_reader = getattr(bench, '_result_netlist', None)
        netlist = result_reader() if callable(result_reader) else getattr(bench, 'netlist', None)
        if netlist is None:
            return False
        if not getattr(netlist, 'sim_results_loaded', False):
            return False
        # A bench whose latest sweep retained several corners answers False
        # too (TASK-218): reading .x/.y right now would raise rather than
        # return data, which is exactly what this property reports. Use
        # .corner(name) to pick one, and that Signal is bound.
        return len(getattr(bench, '_sweep_table', None) or {}) <= 1

    def resolve(self, module=None, *, analysis=None, corner=None,
                _readers=None):
        """Return a Signal carrying data, loading it if this one is unbound.

        Identity on a Signal that already carries data. On an unbound leaf,
        loads ``name`` through ``module.load_signal(name, selector)`` —
        *module* defaults to the bench's current netlist — and returns what
        the loader returns. **Nothing is stored back**: the next read
        resolves again, so a probe can never show a stale run.

        This is the explicit form, for scripts that hold a Module; reading
        ``.x``/``.y`` calls it for you.

        Parameters
        ----------
        module : Module, optional
            Where to load from. Defaults to each leaf's own bench's current
            netlist, so an expression may span benches.
        analysis : {'tran', 'ac', 'dc', 'noise', 'op'}, optional
            Which plot to read from a rawfile holding more than one. A **default**, not an override: a leaf created as
            ``tb.probe(name, analysis='ac')`` keeps its own word, and a leaf
            that names none reads this one. Passed down through a recipe, so
            one call settles a whole expression. ``None`` — the usual case —
            reaches ``load_signal`` as the argument list it has always had,
            and a single-plot rawfile needs nothing here.
        corner : str, optional
            Read a retained corner's rawfile instead of each bench's current
            results. Every leaf is resolved against its OWN
            bench's entry for that corner name, so an expression spanning
            benches works the way it does without this argument. Use
            :meth:`corner`, which is this argument with a name.

        _readers : optional
            Internal cache shared across the leaves of one resolution; leave
            it unset.

        Returns
        -------
        Signal
            ``self`` when it already carries data; otherwise a new Signal
            holding the loaded (or recomputed) samples.

        Raises
        ------
        UnboundSignalError
            When the bench has not run, has no netlist, was never attached,
            or has been garbage-collected; when *corner* names a corner the
            bench did not retain; or when the bench retained MORE than one
            corner and no *corner* was named, since a sweep's results cannot
            be guessed between (the message lists them and points at
            :meth:`corner` and :meth:`plot`). Deliberately not an
            ``AttributeError`` — "no data yet" must propagate, not read as
            "this object has no such attribute".
        """
        if not self._is_unbound():
            return self
        children = self.__dict__.get('_underlying_signals')
        if children is not None:
            return type(self)._make_bus_handle(
                self.name, tuple(child.resolve(module, analysis=analysis, corner=corner,
                                              _readers=_readers) for child in children),
                indices=self._indices)
        recipe = self.__dict__.get('_recipe')
        if recipe is not None:
            head = recipe[0]
            if isinstance(head, str):
                # (method_name, receiver, args, kwargs) — the transform
                # decorator's recipe shape (TASK-215.4). The receiver
                # expression resolves first (recursively — a chain like
                # ``vcm.derivative()`` resolves ``vcm`` all the way down to
                # its own leaves first); any Signal-valued arg/kwarg
                # resolves the same way, a non-Signal one passes through.
                # Calling the method by NAME on the resolved receiver
                # re-invokes the (now-bound) decorated method, so the real
                # transform and its argument validation (window's
                # ``x2 <= x1``, interpolate's complex check) are defined
                # exactly once, in the wrapped implementation.
                _, receiver, args, kwargs = recipe
                resolved_receiver = (
                    receiver.resolve(module, analysis=analysis,
                                     corner=corner, _readers=_readers)
                    if isinstance(receiver, Signal) else receiver
                )
                resolved_args = tuple(
                    a.resolve(module, analysis=analysis,
                              corner=corner, _readers=_readers)
                    if isinstance(a, Signal) else a
                    for a in args
                )
                resolved_kwargs = {
                    k: (v.resolve(module, analysis=analysis,
                                  corner=corner, _readers=_readers)
                        if isinstance(v, Signal) else v)
                    for k, v in kwargs.items()
                }
                return getattr(resolved_receiver, head)(
                    *resolved_args, **resolved_kwargs
                )
            # (op, *operands) — TASK-215.3's operator branch. Each Signal
            # operand resolves itself first (recursively, so a chain of
            # unbound expressions resolves leaf by leaf, each against its
            # own bench); a scalar operand passes through unchanged.
            # Calling ``op`` on the resolved operands re-invokes the bound
            # dunder that already does the real computation and its
            # data-level checks (xlabel, xvalues, exact type), so that
            # logic is defined exactly once.
            operands = recipe[1:]
            resolved = [
                operand.resolve(module, analysis=analysis,
                                corner=corner, _readers=_readers)
                if isinstance(operand, Signal) else operand
                for operand in operands
            ]
            return head(*resolved)
        netlist = (module if module is not None
                   else self._bench_netlist(corner=corner, readers=_readers))
        # This leaf's own word wins; *analysis* is the fallback the caller
        # settled for the whole expression (TASK-212). With neither, the call
        # is the one a single-analysis run has always made.
        plot = self.__dict__.get('_analysis') or analysis
        if plot is None:
            return netlist.load_signal(self.name, self._selector)
        return netlist.load_signal(self.name, self._selector, analysis=plot)

    def snapshot(self, module=None, *, analysis=None, corner=None):
        """Freeze this signal's current data into an ordinary Signal.

        A probe (``tb.probe('vout')``) is a live reference: it reads the
        bench's latest run each time. A snapshot is a plain Signal holding
        the samples as they are *now*, so it keeps them when the bench
        re-runs and after the bench is gone. Use it to (1) keep an earlier
        run while re-running the same bench, e.g. in a loop over a
        parameter, or (2) return results from a function whose bench is
        local. Reading ``.x``/``.y``, registering metrics and plotting
        never need it.

        Exactly :meth:`resolve` under a name that says what you get; the
        arguments and errors are the same.

        Parameters
        ----------
        module : Module, optional
            Where to load from; see :meth:`resolve`.
        analysis : {'tran', 'ac', 'dc', 'noise', 'op'}, optional
            Default plot to read from a multi-plot rawfile; see :meth:`resolve`.
        corner : str, optional
            Read this retained corner instead of the current results.

        Returns
        -------
        Signal
            A Signal carrying data; independent of later runs of the bench.

        Raises
        ------
        UnboundSignalError
            As :meth:`resolve`.
        """
        return self.resolve(module, analysis=analysis, corner=corner)

    def corner(self, name, *, analysis=None):
        """Return this Signal as measured in one retained corner.

        ``tb.run_corners(['tt', 'ff', 'ss'])`` runs three simulations, and the
        reports it hands back reference ONE live netlist — the last corner's.
        The earlier corners survive as files, and the bench remembers where
        (``tb.sweep_corners``). This method reads one of them::

            res = tb.run_corners(['tt', 'ff', 'ss'])
            vcm.corner('ff').y            # ff's samples
            vcm.corner('ff').plot()       # one corner, drawn on its own
            vcm.plot()                    # all three, overlaid

        Every leaf of a derived expression is resolved against its own
        bench's file for that corner, so ``((vp + vm) / 2).corner('ff')`` is
        the common mode as ff measured it, not a mixture. The result is an
        ordinary bound Signal carrying data — ``.y``, ``.plot()``, transforms
        and metrics all work on it as usual.

        The live netlist is never disturbed: the corner's file is read into a
        separate result object, so the bench's current results (and anything
        already derived from them) are exactly what they were.

        Parameters
        ----------
        name : str
            A corner of the latest sweep. Anything else raises, naming what
            was retained — including after a sweep in overwrite output mode,
            which keeps only the last corner's file (see
            :attr:`sweep_corners <silk.circuit.ModuleTestbench.sweep_corners>`).
        analysis : {'tran', 'ac', 'dc', 'noise'}, optional
            Which plot to read, for a corner file holding more than one; the same default-not-override rule
            :meth:`resolve` documents.

        Returns
        -------
        Signal
            A bound Signal carrying that corner's samples.

        Raises
        ------
        UnboundSignalError
            No sweep retained, an unknown corner name, or no bench to ask.
        ValueError
            Called on a Signal that already carries data — it came from one
            run and there is nothing to select.
        """
        if not self._is_unbound():
            raise ValueError(
                f"Signal {self.name!r} already carries data, so there is no "
                f"corner to select: .corner() picks one run out of a "
                f"testbench's latest sweep, and works on a probe "
                f"(tb.probe(...)) or an expression built from one."
            )
        return self.resolve(analysis=analysis, corner=str(name), _readers={})

    def point(self, key, *, analysis=None):
        """Read one deterministic grid point (corner, Celsius, supply volts).

        Deferred expression leaves use the same point and shared readers.
        Monte Carlo requires sample((point, index)); bound signals refuse.
        """
        if not self._is_unbound():
            raise ValueError('Signal already carries data; .point() selects a deferred probe.')
        bench = self._first_leaf()._bench_or_none()
        selector = getattr(bench, '_point_corner', None)
        if selector is None:
            raise UnboundSignalError('No live testbench with retained points; keep the bench alive.')
        return self.resolve(analysis=analysis, corner=selector(key), _readers={})

    def sample(self, key, *, analysis=None):
        """Return this Signal as measured in one Monte Carlo sample.

        After a Monte Carlo run (``tb.set_monte_carlo(...)`` then
        ``tb.run()``, or ``tb.run_monte_carlo(...)``) every sample's files
        stay in ``<testbench dir>/<corner>/mc_NNNN/``; this reads one of them
        the way :meth:`corner` reads one corner of a sweep::

            mc = tb.run_monte_carlo(n=200)
            vout.sample(('tt', 17)).y

        After such a run ``vout.corner('tt')`` refuses: the samples share
        that corner.

        Parameters
        ----------
        key : tuple
            ``(corner, sample index)``, a key of the run's
            :class:`~silk.circuit.monte_carlo.MonteCarloResults`.
        analysis : {'tran', 'ac', 'dc', 'noise'}, optional
            As for :meth:`corner`.

        Raises
        ------
        UnboundSignalError
            The bench's latest run was not Monte Carlo, *key* is not one of
            its retained samples, or there is no bench to ask.
        TypeError
            *key* is not a ``(corner, index)`` pair.
        ValueError
            Called on a Signal that already carries data.
        """
        if not self._is_unbound():
            raise ValueError(
                f"Signal {self.name!r} already carries data, so there is no "
                f"sample to select: .sample() picks one run out of a "
                f"testbench's latest Monte Carlo run, and works on a probe "
                f"(tb.probe(...)) or an expression built from one."
            )
        bench = self._first_leaf()._bench_or_none()
        selector = getattr(bench, '_sample_corner', None)
        if selector is None:
            raise UnboundSignalError(
                f"probe {self.name!r}: no live testbench with Monte Carlo "
                f"samples to read from; keep a reference to the testbench "
                f"for as long as you read its probes."
            )
        return self.resolve(analysis=analysis, corner=selector(key),
                            _readers={})

    # -- unbound-state helpers (underscore-named: reached by normal lookup,
    # so they never re-enter __getattr__) ------------------------------

    def _is_unbound(self):
        """True when this Signal carries no samples (the absence rule)."""
        children = self.__dict__.get('_underlying_signals')
        if children is not None:
            return any(child._is_unbound() for child in children)
        return 'y' not in self.__dict__

    def _bench_or_none(self):
        """The testbench, or ``None`` if unattached or garbage-collected."""
        ref = self.__dict__.get('_bench')
        return None if ref is None else ref()

    def _first_leaf(self):
        """The first deferred leaf, or the leftmost leaf when all are bound.

        A derived Signal has no bench of its own — its leaves do. Corner
        selection needs ONE sweep table to validate a name against and to
        enumerate, and the first deferred leaf's bench is the one that names the
        expression (``(vp + vm) / 2`` is vp's bench). Resolution itself still
        goes leaf by leaf, each against its own bench, so an expression
        spanning benches reads each of them for the named corner. A missing or
        expired owner is not skipped in favor of a later live bench.
        """
        for leaf in self._walk_leaves():
            if leaf._is_unbound():
                return leaf
        return next(self._walk_leaves())

    def _corners_to_draw(self, corners):
        """Which corners :meth:`plot` should overlay, or ``None`` for none.

        ``None`` means "draw the current results", which is what plotting did
        previously and what a bench that ran once still does.
        """
        if not self._is_unbound():
            if corners is None:
                return None
            raise ValueError(
                f"Signal {self.name!r} already carries data, so corners= has "
                f"nothing to select from; it applies to a probe "
                f"(tb.probe(...)) or an expression built from one."
            )
        bench = self._first_leaf()._bench_or_none()
        if bench is None:
            if corners is None:
                return None
            raise UnboundSignalError(
                f"probe {self.name!r}: no live testbench to take corners "
                f"from; keep a reference to the testbench for as long as you "
                f"read its probes."
            )
        # Duck-typed the same way every other bench read here is: an object
        # standing in for a testbench that keeps no sweep simply has no
        # corners, and asking for none is answered without it.
        selector = getattr(bench, '_corners_to_draw', None)
        if selector is None:
            if corners is None:
                return None
            raise UnboundSignalError(
                f"probe {self.name!r}: its testbench keeps no corner sweep, "
                f"so corners={corners!r} cannot be selected."
            )
        return selector(corners)

    def _bench_netlist(self, corner=None, readers=None):
        """What to load from: the bench's netlist, or one corner's rawfile.

        Every message names the probe, and the bench whenever there is one
        left to name.

        With *corner* given, returns the bench's read-only stand-in for that
        corner's file (``ModuleTestbench._corner_netlist``) - the live
        netlist's results are neither read nor touched, and whether it
        currently holds results does not matter. *readers* is the per-call
        reader dict shared by the leaves of one expression.

        With no *corner*, returns the live netlist as it always has, with one
        addition: a bench whose latest sweep retained MORE than
        one corner has no single "current results" to mean, so this refuses
        and names them instead of silently handing back the last corner.
        """
        ref = self.__dict__.get('_bench')
        if isinstance(ref, _DetachedBenchRef):
            raise UnboundSignalError(
                f"probe {self.name!r} is detached from its bench (unpickled); "
                f"create it again with tb.probe"
            )
        if ref is None:
            raise UnboundSignalError(
                f"probe {self.name!r}: no testbench is attached, so there is "
                f"nothing to resolve against; create it with tb.probe(), or "
                f"pass a module to resolve(module)"
            )
        bench = ref()
        if bench is None:
            raise UnboundSignalError(
                f"probe {self.name!r}: its testbench has been "
                f"garbage-collected, so the probe can no longer be resolved; "
                f"keep a reference to the testbench for as long as you read "
                f"its probes"
            )
        bench_name = getattr(bench, 'test_name', None) or '<unnamed>'
        result_reader = getattr(bench, '_result_netlist', None)
        netlist = result_reader() if callable(result_reader) else getattr(bench, 'netlist', None)
        if netlist is None:
            raise UnboundSignalError(
                f"probe {self.name!r}: bench {bench_name!r} has no netlist, so "
                f"it has no results; build the test "
                f"(add_module_under_test / add_sim_type) and run it first"
            )
        if corner is not None:
            return bench._corner_netlist(corner, readers)
        if not getattr(netlist, 'sim_results_loaded', False):
            raise UnboundSignalError(
                f"probe {self.name!r}: bench {bench_name!r} has not run yet; "
                f"call tb.run() before reading .x/.y/.xlabel/.ylabel"
            )
        retained = getattr(bench, '_sweep_table', None) or {}
        if len(retained) > 1 and any(isinstance(key, tuple) for key in retained):
            selector = 'sample' if getattr(bench, '_mc_samples', None) else 'point'
            raise UnboundSignalError(
                f'probe {self.name!r} has several grid runs; use .{selector}'
                f'({next(iter(retained))!r}) to select one.')
        if len(retained) > 1:
            names = ', '.join(repr(c) for c in retained)
            raise UnboundSignalError(
                f"probe {self.name!r}: bench {bench_name!r} ran "
                f"{len(retained)} corners: {names}. A probe reads ONE run's "
                f"results and this sweep retained several, so .x/.y would "
                f"have to guess which. Name one - "
                f"probe.corner({next(iter(retained))!r}).y - or draw them "
                f"together with probe.plot() / tb.plot(probe)."
            )
        return netlist

    # ------------------------------------------------------------------
    # Arithmetic operators
    #
    # Each gets one extra branch (TASK-215.3, design section 4): if either
    # operand is an unbound Signal, build a recipe instead of computing —
    # ``_recipe_signal`` below records ``(op, *operands)`` with ``op`` a
    # module-level ``operator.*`` builtin (never a lambda or bound method,
    # so the metric list still deep-copies and, for named-function metrics,
    # still pickles — section 3.0). The declared ``type`` on the recipe
    # follows the same formula the bound branch already uses just below it
    # (scalar keeps type; Signal-Signal multiply/divide composes 'a*b'/
    # 'a/b'; power gives 'type^k'), so a chain of deferred operators keeps
    # exactly the type a resolved chain would have. The bound-bound and
    # bound-scalar paths are untouched.
    # ------------------------------------------------------------------

    def _is_bus(self):
        """True for a bus Signal (an aggregate of bits), scalar or not."""
        return self.__dict__.get('_underlying_signals') is not None

    def _bus_binary(self, other, op, symbol, reflected=False):
        """Element-wise arithmetic where one or both operands are buses.

        A bus behaves like a numpy array: two buses of equal length combine
        by position, a number (or a non-bus Signal) applies to every bit, and
        a length mismatch raises :class:`~silk.exceptions.BusPairingError`.
        A one-bus result keeps that bus's declared bit numbers. A two-bus
        result is numbered 0, 1, 2, ... by position (``e = d + c`` gives
        ``e[0] = d[8] + c[0]`` for ``d`` declared from 8). Each bit
        goes through the ordinary scalar operator, so unbound probes stay
        deferred and bound ones compute now.
        """
        if self._is_bus() and isinstance(other, Signal) and other._is_bus():
            if len(self._underlying_signals) != len(other._underlying_signals):
                raise BusPairingError(
                    f'Cannot combine bus {self.name!r} (width {len(self._underlying_signals)}) '
                    f'with bus {other.name!r} (width {len(other._underlying_signals)}) '
                    f'for {symbol!r}: buses combine by position and need equal length.')
            bits = tuple(op(a, b) for a, b in
                         zip(self._underlying_signals, other._underlying_signals))
            name = f'{self.name}{symbol}{other.name}'
            return type(self)._make_bus_handle(name, bits)
        if not (isinstance(other, Signal) or _is_real_scalar(other)):
            return NotImplemented
        if self._is_bus():
            bus, name = self, (f'{other}{symbol}{self.name}' if reflected
                               else f'{self.name}{symbol}{getattr(other, "name", other)}')
            pair = (lambda b: op(other, b)) if reflected else (lambda b: op(b, other))
        else:
            # scalar Signal on the left, bus on the right
            bus, name = other, f'{self.name}{symbol}{other.name}'
            pair = lambda b: op(self, b)  # noqa: E731
        bits = tuple(pair(b) for b in bus._underlying_signals)
        return type(self)._make_bus_handle(name, bits, indices=bus._indices)

    def _bus_transform(self, method_name, args, kwargs, name=None):
        """Apply a unary Signal method to each bit, keeping declared bit numbers."""
        for arg in (*args, *kwargs.values()):
            if isinstance(arg, Signal) and arg._is_bus():
                raise TypeError(
                    f'{method_name}() on bus {self.name!r} takes no bus-valued argument; '
                    f'its arguments are forwarded unchanged to every bit, so pass '
                    f'what the scalar method accepts.')
        bits = tuple(getattr(b, method_name)(*args, **kwargs)
                     for b in self._underlying_signals)
        if name is None:
            parts = [repr(a) for a in args] + [f'{k}={v!r}' for k, v in kwargs.items()]
            name = f'{method_name}({", ".join([self.name, *parts])})'
        return type(self)._make_bus_handle(name, bits, indices=self._indices)

    def _recipe_signal(self, name, sig_type, op, *operands):
        """Build an unbound Signal recording a deferred arithmetic recipe.

        ``operands`` are exactly what ``resolve()`` will resolve (each
        Signal operand recursively, each scalar operand as-is) and hand to
        ``op``. No data is read or stored now — the recipe is the only
        state kept.
        """
        obj = object.__new__(type(self))
        obj.name = name
        obj.type = sig_type
        obj._selector = None
        obj._analysis = None
        obj._bench = None
        obj._recipe = (op,) + operands
        obj.width = 1
        obj._underlying_signals = None
        return obj

    def _check_declared_types(self, other):
        """Declaration-time refusal for ``+``/``-`` (design section 4).

        A declared type on an unbound Signal is only ever ``None`` (a
        data-dependent transform result, e.g. ``fft``/``dbm``)
        or the exact string the bound result would carry once resolved — there is no separate vocabulary to recognise. So
        this refuses, with the exact text :meth:`_check_types` raises for
        two bound Signals, whenever BOTH declared types are non-``None``
        and differ; a ``None`` on either side defers, since the full
        exact-type check still runs when the expression resolves, via the
        bound ``__add__``/``__sub__`` this recipe replays.
        """
        if (
            self.type is not None
            and other.type is not None
            and self.type != other.type
        ):
            raise ValueError(
                f'Signals have different types: '
                f'{self.type!r} vs {other.type!r}'
            )

    def __add__(self, other):
        from silk.circuit.placement_reference import normalize_reference
        other = normalize_reference(other)
        if self._is_bus() or (isinstance(other, Signal) and other._is_bus()):
            return self._bus_binary(other, operator.add, '+', reflected=False)
        if isinstance(other, Signal):
            if self._is_unbound() or other._is_unbound():
                self._check_declared_types(other)
                return self._recipe_signal(
                    self.name + '+' + other.name,
                    self.type,
                    operator.add,
                    self, other,
                )
            self._check_compatible(other)
            return self._new(
                self.name + '+' + other.name,
                self.type,
                self.x,
                self.y + other.y,
            )
        if _is_real_scalar(other):
            if self._is_unbound():
                return self._recipe_signal(
                    f'{self.name}+{other}',
                    self.type,
                    operator.add,
                    self, other,
                )
            return self._new(
                f'{self.name}+{other}',
                self.type,
                self.x,
                self.y + other,
            )
        return NotImplemented

    def __radd__(self, other):
        from silk.circuit.placement_reference import normalize_reference
        other = normalize_reference(other)
        if self._is_bus() or (isinstance(other, Signal) and other._is_bus()):
            return self._bus_binary(other, operator.add, '+', reflected=True)
        if _is_real_scalar(other):
            if self._is_unbound():
                return self._recipe_signal(
                    f'{other}+{self.name}',
                    self.type,
                    operator.add,
                    other, self,
                )
            return self._new(
                f'{other}+{self.name}',
                self.type,
                self.x,
                other + self.y,
            )
        return NotImplemented

    def __sub__(self, other):
        from silk.circuit.placement_reference import normalize_reference
        other = normalize_reference(other)
        if self._is_bus() or (isinstance(other, Signal) and other._is_bus()):
            return self._bus_binary(other, operator.sub, '-', reflected=False)
        if isinstance(other, Signal):
            if self._is_unbound() or other._is_unbound():
                self._check_declared_types(other)
                return self._recipe_signal(
                    self.name + '-' + other.name,
                    self.type,
                    operator.sub,
                    self, other,
                )
            self._check_compatible(other)
            return self._new(
                self.name + '-' + other.name,
                self.type,
                self.x,
                self.y - other.y,
            )
        if _is_real_scalar(other):
            if self._is_unbound():
                return self._recipe_signal(
                    f'{self.name}-{other}',
                    self.type,
                    operator.sub,
                    self, other,
                )
            return self._new(
                f'{self.name}-{other}',
                self.type,
                self.x,
                self.y - other,
            )
        return NotImplemented

    def __rsub__(self, other):
        from silk.circuit.placement_reference import normalize_reference
        other = normalize_reference(other)
        if self._is_bus() or (isinstance(other, Signal) and other._is_bus()):
            return self._bus_binary(other, operator.sub, '-', reflected=True)
        if _is_real_scalar(other):
            if self._is_unbound():
                return self._recipe_signal(
                    f'{other}-{self.name}',
                    self.type,
                    operator.sub,
                    other, self,
                )
            return self._new(
                f'{other}-{self.name}',
                self.type,
                self.x,
                other - self.y,
            )
        return NotImplemented

    def __mul__(self, other):
        from silk.circuit.placement_reference import normalize_reference
        other = normalize_reference(other)
        if self._is_bus() or (isinstance(other, Signal) and other._is_bus()):
            return self._bus_binary(other, operator.mul, '*', reflected=False)
        if isinstance(other, Signal):
            new_type = (self.type + '*' + other.type
                        if self.type is not None and other.type is not None
                        else None)
            if self._is_unbound() or other._is_unbound():
                return self._recipe_signal(
                    self.name + '*' + other.name,
                    new_type,
                    operator.mul,
                    self, other,
                )
            self._check_xlabels(other)
            self._check_xvalues(other)
            return self._new(
                self.name + '*' + other.name,
                new_type,
                self.x,
                self.y * other.y,
            )
        if _is_real_scalar(other):
            if self._is_unbound():
                return self._recipe_signal(
                    f'{self.name}*{other}',
                    self.type,
                    operator.mul,
                    self, other,
                )
            return self._new(
                f'{self.name}*{other}',
                self.type,
                self.x,
                self.y * other,
            )
        return NotImplemented

    def __rmul__(self, other):
        from silk.circuit.placement_reference import normalize_reference
        other = normalize_reference(other)
        if self._is_bus() or (isinstance(other, Signal) and other._is_bus()):
            return self._bus_binary(other, operator.mul, '*', reflected=True)
        if _is_real_scalar(other):
            if self._is_unbound():
                return self._recipe_signal(
                    f'{other}*{self.name}',
                    self.type,
                    operator.mul,
                    other, self,
                )
            return self._new(
                f'{other}*{self.name}',
                self.type,
                self.x,
                other * self.y,
            )
        return NotImplemented

    def __truediv__(self, other):
        from silk.circuit.placement_reference import normalize_reference
        other = normalize_reference(other)
        if self._is_bus() or (isinstance(other, Signal) and other._is_bus()):
            return self._bus_binary(other, operator.truediv, '/', reflected=False)
        if isinstance(other, Signal):
            new_type = (self.type + '/' + other.type
                        if self.type is not None and other.type is not None
                        else None)
            if self._is_unbound() or other._is_unbound():
                return self._recipe_signal(
                    self.name + '/' + other.name,
                    new_type,
                    operator.truediv,
                    self, other,
                )
            self._check_xlabels(other)
            self._check_xvalues(other)
            return self._new(
                self.name + '/' + other.name,
                new_type,
                self.x,
                self.y / other.y,
            )
        if _is_real_scalar(other):
            if self._is_unbound():
                return self._recipe_signal(
                    f'{self.name}/{other}',
                    self.type,
                    operator.truediv,
                    self, other,
                )
            return self._new(
                f'{self.name}/{other}',
                self.type,
                self.x,
                self.y / other,
            )
        return NotImplemented

    def __rtruediv__(self, other):
        from silk.circuit.placement_reference import normalize_reference
        other = normalize_reference(other)
        if self._is_bus() or (isinstance(other, Signal) and other._is_bus()):
            return self._bus_binary(other, operator.truediv, '/', reflected=True)
        if _is_real_scalar(other):
            if self._is_unbound():
                return self._recipe_signal(
                    f'{other}/{self.name}',
                    self.type,
                    operator.truediv,
                    other, self,
                )
            return self._new(
                f'{other}/{self.name}',
                self.type,
                self.x,
                other / self.y,
            )
        return NotImplemented

    def __neg__(self):
        if self._is_bus():
            return self._bus_transform('__neg__', (), {}, name=f'-{self.name}')
        if self._is_unbound():
            return self._recipe_signal(
                f'-{self.name}',
                self.type,
                operator.neg,
                self,
            )
        return self._new(
            f'-{self.name}',
            self.type,
            self.x,
            -self.y,
        )

    def __pow__(self, other):
        from silk.circuit.placement_reference import normalize_reference
        other = normalize_reference(other)
        if isinstance(other, Signal):
            raise TypeError(
                f'Signal ** Signal is not supported (got exponent of type '
                f'{type(other).__name__}); the exponent must be a real number'
            )
        if not _is_real_scalar(other):
            raise TypeError(
                f'Signal ** {type(other).__name__} is not supported; '
                f'the exponent must be a real number'
            )
        if self._is_bus():
            return self._bus_binary(other, operator.pow, '**')
        if self._is_unbound():
            return self._recipe_signal(
                f'({self.name})**{other}',
                f'{self.type}^{other}' if self.type is not None else None,
                operator.pow,
                self, other,
            )
        return self._new(
            f'({self.name})**{other}',
            f'{self.type}^{other}' if self.type is not None else None,
            self.x,
            self.y ** other,
        )

    # ------------------------------------------------------------------
    # Transform decorator support (TASK-215.4, design section 4)
    #
    # ``_deferrable`` (module level, above) wraps every Signal-to-Signal
    # transform method; this is the recipe constructor it calls when the
    # receiver is unbound. Same field shape as ``_recipe_signal`` above, so
    # the existing copy/pickle hooks need no special-casing for either
    # recipe shape.
    # ------------------------------------------------------------------

    def _transform_recipe_signal(self, method_name, sig_type, args, kwargs):
        """Build an unbound Signal recording a deferred transform recipe.

        ``self`` — the receiver **expression**, not its current value — is
        retained in the recipe, so ``resolve()`` resolves it fresh each
        time (``slew = vcm.derivative()`` keeps ``vcm``, not a snapshot of
        it). ``method_name`` is the string name of the bound method that
        :meth:`resolve` will invoke on the resolved receiver; ``args`` and
        ``kwargs`` are recorded as given (each resolved if it is itself a
        Signal, at resolution time) — no lambda, no bound method, so
        deepcopy and pickle stay exactly as correct as the operator
        recipe's.
        """
        parts = [repr(a) for a in args] + [f'{k}={v!r}' for k, v in kwargs.items()]
        display_name = (
            f'{method_name}({self.name})' if not parts
            else f'{method_name}({self.name}, {", ".join(parts)})'
        )
        obj = object.__new__(type(self))
        obj.name = display_name
        obj.type = sig_type
        obj._selector = None
        obj._analysis = None
        obj._bench = None
        obj._recipe = (method_name, self, args, kwargs)
        obj.width = 1
        obj._underlying_signals = None
        return obj

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, ax=None, *, mode="auto", logx=None, unwrap=False,
             corners=None, **kwargs):
        """Plot the signal on *ax* (creates one if not provided).

        Real-valued signals (transient/DC) are drawn as x vs y. Complex-valued
        signals (AC sweeps) default to the natural Bode reading — magnitude in
        dB on a log-frequency axis — instead of silently dropping the imaginary
        part. The axis treatment is inferred from the signal metadata and can
        be overridden with the keywords.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to draw on. Created via ``plt.gca()`` if omitted.
        mode : {'auto', 'real', 'mag', 'db', 'phase'}, optional
            What to plot. ``'auto'`` (default) picks ``'real'`` for real
            signals and ``'db'`` for complex signals. ``'mag'`` is linear
            magnitude, ``'db'`` is ``20*log10(|y|)``, ``'phase'`` is the angle
            in degrees.
        logx : bool, optional
            Force a log or linear x-axis. Default: log when the x-axis looks
            like a frequency sweep — i.e. ``xlabel`` contains ``'freq'`` (as in
            ``'frequency'``) — linear otherwise. Pass ``logx`` explicitly for a
            frequency axis labelled differently (e.g. ``'Hz'``).
        unwrap : bool, optional
            Unwrap the phase before plotting (``mode='phase'`` only).
        corners : list of str, optional
            Which retained corners to overlay. Omitted is the
            useful default: after ``tb.run_corners([...])`` in an output mode
            that kept a file per run, ``vcm.plot()`` draws EVERY retained
            corner as its own labelled trace with a legend, and after a plain
            ``tb.run()`` it draws that run, exactly as before. Name a subset
            to draw only those; an unknown name raises, listing the retained
            ones. Default overlays also preserve full retained grid point
            and Monte Carlo sample keys. Each trace is resolved separately,
            leaving the bench's live results untouched. The retained key
            supplies the label, overriding a ``label=`` of your own.
        **kwargs
            Forwarded to the underlying matplotlib plot call.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        if self._is_unbound():
            from silk.circuit.test_placement import placement_plot_traces
            bench = self._first_leaf()._bench_or_none()
            traces = placement_plot_traces(bench, [self], corners) if bench is not None else None
            if traces is not None:
                for label, signal in traces[0]:
                    signal.plot(ax=ax, mode=mode, logx=logx, unwrap=unwrap,
                                **dict(kwargs, label=label))
                ax.legend()
                return ax

        # Corner overlay (TASK-218): one labelled trace per retained corner,
        # each drawn by recursing on the BOUND Signal that corner resolves
        # to, so the mode/logx/unwrap handling below stays the only copy.
        names = self._corners_to_draw(corners)
        if names is not None:
            for corner_name in names:
                self.resolve(corner=corner_name, _readers={}).plot(
                    ax=ax, mode=mode, logx=logx, unwrap=unwrap,
                    **dict(kwargs, label=str(corner_name)))
            ax.legend()
            return ax

        # Resolve ONCE (TASK-215.4): this method reads y (twice), x, xlabel
        # and ylabel below; on an unbound receiver each of those would
        # otherwise hit __getattr__ -> resolve() separately. ``resolve()``
        # is identity when self already carries data, so this costs nothing
        # on the already-bound path.
        sig = self.resolve()

        if sig.y is None:
            raise TypeError(
                f"Signal {sig.name!r} is a bus handle (no scalar y); index a "
                f"bit (sig[i]) or decode (sig.as_int()) before plotting."
            )

        y = np.asarray(sig.y)
        is_complex = np.iscomplexobj(y)

        resolved = "db" if (mode == "auto" and is_complex) else \
                   "real" if mode == "auto" else mode

        if resolved == "real":
            ydata = y.real if is_complex else y
            ylabel = sig.ylabel
        elif resolved == "mag":
            ydata = np.abs(y)
            ylabel = f"|{sig.ylabel}|" if sig.ylabel else "magnitude"
        elif resolved == "db":
            # Floor the magnitude so an exact-zero bin can't raise a
            # divide-by-zero warning (keeps test output pristine).
            mag = np.maximum(np.abs(y), np.finfo(float).tiny)
            ydata = 20.0 * np.log10(mag)
            ylabel = "Magnitude (dB)"
        elif resolved == "phase":
            ydata = (np.unwrap(np.angle(y)) * 180.0 / np.pi
                     if unwrap else np.angle(y, deg=True))
            ylabel = "Phase (deg)"
        else:
            raise ValueError(
                f"unknown plot mode {mode!r}; expected one of "
                "'auto', 'real', 'mag', 'db', 'phase'"
            )

        if logx is None:
            logx = "freq" in (sig.xlabel or "").lower()

        (ax.semilogx if logx else ax.plot)(sig.x, ydata, **kwargs)
        ax.set_xlabel(sig.xlabel)
        ax.set_ylabel(ylabel)
        return ax

    # ------------------------------------------------------------------
    # Bus-handle support
    # ------------------------------------------------------------------

    @classmethod
    def _make_bus_handle(cls, name: str, underlying: tuple, *, indices=None) -> 'Signal':
        """Aggregate Signals without acquiring samples; retain declared numbers."""
        if not underlying:
            raise ValueError("bus signal requires at least one underlying bit")
        handle = cls.__new__(cls)
        handle.name = name
        handle.type = underlying[0].type
        handle.width = len(underlying)
        handle._underlying_signals = tuple(underlying)
        handle._indices = tuple(range(len(underlying))) if indices is None else tuple(indices)
        if len(handle._indices) != handle.width or len(set(handle._indices)) != handle.width:
            raise ValueError('Bus indices must uniquely identify every bit.')
        return handle

    def __len__(self):
        """Return scalar width or aggregate width."""
        return self.width

    def __iter__(self):
        """Iterate bus bits in stored order, including width-one aggregates."""
        if self._underlying_signals is None:
            raise TypeError(f"Signal {self.name!r} is scalar (width=1); iteration is not supported.")
        return iter(self._underlying_signals)

    def __getitem__(self, key):
        """Select bus bits by their declared numbers, retaining numbers in slices."""
        if self._underlying_signals is None:
            raise TypeError(f"Signal {self.name!r} is scalar; subscript not supported.")
        from silk.circuit.declared_indices import select_positions
        positions = select_positions(self._indices, key)
        from numbers import Integral
        if isinstance(key, Integral):
            return self._underlying_signals[positions[0]]
        return type(self)._make_bus_handle(
            self.name, tuple(self._underlying_signals[p] for p in positions),
            indices=tuple(self._indices[p] for p in positions))

    def _signal_children(self):
        """One graph edge enumeration for aggregates and deferred recipes."""
        children = self.__dict__.get('_underlying_signals')
        if children is not None:
            return children
        recipe = self.__dict__.get('_recipe')
        if recipe is None:
            return ()
        if isinstance(recipe[0], str):
            _, receiver, args, kwargs = recipe
            return tuple(x for x in (receiver, *args, *kwargs.values()) if isinstance(x, Signal))
        return tuple(x for x in recipe[1:] if isinstance(x, Signal))

    def _walk_leaves(self, *, logical_probes=False):
        """Visit physical probe leaves; optionally retain logical probe nodes."""
        if logical_probes and self.__dict__.get('_logical_probe'):
            yield self
            return
        children = self._signal_children()
        if children:
            for child in children:
                yield from child._walk_leaves(logical_probes=logical_probes)
        else:
            yield self

    @_deferrable
    def as_int(self, *, threshold: float = 0.5, msb_first: bool = False) -> 'Signal':
        """Decode this bus Signal into an integer-valued scalar Signal.

        Each timepoint's bit-vector is treated as ``y > threshold`` and packed
        into an integer. With ``msb_first=False`` (default) the bus's first
        iterated bit is bit 0 of the integer; with ``msb_first=True`` the first
        iterated bit is the highest-order bit.

        A NaN in any contributing bit makes that sample's code NaN rather
        than logical zero. A contaminated decode emits one structured
        ``MetricWarning`` with the affected sample count.

        Packing uses exact integers before a single conversion to float64.
        Float64 has 53 significant binary bits: larger codes can lose low-order
        integer bits, depending on their value. Exactly representable values
        (including large powers of two and wide zero words) do not warn.
        If any available codes round, one ``binary_code_precision_loss``
        ``MetricWarning`` names ``as_int`` and reports the bus width and rounded
        sample count. Keep the bit Signals to compare whole words exactly.

        Parameters
        ----------
        threshold : float, default 0.5
            A bit is 1 where its signal's ``y`` is greater than this, in the
            bits' own units (volts for a logic bus).
        msb_first : bool, default False
            Take the first bit as the most significant instead of the least.

        Returns
        -------
        Signal
            Type ``'count'``, ylabel ``'code'``, on this signal's ``x``;
            ``y`` holds the integer codes as floats. Deferred if this signal
            is unbound.

        Raises
        ------
        TypeError
            If this Signal is scalar rather than a bus.
        OverflowError
            If an available exact code cannot convert to a finite float64.
            The message names ``as_int``, the zero-based sample index and x
            coordinate. Very wide buses with small codes remain supported.
        """
        if self._underlying_signals is None:
            raise TypeError(
                f"as_int() requires a bus Signal; {self.name!r} is scalar"
            )
        samples = [s.y for s in self._underlying_signals]
        unavailable = np.logical_or.reduce([np.isnan(y) for y in samples])
        bit_arrays = [(y > threshold).astype(int) for y in samples]
        if msb_first:
            bit_arrays = list(reversed(bit_arrays))
        if len(bit_arrays) <= 53:
            code = np.zeros_like(bit_arrays[0])
            for i, bits in enumerate(bit_arrays):
                code = code + (bits << i)
            code = code.astype(float)
        else:
            code = np.full(bit_arrays[0].shape, np.nan, dtype=float)
            rounded_count = 0
            first_rounded = None
            for sample in np.flatnonzero(~unavailable):
                exact = 0
                for bits in reversed(bit_arrays):
                    exact = (exact << 1) | int(bits[sample])
                try:
                    value = float(exact)
                except OverflowError:
                    raise OverflowError(
                        f'as_int: the code at sample {sample} (x={self.x[sample]}) '
                        f'of a {len(bit_arrays)}-bit bus is larger than the largest '
                        'float (about 1.8e308), so it cannot be returned. Keep the '
                        'original bit Signals to compare whole words exactly.'
                    ) from None
                code[sample] = value
                if int(value) != exact:
                    rounded_count += 1
                    if first_rounded is None:
                        first_rounded = (exact, int(value))
            if rounded_count:
                from silk.metrics.diagnostics import emit
                sample_count = int((~unavailable).sum())
                emit(
                    metric='as_int', code='binary_code_precision_loss',
                    message=(
                        f'{rounded_count} of {sample_count} decoded samples were rounded. '
                        f'A {len(bit_arrays)}-bit code needs more than a float\'s '
                        '53 significant bits, so its lowest bits were lost '
                        f'({first_rounded[0]} became {first_rounded[1]}).'
                    ),
                    details={'bit_width': len(bit_arrays),
                             'sample_count': sample_count,
                             'rounded_sample_count': rounded_count},
                    suggestion='Keep the bit Signals to compare whole words exactly.',
                )
        code[unavailable] = np.nan
        if np.any(unavailable):
            from silk.metrics.diagnostics import emit
            emit(metric='as_int', code='nan_bits',
                 message='NaN contributing bits make decoded samples unavailable.',
                 details={'affected_sample_count': int(np.count_nonzero(unavailable))})
        return type(self)(
            name=f'{self.name}.as_int',
            type='count',
            xlabel=self.xlabel,
            ylabel='code',
            xvalues=self.x,
            yvalues=code,
        )

    # ------------------------------------------------------------------
    # Magnitude / phase / dB / real / imag — utilities
    # (mag, phase_deg, phase_rad, phase_deg_unwrapped,
    # phase_rad_unwrapped, db10, db20, dbm, real, imag).
    # ------------------------------------------------------------------

    @_deferrable
    def mag(self):
        """Magnitude waveform: ``|y|``.

        Returns
        -------
        Signal
            ``|y|`` on the same ``x``, same type and ylabel; named
            ``mag(<name>)``. Deferred if this signal is unbound.
        """
        return self._new(
            f'mag({self.name})', self.type, self.x, np.abs(self.y),
        )

    @_deferrable
    def real(self):
        """Real part of y.

        Returns
        -------
        Signal
            ``Re(y)`` on the same ``x``; named ``real(<name>)``. Deferred if
            this signal is unbound.
        """
        return self._new(
            f'real({self.name})', self.type, self.x, np.real(self.y),
        )

    @_deferrable
    def imag(self):
        """Imaginary part of y.

        Returns
        -------
        Signal
            ``Im(y)`` (zeros for real data) on the same ``x``; named
            ``imag(<name>)``. Deferred if this signal is unbound.
        """
        return self._new(
            f'imag({self.name})', self.type, self.x, np.imag(self.y),
        )

    @_deferrable
    def phase_rad(self):
        """Phase in radians, as numpy's principal angle.

        Returns
        -------
        Signal
            Type ``'phase'``, ylabel ``'rad'``, in ``[-π, π]`` by
            ``numpy.angle`` (a negative-zero imaginary part on the negative
            real axis yields ``-π``); named
            ``phase(<name>)``. Deferred if this signal is unbound.
        """
        return self._new(
            f'phase({self.name})', 'phase', self.x,
            np.angle(self.y),
            ylabel='rad',
        )

    @_deferrable
    def phase_deg(self):
        """Phase in degrees, as numpy's principal angle.

        Returns
        -------
        Signal
            Type ``'phase'``, ylabel ``'deg'``, in ``[-180, 180]`` by
            ``numpy.angle`` (a negative-zero imaginary part on the negative
            real axis yields ``-180``); named
            ``phase(<name>)``. Deferred if this signal is unbound.
        """
        return self._new(
            f'phase({self.name})', 'phase', self.x,
            np.degrees(np.angle(self.y)),
            ylabel='deg',
        )

    @_deferrable
    def phase_rad_unwrapped(self):
        """Unwrapped phase in radians.

        Returns
        -------
        Signal
            Type ``'phase'``, ylabel ``'rad'``, unwrapped along ``x``; named
            ``phase(<name>)``. Deferred if this signal is unbound.
        """
        return self._new(
            f'phase({self.name})', 'phase', self.x,
            np.unwrap(np.angle(self.y)),
            ylabel='rad',
        )

    @_deferrable
    def phase_deg_unwrapped(self):
        """Unwrapped phase in degrees.

        Returns
        -------
        Signal
            Type ``'phase'``, ylabel ``'deg'``, unwrapped along ``x``; named
            ``phase(<name>)``. Deferred if this signal is unbound.
        """
        return self._new(
            f'phase({self.name})', 'phase', self.x,
            np.degrees(np.unwrap(np.angle(self.y))),
            ylabel='deg',
        )

    @_deferrable
    def db10(self):
        """10·log10(y). Use for power signals.

        Returns
        -------
        Signal
            Type ``'dB'``, ylabel ``'dB'``; named ``db10(<name>)``. Where
            ``y <= 0`` and ``y`` is real the value is ``-inf`` (zero) or NaN
            (negative), with numpy's divide/invalid warning; complex ``y``
            takes numpy's complex logarithm instead. Deferred if this signal is
            unbound.
        """
        return self._new(
            f'db10({self.name})', 'dB', self.x,
            10.0 * np.log10(self.y),
            ylabel='dB',
        )

    @_deferrable
    def db20(self):
        """20·log10(``|y|``). Use for amplitude signals (auto-takes magnitude
        so complex AC signals work).

        Returns
        -------
        Signal
            Type ``'dB'``, ylabel ``'dB'``; named ``db20(<name>)``. Where
            ``y == 0`` the value is ``-inf``, with numpy's divide warning.
            Deferred if this signal is unbound.
        """
        return self._new(
            f'db20({self.name})', 'dB', self.x,
            20.0 * np.log10(np.abs(self.y)),
            ylabel='dB',
        )

    # ------------------------------------------------------------------
    # Waveform manipulation primitives — method mirrors of selected
    # silk.metrics free functions, for chainable use. Implemented inline
    # (no silk.metrics import) except :meth:`fft` and :meth:`derivative`,
    # which use a local import to break the silk.signals ↔ silk.metrics
    # circular path (derivative delegates so it inherits TASK-106's
    # complex-dtype-preserving gradient rather than duplicating it).
    # ------------------------------------------------------------------

    @_deferrable
    def shift(self, dx: float):
        """Time-shift left by ``dx``: ``x_new = x − dx``.

        Parameters
        ----------
        dx : float
            Shift, in the units of ``x`` (seconds for a transient).

        Returns
        -------
        Signal
            Same ``y``, ``x`` shifted; named ``shift(<name>,<dx>)``.
            Deferred if this signal is unbound.
        """
        return self._new(
            f'shift({self.name},{dx})', self.type,
            np.asarray(self.x, dtype=float) - float(dx),
            self.y,
        )

    @_deferrable
    def window(self, x1: float, x2: float):
        """Restrict to the x-window ``[x1, x2]``.

        Parameters
        ----------
        x1, x2 : float
            Window bounds in the units of ``x`` (inclusive); ``x2 > x1``.

        Returns
        -------
        Signal
            Only the samples with ``x1 <= x <= x2`` (possibly none); named
            ``window(<name>,<x1>,<x2>)``. Deferred if this signal is
            unbound.

        Raises
        ------
        ValueError
            If ``x2 <= x1``.
        """
        if x2 <= x1:
            raise ValueError(f'require x2 > x1, got x1={x1}, x2={x2}')
        x = np.asarray(self.x, dtype=float)
        mask = (x >= x1) & (x <= x2)
        return self._new(
            f'window({self.name},{x1},{x2})', self.type,
            x[mask],
            np.asarray(self.y)[mask],
        )

    @_deferrable
    def flip(self):
        """Time-reverse a signal: mirror x to match, reverse y.

        ``x_new = x[0] + x[-1] - x[::-1]`` — the same span, samples
        relocated to their mirrored position. On a uniform grid this is
        x unchanged; on a non-uniform grid (e.g. SPICE's adaptive
        transient steps) a plain y-reversal would misattribute each
        sample to the wrong x.

        Returns
        -------
        Signal
            Reversed ``y`` on the mirrored ``x``; named ``flip(<name>)``.
            Deferred if this signal is unbound.
        """
        return self._new(
            f'flip({self.name})', self.type,
            _mirror_x(self.x),
            np.asarray(self.y)[::-1],
        )

    @_deferrable
    def interpolate(self, x_new):
        """Resample y at the given x values via linear interpolation.

        Delegates the axis validation, orientation and interpolation to the
        shared ``silk.metrics._guards._Axis`` that :func:`silk.metrics.sample`
        also uses, so the two cannot diverge. The source x axis
        must be finite and strictly monotonic: either ascending or
        descending (a DC sweep run downwards) is accepted, and the Signal
        itself is never sorted or modified. A repeated coordinate, a
        fold-back, a nonfinite coordinate, or x/y of different length or
        not 1-D raises ``ValueError``. Targets keep the order given;
        out-of-range targets take the nearest endpoint value; a NaN in a
        consumed sample propagates. A scalar target returns a scalar-x
        result and an empty target an empty one.

        Raises ``ValueError`` on a complex Signal: silk does not
        define what "interpolate a complex trace" should mean here, so this
        is a caller-error request rather than a
        value to silently compute. Interpolate ``.real`` and ``.imag``
        separately if that is what you want. (:func:`silk.metrics.sample`
        does interpolate complex values componentwise.)

        Parameters
        ----------
        x_new : array-like of float
            Target x values, in the units of ``x``. Values outside the
            signal's range take the nearest endpoint value.

        Returns
        -------
        Signal
            ``y`` sampled at ``x_new``; named ``interp(<name>)``. Deferred
            if this signal is unbound.

        Raises
        ------
        ValueError
            If ``y`` is complex, or if the source x axis is not finite and
            strictly monotonic, or x and y differ in length or are not 1-D.
        """
        y = np.asarray(self.y)
        if np.iscomplexobj(y):
            raise ValueError(
                f'interpolate: Signal {self.name!r} has complex y; '
                f'interpolate does not support complex traces. Interpolate '
                f'.real and .imag separately, e.g. '
                f'sig.real().interpolate(x_new) and '
                f'sig.imag().interpolate(x_new)'
            )
        from silk.metrics._guards import _axis  # local: avoid cycle
        x_target = np.asarray(x_new, dtype=float)
        y_target = _axis(self, 'interpolate').interp(
            x_target, y.astype(float))
        return self._new(
            f'interp({self.name})', self.type, x_target, y_target,
        )

    @_deferrable
    def versus(self, x_signal, *, name=None, allow_nonmonotonic=False):
        """Re-parametrise this Signal against another Signal's values.

        Returns a Signal whose x axis is ``x_signal.y`` and whose y is
        ``self.y``: the two signals are read sample-for-sample as one
        sweep, and the sweep variable is dropped. This is how one swept
        quantity is plotted against another, e.g. ``(gm / id).versus(id / w)``
        for a current-density lookup curve.

        Both signals must have the same number of samples (they come from
        the same sweep) and real values. The resulting x must be strictly
        monotonic so the result is a function graph that ``interpolate()``
        and the metrics can read: strictly *decreasing* x is reversed to
        ascending; a reversal or a repeated x value raises. Pass
        ``allow_nonmonotonic=True`` to keep sample order and get a
        parametric trace instead (fine to plot; ``interpolate()`` on it is
        not meaningful).

        Works on unbound probes too: the call is recorded and replayed when
        the testbench has run.

        Parameters
        ----------
        x_signal : Signal
            Supplies the new x values (its ``y``).
        name : str, optional
            Name of the result. Default ``'<self.name> vs <x_signal.name>'``.
        allow_nonmonotonic : bool, optional
            Keep sample order instead of raising on non-monotonic x.

        Raises
        ------
        TypeError
            *x_signal* is not a Signal.
        ValueError
            Sample counts differ, either signal is complex, or x is not
            strictly monotonic (the message names the first offending index)
            and *allow_nonmonotonic* is False.
        """
        if not isinstance(x_signal, Signal):
            raise TypeError(
                f'versus: x_signal must be a Signal, got '
                f'{type(x_signal).__name__}'
            )
        y = np.asarray(self.y)
        xs = np.asarray(x_signal.y)
        if np.iscomplexobj(y) or np.iscomplexobj(xs):
            raise ValueError(
                f'versus: {self.name!r} against {x_signal.name!r} has a '
                f'complex signal; take .real / .mag() of each first'
            )
        if y.shape != xs.shape:
            raise ValueError(
                f'versus: {self.name!r} has {y.size} samples but '
                f'{x_signal.name!r} has {xs.size}; they must come from the '
                f'same sweep'
            )
        xs = xs.astype(float)
        y = y.astype(float)
        result_name = name or f'{self.name} vs {x_signal.name}'
        if not allow_nonmonotonic and xs.size > 1:
            d = np.diff(xs)
            if np.all(d < 0):
                xs, y = xs[::-1], y[::-1]
            elif not np.all(d > 0):
                bad = int(np.flatnonzero(~(d > 0))[0]) if d[0] >= 0 \
                    else int(np.flatnonzero(~(d < 0))[0])
                raise ValueError(
                    f'versus: x values ({x_signal.name!r}) are not strictly '
                    f'monotonic: samples {bad} and {bad + 1} are '
                    f'{float(xs[bad])!r} and {float(xs[bad + 1])!r}. Pass '
                    f'allow_nonmonotonic=True for a parametric trace.'
                )
        out = self._new(result_name, self.type, xs, y)
        out.xlabel = x_signal.name
        return out

    #: Alias retained for compatibility; same behaviour, no warning.
    against = versus

    @_deferrable
    def derivative(self):
        """Numerical dy/dx via central differences.

        Delegates to :func:`silk.metrics.waveform.derivative <silk.metrics.derivative>`,
        which preserves ``y``'s dtype — a complex Signal (e.g. an .ac
        result) differentiates real and imaginary parts together, matching
        d/dw of H(jw), with no ``ComplexWarning``. Local import to avoid
        the silk.signals ↔ silk.metrics circular dependency, same pattern
        as :meth:`fft`. Previously, this method did
        ``np.asarray(self.y, dtype=float)`` unconditionally, silently
        dropping ``Im(y)`` with only a ``ComplexWarning``.

        Returns
        -------
        Signal
            dy/dx on the same ``x``, in ``y`` units per ``x`` unit. Deferred
            if this signal is unbound.
        """
        from silk.metrics.waveform import derivative as _derivative  # local: avoid cycle
        return _derivative(self)

    @_deferrable
    def integral(self):
        """Cumulative trapezoidal integral as a Signal (``y[0] = 0``).

        Complex-aware (like ``derivative``): the cumulative trapezoid of a complex ``y`` is well
        defined (real and imaginary parts integrate independently), so a
        complex Signal (e.g. an .ac result) comes back complex with no
        ``ComplexWarning``. Previously, this method did
        ``np.asarray(self.y, dtype=float)`` unconditionally, silently
        dropping ``Im(y)``. Computed inline (unlike ``derivative``, which
        delegates to ``silk.metrics.waveform``) because
        ``silk.metrics.waveform`` has no equivalent free function to
        delegate to.

        Returns
        -------
        Signal
            Integral of ``y`` over ``x`` (``y`` units times ``x`` units),
            zero at the first sample; named ``integ(<name>)``. Deferred if
            this signal is unbound.
        """
        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y)
        if not np.iscomplexobj(y):
            y = y.astype(float)
        out = np.zeros_like(y)
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))
        return self._new(
            f'integ({self.name})', self.type, x, out,
        )

    @_deferrable
    def fft(self, *, tclk: float, tstart: float = 0.0,
            n_samples=None, window='rectangular', scaling=None):
        """Spectrum as a Signal (xlabel='frequency'); see ``metrics.fft``
        for ``window=`` and ``scaling=``.

        Local import to avoid the silk.signals ↔ silk.metrics circular
        dependency.

        Parameters
        ----------
        tclk : float
            Sampling interval in seconds (> 0).
        tstart : float, default 0.0
            First sampling time in seconds.
        n_samples : int, optional
            Sample count; the largest fitting power of two when omitted.
        window : str or None, default 'rectangular'
            ``'hann'``, ``'hamming'``, ``'blackman'``, ``'rectangular'`` or
            ``None``.
        scaling : {None, 'amplitude', 'power'}, optional
            Output scaling; see :func:`silk.metrics.fft`.

        Returns
        -------
        Signal
            Spectrum with ``x`` in hertz. Raw (unscaled) ``y`` is squared
            magnitude. Deferred if this signal is unbound.

        Raises
        ------
        ValueError
            As :func:`silk.metrics.fft` (non-transient input, bad ``tclk``,
            window outside the trace, unknown window or scaling).
        """
        from silk.metrics.spectral import fft as _fft  # local: avoid cycle
        return _fft(self, tclk=tclk, tstart=tstart, n_samples=n_samples,
                    window=window, scaling=scaling)

    # ------------------------------------------------------------------

    @_deferrable
    def dbm(self, z0: float = 50.0):
        """Power in dBm: ``10·log10(|y|² / z0 / 1e-3)`` — voltage across
        a ``z0``-Ω load referenced to 1 mW.

        Parameters
        ----------
        z0 : float, default 50.0
            Reference impedance in ohms.

        Returns
        -------
        Signal
            Type ``'power'``, ylabel ``'dBm'``; named ``dbm(<name>)``. Zero
            amplitude gives ``-inf``. Deferred if this signal is unbound.

        Raises
        ------
        ValueError
            If ``z0 <= 0``.
        """
        if z0 <= 0:
            raise ValueError(f'z0 must be > 0, got {z0}')
        power_w = (np.abs(self.y) ** 2) / z0
        return self._new(
            f'dbm({self.name})', 'power', self.x,
            10.0 * np.log10(power_w / _DBM_REF_W),
            ylabel='dBm',
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new(self, name, sig_type, x, y, ylabel=None):
        """Return a new instance of the same concrete class.

        ``ylabel`` defaults to ``self.ylabel`` (preserves the original
        semantic). Pass an explicit string to override (e.g. when
        converting a voltage waveform to a dB or phase waveform).
        """
        obj = object.__new__(type(self))
        Signal.__init__(
            obj, name, sig_type, self.xlabel,
            self.ylabel if ylabel is None else ylabel,
            x, y,
        )
        return obj

    def _check_compatible(self, other):
        self._check_xlabels(other)
        self._check_types(other)
        self._check_xvalues(other)

    def _check_xlabels(self, other):
        if self.xlabel != other.xlabel:
            raise ValueError(
                f'Signals have different x-labels: '
                f'{self.xlabel!r} vs {other.xlabel!r}'
            )

    def _check_types(self, other):
        if self.type != other.type:
            raise ValueError(
                f'Signals have different types: '
                f'{self.type!r} vs {other.type!r}'
            )

    def _check_xvalues(self, other):
        """Require exactly equal coordinate shapes and values, without resampling.

        Equal copied arrays and equal values of different dtypes are accepted.
        """
        x, other_x = self.x, other.x
        context = (
            f'X-values in both signals do not match: '
            f'{self.name!r} ({x.size} samples, shape {x.shape}) and '
            f'{other.name!r} ({other_x.size} samples, shape {other_x.shape}); '
        )
        remedy = (
            'Use matching recorded coordinates or, when resampling is appropriate, '
            'silk.metrics.sample(other, x_new=reference.x).'
        )
        if x.shape != other_x.shape:
            raise ValueError(context + 'shape/length mismatch. ' + remedy)
        if not np.array_equal(x, other_x):
            index = int(np.flatnonzero(x != other_x)[0])
            raise ValueError(
                context + f'first differing index {index}: '
                f'{x.flat[index].item()!r} vs {other_x.flat[index].item()!r}. ' + remedy
            )
