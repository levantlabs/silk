"""Source factory (spec 2026-07-07 §5.2).

Generates the declarative source surfaces from ``source_metadata.yaml``,
delegating to the ``VoltageSource``/``CurrentSource`` + waveform classes (the
"referenced behavior"). Two surfaces over one truth:

- **B** — quantity namespace: ``sources.voltage.pulse(...)`` / ``sources.current.dc(...)``
- **A** — flat alias: ``sources.vpulse(...)`` / ``sources.idc(...)`` — the *same*
  function object as B (aliases, not copies).

Every generated constructor builds a real ``VoltageSource``/``CurrentSource``,
so ``add_source``, the optimization-variable API, and emission are unchanged;
waveform kinds fill the source's specific slot (``pulse=``/``pwl=``/… — for
byte-identical emission incl. PWL's series trick) or the generic ``waveform=``
slot for a new kind that has no specific kwarg.
"""
import inspect
from types import SimpleNamespace

from silk.sources import classes as _classes
from silk.sources import waveforms as _waveforms
from silk.sources.schema import _load_metadata


def _resolve_terminals(p, n, positive, negative):
    """Accept terminals as ``p``/``n`` OR ``positive``/``negative`` (both are
    supported — decided). Returns ``(positive, negative)``."""
    pos = positive if positive is not None else p
    neg = negative if negative is not None else n
    if pos is None or neg is None:
        raise TypeError(
            "source requires two terminals — pass p/n positionally or "
            "positive=/negative= as keywords")
    return pos, neg


def _make_scalar_ctor(Cls, pins, scalar, other):
    """A scalar kind (``dc``/``ac``): the positional value *is* that scalar; the
    other scalar is an optional keyword offset. Requires the value (no silent
    0 V/0 A supply) and rejects extras (spec §5.2, M2)."""
    def ctor(name, p=None, n=None, value=None, *, positive=None, negative=None, **kw):
        pos, neg = _resolve_terminals(p, n, positive, negative)
        # Accept the quantity by name (e.g. dc=/ac=) as an alias for value=.
        if scalar in kw:
            if value is not None:
                raise TypeError(f"{name!r}: pass the {scalar!r} value once (as value= or {scalar}=), not both")
            value = kw.pop(scalar)
        if value is None:
            raise TypeError(f"{name!r}: {Cls.__name__} {scalar!r} requires a value "
                            f"(positionally, as value=, or as {scalar}=)")
        extra = set(kw) - ({other} if other else set())
        if extra:
            raise TypeError(f"{name!r}: unexpected argument(s) {sorted(extra)}")
        args = {pins[0]: pos, pins[1]: neg, scalar: value}
        if other in kw:
            args[other] = kw[other]
        return Cls(name, **args)
    ctor.__name__ = f"{Cls.__name__.lower()}_{scalar}"
    ctor.__qualname__ = ctor.__name__
    unit = 'volts' if Cls._spice_prefix == 'V' else 'amps'
    ctor.__doc__ = (
        f"Construct a :class:`silk.sources.{Cls.__name__}` with ``{scalar}=<value>``.\n\n"
        f"Parameters\n----------\n"
        f"name : str\n    Instance name.\n"
        f"p : str, optional\n    Positive terminal (net name or "
        f":class:`~silk.circuit.Net`). Give either ``p`` and ``n`` or the keywords\n"
        f"    ``positive`` and ``negative``.\n"
        f"n : str, optional\n    Negative terminal.\n"
        f"value : float, optional\n    The ``{scalar}`` value in {unit}. It may instead be "
        f"passed as the keyword ``{scalar}``; one of the two is required.\n"
        f"positive : str, optional\n    Positive terminal, keyword-only form of ``p``.\n"
        f"negative : str, optional\n    Negative terminal, keyword-only form of ``n``.\n"
        f"**kw\n    " + (f"Optionally ``{other}`` (the other offset, in {unit}) and "
                           if other else "Optionally ") +
        f"the keyword ``{scalar}``. Anything else raises ``TypeError``.\n\n"
        f"Returns\n-------\n"
        f"silk.sources.{Cls.__name__}\n    The new source, with ``{scalar}`` set and no waveform.\n\n"
        f"Raises\n------\n"
        f"TypeError\n    If either terminal is missing, the value is missing or given "
        f"twice, or an unexpected keyword is passed.\n"
        f"ValueError\n    If a terminal is empty or contains whitespace; see "
        f":class:`silk.sources.SourceBase`.")
    return ctor


def _make_waveform_ctor(Cls, pins, kind, Emitter, slot):
    """A waveform kind: build the emitter from its own params and fill ``slot``
    (the specific kwarg where one exists — byte-identical, incl. PWL's series
    trick — else the generic ``waveform=`` slot). ``dc``/``ac`` are optional
    coexisting offsets."""
    def ctor(name, p=None, n=None, *wf_args,
             positive=None, negative=None, dc=0, ac=0, **wf_kwargs):
        pos, neg = _resolve_terminals(p, n, positive, negative)
        wave = Emitter(*wf_args, **wf_kwargs)
        args = {pins[0]: pos, pins[1]: neg, 'dc': dc, 'ac': ac, slot: wave}
        return Cls(name, **args)
    wf_params = [p for p in inspect.signature(Emitter.__init__).parameters if p != 'self']
    ctor.__name__ = f"{Cls.__name__.lower()}_{kind}"
    ctor.__qualname__ = ctor.__name__
    unit = 'volts' if Cls._spice_prefix == 'V' else 'amps'
    params = inspect.signature(Emitter.__init__).parameters
    positional = [n for n, q in params.items() if n != 'self'
                  and q.kind is not inspect.Parameter.KEYWORD_ONLY]
    keyword_only = [n for n, q in params.items() if q.kind is inspect.Parameter.KEYWORD_ONLY]
    order = ', '.join(positional) if positional else 'none'
    kw_only = (f" Keyword-only: {', '.join(keyword_only)}." if keyword_only else '')
    ctor.__doc__ = (
        f"Construct a :class:`silk.sources.{Cls.__name__}` with a "
        f":class:`silk.sources.{Emitter.__name__}` waveform.\n\n"
        f"Parameters\n----------\n"
        f"name : str\n    Instance name.\n"
        f"p : str, optional\n    Positive terminal (net name or "
        f":class:`~silk.circuit.Net`). Give either ``p`` and ``n`` or the keywords\n"
        f"    ``positive`` and ``negative``.\n"
        f"n : str, optional\n    Negative terminal.\n"
        f"*wf_args\n    Positional waveform parameters, in the order {order}.\n"
        f"positive : str, optional\n    Positive terminal, keyword-only form of ``p``.\n"
        f"negative : str, optional\n    Negative terminal, keyword-only form of ``n``.\n"
        f"dc : float, optional\n    Separate DC offset in {unit}. Default ``0``.\n"
        f"ac : float, optional\n    Small-signal AC magnitude in {unit}. Default ``0``.\n"
        f"**wf_kwargs\n    Waveform parameters by name; see "
        f":class:`silk.sources.{Emitter.__name__}` for their meaning and units."
        f"{kw_only}\n\n"
        f"Returns\n-------\n"
        f"silk.sources.{Cls.__name__}\n    The new source carrying the waveform.\n\n"
        f"Raises\n------\n"
        f"TypeError\n    If a terminal is missing, or the waveform arguments do not "
        f"match its constructor.\n"
        f"ValueError\n    If the waveform or a terminal rejects its value; see "
        f":class:`silk.sources.{Emitter.__name__}`.")
    return ctor


def build_source_surfaces():
    """Return ``(namespaces, flat)``: ``namespaces = {'voltage': ns, 'current':
    ns}`` (surface B) and ``flat = {'vdc': ctor, 'vpulse': ctor, …}`` (surface A,
    the *same* callables). Driven entirely by ``source_metadata.yaml``: a new
    quantity or (simple) waveform entry auto-produces both surfaces."""
    meta = _load_metadata()
    wf_map = meta.get('waveforms', {})              # kind -> emitter class name
    namespaces, flat = {}, {}

    for class_name, sm in meta['sources'].items():
        quantity, alias = sm.get('quantity'), sm.get('alias')
        if quantity is None or alias is None:
            continue                                # not factory-enabled
        Cls = getattr(_classes, class_name)
        pins = sm['pins']
        scalars = sm.get('scalar_params', [])
        ns = SimpleNamespace()

        for i, sk in enumerate(scalars):            # dc, ac
            other = scalars[1 - i] if len(scalars) == 2 else None
            ctor = _make_scalar_ctor(Cls, pins, sk, other)
            setattr(ns, sk, ctor)
            flat[alias + sk] = ctor                 # e.g. 'vdc' (same object)

        cls_params = set(inspect.signature(Cls.__init__).parameters)
        for kind, emitter_name in wf_map.items():   # pulse, pwl, sine, …
            Emitter = getattr(_waveforms, emitter_name)
            slot = kind if kind in cls_params else 'waveform'
            ctor = _make_waveform_ctor(Cls, pins, kind, Emitter, slot)
            setattr(ns, kind, ctor)
            flat[alias + kind] = ctor               # e.g. 'vpulse' (same object)

        namespaces[quantity] = ns

    return namespaces, flat
