"""make_device_class — creates Python classes from YAML device specs."""

import pickle
from dataclasses import dataclass, field
from typing import Optional

from silk.utils._utils import _render_param, render_pdk_device_line
from silk.elements.base import (
    AnalogElement, _PIN_UNSET, _validate_pin_value, param_context,
)
from silk.utils.deck_text import refuse_line_breaks


# ---------------------------------------------------------------------------
# Lightweight parameter constraint for factory-generated classes
# ---------------------------------------------------------------------------

@dataclass
class DeviceParamConstraint:
    """Parameter constraint for factory-generated device classes.

    Lighter than the PDK's ``ParameterConstraint`` — no min/max validation
    logic; just holds default, pdk_unit, and optional bounds for display.
    """
    default: object = None       # default value (None = required)
    pdk_unit: Optional[str] = None
    minval: Optional[str] = None
    maxval: Optional[str] = None
    resolution: Optional[str] = None

    def violation(self, value, param_name):
        """How *value* (numeric, already in the PDK's OWN unit) violates this
        constraint — a message string — or None if it is fine.

        Vocabulary matches :meth:`ParameterConstraint.validate_value`:
        ``'-1'`` on either bound means unbounded; ``pdk_unit is None`` means
        dimensionless, which must be an integer (``m=1.5`` is not a device).
        Comparisons carry a 1e-9 relative tolerance so a boundary value that
        went through unit conversion (0.22 µm → 2.1999999999999998e-07 m)
        does not false-positive against its own minimum.
        """
        import math
        unit = f' {self.pdk_unit}' if self.pdk_unit else ''
        if self.pdk_unit is None and not math.isclose(
                value, round(value), rel_tol=0.0, abs_tol=1e-9):
            return (f'{param_name}={value} must be an integer '
                    f'(dimensionless parameter)')

        def _end(v):
            if v in (None, '-1', -1):
                return None
            try:
                f = float(v)
            except (ValueError, TypeError):
                return None
            return None if f == -1 else f

        lo, hi = _end(self.minval), _end(self.maxval)
        if (lo is not None and value < lo
                and not math.isclose(value, lo, rel_tol=1e-9)):
            return (f'{param_name}={value}{unit} is below the PDK minimum '
                    f'{self.minval}{unit}')
        if (hi is not None and value > hi
                and not math.isclose(value, hi, rel_tol=1e-9)):
            return (f'{param_name}={value}{unit} exceeds the PDK maximum '
                    f'{self.maxval}{unit}')
        return None


# ---------------------------------------------------------------------------
# Attribute-access helpers for param descriptors
# ---------------------------------------------------------------------------

class ParamAccessor:
    """Returned when a ParamDescriptor is accessed on a class.

    Carries the constraint metadata and exposes ``.variable()`` to create
    a pre-configured :class:`~silk.circuit.variable.Variable`.

    Example::

        pdk.nmos1p8.w  →  ParamAccessor for param 'w'
        pdk.nmos1p8.w.variable(name='w_n', value=1.0)  →  Variable
    """

    def __init__(self, owner_cls, param_name, constraint=None):
        self._cls = owner_cls
        self._param = param_name
        self._constraint = constraint

    def variable(self, *, name, value=None, **kwargs):
        """Create a Variable configured from PDK constraints for this param.

        Parameters
        ----------
        name : str
            Variable name — required to avoid silent clashes in VariableSpace.
        value : float, optional
            Nominal / initial value.
        **kwargs
            Forwarded to :class:`~silk.circuit.variable.Variable`.
            Caller-supplied ``bounds`` and ``resolution`` override PDK values.
        """
        from silk.circuit.variable import Variable
        _name = name
        c = self._constraint
        if 'bounds' not in kwargs:
            # Shared derivation with AnalogElement.variable() — '-1' means an
            # unbounded end, and a None end must be handled by the variable
            # (validate/clip are None-safe), never crash it.
            kwargs['bounds'] = AnalogElement._constraint_bounds(c)
        if 'resolution' not in kwargs and c is not None and c.resolution is not None:
            try:
                kwargs['resolution'] = float(c.resolution)
            except (ValueError, TypeError):
                pass
        return Variable(_name, value=value, **kwargs)

    def __repr__(self):
        return f'ParamAccessor({self._cls.__name__}.{self._param})'


class ParamDescriptor:
    """Non-data descriptor placed on device/source classes for each param name.

    - Class access  ``(MyDevice.w)``  → :class:`ParamAccessor` with constraint metadata.
    - Instance access ``(mn.w)``      → ``mn.params.get('w')`` for :class:`AnalogElement`
                                        instances (params live in the ``params`` dict,
                                        not in ``instance.__dict__``, so the descriptor
                                        is always invoked for instance access).
                                        For sources, ``self.dc = value`` in ``__init__``
                                        stores the value in ``instance.__dict__['dc']``
                                        which shadows the descriptor automatically
                                        (non-data descriptor — no ``__set__``).
    """

    def __init__(self, param_name, constraint=None):
        self._param = param_name
        self._constraint = constraint

    def __get__(self, obj, objtype=None):
        if obj is None:
            # Class-level access: return the accessor
            return ParamAccessor(objtype, self._param, self._constraint)
        # Instance-level access: return current value from params dict
        # (sources store values in __dict__ and shadow this descriptor automatically)
        if hasattr(obj, 'params') and isinstance(getattr(obj, 'params', None), dict):
            return obj.params.get(self._param)
        return obj.__dict__.get(self._param)

    def __repr__(self):
        return f'ParamDescriptor({self._param!r})'


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_pin_names(spec: dict) -> tuple:
    """Extract pin names from a YAML spec, handling dict and list formats."""
    pins = spec.get('pinList', [])
    if isinstance(pins, dict):
        return tuple(pins.keys())
    return tuple(pins)


def _build_pin_descriptions(spec: dict) -> dict:
    """Extract {pin_name: description} from a YAML spec.

    Returns an empty dict for list-format pinLists (no descriptions).
    """
    pins = spec.get('pinList', [])
    if isinstance(pins, dict):
        return dict(pins)   # {d: drain, g: gate, ...}
    return {}               # list format has no descriptions


def _build_constraints(spec: dict) -> dict:
    """Build {param_name: DeviceParamConstraint} from a YAML spec.

    Handles both:
    - PDK format:  ``paramList: {name: {minval, maxval, pdk_unit}}``
    - Ideal format: ``paramConstraints: [{name, default, unit}]``
    """
    if 'paramList' in spec:
        result = {}
        for name, data in spec['paramList'].items():
            result[name] = DeviceParamConstraint(
                # explicit default (a numeric 0 counts), or minval as fallback when absent
                default=(data['default'] if data.get('default') is not None
                         else data.get('minval')),
                pdk_unit=data.get('pdk_unit'),
                minval=str(data['minval']),
                maxval=str(data['maxval']),
                # Manufacturing grid (e.g. sky130 w/l resolution: 0.005). Carried
                # so .variable() can inherit it — its inheritance code already
                # reads c.resolution, but this branch used to drop the field, so
                # PDK-derived variables silently sampled off-grid.
                resolution=(str(data['resolution'])
                            if data.get('resolution') is not None else None),
            )
        return result

    if 'paramConstraints' in spec:
        result = {}
        for entry in spec['paramConstraints']:
            result[entry['name']] = DeviceParamConstraint(
                default=entry.get('default'),
                # Ideal elements (silk/elements/analog_devices.yaml) keep the
                # 'unit:' key: it's a general annotation ('Ohm', 'F', 'V'), not
                # the PDK-prefix vocabulary pdk_unit: names.
                pdk_unit=entry.get('unit'),
            )
        return result

    return {}


def _apply_aliases(cls_name: str, kwargs: dict, alias_map: dict, kind_word: str) -> None:
    """Translate ``kwargs`` through ``alias_map`` ({alias: canonical}) in place.

    Data-driven: *alias_map* comes straight from the YAML
    spec's ``pinAliases``/``paramAliases``, so this function knows nothing
    device-specific — it works the same for a pin alias ('pos' -> 'p') or a
    parameter alias ('resistance' -> 'value').

    Raises
    ------
    ValueError
        An alias and its canonical name are both supplied, or two different
        aliases both target the same canonical name — either way, the
        conflicting names are both named in the message.
    """
    # Snapshotted before any alias is resolved, so a conflict is reported
    # against what the caller actually wrote — not against a canonical name
    # this function itself just filled in from an earlier alias.
    original_canonicals = {c for c in alias_map.values() if c in kwargs}
    supplied_via = {}
    for alias, canonical in alias_map.items():
        if alias not in kwargs:
            continue
        if canonical in original_canonicals:
            raise ValueError(
                f"{cls_name}(): got both {kind_word} {canonical!r} and its "
                f"alias {alias!r} — pass only one"
            )
        if canonical in supplied_via:
            raise ValueError(
                f"{cls_name}(): got both {kind_word} aliases "
                f"{supplied_via[canonical]!r} and {alias!r} for {canonical!r} "
                f"— pass only one"
            )
        kwargs[canonical] = kwargs.pop(alias)
        supplied_via[canonical] = alias


# The device-parameter renderer lives in silk/utils/_utils.py — ONE spelling
# for both build paths (TASK-73). Re-exported here because this module's
# name for it is the one silk.sources.classes and the generated classes use.


# ---------------------------------------------------------------------------
# Pickling by name (TASK-301.8)
# ---------------------------------------------------------------------------
#
# A circuit composed from blocks built under two separate load_pdk('sky130')
# calls holds device instances whose classes are structurally identical
# (same YAML, same device_name) but are two different *class objects* — the
# module-namespace registration below keeps only the last (PDK) or first
# (ideal) one. Default pickling resolves an instance's class by module +
# qualname and demands the object it finds back be `is` the one it pickled
# (Python's save_reduce identity check), which fails for whichever load's
# class is no longer in that slot: "Can't pickle <class '...nmos1p8'>: it's
# not the same object as ...nmos1p8".
#
# The fix reconstructs by NAME instead of by identity: pickle the device
# type name and rebuild against whatever class is CURRENTLY registered for
# it in this module, in whichever process unpickles. A multiprocessing.Pool
# worker forked from a process that already ran every load_pdk() call (the
# normal case — those calls happen at module import time, before the pool
# is created) always has a structurally-equivalent class registered, so the
# rebuilt instance behaves identically for netlisting and simulation. This
# only changes pickling; it does not touch registration order, class
# identity, or any isinstance/type check within a single process — the
# TASK-312.3 last/first-registered-wins policy below is unchanged.
#
# "Structurally equivalent" has exactly one hole: two ``load_pdk()`` calls
# for the same PDK NAME can still disagree on ``user_units``, and
# ``_param_scales`` (baked into the class by ``BasePDK._param_scales()`` at
# parse time) is the one class attribute that tracks that — everything else
# a factory class carries (pin/param names, constraints, ``_pdk_device``,
# ``_spice_prefix``, ...) comes straight from the bundled YAML entry for
# that device name and cannot differ between two loads of the same PDK.
# Silently substituting "whatever nmos1p8 is registered right now" for an
# instance built under a different ``user_units`` re-scaled every
# caller-supplied value by the WRONG factor (caught by
# ``tests/db/test_unit_identity.py``: an 'si'-unit rebuild of a 'u'-unit
# ``w=2.2`` device rendered ``w=2.2Meg``). So the scales an instance was
# built with travel with it and are restored onto a private, cached
# subclass when the currently-registered class disagrees — the common
# case (same ``user_units`` everywhere, e.g. every shipped example) never
# creates one; `cls` stays the exact currently-registered class.
#
# Hand-written AnalogElement subclasses that are NOT factory output (e.g.
# ``silk.elements.passives.Resistor``, which subclasses a factory-generated
# ideal class but has its own stable module + qualname) inherit this
# __reduce__ too, since it lives on the base factory class — so it checks
# `type(self)` at call time and falls back to plain, identity-based
# reconstruction for anything not living in this module.

#: {(device_type_name, sorted param_scales items): class}, process-local.
#: Keeps a single rebuilt class per distinct scale set instead of minting a
#: new subclass on every unpickle, so repeated round-trips of instances
#: from the same load share one class object (and one another's
#: isinstance/type checks), same as the classes ``make_device_class``
#: itself registers.
_rebuilt_class_cache = {}


def _rebuild_factory_element(device_type_name, param_scales, state=None):
    """Reconstruct a factory-generated :class:`AnalogElement` by device name.

    Looks up ``device_type_name`` in *this* module's namespace at
    unpickling time rather than trusting the exact class object that was
    pickled, and restores ``param_scales`` (the one attribute that can
    legitimately differ between two loads of the same PDK) — see the note
    above.
    """
    import sys
    cache_key = (device_type_name, tuple(sorted(param_scales.items())))
    cls = _rebuilt_class_cache.get(cache_key)
    if cls is None:
        try:
            base_cls = sys.modules[__name__].__dict__[device_type_name]
        except KeyError:
            raise pickle.PicklingError(
                f"cannot unpickle {device_type_name!r}: no class registered "
                f"in {__name__} in this process — the PDK or element module "
                f"that defines it was never imported here"
            ) from None
        if dict(base_cls._param_scales) == param_scales:
            cls = base_cls
        else:
            cls = type(base_cls.__name__, (base_cls,),
                       {'_param_scales': dict(param_scales)})
        _rebuilt_class_cache[cache_key] = cls
    obj = cls.__new__(cls)
    if state is not None:
        obj.__dict__.update(state)
    return obj


def _rebuild_plain_element(cls, state=None):
    """Default-shaped reconstruction for an ``AnalogElement`` subclass with
    its own stable class identity (not factory output)."""
    obj = cls.__new__(cls)
    if state is not None:
        obj.__dict__.update(state)
    return obj


def _factory_element_reduce(self):
    from silk.circuit.object_holders import detached_state
    state = detached_state(self)
    cls = type(self)
    if cls.__module__ == __name__:
        return (_rebuild_factory_element,
                (cls._device_type_name, dict(cls._param_scales)), state)
    return (_rebuild_plain_element, (cls,), state)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_device_class(device_name: str, spec: dict, is_pdk: bool = True,
                      param_scales: dict = None):
    """Create a Python class for a device from its YAML spec.

    Parameters
    ----------
    device_name : str
        YAML key for the device (e.g. ``'nmos1p8'``, ``'resistor'``).
    spec : dict
        Parsed YAML entry for this device.
    is_pdk : bool
        ``True`` for PDK devices (default SPICE prefix ``XM``),
        ``False`` for ideal elements (prefix from YAML ``spice_prefix``).

    Returns
    -------
    type
        A subclass of :class:`AnalogElement` with a constructor whose
        keyword arguments match the device's pins and parameters.
    """
    pin_names         = _build_pin_names(spec)
    pin_descriptions  = _build_pin_descriptions(spec)
    param_constraints = _build_constraints(spec)
    param_names       = tuple(param_constraints.keys())
    param_defaults    = {p: c.default for p, c in param_constraints.items()}
    # Alternate spellings for a pin or parameter name (e.g. 'pos' -> 'p',
    # 'resistance' -> 'value'), read entirely from the YAML spec — nothing
    # device-specific is hard-coded in Python (TASK-299 AC#2). Metadata only:
    # never added to _pin_names/_param_names/_param_constraints, so
    # export_schema() (silk/elements/schema.py) never sees them.
    pin_aliases       = dict(spec.get('pinAliases') or {})
    param_aliases     = dict(spec.get('paramAliases') or {})
    pdk_device_str    = spec.get('PDKDevice', '')
    category          = spec.get('category', '')
    # Per-device op-point declaration (TASK-330.3 gate finding), read from
    # the SAME dict PDKDevice.from_dict parses for the add_element path, so
    # the two families can never disagree about a model. Undeclared (either
    # key absent) means None/None -- save_device_params skips this device
    # entirely rather than guessing.
    _op_params_raw    = spec.get('op_params')
    op_params_tuple   = tuple(_op_params_raw) if _op_params_raw else None
    op_internal_name  = spec.get('op_internal_name')
    if op_internal_name is not None:
        op_internal_name = str(op_internal_name)
    if not op_params_tuple or not op_internal_name:
        op_params_tuple = None
        op_internal_name = None
    # Third per-device declaration alongside the pair above (TASK-330.4):
    # channel polarity ('n'/'p'), also read from the same dict, also
    # validated once at PDKDevice.from_dict time (never re-validated here).
    polarity = spec.get('polarity')
    if polarity is not None:
        polarity = str(polarity).lower()
    # Fourth per-device declaration (TASK-330.5): which simulator backends
    # this device's op-point addressing has been verified on by a real run,
    # paired with op_params/op_internal_name exactly like those two are
    # paired with each other -- see PDKDevice.op_backends. Also read from
    # the same dict, also validated once at PDKDevice.from_dict time.
    _op_backends_raw  = spec.get('op_backends')
    op_backends_tuple = tuple(_op_backends_raw) if _op_backends_raw else None
    if not op_params_tuple or not op_internal_name or not op_backends_tuple:
        op_params_tuple   = None
        op_internal_name  = None
        op_backends_tuple = None
    # Optional source class pointer: a `source_class:` string in the YAML spec
    # names the rich SourceBase class (VoltageSource, CurrentSource, etc.) that
    # this ideal source corresponds to. None for non-source elements.
    source_class      = spec.get('source_class')
    # Default prefix: XM for PDK subcircuit devices, from YAML for ideal
    spice_prefix      = spec.get('spice_prefix', 'XM' if is_pdk else 'X')

    # ------------------------------------------------------------------
    # __init__ for the generated class
    # ------------------------------------------------------------------

    def __init__(self, name, **kwargs):
        AnalogElement.__init__(self, name)

        # Resolve alias spellings ('pos'/'neg', 'resistance'/'res', ...) to
        # their canonical pin/parameter name before either resolution loop
        # below ever sees them (TASK-299 AC#2).
        _apply_aliases(self.__class__.__name__, kwargs,
                       self.__class__._pin_aliases, 'pin')
        _apply_aliases(self.__class__.__name__, kwargs,
                       self.__class__._param_aliases, 'parameter')

        # Resolve nets — every pin must be supplied, and what it is supplied
        # WITH must be able to be one node token on the device line. The
        # presence check alone used to let `g=''` and `g=None` through: the
        # first emitted a token-short line that ngspice mis-parsed into an
        # invented node, the second died inside ' '.join at netlist time
        # naming neither the instance nor the pin (TASK-41).
        for pin in self.__class__._pin_names:
            if pin not in kwargs:
                raise TypeError(
                    f"{self.__class__.__name__}() missing required net for pin '{pin}'"
                )
            self.nets[pin] = _validate_pin_value(self, pin, kwargs.pop(pin))

        # Resolve params — use defaults where available.
        #
        # Track WHICH came from the caller (in _caller_supplied, initialised by
        # AnalogElement.__init__ and also fed by __setattr__). Only those are
        # unit-converted at emission: a catalog `default:` is already in the
        # PDK's own unit, like the `minval`/`maxval` beside it, and is a fixed
        # physical geometry — it must not change meaning when the author
        # switches user_units.
        for p, constraint in self.__class__._param_constraints.items():
            if p in kwargs:
                self.params[p] = refuse_line_breaks(
                    kwargs.pop(p), who=param_context(self, p))
                self._caller_supplied.add(p)
            elif constraint.default is not None:
                self.params[p] = constraint.default
            else:
                raise TypeError(
                    f"{self.__class__.__name__}() missing required parameter '{p}'"
                )

        if kwargs:
            raise TypeError(
                f"{self.__class__.__name__}() got unexpected keyword arguments: "
                f"{list(kwargs.keys())}"
            )

        # Auto-register with any module currently active via `with mod:`.
        # Deferred import to avoid a circular import at load time.
        from silk.circuit.modules import _active_module
        _mod = _active_module()
        if _mod is not None:
            _mod.add(self)

    # ------------------------------------------------------------------
    # to_spice for the generated class
    # ------------------------------------------------------------------

    def to_spice(self):
        prefix   = self.__class__._spice_prefix
        pdk_dev  = self.__class__._pdk_device
        # Final structural guard: exactly one node token per pin, in pin
        # order. The constructor and AnalogElement.__setattr__ already reject
        # anything else, so this only fires when `device.nets` was mutated as
        # a raw dict — but it is what makes "never fewer node tokens than
        # _pin_names" true of the emitter itself rather than of its callers.
        nets     = [_validate_pin_value(self, p, self.nets.get(p, _PIN_UNSET))
                    for p in self.__class__._pin_names]
        net_str  = ' '.join(nets)
        # Scale the caller's value into the PDK's own unit before formatting.
        # Empty under user_units='pdk' (the caller already writes PDK units), so
        # this is a no-op for every circuit written before scales existed.
        scales = self.__class__._param_scales
        supplied = self._caller_supplied
        # Fail LOUD on constraint violations, at the moment the deck line is
        # produced (2026-08-23 review): before this, w=-1 / l=0 / m=1.5 were
        # written verbatim into the simulation deck with no check anywhere on
        # this path. Only caller-supplied numeric values are checked — a
        # catalog default is the PDK author's own number, and a string value
        # is a SPICE literal passed through, not a quantity silk can judge.
        # Values are compared AFTER unit conversion, in the PDK's own unit —
        # the same unit the catalog's minval/maxval are written in. (The
        # legacy add_element path keeps its historical warn-and-proceed
        # behavior via ParameterConstraint.validate_value.)
        if pdk_dev:
            problems = []
            constraints = getattr(self.__class__, '_param_constraints', {}) or {}
            for p in self.__class__._param_names:
                if p not in supplied:
                    continue
                c = constraints.get(p)
                if c is None:
                    continue
                val = self.params[p]
                if hasattr(val, 'value'):        # Variable duck-type
                    val = val.value
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    continue
                msg = c.violation(val * scales.get(p, 1.0), p)
                if msg is not None:
                    problems.append(msg)
            if problems:
                from silk.exceptions import CircuitError
                raise CircuitError(
                    f'{self.name} ({pdk_dev}): invalid parameter value(s): '
                    + '; '.join(problems)
                    + '. Caller-supplied values are checked against the PDK '
                      'catalog (min/max/integer) when the netlist is emitted.'
                )
        if pdk_dev:
            # PDK subcircuit: XMmn1 vout vin gnd gnd sky130_... w=2u l=150n
            # through the one PDK line writer (TASK-416).
            return render_pdk_device_line(
                prefix, self.name, nets, pdk_dev,
                [(p, self.params[p], scales.get(p, 1.0) if p in supplied else 1.0)
                 for p in self.__class__._param_names])
        from silk.circuit.emitted_names import emitted_component
        from silk.utils.deck_text import card_text
        instance = (card_text(prefix, who=f'{self.name!r} SPICE prefix')
                    + emitted_component(self.name))
        param_str = ' '.join(
            f'{p}={_render_param(self.params[p], scales.get(p, 1.0) if p in supplied else 1.0, who=param_context(self, p))}'
            for p in self.__class__._param_names
        )
        # Ideal element: Rr1 vout gnd 1k   (single value, no name=val)
        # For single-param ideal elements, emit value directly
        if len(self.__class__._param_names) == 1:
            p = self.__class__._param_names[0]
            # vsource/isource: V/I prefix, DC keyword
            if prefix in ('V', 'I'):
                line = f'{instance} {net_str} DC {_render_param(self.params[p], who=param_context(self, p))}'
            else:
                line = f'{instance} {net_str} {_render_param(self.params[p], who=param_context(self, p))}'
        else:
            line = f'{instance} {net_str}'
            if param_str:
                line += f' {param_str}'
        return line + '\n'

    # ------------------------------------------------------------------
    # Build the class with type()
    # ------------------------------------------------------------------

    cls = type(device_name, (AnalogElement,), {
        '__init__':            __init__,
        'to_spice':            to_spice,
        '__reduce__':          _factory_element_reduce,
        '_pin_names':          pin_names,
        '_pin_descriptions':   pin_descriptions,
        '_param_names':        param_names,
        '_param_defaults':     param_defaults,
        '_param_constraints':  param_constraints,
        '_spice_prefix':       spice_prefix,
        '_param_scales':       dict(param_scales or {}),
        '_pdk_device':         pdk_device_str,
        '_op_params':          op_params_tuple,
        '_op_internal_name':   op_internal_name,
        '_polarity':           polarity,
        '_op_backends':        op_backends_tuple,
        '_source_class':       source_class,
        '_device_type_name':   device_name,
        '_is_pdk_device':      is_pdk,
        '_category':           category,
        '_pin_aliases':        pin_aliases,
        '_param_aliases':      param_aliases,
        'device_name':         device_name,   # backward-compat attribute
    })

    # Add a ParamDescriptor for each param so that MyDevice.w returns a
    # ParamAccessor (and mn.w returns mn.params.get('w') via the descriptor).
    for pname, constraint in param_constraints.items():
        setattr(cls, pname, ParamDescriptor(pname, constraint))

    # Register in this module's namespace so pickle can round-trip
    # ``AnalogElement`` instances across process boundaries (used by
    # silk.faults.FaultCampaign's multiprocessing pool). For a PDK device,
    # last-registered wins for a given ``device_name`` — in practice PDK
    # YAMLs produce structurally equivalent classes for the same name, so
    # this is safe. For an IDEAL element (is_pdk=False, e.g. 'resistor'),
    # first-registered wins instead: the package-level singleton
    # (``silk.elements.elements``, built at import time) always registers
    # first, so its classes stay the pickle-stable ones; a later, private
    # ``ElementLibrary(custom_yaml_path)`` (e.g. built by a test to load an
    # alternate catalog) must not silently invalidate every already-pickled
    # instance built from the singleton by overwriting this slot under it
    # (TASK-312.3: surfaced once tests started pickling Modules containing
    # an ``elements.resistor`` built via ``Module.add()`` instead of the
    # hand-written ``passives.Resistor``, which has its own stable identity
    # at ``silk.elements.passives.Resistor`` and never hit this path).
    import sys as _sys
    cls.__module__ = __name__
    module_ns = _sys.modules[__name__].__dict__
    if is_pdk or device_name not in module_ns:
        module_ns[device_name] = cls

    return cls
