"""Base PDK class and shared data structures for silk PDK implementations.

All PDK classes should inherit from BasePDK. The standard silk YAML schema
is parsed here; subclasses override initializeDeviceList() for other formats.
Corner-to-file mapping is data, not code: a manifest corner's ``includes:``
list names the (file, section) pairs to emit. (``getCornerPath`` used to be
documented as the hook for this; it was never called by anything and is gone.)
"""

import sys
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from silk.exceptions import CircuitError, DuplicateNameError, UndeclaredNetError, SimulationError, PDKError
from silk.utils._utils import render_pdk_device_line
from silk.elements.base import _PIN_UNSET, pin_context, validate_net_token


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader that raises on duplicate mapping keys.

    PyYAML's default behavior is to silently accept duplicate keys in a
    mapping node (the last one wins). For silk PDK YAMLs this is unsafe:
    a typo or merge-conflict leaving two definitions of the same device
    would silently change that device's parameters without any warning.
    This loader enforces uniqueness at the mapping-construction layer.
    """


def _strict_construct_mapping(loader, node, deep=False):
    """Replacement construct_mapping that raises on duplicate keys."""
    if not isinstance(node, yaml.MappingNode):
        raise yaml.constructor.ConstructorError(
            None, None,
            f'expected a mapping node, but found {node.id}',
            node.start_mark,
        )
    seen: set = set()
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            hash(key)
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                'while constructing a mapping', node.start_mark,
                f'found unhashable key ({exc})', key_node.start_mark,
            )
        if key in seen:
            raise yaml.constructor.ConstructorError(
                'while constructing a mapping', node.start_mark,
                f'found duplicate key {key!r}', key_node.start_mark,
            )
        seen.add(key)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _strict_construct_mapping,
)

# ---------------------------------------------------------------------------
# User-units constants — pass as user_units= to any PDK constructor
# ---------------------------------------------------------------------------

#: ``user_units`` value meaning the author writes each parameter in the unit
#: the PDK catalog declares for it (for example micrometres for W/L in sky130).
#: No value is rescaled. It is the default of a bare PDK constructor, which
#: prints a notice to standard error; :func:`~silk.pdk.load_pdk` takes the
#: manifest's ``user_units`` instead, which every shipped manifest sets to
#: ``'u'``. Any prefix token (``'u'``, ``'n'``, ``'p'``, ``'m'``, ``'base'``) is
#: also a valid ``user_units``: it names the unit the circuit author writes in,
#: the portable setting, since it holds one author-facing unit across PDKs that
#: disagree (sky130 takes micrometres, GF180MCU and SG13G2 take metres).
UNIT_PDK = 'pdk'
#: ``user_units`` value meaning the author writes SI base units (metres,
#: farads). The PDK converts them to its own units before writing the netlist
#: and before range checking.
UNIT_SI = 'si'

# SI base unit → the PREFIX the PDK writes its numbers in.
#
# Note these are prefixes, not units: the same table serves lengths (metres →
# µm) and capacitances (farads → pF), which is why the tokens are prefix names
# and not 'micrometre'/'picofarad'.
#
# 'base' means NO prefix — the PDK's models take the SI base unit as-is. It has
# a word rather than a letter because every letter is spoken for: 'm' is milli,
# so a metres-native PDK could not previously declare itself at all. GF180MCU
# and IHP SG13G2 are both base-unit PDKs (their .subckt lines read `w=10u`, so
# the model expects metres); sky130 is 'u' (its subckts take µm).
_SI_TO_PDK_SCALE = {
    'base': 1.0,   # no prefix — SI base unit (metres, farads)
    'u': 1e6,      # metres → micrometres
    'n': 1e9,      # metres → nanometres
    'p': 1e12,     # farads → picofarads
    'm': 1e3,      # metres → millimetres
}

# How many times a prefix applies, per QUANTITY. A length in µm scales by 1e6;
# an AREA in µm² scales by 1e6 SQUARED. Without this, an author writing microns
# had diode `area` scaled by 1e6 instead of 1e12 — 0.09 µm² emitted as 90000.
#
# The root confusion is that `pdk_unit:` names a prefix but is used on parameters of
# different dimensions: sky130 tags diode area `p` because 1e12 happens to be
# the m²→µm² factor, not because an area is measured in picofarads. Declaring
# the quantity separates the two.
# Only the two dimensions a catalog actually declares. 'capacitance' and
# 'resistance' used to sit here as well and were dead weight: nothing declared
# them, and they were wrong in principle — this table is indexed by a LENGTH
# prefix, so a farad or an ohm has no business in it. Add a dimension when a
# catalog needs one, not in anticipation of one.
_QUANTITY_EXPONENT = {
    'length': 1,
    'area': 2,
}


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised for a malformed or inconsistent PDK device catalog.

    Raised while a PDK loads or is queried for catalog-derived facts: YAML that
    does not parse, a required catalog field that is missing, an unrecognised
    ``op_params``, ``op_backends`` or ``polarity`` entry, a length parameter
    without a recognised ``pdk_unit``, an unknown ``quantity``, or a catalog
    that tags its lengths with more than one unit prefix. A parameter value
    outside its range is not an error of this kind: it is reported by
    :meth:`~silk.pdk.ParameterConstraint.validate_value` as a warning on
    standard output, or rejected when the netlist is emitted.

    It derives from :class:`Exception` directly, not from the silk
    ``CircuitError`` hierarchy.
    """
    pass


@dataclass
class Pin:
    """One pin of a catalog device in the legacy ``add_element`` style.

    Parameters
    ----------
    name : str
        Short pin name as written in the netlist, such as ``'d'`` or ``'g'``.
    net_name : str
        Descriptive name from the catalog, such as ``'drain'`` or ``'gate'``.
    net : str, optional
        Net the pin is connected to; ``None`` until the device is connected.

    Examples
    --------
    >>> from silk.pdk import Pin
    >>> pin = Pin('d', 'drain')
    >>> pin.name, pin.net_name, pin.net
    ('d', 'drain', None)
    """
    #: Short pin name used in the netlist (for example ``'d'``, ``'g'``).
    name: str
    #: Descriptive name from the catalog (for example ``'drain'``, ``'gate'``).
    net_name: str
    #: Net the pin is connected to; ``None`` until the device is connected.
    net: Optional[str] = None


@dataclass
class ParameterConstraint:
    """Range, unit and default of one device parameter in the catalog.

    Numbers given for ``minval`` and ``maxval`` are stored as strings. A
    ``pdk_unit`` left as ``None`` is inferred from the last character of
    ``minval`` when that is ``u``, ``n``, ``p`` or ``m``; a plain number leaves
    it ``None``, which marks the parameter dimensionless.

    Parameters
    ----------
    minval : str
        Lower bound in the PDK's own unit; ``'-1'`` means unbounded.
    maxval : str
        Upper bound in the PDK's own unit; ``'-1'`` means unbounded.
    value : object
        The value the caller supplied, or ``None``. Filled when the device is
        instantiated.
    pdk_unit : str, optional
        Unit prefix the PDK's models take this parameter in: ``'base'``,
        ``'u'``, ``'n'``, ``'p'`` or ``'m'``. ``None`` means dimensionless.
    quantity : {'length', 'area'}, optional
        Dimension of the parameter: a unit prefix applies once to a length and
        twice to an area. ``None`` is read as ``'length'``.
    default : object, optional
        Value emitted when the caller supplies none; already in the PDK's own
        unit and never rescaled.

    Examples
    --------
    >>> from silk.pdk import ParameterConstraint
    >>> constraint = ParameterConstraint(minval='0.42u', maxval='-1', value=None)
    >>> constraint.pdk_unit
    'u'
    """
    #: Lower bound in the PDK's own unit; ``'-1'`` means unbounded.
    minval: str
    #: Upper bound in the PDK's own unit; ``'-1'`` means unbounded.
    maxval: str
    #: Value the caller supplied; ``None`` until the device is instantiated.
    value: object
    #: Unit prefix the PDK takes this parameter in; ``None`` is dimensionless.
    pdk_unit: Optional[str] = None
    #: Dimension (``'length'`` or ``'area'``) deciding how often the unit
    #: prefix applies when converting from the author's unit.
    quantity: Optional[str] = None
    #: Value emitted when the caller supplies none (the PDK's own unit, never
    #: rescaled). Resolved as the factory path resolves it, so both APIs fill
    #: an omitted parameter with the same number.
    default: Optional[object] = None

    emit_if_supplied: bool = False
    emission_set: bool = False

    def __post_init__(self):
        self.minval = self._normalize(self.minval)
        self.maxval = self._normalize(self.maxval)
        # Infer unit from minval suffix if not explicitly provided
        if self.pdk_unit is None and isinstance(self.minval, str):
            for suffix in ['u', 'n', 'p', 'm']:
                if self.minval.endswith(suffix):
                    self.pdk_unit = suffix
                    break

    def _normalize(self, value) -> str:
        if isinstance(value, (int, float)):
            return str(value)
        return value

    def validate_value(self, value, param_name: str) -> None:
        """Check *value* against the constraints, warning on violations.

        Checks, in order: a dimensionless parameter (``pdk_unit`` is ``None``,
        for example the multiplier ``m``) must be an integer; the value must
        be at least ``minval`` unless that is ``'-1'``; and at most ``maxval``
        unless that is ``'-1'``.

        Each violation prints a line beginning ``WARNING`` on standard output.
        Nothing is raised, so the simulation can still proceed.

        Parameters
        ----------
        value : float
            The value, already in the PDK's own unit. A value that cannot be
            converted to ``float`` prints a warning and skips all checks.
        param_name : str
            Parameter name, used in the warning text.
        """
        try:
            fval = float(value)
        except (ValueError, TypeError):
            print(f"WARNING: Could not validate '{param_name}={value}' — not numeric")
            return

        unit_str = f" {self.pdk_unit}" if self.pdk_unit else ""

        # Dimensionless parameters (no unit) must be integers
        if self.pdk_unit is None:
            if fval != int(fval):
                print(f"WARNING: '{param_name}' must be an integer, got {value}")

        # Minimum bound
        if self.minval not in ('-1', None):
            try:
                if fval < float(self.minval):
                    print(f"WARNING: '{param_name}={fval}{unit_str}' "
                          f"is below minimum {self.minval}{unit_str}")
            except ValueError:
                pass

        # Maximum bound (-1 means unbounded)
        if self.maxval not in ('-1', None):
            try:
                max_f = float(self.maxval)
                if max_f != -1 and fval > max_f:
                    print(f"WARNING: '{param_name}={fval}{unit_str}' "
                          f"exceeds maximum {self.maxval}{unit_str}")
            except ValueError:
                pass


#: Instance-name prefix for a PDK device whose catalog entry does not declare
#: ``spice_prefix:``. ``X`` makes it a subcircuit call (what every sky130 and
#: most GF180MCU/SG13G2 devices are); the ``M`` is silk's naming convention so
#: MOS instances read as ``XM<name>``. Shared by both emission paths.
DEFAULT_PDK_SPICE_PREFIX = 'XM'


@dataclass
class FaultDimSchema:
    """One dimension (``w`` or ``l``) of a device's ``fault_params:`` block.

    Parsed verbatim from the catalog for ``silk.faults`` (see
    :class:`~silk.pdk.basepdk.FaultParams`). ``kind`` selects how a perturbed
    value is resolved: ``'continuous'`` clamps to ``[min, max]``,
    ``'quantized'`` rounds to a whole number of ``step`` and clamps the count to
    ``[min_count, max_count]`` (``min_count`` defaults to 1), and ``'discrete'``
    snaps to the nearest of ``allowed``. Every bound is in the parameter's own
    PDK unit and is never converted.

    Parameters
    ----------
    kind : {'continuous', 'quantized', 'discrete'}
        How a perturbed value is resolved. Not validated here.
    min : float, optional
        Lower bound of a ``'continuous'`` dimension.
    max : float, optional
        Upper bound of a ``'continuous'`` dimension.
    step : float, optional
        Quantum of a ``'quantized'`` dimension.
    min_count : int, optional
        Smallest count of ``step`` for a ``'quantized'`` dimension.
    max_count : int, optional
        Largest count of ``step`` for a ``'quantized'`` dimension.
    allowed : list of float, optional
        The permitted values of a ``'discrete'`` dimension.
    """
    #: Resolution kind, ``'continuous'``, ``'quantized'`` or ``'discrete'``.
    kind: str
    #: Lower bound (``'continuous'``), in the parameter's PDK unit.
    min: Optional[float] = None
    #: Upper bound (``'continuous'``), in the parameter's PDK unit.
    max: Optional[float] = None
    #: Quantum (``'quantized'``), in the parameter's PDK unit.
    step: Optional[float] = None
    #: Smallest count of ``step`` (``'quantized'``).
    min_count: Optional[int] = None
    #: Largest count of ``step`` (``'quantized'``).
    max_count: Optional[int] = None
    #: Permitted values (``'discrete'``), in the parameter's PDK unit.
    allowed: Optional[List[float]] = None


@dataclass
class FaultParams:
    """A device's ``fault_params:`` schema: W/L perturbation bounds.

    Used by ``silk.faults`` and parsed by
    :meth:`PDKDevice.from_dict <silk.pdk.PDKDevice.from_dict>`. Every bound on
    :attr:`w` and :attr:`l` (``min``/``max``/``step``/``allowed``) is recorded
    in the parameter's own ``pdk_unit``, the same unit as the ``minval`` and
    ``maxval`` of the device's ``w`` and ``l``
    :class:`~silk.pdk.ParameterConstraint` (micrometres for sky130), and is
    never converted here or by any reader of this attribute.
    ``silk.faults`` compares a perturbed value against these bounds directly
    in that PDK-native unit; no author-to-PDK scale factor is carried.

    Parameters
    ----------
    w : FaultDimSchema
        Schema of the width parameter.
    l : FaultDimSchema
        Schema of the length parameter.
    """
    #: Schema of the width parameter, in its PDK unit.
    w: FaultDimSchema
    #: Schema of the length parameter, in its PDK unit.
    l: FaultDimSchema


def _parse_fault_dim(d: Optional[dict]) -> FaultDimSchema:
    d = d or {}
    return FaultDimSchema(
        kind=d.get('kind', 'continuous'),
        min=d.get('min'),
        max=d.get('max'),
        step=d.get('step'),
        min_count=d.get('min_count'),
        max_count=d.get('max_count'),
        allowed=d.get('allowed'),
    )


@dataclass
class JunctionRatingsProvenance:
    """Where a device's ``junction_ratings`` numbers come from.

    Parsed from the catalog's ``junction_ratings_provenance:`` block. A rating
    is a foundry-stated maximum operating voltage and silk adds no margin of
    its own, so every number must be traceable. ``source`` cites the PDK
    document, foundry publication or model-file parameter that states it, and
    covers every rating key not listed under ``heuristic``. ``heuristic`` maps a
    rating key to the rule it was derived by when the foundry publishes no
    number for that pair. The rule text must open with one of the two forms
    that ``silk.pdk.validate`` can re-check against the numbers, ``= <key>_max``
    (equal to another rating of the same device) or ``<factor> x nominal
    <V> V``, followed by free-text justification. The derivation convention is
    written in ``docs/adding-a-pdk.md``.

    Parameters
    ----------
    source : str, optional
        Citation of the document or model parameter that states the ratings.
    heuristic : dict of str to str, optional
        Rating key to the derivation rule, for ratings the foundry does not
        publish.
    """
    #: Citation of the document or model parameter that states the ratings.
    source: Optional[str] = None
    #: Rating key to the rule it was derived by, for ratings not published.
    heuristic: Dict[str, str] = field(default_factory=dict)


def _parse_junction_provenance(d) -> Optional['JunctionRatingsProvenance']:
    """``junction_ratings_provenance:`` block -> dataclass, or ``None`` when absent.

    Deliberately lenient about SHAPE (a string where a mapping was expected is
    kept as-is): the catalog validator (``silk.pdk.validate``) is where a
    malformed block is reported with the device name attached, and raising
    here would take the whole PDK down with it.
    """
    if d is None:
        return None
    if not isinstance(d, dict):
        return JunctionRatingsProvenance(source=d if isinstance(d, str) else None,
                                         heuristic={})
    heur = d.get('heuristic') or {}
    if not isinstance(heur, dict):
        heur = {'__malformed__': str(heur)}
    return JunctionRatingsProvenance(
        source=d.get('source'),
        heuristic={str(k): v for k, v in heur.items()},
    )


# ---------------------------------------------------------------------------
# Junction ratings: catalog value -> signed operating range (TASK-91)
# ---------------------------------------------------------------------------

#: The ONE sign convention for a junction pair named ``Vxy``: the pair voltage
#: is ``V(x) - V(y)`` (``Vbs = V(b) - V(s)``, ``Vgs = V(g) - V(s)``, ``Vak =
#: V(a) - V(k)``, ``Vbe = V(b) - V(e)``), exactly as ``silk.lint.rules.stress``
#: computes it from the terminal waveforms. A signed rating range is read in
#: that convention: sky130's ``VBS = +0.3 to -1.95 V`` for an NMOS is
#: ``{min: -1.95, max: 0.3}`` — the body may sit at most 0.3 V ABOVE the
#: source (forward-biased junction) and at most 1.95 V below it.
JUNCTION_PAIR_SIGN_CONVENTION = 'Vxy = V(x) - V(y)'


def parse_junction_rating(key, raw) -> Tuple[float, float]:
    """Catalog ``junction_ratings`` value -> normalised ``(lo, hi)`` in volts.

    Two forms are accepted (docs/adding-a-pdk.md, "Junction ratings"):

    * a bare number ``m`` — the sign-blind shorthand meaning ``[-m, +m]``,
      the legacy magnitude-only form;
    * a mapping ``{min: lo, max: hi}`` — a signed range in the
      :data:`JUNCTION_PAIR_SIGN_CONVENTION`, for a pair whose published
      operating limits are asymmetric (sky130 NMOS ``Vbs``: ``{min: -1.95,
      max: 0.3}``).

    This is the only place the shorthand is expanded: the catalog loader,
    the validator's fallback for a device that carries magnitudes alone, and
    STR-001's fallback for the same all call it, so the two forms cannot
    drift apart. Raises ``ValueError``/``TypeError`` for a value that is not
    a number or a ``{min, max}`` mapping with numeric entries — the same
    failure a non-numeric magnitude produced before — but deliberately does
    NOT check ``lo < hi`` or finiteness: those are catalog-convention errors
    that ``silk.pdk.validate`` reports with the device name attached
    (``rating_convention``), and raising here would take the whole PDK down.
    """
    if isinstance(raw, dict):
        missing = [k for k in ('min', 'max') if k not in raw]
        extra = sorted(k for k in raw if k not in ('min', 'max'))
        if missing or extra:
            raise ValueError(
                f'junction rating {key!r}: a signed range is {{min: <V>, max: <V>}}'
                f' (missing {missing}, unexpected {extra})')
        return float(raw['min']), float(raw['max'])
    if isinstance(raw, bool):
        raise TypeError(f'junction rating {key!r}: got a bool, expected volts')
    m = float(raw)
    return -m, m


@dataclass
class PDKDevice:
    """One device of a PDK catalog, as the legacy ``add_element`` path sees it.

    Built by :meth:`from_dict` from an entry of the device-details YAML.
    ``pdk.<device>(...)`` factory classes are generated from the same entry;
    this record is what ``BasePDK.generate_netlist_entry`` consumes.

    Parameters
    ----------
    device_name : str
        The catalog key, such as ``'nmos1p8'``.
    pdk_device : str
        The model or subcircuit name written in the netlist, such as
        ``'sky130_fd_pr__nfet_01v8'``.
    param_list : dict of str to ParameterConstraint
        Parameters in netlist order.
    pin_list : dict of str to Pin
        Pins in netlist order, keyed by short name.
    junction_ratings : dict of str to float, optional
        Largest permitted junction voltage magnitude per pair, volts.
    junction_ranges : dict of str to tuple of float, optional
        Signed ``(lo, hi)`` operating range per pair, volts.
    junction_ratings_provenance : JunctionRatingsProvenance, optional
        Where the ratings come from.
    spice_prefix : str, optional
        SPICE instance-name prefix; ``'XM'`` by default.
    fault_params : FaultParams, optional
        Fault-injection W/L schema.
    op_params : tuple of str, optional
        Verified operating-point vector names.
    op_internal_name : str, optional
        Internal instance name used in operating-point vector addresses.
    polarity : {'n', 'p'}, optional
        Channel polarity.
    op_backends : tuple of str, optional
        Simulator backends the operating-point addressing was verified on.

    Examples
    --------
    >>> from silk.pdk import PDKDevice
    >>> entry = {'PDKDevice': 'res_model',
    ...          'paramList': {'w': {'minval': 1, 'maxval': -1, 'pdk_unit': 'u'}},
    ...          'pinList': {'p': 'plus', 'n': 'minus'}}
    >>> device = PDKDevice.from_dict('res1', entry)
    >>> device.spice_prefix, list(device.pin_list)
    ('XM', ['p', 'n'])
    """
    #: The catalog key, such as ``'nmos1p8'``.
    device_name: str
    #: Model or subcircuit name written in the netlist.
    pdk_device: str
    #: Parameters (:class:`~silk.pdk.ParameterConstraint`) in netlist order.
    param_list: Dict[str, ParameterConstraint]
    #: Pins (:class:`~silk.pdk.Pin`) in netlist order, keyed by short name.
    pin_list: Dict[str, Pin]
    #: Largest junction voltage magnitude per pair, ``{'<pair>_max': volts}``.
    #: It is the number itself for a catalog magnitude and ``max(|lo|, |hi|)``
    #: for a signed range (1.95 for the sky130 NMOS ``Vbs`` range
    #: ``{min: -1.95, max: 0.3}``). Consumers that score ``|V|`` read this;
    #: sign-aware ones read :attr:`junction_ranges`.
    junction_ratings: Dict[str, float] = field(default_factory=dict)
    #: Signed operating range per pair, ``{'<pair>_max': (lo, hi)}`` in volts,
    #: in the convention ``Vxy = V(x) - V(y)``. A catalog magnitude ``m``
    #: appears as ``(-m, m)``.
    junction_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    #: Provenance of the ratings
    #: (:class:`~silk.pdk.basepdk.JunctionRatingsProvenance`), or ``None`` when
    #: the catalog declares none (``silk.pdk.validate`` reports that as
    #: ``unsourced_rating`` for a device that carries ratings).
    junction_ratings_provenance: Optional[JunctionRatingsProvenance] = None
    #: SPICE instance-name prefix the netlister writes for this device. It is
    #: the catalog's ``spice_prefix:``, or ``'XM'`` (a subcircuit call) when the
    #: catalog is silent. A ``.model`` primitive declares its own prefix, for
    #: example ``D`` for a diode.
    spice_prefix: str = DEFAULT_PDK_SPICE_PREFIX
    #: This device's ``fault_params:`` schema
    #: (:class:`~silk.pdk.basepdk.FaultParams`), or ``None`` when the catalog
    #: declares none. ``silk.faults`` never substitutes another device's schema.
    fault_params: Optional[FaultParams] = None
    #: Names of the operating-point ('@' vector) parameters this model is
    #: verified to expose, a subset of the recognised vocabulary. ``None``
    #: means undeclared: no '@' saves are made for the device. A device
    #: declares ``op_params``, :attr:`op_internal_name` and :attr:`op_backends`
    #: together; if any of the three is missing, all three are ``None``.
    op_params: Optional[Tuple[str, ...]] = None
    #: Literal internal instance name the simulator needs before ``[<param>]``
    #: in this model's '@' vector address, for example
    #: ``'msky130_fd_pr__nfet_01v8'``. It cannot be derived from
    #: :attr:`pdk_device`. ``None`` when undeclared.
    op_internal_name: Optional[str] = None
    #: Channel polarity, ``'n'`` or ``'p'``, declared per device because it
    #: cannot be inferred from the model name. ``None`` when undeclared; it is
    #: never guessed.
    polarity: Optional[str] = None
    #: Simulator backends on which this device's operating-point addressing was
    #: verified by a real run. Paired with :attr:`op_params`; ``None`` when
    #: undeclared.
    op_backends: Optional[Tuple[str, ...]] = None

    _emission_spec: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.param_list = self.param_list

    def __setattr__(self, name, value):
        from silk.elements._pdk_emission import parameter_values, refuse_derived_inputs
        spec = vars(self).get('_emission_spec', {})
        refuse_derived_inputs((name,), spec, vars(self).get('device_name', 'device'))
        if name == 'param_list':
            value = parameter_values(value, spec)
        object.__setattr__(self, name, value)

    @classmethod
    def from_dict(cls, name: str, data: dict) -> 'PDKDevice':
        """Create a PDKDevice from one entry of the standard silk YAML schema.

        ``spice_prefix`` is read here with the same default the factory path
        applies, so the ``add_element`` path and the class path cannot
        disagree about how a device is emitted.

        Parameters
        ----------
        name : str
            The catalog key of the device.
        data : dict
            The parsed entry. ``PDKDevice``, ``paramList`` (each parameter with
            ``minval`` and ``maxval``, and optionally ``default``,
            ``pdk_unit`` and ``quantity``) and ``pinList`` are required.
            ``junction_ratings``, ``junction_ratings_provenance``,
            ``spice_prefix``, ``fault_params``, ``op_params``,
            ``op_internal_name``, ``op_backends`` and ``polarity`` are optional.

        Returns
        -------
        PDKDevice
            The parsed device. A parameter whose ``default`` is missing or
            null takes its ``minval`` as default; an explicit non-null
            ``default`` (including ``0``) is kept. The three
            ``op_*`` declarations are kept only if all three are present.

        Raises
        ------
        KeyError
            A required key is missing. ``BasePDK`` reports this as
            :class:`~silk.pdk.basepdk.ValidationError` while loading a catalog.
        silk.pdk.basepdk.ValidationError
            ``op_params`` names a parameter outside the recognised vocabulary,
            ``op_backends`` names an unregistered backend, or ``polarity`` is
            not ``'n'`` or ``'p'``.
        ValueError
            A ``junction_ratings`` entry is neither a number nor a
            ``{min, max}`` mapping of numbers.
        """
        from silk.elements._pdk_emission import validate_emission_spec
        try:
            emission_spec = validate_emission_spec(name, data)
        except (ValueError, TypeError) as exc:
            raise ValidationError(str(exc)) from exc
        param_list = {}
        for param_name, constraints in data['paramList'].items():
            param_list[param_name] = ParameterConstraint(
                minval=str(constraints['minval']),
                maxval=str(constraints['maxval']),
                pdk_unit=constraints.get('pdk_unit'),
                quantity=constraints.get('quantity'),
                # Same expression as the factory path: an explicit default (a numeric
                # zero counts), or minval as the fallback when absent. Kept literally identical so the two
                # cannot drift apart again.
                default=(constraints['default'] if constraints.get('default') is not None
                         else constraints.get('minval')),
                value=None,
                emit_if_supplied=constraints.get('emit_if_supplied', False),
            )
        pin_list = {}
        for pin_short, pin_full in data['pinList'].items():
            pin_list[pin_short] = Pin(name=pin_short, net_name=pin_full, net=None)

        junction_ranges = {
            k: parse_junction_rating(k, v)
            for k, v in (data.get('junction_ratings') or {}).items()
        }
        junction_ratings = {
            k: max(abs(lo), abs(hi)) for k, (lo, hi) in junction_ranges.items()
        }
        junction_ratings_provenance = _parse_junction_provenance(
            data.get('junction_ratings_provenance'))

        fault_params = None
        if 'fault_params' in data:
            fp = data['fault_params'] or {}
            fault_params = FaultParams(
                w=_parse_fault_dim(fp.get('w')),
                l=_parse_fault_dim(fp.get('l')),
            )

        op_params = _parse_op_params(name, data.get('op_params'))
        op_internal_name = data.get('op_internal_name')
        if op_internal_name is not None:
            op_internal_name = str(op_internal_name)
        # Paired declaration (TASK-330.3): one without the other is not
        # enough information to build a working '@' address, so it is
        # undeclared either way rather than half-trusted.
        op_backends = _parse_op_backends(name, data.get('op_backends'))
        # Triple-paired declaration (TASK-330.5, extends TASK-330.3's pair):
        # any one of the three missing leaves the device fully undeclared --
        # half of a working address plus a backend list nobody can use (or
        # vice versa) is not enough to capture anything.
        if not op_params or not op_internal_name or not op_backends:
            op_params = None
            op_internal_name = None
            op_backends = None
        polarity = _parse_polarity(name, data.get('polarity'))

        return cls(
            device_name=name,
            pdk_device=data['PDKDevice'],
            param_list=param_list,
            _emission_spec=emission_spec,
            pin_list=pin_list,
            junction_ratings=junction_ratings,
            junction_ranges=junction_ranges,
            junction_ratings_provenance=junction_ratings_provenance,
            spice_prefix=str(data.get('spice_prefix', DEFAULT_PDK_SPICE_PREFIX)),
            fault_params=fault_params,
            op_params=op_params,
            op_internal_name=op_internal_name,
            polarity=polarity,
            op_backends=op_backends,
        )


def _parse_op_params(device_name: str, raw) -> Optional[Tuple[str, ...]]:
    """Validate a catalog ``op_params:`` list against the fixed vocabulary
    (:data:`silk.simulation.backend.DEVICE_OP_PARAMS`) at PDK-load time, so a
    typo is a load-time :class:`ValidationError` naming the device and the
    bad name, never a silent no-op or a later ngspice abort with no
    diagnosis."""
    if not raw:
        return None
    from silk.simulation.backend import DEVICE_OP_PARAMS
    names = tuple(raw)
    unknown = [n for n in names if n not in DEVICE_OP_PARAMS]
    if unknown:
        raise ValidationError(
            f"device {device_name!r}: op_params names {unknown} are not in "
            f"the recognised op-point parameter vocabulary "
            f"{list(DEVICE_OP_PARAMS)}."
        )
    return names


def _parse_op_backends(device_name: str, raw) -> Optional[Tuple[str, ...]]:
    """Validate a catalog ``op_backends:`` list against the registered
    simulator backend names at PDK-load time -- same reasoning
    as :func:`_parse_op_params`: a typo here is a load-time
    :class:`ValidationError`, never a silently-wrong "which backend can
    read this device" answer discovered mid-capture."""
    if not raw:
        return None
    from silk.simulation.backend import SimulatorBackend
    valid = sorted(SimulatorBackend._registry)
    names = tuple(raw)
    unknown = [n for n in names if n not in valid]
    if unknown:
        raise ValidationError(
            f"device {device_name!r}: op_backends names {unknown} are not "
            f"recognised simulator backends {valid}."
        )
    return names


def _parse_polarity(device_name: str, raw) -> Optional[str]:
    """Validate a catalog ``polarity:`` value ('n'/'p') at PDK-load time -- same reasoning as :func:`_parse_op_params`: a typo here
    is a load-time :class:`ValidationError`, never a silently-wrong table
    column later."""
    if raw is None:
        return None
    polarity = str(raw).lower()
    if polarity not in ('n', 'p'):
        raise ValidationError(
            f"device {device_name!r}: polarity {raw!r} must be 'n' or 'p'."
        )
    return polarity


# ---------------------------------------------------------------------------
# Device geometry in ONE canonical unit — the accessor every geometry
# consumer (silk.layout first) reads instead of interpreting units itself
# ---------------------------------------------------------------------------

#: Micrometres are silk's canonical LENGTH unit for everything downstream of
#: the PDK — layout estimation, parasitics, EM sizing. The factor from a PDK's
#: own prefix to µm is derived from the same table emission uses, so there is
#: no second unit vocabulary anywhere in silk.
_CANONICAL_LENGTH_PREFIX = 'u'


def _pdk_unit_to_um(pdk_unit, param_name, owner):
    """Factor taking a LENGTH in the catalog prefix *pdk_unit* to micrometres.

    ``'u'`` → 1, ``'base'`` (metres) → 1e6, ``'n'`` → 1e-3. An untagged or
    unknown prefix is a catalog defect for a geometry parameter and is
    reported, never guessed: before this accessor existed the layout model
    assumed "untagged means µm", which is exactly the kind of silent
    reinterpretation that produced a 1e12 area error.
    """
    if pdk_unit not in _SI_TO_PDK_SCALE:
        raise ValidationError(
            f"{owner}: geometry parameter {param_name!r} has "
            f"{'no' if pdk_unit is None else 'unknown'} pdk_unit"
            f"{'' if pdk_unit is None else ' ' + repr(pdk_unit)} in the device "
            f"catalog; expected one of {sorted(_SI_TO_PDK_SCALE)}. silk will not "
            f"guess the unit of a length.")
    return _SI_TO_PDK_SCALE[_CANONICAL_LENGTH_PREFIX] / _SI_TO_PDK_SCALE[pdk_unit]


def _numeric_or_none(raw):
    """Unwrap a Variable and return ``(value, is_literal)``.

    ``value`` is a float, or None when the parameter is unassigned or is a
    deck token silk cannot evaluate (``'1u'``, a parameter reference).
    ``is_literal`` is True for ANY string: emission writes a string verbatim
    and never unit-scales it, so a plain-decimal string such as the
    ``'1e-05'`` PyYAML hands back for a dot-less catalog default is a number
    in the PDK's OWN unit — read it that way here too, so the two paths
    agree, but never apply the author→PDK factor to it.
    """
    if hasattr(raw, 'value'):          # Variable duck-type
        raw = raw.value
    if raw is None or isinstance(raw, bool):
        return None, False
    if isinstance(raw, str):
        try:
            return float(raw), True
        except ValueError:
            return None, True
    try:
        return float(raw), False
    except (TypeError, ValueError):
        return None, False


def _lookup(mapping, *names):
    """First (name, entry) present in *mapping* among *names*, else (None, None)."""
    for n in names:
        if n in mapping:
            return n, mapping[n]
    return None, None


def device_geometry_um(device, pdk=None):
    """Return ``(W_um, L_um, M)`` for a device, or ``None`` if it has no W/L.

    The single accessor for device geometry. Every consumer that needs a
    device's channel geometry (``silk.layout`` area and wire models, and
    anything after them) reads it here, in canonical micrometres, and never
    interprets ``pdk_unit:`` or ``user_units`` itself. The conversion is the
    same two steps netlist emission performs:

    1. author's unit → the PDK's own unit — the per-parameter factor
       ``BasePDK._param_scales`` / ``BasePDK._scale_factor`` build,
       applied ONLY to caller-supplied values (a catalog ``default:`` is
       already in the PDK's unit and never scales);
    2. the PDK's unit → micrometres, from the parameter's ``pdk_unit:`` tag
       via ``_SI_TO_PDK_SCALE`` (sky130 ``u`` → ×1; GF180MCU/SG13G2
       ``base`` → ×1e6).

    Two device representations are understood:

    - a factory-built :class:`~silk.elements.base.AnalogElement`
      (``pdk.nmos1p8(...)`` inside ``with mod:``): its class carries
      ``_param_scales`` and ``_param_constraints``, so *pdk* is not needed;
    - a :class:`PDKDevice` copy (``add_element(pdk.nmos1p8, ...)``): the
      author→PDK factor lives on the PDK instance, so *pdk* is REQUIRED.

    ``M`` defaults to 1 only when the device declares NO multiplier
    parameter at all — an absent ``m``/``M`` IS 1 in SPICE, so that is not a
    guess.

    Returns ``None`` when the device declares no W/L at all (passives, ideal
    elements); when W or L is not a number silk can judge (unassigned
    ``Variable``, or a string — a SPICE literal / parameter reference);
    or when the device DOES declare a multiplier but that multiplier is not
    a number silk can judge (same unassigned-``Variable``-or-string
    case). A declared-but-unresolved ``m``/``M`` is never silently treated
    as 1: the device has no judgeable footprint until the design space is
    pinned, exactly the reasoning that makes an unresolved W/L return
    ``None`` rather than guess.

    Parameters
    ----------
    device : AnalogElement or PDKDevice
        A factory-built device or an ``add_element``-style copy.
    pdk : BasePDK, optional
        The PDK the device was added from. Required when *device* is a
        :class:`~silk.pdk.PDKDevice`; ignored for a factory-built device.

    Returns
    -------
    tuple of float or None
        ``(W_um, L_um, M)``: width and length in micrometres and the
        multiplier, or ``None`` when no judgeable geometry exists (see above).

    Raises
    ------
    silk.pdk.basepdk.ValidationError
        A W or L is declared but its catalog entry carries no recognised
        ``pdk_unit:``. A length without a unit is a catalog defect; silk does
        not guess.
    TypeError
        *device* is neither representation, or a ``PDKDevice`` was passed
        without its *pdk*.

    Examples
    --------
    >>> from silk.pdk import load_pdk, device_geometry_um
    >>> pdk = load_pdk('sky130')
    >>> device = pdk.nmos1p8('m1', d='o', g='i', s='gnd', b='gnd', w=4.0, l=0.5)
    >>> device_geometry_um(device)
    (4.0, 0.5, 1.0)
    """
    from silk.circuit.device_view import _inspection_device
    device = _inspection_device(device)
    from silk.elements.base import AnalogElement

    if isinstance(device, AnalogElement):
        cls = type(device)
        constraints = getattr(cls, '_param_constraints', None) or {}
        scales = getattr(cls, '_param_scales', None) or {}
        supplied = getattr(device, '_caller_supplied', None) or ()
        params = getattr(device, 'params', None) or {}
        owner = f'{cls.__name__} {getattr(device, "name", "?")}'

        def _length_um(*names):
            p, raw = _lookup(params, *names)
            if p is None:
                return None, False
            val, literal = _numeric_or_none(raw)
            if val is None:
                return None, True
            if p in supplied and not literal:
                val *= scales.get(p, 1.0)          # author unit → PDK unit
            c = constraints.get(p)
            unit = c.pdk_unit if c is not None else None
            return val * _pdk_unit_to_um(unit, p, owner), True

        W_um, has_w = _length_um('w', 'W')
        L_um, has_l = _length_um('l', 'L')
        if not (has_w and has_l):
            return None                               # no geometry declared
        if W_um is None or L_um is None:
            return None                               # literal / unassigned
        _p, m_raw = _lookup(params, 'm', 'M')
        if _p is None:
            M = 1.0                                   # no multiplier declared — SPICE default
        else:
            M = _numeric_or_none(m_raw)[0]
            if M is None:
                return None                           # declared but unresolved — same rule as W/L
        return W_um, L_um, M

    if isinstance(device, PDKDevice):
        if pdk is None:
            raise TypeError(
                f'device_geometry_um({device.device_name}): a PDKDevice '
                f'(add_element style) carries no unit scale of its own — pass '
                f'the PDK it was added from (module.pdk_details).')
        pl = device.param_list
        owner = f'{device.device_name} ({device.pdk_device})'

        def _length_um(*names):
            p, c = _lookup(pl, *names)
            if p is None:
                return None, False
            raw = c.value
            if raw is None:                           # caller omitted it
                raw = c.default
                scale = 1.0                           # a default is PDK-unit
            else:
                scale = pdk._scale_factor(c.pdk_unit, c.quantity)
            val, literal = _numeric_or_none(raw)
            if val is None:
                return None, True
            if literal:
                scale = 1.0                           # emitted verbatim
            return val * scale * _pdk_unit_to_um(c.pdk_unit, p, owner), True

        W_um, has_w = _length_um('w', 'W')
        L_um, has_l = _length_um('l', 'L')
        if not (has_w and has_l):
            return None
        if W_um is None or L_um is None:
            return None
        _p, mc = _lookup(pl, 'm', 'M')
        if mc is None:
            M = 1.0                                   # no multiplier declared — SPICE default
        else:
            M = _numeric_or_none(mc.value if mc.value is not None else mc.default)[0]
            if M is None:
                return None                           # declared but unresolved — same rule as W/L
        return W_um, L_um, M

    raise TypeError(
        f'device_geometry_um: {type(device).__name__} is neither a factory '
        f'AnalogElement nor a PDKDevice; silk cannot read its geometry without '
        f'guessing a unit.')


# ---------------------------------------------------------------------------
# Base PDK class
# ---------------------------------------------------------------------------

class BasePDK:
    """Base class for all silk PDK implementations.

    Provides standard YAML parsing, unit mode handling, parameter validation,
    and SPICE netlist entry generation. Subclasses implement PDK-specific
    behaviour by overriding:

    - ``initializeDeviceList()``  — for non-standard source formats

    Corner-specific model files are declared in the manifest (a corner's
    ``includes:`` list), not by subclassing.

    Normally a PDK is obtained from :func:`~silk.pdk.load_pdk`, which reads
    the manifest and passes these arguments. Each device of the catalog is then
    an attribute: ``pdk.nmos1p8(...)`` is a generated class whose constructor
    takes the instance name positionally and pins and parameters as keywords
    (see the factory page of the reference). Constructing the class directly
    with ``user_units=UNIT_PDK`` (its default) prints a notice to standard
    error. The constructor does not check that ``library_location`` exists;
    a simulation that needs missing models fails later with a
    ``SimulationError``.

    Parameters
    ----------
    model_list : str
        Path to the device-details YAML catalog.
    library_location : str
        Path to the model library file the netlist includes.
    user_units : str, optional
        The unit the circuit author writes in: :py:data:`~silk.pdk.UNIT_PDK`
        (default), :py:data:`~silk.pdk.UNIT_SI`, or a prefix token (``'u'``,
        ``'n'``, ``'p'``, ``'m'``, ``'base'``).
    debug : bool, optional
        Print the parsed catalog and netlist lines while working.
    corners : dict, optional
        Corner name to metadata (``kind``, ``desc`` and optionally
        ``includes``). Empty by default.
    default_corner : str, optional
        Corner used when none is requested. Not checked against *corners*.
    extra_includes : list of str, optional
        Extra model files to include after the corner library line.
    pdk_name : str, optional
        Canonical PDK name from the manifest.
    osdi_files : list of str, optional
        Compiled Verilog-A model files the device models require. silk loads
        them at run time and does not compile them.
    xyce_library_location : str, optional
        Xyce-native counterpart of *library_location*; ``None`` when the
        manifest declares none.
    xyce_extra_includes : list of str, optional
        Xyce-native counterpart of *extra_includes*; ``None`` when the manifest
        declares none.
    statistical : object, optional
        Monte Carlo facts parsed from the manifest (a ``StatisticalModels``);
        ``None`` when the PDK declares none.

    Attributes
    ----------
    model_names : list of str
        Names of the catalog devices, in catalog order.
    model_list : str
        Path to the device-details YAML.
    library_location : str
        Path to the model library file.
    user_units : str
        The author's unit mode, as passed.
    corners : dict
        Corner name to its manifest metadata.
    default_corner : str
        The corner used when none is requested.
    extra_includes : list of str
        Extra model files included after the corner library line.
    osdi_files : list of str
        Compiled Verilog-A model files the device models require.
    xyce_library_location : str or None
        Xyce-native library path, if declared.
    xyce_extra_includes : list of str or None
        Xyce-native extra includes, if declared.
    statistical : object or None
        Monte Carlo facts, if declared.
    pdk_name : str or None
        Canonical PDK name.
    debug : bool
        Whether verbose output is on.

    Raises
    ------
    silk.pdk.basepdk.ValidationError
        The catalog YAML does not parse or lacks a required field.
    silk.exceptions.PDKError
        A device is defined twice in the catalog, or a device, pin or
        parameter name is not a valid Python identifier.
    OSError
        *model_list* cannot be opened.

    Examples
    --------
    >>> from silk.pdk import load_pdk
    >>> pdk = load_pdk('sky130')
    >>> pdk.model_names[:3]
    ['nmos1p8', 'nmos1p8_lvt', 'nmos3p3_native']
    >>> pdk.default_corner
    'tt'
    """

    def __init__(self, model_list, library_location, user_units=UNIT_PDK,
                 debug=False, *, corners=None, default_corner='tt',
                 extra_includes=None, pdk_name=None, osdi_files=None,
                 xyce_library_location=None, xyce_extra_includes=None,
                 statistical=None):
        self.model_list = model_list
        self.library_location = library_location
        self.user_units = user_units
        self.debug = debug
        self.corners = corners or {}
        self.default_corner = default_corner
        self.extra_includes = list(extra_includes or [])
        # Optional Xyce-native twins of library_location/extra_includes (and,
        # per corner, of `includes:` — see that key's `xyce_includes`
        # sibling). Generic and manifest-driven: any PDK may declare them (see
        # silk/pdk/_loader.py); XyceNetlister prefers them when present and
        # falls back to the ngspice-oriented fields above otherwise, so a PDK
        # that declares none of this is untouched, byte for byte.
        self.xyce_library_location = xyce_library_location
        self.xyce_extra_includes = (
            list(xyce_extra_includes) if xyce_extra_includes is not None else None)
        # Compiled Verilog-A models (.osdi) this PDK's device models require —
        # resolved absolute paths from the manifest's `library.osdi:` list.
        # silk does NOT compile these (that is environment setup, like
        # installing ngspice itself); it loads them at run time via `pre_osdi`
        # lines in the emitted control block, and fails loudly at netlist time
        # if they are declared but not compiled.
        self.osdi_files = list(osdi_files or [])
        # StatisticalModels from the manifest's `library.statistical:` block,
        # or None when the PDK declares no Monte Carlo facts.
        self.statistical = statistical
        self.pdk_name = pdk_name
        self.model_names = []
        self._devices = {}
        self.count_devices_used = 0
        self.count_devices_instantiated = 0
        self.devices_used_list = []
        self.modelCount = 0

        self.initializeDeviceList()

        if user_units == UNIT_PDK:
            print(f"User units: PDK defaults — values are taken in each parameter's "
                  f"own pdk_unit (e.g. µm for W/L in sky130). Pass user_units='u' to "
                  f"write microns, or use load_pdk(), whose manifest sets user_units.",
                  file=sys.stderr)

    def __getstate__(self):
        """Drop the per-device class objects before pickling.

        ``initializeDeviceList()`` calls ``make_device_class()`` once per
        YAML device on every ``load_pdk()`` call, so two loads of the same
        PDK (the normal case for a circuit composed from blocks built in
        separate modules) hold two different class objects for the same
        device name — ``self._device_classes`` and the per-device instance
        attributes it also sets (``self.nmos1p8``, ...) can each carry
        EITHER one, depending on which load produced this particular
        ``BasePDK`` instance. Pickling a class object directly demands the
        exact object back by module+qualname (the same identity check
        :func:`silk.elements.factory._factory_element_reduce` works around
        for device *instances*), so shipping these to a worker raises
        ``PicklingError: ... it's not the same object as ...`` regardless
        of that fix. Nothing reads them after a module is built — building
        devices from ``pdk.nmos1p8(...)`` happens before a run is ever
        dispatched to a worker, and ``generate_netlist()``/``simulate()``
        only need this PDK's metadata (``model_list``, ``library_location``,
        ``corners``, ...) — so they are simply left out of the pickled
        state rather than reconstructed.
        """
        state = self.__dict__.copy()
        for device_name in state.get('_device_classes', {}):
            state.pop(device_name, None)
        state['_device_classes'] = {}
        return state

    # ------------------------------------------------------------------
    # Device loading
    # ------------------------------------------------------------------

    def initializeDeviceList(self):
        """Load device definitions from the PDK source file.

        Called once by the constructor. The default implementation parses the
        standard silk YAML schema from ``model_list``: it fills ``model_names``
        and the device records, and creates one device class per entry,
        reachable as ``pdk.<device_name>``. Override it to support another
        source format: populate ``self._devices`` with :class:`PDKDevice`
        objects and call ``setattr(self, device_name, device)`` for each one.

        Raises
        ------
        silk.pdk.basepdk.ValidationError
            The YAML does not parse, or an entry lacks a required field.
        silk.exceptions.PDKError
            A device is defined twice in the YAML, or a device, pin or
            parameter name is not a valid Python identifier.
        OSError
            ``model_list`` cannot be opened.
        """
        self._parse_standard_yaml(self.model_list)

    def _parse_standard_yaml(self, yaml_path: str) -> None:
        """Parse the standard silk YAML device schema.

        Populates ``self._devices``, ``self.model_names``, and creates a named
        attribute on the PDK instance for each device (e.g. ``pdk.nmos1p8``).
        Also populates ``self._device_classes`` with dynamically-created
        :class:`AnalogElement` subclasses (used by the new instantiation API).
        """
        from silk.elements.factory import make_device_class
        from silk.elements._catalog import _validate_device_names

        self._device_classes = {}

        with open(yaml_path, 'r') as f:
            try:
                # _StrictSafeLoader raises on duplicate mapping keys, so a
                # PDK YAML with two entries for the same device name fails
                # loudly instead of silently using the second definition.
                config_data = yaml.load(f, Loader=_StrictSafeLoader)
                if self.debug:
                    print(config_data)
                for device_name, device_data in config_data.items():
                    _validate_device_names(device_name, device_data, yaml_path, PDKError)
                    device = PDKDevice.from_dict(device_name, device_data)
                    self._devices[device_name] = device   # internal: keep PDKDevice
                    cls = make_device_class(
                        device_name, device_data, is_pdk=True,
                        param_scales=self._param_scales(device_data))
                    self._device_classes[device_name] = cls
                    setattr(self, device_name, cls)       # public: expose the class
            except yaml.constructor.ConstructorError as e:
                # Duplicate-key error from _StrictSafeLoader is raised here.
                if 'duplicate key' in str(e):
                    raise PDKError(
                        f'Error: device defined multiple times in YAML: {e}'
                    ) from e
                raise ValidationError(f"Error parsing YAML: {e}") from e
            except yaml.YAMLError as e:
                raise ValidationError(f"Error parsing YAML: {e}") from e
            except KeyError as e:
                raise ValidationError(f"Missing required field in YAML: {e}") from e

        self.model_names = list(self._devices.keys())

        if self.debug:
            print(f'PDK loaded: {len(self.model_names)} devices '
                  f'({", ".join(self.model_names)})')

    # ------------------------------------------------------------------
    # Device geometry (canonical micrometres)
    # ------------------------------------------------------------------

    def device_geometry_um(self, device):
        """Return ``(W_um, L_um, M)`` for *device*, or ``None`` if it has no W/L.

        The method form of :func:`device_geometry_um` bound to this PDK, for
        an ``add_element``-style :class:`PDKDevice` whose author-to-PDK factor
        depends on this instance's ``user_units``.

        Parameters
        ----------
        device : AnalogElement or PDKDevice
            The device to measure.

        Returns
        -------
        tuple of float or None
            Width and length in micrometres and the multiplier, or ``None``.

        Raises
        ------
        silk.pdk.basepdk.ValidationError
            A W or L has no recognised ``pdk_unit`` in the catalog.
        TypeError
            *device* is neither a factory-built device nor a ``PDKDevice``.
        """
        return device_geometry_um(device, self)

    @property
    def length_unit_in_metres(self):
        """Metres per unit of this catalog's length parameters, or ``None``.

        ``1e-6`` for sky130 (``pdk_unit: u`` — its models take µm), ``1.0``
        for GF180MCU and SG13G2 (``base`` — their models take metres),
        ``None`` for a catalog that declares no length at all (nothing to
        rescale). This is the number a simulator needs when it reads the
        deck's bare W/L as metres and must be told otherwise —
        ``silk.circuit.xyce_netlister.XyceNetlister`` emits it as
        ``.options parser scale=<value>`` and omits the line at 1.0.

        Derived from the catalog, never declared twice: every length-tagged
        parameter is read and they must all agree, because one deck-wide
        scale cannot serve two prefixes. A catalog that mixes them raises
        :class:`~silk.pdk.basepdk.ValidationError`.
        Area parameters are not consulted; SPICE applies ``scale`` squared to
        an area, so a catalog tagging its areas ``p`` next to ``u`` lengths
        (sky130) is consistent by construction.

        """
        prefixes = set()
        for device in getattr(self, '_devices', {}).values():
            for constraint in device.param_list.values():
                if constraint.pdk_unit not in _SI_TO_PDK_SCALE:
                    continue      # dimensionless (m, nf, ng) or untagged
                if (constraint.quantity or 'length') != 'length':
                    continue      # areas follow the length prefix squared
                prefixes.add(constraint.pdk_unit)
        if not prefixes:
            return None
        if len(prefixes) > 1:
            raise ValidationError(
                f"the {getattr(self, 'name', 'loaded')!s} device catalog tags "
                f"its lengths with more than one prefix {sorted(prefixes)}; a "
                f"simulator-wide length scale cannot serve two. Tag every "
                f"length in the catalog with the same pdk_unit.")
        return 1.0 / _SI_TO_PDK_SCALE[prefixes.pop()]

    # ------------------------------------------------------------------
    # Netlist generation
    # ------------------------------------------------------------------

    def _param_scales(self, device_data):
        """Per-parameter factor taking a CALLER's value to the PDK's number.

        Both sides name a prefix from the same vocabulary, so the factor is
        just ``scale[what the PDK wants] / scale[what the author writes]``:

        =================  ==========  ==========  ==============
        author writes      PDK wants   factor      ``w = 1`` emits
        =================  ==========  ==========  ==============
        ``u`` (µm)         ``u`` (µm)  1           ``1``
        ``u`` (µm)         ``base``    1e-6        ``1u``  (= 1e-6 m)
        ``si`` (metres)    ``u`` (µm)  1e6         ``1e6``
        =================  ==========  ==========  ==============

        So ``user_units='u'`` lets one circuit source write ``w=1`` for one
        micron and have every PDK receive the number its own models expect —
        which is the point. ``user_units='pdk'`` means the author already writes
        the PDK's unit, so nothing scales and decks are byte-identical to
        before scaling existed.

        Parameters with no ``pdk_unit:`` are dimensionless (``m``, ``nf``, ``ng``)
        and never scale.
        """
        if self.user_units == UNIT_PDK:
            return {}
        scales = {}
        for pname, c in (device_data.get('paramList') or {}).items():
            # Every parameter that declares a convertible unit gets an entry,
            # identity factors included. An empty map must keep meaning "this
            # PDK does not convert", not "we never saw the parameters".
            if not isinstance(c, dict) or c.get('pdk_unit') not in _SI_TO_PDK_SCALE:
                continue
            scales[pname] = self._scale_factor(c.get('pdk_unit'), c.get('quantity'))
        return scales

    def _scale_factor(self, unit, quantity=None) -> float:
        """Factor taking ONE caller-supplied value from the author's unit to
        the PDK's own, for a parameter declaring *unit* and *quantity*.

        The single source of truth for unit conversion. Both emission paths
        call it — the factory path via :meth:`_param_scales` (per class, at
        class-creation time) and the legacy ``add_element`` path per parameter
        in :meth:`generate_netlist_entry`. They diverged once: the legacy path
        carried its own ``user_units == UNIT_SI`` test, which no prefix mode
        satisfies, so under the shipped default (``user_units: u``) it converted
        nothing and emitted ``w=1.0`` — one METRE — into GF180MCU's
        metre-expecting models, while the factory path emitted the correct
        ``w=1u`` for the same circuit.

        Returns 1.0 for a dimensionless parameter (``m``, ``ng``, ``nf``), for
        an unrecognised unit, and for ``user_units='pdk'``.
        """
        if self.user_units == UNIT_PDK or unit not in _SI_TO_PDK_SCALE:
            return 1.0
        # The author's own prefix. 'si' is spelled out as 'base' — writing SI
        # base units IS writing the unprefixed unit.
        author = 1.0 if self.user_units == UNIT_SI \
            else _SI_TO_PDK_SCALE[self.user_units]
        try:
            exponent = _QUANTITY_EXPONENT[quantity or 'length']
        except KeyError:
            raise ValidationError(
                f"unknown quantity {quantity!r} in the device catalog; "
                f"expected one of {sorted(_QUANTITY_EXPONENT)}") from None
        return _SI_TO_PDK_SCALE[unit] / (author ** exponent)

    def generate_netlist_entry(self, name, details):
        """Generate the SPICE instance line for a device.

        Converts each caller-supplied value from the author's unit to the
        PDK's own (the same factor the factory path applies), checks it with
        :meth:`ParameterConstraint.validate_value` (violations print a warning
        and are written anyway), and writes the line. A parameter the caller
        left unset is written with its catalog default, unscaled. String values
        pass through as SPICE literals.

        Parameters
        ----------
        name : str
            Instance name, appended to the device's ``spice_prefix``.
        details : dict
            The module's element entry; ``details['device']`` is the
            :class:`PDKDevice` with its pins connected and parameters set.

        Returns
        -------
        str
            One line ``<prefix><name> <nets> <model> <param>=<value> ...``
            ending in a newline.

        Raises
        ------
        TypeError
            A pin is unconnected or holds something that is not a net name.
        ValueError
            A pin holds a string that is not a valid net token.
        silk.pdk.basepdk.ValidationError
            A parameter's ``quantity`` in the catalog is unknown.
        """
        # The catalog-declared prefix (PDKDevice.spice_prefix), never a literal:
        # a `.model` primitive emits e.g. 'D<name>', a subckt device 'XM<name>'.
        # Tokens are joined with single spaces at the end — the factory path's
        # spelling, and the canonical one (TASK-73). This path used to append a
        # trailing space after every token, so the same device built the two
        # ways differed by one byte at end of line.
        device = details['device']
        from silk.elements._pdk_emission import refuse_derived_inputs, validate_required_parameters
        refuse_derived_inputs(set(device.param_list) | vars(device).keys(),
                              device._emission_spec, name)
        validate_required_parameters(device.param_list, device._emission_spec, name)
        # Final structural guard, the twin of AnalogElement.to_spice's: exactly
        # one node token per pin, in pin order. add_element and connect_element
        # already reject anything else, so this only fires when pin_list was
        # mutated directly (TASK-75 AC #3). The rule is the shared chokepoint,
        # so the message matches the factory path's for the same bad value.
        type_name = getattr(device, 'device_name', '') or type(device).__name__
        pins = [validate_net_token(
                    device.pin_list[p].net if device.pin_list[p].net is not None
                    else _PIN_UNSET,
                    who=pin_context(type_name, name, p), kind='pin')
                for p in device.pin_list]

        params = []
        for p, constraint in device.param_list.items():
            if self.debug:
                print(p)

            raw = constraint.value
            if raw is None and constraint.emit_if_supplied and not constraint.emission_set:
                continue
            from silk.elements._pdk_emission import validate_strict
            validate_strict(constraint.default if raw is None else raw, p,
                            device._emission_spec.get('paramList', {}).get(p, {}), name)
            if raw is None:
                # Caller omitted it. Fill from the catalog, and do NOT scale:
                # a `default:` is already in the PDK's own unit and is a fixed
                # physical geometry (see _scale_factor / the factory path).
                params.append((p, constraint.default, 1.0))
                continue
            param_val = raw.value if hasattr(raw, 'value') else raw

            # Author's unit -> the PDK's own. A string passes through: it is
            # already a SPICE literal ('1u', or a parameter reference).
            scale = self._scale_factor(constraint.pdk_unit, constraint.quantity)
            if scale != 1.0 and not isinstance(param_val, str):
                try:
                    param_val = float(param_val) * scale
                except (ValueError, TypeError):
                    pass  # symbolic/Variable — skip conversion

            # Validate in PDK units
            constraint.validate_value(param_val, p)
            params.append((p, param_val, 1.0))

        # The one writer, shared with AnalogElement.to_spice (TASK-416).
        str_netlist = render_pdk_device_line(
            device.spice_prefix, name, pins, device.pdk_device, params,
            derived_params=device._emission_spec.get('derivedParams'))
        if self.debug:
            print(str_netlist, end='')

        return str_netlist
