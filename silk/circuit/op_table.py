"""silk.circuit.op_table — SpecReport.op_table(): per-device operating-point
rows.

Built entirely from the plain-data inventory
:func:`~silk.circuit.modules_results.capture_op_device_records` captures at
evaluate() time (a report keeps no ``Module``
reference and never re-reads a rawfile; ``op_table()`` applies selected
region policy to already-captured floats). No pandas dependency; rows are
frozen dataclasses, never a numpy structured array (the per-column source
marker is per-cell metadata a structured array would need parallel shadow
arrays to carry).
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Union
import difflib
import math

from silk.simulation.backend import DEVICE_OP_PARAMS
from silk.circuit.region_thresholds import (
    RegionThresholdsError, resolve_policy, shipped_default,
)

#: The closed set of region strings _classify_region ever returns (besides
#: None) — OpTable.filter(region=...)'s own typo guard validates against
#: this: a caller error (an unknown region name) raises rather
#: than silently returning an empty table, mirroring the name-lookup KeyError
#: below.
_KNOWN_REGIONS = frozenset(('cutoff', 'subthreshold', 'triode', 'saturation'))

#: Physical constants for the default subthreshold margin (3*kT/q).
_BOLTZMANN_J_PER_K = 1.380649e-23
_ELECTRON_CHARGE_C = 1.602176634e-19
#: Fallback ambient when a report carries no scoped temperature (unscoped /
#: historical / hand-built reports).
_DEFAULT_TEMPERATURE_C = 27.0

#: Raw column order mirrors DEVICE_OP_PARAMS exactly — all 17
#: are exposed.
RAW_COLUMNS: Tuple[str, ...] = DEVICE_OP_PARAMS
DERIVED_COLUMNS: Tuple[str, ...] = ('vov', 'gm_id', 'gm_gds', 'ro', 'ft', 'vds_margin')


def _kt_over_q(temperature_c: float) -> float:
    t_kelvin = temperature_c + 273.15
    return _BOLTZMANN_J_PER_K * t_kelvin / _ELECTRON_CHARGE_C


def _policy_margin(policy, operating_conditions):
    if 'subthreshold_margin_v' in policy:
        return policy['subthreshold_margin_v']
    if getattr(operating_conditions, 'temperature_kind', None) == 'none':
        raise ValueError(
            'The evidence temperature is none, so kT/q is undefined; '
            'pass region_thresholds= explicitly to op_table().')
    temperature_c = getattr(operating_conditions, 'temperature_c', None)
    if temperature_c is None:
        temperature_c = _DEFAULT_TEMPERATURE_C
    return policy['subthreshold_margin_kt_q'] * _kt_over_q(temperature_c)


@dataclass(frozen=True)
class RegionThresholds:
    """Explicit operating-region thresholds, in amps and volts.

    Omitted constructor fields use the shipped ``default`` policy at 27 C.
    :meth:`at_temperature` resolves the selected user/per-PDK policy and
    derives a thermal margin from the report's captured ambient instead.
    An explicit object passed to a report overrides the selected file.

    Parameters
    ----------
    subthreshold_margin_v : float, optional
        Margin in V below ``vth`` at which ``vgs`` counts as cut off rather than
        subthreshold. Default: the shipped policy's margin at 27 C.
    min_current_a : float, optional
        Current in A; with ``vgs < vth``, a drain current below this is cutoff.
    weak_sat_margin_v : float, optional
        Margin in V; a saturated device with ``0 < vds - vdsat`` below this
        is flagged as weakly saturated.
    """
    subthreshold_margin_v: float = field(
        default_factory=lambda: _policy_margin(shipped_default(), None))
    min_current_a: float = field(default_factory=lambda: shipped_default()['min_current_a'])
    weak_sat_margin_v: float = field(default_factory=lambda: shipped_default()['weak_sat_margin_v'])

    @classmethod
    def at_temperature(cls, operating_conditions, *, pdk_name=None) -> 'RegionThresholds':
        """Resolve policy and apply its thermal multiplier at the run ambient.

        Unknown/unscoped temperature falls back to 27 C. Explicitly absent
        temperature refuses a thermal margin; a fixed-volts policy works.
        Missing PDK identity selects the data's ``default`` entry.

        Parameters
        ----------
        operating_conditions : object
            A report's captured conditions; only ``temperature_kind`` and
            ``temperature_c`` (degrees Celsius) are read. ``None`` means an
            unscoped run (27 C).
        pdk_name : str, optional
            PDK whose region-threshold policy to select.

        Returns
        -------
        RegionThresholds
            Thresholds in volts and amperes, with the thermal margin
            evaluated at the run ambient.

        Raises
        ------
        ValueError
            If the policy needs a thermal margin and the evidence temperature
            is explicitly absent, or the selected policy data is invalid
            (a ``RegionThresholdsError``, which is a ``ValueError``).
        """
        policy = resolve_policy(pdk_name)
        return cls(subthreshold_margin_v=_policy_margin(policy, operating_conditions),
                   min_current_a=policy['min_current_a'],
                   weak_sat_margin_v=policy['weak_sat_margin_v'])


def _classify_region(vgs, vth, vds, vdsat, id_, thresholds: RegionThresholds):
    """Region classification applied identically to both polarities.

    ngspice reports PMOS vgs/vds/id as
    positive source-referenced magnitudes on every declared PDK, sky130 and
    gf180mcu alike, so no sign correction is needed here.

    ``cutoff`` iff ``vgs < vth - subthreshold_margin_v``, OR (``vgs < vth``
    AND ``abs(id_) < min_current_a``). Low current alone does not imply cutoff,
    which mislabelled a fully-ON switch/pass device carrying no current
    because its load is off (measured: sky130 xst2.XMMP, vgs=1.64 > vth=0.74,
    vds ≈ 0, id=1.6e-11 — "cutoff" under the old rule, wrong for a device
    whose gate is driving it on). A device with ``vgs >= vth`` is now NEVER
    cutoff on current alone: it is ``triode`` when ``vds < vdsat``, else
    ``saturation`` — low/no current there is a real (if unusual) operating
    point, not an indication the device is off.

    Returns ``(region, saturation_weak)``, both ``None`` when any of the
    five inputs is unavailable (``None``) or non-finite (``nan``/``inf`` —
    every comparison against ``nan`` is False, so it would otherwise fall
    through to ``'saturation'``) — never a default bucket on
    incomplete data.
    """
    if any(v is None or not math.isfinite(v) for v in (vgs, vth, vds, vdsat, id_)):
        return None, None
    if vgs < vth - thresholds.subthreshold_margin_v:
        return 'cutoff', False
    if vgs < vth:
        if abs(id_) < thresholds.min_current_a:
            return 'cutoff', False
        return 'subthreshold', False
    # vgs >= vth: never cutoff on current alone.
    if vds < vdsat:
        return 'triode', False
    weak = 0.0 < (vds - vdsat) < thresholds.weak_sat_margin_v
    return 'saturation', weak


@dataclass(frozen=True)
class OpRow:
    """One MOS device's operating-point row.

    ``name`` is the designer's OWN hierarchical instance name
    — dot-joined from the exact strings passed to
    ``add_module``/``add_element``/``add``, e.g. ``'dut.st1.MN'`` — never the
    emitted SPICE prefixes (``'x'`` per submodule level, the device's own
    ``spice_prefix``/``'XM'``/... at the leaf). ``None`` on a row built from
    an ``op_devices`` inventory captured without designer names (no ``'name'``
    key in the record) — ``OpTable.__getitem__``'s string form refuses
    to look up on such a table (see its docstring).

    Both identity fields are rooted at the testbench: a DUT device has
    name 'dut.st1.MN' and path 'xdut.xst1.XMMN'; a helper placed as
    'st1' has name 'st1.MN' and path 'xst1.XMMN'. Historical captures
    retain their original labels. Names and paths are separate fields.

    ``sources`` carries a marker per raw/derived column name (never for
    ``name``/``path``/``model``/``polarity``/``w_um``/``l_um``/``m``, which
    are already directly nullable):

    - ``'sim'`` — a real :meth:`silk.simulation.backend.SpiceBackend.device_op_vectors`
      reading: the parameter IS declared for this device's model.
    - ``'derived'`` — computed here from ``'sim'`` inputs.
    - ``'derived_estimate'`` — ``ft`` only (``gm/(2*pi*cgg)`` is a
      small-signal estimate, not a directly measured transit frequency).
    - ``None`` — unavailable: the model does not declare this raw
      parameter, or a derived column's own inputs were unavailable.

    A column with source ``None`` is ``nan``, never a silently-omitted key
    — an undeclared device's row is all-``None``/all-``nan`` beyond
    ``path`` and ``model``.

    Parameters
    ----------
    path : str
        Emitted path from the testbench root, e.g. ``'xdut.xst1.XMMN'``.
    model : str or None
        Model name of the device, or ``None``.
    polarity : str or None
        ``'n'`` or ``'p'`` as captured, or ``None``.
    w_um : float or None
        Channel width in um (the PDK-native unit), or ``None``.
    l_um : float or None
        Channel length in um, or ``None``.
    m : float or None
        Multiplier (parallel device count), or ``None``.
    id : float
        Drain current in A.
    vgs : float
        Gate-source voltage in V (source-referenced magnitude for both
        polarities).
    vds : float
        Drain-source voltage in V.
    vbs : float
        Bulk-source voltage in V.
    vth : float
        Threshold voltage in V.
    vdsat : float
        Saturation voltage in V.
    gm : float
        Transconductance in S.
    gds : float
        Output conductance in S.
    gmbs : float
        Bulk transconductance in S.
    cgg : float
        Total gate capacitance in F.
    cgs : float
        Gate-source capacitance in F.
    cgd : float
        Gate-drain capacitance in F.
    cgso : float
        Gate-source overlap capacitance in F.
    cgdo : float
        Gate-drain overlap capacitance in F.
    cgbo : float
        Gate-bulk overlap capacitance in F.
    capbd : float
        Bulk-drain junction capacitance in F.
    capbs : float
        Bulk-source junction capacitance in F.
    vov : float
        Overdrive ``vgs - vth`` in V; ``nan`` when an input is unavailable.
    gm_id : float
        ``gm / id`` in 1/V; ``nan`` when ``id`` is zero or unavailable.
    gm_gds : float
        Intrinsic gain ``gm / gds`` (unitless, V/V); ``nan`` when ``gds`` is
        zero or unavailable.
    ro : float
        Output resistance ``1 / gds`` in ohms; ``nan`` when ``gds`` is zero or
        unavailable.
    ft : float
        Estimated transit frequency ``gm / (2 pi cgg)`` in Hz; a
        small-signal estimate, ``nan`` when unavailable.
    vds_margin : float
        ``vds - vdsat`` in V; ``nan`` when an input is unavailable.
    region : str or None
        ``'cutoff'``, ``'subthreshold'``, ``'triode'`` or ``'saturation'``;
        ``None`` when any classification input is unavailable or not finite.
    saturation_weak : bool or None
        True when saturated within ``weak_sat_margin_v`` of ``vdsat``;
        ``None`` where ``region`` is ``None``.
    sources : dict of str to str or None
        Per-column provenance: ``'sim'``, ``'derived'``, ``'derived_estimate'``
        or ``None`` (unavailable).
    name : str, optional
        The designer's hierarchical instance name, e.g. ``'st1.MN'``;
        ``None`` for a record captured before names existed.
    """
    path: str
    model: Optional[str]
    polarity: Optional[str]
    w_um: Optional[float]
    l_um: Optional[float]
    m: Optional[float]

    id: float
    vgs: float
    vds: float
    vbs: float
    vth: float
    vdsat: float
    gm: float
    gds: float
    gmbs: float
    cgg: float
    cgs: float
    cgd: float
    cgso: float
    cgdo: float
    cgbo: float
    capbd: float
    capbs: float

    vov: float
    gm_id: float
    gm_gds: float
    ro: float
    ft: float
    vds_margin: float

    region: Optional[str]
    saturation_weak: Optional[bool]

    sources: Dict[str, Optional[str]]

    #: Designer's hierarchical name. Defaulted to ``None`` (and
    #: placed last, after every non-default field a frozen dataclass
    #: requires that ordering for) so an old-style record with no ``'name'``
    #: key still builds a valid row — the rendering placeholder ('—') falls
    #: out of the existing generic ``value is None`` handling in
    #: ``_cell_text``, no special case needed there.
    name: Optional[str] = None


#: ``name`` leads (console/Markdown/HTML renderings show
#: the designer's own name first); ``path`` (the emitted SPICE path) stays
#: right after it, so nothing is lost, just demoted to a secondary column.
_COLUMNS: Tuple[str, ...] = (
    ('name', 'path', 'model', 'polarity', 'w_um', 'l_um', 'm')
    + RAW_COLUMNS + DERIVED_COLUMNS + ('region',)
)


def _cell_text(row: OpRow, col: str) -> str:
    value = getattr(row, col)
    if col == 'region':
        # saturation_weak lives on its own OpRow field (a real bool a
        # caller can branch on), but rendering folds it into the 'region'
        # cell text -- reviewer review of b7639171: it was on OpRow but
        # never shown in __str__/to_markdown, so a weakly-saturated device
        # (measured: sky130 xst1.XMMN, vds - vdsat = 31 mV) looked
        # identical to a comfortably-saturated one at a glance.
        if value == 'saturation' and row.saturation_weak:
            return 'saturation (weak)'
    if value is None:
        return '—'
    if isinstance(value, float):
        if math.isnan(value):
            # A HOLE (this column's source is None -- the simulator/backend
            # never supplied it, e.g. gmbs/the capacitances on Xyce) is a
            # DIFFERENT fact from "a real reading that happens to compute to
            # nan" (source 'sim'/'derived'/'derived_estimate' -- e.g. ro is
            # nan when a genuinely-read gds is exactly 0.0). Only a hole
            # gets the placeholder; the sources dict is what distinguishes
            # them (TASK-330.6, reviewer ruling on AC #5: printing the bare
            # string 'nan' for every nan alike, hole or not, was misleading
            # on every surface this renders to -- console, Markdown, and
            # HTML -- not only HTML, so the fix belongs here, once, rather
            # than only in the HTML renderer).
            return '—' if row.sources.get(col) is None else 'nan'
        return f'{value:.6g}'
    return str(value)


@dataclass(frozen=True)
class OpTable:
    """The full per-device operating-point table — ``report.op_table()``'s
    return value. Iterable over its ``OpRow`` rows;
    ``__str__``/:meth:`to_markdown` render a console/Markdown table.

    ``table[0]`` indexes positionally in depth-first bench insertion order;
    a helper placed before the DUT appears first.
    ``table['dut.st1.MN']`` looks a DUT row up by ``OpRow.name``
    instead — exact-match, case-sensitive, same as silk's own per-scope
    instance-name uniqueness (``'MN'``/``'mn'`` are different names, never
    conflated). An unknown name raises :class:`KeyError` listing the
    closest known names (``difflib``); a table built from an ``op_devices``
    inventory captured without designer names (every row's ``name`` is ``None``
    — no rows were ever indexed) raises explaining the report predates
    designer names, rather than a bare miss with an empty suggestion list.

    Parameters
    ----------
    rows : tuple of OpRow
        One row per MOS device in the DUT and helpers, in depth-first
        testbench insertion order.
    region_thresholds : RegionThresholds
        The thresholds the rows' ``region`` was classified with.

    Raises
    ------
    ValueError
        If two rows share the same designer ``name``.
    """
    rows: Tuple[OpRow, ...]
    region_thresholds: RegionThresholds

    def __post_init__(self):
        index: Dict[str, OpRow] = {}
        for row in self.rows:
            if row.name is None:
                continue
            if row.name in index:
                raise ValueError(
                    f"OpTable: two rows share the designer name {row.name!r} "
                    f"({index[row.name].path!r} and {row.path!r}) -- names "
                    f"must be unique; this indicates a construction bug, "
                    f"not a real circuit (add_module/add_element already "
                    f"refuses a duplicate name within one scope).")
            index[row.name] = row
        object.__setattr__(self, '_name_index', index)

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if not isinstance(key, str):
            return self.rows[key]
        if not self._name_index:
            raise KeyError(
                f"{key!r}: this op_table has no device names to look up -- "
                f"its op_devices inventory was captured before designer "
                f"names existed (TASK-330.7). Index by position instead, "
                f"e.g. table[0].")
        try:
            return self._name_index[key]
        except KeyError:
            close = difflib.get_close_matches(key, self._name_index, n=3, cutoff=0.6)
            candidates = close or sorted(self._name_index)
            label = 'close matches' if close else 'known names'
            raise KeyError(
                f"{key!r}: no device with this name in op_table "
                f"({label}: {', '.join(candidates)}).") from None

    def filter(self, predicate: Optional[Callable[[OpRow], bool]] = None, *,
              region: Optional[Union[str, Tuple[str, ...]]] = None,
              saturation_weak: Optional[bool] = None) -> 'OpTable':
        """A new :class:`OpTable` (same ``region_thresholds``) holding only
        the rows that match. Still iterable/renderable/name-indexable, so
        it composes: ``table.filter(region='cutoff').to_markdown()``.

        Two ways to ask: the everyday shortcuts
        ``region=`` (a region string, or an iterable of them -- validated
        against the known set, so a typo raises rather than silently
        returning zero rows) and/or ``saturation_weak=`` (exact match on
        that field -- already implies ``region == 'saturation'``, since
        that flag is only ever set there); or a raw ``predicate=`` callable
        for anything else, e.g. the OR case ("cutoff or weakly saturated")
        neither shortcut alone expresses:
        ``table.filter(lambda r: r.region == 'cutoff' or r.saturation_weak)``.
        ``predicate=`` is mutually exclusive with the two shortcuts (raises
        ``TypeError`` if combined -- an ambiguous AND/ignore choice
        otherwise).
        """
        if predicate is not None and (region is not None or saturation_weak is not None):
            raise TypeError(
                "OpTable.filter(): pass either predicate=, or region=/"
                "saturation_weak=, not both.")
        if predicate is None:
            regions = None
            if region is not None:
                regions = {region} if isinstance(region, str) else set(region)
                unknown = regions - _KNOWN_REGIONS
                if unknown:
                    raise ValueError(
                        f"OpTable.filter(): unknown region(s) "
                        f"{sorted(unknown)!r}; known regions are "
                        f"{sorted(_KNOWN_REGIONS)!r}.")

            def predicate(row: OpRow) -> bool:
                if regions is not None and row.region not in regions:
                    return False
                if saturation_weak is not None and row.saturation_weak != saturation_weak:
                    return False
                return True

        return OpTable(rows=tuple(row for row in self.rows if predicate(row)),
                       region_thresholds=self.region_thresholds)

    def __str__(self) -> str:
        if not self.rows:
            return '(no MOS devices)'
        lines = [' '.join(_COLUMNS)]
        lines += [' '.join(_cell_text(row, c) for c in _COLUMNS) for row in self.rows]
        return '\n'.join(lines)

    def to_markdown(self) -> str:
        """Render as a GitHub-flavored Markdown table.

        Returns
        -------
        str
            One header row, a separator and one row per device; unavailable
            cells show an em dash. ``'_(no MOS devices)_'`` for an empty
            table.
        """
        if not self.rows:
            return '_(no MOS devices)_'
        out = ['| ' + ' | '.join(_COLUMNS) + ' |',
               '| ' + ' | '.join('---' for _ in _COLUMNS) + ' |']
        out += ['| ' + ' | '.join(_cell_text(row, c) for c in _COLUMNS) + ' |'
                for row in self.rows]
        return '\n'.join(out)


def _sub(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return float('nan')
    return a - b


def _div(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None or b == 0:
        return float('nan')
    return a / b


def build_op_table(records, region_thresholds, operating_conditions, *, pdk_name=None) -> OpTable:
    """Build an :class:`OpTable` from
    :func:`~silk.circuit.modules_results.capture_op_device_records`'s
    plain-data records (a report's stored ``_op_devices``).
    Uses already-captured floats plus selected policy data: never touches
    a ``Module`` or a rawfile."""
    thresholds = (region_thresholds if region_thresholds is not None
                  else RegionThresholds.at_temperature(operating_conditions, pdk_name=pdk_name))
    rows = []
    for record in records:
        declared = set(record.get('declared_params') or ())
        values = record.get('op_values') or {}
        raw: Dict[str, float] = {}
        sources: Dict[str, Optional[str]] = {}
        for param in RAW_COLUMNS:
            raw[param] = values.get(param, float('nan'))
            sources[param] = 'sim' if param in declared else None

        def sim(name: str) -> Optional[float]:
            return raw[name] if sources[name] == 'sim' else None

        vgs, vth, vds, vdsat = sim('vgs'), sim('vth'), sim('vds'), sim('vdsat')
        id_, gm, gds, cgg = sim('id'), sim('gm'), sim('gds'), sim('cgg')

        # Source markers are INPUT-based ('derived' iff every input this
        # column needs is itself 'sim'), never output-based: a genuine
        # mathematical degeneracy (e.g. gds == 0, so ro is nan) must not be
        # mistaken for "the backend never declared this parameter" —
        # those are different facts, and only the marker distinguishes them.
        vov = _sub(vgs, vth)
        sources['vov'] = 'derived' if (vgs is not None and vth is not None) else None
        gm_id = _div(gm, id_)
        sources['gm_id'] = 'derived' if (gm is not None and id_ is not None) else None
        gm_gds = _div(gm, gds)
        sources['gm_gds'] = 'derived' if (gm is not None and gds is not None) else None
        ro = _div(1.0, gds) if gds is not None else float('nan')
        sources['ro'] = 'derived' if gds is not None else None
        ft = _div(gm, 2 * math.pi * cgg) if (gm is not None and cgg is not None) else float('nan')
        sources['ft'] = 'derived_estimate' if (gm is not None and cgg is not None) else None
        vds_margin = _sub(vds, vdsat)
        sources['vds_margin'] = 'derived' if (vds is not None and vdsat is not None) else None

        region, weak = _classify_region(vgs, vth, vds, vdsat, id_, thresholds)

        rows.append(OpRow(
            path=record.get('path'),
            name=record.get('name'),          # None on a pre-330.7 record
            model=record.get('model'),
            polarity=record.get('polarity'),
            w_um=record.get('w_um'),
            l_um=record.get('l_um'),
            m=record.get('m'),
            **raw,
            vov=vov, gm_id=gm_id, gm_gds=gm_gds, ro=ro, ft=ft, vds_margin=vds_margin,
            region=region, saturation_weak=weak,
            sources=sources,
        ))
    return OpTable(rows=tuple(rows), region_thresholds=thresholds)
