"""silk.circuit.net — first-class Net and PortBinding objects.

A ``Net`` represents a named electrical node in a :class:`~silk.circuit.modules.Module`.
Phase 2a promotes per-net state onto this class directly: ``voltage``
(behavioral waveform), ``domain`` (VoltageDomain assignment),
``classification`` (PortRole — a property with auto-inference from
bound pin roles plus explicit override), and ``is_critical`` (a
designer-facing layout annotation flag). The Phase 1 read-through
voltage property has been replaced with a stored field; nothing on
``Module`` proxies to ``_behavioral_results`` anymore — that dict has
been removed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np

if TYPE_CHECKING:
    from silk.circuit.modules import Module
    from silk.circuit.circuit_elements import PortRole
    from silk.circuit.voltage_domain import VoltageDomain


@dataclass
class PortBinding:
    """One pin's contribution to a net.

    Used by ``Net.pins`` and by interface/domain checkers. The ``kind``
    field identifies which kind of owner this binding came from:
    ``'submodule'``, ``'device'``, ``'passive'``, or ``'source'``. Checkers
    that currently only reason about submodule ports filter by ``kind``.

    Parameters
    ----------
    net : str
        Name of the net the pin is bound to.
    instance : str
        Name of the instance (submodule, device, passive or source) that
        owns the pin.
    port : str
        Pin or port name on that instance.
    direction : str, optional
        :class:`~silk.circuit.PortDir` value as a string, or ``None``.
    role : str, optional
        :class:`~silk.circuit.PortRole` value as a string, or ``None``.
    domain : VoltageDomain, optional
        Voltage domain of the pin, or ``None``.
    kind : {'submodule', 'device', 'passive', 'source'}, optional
        Kind of owner the binding came from.
    role_inferred : bool, default False
        True when ``role`` came from ``infer_port_roles`` rather than a
        declaration (submodule pins only).
    avg_current : float, optional
        Signed average current in amperes; positive flows from the device
        into the net. Set by ``Module.annotate_currents()``; ``None`` until
        annotated.
    peak_current : float, optional
        Maximum ``|i(t)|`` over the simulated window, amperes. Set by
        ``annotate_currents()``.
    rms_current : float, optional
        RMS of ``|i(t)|`` over the window, amperes. Set by
        ``annotate_currents()``.
    current_vector : str, optional
        Name of the rawfile vector the currents were read from.
    _current_source : str, optional
        Internal: where the currents came from (``'spice'``, ``'user'`` or
        ``'db'``).
    """
    net: str
    instance: str
    port: str
    direction: Optional[str] = None   # PortDir value (string) or None
    role: Optional[str] = None        # PortRole value (string) or None
    domain: Optional[Any] = None      # VoltageDomain or None
    kind: Optional[str] = None        # 'submodule' | 'device' | 'passive' | 'source' | None
    role_inferred: bool = False       # role came from infer_port_roles (submodule only; TASK-42)
    # --- simulated per-pin currents ---
    # Populated by Module.annotate_currents() from a transient SPICE run.
    avg_current: Optional[float] = None       # signed A; + = device → net
    peak_current: Optional[float] = None      # A, max |i(t)| over the window
    rms_current: Optional[float] = None       # A, RMS |i(t)| over the window
    current_vector: Optional[str] = None      # rawfile vector this came from
    _current_source: Optional[str] = None     # 'spice' | 'user' | 'db' | None


@dataclass
class Net:
    """A named electrical node in a :class:`~silk.circuit.modules.Module`.

    Attributes
    ----------
    name : str
        Net name (e.g. ``'vout'``).
    module : Module
        Parent module this net belongs to. Back-reference so checkers and
        layout code can get to module-level context from a net.
    parameters : list
        Legacy parameter list from :meth:`Module.add_net` (rarely used).
    pins : list of PortBinding
        Pins currently bound to this net. Maintained via push updates by
        ``PyCircuitElements._bind_pin_to_net``.
    voltage : np.ndarray, optional
        Behavioral simulation waveform for this net, populated by
        :meth:`Module.simulate_behavioral`. ``None`` before simulation.
    domain : VoltageDomain, optional
        Voltage-domain assignment for this net, populated by
        :meth:`Module.assign_net_domain`. ``None`` when unassigned.
    is_critical : bool
        Designer flag marking this net as critical for layout (short
        routing, priority placement). Phase 2a just stores it; layout
        consumers land later.
    estimated_length : float, optional
        Statistical wirelength in um, set by ``estimate_layout()``.
    estimated_width : float, optional
        Widest wire required on the net, in um, set by ``estimate_layout()``.
    estimated_r : float, optional
        Largest pin-to-pin resistance in ohms (directionless), set by
        ``estimate_layout()``.
    estimated_c : float, optional
        Total capacitance in fF, set by ``estimate_layout()``.
    peak_current : float, optional
        Conservative per-segment peak current in A: half the sum of the
        pins' peak currents, valid for any routing tree. Set by
        ``Module.annotate_currents()``; ``None`` unless every pin of a
        non-interface net was annotated.
    rms_current : float, optional
        Conservative per-segment RMS current in A (half the sum of the
        pins' RMS currents); same population rule as ``peak_current``.
    avg_current : float, optional
        Conservative per-segment DC current in A (half the sum of the
        pins' absolute averages); same population rule as ``peak_current``.
    max_ir_drop_mV : float, optional
        IR-drop budget in mV (supply nets), read by the layout checkers.
    max_j_factor : float, optional
        Fraction of the layer's maximum current density allowed, in (0, 1]
        (unitless).
    max_length_um : float, optional
        Routed-length budget in um.
    width : int
        Number of bits: 1 for a scalar net, more for a bus handle.
    _classification_override : PortRole, optional
        Internal: explicit classification set through ``classification``.
    _current_source : str, optional
        Internal: where the net's currents came from (``'spice'``,
        ``'user'`` or ``'db'``).
    _index_vector : tuple, optional
        Internal: for a bus handle, the ``(name, bit)`` pairs it spans.
    _underlying_pins : tuple, optional
        Internal: for a bus handle, the scalar nets it aggregates.
    """
    name: str
    module: 'Module'
    parameters: List[Any] = field(default_factory=list)
    pins: List[PortBinding] = field(default_factory=list)

    # Phase 2a stored state
    voltage: Optional[np.ndarray] = None
    domain: Optional['VoltageDomain'] = None
    is_critical: bool = False
    _classification_override: Optional['PortRole'] = None

    # Phase 1 layout fields — populated by estimate_layout()
    estimated_length: Optional[float] = None   # um  — statistical wirelength
    estimated_width:  Optional[float] = None   # um  — max width required across net
    estimated_r:      Optional[float] = None   # ohms — max pin-to-pin R, directionless
    estimated_c:      Optional[float] = None   # fF   — total capacitance

    # Conservative worst-segment aggregates, refreshed from the pins by
    # Module.annotate_currents(). Each is an UPPER BOUND
    # on any routed segment's current, valid for EVERY possible routing tree of
    # this net: for a cut splitting the pins into sides S and S̄ the segment
    # current is Σ_S i_p (= −Σ_S̄ i_p by KCL), hence bounded by ½·Σ_all|i_p|.
    # Read by estimate_layout()/silk_adapter for EM wire sizing. For a 2-pin net
    # the bound is exact.
    #
    # The ½·Σ bound is KCL-derived and only conservative over the COMPLETE pin
    # set of a net, so aggregates are populated ONLY for fully-annotated,
    # non-interface nets. Nets with zero annotated pins AND partially-annotated
    # nets (a skipped pin, or a subckt port net whose boundary current is not a
    # pin) keep None — never 0.0 or a half-bound, which silk_adapter would
    # forward as a real (under-sized) EM hint. Per-pin currents on the annotated
    # pins are still set; only the net-level aggregate is withheld.
    peak_current:     Optional[float] = None   # A  — ½·Σ_p peak_p (pointwise bound)
    rms_current:      Optional[float] = None   # A  — ½·Σ_p rms_p (Minkowski bound)
    avg_current:      Optional[float] = None   # A  — ½·Σ_p |avg_p| (signed-avg/DC bound)
    _current_source:  Optional[str]   = None   # 'spice' | 'user' | 'db' | None

    # Layout budgets — designer intent consumed by EM/IR/length checkers.
    # Moved here from silk.constraints (was MaxIRDrop / MaxCurrentDensity /
    # MaxNetLength) because the net is the natural owner of its own
    # properties; see silk/constraints/DESIGN.md Q5.
    max_ir_drop_mV:   Optional[float] = None   # mV  — IR-drop budget (supply nets)
    max_j_factor:     Optional[float] = None   # —   — fraction of layer J_max (0,1]
    max_length_um:    Optional[float] = None   # um  — routed-length budget

    # Bus-handle support. For scalar Nets these are the
    # default values; bus handles are constructed via _make_bus_handle.
    width: int = 1
    _index_vector: Optional[tuple] = None        # tuple[tuple[str, int], ...] | None
    _underlying_pins: Optional[tuple] = None     # tuple[Net, ...] | None  (frozen attribute name)

    def __setattr__(self, key, value):
        if key == 'name':
            from silk.elements.base import validate_net_token
            value = validate_net_token(value, who='Net.name', kind='net')
        object.__setattr__(self, key, value)

    @property
    def fanout(self) -> int:
        """Number of connected pins."""
        return len(self.pins)

    @property
    def classification(self) -> Optional['PortRole']:
        """The net's classification — either the explicit override, or
        inferred from the roles of bound pins.

        Inference priority (first non-None wins):
            POWER > GROUND > CLOCK > DIGITAL > ANALOG

        Returns ``None`` when no override is set and no bound pin has a
        role. Setting to ``None`` clears the override and reverts to
        inference on the next read.
        """
        if self._classification_override is not None:
            return self._classification_override
        return self._infer_classification_from_pins()

    @classification.setter
    def classification(self, value: Optional['PortRole']) -> None:
        self._classification_override = value

    def _infer_classification_from_pins(self) -> Optional['PortRole']:
        """Walk bound pins and pick the first role in priority order."""
        from silk.circuit.circuit_elements import PortRole
        pin_roles = {b.role for b in self.pins if b.role is not None}
        for role in (
            PortRole.POWER, PortRole.GROUND,
            PortRole.CLOCK, PortRole.DIGITAL, PortRole.ANALOG,
        ):
            if role in pin_roles:
                return role
        return None

    @classmethod
    def _make_bus_handle(cls, name: str, underlying: tuple, index_vector: tuple) -> 'Net':
        """Construct a bus-handle Net referencing already-existing scalar Nets.

        Used by Module.add_net(name, bits=...) and by slicing operations.
        The handle does NOT own its own pin storage — it aggregates over
        ``underlying``, which must be a tuple of scalar Nets that already
        live in the module's net_dict or are declared port bits that will
        enter that dict when first wired or explicitly added.
        """
        handle = cls.__new__(cls)
        handle.name = name
        # Initialize all the same per-instance fields scalar Net would have.
        # Bus handles are aggregates and do not have their own per-bit data;
        # per-bit data lives on each underlying scalar Net.
        handle.module = None
        handle.parameters = []
        handle.pins = []
        handle.voltage = None
        handle.domain = None
        handle.is_critical = False
        handle._classification_override = None
        handle.estimated_length = None
        handle.estimated_width = None
        handle.estimated_r = None
        handle.estimated_c = None
        handle.peak_current = None
        handle.rms_current = None
        handle.avg_current = None
        handle._current_source = None
        handle.max_ir_drop_mV = None
        handle.max_j_factor = None
        handle.max_length_um = None
        # Bus-specific:
        handle.width = len(underlying)
        handle._underlying_pins = tuple(underlying)
        handle._index_vector = tuple(index_vector)
        return handle

    def __len__(self) -> int:
        """Return the width of this net (1 for scalar, N for bus-handle)."""
        return self.width

    def __iter__(self):
        """Iterate over underlying scalar Nets in order. Raises TypeError for scalar nets."""
        if self._underlying_pins is None:
            raise TypeError(
                f"Net {self.name!r} is scalar (width=1); iteration is not supported."
            )
        return iter(self._underlying_pins)

    def __getitem__(self, key):
        """Index or slice this net. int returns the underlying scalar Net; slice returns
        a new bus-handle Net with sliced indices. Raises TypeError for scalar nets."""
        if self._underlying_pins is None:
            raise TypeError(
                f"Net {self.name!r} is scalar (width=1); subscript is not supported."
            )
        from silk.circuit.declared_indices import select_positions
        positions = select_positions(tuple(k for _, k in self._index_vector), key)
        from numbers import Integral
        if isinstance(key, Integral):
            return self._underlying_pins[positions[0]]
        return type(self)._make_bus_handle(
            name=self.name, underlying=tuple(self._underlying_pins[p] for p in positions),
            index_vector=tuple(self._index_vector[p] for p in positions),
        )
