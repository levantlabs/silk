"""The Module circuit container, combining construction, simulation and reporting methods."""

# Module class - UPDATED VERSION
# This includes the devices in the module
# It should also allow for defined verification tasks, constraints, etc.

import os
import copy
from typing import Any, Dict, List, Optional, Tuple, Union
from silk.circuit.circuit_elements import CircuitGraph, ElementType
# ModuleTestbench is imported lazily in create_test() to avoid circular imports
from silk.exceptions import CircuitError, DuplicateNameError, UndeclaredNetError, SimulationError, PDKError



# Shared, importable definitions live in _module_base.py
# so the per-subsystem "part" files can import them without a circular
# dependency on this module. Re-exported here so existing imports
# (silk.circuit.modules._active_module, ...) keep working and
# the _active context singleton stays canonical.
from silk.circuit._module_base import (  # noqa: F401
    _active,
    _active_module,
    _push_active,
    _pop_active,
    _NetAccessor,
    ModulePin,
    SubcircuitInstance,
)

# Per-subsystem "part" mixins. Each holds a disjoint group
# of Module's methods, split into its own file for navigability. They carry no
# state — Module inherits them and every ``self`` is the Module instance.
from silk.circuit.modules_faults import _FaultMethods
from silk.circuit.modules_db import _DbMethods
from silk.circuit.modules_testbench import _TestbenchMethods
from silk.circuit.modules_netlist import _NetlistMethods
from silk.circuit.modules_results import _ResultsMethods
from silk.circuit.modules_sim import _SimMethods
from silk.circuit.modules_spec import _SpecMethods
from silk.circuit.modules_structure import _StructureMethods, _validate_module_name


def _reconstruct_module(cls):
    """Pickle helper used by :meth:`Module.__reduce_ex__` — creates a
    bare instance (bypassing ``__init__``) that ``__setstate__`` fills
    in."""
    return cls.__new__(cls)


class Module(_StructureMethods, _FaultMethods, _DbMethods, _TestbenchMethods,
             _NetlistMethods, _ResultsMethods, _SimMethods, _SpecMethods):

    """A circuit module containing elements, nets and registered testbenches.

    A top-level module owns an analysis and simulation results; a module
    with ``module_name`` set is emitted as a ``.subckt`` block and owns no
    analysis. The constructor creates ``dir_folder`` if it is missing.

    Parameters
    ----------
    pdk_name : str
        Name of the PDK this module is built on (for example ``'sky130'``).
    dir_folder : str
        Output directory for netlists and simulator output; created, with
        any missing parents, when absent.
    netlistname : str
        Netlist file name inside ``dir_folder``, stored unchanged. Put folders
        in ``dir_folder``; a slash in this name is refused at deck preparation.
    pdk_details : BasePDK
        The loaded PDK object, as returned by :func:`silk.pdk.load_pdk`.
    debug : bool, optional
        Print diagnostics while elements are added. Default False.
    strict_nets : bool, optional
        Require every net to be declared before an element uses it; an
        undeclared net raises :class:`~silk.exceptions.UndeclaredNetError`.
        Default False.
    module_name : str, optional
        Subcircuit name. When set, the module is emitted as a ``.subckt``
        and cannot own an analysis. Must be a valid module name.
    db_path : str, optional
        SQLite database to bind for tracking; the database is opened lazily
        on the first tracking call.
    version : str, optional
        Design version recorded with tracked runs. Default ``"1.0"``.
    stage : silk.db.DesignStage, optional
        Design stage recorded with tracked runs; ``None`` means
        ``DesignStage.DEV``.
    created_by : str, optional
        Author recorded with tracked runs. Default ``"user"``.
    variants : bool, optional
        When True, a body that differs from an already-registered body under
        the same ``module_name`` is emitted under a structure-hash-suffixed
        name instead of raising. Default False.

    Attributes
    ----------
    results_file : str or None
        Path of the simulator rawfile (or behavioral result file) published
        by the latest completed run in this session; ``None`` before any run
        and after the results are cleared. Reading the attribute never
        raises. Whether waveform data is available is a separate question:
        a rawfile loaded with :meth:`load_file` can supply data while this
        stays ``None``, and reading signals with no data loaded raises
        ``SimulationError``.

    Examples
    --------
    >>> from tempfile import TemporaryDirectory
    >>> from silk import Module, load_pdk
    >>> folder = TemporaryDirectory()
    >>> pdk = load_pdk('sky130')
    >>> dut = Module('sky130', folder.name, 'demo', pdk)
    >>> from silk.elements import elements
    >>> dut.add(elements.resistor('rload', p='out', n='gnd', value=1000))
    >>> 'rload' in dut.elem_dict
    True
    """

    omit_device_internal_nodes = False

    def __init__(self, pdk_name, dir_folder, netlistname, pdk_details, debug=False, strict_nets=False, module_name=None,
                 db_path=None, version="1.0", stage=None, created_by="user", variants=False):
        self.pdk_name = pdk_name
        # Both end up in paths silk writes into the deck (TASK-593); refuse
        # before the directory below is created.
        from silk.utils.deck_text import refuse_line_breaks
        refuse_line_breaks(dir_folder, who='output directory (dir_folder)')
        refuse_line_breaks(netlistname, who='netlist file name (netlistname)')
        self.netlistname = netlistname
        self.dir_folder = dir_folder
        # Configured paths stay distinct from a prepared run's concrete files.
        self._run_files = None
        self._output_mode = 'new'
        self._run_preparation = None
        self._temperature_preparation = None
        self._supply_preparation = None
        self._deck_preparation = None
        self.operating_conditions = None
        self._run_consumed = False
        # Create the output directory up front. Deferring this produced a bare
        # FileNotFoundError('out/x.cir') at netlist time that never said the
        # *directory* was the problem — the #1 first-run footgun.
        os.makedirs(dir_folder, exist_ok=True)
        self.debug = debug

        self.device_name = 'module'  # This is for the add_element in pyNetlist

        # This is used for instantiation in the netlist
        if module_name is not None:
            _validate_module_name(module_name)
        self.module_name = module_name
        # TASK-292.1: opt-in per-Module. With variants=True, a body that
        # differs from an already-registered body under the same
        # module_name is emitted under a deterministic structure-hash-
        # derived suffixed name instead of raising; see
        # register_module_definition. A plain settable attribute — a
        # builder that names its module via define_module_name() instead
        # of this constructor kwarg can set mod.variants = True directly.
        self.variants = variants
        self.pin_list = None

        # Load PDK details
        self.pdk_details = pdk_details

        # ===== Use CircuitGraph for element management =====
        self._element_mgr = CircuitGraph(pdk_details, debug=debug, strict_nets=strict_nets)
        # Wire back-reference so Net objects created by the element manager
        # can hold a reference to this Module.
        self._element_mgr._module_ref = self

        # The element graph (elem_dict / net_dict / module_dict) is owned by
        # the CircuitGraph and exposed on Module as read-only views via
        # properties in _StructureMethods (B2 boundary) — NOT aliased here, so
        # there is a single owner and all mutation flows through CircuitGraph.
        self.module_names = self._element_mgr.module_names
        # ============================================================

        # Simulator selection must precede netlister/runner/backend use. The
        # validating setter rejects unknown names; the netlister is then built
        # lazily from the selected backend (symmetric with the lazy runner).
        self.simulator = 'ngspice'
        # The constructor default is not a deliberate embedded-test override.
        self._simulator_explicit = False

        self.passives_dict = {}  # Dictionary of passive elements (R, C, L)

        self.output_file_available = False
        self.sim_results_loaded = False
        self.sim_xaxis_values = ['time', 'frequency']
        self.custom_lines = []
        # Simulator-specific option key/value pairs (rendered by the active
        # simulator's YAML config — see silk/simulators/<name>.yaml).
        # Populate via :meth:`set_simulator_options`.
        self.simulator_options = {}
        # This Module's OWN explicit options (a subset of simulator_options,
        # excluding 'temp' which inherits through _temperature_explicit
        # instead) — what a test this Module owns inherits as its defaults
        # (TASK-224). Never includes keys written by inheritance itself.
        self._explicit_simulator_options = {}
        # Keys most recently written onto simulator_options BY inheritance
        # (silk.simulation.conditions.apply_inherited_simulator_options), so
        # a key an owner stops supplying can be un-inherited on the next run
        # instead of sticking around as a stale default.
        self._inherited_simulator_option_keys = frozenset()
        self._temperature_basis = 'native-and-observed'
        # The constructor default is not a deliberate temperature choice, so a
        # registered test inherits its live owner's setting until one of this
        # Module's own temperature setters is called (TASK-162, same shape as
        # ``_simulator_explicit``).
        self._temperature_explicit = False
        # Primary supply (TASK-158). The constructor default is not a
        # deliberate choice either, so a registered test inherits its live
        # owner's setting until one of this Module's own supply setters is
        # called. ``_declared_supply`` is the per-invocation
        # ``run(voltage=...)`` assertion, consumed by one launch.
        self._supply_voltage = None
        self._supply_source = None
        self._supply_explicit = False
        self._declared_supply = None

        # Simulation state (populated when Module is used as top-level netlist)
        self.sig_sources_dict = {}
        self.sim_analysis_dict = {}
        self.results_file = None
        self.circuit_name = None
        self.strict_nets = strict_nets
        # Opt-in: emit the simulator's device-current save directive (ngspice
        # `.options savecurrents`) so annotate_currents() can read per-pin
        # currents. Off by default — it inflates rawfiles. Set BEFORE netlist
        # generation.
        self.save_device_currents = False
        # How terminal currents are obtained when save_device_currents is on:
        # None -> use the active simulator's YAML default ('savecurrents');
        # 'ammeter_wrapper' -> wrap devices in 0 V-ammeter subckts (simulator-
        # independent, no drain-only/polarity limitation; MOS-terminal backlog).
        self.terminal_current_method = None
        # Opt-in (TASK-330.3): emit '@' device-parameter save lines for every
        # PDK MOS device, captured on a dedicated operating-point solve only
        # (never per-timestep/per-sweep-point — that would inflate a .tran/.dc
        # rawfile 4x+, measured TASK-330.1). Off by default. Set BEFORE
        # netlist generation, mirroring save_device_currents. Contradicts
        # save_device_currents (any terminal_current_method): that flag
        # unconditionally forces ngspice batch-write mode (no .control block
        # at all — the escape hatch for a zero-length '@m...[ib]' vector
        # .options savecurrents registers, which otherwise silently aborts a
        # .control 'write'), leaving no place for the interactive 'op'
        # save_device_params needs. Raises at generation time if both are set.
        self.save_device_params = False
        # Keep simulator-created device nodes unless filtering is explicitly requested.
        self.omit_device_internal_nodes = False
        # Opt-in simulator save list (set via set_save_list()): None = the
        # default `save all` — every node reaches the rawfile, which is what
        # the waveform explorer / annotate_currents / no-re-simulate loop
        # assume. A resolved list restricts the RAWFILE ITSELF (the only knob
        # that shrinks simulator write I/O and on-disk size — backlog
        # 2026-08-12-simulator-save-list.md).
        self._save_list = None

        # ===== NEW: Test management =====
        self._tests = {}  # Dictionary of test_name -> ModuleTestbench
        # ================================

        # ===== Database tracking (lazy — created on first tracking call) =====
        self._db_path = db_path
        self._db_version = version
        self._db_stage = stage
        self._db_created_by = created_by
        self._tracker = None

        # ===== Behavioral simulation =====
        self._behavioral_model = None       # callable set by set_behavioral_model()
        self._behavioral_params = None      # optional dict passed as third arg to model when set
        self._behavioral_form = 'ports'     # electrical ports are the model interface
        self._behavioral_execution = 'vector'  # calling contract: 'vector' or 'sample'
        self._specs = []                    # list of Spec objects declared on this module (Task 12)
        # Canonical path of this live instance inside its current root. It is
        # used for standalone runs; parent placements carry their own owner paths.
        self._spec_owner_path = '$'
        self._behavioral_mode = False       # True after simulate_behavioral()
        self._behavioral_t = None           # time vector

        # ===== Voltage domain tracking =====
        self._domains: list = []          # VoltageDomain objects declared on this module
        # ====================================

        # ===== Lint marker tracking =====
        self._no_connects: list = []      # (instance_name, net_name, NoConnect) tuples
        self._no_connect_ports: list = [] # NoConnectPort markers, one per waived submodule pin
        self._buses: dict = {}            # name → BusSpec
        self._bus_handles: dict = {}      # bus_name → bus-handle Net (width=N)
        # =================================

        # ===== Layout estimation state =====
        self.bounding_box = None          # Optional[Tuple[float, float]] — (width_um, height_um)
        self.half_circuits = {}           # Dict[str, HalfCircuit] — matched device declarations

        # Registered tests derive this complete descriptor from successful
        # operating conditions. Plain Modules and partial PVT stay unscoped.
        self.characterization_context = None
        from silk.circuit.placement_reference import register_module
        register_module(self)

    # ------------------------------------------------------------------
    # Pickle support — primarily for silk.faults.FaultCampaign, which
    # serializes a circuit-only worker payload. Behavioural-simulation
    # closures (``_behavioral_model``, CalibrationBinding callbacks,
    # database trackers) and the loaded run result (``_raw``, which wraps a
    # native rawfile object) are not needed for netlist generation and are
    # routinely non-picklable — drop them at pickle time and restore ``None``
    # on the other side.
    # ------------------------------------------------------------------

    _UNPICKLABLE_ATTRS = (
        '_behavioral_model',
        '_tracker',
        '_raw',
    )

    def __reduce_ex__(self, protocol):
        # Only strip the unpicklable attributes when genuinely pickling
        # (i.e. via pickle/copyreg). ``copy.deepcopy`` goes through
        # ``__deepcopy__`` below instead, so behavioural-model callables
        # survive an explicit deepcopy.
        from silk.circuit.test_placement import selected_probe_pickle_bindings
        state = dict(self.__dict__)
        bindings = selected_probe_pickle_bindings(self)
        if bindings:
            state['_selected_probe_bindings'] = bindings
        state.pop('_test_evidence', None)
        state.pop('_placement_reader', None)
        for attr in self._UNPICKLABLE_ATTRS:
            state[attr] = None
        return (_reconstruct_module, (type(self),), state)

    def __setstate__(self, state: dict) -> None:
        bindings = state.pop('_selected_probe_bindings', ())
        self.__dict__.update(state)
        from silk.circuit.placement_reference import register_module
        register_module(self)
        from silk.circuit.object_holders import restore_module
        restore_module(self)
        if bindings:
            from silk.circuit.test_placement import restore_selected_probe_bindings
            restore_selected_probe_bindings(self, bindings)

    def __deepcopy__(self, memo):
        import copy as _copy
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k in ('_test_evidence', '_placement_reader'):
                continue
            try:
                new.__dict__[k] = _copy.deepcopy(v, memo)
            except (TypeError, RecursionError):
                new.__dict__[k] = v
        from silk.circuit.placement_reference import register_module
        register_module(new)
        from silk.circuit.object_holders import restore_module
        restore_module(new)
        return new

    def __enter__(self):
        """Use the module as a context manager to auto-register elements.

        Devices, sources, and passives constructed inside a ``with mod:``
        block are automatically registered with ``mod``::

            with fcamp:
                pdk.nmos1p8('mn1', d='vd', g='vg', s='gnd', b='gnd', w=1, l=0.18)
                sources.CurrentSource('ibias', positive='avdd', negative='vb', dc=1e-3)
                passives.Capacitor('cl', pos='vout', neg='gnd', capacitance=1e-12)

        Nesting is supported — the innermost active module wins. Outside
        any ``with`` block, constructors behave exactly as before (the
        caller must invoke ``.add()`` / ``.add_source()`` /
        ``.add_passive()`` explicitly).
        """
        _push_active(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _pop_active()
        return False
