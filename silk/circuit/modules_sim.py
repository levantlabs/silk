"""Simulator configuration, backend selection and run methods for :class:`~silk.circuit.modules.Module`.

Split out of ``modules.py`` as a per-subsystem **part**. ``_SimMethods`` is a pure mixin: no state, no separate identity. Owns the ``simulator`` / ``backend`` / ``netlister`` properties and the run pipeline; backend/analysis/simulator-config modules are imported lazily inside the methods that need them.
"""

from silk.circuit.test_placement import execution_location, execution_supply_declaration

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING
from silk.exceptions import (
    NotSupportedError, SimulationConfigurationError, SimulationError,
)

if TYPE_CHECKING:  # pragma: no cover
    from silk.circuit.modules import Module
    from silk.circuit._module_protocol import _ModuleAttrs
    _Base = _ModuleAttrs
else:
    _Base = object



def _axis_declaration(module, axis, inherited=None):
    """Nearest authored list/clear/scalar choice, distinct from no choice."""
    name = '_' + axis + '_list'
    if module is not None and name in vars(module):
        return vars(module)[name]
    if getattr(module, '_' + axis + '_explicit', False):
        return (None, None)
    return inherited


def _validate_axis(values, axis):
    from silk.exceptions import SimulationConfigurationError
    from silk.simulation.conditions import validate_temperature, validate_supply
    if values is None:
        return None
    try:
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError('expected a nonempty list or tuple of numbers')
        validate = validate_temperature if axis == 'temperature' else validate_supply
        result = tuple(validate(value) for value in values)
        if any(value is None for value in result):
            raise ValueError('None clears the list; it is not a list element')
        result = tuple(0.0 if value == 0 else float(value) for value in result)
        return result
    except (TypeError, ValueError) as exc:
        raise SimulationConfigurationError(f'Invalid {axis} list: {exc}') from exc

# This scope records a fact about one attempt, not a persistent module flag.
_attempt_stage = ContextVar('silk_attempt_stage', default=None)


@contextmanager
def _track_attempt_stage():
    # Independent of result invalidation: dispatch/test doubles may not prove a launch.
    state = ['execution']
    token = _attempt_stage.set(state)
    try:
        yield state
    finally:
        _attempt_stage.reset(token)


def _mark_attempt_stage(stage):
    current = _attempt_stage.get()
    if current is not None:
        current[0] = stage


_result_replacement = ContextVar('silk_result_replacement', default=None)


@contextmanager
def _track_result_replacement(module):
    state = [False]
    token = _result_replacement.set((module, state))
    try:
        yield state
    finally:
        _result_replacement.reset(token)


def _begin_result_replacement(module):
    attempt = _result_replacement.get()
    if attempt is not None and attempt[0] is module:
        attempt[1][0] = True
    _invalidate_current_results(module)


def _invalidate_current_results(module):
    """Drop live results and implicit reload paths, leaving historical evidence."""
    module._raw = None
    module._prepared_device_param_requests = ()
    module._behavioral_element_notes = ()
    module.simResultsAll = None
    module.sim_results_loaded = False
    module.results_file = None
    module.dataFile = None
    module.output_file_available = False
    module.characterization_context = None
    module.operating_conditions = None
    module._last_sim_corner = None
    module._last_sim_label = None
    module._launch_record = None


def _refuse_deckless_backend(backend, call):
    """Refuse a deck operation on a backend that runs modules, not decks."""
    if backend.runs_module:
        raise NotSupportedError(
            f"{call} needs a SPICE deck, and the {backend.simulator_name!r} "
            f"backend has none: it runs the Module itself. Call simulate(), "
            f"or set simulator to a SPICE backend ('ngspice', 'xyce') to "
            f"emit and run a netlist.")


def _simulate_module(module, backend, corner=None):
    """``Module.simulate()`` on a backend that runs modules.

    The backend's ``run_module`` template applies the data-presence gate.
    Results are published only after it returns, so a refused or aborted run
    leaves no current results.

    The run location is selected exactly as a SPICE run's is:
    overwrite mode reuses the deterministic ``dir_folder`` and removes the
    previous rawfile before the run, so a failed run leaves no stale file
    behind; new mode allocates a persistent ``run-*`` child per call. The
    backend writes its rawfile there, and ``results_file`` names it.
    """
    from silk.simulation._setup_check import (
        refuse_ambient_temperature, require_static_setup,
    )
    refuse_ambient_temperature(module)
    from silk.simulation.conditions import prepare_supply, _classify_supply_sweep
    # Before the run, so a configured supply is the value the engine drives
    # (the counterpart of rendering it into a deck). Resolution failures are
    # configuration errors and refuse here, before any output is touched.
    supply_request = _classify_supply_sweep(module, prepare_supply(module))
    # The shared static admission checks (TASK-590), before any result is
    # replaced or output touched: a static refusal keeps earlier results.
    require_static_setup(module)
    _begin_result_replacement(module)
    module._behavioral_mode = False
    from silk.circuit.modules_netlist import _prepare_run_files
    files = _prepare_run_files(module)
    files.clear_previous_outputs()
    from silk.circuit.view_nodes import _capture_addresses
    addresses = _capture_addresses(module)
    _mark_attempt_stage('simulation')
    # The in-process engine is entered here: its own files, never a deck or an
    # earlier SPICE preparation's paths (TASK-499).
    from silk.simulation.launch_fact import note_launch
    note_launch(True, files)
    raw = backend.run_module(module, run_files=files)
    raw._view_addresses = addresses
    _publish_module_run(module, backend, raw, supply_request, corner)


def _publish_module_run(module, backend, raw, supply_request, corner):
    """Make one gated module-level run *module*'s current results.

    ``Net.voltage`` and ``_behavioral_t`` are written from the rawfile's own
    vectors (lint STR-001 reads them), and the RawResult becomes the results
    every SPICE reader reads. Provenance records that silk ran it,
    on the ``'behavioral'`` backend; corner and temperature are proven none;
    the supply is configured-and-driven, a verified DC sweep, proven none, or unknown.
    """
    from dataclasses import replace
    from silk.simulation.behavioral_backend import behavioral_net_arrays
    from silk.simulation.conditions import (
        OperatingConditions, driven_supply, has_no_supply,
    )
    arrays = behavioral_net_arrays(raw.native)
    conditions = OperatingConditions(
        'silk', backend.simulator_name, None,
        corner_kind='none', corner_basis='proven-behavioral',
        temperature_kind='none', temperature_basis='proven-behavioral',
    )
    if supply_request.opaque:
        conditions = replace(conditions, supply_source=supply_request.source)
    elif supply_request.sweep is not None:
        from silk.simulation.backend import select_plot
        dc_plot = select_plot(raw.native, 'dc')
        driven_supply(supply_request, module, behavioral_net_arrays(dc_plot),
                      axis=dc_plot.vectors[0].data)
        if supply_request.configured and raw.native.current_plot is not dc_plot:
            driven_supply(supply_request, module, arrays)
        conditions = replace(conditions, supply_kind='sweep',
                             supply_basis='configured',
                             supply_source=supply_request.source,
                             supply_sweep=supply_request.sweep)
    elif supply_request.configured:
        volts, basis, source = driven_supply(supply_request, module, arrays)
        conditions = replace(conditions, supply_voltage_v=volts,
                             supply_kind='fixed', supply_basis=basis,
                             supply_source=source)
    elif has_no_supply(module):
        conditions = replace(conditions, supply_kind='none',
                             supply_basis='proven-no-supply')
    from silk.circuit.modules_results import _write_behavioral_voltages
    _write_behavioral_voltages(module, arrays)
    module._behavioral_t = raw.native.vectors[0].data
    module._behavioral_mode = True
    module._raw = raw
    module.simResultsAll = raw.native
    module.results_file = raw.source_path
    module.dataFile = raw.source_path
    module.sim_results_loaded = True
    module.operating_conditions = conditions
    from silk.simulation.provenance import launch_record
    module._launch_record = launch_record(module.simulator)
    module._last_sim_corner = conditions.scalar('corner')
    module._last_sim_label = None if corner is None else str(corner)


def _require_osdi_compiled(pdk_details):
    """Fail a RUN, with instructions, when the PDK's OSDI plugins are missing.

    A PDK whose device models are compiled Verilog-A declares them in its
    manifest (``library.osdi:``); the emitted deck loads them via ``pre_osdi``.
    Generating that deck never requires the files — it is valid wherever the
    plugins ARE compiled — but running it here does, and ngspice's own failure
    is a parse error that never mentions OSDI. Same contract as ngspice
    itself: silk emits without it, and fails loudly at run time without it.
    Module-level so it does not enlarge ``dir(Module)`` (API golden).
    """
    import os
    from silk.exceptions import SimulationError
    missing = [p for p in getattr(pdk_details, 'osdi_files', None) or []
               if not os.path.isfile(p)]
    if missing:
        raise SimulationError(
            "This PDK's device models are compiled Verilog-A (OSDI), and the "
            "compiled plugins are not on disk:\n  " + "\n  ".join(missing) +
            "\nsilk does not compile them (environment setup, like installing "
            "ngspice). Install openvaf-r (or openvaf), then run the PDK's own "
            "compile script — for SG13G2: "
            "$PDK_ROOT/ihp_sg13g2/ihp-sg13g2/libs.tech/ngspice/install.py. "
            "python -m silk.doctor reports this state per PDK."
        )


def _require_pdk_library(pdk_details):
    """Fail a RUN, with instructions, when the PDK's model library is missing.

    ``load_pdk`` resolves the ``.lib`` path but no longer stat-s it:
    the catalog, units and corners all ship inside silk, so a PDK that is not
    downloaded still loads usefully, and a deck that names no PDK device runs
    without it. The one thing that genuinely cannot work is running a deck
    whose ``.lib`` line points at a file that is not there — ngspice's own
    failure for that is ``Could not find library file``, which never says
    which PDK or how to install it. Same contract as
    :func:`_require_osdi_compiled`, and as ngspice itself: silk emits without
    it, and fails loudly at run time without it.

    Module-level so it does not enlarge ``dir(Module)`` (API golden).
    """
    import os
    from silk.exceptions import SimulationError
    lib = getattr(pdk_details, 'library_location', None)
    # An empty/None location is a deliberate no-library PDK object (schema
    # export); ``os.devnull`` is the documented "this PDK has no lib file"
    # placeholder every PDK-free netlisting example uses, spelled exactly as
    # silk.pdk.validate._is_placeholder_lib reads it. Neither is a missing
    # install, and neither should be turned into one here.
    if not lib or os.path.isfile(lib):
        return
    if os.path.realpath(lib) == os.path.realpath(os.devnull):
        return
    # Scoped to the manifest's primary lib file, which is exactly what
    # `python -m silk.doctor` reports per PDK — one answer to "is this PDK
    # installed", not two that can disagree.
    name = getattr(pdk_details, 'pdk_name', None) or type(pdk_details).__name__
    raise SimulationError(
        f"This deck instantiates {name} devices, and the PDK's model library "
        f"is not on disk:\n  {lib}\n"
        "silk ships the device catalog, not the foundry models — install the "
        "PDK (see docs/installing-the-pdk.md), set $PDK_ROOT, or pass "
        "library_location= to load_pdk(). python -m silk.doctor reports this "
        "state per PDK. A testbench built from ideal elements, passives and "
        "sources only needs no PDK at all and runs as-is."
    )


class _SimMethods(_Base):
    """Simulator-config / run / analysis-type methods + backend properties mixed into ``Module``. Holds no state."""

    def set_temperatures(self, temperatures: list[float] | tuple[float, ...] | None) -> 'Module':
        """Declare ambient Celsius points for registered tests; None clears.

        Lists inherit live through owners. A scalar setter replaces the list;
        an explicit clear suppresses inherited lists and uses scalar settings.
        Direct simulate() requires scalar settings instead.
        """
        values = _validate_axis(temperatures, 'temperature')
        if self.module_name and not self._tests:
            raise ValueError('Set temperatures on a testbench or a block with registered tests.')
        self._temperature_list = (values, None)
        return self

    def set_supply_voltages(self, voltages: list[float] | tuple[float, ...] | None,
                            *, source: str | None = None) -> 'Module':
        """Declare supply points for registered tests, using scalar source rules.

        None clears the list and suppresses inherited lists. Source selection
        is resolved against each executing test, never guessed from a name.
        """
        from silk.exceptions import SimulationConfigurationError
        values = _validate_axis(voltages, 'supply')
        if source is not None and (not isinstance(source, str) or not source.strip()):
            raise SimulationConfigurationError('source must name a voltage source.')
        if values is None and source is not None:
            raise SimulationConfigurationError('Clearing supply voltages takes no source.')
        if self.module_name and not self._tests:
            raise ValueError('Set supply voltages on a testbench or a block with registered tests.')
        self._supply_list = (values, source)
        return self

    def set_temperature(self, temperature_c: float | None) -> 'Module':
        """Configure ambient TEMP in Celsius on this Module and its own tests.

        Finite real values above -273.15 C are accepted; ``None`` clears TEMP
        without changing model-reference TNOM. Returns self. The setting is
        shared with ``set_simulator_options(temp=...)``; the last call wins.
        Application is verified at execution on the backends that can report
        the temperature they solved at — ngspice (run log) and Xyce (a
        ``{TEMP}`` probe printed beside the rawfile). Any other backend
        refuses the request rather than labelling it applied.

        The setter is refused only where nothing would apply it: a subcircuit
        that owns **no registered tests**. An instantiated
        ``.subckt`` carries no simulator options of its own, so a setting
        there would be silently ignored. A block that owns tests — the
        canonical embedded-testbench shape — accepts it and becomes their
        default, which is how one call reaches all of a block's tests without
        a wrapper circuit. Register the tests first: the refusal reads the
        Module as it is at the call.

        The setting applies to this Module's own ``simulate()`` runs *and*,
        by inheritance, to every registered test that has no temperature of
        its own: each test resolves the setting from its live
        owner at every run, so a later change or ``set_temperature(None)`` is
        followed rather than pinned. A test that calls its own setter keeps
        that value, which always wins over the inherited one. Under a
        recursive ``run_all_tests`` the nearest setting up the live tree is
        the one in force. ``None`` is itself a deliberate setting: it opts a
        test out of its owner's temperature for that test's own runs.

        Parameters
        ----------
        temperature_c : float or None
            Ambient temperature in degrees Celsius, finite and above
            -273.15; ``None`` clears the setting.

        Returns
        -------
        Module
            This module, for chaining.

        Raises
        ------
        TypeError
            If ``temperature_c`` is neither a real number nor ``None``
            (booleans are refused).
        ValueError
            If it is NaN, infinite, or at or below absolute zero, or this
            is a subcircuit that owns no registered tests.
        """
        from silk.simulation.conditions import validate_temperature
        if self.module_name and not self._tests:
            raise ValueError(
                'Set temperature where something applies it: this subcircuit '
                'owns no registered tests, so the setting would be ignored — '
                'an instantiated .subckt carries no simulator options of its '
                'own. Set it on the testbench, on the top-level Module, or on '
                'this block once its tests are registered.')
        value = validate_temperature(temperature_c)
        if value is None:
            self.simulator_options.pop('temp', None)
        else:
            self.simulator_options['temp'] = value
        self._temperature_basis = 'configured-and-observed'
        # A deliberate choice on THIS Module, including a deliberate clear:
        # an embedded test of this Module stops inheriting from here on.
        self._temperature_explicit = True
        self._temperature_list = (None, None)
        return self

    def set_supply_voltage(self, volts: float | None, *, source: str | None = None) -> 'Module':
        """Configure the primary supply in volts on this executing Module.

        Sets the DC value of one voltage source before the deck is emitted, so
        the supply a report names is the supply the simulator was given rather
        than a caller declaration next to it. Returns self. The
        value is re-applied at every emission and confirmed against the deck
        that is actually launched; a mismatch refuses instead of reporting.

        Refused only where nothing would apply it, exactly as
        ``set_temperature`` is: a subcircuit that owns **no registered tests**. A block that owns tests accepts the setting and becomes
        their default supply; register the tests before calling it.

        Parameters
        ----------
        volts : float or None
            Finite real volts, or ``None`` to stop configuring a supply.
            Clearing is deliberate, like ``set_temperature(None)``: it opts
            this Module (and a test of it) out of a supply setting rather than
            resuming inheritance. It does **not** restore a value a previous
            call overwrote -- the source keeps the DC value it was last given,
            which is the one the deck will carry.
        source : str, optional
            The name of the voltage source to set. Omit it and silk resolves
            the supply from the circuit itself, never from a name: among the
            purely-DC voltage sources referenced to ground, the one whose
            positive net is classified ``POWER`` by a declared port role, or
            -- when no rail is role-classified -- the only such source in the
            bench. Several equally supply-shaped sources, or none, refuse with
            a message naming what was found; pass ``source=`` to settle it.
            A time-varying source is never a supply: no scalar setpoint is
            true of it, so record its waveform knobs as test parameters.

        Returns
        -------
        Module
            This module, for chaining.

        Raises
        ------
        ValueError
            If this is a subcircuit that owns no registered tests, ``source``
            is not a non-empty string, or ``source`` is given with
            ``volts=None``.
        silk.exceptions.CharacterizationContextError
            If ``volts`` is not a finite real number.

        Notes
        -----
        The setting applies to this Module's own ``simulate()`` runs *and*, by
        inheritance, to every registered test that has no supply of its own,
        exactly as ambient temperature inherits. The source is
        re-resolved against the executing Module at each run, which is what
        lets one setting on a circuit reach the separate testbench Module that
        actually holds the rail. A test that calls its own setter keeps that
        value, which always wins over the inherited one; under a recursive
        ``run_all_tests`` the nearest setting up the live tree is in force.
        """
        from silk.simulation.conditions import validate_supply
        if self.module_name and not self._tests:
            raise ValueError(
                'Set the supply where something applies it: this subcircuit '
                'owns no registered tests, so the setting would be ignored — '
                'an instantiated .subckt holds no rail of its own. Set it on '
                'the testbench, on the top-level Module, or on this block '
                'once its tests are registered.')
        if source is not None and (not isinstance(source, str) or not source.strip()):
            raise ValueError('source must be the name of a voltage source, or None.')
        value = validate_supply(volts)
        if value is None and source is not None:
            raise ValueError('set_supply_voltage(None) clears the setting and '
                             'takes no source; pass a voltage to name one.')
        self._supply_voltage = value
        self._supply_source = source
        # A deliberate choice on THIS Module, including a deliberate clear:
        # an embedded test of this Module stops inheriting from here on.
        self._supply_explicit = True
        self._supply_list = (None, None)
        return self

    def set_output_options(self, mode: str = 'new', *, layout: str = 'flat') -> None:
        """Configure where subsequent simulation attempts write their artifacts.

        Parameters
        ----------
        mode : {'overwrite', 'new'}, default 'new'
            Reuse the configured directory, or allocate a persistent unique
            child per prepared deck. Embedded tests read their live owner's
            setting at run time; recursive ``run_all_tests`` uses its root's
            setting for that invocation without changing child settings.
            ``run_label`` still selects the stable test/label base. Cache
            hits and explicit file evaluation do not prepare new artifacts.
        layout : {'flat'}, default 'flat'
            Folder layout, keyword-only. ``'flat'`` writes each embedded test
            to ``test_<module>~<test>/``. Nothing else is supported yet.

        Raises
        ------
        ValueError
            Unsupported mode or unknown layout. The previous setting and
            artifacts are kept.
        NotSupportedError
            ``layout='nested'`` (``test_<module>/<test>/``) is not supported
            yet. Settings are unchanged.

        Notes
        -----
        This setter only validates and configures: it does not allocate,
        generate, run or delete files. Overwrite attempts discard the previous
        same-location output even on failure. Save successful ``results_file``
        paths for later readback; reports still reference a live Module.
        In new mode, regenerate the deck before each ``run_simulator`` call.
        Unique files do not make concurrent runs on one Module safe.
        """
        if not isinstance(mode, str) or mode not in ('overwrite', 'new'):
            raise ValueError(
                f"mode must be 'overwrite' or 'new'; got {mode!r}. "
                "Use set_output_options(mode='overwrite') or mode='new'."
            )
        if layout == 'nested':
            from silk.exceptions import NotSupportedError
            raise NotSupportedError(
                "set_output_options(layout='nested') is not supported yet; "
                "the flat layout test_<module>~<test>/ is used. Get paths "
                "from results_file, prepare() or the report.")
        if layout != 'flat':
            raise ValueError(
                f"layout must be 'flat' (the only supported value); got "
                f"{layout!r}.")
        self._output_mode = mode

    def add_sim_type(self, simType):
        """Register a simulation analysis.

        Parameters
        ----------
        simType : AnalysisBase
            An analysis object — ``AC``, ``Transient``, ``DC``, or ``Noise``
            from ``silk.simulation.analysis``. The legacy string form
            (``add_sim_type('transient', [tstep, tstop])``) has been removed.

        Raises
        ------
        ValueError
            If this module is a subcircuit (``module_name`` is set).
            Subcircuits emit a ``.subckt`` block and cannot own a sim
            analysis — wrap them in a top-level testbench Module and
            call ``add_sim_type`` there instead.
        TypeError
            If *simType* is not an :class:`~silk.simulation.analysis.AnalysisBase`.
        """
        if self.module_name:
            raise ValueError(
                f"add_sim_type() called on subcircuit "
                f"{self.module_name!r}. Subcircuits emit a .subckt block "
                f"and do not run simulations themselves. Create a "
                f"top-level testbench Module (no module_name), instantiate "
                f"this one via tb.add_element(...), and call add_sim_type "
                f"on the testbench."
            )
        from silk.simulation.analysis import AnalysisBase
        if not isinstance(simType, AnalysisBase):
            raise TypeError(
                "add_sim_type() requires an analysis object (AC, Transient, DC, "
                "or Noise from silk.simulation.analysis). The legacy string form "
                "— e.g. add_sim_type('transient', [tstep, tstop]) — was removed; "
                "use add_sim_type(Transient(step=..., stop=...)) instead. "
                f"Got {type(simType).__name__}."
            )
        type_key = type(simType).__name__.lower()
        self.sim_analysis_dict[type_key] = simType

    def set_mismatch(self, enabled, *, devices=None):
        """Switch statistical mismatch on or off for this block (Monte Carlo).

        Per block only: an off block emits ``.param <mismatch switch>=0`` as
        the first line of its ``.subckt`` in a mismatch or both sample deck,
        so it and every block below it draw no mismatch while the rest of the
        design still does. Process-sample decks and every non-Monte-Carlo
        deck are unchanged. Call it before the block is instantiated with
        ``add_element``: the parent stores a copy of the block at that point.

        Parameters
        ----------
        enabled : bool
            ``False`` turns mismatch off for this block's hierarchy.
        devices : None
            Reserved. Per-device control is not supported.

        Raises
        ------
        TypeError
            If *enabled* is not a bool.
        NotSupportedError
            If *devices* is given (per block only), and, when a Monte Carlo
            deck is generated for Xyce, if any block has mismatch off.
        ValueError
            On a top-level testbench (no ``module_name``).
        """
        from silk.exceptions import NotSupportedError
        if not isinstance(enabled, bool):
            raise TypeError(
                f"set_mismatch() requires a bool, got {enabled!r}")
        if devices is not None:
            raise NotSupportedError(
                "set_mismatch(devices=...) is not supported: mismatch is "
                "switched per block only. An instance-line switch is "
                "silently ignored by ngspice, so per-device control cannot "
                "work; put the devices in their own block instead.")
        if not self.module_name:
            raise ValueError(
                "set_mismatch() called on a top-level testbench. Mismatch "
                "is switched per block: call it on the subcircuit Module "
                "(one with module_name) before instantiating it.")
        self._mismatch_enabled = enabled

    @property
    def mismatch_enabled(self):
        """``False`` once :meth:`set_mismatch` turned mismatch off here."""
        return getattr(self, '_mismatch_enabled', True)

    def add_custom_line(self, line):
        """Add a raw SPICE line to the netlist.

        The line belongs to this circuit: on a subcircuit it is emitted
        verbatim inside its ``.subckt``. It must be legal for the simulator
        and subcircuit scope.
        """
        self.custom_lines.append(line)

    def set_simulator_options(self, **options):
        """Set simulator options for this Module's own runs, and for every
        test it owns that has not set the same key itself.

        The options are rendered into the netlist by the currently-active
        simulator's YAML config (``silk/simulators/<simulator>.yaml``). The
        YAML defines the emit syntax, the preset vocabulary, and a known-
        options allowlist.

        Pass ``preset=<name>`` to expand one of the simulator's named presets
        (e.g. ``'default'``, ``'tight'``, ``'fast'``). Additional kwargs are
        merged on top of the preset, overriding or adding keys. Calling the
        method multiple times accumulates options.

        A Module that owns registered tests (the embedded-testbench shape)
        has its options reach every one of them as defaults: resolved from
        this live Module at every run, so a later change here is followed,
        exactly as ``set_temperature`` inherits. A test that
        calls its own ``set_simulator_options`` -- including
        ``tb.set_simulator_options``, ``ModuleTestbench``'s thin delegate to
        ``tb.netlist.set_simulator_options`` -- overrides the inherited
        default key by key; nothing is silently dropped. ``temp`` inherits
        through ``set_temperature``'s own path instead and is unaffected by
        this paragraph.

        Examples
        --------
        On a Module used directly as its own testbench::

            tb.simulator = 'ngspice'                                  # default
            tb.set_simulator_options(preset='tight')                  # preset only
            tb.set_simulator_options(preset='tight', method='trap')   # override a key
            tb.set_simulator_options(reltol=1e-6)                     # raw key

        On a ``ModuleTestbench`` (``tb = mod.create_test(...)``), the same
        calls reach ``tb.netlist`` through its thin delegate -- ``tb.simulator``
        is not one of its attributes, so assign ``tb.netlist.simulator``
        there instead::

            tb.netlist.simulator = 'ngspice'                          # default
            tb.set_simulator_options(preset='tight')                  # -> tb.netlist
        """
        from silk.simulators import load_simulator_config
        cfg = load_simulator_config(self.simulator)
        preset = options.pop('preset', None)
        if preset is not None:
            options = {**cfg.resolve_preset(preset), **options}
        from silk.simulators._loader import _refuse_option_line_breaks
        _refuse_option_line_breaks(options)
        cfg.reject_unknown(options)
        if 'temp' in options:
            from silk.simulation.conditions import validate_temperature
            validate_temperature(options['temp'], native=True)
            self._temperature_basis = 'native-and-observed'
            # Same shared setting as set_temperature(), so the same deliberate
            # choice: this Module's tests stop inheriting a temperature.
            self._temperature_explicit = True
            self._temperature_list = (None, None)
            if options['temp'] is None:
                self.simulator_options.pop('temp', None)
                options.pop('temp')
        self.simulator_options.update(options)
        cfg.warn_unknown(options)
        # This Module's OWN explicit settings (excluding temp, which is
        # tracked by _temperature_explicit instead) — what a test this
        # Module owns inherits as defaults (TASK-224).
        self._explicit_simulator_options.update(
            {k: v for k, v in options.items() if k != 'temp'}
        )

    # Backwards-compatible alias — removed in a future release.
    def set_spice_options(self, **options):
        """Deprecated: use :meth:`set_simulator_options` instead."""
        import warnings
        warnings.warn(
            'set_spice_options() is deprecated; use set_simulator_options().',
            DeprecationWarning, stacklevel=2,
        )
        self.set_simulator_options(**options)

    def netlist_header(self, header):
        """Set an optional netlist header comment (overwrites this attribute)."""
        self.netlist_header = header

    @property
    def backend(self):
        """The SimulatorBackend selected by ``self.simulator`` (stateless)."""
        from silk.simulation.backend import SimulatorBackend
        return SimulatorBackend.get(self.simulator)

    @property
    def simulator(self):
        """Selected backend, e.g. ``'ngspice'`` or ``'xyce'``.

        Registered tests inherit their live owner's selection at each run.
        Assigning ``tb.netlist.simulator`` explicitly overrides inheritance,
        including an assignment of the default ``'ngspice'``. Selection alone
        does not prepare files or launch a simulator. Numerical options and
        executable paths are configured separately.
        """
        return self._simulator

    @simulator.setter
    def simulator(self, name):
        # Fail fast: reject an unregistered simulator at assignment time rather
        # than later when the backend is first used.
        from silk.simulation.backend import SimulatorBackend
        SimulatorBackend.get(name)
        self._simulator = name
        self._simulator_explicit = True

    @property
    def netlister(self):
        """Netlister for the current simulator, built lazily from its backend.

        Rebuilt automatically when ``simulator`` changes, so a post-construction
        switch (``mod.simulator = 'xyce'``) yields a consistent netlister with no
        extra step — dissolving the old eager-netlister / lazy-runner asymmetry.
        """
        if (getattr(self, '_netlister', None) is None
                or getattr(self, '_netlister_sim', None) != self._simulator):
            self._netlister = self.backend.make_netlister(self.pdk_details)
            self._netlister_sim = self._simulator
        return self._netlister

    @netlister.setter
    def netlister(self, value):
        self._netlister = value
        self._netlister_sim = self._simulator

    def run_simulator(self, auto_track_structure=False, timeout=None):
        """Run the simulator on the generated netlist.

        File-producing runners remove exact previous outputs before launch.
        A replacement attempt invalidates live results and automatic reload
        paths; failure leaves no current waveform or successful context.
        Historical scalar reports and stored evidence are retained.
        In new mode, this consumes the location selected by
        ``generate_netlist()``. An absent, mismatched or consumed preparation
        raises ``SimulationError`` before cleanup; regenerate before retrying.
        In either mode, deck-affecting configuration changed since generation
        also raises before cleanup, naming the first changed deck line: the
        prepared file is untouched by such a change, so nothing else would
        notice that the launched deck is not the configuration being reported.

        Parameters
        ----------
        auto_track_structure : bool, optional
            Accepted for backward compatibility; it has no effect.
        timeout : float, optional
            Maximum seconds to wait for the simulator. None means no limit.

        Raises
        ------
        silk.exceptions.SimulationConfigurationError
            In new output mode, if the deck was not prepared by
            ``generate_netlist()`` or was already consumed; in either mode,
            if deck-affecting configuration changed since generation. In
            overwrite mode an externally supplied deck is accepted without a
            prior ``generate_netlist()``.
        silk.exceptions.SimulationError
            If the simulator fails or produces no data.
        """
        backend = self.backend
        _refuse_deckless_backend(backend, 'run_simulator()')
        from silk.circuit.modules_netlist import _require_fresh_deck
        from hashlib import sha256
        from pathlib import Path
        from silk.simulation.conditions import (
            OperatingConditions, observed_conditions, observed_supply,
            prepare_supply, prepare_temperature,
        )
        # The per-invocation supply declaration (TASK-158): a
        # ModuleTestbench.run(voltage=...) assertion scopes exactly one
        # launch, so consume it before any refusal below can leave it set.
        declared_supply = execution_supply_declaration(self)
        directory, filename, mode = execution_location(self)
        self._declared_supply = None
        request = prepare_temperature(self)
        temperature_preparation = self._temperature_preparation
        files = self._run_files
        preparation = self._run_preparation
        if (mode == 'new'
                or (preparation is not None and preparation[3] == 'new')):
            expected = (directory, filename, self.simulator,
                        mode, files)
            if (preparation != expected or self._run_consumed
                    or files is None or not os.path.isfile(files.netlist_path)):
                raise SimulationConfigurationError(
                    'New-mode simulation requires an unconsumed prepared deck '
                    'matching the current output paths, mode and simulator. '
                    'Call generate_netlist() again (or simulate()) to prepare '
                    'the next run before calling run_simulator().'
                )
        launch_options = {} if files is None else {'run_files': files}
        sample = getattr(self, '_mc_sample', None)
        if sample is not None and files is not None:
            sample_args = backend.sample_args(sample)
            if sample_args:
                launch_options['extra_args'] = sample_args
        if files is None:
            # A caller may supply an existing deck without generating it here.
            # Consume those explicit locations without preparing or moving it.
            files = backend._resolve_run_files(filename, directory)
        if temperature_preparation is not None:
            prepared_request, prepared_corner, deck_digest = temperature_preparation
            try:
                current_digest = sha256(Path(files.netlist_path).read_bytes()).digest()
            except OSError as exc:
                raise SimulationConfigurationError(
                    'Prepared deck is unavailable. Call generate_netlist() '
                    'before run_simulator().') from exc
            if (request.temperature_identity != prepared_request.temperature_identity
                    or current_digest != deck_digest):
                raise SimulationConfigurationError(
                    'Temperature configuration or prepared deck changed. Call '
                    'generate_netlist() again before run_simulator().')
        elif request.temperature is not None:
            raise SimulationConfigurationError(
                'Managed TEMP requires a prepared deck. Call generate_netlist() '
                'before run_simulator(), or use simulate().')
        target = getattr(self, '_prepared_ngspice_version', None)
        if target is not None:
            from silk.simulation._ngspice_version import ngspice_version
            from silk.simulation._ngspice_compat import _release
            installed = ngspice_version()
            if installed is None or _release(installed) != _release(target):
                raise SimulationConfigurationError(
                    f'Prepared deck targets ngspice {target}, but installed ngspice '
                    f'is {installed or "unknown"}. Run this deck externally or call '
                    'generate_netlist() again for the installed release.')
        # General staleness gate (TASK-160), deliberately last of the three so
        # the narrower refusals above keep their own wording: the deck that the
        # live configuration would emit now must be the deck that was prepared.
        # Both checks above look at the file on disk, which a configuration
        # change never touches.
        _require_fresh_deck(self)
        # Only a generated preparation ties this live inventory to the deck.
        # External/fault decks deliberately have none, and remain unchecked.
        if self._deck_preparation is not None:
            from silk.simulation._setup_check import require_static_setup
            _mark_attempt_stage('preflight')
            require_static_setup(self)
        # Both environment gates below are scoped to what the EMITTED deck
        # actually references. A deck of nothing but ideal elements, passives
        # and sources carries no `.lib` and no `pre_osdi` (see
        # `deck_uses_pdk_devices`), so demanding an installed model library or
        # compiled plugins for it would refuse a run that works — exactly the
        # PDK-free rung TASK-71 opened.
        #
        # The same proof decides the corner this launch records (design
        # 2026-10-01 section 3.1): a deck with no PDK device has no corner.
        from silk.simulation.conditions import has_no_pdk_device, has_no_supply
        no_pdk_device = has_no_pdk_device(self)
        if not no_pdk_device:
            _require_pdk_library(self.pdk_details)
            _require_osdi_compiled(self.pdk_details)
        supply_request = prepare_supply(self, apply=False)
        prepared_supply = self._supply_preparation
        if (prepared_supply is not None
                and supply_request.supply_identity != prepared_supply.supply_identity):
            raise SimulationConfigurationError(
                'Supply configuration changed since the deck was prepared. '
                'Call generate_netlist() again before run_simulator().')
        if declared_supply is not None and supply_request.opaque:
            raise SimulationConfigurationError(
                'The custom DC analysis prevents confirming one supply voltage.')
        if declared_supply is not None and supply_request.sweep is not None:
            raise SimulationConfigurationError(
                'A designated supply DC sweep cannot be confirmed as one voltage.')
        supply_record = None
        if supply_request.configured or declared_supply is not None or supply_request.sweep is not None:
            if (declared_supply is not None
                    and declared_supply != supply_request.voltage):
                raise SimulationConfigurationError(
                    f'Declared supply {declared_supply!r} V is not what this '
                    f'deck runs at ({supply_request.voltage!r} V). Use '
                    'set_supply_voltage(volts) to configure it.')
            try:
                deck_text = Path(files.netlist_path).read_text()
            except OSError as exc:
                raise SimulationConfigurationError(
                    'Prepared deck is unavailable, so the supply it carries '
                    'cannot be confirmed. Call generate_netlist() before '
                    'run_simulator().') from exc
            # Refuses BEFORE the launch: an unconfirmable supply must not
            # become a reported condition, and must not cost a simulation.
            supply_record = observed_supply(supply_request, deck_text)
        if mode == 'new':
            files.claim_new_attempt()
        requests = (getattr(self, '_prepared_device_param_requests', ())
                    if temperature_preparation is not None else ())
        from copy import deepcopy
        producing_prepared = deepcopy(getattr(self, '_prepared_deck_record', None) or {})
        self._run_consumed = True
        from silk.simulation.backend import SimulatorBackend
        if (self.simulator not in ('ngspice', 'xyce') or
                getattr(backend.run, '__func__', None) is not _BUILTIN_BACKEND_RUN):
            _begin_result_replacement(self)
        self._behavioral_mode = False
        from silk.circuit.view_nodes import _capture_addresses
        addresses = _capture_addresses(self) if temperature_preparation is not None else None
        _mark_attempt_stage('simulation')
        from silk.simulation.generation import replacement_callback
        with replacement_callback(lambda: _begin_result_replacement(self)):
            raw = backend.run(
                netlist_path=files.netlist_path,
                results_dir=files.directory,
                debug=self.debug,
                timeout=timeout,
                **launch_options,
            )
        raw._view_addresses = addresses
        raw._device_param_requests = requests
        from silk.simulation._device_requests import warn_missing_requests
        warn_missing_requests(raw)
        # A deck handed in without generate_netlist() was not prepared from
        # this module, so neither proof describes it: corner and supply stay
        # unknown.
        conditions = (
            observed_conditions(request, str(prepared_corner), raw,
                                no_pdk_device=no_pdk_device)
            if temperature_preparation is not None
            else OperatingConditions('silk', self.simulator, None)
        )
        from dataclasses import replace
        if supply_request.opaque:
            conditions = replace(conditions, supply_source=supply_request.source)
        elif supply_record is not None:
            volts, basis, source = supply_record
            covered_sweep = supply_request.sweep if files.completion is not None else None
            conditions = replace(
                conditions, supply_voltage_v=None if supply_request.sweep else volts,
                supply_kind=('sweep' if covered_sweep else 'unknown')
                if supply_request.sweep else 'fixed',
                supply_basis=('configured' if covered_sweep else 'unknown')
                if supply_request.sweep else basis,
                supply_source=source, supply_sweep=covered_sweep)
        elif temperature_preparation is not None and has_no_supply(self):
            # Nothing configured or declared (supply_record is None), and the
            # stated rule proves the circuit has no supply (section 3.3).
            conditions = replace(conditions, supply_kind='none',
                                 supply_basis='proven-no-supply')
        self._raw = raw
        from silk.simulation.generation import register_owner
        register_owner(self, raw)
        self.results_file = self._raw.source_path
        self.dataFile = self._raw.source_path
        self.simResultsAll = self._raw.native
        self.sim_results_loaded = True
        self.operating_conditions = conditions
        # TASK-306: what silk just launched (binary, its source, version),
        # for RunProvenance. Cached per process; never raises.
        from silk.simulation.provenance import launch_record
        self._launch_record = launch_record(backend.simulator_name)
        self._launch_record['prepared'] = producing_prepared
        self._launch_record['normalized_deck_sha256'] = (
            producing_prepared.get('normalized_deck_sha256')
            if temperature_preparation is not None else None)
        self._launch_record['launch_corner_label'] = (
            str(prepared_corner) if temperature_preparation is not None else None)
        # The evidence (ConditionState.NONE on a PDK-free deck) and, apart
        # from it, the corner this launch was requested at: the run's label.
        self._last_sim_corner = conditions.scalar('corner')
        self._last_sim_label = (None if temperature_preparation is None
                                else str(prepared_corner))

    def read_raw_file(self, *, log=None, reason=None):
        """Make the most recent run's results current.

        ``run_simulator`` now runs *and* loads (via ``backend.run`` →
        ``RawResult``); this consumes that result. Call ``run_simulator``
        (or ``simulate``) first.

        When there is no run to consume, the rawfile named by ``load_file()``
        is loaded through ``backend.load_raw``, which gates it for data
        presence: an aborted run's rawfile (the synthetic ``constants`` plot)
        raises ``SimulationAborted`` rather than loading as results. ``log`` (a simulator log path to scrape a reason from, the
        way ``run()`` does) and ``reason`` (already scraped) are optional
        message enrichment for a run that happened elsewhere.

        Raises
        ------
        SimulationAborted
            The named rawfile carries no analysis data rows.
        """
        raw = getattr(self, '_raw', None)
        if raw is None:
            # No run this session — load a pre-existing rawfile by path.
            path = getattr(self, 'dataFile', None) or getattr(self, 'results_file', None)
            if path is None:
                raise SimulationError(
                    'read_raw_file(): no results loaded and no rawfile path set. '
                    'Call run_simulator()/simulate() first, or set dataFile.'
                )
            self.operating_conditions = None
            self.characterization_context = None
            self.sim_results_loaded = False
            self.simResultsAll = None
            self.results_file = None
            # A handed-in rawfile: silk launched nothing (TASK-306).
            self._launch_record = None
            raw = self.backend.load_raw(
                path, simulator=getattr(self, '_raw_simulator_hint', None),
                log_path=log, reason=reason,
            )
            self._raw = raw
        # The cached branch needs no gate: ``_raw`` can only come from
        # ``backend.run()``, which already ran _require_data on it.
        self.simResultsAll = raw.native
        self.sim_results_loaded = True

    def simulate(self, corner=None, auto_track_structure=False, timeout=None):
        """Generate netlist, run the simulator, and load results.

        Uses this Module's ``set_output_options`` setting. New mode prepares
        one persistent child per call; overwrite mode replaces deterministic
        outputs. Save successful ``results_file`` paths for later readback.

        Parameters
        ----------
        corner : str, optional
            Process corner. Defaults to ``pdk.default_corner`` (falls back
            to ``'tt'`` if the PDK doesn't declare one). When the PDK
            declares a non-empty ``corners`` mapping, the resolved corner
            is validated against it and a ``ValueError`` is raised for an
            unknown corner. Legacy/manifest-less PDKs (empty ``corners``)
            skip validation entirely.
        auto_track_structure : bool, optional
            Accepted for backward compatibility; it has no effect.
        timeout : float, optional
            Maximum seconds to wait for the simulator. None means no limit.

        Returns
        -------
        Module
            Returns self for chaining: ``sig = mod.simulate().load_signal('vout')``

        Raises
        ------
        ValueError
            If ``corner`` is not one of the PDK's declared corners.
        silk.exceptions.SimulationError
            If the simulator fails or produces no data.
        """
        _mark_attempt_stage('preflight')
        for axis in ('temperature', 'supply'):
            declaration = _axis_declaration(self, axis)
            if declaration is not None and declaration[0] is not None:
                raise SimulationConfigurationError(
                    f'Module.simulate() requires one {axis}; use a registered test for lists.')
        pdk = self.pdk_details
        if corner is None:
            corner = getattr(pdk, 'default_corner', 'tt')
        corners = getattr(pdk, 'corners', {}) or {}
        if corners and corner not in corners:
            raise ValueError(
                f'unknown corner {corner!r}; available: {sorted(corners)}')
        backend = self.backend
        if backend.runs_module:
            # No deck (TASK-448.1): the backend runs this Module itself.
            _simulate_module(self, backend, corner)
            return self
        self.generate_netlist(corner=corner)
        self.run_simulator(auto_track_structure=auto_track_structure, timeout=timeout)
        self.read_raw_file()
        # Publish the corner only after this attempt produced usable results:
        # the requested one when it reached the deck, ConditionState.NONE
        # when the deck instantiates no PDK device (design section 3.1).
        conditions = self.operating_conditions
        self._last_sim_corner = (
            conditions.scalar('corner')
            if conditions is not None and conditions.corner_kind == 'none'
            else corner)
        return self

    def view(self, *, out_path=None, open=True, serve=False, port=0,
             title=None, signals=None, exclude=None, max_depth=None,
             inline=False, context=None):
        """Open the interactive waveform explorer on this module's last run.

        The one-liner after a run: ``tb.simulate(); tb.view()`` writes
        ``view.html`` beside the rawfile with sidecar files and opens it;
        ``inline=True`` writes a single self-contained file.
        Nothing else is required — no output path, no ``CircuitReport``.

        Parameters
        ----------
        out_path : str or os.PathLike, optional
            Destination ``.html`` path. Defaults to ``view.html`` beside the
            rawfile (``self.results_file``).
        open : bool, default True
            Open the emitted page in the default browser. Set ``open=False``
            to emit only (e.g. in a headless run) and use the returned path.
        serve : bool, default False
            Instead of a one-shot ``file://`` page, serve the output over
            ``http://127.0.0.1`` and re-emit whenever the rawfile changes
            (live reload while you iterate). **Blocks until Ctrl-C.** Not
            combinable with *inline*.
        port : int, default 0
            Port for *serve* (0 = OS-assigned, printed on stdout).
        title : str, optional
            Page title. Defaults to the rawfile's own title/basename.
        signals, exclude, max_depth : optional
            Selection filters, identical in form to the CLI's
            ``--signals`` / ``--exclude`` / ``--max-depth`` and to
            ``silk.signals.view.emit_view`` (case-insensitive exact /
            glob / hierarchy-run selectors). An array index is written
            ``[k]`` or ``<k>`` and square brackets are never a glob
            character class (see ``docs/simulator-save-list.md``).
        inline : bool, default False
            Emit a single embedded-data file with no sidecars (emailable /
            committable), at the cost of live auto-reload.
        context : dict, optional
            Run-provenance overlay shown in the page's info popover; shape
            per ``silk.signals.view.emit_view``. For auto-populated
            provenance (test name, corner, PDK, spec outcomes) route through
            :meth:`silk.circuit.CircuitReport.add_waveform_view` instead.

        Returns
        -------
        str
            The path to the emitted HTML page.

        Raises
        ------
        silk.signals.view.template.ViewError
            If this module has not been simulated yet (no ``results_file``),
            or *serve* is combined with *inline*.

        Notes
        -----
        The signal list shows the ``[k]`` spelling of array instances as its
        label (``labels='designer'``); raw rawfile names stay the identity.
        Imports ``silk.signals.view`` lazily, so a module that never calls
        this pulls in none of it — the same contract as
        :meth:`silk.circuit.CircuitReport.add_waveform_view`, which this
        method complements: use ``view()`` for a quick look during a run, and
        ``report.add_waveform_view()`` when you want the page linked from a
        report with auto-discovered provenance.
        """
        import os
        import webbrowser
        from silk.signals.view import emit_view, ViewError
        from silk.signals.view.serve import run_server

        source = getattr(self, 'results_file', None)
        if not source:
            raise ViewError(
                "view(): no results to show — this module has not been "
                "simulated yet (results_file is None). Call simulate() (or "
                "run_simulator()) first, then view()."
            )
        if serve and inline:
            raise ViewError(
                "view(): serve=True is not combinable with inline=True — an "
                "inline page has no sidecar to re-emit and watch. Use one or "
                "the other."
            )
        if out_path is None:
            results_dir = os.path.dirname(os.path.abspath(source)) or '.'
            out_path = os.path.join(results_dir, 'view.html')

        from silk.circuit.view_nodes import _view_node_context
        from silk.signals.rawfile import read_rawfile
        from silk.simulation.generation import read_retained
        checked_source = read_retained(self) if getattr(self, '_raw', None) is not None else source
        if serve and getattr(getattr(self, '_raw', None), '_generation', None) is not None:
            raise ViewError('Managed result views retain one generation and cannot watch a mutable public path. Use serve=False, or serve the rawfile path explicitly.')
        nodes = _view_node_context(self, checked_source) if getattr(self, '_raw', None) is not None else None

        if serve:
            # One-shot emit first so a page exists before the browser opens,
            # then hand off to the watch/re-emit server (blocks until Ctrl-C).
            emit_kwargs = dict(title=title, signals=signals, exclude=exclude,
                               max_depth=max_depth, context=context,
                               labels='designer', **({'node_context': nodes} if nodes else {}))
            result = emit_view(checked_source, out_path, **emit_kwargs)
            run_server(out_path, source, port=port, open_browser=open,
                       emit_kwargs=emit_kwargs)
            return result.html_path

        result = emit_view(checked_source, out_path, title=title, signals=signals,
                           exclude=exclude, max_depth=max_depth, inline=inline,
                           context=context, labels='designer',
                           **({'node_context': nodes} if nodes else {}))
        if open:
            webbrowser.open('file://' + os.path.abspath(result.html_path))
        return result.html_path


# Custom launch providers retain the legacy module-level replacement boundary.
from silk.simulation.backend import SimulatorBackend as _SimulatorBackend
_BUILTIN_BACKEND_RUN = _SimulatorBackend.run
