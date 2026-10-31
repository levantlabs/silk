"""Type-checker-only declaration of the base-owned :class:`Module` state that
the per-subsystem *part* mixins read and write.

Each part does::

    if TYPE_CHECKING:
        from silk.circuit._module_protocol import _ModuleAttrs
        _Base = _ModuleAttrs
    else:
        _Base = object

    class _FooMethods(_Base):
        ...

so that:

- **at runtime** the part inherits plain ``object`` — no state, no annotations,
  and therefore no ``__annotations__`` dunder leaking into ``dir(Module)`` and
  no PEP 563 signature stringification (both would break the API golden);
- **for type checkers** the part inherits ``_ModuleAttrs``, so ``self.elem_dict``
  et al. resolve to their declared types.

This module is never imported at runtime, so it adds zero runtime coupling.
Grow it as parts surface more base-owned attributes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


class _ModuleAttrs(Protocol):
    # --- element graph (aliased from CircuitGraph) ---
    _element_mgr: Any
    elem_dict: Dict[str, Any]
    net_dict: Dict[str, Any]
    module_dict: Dict[str, Any]
    module_names: List[str]

    # --- identity / io ---
    module_name: Optional[str]
    netlistname: str
    dir_folder: str
    pdk_name: str
    pdk_details: Any
    device_name: str
    pin_list: Any
    debug: bool
    save_device_currents: bool
    save_device_params: bool
    omit_device_internal_nodes: bool

    # --- sources / passives / analyses ---
    passives_dict: Dict[str, Any]
    sig_sources_dict: Dict[str, Any]
    sim_analysis_dict: Dict[str, Any]
    simulator_options: Any
    custom_lines: List[str]

    # --- results / simulation ---
    _run_files: Any
    _output_mode: str
    _run_preparation: Any
    _temperature_preparation: Any
    _deck_preparation: Any
    _temperature_basis: str
    _temperature_explicit: bool
    _supply_preparation: Any
    _supply_voltage: Any
    _supply_source: Any
    _supply_explicit: bool
    _declared_supply: Any
    operating_conditions: Any
    _run_consumed: bool
    _raw: Any
    simResultsAll: Any
    results_file: Any
    dataFile: Any
    output_file_available: Any   # property
    sim_results_loaded: Any      # property
    _simulator: Any
    _simulator_explicit: bool
    _netlister: Any
    _netlister_sim: Any
    netlister: Any       # property (defined in _SimMethods)
    simulator: Any       # property (defined in _SimMethods)
    backend: Any         # property (defined in _SimMethods)
    circuit_name: Any

    # --- behavioral model ---
    _behavioral_model: Any
    _behavioral_params: Any
    _behavioral_form: Any
    _behavioral_execution: Any
    _behavioral_mode: Any
    _behavioral_t: Any

    # --- domains / buses / connectivity ---
    _domains: Any
    _buses: Dict[str, Any]
    _bus_handles: Dict[str, Any]
    _no_connects: Any
    half_circuits: Any

    # --- testbenches ---
    _tests: Dict[str, Any]

    # --- specs ---
    _specs: List[Any]

    # --- db / tracking ---
    _tracker: Any
    _db_path: Optional[str]
    _db_version: Any
    _db_stage: Any
    _db_created_by: Any
