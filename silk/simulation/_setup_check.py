"""The static admission checks of a simulation setup, shared by the public
:func:`validate_simulation_setup` and silk's own runs.

Everything here is read-only: no model, waveform or selector callable is
invoked, no result, note or provenance is written, and nothing scales with the
transient length. A run calls the same pieces at its existing admission sites,
so the public call and the run raise the same class and message.
"""
from silk.exceptions import NotSupportedError


def private_view(module, memo=None):
    """A private copy of the graph structures the static checks read.

    Reading ``elem_dict``/``net_dict``/``module_dict`` refreshes placement
    pins, bindings and definition records in place, so checking the live graph
    would write to it. This view owns every dictionary, placement entry, pin,
    net, binding and definition record those refreshes touch, down the whole
    hierarchy (a repeated child stays one shared view). Sources, devices,
    models and waveforms are shared and only read. It is built without
    pickling or live-graph registration, and is never published.
    """
    from copy import copy
    from silk.circuit.circuit_elements import ElementType
    memo = {} if memo is None else memo
    if id(module) in memo:
        return memo[id(module)]
    view = object.__new__(type(module))
    view.__dict__.update(module.__dict__)
    memo[id(module)] = view
    graph = copy(module._element_mgr)
    graph.__dict__ = {key: copy(value) if isinstance(value, (dict, list, set)) else value
                      for key, value in vars(graph).items()}
    view._element_mgr = graph
    graph._module_ref = view
    view.simulator_options = dict(module.simulator_options)
    graph.net_dict = {name: copy(net) for name, net in graph.net_dict.items()}
    for net in graph.net_dict.values():
        net.module = view
        net.pins = [copy(binding) for binding in net.pins]
    graph.elem_dict = {name: dict(entry) for name, entry in graph.elem_dict.items()}
    for entry in graph.elem_dict.values():
        if entry['type'] == ElementType.MODULE:
            entry['device'] = private_view(entry['device'], memo)
        if 'pin_list' in entry:
            entry['pin_list'] = {name: copy(pin) for name, pin in entry['pin_list'].items()}
    graph.module_dict = {
        name: dict(entry, device=private_view(entry['device'], memo),
                   **({'_source_ids': set(entry['_source_ids'])}
                      if '_source_ids' in entry else {}))
        for name, entry in graph.module_dict.items()}
    return view


def refuse_ambient_temperature(module):
    """The behavioral engine applies no ambient ``TEMP``."""
    if module.simulator_options.get('temp') is not None:
        raise NotSupportedError(
            f"The {module.backend.simulator_name!r} backend does not apply "
            f"ambient TEMP; clear set_temperature(None).")


def require_static_setup(module):
    """Ground-interface and connectivity checks, then behavioral static refusals.

    Order matches a behavioral run: interfaces and ground connectivity, switches, analyses,
    then the engine's read-only preparation (``_flatten(report=False)``) and
    the DC sweep's own static refusals. Both engines refuse formal ``0`` and
    case-insensitive ``gnd`` with the shared ``CircuitError`` diagnostic.
    Modeled leaves expose their interfaces but replace their contents.
    Other SPICE refusals are decided by the deck or by the simulator.
    """
    from silk.simulation.topology import require_ground_connections
    require_ground_connections(module)
    if module.simulator != 'behavioral':
        return
    from silk.simulation.analysis import DC
    from silk.simulation.behavioral import BehavioralEngine
    from silk.simulation.behavioral_backend import (
        _declared_analyses, _refuse_inapplicable_switches,
    )
    _refuse_inapplicable_switches(module)
    analyses = _declared_analyses(module)
    engine = BehavioralEngine()
    # report=False: no approximation notices and no change to the module's
    # recorded notes; a later real run emits them once.
    _, sources = engine._flatten(module, report=False)
    for analysis in analyses:
        if isinstance(analysis, DC):
            engine._dc_target(sources, analysis)


def validate_simulation_setup(module):
    """Validate the assembled simulation setup for its selected simulator without running it or changing it. Returns None if no static refusal is found; this does not guarantee convergence or successful execution.

    *module* is the assembled top-level module with its effective simulator
    and settings. The first static refusal is raised with the exception class
    and message the run raises for the same input; later checks are not
    attempted, because they rely on what earlier ones established.

    Checks covered, in order:

    * behavioral only: ambient temperature (``set_temperature``) refuses;
    * a configured supply voltage must resolve to a source
      (``SimulationConfigurationError``);
    * the existing topology walk checks each entered interface, including
      modeled-leaf interfaces, for formal ``0`` or case-insensitive ``gnd``
      (``CircuitError``), then checks source ground connectivity
      (``SimulationError``). Rename the formal to ``vss`` and wire it to
      parent ground; actual ``0``/``gnd`` net handling is unchanged;
    * behavioral only, the engine's static refusals (``NotSupportedError`` or
      ``SimulationError``): custom SPICE text and device-capture switches,
      analyses other than transient and DC (and none declared), transistors
      with no model above them, empty blocks without a model, unsupported
      sources, series capacitors, resistive dividers and ambiguous R/L paths,
      static source-reference refusals, and a DC sweep of ambient temperature
      or of a source that does not exist.

    Not covered: installed PDK models, compiled plugins, deck freshness or
    prepared-deck state, refusals decided during a run (they depend on what
    models return, such as the ports engine's conflicts), and electrical
    solvability. A clean return is not a certificate.

    Nothing is changed: source and ``Variable`` values, results, notes,
    provenance, request inventories, run state and output files stay as they
    were, and no stateful model, waveform or selector callable is called.
    """
    check_view(private_view(module))


def check_view(module):
    """The checks of :func:`validate_simulation_setup` on an already private view."""
    if module.simulator == 'behavioral':
        refuse_ambient_temperature(module)
    from silk.simulation.conditions import prepare_supply
    prepare_supply(module, apply=False)
    require_static_setup(module)
