"""SPICE netlist emission methods for :class:`~silk.circuit.modules.Module`.

Split out of ``modules.py`` as a per-subsystem **part**. ``_NetlistMethods`` is a pure mixin: no state, no separate identity. The heavy lifting is delegated to ``self.netlister`` (a per-backend netlister object); these three are the only netlist methods that live on ``Module`` itself.
"""

from silk.circuit.emitted_names import emitted_submodule_name
from silk.circuit.test_placement import execution_location

from typing import TYPE_CHECKING
from functools import wraps
from silk.circuit.module_placement import with_definition_scope
from silk.circuit.wiring_consistency import with_wiring_scope
from silk.simulation._ngspice_compat import _with_ngspice_version
from silk.exceptions import NotASubcircuitError, CircuitError

if TYPE_CHECKING:  # pragma: no cover
    from silk.circuit._module_protocol import _ModuleAttrs
    _Base = _ModuleAttrs
else:
    _Base = object


# Recorded as the freshness preparation while a generation is in flight. A
# generation that raises leaves it behind, and a launch refuses on it: the deck
# on disk is then from an earlier configuration and nothing vouches for it.
# Deliberately a plain string so a pickled Module (the fault runner) round-trips.
_GENERATION_INCOMPLETE = 'silk:generation-incomplete'


def describe_module(module) -> str:
    """A name for *module* that an error message can say out loud: its
    subcircuit name if it has one, else the circuit name, else the deck file
    it emits. A top-level Module has no ``module_name`` by definition, so the
    fallbacks are what identify it in the refusals below."""
    return (getattr(module, 'module_name', None)
            or getattr(module, 'circuit_name', None)
            or str(getattr(module, 'netlistname', '') or '').split('/')[-1]
            or '<unnamed module>')


def require_subcircuit(module, context):
    """Refuse *module* as a subcircuit when it has no ``module_name``.

    The shared emission-time diagnostic for a missing subcircuit name. ``None`` used
    to reach a ``str + None`` concatenation here and raise a bare
    ``TypeError`` naming neither the module nor the mistake; name both, and
    say how to fix it. *context* is the caller's half of the sentence, e.g.
    ``"Cannot instantiate 'u1'"``.
    """
    if getattr(module, 'module_name', None) is None:
        raise NotASubcircuitError(
            f"{context}: {describe_module(module)!r} has no module_name, so "
            f"it is a top-level circuit and there is no .subckt to reference. "
            f"Call define_module_name('<name>') on it (and declare its ports "
            f"with define_module_ports({{...}})) to make it a subcircuit, or "
            f"emit it as a standalone deck with generate_netlist().")


def require_declared_ports(module, context):
    """Refuse a named *module* whose port table was never declared.

    ``pin_list`` is ``None`` (not empty) until ``define_module_ports`` runs,
    and the netlister used to meet that ``None`` as a bare ``AttributeError``. No silent default to zero ports: say so. *context* is the
    caller's half of the sentence.
    """
    if getattr(module, 'pin_list', None) is None:
        raise NotASubcircuitError(
            f"{context}: {describe_module(module)!r} has no declared ports. "
            f"Call define_module_ports({{...}}) on it before use; pass {{}} "
            f"for a block that has no pins.")


def require_composable_child(module, elem_name, *, require_ports=False):
    """Refuse *module* as a child of another Module before ``add_element``
    accepts it.

    ``add_module_under_test`` and its testbench sibling
    ``add_module`` already refuse a module_name-less module where the mistake
    is made. ``Module.add_element`` — the general composition API — did not:
    it accepted it and left the refusal to surface much later, at the
    parent's ``generate_netlist()``.

    Names and port declarations are required before placement so wiring can
    be validated at the call site. The placed definition remains live.

    *require_ports*, passed only when ``add_element`` was actually given
    wiring to bind, also refuses a module with a ``module_name`` but no
    declared ports at all (``pin_list`` is still ``None``): passing wiring
    for such a child used to reach ``connect_element``'s ``pinlist.keys()``
    and raise a bare ``AttributeError`` naming neither the module nor the
    mistake. A portless module added with no wiring is a legitimate zero-pin
    (e.g. behavioral-model) subcircuit — pinned by
    ``tests/specs/test_testbench_integration.py::test_composition_copies_protected_specs_and_runtime_relationships``,
    which builds exactly that and wires it with ``nets={}``.

    ``Module.add_submodule`` deliberately does NOT call this because it also
    supports purely organizational parent/child registration for the recursive
    test-walking tree; see its docstring for this distinction.
    """
    if getattr(module, 'module_name', None) is None:
        raise NotASubcircuitError(
            f"Cannot add {elem_name!r}: {describe_module(module)!r} has no "
            f"module_name, so it is a top-level circuit and there is no "
            f".subckt to reference. A child must be a subcircuit — call "
            f"define_module_name('<name>') on it (and declare its ports "
            f"with define_module_ports({{...}})) before adding it as a "
            f"child.")
    if require_ports and getattr(module, 'pin_list', None) is None:
        raise NotASubcircuitError(
            f"Cannot add {elem_name!r}: {describe_module(module)!r} has no "
            f"declared ports. Call define_module_ports({{...}}) on it before "
            f"adding it as a child.")


def _prepare_run_files(module, *, temperature=False):
    """Select one concrete location before emission without changing the root.

    *temperature* says this run configures a fixed ambient TEMP, so a backend
    that confirms temperature through a written record (Xyce) gets a path for
    it. Backends that scrape their own log (ngspice) ignore it.
    """
    # A failed regeneration must never leave an earlier new-mode deck reusable.
    # Keep its provenance so switching modes cannot bypass that protection.
    module._run_consumed = True
    # Same reasoning for the freshness record (TASK-160). Clearing it to None
    # would be wrong: None means "no preparation, no opinion", which is how an
    # externally supplied deck is treated, and a failed regeneration would then
    # let the previous deck launch unchallenged. Mark it instead.
    module._prepared_device_param_requests = ()
    module._deck_preparation = _GENERATION_INCOMPLETE
    directory, filename, mode = execution_location(module)
    files = module.backend._resolve_run_files(
        filename, directory,
        analyses=list(module.sim_analysis_dict.values()),
        temperature=temperature,
    )
    # Output paths are written into the deck (ngspice's write line, Xyce's
    # .print file=): refuse a line break before any folder is made (TASK-593).
    from silk.utils.deck_text import refuse_line_breaks
    refuse_line_breaks(files.directory, who='output directory (dir_folder)')
    if mode == 'new':
        files = files.prepare_new()
    else:
        files.prepare()
    return files


def _resolve_terminal_current_method(module):
    """Terminal-current method for this run: the per-Module override
    (``module.terminal_current_method``) if set, else the active simulator's
    YAML default (``'savecurrents'`` when unspecified). Module-level so it does
    not enlarge ``dir(Module)`` (API golden)."""
    override = getattr(module, 'terminal_current_method', None)
    if override is not None:
        return override
    try:
        from silk.simulators import load_simulator_config
        cfg = load_simulator_config(module.simulator)
        return getattr(cfg, 'terminal_current_method', 'savecurrents')
    except Exception:
        return 'savecurrents'


def _validate_authored_nets(module):
    """Recheck imported/mutated graph names before emission.

    Each interface's shared ground-port refusal precedes its generic formal
    name validation, including nested definitions and placement pin records.
    Actual net values retain their ground exception.
    """
    from silk.elements.base import AnalogElement, validate_net_token
    from silk.circuit.emitted_names import validate_instance_name
    from silk.circuit.ports import ElementType
    from silk.circuit.ground_port import _require_entered_ground_interfaces

    seen = set()

    def pins(pin_list, who):
        for name, pin in (pin_list or {}).items():
            validate_net_token(name, who=f'{who} port', kind='port')
            validate_net_token(pin.name, who=f'{who} port', kind='port')
            if pin.net is not None:
                validate_net_token(pin.net, who=f'{who} pin {name!r}')

    def walk(mod, path):
        if id(mod) in seen:
            return
        seen.add(id(mod))
        _require_entered_ground_interfaces(mod)
        for name, net in mod.net_dict.items():
            validate_net_token(name, who=f'{path} net', kind='net')
            validate_net_token(net.name, who=f'{path} net', kind='net')
        pins(mod.pin_list, path)
        for name, entry in mod.elem_dict.items():
            validate_instance_name(name)
            device = entry['device']
            who = f'{path} instance {name!r}'
            if entry['type'] == ElementType.MODULE:
                pins(entry.get('pin_list', device.pin_list), who)
                walk(device, who)
            else:
                if hasattr(device, 'name'):
                    validate_instance_name(device.name)
                for pin, net in getattr(device, 'nets', {}).items():
                    validate_net_token(net, who=f'{who} pin {pin!r}')
                pins(getattr(device, 'pin_list', {}), who)
        for registry in (mod.sig_sources_dict, mod.passives_dict):
            for name, device in registry.items():
                validate_instance_name(name)
                validate_instance_name(device.name)
                who = f'{path} instance {name!r}'
                if isinstance(device, AnalogElement):
                    # Ideal elements (including add_passive registrations) use
                    # their declared pin names, not legacy pos/neg attributes.
                    terminals = device.nets.items()
                else:
                    terminals = ((pin, getattr(device, pin)) for pin in
                                 getattr(device, '_node_terminals', ('pos', 'neg')))
                for terminal, net in terminals:
                    validate_net_token(net, who=f'{who} terminal {terminal!r}')
        for definition in mod.module_dict.values():
            walk(definition['device'], path)

    walk(module, f'Module {module.module_name or "<top>"!r}')


def _with_authored_name_validation(fn):
    """Validate interfaces before emission's wiring views synchronize them."""
    @wraps(fn)
    def checked(module, *args, **kwargs):
        _validate_authored_nets(module)
        return fn(module, *args, **kwargs)
    return checked


@_with_authored_name_validation
@with_wiring_scope
def _emit_deck_text(module, corner, files, circuit_name=None):
    """Build the top-level deck text for *corner* into *files*.

    THE ONE emission call site. ``generate_netlist`` writes what this returns;
    :func:`_require_fresh_deck` re-runs it at launch to learn whether the
    prepared deck still describes the live configuration. Sharing
    one call site is what makes the freshness comparison complete: a new
    emission input cannot reach the deck without also reaching the check.

    Side-effect free on disk. ``silk/circuit/spice_netlister.py`` and the Xyce
    netlister contain no filesystem calls at all, and
    ``generate_complete_netlist`` resets its own accumulators on entry, so a
    second call is idempotent. It does NOT allocate run files -- *files* is
    supplied by the caller, because :func:`_prepare_run_files` is the
    side-effecting half (it consumes the previous preparation and creates a
    new-mode directory) and must run exactly once, at generation.

    *circuit_name* defaults to the live derivation and is passed explicitly by
    the freshness check, which replays the name the prepared deck was titled
    with -- see :func:`_require_fresh_deck`.
    """
    if module.omit_device_internal_nodes:
        from silk.simulation._internal_nodes import require_filterable
        certificate = require_filterable(module, corner)
    if circuit_name is None:
        circuit_name = module.circuit_name or execution_location(module)[1].split('/')[-1]
    from silk.circuit.node_paths import default_output_nodes
    text = module.netlister.generate_complete_netlist(
        circuit_name=circuit_name,
        corner=corner,
        module_dict=module.module_dict,
        elem_dict=module.elem_dict,
        sig_sources_dict=module.sig_sources_dict,
        sig_source_details=None,
        sim_analysis_dict=module.sim_analysis_dict,
        custom_lines=module.custom_lines,
        dir_folder=files.directory,
        run_files=files,
        passives_dict=module.passives_dict,
        simulator=module.simulator,
        simulator_options=module.simulator_options,
        output_vars=default_output_nodes(module),
        save_device_currents=module.save_device_currents,
        save_list=_resolve_save_list(module),
        terminal_current_method=_resolve_terminal_current_method(module),
        save_device_params=module.save_device_params,
        device_probe_requests=getattr(module, '_pending_device_probes', ()),
        current_probe_names=getattr(module, '_pending_current_probes', ()),
        current_probe_analyses=getattr(module, '_pending_current_probe_analyses', None),
        mc_sample=getattr(module, '_mc_sample', None),
    )
    if module.omit_device_internal_nodes:
        from silk.simulation._ngspice_compat import NgspiceCompatibility
        title, body = text.split('\n', 1)
        text = (title + '\n' +
                f'*** omit_device_internal_nodes=True; {certificate}; '
                f'ngspice {NgspiceCompatibility().version}\n' + body)
        # Only built-in emission reaches here after content admission. Keep
        # additive current/parameter operands on the same batch save line.
        text = ''.join(
            line.replace('save all', 'save nointernals', 1)
            if line.startswith(('save all', '.save all')) else line
            for line in text.splitlines(keepends=True))
    return text


def _quiet_deck_text(module, corner, files, circuit_name):
    """:func:`_emit_deck_text` with its advisory output suppressed.

    Emission is allowed to talk to the user: ``_resolve_save_list`` prints how
    many nodes survive the list, a save list alongside a Noise analysis prints
    a note, and ``save_device_currents`` without a simulator directive warns.
    Those belong to the generation the caller asked for, not to a guard that
    runs on every launch, so the freshness re-emission says nothing.
    """
    import contextlib
    import io
    import warnings

    sink = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            return _emit_deck_text(module, corner, files, circuit_name)


def _first_difference(prepared, current):
    """Describe the first place where the two decks part company.

    Aligned with :class:`difflib.SequenceMatcher` rather than compared index
    by index, so an inserted line names the inserted line instead of the blank
    that everything after it shifted onto.

    Returns None when there is nothing to compare against (the prepared deck
    is unreadable), so the refusal degrades to naming the fact of the change
    rather than inventing a location.
    """
    if prepared is None:
        return None
    import difflib

    was_lines = prepared.splitlines()
    now_lines = current.splitlines()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, was_lines, now_lines, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        where = f'first change at deck line {j1 + 1}'
        if tag == 'insert':
            return f'{where}: the deck now adds {now_lines[j1].strip()!r}'
        if tag == 'delete':
            return f'{where}: {was_lines[i1].strip()!r} is no longer emitted'
        return (f'{where}: prepared {was_lines[i1].strip()!r}, '
                f'now {now_lines[j1].strip()!r}')
    return None


def _require_fresh_deck(module):
    """Refuse to launch a deck that no longer matches the configuration.

    Silk refuses to relaunch a consumed preparation, and Silk refuses
    when the temperature request or the ON-DISK deck digest moved. Neither
    sees a configuration change made after ``generate_netlist()``: the file on
    disk is untouched, so its digest still matches, and the previous deck
    launches while every downstream consumer attributes the result to the new
    configuration. This closes that hole generally, by re-emitting the deck
    from the live configuration and comparing it against what was emitted.

    "Deck-affecting" therefore means exactly what the netlister renders --
    sources, elements, passives, submodules, analyses, custom lines, the
    simulator and its options, the save list, the terminal-current method, the
    corner recorded at generation and the prepared file paths. Configuration
    that never reaches the deck (``debug``, ``timeout``) is deliberately not
    deck-affecting and never refuses.

    One deliberate exclusion: the deck's title comment, which is derived from
    the deck file name ``netlistname``. A prepared run keeps its own files when
    the configured root moves afterwards -- that is the documented contract of
    a consumed preparation, and the deck that runs is still exactly the deck that was
    prepared. The guard therefore replays the recorded title and compares
    circuit configuration, not the configured output name. New mode's own
    refusal already owns path changes.

    Silent when there is no preparation to check: a caller that supplies an
    existing deck by path, or drives a backend or runner directly, has no
    recorded configuration and gets no opinion from here.
    """
    from silk.exceptions import DeckTextError, SimulationConfigurationError
    from hashlib import sha256

    preparation = getattr(module, '_deck_preparation', None)
    if preparation is None:
        return
    if preparation == _GENERATION_INCOMPLETE:
        raise SimulationConfigurationError(
            'generate_netlist() did not complete, so the deck on disk is from '
            'an earlier configuration. Call generate_netlist() again (or '
            'simulate()) to prepare the next run before calling '
            'run_simulator().')
    corner, files, circuit_name, digest = preparation
    try:
        current = _quiet_deck_text(module, corner, files, circuit_name)
    except DeckTextError:
        # A forbidden line break keeps its own type and message (TASK-593);
        # every other emission failure is a stale configuration.
        raise
    except Exception as exc:
        raise SimulationConfigurationError(
            'Circuit configuration changed after generate_netlist() and no '
            f'longer emits a deck ({type(exc).__name__}: {exc}). Call '
            'generate_netlist() again (or simulate()) to prepare the next run '
            'before calling run_simulator().') from exc
    if sha256(current.encode()).digest() == digest:
        # Re-emission has proved these exact saves belong to the prepared
        # deck, including a retry after failed-result invalidation.
        module._prepared_device_param_requests = tuple(
            getattr(module.netlister, '_device_param_requests', ()))
        return
    try:
        with open(files.netlist_path) as f:
            prepared = f.read()
    except OSError:
        prepared = None
    detail = _first_difference(prepared, current)
    raise SimulationConfigurationError(
        'Circuit configuration changed after generate_netlist(): the deck '
        'that would be generated now differs from the prepared one'
        + (f' ({detail})' if detail else '')
        + '. Call generate_netlist() again (or simulate()) to prepare the '
          'next run before calling run_simulator().')


def _reject_model_only_blocks(module):
    """Refuse a SPICE deck that instantiates a block with a behavioural model
    and no SPICE content.

    Such a block emits an empty ``.subckt``; the simulator runs it and the
    metric then fails with "not found in rawfile", which does not say why.
    A twin (model plus elements, sources, passives or children) has SPICE
    content and is unaffected. Walks the instantiated tree.
    """
    from silk.circuit.ports import ElementType
    from silk.exceptions import NotSupportedError

    def walk(mod, path):
        for inst_name, entry in mod.elem_dict.items():
            if entry['type'] != ElementType.MODULE:
                continue
            sub = entry['device']
            here = f'{path}.{inst_name}' if path else inst_name
            has_content = bool(sub.elem_dict or sub.sig_sources_dict
                               or sub.passives_dict or sub.module_dict)
            if sub._behavioral_model is not None and not has_content:
                raise NotSupportedError(
                    f"Block '{here}' (module '{sub.module_name}') has a "
                    f"behavioural model and no SPICE content, so a SPICE "
                    f"netlist would instantiate an empty .subckt. Set "
                    f"simulator = 'behavioral' to run it from its model, or "
                    f"give the block SPICE elements.")
            walk(sub, here)

    walk(module, '')


def _reject_unportable_ground_ports(module):
    """Apply the shared ``0``/case-insensitive ``gnd`` interface refusal.

    This is backend-independent and precedes rendering and file writes.
    The authored-name walk also applies it to nested definitions and pins
    before generic formal-name checks, including pure deck rendering.
    """
    from silk.circuit.ground_port import _require_portable_ground_ports
    _require_portable_ground_ports(module)


def _mismatch_off_line(module, netlister):
    """``.param <switch>=0`` for a block with mismatch off, in a mismatch or
    both Monte Carlo sample deck; otherwise the empty string."""
    sample = getattr(netlister, '_mc_sample', None)
    if (sample is None or module.mismatch_enabled
            or sample.variation not in ('mismatch', 'both')):
        return ''
    stat = getattr(netlister.pdk_details, 'statistical', None)
    if stat is None:
        return ''
    from silk.utils.deck_text import card_text
    return f".param {card_text(stat.mismatch_switch, who='PDK mismatch switch')}=0\n"


def _subckt_text(module):
    """The ``.subckt`` block for *module*: nested definitions, header, device
    instances, sources, passives, custom lines, footer.

    The one place that sequence lives. ``generate_netlist(subckt=True)``
    wraps it in the emission-time contracts (the ``gnd``-port guard above,
    fault-marker realisation); the subcircuit-dedup cross-check in
    :mod:`silk.circuit.circuit_elements` calls it bare, because that check
    compares two bodies' text and must not turn a design defect the lint
    catalog already reports (STRUCT-012) into an ``add_element`` failure.

    The terminal-current method + shared ammeter-wrapper accumulator are read
    from this module's netlister (set by the parent's
    ``generate_module_definitions``) and propagated one level deeper, so
    in-module devices are wrapped and their (deduped) wrapper defs bubble up
    to the top level.
    """
    netlister = module.netlister
    method = getattr(netlister, '_terminal_current_method', None)
    wrappers = getattr(netlister, '_ammeter_wrappers', None)
    emitted = getattr(netlister, '_emitted_subckts', None)
    str_module = netlister.generate_module_definitions(
        module.module_dict, method, wrappers, emitted,
        mc_sample=getattr(netlister, '_mc_sample', None))
    str_module += module.generate_subckt(0)
    str_module += _mismatch_off_line(module, netlister)
    str_module += netlister.generate_device_instances(module.elem_dict)
    str_module += netlister.generate_sources(module.sig_sources_dict)
    str_module += netlister.generate_passives(module.passives_dict)
    str_module += netlister.generate_custom_lines(module.custom_lines)
    str_module += module.generate_subckt(1)
    return str_module


def _save_list_inventory(module):
    """Every node-voltage path this design can put in a rawfile, in silk's
    dot-separated (ngspice) spelling: top-level nets plus subckt-internal
    nets as ``x<instance>.<net>``, recursively.

    Two exclusions, both structural:

    * a subckt PORT net never gets its own hierarchical vector — the
      simulator aliases it to the parent net it is bound to (the same rule
      :meth:`SubcircuitInstance._external_path` implements for readback), so
      only the parent's name is listed;
    * the ground node ``0`` has no vector.

    Device-internal parasitic nodes (``…#body``) are absent by construction —
    they are the simulator's, not silk's — which is exactly what makes the
    enumerated list a "design signals only" filter (the measured 70% of a
    real rawfile; backlog 2026-08-12-simulator-save-list.md).
    """
    from silk.circuit.ports import ElementType
    paths = []

    def walk(mod, prefix, is_sub):
        ports = set(mod.pin_list or {}) if is_sub else set()
        for net in mod.net_dict:
            if net == '0' or net in ports:
                continue
            paths.append(prefix + net)
        for source in mod.sig_sources_dict.values():
            layout = getattr(source, 'physical_layout', None)
            node = layout().internal_node if layout is not None else None
            if node is not None:
                paths.append(prefix + node)
        for name, entry in (mod.elem_dict or {}).items():
            if entry.get('type') == ElementType.MODULE:
                walk(entry['device'], prefix + emitted_submodule_name(name) + '.', True)

    walk(module, '', False)
    return paths


def _resolve_save_list(module):
    """Expand the module's stored save-list spec against the design's net
    inventory. Returns None (no list set — emit ``save all``) or the list of
    dot-separated node paths to save.

    Fail-loud is silk's ONLY typo defense here: verified on ngspice-42 that
    an unknown node in a ``save`` line is dropped SILENTLY (no warning, rc 0),
    so a selector that matches nothing must die at netlist time, not at
    readback after the simulation has been paid for.
    """
    spec = getattr(module, '_save_list', None)
    if spec is None:
        return None
    from silk.circuit.node_paths import select_save_nodes
    from silk.signals.view.template import ViewError
    inventory = _save_list_inventory(module)
    try:
        selected = select_save_nodes(
            module, inventory, signals=spec['signals'], exclude=spec['exclude'],
            max_depth=spec['max_depth'])
    except ViewError as e:
        raise CircuitError(f'set_save_list: {e}') from e
    import sys
    print(f'save list active: rawfile will hold {len(selected)} of '
          f'{len(inventory)} node voltages. Tools that assume a complete '
          f'rawfile (waveform explorer, annotate_currents, device-stress '
          f'lint) see only these.', file=sys.stderr)
    return selected


class _NetlistMethods(_Base):
    """SPICE netlist emission methods mixed into ``Module``. Holds no state."""

    def set_save_list(self, signals=None, *, exclude=None, max_depth=None):
        """Restrict which node voltages the simulator writes to the rawfile.

        **Opt-in, for runs whose ``save all`` rawfile is too big to store** —
        this is the only knob that shrinks the rawfile itself (simulator
        write I/O, disk) rather than pages built from it. The default (no
        call, or ``set_save_list(None)``) keeps ``save all``: every node
        including subckt internals, which is what the waveform explorer,
        ``annotate_currents`` and the no-re-simulate loop rely on.

        Selectors use the same grammar as ``emit_view(signals=...)``
        (``silk.signals.view.select.matches_selector``): full-name match,
        glob (``*`` and ``?``), or hierarchy segment run — all
        case-insensitive. Square brackets are an array index, never a glob
        character class: ``xx1[0].xxy[1].g[0]`` selects the same node as
        ``xx1<0>.xxy<1>.g[0]``, ``xx1[*]`` every copy, and a bracketed leaf
        net (``d[0]``, ``g[0]``) matches only itself. They are
        expanded at netlist-generation time against silk's own net inventory
        (top-level nets + ``x<instance>.<net>`` subckt internals), and a
        selector that matches nothing raises with near-miss suggestions in
        the ``[k]`` spelling.

        Designer paths also start at this Module: on ``tb.netlist``, use
        ``dut.x1[0].mid``; on a standalone circuit use its actual placements.
        Connected port paths select the parent net. Use bare node selectors,
        not ``v(...)`` wrappers. Emitted selector matches win over a different
        designer match set with a warning; globs can select several targets.
        Both include and exclude check the complete physical inventory.

        Parameters
        ----------
        signals : list of str or str, optional
            Selectors to include. ``None`` with no other argument clears the
            save list (back to ``save all``); ``None`` with ``exclude=`` or
            ``max_depth=`` starts from every known node.
        exclude : list of str or str, optional
            Selectors to subtract from the selection.
        max_depth : int, optional
            Keep only nodes at hierarchy depth <= max_depth (0 = top-level
            nets only).

        Notes
        -----
        Node voltages only for now — a selector shaped like a current
        (``i(...)``, ``@m...[id]``) raises. Incompatible with
        ``save_device_currents`` (whose batch-write mode is built on
        ``.save all``). A deck with a Noise analysis ignores the list (the
        noise output path writes its own vectors). On Xyce the list is
        rendered with ``:`` hierarchy separators automatically.
        """
        if signals is None and exclude is None and max_depth is None:
            self._save_list = None
            return

        def _norm(v, kwarg):
            if v is None:
                return None
            if isinstance(v, str):
                v = [v]
            v = list(v)
            for s in v:
                if not isinstance(s, str):
                    raise TypeError(
                        f'set_save_list {kwarg}= entries must be strings, '
                        f'got {s!r}')
                if s.lower().startswith('i(') or s.startswith('@'):
                    from silk.exceptions import NotSupportedError
                    raise NotSupportedError(
                        f'set_save_list selector {s!r}: a save list can name '
                        f'node voltages only for now — branch/device currents '
                        f'are not yet supported (backlog '
                        f'2026-08-12-simulator-save-list.md).')
            return v

        self._save_list = {
            'signals': _norm(signals, 'signals'),
            'exclude': _norm(exclude, 'exclude'),
            'max_depth': max_depth,
        }

    def generate_subckt(self, which_line):
        """Generate a subcircuit definition line.

        Parameters
        ----------
        which_line : int
            ``0`` for the ``.subckt <name> <pins...>`` header; any other
            value for the closing ``.ends`` line.

        Returns
        -------
        str
            The line, newline-terminated.

        Raises
        ------
        silk.exceptions.NotASubcircuitError
            If ``which_line`` is ``0`` and this Module has no
            ``module_name``. A top-level circuit has no ``.subckt`` header to
            write; it used to concatenate ``None`` and raise a
            bare ``TypeError`` from here.
        silk.exceptions.CircuitError
            If the header declares global ``0`` or case-insensitive ``gnd``.
        ValueError
            If a formal header name violates the authored identifier grammar.
        """
        if which_line == 0:
            require_subcircuit(self, 'Cannot emit a .subckt definition')
            _reject_unportable_ground_ports(self)
            from silk.elements.base import validate_net_token
            for name, pin in (self.pin_list or {}).items():
                validate_net_token(name, who='subcircuit header port', kind='port')
                validate_net_token(pin.name, who='subcircuit header port', kind='port')
            # module_name is a plain attribute: recheck what is written
            # (TASK-593), not only what the constructor saw.
            from silk.utils.deck_text import refuse_line_breaks
            strsubckt = '.subckt ' + refuse_line_breaks(
                getattr(self.netlister, '_subckt_name', None) or self.module_name,
                who='subcircuit name') + ' '
            for p in self.pin_list:
                strsubckt += p + ' '
        else:
            strsubckt = '.ends'
        return strsubckt + '\n'

    @_with_ngspice_version
    @_with_authored_name_validation
    @with_wiring_scope
    @with_definition_scope
    def generate_netlist(self, corner=None, subckt=None, fault_set=None,
                         manifest=None, seed=0, *, ngspice_version=None):
        """Generate SPICE netlist.

        If ``module_name`` is set, generates a ``.subckt`` block and returns
        the SPICE string without writing to disk (subcircuit mode).

        If ``module_name`` is None, generates a complete top-level simulation
        netlist and returns its file path. Overwrite mode writes
        ``dir_folder/netlistname``; new mode allocates a persistent child of
        the configured folder before emission. ``dir_folder`` remains the
        configured folder and ``netlistname`` the file name, in both modes.
        Relative folders give relative returned paths; absolute folders give
        absolute paths.
        ``run_simulator()`` consumes this preparation without relocating it.
        New mode requires regeneration before every subsequent launch.

        Parameters
        ----------
        corner : str, optional
            Process corner (used only in top-level mode). ``None`` resolves the
            PDK's declared ``default_corner`` from its manifest — the same
            resolution ``simulate()`` performs. The default used to be the
            literal ``'tt'``, which is sky130's spelling and no one else's:
            GF180MCU calls its typical corner ``typical``, so every emitted
            deck asked ngspice for a section that does not exist.
        subckt : bool, optional
            ``True`` forces subcircuit mode, ``False`` forces a top-level
            deck, ``None`` (default) picks subcircuit mode when
            ``module_name`` is set.
        fault_set : FaultSet, optional
            Faults realised and marked for this emission only; the markers
            are cleared afterwards. A faulted deck records no freshness
            preparation, so it is not reproducible from module state.
        manifest : optional
            Sink that receives the fault records realised for this emission.
        seed : int, optional
            Seed of the random generator used to realise ``fault_set``.

        ngspice_version : str, optional
            Target ngspice release (for example ``'45.2'`` or ``'47+'``).
            Overrides installed-release compatibility decisions for this call
            only, including subcircuits and completion certification. Requires
            the ngspice backend. ``None`` uses the installed release. An explicit
            target needs no local binary; run the deck externally when the local
            release differs or is unknown.

        Returns
        -------
        str
            The ``.subckt`` text in subcircuit mode; the actual written file
            path in top-level mode.

        Raises
        ------
        ConfigurationError
            If the configured deck file name contains ``/``; put its folder
            in ``dir_folder`` instead.
        ValueError
            If ``subckt=False`` is given while ``module_name`` is set.
        NotSupportedError
            If the selected simulator backend has no deck (behavioural), or
            the module holds model-only blocks, for a top-level deck.
        CircuitError
            If authored nets are invalid or a ground port is unportable
            (raised by the pre-emission checks).
        """
        _reject_unportable_ground_ports(self)

        if corner is None:
            corner = getattr(self.pdk_details, 'default_corner', 'tt')
        # Before any folder or deck is written (TASK-593).
        from silk.utils.deck_text import validate_corner_name
        validate_corner_name(corner)

        # subckt kwarg: explicit True/False overrides auto-detect
        # subckt=None (default) → auto-detect via module_name
        is_subckt = (subckt is True) or (subckt is None and self.module_name is not None)
        if subckt is False and self.module_name is not None:
            raise ValueError(
                "Module.generate_netlist(subckt=False) is not valid when module_name is set. "
                "Use generate_netlist() or generate_netlist(subckt=True) for subcircuit mode."
            )

        if not is_subckt:
            # A top-level deck only: a behavioural-selected block instantiated
            # in a SPICE bench still emits its .subckt (TASK-448.1).
            from silk.circuit.modules_sim import _refuse_deckless_backend
            _refuse_deckless_backend(self.backend, 'generate_netlist()')
            # After the backend check, so a behavioural top is told it has no
            # deck rather than to set the simulator it already set (448.8).
            _reject_model_only_blocks(self)

        # Fault injection: realize + mark before emission, clear after.
        marked: list = []
        fault_token = None
        manifest_start = len(manifest) if manifest is not None else None
        try:
            if fault_set is not None:
                import numpy as _np
                rng = _np.random.default_rng(seed)
                marked = self._silk_apply_fault_markers(
                    fault_set, rng, manifest=manifest)
                from silk.circuit.modules_faults import _prepare_fault_definitions
                from silk.circuit.module_placement import _fault_node
                fault_token = _fault_node.set(_prepare_fault_definitions(self, marked))
            if is_subckt:
                # Subcircuit mode — return .subckt string.
                return _subckt_text(self)
            else:
                # Top-level simulation mode — write to file
                from hashlib import sha256
                from silk.simulation.conditions import (
                    prepare_supply, prepare_temperature,
                )
                temperature_request = prepare_temperature(self)
                # Before emission, so a configured supply is what the
                # netlister renders (TASK-158). Resolution failures are
                # configuration errors and refuse here, not after a launch.
                self._supply_preparation = prepare_supply(self)
                files = _prepare_run_files(
                    self, temperature=temperature_request.temperature is not None)
                netlist_content = _emit_deck_text(self, corner, files)
                from dataclasses import replace
                from silk.simulation.completion import certify_generated
                completion, completion_reason = certify_generated(self, netlist_content)
                files = replace(files, completion=completion,
                                completion_unverified_reason=completion_reason)
                with open(files.netlist_path, 'w') as f:
                    f.write(netlist_content)
                self._prepared_device_param_requests = tuple(
                    getattr(self.netlister, '_device_param_requests', ()))
                self._run_files = files
                self._prepared_ngspice_version = ngspice_version
                digest = sha256(netlist_content.encode()).digest()
                # TASK-306: the prepared deck's provenance facts (its hash,
                # the PDK .lib lines it names, options, save list, seed).
                from silk.simulation.provenance import prepared_deck_record
                self._prepared_deck_record = prepared_deck_record(
                    self, netlist_content)
                from silk.circuit.test_execution import normalized_deck_digest
                normalized, _reason = normalized_deck_digest(self, corner)
                self._prepared_deck_record['normalized_deck_sha256'] = normalized
                self._temperature_preparation = (
                    temperature_request, str(corner), digest,
                )
                # Freshness record (TASK-160). Fault markers are applied to the
                # live design for this emission only and cleared below, so a
                # faulted deck is not reproducible from module state: record
                # nothing rather than refuse a launch on a difference the guard
                # itself created. The fault runner drives the backend directly
                # and never reaches run_simulator(); the TASK-139 on-disk
                # digest check still covers that deck.
                self._deck_preparation = (
                    None if marked else
                    (corner, files,
                     self.circuit_name or execution_location(self)[1].split('/')[-1],
                     digest))
                self._run_preparation = (
                    *execution_location(self)[:2], self.simulator,
                    execution_location(self)[2], files,
                )
                self._run_consumed = False
                return files.netlist_path
        except BaseException:
            if manifest is not None:
                del manifest[manifest_start:]
            raise
        finally:
            if fault_token is not None:
                _fault_node.reset(fault_token)
            if marked:
                self._silk_clear_fault_markers(marked)

    def generate_netlist_entry(self, instance_name):
        """Generate the instantiation line for this module.

        Parameters
        ----------
        instance_name : str
            Instance name to emit; it is converted to the emitted submodule
            spelling.

        Returns
        -------
        str
            ``<instance> <nets...> <module_name>``, newline-terminated, with
            nets in pin order.

        Raises
        ------
        silk.exceptions.NotASubcircuitError
            If this Module has no ``module_name``. Only a subcircuit can be
            instantiated, and the refusal names the module rather than
            letting ``None`` reach the concatenation below as a bare
            ``TypeError``. The testbench path refuses earlier
            still, at ``add_module_under_test``.
        silk.exceptions.CircuitError
            If the interface declares global ``0`` or case-insensitive
            ``gnd``, or a pin has no net connected.
        """
        require_subcircuit(self, f'Cannot instantiate {instance_name!r}')
        str_entry = emitted_submodule_name(instance_name) + ' '
        device = self
        pin_list = device.pin_list

        _reject_unportable_ground_ports(device)
        from silk.elements.base import validate_net_token
        for p in pin_list:
            if pin_list[p].net == None:
                raise CircuitError('ERROR: In module instantiation {}, net for pin {} is undefined'.format(instance_name, p))
            validate_net_token(p, who=f'Module {instance_name!r} port', kind='port')
            validate_net_token(pin_list[p].name, who=f'Module {instance_name!r} port', kind='port')
            str_entry += validate_net_token(
                pin_list[p].net, who=f'Module {instance_name!r} pin {p!r}') + ' '

        from silk.utils.deck_text import refuse_line_breaks
        str_entry += refuse_line_breaks(
            device.module_name,
            who=f'subcircuit name of instance {instance_name!r}') + '\n'
        return str_entry
