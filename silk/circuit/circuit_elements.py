"""Circuit graph storage, device placement, net connections and subcircuit instances."""

# Python based circuit element manager
# Maintained by the silk contributors
# Shared functionality for managing circuit elements in Module
# This includes devices, nets, and module instantiation

from __future__ import annotations

import copy
from typing import Optional, TYPE_CHECKING

from silk.exceptions import CircuitError, DuplicateNameError, UndeclaredNetError, SimulationError, PDKError
from silk.elements.base import AnalogElement, pin_context, validate_net_token
from silk.circuit.net import Net, PortBinding
from silk.circuit._behavioral_identity import behavioral_key, custom_key, definition_hash
# Port/element vocabulary now lives in silk.circuit.ports.
# Re-exported here so existing imports — including string-literal imports in
# generated user code — keep working unchanged (identical class objects).
from silk.circuit.ports import ElementType, PortDir, PortRole, Port

if TYPE_CHECKING:
    from silk.circuit.modules import Module


def _bus_port_bit_net(module, name):
    """Find a deferred port-bit Net without materializing other graph nets."""
    ports = {name.casefold() for name in (getattr(module, 'pin_list', None) or {})}
    for handle in (getattr(module, '_bus_handles', {}) or {}).values():
        if not all(bit.name.casefold() in ports for bit in handle._underlying_pins):
            continue
        for bit in handle._underlying_pins:
            if bit.name.casefold() == name.casefold():
                return bit
    return None


def _module_structure_hash(module) -> str:
    """Structure digest for naming a built Module.

    Provable source trees reuse the database canonicalizer. Opaque source trees
    retain a source-omitting naming hint, which cannot prove persistence or run
    sharing. Import lazily to keep :mod:`silk.db` off the common assembly path.
    """
    return _module_structure_projection(module)[0]


def _module_structure_projection(module):
    """Return a naming digest and whether sources prove structural identity.

    Opaque sources retain the old source-omitting naming hint. Its emitted
    body must still pass collision checks; persistence never uses this hint.
    """
    from silk.db.structure_tracker import CircuitStructureTracker
    from silk.db._source_identity import _UnprovableSource
    tracker = CircuitStructureTracker(None, None)
    try:
        struct = tracker.extract_structure(module)
    except _UnprovableSource:
        tracker = CircuitStructureTracker(None, None, _sources=False)
        struct = tracker.extract_structure(module)
        return tracker.compute_structure_hash(struct)[0], False
    return tracker.compute_structure_hash(struct)[0], True


# Netlister state that ``generate_module_definitions`` primes on a submodule
# before emitting it. Pinned to one fixed value on BOTH bodies while the
# dedup cross-check emits them, so a definition that was already written
# into an ammeter-wrapped parent deck compares against a never-emitted
# rebuild on equal terms; restored afterwards so the check leaves no trace.
_EMISSION_STATE = ('_terminal_current_method', '_ammeter_wrappers',
                   '_emitted_subckts', '_mc_sample', '_subckt_name')
_UNSET = object()


def _module_subckt_text(module, name=None) -> str:
    """The ``.subckt`` block a built Module would emit, under pinned netlister
    state.

    The text is what the deck actually carries, so it is the ground truth
    the structure fingerprint approximates: a binding the fingerprint
    forgets, such as a child instance's port bindings, still
    shows up here as a different ``x<instance>`` line. Emitted with a fresh
    ``emitted`` set so every nested definition is written out in full on
    both sides.

    Uses the bare emission sequence (``modules_netlist._subckt_text``), not
    ``generate_netlist(subckt=True)``: this is a comparison of two bodies'
    text, and the emission-time contracts that wrap a real deck (the
    ``gnd``-port guard, fault markers) are not its business — a block the
    lint catalog already flags must still register for linting.
    """
    from silk.circuit.modules_netlist import _subckt_text
    netlister = module.netlister
    saved = {name: getattr(netlister, name, _UNSET) for name in _EMISSION_STATE}
    try:
        netlister._terminal_current_method = 'savecurrents'
        netlister._ammeter_wrappers = {}
        netlister._emitted_subckts = set()
        netlister._mc_sample = None
        netlister._subckt_name = name or module.module_name
        return _subckt_text(module)
    finally:
        for name, value in saved.items():
            if value is _UNSET:
                delattr(netlister, name)
            else:
                setattr(netlister, name, value)


def _first_text_difference(a: str, b: str) -> str:
    """Describe where two emitted subcircuit texts first diverge."""
    a_lines, b_lines = a.splitlines(), b.splitlines()
    for idx, (line_a, line_b) in enumerate(zip(a_lines, b_lines), 1):
        if line_a != line_b:
            return f"line {idx}: registered {line_a!r} vs new {line_b!r}"
    return (f"registered text has {len(a_lines)} lines, new text has "
            f"{len(b_lines)} lines")


def _casefold_module_lookup(module_dict, name):
    """Return the ``module_dict`` key ``name`` case-folds onto, or ``None``.

    SPICE ``.subckt`` names are case-insensitive (ngspice folds to lower,
    Xyce to upper — measured 2026-09-27): ``'Bias'`` and
    ``'bias'`` are ONE namespace entry to the simulator even though a plain
    dict keeps them as two independent keys. Mirrors the net
    (``CircuitGraph._net_casefold``) and port (``_port_casefold_index``)
    case-fold guards. The first-registered spelling is canonical — it is
    what ``.subckt``/X lines actually emit — so a later call under a
    different-case spelling of the same name resolves onto it here, before
    ``module_dict`` is ever keyed on the new spelling.
    """
    if name in module_dict:
        return name
    folded = name.casefold()
    for key in module_dict:
        if key.casefold() == folded:
            return key
    return None


def definition_names(elemDevice, digest):
    """The one naming helper: ``(emitted, structural)`` variant names.

    ``emitted`` folds authored custom-line text into the digest so bodies that
    differ only in raw text get distinct ``.subckt`` names. ``structural`` is
    the custom-blind name (the legacy one) that structural identity records,
    so raw text never moves a DB structure digest or a wrapper's body hash.
    """
    base = elemDevice.module_name
    model = behavioral_key(elemDevice)
    structural = f"{base}_{definition_hash(digest, model)[:8]}"
    emitted = f"{base}_{definition_hash(digest, model, custom_key(elemDevice))[:8]}"
    return emitted, structural


def structural_definition_name(module_dict, name):
    """Custom-blind name of the registered definition *name*."""
    entry = module_dict.get(name)
    return entry.get('_structural_name', name) if isinstance(entry, dict) else name


def register_module_definition(module_dict, elemDevice, *, validate=False, definition_name=None) -> tuple[str, bool, bool]:
    """Index live definitions by their current name.

    Placement only registers references. A fresh deck-wide registry uses
    ``validate=True`` at emission to compare different objects claiming the
    same name; no placement-time fingerprint is trusted after a mutation.
    """
    base_name = elemDevice.module_name
    variants = getattr(elemDevice, 'variants', False)
    digest = None
    structural_name = None
    if definition_name is not None:
        requested_name = definition_name
    elif variants:
        digest = _module_structure_hash(elemDevice)
        requested_name, structural_name = definition_names(elemDevice, digest)
    else:
        requested_name = base_name
    name = _casefold_module_lookup(module_dict, requested_name)
    if name is None:
        name = requested_name
    entry = module_dict.get(name)
    if entry is None:
        new_entry = {
            'device': elemDevice,
            'defined': False,
            'numberInstantiated': 1,
            '_source_ids': {id(elemDevice)},
            '_structural_name': structural_name or requested_name,
        }
        if digest is not None:
            new_entry['_body_hash'] = digest  # already computed above; skip the lazy recompute below
        module_dict[name] = new_entry
        return name, True, True

    source_ids = entry.setdefault('_source_ids', set())
    source_id_added = id(elemDevice) not in source_ids
    if source_id_added and validate:
        # A distinct object claims a name already in use. Allowed only if it is
        # structurally the same circuit (a legitimate identical rebuild); a real
        # difference is a collision that must never be silently merged. Under
        # variants=True this branch is reached only by a genuine identical
        # rebuild (same digest -> same name, hash matches trivially) or the
        # astronomically unlikely truncated-hash collision between two
        # different bodies — which this cross-check still catches and raises
        # on, rather than silently merging (TASK-292.1 AC#1 addendum item 4).
        if behavioral_key(entry['device']) != behavioral_key(elemDevice):
            raise CircuitError(
                f"module_name {requested_name!r} is claimed by two different "
                f"definitions: the behavioral model differs from already-registered "
                f"module_name {name!r} (code object or explicit parameters). "
                "Give one of them a distinct name via define_module_name(...) "
                "or use variants=True."
            )
        if '_body_hash' not in entry:
            entry['_body_hash'] = _module_structure_hash(entry['device'])
        this_hash = digest if digest is not None else _module_structure_hash(elemDevice)
        if this_hash != entry['_body_hash']:
            if name != requested_name:
                raise CircuitError(
                    f"module_name {requested_name!r} is claimed by two "
                    f"structurally different circuits: it differs from "
                    f"already-registered module_name {name!r} only by case. "
                    f"SPICE has a single case-insensitive namespace for "
                    f".subckt names (ngspice folds to lower, Xyce to upper), "
                    f"so silk would emit two definitions that the simulator "
                    f"collapses into one, and every instance of the other "
                    f"body would silently reference the wrong circuit. Give "
                    f"one of them a distinct name via define_module_name(...)."
                )
            raise CircuitError(
                f"module_name {name!r} is claimed by two structurally different "
                f"circuits. SPICE has a single namespace for .subckt names, so "
                f"silk would emit one definition and every instance of the other "
                f"would silently reference the wrong circuit. Give one of them a "
                f"distinct name via define_module_name(...)."
            )
        # Belt and braces (TASK-126): hash equality alone is no longer enough
        # to merge two definitions. The fingerprint is a summary that can
        # omit something identity-bearing (TASK-125: every child instance's
        # port bindings, silently, since the tracker shipped); the emitted
        # ``.subckt`` text is what the deck carries, so the two bodies must
        # also agree byte-for-byte there. A body that cannot be emitted
        # cannot be proven equal, so that fails here too rather than being
        # merged on the hash alone.
        try:
            if '_body_text' not in entry:
                entry['_body_text'] = _module_subckt_text(entry['device'], name)
            new_text = _module_subckt_text(elemDevice, name)
        except CircuitError:
            raise
        except Exception as exc:
            raise CircuitError(
                f"module_name {name!r} is claimed by a second circuit object "
                f"whose .subckt text could not be emitted to cross-check the "
                f"structure hash: {type(exc).__name__}: {exc}"
            ) from exc
        if new_text != entry['_body_text']:
            if name != requested_name:
                raise CircuitError(
                    f"module_name {requested_name!r} is claimed by two "
                    f"circuits whose structure hashes agree but whose "
                    f"emitted .subckt text differs "
                    f"({_first_text_difference(entry['_body_text'], new_text)}); "
                    f"it differs from already-registered module_name {name!r} "
                    f"only by case. SPICE has a single case-insensitive "
                    f"namespace for .subckt names (ngspice folds to lower, "
                    f"Xyce to upper), so silk would emit two definitions "
                    f"that the simulator collapses into one, and every "
                    f"instance of the other body would silently reference "
                    f"the wrong circuit. Give one of them a distinct name "
                    f"via define_module_name(...)."
                )
            raise CircuitError(
                f"module_name {name!r} is claimed by two circuits whose "
                f"structure hashes agree but whose emitted .subckt text "
                f"differs ({_first_text_difference(entry['_body_text'], new_text)}). "
                f"SPICE has a single namespace for .subckt names, so silk "
                f"would emit one definition and every instance of the other "
                f"would silently reference the wrong circuit. Give one of "
                f"them a distinct name via define_module_name(...)."
            )
    if source_id_added:
        source_ids.add(id(elemDevice))
    entry['numberInstantiated'] += 1
    return name, False, source_id_added


def unregister_module_definition(module_dict, elemDevice, registration) -> None:
    """Undo a :func:`register_module_definition` call whose ``add_element``
    was then rejected by the binding step that follows it.
    ``registration`` is that call's return value.

    A freshly created entry (``created`` True) is purged outright — this
    call is the only thing depending on it, so nothing is left orphaned in
    ``module_dict`` to collide with a later, structurally different module
    of the same name. A deduped registration (a second-or-later
    instantiation of an already-registered module) instead rolls back only
    what this call added: the ``numberInstantiated`` bump, and this
    object's id in ``_source_ids`` if this call is what put it there — an
    earlier, successful instance's entry must come back exactly as it was,
    not merely present.

    """
    name, created, source_id_added = registration
    # ``name`` is the key register_module_definition actually used — the
    # bare module_name, or the variants=True suffix. elemDevice.module_name
    # itself never changes (TASK-292.1 AC#4), so re-deriving the key from it
    # would look up the wrong entry under variants=True; use the returned
    # name instead.
    if created:
        module_dict.pop(name, None)
        return
    entry = module_dict.get(name)
    if entry is None:
        return
    entry['numberInstantiated'] -= 1
    if source_id_added:
        source_ids = entry.get('_source_ids')
        if source_ids is not None:
            source_ids.discard(id(elemDevice))


class CircuitGraph:
    """The in-memory circuit graph — devices + nets + the ``_bind_pin_to_net``
    chokepoint that guards ``Net.pins`` — managed on behalf of :class:`Module`.

    Renamed from ``PyCircuitElements``; the ``Py`` prefix
    was a dead pynetlist artifact. ``PyCircuitElements`` remains as a
    backward-compatible alias at the bottom of this module.
    """

    def __init__(self, pdk_details, debug=False, strict_nets=False):
        self.pdk_details = pdk_details
        self.debug = debug
        self.strict_nets = strict_nets

        # Element storage
        self.elem_dict = {}        # Dictionary of all elements
        self._instance_names = {}
        self._placement_tokens = {}
        self.net_dict = {}         # Dictionary of all nets — dict[str, Net] after refactor
        # Case-insensitive net identity (G16, 2026-08-24): SPICE folds node
        # names (ngspice to lower, Xyce to upper), so names differing only by
        # case are ONE electrical node — silk merges them onto the first-seen
        # spelling and warns. casefold index {name.casefold(): canonical}.
        self._net_casefold = {}
        self._case_merge_warned = set()
        self.module_dict = {}      # Dictionary of all modules (subcircuit definitions)

        # Duck-typing marker for module-ness: exposed as Module.module_names
        # (self.module_names = self._element_mgr.module_names in
        # modules.py), read via hasattr() by
        # CircuitStructureTracker._is_pymodule. Held here rather than on
        # add_element's old-style path, which no longer matches devices
        # against it (TASK-108 — that branch was dead and is deleted).
        self.module_names = ['module']

        # Back-reference to parent Module — set by Module.__init__ right after
        # this CircuitGraph instance is constructed, so that _ensure_net
        # can pass it into new Net objects as Net.module.
        self._module_ref: Optional['Module'] = None

        # Reverse index for the pin-binding chokepoint. Maps (instance, port)
        # to the name of the net that pin is currently bound to. Used by
        # _bind_pin_to_net to clean up stale bindings when a pin is rewired.
        self._pin_to_net: dict[tuple[str, str], str] = {}

    def _canonical_net_name(self, net_name: str) -> str:
        """Fold a net name onto its case-canonical spelling (G16).

        SPICE simulators treat node names case-insensitively — ngspice
        folds to lower case, Xyce to upper (both proven empirically,
        2026-08-24), so ``'VDD'`` and ``'vdd'`` are ONE electrical node in
        every simulation. silk therefore treats net names that differ only
        by case as the same net: the first-seen spelling is canonical (it
        is what the netlist emits), and a WARNING is printed the first
        time each variant merges so the author can fix the spelling.
        """
        if net_name in self.net_dict:
            return net_name
        idx = self._net_casefold
        if len(idx) != len(self.net_dict):
            idx.clear()
            for k in self.net_dict:
                idx[k.casefold()] = k
        canonical = idx.get(net_name.casefold())
        if canonical is None:
            deferred = _bus_port_bit_net(self._module_ref, net_name)
            canonical = deferred.name if deferred is not None else None
        if canonical is None or canonical == net_name:
            return net_name
        if (canonical, net_name) not in self._case_merge_warned:
            self._case_merge_warned.add((canonical, net_name))
            import sys as _sys
            print(f"WARNING: net '{net_name}' differs only by case from "
                  f"existing net '{canonical}'. SPICE simulators fold "
                  f"node-name case (ngspice to lower, Xyce to upper), so "
                  f"these are ONE electrical node — silk merges them and "
                  f"uses '{canonical}'. Rename one if you meant two nets.",
                  file=_sys.stderr)
        return canonical

    def _accept_net(self, net_name):
        """Decide acceptance without materializing nets or canonical caches."""
        from silk.circuit.bus_spec import _guard_scalar_alias
        _guard_scalar_alias(self._module_ref, net_name, connection=True)
        canonical = net_name
        if net_name not in self.net_dict:
            if len(self._net_casefold) == len(self.net_dict):
                canonical = self._net_casefold.get(net_name.casefold(), net_name)
            else:
                canonical = next((n for n in self.net_dict if n.casefold() == net_name.casefold()), net_name)
        existing = self.net_dict.get(canonical)
        if existing is None:
            existing = _bus_port_bit_net(self._module_ref, canonical)
        if existing is None and self.strict_nets:
            raise UndeclaredNetError(
                f'ERROR: net "{net_name}" is not declared. '
                f'Call add_net("{net_name}") before connecting it.')
        return (existing.name if existing is not None else canonical), existing

    def _ensure_net(self, net_name: str) -> 'Net':
        """Return the Net for net_name, creating it if missing.

        Honors ``strict_nets``: raises :class:`UndeclaredNetError` if the
        net does not exist and ``strict_nets`` is True. Otherwise creates
        a new :class:`Net` with a back-reference to this element manager's
        parent module.

        This method is the single place in silk where new ``Net`` objects
        are constructed during circuit building. Net names that differ
        only by case resolve to the SAME net (see
        :meth:`_canonical_net_name`).
        """
        # NOT validated here, deliberately. This is the one place new Nets are
        # constructed, so a node-token check here would be the natural floor —
        # but it would report a bad value as a bare net name with no owner or
        # terminal named. Every path that reaches here validates first with
        # its own context: devices (add_element / connect_element / add_net /
        # generate_netlist_entry, TASK-41/75), sources (``SourceBase``
        # terminals, TASK-76) and passives (``PassiveBase``, TASK-76).
        _, existing = self._accept_net(net_name)
        net_name = self._canonical_net_name(net_name)
        if existing is not None:
            self.net_dict.setdefault(existing.name, existing)
            return existing
        net = Net(name=net_name, module=self._module_ref)
        self.net_dict[net_name] = net
        return net

    def _bind_pin_to_net(self, net_name: str, binding: 'PortBinding', *, undo=None) -> 'Net':
        """Register a pin binding on the Net side.

        This is the single chokepoint for pin-to-net binding. Every
        add_element, connect_element, add_passive, add_source path funnels
        through here to keep ``Net.pins`` consistent.

        Handles both new connections and rewires:

        - Looks up any previous binding for ``(binding.instance, binding.port)``
          via the reverse index ``_pin_to_net``.
        - Ensures the target net exists (via :meth:`_ensure_net`, honoring
          ``strict_nets``) — BEFORE anything is removed, so a rejected net
          raises with ``Net.pins`` and ``_pin_to_net`` untouched.
        - If the previous binding pointed at a different net, removes the
          stale PortBinding from that net.
        - Appends the new PortBinding to the target net.
        - Updates the reverse index.

        Owner-side bookkeeping (device.nets, passive.pos/neg, source
        terminals, submodule pin_list) is the caller's responsibility.
        This method only touches ``Net.pins`` and ``_pin_to_net``. Callers
        that hold owner-side state write it AFTER this returns (see
        ``AnalogElement.__setattr__`` / ``SourceBase.__setattr__``), so an
        :class:`UndeclaredNetError` from here leaves the owner untouched
        too.
        """
        # Canonicalize BEFORE touching the reverse index — otherwise a
        # case-variant spelling ('vdd' for an existing 'VDD') would be
        # stored raw in _pin_to_net, and later rewires would miss the
        # stale-binding cleanup (its guard checks net_dict membership).
        # binding.net is canonicalized too, so Net.pins entries always
        # agree with their Net's name; the OWNER-side strings (device.nets,
        # passive.pos/...) deliberately keep the author's spelling — the
        # deck emits verbatim (simulators fold case), the model is
        # canonical.
        net_name = self._canonical_net_name(net_name)
        if binding.net != net_name:
            binding.net = net_name
        key = (binding.instance, binding.port)

        old_net_name = self._pin_to_net.get(key)
        # Same-net rebind is a no-op — avoid duplicate PortBindings.
        if old_net_name == net_name:
            return self.net_dict[net_name]
        # The strict_nets gate runs BEFORE the stale binding is dropped.
        # Pre-fix (TASK-100) it ran after: on a strict module a rewire to an
        # undeclared net raised UndeclaredNetError with the old Net.pins
        # entry already gone while _pin_to_net still named the old net —
        # and because of that stale reverse entry, rewiring BACK to the old
        # net was then the same-net no-op above, so the old net's pins never
        # recovered. Now a rejected net changes nothing on the Net side.
        created = net_name not in self.net_dict
        net = self._ensure_net(net_name)
        removed = []
        # The `old_net_name in self.net_dict` guard assumes net_dict is
        # append-only in Phase 1. If future work adds net removal, this
        # branch may silently skip cleanup for bindings whose old net
        # has been deleted — revisit at that point.
        if old_net_name and old_net_name != net_name and old_net_name in self.net_dict:
            old_net = self.net_dict[old_net_name]
            removed = [(old_net, i, b) for i, b in enumerate(old_net.pins)
                       if b.instance == binding.instance and b.port == binding.port]
            old_net.pins = [
                b for b in old_net.pins
                if not (b.instance == binding.instance and b.port == binding.port)
            ]

        net.pins.append(binding)
        self._pin_to_net[key] = net_name
        if undo is not None:
            undo.append((key, old_net_name, binding, removed, created))
        return net

    def _registry(self, registry):
        return self.elem_dict if registry == 'elem_dict' else getattr(self._module_ref, registry)

    def _check_instance_name(self, registry, slot, device, *, staged=None, retiring=None):
        """Preflight only this placement's slots and physical card tokens."""
        from silk.circuit import emitted_names
        emitted_names.validate_instance_name(slot)
        existing = self._registry(registry).get(slot)
        old_device = existing.get('device') if registry == 'elem_dict' and existing is not None else existing
        retired = retiring or {}
        if old_device is not None and not any(
                old[0] == registry and old[1] == slot and old[2] is old_device
                for token in self._placement_tokens.get((registry, slot), ())
                if (old := retired.get(token.lower())) is not None):
            if old_device is device:
                return None
            raise DuplicateNameError(
                f"Cannot add instance {slot!r}: destination {registry}[{slot!r}] "
                f"already contains instance {slot!r}; an exact destination-slot clash would overwrite it.")
        if isinstance(device, AnalogElement) or registry in ('passives_dict', 'sig_sources_dict'):
            tokens = emitted_names.element_names(device).card_names
        else:
            tokens = (emitted_names.emitted_instance_name(device, slot),)
        folded = set()
        for token in tokens:
            key = token.lower()
            if key in folded:
                raise DuplicateNameError(f"Instance {slot!r} declares duplicate SPICE card {token!r}.")
            folded.add(key)
            previous = (staged or {}).get(key) or self._instance_names.get(key)
            if previous is None:
                continue
            removed = retired.get(key)
            if removed is not None and previous[0:2] == removed[0:2] and previous[2] is removed[2]:
                previous = (staged or {}).get(key)
                if previous is None:
                    continue
            old_registry, old_slot, old_obj = previous
            old_tokens = self._placement_tokens.get((old_registry, old_slot))
            if old_tokens is None:
                old_tokens = emitted_names.element_names(old_obj).card_names
            old_token = next(t for t in old_tokens if t.lower() == key)
            if token != old_token:
                reason = (f"emitted SPICE name {token!r} conflicts with instance "
                          f"{old_slot!r} ({old_token!r}); SPICE folds instance-name case.")
            else:
                reason = (f"emitted SPICE name {token!r} is already placed as instance "
                          f"{old_slot!r} through {old_registry}; both paths emit the same SPICE instance.")
            raise DuplicateNameError(f"Cannot add instance {slot!r} through {registry}: {reason}")
        return tokens

    def _register_instance(self, registry, slot, device, tokens):
        from silk.circuit.object_holders import register
        register(self._module_ref, registry, slot, device)
        owner = (registry, slot, device)
        for token in tokens:
            self._instance_names[token.lower()] = owner
        self._placement_tokens[registry, slot] = tokens
        if isinstance(device, AnalogElement) or registry in ('passives_dict', 'sig_sources_dict'):
            object.__setattr__(device, '_name_locked', True)

    def _retire_instance(self, registry, slot, device):
        from silk.circuit.object_holders import retire
        retire(self._module_ref, registry, slot, device)
        tokens = self._placement_tokens.get((registry, slot), ())
        for token in tokens:
            owner = self._instance_names.get(token.lower())
            if owner is not None and owner[:2] == (registry, slot) and owner[2] is device:
                del self._instance_names[token.lower()]
        current = self._registry(registry).get(slot)
        current = current.get('device') if registry == 'elem_dict' and current is not None else current
        if current is device:
            self._placement_tokens.pop((registry, slot), None)

    def _purge_partial_instance(self, registry, slot, device, undo):
        """Undo only bindings changed by the attempted placement."""
        self._retire_instance(registry, slot, device)
        current = self._registry(registry).get(slot)
        current_device = current.get('device') if registry == 'elem_dict' and current is not None else current
        if current_device is device:
            self._registry(registry).pop(slot, None)
        self._undo_bindings(undo)

    def _snapshot_net_name_state(self):
        """Copy net-name bookkeeping before a reversible binding edit."""
        return dict(self._net_casefold), set(self._case_merge_warned)

    def _restore_net_name_state(self, state):
        """Restore net-name bookkeeping after the binding edits were undone."""
        self._net_casefold, self._case_merge_warned = state

    def _undo_bindings(self, undo):
        """Replay binding changes without removing their existing placement."""
        for key, old_name, added, removed, created in reversed(undo):
            net = self.net_dict[added.net]
            net.pins[:] = [pin for pin in net.pins if pin is not added]
            for old_net, position, pin in removed:
                old_net.pins.insert(position, pin)
            if old_name is None:
                self._pin_to_net.pop(key, None)
            else:
                self._pin_to_net[key] = old_name
            if created and not net.pins:
                self.remove_net(net.name)

    def add_element(self, elemDevice, elemName, nets=None, parameters=None):  # noqa: C901
        """Add element to circuit.

        Args:
            elemDevice: Device class (new-style), device instance (new-style),
                        or PDKDevice (old-style) to add. A Module (submodule)
                        target is not supported here — go through
                        Module.add_element (modules_structure.py), which
                        handles submodule registration itself.
            elemName: Unique name for this instance.
            nets: Dictionary or list of net connections.
            parameters: Dictionary or list of parameter values.
        """
        from silk.circuit.device_view import _refuse_device_view
        _refuse_device_view(elemDevice, "add_element()")
        from silk.circuit.emitted_names import validate_instance_name
        validate_instance_name(elemName)

        # ------------------------------------------------------------------
        # Three-way detection: new-style class, new-style instance, old-style
        # ------------------------------------------------------------------

        if isinstance(elemDevice, type) and issubclass(elemDevice, AnalogElement):
            # Class-argument path: elemDevice IS the class (e.g.
            # add_element(pdk.nmos1p8, 'mn1', ...)). The class is used only
            # as a lookup key — the manager deep-copies the internal
            # PDKDevice template for that device type and stores the copy.
            # SPICE generation for this path goes through the legacy
            # pdk_details.generate_netlist_entry() route because the stored
            # PDKDevice does not have a to_spice() method; the netlister
            # detects that by checking isinstance(device, AnalogElement).
            elemType = elemDevice._device_type_name
            if elemType not in self.pdk_details.model_names:
                raise CircuitError(
                    f'ERROR: Device {elemType!r} not found in PDK.'
                )
            # Deepcopy the internal PDKDevice (not the class)
            new_device = copy.deepcopy(self.pdk_details._devices[elemType])
            # Every net value has to survive as one node token — checked here,
            # BEFORE the element is registered, so a bad dict leaves nothing
            # behind (TASK-75 AC #1). Without it, `g=''` was bound as a falsy
            # no-op and emitted a line one node short that ngspice mis-parsed
            # into an invented node, and `g=None` died with a bare
            # AttributeError from inside net-name canonicalization.
            if nets is not None:
                # Pin order read off the stored PDKDevice, which is what the
                # positional form actually binds against.
                _validate_connections(elemDevice, elemName, nets,
                                      list(new_device.pin_list))
            element = {
                'type': ElementType.DEVICE,
                'device': new_device,
                'typeName': elemType,
            }
            tokens = self._check_instance_name('elem_dict', elemName, new_device)
            if tokens is None:
                return
            self.elem_dict[elemName] = element
            undo = []
            try:
                # Apply nets and parameters via existing methods
                if nets is not None:
                    _apply_connections(self, elemName, nets, undo=undo)
                if parameters is not None:
                    _apply_parameters(self, elemName, parameters)
            except Exception:
                # A failure partway through (unknown pin, undeclared net
                # under strict_nets, pins/nets length mismatch, ...) must not
                # leave elemName half-wired (TASK-43 AC #4).
                self._purge_partial_instance('elem_dict', elemName, new_device, undo)
                raise
            self._register_instance('elem_dict', elemName, new_device, tokens)
            return

        if isinstance(elemDevice, AnalogElement):
            # New-style: elemDevice is already an instance
            # Store directly — spice_netlister calls instance.to_spice()
            self._add_analog_element(elemDevice)
            return

        # ------------------------------------------------------------------
        # Old-style path (PDKDevice) — MODULE branch removed (TASK-108)
        # ------------------------------------------------------------------
        # This path used to also handle device_name == 'module', calling
        # register_module_definition with no rollback if
        # the binding step below then rejected the call — the same defect
        # TASK-98 fixed on the live path (modules_structure.py). It was dead
        # code: every Module sets device_name = 'module', but
        # Module.add_element intercepts a Module target and handles
        # registration/rollback itself (its own register_module_definition /
        # unregister_module_definition pair) before ever delegating here —
        # confirmed by grepping every caller of CircuitGraph.add_element /
        # PyCircuitElements.add_element / _element_mgr.add_element in silk/
        # and tests/: none passes a module-typed elemDevice. Deleted rather
        # than fixed, so there is exactly one copy of that rollback logic to
        # maintain.

        elemType = elemDevice.device_name

        if elemType not in self.pdk_details.model_names:
            raise CircuitError(f'ERROR: Element name not found in PDK. Please create {elemName} before adding it to the netlist.')

        if self.debug:
            print('Element Name found in pdk')

        # Get original device and copy it
        original_device = self.pdk_details._devices.get(elemType)
        if original_device is None:
            # Fallback: getattr on PDK (handles edge cases)
            original_device = getattr(self.pdk_details, elemType)

        new_device = copy.deepcopy(original_device)

        # Create element entry
        element = {
            'type': ElementType.DEVICE,
            'device': new_device,
            'typeName': new_device.device_name
        }

        pins = new_device.pin_list
        params = new_device.param_list

        if self.debug:
            print(element)

        # Same pre-registration net-value check as the class-argument path
        # above (TASK-75 AC #1).
        if nets is not None:
            _validate_connections(new_device, elemName, nets, list(pins))

        tokens = self._check_instance_name('elem_dict', elemName, new_device)
        if tokens is None:
            return
        self.elem_dict[elemName] = element
        undo = []
        try:
            # Handle net connections
            if nets is not None:
                _apply_connections(self, elemName, nets, undo=undo)

            # Handle parameters
            if parameters is not None:
                _apply_parameters(self, elemName, parameters)
        except Exception:
            # See the class-argument path above (TASK-43 AC #4): a failure
            # here must not leave elemName half-wired in elem_dict.
            self._purge_partial_instance('elem_dict', elemName, new_device, undo)
            raise

        self._register_instance('elem_dict', elemName, new_device, tokens)

    def _add_analog_element(self, device: AnalogElement):
        """Store a factory-created AnalogElement instance in elem_dict.

        Called by :meth:`add` and by :meth:`add_element` when passed an
        instance. SPICE generation uses ``instance.to_spice()`` directly.

        Each connected pin funnels through :meth:`_bind_pin_to_net` so
        that ``Net.pins`` stays consistent. Supported devices receive weak
        holder records after registration; custom emitters retain the
        single-owner back-reference used by inherited pin setters.
        """
        from silk.circuit.device_view import _refuse_device_view
        _refuse_device_view(device, "_add_analog_element()")
        name = device.name
        from silk.circuit.emitted_names import validate_instance_name
        validate_instance_name(name)
        tokens = self._check_instance_name('elem_dict', name, device)
        if tokens is None:
            return

        # Record in elem_dict first so the device is registered before any
        # potential error from _bind_pin_to_net would leave dangling state.
        # The stored device IS an AnalogElement instance; downstream code
        # (netlister, update_element, junction_check) distinguishes this
        # from the legacy PDKDevice path via isinstance(device, AnalogElement).
        self.elem_dict[name] = {
            'type': ElementType.DEVICE,
            'device': device,
            'typeName': device._device_type_name,
        }

        undo = []
        try:
            # Register per-pin bindings through the chokepoint.
            for pin_name in device._pin_names:
                net_name = device.nets.get(pin_name)
                if not net_name:
                    continue
                self._bind_pin_to_net(net_name, PortBinding(
                    net=net_name,
                    instance=name,
                    port=pin_name,
                    direction=None,   # devices don't carry direction/role on Pin
                    role=None,
                    kind='device',
                ), undo=undo)

        except Exception:
            self._purge_partial_instance('elem_dict', name, device, undo)
            raise
        self._register_instance('elem_dict', name, device, tokens)

        # Legacy owner for custom emitters. Supported factory devices use
        # the weak holders installed by successful registration instead.
        object.__setattr__(device, '_module', self._module_ref)

    def add_net(self, netname, parameters=None, classification=None):
        """Add a net to the circuit.

        Creates a new :class:`Net` object and stores the optional legacy
        parameters on ``Net.parameters``. Calling this method on an
        already-registered net is a no-op that does not overwrite
        existing parameters.

        This is the declaration path; it ignores ``strict_nets`` because
        it is itself how a user declares nets under strict mode. The
        ``strict_nets`` flag only gates connection-time net auto-creation
        (see :meth:`_ensure_net`).

        Parameters
        ----------
        netname : str
            Name of the net to add.
        parameters : list, optional
            Legacy parameter list, stored on ``Net.parameters``.
        classification : PortRole, optional
            Explicit classification override for this net. When set,
            ``net.classification`` returns this value regardless of
            bound-pin roles. When omitted, ``net.classification`` is
            inferred from pin roles at read time (see
            :attr:`silk.circuit.net.Net.classification`).
        """
        if parameters is None:
            parameters = []
        # A net name is one SPICE node token too (TASK-75 AC #4): '' and
        # 'a b' used to register as nets, and the second one then emitted TWO
        # node tokens on every device line that referenced it. Checked before
        # canonicalization, which would otherwise raise a bare
        # AttributeError on a non-string.
        netname = validate_net_token(netname, who='add_net', kind='net')
        from silk.circuit.bus_spec import _guard_scalar_alias
        _guard_scalar_alias(self._module_ref, netname)
        netname = self._canonical_net_name(netname)
        if netname in self.net_dict:
            if self.debug:
                print(f'Net {netname} already exists. No need to re-add it.')
            return

        net = _bus_port_bit_net(self._module_ref, netname)
        if net is None:
            net = Net(name=netname, module=self._module_ref)
        netname = net.name
        net.parameters = list(parameters)
        if classification is not None:
            net._classification_override = classification
        self.net_dict[netname] = net

    def remove_net(self, netname):
        """Remove a net and purge any reverse-index bindings pointing at it.

        The mutation entry point for net deletion, so callers do not have to
        reach into ``net_dict`` directly (which is exposed read-only on
        ``Module``). Used to prune generated nets — e.g. layout parasitic
        midpoint nets. Cleaning the ``_pin_to_net`` reverse index here prevents
        a later rebind of a pin that was on the removed net from trying to
        clean up a now-deleted net.

        Raises ``KeyError`` if *netname* is not present (matches ``del`` on a
        dict — callers that prune a known set are unaffected).
        """
        del self.net_dict[netname]
        # Length-based staleness detection in _canonical_net_name can miss a
        # remove+add of a different name (same count) — drop the index.
        self._net_casefold.clear()
        stale = [key for key, bound in self._pin_to_net.items()
                 if bound == netname]
        for key in stale:
            del self._pin_to_net[key]

    def connect_element(self, elemName, pins, nets, _skipCheck=False, *, _undo=None):
        """Connect a defined element to a set of nets.

        Handles both DEVICE (Pin objects with only ``.net``) and MODULE
        (ModulePin objects with ``.net``, ``.direction``, ``.role``,
        ``.domain``) entries. Each per-pin connection funnels through
        :meth:`_bind_pin_to_net`, which keeps ``Net.pins`` consistent and
        handles rewires automatically.

        Args:
            elemName: Name of element to connect
            pins: List of pin names
            nets: List of net names (must match length of pins)
            _skipCheck: Internal flag to skip existence check
        """
        if elemName not in self.elem_dict.keys() and not _skipCheck:
            raise CircuitError('ERROR: Element does not exist. Run add_element first.')

        # Lengths must match exactly, checked before anything is written
        # (TASK-43 AC #3): a short `nets` used to die with a bare
        # `IndexError` on `nets[idx]` mid-loop, after already binding the
        # earlier pins; a long `nets` silently dropped the extras (no error,
        # no net created for them). Both were reported by
        # test_connect_element_length_mismatch pre-fix.
        if len(pins) != len(nets):
            raise CircuitError(
                f'ERROR: connect_element({elemName!r}): pins has {len(pins)} '
                f'entries but nets has {len(nets)}; lengths must match.'
            )

        # Get element info
        elemType = self.elem_dict[elemName]['type']
        elemDevice = self.elem_dict[elemName]['device']

        if elemType == ElementType.DEVICE:
            pinlist = self.elem_dict[elemName].get('pin_list', elemDevice.pin_list)
            binding_kind = 'device'
        elif elemType == ElementType.MODULE:
            pinlist = self.elem_dict[elemName].get('pin_list', elemDevice.pin_list)
            binding_kind = 'submodule'
        else:
            raise CircuitError(f'Unsupported element type {elemType!r} for element {elemName!r}')

        # Connect pins to nets
        type_name = _element_type_name(elemDevice)
        for idx, p in enumerate(pins):
            # The shared chokepoint (TASK-75): the value must be able to be
            # ONE node token, and a Net normalises to its name here exactly
            # as it does on the factory path. Runs before the net is created
            # so a rejected value leaves net_dict untouched.
            net_name = validate_net_token(
                nets[idx], who=pin_context(type_name, elemName, p), kind='pin')

            # Ensure the target net exists (raises if strict_nets).
            net_is_new = net_name not in self.net_dict
            if net_is_new and self.debug:
                print(f'WARNING: net {net_name} is undefined. add_net has been run in the background.')

            # Check pin exists
            if p not in pinlist.keys():
                raise CircuitError(f'ERROR: Pin {p} is not in defined pins for element {elemName}')

            # Build a PortBinding and funnel through the chokepoint. For DEVICE
            # entries, Pin objects carry only ``.net`` — direction/role/domain
            # are None. For MODULE entries, ModulePin carries direction/role/
            # domain (as Enum values) which must be unwrapped to strings to
            # match the existing PortBinding format.
            pin_obj = pinlist[p]
            direction_raw = getattr(pin_obj, 'direction', None)
            role_raw = getattr(pin_obj, 'role', None)
            domain = getattr(pin_obj, 'domain', None)
            self._bind_pin_to_net(net_name, PortBinding(
                net=net_name,
                instance=elemName,
                port=p,
                direction=direction_raw.value if hasattr(direction_raw, 'value') else direction_raw,
                role=role_raw.value if hasattr(role_raw, 'value') else role_raw,
                domain=domain,
                kind=binding_kind,
                # Role provenance rides along (ModulePin only; TASK-42).
                role_inferred=bool(getattr(pin_obj, 'role_inferred', False)),
            ), undo=_undo)
            pinlist[p].net = net_name

        if elemType != ElementType.MODULE:
            elemDevice.pin_list = pinlist
        self.elem_dict[elemName]['device'] = elemDevice

    def update_element(self, elemName, params, paramValues, _skipCheck=False):
        """Update parameter values in element

        Args:
            elemName: Name of element to update
            params: List of parameter names
            paramValues: List of parameter values (must match length of params)
            _skipCheck: Internal flag to skip existence check
        """
        if elemName not in self.elem_dict.keys() and not _skipCheck:
            raise CircuitError('ERROR: Element does not exist. Run add_element first.')

        if len(paramValues) < len(params):
            device = self.elem_dict[elemName]['device']
            constraints = getattr(device, 'param_list', {})
            trailing = params[len(paramValues):]
            if all(p in constraints and constraints[p].emit_if_supplied for p in trailing):
                params = params[:len(paramValues)]
        if len(params) != len(paramValues):
            raise CircuitError(f'ERROR: params length ({len(params)}) does not match paramValues length ({len(paramValues)})')

        # Get element info
        entry = self.elem_dict[elemName]
        elemType = entry['type']
        elemDevice = entry['device']
        derived = getattr(elemDevice, '_emission_spec', {}).get('derivedParams', {})
        readonly = set(params).intersection(derived)
        if readonly:
            raise TypeError(f'{elemName}: derived parameters {sorted(readonly)} are read-only')

        # AnalogElement instances store parameters in a `.params` dict;
        # legacy PDKDevice stores them in `.param_list[p].value`. Dispatch
        # on the type of the stored device rather than an auxiliary flag.
        if isinstance(elemDevice, AnalogElement):
            declared = elemDevice.__class__._param_names
            for idx, p in enumerate(params):
                if p in declared:
                    # Same path as ``mn.w = v``: records the parameter as
                    # caller-supplied so emission unit-converts and
                    # constraint-checks it, even when it was defaulted.
                    setattr(elemDevice, p, paramValues[idx])
                else:
                    elemDevice.params[p] = paramValues[idx]
            return

        if elemType != ElementType.DEVICE:
            raise CircuitError(f'update_element() only supports DEVICE elements; got {elemType!r} for {elemName!r}')

        tempParams = elemDevice.param_list

        # Import Variable here to avoid circular import
        from silk.circuit.variable import Variable

        for idx, p in enumerate(params):
            # Check if value is already a Variable or needs to be wrapped
            if isinstance(paramValues[idx], (int, float)):
                modVar = Variable(name=p)
                modVar.value = paramValues[idx]
            else:
                modVar = paramValues[idx]

            tempParams[p].value = modVar

        elemDevice.param_list = tempParams
        self.elem_dict[elemName]['device'] = elemDevice


# ---------------------------------------------------------------------------
# Module-level helpers used by add_element
# ---------------------------------------------------------------------------

def _element_type_name(elemDevice) -> str:
    """The device-type name a pin-value error should report.

    Matches what the factory path prints (``owner.__class__.__name__``, which
    for a generated device class IS the catalog key, e.g. ``'nmos1p8'``) for
    every shape ``add_element`` accepts — the class itself, an instance, a
    legacy ``PDKDevice`` copy, or a ``Module`` — so the same mistake reads the
    same on both construction styles.
    """
    if isinstance(elemDevice, type):
        return elemDevice.__name__
    if isinstance(elemDevice, AnalogElement):
        return type(elemDevice).__name__
    # A submodule reports as its own module name: `device_name` on a Module is
    # the literal marker 'module', which names nothing the author would
    # recognise.
    mod_name = getattr(elemDevice, 'module_name', None)
    if isinstance(mod_name, str) and mod_name:
        return mod_name
    name = getattr(elemDevice, 'device_name', None)
    return name if isinstance(name, str) and name else type(elemDevice).__name__


def _connection_pairs(nets, pin_order):
    """``[(pin, value)]`` from the three shapes ``nets=`` accepts.

    The shape dispatch is the one :func:`_apply_connections` has always used;
    it lives here so the pre-registration check and the binding itself read
    the SAME pairs out of the same argument. The positional form is zipped
    against *pin_order*, which drops a too-long list and shortens a too-short
    one — both are reported later by ``connect_element`` exactly as before
    (extras ignored / ``IndexError``); this helper only decides what to
    validate.
    """
    if isinstance(nets, dict):
        return list(nets.items())
    if nets and isinstance(nets[0], (tuple, list)):
        return [(p, n) for p, n in nets]
    return list(zip(pin_order, nets))


def _validate_connections(elemDevice, elemName, nets, pin_order):
    """Reject any net value that cannot be one SPICE node token.

    Called by ``add_element`` BEFORE the element is registered, so a rejected wiring dict leaves ``elem_dict`` and ``net_dict``
    untouched. The per-value rule is the factory path's own chokepoint
    (:func:`silk.elements.base.validate_net_token`), not a second copy of it.
    """
    type_name = _element_type_name(elemDevice)
    for pin, value in _connection_pairs(nets, pin_order):
        validate_net_token(value, who=pin_context(type_name, elemName, pin),
                           kind='pin')


def _apply_connections(circuit, elemName, nets, *, undo=None):
    """Apply net connections to a newly-added element.

    ``nets`` may be:
    - a dict ``{pin: net}``
    - a list of ``(pin, net)`` tuples
    - a positional list ``[net, net, ...]`` matched by pin_list order
    """
    elemDevice = circuit.elem_dict[elemName]['device']

    if isinstance(nets, dict):
        pins = list(nets.keys())
        net_values = list(nets.values())
    elif nets and isinstance(nets[0], (tuple, list)):
        # List of (pin, net) tuples
        pins = [p for p, n in nets]
        net_values = [n for p, n in nets]
    else:
        # Positional list — match by pin_list order. Kept unzipped (not via
        # _connection_pairs) so the historical length behaviour is unchanged:
        # connect_element still IndexErrors on a short list and ignores extras.
        pins = list(elemDevice.pin_list.keys())
        net_values = list(nets)

    circuit.connect_element(elemName, pins, net_values, _skipCheck=True, _undo=undo)


def _apply_parameters(circuit, elemName, parameters):
    """Apply parameter values to a newly-added element.

    ``parameters`` may be:
    - a dict ``{param: value}``
    - a list of ``(param, value)`` tuples
    - a positional list ``[value, value, ...]`` matched by param_list order
    """
    elemDevice = circuit.elem_dict[elemName]['device']

    if isinstance(parameters, dict):
        params = list(parameters.keys())
        param_values = list(parameters.values())
    elif parameters and isinstance(parameters[0], (tuple, list)):
        # List of (param, value) tuples
        params = [p for p, v in parameters]
        param_values = [v for p, v in parameters]
    else:
        params = list(elemDevice.param_list.keys())
        param_values = list(parameters)
        trailing = params[len(param_values):]
        if all(elemDevice.param_list[p].emit_if_supplied for p in trailing):
            params = params[:len(param_values)]

    circuit.update_element(elemName, params, param_values, _skipCheck=True)


# ---------------------------------------------------------------------------
# Backward-compatible alias. ``PyCircuitElements`` was
# the pynetlist-era name; kept so external code and the pre-split golden pickle
# import path (``silk.circuit.circuit_elements.PyCircuitElements``) keep
# resolving to the renamed class.
# ---------------------------------------------------------------------------
PyCircuitElements = CircuitGraph
