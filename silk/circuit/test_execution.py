"""Invocation-local simulation equivalence and retained launch evidence."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, replace, field
from uuid import uuid4
import os
import re
from silk.circuit.module_placement import with_definition_scope

_current = ContextVar('silk_test_invocation', default=None)


@dataclass(frozen=True)
class RunEvidence:
    """Frozen run facts; never a reference to an executing netlist container."""
    state: tuple
    corner: str
    conditions: object
    run_id: str = field(default_factory=lambda: uuid4().hex)
    run: object = None

    @classmethod
    def capture(cls, result):
        from silk.circuit.module_test import _SIMULATED_STATE_ATTRS
        state = {name: getattr(result.netlist, name)
                 for name in _SIMULATED_STATE_ATTRS if hasattr(result.netlist, name)}
        return cls(tuple(deepcopy(state).items()), result.corner_str,
                   deepcopy(result.conditions),
                   run_id=getattr(result, 'run_id', None) or uuid4().hex,
                   run=getattr(result, 'run', None))

    def result(self, spec):
        from types import SimpleNamespace
        from silk.circuit.module_test import _RunSimResult
        return _RunSimResult(SimpleNamespace(**dict(self.state)),
                             self.corner, deepcopy(self.conditions),
                             spec.invocation, spec.cache_key, self.run_id, self.run)


def sharing_identity(test, selection):
    """Restrict reuse to placements of one definition's named test."""
    from silk.circuit.module_test import ModuleTestbench
    if (selection is None or not selection.path
            or type(test).run is not ModuleTestbench.run
            or test.netlist.simulator not in {'ngspice', 'xyce'}):
        return None
    from silk.circuit.circuit_elements import _module_structure_projection
    from silk.circuit._behavioral_identity import behavioral_key, definition_hash
    from silk.exceptions import CircuitError
    definition = test.module
    try:
        structural_digest, provable = _module_structure_projection(definition)
        if not provable:
            return None
        digest = definition_hash(structural_digest,
                                 behavioral_key(definition))
    except CircuitError:
        # An opaque definition cannot prove equivalence; run each placement.
        return None
    return definition.module_name, test.test_name, digest


def normalized_deck_text(module, corner):
    """Disk-free normalized built-in emission, or an exclusion reason.

    Only silk-generated RunFiles paths and the fallback title are normalized.
    Eligibility is checked before definition validation can render any bodies.
    Covered decks use ordinary definition validation/name resolution. Lookup
    callers supply an isolated view. Completion policy is deliberately
    independent (including for pure Op).
    """
    from silk.simulation.completion import _opaque_emission_reason
    if module.simulator not in {'ngspice', 'xyce'}:
        return None, 'not a supported SPICE backend'
    if getattr(module, '_mc_sample', None):
        return None, 'Monte Carlo launch'
    reason = _opaque_emission_reason(module)
    if reason:
        return None, reason
    return _normalized_covered_deck_text(module, corner)


@with_definition_scope
def _normalized_covered_deck_text(module, corner):
    """Emit an eligible deck within its complete definition scope."""
    from silk.circuit.modules_netlist import _quiet_deck_text
    files = module.backend._resolve_run_files(
        '/__silk_execution__/deck.cir', '/__silk_execution__',
        analyses=list(module.sim_analysis_dict.values()), temperature=True)
    deck = _quiet_deck_text(module, corner, files,
                            module.circuit_name or 'silk test')
    if re.search(r'\b(gauss|agauss|unif|aunif|random|sgauss|sunif|trnoise|trrandom)\b', deck, re.I):
        return None, 'stochastic deck'
    for line in deck.splitlines():
        if line.lstrip().lower().startswith(('.include ', '.lib ')):
            parts = line.strip().split()
            if len(parts) < 2 or not os.path.isabs(parts[1].strip('"\'')):
                return None, 'relative include or library path'
    return deck, None


def normalized_deck_digest(module, corner):
    """Return a versioned digest/reason pair for normalized emission."""
    from hashlib import sha256
    deck, reason = normalized_deck_text(module, corner)
    return (None if deck is None else
            'silk.normalized-deck/v1:' + sha256(deck.encode('utf-8')).hexdigest()), reason


def equivalence(spec):
    """Compare emitted inputs before allocating or cleaning any run output.

    Only generated file locations are substituted, by emitting against one
    synthetic RunFiles. Opaque directives and stochastic inputs decline reuse.
    """
    if spec.sharing_identity is None:
        return None
    from silk.circuit.test_placement import prepared_execution, _active
    from silk.circuit.module_test import ModuleTestbench
    module = spec.netlist
    for selected in reversed(_active.get()):
        if selected.test.netlist is module:
            if type(selected.test).run is not ModuleTestbench.run:
                return None
            break
    with prepared_execution(spec):
        deck, reason = normalized_deck_text(module, spec.corner_str)
    if reason is not None:
        return None
    return (spec.sharing_identity, module.simulator, spec.corner_str, deck,
            repr(spec.declared_supply))


class Invocation:
    def __init__(self):
        self.runs = {}
        self.failures = {}
        self.observed = {}
        self.locations = {}

    def prepare(self, spec):
        if spec.execution_prepared:
            return spec
        key = equivalence(spec)
        identity = key if key is not None else object()
        location = spec.location
        if location is None:
            return spec
        directory, filename, mode = location
        destination = os.path.abspath(filename)
        previous = self.locations.get(destination)
        if previous is not None and previous != identity:
            # Only distinct simulations competing for one output need another
            # directory. Identical aliases keep main's original spelling.
            from silk.circuit.test_placement import _active
            selection = next((s for s in reversed(_active.get())
                              if s.test.netlist is spec.netlist), None)
            owner = selection.owner_path if selection is not None else str(len(self.locations))
            from urllib.parse import quote
            component = quote(owner, safe='')
            if len(component.encode()) > 240:
                from hashlib import sha256
                component = 'owner~' + sha256(component.encode()).hexdigest()
            directory = os.path.join(directory, component)
            filename = os.path.join(directory, os.path.basename(filename))
            os.makedirs(directory, exist_ok=True)
            location = (directory, filename, mode)
            if selection is not None:
                selection.location = location
            spec = replace(spec, location=location, tb_dir=directory)
        self.locations[os.path.abspath(filename)] = identity
        return replace(spec, execution_prepared=True)


@contextmanager
def invocation():
    value = Invocation()
    token = _current.set(value)
    try:
        yield value
    finally:
        from silk.circuit.test_placement import finish_invocation
        finish_invocation(value)
        _current.reset(token)


def current_invocation():
    return _current.get()
