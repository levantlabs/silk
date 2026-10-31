"""Per-call deck comparisons; never persistent validity or waveform caching."""
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import replace
import re
from types import SimpleNamespace
import warnings

from silk.db.models import DeckCheck
from silk.exceptions import EvidenceMismatchError


class _NotCheckable(Exception):
    """A deliberate coverage refusal, distinct from invalid configuration."""


def _emission_view(module, memo=None):
    """Copy emission topology/state without copying retained waveforms.

    Subcircuit emission primes child netlisters and refreshes graph registries.
    Isolate all those children, ports, bindings and dictionaries, not just the
    top-level netlister. Repeated definitions retain shared identity in the view.
    """
    from silk.circuit.circuit_elements import ElementType
    root = memo is None
    memo = {} if memo is None else memo
    if id(module) in memo:
        return memo[id(module)]
    view = copy(module)
    if '_source_context' in vars(module):
        view._source_context = copy(module._source_context)
    memo[id(module)] = view
    graph = copy(module._element_mgr)
    graph.__dict__ = {key: copy(value) if isinstance(value, (dict, list, set)) else value
                      for key, value in vars(graph).items()}
    view._element_mgr = graph
    graph._module_ref = view
    view.simulator_options = dict(module.simulator_options)
    view.sig_sources_dict = {}
    for name, source in module.sig_sources_dict.items():
        key = ('source', id(source))
        if key not in memo:
            memo[key] = copy(source)
        view.sig_sources_dict[name] = memo[key]
    view.pin_list = (None if module.pin_list is None else
                     {name: copy(pin) for name, pin in module.pin_list.items()})
    graph.net_dict = {name: copy(net) for name, net in graph.net_dict.items()}
    for net in graph.net_dict.values():
        net.module = view
        net.pins = [copy(binding) for binding in net.pins]
    graph.elem_dict = {name: dict(entry) for name, entry in graph.elem_dict.items()}
    for entry in graph.elem_dict.values():
        if entry['type'] == ElementType.MODULE:
            entry['device'] = _emission_view(entry['device'], memo)
        if 'pin_list' in entry:
            entry['pin_list'] = {name: copy(pin) for name, pin in entry['pin_list'].items()}
    graph.module_dict = {name: dict(entry, device=_emission_view(entry['device'], memo))
                         for name, entry in graph.module_dict.items()}
    view.netlister = deepcopy(view.netlister)
    if root:
        from silk.circuit.object_holders import register, supported
        # Ordinary device entries intentionally remain shared read-only views.
        # Only copied sources acquire this view's wiring subscriptions.
        contexts = {id(s): getattr(s, '_module', None)
                    for key, m in memo.items() if isinstance(key, int)
                    for s in m.sig_sources_dict.values() if supported(s)}
        for key, copied_module in list(memo.items()):
            if not isinstance(key, int):
                continue
            for name, source in copied_module.sig_sources_dict.items():
                if supported(source):
                    context = contexts[id(source)]
                    object.__setattr__(source, '_module', memo.get(id(context), copied_module))
                    register(copied_module, 'sig_sources_dict', name, source)
    return view


@contextmanager
def _configured_entry(entry, run=None):
    """A temporary selection and emission view, without registering evidence."""
    from silk.circuit.test_placement import _active, ExecutionSelection, TestEvidence
    from silk.circuit.module_test import (
        _configure_capture, _configure_conditions, _planned_conditions,
        _probe_preflight_reason, ModuleTestbench,
    )
    # Testbench copying is a serialization path that drops free probes.
    test = object.__new__(type(entry.test))
    test.__dict__.update(vars(entry.test))
    test.netlist = _emission_view(test.netlist) if test.netlist is not None else None
    entry = replace(entry, test=test)
    if type(test).run is not ModuleTestbench.run:
        raise _NotCheckable('custom test run')
    if not test._module_added:
        raise ValueError(f'Must call add_module_under_test() before running test {test.test_name}')
    if test.netlist is None:
        raise ValueError(f'Netlist not created for test {test.test_name}')
    test._check_sweep_sources()
    token = _active.set((*_active.get(), ExecutionSelection(entry, TestEvidence())))
    try:
        module = test.netlist
        with _planned_conditions(test, run):
            _configure_conditions(test, module)
            if module.simulator not in {'ngspice', 'xyce'}:
                raise _NotCheckable('not a supported SPICE backend')
            if test.monte_carlo is not None:
                raise _NotCheckable('Monte Carlo test')
            reason = _probe_preflight_reason(test, module.simulator)
            if reason is not None:
                raise _NotCheckable(reason)
            _configure_capture(test, module)
            # Actual preparation clears a previous sample for an ordinary run.
            module._mc_sample = None
            yield module
    finally:
        _active.reset(token)


class CheckContext:
    """One public lookup: reuse current digests, never historical verdicts."""

    def __init__(self, tracker):
        self.db = tracker.db
        from silk.circuit.modules_testbench import _walk_test_entries
        self.entries = {}
        for entry, _, _ in _walk_test_entries(_emission_view(tracker._module), recursive=True):
            self.entries.setdefault((entry.owner_path, entry.test_name), []).append(entry)
        self.current = {}
        self.unchecked = set()

    def _unknown(self, reason, producer=''):
        detail = f'{producer}: {reason}' if producer else reason
        self.unchecked.add(detail)
        return DeckCheck('not checked', detail)

    def row(self, record):
        provenance = self.db.get_run_provenance(record.provenance_id)
        binding = None if provenance is None else provenance.producing_test
        name = binding.get('test_name') if isinstance(binding, dict) else None
        path = binding.get('owner_path') if isinstance(binding, dict) else None
        producer = f"test {name!r} at {path}" if name else ''
        if (provenance is None or provenance.run_origin != 'silk-launch'
                or provenance.simulator not in {'ngspice', 'xyce'}):
            check = self._unknown('no covered silk SPICE launch', producer)
        elif (not isinstance(provenance.normalized_deck_sha256, str)
              or not re.fullmatch(r'silk\.normalized-deck/v1:[0-9a-f]{64}',
                                  provenance.normalized_deck_sha256)):
            check = self._unknown('missing or unsupported normalized deck; rerun a covered SPICE bench', producer)
        elif (not isinstance(name, str) or not isinstance(path, str)
              or not isinstance(provenance.launch_corner_label, str)):
            check = self._unknown('missing producing test or launch corner label; rerun', producer)
        else:
            entries = self.entries.get((path, name), ())
            if len(entries) != 1:
                check = self._unknown('producing test is missing, renamed or ambiguous on this root', producer)
            else:
                entry = entries[0]
                if entry.test.netlist is None or not entry.test._module_added:
                    return replace(record, deck_check=self._unknown(
                        'producing test is registered but not wired in this session; wire it to check',
                        producer))
                backend = (entry.test.netlist.simulator if entry.test.netlist._simulator_explicit
                           else entry.module.simulator)
                key = (path, name, provenance.launch_corner_label, backend)
                if key not in self.current:
                    self.current[key] = self._emit(entry, provenance.launch_corner_label)
                digests, reason = self.current[key]
                if reason is not None:
                    check = self._unknown(reason, producer)
                elif (backend != provenance.simulator
                      or provenance.normalized_deck_sha256 not in digests):
                    corner = provenance.launch_corner_label
                    rerun = (f"dut.run_test({name!r}, corner={corner!r}, store=True)"
                             if not entry.path else
                             f"dut.store_report(dut.get_test({name!r}, instance={entry.path!r}).run(corner={corner!r}))")
                    if entry.test.corners is not None or self._has_grid(entry):
                        selected = (f'dut.get_test({name!r}, instance={entry.path!r})'
                                    if entry.path else f'dut.get_test({name!r})')
                        rerun = (f'for report in {selected}.run(corner={corner!r}).values(): '
                                 'dut.store_report(report)')
                    label = getattr(record, 'parameter', 'measurement')
                    check = DeckCheck('stale',
                        f"{label} was stored from {producer}, whose netlist has changed since "
                        f"(deck differs or backend changed); rerun it: "
                        f"{rerun}")
                else:
                    check = DeckCheck('checked', 'producing deck and backend match')
        return replace(record, deck_check=check)

    @staticmethod
    def _has_grid(entry):
        from silk.circuit.test_placement import _active, ExecutionSelection, TestEvidence
        token = _active.set((*_active.get(), ExecutionSelection(entry, TestEvidence())))
        try:
            return entry.test._has_condition_grid()
        finally:
            _active.reset(token)

    @staticmethod
    def _emit(entry, corner):
        from silk.circuit.module_test import ModuleTestbench
        from silk.circuit.test_execution import normalized_deck_digest
        from silk.circuit.test_placement import (
            _active, ExecutionSelection, TestEvidence, prepared_execution,
        )
        try:
            # Refuse before planning: Monte Carlo planning can allocate output.
            if type(entry.test).run is not ModuleTestbench.run:
                raise _NotCheckable('custom test run')
            if entry.test._variable_lists:
                raise _NotCheckable('Variable-list test plan')
            if entry.test.monte_carlo is not None:
                with _configured_entry(entry):
                    raise _NotCheckable('Monte Carlo test')
            token = _active.set((*_active.get(), ExecutionSelection(entry, TestEvidence())))
            try:
                planned = entry.test._plan_runs(corner=corner)
            finally:
                _active.reset(token)
            digests = set()
            for run in planned if planned is not None else (None,):
                with _configured_entry(entry, run) as module:
                    corners = getattr(module.pdk_details, 'corners', {}) or {}
                    if corners and corner not in corners:
                        raise ValueError(f'unknown corner {corner!r}; available: {sorted(corners)}')
                    spec = SimpleNamespace(netlist=module, declared_supply=None,
                                           location=None)
                    with prepared_execution(spec):
                        digest, reason = normalized_deck_digest(module, corner)
                    if reason is not None:
                        raise _NotCheckable(reason)
                    digests.add(digest)
            return digests, None
        except _NotCheckable as exc:
            # Only a deliberate no-preflight/coverage refusal is unknown.
            # Invalid settings/emission continue to raise their typed errors.
            return set(), str(exc)

    def aggregate(self, evidence):
        metrics = tuple(self.row(row) for row in evidence.metrics)
        specs = tuple(replace(spec, measurements=tuple(self.row(row) for row in spec.measurements))
                      for spec in evidence.specs)
        currents = replace(evidence.current_set, deck_check=self._unknown(
            'current annotations are not frozen at launch'))
        stale = [row.deck_check.reason for row in metrics
                 if row.deck_check.state == 'stale']
        stale += [row.deck_check.reason for spec in specs for row in spec.measurements
                  if row.deck_check.state == 'stale']
        check = (DeckCheck('stale', '; '.join(stale)) if stale else
                 DeckCheck('not checked', currents.deck_check.reason))
        return replace(evidence, metrics=metrics, specs=specs,
                       current_set=currents, deck_check=check)

    @staticmethod
    def refuse_stale(records):
        reasons = [row.deck_check.reason for row in records
                   if row.deck_check.state == 'stale']
        if reasons:
            raise EvidenceMismatchError('; '.join(reasons))

    def warn(self):
        if self.unchecked:
            warnings.warn('Stored evidence not checked: ' + '; '.join(sorted(self.unchecked)),
                          UserWarning, stacklevel=3)
