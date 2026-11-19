"""Private selected-test context and root-owned evidence (no circuit copies)."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from inspect import getattr_static
import weakref

from silk.exceptions import ConfigurationError

_active = ContextVar('silk_test_selection', default=())


#: Selected-test methods whose unbound results must stay bound to that selected test.
_SELECTED_PROBE_CALLS = frozenset({'probe', 'probe_bus'})


@dataclass
class TestEvidence:
    values: dict = field(default_factory=lambda: {
        '_sweep_table': {}, '_sweep_states': {}, '_sweep_generations': {},
        '_mc_samples': None, '_record_measurements': True,
        '_cache': {},
    })
    report: object = None
    run_id: object = None
    results: object = None
    corners: dict = field(default_factory=dict)


@dataclass(eq=False)
class EvidenceIndex:
    selectors: dict = field(default_factory=dict)
    records: dict = field(default_factory=dict)
    annotations: dict = field(default_factory=dict)
    root_ref: object = None
    scoped_specs: dict = field(default_factory=dict)
    projection_associations: dict = field(default_factory=dict)


def evidence_index(root):
    index = root.__dict__.get('_test_evidence')
    if index is None:
        index = EvidenceIndex(root_ref=weakref.ref(root))
        root.__dict__['_test_evidence'] = index
    return index


def authored_test(test):
    return test._authored if isinstance(test, SelectedTestbench) else test


@dataclass
class ExecutionSelection:
    entry: object
    evidence: TestEvidence
    location: object = None
    declared_supply: object = None
    separate_output: bool = False
    executing: bool = False
    reader: object = None

    @property
    def test(self):
        return authored_test(self.entry.test)

    def __getattr__(self, name):
        return getattr(self.entry, name)


def current_placement(test):
    for selection in reversed(_active.get()):
        if selection.test is test:
            return selection
    return None


class CampaignSelection:
    """Private, non-publishing selection used to score one fault-campaign run.

    While active, :func:`current_placement` returns it for the authored bench,
    so every result read (probes captured in user lambdas included) and every
    result write goes to this run's reader and throwaway evidence, never to the
    authored netlist or a retained nominal record. It has no ``entry``: it is
    not part of any evidence index.
    """
    executing = False
    location = None
    declared_supply = None
    separate_output = False
    path = ()
    entry = None

    def __init__(self, test, run):
        self.test = test
        self.run = run
        self.reader = run.reader
        self.owner_path = test.owner_path
        self.report_key = test.report_key
        self.module_name = test._placement_module_name
        self.inherited_supply = test._inherited_supply()
        self.inherited_temperature = test._inherited_temperature()
        self.output_mode = test.module._output_mode
        self.evidence = TestEvidence()
        self.evidence.values['_record_measurements'] = False
        # The explicit-corner probe route finds the run's reader here.
        self.evidence.corners[run.label] = run.reader


def campaign_run(test):
    """The active fault-campaign run scoring *test*, or ``None``."""
    selection = current_placement(test)
    return getattr(selection, 'run', None) if isinstance(selection, CampaignSelection) else None


@contextmanager
def campaign_evaluation(test, run):
    """Score *test* against *run* without touching the authored bench."""
    if current_placement(test) is not None:
        from silk.exceptions import NotSupportedError
        raise NotSupportedError(
            f'A fault campaign cannot score test {test.test_name!r} while a '
            f'placement selection for it is active.')
    from silk.specs.attempt import _evaluation_attempt
    import time
    selection = CampaignSelection(test, run)
    token = _active.set((*_active.get(), selection))
    # Secondary guard: even if a publication path were reached, an attempt
    # frame marked suppressed records nothing on any Spec.
    frame = _evaluation_attempt.set((
        {'suppressed': True, 'attempt_id': 'campaign', 'requested_conditions': None},
        time.time()))
    try:
        yield
    finally:
        _evaluation_attempt.reset(frame)
        _active.reset(token)


class ResultAttribute:
    """Route result bookkeeping through the active owner, never configuration."""
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, test, owner=None):
        if test is None:
            return self
        selection = current_placement(test)
        values = test.__dict__ if selection is None else selection.evidence.values
        if selection is None:
            tracked, record = latest_record(test)
            if tracked:
                values = record.values if record is not None else {}
        try:
            return values[self.name]
        except KeyError:
            raise AttributeError(self.name) from None

    def __set__(self, test, value):
        selection = current_placement(test)
        values = test.__dict__ if selection is None else selection.evidence.values
        values[self.name] = value


@contextmanager
def execution_scope(entry, *, executing=False):
    """Bind placement evidence; only executing scopes restore temporary conditions."""
    test = authored_test(entry.test)
    active = current_placement(test)
    if (active is not None and active.entry == entry
            and (active.executing or not executing)):
        yield test
        return
    root = entry._root_ref()
    if root is None:
        raise ConfigurationError('The root for this selected test no longer exists.')
    key = (entry.owner_path, id(test))
    record = evidence_index(root).records.setdefault(key, TestEvidence())
    selection = ExecutionSelection(entry, record, executing=executing)
    netlist = getattr(test, 'netlist', None)
    saved = None
    if executing and netlist is not None:
        # Preparation helpers derive these effective settings. Restore
        # their authored inputs; owner/path/selection metadata stays in ContextVar.
        saved = (netlist._simulator, netlist.netlister, dict(netlist.simulator_options),
                 netlist._inherited_simulator_option_keys, netlist._temperature_basis,
                 netlist._supply_voltage, netlist._supply_source,
                 [(source, source.dc) for source in netlist.sig_sources_dict.values()
                  if hasattr(source, 'dc')])
    token = _active.set((*_active.get(), selection))
    try:
        from silk.specs.evidence import owner_scope
        with owner_scope(evidence_index(root), entry.owner_path):
            yield test
    finally:
        if saved is not None:
            (netlist._simulator, netlist.netlister, options, netlist._inherited_simulator_option_keys,
             netlist._temperature_basis, netlist._supply_voltage,
             netlist._supply_source, sources) = saved
            netlist.simulator_options.clear()
            netlist.simulator_options.update(options)
            for source, value in sources:
                source.dc = value
        _active.reset(token)


@contextmanager
def unselected_scope(test):
    token = _active.set(tuple(s for s in _active.get() if s.test is not test))
    try:
        yield test
    finally:
        _active.reset(token)


class _SelectedBenchRef(weakref.ref):
    """Retain selected provenance even after the weak target expires."""
    __slots__ = ()


class SelectedTestbench:
    """A root-qualified view with shared configuration and fluent selection."""
    def __init__(self, root, entry):
        self._root = weakref.ref(root)
        self._authored = entry.test
        self._owner_path = entry.owner_path

    def __setattr__(self, name, value):
        if (name.startswith('_')
                or isinstance(getattr_static(type(self), name, None), property)):
            object.__setattr__(self, name, value)
            return
        prop = getattr_static(type(self._authored), name, None)
        if not isinstance(prop, property):
            object.__setattr__(self, name, value)
            return
        if prop.fset is None:
            # Let Python refuse without creating placement evidence.
            prop.__set__(self._authored, value)
        else:
            with execution_scope(self._entry()):
                prop.__set__(self._authored, value)

    def __getstate__(self):
        raise TypeError(
            'SelectedTestbench holds a weakref (weak reference) to its root '
            'and cannot be pickled.')

    def __deepcopy__(self, memo):
        from copy import deepcopy
        new = type(self).__new__(type(self))
        memo[id(self)] = new
        root = self._root()
        if root is None:
            raise ConfigurationError('Cannot copy a selected test whose root has expired.')
        copied_root = deepcopy(root, memo)
        new._root = weakref.ref(copied_root)
        new._authored = deepcopy(self._authored, memo)
        new._owner_path = self._owner_path
        evidence_index(copied_root).selectors[(new._owner_path, id(new._authored))] = new
        return new

    def _entry(self):
        root = self._root()
        if root is None:
            raise ConfigurationError('The root for this selected test no longer exists.')
        for entry in root.list_test_entries():
            if entry.owner_path == self._owner_path and entry.test is self:
                return entry
        raise ConfigurationError(f'Selected test at {self._owner_path!r} was removed or replaced.')

    def validate_simulation_setup(self):
        """Read-only setup check for this selected bench.

        Unlike the other bench methods this never enters ``execution_scope``
        (which records evidence and may materialize the lazy netlister): it reads
        the entry's inherited conditions from an unpruned walk and checks a
        private view.
        """
        root = self._root()
        if root is None:
            raise ConfigurationError('The root for this selected test no longer exists.')
        from silk.circuit.modules_testbench import _walk_test_entries
        from silk.simulation._setup_check import private_view
        # Walk a private graph: the walk refreshes definitions and syncs pins.
        for entry, _, _ in _walk_test_entries(private_view(root), recursive=True):
            if entry.owner_path == self._owner_path and entry.test is self._authored:
                return self._authored._validate_setup(
                    entry.inherited_temperature, entry.inherited_supply)
        raise ConfigurationError(f'Selected test at {self._owner_path!r} was removed or replaced.')

    @property
    def save_device_params(self):
        """The authored bench's per-test capture setting."""
        return self._authored.save_device_params

    @save_device_params.setter
    def save_device_params(self, value):
        # Configure the executing (authored) bench, validated by its setter,
        # exactly as other selected-handle configuration does.
        self._authored.save_device_params = value

    def __getattr__(self, name):
        entry = self._entry()
        with execution_scope(entry):
            if name == 'report':
                return current_placement(self._authored).evidence.report
            if name == 'metrics':
                return dict(getattr(self._authored, '_metric_values', {}))
            value = getattr(self._authored, name)
            if name == 'netlist':
                return None if value is None else _SelectedNetlist(self)
        if not callable(value):
            return value

        @wraps(value)
        def call(*args, **kwargs):
            if name in _SELECTED_PROBE_CALLS:
                from silk.circuit.placement_reference import foreign_bench_reference
                for arg in (*args, *kwargs.values()):
                    if foreign_bench_reference(arg, self):
                        raise ConfigurationError(
                            'This bench reference was made by another selected test '
                            'or the authored bench; call bus() on the selected test '
                            'you probe.')
            with execution_scope(self._entry(), executing=(name in {'run', 'prepare', 'run_corners', 'run_monte_carlo'} or (name == 'evaluate' and kwargs.get('rawfile') is not None))):
                previous_report = current_placement(self._authored).evidence.report
                if name in _SELECTED_PROBE_CALLS:
                    from silk.circuit.placement_reference import selected_probe_call
                    flag = selected_probe_call.set(True)
                    try:
                        result = getattr(self._authored, name)(*args, **kwargs)
                    finally:
                        selected_probe_call.reset(flag)
                else:
                    result = getattr(self._authored, name)(*args, **kwargs)
                record = current_placement(self._authored).evidence
                if name in {'run', 'evaluate', 'run_corners', 'run_monte_carlo'}:
                    # An empty corner request returns a collection but publishes
                    # no replacement evidence to wrap. Keep the prior report.
                    if name != 'run_corners' or record.report is not previous_report:
                        record.report = result
                if name == 'bus':
                    result._set_owner(self)
                if name in _SELECTED_PROBE_CALLS:
                    for leaf in result._walk_leaves():
                        if leaf._is_unbound():
                            leaf._bench = _SelectedBenchRef(self)
                if name == 'add_metric':
                    # Declarations remain shared; returned values and fluent
                    # calls belong to the receiver that registered them.
                    from silk.circuit.metric_handle import MetricHandle
                    if isinstance(result, MetricHandle):
                        result._bench = self
                return self if result is self._authored else result
        return call


class _SelectedNetlist:
    """Shared circuit configuration with late-bound selected result reads."""
    _RESULT_METHODS = frozenset({
        'view', 'load_signal', 'get_signal_names', 'plot', 'read_raw_file',
        'annotate_currents',
    })
    _EXECUTION_METHODS = frozenset({
        'load_file', 'run_simulator', 'simulate', 'simulate_behavioral',
    })

    def __init__(self, selected):
        object.__setattr__(self, '_selected', selected)

    def _module(self):
        self._selected._entry()
        return self._selected._authored.netlist

    @property
    def __class__(self):
        return self._selected._authored.netlist.__class__

    def _unavailable(self, name):
        raise ConfigurationError(
            f'Selected test {self._selected._authored.test_name!r} at '
            f'{self._selected._owner_path!r} has no retained results for '
            f'{name!r}. Call run() or evaluate(rawfile=...) on the selected test first.')

    def _reader(self, name):
        with execution_scope(self._selected._entry()):
            reader = self._selected._authored._result_netlist()
        if reader is None or not reader.__dict__.get('sim_results_loaded', False):
            self._unavailable(name)
        return reader

    def __getattr__(self, name):
        from silk.circuit.module_test import _SIMULATED_STATE_ATTRS
        module = self._module()
        if ((name in _SIMULATED_STATE_ATTRS and name not in {
                'sim_analysis_dict', 'simulator_options', 'save_device_params',
                'save_device_currents', 'terminal_current_method', '_pending_device_probes'})
                or name in {'characterization_context'}):
            reader = self._reader(name)
            # Missing result state must never pick up a sibling's latest state.
            if name not in reader.__dict__:
                self._unavailable(name)
            return getattr(reader, name)
        value = getattr(module, name)
        if not callable(value):
            return value

        @wraps(value)
        def call(*args, **kwargs):
            if name in self._EXECUTION_METHODS:
                self._selected._entry()
                raise ConfigurationError(
                    f'Selected test {self._selected._authored.test_name!r} at '
                    f'{self._selected._owner_path!r}: netlist.{name}() cannot '
                    'publish selected results. Use the selected test\'s run() '
                    'or evaluate(rawfile=...) instead.')
            if name in self._RESULT_METHODS:
                return getattr(self._reader(name), name)(*args, **kwargs)
            authored = self._module()
            result = getattr(authored, name)(*args, **kwargs)
            return self if result is authored else result
        return call

    def __setattr__(self, name, value):
        setattr(self._module(), name, value)

    def __delattr__(self, name):
        delattr(self._module(), name)

    def __enter__(self):
        module = self._module()
        result = module.__enter__()
        return self if result is module else result

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._module().__exit__(exc_type, exc_val, exc_tb)

    def __copy__(self):
        return type(self)(self._selected)

    def __deepcopy__(self, memo):
        from copy import deepcopy
        new = type(self).__new__(type(self))
        memo[id(self)] = new
        object.__setattr__(new, '_selected', deepcopy(self._selected, memo))
        return new

    def __reduce_ex__(self, protocol):
        raise TypeError(
            'A selected netlist holds a weakref to its root and cannot be pickled '
            'or sent to a simulation worker. Construct FaultCampaign with the '
            'authored testbench netlist instead.')


def selected_probe_pickle_bindings(root):
    """Graph-owned selected bindings; Signals alone never serialize a root."""
    from silk.circuit.module_test import _metric_slot_leaves
    from silk.circuit.circuit_elements import ElementType
    pending = [root]
    visited = set()
    leaves = set()
    bindings = []
    while pending:
        module = pending.pop()
        if id(module) in visited:
            continue
        visited.add(id(module))
        for test in module._tests.values():
            for leaf in _metric_slot_leaves(test._metrics):
                ref = leaf.__dict__.get('_bench')
                selected = ref() if ref is not None else None
                if (isinstance(selected, SelectedTestbench) and selected._root() is root
                        and id(leaf) not in leaves):
                    leaves.add(id(leaf))
                    bindings.append((leaf, selected._owner_path, selected._authored))
        pending.extend(entry['device'] for entry in module.elem_dict.values()
                       if entry.get('type') == ElementType.MODULE)
    return bindings


def restore_selected_probe_bindings(root, bindings):
    """Restore exact graph/owner/test bindings using pickle's shared objects."""
    selectors = evidence_index(root).selectors
    for leaf, owner_path, authored in bindings:
        key = (owner_path, id(authored))
        selected = selectors.get(key)
        if selected is None:
            selected = SelectedTestbench.__new__(SelectedTestbench)
            selected._root = weakref.ref(root)
            selected._authored = authored
            selected._owner_path = owner_path
            selectors[key] = selected
        leaf._bench = _SelectedBenchRef(selected)


def selected_test(root, entry):
    if entry.path == '':
        return entry.test
    key = (entry.owner_path, id(entry.test))
    selectors = evidence_index(root).selectors
    if key not in selectors:
        selectors[key] = SelectedTestbench(root, entry)
    return selectors[key]


_worker_execution = ContextVar('silk_prepared_execution', default=None)


def execution_location(module):
    for selection in reversed(_active.get()):
        if getattr(selection.test, 'netlist', None) is module and selection.location is not None:
            return selection.location
    worker = _worker_execution.get()
    if worker is not None and worker.netlist is module and worker.location is not None:
        return worker.location
    name = module.netlistname
    if '/' in name:
        raise ConfigurationError(
            f'netlistname {name!r} is the deck file name, not a path; put the '
            f'folder in dir_folder ({module.dir_folder!r}).')
    return module.dir_folder, module.dir_folder + '/' + name, module._output_mode


def execution_supply_declaration(module):
    for selection in reversed(_active.get()):
        if getattr(selection.test, 'netlist', None) is module:
            return selection.declared_supply
    worker = _worker_execution.get()
    if worker is not None and worker.netlist is module:
        return worker.declared_supply
    return module._declared_supply


@contextmanager
def prepared_execution(spec):
    token = _worker_execution.set(spec)
    saved = [(var, var._value) for var, value in getattr(spec, 'variable_values', ())]
    try:
        for var, value in getattr(spec, 'variable_values', ()):
            var.value = value
        yield
    finally:
        for var, value in saved:
            var._value = value
        _worker_execution.reset(token)


class EvidenceResults:
    """A run's result reader over the live circuit, not a circuit copy."""
    @property
    def __class__(self):
        # Standard proxy identity: existing Module consumers must dispatch
        # on this circuit view while reading its retained simulation state.
        return self._module.__class__

    def __init__(self, module, *, authored=None, _retained=False, _state=None):
        from copy import copy, deepcopy
        from silk.circuit.module_test import _SIMULATED_STATE_ATTRS
        self._module = authored if authored is not None else module
        values = vars(module)
        state = (dict(_state) if _state is not None else
                 {name: values[name] for name in _SIMULATED_STATE_ATTRS
                  if name in values})
        if _state is not None:
            self.__dict__.update(state)
            from silk.simulation.backend import SimulatorBackend
            self.backend = SimulatorBackend.get(self.simulator)
            return
        # A native-only reader has no RawResult. Keep that explicit so a
        # waveform read can use its own native data, never the live circuit's.
        state.setdefault('_raw', None)
        # External rawfiles have no launch/preparation/sample facts. Snapshot
        # their absence too, rather than borrowing a later run's metadata.
        for name in ('_last_sim_corner', '_last_sim_label', '_launch_record',
                     '_prepared_deck_record', '_mc_sample'):
            state.setdefault(name, None)
        state.setdefault('_behavioral_mode', False)
        self.__dict__.update(state)
        conditions = state.get('operating_conditions')
        self.simulator = getattr(conditions, 'backend', None) or self._module.simulator
        from silk.simulation.backend import SimulatorBackend
        self.backend = SimulatorBackend.get(self.simulator)
        if (state.get('_raw') is None and 'simResultsAll' not in state
                and state.get('sim_results_loaded') and state.get('results_file')):
            self._raw = self.backend.load_raw(state['results_file'],
                                               simulator=self.simulator)
        self.characterization_context = getattr(module, 'characterization_context', None)
        self.sim_analysis_dict = deepcopy(
            getattr(module, 'sim_analysis_dict', self._module.sim_analysis_dict))
        self.simulator_options = dict(
            getattr(module, 'simulator_options', self._module.simulator_options))
        for name in ('save_device_params', 'save_device_currents',
                     'terminal_current_method', '_pending_device_probes'):
            setattr(self, name, deepcopy(
                getattr(module, name, getattr(self._module, name, None))))
        self.dir_folder, path, _ = execution_location(self._module)
        self.netlistname = path.split('/')[-1]
        self._retained = _retained
        if _retained:
            from silk.signals.rawfile import RawFile, read_rawfile
            from silk.simulation.backend import RawResult
            from silk.simulation.generation import attach, artifact_read
            from silk.exceptions import NotSupportedError
            raw = self._raw
            native = getattr(raw, 'native', None) if raw is not None else state.get('simResultsAll')
            if isinstance(native, RawFile) and native.source_path:
                raw = copy(raw) if raw is not None else RawResult(native, native.source_path)
                if (getattr(raw, '_generation', None) is None
                        and getattr(native, '_generation', None) is not None):
                    raw._generation = native._generation
                if getattr(raw, '_generation', None) is not None:
                    with artifact_read(raw, 'rawfile_path') as source:
                        raw.native = read_rawfile(source, lazy=True)
                        attach(raw, raw._generation)
                else:
                    stores = {id(vector._store): vector._store
                              for plot in native.plots
                              for vector in plot.vectors if vector._store is not None}
                    source = next(iter(stores.values())).path if stores else native.source_path
                    for store in stores.values():
                        try:
                            with open(store.path, 'rb') as handle:
                                store.check(handle, 'retention')
                        except OSError as exc:
                            raise store.error('retention', str(exc)) from exc
                    indexed = read_rawfile(source, lazy=True)
                    for store in stores.values():
                        try:
                            with open(store.path, 'rb') as handle:
                                store.check(handle, 'retention')
                        except OSError as exc:
                            raise store.error('retention', str(exc)) from exc
                    raw.native = indexed
                for plot in raw.native.plots:
                    for vector in plot.vectors:
                        vector._cache_data = False
                self.simResultsAll = raw.native
            elif (native is None and self.simulator == 'xyce'
                  and getattr(getattr(raw, 'run_files', None), 'noise_path', None)):
                # Xyce noise-only evidence is a companion-file descriptor.
                raw = copy(raw)
                self.simResultsAll = None
            else:
                raise NotSupportedError(
                    f"Backend {self.simulator!r} cannot retain this result: "
                    "a rereadable result artifact is required.")
            raw._preflight_unavailable_params = deepcopy(
                getattr(raw, '_preflight_unavailable_params',
                        getattr(getattr(module, 'netlister', None),
                                '_preflight_unavailable_params', None)) or {})
            self._raw = raw

    def __copy__(self):
        new = type(self).__new__(type(self))
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo):
        from copy import deepcopy
        new = type(self).__new__(type(self))
        memo[id(self)] = new
        for name, value in self.__dict__.items():
            # Ownership remains with the live root and original definition;
            # copying the reader must not publish into a copied owner.
            if name == '_evidence_owner':
                new.__dict__[name] = value
                continue
            try:
                new.__dict__[name] = deepcopy(value, memo)
            except (TypeError, RecursionError):
                # Match Module's treatment of native rawfile readers.
                new.__dict__[name] = value
        return new

    def __reduce_ex__(self, protocol):
        if '_evidence_owner' not in self.__dict__:
            from silk.circuit.modules import _reconstruct_module
            state = dict(self.__dict__)
            state.pop('backend', None)
            return _reconstruct_module, (type(self),), state
        raise TypeError(
            'EvidenceResults cannot be pickled: retained results reference a '
            'live circuit and process-local evidence owner. Use copy.copy() '
            'or copy.deepcopy() for an in-process reader copy.')

    def __getattr__(self, name):
        from types import MethodType
        from silk.circuit.module_test import _SIMULATED_STATE_ATTRS
        if (name in _SIMULATED_STATE_ATTRS
                or name in {'characterization_context'}):
            owner = self.__dict__.get('_evidence_owner')
            where = f' for owner {owner[2]!r}' if owner is not None else ''
            raise ConfigurationError(
                f'Retained simulation results{where} are missing {name!r}. '
                'Call run() or evaluate(rawfile=...) on the testbench '
                'to obtain complete results.')
        if name == 'backend':
            from silk.simulation.backend import SimulatorBackend
            return SimulatorBackend.get(self.simulator)
        module = object.__getattribute__(self, '_module')
        value = getattr(module, name)
        if isinstance(value, MethodType) and value.__self__ is module:
            return MethodType(value.__func__, self)
        return value


def publish_report(test, report, module=None):
    selection = current_placement(test)
    if selection is None:
        return
    results = EvidenceResults(module if module is not None else test.netlist, authored=test.netlist, _retained=True)
    from silk.circuit.modules_results import DUT_ELEMENT_NAME
    from silk.circuit.emitted_names import emitted_submodule_name
    results._evidence_owner = (selection.entry._root_ref, selection.entry.module,
                               selection.owner_path, emitted_submodule_name(DUT_ELEMENT_NAME))
    report.sim_results = results
    test._sweep_states = {
        key: {name: value for name, value in vars(results).items()
              if name not in {'_module', '_evidence_owner', 'backend'}}
        for key in test._sweep_table}
    selection.evidence.report = report
    selection.evidence.results = results
    if selection.evidence.run_id is None:
        from uuid import uuid4
        selection.evidence.run_id = uuid4().hex
    results._run_id = selection.evidence.run_id
    observe_selection(test)
    for corner in test._sweep_table:
        selection.evidence.corners[corner] = results
    # Single-run readback on the original authored bench retains main's
    # identity. Invocation-level ambiguity is finalized by the coordinator.
    test.__dict__.update(selection.evidence.values)


def retain_plan(test, reports, *, samples=False):
    """Retain each successful corner/sample under its public sweep label."""
    from silk.circuit.module_test import _sample_label
    test._sweep_states = {
        _sample_label(key) if samples else key: {
            name: value for name, value in vars(report.sim_results).items()
            if name not in {'_module', '_evidence_owner', 'backend'}}
        for key, report in reports.items()
        if (_sample_label(key) if samples else key) in test._sweep_table}
    selection = current_placement(test)
    if selection is None:
        return
    if reports:
        latest = next(reversed(reports.values()))
        selection.evidence.report = latest
        selection.evidence.results = latest.sim_results
        selection.evidence.values['_metric_values'] = dict(latest.metrics)
    selection.evidence.corners = {
        _sample_label(key) if samples else key: report.sim_results
        for key, report in reports.items()}
    selection.evidence.run_id = tuple(
        reader._run_id for reader in selection.evidence.corners.values())
    test.__dict__.update(selection.evidence.values)


# Weak on both sides: readback compatibility must not keep a hierarchy alive.
_latest = weakref.WeakKeyDictionary()


def latest_record(test):
    from silk.exceptions import UnboundSignalError
    reference, keys = _latest.get(test, (lambda: None, ()))
    index = reference()
    if index is None or not any(path for path, _ in keys):
        return False, None
    records = [(path, index.records.get(key)) for path, key in keys]
    runs = {record.run_id for _, record in records
            if record is not None and record.results is not None}
    if len(runs) > 1:
        choices = ', '.join(repr(path) for path, _ in records)
        raise UnboundSignalError(
            f'Test {test.test_name!r} has several simulation results; select '
            f'root.get_test({test.test_name!r}, instance=...) from {choices}.')
    return True, next((record for _, record in records
                       if record is not None and record.results is not None), None)


def observe_selection(test):
    selection = current_placement(test)
    if selection is None:
        _latest.pop(test, None)
        return
    root = selection.entry._root_ref()
    index = evidence_index(root)
    item = (selection.path, (selection.owner_path, id(test)))
    from silk.circuit.test_execution import current_invocation
    invocation = current_invocation()
    if invocation is None:
        _latest[test] = (weakref.ref(index), (item,))
    else:
        reference, keys = invocation.observed.setdefault(test, (weakref.ref(index), []))
        if item not in keys:
            keys.append(item)


def finish_invocation(invocation):
    for test, (reference, keys) in invocation.observed.items():
        _latest[test] = (reference, tuple(keys))


def begin_execution(test):
    """Replace selected live evidence only after replacement is established."""
    selection = current_placement(test)
    if isinstance(selection, CampaignSelection):
        return
    observe_selection(test)
    if selection is not None:
        record = selection.evidence
        record.report = record.results = record.run_id = None
        record.corners.clear()
        record.values.pop('_metric_values', None)
        record.values['_sweep_table'] = {}
        record.values['_sweep_generations'] = {}
        record.values['_sweep_states'] = {}
        record.values['_mc_samples'] = None


def placement_plot_traces(test, signals, corners=None):
    """Resolve one trace per retained run; aliases contribute labels only."""
    from silk.exceptions import UnboundSignalError
    if isinstance(test, SelectedTestbench):
        entry = test._entry()
        root = entry._root_ref()
        keys = [(entry.path, (entry.owner_path, id(test._authored)))]
        index = evidence_index(root)
    else:
        selection = current_placement(test)
        if selection is not None:
            root = selection.entry._root_ref()
            index = evidence_index(root)
            keys = [(selection.path, (selection.owner_path, id(test)))]
        else:
            reference, keys = _latest.get(test, (lambda: None, ()))
            index = reference()
            if index is None:
                return None
            root = index.root_ref()
    if root is None or not any(path for path, _ in keys):
        return None
    entries = {(entry.owner_path, id(authored_test(entry.test))): entry
               for entry in root.list_test_entries()}
    runs = {}
    requested = [corners] if isinstance(corners, str) else corners
    for path, key in keys:
        record = index.records.get(key)
        if record is None or record.results is None or key not in entries:
            continue
        table = record.values.get('_sweep_table', {})
        names = list(table) if requested is None else list(requested)
        if not names or any(name not in table for name in names):
            raise UnboundSignalError(f'Unknown corners {names!r}; retained {list(table)!r}.')
        for corner in names:
            reader = record.corners.get(corner)
            if reader is None:
                continue
            run_key = (getattr(reader, '_run_id', record.run_id), corner)
            if run_key not in runs:
                runs[run_key] = ([], entries[key], corner)
            runs[run_key][0].append(path)
    if not runs:
        raise UnboundSignalError('No successful placement results are retained; run the test first.')
    traces = [[] for _ in signals]
    for aliases, entry, corner in runs.values():
        label = f"{', '.join(aliases)} / {corner}"
        with execution_scope(entry):
            for output, signal in zip(traces, signals):
                output.append((label, signal.resolve(corner=corner)))
    return traces


def prune_evidence(root, entries):
    index = root.__dict__.get('_test_evidence')
    if index is None:
        return
    valid = {(entry.owner_path, id(authored_test(entry.test))) for entry in entries}
    for mapping in (index.selectors, index.records):
        for key in list(mapping):
            if key not in valid:
                del mapping[key]
    instances = list(root.iter_instances(include_self=True))
    owners = {instance.owner_path for instance in instances}
    declarations = {(instance.owner_path, id(spec)) for instance in instances
                    for spec in instance.module._specs}
    for key in list(index.projection_associations):
        if (key[0], key[1]) not in declarations:
            del index.projection_associations[key]
    for key in list(index.annotations):
        if key[0] not in owners:
            del index.annotations[key]
    for reference in list(index.scoped_specs.values()):
        spec = reference()
        if spec is None:
            continue
        spec.__dict__['_attempts'] = [
            (scope, a) for scope, a in spec.__dict__.get('_attempts', ())
            if scope is None or scope[0]() is not index or (scope[1], id(spec)) in declarations]
        slots = spec.measurements
        for key, measurement in list(slots.all_items()):
            scope = getattr(measurement, '_owner_scope', None)
            if scope is not None and scope[0]() is index and (scope[1], id(spec)) not in declarations:
                del slots[key]
