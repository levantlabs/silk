"""Testbench authoring/running methods for :class:`~silk.circuit.modules.Module`.

Split out of ``modules.py`` as a per-subsystem **part**. ``_TestbenchMethods`` is a pure mixin: no state, no separate identity. ``run_all_tests`` / ``get_all_testable_modules`` carry a documented lazy ``from silk.circuit.modules import Module`` for their submodule ``isinstance`` gate (see module-level note).
"""

from silk.circuit.emitted_names import emitted_submodule_name
import inspect
import weakref
from silk.circuit.test_placement import execution_scope, selected_test, authored_test, current_placement
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING
from silk.circuit.module_test import (
    ModuleTestbench, _RunSpec, _plan_tasks, _planned_outcome,
)
from silk.circuit.test_failure import TestFailure
from silk.exceptions import ConfigurationError, RunArgumentError

if TYPE_CHECKING:  # pragma: no cover
    from silk.circuit._module_protocol import _ModuleAttrs
    _Base = _ModuleAttrs
else:
    _Base = object


def _check_run_arguments(test, test_key, config):
    """Refuse a per-test *config* the test's own ``run`` cannot accept, before
    the call that would otherwise answer with a builtin ``TypeError``.

    An unexpected keyword is a caller-setup error in exactly the sense
    :class:`~silk.exceptions.ConfigurationError` marks, but Python raises it
    from the call itself and a builtin cannot inherit that marker, so it used
    to be folded into the results dictionary like a circuit failure. Binding the arguments to the callee's signature *first* moves
    the failure outside the swallow entirely, and the refusal raised here is a
    silk exception that opted into the marker at its own definition. Nothing
    is classified by exception type or message text at the call site: this
    inspects the CALLEE's signature, never the error a run produced.

    Runs before this test's placement is bound, so a refusal leaves the
    test exactly as it was found. A callee whose signature cannot be
    introspected is left to the call itself, exactly as before.
    """
    run = getattr(test, 'run', None)
    try:
        signature = inspect.signature(run)
    except (TypeError, ValueError):
        return
    try:
        signature.bind(**config)
    except TypeError as error:
        raise RunArgumentError(
            f"Test {test_key!r} cannot be run as asked: {error}. "
            f"run_all_tests forwards its keyword arguments to every test's "
            f"run(), so this affects the whole tree; nothing was run."
        ) from error


def _execution_nodes(module, *, recursive, ambient=None, supply=None):
    entries = [entry for entry, _, _ in _walk_test_entries(
        module, recursive=True, ambient=ambient, supply=supply)]
    from silk.circuit.test_placement import prune_evidence
    prune_evidence(module, entries)
    if not recursive:
        entries = [entry for entry in entries if entry.path == '']
    keys = _test_report_keys([(entry.report_key, entry.owner_path) for entry in entries])
    entries = [replace(entry, report_key=key) for entry, key in zip(entries, keys)]
    return tuple(replace(entry, test=selected_test(module, entry)) for entry in entries)


def _check_process_sources(nodes, *, standalone=None, corner=None):
    """One invocation promises one chip: reject conflicting process sources first."""
    from silk.exceptions import SimulationError, VariationError
    from silk.simulation._behavioral_variation import resolve_test
    from contextlib import nullcontext
    processes = {}
    entries = nodes if standalone is None else (None,)
    for entry in entries:
        scope = execution_scope(entry) if entry is not None else nullcontext(standalone)
        with scope as test:
            if (not isinstance(test, ModuleTestbench)
                    or test.netlist is None
                    or test._effective_simulator() != 'behavioral'):
                continue
            from silk.circuit.module_test import _planned_conditions
            planned = test._plan_runs(corner=corner, _deterministic=True) if (test._variable_lists or test._groups) else None
            for point in planned or (None,):
                try:
                    with _planned_conditions(test, point):
                        records = resolve_test(test).records
                except ConfigurationError:
                    raise
                except SimulationError:
                    # Preserve the normal walk's per-test simulation failure policy.
                    continue
                for record in records:
                    resolved = record.resolved
                    if resolved.variation != 'process':
                        continue
                    report_key = entry.report_key if entry is not None else test.test_name
                    place = (f'test {report_key!r}, point {getattr(point, "key", None)!r}, '
                             f'block {record.block!r}, param {record.param!r}')
                    previous, previous_place = processes.setdefault(resolved.name, (resolved, place))
                    if (previous.kind, previous.fields) != (resolved.kind, resolved.fields):
                        raise VariationError(
                            f'process {resolved.name!r} conflicts between {previous_place} and {place}')


def _announce_test_owner(entry, announced):
    """Preserve the recursive runner's banner once per tested child owner."""
    if entry.path and entry.owner_path not in announced:
        name = entry.owner_path.rsplit('/', 1)[-1].replace('~1', '/').replace('~0', '~')
        print(f"\n--- Testing submodule: {name} ---")
        announced.add(entry.owner_path)


def _run_test_entries(module, *, recursive, config, output_mode, ambient=None,
                      supply=None):
    """Run tests from one live module-instance tree.

    Entries retain both the legacy result key and the canonical owner path so
    repeated instances can be disambiguated after the complete tree is known.

    A test that RAISES is recorded as a :class:`~silk.circuit.test_failure.TestFailure`
    (a dict equal to ``{'error': str(error)}``) and the sweep
    continues: one circuit that will not converge must not abandon the others,
    and that aggregate channel is how multi-test runs have always reported
    trouble. The one exception is a
    :class:`~silk.exceptions.ConfigurationError`, which propagates and stops
    the run: it says the *request* was wrong rather than that a
    circuit misbehaved, so every test in the tree is about to fail for the
    same reason, and a dictionary of identical error strings would read like
    output. Classification is by that marker base class alone -- never by
    message text, and never by a list of types kept here.

    A builtin cannot inherit that marker, so the two caller-setup failures
    that used to arrive as builtins are moved off the builtin path at their
    own raise sites instead: arguments that do not fit this test's
    ``run`` are refused by ``_check_run_arguments`` *before* the call, and an
    unusable ``run_label`` or output directory is
    :class:`~silk.exceptions.OutputPathError` where ``_output_paths`` detects
    it. A plain ``ValueError`` or ``TypeError`` raised from inside a run is
    still a per-test result, exactly as any other run failure raises before metric extraction.

    *ambient* is the temperature setting in force from further up this walk,
    and *supply* is the primary-supply setting in force the same way. A module that set its own replaces it for itself and its
    descendants, so the nearest setting wins and a test that set its own beats
    every ancestor. Nothing here changes a child's stored setting:
    the propagated value lives in the placement binding for this invocation.
    """
    nodes = _execution_nodes(module, recursive=recursive, ambient=ambient, supply=supply)
    entries = []
    announced = set()
    for entry in nodes:
        test = entry.test
        key = entry.report_key
        _announce_test_owner(entry, announced)
        _check_run_arguments(test, key, config)
        print(f"\n{'=' * 60}\nRunning test: {key}\n{'=' * 60}")
        try:
            with execution_scope(entry, executing=True):
                value = test.run(**config)
            print(f"✓ Test {entry.test_name} completed successfully")
        except ConfigurationError:
            print(f"✗ Test {entry.test_name} cannot run as configured; stopping the run")
            raise
        except Exception as error:
            print(f"✗ Test {entry.test_name} failed: {error}")
            value = TestFailure.from_exception(error)
        entries.append((key, entry.owner_path, value))

    return entries


@dataclass(frozen=True)
class TestEntry:
    """A registered test at one hierarchy location, as run_all_tests sees it.

    Parameters
    ----------
    path : str
        Dotted instance path from the root module (``''`` for the root).
    spice_path : str
        The same path spelled with emitted SPICE instance names.
    module : Module
        The module that owns the test at this location.
    owner_path : str
        Spec owner path of that module (``'$'`` for the root).
    test_name : str
        Name the test was registered under.
    test : ModuleTestbench
        The selected testbench handle for this placement.
    report_key : str
        ``'<module_name>.<test_name>'``, the key in the run results.
    module_name : str, optional
        Subcircuit name of the owning module, if any.
    inherited_temperature : optional
        Ambient temperature setting inherited from the nearest owner.
    inherited_supply : optional
        Supply setting inherited from the nearest owner.
    output_mode : str, optional
        Root output mode, ``'overwrite'`` (default) or ``'new'``.
    _root_ref : weakref.ref, optional
        Internal weak reference to the root module; not part of equality or
        ``repr``.
    """

    #: Dotted instance path from the root module (``''`` for the root).
    path: str
    #: The same path spelled with emitted SPICE instance names.
    spice_path: str
    #: The module that owns the test at this location.
    module: object
    #: Spec owner path of that module (``'$'`` for the root).
    owner_path: str
    #: Name the test was registered under.
    test_name: str
    #: The selected testbench handle for this placement.
    test: object
    #: ``'<module_name>.<test_name>'``, the key in the run results.
    report_key: str
    #: Subcircuit name of the owning module, if any.
    module_name: str | None = None
    #: Ambient temperature setting inherited from the nearest owner.
    inherited_temperature: object = None
    #: Supply setting inherited from the nearest owner.
    inherited_supply: object = None
    #: Root output mode, ``'overwrite'`` or ``'new'``.
    output_mode: str = 'overwrite'
    _root_ref: object = field(default=None, repr=False, compare=False)
    inherited_axes: object = None


def _walk_test_entries(module, *, recursive, ambient=None, supply=None,
                       path='', spice_path='', owner_path='$', module_name=None,
                       ancestors=frozenset(), root=None, axes=None):
    from silk.circuit.module_placement import instance_children
    from silk.circuit.modules_structure import _child_spec_owner_path
    from silk.simulation.conditions import own_supply_setting, own_temperature_setting
    from silk.exceptions import CircuitError

    if root is None:
        root = module
    if id(module) in ancestors:
        raise CircuitError(f'Recursive module placement at {path!r}.')
    from silk.circuit.modules_sim import _axis_declaration
    axes = {axis: _axis_declaration(module, axis, (axes or {}).get(axis))
            for axis in ('temperature', 'supply')}
    own_temperature = own_temperature_setting(module)
    if own_temperature is not None:
        ambient = own_temperature
    own_supply = own_supply_setting(module)
    if own_supply is not None:
        supply = own_supply
    name = module_name if module_name is not None else module.module_name
    for test_name, test in module._tests.items():
        yield TestEntry(path, spice_path, module, owner_path, test_name, test,
                        f'{name}.{test_name}', name, ambient, supply, root._output_mode,
                        weakref.ref(root), axes), ambient, supply
    if not recursive:
        return
    for instance, child, name, _placed in instance_children(module):
        child_path = f'{path}.{instance}' if path else instance
        child_spice_path = f'{spice_path}.{emitted_submodule_name(instance)}' if spice_path else emitted_submodule_name(instance)
        yield from _walk_test_entries(
            child, recursive=True, ambient=ambient, supply=supply,
            path=child_path, spice_path=child_spice_path,
            owner_path=_child_spec_owner_path(owner_path, instance),
            module_name=name, ancestors=ancestors | {id(module)}, root=root, axes=axes)


def _iter_test_nodes(module, *, recursive, ambient=None, supply=None):
    """The shared runner traversal; listing projects the same entries."""
    for entry, ambient, supply in _walk_test_entries(
            module, recursive=recursive, ambient=ambient, supply=supply):
        yield entry.report_key, entry.owner_path, entry.test, ambient, supply


def _test_report_keys(identities):
    """Preserve unique legacy keys; qualify repeated keys by owner path."""
    from collections import Counter
    counts = Counter(key for key, _ in identities)
    return [key if counts[key] == 1 else f'{owner}:{key}'
            for key, owner in identities]


def _simulate_run_task(run_dir, spec):
    """Transport the shared attempt outcome, including typed errors.

    The module walk propagates ConfigurationError and records ordinary
    errors as text after the shared consumer applies the state transition.
    """
    from silk.circuit.module_test import _simulate_run_outcome
    return _simulate_run_outcome(spec)


def _run_test_entries_parallel(module, *, recursive, config, output_mode,
                               n_workers, ambient=None, supply=None):
    """Parallel counterpart of :func:`_run_test_entries`.

    Same traversal order (:func:`_iter_test_nodes`), ambient/supply
    inheritance, ``_check_run_arguments`` refusal and
    ``ConfigurationError``-vs-ordinary-error classification as the serial
    walker, so the two produce identical entries for a tree that runs
    cleanly. What differs: an eligible test — a plain ``ModuleTestbench``
    (``run`` not overridden by a subclass) not resolved to an in-process
    backend — is split into ``_prepare_run`` (here, in this process, in
    traversal order: every test's
    inherited conditions resolved in the parent first) and ``_simulate_run``
    (dispatched, once every entry has been prepared, to the shared
    :mod:`silk.simulation.executor` pool) instead of one in-process
    ``test.run(**config)`` call; ``_evaluate_run`` runs back here, one at a
    time, in submission order, once every dispatched simulation has
    returned. This is why the parallel path never calls
    ``ModuleTestbench.run`` itself for a split test — forwarding the
    corner/temperature/voltage/run_label prepare -> simulate -> evaluate
    sequence *is* what ``run()`` does, just with the middle third moved to a
    worker.

    A test whose subclass overrides ``run`` (the examples' ``PVTTestbench``/
    ``_TxTest``) is not eligible for the split — its own override may not
    even simulate anything through this bench's netlist, so it still runs
    with one in-process ``test.run(**config)`` call, exactly as serial. A
    split-eligible test whose netlist resolved to the ``behavioral`` backend
    prepares the same way but always simulates and evaluates in-process too
    (never dispatched to the pool): behavioral models may be closures, so it
    always behaves as ``n_workers=1`` regardless of what this call requested.

    Because every test is prepared before any of them simulate, a
    ``ConfigurationError`` from a later test's ``_prepare_run`` (or from
    ``_check_run_arguments``) stops the walk, exactly as serial, before its
    executor batch is even built — but unlike serial, an earlier test that
    would already have fully run by that point has not simulated yet here.
    Nothing observes that ordering difference: both paths raise instead of
    returning a dict, and no test's report or database record from that call
    is ever produced either way.

    A split-eligible test with a declared plan (``set_corners``)
    is expanded here, not in ``run()``: :meth:`ModuleTestbench._plan_runs`
    gives its runs, every one is prepared in this process (inside this
    test's placement binding) and dispatched to the same pool as every
    other test's run, and once the pool returns its outcomes are evaluated
    in plan order and collected by ``_collect_runs`` into the test's
    :class:`~silk.circuit.spec_report.CornerResults` — the same object the
    serial walk's ``test.run(**config)`` returns. A Monte Carlo plan
    (``set_monte_carlo``) runs in place instead, through
    ``_run_plan(n_workers=...)``: its samples are prepared, simulated on the
    same executor and evaluated in batches of ``n_workers``, so the walk never
    holds one deep-copied netlist per sample. It returns the same
    :class:`~silk.circuit.monte_carlo.MonteCarloResults` as the serial walk.
    """
    from silk.simulation.executor import RunTask, run_tasks

    nodes = _execution_nodes(module, recursive=recursive, ambient=ambient, supply=supply)
    entries = []
    prepared_groups = {}
    pending = []  # (position, legacy_key, owner_path, test, spec)
    plans = []    # (position, legacy_key, owner_path, test, planned, prepared, errors)

    announced = set()
    for position, entry in enumerate(nodes):
        legacy_key, owner_path, test = entry.report_key, entry.owner_path, authored_test(entry.test)
        entries.append(None)
        _announce_test_owner(entry, announced)
        _check_run_arguments(test, legacy_key, config)
        print(f"\n{'=' * 60}")
        print(f"Running test: {legacy_key}")
        print(f"{'=' * 60}")

        splittable = (isinstance(test, ModuleTestbench)
                     and type(test).run is ModuleTestbench.run)
        try:
            with execution_scope(entry, executing=True):
                if not splittable:
                    value = test.run(**config)
                    entries[position] = (legacy_key, owner_path, value)
                    continue
                planned = test._plan_runs(**config)
                if planned is not None:
                    run_config = {k: v for k, v in config.items() if k != 'corner'}
                    if any(run.sample is not None for run in planned):
                        # Monte Carlo samples run in batches of n_workers through
                        # the same executor, prepared one batch at a time rather
                        # than all up front (one deep-copied netlist per sample).
                        entries[position] = (legacy_key, owner_path,
                                             test._run_plan(planned,
                                                            n_workers=n_workers,
                                                            **run_config))
                        continue
                    prepared, errors = test._prepare_plan(planned, **run_config)
                    if any(spec.netlist.simulator == 'behavioral'
                           for spec in prepared.values()
                           if isinstance(spec, _RunSpec)):
                        # In-process backends never go to the pool (see above).
                        entries[position] = (legacy_key, owner_path,
                                             test._run_plan(planned, **run_config))
                        continue
                    plans.append((position, legacy_key, owner_path, test,
                                  planned, prepared, errors, entry))
                    continue
                prepared = test._prepare_run(**config)
                if not isinstance(prepared, _RunSpec):
                    # A cache hit: _prepare_run() already returned the SpecReport
                    # (dead in this release — TASK-284 forces the cache off —
                    # kept so a future working cache needs no change here).
                    entries[position] = (legacy_key, owner_path, prepared)
                    continue
                if prepared.netlist.simulator == 'behavioral':
                    from silk.circuit.module_test import _simulate_run_outcome
                    entries[position] = (legacy_key, owner_path,
                                         test._consume_run_outcome(
                                             _simulate_run_outcome(prepared, test._simulate_run)))
                    continue
                from silk.circuit.test_execution import current_invocation
                prepared = current_invocation().prepare(prepared)
                from silk.simulation._worker_pickle import copy_simulation_netlist
                from silk.circuit.test_execution import equivalence
                key = equivalence(prepared)
                snapshot = prepared_groups.get(key) if key is not None else None
                if snapshot is None:
                    snapshot = copy_simulation_netlist(prepared.netlist)
                    if key is not None:
                        prepared_groups[key] = snapshot
                prepared = replace(prepared, netlist=snapshot)
                pending.append((position, legacy_key, owner_path, test, prepared, entry))
        except ConfigurationError:
            raise
        except Exception as error:
            print(f"✗ Test {legacy_key} failed: {error}")
            entries[position] = (legacy_key, owner_path, TestFailure.from_exception(error))

    if pending or plans:
        from silk.circuit.test_execution import equivalence, RunEvidence
        groups, tasks, consumers = {}, [], []
        for _, _, _, _, spec, _ in pending:
            key = equivalence(spec)
            index = groups.get(key) if key is not None else None
            if index is None:
                index = len(tasks)
                tasks.append(RunTask(fn=_simulate_run_task, run_dir=spec.tb_dir,
                                     args=(spec,), simulator=spec.netlist.simulator))
                if key is not None:
                    groups[key] = index
            consumers.append(index)
        plan_consumers = []
        for _, _, _, _, planned, prepared, _, entry in plans:
            selected = {}
            for run in planned:
                spec = prepared.get(run.key)
                if not isinstance(spec, _RunSpec):
                    continue
                from silk.circuit.test_execution import current_invocation
                with execution_scope(entry, executing=True):
                    spec = current_invocation().prepare(spec)
                prepared[run.key] = spec
                key = equivalence(spec)
                index = groups.get(key) if key is not None else None
                if index is None:
                    index = len(tasks)
                    from silk.circuit.module_test import _planned_simulate_task
                    tasks.append(RunTask(fn=_planned_simulate_task,
                                         run_dir=spec.tb_dir, args=(spec,),
                                         simulator=spec.netlist.simulator))
                    if key is not None:
                        groups[key] = index
                selected[run.key] = index
            plan_consumers.append(selected)
        outcomes = list(run_tasks(tasks, n_workers=n_workers))
        frozen = {index: RunEvidence.capture(outcome.value.result)
                  for index, outcome in enumerate(outcomes)
                  if outcome.error is None and outcome.value.error is None}
        # Each consumer gets its own invocation identity and evaluation; only
        # immutable launch facts are retained across those evaluations.
        def consumer_outcome(index, spec):
            outcome = outcomes[index]
            if index in frozen:
                return replace(outcome, value=replace(
                    outcome.value, result=frozen[index].result(spec)))
            return outcome

        work = []
        for item, index in zip(pending, consumers):
            work.append((item[0], item, consumer_outcome(index, item[4])))
        for item, selected in zip(plans, plan_consumers):
            work.append((item[0], item, {
                key: _planned_outcome(consumer_outcome(index, item[5][key]), item[5][key])
                for key, index in selected.items()}))
        work.sort(key=lambda w: w[0])
        # Every task has executed. Reconcile its replacement facts even if
        # an earlier consumer's configuration error prevents evaluation.
        from silk.circuit.module_test import _reconcile_run_outcome
        for _, item, outcome in work:
            with execution_scope(item[-1], executing=True):
                if isinstance(outcome, dict):
                    for value in outcome.values():
                        _reconcile_run_outcome(item[3], value)
                else:
                    _reconcile_run_outcome(item[3], _planned_outcome(outcome, item[4]))
        for _, item, outcome in work:
            if isinstance(outcome, dict):
                (position, legacy_key, owner_path, test, planned, prepared,
                 errors, entry) = item
                try:
                    with execution_scope(entry, executing=True):
                        reports = test._evaluate_plan(planned, prepared, outcome,
                                                      errors)
                        value = test._collect_runs(planned, reports, errors)
                        from silk.circuit.test_placement import current_placement
                        current_placement(test).evidence.report = value
                except ConfigurationError:
                    raise
                except Exception as error:
                    print(f"✗ Test {legacy_key} failed: {error}")
                    value = TestFailure.from_exception(error)
                entries[position] = (legacy_key, owner_path, value)
                continue
            position, legacy_key, owner_path, test, _spec, entry = item
            try:
                with execution_scope(entry, executing=True):
                    report = test._consume_run_outcome(_planned_outcome(outcome, _spec))
            except ConfigurationError:
                raise
            except Exception as error:
                report = TestFailure.from_exception(error)
            entries[position] = (legacy_key, owner_path, report)

    return entries


def _format_test_results(entries):
    """Preserve legacy keys unless multiple live instances would overwrite."""
    keys = _test_report_keys([(key, owner) for key, owner, _ in entries])
    return {key: value for key, (_, _, value) in zip(keys, entries)}


class _TestbenchMethods(_Base):
    """Testbench authoring/running methods mixed into ``Module``. Holds no state."""

    def create_test(self, name='default_test', parameters=None, *, strict_nets=None,
                    save_device_params=None):
        """Create and register a uniquely named test for this module.

        The test's executing ``netlist`` Module inherits this live module's
        simulator at run time unless ``test.netlist.simulator`` is explicitly
        assigned. Unset numerical options and temperature/supply settings
        inherit from the live owner; explicit bench settings take precedence.

        Parameters
        ----------
        name : str
            Name for the test
        parameters : dict, optional
            Optional declared test-parameter map — part of this
            test's characterization identity (defaults materialized by
            the caller; never read by netlisting). See
            ``ModuleTestbench.__init__``.
        strict_nets : bool, optional
            Keyword-only override for the testbench netlist's
            ``strict_nets`` (default ``None`` inherits it from this
            module — see ``ModuleTestbench.__init__``).
        save_device_params : bool, optional
            Keyword-only. ``None`` (default) inherits this
            module's ``save_device_params``; ``True``/``False`` enables or
            disables operating-point device-parameter capture for this test
            alone. Also ``tb.save_device_params = ...``.

        Returns
        -------
        ModuleTestbench
            ModuleTestbench instance that can be configured

        Raises
        ------
        DuplicateNameError
            If this module already owns a test with ``name``.

        Examples
        --------
        ::

            test = module.create_test('transient')
            test.add_module_under_test({'vdd': 'vdd', 'gnd': '0'})
            test.add_source('voltage', 'supply', nets={'positive': 'vdd', 'negative': '0'},
                          parameters={'dc': 3.3})
            test.add_metric('avg_power', lambda nr: calculate_power(nr))
        """
        from silk.circuit.module_test import ModuleTestbench
        return ModuleTestbench(self, name, parameters=parameters, strict_nets=strict_nets,
                            save_device_params=save_device_params)

    def run_test(self, name='default_test', *, store=False,
                 store_options=None, **config):
        """Run a specific test, optionally persisting its report.

        The testbench executes with this live owner's current simulator unless
        its ``netlist.simulator`` was explicitly assigned, with either storage
        setting. Ambient TEMP and the primary supply follow the same rule:
        the test uses this owner's current ``set_temperature`` /
        ``set_supply_voltage`` setting unless the test set its own, and the
        value is re-read at every run. See ``ModuleTestbench.run``.

        Parameters
        ----------
        name : str
            Name of test to run
        store : bool, default False
            Keyword-only, default ``False``. When ``True``, the
            completed :class:`~silk.circuit.spec_report.SpecReport` is
            persisted through the existing ``store_report`` contract
            (metrics, compatibility pass/fail rows and first-class spec
            measurements, one transaction, append-on-repeat) against the
            circuit version this module already has bound. Nothing is
            created or changed implicitly: ``Module(db_path=...)`` and
            ``create_circuit_version()`` are the caller's job, and every
            readiness failure that can be known before the run (no
            ``db_path``, no bound version, a locked or unsealed version,
            a live design that has drifted from the bound snapshot) is
            raised **before** the simulator starts. The default leaves
            execution byte-for-byte as before: no tracker is touched.
        store_options : dict, optional
            Keyword-only dict of ``store_report`` keyword
            arguments to forward -- ``tool_name``, ``units``,
            ``sim_type``, ``source``, ``conditions``. ``corner``,
            ``temperature`` and ``voltage`` are rejected here: the run's
            own immutable operating conditions are authoritative,
            including unknown temperature or supply.
        **config
            Forwarded to ``ModuleTestbench.run`` (``corner='ff'``,
            ``run_label='baseline'``, and ``temperature=``/``voltage=``
            as deprecated TEMP / supply assertions). Use
            test.set_temperature() and test.set_supply_voltage() to
            configure them. Verified fixed temperature plus an established
            supply gives a scoped report; otherwise storage is explicitly
            unscoped. The label only organizes output files.

        Returns
        -------
        SpecReport
            :class:`~silk.circuit.spec_report.SpecReport`. With
            ``store=True`` the storage receipt is on ``report.storage``:
            ``store_report``'s ``metrics_stored`` / ``specs_stored`` /
            ``skipped`` plus ``cached`` -- always ``False`` in this release:
            the report cache is disabled, so ``run()``
            always simulates. The field is kept for forward compatibility
            with a future working cache.

        Raises
        ------
        ValueError
            Unknown ``name``, or ``store_options`` carrying
            ``corner``/``temperature``/``voltage``.
        RuntimeError
            ``store=True`` without a ``db_path`` or without a
            bound circuit version (before the run).
        LockedVersionError, DatabaseError, EvidenceMismatchError
            ``store=True`` readiness failures (before the run).
        SimulationAborted
            The run produced no data; nothing is stored.
        ReportStorageError
            The run completed but persisting the report
            failed -- ``.report`` holds the completed report, the
            original error is chained. ``store_report``'s transaction has
            rolled back; the live Spec measurements the run recorded
            stand.
        """
        if name not in self._tests:
            raise ValueError(f"No test '{name}' for module {self.module_name}. Available: {list(self._tests.keys())}")

        test = self._tests[name]
        if store and (getattr(test, 'corners', None) is not None
                      or getattr(test, 'monte_carlo', None) is not None
                      or test._has_condition_grid()):
            from silk.exceptions import MonteCarloError
            raise MonteCarloError(
                f"run_test({name!r}, store=True): this test declares "
                f"{'Monte Carlo samples' if test.monte_carlo is not None else 'Variable lists' if test._variable_lists else 'a temperature/supply list' if test._has_condition_grid() else 'a corner list'}"
                f", and store=True stores one SpecReport; storing declared "
                f"runs is not supported yet. Run without store=True.")
        if not store:
            if store_options:
                raise ValueError(
                    "run_test(): store_options given without store=True"
                )
            return test.run(**config)

        # --- store=True: fail before the simulator for anything knowable now.
        options = dict(store_options or {})
        reserved = sorted(
            key for key in ('corner', 'temperature', 'voltage')
            if key in options
        )
        if reserved:
            raise ValueError(
                f"run_test(store=True): store_options may not set "
                f"{reserved}; the run's own corner and characterization "
                f"context (pass corner=/temperature=/voltage= to run_test "
                f"itself) are the only source of that provenance"
            )
        if self._tracker is None and self._db_path is None:
            raise RuntimeError(
                f"run_test(store=True) on module {self.module_name!r}: no "
                f"database is bound. Build the module with "
                f"Module(..., db_path='...') and call "
                f"create_circuit_version() before storing a run."
            )
        tracker = self._ensure_tracker()
        tracker.require_report_storage_ready()

        cached_before = list(getattr(test, '_cache', {}).values())
        report = test.run(**config)
        cached = any(report is item for item in cached_before)

        from silk.exceptions import ReportStorageError
        try:
            receipt = tracker.store_report(report, **options)
        except Exception as error:
            raise ReportStorageError(
                f"run_test({name!r}, store=True): the test ran to completion "
                f"(corner={report.corner!r}) but storing its report failed: "
                f"{error}. The report is on this error's .report; nothing "
                f"from it was written, and the live Spec measurements the "
                f"run recorded are unchanged.",
                report=report,
            ) from error
        report.storage = dict(receipt, cached=cached)
        return report

    def run_all_tests(self, recursive=True, *, n_workers=None, **config):
        """Run all tests for this module and optionally submodules.

        Each test inherits its own live owner's simulator unless its
        ``netlist.simulator`` was explicitly assigned. Descendant owners keep
        their selections, so one recursive invocation can use mixed backends.

        ``n_workers``: ``None`` or ``1`` (the default) runs
        exactly as before — one in-process ``test.run(**config)`` call per
        test, in the same traversal order. Any other value, resolved through
        ``silk.simulation.executor.resolve_n_workers`` (an int capped at
        ``os.cpu_count()``, or ``'all'`` for every CPU), switches to a
        parallel walk: every test's inherited conditions are resolved in
        this process first (the prepare step), simulation is dispatched to a
        shared worker pool, and each result is evaluated back in this
        process, one at a time, in the same order the serial walk would have
        used — so a tree that runs cleanly gets identical per-test reports,
        this call's run summary and any database records either way. A test
        whose ``run`` is overridden by a subclass, or whose netlist resolved
        to the ``behavioral`` backend, always runs in-process
        regardless of ``n_workers`` (behavioral models may be closures).
        Each additional worker is a full extra OS process with its own
        parsed PDK, not a shared one, so size ``n_workers`` for memory, not
        only CPU count. ``ConfigurationError`` still stops the whole call and
        an ordinary run failure still becomes that test's ``{'error': ...}``
        entry either way — see Raises below; the parallel walk just resolves
        every test's *prepare* step before any of them *simulate*, so on a
        ``ConfigurationError`` an earlier test that would already have fully
        run serially may not have simulated yet here (nothing observes the
        difference: both paths raise instead of returning a dictionary).

        Ambient TEMP and the primary supply inherit down the tree instead: a
        temperature or supply set on this root applies to every test in the
        walk, including tests owned by submodules, which is what makes one
        ``set_temperature`` / ``set_supply_voltage`` call a whole-tree
        setting. The nearest setting wins — a test that set its own keeps it,
        and below a submodule that set its own (any submodule owning tests of
        its own may) that one applies to its subtree. The
        propagated value is in force for this invocation only and changes no
        child's stored setting. An inherited supply names a source that is
        re-resolved against each test's own netlist, so one root setting
        reaches every testbench that has a rail of its own.

        This invoking root's output mode applies to ordinary silk testbenches
        throughout this invocation, including legacy registered submodules.
        Child settings are retained for later independent runs. Cache hits
        retain their existing evidence without allocating or cleaning files.

        Parameters
        ----------
        recursive : bool, default True
            If True, also run tests on submodules
        n_workers : int or str, optional
            Keyword-only. See above.
        **config
            Additional configuration passed to tests, including
            ``run_label`` for output organization. Full live instance
            paths isolate each test automatically; labels do not change
            the report keys.

        Returns
        -------
        dict
            Dictionary mapping test names to results. Unique test names retain
            the ``module_name.test_name`` form. Repeated live instances use
            ``owner_path:module_name.test_name`` to prevent overwrites.

            A test that fails is recorded as a ``TestFailure`` (a dict equal
            to ``{'error': str(error)}`` that also carries ``error_type``,
            ``exception``, ``details`` and ``run``) under
            its own key and the sweep continues, so one circuit that will not
            converge never abandons the rest. That covers
            :class:`~silk.exceptions.SimulationAborted`, convergence failures,
            a backend exiting nonzero, and a malformed or missing rawfile.

        Raises
        ------
        RunArgumentError
            ``config`` does not fit a test's own ``run``
            signature -- a mis-typed or unsupported keyword such as
            ``store=True``. Checked against each test's signature before
            it is called, so nothing is run and no dictionary is
            returned. A ``TypeError`` subclass.
        OutputPathError
            ``run_label`` is not a usable label, or the
            output directory it selects would leave the DUT output root.
            A ``ValueError`` subclass, raised before any directory is
            created.
        ConfigurationError
            Any error marked
            :class:`~silk.exceptions.ConfigurationError` propagates out of
            this call instead of being folded into the results, and no
            dictionary is returned. These say the *request* is
            wrong rather than that a circuit misbehaved -- a
            ``run(temperature=)`` assertion that does not match configured
            TEMP (use ``test.set_temperature(...)``), a ``run(voltage=)``
            assertion the deck contradicts or a supply that cannot be
            established (use ``test.set_supply_voltage(...)``), a prepared
            deck that is missing, consumed or stale, a duplicate or
            unreadable temperature declaration in custom SPICE, a cached
            report asked
            for conditions it was not run under, and a backend asked for a
            capability it does not have -- so every test in the tree would
            fail for the same reason.
        """
        from silk.simulation.executor import resolve_n_workers
        resolved_workers = resolve_n_workers(n_workers)
        from silk.circuit.test_execution import invocation
        with invocation():
            _check_process_sources(_execution_nodes(self, recursive=recursive))
            if resolved_workers <= 1:
                entries = _run_test_entries(self, recursive=recursive, config=config,
                                            output_mode=self._output_mode)
            else:
                entries = _run_test_entries_parallel(
                    self, recursive=recursive, config=config,
                    output_mode=self._output_mode, n_workers=resolved_workers)
        return _format_test_results(entries)

    def list_test_entries(self, *, recursive=True):
        """Return immutable TestEntry records in run_all_tests traversal order.

        Each record exposes path, spice_path, module, owner_path, test_name,
        test, report_key, module_name, inherited_temperature, inherited_supply
        and output_mode. The test handle selects this owner for execution and
        readback. Listing never runs or prepares a test. The live module
        and bench handles may be configured before running the listed tests.

        Parameters
        ----------
        recursive : bool, optional
            Include tests of placed submodules (default True); False lists
            only this module's own tests.

        Returns
        -------
        list of TestEntry
            One record per registered test.
        """
        return _execution_nodes(self, recursive=recursive)

    def get_test(self, name, *, instance=None):
        """Return the registered testbench named ``name`` on this module.

        With ``instance=None`` or ``''``, return the original local bench.
        A non-empty instance path (for example ``'u1.u2'``) returns a
        stable selected handle with that placement's evidence and inherited
        execution conditions. Lookup does not run a simulation.

        Parameters
        ----------
        name : str
            Registered test name.
        instance : str, optional
            Dotted instance path of a placed submodule, such as ``'u1.u2'``.

        Returns
        -------
        ModuleTestbench
            The selected testbench handle.

        Raises
        ------
        silk.exceptions.ConfigurationError
            Unknown or ambiguous test/instance, with the available
            selections included in the message.
        """
        if instance not in (None, ''):
            entries = self.list_test_entries()
            matches = [e for e in entries if e.path == instance and e.test_name == name]
            if len(matches) == 1:
                return matches[0].test
            available = [(e.path, e.test_name) for e in entries]
            raise ConfigurationError(
                f"No unambiguous test {name!r} at instance {instance!r}. "
                f"Available (instance, test) selections: {available}")
        if name not in self._tests:
            available = (f"Registered tests: {list(self._tests)}"
                         if self._tests else "Registered tests: none.")
            owner = self.module_name or self.dir_folder + '/' + self.netlistname
            raise ConfigurationError(
                f"No test {name!r} for module {owner!r}. {available}")
        return self._tests[name]

    def list_tests(self):
        """List the names of the tests registered on this module.

        Returns
        -------
        list of str
            Test names in registration order; submodule tests are not
            included (see :meth:`list_test_entries`).
        """
        return list(self._tests.keys())

    def has_tests(self):
        """Report whether this module has registered tests of its own.

        Returns
        -------
        bool
            True when at least one test is registered on this module.
        """
        return len(self._tests) > 0

    def get_all_testable_modules(self, include_self=True):
        """List all modules (including submodules) that have tests.

        Parameters
        ----------
        include_self : bool, optional
            Whether to include this module in the list. Default True.

        Returns
        -------
        list of dict
            One dictionary per module with the keys ``'module'`` (its
            ``module_name``) and ``'tests'`` (its test names).
        """
        testable = []

        if include_self and self.has_tests():
            testable.append({
                'module': self.module_name,
                'tests': self.list_tests()
            })

        # Check submodules
        from silk.circuit.modules import Module  # lazy — see run_all_tests
        for module_name, module_info in self.module_dict.items():
            submodule = module_info['device']
            if isinstance(submodule, Module):
                testable.extend(submodule.get_all_testable_modules(include_self=True))

        return testable
