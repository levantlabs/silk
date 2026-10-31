"""Top-level campaign orchestration for silk fault injection.

``FaultCampaign`` runs each realization through
``silk.simulation.executor.run_tasks`` (via
``silk.faults.runner.run_one_for_executor``, wrapping
``silk.faults.runner.run_one``), serially in-process by default or across a
``multiprocessing.Pool`` for ``n_workers > 1``, and aggregates the per-run
``RunResult`` objects into a ``CampaignResult`` with accessors for pass/fail
fractions, measurement distributions and fault correlation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import numpy as np

from silk.exceptions import NotSupportedError
from silk._json_codec import encode_json
from silk.simulation.executor import RunTask, resolve_n_workers, run_tasks


@dataclass
class RunResult:
    """Outcome of one fault-injection run.

    Attributes
    ----------
    run_id : int
        Index of the run within its campaign, ``0`` to ``n_runs - 1``.
    seed : int
        Seed of the run's random generator, derived from the campaign seed
        and ``run_id``.
    status : str
        ``'passed'`` (spec verification requested; at least one checked spec,
        every one freshly measured and passing, none unbound), ``'failed'``
        (at least one checked spec failed on this run's own output),
        ``'unverified'`` (verification requested but nothing failed and
        nothing was fully verified: no checked spec, or a spec unmeasured),
        ``'simulated'`` (``reconcile_specs=False``: executed, nothing scored)
        or ``'errored'`` (the run raised; see ``error``).
    manifest : silk.faults.FaultManifest
        The faults applied in this run. For an errored run it may be empty or
        partial.
    spec_outcomes : list
        One read-only row per spec, freshly extracted from this run's own
        output; empty when verification was off or the run errored. Each row
        has ``spec`` (``name`` and ``key``), ``owner_path``, ``status``
        (``'PASS'``, ``'FAIL'`` or ``'UNMEASURED'``), ``measurement`` (a
        detached :class:`silk.specs.Measurement`, or ``None`` when there is no
        usable value), ``attempt_kind``, ``reason`` and ``checked``.
    rawfile_path : str or None
        Path of the kept simulator rawfile, or ``None`` when waveforms were
        not archived or the run errored.
    error : str or None
        Exception type, message and traceback for an errored run, else
        ``None``.
    wall_time_s : float
        Wall-clock duration of the run in seconds.
    evaluation : str or None
        ``'fresh-v1'`` when the run was scored from its own output (that alone
        does not mean verified; ``status`` says so), ``'execution-only'`` for
        a ``reconcile_specs=False`` run, ``None`` for anything older.
    test_name : str or None
        The campaign testbench whose specs were checked, if any.
    backend : str or None
        The simulator that produced the run.
    requested_corner : str or None
        The corner label the run was requested at (not a proof; PDK-free
        decks prove corner none).
    no_pdk_device : bool or None
        Whether the emitted deck carried no PDK device.
    excluded : list of dict
        Specs that were not checked (another placement or another test) with
        the reason, so a pass is read for the checked scope only.

    Examples
    --------
    >>> from silk.faults import RunResult
    >>> from silk.faults.types import FaultManifest
    >>> run = RunResult(run_id=0, seed=1, status='passed',
    ...                 manifest=FaultManifest(seed=1, input_spec={}))
    >>> run.status, run.error
    ('passed', None)
    """
    run_id: int
    seed: int
    status: str
    manifest: Any
    spec_outcomes: List[Any] = field(default_factory=list)
    rawfile_path: Optional[str] = None
    error: Optional[str] = None
    wall_time_s: float = 0.0
    evaluation: Optional[str] = None
    test_name: Optional[str] = None
    backend: Optional[str] = None
    requested_corner: Optional[str] = None
    no_pdk_device: Optional[bool] = None
    excluded: List[Any] = field(default_factory=list)


def _derive_run_seed(campaign_seed: int, run_id: int) -> int:
    """Deterministic per-run seed from the campaign seed. Stable across
    ``n_runs`` — two campaigns with the same ``campaign_seed`` produce
    the same seed for a given ``run_id``."""
    return int(np.random.SeedSequence(
        entropy=(campaign_seed, run_id)).generate_state(1)[0])


@dataclass
class CampaignResult:
    """Aggregated results of a fault campaign.

    Returned by :meth:`silk.faults.FaultCampaign.run` and
    :meth:`silk.faults.FaultCampaign.resume`. Spec names are the ``name`` of
    the :class:`silk.specs.Spec` carried by each run's outcome.

    Attributes
    ----------
    campaign_id : str
        ``'campaign_<seed>'``.
    module_name : str
        The module's ``module_name``, or ``'top'`` when it has none.
    n_runs : int
        Number of runs the campaign was configured for.
    fault_set : silk.faults.FaultSet
        The fault set the runs were drawn from.
    seed : int
        Base seed of the campaign.
    runs : list of RunResult
        Per-run results ordered by ``run_id`` after :meth:`silk.faults.FaultCampaign.resume`.
    out_dir : str
        Directory holding ``summary.json`` and the per-run subfolders.

    Examples
    --------
    >>> from silk.faults import CampaignResult, FaultSet
    >>> result = CampaignResult('campaign_0', 'top', 0, FaultSet(), 0, [], 'out')
    >>> result.pass_fraction('gain')
    nan
    >>> result.measurement_distribution('gain').size
    0
    """
    campaign_id: str
    module_name: str
    n_runs: int
    fault_set: Any
    seed: int
    runs: List[RunResult]
    out_dir: str

    def pass_fraction(self, spec_name: str) -> float:
        """Return the fraction of usable runs in which a spec passed.

        Parameters
        ----------
        spec_name : str
            Name of the spec.

        Returns
        -------
        float
            Passing runs divided by the runs with a checked pass or fail for
            the spec (dimensionless, 0 to 1). Unmeasured, unchecked,
            simulated and errored runs are not counted. ``nan`` when no run
            has usable evidence.

        Raises
        ------
        ValueError
            The name matches specs with more than one identity.
        """
        return self._pass_fraction(self._identity(spec_name))

    def _pass_fraction(self, identity) -> float:
        rows = self._usable_rows(identity)
        if not rows:
            return float('nan')
        return sum(1 for o in rows if o.status == 'PASS') / len(rows)

    def measurement_distribution(self, spec_name: str) -> np.ndarray:
        """Return one spec's freshly measured values across the runs.

        Parameters
        ----------
        spec_name : str
            Name of the spec.

        Returns
        -------
        numpy.ndarray
            1-D float array, in the spec's measurement units, with one entry
            per run that has a fresh real measurement (infinities included).
            Extractor errors, NaN and unbound specs contribute nothing.

        Raises
        ------
        ValueError
            The name matches specs with more than one identity.
        """
        return self._measurement_distribution(self._identity(spec_name))

    def _measurement_distribution(self, identity) -> np.ndarray:
        vals: List[float] = []
        for r in self.runs:
            o = self._row_for(r, identity)
            if o is None:
                continue
            if not self._is_usable(o):
                continue
            v = _measured_value(o)
            if v is not None:
                vals.append(v)
        return np.array(vals, dtype=float)

    def runs_failing(self, spec_name: str) -> List[RunResult]:
        """Return the runs in which a spec failed.

        Parameters
        ----------
        spec_name : str
            Name of the spec.

        Returns
        -------
        list of RunResult
            Runs whose checked outcome for the spec has status ``FAIL``, in
            run order.

        Raises
        ------
        ValueError
            The name matches specs with more than one identity.
        """
        identity = self._identity(spec_name)
        return [r for r in self.runs
                if self._is_usable(self._row_for(r, identity))
                and self._row_for(r, identity).status == 'FAIL']

    def correlate_faults_with_failure(self, spec_name: str) -> dict:
        """Estimate how strongly each applied fault goes with a spec failing.

        For every ``(fault_type, target)`` seen in any run's manifest the
        value is ``P(fail | fault active) - P(fail | fault inactive)``,
        computed over the runs with usable evidence for the spec (a checked
        pass or fail). Errored, unmeasured and simulated runs are excluded.

        Parameters
        ----------
        spec_name : str
            Name of the spec.

        Returns
        -------
        dict
            Maps ``'<fault_type>@<target>'`` to the difference, a number from
            -1 to 1. A fault with no usable run on either side (active in
            every usable run, or in none) is omitted because the conditional
            probabilities are undefined.

        Raises
        ------
        ValueError
            The name matches specs with more than one identity.
        """
        return self._correlate(self._identity(spec_name))

    def _correlate(self, identity) -> dict:
        usable = [r for r in self.runs
                  if self._is_usable(self._row_for(r, identity))]
        all_keys: set = set()
        for r in usable:
            for af in r.manifest.applied:
                all_keys.add(f'{af.fault_type}@{af.target}')
        corr: dict = {}
        for key in all_keys:
            fault_type, target = key.split('@', 1)
            active = [r for r in usable
                      if any(af.fault_type == fault_type
                             and af.target == target
                             for af in r.manifest.applied)]
            inactive = [r for r in usable if r not in active]
            if not active or not inactive:
                continue
            p_active = (sum(1 for r in active if self._is_fail(r, identity))
                        / len(active))
            p_inactive = (sum(1 for r in inactive if self._is_fail(r, identity))
                          / len(inactive))
            corr[key] = p_active - p_inactive
        return corr

    def generate_report(self):
        """Return a report object bound to this result.

        Returns
        -------
        silk.faults.report.CampaignReport
            Renders the result as text, markdown, HTML or JSON.
        """
        from silk.faults.report import CampaignReport
        return CampaignReport(self)

    def _spec_identities(self) -> list:
        """Every ``(owner_path, spec_key, spec_name)`` seen, in first-seen order."""
        seen: dict = {}
        for r in self.runs:
            if r.status in ('errored', 'simulated'):
                continue
            for o in r.spec_outcomes:
                key = _identity_of(o)
                seen.setdefault(key, _name_of(o))
        return [(k[0], k[1], n) for k, n in seen.items()]

    def _identity(self, spec_name):
        """Resolve a display name to one identity; identities are private."""
        if not isinstance(spec_name, str):
            raise TypeError(
                f'spec must be a spec name (str), got {type(spec_name).__name__}')
        matches = [(o, k) for o, k, n in self._spec_identities() if n == spec_name]
        if len(matches) > 1:
            raise ValueError(
                f'Spec name {spec_name!r} is ambiguous in this campaign; it '
                f'names {len(matches)} specs: '
                + ', '.join(f'{o}/{k}' for o, k in matches))
        return matches[0] if matches else (None, spec_name)

    @staticmethod
    def _row_for(run: RunResult, identity):
        # One eligibility rule for every statistic: only runs that were
        # evaluated (not errored, not execution-only) contribute rows.
        if run.status in ('errored', 'simulated'):
            return None
        for o in run.spec_outcomes:
            if _identity_of(o) == identity:
                return o
        # Rows without an owner/key (hand-built outcomes) match by name.
        if identity[0] is None:
            for o in run.spec_outcomes:
                if _name_of(o) == identity[1]:
                    return o
        return None

    @staticmethod
    def _is_usable(o) -> bool:
        return (o is not None and getattr(o, 'checked', True)
                and str(getattr(o, 'status', '')).upper() in ('PASS', 'FAIL'))

    def _usable_rows(self, identity) -> list:
        return [o for o in (self._row_for(r, identity) for r in self.runs)
                if self._is_usable(o)]

    def _is_fail(self, run, identity) -> bool:
        o = self._row_for(run, identity)
        return self._is_usable(o) and str(o.status).upper() == 'FAIL'


def _identity_of(o):
    spec = getattr(o, 'spec', None)
    return (getattr(o, 'owner_path', None),
            getattr(spec, 'key', None) or getattr(spec, 'name', None))


def _name_of(o):
    return getattr(getattr(o, 'spec', None), 'name', None)


def _measured_value(o):
    meas = getattr(o, 'measurement', None)
    v = getattr(meas, 'value', None)
    return None if v is None else float(v)


def _detached_fault_set(fault_set):
    from silk.faults.set import FaultSet
    return FaultSet.from_json(fault_set.to_json())


def _remove_if_present(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _try_remove(path):
    """Remove *path*; return a diagnostic instead of raising on failure."""
    try:
        os.remove(path)
    except FileNotFoundError:
        return None
    except OSError as e:
        return f'could not remove {path!r}: {type(e).__name__}: {e}'
    return None


def _require_fault_backend(module):
    if module.simulator == 'behavioral':
        raise NotSupportedError(
            "FaultCampaign is not supported on the 'behavioral' backend: "
            "faults alter SPICE decks, which behavioral models do not use. "
            "Select a SPICE backend to run a fault campaign.")


class FaultCampaign:
    """Orchestrate a Monte Carlo fault campaign over a ``Module``.

    Each run realizes the fault set with its own seed, simulates the faulted
    netlist and (optionally) evaluates the module's specs. The per-run seed
    is derived from ``seed`` and the run index, so a given ``seed`` always
    reproduces the same draws and the seeds do not depend on ``n_runs``.

    Parameters
    ----------
    module : silk.circuit.Module
        The circuit. It needs at least one registered analysis.
    fault_set : silk.faults.FaultSet
        The fault definitions to draw from.
    n_runs : int
        Number of runs.
    seed : int, default 0
        Base RNG seed for reproducible fault draws.
    out_dir : str, default 'campaign_out/'
        Directory for per-run netlists and rawfiles; created on run.
    n_workers : int or str, optional
        Number of parallel worker processes, resolved by
        ``silk.simulation.executor.resolve_n_workers``. **Defaults to
        serial** (``None`` or ``1`` — in-process, no ``multiprocessing.Pool``,
        the easiest mode to debug). Pass an int ``N`` for a pool of ``N``
        workers (capped at ``os.cpu_count()``), or ``'all'`` for every CPU.
    archive_waveforms : bool, default False
        Keep each run's rawfile instead of deleting it once the parent has
        scored it. Because scoring happens after the batch executes, every
        run's rawfile of a batch is on disk at once until scoring finishes;
        ``False`` does not bound disk use to the runs currently executing.
    reconcile_specs : bool, default True
        Score the specs of the testbench whose ``netlist`` is ``module``
        (pass ``tb.netlist``) from each run's own faulted output, in the
        parent process. A pass means a pass for that bench's ``dut``
        placement only; other tests' and placements' specs are listed in
        ``RunResult.excluded``. Old recorded measurements are never read, and
        callables run on the authored bench with its silk-managed results
        untouched (side effects inside user callables are not rolled back).
        A module that is no bench's netlist has no bindable specs, so its
        runs are ``'unverified'``. ``False`` only executes: runs are
        ``'simulated'`` and nothing is verified.
    progress : bool, default True
        Stored on the instance; the current implementation does not use it.

    Raises
    ------
    NotSupportedError
        The behavioral backend cannot apply SPICE fault edits. Checked at
        construction and again before run/resume in case the backend changes.
    ValueError
        The module has no registered analysis (at construction), or
        ``n_workers`` is not ``None``, an int of at least 1, or ``'all'``
        (when the campaign runs).

    Examples
    --------
    >>> from tempfile import TemporaryDirectory
    >>> from silk import Module, load_pdk
    >>> folder = TemporaryDirectory()
    >>> pdk = load_pdk('sky130')
    >>> dut = Module('sky130', folder.name, 'demo', pdk)
    >>> from silk.faults import FaultSet, FaultCampaign
    >>> from silk import Transient
    >>> dut.add_sim_type(Transient(step=1e-9, stop=10e-9))
    >>> campaign = FaultCampaign(dut, FaultSet(), n_runs=2, seed=7, out_dir=folder.name, progress=False)
    >>> campaign.n_runs, campaign.seed
    (2, 7)
    """

    def __init__(
        self,
        module,
        fault_set,
        n_runs: int,
        seed: int = 0,
        out_dir: str = 'campaign_out/',
        n_workers: Union[None, int, str] = None,
        archive_waveforms: bool = False,
        reconcile_specs: bool = True,
        progress: bool = True,
    ):
        _require_fault_backend(module)
        if not getattr(module, 'sim_analysis_dict', None):
            raise ValueError(
                "FaultCampaign requires a module with at least one analysis "
                "registered — call module.add_sim_type(Transient(...)) (or "
                "AC/DC/Noise) before building the campaign. Without one, "
                "every fault run emits a netlist with no analysis command "
                "and the simulator aborts with no data rows."
            )
        self.module = module
        self.fault_set = fault_set
        self.n_runs = n_runs
        self.seed = seed
        self.out_dir = out_dir
        # Stored raw (None/int/'all') and resolved per-_execute() call via
        # resolve_n_workers(), since that also depends on module.backend and
        # a caller may swap module.simulator between run() and resume().
        self.n_workers = n_workers
        self.archive_waveforms = archive_waveforms
        self.reconcile_specs = reconcile_specs
        self.progress = progress

    def run(self) -> CampaignResult:
        """Run every fault realization and aggregate the results.

        Writes each run's files under ``out_dir/runs/run_NNNN`` and a
        ``summary.json`` in ``out_dir``. A run that fails does not abort the
        campaign: it is returned with status ``'errored'``.

        Returns
        -------
        CampaignResult
            Results for runs ``0`` to ``n_runs - 1``.

        Raises
        ------
        NotSupportedError
            The module's simulator is the behavioral backend.
        ValueError
            ``n_workers`` is invalid.
        """
        binding = self._bind()
        return self._finalize(
            self._execute(missing_ids=list(range(self.n_runs)), binding=binding))

    def resume(self, prior_result: CampaignResult) -> CampaignResult:
        """Re-run the runs a previous result lacks, that errored, or that this
        campaign could not have produced.

        A prior run is kept only if it did not error, was scored the way this
        campaign would score it now (``evaluation`` ``'fresh-v1'`` with
        ``reconcile_specs=True``, ``'execution-only'`` without), and belongs
        to this campaign (same ``campaign_id``, ``seed``, fault set and
        checked testbench). Anything else, including every result recorded
        before fresh evaluation existed, is run again with the same per-run
        seeds. Resume assumes the module, metrics and specs are unchanged
        since the prior result; it cannot check that.

        Parameters
        ----------
        prior_result : CampaignResult
            Result of an earlier :meth:`run` or :meth:`resume`.

        Returns
        -------
        CampaignResult
            A result whose ``runs`` merge the kept and new runs, sorted by
            ``run_id``. ``summary.json`` is written from this merged result.

        Raises
        ------
        NotSupportedError
            The module's simulator is the behavioral backend.
        ValueError
            ``n_workers`` is invalid.
        """
        binding = self._bind()
        kept = [r for r in prior_result.runs
                if self._compatible(r, prior_result, binding)]
        done = {r.run_id for r in kept}
        missing = [i for i in range(self.n_runs) if i not in done]
        new_runs = self._execute(missing_ids=missing, binding=binding)
        merged = kept + new_runs
        merged.sort(key=lambda r: r.run_id)
        return self._finalize(merged)

    def _bind(self):
        """The campaign's binding, resolved once per run/resume, or ``None``
        when only executing."""
        _require_fault_backend(self.module)
        self._last_binding = None
        if not self.reconcile_specs:
            return None
        from silk.faults.evaluation import preflight, resolve_binding
        binding = resolve_binding(self.module)
        if binding.bench is not None:
            from silk.circuit.test_placement import current_placement
            if current_placement(binding.bench) is not None:
                raise NotSupportedError(
                    'A fault campaign cannot start while a placement '
                    f'selection for test {binding.bench.test_name!r} is active.')
        preflight(binding)
        self._last_binding = binding
        return binding

    def _expected_evaluation(self) -> str:
        from silk.faults.evaluation import (
            EVALUATION_EXECUTION_ONLY, EVALUATION_FRESH)
        return EVALUATION_FRESH if self.reconcile_specs else EVALUATION_EXECUTION_ONLY

    def _compatible(self, prior, prior_result, binding) -> bool:
        """Whether *prior* may stand in for a fresh run of this campaign."""
        if prior.status == 'errored':
            return False
        if prior.evaluation != self._expected_evaluation():
            return False
        if (prior_result.campaign_id != f'campaign_{self.seed}'
                or prior_result.seed != self.seed
                or prior_result.fault_set.to_json() != self.fault_set.to_json()):
            return False
        expected_test = None if binding is None else binding.test_name
        return prior.test_name == expected_test

    def _finalize(self, runs) -> CampaignResult:
        result = CampaignResult(
            campaign_id=f'campaign_{self.seed}',
            module_name=getattr(self.module, 'module_name', None) or 'top',
            n_runs=self.n_runs,
            # A detached copy: resume compares against what the prior runs
            # were drawn from, not the caller's (mutable) current set.
            fault_set=_detached_fault_set(self.fault_set),
            seed=self.seed,
            runs=runs,
            out_dir=self.out_dir,
        )
        binding = getattr(self, '_last_binding', None)
        scope = None
        if binding is not None and binding.bench is not None:
            scope = {'test_name': binding.bench.test_name,
                     'owner_path': binding.dut_path}
        with open(os.path.join(self.out_dir, 'summary.json'), 'w') as f:
            json.dump(encode_json({
                'campaign_id': result.campaign_id,
                'seed': result.seed,
                'n_runs': result.n_runs,
                'checked_scope': scope,
                'runs': [{'run_id': r.run_id, 'status': r.status,
                          'seed': r.seed, 'evaluation': r.evaluation}
                         for r in result.runs],
            }), f, allow_nan=False, indent=2)
        return result

    def _execute(self, missing_ids: List[int], binding=None) -> List[RunResult]:
        from silk.faults.runner import run_one_for_executor
        from silk.faults.types import FaultManifest

        _require_fault_backend(self.module)
        os.makedirs(self.out_dir, exist_ok=True)
        runs_dir = os.path.join(self.out_dir, 'runs')
        os.makedirs(runs_dir, exist_ok=True)
        from silk.simulation._worker_pickle import dump_worker_netlist
        module_pickle = dump_worker_netlist(self.module)

        run_seeds = []
        tasks = []
        for run_id in missing_ids:
            run_seed = _derive_run_seed(self.seed, run_id)
            run_seeds.append(run_seed)
            run_dir = os.path.join(runs_dir, f'run_{run_id:04d}')
            # A replaced attempt must never leave its predecessor's verdict
            # beside the new result.
            _remove_if_present(os.path.join(run_dir, 'outcome.json'))
            tasks.append(RunTask(
                fn=run_one_for_executor,
                simulator=self.module.simulator,
                run_dir=run_dir,
                args=(module_pickle, self.fault_set, run_seed, run_id,
                      self.archive_waveforms, self.reconcile_specs),
            ))

        workers = resolve_n_workers(self.n_workers, backend=self.module.backend)
        outcomes = run_tasks(tasks, n_workers=workers)

        results = []
        for run_id, run_seed, outcome in zip(missing_ids, run_seeds, outcomes):
            if outcome.error is not None:
                # run_one() never raises (it catches and returns an
                # 'errored' RunResult itself); this only fires if the
                # executor infrastructure failed before run_one ran.
                results.append(RunResult(
                    run_id=run_id, seed=run_seed, status='errored',
                    manifest=FaultManifest(
                        seed=run_seed,
                        input_spec=self.fault_set.to_json(),
                        applied=[],
                    ),
                    error=outcome.error,
                ))
            else:
                results.append(self._score(
                    outcome.value, binding,
                    os.path.join(runs_dir, f'run_{run_id:04d}')))
        return results

    def _score(self, run: RunResult, binding, run_dir: str) -> RunResult:
        """Score one executed run in the parent, then dispose of its rawfile.

        Evaluation reads the run's own rawfile; the rawfile is removed (unless
        archiving) only after scoring, whether or not scoring succeeded.
        """
        from silk.faults.evaluation import (
            EVALUATION_EXECUTION_ONLY, EVALUATION_FRESH, evaluate_run,
            excluded_to_json, outcome_to_json, run_status)
        if run.status == 'errored':
            return run
        rawfile = run.rawfile_path
        outcome_path = os.path.join(run_dir, 'outcome.json')
        problems = []
        outcomes, excluded = [], []
        try:
            if binding is None:
                evaluation = EVALUATION_EXECUTION_ONLY
                status = 'simulated'
            else:
                outcomes, excluded = evaluate_run(
                    binding, run_id=run.run_id, seed=run.seed,
                    backend=run.backend, rawfile_path=rawfile,
                    label=run.requested_corner,
                    no_pdk_device=bool(run.no_pdk_device))
                status = run_status(outcomes)
                evaluation = EVALUATION_FRESH
        except Exception as e:
            import traceback
            problems.append(f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
        finally:
            # Disposal is contained per run: a failure here must not stop
            # later runs from being scored.
            if not self.archive_waveforms:
                try:
                    if rawfile and os.path.exists(rawfile):
                        os.remove(rawfile)
                    run.rawfile_path = None
                except OSError as e:
                    problems.append(
                        f'could not remove rawfile {rawfile!r}: '
                        f'{type(e).__name__}: {e}')
        if not problems:
            try:
                rows = [outcome_to_json(o) for o in outcomes]
                seen = {(r['owner_path'], r['spec_key']) for r in rows}
                rows += [excluded_to_json(x) for x in excluded
                         if (x['owner_path'], x['spec_key']) not in seen]
                tmp = outcome_path + '.tmp'
                with open(tmp, 'w') as f:
                    json.dump(encode_json(rows), f, allow_nan=False, indent=2)
                os.replace(tmp, outcome_path)
            except Exception as e:
                problems.append(f'could not write {outcome_path!r}: '
                                f'{type(e).__name__}: {e}')
                problems += [m for m in (_try_remove(outcome_path + '.tmp'),) if m]
        if problems:
            problems += [m for m in (_try_remove(outcome_path),) if m]
            run.status = 'errored'
            run.spec_outcomes = []
            run.excluded = []
            run.evaluation = None
            run.error = '\n'.join(problems)
            return run
        run.status = status
        run.evaluation = evaluation
        run.spec_outcomes = outcomes
        run.excluded = excluded
        if binding is not None:
            run.test_name = binding.test_name
        return run
