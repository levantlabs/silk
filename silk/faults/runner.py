"""Per-run worker for :class:`silk.faults.FaultCampaign`.

``run_one`` is a pure function — no shared state — so it is safe to
invoke from a ``multiprocessing.Pool``. Simulator execution is delegated to
``_run_ngspice``, which tests can patch to stay fast. Spec evaluation happens
in the parent, on each run's rawfile (see :mod:`silk.faults.evaluation`).
"""

from __future__ import annotations

import json
import os
import pickle
import time
import traceback

from silk._json_codec import encode_json
from silk.faults.campaign import RunResult
from silk.faults.types import FaultManifest


def run_one(
    module_pickle: bytes,
    fault_set,
    seed: int,
    run_id: int,
    run_dir: str,
    archive_waveforms: bool,
    reconcile_specs: bool,
) -> RunResult:
    """Execute one fault-injection run.

    Unpickles the Module snapshot, realizes ``fault_set`` at ``seed``,
    prepares and writes a faulted deck using the serialized circuit's output
    mode (``run_dir/netlist.cir`` by default, a persistent child in new mode),
    calls the simulator via :func:`_run_ngspice` and persists
    ``manifest.json``. It neither scores specs nor deletes the rawfile: the
    parent evaluates each run's rawfile (``reconcile_specs`` is accepted for
    compatibility and applied by the parent). Any exception is
    captured and returned as ``status='errored'`` — this function never
    raises because a crashed run must not poison the worker pool. Control
    files stay in ``run_dir``; ``archive_waveforms=False`` still discards the
    waveform from the prepared child in new mode.
    """
    t0 = time.time()
    os.makedirs(run_dir, exist_ok=True)

    manifest = FaultManifest(
        seed=seed,
        input_spec=fault_set.to_json(),
        applied=[],
    )

    try:
        module = pickle.loads(module_pickle)
        module.netlistname = 'netlist.cir'
        module.dir_folder = run_dir

        module.generate_netlist(
            corner='tt',
            fault_set=fault_set,
            manifest=manifest.applied,
            seed=seed,
        )

        files = module._run_files
        ng_result = _run_ngspice(
            files.netlist_path, files.directory, module.backend, run_files=files)

        rawfile_path = ng_result.get('rawfile_path')
        from silk.simulation.conditions import has_no_pdk_device
        no_pdk_device = bool(has_no_pdk_device(module))

        with open(os.path.join(run_dir, 'manifest.json'), 'w') as f:
            json.dump(encode_json(manifest.to_json()), f, allow_nan=False, indent=2)

        # The parent scores the rawfile (fresh evaluation) and removes it when
        # waveforms are not archived; the worker only simulates.
        return RunResult(
            run_id=run_id,
            seed=seed,
            status='simulated',
            manifest=manifest,
            rawfile_path=rawfile_path,
            wall_time_s=time.time() - t0,
            backend=module.simulator,
            requested_corner='tt',
            no_pdk_device=no_pdk_device,
        )
    except Exception as e:
        err = f'{type(e).__name__}: {e}\n{traceback.format_exc()}'
        return RunResult(
            run_id=run_id,
            seed=seed,
            status='errored',
            manifest=manifest,
            error=err,
            wall_time_s=time.time() - t0,
        )


def run_one_for_executor(
    run_dir: str,
    module_pickle: bytes,
    fault_set,
    seed: int,
    run_id: int,
    archive_waveforms: bool,
    reconcile_specs: bool,
):
    """Adapter for :func:`silk.simulation.executor.run_tasks`.

    A ``RunTask.fn`` is always called as ``fn(run_dir, *args, **kwargs)``
    with ``run_dir`` first; :func:`run_one` takes it as its fifth positional
    argument instead (kept as-is so existing keyword-argument callers and
    tests are undisturbed). This just reorders the call.
    """
    return run_one(module_pickle, fault_set, seed, run_id, run_dir,
                    archive_waveforms, reconcile_specs)


def _run_ngspice(netlist_path: str, run_dir: str, backend, *, run_files=None) -> dict:
    """Invoke the module's selected simulator via *backend*. Returns a dict
    with ``rawfile_path`` and ``status``. Raises on failure so
    :func:`run_one` can report ``status='errored'``."""
    kwargs = {} if run_files is None else {'run_files': run_files}
    raw = backend.run(
        netlist_path=netlist_path,
        results_dir=run_dir,
        debug=False,
        **kwargs,
    )
    return {'rawfile_path': raw.source_path, 'status': 'ok'}


def _outcome_to_json(outcome) -> dict:
    """JSON-friendly projection of a ``SpecOutcome``. Tolerant of missing
    fields so mock outcomes in tests still round-trip."""
    spec = getattr(outcome, 'spec', None)
    meas = getattr(outcome, 'measurement', None)
    return {
        'spec_name': getattr(spec, 'name', None),
        'spec_key': getattr(spec, 'key', None),
        'owner_path': getattr(outcome, 'owner_path', None),
        'report_key': getattr(outcome, 'report_key', None),
        'status': getattr(outcome, 'status', None),
        'measurement_value': getattr(meas, 'value', None),
    }
