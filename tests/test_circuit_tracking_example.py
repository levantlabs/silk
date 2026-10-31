"""``circuit_tracking_example`` runs end to end without a simulator or PDK models.

The example is the runnable walkthrough of the database tracking workflow
(seal, store, reopen, restore, promote, export). Its import and stage-listing
sections are checked here: the projection it writes must import into a second
database, and the stage listing must report the sealed snapshots the example
created. The run is a subprocess with ``SILK_EXAMPLES_OUT`` redirected into
``tmp_path`` and ``PDK_ROOT`` pointed at an empty directory, so a dependency on
installed sky130 models would fail here instead of passing on a configured
machine.
"""
import os
import subprocess
import sys

from silk.db import Database, DesignStage
from silk.examples import SILK_EXAMPLES_OUT_ENV

_REPO = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def _run_example(tmp_path):
    out = tmp_path / 'out'
    env = dict(
        os.environ,
        MPLBACKEND='Agg',
        PYTHONPATH=_REPO,
        PDK_ROOT=str(tmp_path / 'no-pdk'),
        **{SILK_EXAMPLES_OUT_ENV: str(out)},
    )
    cp = subprocess.run(
        [sys.executable, '-m', 'silk.examples.circuit_tracking_example'],
        cwd=str(tmp_path), env=env, capture_output=True, text=True,
        timeout=300,
    )
    assert cp.returncode == 0, cp.stderr[-2000:]
    return out, cp.stdout


def test_example_runs_end_to_end_and_writes_only_under_its_output_folder(
        tmp_path):
    out, stdout = _run_example(tmp_path)

    assert 'locked write rejected' in stdout
    assert (out / 'circuit_tracking.db').is_file()
    assert (out / 'circuit_tracking_projection.json').is_file()
    assert (out / 'circuit_tracking_imported.db').is_file()
    # Nothing else appears beside the output folder.
    assert sorted(p.name for p in tmp_path.iterdir()) == ['out']


def test_imported_projection_contains_the_sealed_circuit(tmp_path):
    out, stdout = _run_example(tmp_path)

    assert 'imported projection into' in stdout
    assert ('  tracked_amp 1.0 stage=dev: 3 devices, 5 nets, 8 results'
            in stdout)

    with Database(str(out / 'circuit_tracking_imported.db')) as imported:
        circuits = imported.list_circuits()
        assert [(c['circuit_name'], c['version'], c['stage'])
                for c in circuits] == [('tracked_amp', '1.0', 'dev')]
        summary = imported.get_circuit_summary('tracked_amp', '1.0')
        assert sorted(d['name'] for d in summary['devices']) == [
            'mn_in', 'mp_load', 'r_deg']
        assert len(summary['simulation_results']) == 8


def test_stage_listing_reports_the_latest_sealed_snapshot_per_stage(
        tmp_path):
    out, stdout = _run_example(tmp_path)

    assert ('latest sealed snapshot by design stage:\n'
            '  dev: tracked_amp 1.0\n'
            '  stable: tracked_amp 1.0.stable\n'
            '  post_layout: (none)\n'
            '  tapeout: (none)\n') in stdout

    with Database(str(out / 'circuit_tracking.db')) as database:
        assert [(r['circuit_name'], r['version'])
                for r in database.latest_at_stage(DesignStage.DEV)] == [
            ('tracked_amp', '1.0')]
        assert [(r['circuit_name'], r['version'])
                for r in database.latest_at_stage(DesignStage.STABLE)] == [
            ('tracked_amp', '1.0.stable')]
        assert database.latest_at_stage(DesignStage.TAPEOUT) == []
