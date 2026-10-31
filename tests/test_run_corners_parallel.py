"""run_corners(n_workers=): parallel corners; each corner in its own folder
(TASK-301.3).

Every corner ALWAYS gets its own ``<testbench dir>/<corner>/`` folder now,
serial or parallel (earlier releases shared an unlabeled output folder) --
proved on a PDK-free RC
deck, which needs no PDK install. Serial (``n_workers=1``) and parallel
(``n_workers>1``) runs of the same benches must give identical
CornerResults, Spec measurements and deck/rawfile content: proved here on
the PDK-free deck and, in the slow profile, on a real sky130 NMOS DUT (the
shape ``_build_nmos_tb`` in ``tests/test_run_split.py`` already proves
picklable/deep-copyable, avoiding the ideal-element pickle-registry
collision documented there).
"""
import copy
import os
import time
from dataclasses import replace

import pytest

from silk import load_pdk, sources
from silk.circuit.modules import Module
from silk.circuit.module_test import ModuleTestbench, _RunSpec
from silk.circuit.spec_report import CornerResults, CornerRunError
from silk.simulation import ConditionState, Transient
from tests._env import needs_ngspice, needs_pdk
from tests.test_run_split import _build_lowpass_tb, _build_nmos_tb


def _rawfile_bytes_ignoring_timestamp(path):
    """A rawfile's bytes with its ``Date:`` header line blanked out.

    ngspice stamps that line with real wall-clock time (second
    resolution), so two separate launches of the identical deck -- exactly
    what a serial-vs-parallel comparison runs -- never agree on it even
    when every actual simulation byte matches. Comparing the rest verifies
    the claim this task actually makes (identical deck/rawfile CONTENT),
    without depending on both launches finishing within the same second.
    """
    with open(path, 'rb') as f:
        data = f.read()
    lines = data.split(b'\n')
    return b'\n'.join(b'Date: <ignored>' if line.startswith(b'Date:') else line
                      for line in lines)


@needs_ngspice
class TestSerialAndParallelAreIdenticalPdkFree:
    """AC #1: run_corners(n_workers=1) and run_corners(n_workers=4) on the
    same bench give identical CornerResults, Spec measurements and
    deck/rawfile content (paths aside)."""

    def _run_both(self, tmp_path, corners=('tt', 'ff', 'ss')):
        dut_a, tb_a = _build_lowpass_tb(str(tmp_path / 'a'), name='rc')
        dut_b, tb_b = _build_lowpass_tb(str(tmp_path / 'b'), name='rc')
        serial = tb_a.run_corners(list(corners), n_workers=1)
        parallel = tb_b.run_corners(list(corners), n_workers=4)
        return dut_a, tb_a, serial, dut_b, tb_b, parallel

    def test_identical_corner_results_and_metrics(self, tmp_path):
        dut_a, tb_a, serial, dut_b, tb_b, parallel = self._run_both(tmp_path)
        try:
            assert isinstance(serial, CornerResults)
            assert isinstance(parallel, CornerResults)
            assert set(serial) == set(parallel) == {'tt', 'ff', 'ss'}
            for corner in ('tt', 'ff', 'ss'):
                assert serial[corner].metrics == parallel[corner].metrics
                assert serial[corner].results[0].passed == parallel[corner].results[0].passed
            assert serial.passed == parallel.passed is True
        finally:
            dut_a.close()
            dut_b.close()

    def test_identical_rawfile_bytes_per_corner(self, tmp_path):
        dut_a, tb_a, serial, dut_b, tb_b, parallel = self._run_both(tmp_path)
        try:
            for corner in ('tt', 'ff', 'ss'):
                assert (_rawfile_bytes_ignoring_timestamp(serial[corner].rawfile_path)
                        == _rawfile_bytes_ignoring_timestamp(parallel[corner].rawfile_path)), corner
        finally:
            dut_a.close()
            dut_b.close()

    def test_n_workers_all_also_matches(self, tmp_path):
        dut_a, tb_a = _build_lowpass_tb(str(tmp_path / 'a'), name='rc')
        try:
            res_all = tb_a.run_corners(['tt', 'ff'], n_workers='all')
            assert set(res_all) == {'tt', 'ff'}
            assert res_all.passed is True
        finally:
            dut_a.close()


@needs_ngspice
class TestEachCornerHasItsOwnFolder:
    """AC #2: each corner has its own folder in serial and parallel; no
    .cir/.raw/.log path is shared; report.rawfile_path is readable after
    the sweep, for every corner."""

    @pytest.mark.parametrize('n_workers', [None, 1, 3])
    def test_no_shared_paths_and_every_path_readable(self, tmp_path, n_workers):
        dut, tb = _build_lowpass_tb(str(tmp_path), name='rc')
        try:
            corners = ['tt', 'ff', 'ss']
            results = tb.run_corners(corners, n_workers=n_workers)
            paths = [results[c].rawfile_path for c in corners]
            assert len(set(paths)) == len(corners), 'a rawfile path was shared'
            decks = set()
            logs = set()
            for corner in corners:
                corner_dir = os.path.dirname(results[corner].rawfile_path)
                assert os.path.basename(os.path.dirname(corner_dir)) == corner
                decks.add(next(f for f in os.listdir(corner_dir) if f.endswith('.cir')))
                logs.add('output.log')  # always present, one per folder -- checked below
                assert os.path.isfile(os.path.join(corner_dir, 'output.log'))
            for path in paths:
                assert os.path.isfile(path)
            # tb.sweep_corners retains every corner's path (TASK-301.3:
            # unconditionally now, not just in 'new' output mode).
            assert tb.sweep_corners == dict(zip(corners, paths))
        finally:
            dut.close()


@needs_ngspice
class TestOneCornerFailingReportsErrorAndRestComplete:
    """AC #3: one corner failing is reported as that corner's error and the
    rest complete."""

    @pytest.mark.parametrize('n_workers', [None, 2])
    def test_failing_corner_does_not_stop_the_others(self, tmp_path, n_workers):
        dut, tb = _build_lowpass_tb(str(tmp_path), name='rc')
        try:
            real_prepare = tb._prepare_run

            def flaky_prepare(corner, **kwargs):
                if str(corner) == 'ff':
                    raise RuntimeError('synthetic failure for corner ff')
                return real_prepare(corner, **kwargs)

            tb._prepare_run = flaky_prepare
            with pytest.raises(CornerRunError) as excinfo:
                tb.run_corners(['tt', 'ff', 'ss'], n_workers=n_workers)
            err = excinfo.value
            assert set(err.errors) == {'ff'}
            assert 'synthetic failure for corner ff' in err.errors['ff']
            assert set(err.results) == {'tt', 'ss'}
            assert err.results['tt'].passed is True
            assert err.results['ss'].passed is True
            # The successful corners' folders/files exist despite the
            # failure elsewhere in the same sweep.
            for corner in ('tt', 'ss'):
                assert os.path.isfile(err.results[corner].rawfile_path)
        finally:
            dut.close()


@needs_ngspice
def test_netlist_identity_unchanged_after_parallel_run_corners(tmp_path):
    """Carried from the TASK-301.2 gate: tb.netlist must be the SAME object
    before and after run_corners(n_workers>1) -- _evaluate_run copies
    simulated state onto the live netlist rather than rebinding it."""
    dut, tb = _build_lowpass_tb(str(tmp_path), name='rc')
    try:
        netlist_before = tb.netlist
        results = tb.run_corners(['tt', 'ff', 'ss'], n_workers=3)
        assert tb.netlist is netlist_before
        assert results.passed is True
    finally:
        dut.close()


@needs_ngspice
@pytest.mark.filterwarnings('ignore:run\\(voltage=...\\) is deprecated:DeprecationWarning')
def test_first_corner_evaluates_with_its_own_state_not_the_last_prepared_corner(tmp_path):
    """Reviewer note (TASK-301.5 cross-check): run_corners() prepares every
    corner in this process before simulating or evaluating any of them, so
    by the time the FIRST corner is evaluated, the live netlist's own
    preparation-phase bookkeeping (``_declared_supply`` and friends, set
    fresh by every ``_prepare_run()`` call) already belongs to the LAST
    corner prepared. Drives ``_prepare_run``/``_simulate_run``/
    ``_evaluate_run`` directly, exactly as ``run_corners()`` does, with two
    corners configured at different temperature AND declared supply, and
    evaluates only the first. The report and characterization context must
    belong to that first corner, not the second -- proving
    ``_SIMULATED_STATE_ATTRS`` plus ``_evaluate_run``'s explicit
    ``operating_conditions``/``characterization_context`` overwrite are
    enough, with no extension needed: ``_declared_supply`` is consumed
    inside ``run_simulator()`` on each corner's own deep-copied netlist
    (never the live one) and is never read anywhere in ``evaluate()``.
    """
    dut, tb = _build_lowpass_tb(str(tmp_path), name='rc')
    try:
        specs = {}
        for corner, temp, volt in (('tt', 25.0, 1.0), ('ff', 85.0, 1.1)):
            tb.set_temperature(temp)
            tb.set_supply_voltage(volt)
            spec = tb._prepare_run(corner, voltage=volt, corner_subdir=corner)
            assert isinstance(spec, _RunSpec)
            specs[corner] = replace(spec, netlist=copy.deepcopy(spec.netlist))
        # The live netlist now holds 'ff' (the LAST corner prepared) --
        # exactly the state run_corners() leaves behind after its own
        # prepare-every-corner-first loop.
        assert tb.netlist._declared_supply == 1.1

        result_tt = ModuleTestbench._simulate_run(specs['tt'])
        report_tt = tb._evaluate_run(result_tt)

        # rc is ideal: the corner is the label, the evidence is none
        # (TASK-406), so temperature and supply tell the corners apart.
        assert report_tt.corner == 'tt'
        assert report_tt.operating_conditions.corner_kind == 'none'
        assert report_tt.operating_conditions.temperature_c == 25.0
        assert report_tt.operating_conditions.supply_voltage_v == 1.0
        assert tb.netlist.characterization_context.corner is ConditionState.NONE
        assert tb.netlist.characterization_context.temperature == 25.0
        assert tb.netlist.characterization_context.voltage == 1.0
        # rc's low-pass settles toward its own supply; distinguishes a
        # report that actually read tt's 1.0 V source from one that read
        # ff's stale 1.1 V leaking in some other way.
        assert report_tt.metrics['vout_final'] == pytest.approx(1.0, abs=0.05)
    finally:
        dut.close()


@pytest.mark.slow
@needs_pdk
@needs_ngspice
def test_serial_and_parallel_are_identical_on_a_real_sky130_bench(tmp_path):
    """AC #1's slow leg: same claim, a real sky130 NMOS DUT this time.

    Same ``name`` ('nmos') for both instances, different directories: the
    DUT's ``module_name`` (and so the deck's Title line) is derived from
    ``name``, not the directory, so a byte-for-byte rawfile comparison needs
    a matching name -- unlike ``tests.test_run_split``'s own 'ref'/'worker'
    pair, which only ever compares ``.metrics``, never raw bytes.
    """
    pdk = load_pdk('sky130')
    dut_a, tb_a = _build_nmos_tb(str(tmp_path / 'a'), pdk, 'nmos')
    dut_b, tb_b = _build_nmos_tb(str(tmp_path / 'b'), pdk, 'nmos')
    try:
        corners = ['tt', 'ff', 'ss']
        serial = tb_a.run_corners(corners, n_workers=1)
        parallel = tb_b.run_corners(corners, n_workers=3)
        assert set(serial) == set(parallel) == set(corners)
        for corner in corners:
            assert serial[corner].metrics == parallel[corner].metrics
            assert (_rawfile_bytes_ignoring_timestamp(serial[corner].rawfile_path)
                    == _rawfile_bytes_ignoring_timestamp(parallel[corner].rawfile_path)), corner
    finally:
        dut_a.close()
        dut_b.close()


@pytest.mark.slow
@needs_pdk
@needs_ngspice
def test_measured_speedup_on_a_five_corner_sky130_bench(tmp_path):
    """AC #3: measured speed-up on a 5-corner sky130 bench, recorded in the
    task notes (not asserted here beyond 'still correct' -- wall-clock
    speed-up is a measurement, not a fast-suite gate)."""
    pdk = load_pdk('sky130')
    corners = ['tt', 'ff', 'ss', 'sf', 'fs']

    dut_s, tb_s = _build_nmos_tb(str(tmp_path / 'serial'), pdk, 'serial')
    t0 = time.time()
    res_serial = tb_s.run_corners(corners, n_workers=1)
    t_serial = time.time() - t0

    dut_p, tb_p = _build_nmos_tb(str(tmp_path / 'parallel'), pdk, 'parallel')
    t0 = time.time()
    res_parallel = tb_p.run_corners(corners, n_workers=5)
    t_parallel = time.time() - t0

    try:
        assert res_serial.passed == res_parallel.passed is True
        print(f'\nrun_corners 5-corner sky130 timing: serial={t_serial:.2f}s '
              f'parallel(n_workers=5)={t_parallel:.2f}s '
              f'speed-up={t_serial / t_parallel:.2f}x')
    finally:
        dut_s.close()
        dut_p.close()


@needs_ngspice
@pytest.mark.parametrize('axis',['temperature','supply','both'])
@pytest.mark.parametrize('local',[False,True],ids=['owner','local'])
@pytest.mark.parametrize('n_workers',[1,2])
def test_run_corners_ignores_plan_lists_but_keeps_scalar_inputs(
        tmp_path,axis,local,n_workers):
    from pathlib import Path
    dut,tb=_build_lowpass_tb(str(tmp_path))
    try:
        tb.set_temperature(27).set_supply_voltage(1.2,source='vin')
        target=tb if local else dut
        if axis in ('temperature','both'):
            target.set_temperatures([-40,125])
        if axis in ('supply','both'):
            target.set_supply_voltages([1.6,1.8],source='vin')
        declarations={id(mod):{name:vars(mod).get(name) for name in
                     ('_temperature_list','_supply_list')} for mod in [dut,tb.netlist]}
        result=tb.run_corners(['tt','ss'],n_workers=n_workers)
        assert list(result)==['tt','ss']
        assert result.summary()=='CornerResults — PASS\n  [PASS]  corner=tt\n  [PASS]  corner=ss'
        assert set(tb.sweep_corners)=={'tt','ss'}
        for corner,report in result.items():
            assert Path(report.rawfile_path).parent.parent.name==corner
            assert report.operating_conditions.temperature_c==27.0
            assert report.operating_conditions.supply_voltage_v==1.2
        for mod in [dut,tb.netlist]:
            assert {name:vars(mod).get(name) for name in
                    ('_temperature_list','_supply_list')}==declarations[id(mod)]
        if local:
            from silk.exceptions import SimulationConfigurationError
            with pytest.raises(SimulationConfigurationError,match='Module.simulate'):
                tb.netlist.simulate()
    finally:
        dut.close()


@needs_ngspice
@pytest.mark.parametrize('n_workers',[1,2])
def test_run_corners_with_local_lists_keeps_failed_corner_and_retained_success(
        tmp_path,monkeypatch,n_workers):
    dut,tb=_build_lowpass_tb(str(tmp_path))
    try:
        tb.set_temperature(27).set_supply_voltage(1.2,source='vin')
        tb.set_temperatures([-40,125]).set_supply_voltages([1.6,1.8],source='vin')
        declarations=(tb.netlist._temperature_list,tb.netlist._supply_list)
        original=tb._prepare_run
        def prepare(corner,*args,**kwargs):
            if corner=='ss':
                raise RuntimeError('selected corner preparation failed')
            return original(corner,*args,**kwargs)
        monkeypatch.setattr(tb,'_prepare_run',prepare)
        with pytest.raises(CornerRunError) as caught:
            tb.run_corners(['tt','ss'],n_workers=n_workers)
        assert list(caught.value.results)==['tt']
        assert list(caught.value.errors)==['ss']
        assert set(tb.sweep_corners)=={'tt'}
        assert (tb.netlist._temperature_list,tb.netlist._supply_list)==declarations
    finally:
        dut.close()


def test_local_lists_reach_real_simulate_guard_with_a_captured_backend(tmp_path,monkeypatch):
    from pathlib import Path
    from tests._mock_rawfile import write_complete_rawfile
    dut,tb=_build_lowpass_tb(str(tmp_path))
    try:
        tb.set_temperature(27).set_supply_voltage(1.2,source='vin')
        tb.set_temperatures([-40,125]).set_supply_voltages([1.6,1.8],source='vin')
        backend=tb.netlist.backend
        launches=[]
        def launch(_backend,netlist_path,results_dir,**kwargs):
            launches.append(netlist_path)
            raw=Path(results_dir)/'captured.raw'
            write_complete_rawfile(str(raw),Path(netlist_path).read_text(),values={'v(vout)':1.25})
            result=backend.load_raw(str(raw))
            result.ambient_temperatures=(27.0,)
            return result
        monkeypatch.setattr(type(backend),'run',launch)
        result=tb.run_corners(['tt','ss'],n_workers=1)
        assert list(result)==['tt','ss']
        assert len(launches)==2
        assert tb.netlist._temperature_list==((-40.0,125.0),None)
        assert tb.netlist._supply_list==((1.6,1.8),'vin')
    finally:
        dut.close()
