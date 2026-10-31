"""TASK-301.7 AC#5: probe-based metrics must keep working in parallel.

Designers use ``tb.probe()``/unbound Signals (TASK-215), not lambdas, for
metrics (owner ruling, 2026-09-27). Probes already survive the pickle round
trip (``Signal.__getstate__``/``ModuleTestbench.__getstate__``, TASK-215.2/
215.8) independently of the lambda-metric fix in this task -- this is a
regression guard, not a repro of the reported bug.

``silk/examples/source_follower_example.py`` owns the exact two-tests-one-DUT
shape the reported bug needs (``mod._tests['ac']`` then ``mod._tests['zout']``,
both probe-based): ``zout``'s netlist embeds a DUT copy carrying a replica of
``ac``'s Spec, so this is also, incidentally, a real-example check that the
fix did not regress the probe path while cutting ``Spec._owning_testbench``
for pickling.

Reviewer measurement on main 3df075f4 (recorded in the task notes): run_all_tests
serial vs n_workers=2 identical (gain_vv 0.83949, bw_hz 1.2589e8,
rout_ohm 2418.0), run_corners tt/ss/ff serial vs n_workers=3 identical.
"""
import pytest

import silk.examples.source_follower_example as _sfe
from silk import load_pdk
from silk.examples.source_follower_example import build_source_follower
from tests._env import needs_ngspice, needs_pdk

pytestmark = [needs_pdk, needs_ngspice]


@pytest.fixture
def isolated_pdk(monkeypatch):
    """Re-point the example module's frozen ``pdk`` global at a FRESH
    ``load_pdk('sky130')`` call, made right here.

    ``source_follower_example.pdk`` is set once, at that module's own
    import time. Any OTHER ``load_pdk('sky130')`` call anywhere else in
    the same process (another example module's own import-time global,
    ``tests._env``'s availability probe, ...) re-registers
    ``silk.elements.factory``'s device classes by name -- 'last
    registration wins' -- and leaves the example module's already-loaded
    ``pdk`` pointing at now-stale classes, unpicklable even though nothing
    about THIS test's circuit changed (see
    ``tests/test_run_all_tests_parallel.py``'s module docstring). Loading
    fresh here, immediately before building the two modules this test
    compares, guarantees no OTHER ``load_pdk`` call can land in between.
    """
    monkeypatch.setattr(_sfe, 'pdk', load_pdk('sky130'))


class TestRunAllTestsParallel:
    def test_n_workers_2_matches_serial(self, tmp_path, isolated_pdk):
        serial_mod = build_source_follower()
        serial_mod.dir_folder = str(tmp_path / 'serial')
        serial = serial_mod.run_all_tests(corner='tt', n_workers=1)

        parallel_mod = build_source_follower()
        parallel_mod.dir_folder = str(tmp_path / 'parallel')
        parallel = parallel_mod.run_all_tests(corner='tt', n_workers=2)

        expected_keys = {f'source_follower_nmos.{name}' for name in ('ac', 'zout')}
        assert set(serial) == set(parallel) == expected_keys
        for key in expected_keys:
            s, p = serial[key], parallel[key]
            assert not isinstance(s, dict), s
            assert not isinstance(p, dict), p
            assert p.metrics == pytest.approx(s.metrics)
            assert [r.passed for r in p.results] == [r.passed for r in s.results]


class TestRunCornersParallel:
    def test_zout_n_workers_2_matches_serial(self, tmp_path, isolated_pdk):
        """``zout``'s own netlist is the one that embeds ``ac``'s Spec."""
        serial_mod = build_source_follower()
        serial_mod.dir_folder = str(tmp_path / 'serial')
        serial = serial_mod._tests['zout'].run_corners(
            corners=['tt', 'ff'], n_workers=1)

        parallel_mod = build_source_follower()
        parallel_mod.dir_folder = str(tmp_path / 'parallel')
        parallel = parallel_mod._tests['zout'].run_corners(
            corners=['tt', 'ff'], n_workers=2)

        assert set(serial) == set(parallel) == {'tt', 'ff'}
        for corner in serial:
            assert parallel[corner].metrics == pytest.approx(serial[corner].metrics)


@pytest.mark.slow
def test_calibration_reaches_the_behavioral_instance(isolated_pdk):
    """The comparison must execute the newly calibrated offset, not its seed."""
    from silk.examples.source_follower_calibrate import calibrate, _STEP_LO

    mod, _, _, after, _, (_, behavioral) = calibrate()
    expected_initial = after["gain"] * _STEP_LO - after["offset"]
    assert after["offset"] != pytest.approx(0.55)
    assert behavioral[0] == pytest.approx(expected_initial)
    assert mod._behavioral_params["offset"] == after["offset"]
