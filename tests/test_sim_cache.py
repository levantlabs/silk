"""Tests for ModuleTestbench's report cache -- disabled in 0.2.0 (TASK-284).

cache=True no longer caches: it emits one UserWarning at construction (the
fingerprint hashes device NAMES, specs and the simulator only, so it cannot
detect a changed device value, metric or analysis) and the testbench then
behaves exactly like cache=False -- every run() simulates. See TASK-130 for
the eventual working cache.
"""
import os
from contextlib import contextmanager

import pytest
import silk.pdk
from unittest.mock import patch
from silk.circuit.spec_report import SpecReport
from silk.simulation import OperatingConditions


def _make_report(corner, gain=65.0):
    r = SpecReport(test_name=str(corner), module_name='inv', metrics={'gain': gain})
    r.operating_conditions = OperatingConditions('silk', 'ngspice', str(corner))
    r.check('Gain > 60', 'gain', '>', 60.0)
    return r


@contextmanager
def _mock_simulate(tb, evaluate_side_effect):
    """Stub out the two steps after ``_prepare_run`` (TASK-301.2 split
    ``run()`` into prepare/simulate/evaluate). ``_simulate_run`` becomes a
    passthrough -- so no real simulator ever launches -- and is what these
    tests count calls on, exactly as they counted the old ``_simulate``
    mock; ``_evaluate_run`` gets the caller's report side_effect, called
    with the passed-through ``_RunSpec`` (its ``.corner_str`` stands in for
    the old mock's bare ``corner`` argument).
    """
    with patch.object(tb, '_simulate_run', side_effect=lambda spec: spec) as mock_sim, \
         patch.object(tb, '_evaluate_run', side_effect=evaluate_side_effect):
        yield mock_sim


def _make_tb(tmp_path, name='test', cache=False):
    """Return a wired ModuleTestbench; tests stub only simulation itself."""
    from silk.circuit.modules import Module
    from silk.circuit.module_test import ModuleTestbench
    pdk_yaml = os.path.join(os.path.dirname(silk.pdk.__file__), 'data', 'skywater_130_device_details.yaml')
    try:
        from silk.pdk.basepdk import BasePDK
        pdk = BasePDK(pdk_yaml, library_location='/dev/null')
    except Exception:
        pytest.skip('PDK not available')
    mod = Module('PDK_skywater130', str(tmp_path), 'test.cir', pdk_details=pdk)
    mod.define_module_name('inv')
    mod.define_module_nets({'in': None, 'out': None, 'vdd': None, 'gnd': None})
    tb = ModuleTestbench(mod, name, cache=cache)
    tb.add_module_under_test({'in': 'vin', 'out': 'out', 'vdd': 'vdd', 'gnd': '0'})
    return tb


class TestSimCacheConstruction:
    def test_cache_disabled_by_default(self, tmp_path):
        tb = _make_tb(tmp_path)
        assert tb._cache_enabled is False
        assert tb._cache == {}

    def test_cache_true_is_forced_off(self, tmp_path):
        """cache=True no longer enables caching -- it is forced to False."""
        with pytest.warns(UserWarning, match='TASK-130'):
            tb = _make_tb(tmp_path, cache=True)
        assert tb._cache_enabled is False

    def test_cache_true_warns_once(self, tmp_path):
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _make_tb(tmp_path, cache=True)
        cache_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(cache_warnings) == 1

    def test_cache_false_emits_no_warning(self, tmp_path):
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _make_tb(tmp_path, cache=False)
        assert len(caught) == 0

    def test_cache_starts_empty(self, tmp_path):
        with pytest.warns(UserWarning):
            tb = _make_tb(tmp_path, cache=True)
        assert len(tb._cache) == 0


class TestSimCacheAlwaysSimulates:
    def test_cache_true_still_simulates_every_run(self, tmp_path):
        """Second run() with the same corner no longer hits a cache."""
        with pytest.warns(UserWarning):
            tb = _make_tb(tmp_path, cache=True)
        with _mock_simulate(tb, lambda spec: _make_report(spec.corner_str)) as mock_sim:
            r1 = tb.run(corner='tt')
            r2 = tb.run(corner='tt')
            assert mock_sim.call_count == 2
        assert r1 is not r2

    def test_cache_miss_different_corners(self, tmp_path):
        """Different corners produce separate simulations, as always."""
        with pytest.warns(UserWarning):
            tb = _make_tb(tmp_path, cache=True)
        with _mock_simulate(tb, lambda spec: _make_report(spec.corner_str)) as mock_sim:
            tb.run(corner='tt')
            tb.run(corner='ff')
            assert mock_sim.call_count == 2

    def test_no_cache_always_simulates(self, tmp_path):
        """Without cache, run() always drives a fresh simulate/evaluate."""
        tb = _make_tb(tmp_path, cache=False)
        with _mock_simulate(tb, lambda spec: _make_report(spec.corner_str)) as mock_sim:
            tb.run(corner='tt')
            tb.run(corner='tt')
            assert mock_sim.call_count == 2

    def test_cache_true_never_populates_cache(self, tmp_path):
        """After run(), the (unused) cache dict stays empty."""
        with pytest.warns(UserWarning):
            tb = _make_tb(tmp_path, cache=True)
        with _mock_simulate(tb, lambda spec: _make_report(spec.corner_str)):
            tb.run(corner='tt')
        assert tb._cache == {}


class TestSimCacheDeviceValueChange:
    def test_device_value_change_is_reflected(self, tmp_path):
        """A device-value edit the fingerprint cannot see (it hashes device
        NAMES, not values) is still reflected, because every run() now
        re-simulates regardless of cache=True (TASK-284 AC #2)."""
        with pytest.warns(UserWarning):
            tb = _make_tb(tmp_path, cache=True)
        reports = [_make_report('tt', gain=65.0), _make_report('tt', gain=40.0)]
        with _mock_simulate(tb, reports) as mock_sim:
            r1 = tb.run(corner='tt')
            # Stand-in for editing a device value between runs: the
            # fingerprint (device names/specs/simulator) would not change,
            # but caching being off means run() re-simulates anyway.
            r2 = tb.run(corner='tt')
        assert mock_sim.call_count == 2
        assert r1.metrics['gain'] == 65.0
        assert r2.metrics['gain'] == 40.0


class TestSimCacheClear:
    def test_clear_cache_stays_empty(self, tmp_path):
        with pytest.warns(UserWarning):
            tb = _make_tb(tmp_path, cache=True)
        with _mock_simulate(tb, lambda spec: _make_report(spec.corner_str)):
            tb.run(corner='tt')
        assert tb._cache == {}
        tb.clear_cache()
        assert tb._cache == {}

    def test_clear_cache_does_not_raise(self, tmp_path):
        tb = _make_tb(tmp_path, cache=False)
        tb.clear_cache()
        assert tb._cache == {}

    def test_after_clear_simulates_again(self, tmp_path):
        """clear_cache() remains callable and harmless; run() simulates
        every time regardless, with or without it."""
        with pytest.warns(UserWarning):
            tb = _make_tb(tmp_path, cache=True)
        with _mock_simulate(tb, lambda spec: _make_report(spec.corner_str)) as mock_sim:
            tb.run(corner='tt')
            tb.clear_cache()
            tb.run(corner='tt')
            assert mock_sim.call_count == 2


class TestSimCacheCornerObject:
    def test_corner_object_does_not_hit_cache(self, tmp_path):
        """Corner enum and matching string each simulate -- no cache to hit."""
        from silk.simulation.corner import Corner
        with pytest.warns(UserWarning):
            tb = _make_tb(tmp_path, cache=True)
        with _mock_simulate(tb, lambda spec: _make_report(spec.corner_str)) as mock_sim:
            tb.run(corner=Corner.TT)
            tb.run(corner='tt')
            assert mock_sim.call_count == 2


class TestSimCacheFingerprint:
    def test_fingerprint_is_string(self, tmp_path):
        tb = _make_tb(tmp_path)
        fp = tb._fingerprint()
        assert isinstance(fp, str)
        assert len(fp) > 0

    def test_fingerprint_same_state(self, tmp_path):
        """Same module state → same fingerprint."""
        tb = _make_tb(tmp_path)
        assert tb._fingerprint() == tb._fingerprint()

    def test_fingerprint_changes_with_spec(self, tmp_path):
        """Adding a spec changes the fingerprint."""
        tb = _make_tb(tmp_path)
        fp1 = tb._fingerprint()
        tb.add_spec('Gain > 60', 'gain', '>', 60.0)
        fp2 = tb._fingerprint()
        assert fp1 != fp2
