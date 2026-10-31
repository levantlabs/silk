"""Tests for Net layout + current stub fields (Phase 1 layout model)."""
from types import SimpleNamespace
import pytest
from silk.circuit.net import Net


def _net(name='n1'):
    return Net(name=name, module=SimpleNamespace())


class TestEstimatedFields:

    def test_estimated_length_defaults_to_none(self):
        assert _net().estimated_length is None

    def test_estimated_width_defaults_to_none(self):
        assert _net().estimated_width is None

    def test_estimated_r_defaults_to_none(self):
        assert _net().estimated_r is None

    def test_estimated_c_defaults_to_none(self):
        assert _net().estimated_c is None

    def test_estimated_length_roundtrip(self):
        n = _net()
        n.estimated_length = 42.5
        assert n.estimated_length == 42.5

    def test_estimated_width_roundtrip(self):
        n = _net()
        n.estimated_width = 0.5
        assert n.estimated_width == 0.5

    def test_estimated_r_roundtrip(self):
        n = _net()
        n.estimated_r = 0.12
        assert n.estimated_r == 0.12

    def test_estimated_c_roundtrip(self):
        n = _net()
        n.estimated_c = 15.3
        assert n.estimated_c == 15.3

    def test_fields_independent_across_instances(self):
        a, b = _net('a'), _net('b')
        a.estimated_r = 1.0
        assert b.estimated_r is None


class TestCurrentStubFields:

    def test_peak_current_defaults_to_none(self):
        assert _net().peak_current is None

    def test_rms_current_defaults_to_none(self):
        assert _net().rms_current is None

    def test_avg_current_defaults_to_none(self):
        assert _net().avg_current is None

    def test_current_source_defaults_to_none(self):
        assert _net()._current_source is None

    def test_peak_current_roundtrip(self):
        n = _net()
        n.peak_current = 2.1e-3
        assert n.peak_current == pytest.approx(2.1e-3)

    def test_current_source_string(self):
        n = _net()
        n._current_source = 'spice'
        assert n._current_source == 'spice'

    def test_current_fields_independent_across_instances(self):
        a, b = _net('a'), _net('b')
        a.peak_current = 1e-3
        assert b.peak_current is None
