"""Tests for AnalogElement.estimated_area field (Phase 1 layout model)."""
import pytest


def _make_device():
    """Return a real factory-generated NFET instance for testing."""
    from silk.elements.factory import make_device_class
    spec = {
        'PDKDevice': 'sky130_fd_pr__nfet_01v8',
        'paramList': {
            'W': {'minval': 0.42e-6, 'maxval': -1, 'unit': 'u'},
            'L': {'minval': 0.15e-6, 'maxval': -1, 'unit': 'u'},
            'M': {'minval': 1, 'maxval': -1},
        },
        'pinList': {'d': 'drain', 'g': 'gate', 's': 'source', 'b': 'bulk'},
    }
    Nfet = make_device_class('test_nfet', spec, is_pdk=True)
    return Nfet(name='mn1', d='vout', g='vin', s='vss', b='vss',
                W=4e-6, L=180e-9, M=1)


class TestEstimatedArea:

    def test_defaults_to_none(self):
        dev = _make_device()
        assert dev.estimated_area is None

    def test_can_be_set(self):
        dev = _make_device()
        dev.estimated_area = 14.4
        assert dev.estimated_area == pytest.approx(14.4)

    def test_can_be_cleared_to_none(self):
        dev = _make_device()
        dev.estimated_area = 10.0
        dev.estimated_area = None
        assert dev.estimated_area is None

    def test_independent_across_instances(self):
        a = _make_device()
        b = _make_device()
        b.name = 'mn2'
        a.estimated_area = 5.0
        assert b.estimated_area is None
