"""Tests for Module.half_circuits registry and bounding_box field."""
import pytest
from silk.layout.matching import HalfCircuit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_module(pdk, tmp_path):
    from silk.circuit.modules import Module
    return Module(
        'PDK_skywater130',
        str(tmp_path),
        'test.cir',
        pdk_details=pdk,
        module_name='test_amp',
    )


def _add_nfet(module, pdk, name, d='vout', g='vin', s='vss', b='vss'):
    module.add(pdk.nmos1p8(name, d=d, g=g, s=s, b=b, w=4, l=0.18, m=1))


# ---------------------------------------------------------------------------
# bounding_box field
# ---------------------------------------------------------------------------

class TestBoundingBox:

    def test_bounding_box_defaults_to_none(self, pdk, tmp_path):
        m = _make_module(pdk, tmp_path)
        assert m.bounding_box is None

    def test_bounding_box_can_be_set(self, pdk, tmp_path):
        m = _make_module(pdk, tmp_path)
        m.bounding_box = (10.0, 10.0)
        assert m.bounding_box == (10.0, 10.0)


# ---------------------------------------------------------------------------
# half_circuits registry
# ---------------------------------------------------------------------------

class TestHalfCircuitsRegistry:

    def test_half_circuits_defaults_to_empty_dict(self, pdk, tmp_path):
        m = _make_module(pdk, tmp_path)
        assert m.half_circuits == {}

    def test_declare_half_circuit_happy_path(self, pdk, tmp_path):
        m = _make_module(pdk, tmp_path)
        _add_nfet(m, pdk, 'mn1', d='vout_p')
        _add_nfet(m, pdk, 'mn2', d='vout_n')
        hc = m.declare_half_circuit('diff_in', left=['mn1'], right=['mn2'])
        assert isinstance(hc, HalfCircuit)
        assert hc.name == 'diff_in'
        assert 'diff_in' in m.half_circuits
        assert m.half_circuits['diff_in'] is hc

    def test_declare_half_circuit_returns_half_circuit(self, pdk, tmp_path):
        m = _make_module(pdk, tmp_path)
        _add_nfet(m, pdk, 'mn1')
        _add_nfet(m, pdk, 'mn2')
        result = m.declare_half_circuit('pair', left=['mn1'], right=['mn2'])
        assert isinstance(result, HalfCircuit)

    def test_declare_half_circuit_unknown_member_raises(self, pdk, tmp_path):
        m = _make_module(pdk, tmp_path)
        _add_nfet(m, pdk, 'mn1')
        with pytest.raises(ValueError, match="not in elem_dict"):
            m.declare_half_circuit('pair', left=['mn1'], right=['ghost'])

    def test_declare_half_circuit_duplicate_name_raises(self, pdk, tmp_path):
        m = _make_module(pdk, tmp_path)
        _add_nfet(m, pdk, 'mn1')
        _add_nfet(m, pdk, 'mn2')
        m.declare_half_circuit('pair', left=['mn1'], right=['mn2'])
        with pytest.raises(ValueError, match="already declared"):
            m.declare_half_circuit('pair', left=['mn1'], right=['mn2'])

    def test_declare_match_pair_sugar(self, pdk, tmp_path):
        m = _make_module(pdk, tmp_path)
        _add_nfet(m, pdk, 'mn1')
        _add_nfet(m, pdk, 'mn2')
        hc = m.declare_match_pair('pair', 'mn1', 'mn2')
        assert hc.left == ('mn1',)
        assert hc.right == ('mn2',)

    def test_declare_match_pair_stored_in_registry(self, pdk, tmp_path):
        m = _make_module(pdk, tmp_path)
        _add_nfet(m, pdk, 'mn1')
        _add_nfet(m, pdk, 'mn2')
        m.declare_match_pair('p', 'mn1', 'mn2')
        assert 'p' in m.half_circuits

    def test_strategy_passed_through(self, pdk, tmp_path):
        m = _make_module(pdk, tmp_path)
        _add_nfet(m, pdk, 'mn1')
        _add_nfet(m, pdk, 'mn2')
        hc = m.declare_half_circuit('pair', left=['mn1'], right=['mn2'],
                                    strategy='common_centroid')
        assert hc.strategy == 'common_centroid'
