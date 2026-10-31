"""Tests for estimate_net_length (statistical wirelength heuristic)."""
import math
import pytest
from silk.layout.wirelength_model import estimate_net_length
from silk.circuit.circuit_elements import PortRole


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeNet:
    def __init__(self, fanout=2):
        self._fanout = fanout

    @property
    def fanout(self):
        return self._fanout


def _net(fanout=2):
    return _FakeNet(fanout=fanout)


# Module with bounding_box set to 10×10 um
def _mod_10x10():
    from types import SimpleNamespace
    m = SimpleNamespace()
    m.bounding_box = (10.0, 10.0)
    return m


# ---------------------------------------------------------------------------
# Per-classification branches
# ---------------------------------------------------------------------------

class TestWirelengthModel:

    def test_power_net_length(self):
        """POWER nets: 2 * perimeter = 2 * 2*(W+H)."""
        m = _mod_10x10()
        length = estimate_net_length(_net(), m, PortRole.POWER)
        perimeter = 2.0 * (10.0 + 10.0)   # 40 um
        assert length == pytest.approx(2.0 * perimeter)

    def test_ground_net_length(self):
        """GROUND gets same treatment as POWER."""
        m = _mod_10x10()
        length = estimate_net_length(_net(), m, PortRole.GROUND)
        perimeter = 2.0 * (10.0 + 10.0)
        assert length == pytest.approx(2.0 * perimeter)

    def test_clock_net_length(self):
        """CLOCK nets: 0.8 * perimeter."""
        m = _mod_10x10()
        length = estimate_net_length(_net(), m, PortRole.CLOCK)
        perimeter = 2.0 * (10.0 + 10.0)
        assert length == pytest.approx(0.8 * perimeter)

    def test_signal_net_uses_rent_rule(self):
        """Signal nets: L_char * sqrt(fanout) * C_signal."""
        m = _mod_10x10()
        net = _net(fanout=4)
        length = estimate_net_length(net, m, PortRole.ANALOG)
        L_char = math.sqrt(10.0 * 10.0)   # 10 um
        expected = L_char * math.sqrt(max(4, 2)) * 0.7
        assert length == pytest.approx(expected)

    def test_none_classification_treated_as_signal(self):
        """None classification falls through to the signal path."""
        m = _mod_10x10()
        net = _net(fanout=2)
        length = estimate_net_length(net, m, None)
        L_char = 10.0
        expected = L_char * math.sqrt(2) * 0.7
        assert length == pytest.approx(expected)

    def test_fanout_floored_at_two(self):
        """Single-pin net (fanout=1) doesn't produce a degenerate length."""
        m = _mod_10x10()
        net = _net(fanout=1)
        length_1 = estimate_net_length(net, m, PortRole.ANALOG)
        net2 = _net(fanout=2)
        length_2 = estimate_net_length(net2, m, PortRole.ANALOG)
        assert length_1 == pytest.approx(length_2)

    def test_larger_fanout_gives_longer_wire(self):
        """Rent's rule: more fanout → longer wire."""
        m = _mod_10x10()
        l4 = estimate_net_length(_net(fanout=4), m, PortRole.ANALOG)
        l16 = estimate_net_length(_net(fanout=16), m, PortRole.ANALOG)
        assert l16 > l4

    def test_returns_positive_float(self):
        m = _mod_10x10()
        assert estimate_net_length(_net(), m, PortRole.POWER) > 0
