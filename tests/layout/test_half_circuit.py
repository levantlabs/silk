"""Tests for HalfCircuit frozen dataclass and LayoutInferenceError."""
import pytest
from silk.layout.matching import HalfCircuit, LayoutInferenceError


class TestHalfCircuitValidation:

    def test_unequal_sides_raises(self):
        with pytest.raises(ValueError, match="equal lengths"):
            HalfCircuit(name='pair', left=('mn1', 'mn3'), right=('mn2',))

    def test_empty_sides_raises(self):
        with pytest.raises(ValueError, match="empty sides"):
            HalfCircuit(name='pair', left=(), right=())

    def test_overlapping_members_raises(self):
        with pytest.raises(ValueError, match="both sides"):
            HalfCircuit(name='pair', left=('mn1',), right=('mn1',))

    def test_valid_pair(self):
        hc = HalfCircuit(name='diff_in', left=('mn1',), right=('mn2',))
        assert hc.name == 'diff_in'
        assert hc.left == ('mn1',)
        assert hc.right == ('mn2',)

    def test_valid_multi_member(self):
        hc = HalfCircuit(
            name='cascode',
            left=('mn1', 'mn3'),
            right=('mn2', 'mn4'),
        )
        assert len(hc.left) == 2
        assert len(hc.right) == 2

    def test_strategy_defaults_to_none(self):
        hc = HalfCircuit(name='p', left=('a',), right=('b',))
        assert hc.strategy is None

    def test_strategy_can_be_set(self):
        hc = HalfCircuit(name='p', left=('a',), right=('b',), strategy='common_centroid')
        assert hc.strategy == 'common_centroid'

    def test_frozen_raises_on_mutation(self):
        hc = HalfCircuit(name='p', left=('a',), right=('b',))
        with pytest.raises(Exception):  # FrozenInstanceError
            hc.name = 'q'

    def test_lists_converted_to_tuples(self):
        """Passing lists should silently become tuples via __post_init__."""
        hc = HalfCircuit(name='p', left=['mn1'], right=['mn2'])
        assert isinstance(hc.left, tuple)
        assert isinstance(hc.right, tuple)


class TestLayoutInferenceError:

    def test_is_exception(self):
        err = LayoutInferenceError("test")
        assert isinstance(err, Exception)

    def test_message_preserved(self):
        err = LayoutInferenceError("bad match config")
        assert "bad match config" in str(err)
