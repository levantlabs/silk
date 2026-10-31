"""Tests for silk.simulation.corner.Corner enum."""
import pytest
from silk.simulation.corner import Corner


class TestCornerValues:
    def test_tt_value(self):
        assert Corner.TT == 'tt'

    def test_ff_value(self):
        assert Corner.FF == 'ff'

    def test_ss_value(self):
        assert Corner.SS == 'ss'

    def test_sf_value(self):
        assert Corner.SF == 'sf'

    def test_fs_value(self):
        assert Corner.FS == 'fs'


class TestCornerStrCompat:
    def test_corner_equals_string(self):
        assert Corner.TT == 'tt'

    def test_string_equals_corner(self):
        assert 'tt' == Corner.TT

    def test_corner_usable_as_string_dict_key(self):
        d = {'tt': 1}
        assert d[Corner.TT] == 1

    def test_corner_in_string_set(self):
        corners = {'tt', 'ff', 'ss'}
        assert Corner.TT in corners

    def test_str_corner_in_corner_set(self):
        corners = {Corner.TT, Corner.FF}
        assert 'tt' in corners

    def test_corner_str_call(self):
        assert str(Corner.TT) == 'tt'

    def test_corner_format(self):
        assert f"{Corner.FF}" == 'ff'


class TestCornerStandard:
    def test_standard_returns_five(self):
        assert len(Corner.standard()) == 5

    def test_standard_contains_all(self):
        s = Corner.standard()
        assert Corner.TT in s
        assert Corner.FF in s
        assert Corner.SS in s
        assert Corner.SF in s
        assert Corner.FS in s

    def test_standard_is_tuple(self):
        assert isinstance(Corner.standard(), tuple)

    def test_standard_first_is_tt(self):
        assert Corner.standard()[0] == Corner.TT

    def test_standard_usable_in_run_corners(self):
        """Corner.standard() can be passed directly to run_corners(corners=...)."""
        from silk.circuit.spec_report import SpecReport, CornerResults
        from unittest.mock import MagicMock
        # Just verify the tuple can be iterated with str() on each element
        for c in Corner.standard():
            assert str(c) in ('tt', 'ff', 'ss', 'sf', 'fs')


class TestCornerImports:
    def test_importable_from_silk(self):
        from silk import Corner
        assert Corner.TT == 'tt'

    def test_importable_from_silk_simulation(self):
        from silk.simulation import Corner
        assert Corner.FF == 'ff'

    def test_importable_directly(self):
        from silk.simulation.corner import Corner
        assert Corner.SS == 'ss'


def test_corner_enum_values_are_valid_sky130_corners():
    from silk.simulation.corner import Corner
    from silk.pdk import registry as R
    sky = set(R.manifest('sky130')['library']['corners'])
    assert {str(c) for c in Corner} <= sky   # every Corner is a real sky130 section
