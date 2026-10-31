"""Tests for RoutingModel frozen dataclass."""
import pytest
from silk.layout.routing_model import RoutingModel


def _default():
    return RoutingModel(
        sheet_resistance=0.1,
        cap_per_um2=0.04,
        j_max=1e-3,
        min_width=0.14,
        max_width=10.0,
    )


class TestRoutingModelConstruction:

    def test_required_fields_stored(self):
        rm = _default()
        assert rm.sheet_resistance == pytest.approx(0.1)
        assert rm.cap_per_um2 == pytest.approx(0.04)
        assert rm.j_max == pytest.approx(1e-3)
        assert rm.min_width == pytest.approx(0.14)
        assert rm.max_width == pytest.approx(10.0)

    def test_thickness_defaults(self):
        rm = _default()
        assert rm.thickness == pytest.approx(0.36)

    def test_thickness_can_be_set(self):
        rm = RoutingModel(
            sheet_resistance=0.1, cap_per_um2=0.04, j_max=1e-3,
            min_width=0.14, max_width=10.0, thickness=0.5,
        )
        assert rm.thickness == pytest.approx(0.5)

    def test_frozen_raises_on_mutation(self):
        rm = _default()
        with pytest.raises(Exception):
            rm.sheet_resistance = 0.2


class TestRoutingModelId:

    def test_id_is_string(self):
        assert isinstance(_default().id, str)

    def test_id_deterministic(self):
        a = _default()
        b = _default()
        assert a.id == b.id

    def test_id_changes_with_sheet_resistance(self):
        a = _default()
        b = RoutingModel(
            sheet_resistance=0.2, cap_per_um2=0.04, j_max=1e-3,
            min_width=0.14, max_width=10.0,
        )
        assert a.id != b.id

    def test_id_contains_all_key_parameters(self):
        rm = _default()
        # id should encode the five key parameters
        assert '0.1' in rm.id or '1e-01' in rm.id.lower()


# ---------------------------------------------------------------------------
# RoutingModel.from_pdk / stack_from_pdk (TASK-25 — manifest-sourced stack)
# ---------------------------------------------------------------------------

class TestRoutingModelFromPdk:

    def test_from_pdk_by_name_returns_default_layer(self):
        rm = RoutingModel.from_pdk('sky130')
        assert isinstance(rm, RoutingModel)
        # Transcribed byte-for-byte from
        # silk/examples/net_current_persistence_example.py's MET1.
        assert rm.sheet_resistance == pytest.approx(0.125)
        assert rm.cap_per_um2 == pytest.approx(0.04)
        assert rm.j_max == pytest.approx(0.78e-3)
        assert rm.min_width == pytest.approx(0.14)
        assert rm.max_width == pytest.approx(10.0)
        assert rm.thickness == pytest.approx(0.36)   # dataclass default

    def test_from_pdk_explicit_layer_matches_default(self):
        rm_default = RoutingModel.from_pdk('sky130')
        rm_named = RoutingModel.from_pdk('sky130', layer='met1')
        assert rm_named.id == rm_default.id

    def test_from_pdk_accepts_pdk_object(self):
        class _FakePdk:
            pdk_name = 'sky130'

        rm = RoutingModel.from_pdk(_FakePdk())
        assert rm.j_max == pytest.approx(0.78e-3)

    def test_from_pdk_unknown_layer_raises(self):
        with pytest.raises(ValueError, match="met2"):
            RoutingModel.from_pdk('sky130', layer='met2')

    def test_from_pdk_pdk_with_no_routing_section_raises_naming_task(self):
        # GF180MCU has no authoritative sky130-independent metal-stack data
        # in-repo yet (TASK-25 scope note) — the manifest omits 'routing:'.
        with pytest.raises(ValueError, match="declares no 'routing' section"):
            RoutingModel.from_pdk('gf180mcu')

    def test_from_pdk_unknown_pdk_raises(self):
        with pytest.raises(Exception):
            RoutingModel.from_pdk('not_a_real_pdk')

    def test_from_pdk_object_with_no_pdk_name_raises(self):
        class _Nameless:
            pass

        with pytest.raises(ValueError):
            RoutingModel.from_pdk(_Nameless())

    def test_stack_from_pdk_returns_ordered_mapping(self):
        stack = RoutingModel.stack_from_pdk('sky130')
        assert list(stack.keys()) == ['met1']
        assert isinstance(stack['met1'], RoutingModel)
        assert stack['met1'].id == RoutingModel.from_pdk('sky130').id

    def test_stack_from_pdk_no_routing_section_raises(self):
        with pytest.raises(ValueError, match="declares no 'routing' section"):
            RoutingModel.stack_from_pdk('sg13g2')
