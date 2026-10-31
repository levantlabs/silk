"""Tests for LayoutEstimate dataclasses, compute_scalars, estimate_layout, capture_layout."""
import time
import pytest


# ---------------------------------------------------------------------------
# LayoutWarning (in _types)
# ---------------------------------------------------------------------------

class TestLayoutWarning:

    def test_construction(self):
        from silk.layout._types import LayoutWarning
        w = LayoutWarning(net_name='vout', reason='no_current_data',
                          severity='info', detail='no current')
        assert w.net_name == 'vout'
        assert w.reason == 'no_current_data'

    def test_frozen_raises_on_mutation(self):
        from silk.layout._types import LayoutWarning
        w = LayoutWarning(net_name='a', reason='r', severity='info', detail='d')
        with pytest.raises(Exception):
            w.reason = 'other'

    def test_hashable(self):
        from silk.layout._types import LayoutWarning
        w = LayoutWarning(net_name='a', reason='r', severity='info', detail='d')
        assert hash(w) is not None


# ---------------------------------------------------------------------------
# Sub-dataclasses
# ---------------------------------------------------------------------------

class TestNetLayout:

    def test_construction(self):
        from silk.layout.estimator import NetLayout
        nl = NetLayout(length=50.0, width=0.5, r=0.1, c=10.0, equivalence_class=None)
        assert nl.length == 50.0
        assert nl.equivalence_class is None

    def test_frozen(self):
        from silk.layout.estimator import NetLayout
        nl = NetLayout(length=1.0, width=0.14, r=0.01, c=0.5, equivalence_class=None)
        with pytest.raises(Exception):
            nl.length = 99.0

    def test_equivalence_class_can_be_set(self):
        from silk.layout.estimator import NetLayout
        nl = NetLayout(length=1.0, width=0.14, r=0.01, c=0.5, equivalence_class='cls_vout_p')
        assert nl.equivalence_class == 'cls_vout_p'

    def test_c_farads_converts_femtofarads(self):
        """AUDIT-02 row 27 / TASK-25: the sole fF->F accessor."""
        from silk.layout.estimator import NetLayout
        nl = NetLayout(length=1.0, width=0.14, r=0.01, c=10.0, equivalence_class=None)
        assert nl.c_farads == pytest.approx(10.0e-15)

    def test_c_farads_zero(self):
        from silk.layout.estimator import NetLayout
        nl = NetLayout(length=1.0, width=0.14, r=0.01, c=0.0, equivalence_class=None)
        assert nl.c_farads == pytest.approx(0.0)


class TestDeviceLayout:

    def test_construction(self):
        from silk.layout.estimator import DeviceLayout
        dl = DeviceLayout(area=1.8)
        assert dl.area == pytest.approx(1.8)

    def test_frozen(self):
        from silk.layout.estimator import DeviceLayout
        dl = DeviceLayout(area=1.0)
        with pytest.raises(Exception):
            dl.area = 2.0


class TestLayoutEstimate:

    def _make(self):
        from silk.layout.estimator import LayoutEstimate, NetLayout, DeviceLayout
        from silk.layout._types import LayoutWarning
        return LayoutEstimate(
            computation_mode='statistical',
            routing_model_id='0.1|0.04|0.001|0.14|10|0.36',
            timestamp=time.time(),
            nets={'vout': NetLayout(50.0, 0.5, 0.1, 10.0, None)},
            devices={'mn1': DeviceLayout(1.8)},
            bounding_box=(10.0, 10.0),
            equivalence_classes={},
            warnings=(),
        )

    def test_construction(self):
        e = self._make()
        assert e.computation_mode == 'statistical'
        assert 'vout' in e.nets

    def test_frozen(self):
        e = self._make()
        with pytest.raises(Exception):
            e.computation_mode = 'other'

    def test_warnings_is_tuple(self):
        e = self._make()
        assert isinstance(e.warnings, tuple)


# ---------------------------------------------------------------------------
# compute_scalars
# ---------------------------------------------------------------------------

class TestComputeScalars:

    def _rm(self):
        from silk.layout.routing_model import RoutingModel
        return RoutingModel(sheet_resistance=0.1, cap_per_um2=0.04,
                            j_max=1e-3, min_width=0.14, max_width=10.0)

    def test_current_aware_width(self):
        from silk.layout.estimator import compute_scalars
        from silk.layout.matching import _CanonicalInputs
        canon = _CanonicalInputs(peak_current=2e-3, fanout=2,
                                  classification=None, length=50.0)
        width, r, c = compute_scalars(canon, self._rm())
        # width = current / j_max = 2e-3 / 1e-3 = 2.0 um
        assert width == pytest.approx(2.0)

    def test_topology_only_uses_min_width(self):
        from silk.layout.estimator import compute_scalars
        from silk.layout.matching import _CanonicalInputs
        canon = _CanonicalInputs(peak_current=None, fanout=2,
                                  classification=None, length=50.0)
        width, r, c = compute_scalars(canon, self._rm())
        assert width == pytest.approx(0.14)   # min_width

    def test_width_clamped_to_max(self):
        from silk.layout.estimator import compute_scalars
        from silk.layout.matching import _CanonicalInputs
        canon = _CanonicalInputs(peak_current=100e-3, fanout=2,
                                  classification=None, length=50.0)
        width, r, c = compute_scalars(canon, self._rm())
        assert width == pytest.approx(10.0)   # max_width

    def test_r_formula(self):
        from silk.layout.estimator import compute_scalars
        from silk.layout.matching import _CanonicalInputs
        # width = 1.0, length = 100um, sheet_R = 0.1
        canon = _CanonicalInputs(peak_current=1e-3, fanout=2,
                                  classification=None, length=100.0)
        width, r, c = compute_scalars(canon, self._rm())
        # width = 1.0 um; R = L * Rs / W = 100 * 0.1 / 1.0 = 10 ohm
        assert r == pytest.approx(10.0)

    def test_c_formula(self):
        from silk.layout.estimator import compute_scalars
        from silk.layout.matching import _CanonicalInputs
        # width=1.0, length=100um, cap_per_um2=0.04 fF/um²
        canon = _CanonicalInputs(peak_current=1e-3, fanout=2,
                                  classification=None, length=100.0)
        width, r, c = compute_scalars(canon, self._rm())
        # C = L * W * cap = 100 * 1.0 * 0.04 = 4 fF
        assert c == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# compute_scalars — layer escalation (TASK-25)
# ---------------------------------------------------------------------------

class TestComputeScalarsEscalation:

    def _met1(self):
        from silk.layout.routing_model import RoutingModel
        # min_width * j_max = 0.14 * 1e-3 = 1.4e-4 A EM headroom at min-width.
        return RoutingModel(sheet_resistance=0.1, cap_per_um2=0.04,
                            j_max=1e-3, min_width=0.14, max_width=10.0)

    def _met2(self):
        from silk.layout.routing_model import RoutingModel
        # Different constants so escalated R/C are visibly distinguishable.
        return RoutingModel(sheet_resistance=0.05, cap_per_um2=0.08,
                            j_max=1.2e-3, min_width=0.28, max_width=20.0)

    def _canon(self, peak_current):
        from silk.layout.matching import _CanonicalInputs
        return _CanonicalInputs(peak_current=peak_current, fanout=2,
                                classification=None, length=100.0)

    def test_single_routing_model_never_escalates(self):
        """A bare RoutingModel is unchanged from Phase 1 — identical
        arithmetic, no escalation_rule call, no behaviour change."""
        from silk.layout.estimator import compute_scalars
        rm = self._met1()
        canon = self._canon(1e-3)
        width, r, c = compute_scalars(canon, rm)
        assert width == pytest.approx(1.0)
        assert r == pytest.approx(100.0 * 0.1 / 1.0)
        assert c == pytest.approx(100.0 * 1.0 * 0.04)

    def test_low_current_stays_on_lowest_layer(self):
        from silk.layout.estimator import compute_scalars
        canon = self._canon(1e-5)   # well under met1's 1.4e-4 A headroom
        width, r, c = compute_scalars(canon, [self._met1(), self._met2()])
        met1 = self._met1()
        assert width == pytest.approx(met1.min_width)
        assert r == pytest.approx(100.0 * met1.sheet_resistance / met1.min_width)

    def test_high_current_escalates_to_next_layer(self):
        from silk.layout.estimator import compute_scalars
        canon = self._canon(5e-3)   # exceeds met1's 1.4e-4 A min-width headroom
        width, r, c = compute_scalars(canon, [self._met1(), self._met2()])
        met2 = self._met2()
        expected_width = max(met2.min_width, min(met2.max_width, 5e-3 / met2.j_max))
        assert width == pytest.approx(expected_width)
        assert r == pytest.approx(100.0 * met2.sheet_resistance / expected_width)
        assert c == pytest.approx(100.0 * expected_width * met2.cap_per_um2)

    def test_escalated_r_differs_from_unescalated(self):
        """AC#2: pre-route R of an escalated net is priced on the escalated
        layer, not the lowest one."""
        from silk.layout.estimator import compute_scalars
        canon = self._canon(5e-3)
        _, r_stack, _ = compute_scalars(canon, [self._met1(), self._met2()])
        _, r_met1_only, _ = compute_scalars(canon, self._met1())
        assert r_stack != pytest.approx(r_met1_only)

    def test_final_layer_is_fallback_even_if_still_em_constrained(self):
        """A current that would violate EM even on the top layer's min-width
        still prices on the top layer (matches Phase-1's own max_width clamp
        for a single-layer stack — no infinite escalation)."""
        from silk.layout.estimator import compute_scalars
        canon = self._canon(1.0)   # absurdly high
        width, r, c = compute_scalars(canon, [self._met1(), self._met2()])
        met2 = self._met2()
        assert width == pytest.approx(met2.max_width)

    def test_none_current_uses_lowest_layer_min_width(self):
        from silk.layout.estimator import compute_scalars
        canon = self._canon(None)
        width, r, c = compute_scalars(canon, [self._met1(), self._met2()])
        assert width == pytest.approx(self._met1().min_width)

    def test_dict_stack_same_as_sequence_stack(self):
        from silk.layout.estimator import compute_scalars
        canon = self._canon(5e-3)
        seq_result = compute_scalars(canon, [self._met1(), self._met2()])
        dict_result = compute_scalars(
            canon, {'met1': self._met1(), 'met2': self._met2()})
        assert seq_result == pytest.approx(dict_result)

    def test_custom_escalation_rule_is_honored(self):
        from silk.layout.estimator import compute_scalars

        def always_top(peak_current, stack):
            return stack[-1]

        canon = self._canon(1e-5)   # would normally stay on met1
        width, r, c = compute_scalars(
            canon, [self._met1(), self._met2()], escalation_rule=always_top)
        assert width == pytest.approx(self._met2().min_width)

    def test_empty_stack_raises(self):
        from silk.layout.estimator import compute_scalars
        canon = self._canon(1e-3)
        with pytest.raises(ValueError):
            compute_scalars(canon, [])

    def test_non_routing_model_stack_entry_raises(self):
        from silk.layout.estimator import compute_scalars
        canon = self._canon(1e-3)
        with pytest.raises(TypeError):
            compute_scalars(canon, [self._met1(), 'not a RoutingModel'])


class TestDefaultEscalationRule:

    def _met1(self):
        from silk.layout.routing_model import RoutingModel
        return RoutingModel(sheet_resistance=0.1, cap_per_um2=0.04,
                            j_max=1e-3, min_width=0.14, max_width=10.0)

    def _met2(self):
        from silk.layout.routing_model import RoutingModel
        return RoutingModel(sheet_resistance=0.05, cap_per_um2=0.08,
                            j_max=1.2e-3, min_width=0.28, max_width=20.0)

    def test_escalates_only_above_minimum_width_current_boundary(self):
        """Escalate exactly when current exceeds min_width * j_max."""
        from silk.layout.estimator import default_escalation_rule
        met1 = self._met1()
        stack = (met1, self._met2())
        boundary = met1.min_width * met1.j_max
        assert default_escalation_rule(boundary, stack) is met1
        assert default_escalation_rule(boundary * 1.0001, stack) is not met1

    def test_single_layer_stack_always_returns_it(self):
        from silk.layout.estimator import default_escalation_rule
        met1 = self._met1()
        assert default_escalation_rule(1.0, (met1,)) is met1
        assert default_escalation_rule(None, (met1,)) is met1


# ---------------------------------------------------------------------------
# estimate_layout integration
# ---------------------------------------------------------------------------

class TestEstimateLayout:

    def _make_diff_amp(self, pdk, tmp_path):
        """Build a minimal diff amp (2 NFETs) for integration tests."""
        from silk.circuit.modules import Module
        amp = Module('PDK_skywater130', str(tmp_path), 'amp.cir',
                     pdk_details=pdk, module_name='amp')
        amp.add(pdk.nmos1p8('mn1', d='vout_p', g='vin_p', s='tail', b='vss', w=4, l=0.18, m=1))
        amp.add(pdk.nmos1p8('mn2', d='vout_n', g='vin_n', s='tail', b='vss', w=4, l=0.18, m=1))
        return amp

    def _routing_model(self):
        from silk.layout.routing_model import RoutingModel
        return RoutingModel(sheet_resistance=0.1, cap_per_um2=0.04,
                            j_max=1e-3, min_width=0.14, max_width=10.0)

    def test_returns_layout_estimate(self, pdk, tmp_path):
        from silk.layout.estimator import estimate_layout, LayoutEstimate
        amp = self._make_diff_amp(pdk, tmp_path)
        est = estimate_layout(amp, self._routing_model())
        assert isinstance(est, LayoutEstimate)

    def test_computation_mode_is_statistical(self, pdk, tmp_path):
        from silk.layout.estimator import estimate_layout
        amp = self._make_diff_amp(pdk, tmp_path)
        est = estimate_layout(amp, self._routing_model())
        assert est.computation_mode == 'statistical'

    def test_net_fields_populated(self, pdk, tmp_path):
        from silk.layout.estimator import estimate_layout
        amp = self._make_diff_amp(pdk, tmp_path)
        estimate_layout(amp, self._routing_model())
        for net in amp.net_dict.values():
            assert net.estimated_length is not None
            assert net.estimated_width is not None
            assert net.estimated_r is not None
            assert net.estimated_c is not None

    def test_device_estimated_area_populated(self, pdk, tmp_path):
        from silk.layout.estimator import estimate_layout
        amp = self._make_diff_amp(pdk, tmp_path)
        estimate_layout(amp, self._routing_model())
        for entry in amp.elem_dict.values():
            assert entry['device'].estimated_area is not None

    def test_module_bounding_box_set(self, pdk, tmp_path):
        from silk.layout.estimator import estimate_layout
        amp = self._make_diff_amp(pdk, tmp_path)
        estimate_layout(amp, self._routing_model())
        assert amp.bounding_box is not None

    def test_matched_pair_symmetrization(self, pdk, tmp_path):
        """vout_p and vout_n must get identical scalars when declared matched."""
        from silk.layout.estimator import estimate_layout
        amp = self._make_diff_amp(pdk, tmp_path)
        amp.declare_match_pair('diff_in', 'mn1', 'mn2')
        # Slightly asymmetric input currents
        amp.net_dict['vout_p'].peak_current = 2.1e-3
        amp.net_dict['vout_n'].peak_current = 1.8e-3
        estimate_layout(amp, self._routing_model())
        assert amp.net_dict['vout_p'].estimated_r == pytest.approx(
            amp.net_dict['vout_n'].estimated_r)
        assert amp.net_dict['vout_p'].estimated_c == pytest.approx(
            amp.net_dict['vout_n'].estimated_c)

    def test_no_current_emits_warnings(self, pdk, tmp_path):
        """With no currents set, all nets get no_current_data warnings."""
        from silk.layout.estimator import estimate_layout
        amp = self._make_diff_amp(pdk, tmp_path)
        est = estimate_layout(amp, self._routing_model())
        reasons = [w.reason for w in est.warnings]
        assert 'no_current_data' in reasons

    def test_estimate_has_routing_model_id(self, pdk, tmp_path):
        from silk.layout.estimator import estimate_layout
        amp = self._make_diff_amp(pdk, tmp_path)
        rm = self._routing_model()
        est = estimate_layout(amp, rm)
        assert est.routing_model_id == rm.id

    def test_equivalence_classes_populated_for_matched_pair(self, pdk, tmp_path):
        from silk.layout.estimator import estimate_layout
        amp = self._make_diff_amp(pdk, tmp_path)
        amp.declare_match_pair('diff_in', 'mn1', 'mn2')
        est = estimate_layout(amp, self._routing_model())
        # At least one size-2 class (vout_p/vout_n)
        assert any(len(v) >= 2 for v in est.equivalence_classes.values())


# ---------------------------------------------------------------------------
# estimate_layout — required_width_exceeds_max warning (TASK-204)
# ---------------------------------------------------------------------------

class TestRequiredWidthExceedsMaxWarning:
    """AC#1-3: notice when the current-based required width exceeds the
    selected layer's max_width; no notice below/at the limit, at zero
    current, or when current data is simply absent."""

    def _make_single_net_module(self, tmp_path):
        from silk.circuit.modules import Module
        from silk.elements import ElementLibrary
        mod = Module(None, str(tmp_path), 'routing.cir', pdk_details=None)
        elements = ElementLibrary()
        with mod:
            elements.resistor('r1', p='route', n='return', value=1000.0)
        return mod

    def _rm(self):
        from silk.layout.routing_model import RoutingModel
        return RoutingModel(sheet_resistance=0.1, cap_per_um2=0.04,
                            j_max=1e-3, min_width=0.14, max_width=10.0)

    def _reasons(self, est):
        return [w.reason for w in est.warnings]

    def test_capped_current_emits_warning(self, tmp_path):
        """AC#1/AC#2: 20 mA at j_max=1e-3 A/um requires 20 um, capped at
        max_width=10 um — the exact TASK-204 reproducer."""
        from silk.layout.estimator import estimate_layout
        mod = self._make_single_net_module(tmp_path)
        for net in mod.net_dict.values():
            net.peak_current = 0.020
        est = estimate_layout(mod, self._rm())
        assert est.nets['route'].width == pytest.approx(10.0)   # unchanged (AC#7)
        cap_warnings = [w for w in est.warnings if w.reason == 'required_width_exceeds_max']
        # Both pins of the resistor (route, return) share the 20 mA
        # annotation, so both singleton nets are individually capped.
        assert {w.net_name for w in cap_warnings} == {'route', 'return'}
        w = next(w for w in cap_warnings if w.net_name == 'route')
        assert w.severity == 'warn'
        mod.close()

    def test_detail_reports_current_jmax_required_and_capped_width(self, tmp_path):
        from silk.layout.estimator import estimate_layout
        mod = self._make_single_net_module(tmp_path)
        for net in mod.net_dict.values():
            net.peak_current = 0.020
        est = estimate_layout(mod, self._rm())
        w = next(w for w in est.warnings if w.reason == 'required_width_exceeds_max')
        detail = w.detail
        assert '0.02' in detail                # sizing current
        assert '0.001' in detail                # selected layer j_max
        assert '20' in detail                   # required width
        assert '10' in detail                   # capped width
        assert 'single-wire' in detail
        assert 'does not model or verify' in detail.lower() or \
               'does not model' in detail.lower()
        mod.close()

    def test_no_warning_below_max(self, tmp_path):
        from silk.layout.estimator import estimate_layout
        mod = self._make_single_net_module(tmp_path)
        for net in mod.net_dict.values():
            net.peak_current = 0.002   # 2 mA -> required 2um < max_width 10um
        est = estimate_layout(mod, self._rm())
        assert 'required_width_exceeds_max' not in self._reasons(est)
        mod.close()

    def test_no_warning_exactly_at_max(self, tmp_path):
        from silk.layout.estimator import estimate_layout
        mod = self._make_single_net_module(tmp_path)
        for net in mod.net_dict.values():
            net.peak_current = 0.010   # 10 mA -> required exactly 10um == max_width
        est = estimate_layout(mod, self._rm())
        assert est.nets['route'].width == pytest.approx(10.0)
        assert 'required_width_exceeds_max' not in self._reasons(est)
        mod.close()

    def test_no_warning_at_zero_current(self, tmp_path):
        from silk.layout.estimator import estimate_layout
        mod = self._make_single_net_module(tmp_path)
        for net in mod.net_dict.values():
            net.peak_current = 0.0
        est = estimate_layout(mod, self._rm())
        assert 'required_width_exceeds_max' not in self._reasons(est)
        mod.close()

    def test_no_warning_when_current_data_absent(self, tmp_path):
        """AC#3: absent current data alone must not trigger the new notice
        — the existing no_current_data warning is unchanged."""
        from silk.layout.estimator import estimate_layout
        mod = self._make_single_net_module(tmp_path)
        est = estimate_layout(mod, self._rm())
        reasons = self._reasons(est)
        assert 'required_width_exceeds_max' not in reasons
        assert 'no_current_data' in reasons
        mod.close()

    def test_capped_width_r_c_match_uncapped_formula(self, tmp_path):
        """AC#7: R/C are still computed from the (capped) width, unchanged."""
        from silk.layout.estimator import estimate_layout
        mod = self._make_single_net_module(tmp_path)
        for net in mod.net_dict.values():
            net.peak_current = 0.020
        rm = self._rm()
        est = estimate_layout(mod, rm)
        nl = est.nets['route']
        assert nl.r == pytest.approx(nl.length * rm.sheet_resistance / nl.width)
        assert nl.c == pytest.approx(nl.length * nl.width * rm.cap_per_um2)
        mod.close()

    def test_later_run_below_limit_has_no_stale_notice(self, tmp_path):
        """AC#8: a later run below the limit carries no leftover notice
        from a prior capped run."""
        from silk.layout.estimator import estimate_layout
        mod = self._make_single_net_module(tmp_path)
        for net in mod.net_dict.values():
            net.peak_current = 0.020
        first = estimate_layout(mod, self._rm())
        assert 'required_width_exceeds_max' in self._reasons(first)
        for net in mod.net_dict.values():
            net.peak_current = 0.001
        second = estimate_layout(mod, self._rm())
        assert 'required_width_exceeds_max' not in self._reasons(second)
        # The earlier estimate keeps its own notices (run-scoped, immutable).
        assert 'required_width_exceeds_max' in self._reasons(first)
        mod.close()

    def test_capture_layout_still_has_no_warnings(self, tmp_path):
        """A fresh capture_layout() keeps its empty-warnings contract even
        after a capped run populated Net state."""
        from silk.layout.estimator import estimate_layout, capture_layout
        mod = self._make_single_net_module(tmp_path)
        for net in mod.net_dict.values():
            net.peak_current = 0.020
        estimate_layout(mod, self._rm())
        snapshot = capture_layout(mod)
        assert snapshot.warnings == ()
        mod.close()


# ---------------------------------------------------------------------------
# required_width_exceeds_max — multilayer stack + escalation-rule call count
# (AC#4, AC#5)
# ---------------------------------------------------------------------------

class TestRequiredWidthExceedsMaxMultilayer:

    _make_diff_amp = TestEstimateLayout._make_diff_amp

    def _met1(self):
        from silk.layout.routing_model import RoutingModel
        return RoutingModel(sheet_resistance=0.1, cap_per_um2=0.04,
                            j_max=1e-3, min_width=0.14, max_width=1.0)

    def _met2(self):
        from silk.layout.routing_model import RoutingModel
        return RoutingModel(sheet_resistance=0.05, cap_per_um2=0.08,
                            j_max=1.2e-3, min_width=0.28, max_width=10.0)

    def test_final_layer_cap_reports_that_layer_not_first(self, pdk, tmp_path):
        """A net that escalates all the way to the top layer, and still
        exceeds ITS max_width, must be reported against the chosen (top)
        layer's j_max/max_width — not the first layer's."""
        from silk.layout.estimator import estimate_layout
        amp = self._make_diff_amp(pdk, tmp_path)
        met1, met2 = self._met1(), self._met2()
        amp.net_dict['vout_p'].peak_current = 1.0     # forces escalation and still overflows met2
        amp.net_dict['vout_n'].peak_current = 1e-6    # stays on met1, no warning
        est = estimate_layout(amp, [met1, met2])
        assert amp.net_dict['vout_p'].estimated_width == pytest.approx(met2.max_width)
        w = next(w for w in est.warnings
                 if w.reason == 'required_width_exceeds_max' and w.net_name == 'vout_p')
        assert f'{met2.j_max:.4g}' in w.detail
        assert f'{met2.max_width:.4g}' in w.detail
        assert not any(w.reason == 'required_width_exceeds_max' and w.net_name == 'vout_n'
                       for w in est.warnings)

    def test_successful_escalation_avoids_cap(self, pdk, tmp_path):
        """AC#4: escalating to a layer that CAN accommodate the current
        produces no width-cap warning."""
        from silk.layout.estimator import estimate_layout
        amp = self._make_diff_amp(pdk, tmp_path)
        met1, met2 = self._met1(), self._met2()
        # 5 mA: overflows met1 (max_width 1.0um -> 1mA cap) but fits on met2
        # (5e-3 / 1.2e-3 ~= 4.17um < 10um max_width).
        amp.net_dict['vout_p'].peak_current = 5e-3
        amp.net_dict['vout_n'].peak_current = 5e-3
        est = estimate_layout(amp, [met1, met2])
        assert amp.net_dict['vout_p'].estimated_width < met2.max_width
        assert 'required_width_exceeds_max' not in [w.reason for w in est.warnings]

    def test_escalation_rule_called_once_per_class(self, pdk, tmp_path):
        """AC#5: the warning logic must not call escalation_rule an extra
        time to re-discover the chosen layer."""
        from silk.layout.estimator import estimate_layout, default_escalation_rule
        amp = self._make_diff_amp(pdk, tmp_path)
        met1, met2 = self._met1(), self._met2()
        amp.net_dict['vout_p'].peak_current = 1.0   # triggers the cap warning path
        amp.net_dict['vout_n'].peak_current = 1e-6

        calls = []

        def counting_rule(peak_current, stack):
            calls.append(peak_current)
            return default_escalation_rule(peak_current, stack)

        est = estimate_layout(amp, [met1, met2], escalation_rule=counting_rule)
        # Two singleton classes (vout_p, vout_n unmatched) -> exactly one
        # escalation_rule call per class, same as before this task.
        assert len(calls) == len(est.nets)
        assert any(w.reason == 'required_width_exceeds_max' for w in est.warnings)


# ---------------------------------------------------------------------------
# required_width_exceeds_max — matched equivalence classes (AC#6)
# ---------------------------------------------------------------------------

class TestRequiredWidthExceedsMaxMatchedClass:

    _make_diff_amp = TestEstimateLayout._make_diff_amp

    def _rm(self):
        from silk.layout.routing_model import RoutingModel
        return RoutingModel(sheet_resistance=0.1, cap_per_um2=0.04,
                            j_max=1e-3, min_width=0.14, max_width=10.0)

    def test_one_notice_per_affected_member_not_duplicated(self, pdk, tmp_path):
        from silk.layout.estimator import estimate_layout
        amp = self._make_diff_amp(pdk, tmp_path)
        amp.declare_match_pair('diff_in', 'mn1', 'mn2')
        amp.net_dict['vout_p'].peak_current = 0.020   # exceeds max_width
        amp.net_dict['vout_n'].peak_current = 0.001   # would not, alone
        est = estimate_layout(amp, self._rm())
        cap_warnings = [w for w in est.warnings if w.reason == 'required_width_exceeds_max']
        assert {w.net_name for w in cap_warnings} == {'vout_p', 'vout_n'}
        assert len(cap_warnings) == 2   # exactly one per member, no duplicates

    def test_uses_max_across_members_and_explains_class_context(self, pdk, tmp_path):
        """AC#6: sizing uses the class-derived maximum current, and the
        detail says so rather than implying vout_n independently measured it."""
        from silk.layout.estimator import estimate_layout
        amp = self._make_diff_amp(pdk, tmp_path)
        amp.declare_match_pair('diff_in', 'mn1', 'mn2')
        amp.net_dict['vout_p'].peak_current = 0.020
        amp.net_dict['vout_n'].peak_current = 0.001
        est = estimate_layout(amp, self._rm())
        assert est.nets['vout_p'].width == pytest.approx(est.nets['vout_n'].width)
        w = next(w for w in est.warnings
                 if w.reason == 'required_width_exceeds_max' and w.net_name == 'vout_n')
        assert 'class' in w.detail.lower()
        assert '2' in w.detail   # class has 2 matched members


# ---------------------------------------------------------------------------
# estimate_layout — layer escalation (TASK-25)
# ---------------------------------------------------------------------------

class TestEstimateLayoutEscalation:

    _make_diff_amp = TestEstimateLayout._make_diff_amp
    _routing_model = TestEstimateLayout._routing_model

    def _met2(self):
        from silk.layout.routing_model import RoutingModel
        return RoutingModel(sheet_resistance=0.05, cap_per_um2=0.08,
                            j_max=1.2e-3, min_width=0.28, max_width=20.0)

    def test_high_current_net_prices_on_escalated_layer(self, pdk, tmp_path):
        from silk.layout.estimator import estimate_layout
        amp = self._make_diff_amp(pdk, tmp_path)
        amp.net_dict['vout_p'].peak_current = 2e-4   # escalates past met1, min-width on met2
        amp.net_dict['vout_n'].peak_current = 1e-6   # stays on met1
        met1, met2 = self._routing_model(), self._met2()
        estimate_layout(amp, [met1, met2])
        assert amp.net_dict['vout_p'].estimated_width == pytest.approx(met2.min_width)
        assert amp.net_dict['vout_n'].estimated_width == pytest.approx(met1.min_width)
        # Different sheet_resistance -> different R even though both nets
        # have the same statistical length.
        assert amp.net_dict['vout_p'].estimated_r != pytest.approx(
            amp.net_dict['vout_n'].estimated_r)

    def test_routing_model_id_covers_every_stack_layer(self, pdk, tmp_path):
        from silk.layout.estimator import estimate_layout
        amp = self._make_diff_amp(pdk, tmp_path)
        met1, met2 = self._routing_model(), self._met2()
        est = estimate_layout(amp, [met1, met2])
        assert met1.id in est.routing_model_id
        assert met2.id in est.routing_model_id

    def test_stack_from_pdk_output_accepted_directly(self, pdk, tmp_path):
        from silk.layout.estimator import estimate_layout
        from silk.layout.routing_model import RoutingModel
        amp = self._make_diff_amp(pdk, tmp_path)
        stack = RoutingModel.stack_from_pdk('sky130')   # {'met1': RoutingModel}
        est = estimate_layout(amp, stack)
        assert est.routing_model_id == stack['met1'].id

    def test_custom_escalation_rule_threaded_through(self, pdk, tmp_path):
        from silk.layout.estimator import estimate_layout
        amp = self._make_diff_amp(pdk, tmp_path)
        amp.net_dict['vout_p'].peak_current = 1e-6   # would stay on met1 by default
        met1, met2 = self._routing_model(), self._met2()

        def always_top(peak_current, stack):
            return stack[-1]

        estimate_layout(amp, [met1, met2], escalation_rule=always_top)
        assert amp.net_dict['vout_p'].estimated_width == pytest.approx(met2.min_width)


# ---------------------------------------------------------------------------
# capture_layout
# ---------------------------------------------------------------------------

class TestCaptureLayout:

    def _make_amp(self, pdk, tmp_path):
        from silk.circuit.modules import Module
        amp = Module('PDK_skywater130', str(tmp_path), 'amp.cir',
                     pdk_details=pdk, module_name='amp')
        amp.add(pdk.nmos1p8('mn1', d='vout', g='vin', s='vss', b='vss', w=4, l=0.18, m=1))
        return amp

    def _rm(self):
        from silk.layout.routing_model import RoutingModel
        return RoutingModel(sheet_resistance=0.1, cap_per_um2=0.04,
                            j_max=1e-3, min_width=0.14, max_width=10.0)

    def test_returns_layout_estimate(self, pdk, tmp_path):
        from silk.layout.estimator import estimate_layout, capture_layout, LayoutEstimate
        amp = self._make_amp(pdk, tmp_path)
        estimate_layout(amp, self._rm())
        snapshot = capture_layout(amp)
        assert isinstance(snapshot, LayoutEstimate)

    def test_computation_mode_is_captured(self, pdk, tmp_path):
        from silk.layout.estimator import estimate_layout, capture_layout
        amp = self._make_amp(pdk, tmp_path)
        estimate_layout(amp, self._rm())
        snapshot = capture_layout(amp)
        assert snapshot.computation_mode == 'captured'

    def test_empty_warnings(self, pdk, tmp_path):
        """capture_layout never emits warnings — they belong to the run."""
        from silk.layout.estimator import estimate_layout, capture_layout
        amp = self._make_amp(pdk, tmp_path)
        estimate_layout(amp, self._rm())
        snapshot = capture_layout(amp)
        assert snapshot.warnings == ()

    def test_scalars_match_net_state(self, pdk, tmp_path):
        """capture_layout reads current Net state, not recomputing."""
        from silk.layout.estimator import estimate_layout, capture_layout
        amp = self._make_amp(pdk, tmp_path)
        estimate_layout(amp, self._rm())
        # Manually override to a known value
        amp.net_dict['vout'].estimated_r = 99.9
        snapshot = capture_layout(amp)
        assert snapshot.nets['vout'].r == pytest.approx(99.9)
