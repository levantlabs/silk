"""Integration: the persistence payoff — currents on a fresh build, no re-sim."""
import pytest

from tests._env import needs_ngspice, needs_pdk
from tests.test_net_current_store import (
    _build_dut, _build_tb, _annotate_definition, _fake_report)


def test_fresh_build_has_currents_without_simulation(pdk, tmp_dir, tmp_path):
    """The backlog's blocked-consumer scenario: store in session 1, load in
    session 2, layout consumers read Net.peak_current with zero SPICE."""
    from silk.layout import RoutingModel, estimate_layout

    routing = RoutingModel(
        sheet_resistance=0.1,
        cap_per_um2=0.04,
        j_max=0.5e-3,
        min_width=0.14,
        max_width=10.0,
    )
    db_path = str(tmp_path / 'chip.db')
    tb = _build_tb(pdk, tmp_dir, db_path=db_path)
    measured_dut = _annotate_definition(tb)
    tb.store_net_currents(_fake_report(), corner='tt')
    measured_layout = estimate_layout(measured_dut, routing)
    del tb                                              # "session 2"
    fresh = _build_dut(pdk, tmp_dir)
    assert fresh.net_dict['mid'].peak_current is None   # nothing yet
    fresh.load_net_currents('tt', db_path=db_path)
    assert fresh.net_dict['mid'].peak_current == pytest.approx(1e-3)
    assert fresh.net_dict['mid']._current_source == 'db'

    loaded_layout = estimate_layout(fresh, routing)
    assert loaded_layout.nets['mid'].width == pytest.approx(
        measured_layout.nets['mid'].width
    )
    assert loaded_layout.nets['mid'].width == pytest.approx(2.0)
    assert fresh.net_dict['A'].peak_current is None
    assert loaded_layout.nets['A'].width == pytest.approx(routing.min_width)


def test_simulate_records_last_corner(pdk, tmp_dir, mock_ngspice):
    """simulate() records the resolved corner so store callers can pass it."""
    from silk import sources
    from silk.simulation import Transient
    from silk.circuit.modules import Module
    tb = Module('PDK_skywater130', tmp_dir, 'c.cir', pdk_details=pdk)
    tb.add_net('a')
    tb.add_source(sources.VoltageSource('v1', positive='a', negative='0', dc=1.0))
    from silk.elements import elements
    tb.add(elements.resistor('r1', p='a', n='0', value=1000.0))
    tb.add_sim_type(Transient(step=1e-9, stop=1e-8))
    tb.simulate(corner='tt')
    # An ideal deck: no corner reached the simulator (TASK-406). The
    # evidence is ConditionState.NONE; the requested corner is the label.
    from silk.simulation import ConditionState
    assert getattr(tb, '_last_sim_corner', None) is ConditionState.NONE
    assert tb._last_sim_label == 'tt'


def test_simulate_default_corner_records_resolved_value(pdk, tmp_dir, mock_ngspice):
    """corner=None (the default) must record the RESOLVED corner
    (pdk.default_corner, 'tt' here) in _last_sim_corner — never None — so a
    store_net_currents(corner=tb._last_sim_corner) call downstream never sees
    a None corner."""
    from silk import sources
    from silk.simulation import Transient
    from silk.circuit.modules import Module
    tb = Module('PDK_skywater130', tmp_dir, 'c.cir', pdk_details=pdk)
    tb.add_net('a')
    tb.add_source(sources.VoltageSource('v1', positive='a', negative='0', dc=1.0))
    from silk.elements import elements
    tb.add(elements.resistor('r1', p='a', n='0', value=1000.0))
    tb.add_sim_type(Transient(step=1e-9, stop=1e-8))
    tb.simulate(corner=None)
    # Never None: an ideal deck records ConditionState.NONE (TASK-406), and
    # the resolved default stays the run's label.
    from silk.simulation import ConditionState
    assert getattr(tb, '_last_sim_corner', None) is ConditionState.NONE
    assert tb._last_sim_label == 'tt'


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_end_to_end_em_driver_round_trip(tmp_path, monkeypatch):
    """Real currents survive storage, including rail pin sizing inputs."""
    from silk import sources, load_pdk
    from silk.circuit.modules import Module
    from silk.examples.net_current_persistence_example import build_current_sink
    from silk.simulation import Transient

    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    db_path = str(tmp_path / 'em.db')
    tb = Module('sky130', str(tmp_path), 'persist.cir',
                pdk_details=load_pdk('sky130'))
    tb.add_module(build_current_sink(), 'dut', nets={'VDD': 'vdd', 'VSS': '0'})
    tb.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
    tb.add_sim_type(Transient(step=5e-9, stop=2e-6))
    tb.save_device_currents = True
    tb.terminal_current_method = 'ammeter_wrapper'
    tb.simulate(corner='tt')
    report = tb.annotate_currents(tstart=1e-6)
    dut = tb.elem_dict['dut']['device']
    tb.store_net_currents(report, corner='tt', db_path=db_path)

    fresh = build_current_sink()
    assert fresh.results_file is None
    assert all(net.avg_current is None for net in fresh.net_dict.values())
    fresh.load_net_currents('tt', db_path=db_path)
    assert fresh.results_file is None
    for name in ('out', 'vbias'):
        measured, restored = dut.net_dict[name], fresh.net_dict[name]
        assert measured.avg_current > 0
        assert restored._current_source == 'db'
        for field in ('avg_current', 'peak_current', 'rms_current'):
            assert getattr(restored, field) == pytest.approx(getattr(measured, field))
    assert fresh.net_dict['out'].avg_current > 1e-3

    for name in ('VDD', 'VSS'):
        measured, restored = dut.net_dict[name], fresh.net_dict[name]
        # Interface aggregates stay withheld; sizing requires every pin.
        assert measured.avg_current is None and restored.avg_current is None
        assert restored._current_source is None
        assert len(restored.pins) == len(measured.pins) > 0
        for live_pin, db_pin in zip(measured.pins, restored.pins):
            assert live_pin.avg_current is not None
            assert db_pin._current_source == 'db'
            assert db_pin.avg_current == pytest.approx(live_pin.avg_current)
        # Conservative rail bound: half the absolute pin sum plus the
        # absolute KCL boundary current. Compare the actual sizing inputs.
        bounds = []
        for net in (measured, restored):
            boundary = -sum(pin.avg_current for pin in net.pins)
            bounds.append(0.5 * (sum(abs(pin.avg_current) for pin in net.pins)
                                 + abs(boundary)))
        assert bounds[0] > 1e-3
        assert bounds[1] == pytest.approx(bounds[0])
        assert max(bounds[1] / 0.78e-3, 0.14) == pytest.approx(
            max(bounds[0] / 0.78e-3, 0.14))
