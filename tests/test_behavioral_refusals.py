"""Behavioral runs refuse SPICE faults and inactive statistical campaigns."""
import pytest

from silk import load_pdk
from silk.circuit.modules import Module
from silk.exceptions import MonteCarloError, NotSupportedError, SimulationError
from silk.faults import FaultCampaign, FaultSet
from silk.simulation import Transient
from silk.sources import vdc


def _module(tmp_path):
    mod = Module('sky130', str(tmp_path / 'module'), 'test.cir', load_pdk('sky130'))
    mod.add_sim_type(Transient(step=1e-9, stop=3e-9))
    return mod


@pytest.mark.parametrize('analysis', [True, False])
def test_fault_campaign_refuses_behavioral_at_construction(tmp_path, analysis):
    mod = _module(tmp_path)
    mod.simulator = 'behavioral'
    if not analysis:
        mod.sim_analysis_dict.clear()
    output = tmp_path / 'campaign'
    with pytest.raises(NotSupportedError, match='behavioral.*backend'):
        FaultCampaign(mod, FaultSet(), 1, out_dir=str(output))
    assert not output.exists()


def test_fault_campaign_rechecks_backend_before_execution(tmp_path):
    mod = _module(tmp_path)
    output = tmp_path / 'campaign'
    campaign = FaultCampaign(mod, FaultSet(), 1, out_dir=str(output))
    mod.simulator = 'behavioral'
    with pytest.raises(NotSupportedError, match='behavioral.*backend'):
        campaign.run()
    assert not output.exists()


@pytest.mark.parametrize('selection', ['owner', 'test'])
def test_monte_carlo_requires_active_declaration(pdk, tmp_path, selection):
    from tests._variation_bench import chip
    mod, tb = chip(pdk, tmp_path, {'offset': 0.0})
    target = mod if selection == 'owner' else tb.netlist
    target.simulator = 'behavioral'
    assert tb.set_monte_carlo(2) is tb
    with pytest.raises(MonteCarloError, match='no active'):
        tb.run()
    with pytest.raises(MonteCarloError, match='no active'):
        tb.run_monte_carlo(2)
    assert not list(tmp_path.rglob('*.raw'))


def test_declared_monte_carlo_rechecks_backend_and_can_be_cleared(pdk, tmp_path):
    from tests._variation_bench import chip
    mod, tb = chip(pdk, tmp_path, {'offset': 0.0})
    mod.simulator = 'ngspice'
    tb.set_monte_carlo(2)
    mod.simulator = 'behavioral'
    with pytest.raises(MonteCarloError, match='no active'):
        tb.run()
    assert tb.monte_carlo.n == 2
    assert tb.set_monte_carlo(None) is tb
    assert tb.monte_carlo is None


@pytest.mark.parametrize('entry', ['simulate', 'simulate_behavioral'])
def test_current_annotation_keeps_simulation_error_with_reason(tmp_path, entry):
    mod = _module(tmp_path)
    mod.add(vdc('vin', positive='out', negative='0', value=1.0))
    if entry == 'simulate':
        mod.simulator = 'behavioral'
        mod.simulate()
    else:
        mod.simulate_behavioral(t_step=1e-9, t_stop=3e-9)
    with pytest.raises(SimulationError, match='behavioral.*backend.*electrical solve'):
        mod.annotate_currents()
