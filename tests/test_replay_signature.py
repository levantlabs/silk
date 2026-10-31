"""The backend-neutral replay projection uses effective inputs without writes."""
import pytest

from silk.circuit.monte_carlo import MonteCarloSettings
from silk.circuit.variable import Variable
from silk.simulation._replay import experiment_signature, changed_inputs
from tests.test_monte_carlo import _bench


def _model(ports, t, params):
    ports.d.v = ports.g.v * params['gain']


def _enable_model(tb, backend='behavioral'):
    tb.module.set_behavioral_model(_model, params={'gain': 2}, execution='vector')
    tb.module.simulator = backend
    if backend == 'behavioral':
        # The transistor rig clamps its drain for SPICE; the modeled OUT
        # owns that slot in behavioral mode. Keep a separate bias source so
        # replay can still track every source without conflicting writers.
        tb.netlist.sig_sources_dict['vd'].positive = 'drain_bias'


def _signature(tb):
    result = experiment_signature(tb, MonteCarloSettings(1), {})
    result.require_valid()
    return result


@pytest.mark.parametrize('backend', ['ngspice', 'behavioral'])
@pytest.mark.parametrize('change,label', [('source', 'source $/vg'),
                                         ('analysis', 'analysis transient'),
                                         ('options', 'simulator option reltol'),
                                         ('supply', 'supply')])
def test_common_inputs_have_backend_independent_labels(tmp_path, backend, change, label):
    tb = _bench(tmp_path, 'sky130', 'nmos1p8')
    _enable_model(tb, backend)
    before = _signature(tb)
    if change == 'source':
        tb.netlist.sig_sources_dict['vg'].dc = 2
    elif change == 'analysis':
        next(iter(tb.netlist.sim_analysis_dict.values())).stop = 5e-9
    elif change == 'options':
        tb.set_simulator_options(reltol=1e-5)
    else:
        tb.module.set_supply_voltage(1.8, source='vd')
    assert label in changed_inputs(before, _signature(tb))


def test_effective_supply_projection_is_non_mutating(tmp_path):
    tb = _bench(tmp_path, 'sky130', 'nmos1p8')
    tb.module.set_supply_voltage(1.8, source='vd')
    tb.module.set_temperature(25)
    tb.module.set_simulator_options(reltol=1e-5)
    netlist = tb.netlist
    original = (netlist.sig_sources_dict['vd'].dc, dict(netlist.simulator_options),
                netlist._supply_voltage, netlist._supply_source, netlist._simulator)
    before = _signature(tb)
    assert original == (netlist.sig_sources_dict['vd'].dc, netlist.simulator_options,
                        netlist._supply_voltage, netlist._supply_source, netlist._simulator)
    tb._prepare_run('tt')
    assert changed_inputs(before, _signature(tb)) == ()


def test_signature_freezes_mutable_values(tmp_path):
    tb = _bench(tmp_path, 'sky130', 'nmos1p8')
    bias = Variable('bias', value=1, bounds=(0, 3))
    tb.netlist.sig_sources_dict['vg'].dc = bias
    before = _signature(tb)
    bias.value = 2
    tb.pdk_details.extra_includes.append('extra.lib')
    changes = changed_inputs(before, _signature(tb))
    assert 'source $/vg' in changes
    assert any('extra_includes' in label for label in changes)


def test_behavioral_contributor_tracks_models_but_stops_at_modeled_leaves(tmp_path):
    tb = _bench(tmp_path, 'sky130', 'nmos1p8')
    _enable_model(tb)
    before = _signature(tb)
    tb.module.elem_dict['M1']['device'].params['w'] = 3
    assert changed_inputs(before, _signature(tb)) == ()
    tb.module._behavioral_params['gain'] = 4
    assert any(label.startswith('model ') for label in changed_inputs(before, _signature(tb)))


def test_invalid_source_does_not_hide_other_changed_inputs(tmp_path):
    tb = _bench(tmp_path, 'sky130', 'nmos1p8')
    before = _signature(tb)
    tb.netlist.sig_sources_dict['vg'].dc = Variable('unset', value=None)
    tb.set_temperature(125)
    current = experiment_signature(tb, MonteCarloSettings(1), {})
    changes = changed_inputs(before, current)
    assert 'temperature' in changes
    assert any('source $/vg' in label for label in changes)
    assert current.invalid


@pytest.mark.parametrize('change,label', [('source', 'source $/vg'),
                                         ('analysis', 'analysis transient')])
def test_signature_preserves_behavioral_numeric_precision(tmp_path, change, label):
    tb = _bench(tmp_path, 'sky130', 'nmos1p8')
    _enable_model(tb)
    original = _signature(tb)
    value = (tb.netlist.sig_sources_dict['vg'] if change == 'source'
             else next(iter(tb.netlist.sim_analysis_dict.values())))
    text = value.to_spice()
    if change == 'source':
        value.dc = 1.0000000000001
    else:
        value.step = 1.0000000000001e-9
    assert value.to_spice() == text  # SPICE formatting is deliberately lossy.
    assert label in changed_inputs(original, _signature(tb))


def test_behavioral_draw_scheme_is_a_replay_input(pdk, tmp_path, monkeypatch):
    from silk.simulation import variation
    from tests._variation_bench import chip
    _, tb = chip(pdk, tmp_path, {'offset': variation.Normal(0, 0.001)})
    before = _signature(tb)
    monkeypatch.setattr(variation, 'DRAW_SCHEME', variation.DRAW_SCHEME + 1)
    assert 'draw scheme' in changed_inputs(before, _signature(tb))


def test_behavioral_declaration_records_include_process_identity(pdk, tmp_path):
    from silk.simulation.variation import Normal
    from tests._variation_bench import chip
    top, tb = chip(pdk, tmp_path, {'offset': Normal(0, 0.001)})
    before = _signature(tb)
    top.elem_dict['c0']['device']._behavioral_params['offset'] = Normal(
        0, 0.001, variation='process', name='shift')
    changes = changed_inputs(before, _signature(tb))
    assert any(label.startswith('declaration ') for label in changes)


def test_campaign_output_mode_does_not_change_signature(tmp_path):
    tb = _bench(tmp_path, 'sky130', 'nmos1p8')
    tb.set_monte_carlo(2)
    before = experiment_signature(tb, tb.monte_carlo, {})
    before.require_valid()
    tb.set_monte_carlo(2, output_mode='new')
    after = experiment_signature(tb, tb.monte_carlo, {})
    after.require_valid()
    assert changed_inputs(before, after) == ()
