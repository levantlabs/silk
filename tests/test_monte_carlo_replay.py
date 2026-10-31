"""Replay must refuse a different managed experiment before touching its files."""
from dataclasses import replace
from pathlib import Path

import pytest

from silk import sources
from silk.circuit.modules import Module
from silk.exceptions import MonteCarloError
from tests.test_monte_carlo import events, fake_sim, sky130_tb  # noqa: F401


def _files(root):
    return {str(p.relative_to(root)): p.read_bytes()
            for p in Path(root).rglob('*') if p.is_file()}


def _refuses(mc, labels, root, key=('tt', 0)):
    before = _files(root)
    with pytest.raises(MonteCarloError) as caught:
        mc.rerun(key)
    message = str(caught.value).lower()
    assert 'changed' in message
    for label in labels:
        assert label.lower() in message
    assert _files(root) == before


def test_rerun_refuses_temperature_and_bias_changes(sky130_tb, fake_sim, tmp_path):
    tb = sky130_tb
    tb.set_temperature(25)
    mc = tb.run_monte_carlo(1)
    digest = tb._structure_hash()
    tb.set_temperature(125)
    tb.netlist.sig_sources_dict['vg'].dc = 2
    # The combined DUT+bench structure component now includes bench sources;
    # replay's independent source and temperature guards still refuse first.
    assert tb._structure_hash() != digest
    _refuses(mc, ['temperature', 'source', 'vg'], tmp_path)


@pytest.mark.parametrize('change,label', [
    ('temperature', 'temperature'), ('source_dc', 'source'),
    ('source_ac', 'source'), ('source_waveform', 'source'),
    ('source_wiring', 'source'), ('analysis', 'analysis'),
    ('supply', 'supply'), ('supply_source', 'supply'),
    ('simulator', 'backend'), ('option', 'reltol'),
    ('option_add', 'abstol'), ('option_remove', 'reltol'),
    ('mc_n', 'n'), ('mc_seed', 'seed'), ('mc_variation', 'variation'),
    ('mc_record_draws', 'record_draws'), ('mc_correlate', 'correlate'),
    ('mc_min_yield', 'min_yield'), ('mc_clear', 'monte carlo'),
    ('corners', 'corner'), ('device', 'device'),
    ('mismatch', 'mismatch'), ('custom', 'custom'),
    ('library', 'library'), ('section', 'statistical'),
])
def test_rerun_refuses_each_experiment_input(
        sky130_tb, fake_sim, tmp_path, change, label):
    tb = sky130_tb
    tb.set_temperature(25)
    tb.set_simulator_options(reltol=1e-3)
    tb.set_supply_voltage(1, source='vd')
    tb.set_monte_carlo(2)
    if change == 'source_waveform':
        tb.add_source(sources.VoltageSource(
            'wave', 'wave', '0', pulse=sources.Pulse(
                0, 1, tr=1e-10, tf=1e-10, pw=1e-9, per=2e-9)))
    mc = tb.run()
    src = tb.netlist.sig_sources_dict['wave' if change == 'source_waveform' else 'vg']
    if change == 'temperature':
        tb.set_temperature(125)
    elif change == 'source_dc':
        src.dc = 2
    elif change == 'source_ac':
        src.ac = 2
    elif change == 'source_waveform':
        src.pulse.v2 = 2
    elif change == 'source_wiring':
        src.positive = 'vout'
    elif change == 'analysis':
        next(iter(tb.netlist.sim_analysis_dict.values())).stop = 3e-9
    elif change == 'supply':
        tb.set_supply_voltage(2, source='vd')
    elif change == 'supply_source':
        tb.set_supply_voltage(1, source='vg')
    elif change == 'simulator':
        tb.netlist.simulator = 'xyce'
    elif change == 'option':
        tb.set_simulator_options(reltol=1e-4)
    elif change == 'option_add':
        tb.set_simulator_options(abstol=1e-10)
    elif change == 'option_remove':
        tb.netlist.simulator_options.pop('reltol')
        tb.netlist._explicit_simulator_options.pop('reltol')
    elif change == 'mc_clear':
        tb.set_monte_carlo(None)
    elif change.startswith('mc_'):
        field = change[3:]
        value = dict(n=3, seed=5, variation='both', record_draws=True,
                     correlate=(tb.probe('vout'),), min_yield=0.8)[field]
        tb._monte_carlo = replace(tb.monte_carlo, **{field: value})
    elif change == 'corners':
        tb.set_corners(['ss'])
    elif change == 'device':
        tb.module.elem_dict['M1']['device'].params['w'] = 2
    elif change == 'mismatch':
        tb.module.set_mismatch(False)
    elif change == 'custom':
        tb.netlist.custom_lines.append('.ic v(vout)=0.1')
    elif change == 'library':
        tb.pdk_details.library_location = '/different/model.lib'
    elif change == 'section':
        tb.pdk_details.statistical.mismatch_sections['tt'] = 'changed_mm'
    _refuses(mc, [label], tmp_path)


def test_rerun_refusal_preserves_sample_files(sky130_tb, fake_sim, tmp_path):
    mc = sky130_tb.run_monte_carlo(1, variation='both')
    wrappers = list(tmp_path.rglob('mc_both.lib.spice'))
    assert wrappers
    wrappers[0].write_text('do not overwrite this evidence')
    sky130_tb.set_temperature(125)
    _refuses(mc, ['temperature'], tmp_path)


def test_signature_is_captured_before_preparation(sky130_tb, fake_sim, monkeypatch, tmp_path):
    tb = sky130_tb
    prepare = tb._prepare_plan

    def change(*args, **kwargs):
        tb.netlist.sig_sources_dict['vg'].dc = 2
        return prepare(*args, **kwargs)

    monkeypatch.setattr(tb, '_prepare_plan', change)
    mc = tb.run_monte_carlo(1)
    _refuses(mc, ['source', 'vg'], tmp_path)


@pytest.mark.parametrize('route', ['one_off', 'declared', 'selected', 'serial', 'parallel', 'corner'])
def test_rerun_guard_covers_campaign_entry_points(sky130_tb, fake_sim, tmp_path, route):
    tb = sky130_tb
    top = Module('sky130', str(tmp_path), 'top.cir', tb.pdk_details)
    top.add_module(tb.module, 'child', d='d', g='g', s='0')
    tb.set_monte_carlo(1)
    if route == 'one_off':
        mc = tb.run_monte_carlo(1)
    elif route == 'selected':
        mc = top.get_test(tb.test_name, instance='child').run()
    elif route in ('serial', 'parallel'):
        mc = next(iter(top.run_all_tests(n_workers=1 if route == 'serial' else 2).values()))
    else:
        mc = tb.run()
    if route == 'corner':
        mc = mc.corner('tt')
    tb.set_temperature(125)
    _refuses(mc, ['temperature'], tmp_path)
    top.close()


@pytest.mark.parametrize('field', ['temperature', 'supply', 'options'])
def test_rerun_uses_live_inherited_inputs(sky130_tb, fake_sim, tmp_path, field):
    tb = sky130_tb
    top = Module('sky130', str(tmp_path), 'top.cir', tb.pdk_details)
    top.add_module(tb.module, 'child', d='d', g='g', s='0')
    owner = tb.module if field == 'options' else top
    setter = {'temperature': lambda v: owner.set_temperature(v),
              'supply': lambda v: owner.set_supply_voltage(v, source='vd'),
              'options': lambda v: owner.set_simulator_options(reltol=v)}[field]
    setter(1)
    mc = top.get_test(tb.test_name, instance='child').run_monte_carlo(1)
    assert mc.rerun(('tt', 0)).metrics == mc.reports[('tt', 0)].metrics
    setter(2)
    _refuses(mc, ['reltol' if field == 'options' else field], tmp_path)
    top.close()


def test_rerun_refuses_removed_placement(sky130_tb, fake_sim, tmp_path):
    tb = sky130_tb
    top = Module('sky130', str(tmp_path), 'top.cir', tb.pdk_details)
    top.add_module(tb.module, 'child', d='d', g='g', s='0')
    mc = top.get_test(tb.test_name, instance='child').run_monte_carlo(1)
    tb.module._tests.pop(tb.test_name)
    _refuses(mc, ['placement'], tmp_path)
    top.close()


def test_rerun_one_off_settings_and_preparation_are_stable(sky130_tb, fake_sim):
    tb = sky130_tb
    tb.set_monte_carlo(3, seed=9)
    tb.module.set_temperature(25)
    tb.module.set_supply_voltage(1.8, source='vd')
    tb.module.set_simulator_options(reltol=1e-5)
    mc = tb.run_monte_carlo(1, seed=42)
    for _ in range(2):
        assert mc.rerun(('tt', 0)).metrics == mc.reports[('tt', 0)].metrics
    assert tb.monte_carlo.seed == 9


def test_rerun_refuses_changed_seed_mapping(sky130_tb, fake_sim, tmp_path):
    mc = sky130_tb.run_monte_carlo(1)
    mc.seeds[('tt', 0)] += 1
    _refuses(mc, ['seed'], tmp_path)


def test_failed_sample_replay_checks_inputs(sky130_tb, fake_sim, tmp_path):
    fake_sim.fail_on(('tt', 0))
    mc = sky130_tb.run_monte_carlo(2)
    assert ('tt', 0) in mc.failed
    sky130_tb.set_temperature(125)
    _refuses(mc, ['temperature'], tmp_path)


@pytest.mark.parametrize('nested', [False, True])
def test_rerun_refuses_changed_controlled_or_nested_source(sky130_tb, fake_sim, tmp_path, nested):
    tb = sky130_tb
    owner = tb.module if nested else tb.netlist
    source = sources.VCVS('amplifier', 'aux', '0', 'g' if nested else 'vg', '0', 2)
    owner.add_source(source)
    mc = tb.run_monte_carlo(1)
    source.gain = 3
    _refuses(mc, ['source', 'amplifier'], tmp_path)


def test_rerun_changed_invalid_input_names_field(sky130_tb, fake_sim, tmp_path):
    mc = sky130_tb.run_monte_carlo(1)
    next(iter(sky130_tb.netlist.sim_analysis_dict.values())).stop = -1
    _refuses(mc, ['analysis'], tmp_path)


def test_rerun_refuses_changed_simulator_executable(sky130_tb, fake_sim, tmp_path, monkeypatch):
    mc = sky130_tb.run_monte_carlo(1)
    monkeypatch.setenv('SILK_NGSPICE_BIN', '/different/ngspice')
    _refuses(mc, ['simulator executable'], tmp_path)


def test_rerun_evaluates_current_metrics_and_specs(sky130_tb, fake_sim):
    tb = sky130_tb
    mc = tb.run_monte_carlo(1)
    tb.add_metric('new_metric', lambda ckt: 42.0)
    tb.add_spec('new_spec', 'new_metric', '>', 40)
    replay = mc.rerun(('tt', 0))
    assert replay.metrics['new_metric'] == 42.0
    assert 'new_metric' not in mc.reports[('tt', 0)].metrics
    assert replay.passed


def test_rerun_local_overrides_mask_changed_inherited_settings(sky130_tb, fake_sim):
    tb = sky130_tb
    tb.set_temperature(25)
    tb.set_supply_voltage(1, source='vd')
    tb.set_simulator_options(reltol=1e-3)
    mc = tb.run_monte_carlo(1)
    tb.module.set_temperature(125)
    tb.module.set_supply_voltage(2, source='vg')
    tb.module.set_simulator_options(reltol=1e-5)
    assert mc.rerun(('tt', 0)).metrics == mc.reports[('tt', 0)].metrics


@pytest.mark.parametrize('field', ['save_device_currents', 'terminal_current_method',
                                    'save_device_params', 'omit_device_internal_nodes', 'save_list'])
def test_rerun_refuses_changed_capture_configuration(sky130_tb, fake_sim, tmp_path, field):
    tb = sky130_tb
    if field == 'terminal_current_method':
        tb.module.save_device_currents = True
        tb.module.terminal_current_method = 'savecurrents'
    mc = tb.run_monte_carlo(1)
    if field == 'save_list':
        tb.netlist.set_save_list(['vout'])
    elif field == 'omit_device_internal_nodes':
        tb.module.omit_device_internal_nodes = True
    else:
        setattr(tb.module, field, 'ammeter_wrapper' if field == 'terminal_current_method' else True)
    _refuses(mc, [field], tmp_path)


def test_rerun_refuses_replaced_statistical_configuration(sky130_tb, fake_sim, tmp_path):
    from copy import copy
    tb = sky130_tb
    mc = tb.run_monte_carlo(1, variation='both')
    # The wrapper writer consumes tb.pdk_details, independently of the
    # netlist/DUT PDK references whose device structure remains unchanged.
    tb.pdk_details = copy(tb.pdk_details)
    tb.pdk_details.statistical = replace(
        tb.pdk_details.statistical, process_includes=('/different/process.spice',))
    _refuses(mc, ['statistical'], tmp_path)


def test_rerun_refuses_removed_netlist(sky130_tb, fake_sim, tmp_path):
    mc = sky130_tb.run_monte_carlo(1)
    sky130_tb.netlist = None
    _refuses(mc, ['netlist'], tmp_path)


def test_rerun_source_paths_cannot_alias(sky130_tb, fake_sim, tmp_path):
    tb = sky130_tb
    tb.module.add_source(sources.vdc('bias', positive='g', negative='s', value=1))
    from silk.exceptions import CircuitError
    # A local name can no longer impersonate the hierarchy path dut/bias.
    with pytest.raises(CircuitError, match='invalid instance name'):
        tb.netlist.add_source(sources.vdc('dut/bias', positive='vg', negative='0', value=1))
    assert 'dut/bias' not in tb.netlist.sig_sources_dict
    tb.netlist.add_source(sources.vdc('dut_bias', positive='vg', negative='0', value=1))
    mc = tb.run_monte_carlo(1)
    tb.netlist.sig_sources_dict['dut_bias'].dc = 2
    _refuses(mc, ['source', 'bias'], tmp_path)


@pytest.fixture
def legacy_tb(sky130_tb):
    tb = sky130_tb
    with pytest.warns(DeprecationWarning):
        tb.module.add_element(
            tb.pdk_details.nmos1p8, 'legacy',
            nets={'d': 'd', 'g': 'g', 's': 's', 'b': 's'},
            parameters={'w': 1.0, 'l': 0.5})
    assert 'XMlegacy ' in Path(tb.prepare()).read_text()
    return tb


def test_legacy_device_campaign_and_rerun_succeed(legacy_tb, fake_sim):
    mc = legacy_tb.run_monte_carlo(1)
    original_deck = fake_sim.deck_text(('tt', 0))
    assert 'XMlegacy ' in original_deck
    report = mc.rerun(('tt', 0))
    assert report.metrics == mc.reports[('tt', 0)].metrics
    assert fake_sim.deck_text(('tt', 0)) == original_deck


@pytest.mark.parametrize('change', ['parameter', 'connection'])
def test_legacy_device_changes_refuse_before_writes(
        legacy_tb, fake_sim, tmp_path, change):
    mc = legacy_tb.run_monte_carlo(1)
    device = legacy_tb.module.elem_dict['legacy']['device']
    if change == 'parameter':
        device.param_list['w'].value = 2.0
    else:
        device.pin_list['g'].net = 'd'
    _refuses(mc, ['device', 'legacy'], tmp_path)


@pytest.mark.parametrize('nested', [False, True])
def test_formal_port_order_change_refuses_before_writes(
        sky130_tb, fake_sim, tmp_path, monkeypatch, nested):
    tb = sky130_tb
    top = None
    selected = tb
    if nested:
        top = Module('sky130', str(tmp_path / 'top'), 'top.cir', tb.pdk_details)
        top.add_module(tb.module, 'child', d='d', g='g', s='0')
        selected = top.get_test(tb.test_name, instance='child')
    try:
        mc = selected.run_monte_carlo(1, variation='both')
        assert mc.rerun(('tt', 0)).metrics == mc.reports[('tt', 0)].metrics
        digest = tb._structure_hash()
        tb.module.pin_list = dict(reversed(list(tb.module.pin_list.items())))
        assert tb._structure_hash() == digest

        def forbid_write(*args, **kwargs):
            pytest.fail('changed formal ports reached replay file preparation')

        monkeypatch.setattr('silk.circuit.module_test._write_wrapper', forbid_write)
        monkeypatch.setattr(tb, '_prepare_run', forbid_write)
        _refuses(mc, ['formal ports'], tmp_path)
    finally:
        if top is not None:
            top.close()
