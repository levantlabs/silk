"""Integration smoke test for fault injection.

Runs a real 3-run SPICE campaign on the flash ADC. Excluded from default
pytest runs — enable via ``pytest -m slow``. Requires ngspice on PATH
and the sky130 library on disk (resolved via
``silk.examples.flash_adc_example._pdk``).
"""

import os
import sys

import pytest


@pytest.mark.slow
def test_flash_adc_fault_campaign_smoke(tmp_path):
    from silk.faults import FaultCampaign, FaultSet, stuck_open

    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..',
        'silk', 'examples'))
    from flash_adc_example import make_flash_adc, T_STEP, CLK_PER  # noqa: E402
    from silk.simulation.analysis import Transient

    adc, _ = make_flash_adc()
    # make_flash_adc() returns a bare module — the analysis lives in the
    # example's __main__ block. Without one, every fault run emits a netlist
    # with no .tran and ngspice aborts with no data rows (this test passed
    # silently for months pre-abort-detection while simulating nothing).
    # 4 clock cycles is enough to exercise the fault and keep 3 runs quick.
    adc.add_sim_type(Transient(step=T_STEP, stop=4 * CLK_PER))

    fs = FaultSet([stuck_open('xcomp1.mn1')])

    campaign = FaultCampaign(
        adc, fs, n_runs=3, seed=0,
        out_dir=str(tmp_path / 'campaign'),
        n_workers=1,
        archive_waveforms=False,
        reconcile_specs=False,
        progress=False,
    )
    result = campaign.run()

    assert len(result.runs) == 3
    errored = [r for r in result.runs if r.status == 'errored']
    assert not errored, f'Errored runs: {[r.error for r in errored]}'
    # reconcile_specs=False executes only; nothing is verified.
    assert {r.status for r in result.runs} == {'simulated'}

    for r in result.runs:
        assert len(r.manifest.applied) == 1
        assert any(a.fault_type == 'stuck_open'
                   and a.target == 'xcomp1.mn1'
                   for a in r.manifest.applied), (
                   f'Expected stuck_open on xcomp1.mn1 in run {r.run_id}, '
                   f'got {[(a.fault_type, a.target) for a in r.manifest.applied]}')

    report = result.generate_report()
    report.save(str(tmp_path / 'campaign.md'))
    md = (tmp_path / 'campaign.md').read_text()
    assert md.startswith('# Campaign')


@pytest.mark.slow
def test_shared_placement_campaign_isolates_left_and_preserves_probe_paths(tmp_path):
    from pathlib import Path
    import numpy as np
    from tests._env import HAVE_NGSPICE, HAVE_PDK
    if not HAVE_NGSPICE or not HAVE_PDK:
        pytest.skip('needs ngspice and installed sky130 models')
    from silk import Module, load_pdk, sources
    from silk.elements import elements
    from silk.faults import FaultCampaign, FaultSet, stuck_open
    from silk.metrics import final_value
    from silk.simulation import Transient

    pdk = load_pdk('sky130')
    leaf = Module('sky130', str(tmp_path), 'pull.cir', pdk_details=pdk, module_name='pull')
    leaf.define_module_ports({'out': 'out', 'vdd': 'power', 'vg': 'in', 'vss': 'ground'})
    leaf.add(elements.resistor('load', p='vdd', n='out', value=1e4))
    leaf.add(pdk.nmos1p8('mn1', d='out', g='vg', s='vss', b='vss', w=4, l=.5))
    leaf.add(elements.resistor('sense', p='out', n='tap', value=1e3))
    leaf.add(elements.capacitor('sense_load', p='tap', n='vss', value=1e-15))
    pair = Module('sky130', str(tmp_path), 'pair.cir', pdk_details=pdk, module_name='pair')
    pair.define_module_ports({'left': 'out', 'right': 'out', 'vdd': 'power',
                              'vg': 'in', 'vss': 'ground'})
    for side in ('left', 'right'):
        pair.add_module(leaf, side, nets={'out': side, 'vdd': 'vdd', 'vg': 'vg', 'vss': 'vss'})
    tb = pair.create_test('levels')
    tb.add_module_under_test({'left': 'left', 'right': 'right', 'vdd': 'vdd', 'vg': 'vg', 'vss': '0'})
    for name in ('vdd', 'vg'):
        tb.add_source(sources.VoltageSource(name, positive=name, negative='0', dc=1.8))
    tb.add_sim_type(Transient(step=1e-9, stop=20e-9))
    tb.netlist.set_save_list(signals=['left', 'right', 'xdut.xleft.tap', 'xdut.xright.tap'])
    for side in ('left', 'right'):
        tb.add_metric(side, final_value, sig=tb.probe(side))
        tb.add_spec(side + ' low', side, '<', .9)
        tb.add_metric(side + '_tap', final_value, sig=tb.probe('dut.' + side + '.tap'))
        tb.add_spec(side + ' tap low', side + '_tap', '<', .9)
    nominal = tb.run()
    assert nominal.passed
    original = tb.probe('dut.left.tap').snapshot()
    report = FaultCampaign(
        tb.netlist, FaultSet([stuck_open('xdut.xleft.mn1')]), n_runs=1, seed=7,
        out_dir=str(tmp_path / 'campaign'), archive_waveforms=True,
        n_workers=1, progress=False).run()
    run = report.runs[0]
    assert run.error is None, run.error
    outcomes = {o.spec.name: o for o in run.spec_outcomes}
    assert outcomes['left low'].measurement.value > 1.7
    assert outcomes['left tap low'].measurement.value > 1.7
    assert outcomes['right low'].measurement.value == pytest.approx(nominal.metrics['right'], abs=1e-6)
    assert outcomes['right tap low'].measurement.value == pytest.approx(nominal.metrics['right_tap'], abs=1e-6)
    assert outcomes['right low'].status == 'PASS'
    assert outcomes['right tap low'].status == 'PASS'
    assert [r.target for r in run.manifest.applied] == ['xdut.xleft.mn1']
    first = next(r for r in run.manifest.applied if r.target == 'xdut.xleft.mn1')
    assert first.fault_type == 'stuck_open'
    deck = Path(run.rawfile_path).with_name('netlist.cir').read_text()
    assert '__silk_fault_pair_78647574' in deck
    assert '__silk_fault_pull_786475742e786c656674' in deck
    assert first.netlist_comment in deck
    assert 'xdut.xleft.tap' in deck
    np.testing.assert_array_equal(tb.probe('dut.left.tap').y, original.y)
    again = tb.run()
    assert again.passed
    for metric in ('left', 'right', 'left_tap', 'right_tap'):
        assert again.metrics[metric] == pytest.approx(nominal.metrics[metric], abs=1e-6)


@pytest.mark.slow
def test_random_vth_manifest_matches_simulated_voltage_card(tmp_path):
    import json
    import re
    from pathlib import Path
    from tests._env import HAVE_NGSPICE, HAVE_PDK
    if not HAVE_NGSPICE or not HAVE_PDK:
        pytest.skip('needs ngspice and installed sky130 models')
    from silk import Module, load_pdk, sources
    from silk.elements import elements
    from silk.faults import FaultCampaign, FaultSet, vth_shift
    from silk.simulation import Transient

    pdk = load_pdk('sky130')
    leaf = Module('sky130', str(tmp_path), 'pull.cir', pdk_details=pdk,
                  module_name='pull')
    leaf.define_module_ports({'out': 'out', 'vdd': 'power', 'vg': 'in', 'vss': 'ground'})
    leaf.add(elements.resistor('load', p='vdd', n='out', value=1e4))
    leaf.add(pdk.nmos1p8('mn1', d='out', g='vg', s='vss', b='vss', w=4, l=.5))
    tb = leaf.create_test('level')
    tb.add_module_under_test({'out': 'out', 'vdd': 'vdd', 'vg': 'vg', 'vss': '0'})
    for name, value in [('vdd', 1.8), ('vg', .6)]:
        tb.add_source(sources.VoltageSource(name, positive=name, negative='0', dc=value))
    tb.add_sim_type(Transient(step=1e-9, stop=10e-9))
    result = FaultCampaign(
        tb.netlist, FaultSet([vth_shift('xdut.mn1', delta=.05, sigma=.02)]),
        n_runs=1, seed=7, out_dir=str(tmp_path / 'campaign'), n_workers=1,
        archive_waveforms=True, reconcile_specs=False, progress=False).run()
    run = result.runs[0]
    assert run.error is None, run.error
    deck = Path(run.rawfile_path).with_name('netlist.cir').read_text()
    paths = list((tmp_path / 'campaign').rglob('manifest.json'))
    assert len(paths) == 1
    applied = json.loads(paths[0].read_text())['applied']
    voltage = float(re.search(r'^V__silk_fault_vt_786475742e6d6e31 __silk_fault_offset_786475742e6d6e31 0 DC (\S+)',
                              deck, re.M)[1])
    assert voltage == pytest.approx(.0665653, abs=5e-8)
    for record in applied:
        assert record['target'] == 'xdut.mn1'
        assert record['realized_params']['delta_total'] == pytest.approx(voltage, abs=5e-8)
    assert len(applied) == 1
