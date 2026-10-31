import os
import shutil

import pytest

from silk.examples.ota_5t_guide import build_5t_ota
from silk.simulation.runner import ngspice_executable
from tests._env import needs_pdk

# Every test here builds against the real installed PDK.
pytestmark = needs_pdk


def test_ota_builds_with_five_devices():
    mod = build_5t_ota()
    assert mod.module_name == 'ota_5t'
    assert len(mod.elem_dict) == 5
    for p in ('VDD', 'VSS', 'VINP', 'VINM', 'VOUT', 'IBIAS'):
        assert p in mod.pin_list


def test_ota_embeds_testbench_and_design_spec():
    """The module carries its own characterization — the testbench-as-part-
    of-module pattern. No external test/spec objects to pass around."""
    mod = build_5t_ota()
    # Both testbenches are owned by the module.
    assert 'ac' in mod.list_tests()
    assert 'tran' in mod.list_tests()
    # tb.add_spec declared first-class design Specs on the module.
    gain_spec = next(s for s in mod._specs if s.metric == 'gain_dB')
    assert gain_spec.op == '>'
    assert gain_spec.target == 20.0
    bw_spec = next(s for s in mod._specs if s.metric == 'bw_hz')
    assert bw_spec.op == '>'
    assert bw_spec.target == 10e6
    settle_spec = next(s for s in mod._specs if s.metric == 't_settle_1pct')
    assert settle_spec.op == '<'
    assert settle_spec.target == 100e-9
    assert 'noise' in mod.list_tests()
    noise_spec = next(s for s in mod._specs if s.metric == 'vn_in_uV')
    assert noise_spec.op == '<'
    assert noise_spec.target == 2000.0


def test_ota_generates_subckt_netlist():
    mod = build_5t_ota()
    netlist = mod.generate_netlist()           # module_name set → subckt string
    assert netlist is not None
    assert '.subckt' in netlist.lower()
    # five MOS device instance lines
    assert sum(line.strip().lower().startswith('x') or
               line.strip().lower().startswith('m')
               for line in netlist.splitlines()) >= 5


@pytest.mark.parametrize('bench', [None, 'ac', 'tran', 'noise'])
def test_ota_design_sizes_reach_every_dut(bench):
    mod = build_5t_ota(pair_w=5.0, pair_l=0.6, mirror_w=9.0,
                       mirror_l=0.7, tail_w=7.0, tail_l=1.2)
    dut = mod if bench is None else mod._tests[bench].netlist.elem_dict['dut']['device']
    for name, width, length in [('M1', 5.0, 0.6), ('M2', 5.0, 0.6),
                                ('M3', 9.0, 0.7), ('M4', 9.0, 0.7),
                                ('M5', 7.0, 1.2)]:
        params = dut.elem_dict[name]['device'].params
        assert params['w'] == width
        assert params['l'] == length
        line = next(line for line in dut.generate_netlist().splitlines()
                    if line.startswith(f'XM{name} '))
        assert f'w={width:g} l={length:g}' in line


@pytest.mark.parametrize('bench', ['ac', 'tran', 'noise'])
def test_ota_tail_bias_changes_only_bias_source(bench):
    default = build_5t_ota()._tests[bench].netlist.sig_sources_dict
    changed = build_5t_ota(tail_bias=0.85)._tests[bench].netlist.sig_sources_dict
    assert changed.keys() == default.keys()
    assert changed['vbias'].dc == 0.85
    assert changed['vbias'].positive == 'ibias'
    assert changed['vbias'].negative == '0'
    for name in default.keys() - {'vbias'}:
        assert changed[name].to_spice() == default[name].to_spice()


def test_ota_design_knobs_are_keyword_only():
    with pytest.raises(TypeError):
        build_5t_ota(5.0)


@pytest.mark.slow
@pytest.mark.skipif(shutil.which(ngspice_executable()) is None,
                    reason="ngspice not installed")
def test_ota_characterizes_itself_and_reconciles(tmp_path):
    from silk.examples.ota_5t_guide import (
        build_5t_ota, run_characterization, plot_transfer_function)
    mod = build_5t_ota()
    results, recon = run_characterization(mod)

    # The AC testbench ran and both AC metrics were extracted.
    ac_metrics = results['ota_5t.ac'].metrics
    assert ac_metrics['gain_dB'] > 20.0, f"expected >20 dB DC gain, got {ac_metrics['gain_dB']}"
    assert ac_metrics['bw_hz'] > 10e6, f"expected >10 MHz BW, got {ac_metrics['bw_hz']}"

    # The transient testbench ran and settled quickly (unity-gain follower).
    t_settle = results['ota_5t.tran'].metrics['t_settle_1pct']
    assert 0 <= t_settle < 100e-9, f"expected <100 ns settling, got {t_settle}"

    # The noise testbench ran; input-referred noise is a sane positive number.
    vn = results['ota_5t.noise'].metrics['vn_in_uV']
    assert 0 < vn < 2000.0, f"expected <2 mV input-referred noise, got {vn} uV"

    # Running the testbenches auto-recorded the measurements onto the module's
    # design specs, so reconciliation passes with no manual bookkeeping.
    assert recon.fail_count == 0
    assert recon.pass_count >= 4
    for metric in ('gain_dB', 'bw_hz', 't_settle_1pct', 'vn_in_uV'):
        outcome = next(o for o in recon.outcomes if o.spec.metric == metric)
        assert outcome.status == 'PASS', f"{metric} did not pass: {outcome.status}"

    # The transfer-function plot is saved as a non-empty PNG.
    png = plot_transfer_function(
        mod._tests['ac'], results['ota_5t.ac'], str(tmp_path / 'tf.png'))
    assert os.path.exists(png) and os.path.getsize(png) > 0
