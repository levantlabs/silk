"""Monte Carlo sample deck emission (TASK-434): generated text only, no simulator."""
from pathlib import Path

import pytest

from silk import Module, Port, load_pdk, sources
from silk.circuit.monte_carlo import _SampleSpec, write_both_wrapper
from silk.simulation import Transient


@pytest.fixture
def nmos_module(tmp_path):
    pdk = load_pdk('sky130')
    dut = Module('PDK_skywater130', str(tmp_path), 'mc.cir',
                 pdk_details=pdk, module_name='nmos_mc')
    dut.define_module_ports({'D': Port.analog_output(), 'G': Port.analog_input(),
                             'S': Port.ground()})
    with dut:
        pdk.nmos1p8('m1', d='D', g='G', s='S', b='S', w=2.0, l=0.5)
    tb = dut.create_test('mc')
    tb.add_module_under_test({'D': 'drain', 'G': 'gate', 'S': '0'})
    tb.add_source(sources.VoltageSource('drain', positive='drain', negative='0', dc=0.8))
    tb.add_source(sources.VoltageSource('gate', positive='gate', negative='0', dc=1.0))
    tb.add_sim_type(Transient(step=1e-9, stop=5e-9))
    return tb.netlist


def _deck(netlist, corner):
    """generate_netlist writes the deck and returns None; read it back."""
    netlist.set_output_options(mode='overwrite')
    netlist.generate_netlist(corner=corner)
    with open(netlist._run_files.netlist_path) as f:
        return f.read()


def test_mismatch_sample_deck_names_the_mm_section_and_seed(nmos_module):
    nmos_module._mc_sample = _SampleSpec(index=1, seed=12345, variation='mismatch',
                                         section='ss_mm', wrapper_path=None)
    deck = _deck(nmos_module, corner='ss')
    assert ' ss_mm\n' in deck and '.options seed=12345' in deck.lower()


def test_process_sample_deck_names_the_process_section(nmos_module):
    st = load_pdk('sky130').statistical
    nmos_module._mc_sample = _SampleSpec(index=0, seed=7, variation='process',
                                         section=st.process_section, wrapper_path=None)
    deck = _deck(nmos_module, corner='tt')
    assert f' {st.process_section}\n' in deck and 'seed=7' in deck


def test_both_sample_deck_includes_the_wrapper(nmos_module, tmp_path):
    st = load_pdk('sky130').statistical
    wrapper = write_both_wrapper(str(tmp_path / 'mc_both.lib.spice'), st)
    nmos_module._mc_sample = _SampleSpec(index=0, seed=3, variation='both',
                                         section=None, wrapper_path=wrapper)
    deck = _deck(nmos_module, corner='tt')
    assert f'.lib "{wrapper}" mc_both\n' in deck and 'seed=3' in deck


def test_both_wrapper_sets_both_switches_with_absolute_includes(tmp_path):
    st = load_pdk('sky130').statistical
    path = write_both_wrapper(str(tmp_path / 'mc_both.lib.spice'), st)
    text = open(path).read()
    assert f'.param {st.mismatch_switch}=1' in text
    assert f'.param {st.process_switch}=1' in text
    assert '.lib mc_both' in text and '.endl mc_both' in text
    includes = [l for l in text.splitlines() if l.startswith('.include')]
    assert len(includes) == len(st.process_includes) > 0
    assert all(l.split('"')[1].startswith('/') for l in includes)
    assert write_both_wrapper(path, st) == path and open(path).read() == text


def test_no_sample_means_an_identical_deck(nmos_module):
    before = _deck(nmos_module, corner='tt')
    nmos_module._mc_sample = None
    assert _deck(nmos_module, corner='tt') == before
    assert 'seed' not in before.lower()


def _sample():
    return _SampleSpec(index=1, seed=12345, variation='mismatch',
                       section='ss_mm', wrapper_path=None)


def test_xyce_sample_deck_samples_with_use_expr_and_one_sample(nmos_module):
    nmos_module.simulator = 'xyce'
    nmos_module._mc_sample = _sample()
    deck = _deck(nmos_module, corner='ss')
    assert '.SAMPLING useExpr=true\n' in deck
    assert '.options SAMPLES numsamples=1 sample_type=MC\n' in deck
    assert ' ss_mm\n' in deck
    # The seed is -randseed, not a deck option Xyce would not know.
    assert 'seed' not in deck.lower()
    assert deck.rstrip().endswith('.end')
    assert deck.index('.SAMPLING') > deck.index('.print')


def test_xyce_deck_without_a_sample_is_unchanged(nmos_module):
    nmos_module.simulator = 'xyce'
    deck = _deck(nmos_module, corner='tt')
    assert 'SAMPLING' not in deck and 'SAMPLES' not in deck


def test_xyce_sample_command_ends_with_randseed(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace
    from silk.circuit.monte_carlo import derive_sample_seed
    from silk.simulation import backend as backend_mod, xyce_runner
    sample = _SampleSpec(index=3, seed=derive_sample_seed(0, 3),
                         variation='mismatch', section='tt_mm',
                         wrapper_path=None)
    args = backend_mod.XyceBackend().sample_args(sample)
    assert args == ('-randseed', str(derive_sample_seed(0, 3)))
    assert backend_mod.NgspiceBackend().sample_args(sample) == ()
    seen = {}

    def fake_run(cmd, **kw):
        seen['cmd'] = cmd
        return SimpleNamespace(returncode=1, stdout='', stderr='')
    monkeypatch.setattr(subprocess, 'run', fake_run)
    monkeypatch.setattr(xyce_runner, '_xyce_executable', lambda: 'Xyce')
    (tmp_path / 'd.cir').write_text('Sample fixture\n.end\n')
    runner = xyce_runner.XyceRunner(str(tmp_path / 'd.cir'), str(tmp_path),
                                    extra_args=args)
    with pytest.raises(Exception):
        runner.run()
    staged = Path(seen['cmd'][1])
    assert seen['cmd'] == ['Xyce', str(staged), *args]
    assert staged.parent == tmp_path and staged.name.startswith('.silk-')
    assert staged.suffix == '.cir'
    assert staged.read_bytes() == (tmp_path / 'd.cir').read_bytes()
    assert runner.netlist_path == str(tmp_path / 'd.cir')


def test_xyce_runner_without_extra_args_has_only_the_deck_argument(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace
    from silk.simulation import xyce_runner
    seen = {}
    monkeypatch.setattr(subprocess, 'run', lambda cmd, **kw: seen.update(cmd=cmd)
                        or SimpleNamespace(returncode=1, stdout='', stderr=''))
    monkeypatch.setattr(xyce_runner, '_xyce_executable', lambda: 'Xyce')
    (tmp_path / 'd.cir').write_text('Sample fixture\n.end\n')
    runner = xyce_runner.XyceRunner(str(tmp_path / 'd.cir'), str(tmp_path))
    with pytest.raises(Exception):
        runner.run()
    staged = Path(seen['cmd'][1])
    assert seen['cmd'] == ['Xyce', str(staged)]
    assert staged.parent == tmp_path and staged.name.startswith('.silk-')
    assert staged.suffix == '.cir'
    assert staged.read_bytes() == (tmp_path / 'd.cir').read_bytes()
    assert runner.netlist_path == str(tmp_path / 'd.cir')


def test_xyce_sample_deck_with_a_per_corner_includes_corner_is_refused(
        nmos_module, monkeypatch):
    from silk.exceptions import NotSupportedError
    nmos_module.simulator = 'xyce'
    nmos_module._mc_sample = _sample()
    monkeypatch.setattr(type(nmos_module.netlister), '_corner_includes',
                        lambda self, meta: ['a.lib'])
    with pytest.raises(NotSupportedError):
        nmos_module.generate_netlist(corner='ss')


def test_sample_on_a_per_corner_includes_corner_is_refused(nmos_module, monkeypatch):
    from silk.exceptions import NotSupportedError
    corners = nmos_module.pdk_details.corners
    monkeypatch.setitem(corners, 'ss', {
        'includes': [{'file': '/x/a.lib', 'section': 'a'},
                     {'file': '/x/b.lib', 'section': 'b'}]})
    nmos_module._mc_sample = _sample()
    with pytest.raises(NotSupportedError, match="'ss'"):
        nmos_module.generate_netlist(corner='ss')


def test_xyce_draws_come_from_the_res_file_under_designer_names(tmp_path):
    import os
    from silk.circuit.modules_results import xyce_draw_name_map
    from silk.circuit.monte_carlo_draws import parse_xyce_res
    from tests.test_monte_carlo_mismatch_off import build_two_pairs
    top = build_two_pairs(tmp_path, off_pair=False)
    record = load_pdk('sky130').statistical.record
    name_map, params, skipped = xyce_draw_name_map(top.elem_dict, record)
    assert name_map['XXOFF:XMM1'] == 'xoff.m1' and skipped == ()
    assert params['xon.m2'] == ('vth0', 'toxe', 'voff', 'nfactor')
    saved = os.path.join(os.path.dirname(__file__), 'data',
                         'xy_block_mismatch_off_one_sample.res')
    # The saved file is from a hand-written deck (instances xoff/xm1); silk's
    # own deck spells them xxoff/xmm1. Respell the columns, keep the numbers.
    res = str(tmp_path / 'd.cir.res')
    with open(saved) as f, open(res, 'w') as out:
        out.write(f.read().replace('XOFF:XM', 'XXOFF:XMM')
                  .replace('XON:XM', 'XXON:XMM'))
    draws = parse_xyce_res(res, name_map)
    assert draws[('xoff.m1', 'vth0')] == pytest.approx(2.01835591)
    assert {d for d, _ in draws} == {'xoff.m1', 'xoff.m2', 'xon.m1', 'xon.m2'}
