"""Per-block mismatch off (TASK-430): generated decks only, no simulator."""
import pytest

from silk import Module, Port, load_pdk, sources
from silk.circuit.monte_carlo import _SampleSpec
from silk.exceptions import NotSupportedError
from silk.simulation import Op


def _pair(tmp_path, pdk, name, off):
    blk = Module('sky130', str(tmp_path), f'{name}.cir', pdk, module_name=name)
    blk.define_module_ports({'g1': Port.analog_input(),
                             'g2': Port.analog_input(),
                             'vss': Port.ground()})
    if off:
        blk.set_mismatch(False)
    with blk:
        pdk.nmos1p8('m1', d='g1', g='g1', s='vss', b='vss', w=5.0, l=2.0)
        pdk.nmos1p8('m2', d='g2', g='g2', s='vss', b='vss', w=5.0, l=2.0)
    return blk


def build_two_pairs(tmp_path, off_pair=True, analysis=None):
    """Top level holding ``pair_off`` (mismatch off) and ``pair_on``."""
    pdk = load_pdk('sky130')
    top = Module('sky130', str(tmp_path), 'top.cir', pdk)
    top.add_module(_pair(tmp_path, pdk, 'pair_off', off_pair), 'xoff',
                    nets={'g1': 'a1', 'g2': 'a2', 'vss': '0'})
    top.add_module(_pair(tmp_path, pdk, 'pair_on', False), 'xon',
                    nets={'g1': 'b1', 'g2': 'b2', 'vss': '0'})
    for n in ('a1', 'a2', 'b1', 'b2'):
        top.add_source(sources.CurrentSource(
            f'i{n}', positive='0', negative=n, dc=20e-6))
    top.add_sim_type(analysis or Op())
    return top


def _sample(variation='mismatch', simulator_section='tt_mm'):
    return _SampleSpec(index=0, seed=7, variation=variation,
                       section=simulator_section, wrapper_path=None)


def _deck(top, corner='tt'):
    top.set_output_options(mode='overwrite')
    top.generate_netlist(corner=corner)
    with open(top._run_files.netlist_path) as f:
        return f.read()


def _block(deck, name):
    start = deck.index(f'.subckt {name} ')
    return deck[start:deck.index('.ends', start)].splitlines()


def test_default_is_enabled(tmp_path):
    assert _pair(tmp_path, load_pdk('sky130'), 'p', False).mismatch_enabled
    assert not _pair(tmp_path, load_pdk('sky130'), 'q', True).mismatch_enabled


def test_mismatch_deck_has_the_param_as_first_line_in_that_block(tmp_path):
    top = build_two_pairs(tmp_path)
    top._mc_sample = _sample()
    deck = _deck(top)
    off, on = _block(deck, 'pair_off'), _block(deck, 'pair_on')
    assert off[1] == '.param mc_mm_switch=0'
    assert not any(l.startswith('.param') for l in on)
    assert deck.count('.param mc_mm_switch=0') == 1


def test_both_deck_also_switches_the_block_off(tmp_path):
    top = build_two_pairs(tmp_path)
    top._mc_sample = _SampleSpec(index=0, seed=7, variation='both',
                                 section='mc_both', wrapper_path=None)
    # a both-sample names the wrapper section; only the emitted line matters
    assert '.param mc_mm_switch=0' in '\n'.join(_block(_deck(top), 'pair_off'))


def test_process_deck_is_unchanged(tmp_path):
    top = build_two_pairs(tmp_path)
    top._mc_sample = _sample('process', 'mc')
    assert 'mc_mm_switch' not in _deck(top)


def test_non_monte_carlo_deck_is_byte_identical(tmp_path):
    off = _deck(build_two_pairs(tmp_path / 'a', off_pair=True))
    on = _deck(build_two_pairs(tmp_path / 'b', off_pair=False))
    assert 'mc_mm_switch' not in off
    assert off.replace(str(tmp_path / 'a'), 'D') == on.replace(str(tmp_path / 'b'), 'D')


def test_a_plain_deck_after_a_sample_deck_carries_no_switch(tmp_path):
    top = build_two_pairs(tmp_path)
    top._mc_sample = _sample()
    assert 'mc_mm_switch' in _deck(top)
    top._mc_sample = None
    assert 'mc_mm_switch' not in _deck(top)


def test_devices_argument_is_refused_per_block_only(tmp_path):
    blk = _pair(tmp_path, load_pdk('sky130'), 'p', False)
    with pytest.raises(NotSupportedError, match='per block only'):
        blk.set_mismatch(False, devices=['M1'])


def test_top_level_testbench_raises_value_error(tmp_path):
    top = Module('sky130', str(tmp_path), 't.cir', load_pdk('sky130'))
    with pytest.raises(ValueError, match='top-level'):
        top.set_mismatch(False)


@pytest.mark.parametrize('bad', [0, 1, None, 'no'])
def test_non_bool_raises_type_error(tmp_path, bad):
    blk = _pair(tmp_path, load_pdk('sky130'), 'p', False)
    with pytest.raises(TypeError, match='bool'):
        blk.set_mismatch(bad)


def test_xyce_sample_deck_with_a_block_off_is_refused(tmp_path):
    top = build_two_pairs(tmp_path)
    top.simulator = 'xyce'
    top._mc_sample = _sample()
    with pytest.raises(NotSupportedError):
        top.generate_netlist(corner='tt')


def test_xyce_block_off_refusal_names_the_block_and_the_scope(tmp_path):
    from silk.simulation import Transient
    top = build_two_pairs(tmp_path, analysis=Transient(step=1e-9, stop=5e-9))
    top.simulator = 'xyce'
    top._mc_sample = _sample()
    with pytest.raises(NotSupportedError, match="pair_off.*subcircuit-local"):
        top.generate_netlist(corner='tt')
