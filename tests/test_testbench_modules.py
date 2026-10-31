"""Helper subcircuits share the existing composition path without becoming DUTs."""
import os
from pathlib import Path

import pytest

from silk import Module, sources
from silk.circuit.circuit_elements import CircuitError
from silk.elements import elements
from silk.exceptions import DuplicateNameError, NotASubcircuitError, UndeclaredNetError
from silk.simulation import Transient
from tests._env import needs_ngspice


def _block(pdk, root, name):
    mod = Module('sky130', str(root), f'{name}.cir', pdk_details=pdk,
                 module_name=name)
    mod.define_module_ports({'out': 'output', 'vss': 'ground'})
    return mod


@pytest.fixture
def blocks(pdk, tmp_path):
    dut = _block(pdk, tmp_path, 'load')
    dut.add(elements.resistor('load', p='out', n='vss', value=1000))
    leaf = _block(pdk, tmp_path, 'bias_leaf')
    leaf.add_source(sources.VoltageSource('bias', positive='out', negative='vss', dc=1))
    helper = _block(pdk, tmp_path, 'biasgen')
    helper.add_module(leaf, 'leaf', nets={'out': 'out', 'vss': 'vss'})
    return dut, helper


def _deck(tb):
    tb_netlist_deck_path = tb.netlist.generate_netlist()
    return Path(tb_netlist_deck_path).read_text()


@pytest.mark.parametrize('helpers_first', [False, True])
def test_multiple_nested_helpers_share_definitions(blocks, helpers_first):
    dut, helper = blocks
    tb = dut.create_test('nested')
    if not helpers_first:
        tb.add_module_under_test({'out': 'out', 'vss': '0'})
    first = tb.add_module(helper, 'bias1', {'out': 'out', 'vss': '0'})
    tb.add_module(helper, 'bias2', {'out': 'aux', 'vss': '0'})
    assert repr(first) == "SubcircuitInstance('xbias1' of 'biasgen')"
    assert repr(first.leaf) == "SubcircuitInstance('xleaf' of 'bias_leaf')"
    if helpers_first:
        assert tb.netlist is not None
        assert not tb._module_added
        with pytest.raises(ValueError, match='add_module_under_test'):
            tb.run()
        tb.add_module_under_test({'out': 'out', 'vss': '0'})
    # The direct composition API remains usable beside the convenience API.
    tb.netlist.add_module(helper, 'legacy', nets={'out': 'other', 'vss': '0'})
    deck = _deck(tb)
    for line in ('xdut out 0 load', 'xbias1 out 0 biasgen',
                 'xbias2 aux 0 biasgen', 'xlegacy other 0 biasgen',
                 'xleaf out vss bias_leaf'):
        assert line in deck
    for name in ('load', 'biasgen', 'bias_leaf'):
        assert deck.lower().count(f'.subckt {name} ') == 1
    assert tb.module is dut
    with pytest.raises(ValueError, match='Module already added'):
        tb.add_module_under_test({'out': 'out', 'vss': '0'})


def test_helper_preserves_metric_spec_and_report_ownership(blocks):
    dut, helper = blocks
    tb = dut.create_test('ownership')
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    tb.add_metric('voltage', type='average', signal='out')
    spec = tb.add_spec('Voltage limit', 'voltage', '<', 3, key='limit')
    raw = Path(__file__).parent / 'fixtures' / 'rawfiles' / 'e_ng_tran.raw'
    baseline = tb.evaluate(rawfile=str(raw))
    tb.add_module(helper, 'bias1', {'out': 'out', 'vss': '0'})
    report = tb.evaluate(rawfile=str(raw))
    assert tb.module is dut
    assert spec._owning_testbench is tb
    assert dut._specs == [spec]
    assert helper._specs == []
    assert report.module_name == 'load'
    assert report.test_name == 'ownership'
    assert report.owner_path == '$'
    assert report.metrics == baseline.metrics
    assert report.passed
    assert report.results[0].owner_path == '$'


def test_unknown_helper_port_uses_existing_error(blocks):
    dut, helper = blocks
    tb = dut.create_test('unknown')
    with pytest.raises(CircuitError, match='Pin typo is not in defined pins for element bias1'):
        tb.add_module(helper, 'bias1', {'out': 'out', 'typo': '0'})
    assert 'bias1' not in tb.netlist.elem_dict


def test_missing_helper_port_fails_before_writing_deck(blocks):
    dut, helper = blocks
    tb = dut.create_test('missing')
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_module(helper, 'bias1', {'out': 'out'})
    with pytest.raises(CircuitError, match='In module instantiation bias1, net for pin vss is undefined'):
        _deck(tb)
    path = Path(os.path.join(tb.netlist.dir_folder, tb.netlist.netlistname))
    assert not path.exists() or not path.read_text()


def test_strict_helper_nets_and_net_handles(blocks):
    dut, helper = blocks
    tb = dut.create_test('strict', strict_nets=True)
    with pytest.raises(UndeclaredNetError, match='typo'):
        tb.add_module(helper, 'bias1', {'out': 'typo', 'vss': '0'})
    assert 'bias1' not in tb.netlist.elem_dict
    out = tb.netlist.add_net('out')
    tb.netlist.add_net('0')
    tb.add_module(helper, 'bias1', {'out': out, 'vss': '0'})
    tb.add_module_under_test({'out': out, 'vss': '0'})
    assert 'xbias1 out 0 biasgen' in _deck(tb)


@pytest.mark.parametrize('name', ['bias1', 'dut'])
def test_helper_duplicate_names_are_rejected(blocks, name):
    dut, helper = blocks
    tb = dut.create_test('duplicates')
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_module(helper, 'bias1', {'out': 'out', 'vss': '0'})
    # The same helper in the same slot is idempotent; a distinct object clashes.
    import copy
    other = copy.deepcopy(helper)
    with pytest.raises(DuplicateNameError, match=name):
        tb.add_module(other, name, {'out': 'aux', 'vss': '0'})


def test_helper_named_dut_does_not_satisfy_dut_requirement(blocks):
    dut, helper = blocks
    tb = dut.create_test('reserved')
    tb.add_module(helper, 'dut', {'out': 'out', 'vss': '0'})
    with pytest.raises(DuplicateNameError, match='dut'):
        tb.add_module_under_test({'out': 'out', 'vss': '0'})
    with pytest.raises(ValueError, match='add_module_under_test'):
        tb.run()
    assert tb.module is dut


@pytest.mark.parametrize('child', [None, 'not a module'])
def test_helper_must_be_a_module(blocks, child):
    dut, _ = blocks
    tb = dut.create_test('type')
    with pytest.raises(TypeError, match='bias1.*must be a Module'):
        tb.add_module(child, 'bias1', {})
    assert tb.netlist is None


def test_helper_must_be_a_named_subcircuit(blocks, pdk, tmp_path):
    dut, _ = blocks
    tb = dut.create_test('top_level')
    top = Module('sky130', str(tmp_path), 'top.cir', pdk_details=pdk)
    with pytest.raises(NotASubcircuitError, match='bias1.*module_name'):
        tb.add_module(top, 'bias1', {})
    assert tb.netlist is None


@pytest.mark.slow
@needs_ngspice
def test_nested_helper_drives_dut_in_real_simulation(blocks):
    dut, helper = blocks
    tb = dut.create_test('driven')
    tb.add_module(helper, 'bias1', {'out': 'out', 'vss': '0'})
    tb.add_module(helper, 'bias2', {'out': 'aux', 'vss': '0'})
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    tb.add_metric('voltage', type='average', signal='out')
    tb.add_spec('Driven', 'voltage', '>', 0.9)
    report = tb.run()
    assert report.metrics['voltage'] == pytest.approx(1)
    assert report.passed
    assert report.module_name == 'load'
