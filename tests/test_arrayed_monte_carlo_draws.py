"""TASK-520: showmod quoting for arrayed instance selectors."""
from pathlib import Path

import numpy as np
import pytest

from silk import load_pdk, sources
from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.circuit.monte_carlo_draws import parse_showmod, showmod_lines
from silk.circuit.spice_netlister import PySpiceNetlister
from silk.simulation import Op
from tests._env import needs_ngspice, needs_pdk


def test_control_deck_single_quotes_every_complete_showmod_selector(tmp_path):
    paths = ['xmplain', 'xmleaf<0>', 'xparent<0>.xmplain',
             'xparent<0>.xmleaf<0>']
    lines = showmod_lines([(path, 'model', ('vth0', 'toxe')) for path in paths])
    deck = PySpiceNetlister(load_pdk('sky130')).generate_control_block(
        str(tmp_path), draw_lines=lines)
    emitted = [line for line in deck.splitlines() if line.startswith('showmod ')]
    assert emitted == [f"showmod 'm.{path}.mmodel' : vth0 toxe" for path in paths]


@pytest.mark.parametrize('path,model', [("xm'leaf", 'model'), ('xmleaf', "mod'el")])
def test_showmod_refuses_single_quote_in_selector(path, model):
    with pytest.raises(ValueError, match='single quote.*rename'):
        showmod_lines([(path, model, ('vth0',))])


def _bench(tmp_path):
    pdk = load_pdk('sky130')
    def block(name):
        mod = Module('sky130', str(tmp_path), name + '.cir', pdk,
                     module_name=name)
        mod.define_module_ports({'d': PortDir.INOUT, 'vss': PortDir.GROUND})
        for device in ('plain', 'leaf[0]'):
            mod.add(pdk.nmos1p8(device, d='d', g='d', s='vss', b='vss',
                               w=5, l=2))
        return mod
    dut = block('bank')
    dut.add_module(block('cell'), 'parent[0]', nets={'d': 'd', 'vss': 'vss'})
    tb = dut.create_test('draws')
    tb.add_module_under_test({'d': 'd', 'vss': '0'})
    tb.add(sources.vdc('bias', positive='d', negative='0', value=1))
    tb.add_sim_type(Op())
    return tb, tuple(pdk.statistical.record['nmos1p8'])


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_one_sample_records_all_four_plain_and_arrayed_device_draws(tmp_path):
    tb, params = _bench(tmp_path)
    mc = tb.run_monte_carlo(1, record_draws=True, n_workers=1)
    assert not mc.failed and len(mc.reports) == 1
    names = ['dut.plain', 'dut.leaf[0]', 'dut.parent[0].plain',
             'dut.parent[0].leaf[0]']
    requested = [(name, params) for name in names]
    assert list(mc.draws.columns) == [(name, p) for name in names for p in params]
    report = next(iter(mc.reports.values()))
    log = Path(report.rawfile_path).with_name('output.log').read_text()
    parsed = parse_showmod(log, requested)
    assert list(parsed) == list(mc.draws.columns)
    assert len({name for name, _ in parsed}) == 4
    assert len(mc.draws.values) == 1
    values = next(iter(mc.draws.values.values()))
    assert np.isfinite(values).all()
    np.testing.assert_array_equal(values, list(parsed.values()))
