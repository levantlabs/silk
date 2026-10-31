"""Smoke tests to verify all conftest fixtures instantiate correctly."""

import pytest


def test_tmp_dir(tmp_dir):
    assert isinstance(tmp_dir, str)


def test_pdk(pdk):
    assert 'nmos1p8' in pdk.model_names
    assert 'pmos1p8' in pdk.model_names
    assert hasattr(pdk, 'nmos1p8')
    assert hasattr(pdk, 'pmos1p8')


def test_empty_netlist(empty_netlist):
    assert len(empty_netlist.elem_dict) == 0
    assert len(empty_netlist.sig_sources_dict) == 0


def test_basic_netlist(basic_netlist):
    assert 'mn1' in basic_netlist.elem_dict
    assert 'mp1' in basic_netlist.elem_dict
    assert len(basic_netlist.elem_dict) == 2


def test_netlist_with_sources(netlist_with_sources):
    assert 'vdd_supply' in netlist_with_sources.sig_sources_dict
    assert 'vin' in netlist_with_sources.sig_sources_dict
    assert 'transient' in netlist_with_sources.sim_analysis_dict


def test_sample_signal(sample_signal):
    assert sample_signal.name == 'v(out)'
    assert len(sample_signal.x) == 100


def test_sample_signal_pair(sample_signal_pair):
    sig1, sig2 = sample_signal_pair
    assert sig1.name == 'v(out)'
    assert sig2.name == 'v(ref)'


def test_in_memory_db(in_memory_db):
    assert in_memory_db is not None


def test_db_with_circuit(db_with_circuit):
    db, circuit_id = db_with_circuit
    assert circuit_id is not None


def test_basic_module(basic_module):
    assert basic_module.module_name == 'inv_cell'
    assert 'mn' in basic_module.elem_dict
    assert 'mp' in basic_module.elem_dict




def test_netlister(netlister):
    header = netlister.generate_header('test')
    assert 'test' in header


def test_mock_ngspice(mock_ngspice, tmp_dir):
    import subprocess
    from pathlib import Path
    deck = Path(tmp_dir) / 'test.cir'
    deck.write_text('mock\n.op\n.end\n')
    subprocess.run(['ngspice', '-b', '-r', f'{tmp_dir}/raw.raw', '-o', f'{tmp_dir}/out.log', str(deck)])
    assert mock_ngspice['called']
    assert 'ngspice' in mock_ngspice['command']



@pytest.mark.parametrize('card,kind,points,last', [
    ('.tran 1n 100n', 'tran', 2, 100e-9),
    ('.tran 2n 17n', 'tran', 2, 17e-9),
    ('.dc vin 0 1 .3', 'dc', 4, .9),
    ('.dc vin 1 0 -.3', 'dc', 4, .1),
    ('.dc vin 1 1.0000001 1e-8', 'dc', 11, 1.0000001),
    ('.dc vin 1 .9999999 -1e-8', 'dc', 11, .9999999),
    ('.ac lin 11 1 1.0000001', 'ac', 11, 1.0000001),
    ('.ac dec 2 1 9', 'ac', 2, 9),
    ('.ac oct 2 1 3', 'ac', 4, 8 ** .5),
    ('.noise v(out) vin oct 2 1 3', 'noise', 4, 8 ** .5),
    ('.op', 'op', 1, 0),
])
def test_mock_results_match_launched_request(mock_ngspice, tmp_path, card, kind, points, last):
    import subprocess
    from silk.signals.rawfile import read_rawfile
    from silk.signals.analysis_kind import plot_silk_analysis
    deck, raw = tmp_path / 'mock.cir', tmp_path / 'mock.raw'
    deck.write_text('mock\n' + card + '\n.end\n.ac lin 100 1 100\n')
    subprocess.run(['ngspice', '-b', '-r', str(raw), '-o', str(tmp_path / 'out.log'), str(deck)])
    plots = read_rawfile(raw).plots
    assert len(plots) == 1
    assert plot_silk_analysis(plots[0]) == kind
    assert plots[0].npoints == points
    assert plots[0].vectors[0].data.real[-1] == pytest.approx(last, rel=1e-12)


def test_mock_results_follow_multiple_requests_and_later_launch(mock_ngspice, tmp_path):
    import subprocess
    from silk.signals.rawfile import read_rawfile
    deck, raw = tmp_path / 'mock.cir', tmp_path / 'mock.raw'
    argv = ['ngspice', '-b', '-r', str(raw), '-o', str(tmp_path / 'out.log'), str(deck)]
    deck.write_text('mock\n.tran 1n 17n\n.ac lin 5 1 9\n.end\n')
    subprocess.run(argv)
    plots = read_rawfile(raw).plots
    assert [p.npoints for p in plots] == [2, 5]
    assert [p.vectors[0].data.real[-1] for p in plots] == [17e-9, 9]
    deck.write_text('mock\n.tran 1n 33n\n.end\n')
    subprocess.run(argv)
    assert read_rawfile(raw).plots[0].vectors[0].data[-1] == pytest.approx(33e-9)
