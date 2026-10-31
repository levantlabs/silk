"""Characterization guard: non-noise emission must stay byte-identical
through the noise work (Tasks 2-3). These literals are the CURRENT output,
captured from main on 2026-06-25."""
import os
from silk.circuit.spice_netlister import PySpiceNetlister
from silk.simulation import AC, Transient, DC, Noise
from silk.sources.classes import VoltageSource
from silk.examples.ota_5t_guide import build_5t_ota
from silk.pdk.basepdk import BasePDK
from silk.circuit.modules import Module
from silk import sources

_PDK_YAML = os.path.join(os.path.dirname(__file__), '..', 'silk', 'pdk',
                         'data', 'skywater_130_device_details.yaml')


def _generate_top_netlist(dir_folder):
    """Build a fixed top-level netlist and return its text with the
    env-specific dir_folder normalized to '<DIR>'. Deterministic: the .lib
    path is a fixed placeholder; the only env-specific bit is the control
    block's `write <dir>/rawfile.raw` line."""
    pdk = BasePDK(_PDK_YAML, library_location='/PDK/sky130.lib.spice')
    top = Module('PDK_skywater130', dir_folder, 'top_fixture.cir', pdk_details=pdk)
    top.set_output_options(mode='overwrite')
    top.add_net('vout'); top.add_net('vdd'); top.add_net('0')
    top.add(pdk.nmos1p8('M1', d='vout', g='vdd', s='0', b='0', w=1.0, l=0.15, m=1))
    top.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
    top.add_source(sources.VoltageSource('vin', positive='vout', negative='0', dc=0.9, ac=1.0))
    top.add_sim_type(AC('dec', 10, 1, 1e9))
    top.generate_netlist(corner='tt')   # top-level mode → writes the file
    content = open(os.path.join(dir_folder, 'top_fixture.cir')).read()
    return content.replace(dir_folder, '<DIR>')


_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), 'golden')


def _netlister():
    # These methods touch no PDK state, so a bare instance is fine.
    return PySpiceNetlister.__new__(PySpiceNetlister)


def test_default_control_block_unchanged():
    expected = ('.control\nsave all\nrun\nset wr_singlescale\nset wr_vecnames\n'
                'option numdgt=3\nwrite /out/rawfile.raw\nquit 0\n.endc\n.end\n')
    assert _netlister().generate_control_block('/out') == expected


def test_ac_analysis_line_unchanged():
    assert AC('dec', 20, 1, 1e9).to_spice() == '.ac dec 20 1 1G\n'


def test_transient_analysis_line_unchanged():
    assert Transient(1e-9, 2e-6).to_spice() == '.tran 1n 2u\n'
    assert Transient(1e-9, 2e-6, maxstep=20e-12, uic=True).to_spice() == '.tran 1n 2u 0 20p uic\n'
    assert Transient(1e-9, 2e-6, start=1e-6).to_spice() == '.tran 1n 2u 1u\n'
    assert Transient(1e-9, 2e-6, maxstep=20e-12).to_spice() == '.tran 1n 2u 0 20p\n'


def test_dc_analysis_line_unchanged():
    assert DC('vsweep', 0, 1.8, 0.01).to_spice() == '.dc vsweep 0 1.8 0.01\n'   # plain band, not '10m'


def test_noise_analysis_line_unchanged():
    n = Noise('vout', '0', 'vinp', 'dec', 10, 1, 1e9)
    assert n.to_spice() == '.noise v(vout,0) vinp dec 10 1 1G 1\n'
    assert n.to_spice(resolve_source=lambda s: 'V' + s) == '.noise v(vout,0) Vvinp dec 10 1 1G 1\n'


def test_generate_analysis_commands_full_block_unchanged():
    """The whole analysis block for an AC + a Noise analysis, with source
    resolution, is the byte-identical contract the renderer must reproduce."""
    net = PySpiceNetlister.__new__(PySpiceNetlister)
    sources = {'vinp': VoltageSource('vinp', positive='vinp', negative='0', dc=0.9, ac=1.0)}
    analyses = {'ac': AC('dec', 20, 1, 1e9), 'noise': Noise('vout', '0', 'vinp', 'dec', 10, 1, 1e9)}
    out = net.generate_analysis_commands(analyses, sig_sources_dict=sources)
    assert out == (
        '*** AC analysis\n.ac dec 20 1 1G\n'
        '*** Noise analysis\n.noise v(vout,0) Vvinp dec 10 1 1G 1\n'
    )


def test_ota_5t_subckt_golden():
    """The 5T OTA subckt emission is byte-identical to the committed golden.
    Guards subckt-mode emission (which stays silk-level through the C1 refactor)."""
    with open(os.path.join(_GOLDEN_DIR, 'ota_5t.subckt'), encoding='utf-8') as f:
        expected = f.read()
    assert build_5t_ota().generate_netlist() == expected


def test_top_level_netlist_golden(tmp_path):
    """The complete top-level netlist (header, .lib, devices, sources, options,
    analysis, control block) is byte-identical to the committed golden. This is
    the byte-identical guard for the C1 netlist-wrapper refactor."""
    with open(os.path.join(_GOLDEN_DIR, 'top_netlist_ac.cir'), encoding='utf-8') as f:
        expected = f.read()
    assert _generate_top_netlist(str(tmp_path)) == expected


def test_control_blocks_end_with_quit_zero():
    # ``exit`` is only an alias defined by ngspice's startup file; see
    # tests/test_ngspice_quit_terminator.py.
    assert _netlister().generate_control_block('/out').endswith('quit 0\n.endc\n.end\n')
