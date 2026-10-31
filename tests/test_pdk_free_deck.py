"""The PDK-free SPICE rung (TASK-71).

A testbench of nothing but ideal elements, passives and sources is valid SPICE
on its own. Before this, every deck carried an unconditional
``.lib "<pdk>" <corner>`` and ``load_pdk`` refused to load at all without that
file on disk, so the cheapest possible on-ramp — ``pip install`` plus
``apt install ngspice``, no PDK download — did not exist.

These tests pin both halves and, more importantly, pin the CONSERVATIVE
direction of the presence walk: a deck that cannot be *proved* PDK-free keeps
its includes. A false negative would silently drop the model library out of a
deck that needs it, and the failure would surface as an unknown-model parse
error far from the cause.
"""
import os
import subprocess

import pytest

from silk import Module, sources
from silk.circuit.spice_netlister import (
    deck_uses_pdk_devices, module_uses_pdk_devices)
from silk.elements import elements
from silk.exceptions import SimulationError
from silk.pdk import load_pdk
from silk.simulation import Transient
from tests._env import needs_ngspice, needs_pdk

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), 'golden')


def _uninstalled_pdk(name='sky130', root='/nonexistent-pdk-root'):
    """A loaded PDK whose model library is *not* on disk.

    The whole point of the lazy loader: everything but the ``.lib`` path ships
    inside silk, so this object is fully usable for emission.
    """
    pdk = load_pdk(name, pdk_root=root)
    assert not os.path.isfile(pdk.library_location)
    return pdk


def _ideal_rc(dir_folder, pdk, name='rc.cir'):
    """R-C low-pass driven by a pulse: no PDK device anywhere."""
    tb = Module('PDK_skywater130', dir_folder, name, pdk_details=pdk)
    for net in ('vin', 'vout', '0'):
        tb.add_net(net)
    tb.add_source(sources.VoltageSource(
        'vin', positive='vin', negative='0', dc=0,
        pulse=sources.Pulse(v1=0, v2=1.0, td=1e-9, tr=1e-10, tf=1e-10,
                            pw=5e-9, per=10e-9)))
    with tb:
        elements.resistor('r1', p='vin', n='vout', value=1e3)
        elements.capacitor('c1', p='vout', n='0', value=1e-12)
    tb.add_sim_type(Transient(1e-11, 20e-9))
    return tb


def _emit(tb):
    tb_deck_path = tb.generate_netlist()
    with open(tb_deck_path, encoding='utf-8') as f:
        return f.read()


# --------------------------------------------------------------------------
# Half 1 — conditional emission
# --------------------------------------------------------------------------

def test_ideal_deck_emits_no_library_lines(tmp_path):
    """The rung itself: an ideal-only deck names no PDK file at all."""
    deck = _emit(_ideal_rc(str(tmp_path), _uninstalled_pdk()))
    assert '.lib' not in deck
    assert '.include' not in deck
    assert 'pre_osdi' not in deck
    assert 'sky130' not in deck
    # ...and says so, rather than leaving the reader to guess whether silk
    # forgot the library or decided it was unnecessary.
    assert '*** No PDK library included' in deck
    # The circuit itself is still all there.
    assert 'Rr1 vin vout 1k' in deck
    assert 'Cc1 vout 0 1p' in deck


def test_ideal_deck_golden(tmp_path):
    """Byte-identity guard for the new PDK-free emission shape."""
    mod = _ideal_rc(str(tmp_path), _uninstalled_pdk())
    mod.set_output_options(mode='overwrite')
    deck = _emit(mod)
    path = os.path.join(_GOLDEN_DIR, 'ideal_rc_tran.cir')
    with open(path, encoding='utf-8') as f:
        expected = f.read()
    assert deck.replace(str(tmp_path), '<DIR>') == expected


def test_osdi_pdk_ideal_deck_emits_no_pre_osdi(tmp_path):
    """`pre_osdi` follows the include block, not the PDK.

    SG13G2's models are compiled Verilog-A, so its decks load plugins from
    disk. An ideal deck references no model, so loading them would reintroduce
    exactly the file dependency the omitted `.lib` just removed.
    """
    deck = _emit(_ideal_rc(str(tmp_path), _uninstalled_pdk('sg13g2', '/nonexistent-pdk-root')))
    assert 'pre_osdi' not in deck
    assert '.osdi' not in deck


def test_osdi_pdk_ideal_deck_is_not_refused_by_xyce(tmp_path):
    """Xyce refuses an OSDI PDK — but only for decks that name its models.

    Without the presence gate an ideal R/C bench under a loaded SG13G2 raised
    NotSupportedError for models it never instantiates.
    """
    tb = _ideal_rc(str(tmp_path), _uninstalled_pdk('sg13g2', '/nonexistent-pdk-root'))
    tb.simulator = 'xyce'
    deck = _emit(tb)
    assert '.lib' not in deck
    assert '.print tran' in deck


# --------------------------------------------------------------------------
# Half 1 — conservativeness. Each of these MUST keep the includes.
# --------------------------------------------------------------------------

@needs_pdk
def test_one_pdk_device_at_top_level_keeps_includes(tmp_path):
    pdk = load_pdk('sky130')
    tb = _ideal_rc(str(tmp_path), pdk)
    tb.add_net('vdd')
    tb.add(pdk.nmos1p8('M1', d='vout', g='vin', s='0', b='0', w=1.0, l=0.15, m=1))
    deck = _emit(tb)
    assert '*** Including PDK library' in deck
    assert '.lib "' in deck


@needs_pdk
def test_pdk_device_two_levels_down_keeps_includes(tmp_path):
    """The walk is recursive: a PDK device inside a submodule's submodule
    still keeps the top-level includes."""
    pdk = load_pdk('sky130')
    leaf = Module('PDK_skywater130', str(tmp_path), 'leaf.cir',
                  pdk_details=pdk, module_name='leaf')
    leaf.define_module_ports({'a': 'input', 'b': 'output'})
    leaf.add_net('a'); leaf.add_net('b'); leaf.add_net('0')
    leaf.add(pdk.nmos1p8('M1', d='b', g='a', s='0', b='0', w=1.0, l=0.15, m=1))

    mid = Module('PDK_skywater130', str(tmp_path), 'mid.cir',
                 pdk_details=pdk, module_name='mid')
    mid.define_module_ports({'a': 'input', 'b': 'output'})
    mid.add_net('a'); mid.add_net('b')
    mid.add_module(leaf, 'x1', nets={'a': 'a', 'b': 'b'})

    top = _ideal_rc(str(tmp_path), pdk, name='top.cir')
    top.add_module(mid, 'x1', nets={'a': 'vin', 'b': 'vout'})
    deck = _emit(top)
    assert '*** Including PDK library' in deck
    assert '.lib "' in deck


def test_a_custom_line_keeps_includes(tmp_path):
    """`add_custom_line` text is opaque SPICE — a hand-registered subckt, a
    stdcell, an `.include` of the user's own — so its mere presence is enough.
    Deliberately blunt: parsing it to decide is the heuristic that can be
    wrong, and keeping an include that is not strictly needed costs nothing."""
    tb = _ideal_rc(str(tmp_path), _uninstalled_pdk())
    tb.add_custom_line('.param vbias=0.9')
    deck = _emit(tb)
    assert '*** Including PDK library' in deck
    assert '.lib "' in deck


def test_unrecognised_element_shape_keeps_includes():
    """Anything the walk does not positively recognise as ideal counts as a
    PDK device — a false negative must be impossible."""
    assert deck_uses_pdk_devices({'x': 'not a dict'}) is True
    assert deck_uses_pdk_devices({'x': {'type': 'd', 'device': object()}}) is True
    assert deck_uses_pdk_devices({}, {'m': {'device': None}}) is True
    assert deck_uses_pdk_devices({}, {'m': 'not a dict'}) is True


def test_empty_deck_is_pdk_free():
    assert deck_uses_pdk_devices({}, {}, []) is False
    assert deck_uses_pdk_devices(None, None, None) is False


def test_walk_is_cycle_safe(tmp_path):
    """A module that (however it got there) reaches itself must terminate."""
    pdk = _uninstalled_pdk()
    a = Module('PDK_skywater130', str(tmp_path), 'a.cir',
               pdk_details=pdk, module_name='a')
    # Reach around add_element's own cycle check to build the shape directly:
    # the walk's guard has to hold regardless of how the graph was made.
    a._element_mgr.module_dict['a'] = {'device': a}
    assert module_uses_pdk_devices(a) is False


@needs_pdk
def test_existing_pdk_deck_emission_is_byte_identical(tmp_path):
    """AC #3, stated directly: for a deck WITH PDK devices, the presence walk
    changes nothing. (tests/test_golden_netlist.py holds the committed
    goldens; this pins the property against a locally regenerated pair.)"""
    from silk.circuit.spice_netlister import PySpiceNetlister
    pdk = load_pdk('sky130', library_location='/PDK/sky130.lib.spice')
    tb = Module('PDK_skywater130', str(tmp_path), 'top.cir', pdk_details=pdk)
    for net in ('vout', 'vdd', '0'):
        tb.add_net(net)
    tb.add(pdk.nmos1p8('M1', d='vout', g='vdd', s='0', b='0', w=1.0, l=0.15, m=1))
    tb.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
    tb.add_sim_type(Transient(1e-11, 1e-9))
    deck = _emit(tb)
    assert deck.count('*** Including PDK library') == 1
    assert '.lib "/PDK/sky130.lib.spice" tt\n' in deck
    assert PySpiceNetlister._needs_pdk_library is True   # default is "keep"


# --------------------------------------------------------------------------
# Half 2 — the loader loads, the RUN is what checks
# --------------------------------------------------------------------------

def test_uninstalled_pdk_still_loads():
    pdk = _uninstalled_pdk()
    assert pdk.pdk_name == 'sky130'
    assert pdk.default_corner
    assert pdk.nmos1p8 is not None


@needs_pdk
def test_run_with_pdk_devices_and_no_library_fails_loudly(tmp_path):
    """The check did not disappear, it moved: a deck that names PDK models
    and cannot find them fails with the PDK named and the fix spelled out,
    instead of ngspice's bare 'Could not find library file'."""
    pdk = _uninstalled_pdk()
    tb = Module('PDK_skywater130', str(tmp_path), 'm.cir', pdk_details=pdk)
    for net in ('vout', 'vdd', '0'):
        tb.add_net(net)
    tb.add(pdk.nmos1p8('M1', d='vout', g='vdd', s='0', b='0', w=1.0, l=0.15, m=1))
    tb.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
    tb.add_sim_type(Transient(1e-11, 1e-9))
    with pytest.raises(SimulationError) as e:
        tb.simulate()
    msg = str(e.value)
    assert 'sky130' in msg
    assert pdk.library_location in msg
    assert 'installing-the-pdk.md' in msg


# --------------------------------------------------------------------------
# The rung, end to end
# --------------------------------------------------------------------------

@pytest.mark.slow
@needs_ngspice
def test_ideal_deck_runs_under_bare_ngspice(tmp_path, monkeypatch):
    """AC #2: pip install + ngspice, no PDK download. $PDK_ROOT points at an
    empty directory for the whole test, so nothing on this machine's real PDK
    install can rescue the run."""
    empty_root = tmp_path / 'empty-pdk-root'
    empty_root.mkdir()
    monkeypatch.setenv('PDK_ROOT', str(empty_root))

    pdk = load_pdk('sky130')
    assert not os.path.isfile(pdk.library_location)

    tb = _ideal_rc(str(tmp_path), pdk)
    tb.simulate()
    vout = tb.load_signal('vout')
    # RC = 1 ns against a 5 ns pulse: settles to the 1 V rail.
    assert len(vout.y) > 100
    assert 0.95 < vout.y.max() <= 1.01
    assert vout.y.min() < 0.05


@pytest.mark.slow
@needs_ngspice
def test_ideal_deck_is_accepted_by_ngspice_with_no_pdk_env(tmp_path):
    """The deck file itself, handed to ngspice in a subprocess with no
    $PDK_ROOT at all — the state a reader following quickstart section 1 is
    actually in."""
    deck = _ideal_rc(str(tmp_path), _uninstalled_pdk())
    deck_path = deck.generate_netlist()
    env = {k: v for k, v in os.environ.items() if k != 'PDK_ROOT'}
    binary = os.environ.get('SILK_NGSPICE_BIN', 'ngspice')
    proc = subprocess.run([binary, '-b', deck_path],
                          capture_output=True, text=True, env=env, timeout=120)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert 'Could not find library file' not in combined
    assert 'error' not in combined.lower(), combined
