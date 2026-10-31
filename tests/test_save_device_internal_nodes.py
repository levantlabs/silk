"""Strict, opt-in ngspice device-internal-node filtering."""
import os
from pathlib import Path

import pytest

from silk import sources
from silk.circuit.modules import Module
from silk.exceptions import NotSupportedError
from silk.pdk import load_pdk
from silk.simulation import Transient


def bench(tmp_path):
    mod = Module('sky130', str(tmp_path), 'save.cir', pdk_details=load_pdk('sky130'))
    with mod:
        sources.VoltageSource('supply', positive='out', negative='0', dc=1)
    mod.add_sim_type(Transient(step=1e-9, stop=10e-9))
    return mod


def version(monkeypatch, token):
    monkeypatch.setattr('silk.simulation._ngspice_version.ngspice_version', lambda: token)


def deck(mod):
    mod_deck_path = mod.generate_netlist()
    return Path(mod_deck_path).read_text()


def test_default_preserves_save_all(tmp_path):
    mod = bench(tmp_path)
    assert mod.omit_device_internal_nodes is False
    assert '\nsave all\n' in deck(mod)


@pytest.mark.parametrize('batch', [False, True])
def test_filter_emission(tmp_path, monkeypatch, batch):
    version(monkeypatch, '47')
    mod = bench(tmp_path)
    mod.omit_device_internal_nodes = True
    mod.save_device_currents = batch
    text = deck(mod)
    assert ('\n.save nointernals\n' if batch else '\nsave nointernals\n') in text
    assert 'save all' not in text


@pytest.mark.parametrize('token', ['42', '43', '44', None, 'unknown'])
def test_unsupported_version_refuses(tmp_path, monkeypatch, token):
    version(monkeypatch, token)
    mod = bench(tmp_path)
    mod.omit_device_internal_nodes = True
    with pytest.raises(NotSupportedError, match='omit_device_internal_nodes.*45'):
        deck(mod)


def test_selective_save_refuses(tmp_path, monkeypatch):
    version(monkeypatch, '47')
    mod = bench(tmp_path)
    mod.omit_device_internal_nodes = True
    mod.set_save_list(['out'])
    with pytest.raises(NotSupportedError, match='set_save_list'):
        deck(mod)


def test_custom_content_refuses(tmp_path, monkeypatch):
    version(monkeypatch, '47')
    mod = bench(tmp_path)
    mod.omit_device_internal_nodes = True
    mod.add_custom_line('.param harmless=1')
    with pytest.raises(NotSupportedError, match='custom'):
        deck(mod)


def test_xyce_refuses(tmp_path):
    mod = bench(tmp_path)
    mod.simulator = 'xyce'
    mod.omit_device_internal_nodes = True
    with pytest.raises(NotSupportedError, match='ngspice'):
        deck(mod)


def test_behavioral_refuses(tmp_path):
    from silk.simulation.behavioral_backend import _refuse_inapplicable_switches
    mod = bench(tmp_path)
    mod.omit_device_internal_nodes = True
    with pytest.raises(NotSupportedError, match='behavioral'):
        _refuse_inapplicable_switches(mod)


def test_unverified_pdk_refuses(tmp_path, monkeypatch):
    version(monkeypatch, '47')
    mod = bench(tmp_path)
    with mod:
        mod.pdk_details.nmos1p8('mn', d='out', g='out', s='0', b='0', w=1, l=0.15)
    mod.pdk_details.library_location = str(tmp_path / 'unverified.lib')
    mod.omit_device_internal_nodes = True
    with pytest.raises(NotSupportedError, match='unverified'):
        deck(mod)


def test_dut_inheritance(tmp_path, monkeypatch):
    from silk.circuit.circuit_elements import PortDir
    version(monkeypatch, '47')
    dut = Module('sky130', str(tmp_path), 'dut.cir',
                 pdk_details=load_pdk('sky130'), module_name='dut')
    dut.define_module_ports({'out': PortDir.OUT})
    with dut:
        sources.VoltageSource('supply', positive='out', negative='0', dc=1)
    dut.omit_device_internal_nodes = True
    tb = dut.create_test('startup')
    tb.add_module_under_test({'out': 'out'})
    tb.netlist.add_sim_type(Transient(step=1e-9, stop=10e-9))
    deck_path = tb.prepare()
    assert '\nsave nointernals\n' in Path(deck_path).read_text()


@pytest.fixture
def certified_bundle(tmp_path, monkeypatch):
    """A tiny exact certificate exercises content changes without a PDK install."""
    import hashlib
    import silk.simulation._internal_nodes as policy
    root = tmp_path / 'pdk'
    root.mkdir()
    library = root / 'library.spice'
    child = root / 'primitive.spice'
    library.write_text('.lib tt\n.include "primitive.spice"\n.endl tt\n')
    child.write_text('.subckt sky130_fd_pr__nfet_01v8 d g s b\n.ends\n')
    cert = {'id': 'fixture', 'root': 'pdk', 'library': 'library.spice',
            'sections': ['tt'], 'extra_includes': [],
            'factories': {'nmos1p8': 'sky130_fd_pr__nfet_01v8'},
            'files': {p.name: {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                              'includes': []} for p in (library, child)}}
    cert['files']['library.spice']['includes'] = [
        {'line': 2, 'token': 'primitive.spice', 'target': 'primitive.spice'}]
    monkeypatch.setattr(policy, '_certificate', lambda: cert, raising=False)
    monkeypatch.setenv('PDK_ROOT', str(tmp_path))
    version(monkeypatch, '47')
    mod = bench(tmp_path / 'out')
    mod.pdk_details.library_location = str(library)
    mod.pdk_details.extra_includes = []
    with mod:
        mod.pdk_details.nmos1p8('mn', d='out', g='out', s='0', b='0', w=1, l=0.15)
    mod.omit_device_internal_nodes = True
    return mod, child


def test_exact_certificate_admits_and_preserves_parameter_saves(certified_bundle):
    mod, _ = certified_bundle
    mod.save_device_params = True
    text = deck(mod)
    assert '\nsave nointernals\n' in text
    assert '[gm]' in text


@pytest.mark.parametrize('change', ['content', 'missing', 'redirect', 'extra', 'factory'])
def test_certificate_refuses_changes(certified_bundle, change):
    mod, child = certified_bundle
    if change == 'content':
        child.write_text('.subckt extracted d g s b\nr1 d circuit#node 1\n.ends\n')
    elif change == 'missing':
        child.unlink()
    elif change == 'redirect':
        target = child.with_name('redirect.spice')
        child.rename(target)
        child.symlink_to(target)
    elif change == 'extra':
        mod.pdk_details.extra_includes.append(str(child))
    else:
        mod.elem_dict['mn']['device'].__class__ = type(
            'Unverified', (type(mod.elem_dict['mn']['device']),),
            {'_pdk_device': 'extracted'})
    with pytest.raises(NotSupportedError, match='unverified'):
        deck(mod)


@pytest.mark.parametrize('kind', ['resistor', 'source', 'include'])
def test_opaque_literals_refuse(tmp_path, monkeypatch, kind):
    from silk.elements import elements
    version(monkeypatch, '47')
    mod = bench(tmp_path)
    from silk.exceptions import DeckTextError
    payload = '1k\nRhidden out circuit#node 1\nCstored circuit#node 0 1p'
    if kind == 'include':
        payload = '1k\n.include "unverified.spice"'
    mod.omit_device_internal_nodes = True
    # TASK-593: a multi-line value never reaches the deck. It is refused
    # where it is set, before this certificate's own opaque-card check.
    with pytest.raises(DeckTextError, match='line break'):
        if kind == 'source':
            mod.sig_sources_dict['supply'].dc = payload
        else:
            with mod:
                elements.resistor('r', p='out', n='0', value=payload)
    text = deck(mod)
    assert 'Rhidden' not in text and 'unverified.spice' not in text


def test_legacy_pdk_device(certified_bundle):
    mod, _ = certified_bundle
    mod.add_element(mod.pdk_details._devices['nmos1p8'], 'legacy',
                    nets={'d': 'out', 'g': 'out', 's': '0', 'b': '0'},
                    parameters={'w': 1, 'l': 0.15})
    assert '\nsave nointernals\n' in deck(mod)


@pytest.mark.parametrize('token', ['45', '45.2'])
def test_supported_boundary(tmp_path, monkeypatch, token):
    version(monkeypatch, token)
    mod = bench(tmp_path)
    mod.omit_device_internal_nodes = True
    assert '\nsave nointernals\n' in deck(mod)


def test_model_changes_after_prepare_refuse(certified_bundle):
    from silk.circuit.modules_netlist import _require_fresh_deck
    from silk.exceptions import SimulationConfigurationError
    mod, child = certified_bundle
    deck(mod)
    child.write_text(child.read_text() + '* changed after preparation\n')
    with pytest.raises(SimulationConfigurationError, match='unverified model content'):
        _require_fresh_deck(mod)


def test_flag_changes_after_prepare_refuse(tmp_path, monkeypatch):
    from silk.circuit.modules_netlist import _require_fresh_deck
    from silk.exceptions import SimulationConfigurationError
    version(monkeypatch, '47')
    mod = bench(tmp_path)
    deck(mod)
    mod.omit_device_internal_nodes = True
    with pytest.raises(SimulationConfigurationError):
        _require_fresh_deck(mod)


@pytest.mark.parametrize('pattern', ['bits', 'prbs'])
def test_builtin_waveform_continuations(tmp_path, monkeypatch, pattern):
    version(monkeypatch, '47')
    mod = bench(tmp_path)
    wave = (sources.BitPattern('01', v1=0, v2=1, ui='1n', tr='10p', tf='10p')
            if pattern == 'bits' else
            sources.PRBS(7, 16, v1=0, v2=1, ui='1n', tr='10p', tf='10p'))
    mod.add_source(sources.VoltageSource('bits', positive='stim', negative='0', waveform=wave))
    mod.omit_device_internal_nodes = True
    assert '\nsave nointernals\n' in deck(mod)
