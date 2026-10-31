"""``vth_shift`` shifts the gate by ``delta`` in a real simulation (TASK-415).

The old realization was one E-source line with a trailing offset,
``Evt_t vp_shifted 0 g 0 1 <delta>``. ngspice read the last two numbers as
polynomial coefficients, so the faulted gate sat at ``1 + delta * V(g)``
instead of ``V(g) + delta``, and Xyce refused the line. Netlist-text tests
cannot catch that; this one runs the deck and reads the shifted gate node.
"""

import numpy as np
import pytest

from tests._env import needs_ngspice, needs_pdk, HAVE_XYCE

_VG, _DELTA = 0.6, 0.05


def _shifted_gate(tmp_path, simulator, leaf):
    from silk import load_pdk
    from silk.circuit.modules import Module
    from silk.faults import FaultSet, vth_shift
    from silk.faults.runner import _run_ngspice
    from silk.signals.rawfile import read_rawfile
    from silk.simulation import Transient
    from silk.sources.classes import VoltageSource

    pdk = load_pdk('sky130')
    shifted_node = {'mn': '__silk_fault_shifted_6d6e',
                    'mi[3]': '__silk_fault_shifted_6d695b335d'}[leaf]
    mod = Module('sky130', str(tmp_path), 'vth.cir', pdk_details=pdk)
    for net in ('d', 'g', '0'):
        mod.add_net(net)
    mod.simulator = simulator
    with mod:
        pdk.nmos1p8(leaf, d='d', g='g', s='0', b='0', w=1.0, l=0.15)
        VoltageSource('vg', positive='g', negative='0', dc=_VG)
        VoltageSource('vd', positive='d', negative='0', dc=1.8)
    mod.add_sim_type(Transient(step=1e-9, stop=10e-9))
    mod.generate_netlist(corner='tt',
                         fault_set=FaultSet([vth_shift(leaf, delta=_DELTA)]))
    files = mod._run_files
    if simulator == 'xyce':
        from pathlib import Path
        deck = Path(files.netlist_path)
        lines = deck.read_text().splitlines(keepends=True)
        amended = 0
        for i, line in enumerate(lines):
            if line.startswith('.print tran format=raw '):
                lines[i] = line.rstrip('\n') + f' v({shifted_node})\n'
                amended += 1
        assert amended == 1
        deck.write_text(''.join(lines))
    result = _run_ngspice(files.netlist_path, files.directory, mod.backend,
                          run_files=files)
    raw = read_rawfile(result['rawfile_path'])
    by_name = {v.name.lower(): v for v in raw.vectors}
    return (np.asarray(by_name[f'v({shifted_node})'].data),
            np.asarray(by_name['v(g)'].data))


@pytest.mark.slow
@needs_pdk
@needs_ngspice
@pytest.mark.parametrize('leaf', ['mn', 'mi[3]'])
def test_vth_shift_gate_is_gate_plus_delta_ngspice(tmp_path, leaf):
    shifted, gate = _shifted_gate(tmp_path, 'ngspice', leaf)
    assert np.allclose(gate, _VG, atol=1e-9)
    assert np.allclose(shifted, _VG + _DELTA, atol=1e-6)


@pytest.mark.slow
@pytest.mark.requires_xyce
@needs_pdk
@pytest.mark.skipif(not HAVE_XYCE, reason="needs configured Xyce")
@pytest.mark.parametrize('leaf', ['mn', 'mi[3]'])
def test_vth_shift_gate_is_gate_plus_delta_xyce(tmp_path, leaf):
    shifted, gate = _shifted_gate(tmp_path, 'xyce', leaf)
    assert np.allclose(gate, _VG, atol=1e-9)
    assert np.allclose(shifted, _VG + _DELTA, atol=1e-6)


@pytest.mark.slow
@needs_pdk
@needs_ngspice
@pytest.mark.parametrize('kind', ['short', 'shift'])
def test_indexed_parent_helpers_parse_in_ngspice(tmp_path, kind):
    from silk import load_pdk
    from silk.circuit.modules import Module
    from silk.faults import FaultSet, short_ds, vth_shift
    from silk.faults.runner import _run_ngspice
    from silk.signals.rawfile import read_rawfile
    from silk.simulation import Transient
    from silk.sources.classes import VoltageSource

    pdk = load_pdk('sky130')
    sub = Module('sky130', str(tmp_path), 'sub.cir', pdk_details=pdk,
                 module_name='sub')
    sub.define_module_ports({'d': 'inout', 'g': 'inout', 's': 'inout'})
    sub.add(pdk.nmos1p8('m', d='d', g='g', s='s', b='s', w=1, l=.15, m=2))
    mod = Module('sky130', str(tmp_path), 'indexed.cir', pdk_details=pdk)
    for n in ('d', 'g', '0'):
        mod.add_net(n)
    mod.add_module(sub, 'u[8]', nets={'d': 'd', 'g': 'g', 's': '0'})
    mod.add(VoltageSource('vg', positive='g', negative='0', dc=_VG))
    mod.add(VoltageSource('vd', positive='d', negative='0', dc=1.8))
    mod.add_sim_type(Transient(step=1e-9, stop=10e-9))
    fault = short_ds('xu[8].m') if kind == 'short' else vth_shift('xu[8].m', delta=_DELTA)
    mod.generate_netlist(corner='tt', fault_set=FaultSet([fault]))
    files = mod._run_files
    result = _run_ngspice(files.netlist_path, files.directory, mod.backend,
                          run_files=files)
    raw = read_rawfile(result['rawfile_path'])
    by_name = {v.name.lower(): v for v in raw.vectors}
    assert np.allclose(by_name['v(g)'].data, _VG, atol=1e-9)
    assert np.asarray(by_name['time'].data)[-1] == pytest.approx(10e-9)
