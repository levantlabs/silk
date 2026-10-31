import numpy as np
from silk.circuit.spice_netlister import PySpiceNetlister
from silk.simulation import Noise
from silk.sources.classes import VoltageSource, CurrentSource
from silk.signals.spice import NgspiceSignal


def test_source_instance_name():
    assert VoltageSource('vinp', positive='vinp', negative='0', dc=0.9).instance_name == 'Vvinp'
    assert CurrentSource('ib', positive='vdd', negative='b', dc=1e-6).instance_name == 'Iib'


def test_noise_to_spice_resolves_source_and_emits_pts():
    n = Noise('vout', '0', 'vinp', 'dec', 10, 1, 1e9)
    # default (no resolver): unchanged source name, but pts_per_summary present
    assert n.to_spice() == '.noise v(vout,0) vinp dec 10 1 1G 1\n'
    # with a resolver: source resolved to its instance name
    assert n.to_spice(resolve_source=lambda s: 'V' + s) == '.noise v(vout,0) Vvinp dec 10 1 1G 1\n'


def test_generate_analysis_commands_resolves_noise_source():
    net = PySpiceNetlister.__new__(PySpiceNetlister)
    sources = {'vinp': VoltageSource('vinp', positive='vinp', negative='0', dc=0.9, ac=1.0)}
    out = net.generate_analysis_commands({'noise': Noise('vout', '0', 'vinp', 'dec', 10, 1, 1e9)},
                                         sig_sources_dict=sources)
    assert '.noise v(vout,0) Vvinp dec 10 1 1G 1' in out


def test_control_block_noise_inserts_setplot():
    net = PySpiceNetlister.__new__(PySpiceNetlister)
    out = net.generate_control_block('/out', noise=True)
    # setplot must come immediately before the write line
    assert 'setplot noise1\nwrite /out/rawfile.raw\n' in out


def test_control_block_default_unchanged():
    net = PySpiceNetlister.__new__(PySpiceNetlister)
    assert 'setplot' not in net.generate_control_block('/out')
    assert 'setplot' not in net.generate_control_block('/out', noise=False)


class _FakeVec:
    def __init__(self, name, data, type='voltage'):
        self.name = name
        self.data = np.asarray(data)
        self.type = type


class _FakeRaw:
    def __init__(self, vectors):
        self.vectors = vectors
        self.nvars = len(vectors)


def test_from_rawfile_loads_bare_noise_vector():
    raw = _FakeRaw([
        _FakeVec('frequency', [1.0, 10.0, 100.0], type='frequency'),
        _FakeVec('inoise_spectrum', [1e-5, 5e-6, 2e-6], type='voltage-density'),
        _FakeVec('onoise_spectrum', [1e-3, 5e-4, 2e-4], type='voltage-density'),
    ])
    sig = NgspiceSignal.from_rawfile(raw, 'inoise_spectrum')
    assert sig.xlabel == 'frequency'
    assert list(sig.x) == [1.0, 10.0, 100.0]
    assert list(sig.y) == [1e-5, 5e-6, 2e-6]


def test_from_rawfile_node_voltage_path_unchanged():
    raw = _FakeRaw([
        _FakeVec('time', [0.0, 1e-9, 2e-9], type='time'),
        _FakeVec('v(out)', [0.0, 0.5, 1.0], type='voltage'),
    ])
    sig = NgspiceSignal.from_rawfile(raw, 'out')
    assert list(sig.y) == [0.0, 0.5, 1.0]
