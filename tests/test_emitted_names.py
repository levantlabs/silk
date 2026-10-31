"""TASK-520: designer instance spelling versus emitted SPICE components."""
import pytest

from silk import sources, load_pdk
from silk.circuit.modules import Module
from silk.circuit.emitted_names import emitted_instance_name
from silk.circuit.spice_netlister import PySpiceNetlister
from silk.elements import elements, passives
from silk.exceptions import CircuitError


def _objects(name):
    pdk = load_pdk('sky130')
    return [
        (pdk.nmos1p8(name, d='d[0]', g='g[0]', s='0', b='0', w=1, l=.15), 'XM'),
        (elements.resistor(name, p='d[0]', n='0', value=1000), 'R'),
        (elements.capacitor(name, p='d[0]', n='0', value=1e-12), 'C'),
        (elements.inductor(name, p='d[0]', n='0', value=1e-6), 'L'),
        (passives.Resistor(name, 'd[0]', '0', 1000), 'R'),
        (sources.VoltageSource(name, positive='d[0]', negative='0', dc=1), 'V'),
        (sources.CurrentSource(name, positive='d[0]', negative='0', dc=.001), 'I'),
        (sources.VCVS(name, 'd[0]', '0', 'g[0]', '0', 2), 'E'),
        (sources.VCCS(name, 'd[0]', '0', 'g[0]', '0', .001), 'G'),
        (sources.CCVS(name, 'd[0]', '0', 1000, 'bias'), 'H'),
        (sources.CCCS(name, 'd[0]', '0', 2, 'bias'), 'F'),
        (sources.BehavioralVoltage(name, 'd[0]', '0', 'v(g[0]) < 1 ? 0 : 1'), 'B'),
        (sources.BehavioralCurrent(name, 'd[0]', '0', 'v(g[0]) > 1 ? 0 : 1'), 'B'),
    ]


@pytest.mark.parametrize('index', range(13))
def test_all_emitters_agree_with_emitted_instance_name(index):
    obj, prefix = _objects('MiX[0]')[index]
    # F/H standalone emission accepts a raw controlling source token.
    assert obj.to_spice().split()[0] == prefix + 'MiX<0>'
    assert emitted_instance_name(obj, obj.name) == prefix + 'MiX<0>'
    assert obj.name == 'MiX[0]'
    assert 'd[0]' in obj.to_spice()


@pytest.mark.parametrize('name', ['bad<0>', 'bad<', 'bad>'])
@pytest.mark.parametrize('index', range(13))
def test_renamed_instances_refuse_literal_angles(index, name):
    obj, _ = _objects('ok')[index]
    obj.name = name
    with pytest.raises(CircuitError, match='ASCII'):
        obj.to_spice()


def test_module_emitters_and_underscore_neighbor(tmp_path):
    child = Module('sky130', str(tmp_path), 'child.cir', pdk_details=load_pdk('sky130'), module_name='child')
    from silk.circuit.circuit_elements import PortDir
    child.define_module_ports({'p': PortDir.IN, 'n': PortDir.OUT})
    parent = Module('sky130', str(tmp_path), 'top.cir', pdk_details=load_pdk('sky130'))
    for name in ['X1[0]', 'X1_0']:
        parent.add_module(child, name, nets={'p': 'g[0]', 'n': '0'})
    from silk.circuit.module_placement import instance_line
    assert instance_line('X1[0]', parent.elem_dict['X1[0]']).startswith('xX1<0> ')
    assert instance_line('X1_0', parent.elem_dict['X1_0']).startswith('xX1_0 ')
    child.pin_list['p'].net = 'g[0]'
    child.pin_list['n'].net = '0'
    assert child.generate_netlist_entry('X1[0]').startswith('xX1<0> ')
    assert set(parent.elem_dict) == {'X1[0]', 'X1_0'}
    with pytest.raises(CircuitError, match='ASCII'):
        parent.add_module(child, 'X1<0>', nets={'p': 'g[0]', 'n': '0'})


def test_arrayed_pwl_physical_suffixes_and_internal_node():
    src = sources.VoltageSource('bias[0]', positive='out[0]', negative='0',
                                dc=.2, pwl=[(0, 0), (1e-9, 1)])
    lines = [line.split() for line in src.to_spice().splitlines()]
    assert lines[0][:3] == ['Vbias<0>__silk_pwl', 'out[0]', '__silk_src_Vbias[0]']
    assert lines[1][:3] == ['Vbias<0>__silk_dc', '__silk_src_Vbias[0]', '0']


def test_generated_control_quotes_each_complete_vector_argument(tmp_path):
    netlister = PySpiceNetlister(load_pdk('sky130'))
    address = '@m.xX1<0>.XMMA.m0[id]'
    text = netlister.generate_control_block(
        dir_folder=str(tmp_path), save_list=['xX1<0>.xY<1>.g[0]'],
        probe_save_lines=[address], save_device_params=True, op_save_lines=[address])
    saves = [line for line in text.splitlines() if line.startswith('save ')]
    assert saves == ['save "v(xX1<0>.xY<1>.g[0])"',
                     f'save "{address}"', f'save "{address}"']


def test_post_preflight_request_snapshot_excludes_suppressed_probes(tmp_path, monkeypatch):
    from silk.circuit.circuit_elements import PortDir
    from silk.simulation import Transient
    pdk = load_pdk('sky130')
    dut = Module('sky130', str(tmp_path), 'dut.cir', pdk, module_name='dut')
    dut.define_module_ports({'d': PortDir.OUT})
    dut.add(pdk.nmos1p8('M[0]', d='d', g='d', s='0', b='0', w=1, l=.15))
    tb = dut.create_test('check')
    tb.add_module_under_test({'d': 'd'})
    tb.add_sim_type(Transient(step=1e-9, stop=3e-9))
    tb.probe('M[0]:cgso')
    monkeypatch.setattr(PySpiceNetlister, '_ngspice_preflight_filter', lambda *args: [])
    spec = tb._prepare_run(corner='tt')
    spec.netlist.generate_netlist(corner=spec.corner_str)
    assert not spec.netlist._prepared_device_param_requests


@pytest.mark.parametrize('kind', ['factory', 'legacy', 'module', 'source', 'passive'])
@pytest.mark.parametrize('name', ['X<0>', 'X<', 'X>'])
def test_literal_angles_refused_at_registration(tmp_path, kind, name):
    parent = Module('sky130', str(tmp_path), 'top.cir', load_pdk('sky130'))
    with pytest.raises(CircuitError, match='ASCII'):
        if kind == 'factory':
            parent.add(elements.resistor(name, p='d', n='0', value=1000))
        elif kind == 'legacy':
            parent.add_element(parent.pdk_details.nmos1p8, name,
                               nets={'d': 'd', 'g': 'd', 's': '0', 'b': '0'},
                               parameters={'w': 1, 'l': .15})
        elif kind == 'module':
            child = Module('sky130', str(tmp_path), 'child.cir', load_pdk('sky130'), module_name='child')
            child.define_module_ports({})
            parent.add_module(child, name, nets={})
        elif kind == 'source':
            parent.add_source(sources.VoltageSource(name, positive='d', negative='0'))
        else:
            parent.add_passive(passives.Resistor(name, 'd', '0', 1000))


def test_authored_expression_and_bracketed_nets_are_unchanged():
    expression = 'v(a[0]) < 1 ? 0 : v(a[0]) > 2'
    source = sources.BehavioralVoltage('B[0]', 'out[0]', '0', expression)
    assert source.to_spice() == f'BB<0> out[0] 0 V={{{expression}}}\n'


def test_batch_device_probe_snapshot_matches_emitted_saves(tmp_path, monkeypatch):
    from silk.simulation import Transient
    netlister = PySpiceNetlister(load_pdk('sky130'))
    address = '@m.xX<0>.XMMA.m0[gm]'
    monkeypatch.setattr(netlister, '_resolve_device_probe_lines', lambda *a, **k: [address])
    text = netlister.generate_output_block(str(tmp_path),
        {'tran': Transient(step=1e-9, stop=3e-9)}, save_device_currents=True)
    assert address in text
    assert [(r[0], r[2]) for r in netlister._device_param_requests] == [('tran1', address)]


def test_mixed_noise_has_no_inferred_device_parameter_request():
    from silk.simulation import Transient, Noise
    from silk.simulation._device_requests import snapshot_requests
    address = '@m.xX<0>.XMMA.m0[gm]'
    requests = snapshot_requests({}, [], [address], {
        'tran': Transient(step=1e-9, stop=3e-9),
        'noise': Noise(output='d', ref=None, source='vg', variation='dec',
                       points=10, fstart=1e3, fstop=1e6)})
    assert [(r[0], r[2]) for r in requests] == [('tran1', address)]


def test_fresh_prepared_deck_recovers_requests_after_failed_attempt(tmp_path):
    from silk.simulation import Op
    from silk.circuit.modules_netlist import _require_fresh_deck
    pdk = load_pdk('sky130')
    top = Module('sky130', str(tmp_path), 'top.cir', pdk)
    top.add(pdk.nmos1p8('M[0]', d='d', g='d', s='0', b='0', w=1, l=.15))
    top.add_sim_type(Op())
    top.save_device_params = True
    top.generate_netlist()
    requested = top._prepared_device_param_requests
    assert requested
    # Result invalidation clears live proof even if the prepared deck survives.
    top._prepared_device_param_requests = ()
    _require_fresh_deck(top)
    assert top._prepared_device_param_requests == requested


@pytest.mark.parametrize('name', ['__silk_x', 'in__silk_dc', 'IN__SILK_DC',
                                  'vs[0]__silk_wave', 'vs_0__SiLk_pwl'])
def test_public_instance_reservation_matches_construction_and_add(tmp_path, name):
    import silk.circuit as circuit
    from silk.circuit.emitted_names import validate_instance_name
    public = getattr(circuit, 'validate_instance_name', None)
    assert public is validate_instance_name
    message = (f"Instance name {name!r} contains the reserved __silk_ prefix. "
               'The __silk_ prefix is reserved for silk-generated names; '
               'rename this instance.')
    with pytest.raises(CircuitError) as exc:
        public(name)
    assert str(exc.value) == message
    for index in range(13):
        mod = Module('sky130', str(tmp_path / str(index)), 'top.cir',
                     pdk_details=load_pdk('sky130'))
        obj, _ = _objects('ok')[index]
        obj.name = name
        with pytest.raises(CircuitError) as exc:
            mod.add(obj)
        assert str(exc.value) == message
        assert not mod.elem_dict and not mod.sig_sources_dict and not mod.net_dict
    with pytest.raises(CircuitError) as exc:
        sources.VoltageSource(name, positive='out', negative='0', dc=1)
    assert str(exc.value) == message
    with pytest.raises(CircuitError) as exc:
        elements.resistor(name, p='out', n='0', value=1000)
    assert str(exc.value) == message


@pytest.mark.parametrize('path', ['add_element', 'add_module', 'add_passive',
                                  'add_source', 'manager', 'placed_rename'])
def test_reserved_instance_add_paths_are_atomic(tmp_path, path):
    mod = Module('sky130', str(tmp_path), 'top.cir', pdk_details=load_pdk('sky130'))
    name = 'bad__SILK__silk_dc'
    if path in ('add_module', 'add_element'):
        child = Module('sky130', str(tmp_path), 'child.cir',
                       pdk_details=load_pdk('sky130'), module_name='child')
        child.define_module_ports({'p': 'in'})
        call = lambda: getattr(mod, path)(child, name, nets={'p': 'out'})
    elif path == 'add_passive':
        obj = passives.Resistor('ok', 'out', '0', 1000)
        obj.name = name
        call = lambda: mod.add_passive(obj)
    elif path == 'add_source':
        obj = sources.VoltageSource('ok', 'out', '0', dc=1)
        obj.name = name
        call = lambda: mod.add_source(obj)
    elif path == 'manager':
        call = lambda: mod._element_mgr.add_element(load_pdk('sky130').nmos1p8, name)
    else:
        obj = elements.resistor('ok', p='out', n='0', value=1000)
        mod.add(obj)
        # The public setter refuses corrupting a placed name before re-add.
        call = lambda: setattr(obj, 'name', name)
    before = (dict(mod.elem_dict), dict(mod.sig_sources_dict), dict(mod.net_dict))
    reason = 'read-only' if path == 'placed_rename' else 'reserved __silk_ prefix'
    with pytest.raises(CircuitError, match=reason):
        call()
    assert (mod.elem_dict, mod.sig_sources_dict, mod.net_dict) == before
    if path == 'placed_rename':
        assert obj.name == 'ok'
        mod.add(obj)
        assert mod.elem_dict['ok']['device'] is obj


@pytest.mark.parametrize('name', ['in', 'vs[0]', 'vs_0'])
@pytest.mark.parametrize('slot,suffix', [('pwl', 'pwl'), ('sine', 'wave'),
                                       ('waveform', 'wave')])
@pytest.mark.parametrize('cls,prefix', [(sources.VoltageSource, 'V'),
                                      (sources.CurrentSource, 'I')])
def test_reserved_cascade_cards_leave_authored_suffix_neighbor_distinct(name, slot, suffix, cls, prefix):
    wave = sources.Sine(0, .2, 1e6) if slot == 'sine' else sources.PWL([(0, 0), (1e-9, 1)])
    src = cls(name, 'out', '0', dc=1.2, **{slot: wave})
    neighbor_name = name.replace('[0]', '_dc[0]') if '[' in name else name + '_dc'
    neighbor = cls(neighbor_name, 'other', '0', dc=.5)
    neighbor_emitted = neighbor_name.replace('[', '<').replace(']', '>')
    emitted = name.replace('[', '<').replace(']', '>')
    cards = [line.split()[0] for line in (src.to_spice() + neighbor.to_spice()).splitlines()]
    assert cards == [prefix + emitted + '__silk_' + suffix,
                     prefix + emitted + '__silk_dc', prefix + neighbor_emitted]
    assert len(set(card.lower() for card in cards)) == 3


@pytest.mark.parametrize('name', ['', 'bad name', 'bad\tname', '__silk_y',
                                  '__SILK_y', 'xdut.__silk_y', 'xdut:__SiLk_y'])
def test_public_net_validation_matches_add_net(tmp_path, name):
    import silk.circuit as circuit
    public = getattr(circuit, 'validate_net_name', None)
    assert callable(public)
    mod = Module('sky130', str(tmp_path), 'top.cir', pdk_details=load_pdk('sky130'))
    with pytest.raises(ValueError) as public_exc:
        public(name)
    with pytest.raises(ValueError) as add_exc:
        mod.add_net(name)
    assert str(public_exc.value) == str(add_exc.value)
    assert not mod.net_dict


def test_public_net_validation_preserves_component_prefix_rule():
    from silk.circuit import validate_net_name
    for name in ['out[0]', 'xdut_out', 'in__silk_y', 'a_in__silk_y']:
        assert validate_net_name(name) is None
