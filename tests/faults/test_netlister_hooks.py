"""Netlister + Module integration tests for fault injection.

Covers:
  * ``Module._silk_fault_all_device_paths()`` — hierarchy enumeration.
  * ``Module.generate_netlist(fault_set=...)`` — end-to-end fault
    injection flowing through the netlister.
  * ``SubcircuitInstance`` exposes device handles with
    ``_silk_fault_target_path`` for the sugar resolver.
"""

import pytest

from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.faults import FaultSet, stuck_open


def _make_sub(pdk, tmp_path):
    """Single-device subcircuit used by the integration tests."""
    sub = Module('PDK_skywater130', str(tmp_path), 'sub.cir',
                 pdk_details=pdk, module_name='sub')
    sub.define_module_nets({'d': PortDir.INOUT, 'g': PortDir.INOUT,
                            's': PortDir.INOUT, 'b': PortDir.INOUT})
    with sub:
        pdk.nmos1p8('mn1', d='d', g='g', s='s', b='b',
                    w=1.0, l=0.18, m=1)
    return sub


def test_silk_fault_all_device_paths_walks_hierarchy(pdk, tmp_dir):
    sub = _make_sub(pdk, tmp_dir)
    top = Module('PDK_skywater130', tmp_dir, 'top.cir', pdk_details=pdk)
    for name in ('g1', 'd1', 'g2', 'd2', 'gnd'):
        top.add_net(name)
    top.add_module(sub, 'u1',
                    nets={'g': 'g1', 'd': 'd1', 's': 'gnd', 'b': 'gnd'})
    top.add_module(sub, 'u2',
                    nets={'g': 'g2', 'd': 'd2', 's': 'gnd', 'b': 'gnd'})

    paths = top._silk_fault_all_device_paths()
    assert 'xu1.mn1' in paths
    assert 'xu2.mn1' in paths
    assert len(paths) == 2


def test_generate_netlist_accepts_fault_set_kwarg(pdk, tmp_dir):
    sub = _make_sub(pdk, tmp_dir)
    top = Module('PDK_skywater130', tmp_dir, 'top.cir', pdk_details=pdk)
    for name in ('g1', 'd1', 'gnd'):
        top.add_net(name)
    top.add_module(sub, 'u1',
                    nets={'g': 'g1', 'd': 'd1', 's': 'gnd', 'b': 'gnd'})

    fs = FaultSet([stuck_open('xu1.mn1')])
    top_deck_path = top.generate_netlist(fault_set=fs)
    with open(top_deck_path) as f:
        text = f.read()
    assert 'FAULT: stuck_open on xu1.mn1' in text
    bodies, _ = _definitions(text)
    assert 'XMmn1' not in bodies['__silk_fault_sub_787531']
    assert 'XMmn1' in bodies['sub']


def test_preparation_threads_owning_netlister_pdk_into_realization(pdk, tmp_dir):
    """Fault bounds use the owning module's netlister catalog at preparation."""
    from unittest.mock import patch

    import silk.faults.application as application

    sub = _make_sub(pdk, tmp_dir)
    top = Module('PDK_skywater130', tmp_dir, 'top.cir', pdk_details=pdk)
    for name in ('g1', 'd1', 'gnd'):
        top.add_net(name)
    top.add_module(sub, 'u1',
                    nets={'g': 'g1', 'd': 'd1', 's': 'gnd', 'b': 'gnd'})

    fs = FaultSet([stuck_open('xu1.mn1')])
    with patch.object(application, '_realize_fault',
                      wraps=application._realize_fault) as spy:
        top.generate_netlist(fault_set=fs)

    assert spy.call_count == 1
    _, kwargs = spy.call_args
    # Resolve the catalog through the actual placed owner, as emission does.
    threaded_pdk = kwargs.get('pdk')
    assert threaded_pdk is top.elem_dict['u1']['device'].netlister.pdk_details
    assert threaded_pdk is not None
    assert threaded_pdk._devices['nmos1p8'].fault_params.w.min == \
        pdk._devices['nmos1p8'].fault_params.w.min == 0.42


def test_w_variation_clamps_through_the_real_netlister_with_real_pdk(pdk, tmp_dir):
    """The clamping ``w_variation`` uses once the PDK is threaded in,
    driven through the real ``generate_netlist(fault_set=...)`` path on a
    factory-built device: the manifest records the sky130 bound and the
    emitted line carries it."""
    from silk.faults import w_variation

    sub = Module('PDK_skywater130', tmp_dir, 'sub_clamp.cir',
                 pdk_details=pdk, module_name='sub')
    sub.define_module_nets({'d': PortDir.INOUT, 'g': PortDir.INOUT,
                            's': PortDir.INOUT, 'b': PortDir.INOUT})
    with sub:
        pdk.nmos1p8('mn1', d='d', g='g', s='s', b='b', w=0.42, l=0.18, m=1)
    top = Module('PDK_skywater130', tmp_dir, 'top_clamp.cir', pdk_details=pdk)
    for name in ('g1', 'd1', 'gnd'):
        top.add_net(name)
    top.add_module(sub, 'u1',
                    nets={'g': 'g1', 'd': 'd1', 's': 'gnd', 'b': 'gnd'})

    manifest = []
    top_deck_path = top.generate_netlist(
        fault_set=FaultSet([w_variation('xu1.mn1', sigma=10000.0)]),
        manifest=manifest, seed=42)
    assert len(manifest) == 1
    rp = manifest[0].realized_params
    assert rp['clamped'] is True
    assert rp['w_new'] in (0.42, 100.0)
    with open(top_deck_path) as f:
        bodies, _ = _definitions(f.read())
        line = next(ln for ln in bodies['__silk_fault_sub_787531'].splitlines()
                    if ln.startswith('XMmn1 '))
    assert f'w={rp["w_new"]:g}' in line.split()


def test_subcircuit_instance_exposes_device_fault_target(pdk, tmp_dir):
    sub = _make_sub(pdk, tmp_dir)
    top = Module('PDK_skywater130', tmp_dir, 'top.cir', pdk_details=pdk)
    for name in ('g1', 'd1', 'gnd'):
        top.add_net(name)
    u1 = top.add_module(sub, 'u1',
                         nets={'g': 'g1', 'd': 'd1',
                               's': 'gnd', 'b': 'gnd'})
    mn1_handle = u1.mn1
    assert getattr(mn1_handle, '_silk_fault_target_path', None) == 'xu1.mn1'


def test_w_variation_on_factory_device_emits_through_real_netlister(pdk, tmp_dir):
    """TASK-31 regression: a re-emitting fault on a factory-built device.

    short_ds (like w_variation/l_variation/vth_shift) re-emits the device;
    it once crashed through a device adapter that lacked ``spice_prefix``.
    Since TASK-415 the re-emitted device is a copy of the element, written
    by its own ``to_spice()``. Drive one through the REAL
    generate_netlist(fault_set=...) path and assert on the line.
    """
    from silk.faults import short_ds

    sub = _make_sub(pdk, tmp_dir)
    top = Module('PDK_skywater130', tmp_dir, 'top_short.cir', pdk_details=pdk)
    for name in ('g1', 'd1', 'gnd'):
        top.add_net(name)
    top.add_module(sub, 'u1',
                    nets={'g': 'g1', 'd': 'd1', 's': 'gnd', 'b': 'gnd'})

    fs = FaultSet([short_ds('xu1.mn1')])
    top_deck_path = top.generate_netlist(fault_set=fs)
    with open(top_deck_path) as f:
        text = f.read()
    # The faulted device line is re-emitted with the declared prefix, and
    # the short's shunt resistor rides alongside it.
    assert 'XMmn1' in text
    assert 'R__silk_fault_short_' in text


_RE_EMITTING_FAULTS = [
    # (sugar name, kwargs, marker lines expected alongside the re-emitted
    #  device line).  Every one of these re-emits the device through its
    #  own writer (TASK-415).
    ('short_ds',    {},                ['FAULT: short_ds on xu1.mn1', 'R__silk_fault_short_']),
    ('short_gs',    {},                ['FAULT: short_gs on xu1.mn1', 'R__silk_fault_short_']),
    ('short_gd',    {},                ['FAULT: short_gd on xu1.mn1', 'R__silk_fault_short_']),
    ('w_variation', {'sigma': 0.05},   ['FAULT: w_variation on xu1.mn1']),
    ('l_variation', {'sigma': 0.05},   ['FAULT: l_variation on xu1.mn1']),
    ('vth_shift',   {'delta': 0.02},   ['FAULT: vth_shift on xu1.mn1',
                                        'E__silk_fault_vt_7875312e6d6e31 __silk_fault_shifted_7875312e6d6e31 __silk_fault_offset_7875312e6d6e31 g 0 1',
                                        'V__silk_fault_vt_7875312e6d6e31 __silk_fault_offset_7875312e6d6e31 0 DC 0.02']),
]


@pytest.mark.parametrize('sugar_name, kwargs, markers', _RE_EMITTING_FAULTS,
                         ids=[row[0] for row in _RE_EMITTING_FAULTS])
def test_every_re_emitting_fault_type_survives_real_netlister_on_factory_device(
        pdk, tmp_dir, sugar_name, kwargs, markers):
    """TASK-30 regression: all six re-emitting fault types (AC#1) drive
    through the REAL ``Module.generate_netlist(fault_set=...)`` path on a
    factory-built (``pdk.nmos1p8``) device and emit a prefixed device line
    (they once raised AttributeError through a device adapter).
    """
    import silk.faults as faults

    sugar = getattr(faults, sugar_name)
    sub = _make_sub(pdk, tmp_dir)
    top = Module('PDK_skywater130', tmp_dir, f'top_{sugar_name}.cir',
                 pdk_details=pdk)
    for name in ('g1', 'd1', 'gnd'):
        top.add_net(name)
    top.add_module(sub, 'u1',
                    nets={'g': 'g1', 'd': 'd1', 's': 'gnd', 'b': 'gnd'})

    fs = FaultSet([sugar('xu1.mn1', **kwargs)])
    top_deck_path = top.generate_netlist(fault_set=fs)
    with open(top_deck_path) as f:
        text = f.read()

    bodies, _ = _definitions(text)
    faulted = bodies['__silk_fault_sub_787531']
    device_lines = [ln for ln in faulted.splitlines() if ln.startswith('XMmn1 ')]
    assert len(device_lines) == 1, text
    # Re-emitted with the catalog prefix, the factory pins, and the model.
    assert device_lines[0].split()[0] == 'XMmn1'
    assert 'sky130_fd_pr__nfet_01v8' in device_lines[0]
    for marker in markers:
        assert marker in faulted, (marker, faulted)


_FAULT_ATTRS = ('_silk_pending_fault', '_silk_fault_rng', '_silk_fault_sink',
                '_silk_fault_realization')


def _authored_fault_state(module):
    """Snapshot authored devices and registrations, including shared definitions."""
    seen = set()
    state = []

    def visit(mod):
        if id(mod) in seen:
            return
        seen.add(id(mod))
        state.append((id(mod), mod.module_name, tuple(mod.elem_dict), tuple(mod.module_dict),
                      tuple((name, entry['device'].module_name,
                             tuple((p, pin.net) for p, pin in entry.get('pin_list', {}).items()))
                            for name, entry in mod.elem_dict.items()
                            if isinstance(entry['device'], Module))))
        for entry in mod.elem_dict.values():
            dev = entry['device']
            if isinstance(dev, Module):
                visit(dev)
            else:
                assert not any(hasattr(dev, attr) for attr in _FAULT_ATTRS)
                state.append((id(dev), dict(dev.params), dict(dev.nets)))
        for entry in mod.module_dict.values():
            visit(entry['device'])

    visit(module)
    return state


def _cleanup_circuit(pdk, tmp_path, nested):
    sub = _make_sub(pdk, tmp_path)
    if not nested:
        sub.module_name = None
        sub.circuit_name = 'fault cleanup'
        return sub, sub, 'mn1'
    top = Module('PDK_skywater130', str(tmp_path), 'top.cir', pdk_details=pdk)
    top.circuit_name = 'fault cleanup'
    for name in ('d', 'g', 's', 'b'):
        top.add_net(name)
    for name in ('u1', 'u2'):
        top.add_module(sub, name, nets={n: n for n in ('d', 'g', 's', 'b')})
    top._silk_resolve_fault_target('xu1.mn1')
    return top, top.module_dict['sub']['device'], 'xu1.mn1'


def _read_cleanup_deck(mod):
    from pathlib import Path
    mod.set_output_options(mode='overwrite')
    return Path(mod.generate_netlist()).read_bytes()


@pytest.mark.parametrize('nested', [False, True])
def test_failed_target_preparation_preserves_same_authored_graph(pdk, tmp_path, nested):
    from silk.faults import short_gd
    mod, definition, target = _cleanup_circuit(pdk, tmp_path, nested)
    baseline = _read_cleanup_deck(mod)
    subckt = definition.generate_netlist(subckt=True) if nested else None
    before = _authored_fault_state(mod)
    manifest = []
    with pytest.raises(KeyError, match='missing'):
        mod.generate_netlist(
            fault_set=FaultSet([short_gd(target), stuck_open('missing')]),
            manifest=manifest)
    assert _read_cleanup_deck(mod) == baseline
    if nested:
        assert definition.generate_netlist(subckt=True) == subckt
    assert _authored_fault_state(mod) == before
    assert manifest == []


@pytest.mark.parametrize('nested', [False, True])
def test_successful_fault_then_nominal_preserves_same_authored_graph(pdk, tmp_path, nested):
    from pathlib import Path
    mod, definition, target = _cleanup_circuit(pdk, tmp_path, nested)
    baseline = _read_cleanup_deck(mod)
    before = _authored_fault_state(mod)
    manifest = []
    mod.generate_netlist(fault_set=FaultSet([stuck_open(target)]), manifest=manifest)
    assert f'FAULT: stuck_open on {target}' in Path(mod._run_files.netlist_path).read_text()
    assert manifest
    assert _authored_fault_state(mod) == before
    assert _read_cleanup_deck(mod) == baseline


@pytest.mark.parametrize('subckt', [False, True])
@pytest.mark.parametrize('prepopulated', [False, True])
def test_emission_failure_rolls_back_manifest_in_place(pdk, tmp_path, monkeypatch,
                                                       subckt, prepopulated):
    import silk.circuit.modules_netlist as netlist
    import silk.simulation.completion as completion
    mod, definition, target = _cleanup_circuit(pdk, tmp_path, True)
    if subckt:
        mod, target = definition, 'mn1'
    baseline = _read_cleanup_deck(mod) if not subckt else mod.generate_netlist(subckt=True)
    before = _authored_fault_state(mod)
    earlier = object()
    manifest = [earlier] if prepopulated else []
    original = manifest
    entry = list(manifest)
    error = RuntimeError('injected after fault recording')
    grew = []
    real_subckt = netlist._subckt_text

    def fail(*args, **kwargs):
        if subckt:
            real_subckt(*args, **kwargs)
        assert len(manifest) > len(entry)
        grew.append(True)
        raise error

    with monkeypatch.context() as patcher:
        patcher.setattr(netlist if subckt else completion,
                        '_subckt_text' if subckt else 'certify_generated', fail)
        for _ in range(2):
            with pytest.raises(RuntimeError) as caught:
                mod.generate_netlist(subckt=True if subckt else None,
                                     fault_set=FaultSet([stuck_open(target)]),
                                     manifest=manifest)
            assert caught.value is error
            assert _authored_fault_state(mod) == before
            assert manifest is original
            assert manifest == entry
    assert grew == [True, True]
    clean = mod.generate_netlist(subckt=True) if subckt else _read_cleanup_deck(mod)
    assert clean == baseline


def test_fixed_seed_selection_and_independent_shared_placements(pdk, tmp_path):
    from pathlib import Path
    from silk.faults import w_variation
    mod, definition, target = _cleanup_circuit(pdk, tmp_path, True)
    mod.set_output_options(mode='overwrite')
    with mod:
        pdk.nmos1p8('mn2', d='d', g='g', s='s', b='b', w=2.0, l=0.18)
    faults = FaultSet([w_variation('xu2.mn1', sigma=0.1)])
    faults.sprinkle(r'.*', w_variation(sigma=0.2), rate=0.8)
    decks, manifests = [], []
    for _ in range(2):
        manifest = []
        mod.generate_netlist(fault_set=faults, seed=7, manifest=manifest)
        manifests.append(manifest)
        decks.append(Path(mod._run_files.netlist_path).read_bytes())
    assert decks[0] == decks[1]
    assert [f.target for f in faults._last_realized] == ['xu2.mn1', 'xu1.mn1', 'mn2']
    bodies, top = _definitions(decks[0].decode())
    assert 'FAULT:' not in bodies['sub']
    for target in ('xu1.mn1', 'xu2.mn1'):
        generated = '__silk_fault_sub_' + target.split('.')[0].encode().hex()
        assert f'FAULT: w_variation on {target}' in bodies[generated]
        assert f"{target.split('.')[0]} d g s b {generated}" in top
    assert 'FAULT: w_variation on mn2' in top
    for target in ('xu2.mn1', 'xu1.mn1', 'mn2'):
        body = (top if target == 'mn2' else
                bodies['__silk_fault_sub_' + target.split('.')[0].encode().hex()])
        assert sum(r.target == target for r in manifests[0]) == 1
        assert sum(r.target == target for r in manifests[1]) == 1
        first = next(r for r in manifests[0] if r.target == target)
        again = next(r for r in manifests[1] if r.target == target)
        assert first.realized_params == again.realized_params
        assert first.netlist_comment in body
        local = target.split('.')[-1]
        line = next(l for l in body.splitlines() if l.startswith('XM' + local + ' '))
        assert f'w={first.realized_params["w_new"]:g}' in line.split()
    assert _authored_fault_state(mod)


def _definitions(text):
    """Split a flat emitted deck into named bodies and top-level text."""
    bodies, top, body = {}, [], None
    for line in text.splitlines():
        if line.lower().startswith('.subckt '):
            name = line.split()[1]
            assert name not in bodies
            body = bodies.setdefault(name, [])
        elif line.lower().startswith('.ends'):
            body = None
        elif body is None:
            top.append(line)
        else:
            body.append(line)
    return {name: '\n'.join(lines) for name, lines in bodies.items()}, '\n'.join(top)


def _placement_pair(pdk, tmp_path, *, distinct=False, indexed=False, nested=False):
    leaf = _make_sub(pdk, tmp_path)
    pair = Module('sky130', str(tmp_path), 'pair.cir', pdk_details=pdk,
                  module_name='pair' if nested else None)
    ports = ('d', 'g', 's', 'b')
    if nested:
        pair.define_module_ports({n: 'inout' for n in ports})
    else:
        for n in ports:
            pair.add_net(n)
    names = ('u[0]', 'u[1]') if indexed else ('left', 'right')
    for index, name in enumerate(names):
        child = _make_sub(pdk, tmp_path) if distinct and index else leaf
        pair.add_module(child, name, nets={n: n for n in ports})
    if not nested:
        return pair, names
    top = Module('sky130', str(tmp_path), 'top.cir', pdk_details=pdk)
    for n in ports:
        top.add_net(n)
    for name in ('dut', 'other'):
        top.add_module(pair, name, nets={n: n for n in ports})
    return top, names


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('distinct', [False, True])
@pytest.mark.parametrize('indexed', [False, True])
@pytest.mark.parametrize('both', [False, True])
def test_shared_placements_have_independent_faulted_definitions(
        pdk, tmp_path, simulator, distinct, indexed, both):
    from pathlib import Path
    from silk.faults import w_variation
    mod, names = _placement_pair(pdk, tmp_path, distinct=distinct, indexed=indexed)
    mod.simulator = simulator
    baseline = _read_cleanup_deck(mod)
    before = _authored_fault_state(mod)
    targets = [f'x{name}.mn1' for name in names[:2 if both else 1]]
    faults = FaultSet([w_variation(t, sigma=0.1) for t in targets] +
                      [stuck_open(targets[0])])
    manifest = []
    text = Path(mod.generate_netlist(fault_set=faults, manifest=manifest,
                                    seed=7)).read_text()
    bodies, top = _definitions(text)
    healthy = bodies['sub']
    assert 'FAULT:' not in healthy
    assert 'XMmn1 d g s b sky130_fd_pr__nfet_01v8 m=1 w=1 l=0.18' in healthy
    for index, name in enumerate(names):
        emitted = 'x' + name.replace('[', '<').replace(']', '>')
        instance = next(l.split() for l in top.splitlines() if l.startswith(emitted + ' '))
        assert instance[1:-1] == ['d', 'g', 's', 'b']
        if index < len(targets):
            target = targets[index]
            generated = '__silk_fault_sub_' + ('x' + name).encode().hex()
            assert instance[-1] == generated
            body = bodies[generated]
            assert f'FAULT: w_variation on {target}' in body
            assert 'stuck_open' not in body
            assert sum(r.target == target for r in manifest) == 1
            record = next(r for r in manifest if r.target == target)
            assert record.fault_type == 'w_variation'
            line = next(l for l in body.splitlines() if l.startswith('XMmn1 '))
            assert line.split()[1:6] == ['d', 'g', 's', 'b', 'sky130_fd_pr__nfet_01v8']
            assert f'w={record.realized_params["w_new"]:g}' in line.split()
        else:
            assert instance[-1] == 'sub'
    assert _authored_fault_state(mod) == before
    assert _read_cleanup_deck(mod) == baseline


def test_nested_fault_routes_only_targeted_ancestry(pdk, tmp_path):
    from pathlib import Path
    mod, _ = _placement_pair(pdk, tmp_path, nested=True)
    baseline = _read_cleanup_deck(mod)
    before = _authored_fault_state(mod)
    manifest = []
    text = Path(mod.generate_netlist(
        fault_set=FaultSet([stuck_open('xdut.xleft.mn1')]), manifest=manifest)).read_text()
    bodies, top = _definitions(text)
    parent = '__silk_fault_pair_' + b'xdut'.hex()
    child = '__silk_fault_sub_' + b'xdut.xleft'.hex()
    assert f'xdut d g s b {parent}' in top
    assert 'xother d g s b pair' in top
    assert f'xleft d g s b {child}' in bodies[parent]
    assert 'xright d g s b sub' in bodies[parent]
    assert 'XMmn1 ' not in bodies[child]
    assert 'FAULT: stuck_open on xdut.xleft.mn1' in bodies[child]
    assert next(r for r in manifest if r.target == 'xdut.xleft.mn1').fault_type == 'stuck_open'
    assert 'XMmn1 ' in bodies['sub']
    assert 'xleft d g s b sub' in bodies['pair']
    for body in [top, *bodies.values()]:
        for line in body.splitlines():
            if line.startswith('x'):
                assert line.split()[-1] in bodies
    assert _authored_fault_state(mod) == before
    assert _read_cleanup_deck(mod) == baseline


@pytest.mark.parametrize('nested', [False, True])
def test_faulted_preparation_is_not_reusable(pdk, tmp_path, nested):
    mod, _, target = _cleanup_circuit(pdk, tmp_path, nested)
    mod.generate_netlist(fault_set=FaultSet([stuck_open(target)]))
    assert mod._deck_preparation is None
    mod.generate_netlist(fault_set=FaultSet([]))
    assert mod._deck_preparation is not None
    mod.generate_netlist()
    assert mod._deck_preparation is not None


@pytest.mark.parametrize('phase', ['child', 'normalized'])
def test_fault_overlay_failure_restores_clean_deck(pdk, tmp_path, monkeypatch, phase):
    import silk.circuit.modules_netlist as netlist
    mod, _ = _placement_pair(pdk, tmp_path, nested=True)
    baseline = _read_cleanup_deck(mod)
    before = _authored_fault_state(mod)
    earlier = object()
    manifest = [earlier]
    real = netlist._subckt_text if phase == 'child' else netlist._quiet_deck_text

    def fail(*args, **kwargs):
        if phase == 'child' and not (getattr(args[0].netlister, '_subckt_name', '') or '').startswith('__silk_fault_'):
            return real(*args, **kwargs)
        real(*args, **kwargs)
        assert len(manifest) > 1
        raise RuntimeError('injected overlay failure')

    with monkeypatch.context() as patcher:
        patcher.setattr(netlist, '_subckt_text' if phase == 'child' else '_quiet_deck_text', fail)
        with pytest.raises(RuntimeError, match='injected overlay failure'):
            mod.generate_netlist(fault_set=FaultSet([stuck_open('xdut.xleft.mn1')]),
                                 manifest=manifest)
    assert manifest == [earlier]
    assert _authored_fault_state(mod) == before
    assert _read_cleanup_deck(mod) == baseline


@pytest.mark.parametrize('collision', ['placed', 'legacy', 'root'])
def test_generated_names_avoid_whole_definition_namespace(pdk, tmp_path, collision):
    from pathlib import Path
    mod, _ = _placement_pair(pdk, tmp_path)
    stem = '__silk_fault_sub_' + b'xleft'.hex()
    other = _make_sub(pdk, tmp_path)
    other.module_name = stem.upper()
    if collision == 'placed':
        mod.add_module(other, 'extra', nets={n: n for n in ('d', 'g', 's', 'b')})
    elif collision == 'legacy':
        # Legacy organizational registration without a placed element entry.
        mod._element_mgr.module_dict[stem.upper()] = {'device': other, 'defined': False}
    else:
        mod.module_name = stem.upper()
        mod.define_module_ports({n: 'inout' for n in ('d', 'g', 's', 'b')})
    baseline = mod.generate_netlist(subckt=True) if collision == 'root' else _read_cleanup_deck(mod)
    manifest = []
    result = mod.generate_netlist(fault_set=FaultSet([stuck_open('xleft.mn1')]), manifest=manifest)
    text = result if collision == 'root' else Path(result).read_text()
    bodies, top = _definitions(text)
    assert stem + '_1' in bodies
    assert next(r for r in manifest if r.target == 'xleft.mn1').netlist_comment in bodies[stem + '_1']
    assert 'XMmn1 ' not in bodies[stem + '_1']
    assert stem.upper() in bodies
    owner = bodies[stem.upper()] if collision == 'root' else top
    assert f'xleft d g s b {stem}_1' in owner
    clean = mod.generate_netlist(subckt=True) if collision == 'root' else _read_cleanup_deck(mod)
    assert clean == baseline


def test_generated_name_uses_resolved_variant_name(pdk, tmp_path):
    from pathlib import Path
    mod, _ = _placement_pair(pdk, tmp_path)
    mod.elem_dict['left']['device'].variants = True
    healthy_name = next(iter(mod.module_dict))
    assert healthy_name.startswith('sub_')
    manifest = []
    text = Path(mod.generate_netlist(fault_set=FaultSet([stuck_open('xleft.mn1')]),
                                    manifest=manifest)).read_text()
    bodies, top = _definitions(text)
    name = '__silk_fault_' + healthy_name + '_' + b'xleft'.hex()
    assert name in bodies
    assert next(r for r in manifest if r.target == 'xleft.mn1').netlist_comment in bodies[name]
    assert 'XMmn1 ' not in bodies[name]
    assert f'xleft d g s b {name}' in top
    assert f'xright d g s b {healthy_name}' in top
    assert mod.elem_dict['left']['device'].module_name == 'sub'


def test_multiple_targets_reuse_each_affected_containing_definition(pdk, tmp_path):
    from pathlib import Path
    from silk.faults import w_variation
    mod, _ = _placement_pair(pdk, tmp_path, nested=True)
    leaf = mod.elem_dict['dut']['device'].elem_dict['left']['device']
    leaf.add(pdk.nmos1p8('mn2', d='d', g='g', s='s', b='b', w=2, l=.18))
    targets = ['xdut.xleft.mn1', 'xdut.xleft.mn2', 'xdut.xright.mn1']
    manifest = []
    text = Path(mod.generate_netlist(
        fault_set=FaultSet([w_variation(t, sigma=.1) for t in targets]),
        manifest=manifest, seed=5)).read_text()
    bodies, top = _definitions(text)
    generated = [n for n in bodies if n.startswith('__silk_fault_')]
    assert set(generated) == {'__silk_fault_pair_' + b'xdut'.hex(),
                              '__silk_fault_sub_' + b'xdut.xleft'.hex(),
                              '__silk_fault_sub_' + b'xdut.xright'.hex()}
    for target in targets:
        ancestry, local = target.rsplit('.', 1)
        body = bodies['__silk_fault_sub_' + ancestry.encode().hex()]
        record = next(r for r in manifest if r.target == target)
        assert record.netlist_comment in body
        line = next(l for l in body.splitlines() if l.startswith('XM' + local + ' '))
        assert f'w={record.realized_params["w_new"]:g}' in line.split()
    assert 'xother d g s b pair' in top
    assert 'FAULT:' not in bodies['sub']


@pytest.mark.parametrize('indexed', [False, True])
def test_xyce_fault_added_ground_reports_target_ancestry_and_cleans_up(
        pdk, tmp_path, monkeypatch, indexed):
    from silk.exceptions import CircuitError
    from silk.faults import application
    mod, names = _placement_pair(pdk, tmp_path, nested=True, indexed=indexed)
    mod.simulator = 'xyce'
    baseline = _read_cleanup_deck(mod)
    before = _authored_fault_state(mod)
    target = 'xdut.x' + names[0] + '.mn1'
    expected = 'xdut.x' + names[0] + '.gnd'
    manifest = [object()]
    previous = list(manifest)
    real = application._realize_fault

    def realize(*args, **kwargs):
        result = real(*args, **kwargs)
        result.added.append(application._AddedElement(
            kind='resistor', name='extra', nets={'p': 'd', 'n': 'gnd'}, value=1000))
        return result

    with monkeypatch.context() as patcher:
        patcher.setattr(application, '_realize_fault', realize)
        for _ in range(2):
            with pytest.raises(CircuitError) as caught:
                mod.generate_netlist(fault_set=FaultSet([stuck_open(target)]),
                                     manifest=manifest)
            assert repr(expected) in str(caught.value)
            assert '__silk_fault_' not in str(caught.value)
            assert manifest == previous
            assert _authored_fault_state(mod) == before
    assert _read_cleanup_deck(mod) == baseline


@pytest.mark.parametrize('with_manifest', [False, True])
@pytest.mark.parametrize('api', ['factory', 'stored'])
def test_prepared_fault_writes_do_not_draw_or_record(pdk, tmp_path, monkeypatch,
                                                    with_manifest, api):
    import copy
    import numpy as np
    from silk.circuit.module_placement import _fault_node
    from silk.circuit.modules_faults import _prepare_fault_definitions
    from silk.faults import vth_shift
    import silk.faults.application as application

    mod, _, target = _cleanup_circuit(pdk, tmp_path, True)
    if api == 'stored':
        import warnings
        sub = Module('sky130', str(tmp_path), 'stored.cir',
                     pdk_details=pdk, module_name='stored')
        sub.define_module_ports({n: 'inout' for n in ('d', 'g', 's', 'b')})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            sub.add_element(pdk.nmos1p8, 'mn1',
                            nets={n: n for n in ('d', 'g', 's', 'b')},
                            parameters={'w': 1, 'l': .18})
        mod = Module('sky130', str(tmp_path), 'top.cir', pdk_details=pdk)
        mod.add_module(sub, 'u1', nets={n: n for n in ('d', 'g', 's', 'b')})
    rng = np.random.default_rng(7)
    manifest = [] if with_manifest else None
    calls = []
    realize = application._realize_fault

    def observe(*args, **kwargs):
        calls.append(args[1].target)
        return realize(*args, **kwargs)

    monkeypatch.setattr(application, '_realize_fault', observe)
    marked = mod._silk_apply_fault_markers(
        FaultSet([vth_shift(target, delta=.05, sigma=.02), stuck_open(target)]),
        rng, manifest)
    assert calls == [target]
    if manifest is not None:
        assert [r.target for r in manifest] == [target]
    state = copy.deepcopy(rng.bit_generator.state)
    prefix = list(manifest) if manifest is not None else None
    token = _fault_node.set(_prepare_fault_definitions(mod, marked))
    try:
        first = mod.netlister.generate_device_instances(mod.elem_dict)
        # Render definitions as well as the top-level instance cards.
        first += mod.netlister.generate_module_definitions(mod.module_dict)
        second = mod.netlister.generate_device_instances(mod.elem_dict)
        second += mod.netlister.generate_module_definitions(mod.module_dict)
        assert first == second
        assert 'FAULT: vth_shift' in first
        assert calls == [target]
        assert rng.bit_generator.state == state
        assert manifest == prefix
    finally:
        _fault_node.reset(token)
        mod._silk_clear_fault_markers(marked)
    assert all(not hasattr(d, attr) for d in marked for attr in _FAULT_ATTRS)
    assert not any(hasattr(mod._silk_resolve_fault_target(target), attr)
                   for attr in _FAULT_ATTRS)


@pytest.mark.parametrize('subckt', [False, True])
def test_second_realization_failure_rolls_back_and_retries(pdk, tmp_path,
                                                          monkeypatch, subckt):
    import silk.faults.application as application
    mod, _, target = _cleanup_circuit(pdk, tmp_path, True)
    if subckt:
        mod.module_name = 'outer'
        mod.define_module_ports({n: 'inout' for n in ('d', 'g', 's', 'b')})
    before = _authored_fault_state(mod)
    nominal = mod.generate_netlist(subckt=True) if subckt else _read_cleanup_deck(mod)
    earlier = object()
    manifest = [earlier]
    original = manifest
    realize = application._realize_fault
    faults = FaultSet([stuck_open(target), stuck_open('xu2.mn1')])
    calls = []

    def fail_second(device, fault, rng, **kwargs):
        calls.append(fault.target)
        if len(calls) == 2:
            assert len(manifest) == 2
            raise RuntimeError('second realization sentinel')
        return realize(device, fault, rng, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(application, '_realize_fault', fail_second)
        with pytest.raises(RuntimeError, match='second realization sentinel'):
            mod.generate_netlist(subckt=subckt, fault_set=faults, manifest=manifest)
    assert calls == [target, 'xu2.mn1']
    assert manifest is original
    assert manifest == [earlier]
    assert _authored_fault_state(mod) == before
    clean = mod.generate_netlist(subckt=True) if subckt else _read_cleanup_deck(mod)
    assert clean == nominal
    mod.generate_netlist(subckt=subckt, fault_set=faults, manifest=manifest)
    assert [r.target for r in manifest[1:]] == [target, 'xu2.mn1']
    assert _authored_fault_state(mod) == before


def test_preparation_uses_each_owning_netlister_catalog(pdk, tmp_path, monkeypatch):
    from copy import deepcopy
    from pathlib import Path
    from silk.faults import w_variation
    import silk.faults.application as application

    # The very same authored element belongs to two different module contexts.
    shared = pdk.nmos1p8('mn', d='d', g='g', s='s', b='b', w=1, l=.18)
    top = Module('sky130', str(tmp_path), 'contexts.cir', pdk_details=pdk)
    for n in ('d', 'g', 's', 'b'):
        top.add_net(n)
    owners = []
    for name, minimum in [('left', 2.0), ('right', 3.0)]:
        catalog = deepcopy(pdk)
        catalog._devices['nmos1p8'].fault_params.w.min = minimum
        child = Module('sky130', str(tmp_path), name + '.cir',
                       pdk_details=catalog, module_name=name)
        child.define_module_ports({n: 'inout' for n in ('d', 'g', 's', 'b')})
        child.add(shared)
        top.add_module(child, name, nets={n: n for n in ('d', 'g', 's', 'b')})
        owners.append(child)
    assert owners[0].elem_dict['mn']['device'] is shared
    assert owners[1].elem_dict['mn']['device'] is shared
    calls = []
    realize = application._realize_fault

    def observe(device, fault, rng, *, pdk):
        calls.append((fault.target, pdk))
        return realize(device, fault, rng, pdk=pdk)

    monkeypatch.setattr(application, '_realize_fault', observe)
    manifest = []
    text = Path(top.generate_netlist(fault_set=FaultSet([
        w_variation('xright.mn', sigma=0), w_variation('xleft.mn', sigma=0)]),
        manifest=manifest)).read_text()
    assert calls == [('xright.mn', owners[1].netlister.pdk_details),
                     ('xleft.mn', owners[0].netlister.pdk_details)]
    assert [r.realized_params['w_new'] for r in manifest] == [3, 2]
    bodies, _ = _definitions(text)
    assert 'w=2 ' in bodies['__silk_fault_left_786c656674']
    assert 'w=3 ' in bodies['__silk_fault_right_787269676874']
    assert shared.w == 1
    assert not any(hasattr(shared, attr) for attr in _FAULT_ATTRS)


def test_delegating_replacement_netlister_supplies_realization_context(pdk, tmp_path,
                                                                     monkeypatch):
    from copy import deepcopy
    from pathlib import Path
    from silk.faults import w_variation
    import silk.faults.application as application

    mod, _, target = _cleanup_circuit(pdk, tmp_path, False)
    catalog = deepcopy(pdk)
    catalog._devices['nmos1p8'].fault_params.w.min = 5.0
    mod.netlister = mod.backend.make_netlister(catalog)
    calls = []
    realize = application._realize_fault

    def observe(*args, **kwargs):
        calls.append(kwargs['pdk'])
        return realize(*args, **kwargs)

    monkeypatch.setattr(application, '_realize_fault', observe)
    manifest = []
    text = Path(mod.generate_netlist(
        fault_set=FaultSet([w_variation(target, sigma=0)]), manifest=manifest)).read_text()
    assert calls == [catalog]
    assert len(manifest) == 1
    assert manifest[0].realized_params['w_new'] == 5
    assert 'w=5 ' in text


@pytest.mark.parametrize('sprinkle_seed, draws', [
    (19, [.0012301533574825742, .2987455375084699, -.2741378553622176]),
    (None, [-.8905918387572742, -.45467078517172255, -.9916465549964624]),
])
def test_random_realizations_follow_request_order_after_selection(
        pdk, tmp_path, sprinkle_seed, draws):
    from pathlib import Path
    from silk.faults import vth_shift
    mod, _, _ = _cleanup_circuit(pdk, tmp_path, True)
    mod.add(pdk.nmos1p8('mn2', d='d', g='g', s='s', b='b', w=2, l=.18))
    mod.set_output_options(mode='overwrite')
    # Reverse placement order, nested ahead of top-level, duplicate last.
    targets = ['xu2.mn1', 'xu1.mn1', 'mn2']
    faults = FaultSet([vth_shift(t, delta=.05, sigma=.02) for t in targets[:2]])
    faults.add(stuck_open(targets[0]))
    faults.sprinkle(r'.*', vth_shift(delta=.05, sigma=.02), rate=1, seed=sprinkle_seed)
    decks = []
    for _ in range(2):
        manifest = []
        decks.append(Path(mod.generate_netlist(fault_set=faults, manifest=manifest,
                                               seed=7)).read_text())
        assert [r.target for r in manifest] == targets
        for record, draw in zip(manifest, draws):
            assert record.fault_type == 'vth_shift'
            assert record.realized_params['delta_jitter'] == pytest.approx(.02 * draw)
            assert record.realized_params['delta_total'] == pytest.approx(.05 + .02 * draw)
            assert record.netlist_comment in decks[-1]
    assert decks[0] == decks[1]


def test_variable_is_realized_again_only_for_a_new_preparation(pdk, tmp_path):
    import numpy as np
    from silk import Variable
    from silk.faults import w_variation
    mod, _, target = _cleanup_circuit(pdk, tmp_path, False)
    width = Variable('width', 2)
    device = mod.elem_dict[target]['device']
    device.w = width
    faults = FaultSet([w_variation(target, sigma=0)])
    first = mod._silk_apply_fault_markers(faults, np.random.default_rng(7))
    try:
        assert first[0]._silk_fault_realization.instances[0].params['w'] == 2
        width.value = 4
        one = mod.netlister.generate_device_instances({target: {'type': mod.elem_dict[target]['type'],
                                                                'device': first[0]}})
        two = mod.netlister.generate_device_instances({target: {'type': mod.elem_dict[target]['type'],
                                                                'device': first[0]}})
        assert one == two
        assert 'w=2 ' in one
        second = mod._silk_apply_fault_markers(faults, np.random.default_rng(7))
        try:
            assert second[0]._silk_fault_realization.instances[0].params['w'] == 4
        finally:
            mod._silk_clear_fault_markers(second)
    finally:
        mod._silk_clear_fault_markers(first)
    assert device.w is width
    assert width.value == 4
    assert not any(hasattr(device, attr) for attr in _FAULT_ATTRS)


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
def test_normalized_fingerprint_and_launch_describe_one_fault_realization(
        pdk, tmp_path, monkeypatch, simulator):
    from hashlib import sha256
    from pathlib import Path
    from silk.faults import vth_shift
    from silk.simulation import Transient
    from silk.simulation.backend import RawResult
    import silk.circuit.modules_netlist as netlisting

    mod, _, target = _cleanup_circuit(pdk, tmp_path, True)
    mod.simulator = simulator
    mod.add_sim_type(Transient(step=1e-9, stop=10e-9))
    emit = netlisting._emit_deck_text
    emissions = []

    def observe(*args, **kwargs):
        text = emit(*args, **kwargs)
        emissions.append(text)
        return text

    monkeypatch.setattr(netlisting, '_emit_deck_text', observe)
    manifest = []
    path = mod.generate_netlist(fault_set=FaultSet([vth_shift(target, delta=.05, sigma=.02)]),
                               manifest=manifest, seed=7)
    assert mod._deck_preparation is None
    assert len(manifest) == 1
    assert len(emissions) == 2
    written, normalized = emissions
    assert Path(path).read_text() == written

    def fault_cards(text):
        return [l for l in text.splitlines() if l.startswith(('*** FAULT:', 'V__silk_fault_vt_', 'E__silk_fault_vt_'))
                or '__silk_fault_shifted_' in l and l.startswith('XM')]

    assert len(fault_cards(written)) == 4
    assert fault_cards(written) == fault_cards(normalized)
    digest = 'silk.normalized-deck/v1:' + sha256(normalized.encode()).hexdigest()
    assert mod._prepared_deck_record['normalized_deck_sha256'] == digest
    launched = []

    def run(backend, **kwargs):
        launched.append(Path(kwargs['netlist_path']).read_text())
        return RawResult(native=None, source_path=mod._run_files.rawfile_path)

    monkeypatch.setattr(type(mod.backend), 'run', run)
    mod.run_simulator()
    assert launched == [written]
    assert mod._launch_record['normalized_deck_sha256'] == digest
    assert mod._launch_record['prepared']['normalized_deck_sha256'] == digest


def test_generated_token_scope_keeps_declarations_and_roles_reserved(tmp_path, pdk):
    from silk.circuit.emitted_names import (
        _generated_token_scope, validate_instance_name, validate_net_name,
    )
    from silk.elements.base import validate_net_token
    from silk.exceptions import CircuitError
    from silk.sources.classes import VoltageSource
    from silk.elements import elements

    shifted = '__silk_fault_shifted_6d6e'
    helper = '__silk_fault_vt_6d6e'
    mod = Module('sky130', str(tmp_path), 'reserved.cir', pdk_details=pdk)
    with _generated_token_scope(instances=frozenset({helper}), nets=frozenset({shifted})):
        for kind in ('pin', 'terminal'):
            assert validate_net_token(shifted, who='test', kind=kind) == shifted
            for forbidden in (helper, '__silk_unrelated'):
                with pytest.raises(ValueError, match='reserved'):
                    validate_net_token(forbidden, who='test', kind=kind)
        validate_instance_name(helper)
        for forbidden in (shifted, '__silk_unrelated'):
            with pytest.raises(CircuitError, match='reserved'):
                validate_instance_name(forbidden)
        for kind in ('net', 'port'):
            with pytest.raises(ValueError, match='reserved'):
                validate_net_token(shifted, who='test', kind=kind)
        with pytest.raises(ValueError, match='reserved'):
            validate_net_name(shifted)
        with pytest.raises(ValueError, match='reserved'):
            mod.add_net(shifted)
        with pytest.raises(ValueError, match='reserved'):
            mod.define_module_ports({shifted: 'input'})
    for build in (
        lambda: VoltageSource(helper, positive='g', negative='0', dc=1),
        lambda: pdk.nmos1p8(helper, d='d', g='g', s='0', b='0'),
        lambda: elements.resistor(helper, p='d', n='0', value=1),
    ):
        with pytest.raises(CircuitError, match='reserved'):
            build()
    with pytest.raises(ValueError, match='reserved'):
        validate_net_token(shifted, who='test')


def test_generated_token_scope_restores_outer_and_checks_grammar():
    from silk.circuit.emitted_names import _generated_token_scope, validate_instance_name
    from silk.elements.base import validate_net_token
    from silk.exceptions import CircuitError
    outer, inner = '__silk_outer', '__silk_inner'
    with _generated_token_scope(instances={outer}, nets={outer}):
        for fail in (False, True):
            try:
                with _generated_token_scope(instances={inner}, nets={inner}):
                    validate_instance_name(inner)
                    assert validate_net_token(inner, who='test') == inner
                    with pytest.raises(CircuitError, match='reserved'):
                        validate_instance_name(outer)
                    with pytest.raises(ValueError, match='reserved'):
                        validate_net_token(outer, who='test')
                    if fail:
                        raise RuntimeError('scope failed')
            except RuntimeError:
                pass
            validate_instance_name(outer)
            assert validate_net_token(outer, who='test') == outer
            with pytest.raises(CircuitError, match='reserved'):
                validate_instance_name(inner)
        with _generated_token_scope(instances={'__silk_x.y'}, nets={'__silk_x.y'}):
            with pytest.raises(CircuitError, match='invalid instance name'):
                validate_instance_name('__silk_x.y')
            with pytest.raises(ValueError, match='invalid net name'):
                validate_net_token('__silk_x.y', who='test')
    with pytest.raises(CircuitError, match='reserved'):
        validate_instance_name(outer)
    with pytest.raises(ValueError, match='reserved'):
        validate_net_token(outer, who='test')


@pytest.mark.parametrize('stage', ['constructor', 'writer'])
def test_fault_helper_exception_restores_token_scope(tmp_path, pdk, monkeypatch, stage):
    from pathlib import Path
    from silk.circuit.emitted_names import (
        _generated_token_scope, validate_instance_name,
    )
    from silk.elements.base import validate_net_token
    from silk.elements import elements
    from silk.exceptions import CircuitError
    from silk.faults import vth_shift
    from silk.sources.classes import VCVS

    mod = Module('sky130', str(tmp_path), 'exception.cir', pdk_details=pdk)
    mod.set_output_options(mode='overwrite')
    for n in ('d', 'g', '0'):
        mod.add_net(n)
    mod.add(pdk.nmos1p8('mn', d='d', g='g', s='0', b='0', w=1, l=.15))
    clean = Path(mod.generate_netlist()).read_text()
    def fail(*args, **kwargs):
        validate_instance_name('__silk_fault_vt_6d6e')
        assert validate_net_token('__silk_fault_shifted_6d6e', who='test')
        raise RuntimeError('injected helper failure')
    with monkeypatch.context() as patch:
        patch.setattr(VCVS, '__init__' if stage == 'constructor' else 'to_spice', fail)
        with _generated_token_scope(instances={'__silk_outer'}, nets={'__silk_outer'}):
            with pytest.raises(RuntimeError, match='injected helper failure'):
                mod.generate_netlist(fault_set=FaultSet([vth_shift('mn', delta=.05)]))
            validate_instance_name('__silk_outer')
            assert validate_net_token('__silk_outer', who='test')
            with pytest.raises(CircuitError, match='reserved'):
                validate_instance_name('__silk_fault_vt_6d6e')
            with pytest.raises(ValueError, match='reserved'):
                validate_net_token('__silk_fault_shifted_6d6e', who='test')
    with pytest.raises(CircuitError, match='reserved'):
        elements.resistor('__silk_fault_vt_6d6e', p='d', n='0', value=1)
    assert Path(mod.generate_netlist()).read_text() == clean


@pytest.mark.parametrize('kind,params', [
    ('w_variation', {'sigma': float('nan')}),
    ('l_variation', {'sigma': float('nan')}),
    ('vth_shift', {'delta': float('nan')}),
    ('vth_shift', {'delta': .02, 'sigma': -.1}),
])
def test_invalid_numeric_fault_preserves_existing_deck(pdk, tmp_path, monkeypatch,
                                                       kind, params):
    from pathlib import Path
    from silk.faults import Fault
    mod, _, target = _cleanup_circuit(pdk, tmp_path, False)
    path = Path(mod.generate_netlist())
    before = path.read_bytes()
    files = mod._run_files
    def forbidden(*args, **kwargs):
        pytest.fail('invalid declaration reached run-file allocation')
    monkeypatch.setattr(type(mod.backend), '_resolve_run_files', forbidden)
    with pytest.raises(ValueError, match=next(iter(params)) if len(params) == 1 else 'sigma'):
        mod.generate_netlist(fault_set=FaultSet([Fault(target, kind, params)]))
    assert path.read_bytes() == before
    assert mod._run_files is files
