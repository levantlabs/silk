"""Tests for SubcircuitInstance — handle returned by Module.add_element()
when the added element is itself a Module (subcircuit).

Covers attribute access to internal nets, port resolution (which must map
to the external top-level net, not the hierarchical path), nested
subcircuits (``bias_inst.fcamp.vbiasn``), __dir__ autocomplete, and the
clear-error path when a net is read before simulate()."""

import pytest
from silk.circuit.modules import Module, SubcircuitInstance
from silk import sources


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _make_leaf(pdk, tmp_dir):
    """Simple subcircuit: one NMOS, three internal nets (vin, vout, vdd, gnd)
    + one extra internal net (vint) exposed only inside."""
    leaf = Module('PDK_skywater130', tmp_dir, 'leaf.cir',
                  pdk_details=pdk, module_name='leaf')
    with leaf:
        pdk.nmos1p8('mn', d='vout', g='vin', s='vint', b='gnd',
                    w=1, l=0.18)
        pdk.nmos1p8('mn2', d='vint', g='vin', s='gnd', b='gnd',
                    w=1, l=0.18)
    from silk import Port, PortDir
    leaf.define_module_ports({
        'vin':  Port(dir=PortDir.IN),
        'vout': Port(dir=PortDir.OUT),
        'vdd':  Port(dir=PortDir.IN),
        'gnd':  Port(dir=PortDir.IN),
    })
    return leaf


def _make_wrapper(pdk, tmp_dir, leaf):
    """Wrapper subcircuit that instantiates the leaf inside."""
    wrap = Module('PDK_skywater130', tmp_dir, 'wrap.cir',
                  pdk_details=pdk, module_name='wrap')
    wrap.add_module(leaf, 'child', nets={
        'vin':  'vin',
        'vout': 'mid',        # rename at this level
        'vdd':  'vdd',
        'gnd':  'gnd',
    })
    from silk import Port, PortDir
    wrap.define_module_ports({
        'vin':  Port(dir=PortDir.IN),
        'mid':  Port(dir=PortDir.OUT),
        'vdd':  Port(dir=PortDir.IN),
        'gnd':  Port(dir=PortDir.IN),
    })
    return wrap


# ---------------------------------------------------------------------------
# Basic handle properties
# ---------------------------------------------------------------------------

class TestHandleReturn:

    def test_add_element_returns_handle_for_module(self, pdk, tmp_dir):
        leaf = _make_leaf(pdk, tmp_dir)
        tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        h = tb.add_module(leaf, 'child', nets={
            'vin': 'vin', 'vout': 'vout', 'vdd': 'vdd', 'gnd': '0'})
        assert isinstance(h, SubcircuitInstance)

    def test_add_element_returns_none_for_raw_device(self, pdk, tmp_dir):
        tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        with tb:
            mn = pdk.nmos1p8('mn1', d='out', g='in', s='gnd', b='gnd',
                             w=1, l=0.18)
        # Direct add_element with a raw AnalogElement returns None
        # (backward-compat with existing callers).
        ret = tb.add_element(mn.__class__, 'mn2',
                              nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
                              parameters={'w': 1, 'l': 0.18})
        assert ret is None

    def test_instance_name_has_x_prefix(self, pdk, tmp_dir):
        leaf = _make_leaf(pdk, tmp_dir)
        tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        h = tb.add_module(leaf, 'fcamp', nets={
            'vin': 'vin', 'vout': 'vout', 'vdd': 'vdd', 'gnd': '0'})
        assert h.instance_name == 'xfcamp'      # silk prepends 'x'


# ---------------------------------------------------------------------------
# Path resolution (no sim needed — purely structural)
# ---------------------------------------------------------------------------

class TestPathResolution:

    def test_internal_net_path(self, pdk, tmp_dir):
        leaf = _make_leaf(pdk, tmp_dir)
        tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        h = tb.add_module(leaf, 'fcamp', nets={
            'vin': 'vin', 'vout': 'vout', 'vdd': 'vdd', 'gnd': '0'})
        # vint is internal-only (not a port) → hierarchical path
        assert h.path('vint') == 'xfcamp.vint'

    def test_port_path_maps_to_external_name(self, pdk, tmp_dir):
        """A port must resolve to the TOP-LEVEL net the user mapped it to,
        not the hierarchical path — ngspice aliases ports in the rawfile."""
        leaf = _make_leaf(pdk, tmp_dir)
        tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        h = tb.add_module(leaf, 'fcamp', nets={
            'vin':  'my_in',    # different name at top level
            'vout': 'my_out',
            'vdd':  'vdd',
            'gnd':  '0'})
        assert h.path('vin')  == 'my_in'
        assert h.path('vout') == 'my_out'
        assert h.path('gnd')  == '0'

    def test_hierarchical_internal_net(self, pdk, tmp_dir):
        """Two-level nesting: bias wraps leaf. Access bias.child.vint."""
        leaf = _make_leaf(pdk, tmp_dir)
        wrap = _make_wrapper(pdk, tmp_dir, leaf)
        tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        bias = tb.add_module(wrap, 'bias', nets={
            'vin': 'vin', 'mid': 'mid', 'vdd': 'vdd', 'gnd': '0'})
        child = bias.child                  # attribute → nested handle
        assert isinstance(child, SubcircuitInstance)
        assert child.path('vint') == 'xbias.xchild.vint'

    def test_hierarchical_port_walks_chain(self, pdk, tmp_dir):
        """Child's port vout is mapped to wrap's internal net 'mid'; wrap's
        'mid' is a port of wrap, mapped to tb's 'mid'. So leaf.vout should
        resolve all the way to top-level 'mid'."""
        leaf = _make_leaf(pdk, tmp_dir)
        wrap = _make_wrapper(pdk, tmp_dir, leaf)
        tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        bias = tb.add_module(wrap, 'bias', nets={
            'vin': 'vin', 'mid': 'mid_top', 'vdd': 'vdd', 'gnd': '0'})
        child = bias.child
        assert child.path('vout') == 'mid_top'


# ---------------------------------------------------------------------------
# __dir__ autocomplete
# ---------------------------------------------------------------------------

class TestDir:

    def test_dir_includes_nets(self, pdk, tmp_dir):
        leaf = _make_leaf(pdk, tmp_dir)
        tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        h = tb.add_module(leaf, 'fcamp', nets={
            'vin': 'vin', 'vout': 'vout', 'vdd': 'vdd', 'gnd': '0'})
        d = dir(h)
        assert 'vint' in d            # internal net
        assert 'vin' in d             # port
        assert 'instance_name' in d   # real attribute still visible

    def test_dir_includes_children(self, pdk, tmp_dir):
        leaf = _make_leaf(pdk, tmp_dir)
        wrap = _make_wrapper(pdk, tmp_dir, leaf)
        tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        bias = tb.add_module(wrap, 'bias', nets={
            'vin': 'vin', 'mid': 'mid', 'vdd': 'vdd', 'gnd': '0'})
        assert 'child' in dir(bias)


# ---------------------------------------------------------------------------
# AttributeError on unknown names
# ---------------------------------------------------------------------------

class TestUnknownAttr:

    def test_unknown_net_raises_attribute_error(self, pdk, tmp_dir):
        leaf = _make_leaf(pdk, tmp_dir)
        tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        h = tb.add_module(leaf, 'fcamp', nets={
            'vin': 'vin', 'vout': 'vout', 'vdd': 'vdd', 'gnd': '0'})
        with pytest.raises(AttributeError, match='no net or sub-instance'):
            _ = h.nonexistent_net

    def test_dunder_attrs_not_intercepted(self, pdk, tmp_dir):
        """Python internals shouldn't trigger net lookup."""
        leaf = _make_leaf(pdk, tmp_dir)
        tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        h = tb.add_module(leaf, 'fcamp', nets={
            'vin': 'vin', 'vout': 'vout', 'vdd': 'vdd', 'gnd': '0'})
        # These should all be AttributeError without going through our
        # net-resolution path.
        with pytest.raises(AttributeError):
            _ = h.__does_not_exist__


# ---------------------------------------------------------------------------
# Pre-sim access — helpful error
# ---------------------------------------------------------------------------

class TestPreSimAccess:

    def test_access_before_simulate_errors_cleanly(self, pdk, tmp_dir):
        leaf = _make_leaf(pdk, tmp_dir)
        tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        h = tb.add_module(leaf, 'fcamp', nets={
            'vin': 'vin', 'vout': 'vout', 'vdd': 'vdd', 'gnd': '0'})
        # Structural lookup is valid before a run; the first read refuses.
        from silk.exceptions import UnboundSignalError
        reference = h.vint
        with pytest.raises(UnboundSignalError, match='run'):
            _ = reference.y


@pytest.mark.parametrize('name', ['bus', 'child', 'device', 'instance_name',
                                  'net', 'path', 'subcircuit'])
@pytest.mark.parametrize('kind', ['net', 'port', 'bus', 'child', 'device',
                                  'passive', 'child_array', 'device_array',
                                  'ambiguous'])
def test_declared_placement_names_take_precedence(pdk, tmp_dir, name, kind):
    """Shadowed tools must not redirect probes, faults, or restored handles."""
    import copy
    import pickle
    from silk.elements import elements
    from silk.exceptions import CircuitError
    from silk.faults import stuck_open

    leaf = Module(None, tmp_dir, 'names.cir', pdk, module_name='names')
    leaf.define_module_ports({'out': 'output'})
    leaf.add(pdk.nmos1p8('sense', d='anchor', g='out', s='0', b='0', w=1, l=.5))
    leaf.add_net('_private')
    leaf.add_net('__class__')
    wiring = {'out': 'parent_out'}
    if kind in ('net', 'ambiguous'):
        leaf.add_net(name)
    if kind == 'port':
        leaf.define_module_ports({name: 'output'})
        wiring[name] = 'parent_named'
    elif kind == 'bus':
        leaf.add_net(name, bits=range(2, 4))
    elif kind in ('child', 'child_array'):
        inner = Module(None, tmp_dir, 'inner.cir', pdk, module_name='inner')
        inner.define_module_ports({'out': 'output'})
        inner.add(pdk.nmos1p8('sense', d='tap', g='out', s='0', b='0', w=1, l=.5))
        leaf.add_module(inner, name + ('[3]' if kind == 'child_array' else ''),
                        out='anchor')
    elif kind in ('device', 'device_array'):
        leaf.add(pdk.nmos1p8(name + ('[3]' if kind == 'device_array' else ''),
                            d='anchor', g='out', s='0', b='0', w=1, l=.5))
    elif kind in ('passive', 'ambiguous'):
        leaf.add(elements.resistor(name, p='anchor', n='out', value=1000))

    owner = Module(None, tmp_dir, 'owner.cir', pdk, module_name='owner')
    tb = owner.create_test('read')
    stage = tb.add_module(leaf, 'stage', nets=wiring)
    tb.netlist.define_module_name('read_deck')
    tb.netlist.define_module_ports({})
    deck = tb.netlist.generate_netlist(subckt=True)
    copies = [(tb, stage), (tb, copy.copy(stage)),
              copy.deepcopy((tb, stage)), pickle.loads(pickle.dumps((tb, stage)))]
    for bench, handle in copies:
        assert handle.__class__ is SubcircuitInstance
        with pytest.raises(AttributeError):
            handle._private
        if kind == 'ambiguous':
            with pytest.raises(CircuitError, match=f"Ambiguous placement attribute '{name}'"):
                getattr(handle, name)
            continue
        item = getattr(handle, name)
        if kind in ('net', 'port'):
            expected = 'parent_named' if kind == 'port' else 'xstage.' + name
            assert bench.probe(item).name == expected
            fault_item, target = handle.sense, 'xstage.sense'
        elif kind == 'bus':
            assert bench.probe(item[2]).name == f'xstage.{name}[2]'
            assert bench.probe(item[3]).name == f'xstage.{name}[3]'
            fault_item, target = handle.sense, 'xstage.sense'
        elif kind in ('child', 'child_array'):
            child = item[3] if kind == 'child_array' else item
            key = name + ('[3]' if kind == 'child_array' else '')
            emitted = name + ('<3>' if kind == 'child_array' else '')
            assert bench.probe(child.tap).name == f'xstage.x{emitted}.tap'
            fault_item, target = child.sense, f'xstage.x{key}.sense'
        else:
            fault_item = item[3] if kind == 'device_array' else item
            target = 'xstage.' + name + ('[3]' if kind == 'device_array' else '')
        fault = stuck_open(fault_item)
        assert fault.target == target
        assert bench.netlist._silk_resolve_fault_target(target).name == target.split('.')[-1]
        assert bench.netlist.generate_netlist(subckt=True) == deck
