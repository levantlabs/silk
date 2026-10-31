"""Bus-handle lookup via mod.nets.<bus_name> — silk runtime regression test.

Companion to tests/test_codegen_submodules.py: this validates the silk
runtime contract that the hierarchical codegen depends on.
"""
import pytest

from silk.circuit.modules import Module
from silk.circuit.net import Net


class TestBusHandleAccessor:
    def _mod(self):
        # Construct a minimal Module without a real PDK to test accessor.
        m = Module(
            pdk_name='test', dir_folder='/tmp', netlistname='t.cir',
            pdk_details=None, module_name='t',
        )
        return m

    def test_add_net_with_bits_stores_handle(self):
        m = self._mod()
        returned = m.add_net('mybus', bits=4)
        assert isinstance(returned, Net)
        assert returned.width == 4
        # The handle is also stored on the module.
        assert 'mybus' in m._bus_handles
        assert m._bus_handles['mybus'] is returned

    def test_mod_nets_attr_returns_bus_handle(self):
        m = self._mod()
        m.add_net('mybus', bits=4)
        handle = m.nets.mybus
        assert isinstance(handle, Net)
        assert handle.width == 4
        assert handle.name == 'mybus'

    def test_mod_nets_attr_scalar_still_works(self):
        m = self._mod()
        m.add_net('scalar')
        handle = m.nets.scalar
        assert isinstance(handle, Net)
        assert handle.width == 1

    def test_mod_nets_getitem_returns_bus_handle(self):
        m = self._mod()
        m.add_net('mybus', bits=4)
        handle = m.nets['mybus']
        assert isinstance(handle, Net)
        assert handle.width == 4

    def test_bus_name_in_nets(self):
        m = self._mod()
        m.add_net('mybus', bits=4)
        assert 'mybus' in m.nets

    def test_unknown_bus_name_raises(self):
        m = self._mod()
        with pytest.raises(AttributeError, match='No net'):
            m.nets.nonexistent

    def test_dir_includes_bus_name(self):
        m = self._mod()
        m.add_net('scalar')
        m.add_net('mybus', bits=4)
        names = dir(m.nets)
        assert 'scalar' in names
        assert 'mybus' in names

    def test_scalar_add_net_does_not_create_bus_handle(self):
        # Sanity: scalar add_net should NOT register in _bus_handles.
        m = self._mod()
        m.add_net('plain')
        assert 'plain' not in m._bus_handles
