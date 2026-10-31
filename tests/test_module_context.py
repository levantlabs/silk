"""Tests for Module as a context manager (``with mod:``) — auto-registers
devices, sources, and passives created inside the block with the active
module."""

import pytest
from silk import sources
from silk.elements import passives
from silk.circuit.modules import Module, _active_module
from silk.exceptions import DuplicateNameError


# ---------------------------------------------------------------------------
# Basic auto-registration
# ---------------------------------------------------------------------------

class TestBuildingAutoRegistration:

    def test_device_auto_registered(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        with mod:
            mn = pdk.nmos1p8('mn1', d='vo', g='vi', s='gnd', b='gnd',
                             w=1, l=0.18)
        assert 'mn1' in mod.elem_dict
        assert mod.elem_dict['mn1']['device'] is mn

    def test_source_auto_registered(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        with mod:
            s = sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8)
        assert 'vdd' in mod.sig_sources_dict
        assert mod.sig_sources_dict['vdd'] is s

    def test_current_source_auto_registered(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        with mod:
            s = sources.CurrentSource('ibias', positive='a', negative='b', dc=1e-4)
        assert 'ibias' in mod.sig_sources_dict
        assert mod.sig_sources_dict['ibias'] is s

    def test_passive_auto_registered(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        with mod:
            r = passives.Resistor('r1', pos='a', neg='b', resistance=1e3)
            c = passives.Capacitor('c1', pos='a', neg='0', capacitance=1e-12)
        assert mod.passives_dict['r1'] is r
        assert mod.passives_dict['c1'] is c

    def test_mixed_elements_all_registered(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        with mod:
            pdk.nmos1p8('mn1', d='vo', g='vi', s='gnd', b='gnd', w=1, l=0.18)
            sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8)
            passives.Resistor('r1', pos='vo', neg='0', resistance=1e3)
        assert 'mn1' in mod.elem_dict
        assert 'vdd' in mod.sig_sources_dict
        assert 'r1' in mod.passives_dict


# ---------------------------------------------------------------------------
# Backward compatibility — outside the block, nothing auto-registers
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    def test_device_outside_block_not_registered(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        # No `with mod:` — caller must add() explicitly.
        mn = pdk.nmos1p8('mn1', d='vo', g='vi', s='gnd', b='gnd', w=1, l=0.18)
        assert 'mn1' not in mod.elem_dict
        # Explicit add() still works
        mod.add(mn)
        assert 'mn1' in mod.elem_dict

    def test_source_outside_block_not_registered(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        s = sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8)
        assert 'vdd' not in mod.sig_sources_dict
        mod.add_source(s)
        assert 'vdd' in mod.sig_sources_dict

    def test_passive_outside_block_not_registered(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        r = passives.Resistor('r1', pos='a', neg='b', resistance=1e3)
        assert 'r1' not in mod.passives_dict
        mod.add_passive(r)
        assert 'r1' in mod.passives_dict


# ---------------------------------------------------------------------------
# Idempotency — re-adding same instance is a no-op
# ---------------------------------------------------------------------------

class TestIdempotency:

    def test_device_explicit_add_inside_block_is_idempotent(self, pdk, tmp_dir):
        """Mixing styles — creating a device inside the block AND calling
        mod.add() on it explicitly — should not double-register."""
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        with mod:
            mn = pdk.nmos1p8('mn1', d='vo', g='vi', s='gnd', b='gnd',
                             w=1, l=0.18)
            mod.add(mn)  # should be a no-op, not raise
        assert 'mn1' in mod.elem_dict

    def test_source_explicit_add_inside_block_is_idempotent(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        with mod:
            s = sources.VoltageSource('v', positive='a', negative='0', dc=1)
            mod.add_source(s)

    def test_passive_explicit_add_inside_block_is_idempotent(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        with mod:
            r = passives.Resistor('r1', pos='a', neg='b', resistance=1e3)
            mod.add_passive(r)

    def test_duplicate_name_different_instance_still_raises(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        with mod:
            pdk.nmos1p8('mn1', d='vo', g='vi', s='gnd', b='gnd', w=1, l=0.18)
        # Second device w/ same name but different instance must raise.
        with pytest.raises(DuplicateNameError):
            mod.add(pdk.nmos1p8('mn1', d='vo', g='vi', s='gnd', b='gnd',
                                 w=1, l=0.18))


# ---------------------------------------------------------------------------
# Nesting — innermost module wins
# ---------------------------------------------------------------------------

class TestNesting:

    def test_innermost_module_wins(self, pdk, tmp_dir):
        outer = Module('PDK_skywater130', tmp_dir, 'outer.cir',
                       pdk_details=pdk, module_name='outer')
        inner = Module('PDK_skywater130', tmp_dir, 'inner.cir',
                       pdk_details=pdk, module_name='inner')
        with outer:
            pdk.nmos1p8('mouter', d='vo', g='vi', s='gnd', b='gnd',
                        w=1, l=0.18)
            with inner:
                pdk.nmos1p8('minner', d='vo', g='vi', s='gnd', b='gnd',
                            w=1, l=0.18)
            # After inner block exits, outer is active again.
            pdk.nmos1p8('mouter2', d='vo', g='vi', s='gnd', b='gnd',
                        w=1, l=0.18)
        assert 'mouter' in outer.elem_dict
        assert 'mouter2' in outer.elem_dict
        assert 'minner' in inner.elem_dict
        assert 'minner' not in outer.elem_dict
        assert 'mouter' not in inner.elem_dict


# ---------------------------------------------------------------------------
# Exception safety — stack unwinds even if body raises
# ---------------------------------------------------------------------------

class TestExceptionSafety:

    def test_exception_inside_block_pops_stack(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        assert _active_module() is None
        with pytest.raises(RuntimeError):
            with mod:
                assert _active_module() is mod
                raise RuntimeError('boom')
        # Stack must be empty again.
        assert _active_module() is None

    def test_module_not_active_outside_block(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        assert _active_module() is None
        with mod:
            assert _active_module() is mod
        assert _active_module() is None


def test_module_creates_missing_output_dir(tmp_path):
    """Module must create its output directory up front — deferring produced
    a bare FileNotFoundError at netlist time that never named the directory
    as the problem (first-run footgun)."""
    import os
    from unittest.mock import MagicMock
    from silk.circuit.modules import Module
    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}
    target = tmp_path / 'not' / 'yet' / 'created'
    assert not target.exists()
    Module('TEST', str(target), 'x.cir', pdk_details=pdk)
    assert target.is_dir()
