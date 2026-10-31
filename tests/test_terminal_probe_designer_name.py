"""Terminal-current probes by designer name (TASK-342).

``tb.probe('st1.MN:d')`` resolves the prefix as a designer name against the
DUT (the same lookup ``'st1.MN:gm'`` uses) when the pre-existing emitted-path
reading (``'xdut.xst1.MN:d'``) does not resolve. Both spellings stay accepted;
when a prefix resolves under both to DIFFERENT devices the emitted reading wins
and a UserWarning names the spelling that selects the shadowed device.
"""
from __future__ import annotations

import math
import os
import warnings

import numpy as np
import pytest

from silk import sources
from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.simulation import analysis

from tests._env import needs_pdk, needs_ngspice
from tests._env import scratch_dir

_PORTS = {'vdd': PortDir.INOUT, 'in': PortDir.IN, 'out': PortDir.OUT,
          'vss': PortDir.GROUND}


def _stage(pdk, tmp):
    stage = Module('sky130', tmp, 'stage.cir', pdk_details=pdk,
                   module_name='stage')
    stage.define_module_ports(_PORTS)
    stage.add_element(pdk.nmos1p8, 'MN',
                      nets={'d': 'out', 'g': 'in', 's': 'vss', 'b': 'vss'},
                      parameters={'w': 1.0, 'l': 0.15})
    return stage


def _bench(helper_beside_dut=False, ammeter=True, dut_child='st1',
           helper_name='st1'):
    from silk.pdk import load_pdk
    pdk = load_pdk('sky130')
    tmp = scratch_dir()
    stage = _stage(pdk, tmp)
    dut = Module('sky130', tmp, 'dut.cir', pdk_details=pdk, module_name='dut')
    dut.define_module_ports(_PORTS)
    dut.add_module(stage, dut_child,
                   nets={'vdd': 'vdd', 'in': 'in', 'out': 'out', 'vss': 'vss'})
    if ammeter:
        dut.save_device_currents = True
        dut.terminal_current_method = 'ammeter_wrapper'
    t = dut.create_test('term')
    t.add_module_under_test({'vdd': 'vdd', 'in': 'in', 'out': 'out',
                             'vss': '0'})
    if helper_beside_dut:
        # a sibling 'st1' at the bench top: its emitted path 'st1.MN'
        # collides with the DUT's designer name 'st1.MN'
        t.add_module(stage, helper_name,
                     nets={'vdd': 'vdd', 'in': 'in', 'out': 'out', 'vss': '0'})
    t.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0',
                                       dc=1.8))
    t.add_source(sources.VoltageSource('vin', positive='in', negative='0',
                                       dc=0.9))
    # bias the drain so the device actually conducts
    t.add_source(sources.VoltageSource('vout', positive='out', negative='0',
                                       dc=1.0))
    t.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
    return dut, t


@needs_pdk
class TestDesignerNameTerminal:
    def test_designer_name_builds_the_same_vector_as_the_emitted_path(self):
        _dut, t = _bench()
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            by_designer = t.probe('st1.MN:d').name
            by_emitted = t.probe('xdut.xst1.MN:d').name
        assert by_designer == by_emitted == 'v.xdut.xst1.xmmn.vamm_d'

    @pytest.mark.parametrize('port', ['d', 'g', 's', 'b', 'D'])
    def test_every_terminal(self, port):
        _dut, t = _bench()
        assert t.probe(f'st1.MN:{port}').name.endswith(
            f'.vamm_{port.lower()}')

    def test_top_level_dut_device_by_designer_name(self):
        from silk.pdk import load_pdk
        pdk = load_pdk('sky130')
        tmp = scratch_dir()
        dut = Module('sky130', tmp, 'dut.cir', pdk_details=pdk,
                     module_name='dut')
        dut.define_module_ports(_PORTS)
        dut.add_element(pdk.nmos1p8, 'M1',
                        nets={'d': 'out', 'g': 'in', 's': 'vss', 'b': 'vss'},
                        parameters={'w': 1.0, 'l': 0.15})
        dut.save_device_currents = True
        dut.terminal_current_method = 'ammeter_wrapper'
        t = dut.create_test('top')
        t.add_module_under_test({'vdd': 'vdd', 'in': 'in', 'out': 'out',
                                 'vss': '0'})
        t.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
        assert t.probe('M1:d').name == t.probe('xdut.M1:d').name \
            == 'v.xdut.xmm1.vamm_d'

    def test_ammeter_opt_in_still_required_for_designer_spelling(self):
        _dut, t = _bench(ammeter=False)
        with pytest.raises(ValueError, match='save_device_currents'):
            t.probe('st1.MN:d')

    def test_unresolved_keeps_the_emitted_error_and_adds_the_designer_hint(self):
        _dut, t = _bench()
        with pytest.raises(ValueError) as exc:
            t.probe('st1.MX:d')
        msg = str(exc.value)
        assert "'st1' is not a submodule instance" in msg   # original text
        assert 'designer name' in msg and 'st1.MN' in msg  # appended hint

    def test_unknown_instance_path_message_unchanged_plus_hint(self):
        _dut, t = _bench()
        with pytest.raises(ValueError) as exc:
            t.probe('xnope.m1:d')
        assert "'xnope'" in str(exc.value) and 'designer name' in str(exc.value)

    def test_designer_name_is_exact_case(self):
        _dut, t = _bench()
        with pytest.raises(ValueError):
            t.probe('st1.mn:d')          # designer lookup is exact-case
        assert t.probe('xdut.xst1.mn:d').name == 'v.xdut.xst1.xmmn.vamm_d'

    def test_collision_emitted_wins_and_warns_naming_shadowed_spelling(self):
        _dut, t = _bench(helper_beside_dut=True)
        with pytest.warns(UserWarning) as rec:
            sig = t.probe('st1.MN:d')
        assert sig.name == 'v.xst1.xmmn.vamm_d'        # emitted (bench) wins
        text = str(rec[0].message)
        assert "'xdut.xst1.MN:d'" in text
        assert 'emitted-path' in text and 'designer' in text

    def test_shadowed_device_reachable_by_emitted_spelling_without_warning(self):
        _dut, t = _bench(helper_beside_dut=True)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert t.probe('xdut.xst1.MN:d').name == 'v.xdut.xst1.xmmn.vamm_d'

    def test_shared_definition_at_different_instance_paths_warns(self, monkeypatch):
        """A shared device object still denotes two different placed instances."""
        import silk.circuit.module_test as mt
        _dut, t = _bench(helper_beside_dut=True)
        emitted_dev = t.netlist.elem_dict['st1']['device'].elem_dict['MN']['device']
        real = mt.find_designer_device

        def same(dut, name):
            hit = real(dut, name)
            return None if hit is None else (hit[0], hit[1], emitted_dev)
        monkeypatch.setattr(mt, 'find_designer_device', same)
        with pytest.warns(UserWarning, match='emitted-path'):
            assert t.probe('st1.MN:d').name == 'v.xst1.xmmn.vamm_d'


class TestEmittedSpellingsBuildTheRealVector:
    """The emitted walk accepts a missing 'x' and any case; the vector is
    built from the names it MATCHED (the netlist's own spelling)."""

    @pytest.mark.parametrize('spelling', [
        'st1.MN:d', 'xst1.MN:d', 'XST1.mn:D', 'xst1.mn:d'])
    def test_bench_level_subcircuit_spellings_agree(self, spelling):
        _dut, t = _bench(dut_child='core', helper_name='st1',
                         helper_beside_dut=True)
        assert t.probe(spelling).name == 'v.xst1.xmmn.vamm_d'

    @pytest.mark.parametrize('spelling', [
        'xdut.xst1.MN:d', 'dut.st1.MN:d', 'XDUT.XST1.mn:d'])
    def test_dut_path_spellings_agree(self, spelling):
        _dut, t = _bench()
        assert t.probe(spelling).name == 'v.xdut.xst1.xmmn.vamm_d'


@pytest.mark.slow
@needs_pdk
@needs_ngspice
def test_real_ngspice_bench_level_subcircuit_missing_x_reads_real_vector():
    dut, t = _bench(dut_child='core', helper_name='st1', helper_beside_dut=True)
    bare, typed = t.probe('st1.MN:d'), t.probe('xst1.MN:d')
    dut.run_test('term', corner='tt')
    yb, yt = np.asarray(bare.y), np.asarray(typed.y)
    assert len(yb) > 1 and not np.isnan(yb).any()
    assert np.array_equal(yb, yt)
    assert np.max(np.abs(yb)) > 0


@pytest.mark.slow
@needs_pdk
@needs_ngspice
def test_real_ngspice_designer_and_emitted_read_identical_arrays():
    dut, t = _bench()
    by_designer = t.probe('st1.MN:d')
    by_emitted = t.probe('xdut.xst1.MN:d')
    dut.run_test('term', corner='tt')
    yd, ye = np.asarray(by_designer.y), np.asarray(by_emitted.y)
    assert len(yd) > 1 and not np.isnan(yd).any()
    assert np.array_equal(yd, ye)
    assert np.max(np.abs(yd)) > 0            # a real current, not a phantom zero
