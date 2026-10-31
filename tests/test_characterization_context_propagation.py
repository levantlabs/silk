"""TASK-3 (DB-02-CONTEXT) propagation tests.

Covers the acceptance criteria of backlog TASK-3
("DB-02-CONTEXT: runtime characterization identity"):

- AC1: a ModuleTestbench run exposes owner path, test name, declared
  parameters, and PVT fields to the persistence layer.
- AC2: SpecReport carries the owner path in addition to module_name and
  test_name.
- AC3: plain-Module workflows stay explicitly unscoped; nothing is ever
  invented.
- AC4: emitted netlists are byte-identical whether or not a characterization
  context is attached (design §9.1, metadata-only propagation).

"""
from unittest.mock import patch

import pytest
from tests._temperature import mark_simulated

from silk.circuit.modules import Module
from silk.circuit.module_test import ModuleTestbench
from silk.circuit.spec_report import SpecReport
from silk.db.context import CharacterizationContext
from silk.exceptions import CharacterizationContextError
from silk import sources
from silk.simulation.analysis import Transient


# ---------------------------------------------------------------------------
# Shared helpers (mirrors tests/test_module_test.py's configured_module, kept
# local so this file has no cross-file fixture dependency).
# ---------------------------------------------------------------------------

def _configured_module(pdk, tmp_dir, module_name='ctx_dut'):
    mod = Module('PDK_skywater130', tmp_dir, 'dut.cir', pdk_details=pdk)
    mod.define_module_name(module_name)
    mod.define_module_nets({'in': None, 'out': None, 'vdd': None, 'vss': None})
    mod.add(pdk.nmos1p8('mn', d='out', g='in', s='vss', b='vss', w=1, l=0.15, m=1))
    mod.add(pdk.pmos1p8('mp', d='out', g='in', s='vdd', b='vdd', w=2, l=0.15, m=1))
    return mod


@pytest.fixture
def configured_module(pdk, tmp_dir):
    """A Module with name, pins, and devices — ready for ModuleTestbench.
    Mirrors tests/test_module_test.py's fixture of the same name."""
    return _configured_module(pdk, tmp_dir)


def _wired_test(mod, name='t1', parameters=None):
    """A ModuleTestbench with the DUT added, a source, and a sim type — ready
    to run() (with .netlist.simulate mocked out; no real ngspice needed)."""
    test = mod.create_test(name, parameters=parameters)
    test.add_module_under_test({
        'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0',
    })
    test.add_source(sources.VoltageSource('vsup', positive='vdd', negative='0', dc=1.8))
    test.add_source(sources.VoltageSource('vin_src', positive='vin', negative='0', dc=0.9))
    test.add_sim_type(Transient(step=1e-9, stop=1e-8))
    return test


def _run_without_simulating(test, **kwargs):
    """Run a ModuleTestbench without invoking ngspice — only netlist.simulate
    is stubbed, so context-building/metric-extraction/report-assembly run for
    real. The stub also sets ``sim_results_loaded`` (TASK-115: ``run()``
    calls the public ``evaluate()``, whose guard checks that flag).

    A declared supply is configured first (TASK-158): this bench has two
    grounded DC sources, so which one is the rail is the author's choice, and
    the stub then establishes it the way a real launch's deck check would."""
    voltage = kwargs.get('voltage')
    if voltage is not None:
        test.set_supply_voltage(voltage, source='vsup')
    def fake_simulate(*_a, **_kw):
        mark_simulated(test.netlist, _kw.get('corner', 'tt'), voltage=voltage)
    with patch.object(test.netlist, 'simulate', side_effect=fake_simulate):
        return test.run(**kwargs)


# ---------------------------------------------------------------------------
# ModuleTestbench.parameters — declared test-parameter map (AC1)
# ---------------------------------------------------------------------------

class TestDeclaredParameters:
    def test_default_is_empty_dict(self, configured_module):
        test = configured_module.create_test('t1')
        assert test.parameters == {}

    def test_create_test_accepts_parameters(self, configured_module):
        test = configured_module.create_test('t1', parameters={'cload_f': 2e-12})
        assert test.parameters == {'cload_f': 2e-12}

    def test_direct_construction_accepts_parameters(self, configured_module):
        test = ModuleTestbench(configured_module, 'direct', parameters={'n': 4})
        assert test.parameters == {'n': 4}

    def test_parameters_snapshot_independent_of_caller_dict(self, configured_module):
        original = {'cload_f': 2e-12}
        test = configured_module.create_test('t1', parameters=original)
        original['cload_f'] = 999.0
        assert test.parameters == {'cload_f': 2e-12}

    def test_reserved_pvt_name_rejected_at_construction(self, configured_module):
        with pytest.raises(CharacterizationContextError):
            configured_module.create_test('t1', parameters={'temperature_c': 27.0})

    def test_non_mapping_parameters_rejected(self, configured_module):
        with pytest.raises(CharacterizationContextError):
            configured_module.create_test('t1', parameters=['not', 'a', 'dict'])


# ---------------------------------------------------------------------------
# ModuleTestbench.owner_path
# ---------------------------------------------------------------------------

class TestOwnerPath:
    def test_top_level_module_owner_path_is_root(self, configured_module):
        test = configured_module.create_test('t1')
        assert test.owner_path == '$'

    def test_embedded_instance_owner_path_is_rebased(self, pdk, tmp_dir):
        parent = Module('PDK_skywater130', tmp_dir, 'parent.cir', pdk_details=pdk)
        child = _configured_module(pdk, tmp_dir, module_name='child_mod')
        parent.add_module(child, 'inst1', nets={
            'in': 'a', 'out': 'b', 'vdd': 'vdd', 'vss': '0',
        })
        child_copy = parent.elem_dict['inst1']['device']
        test = child_copy.create_test('t1')
        assert test.owner_path == '$'
        assert parent.list_test_entries()[0].owner_path == '$/inst1'

    def test_embedded_instance_name_refuses_slash(self, pdk, tmp_dir):
        from silk.exceptions import CircuitError
        parent = Module('PDK_skywater130', tmp_dir, 'parent.cir', pdk_details=pdk)
        child = _configured_module(pdk, tmp_dir, module_name='child_mod')
        with pytest.raises(CircuitError, match='ASCII'):
            parent.add_module(child, 'inst/1', nets={
                'in': 'a', 'out': 'b', 'vdd': 'vdd', 'vss': '0',
            })


# ---------------------------------------------------------------------------
# run() derives the complete descriptor from observed fixed temperature and
# declared supply. Legacy temperature is an assertion requiring explicit setup;
# the descriptor itself still requires all six fields.
# ---------------------------------------------------------------------------

class TestRunContextPropagation:
    def test_default_run_is_unscoped(self, configured_module):
        test = _wired_test(configured_module, parameters={'cload_f': 2e-12})
        report = _run_without_simulating(test, corner='tt')
        assert report.characterization_context is None
        assert test.netlist.characterization_context is None
        # Owner path / declared parameters are still exposed — AC1's "always
        # available" half — even when this run stays unscoped.
        assert report.owner_path == '$'
        assert report.parameters == {'cload_f': 2e-12}

    def test_full_pvt_builds_scoped_context_on_report_and_netlist(self, configured_module):
        test = _wired_test(configured_module, parameters={'cload_f': 2e-12})
        test.set_temperature(27)
        report = _run_without_simulating(
            test, corner='tt', temperature=27.0, voltage=1.8,
        )
        ctx = report.characterization_context
        assert isinstance(ctx, CharacterizationContext)
        assert ctx.owner_path == '$'
        assert ctx.test_name == test.test_name
        assert ctx.parameters == {'cload_f': 2e-12}
        assert ctx.corner == 'tt'
        assert ctx.temperature == 27.0
        assert ctx.voltage == 1.8
        # Same descriptor object rode onto the generated simulation Module.
        assert test.netlist.characterization_context is ctx

    def test_only_temperature_raises(self, configured_module):
        test = _wired_test(configured_module)
        with pytest.raises(CharacterizationContextError):
            _run_without_simulating(test, corner='tt', temperature=27.0)

    def test_only_voltage_uses_observed_temperature(self, configured_module):
        test = _wired_test(configured_module)
        report = _run_without_simulating(test, corner='tt', voltage=1.8)
        assert report.characterization_context.temperature == 27
        assert report.characterization_context.voltage == 1.8

    def test_run_corners_forwards_pvt_to_every_corner(self, configured_module):
        test = _wired_test(configured_module, parameters={'n': 4})
        test.set_temperature(27)
        test.set_supply_voltage(1.8, source='vsup')
        # side_effect (not bare return_value=None) also sets sim_results_loaded
        # so evaluate()'s "nothing loaded" guard passes each corner's run()
        # (TASK-115: run() calls the public evaluate()).
        #
        # Patched on the CLASS, as a plain function (not a Mock instance
        # attribute), so normal method binding passes it the actual netlist
        # instance being simulated: run_corners() (TASK-301.3) deep-copies
        # each corner's netlist before simulating, so a version that closed
        # over test.netlist would mark the wrong (original) object simulated
        # and leave each corner's own copy — the one run_corners() actually
        # evaluates — believing nothing had loaded.
        def fake_simulate(netlist, *_a, **_kw):
            mark_simulated(netlist, _kw.get('corner', 'tt'), voltage=1.8)
        with patch.object(type(test.netlist), 'simulate', fake_simulate):
            results = test.run_corners(
                corners=('tt', 'ff'), temperature=27.0, voltage=1.8,
            )
        for corner in ('tt', 'ff'):
            ctx = results[corner].characterization_context
            assert ctx.corner == corner
            assert ctx.temperature == 27.0

    def test_run_corners_default_stays_unscoped(self, configured_module):
        test = _wired_test(configured_module)
        def fake_simulate(netlist, *_a, **_kw):
            mark_simulated(netlist, _kw.get('corner', 'tt'))
        with patch.object(type(test.netlist), 'simulate', fake_simulate):
            results = test.run_corners(corners=('tt', 'ff'))
        for corner in ('tt', 'ff'):
            assert results[corner].characterization_context is None


# ---------------------------------------------------------------------------
# SpecReport — owner_path in addition to module_name/test_name (AC2)
# ---------------------------------------------------------------------------

class TestSpecReportOwnerPath:
    def test_defaults_to_root(self):
        report = SpecReport(test_name='t', module_name='m', metrics={})
        assert report.owner_path == '$'
        assert report.parameters == {}
        assert report.characterization_context is None

    def test_explicit_owner_path(self):
        report = SpecReport(
            test_name='t', module_name='m', metrics={}, owner_path='$/amp1',
        )
        assert report.owner_path == '$/amp1'


# ---------------------------------------------------------------------------
# Plain-Module workflows stay explicitly unscoped (AC3)
# ---------------------------------------------------------------------------

class TestPlainModuleUnscoped:
    def test_default_characterization_context_is_none(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'plain.cir', pdk_details=pdk)
        assert mod.characterization_context is None

    def test_store_net_currents_style_module_never_gets_a_context(self, pdk, tmp_dir):
        """Mirrors silk/examples/net_current_persistence_example.py: a plain
        top-level Module that never touches ModuleTestbench. Nothing here
        should ever populate characterization_context — no ModuleTestbench
        exists to build one, and TASK-3 never infers one."""
        tb = Module('PDK_skywater130', tmp_dir, 'sink_tb.cir', pdk_details=pdk)
        tb.add_net('vdd')
        tb.add(pdk.nmos1p8('mn', d='vdd', g='vdd', s='0', b='0', w=1, l=0.15, m=1))
        assert tb.characterization_context is None


# ---------------------------------------------------------------------------
# Deck-byte identity (AC4 / hard constraint): context propagation is Python
# metadata only. Attaching a CharacterizationContext, or declaring test
# parameters, must never change a single byte of the emitted netlist.
# ---------------------------------------------------------------------------

class TestDeckByteIdentity:
    def test_attaching_context_after_generation_does_not_change_deck(
        self, configured_module,
    ):
        test = _wired_test(configured_module, parameters={'cload_f': 2e-12})
        test.netlist.set_output_options(mode='overwrite')
        test_netlist_deck_path = test.netlist.generate_netlist(corner='tt')
        with open(test_netlist_deck_path) as fh:
            deck_before = fh.read()

        test.netlist.characterization_context = CharacterizationContext(
            owner_path='$', test_name=test.test_name,
            parameters={'cload_f': 2e-12}, corner='tt',
            temperature=27.0, voltage=1.8,
        )
        test.netlist.set_output_options(mode='overwrite')
        test_netlist_deck_path = test.netlist.generate_netlist(corner='tt')
        with open(test_netlist_deck_path) as fh:
            deck_after = fh.read()

        assert deck_before == deck_after

    def test_context_carrying_run_emits_byte_identical_deck_to_context_free_run(
        self, pdk, tmp_path,
    ):
        """Two structurally identical testbenches — same module/test names,
        same devices/sources/analysis — one with declared parameters + a
        full PVT context, one with neither. Their emitted decks differ only
        in the (normalized-away) results directory path."""
        def _build(dir_folder, *, parameters):
            mod = _configured_module(pdk, str(dir_folder), module_name='byte_dut')
            test = _wired_test(mod, name='same_test', parameters=parameters)
            return test

        plain_dir = tmp_path / 'plain'
        ctx_dir = tmp_path / 'ctx'
        plain_dir.mkdir()
        ctx_dir.mkdir()

        plain = _build(plain_dir, parameters=None)
        ctx = _build(ctx_dir, parameters={'cload_f': 2e-12, 'vin_amplitude_v': 0.1})
        ctx.netlist.characterization_context = CharacterizationContext(
            owner_path=ctx.owner_path, test_name=ctx.test_name,
            parameters=ctx.parameters, corner='tt',
            temperature=27.0, voltage=1.8,
        )

        plain.netlist.set_output_options(mode='overwrite')
        ctx.netlist.set_output_options(mode='overwrite')
        plain_netlist_deck_path = plain.netlist.generate_netlist(corner='tt')
        with open(plain_netlist_deck_path) as fh:
            plain_deck = fh.read().replace(str(plain_dir), '<TBDIR>')

        ctx_netlist_deck_path = ctx.netlist.generate_netlist(corner='tt')
        with open(ctx_netlist_deck_path) as fh:
            ctx_deck = fh.read().replace(str(ctx_dir), '<TBDIR>')

        assert plain_deck == ctx_deck


def test_legacy_owner_path_segment_escaping():
    from silk.db.identity import _escape_owner_path_segment
    assert _escape_owner_path_segment('inst/1') == 'inst~11'
