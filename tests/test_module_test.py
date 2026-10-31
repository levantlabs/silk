from silk.circuit.spec_report import SpecReport
"""
Tests for ModuleTestbench — testbench creation, source/sim delegation,
metric extraction, and run() lifecycle.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from silk.circuit.module_test import ModuleTestbench
from silk.circuit.modules import Module
from silk.exceptions import DuplicateNameError
from silk import sources
from silk.simulation.analysis import Transient, AC, DC, Noise
from silk.signals.base import Signal
# Aliased: ``Transient`` is already the ANALYSIS class above. TASK-196's
# annotation vocabulary happens to share the word.
from silk.signals.schema import Transient as TransientSignal
from tests._env import needs_ngspice, needs_pdk


# ---------------------------------------------------------------------------
# Helper: create a configured module for testing
# ---------------------------------------------------------------------------

@pytest.fixture
def configured_module(pdk, tmp_dir):
    """A Module with name, pins, and devices — ready for ModuleTestbench."""
    mod = Module('PDK_skywater130', tmp_dir, 'dut.cir', pdk_details=pdk)
    mod.define_module_name('test_inv')
    mod.define_module_nets({'in': None, 'out': None, 'vdd': None, 'vss': None})

    mod.add(pdk.nmos1p8('mn', d='out', g='in', s='vss', b='vss', w=1, l=0.15, m=1))
    mod.add(pdk.pmos1p8('mp', d='out', g='in', s='vdd', b='vdd', w=2, l=0.15, m=1))
    return mod


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------

class TestPyModuleTestConstructor:

    def test_create_via_module(self, configured_module):
        """create_test() on a Module returns a ModuleTestbench instance."""
        test = configured_module.create_test('basic')
        assert isinstance(test, ModuleTestbench)

    def test_direct_constructor(self, configured_module):
        """Direct construction registers the test on its owning module."""
        test = ModuleTestbench(configured_module, 'direct')
        assert test.test_name == 'direct'
        assert test.module is configured_module
        assert configured_module._tests['direct'] is test
        assert configured_module.list_tests() == ['direct']

    def test_direct_constructor_is_visible_to_run_all_tests(
        self, configured_module
    ):
        """run_all_tests() includes directly constructed testbenches."""
        test = ModuleTestbench(configured_module, 'direct')
        result = object()
        test.run = MagicMock(return_value=result)

        assert configured_module.run_all_tests(recursive=False) == {
            'test_inv.direct': result,
        }
        test.run.assert_called_once_with()

    @pytest.mark.parametrize(
        ('first_entry', 'second_entry'),
        [
            ('create', 'create'),
            ('create', 'direct'),
            ('direct', 'create'),
            ('direct', 'direct'),
        ],
    )
    def test_duplicate_name_rejected_and_first_test_preserved(
        self, configured_module, first_entry, second_entry
    ):
        """Every entry-point ordering shares one fail-loud registry."""
        def construct(entry):
            if entry == 'create':
                return configured_module.create_test('duplicate')
            return ModuleTestbench(configured_module, 'duplicate')

        first = construct(first_entry)
        first.add_metric('kept', lambda _ckt: 1.0)
        original_metrics = list(first._metrics)

        with pytest.raises(
            DuplicateNameError,
            match="Test 'duplicate'.*module 'test_inv'",
        ):
            construct(second_entry)

        assert configured_module._tests == {'duplicate': first}
        assert first._metrics == original_metrics

    def test_duplicate_default_name_rejected(self, configured_module):
        """The implicit default_test name follows the same uniqueness rule."""
        first = configured_module.create_test()

        with pytest.raises(DuplicateNameError, match="'default_test'"):
            configured_module.create_test()

        assert configured_module._tests == {'default_test': first}

    def test_same_name_allowed_on_different_modules(
        self, configured_module, pdk, tmp_dir
    ):
        """Test-name uniqueness is scoped to one owning module."""
        other = Module(
            'PDK_skywater130', os.path.join(tmp_dir, 'other'), 'other.cir',
            pdk_details=pdk, module_name='other_inv',
        )

        first = configured_module.create_test('shared')
        second = ModuleTestbench(other, 'shared')

        assert configured_module._tests['shared'] is first
        assert other._tests['shared'] is second

    def test_default_name(self, configured_module):
        """Default test name is 'default_test'."""
        test = ModuleTestbench(configured_module)
        assert test.test_name == 'default_test'

    def test_metrics_initially_empty(self, configured_module):
        """Metrics list starts empty."""
        test = ModuleTestbench(configured_module, 'mt')
        assert test._metrics == []

    def test_netlist_initially_none(self, configured_module):
        """Internal netlist is None until add_module_under_test or add_source."""
        test = ModuleTestbench(configured_module, 'mt')
        assert test.netlist is None

    def test_module_not_added_initially(self, configured_module):
        """_module_added is False initially."""
        test = ModuleTestbench(configured_module, 'mt')
        assert test._module_added is False


# ---------------------------------------------------------------------------
# add_module_under_test tests
# ---------------------------------------------------------------------------

class TestAddModuleUnderTest:

    def test_sets_module_added_flag(self, configured_module):
        """add_module_under_test sets _module_added to True."""
        test = configured_module.create_test('t1')
        test.add_module_under_test({
            'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0'
        })
        assert test._module_added is True

    def test_creates_internal_netlist(self, configured_module):
        """add_module_under_test creates the internal PyNetlist."""
        test = configured_module.create_test('t1')
        test.add_module_under_test({
            'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0'
        })
        assert test.netlist is not None

    def test_dut_in_netlist_elem_dict(self, configured_module):
        """The DUT instance appears in the internal netlist's elem_dict."""
        test = configured_module.create_test('t1')
        test.add_module_under_test({
            'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0'
        })
        assert 'dut' in test.netlist.elem_dict

    def test_returns_self_for_chaining(self, configured_module):
        """add_module_under_test returns self for method chaining."""
        test = configured_module.create_test('t1')
        result = test.add_module_under_test({
            'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0'
        })
        assert result is test

    def test_double_add_raises(self, configured_module):
        """Calling add_module_under_test twice raises ValueError."""
        test = configured_module.create_test('t1')
        test.add_module_under_test({
            'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0'
        })
        with pytest.raises(ValueError, match="already added"):
            test.add_module_under_test({
                'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0'
            })


# ---------------------------------------------------------------------------
# add_source delegation tests
# ---------------------------------------------------------------------------

class TestAddSourceDelegation:

    def test_source_in_internal_netlist(self, configured_module):
        """add_source() creates a source in the internal netlist."""
        test = configured_module.create_test('t1')
        test.add_module_under_test({
            'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0'
        })
        test.add_source(sources.VoltageSource('vsup', positive='vdd', negative='0', dc=1.8))

        assert 'vsup' in test.netlist.sig_sources_dict

    def test_source_returns_self(self, configured_module):
        """add_source returns self for chaining."""
        test = configured_module.create_test('t1')
        test.add_module_under_test({
            'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0'
        })
        result = test.add_source(sources.VoltageSource('vsup', positive='vdd', negative='0', dc=1.8))
        assert result is test

    def test_source_before_add_module_creates_netlist(self, configured_module):
        """add_source before add_module_under_test creates the internal netlist."""
        test = configured_module.create_test('t1')
        test.add_source(sources.VoltageSource('vsup', positive='vdd', negative='0', dc=1.8))
        assert test.netlist is not None
        assert 'vsup' in test.netlist.sig_sources_dict

    def test_source_with_empty_terminals_never_reaches_the_testbench(self, configured_module):
        """TASK-76: the placeholder ``positive='', negative=''`` this test once
        used is rejected at construction, before ``add_source`` can run."""
        test = configured_module.create_test('t1')
        with pytest.raises(ValueError, match=r"VoltageSource 'vsup' terminal 'positive'"):
            test.add_source(sources.VoltageSource('vsup', positive='', negative='', dc=1.8))
        assert test.netlist is None


# ---------------------------------------------------------------------------
# add_sim_type delegation tests
# ---------------------------------------------------------------------------

class TestAddSimTypeDelegation:

    def test_simtype_in_internal_netlist(self, configured_module):
        """add_sim_type() populates the internal netlist's sim_analysis_dict."""
        test = configured_module.create_test('t1')
        test.add_module_under_test({
            'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0'
        })
        test.add_sim_type(Transient(step=1e-9, stop=1e-7))

        assert 'transient' in test.netlist.sim_analysis_dict

    def test_simtype_returns_self(self, configured_module):
        """add_sim_type returns self for chaining."""
        test = configured_module.create_test('t1')
        test.add_module_under_test({
            'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0'
        })
        result = test.add_sim_type(Transient(step=1e-9, stop=1e-7))
        assert result is test


# ---------------------------------------------------------------------------
# add_custom_line delegation tests
# ---------------------------------------------------------------------------

class TestAddCustomLineDelegation:

    def test_custom_line_delegated(self, configured_module):
        """add_custom_line() appends to internal netlist's custom_lines."""
        test = configured_module.create_test('t1')
        test.add_module_under_test({
            'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0'
        })
        test.add_custom_line('cload vout 0 10p')
        assert 'cload vout 0 10p' in test.netlist.custom_lines

    def test_custom_line_returns_self(self, configured_module):
        """add_custom_line returns self for chaining."""
        test = configured_module.create_test('t1')
        result = test.add_custom_line('rload out vdd 1k')
        assert result is test


# ---------------------------------------------------------------------------
# add_metric tests
# ---------------------------------------------------------------------------

class TestAddMetric:

    def test_metric_added_to_list(self, configured_module):
        """add_metric() appends to the _metrics list."""
        test = configured_module.create_test('t1')
        test.add_metric('vout_avg', lambda nr: 1.0)
        assert len(test._metrics) == 1
        assert test._metrics[0]['name'] == 'vout_avg'

    def test_multiple_metrics(self, configured_module):
        """Multiple metrics can be added."""
        test = configured_module.create_test('t1')
        test.add_metric('m1', lambda nr: 1.0)
        test.add_metric('m2', lambda nr: 2.0)
        assert len(test._metrics) == 2

    def test_metric_preserves_chaining(self, configured_module):
        """add_metric returns a handle that preserves builder chaining."""
        test = configured_module.create_test('t1')
        result = test.add_metric('m1', lambda nr: 1.0)
        assert result.add_custom_line('* chained after metric') is test

    def test_metric_extractor_stored(self, configured_module):
        """The extractor function is stored and callable."""
        func = lambda nr: 42
        test = configured_module.create_test('t1')
        test.add_metric('answer', func)
        assert test._metrics[0]['extractor'] is func


# ---------------------------------------------------------------------------
# run() tests
# ---------------------------------------------------------------------------

class TestRun:

    def _setup_test(self, configured_module):
        """Helper to set up a fully configured test."""
        test = configured_module.create_test('run_test')
        test.add_module_under_test({
            'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0'
        })
        test.add_source(sources.VoltageSource('vsup', positive='vdd', negative='0', dc=1.8))
        test.add_source(sources.VoltageSource('vin_src', positive='vin', negative='0', dc=0.9))
        test.add_sim_type(Transient(step=1e-9, stop=1e-7))
        return test

    def test_run_without_module_raises(self, configured_module):
        """run() without add_module_under_test raises ValueError."""
        test = configured_module.create_test('t1')
        with pytest.raises(ValueError, match="add_module_under_test"):
            test.run()

    @needs_ngspice

    def test_run_with_mock_returns_dict(self, configured_module, mock_ngspice):
        """run() with mocked simulation returns a result dict with expected keys."""
        test = self._setup_test(configured_module)

        # Mock load_file and read_raw_file since we don't have real sim output
        test.netlist.load_file = MagicMock()
        test.netlist.read_raw_file = MagicMock()

        result = test.run(corner='tt')

        assert isinstance(result, SpecReport)
        assert result.test_name == 'run_test'
        assert result.module_name == 'test_inv'
        assert hasattr(result, 'metrics')
        assert hasattr(result, 'sim_results')

    @needs_ngspice

    def test_run_calls_generate_and_simulate(self, configured_module, mock_ngspice):
        """run() calls generate_netlist and run_simulator."""
        test = self._setup_test(configured_module)

        test.netlist.load_file = MagicMock()
        test.netlist.read_raw_file = MagicMock()

        report = test.run(corner='tt')

        # Verify ngspice was called
        assert mock_ngspice['called']

        # Verify netlist file was generated
        assert os.path.isfile(report.run.deck_path)

    @needs_ngspice

    def test_run_extracts_metrics(self, configured_module, mock_ngspice):
        """run() calls each metric extractor with the netlist object."""
        test = self._setup_test(configured_module)

        extractor_called = {'called': False, 'arg': None}

        def mock_extractor(nr):
            extractor_called['called'] = True
            extractor_called['arg'] = nr
            return 42.0

        test.add_metric('test_metric', mock_extractor)

        test.netlist.load_file = MagicMock()
        test.netlist.read_raw_file = MagicMock()

        result = test.run(corner='tt')

        assert extractor_called['called']
        assert extractor_called['arg'] is test.netlist
        assert result.metrics['test_metric'] == 42.0

    @needs_ngspice

    def test_run_handles_metric_error(self, configured_module, mock_ngspice):
        """run() gracefully handles metric extraction errors."""
        test = self._setup_test(configured_module)

        def bad_extractor(nr):
            raise RuntimeError("metric failed")

        test.add_metric('bad_metric', bad_extractor)

        test.netlist.load_file = MagicMock()
        test.netlist.read_raw_file = MagicMock()

        result = test.run(corner='tt')

        # Should contain error string, not raise
        assert 'Error' in str(result.metrics['bad_metric'])

    @needs_ngspice

    def test_run_with_different_corner(self, configured_module, mock_ngspice):
        """run() passes corner parameter to generate_netlist."""
        test = self._setup_test(configured_module)

        test.netlist.load_file = MagicMock()
        test.netlist.read_raw_file = MagicMock()

        report = test.run(corner='ff')

        # Verify the netlist file contains 'ff' corner
        content = open(report.run.deck_path).read()
        assert 'ff' in content


# ---------------------------------------------------------------------------
# has_tests / run_all_tests via Module tests
# ---------------------------------------------------------------------------

class TestModuleTestIntegration:

    def test_has_tests_via_module(self, configured_module):
        """has_tests() returns True after create_test()."""
        configured_module.create_test('t1')
        assert configured_module.has_tests() is True

    def test_run_all_tests_runs_all(self, configured_module, mock_ngspice):
        """run_all_tests() runs all defined tests and returns results dict."""
        test1 = configured_module.create_test('test_a')
        test1.add_module_under_test({
            'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0'
        })
        test1.add_source(sources.VoltageSource('vsup', positive='vdd', negative='0', dc=1.8))
        test1.add_sim_type(Transient(step=1e-9, stop=1e-7))

        # Mock load_file/read_raw_file on the internal netlist
        test1.netlist.load_file = MagicMock()
        test1.netlist.read_raw_file = MagicMock()

        results = configured_module.run_all_tests(recursive=False)

        assert isinstance(results, dict)
        # Key format is "module_name.test_name"
        expected_key = f"{configured_module.module_name}.test_a"
        assert expected_key in results

    def test_run_test_nonexistent_raises(self, configured_module):
        """run_test() with unknown name raises ValueError."""
        with pytest.raises(ValueError, match="No test"):
            configured_module.run_test('nonexistent')

    def test_get_all_testable_modules(self, configured_module):
        """get_all_testable_modules() includes module with tests."""
        configured_module.create_test('t1')
        testable = configured_module.get_all_testable_modules()
        assert len(testable) == 1
        assert testable[0]['module'] == configured_module.module_name
        assert 't1' in testable[0]['tests']

    def test_get_all_testable_modules_empty(self, configured_module):
        """get_all_testable_modules() returns empty when no tests defined."""
        testable = configured_module.get_all_testable_modules()
        assert len(testable) == 0


# ---------------------------------------------------------------------------
# Method chaining test
# ---------------------------------------------------------------------------

class TestMethodChaining:

    def test_full_chain(self, configured_module):
        """All builder methods support chaining."""
        test = configured_module.create_test('chain')
        result = (
            test
            .add_module_under_test({
                'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0'
            })
            .add_source(sources.VoltageSource('vsup', positive='vdd', negative='0', dc=1.8))
            .add_sim_type(Transient(step=1e-9, stop=1e-7))
            .add_custom_line('cload vout 0 10p')
            .add_metric('dummy', lambda nr: 0)
        )
        assert result.add_custom_line("* chained after metric") is test


# ---------------------------------------------------------------------------
# Lazy netlist creation in add_sim_type / run() guard (lines 105, 144)
# ---------------------------------------------------------------------------

class TestLazyNetlistCreation:

    def test_addSimType_before_add_module_creates_netlist(self, configured_module):
        """add_sim_type() with netlist=None triggers lazy _create_netlist()."""
        test = configured_module.create_test('lazy_sim')
        assert test.netlist is None
        test.add_sim_type(Transient(step=1e-9, stop=1e-7))
        assert test.netlist is not None
        assert 'transient' in test.netlist.sim_analysis_dict

    def test_run_raises_when_module_added_but_netlist_none(self, configured_module):
        """run() raises ValueError when _module_added=True but netlist is None."""
        test = configured_module.create_test('no_netlist')
        test._module_added = True  # bypass normal setup
        # netlist stays None
        with pytest.raises(ValueError):
            test.run(corner='tt')


# ---------------------------------------------------------------------------
# TestRunCorners
# ---------------------------------------------------------------------------

def _make_spec_report(corner, *args, **kwargs):
    """Helper: a passing SpecReport for the given corner.

    Patched onto ``_prepare_run`` (TASK-301.3: run_corners() no longer
    calls ``run()`` directly, it drives prepare/simulate/evaluate itself) —
    a non-``_RunSpec`` return is treated exactly like a cache hit, so
    run_corners() takes it as this corner's finished report without
    simulating or evaluating anything. ``*args, **kwargs`` swallow the real
    call's ``corner_subdir=``/PVT kwargs.
    """
    r = SpecReport(test_name=corner, module_name='inv', metrics={'gain': 65.0})
    r.check('Gain > 60', 'gain', '>', 60.0)
    return r


class TestRunCorners:

    def test_run_corners_returns_corner_results(self, configured_module):
        """run_corners() returns a CornerResults object."""
        from silk.circuit.spec_report import CornerResults
        from unittest.mock import patch
        test = configured_module.create_test('c1')
        with patch.object(test, '_prepare_run', side_effect=_make_spec_report):
            result = test.run_corners(corners=('tt', 'ff'))
        assert isinstance(result, CornerResults)

    def test_run_corners_keys(self, configured_module):
        """run_corners() dict keys match the requested corners."""
        from unittest.mock import patch
        test = configured_module.create_test('c2')
        with patch.object(test, '_prepare_run', side_effect=_make_spec_report):
            result = test.run_corners(corners=('tt', 'ff'))
        assert set(result.keys()) == {'tt', 'ff'}

    def test_run_corners_default_five(self, configured_module):
        """Default corners are tt, ff, ss, sf, fs."""
        from unittest.mock import patch
        test = configured_module.create_test('c3')
        with patch.object(test, '_prepare_run', side_effect=_make_spec_report):
            result = test.run_corners()
        assert set(result.keys()) == {'tt', 'ff', 'ss', 'sf', 'fs'}

    def test_run_corners_values_are_spec_reports(self, configured_module):
        """Each value in run_corners() result is a SpecReport."""
        from unittest.mock import patch
        test = configured_module.create_test('c4')
        with patch.object(test, '_prepare_run', side_effect=_make_spec_report):
            result = test.run_corners(corners=('tt',))
        assert all(isinstance(r, SpecReport) for r in result.values())

    def test_run_corners_passed_all_pass(self, configured_module):
        """CornerResults.passed is True when all corners pass."""
        from unittest.mock import patch
        test = configured_module.create_test('c5')
        with patch.object(test, '_prepare_run', side_effect=_make_spec_report):
            result = test.run_corners(corners=('tt', 'ff'))
        assert result.passed is True

    def test_run_corners_passed_one_fail(self, configured_module):
        """CornerResults.passed is False when any corner fails."""
        from silk.circuit.spec_report import SpecReport as SR
        from unittest.mock import patch

        def make_failing(corner, *args, **kwargs):
            r = SR(test_name=corner, module_name='inv', metrics={'gain': 50.0})
            r.check('Gain > 60', 'gain', '>', 60.0)
            return r

        test = configured_module.create_test('c6')
        with patch.object(test, '_prepare_run', side_effect=make_failing):
            result = test.run_corners(corners=('tt',))
        assert result.passed is False

    def test_run_corners_getitem(self, configured_module):
        """CornerResults[corner] returns the SpecReport for that corner."""
        from unittest.mock import patch
        test = configured_module.create_test('c7')
        with patch.object(test, '_prepare_run', side_effect=_make_spec_report):
            result = test.run_corners(corners=('tt',))
        assert result['tt'].passed is True

    def test_corner_results_summary_contains_pass(self, configured_module):
        """CornerResults.summary() contains PASS when all pass."""
        from unittest.mock import patch
        test = configured_module.create_test('c8')
        with patch.object(test, '_prepare_run', side_effect=_make_spec_report):
            result = test.run_corners(corners=('tt',))
        assert 'PASS' in result.summary()

    def test_corner_results_importable_from_silk(self):
        from silk import CornerResults
        assert CornerResults is not None


# ---------------------------------------------------------------------------
# ModuleTestbench.add(source) front door (mirrors Module.add — Task 2)
# ---------------------------------------------------------------------------

class TestTestbenchAdd:

    def _tb(self, configured_module):
        """Minimal ModuleTestbench, built via the shared `configured_module`
        fixture the same way every other test in this module does."""
        return configured_module.create_test('add_test')

    def test_add_routes_source_to_add_source(self, monkeypatch, configured_module):
        """add() must delegate each source to add_source and return self (chaining)."""
        tb = self._tb(configured_module)
        seen = []
        monkeypatch.setattr(tb, 'add_source', lambda s: seen.append(s))
        src = sources.vdc('vdd', p='vdd', n='gnd', dc=1.8)
        assert tb.add(src) is tb
        assert seen == [src]

    def test_add_rejects_non_source(self, configured_module):
        """add() forwards non-sources to Module.add() (TASK-350), which
        raises its own TypeError for anything that is not an element."""
        tb = self._tb(configured_module)
        with pytest.raises(TypeError, match=r'Module\.add\(\)'):
            tb.add(object())

    def test_add_routes_ideal_passive_into_netlist(self, configured_module):
        """add() routes an ideal R/C/L into the testbench's own netlist via
        Module.add() (elem_dict), exactly like a plain Module — not through
        add_source (TASK-312.6, Part 1)."""
        from silk.elements import elements
        tb = self._tb(configured_module)
        r = elements.resistor('rload', p='out', n='0', value=1e3)
        assert tb.add(r) is tb
        assert tb.netlist.elem_dict['rload']['device'] is r
        assert 'rload' not in tb.netlist.passives_dict

    @needs_ngspice
    def test_add_and_add_passive_produce_identical_deck_and_currents(self, tmp_dir):
        """AC#1 runtime check: an ideal R added via tb.add() vs the same R
        added via tb.add_passive() must emit identical deck element lines
        (order/banner may differ, per TASK-312.1's precedent) and identical
        annotated per-pin currents from a real transient run.

        PDK-free (TASK-71 rung): no PDK device anywhere in the hierarchy, so
        this runs under bare ngspice with no PDK installed.
        """
        from silk.pdk import load_pdk
        from silk.circuit.modules import Module
        from silk.circuit.circuit_elements import PortDir
        from silk.elements import elements

        pdk = load_pdk('sky130', pdk_root='/nonexistent-pdk-root')

        def _build(build):
            dut = Module('PDK_skywater130', tmp_dir, f'dut_{build}.cir',
                         pdk_details=pdk, module_name='buf')
            dut.define_module_ports({'in': PortDir.IN, 'out': PortDir.OUT,
                                     'vss': PortDir.GROUND})
            with dut:
                elements.resistor('rseries', p='in', n='out', value=1e3)
            test = dut.create_test(f'tb_{build}')
            test.add_module_under_test({'in': 'vin', 'out': 'vout', 'vss': '0'})
            test.add_source(sources.VoltageSource('vin', positive='vin',
                                                   negative='0', dc=1.0))
            r = elements.resistor('rload', p='vout', n='0', value=1e3)
            if build == 'add':
                test.add(r)
            else:
                test.add_passive(r)
            test.add_sim_type(Transient(1e-9, 20e-9))
            dut.save_device_currents = True
            return test

        t_add = _build('add')
        t_ap = _build('add_passive')
        r_add = t_add.run(corner='tt')
        r_ap = t_ap.run(corner='tt')

        with open(r_add.run.deck_path) as f:
            deck_add = f.read()
        with open(r_ap.run.deck_path) as f:
            deck_ap = f.read()

        def elem_lines(deck):
            return sorted(l for l in deck.splitlines() if l[:1] in ('R', 'C', 'L', 'X'))
        assert elem_lines(deck_add) == elem_lines(deck_ap)

        t_add.netlist.annotate_currents()
        t_ap.netlist.annotate_currents()

        def pin_by_instance(net, instance):
            return next(b for b in net.pins if b.instance == instance)

        p_add = pin_by_instance(t_add.netlist.net_dict['vout'], 'rload')
        p_ap = pin_by_instance(t_ap.netlist.net_dict['vout'], 'rload')
        assert p_add.avg_current == p_ap.avg_current
        assert p_add.avg_current is not None


# ---------------------------------------------------------------------------
# add_metric declarative overload (Task 3): type= / signal= / **params
# ---------------------------------------------------------------------------

class TestDeclarativeAddMetric:

    def _tb(self, configured_module):
        """Minimal ModuleTestbench, built via the shared `configured_module`
        fixture the same way every other test in this module does."""
        return configured_module.create_test('declarative_metric_test')

    def test_declarative_builds_extractor(self, configured_module):
        """Declarative form (type=, signal=) appends a metric with a callable extractor."""
        tb = self._tb(configured_module)
        tb.add_metric('gain', type='dc_gain_dB', signal='xdut.vout')
        m = tb._metrics[-1]
        assert m['name'] == 'gain' and callable(m['extractor'])

    def test_callable_form_unchanged(self, configured_module):
        """Regression guard: the pre-existing callable form is untouched."""
        tb = self._tb(configured_module)
        f = lambda ckt: 1.0
        tb.add_metric('g', f)
        assert tb._metrics[-1]['extractor'] is f

    def test_both_or_neither_raises(self, configured_module):
        """Exactly one of extractor_func/type is required."""
        tb = self._tb(configured_module)
        with pytest.raises(TypeError):
            tb.add_metric('g')                         # neither
        with pytest.raises(TypeError):
            tb.add_metric('g', (lambda c: 0), type='dc_gain_dB', signal='s')  # both

    def test_declarative_requires_signal(self, configured_module):
        """Declarative form without signal= raises TypeError."""
        tb = self._tb(configured_module)
        with pytest.raises(TypeError, match='signal'):
            tb.add_metric('g', type='dc_gain_dB')

    def test_extractor_calls_metric_on_loaded_signal(self, monkeypatch, configured_module):
        """The built extractor resolves fn via get_metric(type) and calls it as
        fn(ckt.load_signal(signal), **params) at extraction time — not eagerly."""
        tb = self._tb(configured_module)
        sentinel_loaded = object()
        sentinel_result = object()
        calls = []

        def fake_fn(sig: TransientSignal, **params):
            calls.append((sig, params))
            return sentinel_result

        # add_metric reads the metric's signal-input slots out of the schema
        # to bind them (TASK-198), so the stand-in has to be describable by it:
        # one annotated Signal parameter and the @metric metadata, without
        # entering the real registry. Since TASK-196 the annotation must also
        # name a KIND — schema v2 publishes one per slot and refuses a bare
        # Signal — so the stand-in uses the same ``Transient`` alias the real
        # transient metrics do (imported as TransientSignal here).
        fake_fn._silk_metric = {'domain': 'dc', 'unit': 'dB', 'param_units': {}}
        monkeypatch.setattr('silk.signals.schema.get_metric', lambda name: fake_fn)
        tb.add_metric('g', type='dc_gain_dB', signal='xdut.vout', foo=1)

        fake_ckt = MagicMock()
        fake_ckt.load_signal.return_value = sentinel_loaded

        extractor = tb._metrics[-1]['extractor']
        result = extractor(fake_ckt)

        fake_ckt.load_signal.assert_called_once_with('xdut.vout')
        assert calls == [(sentinel_loaded, {'foo': 1})]
        assert result is sentinel_result


# ---------------------------------------------------------------------------
# xdut probe contract lock-in (Task 2): DUT is instantiated as 'dut' -> xdut.*
# ---------------------------------------------------------------------------

class TestXdutContract:

    def test_dut_instance_is_xdut(self, configured_module):
        """add_module_under_test instantiates the DUT as instance 'dut', which
        becomes 'xdut' in the generated SPICE (X-prefix) and thus 'xdut.<port>'
        in signal probes. Locks existing behavior."""
        tb = configured_module.create_test('xdut_test')
        tb.add_module_under_test({
            'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0'
        })
        assert 'dut' in tb.netlist.elem_dict


# ---------------------------------------------------------------------------
# run_corners() / run() corner defaults derive from the testbench PDK
# (Task 6 — corner reconciliation)
# ---------------------------------------------------------------------------

def test_run_corners_default_derives_from_pdk(monkeypatch):
    from silk.circuit.module_test import ModuleTestbench
    from silk.circuit.spec_report import SpecReport
    calls = []
    tb = ModuleTestbench.__new__(ModuleTestbench)          # bare instance
    tb.pdk_details = type('P', (), {'corners': {'tt': {'kind': 'process'},
        'ss': {'kind': 'process'}, 'll': {'kind': 'rc'}}, 'default_corner': 'tt'})()
    # Match an explicitly selected netlist backend for effective resolution.
    tb.netlist = type('N', (), {
        'simulator': 'ngspice', '_simulator_explicit': True})()
    # A cache hit returns a SpecReport, including its measured metric values.
    report = SpecReport(test_name='t', module_name='dut', metrics={})
    monkeypatch.setattr(tb, '_prepare_run',
                        lambda corner, **kw: calls.append(corner) or report)
    tb.run_corners()                                       # no explicit corners
    assert set(calls) == {'tt', 'ss'}                      # process-kind only


class TestPerTestOutputIsolation:
    """2026-08-24 results-file review, B1: both runners hardcode
    <dir>/rawfile.raw, so testbenches sharing the DUT's dir_folder silently
    clobbered each other's results. Each testbench now owns a
    test_<module>~<test>/ subdirectory."""

    def test_each_testbench_gets_its_own_directory(self, pdk, tmp_dir):
        m = Module('PDK_skywater130', tmp_dir, 'dut.cir', pdk_details=pdk,
                   module_name='isodut')
        t1 = ModuleTestbench(m, 'alpha')
        t2 = ModuleTestbench(m, 'beta')
        t1._create_netlist()
        t2._create_netlist()
        d1 = t1.netlist.dir_folder
        d2 = t2.netlist.dir_folder
        assert d1 != d2
        assert d1 == os.path.join(tmp_dir, 'test_isodut~alpha')
        assert d2 == os.path.join(tmp_dir, 'test_isodut~beta')
        # the future rawfiles therefore cannot collide
        assert os.path.join(d1, 'rawfile.raw') != os.path.join(d2, 'rawfile.raw')

    def test_deck_lands_inside_the_test_directory(self, pdk, tmp_dir):
        m = Module('PDK_skywater130', tmp_dir, 'dut.cir', pdk_details=pdk,
                   module_name='isodut')
        t = ModuleTestbench(m, 'gamma')
        t._create_netlist()
        assert os.path.join(t.netlist.dir_folder, t.netlist.netlistname).startswith(
            os.path.join(tmp_dir, 'test_isodut~gamma'))


class TestLoadFileResolution:
    """2026-08-24 results-file review, B2: load_file('rawfile.raw') was
    CWD-relative (quickstart's own snippet crashed in a fresh process) and
    was silently masked by the in-memory cache in-session."""

    def test_relative_name_resolves_against_dir_folder(self, pdk, tmp_dir):
        m = Module('PDK_skywater130', tmp_dir, 'r.cir', pdk_details=pdk)
        target = os.path.join(tmp_dir, 'some.raw')
        open(target, 'w').write('')
        m.load_file('some.raw')
        assert m.dataFile == target

    def test_absolute_path_kept_verbatim(self, pdk, tmp_dir):
        m = Module('PDK_skywater130', tmp_dir, 'r.cir', pdk_details=pdk)
        m.load_file('/nonexistent/abs.raw')
        assert m.dataFile == '/nonexistent/abs.raw'

    def test_pointing_at_a_different_file_drops_the_cache(self, pdk, tmp_dir):
        m = Module('PDK_skywater130', tmp_dir, 'r.cir', pdk_details=pdk)

        class _FakeRaw:
            source_path = os.path.join(tmp_dir, 'old.raw')
        m._raw = _FakeRaw()
        other = os.path.join(tmp_dir, 'new.raw')
        open(other, 'w').write('')
        m.load_file('new.raw')
        assert m._raw is None          # cache dropped -> read_raw_file loads the file

    def test_same_file_keeps_the_cache(self, pdk, tmp_dir):
        m = Module('PDK_skywater130', tmp_dir, 'r.cir', pdk_details=pdk)
        same = os.path.join(tmp_dir, 'rawfile.raw')
        open(same, 'w').write('')

        class _FakeRaw:
            source_path = same
        fake = _FakeRaw()
        m._raw = fake
        m.load_file(f'{tmp_dir}/rawfile.raw')
        assert m._raw is fake          # simulate()'s own path stays fast


class TestResultsFileProperty:
    """2026-08-25 review, B1: ``ModuleTestbench`` had no ``.results_file`` —
    the path lives on ``self.netlist`` — so the documented
    ``emit_view(tb.results_file, ...)``/``report.add_waveform_view(tb)``
    bridge was unrunnable. ``results_file`` now proxies
    ``self.netlist.results_file``, mirroring ``Module``'s own
    None-before-simulation contract rather than raising."""

    def test_none_before_netlist_created(self, configured_module):
        test = ModuleTestbench(configured_module, 'mt')
        assert test.netlist is None
        assert test.results_file is None

    def test_none_after_netlist_created_but_unsimulated(self, configured_module):
        test = ModuleTestbench(configured_module, 'mt')
        test._create_netlist()
        assert test.netlist is not None
        assert test.results_file is None

    def test_proxies_netlist_results_file(self, configured_module):
        test = ModuleTestbench(configured_module, 'mt')
        test._create_netlist()
        test.netlist.results_file = '/fake/path/rawfile.raw'
        assert test.results_file == '/fake/path/rawfile.raw'

    @needs_pdk
    @needs_ngspice
    def test_live_run_populates_results_file_in_per_test_subdir(self, tmp_dir):
        """End-to-end with a real PDK + ngspice (not the fake-lib-path
        `pdk` fixture, which never actually simulates): after run(),
        tb.results_file is a real rawfile inside the per-test subdirectory,
        and both documented consumers — emit_view() directly, and
        report.add_waveform_view(tb) — work off of it."""
        from silk import load_pdk
        from silk.circuit.report import CircuitReport, TopologySummary
        from silk.circuit.interface_check import AnnotationCoverage
        from silk.signals.view import emit_view

        pdk = load_pdk('sky130')
        mod = Module('PDK_skywater130', tmp_dir, 'dut.cir', pdk_details=pdk)
        mod.define_module_name('live_inv')
        mod.define_module_nets({'in': None, 'out': None, 'vdd': None, 'vss': None})
        mod.add(pdk.nmos1p8('mn', d='out', g='in', s='vss', b='vss', w=1, l=0.15, m=1))
        mod.add(pdk.pmos1p8('mp', d='out', g='in', s='vdd', b='vdd', w=2, l=0.15, m=1))

        test = mod.create_test('live')
        test.add_module_under_test({
            'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0'
        })
        test.add_source(sources.VoltageSource('vsup', positive='vdd', negative='0', dc=1.8))
        test.add_source(sources.VoltageSource('vin_src', positive='vin', negative='0', dc=0.9))
        test.add_sim_type(Transient(step=1e-9, stop=1e-7))

        assert test.results_file is None       # unsimulated: no crash, no value
        test.run(corner='tt')

        expected_base = os.path.join(tmp_dir, 'test_live_inv~live')
        assert test.results_file is not None
        assert os.path.isfile(test.results_file)
        expected_dir = os.path.dirname(os.path.abspath(test.results_file))
        assert os.path.dirname(expected_dir) == expected_base

        # Consumer 1: emit_view() directly off tb.results_file, as taught by
        # docs/agent-guide.md's "make it a habit" snippet.
        view_out = os.path.join(tmp_dir, 'live_view.html')
        result = emit_view(test.results_file, view_out)
        assert os.path.isfile(result.html_path)
        assert os.path.dirname(os.path.abspath(result.html_path)) == expected_dir \
            or os.path.dirname(os.path.abspath(result.html_path)) == tmp_dir

        # Consumer 2: report.add_waveform_view(tb), as taught by
        # docs/reports.md.
        report = CircuitReport(
            module_name='live_inv',
            generated_at='2026-08-25T00:00:00',
            topology=TopologySummary(module_name='live_inv', instances=[], total_nets=0),
            annotation=AnnotationCoverage(total=0, annotated=0, unannotated_ports=[]),
            violations=[],
            simulation=None,
        )
        report.add_waveform_view(test)
        assert len(report.waveform_views) == 1
        html_path, label = report.waveform_views[0]
        assert os.path.isfile(html_path)
        assert os.path.dirname(os.path.abspath(html_path)) == expected_dir
        # ModuleTestbench proxies .results_file but not .module_name (that
        # gap is pre-existing and out of scope for B1 — see
        # add_waveform_view's own comment on context_module), so the label
        # falls back to the rawfile's basename rather than 'live_inv'.
        assert label


class TestGetTest:
    def test_returns_registered_object_without_changing_registry(self, configured_module):
        mod = configured_module
        first = mod.create_test('first')
        second = mod.create_test('second')
        assert mod.get_test('first') is first
        assert mod.get_test('second') is second
        assert mod.list_tests() == ['first', 'second']

    @pytest.mark.parametrize('names', [[], ['ac', 'tran']])
    def test_unknown_name_identifies_module_and_available_tests(self, configured_module, names):
        from silk.exceptions import ConfigurationError
        mod = configured_module
        for name in names:
            mod.create_test(name)
        with pytest.raises(ConfigurationError) as caught:
            mod.get_test('missing')
        message = str(caught.value)
        assert 'missing' in message
        assert 'test_inv' in message
        if names:
            assert all(repr(name) in message for name in names)
        else:
            assert 'none' in message.lower()
        assert mod.list_tests() == names

    def test_does_not_search_submodule_tests(self, configured_module, pdk, tmp_dir):
        from silk.exceptions import ConfigurationError
        child = configured_module
        child.create_test('child_only')
        parent = Module('PDK_skywater130', tmp_dir, 'parent.cir',
                          pdk_details=pdk, module_name='parent')
        parent.add_module(child, 'dut', nets={
            'in': 'in', 'out': 'out', 'vdd': 'vdd', 'vss': 'gnd'})
        with pytest.raises(ConfigurationError, match='parent'):
            parent.get_test('child_only')
