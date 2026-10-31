"""Tests for SpecReport and ModuleTestbench.add_spec()."""
import os

import pytest
import silk.pdk
from silk.circuit.spec_report import SpecReport, SpecResult


class TestSpecReportCheck:
    def setup_method(self):
        self.report = SpecReport('tb', 'ldo', {'gain_dB': 65.0, 'bw': 1e6, 'pm': 55.0})

    def test_passing_spec_greater_than(self):
        self.report.check('Gain > 60 dB', 'gain_dB', '>', 60.0)
        assert self.report.results[-1].passed

    def test_failing_spec_greater_than(self):
        self.report.check('Gain > 70 dB', 'gain_dB', '>', 70.0)
        assert not self.report.results[-1].passed

    def test_passing_spec_less_than(self):
        self.report.check('BW < 2 MHz', 'bw', '<', 2e6)
        assert self.report.results[-1].passed

    def test_failing_spec_less_than(self):
        self.report.check('BW < 500 kHz', 'bw', '<', 500e3)
        assert not self.report.results[-1].passed

    def test_passing_spec_gte(self):
        self.report.check('PM >= 45', 'pm', '>=', 45.0)
        assert self.report.results[-1].passed

    def test_passing_spec_eq(self):
        self.report.check('Exact', 'gain_dB', '==', 65.0)
        assert self.report.results[-1].passed

    def test_equality_is_exact_like_first_class_spec_evaluation(self):
        report = SpecReport('t', 'm', {'x': 1.0000000005})
        report.check('Exact', 'x', '==', 1.0)
        assert not report.results[-1].passed

    def test_missing_metric_fails(self):
        self.report.check('Unknown metric', 'nonexistent', '>', 0.0)
        assert not self.report.results[-1].passed

    def test_error_string_metric_fails(self):
        report = SpecReport('t', 'm', {'gain': 'Error: sim failed'})
        report.check('Gain > 60', 'gain', '>', 60.0)
        assert not report.results[-1].passed

    def test_invalid_op_raises(self):
        with pytest.raises(ValueError, match='Unknown operator'):
            self.report.check('Bad op', 'gain_dB', '!=', 60.0)

    def test_chaining(self):
        result = (
            self.report
            .check('Gain > 60', 'gain_dB', '>', 60.0)
            .check('PM > 45', 'pm', '>', 45.0)
        )
        assert result is self.report
        assert len(self.report.results) == 2

    def test_check_preserves_declaration_identity(self):
        self.report.check(
            'Gain > 60', 'gain_dB', '>', 60.0,
            spec_key='gain', owner_path='$/core',
        )
        result = self.report.results[-1]
        assert result.spec_key == 'gain'
        assert result.owner_path == '$/core'


class TestSpecReportPassed:
    def test_all_pass(self):
        report = SpecReport('t', 'm', {'gain': 65.0, 'pm': 55.0})
        report.check('G', 'gain', '>', 60.0).check('PM', 'pm', '>', 45.0)
        assert report.passed

    def test_one_fail(self):
        report = SpecReport('t', 'm', {'gain': 55.0, 'pm': 55.0})
        report.check('G', 'gain', '>', 60.0).check('PM', 'pm', '>', 45.0)
        assert not report.passed

    def test_no_specs_passes(self):
        report = SpecReport('t', 'm', {'gain': 65.0})
        assert report.passed


class TestSpecReportSummary:
    def test_summary_contains_pass(self):
        report = SpecReport('tb', 'ldo', {'gain': 65.0})
        report.check('Gain > 60', 'gain', '>', 60.0)
        s = report.summary()
        assert 'PASS' in s
        assert 'tb' in s
        assert 'ldo' in s

    def test_summary_contains_fail(self):
        report = SpecReport('tb', 'ldo', {'gain': 50.0})
        report.check('Gain > 60', 'gain', '>', 60.0)
        s = report.summary()
        assert 'FAIL' in s

    def test_summary_no_specs(self):
        report = SpecReport('tb', 'ldo', {})
        assert 'no specs' in report.summary()

    def test_repr(self):
        report = SpecReport('tb', 'ldo', {'gain': 65.0})
        report.check('G', 'gain', '>', 60.0)
        r = repr(report)
        assert 'SpecReport' in r
        assert 'passed=True' in r


class TestSpecResultDataclass:
    def test_fields(self):
        r = SpecResult('name', 'metric', '>', 60.0, 65.0, True)
        assert r.name == 'name'
        assert r.metric_name == 'metric'
        assert r.op == '>'
        assert r.target == 60.0
        assert r.value == 65.0
        assert r.passed is True
        assert r.spec_key is None
        assert r.owner_path is None


class TestModuleTestbenchAddSpec:
    def test_add_spec_returns_spec_object(self, tmp_path):
        """add_spec() returns the new Spec; chaining is preserved via
        Spec.add_spec forwarder (covered in test_testbench_integration.py)."""
        from silk.circuit.modules import Module
        from silk.circuit.module_test import ModuleTestbench
        from silk.specs import Spec

        pdk_yaml = os.path.join(os.path.dirname(silk.pdk.__file__), 'data', 'skywater_130_device_details.yaml')
        try:
            from silk.pdk.basepdk import BasePDK
            pdk = BasePDK(pdk_yaml, library_location='/dev/null')
        except Exception:
            pytest.skip('PDK not available')

        mod = Module('PDK_skywater130', str(tmp_path), 'test.cir', pdk_details=pdk)
        mod.define_module_name('inv')
        mod.define_module_nets({'in': None, 'out': None, 'vdd': None, 'gnd': None})

        tb = ModuleTestbench(mod, 'test')
        result = tb.add_metric('gain', lambda ckt: 65.0)
        result2 = result.add_spec('Gain > 60', 'gain', '>', 60.0)
        assert isinstance(result2, Spec)
        assert result2.name == 'Gain > 60'
        assert len(tb._specs) == 1

    def test_specs_stored(self, tmp_path):
        from silk.circuit.modules import Module
        from silk.circuit.module_test import ModuleTestbench

        pdk_yaml = os.path.join(os.path.dirname(silk.pdk.__file__), 'data', 'skywater_130_device_details.yaml')
        try:
            from silk.pdk.basepdk import BasePDK
            pdk = BasePDK(pdk_yaml, library_location='/dev/null')
        except Exception:
            pytest.skip('PDK not available')

        mod = Module('PDK_skywater130', str(tmp_path), 'test.cir', pdk_details=pdk)
        mod.define_module_name('inv')
        mod.define_module_nets({'in': None, 'out': None, 'vdd': None, 'gnd': None})

        tb = ModuleTestbench(mod)
        tb.add_spec('Gain > 60', 'gain_dB', '>', 60.0)
        tb.add_spec('PM >= 45', 'pm', '>=', 45.0)
        assert len(tb._specs) == 2
        assert tb._specs[0]['op'] == '>'
        assert tb._specs[1]['target'] == 45.0


class TestImportPaths:
    def test_spec_report_importable_from_silk_circuit(self):
        from silk.circuit import SpecReport, SpecResult
        assert SpecReport is not None
        assert SpecResult is not None

    def test_spec_report_importable_directly(self):
        from silk.circuit.spec_report import SpecReport
        assert SpecReport is not None


@pytest.mark.parametrize('value', [3+4j, 3+0j, complex(float('nan'), 1)])
def test_complex_report_metric_is_preserved_and_spec_refused(value):
    report = SpecReport('ac', 'dut', {'mean': value, 'gain': 5.0})
    report.check('complex bound', 'mean', '>', 1)
    report.check('real bound', 'gain', '>', 1)
    assert not report.passed
    assert report.results[1].passed
    assert report.metrics['mean'] is value
    assert report.results[0].value is value
    for text in ('mean', 'complex', '.real', 'abs'):
        assert text in report.summary()
    assert report.metric_warnings == {}


def test_numpy_complex_spec_is_not_silently_real_cast():
    import numpy as np
    import warnings
    value = np.complex128(3+4j)
    with warnings.catch_warnings():
        warnings.simplefilter('error', np.exceptions.ComplexWarning)
        report = SpecReport('ac', 'dut', {'mean': value}).check('bound', 'mean', '>', 1)
    assert not report.passed
    assert report.metrics['mean'] is value
    assert report.results[0].value is value
    assert '.real' in report.summary() and 'abs' in report.summary()
