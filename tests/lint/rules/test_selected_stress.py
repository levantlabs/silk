"""Retained hierarchy readers preserve STR-001's circuit and waveform inputs."""
import copy
import pickle

import numpy as np
import pytest

from silk import Module, check_junction_overrange
from silk.circuit.junction_check import junction_stress_coverage
from tests._env import needs_ngspice, needs_pdk


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_ldo_stress_metric_and_report_reader_match_alone_and_placed(tmp_path, monkeypatch):
    from silk.examples.ldo_regulator_example import (
        build_ldo, attach_reliability_test, STRESS_METRIC,
    )
    monkeypatch.chdir(tmp_path)
    ldo = build_ldo(with_tests=False)
    attach_reliability_test(ldo)
    top = Module(ldo.pdk_name, str(tmp_path), 'top.cir', ldo.pdk_details)
    top.add_module(ldo, 'u_ldo', nets={name: name for name in ldo.pin_list})
    try:
        original = ldo.get_test('reliability').run()
        expected = check_junction_overrange(original.sim_results)
        coverage = junction_stress_coverage(original.sim_results).summary()
        assert original.metrics[STRESS_METRIC] == len(expected) == 7
        def signature(findings):
            return [(v.location.instance_path, v.location.terminal, v.measured_v)
                    for v in findings]
        routes = (
            lambda: next(iter(ldo.run_all_tests().values())),
            lambda: top.get_test('reliability', instance='u_ldo').run(),
            lambda: next(iter(top.run_all_tests().values())),
        )
        for run in routes:
            report = run()
            assert report.metrics[STRESS_METRIC] == len(expected)
            assert isinstance(report.sim_results, Module)
            assert report.sim_results.elem_dict == original.sim_results.elem_dict
            assert report.sim_results.elem_dict['dut']['device'] is ldo
            assert report.sim_results.net_dict == original.sim_results.net_dict
            actual = check_junction_overrange(report.sim_results)
            assert signature(actual) == signature(expected)
            assert junction_stress_coverage(report.sim_results).summary() == coverage
            for clone in (copy.copy, copy.deepcopy):
                reader = clone(report.sim_results)
                assert reader is not report.sim_results
                assert isinstance(reader, Module)
                np.testing.assert_array_equal(reader.load_signal('vreg').y,
                                              report.sim_results.load_signal('vreg').y)
                assert signature(check_junction_overrange(reader)) == signature(expected)
                assert junction_stress_coverage(reader).summary() == coverage
            # A reader owns live hierarchy references and a process-local
            # evidence owner; serialization must refuse rather than detach it.
            with pytest.raises(TypeError, match='EvidenceResults'):
                pickle.dumps(report.sim_results)
    finally:
        top.close()
        ldo.close()


def test_reader_without_delegate_has_no_recursive_fallback():
    from silk.circuit.test_placement import EvidenceResults
    reader = EvidenceResults.__new__(EvidenceResults)
    with pytest.raises(AttributeError, match='_module'):
        getattr(reader, 'missing')
