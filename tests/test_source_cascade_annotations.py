"""Composite current evidence is measured, summed before statistics, and unnamed."""
import numpy as np
import pytest
from silk import sources
from silk.circuit.modules import Module
from silk.circuit.modules_results import _annotate_level, CurrentAnnotationReport
from silk.simulation.backend import RawResult, NgspiceBackend
from silk.signals.rawfile import RawFile, RawPlot, RawVector


@pytest.mark.parametrize('missing', [False, True])
def test_annotation_uses_both_parallel_operands_and_no_single_vector(tmp_path, pdk, missing):
    mod = Module('sky130', str(tmp_path), 'top.cir', pdk_details=pdk)
    mod.add_source(sources.CurrentSource('stim', 'out', '0', dc=2, pwl=[0, -1, 1, -1]))
    vectors = [RawVector('time', 'time', data=np.array([0., 1., 2.])),
               RawVector('i(@istim__silk_pwl[current])', 'current', data=np.array([-1., -1., -1.]))]
    if not missing:
        vectors.append(RawVector('i(@istim__silk_dc[current])', 'current', data=np.array([2., 2., 2.])))
    raw = RawResult(source_path="fixture.raw", native=RawFile([RawPlot({'plotname': 'Transient Analysis'}, vectors, npoints=3, real=True, padded=True)]))
    report = CurrentAnnotationReport()
    _annotate_level(mod, NgspiceBackend(), raw, '', None, False, report)
    binding = mod.net_dict['out'].pins[0]
    assert binding.current_vector is None
    if missing:
        assert binding.avg_current is None and report.skips
    else:
        assert binding.avg_current == pytest.approx(-1)
        assert binding.rms_current == pytest.approx(1)  # NOT rms(-1) + rms(2) = 3
        assert binding.peak_current == pytest.approx(1)
        assert binding._current_source == 'spice'


def test_derived_current_provenance_survives_database_readback(tmp_path, pdk):
    from silk.db.circuit_db import Database
    child = Module('sky130', str(tmp_path), 'child.cir', pdk_details=pdk, module_name='child')
    child.define_module_ports({})
    child.add_source(sources.CurrentSource('stim', 'out', '0', dc=2, pwl=[0, -1, 1, -1]))
    db_path = str(tmp_path / 'currents.db')
    top = Module('sky130', str(tmp_path), 'top.cir', pdk_details=pdk, db_path=db_path)
    top.add_module(child, 'dut', nets={})
    definition = top.module_dict['child']['device']
    vectors = [RawVector('time', 'time', data=np.array([0., 1.])),
               RawVector('i(@i.xdut.istim__silk_pwl[current])', 'current', data=np.array([-1., -1.])),
               RawVector('i(@i.xdut.istim__silk_dc[current])', 'current', data=np.array([2., 2.]))]
    raw = RawResult(source_path='measured.raw', native=RawFile([
        RawPlot({'plotname': 'Transient Analysis'}, vectors, npoints=2, real=True, padded=True)]))
    report = CurrentAnnotationReport(results_file='measured.raw', analysis='transient')
    _annotate_level(definition, NgspiceBackend(), raw, 'xdut', None, False, report)
    assert definition.net_dict['out'].pins[0].avg_current == pytest.approx(-1)
    versions = top.store_net_currents(report, corner='tt')
    version = versions['child']
    top.close()
    with Database(db_path) as db:
        _, rows = db.get_net_current_set(version, 'tt')
    pins = [r for r in rows if r['pin_id']]
    assert pins and all(r['current_vector'] is None for r in pins)
    assert all(r['rms_a'] == pytest.approx(1) for r in pins)
    assert all(r['source'] == 'spice' for r in pins)
