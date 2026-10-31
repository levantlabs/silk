"""An AC current-source device value must never masquerade as a measurement."""
import numpy as np
import pytest
from silk import sources
from silk.circuit.module_test import _load_metric_path
from silk.exceptions import NotSupportedError
from silk.signals.rawfile import RawFile, RawPlot, RawVector, write_rawfile
from silk.simulation import AC
from tests.test_raw_current_paths import _bench


@pytest.mark.parametrize('name', ['stim', 'Istim__silk_pwl', 'Istim__silk_dc', '@istim__silk_dc[current]'])
@pytest.mark.parametrize('path', ['load', 'probe', 'metric'])
def test_ac_current_source_read_refuses_with_ammeter_guidance(tmp_path, name, path):
    src = sources.CurrentSource('stim', 'a', '0', dc=.2, pwl=[0, .5, 1, 1])
    tb = _bench(tmp_path, [], [src])
    raw = tmp_path / 'ac.raw'
    vectors = [RawVector('frequency', 'frequency', data=np.array([1e3, 2e3])),
               RawVector('i(@istim__silk_dc[current])', 'current', data=np.array([.2+0j, .2+0j])),
               RawVector('i(@istim__silk_pwl[current])', 'current', data=np.array([.5+0j, .5+0j]))]
    write_rawfile(raw, RawFile([RawPlot({'plotname': 'AC Analysis'}, vectors, npoints=2, real=False, padded=True)]))
    tb.netlist.load_file(str(raw)); tb.netlist.read_raw_file()
    with pytest.raises(NotSupportedError, match='ac=.*0 V VoltageSource'):
        if path == 'load':
            tb.netlist.load_signal(name, 'I', analysis='ac')
        elif path == 'metric':
            _load_metric_path(tb.netlist, f'i({name})', analysis='ac')
        else:
            tb.probe(f'i({name})', analysis='ac').y


def test_ac_current_annotation_names_ammeter_remedy(tmp_path):
    tb = _bench(tmp_path, [], [sources.CurrentSource('stim', 'a', '0', dc=.2)])
    tb.netlist.sim_analysis_dict = {'ac': AC('lin', 3, 1e3, 2e3)}
    with pytest.raises(NotSupportedError, match='ac=.*0 V VoltageSource'):
        tb.netlist.annotate_currents()


@pytest.mark.parametrize('operation', ['load', 'annotation'])
def test_loaded_ac_plot_cannot_bypass_current_source_refusal(tmp_path, operation):
    tb = _bench(tmp_path, [], [sources.CurrentSource('stim', 'a', '0', ac=1)])
    raw = tmp_path / 'ac.raw'
    write_rawfile(raw, RawFile([RawPlot({'plotname': 'AC Analysis'}, [
        RawVector('frequency', 'frequency', data=np.array([1e3, 2e3])),
        RawVector('i(@istim[current])', 'current', data=np.array([0j, 0j]))],
        npoints=2, real=False, padded=True)]))
    tb.netlist.load_file(str(raw)); tb.netlist.read_raw_file()
    with pytest.raises(NotSupportedError, match='ac=.*0 V VoltageSource'):
        if operation == 'load':
            tb.netlist.load_signal('i(@istim[current])', analysis='ac')
        else:
            tb.netlist.annotate_currents()
