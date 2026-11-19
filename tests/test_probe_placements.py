"""Placement selection mirrors corner selection for numerical reads and plots."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

from tests._env import needs_ngspice
from tests.circuit.test_selected_testbench import leaf, root


@needs_ngspice
def test_derived_plot_overlays_distinct_runs_with_alias_labels(tmp_path):
    child, tb = leaf(tmp_path)
    signal = tb.probe('out')
    derived = signal * 2
    top = root(tmp_path)
    parents = []
    try:
        for index, voltage in enumerate((1, 2, 1)):
            carrier = root(tmp_path)
            carrier.set_supply_voltage(voltage)
            carrier.define_module_name('carrier')
            carrier.define_module_ports({})
            carrier.add_module(child, 'cell', out='out', vss='0')
            top.add_module(carrier, 'u'+str(index), nets={})
            parents.append(carrier)
        top.run_all_tests()
        ax = derived.plot()
        assert len(ax.lines) == 2
        assert [line.get_ydata()[-1] for line in ax.lines] == pytest.approx([2, 4])
        labels = [line.get_label() for line in ax.lines]
        assert all('tt' in label for label in labels)
        assert 'u0.cell' in labels[0] and 'u2.cell' in labels[0]
        assert 'u1.cell' in labels[1]
        figure = tb.plot(derived, show=False)
        assert len(figure.axes[0].lines) == 2
        selected = top.get_test('step', instance='u1.cell')
        assert (selected.probe('out') * 2).y[-1] == pytest.approx(4)
    finally:
        plt.close('all')
        top.close()
        child.close()
        for carrier in parents:
            carrier.close()


def test_point_reads_selected_placements_at_shared_corner_keys(tmp_path, monkeypatch):
    from pathlib import Path
    import numpy as np
    from tests._mock_rawfile import write_complete_rawfile

    child, authored = leaf(tmp_path)
    top = root(tmp_path)
    carriers = []
    try:
        for index, name in enumerate(('one', 'two')):
            carrier = root(tmp_path)
            carrier.set_supply_voltage(index + 1)
            carrier.define_module_name('carrier')
            carrier.define_module_ports({})
            carrier.add_module(child, 'cell', out='out', vss='0')
            top.add_module(carrier, name, nets={})
            carriers.append(carrier)
        benches = [top.get_test('step', instance=name + '.cell') for name in ('one', 'two')]
        backend = authored.netlist.backend
        launches = []

        def launch(_backend, netlist_path, results_dir, **kwargs):
            launches.append(netlist_path)
            raw = Path(results_dir) / 'selected.raw'
            write_complete_rawfile(raw, Path(netlist_path).read_text(),
                                   values={'v(out)': float(len(launches))})
            return backend.load_raw(str(raw))

        monkeypatch.setattr(type(backend), 'run', launch)
        authored.set_corners(['tt', 'ss'])
        reports = top.run_all_tests()
        assert len(reports) == 2
        a, b = [bench.probe('out') for bench in benches]
        for key, expected in [('tt', [1., 3.]), ('ss', [2., 4.])]:
            for probe, value in zip((a, b), expected):
                selected = probe.point(key)
                assert selected.bound
                np.testing.assert_array_equal(selected.y, np.full(selected.y.shape, value))
            np.testing.assert_allclose((a + a - b).point(key).y, 2 * expected[0] - expected[1])
            readers = {}
            first = benches[0]._corner_netlist(key, readers)
            assert benches[0]._corner_netlist(key, readers) is first
            assert benches[1]._corner_netlist(key, readers) is not first
        from silk.exceptions import UnboundSignalError
        with pytest.raises(UnboundSignalError, match='several simulation results'):
            authored.probe('out').point('tt')
    finally:
        top.close()
        child.close()
        for carrier in carriers:
            carrier.close()
