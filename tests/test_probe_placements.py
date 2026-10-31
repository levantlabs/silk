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
