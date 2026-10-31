"""Regression check for TASK-561: a leaked pyplot figure must not reach later tests.

``Signal.plot()`` with no ``ax`` draws on ``plt.gca()``. The autouse
``_close_pyplot_figures`` fixture in ``tests/conftest.py`` closes every figure
after each test, so the figure left open by the first test below is gone before
the second test plots. Without that fixture the second test reads the leaked
axes and its first line is the leaker's, not its own trace.

pytest runs a file's tests in definition order, so the pair is deterministic.
"""
import numpy as np

from silk.signals import Signal


def test_leaves_a_figure_open_on_purpose():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.add_subplot(111).plot([0, 1], [9., 9.])
    # Deliberately not closed: the next test must not see this figure.


def test_plot_without_ax_draws_on_a_fresh_axis_after_a_leak():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sig = Signal('y', 'voltage', 'time', 'V', [0, 1, 2], [1., 2., 3.])
    ax = sig.plot()
    try:
        assert len(ax.lines) == 1
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [1., 2., 3.])
    finally:
        plt.close(ax.figure)
