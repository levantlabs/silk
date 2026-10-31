"""
Tests for PyNetlistAnalysis.NgspiceSignal — signal arithmetic and plotting.

Covers __add__, __sub__, __mul__, x-value preservation, error checks
(mismatched xlabel, x-values), and plot().
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI

from silk.exceptions import CircuitError
from silk.signals import Signal, NgspiceSignal


# ---------------------------------------------------------------------------
# Arithmetic: __add__
# ---------------------------------------------------------------------------

class TestSignalAdd:

    def test_add_y_values(self, sample_signal_pair):
        """Sum of two signals has correct y-values."""
        sig1, sig2 = sample_signal_pair
        result = sig1 + sig2
        np.testing.assert_array_almost_equal(result.y, sig1.y + sig2.y)

    def test_add_preserves_x_values(self, sample_signal_pair):
        """Addition preserves the x-axis values."""
        sig1, sig2 = sample_signal_pair
        result = sig1 + sig2
        np.testing.assert_array_equal(result.x, sig1.x)

    def test_add_returns_pyspice_signal(self, sample_signal_pair):
        """Result of addition is a NgspiceSignal instance."""
        sig1, sig2 = sample_signal_pair
        result = sig1 + sig2
        assert isinstance(result, NgspiceSignal)
        assert isinstance(result, Signal)

    def test_add_name_contains_operands(self, sample_signal_pair):
        """Result name contains both operand names joined by '+'."""
        sig1, sig2 = sample_signal_pair
        result = sig1 + sig2
        assert sig1.name in result.name
        assert sig2.name in result.name
        assert '+' in result.name


# ---------------------------------------------------------------------------
# Arithmetic: __sub__
# ---------------------------------------------------------------------------

class TestSignalSub:

    def test_sub_y_values(self, sample_signal_pair):
        """Difference of two signals has correct y-values."""
        sig1, sig2 = sample_signal_pair
        result = sig1 - sig2
        np.testing.assert_array_almost_equal(result.y, sig1.y - sig2.y)

    def test_sub_preserves_x_values(self, sample_signal_pair):
        """Subtraction preserves the x-axis values."""
        sig1, sig2 = sample_signal_pair
        result = sig1 - sig2
        np.testing.assert_array_equal(result.x, sig1.x)

    def test_sub_returns_pyspice_signal(self, sample_signal_pair):
        """Result of subtraction is a NgspiceSignal instance."""
        sig1, sig2 = sample_signal_pair
        result = sig1 - sig2
        assert isinstance(result, NgspiceSignal)
        assert isinstance(result, Signal)


# ---------------------------------------------------------------------------
# Arithmetic: __mul__
# ---------------------------------------------------------------------------

class TestSignalMul:

    def test_mul_y_values(self, sample_signal_pair):
        """Product of two signals has correct y-values."""
        sig1, sig2 = sample_signal_pair
        result = sig1 * sig2
        np.testing.assert_array_almost_equal(result.y, sig1.y * sig2.y)

    def test_mul_preserves_x_values(self, sample_signal_pair):
        """Multiplication preserves the x-axis values."""
        sig1, sig2 = sample_signal_pair
        result = sig1 * sig2
        np.testing.assert_array_equal(result.x, sig1.x)

    def test_mul_returns_pyspice_signal(self, sample_signal_pair):
        """Result of multiplication is a NgspiceSignal instance."""
        sig1, sig2 = sample_signal_pair
        result = sig1 * sig2
        assert isinstance(result, NgspiceSignal)
        assert isinstance(result, Signal)


# ---------------------------------------------------------------------------
# Error checks: mismatched xlabel
# ---------------------------------------------------------------------------

class TestMismatchedXLabel:

    def test_add_mismatched_xlabel_raises(self, sample_signal):
        """Adding signals with different xlabels raises ValueError."""
        other = NgspiceSignal(
            'v(ref)', 'voltage', 'frequency',  # different xlabel
            'voltage', sample_signal.x, sample_signal.y,
        )
        with pytest.raises(ValueError):
            _ = sample_signal + other

    def test_sub_mismatched_xlabel_raises(self, sample_signal):
        """Subtracting signals with different xlabels raises ValueError."""
        other = NgspiceSignal(
            'v(ref)', 'voltage', 'frequency',
            'voltage', sample_signal.x, sample_signal.y,
        )
        with pytest.raises(ValueError):
            _ = sample_signal - other


# ---------------------------------------------------------------------------
# Error checks: mismatched type
# ---------------------------------------------------------------------------

class TestMismatchedType:

    def test_add_mismatched_type_raises(self, sample_signal):
        """Adding signals with different types raises ValueError."""
        other = NgspiceSignal(
            'i(drain)', 'current', 'time',  # different type
            'current', sample_signal.x, np.zeros(100),
        )
        with pytest.raises(ValueError, match='different types'):
            _ = sample_signal + other


# ---------------------------------------------------------------------------
# Error checks: mismatched x-values
# ---------------------------------------------------------------------------

class TestMismatchedXValues:

    def test_add_mismatched_xvalues_raises(self, sample_signal):
        """Adding signals with different x-values raises ValueError."""
        different_x = np.linspace(0, 2e-6, 100)  # different range
        other = NgspiceSignal(
            'v(ref)', 'voltage', 'time', 'voltage',
            different_x, np.zeros(100),
        )
        with pytest.raises(ValueError):
            _ = sample_signal + other


# ---------------------------------------------------------------------------
# plot()
# ---------------------------------------------------------------------------

class TestPlot:

    def test_plot_returns_axes(self, sample_signal):
        """plot() returns a matplotlib Axes object."""
        import matplotlib.pyplot as plt
        ax = sample_signal.plot()
        assert ax is not None
        assert hasattr(ax, 'plot')  # quacks like an Axes
        plt.close('all')

    def test_plot_with_provided_axes(self, sample_signal):
        """plot() uses the provided axes when given."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        returned_ax = sample_signal.plot(ax=ax)
        assert returned_ax is ax
        plt.close('all')

    def test_plot_sets_labels(self, sample_signal):
        """plot() sets xlabel and ylabel on the axes."""
        import matplotlib.pyplot as plt
        ax = sample_signal.plot()
        assert ax.get_xlabel() == sample_signal.xlabel
        assert ax.get_ylabel() == sample_signal.ylabel
        plt.close('all')

    def test_plot_real_signal_uses_linear_axis(self, sample_signal):
        """A real, time-domain signal stays a linear x vs y line plot."""
        import matplotlib.pyplot as plt
        ax = sample_signal.plot()
        assert ax.get_xscale() == 'linear'
        plt.close('all')

    @staticmethod
    def _ac_signal():
        """A complex-valued AC-sweep signal (frequency on x)."""
        freq = np.logspace(0, 8, 50)
        # First-order rolloff: DC gain 100, pole at 1 MHz.
        h = 100.0 / (1.0 + 1j * freq / 1e6)
        return NgspiceSignal('v(out)', 'voltage', 'frequency', 'voltage',
                             freq, h)

    def test_plot_complex_defaults_to_db_on_log_axis(self):
        """A complex AC signal plots magnitude in dB on a log-frequency axis."""
        import matplotlib.pyplot as plt
        ax = self._ac_signal().plot()
        assert ax.get_yscale() == 'linear'      # y is dB (linear axis), x is log
        assert ax.get_xscale() == 'log'
        assert ax.get_ylabel() == 'Magnitude (dB)'
        # DC magnitude is 100 → 40 dB; the first (lowest-freq) point ≈ 40 dB.
        line = ax.get_lines()[0]
        assert line.get_ydata()[0] == pytest.approx(40.0, abs=0.5)
        plt.close('all')

    def test_plot_phase_mode(self):
        """mode='phase' plots degrees and labels the axis accordingly."""
        import matplotlib.pyplot as plt
        ax = self._ac_signal().plot(mode='phase')
        assert ax.get_ylabel() == 'Phase (deg)'
        plt.close('all')

    def test_plot_mode_override_real_on_complex(self):
        """mode='real' forces the real part even for a complex signal."""
        import matplotlib.pyplot as plt
        ax = self._ac_signal().plot(mode='real', logx=False)
        assert ax.get_xscale() == 'linear'
        plt.close('all')

    def test_plot_unknown_mode_raises(self, sample_signal):
        """An unknown mode is a clear error, not a silent fallback."""
        with pytest.raises(ValueError):
            sample_signal.plot(mode='bogus')

    def test_plot_bus_handle_raises(self, sample_signal):
        """Plotting a width>1 bus handle (no scalar y) raises a clear error."""
        from silk.signals import NgspiceSignal
        bus = NgspiceSignal._make_bus_handle('d', (sample_signal, sample_signal))
        with pytest.raises(TypeError):
            bus.plot()


# ---------------------------------------------------------------------------
# Signal properties
# ---------------------------------------------------------------------------

class TestSignalProperties:

    def test_signal_name(self, sample_signal):
        """Signal stores its name correctly."""
        assert sample_signal.name == 'v(out)'

    def test_signal_type(self, sample_signal):
        """Signal stores its type correctly."""
        assert sample_signal.type == 'voltage'

    def test_signal_xlabel(self, sample_signal):
        """Signal stores its xlabel correctly."""
        assert sample_signal.xlabel == 'time'

    def test_signal_data_length(self, sample_signal):
        """Signal x and y arrays have the same length."""
        assert len(sample_signal.x) == len(sample_signal.y)
        assert len(sample_signal.x) == 100
