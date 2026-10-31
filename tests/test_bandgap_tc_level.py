"""The bandgap TC testbench must check the *level*, not only the flatness.

`tc_ppm_per_C` is `(max-min)/mean/dT` — a flatness measure, blind to the level
V_ref is flat at. A bandgap core that converges to a wrong-but-flat equilibrium
therefore scores a near-perfect TC and would pass on TC alone. The `tc`
testbench now also specs the sweep's *mean* into the acceptance band, closing
that hole in the same simulation (no reliance on the separate `settle` run).

The assertions here exercise the metric functions against synthetic sweeps (no
SPICE run), but the example module calls ``load_pdk`` at import, so the module
imports it lazily via a fixture and is marked ``needs_pdk`` — importing it at
file scope would call ``load_pdk`` during collection and break the suite on a
PDK-less machine (guarded by ``test_suite_runs_without_a_pdk.py``).
"""
import numpy as np
import pytest

from silk.metrics import average

from tests._env import needs_pdk

pytestmark = needs_pdk


@pytest.fixture(scope="module")
def bg():
    """The bandgap example, imported lazily so collection never calls load_pdk."""
    import silk.examples.bandgap_example as _bg
    return _bg


class _StubCkt:
    """Minimal stand-in for the simulated netlist: only `load_signal('vref')`."""
    def __init__(self, x, y):
        from silk.signals.base import Signal
        self._sig = Signal('v(vref)', 'voltage', 'temperature', 'voltage',
                           np.asarray(x, dtype=float), np.asarray(y, dtype=float))

    def load_signal(self, name):
        assert name == "vref"
        return self._sig


# A representative .dc temp -40..125 axis (the metrics only use its span/order).
_T = np.array([-40., -20., 5., 30., 50., 75., 100., 125.])


def _in_band(bg, mean):
    return bg.VREF_LO < mean < bg.VREF_HI


class TestWrongEquilibriumIsCaught:
    """The load-bearing case (per the backlog doc): a flat sweep at the WRONG
    level must score TC ~ 0 (blind) AND fail the mean band (caught). Asserting
    only the second half would pass against the pre-fix code and prove nothing,
    so both halves are asserted together."""

    def test_flat_wrong_level_scores_perfect_tc_but_fails_mean_band(self, bg):
        v = np.full_like(_T, 0.649)  # spurious equilibrium: flat, in-guard, out-of-band
        ckt = _StubCkt(_T, v)

        # (a) TC is blind: a flat curve is a perfect ~0 ppm/C, and it clears the
        #     dead-core guard (min 0.649 V > 0.1 V), so no RuntimeError.
        tc = bg._tc_ppm_per_C(ckt)
        assert tc == pytest.approx(0.0, abs=1e-9)
        assert tc < bg.TC_TARGET_PPM  # would PASS the TC spec on its own

        # (b) The mean level is the check that catches it.
        mean = average(ckt.load_signal('vref'))
        assert mean == pytest.approx(0.649)
        assert not _in_band(bg, mean)
        assert mean < bg.VREF_LO  # below the band → the lower mean-spec fails

    def test_dead_core_still_raises_before_reaching_the_mean_check(self, bg):
        # The pre-existing dead-core guard is unchanged: a ~0 V core is rejected
        # by _tc_ppm_per_C itself, never scored.
        v = np.zeros_like(_T)
        with pytest.raises(RuntimeError, match="did not start"):
            bg._tc_ppm_per_C(_StubCkt(_T, v))


class TestShippingVariantsUnchanged:
    """The mean check must leave both ngspice variants' verdicts exactly as they
    were: A passes, B still fails on TC alone (its mean stays in band, so the
    new spec adds no failure). Values are the measured ngspice sweeps."""

    def test_variant_a_passes_tc_and_mean_in_band(self, bg):
        vA = np.array([1.200, 1.204, 1.209, 1.211, 1.211, 1.209, 1.205, 1.201])
        ckt = _StubCkt(_T, vA)
        assert bg._tc_ppm_per_C(ckt) < bg.TC_TARGET_PPM
        assert _in_band(bg, average(ckt.load_signal('vref')))

    def test_variant_b_fails_tc_but_mean_stays_in_band(self, bg):
        # B's cold runaway to ~1.55 V wrecks TC, but the excursion averages out:
        # the mean is comfortably in band, so the mean spec does NOT add a second
        # failure — B keeps failing on TC alone, its printed verdict unchanged.
        vB = np.array([1.547, 1.261, 1.239, 1.225, 1.215, 1.204, 1.194, 1.185])
        ckt = _StubCkt(_T, vB)
        assert bg._tc_ppm_per_C(ckt) > bg.TC_TARGET_PPM  # fails TC
        assert _in_band(bg, average(ckt.load_signal('vref')))          # mean spec passes


class TestTestbenchWiring:
    """attach_tc_test must actually register the mean-band specs, not just make
    the metric callable."""

    def test_tc_testbench_specs_the_mean_into_the_band(self, bg):
        m = bg.build_bandgap("pmos")
        tb = bg.attach_tc_test(m, input_type="pmos")
        mean_specs = [s for s in tb._specs if s["metric"] == "vref_mean"]
        ops = {(s["op"], s["target"]) for s in mean_specs}
        assert (">", bg.VREF_LO) in ops
        assert ("<", bg.VREF_HI) in ops
        # The original TC spec is still there.
        assert any(s["metric"] == "tc_ppm_per_C" for s in tb._specs)
