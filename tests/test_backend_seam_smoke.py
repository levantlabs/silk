"""Second-backend seam smoke (Plan C3 / T9).

Proves the simulator-agnostic claim end to end with NO real simulator and NO
edits to ``modules.py``: a brand-new backend, registered under a name, drives a
``Module`` through the four core seams — render analysis, emit netlist, run, load
signal — purely by being selected via ``self.simulator``. (The fifth seam,
``device_current_vectors`` for current annotation, is optional: ``SpiceBackend``'s
base raises ``NotSupportedError``, so a fake backend need not implement it.)

The Xyce e2e (``test_xyce_netlister.py``) proves the real second backend; this
proves the *contract* is closed (a fake backend with synthetic results plugs in
with zero core changes), and runs everywhere with no external dependency.

It also proves the two TASK-72 hardening properties of that contract, which are
exactly what a fake third backend is the right witness for:

* the data-presence gate is **structural** — a backend that produces no data
  raises ``SimulationAborted`` with zero gate code of its own, because
  ``SimulatorBackend.run`` is a template method around ``_launch``;
* a backend cannot take ``run`` back and skip the gate — defining ``run`` on a
  backend subclass is refused at ``class`` statement time.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from silk.exceptions import SimulationAborted
from silk.simulation.backend import SimulatorBackend, SpiceBackend, RawResult
from silk.simulation.analysis import AC
from silk.circuit.modules import Module
from silk import sources


# ---------------------------------------------------------------------------
# A complete fake backend: inherits render_analysis + load_signal + load_raw
# from SpiceBackend; supplies a real netlister (so emission works) and a run()
# that returns synthetic results instead of launching a simulator.
# ---------------------------------------------------------------------------

_FAKE_FREQ = np.logspace(0, 6, 13)
_FAKE_VOUT = np.linspace(2.0, 0.5, 13)


@SimulatorBackend.register('faketb')
class _FakeTBBackend(SpiceBackend):
    ran_with = None

    def make_netlister(self, pdk_details):
        # Reuse silk's real SPICE body/wrapper emission — emission is not the
        # thing under test here (the run/load seam is), and it lets
        # generate_netlist write a real deck.
        from silk.circuit.spice_netlister import PySpiceNetlister
        return PySpiceNetlister(pdk_details)

    def _launch(self, netlist_path, results_dir, *, debug=False, timeout=None):
        # Note what a backend author writes here: no _require_data call, no
        # abort bookkeeping. The gate is applied by SimulatorBackend.run(),
        # which called this. Returning the bare RawResult is the short form of
        # the (RawResult, log_path, reason) contract, for a backend with
        # neither a log file nor a pre-scraped reason.
        type(self).ran_with = dict(netlist_path=netlist_path, results_dir=results_dir)
        native = SimpleNamespace(
            nvars=2,
            vectors=[
                SimpleNamespace(name='frequency', data=_FAKE_FREQ, type='frequency'),
                SimpleNamespace(name='v(out)', data=_FAKE_VOUT, type='voltage'),
            ],
        )
        return RawResult(native=native, source_path=results_dir + '/rawfile.raw')


@SimulatorBackend.register('faketb_nodata')
class _NoDataBackend(_FakeTBBackend):
    """A backend whose simulator "ran" and produced nothing.

    The point of this class is what it does **not** contain: it has no call to
    ``_require_data``, no exit-code check, no opt-in of any kind. It is the
    backend a careless third-backend author writes.
    """

    def _launch(self, netlist_path, results_dir, *, debug=False, timeout=None):
        native = SimpleNamespace(nvars=0, vectors=[], npoints=0)
        return RawResult(native=native, source_path=results_dir + '/rawfile.raw')


def _build_top_module(tmp_path):
    pdk_yaml = os.path.join(os.path.dirname(__file__), '..', 'silk', 'pdk',
                            'data', 'skywater_130_device_details.yaml')
    from silk.pdk.basepdk import BasePDK
    # A real (never-parsed) file: the fake backend stands in for the
    # simulator, not for an uninstalled PDK, and a deck with PDK devices in it
    # checks the model library exists before it runs (TASK-71).
    lib = tmp_path / 'sky130.lib.spice'
    lib.write_text('* stub sky130 lib: never parsed\n')
    pdk = BasePDK(pdk_yaml, library_location=str(lib))
    mod = Module('PDK_skywater130', str(tmp_path), 'seam.cir', pdk_details=pdk)
    for n in ('out', 'vdd', '0'):
        mod.add_net(n)
    mod.add(pdk.nmos1p8('M1', d='out', g='vdd', s='0', b='0', w=1.0, l=0.15, m=1))
    mod.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
    mod.add_source(sources.VoltageSource('vin', positive='out', negative='0', dc=0.9, ac=1.0))
    mod.add_sim_type(AC('dec', 10, 1, 1e9))
    return mod


def test_fake_backend_drives_all_four_seams(tmp_path):
    """Selecting an arbitrary registered backend drives generate→run→load with
    no modules.py changes — the simulator-agnostic contract is closed."""
    mod = _build_top_module(tmp_path)
    mod.simulator = 'faketb'

    # Seam 1 (render) + Seam 2 (emit): the netlister came from the fake backend
    # and writes a deck containing the rendered analysis line.
    deck_path = mod.generate_netlist(corner='tt')
    deck = open(deck_path).read()
    assert '.ac dec 10 1 1G' in deck            # render_analysis (shared SPICE)

    # Seam 3 (run): backend.run() returns the fake RawResult.
    mod.run_simulator()
    assert _FakeTBBackend.ran_with['netlist_path'] == deck_path
    assert mod.sim_results_loaded is True

    # Seam 4 (load): backend.load_signal reads the synthetic native result.
    sig = mod.load_signal('out', 'V')
    np.testing.assert_array_equal(sig.y, _FAKE_VOUT)
    assert sig.xlabel == 'frequency'


def test_fake_backend_netlister_is_lazy_and_selected(tmp_path):
    """The netlister is sourced from the selected backend (C2c lazy property)."""
    mod = _build_top_module(tmp_path)
    from silk.circuit.spice_netlister import PySpiceNetlister
    assert isinstance(mod.netlister, PySpiceNetlister)   # ngspice default
    mod.simulator = 'faketb'
    # Still a PySpiceNetlister here (the fake reuses it), but rebuilt via the
    # faketb backend — proven by the run/load path above using 'faketb'.
    assert isinstance(mod.netlister, PySpiceNetlister)


def test_unknown_backend_still_fails_fast(tmp_path):
    """Sanity: the validating selector rejects an unregistered backend."""
    mod = _build_top_module(tmp_path)
    with pytest.raises(ValueError):
        mod.simulator = 'no_such_backend'


# ---------------------------------------------------------------------------
# TASK-72: the gate is structural, not a per-backend courtesy.
# ---------------------------------------------------------------------------

def test_no_data_backend_aborts_without_opting_in(tmp_path):
    """A backend that produces no data rows raises SimulationAborted even
    though it contains ZERO gate code — the template method applied it."""
    backend = SimulatorBackend.get('faketb_nodata')
    with pytest.raises(SimulationAborted) as ei:
        backend.run(str(tmp_path / 'deck.cir'), str(tmp_path))
    assert ei.value.data_rows == 0
    # Attributed to the name the backend is registered under, not a string the
    # backend had to remember to pass.
    assert ei.value.simulator == 'faketb_nodata'


def test_no_data_backend_aborts_through_module(tmp_path):
    """Same gate, reached the way a user reaches it: Module.run_simulator()."""
    mod = _build_top_module(tmp_path)
    mod.simulator = 'faketb_nodata'
    mod.generate_netlist(corner='tt')
    with pytest.raises(SimulationAborted):
        mod.run_simulator()
    assert mod.sim_results_loaded is False


def test_backend_defining_run_is_rejected():
    """A backend cannot take run() back and skip the gate: the class statement
    itself fails, before anyone can select that backend."""
    with pytest.raises(TypeError) as ei:
        class _UngatedBackend(SpiceBackend):
            def run(self, netlist_path, results_dir, *, debug=False, timeout=None):
                return RawResult(native=None, source_path='')
    assert '_launch' in str(ei.value)


def test_launch_returning_junk_says_so(tmp_path):
    """A _launch that returns the wrong shape fails with a message naming the
    contract, not obscurely inside the gate."""
    @SimulatorBackend.register('faketb_junk')
    class _JunkBackend(_FakeTBBackend):
        def _launch(self, netlist_path, results_dir, *, debug=False, timeout=None):
            return 'not a RawResult'

    with pytest.raises(TypeError) as ei:
        SimulatorBackend.get('faketb_junk').run(str(tmp_path / 'd.cir'), str(tmp_path))
    assert '_launch' in str(ei.value)
