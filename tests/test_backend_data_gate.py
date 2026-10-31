"""Layer 2: the agnostic data-presence gate (_require_data)."""
from types import SimpleNamespace

import pytest
from silk.signals.rawfile import read_rawfile
from silk.simulation.backend import (
    DATA_PRESENCE_HOOK, RawResult, SimulatorBackend, SpiceBackend,
    _has_data_rows, _require_data,
)
from silk.exceptions import NotSupportedError, SimulationAborted


def _write_raw(tmp_path, npoints, values_block=""):
    raw = (
        "Title: t\nDate: d\nPlotname: Transient Analysis\nFlags: real\n"
        "No. Variables: 2\nNo. Points: %d\n"
        "Variables:\n\t0\ttime\ttime\n\t1\tv(out)\tvoltage\n"
        "Values:\n%s" % (npoints, values_block)
    )
    p = tmp_path / 'rawfile.raw'
    p.write_text(raw)
    return str(p)


def _write_constants_raw(tmp_path):
    # ngspice's write-with-no-analysis fallback: when an analysis aborts, the run
    # can leave ONLY the synthetic 'constants' plot (pi/boltz/echarge/…) with
    # npoints=1. A REAL ngspice-42 abort emits exactly this plotname; the reader
    # exposes the last plot as native.current_plot, so the name lives there.
    raw = (
        "Title: Constant values\nDate: d\nPlotname: constants\nFlags: real\n"
        "No. Variables: 3\nNo. Points: 1\n"
        "Variables:\n\t0\tpi\tnotype\n\t1\tboltz\tnotype\n\t2\techarge\tnotype\n"
        "Values:\n0\t3.14159265358979\n\t1.38064852e-23\n\t1.60217662e-19\n"
    )
    p = tmp_path / 'rawfile.raw'
    p.write_text(raw)
    return str(p)


def test_zero_rows_raises_aborted(tmp_path):
    path = _write_raw(tmp_path, 0)
    raw = RawResult(native=read_rawfile(path), source_path=path)
    with pytest.raises(SimulationAborted) as ei:
        _require_data(raw, simulator='ngspice')
    assert ei.value.data_rows == 0


def test_constants_only_plot_raises_aborted(tmp_path):
    # A constants-only rawfile reports npoints=1 (would look like data) but is an
    # aborted run: the analysis produced nothing. The gate must reject it.
    path = _write_constants_raw(tmp_path)
    native = read_rawfile(path)
    assert native.npoints == 1                            # would (wrongly) look like data
    assert native.current_plot.plotname == 'constants'
    raw = RawResult(native=native, source_path=path)
    with pytest.raises(SimulationAborted) as ei:
        _require_data(raw, simulator='ngspice')
    assert ei.value.data_rows == 0
    assert ei.value.simulator == 'ngspice'


def test_rows_present_does_not_raise(tmp_path):
    values = "\t0\t0.0\n\t\t1.0\n\t1\t1e-9\n\t\t1.8\n"
    path = _write_raw(tmp_path, 2, values)
    raw = RawResult(native=read_rawfile(path), source_path=path)
    _require_data(raw, simulator='ngspice')  # must not raise


def test_native_none_does_not_raise(tmp_path):
    # Xyce noise-only run: no rawfile, native is None. Legitimate.
    raw = RawResult(native=None, source_path=str(tmp_path / 'rawfile.raw'))
    _require_data(raw, simulator='xyce')  # must not raise


def test_reason_from_log_attached(tmp_path):
    log = tmp_path / 'output.log'
    log.write_text("Timestep too small; time = 1e-18\nrun simulation(s) aborted\n")
    path = _write_raw(tmp_path, 0)
    raw = RawResult(native=read_rawfile(path), source_path=path)
    with pytest.raises(SimulationAborted) as ei:
        _require_data(raw, simulator='ngspice', log_path=str(log))
    assert 'Timestep too small' in (ei.value.reason or '')


# ---------------------------------------------------------------------------
# TASK-72 item 2: data presence for a NON-NUTMEG native.
#
# The two SPICE backends both read their rawfile through silk's nutmeg reader,
# so `native` always has .npoints/.vectors. A future non-SPICE backend's native
# has neither, and silk must not guess: guessing "no data" would flag every one
# of that backend's runs as an abort (a false red). These pin the two seams a
# non-nutmeg backend answers through, and the loud failure when it answers
# through neither.
# ---------------------------------------------------------------------------

class _NonNutmegNative:
    """A native result that is not a rawfile and answers for itself."""

    def __init__(self, rows):
        self.rows = rows

    def silk_has_data(self):
        return self.rows > 0


class _OpaqueNative:
    """A native that is neither nutmeg-shaped nor implements the hook."""


def test_non_nutmeg_native_with_data_passes(tmp_path):
    raw = RawResult(native=_NonNutmegNative(rows=128), source_path='mem://run')
    _require_data(raw, simulator='fakesim')          # must not raise


def test_non_nutmeg_native_without_data_aborts(tmp_path):
    raw = RawResult(native=_NonNutmegNative(rows=0), source_path='mem://run')
    with pytest.raises(SimulationAborted) as ei:
        _require_data(raw, simulator='fakesim')
    assert ei.value.data_rows == 0
    assert ei.value.simulator == 'fakesim'


def test_hook_name_is_the_documented_one():
    """The protocol is a named constant, not a string repeated in two files."""
    assert DATA_PRESENCE_HOOK == 'silk_has_data'
    assert hasattr(_NonNutmegNative, DATA_PRESENCE_HOOK)


def test_hook_is_looked_up_on_the_type_not_the_instance():
    """A stand-in that answers every getattr (a MagicMock in someone's test)
    has not implemented the protocol — the nutmeg fields it was given are still
    what decides, exactly as before the hook existed."""
    from unittest.mock import MagicMock
    mock = MagicMock()
    mock.npoints = 2
    mock.current_plot.plotname = 'Transient Analysis'
    assert _has_data_rows(mock) is True
    mock.npoints = 0
    assert _has_data_rows(mock) is False


def test_hook_must_return_a_bool():
    class _Sloppy:
        def silk_has_data(self):
            return 42
    with pytest.raises(TypeError) as ei:
        _has_data_rows(_Sloppy())
    assert DATA_PRESENCE_HOOK in str(ei.value)


def test_unanswerable_native_raises_actionably():
    """Neither nutmeg-shaped nor hooked: silk says so and names both seams,
    instead of silently reporting 'no data' for every run of that backend."""
    with pytest.raises(NotSupportedError) as ei:
        _has_data_rows(_OpaqueNative())
    msg = str(ei.value)
    assert DATA_PRESENCE_HOOK in msg and 'has_data()' in msg


def test_backend_can_override_has_data(tmp_path):
    """The other seam: a backend answers for a native it cannot change."""

    @SimulatorBackend.register('nonnutmeg')
    class _NonNutmegBackend(SpiceBackend):
        def make_netlister(self, pdk_details):        # pragma: no cover - unused
            raise NotImplementedError

        def has_data(self, native):
            return len(native.samples) > 0

        def _launch(self, netlist_path, results_dir, *, debug=False, timeout=None):
            native = SimpleNamespace(samples=self.samples)
            return RawResult(native=native, source_path='mem://run')

    backend = SimulatorBackend.get('nonnutmeg')
    backend.samples = [1.0, 2.0]
    backend.run('deck.cir', str(tmp_path))            # must not raise

    empty = SimulatorBackend.get('nonnutmeg')
    empty.samples = []
    with pytest.raises(SimulationAborted) as ei:
        empty.run('deck.cir', str(tmp_path))
    assert ei.value.simulator == 'nonnutmeg'


def test_nutmeg_backends_still_use_the_nutmeg_shape(tmp_path):
    """Regression: the hook lookup did not displace the nutmeg path the two
    SPICE backends depend on (constants carve-out included)."""
    ngspice = SimulatorBackend.get('ngspice')
    path = _write_constants_raw(tmp_path)
    assert ngspice.has_data(read_rawfile(path)) is False
    ok_dir = tmp_path / 'ok'
    ok_dir.mkdir()
    good = _write_raw(ok_dir, 2, "\t0\t0.0\n\t\t1.0\n\t1\t1e-9\n\t\t1.8\n")
    assert ngspice.has_data(read_rawfile(good)) is True
