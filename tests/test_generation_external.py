"""Real external and managed execution share prepared relative-include decks."""
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from silk.circuit.modules_sim import _track_result_replacement
from silk.exceptions import SimulationConfigurationError
from silk.simulation.runner import ngspice_executable
from tests._env import HAVE_XYCE, needs_ngspice, xyce_executable
from tests.test_run_split import _build_lowpass_tb


pytestmark = pytest.mark.slow


@pytest.mark.parametrize('simulator', [
    pytest.param('ngspice', marks=needs_ngspice),
    pytest.param('xyce', marks=[pytest.mark.requires_xyce,
                              pytest.mark.skipif(not HAVE_XYCE, reason='needs configured Xyce')]),
])
def test_external_prepared_relative_include_then_managed_run(tmp_path, monkeypatch, simulator):
    dut, tb = _build_lowpass_tb(str(tmp_path), name='external')
    try:
        dut.simulator = simulator
        dut.set_output_options(mode='overwrite')
        # Opaque custom lines conservatively retain a library include. This
        # local empty library keeps the real circuit independent of a PDK.
        library = tmp_path / 'fixture.lib'
        library.write_text('.lib tt\n.param fixture_unused=1\n.endl tt\n')
        pdk = SimpleNamespace(library_location=str(library), corners={}, extra_includes=[],
                              length_unit_in_metres=None)
        dut.pdk_details = tb.pdk_details = tb.netlist.pdk_details = pdk
        tb.netlist.add_custom_line('.include "relative-load.inc"')
        prepared = Path(tb.prepare())
        files = tb.netlist._run_files
        include = prepared.parent / 'relative-load.inc'
        include.write_text('Rexternal_load vout 0 1000\n')
        monkeypatch.chdir(prepared.parent)
        deck = prepared.read_bytes()
        assert b'.include "relative-load.inc"' in deck
        assert not list(tmp_path.rglob('.silk-generation-*'))

        if simulator == 'ngspice':
            command = [ngspice_executable(), '-b', '-r', files.rawfile_path,
                       '-o', files.logfile_path, str(prepared)]
        else:
            command = [xyce_executable(), str(prepared)]
        completed = subprocess.run(command, cwd=files.directory,
                                   capture_output=True, text=True, timeout=60)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        external_bytes = Path(files.rawfile_path).read_bytes()
        report = tb.evaluate(rawfile=files.rawfile_path)
        assert report.operating_conditions.origin == 'external'
        assert getattr(tb.netlist._raw, '_generation', None) is None
        assert report.metrics['vout_final'] == pytest.approx(0.5, rel=1e-5)
        external_y = tb.netlist.load_signal('vout').y.copy()
        old_raw = tb.netlist._raw

        # Preparing has no replacement side effect, and a stale prepared
        # request must fail before any external evidence is invalidated.
        assert Path(tb.prepare()) == prepared
        assert tb.netlist._raw is old_raw
        assert Path(files.rawfile_path).read_bytes() == external_bytes
        tb.netlist.sig_sources_dict['vin'].dc = 2
        with _track_result_replacement(tb.netlist) as started:
            with pytest.raises(SimulationConfigurationError, match='Circuit configuration changed'):
                tb.netlist.run_simulator()
        assert started == [False]
        assert tb.netlist._raw is old_raw
        assert Path(files.rawfile_path).read_bytes() == external_bytes
        assert prepared.read_bytes() == deck
        assert not list(tmp_path.rglob('.silk-generation-*'))

        tb.netlist.sig_sources_dict['vin'].dc = 1
        managed = tb.run()
        assert managed.metrics['vout_final'] == pytest.approx(report.metrics['vout_final'])
        np.testing.assert_allclose(tb.netlist.load_signal('vout').y, external_y)
        generation = tb.netlist._raw._generation
        assert generation.accepted
        assert Path(generation.staged.netlist_path).parent == prepared.parent
        assert Path(generation.staged.netlist_path) != prepared
        assert prepared.read_bytes() == deck
        assert report.operating_conditions.origin == 'external'
    finally:
        dut.close()
