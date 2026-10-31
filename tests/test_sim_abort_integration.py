"""End-to-end: a real ngspice run that aborts must raise SimulationAborted,
and a real valid run must still succeed (no false positive)."""
import os
import shutil
from pathlib import Path
import pytest
from silk.simulation.backend import SimulatorBackend
from silk.exceptions import SimulationAborted
from silk.simulation.runner import ngspice_executable

pytestmark = pytest.mark.slow

ngspice_missing = shutil.which(ngspice_executable()) is None


@pytest.mark.skipif(ngspice_missing, reason="ngspice not installed")
def test_aborted_run_raises(tmp_path):
    """A deck that produces no analysis output makes ngspice exit 0 and write
    no rawfile — the exact 'rc=0 but no usable result' class this feature
    targets. silk must raise SimulationAborted, not crash on the missing file
    or report a false success. (First line is the SPICE title/comment line.)

    The no-row variant is covered by test_backend_data_gate.py; partial-row
    output is covered below using an impossible equation after 1.5 ns."""
    deck = tmp_path / 'abort.cir'
    deck.write_text(
        "* deck with no analysis directive -> no rawfile, exit 0\n"
        "v1 1 0 1\n"
        "r1 1 0 1k\n"
        ".end\n"
    )
    backend = SimulatorBackend.get('ngspice')
    with pytest.raises(SimulationAborted):
        backend.run(str(deck), str(tmp_path))


@pytest.mark.skipif(ngspice_missing, reason="ngspice not installed")
def test_valid_run_succeeds(tmp_path):
    deck = tmp_path / 'ok.cir'
    deck.write_text(
        "* valid rc network (first line is the title/comment line)\n"
        ".tran 1n 10n\n"
        "v1 1 0 1\n"
        "r1 1 2 1k\n"
        "c1 2 0 1p\n"
        ".end\n"
    )
    backend = SimulatorBackend.get('ngspice')
    raw = backend.run(str(deck), str(tmp_path))       # must not raise
    assert raw.native is not None
    assert raw.native.npoints > 0


@pytest.mark.skipif(ngspice_missing, reason='ngspice not installed')
@pytest.mark.parametrize('reuse_module', [True, False])
def test_same_directory_success_then_no_output_never_resurrects_data(tmp_path, reuse_module):
    """A .control quit leaves -r untouched: cleanup must remove the old file."""
    from pathlib import Path
    import numpy as np
    from silk.exceptions import SimulationError
    from tests.test_run_files import _module

    directory = tmp_path / 'reused'
    first = _module(directory)
    first.set_output_options(mode='overwrite')
    first.sig_sources_dict['vin'].dc = 1.8
    first.simulate()
    saved_path = Path(first.results_file)
    np.testing.assert_allclose(first.load_signal('out').y, 1.8)
    assert saved_path.is_file()
    attempt = first if reuse_module else _module(directory)
    attempt.set_output_options(mode='overwrite')
    attempt.sig_sources_dict['vin'].dc = 0.9
    attempt.add_custom_line('.control\nquit\n.endc')
    for _ in range(2):
        with pytest.raises(SimulationAborted):
            attempt.simulate()
        assert not saved_path.exists()
        assert attempt._raw is attempt.simResultsAll is None
        assert attempt.dataFile is attempt.results_file is None
        assert attempt._last_sim_corner is None
        assert not attempt.sim_results_loaded
        with pytest.raises(SimulationError):
            attempt.load_signal('out')
        with pytest.raises(SimulationError):
            attempt.read_raw_file()
    deck = Path(os.path.join(attempt.dir_folder, attempt.netlistname)).read_text()
    assert '0.9' in deck
    assert deck.index('quit') < deck.index('run\n')
    # The same no-output request aborts in an unused directory too.
    fresh = _module(tmp_path / 'fresh')
    fresh.set_output_options(mode='overwrite')
    fresh.sig_sources_dict['vin'].dc = 0.9
    fresh.add_custom_line('.control\nquit\n.endc')
    with pytest.raises(SimulationAborted):
        fresh.simulate()
    fresh_deck = Path(os.path.join(fresh.dir_folder, fresh.netlistname)).read_text()
    assert fresh_deck.replace(str(tmp_path / 'fresh'), str(directory)) == deck
    # A later valid replacement publishes the changed supply, on the same path.
    attempt.custom_lines.clear()
    attempt.simulate()
    assert attempt.results_file == str(saved_path)
    np.testing.assert_allclose(attempt.load_signal('out').y, 0.9)


@pytest.mark.skipif(ngspice_missing, reason='ngspice not installed')
def test_direct_backend_rejects_stale_module_control_block_output(tmp_path):
    from tests.test_run_files import _module
    from pathlib import Path

    mod = _module(tmp_path)
    mod.set_output_options(mode='overwrite')
    mod.generate_netlist()
    backend = SimulatorBackend.get('ngspice')
    first = backend.run(os.path.join(mod.dir_folder, mod.netlistname), mod.dir_folder)
    assert first.native.npoints > 0
    mod.add_custom_line('.control\nquit\n.endc')
    mod.generate_netlist()
    with pytest.raises(SimulationAborted):
        backend.run(os.path.join(mod.dir_folder, mod.netlistname), mod.dir_folder)
    assert not Path(first.source_path).exists()


@pytest.mark.skipif(ngspice_missing, reason='ngspice not installed')
def test_mid_transient_abort_with_partial_rawfile(tmp_path):
    """An impossible B-source equation after 1.5 ns leaves partial rows, rc=0."""
    from silk.signals.rawfile import read_rawfile
    from silk import Module, load_pdk, sources
    from silk.elements import elements
    from silk.simulation import Transient
    mod = Module('sky130', str(tmp_path), 'partial.cir', load_pdk('sky130'))
    mod.set_output_options(mode='overwrite')
    mod.add_source(sources.BehavioralVoltage('b1', positive='n', negative='0',
                                           expression='time > 1.5e-9 ? v(n) + 1 : 0'))
    mod.add(elements.resistor('r1', p='n', n='0', value=1e6))
    mod.add_sim_type(Transient(10e-12, 3e-9))
    mod.generate_netlist()
    assert mod._run_files.completion is not None
    with pytest.raises(SimulationAborted) as error:
        SimulatorBackend.get('ngspice').run(os.path.join(mod.dir_folder, mod.netlistname), str(tmp_path), run_files=mod._run_files)
    assert 'tran' in str(error.value)
    assert 'requested stop 3e-09' in str(error.value)
    assert 'Timestep too small' in error.value.reason
    assert not (tmp_path / 'rawfile.raw').exists()
    assert error.value.log_path == str(tmp_path / 'output.log')
    assert Path(error.value.log_path).is_file()
    # Only the diagnostic log is published; partial data stays private.
    retained_rawfiles = list(tmp_path.glob('.silk-generation-*/rawfile.raw'))
    assert len(retained_rawfiles) == 1
    native = read_rawfile(retained_rawfiles[0])
    assert native.npoints > 1
    assert 0 < native.vectors[0].data[-1] < 3e-9
    assert error.value.data_rows == native.npoints


@pytest.mark.skipif(ngspice_missing, reason='ngspice not installed')
@pytest.mark.parametrize('card,endpoint,noise', [
    ('.dc vin 0 1 .3', .9, False),
    ('.dc vin 1 -1 -.3', -.8, False),
    ('.ac dec 2 1 9', 9, False),
    ('.ac oct 2 1 3', 2 ** 1.5, False),
    ('.noise v(out) vin oct 2 1 3', 2 ** 1.5, True),
])
def test_complete_nondivisible_sweep_grid(tmp_path, card, endpoint, noise):
    """A valid stepped sweep need not reach a stop between its grid points."""
    from silk import Module, load_pdk, sources
    from silk.elements import elements
    from silk.simulation import DC, AC, Noise
    fields = card.split()
    if fields[0] == '.dc':
        analysis = DC('vin', *fields[2:5])
    elif fields[0] == '.ac':
        analysis = AC(fields[1], int(fields[2]), fields[3], fields[4])
    else:
        analysis = Noise('out', '0', 'vin', fields[3], int(fields[4]), fields[5], fields[6])
    mod = Module('sky130', str(tmp_path), 'grid.cir', load_pdk('sky130'))
    mod.add_source(sources.VoltageSource('vin', positive='in', negative='0', dc=1, ac=1))
    mod.add(elements.resistor('r1', p='in', n='out', value=1000))
    mod.add(elements.capacitor('c1', p='out', n='0', value=1e-12))
    mod.add_sim_type(analysis)
    mod.generate_netlist()
    assert mod._run_files.completion is not None
    raw = SimulatorBackend.get('ngspice').run(os.path.join(mod.dir_folder, mod.netlistname), str(tmp_path), run_files=mod._run_files)
    from silk.signals.analysis_kind import plot_silk_analysis
    kind = 'noise' if noise else card.split()[0][1:]
    plot = next(p for p in raw.native.plots if plot_silk_analysis(p) == kind
                and (not noise or p.vectors[0].name == 'frequency'))
    if kind == 'ac' and fields[1] == 'dec':
        from silk.simulation._ngspice_version import ngspice_version
        version = ngspice_version()
        if version == '47':
            endpoint = 10 ** .5
        elif version != '42':
            pytest.skip('nondivisible DEC measurement is pinned on ngspice 42 and 47')
    assert plot.vectors[0].data[-1] == pytest.approx(endpoint)


@pytest.mark.skipif(ngspice_missing, reason='ngspice not installed')
@pytest.mark.parametrize('case,last', [('after_end', 10e-9), ('control_override', 2e-9), ('repeated', 10e-9)])
def test_external_deck_false_abort_controls(tmp_path, case, last):
    """Measured reviewer decks are unverified, and must keep usable data."""
    card = '.tran 1n 10n\n'
    if case == 'repeated':
        card += '.tran 1n 20n\n'
    command = 'tran 1n 2n' if case == 'control_override' else 'run'
    text = ('* external control\nvin n 0 1\nr1 n 0 1k\n' + card +
            f'.control\n{command}\nwrite {tmp_path}/rawfile.raw\nquit\n.endc\n.end\n')
    if case == 'after_end':
        text += '.ac lin 3 1 100\n'
    deck = tmp_path / 'external.cir'
    deck.write_text(text)
    raw = SimulatorBackend.get('ngspice').run(str(deck), str(tmp_path))
    assert raw.native.vectors[0].data[-1] == pytest.approx(last)


def _legacy_serializer(tmp_path, monkeypatch):
    """Rebuild the real serializer with frozen pre-TASK-540 lane stimuli."""
    import json
    from pathlib import Path
    from silk import Module, sources
    from silk.examples import serializer_example as example
    frozen = json.loads((Path(__file__).parent / 'fixtures/analysis_completion/serializer_4g_lane_knots.json').read_text())
    captured = []

    class Captured(BaseException):
        pass

    def capture(mod, *args, **kwargs):
        captured.append(mod)
        raise Captured()

    with monkeypatch.context() as patch:
        patch.setattr(Module, 'simulate', capture)
        patch.setattr(example, 'example_output_dir', lambda: str(tmp_path))
        with pytest.raises(Captured):
            example._run_cml(frozen['prbs_bits'], tbit=frozen['tbit'],
                             n_words=frozen['n_words'], probe='dp')
    mod = captured[0]
    for name, points in frozen['lane_points'].items():
        wave = sources.PWL()
        for time, value in points:
            wave.add(time, value)
        source = mod.sig_sources_dict[name]
        previous_names = source.physical_instance_names()
        # Replace BitPattern with PWL in the same public waveform slot.
        mod.update_source(name, ['waveform', 'pwl'], [wave, None])
        assert source.waveform is wave and source.pwl is None
        assert source.selected_waveform() is wave
        assert source.physical_instance_names() == previous_names
        assert len(wave) == 10
        assert wave[0][1] == wave[1][1] == wave[2][1]  # redundant first-word knots
    mod.generate_netlist()
    assert mod._run_files.completion is not None
    assert mod._run_files.completion.requests[0].stop == pytest.approx(frozen['stop'])
    return mod


@pytest.mark.skipif(ngspice_missing, reason='ngspice not installed')
def test_pre540_serializer_certifies_and_rejects_partial_run(tmp_path, monkeypatch):
    from tests._env import pdk_available
    if not pdk_available():
        pytest.skip('needs installed sky130 models')
    mod = _legacy_serializer(tmp_path, monkeypatch)
    with pytest.raises(SimulationAborted) as error:
        mod.run_simulator()
    assert 'requested stop 1.25e-08' in str(error.value)
    assert 'Timestep too small' in error.value.reason
    assert error.value.data_rows > 1
    assert not mod.sim_results_loaded


@pytest.mark.skipif(ngspice_missing, reason='ngspice not installed')
@pytest.mark.parametrize('ppd,stop', [(4, 2000), (10, 2000), (1, 9000)])
def test_ngspice_subdecade_dec_version_grid(tmp_path, ppd, stop):
    """Hub legs select ngspice-47 with a per-command SILK_NGSPICE_BIN override."""
    from silk.simulation._ngspice_version import ngspice_version
    version = ngspice_version()
    if version not in ('42', '47'):
        pytest.skip('sub-decade measurements are pinned on ngspice 42 and 47')
    from silk import Module, load_pdk, sources
    from silk.elements import elements
    from silk.simulation import AC
    import numpy as np
    mod = Module('sky130', str(tmp_path), 'version-grid.cir', load_pdk('sky130'))
    mod.add_source(sources.VoltageSource('vin', positive='in', negative='0', dc=1, ac=1))
    mod.add(elements.resistor('r1', p='in', n='out', value=1000))
    mod.add(elements.capacitor('c1', p='out', n='0', value=1e-12))
    if ppd == 1:
        with pytest.raises(ValueError, match='at least 2 points per decade'):
            AC('dec', ppd, 1000, stop)
        return
    mod.add_sim_type(AC('dec', ppd, 1000, stop))
    mod.generate_netlist()
    assert mod._run_files.completion.requests[0].endpoint_unverified_reason is None
    raw = SimulatorBackend.get('ngspice').run(os.path.join(mod.dir_folder, mod.netlistname), str(tmp_path), run_files=mod._run_files)
    steps = int(np.floor(ppd * np.log10(stop / 1000)))
    expected = (np.geomspace(1000, stop, steps + 1) if version == '42'
                else 1000 * 10 ** (np.arange(steps + 1) / ppd))
    np.testing.assert_allclose(raw.native.vectors[0].data.real, expected, rtol=1e-8)


@pytest.mark.skipif(ngspice_missing, reason='ngspice not installed')
def test_parse_failure_names_card_and_reason(tmp_path):
    from silk.simulation.runner import NgspiceRunner
    from silk.exceptions import SimulationError
    deck = tmp_path / 'duplicate.cir'
    deck.write_text('* duplicate device\nv1 out 0 1\n.subckt twin out\n'
                    'R1 out 0 1k\nr1 out 0 1k\n.ends twin\n'
                    'xdut out twin\n.tran 1n 10n\n.end\n')
    with pytest.raises(SimulationError) as error:
        NgspiceRunner(str(deck), str(tmp_path)).run()
    assert 'r.xdut.r1 out 0 1k' in str(error.value)
    assert 'device already exists, bail out' in str(error.value)
