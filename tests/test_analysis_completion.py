"""Partial saved data must never become a completed analysis."""
import pytest

from silk.exceptions import SimulationAborted
from silk.signals.rawfile import read_rawfile
from silk.simulation.backend import RawResult, SimulatorBackend


@pytest.mark.parametrize('source_type', ['VoltageSource', 'CurrentSource'])
@pytest.mark.parametrize('override', ['class', 'instance'])
@pytest.mark.parametrize('callback', ['counting', 'raising', 'rendering'])
def test_custom_waveform_selector_is_declined_without_callbacks(tmp_path, source_type, override, callback):
    from silk import sources
    from silk.simulation.completion import _opaque_emission_reason, certify_generated
    from tests.test_run_files import _module
    from tests._selector_control import install_selector
    mod = _module(tmp_path)
    source = getattr(sources, source_type)('extra', positive='out', negative='0',
                                         sine=sources.Sine(0, 1, 1e3))
    wave = install_selector(source, override, callback)
    mod.add_source(source)
    reason = _opaque_emission_reason(mod)
    assert reason is not None
    assert 'custom waveform selector' in reason
    plan, reason = certify_generated(mod, '.tran 1n 10n\n.end\n')
    assert plan is None and 'custom waveform selector' in reason
    assert (wave.selector_calls, wave.renderer_calls, wave.vampl) == (0, 0, 1)


@pytest.mark.parametrize('source_type', ['VoltageSource', 'CurrentSource'])
def test_inherited_waveform_selector_remains_certified(tmp_path, source_type):
    from silk import sources
    from tests.test_run_files import _module
    from tests._selector_control import install_selector
    mod = _module(tmp_path)
    source = getattr(sources, source_type)('extra', positive='out', negative='0',
                                         sine=sources.Sine(0, 1, 1e3))
    wave = install_selector(source, 'inherited', 'counting')
    mod.add_source(source)
    mod.generate_netlist()
    assert mod._run_files.completion is not None
    assert (wave.selector_calls, wave.renderer_calls, wave.vampl) == (0, 0, 1)


def _plot(kind, endpoint, points=2, *, start=None):
    names = {'tran': 'Transient Analysis', 'dc': 'DC transfer characteristic',
             'ac': 'AC Analysis', 'noise': 'Noise Spectral Density Curves'}
    axis = 'time' if kind == 'tran' else 'frequency' if kind in ('ac', 'noise') else 'v(v-sweep)'
    if start is None:
        start = 1.0 if kind in ('ac', 'noise') else 0.0
    return (f'Title: test\nDate: today\nPlotname: {names[kind]}\nFlags: real\n'
            f'No. Variables: 2\nNo. Points: {points}\nVariables:\n'
            f'0\t{axis}\t{axis}\n1\tv(out)\tvoltage\n'
            'Values:\n' + ''.join(f'{i}\t{start + (endpoint - start) * i / (points - 1) if i < points - 1 else endpoint}\n\t1\n' for i in range(points)))


def _trusted_files(deck, simulator, *, noise=False):
    """Synthetic certificate; generation provenance is tested separately."""
    from dataclasses import replace
    from hashlib import sha256
    from silk.simulation.completion import CompletionPlan, _requests
    from silk.simulation.run_files import RunFiles
    text = deck.read_text()
    requests = _requests(text, simulator)
    plan = CompletionPlan(simulator, sha256(text.encode()).digest(), requests) if requests is not None else None
    return replace(RunFiles.resolve(str(deck), str(deck.parent), noise=noise), completion=plan)


def _run(tmp_path, monkeypatch, simulator, card, plots):
    deck = tmp_path / 'test.cir'
    deck.write_text('* completion test\n' + card + '\n.end\n')
    path = tmp_path / 'rawfile.raw'
    path.write_text(plots)
    raw = RawResult(read_rawfile(str(path)), str(path), run_files=_trusted_files(deck, simulator))
    backend = SimulatorBackend.get(simulator)
    monkeypatch.setattr(backend, '_launch', lambda *a, **kw: (raw, None, None))
    return backend.run(str(deck), str(tmp_path))


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('kind,card,end', [
    ('tran', '.tran 1n 10n', 2e-9), ('dc', '.dc vin 0 1 .1', .5),
    ('dc', '.dc vin 1 -1 -.1', 0), ('ac', '.ac dec 10 1 100', 10),
    ('noise', '.noise v(out) vin dec 10 1 100', 10),
    ('ac', '.ac lin 3 1 100', 90),
    ('noise', '.noise v(out) vin lin 3 1 100', 90)])
def test_short_analysis_aborts(tmp_path, monkeypatch, simulator, kind, card, end):
    with pytest.raises(SimulationAborted, match=kind) as error:
        _run(tmp_path, monkeypatch, simulator, card, _plot(kind, end))
    assert 'requested stop' in str(error.value)
    assert 'last' in str(error.value)
    assert error.value.data_rows == 2


@pytest.mark.parametrize('kind,card,end,points', [
    ('tran', '.tran 1n 10n', 10e-9 * (1 - 1e-8), 2),
    ('dc', '.dc vin 0 1 .3', .9, 4), ('dc', '.dc vin 1 -1 -.3', -.8, 7),
    ('ac', '.ac dec 2 1 9', 9, 2),
    ('ac', '.ac oct 2 1 3', 2 ** 1.5, 4),
    ('noise', '.noise v(out) vin oct 2 1 3', 2 ** 1.5, 4),
    ('ac', '.ac lin 3 1 100', 100, 3)])
def test_complete_grid_passes(tmp_path, monkeypatch, kind, card, end, points):
    _run(tmp_path, monkeypatch, 'ngspice', card, _plot(kind, end, points))


def test_each_plot_checked(tmp_path, monkeypatch):
    with pytest.raises(SimulationAborted, match='tran'):
        _run(tmp_path, monkeypatch, 'ngspice', '.tran 1n 10n\n.ac lin 3 1 100',
             _plot('tran', 2e-9) + _plot('ac', 100))


def test_missing_second_analysis_aborts(tmp_path, monkeypatch):
    with pytest.raises(SimulationAborted, match='ac'):
        _run(tmp_path, monkeypatch, 'ngspice', '.tran 1n 10n\n.ac lin 3 1 100',
             _plot('tran', 10e-9))


def test_log_only_enriches_data_verdict(tmp_path, monkeypatch):
    deck = tmp_path / 'test.cir'
    deck.write_text('* test\n.tran 1n 10n\n.end\n')
    log = tmp_path / 'output.log'
    log.write_text('Timestep too small; time = 2e-9\n')
    path = tmp_path / 'rawfile.raw'
    backend = SimulatorBackend.get('ngspice')
    def launch(*args, **kwargs):
        return RawResult(read_rawfile(str(path)), str(path), run_files=_trusted_files(deck, 'ngspice')), str(log), None
    monkeypatch.setattr(backend, '_launch', launch)
    path.write_text(_plot('tran', 2e-9))
    with pytest.raises(SimulationAborted) as error:
        backend.run(str(deck), str(tmp_path))
    assert 'Timestep too small' in error.value.reason
    path.write_text(_plot('tran', 10e-9))
    backend.run(str(deck), str(tmp_path))


@pytest.mark.parametrize('endpoint,aborted', [(10, True), (100, False)])
def test_xyce_noise_csv_completion(tmp_path, monkeypatch, endpoint, aborted):
    from silk.simulation.run_files import RunFiles
    deck = tmp_path / 'noise.cir'
    deck.write_text('* test\n.noise v(out) vin dec 10 1 100\n.end\n')
    files = _trusted_files(deck, 'xyce', noise=True)
    from pathlib import Path
    Path(files.noise_path).write_text('FREQ,ONOISE,INOISE\n' + ''.join(f'{1 + (endpoint - 1) * i / 20},1,1\n' for i in range(21)))
    raw = RawResult(None, files.noise_path, run_files=files)
    backend = SimulatorBackend.get('xyce')
    monkeypatch.setattr(backend, '_launch', lambda *a, **kw: (raw, None, None))
    if aborted:
        with pytest.raises(SimulationAborted, match='noise'):
            backend.run(str(deck), str(tmp_path))
    else:
        backend.run(str(deck), str(tmp_path))


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
def test_direct_runner_load_keeps_data_presence_only(tmp_path, simulator):
    from silk.simulation.runner import NgspiceRunner
    from silk.simulation.xyce_runner import XyceRunner
    deck = tmp_path / 'test.cir'
    deck.write_text('* test\n.tran 1n 10n\n.end\n')
    (tmp_path / 'rawfile.raw').write_text(_plot('tran', 2e-9))
    cls = NgspiceRunner if simulator == 'ngspice' else XyceRunner
    runner = cls(str(deck), str(tmp_path))
    assert runner.load().npoints == 2
    assert runner.results is not None


def test_run_all_tests_records_partial_run_as_failure(tmp_path, monkeypatch):
    from tests.test_run_all_tests_error_classification import _dut
    mod, tb = _dut(tmp_path, 'dut', 'dut.cir')
    backend = SimulatorBackend.get('ngspice')
    def launch(self, files, **kwargs):
        from pathlib import Path
        Path(files.rawfile_path).write_text(_plot('tran', 1e-6))
        return RawResult(read_rawfile(files.rawfile_path), files.rawfile_path,
                         run_files=files), None, None
    monkeypatch.setattr(type(backend), '_launch_prepared', launch)
    results = mod.run_all_tests()
    assert 'error' in results['dut.level']
    assert 'tran analysis aborted' in results['dut.level']['error']
    assert not tb.netlist.sim_results_loaded


def test_monte_carlo_records_partial_sample_as_failure(tmp_path, monkeypatch):
    from tests.test_monte_carlo import _bench
    from silk.pdk import statistical
    monkeypatch.setattr(statistical, 'require_library_sections', lambda *a, **kw: None)
    monkeypatch.setenv('PDK_ROOT', str(tmp_path / 'missing-pdk'))
    tb = _bench(tmp_path, 'sky130', 'nmos1p8')
    import os
    tb.netlist.pdk_details.library_location = os.devnull
    tb.set_monte_carlo(n=2)
    backend = SimulatorBackend.get('ngspice')
    def launch(self, files, **kwargs):
        from pathlib import Path
        end = 1e-9 if 'mc_0000' in files.directory else 2e-9
        Path(files.rawfile_path).write_text(_plot('tran', end))
        return RawResult(read_rawfile(files.rawfile_path), files.rawfile_path,
                         run_files=files), None, None
    monkeypatch.setattr(type(backend), '_launch_prepared', launch)
    campaign = tb.run()
    assert set(campaign.failed) == {('tt', 0)}
    assert set(campaign.reports) == {('tt', 1)}
    assert all('tran analysis aborted' in failure.error for failure in campaign.failed.values())


def test_integrated_noise_totals_are_not_a_frequency_axis(tmp_path, monkeypatch):
    totals = _plot('noise', 1e-10).replace('Noise Spectral Density Curves', 'Integrated Noise')
    totals = totals.replace('frequency\tfrequency', 'onoise_total\tvoltage')
    _run(tmp_path, monkeypatch, 'ngspice', '.noise v(out) vin dec 10 1 100',
         _plot('noise', 100, 21) + totals)


@pytest.mark.parametrize('kind,card,end', [
    ('tran', '.tran 1n 10n', 10e-9 * (1 - 2e-7)),
    ('tran', '.tran 1e-19 1e-18', 5e-19),
    ('dc', '.dc vin 1 0 -.1', 2e-7),
    ('ac', '.ac lin 3 1 100', float('nan')),
    ('ac', '.ac lin 3 1 100', float('inf')),
])
def test_outside_tolerance_or_nonfinite_endpoint_aborts(tmp_path, monkeypatch, kind, card, end):
    with pytest.raises(SimulationAborted):
        _run(tmp_path, monkeypatch, 'ngspice', card, _plot(kind, end))


def test_noise_totals_alone_do_not_prove_frequency_completion(tmp_path, monkeypatch):
    totals = _plot('noise', 100).replace('Noise Spectral Density Curves', 'Integrated Noise')
    totals = totals.replace('frequency\tfrequency', 'onoise_total\tvoltage')
    with pytest.raises(SimulationAborted, match='noise'):
        _run(tmp_path, monkeypatch, 'ngspice', '.noise v(out) vin dec 10 1 100', totals)


def test_prepared_run_uses_the_launched_deck_request(tmp_path, monkeypatch):
    from silk.simulation.run_files import RunFiles
    original = tmp_path / 'original.cir'
    original.write_text('* old request\n.tran 1n 10n\n.end\n')
    prepared = tmp_path / 'prepared.cir'
    prepared.write_text('* actual request\n.tran 1n 20n\n.end\n')
    files = _trusted_files(prepared, 'ngspice')
    from pathlib import Path
    Path(files.rawfile_path).write_text(_plot('tran', 10e-9))
    raw = RawResult(read_rawfile(files.rawfile_path), files.rawfile_path)
    backend = SimulatorBackend.get('ngspice')
    monkeypatch.setattr(backend, '_launch_prepared', lambda *a, **kw: (raw, None, None))
    with pytest.raises(SimulationAborted, match='requested stop 2e-08'):
        backend.run(str(original), str(tmp_path), run_files=files)


@pytest.mark.parametrize('pathway', ['loader', 'runner', 'backend'])
def test_xyce_header_only_with_blank_point_count_raises_typed_error(tmp_path, monkeypatch, pathway):
    """Xyce 7.10's measured late abort leaves this header and no binary bytes."""
    path = tmp_path / 'rawfile.raw'
    path.write_bytes(
        b'Title: Xyce abort\nDate: today\nPlotname: Transient Analysis\n'
        b'Flags: real\nNo. Variables: 2\nNo. Points:                 \n'
        b'Variables:\n0\tTIME\ttime\n1\tV(N)\tvoltage\nBinary:\n'
    )
    deck = tmp_path / 'test.cir'
    deck.write_text('* measured request\n.tran 10p 3n\n.end\n')
    backend = SimulatorBackend.get('xyce')
    with pytest.raises(SimulationAborted) as error:
        if pathway == 'loader':
            backend.load_raw(str(path), simulator='xyce')
        elif pathway == 'runner':
            from silk.simulation.xyce_runner import XyceRunner
            runner = XyceRunner(str(deck), str(tmp_path))
            runner.stdout = 'Time step too small near step number: 10  Exiting transient loop.'
            runner.load()
        else:
            raw = RawResult(read_rawfile(str(path)), str(path))
            monkeypatch.setattr(backend, '_launch', lambda *a, **kw: (raw, None, None))
            backend.run(str(deck), str(tmp_path))
    assert error.value.data_rows == 0
    assert error.value.simulator == 'xyce'
    if pathway == 'runner':
        assert runner.results is None
        assert 'Time step too small' in error.value.reason


@pytest.mark.parametrize('version,ppd,stop,frequencies', [
    ('42', 4, 2000, [1000, 2000]),
    ('47', 4, 2000, [1000, 1778.279410038923]),
    ('42', 10, 2000, [1000, 1259.921049895, 1587.401051968, 2000]),
    ('47', 10, 2000, [1000, 1258.925411794, 1584.893192461, 1995.26231496888]),
    ('47', 1, 9000, [1000]),
])
def test_subdecade_dec_accepts_measured_version_grids(tmp_path, monkeypatch, version, ppd, stop, frequencies):
    monkeypatch.setattr('silk.simulation._ngspice_version.ngspice_version', lambda: version)
    text = _plot('ac', frequencies[-1], len(frequencies), start=frequencies[0])
    header = text.split('Values:')[0].replace('Title: test', 'Title: ngspice-' + version)
    text = header + 'Values:\n' + ''.join(f'{i}\t{freq}\n\t1\n' for i, freq in enumerate(frequencies))
    raw = _run(tmp_path, monkeypatch, 'ngspice', f'.ac dec {ppd} 1000 {stop}', text)
    request = raw.run_files.completion.requests[0]
    assert request.endpoint_unverified_reason is None
    assert raw.native.npoints == len(frequencies)


def test_subdecade_dec_does_not_disable_other_requests(tmp_path, monkeypatch):
    monkeypatch.setattr('silk.simulation._ngspice_version.ngspice_version', lambda: '47')
    with pytest.raises(SimulationAborted, match='tran'):
        _run(tmp_path, monkeypatch, 'ngspice', '.ac dec 4 1000 2000\n.tran 1n 10n',
             _plot('ac', 1778.279410038923, start=1000) + _plot('tran', 2e-9))


def test_dec_full_decade_retains_completion_verdict(tmp_path, monkeypatch):
    with pytest.raises(SimulationAborted, match='ac'):
        _run(tmp_path, monkeypatch, 'ngspice', '.ac dec 4 1000 10000',
             _plot('ac', 5623.413, 4, start=1000))


def test_abort_reason_prefers_timestep_failure_over_earlier_warning(tmp_path):
    from silk.simulation.runner import abort_reason_from_log
    log = tmp_path / 'output.log'
    failure = 'doAnalyses: TRAN: Timestep too small; time = 1.5e-9'
    log.write_text('Warning: singular matrix: check nodes n and b1#branch\n'
                   + failure + '\nrun simulation(s) aborted\n')
    assert abort_reason_from_log(str(log)) == (
        failure + ' (earlier: Warning: singular matrix: check nodes n and b1#branch)')


def test_endpoint_policy_is_limited_to_spice_backends(tmp_path):
    from silk.simulation.completion import require_complete
    deck = tmp_path / 'test.cir'
    deck.write_text('* test\n.tran 1n 10n\n.end\n')
    path = tmp_path / 'rawfile.raw'
    path.write_text(_plot('tran', 2e-9))
    raw = RawResult(read_rawfile(str(path)), str(path))
    require_complete(raw, str(deck), simulator='behavioral')


@pytest.mark.parametrize('endpoint,aborted', [(3.162278, False), (1.0, True)])
def test_measured_xyce_dec_ac_grid(tmp_path, monkeypatch, endpoint, aborted):
    if aborted:
        with pytest.raises(SimulationAborted, match='ac'):
            _run(tmp_path, monkeypatch, 'xyce', '.ac dec 2 1 9', _plot('ac', endpoint))
    else:
        _run(tmp_path, monkeypatch, 'xyce', '.ac dec 2 1 9', _plot('ac', endpoint))


@pytest.mark.parametrize('complete', [False, True])
def test_measured_xyce_oct_noise_grid(tmp_path, monkeypatch, complete):
    from pathlib import Path
    from silk.simulation.run_files import RunFiles
    deck = tmp_path / 'test.cir'
    deck.write_text('* measured Xyce grid\n.noise v(out) vin oct 2 1 3\n.end\n')
    files = _trusted_files(deck, 'xyce', noise=True)
    frequencies = [1, 1.41421356, 2, 2.82842712]
    if not complete:
        frequencies.pop()
    Path(files.noise_path).write_text('FREQ,ONOISE,INOISE\n' +
                                    ''.join(f'{f},1,1\n' for f in frequencies))
    raw = RawResult(None, files.noise_path, run_files=files)
    backend = SimulatorBackend.get('xyce')
    monkeypatch.setattr(backend, '_launch', lambda *a, **kw: (raw, None, None))
    if complete:
        backend.run(str(deck), str(tmp_path))
    else:
        with pytest.raises(SimulationAborted, match='noise'):
            backend.run(str(deck), str(tmp_path))


def test_external_deck_has_no_endpoint_verdict(tmp_path, monkeypatch):
    deck = tmp_path / 'external.cir'
    deck.write_text('* external\n.tran 1n 10n\n.end\n.ac lin 3 1 100\n')
    path = tmp_path / 'rawfile.raw'
    path.write_text(_plot('tran', 2e-9))
    raw = RawResult(read_rawfile(str(path)), str(path))
    backend = SimulatorBackend.get('ngspice')
    monkeypatch.setattr(backend, '_launch', lambda *a, **kw: (raw, None, None))
    assert backend.run(str(deck), str(tmp_path)).native.npoints == 2


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('kind,card,stop', [
    ('dc', '.dc vin 1 1.0000001 1e-8', 1.0000001),
    ('dc', '.dc vin 1.0000001 1 -1e-8', 1.0),
    ('ac', '.ac lin 11 1 1.0000001', 1.0000001),
    ('noise', '.noise v(out) vin lin 11 1 1.0000001', 1.0000001),
])
@pytest.mark.parametrize('output', ['first_only', 'short_endpoint', 'missing_row', 'complete', 'rounded'])
def test_narrow_sweep_requires_final_step_and_count(tmp_path, monkeypatch, simulator,
                                                   kind, card, stop, output):
    from silk.utils._utils import _parse_si
    fields = card.split()
    start = _parse_si(fields[2 if kind == 'dc' else 3 if kind == 'ac' else 5])
    points = 1 if output == 'first_only' else 10 if output == 'missing_row' else 11
    end = start if output == 'first_only' else stop
    if output == 'short_endpoint':
        end = stop - (1e-8 if stop > start else -1e-8)
    if output == 'rounded':
        end = stop - (1e-10 if stop > start else -1e-10)
    # Explicit axis values; output rounding is smaller than the final step.
    names = {'dc': 'DC transfer characteristic', 'ac': 'AC Analysis',
             'noise': 'Noise Spectral Density Curves'}
    axis = 'v(v-sweep)' if kind == 'dc' else 'frequency'
    values = [start + (end - start) * i / max(points - 1, 1) for i in range(points)]
    plot = (f'Title: narrow\nPlotname: {names[kind]}\nFlags: real\n'
            f'No. Variables: 2\nNo. Points: {points}\nVariables:\n'
            f'0\t{axis}\t{axis}\n1\tv(out)\tvoltage\nValues:\n' +
            ''.join(f'{i}\t{x:.12g}\n\t1\n' for i, x in enumerate(values)))
    if output in ('complete', 'rounded'):
        _run(tmp_path, monkeypatch, simulator, card, plot)
    else:
        with pytest.raises(SimulationAborted):
            _run(tmp_path, monkeypatch, simulator, card, plot)


@pytest.mark.parametrize('plots', [
    _plot('tran', 2e-9) + _plot('tran', 10e-9),
    _plot('tran', 10e-9) + _plot('tran', 2e-9),
])
def test_ambiguous_same_kind_plots_have_no_endpoint_verdict(tmp_path, monkeypatch, plots):
    # Even a missing second analysis must not be judged before association.
    _run(tmp_path, monkeypatch, 'ngspice', '.tran 1n 10n\n.ac lin 3 1 100', plots)


def test_unique_plot_association_checks_each_request(tmp_path, monkeypatch):
    with pytest.raises(SimulationAborted, match='ac'):
        _run(tmp_path, monkeypatch, 'ngspice', '.tran 1n 10n\n.ac lin 3 1 100',
             _plot('tran', 10e-9) + _plot('ac', 50, 3))


@pytest.mark.parametrize('custom', [
    '.dc vin 0 1 .5 vbias 0 2 1',
    '.tran 1n 10n\n.tran 1n 20n',
    '.tran 1n 10n\n.step vin 0 2 1',
    '.tran 1n 10n\n+ uic',
    '.control\ntran 1n 2n\n.endc',
])
def test_custom_analysis_content_is_not_certified(tmp_path, custom):
    from tests.test_run_files import _module
    mod = _module(tmp_path)
    mod.add_custom_line(custom)
    mod.generate_netlist()
    assert mod._run_files.completion is None
    assert mod._run_files.completion_unverified_reason == 'Module.add_custom_line()/custom_lines'


def test_generation_certificate_survives_pickle_and_is_bound_to_bytes(tmp_path, monkeypatch):
    import pickle
    from pathlib import Path
    from tests.test_run_files import _module
    mod = _module(tmp_path)
    mod.generate_netlist()
    files = pickle.loads(pickle.dumps(mod._run_files))
    assert files.completion.requests[0].stop == 1e-6
    Path(files.rawfile_path).write_text(_plot('tran', 2e-9))
    raw = RawResult(read_rawfile(files.rawfile_path), files.rawfile_path, run_files=files)
    backend = SimulatorBackend.get('ngspice')
    monkeypatch.setattr(backend, '_launch_prepared', lambda *a, **kw: (raw, None, None))
    with pytest.raises(SimulationAborted):
        backend.run(files.netlist_path, files.directory, run_files=files)
    # An external alteration loses the endpoint guarantee, retaining presence.
    with Path(files.netlist_path).open('a') as output:
        output.write('* changed after generation\n')
    backend.run(files.netlist_path, files.directory, run_files=files)


def test_nested_dc_parser_is_explicitly_unverified():
    from silk.simulation.completion import _requests
    for simulator in ('ngspice', 'xyce'):
        assert _requests('* nested\n.dc vin 0 1 .5 vbias 0 2 1\n.end\n', simulator) is None


def test_effective_deck_boundary_ignores_analysis_after_end():
    from silk.simulation.completion import _requests
    requests = _requests('* title\n.tran 1n 10n\n.end\n.ac lin 3 1 100\n', 'ngspice')
    assert [r.kind for r in requests] == ['tran']


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
def test_nested_dc_external_first_slice_has_presence_only(tmp_path, monkeypatch, simulator):
    _run(tmp_path, monkeypatch, simulator, '.dc vin 0 1 .5 vbias 0 2 1', _plot('dc', 1, 3))


@pytest.mark.parametrize('api', ['analysis', 'netlister', 'source', 'waveform', 'child_custom_line'])
def test_opaque_extension_api_entries_are_not_certified(tmp_path, api):
    from tests.test_run_files import _module
    from silk import Module, sources
    from silk.simulation import Transient
    from silk.circuit.spice_netlister import PySpiceNetlister
    mod = _module(tmp_path)
    if api == 'analysis':
        class CustomTransient(Transient):
            pass
        mod.add_sim_type(CustomTransient(1e-7, 1e-6))
    elif api == 'netlister':
        class CustomNetlister(PySpiceNetlister):
            pass
        mod.netlister = CustomNetlister(mod.pdk_details)
    elif api == 'source':
        class CustomSource(sources.VoltageSource):
            def to_spice(self):
                return super().to_spice()
        mod.add_source(CustomSource('extra', positive='out', negative='0', dc=1))
    elif api == 'waveform':
        class CustomPWL(sources.PWL):
            def to_spice(self):
                return super().to_spice()
        mod.add_source(sources.VoltageSource('extra', positive='out', negative='0',
                                             pwl=CustomPWL().add(0, 1).add(1e-6, 1)))
    else:
        from silk.circuit.circuit_elements import PortDir
        child = Module('sky130', str(tmp_path), 'child.cir', mod.pdk_details, module_name='child')
        child.define_module_ports({'p': PortDir.IN})
        child.add_custom_line('.param x=1')
        mod.add_module(child, 'child', nets={'p': 'out'})
    mod.generate_netlist()
    assert mod._run_files.completion is None
    assert mod._run_files.completion_unverified_reason


def test_builtin_pwl_knots_do_not_remove_certification(tmp_path):
    from tests.test_run_files import _module
    from silk import sources
    mod = _module(tmp_path)
    mod.add_source(sources.VoltageSource('extra', positive='out', negative='0',
                                         pwl=sources.PWL().add(0, 1).add(1e-7, 1).add(2e-7, 1)))
    mod.generate_netlist()
    assert mod._run_files.completion is not None


@pytest.mark.parametrize('emitter', ['pdk', 'child_netlist', 'child_subckt'])
def test_custom_pdk_and_module_emission_hooks_are_not_certified(tmp_path, monkeypatch, emitter):
    from tests.test_run_files import _module
    from silk import Module
    from silk.circuit.circuit_elements import PortDir
    mod = _module(tmp_path)
    if emitter == 'pdk':
        original = mod.pdk_details.generate_netlist_entry
        monkeypatch.setattr(mod.pdk_details, 'generate_netlist_entry', lambda *a, **kw: original(*a, **kw))
    else:
        class Child(Module):
            def generate_netlist(self, *args, **kwargs):
                return super().generate_netlist(*args, **kwargs)
            def generate_subckt(self, *args, **kwargs):
                return super().generate_subckt(*args, **kwargs)
        if emitter == 'child_netlist':
            Child.generate_subckt = Module.generate_subckt
        else:
            Child.generate_netlist = Module.generate_netlist
        child = Child('sky130', str(tmp_path), 'child.cir', mod.pdk_details, module_name='child')
        child.define_module_ports({'p': PortDir.IN})
        mod.add_module(child, 'child', nets={'p': 'out'})
    mod.generate_netlist()
    assert mod._run_files.completion is None


def test_missing_certified_noise_csv_is_an_aborted_analysis(tmp_path, monkeypatch):
    deck = tmp_path / 'test.cir'
    deck.write_text('* certified noise\n.noise v(out) vin dec 1 1 10\n.end\n')
    files = _trusted_files(deck, 'xyce', noise=True)
    path = tmp_path / 'rawfile.raw'
    path.write_text(_plot('tran', 1))
    raw = RawResult(read_rawfile(str(path)), str(path), run_files=files)
    backend = SimulatorBackend.get('xyce')
    monkeypatch.setattr(backend, '_launch', lambda *a, **kw: (raw, None, None))
    with pytest.raises(SimulationAborted, match='noise') as error:
        backend.run(str(deck), str(tmp_path))
    assert error.value.data_rows == 0


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('kind,descending', [('dc', False), ('dc', True), ('ac', False), ('noise', False)])
def test_real_emission_certifies_narrow_numeric_requests(tmp_path, simulator, kind, descending):
    from tests.test_run_files import _module
    from silk.simulation import DC, AC, Noise
    if kind == 'dc':
        analysis = DC('vin', '1.0000001' if descending else '1',
                      '1' if descending else '1.0000001', '-1e-8' if descending else '1e-8')
    elif kind == 'ac':
        analysis = AC('lin', 11, '1', '1.0000001')
    else:
        analysis = Noise('out', '0', 'vin', 'lin', 11, '1', '1.0000001')
    mod = _module(tmp_path, simulator, [analysis])
    mod.generate_netlist()
    assert mod._run_files.completion is not None
    request, = mod._run_files.completion.requests
    assert request.kind == kind
    assert request.count == 11
    assert request.final_interval == pytest.approx(1e-8)
    assert request.end == (1 if descending else 1.0000001)


@pytest.mark.parametrize('simulator,card,kind,hint', [
    ('ngspice', '.ac lin 2 1000 2000', 'ac', True),
    ('xyce', '.ac lin 2 1000 2000', 'ac', False),
    ('ngspice', '.ac lin 3 1000 2000', 'ac', False),
    ('ngspice', '.ac dec 2 1000 10000', 'ac', False),
    ('ngspice', '.noise v(out) vin lin 2 1000 2000', 'noise', False),
    ('ngspice', '.dc vin 1000 2000 1000', 'dc', False),
])
def test_ngspice_two_point_lin_ac_diagnostic(tmp_path, monkeypatch, simulator, card, kind, hint):
    with pytest.raises(SimulationAborted) as caught:
        _run(tmp_path, monkeypatch, simulator, card, _plot(kind, 1000, 1, start=1000))
    assert caught.value.data_rows == 1
    message = str(caught.value)
    assert ('ngspice wrote 1 of 2 requested LIN AC points' in message) == hint
    if hint:
        assert 'writes only the start frequency for a 2-point LIN AC sweep' in message
        assert 'use 3 or more points or Xyce' in message
        assert 'requested stop 2000' in message
        assert 'last frequency reached 1000.0' in message


def test_complete_two_point_lin_ac_still_passes(tmp_path, monkeypatch):
    raw = _run(tmp_path, monkeypatch, 'ngspice', '.ac lin 2 1000 2000',
               _plot('ac', 2000, 2, start=1000))
    assert raw.native.npoints == 2


def test_subdecade_dec_still_requires_data(tmp_path, monkeypatch):
    with pytest.raises(SimulationAborted) as caught:
        _run(tmp_path, monkeypatch, 'ngspice', '.ac dec 4 1000 2000',
             _plot('ac', 1778.28, 0, start=1000))
    assert caught.value.data_rows == 0


def test_xyce_sample_step_single_point_dc_is_complete(tmp_path, monkeypatch):
    text = ('Title: sample\nDate: today\n'
            'Plotname: Step Analysis: Step 1 of 1 params: name = ... DC transfer characteristic\n'
            'Flags: real\nNo. Variables: 3\nNo. Points: 1\nVariables:\n'
            '0 sweep sweep\n1 V(G1) voltage\n2 V(G2) voltage\nValues:\n'
            '0 2e-5\n\t.7\n\t.7\n')
    raw = _run(tmp_path, monkeypatch, 'xyce', '.dc Ii1 20u 20u 1u\n.options SAMPLES numsamples=1', text)
    assert raw.native.npoints == 1
    assert [v.name for v in raw.native.vectors] == ['sweep', 'V(G1)', 'V(G2)']


@pytest.mark.parametrize('version,end', [('42', 1778.279410038923), ('43', 1500), ('47', 1500)])
def test_subdecade_dec_rejects_truncated_version_grid(tmp_path, monkeypatch, version, end):
    monkeypatch.setattr('silk.simulation._ngspice_version.ngspice_version', lambda: version)
    with pytest.raises(SimulationAborted, match='ac'):
        _run(tmp_path, monkeypatch, 'ngspice', '.ac dec 4 1000 2000',
             _plot('ac', end, start=1000))


def test_unknown_ngspice_subdecade_grid_remains_unverified(tmp_path, monkeypatch):
    monkeypatch.setattr('silk.simulation._ngspice_version.ngspice_version', lambda: None)
    raw = _run(tmp_path, monkeypatch, 'ngspice', '.ac dec 4 1000 2000',
               _plot('ac', 1778.279410038923, start=1000))
    assert raw.run_files.completion.requests[0].endpoint_unverified_reason


@pytest.mark.parametrize('terminator', ['quit 0', 'exit'])
def test_generated_control_block_keeps_its_completion_certificate(terminator):
    from silk.circuit.spice_netlister import PySpiceNetlister
    from silk.simulation.completion import _requests
    control = PySpiceNetlister.__new__(PySpiceNetlister).generate_control_block('/out')
    assert control.endswith('quit 0\n.endc\n.end\n')
    deck = '* gen\nR1 a 0 1\n.tran 1n 1u\n' + control.replace('quit 0', terminator)
    requests = _requests(deck, 'ngspice')
    assert requests is not None and [r.kind for r in requests] == ['tran']
