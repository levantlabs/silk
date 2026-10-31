"""Version-policy boundaries and user-visible advice, without a simulator."""
import inspect
from pathlib import Path
import warnings

import pytest

from silk import doctor
from silk.simulation import _ngspice_version


def policy(version):
    from silk.simulation._ngspice_compat import NgspiceCompatibility
    return NgspiceCompatibility(version)


@pytest.mark.parametrize('version,nosub,osdi,nointernals,pwl', [
    ('42', False, False, False, False),
    ('43', True, False, False, False),
    ('44.2', True, True, False, False),
    ('45.2', True, True, True, False),
    ('47', True, True, True, True),
    ('47+', True, True, True, True),
    (None, False, False, False, False),
])
def test_feature_boundaries(version, nosub, osdi, nointernals, pwl):
    p = policy(version)
    assert [p.has(name) for name in ('save_nosub', 'osdi_v04',
                                    'save_nointernals', 'current_pwl_timing')] == [
                                        nosub, osdi, nointernals, pwl]


@pytest.mark.parametrize('version,affected', [
    ('38', False), ('39', True), ('42', True), ('42+', True),
    ('43', False), ('45.2', False), ('47+', False), (None, True),
])
def test_runaway_is_conservative_for_unknown_builds(version, affected):
    assert policy(version).has_bug('dec_subdecade_runaway') is affected
    assert policy(version).has_bug('ac_lin_two')


@pytest.mark.parametrize('version,expected', [('42', True), ('42.0', False),
                                             ('42+', False), ('43', False), (None, False)])
def test_waveform_verification_preserves_exact_release_rule(version, expected):
    assert policy(version).waveforms_verified is expected


def test_selected_version_reuses_existing_probe(monkeypatch):
    from silk.simulation._ngspice_compat import NgspiceCompatibility
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: '45.2')
    assert NgspiceCompatibility().version == '45.2'
    assert not NgspiceCompatibility().has('sffm_carrier_first')
    assert policy('46').has('sffm_carrier_first')


@pytest.mark.parametrize('version,status', [('41', doctor.WARN), ('42', doctor.OK)])
def test_doctor_advice_is_actionable_and_nonfatal(monkeypatch, capsys, version, status):
    monkeypatch.setattr(doctor.shutil, 'which', lambda _: '/fake/ngspice')
    monkeypatch.setattr(doctor, '_version_line', lambda *a: 'ngspice-' + version)
    checks = doctor.check_ngspice()
    advice = next(c for c in checks if c.name == 'ngspice support')
    assert advice.status == status
    assert advice.detail == policy(version).advice
    assert 'an AC sweep with dec spacing too short to hold one step' in advice.detail
    assert 'fixed in 43' in advice.detail
    assert 'Update ngspice when possible' in advice.detail
    assert 'LIN' not in advice.detail  # no update can fix an unfixed bug
    if version == '41':
        assert 'silk supports releases from 42 onward' in advice.detail
    monkeypatch.setattr(doctor, 'run_checks', lambda: checks)
    assert doctor.main([]) == 0
    assert 'Update ngspice when possible' in capsys.readouterr().out


def test_no_speculative_update_advice():
    assert policy(None).advice is None
    assert policy('47+').advice is None


@pytest.mark.parametrize('version,affected', [
    ('42', False), ('44.2', False), ('45', True),
    ('45.2', True), ('46', True), ('47', False),
])
def test_sourcepath_slowdown_warning(monkeypatch, version, affected):
    from silk.simulation import _ngspice_compat as compat
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: version)
    monkeypatch.setattr(compat, '_advised_binaries', set())
    monkeypatch.setattr(compat.shutil, 'which', lambda _: '/fake/ngspice')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        compat.warn_version_advice('ngspice')
    slowdown = [w for w in caught if 'sky130' in str(w.message)]
    assert bool(slowdown) is affected
    if affected:
        assert len(slowdown) == 1
        assert slowdown[0].category is compat.NgspiceVersionWarning
        message = str(slowdown[0].message)
        assert f'ngspice {version}' in message
        assert 'several times slower' in message
        assert 'fixed in 47' in message
    assert policy(version).has_bug('sourcepath_duplicate_slowdown') is affected


def test_runtime_advice_once_per_canonical_binary(monkeypatch, tmp_path):
    from silk.simulation import _ngspice_compat as compat
    first = tmp_path / 'ngspice'
    first.touch()
    alias = tmp_path / 'alias'
    alias.symlink_to(first)
    second = tmp_path / 'other'
    second.touch()
    monkeypatch.setattr(compat, '_advised_binaries', set())
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: '42')
    monkeypatch.setattr(compat.shutil, 'which', lambda p: p)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for exe in (str(first), str(alias), str(second), str(first)):
            compat.warn_version_advice(exe)
    assert len(caught) == 2
    assert all(w.category is compat.NgspiceVersionWarning for w in caught)
    assert all(str(w.message) == policy('42').advice for w in caught)


def test_runtime_missing_binary_is_silent(monkeypatch):
    from silk.simulation import _ngspice_compat as compat
    monkeypatch.setattr(compat.shutil, 'which', lambda _: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('error')
        compat.warn_version_advice('missing')
    assert not caught


@pytest.mark.parametrize('version', ['41', '42'])
def test_runner_emits_filterable_advice(monkeypatch, tmp_path, version):
    from types import SimpleNamespace
    from silk.simulation import runner, _ngspice_compat as compat
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: version)
    monkeypatch.setattr(compat, '_advised_binaries', set())
    monkeypatch.setattr(compat.shutil, 'which', lambda _: '/fake/ngspice')
    def launch(argv, **kwargs):
        Path(argv[argv.index('-r') + 1]).write_text('stub')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, 'run', launch)
    (tmp_path / 'deck.cir').write_text('Warning fixture\n.end\n')
    run = runner.NgspiceRunner(str(tmp_path / 'deck.cir'), str(tmp_path))
    with pytest.warns(compat.NgspiceVersionWarning) as caught:
        caller_line = inspect.currentframe().f_lineno + 1
        run.run()
    assert caught[0].filename == __file__
    assert caught[0].lineno == caller_line
    assert str(caught[0].message) == policy(version).advice
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        run.run()  # second launch is silent even with warnings as errors
    monkeypatch.setattr(compat, '_advised_binaries', set())
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        warnings.simplefilter('ignore', compat.NgspiceVersionWarning)
        run.run()  # callers can filter just this information category


def test_concurrent_launches_advise_once(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    import time
    from silk.simulation import _ngspice_compat as compat
    monkeypatch.setattr(compat, '_advised_binaries', set())
    monkeypatch.setattr(compat.shutil, 'which', lambda _: '/fake/ngspice')
    def version():
        time.sleep(.02)  # a real uncached subprocess probe releases the GIL
        return '42'
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', version)
    ready = Barrier(4)
    def launch(_):
        ready.wait(timeout=5)
        compat.warn_version_advice('ngspice')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(launch, range(4)))
    assert len(caught) == 1


@pytest.mark.parametrize('version', ['41', '42'])
def test_first_launch_under_warning_errors_still_runs(monkeypatch, tmp_path, version):
    from types import SimpleNamespace
    from silk.simulation import runner, _ngspice_compat as compat
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: version)
    monkeypatch.setattr(compat, '_advised_binaries', set())
    monkeypatch.setattr(compat.shutil, 'which', lambda _: '/fake/ngspice')
    launches = []
    def launch(argv, **kwargs):
        assert '-b' in argv
        launches.append(argv)
        Path(argv[argv.index('-r') + 1]).write_text('successful launch')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, 'run', launch)
    (tmp_path / 'deck.cir').write_text('Warning fixture\n.end\n')
    run = runner.NgspiceRunner(str(tmp_path / 'deck.cir'), str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        filters = list(warnings.filters)
        run.run()  # fresh cache, no category-specific filter
        assert Path(run.rawfile_path).read_text() == 'successful launch'
        run.run()
        assert warnings.filters == filters
        with pytest.raises(UserWarning, match='unrelated'):
            warnings.warn('unrelated', UserWarning)
    assert len(launches) == 2
    # Even switching back to visible warnings must not re-advise this binary.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        compat.warn_version_advice('ngspice')
    assert not caught


@pytest.mark.parametrize('kind', ['warning', 'exception', 'other_advisory'])
def test_advice_does_not_swallow_warning_handler_failures(monkeypatch, kind):
    from silk.simulation import _ngspice_compat as compat
    error_type = {'warning': UserWarning, 'exception': RuntimeError,
                  'other_advisory': compat.NgspiceVersionWarning}[kind]
    error = error_type('unrelated handler failure')
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: '42')
    monkeypatch.setattr(compat, '_advised_binaries', set())
    monkeypatch.setattr(compat.shutil, 'which', lambda _: '/fake/ngspice')
    def broken_handler(*args, **kwargs):
        raise error
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        monkeypatch.setattr(warnings, 'showwarning', broken_handler)
        with pytest.raises(error_type) as caught:
            compat.warn_version_advice('ngspice')
    assert caught.value is error


BUG_TEXT = ('an AC sweep with dec spacing too short to hold one step '
            '(for example .ac dec 1 1k 2k) never finishes and uses unbounded memory')


@pytest.mark.parametrize('version,expected', [
    ('42', 'ngspice 42 works with silk, but a newer release fixes a bug that affects it: '
     + BUG_TEXT + ' (fixed in 43). Update ngspice when possible.'),
    ('41', 'ngspice 41 is older than the minimum version silk supports '
     '(silk supports releases from 42 onward). A newer release fixes a bug that affects it: '
     + BUG_TEXT + ' (fixed in 43). Update ngspice when possible.'),
    ('38', 'ngspice 38 is older than the minimum version silk supports '
     '(silk supports releases from 42 onward). Update ngspice when possible.'),
])
def test_plain_advice_text(version, expected):
    assert policy(version).advice == expected


def test_advice_uses_bug_row_text(monkeypatch):
    p = policy('42')
    row = p.policy['behaviours']['dec_subdecade_runaway']
    monkeypatch.setitem(row, 'user_message', 'a different bug explanation')
    assert 'a different bug explanation (fixed in 43)' in p.advice
    assert BUG_TEXT not in p.advice


@pytest.mark.parametrize('module_name', ['__main__', 'silk.examples.user_example'])
def test_example_warning_points_at_example(monkeypatch, tmp_path, module_name):
    from types import SimpleNamespace
    from silk.simulation import runner, _ngspice_compat as compat
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: '42')
    monkeypatch.setattr(compat, '_advised_binaries', set())
    monkeypatch.setattr(compat.shutil, 'which', lambda _: '/fake/ngspice')
    def launch(argv, **kwargs):
        Path(argv[argv.index('-r') + 1]).write_text('stub')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, 'run', launch)
    (tmp_path / 'deck.cir').write_text('Warning fixture\n.end\n')
    run = runner.NgspiceRunner(str(tmp_path / 'deck.cir'), str(tmp_path))
    # Add a framework wrapper: attribution must traverse more than runner.run.
    wrapper = {'__name__': 'silk.simulation.backend', 'run': run}
    exec(compile('def simulate():\n    run.run()\n',
                 str(Path(runner.__file__).with_name('backend.py')),
                 'exec'), wrapper)
    filename = str(Path(runner.__file__).parents[1]
                   / 'examples' / 'user_example.py')
    with pytest.warns(compat.NgspiceVersionWarning) as caught:
        exec(compile('# user example\nsimulate()\n', filename, 'exec'),
             {'__name__': module_name, 'simulate': wrapper['simulate']})
    assert len(caught) == 1
    assert caught[0].filename == filename
    assert caught[0].lineno == 2


@pytest.mark.parametrize('target', ['45.2', '42', '47+'])
def test_generation_target_restores_default(tmp_path, monkeypatch, target):
    from tests.test_run_files import _module
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: None)
    mod = _module(tmp_path)
    mod.omit_device_internal_nodes = target != '42'
    mod.generate_netlist(ngspice_version=target)
    text = Path(mod._run_files.netlist_path).read_text()
    if target != '42':
        assert f'ngspice {target}' in text
        assert 'save nointernals' in text
    assert policy(None).version is None
    from silk.simulation._ngspice_compat import NgspiceCompatibility
    assert NgspiceCompatibility().version is None
    mod.omit_device_internal_nodes = False
    mod.generate_netlist()


@pytest.mark.parametrize('target', ['', 'ngspice-45', ' 42', '42x', 42, 45.2, [], True])
def test_generation_target_invalid(tmp_path, target):
    from tests.test_run_files import _module
    with pytest.raises(ValueError, match='ngspice_version'):
        _module(tmp_path).generate_netlist(ngspice_version=target)


@pytest.mark.parametrize('simulator', ['xyce', 'behavioral'])
def test_generation_target_non_ngspice(tmp_path, simulator):
    from tests.test_run_files import _module
    with pytest.raises(ValueError, match='ngspice'):
        _module(tmp_path, simulator).generate_netlist(ngspice_version='45.2')


def test_generation_target_completion_and_default_bytes(tmp_path, monkeypatch):
    from tests.test_run_files import _module
    from silk.simulation import AC
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: '42')
    mod = _module(tmp_path, analyses=[AC('dec', 10, 100, 150)])
    mod.set_output_options(mode='overwrite')
    mod.generate_netlist()
    original = Path(mod._run_files.netlist_path).read_bytes()
    assert mod._run_files.completion.requests[0].end == 150
    mod.generate_netlist(ngspice_version='45.2')
    assert mod._run_files.completion.requests[0].end == pytest.approx(100 * 10**0.1)
    mod.generate_netlist()
    assert mod._run_files.completion.requests[0].end == 150
    assert Path(mod._run_files.netlist_path).read_bytes() == original


def test_target_am_children_normalized_and_failure_restoration(tmp_path, monkeypatch):
    from silk import Module, sources
    from silk.sources.waveforms import AM
    from silk.exceptions import CircuitError
    from tests.test_run_files import _module
    from silk.circuit.test_execution import normalized_deck_digest
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: '42')
    mod = _module(tmp_path)
    children = []
    for name in ('one', 'two'):
        child = Module(None, str(tmp_path), name + '.cir', None, module_name='am_child')
        child.define_module_ports({'out': 'out'})
        child.add_source(sources.VoltageSource('wave', positive='out', negative='0',
                                             am=AM(2, 0.5, 1000, 1e6, phases=30)))
        mod.add_module(child, name, nets={'out': name})
        children.append(child)
    mod.generate_netlist(ngspice_version='45.2')
    deck = Path(mod._run_files.netlist_path).read_text()
    assert 'AM(0 1.0 2 1000 1000000.0 0 30 0)' in deck
    targeted_digest = mod._prepared_deck_record['normalized_deck_sha256']
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: '45.2')
    assert targeted_digest == normalized_deck_digest(mod, 'tt')[0]
    assert 'AM(0 1.0' in children[0].generate_netlist(ngspice_version='45.2')
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: '42')
    with pytest.raises(CircuitError, match='phases'):
        mod.generate_netlist()
    with pytest.raises(CircuitError, match='requires an identifiable'):
        mod.generate_netlist(ngspice_version='41')
    for child in children:
        child.sig_sources_dict['wave'].am.phases = 0
    mod.generate_netlist()
    assert 'AM(2 0.5' in Path(mod._run_files.netlist_path).read_text()


@pytest.mark.parametrize('kind,target,allowed', [
    ('sffm', '42', True), ('sffm', '45.2', False),
    ('pwl', '47+', True), ('pwl', '45.2', False),
    ('filter', '45.2', True), ('filter', '42', False),
])
def test_target_policy_refusals(tmp_path, monkeypatch, kind, target, allowed):
    from silk import sources
    from silk.sources.waveforms import SFFM, PWL
    from silk.exceptions import NotSupportedError
    from tests.test_run_files import _module
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: None)
    mod = _module(tmp_path)
    if kind == 'sffm':
        mod.add_source(sources.VoltageSource('wave', positive='out', negative='0',
                                             sffm=SFFM(0, 1, 1e6, 2, 1e3)))
    elif kind == 'pwl':
        mod.add_source(sources.CurrentSource('wave', positive='out', negative='0',
                                             pwl=PWL([(0, 0), (1, 1)], r=0, td=1)))
    else:
        mod.omit_device_internal_nodes = True
    if allowed:
        mod.generate_netlist(ngspice_version=target)
    else:
        with pytest.raises(NotSupportedError):
            mod.generate_netlist(ngspice_version=target)
    from silk.simulation._ngspice_compat import NgspiceCompatibility
    assert NgspiceCompatibility().version is None
