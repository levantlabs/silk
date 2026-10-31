"""silk.settings — the user settings file for tool and PDK locations (TASK-452).

Every test here points HOME, XDG_CONFIG_HOME and SILK_SETTINGS at tmp_path
(the ``isolated`` fixture), so none reads or writes the real
``~/.config/silk/silk.env`` on this shared machine.
"""
import os
import shutil
import stat

import pytest

from silk import doctor, settings
from silk.exceptions import ConfigurationError, SettingsError

KEYS = list(settings.KNOWN_SETTINGS)


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """No settings in the environment; HOME and XDG_CONFIG_HOME in tmp_path;
    SILK_SETTINGS unset (so the location rules apply). Returns the fake HOME."""
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.delenv('XDG_CONFIG_HOME', raising=False)
    monkeypatch.delenv('SILK_SETTINGS', raising=False)
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    return home


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def settings_file(isolated, monkeypatch, tmp_path):
    """Write *text* to a tmp settings file named by SILK_SETTINGS."""
    def write(text):
        path = _write(tmp_path / 'cfg' / 'silk.env', text)
        monkeypatch.setenv('SILK_SETTINGS', str(path))
        return path
    return write


# ---------------------------------------------------------------- location

class TestLocation:
    def test_default_is_home_config(self, isolated):
        assert settings.settings_path() == str(
            isolated / '.config' / 'silk' / 'silk.env')

    def test_home_config_file_is_read(self, isolated):
        _write(isolated / '.config' / 'silk' / 'silk.env', 'PDK_ROOT=/from/home\n')
        assert settings.lookup('PDK_ROOT') == settings.Setting(
            'PDK_ROOT', '/from/home', 'file')

    def test_xdg_config_home_wins_over_home(self, isolated, monkeypatch, tmp_path):
        _write(isolated / '.config' / 'silk' / 'silk.env', 'PDK_ROOT=/from/home\n')
        xdg = tmp_path / 'xdg'
        _write(xdg / 'silk' / 'silk.env', 'PDK_ROOT=/from/xdg\n')
        monkeypatch.setenv('XDG_CONFIG_HOME', str(xdg))
        assert settings.settings_path() == str(xdg / 'silk' / 'silk.env')
        assert settings.get('PDK_ROOT') == '/from/xdg'

    def test_silk_settings_wins_over_xdg(self, isolated, monkeypatch, tmp_path):
        xdg = tmp_path / 'xdg'
        _write(xdg / 'silk' / 'silk.env', 'PDK_ROOT=/from/xdg\n')
        monkeypatch.setenv('XDG_CONFIG_HOME', str(xdg))
        named = _write(tmp_path / 'elsewhere.env', 'PDK_ROOT=/from/named\n')
        monkeypatch.setenv('SILK_SETTINGS', str(named))
        assert settings.settings_path() == str(named)
        assert settings.get('PDK_ROOT') == '/from/named'

    @pytest.mark.parametrize('value', ['', os.devnull])
    def test_empty_or_devnull_silk_settings_disables_the_file(
            self, isolated, monkeypatch, value):
        _write(isolated / '.config' / 'silk' / 'silk.env', 'PDK_ROOT=/from/home\n')
        monkeypatch.setenv('SILK_SETTINGS', value)
        assert settings.settings_path() is None
        assert settings.settings_file_status() == (None, 'disabled')
        assert settings.lookup('PDK_ROOT').source == 'default'

    def test_missing_file_named_by_silk_settings_raises(
            self, isolated, monkeypatch, tmp_path):
        missing = tmp_path / 'nope.env'
        monkeypatch.setenv('SILK_SETTINGS', str(missing))
        with pytest.raises(SettingsError, match='does not exist'):
            settings.get('PDK_ROOT')

    def test_missing_default_file_is_not_an_error(self, isolated):
        assert settings.settings_file_status()[1] == 'not found'
        assert settings.lookup('PDK_ROOT').source == 'default'


# -------------------------------------------------------------- precedence

_FILE_VALUE = {k: f'/file/{k.lower()}' for k in KEYS}


class TestPrecedence:
    @pytest.mark.parametrize('key', KEYS)
    def test_env_beats_file(self, settings_file, monkeypatch, key):
        settings_file(f'{key}={_FILE_VALUE[key]}\n')
        monkeypatch.setenv(key, '/from/env')
        assert settings.lookup(key) == settings.Setting(key, '/from/env', 'env')

    @pytest.mark.parametrize('key', KEYS)
    def test_file_beats_default(self, settings_file, key):
        settings_file(f'{key}={_FILE_VALUE[key]}\n')
        assert settings.lookup(key) == settings.Setting(
            key, _FILE_VALUE[key], 'file')

    @pytest.mark.parametrize('key', KEYS)
    def test_empty_env_falls_through_to_file(self, settings_file, monkeypatch, key):
        settings_file(f'{key}={_FILE_VALUE[key]}\n')
        monkeypatch.setenv(key, '')
        assert settings.lookup(key).source == 'file'

    def test_defaults_without_a_file(self, isolated, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        assert settings.get('PDK_ROOT') == str(isolated / 'pdk')
        assert settings.get('SILK_NGSPICE_BIN') == 'ngspice'
        assert settings.get('SILK_XYCE_BIN') == (shutil.which('Xyce') or 'Xyce')
        assert settings.get('SILK_EXAMPLES_OUT') == str(tmp_path / 'silk-output')
        assert settings.get('SILK_FIREFOX') is None
        assert settings.get('SILK_CHROME') is None
        assert all(s.source == 'default' for s in settings.all_settings())

    def test_empty_file_value_counts_as_unset(self, settings_file):
        settings_file('SILK_NGSPICE_BIN=\n')
        assert settings.lookup('SILK_NGSPICE_BIN').source == 'default'

    def test_unknown_name_is_a_key_error(self, isolated):
        with pytest.raises(KeyError, match='not a silk setting'):
            settings.get('PDK_RUT')


# ------------------------------------------------------------------ format

class TestFormat:
    def test_comments_blank_lines_and_export(self, settings_file):
        settings_file('# a comment\n'
                      '\n'
                      '   # indented comment\n'
                      'export PDK_ROOT=/pdk\n'
                      'export\tSILK_XYCE_BIN = /opt/Xyce\n')
        assert settings.get('PDK_ROOT') == '/pdk'
        assert settings.get('SILK_XYCE_BIN') == '/opt/Xyce'

    def test_inline_comment_after_unquoted_value(self, settings_file):
        settings_file('PDK_ROOT=/pdk   # where the PDKs live\n'
                      'SILK_CHROME=/odd#name\n')
        assert settings.get('PDK_ROOT') == '/pdk'
        assert settings.get('SILK_CHROME') == '/odd#name'

    def test_double_and_single_quotes(self, settings_file):
        settings_file('SILK_EXAMPLES_OUT="/runs with spaces"  # ok\n'
                      "SILK_FIREFOX='/a # not a comment'\n")
        assert settings.get('SILK_EXAMPLES_OUT') == '/runs with spaces'
        assert settings.get('SILK_FIREFOX') == '/a # not a comment'

    def test_tilde_is_expanded(self, settings_file, isolated):
        settings_file('PDK_ROOT=~/pdk\nSILK_XYCE_BIN="~/xyce/bin/Xyce"\n')
        assert settings.get('PDK_ROOT') == str(isolated / 'pdk')
        assert settings.get('SILK_XYCE_BIN') == str(isolated / 'xyce/bin/Xyce')

    def test_file_edit_is_picked_up(self, settings_file):
        path = settings_file('PDK_ROOT=/one\n')
        assert settings.get('PDK_ROOT') == '/one'
        path.write_text('PDK_ROOT=/two-longer\n')
        assert settings.get('PDK_ROOT') == '/two-longer'

    @pytest.mark.parametrize('text, line, words', [
        ('PDK_ROOT=/pdk\nthis is not a setting\n', 2, 'expected KEY=value'),
        ('PDK_ROOT="/pdk\n', 1, 'unterminated'),
        ("# c\nPDK_ROOT='/pdk' trailing\n", 2, 'after the closing quote'),
        ('PDK_ROOT=/a\nPDK_ROOT=/b\n', 2, 'already set on line 1'),
        ('1BAD=x\n', 1, 'expected KEY=value'),
    ])
    def test_malformed_line_names_file_and_line(self, settings_file, text,
                                                 line, words):
        path = settings_file(text)
        with pytest.raises(SettingsError) as exc:
            settings.get('PDK_ROOT')
        assert f'{path}:{line}:' in str(exc.value)
        assert words in str(exc.value)

    def test_settings_error_is_a_configuration_error(self):
        assert issubclass(SettingsError, ConfigurationError)

    def test_unknown_keys_are_kept(self, settings_file):
        settings_file('PDK_ROOT=/pdk\nSILK_NGSPCE_BIN=/typo\n')
        assert settings.unknown_keys() == [('SILK_NGSPCE_BIN', 2)]


# ------------------------------------------------------------------ doctor

class TestDoctor:
    def test_reports_path_and_every_setting_with_source(
            self, settings_file, monkeypatch):
        path = settings_file('PDK_ROOT=/file/pdk\n')
        monkeypatch.setenv('SILK_XYCE_BIN', '/env/Xyce')
        rows = {c.name: c for c in doctor.check_settings()}
        assert rows['settings file'].detail == f'{path} (found)'
        assert rows['setting PDK_ROOT'].detail == '/file/pdk  (source: file)'
        assert rows['setting SILK_XYCE_BIN'].detail == '/env/Xyce  (source: env)'
        assert rows['setting SILK_NGSPICE_BIN'].detail == \
            'ngspice  (source: default)'
        assert all(f'setting {k}' in rows for k in KEYS)

    def test_not_found_and_disabled_states(self, isolated, monkeypatch):
        head = doctor.check_settings()[0]
        assert head.status == doctor.OK
        assert head.detail == (f"{isolated / '.config' / 'silk' / 'silk.env'} "
                               f"(not found)")
        monkeypatch.setenv('SILK_SETTINGS', '')
        assert 'disabled by SILK_SETTINGS' in doctor.check_settings()[0].detail

    def test_unknown_key_is_a_warning(self, settings_file):
        path = settings_file('PDK_ROOT=/pdk\nPDKROOT=/typo\n')
        warns = [c for c in doctor.check_settings() if c.status == doctor.WARN]
        assert len(warns) == 1
        assert warns[0].name == 'setting PDKROOT'
        assert f'{path}:2' in warns[0].detail
        assert 'PDK_ROOT' in warns[0].fix

    def test_malformed_file_fails_and_names_the_line(self, settings_file):
        path = settings_file('garbage\n')
        head = doctor.check_settings()[0]
        assert head.status == doctor.FAIL
        assert f'{path}:1:' in head.detail
        # Every other check that needs a setting fails rather than crashing.
        rows = doctor.run_checks()
        assert any(c.name == 'ngspice' and c.status == doctor.FAIL
                   and 'cannot resolve settings' in c.detail for c in rows)

    def test_location_fix_hints_name_the_settings_file(self, isolated):
        path = str(isolated / '.config' / 'silk' / 'silk.env')
        fix = doctor.check_pdk_root()[0].fix
        assert 'export PDK_ROOT=' in fix and path in fix
        assert path in doctor._lib_fix()
        assert path in doctor._osdi_version_fix()

    def test_ngspice_and_xyce_hints_name_the_settings_file(
            self, isolated, monkeypatch, tmp_path):
        path = str(isolated / '.config' / 'silk' / 'silk.env')
        monkeypatch.setenv('PATH', str(tmp_path))       # nothing on PATH
        ng = doctor.check_ngspice()[0]
        assert ng.status == doctor.FAIL and path in ng.fix
        xy = doctor.check_xyce()[0]
        assert xy.status == doctor.SKIP and path in xy.detail

    def test_pdk_root_from_file_passes(self, settings_file, tmp_path):
        root = tmp_path / 'pdk'
        root.mkdir()
        settings_file(f'PDK_ROOT={root}\n')
        check = doctor.check_pdk_root()[0]
        assert check.status == doctor.OK
        assert check.detail == f'{root} (settings file)'

    def test_json_carries_the_settings_rows(self, settings_file, capsys):
        import json
        path = settings_file('PDK_ROOT=/file/pdk\nNOPE=1\n')
        doctor.main(['--json'])
        rows = {r['name']: r for r in json.loads(capsys.readouterr().out)}
        assert rows['settings file']['detail'] == f'{path} (found)'
        assert rows['setting PDK_ROOT']['detail'] == '/file/pdk  (source: file)'
        assert rows['setting NOPE']['status'] == 'warn'


# --------------------------------------------------------- read sites

def _fake_exe(path):
    path.write_text('#!/bin/sh\nexit 0\n')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class TestReadSites:
    """Each place silk reads a location resolves through the settings file."""

    def test_ngspice_runner(self, settings_file, tmp_path):
        from silk.simulation.runner import ngspice_executable
        exe = _fake_exe(tmp_path / 'my-ngspice')
        settings_file(f'SILK_NGSPICE_BIN={exe}\n')
        assert ngspice_executable() == str(exe)

    def test_xyce_runner(self, settings_file):
        from silk.simulation.xyce_runner import _xyce_executable
        settings_file('SILK_XYCE_BIN=/opt/xyce/bin/Xyce\n')
        assert _xyce_executable() == '/opt/xyce/bin/Xyce'

    def test_load_pdk_root(self, settings_file, tmp_path):
        from silk.pdk import load_pdk
        settings_file(f'PDK_ROOT={tmp_path}/pdkroot\n')
        lib = load_pdk('sky130').library_location
        assert lib.startswith(f'{tmp_path}/pdkroot/')

    def test_validate_lib_root_and_default_lib(self, settings_file, tmp_path):
        from silk.pdk import validate
        root = tmp_path / 'pdkroot'
        root.mkdir()
        settings_file(f'PDK_ROOT={root}\n')
        assert validate._lib_root(object(), None) == str(root)
        assert validate._default_lib().startswith(str(root))

    def test_examples_output_dir(self, settings_file, tmp_path):
        from silk.examples import example_output_dir
        settings_file(f'SILK_EXAMPLES_OUT={tmp_path}/runs\n')
        assert example_output_dir('x', create=False) == f'{tmp_path}/runs/x'

    def test_provenance_names_the_setting(self, settings_file, tmp_path):
        """Vocabulary unchanged: a file value is still 'SILK_NGSPICE_BIN'."""
        from silk.simulation.provenance import _pdk_root, launch_record
        exe = _fake_exe(tmp_path / 'ngspice')
        settings_file(f'SILK_NGSPICE_BIN={exe}\nPDK_ROOT={tmp_path}/r\n')
        rec = launch_record('ngspice')
        assert rec['simulator_binary_source'] == 'SILK_NGSPICE_BIN'
        assert _pdk_root() == f'{tmp_path}/r'

    def test_provenance_default_is_path(self, isolated):
        from silk.simulation.provenance import launch_record
        assert launch_record('ngspice')['simulator_binary_source'] == 'PATH'

    def test_test_probes(self, settings_file, tmp_path):
        from tests import _env
        ff = _fake_exe(tmp_path / 'firefox')
        settings_file(f'SILK_FIREFOX={ff}\nSILK_CHROME=/c\n')
        assert _env.browser_executable() == str(ff)
        assert _env.chrome_setting() == '/c'
