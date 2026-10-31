"""Catalogs must reject names that cannot be used by Python callers."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from silk import export_schema, load_pdk
from silk.elements import ElementLibrary
from silk.exceptions import CircuitError, PDKError
from silk.pdk import BasePDK, available
from silk.sources import schema as source_schema

BAD_NAMES = ['bad-name', 'bad.name', 'bad+name', 'class', '1name']


@pytest.mark.parametrize('name', BAD_NAMES + ['', 1, None])
def test_public_identifier_validator_refuses_invalid_names(name):
    from silk.utils import require_python_identifier
    with pytest.raises(ValueError) as exc:
        require_python_identifier(name, 'device type')
    assert str(exc.value) == (
        f'device type {name!r} must be a valid Python identifier and '
        'must not be a Python keyword')


@pytest.mark.parametrize('name', ['valid_name', '_private', 'V2', 'café', 'match'])
def test_public_identifier_validator_preserves_valid_names(name):
    from silk.utils import require_python_identifier
    assert require_python_identifier(name, 'device type') == name


def _device():
    return {'PDKDevice': 'model-with.dots', 'pinList': {'p': 'positive'},
            'paramList': {'w': {'minval': 1, 'maxval': 10}}}


@pytest.mark.parametrize('bad', BAD_NAMES)
@pytest.mark.parametrize('field', ['device', 'pinList', 'pinAliases', 'paramList',
                                   'paramAliases', 'paramConstraints'])
@pytest.mark.parametrize('loader,error', [(BasePDK, PDKError), (ElementLibrary, CircuitError)])
def test_catalog_load_refuses_invalid_keys(tmp_path, bad, field, loader, error):
    spec = _device()
    name = bad if field == 'device' else 'demo'
    if field == 'paramConstraints':
        spec[field] = [{'name': bad, 'default': 1}]
    elif field != 'device':
        spec[field] = {bad: {'minval': 1, 'maxval': 10} if field == 'paramList' else 'p'}
    path = tmp_path / 'devices.yaml'
    path.write_text(yaml.safe_dump({name: spec}))
    with pytest.raises(error) as exc:
        if loader is BasePDK:
            loader(str(path), library_location="")
        else:
            loader(str(path))
    message = str(exc.value)
    assert str(path) in message
    assert repr(name) in message
    assert repr(bad) in message
    assert field in message
    assert 'Python identifier' in message


@pytest.mark.parametrize('bad', BAD_NAMES)
def test_ideal_pin_list_sequence_refuses_invalid_name(tmp_path, bad):
    path = tmp_path / 'analog_devices.yaml'
    path.write_text(yaml.safe_dump({'demo': {'pinList': [bad], 'paramConstraints': []}}))
    with pytest.raises(CircuitError, match='pinList'):
        ElementLibrary(path)


SOURCE_FIELDS = ['source', 'pins', 'scalar_params', 'references', 'expressions',
                 'quantity', 'alias', 'waveform_kind', 'waveform_class',
                 'param_units', 'param_kinds']


@pytest.mark.parametrize('bad', BAD_NAMES)
@pytest.mark.parametrize('field', SOURCE_FIELDS)
def test_source_metadata_refuses_python_names(tmp_path, monkeypatch, bad, field):
    meta = {'sources': {'Demo': {}}, 'waveforms': {}}
    if field == 'source':
        meta['sources'] = {bad: {}}
    elif field in ('quantity', 'alias'):
        meta['sources']['Demo'][field] = bad
    elif field in ('pins', 'scalar_params', 'references', 'expressions'):
        meta['sources']['Demo'][field] = [bad]
    elif field == 'waveform_kind':
        meta['waveforms'] = {bad: 'Pulse'}
    elif field == 'waveform_class':
        meta['waveforms'] = {'pulse': bad}
    else:
        meta[field] = {bad: 'unit'}
    path = tmp_path / 'source_metadata.yaml'
    path.write_text(yaml.safe_dump(meta))
    monkeypatch.setattr(source_schema, '_META_PATH', path)
    with pytest.raises(CircuitError) as exc:
        source_schema._load_metadata()
    assert str(path) in str(exc.value)
    assert repr(bad) in str(exc.value)
    assert 'Python identifier' in str(exc.value)
    if field in ('pins', 'scalar_params', 'references', 'expressions', 'quantity', 'alias'):
        assert 'Demo' in str(exc.value)


def test_shipped_catalogs_load_and_export():
    for name in available():
        pdk = load_pdk(name)
        schema = export_schema(pdk, include_metadata=False)
        assert set(schema) == set(pdk.model_names)
        assert schema
    assert 'resistor' in export_schema(ElementLibrary())
    assert 'VoltageSource' in source_schema.export_source_schema()


def _cli(*args):
    root = Path(__file__).resolve().parents[1]
    return subprocess.run([sys.executable, '-m', *args], cwd=root,
                          env={**os.environ, 'PYTHONPATH': str(root)},
                          text=True, capture_output=True)


@pytest.mark.parametrize('name', ['sky130', 'skywater130'])
def test_schema_cli_registered_pdk(name):
    result = _cli('silk.elements.schema', '--pdk', name)
    assert result.returncode == 0, result.stderr
    schema = json.loads(result.stdout)
    assert schema['nmos1p8']['pdk_model'] == 'sky130_fd_pr__nfet_01v8'


def test_schema_cli_yaml_path(tmp_path):
    path = tmp_path / 'devices.yaml'
    path.write_text(yaml.safe_dump({'demo': _device()}))
    result = _cli('silk.elements.schema', '--pdk', str(path))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['demo']['pdk_model'] == 'model-with.dots'


def test_schema_cli_help():
    result = _cli('silk.elements.schema', '--help')
    assert result.returncode == 0
    assert 'registered PDK' in result.stdout
    assert 'pynetlist/' not in result.stdout


@pytest.mark.parametrize('registered', [True, False])
def test_validate_cli_reports_catalog_error(tmp_path, monkeypatch, capsys, registered):
    from silk.pdk import registry
    from silk.pdk.validate import main
    path = tmp_path / 'devices.yaml'
    path.write_text(yaml.safe_dump({'bad-name': _device()}))
    if registered:
        (tmp_path / 'demo.pdk.yaml').write_text(yaml.safe_dump({
            'name': 'demo', 'device_details': 'devices.yaml',
            'library': {'lib_path': 'absent.spice', 'default_corner': 'tt',
                        'corners': {'tt': {}}}}))
        monkeypatch.setattr(registry, '_DATA_DIR', str(tmp_path))
        monkeypatch.setattr(registry, '_CACHE', None)
        args = ['--pdk', 'demo']
    else:
        args = ['--yaml', str(path)]
    assert main(args) == 1
    output = capsys.readouterr().out
    assert str(path) in output
    assert 'bad-name' in output
    assert 'Python identifier' in output


@pytest.mark.parametrize('bad', BAD_NAMES)
def test_ideal_source_class_refuses_invalid_python_name(tmp_path, bad):
    path = tmp_path / 'analog_devices.yaml'
    path.write_text(yaml.safe_dump({'demo': {
        'pinList': ['p', 'n'], 'paramConstraints': [], 'source_class': bad}}))
    with pytest.raises(CircuitError) as exc:
        ElementLibrary(path)
    assert str(path) in str(exc.value)
    assert 'demo' in str(exc.value)
    assert 'source_class' in str(exc.value)
    assert repr(bad) in str(exc.value)


@pytest.mark.parametrize('field,part', [('scalar_params', 'sync'), ('waveforms', 'nd')])
def test_source_metadata_refuses_keyword_factory_name(tmp_path, monkeypatch, field, part):
    meta = {'sources': {'VoltageSource': {
        'alias': 'a', 'quantity': 'voltage', 'scalar_params': ['dc']}},
        'waveforms': {'pulse': 'Pulse'}}
    if field == 'scalar_params':
        meta['sources']['VoltageSource'][field] = [part]
    else:
        meta[field] = {part: 'Pulse'}
    path = tmp_path / 'source_metadata.yaml'
    path.write_text(yaml.safe_dump(meta))
    monkeypatch.setattr(source_schema, '_META_PATH', path)
    with pytest.raises(CircuitError) as exc:
        source_schema._load_metadata()
    assert str(path) in str(exc.value)
    assert 'VoltageSource' in str(exc.value)
    assert repr('a' + part) in str(exc.value)
    assert 'Python keyword' in str(exc.value)


def _bad_catalog():
    return {
        'bad-device': {
            'PDKDevice': 'model',
            'pinList': {'bad.pin': 'pin', '1pin': 'another pin'},
            'pinAliases': {'bad+alias': 'bad.pin'},
            'paramList': {'class': {'minval': 1, 'maxval': 10}},
            'paramAliases': {'bad-param': 'class'},
        },
        'second': {'PDKDevice': 'model', 'pinList': {'p': 'pin'},
                   'paramList': {'2width': {'minval': 1, 'maxval': 10}}},
    }


def _assert_all_bad_names(output, path):
    assert str(path) in output
    for field, key in [('device key', 'bad-device'), ('pinList', 'bad.pin'),
                       ('pinList', '1pin'), ('pinAliases', 'bad+alias'),
                       ('paramList', 'class'), ('paramAliases', 'bad-param'),
                       ('paramList', '2width')]:
        assert any(field in line and repr(key) in line for line in output.splitlines())
    assert output.count('must be a valid Python identifier') == 7


def test_validate_module_lists_every_bad_key_before_library_resolution(tmp_path):
    path = tmp_path / 'devices.yaml'
    path.write_text(yaml.safe_dump(_bad_catalog(), sort_keys=False))
    result = _cli('silk.pdk.validate', '--yaml', str(path), '--lib',
                  str(tmp_path / 'missing.spice'))
    assert result.returncode == 1, result.stderr
    _assert_all_bad_names(result.stdout, path)
    assert 'Traceback' not in result.stderr
    with pytest.raises(PDKError) as exc:
        BasePDK(str(path), library_location='')
    assert str(exc.value).count('must be a valid Python identifier') == 1
    assert 'bad-device' in str(exc.value)
    assert '2width' not in str(exc.value)


def test_validate_registered_catalog_lists_every_bad_key(tmp_path, monkeypatch, capsys):
    from silk.pdk import registry
    from silk.pdk.validate import main
    path = tmp_path / 'devices.yaml'
    path.write_text(yaml.safe_dump(_bad_catalog()))
    (tmp_path / 'demo.pdk.yaml').write_text(yaml.safe_dump({
        'name': 'demo', 'device_details': 'devices.yaml',
        'library': {'lib_path': 'absent.spice', 'default_corner': 'tt',
                    'corners': {'tt': {}}}}))
    monkeypatch.setattr(registry, '_DATA_DIR', str(tmp_path))
    monkeypatch.setattr(registry, '_CACHE', None)
    assert main(['--pdk', 'demo']) == 1
    _assert_all_bad_names(capsys.readouterr().out, path)


@pytest.mark.parametrize('unresolved', ['pdk', 'catalog', 'library'])
def test_validate_unresolvable_inputs_remain_exit_two(tmp_path, unresolved):
    path = tmp_path / 'devices.yaml'
    if unresolved == 'pdk':
        args = ['--pdk', 'unknown_task550_pdk']
    else:
        if unresolved == 'library':
            path.write_text(yaml.safe_dump({'demo': _device()}))
        args = ['--yaml', str(path), '--lib', str(tmp_path / 'missing.spice')]
    result = _cli('silk.pdk.validate', *args)
    assert result.returncode == 2, result.stderr
    assert 'validate:' in result.stdout
    assert 'Traceback' not in result.stderr


def test_validate_clean_catalog_and_library_remain_exit_zero(tmp_path):
    path = tmp_path / 'devices.yaml'
    path.write_text(yaml.safe_dump({'demo': _device()}))
    library = tmp_path / 'models.spice'
    library.write_text('.subckt model-with.dots p\n.ends model-with.dots\n')
    result = _cli('silk.pdk.validate', '--yaml', str(path), '--lib', str(library))
    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 devices, no issues' in result.stdout
