"""Customer region policy reaches tables, HTML, and preflight diagnostics."""
from types import SimpleNamespace

import pytest
import yaml

from silk.circuit.op_table import RegionThresholds
from silk.circuit.spec_report import SpecReport
from silk.simulation.provenance import RunProvenance


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    monkeypatch.setenv('SILK_SETTINGS', '')
    monkeypatch.delenv('SILK_REGION_THRESHOLDS', raising=False)


def policy():
    return dict(min_current_a=1e-9, weak_sat_margin_v=0.05,
                subthreshold_margin_kt_q=3.0)


def select_file(tmp_path, monkeypatch, entries):
    path = tmp_path / 'customer.yaml'
    path.write_text(yaml.safe_dump({'version': 1, 'pdks': entries}))
    monkeypatch.setenv('SILK_REGION_THRESHOLDS', str(path))
    return path


def report(pdk=None, temperature=None):
    values = dict(vgs=0.35, vth=0.4, vds=0.9, vdsat=0.2, id=1e-7)
    return SpecReport('op', 'dut', {}, op_devices=[{
        'path': 'xm1', 'name': 'm1', 'model': 'm', 'polarity': 'n',
        'w_um': 1.0, 'l_um': 0.5, 'm': 1.0, 'declared_params': tuple(values),
        'op_values': values,
    }], provenance=RunProvenance(pdk_name=pdk) if pdk else None,
        operating_conditions=temperature)


def test_user_policy_selected_once_per_pdk(tmp_path, monkeypatch):
    strict = policy()
    strict['min_current_a'] = 1e-6
    select_file(tmp_path, monkeypatch, {'sky130': strict})
    assert report('sky130').op_table()[0].region == 'cutoff'
    assert report('gf180mcu').op_table()[0].region == 'subthreshold'
    assert report('sky130').op_table(RegionThresholds())[0].region == 'subthreshold'


def test_default_entry_is_overridable(tmp_path, monkeypatch):
    strict = policy()
    strict['min_current_a'] = 1e-6
    select_file(tmp_path, monkeypatch, {'default': strict})
    assert report().op_table()[0].region == 'cutoff'
    assert report('sky130').op_table()[0].region == 'subthreshold'


def test_settings_file_and_environment_precedence(tmp_path, monkeypatch):
    strict = policy()
    strict['min_current_a'] = 1e-6
    path = select_file(tmp_path, monkeypatch, {'default': strict})
    settings = tmp_path / 'silk.env'
    settings.write_text(f'SILK_REGION_THRESHOLDS={path}\n')
    monkeypatch.setenv('SILK_SETTINGS', str(settings))
    monkeypatch.delenv('SILK_REGION_THRESHOLDS')
    assert report().op_table()[0].region == 'cutoff'
    other = tmp_path / 'other'
    other.mkdir()
    select_file(other, monkeypatch, {'default': policy()})
    assert report().op_table()[0].region == 'subthreshold'


def test_thermal_and_fixed_margins(tmp_path, monkeypatch):
    thermal = policy()
    thermal['subthreshold_margin_kt_q'] = 2.0
    select_file(tmp_path, monkeypatch, {'default': thermal})
    hot = report(temperature=SimpleNamespace(temperature_c=125.0)).op_table()
    assert hot.region_thresholds.subthreshold_margin_v == pytest.approx(0.068619824766)
    no_temp = report(temperature=SimpleNamespace(temperature_kind='none'))
    with pytest.raises(ValueError, match='region_thresholds='):
        no_temp.op_table()
    fixed = dict(min_current_a=1e-9, weak_sat_margin_v=0.05,
                 subthreshold_margin_v=0.01)
    select_file(tmp_path, monkeypatch, {'default': fixed})
    assert no_temp.op_table()[0].region == 'cutoff'


def test_registered_pdks_have_explicit_shipped_sets():
    from silk.circuit.region_thresholds import load_thresholds_file, DEFAULT_THRESHOLDS_PATH
    from silk.pdk import registry
    sets = load_thresholds_file(DEFAULT_THRESHOLDS_PATH)
    assert set(registry.available()) <= sets.keys()
    assert 'default' in sets
    for name in [None, *registry.available()]:
        table = report(name).op_table()
        assert table[0].region == 'subthreshold'
        assert table.region_thresholds.min_current_a == 1e-9
        assert table.region_thresholds.weak_sat_margin_v == 0.05
        assert table.region_thresholds.subthreshold_margin_v == pytest.approx(0.077594777359)


def test_alias_uses_canonical_policy(tmp_path, monkeypatch):
    from silk.pdk import registry
    strict = policy()
    strict['min_current_a'] = 1e-6
    select_file(tmp_path, monkeypatch, {'sky130': strict})
    for alias in registry.manifest('sky130')['aliases']:
        assert report(alias).op_table()[0].region == 'cutoff'


def test_unknown_pdk_requires_its_own_set(tmp_path, monkeypatch):
    from silk.circuit.region_thresholds import RegionThresholdsError
    with pytest.raises(RegionThresholdsError, match='custom'):
        report('custom').op_table()
    select_file(tmp_path, monkeypatch, {'custom': policy()})
    assert report('custom').op_table()[0].region == 'subthreshold'


@pytest.mark.parametrize('text,key', [
    ('[]', '<root>'),
    ('version: 2\npdks: {}', 'version'),
    ('version: true\npdks: {}', 'version'),
    ('version: 1\npdks: []', 'pdks'),
    ('version: 1\npdks: {}\nextra: 1', 'extra'),
    ('version: 1\npdks: {sky130: {}}', 'min_current_a'),
    ('version: 1\npdks: {sky130: null}', 'sky130'),
    ('version: 1\nversion: 1\npdks: {}', 'version'),
    ('version: [', '<root>'),
])
def test_malformed_file_names_file_and_key(tmp_path, monkeypatch, text, key):
    from silk.circuit.region_thresholds import RegionThresholdsError
    path = tmp_path / 'bad.yaml'
    path.write_text(text)
    monkeypatch.setenv('SILK_REGION_THRESHOLDS', str(path))
    with pytest.raises(RegionThresholdsError) as caught:
        report('sky130').op_table()
    assert str(path) in str(caught.value)
    assert key in str(caught.value)


@pytest.mark.parametrize('key,value', [
    ('min_current_a', -1), ('min_current_a', True), ('min_current_a', '1n'),
    ('weak_sat_margin_v', float('inf')), ('subthreshold_margin_kt_q', float('nan')),
    ('typo', 1), ('subthreshold_margin_v', 0.1),
])
def test_invalid_threshold_names_key(tmp_path, monkeypatch, key, value):
    from silk.circuit.region_thresholds import RegionThresholdsError
    entry = policy()
    entry[key] = value
    path = select_file(tmp_path, monkeypatch, {'sky130': entry})
    with pytest.raises(RegionThresholdsError) as caught:
        report('sky130').op_table()
    assert str(path) in str(caught.value)
    assert key in str(caught.value)


def test_missing_file_fails_but_override_bypasses_it(tmp_path, monkeypatch):
    from silk.circuit.region_thresholds import RegionThresholdsError
    path = tmp_path / 'missing.yaml'
    monkeypatch.setenv('SILK_REGION_THRESHOLDS', str(path))
    with pytest.raises(RegionThresholdsError, match='missing.yaml'):
        report().op_table()
    assert report().op_table(RegionThresholds())[0].region == 'subthreshold'


def test_html_uses_each_reports_policy_and_override(tmp_path, monkeypatch):
    from silk.circuit.modules import Module
    from silk.circuit.report import generate_report
    strict = policy()
    strict['min_current_a'] = 1e-6
    select_file(tmp_path, monkeypatch, {'sky130': strict})
    from silk.pdk import load_pdk
    dut = Module('PDK_skywater130', str(tmp_path), 'dut.cir',
                 pdk_details=load_pdk('sky130'))
    reports = [report('sky130'), report('gf180mcu')]
    generated = generate_report(dut, auto_discover_config=False, spec_reports=reports)
    assert [s.table[0].region for s in generated.op_tables] == ['cutoff', 'subthreshold']
    html = generated._to_html()
    assert '>cutoff<' in html and '>subthreshold<' in html
    overridden = generate_report(dut, auto_discover_config=False, spec_reports=reports,
                                 region_thresholds=RegionThresholds())
    assert [s.table[0].region for s in overridden.op_tables] == ['subthreshold'] * 2
    assert '>cutoff<' not in overridden._to_html()


def test_doctor_validates_selected_file(tmp_path, monkeypatch):
    from silk import doctor
    assert doctor.check_region_thresholds in doctor.REQUIRED_CHECKS
    path = select_file(tmp_path, monkeypatch, {'sky130': policy()})
    setting = next(c for c in doctor.check_settings() if c.name == 'setting SILK_REGION_THRESHOLDS')
    assert str(path) in setting.detail and 'source: env' in setting.detail
    assert doctor.check_region_thresholds()[0].status == doctor.OK
    path.write_text('version: 1\npdks: {sky130: {}}')
    result = doctor.check_region_thresholds()[0]
    assert result.status == doctor.FAIL
    assert str(path) in result.detail and 'min_current_a' in result.detail
    assert result.fix
    monkeypatch.setattr(doctor, 'REQUIRED_CHECKS', [doctor.check_region_thresholds])
    monkeypatch.setattr(doctor, 'OPTIONAL_CHECKS', [])
    assert doctor.main(['--json']) == 1


def test_weak_margin_changes_rendered_flag(tmp_path, monkeypatch):
    strict = policy()
    strict['weak_sat_margin_v'] = 0.08
    select_file(tmp_path, monkeypatch, {'default': strict})
    sr = report()
    sr.op_devices[0]['op_values'].update(vgs=1.0, vds=0.26, vdsat=0.2)
    assert sr.op_table()[0].saturation_weak is True
    assert 'saturation (weak)' in sr.op_table().to_markdown()
    assert sr.op_table(RegionThresholds())[0].saturation_weak is False


def test_file_reedit_is_seen_without_rerunning_report(tmp_path, monkeypatch):
    path = select_file(tmp_path, monkeypatch, {'default': policy()})
    sr = report()
    assert sr.op_table()[0].region == 'subthreshold'
    path.write_text(path.read_text().replace('1.0e-09', '1.0e-06'))
    assert sr.op_table()[0].region == 'cutoff'


def test_non_utf8_file_has_typed_diagnostic(tmp_path, monkeypatch):
    from silk.circuit.region_thresholds import RegionThresholdsError
    path = tmp_path / 'bad-encoding.yaml'
    path.write_bytes(b'\xff')
    monkeypatch.setenv('SILK_REGION_THRESHOLDS', str(path))
    with pytest.raises(RegionThresholdsError, match='bad-encoding.yaml.*<root>'):
        report().op_table()


@pytest.mark.parametrize('tagged', ['!!float nope', '!!int nope', '!!bool nope', '!!float \"\"',
                                    '!!timestamp 2026-99-99', '!!timestamp nope'])
def test_malformed_tagged_scalar_is_typed_and_doctor_reports_fail(tmp_path, monkeypatch, tagged):
    from silk.circuit.region_thresholds import RegionThresholdsError
    from silk import doctor
    path = select_file(tmp_path, monkeypatch, {'default': policy()})
    path.write_text(path.read_text().replace('min_current_a: 1.0e-09',
                                            f'min_current_a: {tagged}'))
    with pytest.raises(RegionThresholdsError) as caught:
        report().op_table()
    assert str(path) in str(caught.value)
    assert 'min_current_a' in str(caught.value)
    assert doctor.check_region_thresholds()[0].status == doctor.FAIL


_BAD_MAPPING_TAGS = [
    ('version: 1\npdks: !!map [sky130]', 'pdks'),
    ('version: 1\npdks: !!map nope', 'pdks'),
    ('version: 1\npdks:\n  sky130: !!map [min_current_a]', 'pdks.sky130'),
    ('version: 1\npdks:\n  sky130: !!map nope', 'pdks.sky130'),
]


@pytest.mark.parametrize('text,key', _BAD_MAPPING_TAGS)
def test_mapping_tag_node_kind_has_file_and_key(tmp_path, text, key):
    from silk.circuit.region_thresholds import load_thresholds_file, RegionThresholdsError
    path = tmp_path / 'bad-map.yaml'
    path.write_text(text)
    with pytest.raises(RegionThresholdsError) as caught:
        load_thresholds_file(path)
    assert str(path) in str(caught.value)
    assert key in str(caught.value)


@pytest.mark.parametrize('text,key', _BAD_MAPPING_TAGS)
def test_doctor_json_handles_mapping_tag_node_kind(tmp_path, monkeypatch, text, key):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    path = tmp_path / 'bad-map.yaml'
    path.write_text(text)
    monkeypatch.setenv('SILK_REGION_THRESHOLDS', str(path))
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root), MPLBACKEND='Agg')
    result = subprocess.run([sys.executable, '-m', 'silk.doctor', '--json'],
                            cwd=root, env=env, capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 1
    assert 'Traceback' not in result.stderr
    rows = json.loads(result.stdout)
    row = next(r for r in rows if r['name'] == 'region thresholds')
    assert row['status'] == 'FAIL'
    assert row['fix']
    assert str(path) in row['detail']
    assert key in row['detail']
