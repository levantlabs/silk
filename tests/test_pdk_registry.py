import pytest
from silk.pdk import registry as R


def test_available_includes_sky130():
    assert 'sky130' in R.available()


def test_registry_api_exported_from_silk_pdk():
    """silk.pdk.__init__ re-exports the discovery API (not just load_pdk)."""
    from silk.pdk import available, manifest, PDKRegistryError
    assert 'sky130' in available()
    assert manifest('sky130')['name'] == 'sky130'
    assert PDKRegistryError is R.PDKRegistryError


def test_registry_error_is_valueerror_backcompat():
    """Historical load_pdk contract raised ValueError for unknown names;
    PDKRegistryError must remain catchable as ValueError."""
    assert issubclass(R.PDKRegistryError, ValueError)
    with pytest.raises(ValueError):
        R.manifest('definitely-not-a-pdk')


def test_manifest_fields():
    m = R.manifest('sky130')
    assert m['name'] == 'sky130'
    assert 'class' not in m
    assert m['library']['default_corner'] == 'tt'
    assert m['library']['lib_path'].endswith('sky130A/libs.tech/ngspice/sky130.lib.spice')
    assert len(m['library']['corners']) == 51
    assert m['library']['corners']['tt']['kind'] == 'process'
    assert '_manifest_dir' in m  # absolute dir for resolving device_details/lib


def test_aliases_resolve_to_same_manifest():
    assert R.manifest('skywater130')['name'] == 'sky130'
    assert R.manifest('skywater_130')['name'] == 'sky130'


def test_unknown_name_raises_clear_error():
    with pytest.raises(R.PDKRegistryError) as e:
        R.manifest('nope')
    assert 'nope' in str(e.value) and 'sky130' in str(e.value)  # lists available


def test_default_corner_is_a_declared_corner():
    m = R.manifest('sky130')
    assert m['library']['default_corner'] in m['library']['corners']


def _fresh_registry(monkeypatch, tmp_path):
    monkeypatch.setattr(R, '_DATA_DIR', str(tmp_path))
    monkeypatch.setattr(R, '_CACHE', None)
    return R


def test_duplicate_canonical_name_across_files_raises(monkeypatch, tmp_path):
    (tmp_path / 'a.pdk.yaml').write_text('name: dup\nlibrary: {corners: {tt: {}}, default_corner: tt}\n')
    (tmp_path / 'b.pdk.yaml').write_text('name: dup\nlibrary: {corners: {tt: {}}, default_corner: tt}\n')
    R2 = _fresh_registry(monkeypatch, tmp_path)
    with pytest.raises(R2.PDKRegistryError) as e:
        R2.available()
    assert 'dup' in str(e.value)


def test_alias_collision_across_files_raises(monkeypatch, tmp_path):
    (tmp_path / 'a.pdk.yaml').write_text('name: aa\naliases: [shared]\nlibrary: {corners: {}, default_corner: tt}\n')
    (tmp_path / 'b.pdk.yaml').write_text('name: bb\naliases: [shared]\nlibrary: {corners: {}, default_corner: tt}\n')
    R2 = _fresh_registry(monkeypatch, tmp_path)
    with pytest.raises(R2.PDKRegistryError):
        R2.available()


def test_malformed_manifest_raises(monkeypatch, tmp_path):
    (tmp_path / 'bad.pdk.yaml').write_text('name: x\n  bad: : indent\n')
    R2 = _fresh_registry(monkeypatch, tmp_path)
    with pytest.raises(R2.PDKRegistryError):
        R2.available()


def test_non_list_aliases_raises(monkeypatch, tmp_path):
    (tmp_path / 'a.pdk.yaml').write_text('name: x\naliases: notalist\nlibrary: {corners: {}, default_corner: tt}\n')
    R2 = _fresh_registry(monkeypatch, tmp_path)
    with pytest.raises(R2.PDKRegistryError):
        R2.available()
