import pytest
from silk import doctor, load_pdk
from silk.exceptions import (ConfigurationError, MissingLibrarySectionError,
                             MonteCarloError, PDKError)
from silk.pdk.statistical import library_sections, require_library_sections


def _lib(tmp_path, *sections):
    p = tmp_path / 'models.lib.spice'
    p.write_text(''.join(f'.lib {s}\n.param x=1\n.endl {s}\n' for s in sections))
    return str(p)


def test_sky130_manifest_declares_its_statistics():
    pdk = load_pdk('sky130')
    st = pdk.statistical
    assert st.mismatch_sections['tt'] == 'tt_mm'
    assert st.mismatch_sections['ss'] == 'ss_mm'
    assert st.process_section == 'mc' and st.process_corners == ('tt',)
    assert st.mismatch_switch == 'mc_mm_switch'
    assert st.process_switch == 'mc_pr_switch'
    assert st.record['nmos1p8'] == ('vth0', 'toxe', 'voff', 'nfactor')
    assert st.min_open_pdks == '1.0.306'
    assert all(p.startswith('/') for p in st.process_includes)
    assert len(st.process_includes) == 2


_DRAWN = {'nmos1p8', 'nmos1p8_lvt', 'nmos3p3_native', 'nmos5v',
          'nmos5v_native', 'pmos1p8', 'pmos1p8_lvt', 'pmos1p8_hvt', 'pmos5v'}


def test_record_lists_only_devices_whose_models_draw_mismatch():
    st = load_pdk('sky130').statistical
    assert set(st.record) == _DRAWN
    assert all(p == ('vth0', 'toxe', 'voff', 'nfactor')
               for p in st.record.values())


def test_recorded_parameters_are_drawn_by_the_installed_models():
    import os
    import re
    import yaml
    pdk = load_pdk('sky130')
    root = os.environ.get('PDK_ROOT', os.path.expanduser('~/pdk'))
    spice = os.path.join(root, 'sky130A/libs.ref/sky130_fd_pr/spice')
    with open(pdk.model_list) as f:
        details = yaml.safe_load(f)
    for dev, params in pdk.statistical.record.items():
        path = os.path.join(spice, details[dev]['PDKDevice'] + '.pm3.spice')
        if not os.path.isfile(path):
            pytest.skip(f'sky130 model file not installed: {path}')
        drawn = ' '.join(l for l in open(path).read().splitlines()
                         if 'mc_mm_switch' in l.lower()).lower()
        for p in params:
            assert re.search(rf'\+\s*{p}\s*=', drawn) or f'{p} =' in drawn, \
                f'{dev}: {p} not drawn in {path}'


def test_process_includes_resolve_against_pdk_root(tmp_path):
    st = load_pdk('sky130', pdk_root=str(tmp_path)).statistical
    assert all(p.startswith(str(tmp_path)) for p in st.process_includes)


def test_library_sections_reads_lib_lines_case_insensitively(tmp_path):
    path = _lib(tmp_path, 'tt', 'TT_MM')
    assert library_sections(path) == frozenset({'tt', 'tt_mm'})


def test_library_sections_cache_follows_mtime(tmp_path):
    import os
    path = _lib(tmp_path, 'tt')
    assert library_sections(path) == frozenset({'tt'})
    with open(path, 'a') as f:
        f.write('.lib ss\n.endl ss\n')
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert library_sections(path) == frozenset({'tt', 'ss'})


def test_missing_section_raises_a_configuration_error_naming_it(tmp_path):
    pdk = load_pdk('sky130', library_location=_lib(tmp_path, 'tt'))
    with pytest.raises(MissingLibrarySectionError) as err:
        require_library_sections(pdk, ['tt_mm', 'mc'])
    assert isinstance(err.value, PDKError)
    assert isinstance(err.value, ConfigurationError)
    msg = str(err.value)
    assert 'tt_mm' in msg and 'mc' in msg and 'models.lib.spice' in msg
    assert 'open_pdks' in msg


def test_present_sections_pass_case_insensitively(tmp_path):
    pdk = load_pdk('sky130', library_location=_lib(tmp_path, 'TT_MM'))
    require_library_sections(pdk, ['tt_mm'])


def test_a_pdk_without_the_block_has_statistical_none():
    assert load_pdk('gf180mcu').statistical is None


def test_monte_carlo_error_is_a_configuration_error_and_value_error():
    assert issubclass(MonteCarloError, ConfigurationError)
    assert issubclass(MonteCarloError, ValueError)


def test_doctor_warns_per_missing_statistical_section(monkeypatch, tmp_path):
    root = tmp_path / 'sky130A' / 'libs.tech' / 'ngspice'
    root.mkdir(parents=True)
    (root / 'sky130.lib.spice').write_text('.lib tt\n.endl tt\n')
    monkeypatch.setenv('PDK_ROOT', str(tmp_path))
    rows = [c for c in doctor.check_pdks() if c.name.startswith('pdk sky130')]
    assert all(c.status != doctor.FAIL for c in rows)
    warned = {c.name: c for c in rows if c.status == doctor.WARN}
    # one warn row per missing section: the five mismatch corners and 'mc'
    for sec in ('tt_mm', 'ss_mm', 'ff_mm', 'sf_mm', 'fs_mm', 'mc'):
        row = warned[f'pdk sky130 section {sec}']
        assert sec in row.detail and row.fix
    assert 'pdk sky130 section tt' not in warned
