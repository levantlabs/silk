"""Tests for load_pdk() convenience function."""
import os
import pytest
from silk.pdk import load_pdk
from silk.pdk.registry import PDKRegistryError
from silk.pdk import BasePDK
from tests._env import needs_pdk, needs_pdk_named


def _stub_pdk_root(root):
    """Create a stub sky130.lib.spice at the manifest-expected relative path.

    Lets tests exercise real pdk_root/PDK_ROOT path-resolution logic without
    requiring an actual PDK checkout on the machine running the tests.
    """
    lib_dir = root / 'sky130A' / 'libs.tech' / 'ngspice'
    lib_dir.mkdir(parents=True)
    (lib_dir / 'sky130.lib.spice').write_text('* stub sky130.lib.spice\n')
    return root


class TestLoadPdkSignature:
    def test_unknown_pdk_raises_value_error(self):
        # Unknown names now resolve through the registry, which raises
        # PDKRegistryError (not ValueError) and names the unknown PDK.
        with pytest.raises(PDKRegistryError, match='tsmc28'):
            load_pdk('tsmc28')

    def test_sky130_alias_sky130(self, tmp_path):
        pdk = load_pdk('sky130', pdk_root=str(_stub_pdk_root(tmp_path)))
        assert type(pdk) is BasePDK
        assert pdk.pdk_name == 'sky130'

    def test_sky130_alias_skywater130(self, tmp_path):
        pdk = load_pdk('skywater130', pdk_root=str(_stub_pdk_root(tmp_path)))
        assert type(pdk) is BasePDK
        assert pdk.pdk_name == 'sky130'

    def test_sky130_alias_skywater_130(self, tmp_path):
        pdk = load_pdk('skywater_130', pdk_root=str(_stub_pdk_root(tmp_path)))
        assert type(pdk) is BasePDK
        assert pdk.pdk_name == 'sky130'

    def test_pdk_root_kwarg_sets_lib_path(self, tmp_path):
        pdk = load_pdk('sky130', pdk_root=str(_stub_pdk_root(tmp_path)))
        assert str(tmp_path) in pdk.library_location

    def test_pdk_root_env_var_used(self, monkeypatch, tmp_path):
        monkeypatch.setenv('PDK_ROOT', str(_stub_pdk_root(tmp_path)))
        pdk = load_pdk('sky130')
        assert str(tmp_path) in pdk.library_location

    def test_pdk_root_defaults_to_home_pdk(self, monkeypatch, tmp_path):
        monkeypatch.delenv('PDK_ROOT', raising=False)
        _stub_pdk_root(tmp_path / 'pdk')
        monkeypatch.setenv('HOME', str(tmp_path))
        pdk = load_pdk('sky130')
        home_pdk = os.path.expanduser('~/pdk')
        assert home_pdk in pdk.library_location

    def test_yaml_path_is_bundled(self, tmp_path):
        pdk = load_pdk('sky130', pdk_root=str(_stub_pdk_root(tmp_path)))
        assert os.path.isfile(pdk.model_list)

    def test_importable_from_silk(self):
        from silk import load_pdk as lp
        assert callable(lp)

    def test_importable_from_silk_pdk(self):
        from silk.pdk import load_pdk as lp
        assert callable(lp)


@pytest.mark.skipif(not os.environ.get('PDK_ROOT'), reason='needs installed PDK')
@needs_pdk
def test_load_pdk_populates_library_config():
    from silk import load_pdk
    p = load_pdk('sky130')
    assert p.pdk_name == 'sky130'
    assert p.default_corner == 'tt' and len(p.corners) == 51
    assert p.library_location.endswith('sky130A/libs.tech/ngspice/sky130.lib.spice')
    assert any('cap_vpp_11p5x11p7' in i for i in p.extra_includes)


@pytest.mark.skipif(not os.environ.get('PDK_ROOT'), reason='needs installed PDK')
@needs_pdk
def test_load_pdk_aliases_still_resolve():
    from silk import load_pdk
    assert load_pdk('skywater130').pdk_name == 'sky130'
    assert load_pdk('skywater_130').pdk_name == 'sky130'


def test_load_pdk_missing_lib_loads_and_resolves_the_path(tmp_path):
    """A missing model library no longer stops the load (TASK-71).

    Everything ``load_pdk`` returns except the ``.lib`` path ships inside
    silk, so an uninstalled PDK still yields a usable object: behavioral
    simulation, schema export, and SPICE decks that name no PDK device. The
    path is still RESOLVED against the root given, so the run-time gate and
    ``python -m silk.doctor`` can say exactly what is missing.
    """
    from silk import load_pdk
    pdk = load_pdk('sky130', pdk_root=str(tmp_path))
    assert pdk.pdk_name == 'sky130'
    assert str(pdk.library_location).startswith(str(tmp_path))
    assert not os.path.isfile(pdk.library_location)
    # The catalog came from inside the package, not from the missing root.
    assert pdk.nmos1p8 is not None


def test_load_pdk_unknown_name():
    from silk import load_pdk
    with pytest.raises(Exception) as e:
        load_pdk('nope')
    assert 'nope' in str(e.value)


def test_load_pdk_invalid_unit_mode_raises_value_error(tmp_path):
    from silk import load_pdk
    with pytest.raises(ValueError) as e:
        load_pdk('sky130', pdk_root=str(_stub_pdk_root(tmp_path)),
                  user_units='bogus')
    assert 'bogus' in str(e.value)
    # The message now lists the author-unit prefixes too, since user_units
    # accepts any of them ('u' = "I write microns").
    assert "'pdk'" in str(e.value) and "'si'" in str(e.value)
    assert 'base' in str(e.value) and 'u' in str(e.value)


# ---------------------------------------------------------------------------
# The PDK's declared default corner must actually be used
# ---------------------------------------------------------------------------

@needs_pdk_named('gf180mcu')
def test_generate_netlist_uses_the_pdks_declared_default_corner(tmp_path):
    """`generate_netlist()` used to default to the literal `'tt'`.

    That is sky130's spelling and nobody else's — GF180MCU calls its typical
    corner `typical` — so every deck emitted for another PDK asked ngspice for
    a section that does not exist and the run died with
    "section definition tt not found". `simulate()` already resolved the
    manifest's `default_corner`; only the netlist path did not, so the two
    disagreed about what corner a run actually was.
    """
    from silk import load_pdk, sources
    from silk.circuit.modules import Module

    pdk = load_pdk('gf180mcu')
    assert pdk.default_corner == 'typical', 'fixture assumption'

    tb = Module('gf180mcu', str(tmp_path), 'x.cir', pdk_details=pdk)
    tb.add_net('d')
    with tb:
        pdk.nmos3p3('M1', d='d', g='d', s='0', b='0', w=10e-6, l=0.28e-6)
    tb.add_source(sources.vdc('d', positive='d', negative='0', dc=1.0))
    tb.generate_netlist()

    deck = (tmp_path / 'x.cir').read_text()
    lib = [l for l in deck.splitlines() if l.lower().startswith('.lib')]
    assert lib, deck
    assert lib[0].endswith(' typical'), lib[0]

    tb.generate_netlist(corner='ss')
    lib = [l for l in (tmp_path / 'x.cir').read_text().splitlines()
           if l.lower().startswith('.lib')]
    assert lib[0].endswith(' ss'), 'an explicit corner must still win'


# ---------------------------------------------------------------------------
# Per-corner include lists: one corner, several (file, section) pairs
# ---------------------------------------------------------------------------

@needs_pdk_named('sg13g2')
def test_a_corner_can_declare_several_lib_files(tmp_path):
    """One corner, six `.lib` lines.

    SG13G2 splits device families across separate corner files, and the
    section name is NOT derivable from the corner name — MOS says `mos_tt`,
    resistors `res_typ`, caps `cap_typ`, diodes `dio_tt`, HBTs `hbt_typ`.
    cornerMOSlv.lib and cornerMOShv.lib even use IDENTICAL section names, so
    the file is the only thing telling low- from high-voltage apart. Hence
    explicit (file, section) pairs rather than a derived suffix.
    """
    from silk import load_pdk, sources
    from silk.circuit.modules import Module

    pdk = load_pdk('sg13g2')
    tb = Module('sg13g2', str(tmp_path), 'x.cir', pdk_details=pdk)
    tb.add_net('n')
    with tb:
        pdk.nmos_lv('M1', d='n', g='n', s='0', b='0', w=1e-6, l=0.34e-6)
    tb.add_source(sources.vdc('n', positive='n', negative='0', dc=1.0))
    tb.generate_netlist()

    libs = [l.strip() for l in (tmp_path / 'x.cir').read_text().splitlines()
            if l.lower().startswith('.lib')]
    assert len(libs) == 6, libs
    assert sum('cornerMOSlv.lib" mos_tt' in l for l in libs) == 1
    assert sum('cornerMOShv.lib" mos_tt' in l for l in libs) == 1
    assert any('cornerRES.lib" res_typ' in l for l in libs)
    assert any('cornerHBT.lib" hbt_typ' in l for l in libs)

    # A different corner must move EVERY family, not just the MOS ones.
    tb.generate_netlist(corner='ss')
    libs = [l.strip() for l in (tmp_path / 'x.cir').read_text().splitlines()
            if l.lower().startswith('.lib')]
    assert any('mos_ss' in l for l in libs)
    assert any('res_wcs' in l for l in libs), 'passives must track the corner too'


@needs_pdk
def test_a_pdk_without_per_corner_includes_still_emits_one_lib_line(tmp_path):
    """Backward compatibility: sky130 and GF180MCU declare no `includes:`, so
    they keep the historical single line with the corner name as the section."""
    from silk import load_pdk, sources
    from silk.circuit.modules import Module

    pdk = load_pdk('sky130')
    tb = Module('sky130', str(tmp_path), 'x.cir', pdk_details=pdk)
    tb.add_net('n')
    with tb:
        pdk.nmos1p8('M1', d='n', g='n', s='0', b='0', w=1.0, l=0.15)
    tb.add_source(sources.vdc('n', positive='n', negative='0', dc=1.0))
    from pathlib import Path
    deck_path = tb.generate_netlist()

    libs = [l for l in Path(deck_path).read_text().splitlines()
            if l.lower().startswith('.lib')]
    assert len(libs) == 1 and libs[0].strip().endswith(' tt'), libs


# ---------------------------------------------------------------------------
# Units: one circuit source, correct deck for every PDK
# ---------------------------------------------------------------------------

@needs_pdk
def test_unit_mode_pdk_scales_nothing(tmp_path):
    """`pdk` means "the author writes whatever this PDK wants", so nothing
    scales. No longer the default — the manifests declare `u` — but it stays
    supported for a PDK whose parameters are not dimensioned."""
    from silk import load_pdk
    pdk = load_pdk('sky130', user_units='pdk')
    assert pdk.nmos1p8._param_scales == {}
    dev = pdk.nmos1p8('M1', d='a', g='b', s='0', b='0', w=4.0, l=0.5)
    assert 'w=4 l=0.5' in dev.to_spice()


@needs_pdk
def test_the_manifest_default_emits_identically_to_no_scaling(tmp_path):
    """sky130's manifest now declares `user_units: u`, and sky130's own
    parameters are already in µm — so every scale is exactly 1.0 and every
    deck ever written against this PDK is byte-identical. The team-wide
    default moved without moving a single number."""
    from silk import load_pdk
    default = load_pdk('sky130')
    assert default.user_units == 'u', 'the manifest carries the team unit'
    assert set(default.nmos1p8._param_scales.values()) == {1.0}
    assert set(default.diode_std._param_scales.values()) == {1.0}, \
        'area must scale by 1 too — quantity: area, not a bare prefix'
    assert 'w=4 l=0.5' in default.nmos1p8(
        'M1', d='a', g='b', s='0', b='0', w=4.0, l=0.5).to_spice()


@needs_pdk
def test_area_scales_by_the_square_of_the_prefix(tmp_path):
    """A length in µm scales by 1e6; an AREA in µm² by 1e6 SQUARED.

    Without `quantity: area`, an author writing microns had diode `area`
    scaled by 1e6 rather than 1e12 — 0.09 µm² emitted as 90000. sky130 tags
    that parameter `pdk_unit: p` because 1e12 happens to be the m²→µm² factor,
    not because an area is measured in picofarads.
    """
    from silk import load_pdk
    assert load_pdk('sky130', user_units='u').diode_std._param_scales['area'] == 1.0
    assert load_pdk('sky130', user_units='si').diode_std._param_scales['area'] == 1e12


@needs_pdk
def test_si_mode_converts_into_the_pdks_own_unit(tmp_path):
    """sky130's subckts take µm, so SI metres must arrive as µm numbers.

    Before this, the factory path ignored `pdk_unit:` and `user_units` entirely and
    just SI-formatted whatever it was handed: w=1e-6 became `w=1u`, which
    sky130 reads as 1e-6 µm — a one-picometre device, emitted silently.
    """
    from silk import load_pdk
    pdk = load_pdk('sky130', user_units='si')
    assert pdk.nmos1p8._param_scales == {'w': 1e6, 'l': 1e6}
    dev = pdk.nmos1p8('M1', d='a', g='b', s='0', b='0', w=1e-6, l=0.3e-6)
    assert 'w=1 l=0.3' in dev.to_spice(), dev.to_spice()


@needs_pdk_named('gf180mcu')
def test_authoring_in_microns_lands_correctly_on_a_metres_pdk(tmp_path):
    """The portability case, stated the way a designer states it: I write
    `w = 1` meaning one micron, and silk hands each PDK what its models want.

    sky130's models take µm, so `w=1` stays `1`. GF180MCU's take metres, so the
    same `w=1` must become 1e-6 — emitted as `1u`, which is the same quantity.
    """
    from silk import load_pdk
    sky = load_pdk('sky130', user_units='u')
    gf = load_pdk('gf180mcu', user_units='u')
    assert sky.nmos1p8._param_scales == {'w': 1.0, 'l': 1.0}
    assert gf.nmos3p3._param_scales == {'w': 1e-6, 'l': 1e-6}

    a = sky.nmos1p8('M1', d='a', g='b', s='0', b='0', w=1.0, l=0.3)
    b = gf.nmos3p3('M1', d='a', g='b', s='0', b='0', w=1.0, l=0.3)
    assert 'w=1 l=0.3' in a.to_spice(), a.to_spice()
    assert 'w=1u l=300n' in b.to_spice(), b.to_spice()


@needs_pdk_named('gf180mcu')
def test_the_same_source_is_correct_on_a_base_unit_pdk(tmp_path):
    """GF180MCU's subckts take METRES, sky130's take µm. Under `si` the caller
    writes SI in both cases and each PDK gets the number its models expect —
    which is the whole point: the numbers in circuit code stop moving."""
    from silk import load_pdk
    pdk = load_pdk('gf180mcu', user_units='si')
    assert pdk.nmos3p3._param_scales == {'w': 1.0, 'l': 1.0}
    dev = pdk.nmos3p3('M1', d='a', g='b', s='0', b='0', w=1e-6, l=0.3e-6)
    # 1e-6 m and 3e-7 m, SI-formatted — not µm numbers
    assert 'w=1u l=300n' in dev.to_spice(), dev.to_spice()


@needs_pdk
def test_dimensionless_parameters_never_scale(tmp_path):
    """`m`, `nf`, `ng` declare no unit and must survive `si` mode untouched."""
    from silk import load_pdk
    pdk = load_pdk('sky130', user_units='si')
    assert 'm' not in pdk.nmos1p8._param_scales
    dev = pdk.nmos1p8('M1', d='a', g='b', s='0', b='0', w=1e-6, l=0.3e-6, m=4)
    assert 'm=4' in dev.to_spice()


def test_base_is_a_no_prefix_scale():
    """'base' exists because every letter was taken — 'm' is milli, so a
    metres-native PDK could not previously declare its unit at all."""
    from silk.pdk.basepdk import _SI_TO_PDK_SCALE
    assert _SI_TO_PDK_SCALE['base'] == 1.0
    assert _SI_TO_PDK_SCALE['m'] == 1e3, "'m' still means milli, not metre"


@needs_pdk
def test_unit_mode_accepts_any_prefix_the_author_writes_in(tmp_path):
    """`user_units` names the AUTHOR's unit; `pdk_unit:` names the PDK's. Both draw
    on the same prefix vocabulary, so the factor is simply
    scale[pdk] / scale[author]."""
    from silk import load_pdk
    assert load_pdk('sky130', user_units='u').nmos1p8._param_scales['w'] == 1.0
    assert load_pdk('sky130', user_units='n').nmos1p8._param_scales['w'] == 1e-3
    assert load_pdk('sky130', user_units='si').nmos1p8._param_scales['w'] == 1e6


@needs_pdk
def test_an_unknown_unit_mode_lists_what_is_valid(tmp_path):
    from silk import load_pdk
    with pytest.raises(ValueError) as e:
        load_pdk('sky130', user_units='micron')
    assert 'micron' in str(e.value) and 'base' in str(e.value)
