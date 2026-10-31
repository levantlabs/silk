"""GF180MCU's resistors, capacitors, diodes and bipolars.

The catalog for these was generated from the installed PDK, and the point of
these tests is that it STAYS generated: every assertion below re-reads
sm141064.ngspice (or an xschem symbol) and checks silk against it, rather
than against a number copied into the test. A test that hardcodes the
answer cannot catch the failure that matters here — a pin order or a unit
that silently disagrees with the model.
"""
import math
import os
import re

import pytest

from tests._env import needs_pdk_named, needs_ngspice

pytestmark = [
    needs_pdk_named('gf180mcu'),
    pytest.mark.requires_gf180,
]

RESISTORS = [
    'res_nplus_u', 'res_pplus_u', 'res_nplus_s', 'res_pplus_s', 'res_nwell',
    'res_npolyf_u', 'res_ppolyf_u', 'res_npolyf_s', 'res_ppolyf_s',
    'res_ppolyf_1k', 'res_ppolyf_2k', 'res_ppolyf_3k',
    'res_ppolyf_1k_6p0', 'res_ppolyf_2k_6p0',
    'res_rm1', 'res_rm2', 'res_rm3', 'res_rm4',
    'res_tm6k', 'res_tm9k', 'res_tm11k', 'res_tm30k',
]
MIMCAPS = ['cap_mim_1f0', 'cap_mim_1f5', 'cap_mim_2f0']
MOSCAPS = ['cap_nmos3p3', 'cap_pmos3p3', 'cap_nmos6p0', 'cap_pmos6p0',
           'cap_nmos3p3_b', 'cap_pmos3p3_b', 'cap_nmos6p0_b', 'cap_pmos6p0_b']
DIODES = ['diode_nd2ps_3p3', 'diode_pd2nw_3p3', 'diode_nd2ps_6p0',
          'diode_pd2nw_6p0', 'diode_nw2ps_3p3', 'diode_nw2ps_6p0',
          'diode_pw2dw', 'diode_dw2ps', 'diode_schottky']
BJTS = ['npn_10x10', 'npn_05x05', 'npn_0p54x16', 'npn_0p54x08',
        'npn_0p54x04', 'npn_0p54x02',
        'pnp_10x10', 'pnp_05x05', 'pnp_10x0p42', 'pnp_05x0p42']

NEW_DEVICES = RESISTORS + MIMCAPS + MOSCAPS + DIODES + BJTS


@pytest.fixture(scope='module')
def pdk():
    from silk import load_pdk
    return load_pdk('gf180mcu')


@pytest.fixture(scope='module')
def model_text(pdk):
    return open(pdk.library_location, errors='ignore').read()


@pytest.fixture(scope='module')
def subckts(model_text):
    """{name: [terminal, ...]} for every .subckt in the model file."""
    out = {}
    for m in re.finditer(r'^\s*\.subckt\s+(\S+)\s+(.*)$', model_text,
                         re.I | re.M):
        terms = [t for t in m.group(2).split() if '=' not in t]
        out.setdefault(m.group(1), terms)
    return out


# ---------------------------------------------------------------------------
# The catalog covers every family
# ---------------------------------------------------------------------------

def test_every_new_family_is_in_the_catalog(pdk):
    """Before this, `.LIB typical` pulled in the FETs and nothing else, so the
    catalog was MOSFETs-only and a GF180MCU design could not use a resistor."""
    missing = [d for d in NEW_DEVICES if d not in pdk.model_names]
    assert not missing, missing
    assert len(pdk.model_names) == 9 + len(NEW_DEVICES), pdk.model_names


# ---------------------------------------------------------------------------
# Pin order is the .subckt's, not a guess
# ---------------------------------------------------------------------------

def test_subckt_devices_carry_the_subckt_s_own_terminal_count(pdk, subckts):
    """A pinList one pin short of its .subckt silently shifts every net."""
    for name in RESISTORS + MIMCAPS + MOSCAPS + BJTS:
        dev = pdk._devices[name]
        terms = subckts[dev.pdk_device]
        assert len(dev.pin_list) == len(terms), (
            f'{name} -> .subckt {dev.pdk_device} {" ".join(terms)}')


def test_bjt_pin_order_is_read_from_the_subckt(pdk, subckts):
    """The BJT .subckt names its terminals (`c b e s`), so this is the one
    family where the file states the order outright — and where the xschem
    symbol's own pinnumbers disagree with it (B=1, E=2, C=3). The file wins."""
    for name in BJTS:
        dev = pdk._devices[name]
        terms = [t.lower() for t in subckts[dev.pdk_device]]
        assert list(dev.pin_list) == terms, name
    assert list(pdk._devices['npn_10x10'].pin_list) == ['c', 'b', 'e', 's']
    assert list(pdk._devices['pnp_10x10'].pin_list) == ['c', 'b', 'e']


def test_resistors_carry_the_well_or_substrate_tie(pdk):
    """Pin 3 is a real terminal, not an optional one: leaving it off shifts
    nothing but leaves the junction/field-oxide plate floating."""
    for name in RESISTORS:
        pins = list(pdk._devices[name].pin_list)
        assert pins[:2] == ['a', 'b'], name
        # metal resistors sit on oxide: no junction, and no third terminal
        is_metal = name.startswith(('res_rm', 'res_tm'))
        assert (pins == ['a', 'b', 'c']) is not is_metal, (name, pins)


# ---------------------------------------------------------------------------
# Diodes are .model primitives, not subcircuits
# ---------------------------------------------------------------------------

def test_diodes_emit_a_D_line_because_they_are_model_cards(pdk, model_text,
                                                           subckts):
    """GF180MCU's diodes are `.model <name> d level=3` cards. Emitted as `XM…`
    they would bind to a subcircuit that does not exist."""
    for name in DIODES:
        dev = pdk._devices[name]
        assert dev.pdk_device not in subckts, name
        assert re.search(r'^\s*\.model\s+' + re.escape(dev.pdk_device) + r'\s+d\b',
                         model_text, re.I | re.M), name
        cls = getattr(pdk, name)
        assert cls._spice_prefix == 'D', name
        line = cls('D1', a='p', k='n').to_spice()
        assert line.startswith('DD1 p n ' + dev.pdk_device), line


def test_diodes_emit_a_D_line_on_the_add_element_path(pdk, tmp_path):
    """The legacy ``add_element`` path stores a PDKDevice, not the factory
    class, and used to spell ``XM`` regardless of the catalog (TASK-23). The
    deck it writes must carry the same ``D`` primitive the class path emits."""
    from silk.circuit.modules import Module
    from silk.circuit.circuit_elements import PortDir
    from silk.circuit.emitted_names import emitted_instance_name
    for name in DIODES:
        dev = pdk._devices[name]
        assert dev.spice_prefix == 'D' == getattr(pdk, name)._spice_prefix, name
        assert emitted_instance_name(dev, 'D1') == 'DD1', name
    name = DIODES[0]
    m = Module('gf180mcu', str(tmp_path), f'{name}.cir', pdk_details=pdk,
               module_name='sub')
    m.define_module_ports({'p': PortDir.IN, 'n': PortDir.OUT})
    m.add(getattr(pdk, name)('D1', a='p', k='n'))
    deck = m.generate_netlist(corner='tt').splitlines()   # subckt text
    want = 'DD1 p n ' + pdk._devices[name].pdk_device + ' '
    assert any(l.startswith(want) for l in deck), (want, deck)
    assert not any(l.startswith('XMD1') for l in deck), deck


def test_the_other_families_still_emit_subcircuit_calls(pdk):
    for name in RESISTORS + MIMCAPS + MOSCAPS + BJTS:
        assert getattr(pdk, name)._spice_prefix.startswith('X'), name


# ---------------------------------------------------------------------------
# Units: a diode area is an area
# ---------------------------------------------------------------------------

def test_a_diode_area_scales_by_the_square_not_the_prefix():
    """`pdk_unit: base` + `quantity: area` is what stops an author's 1 um^2 from
    being emitted as 1e-6 m^2 — a million times too big. The perimeter beside
    it is a length and must scale once."""
    from silk import load_pdk
    pdk = load_pdk('gf180mcu', user_units='u')       # author writes microns
    scales = pdk.diode_nd2ps_3p3._param_scales
    assert scales['area'] == pytest.approx(1e-12)
    assert scales['pj'] == pytest.approx(1e-6)
    assert 'm' not in scales, 'the multiplier is dimensionless'
    line = pdk.diode_nd2ps_3p3('D1', a='p', k='0', area=1.0, pj=4.0).to_spice()
    assert 'area=1p pj=4u' in line, line


def test_resistor_and_cap_geometry_is_metres_like_the_rest_of_this_pdk():
    from silk import load_pdk
    pdk = load_pdk('gf180mcu', user_units='u')
    assert pdk.res_nplus_u._param_scales == {'r_width': 1e-6, 'r_length': 1e-6}
    assert pdk.cap_mim_1f5._param_scales == {'c_width': 1e-6, 'c_length': 1e-6}


# ---------------------------------------------------------------------------
# Defaults come from GF's own xschem symbols
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('device,symbol', [
    ('res_nplus_u', 'nplus_u'),
    ('res_nwell', 'nwell'),          # 3 um square, not 1
    ('res_rm1', 'rm1'),              # 0.6 um square, not 1
    ('cap_mim_1f5', 'cap_mim_1f5fF'),
])
def test_geometry_defaults_are_the_xschem_symbol_s_template(device, symbol, pdk):
    """The .subckt's own default is `r_length=l`, a reference to a global the
    PDK never defines — so the defaults come from the geometry GF's own
    schematics instantiate. Re-read here so a blanket 1 um cannot creep in."""
    variant = os.path.dirname(os.path.dirname(os.path.dirname(
        pdk.library_location)))                       # .../gf180mcuD
    sym = os.path.join(variant, 'libs.tech', 'xschem', 'symbols',
                       symbol + '.sym')
    text = open(sym, errors='ignore').read()
    tmpl = dict(tok.split('=', 1) for tok in
                re.search(r'template="(.*?)"', text, re.S).group(1).split()
                if '=' in tok)
    cls = getattr(pdk, device)
    prefix = 'r_' if device.startswith('res_') else 'c_'
    for key, param in (('W', prefix + 'width'), ('L', prefix + 'length')):
        # float() because the default is a quoted SPICE literal, not a number
        # — see test_geometry_defaults_are_unit_mode_independent.
        assert float(cls._param_defaults[param]) == pytest.approx(float(tmpl[key]))


def test_geometry_defaults_are_unit_mode_independent():
    """Defaults are in the PDK's own unit and are emitted UNSCALED — only
    caller-supplied values are converted — so the same omitted parameter emits
    the same geometry whatever unit the author writes in.

    Unquoted they would be at PyYAML's mercy: `1e-06` has no decimal point, so
    YAML 1.1 types it a string and it survives, while `1.2e-05` has one, comes
    back a float, and is then scaled a second time into 1.2e-11 m. That is not
    hypothetical — diode_pw2dw's perimeter is exactly such a number.
    """
    from silk import load_pdk
    for build in (lambda p: p.res_nwell('R1', a='a', b='b', c='c'),
                  lambda p: p.diode_pw2dw('D1', a='a', k='k'),
                  lambda p: p.cap_mim_1f5('C1', plus='p', minus='m')):
        emitted = {mode: build(load_pdk('gf180mcu', user_units=mode)).to_spice()
                   for mode in ('u', 'si', 'pdk')}
        assert len(set(emitted.values())) == 1, emitted
    pdk = load_pdk('gf180mcu', user_units='u')
    assert 'r_width=3e-06 r_length=3e-06' in \
        pdk.res_nwell('R1', a='a', b='b', c='c').to_spice()
    assert 'area=9e-12 pj=12u' in \
        pdk.diode_pw2dw('D1', a='a', k='k').to_spice()


# ---------------------------------------------------------------------------
# Corners: one .lib line per family, at every corner
# ---------------------------------------------------------------------------

_FAMILIES = ('res', 'moscap', 'mimcap', 'cap_mim', 'diode', 'bjt')


def test_every_corner_pulls_in_every_device_family(pdk, tmp_path):
    """A corner that moves the FETs but leaves the resistors typical is a
    silently wrong corner run, which is worse than a missing one."""
    from silk import sources
    from silk.circuit.modules import Module

    for corner in pdk.corners:
        tb = Module('gf180mcu', str(tmp_path), f'{corner}.cir', pdk_details=pdk)
        tb.add_net('d')
        with tb:
            pdk.nmos3p3('M1', d='d', g='d', s='0', b='0', w=10e-6, l=0.28e-6)
        tb.add_source(sources.vdc('d', positive='d', negative='0', dc=1.0))
        tb.generate_netlist(corner=corner)
        sections = [l.rsplit(None, 1)[-1] for l in
                    (tmp_path / f'{corner}.cir').read_text().splitlines()
                    if l.lower().startswith('.lib ')]
        assert len(sections) == 7, (corner, sections)
        for family in _FAMILIES:
            assert any(s == family or s.startswith(family + '_')
                       for s in sections), (corner, family, sections)


def test_every_corner_section_exists_in_the_model_file(pdk, model_text):
    """`.lib <file> <section>` with a typo'd section is not an error silk can
    see — ngspice reports "section definition not found" at run time."""
    defined = {m.group(1).lower() for m in
               re.finditer(r'^\s*\.lib\s+([A-Za-z_][\w]*)\s*$', model_text,
                           re.I | re.M)}
    defined |= {m.group(1).lower() for m in
                re.finditer(r'^\s*\.lib\s+([A-Za-z_][\w]*)\s*$', model_text,
                            re.I | re.M)}
    for corner, meta in pdk.corners.items():
        for inc in meta['includes']:
            assert inc['section'].lower() in defined, (corner, inc['section'])


def test_the_mos_corner_still_moves_with_the_corner_name(pdk, tmp_path):
    """The FET section is the corner's own name, as before the passives
    landed — the regression the `default_corner` fix was about."""
    from silk import sources
    from silk.circuit.modules import Module
    for corner, mos in (('typical', 'typical'), ('ss', 'ss'), ('ff', 'ff')):
        tb = Module('gf180mcu', str(tmp_path), 'c.cir', pdk_details=pdk)
        tb.add_net('d')
        with tb:
            pdk.nmos3p3('M1', d='d', g='d', s='0', b='0', w=10e-6, l=0.28e-6)
        tb.add_source(sources.vdc('d', positive='d', negative='0', dc=1.0))
        tb.generate_netlist(corner=corner)
        first = [l for l in (tmp_path / 'c.cir').read_text().splitlines()
                 if l.lower().startswith('.lib ')][0]
        assert first.endswith(' ' + mos), first


# ---------------------------------------------------------------------------
# The validator agrees
# ---------------------------------------------------------------------------

def test_the_pdk_validates_clean():
    """Arity + emission prefix of all 61 devices against the real
    .subckt/.model lines, through silk's own validator."""
    from silk import load_pdk
    from silk.pdk.validate import validate_pdk
    issues = validate_pdk(load_pdk('gf180mcu'))
    assert issues == [], [(i.device, i.kind, i.detail) for i in issues]


# ---------------------------------------------------------------------------
# ngspice: the numbers, against closed form
# ---------------------------------------------------------------------------

@pytest.mark.slow
@needs_ngspice
def test_a_diffusion_resistor_matches_its_sheet_resistance(tmp_path):
    """nplus_u, 1 x 10 um, at `.temp 25` (the models' own tnom, so every
    temperature coefficient is exactly 1):

        R = rsh * (L - 2*r_dl)/(W - 2*r_dw) + 2 * rsh_t * 1u/(W - 2*r_dw)

    with rsh = 60 (res_typical), r_dl = 2e-11, r_dw = -5e-8, rsh_t = 18.5.
    A resistor that binds but reads the wrong sheet — or takes W/L in the
    wrong unit — passes every static check and fails this one.
    """
    import numpy as np
    from silk import load_pdk, sources
    from silk.circuit.modules import Module
    from silk.simulation import Transient

    pdk = load_pdk('gf180mcu', user_units='si')
    W, L = 1e-6, 10e-6
    r_w = W - 2 * (-5e-8)
    R = 60.0 * (L - 2 * 2e-11) / r_w + 2 * 18.5 * (1e-6 / r_w)

    tb = Module('gf180mcu', str(tmp_path), 'r.cir', pdk_details=pdk)
    tb.add_custom_line('.temp 25')
    for n in ('a', 'sub'):
        tb.add_net(n)
    with tb:
        pdk.res_nplus_u('R1', a='a', b='0', c='sub', r_width=W, r_length=L, m=1)
    tb.add_source(sources.VoltageSource('a', positive='a', negative='0', dc=1.0))
    tb.add_source(sources.VoltageSource('sub', positive='sub', negative='0', dc=0))
    tb.add_sim_type(Transient(step=1e-9, stop=5e-9))
    tb.simulate(timeout=120)

    i = abs(float(np.asarray(tb.load_signal('va', 'I').y)[-1]))
    assert i == pytest.approx(1.0 / R, rel=1e-3), (i, 1.0 / R)


@pytest.mark.slow
@needs_ngspice
def test_the_moscap_pin_called_gate_really_is_the_gate(tmp_path):
    """The .subckt numbers its nodes 1/2 and the symbol's pinnumbers contradict
    its own pin order, so silk's `g`/`b` naming is settled by the model
    equation: c = A*(cvar1 + cvar2*tanh(cvar3*v(1,2) + cvar4)) peaks when
    node 1 is a threshold ABOVE node 2. Bias silk's `g` both ways and the
    capacitance must swing by (cvar1+cvar2)/(cvar1-cvar2) = 173x — if `g` were
    the bulk the swing would be inverted.
    """
    import numpy as np
    from silk import load_pdk, sources
    from silk.circuit.modules import Module
    from silk.simulation import AC

    pdk = load_pdk('gf180mcu', user_units='si')
    W = L = 10e-6
    f = 1e6
    caps = {}
    for vg in (3.3, -3.3):
        tb = Module('gf180mcu', str(tmp_path), f'c{vg:+.0f}.cir', pdk_details=pdk)
        tb.add_custom_line('.temp 25')
        tb.add_net('g')
        with tb:
            pdk.cap_nmos3p3('C1', g='g', b='0', c_width=W, c_length=L, m=1)
        tb.add_source(sources.VoltageSource('g', positive='g', negative='0',
                                            dc=vg, ac=1))
        tb.add_sim_type(AC('lin', 1, f, f))
        tb.simulate(timeout=120)
        i = abs(complex(np.asarray(tb.load_signal('vg', 'I').y)[-1]))
        caps[vg] = i / (2 * math.pi * f)
        expect = W * L * (0.002003 + 0.00198 * math.tanh(6.25 * vg - 3.9375))
        assert caps[vg] == pytest.approx(expect, rel=1e-3), (vg, caps[vg], expect)

    assert caps[3.3] / caps[-3.3] == pytest.approx(
        (0.002003 + 0.00198) / (0.002003 - 0.00198), rel=1e-2)
