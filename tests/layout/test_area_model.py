"""Tests for compute_device_area, compute_module_bounding_box and the unit
contract behind them (TASK-15).

Unit contract under test: ``silk.layout`` owns NO unit knowledge. Device
geometry is read through ONE accessor, ``silk.pdk.basepdk.device_geometry_um``,
which converts author unit -> PDK unit -> micrometres from the catalog's
``pdk_unit:`` tags and the PDK's ``user_units``. Anything the accessor cannot
read without guessing a unit fails loudly.

The fakes here are real factory-built elements from DECLARED-unit specs (no
installed PDK library is needed — a bundled catalog plus a fake lib path is
enough to build a PDK object, exactly as ``tests/conftest.py::pdk`` does).
"""
import math
import os

import pytest

from silk.layout.area_model import (
    compute_device_area, compute_entry_area, compute_module_bounding_box,
    DEFAULT_PACKING_FACTOR, DEFAULT_ROUTING_OVERHEAD)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PINS = {'d': 'drain', 'g': 'gate', 's': 'source', 'b': 'bulk'}
_DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'silk', 'pdk', 'data')

#: (registry name, bundled catalog, MOSFET device, catalog default W/L in µm)
_PDKS = [
    ('sky130',   'skywater_130_device_details.yaml', 'nmos1p8', (1.0, 0.15)),
    ('gf180mcu', 'gf180mcu_device_details.yaml',     'nmos3p3', (10.0, 0.28)),
    ('sg13g2',   'sg13g2_device_details.yaml',       'nmos_lv', (0.35, 0.34)),
]


def _metre_native_class(name='nfet_test', param_scales=None, extra=None):
    """A factory class whose W/L are declared in SI metres (``pdk_unit: base``)
    — the declared-unit form of the old ``{'W': 4e-6}`` test fakes."""
    from silk.elements.factory import make_device_class
    spec = {
        'PDKDevice': 'test_nfet',
        'paramList': {
            'W': {'minval': -1, 'maxval': -1, 'pdk_unit': 'base'},
            'L': {'minval': -1, 'maxval': -1, 'pdk_unit': 'base'},
            'M': {'minval': 1, 'maxval': -1},
            **(extra or {}),
        },
        'pinList': dict(_PINS),
    }
    return make_device_class(name, spec, is_pdk=True, param_scales=param_scales)


def _dev_si(W, L, M=1, name='mn1'):
    """Metre-native element, author writes metres (no scales)."""
    return _metre_native_class()(name, d='d', g='g', s='s', b='b', W=W, L=L, M=M)


class _FakeModule:
    """Module mock: elem_dict only, no PDK calls."""
    def __init__(self, devices):
        from silk.circuit.ports import ElementType
        self.elem_dict = {
            name: {'device': dev, 'type': ElementType.DEVICE,
                   'typeName': type(dev)._device_type_name}
            for name, dev in devices.items()
        }
        self.bounding_box = None


def _bundled_pdk(catalog, user_units='u'):
    """A generic BasePDK on a bundled catalog with a fake lib path — needs no
    installed PDK. ``user_units='u'`` is what every shipped manifest sets."""
    from silk.pdk.basepdk import BasePDK
    return BasePDK(os.path.join(_DATA, catalog), '/fake/lib.spice',
                   user_units=user_units)


def _module(pdk, name, tmp_path):
    from silk.circuit.modules import Module
    return Module('PDK_test', str(tmp_path), f'{name}.cir', pdk_details=pdk,
                  module_name=name)


def _factory_and_add_element(pdk, device, tmp_path, **params):
    """Same device, same numbers, both construction styles. Returns the two
    modules (each holding exactly one instance 'M1')."""
    cls = getattr(pdk, device)
    a = _module(pdk, 'a', tmp_path)
    with a:
        cls('M1', d='d', g='g', s='s', b='s', **params)
    b = _module(pdk, 'b', tmp_path)
    b.add_element(cls, 'M1', nets={'d': 'd', 'g': 'g', 's': 's', 'b': 's'},
                  parameters=dict(params))
    return a, b


# ---------------------------------------------------------------------------
# compute_device_area
# ---------------------------------------------------------------------------

class TestComputeDeviceArea:

    def test_mosfet_basic(self):
        """MOSFET: area = W_um * L_um * M * packing_factor (default 2.5)."""
        dev = _dev_si(W=4e-6, L=180e-9, M=1)
        area = compute_device_area(dev)
        # W=4um, L=0.18um, M=1, packing=2.5
        expected = 4.0 * 0.18 * 1 * 2.5
        assert area == pytest.approx(expected)

    def test_mosfet_with_multiplier(self):
        dev = _dev_si(W=2e-6, L=180e-9, M=4)
        area = compute_device_area(dev)
        expected = 2.0 * 0.18 * 4 * 2.5
        assert area == pytest.approx(expected)

    def test_mosfet_custom_packing_factor(self):
        dev = _dev_si(W=4e-6, L=180e-9, M=1)
        area = compute_device_area(dev, packing_factor=3.0)
        expected = 4.0 * 0.18 * 1 * 3.0
        assert area == pytest.approx(expected)

    def test_device_without_wlm_returns_default(self):
        """Devices without W/L/M (passives, etc.) get a 1.0 um² default."""
        from silk.elements.factory import make_device_class
        Res = make_device_class('res_test', {
            'PDKDevice': 'test_res',
            'paramList': {'r': {'minval': -1, 'maxval': -1, 'default': 1000}},
            'pinList': {'p': 'plus', 'n': 'minus'},
        }, is_pdk=True)
        dev = Res('r1', p='a', n='b')
        assert compute_device_area(dev) == pytest.approx(1.0)

    def test_returns_positive_float(self):
        dev = _dev_si(W=1e-6, L=100e-9, M=1)
        assert compute_device_area(dev) > 0

    def test_default_multiplier_is_one(self):
        """A device declaring no `m` at all counts once."""
        from silk.elements.factory import make_device_class
        N = make_device_class('nfet_nom', {
            'PDKDevice': 'x',
            'paramList': {'w': {'minval': -1, 'maxval': -1, 'pdk_unit': 'u'},
                          'l': {'minval': -1, 'maxval': -1, 'pdk_unit': 'u'}},
            'pinList': dict(_PINS),
        }, is_pdk=True)
        dev = N('m', d='d', g='g', s='s', b='b', w=2.0, l=1.0)
        assert compute_device_area(dev, packing_factor=1.0) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Fail-loud contract: silk.layout guesses NO unit
# ---------------------------------------------------------------------------

class TestFailLoud:

    def test_untagged_length_raises(self):
        """A W/L with no pdk_unit is a catalog defect, not 'probably µm'."""
        from silk.elements.factory import make_device_class
        from silk.pdk.basepdk import ValidationError
        N = make_device_class('nfet_untagged', {
            'PDKDevice': 'x',
            'paramList': {'w': {'minval': -1, 'maxval': -1},
                          'l': {'minval': -1, 'maxval': -1}},
            'pinList': dict(_PINS),
        }, is_pdk=True)
        dev = N('m', d='d', g='g', s='s', b='b', w=4.0, l=0.18)
        with pytest.raises(ValidationError, match="no pdk_unit"):
            compute_device_area(dev)

    def test_unknown_prefix_raises(self):
        from silk.elements.factory import make_device_class
        from silk.pdk.basepdk import ValidationError
        N = make_device_class('nfet_badunit', {
            'PDKDevice': 'x',
            'paramList': {'w': {'minval': -1, 'maxval': -1, 'pdk_unit': 'furlong'},
                          'l': {'minval': -1, 'maxval': -1, 'pdk_unit': 'u'}},
            'pinList': dict(_PINS),
        }, is_pdk=True)
        dev = N('m', d='d', g='g', s='s', b='b', w=4.0, l=0.18)
        with pytest.raises(ValidationError, match="unknown pdk_unit 'furlong'"):
            compute_device_area(dev)

    def test_bare_params_dict_is_not_a_device(self):
        """The retired SI-metres branch: an object with only a `params` dict
        is an unknown shape and must not be read as metres."""
        class _Legacy:
            params = {'W': 4e-6, 'L': 180e-9, 'M': 1}
        with pytest.raises(TypeError, match="neither a factory AnalogElement"):
            compute_device_area(_Legacy())

    def test_pdkdevice_needs_its_pdk(self, tmp_path):
        """An add_element PDKDevice has no scale of its own: the PDK must be
        passed (estimate_layout does so from module.pdk_details)."""
        pdk = _bundled_pdk('skywater_130_device_details.yaml')
        b = _module(pdk, 'b', tmp_path)
        b.add_element(pdk.nmos1p8, 'M1',
                      nets={'d': 'd', 'g': 'g', 's': 's', 'b': 's'},
                      parameters={'w': 16.0, 'l': 0.5})
        dev = b.elem_dict['M1']['device']
        with pytest.raises(TypeError, match="pass the PDK"):
            compute_device_area(dev)
        assert compute_device_area(dev, pdk=pdk) == pytest.approx(16 * 0.5 * 2.5)
        assert pdk.device_geometry_um(dev) == pytest.approx((16.0, 0.5, 1.0))

    def test_submodule_entry_keeps_placeholder(self):
        """A sub-module instance is a KNOWN shape with no geometry here — it
        keeps the 1 µm² placeholder instead of tripping the unknown-shape
        error."""
        from silk.circuit.ports import ElementType
        entry = {'device': object(), 'type': ElementType.MODULE, 'typeName': 'sub'}
        assert compute_entry_area(entry) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_module_bounding_box
# ---------------------------------------------------------------------------

class TestComputeModuleBoundingBox:

    def test_two_identical_fets(self):
        """Two 4um/0.18um FETs, routing_overhead=2.0."""
        dev1 = _dev_si(W=4e-6, L=180e-9, M=1)
        dev2 = _dev_si(W=4e-6, L=180e-9, M=1, name='mn2')
        mod = _FakeModule({'mn1': dev1, 'mn2': dev2})
        w, h = compute_module_bounding_box(mod, routing_overhead=2.0)
        per_dev = 4.0 * 0.18 * 1 * 2.5      # 1.8 um²
        total = 2 * per_dev * 2.0            # two devices * overhead
        L_char = math.sqrt(total)
        assert w == pytest.approx(L_char)
        assert h == pytest.approx(L_char)

    def test_square_approximation(self):
        """Phase 1 always returns a square (w == h)."""
        dev = _dev_si(W=2e-6, L=180e-9, M=2)
        mod = _FakeModule({'mn1': dev})
        w, h = compute_module_bounding_box(mod)
        assert w == pytest.approx(h)

    def test_empty_module_returns_minimum(self):
        """Empty module doesn't crash; returns a small floor value."""
        mod = _FakeModule({})
        w, h = compute_module_bounding_box(mod)
        assert w > 0
        assert h > 0

    def test_sets_bounding_box_on_module(self):
        """The function writes the result back to module.bounding_box."""
        dev = _dev_si(W=4e-6, L=180e-9, M=1)
        mod = _FakeModule({'mn1': dev})
        compute_module_bounding_box(mod)
        assert mod.bounding_box is not None
        assert len(mod.bounding_box) == 2

    def test_routing_overhead_default_is_two(self):
        """Default routing_overhead=2.0."""
        dev = _dev_si(W=4e-6, L=180e-9, M=1)
        mod_a = _FakeModule({'mn1': dev})
        mod_b = _FakeModule({'mn1': dev})
        a = compute_module_bounding_box(mod_a, routing_overhead=2.0)
        b = compute_module_bounding_box(mod_b)
        assert a == pytest.approx(b)
        assert DEFAULT_ROUTING_OVERHEAD == 2.0


# ---------------------------------------------------------------------------
# Factory-generated elements (``pdk.nmos1p8(...)`` / ``with mod:`` style)
#
# Regression for the 1e12 footprint bug found by the TASK-11.3 walkthrough:
# factory instances store parameters in the AUTHOR's unit (microns by default)
# but fell through to the SI-metres branch, so a w=16 device was sized as
# 16 metres wide. These build the class straight from the bundled catalog
# spec, so they need no installed PDK library.
# ---------------------------------------------------------------------------

def _sky130_nmos_spec():
    import yaml
    with open(os.path.join(_DATA, 'skywater_130_device_details.yaml')) as f:
        return yaml.safe_load(f)['nmos1p8']


class TestFactoryElementArea:

    def test_factory_micron_pdk_author_writes_microns(self):
        """sky130 (pdk_unit u), author writes microns: scale 1, unit 1."""
        from silk.elements.factory import make_device_class
        Nmos = make_device_class('nmos1p8', _sky130_nmos_spec(), is_pdk=True,
                                 param_scales={'w': 1.0, 'l': 1.0})
        dev = Nmos('M1', d='d', g='g', s='s', b='s', w=16.0, l=0.5)
        assert compute_device_area(dev) == pytest.approx(16.0 * 0.5 * 2.5)

    def test_factory_catalog_defaults_are_pdk_units_unscaled(self):
        """A catalog default (w=1.0 l=0.15 µm) is never scaled, even when the
        class carries a non-identity author→PDK factor."""
        from silk.elements.factory import make_device_class
        Nmos = make_device_class('nmos1p8', _sky130_nmos_spec(), is_pdk=True,
                                 param_scales={'w': 1e6, 'l': 1e6})  # user_units='si'
        dev = Nmos('M1', d='d', g='g', s='s', b='s')
        assert compute_device_area(dev) == pytest.approx(1.0 * 0.15 * 2.5)

    def test_factory_si_author_on_micron_pdk(self):
        """user_units='si': author writes metres, class scales 1e6 to µm."""
        from silk.elements.factory import make_device_class
        Nmos = make_device_class('nmos1p8', _sky130_nmos_spec(), is_pdk=True,
                                 param_scales={'w': 1e6, 'l': 1e6})
        dev = Nmos('M1', d='d', g='g', s='s', b='s', w=16e-6, l=0.5e-6)
        assert compute_device_area(dev) == pytest.approx(16.0 * 0.5 * 2.5)

    def test_factory_metre_native_pdk_author_writes_microns(self):
        """GF180MCU-style catalog (pdk_unit base = metres), author writes
        microns: the class scales 1e-6 to metres, the accessor brings the
        metres back to µm. Same source, same footprint as on sky130."""
        Nmos = _metre_native_class(param_scales={'W': 1e-6, 'L': 1e-6})
        dev = Nmos('M1', d='d', g='g', s='s', b='s', W=16.0, L=0.5, M=2)
        assert compute_device_area(dev) == pytest.approx(16.0 * 0.5 * 2 * 2.5)

    def test_factory_metre_native_defaults_unscaled(self):
        """A metre-valued catalog default lands in µm² without the class
        scale being applied to it."""
        from silk.elements.factory import make_device_class
        spec = {
            'PDKDevice': 'nfet_03v3',
            'paramList': {
                'w': {'minval': -1, 'maxval': -1, 'default': 1e-05,
                      'pdk_unit': 'base'},
                'l': {'minval': -1, 'maxval': -1, 'default': 2.8e-07,
                      'pdk_unit': 'base'},
                'm': {'minval': 1, 'maxval': -1},
            },
            'pinList': dict(_PINS),
        }
        Nmos = make_device_class('nfet_03v3', spec, is_pdk=True,
                                 param_scales={'w': 1e-6, 'l': 1e-6})
        dflt = Nmos('M2', d='d', g='g', s='s', b='s')
        assert compute_device_area(dflt) == pytest.approx(10.0 * 0.28 * 2.5)

    def test_factory_pdk_units_mode_has_no_scales(self):
        """user_units='pdk' on a metre-native PDK: no class scales, the author
        already writes metres, the pdk_unit alone converts to µm."""
        Nmos = _metre_native_class(param_scales={})
        dev = Nmos('M1', d='d', g='g', s='s', b='s', W=16e-6, L=0.5e-6)
        assert compute_device_area(dev) == pytest.approx(16.0 * 0.5 * 2.5)

    def test_factory_spice_literal_geometry_has_no_area(self):
        """A string parameter is a SPICE literal, not a quantity — fall back."""
        from silk.elements.factory import make_device_class
        Nmos = make_device_class('nmos1p8', _sky130_nmos_spec(), is_pdk=True,
                                 param_scales={'w': 1.0, 'l': 1.0})
        dev = Nmos('M1', d='d', g='g', s='s', b='s', w='wn', l=0.5)
        assert compute_device_area(dev) == pytest.approx(1.0)

    def test_assigned_parameter_is_caller_supplied(self):
        """``mn.w = 5`` after construction is the author's unit, like w=5 at
        construction — the accessor scales it the same way."""
        Nmos = _metre_native_class(param_scales={'W': 1e-6, 'L': 1e-6})
        dev = Nmos('M1', d='d', g='g', s='s', b='s', W=1.0, L=1.0)
        dev.W = 16.0
        dev.L = 0.5
        assert compute_device_area(dev, packing_factor=1.0) == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# AC3: factory / add_element equivalence on sky130 AND the metre-native PDKs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name, catalog, device, default_wl', _PDKS,
                         ids=[p[0] for p in _PDKS])
class TestConstructionStylesAgree:
    """The two construction styles of the same device must estimate the same
    footprint — the exact discrepancy the walkthrough hit — on a µm-native and
    on metre-native PDKs. Runs on the bundled catalogs; no installed PDK."""

    def test_caller_supplied_geometry(self, name, catalog, device, default_wl, tmp_path):
        pdk = _bundled_pdk(catalog)                          # user_units='u'
        a, b = _factory_and_add_element(pdk, device, tmp_path, w=16.0, l=0.5, m=2)
        area_a = compute_device_area(a.elem_dict['M1']['device'], pdk=pdk)
        area_b = compute_device_area(b.elem_dict['M1']['device'], pdk=pdk)
        assert area_a == pytest.approx(area_b) == pytest.approx(16.0 * 0.5 * 2 * 2.5)
        # And through the PDK-bound accessor, in canonical µm.
        assert pdk.device_geometry_um(a.elem_dict['M1']['device']) == pytest.approx((16.0, 0.5, 2.0))
        assert pdk.device_geometry_um(b.elem_dict['M1']['device']) == pytest.approx((16.0, 0.5, 2.0))

    def test_catalog_defaults(self, name, catalog, device, default_wl, tmp_path):
        """Omitted W/L fall back to the catalog default — in the PDK's own
        unit (metres for GF180MCU/SG13G2) — and both styles agree."""
        pdk = _bundled_pdk(catalog)
        a, b = _factory_and_add_element(pdk, device, tmp_path)
        W, L = default_wl
        for m in (a, b):
            assert compute_device_area(m.elem_dict['M1']['device'], pdk=pdk) \
                == pytest.approx(W * L * 2.5)

    def test_si_author(self, name, catalog, device, default_wl, tmp_path):
        """user_units='si': the author writes metres on every PDK; the
        footprint is still 16 µm × 0.5 µm."""
        pdk = _bundled_pdk(catalog, user_units='si')
        a, b = _factory_and_add_element(pdk, device, tmp_path, w=16e-6, l=0.5e-6)
        for m in (a, b):
            assert compute_device_area(m.elem_dict['M1']['device'], pdk=pdk) \
                == pytest.approx(16.0 * 0.5 * 2.5)

    def test_estimate_layout_end_to_end(self, name, catalog, device, default_wl, tmp_path):
        """The public entry point resolves the PDK from the module itself."""
        from silk.layout import estimate_layout, RoutingModel
        pdk = _bundled_pdk(catalog)
        a, b = _factory_and_add_element(pdk, device, tmp_path, w=16.0, l=0.5)
        rm = RoutingModel(0.1, 0.04, 1e-3, 0.14, 10.0)
        ea, eb = estimate_layout(a, rm), estimate_layout(b, rm)
        assert ea.devices['M1'].area == pytest.approx(20.0)
        assert eb.devices['M1'].area == pytest.approx(20.0)
        assert ea.bounding_box == pytest.approx(eb.bounding_box)
        assert a.elem_dict['M1']['device'].estimated_area == pytest.approx(20.0)


@pytest.mark.parametrize('name', ['sky130', 'gf180mcu'])
def test_installed_pdk_styles_agree(name, tmp_path):
    """Same check through load_pdk() on a real install (manifest user_units)."""
    from tests._env import pdk_available
    if not pdk_available(name):
        pytest.skip(f'needs the {name} PDK installed under $PDK_ROOT')
    from silk import load_pdk
    pdk = load_pdk(name)
    device = 'nmos1p8' if name == 'sky130' else 'nmos3p3'
    a, b = _factory_and_add_element(pdk, device, tmp_path, w=16.0, l=0.5, m=2)
    area_a = compute_device_area(a.elem_dict['M1']['device'], pdk=pdk)
    area_b = compute_device_area(b.elem_dict['M1']['device'], pdk=pdk)
    assert area_a == pytest.approx(area_b) == pytest.approx(16.0 * 0.5 * 2 * 2.5)


# ---------------------------------------------------------------------------
# TASK-28: a DECLARED-but-unresolved m/M must never be silently treated as
# 1 — geometry returns None, exactly like an unresolved W/L (compute_*_area
# then falls back to the 1 um^2 placeholder, same as any other no-geometry
# device). An ABSENT m/M (no such parameter in the catalog at all) keeps
# M=1 — that IS SPICE truth, not a guess. Reviewer repro: sky130 W=16um,
# L=0.5um, packing_factor=2.5 — unresolved m must NOT report 20 um^2;
# m=4 must report 80 um^2.
# ---------------------------------------------------------------------------

class TestUnresolvedMultiplier:

    def _sky130_pdk(self):
        return _bundled_pdk('skywater_130_device_details.yaml')

    # -- AC2/AC4: an assigned Variable m is used exactly --

    def test_factory_unassigned_modulevariable_m_has_no_area(self, tmp_path):
        """W=16, L=0.5: an unresolved m must NOT report 20 um^2 (2.5x)."""
        from silk.circuit.variable import Variable
        pdk = self._sky130_pdk()
        mv = Variable('m', bounds=(1, 16))
        mod = _module(pdk, 'a', tmp_path)
        with mod:
            pdk.nmos1p8('M1', d='d', g='g', s='s', b='s', w=16.0, l=0.5, m=mv)
        dev = mod.elem_dict['M1']['device']
        assert pdk.device_geometry_um(dev) is None
        assert compute_device_area(dev, pdk=pdk) == pytest.approx(1.0)

    def test_factory_assigning_modulevariable_resolves_to_eighty(self, tmp_path):
        """Same Variable, now assigned 4 -> reports 80 um^2 exactly."""
        from silk.circuit.variable import Variable
        pdk = self._sky130_pdk()
        mv = Variable('m', bounds=(1, 16))
        mod = _module(pdk, 'a', tmp_path)
        with mod:
            pdk.nmos1p8('M1', d='d', g='g', s='s', b='s', w=16.0, l=0.5, m=mv)
        mv.value = 4
        dev = mod.elem_dict['M1']['device']
        assert pdk.device_geometry_um(dev) == pytest.approx((16.0, 0.5, 4.0))
        assert compute_device_area(dev, pdk=pdk) == pytest.approx(80.0)

    def test_add_element_unassigned_modulevariable_m_has_no_area(self, tmp_path):
        from silk.circuit.variable import Variable
        pdk = self._sky130_pdk()
        mv = Variable('m', bounds=(1, 16))
        mod = _module(pdk, 'b', tmp_path)
        mod.add_element(pdk.nmos1p8, 'M1',
                        nets={'d': 'd', 'g': 'g', 's': 's', 'b': 's'},
                        parameters={'w': 16.0, 'l': 0.5, 'm': mv})
        dev = mod.elem_dict['M1']['device']
        assert pdk.device_geometry_um(dev) is None
        assert compute_device_area(dev, pdk=pdk) == pytest.approx(1.0)

    def test_add_element_assigning_modulevariable_resolves_to_eighty(self, tmp_path):
        from silk.circuit.variable import Variable
        pdk = self._sky130_pdk()
        mv = Variable('m', bounds=(1, 16))
        mod = _module(pdk, 'b', tmp_path)
        mod.add_element(pdk.nmos1p8, 'M1',
                        nets={'d': 'd', 'g': 'g', 's': 's', 'b': 's'},
                        parameters={'w': 16.0, 'l': 0.5, 'm': mv})
        mv.value = 4
        dev = mod.elem_dict['M1']['device']
        assert pdk.device_geometry_um(dev) == pytest.approx((16.0, 0.5, 4.0))
        assert compute_device_area(dev, pdk=pdk) == pytest.approx(80.0)

    # -- AC5: a non-numeric m parameter reference (SPICE literal / symbolic
    #    reference, not silk's own Variable) — both construction styles.

    def test_factory_non_numeric_m_reference_has_no_area(self, tmp_path):
        pdk = self._sky130_pdk()
        mod = _module(pdk, 'a', tmp_path)
        with mod:
            pdk.nmos1p8('M1', d='d', g='g', s='s', b='s',
                       w=16.0, l=0.5, m='m_param')
        dev = mod.elem_dict['M1']['device']
        assert pdk.device_geometry_um(dev) is None
        assert compute_device_area(dev, pdk=pdk) == pytest.approx(1.0)

    def test_add_element_non_numeric_m_reference_has_no_area(self, tmp_path):
        pdk = self._sky130_pdk()
        mod = _module(pdk, 'b', tmp_path)
        mod.add_element(pdk.nmos1p8, 'M1',
                        nets={'d': 'd', 'g': 'g', 's': 's', 'b': 's'},
                        parameters={'w': 16.0, 'l': 0.5, 'm': 'm_param'})
        dev = mod.elem_dict['M1']['device']
        assert pdk.device_geometry_um(dev) is None
        assert compute_device_area(dev, pdk=pdk) == pytest.approx(1.0)

    # -- AC1: an ABSENT m/M (catalog declares no such parameter at all)
    #    keeps multiplicity 1 — add_element style; the factory-style case
    #    is TestComputeDeviceArea.test_default_multiplier_is_one.

    def test_add_element_absent_multiplier_defaults_to_one(self, tmp_path):
        import yaml
        from silk.pdk.basepdk import BasePDK
        spec = {
            'nfet_nom': {
                'PDKDevice': 'test_nfet',
                'paramList': {
                    'w': {'minval': -1, 'maxval': -1, 'pdk_unit': 'u'},
                    'l': {'minval': -1, 'maxval': -1, 'pdk_unit': 'u'},
                },
                'pinList': dict(_PINS),
            },
        }
        yaml_path = tmp_path / 'nom.yaml'
        yaml_path.write_text(yaml.safe_dump(spec))
        pdk = BasePDK(str(yaml_path), '/fake/lib.spice', user_units='u')
        mod = _module(pdk, 'b', tmp_path)
        mod.add_element(pdk.nfet_nom, 'M1',
                        nets={'d': 'd', 'g': 'g', 's': 's', 'b': 's'},
                        parameters={'w': 2.0, 'l': 1.0})
        dev = mod.elem_dict['M1']['device']
        assert pdk.device_geometry_um(dev) == pytest.approx((2.0, 1.0, 1.0))
        assert compute_device_area(dev, pdk=pdk, packing_factor=1.0) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# AC6: estimate_layout signature defaults agree with the documented ones
# ---------------------------------------------------------------------------

class TestEstimateLayoutSignatureDefaults:

    def test_signature_defaults_match_documented_values(self):
        """inspect.signature must report the SAME numbers the docstring
        (and every caller relying on 'omit the kwarg') already promises."""
        import inspect
        from silk.layout.estimator import estimate_layout
        sig = inspect.signature(estimate_layout)
        assert sig.parameters['packing_factor'].default == DEFAULT_PACKING_FACTOR == 2.5
        assert sig.parameters['routing_overhead'].default == DEFAULT_ROUTING_OVERHEAD == 2.0

    def test_omitting_and_explicit_none_are_equivalent(self, tmp_path):
        """Both paths to 'use the default' land on the same effective value."""
        from silk.layout import estimate_layout, RoutingModel
        pdk = _bundled_pdk('skywater_130_device_details.yaml')
        mod = _module(pdk, 'sig', tmp_path)
        with mod:
            pdk.nmos1p8('M1', d='d', g='g', s='s', b='s', w=16.0, l=0.5)
        rm = RoutingModel(0.1, 0.04, 1e-3, 0.14, 10.0)
        omitted = estimate_layout(mod, rm)
        explicit_none = estimate_layout(mod, rm, packing_factor=None,
                                        routing_overhead=None)
        assert omitted.packing_factor == explicit_none.packing_factor == 2.5
        assert omitted.routing_overhead == explicit_none.routing_overhead == 2.0


# ---------------------------------------------------------------------------
# AC7/AC8: public packing_factor override on estimate_layout
# ---------------------------------------------------------------------------

class TestPackingFactorOverride:

    def _one_device_module(self, tmp_path, **params):
        pdk = _bundled_pdk('skywater_130_device_details.yaml')
        mod = _module(pdk, 'pf', tmp_path)
        with mod:
            pdk.nmos1p8('M1', d='d', g='g', s='s', b='s', **params)
        return mod

    def _rm(self):
        from silk.layout import RoutingModel
        return RoutingModel(0.1, 0.04, 1e-3, 0.14, 10.0)

    def test_default_is_documented_heuristic(self, tmp_path):
        """W=16 µm, L=0.5 µm, M=1 → 20 µm² at the default 2.5."""
        from silk.layout import estimate_layout
        assert DEFAULT_PACKING_FACTOR == 2.5
        est = estimate_layout(self._one_device_module(tmp_path, w=16.0, l=0.5),
                              self._rm())
        assert est.devices['M1'].area == pytest.approx(20.0)
        assert est.packing_factor == 2.5
        assert est.routing_overhead == 2.0

    def test_override_to_one(self, tmp_path):
        """... and 8 µm² at packing_factor=1.0."""
        from silk.layout import estimate_layout
        est = estimate_layout(self._one_device_module(tmp_path, w=16.0, l=0.5),
                              self._rm(), packing_factor=1.0)
        assert est.devices['M1'].area == pytest.approx(8.0)
        assert est.packing_factor == 1.0

    def test_keyword_only(self, tmp_path):
        from silk.layout import estimate_layout
        with pytest.raises(TypeError):
            estimate_layout(self._one_device_module(tmp_path, w=16.0, l=0.5),
                            self._rm(), None, 1.0)

    def test_propagates_to_bounding_box(self, tmp_path):
        """The SAME factor drives per-device area and the module box:
        box = sqrt(Σ area × routing_overhead)."""
        from silk.layout import estimate_layout
        mod = self._one_device_module(tmp_path, w=16.0, l=0.5)
        est = estimate_layout(mod, self._rm(), packing_factor=1.0,
                              routing_overhead=3.0)
        assert est.devices['M1'].area == pytest.approx(8.0)
        assert est.bounding_box[0] == pytest.approx(math.sqrt(8.0 * 3.0))
        assert mod.bounding_box == est.bounding_box
        assert est.routing_overhead == 3.0

    @pytest.mark.parametrize('bad', [0, -1.0, float('inf'), float('nan'), 'x', None])
    def test_rejects_invalid(self, tmp_path, bad):
        from silk.layout import estimate_layout
        mod = self._one_device_module(tmp_path, w=16.0, l=0.5)
        if bad is None:
            # None means "use the default" — never an error.
            assert estimate_layout(mod, self._rm(), packing_factor=None).packing_factor == 2.5
            return
        with pytest.raises(ValueError, match='packing_factor'):
            estimate_layout(mod, self._rm(), packing_factor=bad)
        with pytest.raises(ValueError, match='routing_overhead'):
            estimate_layout(mod, self._rm(), routing_overhead=bad)

    def test_low_level_overrides_still_work(self):
        dev = _dev_si(W=16e-6, L=0.5e-6)
        assert compute_device_area(dev, 1.0) == pytest.approx(8.0)
        mod = _FakeModule({'mn1': dev})
        w, _ = compute_module_bounding_box(mod, 1.0, 1.0)
        assert w == pytest.approx(math.sqrt(8.0))
        with pytest.raises(ValueError):
            compute_device_area(dev, packing_factor=0)

    def test_captured_estimate_has_no_provenance(self, tmp_path):
        from silk.layout import capture_layout, estimate_layout
        mod = self._one_device_module(tmp_path, w=16.0, l=0.5)
        estimate_layout(mod, self._rm(), packing_factor=1.0)
        cap = capture_layout(mod)
        assert cap.packing_factor is None
        assert cap.devices['M1'].area == pytest.approx(8.0)

    def test_report_records_and_round_trips_provenance(self, tmp_path):
        from silk.layout import estimate_layout
        from silk.circuit.report import generate_report, CircuitReport
        mod = self._one_device_module(tmp_path, w=16.0, l=0.5)
        est = estimate_layout(mod, self._rm(), packing_factor=1.5)
        report = generate_report(mod, layout_estimate=est)
        md = report.to_markdown() if hasattr(report, 'to_markdown') else None
        out = str(tmp_path / 'r.json')
        report.save_json(out)
        loaded = CircuitReport.load_json(out)
        assert loaded.layout_estimate.packing_factor == 1.5
        assert loaded.layout_estimate.routing_overhead == 2.0
        md_path = str(tmp_path / 'r.md')
        report.save(md_path)
        text = open(md_path).read()
        assert 'packing factor 1.5' in text
        assert 'not' in text and 'layout truth' in text


class TestChildModuleFootprints:
    """Child estimates are caller-owned micrometre footprints, not W/L data."""

    def _child(self, pdk, tmp_path, count=1):
        child = _module(pdk, 'child', tmp_path)
        child.define_module_ports({'d': 'out', 'g': 'in', 's': 'ground'})
        for i in range(count):
            child.add(pdk.nmos1p8(f'M{i}', d='d', g='g', s='s', b='s', w=16.0, l=0.5, m=1))
        return child

    def _parent(self, pdk, child, tmp_path):
        parent = _module(pdk, 'parent', tmp_path)
        parent.add_module(child, 'x1', nets={'d': 'out', 'g': 'in', 's': 'gnd'})
        return parent

    def _estimate(self, module, **kwargs):
        from silk.layout import estimate_layout, RoutingModel
        return estimate_layout(module, RoutingModel(0.1, 0.04, 1e-3, 0.14, 10),
                               **kwargs)

    @pytest.mark.parametrize('packing_factor', [1.0, 7.0])
    @pytest.mark.parametrize('routing_overhead', [1.0, 2.0])
    def test_estimated_child_footprint_is_reused(
            self, tmp_path, packing_factor, routing_overhead):
        pdk = _bundled_pdk('skywater_130_device_details.yaml')
        child = self._child(pdk, tmp_path, count=100)
        child_est = self._estimate(child)
        assert math.prod(child_est.bounding_box) == pytest.approx(4000)
        parent = self._parent(pdk, child, tmp_path)
        instance = parent.elem_dict['x1']['device']
        assert instance is child
        geometry_before = [pdk.device_geometry_um(e['device'])
                           for e in instance.elem_dict.values()]

        est = self._estimate(parent, packing_factor=packing_factor,
                             routing_overhead=routing_overhead)

        assert est.devices['x1'].area == pytest.approx(4000)
        assert instance.estimated_area == pytest.approx(4000)
        assert math.prod(est.bounding_box) == pytest.approx(4000 * routing_overhead)
        assert parent.bounding_box == est.bounding_box
        assert est.bounding_box[0] == est.bounding_box[1]
        assert instance.bounding_box == child.bounding_box == child_est.bounding_box
        assert [pdk.device_geometry_um(e['device'])
                for e in instance.elem_dict.values()] == geometry_before
        assert [e['device'].estimated_area for e in instance.elem_dict.values()] \
            == pytest.approx([20.0] * 100)

    @pytest.mark.parametrize('catalog', [
        'skywater_130_device_details.yaml', 'gf180mcu_device_details.yaml'])
    @pytest.mark.parametrize('user_units', ['u', 'si'])
    def test_rectangular_box_is_already_in_micrometres(
            self, tmp_path, catalog, user_units):
        pdk = _bundled_pdk(catalog, user_units=user_units)
        child = _module(pdk, 'child', tmp_path)
        child.bounding_box = (20, 200.0)
        parent = _module(pdk, 'parent', tmp_path)
        parent.add_module(child, 'x1')
        est = self._estimate(parent, packing_factor=7, routing_overhead=1)
        assert est.devices['x1'].area == pytest.approx(4000)
        assert math.prod(est.bounding_box) == pytest.approx(4000)
        assert parent.elem_dict['x1']['device'].bounding_box == (20, 200.0)

    def test_each_instance_and_local_transistor_count_once(self, tmp_path):
        pdk = _bundled_pdk('skywater_130_device_details.yaml')
        child = self._child(pdk, tmp_path)
        self._estimate(child)  # 40 µm², including its own routing overhead.
        parent = self._parent(pdk, child, tmp_path)
        parent.add_module(child, 'x2', nets={'d': 'out', 'g': 'in', 's': 'gnd'})
        parent.add(pdk.nmos1p8('local', d='out', g='in', s='gnd', b='gnd', w=16.0, l=0.5, m=2))
        est = self._estimate(parent, packing_factor=3, routing_overhead=2)
        assert {name: device.area for name, device in est.devices.items()} \
            == pytest.approx({'x1': 40, 'x2': 40, 'local': 48})
        assert math.prod(est.bounding_box) == pytest.approx(256)
        first = parent.elem_dict['x1']['device']
        second = parent.elem_dict['x2']['device']
        assert first is second is child
        assert first.estimated_area == second.estimated_area == pytest.approx(40)
        assert first.bounding_box == second.bounding_box == child.bounding_box

    def test_explicit_bottom_up_estimation_of_owned_instances(self, tmp_path):
        pdk = _bundled_pdk('skywater_130_device_details.yaml')
        child = self._child(pdk, tmp_path)
        middle = self._parent(pdk, child, tmp_path)
        top = _module(pdk, 'top', tmp_path)
        top.add_module(middle, 'middle')
        owned_middle = top.elem_dict['middle']['device']
        owned_child = owned_middle.elem_dict['x1']['device']
        assert owned_middle is middle and owned_child is child

        self._estimate(owned_child)
        middle_est = self._estimate(owned_middle)
        top_est = self._estimate(top)

        assert math.prod(owned_child.bounding_box) == pytest.approx(40)
        assert middle_est.devices['x1'].area == pytest.approx(40)
        assert math.prod(owned_middle.bounding_box) == pytest.approx(80)
        assert top_est.devices['middle'].area == pytest.approx(80)
        assert owned_middle.estimated_area == pytest.approx(80)
        assert math.prod(top_est.bounding_box) == pytest.approx(160)
        assert child.bounding_box == owned_child.bounding_box
        assert middle.bounding_box == owned_middle.bounding_box

    @pytest.mark.parametrize('missing_attribute', [False, True])
    def test_missing_estimate_stays_placeholder_until_live_child_is_estimated(
            self, tmp_path, missing_attribute):
        pdk = _bundled_pdk('skywater_130_device_details.yaml')
        child = self._child(pdk, tmp_path)
        parent = self._parent(pdk, child, tmp_path)
        instance = parent.elem_dict['x1']['device']
        if missing_attribute:
            del instance.bounding_box
        est = self._estimate(parent)
        assert est.devices['x1'].area == instance.estimated_area == 1.0
        assert math.prod(est.bounding_box) == pytest.approx(2)
        assert getattr(instance, 'bounding_box', None) is None
        transistor = instance.elem_dict['M0']['device']
        assert getattr(transistor, 'estimated_area', None) is None

        self._estimate(child)  # Late estimates are visible through the reference.
        est = self._estimate(parent)
        assert est.devices['x1'].area == pytest.approx(40)
        assert math.prod(est.bounding_box) == pytest.approx(80)

    def test_small_child_preserves_parent_area_floor(self, tmp_path):
        pdk = _bundled_pdk('skywater_130_device_details.yaml')
        child = self._child(pdk, tmp_path)
        child.bounding_box = (0.1, 0.2)
        parent = self._parent(pdk, child, tmp_path)
        est = self._estimate(parent, routing_overhead=1)
        assert est.devices['x1'].area == pytest.approx(0.02)
        assert est.bounding_box == (1.0, 1.0)

    @pytest.mark.parametrize('bbox', [
        (), (1,), (1, 2, 3), 42, '12', b'12', bytearray(b'12'),
        {'width': 1, 'height': 2}, {1, 2},
        ('1', 2), (1, None), ([1], 2), (True, 2), (1, False), (1 + 2j, 2),
        (0, 2), (1, 0), (-1, 2), (-1, -2),
        (float('nan'), 2), (1, float('nan')),
        (float('inf'), 2), (1, float('-inf')),
        (1e308, 1e308), (1e-300, 1e-300), (10 ** 400, 1),
    ])
    def test_invalid_supplied_box_raises(self, tmp_path, bbox):
        pdk = _bundled_pdk('skywater_130_device_details.yaml')
        parent = self._parent(pdk, self._child(pdk, tmp_path), tmp_path)
        entry = parent.elem_dict['x1']
        entry['device'].bounding_box = bbox
        for compute in (lambda: compute_entry_area(entry, pdk=pdk),
                        lambda: compute_module_bounding_box(parent),
                        lambda: self._estimate(parent)):
            message = 'bounding_box.*finite.*positive.*micrometres'
            with pytest.raises(ValueError, match=message) as exc:
                compute()
            assert repr(bbox) in str(exc.value)
            assert 'Estimate the child' in str(exc.value)


class TestRecursiveFootprints:
    _child = TestChildModuleFootprints._child
    _parent = TestChildModuleFootprints._parent
    _estimate = TestChildModuleFootprints._estimate

    def test_large_child_and_two_levels(self, tmp_path):
        pdk = _bundled_pdk('skywater_130_device_details.yaml')
        parent = self._parent(pdk, self._child(pdk, tmp_path, 100), tmp_path)
        child = parent.elem_dict['x1']['device']
        est = self._estimate(parent, children='estimate')
        assert est.devices['x1'].area == pytest.approx(4000)
        assert math.prod(est.bounding_box) == pytest.approx(8000)
        assert est.child_footprints['x1'] == (est.devices['x1'].area, 'estimated')
        assert all(n.estimated_length is None and n.estimated_r is None
                   and n.estimated_c is None and n.estimated_width is None
                   for n in child.net_dict.values())
        top = _module(pdk, 'top', tmp_path)
        top.add_module(parent, 'middle')
        assert math.prod(self._estimate(top, children='estimate').bounding_box) == pytest.approx(16000)

    def test_shared_module_computed_once_and_cycle_path(self, tmp_path, monkeypatch):
        from silk.circuit.ports import ElementType
        from silk.exceptions import CircuitError
        import silk.layout.area_model as areas
        pdk = _bundled_pdk('skywater_130_device_details.yaml')
        parent = self._parent(pdk, self._child(pdk, tmp_path), tmp_path)
        child = parent.elem_dict['x1']['device']
        middle = _module(pdk, 'middle', tmp_path)
        middle._element_mgr.elem_dict['shared'] = dict(parent.elem_dict['x1'])
        parent._element_mgr.elem_dict['x2'] = dict(parent.elem_dict['x1'])
        parent._element_mgr.elem_dict['middle'] = {'type': ElementType.MODULE, 'device': middle}
        calls = []
        original = areas.compute_device_area
        def counted(device, *args, **kwargs):
            calls.append(device)
            return original(device, *args, **kwargs)
        monkeypatch.setattr(areas, 'compute_device_area', counted)
        est = self._estimate(parent, children='estimate')
        assert len(calls) == 1
        assert math.prod(est.bounding_box) == pytest.approx(320)
        child._element_mgr.elem_dict['back'] = {'type': ElementType.MODULE, 'device': parent}
        with pytest.raises(CircuitError, match='x1.*back'):
            self._estimate(parent, children='estimate')

    @pytest.mark.parametrize('mode', ['reuse', 'estimate'])
    def test_fixed_and_clear(self, tmp_path, mode):
        from silk.layout import set_footprint
        pdk = _bundled_pdk('skywater_130_device_details.yaml')
        child = self._child(pdk, tmp_path)
        set_footprint(child, (100, 50))
        parent = self._parent(pdk, child, tmp_path)
        instance = parent.elem_dict['x1']['device']
        # Invalid descendants prove a fixed child is a leaf.
        instance._element_mgr.elem_dict['broken'] = {'device': object()}
        est = self._estimate(parent, children=mode)
        assert est.child_footprints == {'x1': (5000, 'fixed')}
        assert instance.bounding_box == (100, 50)
        assert instance is child
        # A fixed root still estimates its own entries; the live child sees
        # the invalid device too. Remove the sentinel before checking sizing.
        with pytest.raises(TypeError, match='cannot read its geometry'):
            self._estimate(child, children=mode)
        del child._element_mgr.elem_dict['broken']
        own = self._estimate(child, children=mode)
        assert own.bounding_box == (100, 50)
        assert own.nets and all(n.length > 0 for n in own.nets.values())
        set_footprint(child, None)
        assert child.bounding_box is None
        assert math.prod(self._estimate(child).bounding_box) == pytest.approx(40)

    @pytest.mark.parametrize('box', [(0, 2), (True, 2), '12', (1e308, 1e308), (1, float('nan'))])
    def test_set_footprint_invalid_is_atomic(self, tmp_path, box):
        from silk.layout import set_footprint
        mod = _module(_bundled_pdk('skywater_130_device_details.yaml'), 'm', tmp_path)
        set_footprint(mod, (2, 3))
        with pytest.raises(ValueError):
            set_footprint(mod, box)
        assert mod.bounding_box == (2, 3)

    @pytest.mark.parametrize('mode', ['reuse', 'estimate'])
    @pytest.mark.parametrize('failure', ['box', 'sizing'])
    def test_failure_does_not_mutate_any_level(self, tmp_path, mode, failure):
        from silk.layout import set_footprint
        pdk = _bundled_pdk('skywater_130_device_details.yaml')
        parent = self._parent(pdk, self._child(pdk, tmp_path), tmp_path)
        parent.add_module(self._child(pdk, tmp_path), 'x2',
                          nets={'d': 'out', 'g': 'in', 's': 'gnd'})
        if failure == 'box':
            bad = parent.elem_dict['x2']['device']
            set_footprint(bad, (2, 3))
            bad.bounding_box = (0, 3)
        objects = [parent]
        for e in parent.elem_dict.values():
            c = e['device']
            objects.extend([c, *c.net_dict.values(), *(e['device'] for e in c.elem_dict.values())])
        objects.extend(parent.net_dict.values())
        attrs = ('bounding_box', 'estimated_area', 'estimated_length', 'estimated_width', 'estimated_r', 'estimated_c')
        for index, obj in enumerate(objects):
            for attr in attrs[1:]:
                if hasattr(obj, attr):
                    setattr(obj, attr, float(index + 1))
        before = [[getattr(o, a, None) for a in attrs] for o in objects]
        if failure == 'sizing':
            from silk.layout import estimate_layout, RoutingModel
            calls = 0
            def fail(current, stack):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise ValueError('sizing failed')
                return stack[0]
            run = lambda: estimate_layout(parent, [RoutingModel(0.1, 0.04, 1e-3, 0.14, 10)], children=mode, escalation_rule=fail)
        else:
            run = lambda: self._estimate(parent, children=mode)
        with pytest.raises(ValueError):
            run()
        assert [[getattr(o, a, None) for a in attrs] for o in objects] == before

    def test_provenance_and_no_geometry(self, tmp_path):
        pdk = _bundled_pdk('skywater_130_device_details.yaml')
        child = _module(pdk, 'empty', tmp_path)
        parent = _module(pdk, 'parent', tmp_path)
        parent.add_module(child, 'empty')
        est = self._estimate(parent)
        assert est.child_footprints == {'empty': (1, 'placeholder')}
        assert any(w.reason == 'child_footprint_missing' and w.net_name is None and 'empty' in w.detail for w in est.warnings)
        est = self._estimate(parent, children='estimate')
        assert est.child_footprints == {'empty': (1, 'estimated')}
        assert any(w.reason == 'child_has_no_geometry' and 'empty' in w.detail for w in est.warnings)
        assert self._estimate(parent).child_footprints == {'empty': (1, 'reused')}
        with pytest.raises(ValueError, match='children'):
            self._estimate(parent, children='invalid')

    def test_child_uses_own_pdk(self, tmp_path):
        sky = _bundled_pdk('skywater_130_device_details.yaml')
        si = _bundled_pdk('gf180mcu_device_details.yaml', user_units='si')
        _, child = _factory_and_add_element(si, 'nmos3p3', tmp_path, w=16e-6, l=0.5e-6)
        parent = _module(sky, 'parent', tmp_path)
        parent.add_module(child, 'different')
        est = self._estimate(parent, children='estimate', pdk=sky)
        assert est.devices['different'].area == pytest.approx(40)

    def test_nested_fixed_child_gets_area(self, tmp_path):
        from silk.layout import set_footprint
        pdk = _bundled_pdk('skywater_130_device_details.yaml')
        child = self._child(pdk, tmp_path)
        set_footprint(child, (10, 20))
        middle = self._parent(pdk, child, tmp_path)
        top = _module(pdk, 'top', tmp_path)
        top.add_module(middle, 'middle')
        fixed = top.elem_dict['middle']['device'].elem_dict['x1']['device']
        self._estimate(top, children='estimate')
        assert fixed.estimated_area == 200
        assert fixed.bounding_box == (10, 20)

    def test_invalid_nonfixed_child_box_is_atomic(self, tmp_path):
        pdk = _bundled_pdk('skywater_130_device_details.yaml')
        parent = self._parent(pdk, self._child(pdk, tmp_path), tmp_path)
        child = parent.elem_dict['x1']['device']
        child.bounding_box = (0, 3)
        with pytest.raises(ValueError, match='bounding_box'):
            self._estimate(parent, children='estimate')
        assert parent.bounding_box is None
        assert child.bounding_box == (0, 3)
        assert not hasattr(child, 'estimated_area')

    @pytest.mark.parametrize('behavioral', [False, True])
    def test_geometryless_warning_and_existing_child_nets(self, tmp_path, behavioral):
        pdk = _bundled_pdk('skywater_130_device_details.yaml')
        child = _module(pdk, 'empty', tmp_path)
        child.define_module_ports({'d': 'out', 'g': 'in', 's': 'ground'})
        if behavioral:
            def model_1(ports, t, params):
                ports.d.v = ports.g.v
            child.set_behavioral_model(model_1, execution='vector')
        parent = self._parent(pdk, child, tmp_path)
        instance = parent.elem_dict['x1']['device']
        for net in instance.net_dict.values():
            net.estimated_length, net.estimated_width = 123, 456
            net.estimated_r, net.estimated_c = 789, 10
        est = self._estimate(parent, children='estimate')
        assert any(w.reason == 'child_has_no_geometry' for w in est.warnings)
        for net in instance.net_dict.values():
            assert (net.estimated_length, net.estimated_width, net.estimated_r, net.estimated_c) == (123, 456, 789, 10)

    def test_fixed_root_rejects_overflowing_entry_area(self, tmp_path):
        from silk.layout import set_footprint
        pdk = _bundled_pdk('skywater_130_device_details.yaml')
        mod = self._child(pdk, tmp_path)
        set_footprint(mod, (10, 20))
        with pytest.raises(ValueError, match='area'):
            self._estimate(mod, packing_factor=1e308)
        assert mod.bounding_box == (10, 20)
        assert mod.elem_dict['M0']['device'].estimated_area is None
