"""One device, two build paths, identical bytes (TASK-73).

silk has two supported ways to put a PDK device in a circuit:

* the legacy ``Module.add_element(pdk.nmos1p8, 'M1', nets=…, parameters=…)``
  path, emitted by ``BasePDK.generate_netlist_entry``;
* the factory API ``pdk.nmos1p8('M1', d=…, …, w=…, l=…)``, emitted by the
  generated class's ``to_spice``.

They used to disagree on the *spelling* of the same device: the legacy path
rendered an unscaled float with ``str()`` (``w=4.0`` where the factory wrote
``w=4``) and left a trailing space at end of line. Cosmetic per line, but it
means two build styles of the identical circuit produce non-identical decks —
and deck-diffing, the golden guard and the cross-PDK byte comparisons all lean
on byte identity meaning "same circuit".

Both paths now render every parameter through the one renderer,
``silk.utils._utils._render_param``, and assemble the line with single spaces.
"""
import os

import pytest

from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk.pdk.basepdk import BasePDK

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PDK_YAML = os.path.join(
    _REPO, 'silk', 'pdk', 'data', 'skywater_130_device_details.yaml')
_PDK_LIB = '/fake/sky130.lib.spice'          # fixed, never opened


def _pdk(**kwargs):
    return BasePDK(_PDK_YAML, library_location=_PDK_LIB, **kwargs)


def _legacy_line(pdk, params, device='nmos1p8'):
    """Device line as the ``add_element`` path emits it."""
    mod = Module('PDK_skywater130', '/tmp', 'legacy.cir',
                 pdk_details=pdk, module_name='legacy')
    mod.define_module_ports({'vdd': Port.supply(), 'vss': Port.ground()})
    mod.add_element(getattr(pdk, device), 'M1',
                    nets={'d': 'vdd', 'g': 'vdd', 's': 'vss', 'b': 'vss'},
                    parameters=dict(params))
    return pdk.generate_netlist_entry('M1', mod.elem_dict['M1'])


def _factory_line(pdk, params, device='nmos1p8'):
    """Device line as the factory API emits it."""
    mod = Module('PDK_skywater130', '/tmp', 'factory.cir',
                 pdk_details=pdk, module_name='factory')
    mod.define_module_ports({'vdd': Port.supply(), 'vss': Port.ground()})
    with mod:
        dev = getattr(pdk, device)('M1', d='vdd', g='vdd', s='vss', b='vss',
                                   **params)
    return dev.to_spice()


# Each case is the *same* device expressed to both APIs.
_CASES = {
    # The historical offender: a float that is integral. str() wrote '4.0'.
    'integral_float': {'w': 4.0, 'l': 0.5, 'm': 1},
    # Same numbers as ints — must not change the spelling either.
    'int_values': {'w': 4, 'l': 1, 'm': 1},
    # Parameters left out entirely: both paths fill from the catalog default.
    'defaults_only': {},
    # Only some supplied — mixes a caller value and a catalog default.
    'partial': {'w': 2.0},
    # A SPICE literal string passes through unscaled on both paths.
    'string_literal': {'w': '1u', 'l': 0.5},
    # Values above the plain-decimal band still take the same SI suffix
    # (physically silly, catalog-legal: sky130 sets no W/L maximum).
    'suffix_band': {'w': 2500.0, 'l': 1000.0},
    # A trailing-zero decimal and a repeating one.
    'decimals': {'w': 2.50, 'l': 0.15, 'm': 2},
}


@pytest.mark.parametrize('case', sorted(_CASES))
def test_paths_emit_identical_bytes_pdk_units(case):
    pdk = _pdk()
    params = _CASES[case]
    assert _legacy_line(pdk, params) == _factory_line(pdk, params)


@pytest.mark.parametrize('case', sorted(_CASES))
def test_paths_emit_identical_bytes_under_unit_conversion(case):
    """Same, with the author writing SI base units (metres) — the branch
    where the legacy path *did* run values through ``_fmt``. Byte identity
    has to hold on both sides of that branch, not just the one."""
    pdk = _pdk(user_units='si')
    params = {k: (v * 1e-6 if k in ('w', 'l') and isinstance(v, (int, float))
                  else v)
              for k, v in _CASES[case].items()}
    assert _legacy_line(pdk, params) == _factory_line(pdk, params)


def test_paths_agree_on_a_module_variable():
    """A ``Variable`` renders from its ``.value`` on both paths."""
    pdk = _pdk()
    var = pdk.nmos1p8.w.variable(name='w_n', value=4.0)
    assert _legacy_line(pdk, {'w': var, 'l': 0.5}) == \
        _factory_line(pdk, {'w': var, 'l': 0.5})


def test_paths_agree_for_a_pmos_too():
    """Nothing in the canonicalization is nmos-specific."""
    pdk = _pdk()
    params = {'w': 8.0, 'l': 0.5, 'm': 1}
    assert _legacy_line(pdk, params, device='pmos1p8') == \
        _factory_line(pdk, params, device='pmos1p8')


def test_canonical_form_is_the_factory_spelling():
    """Pin the canonical bytes themselves: ``_fmt`` spelling, single spaces,
    no trailing space before the newline."""
    line = _legacy_line(_pdk(), {'w': 4.0, 'l': 0.5, 'm': 1})
    assert line == ('XMM1 vdd vdd vss vss sky130_fd_pr__nfet_01v8 '
                    'm=1 w=4 l=0.5\n')
    assert not line[:-1].endswith(' ')


def test_one_shared_renderer():
    """Both emitters call the same function object — not two copies that
    could drift apart again."""
    from silk.elements import factory
    from silk.pdk import basepdk
    from silk.utils import _utils
    assert factory._render_param is _utils._render_param
    assert basepdk.render_pdk_device_line is _utils.render_pdk_device_line
    assert factory.render_pdk_device_line is _utils.render_pdk_device_line
