"""``add_element`` and the factory API must emit the SAME device.

silk has two device-emission paths and they are both public:

    mod.add_element(pdk.nmos3p3, 'M1', nets=…, parameters={'w': 1, 'l': 0.3})
    pdk.nmos3p3('M1', d=…, g=…, s=…, b=…, w=1, l=0.3)

They ran on different code — ``basepdk.generate_netlist_entry`` and
``AnalogElement.to_spice`` — and drifted apart the moment ``user_units`` learned
prefixes. The legacy path kept its own ``user_units == UNIT_SI`` test, which no
prefix mode satisfies, so under the shipped default (``user_units: u``) it
converted nothing: for GF180MCU, whose models expect metres, ``w=1`` (one
micron) went out as ``w=1.0`` — one METRE — while the factory path emitted the
correct ``w=1u``. Same circuit, same PDK, 1e6 apart, and ngspice simulates the
giant device without complaint.

It also wrote the literal ``ng=None`` into the deck for any parameter the
caller omitted, because ``PDKDevice.from_dict`` never read ``default:``.

This compares parameter VALUES, not the rendered text. sky130 emits ``w=1.0``
on the legacy path and ``w=1`` on the factory path — identical numbers, and the
legacy spelling is deliberately preserved so decks written before unit
conversion existed stay byte-identical. Text equality would be a stricter gate
than the physics warrants.
"""
import pytest

from silk.utils._utils import _parse_si
from tests._env import needs_pdk, pdk_available


def _registered():
    from silk.pdk import available
    return sorted({n for n in available() if pdk_available(n)})


def _params_from(line):
    """{name: float} from the parameter tail of an emitted instance line."""
    out = {}
    for token in line.split():
        if '=' not in token:
            continue
        key, _, val = token.partition('=')
        out[key] = _parse_si(val)
    return out


def _emit_both(pdk, cls, supplied):
    """The same device through both APIs. Returns (legacy, factory) param dicts."""
    from silk.circuit.modules import Module

    pins = {p: f'n{i}' for i, p in enumerate(cls._pin_names)}
    factory_line = cls('M1', **pins, **supplied).to_spice()

    mod = Module(pdk.pdk_name or 'p', '.', 'unused.cir', pdk)
    mod.add_element(cls, 'M1', nets=dict(pins), parameters=dict(supplied) or None)
    legacy_line = mod.netlister.generate_device_instances(mod.elem_dict).strip().splitlines()[-1]
    return _params_from(legacy_line), _params_from(factory_line)




def _legal_supplied(cls):
    """One caller-supplied value per parameter, legal under the 2026-08-23
    emission-time constraint check: the historical blanket 1.0 (one
    author-unit) clamped into [minval, maxval] where the catalog bounds exclude
    it (sky130's MIM caps: min w = 2 µm; its ESD FET: max l = 0.55 µm) —
    converted from the catalog's PDK
    unit back to author units through the same scale the emitter uses.
    Dimensionless counts stay 1."""
    supplied = {}
    for p in cls._param_names:
        c = cls._param_constraints.get(p)
        if getattr(c, 'pdk_unit', None):
            v = 1.0
            scale = cls._param_scales.get(p, 1.0)
            for end, op in (('minval', max), ('maxval', min)):
                b = getattr(c, end, None)
                if b not in (None, '-1'):
                    try:
                        v = op(v, float(b) / scale)
                    except (ValueError, TypeError):
                        pass
            supplied[p] = v
        else:
            supplied[p] = 1
    return supplied


@needs_pdk
@pytest.mark.parametrize('pdk_name', _registered())
@pytest.mark.parametrize('supply', [True, False], ids=['caller-supplied', 'defaults'])
def test_add_element_and_the_factory_emit_the_same_device(pdk_name, supply, tmp_path, monkeypatch):
    from silk import load_pdk

    monkeypatch.chdir(tmp_path)
    pdk = load_pdk(pdk_name)

    checked = 0
    for dev_name, cls in sorted(pdk._device_classes.items()):
        # A geometry gets 1.0 (one author-unit); a dimensionless count gets 1.
        supplied = _legal_supplied(cls) if supply else {}
        try:
            legacy, factory = _emit_both(pdk, cls, supplied)
        except TypeError:
            continue        # required parameter with no default — not our case
        assert legacy == factory, (
            f'{pdk_name}.{dev_name}: the two APIs disagree\n'
            f'  add_element: {legacy}\n'
            f'  factory    : {factory}')
        checked += 1

    assert checked, f'{pdk_name}: no device emitted through either API'


@needs_pdk
@pytest.mark.parametrize('pdk_name', _registered())
def test_add_element_never_emits_a_none_parameter(pdk_name, tmp_path, monkeypatch):
    """An omitted parameter takes the catalog default. `ng=None` is not a deck."""
    from silk import load_pdk
    from silk.circuit.modules import Module

    monkeypatch.chdir(tmp_path)
    pdk = load_pdk(pdk_name)

    for dev_name, cls in sorted(pdk._device_classes.items()):
        mod = Module(pdk_name, '.', 'unused.cir', pdk)
        mod.add_element(cls, 'M1',
                        nets={p: f'n{i}' for i, p in enumerate(cls._pin_names)})
        line = mod.netlister.generate_device_instances(mod.elem_dict)
        assert 'None' not in line, f'{pdk_name}.{dev_name}: {line.strip()}'


@needs_pdk
@pytest.mark.parametrize('pdk_name', _registered())
def test_a_micron_reaches_the_models_as_a_micron_on_both_paths(pdk_name, tmp_path, monkeypatch):
    """The point of the whole mechanism, stated in physical terms.

    Whatever a PDK's models want, `w=1` written by the author is one micron.
    Read the emitted value back through the PDK's own declared unit and it must
    come out 1 µm — on BOTH APIs. This is what catches a missing conversion
    even if the two paths were consistently wrong together.
    """
    from silk import load_pdk
    from silk.pdk.basepdk import _SI_TO_PDK_SCALE

    monkeypatch.chdir(tmp_path)
    pdk = load_pdk(pdk_name)

    checked = 0
    for dev_name, cls in sorted(pdk._device_classes.items()):
        c = cls._param_constraints.get('w')
        unit = getattr(c, 'pdk_unit', None)
        if unit not in _SI_TO_PDK_SCALE or getattr(c, 'quantity', None):
            continue                      # not a plain length in a known unit
        supplied = _legal_supplied(cls)
        try:
            legacy, factory = _emit_both(pdk, cls, supplied)
        except TypeError:
            continue
        # emitted value / (PDK units per metre) = metres; the author wrote
        # supplied['w'] microns, so that many µm must reach the models.
        for label, params in (('add_element', legacy), ('factory', factory)):
            metres = params['w'] / _SI_TO_PDK_SCALE[unit]
            assert metres == pytest.approx(supplied['w'] * 1e-6, rel=1e-9), (
                f'{pdk_name}.{dev_name} via {label}: author wrote '
                f'w={supplied["w"]} µm, models receive {metres:g} m '
                f'(emitted {params["w"]:g} {unit})')
        checked += 1

    assert checked, f'{pdk_name}: no device has a plain-length w — test is vacuous'


@needs_pdk
@pytest.mark.parametrize('pdk_name', _registered())
def test_assigning_a_parameter_emits_the_same_as_passing_it(pdk_name, tmp_path, monkeypatch):
    """``mn.w = 5`` after construction is the same author statement as ``w=5``
    at construction, and must unit-convert identically.

    It did not: ``AnalogElement.__setattr__`` stored the value without marking
    it caller-supplied, so assigning a parameter that was DEFAULTED at
    construction skipped conversion — on a metres-PDK ``mn.w = 5`` emitted five
    metres. For a binned MOSFET that dies with "could not find a valid
    modelname"; for a resistor, whose length is a continuous model formula, it
    is a silently wrong answer. sky130's identity scale hid it from the suite.
    """
    from silk import load_pdk

    monkeypatch.chdir(tmp_path)
    pdk = load_pdk(pdk_name)

    checked = 0
    for dev_name, cls in sorted(pdk._device_classes.items()):
        pins = {p: f'n{i}' for i, p in enumerate(cls._pin_names)}
        supplied = _legal_supplied(cls)
        try:
            passed = cls('M1', **pins, **supplied).to_spice()
            assigned = cls('M1', **pins)          # everything defaulted...
        except TypeError:
            continue        # a required parameter with no default — not our case
        for p, v in supplied.items():
            setattr(assigned, p, v)               # ...then assigned
        assert assigned.to_spice() == passed, (
            f'{pdk_name}.{dev_name}: assignment and construction disagree\n'
            f'  passed  : {passed.strip()}\n'
            f'  assigned: {assigned.to_spice().strip()}')
        checked += 1

    assert checked, f'{pdk_name}: no device was constructible from defaults'
