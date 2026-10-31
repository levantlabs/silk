"""TASK-75 — a net value that cannot be one SPICE node token is rejected on
BOTH construction styles.

TASK-41 closed this on the factory style (``pdk.nmos1p8(...)`` inside
``with mod:``) only. The legacy style — ``add_element(pdk.nmos1p8, name,
nets={...})``, the one ``docs/`` and the older examples teach — reproduced the
original defect byte for byte, because it binds through ``_apply_connections``
and emits through ``BasePDK.generate_netlist_entry()``, and neither route saw
the factory's validator. Measured on this branch before the fix, sky130
fake-lib PDK:

* ``nets={'g': ''}``    → ``XMmn y  gnd gnd sky130_fd_pr__nfet_01v8 ...`` — one
  node short. ngspice-42 accepts that deck, reads the model name as the
  missing node and invents a net called ``sky130_fd_pr__nfet_01v8`` (0.167 V
  in the .op table): a mis-wired transistor with no error anywhere.
* ``nets={'g': None}``  → a bare ``AttributeError: 'NoneType' object has no
  attribute 'casefold'`` from inside net-name canonicalization, naming neither
  the instance nor the pin.
* ``nets={'g': 'a b'}`` → ``XMmn y a b gnd gnd ...`` — FIVE node tokens for a
  four-pin device. ``add_net('a b')`` registered that net without complaint.

Per decision-1 all three are CALLER ERROR — malformed for every corner, not a
data-dependent degenerate result — so they raise, with the diagnosis in the
message. Both styles now route through the ONE validator
(``silk.elements.base.validate_net_token``), which is what keeps them from
drifting apart again.
"""
import warnings

import pytest

from silk.circuit.modules import Module
from silk.circuit.ports import Port
from silk.exceptions import CircuitError


_MN_NETS = {'d': 'y', 'g': 'in', 's': 'gnd', 'b': 'gnd'}
_MN_PARAMS = {'w': 1, 'l': 0.15}

# (bad value, exception type). Every one of these is a value that cannot be
# ONE node token: the wrong KIND of thing (TypeError) or the right kind with
# an unusable value (ValueError).
_BAD = [
    ('', ValueError),
    ('   ', ValueError),
    ('\t', ValueError),
    ('a b', ValueError),
    ('a\tb', ValueError),
    (None, TypeError),
    (0, TypeError),
    (1.5, TypeError),
    (object(), TypeError),
]
_BAD_IDS = ['empty', 'blank', 'tab', 'space-inside', 'tab-inside',
            'none', 'int', 'float', 'object']


def _module(pdk, tmp_path, name='top', strict=False):
    return Module('PDK_skywater130', str(tmp_path / name), f'{name}.cir',
                  pdk_details=pdk, strict_nets=strict, module_name=name)


def _inverter(pdk, tmp_path):
    """A minimal submodule with four declared ports."""
    inv = Module('PDK_skywater130', str(tmp_path / 'inv'), 'inv.cir',
                 pdk_details=pdk, module_name='inv')
    inv.define_module_ports({
        'vin': Port.analog_input(),
        'vout': Port.analog_output(),
        'vdd': Port.supply(),
        'gnd': Port.ground(),
    })
    with inv:
        pdk.nmos1p8('mn', d='vout', g='vin', s='gnd', b='gnd', **_MN_PARAMS)
    return inv


# ---------------------------------------------------------------------------
# AC #1 / AC #2 — the legacy add_element path
# ---------------------------------------------------------------------------

class TestAddElementNetValues:

    @pytest.mark.parametrize('bad,exc', _BAD, ids=_BAD_IDS)
    def test_bad_net_value_is_typed_and_names_instance_and_pin(
            self, pdk, tmp_path, bad, exc):
        m = _module(pdk, tmp_path)
        with pytest.raises(exc) as e:
            m.add_element(pdk.nmos1p8, 'mn', nets={**_MN_NETS, 'g': bad},
                          parameters=_MN_PARAMS)
        assert "nmos1p8 'mn' pin 'g'" in str(e.value)

    @pytest.mark.parametrize('bad,exc', _BAD, ids=_BAD_IDS)
    def test_nothing_is_registered_when_a_net_value_is_rejected(
            self, pdk, tmp_path, bad, exc):
        """AC #1: rejected *before* the element is registered, so neither the
        instance nor a junk net is left behind."""
        m = _module(pdk, tmp_path)
        with pytest.raises(exc):
            m.add_element(pdk.nmos1p8, 'mn', nets={**_MN_NETS, 'g': bad},
                          parameters=_MN_PARAMS)
        assert 'mn' not in m.elem_dict
        assert m.net_dict == {}

    def test_none_is_a_typeerror_not_a_bare_attributeerror(self, pdk, tmp_path):
        """AC #2. The old failure was ``AttributeError: 'NoneType' object has
        no attribute 'casefold'`` raised inside net-name canonicalization."""
        m = _module(pdk, tmp_path)
        with pytest.raises(TypeError) as e:
            m.add_element(pdk.nmos1p8, 'mn', nets={**_MN_NETS, 'g': None},
                          parameters=_MN_PARAMS)
        assert 'casefold' not in str(e.value)
        assert 'never assigned' in str(e.value)

    def test_int_message_points_at_the_missing_quotes(self, pdk, tmp_path):
        """``s=0`` for the SPICE ground node is the plausible slip."""
        m = _module(pdk, tmp_path)
        with pytest.raises(TypeError, match=r"write '0', not 0"):
            m.add_element(pdk.nmos1p8, 'mn', nets={**_MN_NETS, 's': 0},
                          parameters=_MN_PARAMS)

    def test_tuple_list_form_is_validated_too(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        with pytest.raises(ValueError, match=r"nmos1p8 'mn' pin 'g'"):
            m.add_element(pdk.nmos1p8, 'mn',
                          nets=[('d', 'y'), ('g', ''), ('s', 'gnd'), ('b', 'gnd')],
                          parameters=_MN_PARAMS)
        assert 'mn' not in m.elem_dict

    def test_positional_form_is_validated_and_names_the_pin_it_lands_on(
            self, pdk, tmp_path):
        """The positional list pairs by pin order, so the message names the
        pin the bad value would have been wired to."""
        m = _module(pdk, tmp_path)
        with pytest.raises(ValueError, match=r"nmos1p8 'mn' pin 's'"):
            m._element_mgr.add_element(pdk.nmos1p8, 'mn',
                                       ['y', 'in', 'a b', 'gnd'], _MN_PARAMS)
        assert 'mn' not in m.elem_dict

    def test_list_value_through_the_graph_is_a_typeerror(self, pdk, tmp_path):
        """Through ``Module.add_element`` a list is caught earlier by the
        bus-wiring guard; through ``CircuitGraph`` it reaches the net-value
        check, which reports it as the wrong kind of thing for a pin."""
        m = _module(pdk, tmp_path)
        with pytest.raises(CircuitError, match='multi-source bus wiring'):
            m.add_element(pdk.nmos1p8, 'mn', nets={**_MN_NETS, 'g': ['n1']},
                          parameters=_MN_PARAMS)
        with pytest.raises(TypeError, match=r"nmos1p8 'mn' pin 'g'"):
            m._element_mgr.add_element(pdk.nmos1p8, 'mn',
                                       {**_MN_NETS, 'g': ['n1']}, _MN_PARAMS)

    def test_old_style_pdkdevice_argument_is_validated(self, pdk, tmp_path):
        """The third add_element shape — a PDKDevice, not the class."""
        m = _module(pdk, tmp_path)
        with pytest.raises(ValueError, match=r"nmos1p8 'mn' pin 'g'"):
            m.add_element(pdk._devices['nmos1p8'], 'mn',
                          nets={**_MN_NETS, 'g': ''}, parameters=_MN_PARAMS)
        assert 'mn' not in m.elem_dict

    def test_net_object_is_accepted_and_normalised(self, pdk, tmp_path):
        """A Net is admitted on this path exactly as on the factory one, and
        normalises to its NAME (every downstream consumer reads strings)."""
        m = _module(pdk, tmp_path)
        vbias = m.add_net('vbias')
        m.add_element(pdk.nmos1p8, 'mn', nets={**_MN_NETS, 'g': vbias},
                      parameters=_MN_PARAMS)
        assert m.elem_dict['mn']['device'].pin_list['g'].net == 'vbias'
        assert m.net_dict['vbias'].pins[0].instance == 'mn'

    def test_net_object_with_a_blank_name_is_still_rejected(self, pdk, tmp_path):
        class _FakeNet:
            name = ''

        m = _module(pdk, tmp_path)
        with pytest.raises(ValueError, match=r"nmos1p8 'mn' pin 'g'"):
            m.add_element(pdk.nmos1p8, 'mn', nets={**_MN_NETS, 'g': _FakeNet()},
                          parameters=_MN_PARAMS)


class TestPinAssignmentNetValues:
    """The rewire entry point shares the same chokepoint."""

    @pytest.mark.parametrize('bad,exc', _BAD, ids=_BAD_IDS)
    def test_rewire_to_a_bad_value_is_rejected(self, pdk, tmp_path, bad, exc):
        m = _module(pdk, tmp_path)
        dev = pdk.nmos1p8('mn', **_MN_NETS, **_MN_PARAMS)
        m.add(dev)
        with pytest.raises(exc, match=r"nmos1p8 'mn' pin 'g'"):
            dev.g = bad
        # The rejected rewire left the previous binding untouched.
        assert dev.nets['g'] == 'in'
        assert m.net_dict['in'].pins[0].port == 'g'

    def test_submodule_port_is_validated_before_the_instance_is_registered(
            self, pdk, tmp_path):
        top = _module(pdk, tmp_path)
        with pytest.raises(ValueError, match=r"inv 'u1' pin 'vdd'"):
            top.add_module(_inverter(pdk, tmp_path), 'u1',
                            vin='a', vout='y', vdd='', gnd='gnd')
        assert 'u1' not in top.elem_dict
        assert top.net_dict == {}


# ---------------------------------------------------------------------------
# AC #4 — Module.add_net
# ---------------------------------------------------------------------------

class TestAddNetNames:

    @pytest.mark.parametrize('bad,exc', _BAD, ids=_BAD_IDS)
    def test_a_name_that_cannot_be_one_node_token_is_rejected(
            self, pdk, tmp_path, bad, exc):
        m = _module(pdk, tmp_path)
        with pytest.raises(exc, match='add_net'):
            m.add_net(bad)
        assert m.net_dict == {}

    def test_bus_declaration_is_validated_too(self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        with pytest.raises(ValueError, match='add_net'):
            m.add_net('a b', bits=4)
        assert m.net_dict == {}

    def test_ordinary_names_still_work(self, pdk, tmp_path):
        """Guard against over-reach: everything SPICE accepts as one token
        still declares, including the punctuation bus names use."""
        m = _module(pdk, tmp_path)
        for name in ['vdd', 'VDD_1p8', 'net_0', 'x[3]', 'a[1]', '0']:
            assert m.add_net(name).name == name


# ---------------------------------------------------------------------------
# AC #5 — the two styles agree, message for message
# ---------------------------------------------------------------------------

class TestBothStylesAgree:

    @pytest.mark.parametrize('bad,exc', _BAD, ids=_BAD_IDS)
    def test_same_exception_type_and_message(self, pdk, tmp_path, bad, exc):
        factory, legacy = None, None
        with pytest.raises(exc) as ef:
            pdk.nmos1p8('mn', **{**_MN_NETS, 'g': bad}, **_MN_PARAMS)
        factory = ef.value
        m = _module(pdk, tmp_path)
        with pytest.raises(exc) as el:
            m.add_element(pdk.nmos1p8, 'mn', nets={**_MN_NETS, 'g': bad},
                          parameters=_MN_PARAMS)
        legacy = el.value
        assert type(legacy) is type(factory)
        assert str(legacy) == str(factory)


# ---------------------------------------------------------------------------
# AC #3 — one node token per pin, mechanically, over the whole catalog
# ---------------------------------------------------------------------------

class TestEmittedNodeTokenCount:
    """The property that would have caught the original defect, asserted over
    every device the shipped sky130 catalog defines, through BOTH styles.

    ``tests/test_elements_factory.py::TestEmittedNodeTokenCount`` does the
    factory half (and covers the ideal elements, which the legacy class-argument
    path cannot reach — they are not in ``pdk.model_names``). This one runs the
    legacy emitter and then compares the two node sections directly.
    """

    def _legacy_line(self, pdk, tmp_path, cls, name, nets):
        m = _module(pdk, tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            m.add_element(cls, name, nets=dict(nets))
        deck = m.netlister.generate_device_instances(m.elem_dict)
        return deck.strip().splitlines()[-1]

    def test_every_pdk_device_emits_one_token_per_pin_on_both_styles(
            self, pdk, tmp_path):
        checked = 0
        for dev_name, cls in sorted(pdk._device_classes.items()):
            pins = list(cls._pin_names)
            if not pins:
                continue
            nets = {p: f'n{i}' for i, p in enumerate(pins)}
            line = self._legacy_line(pdk, tmp_path, cls, 'x1', nets)
            tokens = line.split()
            assert tokens[1:1 + len(pins)] == list(nets.values()), (
                f'{dev_name}: node tokens do not match its pins:\n  {line}')
            # And the token right after the nodes cannot be mistaken for one
            # of them — this is the shape ngspice mis-parsed when a node went
            # missing.
            assert tokens[1 + len(pins)] not in nets.values()

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                factory_tokens = cls('x1', **nets).to_spice().split()
            assert factory_tokens[1:1 + len(pins)] == tokens[1:1 + len(pins)]
            checked += 1
        assert checked >= 30, f'only {checked} devices exercised'

    def test_the_legacy_emitter_guards_a_direct_pin_list_mutation(
            self, pdk, tmp_path):
        """AC #3 is a property of the EMITTER, not only of its callers: a
        write straight into ``pin_list`` bypasses both entry points, so
        ``generate_netlist_entry`` re-checks every pin before building the
        line. Without this, that write produced the token-short line."""
        m = _module(pdk, tmp_path)
        m.add_element(pdk.nmos1p8, 'mn', nets=dict(_MN_NETS), parameters=_MN_PARAMS)
        m.elem_dict['mn']['device'].pin_list['g'].net = ''
        with pytest.raises(ValueError, match=r"nmos1p8 'mn' pin 'g'"):
            m.netlister.generate_device_instances(m.elem_dict)

    def test_the_legacy_emitter_guards_a_whitespace_bearing_mutation(
            self, pdk, tmp_path):
        m = _module(pdk, tmp_path)
        m.add_element(pdk.nmos1p8, 'mn', nets=dict(_MN_NETS), parameters=_MN_PARAMS)
        m.elem_dict['mn']['device'].pin_list['g'].net = 'a b'
        with pytest.raises(ValueError, match=r"nmos1p8 'mn' pin 'g'"):
            m.netlister.generate_device_instances(m.elem_dict)


@pytest.mark.parametrize('value', [42, -1, 1.5, 0.0])
def test_numeric_input_remedy_is_a_legal_name(value):
    from silk.elements.base import validate_net_token
    with pytest.raises(TypeError) as exc:
        validate_net_token(value, who='test')
    message = str(exc.value)
    assert f"write '{value}', not {value}" not in message
    assert "'node_42'" in message or "write '0'" in message
    assert validate_net_token('node_42', who='test') == 'node_42'
