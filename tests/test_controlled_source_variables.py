"""A Variable in a controlled-source gain parameter resolves (TASK-228).

Left over from TASK-209, which routed every waveform parameter and a
source's ``dc=``/``ac=`` through the one resolver,
``silk.utils._utils._resolve_value``. The four controlled-source classes in
``silk/sources/classes.py`` still interpolated their gain-like parameter
(VCVS ``gain``, VCCS ``gm``, CCVS ``transresistance``, CCCS ``gain``) raw
into an f-string, so a ``Variable`` there was accepted silently and
wrote its **repr** into the deck::

    Ee1 out 0 inp 0 Variable(name='av', value=10.0)

which ngspice rejects as an undefined parameter. This mirrors
``tests/test_source_waveform_variables.py``: one case per class, proving the
resolved number is emitted, two successive ``space.apply()`` calls change
the emitted text, an expression over the variable resolves too, an unset
variable raises naming the parameter, and plain values stay byte-identical.
"""
from __future__ import annotations

import pytest

from silk import sources
from silk.circuit.variable import Variable
from silk.optimize import VariableSpace

# (class name, build(var) -> source, expected plain-value line)
_CASES = {
    'VCVS': (
        lambda v: sources.VCVS('e1', out_pos='out', out_neg='0',
                               ctrl_pos='inp', ctrl_neg='0', gain=v),
        'Ee1 out 0 inp 0 10\n',
    ),
    'VCCS': (
        lambda v: sources.VCCS('g1', out_pos='out', out_neg='0',
                               ctrl_pos='inp', ctrl_neg='0', gm=v),
        'Gg1 out 0 inp 0 10\n',
    ),
    'CCVS': (
        lambda v: sources.CCVS('h1', out_pos='out', out_neg='0',
                               transresistance=v, ctrl_vname='vmeas'),
        'Hh1 out 0 vmeas 10\n',
    ),
    'CCCS': (
        lambda v: sources.CCCS('f1', out_pos='out', out_neg='0',
                               gain=v, ctrl_vname='vmeas'),
        'Ff1 out 0 vmeas 10\n',
    ),
}


def _var(value=10.0, name='av'):
    return Variable(name=name, value=value, bounds=(0.1, 1000.0))


@pytest.mark.parametrize('cls_name', sorted(_CASES))
class TestAControlledSourceParameterResolvesAndTracksTheSweep:

    def test_no_repr_reaches_the_spice_text(self, cls_name):
        build, _plain = _CASES[cls_name]
        v = _var(10.0)
        text = build(v).to_spice()
        assert 'Variable(' not in text
        assert 'Variable' not in text
        assert v.name not in text

    def test_the_resolved_number_is_emitted(self, cls_name):
        build, _plain = _CASES[cls_name]
        text = build(_var(10.0)).to_spice()
        assert '10.0' in text

    def test_two_successive_applies_change_the_emitted_text(self, cls_name):
        build, _plain = _CASES[cls_name]
        v = _var(10.0)
        src = build(v)
        space = VariableSpace([v])

        space.apply({v.name: 10.0})
        text_first = src.to_spice()
        space.apply({v.name: 25.0})
        text_second = src.to_spice()

        assert text_first != text_second
        assert '10.0' in text_first and '10.0' not in text_second
        assert '25.0' in text_second

    def test_an_expression_over_the_variable_resolves_too(self, cls_name):
        build, _plain = _CASES[cls_name]
        v = _var(10.0)
        src = build(v * 2)
        space = VariableSpace([v])

        space.apply({v.name: 10.0})
        assert '20.0' in src.to_spice()
        space.apply({v.name: 25.0})
        assert '50.0' in src.to_spice()

    def test_an_unassigned_variable_raises_naming_the_parameter(self, cls_name):
        build, _plain = _CASES[cls_name]
        empty = Variable(name='unset')
        with pytest.raises(ValueError) as exc:
            build(empty).to_spice()
        assert 'no value assigned' in str(exc.value)


class TestPlainValuesAreUntouched:
    """Resolution goes through the same shared function the waveform and
    ``dc=`` paths use, but plain numbers still pass through unformatted --
    byte-identical to the deck text silk always wrote."""

    @pytest.mark.parametrize('cls_name', sorted(_CASES))
    def test_the_emitted_text_is_exact(self, cls_name):
        build, expected = _CASES[cls_name]
        assert build(10).to_spice() == expected


@pytest.mark.parametrize('cls_name', sorted(_CASES))
@pytest.mark.parametrize('expression', [False, True])
def test_discovered_controlled_parameter_drives_space(tmp_path, cls_name, expression):
    from silk import Module, load_pdk
    build, _ = _CASES[cls_name]
    var = _var()
    src = build(2 * var if expression else var)
    mod = Module('sky130', str(tmp_path), 'controlled.cir', load_pdk('sky130'))
    mod.add_source(src)
    assert mod.variables() == [var]
    space = VariableSpace(mod.variables())
    space.apply({'av': 25.0})
    assert src.to_spice().endswith('50.0\n' if expression else '25.0\n')
    var.value = None
    assert mod.variables() == [var]
