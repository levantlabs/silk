"""The parameterised amplifier example emits variant-suffixed subcircuits (TASK-523).

Fast and simulator-free: the example's netlist path only. ``load_pdk('sky130')``
reads the packaged catalog and netlist generation needs no installed model
library, so this runs on the bare CI profile too.
"""
import os
import re

import pytest

from silk import load_pdk
from silk.exceptions import CircuitError

_SUBCKT = re.compile(r'^\.subckt\s+(\S+)', re.IGNORECASE | re.MULTILINE)
_SUFFIXED = re.compile(r'^cs_amp_[0-9a-f]{8}$')

SETTING_A = dict(w_in=4.0, c_load=0.1e-12)
SETTING_B = dict(w_in=8.0, c_load=0.3e-12)


@pytest.fixture(scope='module')
def pdk():
    return load_pdk('sky130')


@pytest.fixture
def example(tmp_path, monkeypatch):
    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    from silk.examples import amplifier_variants_example as ex
    return ex


def _x_lines(text):
    """Map instance name -> the subcircuit name each top-level X line uses.

    Only lines after the last ``.ends`` are top-level instances; inside a
    definition, transistor cards also start with ``X`` (sky130 nfet/pfet).
    Silk emits instance names as ``x<name>``, lowercase.
    """
    top = text[text.lower().rfind('.ends'):]
    out = {}
    for line in top.splitlines():
        fields = line.split()
        if fields and fields[0].startswith('x'):
            out[fields[0]] = fields[-1]
    return out


def test_three_placements_two_settings_one_repeat(example, pdk):
    top = example.build_top(pdk, [('xa', SETTING_A), ('xb', SETTING_B),
                             ('xc', SETTING_A)])
    text = example.emit_netlist(top)

    defs = _SUBCKT.findall(text)
    assert len(defs) == 2, defs                       # one per distinct setting
    assert len(set(defs)) == 2
    assert all(_SUFFIXED.match(d) for d in defs), defs

    x = _x_lines(text)
    assert set(x) == {'xxa', 'xxb', 'xxc'}
    assert all(name in defs for name in x.values())
    assert x['xxa'] == x['xxc'] != x['xxb']           # repeat shares a definition


def test_suffix_present_for_a_single_placement(example, pdk):
    top = example.build_top(pdk, [('x1', SETTING_A)])
    text = example.emit_netlist(top)

    defs = _SUBCKT.findall(text)
    assert len(defs) == 1 and _SUFFIXED.match(defs[0]), defs
    assert _x_lines(text) == {'xx1': defs[0]}


def test_same_placements_without_variants_raise(example, pdk):
    with pytest.raises(CircuitError, match="cs_amp"):
        top = example.build_top(pdk, [('xa', SETTING_A), ('xb', SETTING_B)],
                                variants=False)
        example.emit_netlist(top)


def test_child_module_netlist_name_is_cs_amp(example, pdk):
    assert os.path.basename(example.build_cs_amp(pdk).netlistname) == 'cs_amp.cir'


def test_output_lands_under_example_output_dir(example, pdk, tmp_path):
    top = example.build_top(pdk, [('xa', SETTING_A)])
    top.set_output_options(mode='overwrite')
    example.emit_netlist(top)
    assert os.path.isfile(os.path.join(str(tmp_path), 'amplifier_variants',
                                       'amp_variants.cir'))
