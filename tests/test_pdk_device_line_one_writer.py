"""TASK-416: one PDK device-line writer, and faults reach add_element devices."""
import warnings

import pytest

from silk.circuit.modules import Module
from silk.faults import FaultSet, stuck_open
from silk.pdk import load_pdk


def _build(tmp_path, how, pdk_name='sky130'):
    pdk = load_pdk(pdk_name)
    m = Module(pdk_name, str(tmp_path), 'x', pdk)
    for n in ('d', 'g', 's'):
        m.add_net(n)
    if how == 'cls':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            m.add_element(pdk.nmos1p8, 'mn', nets=dict(d='d', g='g', s='s', b='s'),
                          parameters={'w': 4.0, 'l': 1.0, 'm': 2})
    else:
        m.add(pdk.nmos1p8('mn', d='d', g='g', s='s', b='s', w=4.0, l=1.0, m=2))
    return m


def _deck(m, **kw):
    m.generate_netlist(**kw)
    return open(m._run_files.netlist_path).read()


def _line(deck):
    return [ln for ln in deck.splitlines() if ln.startswith('XMmn')]


def test_both_apis_emit_the_same_line_through_one_writer(tmp_path):
    from silk.pdk import basepdk
    from silk.elements import factory
    # one writer: both modules import the same function object
    assert basepdk.render_pdk_device_line is factory.render_pdk_device_line
    a = _line(_deck(_build(tmp_path / 'a', 'cls')))
    b = _line(_deck(_build(tmp_path / 'b', 'factory')))
    assert a == b and len(a) == 1


def test_fault_on_add_element_device_is_applied_like_factory(tmp_path):
    fs = lambda: FaultSet([stuck_open('mn')])  # noqa: E731
    clean = _line(_deck(_build(tmp_path / 'c', 'cls')))
    a = _line(_deck(_build(tmp_path / 'a', 'cls'), fault_set=fs()))
    b = _line(_deck(_build(tmp_path / 'b', 'factory'), fault_set=fs()))
    assert a == b
    assert a != clean          # never silently ignored
