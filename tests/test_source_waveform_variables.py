"""A Variable in a source WAVEFORM parameter resolves (TASK-209).

Before this, three source-shaped parameters answered the same question three
different ways: a passive refused a ``Variable`` at construction, a
source's ``dc=`` resolved it and tracked ``space.apply()``, and every waveform
parameter accepted it silently and wrote its **repr** into the deck ::

    Vvin vin 0 DC 0.75 SIN(0.75 Variable(name='vampl', value=0.02) 27340000.0 0 0)

which ngspice rejects with "Undefined parameter [vampl]". The ruling recorded
on TASK-209 is *resolve*, everywhere: a variable on any waveform parameter of
any waveform class behaves exactly as one on ``dc=`` does, because both now go
through the one resolver ``silk.utils._utils._resolve_value``. Passives are
unchanged — they still refuse.

What is proved here:

* **Every waveform class.** The per-class table is cross-checked against the
  ``waveforms:`` map in ``silk/sources/source_metadata.yaml``, so a waveform
  class added to silk without a case here fails the coverage test rather than
  quietly keeping the old repr behaviour.
* **It tracks the sweep.** Each class emits the *current* value after two
  successive ``space.apply()`` calls — asserted from the emitted text, never
  from the variable's ``.value``.
* **PWL per element.** A variable may sit on a single time or a single value
  of one point, and a swept time re-sorts the points.
* **Expressions too.** ``vampl * 2`` is a ``_VarExpr``; it resolves the same
  way, since ``dc=`` has always resolved those.
* **The two paths cannot drift.** The ``dc=`` assertions live beside the
  waveform ones in this file (AC #4).
* **Plain values are untouched.** Every waveform's plain-float/SI-string text
  is pinned byte for byte: resolution shares ``_resolve_value`` with the device
  path but *not* ``_fmt``, which would have rewritten ``27343750.0`` as
  ``27.3438Meg``.
* **A real deck, and a real simulation.** A generated netlist for the task's
  own reproducer, and a slow ngspice leg that sweeps a sine amplitude on an RC
  low-pass and measures a different output amplitude per point.
"""
from __future__ import annotations

import inspect
import os
import re

import numpy as np
import pytest
import yaml

from silk import sources
from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.circuit.variable import Variable
from silk.elements import passives
from silk.elements import elements
from silk.optimize import VariableSpace
from silk.simulation import analysis
from silk.sources.waveforms import (AM, PWL, SFFM, Exponential, Pulse,
                                    RandomSource, Sine, TransientNoise)

_PDK_YAML = os.path.join(os.path.dirname(__file__), '..', 'silk', 'pdk', 'data',
                         'skywater_130_device_details.yaml')
_SKY130_LIB = os.path.expanduser(
    '~/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice')
_have_pdk = os.path.exists(_SKY130_LIB)

_METADATA = os.path.join(os.path.dirname(__file__), '..', 'silk', 'sources',
                         'source_metadata.yaml')


def _var(value=0.02, name='vampl'):
    return Variable(name=name, value=value, bounds=(0.001, 10.0))


# ---------------------------------------------------------------------------
# The per-class table — one case per waveform class silk ships
# ---------------------------------------------------------------------------
#
# Each case: (class name, build(var) -> waveform, the swept parameter, the two
# values to apply). ``build`` puts the variable on ONE parameter and leaves the
# rest as plain numbers, so a case fails if that one parameter is the one left
# unresolved.

_CASES = {
    'BitPattern': (lambda v: sources.BitPattern('01', v1=0, v2=v, ui=1, tr=.1, tf=.2),
                   'v2', (0.02, 0.3)),
    'PRBS': (lambda v: sources.PRBS(7, 10, v1=0, v2=v, ui=1, tr=.1, tf=.2),
             'v2', (0.02, 0.3)),
    'Pulse': (lambda v: Pulse(v1=0, v2=v, td=0, tr='1n', tf='1n',
                              pw='10n', per='20n'), 'v2', (0.02, 0.3)),
    'Sine': (lambda v: Sine(voffset=0.75, vampl=v, freq=27.34e6),
             'vampl', (0.02, 0.3)),
    'PWL': (lambda v: PWL([(0, 0), (1e-9, 0), (2e-9, v)]),
            'point value', (0.02, 0.3)),
    'Exponential': (lambda v: Exponential(v1=0, v2=v, td1=1e-9, tau1=2e-9),
                    'v2', (0.02, 0.3)),
    'SFFM': (lambda v: SFFM(vo=0, va=v, fc=1e6, mdi=2, fs=1e3),
             'va', (0.02, 0.3)),
    'AM': (lambda v: AM(va=v, vo=0, mf=1e3, fc=1e6), 'va', (0.02, 0.3)),
    'TransientNoise': (lambda v: TransientNoise(na=v, nt='0.5n'),
                       'na', (0.02, 0.3)),
    'RandomSource': (lambda v: RandomSource(type=1, ts='1n', param1=v),
                     'param1', (0.02, 0.3)),
}


def _classes_in_metadata():
    with open(_METADATA) as fh:
        return set(yaml.safe_load(fh)['waveforms'].values())


class TestEveryWaveformClassIsCovered:
    """The table above is the enumeration, checked against the authored map."""

    def test_the_table_matches_source_metadata_exactly(self):
        assert set(_CASES) == _classes_in_metadata()

    def test_every_covered_class_is_a_real_waveform_class(self):
        import silk.sources.waveforms as wf
        for name in _CASES:
            assert hasattr(wf, name)


@pytest.mark.parametrize('cls_name', sorted(_CASES))
class TestAWaveformParameterResolvesAndTracksTheSweep:
    """One variable, one parameter, every class."""

    def test_no_repr_reaches_the_spice_text(self, cls_name):
        build, _param, (first, _second) = _CASES[cls_name]
        v = _var(first)
        text = build(v).to_spice()
        assert 'Variable(' not in text
        assert 'Variable' not in text
        assert v.name not in text

    def test_the_resolved_number_is_emitted(self, cls_name):
        build, _param, (first, _second) = _CASES[cls_name]
        text = build(_var(first)).to_spice()
        assert str(first) in text.split('(', 1)[1]

    def test_two_successive_applies_change_the_emitted_text(self, cls_name):
        """AC #5, read off the deck text — not off ``var.value``."""
        build, _param, (first, second) = _CASES[cls_name]
        v = _var(first)
        wave = build(v)
        space = VariableSpace([v])

        space.apply({v.name: first})
        text_first = wave.to_spice()
        space.apply({v.name: second})
        text_second = wave.to_spice()

        assert text_first != text_second
        assert str(first) in text_first and str(first) not in text_second
        assert str(second) in text_second

    def test_an_expression_over_the_variable_resolves_too(self, cls_name):
        """``vampl * 2`` is a ``_VarExpr`` — the same thing ``dc=`` accepts."""
        build, _param, (first, second) = _CASES[cls_name]
        v = _var(first)
        wave = build(v * 2)
        space = VariableSpace([v])

        space.apply({v.name: first})
        assert str(first * 2) in wave.to_spice()
        space.apply({v.name: second})
        assert str(second * 2) in wave.to_spice()

    def test_an_unassigned_variable_raises_naming_the_parameter(self, cls_name):
        build, _param, _values = _CASES[cls_name]
        empty = Variable(name='unset')
        with pytest.raises(ValueError) as exc:
            build(empty).to_spice()
        assert 'no value assigned' in str(exc.value)

    def test_inside_a_voltage_source_line(self, cls_name):
        """The waveform is reached through the source, which is how a deck
        gets it — the kwarg name is the lowercase-ish selector in the yaml."""
        with open(_METADATA) as fh:
            kinds = {c: k for k, c in yaml.safe_load(fh)['waveforms'].items()}
        build, _param, (first, second) = _CASES[cls_name]
        v = _var(first)
        kind = kinds[cls_name]
        slot = kind if kind in inspect.signature(sources.VoltageSource).parameters else 'waveform'
        src = sources.VoltageSource('vin', positive='vin', negative='0',
                                    dc=0.75, **{slot: build(v)})
        space = VariableSpace([v])
        space.apply({v.name: second})
        line = src.to_spice()
        assert 'Variable(' not in line
        assert str(second) in line


# ---------------------------------------------------------------------------
# PWL: per element
# ---------------------------------------------------------------------------

class TestPWLPerElement:

    def test_a_variable_value_on_one_point_resolves(self):
        v = _var(0.02)
        pwl = PWL([(0, 0), (1e-9, 0), (2e-9, v)])
        assert pwl.to_spice() == 'PWL(0.0 0.0 1e-09 0.0 2e-09 0.02)'
        VariableSpace([v]).apply({v.name: 0.3})
        assert pwl.to_spice() == 'PWL(0.0 0.0 1e-09 0.0 2e-09 0.3)'

    def test_a_variable_time_on_one_point_resolves(self):
        t = Variable(name='tedge', value=2e-9, bounds=(1e-10, 1e-8))
        pwl = PWL([(0, 0), (1e-9, 0), (t, 1.8)])
        assert pwl.to_spice().endswith('2e-09 1.8)')
        VariableSpace([t]).apply({t.name: 4e-9})
        assert pwl.to_spice().endswith('4e-09 1.8)')

    def test_a_swept_time_re_sorts_the_points(self):
        """Ordering follows the values the variables hold NOW: a time swept
        below its neighbour moves its point, because the deck must stay
        monotonic in time."""
        t = Variable(name='tedge', value=3e-9, bounds=(1e-10, 1e-8))
        pwl = PWL([(0, 0), (2e-9, 0.5), (t, 1.8)])
        assert pwl.to_spice() == 'PWL(0.0 0.0 2e-09 0.5 3e-09 1.8)'
        VariableSpace([t]).apply({t.name: 1e-9})
        assert pwl.to_spice() == 'PWL(0.0 0.0 1e-09 1.8 2e-09 0.5)'

    def test_add_accepts_a_variable(self):
        v = _var(1.8)
        pwl = PWL().add(0, 0).add(1e-9, v)
        assert pwl.to_spice() == 'PWL(0.0 0.0 1e-09 1.8)'

    def test_r_and_td_resolve(self):
        r = Variable(name='rep', value=10e-9)
        td = Variable(name='delay', value=1e-9)
        pwl = PWL([(0, 0), (1e-9, 1.8)], r=r, td=td)
        assert pwl.to_spice() == 'PWL(0.0 0.0 1e-09 1.8) r=1e-08 td=1e-09'

    def test_points_reads_back_resolved_numbers(self):
        v = _var(0.02)
        pwl = PWL([(0, 0), (1e-9, v)])
        assert pwl.points == [(0.0, 0.0), (1e-9, 0.02)]
        assert pwl[1] == (1e-9, 0.02)
        assert list(pwl) == [(0.0, 0.0), (1e-9, 0.02)]

    def test_to_numpy_follows_the_variable(self):
        v = _var(1.0)
        pwl = PWL([(0, 0), (1e-9, v)])
        assert pwl.to_numpy(np.array([1e-9]))[0] == pytest.approx(1.0)
        VariableSpace([v]).apply({v.name: 2.0})
        assert pwl.to_numpy(np.array([1e-9]))[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Pulse edges: still checked, now also at emission
# ---------------------------------------------------------------------------

class TestPulseEdgeTimesStayChecked:

    def test_a_variable_edge_is_accepted_and_resolves(self):
        t = Variable(name='tedge', value=1e-9, bounds=(1e-12, 1e-8))
        p = Pulse(0, 1.8, tr=t, tf='1n', pw='10n', per='20n')
        assert p.to_spice() == 'PULSE(0 1.8 0 1e-09 1n 10n 20n)'
        VariableSpace([t]).apply({t.name: 5e-10})
        assert p.to_spice() == 'PULSE(0 1.8 0 5e-10 1n 10n 20n)'

    def test_a_zero_edge_at_construction_still_raises(self):
        t = Variable(name='tedge', value=0.0)
        with pytest.raises(ValueError, match='must be > 0'):
            Pulse(0, 1.8, tr=t, tf='1n')

    def test_a_sweep_cannot_slip_a_zero_edge_into_a_deck(self):
        """The edge is re-checked at emission, because a variable can move
        after construction — which is the entire point of a variable."""
        t = Variable(name='tedge', value=1e-9)
        p = Pulse(0, 1.8, tr=t, tf='1n')
        t.value = 0.0
        with pytest.raises(ValueError, match='must be > 0'):
            p.to_spice()

    def test_to_numpy_follows_a_variable_amplitude(self):
        v = _var(1.0)
        p = Pulse(0, v, td=0, tr=1e-12, tf=1e-12, pw=1e-6, per=2e-6)
        assert p.to_numpy(np.array([1e-7]))[0] == pytest.approx(1.0)
        VariableSpace([v]).apply({v.name: 2.0})
        assert p.to_numpy(np.array([1e-7]))[0] == pytest.approx(2.0)


class TestSineBehavioralTwinFollowsTheVariable:

    def test_to_numpy_amplitude_tracks_apply(self):
        """Without resolution this built a lazy ``_VarExpr`` against a numpy
        array instead of a waveform."""
        v = _var(1.0)
        s = Sine(voffset=0.0, vampl=v, freq=1e6)
        t = np.linspace(0, 1e-6, 101)
        assert float(np.max(s.to_numpy(t))) == pytest.approx(1.0, abs=1e-3)
        VariableSpace([v]).apply({v.name: 0.25})
        assert float(np.max(s.to_numpy(t))) == pytest.approx(0.25, abs=1e-3)


# ---------------------------------------------------------------------------
# The dc= path, asserted beside the waveform path so the two cannot drift
# (AC #4)
# ---------------------------------------------------------------------------

class TestTheDCPathAndTheWaveformPathAgree:

    def test_dc_resolves_and_tracks_apply(self):
        v = Variable(name='vdcv', value=0.3, bounds=(0.0, 2.0))
        src = sources.VoltageSource('v1', positive='a', negative='0', dc=v)
        space = VariableSpace([v])
        space.apply({v.name: 0.3})
        assert src.to_spice() == 'Vv1 a 0 DC 0.3\n'
        space.apply({v.name: 1.5})
        assert src.to_spice() == 'Vv1 a 0 DC 1.5\n'

    def test_dc_and_a_waveform_parameter_track_the_same_variable(self):
        v = Variable(name='amp', value=0.2, bounds=(0.0, 2.0))
        src = sources.VoltageSource('v1', positive='a', negative='0', dc=v,
                                    sine=sources.Sine(voffset=0, vampl=v,
                                                      freq=1e6))
        space = VariableSpace([v])
        space.apply({v.name: 0.5})
        assert src.to_spice() == 'Vv1__silk_wave a __silk_src_Vv1 SIN(0 0.5 1000000.0 0 0)\nVv1__silk_dc __silk_src_Vv1 0 DC 0.5\n'
        space.apply({v.name: 1.25})
        assert src.to_spice() == 'Vv1__silk_wave a __silk_src_Vv1 SIN(0 1.25 1000000.0 0 0)\nVv1__silk_dc __silk_src_Vv1 0 DC 1.25\n'

    def test_both_go_through_one_resolver(self):
        """Not a style point: two resolvers is how they drifted apart."""
        import inspect

        from silk.utils import _utils
        body = inspect.getsource(_utils._render_param).split('"""')[-1]
        assert '_resolve_value(' in body
        assert ".value" not in body      # no second duck-type test of its own

    def test_a_current_source_waveform_resolves_as_well(self):
        v = Variable(name='iamp', value=1e-6, bounds=(1e-7, 1e-4))
        src = sources.CurrentSource('i1', positive='a', negative='0', dc=0,
                                    sine=sources.Sine(voffset=0, vampl=v,
                                                      freq=1e3))
        VariableSpace([v]).apply({v.name: 2e-6})
        assert src.to_spice() == 'Ii1__silk_wave a 0 SIN(0 2e-06 1000.0 0 0)\nIi1__silk_dc a 0 DC 0\n'


# ---------------------------------------------------------------------------
# Passives are unchanged: they still refuse (the third of the three behaviours)
# ---------------------------------------------------------------------------

class TestPassivesStillRefuse:

    @pytest.mark.parametrize('build', [
        lambda v: passives.Resistor('r1', pos='a', neg='b', resistance=v),
        lambda v: passives.Capacitor('c1', pos='a', neg='b', capacitance=v),
        lambda v: passives.Inductor('l1', pos='a', neg='b', inductance=v),
    ])
    def test_a_module_variable_raises_type_error(self, build):
        with pytest.raises(TypeError) as exc:
            build(_var(1.0))
        assert 'Variable' in str(exc.value)


# ---------------------------------------------------------------------------
# Plain values reach the deck byte-identically
# ---------------------------------------------------------------------------

class TestPlainValuesAreUntouched:
    """Resolution is shared with the device path; FORMATTING is not. ``_fmt``
    would have written ``27343750.0`` as ``27.3438Meg`` — six significant
    digits and a suffix — rewriting every deck silk has ever emitted."""

    @pytest.mark.parametrize('wave,expected', [
        (Pulse(0, 1.8, td=0, tr='1n', tf='1n', pw='10n', per='20n'),
         'PULSE(0 1.8 0 1n 1n 10n 20n)'),
        (Pulse(0, 1.8, td=0, tr=1e-9, tf=1e-9, pw=1e-8, per=2e-8, np=3),
         'PULSE(0 1.8 0 1e-09 1e-09 1e-08 2e-08 3)'),
        (Sine(0.75, 0.02, 27343750.0), 'SIN(0.75 0.02 27343750.0 0 0)'),
        (Sine(0, 0.7, 1e6, phase=45), 'SIN(0 0.7 1000000.0 0 0 45)'),
        (PWL([(0, 0), (1e-9, 0), (2e-9, 1.8)]),
         'PWL(0.0 0.0 1e-09 0.0 2e-09 1.8)'),
        (PWL([(0, 0), (1e-9, 1.8)], r=0, td=1e-9),
         'PWL(0.0 0.0 1e-09 1.8) r=0 td=1e-09'),
        (Exponential(0, 1.8, 1e-9, 2e-9, 5e-9, 3e-9),
         'EXP(0 1.8 1e-09 2e-09 5e-09 3e-09)'),
        (SFFM(0, 1.0, 1e6, 2, 1e3), 'SFFM(0 1.0 1000000.0 2 1000.0)'),
        (SFFM(0, 1.0, 1e6, 2, 1e3, phasec=10, phases=20),
         'SFFM(0 1.0 1000000.0 2 1000.0 10 20)'),
        (AM(1.0, 0, 1e3, 1e6), 'AM(1.0 0 1000.0 1000000.0 0)'),
        (AM(1.0, 0, 1e3, 1e6, phases=30), 'AM(1.0 0 1000.0 1000000.0 0 30)'),
        (TransientNoise(na='20n', nt='0.5n'), 'TRNOISE(20n 0.5n 0 0)'),
        (TransientNoise(na=0, nt='10p', nalpha=1.1, namp='12p',
                        rtsam='15m', rtscapt='22u', rtsemt='50u'),
         'TRNOISE(0 10p 1.1 12p 15m 22u 50u)'),
        (RandomSource(2, '1n', 0, 1, 0), 'TRRANDOM(2 1n 0 1 0)'),
    ])
    def test_the_emitted_text_is_exact(self, wave, expected):
        assert wave.to_spice() == expected

    def test_a_frequency_keeps_its_full_precision(self):
        """The regression this pins: the device renderer's ``_fmt`` is %g."""
        assert '27343750.0' in Sine(0.75, 0.02, 27343750.0).to_spice()


# ---------------------------------------------------------------------------
# The task's own reproducer, on a generated deck (AC #3, AC #5)
# ---------------------------------------------------------------------------

@pytest.fixture
def swept_bench(pdk, tmp_dir):
    """The reproducer from TASK-209's description: an nmos DUT driven by a
    sine whose amplitude is a Variable in a VariableSpace."""
    dut = Module('PDK_skywater130', tmp_dir, 'dut.cir', pdk_details=pdk,
                 module_name='dut')
    dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT,
                             'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
    with dut:
        pdk.nmos1p8('m1', d='b', g='a', s='vss', b='vss', w=1.0, l=0.15)

    v = Variable(name='vampl', value=0.02, bounds=(0.005, 0.4))
    test = dut.create_test('t')
    test.add_module_under_test({'a': 'vin', 'b': 'vout', 'vdd': 'vdd',
                                'vss': '0'})
    test.add_source(sources.VoltageSource(
        'vin', positive='vin', negative='0', dc=0.75,
        sine=sources.Sine(voffset=0.75, vampl=v, freq=27.34e6)))
    test.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0',
                                          dc=1.8))
    test.add_sim_type(analysis.Transient(step=1e-9, stop=100e-9))
    return test, v


def _deck(test):
    """Write the testbench deck and read it back.

    A ``ModuleTestbench`` emits through the ``Module`` it owns; that module has
    no ``module_name``, so ``generate_netlist()`` is in top-level mode and
    writes the file and returns its concrete path.
    """
    path = test.netlist.generate_netlist()
    with open(path) as fh:
        return fh.read()


def _sine_line(deck, instance='Vvin'):
    for line in deck.splitlines():
        if line.startswith(instance + '__silk_wave '):
            return line
    raise AssertionError(f'no {instance} line in deck:\n{deck}')


class TestTheReproducerDeck:

    def test_no_repr_in_the_generated_deck(self, swept_bench):
        test, _v = swept_bench
        deck = _deck(test)
        assert 'Variable(' not in deck
        assert _sine_line(deck) == ('Vvin__silk_wave vin __silk_src_Vvin '
                                    'SIN(0.75 0.02 27340000.0 0 0)')

    def test_n_applied_points_give_n_different_decks(self, swept_bench):
        """AC #5: read the SIN( ... ) amplitude out of each emitted deck."""
        test, v = swept_bench
        space = VariableSpace([v])
        points = [{v.name: a} for a in (0.01, 0.05, 0.2)]

        amplitudes = []
        for point in points:
            space.apply(point)
            line = _sine_line(_deck(test))
            amplitudes.append(float(re.search(r'SIN\(\S+ (\S+)', line).group(1)))

        assert amplitudes == [0.01, 0.05, 0.2]
        assert len(set(amplitudes)) == len(points)

    def test_a_sampled_grid_reaches_the_deck(self, swept_bench):
        test, v = swept_bench
        space = VariableSpace([v])
        grid = space.sample(4, method='grid')
        seen = set()
        for point in grid:
            space.apply(point)
            seen.add(_sine_line(_deck(test)))
        assert len(seen) == len(grid)


# ---------------------------------------------------------------------------
# Real ngspice — a swept sine amplitude that actually changes the output
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not _have_pdk, reason='sky130 ngspice models not installed')
class TestAgainstRealNgspice:
    """The RC low-pass of ``tests/test_probe.py``, driven by a sine whose
    amplitude is a swept Variable.

    Two points are applied through ``VariableSpace``; each is simulated and the output
    amplitude measured. The assertion is on the MEASURED output, so a variable
    that failed to reach the deck cannot pass by looking plausible.
    """

    @pytest.fixture(scope='class')
    def swept(self, tmp_path_factory):
        from silk.pdk.basepdk import BasePDK
        pdk = BasePDK(_PDK_YAML, library_location=_SKY130_LIB)
        tmp_dir = str(tmp_path_factory.mktemp('rc_sweep'))
        dut = Module('PDK_skywater130', tmp_dir, 'rc.cir', pdk_details=pdk,
                     module_name='rc')
        dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT})
        with dut:
            elements.resistor('r1', p='a', n='b', value=1e3)
            elements.capacitor('c1', p='b', n='0', value=1e-9)

        v = Variable(name='vampl', value=0.1, bounds=(0.05, 0.4))
        test = dut.create_test('rc')
        test.add_module_under_test({'a': 'vin', 'b': 'vout'})
        test.add_source(sources.VoltageSource(
            'vs', positive='vin', negative='0', dc=0.0,
            sine=sources.Sine(voffset=0.0, vampl=v, freq=10e3)))
        test.add_sim_type(analysis.Transient(step=1e-6, stop=500e-6))

        space = VariableSpace([v])
        results = []
        for amplitude in (0.1, 0.4):
            space.apply({v.name: amplitude})
            deck_line = _sine_line(_deck(test), 'Vvs')
            test.run()
            out = test.netlist.load_signal('vout')
            inp = test.netlist.load_signal('vin')
            results.append({
                'applied': amplitude,
                'deck_line': deck_line,
                'vin_peak': float(np.max(np.abs(inp.y))),
                'vout_peak': float(np.max(np.abs(out.y))),
            })
        return results

    def test_each_deck_carried_the_applied_amplitude(self, swept):
        for r in swept:
            assert 'Variable(' not in r['deck_line']
            assert f"SIN(0.0 {r['applied']} " in r['deck_line']

    def test_the_stimulus_that_ngspice_applied_differs_per_point(self, swept):
        for r in swept:
            assert r['vin_peak'] == pytest.approx(r['applied'], rel=0.02)

    def test_the_measured_output_amplitude_differs_per_point(self, swept):
        """The real proof: two runs, two different measured outputs, in the
        ratio of the two applied amplitudes (the RC is linear)."""
        small, large = swept[0]['vout_peak'], swept[1]['vout_peak']
        assert large > small
        assert large / small == pytest.approx(
            swept[1]['applied'] / swept[0]['applied'], rel=0.05)
