"""evaluate(rawfile=...) declares none per fact; an undeclared corner is unknown (TASK-407)."""
from pathlib import Path

import pytest

from silk import Module, load_pdk, sources
from silk.elements import elements
from silk.exceptions import CharacterizationContextError
from silk.simulation import ConditionState, Transient

NONE = ConditionState.NONE
RAW = str(Path(__file__).parent / 'fixtures/rawfiles/e_ng_tran.raw')


@pytest.fixture
def tb(tmp_path):
    mod = Module('sky130', str(tmp_path), 'dut.cir', load_pdk('sky130'),
                 module_name='load')
    mod.define_module_ports({'out': 'out', 'vss': 'ground'})
    mod.add(elements.resistor('rload', p='out', n='vss', value=1000))
    t = mod.create_test('level')
    t.add_module_under_test({'out': 'out', 'vss': '0'})
    t.add_source(sources.VoltageSource('supply', positive='out', negative='0', dc=1.8))
    t.add_sim_type(Transient(step=1e-6, stop=2e-6))
    t.add_metric('supply', lambda c: float(c.load_signal('out').y[-1]))
    yield t
    mod.close()


def test_none_per_fact_is_recorded_caller_declared_and_scopes(tb):
    r = tb.evaluate(rawfile=RAW, corner=NONE, temperature=NONE, voltage=NONE)
    c = r.operating_conditions
    for fact in ('corner', 'temperature', 'supply'):
        assert c.state(fact) is NONE
        assert c.scalar(fact) is NONE
    assert (c.corner_basis, c.temperature_basis, c.supply_basis) == (
        'caller-declared',) * 3
    ctx = r.characterization_context
    assert (ctx.corner, ctx.temperature, ctx.voltage) == (NONE,) * 3


def test_mixed_fixed_and_none_scopes(tb):
    r = tb.evaluate('tt', rawfile=RAW, temperature=27.0, voltage=NONE)
    ctx = r.characterization_context
    assert (ctx.corner, ctx.temperature, ctx.voltage) == ('tt', 27.0, NONE)
    assert r.operating_conditions.supply_kind == 'none'
    assert r.operating_conditions.supply_voltage_v is None


@pytest.mark.parametrize('fact', ['corner', 'temperature', 'voltage'])
@pytest.mark.parametrize('state', [ConditionState.UNKNOWN, ConditionState.FIXED,
                                   ConditionState.SWEEP])
def test_other_states_raise_and_load_nothing(tb, fact, state):
    kwargs = {'corner': 'tt', 'temperature': 27.0, 'voltage': 1.8, fact: state}
    with pytest.raises(CharacterizationContextError):
        tb.evaluate(rawfile=RAW, **kwargs)
    assert tb.netlist.operating_conditions is None


def test_lone_none_is_recorded_unscoped(tb):
    r = tb.evaluate('tt', rawfile=RAW, voltage=NONE)
    assert r.characterization_context is None
    c = r.operating_conditions
    assert c.supply_kind == 'none' and c.temperature_kind == 'unknown'


def test_undeclared_corner_is_unknown_and_unscoped(tb):
    r = tb.evaluate(rawfile=RAW, temperature=27.0, voltage=1.8)
    c = r.operating_conditions
    assert (c.corner, c.corner_kind, c.corner_basis) == (None, 'unknown', 'unknown')
    assert r.characterization_context is None
    assert (c.temperature_c, c.supply_voltage_v) == (27.0, 1.8)
    assert r.corner is None


def test_set_corner_is_caller_declared(tb):
    tb.set_corner('ff')
    r = tb.evaluate(rawfile=RAW, temperature=27.0, voltage=1.8)
    assert r.operating_conditions.corner == 'ff'
    assert r.operating_conditions.corner_basis == 'caller-declared'
    assert r.characterization_context.corner == 'ff'


def test_redeclaring_external_memory_reuses_none(tb):
    tb.evaluate(rawfile=RAW, corner=NONE, temperature=27.0, voltage=NONE)
    r = tb.evaluate(temperature=35.0)
    ctx = r.characterization_context
    assert (ctx.corner, ctx.temperature, ctx.voltage) == (NONE, 35.0, NONE)


def test_silk_memory_cannot_be_relabelled_none(tb, monkeypatch):
    r = tb.evaluate(rawfile=RAW, corner=NONE, temperature=NONE, voltage=NONE)
    # An external snapshot may be redeclared; confirm_conditions compares scalars.
    from silk.simulation.conditions import confirm_conditions
    confirm_conditions(r.operating_conditions, corner=NONE)
    with pytest.raises(CharacterizationContextError):
        confirm_conditions(r.operating_conditions, corner='tt')


def test_temperature_none_on_a_sweep_axis_raises(tb, monkeypatch):
    from silk.simulation import conditions
    from silk.simulation.conditions import ConditionSweep
    monkeypatch.setattr(conditions, 'external_sweep',
                        lambda raw: ConditionSweep(-40.0, 125.0, 5.0))
    with pytest.raises(CharacterizationContextError, match='sweep'):
        tb.evaluate('tt', rawfile=RAW, temperature=NONE, voltage=1.8)
    assert tb.netlist.operating_conditions is None
    r = tb.evaluate(rawfile=RAW)
    assert r.operating_conditions.temperature_kind == 'sweep'
