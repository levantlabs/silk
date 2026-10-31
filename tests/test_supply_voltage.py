"""The primary supply must be configured and confirmed, never just declared.

TASK-158. ``run(voltage=...)`` was consumed only by the characterization-
context builder: never matched to a source, never emitted, never checked. A
bench wired at 1.8 V and run as ``run(voltage=1.62)`` measured 1.800 V while
the report recorded 1.62. Unlike the temperature bug behind TASK-137/138/139,
there was no second path a careful user could take -- the declaration could
not be made true by any means.

The fix follows the temperature contract: a typed setter that sets the supply
before emission, inheritable from the owning circuit and overridable per test,
confirmed against the deck that is actually launched, with ``run(voltage=...)``
demoted to a deprecated assertion. Which source is the supply is decided from
user-supplied evidence only -- an explicit name, a declared POWER role, or a
bench with a single DC rail -- never from a name list in silk.
"""
import os
import subprocess
from pathlib import Path

import pytest

from tests._mock_rawfile import write_complete_rawfile

from silk import Module, load_pdk, sources
from silk.circuit.module_test import ModuleTestbench
from silk.circuit.ports import PortDir
from silk.elements import elements
from silk.exceptions import (
    ConfigurationError, SimulationConfigurationError, SimulationError,
)
from silk.simulation import Transient
from silk.simulation.conditions import prepare_supply
from tests._env import needs_ngspice, needs_pdk

FIXTURE = Path(__file__).parent / 'fixtures/rawfiles/e_ng_tran.raw'


# --- benches ---------------------------------------------------------------

def _block(tmp_path, pdk, name='block', ports=None):
    """A one-resistor subcircuit that can own a characterization test."""
    block = Module('sky130', str(tmp_path), f'{name}.cir', pdk, module_name=name)
    block.define_module_ports(ports or {'out': PortDir.OUT, 'vss': PortDir.GROUND})
    block.add(elements.resistor('rload', p='out', n='vss', value=1000))
    return block


def _one_rail(block, name='level', dc=1.8):
    """A bench whose only DC source to ground is the rail (rule tier 3)."""
    tb = block.create_test(name)
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_source(sources.VoltageSource('supply', positive='out', negative='0', dc=dc))
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    tb.add_metric('supply', lambda c: float(c.load_signal('out').y[-1]))
    return tb


@pytest.fixture
def pdk():
    return load_pdk('sky130')


@pytest.fixture
def bench(tmp_path, pdk):
    block = _block(tmp_path, pdk)
    yield _one_rail(block)
    block.close()


@pytest.fixture
def two_rails(tmp_path, pdk):
    """A powered DUT plus an input source: two grounded DC sources.

    Only the declared POWER port tells the two apart, which is the whole
    point -- the names ``vsup`` and ``vin_src`` must decide nothing.
    """
    block = _block(tmp_path, pdk, 'powered', ports={
        'in': PortDir.IN, 'out': PortDir.OUT,
        'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
    block.add(elements.resistor('rin', p='in', n='out', value=1000))
    tb = block.create_test('level')
    tb.add_module_under_test({'in': 'vin', 'out': 'out', 'vdd': 'vdd', 'vss': '0'})
    tb.add_source(sources.VoltageSource('vsup', positive='vdd', negative='0', dc=1.8))
    tb.add_source(sources.VoltageSource('vin_src', positive='vin', negative='0', dc=0.9))
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    tb.add_metric('supply', lambda c: float(c.load_signal('out').y[-1]))
    yield tb
    block.close()


@pytest.fixture
def launch(monkeypatch, pdk, mock_ngspice_version):
    """Stub ngspice that records the deck it was handed.

    The deck is the evidence the supply check reads, so a run cannot pass here
    by claiming a voltage the deck never carried.
    """
    monkeypatch.setattr(pdk, 'library_location', os.devnull)
    state = {'decks': []}

    def run(argv, **kwargs):
        deck = Path(argv[-1]).read_text()
        state['decks'].append(deck)
        raw = Path(argv[argv.index('-r') + 1])
        log = Path(argv[argv.index('-o') + 1])
        applied = 27.0
        for line in deck.splitlines():
            for token in line.split():
                if token.lower().startswith('temp='):
                    applied = float(token.split('=', 1)[1])
        log.write_text(
            f'Doing analysis at TEMP = {applied:f} and TNOM = 27.000000\n')
        write_complete_rawfile(raw, Path(argv[-1]).read_text(), template=FIXTURE)
        return subprocess.CompletedProcess(argv, 0, stdout='', stderr='')

    monkeypatch.setattr(subprocess, 'run', run)
    return state


def _rail_line(deck, instance='Vsupply'):
    return next(line for line in deck.splitlines()
                if line.split() and line.split()[0].lower() == instance.lower())


# --- the filed bug ---------------------------------------------------------

def test_a_declaration_the_deck_contradicts_is_refused(bench, launch):
    """The filed case, with the stub standing in for ngspice."""
    with pytest.warns(DeprecationWarning), \
            pytest.raises(SimulationConfigurationError, match='contradicts source'):
        bench.run(voltage=1.62)
    assert launch['decks'] == []          # refused before anything launched


def test_a_matching_declaration_is_confirmed_against_the_deck(bench, launch):
    with pytest.warns(DeprecationWarning):
        report = bench.run(voltage=1.8)
    conditions = report.operating_conditions
    assert (conditions.supply_voltage_v, conditions.supply_basis,
            conditions.supply_source) == (1.8, 'declared-and-emitted', 'supply')
    assert 'DC 1.8' in _rail_line(launch['decks'][0])


def test_configuring_the_supply_changes_the_deck(bench, launch):
    bench.set_supply_voltage(1.62)
    report = bench.run()
    conditions = report.operating_conditions
    assert (conditions.supply_voltage_v, conditions.supply_basis,
            conditions.supply_source) == (1.62, 'configured-and-emitted', 'supply')
    assert 'DC 1.62' in _rail_line(launch['decks'][0])


def test_an_unasked_run_records_the_supply_as_unknown(bench, launch):
    """Nobody claimed a supply, so none is invented -- it stays unknown."""
    conditions = bench.run().operating_conditions
    assert conditions.supply_voltage_v is None
    assert conditions.supply_basis == 'unknown'
    assert conditions.supply_source is None
    assert bench.run().characterization_context is None


# --- how the supply is identified ------------------------------------------

def test_a_declared_power_role_picks_the_rail_out(two_rails, launch):
    """Two grounded DC sources; only the declared POWER port separates them."""
    two_rails.set_supply_voltage(1.62)
    report = two_rails.run()
    assert report.operating_conditions.supply_source == 'vsup'
    assert 'DC 1.62' in _rail_line(launch['decks'][0], 'Vvsup')
    assert 'DC 0.9' in _rail_line(launch['decks'][0], 'Vvin_src')


def test_an_explicit_name_needs_no_roles_at_all(tmp_path, pdk, launch):
    """Nets carry no roles here, so only the author's own name decides."""
    block = _block(tmp_path, pdk, 'plain')
    tb = block.create_test('level')
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_source(sources.VoltageSource('a', positive='out', negative='0', dc=1.8))
    tb.add_source(sources.VoltageSource('b', positive='aux', negative='0', dc=0.9))
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    tb.add_metric('supply', lambda c: float(c.load_signal('out').y[-1]))
    tb.set_supply_voltage(1.62, source='b')
    report = tb.run()
    assert report.operating_conditions.supply_source == 'b'
    assert 'DC 1.62' in _rail_line(launch['decks'][0], 'Vb')
    assert 'DC 1.8' in _rail_line(launch['decks'][0], 'Va')
    block.close()


def test_an_ambiguous_bench_refuses_rather_than_choosing(tmp_path, pdk, launch):
    block = _block(tmp_path, pdk, 'plain')
    tb = block.create_test('level')
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_source(sources.VoltageSource('a', positive='out', negative='0', dc=1.8))
    tb.add_source(sources.VoltageSource('b', positive='aux', negative='0', dc=0.9))
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    tb.add_metric('supply', lambda c: float(c.load_signal('out').y[-1]))
    tb.set_supply_voltage(1.62)
    with pytest.raises(SimulationConfigurationError, match='equally supply-shaped'):
        tb.run()
    # Silent while nobody asks: an unresolved supply is simply unknown.
    tb.set_supply_voltage(None)
    assert tb.run().operating_conditions.supply_voltage_v is None
    block.close()


def test_two_power_rails_are_never_collapsed_into_one(tmp_path, pdk, launch):
    block = _block(tmp_path, pdk, 'dual', ports={
        'out': PortDir.OUT, 'vdda': PortDir.POWER,
        'vddb': PortDir.POWER, 'vss': PortDir.GROUND})
    block.add(elements.resistor('ra', p='vdda', n='out', value=1000))
    block.add(elements.resistor('rb', p='vddb', n='vss', value=1000))
    tb = block.create_test('level')
    tb.add_module_under_test({'out': 'out', 'vdda': 'va', 'vddb': 'vb', 'vss': '0'})
    tb.add_source(sources.VoltageSource('sa', positive='va', negative='0', dc=1.8))
    tb.add_source(sources.VoltageSource('sb', positive='vb', negative='0', dc=1.2))
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    tb.add_metric('supply', lambda c: float(c.load_signal('out').y[-1]))
    tb.set_supply_voltage(1.62)
    with pytest.raises(SimulationConfigurationError, match='declared POWER rail'):
        tb.run()
    # The author's own choice settles it, and the other rail is untouched.
    tb.set_supply_voltage(1.62, source='sa')
    tb.run()
    assert 'DC 1.62' in _rail_line(launch['decks'][-1], 'Vsa')
    assert 'DC 1.2' in _rail_line(launch['decks'][-1], 'Vsb')
    block.close()


def test_a_time_varying_supply_takes_no_scalar_setpoint(tmp_path, pdk, launch):
    block = _block(tmp_path, pdk, 'pulsed')
    tb = block.create_test('level')
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_source(sources.VoltageSource(
        'supply', positive='out', negative='0', dc=0,
        pulse=sources.Pulse(v1=0, v2=1.8, tr='1n', tf='1n', pw='1u', per='2u')))
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    tb.add_metric('supply', lambda c: float(c.load_signal('out').y[-1]))
    # Never a candidate ...
    assert tb.run().operating_conditions.supply_voltage_v is None
    # ... and naming it explicitly says why, instead of writing a DC value
    # next to a waveform.
    tb.set_supply_voltage(1.8, source='supply')
    with pytest.raises(SimulationConfigurationError, match='carries a waveform'):
        tb.run()
    block.close()


def test_a_bench_with_no_source_cannot_confirm_a_declaration(tmp_path, pdk, launch):
    block = _block(tmp_path, pdk, 'sourceless')
    tb = block.create_test('level')
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add(elements.resistor('rtb', p='out', n='0', value=1000))
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    tb.add_metric('supply', lambda c: float(c.load_signal('out').y[-1]))
    with pytest.warns(DeprecationWarning), \
            pytest.raises(SimulationConfigurationError, match='no voltage source'):
        tb.run(voltage=1.8)
    block.close()


def test_a_non_literal_dc_value_establishes_nothing(bench, launch):
    bench.netlist.sig_sources_dict['supply'].dc = '{vdd_param}'
    request = prepare_supply(bench.netlist, apply=False)
    assert request.voltage is None and 'not a literal' in request.reason
    with pytest.warns(DeprecationWarning), \
            pytest.raises(SimulationConfigurationError, match='not a literal'):
        bench.run(voltage=1.8)


def test_a_named_source_that_is_not_there_is_named_back(bench, launch):
    bench.set_supply_voltage(1.8, source='nope')
    with pytest.raises(SimulationConfigurationError, match="'nope' is not on this circuit"):
        bench.run()


# --- confirmation against the launched deck --------------------------------

def test_custom_spice_that_redeclares_the_rail_is_caught(bench, launch):
    bench.set_supply_voltage(1.62)
    bench.add_custom_line('Vsupply out 0 DC 1.8')
    with pytest.raises(SimulationError, match='declares Vsupply 2 times'):
        bench.run()
    assert launch['decks'] == []           # refused before the launch


# --- the setter's own contract ---------------------------------------------

def test_a_subcircuit_owning_no_tests_refuses_the_setter(tmp_path, pdk):
    """The refusal survives where it is right: nothing would apply it.

    TASK-168 widened the guard from "is this a subcircuit" to "is there
    anything here for this setting to apply to". A block that owns registered
    tests now accepts it (see ``tests/test_block_default_conditions.py``);
    one that owns none still does not, because an instantiated ``.subckt``
    holds no rail of its own.
    """
    block = _block(tmp_path, pdk, 'testless')
    try:
        with pytest.raises(ValueError, match='owns no registered tests'):
            block.set_supply_voltage(1.8)
        assert block._supply_explicit is False
        assert block._supply_voltage is None
    finally:
        block.close()


@pytest.mark.parametrize('volts', [True, float('nan'), float('inf'), '1.8'])
def test_the_typed_domain_is_atomic(bench, volts):
    from silk.exceptions import CharacterizationContextError
    with pytest.raises(CharacterizationContextError):
        bench.set_supply_voltage(volts)
    assert bench.netlist._supply_voltage is None


def test_clearing_stops_configuring_and_does_not_restore(bench, launch):
    bench.set_supply_voltage(1.62)
    bench.run()
    bench.set_supply_voltage(None)
    report = bench.run()
    # The setting is gone, so nothing is claimed; the source keeps the value
    # it was last given, which is what the deck carries.
    assert report.operating_conditions.supply_voltage_v is None
    assert 'DC 1.62' in _rail_line(launch['decks'][-1])


def test_a_source_name_needs_a_voltage(bench):
    with pytest.raises(ValueError, match='takes no source'):
        bench.set_supply_voltage(None, source='supply')


# --- inheritance (the TASK-162 shape) --------------------------------------

@pytest.fixture
def tree(tmp_path, pdk):
    """A top-level circuit owning one block, which owns one test.

    The only shape in which an owner setting is reachable: a subcircuit still
    refuses the setter, so the circuit that can hold a whole-tree setting is
    the top-level root of the recursive run.
    """
    block = _block(tmp_path, pdk)
    _one_rail(block)
    root = Module('sky130', str(tmp_path), 'root.cir', pdk)
    root.add_module(block, 'u1', nets={'out': 'chip_out', 'vss': '0'})
    yield root, block
    block.close()
    root.close()


def live_test(root, name='level', instance='u1'):
    return root.elem_dict[instance]['device']._tests[name]


def test_an_owner_setting_reaches_a_registered_test(tree, launch):
    root, _block = tree
    root.set_supply_voltage(1.62)
    results = root.run_all_tests(recursive=True)
    report = results['block.level']
    assert not isinstance(report, dict), report
    assert report.operating_conditions.supply_voltage_v == 1.62
    assert 'DC 1.62' in _rail_line(launch['decks'][-1])


def test_an_owner_setting_is_re_read_at_every_run(tree, launch):
    root, _block = tree
    root.set_supply_voltage(1.62)
    root.run_all_tests(recursive=True)
    root.set_supply_voltage(1.2)
    root.run_all_tests(recursive=True)
    assert 'DC 1.2' in _rail_line(launch['decks'][-1])
    root.set_supply_voltage(None)
    report = root.run_all_tests(recursive=True)['block.level']
    assert report.operating_conditions.supply_voltage_v is None


def test_an_inherited_value_is_not_the_tests_own_setting(tree, launch):
    root, _block = tree
    root.set_supply_voltage(1.62)
    root.run_all_tests(recursive=True)
    assert live_test(root).netlist._supply_explicit is False


def test_a_test_setting_wins_over_its_owner(tree, launch):
    root, _block = tree
    root.set_supply_voltage(1.62)
    live_test(root).set_supply_voltage(1.2)
    report = root.run_all_tests(recursive=True)['block.level']
    assert report.operating_conditions.supply_voltage_v == 1.2


def test_a_test_can_opt_out_of_its_owners_supply(tree, launch):
    root, _block = tree
    root.set_supply_voltage(1.62)
    live_test(root).set_supply_voltage(None)
    report = root.run_all_tests(recursive=True)['block.level']
    assert report.operating_conditions.supply_voltage_v is None
    assert 'DC 1.8' in _rail_line(launch['decks'][-1])


def test_a_recursive_run_changes_no_stored_setting(tree, launch):
    root, _block = tree
    root.set_supply_voltage(1.62)
    root.run_all_tests(recursive=True)
    assert not hasattr(live_test(root), '_inherited_supply_override')


def test_a_misconfigured_supply_stops_the_whole_run(two_rails, launch):
    """A ConfigurationError says the request is wrong, so it propagates out of
    run_all_tests rather than becoming one error string per test (TASK-163)."""
    two_rails.module.create_test('second')
    two_rails.set_supply_voltage(1.62, source='missing')
    with pytest.raises(ConfigurationError):
        two_rails.module.run_all_tests()


# --- cache ------------------------------------------------------------------

def test_changed_supply_cannot_reuse_cached_evidence(tmp_path, pdk, launch):
    block = _block(tmp_path, pdk)
    tb = _one_rail(block)
    tb._cache_enabled = True
    tb.set_supply_voltage(1.62)
    first = tb.run()
    assert tb.run() is first
    tb.set_supply_voltage(1.2)
    with pytest.raises(SimulationConfigurationError, match='clear_cache'):
        tb.run()
    tb.clear_cache()
    assert tb.run().operating_conditions.supply_voltage_v == 1.2
    block.close()


# --- nothing TASK-139/162 established regresses ------------------------------

def test_temperature_and_supply_are_independent(bench, launch):
    bench.set_temperature(85)
    bench.set_supply_voltage(1.62)
    conditions = bench.run().operating_conditions
    assert conditions.temperature_c == 85
    assert conditions.temperature_basis == 'configured-and-observed'
    assert conditions.supply_voltage_v == 1.62
    deck = launch['decks'][-1]
    assert 'temp=85' in deck.lower() and 'DC 1.62' in _rail_line(deck)


def test_the_six_field_descriptor_is_unchanged(bench, launch):
    from silk.db.context import CharacterizationContext
    bench.set_temperature(27)
    bench.set_supply_voltage(1.62)
    context = bench.run().characterization_context
    assert isinstance(context, CharacterizationContext)
    # The bench is ideal, so its corner is none, not the label (TASK-406).
    from silk.simulation import ConditionState
    assert (context.corner, context.temperature, context.voltage) == (
        ConditionState.NONE, 27.0, 1.62)
    assert set(vars(context)) == {'owner_path', 'test_name', 'parameters',
                                  'corner', 'temperature', 'voltage'}


# --- real ngspice ------------------------------------------------------------

@needs_ngspice
@needs_pdk
def test_real_ngspice_reproduces_the_filed_case(tmp_path, pdk):
    """A bench wired at 1.8 V, run with a different declared supply.

    Before TASK-158 this simulated 1.800 V and reported 1.62. It now refuses,
    and configuring 1.62 makes the report true by actually simulating it.
    """
    block = _block(tmp_path, pdk, 'filed')
    tb = _one_rail(block)
    with pytest.warns(DeprecationWarning), \
            pytest.raises(SimulationConfigurationError, match='contradicts source'):
        tb.run(voltage=1.62)

    honest = tb.run(voltage=1.8)
    assert honest.metrics['supply'] == pytest.approx(1.8, abs=1e-6)
    assert honest.operating_conditions.supply_voltage_v == 1.8

    tb.set_supply_voltage(1.62)
    configured = tb.run()
    assert configured.metrics['supply'] == pytest.approx(1.62, abs=1e-6)
    assert configured.operating_conditions.supply_voltage_v == 1.62
    assert configured.operating_conditions.supply_basis == 'configured-and-emitted'
    block.close()


@needs_ngspice
@needs_pdk
@pytest.mark.slow
def test_real_ngspice_owner_setting_reaches_a_transistor_level_test(tmp_path, pdk):
    """The inherited supply on the path an actual characterization run takes."""
    block = _block(tmp_path, pdk, 'block')
    block.add(pdk.nmos1p8('mload', d='out', g='out', s='vss', b='vss', w=1, l=0.15))
    _one_rail(block, 'inherits')
    _one_rail(block, 'overrides').set_supply_voltage(1.2)
    root = Module('sky130', str(tmp_path), 'root.cir', pdk)
    root.add_module(block, 'u1', nets={'out': 'chip_out', 'vss': '0'})
    root.set_supply_voltage(1.62)

    results = root.run_all_tests(recursive=True)
    for name, expected in (('inherits', 1.62), ('overrides', 1.2)):
        report = results[f'block.{name}']
        assert not isinstance(report, dict), report
        assert report.metrics['supply'] == pytest.approx(expected, abs=1e-6)
        assert report.operating_conditions.supply_voltage_v == expected
        assert report.operating_conditions.supply_source == 'supply'
    block.close()
    root.close()
