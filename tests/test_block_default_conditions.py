"""A block that owns registered tests sets one default for all of them.

TASK-168. ``set_temperature`` and ``set_supply_voltage`` refused any Module
with a ``module_name``, which is exactly the canonical embedded-testbench
shape: a block owns its tests and users run them with ``block.run_all_tests()``.
A designer with one block and five tests who wanted all five hot had to set it
five times, or invent a wrapper circuit purely to hold the setting.

The owner ruled for one default with a per-test override, so the guard changed
from "is this a subcircuit" to "is there anything here for this setting to
apply to". A subcircuit that owns NO tests is still refused -- there the
setting would be accepted and then ignored, which is the condition
``module_name`` was only ever a proxy for. Precedence is unchanged from
TASK-162: nearest setting wins.

Both settings are covered together because they share one inheritance
mechanism and shared one refusal; they must not diverge.
"""
import os
import subprocess
from pathlib import Path

import pytest

from tests._mock_rawfile import write_complete_rawfile

from silk import Module, load_pdk, sources
from silk.circuit.ports import PortDir
from silk.elements import elements
from silk.simulation import Transient
from tests._env import needs_ngspice, needs_pdk

FIXTURE = Path(__file__).parent / 'fixtures/rawfiles/e_ng_tran.raw'


# --- benches ---------------------------------------------------------------

def _block(tmp_path, pdk, name='block'):
    """A one-resistor subcircuit that can own characterization tests."""
    block = Module('sky130', str(tmp_path), f'{name}.cir', pdk, module_name=name)
    block.define_module_ports({'out': PortDir.OUT, 'vss': PortDir.GROUND})
    block.add(elements.resistor('rload', p='out', n='vss', value=1000))
    return block


def _test_on(block, name, dc=1.8):
    """A bench whose only grounded DC source is the rail (supply rule tier 3)."""
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
def block(tmp_path, pdk):
    """A block owning two tests: one inherits the default, one overrides it."""
    block = _block(tmp_path, pdk)
    _test_on(block, 'inherits')
    _test_on(block, 'overrides')
    yield block
    block.close()


@pytest.fixture
def launch(monkeypatch, pdk, mock_ngspice_version):
    """Stub ngspice echoing the temperature the deck actually asked for.

    Keyed off the emitted deck rather than a fixed number, so a run cannot
    pass by claiming conditions the deck never carried.
    """
    monkeypatch.setattr(pdk, 'library_location', os.devnull)
    state = {'decks': [], 'temperatures': []}

    def run(argv, **kwargs):
        deck = Path(argv[-1]).read_text()
        raw = Path(argv[argv.index('-r') + 1])
        log = Path(argv[argv.index('-o') + 1])
        applied = 27.0
        for line in deck.splitlines():
            for token in line.split():
                if token.lower().startswith('temp='):
                    applied = float(token.split('=', 1)[1])
        state['decks'].append(deck)
        state['temperatures'].append(applied)
        log.write_text(
            f'Doing analysis at TEMP = {applied:f} and TNOM = 27.000000\n')
        write_complete_rawfile(raw, Path(argv[-1]).read_text(), template=FIXTURE)
        return subprocess.CompletedProcess(argv, 0, stdout='', stderr='')

    monkeypatch.setattr(subprocess, 'run', run)
    return state


def _rail_volts(deck, instance='Vsupply'):
    line = next(l for l in deck.splitlines()
                if l.split() and l.split()[0].lower() == instance.lower())
    tokens = line.split()
    return float(tokens[tokens.index('DC') + 1])


# --- the setter is accepted where something applies it ---------------------

def test_a_block_that_owns_tests_accepts_both_setters(block):
    block.set_temperature(85)
    block.set_supply_voltage(1.62)
    assert block.simulator_options['temp'] == 85
    assert block._temperature_explicit is True
    assert block._supply_voltage == 1.62
    assert block._supply_explicit is True


def test_registering_a_test_is_what_opens_the_setter(tmp_path, pdk):
    """The guard reads the Module as it is at the call, both ways round."""
    bare = _block(tmp_path, pdk, 'late')
    try:
        with pytest.raises(ValueError, match='owns no registered tests'):
            bare.set_temperature(85)
        with pytest.raises(ValueError, match='owns no registered tests'):
            bare.set_supply_voltage(1.62)
        _test_on(bare, 'now')
        assert bare.set_temperature(85) is bare
        assert bare.set_supply_voltage(1.62) is bare
    finally:
        bare.close()


def test_a_subcircuit_owning_no_tests_is_left_unmarked_by_a_refusal(tmp_path, pdk):
    bare = _block(tmp_path, pdk, 'testless')
    try:
        for call in (lambda: bare.set_temperature(85),
                     lambda: bare.set_supply_voltage(1.62)):
            with pytest.raises(ValueError):
                call()
        assert bare._temperature_explicit is False
        assert 'temp' not in bare.simulator_options
        assert bare._supply_explicit is False
        assert bare._supply_voltage is None
    finally:
        bare.close()


def test_a_top_level_module_with_no_tests_is_unaffected(tmp_path, pdk):
    """``Module.simulate()`` on a directly simulated circuit is untouched."""
    top = Module('sky130', str(tmp_path), 'top.cir', pdk)
    try:
        assert top.module_name is None and not top.list_tests()
        top.set_temperature(85)
        top.set_supply_voltage(1.62)
        assert top.simulator_options['temp'] == 85 and top._supply_voltage == 1.62
    finally:
        top.close()


# --- one call reaches every test the block owns ----------------------------

def test_a_block_default_reaches_all_of_its_tests(block, launch):
    """The filed case: no wrapper circuit, no per-test repetition."""
    block.set_temperature(85)
    block.set_supply_voltage(1.62)
    results = block.run_all_tests()
    assert set(results) == {'block.inherits', 'block.overrides'}
    for report in results.values():
        assert not isinstance(report, dict), report
        assert report.operating_conditions.temperature_c == 85
        assert report.operating_conditions.temperature_basis == 'configured-and-observed'
        assert report.operating_conditions.supply_voltage_v == 1.62
    assert launch['temperatures'] == [85.0, 85.0]
    assert [_rail_volts(deck) for deck in launch['decks']] == [1.62, 1.62]


def test_run_test_on_the_block_resolves_the_block_default(block, launch):
    """A single named run takes the owner's setting, like a walked one."""
    block.set_temperature(85)
    block.set_supply_voltage(1.62)
    conditions = block.run_test('inherits').operating_conditions
    assert conditions.temperature_c == 85
    assert conditions.supply_voltage_v == 1.62


def test_a_block_default_is_re_read_at_every_run(block, launch):
    block.set_temperature(85)
    block.set_supply_voltage(1.62)
    block.run_test('inherits')
    block.set_temperature(-40)
    block.set_supply_voltage(1.2)
    block.run_test('inherits')
    block.set_temperature(None)
    block.set_supply_voltage(None)
    cleared = block.run_test('inherits')
    assert launch['temperatures'] == [85.0, -40.0, 27.0]
    # Clearing clears the SETTING, not the source: the rail keeps the DC value
    # it was last given (documented on set_supply_voltage), and the report
    # stops claiming a supply rather than claiming that leftover.
    assert [_rail_volts(deck) for deck in launch['decks']] == [1.62, 1.2, 1.2]
    assert 'temp=' not in launch['decks'][-1]
    assert cleared.operating_conditions.temperature_basis == 'backend-observed'
    assert cleared.operating_conditions.supply_voltage_v is None


def test_an_inherited_block_default_never_becomes_the_tests_own(block, launch):
    block.set_temperature(85)
    block.set_supply_voltage(1.62)
    block.run_all_tests()
    for tb in (block._tests['inherits'], block._tests['overrides']):
        assert tb.netlist._temperature_explicit is False
        assert tb.netlist._supply_explicit is False
        assert not hasattr(tb, '_inherited_temperature_override')
        assert not hasattr(tb, '_inherited_supply_override')


# --- the per-test override still wins --------------------------------------

def test_a_per_test_setting_beats_the_block_default(block, launch):
    block.set_temperature(85)
    block.set_supply_voltage(1.62)
    block._tests['overrides'].set_temperature(-40)
    block._tests['overrides'].set_supply_voltage(1.98)
    results = block.run_all_tests()
    inherited = results['block.inherits'].operating_conditions
    overridden = results['block.overrides'].operating_conditions
    assert (inherited.temperature_c, inherited.supply_voltage_v) == (85, 1.62)
    assert (overridden.temperature_c, overridden.supply_voltage_v) == (-40, 1.98)
    # A later change on the block does not take the test's own setting away.
    block.set_temperature(0)
    block.set_supply_voltage(1.1)
    again = block.run_test('overrides').operating_conditions
    assert (again.temperature_c, again.supply_voltage_v) == (-40, 1.98)


def test_a_test_can_opt_out_of_the_block_default(block, launch):
    block.set_temperature(85)
    block.set_supply_voltage(1.62)
    block._tests['overrides'].set_temperature(None)
    block._tests['overrides'].set_supply_voltage(None)
    conditions = block.run_test('overrides').operating_conditions
    assert conditions.temperature_basis == 'backend-observed'
    assert conditions.supply_voltage_v is None
    assert _rail_volts(launch['decks'][-1]) == 1.8      # the bench's own wiring


# --- nearest wins, against an ancestor -------------------------------------

def test_a_block_default_beats_an_ancestors_setting(tmp_path, pdk, launch):
    """Root, block and test are three levels; the nearest one in force wins."""
    template = _block(tmp_path, pdk, 'child')
    _test_on(template, 'inherits')
    _test_on(template, 'overrides')
    root = Module('sky130', str(tmp_path), 'root.cir', pdk)
    root.add_module(template, 'u1', nets={'out': 'chip_out', 'vss': '0'})
    # add_element deep-copies, so the live instance is the one that runs.
    live = root.elem_dict['u1']['device']
    root.set_temperature(85)
    root.set_supply_voltage(1.62)
    live.set_temperature(-40)
    live.set_supply_voltage(1.2)
    live._tests['overrides'].set_temperature(125)
    live._tests['overrides'].set_supply_voltage(1.98)

    results = root.run_all_tests(recursive=True)
    inherited = results['child.inherits'].operating_conditions
    overridden = results['child.overrides'].operating_conditions
    assert (inherited.temperature_c, inherited.supply_voltage_v) == (-40, 1.2)
    assert (overridden.temperature_c, overridden.supply_voltage_v) == (125, 1.98)
    # The walk changed nothing stored on the block or on its tests.
    assert live.simulator_options['temp'] == -40
    assert live._supply_voltage == 1.2
    assert not hasattr(live._tests['inherits'], '_inherited_temperature_override')
    assert not hasattr(live._tests['inherits'], '_inherited_supply_override')
    assert live._tests['inherits'].netlist._temperature_explicit is False
    template.close()
    root.close()


def test_an_ancestor_still_reaches_a_block_that_set_nothing(tmp_path, pdk, launch):
    """Widening the guard must not cost TASK-162 its whole-tree reach."""
    template = _block(tmp_path, pdk, 'child')
    _test_on(template, 'level')
    root = Module('sky130', str(tmp_path), 'root.cir', pdk)
    root.add_module(template, 'u1', nets={'out': 'chip_out', 'vss': '0'})
    root.set_temperature(85)
    root.set_supply_voltage(1.62)
    conditions = root.run_all_tests(recursive=True)['child.level'].operating_conditions
    assert conditions.temperature_c == 85
    assert conditions.supply_voltage_v == 1.62
    template.close()
    root.close()


# --- real ngspice ----------------------------------------------------------

@needs_ngspice
@needs_pdk
@pytest.mark.slow
def test_real_ngspice_block_default_with_one_test_overriding(tmp_path, pdk):
    """The whole contract on a real simulation: two tests, one of each.

    A real device pulls in the sky130 model library, so both settings are
    proven on the path an actual characterization run takes, and the
    temperature is read back from ngspice's own log rather than asserted.
    """
    block = _block(tmp_path, pdk, 'hotblock')
    block.add(pdk.nmos1p8('mload', d='out', g='out', s='vss', b='vss', w=1, l=0.15))
    _test_on(block, 'inherits')
    overrides = _test_on(block, 'overrides')
    overrides.set_temperature(-40)
    overrides.set_supply_voltage(1.98)

    block.set_temperature(85)            # one call, no wrapper circuit
    block.set_supply_voltage(1.62)

    results = block.run_all_tests()
    expected = {'inherits': (85, 1.62), 'overrides': (-40, 1.98)}
    for name, (temp_c, volts) in expected.items():
        report = results[f'hotblock.{name}']
        assert not isinstance(report, dict), report
        conditions = report.operating_conditions
        assert conditions.temperature_c == temp_c
        assert conditions.temperature_basis == 'configured-and-observed'
        assert conditions.supply_voltage_v == volts
        assert conditions.supply_basis == 'configured-and-emitted'
        # The simulator really solved at that supply, not just declared it.
        assert report.metrics['supply'] == pytest.approx(volts, abs=1e-6)
        log = (Path(report.sim_results.results_file).parent
               / 'output.log').read_text()
        assert f'TEMP = {temp_c:.6f}' in log and 'TNOM = 27.000000' in log
    block.close()
