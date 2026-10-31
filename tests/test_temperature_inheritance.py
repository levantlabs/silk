"""A registered test follows its owner's temperature unless it set its own.

TASK-162. The setter was accepted on a circuit that owns tests and then
ignored, because ``run()`` executes the testbench's own netlist. The owner
ruled for inheritance with a per-test override, on the shape TASK-150
established for the simulator: an explicitness flag the public setter sets and
the constructor clears, and a value re-read at every run instead of copied
once.
"""
import os
import subprocess
from pathlib import Path

import pytest

from tests._mock_rawfile import write_complete_rawfile

from silk import Module, load_pdk, sources
from silk.circuit.module_test import ModuleTestbench
from silk.elements import elements
from silk.exceptions import CharacterizationContextError, SimulationError
from silk.simulation import Transient
from tests._env import needs_ngspice, needs_pdk

FIXTURE = Path(__file__).parent / 'fixtures/rawfiles/e_ng_tran.raw'


def _block(tmp_path, pdk, name):
    """A one-resistor subcircuit that owns its characterization test."""
    block = Module('sky130', str(tmp_path), f'{name}.cir', pdk, module_name=name)
    block.define_module_ports({'out': 'out', 'vss': 'ground'})
    block.add(elements.resistor('rload', p='out', n='vss', value=1000))
    return block


def _test_on(block, name='level'):
    tb = block.create_test(name)
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_source(sources.VoltageSource('supply', positive='out', negative='0', dc=1.8))
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    tb.add_metric('supply', lambda c: float(c.load_signal('out').y[-1]))
    return tb


@pytest.fixture
def tree(tmp_path):
    """A top-level circuit owning one block, which owns one test.

    The whole-tree shape: the root holds a setting that reaches the block's
    test through the recursive walk. Since TASK-168 the block itself may also
    hold one, because it owns a registered test; the nearer setting wins.
    """
    pdk = load_pdk('sky130')
    block = _block(tmp_path, pdk, 'block')
    _test_on(block)
    root = Module('sky130', str(tmp_path), 'root.cir', pdk)
    root.add_module(block, 'u1', nets={'out': 'chip_out', 'vss': '0'})
    yield root, block
    block.close()
    root.close()


def live_test(root, name='level', instance='u1'):
    """The test of the live instance a recursive run actually walks.

    ``add_element`` deep-copies its child, so the registered test that runs is
    the copy's, exactly as TASK-150 requires for per-instance selection.
    """
    return root.elem_dict[instance]['device']._tests[name]


@pytest.fixture
def launch(monkeypatch, mock_ngspice_version):
    """Stub ngspice that reports the temperature the deck actually asked for.

    Echoing the emitted setting rather than a fixed number keeps the run
    honest: the configured-versus-observed check still has to agree, so a test
    cannot pass by claiming a temperature the deck never carried.
    """
    state = {'decks': [], 'temperatures': [], 'observed': None}

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
        if state['observed'] is not None:
            applied = state['observed']
        log.write_text(f'Doing analysis at TEMP = {applied:f} and TNOM = 27.000000\n')
        write_complete_rawfile(raw, Path(argv[-1]).read_text(), template=FIXTURE)
        return subprocess.CompletedProcess(argv, 0, stdout='', stderr='')

    monkeypatch.setattr(subprocess, 'run', run)
    return state


@pytest.fixture
def top_level(tmp_path):
    """A top-level circuit that owns a test — the shape this bug was filed on.

    Its test is wired without ``add_module_under_test``, and still has to be
    after TASK-167: a Module with no ``module_name`` is a top-level circuit,
    so there is no ``.subckt`` for a testbench to instantiate. TASK-167 made
    that refusal explicit — ``add_module_under_test`` now raises
    ``NotASubcircuitError`` naming the module instead of letting the netlister
    fail later on a ``None`` concatenation — so the ordinary wiring is not
    available here BY DESIGN, not by defect. Giving this owner a
    ``module_name`` to make it available would make it a subcircuit and
    destroy the very shape this file is about. Skipping the instantiation
    isolates the temperature contract while keeping the owner/test
    relationship real.
    """
    pdk = load_pdk('sky130')
    owner = Module('sky130', str(tmp_path), 'owner.cir', pdk)
    tb = ModuleTestbench(owner, 'level')
    tb._module_added = True
    tb._create_netlist()
    tb.add_source(sources.VoltageSource('supply', positive='out', negative='0', dc=1.8))
    tb.add(elements.resistor('rload', p='out', n='0', value=1000))
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    tb.add_metric('supply', lambda c: float(c.load_signal('out').y[-1]))
    yield owner, tb
    owner.close()


@pytest.fixture(autouse=True)
def _no_model_library(monkeypatch, request):
    """Keep the stubbed runs independent of an installed PDK on disk."""
    if 'launch' not in request.fixturenames:
        return
    monkeypatch.setattr(load_pdk('sky130'), 'library_location', os.devnull)


# --- the setting is no longer silently ignored -----------------------------

def test_owner_temperature_reaches_a_registered_test(tree, launch):
    root, _block = tree
    root.set_temperature(85)
    results = root.run_all_tests(recursive=True)
    report = results['block.level']
    assert not isinstance(report, dict), report
    assert launch['temperatures'] == [85.0]
    assert report.operating_conditions.temperature_c == 85
    assert report.operating_conditions.temperature_kind == 'fixed'
    assert report.operating_conditions.temperature_basis == 'configured-and-observed'


def test_owner_temperature_is_re_read_at_every_run(tree, launch):
    root, _block = tree
    tb = live_test(root)
    root.set_temperature(85)
    root.run_all_tests(recursive=True)
    root.set_temperature(-40)
    root.run_all_tests(recursive=True)
    root.set_temperature(None)
    third = root.run_all_tests(recursive=True)['block.level']
    assert launch['temperatures'] == [85.0, -40.0, 27.0]
    assert 'temp=' not in launch['decks'][-1]
    # Cleared on the owner means cleared on the test: an inherited value is
    # never pinned by the first run that used it.
    assert 'temp' not in tb.netlist.simulator_options
    assert third.operating_conditions.temperature_basis == 'backend-observed'
    assert third.operating_conditions.temperature_c == 27


def test_an_inherited_value_is_not_the_tests_own_setting(tree, launch):
    root, _block = tree
    tb = live_test(root)
    root.set_temperature(85)
    reports = root.run_all_tests(recursive=True)
    assert reports['block.level'].sim_results.simulator_options['temp'] == 85
    assert tb.netlist._temperature_explicit is False
    assert not hasattr(tb, '_inherited_temperature_override')


# --- the per-test override wins, in both directions ------------------------

@pytest.mark.parametrize('owner,test_value', [(85, -40), (-40, 85), (None, 85)])
def test_a_test_setting_wins_over_its_owner(tree, launch, owner, test_value):
    root, _block = tree
    tb = live_test(root)
    if owner is not None:
        root.set_temperature(owner)
    tb.set_temperature(test_value)
    report = root.run_all_tests(recursive=True)['block.level']
    assert not isinstance(report, dict), report
    assert launch['temperatures'] == [float(test_value)]
    assert report.operating_conditions.temperature_c == test_value
    # A later owner change does not take the test's own setting away.
    root.set_temperature(0)
    root.run_all_tests(recursive=True)
    assert launch['temperatures'][-1] == float(test_value)


def test_a_test_can_opt_out_of_its_owners_temperature(tree, launch):
    root, _block = tree
    tb = live_test(root)
    root.set_temperature(85)
    tb.set_temperature(None)
    report = root.run_all_tests(recursive=True)['block.level']
    assert launch['temperatures'] == [27.0]
    assert 'temp=' not in launch['decks'][-1]
    assert report.operating_conditions.temperature_basis == 'backend-observed'


def test_native_options_are_the_same_deliberate_setting(tree, launch):
    root, _block = tree
    tb = live_test(root)
    root.set_temperature(85)
    tb.netlist.set_simulator_options(temp='-40')
    root.run_all_tests(recursive=True)
    assert launch['temperatures'] == [-40.0]
    assert tb.netlist._temperature_explicit is True


def test_a_child_test_run_outside_the_walk_sees_only_its_own_owner(tree, launch):
    """A propagated setting is in force for that invocation, not stored.

    Running the child's test on its own asks its own owner, which is a
    subcircuit holding no setting, so the run is unconfigured rather than
    quietly still hot from the last recursive run. Same rule the output mode
    already follows.
    """
    root, _block = tree
    instance = root.elem_dict['u1']['device']
    root.set_temperature(85)
    root.run_all_tests(recursive=True)
    assert launch['temperatures'] == [85.0]
    report = instance.run_test('level')
    assert launch['temperatures'] == [85.0, 27.0]
    assert report.operating_conditions.temperature_basis == 'backend-observed'


def test_single_test_run_on_a_top_level_owner_inherits(top_level, launch):
    """``run_test`` on the owner itself resolves that owner's setting."""
    owner, _tb = top_level
    owner.set_temperature(85)
    report = owner.run_test('level')
    assert launch['temperatures'] == [85.0]
    assert report.operating_conditions.temperature_c == 85


def test_the_deprecated_assertion_sees_the_inherited_setting(top_level, launch):
    """``run(temperature=)`` still only checks — against what will be applied."""
    owner, tb = top_level
    owner.set_temperature(85)
    with pytest.warns(DeprecationWarning):
        report = tb.run(temperature=85, voltage=1.8)
    assert report.characterization_context.temperature == 85
    with pytest.warns(DeprecationWarning), pytest.raises(CharacterizationContextError):
        tb.run(temperature=27)


def test_run_corners_inherits_for_every_corner(top_level, launch):
    owner, tb = top_level
    owner.set_temperature(85)
    results = tb.run_corners(['tt', 'ff'])
    assert set(results) == {'tt', 'ff'}
    assert launch['temperatures'] == [85.0, 85.0]
    assert all(r.operating_conditions.temperature_c == 85 for r in results.values())


# --- the recursive rule ----------------------------------------------------

def test_recursive_runs_apply_the_nearest_setting(tmp_path, monkeypatch):
    """Parent sets one, child test sets its own: the child test wins.

    The middle level is exercised through the legacy ``add_submodule``
    registration so that a Module with no ``module_name`` appears inside a
    walked tree; a block that owns tests is covered separately below.
    """
    pdk = load_pdk('sky130')
    root = Module('sky130', str(tmp_path), 'root.cir', pdk)
    mid = Module('sky130', str(tmp_path), 'mid.cir', pdk)
    leaf = _block(tmp_path, pdk, 'leaf')
    inherited = _test_on(leaf, 'inherited')
    overridden = _test_on(leaf, 'overridden')
    overridden.set_temperature(125)
    mid.add_submodule('leaf', leaf)
    root.add_submodule('mid', mid)
    root.set_temperature(85)

    seen = {}

    def record(self, **config):
        seen[self.test_name] = self._inherited_temperature()
        return {'recorded': True}

    monkeypatch.setattr(ModuleTestbench, 'run', record)
    root.run_all_tests(recursive=True)
    # Both tests are reached with the root's setting in force ...
    assert seen['inherited'] == (85.0, 'configured-and-observed')
    assert seen['overridden'] == (85.0, 'configured-and-observed')
    # ... and only the test without its own setting actually takes it.
    assert inherited.netlist._temperature_explicit is False
    assert overridden.netlist._temperature_explicit is True

    # A nearer owner replaces the root's setting for its own subtree.
    mid.set_temperature(-40)
    seen.clear()
    root.run_all_tests(recursive=True)
    assert seen['inherited'] == (-40.0, 'configured-and-observed')
    for module in (root, mid, leaf):
        module.close()


def test_a_recursive_run_changes_no_stored_setting(tree, launch):
    root, _block = tree
    tb = live_test(root)
    root.set_temperature(85)
    root.run_all_tests(recursive=True)
    assert not hasattr(tb, '_inherited_temperature_override')
    assert tb.netlist._temperature_explicit is False


# --- nothing else moves ----------------------------------------------------

def test_a_subcircuit_owning_no_tests_still_refuses(tmp_path):
    """The refusal stays where it is genuinely right (TASK-168).

    An instantiated ``.subckt`` carries no simulator options, so a setting
    there would be accepted and then ignored -- which is the condition the
    old ``module_name`` guard was a proxy for.
    """
    block = _block(tmp_path, load_pdk('sky130'), 'testless')
    try:
        with pytest.raises(ValueError, match='owns no registered tests'):
            block.set_temperature(85)
        assert block._temperature_explicit is False
        assert 'temp' not in block.simulator_options
    finally:
        block.close()


def test_a_refused_setting_does_not_mark_the_circuit(tree):
    root, _block = tree
    with pytest.raises(ValueError):
        root.set_temperature(-300)
    assert root._temperature_explicit is False
    root.set_temperature(85)
    with pytest.raises(TypeError):
        root.set_temperature('85')
    assert root.simulator_options['temp'] == 85


def test_an_inherited_temperature_is_still_confirmed(tree, launch):
    """Honest provenance is not weakened by inheritance (TASK-139)."""
    root, _block = tree
    root.set_temperature(85)
    launch['observed'] = 27.0
    results = root.run_all_tests(recursive=True)
    assert '85' in results['block.level']['error']
    assert live_test(root).netlist.operating_conditions is None


def test_a_changed_inherited_temperature_no_longer_needs_clear_cache(tmp_path, launch):
    """Formerly test_a_changed_inherited_temperature_cannot_reuse_cached_evidence:
    verified a cache=True testbench reused a prior report on a repeat
    run_all_tests, and that a changed temperature then raised
    SimulationConfigurationError until clear_cache() was called (TASK-163's
    cache-provenance guard). The report cache is disabled in 0.2.0
    (TASK-284): cache=True now only warns at construction and every run()
    simulates, so a repeat run produces a fresh report and a changed
    temperature is simply reflected -- no guard, no clear_cache() needed."""
    pdk = load_pdk('sky130')
    block = _block(tmp_path, pdk, 'cached')
    with pytest.warns(UserWarning, match='TASK-130'):
        tb = ModuleTestbench(block, 'level', cache=True)
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_source(sources.VoltageSource('supply', positive='out', negative='0', dc=1.8))
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    tb.add_metric('supply', lambda c: float(c.load_signal('out').y[-1]))
    root = Module('sky130', str(tmp_path), 'cachedroot.cir', pdk)
    root.add_module(block, 'u1', nets={'out': 'chip_out', 'vss': '0'})
    root.set_temperature(85)
    first = root.run_all_tests(recursive=True)['cached.level']
    assert first.operating_conditions.temperature_c == 85
    second = root.run_all_tests(recursive=True)['cached.level']
    assert second is not first   # no cache: a fresh SpecReport every run
    assert launch['temperatures'] == [85.0, 85.0]
    root.set_temperature(-40)
    third = root.run_all_tests(recursive=True)['cached.level']
    assert third.operating_conditions.temperature_c == -40
    block.close()
    root.close()


# --- real ngspice ----------------------------------------------------------

@needs_ngspice
@needs_pdk
@pytest.mark.slow
def test_real_ngspice_owner_setting_and_per_test_override(tmp_path):
    pdk = load_pdk('sky130')
    block = _block(tmp_path, pdk, 'block')
    # A real device pulls in the sky130 model library, so the inherited
    # setting is proven on the path an actual characterization run takes.
    block.add(pdk.nmos1p8('mload', d='out', g='out', s='vss', b='vss', w=1, l=0.15))
    _test_on(block, 'inherits')
    _test_on(block, 'overrides').set_temperature(-40)
    root = Module('sky130', str(tmp_path), 'root.cir', pdk)
    root.add_module(block, 'u1', nets={'out': 'chip_out', 'vss': '0'})
    root.set_temperature(85)

    results = root.run_all_tests(recursive=True)
    for name, expected in (('inherits', 85), ('overrides', -40)):
        report = results[f'block.{name}']
        assert not isinstance(report, dict), report
        assert report.metrics['supply'] == pytest.approx(1.8)
        assert report.operating_conditions.temperature_c == expected
        assert report.operating_conditions.temperature_basis == 'configured-and-observed'
        netlist = report.sim_results
        log = (Path(netlist.results_file).parent / 'output.log').read_text()
        assert f'TEMP = {expected:.6f}' in log and 'TNOM = 27.000000' in log
    block.close()
    root.close()


@needs_ngspice
@needs_pdk
def test_real_ngspice_direct_simulate_on_a_test_owning_circuit(tmp_path):
    """The setter stays usable where it always applied: the circuit's own run."""
    pdk = load_pdk('sky130')
    owner = Module('sky130', str(tmp_path), 'owner.cir', pdk)
    owner.add(elements.resistor('rload', p='out', n='0', value=1000))
    owner.add(sources.VoltageSource('supply', positive='out', negative='0', dc=1.8))
    owner.add_sim_type(Transient(step=1e-6, stop=2e-6))
    tb = owner.create_test('registered')     # the circuit owns a test ...
    assert owner.list_tests() == ['registered']
    owner.set_temperature(85)                # ... and still configures itself
    owner.simulate('tt')
    assert owner.operating_conditions.temperature_c == 85
    assert float(owner.load_signal('out').y[-1]) == pytest.approx(1.8)
    log = (Path(owner.results_file).parent / 'output.log').read_text()
    assert f'TEMP = {85:.6f}' in log and 'TNOM = 27.000000' in log
    assert tb is owner._tests['registered']
    owner.close()
