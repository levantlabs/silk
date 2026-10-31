"""TASK-224: ``tb.set_simulator_options`` and owner-to-test option propagation.

Two defects, one fix. (1) ``tb.set_simulator_options(...)`` raised
``AttributeError`` on a ``ModuleTestbench`` -- the docstring's own spelling
did not work. (2) Calling ``set_simulator_options`` on a Module that owns
tests was accepted, warned nothing, and every key except ``temp`` was
silently dropped from the emitted deck. The owner ruling (2026-09-23): (1)
``tb.set_simulator_options`` becomes a thin delegate to
``tb.netlist.set_simulator_options``; (2) owner options reach every test it
owns as defaults, resolved live at every run exactly as ``set_temperature``
inherits (TASK-162/166/168), and a key the test sets itself always wins.
"""
import subprocess
import warnings
from pathlib import Path

import pytest

from tests._mock_rawfile import write_complete_rawfile

from silk.circuit.modules import Module
from silk.circuit.module_test import ModuleTestbench
from silk.elements import elements
from silk.simulation import Transient
from silk import sources

FIXTURE = Path(__file__).parent / 'fixtures/rawfiles/e_ng_tran.raw'


# ---------------------------------------------------------------------------
# AC1 -- tb.set_simulator_options is a thin delegate to tb.netlist's
# ---------------------------------------------------------------------------

class TestDelegate:

    def test_delegates_preset_expansion(self, pdk, tmp_dir):
        owner = Module('PDK_skywater130', tmp_dir, 'owner.cir', pdk_details=pdk)
        tb = ModuleTestbench(owner, 'level')
        tb._create_netlist()

        reference = Module('PDK_skywater130', tmp_dir, 'ref.cir', pdk_details=pdk)
        reference.set_simulator_options(preset='tight')

        tb.set_simulator_options(preset='tight')
        assert tb.netlist.simulator_options == reference.simulator_options

    def test_delegates_preset_plus_override(self, pdk, tmp_dir):
        owner = Module('PDK_skywater130', tmp_dir, 'owner.cir', pdk_details=pdk)
        tb = ModuleTestbench(owner, 'level')
        tb._create_netlist()
        tb.set_simulator_options(preset='tight', method='trap')
        assert tb.netlist.simulator_options['method'] == 'trap'
        assert tb.netlist.simulator_options['trtol'] == 1

    def test_delegates_accumulation(self, pdk, tmp_dir):
        owner = Module('PDK_skywater130', tmp_dir, 'owner.cir', pdk_details=pdk)
        tb = ModuleTestbench(owner, 'level')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tb.set_simulator_options(reltol=1e-7)
            tb.set_simulator_options(abstol=1e-13)
        assert tb.netlist.simulator_options == {'reltol': 1e-7, 'abstol': 1e-13}

    def test_delegates_temp_handling(self, pdk, tmp_dir):
        owner = Module('PDK_skywater130', tmp_dir, 'owner.cir', pdk_details=pdk)
        tb = ModuleTestbench(owner, 'level')
        tb.set_simulator_options(temp='45')
        assert tb.netlist.simulator_options['temp'] == '45'
        assert tb.netlist._temperature_explicit is True
        assert tb.netlist._temperature_basis == 'native-and-observed'

    def test_delegates_unknown_key_warning(self, pdk, tmp_dir):
        owner = Module('PDK_skywater130', tmp_dir, 'owner.cir', pdk_details=pdk)
        tb = ModuleTestbench(owner, 'level')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            tb.set_simulator_options(totally_made_up_option=42)
        assert any('known_options' in str(x.message) for x in w)

    def test_creates_netlist_lazily_like_siblings(self, pdk, tmp_dir):
        """Mirrors add_custom_line/set_temperature: no netlist required yet."""
        owner = Module('PDK_skywater130', tmp_dir, 'owner.cir', pdk_details=pdk)
        tb = ModuleTestbench(owner, 'level')
        assert tb.netlist is None
        tb.set_simulator_options(reltol=1e-6)
        assert tb.netlist is not None

    def test_returns_self_for_chaining(self, pdk, tmp_dir):
        owner = Module('PDK_skywater130', tmp_dir, 'owner.cir', pdk_details=pdk)
        tb = ModuleTestbench(owner, 'level')
        assert tb.set_simulator_options(reltol=1e-6) is tb


# ---------------------------------------------------------------------------
# AC4 -- the docstring's own spellings work, on both receivers it shows
# ---------------------------------------------------------------------------

class TestDocstringSpellings:

    def test_module_receiver_examples(self, pdk, tmp_dir):
        tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        tb.simulator = 'ngspice'
        tb.set_simulator_options(preset='tight')
        tb.set_simulator_options(preset='tight', method='trap')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tb.set_simulator_options(reltol=1e-6)
        assert tb.simulator_options['method'] == 'trap'
        assert tb.simulator_options['reltol'] == 1e-6

    def test_testbench_receiver_examples(self, pdk, tmp_dir):
        owner = Module('PDK_skywater130', tmp_dir, 'owner.cir', pdk_details=pdk)
        tb = ModuleTestbench(owner, 'level')
        tb._create_netlist()
        tb.netlist.simulator = 'ngspice'
        tb.set_simulator_options(preset='tight')
        assert tb.netlist.simulator_options['method'] == 'gear'


# ---------------------------------------------------------------------------
# AC2/AC3 -- owner options reach every test it owns, key by key, live
# ---------------------------------------------------------------------------

@pytest.fixture
def launch(monkeypatch, mock_ngspice_version):
    """Stub ngspice: records every emitted deck, changes nothing about it."""
    state = {'decks': []}

    def run(argv, **kwargs):
        deck = Path(argv[-1]).read_text()
        raw = Path(argv[argv.index('-r') + 1])
        log = Path(argv[argv.index('-o') + 1])
        state['decks'].append(deck)
        log.write_text('Doing analysis at TEMP = 27.000000 and TNOM = 27.000000\n')
        write_complete_rawfile(raw, Path(argv[-1]).read_text(), template=FIXTURE)
        return subprocess.CompletedProcess(argv, 0, stdout='', stderr='')

    monkeypatch.setattr(subprocess, 'run', run)
    return state


@pytest.fixture
def top_level(tmp_path, pdk):
    """A top-level circuit that owns a test directly (no .subckt DUT)."""
    owner = Module('PDK_skywater130', str(tmp_path), 'owner.cir', pdk_details=pdk)
    tb = ModuleTestbench(owner, 'level')
    tb._module_added = True
    tb._create_netlist()
    tb.add_source(sources.VoltageSource('supply', positive='out', negative='0', dc=1.8))
    tb.add(elements.resistor('rload', p='out', n='0', value=1000))
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    tb.add_metric('supply', lambda c: float(c.load_signal('out').y[-1]))
    yield owner, tb
    owner.close()


def test_owner_options_reach_an_owned_test_beyond_temp(top_level, launch):
    owner, tb = top_level
    owner.set_simulator_options(preset='tight')
    tb.run(corner='tt')
    deck = launch['decks'][-1]
    assert 'reltol=' in deck
    assert 'method=gear' in deck
    assert tb.netlist.simulator_options['reltol'] == owner.simulator_options['reltol']


def test_bench_setting_overrides_the_inherited_key(top_level, launch):
    owner, tb = top_level
    owner.set_simulator_options(preset='tight')  # method=gear, reltol=1e-5, ...
    tb.set_simulator_options(method='trap')
    tb.run(corner='tt')
    deck = launch['decks'][-1]
    assert 'method=trap' in deck
    # Untouched keys still inherit from the owner.
    assert 'reltol=' in deck
    assert tb.netlist.simulator_options['method'] == 'trap'
    assert tb.netlist.simulator_options['trtol'] == owner.simulator_options['trtol']


def test_inherited_options_are_not_the_tests_own_setting(top_level, launch):
    owner, tb = top_level
    owner.set_simulator_options(preset='tight')
    tb.run(corner='tt')
    assert 'reltol' not in tb.netlist._explicit_simulator_options
    assert tb.netlist.simulator_options['reltol'] == owner.simulator_options['reltol']


def test_owner_change_is_followed_at_the_next_run(top_level, launch):
    owner, tb = top_level
    owner.set_simulator_options(reltol=1e-5)
    tb.run(corner='tt')
    assert tb.netlist.simulator_options['reltol'] == 1e-5
    owner.set_simulator_options(reltol=1e-9)
    tb.run(corner='tt')
    assert tb.netlist.simulator_options['reltol'] == 1e-9


def test_no_owner_options_means_no_inherited_line(top_level, launch):
    owner, tb = top_level
    tb.run(corner='tt')
    assert '.options' not in launch['decks'][-1]


def test_bench_only_setting_emits_exactly_one_options_line(top_level, launch):
    """Deck-byte-identity guard (AC4): a bench with an owner that set no
    options of its own emits exactly what it emitted before TASK-224."""
    owner, tb = top_level
    tb.set_simulator_options(reltol=1e-4)
    tb.run(corner='tt')
    deck = launch['decks'][-1]
    assert tb.netlist.simulator_options == {'reltol': 1e-4}
    assert deck.count('.options') == 1


# ---------------------------------------------------------------------------
# Direct unit coverage of the merge helper (silk.simulation.conditions)
# ---------------------------------------------------------------------------

class TestApplyInheritedSimulatorOptions:

    def test_own_setting_none_when_never_called(self, pdk, tmp_dir):
        from silk.simulation.conditions import own_simulator_options_setting
        m = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        assert own_simulator_options_setting(m) is None

    def test_own_setting_excludes_temp(self, pdk, tmp_dir):
        from silk.simulation.conditions import own_simulator_options_setting
        m = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        m.set_simulator_options(temp=45)
        assert own_simulator_options_setting(m) is None

    def test_a_key_no_longer_supplied_is_un_inherited(self, pdk, tmp_dir):
        """Owner previously offered {'a': 1, 'b': 2}; now only offers {'a': 1}.
        The bench must stop carrying 'b' as a stale default."""
        from silk.simulation.conditions import apply_inherited_simulator_options
        netlist = Module('PDK_skywater130', tmp_dir, 'n.cir', pdk_details=pdk)
        apply_inherited_simulator_options(netlist, {'a': 1, 'b': 2})
        assert netlist.simulator_options == {'a': 1, 'b': 2}
        apply_inherited_simulator_options(netlist, {'a': 1})
        assert netlist.simulator_options == {'a': 1}

    def test_own_explicit_key_is_never_overwritten_or_removed(self, pdk, tmp_dir):
        from silk.simulation.conditions import apply_inherited_simulator_options
        netlist = Module('PDK_skywater130', tmp_dir, 'n.cir', pdk_details=pdk)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            netlist.set_simulator_options(a=99)
        apply_inherited_simulator_options(netlist, {'a': 1, 'b': 2})
        assert netlist.simulator_options['a'] == 99
        assert netlist.simulator_options['b'] == 2
        # Owner later drops 'b': the explicit 'a' is untouched either way.
        apply_inherited_simulator_options(netlist, {})
        assert netlist.simulator_options == {'a': 99}
