"""TASK-163 — ``run_all_tests`` separates a wrong REQUEST from a bad RESULT.

The swallow in ``_run_test_entries`` is deliberate and stays: one circuit that
will not converge must not abandon the rest of a sweep. What changes is that an
exception marked :class:`~silk.exceptions.ConfigurationError` propagates instead
of becoming one ``{'error': ...}`` entry per test, because it says the caller's
setup is wrong and every test in the tree is about to fail for the same reason.

Classification is by the marker base class only. These tests pin that: the
marker, the two behaviours it separates, and the fact that retyping the affected
raise sites did not move them out of the bases existing callers already catch.
"""
import os
import subprocess
from pathlib import Path

import pytest

from tests._mock_rawfile import write_complete_rawfile

from silk import Module, load_pdk, sources
from silk.elements import elements
from silk.exceptions import (
    CharacterizationContextError, CircuitError, ConfigurationError,
    DatabaseError, NotSupportedError, OutputPathError, RunArgumentError,
    SimulationAborted, SimulationConfigurationError, SimulationError,
)
from silk.simulation import Transient


# ---------------------------------------------------------------------------
# A two-level live tree: parent 'chain' owns a test and a registered submodule
# 'stage' that owns its own test. run_all_tests(recursive=True) visits both.
# ---------------------------------------------------------------------------

def _dut(tmp_path, name, filename, *, child=None):
    mod = Module('sky130', str(tmp_path), filename, load_pdk('sky130'),
                 module_name=name)
    mod.define_module_ports({'out': 'out', 'vss': 'ground'})
    mod.add(elements.resistor('rload', p='out', n='vss', value=1000))
    if child is not None:
        mod.add_submodule('stage', child)
    tb = mod.create_test('level')
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_source(sources.VoltageSource('supply', positive='out', negative='0',
                                        dc=1.8))
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    tb.add_metric('supply', lambda c: float(c.load_signal('out').y[-1]))
    return mod, tb


@pytest.fixture
def tree(tmp_path):
    """parent.run_all_tests() runs exactly two tests: chain.level, stage.level."""
    child, child_tb = _dut(tmp_path / 'c', 'stage', 'stage.cir')
    parent, parent_tb = _dut(tmp_path / 'p', 'chain', 'chain.cir', child=child)
    yield parent, parent_tb, child, child_tb
    parent.close()
    child.close()


@pytest.fixture
def launch(monkeypatch, tree, mock_ngspice_version):
    """Every ngspice launch succeeds, reporting TEMP=27, unless told otherwise.

    ``state['temperatures']`` sets what the log claims; ``state['broken']`` holds
    module names whose launch produces no rawfile at all, which is how a per-test
    simulation failure (SimulationAborted) is provoked for exactly one circuit in
    the tree. ``state['decks']`` records the title of every deck actually launched.
    """
    parent, _parent_tb, child, _child_tb = tree
    for mod in (parent, child):
        monkeypatch.setattr(mod.pdk_details, 'library_location', os.devnull)
    state = {'temperatures': (27.0,), 'broken': set(), 'decks': []}
    fixture = Path(__file__).parent / 'fixtures/rawfiles/e_ng_tran.raw'

    def run(argv, **kwargs):
        raw = Path(argv[argv.index('-r') + 1])
        log = Path(argv[argv.index('-o') + 1])
        deck_text = Path(argv[-1]).read_text()
        # Generation staging changes the filename but preserves the circuit title.
        deck = deck_text.splitlines()[0]
        state['decks'].append(deck)
        log.write_text(''.join(
            f'Doing analysis at TEMP = {t} and TNOM = 27.000000\n'
            for t in state['temperatures']))
        if any(name in deck for name in state['broken']):
            return subprocess.CompletedProcess(argv, 1, stdout='',
                                               stderr='doAnalyses: TRAN:  Timestep too small')
        write_complete_rawfile(raw, deck_text, template=fixture)
        return subprocess.CompletedProcess(argv, 0, stdout='', stderr='')

    monkeypatch.setattr(subprocess, 'run', run)
    return state


# ---------------------------------------------------------------------------
# The marker itself
# ---------------------------------------------------------------------------

def test_marker_is_opt_in_by_inheritance_and_keeps_existing_bases():
    """Every marked class still is what it was; unmarked neighbours stay out."""
    assert issubclass(ConfigurationError, CircuitError)

    # Marked, and each keeps the base its existing callers catch.
    assert issubclass(CharacterizationContextError, ConfigurationError)
    assert issubclass(CharacterizationContextError, DatabaseError)
    assert issubclass(SimulationConfigurationError, ConfigurationError)
    assert issubclass(SimulationConfigurationError, SimulationError)
    assert issubclass(NotSupportedError, ConfigurationError)
    assert issubclass(NotSupportedError, CircuitError)

    # TASK-166: the two caller-setup failures that used to arrive as builtins.
    # Each keeps the builtin base its historical callers already catch, and
    # opts into the marker at its own definition like every other member.
    assert issubclass(RunArgumentError, ConfigurationError)
    assert issubclass(RunArgumentError, TypeError)
    assert issubclass(OutputPathError, ConfigurationError)
    assert issubclass(OutputPathError, ValueError)

    # The builtins themselves stay out: nothing widened to "every ValueError".
    assert not issubclass(ValueError, ConfigurationError)
    assert not issubclass(TypeError, ConfigurationError)

    # Deliberately NOT marked: these describe a run, not a request.
    assert not issubclass(SimulationAborted, ConfigurationError)
    assert not issubclass(SimulationError, ConfigurationError)
    assert not issubclass(DatabaseError, ConfigurationError)


def test_a_new_exception_joins_by_saying_so_not_by_message_text(tree, launch):
    """The runner asks isinstance and nothing else."""
    parent, parent_tb, _child, _child_tb = tree

    class CallerMistake(ConfigurationError):
        pass

    class LooksLikeOne(SimulationError):
        """Same words, no marker — still a per-test result."""

    parent_tb.run = lambda **kw: (_ for _ in ()).throw(
        LooksLikeOne('configuration error: the caller setup is wrong'))
    results = parent.run_all_tests()
    assert 'configuration error' in results['chain.level']['error']

    parent_tb.run = lambda **kw: (_ for _ in ()).throw(CallerMistake('opted in'))
    with pytest.raises(CallerMistake, match='opted in'):
        parent.run_all_tests()


# ---------------------------------------------------------------------------
# (1) Every test in a recursive tree fails for the same configuration reason
# ---------------------------------------------------------------------------

def test_whole_tree_failing_on_one_temperature_assertion_propagates(tree, launch):
    """The migration case: run_all_tests(temperature=85) with no set_temperature.

    Before TASK-163 this returned a dictionary whose every entry was an error
    string. Now the first test stops the run, and nothing is returned at all.
    """
    parent, _parent_tb, _child, _child_tb = tree
    with pytest.deprecated_call():
        with pytest.raises(CharacterizationContextError,
                           match='requires matching explicit fixed TEMP'):
            parent.run_all_tests(temperature=85, recursive=True)
    # It stopped at the first test rather than running the rest of the tree.
    assert launch['decks'] == []


def test_configuration_error_stops_the_tree_before_later_tests_run(tree, launch):
    """Stopping is the point: the sibling test is never launched."""
    parent, parent_tb, child, child_tb = tree
    # Only the parent's test is mis-configured; the child's is fine.
    child_tb.set_temperature(27)
    with pytest.deprecated_call():
        with pytest.raises(CharacterizationContextError):
            parent.run_all_tests(temperature=27, recursive=True)
    assert launch['decks'] == []
    # ...and the child still runs perfectly well on its own afterwards.
    child_tb.set_temperature(None)
    assert child.run_all_tests(recursive=False)['stage.level'].metrics['supply']


def test_a_correctly_migrated_call_still_returns_reports(tree, launch):
    """The fix the error names actually works, through the same entry point."""
    parent, parent_tb, _child, child_tb = tree
    parent_tb.set_temperature(85)
    child_tb.set_temperature(85)
    launch['temperatures'] = (85.0,)
    results = parent.run_all_tests(recursive=True)
    assert set(results) == {'chain.level', 'stage.level'}
    for report in results.values():
        assert report.operating_conditions.temperature_c == 85


# ---------------------------------------------------------------------------
# (2) One circuit misbehaves — the sweep must still finish
# ---------------------------------------------------------------------------

def test_one_test_failing_to_converge_does_not_abort_the_sweep(tree, launch):
    """The long-standing contract, pinned so classification cannot erode it."""
    parent, _parent_tb, _child, _child_tb = tree
    launch['broken'].add('chain')

    results = parent.run_all_tests(recursive=True)

    assert set(results) == {'chain.level', 'stage.level'}
    assert set(results['chain.level']) == {'error'}
    assert any('stage' in d for d in launch['decks'])   # sibling still launched
    good = results['stage.level']
    assert not isinstance(good, dict)
    assert good.metrics['supply']              # a real report, not an error


@pytest.mark.parametrize('error', [
    SimulationAborted('no data rows', simulator='ngspice', data_rows=0),
    SimulationError('ngspice exit 1'),
    SimulationError('Cannot launch ngspice'),
    OSError('rawfile is malformed'),
    RuntimeError('boom'),
    # TASK-166 moved two builtin *caller-setup* failures off the builtin path,
    # and deliberately did not widen the propagating set to builtins in
    # general: a plain ValueError/TypeError out of a run is still a result.
    ValueError('the extractor got a value it could not use'),
    TypeError('the extractor got a type it could not use'),
])
def test_run_failures_stay_per_test_results(tree, launch, error):
    """Aborts, launch failures, nonzero exits and bad rawfiles are results."""
    parent, parent_tb, _child, _child_tb = tree

    def raise_it(**kw):
        raise error

    parent_tb.run = raise_it
    results = parent.run_all_tests(recursive=True)
    assert set(results) == {'chain.level', 'stage.level'}
    assert set(results['chain.level']) == {'error'}
    assert results['stage.level'].metrics['supply']


# ---------------------------------------------------------------------------
# (3) The retyped raise sites: deck preparation and staleness
# ---------------------------------------------------------------------------

def test_stale_deck_refusal_propagates_out_of_run_all_tests(tree, launch):
    """TASK-160's guard is a configuration error: the fix is in the script."""
    parent, parent_tb, _child, _child_tb = tree

    original_run = parent_tb.run

    def run_then_break(**kw):
        # Prepare a deck, change the circuit, then launch the stale one.
        parent_tb.netlist.generate_netlist()
        parent_tb.netlist.add(
            elements.resistor('rextra', p='out', n='0', value=50))
        return parent_tb.netlist.run_simulator()

    parent_tb.run = run_then_break
    with pytest.raises(SimulationConfigurationError,
                       match='Circuit configuration changed'):
        parent.run_all_tests(recursive=True)
    parent_tb.run = original_run
    assert not any('stage' in d for d in launch['decks'])


def test_managed_temp_without_a_prepared_deck_propagates(tree, launch):
    parent, parent_tb, _child, _child_tb = tree
    parent_tb.set_temperature(85)
    parent_tb.netlist.set_output_options(mode='overwrite')

    def run_without_preparing(**kw):
        return parent_tb.netlist.run_simulator()

    parent_tb.run = run_without_preparing
    with pytest.raises(SimulationConfigurationError,
                       match='Managed TEMP requires a prepared deck'):
        parent.run_all_tests(recursive=True)


def test_unsupported_capability_propagates(tree, launch):
    """NotSupportedError: the backend never had it, so no test in the tree can."""
    parent, parent_tb, _child, _child_tb = tree
    parent_tb.netlist.simulator = 'xyce'
    parent_tb.netlist._simulator_explicit = True
    parent_tb.set_temperature(85)

    def ask_for_device_currents(**kw):
        parent_tb.netlist.backend.device_current_vectors(None, '', '')

    parent_tb.run = ask_for_device_currents
    with pytest.raises(NotSupportedError):
        parent.run_all_tests(recursive=True)


# ---------------------------------------------------------------------------
# (4) The retyping did not break the handlers that already exist
# ---------------------------------------------------------------------------

def test_retyped_errors_are_still_caught_by_their_old_handlers(tree, launch):
    """``except SimulationError`` / ``except DatabaseError`` are unaffected."""
    parent, parent_tb, _child, _child_tb = tree
    parent_tb.set_temperature(85)
    try:
        parent_tb.netlist.run_simulator()
    except SimulationError as error:                 # the pre-TASK-163 handler
        assert isinstance(error, SimulationConfigurationError)
    else:                                            # pragma: no cover
        pytest.fail('expected the managed-TEMP refusal')

    with pytest.deprecated_call():
        try:
            parent_tb.run(temperature=35)
        except DatabaseError as error:               # the pre-TASK-163 handler
            assert isinstance(error, CharacterizationContextError)
        else:                                        # pragma: no cover
            pytest.fail('expected the temperature assertion mismatch')


def test_output_mode_override_is_restored_when_a_run_stops(tree, launch):
    """The propagation path still runs the finally that restores the override."""
    parent, parent_tb, _child, _child_tb = tree
    before = dict(parent_tb.__dict__)
    with pytest.deprecated_call():
        with pytest.raises(CharacterizationContextError):
            parent.run_all_tests(temperature=85, recursive=True)
    assert parent_tb.__dict__ == before


# ---------------------------------------------------------------------------
# (5) TASK-166 — the two caller-setup failures that used to arrive as builtins
#
# A builtin cannot inherit the marker, so each is moved off the builtin path at
# its own raise site: the arguments are bound to the callee's signature before
# the call, and an unusable output label is OutputPathError where it is found.
# ---------------------------------------------------------------------------

def test_unexpected_keyword_stops_the_run_instead_of_filling_the_results(tree, launch):
    """``run_all_tests(store=True)``: Python's TypeError, now a raised refusal.

    Before TASK-166 this returned a dictionary whose every entry was the same
    "unexpected keyword argument" string — the exact shape TASK-163 fixed for
    silk's own exceptions, just reached by a builtin instead.
    """
    parent, _parent_tb, _child, _child_tb = tree
    with pytest.raises(RunArgumentError, match='store'):
        parent.run_all_tests(store=True, recursive=True)
    assert launch['decks'] == []       # refused before anything was launched


def test_a_mis_spelled_keyword_names_the_test_and_the_keyword(tree, launch):
    parent, _parent_tb, _child, _child_tb = tree
    with pytest.raises(RunArgumentError) as caught:
        parent.run_all_tests(run_labl='baseline', recursive=True)
    assert 'chain.level' in str(caught.value) and 'run_labl' in str(caught.value)
    # Still a TypeError for anyone who wrote the obvious handler, and the
    # builtin Python would have raised is chained rather than discarded.
    assert isinstance(caught.value, TypeError)
    assert type(caught.value.__cause__) is TypeError
    assert launch['decks'] == []


def test_a_bad_run_label_stops_the_run_and_writes_nothing(tree, launch, tmp_path):
    """The other half: a label that cannot name a directory (TASK-166)."""
    parent, _parent_tb, _child, _child_tb = tree
    before = sorted(p for p in tmp_path.rglob('*'))
    with pytest.raises(OutputPathError, match='run_label'):
        parent.run_all_tests(run_label='../escape', recursive=True)
    assert launch['decks'] == []
    assert sorted(p for p in tmp_path.rglob('*')) == before


def test_the_refusals_are_still_the_builtins_their_callers_catch(tree, launch):
    """Retyping must not move these out of ``except TypeError``/``ValueError``."""
    parent, _parent_tb, _child, _child_tb = tree
    with pytest.raises(TypeError):
        parent.run_all_tests(store=True, recursive=True)
    with pytest.raises(ValueError):
        parent.run_all_tests(run_label='../escape', recursive=True)


def test_a_refusal_leaves_every_test_exactly_as_it_was_found(tree, launch):
    """The argument check runs before the per-invocation overrides are installed."""
    parent, parent_tb, _child, child_tb = tree
    before = [dict(tb.__dict__) for tb in (parent_tb, child_tb)]
    with pytest.raises(RunArgumentError):
        parent.run_all_tests(store=True, recursive=True)
    assert [tb.__dict__ for tb in (parent_tb, child_tb)] == before


def test_a_valid_label_and_corner_still_run_the_whole_tree(tree, launch):
    """The check refuses only what the callee genuinely cannot accept."""
    parent, _parent_tb, _child, _child_tb = tree
    results = parent.run_all_tests(run_label='baseline', corner='tt',
                                   recursive=True)
    assert set(results) == {'chain.level', 'stage.level'}
    for report in results.values():
        assert report.metrics['supply']


def test_a_test_whose_run_takes_anything_is_not_second_guessed(tree, launch):
    """Signature introspection asks the callee, so a ``**kwargs`` run is fine."""
    parent, parent_tb, _child, _child_tb = tree
    seen = {}

    def permissive_run(**kw):
        seen.update(kw)
        return {'ok': True}

    parent_tb.run = permissive_run
    results = parent.run_all_tests(anything_at_all=1, recursive=False)
    assert seen == {'anything_at_all': 1}
    assert results['chain.level'] == {'ok': True}


def test_one_circuit_that_will_not_converge_still_does_not_abort_a_sweep(tree, launch):
    """AC#3 — the swallow is why this change had to happen outside it.

    Same contract as the TASK-163 regression above, re-proven through the
    argument-checked path and with a label in play, so neither the pre-call
    binding nor the retyped ``_output_paths`` can quietly turn a bad *result*
    into a stopped run.
    """
    parent, _parent_tb, _child, _child_tb = tree
    launch['broken'].add('chain')

    results = parent.run_all_tests(run_label='sweep', corner='tt',
                                   recursive=True)

    assert set(results) == {'chain.level', 'stage.level'}
    assert set(results['chain.level']) == {'error'}      # the bad circuit
    assert any('stage' in deck for deck in launch['decks'])
    assert results['stage.level'].metrics['supply']      # the sibling finished
