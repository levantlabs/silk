"""silk.analysis(): the analysis the behavioural engine is dispatching (design 3.5)."""
import subprocess
import sys

import pytest

from silk.exceptions import SimulationError
from silk.simulation import DC, Transient
from silk.sources.classes import VoltageSource
from silk.simulation.variation import Normal
from tests._variation_bench import chip
from tests.test_behavioral_rawresult import _block


def _kind():
    """A helper the model calls synchronously."""
    from silk import analysis
    return analysis()


def _recording_bench(tmp_path, analyses, observed, *, fail_at=None):
    # Observe dispatch through a module global, outside copied closure state.
    global seen
    seen = observed
    def model(ports, t, params):
        seen.append(_kind())
        if fail_at is not None and len(seen) == fail_at:
            raise RuntimeError('model failed')
        ports.vout.v = 2.0 * ports.vin.v

    dut = _block(tmp_path, model)
    tb = dut.create_test('both')
    tb.add_module_under_test({'vin': 'vin', 'vout': 'vout'})
    tb.add_source(VoltageSource('drive', positive='vin', negative='gnd', dc=0.5))
    for analysis in analyses:
        tb.add_sim_type(analysis)
    tb.netlist.simulator = 'behavioral'
    return tb


class _OutputValue:
    """A published scalar asks for analysis during voltage conversion."""
    def __init__(self, log):
        self.log = log

    def __float__(self):
        self.log.append(('publish', _kind_or_raised()))
        return 1.0


def _kind_or_raised():
    try:
        return _kind()
    except SimulationError:
        return 'raised'


def _bench_with(tmp_path, model, analysis):
    dut = _block(tmp_path, model)
    tb = dut.create_test('one')
    tb.add_module_under_test({'vin': 'vin', 'vout': 'vout'})
    tb.add_source(VoltageSource('drive', positive='vin', negative='gnd', dc=0.5))
    tb.add_sim_type(analysis)
    tb.netlist.simulator = 'behavioral'
    return tb


def _raises_outside():
    from silk import analysis
    with pytest.raises(SimulationError, match='behavioural model'):
        analysis()


def test_exported_at_the_root():
    import silk
    from silk import analysis
    assert silk.analysis is analysis
    assert 'analysis' in silk.__all__


def test_raises_before_any_run():
    _raises_outside()


@pytest.mark.parametrize('first', ['dc', 'transient'])
def test_value_follows_each_dispatched_analysis_in_one_launch(tmp_path, first):
    dc = DC('drive', 0, 1, 0.5)                        # three points
    tran = Transient(step=1e-9, stop=2e-9)
    seen = []
    tb = _recording_bench(tmp_path, [dc, tran] if first == 'dc' else [tran, dc], seen)
    tb.netlist.simulate()
    expected = (['dc'] * 3 + ['transient']) if first == 'dc' else (['transient'] + ['dc'] * 3)
    assert seen == expected
    _raises_outside()


def test_reset_after_a_model_failure(tmp_path):
    seen = []
    tb = _recording_bench(tmp_path, [DC('drive', 0, 1, 0.5)], seen, fail_at=2)
    with pytest.raises(Exception, match='model failed'):
        tb.netlist.simulate()
    assert seen == ['dc', 'dc']
    _raises_outside()


def test_scope_resets_on_return_failure_and_cancellation():
    from silk.simulation._analysis_context import analysis_scope
    from silk import analysis

    with analysis_scope('dc'):
        assert analysis() == 'dc'
    _raises_outside()

    with pytest.raises(RuntimeError):
        with analysis_scope('transient'):
            raise RuntimeError('boom')
    _raises_outside()

    class Cancelled(BaseException):
        pass

    with pytest.raises(Cancelled):
        with analysis_scope('transient'):
            raise Cancelled()
    _raises_outside()


def test_inner_scope_restores_its_callers_value():
    from silk.simulation._analysis_context import analysis_scope
    from silk import analysis
    with analysis_scope('transient'):
        with analysis_scope('dc'):
            assert analysis() == 'dc'
        assert analysis() == 'transient'
    _raises_outside()


def test_scope_refuses_an_unknown_kind():
    from silk.simulation._analysis_context import analysis_scope
    with pytest.raises(ValueError, match="'transient' or 'dc'"):
        with analysis_scope('ac'):
            pass


def test_monte_carlo_samples_and_a_replay_set_it_again(pdk, tmp_path, monkeypatch):
    seen = []
    original_kind = _kind

    def observe_kind():
        kind = original_kind()
        seen.append(kind)
        return kind

    # Global helper calls retain observation across cloned model placements.
    monkeypatch.setattr(sys.modules[__name__], '_kind', observe_kind)

    def model(ports, t, params):
        _kind()
        ports.y.v = ports.a.v + params['offset']

    top, tb = chip(pdk, tmp_path, {'offset': Normal(nominal=0.0, sigma=1e-3)})
    top.elem_dict['c0']['device'].set_behavioral_model(
        model, params={'offset': Normal(nominal=0.0, sigma=1e-3)}, execution='vector')
    mc = tb.set_monte_carlo(n=2, seed=1).run()
    assert seen == ['transient'] * 6  # two samples, three placements
    _raises_outside()
    mc.rerun(list(mc.keys())[0])
    assert seen == ['transient'] * 9  # replay invokes all three placements
    _raises_outside()


def test_a_run_leaves_behavioral_identity_unchanged(tmp_path):
    from silk.circuit._behavioral_identity import behavioral_key
    seen = []
    tb = _recording_bench(tmp_path, [DC('drive', 0, 1, 0.5)], seen)
    before = behavioral_key(tb.netlist)
    tb.netlist.simulate()
    assert behavioral_key(tb.netlist) == before


@pytest.mark.parametrize('order', [
    'import silk.simulation.analysis\nimport silk',
    'import silk\nimport silk.simulation.analysis',
])
def test_import_order_and_no_submodule(order):
    script = order + '''
import importlib.util
import types
from silk import analysis
import silk
assert silk.analysis is analysis and callable(analysis)
assert importlib.util.find_spec('silk.analysis') is None
assert isinstance(silk.simulation.analysis, types.ModuleType)
assert silk.simulation.analysis.Transient is silk.Transient
from silk.exceptions import SimulationError
for call in (silk.analysis, analysis):
    try:
        call()
    except SimulationError:
        pass
    else:
        raise AssertionError('analysis() returned outside a model')
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_importing_it_does_not_import_the_engine():
    script = '''
import sys
import silk.simulation._analysis_context
assert 'silk.simulation.behavioral' not in sys.modules, 'engine imported'
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('kind,analysis,calls', [
    ('dc', DC('drive', 0, 1, 0.5), 3),
    ('transient', Transient(step=1e-9, stop=2e-9), 1),
])
def test_scope_covers_dispatched_analysis_including_voltage_publication(tmp_path, kind, analysis, calls):
    # Module globals deliberately observe writes from run-local model copies.
    global log
    log = []

    def model(ports, t, params):
        log.append(('model', _kind_or_raised()))
        ports.vout.v = _OutputValue(log)

    _bench_with(tmp_path, model, analysis).netlist.simulate()
    assert log == [('model', kind), ('publish', kind)] * calls
    _raises_outside()




class _AnalysisNumber:
    def __float__(self):
        return 1.0 if _kind() == 'transient' else 2.0


@pytest.mark.parametrize('declared,expected', [
    (Transient(step=1e-9, stop=2e-9), 1.0),
    (DC('drive', 0, 1, 0.5), 2.0),
])
def test_analysis_covers_result_conversion_in_the_dispatch(tmp_path, declared, expected):
    import numpy as np
    def model(ports, t, params):
        ports.vout.v = np.array([_AnalysisNumber() for _ in t], dtype=object)
    tb = _bench_with(tmp_path, model, declared)
    tb.netlist.simulate()
    np.testing.assert_array_equal(tb.netlist.load_signal('vout').y, [expected] * 3)
    _raises_outside()
