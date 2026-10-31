"""Aggregate evidence is attributed, immutable, and uses one storage codec."""
from dataclasses import replace, FrozenInstanceError
import json
import math
import pickle

import pytest

from silk.circuit.monte_carlo import MonteCarloResults, MonteCarloSettings, SampleFailure
from silk.circuit.spec_report import SpecReport
from silk.exceptions import MonteCarloError
from silk.simulation.conditions import OperatingConditions, ConditionState


def aggregate(**changes):
    from silk.specs import AggregateMeasurement
    fields = dict(metric='gain_dB', statistic='std', variation='mismatch',
                  selected_corner='tt', samples_total=4, samples_used=2,
                  samples_failed=1, samples_unavailable=1, used_indices=(0, 1),
                  sample_seeds=((0, 1), (1, 2), (2, 3), (3, 4)),
                  campaign_id='campaign-1', base_seed=2**70, requested_samples=4,
                  structure_hash=None, draw_scheme=None)
    return AggregateMeasurement(**(fields | changes))


def campaign(corners=('tt',), *, backend='ngspice'):
    reports, failed, seeds = {}, {}, {}
    for corner in corners:
        for index, value in enumerate([1., 3., math.nan]):
            reports[corner, index] = SpecReport('ac', 'm', {'gain_dB': value},
                operating_conditions=(None if backend is None else
                    OperatingConditions('silk', backend, None, corner_kind='none',
                        corner_basis='proven-behavioral') if backend == 'behavioral' else
                    OperatingConditions('silk', backend, corner)))
        failed[corner, 3] = SampleFailure('no data', 4)
        seeds.update({(corner, i): i + 1 for i in range(4)})
    return MonteCarloResults(reports, settings=MonteCarloSettings(n=4, seed=2**70),
                             seeds=seeds, failed=failed)


def test_aggregate_codec_and_pickle_round_trip():
    from silk.specs.aggregate import encode_aggregate, decode_aggregate
    value = aggregate()
    assert decode_aggregate(json.loads(json.dumps(encode_aggregate(value)))) == value
    assert pickle.loads(pickle.dumps(value)) == value
    with pytest.raises(FrozenInstanceError):
        value.samples_total = 0
    assert isinstance(value.sample_seeds, tuple)
    assert all(isinstance(pair, tuple) for pair in value.sample_seeds)


@pytest.mark.parametrize('changes', [dict(samples_used=True), dict(samples_total=5),
    dict(samples_used=1, samples_unavailable=2), dict(used_indices=(0, 0)),
    dict(used_indices=(0, 9)), dict(sample_seeds=((0, 1), (1, 2))),
    dict(sample_seeds=((0, 1), (0, 2), (2, 3), (3, 4)))])
def test_inconsistent_inventory_refuses(changes):
    with pytest.raises((TypeError, ValueError)):
        aggregate(**changes)


def test_codec_rejects_invalid_scheme():
    from silk.specs.aggregate import encode_aggregate, decode_aggregate
    payload = encode_aggregate(aggregate())
    payload['scheme'] = 'silk.measurement.aggregate/v999'
    with pytest.raises(ValueError, match='scheme'):
        decode_aggregate(payload)


def test_bridge_counts_source_std_and_identity():
    mc = campaign()
    with pytest.warns(Warning):
        m = mc.measurement('gain_dB')
    assert m.value == pytest.approx(math.sqrt(2))
    assert m.source == 'spice' and m.corner == 'tt'
    assert m.aggregate.samples_used == 2
    assert m.aggregate.samples_failed == 1
    assert m.aggregate.samples_unavailable == 1
    assert m.aggregate.samples_total == 4
    assert m.aggregate.base_seed == 2**70
    assert m.sim_run_id == m.aggregate.campaign_id
    with pytest.warns(Warning):
        selected = mc.corner('tt').measurement('gain_dB')
        repeated = campaign().measurement('gain_dB')
    assert selected.sim_run_id == m.sim_run_id != repeated.sim_run_id


def test_multiple_corners_need_selection():
    mc = campaign(('tt', 'ss'))
    with pytest.raises(MonteCarloError, match='corner'):
        mc.measurement('gain_dB')
    with pytest.warns(Warning):
        assert mc.measurement('gain_dB', corner='ss').aggregate.selected_corner == 'ss'


def test_campaign_label_does_not_overwrite_effective_none():
    mc = campaign(backend='behavioral')
    with pytest.warns(Warning):
        m = mc.measurement('gain_dB')
    assert m.corner is ConditionState.NONE
    assert m.aggregate.selected_corner == 'tt'


def test_unknown_source_is_not_guessed():
    mc = campaign(backend=None)
    with pytest.raises(MonteCarloError, match='source|attribution'):
        mc.measurement('gain_dB')


def test_wholly_failed_declared_corner_uses_captured_source():
    mc = MonteCarloResults({}, settings=MonteCarloSettings(n=2),
        seeds={('ss', 0): 1, ('ss', 1): 2},
        failed={('ss', 0): SampleFailure('failed', 1), ('ss', 1): SampleFailure('failed', 2)},
        metric_names=['gain_dB'], campaign_source='spice')
    with pytest.warns(Warning, match='every sample failed'):
        m = mc.measurement('gain_dB')
    assert math.isnan(m.value)
    assert (m.corner, m.temperature, m.voltage) == (None, None, None)
    assert (m.aggregate.selected_corner, m.aggregate.samples_total, m.aggregate.samples_failed) == ('ss', 2, 2)
    assert m.aggregate.samples_used == 0 and m.aggregate.used_indices == ()


def test_wholly_failed_without_source_refuses_attribution_only():
    mc = MonteCarloResults({}, settings=MonteCarloSettings(n=1),
        seeds={('ss', 0): 1}, failed={('ss', 0): SampleFailure('failed', 1)},
        metric_names=['gain_dB'])
    with pytest.raises(MonteCarloError, match='source attribution'):
        mc.measurement('gain_dB')
    with pytest.warns(Warning, match='every sample failed'):
        assert math.isnan(mc.stat('gain_dB', 'mean'))


def test_behavioral_draw_identity_and_view_are_captured():
    mc = MonteCarloResults({}, settings=MonteCarloSettings(n=1, seed=2**100),
        seeds={('ss', 0): 1}, failed={('ss', 0): SampleFailure('failed', 1)},
        metric_names=['gain_dB'], campaign_source='behavioral', draw_scheme=1)
    with pytest.warns(Warning):
        m = mc.corner('ss').measurement('gain_dB')
    assert m.aggregate.draw_scheme == 1 and m.aggregate.base_seed == 2**100
    assert m.source == 'behavioral'


def test_captured_metadata_is_not_replaced_by_mutated_report():
    mc = campaign()
    for report in mc.reports.values():
        report.operating_conditions = None
    with pytest.warns(Warning):
        m = mc.measurement('gain_dB')
    assert m.source == 'spice' and m.corner == 'tt'


def test_empty_declared_inventory_with_source_returns_unavailable():
    mc = MonteCarloResults({}, settings=MonteCarloSettings(n=1), seeds={},
                           metric_names=['gain_dB'], campaign_source='spice')
    with pytest.warns(Warning, match='0 used'):
        m = mc.measurement('gain_dB')
    assert math.isnan(m.value)
    assert m.aggregate.samples_total == m.aggregate.samples_used == 0
    assert m.aggregate.selected_corner is None and m.corner is None


def test_inconsistent_attribution_only_refuses_the_measurement_bridge():
    reports = {('tt', 0): SpecReport('ac', 'm', {'gain_dB': 1},
                    operating_conditions=OperatingConditions('silk', 'ngspice', 'tt')),
               ('tt', 1): SpecReport('ac', 'm', {'gain_dB': 3},
                    operating_conditions=OperatingConditions('silk', 'xyce', 'tt',
                        temperature_kind='fixed', temperature_c=80))}
    mc = MonteCarloResults(reports, settings=MonteCarloSettings(n=2),
                           seeds={('tt', 0): 1, ('tt', 1): 2})
    assert mc.stat('gain_dB', 'mean') == 2
    with pytest.raises(MonteCarloError, match='inconsistent'):
        mc.measurement('gain_dB')


def test_tool_identity_is_captured_and_mixed_tools_refuse():
    from silk.simulation.provenance import RunProvenance
    reports = {('tt', i): SpecReport('custom-name', 'm', {'gain_dB': value},
        operating_conditions=OperatingConditions('silk', 'ngspice', 'tt'),
        provenance=RunProvenance(simulator='ngspice', simulator_version='ngspice-42'))
        for i, value in enumerate((1., 3.))}
    mc = MonteCarloResults(reports, settings=MonteCarloSettings(n=2), seeds={('tt', 0): 1, ('tt', 1): 2})
    assert mc.measurement('gain_dB').tool_name == 'ngspice-42'
    assert mc.measurement('gain_dB').sim_type is None
    reports['tt', 1].operating_conditions = OperatingConditions('silk', 'xyce', 'tt')
    reports['tt', 1].provenance = RunProvenance(simulator='xyce', simulator_version='7.10')
    mixed = MonteCarloResults(reports, settings=mc.settings, seeds=mc.seeds)
    assert mixed.stat('gain_dB', 'mean') == 2
    with pytest.raises(MonteCarloError, match='inconsistent'):
        mixed.measurement('gain_dB')


@pytest.mark.parametrize('field', ['used_indices', 'sample_seeds'])
def test_wire_collections_must_be_json_arrays(field):
    from silk.specs.aggregate import encode_aggregate, decode_aggregate
    payload = encode_aggregate(aggregate(samples_total=0, samples_used=0,
        samples_failed=0, samples_unavailable=0, used_indices=(), sample_seeds=()))
    payload[field] = {}
    with pytest.raises(ValueError, match='array'):
        decode_aggregate(payload)


def test_behavioral_campaign_bridge_retains_launch_identity(pdk, tmp_path):
    from tests._variation_bench import chip
    from silk.simulation.variation import DRAW_SCHEME, Normal
    _, bench = chip(pdk, tmp_path, {'offset': Normal(0, 0.001)})
    mc = bench.set_monte_carlo(n=3, seed=2**100).run()
    measurement = mc.measurement('y0')
    assert measurement.value == mc.stat('y0', 'std')
    assert measurement.source == 'behavioral'
    assert measurement.corner is ConditionState.NONE
    assert measurement.aggregate.draw_scheme == DRAW_SCHEME
    assert measurement.aggregate.base_seed == 2**100
    assert measurement.aggregate.samples_total == measurement.aggregate.samples_used == 3
    assert measurement.aggregate.sample_seeds == tuple((index, seed)
        for (_, index), seed in sorted(mc.seeds.items()))
    bench.netlist.simulator = 'ngspice'
    assert mc.measurement('y0').aggregate == measurement.aggregate
    assert mc.measurement('y0').source == 'behavioral'


def test_attempt_snapshot_preserves_aggregate_and_defensive_copy():
    from silk.specs import Measurement, Spec, SpecAttempt
    spec = Spec('gain', module=None, metric='gain_dB', op='<=', target=10)
    measurement = Measurement('spice', 2., sim_run_id='campaign-1', aggregate=aggregate())
    spec.record_measurement(measurement)
    attempt = spec.latest_attempt()
    assert attempt.measurement.aggregate == aggregate()
    restored = SpecAttempt.from_dict(json.loads(json.dumps(attempt.as_dict())))
    assert restored.measurement.aggregate == aggregate()
    measurement.aggregate = None
    snapshot = restored.measurement
    snapshot.aggregate = None
    assert restored.measurement.aggregate == aggregate()


def test_unavailable_attempt_aggregate_codec_and_merge():
    from silk.specs import SpecAttempt
    from silk.specs.attempt import _merge
    evidence = aggregate(samples_used=0, samples_failed=4, samples_unavailable=0, used_indices=())
    attempt = SpecAttempt(source='spice', kind='unavailable', reason='every sample failed',
                          attempt_id='failed-campaign', aggregate=evidence,
                          requested_conditions={'corner': 'tt', 'temperature': 125},
                          proved_conditions=None)
    restored = SpecAttempt.from_dict(json.loads(json.dumps(attempt.as_dict())))
    assert restored.aggregate == evidence
    assert restored.measurement is None
    assert restored.requested_conditions == {'corner': 'tt', 'temperature': 125}
    assert restored.proved_conditions is None
    assert pickle.loads(pickle.dumps(restored)) == restored
    with pytest.raises(FrozenInstanceError):
        restored.aggregate = None
    absent = SpecAttempt(source='spice', kind='unavailable', reason='every sample failed',
                         attempt_id='failed-campaign')
    assert _merge(absent, restored).aggregate == evidence
    assert _merge(restored, absent).aggregate == evidence
    conflicting = SpecAttempt(source='spice', kind='unavailable', reason='every sample failed',
                              attempt_id='failed-campaign',
                              aggregate=replace(evidence, campaign_id='different-campaign'))
    with pytest.raises(ValueError, match='failed-campaign.*aggregate'):
        _merge(restored, conflicting)


@pytest.mark.parametrize('kind', ['measurement', 'extractor_error', 'simulation_error', 'transport_error'])
def test_separate_attempt_aggregate_requires_unavailable(kind):
    from silk.specs import Measurement, SpecAttempt
    kwargs = {'measurement': Measurement('spice', 2., passes=True)} if kind == 'measurement' else {'reason': 'failed'}
    with pytest.raises(ValueError, match='wrong-kind.*unavailable'):
        SpecAttempt(source='spice', kind=kind, attempt_id='wrong-kind', aggregate=aggregate(), **kwargs)


@pytest.mark.parametrize('bad', [{'scheme': 'unknown'}, {'scheme': 'silk.measurement.aggregate/v1'}])
def test_unavailable_attempt_malformed_aggregate_names_attempt(bad):
    from silk.specs import SpecAttempt
    wire = SpecAttempt(source='spice', kind='unavailable', reason='no usable samples',
                       attempt_id='bad-aggregate-attempt').as_dict()
    wire['aggregate'] = bad
    with pytest.raises(ValueError, match='bad-aggregate-attempt.*aggregate'):
        SpecAttempt.from_dict(wire)


def test_legacy_attempt_without_aggregate_stays_unknown():
    from silk.specs import SpecAttempt
    wire = SpecAttempt(source='spice', kind='unavailable', reason='no usable samples').as_dict()
    wire.pop('aggregate', None)
    assert SpecAttempt.from_dict(wire).aggregate is None


@pytest.mark.parametrize('change', [{'sim_run_id': 'different-campaign'},
                                   {'sim_run_id': None},
                                   {'aggregate': aggregate(metric='unrelated_metric')}])
def test_unavailable_conversion_validates_original_attribution_before_recording(change):
    from silk.specs import Measurement, Spec
    spec = Spec('gain', module=None, metric='gain_dB', op='<=', target=10)
    measurement = Measurement('spice', math.nan, sim_run_id='campaign-1', aggregate=aggregate())
    for name, value in change.items():
        setattr(measurement, name, value)
    with pytest.raises(ValueError, match='aggregate'):
        spec.record_measurement(measurement)
    assert spec.attempt_history() == ()
    assert spec.measurements == {}
