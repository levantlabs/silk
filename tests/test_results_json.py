"""Collection wire contract: identity, ordering, failures and offline evidence."""
import copy
import json
from pathlib import Path

import pytest

from silk.circuit import spec_report as api
from silk.circuit.spec_report import CornerResults, SpecReport

FIXTURES = Path(__file__).parent / 'fixtures' / 'reports'


def record():
    return SpecReport.load_json(FIXTURES / 'spec_report_v1_wire.json')


def test_collection_adapters_are_public():
    for name in ('results_to_dict', 'results_from_dict', 'save_results_json', 'load_results_json'):
        assert callable(getattr(api, name, None)), name


def test_tests_corners_failures_and_order(tmp_path):
    sr = record()
    results = {'$/left:mirror.bias': CornerResults({'tt': sr}, failed={'ss': 'original error'},
                                                 skip_unavailable=True),
               'failed:test': {'error': 'Error: failed test'}, 'last': sr}
    wire = api.results_to_dict(results)
    assert (wire['format'], wire['schema_version'], wire['kind']) == ('silk.test-results', 1, 'tests')
    assert [e['key'] for e in wire['entries']] == list(results)
    corners = wire['entries'][0]['result']
    assert corners['skip_unavailable'] is True
    assert [e['key'] for e in corners['entries']] == ['tt', 'ss']
    assert corners['entries'][1]['result'] == {'kind': 'failure', 'error': 'original error'}
    assert corners['entries'][0]['result'] == json.loads((FIXTURES / 'spec_report_v1_wire.json').read_text())
    path = tmp_path / 'results.json'
    api.save_results_json(results, path)
    restored = api.load_results_json(path)
    assert list(restored) == list(results)
    assert restored['failed:test'] == {'error': 'Error: failed test'}
    assert restored['$/left:mirror.bias'].failed == {'ss': 'original error'}
    assert restored['$/left:mirror.bias']._skip_unavailable is True
    assert api.results_to_dict(restored) == wire


@pytest.mark.parametrize('results,kind', [({}, 'tests'), ([], 'records'), (CornerResults({}), 'corners')])
def test_empty_collections(results, kind):
    wire = api.results_to_dict(results)
    assert wire['kind'] == kind
    assert wire['entries'] == []
    assert api.results_to_dict(api.results_from_dict(wire)) == wire


def test_records_keep_duplicates_and_detach():
    sr = record()
    wire = api.results_to_dict([sr, sr])
    assert wire['kind'] == 'records'
    assert len(wire['entries']) == 2
    loaded = api.results_from_dict(wire)
    assert loaded[0] is not loaded[1]
    assert api.results_to_dict(loaded) == wire
    loaded[0].metrics['added'] = 3
    assert 'added' not in loaded[1].metrics


@pytest.mark.parametrize('value', [object(), (), [object()], {1: SpecReport('t', 'm', {})},
                                 {'t': {'error': 2}}, {'t': {'custom': 3}}])
def test_unsupported_containers(value):
    with pytest.raises((TypeError, ValueError), match=r'\$'):
        api.results_to_dict(value)


def test_monte_carlo_corner_only_supported():
    from silk.circuit.monte_carlo import MonteCarloResults, MonteCarloSettings
    mc = MonteCarloResults({('tt', 0): SpecReport('ac', 'amp', {'gain': 2.0})}, seeds={('tt', 0): 7},
                           settings=MonteCarloSettings(n=1))
    restored = api.results_from_dict(api.results_to_dict(mc))
    assert restored.table() == mc.table()
    assert restored.summary() == mc.summary()


def test_duplicate_corner_keys_and_cycles_rejected():
    sr = record()
    with pytest.raises(ValueError, match='duplicate'):
        api.results_to_dict(CornerResults({'tt': sr}, failed={'tt': 'failed'}))
    circular = {}
    circular['self'] = circular
    with pytest.raises((TypeError, ValueError), match=r'\$'):
        api.results_to_dict(circular)


@pytest.mark.parametrize('change,field', [
    (lambda d: d.pop('entries'), 'entries'),
    (lambda d: d.update(schema_version=True), 'schema_version'),
    (lambda d: d.update(schema_version=3), 'schema_version'),
    (lambda d: d.update(format='other'), 'format'),
    (lambda d: d.update(kind='other'), 'kind'),
    (lambda d: d.update(entries={}), 'entries'),
    (lambda d: d['entries'].append(copy.deepcopy(d['entries'][0])), 'duplicate'),
    (lambda d: d['entries'][0].update(key=3), 'key'),
    (lambda d: d['entries'][0].update(result={'kind': 'failure', 'error': 5}), 'error'),
])
def test_malformed_wire(change, field):
    wire = api.results_to_dict({'t': record()})
    change(wire)
    with pytest.raises(ValueError, match=field):
        api.results_from_dict(wire)


def test_invalid_data_does_not_truncate_destination(tmp_path):
    path = tmp_path / 'existing.json'
    path.write_text('keep')
    with pytest.raises(TypeError):
        api.save_results_json([SpecReport('t', 'm', {'bad': object()})], path)
    assert path.read_text() == 'keep'
    path.write_text('{}')
    with pytest.raises(ValueError, match='existing.json.*format'):
        api.load_results_json(path)


def test_corner_skip_policy_survives_load():
    import math
    import warnings
    for skip in (False, True):
        source = CornerResults({'good': SpecReport('t', 'm', {'x': 4}),
                                'nan': SpecReport('t', 'm', {'x': float('nan')})},
                               failed={'failed': 'no data'}, skip_unavailable=skip)
        restored = api.results_from_dict(api.results_to_dict(source))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            value = restored.stat('x', 'mean')
        assert value == 4 if skip else math.isnan(value)
        assert restored.passed is False


@pytest.mark.parametrize('payload,field', [
    ({'format': 'silk.test-results', 'schema_version': 1, 'kind': 'corners', 'entries': []}, 'skip_unavailable'),
    ({'format': 'silk.test-results', 'schema_version': 1, 'kind': 'corners', 'entries': [], 'skip_unavailable': 1}, 'skip_unavailable'),
    ({'format': 'silk.test-results', 'schema_version': 1, 'kind': 'records', 'entries': [{'kind': 'failure', 'error': 'no'}]}, 'format'),
])
def test_malformed_corner_and_record_nodes(payload, field):
    with pytest.raises(ValueError, match=field):
        api.results_from_dict(payload)


def test_frozen_collection_fixture(tmp_path):
    wire = json.loads((FIXTURES / 'test_results_v1_wire.json').read_text())
    restored = api.results_from_dict(wire)
    assert list(restored) == ['$/left:mirror.bias', 'failed:test', 'last']
    assert restored['failed:test'] == {'error': 'Error: failed test'}
    assert restored['$/left:mirror.bias'].failed == {'ss': 'original error'}
    assert restored['last'].op_devices is None
    api.save_results_json(restored, tmp_path / 'collection.json')
    assert json.loads((tmp_path / 'collection.json').read_text()) == wire


def grid_collection(mc=False):
    from silk.circuit.monte_carlo import MonteCarloResults, MonteCarloSettings, SampleFailure
    points = [('ss', -40.0, 1.6), ('tt', 25.0, 1.8), ('ff', 125.0, 2.0)]
    keys = [(p, 0) for p in points] if mc else points
    inventory = [dict(key=k, corner=p[0], temperature=p[1], supply=p[2],
                                           supply_source='vdd') for k, p in zip(keys, points)]
    sr = SpecReport('ac', 'amp', {'gain': 2.0}).check('gain floor', 'gain', '>', 1.0)
    kw = dict(run_inventory=inventory, metric_names=['gain'],
              check_declarations=[dict(key='gain floor', name='gain floor', metric='gain', op='>', target=1.0)])
    if mc:
        return MonteCarloResults({keys[1]: sr}, failed={keys[0]: SampleFailure('no data', 7)},
            seeds={k: 7+i for i, k in enumerate(keys)}, settings=MonteCarloSettings(n=1), **kw)
    return CornerResults({keys[1]: sr}, failed={keys[0]: 'no data'}, skip_unavailable=True, **kw)


@pytest.mark.parametrize('mc', [False, True])
def test_grid_roundtrip_preserves_inventory_failures_and_summary(tmp_path, mc):
    import warnings
    source = grid_collection(mc)
    path = tmp_path / 'grid.json'
    api.save_results_json({'ac': source}, path)
    restored = api.load_results_json(path)['ac']
    assert type(restored) is type(source)
    assert restored.table() == source.table()
    assert restored._table_columns() == source._table_columns()
    assert restored._ordered_keys() == source._ordered_keys()
    assert restored.failed == source.failed
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert restored.summary() == source.summary()
    assert api.results_to_dict(restored) == api.results_to_dict(source)


@pytest.mark.parametrize('point', [('tt', 27.0, None), ('tt', None, 1.8), ('tt', None, None)])
def test_grid_coordinates_allow_unconfigured_axes(point):
    original = CornerResults({point: SpecReport('ac', 'amp', {'gain': 2.0})})
    restored = api.results_from_dict(api.results_to_dict(original))
    assert list(restored.reports) == [point]


def test_monte_carlo_metadata_draws_and_empty_inventory_roundtrip():
    import numpy as np
    from silk.circuit.monte_carlo_draws import DrawTable
    source = grid_collection(True)
    source.draws = DrawTable((('m1', 'vth'),), {key: np.array([.3]) for key in source.seeds}, True)
    source.rerun_samples = (next(iter(source.seeds)),)
    restored = api.results_from_dict(api.results_to_dict(source))
    assert restored._rerun_hook is None
    assert restored._campaign_id == source._campaign_id
    assert restored.settings == source.settings
    assert restored.rerun_samples == source.rerun_samples
    assert restored._conditions == source._conditions
    assert restored.draws.columns == source.draws.columns
    for key in source.draws.values:
        np.testing.assert_array_equal(restored.draws.values[key], source.draws.values[key])
    empty = CornerResults({}, run_inventory=[], metric_names=['gain'], check_declarations=[])
    restored = api.results_from_dict(api.results_to_dict(empty))
    assert restored._run_inventory == ()
    assert restored._table_metric_names == ('gain',)
    assert restored.passed is False


@pytest.mark.parametrize('change,field', [
    (lambda d: d['entries'][0]['key'].update(temperature=True), 'key'),
    (lambda d: d['entries'].append(copy.deepcopy(d['entries'][0])), 'duplicate'),
    (lambda d: d.update(run_inventory=[]), 'inventory'),
    (lambda d: d.update(metric_names=[3]), 'metric_names'),
    (lambda d: d.update(check_declarations=[{}]), 'check_declarations'),
])
def test_malformed_grid_wire(change, field):
    wire = api.results_to_dict(grid_collection())
    change(wire)
    with pytest.raises(ValueError, match=field):
        api.results_from_dict(wire)


@pytest.mark.parametrize('field,value', [
    ('conditions', [dict(point=dict(kind='point', corner='tt', temperature=25., supply=1.8), records=[[]])]),
    ('draws', dict(columns=None, values=[], raw=True, skipped_types=[])),
])
def test_malformed_monte_carlo_metadata_names_field(field, value):
    wire = api.results_to_dict(grid_collection(True))
    wire['monte_carlo'][field] = value
    with pytest.raises(ValueError, match=field):
        api.results_from_dict(wire)


def test_malformed_monte_carlo_spec_inventory_names_field():
    wire = api.results_to_dict(grid_collection(True))
    wire['monte_carlo']['spec_declarations'] = [{}]
    with pytest.raises(ValueError, match='spec_declarations'):
        api.results_from_dict(wire)


@pytest.mark.parametrize('field,bad', [
    ('temperature', ['bad']), ('temperature', True), ('temperature', float('inf')),
    ('supply', 'bad'), ('supply', False), ('supply_source', ['bad']),
    ('corner', 3),
])
@pytest.mark.parametrize('file_reader', [False, True])
def test_v3_inventory_scalar_validation_has_indexed_paths(tmp_path, field, bad, file_reader):
    wire = api.results_to_dict(grid_collection())
    wire['run_inventory'][0][field] = bad
    path = tmp_path / 'bad-inventory.json'
    if file_reader:
        from silk._json_codec import encode_json
        path.write_text(json.dumps(encode_json(wire, complex_values=True), allow_nan=False))
    with pytest.raises(ValueError) as exc:
        api.load_results_json(path) if file_reader else api.results_from_dict(wire)
    assert f'$.run_inventory[0].{field}' in str(exc.value)
    if file_reader:
        assert str(path) in str(exc.value)


@pytest.mark.parametrize('location', ['point', 'inventory'])
@pytest.mark.parametrize('file_reader', [False, True])
def test_huge_coordinate_integer_has_validation_path(tmp_path, location, file_reader):
    wire = api.results_to_dict(grid_collection())
    if location == 'point':
        wire['entries'][0]['key']['temperature'] = 10**400
        expected_path = '$.key'
    else:
        wire['run_inventory'][0]['temperature'] = 10**400
        expected_path = '$.run_inventory[0].temperature'
    path = tmp_path / 'huge-coordinate.json'
    path.write_text(json.dumps(wire))
    with pytest.raises(ValueError) as exc:
        api.load_results_json(path) if file_reader else api.results_from_dict(wire)
    assert expected_path in str(exc.value)
    if file_reader:
        assert str(path) in str(exc.value)


@pytest.mark.parametrize('bad', [None, [], '', 3])
@pytest.mark.parametrize('file_reader', [False, True])
def test_v3_campaign_identity_is_required_on_restore(tmp_path, bad, file_reader):
    wire = api.results_to_dict(grid_collection(True))
    wire['monte_carlo']['campaign_id'] = bad
    path = tmp_path / 'bad-campaign.json'
    path.write_text(json.dumps(wire))
    with pytest.raises(ValueError) as exc:
        api.load_results_json(path) if file_reader else api.results_from_dict(wire)
    assert '$.monte_carlo.campaign_id' in str(exc.value)
    if file_reader:
        assert str(path) in str(exc.value)


@pytest.mark.parametrize('index,bad', [(0, []), (1, []), (2, []), (3, 'bad'), (4, []), (5, [])])
@pytest.mark.parametrize('file_reader', [False, True])
def test_v3_condition_record_values_have_indexed_paths(tmp_path, index, bad, file_reader):
    wire = api.results_to_dict(grid_collection(True))
    wire['monte_carlo']['conditions'][0]['records'][0][index] = bad
    path = tmp_path / 'bad-conditions.json'
    path.write_text(json.dumps(wire))
    with pytest.raises(ValueError) as exc:
        api.load_results_json(path) if file_reader else api.results_from_dict(wire)
    assert f'$.monte_carlo.conditions[0].records[0][{index}]' in str(exc.value)
    if file_reader:
        assert str(path) in str(exc.value)


def test_v3_nullable_inventory_and_tagged_conditions_keep_identity():
    wire = api.results_to_dict(grid_collection(True))
    for run in wire['run_inventory']:
        run.update(temperature=None, supply=None, supply_source=None)
    wire['monte_carlo']['conditions'][0]['records'][0][2] = {'state': 'sweep'}
    wire['monte_carlo']['conditions'][0]['records'][0][3] = {'state': 'none'}
    restored = api.results_from_dict(wire)
    assert restored._campaign_id == wire['monte_carlo']['campaign_id']
    assert api.results_to_dict(restored) == wire


@pytest.mark.parametrize('value', [True, 1, 1.0, '1'])
def test_variable_point_wire_retains_scalar_types(value):
    point = ('tt', 27.0, None, (('mode', value),))
    source = CornerResults({point: record()}, run_inventory=[dict(
        key=point, corner='tt', temperature=27.0, supply=None, supply_source=None,
        variables=[['mode', value]], groups=[])])
    wire = api.results_to_dict({'test': source})
    assert wire['schema_version'] == 4
    saved = wire['entries'][0]['result']
    assert saved['schema_version'] == 4
    assert type(saved['entries'][0]['key']['variables'][0][1]) is type(value)
    restored = api.results_from_dict(wire)['test']
    got = next(iter(restored.keys()))
    assert got == point and type(got[3][0][1]) is type(value)
    assert restored.table()[0]['run.mode'] == value
    assert api.results_to_dict({'test': restored}) == wire


def test_grouped_pvt_only_wire_roundtrip_and_failure_inventory():
    source = grid_collection()
    source._run_inventory = tuple(dict(run, variables=[], groups=[['corner', 'temperature', 'supply']])
                                  for run in source._run_inventory)
    wire = api.results_to_dict({'grouped': source})
    assert wire['schema_version'] == 4
    assert wire['entries'][0]['result']['schema_version'] == 4
    restored = api.results_from_dict(wire)['grouped']
    assert restored.failed == source.failed
    assert restored._ordered_keys() == source._ordered_keys()
    assert api.results_to_dict({'grouped': restored}) == wire
    assert api.results_to_dict(grid_collection())['schema_version'] == 3


def test_grouped_wire_refuses_unknown_or_repeated_members():
    source = grid_collection()
    source._run_inventory = tuple(dict(run, variables=[], groups=[['corner', 'temperature']])
                                  for run in source._run_inventory)
    wire = api.results_to_dict(source)
    for groups in ([['corner', 'unknown']], [['corner'], ['corner']]):
        broken = copy.deepcopy(wire)
        for run in broken['run_inventory']:
            run['groups'] = groups
        with pytest.raises(ValueError, match='groups'):
            api.results_from_dict(broken)


@pytest.mark.parametrize('variables', [False, True])
@pytest.mark.parametrize('failure_first', [False, True])
def test_v4_envelope_keeps_highest_version_with_enriched_failure(variables, failure_first):
    from silk.circuit.test_failure import TestFailure
    source = grid_collection()
    if variables:
        point = ('tt', 27.0, None, (('trim', 1),))
        source = CornerResults({point: record()}, run_inventory=[dict(
            key=point, corner='tt', temperature=27.0, supply=None, supply_source=None,
            variables=[['trim', 1]], groups=[])])
    else:
        source._run_inventory = tuple(dict(run, variables=[], groups=[['corner', 'temperature']])
                                      for run in source._run_inventory)
    failure = TestFailure.from_exception(ValueError('unavailable'))
    entries = [('grouped', source), ('failed', failure)]
    wire = api.results_to_dict(dict(reversed(entries) if failure_first else entries))
    assert wire['schema_version'] == 4
    assert api.results_to_dict(api.results_from_dict(wire)) == wire
