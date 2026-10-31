"""Decoded collection builders shared by standalone and circuit reports."""
from silk.circuit._spec_report_json import (
    FORMAT as REPORT_FORMAT, _record, _require, plain_data, report_data, restore_report,
)
from silk.circuit.test_failure import RunRecord, TestFailure
from silk.simulation.conditions import encode_condition, decode_condition

FORMAT = 'silk.test-results'


def _unsupported(value, path):
    raise TypeError(f'{path}: unsupported {type(value).__name__}; expected a tests dict, '
                    'CornerResults, MonteCarloResults, or list of SpecReports')


def _report_data(value, path):
    from silk.circuit.spec_report import SpecReport
    if not isinstance(value, SpecReport):
        _unsupported(value, path)
    try:
        return report_data(value)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f'{path}: {exc}') from exc


def results_data(results, path='$'):
    """Build a fresh unencoded collection; never consult a runner or rawfile."""
    from silk.circuit.spec_report import CornerResults, SpecReport
    from silk.circuit.monte_carlo import MonteCarloResults
    if type(results) is MonteCarloResults or (type(results) is CornerResults and (
            results._run_inventory is not None or results._is_grid())):
        return _collection_data(results, path)
    if type(results) is list:
        return dict(format=FORMAT, schema_version=1, kind='records',
                    entries=[_report_data(sr, f'{path}.entries[{i}]')
                             for i, sr in enumerate(results)])
    if type(results) not in (dict, CornerResults):
        _unsupported(results, path)
    corners = type(results) is CornerResults
    reports = results.reports if corners else results
    entries = []
    version = 1     # Highest envelope version required by its entries
    for key, result in reports.items():
        entrypath = f'{path}.entries[{len(entries)}]'
        if not isinstance(key, str):
            raise TypeError(f'{entrypath}.key: mapping keys must be strings')
        if isinstance(result, SpecReport):
            data = _report_data(result, f'{entrypath}.result')
        elif not corners and type(result) in (CornerResults, MonteCarloResults):
            data = results_data(result, f'{entrypath}.result')
        elif not corners and isinstance(result, TestFailure):
            if set(result) != {'error'}:
                raise ValueError(f'{entrypath}.result: a failure holds exactly the key '
                                 f"'error', not {sorted(map(str, result))}")
            _require(result['error'], str, f'{entrypath}.result.error')
            data = dict(kind='failure', error=result['error'])
            if result.error_type is not None:
                data.update(error_type=result.error_type,
                            error_class=result.error_class,
                            details=plain_data(result.details, f'{entrypath}.result.details'),
                            run=None if result.run is None else result.run.to_dict())
                version = max(version, 2)
        elif not corners and type(result) is dict and set(result) == {'error'}:
            _require(result['error'], str, f'{entrypath}.result.error')
            data = dict(kind='failure', error=result['error'])
        else:
            _unsupported(result, f'{entrypath}.result')
        entries.append(dict(key=key, result=data))
        if data.get('schema_version') == 4 and data.get('format') == FORMAT:
            version = 4
    data = dict(format=FORMAT, schema_version=version,
                kind='corners' if corners else 'tests', entries=entries)
    if corners:
        _require(results._skip_unavailable, bool, f'{path}.skip_unavailable')
        data['skip_unavailable'] = results._skip_unavailable
        for key, error in results.failed.items():
            entrypath = f'{path}.entries[{len(entries)}]'
            if not isinstance(key, str):
                raise TypeError(f'{entrypath}.key: mapping keys must be strings')
            if key in reports:
                raise ValueError(f'{entrypath}.key: duplicate key {key!r}')
            _require(error, str, f'{entrypath}.result.error')
            entries.append(dict(key=key, result=dict(kind='failure', error=error)))
    return data


def _restore_report(data, path):
    from silk.circuit.spec_report import SpecReport
    try:
        return restore_report(data, SpecReport)
    except ValueError as exc:
        raise ValueError(f'{path}: {exc}') from exc


def restore_results(data, path='$'):
    """Restore already-decoded nodes, preserving keys, order and corner policy."""
    from silk.circuit.spec_report import CornerResults
    if isinstance(data, dict) and type(data.get('schema_version')) is int and data['schema_version'] in (3, 4) and data.get('kind') != 'tests':
        return _restore_collection(data, path)
    data = _record(data, ('format', 'schema_version', 'kind', 'entries') +
                   (('skip_unavailable',) if isinstance(data, dict) and
                    data.get('kind') == 'corners' else ()), path)
    if data['format'] != FORMAT:
        raise ValueError(f'{path}.format: expected {FORMAT!r}')
    if type(data['schema_version']) is not int or data['schema_version'] not in (1, 2, 4):
        raise ValueError(f'{path}.schema_version: expected supported integer version 1, 2 or 4')
    kind = data['kind']
    if data['schema_version'] == 4 and kind != 'tests':
        raise ValueError(f'{path}.schema_version: version 4 envelope requires tests')
    if kind not in ('tests', 'corners', 'records'):
        raise ValueError(f'{path}.kind: expected tests, corners or records')
    _require(data['entries'], list, f'{path}.entries')
    if kind == 'records':
        return [_restore_report(sr, f'{path}.entries[{i}]')
                for i, sr in enumerate(data['entries'])]
    if kind == 'corners':
        _require(data['skip_unavailable'], bool, f'{path}.skip_unavailable')
    reports, failed, seen = {}, {}, set()
    for i, entry in enumerate(data['entries']):
        entrypath = f'{path}.entries[{i}]'
        entry = _record(entry, ('key', 'result'), entrypath)
        key = entry['key']
        _require(key, str, f'{entrypath}.key')
        if key in seen:
            raise ValueError(f'{entrypath}.key: duplicate key {key!r}')
        seen.add(key)
        resultpath = f'{entrypath}.result'
        result = entry['result']
        _require(result, dict, resultpath)
        if result.get('kind') == 'failure':
            enriched = 'error_type' in result
            if enriched and (kind == 'corners' or data['schema_version'] < 2):
                raise ValueError(f'{resultpath}: error_type needs a version 2 tests collection')
            failure = _record(result, ('kind', 'error') + (
                ('error_type', 'error_class', 'details', 'run') if enriched else ()),
                resultpath)
            _require(failure['error'], str, f'{resultpath}.error')
            if kind == 'corners':
                failed[key] = failure['error']
            elif enriched:
                _require(failure['error_type'], str, f'{resultpath}.error_type')
                _require(failure['error_class'], str, f'{resultpath}.error_class')
                _require(failure['details'], dict, f'{resultpath}.details')
                run = failure['run']
                reports[key] = TestFailure(
                    failure['error'], error_type=failure['error_type'],
                    error_class=failure['error_class'], details=failure['details'],
                    run=None if run is None else RunRecord.from_dict(run, f'{resultpath}.run'),
                    exception_is_original=False)
            else:
                reports[key] = {'error': failure['error']}
        elif result.get('format') == REPORT_FORMAT:
            reports[key] = _restore_report(result, resultpath)
        elif kind == 'tests' and result.get('kind') in ('corners', 'monte_carlo'):
            reports[key] = restore_results(result, resultpath)
        else:
            raise ValueError(f'{resultpath}: expected a per-test record, '
                             'failure node or nested corners for tests')
    if kind == 'corners':
        return CornerResults(reports, failed=failed, skip_unavailable=data['skip_unavailable'])
    return reports


def successful_reports(results):
    """Iterate records from a validated/restored collection in supplied order."""
    from silk.circuit.spec_report import CornerResults, SpecReport
    if isinstance(results, list):
        yield from results
    else:
        values = results.reports.values() if isinstance(results, CornerResults) else results.values()
        for value in values:
            if isinstance(value, SpecReport):
                yield value
            elif isinstance(value, CornerResults):
                yield from successful_reports(value)


def _key_data(key, path, *, sample=False):
    """Explicit coordinate tags keep lists of evidence separate from run keys."""
    if sample:
        if not isinstance(key, tuple) or len(key) != 2 or type(key[1]) is not int or key[1] < 0:
            raise ValueError(f'{path}: expected (point, nonnegative sample index)')
        return dict(kind='sample', point=_key_data(key[0], path + '.point'), sample=key[1])
    if isinstance(key, str):
        return dict(kind='corner', corner=key)
    if isinstance(key, tuple) and len(key) in (3, 4) and isinstance(key[0], str):
        import math
        from numbers import Real
        for value in key[1:3]:
            try:
                invalid = value is not None and (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value))
            except OverflowError:
                invalid = True
            if invalid:
                raise ValueError(f'{path}: temperature and supply must be finite numbers or null')
        data = dict(kind='point', corner=key[0], temperature=key[1], supply=key[2])
        if len(key) == 4:
            if not isinstance(key[3], tuple) or not key[3]:
                raise ValueError(f'{path}: variables must be nonempty named coordinates')
            names = set()
            for pair in key[3]:
                if not isinstance(pair, tuple) or len(pair) != 2 or not isinstance(pair[0], str):
                    raise ValueError(f'{path}: expected (Variable name, value) pairs')
                name, value = pair
                if name in names or not isinstance(value, (str, bool, int, float)):
                    raise ValueError(f'{path}: duplicate Variable name or invalid value')
                if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
                    raise ValueError(f'{path}: Variable coordinates must be finite')
                names.add(name)
            data['variables'] = [list(pair) for pair in key[3]]
        return plain_data(data)
    raise ValueError(f'{path}: expected corner string or (corner, temperature, supply)')


def _restore_key(data, path, *, sample=False):
    _require(data, dict, path)
    if sample:
        data = _record(data, ('kind', 'point', 'sample'), path)
        key = (_restore_key(data['point'], path + '.point'), data['sample'])
    elif data.get('kind') == 'corner':
        key = _record(data, ('corner',), path)['corner']
    elif data.get('kind') == 'point':
        data = _record(data, ('kind', 'corner', 'temperature', 'supply') + (('variables',) if 'variables' in data else ()), path)
        key = (data['corner'], data['temperature'], data['supply'])
        if 'variables' in data:
            _require(data['variables'], list, path + '.variables')
            key += (tuple(tuple(pair) for pair in data['variables']),)
    else:
        raise ValueError(f'{path}.kind: expected corner or point')
    if _key_data(key, path, sample=sample) != data:
        # Only the defined coordinate representation is accepted.
        raise ValueError(f'{path}: invalid coordinate representation')
    return key


def _collection_data(results, path):
    from dataclasses import asdict
    from silk.circuit.monte_carlo import MonteCarloResults
    mc = type(results) is MonteCarloResults
    key_data = lambda key: _key_data(key, path + '.key', sample=mc)
    entries = []
    for key, report in results.reports.items():
        if key in results.failed:
            raise ValueError(f'{path}: duplicate key {key!r}')
        entries.append(dict(key=key_data(key), result=_report_data(report, path + '.result')))
    for key, failure in results.failed.items():
        error = failure.error if mc else failure
        _require(error, str, path + '.error')
        result = dict(kind='failure', error=error)
        if mc:
            _require(failure.seed, int, path + '.seed')
            result['seed'] = failure.seed
        entries.append(dict(key=key_data(key), result=result))
    inventory = None if results._run_inventory is None else [
        dict(run, key=key_data(run['key'])) for run in results._run_inventory]
    extended = any(run.get('variables') or run.get('groups') for run in results._run_inventory or ())
    extended = extended or bool(results._variable_names())
    data = dict(format=FORMAT, schema_version=4 if extended else 3, kind='monte_carlo' if mc else 'corners',
                entries=entries, skip_unavailable=results._skip_unavailable,
                run_inventory=inventory, metric_names=list(results._table_metric_names),
                check_declarations=list(results._check_declarations))
    if mc:
        draws = results.draws
        data['monte_carlo'] = dict(
            settings=asdict(results.settings),
            seeds=[dict(key=key_data(k), seed=v) for k, v in results.seeds.items()],
            spec_declarations=[d._asdict() for d in results._spec_declarations],
            structure_hash=results.structure_hash, campaign_id=results._campaign_id,
            campaign_source=results._campaign_source, draw_scheme=results._draw_scheme,
            conditions=[dict(point=_key_data(k, path + '.conditions'),
                             records=[[r[0], *(encode_condition(v) for v in r[1:4]), *r[4:]]
                                      for r in records])
                        for k, records in results._conditions],
            rerun_samples=[key_data(k) for k in results.rerun_samples],
            draws=None if draws is None else dict(columns=draws.columns, raw=draws.raw,
                skipped_types=draws.skipped_types,
                values=[dict(key=key_data(k), values=v.tolist()) for k, v in draws.values.items()]))
    return plain_data(data, path)


def _restore_collection(data, path):
    from silk.circuit.spec_report import CornerResults
    from silk.circuit.monte_carlo import MonteCarloResults, MonteCarloSettings, SampleFailure
    from silk.circuit.monte_carlo_draws import DrawTable
    from silk.exceptions import MonteCarloError
    if data.get('kind') not in ('corners', 'monte_carlo'):
        raise ValueError(f'{path}.schema_version: version 3 requires corners or monte_carlo')
    fields = ('format', 'schema_version', 'kind', 'entries', 'skip_unavailable',
              'run_inventory', 'metric_names', 'check_declarations')
    mc = data.get('kind') == 'monte_carlo'
    data = _record(data, fields + (('monte_carlo',) if mc else ()), path)
    if data['format'] != FORMAT or data['kind'] not in ('corners', 'monte_carlo'):
        raise ValueError(f'{path}: version 3 expects corners or monte_carlo with format {FORMAT}')
    _require(data['skip_unavailable'], bool, path + '.skip_unavailable')
    _require(data['entries'], list, path + '.entries')
    _require(data['metric_names'], list, path + '.metric_names')
    for name in data['metric_names']:
        _require(name, str, path + '.metric_names')
    _require(data['check_declarations'], list, path + '.check_declarations')
    for declaration in data['check_declarations']:
        _declaration(declaration, path + '.check_declarations', check=True)
    restore_key = lambda value: _restore_key(value, path + '.key', sample=mc)
    reports, failed, seen = {}, {}, set()
    for i, item in enumerate(data['entries']):
        entrypath = f'{path}.entries[{i}]'
        item = _record(item, ('key', 'result'), entrypath)
        key = restore_key(item['key'])
        if key in seen:
            raise ValueError(f'{entrypath}: duplicate key {key!r}')
        seen.add(key)
        result = item['result']
        _require(result, dict, entrypath + '.result')
        if result.get('kind') == 'failure':
            result = _record(result, ('error',) + (('seed',) if mc else ()), entrypath)
            _require(result['error'], str, entrypath + '.error')
            if mc:
                _require(result['seed'], int, entrypath + '.seed')
            failed[key] = SampleFailure(result['error'], result['seed']) if mc else result['error']
        else:
            reports[key] = _restore_report(result, entrypath + '.result')
    inventory = data['run_inventory']
    if inventory is not None:
        _require(inventory, list, path + '.run_inventory')
        restored_inventory = []
        plan_metadata = None
        for i, run in enumerate(inventory):
            runpath = f'{path}.run_inventory[{i}]'
            run = _record(run, ('key', 'corner', 'temperature', 'supply', 'supply_source') +
                          (('variables', 'groups') if data['schema_version'] == 4 else ()), runpath)
            if data['schema_version'] == 4:
                _require(run['variables'], list, runpath + '.variables')
                _require(run['groups'], list, runpath + '.groups')
                if any(not isinstance(group, list) or not group or
                       not all(isinstance(name, str) for name in group) for group in run['groups']):
                    raise ValueError(f'{runpath}.groups: expected nonempty named groups')
            run['key'] = _restore_key(run['key'], runpath + '.key', sample=mc)
            if data['schema_version'] == 4:
                point = run['key'][0] if mc else run['key']
                expected = [list(pair) for pair in point[3]] if isinstance(point, tuple) and len(point) == 4 else []
                if run['variables'] != expected or any(
                        type(pair[1]) is not type(want[1]) for pair, want in zip(run['variables'], expected)):
                    raise ValueError(f'{runpath}.variables: must match the key coordinates and types')
                labels = ['corner', 'temperature', 'supply'] + [pair[0] for pair in expected]
                members = [name for group in run['groups'] for name in group]
                if len(set(members)) != len(members) or any(name not in labels for name in members):
                    raise ValueError(f'{runpath}.groups: unknown or repeated group member')
                metadata = ([pair[0] for pair in expected], run['groups'])
                if plan_metadata is not None and metadata != plan_metadata:
                    raise ValueError(f'{runpath}.groups: inconsistent plan metadata')
                plan_metadata = metadata
            _require(run['corner'], str, runpath + '.corner', nullable=True)
            for field in ('temperature', 'supply'):
                _coordinate(run[field], runpath + '.' + field)
            _require(run['supply_source'], str, runpath + '.supply_source', nullable=True)
            restored_inventory.append(run)
        inventory = restored_inventory
        keys = [run['key'] for run in inventory]
        if len(set(keys)) != len(keys) or not seen.issubset(keys):
            raise ValueError(f'{path}.run_inventory: duplicate keys or missing recorded runs')
    kw = dict(run_inventory=inventory, metric_names=data['metric_names'],
              check_declarations=data['check_declarations'])
    if not mc:
        return CornerResults(reports, failed=failed, skip_unavailable=data['skip_unavailable'], **kw)
    meta = _record(data['monte_carlo'], ('settings', 'seeds', 'spec_declarations', 'structure_hash',
        'campaign_id', 'campaign_source', 'draw_scheme', 'conditions', 'rerun_samples', 'draws'), path + '.monte_carlo')
    _require(meta['campaign_id'], str, path + '.monte_carlo.campaign_id')
    if not meta['campaign_id'].strip():
        raise ValueError(f'{path}.monte_carlo.campaign_id: expected a nonempty captured identity')
    _require(meta['settings'], dict, path + '.monte_carlo.settings')
    _require(meta['seeds'], list, path + '.monte_carlo.seeds')
    seeds = {}
    for item in meta['seeds']:
        item = _record(item, ('key', 'seed'), path + '.monte_carlo.seeds')
        key = restore_key(item['key'])
        if key in seeds:
            raise ValueError(f'{path}.monte_carlo.seeds: duplicate key')
        _require(item['seed'], int, path + '.monte_carlo.seeds.seed')
        seeds[key] = item['seed']
    for name in ('conditions', 'rerun_samples', 'spec_declarations'):
        _require(meta[name], list, path + '.monte_carlo.' + name)
    for declaration in meta['spec_declarations']:
        _declaration(declaration, path + '.monte_carlo.spec_declarations')
    conditions = []
    for i, item in enumerate(meta['conditions']):
        conditionpath = f'{path}.monte_carlo.conditions[{i}]'
        item = _record(item, ('point', 'records'), conditionpath)
        _require(item['records'], list, conditionpath + '.records')
        records = []
        for j, record in enumerate(item['records']):
            recordpath = f'{conditionpath}.records[{j}]'
            _require(record, list, recordpath)
            if len(record) != 6:
                raise ValueError(f'{recordpath}: expected six fields')
            for index in (0, 4, 5):
                _require(record[index], str, f'{recordpath}[{index}]', nullable=True)
            decoded = []
            for index, fact in enumerate(('corner', 'temperature', 'supply'), 1):
                try:
                    decoded.append(decode_condition(fact, record[index]))
                except ValueError as exc:
                    raise ValueError(f'{recordpath}[{index}]: {exc}') from exc
            records.append((record[0], *decoded, *record[4:]))
        conditions.append((_restore_key(item['point'], conditionpath + '.point'), tuple(records)))
    draws = meta['draws']
    if draws is not None:
        draws = _record(draws, ('columns', 'values', 'raw', 'skipped_types'), path + '.draws')
        _require(draws['raw'], bool, path + '.draws.raw')
        for field in ('columns', 'skipped_types'):
            _require(draws[field], list, path + '.draws.' + field)
        for column in draws['columns']:
            _require(column, list, path + '.draws.columns')
            if len(column) != 2 or not all(isinstance(v, str) for v in column):
                raise ValueError(f'{path}.draws.columns: expected device/parameter string pairs')
        for name in draws['skipped_types']:
            _require(name, str, path + '.draws.skipped_types')
        _require(draws['values'], list, path + '.draws.values')
        values = {}
        for item in draws['values']:
            item = _record(item, ('key', 'values'), path + '.draws.values')
            key = restore_key(item['key'])
            if key in values:
                raise ValueError(f'{path}.draws.values: duplicate key')
            values[key] = item['values']
        try:
            draws = DrawTable(tuple(map(tuple, draws['columns'])), values,
                              draws['raw'], tuple(draws['skipped_types']))
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{path}.draws: {exc}') from exc
    try:
        result = MonteCarloResults(reports, failed=failed, settings=MonteCarloSettings(**meta['settings']),
            seeds=seeds, structure_hash=meta['structure_hash'], draws=draws,
            spec_declarations=meta['spec_declarations'], _campaign_id=meta['campaign_id'],
            campaign_source=meta['campaign_source'], draw_scheme=meta['draw_scheme'],
            _conditions=conditions, rerun_samples=[restore_key(k) for k in meta['rerun_samples']], **kw)
    except (TypeError, ValueError, MonteCarloError) as exc:
        raise ValueError(f'{path}.monte_carlo: {exc}') from exc
    result._skip_unavailable = data['skip_unavailable']
    return result


def _declaration(value, path, *, check=False):
    """Validate summary inputs before collection constructors consume them."""
    from numbers import Real
    from silk.circuit.spec_report import _OPS
    names = ('key', 'name', 'metric', 'op') if check else ('name', 'metric', 'op')
    _record(value, (*names, 'target'), path)
    for name in names:
        _require(value[name], str, path + '.' + name)
    if value['op'] not in _OPS:
        raise ValueError(f'{path}.op: expected one of {sorted(_OPS)}')
    if isinstance(value['target'], bool) or not isinstance(value['target'], Real):
        raise ValueError(f'{path}.target: expected a real number')
    if check and value.get('owner_path') is not None:
        _require(value['owner_path'], str, path + '.owner_path')


def _coordinate(value, path):
    """Validate a nullable requested coordinate without inventing a value."""
    import math
    from numbers import Real
    try:
        invalid = value is not None and (isinstance(value, bool) or not isinstance(value, Real)
                                          or not math.isfinite(value))
    except OverflowError:
        invalid = True
    if invalid:
        raise ValueError(f'{path}: expected a finite real number or null')
