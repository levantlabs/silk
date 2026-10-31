"""Run-local statistical resolution of executed behavioural model declarations."""
from dataclasses import dataclass
from typing import Optional

from silk.exceptions import NotSupportedError, VariationError
from silk.simulation.variation import (Resolved, is_declaration, owner_segments, draw_u,
                            mismatch_key, bench_key, process_key)


@dataclass(frozen=True)
class VariationScope:
    owner_path: str = '$'
    test_name: Optional[str] = None
    dut_instance: Optional[str] = None


@dataclass(frozen=True)
class DeclarationRecord:
    block: str
    display: str
    param: str
    key: tuple
    resolved: Resolved
    mismatch_enabled: bool


@dataclass
class RunResolution:
    params: dict
    records: tuple
    draws: dict


def _refuse_nested(value, place, seen=None):
    """Inspect built-in containers and object arrays without following cycles."""
    import numpy as np
    if is_declaration(value):
        raise NotSupportedError(
            f'{place}: a statistical declaration must be a top-level params value')
    if not isinstance(value, (dict, list, tuple, set, frozenset, np.ndarray, np.void)):
        return
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, dict):
        for key, item in value.items():
            _refuse_nested(key, f'{place} key {key!r}', seen)
            _refuse_nested(item, f'{place}[{key!r}]', seen)
    elif isinstance(value, np.void):
        if value.dtype.hasobject:
            for name in value.dtype.names or ():
                _refuse_nested(value[name], f'{place}[{name!r}]', seen)
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            for index, item in enumerate(value.flat):
                _refuse_nested(item, f'{place}[{index}]', seen)
    else:
        for index, item in enumerate(value):
            _refuse_nested(item, f'{place}[{index}]', seen)


def validate_params(params, block):
    """Refuse nested declarations at attachment and again at run preparation."""
    if params is not None:
        for param, value in params.items():
            if not is_declaration(value):
                _refuse_nested(value, f'block {block!r} param {param!r}')


def refuse_calibration(params, param, block):
    if is_declaration(params[param]):
        raise NotSupportedError(
            f'block {block!r} parameter {param!r}: calibrating a statistical '
            'declaration needs a field target; use CalibrationBinding(field=...)')


def resolve_run(blocks: dict, *, scope: Optional[VariationScope] = None,
                sample=None) -> RunResolution:
    """Resolve once before any model executes; preserve plain parameter dicts."""
    frozen = {(r.block, r.param): r.resolved
              for r in (getattr(sample, 'behavioral_records', None) or ())}
    scope = scope or VariationScope()
    params_by_block, records, processes = {}, [], {}
    for block in blocks.values():
        params = getattr(block.device, '_behavioral_params', None)
        if params is None:
            continue
        validate_params(params, block.path)
        segments = block.segments
        if scope.dut_instance is None:
            key = ('mismatch', segments)
        elif segments and segments[0] == scope.dut_instance:
            key = ('mismatch', owner_segments(scope.owner_path) + segments[1:])
        else:
            key = ('bench', scope.owner_path, scope.test_name, segments)
        for param, value in params.items():
            if not is_declaration(value):
                continue
            try:
                resolved = (frozen[(block.path, param)] if frozen else value.resolve())
            except VariationError as exc:
                raise VariationError(f'block {block.path!r} param {param!r}: {exc}') from exc
            record = DeclarationRecord(
                block.path, '.'.join(segments), param,
                ('process', resolved.name) if resolved.variation == 'process' else key,
                resolved, block.mismatch)
            if resolved.variation == 'process':
                previous = processes.setdefault(resolved.name, record)
                if (previous.resolved.kind, previous.resolved.fields) != (resolved.kind, resolved.fields):
                    raise VariationError(
                        f'process {resolved.name!r} conflicts between '
                        f'block {previous.block!r} param {previous.param!r} and '
                        f'block {block.path!r} param {param!r}')
            if block.path not in params_by_block:
                params_by_block[block.path] = dict(params)
            params_by_block[block.path][param] = resolved.nominal
            records.append(record)
    resolution = RunResolution(params_by_block, tuple(records), {})
    if sample is not None:
        for record in records:
            if not active(record, sample.variation):
                continue
            kind, *parts = record.key
            if kind == 'process':
                key = process_key(sample.base_seed, sample.index, *parts)
                column = ('process', record.resolved.name)
            elif kind == 'bench':
                key = bench_key(sample.base_seed, sample.index, *parts, record.param)
                column = (record.display, record.param)
            else:
                key = mismatch_key(sample.base_seed, sample.index, *parts, record.param)
                column = (record.display, record.param)
            value = record.resolved.at(draw_u(key))
            resolution.params[record.block][record.param] = value
            resolution.draws[column] = value
    return resolution


def active(record, variation):
    """Whether this declaration draws under the requested variation filter."""
    resolved = record.resolved
    return (resolved.has_spread
            and variation in (resolved.variation, 'both')
            and (resolved.variation == 'process' or record.mismatch_enabled))


def resolve_test(tb):
    """Resolve a selected bench without executing models or writing output."""
    from silk.circuit.module_test import DUT_ELEMENT_NAME
    from silk.simulation.behavioral import BehavioralEngine
    from silk.simulation.conditions import apply_inherited_supply
    netlist = tb.netlist
    if netlist is None:
        return RunResolution({}, (), {})
    # The engine needs effective supply rails for passive classification.
    # Discover with the same inheritance as preparation, but leave the
    # authored settings and source values untouched, including on refusal.
    saved = netlist._supply_voltage, netlist._supply_source
    try:
        if not netlist._supply_explicit:
            apply_inherited_supply(netlist, tb._inherited_supply())
        blocks, _ = BehavioralEngine()._flatten(netlist, report=False)
    finally:
        netlist._supply_voltage, netlist._supply_source = saved
    return resolve_run(blocks, scope=VariationScope(
        owner_path=tb.owner_path, test_name=tb.test_name,
        dut_instance=DUT_ELEMENT_NAME))
