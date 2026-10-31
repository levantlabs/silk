"""Run-local callable state and attachment snapshots for placed models."""
from copy import deepcopy
from functools import partial
from types import FunctionType, MethodType, ModuleType

from silk.exceptions import SimulationError


def _cell(value):
    return (lambda: value).__closure__[0]


def _clone(fn, memo=None):
    memo = {} if memo is None else memo
    if id(fn) in memo:
        return memo[id(fn)]
    if isinstance(fn, ModuleType):
        return fn
    if isinstance(fn, FunctionType):
        # Register every cell before cloning captured helpers. Functions within
        # one placement must retain their original shared nonlocal bindings.
        pending = []
        cells = []
        for source in fn.__closure__ or ():
            if id(source) not in memo:
                memo[id(source)] = _cell(None)
                pending.append(source)
            cells.append(memo[id(source)])
        cells = tuple(cells)
        clone = FunctionType(fn.__code__, fn.__globals__, fn.__name__,
                             fn.__defaults__, cells or None)
        memo[id(fn)] = clone
        for source in pending:
            target = memo[id(source)]
            try:
                value = source.cell_contents
            except ValueError:
                del target.cell_contents
                continue
            target.cell_contents = _clone(value, memo)
        clone.__defaults__ = _clone(fn.__defaults__, memo)
        clone.__kwdefaults__ = _clone(fn.__kwdefaults__, memo)
        clone.__dict__.update(deepcopy(fn.__dict__, memo))
        return clone
    if isinstance(fn, partial):
        clone = partial(fn.func)
        memo[id(fn)] = clone
        clone.__setstate__((_clone(fn.func, memo), _clone(fn.args, memo),
                            _clone(fn.keywords, memo), _clone(fn.__dict__, memo)))
        return clone
    if isinstance(fn, MethodType):
        return MethodType(fn.__func__, deepcopy(fn.__self__, memo))
    if type(fn) is list:
        clone = []
        memo[id(fn)] = clone
        clone.extend(_clone(value, memo) for value in fn)
        return clone
    if type(fn) is dict:
        clone = {}
        memo[id(fn)] = clone
        for key, value in fn.items():
            clone[_clone(key, memo)] = _clone(value, memo)
        return clone
    if type(fn) is tuple:
        values = [_clone(value, memo) for value in fn]
        # A mutable child can lead back to this tuple and finish its copy first.
        if id(fn) not in memo:
            memo[id(fn)] = tuple(values)
        return memo[id(fn)]
    return deepcopy(fn, memo)


def attachments(blocks, resolution):
    """Freeze form/contract/callable, with independent state for ports placements.

    Every run copies each placement's callable, closure cells and captured
    state. Writes to captured Python objects affect the copy; module-level
    globals are not reset. Live parameters remain separate
    from callable state and keep the existing per-path resolution.
    """
    modeled = [b for b in blocks.values() if hasattr(b.device, '_behavioral_model')]
    result = {}
    for block in modeled:
        device = block.device
        fn = device._behavioral_model
        try:
            fn = _clone(fn)
        except Exception as exc:
            raise SimulationError(
                f"Model {block.path!r}: cannot copy its callable state for this "
                "placement; use copyable model state and explicit parameters.") from exc
        params = resolution.params.get(block.path, device._behavioral_params)
        result[block.path] = fn, 'ports', device._behavioral_execution, params
    return result
