"""Circuit-only serialization for simulation workers.

Ordinary Module pickles retain their registered benches. Workers evaluate no
metrics: omitting test registries here keeps live DUT references from dragging
local metric callables into a simulation payload.
"""
import io
import pickle


def dump_worker_netlist(module):
    from silk.circuit.modules import Module

    class CircuitPickler(pickle.Pickler):
        def reducer_override(self, obj):
            if isinstance(obj, Module):
                factory, args, state = obj.__reduce_ex__(pickle.HIGHEST_PROTOCOL)
                return factory, args, dict(state, _tests={}, simResultsAll=None)
            return NotImplemented

    output = io.BytesIO()
    CircuitPickler(output, protocol=pickle.HIGHEST_PROTOCOL).dump(module)
    return output.getvalue()


def copy_simulation_netlist(module):
    """Snapshot a corner/sample circuit without copying its bench registry.

    Unlike the worker pickle, an in-process snapshot retains behavioral
    callables. Explicit user deepcopy still copies the full bench graph.
    """
    import copy
    memo = {}
    for instance in module.iter_instances(include_self=True):
        owner = instance.module
        memo[id(owner._tests)] = {}
        for bench in owner._tests.values():
            memo[id(bench)] = None
        for spec in owner._specs:
            bench = getattr(spec, '_owning_testbench', None)
            if bench is not None:
                memo[id(bench)] = None
    return copy.deepcopy(module, memo)
