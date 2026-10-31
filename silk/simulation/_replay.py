"""Private, in-memory identity of the managed inputs to a Monte Carlo run.

Preparation and replay use the same effective configuration. Backend traversal
supplies executed sources and backend facts; freezing and comparison happen here
once. Results, generated files and metric/spec evaluation are not inputs here.
"""
from copy import copy
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from shutil import which
from urllib.parse import quote

from silk.circuit._behavioral_identity import _dump, _value_key
from silk.exceptions import CircuitError, MonteCarloError
from silk.utils._utils import _resolve_value


_UNSET = object()


def apply_effective_settings(tb, netlist, *, temperature=_UNSET, supply=_UNSET):
    """Resolve the settings preparation consumes, onto *netlist*.

    Signature construction passes a shallow view with its own option mapping;
    preparation passes the live netlist. This does not apply source voltages.
    ``temperature`` and ``supply`` give the inherited settings directly, for
    callers (the read-only setup check) that must not enter a placement scope
    to have the bench look them up.
    """
    from silk.simulation.conditions import (
        apply_inherited_simulator_options, apply_inherited_supply,
        apply_inherited_temperature, own_simulator_options_setting,
    )
    if not netlist._simulator_explicit:
        netlist._simulator = tb.module.simulator
    if not netlist._temperature_explicit:
        apply_inherited_temperature(
            netlist, tb._inherited_temperature() if temperature is _UNSET else temperature)
    if not netlist._supply_explicit:
        apply_inherited_supply(
            netlist, tb._inherited_supply() if supply is _UNSET else supply)
    apply_inherited_simulator_options(
        netlist, own_simulator_options_setting(tb.module))


def _plain(value):
    """Resolve explicit values without evaluating deferred waveform probes."""
    from silk.signals import Signal
    if isinstance(value, Signal):
        data = vars(value)
        bench = data.get('_bench')
        owner = bench() if bench is not None else None
        return ('probe', data.get('name'), data.get('_selector'),
                data.get('_analysis'), getattr(owner, 'owner_path', None),
                getattr(owner, 'test_name', None),
                _plain(data.get('_recipe')), _plain(data.get('_underlying_signals')))
    # Sources, analyses and waveform objects store authored values. Keep the
    # resolved numeric precision as well as emitted text: behavioral execution
    # does not round through SPICE formatting. Never traverse source ownership.
    if callable(getattr(value, 'to_spice', None)):
        return (type(value).__module__, type(value).__qualname__,
                _plain({key: item for key, item in vars(value).items()
                        if key != '_module'}))
    if isinstance(value, Enum):
        return (type(value).__qualname__, value.value)
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {_plain(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain(v) for v in value)
    return _resolve_value(value)


@dataclass(frozen=True)
class ExperimentSignature:
    items: tuple
    invalid: tuple = ()

    def require_valid(self):
        if self.invalid:
            raise MonteCarloError(
                'Cannot record Monte Carlo experiment: ' + '; '.join(self.invalid))


class _Inputs:
    """Freeze labelled fields independently so one invalid edit hides none."""
    def __init__(self):
        self.items = {}
        self.invalid = []

    def add(self, label, read):
        if label in self.items:
            raise MonteCarloError(f'Duplicate replay input {label!r}; input paths must be unique.')
        try:
            self.items[label] = _dump(_value_key(_plain(read())))
        except (CircuitError, ValueError, TypeError, AttributeError, KeyError,
                RecursionError) as exc:
            detail = f'{label}: {exc}'
            self.invalid.append(detail)
            self.items[label] = ('invalid', detail)

    def signature(self):
        return ExperimentSignature(tuple(self.items.items()), tuple(self.invalid))


def changed_inputs(recorded, current):
    """Every changed/invalid field, including additions and removals."""
    missing = object()
    before, after = dict(recorded.items), dict(current.items)
    changed = [label for label in dict.fromkeys((*before, *after))
               if before.get(label, missing) != after.get(label, missing)]
    changed.extend(current.invalid)
    return tuple(dict.fromkeys(changed))


def _spice_inputs(tb, netlist, inputs):
    """SPICE structure plus facts the structural digest intentionally omits."""
    from silk import settings
    setting = {'ngspice': 'SILK_NGSPICE_BIN', 'xyce': 'SILK_XYCE_BIN'}[netlist.simulator]
    executable = settings.get(setting)
    inputs.add('simulator executable', lambda: (executable, which(executable)))
    inputs.add('SPICE structure', lambda: tb._structure_hash(_netlist=netlist))
    # These are propagated from the owner by _prepare_run, and can change
    # electrical topology (ammeter wrappers), analyses and saved evidence.
    for name in ('save_device_currents', 'terminal_current_method', 'save_device_params',
                 'omit_device_internal_nodes'):
        inputs.add(f'capture {name}', lambda: getattr(tb.module, name))
    inputs.add('capture save_list', lambda: getattr(netlist, '_save_list', None))
    sources = []

    def pdk_inputs(pdk, path):
        for name in ('library_location', 'extra_includes', 'corners',
                     'default_corner', 'statistical', 'user_units',
                     'xyce_library_location', 'xyce_extra_includes', 'osdi_files'):
            inputs.add(f'PDK {path} {name}', lambda: getattr(pdk, name, None))

    # Planning and the both-variation wrapper use the bench's PDK directly;
    # it need not still be the same object as the netlist's or the DUT's PDK.
    pdk_inputs(tb.pdk_details, 'testbench')

    def visit(module, path):
        # Reuse the emitter's complete device dispatch on a clean instance,
        # without sample/ammeter state left on the live netlister by prior runs.
        emitter = type(module.netlister)(module.pdk_details)
        inputs.add(f'formal ports {path}', lambda: (
            None if module.pin_list is None else
            tuple((name, pin.net, pin.direction)
                  for name, pin in module.pin_list.items())))
        inputs.add(f'mismatch {path}', lambda: module.mismatch_enabled)
        inputs.add(f'custom text {path}', lambda: module.custom_lines)
        inputs.add(f'device order {path}', lambda: tuple(module.elem_dict))
        inputs.add(f'source order {path}', lambda: tuple(module.sig_sources_dict))
        for name, source in module.sig_sources_dict.items():
            sources.append((f'{path}/{quote(name, safe="")}', source))
        for name, passive in module.passives_dict.items():
            inputs.add(f'passive {path}/{quote(name, safe="")}', passive.to_spice)
        for name, entry in module.elem_dict.items():
            device = entry['device']
            child = f'{path}/{quote(name, safe="")}'
            if hasattr(device, 'elem_dict'):
                inputs.add(f'instance {child}', lambda: (
                    device.module_name,
                    [(p, pin.net, pin.direction)
                     for p, pin in (entry.get('pin_list', device.pin_list) or {}).items()]))
                visit(device, child)
            else:
                inputs.add(f'device {child}', lambda: (
                    emitter.generate_device_instances({name: entry})))
        pdk_inputs(module.pdk_details, path)

    visit(netlist, '$')
    return sources


def _behavioral_inputs(tb, netlist, inputs):
    """Executed models, resolved declarations and their effective draw inputs."""
    from silk.circuit._behavioral_identity import _own_model_key
    from silk.simulation._behavioral_elements import AlgebraicModel
    from silk.simulation.behavioral import BehavioralEngine
    from silk.circuit.module_test import DUT_ELEMENT_NAME
    from silk.simulation._behavioral_variation import VariationScope, resolve_run
    from silk.simulation.variation import DRAW_SCHEME

    blocks, sources = BehavioralEngine()._flatten(netlist, report=False)
    resolution = resolve_run(blocks, scope=VariationScope(
        owner_path=tb.owner_path, test_name=tb.test_name,
        dut_instance=DUT_ELEMENT_NAME))
    for record in resolution.records:
        inputs.add(f'declaration {record.display}:{record.param}', lambda: (
            record.resolved.kind, record.resolved.fields, record.key, record.param))
    inputs.add('draw scheme', lambda: DRAW_SCHEME)
    for key, block in blocks.items():
        inputs.add(f'mismatch {key!r}', lambda: block.mismatch)
        inputs.add(f'model bindings {key!r}', lambda: block.pins)
        if isinstance(block.device, AlgebraicModel):
            inputs.add(f'model {key!r}', lambda: block.device)
        else:
            inputs.add(f'model {key!r}', lambda: _own_model_key(block.device))
    for path, _, positive, negative in sources:
        inputs.add(f'source bindings $/{quote(path, safe="")}', lambda: (positive, negative))
    return [(f'$/{quote(path, safe="")}', source) for path, source, _, _ in sources]


def experiment_signature(tb, settings, config, *, backend_items=None, _point=False):
    """Project a managed experiment without preparing a run or writing files.

    A backend contributor takes ``(tb, effective_netlist, inputs)`` and returns
    ``(path, source)`` pairs for common source serialization. It adds only its
    backend facts with ``inputs.add(label, reader)``. Both backends share this
    signature and its comparator.
    """
    from silk.simulation.conditions import prepare_supply
    if not _point and (tb._variable_lists or tb._groups):
        from silk.circuit.module_test import _planned_conditions
        inputs = _Inputs()
        inputs.add('Variable lists', lambda: [(var.name, values) for var, values in tb._variable_lists])
        inputs.add('groups', lambda: [[m if isinstance(m, str) else m.name for m in g] for g in tb._groups])
        points = tb._plan_runs(corner=config.get('corner'), _deterministic=True) or ()
        for i, run in enumerate(points):
            with _planned_conditions(tb, run):
                signature = experiment_signature(tb, settings, config, backend_items=backend_items, _point=True)
                inputs.add(f'point {i} coordinates', lambda: run.key)
                for label, value in signature.items:
                    inputs.items[f'point {i} {label}'] = value
                inputs.invalid.extend(f'point {i}: {reason}' for reason in signature.invalid)
        return inputs.signature()
    inputs = _Inputs()
    inputs.add('netlist', lambda: tb.netlist is not None)
    if tb.netlist is None:
        inputs.invalid.append('netlist: the recorded testbench has no netlist')
        return inputs.signature()
    netlist = copy(tb.netlist)
    netlist.simulator_options = dict(tb.netlist.simulator_options)
    apply_effective_settings(tb, netlist)
    inputs.add('placement', lambda: (tb.owner_path, tb.test_name))
    inputs.add('backend', lambda: netlist.simulator)
    inputs.add('temperature', lambda: (netlist.simulator_options.get('temp'),
                                      netlist._temperature_basis))
    inputs.add('supply', lambda: (netlist._supply_voltage, netlist._supply_source))
    for name, value in netlist.simulator_options.items():
        if name != 'temp':
            inputs.add(f'simulator option {name}', lambda: value)
    inputs.add('condition axes', lambda: tb._condition_axes())
    inputs.add('corner declaration', lambda: (tb.corners, tb.corner, tb._default_corner()))
    for field in ('temperature', 'voltage'):
        inputs.add(f'run {field}', lambda: config.get(field))
    inputs.add('analysis order', lambda: tuple(netlist.sim_analysis_dict))
    for name, analysis in netlist.sim_analysis_dict.items():
        inputs.add(f'analysis {name}', lambda: (analysis.to_spice(), analysis))
    for prefix, declaration in (('Monte Carlo request', settings),
                                ('Monte Carlo declaration', tb.monte_carlo)):
        inputs.add(prefix, lambda: declaration is not None)
        if declaration is not None:
            for field in fields(declaration):
                if field.name in ('samples_per_process', 'output_mode'):
                    continue    # execution-only: results are identical either way
                inputs.add(f'{prefix} {field.name}',
                           lambda: getattr(declaration, field.name))
    supply = None

    def resolve_supply():
        nonlocal supply
        supply = prepare_supply(netlist, apply=False)
        return supply.supply_identity

    inputs.add('effective supply', resolve_supply)
    if supply is not None and supply.voltage is not None:
        # Use the same effective source value for structure and the existing
        # source guard. Preparation writes it later; signature capture must not
        # mutate the authored source or mistake that write for input drift.
        netlist.sig_sources_dict = dict(netlist.sig_sources_dict)
        effective = copy(netlist.sig_sources_dict[supply.source])
        effective.dc = supply.voltage
        netlist.sig_sources_dict[supply.source] = effective
    contributor = backend_items or (
        _behavioral_inputs if netlist.simulator == 'behavioral' else _spice_inputs)
    try:
        sources = contributor(tb, netlist, inputs)
    except (CircuitError, ValueError, TypeError, AttributeError, KeyError,
            RecursionError) as exc:
        inputs.invalid.append(f'backend inputs: {exc}')
        sources = []
    for path, source in sources:
        effective = copy(source)
        if supply is not None and path == f'$/{quote(supply.source or "", safe="")}' and supply.voltage is not None:
            effective.dc = supply.voltage
        inputs.add(f'source {path}', lambda: (effective.to_spice(), effective))
    return inputs.signature()
