"""Private proof of device parameters actually requested by one prepared run.

The tuple snapshot travels on RawResult, including retained/shared results and
worker copies. It is never reconstructed from a reader's live circuit.
"""
from collections import defaultdict


def snapshot_requests(elem_dict, op_lines, probe_lines, analyses):
    """Freeze post-preflight ngspice requests as (plot, designer, address, param)."""
    from silk.circuit.emitted_names import emitted_instance_name
    from silk.circuit.modules_results import (
        _iter_mos_device_paths, _device_op_internal_name, ngspice_device_op_line,
    )
    from silk.circuit.spice_netlister import NGSPICE_PLOT_NAMES
    from silk.simulation import Noise
    owners = {}
    for path, designer_path, name, device in _iter_mos_device_paths(elem_dict):
        internal = _device_op_internal_name(device)
        if not internal:
            continue
        leaf = emitted_instance_name(device, name)
        stem = ngspice_device_op_line(path, leaf, internal, '').removesuffix('[]')
        owners[stem.lower()] = '.'.join(filter(None, (designer_path, name)))

    requests = []
    def add(lines, plot):
        for address in lines or ():
            stem, _, suffix = address.rpartition('[')
            requests.append((plot, owners.get(stem.lower(), stem), address, suffix.rstrip(']')))
    for analysis in (analyses or {}).values():
        # Noise emits spectral vectors, not pre-run device parameter saves.
        if isinstance(analysis, Noise):
            continue
        plot = NGSPICE_PLOT_NAMES.get(type(analysis))
        if plot is not None:
            add(probe_lines, plot)
    add(op_lines, 'op1')
    return tuple(dict.fromkeys(requests))


def warn_missing_requests(raw):
    """Warn for absent/empty requested vectors, never for their numeric values."""
    from silk.metrics import diagnostics
    requests = getattr(raw, '_device_param_requests', ())
    if not requests:
        return
    from silk.signals.analysis_kind import plot_silk_analysis
    plots = {}
    for plot in getattr(raw.native, 'plots', (raw.native,)):
        # Raw plots carry descriptive names, not nutmeg's numbered identifiers.
        kind = plot_silk_analysis(plot)
        plots[kind] = plot
    kinds = {'op1': 'op', 'tran1': 'tran', 'ac1': 'ac', 'dc1': 'dc',
             'noise1': 'noise'}
    by_plot = {}
    missing = defaultdict(list)
    for plot_name, designer, address, param in requests:
        if plot_name not in by_plot:
            plot = plots.get(kinds.get(plot_name, plot_name))
            names = {}
            for vector in (() if plot is None else plot.vectors):
                name = vector.name.lower()
                if name.startswith(('i(', 'v(')) and name.endswith(')'):
                    name = name[2:-1]
                names[name] = len(vector.data)
            by_plot[plot_name] = names
        if not by_plot[plot_name].get(address.lower(), 0):
            missing[(plot_name, designer)].append((param, address))
    for (plot, designer), absent in missing.items():
        params = ', '.join(param for param, _ in absent)
        addresses = ', '.join(address for _, address in absent)
        diagnostics.emit(
            metric='device_parameters', code='op_parameter_missing',
            message=f"Instance {designer!r}, plot {plot!r}: requested parameters "
                    f"{params} are absent or empty in this run's rawfile ({addresses}).",
            details={'device': designer, 'plot': plot, 'parameters': params,
                     'addresses': addresses},
            suggestion='Inspect this run\'s simulator log and emitted save addresses; '
                       'missing data is not an operating-point measurement.')
