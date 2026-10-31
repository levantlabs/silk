"""Subprocess driver for ``tests/test_examples_hierarchy_audit.py`` (TASK-126).

Runs ONE shipped example with SPICE stubbed, records every ``Module`` it
builds, and audits structure extraction over each of them. Written as a
plain script (no pytest import) so the test can run it in a fresh
interpreter per example: examples mutate global state (PDK cache, matplotlib,
``sys.argv``) and one example's crash must never be attributed to another.

Usage::

    python tests/_examples_hierarchy_driver.py <example.py> <report.json>

Exit status 0 means the driver ran to completion and wrote the report; the
verdict is in the JSON, and every non-stub exception is in it verbatim.
Nothing here swallows an exception: a stopped pass is recorded with its
type and traceback, and the audit itself runs with no ``try`` around it.
"""
from __future__ import annotations

import importlib
import inspect
import io
import json
import os
import runpy
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr


class SimulationStubbed(BaseException):
    """Raised by every stubbed simulator entry point.

    A ``BaseException`` on purpose: several examples wrap a testbench run in
    ``except Exception`` and carry on with a results dict, which under a dead
    simulator ends in an unrelated ``AttributeError`` several calls later.
    Nothing in an example catches ``BaseException``, so a pass stops exactly
    at its first simulation and the report can say so by type.
    """


class SimulationStubbedInWorker(Exception):
    """The stub as raised inside a ``multiprocessing`` pool worker.

    A pool worker only catches ``Exception``: a ``BaseException`` kills the
    worker, the pool silently replaces it, the task is lost and the parent
    waits forever (the fault-injection example hung the audit that way). An
    ordinary exception is pickled back to the parent, where the driver still
    recognises it as the stub. A campaign that records per-run errors by
    design (``FaultCampaign``) then completes with every run errored and no
    simulation run; that is reported as a completed pass, which is exactly
    right for an audit of module builds.
    """


_MAIN_PID = os.getpid()
_STUB_TEXT = 'SPICE is stubbed in the hierarchy audit'


def _stub(*_args, **_kwargs):
    if os.getpid() != _MAIN_PID:
        raise SimulationStubbedInWorker(_STUB_TEXT)
    raise SimulationStubbed(_STUB_TEXT)


def _is_stub(exc: BaseException) -> bool:
    # A worker's stub can come back as text only: run_all_tests folds a
    # test whose every run failed into {'error': message}, and an example
    # that then raises on that entry carries the stub's message, not the
    # stub exception itself.
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, (SimulationStubbed, SimulationStubbedInWorker)):
            return True
        if _STUB_TEXT in str(exc):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


def install_stubs():
    from silk.circuit.modules import Module
    from silk.simulation.backend import SimulatorBackend
    from silk.simulation import _ngspice_version
    from silk.simulation.runner import NgspiceRunner
    from silk.simulation.xyce_runner import XyceRunner

    # This structural audit targets the measured release, without probing a binary.
    _ngspice_version.ngspice_version = lambda: '42'

    Module.run_simulator = _stub
    SimulatorBackend.run = _stub
    NgspiceRunner.run = _stub
    XyceRunner.run = _stub


BUILT: list = []


def install_recorder():
    from silk.circuit.modules import Module
    original = Module.__init__

    def recording_init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        BUILT.append(self)

    Module.__init__ = recording_init


def _outcome_of(exc: BaseException) -> dict:
    if _is_stub(exc):
        return {'outcome': 'stopped_at_first_simulation'}
    if isinstance(exc, SystemExit):
        # Recorded by code, judged by the test: 0 is a completed run.
        return {'outcome': 'exit', 'code': exc.code,
                'traceback': ''.join(traceback.format_exception(exc))}
    return {'outcome': 'error', 'type': type(exc).__name__, 'detail': str(exc),
            'traceback': ''.join(traceback.format_exception(exc))}


def run_as_main(path: str) -> dict:
    """Pass 1: execute the example as ``__main__``."""
    saved_argv = sys.argv
    sys.argv = [path]
    try:
        runpy.run_path(path, run_name='__main__')
    except (Exception, SimulationStubbed, SystemExit) as exc:
        return _outcome_of(exc)
    finally:
        sys.argv = saved_argv
    return {'outcome': 'completed'}


def _synthesize(name: str, builder: str, pdk_cache: dict, out_dir: str):
    """Value for a required builder parameter, by name; None if unknowable."""
    if name in ('pdk', 'pdk_inst', 'pdk130'):
        from silk import load_pdk
        return pdk_cache.setdefault('sky130', load_pdk('sky130'))
    if name in ('out_dir', 'dir_folder', 'work_dir', 'outdir'):
        return out_dir
    if name in ('fname', 'netlist_name'):
        return f'{builder}.cir'
    if name == 'n':
        return 4
    return None


def run_builders(stem: str, out_dir: str) -> list:
    """Pass 2: call every module-level ``build_*``/``make_*`` builder."""
    module = importlib.import_module(f'silk.examples.{stem}')
    pdk_cache: dict = {}
    results = []
    for name, func in sorted(vars(module).items()):
        if not (name.startswith('build_') or name.startswith('make_')):
            continue
        if not inspect.isfunction(func) or func.__module__ != module.__name__:
            continue
        kwargs, missing = {}, []
        for param in inspect.signature(func).parameters.values():
            if param.default is not inspect.Parameter.empty:
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            value = _synthesize(param.name, name, pdk_cache, out_dir)
            if value is None:
                missing.append(param.name)
            else:
                kwargs[param.name] = value
        if missing:
            results.append({'builder': name, 'outcome': 'skipped',
                            'detail': f'needs live argument(s) {missing}'})
            continue
        before = len(BUILT)
        try:
            func(**kwargs)
        except (Exception, SimulationStubbed, SystemExit) as exc:
            record = _outcome_of(exc)
        else:
            record = {'outcome': 'completed'}
        record.update(builder=name, modules_built=len(BUILT) - before)
        results.append(record)
    return results


def _children(module):
    """Direct child Module instances: elem_dict entries whose device IS a
    Module, independent of the entry's type tag."""
    from silk.circuit.modules import Module
    out = {}
    for name, entry in module.elem_dict.items():
        device = entry.get('device') if isinstance(entry, dict) else entry
        if isinstance(device, Module):
            out[name] = device
    return out


def _expected_instances(module, prefix=''):
    """Every child instance at every depth: path -> {port: parent net}."""
    expected = {}
    for name, child in _children(module).items():
        path = f'{prefix}.{name}' if prefix else name
        pins = module.elem_dict[name].get('pin_list', {})
        expected[path] = {port: pin.net for port, pin in pins.items()}
        expected.update(_expected_instances(child, path))
    return expected


def audit_module(module) -> dict:
    """Compare both extractors against the module tree. No try/except: an
    extractor failure propagates to the driver's top level and is reported
    as a driver error, which fails the test."""
    from silk.db.structure_tracker import CircuitStructureTracker
    from silk.exceptions import CircuitError
    tracker = CircuitStructureTracker(None, None)
    expected = _expected_instances(module)
    problems = []
    summary = {
        'name': getattr(module, 'module_name', None) or getattr(module, 'circuit_name', None)
        or getattr(module, 'netlistname', '?'),
        'class': type(module).__name__,
        'direct_children': len(_children(module)),
        'instances_all_depths': len(expected),
    }

    # A child instance the example left with an unbound port (the lint
    # coverage example builds one ON PURPOSE) cannot yield a complete record;
    # TASK-127's contract there is a loud CircuitError naming the instance,
    # never a silent None. Assert that contract, and report the instance.
    unbound = sorted(path for path, ports in expected.items()
                     if any(not isinstance(net, str) or not net for net in ports.values()))
    if unbound:
        summary['unbound_by_example'] = unbound
        for extractor in ('extract_structure', '_extract_module_structure'):
            try:
                getattr(tracker, extractor)(module)
            except CircuitError as exc:
                if not any(path in str(exc) for path in unbound):
                    problems.append(f'{extractor} raised without naming an unbound '
                                    f'instance {unbound}: {exc}')
            else:
                problems.append(f'{extractor} silently accepted unbound port(s) on '
                                f'{unbound}')
        summary.update(records=None, ok=not problems, problems=problems)
        return summary

    hierarchy = tracker.extract_structure(module)['hierarchy']
    actual = {record['instance_name']: record['ports'] for record in hierarchy}
    if len(hierarchy) != len(expected):
        problems.append(f'{len(expected)} child instance(s) at every depth but '
                        f'{len(hierarchy)} hierarchy record(s)')
    if len(actual) != len(hierarchy):
        problems.append('duplicate instance paths in hierarchy records')
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        differing = sorted(p for p in set(expected) & set(actual)
                           if expected[p] != actual[p])
        problems.append(f'hierarchy port maps differ: missing={missing} '
                        f'extra={extra} differing={differing}')
    for path, ports in actual.items():
        if not ports:
            problems.append(f'{path}: empty port map')
        for port, net in ports.items():
            if not isinstance(net, str) or not net:
                problems.append(f'{path}: port {port!r} bound to {net!r}')

    def walk(parent, prefix=''):
        payload = tracker._extract_module_structure(parent)
        children = _children(parent)
        if set(payload['submodules']) != set(children):
            problems.append(f'definition payload of {prefix or "<root>"}: submodules '
                            f'{sorted(payload["submodules"])} != children '
                            f'{sorted(children)}')
        for name, child in children.items():
            path = f'{prefix}.{name}' if prefix else name
            record = payload['submodules'].get(name)
            if record is not None and record['ports'] != expected[path]:
                problems.append(f'definition payload of {path}: ports {record["ports"]} '
                                f'!= bindings {expected[path]}')
            walk(child, path)

    walk(module)
    summary.update(records=len(hierarchy), ok=not problems, problems=problems)
    return summary


def main(path: str, report_path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(os.getcwd(), 'builder_out')
    os.makedirs(out_dir, exist_ok=True)
    report = {'example': os.path.basename(path), 'driver_error': None}
    log = io.StringIO()
    try:
        install_stubs()
        install_recorder()
        with redirect_stdout(log), redirect_stderr(log):
            report['pass1_main'] = run_as_main(path)
            report['pass2_builders'] = run_builders(stem, out_dir)
        seen, modules = set(), []
        for module in BUILT:
            if id(module) in seen:
                continue
            seen.add(id(module))
            modules.append(audit_module(module))
        report['modules'] = modules
    except BaseException as exc:  # reported verbatim, never swallowed
        report['driver_error'] = ''.join(traceback.format_exception(exc))
    report['log_tail'] = log.getvalue()[-4000:]
    with open(report_path, 'w') as fh:
        json.dump(report, fh, indent=1)
    return 0 if report['driver_error'] is None else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
