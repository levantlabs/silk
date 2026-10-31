"""Overwrite, retain and reread ngspice results from a resistive load.

Run five transient simulations at 1 through 5 V and measure the final output
voltage. Overwrite mode reuses a path; new mode retains distinct directories.
Print path comparisons and disk readback values, and check that retained
rawfiles keep their contents. Artifacts default to ``silk-output/output_modes/``
(or under ``SILK_EXAMPLES_OUT``); run_example also accepts output_root.

A report retains its scalar metrics, but report.sim_results refers to the live
Module. Save results_file after each successful run to read that waveform from
disk. Overwrite mode removes the expected previous result and log files before
launch, so a failed replacement can lose them. New mode preserves earlier runs.

Needs ngspice but no installed PDK models: the deck uses only ideal elements.
Run from the repository root::

    MPLBACKEND=Agg .venv/bin/python -m silk.examples.output_modes_example
"""

from __future__ import annotations

from math import isclose
from pathlib import Path

from silk import Module, Port, load_pdk, sources
from silk.elements import elements
from silk.exceptions import SimulationError
from silk.metrics import final_value
from silk.signals import NgspiceSignal
from silk.signals.rawfile import read_rawfile
from silk.simulation import Transient
from silk.examples import example_output_dir


DEFAULT_OUTPUT_ROOT = Path(example_output_dir('output_modes', create=False))
LOAD_OHM = 1000.0
INPUT_VOLTAGE = 1.0


def build_load(*, output_root: str | Path = DEFAULT_OUTPUT_ROOT,
               resistance: float = LOAD_OHM,
               input_voltage: float = INPUT_VOLTAGE) -> Module:
    """Build an ideal load with its voltage characterization already registered.

    Parameters
    ----------
    output_root : str or pathlib.Path
        Directory for generated artifacts.
    resistance : float
        Load resistance in ohms.
    input_voltage : float
        Initial DC stimulus in volts. The embedded test measures its last
        transient sample and checks a simple positive-voltage spec.

    Returns
    -------
    Module
        Resistive load owning the ``level`` test; run with ``run_test('level')``
        or ``run_all_tests()``.
    """
    mod = Module('sky130', str(Path(output_root).resolve()), 'load.cir',
                 pdk_details=load_pdk('sky130'), module_name='load')
    mod.define_module_ports({'out': Port.analog_input(), 'vss': Port.ground()})
    mod.add(elements.resistor('load', p='out', n='vss', value=resistance))

    tb = mod.create_test('level')
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_source(sources.VoltageSource(
        'supply', positive='out', negative='0', dc=input_voltage))
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    # Measure the final output voltage from the declared probe.
    tb.add_metric('voltage', final_value, sig=tb.probe('out'))
    tb.add_spec('Positive voltage', 'voltage', '>', 0.0)
    return mod


def read_voltage(path: str | Path) -> float:
    """Read the final output voltage from a saved rawfile, freshly from disk."""
    raw = read_rawfile(path)
    signal = NgspiceSignal.from_rawfile(raw, 'out')
    return final_value(signal)


def _check_voltage(actual: float, expected: float) -> None:
    if not isclose(actual, expected, rel_tol=0, abs_tol=1e-9):
        raise SimulationError(f'Expected {expected:g} V, read {actual:g} V')


def run_example(*, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Path]:
    """Run five simulations and verify overwrite, retention and disk readback.

    Parameters
    ----------
    output_root : str or pathlib.Path
        Output directory, reusable across invocations without cleanup.

    Returns
    -------
    dict[str, pathlib.Path]
        Actual successful paths captured at each launch: ``overwrite_first``,
        ``overwrite_second``, ``new_first``, ``new_second``, and ``restored``.
        The three overwrite paths alias one file, containing 5 V on return;
        the two retained files contain 3 V and 4 V respectively.
    """
    mod = build_load(output_root=output_root, input_voltage=1.0)

    mod.set_output_options(mode='overwrite')

    # 1. Explicit overwrite: capture paths before the live Module changes.
    first = mod.run_test('level')
    first_path = Path(first.sim_results.results_file)
    _check_voltage(first.metrics['voltage'], 1.0)
    _check_voltage(read_voltage(first_path), 1.0)

    # Keep this source handle only to edit the live stimulus, not as evidence.
    supply = first.sim_results.sig_sources_dict['supply']
    supply.dc = 2.0
    second = mod.run_test('level')
    second_path = Path(second.sim_results.results_file)
    _check_voltage(second.metrics['voltage'], 2.0)
    if first_path != second_path:
        raise SimulationError('Overwrite did not reuse its successful path')
    overwritten = read_voltage(first_path)
    _check_voltage(overwritten, 2.0)
    print(f'Overwrite: 1 V then 2 V; same path: {first_path == second_path}')
    print(f'  Fresh disk read through the first path: {overwritten:g} V')
    print(f'  First scalar report: {first.metrics["voltage"]:g} V; '
          f'its live Module: {first.sim_results.load_signal("out").y[-1]:g} V')

    # 2. The test already exists; its owning circuit supplies the new policy.
    mod.set_output_options(mode='new')
    supply.dc = 3.0
    third = mod.run_test('level')
    third_path = Path(third.sim_results.results_file)
    _check_voltage(third.metrics['voltage'], 3.0)
    third_bytes = third_path.read_bytes()

    supply.dc = 4.0
    fourth = mod.run_test('level')
    fourth_path = Path(fourth.sim_results.results_file)
    _check_voltage(fourth.metrics['voltage'], 4.0)
    fourth_bytes = fourth_path.read_bytes()
    if (third_path.parent == fourth_path.parent
            or third_path.parent.parent != first_path.parent
            or fourth_path.parent.parent != first_path.parent):
        raise SimulationError('New-mode results are not distinct base children')

    # Both reads happen after both launches, independently of any live report.
    retained = read_voltage(third_path), read_voltage(fourth_path)
    _check_voltage(retained[0], 3.0)
    _check_voltage(retained[1], 4.0)
    print('New: distinct retained directories; '
          f'fresh disk reads: {retained[0]:g} V and {retained[1]:g} V')

    # 3. Return to the stable base; the earlier children remain intact.
    mod.set_output_options(mode='overwrite')
    supply.dc = 5.0
    fifth = mod.run_test('level')
    fifth_path = Path(fifth.sim_results.results_file)
    _check_voltage(fifth.metrics['voltage'], 5.0)
    if fifth_path != first_path:
        raise SimulationError('Switching to overwrite did not return to the base')
    _check_voltage(read_voltage(fifth_path), 5.0)
    for path, expected, saved_bytes in ((third_path, 3.0, third_bytes),
                                        (fourth_path, 4.0, fourth_bytes)):
        _check_voltage(read_voltage(path), expected)
        if path.read_bytes() != saved_bytes:
            raise SimulationError(f'Retained rawfile changed: {path}')
    print('Back to overwrite: original path now reads 5 V; '
          'retained files still read 3 V and 4 V, byte-for-byte unchanged')

    for path in (fifth_path, third_path, fourth_path):
        print(f'  Result: {path}')
        names = sorted(p.name for p in path.parent.iterdir() if p.is_file())
        print(f'  Artifacts beside it: {", ".join(names)}')
    return {'overwrite_first': first_path, 'overwrite_second': second_path,
            'new_first': third_path, 'new_second': fourth_path,
            'restored': fifth_path}


if __name__ == '__main__':
    run_example()
