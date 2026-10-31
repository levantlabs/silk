"""Generate an AC testbench module from a Python specification dictionary.

The test drives an amplifier input, measures DC gain and checks it against
20 dB. This script generates the source without building or simulating a
circuit. It prints the source and validation errors for an invalid builder
name and metric type, and writes ``silk-output/ac_gain_tests.py`` (or under
``SILK_EXAMPLES_OUT``).

Generation needs neither models nor a simulator. To execute the tests, define
or import ``_build_ac_gain(pdk)`` in the generated module's namespace. The caller
supplies that builder and a PDK object. Call ``build_ac_gain_with_tests(pdk)``
to build and attach tests, then call the returned module's ``run_all_tests()``.
Executing these AC tests needs ngspice and any models used by the circuit.

Run: ``python -m silk.examples.testbench_codegen_example``.
See ``docs/codegen-contract.md`` for the specification format.
"""
import os

from silk.codegen import Generator
from silk.codegen.testbench_generator import TestbenchSpecError, generate_testbenches
from silk.examples import example_output_dir

OUT_DIR = example_output_dir(create=False)

# The spec is plain JSON-able data. ``rigs`` are shared source groups; each
# test names the rigs it applies plus its own sources, analyses, metrics and
# specs. Source entries are the same ``SourceSpec`` shape the circuit
# generator accepts, so a tool maps one vocabulary onto both.
AC_GAIN_SPEC = {
    "circuit_name": "ac_gain",
    "circuit_builder": "_build_ac_gain",      # the caller's builder: mod = _build_ac_gain(pdk)
    "rigs": [
        {"name": "supply",
         "sources": [{"type": "VoltageSource", "instance": "vdd",
                      "positive": "vdd", "negative": "gnd", "dc": 1.8}]},
    ],
    "tests": [
        {"name": "ac",
         "dut_portmap": {"vout": "vout", "vin": "vin", "vdd": "vdd", "vss": "gnd"},
         "applied_rigs": ["supply"],
         "sources": [{"type": "VoltageSource", "instance": "vin",
                      "positive": "vin", "negative": "gnd", "dc": 0.9, "ac": 1.0}],
         "analyses": [{"kind": "ac", "variation": "dec", "points": 20,
                       "fstart": 1.0, "fstop": 1e9}],
         # Declarative metric: ``type`` is a registered single-signal metric,
         # ``signal`` a rawfile signal name. The DUT is instanced as ``dut``,
         # so a DUT-internal node is ``xdut.<net>``; a DUT *port* aliases to
         # the testbench net it is mapped to above, so read that net.
         "metrics": [{"name": "gain_dB", "type": "dc_gain_dB",
                      "signal": "vout", "params": {}}],
         "specs": [{"name": "DC gain > 20 dB", "metric": "gain_dB",
                    "op": ">", "target": 20.0}]},
    ],
}


def generate(spec=AC_GAIN_SPEC, out_dir=OUT_DIR):
    """Generate the testbench module for *spec*; write it under *out_dir*.

    Return ``(source, path)``. The Generator method delegates to the
    module-level generator.
    """
    gen = Generator()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{spec['circuit_name']}_tests.py")
    gen.generate_testbenches_to_file(spec, path)
    with open(path) as generated:
        code = generated.read()
    return code, path


def show_builder_contract(spec=AC_GAIN_SPEC):
    """``circuit_builder`` must be a bare identifier — a dotted path is refused."""
    bad = dict(spec, circuit_builder='pkg.builders.build_ac_gain')
    try:
        generate_testbenches(bad)
    except TestbenchSpecError as err:
        return str(err)
    raise AssertionError('a dotted circuit_builder must be rejected')


def show_error_contract(spec=AC_GAIN_SPEC):
    """An unknown metric ``type`` is refused at generate time, by name."""
    bad_test = dict(spec['tests'][0])
    bad_test['metrics'] = [{"name": "vout_final", "type": "not_a_metric",
                            "signal": "vout", "params": {}}]
    bad = dict(spec, tests=[bad_test])
    try:
        generate_testbenches(bad)
    except TestbenchSpecError as err:
        return str(err)
    raise AssertionError('an unknown metric type must be rejected')


def main():
    code, path = generate()
    print('--- generated testbench module -------------------------------------')
    print(code)
    print(f'--- written to {path}')
    print()
    print('--- circuit_builder emission contract -------------------------------')
    print('circuit_builder="pkg.builders.build_ac_gain" ->')
    print('  TestbenchSpecError:', show_builder_contract())
    print()
    print('--- error contract --------------------------------------------------')
    print('metric type="not_a_metric" ->')
    print('  TestbenchSpecError:', show_error_contract())
    print()
    print('Generation needs no models or simulator. Define or import '
          '_build_ac_gain(pdk) in the generated module, call '
          'build_ac_gain_with_tests(pdk), then call run_all_tests() on its result.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
