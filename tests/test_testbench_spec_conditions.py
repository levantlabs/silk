"""TASK-371: per-test ``corner`` / ``temperature_c`` and the top-level
``save_device_params`` key of a TestbenchSpec, plus the public
``ModuleTestbench.set_corner`` setter they emit to.

Corner precedence for a test run, highest first (documented on
``ModuleTestbench.set_corner`` and in docs/codegen-contract.md):

1. a corner passed to the call (``tb.run(corner=)``, ``run_all_tests(corner=)``);
2. the test's own corner (``tb.set_corner`` / the spec's test ``corner``);
3. the PDK's ``default_corner``.

The CircuitSpec top-level ``corner`` is not in that list: it only reaches the
standalone script's ``ckt.simulate(corner=...)``. A subcircuit DUT generated
from a CircuitSpec carries no corner, so it never reaches a testbench run.
"""
import os
import copy
import re

import pytest

from silk.codegen.testbench_generator import (
    TestbenchSpec, TestbenchSpecError, generate_testbenches,
)
from silk.exceptions import ConfigurationError, UnknownCornerError


def _spec(save_device_params=None, **test_over):
    t = {"name": "op", "dut_portmap": {"d": "vd", "g": "vg", "s": "0", "b": "0"},
         "applied_rigs": [], "sources": [], "analyses": [{"kind": "op"}],
         "metrics": [], "specs": []}
    t.update(test_over)
    d = {"circuit_name": "nfet", "circuit_builder": "_build_nfet",
         "rigs": [], "tests": [t]}
    if save_device_params is not None:
        d["save_device_params"] = save_device_params
    return d


# --- emission ----------------------------------------------------------------

def test_all_three_keys_emit_plain_public_calls():
    code = generate_testbenches(_spec(save_device_params=True,
                                      corner='ff', temperature_c=85))
    assert code.splitlines()[:6] == [
        "from silk.simulation import analysis",
        "",
        "def _attach_op(mod):",
        "    mod.save_device_params = True",
        "    tb = mod.create_test('op')",
        "    tb.add_module_under_test({'d': 'vd', 'g': 'vg', 's': '0', 'b': '0'})",
    ]
    assert "    tb.set_corner('ff')\n    tb.set_temperature(85.0)\n" in code
    compile(code, '<generated>', 'exec')


def test_save_device_params_is_emitted_in_every_attach_function():
    d = _spec(save_device_params=True)
    second = copy.deepcopy(d['tests'][0])
    second['name'] = 'op2'
    d['tests'].append(second)
    code = generate_testbenches(d)
    assert code.count("    mod.save_device_params = True\n") == 2


def test_float_and_negative_temperatures_emit_as_floats():
    assert "tb.set_temperature(-40.0)" in generate_testbenches(_spec(temperature_c=-40))
    assert "tb.set_temperature(27.5)" in generate_testbenches(_spec(temperature_c=27.5))


def test_numpy_temperature_does_not_leak_a_numpy_repr():
    np = pytest.importorskip('numpy')
    code = generate_testbenches(_spec(temperature_c=np.float64(85.0)))
    assert "tb.set_temperature(85.0)" in code


def test_absent_none_and_false_keys_are_byte_identical():
    base = generate_testbenches(_spec())
    assert generate_testbenches(_spec(save_device_params=False)) == base
    assert generate_testbenches(_spec(corner=None, temperature_c=None)) == base
    for word in ('save_device_params', 'set_corner', 'set_temperature'):
        assert word not in base


def test_from_dict_reads_save_device_params():
    assert TestbenchSpec.from_dict(_spec()).save_device_params is False
    assert TestbenchSpec.from_dict(_spec(True)).save_device_params is True


# --- validation --------------------------------------------------------------

def test_null_save_device_params_reads_as_absent():
    d = _spec()
    d['save_device_params'] = None
    assert generate_testbenches(d) == generate_testbenches(_spec())


@pytest.mark.parametrize('bad', [1, 0, 'true', 0.0])
def test_non_boolean_save_device_params_is_refused(bad):
    d = _spec()
    d['save_device_params'] = bad
    with pytest.raises(TestbenchSpecError, match='save_device_params'):
        generate_testbenches(d)


@pytest.mark.parametrize('bad', ['85', True, False, float('nan'), float('inf'),
                                 -float('inf'), -273.15, -300, [85]])
def test_bad_temperature_c_is_refused(bad):
    with pytest.raises(TestbenchSpecError, match=r"test 'op': temperature_c"):
        generate_testbenches(_spec(temperature_c=bad))


@pytest.mark.parametrize('bad', ['', 'f f', "ff'); evil()", 3, True])
def test_malformed_corner_is_refused_at_generation(bad):
    with pytest.raises(TestbenchSpecError, match=r"test 'op': corner"):
        generate_testbenches(_spec(corner=bad))


# --- ModuleTestbench.set_corner ----------------------------------------------

@pytest.fixture
def sky130_dut(tmp_path):
    """A sky130 DUT built from the packaged catalog: no model library or
    simulator is needed to construct it or to resolve corners."""
    from silk import load_pdk
    from silk.circuit.circuit_elements import PortDir
    from silk.circuit.modules import Module
    pdk = load_pdk('sky130')

    def build(_pdk=None):
        mod = Module('PDK_skywater130', str(tmp_path), 'nfet.cir',
                     pdk_details=pdk, module_name='nfet')
        mod.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN,
                                 's': PortDir.GROUND, 'b': PortDir.GROUND})
        mod.add(pdk.nmos1p8('M1', d='d', g='g', s='s', b='b', w=1.0, l=0.15))
        return mod
    return pdk, build


def _bench(build):
    from silk import sources
    from silk.simulation import analysis
    mod = build()
    tb = mod.create_test('op')
    tb.add_module_under_test({'d': 'vd', 'g': 'vg', 's': '0', 'b': '0'})
    tb.add(sources.vdc('vd', positive='vd', negative='0', value=1.0))
    tb.add(sources.vdc('vg', positive='vg', negative='0', value=1.0))
    tb.add_sim_type(analysis.Op())
    return mod, tb


def test_set_corner_stores_validates_and_clears(sky130_dut):
    _, build = sky130_dut
    _, tb = _bench(build)
    assert tb.corner is None
    assert tb.set_corner('ff') is tb
    assert tb.corner == 'ff'
    tb.set_corner(None)
    assert tb.corner is None


def test_unknown_corner_is_a_typed_error_listing_the_valid_corners(sky130_dut):
    pdk, build = sky130_dut
    _, tb = _bench(build)
    tb.set_corner('ff')
    with pytest.raises(UnknownCornerError) as exc:
        tb.set_corner('fff')
    msg = str(exc.value)
    assert "unknown corner 'fff'" in msg
    for name in ('tt', 'ff', 'ss', 'sf', 'fs'):
        assert repr(name) in msg
    # A configuration error (stops run_all_tests) and a ValueError (the
    # type Module.simulate raises for the same mistake).
    assert isinstance(exc.value, ConfigurationError)
    assert isinstance(exc.value, ValueError)
    # The refused name never replaced the earlier one.
    assert tb.corner == 'ff'


def test_set_corner_type_errors(sky130_dut):
    _, build = sky130_dut
    _, tb = _bench(build)
    with pytest.raises(TypeError):
        tb.set_corner(3)
    with pytest.raises(UnknownCornerError):
        tb.set_corner('')


def test_corner_enum_is_accepted_as_its_name(sky130_dut):
    from silk.simulation.corner import Corner
    _, build = sky130_dut
    _, tb = _bench(build)
    tb.set_corner(Corner.SS)
    assert tb.corner == 'ss' and type(tb.corner) is str


def test_pdk_free_bench_accepts_any_corner_name(tmp_path):
    from silk.circuit.circuit_elements import PortDir
    from silk.circuit.modules import Module
    mod = Module('rc', str(tmp_path), 'rc.cir', pdk_details=None, module_name='rc')
    mod.define_module_ports({'a': PortDir.INOUT})
    tb = mod.create_test('t')
    assert tb.set_corner('whatever').corner == 'whatever'


@pytest.fixture
def captured_corners(monkeypatch):
    """Stop each run after preparation and record the corner it resolved."""
    from silk.circuit.module_test import ModuleTestbench
    seen = []

    def fake_simulate(self, spec):
        seen.append(spec.corner_str)
        return spec

    monkeypatch.setattr(ModuleTestbench, '_simulate_run', fake_simulate)
    monkeypatch.setattr(ModuleTestbench, '_evaluate_run',
                        lambda self, result: result.corner_str)
    return seen


def test_corner_precedence_run_argument_over_test_over_pdk_default(
        sky130_dut, captured_corners):
    pdk, build = sky130_dut
    mod, tb = _bench(build)
    tb.run()                              # nothing set: the PDK default
    tb.set_corner('ff')
    tb.run()                              # the test's own corner
    tb.run(corner='ss')                   # the call wins
    mod.run_all_tests(corner='sf')        # forwarded to every test: still wins
    mod.run_all_tests()                   # no corner: the test's own again
    tb.set_corner(None)
    tb.run()                              # cleared: back to the PDK default
    assert captured_corners == [pdk.default_corner, 'ff', 'ss', 'sf', 'ff',
                                pdk.default_corner]


def test_circuitspec_corner_never_reaches_a_generated_subcircuit():
    """The CircuitSpec ``corner`` is the standalone script's run corner; a
    subcircuit DUT (``ports`` present) is emitted without it, so it cannot
    take part in a testbench run's corner precedence."""
    from silk.codegen.generator import Generator
    spec = {"circuit_name": "nfet", "pdk": "sky130",
            "ports": {"d": {"dir": "output"}, "g": {"dir": "input"}},
            "devices": [{"type": "nmos1p8", "instance": "M1",
                         "nets": {"d": "d", "g": "g", "s": "0", "b": "0"},
                         "params": {"w": 1.0, "l": 0.15}}]}
    tt = Generator().generate({**spec, "corner": "tt"})
    ff = Generator().generate({**spec, "corner": "ff"})
    assert tt == ff
    assert 'corner' not in tt


# --- generated code, built (no simulation) -----------------------------------

def _exec_generated(spec, build):
    ns = {'_build_nfet': build}
    exec(generate_testbenches(spec), ns)
    return ns['build_nfet_with_tests']


def test_generated_bench_carries_all_three_settings(sky130_dut):
    from silk.simulation.conditions import own_temperature_setting
    pdk, build = sky130_dut
    builder = _exec_generated(
        _spec(save_device_params=True, corner='ff', temperature_c=85), build)
    mod = builder(pdk)
    tb = mod._tests['op']
    assert mod.save_device_params is True
    assert tb.corner == 'ff'
    assert own_temperature_setting(tb.netlist)[0] == 85.0


def test_generated_bench_without_the_keys_is_unchanged(sky130_dut):
    from silk.simulation.conditions import own_temperature_setting
    pdk, build = sky130_dut
    mod = _exec_generated(_spec(), build)(pdk)
    tb = mod._tests['op']
    assert mod.save_device_params is False
    assert tb.corner is None
    assert own_temperature_setting(tb.netlist) is None


def test_generated_bench_with_unknown_corner_fails_when_built(sky130_dut):
    pdk, build = sky130_dut
    builder = _exec_generated(_spec(corner='zz'), build)
    with pytest.raises(UnknownCornerError, match=r"unknown corner 'zz'.*'tt'"):
        builder(pdk)


# --- slow: real ngspice + sky130 ---------------------------------------------

@pytest.mark.slow
def test_slow_generated_bench_runs_at_its_corner_and_temperature_with_op_table(
        sky130_dut):
    from tests._env import HAVE_NGSPICE, HAVE_PDK
    if not (HAVE_PDK and HAVE_NGSPICE):
        pytest.skip('needs an installed sky130 PDK and the ngspice binary')
    from silk.exceptions import SimulationError

    pdk, build = sky130_dut

    spec = _spec(save_device_params=True, corner='ff', temperature_c=85,
                 sources=[
                     {"type": "VoltageSource", "instance": "vd",
                      "positive": "vd", "negative": "0", "dc": 1.0},
                     {"type": "VoltageSource", "instance": "vg",
                      "positive": "vg", "negative": "0", "dc": 1.0}])
    mod = _exec_generated(spec, build)(pdk)
    report = mod.run_test('op')                  # no corner passed
    tb = mod._tests['op']

    deck = open(report.run.deck_path).read()
    # The deck runs at the test's corner, not the PDK default 'tt' ...
    lib_lines = [ln for ln in deck.splitlines() if ln.lower().startswith('.lib')]
    assert any(re.search(r'\sff\s*$', ln) for ln in lib_lines), lib_lines
    assert not any(re.search(r'\stt\s*$', ln) for ln in lib_lines), lib_lines
    # ... and at the test's temperature.
    assert re.search(r'temp\s*=\s*85', deck, re.IGNORECASE), deck
    assert report.corner == 'ff'
    assert report.operating_conditions.temperature_c == 85.0

    table = report.op_table()
    assert len(table) == 1
    row = table[0]
    assert row.model == 'sky130_fd_pr__nfet_01v8'
    assert row.id > 0 and row.gm > 0

    # Without the key the existing refusal is unchanged.
    spec_off = _spec(corner='ff', temperature_c=85, sources=spec['tests'][0]['sources'])
    mod_off = _exec_generated(spec_off, build)(pdk)
    report_off = mod_off.run_test('op')
    assert report_off.op_devices is None
    with pytest.raises(SimulationError, match='save_device_params'):
        report_off.op_table()
