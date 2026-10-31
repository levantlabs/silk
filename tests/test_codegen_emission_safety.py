"""Security regression tests for Python code emitted by silk.codegen."""

import re

import pytest

from silk.codegen import generator as g
from silk.codegen.generator import (
    CircuitSpec,
    DeviceSpec,
    Generator,
    SourceSpec,
    SubmoduleSpec,
)
from silk.codegen.testbench_generator import TestbenchSpecError


class _CaptureFactory:
    def __getattr__(self, factory_name):
        def build(instance, **kwargs):
            return {
                "factory": factory_name,
                "instance": instance,
                "kwargs": kwargs,
            }
        return build


class _CaptureModule:
    def __init__(self):
        self.added = []

    def add(self, *elements):
        self.added.extend(elements)

    def add_element(self, module, instance, *, nets, parameters=None):
        item = {
            "module": module,
            "instance": instance,
            "nets": nets,
            "parameters": parameters,
        }
        self.added.append(item)
        return item

    def add_module(self, module, instance, *, nets):
        return self.add_element(module, instance, nets=nets)


def test_generate_array_templates_keep_existing_braces_as_literal_data():
    payload = "net[$i]{globals().__setitem__('compromised', True)}"
    spec = CircuitSpec(
        "safe_arrays",
        devices=[
            DeviceSpec("nmos1p8", "M", {"d": payload}, {}, array=1),
        ],
        ideal_elements=[
            DeviceSpec(
                "resistor",
                "VS",
                {"p": payload, "n": "gnd"},
                {"value": 1000},
                array=1,
            ),
        ],
        sources=[
            SourceSpec(
                "VoltageSource",
                "VIN",
                payload,
                "gnd",
                dc=1.0,
                array=1,
            ),
        ],
        submodules=[
            SubmoduleSpec(
                "X",
                ".child",
                "child",
                {"a": payload},
                array=1,
            ),
        ],
    )
    code = Generator().generate(spec)
    emitted = [
        line for line in code.splitlines()
        if line.startswith(("ckt.add(*[", "for i in range("))
    ]
    assert len(emitted) == 4
    factory = _CaptureFactory()
    ckt = _CaptureModule()
    namespace = {
        "pdk": factory,
        "elements": factory,
        "sources": factory,
        "ckt": ckt,
        "child_child": object(),
        "compromised": False,
    }

    for line in emitted:
        exec(line, namespace)

    expected = "net[0]{globals().__setitem__('compromised', True)}"
    by_instance = {}
    for item in ckt.added:
        by_instance[item["instance"]] = item
    assert by_instance["M[0]"]["kwargs"]["d"] == expected
    assert by_instance["VS[0]"]["kwargs"]["p"] == expected
    assert by_instance["VIN[0]"]["kwargs"]["positive"] == expected
    assert by_instance["X[0]"]["nets"]["a"] == expected
    assert namespace["compromised"] is False


def test_generated_submodule_alias_avoids_load_pdk_binding():
    code = Generator().generate(CircuitSpec(
        "safe_alias",
        submodules=[SubmoduleSpec("xchild", ".pdk", "load", {})],
    ))

    assert "from .pdk import load as load_pdk_1" in code
    assert "ckt.add_module(load_pdk_1, 'xchild', nets={})" in code


def test_generate_testbenches_rejects_non_string_waveform_type():
    spec = {
        "circuit_name": "unsafe_tb",
        "tests": [{
            "name": "unsafe",
            "dut_portmap": {},
            "sources": [{
                "type": "VoltageSource",
                "instance": "vin",
                "positive": "in",
                "negative": "gnd",
                "waveform": {"type": 7},
            }],
        }],
    }

    with pytest.raises(TestbenchSpecError, match="Source .*vin.*waveform.*type"):
        Generator().generate_testbenches(spec)


@pytest.mark.parametrize(
    "instance",
    [
        "os",
        "_pdk_root",
        "_lib_path",
        "Net",
        "Module",
        "Port",
        "PortDir",
        "PortRole",
        "VoltageDomain",
        "load_pdk",
        "v",
    ],
)
def test_non_live_generator_names_remain_valid_instances(instance):
    code = Generator().generate(CircuitSpec(
        "compatible_name",
        devices=[DeviceSpec("nmos1p8", instance, {}, {})],
    ))

    assert f".add(pdk.nmos1p8({instance!r}, ))" in code
    assert f"{instance} = " not in code


@pytest.mark.parametrize(
    "spec, instance",
    [
        (
            CircuitSpec(
                "standalone_mod",
                devices=[DeviceSpec("nmos1p8", "mod", {}, {})],
            ),
            "mod",
        ),
        (
            CircuitSpec(
                "module_ckt",
                ports={"in": g.Port(g.PortDir.IN)},
                devices=[DeviceSpec("nmos1p8", "ckt", {}, {})],
            ),
            "ckt",
        ),
        (
            CircuitSpec(
                "standalone_analysis",
                devices=[DeviceSpec("nmos1p8", "analysis", {}, {})],
            ),
            "analysis",
        ),
        (
            CircuitSpec(
                "standalone_sources",
                devices=[DeviceSpec("nmos1p8", "sources", {}, {})],
            ),
            "sources",
        ),
        (
            CircuitSpec(
                "standalone_elements",
                devices=[DeviceSpec("nmos1p8", "elements", {}, {})],
            ),
            "elements",
        ),
    ],
)
def test_output_mode_allows_names_without_a_live_collision(spec, instance):
    code = Generator().generate(spec)

    assert f".add(pdk.nmos1p8({instance!r}, ))" in code
    assert f"{instance} = " not in code


def test_submodule_alias_avoids_voltage_domains_for_dict_and_direct_specs():
    direct_domain = g.VoltageDomain(name="child_blocks", nominal=1.8)
    specs = [
        {
            "circuit_name": "domain_alias_dict",
            "voltage_domains": {
                "child_blocks": {"nominal": 1.8},
            },
            "ports": {
                "vdd": {"dir": "power", "domain": "child_blocks"},
            },
            "submodules": [{
                "instance_name": "xchild",
                "module_ref": {
                    "import_path": ".blocks",
                    "module_var": "child",
                },
                "nets": {},
            }],
        },
        CircuitSpec(
            "domain_alias_direct",
            voltage_domains={"child_blocks": direct_domain},
            ports={"vdd": g.Port(g.PortDir.POWER, domain=direct_domain)},
            submodules=[
                SubmoduleSpec("xchild", ".blocks", "child", {}),
            ],
        ),
        CircuitSpec(
            "domain_alias_mismatched_key",
            voltage_domains={"supply": direct_domain},
            ports={"vdd": g.Port(g.PortDir.POWER, domain=direct_domain)},
            submodules=[
                SubmoduleSpec("xchild", ".blocks", "child", {}),
            ],
        ),
    ]

    for spec in specs:
        code = Generator().generate(spec)
        assert "from .blocks import child as child_blocks_1" in code
        assert "mod.add_module(child_blocks_1, 'xchild', nets={})" in code


def test_standalone_alias_ignores_unemitted_voltage_domain():
    domain = g.VoltageDomain(name="child_blocks", nominal=1.8)
    code = Generator().generate(CircuitSpec(
        "standalone_domain_alias",
        voltage_domains={"child_blocks": domain},
        submodules=[SubmoduleSpec("xchild", ".blocks", "child", {})],
    ))

    assert "from .blocks import child as child_blocks\n" in code
    assert "ckt.add_module(child_blocks, 'xchild', nets={})" in code


@pytest.mark.parametrize(
    "payload",
    [
        "./sim'); compromised = True; #",
        "./sim\\'); compromised = True; #",
        "./sim\n compromised = True",
        './sim"); compromised = True; #',
    ],
)
def test_portless_output_dir_breakout_stays_data(payload):
    code = Generator().generate(CircuitSpec("safe_dir", output_dir=payload))
    [ctor_line] = [
        line for line in code.splitlines() if line.startswith("ckt = Module(")
    ]

    captured = {}

    def module(pdk_name, output_dir, fname, *, pdk_details):
        captured.update(
            pdk_name=pdk_name, output_dir=output_dir, fname=fname,
        )

    namespace = {"Module": module, "pdk": object(), "compromised": False}
    exec(ctor_line, namespace)

    assert captured["output_dir"] == payload
    assert captured["pdk_name"] == "PDK_sky130"
    assert captured["fname"] == "safe_dir.cir"
    assert namespace["compromised"] is False


def test_module_ctor_lines_stay_byte_identical_for_wellformed_specs():
    standalone = Generator().generate(CircuitSpec("my_ckt"))
    assert (
        "ckt = Module('PDK_sky130', './sim', 'my_ckt.cir', pdk_details=pdk)"
        in standalone.splitlines()
    )

    module = Generator().generate(CircuitSpec(
        "my_mod",
        output_dir="./gen/out",
        ports={"in": g.Port(g.PortDir.IN)},
    ))
    assert (
        "mod = Module('PDK_sky130', './gen/out', 'my_mod.cir', "
        "pdk_details=pdk," in module.splitlines()
    )


@pytest.mark.parametrize('field', ['instance', 'positive', 'negative'])
def test_generate_testbenches_refuses_quote_breakout_without_execution(field):
    from silk.exceptions import CircuitError
    payload = "vin');globals().__setitem__('compromised',True);#"
    # Exercise each authored component separately. A breakout would mutate
    # the generated function's globals, so the marker observes execution.
    source = {'type': 'VoltageSource', 'instance': 'vin',
              'positive': 'out', 'negative': '0', 'dc': 1.2}
    source[field] = payload
    code = Generator().generate_testbenches({
        "circuit_name": "safe_tb",
        "tests": [{
            "name": "safe",
            "dut_portmap": {},
            "sources": [source],
        }],
    })

    class FakeTestbench:
        def __init__(self):
            self.sources = []

        def add_module_under_test(self, _portmap):
            pass

        def add(self, source):
            self.sources.append(source)

    class FakeModule:
        def __init__(self):
            self.testbench = FakeTestbench()

        def create_test(self, _name):
            return self.testbench

    namespace = {"compromised": False}
    exec(code, namespace)
    module = FakeModule()
    error = CircuitError if field == 'instance' else ValueError
    with pytest.raises(error, match='invalid (instance|net) name'):
        namespace["_attach_safe"](module)

    assert module.testbench.sources == []
    assert namespace["compromised"] is False


def test_generate_testbenches_keeps_valid_source_output_byte_identical():
    code = Generator().generate_testbenches({
        "circuit_name": "c",
        "tests": [{
            "name": "t",
            "dut_portmap": {},
            "sources": [{
                "type": "VoltageSource",
                "instance": "vin",
                "positive": "in",
                "negative": "gnd",
                "dc": 1.2,
            }],
        }],
    })

    assert code == (
        "from silk import sources\n"
        "\n"
        "def _attach_t(mod):\n"
        "    tb = mod.create_test('t')\n"
        "    tb.add_module_under_test({})\n"
        "    tb.add(sources.vdc('vin', positive='in', negative='gnd', "
        "value=1.2))\n"
    )


@pytest.mark.parametrize(
    "spec",
    [
        {
            "circuit_name": "unsafe_keyword",
            "devices": [{
                "type": "nmos1p8",
                "instance": "class",
                "nets": {},
                "params": {},
            }],
        },
        CircuitSpec(
            "unsafe_non_string",
            sources=[SourceSpec("VoltageSource", 17, "out", "gnd")],
        ),
        CircuitSpec(
            "unsafe_injection",
            ideal_elements=[DeviceSpec(
                "resistor",
                "r1); compromised = True; #",
                {"p": "out", "n": "gnd"},
                {"value": 1_000},
            )],
        ),
    ],
)
def test_generate_rejects_non_identifier_instance_bindings(spec):
    with pytest.raises(ValueError, match="must be a valid Python identifier"):
        Generator().generate(spec)


def _shadow_spec(tmp_path, ports):
    spec = {
        "circuit_name": "shadow_names",
        "pdk": "sky130",
        "output_dir": str(tmp_path),
        "devices": [
            {"type": "nmos1p8", "instance": "pdk",
             "nets": {"d": "a", "g": "a", "s": "a", "b": "a"},
             "params": {"w": 1.0, "l": 0.18}},
            {"type": "nmos1p8", "instance": "mod", "array": 2,
             "nets": {"d": "a", "g": "a", "s": "a", "b": "a"},
             "params": {"w": 1.0, "l": 0.18}},
        ],
        "ideal_elements": [
            {"type": "resistor", "instance": "elements",
             "nets": {"p": "a", "n": "gnd"}, "params": {"value": 1000}},
            {"type": "resistor", "instance": "sources",
             "nets": {"p": "a", "n": "gnd"}, "params": {"value": 1000}},
            {"type": "resistor", "instance": "ckt", "array": 2,
             "nets": {"p": "a", "n": "gnd"}, "params": {"value": 1000}},
        ],
        "sources": [
            {"type": "VoltageSource", "instance": "tb",
             "positive": "a", "negative": "gnd", "dc": 0.5},
            {"type": "VoltageSource", "instance": "vsrc", "array": 2,
             "positive": "a", "negative": "gnd", "dc": 0.5},
        ],
    }
    if ports:
        spec["ports"] = {"a": {"dir": "inout"}}
    return spec


@pytest.mark.parametrize("ports", [True, False])
def test_generator_local_names_are_legal_instance_names(tmp_path, ports):
    """TASK-351: instance names are string arguments, never Python
    variables, so devices/elements/sources named ``pdk``, ``mod``, ``ckt``,
    ``sources``, ``elements`` or ``tb`` generate code that executes and
    registers every instance under its own name."""
    from tests._env import HAVE_PDK
    if not HAVE_PDK:
        pytest.skip("needs an installed sky130 PDK")
    code = Generator().generate(_shadow_spec(tmp_path, ports))
    for name in ("pdk", "mod", "ckt", "elements", "sources", "tb"):
        # the only top-level binding of these names is the generator's own
        assert len(re.findall(rf"^{name}\s*=", code, re.M)) <= 1, name
    assert "= pdk." not in code and "= elements." not in code
    assert "= sources." not in code
    code = code.replace("ckt.simulate(", "pass  # ckt.simulate(")
    ns = {}
    exec(code, ns)
    module = ns["ckt" if not ports else "mod"]
    names = set(module.elem_dict) | set(module.sig_sources_dict)
    assert {"pdk", "mod[0]", "mod[1]", "elements", "sources", "ckt[0]",
            "ckt[1]", "tb", "vsrc[0]", "vsrc[1]"} <= names


@pytest.mark.parametrize(
    "spec",
    [
        {
            "circuit_name": "unsafe_type",
            "devices": [{
                "type": "nmos1p8); compromised = True; #",
                "instance": "m1",
                "nets": {},
                "params": {},
            }],
        },
        CircuitSpec(
            "unsafe_net_key",
            devices=[DeviceSpec(
                "nmos1p8", "m1",
                {"d=1); compromised = True; #": "out"}, {},
            )],
        ),
        CircuitSpec(
            "unsafe_parameter_key",
            devices=[DeviceSpec("nmos1p8", "m1", {}, {"class": 1.0})],
        ),
        {
            "circuit_name": "unsafe_waveform_key",
            "sources": [{
                "type": "VoltageSource",
                "instance": "vin",
                "positive": "in",
                "negative": "gnd",
                "waveform": {
                    "type": "Pulse",
                    "v1=0); compromised = True; #": 0,
                },
            }],
        },
    ],
)
def test_generate_rejects_caller_controlled_bare_tokens(spec):
    with pytest.raises(ValueError, match="must be a valid Python identifier"):
        Generator().generate(spec)


def test_generate_testbenches_wraps_invalid_waveform_bare_token():
    spec = {
        "circuit_name": "unsafe_tb",
        "tests": [{
            "name": "unsafe",
            "dut_portmap": {},
            "sources": [{
                "type": "VoltageSource",
                "instance": "vin",
                "positive": "in",
                "negative": "gnd",
                "waveform": {
                    "type": "Pulse",
                    "v1=0); compromised = True; #": 0,
                },
            }],
        }],
    }

    with pytest.raises(TestbenchSpecError, match="waveform parameter key"):
        Generator().generate_testbenches(spec)


def test_metadata_derived_source_alias_is_validated_at_emission(monkeypatch):
    monkeypatch.setattr(g, "_source_meta", lambda: {
        "sources": {
            "VoltageSource": {
                "alias": "vdc); compromised = True; #",
            },
        },
        "waveforms": {},
    })
    spec = CircuitSpec(
        "unsafe_alias",
        sources=[SourceSpec("VoltageSource", "vin", "in", "gnd")],
    )

    with pytest.raises(ValueError, match="source alias"):
        Generator().generate(spec)


def test_quote_containing_values_remain_data_across_emitters():
    payload = "out'); compromised = True; #"
    lines = [
        g._device_call(
            DeviceSpec("nmos1p8", "m1", {"d": payload}, {"w": 1.0}),
            "pdk",
        ),
        g._device_array_call(
            DeviceSpec(
                "nmos1p8", "MA", {"d": payload}, {"w": 1.0}, array=2,
            ),
            "pdk",
        ),
        g._source_call(
            SourceSpec("VoltageSource", "vin", payload, "gnd", dc=1.0)
        ),
        g._source_array_call(
            SourceSpec(
                "VoltageSource", "VIN", payload, "gnd", dc=1.0, array=2,
            )
        ),
    ]
    factory = _CaptureFactory()
    namespace = {
        "pdk": factory,
        "sources": factory,
        "compromised": False,
    }

    results = [eval(line, namespace) for line in lines]
    m1, ma, vin, vina = results

    assert m1["kwargs"]["d"] == payload
    assert all(item["kwargs"]["d"] == payload for item in ma)
    assert vin["kwargs"]["positive"] == payload
    assert all(item["kwargs"]["positive"] == payload for item in vina)
    assert namespace["compromised"] is False


def test_identifier_safe_emission_remains_byte_identical():
    assert g._device_call(
        DeviceSpec(
            "nmos1p8", "m1", {"d": "out", "g": "in"},
            {"w": 2.0, "l": 0.18},
        ),
        "pdk",
    ) == (
        "pdk.nmos1p8('m1', d='out', g='in', w=2.0, l=0.18)"
    )
    assert g._device_array_call(
        DeviceSpec(
            "nmos1p8", "M", {"d": "out[$i]", "g": "in"},
            {"w": 2.0}, array=2,
        ),
        "pdk",
    ) == (
        "[pdk.nmos1p8(f'M[{i}]', d=f'out[{i}]', g='in', w=2.0) "
        "for i in range(2)]"
    )
    assert g._source_call(
        SourceSpec("VoltageSource", "vin", "in", "gnd", dc=1.2)
    ) == (
        "sources.vdc('vin', positive='in', negative='gnd', value=1.2)"
    )
    assert g._source_array_call(
        SourceSpec(
            "VoltageSource", "VIN", "in[$i]", "gnd", dc=0.1, array=2,
        )
    ) == (
        "[sources.vdc(f'VIN[{i}]', positive=f'in[{i}]', "
        "negative='gnd', value=0.1) for i in range(2)]"
    )


@pytest.mark.parametrize(
    ("emit", "context"),
    [
        (
            lambda: g._device_call(
                DeviceSpec("nmos1p8", "bad'name", {}, {}), "pdk"
            ),
            "device instance name",
        ),
        (
            lambda: g._device_call(
                DeviceSpec("bad.type", "m1", {}, {}), "pdk"
            ),
            "device type",
        ),
        (
            lambda: g._device_call(
                DeviceSpec("nmos1p8", "m1", {"d=1": "out"}, {}), "pdk"
            ),
            "device net key",
        ),
        (
            lambda: g._device_call(
                DeviceSpec("nmos1p8", "m1", {}, {"w=1": 2.0}), "pdk"
            ),
            "device parameter key",
        ),
        (
            lambda: g._device_array_call(
                DeviceSpec(
                    "nmos1p8", "M", {"d": "out"}, {"class": 2.0}, array=2,
                ),
                "pdk",
            ),
            "device parameter key",
        ),
        (
            lambda: g._source_call(
                SourceSpec("VoltageSource", "bad'name", "out", "gnd")
            ),
            "source instance name",
        ),
        (
            lambda: g._source_call_body(
                "bad.alias",
                "positive='out', negative='gnd'",
                SourceSpec("VoltageSource", "vin", "out", "gnd"),
            ),
            "source alias",
        ),
        (
            lambda: g._source_call(
                SourceSpec(
                    "VoltageSource", "vin", "out", "gnd",
                    waveform={"type": "bad.kind"},
                )
            ),
            "source waveform kind",
        ),
        (
            lambda: g._source_call(
                SourceSpec(
                    "VoltageSource", "vin", "out", "gnd",
                    waveform={"type": "Pulse", "v1=0": 0},
                )
            ),
            "source waveform parameter key",
        ),
    ],
)
def test_bare_emission_tokens_reject_invalid_identifiers(emit, context):
    with pytest.raises(
        ValueError,
        match=rf"{context} .* must be a valid Python identifier",
    ):
        emit()


# ---------------------------------------------------------------------------
# TASK-13: circuit_name / corner emission contract
#
# circuit_name is an identity (SPICE .subckt namespace, DB structure identity,
# module_name= literal, output filename) and must be a valid, non-keyword
# Python identifier. corner must match [A-Za-z0-9_-]+ (syntax only — manifest
# membership is Module.simulate's job). Both are enforced at the emission
# boundary so dict and direct CircuitSpec inputs are held to the same rule,
# and every literal/header emission site is repr/docstring-escaped behind
# the validator as defense in depth.
# ---------------------------------------------------------------------------

_INJECTION_PAYLOADS = [
    pytest.param("x'); compromised = True; #", id="single-quote"),
    pytest.param('x"); compromised = True; #', id="double-quote"),
    pytest.param("x\\'); compromised = True; #", id="backslash-quote"),
    pytest.param("x\ncompromised = True", id="newline"),
    pytest.param("x  # compromised = True", id="comment"),
    pytest.param('x"""\ncompromised = True\n"""', id="triple-quote"),
]

_PORTS = {"in": g.Port(g.PortDir.IN), "out": g.Port(g.PortDir.OUT)}
_PORTS_DICT = {"in": {"dir": "input"}, "out": {"dir": "output"}}


def _spec_variants(field, value):
    """Every (input form x branch) combination carrying *value* in *field*."""
    base = {"circuit_name": "safe", "corner": "tt"}
    base[field] = value
    return [
        pytest.param(dict(base), id="dict-portless"),
        pytest.param({**base, "ports": _PORTS_DICT}, id="dict-ports"),
        pytest.param(
            CircuitSpec(base["circuit_name"], corner=base["corner"]),
            id="direct-portless",
        ),
        pytest.param(
            CircuitSpec(
                base["circuit_name"], corner=base["corner"], ports=dict(_PORTS),
            ),
            id="direct-ports",
        ),
    ]


@pytest.mark.parametrize("payload", _INJECTION_PAYLOADS)
def test_generate_rejects_injected_circuit_name_before_any_output(payload):
    for spec in (p.values[0] for p in _spec_variants("circuit_name", payload)):
        with pytest.raises(
            ValueError,
            match=r"circuit_name .* must be a valid Python identifier and "
                  r"must not be a Python keyword",
        ):
            Generator().generate(spec)


@pytest.mark.parametrize("payload", _INJECTION_PAYLOADS)
def test_generate_rejects_injected_corner_before_any_output(payload):
    for spec in (p.values[0] for p in _spec_variants("corner", payload)):
        with pytest.raises(
            ValueError,
            match=r"corner .* must be a non-empty string of letters, digits, "
                  r"underscores, or hyphens \(\[A-Za-z0-9_-\]\+\)",
        ):
            Generator().generate(spec)


@pytest.mark.parametrize(
    "bad_name",
    ["class", "my circuit", "diff-amp", "1amp", "", None, 42, "amp.v2"],
)
def test_generate_rejects_non_identifier_circuit_names(bad_name):
    with pytest.raises(ValueError, match=r"circuit_name .* Python identifier"):
        Generator().generate(CircuitSpec(bad_name))
    with pytest.raises(ValueError, match=r"circuit_name .* Python identifier"):
        Generator().generate({"circuit_name": bad_name})


@pytest.mark.parametrize("bad_corner", ["", " tt", "tt ", "tt,ss", None, 1, "tt.ss"])
def test_generate_rejects_non_syntactic_corners(bad_corner):
    pattern = r"corner .* \(\[A-Za-z0-9_-\]\+\)"
    with pytest.raises(ValueError, match=pattern):
        Generator().generate(CircuitSpec("safe", corner=bad_corner))
    with pytest.raises(ValueError, match=pattern):
        Generator().generate({"circuit_name": "safe", "corner": bad_corner})


@pytest.mark.parametrize(
    "corner", ["tt", "ff_hh_mm", "typical", "ss-1", "SS", "corner_2"],
)
def test_generate_accepts_syntactic_corners_without_manifest_lookup(corner):
    # Emission must not consult the PDK manifest: 'ss-1' and 'corner_2' are
    # not sky130 corners, yet they are syntactically valid and pass through.
    code = Generator().generate(CircuitSpec("safe", corner=corner))
    assert f"ckt.simulate(corner={corner!r})" in code.splitlines()
    assert f"Corner:  {corner}" in code.splitlines()


def test_generate_rejects_invalid_identity_before_output_or_file(tmp_path):
    target = tmp_path / "out.py"
    with pytest.raises(ValueError, match="circuit_name"):
        Generator().generate_to_file(CircuitSpec("bad name"), str(target))
    assert not target.exists()
    with pytest.raises(ValueError, match="corner"):
        Generator().generate_to_file(
            CircuitSpec("ok", corner="tt;"), str(target),
        )
    assert not target.exists()


def _assignment_targets(tree):
    import ast
    return {
        t.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name)
    }


@pytest.mark.parametrize("payload", _INJECTION_PAYLOADS)
@pytest.mark.parametrize("with_ports", [False, True], ids=["portless", "ports"])
def test_identity_escaping_holds_even_if_validation_is_bypassed(
    monkeypatch, payload, with_ports,
):
    """Defense in depth: with the contract validator disabled, the emitted
    literals and headers still keep the payload as data — no injected
    statement, no terminated docstring."""
    import ast

    monkeypatch.setattr(g, "_validate_identity_fields", lambda spec: None)
    spec = CircuitSpec(
        payload, corner=payload, ports=dict(_PORTS) if with_ports else None,
    )
    code = Generator().generate(spec)

    tree = ast.parse(code)  # would raise on a terminated header/literal
    assert "compromised" not in _assignment_targets(tree)
    # The header parsed as one docstring holding the payload verbatim.
    assert payload in ast.get_docstring(tree, clean=False)

    if with_ports:
        [call] = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "Module"
        ]
        [module_name] = [k.value for k in call.keywords if k.arg == "module_name"]
        assert isinstance(module_name, ast.Constant)
        assert module_name.value == payload
    else:
        [sim] = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "attr", None) == "simulate"
        ]
        [corner] = [k.value for k in sim.keywords if k.arg == "corner"]
        assert isinstance(corner, ast.Constant)
        assert corner.value == payload


def test_identity_emission_lines_stay_byte_identical_for_wellformed_specs():
    standalone = Generator().generate(
        CircuitSpec("my_ckt", corner="ff_hh_mm")
    ).splitlines()
    assert standalone[:5] == [
        '"""Generated by silk CircuitCodeGenerator',
        "Circuit: my_ckt",
        "PDK:     sky130",
        "Corner:  ff_hh_mm",
        '"""',
    ]
    assert "ckt.simulate(corner='ff_hh_mm')" in standalone

    module = Generator().generate(CircuitSpec(
        "Ünit_ötã", corner="tt", ports=dict(_PORTS),
    )).splitlines()
    assert module[0] == '"""Generated subcircuit: Ünit_ötã"""'
    assert "             module_name='Ünit_ötã')" in module


@pytest.mark.parametrize("bad_name", ["class", "ac gain", "x'); y=1 #", None])
def test_generate_testbenches_rejects_non_identifier_circuit_name_with_builder(
    bad_name,
):
    spec = {
        "circuit_name": bad_name,
        "circuit_builder": "_build_c",
        "rigs": [],
        "tests": [],
    }
    if bad_name is None:
        from silk.codegen.testbench_generator import TestbenchSpec
        spec = TestbenchSpec(circuit_name=None, circuit_builder="_build_c")
    with pytest.raises(
        TestbenchSpecError,
        match=r"circuit_name .* must be a valid Python identifier and must "
              r"not be a Python keyword",
    ):
        Generator().generate_testbenches(spec)


def test_generate_testbenches_keeps_informational_circuit_name_without_builder():
    # No circuit_builder -> circuit_name is never emitted, so the identity
    # contract does not apply and existing informational names still work.
    code = Generator().generate_testbenches({
        "circuit_name": "ac gain", "rigs": [], "tests": [],
    })
    assert "ac gain" not in code
    assert "def build_" not in code


# --- TASK-14: circuit_builder is emitted bare as a call target -------------
# ``mod = <circuit_builder>(pdk)`` — the contract is a bare, non-keyword
# Python identifier (the sidecar builder's name). Anything else must raise
# TestbenchSpecError before any output exists, for dict and direct input.

_BAD_BUILDERS = [
    "class",                      # keyword
    "evil(); x",                  # two live statements
    "pkg.mod.build",              # dotted path (rejected by contract)
    "_build_c(pdk); import os",   # call + import
    "x'); y=1 #",                 # quote breakout + comment
    "_build_c\nimport os",        # newline
    "_build_c  # comment",        # comment payload
    "1build",                     # leading digit
    "build-me",                   # hyphen
    "_build_c()",                 # call expression
    42,                           # non-str
]


def _tb_spec_dict(builder):
    return {
        "circuit_name": "ac_gain",
        "circuit_builder": builder,
        "rigs": [],
        "tests": [{
            "name": "t_gain",
            "dut_portmap": {"vin": "in", "vout": "out"},
            "applied_rigs": [],
            "sources": [],
            "analyses": [{"kind": "ac", "start": 1.0, "stop": 1e6, "points": 10}],
            "metrics": [],
            "specs": [],
        }],
    }


@pytest.mark.parametrize("bad", _BAD_BUILDERS, ids=repr)
@pytest.mark.parametrize("direct", [False, True], ids=["dict", "direct"])
def test_generate_testbenches_rejects_non_identifier_circuit_builder(bad, direct):
    from silk.codegen.testbench_generator import TestbenchSpec

    spec = _tb_spec_dict(bad)
    if direct:
        spec = TestbenchSpec(**spec)
    with pytest.raises(
        TestbenchSpecError,
        match=r"circuit_builder .* must be a bare Python identifier .* "
              r"must not be a Python keyword",
    ) as ei:
        Generator().generate_testbenches(spec)
    # The error names the offending value; nothing was returned, so no
    # injected statement can have been emitted.
    assert repr(bad) in str(ei.value)


def test_generate_testbenches_circuit_builder_injection_never_reaches_output(
    monkeypatch,
):
    # Even if a caller sneaks past _validate, the assembler is the only
    # place circuit_builder is interpolated; prove the validator is what
    # stands between the payload and a live statement by checking that the
    # validated path raises and the emitted text of a *valid* spec contains
    # exactly one call-target line built from the builder name.
    payload = "evil(); x"
    with pytest.raises(TestbenchSpecError):
        Generator().generate_testbenches(_tb_spec_dict(payload))

    code = Generator().generate_testbenches(_tb_spec_dict("_build_ac_gain"))
    builder_lines = [l for l in code.splitlines() if "(pdk)" in l and "mod =" in l]
    assert builder_lines == ["    mod = _build_ac_gain(pdk)"]
    assert "evil" not in code


def test_generate_testbenches_wellformed_circuit_builder_lines_byte_identical():
    from silk.codegen.testbench_generator import TestbenchSpec

    d = _tb_spec_dict("_build_ac_gain")
    code_dict = Generator().generate_testbenches(d)
    code_direct = Generator().generate_testbenches(TestbenchSpec(**d))
    assert code_dict == code_direct
    lines = code_dict.splitlines()
    i = lines.index("def build_ac_gain_with_tests(pdk):")
    assert lines[i:i + 4] == [
        "def build_ac_gain_with_tests(pdk):",
        "    mod = _build_ac_gain(pdk)",
        "    _attach_t_gain(mod)",
        "    return mod",
    ]
    compile(code_dict, "<generated>", "exec")


@pytest.mark.parametrize("absent", [None, ""], ids=["None", "empty"])
def test_generate_testbenches_falsy_circuit_builder_still_omits_assembler(absent):
    d = _tb_spec_dict(absent)
    if absent is None:
        del d["circuit_builder"]
    code = Generator().generate_testbenches(d)
    assert "def build_" not in code
    assert "(pdk)" not in code


# --- TASK-321: AC/Noise 'points' is validated + rendered, never bare -------
#
# points used to be interpolated bare into the emitted
# `analysis.AC(...)`/`analysis.Noise(...)` call (unlike fstart/fstop, which
# always went through _fmt_value): an SI string like '1k' produced a
# SyntaxError in the generated module, and any string was pasted verbatim
# into code that is later executed.

def _tb_spec_with_points(points):
    d = _tb_spec_dict("_build_ac_gain")
    d["tests"][0]["analyses"] = [
        {"kind": "ac", "variation": "dec", "points": points,
         "fstart": 1, "fstop": 1e6},
    ]
    return d


def test_generate_testbenches_points_int_is_byte_identical():
    # AC #3: existing (int) behavior through the full testbench path is
    # unchanged, byte for byte.
    code = Generator().generate_testbenches(_tb_spec_with_points(10))
    assert "tb.add_sim_type(analysis.AC('dec', 10, 1, 1e+06))" in code
    compile(code, "<generated>", "exec")


def test_generate_testbenches_points_accepts_si_string():
    # AC #1: an SI string parsing to a positive integer renders as an int
    # literal through the full testbench path.
    code = Generator().generate_testbenches(_tb_spec_with_points('1k'))
    assert "tb.add_sim_type(analysis.AC('dec', 1000, 1, 1e+06))" in code
    compile(code, "<generated>", "exec")


@pytest.mark.parametrize("bad_points", [
    "10.5",
    "x); import os #",
    "inf",     # non-finite (OverflowError on bare int())
    "1e400",   # non-finite (overflows to inf; same OverflowError)
    "nan",     # non-finite (bare ValueError on int(), no field/value)
])
def test_generate_testbenches_points_injection_never_reaches_output(bad_points):
    # AC #2: a non-integer/non-numeric points value is refused, naming the
    # field, before any code is returned — the payload cannot appear in
    # generated output because no output is generated on this path.
    with pytest.raises(ValueError, match=r"points .* must be an int or an SI string") as ei:
        Generator().generate_testbenches(_tb_spec_with_points(bad_points))
    assert repr(bad_points) in str(ei.value)


# ---------------------------------------------------------------------------
# TASK-324: CircuitSpec port names are identifier-validated and rendered
# via repr(), never bare.
#
# Port names used to be interpolated into the emitted
# `mod.define_module_ports({...})` call as `f"'{name}': ..."` — hand-rolled
# single quotes instead of repr() — and CircuitSpec.from_dict never
# validated them, so a port name could break out of the quotes and inject
# arbitrary code into the generated module. Python keywords (e.g. 'in')
# must stay legal: they are emitted as dict keys via repr(), never as bare
# Python locals, and analog designers name ports in/out routinely.
# ---------------------------------------------------------------------------

def _spec_with_port_name(name, *, direct=False):
    if direct:
        return CircuitSpec("safe", corner="tt", ports={name: g.Port(g.PortDir.IN)})
    return {"circuit_name": "safe", "corner": "tt",
            "ports": {name: {"dir": "input"}}}


@pytest.mark.parametrize("payload", _INJECTION_PAYLOADS)
@pytest.mark.parametrize("direct", [False, True], ids=["dict-ports", "direct-ports"])
def test_generate_rejects_injected_port_name_before_any_output(payload, direct):
    spec = _spec_with_port_name(payload, direct=direct)
    with pytest.raises(
        ValueError,
        match=r"port name .* must be an identifier-shaped string",
    ):
        Generator().generate(spec)


def test_generate_rejects_reproducer_port_name_injection():
    # Exact reproducer from the TASK-324 description.
    with pytest.raises(
        ValueError, match=r"port name .* must be an identifier-shaped string"
    ):
        Generator().generate(CircuitSpec.from_dict({
            'circuit_name': 'poc', 'pdk': 'sky130',
            'ports': {"a': None}); PWNED = 1 #": {'dir': 'input'}},
        }))


@pytest.mark.parametrize(
    "bad_name", ["1amp", "my port", "d[3:0]", "", None, 42, "a.b", "a-b"],
)
def test_generate_rejects_non_identifier_port_names(bad_name):
    with pytest.raises(
        ValueError, match=r"port name .* must be an identifier-shaped string"
    ) as ei:
        Generator().generate(
            {"circuit_name": "x", "ports": {bad_name: {"dir": "input"}}}
        )
    assert repr(bad_name) in str(ei.value)


def test_generate_accepts_keyword_port_name_in():
    # A port named 'in' (a Python keyword) still generates, and the
    # emitted module imports and builds.
    code = Generator().generate(CircuitSpec.from_dict({
        "circuit_name": "kwtest",
        "ports": {"in": {"dir": "input"}, "out": {"dir": "output"}},
    }))
    assert (
        "mod.define_module_ports({'in': PortDir.IN, 'out': PortDir.OUT})"
        in code
    )
    compile(code, "<generated>", "exec")


def test_port_dict_lines_stay_byte_identical_for_wellformed_specs():
    code = Generator().generate(CircuitSpec(
        "my_mod", ports={"in": g.Port(g.PortDir.IN), "out": g.Port(g.PortDir.OUT)},
    ))
    assert (
        "mod.define_module_ports({'in': PortDir.IN, 'out': PortDir.OUT})"
        in code.splitlines()
    )


@pytest.mark.parametrize("exponent", [400, 5000])
@pytest.mark.parametrize("sign", [1, -1])
def test_generate_testbenches_huge_points_names_field(exponent, sign):
    with pytest.raises(ValueError, match=r"points .* must be an int or an SI string"):
        Generator().generate_testbenches(_tb_spec_with_points(sign * 10**exponent))


@pytest.mark.parametrize("exponent", [400, 5000])
def test_require_points_huge_int_preserves_context(exponent):
    from silk.codegen.generator import _require_points

    with pytest.raises(ValueError, match=r"custom.points .* must be an int or an SI string"):
        _require_points(10**exponent, context="custom.points")


@pytest.mark.parametrize('renderer', [g._format_submodule_nets_dict, g._format_array_nets_dict])
@pytest.mark.parametrize('module_var', ['mod', 'ckt'])
@pytest.mark.parametrize('bus_name', ["bus'quoted", 'bus\\slash'])
def test_bus_lookup_keys_are_repr_escaped(renderer, module_var, bus_name):
    # Exercise escaping at the renderer boundary without relaxing name validation.
    from types import SimpleNamespace
    handle = object()
    spec = CircuitSpec.from_dict({'circuit_name': 'parent',
                                  'bus_nets': [{'name': bus_name, 'bits': 2}]})
    literal = renderer({'d': bus_name}, spec, module_var=module_var)
    result = eval(compile(literal, '<bus_lookup>', 'eval'),
                  {module_var: SimpleNamespace(nets={bus_name: handle})})
    assert result['d'] is handle


@pytest.mark.parametrize('direct', [False, True])
def test_domain_aliases_avoid_all_live_bindings(tmp_path, monkeypatch, direct):
    import sys
    import types
    from silk import Module, load_pdk
    child = Module('sky130', str(tmp_path), 'child.cir', pdk_details=load_pdk('sky130'),
                   module_name='child')
    child.define_module_ports({})
    blocks = types.ModuleType('Port')
    blocks.domain = child
    monkeypatch.setitem(sys.modules, 'Port', blocks)
    names = ['Module', 'domain_Module', 'Variable', 'Port',
             '__silk_r_nets', 'domain_Port_1']
    spec = {
        'circuit_name': 'collisions', 'output_dir': str(tmp_path),
        'voltage_domains': {name: {'nominal': 1.8} for name in names},
        'ports': {f'p{i}': {'dir': 'input', 'domain': name}
                  for i, name in enumerate(names)},
        'variables': [{'name': 'domain_Variable', 'value': 1000}],
        'ideal_elements': [{'type': 'resistor', 'instance': 'r', 'array': 1,
                            'nets': [{'p': 'p0', 'n': 'gnd'}],
                            'params': {'value': {'variable': 'domain_Variable'}}}],
        'submodules': [{'instance_name': 'x', 'module_ref': {
            'import_path': 'Port', 'module_var': 'domain'}, 'nets': {}}],
    }
    # The natural import alias occupies domain_Port, forcing a suffix.
    if direct:
        spec = CircuitSpec.from_dict(spec)
        spec.voltage_domains = {f'key{i}': domain for i, domain
                                in enumerate(spec.voltage_domains.values())}
    code = Generator().generate(spec)
    assert code == Generator().generate(spec)
    ns = {}
    exec(compile(code, '<generated>', 'exec'), ns)
    mod = ns['mod']
    for i, name in enumerate(names):
        assert mod.pin_list[f'p{i}'].domain.name == name
    assert ns['domain_Module'].name == 'domain_Module'
    assert ns['domain_Module_1'].name == 'Module'
    assert ns['domain_Variable_1'].name == 'Variable'
    assert ns['domain_Port_2'].name == 'Port'
    assert ns['domain___silk_r_nets'] is mod.pin_list['p4'].domain
    assert mod.elem_dict['x']['device'] is child


def test_standalone_output_bytes_with_unemitted_domains():
    import hashlib
    plain = Generator().generate(CircuitSpec('x'))
    assert hashlib.sha256(plain.encode()).hexdigest() == (
        'b7b50b86933172e353a7ffe1df1b0ba9a96ac64b8fd5b6d9e9ad9f0a08584b6a')
    for name in ['AVDD', 'Module', 'range', 'pdk', 'class', 'unused-domain']:
        spec = CircuitSpec('x', voltage_domains={
            'key': g.VoltageDomain(name=name, nominal=1.8)})
        assert Generator().generate(spec) == plain


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('exists', [False, True])
def test_authored_variable_domain_collision_refuses_before_write(tmp_path, direct, exists):
    target = tmp_path / 'generated.py'
    if exists:
        target.write_text('keep me')
    spec = {
        'circuit_name': 'x', 'ports': {},
        'voltage_domains': {'AVDD': {'nominal': 1.8}},
        'variables': [{'name': 'AVDD', 'value': 1000}],
        'ideal_elements': [{'type': 'resistor', 'instance': 'r',
                            'nets': {'p': 'a', 'n': 'gnd'},
                            'params': {'value': {'variable': 'AVDD'}}}],
    }
    if direct:
        spec = CircuitSpec.from_dict(spec)
        spec.voltage_domains = {'different_key': spec.voltage_domains['AVDD']}
    with pytest.raises(ValueError, match='clashes with voltage domain'):
        Generator().generate_to_file(spec, str(target))
    assert target.read_text() == 'keep me' if exists else not target.exists()
