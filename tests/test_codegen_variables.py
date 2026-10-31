"""Module variables in the CircuitSpec (TASK-376).

A spec declares ``variables`` ({name, value}, declaration order) and any
numeric device, element or source parameter may hold ``{"variable": name}``
or ``{"variable": name, "scale": number}``. The generator emits one
``Variable`` per declaration and the reference as ``w_in`` or
``2.0 * w_in``; changing a variable's value moves every referencing line of
the regenerated netlist.
"""

import re

import pytest

from silk import Variable
from silk.codegen.generator import CircuitSpec, Generator
from silk.codegen.testbench_generator import (
    TestbenchSpecError, generate_testbenches)


def _spec(tmp_path=None, **extra):
    """A module-mode spec covering every kind that takes a reference."""
    spec = {
        "circuit_name": "shared",
        "pdk": "sky130",
        "output_dir": str(tmp_path) if tmp_path else "./sim",
        "ports": {"a": {"dir": "inout"}},
        "variables": [
            {"name": "w_in", "value": 2.0},
            {"name": "r_load", "value": 1000.0},
            {"name": "v_bias", "value": 1.0},
        ],
        "devices": [
            {"type": "nmos1p8", "instance": "m1",
             "nets": {"d": "a", "g": "b", "s": "gnd", "b": "gnd"},
             "params": {"w": {"variable": "w_in"}, "l": 0.15}},
            {"type": "nmos1p8", "instance": "m2",
             "nets": {"d": "a", "g": "b", "s": "gnd", "b": "gnd"},
             "params": {"w": {"variable": "w_in", "scale": 2}, "l": 0.15}},
            {"type": "nmos1p8", "instance": "ma", "array": 2,
             "nets": {"d": "a", "g": "b", "s": "gnd", "b": "gnd"},
             "params": {"w": {"variable": "w_in"}, "l": 0.15}},
        ],
        "ideal_elements": [
            {"type": "resistor", "instance": "r1",
             "nets": {"p": "a", "n": "b"},
             "params": {"value": {"variable": "r_load"}}},
            {"type": "capacitor", "instance": "c1",
             "nets": {"p": "a", "n": "gnd"},
             "params": {"value": {"variable": "v_bias", "scale": 1e-12}}},
        ],
        "sources": [
            {"type": "VoltageSource", "instance": "vs",
             "positive": "a", "negative": "gnd",
             "dc": {"variable": "v_bias"}, "ac": {"variable": "v_bias"}},
            {"type": "VoltageSource", "instance": "vp",
             "positive": "b", "negative": "gnd",
             "waveform": {"type": "Pulse", "params": {
                 "v1": 0, "v2": {"variable": "v_bias"}, "tr": 1e-9,
                 "tf": 1e-9, "pw": 1e-8, "per": 2e-8}}},
            {"type": "VoltageSource", "instance": "vw",
             "positive": "c", "negative": "gnd",
             "pwl": [0, 0, 1e-9, {"variable": "v_bias"}]},
            {"type": "VCVS", "instance": "e1", "out_pos": "d",
             "out_neg": "gnd", "ctrl_pos": "a", "ctrl_neg": "gnd",
             "gain": {"variable": "v_bias", "scale": 3}},
        ],
    }
    spec.update(extra)
    return spec


def _run(code):
    ns = {}
    exec(compile(code, "<generated>", "exec"), ns)
    return ns


def _card(netlist, prefix):
    lines = [l for l in netlist.splitlines() if l.startswith(prefix)]
    assert len(lines) == 1, (prefix, lines)
    return lines[0]


class TestEmission:
    def test_emission_golden(self):
        spec = {
            "circuit_name": "pair",
            "pdk": "sky130",
            "ports": {"a": {"dir": "inout"}},
            "variables": [{"name": "w_in", "value": 2.0},
                          {"name": "l_in", "value": 0.15}],
            "devices": [
                {"type": "nmos1p8", "instance": "m1",
                 "nets": {"d": "a", "g": "g", "s": "gnd", "b": "gnd"},
                 "params": {"w": {"variable": "w_in"},
                            "l": {"variable": "l_in"}}},
                {"type": "nmos1p8", "instance": "m2",
                 "nets": {"d": "a", "g": "g", "s": "gnd", "b": "gnd"},
                 "params": {"w": {"variable": "w_in", "scale": 1},
                            "l": {"variable": "l_in"}}},
            ],
        }
        assert Generator().generate(spec) == (
            '"""Generated subcircuit: pair"""\n'
            '\n'
            'from silk import Module\n'
            'from silk.circuit.circuit_elements import PortDir\n'
            'from silk import Variable\n'
            '\n'
            '# --- PDK setup ---\n'
            'from silk import load_pdk\n'
            "pdk = load_pdk('sky130')\n"
            '\n'
            '# --- module variables ---\n'
            "w_in = Variable('w_in', 2.0)\n"
            "l_in = Variable('l_in', 0.15)\n"
            '\n'
            '# --- module definition ---\n'
            "mod = Module('PDK_sky130', './sim', 'pair.cir', pdk_details=pdk,\n"
            "             module_name='pair')\n"
            "mod.define_module_ports({'a': PortDir.INOUT})\n"
            '\n'
            '# --- PDK devices ---\n'
            "mod.add(pdk.nmos1p8('m1', d='a', g='g', s='gnd', b='gnd', "
            "w=w_in, l=l_in))\n"
            "mod.add(pdk.nmos1p8('m2', d='a', g='g', s='gnd', b='gnd', "
            "w=1.0 * w_in, l=l_in))\n"
        )

    def test_standalone_declares_before_the_module(self):
        spec = _spec()
        del spec["ports"]
        code = Generator().generate(spec)
        lines = code.splitlines()
        assert "from silk import Variable" in lines
        block = lines.index("# --- module variables ---")
        assert lines[block + 1:block + 4] == [
            "w_in = Variable('w_in', 2.0)",
            "r_load = Variable('r_load', 1000.0)",
            "v_bias = Variable('v_bias', 1.0)",
        ]
        assert block < lines.index("# --- circuit ---")

    def test_reference_forms_in_every_slot(self):
        code = Generator().generate(_spec())
        assert "w=w_in, l=0.15" in code
        assert "w=2.0 * w_in, l=0.15" in code
        assert "for i in range(2)" in code and "w=w_in" in code
        assert "value=r_load" in code
        assert "value=1e-12 * v_bias" in code
        assert "value=v_bias, ac=v_bias" in code
        assert "v2=v_bias" in code
        assert "points=[0, 0, 1e-09, v_bias]" in code
        assert "gain=3.0 * v_bias" in code

    def test_specs_without_variables_emit_no_variable_code(self):
        spec = _spec()
        for key in ("variables", "devices", "ideal_elements", "sources"):
            spec.pop(key)
        code = Generator().generate(spec)
        assert "Variable" not in code
        assert "module variables" not in code


class TestExecutedRegeneration:
    """AC #4: build, change one value, regenerate, every referrer follows."""

    def test_every_referencing_line_follows_a_value_change(self, tmp_path):
        ns = _run(Generator().generate(_spec(tmp_path)))
        mod = ns["mod"]
        before = mod.generate_netlist()
        assert "w=2 " in _card(before, "XMm1 ")
        assert "w=4 " in _card(before, "XMm2 ")
        assert "w=2 " in _card(before, "XMma<0> ")
        assert "w=2 " in _card(before, "XMma<1> ")

        ns["w_in"].value = 3.5
        ns["r_load"].value = 2000.0
        ns["v_bias"].value = 0.5
        after = mod.generate_netlist()

        assert "w=3.5 " in _card(after, "XMm1 ")
        assert "w=7 " in _card(after, "XMm2 ")
        assert "w=3.5 " in _card(after, "XMma<0> ")
        assert "w=3.5 " in _card(after, "XMma<1> ")
        assert _card(before, "Rr1 ").endswith(" 1k")
        assert _card(after, "Rr1 ").endswith(" 2k")
        assert _card(before, "Cc1 ").endswith(" 1p")
        assert _card(after, "Cc1 ").endswith(" 500f")
        assert _card(after, "Vvs ") == "Vvs a gnd DC 0.5 AC 0.5"
        assert "PULSE(0 0.5 " in _card(after, "Vvp__silk_wave ")
        assert _card(after, "Vvw__silk_pwl ").endswith("PWL(0.0 0.0 1e-09 0.5)")
        assert _card(before, "Ee1 ") == "Ee1 d gnd a gnd 3.0"
        assert _card(after, "Ee1 ") == "Ee1 d gnd a gnd 1.5"

    def test_variables_lists_declared_variables(self, tmp_path):
        ns = _run(Generator().generate(_spec(tmp_path)))
        found = ns["mod"].variables()
        assert all(isinstance(v, Variable) for v in found)
        assert {v.name for v in found} == {"w_in", "r_load", "v_bias"}
        # They are the generated block's objects, not copies.
        assert {id(v) for v in found} == {
            id(ns[n]) for n in ("w_in", "r_load", "v_bias")}


def _error(spec):
    with pytest.raises(ValueError) as exc:
        Generator().generate(spec)
    return str(exc.value)


class TestErrors:
    def test_unknown_variable_names_entry_and_param(self):
        spec = _spec()
        spec["devices"][1]["params"]["w"] = {"variable": "w_inn"}
        msg = _error(spec)
        assert "device 'm2' param 'w' references undefined variable 'w_inn'" in msg

    def test_unknown_variable_in_source_waveform(self):
        spec = _spec()
        spec["sources"][1]["waveform"]["params"]["v2"] = {"variable": "nope"}
        assert ("source 'vp' param 'waveform v2' references undefined "
                "variable 'nope'") in _error(spec)

    def test_reference_without_any_declaration(self):
        spec = _spec()
        del spec["variables"]
        assert "references undefined variable 'w_in'" in _error(spec)

    def test_duplicate_name(self):
        spec = _spec()
        spec["variables"].append({"name": "w_in", "value": 1.0})
        assert "variable 'w_in': declared more than once" in _error(spec)

    @pytest.mark.parametrize("name", ["2w", "w-in", "class", "", 5])
    def test_non_identifier_name(self, name):
        spec = _spec()
        spec["variables"].append({"name": name, "value": 1.0})
        assert "must be a valid Python identifier" in _error(spec)

    @pytest.mark.parametrize("name", ["pdk", "mod", "sources", "elements",
                                      "Variable", "i", "range"])
    def test_name_clashing_with_a_generated_local(self, name):
        spec = _spec()
        spec["variables"].append({"name": name, "value": 1.0})
        spec["devices"][0]["params"]["m"] = {"variable": name}
        assert "clashes with a name the generated code binds" in _error(spec)

    def test_name_clashing_with_a_voltage_domain(self):
        spec = _spec(voltage_domains={"AVDD": {"nominal": 1.8}})
        spec["variables"].append({"name": "AVDD", "value": 1.8})
        spec["sources"][0]["dc"] = {"variable": "AVDD"}
        assert "clashes with voltage domain 'AVDD'" in _error(spec)

    def test_name_clashing_with_a_submodule_alias(self):
        spec = _spec(submodules=[{
            "instance_name": "x1",
            "module_ref": {"import_path": ".inv", "module_var": "mod"},
            "nets": {"a": "a"}}])
        spec["variables"].append({"name": "mod_inv", "value": 1.0})
        spec["devices"][0]["params"]["m"] = {"variable": "mod_inv"}
        assert "clashes with a generated submodule import alias" in _error(spec)

    @pytest.mark.parametrize("value", ["2u", None, True, float("nan"), [1]])
    def test_non_numeric_value(self, value):
        spec = _spec()
        spec["variables"][0]["value"] = value
        assert "must be a finite int or float" in _error(spec)

    @pytest.mark.parametrize("scale", ["2", None, False, float("inf")])
    def test_non_numeric_scale(self, scale):
        spec = _spec()
        spec["devices"][1]["params"]["w"]["scale"] = scale
        msg = _error(spec)
        assert "device 'm2' param 'w' scales variable 'w_in'" in msg

    def test_malformed_reference_object(self):
        spec = _spec()
        spec["devices"][0]["params"]["w"] = {"variable": "w_in", "offset": 1}
        assert "device 'm1' param 'w' must be a number or a variable reference" \
            in _error(spec)

    def test_expression_string_is_not_parsed(self):
        spec = _spec()
        spec["devices"][1]["params"]["w"] = "2 * w_in"
        code = Generator().generate(spec)
        # A string is an ordinary literal, never an expression.
        assert "w='2 * w_in'" in code

    def test_unused_variable_is_an_error(self):
        spec = _spec()
        spec["variables"].append({"name": "spare", "value": 1.0})
        assert "variable(s) ['spare'] declared but not referenced" in _error(spec)

    def test_variable_entry_shape(self):
        spec = _spec()
        spec["variables"][0]["bounds"] = [1, 4]
        with pytest.raises(ValueError, match="unknown key"):
            CircuitSpec.from_dict(spec)
        with pytest.raises(ValueError, match="must be a list"):
            CircuitSpec.from_dict(_spec(variables={"w_in": 2.0}))


class TestTestbenchSpecRejectsReferences:
    """TestbenchSpec declares no variables, so a reference is refused rather
    than emitted as an undefined name."""

    REF = {"variable": "v_bias"}

    def _spec(self, test=None, rig=None):
        t = {"name": "t", "dut_portmap": {"a": "a"},
             "applied_rigs": ["r"] if rig else []}
        t.update(test or {})
        return {"circuit_name": "c", "tests": [t],
                "rigs": [dict({"name": "r"}, **rig)] if rig else []}

    def test_source_reference_refused(self):
        src = {"type": "VoltageSource", "instance": "vs", "positive": "a",
               "negative": "0", "dc": self.REF}
        with pytest.raises(TestbenchSpecError, match="CircuitSpec feature"):
            generate_testbenches(self._spec({"sources": [src]}))

    def test_device_reference_refused(self):
        dev = {"type": "nmos1p8", "instance": "M1",
               "nets": {"d": "y", "g": "a", "s": "0", "b": "0"},
               "params": {"w": self.REF, "l": 0.15}}
        with pytest.raises(TestbenchSpecError,
                           match="entry 'M1' param 'w'.*CircuitSpec feature"):
            generate_testbenches(self._spec(rig={"devices": [dev]}))

    def test_element_reference_refused(self):
        res = {"type": "resistor", "instance": "r1",
               "nets": {"p": "y", "n": "0"}, "params": {"value": self.REF}}
        with pytest.raises(TestbenchSpecError, match="CircuitSpec feature"):
            generate_testbenches(self._spec({"ideal_elements": [res]}))


class TestContractDocExample:
    """docs/codegen-contract.md ``variables`` worked example."""

    def _doc_spec(self, tmp_path):
        return {
            "circuit_name": "pair", "pdk": "sky130",
            "output_dir": str(tmp_path), "ports": {"a": {"dir": "inout"}},
            "variables": [{"name": "w_in", "value": 2.0}],
            "devices": [
                {"type": "nmos1p8", "instance": "m1",
                 "nets": {"d": "a", "g": "inp", "s": "tail", "b": "gnd"},
                 "params": {"w": {"variable": "w_in"}, "l": 0.15}},
                {"type": "nmos1p8", "instance": "m2",
                 "nets": {"d": "b", "g": "inn", "s": "tail", "b": "gnd"},
                 "params": {"w": {"variable": "w_in", "scale": 2.0},
                            "l": 0.15}},
            ],
        }

    def test_doc_example_emits_and_follows(self, tmp_path):
        code = Generator().generate(self._doc_spec(tmp_path))
        assert "w_in = Variable('w_in', 2.0)" in code
        assert "w=w_in, l=0.15" in code and "w=2.0 * w_in, l=0.15" in code
        ns = _run(code)
        ns["w_in"].value = 1.5
        netlist = ns["mod"].generate_netlist()
        assert "w=1.5 " in _card(netlist, "XMm1 ")
        assert "w=3 " in _card(netlist, "XMm2 ")

    def test_scaled_only_variable_is_listed_in_variables(self, tmp_path):
        spec = self._doc_spec(tmp_path)
        spec["devices"].pop(0)
        ns = _run(Generator().generate(spec))
        assert ns["mod"].variables() == [ns["w_in"]]

    def test_variables_lists_nested_expression_once_in_first_use_order(self):
        from silk.circuit.variable import Variable
        from silk.circuit.modules import Module
        from silk.pdk import load_pdk
        pdk = load_pdk("sky130")
        a, b = Variable("a", 1.0), Variable("b", 2.0)
        mod = Module("PDK_skywater130", "/tmp", "v.cir", pdk_details=pdk,
                     module_name="v")
        mod.add(pdk.nmos1p8("m1", d="d", g="g", s="s", b="s",
                            w=(2 * b + a) / 4, l=0.35))
        mod.add(pdk.nmos1p8("m2", d="d", g="g", s="s", b="s",
                            w=a, l=0.35))
        assert mod.variables() == [b, a]

    def test_variables_searches_placed_children(self):
        from silk.circuit.variable import Variable
        from silk.circuit.modules import Module
        from silk.circuit.ports import Port
        from silk.pdk import load_pdk
        pdk = load_pdk("sky130")
        w = Variable("w", 1.0)
        child = Module("PDK_skywater130", "/tmp", "c.cir", pdk_details=pdk,
                       module_name="c")
        child.define_module_ports({"G": Port.analog_input(),
                                   "D": Port.analog_output(),
                                   "S": Port.ground()})
        child.add(pdk.nmos1p8("m1", d="D", g="G", s="S", b="S", w=2 * w,
                              l=0.35))
        assert child.variables() == [w]
        parent = Module("PDK_skywater130", "/tmp", "p.cir", pdk_details=pdk)
        parent.add_module(child, "x1", nets={"G": "a", "D": "b", "S": "0"})
        assert parent.variables() == [w]
