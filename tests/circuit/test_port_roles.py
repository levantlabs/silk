"""Tests for port-role inference (#6).

Organized by task:
  - Task 1: RoleRuleset core (classify + from_yaml)          [TestRoleRulesetClassify, TestRoleRulesetFromYaml]
  - Task 2: shipped default + packaging                       [TestRoleRulesetDefault, TestPackaging]
  - Task 3: infer_port_roles                                  [TestInferPortRoles]
"""
import os
import textwrap

import pytest

from silk.circuit.ports import PortRole


# ---------------------------------------------------------------------------
# Task 1 — RoleRuleset.classify
# ---------------------------------------------------------------------------

class TestRoleRulesetClassify:

    def test_glob_match(self):
        from silk.circuit.port_roles import RoleRuleset
        rs = RoleRuleset().add("*vdd*", PortRole.POWER)
        assert rs.classify("vdd") == PortRole.POWER
        assert rs.classify("avdd18") == PortRole.POWER

    def test_no_match_returns_none(self):
        from silk.circuit.port_roles import RoleRuleset
        rs = RoleRuleset().add("*vdd*", PortRole.POWER)
        assert rs.classify("vout") is None

    def test_empty_ruleset_returns_none(self):
        from silk.circuit.port_roles import RoleRuleset
        assert RoleRuleset().classify("vdd") is None

    def test_add_is_chainable(self):
        from silk.circuit.port_roles import RoleRuleset
        rs = (RoleRuleset()
              .add("*vdd*", PortRole.POWER)
              .add("*clk*", PortRole.CLOCK))
        assert rs.classify("vdd") == PortRole.POWER
        assert rs.classify("clk") == PortRole.CLOCK

    def test_classify_returns_portrole_enum(self):
        from silk.circuit.port_roles import RoleRuleset
        role = RoleRuleset().add("*vdd*", PortRole.POWER).classify("vdd")
        assert isinstance(role, PortRole)

    def test_case_insensitive_name_and_pattern(self):
        from silk.circuit.port_roles import RoleRuleset
        rs = RoleRuleset().add("*VDD*", PortRole.POWER)
        assert rs.classify("vdd") == PortRole.POWER
        assert rs.classify("AVDD_Core") == PortRole.POWER

    def test_none_name_returns_none(self):
        from silk.circuit.port_roles import RoleRuleset
        rs = RoleRuleset().add("*vdd*", PortRole.POWER)
        assert rs.classify(None) is None

    # --- most-specific-first (§5) ---

    def test_exact_literal_beats_wildcard(self):
        from silk.circuit.port_roles import RoleRuleset
        rs = (RoleRuleset()
              .add("*vss*", PortRole.GROUND)   # wildcard
              .add("vss", PortRole.ANALOG))     # exact literal — wins for 'vss'
        assert rs.classify("vss") == PortRole.ANALOG
        # a name only the wildcard reaches still classifies via the wildcard
        assert rs.classify("avss") == PortRole.GROUND

    def test_more_literal_chars_wins(self):
        from silk.circuit.port_roles import RoleRuleset
        rs = (RoleRuleset()
              .add("*v*", PortRole.ANALOG)      # 1 literal char
              .add("*vdd*", PortRole.POWER))     # 3 literal chars — more specific
        assert rs.classify("vdd") == PortRole.POWER

    def test_declaration_order_is_final_tiebreak(self):
        from silk.circuit.port_roles import RoleRuleset
        # equally-specific globs both match 'xy'; earlier declared wins
        rs = (RoleRuleset()
              .add("*x*", PortRole.CLOCK)
              .add("*y*", PortRole.DIGITAL))
        assert rs.classify("xy") == PortRole.CLOCK


# ---------------------------------------------------------------------------
# Task 1 — RoleRuleset.from_yaml
# ---------------------------------------------------------------------------

class TestRoleRulesetFromYaml:

    def _write(self, tmp_path, text):
        p = tmp_path / "rules.yaml"
        p.write_text(textwrap.dedent(text))
        return str(p)

    def test_valid_load_glob(self, tmp_path):
        from silk.circuit.port_roles import RoleRuleset
        path = self._write(tmp_path, """
            version: 1
            rules:
              - {match: "*vdd*", role: power}
              - {match: "*clk*", role: clock}
        """)
        rs = RoleRuleset.from_yaml(path)
        assert rs.classify("vdd") == PortRole.POWER
        assert rs.classify("clk") == PortRole.CLOCK
        assert rs.classify("vout") is None

    def test_regex_rule_matches(self, tmp_path):
        from silk.circuit.port_roles import RoleRuleset
        path = self._write(tmp_path, r"""
            version: 1
            rules:
              - {regex: "^d\\[[0-9]+\\]$", role: digital}
        """)
        rs = RoleRuleset.from_yaml(path)
        assert rs.classify("d[3]") == PortRole.DIGITAL
        assert rs.classify("dbus") is None

    def test_regex_with_metachars_is_not_exact(self, tmp_path):
        # §5/N2: a regex with metachars (even zero * / ?) must NOT rank as exact.
        # An exact-literal glob 'vdd' beats the anchored regex '^vdd$'.
        from silk.circuit.port_roles import RoleRuleset
        path = self._write(tmp_path, r"""
            version: 1
            rules:
              - {regex: "^vdd$", role: digital}
              - {match: "vdd", role: power}
        """)
        rs = RoleRuleset.from_yaml(path)
        assert rs.classify("vdd") == PortRole.POWER

    def test_unknown_role_raises(self, tmp_path):
        from silk.circuit.port_roles import RoleRuleset, RoleRulesetError
        path = self._write(tmp_path, """
            version: 1
            rules:
              - {match: "*vdd*", role: bogus}
        """)
        with pytest.raises(RoleRulesetError):
            RoleRuleset.from_yaml(path)

    def test_missing_pattern_key_raises(self, tmp_path):
        from silk.circuit.port_roles import RoleRuleset, RoleRulesetError
        path = self._write(tmp_path, """
            version: 1
            rules:
              - {role: power}
        """)
        with pytest.raises(RoleRulesetError):
            RoleRuleset.from_yaml(path)

    def test_dual_pattern_key_raises(self, tmp_path):
        from silk.circuit.port_roles import RoleRuleset, RoleRulesetError
        path = self._write(tmp_path, r"""
            version: 1
            rules:
              - {match: "*vdd*", regex: "^v", role: power}
        """)
        with pytest.raises(RoleRulesetError):
            RoleRuleset.from_yaml(path)

    def test_missing_role_key_raises(self, tmp_path):
        from silk.circuit.port_roles import RoleRuleset, RoleRulesetError
        path = self._write(tmp_path, """
            version: 1
            rules:
              - {match: "*vdd*"}
        """)
        with pytest.raises(RoleRulesetError):
            RoleRuleset.from_yaml(path)

    def test_bad_version_raises(self, tmp_path):
        from silk.circuit.port_roles import RoleRuleset, RoleRulesetError
        path = self._write(tmp_path, """
            version: 99
            rules:
              - {match: "*vdd*", role: power}
        """)
        with pytest.raises(RoleRulesetError):
            RoleRuleset.from_yaml(path)

    def test_missing_version_raises(self, tmp_path):
        from silk.circuit.port_roles import RoleRuleset, RoleRulesetError
        path = self._write(tmp_path, """
            rules:
              - {match: "*vdd*", role: power}
        """)
        with pytest.raises(RoleRulesetError):
            RoleRuleset.from_yaml(path)

    def test_invalid_regex_raises(self, tmp_path):
        from silk.circuit.port_roles import RoleRuleset, RoleRulesetError
        # an unterminated character class is a compile-time re.error
        path = self._write(tmp_path, r"""
            version: 1
            rules:
              - {regex: "d[0-9", role: digital}
        """)
        with pytest.raises(RoleRulesetError):
            RoleRuleset.from_yaml(path)

    def test_top_level_not_mapping_raises(self, tmp_path):
        from silk.circuit.port_roles import RoleRuleset, RoleRulesetError
        path = self._write(tmp_path, """
            - {match: "*vdd*", role: power}
        """)
        with pytest.raises(RoleRulesetError):
            RoleRuleset.from_yaml(path)

    def test_rules_not_list_raises(self, tmp_path):
        from silk.circuit.port_roles import RoleRuleset, RoleRulesetError
        path = self._write(tmp_path, """
            version: 1
            rules: "not a list"
        """)
        with pytest.raises(RoleRulesetError):
            RoleRuleset.from_yaml(path)

    def test_rule_not_mapping_raises(self, tmp_path):
        from silk.circuit.port_roles import RoleRuleset, RoleRulesetError
        path = self._write(tmp_path, """
            version: 1
            rules:
              - "just a string"
        """)
        with pytest.raises(RoleRulesetError):
            RoleRuleset.from_yaml(path)

    def test_error_subclasses_value_error(self):
        from silk.circuit.port_roles import RoleRulesetError
        assert issubclass(RoleRulesetError, ValueError)


# ---------------------------------------------------------------------------
# Task 2 — shipped default + packaging
# ---------------------------------------------------------------------------

class TestRoleRulesetDefault:

    def test_default_loads(self):
        from silk.circuit.port_roles import RoleRuleset
        assert RoleRuleset.default() is not None

    def test_default_seed_patterns_classify(self):
        from silk.circuit.port_roles import RoleRuleset
        rs = RoleRuleset.default()
        assert rs.classify("vdd") == PortRole.POWER
        assert rs.classify("vcc") == PortRole.POWER
        assert rs.classify("vss") == PortRole.GROUND
        assert rs.classify("gnd") == PortRole.GROUND
        assert rs.classify("0") == PortRole.GROUND
        assert rs.classify("clk") == PortRole.CLOCK
        assert rs.classify("clock") == PortRole.CLOCK
        # unmatched signal names stay None so #9 still flags them
        assert rs.classify("vout") is None
        assert rs.classify("vin") is None

    def test_default_returns_fresh_instance(self):
        # a fresh instance per call means .add() never mutates a shared singleton
        from silk.circuit.port_roles import RoleRuleset
        a = RoleRuleset.default()
        a.add("*vbias*", PortRole.ANALOG)
        b = RoleRuleset.default()
        assert b.classify("vbias") is None


class TestPackaging:
    """I3 guard (§7) — NOT a simulated wheel. Assert the YAML is discoverable
    at its ``__file__``-relative path AND the pyproject package-data entry
    exists, so the file ships in a wheel."""

    def test_default_yaml_discoverable(self):
        import silk.circuit.port_roles as pr
        assert os.path.isfile(pr._DEFAULT_YAML), pr._DEFAULT_YAML

    def test_pyproject_package_data_entry(self):
        import silk.circuit.port_roles as pr
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(pr.__file__))))
        pyproject = os.path.join(repo_root, "pyproject.toml")
        assert os.path.isfile(pyproject), pyproject
        # The glob hangs off the OWNING PACKAGE (silk.circuit) — data/ has no
        # __init__.py, so a "silk.circuit.data" package-data key silently
        # ships nothing (the pre-0.1.1 wheel bug this test used to encode).
        try:
            import tomllib
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            pkg_data = data["tool"]["setuptools"]["package-data"]
            assert "silk.circuit" in pkg_data
            assert any("data/" in pat and "yaml" in pat
                       for pat in pkg_data["silk.circuit"])
        except ImportError:
            with open(pyproject) as f:
                text = f.read()
            assert '"silk.circuit"' in text and 'data/*.yaml' in text


# ---------------------------------------------------------------------------
# Task 3 — infer_port_roles
# ---------------------------------------------------------------------------

def _pdk():
    from unittest.mock import MagicMock
    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}
    return pdk


def _leaf(tmp_path, name, ports):
    """A bare subcircuit Module with the given port definitions."""
    from silk.circuit.modules import Module
    m = Module('TEST', str(tmp_path), f'{name}.cir', pdk_details=_pdk())
    m.define_module_name(name)
    m.define_module_ports(ports)
    return m


class TestInferPortRoles:

    def test_fills_unset_roles(self, tmp_path):
        from silk.circuit.circuit_elements import PortDir
        from silk.circuit.port_roles import RoleRuleset, infer_port_roles
        leaf = _leaf(tmp_path, 'blk', {
            'vdd': PortDir.IN, 'gnd': PortDir.IN,
            'clk': PortDir.IN, 'vout': PortDir.OUT,
        })
        infer_port_roles(leaf, RoleRuleset.default())
        assert leaf.pin_list['vdd'].role == PortRole.POWER
        assert leaf.pin_list['gnd'].role == PortRole.GROUND
        assert leaf.pin_list['clk'].role == PortRole.CLOCK
        assert leaf.pin_list['vout'].role is None

    def test_sets_portrole_enum(self, tmp_path):
        from silk.circuit.circuit_elements import PortDir
        from silk.circuit.port_roles import RoleRuleset, infer_port_roles
        leaf = _leaf(tmp_path, 'blk', {'vdd': PortDir.IN})
        infer_port_roles(leaf, RoleRuleset.default())
        assert isinstance(leaf.pin_list['vdd'].role, PortRole)

    def test_never_clobbers_explicit_role(self, tmp_path):
        from silk.circuit.circuit_elements import PortDir
        from silk.circuit.port_roles import RoleRuleset, infer_port_roles
        # 'vdd' would classify POWER, but it is hand-set ANALOG — must survive.
        leaf = _leaf(tmp_path, 'blk', {
            'vdd': (PortDir.IN, PortRole.ANALOG),
            'gnd': PortDir.IN,
        })
        report = infer_port_roles(leaf, RoleRuleset.default())
        assert leaf.pin_list['vdd'].role == PortRole.ANALOG   # untouched
        assert leaf.pin_list['gnd'].role == PortRole.GROUND   # filled
        assert 'vdd' not in {n for n, _ in report.filled}
        assert 'vdd' not in report.unmatched                  # skipped, not unmatched

    def test_unmatched_stay_none(self, tmp_path):
        from silk.circuit.circuit_elements import PortDir
        from silk.circuit.port_roles import RoleRuleset, infer_port_roles
        leaf = _leaf(tmp_path, 'blk', {'vout': PortDir.OUT})
        infer_port_roles(leaf, RoleRuleset.default())
        assert leaf.pin_list['vout'].role is None

    def test_report_is_accurate(self, tmp_path):
        from silk.circuit.circuit_elements import PortDir
        from silk.circuit.port_roles import RoleRuleset, infer_port_roles
        leaf = _leaf(tmp_path, 'blk', {
            'vdd': PortDir.IN, 'clk': PortDir.IN,
            'vout': PortDir.OUT, 'vin': PortDir.IN,
        })
        report = infer_port_roles(leaf, RoleRuleset.default())
        assert dict(report.filled) == {
            'vdd': PortRole.POWER, 'clk': PortRole.CLOCK}
        assert set(report.unmatched) == {'vout', 'vin'}

    def test_module_without_ports_is_noop(self, tmp_path):
        from silk.circuit.modules import Module
        from silk.circuit.port_roles import RoleRuleset, infer_port_roles
        m = Module('TEST', str(tmp_path), 'x.cir', pdk_details=_pdk())
        report = infer_port_roles(m, RoleRuleset.default())
        assert report.filled == []
        assert report.unmatched == []

    # --- ordering contract (C1 regression guard, §4.3 / §7) ---

    def test_ordering_infer_before_add_element_passes_coverage(self, tmp_path):
        from silk.circuit.circuit_elements import PortDir
        from silk.circuit.modules import Module
        from silk.circuit.interface_check import require_annotation_coverage
        from silk.circuit.port_roles import RoleRuleset, infer_port_roles
        # annotate the leaf's OWN ports FIRST ...
        leaf = _leaf(tmp_path, 'blk', {
            'vdd': PortDir.IN, 'gnd': PortDir.IN, 'clk': PortDir.IN,
        })
        infer_port_roles(leaf, RoleRuleset.default())
        # ... THEN instantiate into a parent (deep-copies the annotated child).
        top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
        top.add_module(leaf, 'blk1',
                        nets={'vdd': 'vdd', 'gnd': 'gnd', 'clk': 'clk'})
        # coverage reads the deep copy — roles must have reached it.
        cov = require_annotation_coverage(top, min_pct=100.0)
        assert cov.coverage_pct == 100.0

    # --- inferred role flows to net classification when it precedes wiring (I1) ---

    def test_inferred_role_flows_to_net_classification(self, tmp_path):
        from silk.circuit.circuit_elements import PortDir
        from silk.circuit.modules import Module
        from silk.circuit.port_roles import RoleRuleset, infer_port_roles
        leaf = _leaf(tmp_path, 'blk', {'clk': PortDir.IN})
        infer_port_roles(leaf, RoleRuleset.default())   # role set BEFORE wiring
        top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
        top.add_module(leaf, 'blk1', nets={'clk': 'clk_top'})
        assert top.net_dict['clk_top'].classification == PortRole.CLOCK

    # --- de-hardcode proof at the injectable layer (I2, §7) ---

    def test_custom_ruleset_changes_outcome(self, tmp_path):
        from silk.circuit.circuit_elements import PortDir
        from silk.circuit.port_roles import RoleRuleset, infer_port_roles
        # the default does NOT classify 'vbias'
        a = _leaf(tmp_path, 'a', {'vbias': PortDir.IN})
        infer_port_roles(a, RoleRuleset.default())
        assert a.pin_list['vbias'].role is None
        # a custom ruleset does
        custom = RoleRuleset().add('*vbias*', PortRole.ANALOG)
        b = _leaf(tmp_path, 'b', {'vbias': PortDir.IN})
        infer_port_roles(b, custom)
        assert b.pin_list['vbias'].role == PortRole.ANALOG
        # ... and vice-versa: 'vdd' the default classifies, the custom omits
        c = _leaf(tmp_path, 'c', {'vdd': PortDir.IN})
        infer_port_roles(c, custom)
        assert c.pin_list['vdd'].role is None


# ---------------------------------------------------------------------------
# Task 5 — public exports
# ---------------------------------------------------------------------------

class TestExports:

    def test_exported_from_silk_circuit(self):
        import silk.circuit as c
        assert c.RoleRuleset is not None
        assert c.infer_port_roles is not None
        assert "RoleRuleset" in c.__all__
        assert "infer_port_roles" in c.__all__

    def test_exported_from_silk(self):
        import silk
        assert silk.RoleRuleset is not None
        assert silk.infer_port_roles is not None
        assert "RoleRuleset" in silk.__all__
        assert "infer_port_roles" in silk.__all__

    def test_same_object_across_namespaces(self):
        import silk
        import silk.circuit
        from silk.circuit.port_roles import RoleRuleset, infer_port_roles
        assert silk.RoleRuleset is RoleRuleset
        assert silk.circuit.RoleRuleset is RoleRuleset
        assert silk.infer_port_roles is infer_port_roles


# ---------------------------------------------------------------------------
# CLI — python -m silk.circuit.port_roles
# ---------------------------------------------------------------------------

class TestCLI:
    """The ruleset checker: validate (exit 0/1/2) + optional dry-run."""

    def _write(self, tmp_path, text):
        p = tmp_path / "rules.yaml"
        p.write_text(textwrap.dedent(text))
        return str(p)

    def test_len_counts_rules(self):
        from silk.circuit.port_roles import RoleRuleset
        rs = RoleRuleset().add("*vdd*", PortRole.POWER).add("*clk*", PortRole.CLOCK)
        assert len(rs) == 2

    def test_default_ruleset_ok(self, capsys):
        from silk.circuit.port_roles import main
        assert main([]) == 0
        out = capsys.readouterr().out
        assert out.startswith("OK:")
        assert "rule(s)" in out

    def test_valid_file_exit_0(self, tmp_path, capsys):
        from silk.circuit.port_roles import main
        path = self._write(tmp_path, """
            version: 1
            rules:
              - {match: "*vdd*", role: power}
        """)
        assert main(["--file", path]) == 0
        assert "OK:" in capsys.readouterr().out

    def test_invalid_content_exit_1(self, tmp_path, capsys):
        from silk.circuit.port_roles import main
        path = self._write(tmp_path, """
            version: 1
            rules:
              - {match: "*vdd*", role: bogus}
        """)
        assert main(["--file", path]) == 1
        assert "INVALID" in capsys.readouterr().err

    def test_missing_file_exit_2(self, tmp_path, capsys):
        from silk.circuit.port_roles import main
        assert main(["--file", str(tmp_path / "nope.yaml")]) == 2
        assert "cannot read" in capsys.readouterr().err

    def test_names_dry_run(self, capsys):
        from silk.circuit.port_roles import main
        assert main(["--names", "vdd", "vout"]) == 0
        out = capsys.readouterr().out
        assert "vdd" in out and "power" in out
        assert "vout" in out and "(unmatched)" in out

    def test_rolecheck_shim_exposes_main(self):
        # the runnable entry (python -m silk.circuit.rolecheck) delegates here
        from silk.circuit import rolecheck
        from silk.circuit.port_roles import main
        assert rolecheck.main is main


# ---------------------------------------------------------------------------
# TASK-42 — provenance of an inferred role, and the ordering footgun made loud
# ---------------------------------------------------------------------------

class TestInferredRoleProvenance:

    def test_inferred_role_is_marked_and_declared_is_not(self, tmp_path):
        """Pre-fix there was no provenance at all: ``ModulePin.role`` and
        ``PortBinding.role`` were a bare value, so a '*vdd*' guess was
        byte-identical to ``Port.supply()`` once wired. Now ``role_inferred``
        is True only for a role this call filled — False for an explicit
        role, a factory role, and the TASK-45 direction-implied one — and
        the mark reaches the instance copy and ``Net.pins``."""
        from silk.circuit.circuit_elements import PortDir, Port
        from silk.circuit.modules import Module
        from silk.circuit.port_roles import RoleRuleset, infer_port_roles
        leaf = _leaf(tmp_path, 'blk', {
            'vdd': PortDir.IN,                    # -> inferred POWER
            'avdd': Port.supply(),                # declared
            'vss': PortDir.GROUND,                # direction-implied (TASK-45)
            'vin': PortDir.IN,                    # unmatched, stays None
        })
        assert all(pin.role_inferred is False for pin in leaf.pin_list.values())
        infer_port_roles(leaf, RoleRuleset.default())
        assert {n: pin.role_inferred for n, pin in leaf.pin_list.items()} == {
            'vdd': True, 'avdd': False, 'vss': False, 'vin': False}

        top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
        top.add_module(leaf, 'u1', nets={'vdd': 'p', 'avdd': 'q',
                                          'vss': 'g', 'vin': 'i'})
        copy_pins = top.elem_dict['u1']['device'].pin_list
        assert copy_pins['vdd'].role_inferred is True
        assert copy_pins['avdd'].role_inferred is False
        by_net = {n: net.pins[0] for n, net in top.net_dict.items()}
        assert (by_net['p'].role, by_net['p'].role_inferred) == ('power', True)
        assert (by_net['q'].role, by_net['q'].role_inferred) == ('power', False)
        assert (by_net['g'].role, by_net['g'].role_inferred) == ('ground', False)
        assert (by_net['i'].role, by_net['i'].role_inferred) == (None, False)

    def test_non_submodule_bindings_carry_no_provenance(self, tmp_path):
        """Source / passive / device bindings have no role to infer; their
        ``role_inferred`` is the default False."""
        from silk.circuit.modules import Module
        from silk.sources.classes import VoltageSource
        top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
        top.add_source(VoltageSource('v1', positive='a', negative='b', dc=1.0))
        assert [(b.kind, b.role_inferred) for b in top.net_dict['a'].pins] == [
            ('source', False)]

    def test_infer_after_placement_reaches_both_instances(self, tmp_path):
        from silk.circuit.circuit_elements import PortDir
        from silk.circuit.modules import Module
        from silk.circuit.port_roles import RoleRuleset, infer_port_roles
        leaf = _leaf(tmp_path, 'blk', {'vdd': PortDir.IN})
        top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
        top.add_module(leaf, 'u1', nets={'vdd': 'rail1'})
        top.add_module(leaf, 'u2', nets={'vdd': 'rail2'})
        report = infer_port_roles(leaf, RoleRuleset.default())
        assert report.filled == [('vdd', PortRole.POWER)]
        for rail in ('rail1', 'rail2'):
            assert {b.role for b in top.net_dict[rail].pins} == {'power'}
            assert {b.role_inferred for b in top.net_dict[rail].pins} == {True}

    def test_infer_through_public_instance_walk_reaches_parent(self, tmp_path):
        from silk.circuit.circuit_elements import PortDir
        from silk.circuit.modules import Module
        from silk.circuit.port_roles import RoleRuleset, infer_port_roles
        leaf = _leaf(tmp_path, 'blk', {'vdd': PortDir.IN, 'gnd': PortDir.IN})
        top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
        top.add_module(leaf, 'u1', nets={'vdd': 'vdd', 'gnd': 'gnd'})
        for instance in top.iter_instances():
            infer_port_roles(instance.module, RoleRuleset.default())
        assert {b.role for b in top.net_dict['vdd'].pins} == {'power'}
        assert {b.role for b in top.net_dict['gnd'].pins} == {'ground'}

    def test_infer_before_add_element_does_not_warn(self, tmp_path):
        """The supported order stays quiet, including a second call on a
        never-instantiated definition."""
        import warnings
        from silk.circuit.circuit_elements import PortDir
        from silk.circuit.port_roles import RoleRuleset, infer_port_roles
        leaf = _leaf(tmp_path, 'blk', {'vdd': PortDir.IN, 'vin': PortDir.IN})
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            infer_port_roles(leaf, RoleRuleset.default())
            infer_port_roles(leaf, RoleRuleset().add('*vin*', PortRole.ANALOG))
        assert leaf.pin_list['vin'].role == PortRole.ANALOG
