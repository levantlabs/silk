"""Catalog composition guards (TASK-49).

Pins what "the shipped rule set" means: DEFAULT_CATALOG is what
RuleRegistry.default() runs, OPTIONAL_CATALOG holds the opt-in rules,
help_rule resolves every one of them, and the user guide's rule tables
match the code exactly.
"""
import re
from pathlib import Path

import pytest

from silk.circuit.circuit_elements import PortDir, PortRole
from silk.circuit.modules import Module
from silk.lint.help import help_rule, list_rules
from silk.lint.registry import RuleRegistry
from silk.lint.rules import DEFAULT_CATALOG, FULL_CATALOG, OPTIONAL_CATALOG

_GUIDE = Path(__file__).resolve().parents[2] / 'docs' / 'lint_and_coverage_user_guide.md'
_ROW = re.compile(r'^\| ((?:INT|DOM|STR|STRUCT|BUS)-\d{3}) \| (\S+) \| (\S+) \| (\S+) \| (\S+) \|$')


def _ids(catalog):
    return [r.rule_id for r in catalog]


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def test_optional_catalog_is_struct_009_only():
    """STRUCT-010 left the opt-in catalog in TASK-94 (regulated-output
    exemption); STRUCT-009 stays there by the TASK-93 ruling."""
    assert _ids(OPTIONAL_CATALOG()) == ['STRUCT-009']
    assert 'STRUCT-010' in _ids(DEFAULT_CATALOG())


def test_full_catalog_is_default_plus_optional_without_overlap():
    default, optional, full = _ids(DEFAULT_CATALOG()), _ids(OPTIONAL_CATALOG()), _ids(FULL_CATALOG())
    assert full == default + optional
    assert len(set(full)) == len(full)
    assert not set(default) & set(optional)


def test_registry_default_flag_adds_the_optional_rules():
    assert _ids(RuleRegistry.default().rules) == _ids(DEFAULT_CATALOG())
    assert _ids(RuleRegistry.default(include_optional=True).rules) == _ids(FULL_CATALOG())


# ---------------------------------------------------------------------------
# help_rule / list_rules see the full shipped set
# ---------------------------------------------------------------------------

def test_help_rule_resolves_every_implemented_rule_id():
    for rule in FULL_CATALOG():
        text = help_rule(rule.rule_id)
        assert rule.rule_id in text and rule.rationale in text


def test_help_rule_reports_default_membership():
    assert 'In default catalog: yes' in help_rule('INT-001')
    assert 'In default catalog: yes' in help_rule('STRUCT-010')
    text = help_rule('STRUCT-009')
    assert 'In default catalog: no (opt-in' in text
    assert 'include_optional=True' in text


def test_help_rule_still_raises_for_unshipped_ids():
    # STRUCT-011 SINGLE_CONNECTION_NET ships as of TASK-35, STRUCT-012
    # UNPORTABLE_GROUND_PORT as of TASK-36 and STRUCT-013 HIDDEN_SUPPLY_NET
    # as of TASK-93; STRUCT-014 is the next free structural id and stands in
    # as the unshipped one.
    with pytest.raises(KeyError, match='STRUCT-014'):
        help_rule('STRUCT-014')


def test_list_rules_default_vs_full():
    assert list_rules() == _ids(DEFAULT_CATALOG())
    assert list_rules(include_optional=True) == _ids(FULL_CATALOG())


# ---------------------------------------------------------------------------
# The documented opt-in recipe works on a real module
# ---------------------------------------------------------------------------

def test_opt_in_recipe_runs_struct_009_on_a_device_block(pdk, tmp_path):
    blk = Module('PDK_skywater130', str(tmp_path), 'nosupply.cir',
                 pdk_details=pdk, module_name='nosupply')
    blk.define_module_ports({'out': (PortDir.OUT, PortRole.ANALOG),
                             'in': (PortDir.IN, PortRole.ANALOG)})
    with blk:
        pdk.nmos1p8('mn', d='out', g='in', s='gnd_int', b='gnd_int', w=1.0, l=0.15)

    default_ids = {v.rule_id for v in RuleRegistry.default().run(blk).violations}
    assert 'STRUCT-009' not in default_ids
    # The block also hides its ground rail on 'gnd_int', which is the half
    # of STRUCT-009's original job the default catalog DOES do (STRUCT-013,
    # TASK-93). The opt-in half is the port-declaration demand below.
    assert 'STRUCT-013' in default_ids

    opted = RuleRegistry.default(include_optional=True).run(blk)
    visited = {e.rule_id for e in opted.visit_log}
    assert 'STRUCT-009' in visited
    assert 'STRUCT-010' in visited  # default since TASK-94, so here too
    struct009 = [v for v in opted.violations if v.rule_id == 'STRUCT-009']
    assert len(struct009) == 2  # no POWER port, no GROUND port


# ---------------------------------------------------------------------------
# User guide tables == code
# ---------------------------------------------------------------------------

def _guide_tables():
    """Return {section_title: [(id, name, category, severity, needs_sim)]}."""
    tables, current = {}, None
    for line in _GUIDE.read_text().splitlines():
        if line.startswith('### '):
            current = line[4:].strip()
            continue
        m = _ROW.match(line)
        if m and current is not None:
            tables.setdefault(current, []).append(m.groups())
    return tables


def _expected(catalog):
    return [(r.rule_id, r.category, r.default_severity.to_string().upper(),
             'yes' if r.requires_simulation else 'no') for r in catalog]


def test_user_guide_default_table_matches_default_catalog_exactly():
    rows = _guide_tables()['Default catalog']
    got = [(rid, cat, sev, sim) for rid, _name, cat, sev, sim in rows]
    assert sorted(got) == sorted(_expected(DEFAULT_CATALOG()))


def test_user_guide_opt_in_table_matches_optional_catalog_exactly():
    rows = _guide_tables()['Opt-in rules']
    got = [(rid, cat, sev, sim) for rid, _name, cat, sev, sim in rows]
    assert sorted(got) == sorted(_expected(OPTIONAL_CATALOG()))
