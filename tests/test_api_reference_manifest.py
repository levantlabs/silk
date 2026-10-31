"""The API-reference surface manifest covers what docs/api-reference.md documents.

TASK-305.1 (A1). The generated reference is driven by
``scripts/api_surface.yaml``; until the owner has reviewed it, the hand-written
``docs/api-reference.md`` stays the reference of record, so every symbol it
documents has to be a manifest row or the generated one would silently drop it.

Fast suite: no PDK models, no simulator.
"""
import importlib
import re
from pathlib import Path

import pytest

from scripts import docstring_audit as audit

REPO = Path(__file__).resolve().parent.parent
API_REFERENCE = REPO / 'docs' / 'api-reference.md'

# Rows the manifest reaches through its `include:` list (names exported by no
# package's __all__ but documented in docs/api-reference.md).
INCLUDED = (
    'silk.simulation.behavioral.BehavioralEngine',
    'silk.circuit.report_diff.LayoutDiff',
    'silk.circuit.report_diff.LayoutNetDelta',
    'silk.simulation.backend.SimulatorBackend',
    'silk.codegen.generator.SourceSpec',
    'silk.lint.stress_coverage.StressCoverage',
    'silk.circuit.modules.SubcircuitInstance',
    'silk.codegen.testbench_generator.TestbenchSpec',
    'silk.circuit.junction_check.junction_stress_coverage',
    'silk.db.circuit_db',
    'silk.specs.reconciler.reconcile',
    'silk.specs.resolver.resolve_metric',
    'silk.circuit.module_placement.ModuleInstance',
    'silk.circuit.modules_testbench.TestEntry',
    'silk.circuit.monte_carlo.derive_sample_seed',
    'silk.circuit.monte_carlo.wilson_interval',
    'silk.simulation.backend.SpiceBackend',
)


@pytest.fixture(scope='module')
def entries():
    return audit.collect(audit.load_manifest())


def _documented_tokens():
    """Identifiers the hand-written reference documents.

    Backticked identifiers in headings, and every name in its import block.
    """
    text = API_REFERENCE.read_text(encoding='utf-8')
    tokens = set()
    for line in text.splitlines():
        if line.startswith('#'):
            for tok in re.findall(r'`([^`]+)`', line):
                tok = re.sub(r'\(.*', '', tok).strip()
                if re.fullmatch(r'[A-Za-z_][\w.]*', tok):
                    # ``module.x`` is an instance of Module in the prose
                    tokens.add(re.sub(r'^module\.', 'Module.', tok))
    block = re.search(r'from silk import \((.*?)\n\)', text, re.S)
    body = re.sub(r'#.*', '', block.group(1))
    tokens.update(re.findall(r'[A-Za-z_]\w*', body))
    return tokens


def _surface_names(entries):
    """Dotted names at every dot boundary, split by what they name.

    ``top`` holds modules, classes, functions and constants (an undotted token
    may match any of their suffixes); ``members`` holds ``Owner.member`` names,
    which only match when the owner is spelled too.
    """
    top, members = set(), set()
    for e in entries:
        parts = e.name.split('.')
        target = members if e.cls is not None else top
        for k in range(len(parts)):
            target.add('.'.join(parts[k:]))
    for e in entries:                       # an alias row names the same object
        if e.kind == 'alias':
            top.add(e.name.rsplit('.', 1)[-1])
    return top, members


def _on_surface(token, entries):
    """True when *token* (``Name``, ``Owner.member`` or a dotted path) is a row.

    A member must be matched with its owner: ``Module.nets`` is not satisfied
    by ``ModuleTestbench.nets`` or any other member called ``nets``.
    """
    top, members = _surface_names(entries)
    if token in top or token in members:
        return True
    if token.startswith('silk.'):
        objs = {id(e.obj) for e in entries if e.obj is not None}
        parts = token.split('.')
        for i in range(len(parts), 0, -1):
            try:
                obj = importlib.import_module('.'.join(parts[:i]))
            except ImportError:
                continue
            for attr in parts[i:]:
                obj = getattr(obj, attr, None)
            return obj is not None and id(obj) in objs
    return False


# Documented only inside prose or table cells, not as a heading or a table's
# first column: (owner, name) pairs the extraction cannot see.
EXPLICIT = (
    'ModuleInstance',          # return record of Module.iter_instances
    'TestEntry',               # return record of Module.list_test_entries
    'ModuleInstance.path', 'ModuleInstance.spice_path',
    'ModuleInstance.module', 'ModuleInstance.owner_path',
    'ModuleInstance.module_name', 'ModuleInstance.placed',
    'TestEntry.path', 'TestEntry.spice_path', 'TestEntry.module',
    'TestEntry.owner_path', 'TestEntry.test_name', 'TestEntry.test',
    'TestEntry.report_key', 'TestEntry.module_name',
    'TestEntry.inherited_temperature', 'TestEntry.inherited_supply',
    'TestEntry.output_mode',
)


def _table_pairs(text, entries):
    """``(Owner, member)`` for each table row whose first cell names a method.

    The owner is a class named by the nearest preceding heading. A section can
    carry the tables of several related classes under one heading, so a pair
    is kept for the heading's classes that really have the member; names no
    heading class has come back with owner ``None`` for a top-level lookup.
    """
    objs = {e.name.rsplit('.', 1)[-1]: e.obj for e in entries
            if e.kind in ('class', 'exception', 'enum') and e.cls is None}
    owners, out = [], set()
    for line in text.splitlines():
        h = re.match(r'#{2,4}\s+(.*)', line)
        if h:
            owners = [c for c in re.findall(r'`(\w+)', h.group(1)) if c in objs]
            continue
        row = re.match(r'\|\s*`([A-Za-z_]\w*)\(', line)
        if not row or row.group(1).startswith('_'):
            continue
        name = row.group(1)
        having = [o for o in owners if hasattr(objs[o], name)]
        if having:
            out.update((o, name) for o in having)
        elif owners:
            out.add((None, name))
    return out


def test_every_documented_symbol_is_on_the_manifest(entries):
    tokens = _documented_tokens()
    assert len(tokens) > 120, 'the extraction found too little to be a gate'
    missing = sorted(t for t in tokens if not _on_surface(t, entries))
    assert missing == []


def test_explicit_and_table_documented_members_are_on_the_manifest(entries):
    pairs = _table_pairs(API_REFERENCE.read_text(encoding='utf-8'), entries)
    assert len(pairs) > 40, 'the table extraction found too little'
    wanted = {f'{o}.{m}' if o else m for o, m in pairs} | set(EXPLICIT)
    missing = sorted(w for w in wanted if not _on_surface(w, entries))
    assert missing == [], 'documented in docs/api-reference.md, not a row'


def test_a_member_needs_its_owner(entries):
    assert _on_surface('Module.nets', entries)
    assert not _on_surface('ModuleInstance.nets', entries)
    assert not _on_surface('Spec.nets', entries)
    assert not _on_surface('NoSuchOwner.path', entries)


@pytest.mark.parametrize('dotted', INCLUDED)
def test_included_rows_are_collected_once(entries, dotted):
    hits = [e for e in entries if e.name == dotted]
    assert len(hits) == 1, f'{dotted} must be exactly one manifest row'
    assert hits[0].kind != 'unresolved'


@pytest.mark.parametrize('dotted', ['silk.circuit.validate_instance_name',
                                   'silk.circuit.validate_net_name'])
def test_public_name_checks_are_collected_once(entries, dotted):
    hits = [entry for entry in entries if entry.name == dotted]
    assert len(hits) == 1
    assert hits[0].kind == 'function'
