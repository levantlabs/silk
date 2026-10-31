"""Real-circuit contracts for child-instance connectivity in structure v3."""

from collections import Counter
import json
from pathlib import Path

import pytest

from silk import Module, Port
from silk.db.identity import _fingerprint_module_structure, _fingerprint_structure
from silk.db.structure_tracker import CircuitStructureTracker
from silk.elements import elements
from silk.exceptions import CircuitError, EvidenceMismatchError, UnsupportedFingerprintSchemeError

from scripts.audit_hierarchy_identity import _flat, _parent, _registration, _wrap


def _structure(module):
    return CircuitStructureTracker(None, None).extract_structure(module)


def _fingerprint(module):
    return _fingerprint_structure(_structure(module))[0]


def _definition(module):
    tracker = CircuitStructureTracker(None, None)
    return _fingerprint_module_structure(tracker._extract_module_structure(module))


@pytest.mark.parametrize('depth', [1, 2, 3])
def test_rewire_changes_parent_and_ancestor_but_not_leaf(tmp_path, depth):
    original = _parent(str(tmp_path))
    rewired = _parent(str(tmp_path), swap=True)
    assert _fingerprint(original.elem_dict['u']['device']) == _fingerprint(
        rewired.elem_dict['u']['device']
    )
    assert _definition(original.elem_dict['u']['device']) == _definition(
        rewired.elem_dict['u']['device']
    )
    for level in range(2, depth + 1):
        original = _wrap(str(tmp_path), original, f'level{level}')
        rewired = _wrap(str(tmp_path), rewired, f'level{level}')
    assert _fingerprint(original) != _fingerprint(rewired)
    assert _definition(original) != _definition(rewired)
    top = _wrap(str(tmp_path), original, 'top')
    top.add_module(rewired, 'second', nets={
        'a': 'a', 'b': 'b', 'vss': 'vss',
    })
    with pytest.raises(CircuitError, match='two structurally different circuits'):
        top.generate_netlist()
    assert 'second' in top.elem_dict


def test_identical_definitions_stay_reusable_across_external_bindings(tmp_path):
    out = str(tmp_path)
    for first, second in [(_flat(out), _flat(out)), (_parent(out), _parent(out))]:
        result = _registration(out, first, second)
        assert result['circuit_error'] is None
        assert result['definition_count'] == 1
        assert 'xoriginal vin out1 vss ' + first.module_name in result['instance_lines']
        assert 'xchanged vin out2 vss ' + first.module_name in result['instance_lines']


def _tree(out, edit='rebuild'):
    reverse = edit == 'reorder'

    def module(name):
        mod = Module('none', str(out), f'{name}.cir', module_name=name,
                     pdk_details=None)
        ports = [('a', Port.analog_input()), ('b', Port.analog_output()),
                 ('vss', Port.ground())]
        mod.define_module_ports(dict(reversed(ports) if reverse else ports))
        return mod

    stage = module('stage')
    children = [('left', 'leaf_left'), ('right', 'leaf_right')]
    for instance, name in reversed(children) if reverse else children:
        leaf = module(name)
        devices = [
            ('r1', 'a', 'b', 1000 if instance == 'left' else 2000),
            ('r2', 'b', 'vss', 3000),
        ]
        for device, pos, neg, value in reversed(devices) if reverse else devices:
            leaf.add(elements.resistor(device, p=pos, n=neg, value=value))
        if instance == 'left':
            if edit == 'device_terminal':
                leaf.elem_dict['r1']['device'].n = 'vss'
            if edit == 'value':
                leaf.elem_dict['r1']['device'].value = 1500
            if edit == 'add_device_two_levels_down':
                leaf.add(elements.resistor('r3', p='a', n='vss', value=4000))
        swap = (instance == 'right') != (edit == 'swap_children')
        stage.add_module(leaf, instance, nets={
            'a': 'b' if swap else 'a', 'b': 'a' if swap else 'b', 'vss': 'vss',
        })
    if edit == 'child_port':
        stage.connect_module('left', a='vss')
    return _wrap(str(out), stage, 'root')


@pytest.mark.parametrize(('edit', 'changes'), [
    ('device_terminal', True),
    ('child_port', True),
    ('swap_children', True),
    ('value', True),
    ('add_device_two_levels_down', True),
    ('rebuild', False),
    ('reorder', False),
])
def test_structure_mutation_table(tmp_path, edit, changes):
    original, changed = _tree(tmp_path), _tree(tmp_path, edit)
    assert (_fingerprint(original) != _fingerprint(changed)) is changes
    assert (_definition(original) != _definition(changed)) is changes


def test_hierarchy_has_exactly_one_complete_record_per_instance_at_every_depth(tmp_path):
    tree = _wrap(str(tmp_path), _tree(tmp_path), 'outer')
    # Repeat the same definition with different bindings; count instances,
    # including repeated descendants, rather than unique definitions.
    tree.add_module(_tree(tmp_path), 'other', nets={'a': 'b', 'b': 'a', 'vss': 'vss'})
    expected = {}
    tracker = CircuitStructureTracker(None, None)

    def walk(parent, prefix=''):
        payload = tracker._extract_module_structure(parent)
        children = {name: entry['device'] for name, entry in parent.elem_dict.items()
                    if str(entry['type']) in ('m', 'ElementType.MODULE')}
        assert set(payload['submodules']) == set(children)
        for name, child in children.items():
            path = f'{prefix}.{name}' if prefix else name
            ports = {port: pin.net for port, pin in parent.elem_dict[name]['pin_list'].items()}
            expected[path] = ports
            record = payload['submodules'][name]
            assert record['ports'] == ports
            assert record['hash'] == _definition(child).digest
            walk(child, path)

    walk(tree)
    hierarchy = _structure(tree)['hierarchy']
    actual = {record['instance_name']: record['ports'] for record in hierarchy}
    assert len(actual) == len(hierarchy) == len(expected)
    assert actual == expected
    assert Counter(path.count('.') + 1 for path in actual) == {1: 2, 2: 2, 3: 4}


@pytest.mark.parametrize('extractor', ['extract_structure', '_extract_module_structure'])
@pytest.mark.parametrize('damage', ['missing_pin_list', 'missing_net', 'unbound_port'])
def test_incomplete_instance_record_raises_with_path(tmp_path, monkeypatch, extractor, damage):
    parent = _wrap(str(tmp_path), _parent(str(tmp_path)), 'outer')
    entry = parent.elem_dict['stage']['device'].elem_dict['u']
    child = entry['device']
    if damage == 'missing_pin_list':
        monkeypatch.delattr(child, 'pin_list')
    elif damage == 'missing_net':
        monkeypatch.delattr(entry['pin_list']['a'], 'net')
    else:
        monkeypatch.setattr(entry['pin_list']['a'], 'net', '')
    tracker = CircuitStructureTracker(None, None)
    with pytest.raises(CircuitError, match=r'stage\.u.*parent-net bindings'):
        getattr(tracker, extractor)(parent)


def test_rewired_snapshot_has_separate_evidence_and_persisted_bindings(tmp_path):
    db_path = str(tmp_path / 'identity.db')
    original = _parent(str(tmp_path), db_path=db_path)
    rewired = _parent(str(tmp_path), swap=True, db_path=db_path)
    rebuild = _parent(str(tmp_path), db_path=db_path)
    try:
        first_id = original.create_circuit_version()
        original.store_result('sentinel', 0.666, 'V/V')
        assert rewired.create_circuit_version() != first_id
        with pytest.raises(EvidenceMismatchError, match='incompatible structure'):
            rewired.query_results(parameter='sentinel')
        assert rebuild.create_circuit_version() == first_id
        assert [r.value for r in rebuild.query_results(parameter='sentinel')] == [0.666]
    finally:
        for module in (original, rewired, rebuild):
            module.close()

    # A nested definition exercises module payload storage and re-validation.
    nested = _wrap(str(tmp_path), _parent(str(tmp_path)), 'nested')
    top = Module('none', str(tmp_path), 'stored.cir', pdk_details=None,
                 db_path=db_path)
    try:
        top.add_module(nested, 'n', nets={'a': 'in', 'b': 'out', 'vss': '0'})
        top.create_circuit_version()
        row = top._tracker.db.conn.execute(
            "SELECT structure FROM modules WHERE module_name = 'nested'"
        ).fetchone()
        payload = json.loads(row[0])
        assert payload['submodules']['stage']['ports'] == {'a': 'a', 'b': 'b', 'vss': 'vss'}
        assert _fingerprint_module_structure(payload) == _definition(nested)
    finally:
        top.close()


@pytest.mark.parametrize('historical_scheme', ['silk.structure/v1', 'silk.structure/v2'])
def test_legacy_sealed_snapshot_rejected_without_rewriting_evidence(tmp_path, historical_scheme):
    db_path = str(tmp_path / 'legacy.db')
    module = _parent(str(tmp_path), db_path=db_path)
    try:
        version = module.create_circuit_version()
        module.store_result('sentinel', 0.666, 'V/V')
        db = module._tracker.db
        # Simulate a historical identity without changing its digest; a
        # matching digest must never bypass the scheme boundary.
        db.conn.execute("UPDATE circuit_versions SET structure_scheme = ?", (historical_scheme,))
        db.conn.execute("UPDATE modules SET module_hash_scheme = ?", (historical_scheme,))
        db.conn.commit()
        before = list(db.conn.iterdump())
        with pytest.raises(UnsupportedFingerprintSchemeError, match='rebuild.*new database'):
            module.create_circuit_version()
        with pytest.raises(UnsupportedFingerprintSchemeError, match='rebuild'):
            CircuitStructureTracker(db, version)
        with pytest.raises(UnsupportedFingerprintSchemeError, match='rebuild'):
            module.query_results(parameter='sentinel')
        assert list(db.conn.iterdump()) == before
    finally:
        module.close()


@pytest.mark.parametrize(('label', 'swap'), [('original', False), ('rewired', True)])
def test_v3_hierarchical_fingerprints_match_frozen_payloads(tmp_path, label, swap):
    fixture = json.loads((Path(__file__).parents[1] / 'fixtures' /
                         'db_fingerprints_v3.json').read_text())
    case = fixture['cases'][label]
    module = _wrap(str(tmp_path), _parent(str(tmp_path), swap=swap), 'outer')
    tracker = CircuitStructureTracker(None, None)
    assert tracker._extract_module_structure(module) == case['module_payload']
    assert _structure(module)['hierarchy'] == case['hierarchy']
    assert _definition(module).digest == case['module_digest']
    assert _fingerprint(module).digest == case['structure_digest']
    # The frozen v3 payload/digests stay untouched; v4 changes only the tag
    # for this wholly source-free hierarchy.
    assert fixture['structure_scheme'] == 'silk.structure/v3'
    assert _fingerprint(module).scheme == 'silk.structure/v4'


# ---------------------------------------------------------------------------
# TASK-126 guard 1: dedup cross-checks the emitted subcircuit text, so hash
# equality alone can never merge two different bodies again.
# ---------------------------------------------------------------------------

def _constant_hash(monkeypatch):
    import silk.circuit.circuit_elements as circuit_elements
    calls = []

    def constant(module):
        calls.append(module.module_name)
        return 'constant'

    monkeypatch.setattr(circuit_elements, '_module_structure_hash', constant)
    return calls


def test_dedup_rejects_rewired_body_even_when_hash_is_reverted(tmp_path, monkeypatch):
    """The TASK-125 reproduction (two same-named roots, child ports swapped)
    must raise at registration even under the old, binding-blind hash: the
    hash is monkeypatched to a constant, so only the emitted text can tell
    the two bodies apart."""
    calls = _constant_hash(monkeypatch)
    out = str(tmp_path)
    top = _wrap(out, _parent(out), 'top')
    rewired = _parent(out, swap=True)
    top.add_module(rewired, 'second', nets={'a': 'a', 'b': 'b', 'vss': 'vss'})
    with pytest.raises(CircuitError) as info:
        top.generate_netlist()
    message = str(info.value)
    assert 'structure hashes agree' in message
    assert 'emitted .subckt text differs' in message
    # The diagnostic names the divergent instance line, not just "differs".
    assert "'xu a b vss leaf'" in message and "'xu b a vss leaf'" in message
    assert calls, 'the hash must still have been consulted first'
    # Nothing left behind: the rejected instance and its registration.
    assert 'second' in top.elem_dict
    assert top.module_dict['root']['numberInstantiated'] == 2
    assert id(rewired) in top.module_dict['root']['_source_ids']


def test_dedup_still_merges_identical_rebuild_under_constant_hash(tmp_path, monkeypatch):
    """The text cross-check must not over-fire: an identical rebuild (same
    name, same body) dedupes exactly as before, and the deck carries one
    definition."""
    _constant_hash(monkeypatch)
    out = str(tmp_path)
    top = _wrap(out, _parent(out), 'top')
    top.add_module(_parent(out), 'second', nets={'a': 'a', 'b': 'b', 'vss': 'vss'})
    entry = top.module_dict['root']
    assert entry['numberInstantiated'] == 2
    assert len(entry['_source_ids']) == 2
    deck = top.generate_netlist(subckt=True)
    assert deck.count('.subckt root ') == 1
    assert 'xstage a b vss root' in deck and 'xsecond a b vss root' in deck


def test_dedup_text_check_leaves_netlister_state_untouched(tmp_path):
    """Emitting for the cross-check pins the ammeter/emitted netlister state
    on both bodies and restores it, so a later real emission is unaffected."""
    from silk.circuit.circuit_elements import _module_subckt_text
    out = str(tmp_path)
    parent = _parent(out)
    netlister = parent.netlister
    netlister._terminal_current_method = 'ammeter_wrapper'
    netlister._emitted_subckts = {'leaf'}
    assert not hasattr(netlister, '_ammeter_wrappers')
    text = _module_subckt_text(parent)
    # Emitted with a fresh emitted-set: the nested leaf definition is present.
    assert '.subckt leaf ' in text and '.subckt root ' in text
    assert netlister._terminal_current_method == 'ammeter_wrapper'
    assert netlister._emitted_subckts == {'leaf'}
    assert not hasattr(netlister, '_ammeter_wrappers')


def test_dedup_refuses_to_merge_when_text_cannot_be_emitted(tmp_path, monkeypatch):
    """A body whose .subckt text cannot be emitted cannot be proven equal, so
    it is rejected loudly rather than merged on the hash alone."""
    _constant_hash(monkeypatch)
    out = str(tmp_path)
    top = _wrap(out, _parent(out), 'top')
    second = _parent(out)
    monkeypatch.setattr(second.netlister, 'generate_device_instances',
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
    top.add_module(second, 'second', nets={'a': 'a', 'b': 'b', 'vss': 'vss'})
    with pytest.raises(CircuitError, match='could not be emitted.*RuntimeError: boom'):
        top.generate_netlist()
    assert 'second' in top.elem_dict
    assert top.module_dict['root']['numberInstantiated'] == 2


def test_dedup_text_check_registers_a_block_the_lint_catalog_flags(tmp_path):
    """A block with a port named ``gnd`` is a STRUCT-012 finding and is
    refused by real deck emission, but it must still register twice for
    linting: the cross-check compares text, it does not emit a deck."""
    out = str(tmp_path)

    def block():
        mod = Module('none', out, 'sense.cir', module_name='sense', pdk_details=None)
        mod.define_module_ports({'a': Port.analog_input(), 'gnd': Port.ground()})
        mod.add(elements.resistor('r1', p='a', n='gnd', value=1000))
        return mod

    top = Module('none', out, 'top.cir', pdk_details=None)
    top.add_module(block(), 'u1', nets={'a': 'x', 'gnd': 'gnd'})
    top.add_module(block(), 'u2', nets={'a': 'y', 'gnd': 'gnd'})
    assert top.module_dict['sense']['numberInstantiated'] == 2
    with pytest.raises(Exception, match="declares a port named 'gnd'"):
        top.generate_netlist()
