"""Source-sensitive naming and safe opaque fallbacks without a simulator."""
import copy
from types import SimpleNamespace

import pytest

from silk import sources
from silk.circuit.circuit_elements import register_module_definition
from silk.circuit.module_test import ModuleTestbench
from silk.circuit.monte_carlo import MonteCarloSettings
from silk.circuit.test_execution import sharing_identity
from silk.db.structure_tracker import CircuitStructureTracker
from silk.exceptions import CircuitError, MonteCarloError
from silk.simulation._replay import experiment_signature, changed_inputs
from silk.simulation import Transient
from tests.db.test_source_identity import module


def test_builtin_source_variants_get_distinct_names_and_identical_dedup(tmp_path):
    registry = {}
    names = [register_module_definition(registry, module(tmp_path, dc=v, variants=True), validate=True)[0]
             for v in (1, 2, 1)]
    assert names[0] == names[2]
    assert names[0] != names[1]
    assert len(registry) == 2


def test_source_collision_with_variants_false_still_refuses(tmp_path):
    registry = {}
    first = module(tmp_path, dc=1)
    register_module_definition(registry, first, validate=True)
    with pytest.raises(CircuitError, match='two structurally different|text differs'):
        register_module_definition(registry, module(tmp_path, dc=2), validate=True)


def opaque(tmp_path, target, *, variants=False, level=7):
    class CustomSource(sources.VoltageSource):
        def to_spice(self):
            return f'Vbias out vss DC {self.level}\n'

    class CustomWave(sources.Sine):
        def to_spice(self):
            return f'PWL(0 {self.level} 1 {self.level})'

    m = module(tmp_path, variants=variants)
    wave = (CustomWave if target == 'wave_class' else sources.Sine)(vampl=1, freq=1)
    src = (CustomSource if target == 'source_class' else sources.VoltageSource)('bias', 'out', 'vss', waveform=wave)
    src.level = wave.level = level
    if target == 'source_instance':
        src.to_spice = lambda: f'Vbias out vss DC {level}\n'
    if target == 'wave_instance':
        wave.to_spice = lambda: f'PWL(0 {level} 1 {level})'
    m.add_source(src)
    return m


TARGETS = ['source_class', 'wave_class', 'source_instance', 'wave_instance']


@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('variants', [False, True])
def test_custom_emitters_keep_registration_and_emitted_collision_guard(tmp_path, target, variants):
    registry = {}
    a = opaque(tmp_path, target, variants=variants)
    b = opaque(tmp_path, target, variants=variants)
    first = register_module_definition(registry, a, validate=True)[0]
    assert register_module_definition(registry, b, validate=True)[0] == first
    assert len(registry) == 1
    with pytest.raises(CircuitError, match='text differs|two structurally different'):
        register_module_definition(registry, opaque(tmp_path, target, variants=variants, level=8), validate=True)


@pytest.mark.parametrize('target', TARGETS)
def test_custom_emitters_refuse_persistence_without_blocking_campaign_or_sharing(tmp_path, target):
    m = opaque(tmp_path, target)
    with pytest.raises(CircuitError, match='source|Source'):
        CircuitStructureTracker(None, None).extract_structure(m)
    tb = ModuleTestbench(m, 'level')
    tb.add_module_under_test({'out': 'out', 'vss': '0', 'in': 'in'})
    tb.add_sim_type(Transient(step=.1, stop=1))
    assert sharing_identity(tb, SimpleNamespace(path='$/u')) is None
    assert tb._structure_hash() is None
    # This is the exact pre-dispatch require_valid boundary used by _execute_plan.
    before = experiment_signature(tb, MonteCarloSettings(1), {})
    src = tb.netlist.elem_dict['dut']['device'].sig_sources_dict['bias']
    if target.endswith('instance'):
        # Existing replay refuses callable instance state; do not add support.
        with pytest.raises(MonteCarloError, match='unsupported identity value function'):
            before.require_valid()
        assert not any(item.startswith('SPICE structure:') for item in before.invalid)
        return
    before.require_valid()
    # Replay still records emitted custom source content separately.
    src.level = src.waveform.level = 9
    after = experiment_signature(tb, MonteCarloSettings(1), {})
    after.require_valid()
    assert any(label.startswith('source ') for label in changed_inputs(before, after))


def test_inherited_unchanged_source_emitter_is_persistable(tmp_path):
    class Inherited(sources.VoltageSource):
        pass
    m = module(tmp_path)
    m.add_source(Inherited('bias', 'out', 'vss', dc=1))
    structure = CircuitStructureTracker(None, None).extract_structure(m)
    assert structure['sources'][0]['parameters']['dc'] == '1'


def test_opaque_campaign_metadata_remains_none_in_subset_and_aggregate(tmp_path):
    from silk.circuit.monte_carlo import MonteCarloResults
    from silk.circuit.spec_report import SpecReport
    from silk.specs.aggregate import encode_aggregate, decode_aggregate
    m = opaque(tmp_path, 'source_class')
    tb = ModuleTestbench(m, 'level')
    tb.add_module_under_test({'out': 'out', 'vss': '0', 'in': 'in'})
    report = SpecReport('level', 'bias', {'out_V': 7.})
    campaign = MonteCarloResults({('tt', 0): report}, settings=MonteCarloSettings(1),
        seeds={('tt', 0): 1}, structure_hash=tb._structure_hash(), campaign_source='spice')
    assert campaign.structure_hash is None
    assert campaign.corner('tt').structure_hash is None
    measurement = campaign.measurement('out_V', how='mean')
    assert measurement.aggregate.structure_hash is None
    assert decode_aggregate(encode_aggregate(measurement.aggregate)).structure_hash is None


def test_concrete_inherited_source_type_participates_in_identity(tmp_path):
    from tests.db.test_source_identity import identities
    class Inherited(sources.VoltageSource):
        pass
    a, b = module(tmp_path, dc=1), module(tmp_path)
    b.add_source(Inherited('bias', 'out', 'vss', dc=1))
    assert a.sig_sources_dict['bias'].to_spice() == b.sig_sources_dict['bias'].to_spice()
    assert all(x != y for x, y in zip(identities(a), identities(b)))


def test_source_variant_references_match_distinct_emitted_definitions(tmp_path):
    import re
    top = module(tmp_path, name='parent')
    for name, dc in [('a', 1), ('b', 2), ('c', 1)]:
        top.add_module(module(tmp_path, dc=dc, variants=True), name,
                       nets={'out': name, 'vss': 'vss', 'in': 'in'})
    definitions = top.netlister.generate_module_definitions(top.module_dict)
    names = re.findall(r'(?im)^\.subckt\s+(\S+)', definitions)
    assert len(names) == len(set(names)) == 2
    targets = [top.netlister.generate_device_instances({name: top.elem_dict[name]}).splitlines()[-1].split()[-1]
               for name in ('a', 'b', 'c')]
    assert targets[0] == targets[2] and targets[0] != targets[1]
    assert set(targets) == set(names)
    assert 'Vbias out vss DC 1\n' in definitions
    assert 'Vbias out vss DC 2\n' in definitions


def test_forced_source_variant_digest_collision_still_checks_emitted_body(tmp_path, monkeypatch):
    monkeypatch.setattr('silk.circuit.circuit_elements._module_structure_hash', lambda m: 'f' * 64)
    registry = {}
    first = module(tmp_path, dc=1, variants=True)
    register_module_definition(registry, first, validate=True)
    with pytest.raises(CircuitError, match='text differs'):
        register_module_definition(registry, module(tmp_path, dc=2, variants=True), validate=True)


def test_run_sharing_keeps_full_deck_guard_when_structural_hint_agrees(tmp_path):
    from silk.circuit.module_test import _RunSpec
    from silk.circuit.test_execution import equivalence
    m = module(tmp_path, dc=1)
    tb = ModuleTestbench(m, 'level')
    tb.add_module_under_test({'out': 'out', 'vss': '0', 'in': 'in'})
    tb.add_source(sources.VoltageSource('external', 'out', '0', dc=1))
    tb.add_sim_type(Transient(step=.1, stop=1))
    hint = sharing_identity(tb, SimpleNamespace(path='$/u'))
    spec = _RunSpec(tb.netlist, 'tt', str(tmp_path), (), sharing_identity=hint)
    before = equivalence(spec)
    tb.netlist.sig_sources_dict['external'].dc = 2
    assert sharing_identity(tb, SimpleNamespace(path='$/u')) == hint
    after = equivalence(spec)
    assert before is not None and after is not None
    assert before != after  # complete emitted decks still decide equivalence


def test_unprovable_control_reference_does_not_block_naming(tmp_path):
    from silk.circuit.circuit_elements import _module_structure_projection
    m = module(tmp_path, variants=True)
    m.add_source(sources.CCCS('copy', 'out', 'vss', 1, object()))
    with pytest.raises(CircuitError, match='source|Source'):
        CircuitStructureTracker(None, None).extract_structure(m)
    assert _module_structure_projection(m)[1] is False
    register_module_definition({}, m)  # naming only; no emission equivalence claimed


def test_source_aware_replay_uses_effective_supply_without_mutating_authored_source(tmp_path):
    from silk.simulation._replay import apply_effective_settings
    from silk.simulation.conditions import prepare_supply
    m = module(tmp_path, dc=1)
    tb = ModuleTestbench(m, 'level')
    m.set_supply_voltage(1.8, source='external')
    tb.add_module_under_test({'out': 'out', 'vss': '0', 'in': 'in'})
    tb.add_source(sources.VoltageSource('external', 'out', '0', dc=.5))
    tb.add_sim_type(Transient(step=.1, stop=1))
    before = experiment_signature(tb, MonteCarloSettings(1), {})
    before.require_valid()
    assert tb.netlist.sig_sources_dict['external'].dc == .5
    apply_effective_settings(tb, tb.netlist)
    prepare_supply(tb.netlist)
    assert tb.netlist.sig_sources_dict['external'].dc == 1.8
    after = experiment_signature(tb, MonteCarloSettings(1), {})
    after.require_valid()
    assert changed_inputs(before, after) == ()
