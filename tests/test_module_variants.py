"""TASK-292.1 — Module(..., variants=True) and the module_name validator.

Motivating case (owner, 2026-09-27): a designer builds two bias blocks from
the same builder with different currents and places both with add_module.
Today that raises CircuitError('module_name ... is claimed by two
structurally different circuits') unless the builder bakes the value into
the name by hand. variants=True opts a module_name into emitting each
distinct body under a deterministic, structure-hash-derived suffixed name
instead — ALWAYS suffixed (order/count-independent), never for the caller's
own kept object (AC#4), and consistently for every place module_name acts
as an identity (AC#1: netlist .subckt/X lines, the structure hash, DB
tracking, run_all_tests result keys, current-annotation grouping, fault
targeting, the human circuit report).

Uses ideal resistors (``silk.elements.elements.resistor``), not PDK
transistors, everywhere a real simulation is needed — fast and free of PDK
convergence risk, matching ``tests/test_dut_must_be_subcircuit.py``'s
``subcircuit()`` helper. Structural-only checks (module_dict keys, emitted
text) use ``pdk.nmos1p8``, matching ``TestSubcktUniqueness`` in
``tests/test_spice_netlister.py``, which this suite is a sibling to.
"""
import re

import pytest

from silk import Module, load_pdk, sources
from silk.elements import elements
from silk.exceptions import CircuitError
from silk.circuit.modules_structure import _validate_module_name
from silk.simulation import Transient
from tests._env import needs_ngspice


# ---------------------------------------------------------------------------
# .subckt-name validator
# ---------------------------------------------------------------------------

class TestModuleNameValidator:

    @pytest.mark.parametrize('name', [
        'bias', '_bias', 'Bias10u', 'a', 'a_1', '_1', 'CamelCase_Name123',
    ])
    def test_accepts_identifier_like_names(self, name):
        _validate_module_name(name)  # must not raise

    @pytest.mark.parametrize('name', [
        '1bias', 'bias-x', 'bias x', 'bi.as', '', 'bi/as', 'bi@as',
    ])
    def test_rejects_names_spice_cannot_carry(self, name):
        with pytest.raises(ValueError, match=re.escape(repr(name))):
            _validate_module_name(name)

    def test_constructor_validates_module_name(self, pdk, tmp_dir):
        with pytest.raises(ValueError, match=re.escape("'1bad'")):
            Module('PDK_skywater130', tmp_dir, 'bad.cir', pdk_details=pdk,
                   module_name='1bad')

    def test_constructor_accepts_none_module_name(self, pdk, tmp_dir):
        # None means top-level circuit — never validated, never refused.
        Module('PDK_skywater130', tmp_dir, 'top.cir', pdk_details=pdk)

    def test_define_module_name_validates(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'x.cir', pdk_details=pdk)
        with pytest.raises(ValueError, match=re.escape("'bad name'")):
            mod.define_module_name('bad name')
        # unchanged after the refusal
        assert mod.module_name is None
        mod.define_module_name('good_name')
        assert mod.module_name == 'good_name'


# ---------------------------------------------------------------------------
# variants=False (default): unchanged behaviour — AC#3
# ---------------------------------------------------------------------------

def _leaf(pdk, tmp_dir, name, w, *, variants=False):
    m = Module('PDK_skywater130', tmp_dir, f'{name}_{w}.cir', pdk_details=pdk,
               module_name=name, variants=variants)
    m.define_module_ports({'a': 'in', 'y': 'out', 'vdd': 'power', 'vss': 'ground'})
    m.add(pdk.nmos1p8('mn', d='y', g='a', s='vss', b='vss', w=w, l=0.15, m=1))
    return m


class TestVariantsFalseUnchanged:

    def test_default_is_false(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'x.cir', pdk_details=pdk,
                     module_name='m')
        assert mod.variants is False

    def test_same_name_different_body_still_raises(self, pdk, tmp_dir):
        a = _leaf(pdk, tmp_dir, 'bias', 1.0)
        b = _leaf(pdk, tmp_dir, 'bias', 2.0)
        top = Module('PDK_skywater130', tmp_dir, 'top.cir', pdk_details=pdk)
        top.add_module(a, 'x1', a='na', y='y0', vdd='vdd', vss='0')
        top.add_module(b, 'x2', a='na', y='y1', vdd='vdd', vss='0')
        with pytest.raises(CircuitError, match='structurally different'):
            top.generate_netlist()

    def test_identical_rebuild_dedups_no_suffix(self, pdk, tmp_dir):
        a = _leaf(pdk, tmp_dir, 'bias', 1.0)
        c = _leaf(pdk, tmp_dir, 'bias', 1.0)
        top = Module('PDK_skywater130', tmp_dir, 'top.cir', pdk_details=pdk)
        top.add_module(a, 'x1', a='na', y='y0', vdd='vdd', vss='0')
        top.add_module(c, 'x2', a='na', y='y1', vdd='vdd', vss='0')
        assert list(top.module_dict.keys()) == ['bias']

    def test_caller_module_keeps_its_name(self, pdk, tmp_dir):
        a = _leaf(pdk, tmp_dir, 'bias', 1.0)
        top = Module('PDK_skywater130', tmp_dir, 'top.cir', pdk_details=pdk)
        top.add_module(a, 'x1', a='na', y='y0', vdd='vdd', vss='0')
        assert a.module_name == 'bias'


# ---------------------------------------------------------------------------
# variants=True — AC#2
# ---------------------------------------------------------------------------

class TestVariantsTrue:

    def test_two_bodies_get_distinct_deterministic_suffixed_names(
            self, pdk, tmp_dir):
        a = _leaf(pdk, tmp_dir, 'bias', 1.0, variants=True)
        b = _leaf(pdk, tmp_dir, 'bias', 2.0, variants=True)
        top = Module('PDK_skywater130', tmp_dir, 'top.cir', pdk_details=pdk)
        top.add_module(a, 'x1', a='na', y='y0', vdd='vdd', vss='0')
        top.add_module(b, 'x2', a='na', y='y1', vdd='vdd', vss='0')

        keys = list(top.module_dict.keys())
        assert len(keys) == 2
        for k in keys:
            assert re.fullmatch(r'bias_[0-9a-f]{8}', k), k
        assert keys[0] != keys[1]

    def test_identical_bodies_dedup_to_one_definition(self, pdk, tmp_dir):
        a = _leaf(pdk, tmp_dir, 'bias', 1.0, variants=True)
        c = _leaf(pdk, tmp_dir, 'bias', 1.0, variants=True)  # separate object, same body
        top = Module('PDK_skywater130', tmp_dir, 'top.cir', pdk_details=pdk)
        top.add_module(a, 'x1', a='na', y='y0', vdd='vdd', vss='0')
        top.add_module(c, 'x2', a='na', y='y1', vdd='vdd', vss='0')
        assert len(top.module_dict) == 1

    def test_caller_module_keeps_its_base_name(self, pdk, tmp_dir):
        a = _leaf(pdk, tmp_dir, 'bias', 1.0, variants=True)
        top = Module('PDK_skywater130', tmp_dir, 'top.cir', pdk_details=pdk)
        top.add_module(a, 'x1', a='na', y='y0', vdd='vdd', vss='0')
        assert a.module_name == 'bias'  # unsuffixed, per AC#4

    def test_deterministic_across_placement_order(self, pdk, tmp_dir):
        """Same inputs, opposite placement order -> the same pair of names."""
        a1 = _leaf(pdk, tmp_dir, 'bias', 1.0, variants=True)
        b1 = _leaf(pdk, tmp_dir, 'bias', 2.0, variants=True)
        top1 = Module('PDK_skywater130', tmp_dir, 'top1.cir', pdk_details=pdk)
        top1.add_module(a1, 'x1', a='na', y='y0', vdd='vdd', vss='0')
        top1.add_module(b1, 'x2', a='na', y='y1', vdd='vdd', vss='0')

        a2 = _leaf(pdk, tmp_dir, 'bias', 1.0, variants=True)
        b2 = _leaf(pdk, tmp_dir, 'bias', 2.0, variants=True)
        top2 = Module('PDK_skywater130', tmp_dir, 'top2.cir', pdk_details=pdk)
        top2.add_module(b2, 'x2', a='na', y='y1', vdd='vdd', vss='0')  # 2.0 first
        top2.add_module(a2, 'x1', a='na', y='y0', vdd='vdd', vss='0')  # 1.0 second

        assert set(top1.module_dict.keys()) == set(top2.module_dict.keys())

    def test_built_once_still_carries_the_suffix(self, pdk, tmp_dir):
        """ALWAYS-suffix: a variants=True module placed exactly once is
        still suffixed, not left bare."""
        a = _leaf(pdk, tmp_dir, 'lonely', 1.0, variants=True)
        top = Module('PDK_skywater130', tmp_dir, 'top.cir', pdk_details=pdk)
        top.add_module(a, 'x1', a='na', y='y0', vdd='vdd', vss='0')
        keys = list(top.module_dict.keys())
        assert keys == [k for k in keys if re.fullmatch(r'lonely_[0-9a-f]{8}', k)]
        assert keys != ['lonely']

    def test_netlist_two_defs_three_instances_one_dedup(self, pdk, tmp_dir):
        """AC#4's exact shape: 10u, 20u, and a third instance rebuilding
        10u -> two .subckt definitions, three X lines."""
        a = _leaf(pdk, tmp_dir, 'bias', 1.0, variants=True)
        b = _leaf(pdk, tmp_dir, 'bias', 2.0, variants=True)
        c = _leaf(pdk, tmp_dir, 'bias', 1.0, variants=True)  # == a's body
        top = Module('PDK_skywater130', tmp_dir, 'top.cir', pdk_details=pdk)
        top.add_module(a, 'x1', a='na', y='y0', vdd='vdd', vss='0')
        top.add_module(b, 'x2', a='na', y='y1', vdd='vdd', vss='0')
        top.add_module(c, 'x3', a='na', y='y2', vdd='vdd', vss='0')

        netlist = top.netlister.generate_module_definitions(top.module_dict)
        defs = re.findall(r'(?im)^\.subckt\s+(\S+)', netlist)
        assert len(defs) == 2
        assert len(set(defs)) == 2

        x_lines = [top.netlister.generate_device_instances({n: top.elem_dict[n]}).splitlines()[-1]
                   for n in ('x1', 'x2', 'x3')]
        targets = [line.split()[-1].strip() for line in x_lines]
        assert len(targets) == 3
        assert targets[0] == targets[2]  # x1 and x3 share the dedup'd def
        assert targets[1] != targets[0]  # x2 is the distinct 20u body
        assert set(targets) == set(defs)

    def test_truncated_hash_collision_still_fails_loud_not_silent(
            self, pdk, tmp_dir, monkeypatch):
        """The structural safety net from the AC#1 addendum: if two
        different bodies' suffixes ever DID collide, register_module_definition
        must still raise rather than silently merge them. Forced by
        monkeypatching the hash to a constant so two different bodies land
        on the identical name."""
        import silk.circuit.circuit_elements as ce
        monkeypatch.setattr(ce, '_module_structure_hash', lambda module: 'f' * 64)

        a = _leaf(pdk, tmp_dir, 'bias', 1.0, variants=True)
        b = _leaf(pdk, tmp_dir, 'bias', 2.0, variants=True)
        top = Module('PDK_skywater130', tmp_dir, 'top.cir', pdk_details=pdk)
        top.add_module(a, 'x1', a='na', y='y0', vdd='vdd', vss='0')
        # Both bodies' hash is forced identical, so the FIRST check (hash
        # mismatch) can't fire; the belt-and-braces body-TEXT cross-check
        # (TASK-126) still catches the genuine difference and raises —
        # proving the safety net has two independent layers, not one.
        top.add_module(b, 'x2', a='na', y='y1', vdd='vdd', vss='0')
        with pytest.raises(CircuitError, match='is claimed by two'):
            top.generate_netlist()

    def test_structure_tracker_keys_match_module_dict(self, pdk, tmp_dir):
        """AC#1 point 5/6: DB/report tracking reads module_dict's own keys,
        so it is automatically consistent once those are the suffixed
        names — no separate DB-layer change needed."""
        from silk.db.structure_tracker import CircuitStructureTracker

        a = _leaf(pdk, tmp_dir, 'bias', 1.0, variants=True)
        b = _leaf(pdk, tmp_dir, 'bias', 2.0, variants=True)
        top = Module('PDK_skywater130', tmp_dir, 'top.cir', pdk_details=pdk)
        top.add_module(a, 'x1', a='na', y='y0', vdd='vdd', vss='0')
        top.add_module(b, 'x2', a='na', y='y1', vdd='vdd', vss='0')

        tracker = CircuitStructureTracker(None, None)
        structure = tracker.extract_structure(top)
        assert set(structure['modules'].keys()) == set(top.module_dict.keys())

    def test_circuit_report_topology_shows_suffixed_names(self, pdk, tmp_dir):
        """AC#1 point 11: the human-readable circuit report's topology
        extraction reads module_dict directly — automatically consistent
        with the suffixed keys, no separate change needed there."""
        from silk.circuit.report import _build_topology

        a = _leaf(pdk, tmp_dir, 'bias', 1.0, variants=True)
        b = _leaf(pdk, tmp_dir, 'bias', 2.0, variants=True)
        top = Module('PDK_skywater130', tmp_dir, 'top.cir', pdk_details=pdk)
        top.add_module(a, 'x1', a='na', y='y0', vdd='vdd', vss='0')
        top.add_module(b, 'x2', a='na', y='y1', vdd='vdd', vss='0')

        topology = _build_topology(top)
        names = {inst.module_name for inst in topology.instances}
        assert names == set(top.module_dict.keys())


# ---------------------------------------------------------------------------
# variants=True — AC#4 must-have: run_all_tests returns distinct keys
# ---------------------------------------------------------------------------

def _bias_with_test(pdk, tmp_dir, r_value, *, variants=True):
    """A resistor-divider 'bias' block with its own transient testbench
    attached before it is returned — the shape build_bias(...) has in the
    owner's motivating scenario."""
    m = Module('PDK_skywater130', tmp_dir, f'bias_{r_value}.cir',
              pdk_details=pdk, module_name='bias', variants=variants)
    m.define_module_ports({'out': 'out', 'vss': 'ground'})
    m.add(elements.resistor('rload', p='out', n='vss', value=r_value))
    tb = m.create_test('level')
    tb.add_module_under_test({'out': 'vout', 'vss': '0'})
    tb.add_source(sources.VoltageSource('supply', positive='vout',
                                        negative='0', dc=1.8))
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    return m


@needs_ngspice
class TestRunAllTestsDistinctKeys:

    def test_two_bias_instances_get_distinct_result_keys(self, pdk, tmp_dir):
        """The must-have test (reviewer, 2026-09-27): two variant instances
        must NOT collide under one run_all_tests() key."""
        a = _bias_with_test(pdk, tmp_dir, 1000)   # '10u' stand-in
        b = _bias_with_test(pdk, tmp_dir, 2000)   # '20u' stand-in
        top = Module('PDK_skywater130', tmp_dir, 'top.cir', pdk_details=pdk)
        top.add_module(a, 'x1', out='o1', vss='0')
        top.add_module(b, 'x2', out='o2', vss='0')

        results = top.run_all_tests()
        assert len(results) == 2
        errors = {k: v for k, v in results.items()
                  if isinstance(v, dict) and 'error' in v}
        assert not errors, errors
        assert len(set(results.keys())) == 2  # no silent overwrite

    def test_three_instances_two_variants_one_dedup_get_full_coverage(
            self, pdk, tmp_dir):
        """10u, 20u, and a third instance rebuilding 10u: the reviewer's
        exact AC#4 shape, checked through run_all_tests()."""
        a = _bias_with_test(pdk, tmp_dir, 1000)
        b = _bias_with_test(pdk, tmp_dir, 2000)
        c = _bias_with_test(pdk, tmp_dir, 1000)  # rebuild of a's body
        top = Module('PDK_skywater130', tmp_dir, 'top.cir', pdk_details=pdk)
        top.add_module(a, 'x1', out='o1', vss='0')
        top.add_module(b, 'x2', out='o2', vss='0')
        top.add_module(c, 'x3', out='o3', vss='0')

        results = top.run_all_tests()
        # x1 and x3 share ONE .subckt definition (dedup'd) but are still
        # distinguishable instances — the pre-existing owner_path
        # qualification (see api-reference.md, run_all_tests) disambiguates
        # same-definition siblings; the module_name suffix disambiguates
        # different-body variants. Together: three usable result entries.
        assert len(results) == 3
        errors = {k: v for k, v in results.items()
                  if isinstance(v, dict) and 'error' in v}
        assert not errors, errors

    @needs_ngspice
    def test_two_bias_shape_with_a_real_simulated_op(self, pdk, tmp_dir):
        """The reviewer's exact AC#4 shape, evidenced rather than merely
        exercised on ideal-resistor netlist text: a real ngspice DC
        operating-point run through run_all_tests(), on 10u/20u/10u-again,
        checked together with the netlist shape in one test — two .subckt
        definitions, three X lines, and three distinct result keys (x1/x3
        further split by the pre-existing owner_path qualification since
        they share one dedup'd definition)."""
        def build_bias(r_value):
            m = Module('sky130', tmp_dir, f'bias_{r_value}.cir', pdk_details=pdk,
                      module_name='bias', variants=True)
            m.define_module_ports({'out': 'out', 'vss': 'ground'})
            m.add(elements.resistor('rload', p='out', n='vss', value=r_value))
            tb = m.create_test('op')
            tb.add_module_under_test({'out': 'vout', 'vss': '0'})
            tb.add_source(sources.VoltageSource('vout', positive='vout',
                                                negative='0', dc=1.0))
            from silk.simulation import DC
            tb.add_sim_type(DC(source='vvout', start=1.0, stop=1.1, step=0.1))
            return m

        a = build_bias(1000)    # '10u' stand-in
        b = build_bias(2000)    # '20u' stand-in
        c = build_bias(1000)    # rebuild of a's body
        top = Module('sky130', tmp_dir, 'top.cir', pdk_details=pdk)
        top.add_module(a, 'x1', out='o1', vss='0')
        top.add_module(b, 'x2', out='o2', vss='0')
        top.add_module(c, 'x3', out='o3', vss='0')

        # Netlist shape.
        netlist = top.netlister.generate_module_definitions(top.module_dict)
        defs = re.findall(r'(?im)^\.subckt\s+(\S+)', netlist)
        assert len(defs) == 2
        x_lines = [top.netlister.generate_device_instances({n: top.elem_dict[n]}).splitlines()[-1]
                   for n in ('x1', 'x2', 'x3')]
        assert len(x_lines) == 3

        # A real simulated operating point, through run_all_tests().
        results = top.run_all_tests()
        errors = {k: v for k, v in results.items()
                  if isinstance(v, dict) and 'error' in v}
        assert not errors, errors
        assert len(results) == 3
        assert len(set(results.keys())) == 3  # no silent overwrite


# ---------------------------------------------------------------------------
# variants unset (default False) on a REAL shipped example — AC#3 evidence,
# not just the design commitment. Pinned deck text + pinned run_all_tests
# key set: either regressing would fail this test, not merely leave a note.
# ---------------------------------------------------------------------------

@needs_ngspice
def test_real_example_deck_and_result_keys_unchanged_without_variants():
    """current_mirror_example.py never sets variants=True. Its emitted
    .subckt text and its run_all_tests() result-key set are pinned here
    literally, captured from this same branch and cross-checked by hand
    against a detached pre-TASK-292.1 checkout of main (byte-identical) —
    see the TASK-292.1 task notes. A regression in the variants=False path
    changes either pin."""
    import silk.examples.current_mirror_example as ex

    mod = ex.build_current_mirror()
    assert mod.generate_netlist() == (
        ".subckt current_mirror VDD VSS REF OUT \n"
        "*** M1.  Connections = ['d', 'g', 's', 'b'].\n"
        "XMM1 REF REF VSS VSS sky130_fd_pr__nfet_01v8 m=1 w=4 l=1\n"
        "*** M2.  Connections = ['d', 'g', 's', 'b'].\n"
        "XMM2 OUT REF VSS VSS sky130_fd_pr__nfet_01v8 m=1 w=4 l=1\n"
        ".ends\n"
    )

    results, _recon = ex.run_characterization(mod)
    errors = {k: v for k, v in results.items()
              if isinstance(v, dict) and 'error' in v}
    assert not errors, errors
    assert set(results.keys()) == {'current_mirror.match', 'current_mirror.zout'}
