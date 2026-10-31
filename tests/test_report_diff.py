# tests/test_report_diff.py
import os
from pathlib import Path

import numpy as np
import pytest
from unittest.mock import MagicMock

from silk.circuit.modules import Module
from silk.circuit.circuit_elements import PortDir, PortRole
from silk.circuit.report import generate_report, CircuitReport
from silk.circuit.report_diff import (
    diff_reports, ReportDiff,
    SignalDiff, SignalDelta, TopologyDiff, AnnotationDiff, ViolationDiff,
    MagnitudeDiff, MagnitudeDelta, KindMismatch,
)

from silk.lint.locations import (
    NetLocation, PortLocation, InstanceLocation, TerminalLocation, ModuleLocation,
)
from silk.lint.violations import InterfaceViolation, Violation
from silk.lint.severity import Severity

FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'rawfiles'


def _pdk():
    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}
    return pdk


def _sub(tmp_path, name, ports):
    m = Module('TEST', str(tmp_path), f'{name}.cir', pdk_details=_pdk())
    m.define_module_name(name)
    m.define_module_ports(ports)
    return m


def _top_with_sim(tmp_path, vin_mean=0.9, vout_mean=0.91, extra_signals=None):
    comp = _sub(tmp_path, 'comp', {
        'vp':   PortDir.IN,
        'vout': (PortDir.OUT, PortRole.DIGITAL),
    })
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.module_name = 'top'
    top.add_net('vcmp')
    top.add_submodule('comp1', comp)
    top._behavioral_t = np.linspace(0, 1e-9, 100)
    # Phase 2a: populate Net.voltage directly.
    signals = {
        'vin':  np.ones(100) * vin_mean,
        'vout': np.ones(100) * vout_mean,
    }
    if extra_signals:
        for name, val in extra_signals.items():
            signals[name] = np.ones(100) * val
    for name, arr in signals.items():
        if name not in top.net_dict:
            top.add_net(name)
        top.net_dict[name].voltage = arr
    return top


def _top_no_sim(tmp_path):
    comp = _sub(tmp_path, 'comp', {'vp': PortDir.IN})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.module_name = 'top'
    top.add_submodule('comp1', comp)
    return top


# --- Basic ---

def test_diff_returns_report_diff(tmp_path):
    r = generate_report(_top_with_sim(tmp_path))
    diff = diff_reports(r, r)
    assert isinstance(diff, ReportDiff)


def test_diff_labels_default(tmp_path):
    r = generate_report(_top_with_sim(tmp_path))
    diff = diff_reports(r, r)
    assert diff.label_a == 'a'
    assert diff.label_b == 'b'


def test_diff_labels_custom(tmp_path):
    r = generate_report(_top_with_sim(tmp_path))
    diff = diff_reports(r, r, label_a='schematic', label_b='extracted')
    assert diff.label_a == 'schematic'
    assert diff.label_b == 'extracted'


# --- Signals ---

def test_diff_identical_signals_no_changes(tmp_path):
    r = generate_report(_top_with_sim(tmp_path))
    diff = diff_reports(r, r)
    assert diff.signal_diff.changed   == []
    assert diff.signal_diff.added     == []
    assert diff.signal_diff.dropped   == []
    assert len(diff.signal_diff.unchanged) == 2


def test_diff_signal_delta_computed(tmp_path):
    r_a = generate_report(_top_with_sim(tmp_path, vout_mean=0.91))
    r_b = generate_report(_top_with_sim(tmp_path, vout_mean=0.87))
    diff = diff_reports(r_a, r_b)
    changed_names = [d.name for d in diff.signal_diff.changed]
    assert 'vout' in changed_names
    vout = next(d for d in diff.signal_diff.changed if d.name == 'vout')
    assert abs(vout.delta_mean - (-0.04)) < 1e-6


def test_diff_signal_added(tmp_path):
    r_a = generate_report(_top_with_sim(tmp_path))
    r_b = generate_report(_top_with_sim(tmp_path, extra_signals={'vnew': 0.5}))
    diff = diff_reports(r_a, r_b)
    assert 'vnew' in diff.signal_diff.added


def test_diff_signal_dropped(tmp_path):
    r_a = generate_report(_top_with_sim(tmp_path, extra_signals={'vold': 0.5}))
    r_b = generate_report(_top_with_sim(tmp_path))
    diff = diff_reports(r_a, r_b)
    assert 'vold' in diff.signal_diff.dropped


def test_diff_no_simulation_signal_diff_is_none(tmp_path):
    r = generate_report(_top_no_sim(tmp_path))
    diff = diff_reports(r, r)
    assert diff.signal_diff is None


# --- Topology ---

def test_diff_topology_unchanged(tmp_path):
    r = generate_report(_top_with_sim(tmp_path))
    diff = diff_reports(r, r)
    assert diff.topology_diff.instances_added   == []
    assert diff.topology_diff.instances_removed == []
    assert diff.topology_diff.nets_delta        == 0


def test_diff_topology_instance_added(tmp_path):
    comp = _sub(tmp_path, 'comp', {'vp': PortDir.IN})
    top_a = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top_a.module_name = 'top'
    top_a.add_submodule('comp1', comp)
    top_b = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top_b.module_name = 'top'
    top_b.add_submodule('comp1', comp)
    top_b.add_submodule('comp2', comp)
    diff = diff_reports(generate_report(top_a), generate_report(top_b))
    assert 'comp2' in diff.topology_diff.instances_added


# --- Sections filter ---

def test_diff_sections_signals_only(tmp_path):
    r = generate_report(_top_with_sim(tmp_path))
    diff = diff_reports(r, r, sections={'signals'})
    assert diff.signal_diff     is not None
    assert diff.topology_diff   is None
    assert diff.annotation_diff is None
    assert diff.violation_diff  is None


def test_diff_sections_topology_only(tmp_path):
    r = generate_report(_top_with_sim(tmp_path))
    diff = diff_reports(r, r, sections={'topology'})
    assert diff.signal_diff     is None
    assert diff.topology_diff   is not None
    assert diff.annotation_diff is None
    assert diff.violation_diff  is None


# --- JSON path loading ---

def test_diff_from_json_paths(tmp_path):
    r = generate_report(_top_with_sim(tmp_path))
    path_a = str(tmp_path / 'a.json')
    path_b = str(tmp_path / 'b.json')
    r.save_json(path_a)
    r.save_json(path_b)
    diff = diff_reports(path_a, path_b)
    assert isinstance(diff, ReportDiff)
    assert diff.signal_diff.changed == []


# --- Output ---

def test_diff_print_summary_contains_labels(tmp_path, capsys):
    r = generate_report(_top_with_sim(tmp_path))
    diff = diff_reports(r, r, label_a='sch', label_b='ext')
    diff.print_summary()
    out = capsys.readouterr().out
    assert 'sch' in out
    assert 'ext' in out


def test_diff_save_creates_md_file(tmp_path):
    r = generate_report(_top_with_sim(tmp_path))
    diff = diff_reports(r, r)
    out = str(tmp_path / 'diff.md')
    diff.save(out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_diff_save_md_contains_labels(tmp_path):
    r = generate_report(_top_with_sim(tmp_path))
    diff = diff_reports(r, r, label_a='schematic', label_b='extracted')
    out = str(tmp_path / 'diff.md')
    diff.save(out)
    text = open(out).read()
    assert 'schematic' in text
    assert 'extracted' in text


def test_diff_one_sided_simulation_signal_diff_is_none(tmp_path):
    r_sim   = generate_report(_top_with_sim(tmp_path))
    r_nosim = generate_report(_top_no_sim(tmp_path))
    diff = diff_reports(r_sim, r_nosim)
    assert diff.signal_diff is None


# ---------------------------------------------------------------------------
# results-file review 2026-08-24: diff_reports() on saved/loaded JSON paths
# must match diffing the same reports in memory, for layout_diff and
# spec_transitions too — previously load_json silently dropped both
# spec_reconciliation and layout, so diff_reports('a.json', 'b.json') (the
# module docstring's own documented usage) reported layout_diff=None and
# spec_transitions=[] even when the underlying reports differed.
# ---------------------------------------------------------------------------

def _report_with_layout_and_spec(tmp_path, r_ohm, status):
    from silk.layout.estimator import LayoutEstimate, NetLayout
    from silk.specs import Spec, Measurement, SpecOutcome, SpecReconciliationReport

    top = _top_no_sim(tmp_path)
    report = generate_report(top)
    report.layout_estimate = LayoutEstimate(
        computation_mode='estimated',
        routing_model_id='m1',
        timestamp=1.0,
        bounding_box=(10.0, 10.0),
        nets={'vp': NetLayout(length=5.0, width=0.5, r=r_ohm, c=1e-15,
                               equivalence_class=None)},
        devices={},
        equivalence_classes={},
        warnings=(),
    )
    spec = Spec(name='gain', module=top, metric='gain_db', op='>=', target=20.0)
    measurement = Measurement(source='sim', value=25.0, timestamp=1.0)
    outcome = SpecOutcome(spec=spec, measurement=measurement, status=status,
                           source_used='sim')
    report.spec_reconciliation = SpecReconciliationReport(
        module=top, outcomes=[outcome], timestamp=1.0,
    )
    return report


def test_diff_layout_and_spec_round_trip_via_json_matches_in_memory(tmp_path):
    """The regression test for the save_json/load_json bug: diffing two
    reports loaded from JSON must find the same layout-net change and the
    same spec status transition as diffing the same two reports in memory."""
    report_a = _report_with_layout_and_spec(tmp_path, r_ohm=100.0, status='FAIL')
    report_b = _report_with_layout_and_spec(tmp_path, r_ohm=1000.0, status='PASS')

    diff_mem = diff_reports(report_a, report_b)

    path_a = str(tmp_path / 'a.json')
    path_b = str(tmp_path / 'b.json')
    report_a.save_json(path_a)
    report_b.save_json(path_b)
    diff_file = diff_reports(path_a, path_b)

    assert diff_mem.layout_diff is not None
    assert diff_file.layout_diff is not None
    assert len(diff_mem.layout_diff.changed) == 1
    assert diff_mem.layout_diff.changed == diff_file.layout_diff.changed
    assert diff_file.layout_diff.changed[0].name == 'vp'
    assert diff_file.layout_diff.changed[0].r_a == 100.0
    assert diff_file.layout_diff.changed[0].r_b == 1000.0

    assert diff_mem.spec_transitions == diff_file.spec_transitions
    assert len(diff_file.spec_transitions) == 1
    t = diff_file.spec_transitions[0]
    assert t['spec_name'] == 'gain'
    assert t['before_status'] == 'FAIL'
    assert t['after_status'] == 'PASS'


def test_diff_layout_one_sided_is_none(tmp_path):
    """Existing None-handling: when only one report carries a layout
    estimate, layout_diff stays None rather than raising."""
    report_a = generate_report(_top_no_sim(tmp_path))
    report_b = _report_with_layout_and_spec(tmp_path, r_ohm=1000.0, status='PASS')
    assert report_a.layout_estimate is None
    diff = diff_reports(report_a, report_b)
    assert diff.layout_diff is None


def test_diff_spec_transitions_one_sided(tmp_path):
    """Existing None-handling: a spec present only in B still produces a
    transition with before_status=None, rather than raising."""
    report_a = generate_report(_top_no_sim(tmp_path))  # no specs declared
    report_b = _report_with_layout_and_spec(tmp_path, r_ohm=1000.0, status='PASS')
    diff = diff_reports(report_a, report_b)
    assert len(diff.spec_transitions) == 1
    assert diff.spec_transitions[0]['spec_name'] == 'gain'
    assert diff.spec_transitions[0]['before_status'] is None
    assert diff.spec_transitions[0]['after_status'] == 'PASS'


def test_top_level_import():
    from silk import diff_reports, ReportDiff
    assert callable(diff_reports)


# ---------------------------------------------------------------------------
# TASK-101: kind-aware diff — AC/noise reports carry their statistics in
# SimulationSummary.magnitudes (TASK-82), not .signals. Before this fix
# _diff_signals walked only .signals, so two AC (or noise) reports always
# diffed as an empty SignalDiff — changed/added/dropped all [] — reporting
# "Signals: unchanged" for a before/after AC characterization that DID
# change. These reports are built through the real load path (Module ->
# read_raw_file() -> generate_report), on the TASK-74 ngspice fixtures, the
# same fixtures test_report_simulation_kinds.py uses for TASK-82.
# ---------------------------------------------------------------------------

def _report_from_fixture(pdk, tmp_dir, fixture, perturb=None, factor=2.0):
    """Load *fixture* through the real path and summarise it via
    generate_report(). If *perturb* is given (a raw vector name, e.g.
    'v(out)'), that vector's data is scaled by *factor* AFTER loading but
    BEFORE summarising, so the two reports built from the "same" fixture
    differ only in that one node's statistics."""
    mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
    mod.dataFile = str(FIXTURES / fixture)
    mod.read_raw_file()
    if perturb is not None:
        for vec in mod.simResultsAll.vectors:
            if vec.name == perturb:
                vec.data = vec.data * factor
                break
        else:
            raise AssertionError(f'{perturb!r} not found in {fixture} vectors')
    return generate_report(mod, auto_discover_config=False)


def test_diff_ac_reports_reports_changed_magnitudes(pdk, tmp_dir):
    """Pre-fix: both reports' SimulationSummary.signals are [] (TASK-82
    moved AC statistics to .magnitudes), so diff_reports produced an empty
    SignalDiff — changed=[] — even though v(out)'s magnitude was scaled by
    2x. Post-fix: signal_diff is a MagnitudeDiff, 'out' is reported changed
    with the doubled mag_min/mag_max/mag_mean, and 'p1'/'in' (untouched)
    are unchanged."""
    report_a = _report_from_fixture(pdk, tmp_dir, 'e_ng_ac.raw')
    report_b = _report_from_fixture(pdk, tmp_dir, 'e_ng_ac.raw',
                                     perturb='v(out)', factor=2.0)
    diff = diff_reports(report_a, report_b)

    sd = diff.signal_diff
    assert isinstance(sd, MagnitudeDiff)
    assert sd.added == [] and sd.dropped == []
    changed_names = [d.name for d in sd.changed]
    assert changed_names == ['out']
    assert 'p1' in sd.unchanged
    assert 'in' in sd.unchanged

    out = next(d for d in sd.changed if d.name == 'out')
    assert isinstance(out, MagnitudeDelta)
    assert out.mag_min_b == pytest.approx(2.0 * out.mag_min_a)
    assert out.mag_max_b == pytest.approx(2.0 * out.mag_max_a)
    assert out.mag_mean_b == pytest.approx(2.0 * out.mag_mean_a)
    assert out.delta_mag_mean == pytest.approx(out.mag_mean_a)


def test_diff_identical_ac_reports_diff_empty(pdk, tmp_dir):
    """Two identical AC reports still diff empty post-fix — a MagnitudeDiff
    with nothing changed/added/dropped and every node unchanged — so the
    fix does not turn every AC diff into a false positive."""
    report_a = _report_from_fixture(pdk, tmp_dir, 'e_ng_ac.raw')
    report_b = _report_from_fixture(pdk, tmp_dir, 'e_ng_ac.raw')
    diff = diff_reports(report_a, report_b)

    sd = diff.signal_diff
    assert isinstance(sd, MagnitudeDiff)
    assert sd.changed == []
    assert sd.added == []
    assert sd.dropped == []
    assert sorted(sd.unchanged) == ['in', 'out', 'p1']


def test_diff_noise_reports_reports_changed_magnitudes(pdk, tmp_dir):
    """Same kind-aware path for a noise report: onoise_spectrum's density
    doubled is reported as a changed magnitude, the per-device breakdown
    vectors (untouched) are unchanged. Pre-fix this was also an empty
    SignalDiff for the same reason as the AC case."""
    report_a = _report_from_fixture(pdk, tmp_dir, 'e_ng_noise.raw')
    report_b = _report_from_fixture(pdk, tmp_dir, 'e_ng_noise.raw',
                                     perturb='onoise_spectrum', factor=1.5)
    diff = diff_reports(report_a, report_b)

    sd = diff.signal_diff
    assert isinstance(sd, MagnitudeDiff)
    changed_names = [d.name for d in sd.changed]
    assert changed_names == ['onoise_spectrum']
    onoise = next(d for d in sd.changed if d.name == 'onoise_spectrum')
    assert onoise.mag_mean_b == pytest.approx(1.5 * onoise.mag_mean_a)
    assert 'inoise_spectrum' in sd.unchanged


def test_diff_transient_vs_ac_reports_kind_mismatch(pdk, tmp_dir):
    """A transient report diffed against an AC report cannot be compared
    node-for-node (different statistics live under different fields), so
    it must show up as a structural KindMismatch, never an empty or
    partial SignalDiff.

    Pre-fix: stats_a came from the transient report's .signals (non-empty)
    and stats_b from the AC report's .signals (empty, since TASK-82), so
    _diff_signals silently reported every transient signal as 'dropped'
    with no indication the two runs were different analyses at all —
    neither 'no change' nor a mismatch, just a wrong-looking topology-style
    diff of unrelated data."""
    report_a = _report_from_fixture(pdk, tmp_dir, 'e_ng_tran.raw')
    report_b = _report_from_fixture(pdk, tmp_dir, 'e_ng_ac.raw')
    diff = diff_reports(report_a, report_b)

    sd = diff.signal_diff
    assert isinstance(sd, KindMismatch)
    assert sd.analysis_a == 'transient'
    assert sd.analysis_b == 'ac'


def test_diff_ac_print_summary_shows_magnitudes_label(pdk, tmp_dir, capsys):
    """print_summary() labels an AC/noise diff 'Magnitudes', not 'Signals',
    so a reader is not misled into thinking these are voltages."""
    report_a = _report_from_fixture(pdk, tmp_dir, 'e_ng_ac.raw')
    report_b = _report_from_fixture(pdk, tmp_dir, 'e_ng_ac.raw',
                                     perturb='v(out)', factor=2.0)
    diff = diff_reports(report_a, report_b)
    diff.print_summary()
    out = capsys.readouterr().out
    assert 'Magnitudes' in out
    assert 'out' in out


def test_diff_kind_mismatch_print_summary(pdk, tmp_dir, capsys):
    """print_summary() states the kind mismatch explicitly rather than
    printing an unchanged/changed signals line."""
    report_a = _report_from_fixture(pdk, tmp_dir, 'e_ng_tran.raw')
    report_b = _report_from_fixture(pdk, tmp_dir, 'e_ng_ac.raw')
    diff = diff_reports(report_a, report_b)
    diff.print_summary()
    out = capsys.readouterr().out
    assert 'kind mismatch' in out
    assert 'transient' in out and 'ac' in out


def _identity_report(tmp_path, rows):
    from silk.specs import Spec, Measurement, SpecOutcome, SpecReconciliationReport
    report = generate_report(_top_no_sim(tmp_path))
    outcomes = []
    for owner, key, value, name, report_key in rows:
        outcome = SpecOutcome(Spec(name, None, 'gain', '>=', 60, key=key),
                              Measurement('user', value),
                              'PASS' if value >= 60 else 'FAIL')
        # Assignment lets these tests exercise the old diff before the new API.
        outcome.owner_path = owner
        outcome.report_key = report_key
        outcomes.append(outcome)
    report.spec_reconciliation = SpecReconciliationReport(None, outcomes)
    return report


@pytest.mark.parametrize('loaded', [False, True])
def test_same_spec_at_two_placements_is_not_collapsed(tmp_path, loaded):
    a = _identity_report(tmp_path, [('$/x1', 'gain', 50, 'Gain', 'old1'),
                                  ('$/x2', 'gain', 70, 'Gain', 'old2')])
    b = _identity_report(tmp_path, [('$/x1', 'gain', 70, 'Gain', 'new1'),
                                  ('$/x2', 'gain', 50, 'Gain', 'new2')])
    if loaded:
        a.save_json(tmp_path / 'a.json'); b.save_json(tmp_path / 'b.json')
        a, b = str(tmp_path / 'a.json'), str(tmp_path / 'b.json')
    diff = diff_reports(a, b)
    assert len(diff.spec_transitions) == 2
    assert [(r['owner_path'], r['spec_key'], r['report_key'])
            for r in diff.spec_transitions] == [
                ('$/x1', 'gain', 'new1'), ('$/x2', 'gain', 'new2')]


@pytest.mark.parametrize('side', ['before', 'after'])
@pytest.mark.parametrize('owner', [None, '$', '$/x1'])
@pytest.mark.parametrize('value', [50, 70])
def test_diff_rejects_duplicate_reconciliation_identity(tmp_path, side, owner, value):
    from silk.exceptions import ConfigurationError
    duplicate = _identity_report(tmp_path, [(owner, 'gain', 50, 'First', 'test1'),
                                          (owner, 'gain', value, 'Second', 'test2')])
    empty = _identity_report(tmp_path, [])
    args = (duplicate, empty) if side == 'before' else (empty, duplicate)
    with pytest.raises(ConfigurationError) as error:
        diff_reports(*args)
    message = str(error.value)
    for part in (side, owner or 'unknown', 'gain', 'First', 'Second',
                 'test1', 'test2', '1', '2'):
        assert part in message
    assert 'distinct' in message and 'duplicate' in message


@pytest.mark.parametrize('known', ['$', '$/x1'])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('loaded', [False, True])
def test_diff_unknown_owner_never_matches_known_owner(tmp_path, known, reverse, loaded):
    a = _identity_report(tmp_path, [(None, 'gain', 70, 'Gain', None)])
    b = _identity_report(tmp_path, [(known, 'gain', 70, 'Gain', None)])
    if reverse:
        a, b = b, a
    if loaded:
        a.save_json(tmp_path / 'a.json'); b.save_json(tmp_path / 'b.json')
        a, b = str(tmp_path / 'a.json'), str(tmp_path / 'b.json')
    rows = diff_reports(a, b).spec_transitions
    assert [r['owner_path'] for r in rows] == [None, known]
    assert [(r['before_status'], r['after_status']) for r in rows] == (
        [(None, 'PASS'), ('PASS', None)] if reverse
        else [('PASS', None), (None, 'PASS')])


def test_diff_unique_unknown_rows_keep_unknown_identity(tmp_path, capsys):
    a = _identity_report(tmp_path, [(None, 'gain', 50, 'Gain', None)])
    b = _identity_report(tmp_path, [(None, 'gain', 70, 'Gain', None)])
    diff = diff_reports(a, b)
    assert len(diff.spec_transitions) == 1
    assert diff.spec_transitions[0]['owner_path'] is None
    assert 'unknown placement' in str(diff)
    diff.print_summary()
    assert 'unknown placement' in capsys.readouterr().out


def test_diff_mixed_owner_order_is_deterministic(tmp_path):
    empty = _identity_report(tmp_path, [])
    records = [('$/z', 'b', 70, 'Gain', None), (None, 'b', 70, 'Gain', None),
               ('$', 'a', 70, 'Gain', None), (None, 'a', 70, 'Gain', None),
               ('$/a', 'a', 70, 'Gain', None)]
    for inputs in (records, list(reversed(records))):
        rows = diff_reports(empty, _identity_report(tmp_path, inputs)).spec_transitions
        assert [(r['owner_path'], r['spec_key']) for r in rows] == [
            (None, 'a'), (None, 'b'), ('$', 'a'), ('$/a', 'a'), ('$/z', 'b')]


def test_diff_transition_identity_labels(tmp_path, capsys):
    a = _identity_report(tmp_path, [('$/x1', 'old', 50, 'Gain', 'before')])
    b = _identity_report(tmp_path, [('$/x1', 'new', 70, 'Gain', 'after')])
    diff = diff_reports(a, b)
    assert len(diff.spec_transitions) == 2
    assert [r['report_key'] for r in diff.spec_transitions] == ['after', 'before']
    diff.print_summary()
    for text in (str(diff), capsys.readouterr().out):
        for value in ('$/x1', 'old', 'new', 'Gain'):
            assert value in text


@pytest.mark.parametrize('before,after', [
    (PortLocation('vin'), PortLocation('vout')),
    (InstanceLocation('u0/x1'), InstanceLocation('u1/x1')),
    (TerminalLocation('u0/x1', 'g'), TerminalLocation('u1/x1', 'g')),
    (ModuleLocation('left'), ModuleLocation('right')),
    (NetLocation('vin'), NetLocation('vout')),
    (NetLocation('same'), PortLocation('same')),
    (PortLocation('same', 'left'), PortLocation('same', 'right')),
])
def test_violation_location_move(tmp_path, capsys, before, after):
    a = generate_report(_top_no_sim(tmp_path))
    b = generate_report(_top_no_sim(tmp_path))
    old = InterfaceViolation('INT-001', Severity.WARNING, 'm', before)
    new = InterfaceViolation('INT-001', Severity.WARNING, 'm', after)
    a.violations, b.violations = [old], [new]
    diff = diff_reports(a, b, sections={'violations'})
    assert diff.violation_diff.resolved == [old]
    assert diff.violation_diff.introduced == [new]
    assert diff.violation_diff.unchanged == []
    assert diff.violation_diff.resolved[0] is old
    assert diff.violation_diff.introduced[0] is new
    assert 'Violations: 1 introduced, 1 resolved, 0 unchanged' in str(diff)
    diff.print_summary()
    assert 'Violations: 1 introduced, 1 resolved, 0 unchanged' in capsys.readouterr().out


@pytest.mark.parametrize('saved', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
def test_violation_distinct_locations_and_saved_paths(tmp_path, reverse, saved):
    a = generate_report(_top_no_sim(tmp_path))
    b = generate_report(_top_no_sim(tmp_path))
    # Unsorted, with several same-rule ports and the same leaf across kinds.
    a.violations = [InterfaceViolation('INT-001', Severity.WARNING, 'm', loc)
                    for loc in [PortLocation('same'), NetLocation('same'),
                                TerminalLocation('same', 'g'), ModuleLocation('same'),
                                InstanceLocation('same'), PortLocation('other'),
                                PortLocation('same', 'left'), PortLocation('same', 'right')]]
    b.violations = []
    expected = ['instance:same', 'module:same', 'net:same', 'port:left.same',
                'port:other', 'port:right.same', 'port:same', 'terminal:same.g']
    if reverse:
        a, b = b, a
    path_a, path_b = str(tmp_path / 'a.json'), str(tmp_path / 'b.json')
    a.save_json(path_a)
    b.save_json(path_b)
    first, second = (path_a, path_b) if saved else (a, b)
    vd = diff_reports(first, second, sections={'violations'}).violation_diff
    assert [v.location.describe() for v in vd.introduced] == (expected if reverse else [])
    assert [v.location.describe() for v in vd.resolved] == ([] if reverse else expected)
    assert vd.unchanged == []


@pytest.mark.parametrize('before,after', [
    (NetLocation('n', ['a']), NetLocation('n', ['b'])),
    (TerminalLocation('u/x', 'g', ('g', 's')), TerminalLocation('u/x', 'g', ('g', 'd'))),
    (PortLocation('p', 'm'), PortLocation('p', 'm')),
])
def test_violation_identity_ignores_details_and_keeps_last_entry(tmp_path, before, after):
    a = generate_report(_top_no_sim(tmp_path))
    b = generate_report(_top_no_sim(tmp_path))
    first = InterfaceViolation('INT-001', Severity.WARNING, 'first', before)
    last = InterfaceViolation('INT-001', Severity.ERROR, 'last', before, {'old': 1})
    new = InterfaceViolation('INT-001', Severity.INFO, 'changed', after, {'new': 2})
    a.violations, b.violations = [first, last], [new]
    vd = diff_reports(a, b, sections={'violations'}).violation_diff
    assert vd.resolved == vd.introduced == []
    assert len(vd.unchanged) == 1
    assert vd.unchanged[0] is last
    # Rule identity still distinguishes two findings at exactly the same location.
    new.rule_id = 'INT-002'
    vd = diff_reports(a, b, sections={'violations'}).violation_diff
    assert vd.resolved == [last]
    assert vd.introduced == [new]
    assert vd.unchanged == []


def test_violation_net_only_objects_order_and_markdown_bytes(tmp_path):
    a = generate_report(_top_no_sim(tmp_path))
    b = generate_report(_top_no_sim(tmp_path))
    a.violations = [InterfaceViolation('INT-001', Severity.WARNING, 'old', NetLocation(n))
                    for n in ['z', 'keep', 'a']]
    b.violations = [InterfaceViolation('INT-001', Severity.ERROR, 'new', NetLocation(n, ['p']))
                    for n in ['y', 'keep', 'b']]
    diff = diff_reports(a, b, sections={'violations'})
    vd = diff.violation_diff
    assert len(vd.resolved) == len(vd.introduced) == 2
    assert len(vd.unchanged) == 1
    assert vd.resolved[0] is a.violations[2]
    assert vd.resolved[1] is a.violations[0]
    assert vd.introduced[0] is b.violations[2]
    assert vd.introduced[1] is b.violations[0]
    assert vd.unchanged[0] is a.violations[1]
    path = tmp_path / 'net.md'
    diff.save(str(path))
    assert path.read_bytes().split(b'## Interface Violations\n', 1)[1] == (
        b'\n### Introduced (2)\n\n'
        b'| Severity | Kind | Net | Message |\n|---|---|---|---|\n'
        b'| 30 | INT-001 | b | new |\n| 30 | INT-001 | y | new |\n\n'
        b'### Resolved (2)\n\n'
        b'| Severity | Kind | Net | Message |\n|---|---|---|---|\n'
        b'| 20 | INT-001 | a | old |\n| 20 | INT-001 | z | old |\n')


@pytest.mark.parametrize('reverse', [False, True])
def test_violation_mixed_markdown_tables_choose_header_independently(tmp_path, reverse):
    a = generate_report(_top_no_sim(tmp_path))
    b = generate_report(_top_no_sim(tmp_path))
    a.violations = [InterfaceViolation('INT-001', Severity.WARNING, 'old', NetLocation('old'))]
    b.violations = [InterfaceViolation('INT-001', Severity.WARNING, 'new', loc)
                    for loc in [NetLocation('new'), PortLocation('p', 'm')]]
    if reverse:
        a, b = b, a
    diff = diff_reports(a, b, sections={'violations'})
    path = tmp_path / 'mixed.md'
    diff.save(str(path))
    text = path.read_text()
    mixed, net = ('Resolved', 'Introduced') if reverse else ('Introduced', 'Resolved')
    assert (f'### {mixed} (2)\n\n| Severity | Kind | Location | Message |\n'
            '|---|---|---|---|\n| 20 | INT-001 | net:new | new |\n'
            '| 20 | INT-001 | port:m.p | new |') in text
    assert (f'### {net} (1)\n\n| Severity | Kind | Net | Message |\n'
            '|---|---|---|---|\n| 20 | INT-001 | old | old |') in text


def test_violation_base_from_custom_registry_diff_and_save(tmp_path, monkeypatch):
    from silk.lint.registry import RuleRegistry
    from silk.lint.rules.interface import DriverConflictRule

    rule = DriverConflictRule()
    # A custom check on an existing rule instance avoids another Rule subclass.
    monkeypatch.setattr(rule, 'check', lambda scope: iter([
        Violation('INT-001', Severity.WARNING, 'custom', PortLocation(scope.module_name))]))
    registry = RuleRegistry(rules=[rule])
    a_top, b_top = _top_no_sim(tmp_path), _top_no_sim(tmp_path)
    a_top.module_name, b_top.module_name = 'vin', 'vout'
    a = generate_report(a_top, registry=registry, auto_discover_config=False)
    b = generate_report(b_top, registry=registry, auto_discover_config=False)
    assert type(a.violations[0]) is Violation
    diff = diff_reports(a, b, sections={'violations'})
    assert diff.violation_diff.resolved == a.violations
    assert diff.violation_diff.introduced == b.violations
    assert diff.violation_diff.unchanged == []
    path = tmp_path / 'custom.md'
    diff.save(str(path))
    text = path.read_text()
    assert '| 20 | INT-001 | port:vin | custom |' in text
    assert '| 20 | INT-001 | port:vout | custom |' in text
