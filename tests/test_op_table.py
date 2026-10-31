"""``SpecReport.op_table()`` (TASK-330.4) — per-device operating-point rows
built from ``Module.save_device_params``'s captured data (TASK-330.3).

``TestRegionRuleSynthetic`` is PDK-free (synthetic records, no simulator).
Everything else needs a real ngspice + PDK (``needs_pdk``/``needs_ngspice``),
mirroring ``tests/test_op_point_params.py``'s own real-run coverage; the sky130
values are cross-checked against TASK-330.1's hand-read fixture
(``tests/fixtures/op_point/op_probe.sp``) within 1e-6 relative (parent AC#2).
"""
import math
import os

import pytest

from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.circuit.op_table import RegionThresholds, build_op_table
from silk.exceptions import SimulationError
from silk import sources
from silk.simulation import analysis

from tests._env import (needs_pdk, needs_pdk_named, needs_ngspice,
                        needs_osdi_runnable, HAVE_XYCE)
from tests.test_op_point_params import _nested_two_level_bench
from tests._env import scratch_dir

_NGSPICE_47 = os.path.expanduser('~/ngspice/ngspice-47/bin/ngspice')


def _ngspice_bins():
    bins = [None]
    if os.path.isfile(_NGSPICE_47) and os.access(_NGSPICE_47, os.X_OK):
        bins.append(_NGSPICE_47)
    return bins


_UNSET = object()


# TASK-491: geometry bins read from the installed model cards; biases stay
# inside the catalog's junction ratings. A missing/wrong declaration must
# fail on actual OpTable values, not just a metadata equality check.
_SKY130_EXTRA_MOS = [
    ('nmos1p8_lvt', 'n', 1, 1, 1.2, 1.8),
    ('pmos1p8_lvt', 'p', 1, 1, 1.2, 1.8),
    ('pmos1p8_hvt', 'p', 1.26, .15, 1.2, 1.8),
    ('nmos3p3_native', 'n', 10, .5, 1, 3),
    ('nmos5v', 'n', 10, .5, 3, 3),
    ('nmos5v_native', 'n', 10, 2, 1, 3),
    ('pmos5v', 'p', 10, .5, 3, 3),
    ('nmos_esd', 'n', 17.5, .55, 3, 3),
    ('pmos_esd', 'p', 14.5, .55, 3, 3),
]
_SKY130_HIGH_VOLTAGE_MOS = [
    ('nmos20v', 'n', 60, 2, 3, 10),
    ('nmos20v_native', 'n', 60, 2, 1, 10),
    ('nmos20v_zvt', 'n', 60, 4, 3, 10),
    ('nmos20v_iso', 'n', 60, 2, 3, 10),
    ('pmos20v', 'p', 50, 2, 3, 10),
]


@pytest.mark.slow
@needs_pdk
@pytest.mark.parametrize('kind,polarity,w,l,vg,vd,backend', [
    pytest.param(*case, 'ngspice', marks=needs_ngspice, id=case[0] + '-ngspice')
    for case in _SKY130_EXTRA_MOS + _SKY130_HIGH_VOLTAGE_MOS
] + [
    pytest.param(*case, 'xyce', id=case[0] + '-xyce', marks=[
        pytest.mark.requires_xyce,
        pytest.mark.skipif(not HAVE_XYCE, reason='needs Xyce')])
    for case in _SKY130_EXTRA_MOS
])
def test_additional_sky130_mos_op_table(tmp_path, backend, kind, polarity, w, l, vg, vd):
    """Catch missing saves, wrong primitive names, polarity and silent holes.

    The independent drain voltage source measures terminal current, so the
    current/sign check does not repeat OpTable's own calculation.
    """
    from silk.pdk import load_pdk
    pdk = load_pdk('sky130')
    dut = Module('sky130', str(tmp_path), 'dut.cir', pdk_details=pdk, module_name='dut')
    dut.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN,
                             's': PortDir.GROUND, 'b': PortDir.GROUND})
    extra_pins = {'sub': 's'} if kind == 'nmos20v_iso' else {}
    dut.add(getattr(pdk, kind)('M1', d='d', g='g', s='s', b='b', w=w, l=l, m=1,
                             **extra_pins))
    dut.simulator = backend
    tb = dut.create_test('opchk')
    tb.add_module_under_test({'d': 'vd', 'g': 'vg', 's': '0', 'b': '0'})
    sign = 1 if polarity == 'n' else -1
    tb.add_source(sources.VoltageSource('vd', positive='vd', negative='0', dc=sign * vd))
    tb.add_source(sources.VoltageSource('vg', positive='vg', negative='0', dc=sign * vg))
    tb.probe('i(vvd)')
    tb.add_sim_type(analysis.Op())
    dut.save_device_params = True
    report = dut.run_test('opchk', corner='tt')
    assert os.path.isfile(report.rawfile_path)
    row = report.op_table()['dut.M1']
    assert row.polarity == polarity
    assert row.region == ('triode' if kind == 'nmos20v_iso' else 'saturation')
    assert row.id > 1e-8
    branch = float(tb.netlist.load_signal(
        'vvd', 'I', analysis='dc' if backend == 'xyce' else 'op').y[0])
    # Terminal current includes small junction/body currents as well as id.
    assert row.id == pytest.approx(-sign * branch, rel=2e-3, abs=1e-12)
    if '20v' in kind:
        # m1 is behind the wrapper's drift/series circuitry. These are its
        # intrinsic operating-point values, not external terminal voltages.
        assert .98 * vg < row.vgs <= vg
        assert 0 < row.vds < vd
        assert -.02 < row.vbs < .001
    else:
        assert row.vgs == pytest.approx(vg)
        assert row.vds == pytest.approx(vd)
        # Intrinsic bulk potential includes the body-resistance drop.
        assert row.vbs == pytest.approx(0, abs=1e-4)
    assert row.vgs > row.vth
    assert row.vdsat > 0
    if kind == 'nmos20v_iso':
        assert row.vds < row.vdsat
    else:
        assert row.vds > row.vdsat
    available = {'id', 'vgs', 'vds', 'vbs', 'vth', 'vdsat', 'gm', 'gds'}
    holes = {'cgso', 'cgdo', 'cgbo'}
    extra = {'gmbs', 'cgg', 'cgs', 'cgd', 'capbd', 'capbs'}
    if kind in ('nmos20v_native', 'nmos20v_zvt'):
        # Only zero was observed on these wrappers' intrinsic drain caps.
        extra.remove('capbd')
        holes.add('capbd')
    if backend == 'ngspice':
        available |= extra
        assert row.cgg > 0
        assert row.ft > 0
    else:
        holes |= extra | {'ft'}
    for column in available:
        assert row.sources[column] == 'sim', column
        assert math.isfinite(getattr(row, column)), column
        if column != 'vbs':  # a source-tied bulk may legitimately read zero
            assert getattr(row, column) != 0, column
    for column in holes:
        assert row.sources[column] is None, column
        assert math.isnan(getattr(row, column)), column
    assert row.gm > 0
    assert row.gds > 0


def _record(path='xm1', name=_UNSET, model='m', polarity='n', declared=(), values=None, **geom):
    """``name`` defaults to ``_UNSET`` rather than ``None`` so a caller who
    doesn't pass it gets a record with NO ``'name'`` key at all -- the
    actual shape of a pre-TASK-330.7 ``op_devices`` record (never a present
    key holding ``None``), which is what the backward-compatibility ruling
    asks be tested against."""
    base = {p: float('nan') for p in (
        'id', 'vgs', 'vds', 'vbs', 'vth', 'vdsat', 'gm', 'gds', 'gmbs',
        'cgg', 'cgs', 'cgd', 'cgso', 'cgdo', 'cgbo', 'capbd', 'capbs')}
    if values:
        base.update(values)
    rec = {'path': path, 'model': model, 'polarity': polarity,
           'w_um': geom.get('w_um'), 'l_um': geom.get('l_um'), 'm': geom.get('m'),
           'declared_params': tuple(declared), 'op_values': base}
    if name is not _UNSET:
        rec['name'] = name
    return rec


# ---------------------------------------------------------------------------
# PDK-free: region rule + derived columns on synthetic values.
# ---------------------------------------------------------------------------
class TestRegionRuleSynthetic:
    @pytest.mark.parametrize('engine', ['ngspice', 'xyce'])
    def test_bench_inventory_roundtrip_helpers(self, tmp_path, engine):
        from silk.pdk import load_pdk
        from silk.circuit.modules_results import capture_op_device_records
        from silk.circuit.spec_report import SpecReport
        from silk.simulation.backend import SimulatorBackend, DEVICE_OP_PARAMS
        pdk = load_pdk('sky130')
        block = Module('sky130', str(tmp_path), 'block.cir', pdk_details=pdk,
                       module_name='block')
        block.add(pdk.nmos1p8('MN', d='d', g='g', s='s', b='s', w=1, l=.15))
        unknown = pdk.nmos1p8('MU', d='d', g='g', s='s', b='s', w=1, l=.15)
        unknown._op_params = ()
        block.add(unknown)
        dut = Module('sky130', str(tmp_path), 'dut.cir', pdk_details=pdk, module_name='dut')
        dut.add_module(block, 'bias1', nets={})
        bench = Module('sky130', str(tmp_path), 'bench.cir', pdk_details=pdk)
        bench.add_module(block, 'bias1', nets={})
        bench.add_module(dut, 'dut', nets={})
        bench.add_module(block, 'bias2', nets={})
        nested = Module('sky130', str(tmp_path), 'nested.cir',
                        pdk_details=pdk, module_name='nested')
        nested.add_module(block, 'inner', nets={})
        bench.add_module(nested, 'bias3', nets={})
        from silk.circuit.modules_results import device_op_save_lines, xyce_op_print_terms
        saved = (device_op_save_lines(bench.elem_dict) if engine == 'ngspice'
                 else xyce_op_print_terms(bench.elem_dict))
        assert saved
        assert not any('xmmu' in line.lower() for line in saved)
        assert any('xbias3' in line.lower() for line in saved)
        backend = SimulatorBackend.get(engine)
        calls = []
        def read(raw, path, leaf, **kwargs):
            calls.append((path, leaf))
            return {p: float('nan') if leaf == 'XMMU' else 1.0
                    for p in DEVICE_OP_PARAMS}
        backend.device_op_vectors = read
        records = capture_op_device_records(bench, backend, None, pdk=pdk)
        expected = ['bias1.MN', 'bias1.MU', 'dut.bias1.MN',
                    'dut.bias1.MU', 'bias2.MN', 'bias2.MU',
                    'bias3.inner.MN', 'bias3.inner.MU']
        assert [r['name'] for r in records] == expected
        assert [r['path'] for r in records] == [
            'xbias1.XMMN', 'xbias1.XMMU', 'xdut.xbias1.XMMN',
            'xdut.xbias1.XMMU', 'xbias2.XMMN', 'xbias2.XMMU',
            'xbias3.xinner.XMMN', 'xbias3.xinner.XMMU']
        assert calls[2] == ('xdut.xbias1', 'XMMN')
        table = build_op_table(records, None, None)
        for name in ['bias1.MU', 'dut.bias1.MU', 'bias2.MU', 'bias3.inner.MU']:
            row = table[name]
            assert row.region is None
            assert all(v is None for v in row.sources.values())
            assert math.isnan(row.id)
        assert records[1]['declared_params'] == ()
        assert all(math.isnan(v) for v in records[1]['op_values'].values())
        assert table['bias1.MN'].sources['id'] == 'sim'
        report = SpecReport('opchk', 'dut', {}, op_devices=records)
        restored = SpecReport.from_dict(report.to_dict()).op_table()
        assert [r.name for r in restored] == expected
        assert [r.path for r in restored] == [r['path'] for r in records]
        assert restored['bias1.MN'].region == table['bias1.MN'].region

    def test_saturation(self):
        rec = _record(declared=('vgs', 'vth', 'vds', 'vdsat', 'id', 'gm', 'gds', 'cgg'),
                       values={'vgs': 1.0, 'vth': 0.4, 'vds': 0.9, 'vdsat': 0.2,
                               'id': 1e-4, 'gm': 2e-4, 'gds': 1e-5, 'cgg': 1e-15})
        table = build_op_table([rec], None, None)
        row = table[0]
        assert row.region == 'saturation'
        assert row.saturation_weak is False
        assert row.vov == pytest.approx(0.6)
        assert row.vds_margin == pytest.approx(0.7)
        assert row.gm_id == pytest.approx(2.0)
        assert row.gm_gds == pytest.approx(20.0)
        assert row.ro == pytest.approx(1e5)
        assert row.ft == pytest.approx(2e-4 / (2 * math.pi * 1e-15))
        assert row.sources['ft'] == 'derived_estimate'
        assert row.sources['vov'] == 'derived'
        assert row.sources['vgs'] == 'sim'

    def test_saturation_weak_flag(self):
        rec = _record(declared=('vgs', 'vth', 'vds', 'vdsat', 'id'),
                       values={'vgs': 1.0, 'vth': 0.4, 'vds': 0.22, 'vdsat': 0.2,
                               'id': 1e-4})
        row = build_op_table([rec], None, None)[0]
        assert row.region == 'saturation'
        assert row.saturation_weak is True

    def test_triode(self):
        rec = _record(declared=('vgs', 'vth', 'vds', 'vdsat', 'id'),
                       values={'vgs': 1.0, 'vth': 0.4, 'vds': 0.05, 'vdsat': 0.2,
                               'id': 1e-4})
        row = build_op_table([rec], None, None)[0]
        assert row.region == 'triode'
        assert row.saturation_weak is False

    def test_subthreshold(self):
        rec = _record(declared=('vgs', 'vth', 'vds', 'vdsat', 'id'),
                       values={'vgs': 0.35, 'vth': 0.4, 'vds': 0.9, 'vdsat': 0.05,
                               'id': 1e-7})
        row = build_op_table([rec], None, None)[0]
        assert row.region == 'subthreshold'

    def test_cutoff_by_vgs(self):
        rec = _record(declared=('vgs', 'vth', 'vds', 'vdsat', 'id'),
                       values={'vgs': 0.0, 'vth': 0.4, 'vds': 0.9, 'vdsat': 0.05,
                               'id': 1e-15})
        row = build_op_table([rec], None, None)[0]
        assert row.region == 'cutoff'

    def test_cutoff_by_low_current(self):
        """vgs is NOT below vth - margin, but |id| < min_current_a alone
        still cuts off (owner's 'or' rule)."""
        rec = _record(declared=('vgs', 'vth', 'vds', 'vdsat', 'id'),
                       values={'vgs': 0.39, 'vth': 0.4, 'vds': 0.9, 'vdsat': 0.05,
                               'id': 1e-15})
        row = build_op_table([rec], None, None)[0]
        assert row.region == 'cutoff'

    def test_region_none_on_missing_input(self):
        """Undeclared device: every raw column nan/source None, region is
        None, never a default bucket (reviewer ruling)."""
        rec = _record(declared=())
        row = build_op_table([rec], None, None)[0]
        assert row.region is None
        assert row.saturation_weak is None
        for p in ('id', 'vgs', 'vds', 'vth', 'vdsat', 'gm', 'gds', 'cgg',
                  'cgso', 'cgdo', 'cgbo'):
            assert row.sources[p] is None
            assert math.isnan(getattr(row, p))
        for d in ('vov', 'gm_id', 'gm_gds', 'ro', 'ft', 'vds_margin'):
            assert row.sources[d] is None
            assert math.isnan(getattr(row, d))

    def test_custom_thresholds_override(self):
        """min_current_a only ever gates the CUTOFF check when vgs < vth --
        a device with vgs >= vth is never cutoff on current alone (reviewer
        review of b7639171), so this exercises the override on a genuinely
        subthreshold bias, not an on-switch."""
        rec = _record(declared=('vgs', 'vth', 'vds', 'vdsat', 'id'),
                       values={'vgs': 0.35, 'vth': 0.4, 'vds': 0.9, 'vdsat': 0.2,
                               'id': 1e-7})
        strict = RegionThresholds(min_current_a=1.0)   # any real current cuts off
        row = build_op_table([rec], strict, None)[0]
        assert row.region == 'cutoff'

    def test_on_switch_with_no_current_is_never_cutoff(self):
        """Reviewer review of b7639171: a fully-ON switch/pass device
        carrying near-zero current because its LOAD is off (not because the
        device itself is off) must not read 'cutoff'. Real values from the
        reviewer's own sky130 nested bench (xst2.XMMP): vgs=1.64 > vth=0.74,
        vds ~ 0 << vdsat=0.64, id=1.6e-11 (tiny, but vgs is well above vth)."""
        rec = _record(declared=('vgs', 'vth', 'vds', 'vdsat', 'id'),
                       values={'vgs': 1.63975, 'vth': 0.738665,
                               'vds': 5.49511e-08, 'vdsat': 0.643507,
                               'id': 1.57231e-11})
        row = build_op_table([rec], None, None)[0]
        assert row.region == 'triode'          # vds << vdsat, vgs firmly on
        assert row.region != 'cutoff'

    def test_region_thresholds_default_uses_report_temperature(self):
        class _FakeConditions:
            temperature_c = 125.0
        hot = RegionThresholds.at_temperature(_FakeConditions())
        cold = RegionThresholds.at_temperature(None)
        assert hot.subthreshold_margin_v != cold.subthreshold_margin_v
        assert hot.subthreshold_margin_v > cold.subthreshold_margin_v  # higher T -> larger kT/q

    def test_rendering_str_and_markdown(self):
        rec = _record(declared=('vgs', 'vth', 'vds', 'vdsat', 'id', 'gm', 'gds', 'cgg'),
                       values={'vgs': 1.0, 'vth': 0.4, 'vds': 0.9, 'vdsat': 0.2,
                               'id': 1e-4, 'gm': 2e-4, 'gds': 1e-5, 'cgg': 1e-15})
        table = build_op_table([rec], None, None)
        text = str(table)
        assert 'region' in text and 'saturation' in text
        md = table.to_markdown()
        # TASK-330.7: the designer's own name leads, 'path' (the emitted
        # SPICE path) stays right after it.
        assert md.startswith('| name | path')
        assert '---' in md

    def test_rendering_shows_saturation_weak(self):
        """saturation_weak is a real OpRow field, but review of b7639171
        found it wasn't rendered at all -- __str__/to_markdown must show it
        (folded into the region cell as 'saturation (weak)'), not just carry
        it silently on the dataclass."""
        rec = _record(declared=('vgs', 'vth', 'vds', 'vdsat', 'id'),
                       values={'vgs': 1.0, 'vth': 0.4, 'vds': 0.22, 'vdsat': 0.2,
                               'id': 1e-4})
        table = build_op_table([rec], None, None)
        assert table[0].saturation_weak is True
        assert 'saturation (weak)' in str(table)
        assert 'saturation (weak)' in table.to_markdown()
        # A comfortably-saturated device (not weak) must NOT carry the
        # marker, so the two are visually distinguishable.
        comfy = _record(declared=('vgs', 'vth', 'vds', 'vdsat', 'id'),
                        values={'vgs': 1.0, 'vth': 0.4, 'vds': 0.9, 'vdsat': 0.2,
                                'id': 1e-4})
        comfy_table = build_op_table([comfy], None, None)
        assert comfy_table[0].saturation_weak is False
        assert 'saturation (weak)' not in str(comfy_table)

    def test_empty_table_renders_without_crashing(self):
        table = build_op_table([], None, None)
        assert len(table) == 0
        assert list(table) == []
        assert 'no MOS devices' in str(table)
        assert 'no MOS devices' in table.to_markdown()

    def test_declared_but_nan_stays_sim(self):
        """A declared parameter that happens to read nan on this bias (a
        genuine simulator reading, not a catalog gap) keeps source 'sim' --
        declared-ness, not nan-ness, is what the marker records. It also
        keeps rendering the literal 'nan' text -- TASK-330.6's reviewer
        ruling only replaces 'nan' with the placeholder for a true HOLE
        (source None); a real nan-valued reading is not a hole."""
        rec = _record(declared=('cgso',), values={'cgso': float('nan')})
        table = build_op_table([rec], None, None)
        row = table[0]
        assert row.sources['cgso'] == 'sim'
        assert math.isnan(row.cgso)
        assert 'nan' in str(table)
        assert 'nan' in table.to_markdown()

    def test_undeclared_hole_renders_placeholder_not_nan(self):
        """TASK-330.6, reviewer ruling on AC #5: a column the model/backend
        never supplied (source None, e.g. Xyce's gmbs/capacitances) renders
        the SAME '—' placeholder as a None-valued field (path/model/region),
        never the literal string 'nan' -- on console AND Markdown, not just
        HTML, since _cell_text is the single shared renderer for all three.
        cgg is undeclared here, so cgg AND ft (which needs cgg) are both
        holes; id IS declared, so it stays a real reading."""
        rec = _record(declared=('id',), values={'id': 1e-4})
        table = build_op_table([rec], None, None)
        row = table[0]
        assert row.sources['cgg'] is None
        assert math.isnan(row.cgg)
        assert row.sources['ft'] is None
        assert math.isnan(row.ft)
        text = str(table)
        md = table.to_markdown()
        for rendering in (text, md):
            assert 'nan' not in rendering
        # id's own declared, real reading is unaffected by the hole fix.
        assert '0.0001' in text

    # -----------------------------------------------------------------
    # TASK-330.7: designer name -- lookup, filter, backward compatibility.
    # -----------------------------------------------------------------

    def test_lookup_by_designer_name(self):
        rec1 = _record(path='xst1.XMMN', name='st1.MN',
                       declared=('id',), values={'id': 1e-4})
        rec2 = _record(path='xst2.XMMN', name='st2.MN',
                       declared=('id',), values={'id': 2e-4})
        table = build_op_table([rec1, rec2], None, None)
        assert table['st1.MN'].path == 'xst1.XMMN'
        assert table['st2.MN'].path == 'xst2.XMMN'
        # Positional indexing is untouched.
        assert table[0].name == 'st1.MN'
        assert table[1].name == 'st2.MN'

    def test_lookup_case_sensitive_names_coexist(self):
        """'MN' and 'mn' are different names in the same scope (silk's own
        add_element/add_module never conflate them -- Python dict keys are
        case-sensitive), so both rows must coexist and be looked up exactly,
        never folded together."""
        rec_upper = _record(path='XMMN', name='MN', declared=('id',), values={'id': 1e-4})
        rec_lower = _record(path='XMmn', name='mn', declared=('id',), values={'id': 2e-4})
        table = build_op_table([rec_upper, rec_lower], None, None)
        assert table['MN'].path == 'XMMN'
        assert table['mn'].path == 'XMmn'

    def test_unknown_name_raises_with_close_matches(self):
        rec = _record(path='xst1.XMMN', name='st1.MN', declared=('id',), values={'id': 1e-4})
        table = build_op_table([rec], None, None)
        with pytest.raises(KeyError, match='st1.MN'):
            table['st1.mn']          # a case typo -- close, not exact

    def test_unknown_name_with_no_close_match_lists_all_known(self):
        rec = _record(path='xst1.XMMN', name='st1.MN', declared=('id',), values={'id': 1e-4})
        table = build_op_table([rec], None, None)
        with pytest.raises(KeyError, match='st1.MN'):
            table['totally_unrelated']

    def test_duplicate_designer_name_raises_at_build_time(self):
        rec1 = _record(path='xa.XMMN', name='st1.MN', declared=('id',), values={'id': 1e-4})
        rec2 = _record(path='xb.XMMN', name='st1.MN', declared=('id',), values={'id': 2e-4})
        with pytest.raises(ValueError, match='st1.MN'):
            build_op_table([rec1, rec2], None, None)

    def test_backward_compatible_with_pre_330_7_records(self):
        """A hand-built OLD-style record list -- no 'name' key at all, the
        exact shape an op_devices inventory captured before TASK-330.7
        has. build_op_table must still build a valid table; rows fall back
        to name=None (rendered as the '-' placeholder); string lookup
        refuses with a message naming the reason, not a bare KeyError."""
        old_style = [{'path': 'xm1', 'model': 'sky130_fd_pr__nfet_01v8',
                      'polarity': 'n', 'w_um': 1.0, 'l_um': 0.15, 'm': 1,
                      'declared_params': ('id',), 'op_values': {'id': 1e-4}}]
        assert 'name' not in old_style[0]
        table = build_op_table(old_style, None, None)
        row = table[0]
        assert row.name is None
        assert row.path == 'xm1'
        assert '—' in str(table)            # name column renders the hole
        with pytest.raises(KeyError, match='captured before designer names existed'):
            table['xm1']

    def test_filter_by_region(self):
        cutoff = _record(path='xst2.XMMN', name='st2.MN',
                         declared=('vgs', 'vth', 'vds', 'vdsat', 'id'),
                         values={'vgs': 0.0, 'vth': 0.4, 'vds': 0.9, 'vdsat': 0.2, 'id': 0.0})
        sat = _record(path='xst1.XMMN', name='st1.MN',
                      declared=('vgs', 'vth', 'vds', 'vdsat', 'id'),
                      values={'vgs': 1.0, 'vth': 0.4, 'vds': 0.9, 'vdsat': 0.2, 'id': 1e-4})
        table = build_op_table([cutoff, sat], None, None)
        cutoff_rows = table.filter(region='cutoff')
        assert isinstance(cutoff_rows, type(table))
        assert [r.name for r in cutoff_rows] == ['st2.MN']
        assert cutoff_rows['st2.MN'].name == 'st2.MN'       # still name-indexable

    def test_filter_by_saturation_weak(self):
        weak = _record(path='xst1.XMMN', name='st1.MN',
                       declared=('vgs', 'vth', 'vds', 'vdsat', 'id'),
                       values={'vgs': 1.0, 'vth': 0.4, 'vds': 0.22, 'vdsat': 0.2, 'id': 1e-4})
        comfy = _record(path='xst2.XMMN', name='st2.MN',
                        declared=('vgs', 'vth', 'vds', 'vdsat', 'id'),
                        values={'vgs': 1.0, 'vth': 0.4, 'vds': 0.9, 'vdsat': 0.2, 'id': 1e-4})
        table = build_op_table([weak, comfy], None, None)
        weak_rows = table.filter(saturation_weak=True)
        assert [r.name for r in weak_rows] == ['st1.MN']

    def test_filter_predicate_expresses_cutoff_or_weak(self):
        cutoff = _record(path='xst2.XMMN', name='st2.MN',
                         declared=('vgs', 'vth', 'vds', 'vdsat', 'id'),
                         values={'vgs': 0.0, 'vth': 0.4, 'vds': 0.9, 'vdsat': 0.2, 'id': 0.0})
        weak = _record(path='xst1.XMMN', name='st1.MN',
                       declared=('vgs', 'vth', 'vds', 'vdsat', 'id'),
                       values={'vgs': 1.0, 'vth': 0.4, 'vds': 0.22, 'vdsat': 0.2, 'id': 1e-4})
        comfy = _record(path='xst3.XMMN', name='st3.MN',
                        declared=('vgs', 'vth', 'vds', 'vdsat', 'id'),
                        values={'vgs': 1.0, 'vth': 0.4, 'vds': 0.9, 'vdsat': 0.2, 'id': 1e-4})
        table = build_op_table([cutoff, weak, comfy], None, None)
        rows = table.filter(lambda r: r.region == 'cutoff' or r.saturation_weak)
        assert sorted(r.name for r in rows) == ['st1.MN', 'st2.MN']

    def test_filter_unknown_region_raises(self):
        rec = _record(path='xm1', name='m1', declared=('id',), values={'id': 1e-4})
        table = build_op_table([rec], None, None)
        with pytest.raises(ValueError, match='saturated'):
            table.filter(region='saturated')      # typo for 'saturation'

    def test_filter_predicate_and_keyword_mutually_exclusive(self):
        rec = _record(path='xm1', name='m1', declared=('id',), values={'id': 1e-4})
        table = build_op_table([rec], None, None)
        with pytest.raises(TypeError):
            table.filter(lambda r: True, region='cutoff')


# ---------------------------------------------------------------------------
# Real ngspice: end to end via run_test(), cross-checked against 330.1.
# ---------------------------------------------------------------------------
@pytest.mark.slow
@needs_pdk
@needs_ngspice
@pytest.mark.parametrize('ngspice_bin', _ngspice_bins(),
                         ids=lambda b: 'ngspice47' if b else 'ngspice42')
class TestOpTableRealNgspice:
    def _single_nfet_bench(self, monkeypatch, ngspice_bin):
        """Reproduces tests/fixtures/op_point/op_probe.sp's exact NFET bias
        (Vg=1.0 Vd=1.0 Vs=Vb=0, w=1 l=0.15) so the 330.1 hand-read values
        apply unchanged (parent AC#2's 1e-6 relative check)."""
        if ngspice_bin:
            monkeypatch.setenv('SILK_NGSPICE_BIN', ngspice_bin)
        else:
            monkeypatch.delenv('SILK_NGSPICE_BIN', raising=False)
        from silk.pdk import load_pdk
        pdk = load_pdk('sky130')
        tmp = scratch_dir()
        dut = Module('sky130', tmp, 'dut.cir', pdk_details=pdk, module_name='dut')
        dut.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN,
                                 's': PortDir.GROUND, 'b': PortDir.GROUND})
        dut.add(pdk.nmos1p8('M1', d='d', g='g', s='s', b='b', w=1.0, l=0.15, m=1))
        t = dut.create_test('opchk')
        t.add_module_under_test({'d': 'vd', 'g': 'vg', 's': '0', 'b': '0'})
        t.add_source(sources.VoltageSource('vd', positive='vd', negative='0', dc=1.0))
        t.add_source(sources.VoltageSource('vg', positive='vg', negative='0', dc=1.0))
        t.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
        dut.save_device_params = True
        return dut

    def test_values_match_330_1_hand_read_fixture(self, monkeypatch, ngspice_bin):
        dut = self._single_nfet_bench(monkeypatch, ngspice_bin)
        report = dut.run_test('opchk', corner='tt')
        table = report.op_table()
        assert len(table) == 1
        row = table[0]
        assert row.path == 'xdut.XMM1'                      # bench-rooted
        assert row.model == 'sky130_fd_pr__nfet_01v8'
        assert row.polarity == 'n'
        # TASK-330.1's hand-read values (Vgs=1.0V Vds=1.0V, same geometry).
        # The reference values are themselves only recorded to 6 significant
        # figures, so a strict 1e-6 relative bound would fail purely on the
        # reference's OWN rounding (measured: up to ~1.4e-6 relative on
        # vdsat/gds against the full-precision value) -- 2e-6 keeps the
        # comparison meaningful while tolerating that truncation.
        expected = {'gm': 3.825087e-04, 'vth': 0.769391,
                    'vdsat': 0.171507, 'gds': 2.46322e-05, 'id': 6.94215e-05}
        for name, value in expected.items():
            got = getattr(row, name)
            assert got == pytest.approx(value, rel=2e-6), f"{name}: {got} vs {value}"
            assert row.sources[name] == 'sim'

    def test_switch_off_raises_naming_save_device_params(self, monkeypatch, ngspice_bin):
        dut = self._single_nfet_bench(monkeypatch, ngspice_bin)
        dut.save_device_params = False
        report = dut.run_test('opchk', corner='tt')
        assert report.op_devices is None
        with pytest.raises(SimulationError, match='save_device_params'):
            report.op_table()

    def test_region_thresholds_none_uses_reports_own_temperature(self, monkeypatch, ngspice_bin):
        dut = self._single_nfet_bench(monkeypatch, ngspice_bin)
        report = dut.run_test('opchk', corner='tt')
        table = report.op_table()
        expected = RegionThresholds.at_temperature(report.operating_conditions)
        assert table.region_thresholds.subthreshold_margin_v == pytest.approx(
            expected.subthreshold_margin_v)

    def test_op_only_bench_cgso_cgdo_cgbo_are_nan_never_a_silent_zero(self, monkeypatch, ngspice_bin):
        """TASK-330.4 gate finding regression, at the op_table() level (not
        just the backend seam -- see test_op_point_params.py::test_op_only
        for the same check one layer down): an Op()-only sky130 bench (no
        preceding .tran in the same ngspice session) is exactly the case
        where BSIM4 reported cgso/cgdo/cgbo as a SILENT 0 rather than
        nan/unavailable, while every other declared column read identically
        to a .tran bench. Undeclaring the three makes them nan/None here
        unconditionally, regardless of which analysis actually ran."""
        if ngspice_bin:
            monkeypatch.setenv('SILK_NGSPICE_BIN', ngspice_bin)
        else:
            monkeypatch.delenv('SILK_NGSPICE_BIN', raising=False)
        from silk.pdk import load_pdk
        pdk = load_pdk('sky130')
        tmp = scratch_dir()
        dut = Module('sky130', tmp, 'dut.cir', pdk_details=pdk, module_name='dut')
        dut.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN,
                                 's': PortDir.GROUND, 'b': PortDir.GROUND})
        dut.add(pdk.nmos1p8('M1', d='d', g='g', s='s', b='b', w=1.0, l=0.15, m=1))
        t = dut.create_test('opchk')
        t.add_module_under_test({'d': 'vd', 'g': 'vg', 's': '0', 'b': '0'})
        t.add_source(sources.VoltageSource('vd', positive='vd', negative='0', dc=1.0))
        t.add_source(sources.VoltageSource('vg', positive='vg', negative='0', dc=1.0))
        t.add_sim_type(analysis.Op())            # Op()-only -- no preceding .tran
        dut.save_device_params = True
        report = dut.run_test('opchk', corner='tt')
        table = report.op_table()
        assert len(table) == 1
        row = table[0]
        for p in ('cgso', 'cgdo', 'cgbo'):
            assert row.sources[p] is None, f"{p}: must be undeclared (None), never 'sim'"
            value = getattr(row, p)
            assert math.isnan(value), f"{p}: must be nan, got {value!r} (a silent zero, not nan)"
        # Every other declared column stays a real, non-nan reading.
        for p in ('id', 'vgs', 'vds', 'vth', 'vdsat', 'gm', 'gds', 'gmbs',
                  'cgg', 'cgs', 'cgd', 'capbd', 'capbs'):
            assert row.sources[p] == 'sim'
            assert not math.isnan(getattr(row, p)), f"{p} should be readable"

    def test_generate_report_html_includes_op_table(self, monkeypatch, ngspice_bin):
        """TASK-330.6: generate_report(spec_reports=[...]) sources its
        Operating Point Table section from a real ngspice SpecReport
        (330.4's capture), never re-deriving it. Checks the section
        heading, the hole placeholder on an undeclared-by-ngspice-op-only
        column, and the region class -- one real end-to-end pass through
        report.py's new seam."""
        from silk.circuit.report import generate_report
        dut = self._single_nfet_bench(monkeypatch, ngspice_bin)
        report = dut.run_test('opchk', corner='tt')
        assert report.op_devices is not None

        circuit_report = generate_report(dut, auto_discover_config=False,
                                         spec_reports=[report])
        html = circuit_report._to_html()
        assert 'Operating Point Table — opchk (tt)' in html
        assert 'class="op-table"' in html
        assert 'sky130_fd_pr__nfet_01v8' in html
        assert 'class="op-region-saturation"' in html


@pytest.mark.slow
@needs_pdk
@needs_ngspice
class TestOpTableNestedHierarchy:
    """op_table() over the reviewer's own nested dut -> st1/st2 -> MN/MP
    bench (same shape TASK-330.3's gate finding was reproduced with)."""

    def test_sky130_14_of_17_columns_dut_relative_paths(self):
        """TASK-330.4 gate finding: cgso/cgdo/cgbo are undeclared on sky130
        too (BSIM4 fills them only after a transient has run in the same
        ngspice session; outside one they read a SILENT 0, which the arc
        forbids) -- same 14/17 shape as gf180mcu, for a different reason."""
        amp, t = _nested_two_level_bench('sky130', 'nmos1p8', 'pmos1p8',
                                         w=1.0, l=0.15, vdd=1.8)
        amp.save_device_params = True
        report = amp.run_test('opchk', corner='tt')
        table = report.op_table()
        assert len(table) == 4
        paths = sorted(row.path for row in table)
        assert paths == ['xdut.xst1.XMMN', 'xdut.xst1.XMMP', 'xdut.xst2.XMMN', 'xdut.xst2.XMMP']
        unavailable = {'cgso', 'cgdo', 'cgbo'}
        for row in table:
            assert row.sources['id'] == 'sim'
            assert row.polarity in ('n', 'p')
            expect_polarity = 'n' if 'mmn' in row.path.lower() else 'p'
            assert row.polarity == expect_polarity
            for p in unavailable:
                assert row.sources[p] is None
                assert math.isnan(getattr(row, p)), (
                    f"{row.path}.{p}: must be nan, never a silent 0")
            for p in set(('id', 'vgs', 'vds', 'vbs', 'vth', 'vdsat', 'gm',
                          'gds', 'gmbs', 'cgg', 'cgs', 'cgd', 'capbd', 'capbs')):
                assert row.sources[p] == 'sim'

    @pytest.mark.requires_gf180
    @needs_pdk_named('gf180mcu')
    def test_gf180mcu_14_of_17_columns(self):
        amp, t = _nested_two_level_bench('gf180mcu', 'nmos3p3', 'pmos3p3',
                                         w=1.0, l=0.28, vdd=3.3)
        amp.save_device_params = True
        report = amp.run_test('opchk', corner='typical')
        table = report.op_table()
        assert len(table) == 4
        unavailable = {'cgso', 'cgdo', 'cgbo'}
        for row in table:
            for p in unavailable:
                assert row.sources[p] is None
                assert math.isnan(getattr(row, p))
            for p in set(('id', 'vgs', 'vds', 'vbs', 'vth', 'vdsat', 'gm',
                          'gds', 'gmbs', 'cgg', 'cgs', 'cgd', 'capbd', 'capbs')):
                assert row.sources[p] == 'sim'

    def test_html_report_flags_cutoff_and_weak_saturation_rows(self):
        """TASK-330.6, reviewer ruling on AC #6: op-flag applies to BOTH
        weak-saturation AND cutoff rows, not saturation_weak alone --
        xdut.xst1.XMMN is the real weakly-saturated row and xdut.xst2.XMMN is the
        real cutoff row in this same reviewer bench (confirmed by a real
        run: xdut.xst1.XMMN saturation/weak=True, xdut.xst2.XMMN cutoff)."""
        from silk.circuit.report import generate_report
        amp, t = _nested_two_level_bench('sky130', 'nmos1p8', 'pmos1p8',
                                         w=1.0, l=0.15, vdd=1.8)
        amp.save_device_params = True
        report = amp.run_test('opchk', corner='tt')
        table = report.op_table()
        cutoff_rows = [row for row in table if row.region == 'cutoff']
        weak_rows = [row for row in table if row.saturation_weak]
        assert cutoff_rows, 'bench must still produce a real cutoff row'
        assert weak_rows, 'bench must still produce a real weak-saturation row'

        circuit_report = generate_report(amp, auto_discover_config=False,
                                         spec_reports=[report])
        html = circuit_report._to_html()
        for row in cutoff_rows + weak_rows:
            # TASK-330.7: 'name' (the designer's own name) now leads, ahead
            # of 'path' (the emitted SPICE path, still the next cell).
            assert f'<tr class="op-flag">\n<td>{row.name}</td>\n<td>{row.path}</td>' in html, (
                f"{row.name} ({row.path}, region={row.region!r}, "
                f"saturation_weak={row.saturation_weak}) must carry op-flag")

    @pytest.mark.requires_gf180
    @needs_pdk_named('gf180mcu')
    def test_pmos_signs_are_positive_magnitudes_gf180(self):
        """Ruling (D): gf180 PMOS verified separately
        (tests/fixtures/op_point/gf180mcu/pmos3p3_sign_check.*) -- this
        confirms the passthrough end to end through op_table() too."""
        amp, t = _nested_two_level_bench('gf180mcu', 'nmos3p3', 'pmos3p3',
                                         w=1.0, l=0.28, vdd=3.3)
        amp.save_device_params = True
        report = amp.run_test('opchk', corner='typical')
        table = report.op_table()
        pmos_rows = [row for row in table if row.polarity == 'p']
        assert pmos_rows
        for row in pmos_rows:
            assert row.vgs >= 0.0
            assert row.vds >= 0.0

    @pytest.mark.requires_sg13g2
    @needs_osdi_runnable('sg13g2')
    def test_sg13g2_undeclared_all_none_region_none(self):
        amp, t = _nested_two_level_bench('sg13g2', 'nmos_lv', 'pmos_lv',
                                         w=1.0, l=0.13, vdd=1.2)
        amp.save_device_params = True
        with pytest.warns(UserWarning, match="save_device_params.*'sg13g2'"):
            report = amp.run_test('opchk', corner='typical')
        table = report.op_table()
        assert len(table) == 4
        for row in table:
            assert row.region is None
            for p in ('id', 'vgs', 'vds', 'vth', 'vdsat', 'gm', 'gds', 'cgg'):
                assert row.sources[p] is None
                assert math.isnan(getattr(row, p))

    def test_lookup_by_designer_name_matches_emitted_path_row(self):
        """TASK-330.7 AC #2/#(c): a real ngspice run of the nested bench
        (two instances of one stage) names rows by the designer's own
        instance/device names -- table['dut.st1.MN'] returns the SAME row the
        old emitted-path spelling ('xdut.xst1.XMMN') would have addressed, for
        all four devices."""
        amp, t = _nested_two_level_bench('sky130', 'nmos1p8', 'pmos1p8',
                                         w=1.0, l=0.15, vdd=1.8)
        amp.save_device_params = True
        report = amp.run_test('opchk', corner='tt')
        table = report.op_table()
        by_path = {row.path: row for row in table}
        assert set(by_path) == {'xdut.xst1.XMMN', 'xdut.xst1.XMMP', 'xdut.xst2.XMMN', 'xdut.xst2.XMMP'}
        expected = {'dut.st1.MN': 'xdut.xst1.XMMN', 'dut.st1.MP': 'xdut.xst1.XMMP',
                   'dut.st2.MN': 'xdut.xst2.XMMN', 'dut.st2.MP': 'xdut.xst2.XMMP'}
        for designer_name, emitted_path in expected.items():
            row = table[designer_name]
            assert row is by_path[emitted_path]
            assert row.name == designer_name
            assert row.path == emitted_path

    def test_filter_on_real_bench_cutoff_and_weak(self):
        """TASK-330.7 AC #3: the filter exercised on a real bench with at
        least one cutoff and one weakly-saturated device -- the same
        reviewer bench TASK-330.6's op-flag test already measured this on
        (xdut.xst1.XMMN weakly saturated, xdut.xst2.XMMN cutoff)."""
        amp, t = _nested_two_level_bench('sky130', 'nmos1p8', 'pmos1p8',
                                         w=1.0, l=0.15, vdd=1.8)
        amp.save_device_params = True
        report = amp.run_test('opchk', corner='tt')
        table = report.op_table()

        cutoff_rows = table.filter(region='cutoff')
        assert len(cutoff_rows) >= 1
        assert all(row.region == 'cutoff' for row in cutoff_rows)

        weak_rows = table.filter(saturation_weak=True)
        assert len(weak_rows) >= 1
        assert all(row.saturation_weak for row in weak_rows)

        either = table.filter(lambda r: r.region == 'cutoff' or r.saturation_weak)
        assert {r.name for r in either} == {r.name for r in cutoff_rows} | {r.name for r in weak_rows}

        with pytest.raises(KeyError):
            table['dut.st1.mn']              # case typo -- must not silently match


@pytest.mark.slow
@needs_pdk
@needs_ngspice
class TestAnchoredLookup:
    """Ruling (B): the absolute path (DUT instance + relative path) anchors
    the '@' lookup. A HELPER module placed beside the DUT (not inside it),
    sharing the DUT's own relative sub-path name, proves this matters: an
    unanchored relative-only search would find both and come back
    ambiguous (nan) for a device that is really perfectly resolvable."""

    @pytest.mark.parametrize('engine', ['ngspice', pytest.param(
        'xyce', marks=[pytest.mark.requires_xyce,
                      pytest.mark.skipif(not HAVE_XYCE, reason='needs Xyce')])])
    def test_helper_module_with_same_subpath_does_not_confuse_the_dut_row(self, engine):
        from silk.pdk import load_pdk
        pdk = load_pdk('sky130')
        tmp = scratch_dir()
        stage = Module('sky130', tmp, 'stage.cir', pdk_details=pdk, module_name='stage')
        stage.define_module_ports({'vdd': PortDir.INOUT, 'in': PortDir.IN,
                                   'out': PortDir.OUT, 'vss': PortDir.GROUND})
        stage.add(pdk.nmos1p8('MN', d='out', g='in', s='vss', b='vss', w=1.0, l=0.15))
        stage.add(pdk.pmos1p8('MP', d='out', g='in', s='vdd', b='vdd', w=1.0, l=0.15))

        dut = Module('sky130', tmp, 'dut.cir', pdk_details=pdk, module_name='dut')
        dut.define_module_ports({'vdd': PortDir.INOUT, 'in': PortDir.IN,
                                 'out': PortDir.OUT, 'vss': PortDir.GROUND})
        dut.add_module(stage, 'bias1', nets={'vdd': 'vdd', 'in': 'in', 'out': 'out', 'vss': 'vss'})

        dut.simulator = engine
        t = dut.create_test('opchk')
        t.add_module_under_test({'vdd': 'vdd', 'in': 'in', 'out': 'out', 'vss': '0'})
        # Same block name on each side of the DUT boundary, at distinct
        # biases, catches both an ambiguous match and borrowed readings.
        t.add_module(stage, 'bias1', nets={'vdd': 'vdd', 'in': 'helper_in', 'out': 'helper_out', 'vss': '0'})
        t.add_source(sources.VoltageSource('helper_in', positive='helper_in', negative='0', dc=0.6))
        t.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
        t.add_source(sources.VoltageSource('vin', positive='in', negative='0', dc=0.9))
        t.add_sim_type(analysis.Op())
        dut.save_device_params = True

        report = dut.run_test('opchk', corner='tt')
        table = report.op_table()
        assert len(table) == 4
        paths = sorted(row.path for row in table)
        assert paths == ['xbias1.XMMN', 'xbias1.XMMP',
                         'xdut.xbias1.XMMN', 'xdut.xbias1.XMMP']
        assert table['dut.bias1.MN'].vgs == pytest.approx(0.9)
        assert table['bias1.MN'].vgs == pytest.approx(0.6)
        assert table['dut.bias1.MN'].id != pytest.approx(table['bias1.MN'].id)
        from silk.circuit.spec_report import SpecReport
        from silk.circuit.report import generate_report
        restored = SpecReport.from_dict(report.to_dict())
        os.remove(report.rawfile_path)
        assert restored.op_table()['bias1.MN'].vgs == pytest.approx(0.6)
        html = generate_report(dut, auto_discover_config=False,
                               spec_reports=[restored])._to_html()
        for row in restored.op_table():
            assert f'<td>{row.name}</td>' in html
            assert f'<td>{row.path}</td>' in html
        for row in table:
            assert row.sources['id'] == 'sim'
            assert not math.isnan(row.id), (
                "anchored lookup must resolve the DUT's own device despite "
                "a same-named helper module existing alongside it")


# ---------------------------------------------------------------------------
# Real Xyce (TASK-330.5): N()/ID() columns, the 8-name capability narrowing,
# the Op()-only restriction, and an undeclared device completing with a
# rawfile. Skips cleanly where Xyce / the sky130 PDK are absent (mirrors
# tests/test_xyce_netlister.py::TestXyceEndToEnd's own skip pattern).
# ---------------------------------------------------------------------------
_XYCE_BIN = os.path.expanduser('~/xyce/Xyce-7.10.0/bin/Xyce')
_SKY130_LIB = os.path.expanduser(
    '~/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice')
_have_xyce = os.path.exists(_XYCE_BIN) and os.path.exists(_SKY130_LIB)


@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.skipif(not _have_xyce, reason='Xyce binary or sky130 PDK not present')
class TestOpTableRealXyce:
    """Real Xyce 7.10 runs through the actual silk path (Module -> run_test()
    -> XyceBackend -> real Xyce -> RawResult -> op_table()), not a hand-built
    deck -- TASK-330.1's ``xyce_probe_available.cir`` already proved the raw
    N()/ID() addressing works; this proves silk's own emission+capture seam
    (XyceNetlister.generate_output_block + XyceBackend.device_op_vectors +
    op_param_capability) agrees with it end to end."""

    def _single_bench(self, monkeypatch, devname, vg, vd, *, undeclared=False):
        monkeypatch.setenv('SILK_XYCE_BIN', _XYCE_BIN)
        from silk.pdk import load_pdk
        pdk = load_pdk('sky130')
        tmp = scratch_dir()
        dut = Module('sky130', tmp, 'dut.cir', pdk_details=pdk, module_name='dut')
        dut.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN,
                                 's': PortDir.GROUND, 'b': PortDir.GROUND})
        device = getattr(pdk, devname)('M1', d='d', g='g', s='s', b='b', w=1.0, l=0.15, m=1)
        if undeclared:
            device._op_backends = ('ngspice',)
        dut.add(device)
        dut.simulator = 'xyce'
        t = dut.create_test('opchk')
        t.add_module_under_test({'d': 'vd', 'g': 'vg', 's': '0', 'b': '0'})
        t.add_source(sources.VoltageSource('vd', positive='vd', negative='0', dc=vd))
        t.add_source(sources.VoltageSource('vg', positive='vg', negative='0', dc=vg))
        t.add_sim_type(analysis.Op())
        dut.save_device_params = True
        return dut

    def _single_bench_ngspice(self, devname, vg, vd):
        """Same bench as :meth:`_single_bench`, on ngspice -- a real
        cross-check reference, not a hand-read fixture, so a sign-convention
        test can compare like for like on the identical topology/bias."""
        from silk.pdk import load_pdk
        pdk = load_pdk('sky130')
        tmp = scratch_dir()
        dut = Module('sky130', tmp, 'dut.cir', pdk_details=pdk, module_name='dut')
        dut.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN,
                                 's': PortDir.GROUND, 'b': PortDir.GROUND})
        dut.add(getattr(pdk, devname)('M1', d='d', g='g', s='s', b='b', w=1.0, l=0.15, m=1))
        t = dut.create_test('opchk')
        t.add_module_under_test({'d': 'vd', 'g': 'vg', 's': '0', 'b': '0'})
        t.add_source(sources.VoltageSource('vd', positive='vd', negative='0', dc=vd))
        t.add_source(sources.VoltageSource('vg', positive='vg', negative='0', dc=vg))
        t.add_sim_type(analysis.Op())
        dut.save_device_params = True
        return dut

    def test_nmos_matches_330_1_hand_read_fixture(self, monkeypatch):
        """Same NFET bias as TestOpTableRealNgspice's own 330.1 cross-check
        (Vg=1.0 Vd=1.0 Vs=Vb=0, w=1 l=0.15) -- Xyce must agree with ngspice
        (and with 330.1's hand-read values) on the 8 columns it can read."""
        dut = self._single_bench(monkeypatch, 'nmos1p8', vg=1.0, vd=1.0)
        report = dut.run_test('opchk', corner='tt')
        table = report.op_table()
        assert len(table) == 1
        row = table[0]
        assert row.path == 'xdut.XMM1'
        assert row.model == 'sky130_fd_pr__nfet_01v8'
        assert row.polarity == 'n'
        expected = {'gm': 3.825087e-04, 'vth': 0.769391,
                    'vdsat': 0.171507, 'gds': 2.46322e-05, 'id': 6.94215e-05}
        for name, value in expected.items():
            got = getattr(row, name)
            assert got == pytest.approx(value, rel=2e-6), f"{name}: {got} vs {value}"
            assert row.sources[name] == 'sim'
        # gmbs and every capacitance are UNAVAILABLE on Xyce -- nan, source
        # None, never 'sim' regardless of what the model declares for
        # ngspice (ruling B's op_param_capability narrowing).
        for p in ('gmbs', 'cgg', 'cgs', 'cgd', 'cgso', 'cgdo', 'cgbo',
                  'capbd', 'capbs'):
            assert row.sources[p] is None, f"{p}: must be None on Xyce"
            assert math.isnan(getattr(row, p))
        assert row.sources['ft'] is None          # needs cgg, unavailable
        assert math.isnan(row.ft)
        assert row.region == 'saturation'

    def test_pmos_sign_convention_matches_ngspice(self, monkeypatch):
        """PMOS vgs/vds/id come back sign-normalised to the same
        source-referenced positive-magnitude convention 330.1 measured on
        ngspice (never re-derive this from a plain node subtraction).

        Reviewer gate finding (2026-09-27): Xyce's ID() uses the plain SPICE
        branch-current sign (current INTO the drain), negative for a PMOS in
        conduction -- a DIFFERENT convention from ngspice's own '@' vectors,
        which are already positive-magnitude for both polarities. An
        uncorrected read gave id=-1.98358e-05 on Xyce vs +1.98357e-05 on
        ngspice for the same device, and the derived gm_id went negative
        with it. device_op_vectors corrects this using the device's own
        declared polarity (never abs()); this test cross-checks id AND
        gm_id against a real ngspice run of the identical bench, not just
        a bare sign check."""
        ngspice_dut = self._single_bench_ngspice('pmos1p8', vg=-1.0, vd=-1.0)
        ngspice_row = ngspice_dut.run_test('opchk', corner='tt').op_table()[0]

        dut = self._single_bench(monkeypatch, 'pmos1p8', vg=-1.0, vd=-1.0)
        report = dut.run_test('opchk', corner='tt')
        table = report.op_table()
        assert len(table) == 1
        row = table[0]
        assert row.polarity == 'p'
        assert row.vgs > 0.0 and row.vds > 0.0    # positive-magnitude convention
        assert row.id > 0.0                       # positive magnitude, not raw SPICE branch sign
        assert row.gm_id > 0.0
        assert row.sources['vgs'] == 'sim'
        assert row.id == pytest.approx(ngspice_row.id, rel=1e-3)
        assert row.gm_id == pytest.approx(ngspice_row.gm_id, rel=1e-3)

    def test_transient_with_save_device_params_raises_not_supported(self, monkeypatch):
        """Ruling (A): only Op() may pair with save_device_params on Xyce --
        any other analysis would print device columns at every solve point."""
        from silk.exceptions import NotSupportedError
        monkeypatch.setenv('SILK_XYCE_BIN', _XYCE_BIN)
        from silk.pdk import load_pdk
        pdk = load_pdk('sky130')
        tmp = scratch_dir()
        dut = Module('sky130', tmp, 'dut.cir', pdk_details=pdk, module_name='dut')
        dut.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN,
                                 's': PortDir.GROUND, 'b': PortDir.GROUND})
        dut.add(pdk.nmos1p8('M1', d='d', g='g', s='s', b='b', w=1.0, l=0.15, m=1))
        dut.simulator = 'xyce'
        t = dut.create_test('opchk')
        t.add_module_under_test({'d': 'vd', 'g': 'vg', 's': '0', 'b': '0'})
        t.add_source(sources.VoltageSource('vd', positive='vd', negative='0', dc=1.0))
        t.add_source(sources.VoltageSource('vg', positive='vg', negative='0', dc=1.0))
        t.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
        dut.save_device_params = True
        with pytest.raises(NotSupportedError, match='save_device_params on Xyce requires Op'):
            dut.run_test('opchk', corner='tt')

    def test_undeclared_device_still_completes_with_a_rawfile(self, monkeypatch):
        """Ruling (D): an Op()-only Xyce bench with an UNDECLARED device
        (no op_backends listing 'xyce') gets no N()/ID() columns for it, but
        the run is not aborted -- Xyce's abort-on-one-bad-symbol failure
        mode is exactly why an unverified device must never be guessed at."""
        dut = self._single_bench(monkeypatch, 'nmos1p8_lvt', vg=1.0, vd=1.0,
                                 undeclared=True)
        with pytest.warns(UserWarning, match="save_device_params.*'xyce'"):
            report = dut.run_test('opchk', corner='tt')
        table = report.op_table()
        assert len(table) == 1
        row = table[0]
        assert row.model == 'sky130_fd_pr__nfet_01v8_lvt'
        assert row.region is None
        for p in ('id', 'vgs', 'vds', 'vth', 'vdsat', 'gm', 'gds'):
            assert row.sources[p] is None
            assert math.isnan(getattr(row, p))

    def test_generate_report_html_includes_op_table(self, monkeypatch):
        """TASK-330.6, real Xyce: the Operating Point Table section renders
        through the same generate_report(spec_reports=[...]) seam as
        ngspice, with Xyce's narrower column set showing up as hole cells
        (gmbs and every capacitance are outside XYCE_OP_CAPABILITY -- 330.5)
        and the PMOS id sign fix (330.5 gate finding) still holding: a
        positive id, never the raw SPICE branch-current sign."""
        from silk.circuit.report import generate_report
        dut = self._single_bench(monkeypatch, 'pmos1p8', vg=-1.0, vd=-1.0)
        report = dut.run_test('opchk', corner='tt')
        assert report.op_devices is not None
        row = report.op_table()[0]
        assert row.id > 0.0

        circuit_report = generate_report(dut, auto_discover_config=False,
                                         spec_reports=[report])
        html = circuit_report._to_html()
        assert 'Operating Point Table — opchk (tt)' in html
        assert 'class="op-table"' in html
        assert 'sky130_fd_pr__pfet_01v8' in html
        # gmbs is outside XYCE_OP_CAPABILITY -- always a hole on Xyce.
        assert 'class="op-hole"' in html

    def _nested_two_level_op_bench(self, monkeypatch):
        """Same dut(amp) -> st1/st2(stage) -> MN/MP topology as
        tests.test_op_point_params._nested_two_level_bench, but with an
        Op() analysis (ruling (A): save_device_params on Xyce requires
        Op(), never Transient()) -- TASK-330.7's AC #(c) real-run check on
        Xyce needs a nested bench, and the shared helper's Transient()
        can't pair with save_device_params here."""
        monkeypatch.setenv('SILK_XYCE_BIN', _XYCE_BIN)
        from silk.pdk import load_pdk
        pdk = load_pdk('sky130')
        tmp = scratch_dir()
        stage = Module('sky130', tmp, 'stage.cir', pdk_details=pdk, module_name='stage')
        stage.define_module_ports({'vdd': PortDir.INOUT, 'in': PortDir.IN,
                                   'out': PortDir.OUT, 'vss': PortDir.GROUND})
        stage.add(pdk.nmos1p8('MN', d='out', g='in', s='vss', b='vss', w=1.0, l=0.15))
        stage.add(pdk.pmos1p8('MP', d='out', g='in', s='vdd', b='vdd', w=1.0, l=0.15))
        amp = Module('sky130', tmp, 'amp.cir', pdk_details=pdk, module_name='amp')
        amp.define_module_ports({'vdd': PortDir.INOUT, 'in': PortDir.IN,
                                 'out': PortDir.OUT, 'vss': PortDir.GROUND})
        amp.add_module(stage, 'st1', nets={'vdd': 'vdd', 'in': 'in', 'out': 'mid', 'vss': 'vss'})
        amp.add_module(stage, 'st2', nets={'vdd': 'vdd', 'in': 'mid', 'out': 'out', 'vss': 'vss'})
        amp.simulator = 'xyce'
        t = amp.create_test('opchk')
        t.add_module_under_test({'vdd': 'vdd', 'in': 'in', 'out': 'out', 'vss': '0'})
        t.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
        t.add_source(sources.VoltageSource('vin', positive='in', negative='0', dc=0.9))
        t.add_sim_type(analysis.Op())
        amp.save_device_params = True
        return amp

    def test_nested_bench_lookup_by_designer_name_matches_emitted_path_row(self, monkeypatch):
        """TASK-330.7 AC #(c): the nested two-stage bench on REAL Xyce --
        table['dut.st2.MN'] returns the same row the old emitted path
        'xdut.xst2.XMMN' addresses, same as the ngspice cross-check
        (TestOpTableNestedHierarchy.test_lookup_by_designer_name_matches_emitted_path_row)."""
        amp = self._nested_two_level_op_bench(monkeypatch)
        report = amp.run_test('opchk', corner='tt')
        table = report.op_table()
        assert len(table) == 4
        by_path = {row.path: row for row in table}
        assert set(by_path) == {'xdut.xst1.XMMN', 'xdut.xst1.XMMP', 'xdut.xst2.XMMN', 'xdut.xst2.XMMP'}
        expected = {'dut.st1.MN': 'xdut.xst1.XMMN', 'dut.st1.MP': 'xdut.xst1.XMMP',
                   'dut.st2.MN': 'xdut.xst2.XMMN', 'dut.st2.MP': 'xdut.xst2.XMMP'}
        for designer_name, emitted_path in expected.items():
            row = table[designer_name]
            assert row is by_path[emitted_path]
            assert row.name == designer_name


# ---------------------------------------------------------------------------
# TASK-439: non-finite readings are None, ambiguous/absent devices warn.
# ---------------------------------------------------------------------------
class TestNonFiniteAndMatcherWarnings:
    _DECLARED = ('vgs', 'vth', 'vds', 'vdsat', 'id')

    def test_all_nan_row_has_no_region(self):
        rec = _record(declared=self._DECLARED + ('gm', 'cgg'))   # every value nan
        table = build_op_table([rec], None, None)
        row = table[0]
        assert row.region is None
        assert row.saturation_weak is None
        assert math.isnan(row.vov) and math.isnan(row.gm_id) and math.isnan(row.ft)
        assert '—' in str(table).splitlines()[-1]
        assert '—' in table.to_markdown().splitlines()[-1]

    @pytest.mark.parametrize('bad', ['vgs', 'vth', 'vds', 'vdsat', 'id'])
    def test_single_nan_input_is_none(self, bad):
        values = {'vgs': 1.0, 'vth': 0.4, 'vds': 0.9, 'vdsat': 0.2, 'id': 1e-4,
                  bad: float('nan')}
        row = build_op_table([_record(declared=self._DECLARED, values=values)],
                             None, None)[0]
        assert row.region is None

    def _vec(self, name, value=1.0):
        import numpy as np
        from types import SimpleNamespace
        return SimpleNamespace(name=name, data=np.array([value]))

    def _read(self, monkeypatch, names, path='xdut.xi', leaf='xm1'):
        from types import SimpleNamespace
        from silk.simulation import backend as be
        plot = SimpleNamespace(vectors=[self._vec(n) for n in names])
        monkeypatch.setattr(be, 'select_plot', lambda native, analysis: plot)
        raw = SimpleNamespace(native=None)
        return be.NgspiceBackend().device_op_vectors(raw, path, leaf)

    def test_ambiguous_match_warns_with_candidates(self, monkeypatch):
        from silk.metrics.diagnostics import MetricWarning
        names = ['@m.xdut.xi.xm1.msky130[vgs]', '@m.xdut.xi.xm1.other[vgs]']
        with pytest.warns(MetricWarning, match='xm1') as rec:
            out = self._read(monkeypatch, names)
        w = rec[0].message
        assert w.code == 'op_device_ambiguous'
        assert set(w.details['candidates']) == set(names)
        assert math.isnan(out['vgs'])

    @pytest.mark.parametrize('engine', ['ngspice', 'xyce'])
    @pytest.mark.parametrize('reverse', [False, True])
    @pytest.mark.parametrize('helper_present', [False, True])
    def test_rooted_helper_readback(self, monkeypatch, engine, reverse, helper_present):
        from types import SimpleNamespace
        from silk.simulation import backend as be
        if engine == 'ngspice':
            names = ['@m.xdut.xbias1.xmmn.msky130[vgs]',
                     '@m.xbias1.xmmn.msky130[vgs]']
        else:
            names = ['N(XDUT:XBIAS1:XMMN:MSKY130:VGS)',
                     'N(XBIAS1:XMMN:MSKY130:VGS)']
        vectors = [self._vec(names[0], 1.0)]
        if helper_present:
            vectors.append(self._vec(names[1], 0.6))
        if reverse:
            vectors.reverse()
        plot = SimpleNamespace(vectors=vectors)
        monkeypatch.setattr(be, 'select_plot', lambda native, analysis: plot)
        raw = SimpleNamespace(native=plot)
        backend = be.SimulatorBackend.get(engine)
        dut = backend.device_op_vectors(raw, 'xdut.xbias1', 'xmmn')
        assert dut['vgs'] == 1.0
        if engine == 'ngspice' and not helper_present:
            from silk.metrics.diagnostics import MetricWarning
            with pytest.warns(MetricWarning, match='xbias1'):
                helper = backend.device_op_vectors(raw, 'xbias1', 'xmmn')
        else:
            helper = backend.device_op_vectors(raw, 'xbias1', 'xmmn')
        if helper_present:
            assert helper['vgs'] == 0.6
        else:
            assert math.isnan(helper['vgs'])

    @pytest.mark.parametrize('engine', ['ngspice', 'xyce'])
    def test_nested_repeated_helper_readback(self, monkeypatch, engine):
        from types import SimpleNamespace
        from silk.simulation import backend as be
        paths = ['xbias1.xinner', 'xbias2.xinner',
                 'xdut.xbias1.xinner']
        names = ([f'@m.{p}.xmmn.xmdev.msky130[vgs]' for p in paths]
                 if engine == 'ngspice' else
                 [f'N({p.replace(".", ":")}:XMMN:XMDEV:MSKY130:VGS)' for p in paths])
        plot = SimpleNamespace(vectors=[
            self._vec(name, value) for name, value in zip(names, [.5, .7, .9])])
        monkeypatch.setattr(be, 'select_plot', lambda native, analysis: plot)
        backend = be.SimulatorBackend.get(engine)
        for path, expected in zip(paths, [.5, .7, .9]):
            values = backend.device_op_vectors(SimpleNamespace(native=plot), path, 'xmmn')
            assert values['vgs'] == expected

    def test_xyce_genuine_rooted_ambiguity_stays_unavailable(self, monkeypatch):
        from types import SimpleNamespace
        from silk.simulation import backend as be
        plot = SimpleNamespace(vectors=[
            self._vec('N(XBIAS1:XMMN:M1:VGS)', .5),
            self._vec('N(XBIAS1:XMMN:M2:VGS)', .7)])
        monkeypatch.setattr(be, 'select_plot', lambda native, analysis: plot)
        values = be.XyceBackend().device_op_vectors(
            SimpleNamespace(native=plot), 'xbias1', 'xmmn')
        assert math.isnan(values['vgs'])

    def test_different_root_is_not_ambiguous(self, monkeypatch):
        out = self._read(monkeypatch, [
            '@m.xdut.xi.xm1.msky130[vgs]',
            '@m.xhelper.xdut.xi.xm1.msky130[vgs]'])
        assert out['vgs'] == 1.0

    def test_absent_device_warns(self, monkeypatch):
        from silk.metrics.diagnostics import MetricWarning
        with pytest.warns(MetricWarning, match='xdut.xi.xm1') as rec:
            self._read(monkeypatch, ['@m.xdut.other.msky130[vgs]'])
        assert rec[0].message.code == 'op_device_not_found'

    def test_unique_match_is_silent(self, monkeypatch):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            out = self._read(monkeypatch, ['@m.xdut.xi.xm1.msky130[vgs]'])
        assert out['vgs'] == 1.0
