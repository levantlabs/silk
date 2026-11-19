"""Monte Carlo on Xyce + sky130 (TASK-428), against TASK-300's measured facts.

The circuit is TASK-300's ``xy_offset.cir``: a diode-connected matched
``nmos1p8`` pair, W=5 um L=2 um, 20 uA each, read through a one-point DC
sweep (Xyce has no ``.op`` form). TASK-300 measured the offset sigma over 200
samples at ``tt_mm`` as 2.180 mV (95 % CI 1.97-2.39 mV).
"""
import math
import os

import numpy as np
import pytest

from tests._env import HAVE_XYCE, needs_pdk

pytestmark = [
    pytest.mark.slow, pytest.mark.requires_xyce, needs_pdk,
    pytest.mark.skipif(not HAVE_XYCE, reason='needs configured Xyce'),
]

TASK_300_INTERVAL_V = (1.97e-3, 2.39e-3)   # sigma 2.180 mV, 95 % CI


def _offset(ckt):
    def node(name):
        return float(np.asarray(ckt.load_signal(name, 'V').y).ravel()[-1])
    return node('g1') - node('g2')


@pytest.fixture
def pair_tb(tmp_path):
    from silk import load_pdk, sources
    from silk.circuit.circuit_elements import PortDir
    from silk.circuit.modules import Module
    from silk.simulation import analysis
    pdk = load_pdk('sky130')
    mod = Module('PDK_skywater130', str(tmp_path), 'pair.cir',
                 pdk_details=pdk, module_name='pair')
    mod.simulator = 'xyce'
    mod.define_module_ports({'g1': PortDir.INOUT, 'g2': PortDir.INOUT,
                             'vss': PortDir.GROUND})
    mod.add(pdk.nmos1p8('m1', d='g1', g='g1', s='vss', b='vss', w=5.0, l=2.0))
    mod.add(pdk.nmos1p8('m2', d='g2', g='g2', s='vss', b='vss', w=5.0, l=2.0))
    tb = mod.create_test('offset')
    tb.add_module_under_test({'g1': 'g1', 'g2': 'g2', 'vss': '0'})
    tb.add(sources.idc('i1', positive='0', negative='g1', value=20e-6))
    tb.add(sources.idc('i2', positive='0', negative='g2', value=20e-6))
    tb.add_sim_type(analysis.DC('i1', 20e-6, 20e-6, 1e-6))
    tb.add_metric('offset', _offset)
    tb.add_spec('offset small', 'offset', '<', 10e-3)
    return tb


def test_identical_devices_differ_and_the_seed_reproduces(pair_tb):
    mc = pair_tb.run_monte_carlo(3, seed=7)
    assert not mc.failed
    offsets = [r.metrics['offset'] for r in mc.reports.values()]
    assert all(abs(o) > 1e-5 for o in offsets) and len(set(offsets)) == 3
    again = pair_tb.run_monte_carlo(3, seed=7)
    assert ([r.metrics['offset'] for r in again.reports.values()] == offsets)
    assert (mc.rerun(('tt', 1)).metrics['offset'] == offsets[1])


def test_offset_sigma_and_ranking_match_task_300(pair_tb):
    from silk.circuit.monte_carlo_draws import DrawTable, rank_sensitivity
    n = 50
    mc = pair_tb.run_monte_carlo(n, record_draws=True, n_workers=4)
    assert not mc.failed and len(mc.reports) == n
    sigma = mc.stat('offset', 'std')
    half = 1.959963984540054 / math.sqrt(2 * (n - 1))
    lo, hi = sigma * (1 - half), sigma * (1 + half)
    # At n=50 the relative half-width is 19.8%; the historical reference
    # interval stays fixed. The two approximate 95 % intervals must overlap.
    assert lo <= TASK_300_INTERVAL_V[1] and hi >= TASK_300_INTERVAL_V[0], (
        sigma, lo, hi)

    keys = list(mc.reports)
    first = mc.reports[keys[0]]._mc_draws
    columns = tuple(sorted(first))
    assert {d for d, _ in columns} == {'dut.m1', 'dut.m2'}
    table = DrawTable(
        columns=columns, raw=False,
        values={k: [mc.reports[k]._mc_draws[c] for c in columns] for k in keys})
    ranking = rank_sensitivity(
        'offset', {k: mc.reports[k].metrics['offset'] for k in keys}, table)
    assert ranking.rows[0].name.endswith(':vth0'), ranking.rows[:3]
    # TASK-429 integration: the run's own table and ranking carry the same.
    assert mc.draws.raw is False and len(mc.draws.values) == n
    assert set(mc.draws.columns) == set(columns)
    assert mc.sensitivity('offset').rows[0].name.endswith(':vth0')
    import os
    assert os.path.isfile(os.path.join(pair_tb.module.dir_folder,
                                       f'test_{pair_tb.module.module_name}~'
                                       f'{pair_tb.test_name}', 'tt',
                                       'draws.csv'))


@pytest.mark.parametrize('workers', [1, 2])
def test_new_campaign_preserves_corner_outputs(pair_tb, workers):
    from pathlib import Path
    tb = pair_tb
    tb.set_corners(['tt', 'ss'])
    baseline = tb.run_monte_carlo(2, seed=7, record_draws=True, n_workers=workers)
    first = tb.run_monte_carlo(2, seed=7, record_draws=True,
                               n_workers=workers, output_mode='new')
    roots = {corner: Path(first[(corner, 0)].run.deck_path).parent.parent
             for corner in ('tt', 'ss')}
    saved = {p: p.read_bytes() for root in roots.values()
             for p in root.rglob('*') if p.is_file()
             and not any(part.startswith('.silk') for part in p.relative_to(root).parts)}
    second = tb.run_monte_carlo(2, seed=7, record_draws=True,
                                n_workers=workers, output_mode='new')
    assert not baseline.failed and not first.failed and not second.failed
    assert {k: r.metrics for k, r in baseline.items()} == {k: r.metrics for k, r in first.items()}
    assert {k: r.metrics for k, r in second.items()} == {k: r.metrics for k, r in first.items()}
    assert baseline.draws.columns == first.draws.columns == second.draws.columns
    assert baseline.draws.raw == first.draws.raw == second.draws.raw
    for key in baseline.draws.values:
        np.testing.assert_array_equal(baseline.draws.values[key], first.draws.values[key])
        np.testing.assert_array_equal(baseline.draws.values[key], second.draws.values[key])
    for corner, root in roots.items():
        assert root.name.startswith('run-') and root.parent.name == corner
        assert (root / 'draws.csv').is_file()
        assert Path(first[(corner, 1)].run.deck_path).parent.parent == root
        assert Path(second[(corner, 0)].run.deck_path).parent.parent != root
    assert all(p.read_bytes() == data for p, data in saved.items())


def test_prepare_external_sample_uses_derived_seed(pair_tb, tmp_path):
    import subprocess
    from pathlib import Path
    from silk import settings
    from silk.circuit.monte_carlo import derive_sample_seed
    tb = pair_tb.set_monte_carlo(3, seed=11)
    expected = tb.run()[('tt', 2)].metrics['offset']
    deck = Path(tb.prepare(corner='tt', sample=2))
    remote = tmp_path / 'remote'
    remote.mkdir()
    # Xyce writes its RAW output to the path in the deck's .PRINT card.
    raw = remote / 'returned.raw'
    text = deck.read_text().replace(str(tb.netlist._run_files.rawfile_path), str(raw))
    copied = remote / 'deck.cir'
    copied.write_text(text)
    completed = subprocess.run(
        [settings.get('SILK_XYCE_BIN'), '-randseed',
         str(derive_sample_seed(tb.monte_carlo.seed, 2)), str(copied)],
        cwd=remote, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = tb.evaluate('tt', rawfile=raw)
    assert report.metrics['offset'] == pytest.approx(expected, abs=1e-12)
    assert report.provenance.run_origin == 'external-rawfile'
    assert report.provenance.seed is None


@pytest.mark.parametrize('variation', ['mismatch', 'process', 'both'])
def test_resistor_draws_match_exact_sample_cells_and_preserve_metrics(tmp_path, variation):
    from pathlib import Path
    from tests.test_monte_carlo_ngspice import build_resistor_bench
    tb = build_resistor_bench(tmp_path, 'xyce')
    off = tb.run_monte_carlo(1, seed=9, variation=variation, output_mode='new')
    on = tb.run_monte_carlo(2, seed=9, variation=variation, record_draws=True, output_mode='new')
    again = (tb.run_monte_carlo(1, seed=9, variation=variation, record_draws=True, output_mode='new')
             if variation == 'mismatch' else on)
    assert not on.failed
    assert on.draws.raw is False
    assert set(on.draws.columns) == {(name, label) for name in ('dut.r1', 'dut.r[0]')
                                    for label in ('rbody.rsh', 'rhead.rsh')}
    for key, report in on.reports.items():
        if key in off.reports:
            assert report.metrics == off.reports[key].metrics
            np.testing.assert_array_equal(
                report.sim_results.load_signal('i(vbias)', 'I').y,
                off.reports[key].sim_results.load_signal('i(vbias)', 'I').y)
        if key in again.reports:
            assert report.metrics == again.reports[key].metrics
            np.testing.assert_array_equal(on.draws.values[key], again.draws.values[key])
        files = list(Path(report.rawfile_path).parent.glob('*.res'))
        assert len(files) == 1
        header, row = [line.split() for line in files[0].read_text().splitlines() if line.strip()][:2]
        cells = {c.upper(): float(v) for c, v in zip(header[1:], row[1:])}
        for (device, label), value in report._mc_draws.items():
            leaf = 'R1' if device == 'dut.r1' else 'R<0>'
            target, param = label.split('.')
            assert value == cells[f'XDUT:XM{leaf}:{target.upper()}_MODEL:{param.upper()}']
    assert on.seeds[('tt', 0)] != on.seeds[('tt', 1)]
    assert np.any(on.draws.values[('tt', 0)] != on.draws.values[('tt', 1)])
    from silk.circuit import spec_report as codec
    restored = codec.results_from_dict(codec.results_to_dict(on))
    assert restored.draws.columns == on.draws.columns
    assert restored.draws.raw is on.draws.raw
    for key in on.draws.values:
        np.testing.assert_array_equal(restored.draws.values[key], on.draws.values[key])
    with pytest.warns(UserWarning):
        fitted = restored.sensitivity('current', by='device')
    assert fitted.method == 'correlation'
    assert {row.name for row in fitted.rows} == {'dut.r1', 'dut.r[0]'}



def test_mixed_resistor_mos_worker_draws_preserve_keys_and_values(tmp_path):
    from tests.test_monte_carlo_ngspice import build_resistor_bench
    serial = build_resistor_bench(tmp_path / 'serial', 'xyce', mixed=True).run_monte_carlo(
        2, seed=13, record_draws=True)
    workers = build_resistor_bench(tmp_path / 'workers', 'xyce', mixed=True).run_monte_carlo(
        2, seed=13, record_draws=True, n_workers=2)
    assert not serial.failed and not workers.failed
    assert serial.draws.columns == workers.draws.columns
    assert serial.seeds == workers.seeds
    assert set(serial.draws.columns) >= {('dut.m', 'vth0'), ('dut.r1', 'rbody.rsh'),
                                        ('dut.r[0]', 'rhead.rsh')}
    for key in serial.reports:
        assert serial.reports[key].metrics == workers.reports[key].metrics
        np.testing.assert_array_equal(serial.draws.values[key], workers.draws.values[key])
