"""Monte Carlo on ngspice + sky130, against TASK-300's measured facts
(TASK-427).

The circuit is TASK-300's ``ng_offset.sp``: a diode-connected matched
``nmos1p8`` pair, W=5 um L=2 um, 20 uA each, read at the operating point.
``v(g1) - v(g2)`` is the input-referred offset of a diff pair with an ideal
matched load. TASK-300 measured its sigma over 200 samples at ``tt_mm`` as
2.065 mV (95 % CI 1.86-2.27 mV).
"""
import math
import re

import numpy as np
import pytest

from silk.circuit.monte_carlo import _SampleSpec, derive_sample_seed
from tests._env import needs_ngspice, needs_pdk
from tests.test_monte_carlo_mismatch_off import build_two_pairs

pytestmark = [pytest.mark.slow, needs_ngspice, needs_pdk]

TASK_300_SIGMA_V = 2.065e-3


def _offset(ckt):
    def node(name):
        return float(np.asarray(ckt.load_signal(name).y).ravel()[-1])
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
    mod.define_module_ports({'g1': PortDir.INOUT, 'g2': PortDir.INOUT,
                             'vss': PortDir.GROUND})
    mod.add(pdk.nmos1p8('m1', d='g1', g='g1', s='vss', b='vss', w=5.0, l=2.0))
    mod.add(pdk.nmos1p8('m2', d='g2', g='g2', s='vss', b='vss', w=5.0, l=2.0))
    tb = mod.create_test('offset')
    tb.add_module_under_test({'g1': 'g1', 'g2': 'g2', 'vss': '0'})
    tb.add(sources.idc('i1', positive='0', negative='g1', value=20e-6))
    tb.add(sources.idc('i2', positive='0', negative='g2', value=20e-6))
    tb.add_sim_type(analysis.Op())
    tb.add_metric('offset', _offset)
    tb.add_spec('offset small', 'offset', '<', 10e-3)
    return tb


def _rawfile_without_date(path):
    """A rawfile's bytes with its ``Date:`` header line removed: ngspice
    stamps the wall-clock time there, the only line two runs of one deck
    differ in (TASK-300 C1)."""
    with open(path, 'rb') as f:
        data = f.read()
    return re.sub(rb'^Date:[^\n]*\n', b'', data, count=1, flags=re.M)


def test_identical_devices_differ_under_mismatch_and_match_without(pair_tb):
    nominal = pair_tb.run()
    assert nominal.metrics['offset'] == 0.0
    mc = pair_tb.run_monte_carlo(2)
    offsets = [r.metrics['offset'] for r in mc.reports.values()]
    assert all(abs(o) > 1e-5 for o in offsets)
    assert offsets[0] != offsets[1]


def test_same_seed_gives_a_byte_identical_rawfile(pair_tb):
    first = pair_tb.run_monte_carlo(1, seed=11)
    path = first.reports[('tt', 0)].rawfile_path
    before = _rawfile_without_date(path)
    again = pair_tb.run_monte_carlo(1, seed=11)
    assert again.reports[('tt', 0)].rawfile_path == path
    assert _rawfile_without_date(path) == before
    assert (again.reports[('tt', 0)].metrics['offset']
            == first.reports[('tt', 0)].metrics['offset'])
    other = pair_tb.run_monte_carlo(1, seed=12)
    assert _rawfile_without_date(path) != before
    assert (other.reports[('tt', 0)].metrics['offset']
            != first.reports[('tt', 0)].metrics['offset'])


def test_rerun_reproduces_a_sample(pair_tb):
    mc = pair_tb.run_monte_carlo(2, seed=3)
    assert (mc.rerun(('tt', 1)).metrics['offset']
            == mc.reports[('tt', 1)].metrics['offset'])


def test_parallel_samples_match_serial(pair_tb):
    pair_tb.set_monte_carlo(4, seed=5)
    key = f'{pair_tb.module.module_name}.{pair_tb.test_name}'
    serial = pair_tb.module.run_all_tests(n_workers=1)[key]
    parallel = pair_tb.module.run_all_tests(n_workers=2)[key]
    assert ({k: r.metrics['offset'] for k, r in serial.reports.items()}
            == {k: r.metrics['offset'] for k, r in parallel.reports.items()})


def test_diff_pair_offset_sigma_matches_task_300(pair_tb):
    n = 50
    mc = pair_tb.run_monte_carlo(n, n_workers=4)
    assert not mc.failed and len(mc.reports) == n
    sigma = mc.stat('offset', 'std')
    # Approximate 95 % interval for a normal sigma from n samples: the
    # measured value must hold TASK-300's 2.065 mV, as TASK-300's own
    # interval (1.86-2.27 mV) holds this one. At n=50 the relative
    # half-width is 19.8%; the zero-mean bound below is 3 standard errors.
    half = 1.959963984540054 / math.sqrt(2 * (n - 1))
    lo, hi = sigma * (1 - half), sigma * (1 + half)
    assert lo <= TASK_300_SIGMA_V <= hi, (sigma, lo, hi)
    assert abs(mc.stat('offset', 'mean')) < 3 * sigma / math.sqrt(n)


# TASK-430: a block with mismatch off draws nothing while its twin draws.
def test_block_with_mismatch_off_gives_exactly_zero_offset(tmp_path):
    top = build_two_pairs(tmp_path)
    off, on = [], []
    for i in range(4):
        top._mc_sample = _SampleSpec(
            index=i, seed=derive_sample_seed(0, i), variation='mismatch',
            section='tt_mm', wrapper_path=None)
        top.simulate(corner='tt')
        v = lambda n: float(np.asarray(top.load_signal(n, 'V').y).ravel()[-1])
        off.append(v('a1') - v('a2'))
        on.append(v('b1') - v('b2'))
    assert all(x == 0.0 for x in off), off
    assert np.std(on) > 0.0, on


# TASK-429: the drawn parameters rank the offset as TASK-300 decomposed it
# (Xyce, N=400: vth0 57 %, toxe 26 %, voff 17 % of the variance, R^2 0.9998).
def test_record_draws_ranks_the_offset_as_task_300(pair_tb):
    import csv
    import os
    n = 50
    mc = pair_tb.run_monte_carlo(n, record_draws=True, n_workers=4)
    assert not mc.failed and len(mc.draws.values) == n
    assert {d for d, _ in mc.draws.columns} == {'dut.m1', 'dut.m2'}
    s = mc.sensitivity('offset')
    assert s.method == 'least_squares' and s.r_squared > 0.99
    shares = {}
    for row in s.rows:
        param = row.name.split(':')[1]
        shares[param] = shares.get(param, 0.0) + row.share
    assert shares['vth0'] > shares['toxe'] > shares['voff']
    # At n=50 the two-device normal-draw delta approximation gives
    # 95% share half-widths <= 0.090, including the n=400 reference.
    # Keep the original 0.10 bound; derivation is in TASK-546.
    for param, expected in (('vth0', 0.57), ('toxe', 0.26), ('voff', 0.17)):
        assert abs(shares[param] - expected) <= 0.10, shares
    assert shares['nfactor'] < 0.05, shares
    path = os.path.join(pair_tb.module.dir_folder, 'test_pair~offset', 'tt',
                        'draws.csv')
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n
    assert float(rows[0]['dut.m1:vth0']) == mc.draws.values[('tt', 0)][0]
