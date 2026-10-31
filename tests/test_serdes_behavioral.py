"""The SerDes twin produces ordinary behavioral evidence and geometric eyes."""
from pathlib import Path

import pytest

from silk.examples import serdes_tx_example as tx
from silk.sources import prbs_bits


@pytest.mark.parametrize('variant, height', [('cmos', 122.7), ('cml', 109.2)])
def test_tx_reports_have_raw_evidence_and_preserve_bit_recovery(tmp_path, monkeypatch,
                                                              variant, height):
    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    bits = prbs_bits(9, 511, seed=0)
    mod = tx.build_serdes_tx_behavioral(variant=variant, bits=bits)
    reports = mod.run_all_tests()
    for report in reports.values():
        assert report.operating_conditions.backend == 'behavioral'
        assert Path(report.rawfile_path).is_file()
        assert report.passed
    full = reports[f'tx_beh_{variant}.full']
    assert full.metrics['bit_errors'] == 0
    assert full.metrics['eye_h_mv'] == pytest.approx(height, abs=0.1)
    assert full.metrics['eye_w_ps'] == pytest.approx(1097, abs=1)
    off = reports[f'tx_beh_{variant}.off']
    assert off.metrics['bit_errors'] == 79
    assert reports[f'tx_beh_{variant}.ideal_off'].metrics['eye_w_ps'] == pytest.approx(1667, abs=1)
    assert reports[f'tx_beh_{variant}.ideal_full'].metrics['bit_errors'] > 0
