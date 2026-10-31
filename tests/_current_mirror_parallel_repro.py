"""Driver for TestReportedReproducer (tests/test_parallel_lambda_metrics.py).

Runs the exact reproducer from TASK-301.7's task description in its own
process, and compares it against a serial run in a second, separate module
instance -- both inside this one process, since a stray SECOND
``load_pdk('sky130')`` call from importing some OTHER example file is what
corrupts an already-built circuit's device-class references for pickling
(see the caller's module docstring); this script imports exactly one
example module, so that hazard cannot occur here.
"""
import sys

from silk.examples.current_mirror_example import build_current_mirror


def main():
    serial_mod = build_current_mirror()
    serial = serial_mod.run_all_tests(corner='tt', n_workers=1)

    parallel_mod = build_current_mirror()
    parallel = parallel_mod.run_all_tests(corner='tt', n_workers=2)

    expected_keys = {'current_mirror.match', 'current_mirror.zout'}
    assert set(serial) == expected_keys, serial
    assert set(parallel) == expected_keys, parallel

    for key in expected_keys:
        s, p = serial[key], parallel[key]
        assert not isinstance(s, dict), s
        assert not isinstance(p, dict), p
        for name, s_val in s.metrics.items():
            p_val = p.metrics[name]
            assert abs(p_val - s_val) <= 1e-9 * max(1.0, abs(s_val)), (
                key, name, s_val, p_val)
        assert p.results[0].passed == s.results[0].passed

    print('OK')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
