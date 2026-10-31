import pytest
from silk.faults.types import Fault, AppliedFault, FaultManifest


def test_fault_dataclass_is_frozen():
    f = Fault(target='xcomp1.mn_tail', fault_type='stuck_open')
    with pytest.raises(Exception):  # FrozenInstanceError
        f.target = 'changed'


def test_fault_fingerprint_is_stable():
    f1 = Fault(target='xc.mn', fault_type='w_variation', params={'sigma': 0.02})
    f2 = Fault(target='xc.mn', fault_type='w_variation', params={'sigma': 0.02})
    assert f1.fingerprint() == f2.fingerprint()


def test_fault_fingerprint_differs_when_params_differ():
    a = Fault(target='xc.mn', fault_type='w_variation', params={'sigma': 0.01})
    b = Fault(target='xc.mn', fault_type='w_variation', params={'sigma': 0.02})
    assert a.fingerprint() != b.fingerprint()


def test_fault_fingerprint_key_order_independent():
    a = Fault(target='x', fault_type='vth_shift', params={'delta': 0.03, 'sigma': 0.005})
    b = Fault(target='x', fault_type='vth_shift', params={'sigma': 0.005, 'delta': 0.03})
    assert a.fingerprint() == b.fingerprint()


def test_fault_manifest_json_roundtrip():
    m = FaultManifest(
        seed=42,
        input_spec={'faults': [], 'sprinkles': []},
        applied=[
            AppliedFault(
                target='xcomp1.mn_tail',
                fault_type='stuck_open',
                declared_params={},
                realized_params={'m_before': 4, 'm_after': 3, 'deleted': False},
                netlist_comment='*** FAULT: stuck_open on xcomp1.mn_tail (m 4 → 3)',
            ),
        ],
    )
    j = m.to_json()
    round_tripped = FaultManifest.from_json(j)
    assert round_tripped.seed == 42
    assert len(round_tripped.applied) == 1
    assert round_tripped.applied[0].target == 'xcomp1.mn_tail'
    assert round_tripped.applied[0].realized_params['m_after'] == 3
