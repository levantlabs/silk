"""Tests for build_equivalence_classes and compute_canonical_inputs."""
import pytest
from silk.layout.matching import build_equivalence_classes, compute_canonical_inputs


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

class _FakeDev:
    def __init__(self, name, nets_dict, pin_names=('d', 'g', 's', 'b')):
        self.name = name
        self.nets = nets_dict
        self._pin_names = pin_names
        self.params = {'W': 2e-6, 'L': 180e-9, 'M': 1}


class _FakeNet:
    def __init__(self, name, classification=None, fanout=2, peak_current=None):
        self.name = name
        self.classification = classification
        self.fanout = fanout
        self.peak_current = peak_current
        self.estimated_length = None
        self.estimated_width = None
        self.estimated_r = None
        self.estimated_c = None


def _make_module(devs, nets):
    """devs: dict name->_FakeDev, nets: dict name->_FakeNet."""
    from types import SimpleNamespace
    m = SimpleNamespace()
    m.elem_dict = {
        name: {'device': dev, 'type': 'DEVICE', 'typeName': ''}
        for name, dev in devs.items()
    }
    m.net_dict = nets
    m.half_circuits = {}
    m.bounding_box = (10.0, 10.0)
    return m


# ---------------------------------------------------------------------------
# build_equivalence_classes — total coverage of net_dict
# ---------------------------------------------------------------------------

class TestBuildEquivalenceClassesTotal:

    def test_no_half_circuits_all_singletons(self):
        nets = {
            'vout': _FakeNet('vout'),
            'vin':  _FakeNet('vin'),
            'vdd':  _FakeNet('vdd'),
        }
        mod = _make_module({}, nets)
        classes = build_equivalence_classes(mod, [])
        # every net appears exactly once
        all_nets = [n for members in classes.values() for n in members]
        assert sorted(all_nets) == sorted(nets.keys())

    def test_no_half_circuits_all_singleton_size(self):
        nets = {'a': _FakeNet('a'), 'b': _FakeNet('b')}
        mod = _make_module({}, nets)
        classes = build_equivalence_classes(mod, [])
        for members in classes.values():
            assert len(members) == 1

    def test_every_net_in_exactly_one_class(self):
        nets = {'n1': _FakeNet('n1'), 'n2': _FakeNet('n2'), 'n3': _FakeNet('n3')}
        mod = _make_module({}, nets)
        classes = build_equivalence_classes(mod, [])
        all_flat = [n for ms in classes.values() for n in ms]
        assert len(all_flat) == len(nets)
        assert len(set(all_flat)) == len(nets)


class TestBuildEquivalenceClassesPairs:

    def test_diff_pair_drains_unified(self):
        """mn1.d='vout_p', mn2.d='vout_n' → {vout_p, vout_n} in one class."""
        from silk.layout.matching import HalfCircuit
        mn1 = _FakeDev('mn1', {'d': 'vout_p', 'g': 'vin_p', 's': 'tail', 'b': 'vss'})
        mn2 = _FakeDev('mn2', {'d': 'vout_n', 'g': 'vin_n', 's': 'tail', 'b': 'vss'})
        nets = {
            'vout_p': _FakeNet('vout_p'),
            'vout_n': _FakeNet('vout_n'),
            'vin_p':  _FakeNet('vin_p'),
            'vin_n':  _FakeNet('vin_n'),
            'tail':   _FakeNet('tail'),
            'vss':    _FakeNet('vss'),
        }
        mod = _make_module({'mn1': mn1, 'mn2': mn2}, nets)
        mod.half_circuits['diff_in'] = HalfCircuit(
            name='diff_in', left=('mn1',), right=('mn2',)
        )
        warnings = []
        classes = build_equivalence_classes(mod, warnings)

        # vout_p and vout_n should be in the same class
        flat = {name: cid for cid, members in classes.items() for name in members}
        assert flat['vout_p'] == flat['vout_n']
        # vin_p and vin_n also unified (same port 'g' connected)
        assert flat['vin_p'] == flat['vin_n']
        # shared nets stay as singletons (or merged with themselves only)
        # tail connects both devices' 's' → same net on both sides → no union
        assert len([m for m in classes[flat['tail']]] ) >= 1

    def test_shared_net_no_union(self):
        """tail is the same net on both sides: no union needed, stays singleton."""
        from silk.layout.matching import HalfCircuit
        mn1 = _FakeDev('mn1', {'d': 'vout_p', 'g': 'vin_p', 's': 'tail', 'b': 'vss'})
        mn2 = _FakeDev('mn2', {'d': 'vout_n', 'g': 'vin_n', 's': 'tail', 'b': 'vss'})
        nets = {
            'vout_p': _FakeNet('vout_p'), 'vout_n': _FakeNet('vout_n'),
            'vin_p': _FakeNet('vin_p'), 'vin_n': _FakeNet('vin_n'),
            'tail': _FakeNet('tail'), 'vss': _FakeNet('vss'),
        }
        mod = _make_module({'mn1': mn1, 'mn2': mn2}, nets)
        mod.half_circuits['p'] = HalfCircuit('p', left=('mn1',), right=('mn2',))
        warnings = []
        classes = build_equivalence_classes(mod, warnings)
        flat = {name: cid for cid, members in classes.items() for name in members}
        # tail is its own class
        assert len(classes[flat['tail']]) == 1

    def test_large_class_emits_warning(self):
        """Two half-circuits sharing a device cause a >2 class and a warning."""
        from silk.layout.matching import HalfCircuit
        mn1 = _FakeDev('mn1', {'d': 'va', 'g': 'vin', 's': 'vss', 'b': 'vss'})
        mn2 = _FakeDev('mn2', {'d': 'vb', 'g': 'vin', 's': 'vss', 'b': 'vss'})
        mn3 = _FakeDev('mn3', {'d': 'vc', 'g': 'vin', 's': 'vss', 'b': 'vss'})
        nets = {
            'va': _FakeNet('va'), 'vb': _FakeNet('vb'), 'vc': _FakeNet('vc'),
            'vin': _FakeNet('vin'), 'vss': _FakeNet('vss'),
        }
        mod = _make_module({'mn1': mn1, 'mn2': mn2, 'mn3': mn3}, nets)
        mod.half_circuits['p1'] = HalfCircuit('p1', left=('mn1',), right=('mn2',))
        mod.half_circuits['p2'] = HalfCircuit('p2', left=('mn1',), right=('mn3',))
        warnings = []
        classes = build_equivalence_classes(mod, warnings)
        reasons = [w.reason for w in warnings]
        assert any('large_class' in r for r in reasons)


# ---------------------------------------------------------------------------
# compute_canonical_inputs
# ---------------------------------------------------------------------------

class TestComputeCanonicalInputs:

    def test_max_current_selected(self):
        """The higher peak_current across matched members is chosen."""
        nets = [
            _FakeNet('vout_p', peak_current=2.1e-3, fanout=2),
            _FakeNet('vout_n', peak_current=1.8e-3, fanout=2),
        ]
        mod = _make_module({}, {n.name: n for n in nets})
        canon = compute_canonical_inputs(nets, mod, [])
        assert canon.peak_current == pytest.approx(2.1e-3)

    def test_max_fanout_selected(self):
        nets = [
            _FakeNet('a', fanout=3),
            _FakeNet('b', fanout=7),
        ]
        mod = _make_module({}, {n.name: n for n in nets})
        mod.bounding_box = (10.0, 10.0)
        canon = compute_canonical_inputs(nets, mod, [])
        assert canon.fanout == 7

    def test_none_current_falls_back(self):
        """All-None currents → canonical_peak_current is None."""
        nets = [_FakeNet('a', peak_current=None), _FakeNet('b', peak_current=None)]
        mod = _make_module({}, {n.name: n for n in nets})
        canon = compute_canonical_inputs(nets, mod, [])
        assert canon.peak_current is None

    def test_mixed_none_uses_non_none(self):
        """One None, one value → use the non-None value."""
        nets = [_FakeNet('a', peak_current=None), _FakeNet('b', peak_current=1e-3)]
        mod = _make_module({}, {n.name: n for n in nets})
        canon = compute_canonical_inputs(nets, mod, [])
        assert canon.peak_current == pytest.approx(1e-3)

    def test_no_current_emits_warning(self):
        """All-None currents → one warning per net with reason 'no_current_data'."""
        nets = [_FakeNet('a', peak_current=None), _FakeNet('b', peak_current=None)]
        mod = _make_module({}, {n.name: n for n in nets})
        warnings = []
        compute_canonical_inputs(nets, mod, warnings)
        assert len(warnings) == len(nets)
        assert all(w.reason == 'no_current_data' for w in warnings)
