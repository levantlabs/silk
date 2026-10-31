import re
import numpy as np
import pytest

from silk.faults.types import Fault
from silk.faults.sugar import stuck_open, w_variation
from silk.faults.set import FaultSet, _SprinkleSpec


# --- Construction + add ---

def test_empty_faultset():
    fs = FaultSet()
    assert list(fs) == []


def test_faultset_constructor_accepts_iterable():
    fs = FaultSet([stuck_open('xc.a'), stuck_open('xc.b')])
    assert len(list(fs)) == 2


def test_faultset_add():
    fs = FaultSet()
    fs.add(stuck_open('xc.a'))
    fs.add(stuck_open('xc.b'))
    targets = [f.target for f in fs]
    assert targets == ['xc.a', 'xc.b']


# --- sprinkle stores lazy spec ---

def test_sprinkle_stores_spec_without_expanding():
    fs = FaultSet()
    fs.sprinkle(scope=r'xcomp\d+\.mn_tail', fault=stuck_open(), rate=1.0)
    # No explicit faults yet
    assert list(fs) == []
    # But the spec is stored
    assert len(fs._sprinkles) == 1
    assert isinstance(fs._sprinkles[0], _SprinkleSpec)
    assert fs._sprinkles[0].scope == r'xcomp\d+\.mn_tail'


# --- realize: synthetic module with mock elem_dict ---

class _MockModule:
    """Mimics silk Module enough for FaultSet.realize()."""
    def __init__(self, device_paths):
        self._paths = list(device_paths)

    def _silk_fault_all_device_paths(self):
        return list(self._paths)


def test_realize_explicit_only():
    fs = FaultSet([stuck_open('xcomp1.mn_in_p'),
                   w_variation('xcomp2.mn_tail', sigma=0.02)])
    mod = _MockModule(['xcomp1.mn_in_p', 'xcomp2.mn_tail', 'xcomp3.mn_tail'])
    rng = np.random.default_rng(42)
    realized = fs.realize(mod, rng)
    assert len(realized) == 2
    assert {f.target for f in realized} == {'xcomp1.mn_in_p', 'xcomp2.mn_tail'}


def test_realize_sprinkle_rate_one_applies_all_matches():
    fs = FaultSet()
    fs.sprinkle(r'xcomp\d+\.mn_tail', stuck_open(), rate=1.0)
    mod = _MockModule(['xcomp1.mn_tail', 'xcomp2.mn_tail',
                       'xcomp1.mn_in_p'])  # in_p shouldn't match
    rng = np.random.default_rng(42)
    realized = fs.realize(mod, rng)
    assert {f.target for f in realized} == {'xcomp1.mn_tail', 'xcomp2.mn_tail'}
    assert all(f.fault_type == 'stuck_open' for f in realized)


def test_realize_sprinkle_rate_zero_applies_none():
    fs = FaultSet()
    fs.sprinkle(r'xcomp\d+\.mn_tail', stuck_open(), rate=0.0)
    mod = _MockModule(['xcomp1.mn_tail', 'xcomp2.mn_tail'])
    rng = np.random.default_rng(42)
    realized = fs.realize(mod, rng)
    assert realized == []


def test_realize_sprinkle_is_seed_deterministic():
    fs = FaultSet()
    fs.sprinkle(r'xc\d+\.m\w+', stuck_open(), rate=0.5)
    mod = _MockModule([f'xc{i}.m{j}' for i in range(10) for j in range(5)])
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    a = fs.realize(mod, rng_a)
    b = fs.realize(mod, rng_b)
    assert [f.target for f in a] == [f.target for f in b]


def test_realize_sprinkle_different_seeds_differ():
    fs = FaultSet()
    fs.sprinkle(r'xc\d+\.m\w+', stuck_open(), rate=0.5)
    mod = _MockModule([f'xc{i}.m{j}' for i in range(20) for j in range(10)])
    a = fs.realize(mod, np.random.default_rng(1))
    b = fs.realize(mod, np.random.default_rng(2))
    assert [f.target for f in a] != [f.target for f in b]


def test_faults_targeting_returns_explicit_and_sprinkled():
    fs = FaultSet([stuck_open('xc1.mn_tail')])
    fs.sprinkle(r'xc1\.mn_in_p', w_variation(sigma=0.02), rate=1.0)
    mod = _MockModule(['xc1.mn_tail', 'xc1.mn_in_p'])
    rng = np.random.default_rng(42)
    fs.realize(mod, rng)
    assert len(fs.faults_targeting('xc1.mn_tail')) == 1
    assert len(fs.faults_targeting('xc1.mn_in_p')) == 1


# --- JSON roundtrip ---

def test_json_roundtrip_explicit_and_sprinkle():
    fs = FaultSet([stuck_open('xc.a', per_finger=False),
                   w_variation('xc.b', sigma=0.02)])
    fs.sprinkle(r'xc\d+\.mn_tail', stuck_open(), rate=0.005)
    d = fs.to_json()
    fs2 = FaultSet.from_json(d)
    assert [f.target for f in fs2] == ['xc.a', 'xc.b']
    assert len(fs2._sprinkles) == 1
    assert fs2._sprinkles[0].scope == r'xc\d+\.mn_tail'
    assert fs2._sprinkles[0].rate == 0.005


# --- per-sprinkle seed (TASK-605) ---

_PATHS = [f'xc{i}.m{j}' for i in range(6) for j in 'abcd']

def _seeded_selection(seed, rng_seed):
    fs = FaultSet()
    fs.sprinkle(r'xc\d+\.m\w+', stuck_open(), rate=0.5, seed=seed)
    faults = fs.realize(_MockModule(_PATHS), np.random.default_rng(rng_seed))
    return [f.target for f in faults]


def test_sprinkle_seed_ignores_caller_rng():
    assert _seeded_selection(42, 1) == _seeded_selection(42, 2)


def test_sprinkle_different_seeds_differ():
    assert any(_seeded_selection(a, 1) != _seeded_selection(b, 1)
               for a, b in [(1, 2), (3, 4), (5, 6)])


def test_sprinkle_unseeded_follows_caller_rng():
    def sel(r):
        fs = FaultSet()
        fs.sprinkle(r'xc\d+\.m\w+', stuck_open(), rate=0.5)
        return [f.target for f in fs.realize(_MockModule(_PATHS),
                                             np.random.default_rng(r))]
    assert sel(1) == sel(1)


def test_sprinkle_seed_serialized_unchanged():
    fs = FaultSet()
    fs.sprinkle(r'm\d+', stuck_open(), rate=0.5, seed=7)
    assert fs.to_json()['sprinkles'][0]['seed'] == 7
    assert FaultSet.from_json(fs.to_json())._sprinkles[0].seed == 7


@pytest.mark.parametrize('kind,param,value', [
    (kind, param, value)
    for kind, param in [('w_variation', 'sigma'), ('l_variation', 'sigma'),
                        ('vth_shift', 'sigma'), ('vth_shift', 'delta')]
    for value in [float('nan'), float('inf'), -float('inf'), -0.1]
    if param != 'delta' or value != -0.1
])
@pytest.mark.parametrize('route', ['helper', 'direct', 'json', 'mutated', 'template', 'json_template'])
def test_numeric_declarations_refused_before_draws(kind, param, value, route):
    import silk.faults as faults
    params = {'sigma': .02}
    if kind == 'vth_shift':
        params['delta'] = .02
    if route == 'mutated':
        fault = Fault('mn', kind, dict(params))
        fs = FaultSet([fault])
        fault.params[param] = value
    else:
        params[param] = value
        if route in ('template', 'json_template'):
            fs = FaultSet()
            fs.sprinkle('missing', getattr(faults, kind)(**params), rate=0)
            if route == 'json_template':
                import json
                fs = FaultSet.from_json(json.loads(json.dumps(fs.to_json())))
        elif route == 'helper':
            fs = FaultSet([getattr(faults, kind)('mn', **params)])
        else:
            fs = FaultSet([Fault('mn', kind, params)])
            if route == 'json':
                import json
                fs = FaultSet.from_json(json.loads(json.dumps(fs.to_json())))
    previous = [stuck_open('earlier')]
    fs._last_realized = previous
    # An earlier valid sprinkle must not consume draws before the bad declaration.
    fs._sprinkles.insert(0, _SprinkleSpec('.*', stuck_open(), .5))
    rng = np.random.default_rng(42)
    state = rng.bit_generator.state
    with pytest.raises(ValueError) as exc:
        fs.realize(_MockModule(['mn']), rng)
    message = str(exc.value)
    for fragment in (kind, param, repr(value), 'finite',
                     'missing' if route in ('template', 'json_template') else 'mn'):
        assert fragment in message
    if param == 'sigma':
        assert '>= 0' in message
    assert rng.bit_generator.state == state
    assert fs._last_realized is previous


@pytest.mark.parametrize('rate', [float('nan'), float('inf'), -float('inf'), -.1, 1.1])
@pytest.mark.parametrize('roundtrip', [False, True])
def test_invalid_sprinkle_rate_without_matches(rate, roundtrip):
    fs = FaultSet()
    fs.sprinkle('absent', stuck_open(), rate=rate)
    if roundtrip:
        fs = FaultSet.from_json(fs.to_json())
    with pytest.raises(ValueError) as exc:
        fs.realize(_MockModule([]), np.random.default_rng(1))
    for fragment in ('rate', repr(rate), '[0, 1]', 'absent', 'stuck_open'):
        assert fragment in str(exc.value)


@pytest.mark.parametrize('seed', [None, 7])
@pytest.mark.parametrize('rate', [0, .4, 1])
def test_valid_numeric_templates_keep_selection_draw_order(seed, rate):
    from silk.faults import l_variation, vth_shift
    fs = FaultSet([w_variation('mn', sigma=0), l_variation('mn', sigma=10000),
                   vth_shift('mn', delta=-.02), vth_shift('mn', delta=0),
                   vth_shift('mn', delta=.02, sigma=.01)])
    fs.sprinkle('.*', w_variation(sigma=.02), rate=rate, seed=seed)
    paths = ['m1', 'm2', 'm3']
    rng = np.random.default_rng(42)
    expected_rng = np.random.default_rng(42)
    source = np.random.default_rng(seed) if seed is not None else expected_rng
    expected = [p for p, draw in zip(paths, source.random(3)) if draw < rate]
    declared = fs.to_json()
    result = fs.realize(_MockModule(paths), rng)
    assert result[:5] == list(fs)
    assert [f.target for f in result[5:]] == expected
    assert rng.bit_generator.state == expected_rng.bit_generator.state
    assert fs.to_json() == declared


def test_mutated_template_params_refused_even_without_matches():
    template = w_variation(sigma=.02)
    fs = FaultSet()
    fs.sprinkle('absent', template, rate=0)
    template.params['sigma'] = float('nan')
    with pytest.raises(ValueError, match='sigma=nan'):
        fs.realize(_MockModule([]), np.random.default_rng(42))
