import math
import numpy as np
import pytest
import silk
from silk.circuit.variable import Variable
from silk.exceptions import ConfigurationError, VariationError
from silk.simulation.variation import (Normal, Uniform, draw_u, u_from_word, mismatch_key,
                            process_key, bench_key, owner_segments, is_declaration)


def test_exports():
    assert silk.Normal is Normal and silk.Uniform is Uniform
    assert issubclass(VariationError, ConfigurationError)
    assert issubclass(VariationError, ValueError)


def test_nominal_values():
    assert Normal(nominal=0.25, sigma=1e-3).resolve().nominal == 0.25
    u = Uniform(low=0.1, high=0.3).resolve()
    assert u.nominal == pytest.approx(0.2)
    assert Uniform(low=0.5, high=0.5).resolve().has_spread is False
    assert Normal(nominal=1.0, sigma=0.0).resolve().has_spread is False


@pytest.mark.parametrize('make', [
    lambda: Normal(nominal=0.0, sigma=-1e-3),
    lambda: Normal(nominal=math.inf, sigma=1e-3),
    lambda: Normal(nominal=0.0, sigma=math.nan),
    lambda: Uniform(low=2.0, high=1.0),
    lambda: Uniform(low=-1e308, high=1e308),          # high - low overflows
    lambda: Normal(nominal=0.0, sigma=1e-3, variation='Mismatch'),
    lambda: Normal(nominal=0.0, sigma=1e-3, variation='process'),            # no name
    lambda: Normal(nominal=0.0, sigma=1e-3, variation='process', name=''),
    lambda: Normal(nominal=0.0, sigma=1e-3, name='gain'),                    # name on mismatch
])
def test_construction_refusals(make):
    with pytest.raises(VariationError):
        make()


@pytest.mark.parametrize('bad', ['1m', True, None, [1.0], object()])
def test_field_types(bad):
    with pytest.raises(TypeError):
        Normal(nominal=bad, sigma=1e-3)


def test_numpy_scalars_are_numbers():
    r = Normal(nominal=np.float32(0.0), sigma=np.float64(1e-3)).resolve()
    assert r.fields == (0.0, pytest.approx(1e-3))
    assert all(type(f) is float for f in r.fields)


def test_variable_fields_are_read_at_resolution():
    s = Variable('s', value=1e-3, bounds=(-1.0, 1.0))
    decl = Normal(nominal=0.0, sigma=s)
    assert decl.resolve().fields == (0.0, 1e-3)
    s.value = 2e-3
    assert decl.resolve().fields == (0.0, 2e-3)
    s.value = -1e-3
    with pytest.raises(VariationError, match='sigma'):
        decl.resolve()
    assert Normal(nominal=0.0, sigma=s * 0 + 4e-3).resolve().fields[1] == pytest.approx(4e-3)


def test_variable_without_value_raises_at_resolution():
    s = Variable('s', bounds=(0.0, 1.0))
    with pytest.raises(VariationError, match='no value'):
        Normal(nominal=0.0, sigma=s).resolve()


def test_declarations_are_frozen_and_not_compared_by_value():
    a, b = Normal(nominal=0.0, sigma=1e-3), Normal(nominal=0.0, sigma=1e-3)
    with pytest.raises(Exception):
        a.sigma = 2e-3
    assert a != b          # eq=False: silk never relies on declaration equality
    assert is_declaration(a) and is_declaration(Uniform(low=0, high=1))
    assert not is_declaration(1.0)


# Draw scheme 1 goldens (spec §4.2), computed 2026-10-07 on Python 3.12.
GOLDEN = [
    (["silk.variation", 1, 0, 0, "mismatch", ["stage1", "comp3"], "offset_voltage"],
     '7d22f0ad538750e6bac02b82a5fad8ec54230ba421a1c3d491c2a3704093d18f',
     0.48881439429854134, -0.02804183017531213),
    (["silk.variation", 1, 7, 3, "process", "comp_gain"],
     '9016c0c4c32615811b49e1b6a52bff9f42687f48b4c9f0431d210e2ff7d4c096',
     0.5628471832634648, 0.15819184007326814),
    (["silk.variation", 1, 0, 0, "bench", "$", "offset", ["xstim"], "amp"],
     '7eff7c5acb6948f26e9e3d45a050d6bbed9f46d04d166eccceb46cbbd1337fe2',
     0.49608590332664904, -0.009811342799433015),
]


@pytest.mark.parametrize('key, digest, u, z', GOLDEN)
def test_draw_scheme_goldens(key, digest, u, z):
    import hashlib, json
    from silk._json_codec import encode_json
    encoded = json.dumps(encode_json(key), ensure_ascii=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    expected = [
        b'["silk.variation",1,0,0,"mismatch",["stage1","comp3"],"offset_voltage"]',
        b'["silk.variation",1,7,3,"process","comp_gain"]',
        b'["silk.variation",1,0,0,"bench","$","offset",["xstim"],"amp"]',
    ][[entry[0] for entry in GOLDEN].index(key)]
    assert encoded == expected
    assert hashlib.sha256(encoded).hexdigest() == digest
    assert draw_u(key) == u                               # exact
    r = Normal(nominal=0.0, sigma=1.0).resolve()
    assert r.at(u) == pytest.approx(z, rel=1e-12, abs=1e-15)   # inv_cdf: tolerance


def test_key_builders_match_goldens():
    assert mismatch_key(0, 0, ('stage1', 'comp3'), 'offset_voltage') == GOLDEN[0][0]
    assert process_key(7, 3, 'comp_gain') == GOLDEN[1][0]
    assert bench_key(0, 0, '$', 'offset', ('xstim',), 'amp') == GOLDEN[2][0]


def test_u_endpoints_are_inside_the_open_interval():
    lo, hi = u_from_word(0), u_from_word(2 ** 64 - 1)
    assert 0.0 < lo < hi < 1.0
    r = Normal(nominal=0.0, sigma=1.0).resolve()
    assert r.at(lo) == pytest.approx(-8.209536151601386)
    assert r.at(hi) == pytest.approx(8.209536151601386)


def test_uniform_stays_within_limits():
    r = Uniform(low=-1.0, high=3.0).resolve()
    for word in (0, 2 ** 63, 2 ** 64 - 1):
        assert -1.0 <= r.at(u_from_word(word)) <= 3.0


def test_simulator_seed_collision_does_not_repeat_a_chip():
    from silk.circuit.monte_carlo import derive_sample_seed
    assert derive_sample_seed(0, 50699) == derive_sample_seed(0, 55170)
    assert (draw_u(mismatch_key(0, 50699, ('c0',), 'offset'))
            != draw_u(mismatch_key(0, 55170, ('c0',), 'offset')))


def test_distribution_sanity():
    zs = np.array([Normal(nominal=0.0, sigma=1.0).resolve().at(
        draw_u(mismatch_key(0, i, ('c0',), 'offset'))) for i in range(2000)])
    assert abs(zs.mean()) < 0.1 and abs(zs.std() - 1.0) < 0.1
    us = np.array([draw_u(mismatch_key(0, i, ('c0',), 'offset')) for i in range(2000)])
    assert abs(us.mean() - 0.5) < 0.03


@pytest.mark.parametrize('path, segments', [
    ('$', ()), ('$/stage1', ('stage1',)), ('$/stage1/c0', ('stage1', 'c0')),
    ('$/a~1b/c~0d', ('a/b', 'c~d')),
])
def test_owner_segments(path, segments):
    assert owner_segments(path) == segments


@pytest.mark.parametrize('make', [
    lambda: Normal(0, 1, variation=None),
    lambda: Normal(0, 1, variation='process', name=7),
    lambda: Uniform(0, 1, name='offset'),
    lambda: Normal(10**400, 1),
])
def test_additional_declaration_errors(make):
    with pytest.raises(VariationError):
        make()


@pytest.mark.parametrize('bad', [np.bool_(True), complex(1, 0), '1m'])
def test_both_uniform_fields_require_real_numbers(bad):
    with pytest.raises(TypeError):
        Uniform(0, bad)


def test_variable_expression_errors_are_configuration_errors():
    s = Variable('unset', bounds=(0.0, 1.0))
    with pytest.raises(VariationError, match='sigma.*no value'):
        Normal(0, s * 2).resolve()
    s.value = 0.0
    with pytest.raises(VariationError, match='sigma'):
        Normal(0, 1 / s).resolve()


def test_variable_categorical_field_is_checked_at_resolution():
    v = Variable('choice', value='wide', var_type='categorical', categories=['wide'])
    with pytest.raises(VariationError, match='not a number'):
        Normal(0, v).resolve()


def test_draw_overflow_is_a_declaration_error():
    with pytest.raises(VariationError, match='finite'):
        Normal(1e308, 1e308).resolve().at(u_from_word(2**64 - 1))


def test_keyword_only_metadata_and_pickling():
    import pickle
    for cls in (Normal, Uniform):
        with pytest.raises(TypeError):
            cls(0, 1, 'process', 'shared')
        decl = cls(0, 1, variation='process', name='shared')
        assert pickle.loads(pickle.dumps(decl)).resolve() == decl.resolve()
