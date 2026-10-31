"""
Tests for VariableSpace — joint multi-variable parameter space.
"""

import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from silk.circuit.variable import Variable
from silk.optimize.parameter_space import VariableSpace


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cont_vars():
    """Two continuous bounded variables."""
    w = Variable('w', bounds=(0.5e-6, 20e-6))
    l = Variable('l', bounds=(0.15e-6, 2e-6))
    return w, l


@pytest.fixture
def mixed_space():
    """Continuous + integer + categorical."""
    w    = Variable('w',    bounds=(0.5e-6, 20e-6))
    nfin = Variable('nfin', bounds=(1, 32), var_type='integer')
    cor  = Variable('corner', var_type='categorical',
                          categories=['tt', 'ss', 'ff'])
    return VariableSpace([w, nfin, cor])


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestVariableSpaceConstructor:

    def test_from_list_of_variables(self, cont_vars):
        w, l = cont_vars
        ps = VariableSpace([w, l])
        assert ps.names == ['w', 'l']

    def test_from_dict_of_variables(self, cont_vars):
        w, l = cont_vars
        ps = VariableSpace({'w': w, 'l': l})
        assert ps.size == 2

    def test_list_with_duplicate_names_raises(self, cont_vars):
        w, _ = cont_vars
        w2 = Variable('w', bounds=(1.0, 5.0))
        with pytest.raises(ValueError, match='Duplicate'):
            VariableSpace([w, w2])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match='at least one'):
            VariableSpace([])

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match='at least one'):
            VariableSpace({})

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            VariableSpace("w")

    def test_names_preserves_order(self):
        names = ['e', 'a', 'c', 'b', 'd']
        vars_ = [Variable(n, bounds=(0.0, 1.0)) for n in names]
        ps = VariableSpace(vars_)
        assert ps.names == names

    def test_size_property(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        assert ps.size == 2
        assert len(ps) == 2

    def test_repr_contains_names(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        r = repr(ps)
        assert 'w' in r and 'l' in r


# ---------------------------------------------------------------------------
# Random sampling
# ---------------------------------------------------------------------------

class TestSampleRandom:

    def test_returns_n_dicts(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        pts = ps.sample(10)
        assert len(pts) == 10
        assert all(isinstance(p, dict) for p in pts)

    def test_dict_keys_match_names(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        for p in ps.sample(5):
            assert set(p.keys()) == set(ps.names)

    def test_values_within_bounds(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        w_var, l_var = cont_vars
        for p in ps.sample(50):
            assert 0.5e-6 <= p['w'] <= 20e-6
            assert 0.15e-6 <= p['l'] <= 2e-6

    def test_integer_var_returns_ints(self):
        ps = VariableSpace([Variable('n', bounds=(1, 20),
                                            var_type='integer')])
        for p in ps.sample(20):
            assert isinstance(p['n'], int)

    def test_categorical_var_in_categories(self):
        cats = ['tt', 'ss', 'ff']
        ps = VariableSpace([Variable('c', var_type='categorical',
                                            categories=cats)])
        for p in ps.sample(20):
            assert p['c'] in cats

    def test_log_uniform_in_bounds(self):
        ps = VariableSpace([Variable('w', bounds=(1e-9, 1e-6),
                                            scale='log')])
        for p in ps.sample(50):
            assert 1e-9 <= p['w'] <= 1e-6

    def test_n_zero_returns_empty(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        assert ps.sample(0) == []

    def test_unknown_method_raises(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        with pytest.raises(ValueError, match='Unknown sampling method'):
            ps.sample(5, method='bayes')

    def test_sobol_method_raises_mentioning_halton(self, cont_vars):
        """'sobol' was renamed to 'halton' 2026-08-23 — no alias. The old
        name must fail loud and point the caller at the new one."""
        ps = VariableSpace(list(cont_vars))
        with pytest.raises(ValueError, match='halton'):
            ps.sample(5, method='sobol')


# ---------------------------------------------------------------------------
# LHS sampling
# ---------------------------------------------------------------------------

class TestSampleLHS:

    def test_returns_n_dicts(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        assert len(ps.sample(8, 'lhs')) == 8

    def test_dict_keys_correct(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        for p in ps.sample(6, 'lhs'):
            assert set(p.keys()) == set(ps.names)

    def test_values_within_bounds(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        for p in ps.sample(20, 'lhs'):
            assert 0.5e-6 <= p['w'] <= 20e-6
            assert 0.15e-6 <= p['l'] <= 2e-6

    def test_each_var_covers_all_strata(self):
        """Each variable independently has one sample per stratum."""
        ps = VariableSpace([Variable('w', bounds=(0.0, 1.0)),
                             Variable('l', bounds=(0.0, 1.0))])
        n = 10
        pts = ps.sample(n, 'lhs')
        lo, hi = 0.0, 1.0
        for name in ps.names:
            vals = sorted(p[name] for p in pts)
            for i in range(n):
                stratum_lo = lo + i * (hi - lo) / n
                stratum_hi = lo + (i + 1) * (hi - lo) / n
                assert any(stratum_lo <= v <= stratum_hi + 1e-12 for v in vals), \
                    f"{name}: stratum [{stratum_lo:.2f}, {stratum_hi:.2f}] not covered"

    def test_vars_not_identical_rank_order(self):
        """With enough samples the two variables' strata orderings differ."""
        ps = VariableSpace([Variable('w', bounds=(0.0, 1.0)),
                             Variable('l', bounds=(0.0, 1.0))])
        n = 20
        pts = ps.sample(n, 'lhs')
        rank_w = sorted(range(n), key=lambda i: pts[i]['w'])
        rank_l = sorted(range(n), key=lambda i: pts[i]['l'])
        assert rank_w != rank_l, "LHS variables should not be perfectly correlated"

    def test_categorical_falls_back_to_random(self):
        cats = ['tt', 'ss', 'ff']
        ps = VariableSpace([Variable('c', var_type='categorical',
                                            categories=cats)])
        for p in ps.sample(10, 'lhs'):
            assert p['c'] in cats

    def test_log_uniform_within_bounds(self):
        ps = VariableSpace([Variable('w', bounds=(1e-9, 1e-6),
                                            scale='log')])
        for p in ps.sample(20, 'lhs'):
            assert 9e-10 <= p['w'] <= 1.01e-6

    def test_integer_var_returns_ints(self):
        ps = VariableSpace([Variable('n', bounds=(1, 20),
                                            var_type='integer')])
        for p in ps.sample(10, 'lhs'):
            assert isinstance(p['n'], int)

    def test_unbounded_var_raises(self):
        ps = VariableSpace([Variable('w')])
        with pytest.raises(ValueError, match="'w'"):
            ps.sample(5, 'lhs')


# ---------------------------------------------------------------------------
# Halton sampling
# ---------------------------------------------------------------------------

class TestSampleHalton:

    def test_returns_n_dicts(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        assert len(ps.sample(8, 'halton')) == 8

    def test_halton_matches_old_sobol_sequence(self):
        """Rename is not a behavior change — pin the first 5 values of a
        single bounded (0,1) variable to what method='sobol' used to
        produce (base-2 van der Corput / first Halton dimension)."""
        v = Variable('w', bounds=(0.0, 1.0))
        ps = VariableSpace([v])
        pts = ps.sample(5, method='halton')
        vals = [p['w'] for p in pts]
        expected = [0.5, 0.25, 0.75, 0.125, 0.625]
        for got, want in zip(vals, expected):
            assert got == pytest.approx(want)

    def test_values_within_bounds(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        for p in ps.sample(20, 'halton'):
            assert 0.5e-6 <= p['w'] <= 20e-6
            assert 0.15e-6 <= p['l'] <= 2e-6

    def test_two_vars_use_different_sequences(self):
        """w (base-2) and l (base-3) produce different sequences."""
        ps = VariableSpace([Variable('w', bounds=(0.0, 1.0)),
                             Variable('l', bounds=(0.0, 1.0))])
        pts = ps.sample(8, 'halton')
        w_vals = [p['w'] for p in pts]
        l_vals = [p['l'] for p in pts]
        assert w_vals != l_vals

    def test_low_discrepancy_property(self):
        """For n=16, each stratum [i/16, (i+1)/16] covered for each var."""
        ps = VariableSpace([Variable('w', bounds=(0.0, 1.0))])
        n = 16
        pts = ps.sample(n, 'halton')
        vals = sorted(p['w'] for p in pts)
        for i in range(n):
            lo, hi = i / n, (i + 1) / n
            assert any(lo <= v <= hi + 1e-12 for v in vals), \
                f"Stratum [{lo:.3f},{hi:.3f}] not covered"

    def test_categorical_samples_are_declared_categories(self):
        cats = ['tt', 'ss', 'ff']
        ps = VariableSpace([Variable('c', var_type='categorical',
                                            categories=cats)])
        for p in ps.sample(10, 'halton'):
            assert p['c'] in cats

    def test_log_uniform_within_bounds(self):
        ps = VariableSpace([Variable('w', bounds=(1e-9, 1e-6),
                                            scale='log')])
        for p in ps.sample(16, 'halton'):
            assert 9e-10 <= p['w'] <= 1.01e-6

    def test_unbounded_var_raises(self):
        ps = VariableSpace([Variable('w')])
        with pytest.raises(ValueError, match="'w'"):
            ps.sample(5, 'halton')


# ---------------------------------------------------------------------------
# Grid sampling
# ---------------------------------------------------------------------------

class TestSampleGrid:

    def test_total_count_is_n_power_d(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        pts = ps.sample(3, 'grid')
        assert len(pts) == 3 ** 2

    def test_single_var_returns_n_dicts(self):
        ps = VariableSpace([Variable('w', bounds=(0.0, 1.0))])
        assert len(ps.sample(5, 'grid')) == 5

    def test_dict_keys_correct(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        for p in ps.sample(2, 'grid'):
            assert set(p.keys()) == set(ps.names)

    def test_endpoints_covered(self):
        ps = VariableSpace([Variable('w', bounds=(1.0, 5.0))])
        pts = ps.sample(5, 'grid')
        w_vals = [p['w'] for p in pts]
        assert min(w_vals) == pytest.approx(1.0)
        assert max(w_vals) == pytest.approx(5.0)

    def test_categorical_uses_all_categories(self):
        cats = ['tt', 'ss', 'ff']
        ps = VariableSpace([Variable('c', var_type='categorical',
                                            categories=cats)])
        pts = ps.sample(10, 'grid')  # n ignored for categorical
        assert set(p['c'] for p in pts) == set(cats)

    def test_integer_var_returns_ints(self):
        ps = VariableSpace([Variable('n', bounds=(1, 5),
                                            var_type='integer')])
        for p in ps.sample(5, 'grid'):
            assert isinstance(p['n'], int)


# ---------------------------------------------------------------------------
# apply()
# ---------------------------------------------------------------------------

class TestApply:

    def test_sets_variable_values(self, cont_vars):
        w, l = cont_vars
        ps = VariableSpace([w, l])
        ps.apply({'w': 1.5e-6, 'l': 0.3e-6})
        assert w.value == pytest.approx(1.5e-6)
        assert l.value == pytest.approx(0.3e-6)

    def test_unknown_key_raises(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        with pytest.raises(KeyError, match='unknown'):
            ps.apply({'unknown': 1.0})

    def test_partial_point_sets_only_given_keys(self, cont_vars):
        w, l = cont_vars
        w.value = 1.0e-6
        l.value = 0.5e-6
        ps = VariableSpace([w, l])
        ps.apply({'w': 2.0e-6})
        assert w.value == pytest.approx(2.0e-6)
        assert l.value == pytest.approx(0.5e-6)  # unchanged

    def test_apply_then_sample_uses_new_value(self, cont_vars):
        w, l = cont_vars
        ps = VariableSpace([w, l])
        ps.apply({'w': 3.0e-6, 'l': 1.0e-6})
        assert w.value == pytest.approx(3.0e-6)

    def test_out_of_bounds_value_raises(self):
        """apply() must validate before assigning — the reviewed repro:
        validate({'x': 42}) said False, but apply() assigned 42 anyway."""
        x = Variable('x', bounds=(0, 10))
        ps = VariableSpace([x])
        assert ps.validate({'x': 42}) is False
        with pytest.raises(ValueError, match="'x'"):
            ps.apply({'x': 42})
        assert x.value is None  # unchanged — apply() must not have assigned

    def test_invalid_point_is_atomic(self):
        """Two variables, first valid second invalid — neither is assigned."""
        a = Variable('a', bounds=(0, 10), value=1.0)
        b = Variable('b', bounds=(0, 10), value=2.0)
        ps = VariableSpace([a, b])
        with pytest.raises(ValueError):
            ps.apply({'a': 5.0, 'b': 99})
        assert a.value == 1.0  # unchanged, even though 'a' alone was valid
        assert b.value == 2.0  # unchanged


# ---------------------------------------------------------------------------
# Optimizer interface
# ---------------------------------------------------------------------------

class TestOptimizerInterface:

    def test_to_bounds_returns_list_of_tuples(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        b = ps.to_bounds()
        assert b == [(0.5e-6, 20e-6), (0.15e-6, 2e-6)]

    def test_to_bounds_order_matches_names(self):
        a = Variable('a', bounds=(0.0, 1.0))
        b = Variable('b', bounds=(10.0, 20.0))
        ps = VariableSpace([a, b])
        bounds = ps.to_bounds()
        assert bounds[0] == (0.0, 1.0)
        assert bounds[1] == (10.0, 20.0)

    def test_to_bounds_unbounded_raises(self):
        ps = VariableSpace([Variable('w')])
        with pytest.raises(ValueError, match="'w'"):
            ps.to_bounds()

    def test_to_bounds_categorical_returns_index_range(self):
        cats = ['tt', 'ss', 'ff']
        ps = VariableSpace([Variable('c', var_type='categorical',
                                            categories=cats)])
        assert ps.to_bounds() == [(0, 2)]

    def test_to_list_preserves_names_order(self):
        a = Variable('a', bounds=(0.0, 1.0))
        b = Variable('b', bounds=(0.0, 1.0))
        ps = VariableSpace([a, b])
        lst = ps.to_list({'b': 0.9, 'a': 0.1})
        # names order is ['a', 'b']
        assert lst == [0.1, 0.9]

    def test_from_list_roundtrip(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        point = {'w': 3e-6, 'l': 0.5e-6}
        assert ps.from_list(ps.to_list(point)) == point

    def test_from_list_wrong_length_raises(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        with pytest.raises(ValueError, match='Expected 2'):
            ps.from_list([1.0])

    def test_roundtrip_categorical(self):
        cats = ['tt', 'ss', 'ff']
        ps = VariableSpace([Variable('c', var_type='categorical',
                                            categories=cats)])
        point = {'c': 'ss'}
        assert ps.from_list(ps.to_list(point)) == point


# ---------------------------------------------------------------------------
# validate() and clip()
# ---------------------------------------------------------------------------

class TestValidateAndClip:

    def test_validate_valid_point(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        assert ps.validate({'w': 5e-6, 'l': 0.5e-6}) is True

    def test_validate_out_of_bounds(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        assert ps.validate({'w': 100e-6, 'l': 0.5e-6}) is False

    def test_validate_categorical_valid(self):
        ps = VariableSpace([Variable('c', var_type='categorical',
                                            categories=['tt', 'ss', 'ff'])])
        assert ps.validate({'c': 'tt'}) is True

    def test_validate_categorical_invalid(self):
        ps = VariableSpace([Variable('c', var_type='categorical',
                                            categories=['tt', 'ss', 'ff'])])
        assert ps.validate({'c': 'fs'}) is False

    def test_clip_above_max(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        clipped = ps.clip({'w': 100e-6, 'l': 0.5e-6})
        assert clipped['w'] == pytest.approx(20e-6)

    def test_clip_below_min(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        clipped = ps.clip({'w': 0.1e-6, 'l': 0.5e-6})
        assert clipped['w'] == pytest.approx(0.5e-6)

    def test_clip_valid_unchanged(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        point = {'w': 5e-6, 'l': 0.5e-6}
        clipped = ps.clip(point)
        assert clipped['w'] == pytest.approx(5e-6)
        assert clipped['l'] == pytest.approx(0.5e-6)

    def test_clip_integer_rounds(self):
        ps = VariableSpace([Variable('n', bounds=(1, 20),
                                            var_type='integer')])
        clipped = ps.clip({'n': 3.7})
        assert clipped['n'] == 4
        assert isinstance(clipped['n'], int)

    def test_clip_does_not_mutate_input(self, cont_vars):
        ps = VariableSpace(list(cont_vars))
        point = {'w': 100e-6, 'l': 0.5e-6}
        ps.clip(point)
        assert point['w'] == 100e-6  # original unchanged


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_variable_all_methods(self):
        ps = VariableSpace([Variable('w', bounds=(0.0, 1.0))])
        for method in ('random', 'lhs', 'halton', 'grid'):
            pts = ps.sample(4, method)
            assert len(pts) >= 1
            assert all('w' in p for p in pts)

    def test_all_categorical_space_random(self):
        ps = VariableSpace([
            Variable('c1', var_type='categorical', categories=['a', 'b']),
            Variable('c2', var_type='categorical', categories=['x', 'y', 'z']),
        ])
        pts = ps.sample(10, 'random')
        assert len(pts) == 10
        assert all(p['c1'] in ['a', 'b'] for p in pts)

    def test_all_categorical_lhs_fallback(self):
        ps = VariableSpace([
            Variable('c', var_type='categorical', categories=['tt', 'ff']),
        ])
        pts = ps.sample(6, 'lhs')
        assert all(p['c'] in ['tt', 'ff'] for p in pts)

    def test_mixed_types_sample_random(self, mixed_space):
        pts = mixed_space.sample(20, 'random')
        for p in pts:
            assert 0.5e-6 <= p['w'] <= 20e-6
            assert isinstance(p['nfin'], int)
            assert 1 <= p['nfin'] <= 32
            assert p['corner'] in ['tt', 'ss', 'ff']

    def test_mixed_types_sample_lhs(self, mixed_space):
        pts = mixed_space.sample(10, 'lhs')
        assert len(pts) == 10
        for p in pts:
            assert p['corner'] in ['tt', 'ss', 'ff']
            assert isinstance(p['nfin'], int)


# ---------------------------------------------------------------------------
# Public export
# ---------------------------------------------------------------------------

class TestPublicExport:

    def test_importable_from_package(self):
        from silk.optimize import VariableSpace as PS
        from silk import VariableSpace as PS2
        assert PS is PS2 is VariableSpace

    @pytest.mark.parametrize('old', ['Space', 'ParameterSpace'])
    def test_removed_names_raise_import_error(self, old):
        import silk, silk.optimize, silk.optimize.parameter_space as m
        for mod in (silk, silk.optimize, m):
            assert not hasattr(mod, old)
            with pytest.raises(ImportError):
                exec(f'from {mod.__name__} import {old}')

    def test_option_classes_not_importable_from_package(self):
        for name in ('VarType', 'Distribution'):
            with pytest.raises(ImportError):
                exec(f'from silk.optimize.parameter_space import {name}', {})



class TestVariableSpaceGroups:
    """Tests for the groups= correlated-variable feature."""

    @pytest.fixture
    def three_vars(self):
        w_n = Variable('w_n', bounds=(0.42, 20.0))
        w_p = Variable('w_p', bounds=(0.42, 20.0))
        l   = Variable('l',   bounds=(0.15, 1.0))
        return w_n, w_p, l

    def test_no_groups_still_works(self, three_vars):
        w_n, w_p, l = three_vars
        space = VariableSpace([w_n, w_p, l])
        pts = space.sample(10, method='lhs')
        assert len(pts) == 10
        assert set(pts[0].keys()) == {'w_n', 'w_p', 'l'}

    def test_single_group_correct_size(self, three_vars):
        w_n, w_p, l = three_vars
        space = VariableSpace([w_n, w_p, l], groups=[[w_n, w_p]])
        pts = space.sample(12, method='lhs')
        assert len(pts) == 12

    def test_multiple_groups(self, three_vars):
        w_n, w_p, l = three_vars
        vdd = Variable('vdd', bounds=(1.6, 2.0))
        space = VariableSpace([w_n, w_p, l, vdd], groups=[[w_n, w_p], [l, vdd]])
        pts = space.sample(10, method='lhs')
        assert len(pts) == 10
        assert set(pts[0].keys()) == {'w_n', 'w_p', 'l', 'vdd'}

    def test_grouped_vars_share_rank_order_lhs(self, three_vars):
        """w_n and w_p in same group should have identical rank ordering."""
        w_n, w_p, l = three_vars
        space = VariableSpace([w_n, w_p, l], groups=[[w_n, w_p]])
        pts = space.sample(20, method='lhs')
        wn_vals = [p['w_n'] for p in pts]
        wp_vals = [p['w_p'] for p in pts]
        # Same rank ordering means argsort of both columns is identical
        wn_ranks = sorted(range(20), key=lambda i: wn_vals[i])
        wp_ranks = sorted(range(20), key=lambda i: wp_vals[i])
        assert wn_ranks == wp_ranks

    def test_ungrouped_var_is_independent(self, three_vars):
        """l is not grouped with w_n/w_p — its rank order should differ."""
        w_n, w_p, l = three_vars
        import random as _random
        space = VariableSpace([w_n, w_p, l], groups=[[w_n, w_p]])
        _random.seed(42)
        pts = space.sample(20, method='lhs')
        wn_vals = [p['w_n'] for p in pts]
        l_vals  = [p['l']   for p in pts]
        wn_ranks = sorted(range(20), key=lambda i: wn_vals[i])
        l_ranks  = sorted(range(20), key=lambda i: l_vals[i])
        assert wn_ranks != l_ranks  # independent — very unlikely to match

    def test_variable_not_in_space_raises(self, three_vars):
        w_n, w_p, l = three_vars
        outsider = Variable('outsider', bounds=(0.0, 1.0))
        with pytest.raises(ValueError, match="not in this VariableSpace"):
            VariableSpace([w_n, w_p, l], groups=[[w_n, outsider]])

    def test_variable_in_two_groups_raises(self, three_vars):
        w_n, w_p, l = three_vars
        with pytest.raises(ValueError, match="more than one group"):
            VariableSpace([w_n, w_p, l], groups=[[w_n, w_p], [w_n, l]])

    def test_groups_halton(self, three_vars):
        w_n, w_p, l = three_vars
        space = VariableSpace([w_n, w_p, l], groups=[[w_n, w_p]])
        pts = space.sample(12, method='halton')
        assert len(pts) == 12
        wn_ranks = sorted(range(12), key=lambda i: pts[i]['w_n'])
        wp_ranks = sorted(range(12), key=lambda i: pts[i]['w_p'])
        assert wn_ranks == wp_ranks

    def test_groups_random(self, three_vars):
        w_n, w_p, l = three_vars
        space = VariableSpace([w_n, w_p, l], groups=[[w_n, w_p]])
        pts = space.sample(12, method='random')
        assert len(pts) == 12

    def test_groups_grid(self, three_vars):
        """Groups are respected when using grid sampling."""
        w_n, w_p, l = three_vars
        space = VariableSpace([w_n, w_p, l], groups=[[w_n, w_p]])
        pts = space.sample(3, method='grid')
        assert len(pts) == 9
        assert [(p["w_n"], p["w_p"], p["l"]) for p in pts] == [
            (wn, wp, length)
            for wn, wp in zip(w_n.sample(3, "grid"), w_p.sample(3, "grid"))
            for length in l.sample(3, "grid")
        ]
        for p in pts:
            assert 'w_n' in p and 'w_p' in p and 'l' in p

    # ------------------------------------------------------------------
    # Discrete (resolution-constrained) variables in group sampling
    # (lines 248-250, 273-279, 310-312, 329-334)
    # ------------------------------------------------------------------

    @pytest.fixture
    def discrete_var(self):
        """A variable with resolution set — snaps to a finite grid."""
        return Variable('w_d', bounds=(0.0, 1.0), resolution=0.1,
                              enforce_resolution=True)

    @pytest.fixture
    def discrete_space(self, discrete_var):
        """VariableSpace with one discrete and one continuous variable."""
        cont = Variable('x', bounds=(0.0, 1.0))
        return VariableSpace([discrete_var, cont])

    def test_discrete_random_values_on_grid(self, discrete_space):
        """Random sampling of a resolution-constrained var stays on the grid
        (covers lines 248-250)."""
        pts = discrete_space.sample(20, method='random')
        for p in pts:
            # Each value should be a multiple of 0.1 (within floating-point tolerance)
            assert abs(p['w_d'] % 0.1) < 1e-9 or abs(p['w_d'] % 0.1 - 0.1) < 1e-9, \
                f"Value {p['w_d']} is not on the resolution grid"

    def test_discrete_lhs_values_on_grid(self, discrete_space):
        """LHS sampling of a resolution-constrained var stays on the grid
        (covers lines 273-279)."""
        pts = discrete_space.sample(10, method='lhs')
        for p in pts:
            assert abs(p['w_d'] % 0.1) < 1e-9 or abs(p['w_d'] % 0.1 - 0.1) < 1e-9, \
                f"Value {p['w_d']} is not on the resolution grid"

    def test_discrete_halton_values_on_grid(self):
        """Halton sampling of a resolution-constrained var stays on the grid
        (covers lines 310-312)."""
        v = Variable('w_d', bounds=(0.0, 1.0), resolution=0.1,
                           enforce_resolution=True)
        ps = VariableSpace([v])
        pts = ps.sample(10, method='halton')
        for p in pts:
            assert abs(p['w_d'] % 0.1) < 1e-9 or abs(p['w_d'] % 0.1 - 0.1) < 1e-9, \
                f"Value {p['w_d']} is not on the resolution grid"

    def test_discrete_grid_uses_evenly_spaced_subset_when_grid_larger(self):
        """Grid sampling picks n evenly-spaced values when discrete set > n
        (covers lines 329-334)."""
        # resolution=0.01 over [0,1] gives 101 discrete values; request n=5
        v = Variable('w_d', bounds=(0.0, 1.0), resolution=0.01,
                           enforce_resolution=True)
        ps = VariableSpace([v])
        pts = ps.sample(5, method='grid')
        # Should get exactly 5 values
        assert len(pts) == 5
        vals = [p['w_d'] for p in pts]
        # Each value should be on the 0.01 grid
        for val in vals:
            assert abs(val % 0.01) < 1e-9 or abs(val % 0.01 - 0.01) < 1e-9

    def test_discrete_grid_uses_all_values_when_grid_smaller_or_equal(self):
        """Grid sampling uses all discrete values when set size <= n
        (covers the else-branch of lines 329-334)."""
        # resolution=0.25 over [0,1] gives 5 discrete values; request n=10
        v = Variable('w_d', bounds=(0.0, 1.0), resolution=0.25,
                           enforce_resolution=True)
        ps = VariableSpace([v])
        pts = ps.sample(10, method='grid')
        # discrete values are [0.0, 0.25, 0.50, 0.75, 1.0] → 5 points
        assert len(pts) == 5

    def test_discrete_grouped_random_values_on_grid(self, three_vars):
        """Grouped discrete vars in random sampling produce on-grid values
        (covers lines 248-250 with groups)."""
        w_n, w_p, l = three_vars
        # Make w_n discrete
        w_n_d = Variable('w_n', bounds=(0.5, 2.0), resolution=0.5,
                               enforce_resolution=True)
        w_p_d = Variable('w_p', bounds=(0.5, 2.0), resolution=0.5,
                               enforce_resolution=True)
        space = VariableSpace([w_n_d, w_p_d, l], groups=[[w_n_d, w_p_d]])
        pts = space.sample(8, method='random')
        for p in pts:
            assert abs(p['w_n'] % 0.5) < 1e-9 or abs(p['w_n'] % 0.5 - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# _scale_to_var LOG_UNIFORM with lo <= 0 raises (line 353)
# ---------------------------------------------------------------------------

class TestScaleToVarLogUniformError:

    def test_log_uniform_lo_zero_raises(self):
        """LOG_UNIFORM with bounds[0] == 0 raises ValueError in _scale_to_var
        (covers line 353)."""
        v = Variable('x', bounds=(0.0, 1.0),
                           scale='log')
        ps = VariableSpace([v])
        with pytest.raises(ValueError, match="scale='log'"):
            ps.sample(5, method='lhs')

    def test_log_uniform_lo_negative_raises(self):
        """LOG_UNIFORM with bounds[0] < 0 raises ValueError."""
        v = Variable('x', bounds=(-1.0, 1.0),
                           scale='log')
        ps = VariableSpace([v])
        with pytest.raises(ValueError, match="scale='log'"):
            ps.sample(5, method='halton')


class TestNoneBoundEnds:
    """A bound end of None means UNBOUNDED — the shape every PDK-derived
    variable has (real minval, no maxval; catalogs spell 'no bound' as
    '-1'). Before 2026-08-23, validate() raised TypeError on the None end —
    and only for values that had already PASSED the finite check — and
    clip()/_snap() had the same-shaped crash."""

    def test_validate_lower_only(self):
        v = Variable('w', bounds=(0.42, None))
        assert v.validate(0.5) is True          # raised TypeError before
        assert v.validate(1e9) is True          # unbounded above
        assert v.validate(0.1) is False         # below minval still rejected

    def test_validate_upper_only(self):
        v = Variable('x', bounds=(None, 5.0))
        assert v.validate(-1e9) is True
        assert v.validate(4.0) is True
        assert v.validate(6.0) is False

    def test_clip_with_none_ends(self):
        v = Variable('w', bounds=(0.42, None))
        assert v.clip(0.1) == 0.42
        assert v.clip(1e9) == 1e9               # nothing to clip against
        u = Variable('x', bounds=(None, 5.0))
        assert u.clip(6.0) == 5.0

    def test_snap_with_none_upper(self):
        v = Variable('w', bounds=(0.42, None), resolution=0.01,
                           enforce_resolution=True)
        v.value = 0.505                          # crashed in _snap before
        assert abs(v.value - 0.5) < 1e-12 or abs(v.value - 0.51) < 1e-12


class TestConstraintBoundsDerivation:
    """PDK catalogs spell an unbounded end as '-1'; the derivation used to
    hardcode (minval, None) and never read maxval at all."""

    def test_minval_only(self):
        from types import SimpleNamespace
        from silk.elements.base import AnalogElement
        c = SimpleNamespace(minval='0.42', maxval='-1')
        assert AnalogElement._constraint_bounds(c) == (0.42, None)

    def test_real_maxval_now_read(self):
        from types import SimpleNamespace
        from silk.elements.base import AnalogElement
        c = SimpleNamespace(minval='0.42', maxval='10')
        assert AnalogElement._constraint_bounds(c) == (0.42, 10.0)

    def test_both_unbounded_is_none(self):
        from types import SimpleNamespace
        from silk.elements.base import AnalogElement
        c = SimpleNamespace(minval='-1', maxval=None)
        assert AnalogElement._constraint_bounds(c) is None
        assert AnalogElement._constraint_bounds(None) is None

    def test_pdk_derived_variable_validates_without_crashing(self, pdk):
        """End to end through the real sky130 catalog: this exact call
        raised TypeError before (bounds were (0.42, None))."""
        var = pdk.nmos1p8.w.variable(name='w_n', value=1.0)
        assert var.validate(0.5) is True
        assert var.validate(0.1) is False


class TestOnePointGrid:
    def test_grid_n1_on_resolution_variable_no_zerodivision(self):
        """sample(1, 'grid') on a resolution-constrained variable divided by
        (n - 1) == 0 before 2026-08-23."""
        v = Variable('y', bounds=(0.0, 1.0), resolution=0.1,
                           enforce_resolution=True)
        pts = VariableSpace([v]).sample(1, method='grid')
        assert pts == [{'y': 0.0}]              # first grid point, matching
                                                # the continuous n=1 behavior


class TestSamplingRejectsUnboundedEnds:
    """2026-08-23: a None bounds END is legitimate variable state (PDK
    catalogs often set no upper limit, so `.variable()` derives (lo, None))
    — but sampling needs a finite interval. Both samplers used to crash
    with a raw TypeError deep in the scaling arithmetic."""

    def test_space_sample_raises_naming_the_variable(self):
        space = VariableSpace([
            Variable('w_n', 1.0, bounds=(0.42, None)),
            Variable('l_n', 0.5, bounds=(0.15, 1.0)),
        ])
        for method in ('halton', 'lhs', 'random', 'grid'):
            with pytest.raises(ValueError, match="w_n"):
                space.sample(3, method=method)

    def test_space_sample_ok_when_all_bounds_finite(self):
        space = VariableSpace([
            Variable('w_n', 1.0, bounds=(0.42, 20.0)),
        ])
        assert len(space.sample(3, method='halton')) == 3

    def test_module_variable_sample_raises_on_none_end(self):
        v = Variable('w_n', 1.0, bounds=(0.42, None))
        with pytest.raises(ValueError, match="unbounded"):
            v.sample(3, method='grid')


@pytest.mark.parametrize('method, expected', [
    ('random', [(0.7756856902451935, 18.735534453962618),
                (0.30016628491122543, 18.212284183827663),
                (0.22520718999059186, 10.052653045655747)]),
    ('lhs', [(0.25856189674839786, 12.91184481798754),
             (0.741735729996864, 16.684217681885247),
             (0.4333887616370751, 16.07076139460922)]),
    ('halton', [(0.5, 15.0), (0.25, 12.5), (0.75, 17.5)]),
])
def test_grouped_non_grid_sampling_unchanged(method, expected):
    a = Variable('a', bounds=(0, 1))
    b = Variable('b', bounds=(10, 20))
    points = VariableSpace([a, b], groups=[[a, b]]).sample(3, method, seed=7)
    assert len(points) == len(expected)
    for point, expected_pair in zip(points, expected):
        assert (point['a'], point['b']) == pytest.approx(expected_pair)


def test_grid_axes_follow_first_member_and_keep_dict_order():
    u = Variable('u', bounds=(0, 1))
    a = Variable('a', bounds=(10, 20))
    b = Variable('b', bounds=(100, 200))
    c = Variable('c', bounds=(30, 40))
    d = Variable('d', bounds=(300, 400))
    space = VariableSpace([u, a, b, c, d], groups=[[d, b], [c, a]])
    points = space.sample(2, 'grid')
    assert [list(p) for p in points] == [['u', 'a', 'b', 'c', 'd']] * 8
    assert [tuple(p.values()) for p in points] == [
        (0, 10, 100, 30, 300), (0, 10, 200, 30, 400),
        (0, 20, 100, 40, 300), (0, 20, 200, 40, 400),
        (1, 10, 100, 30, 300), (1, 10, 200, 30, 400),
        (1, 20, 100, 40, 300), (1, 20, 200, 40, 400),
    ]


@pytest.mark.parametrize('shortened', [False, True])
def test_grouped_grid_refuses_different_column_lengths(shortened):
    a = (Variable('a', bounds=(0, 1), resolution=1, enforce_resolution=True)
         if shortened else Variable('a', var_type='categorical', categories=['z', 'a']))
    b = Variable('b', bounds=(10, 20))
    space = VariableSpace([a, b], groups=[[b, a]])
    if shortened:
        with pytest.warns(UserWarning):
            with pytest.raises(ValueError, match=r'a.*2.*b.*3'):
                space.sample(3, 'grid')
    else:
        with pytest.raises(ValueError, match=r'a.*2.*b.*3'):
            space.sample(3, 'grid')


def test_grouped_grid_equal_shortened_columns():
    a = Variable('a', bounds=(0, 1), resolution=1, enforce_resolution=True)
    b = Variable('b', bounds=(10, 11), resolution=1, enforce_resolution=True)
    with pytest.warns(UserWarning) as caught:
        points = VariableSpace([a, b], groups=[[a, b]]).sample(3, 'grid')
    assert len(caught) == 2
    assert points == [{'a': 0, 'b': 10}, {'a': 1, 'b': 11}]


def test_grid_ungrouped_output_and_category_order():
    a = Variable('a', var_type='categorical', categories=['z', 'a'])
    b = Variable('b', bounds=(10, 20))
    assert VariableSpace([a, b]).sample(3, 'grid') == [
        {'a': 'z', 'b': 10}, {'a': 'z', 'b': 15}, {'a': 'z', 'b': 20},
        {'a': 'a', 'b': 10}, {'a': 'a', 'b': 15}, {'a': 'a', 'b': 20},
    ]
    grouped = VariableSpace([a, b], groups=[[a, b]])
    assert grouped.sample(0, 'grid') == []
    assert grouped.sample(2, 'grid') == [
        {'a': 'z', 'b': 10}, {'a': 'a', 'b': 20},
    ]


def test_grouped_log_grid_keeps_repeated_values():
    a = Variable('a', bounds=(1, 2), var_type='integer', scale='log')
    b = Variable('b', bounds=(10, 40), scale='log')
    points = VariableSpace([a, b], groups=[[a, b]]).sample(3, 'grid')
    assert [p['a'] for p in points] == [1, 1, 2]
    assert [p['b'] for p in points] == pytest.approx([10, 20, 40])
