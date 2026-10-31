"""Named numeric design parameters and sampling distributions."""

# Variable — a named number passed to any device, source, waveform or passive
# parameter; its current value is written when the netlist is emitted.
# Not tied to a module.

import math
import numpy as np
import warnings
from typing import Any, List, Optional, Tuple


def _sampling_rng(seed):
    """Create one isolated generator for a public sampling call."""
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an int or None (not bool)")
    return np.random.default_rng(seed)


# ----------------------------------------------------------------------
# Arithmetic on Variable — lazy expression nodes.
# A _VarExpr lazily computes its .value from its operands each time it
# is read, so updates to leaf Variable values propagate through
# derived expressions automatically at netlist-generation time.
# ----------------------------------------------------------------------

def _op_mul(a, b):      return a * b
def _op_add(a, b):      return a + b
def _op_sub(a, b):      return a - b
def _op_truediv(a, b):  return a / b
def _op_floordiv(a, b): return a // b
def _op_neg(a):         return -a


# Printing table: operator function -> (symbol, precedence). Higher binds
# tighter; a negation and a negative number literal sit above * and /.
_OP_PRINT = {
    _op_add:      (' + ', 1),
    _op_sub:      (' - ', 1),
    _op_mul:      ('*', 2),
    _op_truediv:  ('/', 2),
    _op_floordiv: ('//', 2),
}
_PREC_UNARY = 3
_PREC_ATOM = 4


def _operand_text(o, min_prec):
    """Text of one operand, parenthesised when it binds looser than *min_prec*."""
    if isinstance(o, _VarExpr):
        text, prec = o._text_prec()
    elif hasattr(o, 'value') and hasattr(o, 'name'):   # a Variable leaf
        text, prec = str(o.name), _PREC_ATOM
    else:
        text = repr(o)
        prec = _PREC_UNARY if text.startswith('-') else _PREC_ATOM
    return f'({text})' if prec < min_prec else text


class _VarArithmeticMixin:
    """Mixin giving arithmetic operators to Variable and _VarExpr.
    Every operator returns a new ``_VarExpr`` — expressions are pure and
    composable. Leaves stay as the sole place value can be assigned."""
    def __mul__(self, other):       return _VarExpr(_op_mul, self, other)
    def __rmul__(self, other):      return _VarExpr(_op_mul, other, self)
    def __add__(self, other):       return _VarExpr(_op_add, self, other)
    def __radd__(self, other):      return _VarExpr(_op_add, other, self)
    def __sub__(self, other):       return _VarExpr(_op_sub, self, other)
    def __rsub__(self, other):      return _VarExpr(_op_sub, other, self)
    def __truediv__(self, other):   return _VarExpr(_op_truediv, self, other)
    def __rtruediv__(self, other):  return _VarExpr(_op_truediv, other, self)
    def __floordiv__(self, other):  return _VarExpr(_op_floordiv, self, other)
    def __rfloordiv__(self, other): return _VarExpr(_op_floordiv, other, self)
    def __neg__(self):              return _VarExpr(_op_neg, self)


class _VarExpr(_VarArithmeticMixin):
    """Lazy arithmetic expression over Variables.

    Created by operators on ``Variable`` / ``_VarExpr``. The
    expression has a read-only ``.value`` property that re-evaluates
    each time it is read — so updating a leaf's ``.value`` propagates
    through any expressions that reference it on the next
    ``generate_netlist()``.

    Expressions are derived, not independent: they cannot be added to
    a ``VariableSpace``. Put the leaf ``Variable`` in the
    parameter space; derived expressions follow automatically.
    """
    def __init__(self, fn, *ops):
        self._fn = fn
        self._ops = ops

    @property
    def value(self):
        vals = []
        for o in self._ops:
            v = o.value if hasattr(o, 'value') else o
            if v is None:
                raise ValueError(
                    f"_VarExpr: variable '{getattr(o, 'name', o)}' "
                    f"has no value assigned (in {self!r})"
                )
            vals.append(v)
        return self._fn(*vals)

    def _text_prec(self):
        """(text, precedence) of this expression, e.g. ``('2*w_in', 2)``."""
        if self._fn is _op_neg:
            return '-' + _operand_text(self._ops[0], _PREC_UNARY), _PREC_UNARY
        sym, prec = _OP_PRINT[self._fn]
        a, b = self._ops
        # The right operand is parenthesised at equal precedence too, so
        # the text always shows how the expression was grouped.
        return _operand_text(a, prec) + sym + _operand_text(b, prec + 1), prec

    def __repr__(self):
        return self._text_prec()[0]


def _leaf_variables(val):
    """Read Variable leaves from stored parameter data without evaluation.

    Visit expression operands, Normal nominal/sigma and Uniform low/high,
    dict keys then values, list/tuple/set/frozenset items, and object-bearing
    ndarray/structured scalar fields. Other objects are opaque. Containers
    are visited once by identity to terminate cycles; Variable occurrences
    may repeat. Ordered containers retain their order; sets have no order.
    """
    # Local import defers loading statistical declaration types until discovery.
    from silk.simulation.variation import Normal, Uniform
    seen = {}

    def walk(value):
        if isinstance(value, Variable):
            yield value
        elif isinstance(value, _VarExpr):
            for operand in value._ops:
                yield from walk(operand)
        elif isinstance(value, Normal):
            yield from walk(value.nominal)
            yield from walk(value.sigma)
        elif isinstance(value, Uniform):
            yield from walk(value.low)
            yield from walk(value.high)
        elif isinstance(value, (dict, list, tuple, set, frozenset, np.ndarray, np.void)):
            if id(value) in seen:
                return
            # Retain references too: numpy iteration can create temporary
            # structured scalars whose ids could otherwise be reused.
            seen[id(value)] = value
            if isinstance(value, dict):
                for key, item in value.items():
                    yield from walk(key)
                    yield from walk(item)
            elif isinstance(value, np.void):
                if value.dtype.hasobject:
                    for name in value.dtype.names or ():
                        yield from walk(value[name])
            elif isinstance(value, np.ndarray):
                if value.dtype.hasobject:
                    for item in value.flat:
                        yield from walk(item)
            else:
                for item in value:
                    yield from walk(item)

    return list(walk(val))


class Variable(_VarArithmeticMixin):
    """Represents a circuit variable that can be shared between components.

    Parameters
    ----------
    name : str
        Identifies the variable: the key in a VariableSpace and an Optimizer's
        results, the name in error messages, and the name in generated
        code. It is not written to the netlist; a device parameter holding
        a Variable is emitted as the variable's current number.
    value : float, optional
        Nominal / initial value.
    bounds : tuple of float, optional
        Hard bounds for the parameter.  Required for sample() and validate().
    var_type : str, optional
        One of 'continuous' (default), 'integer', 'categorical'. Exact lowercase
        options are validated both at construction and on assignment.
    scale : str, optional
        'linear' (default) or 'log'. Every method samples in these coordinates
        before projecting onto the resolution or integer domain. Log requires
        finite bounds with 0 < lo <= hi at sampling time; it cannot be used
        with categorical variables.
    resolution : float, optional
        Manufacturing / simulation grid — the smallest increment the parameter
        can take.  When set and enforce_resolution=True, all sampled and
        assigned values are snapped to the nearest valid grid point.
        Example: resolution=0.005 for Sky130 µm dimensions. Integer variables
        require a whole-number resolution and, when set, a whole-number lower
        bound. Construction and assignment reject incompatible grids, even
        when enforce_resolution=False. An open lower bound is checked when
        sampling; supply a whole-number lower bound first.
    enforce_resolution : bool, optional
        If False the resolution is stored for documentation but not applied,
        allowing continuous optimisers to work without snapping.  Default True.
    categories : list, optional
        Allowed values when var_type='categorical'.

    Examples
    --------
    >>> from silk import Variable
    >>> width = Variable('w', value=2.0, bounds=(1.0, 4.0))
    >>> width.value = 3.0
    >>> width.value
    3.0
    """

    def __init__(
        self,
        name: str,
        value: Optional[float] = None,
        *,
        bounds: Optional[Tuple[float, float]] = None,
        var_type: str = 'continuous',
        scale: str = 'linear',
        resolution: Optional[float] = None,
        enforce_resolution: bool = True,
        categories: Optional[List] = None,
    ):
        self.name               = name
        self.bounds             = bounds
        self.var_type           = var_type
        self.scale              = scale
        self.resolution         = resolution
        self.enforce_resolution = enforce_resolution
        self.categories         = categories
        self._value             = None

        if value is not None:
            self._assign(value)

    @property
    def var_type(self):
        """Kind of the variable, one of ``'continuous'``, ``'integer'`` or ``'categorical'``.

        Assignment raises ``ValueError`` for any other string, for
        ``'categorical'`` combined with ``scale='log'``, and for an integer
        variable whose resolution or lower bound is not a whole number.
        """
        return self._var_type

    @var_type.setter
    def var_type(self, value):
        if value not in ('continuous', 'integer', 'categorical'):
            raise ValueError("var_type must be one of: 'continuous', 'integer', 'categorical'")
        if value == 'categorical' and getattr(self, '_scale', 'linear') == 'log':
            raise ValueError("scale='log' cannot be used with var_type='categorical'")
        self._validate_integer_grid(value, getattr(self, '_resolution', None),
                                    self.bounds)
        self._var_type = value

    @property
    def bounds(self):
        """Hard ``(low, high)`` bounds, or ``None`` when unbounded.

        Required by sampling and validation. Assignment raises ``ValueError``
        when a finite integer interval contains no integer, or an integer
        variable with resolution would get a nonintegral lower bound.
        """
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        self._validate_integer_grid(getattr(self, '_var_type', 'continuous'),
                                    getattr(self, '_resolution', None), value)
        self._bounds = value

    @property
    def resolution(self):
        """Smallest increment the parameter can take, or ``None``.

        Assignment raises ``ValueError`` if the value is not positive, or if
        an integer variable gets a resolution that is not a whole number.
        Snapping to this grid happens only when ``enforce_resolution`` is
        true. Sampling count is chosen separately by sample(n, ...);
        independent sweep-only spacing is not supported.
        """
        return self._resolution

    @resolution.setter
    def resolution(self, value):
        self._validate_integer_grid(self.var_type, value, self.bounds)
        if value is not None and value <= 0:
            raise ValueError(f"resolution must be None or positive, got {value!r}")
        self._resolution = value

    def _validate_integer_grid(self, var_type, resolution, bounds, *, sampling=False):
        """Reject grids that cannot represent an integer variable.

        Open lower bounds are valid catalog state, but must be filled before
        sampling. Validate proposed assignments before mutating any property.
        """
        if var_type != 'integer':
            return
        if (bounds is not None and all(v is not None and math.isfinite(v) for v in bounds)
                and math.ceil(bounds[0]) > math.floor(bounds[1])):
            raise ValueError(
                f"Variable '{self.name}': bounds={bounds!r} contain no integer; "
                "widen the bounds or use var_type='continuous'."
            )
        if resolution is None:
            return
        lo = bounds[0] if bounds is not None else None
        if (not math.isfinite(resolution) or resolution != int(resolution)
                or (lo is not None and (not math.isfinite(lo) or lo != int(lo)))
                or (sampling and lo is None)):
            raise ValueError(
                f"Variable '{self.name}': integer grid requires an integer resolution "
                f"and a whole-number lower bound; resolution={resolution!r}, "
                f"bounds={bounds!r}. Use an integer resolution and whole-number "
                "lower bound, or var_type='continuous'."
            )

    @property
    def scale(self):
        """Sampling coordinates, one of ``'linear'`` or ``'log'``.

        Assignment raises ``ValueError`` for any other string, and for
        ``'log'`` on a categorical variable.
        """
        return self._scale

    @scale.setter
    def scale(self, value):
        if value not in ('linear', 'log'):
            raise ValueError("scale must be one of: 'linear', 'log'")
        if value == 'log':
            if self.var_type == 'categorical':
                raise ValueError("scale='log' cannot be used with var_type='categorical'")
        self._scale = value

    def __repr__(self):
        return f"Variable(name='{self.name}', value={self.value})"

    # ------------------------------------------------------------------
    # Copying keeps the knob (TASK-400)
    # ------------------------------------------------------------------
    # A Variable is one shared knob. add_module and add_module_under_test
    # place a deep copy of the module; a copied Variable inside it would
    # keep its placement-time value while the caller's Variable changed,
    # and the netlist would silently ignore every later assignment. So a
    # copy of anything that holds a Variable points at the same Variable.
    # _VarExpr needs no hook: its leaves resolve here. Pickling does not
    # use these hooks, so the parallel optimizer's pickle snapshot still
    # freezes each design point's values.

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    # ------------------------------------------------------------------
    # value property — unchanged semantics, backward-compatible
    # ------------------------------------------------------------------

    @property
    def value(self) -> Optional[float]:
        """Return the assigned value."""
        return self._value

    @value.setter
    def value(self, val):
        self._assign(val)

    def _assign(self, val, warn_on_snap: bool = True):
        """Check *val* against the bounds, snap it to the grid, store it.

        A value outside the bounds raises ValueError and leaves the variable
        unchanged.  A value moved by grid snapping is stored snapped and, when
        *warn_on_snap*, reported with a UserWarning (VariableSpace.apply passes False:
        optimiser points are snapped by design).
        """
        from silk.simulation.variation import is_declaration
        if is_declaration(val):
            raise TypeError(f"Variable {self.name!r}: a statistical declaration cannot be its value")
        if val is None:
            self._value = val
            return
        if self.var_type != 'categorical' and not self.validate(val):
            raise ValueError(
                f"Variable '{self.name}': value {val!r} is outside its "
                f"bounds {self.bounds!r}; use clip() to clip explicitly"
            )
        snapped = self._snap(val)
        if warn_on_snap and snapped != val and not math.isclose(
                snapped, val, rel_tol=1e-9, abs_tol=1e-9 * self.resolution):
            warnings.warn(
                f"Variable '{self.name}': value {val!r} is off its "
                f"resolution grid ({self.resolution!r}); using {snapped!r}",
                UserWarning, stacklevel=3)
        self._value = snapped

    # ------------------------------------------------------------------
    # Validation and clipping
    # ------------------------------------------------------------------

    def _snap(self, val: float) -> float:
        """Snap *val* to the nearest resolution grid point.

        Returns *val* unchanged if resolution is None or enforce_resolution
        is False.  Always clips to bounds after snapping.
        """
        if self.resolution is None or not self.enforce_resolution:
            return val
        # A bound end of None means unbounded (PDK catalogs spell it '-1');
        # the grid anchors at the lower bound when there is one.
        lo, hi = self.bounds if self.bounds is not None else (None, None)
        # Index zero is the exact lower anchor shared with sampling.
        if lo is not None and val == lo:
            return val
        anchor = lo if lo is not None else 0.0
        steps = round((val - anchor) / self.resolution)
        snapped = anchor + steps * self.resolution
        # Clip to whichever bounds exist
        if lo is not None:
            snapped = max(lo, snapped)
        if hi is not None:
            snapped = min(hi, snapped)
        # Preserve the same normalized terminal point used by sampling.
        if (lo is not None and hi is not None and val == hi
                and math.isfinite(lo) and math.isfinite(hi)):
            last, endpoint = self._resolution_domain()
            if steps == last and endpoint is not None:
                return endpoint
        # Round away float drift (e.g. 0.42 + 3*0.005 = 0.43500000000000006)
        ndigits = max(0, -int(math.floor(math.log10(self.resolution)))) + 2
        cleaned = round(snapped, ndigits)
        if ((lo is not None and cleaned < lo) or
                (hi is not None and cleaned > hi)):
            return snapped
        # Cleanup must not merge distinct neighbouring grid points.
        if not (anchor + (steps - 1) * self.resolution < cleaned
                < anchor + (steps + 1) * self.resolution):
            return snapped
        return cleaned

    def validate(self, val) -> bool:
        """Return True if *val* is a legal value for this variable.

        For CATEGORICAL: must be in self.categories.
        For INTEGER/CONTINUOUS with bounds: must be within [min, max].
        """
        if self.var_type == 'categorical':
            return self.categories is not None and val in self.categories
        if self.bounds is not None:
            # Either end may be None = unbounded (PDK-derived variables
            # usually have a real minval and no maxval). `lo <= val <= hi`
            # raised TypeError on a None end — and only for values that had
            # already PASSED the finite check, the worst possible failure
            # placement.
            lo, hi = self.bounds
            if lo is not None and val < lo:
                return False
            if hi is not None and val > hi:
                return False
        return True

    def clip(self, val):
        """Return *val* clipped to bounds and snapped to resolution grid."""
        if self.var_type == 'categorical':
            return val  # no numeric clipping for categoricals
        if self.bounds is not None:
            lo, hi = self.bounds
            if lo is not None:
                val = max(lo, val)
            if hi is not None:
                val = min(hi, val)
        if self.var_type == 'integer':
            val = int(round(val))
        return self._snap(val)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n: int, method: str = 'random', *, seed: Optional[int] = 0) -> List:
        """Generate *n* sample values for this variable.

        Parameters
        ----------
        n : int
            Number of samples.
        method : str
            One of:

            - ``'random'``: uniform or log-uniform random samples within bounds.
            - ``'grid'``: select up to n distinct linear resolution/integer values;
              continuous count grids and log grids retain n draws.
            - ``'lhs'``: Latin Hypercube Sampling (stratified random).
            - ``'halton'``: low-discrepancy quasi-random (base-2 van der Corput
              sequence; this is Halton's first dimension, not Sobol; for true
              Sobol use scipy.stats.qmc.Sobol).

        seed : int or None
            Reproducible seed (default 0). None uses fresh OS entropy.
            Booleans and non-integers raise TypeError.

        Returns
        -------
        list
            Sampled values (category objects, floats, or ints).
            Categorical grid enumerates all categories in declared order for n > 0.
            Categorical Halton maps floor(u * category_count) to category indices;
            finite prefixes need not cover every category. Use grid for coverage.
            Categorical random and lhs use seeded random draws.
            Linear resolution and integer grids select at most n distinct values
            across the full domain; n=1 selects its first value. A shortened
            numeric grid emits one UserWarning at the public caller, following
            standard filters; it is not persisted in reports. Integers without
            enforced resolution use ceil(lo)..floor(hi). Continuous count grids
            retain n draws (including singleton repeats) and pin both endpoints
            exactly for n>1. Random/LHS/Halton keep n draws.
            Log grids retain n draws, including duplicates after projection. A
            resolution grid anchors at the lower bound; an off-grid upper bound
            projects to the last realizable value below it.
        """
        return self._sample(n, method, _sampling_rng(seed))

    def _sample(self, n, method, rng, *, warning_stacklevel=3):
        """Sample using the caller-owned generator for a joint draw."""
        _validate_sampling_method(method)
        self._validate_integer_grid(self.var_type, self.resolution, self.bounds,
                                    sampling=True)

        if self.var_type == 'categorical':
            if not self.categories:
                raise ValueError(f"Variable '{self.name}' has no categories defined")
            if method == 'grid':
                return self._categorical_grid(n)
            if method == 'halton':
                count = len(self.categories)
                return [self.categories[min(int(u * count), count - 1)]
                        for u in self._halton_samples(0, 1, n)]
            return [self.categories[int(rng.integers(len(self.categories)))] for _ in range(n)]

        if self.bounds is None or None in self.bounds:
            # A None END is legitimate variable state (PDK catalogs often set
            # no upper limit, so `.variable()` derives e.g. (0.42, None)) —
            # but sampling needs a finite interval, and without this check the
            # None reached the arithmetic as a raw TypeError.
            raise ValueError(
                f"Variable '{self.name}' has unbounded end(s): "
                f"bounds={self.bounds}. Sampling needs finite (min, max) — "
                "pass bounds=(lo, hi) explicitly."
            )
        self._require_log_bounds()
        if n == 0:
            return []
        lo, hi = self.bounds

        if (self.scale == 'linear' and self.resolution is not None
                and self.enforce_resolution):
            discrete = self._discrete_values()
            nd = len(discrete)
            if method == 'grid':
                return self._discrete_grid(n, warning_stacklevel=warning_stacklevel + 1)
            if method == 'random':
                return [discrete[int(rng.integers(len(discrete)))] for _ in range(n)]
            raw = (self._lhs_samples(0, 1, n, rng) if method == 'lhs'
                   else self._halton_samples(0, 1, n))
            return [discrete[min(int(u * nd), nd - 1)] for u in raw]

        if method == 'grid' and self.scale == 'linear' and self.var_type == 'integer':
            return self._discrete_grid(n, warning_stacklevel=warning_stacklevel + 1)
        if method == 'grid':
            values = self._grid_samples(lo, hi, n)
        elif method == 'lhs':
            values = self._lhs_samples(lo, hi, n, rng)
        elif method == 'halton':
            values = self._halton_samples(lo, hi, n)
        elif method == 'random':
            values = self._random_samples(lo, hi, n, rng)

        return [self._project_sample(v) for v in values]

    # ------------------------------------------------------------------
    # Internal sampling helpers
    # ------------------------------------------------------------------

    def _categorical_grid(self, n):
        """Enumerate categories in declared order for a nonempty grid request."""
        return list(self.categories) if n else []

    def _random_samples(self, lo, hi, n, rng):
        if self.scale == 'log':
            log_lo, log_hi = math.log(lo), math.log(hi)
            return [math.exp(rng.uniform(log_lo, log_hi)) for _ in range(n)]
        else:  # UNIFORM
            return [rng.uniform(lo, hi) for _ in range(n)]

    def _grid_samples(self, lo, hi, n):
        if self.scale == 'log':
            return self._log_grid_samples(lo, hi, n)
        if n == 1:
            return [lo]
        step = (hi - lo) / (n - 1)
        return [lo] + [min(hi, max(lo, lo + i * step)) for i in range(1, n - 1)] + [hi]

    def _log_grid_samples(self, lo, hi, n):
        if n == 1:
            return [lo]
        log_lo, log_hi = math.log(lo), math.log(hi)
        step = (log_hi - log_lo) / (n - 1)
        return [lo] + [math.exp(log_lo + i * step) for i in range(1, n - 1)] + [hi]

    def _lhs_samples(self, lo, hi, n, rng):
        """Latin Hypercube Sampling — one point per stratum."""
        strata = [(i / n, (i + 1) / n) for i in range(n)]
        rng.shuffle(strata)
        raw = [rng.uniform(s[0], s[1]) for s in strata]
        if self.scale == 'log':
            log_lo, log_hi = math.log(lo), math.log(hi)
            return [math.exp(log_lo + u * (log_hi - log_lo)) for u in raw]
        return [lo + u * (hi - lo) for u in raw]

    def _halton_samples(self, lo, hi, n):
        """Van der Corput sequence (base-2) — deterministic low-discrepancy.

        This is the first dimension of a Halton sequence, not Sobol; for
        true Sobol sequences use scipy.stats.qmc.Sobol.
        """
        def vdc(k):
            r, denom = 0.0, 1.0
            while k:
                denom *= 2
                k, remainder = divmod(k, 2)
                r += remainder / denom
            return r
        raw = [vdc(i + 1) for i in range(n)]
        if self.scale == 'log':
            log_lo, log_hi = math.log(lo), math.log(hi)
            return [math.exp(log_lo + u * (log_hi - log_lo)) for u in raw]
        return [lo + u * (hi - lo) for u in raw]


    def _require_log_bounds(self):
        if self.scale == 'log':
            b = self.bounds
            if (b is None or any(v is None or not math.isfinite(v) for v in b)
                    or not 0 < b[0] <= b[1]):
                raise ValueError(
                    f"Variable '{self.name}': scale='log' requires finite bounds "
                    f"with 0 < lo <= hi; got {b!r}"
                )

    def _resolution_domain(self):
        """Return the final index and optional normalized terminal value.

        Only the nearest boundary index may normalize to the exact upper
        endpoint. The four-ULP allowance accounts for arithmetic drift;
        strict neighbour checks preserve the anchor and distinct points.
        """
        lo, hi = self.bounds
        res = self.resolution
        quotient = (hi - lo) / res
        last = math.floor(quotient)
        nearest = round(quotient)
        candidate = lo + nearest * res
        allowance = 4 * max(math.ulp(lo), math.ulp(nearest * res), math.ulp(hi))
        if (self.var_type != 'integer' and nearest > 0
                and abs(candidate - hi) <= allowance
                and lo + (nearest - 1) * res < hi < lo + (nearest + 1) * res):
            return nearest, hi
        while last > 0 and lo + last * res > hi:
            last -= 1
        return last, None

    def _resolution_value(self, index, last, endpoint):
        lo, hi = self.bounds
        if index == 0:
            value = lo
        elif index == last and endpoint is not None:
            value = endpoint
        else:
            value = lo + index * self.resolution
            ndigits = max(0, -int(math.floor(math.log10(self.resolution)))) + 2
            cleaned = round(value, ndigits)
            if (lo <= cleaned <= hi
                    and lo + (index - 1) * self.resolution < cleaned
                    < lo + (index + 1) * self.resolution):
                value = cleaned
        return int(value) if self.var_type == 'integer' else value

    def _discrete_values(self):
        """Enumerate the lower-bound-anchored, bounds-safe resolution domain."""
        last, endpoint = self._resolution_domain()
        return [self._resolution_value(i, last, endpoint) for i in range(last + 1)]

    def _discrete_grid(self, n, *, warning_stacklevel):
        if self.resolution is not None and self.enforce_resolution:
            discrete = self._discrete_values()
            limiting = "bounds and resolution"
        else:
            lo, hi = self.bounds
            discrete = range(math.ceil(lo), math.floor(hi) + 1)
            limiting = "bounds and integer domain"
        if len(discrete) < n:
            warnings.warn(
                f"{self.name}: requested {n} grid points, but the {limiting} "
                f"allow only {len(discrete)} distinct values. Returning {len(discrete)}.",
                UserWarning, stacklevel=warning_stacklevel)
        if len(discrete) <= n:
            return list(discrete)
        if n == 1:
            return [discrete[0]]
        return [discrete[int(i * (len(discrete) - 1) / (n - 1))] for i in range(n)]

    def _project_sample(self, value):
        """Project a draw to the permitted bounds and applicable numeric domain."""
        lo, hi = self.bounds
        value = min(hi, max(lo, value))
        if self.resolution is not None and self.enforce_resolution:
            last, endpoint = self._resolution_domain()
            index = min(last, max(0, round((value - lo) / self.resolution)))
            value = self._resolution_value(index, last, endpoint)
        if self.var_type == 'integer':
            value = min(math.floor(hi), max(math.ceil(lo), int(round(value))))
        return value


def _validate_sampling_method(method):
    if method not in ('random', 'grid', 'lhs', 'halton'):
        hint = ""
        if method == 'log_grid':
            hint = " Use scale='log' with method='grid'."
        elif method == 'sobol':
            hint = " 'sobol' was renamed to 'halton' (the implementation always was Halton)."
        raise ValueError(
            f"Unknown sampling method {method!r}. Choose from: "
            f"'random', 'grid', 'lhs', 'halton'.{hint}"
        )
