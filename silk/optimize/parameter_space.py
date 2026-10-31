"""Joint parameter space over multiple Variable objects.

VariableSpace collects N Variable objects and operates on them
jointly: generating correlated design points, interfacing with external
optimizers, and applying suggested points back to the variables.

The per-variable Variable.sample() method handles 1-D sampling;
VariableSpace handles the multi-dimensional case where stratification
and quasi-random sequences must be coordinated across variables.
"""

import math
import itertools
from typing import Dict, List, Optional, Tuple, Union

from silk.circuit.variable import _sampling_rng, _validate_sampling_method
from silk.circuit.variable import Variable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_n_primes(n: int) -> List[int]:
    """Return the first n prime numbers (used as Halton sequence bases)."""
    primes = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return primes


def _halton(k: int, base: int) -> float:
    """k-th element of the Halton sequence with the given prime base."""
    r, denom = 0.0, 1.0
    while k:
        denom *= base
        k, remainder = divmod(k, base)
        r += remainder / denom
    return r


# ---------------------------------------------------------------------------
# VariableSpace
# ---------------------------------------------------------------------------

class VariableSpace:
    """Joint parameter space over N :class:`Variable <silk.circuit.Variable>` objects.

    Parameters
    ----------
    variables : list of Variable or dict {name: Variable}
        The variables that define the space.  When a list is passed the
        variable's ``.name`` attribute is used as the key; duplicate names
        raise ``ValueError``.
    groups : list of list of Variable, optional
        Correlation groups.  Variables in the same group share the same
        stratum permutation in LHS and the same sequence in Halton — they
        advance together through the design space rather than being sampled
        independently.  Multiple groups are supported; a variable not listed
        in any group is treated as its own independent group. Grid sampling
        steps grouped columns together by index; their actual lengths must
        match. Groups and ungrouped variables cross in first-member declaration
        order. Random sampling remains independent.

        Example — ``w_n`` and ``w_p`` correlated, ``l`` independent::

            space = VariableSpace([w_n, w_p, l], groups=[[w_n, w_p]])

        Multiple groups::

            space = VariableSpace([w_n, w_p, l, vdd],
                                   groups=[[w_n, w_p], [l, vdd]])

    Examples
    --------
    >>> from silk import Variable, VariableSpace
    >>> width = Variable('w', value=2.0, bounds=(1.0, 4.0))
    >>> space = VariableSpace([width])
    >>> space.apply({'w': 3.0})
    >>> width.value
    3.0
    """

    def __init__(
        self,
        variables: Union[List[Variable], Dict[str, Variable]],
        groups: Optional[List[List[Variable]]] = None,
    ) -> None:
        if isinstance(variables, dict):
            self._vars: Dict[str, Variable] = dict(variables)
        elif isinstance(variables, list):
            seen: set = set()
            self._vars = {}
            for v in variables:
                if v.name in seen:
                    raise ValueError(
                        f"Duplicate variable name '{v.name}' in VariableSpace"
                    )
                seen.add(v.name)
                self._vars[v.name] = v
        else:
            raise TypeError("variables must be a list or dict of Variable")

        if len(self._vars) == 0:
            raise ValueError("VariableSpace requires at least one variable")

        # Build group map: var_name -> group_id
        # Variables in the same group share sampling permutations.
        self._group_map: Dict[str, int] = {}
        next_group = 0
        if groups:
            for group in groups:
                for var in group:
                    if var.name not in self._vars:
                        raise ValueError(
                            f"Variable '{var.name}' in groups= is not in this "
                            f"VariableSpace. Add it to the variables list first."
                        )
                    if var.name in self._group_map:
                        raise ValueError(
                            f"Variable '{var.name}' appears in more than one group."
                        )
                    self._group_map[var.name] = next_group
                next_group += 1
        # Ungrouped variables each get their own independent group id
        for name in self._vars:
            if name not in self._group_map:
                self._group_map[name] = next_group
                next_group += 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def names(self) -> List[str]:
        """Variable names in insertion order."""
        return list(self._vars.keys())

    def variables(self) -> Dict[str, Variable]:
        """Return a new ``{space key: Variable}`` dict in :attr:`names` order.

        The values are the same ``Variable`` objects the space holds (by
        identity), so they can be passed to ``Optimizer.add_objective`` /
        ``add_constraint``. A space key can differ from ``Variable.name``.
        Mutating the returned dict does not change the space.
        """
        return dict(self._vars)

    @property
    def size(self) -> int:
        """Number of variables in the space."""
        return len(self._vars)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"VariableSpace({self.names})"

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n: int, method: str = 'random', *, seed: Optional[int] = 0) -> List[Dict[str, object]]:
        """Generate *n* joint design points.

        Parameters
        ----------
        n : int
            Number of design points.  For ``method='grid'`` this is the
            requested number of points **per numeric variable**. Total output
            is the product of actual lengths of groups and ungrouped columns,
            including categorical choices; finite linear domains may return
            fewer than n values. Grouped columns must have equal lengths.
        method : str
            ``'random'``  — independent random draws per variable.

            ``'lhs'``     — Latin Hypercube Sampling; each variable is
            independently stratified into *n* equal-width strata, with one
            sample per stratum.  Strata are shuffled independently so
            variables are not rank-correlated.

            ``'halton'``  — Halton low-discrepancy sequence; variable *d*
            uses the *d*-th prime as its base.  Provides better space
            coverage than random for small *n*. Categorical variables map
            floor(u * category_count) to indices; finite prefixes need not
            cover every category. Use grid for exhaustive coverage.

            ``'grid'``    — Cross-product of groups and ungrouped Variable grids.
            Grouped variables step together through equal-length columns.
            Axes follow the first member's declaration order; output dicts
            retain variable declaration order. Unequal lengths raise ValueError
            naming each variable and count in the mismatched group.
            Shortened linear numeric columns emit one UserWarning each at the
            public caller. Log columns retain n draws and may repeat.
            Each categorical contributes all categories in declared order
            for n > 0. **Use with caution for D > 3.**

        seed : int or None
            Reproducible seed (default 0). None uses fresh OS entropy.
            One generator is shared by all variables and groups. Booleans
            and non-integers raise TypeError.

        Returns
        -------
        list of dict
            Each dict maps variable names to sampled values.
        """
        return self._sample(n, method, seed=seed)

    def _sample(self, n, method='random', *, seed=0, warning_stacklevel=3):
        """Joint sampling with caller depth carried only on private helpers."""
        rng = _sampling_rng(seed)
        _validate_sampling_method(method)
        if n == 0:
            return []
        self._require_sampling_bounds()
        if method == 'lhs':
            return self._sample_lhs(n, rng)
        if method == 'halton':
            return self._sample_halton(n, rng)
        if method == 'grid':
            return self._sample_grid(n, rng, warning_stacklevel=warning_stacklevel + 1)
        if method == 'random':
            return self._sample_random(n, rng)
    # ------------------------------------------------------------------
    # Internal sampling methods
    # ------------------------------------------------------------------

    def _require_sampling_bounds(self):
        """Fail loud on variables whose bounds cannot be sampled.

        A None end is legitimate variable state — PDK catalogs often declare
        no upper limit, so ``pdk.nmos1p8.w.variable()`` derives e.g.
        ``(0.42, None)`` — but sampling needs a finite interval, and without
        this check the None reached the scaling arithmetic as a raw
        TypeError. Categorical variables are exempt (they sample without an interval).
        """
        bad = []
        for name, var in self._vars.items():
            var._validate_integer_grid(var.var_type, var.resolution, var.bounds,
                                       sampling=True)
            var._require_log_bounds()
            if var.var_type == 'categorical':
                continue
            b = var.bounds
            if b is None or b[0] is None or b[1] is None:
                bad.append(f"'{name}' (bounds={b})")
        if bad:
            raise ValueError(
                "Cannot sample — variable(s) with unbounded end(s): "
                + ", ".join(bad)
                + ". Sampling needs finite (min, max); PDK catalogs often "
                "set no upper limit, so pass bounds explicitly, e.g. "
                "pdk.nmos1p8.w.variable(name='w_n', value=1.0, "
                "bounds=(0.42, 20.0))."
            )

    def _discrete_values(self, var) -> List[float]:
        """Enumerate all valid grid points for a resolution-constrained variable.

        Returns a list ``[lo, lo+res, lo+2*res, ...]`` clipped to ``[lo, hi]``.
        This is the actual realizable set the sampler should draw from.
        """
        return var._discrete_values()

    def _group_permutations(self, n: int, rng) -> Dict[int, List[int]]:
        """Return one shuffled permutation of [0..n-1] per group id.

        Variables sharing a group id will receive the same permutation,
        making them advance through strata in lockstep.
        """
        group_ids = sorted(set(self._group_map.values()))
        return {gid: rng.permutation(n).tolist() for gid in group_ids}

    def _sample_random(self, n: int, rng) -> List[Dict[str, object]]:
        """Random sampling — grouped variables share the same draw order."""
        # For random, sharing the permutation means applying the same index
        # shuffle so grouped variables are drawn from matching positions.
        perms = self._group_permutations(n, rng)
        cols: Dict[str, list] = {}
        for name, var in self._vars.items():
            gid = self._group_map[name]
            perm = perms[gid]
            if var.scale == 'linear' and var.resolution is not None and var.enforce_resolution and var.bounds is not None:
                discrete = self._discrete_values(var)
                full = [discrete[int(rng.integers(len(discrete)))] for _ in range(n)]
                cols[name] = [full[perm[i]] for i in range(n)]
            else:
                full = var._sample(n, 'random', rng)
                cols[name] = [full[perm[i]] for i in range(n)]
        return [{name: cols[name][i] for name in self.names} for i in range(n)]

    def _sample_lhs(self, n: int, rng) -> List[Dict[str, object]]:
        """Latin Hypercube Sampling — stratifies over the discrete grid when
        resolution is set.  Variables in the same group share the same stratum
        permutation so they advance through the design space in lockstep."""
        perms = self._group_permutations(n, rng)
        cols: Dict[str, list] = {}
        for name, var in self._vars.items():
            if var.var_type == 'categorical':
                cols[name] = var._sample(n, 'random', rng)
                continue
            if var.bounds is None:
                raise ValueError(
                    f"Variable '{name}' has no bounds. "
                    "Set bounds=(min, max) before sampling."
                )
            strata = perms[self._group_map[name]]
            if var.scale == 'linear' and var.resolution is not None and var.enforce_resolution:
                discrete = self._discrete_values(var)
                nd = len(discrete)
                col = []
                for s in strata:
                    idx = int((s + rng.random()) / n * nd)
                    col.append(discrete[min(idx, nd - 1)])
                cols[name] = col
            else:
                col = []
                for s in strata:
                    u = (s + rng.random()) / n
                    col.append(self._scale_to_var(u, var))
                cols[name] = col

        return [{name: cols[name][i] for name in self.names} for i in range(n)]

    def _sample_halton(self, n: int, rng) -> List[Dict[str, object]]:
        """Halton low-discrepancy sequence — variables in the same group use
        the same prime base so they follow the same sequence."""
        # Assign one prime per group (not per variable)
        group_ids = sorted(set(self._group_map.values()))
        n_groups = len(group_ids)
        primes = _first_n_primes(n_groups)
        group_prime = {gid: primes[i] for i, gid in enumerate(group_ids)}

        cols: Dict[str, list] = {}
        for name, var in self._vars.items():
            if var.var_type == 'categorical':
                if not var.categories:
                    raise ValueError(f"Variable '{var.name}' has no categories defined")
                base = group_prime[self._group_map[name]]
                count = len(var.categories)
                cols[name] = [var.categories[min(int(_halton(k + 1, base) * count), count - 1)]
                              for k in range(n)]
                continue
            if var.bounds is None:
                raise ValueError(
                    f"Variable '{name}' has no bounds. "
                    "Set bounds=(min, max) before sampling."
                )
            base = group_prime[self._group_map[name]]
            if var.scale == 'linear' and var.resolution is not None and var.enforce_resolution:
                discrete = self._discrete_values(var)
                nd = len(discrete)
                col = [discrete[min(int(_halton(k + 1, base) * nd), nd - 1)]
                       for k in range(n)]
            else:
                col = [self._scale_to_var(_halton(k + 1, base), var) for k in range(n)]
            cols[name] = col

        return [{name: cols[name][i] for name in self.names} for i in range(n)]

    def _sample_grid(self, n: int, rng, *, warning_stacklevel) -> List[Dict[str, object]]:
        """Cross independent axes, stepping grouped grid columns together."""
        per_var = {}
        groups = {}
        for name, var in self._vars.items():
            per_var[name] = var._sample(
                n, 'grid', rng, warning_stacklevel=warning_stacklevel + 1)
            groups.setdefault(self._group_map[name], []).append(name)

        axes = []
        for names in groups.values():
            counts = [len(per_var[name]) for name in names]
            if len(set(counts)) != 1:
                detail = ', '.join(f"'{name}': {count}" for name, count in zip(names, counts))
                raise ValueError(
                    f"Grouped grid variables must have equal column lengths; got {detail}"
                )
            axes.append([
                dict(zip(names, values))
                for values in zip(*(per_var[name] for name in names))
            ])

        points = []
        for combo in itertools.product(*axes):
            values = {name: value for axis in combo for name, value in axis.items()}
            points.append({name: values[name] for name in self.names})
        return points

    # ------------------------------------------------------------------
    # Value scaling (uniform [0,1] → variable domain)
    # ------------------------------------------------------------------

    def _scale_to_var(self, u: float, var: Variable):
        """Map a uniform [0,1] sample *u* to the variable's domain."""
        lo, hi = var.bounds
        if var.scale == 'log':
            log_lo, log_hi = math.log(lo), math.log(hi)
            val = math.exp(log_lo + u * (log_hi - log_lo))
        else:
            val = lo + u * (hi - lo)

        return var._project_sample(val)

    # ------------------------------------------------------------------
    # Apply a point to variables
    # ------------------------------------------------------------------

    def apply(self, point: Dict[str, object]) -> None:
        """Set each variable's value from *point*.

        Points reach ``apply()`` from OUTSIDE :meth:`sample` — external
        optimizers propose out-of-bounds candidates routinely (that's how
        scipy-style algorithms explore), hand-written sweep dicts get
        typo'd, points loaded from a file may be stale. Before this method
        validated, a bad value's failure surfaced far away (at netlist
        emission) or nowhere at all (behavioral-model variables that never
        touch a netlist). So every value in *point* is validated against
        its variable BEFORE any value is assigned — an invalid point never
        partially mutates the space. Values that are within bounds but off
        the resolution grid are not rejected here; they are legal input and
        the value setter's ``_snap()`` handles them.

        Parameters
        ----------
        point : dict
            Mapping of ``{variable_name: value}``.

        Raises
        ------
        KeyError
            If a key in *point* is not a variable in this space.
        ValueError
            If any value fails its variable's :meth:`Variable.validate <silk.circuit.Variable.validate>`
            — the message names the variable, the offending value, and its
            bounds (or categories).
        """
        for name, val in point.items():
            if name not in self._vars:
                raise KeyError(
                    f"Unknown variable '{name}' — not in this VariableSpace"
                )
            var = self._vars[name]
            if not var.validate(val):
                if var.var_type == 'categorical':
                    constraint = f"categories={var.categories}"
                else:
                    constraint = f"bounds={var.bounds}"
                raise ValueError(
                    f"Invalid value for variable '{name}': {val!r} is not "
                    f"valid ({constraint})"
                )
        for name, val in point.items():
            self._vars[name]._assign(val, warn_on_snap=False)

    # ------------------------------------------------------------------
    # Optimizer interface
    # ------------------------------------------------------------------

    def to_bounds(self) -> List[Tuple[float, float]]:
        """Return ``[(lo, hi), ...]`` in variable name order.

        For CATEGORICAL variables returns ``(0, len(categories) - 1)`` as
        integer index bounds — a common convention for mixed-integer
        optimizers.  The caller is responsible for encoding/decoding
        category indices.

        Raises
        ------
        ValueError
            If any non-categorical variable has no bounds set.
        """
        result = []
        for name, var in self._vars.items():
            if var.var_type == 'categorical':
                result.append((0, len(var.categories) - 1))
            elif var.bounds is None:
                raise ValueError(
                    f"Variable '{name}' has no bounds. "
                    "Set bounds=(min, max) before calling to_bounds()."
                )
            else:
                result.append(var.bounds)
        return result

    def to_list(self, point: Dict[str, object]) -> List[object]:
        """Convert a ``{name: value}`` dict to an ordered flat list.

        Order matches :attr:`names`.
        """
        return [point[name] for name in self.names]

    def from_list(self, values: List[object]) -> Dict[str, object]:
        """Convert an ordered flat list back to a ``{name: value}`` dict.

        Parameters
        ----------
        values : list
            Must have the same length as :attr:`size`.

        Raises
        ------
        ValueError
            If length of *values* does not match the number of variables.
        """
        if len(values) != self.size:
            raise ValueError(
                f"Expected {self.size} values, got {len(values)}"
            )
        return dict(zip(self.names, values))

    # ------------------------------------------------------------------
    # Validation and clipping
    # ------------------------------------------------------------------

    def validate(self, point: Dict[str, object]) -> bool:
        """Return ``True`` if every value in *point* is valid for its variable."""
        return all(
            self._vars[name].validate(val)
            for name, val in point.items()
            if name in self._vars
        )

    def clip(self, point: Dict[str, object]) -> Dict[str, object]:
        """Return a new dict with each value clipped to its variable's bounds.

        Does not mutate *point*.
        """
        return {
            name: self._vars[name].clip(val) if name in self._vars else val
            for name, val in point.items()
        }
