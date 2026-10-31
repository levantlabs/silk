"""Statistical declarations for top-level behavioural model parameters.

Runs outside Monte Carlo use nominal floats. Draw scheme 1 is a pure function
of a canonical key, independent of simulator seeds and evaluation order.
"""
import hashlib
import json
import math
import numbers
import statistics
from dataclasses import dataclass, field
from typing import Optional

from silk._json_codec import encode_json
from silk.exceptions import VariationError

DRAW_SCHEME = 1
VARIATIONS = ('mismatch', 'process')
_STANDARD_NORMAL = statistics.NormalDist()


def _variable_types():
    from silk.circuit.variable import Variable, _VarExpr
    return (Variable, _VarExpr)


def _is_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _check_type(owner, name, value):
    if not (_is_number(value) or isinstance(value, _variable_types())):
        raise TypeError(f'{owner}.{name} must be a number, a Variable or a '
                        f'Variable expression, got {type(value).__name__}')


def _field_value(owner, name, value):
    if isinstance(value, _variable_types()):
        try:
            value = value.value
        except (ValueError, TypeError, ArithmeticError) as exc:
            raise VariationError(f'{owner}.{name}: {exc}') from exc
        if value is None:
            raise VariationError(f'{owner}.{name}: its Variable has no value assigned')
        if not _is_number(value):
            raise VariationError(f'{owner}.{name}: its Variable holds {value!r}, not a number')
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise VariationError(f'{owner}.{name} must be finite') from exc
    if not math.isfinite(value):
        raise VariationError(f'{owner}.{name} must be finite, got {value!r}')
    return value


def _validate(declaration, fields):
    owner = type(declaration).__name__
    for name in fields:
        _check_type(owner, name, getattr(declaration, name))
    if not isinstance(declaration.variation, str) or declaration.variation not in VARIATIONS:
        raise VariationError(f'{owner}.variation must be one of {VARIATIONS}')
    if declaration.variation == 'process':
        if not isinstance(declaration.name, str) or not declaration.name:
            raise VariationError(f'{owner}.name must be a nonempty string for process variation')
    elif declaration.name is not None:
        raise VariationError(f'{owner}.name is only allowed for process variation')
    if all(_is_number(getattr(declaration, name)) for name in fields):
        declaration.resolve()


@dataclass(frozen=True, eq=False)
class Normal:
    """Normal variation with a finite nominal value and nonnegative sigma."""
    nominal: object
    sigma: object
    variation: str = field(default='mismatch', kw_only=True)
    name: Optional[str] = field(default=None, kw_only=True)

    def __post_init__(self):
        _validate(self, ('nominal', 'sigma'))

    def resolve(self) -> 'Resolved':
        nominal = _field_value('Normal', 'nominal', self.nominal)
        sigma = _field_value('Normal', 'sigma', self.sigma)
        if sigma < 0:
            raise VariationError('Normal.sigma must be nonnegative')
        return Resolved('Normal', (nominal, sigma), self.variation, self.name)


@dataclass(frozen=True, eq=False)
class Uniform:
    """Uniform variation between finite limits; the nominal is their midpoint."""
    low: object
    high: object
    variation: str = field(default='mismatch', kw_only=True)
    name: Optional[str] = field(default=None, kw_only=True)

    def __post_init__(self):
        _validate(self, ('low', 'high'))

    def resolve(self) -> 'Resolved':
        low = _field_value('Uniform', 'low', self.low)
        high = _field_value('Uniform', 'high', self.high)
        if low > high:
            raise VariationError('Uniform.low must not exceed high')
        if not math.isfinite(high - low) or not math.isfinite(low + (high - low) / 2):
            raise VariationError('Uniform.high - low and midpoint must be finite')
        return Resolved('Uniform', (low, high), self.variation, self.name)


@dataclass(frozen=True)
class Resolved:
    """Read-only snapshot of a resolved variation declaration.

    Obtain this record from ``Normal.resolve()`` or ``Uniform.resolve()``.
    Resolution evaluates Variable-backed fields and validates their scalar
    values. Later changes to those Variables do not change this snapshot.

    Parameters
    ----------
    kind : str
        Distribution name, ``'Normal'`` or ``'Uniform'``.
    fields : tuple of float
        Resolved ``(nominal, sigma)`` or ``(low, high)`` values, respectively.
    variation : str
        Variation scope, ``'mismatch'`` or ``'process'``.
    name : str or None
        Shared process-variation name, or None for mismatch variation.

    Notes
    -----
    Attributes cannot be reassigned. Direct construction does not validate
    the fields; use a declaration's ``resolve()`` method for validated data.

    Examples
    --------
    >>> resolved = Uniform(1.0, 3.0).resolve()
    >>> resolved.nominal
    2.0
    >>> resolved.at(0.25)
    1.5
    """
    kind: str
    """str: Distribution name, either ``'Normal'`` or ``'Uniform'``."""
    fields: tuple
    """tuple of float: Resolved ``(nominal, sigma)`` or ``(low, high)``."""
    variation: str
    """str: Variation scope, either ``'mismatch'`` or ``'process'``."""
    name: Optional[str]
    """str or None: Shared process name, or None for mismatch variation."""

    @property
    def nominal(self) -> float:
        """Return the value used outside Monte Carlo sampling.

        Returns
        -------
        float
            Normal nominal value or the midpoint of the Uniform limits.
        """
        a, b = self.fields
        return a if self.kind == 'Normal' else a + (b - a) / 2

    @property
    def has_spread(self) -> bool:
        """Return whether the distribution has nonzero spread.

        Returns
        -------
        bool
            True for positive Normal sigma or distinct Uniform limits.
        """
        a, b = self.fields
        return b > 0 if self.kind == 'Normal' else a < b

    def at(self, u: float) -> float:
        """Evaluate the distribution at an open-unit-interval quantile.

        Parameters
        ----------
        u : float
            Quantile strictly between zero and one.

        Returns
        -------
        float
            Normal inverse-CDF value or a Uniform interpolation clamped to
            its limits against floating-point rounding.

        Raises
        ------
        silk.exceptions.VariationError
            If the quantile is outside the open interval or the drawn value
            is not finite.
        """
        a, b = self.fields
        if not 0 < u < 1:
            raise VariationError('draw u must be in the open interval (0, 1)')
        value = (a + b * _STANDARD_NORMAL.inv_cdf(u) if self.kind == 'Normal'
                 else min(max(a + (b - a) * u, a), b))
        if not math.isfinite(value):
            raise VariationError(f'{self.kind} drawn value must be finite')
        return value


def is_declaration(value) -> bool:
    return isinstance(value, (Normal, Uniform))


def u_from_word(word: int) -> float:
    """Map an unsigned 64-bit word strictly inside (0, 1), using 52 bits."""
    return ((word >> 12) + 0.5) / 2.0 ** 52


def draw_u(key: list) -> float:
    encoded = json.dumps(encode_json(key), ensure_ascii=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return u_from_word(int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'big'))


def mismatch_key(base_seed: int, index: int, segments, param: str) -> list:
    return ['silk.variation', DRAW_SCHEME, base_seed, index, 'mismatch', list(segments), param]


def bench_key(base_seed: int, index: int, owner_path: str, test_name, segments, param: str) -> list:
    return ['silk.variation', DRAW_SCHEME, base_seed, index, 'bench',
            owner_path, test_name, list(segments), param]


def process_key(base_seed: int, index: int, name: str) -> list:
    return ['silk.variation', DRAW_SCHEME, base_seed, index, 'process', name]


def owner_segments(owner_path: str) -> tuple:
    """Decode the escaped instance names in a spec owner path."""
    return tuple(part.replace('~1', '/').replace('~0', '~')
                 for part in owner_path.split('/')[1:])
