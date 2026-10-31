"""Literal finite bit sequences; no simulator or random state required."""
from numbers import Integral

_TAPS = {7: 6, 9: 5, 15: 14, 23: 18, 31: 28}


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be a literal integer, got {value!r}')
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {value!r}')
    return int(value)


def _normalize_bits(bits):
    if isinstance(bits, str):
        if bits and set(bits) <= {'0', '1'}:
            return bits
    elif isinstance(bits, (list, tuple)):
        if bits and all(isinstance(b, Integral) and not isinstance(b, bool)
                        and b in (0, 1) for b in bits):
            return ''.join(str(int(b)) for b in bits)
    raise ValueError('bits must be a nonempty 0/1 string or list/tuple of integer bits')


def _prbs_parameters(order, length, seed):
    order = _integer('order', order, 1)
    if order not in _TAPS:
        raise ValueError(f'order must be one of {tuple(_TAPS)}, got {order}')
    length = _integer('length', length, 1)
    seed = _integer('seed', seed)
    if seed >= (1 << order) - 1:
        raise ValueError(f'seed must be below {(1 << order) - 1}; '
                         'the all-ones XNOR seed locks up')
    return order, length, seed


def prbs_bits(order, length, *, seed=0):
    """Return a finite tuple of Fibonacci XNOR, post-shift output bits.

    Orders 7, 9, 15, 23 and 31 use taps (n, m) with m=6, 5, 14, 18,
    and 28 respectively. Seed bit 0 is Q_1; feedback enters Q_1 and Qn
    is sampled after shifting. The all-ones seed is forbidden. Explicit
    ``length`` bounds allocation; generation continues across periods.

    Parameters
    ----------
    order : int
        Register length ``n``; one of 7, 9, 15, 23 or 31.
    length : int
        Number of bits to return, at least 1.
    seed : int, optional
        Initial register state as an integer, keyword-only, from ``0`` to
        ``2**order - 2``. Default ``0``.

    Returns
    -------
    tuple of int
        ``length`` bits, each ``0`` or ``1``, deterministic for the given
        arguments.

    Raises
    ------
    TypeError
        If *order*, *length* or *seed* is not a literal integer (a ``bool``
        is rejected).
    ValueError
        If *order* is not a supported register length, *length* is below 1,
        *seed* is negative, or *seed* is ``2**order - 1`` or larger (the
        all-ones XNOR state locks up).

    Examples
    --------
    >>> from silk.sources import prbs_bits
    >>> prbs_bits(7, 8)
    (0, 0, 0, 0, 0, 0, 1, 1)
    >>> prbs_bits(7, 8, seed=1)
    (0, 0, 0, 0, 0, 1, 1, 1)
    """
    order, length, state = _prbs_parameters(order, length, seed)
    mask = (1 << order) - 1
    tap = _TAPS[order]
    result = []
    for _ in range(length):
        feedback = 1 ^ ((state >> (order - 1)) & 1) ^ ((state >> (tap - 1)) & 1)
        state = ((state << 1) | feedback) & mask
        result.append((state >> (order - 1)) & 1)
    return tuple(result)
