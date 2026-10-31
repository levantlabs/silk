"""Selection by declared circuit indices, independent of sequence positions."""
from numbers import Integral
from silk.exceptions import CircuitError


def select_positions(indices, key):
    """Return positions selected by a declared number or numerical slice."""
    indices = tuple(indices)
    if isinstance(key, Integral):
        try:
            return (indices.index(key),)
        except ValueError:
            raise CircuitError(f'Index {key!r} is absent; declared indices: {indices}.') from None
    if not isinstance(key, slice):
        raise TypeError('Circuit aggregate subscript must be an integer or slice.')
    for part in (key.start, key.stop, key.step):
        if part is not None and not isinstance(part, Integral):
            raise TypeError('Declared slice bounds and step must be integers.')
    step = 1 if key.step is None else key.step
    if step == 0:
        raise ValueError('Declared slice step cannot be zero.')
    positions = tuple(range(len(indices)))
    if step < 0:
        positions = positions[::-1]
    anchor = key.start if key.start is not None else indices[positions[0]] if positions else 0
    chosen = tuple(p for p in positions
                   if (key.start is None or (indices[p] >= key.start if step > 0 else indices[p] <= key.start))
                   and (key.stop is None or (indices[p] < key.stop if step > 0 else indices[p] > key.stop))
                   and (indices[p] - anchor) % abs(step) == 0)
    if not chosen:
        raise CircuitError(f'Slice {key!r} selects no declared indices; declared indices: {indices}.')
    return chosen
