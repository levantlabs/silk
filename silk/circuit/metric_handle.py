"""Deferred arithmetic over a testbench's reported metric values."""

import math
import operator
from numbers import Number


_OPERATORS = {
    '+': operator.add, '-': operator.sub, '*': operator.mul,
    '/': operator.truediv, '**': operator.pow, 'abs': abs, 'neg': operator.neg,
}


def _bench_identity(bench):
    """Validate a selected receiver and identify its exact placement."""
    from silk.circuit.test_placement import SelectedTestbench
    if isinstance(bench, SelectedTestbench):
        entry = bench._entry()
        return entry._root_ref(), entry.owner_path, bench._authored
    return None, None, bench


def _evaluate_expression(expression, values):
    """Read measured leaves; never invoke their extractors again."""
    op, *args = expression
    if op == 'metric':
        value = values[args[0]]
        if isinstance(value, str):
            raise ValueError(f"input metric {args[0]!r} failed: {value}")
        return value
    if op == 'number':
        return args[0]
    operands = [_evaluate_expression(arg, values) for arg in args]
    # Even nan ** 0 and 0 ** nan represent an unknown measurement here.
    if any(isinstance(value, Number) and value != value for value in operands):
        return math.nan
    # NumPy scalar division need not raise on zero; check before dividing.
    if (op == '/' and all(isinstance(value, Number) for value in operands)
            and operands[1] == 0):
        return math.nan
    try:
        return _OPERATORS[op](*operands)
    except ZeroDivisionError:
        # Degenerate data returns a number (decision-1): a zero denominator
        # measured from the circuit is not a caller error.
        return math.nan


class MetricHandle:
    """A reported metric or arithmetic expression owned by one testbench.

    ``value`` reads the latest completed evaluation. Arithmetic declares an
    expression; register it with ``tb.add_metric(name, expression)`` to report
    it and check specs. Handles retain identity equality and hashing.

    Parameters
    ----------
    bench : ModuleTestbench
        The testbench whose evaluated metric values the handle reads.
    expression : tuple
        Internal expression tree; handles are created by
        ``ModuleTestbench.add_metric`` and by arithmetic on existing handles,
        not constructed directly.
    """

    def __init__(self, bench, expression):
        self._bench = bench
        self._expression = expression

    @property
    def value(self):
        """The metric's latest evaluated value.

        Returns
        -------
        float or numpy.ndarray
            The value from the testbench's most recent completed run, or
            the arithmetic combination of such values for an expression.

        Raises
        ------
        RuntimeError
            If the testbench has not produced the underlying metric yet
            (run it first, then read ``.value``).
        """
        values = getattr(self._bench, '_metric_values', {})
        try:
            return _evaluate_expression(self._expression, values)
        except KeyError as exc:
            raise RuntimeError(
                'Metric value is unavailable; run the testbench first, then read .value.'
            ) from exc

    @property
    def name(self):
        """The name this metric was registered under with ``add_metric``.

        An unregistered arithmetic expression has no name; reading it raises
        ``ValueError`` (not ``AttributeError``, which ``__getattr__`` would
        mask) telling you to register it.
        """
        if self._expression[0] == 'metric':
            return self._expression[1]
        raise ValueError(
            'This metric expression has no name: register it with '
            "tb.add_metric('<name>', expression) and use the handle that "
            'returns.')

    def __bool__(self):
        raise TypeError(
            'Metric value exists only after evaluation; run the testbench first '
            'and read .value.'
        )

    def __getattr__(self, name):
        # Do not forward pickle/deepcopy hooks or expose bench internals.
        if not name.startswith('_'):
            bench = self.__dict__.get('_bench')
            method = getattr(bench, name, None)
            if callable(method):
                return method
        raise AttributeError(name)

    def _binary(self, other, op, reverse=False):
        owner = _bench_identity(self._bench)
        if isinstance(other, MetricHandle):
            other_owner = _bench_identity(other._bench)
            if (other_owner[0] is not owner[0] or other_owner[1] != owner[1]
                    or other_owner[2] is not owner[2]):
                raise ValueError('Metric arithmetic requires handles from the same testbench.')
            other_expression = other._expression
        elif isinstance(other, Number):
            other_expression = ('number', other)
        else:
            return NotImplemented
        args = (other_expression, self._expression) if reverse else (
            self._expression, other_expression)
        return MetricHandle(self._bench, (op, *args))

    def _belongs_to(self, bench):
        """Allow shared authored recipes or this exact selected registration."""
        from silk.circuit.test_placement import current_placement
        root, path, authored = _bench_identity(self._bench)
        if root is None:
            return authored is bench
        selection = current_placement(bench)
        return (selection is not None and selection.entry is not None
                and selection.entry._root_ref() is root
                and selection.owner_path == path and authored is bench)

    def __add__(self, other):
        return self._binary(other, '+')

    def __radd__(self, other):
        return self._binary(other, '+', True)

    def __sub__(self, other):
        return self._binary(other, '-')

    def __rsub__(self, other):
        return self._binary(other, '-', True)

    def __mul__(self, other):
        return self._binary(other, '*')

    def __rmul__(self, other):
        return self._binary(other, '*', True)

    def __truediv__(self, other):
        return self._binary(other, '/')

    def __rtruediv__(self, other):
        return self._binary(other, '/', True)

    def __pow__(self, other):
        return self._binary(other, '**')

    def __rpow__(self, other):
        return self._binary(other, '**', True)

    def __abs__(self):
        _bench_identity(self._bench)
        return MetricHandle(self._bench, ('abs', self._expression))

    def __neg__(self):
        _bench_identity(self._bench)
        return MetricHandle(self._bench, ('neg', self._expression))


def _metric_key(metric):
    """A metric name string, from a name or a registered ``MetricHandle``."""
    if isinstance(metric, MetricHandle):
        return metric.name
    return metric


class MetricValues(dict):
    """``SpecReport.metrics``: a dict keyed by name that also takes a handle."""

    def __getitem__(self, key):
        return super().__getitem__(_metric_key(key))

    def get(self, key, default=None):
        return super().get(_metric_key(key), default)

    def __contains__(self, key):
        return super().__contains__(_metric_key(key))
