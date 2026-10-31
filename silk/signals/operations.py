"""silk.signals.operations — the fixed list of derived-signal operations.

One vocabulary for waveform math: the testbench codegen spec's
``derived_signals``, the waveform explorer's trace math and silk's Python
:class:`~silk.signals.base.Signal` operations all use these names with these
definitions. There is no expression language and no formula parser — a
derived signal names one operation from this list, its inputs and the
operation's parameters.

Every operation maps to plain public silk Python on Signals: operator
arithmetic for the multi-input operations, ``gain * a + offset`` for
``scale``, and the Signal methods :meth:`~silk.signals.base.Signal.mag`,
:meth:`~silk.signals.base.Signal.derivative` (central differences) and
:meth:`~silk.signals.base.Signal.integral` (cumulative trapezoid with
``y[0] = 0``). All of them defer on unbound probes, so a derived signal built
in a testbench is recomputed from the probes on every run.

The explorer consumes this list as its operation contract: names, input
counts and parameters must match the available Python operations.
"""
import copy

# Each entry: name, min/max input count (max None = no upper limit), whether
# input order matters, parameters, the silk Python it maps to, description.
# 'python' is the display form published by export_signal_operations();
# operation_expression() below renders the same thing for given operands.
_OPERATIONS = (
    {'name': 'sum', 'min_inputs': 2, 'max_inputs': None, 'ordered': False,
     'parameters': [], 'python': 'a + b (+ ...)',
     'description': 'Sum of two or more signals.'},
    {'name': 'difference', 'min_inputs': 2, 'max_inputs': 2, 'ordered': True,
     'parameters': [], 'python': 'a - b',
     'description': 'First input minus second input.'},
    {'name': 'product', 'min_inputs': 2, 'max_inputs': None, 'ordered': False,
     'parameters': [], 'python': 'a * b (* ...)',
     'description': 'Product of two or more signals.'},
    {'name': 'quotient', 'min_inputs': 2, 'max_inputs': 2, 'ordered': True,
     'parameters': [], 'python': 'a / b',
     'description': 'First input divided by second input.'},
    {'name': 'mean', 'min_inputs': 2, 'max_inputs': None, 'ordered': False,
     'parameters': [], 'python': '(a + b + ...) / n',
     'description': 'Point-wise mean of two or more signals.'},
    {'name': 'scale', 'min_inputs': 1, 'max_inputs': 1, 'ordered': True,
     'parameters': [
         {'name': 'gain', 'type': 'float', 'default': 1.0, 'required': False},
         {'name': 'offset', 'type': 'float', 'default': 0.0, 'required': False},
     ],
     'python': 'gain * a + offset',
     'description': 'Scale by gain, then add offset.'},
    {'name': 'abs', 'min_inputs': 1, 'max_inputs': 1, 'ordered': True,
     'parameters': [], 'python': 'a.mag()',
     'description': 'Magnitude |y| (Signal.mag).'},
    {'name': 'derivative', 'min_inputs': 1, 'max_inputs': 1, 'ordered': True,
     'parameters': [], 'python': 'a.derivative()',
     'description': 'dy/dx by central differences (Signal.derivative).'},
    {'name': 'integral', 'min_inputs': 1, 'max_inputs': 1, 'ordered': True,
     'parameters': [], 'python': 'a.integral()',
     'description': 'Cumulative trapezoidal integral with y[0] = 0 '
                    '(Signal.integral).'},
)


def export_signal_operations() -> list:
    """Return the derived-signal operation list as JSON-safe data.

    One dict per operation, in a fixed order::

        {'name': 'scale', 'min_inputs': 1, 'max_inputs': 1, 'ordered': True,
         'parameters': [{'name': 'gain', 'type': 'float', 'default': 1.0,
                         'required': False}, ...],
         'python': 'gain * a + offset',
         'description': 'Scale by gain, then add offset.'}

    ``max_inputs`` is ``None`` (JSON ``null``) when the operation takes any
    number of inputs from ``min_inputs`` up. ``ordered`` says whether
    swapping inputs changes the result. ``python`` is the silk Python the
    operation maps to, written over inputs ``a``, ``b``, …

    The testbench codegen validates a spec's ``derived_signals`` against this
    same list. The returned list is a copy; mutating it changes nothing.
    """
    return copy.deepcopy(list(_OPERATIONS))


def signal_operation(name: str) -> dict:
    """The entry for operation *name*, or ``KeyError`` naming the known ones."""
    for op in _OPERATIONS:
        if op['name'] == name:
            return copy.deepcopy(op)
    raise KeyError(
        f"unknown signal operation {name!r}; known: "
        f"{[op['name'] for op in _OPERATIONS]}")


def operation_expression(name: str, operands: list, params: dict) -> str:
    """Render operation *name* as silk Python source.

    *operands* are Python expressions (already source text) for the inputs,
    in order; *params* the operation's parameter values with defaults filled
    in. Validation is the caller's job: this assumes a known operation, a
    legal operand count and every parameter present.
    """
    if name == 'sum':
        return ' + '.join(operands)
    if name == 'difference':
        return f"{operands[0]} - {operands[1]}"
    if name == 'product':
        return ' * '.join(operands)
    if name == 'quotient':
        return f"{operands[0]} / {operands[1]}"
    if name == 'mean':
        return f"({' + '.join(operands)}) / {len(operands)}"
    if name == 'scale':
        return f"{params['gain']!r} * {operands[0]} + {params['offset']!r}"
    if name == 'abs':
        return f"{operands[0]}.mag()"
    if name == 'derivative':
        return f"{operands[0]}.derivative()"
    if name == 'integral':
        return f"{operands[0]}.integral()"
    raise KeyError(f"unknown signal operation {name!r}")
