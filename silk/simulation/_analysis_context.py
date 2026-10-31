"""The analysis currently dispatched by the behavioral engine.

The root export silk.analysis() reports the live run context, restored on
return or failure. It is never saved as model state or behavioral identity.
"""

from contextlib import contextmanager
from contextvars import ContextVar

from silk.exceptions import SimulationError

_KINDS = ('transient', 'dc')
_current = ContextVar('silk_behavioral_analysis', default=None)


def analysis():
    """Return ``"transient"`` or ``"dc"``: the analysis being dispatched.

    Call it from a behavioural model, or from a helper the model calls::

        from silk import analysis

        def amp(ports, t, params):
            if analysis() == "dc":
                ...

    Raises
    ------
    SimulationError
        Outside a behavioral run: at construction, after a run,
        in post-run metrics and probes, or when a model function is called
        directly. It never returns the most recent run's analysis.
    """
    value = _current.get()
    if value is None:
        raise SimulationError(
            "silk.analysis() is only defined during a behavioral run; call it "
            "from a behavioural model or a helper during the run.")
    return value


@contextmanager
def analysis_scope(kind):
    """Install *kind* for the calls inside the block; restore the caller's value on exit."""
    if kind not in _KINDS:
        raise ValueError(
            f"analysis kind must be 'transient' or 'dc', got {kind!r}")
    token = _current.set(kind)
    try:
        yield kind
    finally:
        _current.reset(token)
