"""TASK-215.1: UnboundSignalError, MetricBindingError, SignalKindError.

Three new exception classes land ahead of the rest of TASK-215 (unbound
Signal state, ``tb.probe()``, the function form of ``add_metric()``):

- ``UnboundSignalError`` (silk.exceptions) — NOT an AttributeError
  subclass, so a defensive ``getattr(sig, 'y', None)`` does not swallow it.
- ``MetricBindingError(ConfigurationError)`` (silk.exceptions).
- ``SignalKindError(ValueError)`` (silk.signals.kinds), raised by the kind
  guard (``_check_kind``) in place of a plain ``ValueError``.

"""
import pytest

from silk.exceptions import (
    CircuitError,
    ConfigurationError,
    MetricBindingError,
    UnboundSignalError,
)
from silk.signals.kinds import (
    SignalKindError,
    TRANSIENT,
    _check_kind,
)


# ---------------------------------------------------------------------------
# Class identity and bases
# ---------------------------------------------------------------------------

def test_unbound_signal_error_is_a_circuit_error():
    assert issubclass(UnboundSignalError, CircuitError)


def test_unbound_signal_error_is_not_an_attribute_error():
    # The load-bearing property: kinds.py and lint/rules/stress.py both use
    # getattr(sig, <attr>, None), which would silently swallow an
    # AttributeError. UnboundSignalError must survive that.
    assert not issubclass(UnboundSignalError, AttributeError)


def test_metric_binding_error_is_a_configuration_error():
    assert issubclass(MetricBindingError, ConfigurationError)
    assert issubclass(MetricBindingError, CircuitError)


def test_signal_kind_error_is_a_value_error():
    assert issubclass(SignalKindError, ValueError)


def test_all_three_are_raisable_and_catchable():
    with pytest.raises(UnboundSignalError):
        raise UnboundSignalError("probe 'vp': bench 'cm' has not run yet")
    with pytest.raises(MetricBindingError):
        raise MetricBindingError("metric 'p_avg': binding failed")
    with pytest.raises(SignalKindError):
        raise SignalKindError("metric: sig must be a transient Signal")


def test_exported_from_silk_exceptions():
    import silk.exceptions as ex
    assert "UnboundSignalError" in ex.__all__
    assert "MetricBindingError" in ex.__all__


def test_exported_from_silk_signals_kinds():
    import silk.signals.kinds as kinds
    assert "SignalKindError" in kinds.__all__


# ---------------------------------------------------------------------------
# The kind guard raises SignalKindError (and existing ValueError handlers
# keep working, since SignalKindError IS a ValueError)
# ---------------------------------------------------------------------------

class _FakeSignal:
    """A minimal stand-in with the attributes _check_kind's predicates read
    (xlabel, y) — TRANSIENT's predicate conflicts with a frequency-domain
    xlabel."""

    def __init__(self, xlabel, y):
        self.xlabel = xlabel
        self.y = y


def test_check_kind_raises_signal_kind_error_on_conflict():
    freq_sig = _FakeSignal(xlabel='frequency', y=[1.0, 2.0, 3.0])
    with pytest.raises(SignalKindError):
        _check_kind(freq_sig, TRANSIENT, 'rise_time')


def test_check_kind_conflict_is_still_caught_as_value_error():
    # Message-pinning + existing "except ValueError" handlers must keep
    # working unchanged (tests/test_signal_kinds.py, tests/metrics/test_dc.py).
    freq_sig = _FakeSignal(xlabel='frequency', y=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match='must be a transient Signal'):
        _check_kind(freq_sig, TRANSIENT, 'rise_time')


def test_check_kind_conflict_message_unchanged():
    freq_sig = _FakeSignal(xlabel='frequency', y=[1.0, 2.0, 3.0])
    with pytest.raises(SignalKindError) as exc:
        _check_kind(freq_sig, TRANSIENT, 'rise_time', slot='sig_out')
    msg = str(exc.value)
    assert "rise_time: sig_out must be a transient Signal" in msg
    assert "xlabel='frequency'" in msg


# ---------------------------------------------------------------------------
# hasattr()/getattr(obj, name, default) must NOT swallow UnboundSignalError
# ---------------------------------------------------------------------------

class _UnboundLike:
    """Stands in for the future unbound Signal's __getattr__ fallback
    (TASK-215.2): reading 'y' (or any non-underscore name) raises
    UnboundSignalError instead of returning/absence; names starting with
    '_' still raise plain AttributeError so copy/pickle keep working."""

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise UnboundSignalError(
            f"probe {name!r}: bench 'cm' has not run yet"
        )


def test_hasattr_does_not_swallow_unbound_signal_error():
    obj = _UnboundLike()
    with pytest.raises(UnboundSignalError):
        hasattr(obj, 'y')


def test_getattr_with_default_does_not_swallow_unbound_signal_error():
    obj = _UnboundLike()
    with pytest.raises(UnboundSignalError):
        getattr(obj, 'y', None)


def test_getattr_default_still_works_for_underscore_names():
    # Sanity check: the dunder/private-name escape hatch behaves like a
    # normal AttributeError-driven getattr default.
    obj = _UnboundLike()
    assert getattr(obj, '_missing', 'fallback') == 'fallback'
