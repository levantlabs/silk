"""SimulationAborted: structured, and is-a SimulationError."""
import pytest
from silk.exceptions import SimulationAborted, SimulationError


def test_is_a_simulation_error():
    err = SimulationAborted("boom", simulator="ngspice")
    assert isinstance(err, SimulationError)


def test_caught_by_existing_simulation_error_handlers():
    with pytest.raises(SimulationError):
        raise SimulationAborted("boom", simulator="ngspice")


def test_carries_structured_fields():
    err = SimulationAborted(
        "aborted", simulator="ngspice",
        reason="Timestep too small", data_rows=0,
        log_path="/tmp/output.log",
    )
    assert err.simulator == "ngspice"
    assert err.reason == "Timestep too small"
    assert err.data_rows == 0
    assert err.log_path == "/tmp/output.log"
    assert "aborted" in str(err)


def test_optional_fields_default_none():
    err = SimulationAborted("aborted", simulator="xyce")
    assert err.reason is None
    assert err.data_rows is None
    assert err.log_path is None


def test_exported():
    import silk.exceptions as ex
    assert "SimulationAborted" in ex.__all__
