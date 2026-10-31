import pytest
from silk.lint.severity import Severity


def test_severity_levels_defined():
    assert Severity.ERROR
    assert Severity.WARNING
    assert Severity.INFO


def test_severity_ordering_error_worst():
    assert Severity.ERROR.value > Severity.WARNING.value
    assert Severity.WARNING.value > Severity.INFO.value


def test_severity_from_string_roundtrip():
    assert Severity.from_string("error") == Severity.ERROR
    assert Severity.from_string("WARNING") == Severity.WARNING
    assert Severity.ERROR.to_string() == "error"


def test_severity_from_string_invalid():
    with pytest.raises(ValueError, match="Unknown severity"):
        Severity.from_string("critical")
