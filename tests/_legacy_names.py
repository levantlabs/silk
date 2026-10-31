"""Build pre-TASK-553 net fixtures for raw-reader compatibility tests only."""
from unittest.mock import patch


def add_legacy_net(module, name):
    """Model an imported graph; restore name validation before any assertions.

    Existing rawfiles may still contain punctuation in node leaves. These
    tests exercise readback of that evidence, never permission to emit it.
    """
    with patch('silk.spice_names.validate_authored_name', return_value=None):
        return module.add_net(name)


def legacy_net(name):
    """An unattached historical Net for aggregate-index reader tests."""
    from silk.circuit.net import Net
    with patch('silk.spice_names.validate_authored_name', return_value=None):
        return Net(name, None)
