import pytest
from silk.lint_markers import NoConnect


def test_noconnect_requires_reason():
    with pytest.raises(TypeError):
        NoConnect()


def test_noconnect_with_reason():
    nc = NoConnect(reason="differential test tap")
    assert nc.reason == "differential test tap"


def test_noconnect_reason_must_be_nonempty():
    with pytest.raises(ValueError, match="empty"):
        NoConnect(reason="")


def test_noconnect_reason_must_be_nonblank():
    with pytest.raises(ValueError, match="empty"):
        NoConnect(reason="   ")


def test_noconnect_is_lint_marker_flag():
    nc = NoConnect(reason="probe tap")
    assert nc.is_lint_marker is True


def test_noconnect_repr_contains_reason():
    nc = NoConnect(reason="probe tap")
    assert "probe tap" in repr(nc)


def test_noconnect_can_be_added_to_module(tmp_path):
    """add_no_connect() registers the marker and creates a PortBinding."""
    from unittest.mock import MagicMock
    from silk.circuit.modules import Module
    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=pdk)
    m.add_net('probe_out')
    m.add_no_connect('probe_out', reason="differential test tap")
    # The net should now have a PortBinding with kind='lint_marker'
    binding = m.net_dict['probe_out'].pins[0]
    assert binding.kind == 'lint_marker'
    assert binding.instance.startswith('_nc_')
