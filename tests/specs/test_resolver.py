"""Dotted-path metric resolver."""
import pytest

from silk.specs.resolver import resolve_metric, ResolveResult


class _FakeNet:
    def __init__(self, name, **kwargs):
        self.name = name
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeDevice:
    def __init__(self, name, **kwargs):
        self.name = name
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeModule:
    def __init__(self):
        self.net_dict = {}
        self.elem_dict = {}


def test_resolve_plain_metric_returns_unresolved():
    """A plain metric name is not a dotted path — return UNRESOLVED."""
    mod = _FakeModule()
    result = resolve_metric(mod, 'noise_figure')
    assert result.resolved is False
    assert result.value is None
    assert result.reason == 'not_a_dotted_path'


def test_resolve_net_attribute():
    mod = _FakeModule()
    mod.net_dict['vout'] = _FakeNet('vout', peak_current=3.2e-3)
    result = resolve_metric(mod, 'net.vout.peak_current')
    assert result.resolved is True
    assert result.value == 3.2e-3
    assert result.reason is None


def test_resolve_device_attribute():
    mod = _FakeModule()
    mod.elem_dict['mn1'] = _FakeDevice('mn1', vgs=0.72)
    result = resolve_metric(mod, 'device.mn1.vgs')
    assert result.resolved is True
    assert result.value == 0.72


def test_resolve_missing_net_returns_unresolved():
    mod = _FakeModule()
    result = resolve_metric(mod, 'net.vout.peak_current')
    assert result.resolved is False
    assert result.reason == 'net_not_found'


def test_resolve_missing_attr_returns_unresolved():
    mod = _FakeModule()
    mod.net_dict['vout'] = _FakeNet('vout')
    result = resolve_metric(mod, 'net.vout.peak_current')
    assert result.resolved is False
    assert result.reason == 'attr_not_found'


def test_resolve_attr_present_but_none_returns_unresolved():
    """Phase 2 fields that are declared but not yet populated should be treated as unresolved."""
    mod = _FakeModule()
    mod.net_dict['vout'] = _FakeNet('vout', peak_current=None)
    result = resolve_metric(mod, 'net.vout.peak_current')
    assert result.resolved is False
    assert result.reason == 'value_is_none'


def test_resolve_unknown_prefix_returns_unresolved():
    mod = _FakeModule()
    result = resolve_metric(mod, 'signal.foo.bar')
    assert result.resolved is False
    assert result.reason == 'unknown_prefix'


def test_resolve_malformed_path_returns_unresolved():
    mod = _FakeModule()
    result = resolve_metric(mod, 'net.vout')  # missing attr
    assert result.resolved is False
    assert result.reason == 'malformed_path'
