"""silk.layout — layout intelligence for silk circuits.

Phase 1 public API (statistical estimation):
    RoutingModel, HalfCircuit, LayoutEstimate, NetLayout, DeviceLayout,
    LayoutWarning, LayoutInferenceError
    estimate_layout, capture_layout, inject_parasitics, default_escalation_rule
"""
from silk.layout.area_model import set_footprint
from silk.layout.routing_model import RoutingModel
from silk.layout.matching import HalfCircuit, LayoutInferenceError
from silk.layout._types import LayoutWarning
from silk.layout.estimator import (
    LayoutEstimate,
    NetLayout,
    DeviceLayout,
    estimate_layout,
    capture_layout,
    default_escalation_rule,
)
from silk.layout.parasitics import inject_parasitics

__all__ = [
    'RoutingModel',
    'HalfCircuit',
    'LayoutInferenceError',
    'LayoutWarning',
    'LayoutEstimate',
    'NetLayout',
    'DeviceLayout',
    'estimate_layout',
    'set_footprint',
    'capture_layout',
    'inject_parasitics',
    'default_escalation_rule',
]
