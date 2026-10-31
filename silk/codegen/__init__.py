"""Generate Python source from circuit descriptions.

:class:`Generator` turns a circuit description dict from any tool (or a
:class:`CircuitSpec`) into a runnable ``Module`` script or subcircuit
definition, and can generate testbench code from a testbench description. The
description format is specified in ``docs/codegen-contract.md``.
:class:`ElementNames` and :func:`source_element_names` report the SPICE names a
saved ``sources[]`` entry will emit.
"""
from silk.codegen.generator import (
    Generator, CircuitSpec, DeviceSpec, BusNetSpec,
    SubmoduleSpec,
)
from silk.circuit.emitted_names import ElementNames, source_element_names

__all__ = ["Generator", "CircuitSpec",
           "DeviceSpec", "BusNetSpec", "SubmoduleSpec",
           "ElementNames", "source_element_names"]
