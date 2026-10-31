"""PortWiring — value-typed entry for SubcircuitInstance._port_map.

Records what each pin on a sub-instance is wired to, plus optional bus
membership so ``inst.d`` (where ``d`` is a bus port on the child) can be
reconstructed as a width-aware Signal.

.. note::

    This was previously named ``PortBinding``, which collided with the
    unrelated :class:`silk.circuit.net.PortBinding` (pin contributions to a
    :class:`~silk.circuit.net.Net`, for interface checking). Renamed to
    ``PortWiring`` to remove that ambiguity; this type is specific to
    ``SubcircuitInstance._port_map`` and carries parent-net wiring plus bus
    membership metadata.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class PortWiring:
    """One entry in SubcircuitInstance._port_map.

    Parameters
    ----------
    parent_net : str
        Name of the parent-side net this pin is bound to.
    bus : Optional[Tuple[str, int]]
        If this pin is a bit of a bus port on the child, the
        ``(bus_name, bit_index)`` pair. ``None`` for scalar ports.
    """
    parent_net: str
    bus: Optional[Tuple[str, int]] = None


# Pickle back-compat: SubcircuitInstance state pickled before the rename
# references ``silk.circuit.port_binding.PortBinding`` by class path; keep it
# resolvable (and identical to PortWiring) so SubcircuitInstance.__setstate__
# still upgrades old pickles. Do NOT use in new code — it is the same class.
PortBinding = PortWiring
