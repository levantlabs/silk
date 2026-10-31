"""silk.layout.wirelength_model — statistical per-net wire length estimation.

Uses module bounding box and net classification/fanout to produce a coarse
wire length estimate without any placement data.  Accuracy is intentionally
rough (~2-3× for individual nets); useful for triage (worst-offender ranking)
not for signoff.
"""
from __future__ import annotations

import math
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from silk.circuit.net import Net
    from silk.circuit.modules import Module

# Signal net Rent's-rule constant: average local net spans ~0.7 × L_char × sqrt(fanout)
_C_SIGNAL = 0.7
# Minimum fanout for Rent's rule — avoids degenerate single-pin estimates
_MIN_FANOUT = 2


def estimate_net_length(
    net,
    module,
    classification: Optional[object],
) -> float:
    """Statistical per-net wire length.

    Uses the module bounding box (must already be set on ``module.bounding_box``)
    and the net's classification and fanout to return an estimated length in μm.

    Parameters
    ----------
    net : Net
        The net being estimated. Uses ``net.fanout``.
    module : Module
        Parent module.  ``module.bounding_box`` must be a ``(width, height)``
        tuple of floats in μm.
    classification : PortRole or None
        The canonical classification for this net (or its equivalence class).
        POWER and GROUND get a global-trunk estimate; CLOCK gets a
        clock-tree estimate; everything else uses Rent's rule.

    Returns
    -------
    float
        Estimated wire length in μm.  Always positive.
    """
    from silk.circuit.circuit_elements import PortRole

    W, H = module.bounding_box
    perimeter = 2.0 * (W + H)
    L_char = math.sqrt(W * H)
    fanout = max(net.fanout, _MIN_FANOUT)

    if classification in (PortRole.POWER, PortRole.GROUND):
        # Power and ground trunks span the full module perimeter
        return 2.0 * perimeter

    if classification == PortRole.CLOCK:
        # Clock tree spans most of the module
        return 0.8 * perimeter

    # Signal (ANALOG, DIGITAL, or None): Rent's-rule local length
    return L_char * math.sqrt(fanout) * _C_SIGNAL
