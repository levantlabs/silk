"""Fault injection: declare device faults, run Monte Carlo campaigns, report.

A :class:`FaultSet` collects faults (:func:`stuck_open`, :func:`short_ds`,
:func:`short_gs`, :func:`short_gd`, :func:`w_variation`, :func:`l_variation`,
:func:`vth_shift`) and random *sprinkles* over a module's device paths. A
:class:`FaultCampaign` realizes the set once per run with a derived seed,
edits each run's SPICE netlist, simulates it and scores the specs of the
campaign testbench (``FaultCampaign(tb.netlist, ...)``) from that run's own
output; a campaign with no bound testbench only executes or stays
unverified. The :class:`CampaignResult` it returns exposes pass fractions,
measurement distributions and fault-to-failure correlations, and
:meth:`CampaignResult.generate_report` renders them.

Faults alter SPICE decks, so campaigns refuse the ``'behavioral'`` simulator
with :class:`silk.exceptions.NotSupportedError`.
"""

from silk.faults.types import Fault, AppliedFault, FaultManifest
from silk.faults.sugar import (
    FaultFactory,
    stuck_open, short_ds, short_gs, short_gd,
    w_variation, l_variation, vth_shift,
)
from silk.faults.set import FaultSet
from silk.faults.campaign import FaultCampaign, RunResult, CampaignResult

__all__ = [
    'Fault', 'AppliedFault', 'FaultManifest',
    'FaultFactory', 'FaultSet',
    'FaultCampaign', 'RunResult', 'CampaignResult',
    'stuck_open', 'short_ds', 'short_gd', 'short_gs',
    'w_variation', 'l_variation', 'vth_shift',
]
