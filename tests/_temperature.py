"""Explicit backend-observation stand-in for tests that stub Module.simulate."""
from silk.simulation.conditions import OperatingConditions


def mark_simulated(module, corner='tt', temperature=27.0, voltage=None,
                   supply_source='vsup'):
    """Stand in for what a real launch establishes.

    A stubbed ``simulate`` emits no deck, so nothing can confirm a supply from
    one (TASK-158). Tests that need an established supply say so here, which
    is the stub's equivalent of the deck check the real run performs.
    """
    module.sim_results_loaded = True
    module.operating_conditions = OperatingConditions(
        'silk', module.simulator, str(corner), 'fixed', temperature,
        'backend-observed',
        supply_voltage_v=voltage,
        supply_basis='unknown' if voltage is None else 'configured-and-emitted',
        supply_source=None if voltage is None else supply_source,
    )
