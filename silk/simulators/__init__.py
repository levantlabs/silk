"""Per-simulator configuration (options syntax, presets, allowlists).

See ``ngspice.yaml`` for the schema. Load via :func:`load_simulator_config`.
"""

from silk.simulators._loader import SimulatorConfig, load_simulator_config

__all__ = ['SimulatorConfig', 'load_simulator_config']
