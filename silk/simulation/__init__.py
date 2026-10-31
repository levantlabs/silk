"""silk.simulation: analysis requests, simulator backends and operating conditions.

* Analysis requests are small typed objects passed to
  ``Module.add_sim_type()``: :class:`Transient`, :class:`AC`, :class:`DC`,
  :class:`Noise` and :class:`Op`, all derived from :class:`AnalysisBase`.
  Each validates its numbers on construction and raises ``ValueError``.
* :class:`Corner` names the five standard process corners.
* :class:`NgspiceRunner` is the thin ngspice subprocess wrapper behind the
  ngspice backend.
* :class:`OperatingConditions`, :class:`ConditionSweep` and
  :class:`ConditionState` record which corner, temperature and supply a set
  of results corresponds to and how silk knows it;
  :data:`CONDITION_VOCABULARY` and :func:`export_conditions_schema` publish
  the closed vocabulary of states and basis words, :func:`encode_condition`
  and :func:`decode_condition` are the JSON wire codec, and
  :func:`has_no_pdk_device` and :func:`has_no_supply` are the two proofs a
  consumer running a silk-generated deck itself can reuse.

Simulator backends (``silk.simulation.backend``) and the behavioral engine
(``silk.simulation.behavioral``) are separate modules; importing this
package registers the ``'ngspice'`` backend.

Examples
--------
>>> from silk.simulation import Transient, AC, DC, Op
>>> Transient(step=1e-9, stop=100e-9).to_spice().strip()
'.tran 1n 100n'
>>> Op().to_spice().strip()
'.op'
"""
from silk.simulation.analysis import Transient, AC, DC, Noise, Op, AnalysisBase
from silk.simulation.runner import NgspiceRunner
from silk.simulation.corner import Corner
from silk.simulation.conditions import (
    OperatingConditions, ConditionSweep, ConditionState, CONDITION_VOCABULARY,
    encode_condition, decode_condition, export_conditions_schema,
    has_no_pdk_device, has_no_supply,
)
from silk.simulation._setup_check import validate_simulation_setup
from silk.simulation import backend  # noqa: F401  (registers the 'ngspice' backend)

__all__ = ["Transient", "AC", "DC", "Noise", "Op", "AnalysisBase", "NgspiceRunner", "Corner",
           "OperatingConditions", "ConditionSweep", "ConditionState", "CONDITION_VOCABULARY",
           "encode_condition", "decode_condition", "export_conditions_schema",
           "has_no_pdk_device", "has_no_supply", "validate_simulation_setup"]
