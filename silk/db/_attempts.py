"""Private links from stored projections to live observation snapshots.

SQL rows retain their existing schema and meaning. These links exist only in
this root's lifetime and are never reconstructed by guessing from row values.
"""
from dataclasses import dataclass
from numbers import Real

from silk.specs.attempt import prepare_measurement, _merge, _pack


@dataclass(frozen=True)
class Projection:
    observation: object
    scope: object
    payload: str


def _payload(m):
    from silk.simulation.conditions import encode_condition
    # Stored row facts, not an observation codec. Version identity is supplied
    # by the surrounding tracker and is not part of the legacy live slot.
    data = {}
    for name in ('source', 'value', 'passes', 'timestamp', 'corner', 'temperature',
                 'voltage', 'sim_run_id', 'notes', 'sim_type', 'tool_name', 'profile_id'):
        value = getattr(m, name)
        if name == 'timestamp' and value is None:
            value = 0.0
        # SQLite REAL affinity normalizes legal integer inputs on readback.
        # This compares one associated row's facts, never distinct observations.
        if name in ('value', 'timestamp', 'temperature', 'voltage') and isinstance(value, Real):
            value = float(value) if value != 0 else 0.0
        data[name] = (encode_condition(value)
                      if name in ('corner', 'temperature', 'voltage') else value)
    return _pack(data)


def stage_observation(db, spec, measurement):
    """Validate against published AND pending observations before SQL commit."""
    if measurement is None:
        return None, None
    attempt, scope = prepare_measurement(spec, measurement)
    if attempt is None:
        return None, scope
    pending = db.__dict__.setdefault('_pending_spec_attempts', {})
    key = (id(spec), scope, attempt.attempt_id)
    previous = pending.get(key)
    if previous is not None:
        attempt = _merge(previous, attempt)
    pending[key] = attempt

    def undo():
        if previous is None:
            pending.pop(key, None)
        else:
            pending[key] = previous

    db._after_rollback(undo)
    db._after_commit(lambda: pending.pop(key, None))
    return attempt, scope


def projection_for(root, spec, owner, row_id, measurement, observation, scope):
    """Stage a row association without publishing it or changing observation facts."""
    from silk.circuit.test_placement import evidence_index
    associations = evidence_index(root).projection_associations
    key = (owner, id(spec), row_id)
    return associations, key, Projection(observation, scope, _payload(measurement))


def attach_projection(measurement, projection):
    if _payload(measurement) != projection.payload:
        raise ValueError('Stored projection conflicts with its recorded row identity')
    measurement._legacy_evidence = True
    if projection.observation is not None:
        measurement._projection_observation = (projection.observation, projection.scope)


def hydrate(root, spec, owner, row_id, measurement):
    """Hydrate legacy rows; only an existing private association proves a run."""
    index = root.__dict__.get('_test_evidence')
    association = (index.projection_associations.get((owner, id(spec), row_id))
                   if index is not None else None)
    measurement._legacy_evidence = True
    if association is not None:
        attach_projection(measurement, association)
    return prepare_measurement(spec, measurement)
