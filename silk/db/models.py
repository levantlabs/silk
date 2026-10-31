"""Immutable value objects returned by :mod:`silk.db` APIs.

Operating conditions (schema v5): every record's ``corner``,
``temperature`` and ``voltage`` is a condition scalar, decoded from the
row's state column. A value is a ``str`` (corner) or ``float``;
``ConditionState.NONE`` means the condition provably does not apply;
``None`` means unknown; ``ConditionState.SWEEP`` (temperature only) means
the evidence spans a temperature range. ``ConditionState`` is a plain Enum
from :mod:`silk.simulation`, so a none corner never equals a corner string.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple, Union
from silk.specs.aggregate import AggregateMeasurement

#: A widened condition scalar (see the module docstring).
ConditionScalar = Union[str, float, Any, None]


@dataclass(frozen=True)
class DeckCheck:
    """A per-call comparison of a stored observation's producing deck with today's.

    Attached to records returned by :class:`~silk.db.CircuitTracker` reads (and
    to the nested records of an evidence selection) as ``deck_check``. Raw
    :class:`~silk.db.Database` reads carry ``None`` instead. The check is
    ephemeral: record equality ignores it and silk does not persist it, although
    Python ``dataclasses.asdict`` and pickle still include it.

    Parameters
    ----------
    state : str
        ``'checked'``, ``'stale'`` or ``'not checked'``.
    reason : str
        Human-readable explanation of the verdict.

    Attributes
    ----------
    state : str
        ``'checked'`` when the producing deck and simulator backend match the
        live bench, ``'stale'`` when the live deck or backend differs from the one
        that produced the observation, ``'not checked'`` when the comparison could
        not be made (for example no covered silk SPICE launch was recorded).
    reason : str
        Why the check reached ``state``. A stale reason names the producing test
        and the call that reruns it.
    """

    state: str
    reason: str


@dataclass(frozen=True)
class Fingerprint:
    """A digest paired with the algorithm that produced it.

    Persisted digests are only comparable within one scheme, so every identity
    in :mod:`silk.db` carries its scheme next to the digest. A comparison whose
    fingerprint has a scheme other than the one the operation expects raises
    :class:`~silk.exceptions.UnsupportedFingerprintSchemeError`.

    Parameters
    ----------
    scheme : str
        Scheme tag, for example ``'silk.structure/v4'`` (circuit structure),
        ``'silk.spec/v1'`` (specification declarations) or
        ``'silk.profile/v1'`` (characterization profiles).
    digest : str
        Lowercase hexadecimal SHA-256 digest under ``scheme``.

    Attributes
    ----------
    scheme : str
        Scheme tag of the digest.
    digest : str
        Lowercase hexadecimal SHA-256 digest.

    Examples
    --------
    >>> from silk.db import Fingerprint
    >>> fp = Fingerprint('silk.structure/v4', '0' * 64)
    >>> fp.scheme
    'silk.structure/v4'
    """

    scheme: str
    digest: str


class _UnscopedProfile:
    """Sentinel that selects only unscoped (``profile_id IS NULL``) evidence.

    ``None`` (the default everywhere) means "no profile filter": both scoped and
    unscoped evidence match. Passing this sentinel narrows a ``profile_id``
    filter to unscoped rows only; passing a ``profile_id`` string narrows it to
    that one profile. The sentinel is never a database identity: it appears only
    in Python filter arguments, never in SQL parameters or stored rows. It is a
    singleton and its ``repr`` is ``UNSCOPED_PROFILE``.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSCOPED_PROFILE"


#: Singleton filter value that selects only explicitly unscoped evidence.
#:
#: Pass it as ``profile_id`` to a :class:`~silk.db.Database` or
#: :class:`~silk.db.CircuitTracker` read or restore to keep only rows stored
#: without a characterization profile. ``None`` applies no profile filter and an
#: exact ``profile_id`` string selects that one profile.
UNSCOPED_PROFILE = _UnscopedProfile()


@dataclass(frozen=True)
class SnapshotIdentity:
    """Independent structure and specification-set identities of one snapshot.

    A sealed circuit snapshot is identified by what the circuit contains
    (``structure``) and by which specs it declares (``spec_set``). Reading
    stored metrics back through a tracker needs a matching ``structure``;
    selecting or restoring characterization evidence also needs a matching
    ``spec_set``.

    Parameters
    ----------
    structure : Fingerprint
        Structure identity, scheme ``'silk.structure/v4'``.
    spec_set : Fingerprint
        Identity of the complete spec declaration set, scheme ``'silk.spec/v1'``.

    Attributes
    ----------
    structure : Fingerprint
        Structure identity.
    spec_set : Fingerprint
        Specification-set identity.

    Examples
    --------
    >>> from silk.db import Fingerprint, SnapshotIdentity
    >>> identity = SnapshotIdentity(
    ...     Fingerprint('silk.structure/v4', 'a' * 64),
    ...     Fingerprint('silk.spec/v1', 'b' * 64))
    >>> identity.spec_set.scheme
    'silk.spec/v1'
    """

    structure: Fingerprint
    spec_set: Fingerprint


@dataclass(frozen=True)
class CharacterizationProfileRecord:
    """One immutable characterization identity at its origin snapshot.

    A profile names which registered test, on which owner, with which declared
    parameters produced evidence. Profiles are content-addressed per origin
    snapshot; they do not include the process, temperature or voltage point.

    Attributes
    ----------
    profile_id : str
        Stable identifier of the profile row.
    origin_version_id : str
        Version id of the snapshot that owns the profile.
    owner_path : str
        Instance path of the owning module (``$`` for the root, ``$/amp1`` for an
        instance).
    test_name : str
        Name of the registered test.
    parameters : dict
        Complete effective declared test parameters (defaults materialised),
        decoded from the stored canonical form.
    profile_scheme : str
        Fingerprint scheme of ``profile_hash``, ``'silk.profile/v1'``.
    profile_hash : str
        SHA-256 digest of the canonical owner, test and parameter payload.
    created_at : str
        Insertion time as written by SQLite (UTC, ``YYYY-MM-DD HH:MM:SS``).
    """

    profile_id: str
    origin_version_id: str
    owner_path: str
    test_name: str
    parameters: Mapping[str, Any]
    profile_scheme: str
    profile_hash: str
    created_at: str


@dataclass(frozen=True)
class NetCurrentRecord:
    """One validated, immutable current row within a stored current set.

    A row is either the net-level aggregate (``pin_id == ''``) or one pin of the
    net (``pin_id == 'instance.port'``). Currents are in amperes.

    Attributes
    ----------
    current_id : str
        Identifier of the row.
    net_name : str
        Name of the net.
    pin_id : str
        Empty string for the net-level aggregate row, otherwise
        ``'<instance>.<port>'``.
    avg_a : float
        Average current in amperes.
    rms_a : float or None
        RMS current in amperes, ``None`` when not recorded.
    peak_a : float or None
        Peak current in amperes, ``None`` when not recorded.
    source : str
        ``'spice'`` or ``'user'``.
    current_vector : str or None
        Per-pin current vector as stored text, ``None`` for aggregate rows or
        when not recorded.
    timestamp : str
        Insertion time as written by SQLite (UTC, to the second,
        ``YYYY-MM-DD HH:MM:SS``).
    """

    current_id: str
    net_name: str
    pin_id: str
    avg_a: float
    rms_a: Optional[float]
    peak_a: Optional[float]
    source: str
    current_vector: Optional[str]
    timestamp: str


@dataclass(frozen=True)
class NetCurrentSetRecord:
    """One stable current-set endpoint, either selected or superseded.

    A current set groups the :class:`NetCurrentRecord` rows stored together for
    one snapshot and one selection key (an unscoped corner, or a profile plus
    corner, temperature and voltage). Replacing a set supersedes the previous
    one and keeps it readable by its id.

    Attributes
    ----------
    current_set_id : str
        Stable identifier of the set.
    profile_id : str or None
        Characterization profile that scopes the set; ``None`` for an unscoped
        set.
    origin_version_id : str
        Version id of the snapshot that owns the set.
    state : str
        ``'current'`` for the selected set, ``'superseded'`` for a retained
        predecessor.
    supersedes_current_set_id : str or None
        Id of the set this one replaced, ``None`` for the first set.
    corner : str or ConditionState or None
        Process corner name, ``ConditionState.NONE`` or ``None`` (unknown).
    temperature : float or ConditionState or None
        Temperature in degrees Celsius, ``ConditionState.NONE``, ``ConditionState.SWEEP`` or
        ``None`` (unknown).
    voltage : float or ConditionState or None
        Supply volts, ``ConditionState.NONE`` or ``None`` (unknown).
    tool_name : str or None
        Tool that produced the currents.
    analysis : str or None
        Analysis the currents came from.
    tstart : float or None
        Start of the report window the currents were taken over, in
        seconds of simulation time (``report.window[0]``).
    results_file : str or None
        Path of the simulator results file the currents came from.
    structure_scheme : str
        Fingerprint scheme of ``structure_hash``.
    structure_hash : str
        Structure digest of the snapshot the set was stored against.
    selected_at : str
        Time the set was stored (UTC, ``YYYY-MM-DD HH:MM:SS``).
    superseded_at : str or None
        Time the set was superseded, ``None`` while it is current.
    currents : tuple of NetCurrentRecord
        The rows of the set, in insertion order.
    provenance_id : str or None
        Id of the stored run-provenance record, resolved with
        :meth:`silk.db.Database.get_run_provenance`; ``None`` when none was
        recorded (every row written before schema v4).
    deck_check : DeckCheck or None
        Ephemeral deck comparison; ``None`` on raw :class:`~silk.db.Database`
        reads. Ignored by equality.
    """

    current_set_id: str
    profile_id: Optional[str]
    origin_version_id: str
    state: str
    supersedes_current_set_id: Optional[str]
    corner: ConditionScalar
    temperature: ConditionScalar
    voltage: ConditionScalar
    tool_name: Optional[str]
    analysis: Optional[str]
    tstart: Optional[float]
    results_file: Optional[str]
    structure_scheme: str
    structure_hash: str
    selected_at: str
    superseded_at: Optional[str]
    currents: Tuple[NetCurrentRecord, ...]
    # Schema v4 (TASK-306): see SimulationResultRecord.provenance_id.
    provenance_id: Optional[str] = None

    deck_check: Optional[DeckCheck] = field(default=None, compare=False)


@dataclass(frozen=True)
class SimulationResultRecord:
    """One typed metric observation and the snapshot where it originated.

    Exactly one of ``value`` and ``value_text`` carries the observation, except
    that a stored row with both columns NULL reads back with ``value`` set to
    ``nan``. The ``corner``, ``temperature`` and ``voltage`` fields are condition
    scalars: a value, :class:`~silk.simulation.ConditionState` ``NONE`` (the
    condition provably does not apply), ``SWEEP`` (temperature only) or ``None``
    (unknown).

    Attributes
    ----------
    result_id : str
        Identifier of the stored row.
    origin_version_id : str
        Version id of the snapshot the observation was stored on. A read through
        a descendant snapshot can return rows from compatible ancestors.
    parameter : str
        Metric name, for example ``'gain_dB'``.
    value : float or None
        Numeric value, ``nan`` for an unavailable numeric value, ``None`` for a
        text observation.
    value_text : str or None
        Text value (for example a pass/fail status or an error), ``None`` for a
        numeric observation.
    unit : str or None
        Unit label stored with the value.
    object_id : str or None
        Identifier of the object the metric is about, when the observation is not
        about the whole circuit.
    object_type : str
        Kind of object the observation is about; ``'circuit'`` by default.
    source : str
        Non-empty label of where the value came from; ``'user'`` by default.
    sim_type : str or None
        Label of the analysis that produced the value; not validated.
    corner : str or ConditionState or None
        Process corner name, ``ConditionState.NONE`` or ``None`` (unknown).
    temperature : float or ConditionState or None
        Temperature in degrees Celsius, ``ConditionState.NONE``, ``ConditionState.SWEEP`` or
        ``None`` (unknown).
    voltage : float or ConditionState or None
        Supply volts, ``ConditionState.NONE`` or ``None`` (unknown).
    tool_name : str or None
        Tool that produced the value.
    conditions : dict or None
        JSON-compatible dictionary of extra conditions stored with the row.
    timestamp : str
        Time recorded with the row: the caller's string, or insertion time as
        written by SQLite (UTC, ``YYYY-MM-DD HH:MM:SS``).
    profile_id : str or None
        Characterization profile that scopes the row; ``None`` for an unscoped
        (legacy or manual) row.
    provenance_id : str or None
        Id of the stored run-provenance record (schema v4), resolved with
        :meth:`silk.db.Database.get_run_provenance`; ``None`` when none was
        recorded.
    deck_check : DeckCheck or None
        Ephemeral deck comparison; ``None`` on raw :class:`~silk.db.Database`
        reads. Ignored by equality.
    """

    result_id: str
    origin_version_id: str
    parameter: str
    value: Optional[float]
    value_text: Optional[str]
    unit: Optional[str]
    object_id: Optional[str]
    object_type: str
    source: str
    sim_type: Optional[str]
    corner: ConditionScalar
    temperature: ConditionScalar
    voltage: ConditionScalar
    tool_name: Optional[str]
    conditions: Optional[Dict[str, Any]]
    timestamp: str
    # DB-02/TASK-6 (design SS4.3/SS8): NULL = explicitly unscoped/legacy/manual.
    # Defaulted (not a bare positional field) so the frozen v1 fixture in
    # tests/db/_v1_fixture/legacy_circuit_db.py -- which must never be updated
    # to track v2 -- keeps constructing this record without passing it.
    profile_id: Optional[str] = None
    # Schema v4 (TASK-306): the run_provenance row this observation was
    # stored with; None = none recorded (every pre-v4 row). Resolve it with
    # Database.get_run_provenance().
    provenance_id: Optional[str] = None

    deck_check: Optional[DeckCheck] = field(default=None, compare=False)


@dataclass(frozen=True)
class SpecMeasurementRecord:
    """One stored spec observation and the snapshot where it was measured.

    Attributes
    ----------
    measurement_id : str
        Identifier of the stored measurement (chosen by the writer).
    origin_version_id : str
        Version id of the snapshot the measurement was stored on.
    origin_spec_id : str
        Id of the spec declaration row, in the origin snapshot, that the
        measurement belongs to.
    source : str
        ``'behavioral'``, ``'spice'``, ``'extraction'`` or ``'user'``.
    value : float
        Measured value in the units of the spec; ``nan`` when the stored value is
        NULL (an unavailable measurement).
    corner : str or ConditionState or None
        Process corner name, ``ConditionState.NONE`` or ``None`` (unknown).
    temperature : float or ConditionState or None
        Temperature in degrees Celsius, ``ConditionState.NONE``, ``ConditionState.SWEEP`` or
        ``None`` (unknown).
    voltage : float or ConditionState or None
        Supply volts, ``ConditionState.NONE`` or ``None`` (unknown).
    sim_type : str or None
        Label of the analysis that produced the value; not validated.
    tool_name : str or None
        Tool that produced the value.
    timestamp : float or None
        Observation time as stored (the live ``Measurement`` convention is Unix
        seconds); ``None`` when not recorded.
    sim_run_id : str or None
        External run-correlation identifier; for an aggregate measurement it
        equals the aggregate's ``campaign_id``.
    notes : str
        Free-text note; the empty string when none was stored.
    passes : bool or None
        Cached verdict stored with the measurement, ``None`` when none was
        cached. Reads verify it against the declaration's operator and target.
    profile_id : str or None
        Characterization profile that scopes the measurement; ``None`` for an
        unscoped one.
    provenance_id : str or None
        Id of the stored run-provenance record (schema v4), resolved with
        :meth:`silk.db.Database.get_run_provenance`; ``None`` when none was
        recorded.
    aggregate : AggregateMeasurement or None
        Monte Carlo attribution when ``value`` summarises a campaign; ``None``
        for a single-run value.
    deck_check : DeckCheck or None
        Ephemeral deck comparison; ``None`` on raw :class:`~silk.db.Database`
        reads. Ignored by equality.
    """

    measurement_id: str
    origin_version_id: str
    origin_spec_id: str
    source: str
    value: float
    corner: ConditionScalar
    temperature: ConditionScalar
    voltage: ConditionScalar
    sim_type: Optional[str]
    tool_name: Optional[str]
    timestamp: Optional[float]
    sim_run_id: Optional[str]
    notes: str
    passes: Optional[bool]
    # See SimulationResultRecord.profile_id above.
    profile_id: Optional[str] = None
    # See SimulationResultRecord.provenance_id above.
    provenance_id: Optional[str] = None

    aggregate: Optional['AggregateMeasurement'] = None

    deck_check: Optional[DeckCheck] = field(default=None, compare=False)


@dataclass(frozen=True)
class SpecDeclarationRecord:
    """One persisted spec declaration with optionally attached measurement history.

    Attributes
    ----------
    spec_id : str
        Identifier of the declaration row in the queried snapshot.
    declaration_version_id : str
        Version id of the snapshot whose declarations were read.
    owner_path : str
        Instance path of the module that owns the spec (``$`` for the root).
    spec_key : str
        Stable key of the spec within its owner.
    name : str
        Display name of the spec.
    metric : str
        Name of the metric the spec constrains.
    op : str
        Comparison operator, one of ``'<'``, ``'<='``, ``'>'``, ``'>='``,
        ``'=='``.
    target : float
        Target value, in the units of ``units``.
    units : str
        Unit label of the target.
    description : str
        Free-text description.
    parent_id : str or None
        Id of the parent declaration row, ``None`` for a top-level spec.
    parent_owner_path : str or None
        Owner path of the parent spec, ``None`` for a top-level spec.
    parent_spec_key : str or None
        Key of the parent spec, ``None`` for a top-level spec.
    declared_at : str
        Location label of the declaration, as supplied by the declaring API.
    definition_scheme : str
        Fingerprint scheme of ``definition_hash``, ``'silk.spec/v1'``.
    definition_hash : str
        SHA-256 digest of the declaration fields.
    created_at : str
        Insertion time as written by SQLite (UTC, ``YYYY-MM-DD HH:MM:SS``).
    measurements : tuple of SpecMeasurementRecord
        Compatible stored measurements, oldest first; empty when measurements
        were not requested or none match.
    """

    spec_id: str
    declaration_version_id: str
    owner_path: str
    spec_key: str
    name: str
    metric: str
    op: str
    target: float
    units: str
    description: str
    parent_id: Optional[str]
    parent_owner_path: Optional[str]
    parent_spec_key: Optional[str]
    declared_at: str
    definition_scheme: str
    definition_hash: str
    created_at: str
    measurements: Tuple[SpecMeasurementRecord, ...]


@dataclass(frozen=True)
class CharacterizationEvidenceRecord:
    """One exact profile and operating-point evidence selection for a live snapshot.

    Returned by :meth:`silk.db.Database.select_characterization_evidence` and
    :meth:`silk.db.CircuitTracker.select_characterization_evidence`. It bundles
    the profile, the metrics, the matching spec measurements and the selected
    current set that were stored for one profile at one exact operating point
    (corner, temperature and voltage).

    Attributes
    ----------
    target_version_id : str
        Version id of the snapshot the selection was made for.
    origin_version_id : str
        Version id of the (possibly ancestor) snapshot that holds the evidence.
    profile_id : str
        Id of the resolved characterization profile.
    current_set_id : str
        Id of the selected current set.
    corner : str or ConditionState
        Process corner name or ``ConditionState.NONE``; a profile is never
        unknown or sweep.
    temperature : float or ConditionState
        Temperature in degrees Celsius or ``ConditionState.NONE``.
    voltage : float or ConditionState
        Supply volts or ``ConditionState.NONE``.
    profile : CharacterizationProfileRecord
        The resolved profile.
    metrics : tuple of SimulationResultRecord
        Metric observations stored under the profile at this point.
    specs : tuple of SpecDeclarationRecord
        Declarations of the target snapshot that have measurements at this
        point; each carries only those measurements.
    current_set : NetCurrentSetRecord
        The selected current set at this point.
    deck_check : DeckCheck or None
        Aggregate deck comparison: ``'stale'`` if any nested metric or
        measurement is stale, otherwise ``'not checked'`` (current annotations
        are never frozen at launch). ``None`` on raw :class:`~silk.db.Database`
        reads. Ignored by equality.
    """

    target_version_id: str
    origin_version_id: str
    profile_id: str
    current_set_id: str
    # A value or ConditionState.NONE (a profile is never unknown or sweep).
    corner: ConditionScalar
    temperature: ConditionScalar
    voltage: ConditionScalar
    profile: CharacterizationProfileRecord
    metrics: Tuple[SimulationResultRecord, ...]
    specs: Tuple[SpecDeclarationRecord, ...]
    current_set: NetCurrentSetRecord

    deck_check: Optional[DeckCheck] = field(default=None, compare=False)


@dataclass(frozen=True)
class CharacterizationRestoreResult:
    """Outcome and stable provenance of one atomic restore into live objects.

    Returned by :meth:`silk.db.CircuitTracker.restore_characterization_evidence`.

    Attributes
    ----------
    target_version_id : str
        Version id of the snapshot the evidence was restored for.
    origin_version_id : str
        Version id of the snapshot that held the evidence.
    profile_id : str
        Id of the characterization profile that was restored.
    current_set_id : str
        Id of the current set that was restored.
    corner : str or ConditionState
        Process corner name or ``ConditionState.NONE``.
    temperature : float or ConditionState
        Temperature in degrees Celsius or ``ConditionState.NONE``.
    voltage : float or ConditionState
        Supply volts or ``ConditionState.NONE``.
    measurement_count : int
        Number of spec measurements attached to live specs.
    current_count : int
        Number of net or pin current records applied to the live module.
    evidence : CharacterizationEvidenceRecord
        The evidence selection that was restored.
    """

    target_version_id: str
    origin_version_id: str
    profile_id: str
    current_set_id: str
    # A value or ConditionState.NONE (a profile is never unknown or sweep).
    corner: ConditionScalar
    temperature: ConditionScalar
    voltage: ConditionScalar
    measurement_count: int
    current_count: int
    evidence: CharacterizationEvidenceRecord


__all__ = [
    "DeckCheck",
    "CharacterizationProfileRecord",
    "CharacterizationEvidenceRecord",
    "CharacterizationRestoreResult",
    "Fingerprint",
    "NetCurrentRecord",
    "NetCurrentSetRecord",
    "SnapshotIdentity",
    "SimulationResultRecord",
    "SpecDeclarationRecord",
    "SpecMeasurementRecord",
    "UNSCOPED_PROFILE",
]
