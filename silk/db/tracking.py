"""Module-bound database tracking.

:class:`CircuitTracker` holds the database logic that
:class:`~silk.circuit.Module` delegates to when a ``db_path`` was supplied.
Its public methods correspond one-to-one to the thin tracking wrappers on
``Module``.
"""

import os
from functools import wraps
import sqlite3
import time
import uuid
from numbers import Complex, Real
from typing import Dict, Tuple, Optional, Any

from silk.db._condition_columns import condition_columns
from silk.db.circuit_db import Database, DesignStage
from silk.db.context import CharacterizationContext
from silk.db.identity import (
    SPEC_FINGERPRINT_SCHEME,
    STRUCTURE_FINGERPRINT_SCHEME,
    _fingerprint_spec_set,
    _fingerprints_equal,
)
from silk.db.models import (
    CharacterizationRestoreResult,
    Fingerprint,
    SnapshotIdentity,
    UNSCOPED_PROFILE,
)
from silk.exceptions import DatabaseError, EvidenceMismatchError, EvidenceNotFoundError


def _deck_path(module, default):
    name = getattr(module, 'netlistname', None)
    folder = getattr(module, 'dir_folder', None)
    if name is None:
        return default
    return name if folder is None else folder + '/' + name


def _corner_value(value):
    """Reduce a Corner-like enum to its exact string value; strings and
    ``ConditionState`` members pass through unchanged (mirrors
    ``silk.db.context._validate_corner``). A plain-Enum ``NONE`` must never
    become the string ``'none'``: that would collide with a corner of that
    name (design 2026-10-01 operating-conditions-none, §2.4.1)."""
    if isinstance(value, str):
        return getattr(value, "value", value)
    return value


def _snapshot_scalars(snapshot):
    """``(corner, temperature, voltage)`` condition scalars of a snapshot."""
    return (
        snapshot.scalar('corner'),
        snapshot.scalar('temperature'),
        snapshot.scalar('supply'),
    )


# Metric-name-suffix → unit convention (silk.circuit.spec_report metrics are
# named like 'gain_dB', 'ugf_MHz' — no separate unit field). Conservative and
# case-sensitive on purpose: an unrecognised suffix falls back to '' rather
# than guessing wrong. Extend as new metric-naming conventions show up.
_UNIT_SUFFIXES = {
    'dB', 'Hz', 'kHz', 'MHz', 'GHz',
    'V', 'mV', 'uV',
    'A', 'mA', 'uA', 'nA',
    's', 'ms', 'us', 'ns', 'ps',
    'W', 'mW', 'uW',
    'pct', 'LSB', 'bits', 'ohm', 'kohm', 'VV',
}


def _write_transaction(method):
    """Give a tracker operation's SQL and live binding one rollback scope."""
    @wraps(method)
    def write(self, *args, **kwargs):
        with self.db.transaction():
            return method(self, *args, **kwargs)
    return write


class CircuitTracker:
    """Database tracking delegate bound to one live module.

    A tracker opens a :class:`~silk.db.Database`, derives the snapshot identity
    of its module, and creates or reuses the matching sealed snapshot
    (:meth:`create_circuit_version`). It then writes metrics, spec measurements
    and net currents against that bound snapshot, and reads them back checked
    against the live design. ``Module`` creates its tracker lazily on the first
    tracking call when it was constructed with a ``db_path``.

    Every write first checks that the live design still matches the bound
    snapshot and raises :class:`~silk.exceptions.EvidenceMismatchError` if it
    does not. Writes run inside one :meth:`silk.db.Database.transaction` scope, so a
    failing call leaves no rows and restores the previous binding.

    Parameters
    ----------
    module : Module
        The live module to track. The circuit name is its ``module_name``, or
        for a top-level module the stem of its netlist file name.
    db_path : str
        Path of the SQLite database, or ``':memory:'``.
    version : str, default "1.0"
        Descriptive version label for snapshots this tracker creates.
    stage : DesignStage, optional
        Design stage of snapshots this tracker creates; ``DesignStage.DEV`` when
        omitted.
    created_by : str, default "user"
        Author recorded with snapshots this tracker creates.

    Attributes
    ----------
    db : Database or None
        The open database; ``None`` after :meth:`close`.
    circuit_name : str
        Circuit name used for snapshots.
    version : str
        Version label of the bound snapshot, once one is bound.
    stage : DesignStage
        Stage of the bound snapshot, once one is bound.
    created_by : str
        Author recorded with new snapshots.
    circuit_id : str or None
        Version id of the bound snapshot; ``None`` until
        :meth:`create_circuit_version` or :meth:`ensure_version` binds one.

    See Also
    --------
    silk.db.Database : The storage layer this class drives.
    """

    def __init__(self, module, db_path, version="1.0",
                 stage=None, created_by="user"):
        self._module = module
        self.db = Database(db_path)
        if module.module_name:
            self.circuit_name = module.module_name
        else:
            # Top-level netlist: derive name from filename (e.g. "ldo_example.cir" → "ldo_example")
            basename = os.path.basename(getattr(module, 'netlistname', 'circuit.cir'))
            self.circuit_name = os.path.splitext(basename)[0]
        self.version = version
        self.stage = stage or DesignStage.DEV
        self.created_by = created_by
        self.circuit_id = None

        # Simulation metadata
        self._current_corner = "tt"
        self._current_temp = None
        self._current_voltage = None

    # -----------------------------------------------------------------
    # Database integration methods
    # -----------------------------------------------------------------

    def _require_write_identity(self, module, version_id):
        """Validate the live owner of an explicit evidence target before writing."""
        if version_id is None:
            raise DatabaseError(
                "No circuit version created. Call create_circuit_version() first."
            )
        snapshot = self._prepare_snapshot(module)
        self.db.require_version_identity(
            version_id, snapshot[2], include_spec_set=True
        )
        return snapshot

    def _bind_version(self, version_id):
        """Publish the SQL-visible binding, with a scope-owned tuple restore."""
        row = self.db._snapshot_row(version_id)
        binding = (version_id, DesignStage(row['stage']), row['version'])
        previous = (self.circuit_id, self.stage, self.version)

        def restore():
            self.circuit_id, self.stage, self.version = previous

        self.db._after_rollback(restore)
        self.circuit_id, self.stage, self.version = binding

    def create_circuit_version(
        self,
        description="",
        netlist_path=None,
        *,
        parent_version_id=None,
    ):
        """Create or reuse the sealed snapshot of the tracked module and bind to it.

        The snapshot identity (structure plus spec declarations) of the live module
        is computed; if a sealed snapshot with that identity exists at the
        tracker's stage it is reused, otherwise a new one is created, filled and
        sealed. Within a grouped :meth:`silk.db.Database.transaction`, a rollback restores
        the previous binding.

        Parameters
        ----------
        description : str, default ""
            Description stored with a new snapshot.
        netlist_path : str, optional
            Netlist path stored with a new snapshot; the module's configured
            folder joined with its deck file name (or ``'circuit.cir'``) when omitted.
        parent_version_id : str, optional
            Parent snapshot (keyword-only). It must be sealed and of the same
            circuit; an existing snapshot with a different parent is refused.

        Returns
        -------
        str
            The version id of the bound snapshot.

        Raises
        ------
        silk.exceptions.DatabaseError
            The parent is invalid, or an existing matching snapshot has a different
            parent.
        """
        if netlist_path is None:
            netlist_path = _deck_path(self._module, 'circuit.cir')
        return self._ensure_version(
            self._module,
            circuit_name=self.circuit_name,
            parent_version_id=parent_version_id,
            description=description,
            netlist_path=netlist_path,
            bind=True,
        )

    def _prepare_snapshot(self, module):
        """Extract all identity-bearing data before any write transaction."""
        from silk.db.structure_tracker import CircuitStructureTracker

        tracker = CircuitStructureTracker(self.db, None)
        structure = tracker.extract_structure(module)
        structure_hash, module_hashes = tracker.compute_structure_hash(structure)
        spec_set, definitions = _fingerprint_spec_set(module)
        identity = SnapshotIdentity(
            structure=Fingerprint(
                STRUCTURE_FINGERPRINT_SCHEME, structure_hash
            ),
            spec_set=spec_set,
        )
        return structure, module_hashes, identity, definitions

    def compute_snapshot_identity_for(self, module):
        """Return the structure and spec-set identity of a module.

        Parameters
        ----------
        module : Module
            Any live module, not necessarily the tracked one.

        Returns
        -------
        SnapshotIdentity
            The structure identity and the identity of the module's spec
            declarations.
        """
        _structure, _modules, identity, _definitions = self._prepare_snapshot(
            module
        )
        return identity

    @_write_transaction
    def _ensure_version(
        self,
        module,
        *,
        circuit_name,
        parent_version_id,
        description,
        netlist_path,
        bind,
    ):
        from silk.db.structure_tracker import CircuitStructureTracker

        structure, module_hashes, identity, definitions = self._prepare_snapshot(
            module
        )
        existing = self.resolve_version(circuit_name, identity)
        if existing is not None:
            if parent_version_id is not None:
                self.db._validate_parent(
                    parent_version_id, circuit_name=circuit_name
                )
                existing_parent = self.db._snapshot_row(existing)[
                    "parent_version_id"
                ]
                if existing_parent != parent_version_id:
                    from silk.exceptions import DatabaseError
                    raise DatabaseError(
                        f"Existing snapshot {existing!r} has parent "
                        f"{existing_parent!r}, not requested parent "
                        f"{parent_version_id!r}"
                    )
            if bind:
                self._bind_version(existing)
            return existing

        try:
            with self.db.transaction():
                version_id = self.db.create_circuit(
                    name=circuit_name,
                    version=self.version,
                    stage=self.stage,
                    created_by=self.created_by,
                    description=description,
                    netlist_path=netlist_path,
                    identity=identity,
                    parent_version_id=parent_version_id,
                )
                structure_tracker = CircuitStructureTracker(self.db, version_id)
                structure_tracker.store_structure(
                    structure, identity.structure.digest, module_hashes
                )
                self.db.store_spec_declarations(version_id, definitions)
                self.db.seal_circuit_version(version_id)
        except sqlite3.IntegrityError:
            existing = self.resolve_version(circuit_name, identity)
            if existing is None:
                raise
            version_id = existing

        if bind:
            self._bind_version(version_id)
        return version_id

    def ensure_version(
        self,
        module,
        *,
        circuit_name=None,
        parent_version_id=None,
    ):
        """Create or reuse the sealed snapshot for a module and bind to it.

        Like :meth:`create_circuit_version` but for an explicit module and circuit
        name; the description is empty.

        Parameters
        ----------
        module : Module
            The live module to snapshot.
        circuit_name : str, optional
            Circuit name; the tracker's own name when omitted (keyword-only).
        parent_version_id : str, optional
            Parent snapshot (keyword-only).

        Returns
        -------
        str
            The version id of the snapshot, which is now the bound one.

        Raises
        ------
        silk.exceptions.DatabaseError
            The parent is invalid, or an existing matching snapshot has a different
            parent.
        """
        return self._ensure_version(
            module,
            circuit_name=circuit_name or self.circuit_name,
            parent_version_id=parent_version_id,
            description="",
            netlist_path=_deck_path(module, None),
            bind=True,
        )

    def compute_structure_hash_for(self, module):
        """Return the structure digest of an arbitrary module.

        Parameters
        ----------
        module : Module
            Any live module, not necessarily the tracked one.

        Returns
        -------
        str
            Lowercase hexadecimal SHA-256 structure digest (scheme
            ``'silk.structure/v4'``).
        """
        from silk.db.structure_tracker import CircuitStructureTracker
        tracker = CircuitStructureTracker(self.db, None)
        structure = tracker.extract_structure(module)
        structure_hash, _module_hashes = tracker.compute_structure_hash(structure)
        return structure_hash

    def resolve_version(self, circuit_name, identity, *, stage=None):
        """Find the sealed snapshot with an exact identity without creating one.

        Parameters
        ----------
        circuit_name : str
            Circuit name.
        identity : SnapshotIdentity
            Identity to match exactly.
        stage : DesignStage, optional
            Stage to search (keyword-only); the tracker's own stage when omitted.

        Returns
        -------
        str or None
            The version id, or ``None`` when no sealed snapshot matches.
        """
        return self.db.find_snapshot(
            circuit_name,
            stage or self.stage,
            identity,
        )

    @_write_transaction
    def store_results(self, results_dict, corner=None, temperature=None,
                      voltage=None, sim_type=None, tool_name=None,
                      source="user", conditions=None):
        """Store several results on the bound snapshot, atomically.

        Parameters
        ----------
        results_dict : dict
            Maps a parameter name to a ``(value, unit)`` pair. A string value is
            stored as a text observation, any other value as a number.
        corner : str or ConditionState, optional
            Process corner; the tracker's current corner (see :meth:`set_corner`,
            initially ``'tt'``) when omitted.
        temperature : float, optional
            Temperature in degrees Celsius; the tracker's current temperature when
            omitted.
        voltage : float, optional
            Supply voltage in volts; the tracker's current voltage when omitted.
        sim_type : str, optional
            Analysis label.
        tool_name : str, optional
            Producing tool.
        source : str, default "user"
            Source label.
        conditions : dict, optional
            JSON-compatible dictionary of extra conditions.

        Raises
        ------
        silk.exceptions.DatabaseError
            No snapshot is bound, or the bound snapshot is not sealed.
        silk.exceptions.EvidenceMismatchError
            The live design no longer matches the bound snapshot.
        silk.exceptions.LockedVersionError
            The bound snapshot is locked.
        ValueError, TypeError
            As :meth:`silk.db.Database.add_simulation_result` for invalid values.
        """
        self._require_write_identity(self._module, self.circuit_id)

        corner = corner or self._current_corner
        temperature = temperature if temperature is not None else self._current_temp
        voltage = voltage if voltage is not None else self._current_voltage

        with self.db.transaction():
            for param, (result_value, unit) in results_dict.items():
                if isinstance(result_value, str):
                    value = None
                    value_text = result_value
                else:
                    value = result_value
                    value_text = None
                self.db.add_simulation_result(
                    circuit_version_id=self.circuit_id,
                    parameter=param,
                    value=value,
                    value_text=value_text,
                    unit=unit,
                    corner=corner,
                    temperature=temperature,
                    voltage=voltage,
                    sim_type=sim_type,
                    tool_name=tool_name,
                    source=source,
                    conditions=conditions,
                )

    def store_result(self, parameter, value, unit, corner=None,
                     temperature=None, voltage=None, sim_type=None,
                     tool_name=None, source="user", conditions=None):
        """Store one result on the bound snapshot.

        A convenience wrapper around :meth:`store_results`.

        Parameters
        ----------
        parameter : str
            Parameter name.
        value : float or str
            Numeric value, or a string for a text observation.
        unit : str
            Unit label.
        corner : str or ConditionState, optional
            Process corner; the tracker's current corner when omitted.
        temperature : float, optional
            Temperature in degrees Celsius.
        voltage : float, optional
            Supply voltage in volts.
        sim_type : str, optional
            Analysis label.
        tool_name : str, optional
            Producing tool.
        source : str, default "user"
            Source label.
        conditions : dict, optional
            JSON-compatible dictionary of extra conditions.

        Raises
        ------
        silk.exceptions.DatabaseError, silk.exceptions.EvidenceMismatchError, silk.exceptions.LockedVersionError
            As :meth:`store_results`.
        """
        self.store_results(
            {parameter: (value, unit)},
            corner=corner, temperature=temperature, voltage=voltage,
            sim_type=sim_type, tool_name=tool_name, source=source,
            conditions=conditions,
        )

    @_write_transaction
    def store_pass_fail(self, spec_name, status, details="", *,
                        source="user", conditions=None):
        """Store a pass/fail status for a named spec as a text observation.

        The observation is stored under the parameter ``'spec_<spec_name>'`` with the
        text ``'<status>'`` or ``'<status> (<details>)'``.

        Parameters
        ----------
        spec_name : str
            Spec name.
        status : str
            Status text, for example ``'pass'`` or ``'fail'``.
        details : str, default ""
            Optional detail appended in parentheses.
        source : str, default "user"
            Source label (keyword-only).
        conditions : dict, optional
            JSON-compatible dictionary of extra conditions (keyword-only).

        Raises
        ------
        silk.exceptions.DatabaseError, silk.exceptions.EvidenceMismatchError, silk.exceptions.LockedVersionError
            As :meth:`store_results`.
        """
        self._require_write_identity(self._module, self.circuit_id)
        value_text = f"{status}"
        if details:
            value_text += f" ({details})"
        self.db.add_simulation_result(
            circuit_version_id=self.circuit_id,
            parameter=f"spec_{spec_name}",
            value_text=value_text,
            source=source,
            conditions=conditions,
        )

    def query_results(
        self,
        *,
        parameter=None,
        source=None,
        sim_type=None,
        corner=None,
        temperature=None,
        voltage=None,
        tool_name=None,
        conditions=None,
        profile_id=None,
        latest_only=False,
        include_ancestors=True,
    ):
        """Read stored metrics compatible with the bound live design.

        The live structure must still match the bound snapshot. Records come from
        the snapshot and, by default, its structure-compatible ancestors, and each
        carries a ``deck_check`` comparing its producing deck with the live bench.
        Stale history is flagged rather than hidden, except that with
        ``latest_only`` a selected stale row raises. Rows that could not be checked
        stay usable and produce one combined :class:`UserWarning`.

        Parameters
        ----------
        parameter : str, optional
            Metric name. All filters are keyword-only and ``None`` does not
            restrict.
        source : str, optional
            Source label.
        sim_type : str, optional
            Analysis label.
        corner : str or ConditionState, optional
            Process corner.
        temperature : float or ConditionState, optional
            Temperature in degrees Celsius.
        voltage : float or ConditionState, optional
            Supply voltage in volts.
        tool_name : str, optional
            Producing tool.
        conditions : dict, optional
            Exact extra-conditions dictionary to match.
        profile_id : str or UNSCOPED_PROFILE, optional
            ``None`` applies no profile filter, :data:`~silk.db.UNSCOPED_PROFILE`
            keeps only unscoped rows, a string keeps only that profile.
        latest_only : bool, default False
            Keep only the newest observation per distinct context.
        include_ancestors : bool, default True
            Also read structure-compatible ancestors.

        Returns
        -------
        list of SimulationResultRecord
            Non-empty list of matching records, ordered as
            :meth:`silk.db.Database.get_simulation_results` orders them.

        Raises
        ------
        RuntimeError
            No snapshot is bound.
        silk.exceptions.EvidenceNotFoundError
            No compatible metric evidence is stored for the filters.
        silk.exceptions.EvidenceMismatchError
            Matching evidence exists only for an incompatible structure, the live
            design differs from the bound snapshot, or (with ``latest_only``) a
            selected row is stale.
        """
        if self.circuit_id is None:
            raise RuntimeError(
                "No circuit version created. Call create_circuit_version() first."
            )
        live_identity = self.compute_snapshot_identity_for(self._module)
        self.db.require_version_identity(
            self.circuit_id, live_identity, include_spec_set=False
        )
        filters = {
            "parameter": parameter,
            "source": source,
            "sim_type": sim_type,
            "corner": corner,
            "temperature": temperature,
            "voltage": voltage,
            "tool_name": tool_name,
            "conditions": conditions,
            "profile_id": profile_id,
            "include_ancestors": include_ancestors,
        }
        records = self.db.get_simulation_results(
            self.circuit_id,
            latest_only=latest_only,
            **filters,
        )
        if records:
            from silk.db._deck_check import CheckContext
            check = CheckContext(self)
            records = [check.row(row) for row in records]
            if latest_only:
                check.refuse_stale(records)
            check.warn()
            return records
        if self.db._has_incompatible_simulation_results(
            self.circuit_id, **filters
        ):
            raise EvidenceMismatchError(
                "Matching metric context exists only for an incompatible "
                "structure identity"
            )
        requested = ", ".join(
            f"{key}={value!r}"
            for key, value in filters.items()
            if key != "include_ancestors" and value is not None
        )
        raise EvidenceNotFoundError(
            "No compatible metric evidence is stored"
            + (f" for context {requested}" if requested else "")
        )

    def get_stored_specs(
        self,
        *,
        include_measurements=True,
        include_ancestors=True,
        profile_id=None,
    ):
        """Return the stored spec declarations of the bound snapshot.

        Parameters
        ----------
        include_measurements : bool, default True
            Attach stored measurements to each declaration (keyword-only).
        include_ancestors : bool, default True
            Also read measurements stored on ancestor snapshots.
        profile_id : str or UNSCOPED_PROFILE, optional
            Filter on which measurements are attached, as
            :meth:`silk.db.Database.get_stored_specs`.

        Returns
        -------
        list of SpecDeclarationRecord
            The declarations, with the filtered measurements attached.

        Raises
        ------
        RuntimeError
            No snapshot is bound.
        """
        if self.circuit_id is None:
            raise RuntimeError(
                "No circuit version created. Call create_circuit_version() first."
            )
        return self.db.get_stored_specs(
            self.circuit_id,
            include_measurements=include_measurements,
            include_ancestors=include_ancestors,
            profile_id=profile_id,
        )

    def _require_bound_read_identity(self, *, include_spec_set):
        if self.circuit_id is None:
            raise RuntimeError(
                "No circuit version created. Call create_circuit_version() "
                "first."
            )
        live_identity = self.compute_snapshot_identity_for(self._module)
        self.db.require_version_identity(
            self.circuit_id,
            live_identity,
            include_spec_set=include_spec_set,
        )
        return self.circuit_id

    def list_characterization_profiles(self, *, include_ancestors=True):
        """List the profiles compatible with the bound live structure.

        Parameters
        ----------
        include_ancestors : bool, default True
            Also read ancestors of the bound snapshot (keyword-only).

        Returns
        -------
        list of CharacterizationProfileRecord
            Profiles, nearest snapshot first.

        Raises
        ------
        RuntimeError
            No snapshot is bound.
        silk.exceptions.EvidenceMismatchError
            The live structure differs from the bound snapshot.
        """
        version_id = self._require_bound_read_identity(include_spec_set=False)
        return self.db.list_characterization_profiles(
            version_id, include_ancestors=include_ancestors
        )

    def resolve_characterization_profile(
        self,
        *,
        profile_id=None,
        context=None,
        include_ancestors=True,
    ):
        """Resolve a profile by exact id or complete descriptor.

        Parameters
        ----------
        profile_id : str, optional
            Exact profile id. Give exactly one of ``profile_id`` and ``context``
            (both keyword-only).
        context : CharacterizationContext, optional
            Complete descriptor.
        include_ancestors : bool, default True
            Also search ancestors of the bound snapshot.

        Returns
        -------
        CharacterizationProfileRecord
            The matching profile.

        Raises
        ------
        RuntimeError
            No snapshot is bound.
        silk.exceptions.EvidenceMismatchError
            The live structure differs from the bound snapshot, or the profile is
            incompatible.
        silk.exceptions.EvidenceNotFoundError
            No compatible profile exists.
        ValueError, TypeError
            As :meth:`silk.db.Database.resolve_characterization_profile`.
        """
        version_id = self._require_bound_read_identity(include_spec_set=False)
        return self.db.resolve_characterization_profile(
            version_id,
            profile_id=profile_id,
            context=context,
            include_ancestors=include_ancestors,
        )

    def select_characterization_evidence(
        self,
        *,
        profile_id=None,
        context=None,
        corner=None,
        temperature=None,
        voltage=None,
        include_ancestors=True,
    ):
        """Select the evidence of one profile at one exact operating point.

        Behaves like :meth:`silk.db.Database.select_characterization_evidence` for the
        bound snapshot, but additionally requires the live structure and spec
        declarations to match it, and returns records flagged with ``deck_check``.
        Stale rows are flagged, not refused. Because current annotations are not
        frozen at launch, a warning for unchecked currents is always emitted once.

        Parameters
        ----------
        profile_id : str, optional
            Exact profile id. Give exactly one of ``profile_id`` and ``context``
            (all parameters are keyword-only).
        context : CharacterizationContext, optional
            Complete descriptor; its corner, temperature and voltage select the
            point.
        corner : str or ConditionState, optional
            Process corner (required with ``profile_id``).
        temperature : float or ConditionState, optional
            Temperature in degrees Celsius.
        voltage : float or ConditionState, optional
            Supply voltage in volts.
        include_ancestors : bool, default True
            Also search ancestors of the bound snapshot.

        Returns
        -------
        CharacterizationEvidenceRecord
            The selected evidence with ``deck_check`` set on it and on its nested
            records.

        Raises
        ------
        RuntimeError
            No snapshot is bound.
        silk.exceptions.EvidenceMismatchError
            The live design differs from the bound snapshot, or the evidence is
            incompatible.
        silk.exceptions.EvidenceNotFoundError, silk.exceptions.EvidenceAmbiguityError, ValueError, TypeError
            As :meth:`silk.db.Database.select_characterization_evidence`.
        """
        version_id = self._require_bound_read_identity(include_spec_set=True)
        evidence = self.db.select_characterization_evidence(
            version_id,
            profile_id=profile_id,
            context=context,
            corner=corner,
            temperature=temperature,
            voltage=voltage,
            include_ancestors=include_ancestors,
        )
        from silk.db._deck_check import CheckContext
        check = CheckContext(self)
        evidence = check.aggregate(evidence)
        check.warn()
        return evidence

    def _check_swept_measurement_attribution(self, candidates):
        """Prove stored sweep links before scalar restoration can collapse points."""
        import json
        from silk._json_codec import encode_json
        candidates = list(candidates)
        if not candidates:
            return
        identities = {(r.origin_version_id, r.origin_spec_id, r.measurement_id): (location, r)
                      for location, r in candidates}
        fields = ('source', 'corner', 'temperature', 'voltage', 'profile_id',
                  'sim_type', 'tool_name', 'provenance_id')
        scopes = {(r.origin_version_id, *(getattr(r, name) for name in fields))
                  for location, r in candidates}
        declarations = self.db.get_stored_specs(
            self.circuit_id, include_measurements=True, include_ancestors=True)
        stored = {(r.origin_version_id, r.origin_spec_id, r.measurement_id): r
                  for declaration in declarations for r in declaration.measurements}
        owners = {(d.declaration_version_id, d.spec_id): d for d in declarations}
        attribution = {}
        rows = self.db.get_simulation_results(self.circuit_id, include_ancestors=True)
        for row in rows:
            conditions = row.conditions or {}
            if 'silk_run_variables' not in conditions or row.value_text is None:
                continue
            scope = (row.origin_version_id, *(getattr(row, name) for name in fields))
            link = conditions.get('silk_spec_measurement')
            identity_fields = {'origin_version_id', 'origin_spec_id', 'measurement_id'}
            if not isinstance(link, dict) or set(link) not in (
                    identity_fields, identity_fields | {'has_measurement'}):
                if scope not in scopes:
                    continue
                raise EvidenceMismatchError('Variable attribution has no provable spec measurement link')
            if (any(not isinstance(link[name], str) or not link[name]
                    for name in ('origin_version_id', 'origin_spec_id'))
                    or (link['measurement_id'] is not None
                        and (not isinstance(link['measurement_id'], str) or not link['measurement_id']))):
                if scope in scopes:
                    raise EvidenceMismatchError('Variable attribution has an invalid spec measurement identity')
                continue
            identity = (link['origin_version_id'], link['origin_spec_id'], link['measurement_id'])
            if link.get('has_measurement', True) is not (link['measurement_id'] is not None):
                if scope in scopes or identity in identities:
                    raise EvidenceMismatchError('Variable attribution has no provable spec measurement link')
                continue
            if link['measurement_id'] is None:
                if scope not in scopes:
                    continue
                if row.origin_version_id != identity[0]:
                    raise EvidenceMismatchError('Variable attribution has an invalid measurement-free status')
                if identity[:2] not in owners:
                    origin_declarations = self.db.get_stored_specs(
                        identity[0], include_measurements=True, include_ancestors=False)
                    compatible = {(d.owner_path, d.spec_key, d.definition_scheme, d.definition_hash)
                                  for d in declarations}
                    owners.update(((d.declaration_version_id, d.spec_id), d)
                                  for d in origin_declarations
                                  if (d.owner_path, d.spec_key, d.definition_scheme, d.definition_hash)
                                  in compatible)
                declaration = owners.get(identity[:2])
                if declaration is None:
                    raise EvidenceMismatchError('Variable attribution has an invalid measurement-free status')
                continue
            record = stored.get(identity)
            if record is None:
                if scope not in scopes:
                    continue
                raise EvidenceMismatchError('Variable attribution references an unavailable spec measurement')
            if row.origin_version_id != record.origin_version_id or any(
                    getattr(row, name) != getattr(record, name) for name in fields):
                if identity in identities or scope in scopes:
                    raise EvidenceMismatchError('Variable attribution conflicts with its spec measurement evidence')
                continue
            if identity not in identities:
                continue
            location, record = identities[identity]
            marker = conditions['silk_run_variables']
            if not isinstance(marker, dict) or not marker:
                raise EvidenceMismatchError('Variable attribution must be a nonempty value map')
            encoded = json.dumps(encode_json(marker, finite_only=True),
                                 sort_keys=True, allow_nan=False)
            if identity in attribution and attribution[identity] != encoded:
                raise EvidenceMismatchError('Conflicting Variable attribution for one spec measurement')
            attribution[identity] = encoded
        slots = {}
        for location, record in candidates:
            identity = (record.origin_version_id, record.origin_spec_id, record.measurement_id)
            slot = (location, record.source, record.corner)
            if record.profile_id is not None:
                slot += (record.profile_id, record.temperature, record.voltage)
            marker = attribution.get(identity)
            if slot in slots and slots[slot] != marker:
                raise EvidenceMismatchError(
                    'Ambiguous Variable attribution: scalar spec restoration cannot select a swept point')
            slots[slot] = marker

    def _stage_characterization_measurements(self, evidence):
        """Validate selected measurements and build live values without apply."""
        from silk.specs import Measurement

        bindings = self._validated_live_spec_bindings()
        self._check_swept_measurement_attribution(
            ((declaration.owner_path, declaration.spec_key), record)
            for declaration in evidence.specs for record in declaration.measurements)
        newest = {}
        for declaration in evidence.specs:
            location = (declaration.owner_path, declaration.spec_key)
            if location not in bindings:
                raise EvidenceMismatchError(
                    "Selected spec measurement has no matching live "
                    f"declaration at {location[0]}/{location[1]}"
                )
            for record in declaration.measurements:
                key = (
                    location,
                    record.profile_id,
                    record.source,
                    record.corner,
                    record.temperature,
                    record.voltage,
                )
                newest[key] = (location, record)

        staged = []
        for location, record in newest.values():
            live_spec, _declaration = bindings[location]
            measurement = Measurement(
                source=record.source,
                value=record.value,
                passes=None,
                corner=record.corner,
                timestamp=(
                    0.0 if record.timestamp is None else record.timestamp
                ),
                sim_run_id=record.sim_run_id,
                notes=record.notes,
                aggregate=record.aggregate,
                temperature=record.temperature,
                voltage=record.voltage,
                sim_type=record.sim_type,
                tool_name=record.tool_name,
                origin_version_id=record.origin_version_id,
                profile_id=record.profile_id,
            )
            evaluated = bool(live_spec.evaluate(measurement))
            if record.passes is not None and record.passes != evaluated:
                raise EvidenceMismatchError(
                    f"Stored spec measurement {record.measurement_id!r} "
                    "cached pass does not match the live declaration"
                )
            measurement.passes = evaluated
            from silk.specs.evidence import stage_for_root
            stage_for_root(self._module, live_spec, measurement, location[0])
            from silk.db._attempts import hydrate
            hydrate(self._module, live_spec, location[0], record.measurement_id, measurement)
            staged.append((live_spec, measurement))
        return staged

    @staticmethod
    def _current_value(row, name):
        if hasattr(row, name):
            return getattr(row, name)
        return row[name]

    def _stage_current_restore(self, rows, *, force, error_type=RuntimeError, owner_path="$"):
        """Resolve all selected current destinations before live mutation."""
        module = self._module
        if owner_path != '$':
            module = next(instance.module for instance in self._module.iter_instances()
                          if instance.owner_path == owner_path)
        pins_by_id = {}
        for net in module.net_dict.values():
            for binding in net.pins:
                pins_by_id[(net.name, f"{binding.instance}.{binding.port}")] = (
                    binding
                )
        actions = []
        for row in rows:
            net_name = self._current_value(row, "net_name")
            pin_id = self._current_value(row, "pin_id")
            if net_name not in module.net_dict:
                raise error_type(
                    f"load_net_currents: stored net {net_name!r} does not "
                    "exist on the live module; structure drift"
                )
            if pin_id != "" and (net_name, pin_id) not in pins_by_id:
                raise error_type(
                    f"load_net_currents: stored pin {pin_id!r} on net "
                    f"{net_name!r} matches no binding on the live module"
                )
            target = (
                module.net_dict[net_name]
                if pin_id == ""
                else pins_by_id[(net_name, pin_id)]
            )
            if target._current_source in ("user", "spice") and not force:
                continue
            actions.append((target, row, pin_id != ""))
        return actions

    def _apply_current_restore(self, actions, *, owner_path="$"):
        """Apply a fully staged current set; this phase cannot validate/fail."""
        module = self._module
        if owner_path != '$':
            from silk.circuit.test_placement import evidence_index
            annotations = evidence_index(module).annotations
            for key in list(annotations):
                if key[0] == owner_path and key[1].startswith('net.'):
                    del annotations[key]
            for _, row, is_pin in actions:
                if not is_pin:
                    name = self._current_value(row, 'net_name')
                    for attribute, column in [('avg_current', 'avg_a'),
                                              ('peak_current', 'peak_a'),
                                              ('rms_current', 'rms_a')]:
                        value = self._current_value(row, column)
                        if value is not None:
                            annotations[(owner_path, f'net.{name}.{attribute}')] = value
            return len(actions)
        for net in module.net_dict.values():
            if net._current_source == "db":
                net.avg_current = None
                net.rms_current = None
                net.peak_current = None
                net._current_source = None
            for binding in net.pins:
                if binding._current_source == "db":
                    binding.avg_current = None
                    binding.rms_current = None
                    binding.peak_current = None
                    binding.current_vector = None
                    binding._current_source = None
        for target, row, is_pin in actions:
            target.avg_current = self._current_value(row, "avg_a")
            target.rms_current = self._current_value(row, "rms_a")
            target.peak_current = self._current_value(row, "peak_a")
            if is_pin:
                target.current_vector = self._current_value(
                    row, "current_vector"
                )
            target._current_source = "db"
        return len(actions)

    def restore_characterization_evidence(
        self,
        *,
        profile_id=None,
        context=None,
        corner=None,
        temperature=None,
        voltage=None,
        include_ancestors=True,
        force=False,
    ):
        """Restore selected profile evidence into the live specs and nets.

        The evidence is selected as in :meth:`select_characterization_evidence`,
        all of its producing decks are checked, and only then are measurements
        attached to live specs and currents applied to the live module, as one
        operation. A stale row raises before any live object is changed and before
        any warning. Rows that could not be checked stay usable and one combined
        warning, which always covers the current set, is emitted.

        Parameters
        ----------
        profile_id : str, optional
            Exact profile id. Give exactly one of ``profile_id`` and ``context``
            (all parameters are keyword-only).
        context : CharacterizationContext, optional
            Complete descriptor.
        corner : str or ConditionState, optional
            Process corner (required with ``profile_id``).
        temperature : float or ConditionState, optional
            Temperature in degrees Celsius.
        voltage : float or ConditionState, optional
            Supply voltage in volts.
        include_ancestors : bool, default True
            Also search ancestors of the bound snapshot.
        force : bool, default False
            Overwrite live currents whose source is ``'user'`` or ``'spice'``;
            otherwise those are kept and only database-sourced or empty ones are
            written.

        Returns
        -------
        CharacterizationRestoreResult
            What was selected and how many measurements and currents were applied.

        Raises
        ------
        RuntimeError
            No snapshot is bound.
        silk.exceptions.EvidenceMismatchError
            A selected row is stale, the live design differs from the bound
            snapshot, a stored net or pin has no live counterpart, or a stored
            verdict disagrees with the live declaration.
        silk.exceptions.EvidenceNotFoundError, silk.exceptions.EvidenceAmbiguityError, ValueError, TypeError
            As :meth:`silk.db.Database.select_characterization_evidence`.
        """
        version_id = self._require_bound_read_identity(include_spec_set=True)
        evidence = self.db.select_characterization_evidence(
            version_id,
            profile_id=profile_id,
            context=context,
            corner=corner,
            temperature=temperature,
            voltage=voltage,
            include_ancestors=include_ancestors,
        )
        from silk.db._deck_check import CheckContext
        check = CheckContext(self)
        evidence = check.aggregate(evidence)
        check.refuse_stale([evidence])
        check.warn()
        staged_measurements = self._stage_characterization_measurements(
            evidence
        )
        owner = evidence.profile.owner_path
        # Profiles can name an external characterization owner on a standalone
        # definition. Route to a placement only when that path exists here.
        if owner not in {i.owner_path for i in self._module.iter_instances()}:
            owner = '$'
        staged_currents = self._stage_current_restore(
            evidence.current_set.currents,
            force=force,
            error_type=EvidenceMismatchError,
            owner_path=owner,
        )

        for live_spec, measurement in staged_measurements:
            live_spec.record_measurement(measurement)
        current_count = self._apply_current_restore(staged_currents, owner_path=owner)
        return CharacterizationRestoreResult(
            target_version_id=evidence.target_version_id,
            origin_version_id=evidence.origin_version_id,
            profile_id=evidence.profile_id,
            current_set_id=evidence.current_set_id,
            corner=evidence.corner,
            temperature=evidence.temperature,
            voltage=evidence.voltage,
            measurement_count=len(staged_measurements),
            current_count=current_count,
            evidence=evidence,
        )

    def _validated_live_spec_bindings(self, *, writing=False):
        """Map stable declaration locations to live Specs and stored rows."""
        if writing:
            _structure, _modules, identity, definitions = self._require_write_identity(
                self._module, self.circuit_id
            )
        else:
            if self.circuit_id is None:
                raise RuntimeError(
                    "No circuit version created. Call create_circuit_version() first."
                )
            _structure, _modules, identity, definitions = self._prepare_snapshot(
                self._module
            )
            self.db.require_version_identity(
                self.circuit_id, identity, include_spec_set=True
            )
        declarations = self.db.get_stored_specs(
            self.circuit_id,
            include_measurements=False,
            include_ancestors=False,
        )
        live_by_location = {
            (definition.owner_path, definition.key): definition
            for definition in definitions
        }
        stored_by_location = {
            (record.owner_path, record.spec_key): record
            for record in declarations
        }
        if set(live_by_location) != set(stored_by_location):
            raise EvidenceMismatchError(
                "Live spec declarations do not match the bound snapshot"
            )

        bindings = {}
        for location, definition in live_by_location.items():
            record = stored_by_location[location]
            if not _fingerprints_equal(
                definition.fingerprint,
                Fingerprint(record.definition_scheme, record.definition_hash),
                expected_scheme=SPEC_FINGERPRINT_SCHEME,
            ):
                raise EvidenceMismatchError(
                    f"Live spec {location[0]}/{location[1]} has a different "
                    "definition from the bound snapshot"
                )
            bindings[location] = (definition.spec, record)
        return bindings

    def restore_spec_measurements(
        self,
        *,
        source=None,
        corner=None,
        temperature=None,
        voltage=None,
        tool_name=None,
        profile_id=None,
        include_ancestors=True,
    ):
        """Attach stored spec observations to the live specs, atomically.

        Matching stored measurements are checked against their producing decks and
        staged; a stale selected observation raises before anything is changed.
        Observations that could not be checked stay usable and produce one combined
        warning. Within the selection the newest wins per live slot: unscoped
        observations per (spec, source, corner), scoped ones per (spec, profile,
        source, corner, temperature, voltage), so two profiles' measurements at the
        same corner both survive.

        Parameters
        ----------
        source : str, optional
            Keep only this source. All filters are keyword-only; ``None`` does not
            restrict.
        corner : str or ConditionState, optional
            Keep only this corner.
        temperature : float or ConditionState, optional
            Keep only this temperature (degrees Celsius).
        voltage : float or ConditionState, optional
            Keep only this voltage.
        tool_name : str, optional
            Keep only this producing tool.
        profile_id : str or UNSCOPED_PROFILE, optional
            ``None`` matches scoped and unscoped observations,
            :data:`~silk.db.UNSCOPED_PROFILE` only unscoped ones, a string only that
            profile.
        include_ancestors : bool, default True
            Also read structure-compatible ancestors.

        Returns
        -------
        int
            Number of measurements attached.

        Raises
        ------
        RuntimeError
            No snapshot is bound.
        silk.exceptions.EvidenceNotFoundError
            No compatible spec measurements match.
        silk.exceptions.EvidenceMismatchError
            Matching evidence exists only as incompatible evidence, a selected row is
            stale, the live design differs from the bound snapshot, or a stored
            verdict disagrees with the live declaration.
        """
        from silk.specs import Measurement

        bindings = self._validated_live_spec_bindings()
        declarations = self.db.get_stored_specs(
            self.circuit_id,
            include_measurements=True,
            include_ancestors=include_ancestors,
        )
        candidates = []
        for declaration in declarations:
            location = (declaration.owner_path, declaration.spec_key)
            for record in declaration.measurements:
                if any(value is not None and getattr(record, name) != value
                       for name, value in (('source', source), ('corner', corner),
                                           ('temperature', temperature), ('voltage', voltage),
                                           ('tool_name', tool_name))):
                    continue
                if profile_id is UNSCOPED_PROFILE:
                    if record.profile_id is not None:
                        continue
                elif profile_id is not None and record.profile_id != profile_id:
                    continue
                candidates.append((location, record))
        self._check_swept_measurement_attribution(candidates)
        selected = {}
        for declaration in declarations:
            location = (declaration.owner_path, declaration.spec_key)
            for record in declaration.measurements:
                if source is not None and record.source != source:
                    continue
                if corner is not None and record.corner != corner:
                    continue
                if (
                    temperature is not None
                    and record.temperature != temperature
                ):
                    continue
                if voltage is not None and record.voltage != voltage:
                    continue
                if tool_name is not None and record.tool_name != tool_name:
                    continue
                if profile_id is UNSCOPED_PROFILE:
                    if record.profile_id is not None:
                        continue
                elif profile_id is not None:
                    if record.profile_id != profile_id:
                        continue
                if record.profile_id is not None:
                    dedup_key = (
                        location,
                        "profile",
                        record.profile_id,
                        record.source,
                        record.corner,
                        record.temperature,
                        record.voltage,
                    )
                else:
                    dedup_key = (location, record.source, record.corner)
                selected[dedup_key] = (location, record)

        if not selected:
            filters = {
                "source": source,
                "corner": corner,
                "temperature": temperature,
                "voltage": voltage,
                "tool_name": tool_name,
                "profile_id": profile_id,
                "include_ancestors": include_ancestors,
            }
            if self.db._has_incompatible_spec_measurements(
                self.circuit_id, **filters
            ):
                raise EvidenceMismatchError(
                    "Matching spec measurement context exists only as "
                    "incompatible evidence"
                )
            requested = ", ".join(
                f"{key}={value!r}"
                for key, value in filters.items()
                if key != "include_ancestors" and value is not None
            )
            raise EvidenceNotFoundError(
                "No compatible spec measurements are stored"
                + (f" for context {requested}" if requested else "")
            )

        from silk.db._deck_check import CheckContext
        check = CheckContext(self)
        selected = {key: (location, check.row(record))
                    for key, (location, record) in selected.items()}
        check.refuse_stale([record for _, record in selected.values()])
        check.warn()

        staged = []
        for location, record in selected.values():
            live_spec, _declaration = bindings[location]
            measurement = Measurement(
                source=record.source,
                value=record.value,
                passes=None,
                corner=record.corner,
                timestamp=(
                    0.0 if record.timestamp is None else record.timestamp
                ),
                sim_run_id=record.sim_run_id,
                notes=record.notes,
                aggregate=record.aggregate,
                temperature=record.temperature,
                voltage=record.voltage,
                sim_type=record.sim_type,
                tool_name=record.tool_name,
                origin_version_id=record.origin_version_id,
                profile_id=record.profile_id,
            )
            evaluated = bool(live_spec.evaluate(measurement))
            if record.passes is not None and record.passes != evaluated:
                raise EvidenceMismatchError(
                    f"Stored spec measurement {record.measurement_id!r} "
                    "cached pass does not match the live declaration"
                )
            measurement.passes = evaluated
            from silk.specs.evidence import stage_for_root
            stage_for_root(self._module, live_spec, measurement, location[0])
            from silk.db._attempts import hydrate
            hydrate(self._module, live_spec, location[0], record.measurement_id, measurement)
            staged.append((live_spec, measurement))

        for live_spec, measurement in staged:
            live_spec.record_measurement(measurement)
        return len(staged)

    @_write_transaction
    def store_report(self, report, corner=None, temperature=None,
                     voltage=None, sim_type=None, tool_name=None,
                     units=None, source=None, conditions=None):
        """Atomically store a spec report: metrics, status rows and spec measurements.

        Every numeric metric of the report becomes a result row (the unit is taken
        from ``units``, else from a recognised name suffix such as ``_dB`` or
        ``_MHz``, else empty). Every spec result becomes a ``'spec_<name>'`` status
        row and, when it has a numeric value, a first-class measurement. Non-numeric
        and complex metric values are not stored and are listed under
        ``'skipped'``. Spec results must carry the declaration identity emitted by
        the testbench and agree with the live declaration, the report's metric
        value and the cached verdict before anything is written.

        Original observation facts come only from the report's detached record,
        including after JSON transport or later runs. Older/manual reports without
        that record retain unknown observation time and correlation metadata;
        storage never borrows a live measurement or stamps the observation now.

        When the report carries a characterization context, every row written
        shares one profile and that exact point. The context is the only source of
        profile identity; a report without one is stored unscoped. When the report
        carries an operating-conditions snapshot, its corner, temperature and
        voltage are the stored evidence and are never filled in from the report's
        run label; explicit arguments are only checked against it. Otherwise the
        corner is the explicit argument, else the report's run corner, else the
        tracker's current corner.

        The call is its own rollback boundary. Live measurement updates are applied
        only after the outermost transaction commits.

        Parameters
        ----------
        report : SpecReport
            The report to store.
        corner : str or ConditionState, optional
            Process corner; must agree with the report's evidence.
        temperature : float, optional
            Temperature in degrees Celsius; must agree with the report's evidence.
        voltage : float, optional
            Supply voltage in volts; must agree with the report's evidence.
        sim_type : str, optional
            Analysis label.
        tool_name : str, optional
            Producing tool.
        units : dict, optional
            Maps a metric name to its unit label.
        source : str, optional
            Source label. Defaults to the report's evidence (``'behavioral'`` or
            ``'spice'``); it may not contradict a recorded backend.
        conditions : dict, optional
            JSON-compatible dictionary of extra conditions. The keys
            ``'silk_operating_conditions'``, ``'silk_metric_warnings'``,
            ``'silk_run_variables'`` and ``'silk_spec_measurement'`` are reserved.

        Returns
        -------
        dict
            ``{'metrics_stored': int, 'specs_stored': int, 'skipped': dict}`` where
            ``skipped`` maps a metric name to the reason it was not stored.

        Raises
        ------
        silk.exceptions.DatabaseError
            No snapshot is bound, or it is not sealed.
        silk.exceptions.EvidenceMismatchError
            The live design differs from the bound snapshot, an explicit
            ``source``, corner, temperature or voltage conflicts with the report's
            evidence, a spec result is missing its identity, repeated, undeclared or
            inconsistent with the live declaration, report metric value or cached
            verdict, a reserved ``conditions`` key was used, or live and stored
            evidence disagree.
        silk.exceptions.LockedVersionError
            The bound snapshot is locked.
        ValueError
            ``source`` is not an allowed spec-measurement source.
        TypeError
            A staged field has an unsupported type.
        """
        from silk.specs import Measurement

        bindings = self._validated_live_spec_bindings(writing=True)
        context = getattr(report, 'characterization_context', None)
        snapshot = getattr(report, 'operating_conditions', None)
        evidence_source = ('behavioral' if getattr(snapshot, 'backend', None)
                           == 'behavioral' else 'spice')
        if source is None:
            source = evidence_source
        elif (getattr(snapshot, 'backend', None) is not None
              and source != evidence_source):
            raise EvidenceMismatchError(
                f'store_report source={source!r} conflicts with the report '
                f'evidence ({evidence_source!r}); omit source or match it.')
        if snapshot is not None:
            from silk.simulation.conditions import storage_conditions
            # The run's own label (report.corner) is accepted as an explicit
            # corner; it names the run, not the evidence (TASK-406).
            storage_conditions(
                snapshot, context, corner, temperature, voltage,
                label=getattr(report, 'corner', None))
            # Schema v5 (TASK-409): the snapshot is the evidence. Its scalars
            # (a value, ConditionState.NONE, SWEEP, or None for unknown) are
            # stored as given, and never filled from the run label.
            corner, temperature, voltage = _snapshot_scalars(snapshot)
            conditions = dict(conditions or {})
            marker = snapshot.to_dict()
            reserved = 'silk_operating_conditions'
            if reserved in conditions and conditions[reserved] != marker:
                raise EvidenceMismatchError(
                    'conditions["silk_operating_conditions"] conflicts with the report snapshot')
            conditions[reserved] = marker
        swept = getattr(report, 'parameters', {}).get('silk_run_variables')
        if conditions is not None and 'silk_spec_measurement' in conditions:
            raise EvidenceMismatchError('conditions.silk_spec_measurement is reserved for stored evidence links')
        if swept is not None:
            import math
            if not isinstance(swept, dict) or not swept or any(
                    not isinstance(name, str) or not isinstance(value, (str, bool, int, float))
                    or (isinstance(value, (int, float)) and not math.isfinite(value))
                    for name, value in swept.items()):
                raise EvidenceMismatchError('Invalid swept Variable attribution on report')
            conditions = dict(conditions or {})
            if 'silk_run_variables' in conditions and conditions['silk_run_variables'] != swept:
                raise EvidenceMismatchError('conditions.silk_run_variables conflicts with the report')
            conditions['silk_run_variables'] = dict(swept)
        elif conditions is not None and 'silk_run_variables' in conditions:
            raise EvidenceMismatchError('conditions.silk_run_variables requires report attribution')
        report_metric_warnings = getattr(report, 'metric_warnings', None) or {}
        _WARNINGS_KEY = self.db._METRIC_WARNINGS_CONDITIONS_KEY
        if conditions is not None and _WARNINGS_KEY in conditions:
            raise EvidenceMismatchError(
                f'conditions[{_WARNINGS_KEY!r}] is reserved for the '
                'per-metric MetricWarnings store_report() attaches '
                '(TASK-337); do not set it explicitly'
            )
        if context is not None:
            for field_name, explicit, from_context in (
                ('corner', corner, context.corner),
                ('temperature', temperature, context.temperature),
                ('voltage', voltage, context.voltage),
            ):
                if (
                    explicit is not None
                    and _corner_value(explicit) != from_context
                ):
                    raise EvidenceMismatchError(
                        f"store_report {field_name}={explicit!r} conflicts "
                        f"with the report's characterization context "
                        f"({from_context!r}); omit the argument or match "
                        "the context exactly"
                    )
            resolved_corner = context.corner
        elif snapshot is not None:
            # The snapshot already confirmed any explicit corner; the run's
            # label (report.corner) is not evidence of what reached the
            # simulator (a PDK-free deck records corner none).
            resolved_corner = corner
        else:
            # Unscoped report (TASK-129): the corner the run actually used
            # outranks the tracker's current corner, and an explicit argument
            # may only confirm it -- never silently prefer either.
            report_corner = getattr(report, 'corner', None)
            if (
                corner is not None
                and report_corner is not None
                and _corner_value(corner) != report_corner
            ):
                raise EvidenceMismatchError(
                    f"store_report corner={corner!r} conflicts with the "
                    f"report's run corner ({report_corner!r}); omit the "
                    "argument or match the run corner exactly"
                )
            if corner is not None:
                resolved_corner = corner
            elif report_corner is not None:
                resolved_corner = report_corner
            else:
                resolved_corner = self._current_corner

        units = units or {}
        skipped: Dict[str, str] = {}
        results_dict = {}
        for name, value in report.metrics.items():
            if isinstance(value, Complex) and not isinstance(value, Real):
                skipped[name] = (f"complex metric value: {value!r}; numeric storage "
                                 "requires result.real or abs(result)")
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                skipped[name] = f"non-numeric value: {value!r}"
                continue
            if name in units:
                unit = units[name]
            else:
                suffix = name.rsplit('_', 1)[-1]
                unit = suffix if suffix in _UNIT_SUFFIXES else ''
            results_dict[name] = (float(value), unit)

        report_results = list(getattr(report, 'results', []) or [])
        if report_results and source not in {
            "behavioral", "spice", "extraction", "user"
        }:
            raise ValueError(
                "Spec measurement source must be one of behavioral, spice, "
                f"extraction, or user; got {source!r}"
            )
        staged_specs = []
        seen_locations = set()
        for result in report_results:
            location = (
                getattr(result, "owner_path", None),
                getattr(result, "spec_key", None),
            )
            if not all(isinstance(item, str) and item for item in location):
                raise EvidenceMismatchError(
                    "Spec result is missing declaration identity: both "
                    "owner_path and spec_key are required"
                )
            if location in seen_locations:
                raise EvidenceMismatchError(
                    f"Spec report repeats declaration {location!r}"
                )
            seen_locations.add(location)
            binding = bindings.get(location)
            if binding is None:
                raise EvidenceMismatchError(
                    f"Spec result {location[0]}/{location[1]} is not declared "
                    "by the bound live design"
                )
            live_spec, declaration = binding

            target = getattr(result, "target", None)
            if isinstance(target, bool) or not isinstance(target, Real):
                raise EvidenceMismatchError(
                    f"Spec result {location!r} has a non-numeric target"
                )
            target_matches = float(target).hex() == float(live_spec.target).hex()
            if (
                result.metric_name != live_spec.metric
                or result.op != live_spec.op
                or not target_matches
                or declaration.metric != live_spec.metric
                or declaration.op != live_spec.op
            ):
                raise EvidenceMismatchError(
                    f"Spec result {location[0]}/{location[1]} definition "
                    "does not match the live declaration"
                )

            value = getattr(result, "value", None)
            metric_value = report.metrics.get(result.metric_name)
            value_is_numeric = (
                not isinstance(value, bool) and isinstance(value, Real)
            )
            metric_is_numeric = (
                not isinstance(metric_value, bool)
                and isinstance(metric_value, Real)
            )
            has_measurement = value_is_numeric and metric_is_numeric
            if has_measurement:
                values_match = (
                    float(value).hex() == float(metric_value).hex()
                )
            elif (isinstance(value, Complex) and not isinstance(value, Real)
                  and isinstance(metric_value, Complex) and not isinstance(metric_value, Real)):
                # Complex evidence is preserved in memory, but has no numeric
                # storage row or Measurement. Compare both components exactly,
                # including NaN and signed zero, before accepting a failed spec.
                values_match = all(
                    float(getattr(value, part)).hex() == float(getattr(metric_value, part)).hex()
                    for part in ('real', 'imag'))
            else:
                values_match = (
                    (value is None or isinstance(value, str))
                    and (metric_value is None or isinstance(metric_value, str))
                    and value == metric_value
                )
            if not values_match or value_is_numeric != metric_is_numeric:
                raise EvidenceMismatchError(
                    f"Spec result {location[0]}/{location[1]} does not match "
                    "the report metric value"
                )
            cached = getattr(result, "passed", None)
            if has_measurement:
                value = float(value)
                probe = Measurement(
                    source=source,
                    value=value,
                    corner=resolved_corner,
                )
                evaluated = bool(live_spec.evaluate(probe))
                cache_matches = (
                    isinstance(cached, bool) and cached == evaluated
                )
            else:
                evaluated = False
                cache_matches = cached is False
            if not cache_matches:
                raise EvidenceMismatchError(
                    f"Spec result {location[0]}/{location[1]} cached pass "
                    "does not match its value, operator, and target"
                )

            # Only report-owned evidence can identify the producing observation.
            from silk.circuit.spec_report import _report_observation
            try:
                observation = _report_observation(result, report)
            except (ValueError, TypeError) as exc:
                raise EvidenceMismatchError(str(exc)) from exc
            if observation is not None and observation.source != source:
                raise EvidenceMismatchError('Report observation conflicts with source')
            live_measurement = observation.measurement if observation is not None else None
            if live_measurement is not None:
                if (live_measurement.source != source
                        or live_measurement.corner != resolved_corner):
                    raise EvidenceMismatchError('Report observation conflicts with source/corner')
                from silk.specs.evidence import stage_for_root
                stage_for_root(self._module, live_spec, live_measurement, location[0])
                live_measurement._projection_observation = (
                    observation, getattr(live_measurement, '_owner_scope', None))

            staged_temperature = (
                context.temperature
                if context is not None
                else (
                    temperature
                    if temperature is not None
                    else (
                        live_measurement.temperature
                        if live_measurement is not None
                        else None
                    )
                )
            )
            staged_voltage = (
                context.voltage
                if context is not None
                else (
                    voltage
                    if voltage is not None
                    else (
                        live_measurement.voltage
                        if live_measurement is not None
                        else None
                    )
                )
            )
            if snapshot is not None:
                staged_temperature = temperature
                staged_voltage = voltage
            staged_sim_type = (
                sim_type
                if sim_type is not None
                else (
                    live_measurement.sim_type
                    if live_measurement is not None
                    else None
                )
            )
            staged_tool_name = (
                tool_name
                if tool_name is not None
                else (
                    live_measurement.tool_name
                    if live_measurement is not None
                    else None
                )
            )
            timestamp = (
                live_measurement.timestamp
                if live_measurement is not None
                else observation.timestamp if observation is not None and has_measurement else None
            )
            sim_run_id = (
                live_measurement.sim_run_id
                if live_measurement is not None
                else None
            )
            notes = (
                live_measurement.notes if live_measurement is not None else ""
            )
            for column, item in (
                ("corner", resolved_corner),
                ("temperature", staged_temperature),
                ("voltage", staged_voltage),
            ):
                condition_columns(column, item, subject="Spec measurement")
            if timestamp is not None and (
                isinstance(timestamp, bool) or not isinstance(timestamp, Real)
            ):
                raise TypeError(
                    "Spec measurement timestamp must be a real number or None"
                )
            for field, item in (
                ("sim_type", staged_sim_type),
                ("tool_name", staged_tool_name),
                ("sim_run_id", sim_run_id),
            ):
                if item is not None and not isinstance(item, str):
                    raise TypeError(
                        f"Spec measurement {field} must be a string or None"
                    )
            if not isinstance(notes, str):
                raise TypeError("Spec measurement notes must be a string")

            staged_specs.append({
                "declaration": declaration,
                "live_spec": live_spec,
                "live_measurement": live_measurement,
                "value": value,
                "passes": evaluated,
                "has_measurement": has_measurement,
                "corner": resolved_corner,
                "temperature": staged_temperature,
                "voltage": staged_voltage,
                "sim_type": staged_sim_type,
                "tool_name": staged_tool_name,
                "timestamp": timestamp,
                "sim_run_id": sim_run_id,
                "notes": notes,
            })

        def resolve_common_context(field, explicit, fallback):
            if snapshot is not None and field in ('temperature', 'voltage'):
                resolved = snapshot.scalar(
                    'temperature' if field == 'temperature' else 'supply')
                for staged in staged_specs:
                    staged[field] = resolved
                return resolved
            values = {
                staged[field]
                for staged in staged_specs
                if staged[field] is not None
            }
            if len(values) > 1:
                raise EvidenceMismatchError(
                    f"Spec report contains conflicting {field} contexts"
                )
            if explicit is not None:
                resolved = explicit
            elif values:
                resolved = next(iter(values))
            else:
                resolved = fallback
            for staged in staged_specs:
                staged[field] = resolved
            return resolved

        resolved_temperature = resolve_common_context(
            "temperature", temperature, self._current_temp
        )
        resolved_voltage = resolve_common_context(
            "voltage", voltage, self._current_voltage
        )
        resolved_sim_type = resolve_common_context(
            "sim_type", sim_type, None
        )
        resolved_tool_name = resolve_common_context(
            "tool_name", tool_name, None
        )
        if context is not None:
            # The context's PVT is authoritative for every row this report
            # writes, including metrics with no attached spec (staged_specs
            # may be empty even though the report is scoped) -- never a
            # merge with a prior live measurement's temperature/voltage.
            resolved_temperature = context.temperature
            resolved_voltage = context.voltage

        # Stage the whole report before registering any live-object writes:
        # they must wait for the outermost SQL commit, including grouped saves.
        live_updates = []
        associations = []
        observations = {}

        with self.db.transaction():
            profile_id = None
            if context is not None:
                profile_record = self.db.store_characterization_profile(
                    self.circuit_id, context
                )
                profile_id = profile_record.profile_id
            # TASK-306: one content-addressed run_provenance row per
            # distinct record; every row this call writes points at it.
            provenance_record = getattr(report, 'provenance', None)
            provenance_id = (
                None if provenance_record is None
                else self.db.store_run_provenance(provenance_record)
            )

            for param, (value, unit) in results_dict.items():
                param_warnings = report_metric_warnings.get(param)
                if param_warnings:
                    param_conditions = dict(conditions or {})
                    param_conditions[_WARNINGS_KEY] = [
                        w.to_dict() for w in param_warnings
                    ]
                else:
                    # No warnings for this output: write the exact same
                    # `conditions` every other metric in this report gets,
                    # byte-identical to before this task.
                    param_conditions = conditions
                self.db.add_simulation_result(
                    circuit_version_id=self.circuit_id,
                    parameter=param,
                    value=value,
                    unit=unit,
                    corner=resolved_corner,
                    temperature=resolved_temperature,
                    voltage=resolved_voltage,
                    sim_type=resolved_sim_type,
                    tool_name=resolved_tool_name,
                    source=source,
                    conditions=param_conditions,
                    profile_id=profile_id,
                    provenance_id=provenance_id,
                )
            for staged in staged_specs:
                live_spec = staged["live_spec"]
                measurement_id = str(uuid.uuid4()) if staged['has_measurement'] else None
                spec_conditions = conditions
                if swept is not None:
                    spec_conditions = dict(conditions)
                    spec_conditions['silk_spec_measurement'] = dict(
                        measurement_id=measurement_id,
                        origin_spec_id=staged['declaration'].spec_id,
                        origin_version_id=self.circuit_id)
                    if measurement_id is None:
                        spec_conditions['silk_spec_measurement']['has_measurement'] = False
                status = "pass" if staged["passes"] else "fail"
                details = (
                    f"{live_spec.metric}={staged['value']} "
                    f"{live_spec.op} {live_spec.target}"
                )
                self.db.add_simulation_result(
                    circuit_version_id=self.circuit_id,
                    parameter=f"spec_{live_spec.name}",
                    value_text=f"{status} ({details})",
                    source=source,
                    sim_type=staged["sim_type"],
                    corner=staged["corner"],
                    temperature=staged["temperature"],
                    voltage=staged["voltage"],
                    tool_name=staged["tool_name"],
                    conditions=spec_conditions,
                    profile_id=profile_id,
                    provenance_id=provenance_id,
                )
                if staged["has_measurement"]:
                    from silk.db._attempts import stage_observation
                    attempt, observation_scope = stage_observation(
                        self.db, live_spec, staged['live_measurement'])
                    if attempt is not None:
                        observations[(id(live_spec), observation_scope, attempt.attempt_id)] = (
                            live_spec, attempt, observation_scope)
                    self.db.insert_spec_measurement(
                        measurement_id,
                        staged["declaration"].spec_id,
                        source,
                        staged["value"],
                        corner=staged["corner"],
                        timestamp=staged["timestamp"],
                        sim_run_id=staged["sim_run_id"],
                        notes=staged["notes"],
                        passes=staged["passes"],
                        temperature=staged["temperature"],
                        voltage=staged["voltage"],
                        sim_type=staged["sim_type"],
                        tool_name=staged["tool_name"],
                        profile_id=profile_id,
                        provenance_id=provenance_id,
                    )
                    if profile_id is not None:
                        # AC5 / design SS4.3: promote the live measurement to
                        # a profile-keyed slot distinct from the legacy
                        # ``source@corner`` string key _simulate() used, so a
                        # second scoped report at the same corner (a
                        # different profile) coexists instead of clobbering
                        # this one in ``live_spec.measurements``.
                        updated = Measurement(
                            source=source,
                            value=staged["value"],
                            passes=staged["passes"],
                            corner=staged["corner"],
                            timestamp=staged["timestamp"] if staged["timestamp"] is not None else 0.0,
                            sim_run_id=staged["sim_run_id"],
                            notes=staged["notes"],
                            temperature=staged["temperature"],
                            voltage=staged["voltage"],
                            sim_type=staged["sim_type"],
                            tool_name=staged["tool_name"],
                            profile_id=profile_id,
                        )
                        from silk.specs.evidence import stage_for_root
                        stage_for_root(self._module, live_spec, updated,
                                       staged["declaration"].owner_path)
                        from silk.db._attempts import projection_for, attach_projection
                        association = projection_for(self._module, live_spec,
                            staged['declaration'].owner_path, measurement_id,
                            updated, attempt, observation_scope)
                        attach_projection(updated, association[2])
                        live_updates.append((live_spec, updated, attempt, observation_scope))
                        associations.append(association)
                    elif staged['live_measurement'] is not None:
                        # Unscoped stores also retain a row link; they do not
                        # publish a new profile slot or change the observation.
                        from silk.db._attempts import projection_for
                        projected = Measurement(source=source, value=staged['value'],
                            passes=staged['passes'], corner=staged['corner'],
                            timestamp=staged['timestamp'], sim_run_id=staged['sim_run_id'],
                            notes=staged['notes'], temperature=staged['temperature'],
                            voltage=staged['voltage'], sim_type=staged['sim_type'],
                            tool_name=staged['tool_name'])
                        associations.append(projection_for(self._module, live_spec,
                            staged['declaration'].owner_path, measurement_id,
                            projected, attempt, observation_scope))

            if live_updates or associations:
                def validate_live_updates():
                    # Genuine observations can be published after store_report
                    # returns but before an enclosing transaction commits.
                    # Merge all known facts now; callbacks only apply this state.
                    from dataclasses import replace
                    from silk.specs.attempt import prepare_attempt
                    pending = self.db.__dict__.get('_pending_spec_attempts', {})
                    validated = {
                        key: prepare_attempt(spec, pending.get(key, attempt), scope)
                        for key, (spec, attempt, scope) in observations.items()
                    }
                    for i, (spec, measurement, attempt, scope) in enumerate(live_updates):
                        if attempt is not None:
                            attempt = validated[(id(spec), scope, attempt.attempt_id)]
                            measurement._projection_observation = (attempt, scope)
                            live_updates[i] = (spec, measurement, attempt, scope)
                    for i, (mapping, key, association) in enumerate(associations):
                        if association.observation is not None:
                            attempt = validated[(key[1], association.scope,
                                                 association.observation.attempt_id)]
                            associations[i] = (mapping, key, replace(association, observation=attempt))

                self.db._before_commit(validate_live_updates)

                def apply_live_updates():
                    for live_spec, measurement, attempt, scope in live_updates:
                        live_spec._publish_measurement(measurement, attempt, scope)
                    for mapping, key, association in associations:
                        mapping[key] = association

                self.db._after_commit(apply_live_updates)

        return {
            'metrics_stored': len(results_dict),
            'specs_stored': len(staged_specs),
            'skipped': skipped,
        }

    def require_report_storage_ready(self):
        """Check, without a report, that a report could be stored.

        Lets a caller reject every readiness failure that can be known before an
        expensive run. Nothing is created or written.

        Returns
        -------
        str
            The version id of the bound snapshot.

        Raises
        ------
        silk.exceptions.DatabaseError
            No snapshot is bound, or it is not sealed.
        silk.exceptions.EvidenceMismatchError
            The live design differs from the bound snapshot.
        silk.exceptions.LockedVersionError
            The bound snapshot is locked.
        """
        self._validated_live_spec_bindings(writing=True)
        self.db._require_sealed(self.circuit_id)
        return self.circuit_id

    def set_corner(self, corner, temperature=None, voltage=None):
        """Set the default operating point for subsequent result storage.

        Parameters
        ----------
        corner : str
            Process corner name.
        temperature : float, optional
            Temperature in degrees Celsius; unchanged when ``None``.
        voltage : float, optional
            Supply voltage in volts; unchanged when ``None``.
        """
        self._current_corner = corner
        if temperature is not None:
            self._current_temp = temperature
        if voltage is not None:
            self._current_voltage = voltage

    def view_summary(self):
        """Return the simulation summary of the bound snapshot.

        Returns
        -------
        str or None
            The summary text rendered by :meth:`silk.db.Database.get_simulation_summary`
            (empty when there are no results), or ``None`` when no snapshot is
            bound.
        """
        if self.circuit_id is None:
            return None
        return self.db.get_simulation_summary(self.circuit_id)

    @_write_transaction
    def promote(self, to_stage, new_version=None, lock=True, description=""):
        """Promote the bound snapshot to a later design stage and bind to the result.

        The new binding is visible immediately inside a grouped transaction and is
        restored if that scope rolls back.

        Parameters
        ----------
        to_stage : str
            ``'stable'``, ``'post_layout'`` or ``'tapeout'``.
        new_version : str, optional
            Version label of the promoted snapshot.
        lock : bool, default True
            Lock the promoted snapshot.
        description : str, default ""
            Description; ``'Promoted from <stage>'`` when empty.

        Returns
        -------
        str
            The version id of the promoted snapshot.

        Raises
        ------
        ValueError
            ``to_stage`` is not one of the three allowed names.
        silk.exceptions.DatabaseError
            No snapshot is bound, or promotion fails as
            :meth:`silk.db.Database.promote_circuit`.
        silk.exceptions.EvidenceMismatchError
            The live design differs from the bound snapshot.
        """
        self._require_write_identity(self._module, self.circuit_id)
        stage_map = {
            'stable': DesignStage.STABLE,
            'post_layout': DesignStage.POST_LAYOUT,
            'tapeout': DesignStage.TAPEOUT,
        }
        if to_stage not in stage_map:
            raise ValueError(
                f"Invalid stage '{to_stage}'. Use one of: {list(stage_map)}"
            )
        new_id = self.db.promote_circuit(
            self.circuit_id,
            stage_map[to_stage],
            new_version=new_version,
            lock=lock,
            created_by=self.created_by,
            description=description or f"Promoted from {self.stage.value}",
        )
        self._bind_version(new_id)
        return new_id

    @_write_transaction
    def add_transistor(self, name, transistor_type, width, length,
                       multiplier=1, model=None):
        """Manually add a transistor to the bound snapshot.

        Parameters
        ----------
        name : str
            Instance name.
        transistor_type : str
            Device type label, for example ``'nmos'``.
        width : float
            Channel width in the caller's units, stored as given.
        length : float
            Channel length in the caller's units, stored as given.
        multiplier : int, default 1
            Parallel multiplier.
        model : str, optional
            Model name.

        Raises
        ------
        silk.exceptions.DatabaseError
            No snapshot is bound, or it is not in the ``building`` state (a
            snapshot made by this tracker is sealed, so this normally fails).
        silk.exceptions.EvidenceMismatchError
            The live design differs from the bound snapshot.
        """
        self._require_write_identity(self._module, self.circuit_id)
        self.db.add_transistor(
            circuit_version_id=self.circuit_id,
            name=name,
            transistor_type=transistor_type,
            width=width,
            length=length,
            multiplier=multiplier,
            model=model,
        )

    @_write_transaction
    def add_net(self, name, net_type="signal"):
        """Manually add a net to the bound snapshot.

        Parameters
        ----------
        name : str
            Net name.
        net_type : str, default "signal"
            Role label.

        Raises
        ------
        silk.exceptions.DatabaseError
            No snapshot is bound, or it is not in the ``building`` state (a
            snapshot made by this tracker is sealed, so this normally fails).
        silk.exceptions.EvidenceMismatchError
            The live design differs from the bound snapshot.
        """
        self._require_write_identity(self._module, self.circuit_id)
        self.db.add_net(
            circuit_version_id=self.circuit_id,
            name=name,
            net_type=net_type,
        )

    def get_corner_summary(self):
        """Return the corners the bound snapshot has results for.

        Returns
        -------
        dict
            As :meth:`silk.db.Database.get_corner_summary`; ``{}`` when no snapshot is bound.
        """
        if self.circuit_id is None:
            return {}
        return self.db.get_corner_summary(self.circuit_id)

    def get_corners_for_parameter(self, parameter, latest_only=False):
        """Return the results of one parameter across corners.

        Parameters
        ----------
        parameter : str
            Metric name.
        latest_only : bool, default False
            Collapse to the newest observation per condition group; see
            :meth:`silk.db.Database.get_corners_for_parameter`.

        Returns
        -------
        list of dict
            As :meth:`silk.db.Database.get_corners_for_parameter`; ``[]`` when no snapshot
            is bound.
        """
        if self.circuit_id is None:
            return []
        return self.db.get_corners_for_parameter(
            self.circuit_id, parameter, latest_only=latest_only)

    def export_to_json(self, output_file):
        """Export the tracked circuit to a JSON interoperability projection.

        Uses :meth:`silk.db.CircuitExporter.export_to_json` on this tracker's database
        and circuit name.

        Parameters
        ----------
        output_file : str
            Path of the JSON file to write.

        Raises
        ------
        RuntimeError
            No snapshot is bound.
        """
        if self.circuit_id is None:
            raise RuntimeError(
                "No circuit version created. Call create_circuit_version() first."
            )
        from silk.db.circuit_import import CircuitExporter
        exporter = CircuitExporter(self.db)
        exporter.export_to_json(self.circuit_name, output_file)

    def close(self):
        """Close the database connection and drop the reference to it.

        ``db`` is ``None`` afterwards; calling this again does nothing.
        """
        if self.db:
            self.db.close()
            self.db = None

    @_write_transaction
    def store_net_currents(
        self,
        report,
        corner,
        *,
        temperature=None,
        voltage=None,
        tool_name=None,
    ):
        """Persist annotated electromigration currents of the module's definitions.

        Reads the annotated currents of every definition copy in the module's
        ``module_dict``, creating a snapshot for a definition that has none, and
        stores one current set per definition in a single transaction. Each stored
        set supersedes the previous set for its selection key and keeps it
        readable.

        Parameters
        ----------
        report : object
            The annotation report. Its ``window`` (first element is stored as the
            window start), ``analysis`` and ``results_file`` attributes are read
            when present.
        corner : str or ConditionState
            Process corner name or ``ConditionState.NONE`` when no corner reached
            the simulator. Required; never defaulted.
        temperature : float, optional
            Temperature in degrees Celsius (keyword-only); taken from the module's
            characterization context, else the tracker's current temperature.
        voltage : float, optional
            Supply voltage in volts (keyword-only); taken like ``temperature``.
        tool_name : str, optional
            Tool that produced the currents (keyword-only).

        Returns
        -------
        dict
            Maps each stored circuit name to the version id its currents were
            stored on.

        Raises
        ------
        RuntimeError
            ``corner`` is ``None``, no annotated definition was found (run
            ``annotate_currents()`` first), or the run's operating conditions record
            an unknown corner.
        TypeError
            The module's characterization context is not a
            ``CharacterizationContext``.
        silk.exceptions.EvidenceMismatchError
            The live design differs from a bound or resolved snapshot.
        silk.exceptions.LockedVersionError
            A target snapshot is locked.
        """
        if self.circuit_id is not None:
            self._require_write_identity(self._module, self.circuit_id)
        if corner is None:
            raise RuntimeError(
                "store_net_currents: corner is required (never defaulted — "
                "a silent 'tt' would mislabel ss/ff data). Use "
                "corner=tb._last_sim_corner from the simulate() call.")
        tstart = report.window[0] if getattr(report, 'window', None) else None
        context = getattr(self._module, "characterization_context", None)
        if context is not None and not isinstance(
            context, CharacterizationContext
        ):
            raise TypeError(
                "Current evidence context must be a CharacterizationContext "
                "or None"
            )
        snapshot = getattr(self._module, 'operating_conditions', None)
        if snapshot is not None:
            from silk.simulation.conditions import storage_conditions
            storage_conditions(
                snapshot, context, corner, temperature, voltage)
            corner, temperature, voltage = _snapshot_scalars(snapshot)
            if corner is None:
                raise RuntimeError(
                    "store_net_currents: the run's operating conditions "
                    "record an unknown corner, and current evidence is "
                    "always selected by its corner (a value or "
                    "ConditionState.NONE).")
        else:
            if temperature is None:
                temperature = (context.temperature if context is not None
                               else self._current_temp)
            if voltage is None:
                voltage = (context.voltage if context is not None
                           else self._current_voltage)
        stored = {}
        pending = []          # identity and rows per definition
        for mn, entry in (self._module.module_dict or {}).items():
            defn = entry.get('device') if isinstance(entry, dict) else None
            if defn is None:
                continue
            rows = self._collect_current_rows(defn)
            if not rows:
                continue
            identity = self.compute_snapshot_identity_for(defn)
            circuit_name = getattr(defn, 'module_name', None) or mn
            pending.append((defn, circuit_name, identity, rows))
        if not pending:
            raise RuntimeError(
                "store_net_currents: no annotated definition copies found in "
                "module_dict — run annotate_currents() first.")
        # TASK-306: the run these currents came from (the testbench module
        # this tracker is bound to holds its launch/prepared-deck facts).
        from silk.simulation.provenance import from_module as _provenance_from_module
        provenance_record = _provenance_from_module(self._module)
        with self.db.transaction():
            provenance_id = self.db.store_run_provenance(provenance_record)
            resolved = []
            for defn, circuit_name, identity, rows in pending:
                version_id = self.resolve_version(circuit_name, identity)
                if version_id is None:
                    version_id = self._ensure_version(
                        defn,
                        circuit_name=circuit_name,
                        parent_version_id=None,
                        description="Current-annotation snapshot",
                        netlist_path=_deck_path(defn, None),
                        bind=False,
                    )
                self._require_write_identity(defn, version_id)
                resolved.append((circuit_name, version_id, identity, rows))

            for circuit_name, version_id, identity, rows in resolved:
                staged_rows = [
                    {
                        'net_name': net_name,
                        'pin_id': pin_id,
                        'avg_a': avg,
                        'rms_a': rms,
                        'peak_a': peak,
                        'source': source,
                        'current_vector': vector,
                    }
                    for net_name, pin_id, avg, rms, peak, source, vector
                    in rows
                ]
                self.db.replace_net_current_set(
                    version_id,
                    corner=corner,
                    structure_scheme=identity.structure.scheme,
                    structure_hash=identity.structure.digest,
                    rows=staged_rows,
                    temperature=temperature,
                    voltage=voltage,
                    tool_name=tool_name,
                    analysis=getattr(report, 'analysis', None),
                    tstart=tstart,
                    results_file=getattr(report, 'results_file', None),
                    context=context,
                    provenance_id=provenance_id,
                )
                stored[circuit_name] = version_id
        return stored

    def load_net_currents(self, corner, force=False):
        """Populate the live module's nets and pins from stored current rows.

        Only the net-level aggregate rows set a net's aggregate; per-pin rows set
        pin currents. Loaded values are marked as database-sourced. Existing live
        currents sourced from ``'user'`` or ``'spice'`` are kept unless ``force``.

        Parameters
        ----------
        corner : str or ConditionState
            Corner of the stored set to load.
        force : bool, default False
            Overwrite live currents sourced from ``'user'`` or ``'spice'``; an
            in-session annotate, store, load round needs it.

        Returns
        -------
        int
            Number of nets and pins updated.

        Raises
        ------
        RuntimeError
            No stored snapshot matches the live identity, no set exists for the
            corner, or a stored net or pin has no live counterpart (structure
            drift).
        silk.exceptions.EvidenceMismatchError
            The live structure differs from the bound snapshot, or the stored
            currents exist only on an incompatible structure.
        """
        module = self._module
        identity = self.compute_snapshot_identity_for(module)
        version_id = self.circuit_id
        if version_id is not None:
            self.db.require_version_identity(
                version_id, identity, include_spec_set=False
            )
        else:
            version_id = self.resolve_version(self.circuit_name, identity)
        if version_id is None:
            raise RuntimeError(
                f"load_net_currents: no stored version of "
                f"{self.circuit_name!r} matches this snapshot identity — the "
                "circuit or its spec declarations changed since store, or "
                "were never stored.")
        _origin_version_id, rows = self.db.get_net_current_set(
            version_id, corner, include_ancestors=True
        )
        actions = self._stage_current_restore(rows, force=force)
        return self._apply_current_restore(actions)

    @staticmethod
    def _collect_current_rows(defn):
        """(net, pin_id, avg, rms, peak, source, vector) rows for one
        definition copy. Aggregate row (pin_id='') only for a populated,
        non-db-sourced net aggregate — partial nets contribute pin rows only,
        so an absent aggregate row round-trips as None (never 0.0)."""
        rows = []
        for net_name, net in defn.net_dict.items():
            if getattr(net, 'width', 1) != 1:
                continue
            for b in net.pins:
                if b.kind == 'lint_marker' or b.avg_current is None:
                    continue
                if b._current_source == 'db':
                    continue
                rows.append((net_name, f'{b.instance}.{b.port}',
                             b.avg_current, b.rms_current, b.peak_current,
                             b._current_source or 'spice', b.current_vector))
            if net.avg_current is not None and net._current_source != 'db':
                rows.append((net_name, '', net.avg_current, net.rms_current,
                             net.peak_current, net._current_source or 'spice',
                             None))
        return rows
