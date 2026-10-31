"""Database-tracking methods for :class:`~silk.circuit.modules.Module`.

Split out of ``modules.py`` as a per-subsystem **part**. ``_DbMethods`` is a pure mixin: no state, no separate identity — every wrapper here lazily builds/uses ``self._tracker`` (``silk.db.tracking.CircuitTracker``).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from silk.circuit._module_protocol import _ModuleAttrs
    _Base = _ModuleAttrs
else:
    _Base = object


def _resolve_db_path(module, db_path):
    """Adopt ``db_path`` onto ``module`` when it has no db binding yet; raise
    ``ValueError`` when it conflicts with an existing binding (a bound
    ``module._db_path`` or a live tracker attached to a different path).
    Passing the same path as the existing binding is a no-op.

    Module-level (not a ``_DbMethods`` method) so it doesn't widen the
    frozen ``dir(Module)`` API surface guarded by the refactor goldens.
    """
    if db_path is None:
        return
    existing = (module._tracker.db.db_path if module._tracker is not None
                else module._db_path)
    if existing is not None and existing != db_path:
        raise ValueError(
            f"db_path mismatch: this module is already bound to "
            f"{existing!r}, but {db_path!r} was passed. Use the "
            f"existing db_path, or build a fresh Module to target a "
            f"different database."
        )
    if module._tracker is None and module._db_path is None:
        module._db_path = db_path


class _DbMethods(_Base):
    """Database-tracking wrappers mixed into ``Module``. Holds no state."""

    # ===== Database tracking wrappers (delegate to CircuitTracker) =====

    def _ensure_tracker(self):
        """Lazily create the CircuitTracker on first tracking call."""
        if self._tracker is not None:
            return self._tracker
        if self._db_path is None:
            raise RuntimeError(
                "No db_path provided. Pass db_path='...' to Module() to enable tracking."
            )
        from silk.db.tracking import CircuitTracker
        self._tracker = CircuitTracker(
            self, self._db_path,
            version=self._db_version,
            stage=self._db_stage,
            created_by=self._db_created_by,
        )
        return self._tracker

    def create_circuit_version(
        self, description="", netlist_path=None, *, parent_version_id=None,
    ):
        """Create (or reuse) a sealed circuit version and bind this module to it.

        Parameters
        ----------
        description : str, default ''
            Free-text description stored with the version.
        netlist_path : str, optional
            Netlist path recorded with the version; defaults to the module's
            ``dir_folder`` joined with its ``netlistname`` file name.
        parent_version_id : str, optional
            Version this one derives from.

        Returns
        -------
        str
            The version id. If a snapshot with the same structure and spec
            declarations already exists, its id is returned and bound.

        Raises
        ------
        RuntimeError
            If the module was built without ``db_path``.
        DatabaseError
            If the existing snapshot has a different parent than requested.
        """
        return self._ensure_tracker().create_circuit_version(
            description,
            netlist_path,
            parent_version_id=parent_version_id,
        )

    def store_results(self, results_dict, corner=None, temperature=None,
                      voltage=None, sim_type=None, tool_name=None,
                      source="user", conditions=None):
        """Store several simulation results in one transaction.

        Parameters
        ----------
        results_dict : dict
            ``{parameter: (value, unit)}``. A string value is stored as text.
        corner : str, optional
            Process-corner label; defaults to the tracker's current corner
            (see :meth:`set_corner`).
        temperature : float, optional
            Temperature in degrees Celsius; defaults to the current value.
        voltage : float, optional
            Supply voltage in volts; defaults to the current value.
        sim_type : str, optional
            Analysis kind, e.g. ``'tran'``.
        tool_name : str, optional
            Simulator or tool that produced the values.
        source : str, default 'user'
            Provenance label of the rows.
        conditions : dict, optional
            Extra operating conditions recorded with each row.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the module was built without ``db_path``.
        DatabaseError
            If no circuit version has been created.
        EvidenceMismatchError
            If the live circuit no longer matches the bound snapshot.
        """
        return self._ensure_tracker().store_results(
            results_dict, corner, temperature, voltage, sim_type, tool_name,
            source, conditions)

    def store_result(self, parameter, value, unit, corner=None,
                     temperature=None, voltage=None, sim_type=None,
                     tool_name=None, source="user", conditions=None):
        """Store a single simulation result.

        Parameters
        ----------
        parameter : str
            Metric or parameter name.
        value : float or str
            The value; a string is stored as text.
        unit : str
            Unit of ``value`` (e.g. ``'V'``).
        corner, temperature, voltage, sim_type, tool_name, source, conditions
            As for :meth:`store_results`.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the module was built without ``db_path``.
        DatabaseError
            If no circuit version has been created.
        EvidenceMismatchError
            If the live circuit no longer matches the bound snapshot.
        """
        return self._ensure_tracker().store_result(
            parameter, value, unit, corner, temperature, voltage, sim_type,
            tool_name, source, conditions)

    def store_pass_fail(self, spec_name, status, details=""):
        """Store a pass/fail result as a ``spec_<name>`` text row.

        Parameters
        ----------
        spec_name : str
            Name of the spec; the row's parameter is ``spec_<spec_name>``.
        status : str
            Outcome text, e.g. ``'PASS'``.
        details : str, default ''
            Appended to the stored text in parentheses.

        Returns
        -------
        None
        """
        return self._ensure_tracker().store_pass_fail(spec_name, status, details)

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
        """Return flagged metric history for this live design.

        ``deck_check`` is a lookup snapshot (checked/stale/not checked),
        ignored by equality. History flags stale rows; selected stale latest
        rows raise. Unknowns stay usable with one combined warning. Raw
        Database reads have ``deck_check=None`` and do not run this check.

        ``profile_id`` (design SS4.3/SS8) is a 3-state filter: ``None`` (the
        default) applies no profile filter; ``silk.db.UNSCOPED_PROFILE``
        matches only explicitly unscoped evidence; an exact ``profile_id``
        string matches only that profile.

        Parameters
        ----------
        parameter, source, sim_type, corner, tool_name : str, optional
            Equality filters on the stored row; ``None`` applies none.
        temperature : float, optional
            Temperature filter in degrees Celsius.
        voltage : float, optional
            Supply-voltage filter in volts.
        conditions : dict, optional
            Filter on the recorded operating conditions.
        profile_id : str, optional
            Characterization-profile filter, as described above.
        latest_only : bool, default False
            Collapse the append-only history to the newest row per key.
        include_ancestors : bool, default True
            Include rows stored on ancestor versions of the bound snapshot.

        Returns
        -------
        list
            Flagged result records.

        Raises
        ------
        RuntimeError
            If no circuit version is bound.
        EvidenceMismatchError
            If the live circuit no longer matches the bound snapshot, or (with
            ``latest_only``) a selected row is stale.
        EvidenceNotFoundError
            If no compatible row matches the filters.
        """
        return self._ensure_tracker().query_results(
            parameter=parameter,
            source=source,
            sim_type=sim_type,
            corner=corner,
            temperature=temperature,
            voltage=voltage,
            tool_name=tool_name,
            conditions=conditions,
            profile_id=profile_id,
            latest_only=latest_only,
            include_ancestors=include_ancestors,
        )

    def get_stored_specs(
        self,
        *,
        include_measurements=True,
        include_ancestors=True,
        profile_id=None,
    ):
        """Return typed historical declarations for the bound snapshot.

        ``profile_id`` (design SS4.3/SS8) filters which measurements are
        included per declaration -- see ``CircuitTracker.get_stored_specs``.

        Parameters
        ----------
        include_measurements : bool, default True
            Attach each declaration's stored measurements.
        include_ancestors : bool, default True
            Include declarations stored on ancestor versions.
        profile_id : str, optional
            Profile filter for the measurements, as in :meth:`query_results`.

        Returns
        -------
        list
            Stored spec declarations.

        Raises
        ------
        RuntimeError
            If no circuit version is bound.
        """
        return self._ensure_tracker().get_stored_specs(
            include_measurements=include_measurements,
            include_ancestors=include_ancestors,
            profile_id=profile_id,
        )

    def list_characterization_profiles(self, *, include_ancestors=True):
        """List stored profiles compatible with this rebuilt structure.

        Parameters
        ----------
        include_ancestors : bool, default True
            Include profiles stored on ancestor versions, nearest first.

        Returns
        -------
        list
            Characterization profile records.

        Raises
        ------
        RuntimeError
            If no circuit version is bound.
        EvidenceMismatchError
            If the live structure no longer matches the bound snapshot.
        """
        return self._ensure_tracker().list_characterization_profiles(
            include_ancestors=include_ancestors
        )

    def resolve_characterization_profile(
        self,
        *,
        profile_id=None,
        context=None,
        include_ancestors=True,
    ):
        """Resolve one exact profile ID or complete descriptor.

        Parameters
        ----------
        profile_id : str, optional
            Exact profile id.
        context : CharacterizationContext, optional
            A complete descriptor to match instead of an id.
        include_ancestors : bool, default True
            Search ancestor versions as well.

        Returns
        -------
        CharacterizationProfileRecord
            The matching profile.

        Raises
        ------
        RuntimeError
            If no circuit version is bound.
        EvidenceMismatchError
            If the live structure no longer matches the bound snapshot.
        """
        return self._ensure_tracker().resolve_characterization_profile(
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
        """Return flagged profile/PVT history, including stale members.

        One combined warning always discloses unchecked current sets.
        ``deck_check`` is an ephemeral snapshot, ignored by equality; raw
        Database records leave it None. The aggregate is never fully checked.

        Parameters
        ----------
        profile_id : str, optional
            Exact profile id.
        context : CharacterizationContext, optional
            A complete descriptor to select by.
        corner : str, optional
            Process corner of the point.
        temperature : float, optional
            Temperature in degrees Celsius.
        voltage : float, optional
            Supply voltage in volts.
        include_ancestors : bool, default True
            Include evidence stored on ancestor versions.

        Returns
        -------
        object
            The aggregate evidence (metrics, spec measurements, currents).

        Raises
        ------
        RuntimeError
            If no circuit version is bound.
        EvidenceMismatchError
            If the live structure or declarations no longer match.
        """
        return self._ensure_tracker().select_characterization_evidence(
            profile_id=profile_id,
            context=context,
            corner=corner,
            temperature=temperature,
            voltage=voltage,
            include_ancestors=include_ancestors,
        )

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
        """Atomically restore a profile/PVT; stale evidence raises first.

        Not-checked values stay usable with one combined warning, always
        including currents whose annotations are not frozen at launch.

        Parameters
        ----------
        profile_id, context, corner, temperature, voltage, include_ancestors
            Selection, as for :meth:`select_characterization_evidence`
            (``temperature`` in degrees Celsius, ``voltage`` in volts).
        force : bool, default False
            Overwrite current annotations the user set by hand.

        Returns
        -------
        CharacterizationRestoreResult
            Provenance and counts of what was restored.

        Raises
        ------
        EvidenceMismatchError
            If the selected evidence is stale or does not match the live
            declarations; nothing is changed.
        """
        return self._ensure_tracker().restore_characterization_evidence(
            profile_id=profile_id,
            context=context,
            corner=corner,
            temperature=temperature,
            voltage=voltage,
            include_ancestors=include_ancestors,
            force=force,
        )

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
        """Attach observations after checking their producing decks.

        Stale selected rows raise before mutation. Not-checked observations
        remain usable with one combined warning; a restored Spec may PASS.

        ``profile_id`` (design SS4.3/SS8) is a 3-state filter: ``None`` (the
        default) matches both scoped and unscoped evidence;
        ``silk.db.UNSCOPED_PROFILE`` matches only explicitly unscoped rows;
        an exact ``profile_id`` string matches only that profile.

        Parameters
        ----------
        source, corner, tool_name : str, optional
            Equality filters on the stored measurements.
        temperature : float, optional
            Temperature filter in degrees Celsius.
        voltage : float, optional
            Supply-voltage filter in volts.
        profile_id : str, optional
            Profile filter, as described above.
        include_ancestors : bool, default True
            Include measurements stored on ancestor versions.

        Returns
        -------
        int
            Number of measurements attached to the live specs.

        Raises
        ------
        EvidenceMismatchError
            If a selected row is stale or has no matching live declaration.
        EvidenceNotFoundError
            If nothing matches the filters.
        """
        return self._ensure_tracker().restore_spec_measurements(
            source=source,
            corner=corner,
            temperature=temperature,
            voltage=voltage,
            tool_name=tool_name,
            profile_id=profile_id,
            include_ancestors=include_ancestors,
        )

    def store_report(self, report, corner=None, temperature=None,
                     voltage=None, sim_type=None, tool_name=None,
                     units=None, source=None, conditions=None):
        """Store a whole testbench run (a SpecReport) in one call.

        Parameters
        ----------
        report : SpecReport
            The report returned by a testbench run.
        corner : str, optional
            Process corner; evidence on the report takes precedence.
        temperature : float, optional
            Temperature in degrees Celsius.
        voltage : float, optional
            Supply voltage in volts.
        sim_type : str, optional
            Analysis kind, e.g. ``'tran'``.
        tool_name : str, optional
            Simulator or tool that produced the report.
        units : dict, optional
            Unit per metric name.
        source : str, optional
            Provenance label; defaults to the backend evidence on the report.
        conditions : dict, optional
            Extra operating conditions recorded with each row.

        Returns
        -------
        dict
            ``{'metrics_stored': int, 'specs_stored': int, 'skipped': ...}``.

        Raises
        ------
        EvidenceMismatchError
            If the report disagrees with the live declarations, the recorded
            backend, or the explicit conditions; nothing is written.
        """
        return self._ensure_tracker().store_report(
            report, corner=corner, temperature=temperature, voltage=voltage,
            sim_type=sim_type, tool_name=tool_name, units=units,
            source=source, conditions=conditions)

    def store_net_currents(
        self,
        report,
        corner,
        db_path=None,
        *,
        temperature=None,
        voltage=None,
        tool_name=None,
    ):
        """Persist annotated EM currents (net_currents table) from this
        testbench's worst-case definition copies.

        Call on the TESTBENCH that ran annotate_currents(); its module_dict
        holds the definition copies. ``db_path`` may be supplied here when the
        Module was built without one; it must match any existing binding.

        Parameters
        ----------
        report : CurrentAnnotationReport
            The report returned by ``annotate_currents()``.
        corner : str or ConditionState
            Process corner of the annotated run (required; never defaulted).
        db_path : str, optional
            Database path to adopt when the module has none.
        temperature : float, optional
            Temperature in degrees Celsius.
        voltage : float, optional
            Supply voltage in volts.
        tool_name : str, optional
            Simulator that produced the run.

        Returns
        -------
        dict
            ``{circuit_name: version_id}`` for each definition stored.

        Raises
        ------
        ValueError
            If ``db_path`` conflicts with the module's existing binding.
        RuntimeError
            If ``corner`` is ``None``, or no database is bound.
        """
        _resolve_db_path(self, db_path)
        return self._ensure_tracker().store_net_currents(
            report,
            corner,
            temperature=temperature,
            voltage=voltage,
            tool_name=tool_name,
        )

    def load_net_currents(self, corner, *, db_path=None, force=False):
        """Load persisted EM currents onto this (fresh) module — the inverse of
        store_net_currents. Loud on stale/missing data.

        Parameters
        ----------
        corner : str or ConditionState
            Process corner whose stored currents to load.
        db_path : str, optional
            Database path to adopt when the module has none.
        force : bool, default False
            Overwrite currents the user or the simulator already set.

        Returns
        -------
        int
            Number of current annotations applied.

        Raises
        ------
        ValueError
            If ``db_path`` conflicts with the module's existing binding.
        RuntimeError
            If no stored version matches this module's structure.
        """
        _resolve_db_path(self, db_path)
        return self._ensure_tracker().load_net_currents(corner, force=force)

    def set_corner(self, corner, temperature=None, voltage=None):
        """Set the current corner for subsequent result storage.

        Parameters
        ----------
        corner : str
            Process-corner label used when a store call gives none.
        temperature : float, optional
            Temperature in degrees Celsius; unchanged when ``None``.
        voltage : float, optional
            Supply voltage in volts; unchanged when ``None``.

        Returns
        -------
        None
        """
        return self._ensure_tracker().set_corner(corner, temperature, voltage)

    def view_summary(self):
        """Return the simulation summary of the bound version.

        Returns
        -------
        str or None
            The rendered summary, or ``None`` if no circuit version has been
            created.
        """
        return self._ensure_tracker().view_summary()

    def promote(self, to_stage, new_version=None, lock=True, description=""):
        """Promote the bound circuit version to a later design stage.

        Parameters
        ----------
        to_stage : {'stable', 'post_layout', 'tapeout'}
            Target stage.
        new_version : str, optional
            Version label of the promoted copy.
        lock : bool, default True
            Lock the promoted version against further writes.
        description : str, default ''
            Description of the promotion; defaults to a generated one.

        Returns
        -------
        str
            The new circuit version id, which is now the bound version.

        Raises
        ------
        ValueError
            If ``to_stage`` is not one of the listed stages.
        """
        return self._ensure_tracker().promote(to_stage, new_version, lock, description)

    def add_transistor(self, name, transistor_type, width, length,
                       multiplier=1, model=None):
        """Manually add a transistor to the bound circuit version.

        Parameters
        ----------
        name : str
            Instance name.
        transistor_type : str
            Device type, e.g. ``'nmos'``.
        width : float
            Channel width, in the PDK's user unit (micrometres for the
            shipped PDKs).
        length : float
            Channel length, in the same unit.
        multiplier : int, default 1
            Parallel multiplier.
        model : str, optional
            Model name.

        Returns
        -------
        None
        """
        return self._ensure_tracker().add_transistor(
            name, transistor_type, width, length, multiplier, model)

    def add_net_to_db(self, name, net_type="signal"):
        """Manually add a net to the bound circuit version.

        Parameters
        ----------
        name : str
            Net name.
        net_type : str, default 'signal'
            Net kind label.

        Returns
        -------
        None
        """
        return self._ensure_tracker().add_net(name, net_type)

    def get_corner_summary(self):
        """Get summary of all corners simulated.

        Returns
        -------
        dict
            ``{corner: [{'corner', 'temperature', 'voltage',
            'display_name'}, ...]}``; empty when no version is bound.
            Temperature is in degrees Celsius, voltage in volts.
        """
        return self._ensure_tracker().get_corner_summary()

    def get_corners_for_parameter(self, parameter, latest_only=False):
        """Get all corner results for a specific parameter.

        Parameters
        ----------
        parameter : str
            Metric or parameter name.
        latest_only : bool, default False
            Collapse the append-only history to the newest row per corner.

        Returns
        -------
        list
            One entry per stored result; empty when no version is bound.
        """
        return self._ensure_tracker().get_corners_for_parameter(
            parameter, latest_only=latest_only)

    def export_to_json(self, output_file):
        """Export this circuit to JSON.

        Parameters
        ----------
        output_file : str
            Path of the JSON file to write.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If no circuit version has been created.
        """
        return self._ensure_tracker().export_to_json(output_file)

    def close(self):
        """Close database connection if tracking was used."""
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None
