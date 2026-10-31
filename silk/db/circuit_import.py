"""Circuit import and export utilities.

:class:`CircuitImporter` and :class:`CircuitExporter` move circuits in and
out of a :class:`~silk.db.Database`: importing from JSON, CSV and SPICE
netlists, and exporting to JSON and CSV.
"""

import json
import csv
import re
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional

from silk._json_codec import decode_json, encode_json
from silk.db.circuit_db import Database, DesignStage
from silk.db.identity import (
    _fingerprint_spec_set_records,
    _fingerprint_structure,
)
from silk.db.models import SnapshotIdentity
from silk.db.structure_tracker import CircuitStructureTracker
from silk.circuit.port_roles import RoleRuleset
from silk.exceptions import DatabaseError
from silk.utils._utils import _parse_si

# A foreign netlist carries no house ruleset, so this call site intentionally
# pins the shipped default (§4.5). It de-hardcodes the former name->role
# ladder without parameterizing the importer; the injectable-ruleset
# guarantee lives at the infer_port_roles layer.
_ROLES = RoleRuleset.default()

# v2 (TASK-409) writes each condition as its wire scalar (a value,
# {"state": "none"|"sweep"} or null) plus a *_state key. v1 held only plain
# values and null; an older silk that reads only v1 refuses a v2 file rather
# than misreading a tagged value.
_PROJECTION_FORMAT = 'silk.circuit-projection/v2'
_PROJECTION_FORMAT_V1 = 'silk.circuit-projection/v1'
_PROJECTION_FORMATS = (_PROJECTION_FORMAT_V1, _PROJECTION_FORMAT)
_PROJECTION_OMISSIONS = (
    'specs',
    'net_currents',
    'parasitics',
    'complete_module_connectivity',
    'raw_runs',
    'waveforms',
    'simulator_logs',
    'calibration',
)


# (JSON key, operating-condition fact) for the three per-row conditions.
_CONDITION_KEYS = (
    ('corner', 'corner'),
    ('temperature', 'temperature'),
    ('voltage', 'supply'),
)


def _exported_conditions(record) -> Dict[str, Any]:
    """The wire scalars and ``*_state`` columns of one stored result."""
    from silk.simulation.conditions import ConditionState, encode_condition

    exported = {}
    for key, _fact in _CONDITION_KEYS:
        scalar = getattr(record, key)
        exported[key] = encode_condition(scalar)
        if scalar is None:
            exported[f'{key}_state'] = 'unknown'
        elif isinstance(scalar, ConditionState):
            exported[f'{key}_state'] = scalar.value
        else:
            exported[f'{key}_state'] = 'fixed'
    return exported


def _imported_conditions(sim: Dict[str, Any], *, legacy: bool = False):
    """Widened ``(corner, temperature, voltage)`` of one imported result.

    A v2 export carries ``corner_state`` / ``temperature_state`` /
    ``voltage_state`` beside the wire scalars, and the two must agree (a
    tagged scalar is self-describing without them). A v1 export
    (``legacy=True``) holds only plain values and null: a state key or a
    tagged value in it raises. It, and any export written before schema v5,
    imports per the
    v4-to-v5 migration table: a value is fixed, null is unknown (never
    none), and a null temperature whose recorded operating conditions say
    ``temperature_kind == 'sweep'`` is a sweep.
    """
    from silk.simulation.conditions import ConditionState, decode_condition

    scalars = []
    for key, fact in _CONDITION_KEYS:
        if legacy and (f'{key}_state' in sim or isinstance(sim.get(key), dict)):
            raise DatabaseError(
                f"Imported simulation result {sim.get('parameter')!r} {key}: "
                f"a {_PROJECTION_FORMAT_V1} projection holds only plain "
                f"values and null; condition states need {_PROJECTION_FORMAT}"
            )
        try:
            scalar = decode_condition(fact, sim.get(key))
        except ValueError as exc:
            raise DatabaseError(
                f"Imported simulation result {sim.get('parameter')!r} "
                f"{key}: {exc}"
            ) from exc
        state_key = f'{key}_state'
        if state_key in sim:
            state = sim[state_key]
            expected = (
                'unknown' if scalar is None
                else scalar.value if isinstance(scalar, ConditionState)
                else 'fixed'
            )
            if state != expected:
                raise DatabaseError(
                    f"Imported simulation result {sim.get('parameter')!r} "
                    f"has {state_key}={state!r} but {key}={sim.get(key)!r} "
                    f"(state {expected!r})"
                )
        elif key == 'temperature' and scalar is None:
            conditions = sim.get('conditions')
            marker = (
                conditions.get('silk_operating_conditions')
                if isinstance(conditions, dict) else None
            )
            if (
                isinstance(marker, dict)
                and marker.get('temperature_kind') == 'sweep'
            ):
                scalar = ConditionState.SWEEP
        scalars.append(scalar)
    return tuple(scalars)


_PASSIVE_KINDS = frozenset({
    'Resistor', 'Capacitor', 'Inductor', 'resistor', 'capacitor', 'inductor',
})
_PASSIVE_FIELDS = ('name', 'kind', 'value', 'pos', 'neg')


def _require_fields(record: Dict[str, Any], fields, what: str) -> None:
    """Raise a DatabaseError naming *record* and the first missing field."""
    for field in fields:
        if field not in record:
            raise DatabaseError(
                f"Imported {what} {record.get('name')!r} has no {field!r} "
                f"field; the importer does not skip records silently"
            )


def _import_structure(record: Dict[str, Any]) -> Dict[str, Any]:
    """Build the v1 structure payload for exactly what the importer stores.

    Every record is stored or refused with a :class:`DatabaseError` naming the
    record and field; none is skipped silently.
    """
    devices = []
    for transistor in record.get('transistors', []):
        _require_fields(transistor, ('name',), 'transistor')
        for dimension in ('width', 'length'):
            if transistor.get(dimension) is None:
                raise DatabaseError(
                    f"Imported transistor {transistor['name']!r} has no "
                    f"{dimension}; the importer does not skip records "
                    f"silently"
                )
        devices.append({
            'name': transistor['name'],
            'type': transistor.get('type') or 'nmos',
            'model': transistor.get('model'),
            'parameters': {
                'width': transistor.get('width'),
                'length': transistor.get('length'),
                'multiplier': transistor.get('multiplier', 1),
                'nodes': transistor.get('nodes'),
            },
        })
    stored_sources = []
    for source in record.get('sources', []):
        _require_fields(source, ('name',), 'source')
        if 'kind' in source and 'type' not in source:
            # A parsed source (SPICE import or hand-written JSON): stored as
            # a <kind>_source device with its terminals and values.
            _require_fields(source, ('positive', 'negative'), 'source')
            devices.append({
                'name': source['name'],
                'type': f"{source['kind']}_source",
                'model': source['kind'],
                'parameters': {
                    'positive': source['positive'],
                    'negative': source['negative'],
                    'dc': source.get('dc'),
                    'ac': source.get('ac'),
                },
            })
            continue
        # A tracked source entry (the shape the exporter writes): stored
        # verbatim through the structure's sources list.
        _require_fields(source, ('type', 'nets', 'parameters'), 'source')
        stored_sources.append(dict(source))
    passives = []
    for passive in record.get('passives', []):
        _require_fields(passive, _PASSIVE_FIELDS, 'passive')
        passives.append(dict(passive))
    structure = {
        'transistors': devices,
        'nets': [dict(net) for net in record.get('nets', [])],
        'hierarchy': [],
        'parameters': [],
        'passives': passives,
        'modules': {},
    }
    if stored_sources:
        structure['sources'] = stored_sources
    return structure


def _import_identity(structure: Dict[str, Any]) -> SnapshotIdentity:
    structure_fingerprint, _module_fingerprints = _fingerprint_structure(
        structure
    )
    return SnapshotIdentity(
        structure=structure_fingerprint,
        spec_set=_fingerprint_spec_set_records(()),
    )


# ---------------------------------------------------------------------------
# SPICE value parser — the canonical one, made lenient for foreign netlists
# ---------------------------------------------------------------------------

def _param_value(token: str):
    """A ``k=v`` value as a number when it parses, else the raw token.

    A parameter reference (``w=wn``) or an expression is kept verbatim so
    the consumer can see it rather than a silent ``None``.
    """
    value = _parse_spice_value(token)
    return token if value is None else value


def _parse_spice_value(s: str) -> Optional[float]:
    """Parse a SPICE numeric value with optional suffix, or ``None``.

    A thin adapter over :func:`silk.utils._utils._parse_si`, which is silk's
    one SPICE-number parser and defines the contract (case-insensitive
    ngspice suffixes, ``M`` = milli, letters after the suffix ignored, so
    ``'2.56us'`` is 2.56e-6). Examples::

        _parse_spice_value('150m')   → 0.15
        _parse_spice_value('1u')     → 1e-6
        _parse_spice_value('2.56us') → 2.56e-6
        _parse_spice_value('1.5e-6') → 1.5e-6

    The importer reads netlists silk did not write, so a value may be an
    expression it cannot evaluate (``{2*wn}``, ``'wn*2'``). Those return
    ``None`` — the field is recorded as unknown rather than aborting the
    whole import — but never silently: a ``UserWarning`` names the token, so
    a null W/L or source value in the database is traceable to its cause.
    """
    try:
        return _parse_si(s)
    except ValueError as exc:
        warnings.warn(
            f'circuit_import: {exc}; the field is stored as None. '
            f'(silk cannot evaluate parameter expressions in foreign '
            f'netlists.)',
            stacklevel=2,
        )
        return None


# ---------------------------------------------------------------------------
# CircuitImporter
# ---------------------------------------------------------------------------

class CircuitImporter:
    """Import circuits and simulation results into a database.

    Imported circuits are stored as sealed snapshots whose identity is derived
    from the imported content, with no spec declarations.

    Parameters
    ----------
    db : Database
        The destination database.

    Attributes
    ----------
    db : Database
        The destination database.

    Examples
    --------
    >>> import os, tempfile
    >>> from silk.db import CircuitExporter, CircuitImporter, Database
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     netlist = os.path.join(folder, 'divider.cir')
    ...     with open(netlist, 'w') as handle:
    ...         print('.subckt divider in out', 'R1 in out 1k', 'R2 out 0 2k',
    ...               '.ends', sep=os.linesep, file=handle)
    ...     with Database(':memory:') as db:
    ...         version_id = CircuitImporter(db).import_from_spice(netlist)
    ...         _ = db.add_simulation_result(
    ...             version_id, 'gain_dB', value=-3.5, unit='dB', sim_type='ac')
    ...         target = os.path.join(folder, 'divider.json')
    ...         CircuitExporter(db).export_to_json('divider', target)
    ...         exported = os.path.exists(target)
    >>> exported
    True
    """

    def __init__(self, db: Database):
        self.db = db

    def _store_snapshot(
        self,
        record,
        *,
        version,
        stage,
        created_by,
        description,
        netlist_path=None,
    ):
        structure = _import_structure(record)
        identity = _import_identity(structure)
        existing = self.db.find_snapshot(record['name'], stage, identity)
        if existing is not None:
            raise DatabaseError(
                f"Import for {record['name']!r} matches existing sealed "
                f"snapshot {existing!r}; refusing to attach imported "
                "evidence to a trusted identity. Import into a different "
                "database or use a different circuit name."
            )

        version_id = self.db.create_circuit(
            name=record['name'],
            version=version,
            stage=stage,
            created_by=created_by,
            description=description,
            netlist_path=netlist_path,
            identity=identity,
        )
        structure_tracker = CircuitStructureTracker(self.db, version_id)
        _digest, module_hashes = structure_tracker.compute_structure_hash(
            structure
        )
        structure_tracker.store_structure(
            structure, identity.structure.digest, module_hashes
        )
        self.db.store_spec_declarations(version_id, ())
        self.db.seal_circuit_version(version_id)
        return version_id

    # ---- JSON import ----

    def import_from_json(self, json_file: str) -> List[str]:
        """Import circuits from a JSON file.

        The file is either hand-written in the shape below or a projection written
        by :meth:`CircuitExporter.export_to_json`. A file without a
        ``projection_format`` marker is treated as hand-written; a marked file must
        be ``'silk.circuit-projection/v1'`` or ``'silk.circuit-projection/v2'``
        (a v1 file imports with null conditions read as unknown). Each circuit is
        imported in its own transaction, so a failure leaves earlier circuits of
        the same file imported.

        Expected content::

            {
                "circuits": [
                    {
                        "name": "amplifier",
                        "version": "1.0",
                        "stage": "dev",
                        "description": "...",
                        "transistors": [
                            {"name": "M1", "type": "nmos", "width": 10e-6, "length": 180e-9}
                        ],
                        "nets": [
                            {"name": "vdd", "type": "power"}
                        ],
                        "simulation_results": [
                            {"parameter": "gain", "value": 45.2, "unit": "dB"}
                        ]
                    }
                ]
            }

        Missing optional fields default to version ``'1.0'``, stage ``'dev'``,
        ``created_by='importer'`` and, per result, ``source='user'``,
        ``sim_type='dc'`` and ``tool_name='unknown'``. Widths and lengths are
        stored as given.

        Parameters
        ----------
        json_file : str
            Path of the JSON file.

        Returns
        -------
        list of str
            The version ids of the created snapshots, in file order.

        Raises
        ------
        silk.exceptions.DatabaseError
            The ``projection_format`` is unsupported, a circuit has the same identity
            as an existing sealed snapshot at that stage, or a result's condition
            fields are inconsistent.
        ValueError
            A circuit names an unknown ``stage``.
        OSError
            The file cannot be read.
        json.JSONDecodeError
            The file is not valid JSON.
        """
        with open(json_file, 'r') as f:
            data = decode_json(json.load(f))

        # A file without a marker is hand-authored input in the documented
        # shape above. A marked projection must be a version this silk reads:
        # v1 imports by the v4-to-v5 rule (null is unknown), v2 as written.
        projection_format = data.get('projection_format')
        if (
            projection_format is not None
            and projection_format not in _PROJECTION_FORMATS
        ):
            raise DatabaseError(
                f"Unsupported projection_format {projection_format!r}; this "
                f"silk imports {', '.join(_PROJECTION_FORMATS)} (or an "
                "unmarked file)"
            )
        legacy = projection_format == _PROJECTION_FORMAT_V1

        circuit_ids = []

        for circuit_data in data.get('circuits', []):
            stage = DesignStage(circuit_data.get('stage', 'dev'))
            with self.db.transaction():
                circuit_id = self._store_snapshot(
                    circuit_data,
                    version=circuit_data.get('version', '1.0'),
                    stage=stage,
                    created_by=circuit_data.get('created_by', 'importer'),
                    description=circuit_data.get('description', ''),
                )
                for sim in circuit_data.get('simulation_results', []):
                    corner, temperature, voltage = _imported_conditions(
                        sim, legacy=legacy)
                    self.db.add_simulation_result(
                        circuit_version_id=circuit_id,
                        parameter=sim['parameter'],
                        value=sim.get('value'),
                        value_text=sim.get('value_text'),
                        unit=sim.get('unit'),
                        object_id=sim.get('object_id'),
                        object_type=sim.get('object_type', 'circuit'),
                        source=sim.get('source', 'user'),
                        sim_type=sim.get('sim_type', 'dc'),
                        corner=corner,
                        temperature=temperature,
                        voltage=voltage,
                        tool_name=sim.get('tool_name', 'unknown'),
                        conditions=sim.get('conditions'),
                        timestamp=sim.get('timestamp'),
                    )
            circuit_ids.append(circuit_id)

        return circuit_ids

    # ---- CSV import ----

    def import_simulation_results_csv(self, csv_file: str,
                                      circuit_version_id: str,
                                      sim_type: str = "dc",
                                      tool_name: str = "unknown") -> int:
        """Import simulation results from a CSV file.

        The file needs a header with a required ``parameter`` column; ``value``,
        ``value_text`` and ``unit`` are read when present. A nonempty
        ``value`` cell is numeric (including ``nan`` for unavailable observations);
        absent or empty text beside it means no text, while nonempty text refuses
        as both values. With an empty numeric cell, a present text cell is kept
        exactly, including empty text and whitespace. An absent text field
        (including a short row) then refuses as neither value.

        CSV ``sim_type``, ``tool_name`` and ``timestamp`` columns are ignored:
        analysis and tool labels come from the arguments, and the database assigns
        a new insertion timestamp. This imports measurement values, not complete
        evidence or metadata. Each row is
        added with :meth:`silk.db.Database.add_simulation_result`. The rows are not
        grouped into a transaction, so a failing row leaves the earlier rows stored
        unless the caller wraps the call in :meth:`silk.db.Database.transaction`.

        Parameters
        ----------
        csv_file : str
            Path of the CSV file.
        circuit_version_id : str
            Version id of a sealed, unlocked snapshot to attach the results to.
        sim_type : str, default "dc"
            Analysis label stored with every row.
        tool_name : str, default "unknown"
            Tool name stored with every row.

        Returns
        -------
        int
            The number of rows imported.

        Raises
        ------
        KeyError
            A row has no ``parameter`` column.
        ValueError
            A ``value`` is not a number, or a row has both or neither of a value and
            text value.
        silk.exceptions.DatabaseError
            The snapshot does not exist or is not sealed.
        silk.exceptions.LockedVersionError
            The snapshot is locked.
        OSError
            The file cannot be read.
        """
        count = 0
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                value = float(row['value']) if row.get('value') else None
                text = row.get('value_text')
                if value is not None and text == '':
                    text = None
                self.db.add_simulation_result(
                    circuit_version_id=circuit_version_id,
                    parameter=row['parameter'],
                    value=value,
                    value_text=text,
                    unit=row.get('unit'),
                    sim_type=sim_type,
                    tool_name=tool_name,
                )
                count += 1
        return count

    # ---- SPICE netlist import ----

    def parse_spice_netlist(self, netlist_file: str, *,
                            device_prefixes=None) -> Dict[str, Any]:
        """Parse a SPICE netlist and extract its circuit elements.

        Recognised lines are devices, passives (``R``, ``C``, ``L``) and sources
        (``V``, ``I``); comments and dot commands other than ``.subckt`` are
        skipped. By default a device is a line starting with ``M`` (not ``MO``) or
        ``XM``, with four nodes then the model. With ``device_prefixes`` — the
        instance prefixes a PDK declares, for example ``('D', 'XM')`` — a device is
        instead any line whose instance name starts with one of them
        (case-insensitive, longest prefix wins); its model is the last token without
        ``=`` and its nodes are the tokens in between, so a two-terminal diode
        parses as faithfully as a four-terminal transistor. Net roles are guessed
        from net names with the shipped default role ruleset and marked
        ``'inferred'``.

        Values that cannot be evaluated (parameter expressions such as
        ``{2*wn}``) are stored as ``None`` and a :class:`UserWarning` names the
        token.

        Parameters
        ----------
        netlist_file : str
            Path of the netlist.
        device_prefixes : iterable of str, optional
            PDK device instance prefixes (keyword-only). ``None`` keeps the default
            ``M``/``XM`` detection.

        Returns
        -------
        dict
            ``{'name': str, 'transistors': list, 'passives': list, 'sources': list,
            'nets': list}``. ``name`` is the ``.subckt`` name or the file stem. Each
            transistor dict has ``name``, ``prefix``, ``nodes``, ``model``, ``type``
            (``'nmos'``, ``'pmos'`` or ``None``, guessed from the model name),
            ``width``, ``length`` (numbers parsed from ``w=``/``l=`` with SPICE
            suffixes applied, so ``w=10u`` gives ``1e-05``),
            ``multiplier`` and ``params``. Each passive dict has ``name``, ``kind``,
            ``pos``, ``neg`` and ``value``; each source dict has ``name``, ``kind``,
            ``positive``, ``negative``, ``dc`` and ``ac``; each net dict has
            ``name``, ``type`` and, when the role was guessed, ``inferred``.

        Raises
        ------
        OSError
            The file cannot be read.
        """
        with open(netlist_file, 'r') as f:
            lines = f.readlines()

        circuit_name = None
        transistors = []
        passives = []
        source_list = []
        nets = set()
        # Longest prefix first so 'XM' wins over 'X' when a PDK declares both.
        prefixes = (None if device_prefixes is None
                    else sorted({str(pf) for pf in device_prefixes},
                                key=len, reverse=True))

        for line in lines:
            line = line.strip()

            if not line or line.startswith('*') or line.startswith('.end'):
                continue

            # Extract subcircuit name
            if line.lower().startswith('.subckt'):
                parts = line.split()
                if len(parts) > 1:
                    circuit_name = parts[1]
                continue

            # Skip control blocks and dot commands
            if line.startswith('.'):
                continue

            upper = line.upper()
            parts = line.split()

            # --- Devices: PDK-declared prefixes, or legacy M... / XM... ---
            if prefixes is None:
                matched = None
                is_device = upper.startswith('XM') or (
                    upper.startswith('M') and not upper.startswith('MO'))
            else:
                matched = next((pf for pf in prefixes
                                if upper.startswith(pf.upper())), None)
                is_device = matched is not None

            if is_device:
                if prefixes is None:
                    if len(parts) < 5:
                        continue
                    nodes = parts[1:5]
                    model = parts[5] if len(parts) > 5 else parts[4]
                    param_str = ' '.join(parts[5:])
                else:
                    # Normalise `w = 4` to `w=4` so an assignment is one token.
                    tokens = re.sub(r'\s*=\s*', '=', line).split()
                    positional = [tok for tok in tokens[1:] if '=' not in tok]
                    if len(positional) < 2:      # need >= 1 node and a model
                        continue
                    model = positional[-1]
                    nodes = positional[:-1]
                    param_str = ' '.join(tok for tok in tokens[1:] if '=' in tok)

                trans = {
                    'name': parts[0],
                    'prefix': matched,
                    'nodes': nodes,
                    'model': model,
                    'type': None,
                    'width': None,
                    'length': None,
                    'multiplier': 1,
                    'params': {
                        k.lower(): _param_value(v)
                        for k, v in re.findall(r'(\w+)\s*=\s*(\S+)', param_str)
                    },
                }

                w_match = re.search(r'\bw\s*=\s*(\S+)', param_str, re.IGNORECASE)
                if w_match:
                    trans['width'] = _parse_spice_value(w_match.group(1))

                l_match = re.search(r'\bl\s*=\s*(\S+)', param_str, re.IGNORECASE)
                if l_match:
                    trans['length'] = _parse_spice_value(l_match.group(1))

                m_match = re.search(r'\bm\s*=\s*(\d+)', param_str, re.IGNORECASE)
                if m_match:
                    trans['multiplier'] = int(m_match.group(1))

                if trans['model']:
                    model_lower = trans['model'].lower()
                    if any(k in model_lower for k in ('nmos', 'nfet', 'n_')):
                        trans['type'] = 'nmos'
                    elif any(k in model_lower for k in ('pmos', 'pfet', 'p_')):
                        trans['type'] = 'pmos'

                transistors.append(trans)
                for node in trans['nodes']:
                    if node and not node.startswith('$'):
                        nets.add(node)

            # --- Passives: R, C, L ---
            elif upper[0] in ('R', 'C', 'L') and len(parts) >= 3:
                kind = {'R': 'resistor', 'C': 'capacitor', 'L': 'inductor'}[upper[0]]
                passive = {
                    'name': parts[0],
                    'kind': kind,
                    'pos': parts[1],
                    'neg': parts[2],
                    'value': _parse_spice_value(parts[3]) if len(parts) > 3 else None,
                }
                passives.append(passive)
                nets.add(parts[1])
                nets.add(parts[2])

            # --- Sources: V, I ---
            elif upper[0] in ('V', 'I') and len(parts) >= 3:
                kind = 'voltage' if upper[0] == 'V' else 'current'
                source = {
                    'name': parts[0],
                    'kind': kind,
                    'positive': parts[1],
                    'negative': parts[2],
                    'dc': None,
                    'ac': None,
                }
                # Parse DC/AC values from remainder
                rest = ' '.join(parts[3:]).upper()
                dc_match = re.search(r'DC\s+(\S+?)(?:V|A)?(?:\s|$)', rest)
                if dc_match:
                    source['dc'] = _parse_spice_value(dc_match.group(1))
                ac_match = re.search(r'AC\s+(\S+?)(?:V|A)?(?:\s|$)', rest)
                if ac_match:
                    source['ac'] = _parse_spice_value(ac_match.group(1))
                source_list.append(source)
                nets.add(parts[1])
                nets.add(parts[2])

        # Classify net types by name via the shipped default role ruleset
        # (patterns de-hardcoded, §4.5). PortRole.POWER.value == 'power'.
        # A name match is a guess, not a statement from the netlist (a
        # foreign SPICE file carries no port-role provenance at all) — mark
        # it 'inferred' the same way infer_port_roles marks ModulePin.role
        # (TASK-42/97), sparsely: the key is present (and True) only for a
        # net the ruleset actually classified, omitted for the 'signal'
        # default (nothing was inferred, so there is nothing to mark).
        net_list = []
        for net in sorted(nets):
            role = _ROLES.classify(net)
            net_type = role.value if role is not None else 'signal'
            entry = {'name': net, 'type': net_type}
            if role is not None:
                entry['inferred'] = True
            net_list.append(entry)

        return {
            'name': circuit_name or Path(netlist_file).stem,
            'transistors': transistors,
            'passives': passives,
            'sources': source_list,
            'nets': net_list,
        }

    def import_from_spice(self, netlist_file: str, version: str = "1.0",
                          stage: DesignStage = DesignStage.DEV,
                          description: str = "") -> str:
        """Import a circuit from a SPICE netlist as a sealed snapshot.

        The netlist is read with :meth:`parse_spice_netlist` and stored with the
        creator ``'spice_importer'`` and the absolute netlist path.

        Parameters
        ----------
        netlist_file : str
            Path of the netlist.
        version : str, default "1.0"
            Version label.
        stage : DesignStage, optional
            Design stage; ``DesignStage.DEV`` by default.
        description : str, default ""
            Description; ``'Imported from <netlist_file>'`` when empty.

        Returns
        -------
        str
            The version id of the new snapshot.

        Raises
        ------
        silk.exceptions.DatabaseError
            A sealed snapshot with the same content identity already exists for this
            circuit name and stage.
        OSError
            The file cannot be read.
        """
        parsed = self.parse_spice_netlist(netlist_file)

        with self.db.transaction():
            circuit_id = self._store_snapshot(
                parsed,
                version=version,
                stage=stage,
                created_by='spice_importer',
                description=description or f"Imported from {netlist_file}",
                netlist_path=str(Path(netlist_file).absolute()),
            )

        return circuit_id


# ---------------------------------------------------------------------------
# CircuitExporter
# ---------------------------------------------------------------------------

class CircuitExporter:
    """Export circuits from a database to JSON and CSV.

    Parameters
    ----------
    db : Database
        The source database.

    Attributes
    ----------
    db : Database
        The source database.
    """

    def __init__(self, db: Database):
        self.db = db

    def export_to_json(self, circuit_name: str, output_file: str,
                       version: str = None, include_results: bool = True):
        """Export a lossy, versioned interoperability projection as JSON.

        The projection can be re-imported with
        :meth:`CircuitImporter.import_from_json`, but it is not a database backup or
        a complete circuit archive: version and evidence UUIDs, specs, net
        currents, parasitics, raw runs and other evidence are omitted (the file's
        ``projection_scope.omits`` lists them). Import recomputes the snapshot
        identity from the content and allocates new ids; an exact identity
        collision in the destination is refused before any evidence is written.
        Nothing is written when the circuit is not found.

        Parameters
        ----------
        circuit_name : str
            Circuit name.
        output_file : str
            Path of the JSON file to write.
        version : str, optional
            Version label; the newest sealed snapshot when omitted.
        include_results : bool, default True
            Also write the simulation results.

        Raises
        ------
        silk.exceptions.DatabaseError
            The selected snapshot is not sealed (its results cannot be read), or the
            label is ambiguous.
        OSError
            The file cannot be written.
        """
        summary = self.db.get_circuit_summary(circuit_name, version)
        if not summary:
            return

        circuit = summary['circuit']

        # Unpack device properties. TWO writer conventions share the
        # devices.properties JSON column: the auto-tracker
        # (create_circuit_version → store_structure) writes PDK-native keys
        # ('w'/'l'/'m'), while the manual add_transistor/import paths write
        # 'width'/'length'/'multiplier'. Read both — and never invent a
        # value: the old `.get('multiplier', 1)` exported multiplier=1 for a
        # real m=4 device (2026-08-24 review, finding B4). Unknown is None.
        transistors = []
        passives = []
        source_records = []
        for d in summary['devices']:
            props = decode_json(json.loads(d['properties'])) if d.get('properties') else {}
            if d['type'] in ('voltage_source', 'current_source'):
                # A parsed source row (SPICE import): its stored facts.
                source_records.append({
                    'name': d['name'],
                    'kind': d['model'],
                    'positive': props.get('positive'),
                    'negative': props.get('negative'),
                    'dc': props.get('dc'),
                    'ac': props.get('ac'),
                })
            elif d['type'] == 'silk_source':
                # The tracked source entry, as stored (the kind marker is
                # added back by the structure store).
                source_records.append(
                    {k: v for k, v in props.items() if k != 'kind'})
            elif d['type'] in _PASSIVE_KINDS:
                passives.append({
                    'name': d['name'],
                    'kind': d['type'],
                    'value': props.get('value'),
                    'pos': props.get('pos'),
                    'neg': props.get('neg'),
                })
            else:
                transistors.append({
                    'name': d['name'],
                    'type': d['type'],
                    'model': d.get('model'),
                    'width': props.get('width', props.get('w')),
                    'length': props.get('length', props.get('l')),
                    'multiplier': props.get('multiplier', props.get('m')),
                })

        export_data = {
            'projection_format': _PROJECTION_FORMAT,
            'projection_scope': {
                'purpose': 'lossy interoperability projection',
                'is_database_backup': False,
                'is_circuit_archive': False,
                'omits': list(_PROJECTION_OMISSIONS),
            },
            'circuits': [{
                'name': circuit['circuit_name'],
                'version': circuit['version'],
                'stage': circuit['stage'],
                'description': circuit.get('description', ''),
                'created_at': circuit.get('created_at', ''),
                'created_by': circuit.get('created_by', ''),
                'transistors': transistors,
                'passives': passives,
                'sources': source_records,
                'nets': [
                    {
                        'name': n['name'],
                        'type': n['net_type'],
                        # Whether net_type came from name-based guessing
                        # rather than a declaration (TASK-97/112), mirrors
                        # get_circuit_summary's 'role_inferred'; False for
                        # both a declared net and a pre-TASK-112 row with
                        # no stored properties -- indistinguishable at
                        # this point.
                        'role_inferred': n.get('role_inferred', False),
                    }
                    for n in summary['nets']
                ],
            }]
        }

        if include_results:
            export_data['circuits'][0]['simulation_results'] = [
                {
                    'parameter': r.parameter,
                    'value': r.value,
                    'value_text': r.value_text,
                    'unit': r.unit,
                    'object_id': r.object_id,
                    'object_type': r.object_type,
                    'source': r.source,
                    'sim_type': r.sim_type,
                    # Schema v5 (TASK-409): each condition as its wire
                    # scalar (a value, {"state": "none"|"sweep"} or null)
                    # plus the stored state column.
                    **_exported_conditions(r),
                    'tool_name': r.tool_name,
                    'conditions': r.conditions,
                    'timestamp': r.timestamp,
                }
                for r in summary['simulation_results']
            ]

        with open(output_file, 'w') as f:
            json.dump(encode_json(export_data), f, allow_nan=False, indent=2)

    def export_results_to_csv(self, circuit_name: str, output_file: str,
                              version: str = None):
        """Export simulation results to a CSV file.

        The header is ``parameter, value, value_text, unit, sim_type, tool_name,
        timestamp``. Numeric observations use the numeric cell, including ``nan``
        for unavailable values; text uses the text cell verbatim, including empty
        text. The CSV importer preserves these values but assigns new metadata
        according to its arguments and database insertion time. Nothing is written
        when the circuit is not found.

        Parameters
        ----------
        circuit_name : str
            Circuit name.
        output_file : str
            Path of the CSV file to write.
        version : str, optional
            Version label; the newest sealed snapshot when omitted.

        Raises
        ------
        silk.exceptions.DatabaseError
            The selected snapshot is not sealed, or the label is ambiguous.
        OSError
            The file cannot be written.
        """
        if version:
            circuit = self.db.get_version(circuit_name, version)
        else:
            circuit = self.db.get_latest(circuit_name)

        if not circuit:
            return

        results = self.db.get_simulation_results(circuit['version_id'])

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['parameter', 'value', 'value_text', 'unit',
                             'sim_type', 'tool_name', 'timestamp'])
            for r in results:
                writer.writerow([
                    r.parameter,
                    r.value if r.value is not None else '',
                    r.value_text or '',
                    r.unit or '',
                    r.sim_type or '',
                    r.tool_name or '',
                    r.timestamp,
                ])
