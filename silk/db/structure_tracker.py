"""
Circuit Structure Tracker
Intelligent tracking of circuit structure with change detection and run counting.

This module handles:
- Automatic structure extraction from Module
- Change detection (only store if structure changed)
- Run counting for each unique structure
- Complete transistor and net capture
- Module filtering (exclude testbench)
- Module-level change tracking with individual module hashes
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from silk._json_codec import decode_json, encode_json
from silk.db.identity import (
    STRUCTURE_FINGERPRINT_SCHEME,
    _fingerprint_module_structure,
    _fingerprint_structure,
    _fingerprints_equal,
    _require_fingerprint_scheme,
)
from silk.db.models import Fingerprint
from silk.exceptions import CircuitError, DatabaseError, EvidenceMismatchError
from silk.utils._utils import _actual_values, _render_param
from silk.db._source_identity import source_entry


_IDEAL_PASSIVE_KIND_BY_PREFIX = {'R': 'Resistor', 'C': 'Capacitor', 'L': 'Inductor'}


def _passive_value(name: str, value: Any) -> Any:
    """A passive's value as stored and hashed: the number it emits.

    A Variable or Variable expression is replaced by its current value. Before, the object itself reached the structure, so the
    devices row held its repr or expression text (``2*r_load``) and an
    expression's structure hash ignored the value. A literal value is
    returned unchanged, so circuits without Variables keep their hashes, and
    a Variable holding 1000.0 hashes the same as the literal 1000.0.
    """
    try:
        return _actual_values(value)
    except ValueError as exc:
        raise CircuitError(
            f"Cannot extract structure for passive {name!r}: {exc}. "
            "Assign the Variable a value before storing or hashing the circuit."
        ) from exc


def _ideal_passive_canonical_entry(name: str, device: Any) -> Optional[Dict[str, Any]]:
    """Canonical ``structure['passives']`` entry for *device*, or ``None``.

    Identifies an ideal two-terminal R/C/L by WHAT it emits (ideal, category
    'passive', spice_prefix R/C/L) rather than by which class built it, so a
    hand-written ``silk.elements.passives.Resistor`` and a YAML-built
    ``silk.elements.elements.resistor`` with the same name/value/nets produce
    the byte-identical entry — one ``silk.structure/v3`` identity shared by
    both families. A hand-written
    ``PassiveBase`` instance (whether read from ``elem_dict`` or
    ``passives_dict``) has no ``_is_pdk_device``/``_category``/``_spice_prefix``
    of its own (Silk added ``_spice_prefix``/``_category``, but not
    ``_is_pdk_device``), so it always returns ``None`` here and today's
    hand-written-passive hashes stay byte-for-byte untouched — see
    :func:`_passive_dict_entry` for the ``passives_dict`` fallback.
    """
    if getattr(device, '_is_pdk_device', True):
        return None
    if getattr(device, '_category', None) != 'passive':
        return None
    kind = _IDEAL_PASSIVE_KIND_BY_PREFIX.get(getattr(device, '_spice_prefix', None))
    if kind is None:
        return None
    pin_names = getattr(type(device), '_pin_names', ())
    param_names = getattr(type(device), '_param_names', ())
    if len(pin_names) != 2 or len(param_names) != 1:
        return None
    nets = getattr(device, 'nets', None) or {}
    params = getattr(device, 'params', None) or {}
    return {
        'name': name,
        'kind': kind,
        'value': _passive_value(name, params.get(param_names[0])),
        'pos': nets.get(pin_names[0]),
        'neg': nets.get(pin_names[1]),
    }


def _passive_dict_entry(name: str, passive: Any) -> Dict[str, Any]:
    """``structure['passives']`` entry for a ``passives_dict`` value.

    ``passives_dict`` can hold either a genuine
    ``PassiveBase`` instance (``.value``/``.pos``/``.neg`` attributes) or the
    ideal ``AnalogElement`` that ``silk.elements.passives.Resistor/Capacitor/
    Inductor`` now build under the hood (``.params``/``.nets``, no ``.pos``/
    ``.neg``). Try the canonical (attribute-agnostic) extraction first; a
    ``PassiveBase`` instance always fails it (see
    :func:`_ideal_passive_canonical_entry`) and falls back to reading its own
    attributes exactly as before.
    """
    return _ideal_passive_canonical_entry(name, passive) or {
        'name': name,
        'kind': type(passive).__name__,
        'value': _passive_value(name, passive.value),
        'pos': passive.pos,
        'neg': passive.neg,
    }


class CircuitStructureTracker:
    """
    Tracks circuit structure with intelligent change detection

    Features:
    - Extracts structure from Module Module elements
    - Computes structure hash for change detection
    - Tracks run count per unique structure
    - Stores complete transistor parameters and connections
    - Tracks individual module changes via module hashes
    """

    def __init__(self, database, circuit_id: Optional[str], debug: bool = False,
                 *, _sources=True):
        """
        Initialize structure tracker

        Args:
            database: Database instance
            circuit_id: Circuit version ID from database (can be None initially)
            debug: Print debug information
        """
        self.db = database
        self.circuit_id = circuit_id  # Can be None, will be set later
        self.debug = debug
        self._sources = _sources

        # Cache of last stored structure
        self._last_structure_hash = None
        self._last_structure_fingerprint = None
        self._current_run_count = 0
        self._last_module_hashes = {}  # Track previous module hashes
        self._last_module_fingerprints = {}

        # Load existing structure hash if available
        if circuit_id:
            self._load_last_structure()

    def _load_last_structure(self):
        """Load the last stored structure hash and run count"""
        row = self.db.conn.execute("""
            SELECT state, structure_scheme, structure_hash, metadata
            FROM circuit_versions
            WHERE version_id = ?
        """, (self.circuit_id,)).fetchone()
        if row is None:
            return
        stored_structure = Fingerprint(
            row["structure_scheme"], row["structure_hash"]
        )
        _require_fingerprint_scheme(
            stored_structure, STRUCTURE_FINGERPRINT_SCHEME
        )
        if row["state"] == "sealed":
            self._last_structure_fingerprint = stored_structure
            self._last_structure_hash = stored_structure.digest
        if row["metadata"]:
            try:
                metadata = decode_json(json.loads(row["metadata"]))
                self._current_run_count = metadata.get('run_count', 0)
            except (json.JSONDecodeError, TypeError):
                pass
        records = self.db.conn.execute(
            """
            SELECT cm.instance_name, m.module_hash_scheme, m.module_hash
            FROM circuit_modules AS cm
            JOIN modules AS m ON m.module_id = cm.module_id
            WHERE cm.circuit_version_id = ?
            """,
            (self.circuit_id,),
        ).fetchall()
        for record in records:
            fingerprint = Fingerprint(
                record["module_hash_scheme"], record["module_hash"]
            )
            _require_fingerprint_scheme(
                fingerprint, STRUCTURE_FINGERPRINT_SCHEME
            )
            instance_name = record["instance_name"]
            self._last_module_fingerprints[instance_name] = fingerprint
            self._last_module_hashes[instance_name] = fingerprint.digest

    def extract_structure(self, netlist) -> Dict[str, Any]:
        """
        Extract complete circuit structure from Module

        Handles Module's dict-based elem_dict structure:
        elem_dict[name] = {'device': <object>, 'type': 'd' or 'm'}

        Recursively extracts devices from Module instances.
        Captures all devices (transistors, resistors, capacitors, etc.)
        Captures passives (resistors, capacitors, inductors) from passives_dict.
        Includes sources owned by this root and placed descendants; registered
        testbenches are outside this traversal.

        Args:
            netlist: Module instance

        Returns:
            dict: Complete structure including devices, nets, hierarchy, and modules with hashes
        """
        structure = {
            'transistors': [],  # Keep name for compatibility (contains all devices)
            'nets': [],
            'hierarchy': [],
            'parameters': [],
            'passives': [],  # Resistors, capacitors, inductors (from passives_dict)
            'modules': {}  # Track modules: {module_path: {devices: [...], hash: '...', device_count: N}}
        }
        self._extract_sources(netlist, structure)

        if self.debug:
            print("\n🔍 Extracting circuit structure...")

        # Track which devices and nets we've already added to avoid duplicates
        processed_devices = set()
        processed_nets = set()

        # Extract devices and subcircuits from elem_dict
        for elem_name, elem_entry in netlist.elem_dict.items():
            if self.debug:
                print(f"  Examining element: {elem_name}")

            # Handle dict-based structure: elem_dict[name] = {'device': obj, 'type': ...}
            # type can be string 'd'/'m' or ElementType.DEVICE/MODULE enum
            if isinstance(elem_entry, dict):
                elem_data = elem_entry.get('device')
                elem_type = str(elem_entry.get('type', ''))

                if self.debug:
                    print(f"    Dict entry: type='{elem_type}', device={type(elem_data).__name__}")

                if elem_type in ('m', 'ElementType.MODULE'):
                    # This is a module
                    module_info = self._extract_module_info(elem_name, elem_data, netlist, local_name=elem_name)
                    structure['hierarchy'].append(module_info)
                    if self.debug:
                        print(f"    ✓ Added module: {elem_name}")

                    # Recursively extract devices from inside the module
                    self._extract_from_module(elem_data, structure, prefix=elem_name, processed=processed_devices, processed_nets=processed_nets)

                elif elem_type in ('d', 'ElementType.DEVICE'):
                    # This is a device (transistor, resistor, etc.)
                    if elem_name not in processed_devices:
                        passive_entry = _ideal_passive_canonical_entry(elem_name, elem_data)
                        if passive_entry is not None:
                            structure['passives'].append(passive_entry)
                            processed_devices.add(elem_name)
                            if self.debug:
                                print(f"    ✓ Added ideal passive: {elem_name}")
                        else:
                            device_info = self._extract_transistor_info(elem_name, elem_data, netlist)
                            if device_info:
                                structure['transistors'].append(device_info)
                                processed_devices.add(elem_name)
                                if self.debug:
                                    print(f"    ✓ Added device: {elem_name}")

            else:
                # Handle direct object (old style, for compatibility)
                if self.debug:
                    print(f"    Direct object: {type(elem_entry).__name__}")

                if self._is_pymodule(elem_entry):
                    # This is a subcircuit/module
                    module_info = self._extract_module_info(elem_name, elem_entry, netlist, local_name=elem_name)
                    structure['hierarchy'].append(module_info)
                    if self.debug:
                        print(f"    ✓ Added module: {elem_name}")

                    # Recursively extract devices from inside the module
                    self._extract_from_module(elem_entry, structure, prefix=elem_name, processed=processed_devices, processed_nets=processed_nets)

                elif self._is_transistor(elem_entry):
                    # This is a device at top level
                    if elem_name not in processed_devices:
                        passive_entry = _ideal_passive_canonical_entry(elem_name, elem_entry)
                        if passive_entry is not None:
                            structure['passives'].append(passive_entry)
                            processed_devices.add(elem_name)
                            if self.debug:
                                print(f"    ✓ Added ideal passive: {elem_name}")
                        else:
                            device_info = self._extract_transistor_info(elem_name, elem_entry, netlist)
                            if device_info:
                                structure['transistors'].append(device_info)
                                processed_devices.add(elem_name)
                                if self.debug:
                                    print(f"    ✓ Added device: {elem_name}")

        # Extract passives (R/C/L) from passives_dict
        for passive_name, passive in (getattr(netlist, 'passives_dict', None) or {}).items():
            structure['passives'].append(_passive_dict_entry(passive_name, passive))

        # Extract nets from net_dict
        for net_name, net_data in netlist.net_dict.items():
            if net_name not in processed_nets:
                net_info = self._extract_net_info(net_name, net_data)
                structure['nets'].append(net_info)
                processed_nets.add(net_name)

        # Process module_dict to track individual module structures and hashes
        # This is ONLY for module hash tracking, not for device extraction
        if hasattr(netlist, 'module_dict'):
            if self.debug:
                print(f"\n  Computing module hashes ({len(netlist.module_dict)} modules)...")

            for module_name, module_entry in netlist.module_dict.items():
                if self.debug:
                    print(f"    Processing module_dict['{module_name}']")

                # module_dict uses dict structure: {'device': obj, 'defined': bool}
                if isinstance(module_entry, dict):
                    module_device = module_entry.get('device')
                    if module_device:
                        module_name = module_entry.get('_structural_name', module_name)
                        # Extract module structure (for hash computation only)
                        module_structure = self._extract_module_structure(module_device, module_name)

                        # Compute hash for this module
                        module_hash = self._compute_module_hash(module_structure)

                        # Store module info with hash
                        structure['modules'][module_name] = {
                            'devices': module_structure['devices'],
                            'nets': module_structure['nets'],
                            'submodules': module_structure['submodules'],
                            'passives': module_structure['passives'],
                            'hash': module_hash,
                            'device_count': len(module_structure['devices'])
                        }
                        if module_structure.get('sources'):
                            structure['modules'][module_name]['sources'] = module_structure['sources']

                        if self.debug:
                            print(f"      ✓ Module '{module_name}' - {len(module_structure['devices'])} devices, hash: {module_hash[:16]}...")

        if self.debug:
            print(f"\n📊 Structure summary:")
            print(f"  Devices: {len(structure['transistors'])}")
            print(f"  Nets: {len(structure['nets'])}")
            print(f"  Hierarchy: {len(structure['hierarchy'])}")
            print(f"  Modules: {len(structure['modules'])}")

        return structure

    def _extract_sources(self, module, structure, prefix=''):
        if not self._sources:
            return
        for name, source in (getattr(module, 'sig_sources_dict', None) or {}).items():
            full_name = f'{prefix}.{name}' if prefix else name
            structure.setdefault('sources', []).append(source_entry(full_name, source))

    def _extract_from_module(self, module, structure: Dict, prefix: str = "",
                             processed: set = None, processed_nets: set = None):
        """
        Recursively extract devices and nets from inside a Module

        Handles both dict-based and direct object structures.
        Captures all devices (type='d') - transistors, resistors, capacitors, etc.

        Args:
            module: Module instance
            structure: Structure dict to populate
            prefix: Hierarchical prefix for naming (e.g., "ldo_top.stage1")
            processed: Set of already processed device names to avoid duplicates
        """
        if processed is None:
            processed = set()
        if processed_nets is None:
            processed_nets = set()
        self._extract_sources(module, structure, prefix)

        if self.debug:
            print(f"    → Recursing into module: {prefix}")

        # Check if module has elem_dict (circuit elements)
        if hasattr(module, 'elem_dict'):
            if self.debug:
                print(f"      Module has elem_dict with {len(module.elem_dict)} elements")

            for elem_name, elem_entry in module.elem_dict.items():
                full_name = f"{prefix}.{elem_name}" if prefix else elem_name

                if self.debug:
                    print(f"      Examining: {full_name}")

                # Handle dict-based structure: {'device': obj, 'type': ...}
                if isinstance(elem_entry, dict):
                    elem_data = elem_entry.get('device')
                    elem_type = str(elem_entry.get('type', ''))

                    if self.debug:
                        print(f"        Dict entry: type='{elem_type}', device={type(elem_data).__name__}")

                    if elem_type in ('m', 'ElementType.MODULE'):
                        structure['hierarchy'].append(
                            self._extract_module_info(full_name, elem_data, module, local_name=elem_name)
                        )
                        self._extract_from_module(elem_data, structure, prefix=full_name, processed=processed, processed_nets=processed_nets)

                    elif elem_type in ('d', 'ElementType.DEVICE'):
                        # Found a device! (transistor, resistor, etc.)
                        if full_name not in processed:
                            passive_entry = _ideal_passive_canonical_entry(full_name, elem_data)
                            if passive_entry is not None:
                                structure['passives'].append(passive_entry)
                                processed.add(full_name)
                                if self.debug:
                                    print(f"        ✓ Added ideal passive: {full_name}")
                            else:
                                device_info = self._extract_transistor_info(full_name, elem_data, module)
                                if device_info:
                                    structure['transistors'].append(device_info)
                                    processed.add(full_name)
                                    if self.debug:
                                        print(f"        ✓ Added device: {full_name}")

                else:
                    # Handle direct object (old style)
                    if self.debug:
                        print(f"        Direct object: {type(elem_entry).__name__}")

                    if self._is_pymodule(elem_entry):
                        structure['hierarchy'].append(
                            self._extract_module_info(full_name, elem_entry, module, local_name=elem_name)
                        )
                        self._extract_from_module(elem_entry, structure, prefix=full_name, processed=processed, processed_nets=processed_nets)

                    elif self._is_transistor(elem_entry):
                        # Found a device!
                        if full_name not in processed:
                            passive_entry = _ideal_passive_canonical_entry(full_name, elem_entry)
                            if passive_entry is not None:
                                structure['passives'].append(passive_entry)
                                processed.add(full_name)
                                if self.debug:
                                    print(f"        ✓ Added ideal passive: {full_name}")
                            else:
                                device_info = self._extract_transistor_info(full_name, elem_entry, module)
                                if device_info:
                                    structure['transistors'].append(device_info)
                                    processed.add(full_name)
                                    if self.debug:
                                        print(f"        ✓ Added device: {full_name}")
        else:
            if self.debug:
                print(f"      Module has no elem_dict")

        # Also extract passives (R/C/L) from this module's passives_dict
        for passive_name, passive in (getattr(module, 'passives_dict', None) or {}).items():
            full_name = f"{prefix}.{passive_name}" if prefix else passive_name
            if full_name not in processed:
                structure['passives'].append(_passive_dict_entry(full_name, passive))
                processed.add(full_name)
                if self.debug:
                    print(f"        ✓ Added passive: {full_name}")

        # Also extract nets from module if available
        if hasattr(module, 'net_dict'):
            for net_name, net_data in module.net_dict.items():
                if net_name not in processed_nets:
                    net_info = self._extract_net_info(net_name, net_data)
                    structure['nets'].append(net_info)
                    processed_nets.add(net_name)
                    if self.debug:
                        print(f"        ✓ Added net: {net_name}")

    def _is_pymodule(self, elem_data) -> bool:
        """Check if element is a Module (subcircuit)"""
        # Check if it's a Module instance
        return type(elem_data).__name__ == 'Module' or \
               hasattr(elem_data, 'module_names') or \
               hasattr(elem_data, 'elem_dict')

    def _is_transistor(self, elem_data) -> bool:
        """
        Check if element is a device (transistor, resistor, capacitor, etc.)

        Returns False for non-device objects (strings, ints, None, etc.)
        that may appear in old-style elem_dict entries.
        """
        # Legacy Module attributes
        if hasattr(elem_data, 'deviceType'):
            return True
        if hasattr(elem_data, 'parameters'):
            return True
        if hasattr(elem_data, 'pin_list'):
            return True

        # Silk AnalogElement attributes
        if hasattr(elem_data, 'params') and hasattr(elem_data, 'nets'):
            return True

        return False

    def _extract_transistor_info(self, name: str, device, netlist) -> Optional[Dict]:
        """
        Extract complete device information (transistor, resistor, capacitor, etc.)

        Returns dict with:
        - name
        - type (if available, e.g., 'nmos', 'pmos', 'resistor')
        - parameters as PDK-native SPICE text (w, l, m, r, c, etc.)
        - all net connections
        - model (if available)
        """
        param_name = None
        try:
            info = {
                'name': name,
                'type': 'device',  # Default type
                'parameters': {},
                'nets': {}
            }

            # Try to get device type (various possible attribute names)
            if hasattr(device, 'deviceType'):
                info['type'] = device.deviceType
            elif hasattr(device, 'device_type'):
                info['type'] = device.device_type
            elif hasattr(device, 'type'):
                info['type'] = device.type
            elif hasattr(device, 'pdk_device'):
                info['type'] = device.pdk_device
            elif hasattr(device, 'model'):
                info['type'] = device.model
            else:
                # Use class name (e.g. 'nmos1p8', 'pmos1p8')
                info['type'] = type(device).__name__

            # Extract all parameters - try multiple formats

            # Canonicalize a COPY using the emitter's unit and rendering rules.
            # Defaults already use PDK units; strings are literal SPICE text.
            # Rendering also removes ordinary conversion roundoff exactly as
            # emission does, without a separate fingerprint tolerance.
            # Format 1: params dict (silk AnalogElement style)
            if hasattr(device, 'params') and isinstance(getattr(device, 'params', None), dict):
                scales = getattr(type(device), '_param_scales', {})
                supplied = getattr(device, '_caller_supplied', ())
                for param_name, param_value in device.params.items():
                    scale = scales.get(param_name, 1.0) if param_name in supplied else 1.0
                    info['parameters'][param_name] = _render_param(param_value, scale)

            # Format 2: param_list (legacy Module PDKDevice style). The
            # immediate owner is essential: ancestors may use another PDK.
            elif hasattr(device, 'param_list'):
                for param_name, constraint in device.param_list.items():
                    if (getattr(constraint, 'emit_if_supplied', False)
                            and constraint.value is None
                            and not getattr(constraint, 'emission_set', False)):
                        continue
                    raw = constraint.value
                    if raw is None:
                        value = constraint.default
                    else:
                        value = raw.value if hasattr(raw, 'value') else raw
                        scale = netlist.pdk_details._scale_factor(
                            constraint.pdk_unit, constraint.quantity)
                        # Match generate_netlist_entry's numeric coercion;
                        # never parse strings or scale omitted defaults.
                        if scale != 1.0 and not isinstance(value, str):
                            try:
                                value = float(value) * scale
                            except (ValueError, TypeError):
                                pass
                    info['parameters'][param_name] = _render_param(value)

            # Format 3: parameters dict
            elif hasattr(device, 'parameters'):
                for param_name, param_value in device.parameters.items():
                    if hasattr(param_value, 'value'):
                        info['parameters'][param_name] = param_value.value
                    else:
                        info['parameters'][param_name] = param_value

            # Extract net connections

            # Format 1: nets dict (silk AnalogElement style)
            if hasattr(device, 'nets') and isinstance(getattr(device, 'nets', None), dict):
                for terminal, net_name in device.nets.items():
                    info['nets'][terminal] = net_name

            # Format 2: pin_list (legacy Module style)
            elif hasattr(device, 'pin_list'):
                for pin_name, pin_obj in device.pin_list.items():
                    if hasattr(pin_obj, 'net'):
                        info['nets'][pin_name] = pin_obj.net

            # Add model if available
            if hasattr(device, 'model'):
                info['model'] = device.model
            elif hasattr(device, 'pdk_device'):
                info['model'] = device.pdk_device

            return info

        except Exception as exc:
            raise CircuitError(
                f"Cannot extract structure for device {name!r} "
                f"(parameter {param_name!r}): {exc}. "
                "Check the device parameter values and owning PDK unit metadata."
            ) from exc

    def _extract_module_info(self, name: str, module, netlist, *, local_name=None) -> Dict:
        """
        Extract module/subcircuit information

        Returns dict with:
        - instance_name
        - module_type
        - ports (port name -> immediate parent net name)
        - parameters
        """
        try:
            from silk.circuit.module_placement import refresh_definitions
            refresh_definitions(netlist)
            placement = getattr(netlist, 'elem_dict', {}).get(
                name if local_name is None else local_name, {})
            if not isinstance(placement, dict):
                placement = {}
            info = {
                'instance_name': name,
                'module_type': placement.get(
                    'structural_name',
                    placement.get('module_name', module.module_name)),
                'ports': {},
                'parameters': {}
            }

            # Module.nets enumerates internal nets. Only the stored instance's
            # pins describe its connections in the immediate parent.
            for port, pin in (placement.get('pin_list', module.pin_list) or {}).items():
                net = pin.net
                if not isinstance(net, str) or not net:
                    raise ValueError(f"port {port!r} has no parent-net binding")
                info['ports'][port] = net

            # Get parameters if any
            if hasattr(module, 'parameters'):
                for param_name, param_value in module.parameters.items():
                    if hasattr(param_value, 'value'):
                        info['parameters'][param_name] = param_value.value
                    else:
                        info['parameters'][param_name] = param_value

            return info

        except Exception as exc:
            raise CircuitError(
                f"Cannot extract structure for module instance {name!r}: {exc}. "
                "Check the instance's declared ports and parent-net bindings."
            ) from exc

    def _extract_net_info(self, name: str, net_data) -> Dict:
        """Extract net information"""
        # Classify net type
        net_type = 'signal'  # default

        name_lower = name.lower()
        if 'vdd' in name_lower or 'vcc' in name_lower or 'vpwr' in name_lower:
            net_type = 'power'
        elif 'gnd' in name_lower or 'vss' in name_lower or name == '0':
            net_type = 'ground'

        return {
            'name': name,
            'type': net_type
        }

    def _extract_module_structure(self, module, prefix: str = "") -> Dict:
        """
        Extract structure of a single module (for module hash computation)

        Args:
            module: Module instance
            prefix: Module path prefix

        Returns:
            dict: {
                'devices': [list of device info],
                'nets': [list of net names],
                'submodules': {name: {hash, instance_name, module_type, ports, parameters}},
                'passives': [list of passive info]
            }
        """
        module_struct = {
            'devices': [],
            'nets': [],
            'submodules': {},
            'passives': []
        }
        self._extract_sources(module, module_struct)

        # Extract devices from this module
        if hasattr(module, 'elem_dict'):
            for elem_name, elem_entry in module.elem_dict.items():
                if isinstance(elem_entry, dict):
                    elem_data = elem_entry.get('device')
                    elem_type = str(elem_entry.get('type', ''))

                    if elem_type in ('d', 'ElementType.DEVICE'):
                        # Device
                        passive_entry = _ideal_passive_canonical_entry(elem_name, elem_data)
                        if passive_entry is not None:
                            module_struct['passives'].append(passive_entry)
                        else:
                            device_info = self._extract_transistor_info(elem_name, elem_data, module)
                            if device_info:
                                module_struct['devices'].append(device_info)

                    elif elem_type in ('m', 'ElementType.MODULE'):
                        full_name = f"{prefix}.{elem_name}" if prefix else elem_name
                        info = self._extract_module_info(full_name, elem_data, module, local_name=elem_name)
                        # Definition payloads stay local and reusable; full
                        # paths belong only to the flattened hierarchy.
                        info['instance_name'] = elem_name
                        sub_struct = self._extract_module_structure(elem_data, full_name)
                        sub_hash = self._compute_module_hash(sub_struct)
                        module_struct['submodules'][elem_name] = {
                            **info, 'hash': sub_hash,
                        }

        # Extract passives (R/C/L) from this module
        for passive_name, passive in (getattr(module, 'passives_dict', None) or {}).items():
            module_struct['passives'].append(_passive_dict_entry(passive_name, passive))

        # Extract nets
        if hasattr(module, 'net_dict'):
            module_struct['nets'] = list(module.net_dict.keys())

        return module_struct

    def _compute_module_hash(self, module_structure: Dict) -> str:
        """
        Compute hash of a single module structure

        Hash is based on:
        - Device names, types, parameters (sorted)
        - Net names (sorted)
        - Submodule hashes and instance port bindings (sorted)
        - Passive names, kinds, values, connections (sorted)

        Args:
            module_structure: Dict from _extract_module_structure()

        Returns:
            str: SHA256 hash of module
        """
        return _fingerprint_module_structure(module_structure).digest

    def _extract_parameters(self, module_name: str, module_data) -> List[Dict]:
        """Extract module-level parameters"""
        params = []

        if hasattr(module_data, 'parameters'):
            for param_name, param_value in module_data.parameters.items():
                params.append({
                    'module': module_name,
                    'name': param_name,
                    'value': param_value.value if hasattr(param_value, 'value') else param_value
                })

        return params

    def compute_structure_hash(self, structure: Dict) -> Tuple[str, Dict[str, str]]:
        """
        Compute hash of structure for change detection

        Hash is based on:
        - Device names, types, parameters, and connections (sorted)
        - Net names and types (sorted)
        - Hierarchy (sorted)
        - Passive names, kinds, values, and connections (sorted)
        - Module hashes (sorted) - NEW!

        Order-independent: Same devices in different order = same hash

        Args:
            structure: Structure dict from extract_structure()

        Returns:
            tuple: (structure_hash: str, module_hashes: Dict[str, str])
        """
        fingerprint, module_fingerprints = _fingerprint_structure(structure)
        structure_hash = fingerprint.digest
        module_hashes = {
            path: module_fingerprint.digest
            for path, module_fingerprint in module_fingerprints.items()
        }

        if self.debug:
            print(f"\n🔐 Structure hash computation:")
            print(f"  Devices: {len(structure.get('transistors', []))}")
            print(f"  Nets: {len(structure.get('nets', []))}")
            print(f"  Modules: {len(module_hashes)}")
            print(f"  Hash: {structure_hash[:16]}...")

        return structure_hash, module_hashes

    def _compare_module_hashes(self, current_modules: Dict) -> Dict[str, bool]:
        """
        Compare current module hashes with stored hashes to detect module-level changes

        Args:
            current_modules: Dict of current modules with their hashes

        Returns:
            Dict[module_name: changed (bool)]
        """
        if not current_modules:
            return {}

        module_changes = {}

        for module_name, module_data in current_modules.items():
            current_fingerprint = Fingerprint(
                module_data.get(
                    'hash_scheme', STRUCTURE_FINGERPRINT_SCHEME
                ),
                module_data.get('hash'),
            )
            _require_fingerprint_scheme(
                current_fingerprint, STRUCTURE_FINGERPRINT_SCHEME
            )
            previous_fingerprint = self._last_module_fingerprints.get(
                module_name
            )

            # Module changed if hash is different or module is new
            module_changes[module_name] = (
                previous_fingerprint is None
                or not _fingerprints_equal(
                    current_fingerprint,
                    previous_fingerprint,
                    expected_scheme=STRUCTURE_FINGERPRINT_SCHEME,
                )
            )

            if self.debug and module_changes[module_name]:
                if previous_fingerprint is None:
                    print(f"  🆕 Module '{module_name}' is NEW")
                else:
                    print(f"  ⚠️  Module '{module_name}' has CHANGED")
                    print(
                        f"      Old hash: "
                        f"{previous_fingerprint.digest[:16]}..."
                    )
                    print(
                        f"      New hash: "
                        f"{current_fingerprint.digest[:16]}..."
                    )

        # Check for deleted modules
        for old_module_name in self._last_module_fingerprints:
            if old_module_name not in current_modules:
                module_changes[old_module_name] = True
                if self.debug:
                    print(f"  🗑️  Module '{old_module_name}' was DELETED")

        return module_changes

    def has_structure_changed(self, current_hash: str) -> bool:
        """
        Check if structure has changed from last stored version

        Args:
            current_hash: Hash of current structure

        Returns:
            bool: True if structure changed or this is first run
        """
        current_fingerprint = Fingerprint(
            STRUCTURE_FINGERPRINT_SCHEME, current_hash
        )
        _require_fingerprint_scheme(
            current_fingerprint, STRUCTURE_FINGERPRINT_SCHEME
        )
        if self._last_structure_fingerprint is None:
            if self.debug:
                print("  → First run, structure is new")
            return True

        changed = not _fingerprints_equal(
            current_fingerprint,
            self._last_structure_fingerprint,
            expected_scheme=STRUCTURE_FINGERPRINT_SCHEME,
        )

        if self.debug:
            if changed:
                print(f"  → Structure changed!")
                print(f"    Last: {self._last_structure_hash[:16]}...")
                print(f"    Current: {current_hash[:16]}...")
            else:
                print(f"  → Structure unchanged (hash: {current_hash[:16]}...)")

        return changed

    def store_structure(self, structure: Dict, structure_hash: str, module_hashes: Dict[str, str] = None):
        """
        Store structure in database

        Args:
            structure: Structure dict from extract_structure()
            structure_hash: Hash of structure
            module_hashes: Dict of module hashes (optional, will be extracted if not provided)
        """
        self.db._require_building(self.circuit_id)
        for module_path, module_data in structure.get('modules', {}).items():
            if not isinstance(module_data, dict):
                raise EvidenceMismatchError(
                    f"Module payload at {module_path!r} is not a mapping"
                )
            module_payload = {
                'devices': module_data.get('devices', []),
                'nets': module_data.get('nets', []),
                'submodules': module_data.get('submodules', {}),
                'passives': module_data.get('passives', []),
            }
            if module_data.get('sources'):
                module_payload['sources'] = module_data['sources']
            expected_module = _fingerprint_module_structure(module_payload)
            if module_data.get('hash') != expected_module.digest:
                raise EvidenceMismatchError(
                    f"Stored module hash for {module_path!r} does not match "
                    "its structure payload"
                )
        computed, computed_modules = _fingerprint_structure(structure)
        if computed.digest != structure_hash:
            raise EvidenceMismatchError(
                "The supplied structure hash does not match the structure payload"
            )
        row = self.db._snapshot_row(self.circuit_id)
        if (
            row["structure_scheme"] != STRUCTURE_FINGERPRINT_SCHEME
            or row["structure_hash"] != structure_hash
        ):
            raise EvidenceMismatchError(
                "The structure payload does not match the building snapshot identity"
            )
        expected_module_hashes = {
            path: fingerprint.digest
            for path, fingerprint in computed_modules.items()
        }
        if module_hashes is None:
            module_hashes = expected_module_hashes
        elif module_hashes != expected_module_hashes:
            raise EvidenceMismatchError(
                "The supplied module hashes do not match the structure payload"
            )

        attached = sum(
            self.db.conn.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE circuit_version_id = ?',
                (self.circuit_id,),
            ).fetchone()[0]
            for table in ("devices", "nets", "circuit_modules")
        )
        if attached:
            raise DatabaseError(
                f"Structure for version {self.circuit_id!r} is insert-only"
            )
        if self.debug:
            print("\n💾 Storing structure in database...")
            print(f"  circuit_id: {self.circuit_id}")
            print(f"  devices in structure: {len(structure.get('transistors', []))}")

        device_count = 0
        for source in structure.get('sources', []):
            self.db.conn.execute(
                """INSERT INTO devices
                    (device_id, circuit_version_id, name, type, model, properties)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), self.circuit_id, source['name'],
                 'silk_source', source['type'],
                 json.dumps(encode_json(dict(source, kind='source')), allow_nan=False)),
            )
            device_count += 1
        for device in structure.get('transistors', []):
            device_type = device.get('type', 'device')
            model = device.get('model', device_type)
            self.db.conn.execute(
                """
                INSERT INTO devices
                    (device_id, circuit_version_id, name, type, model, properties)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    self.circuit_id,
                    device['name'],
                    device_type,
                    model,
                    json.dumps(encode_json(device.get('parameters', {})), allow_nan=False, default=str),
                ),
            )
            device_count += 1

        for passive in structure.get('passives', []):
            self.db.conn.execute(
                """
                INSERT INTO devices
                    (device_id, circuit_version_id, name, type, model, properties)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    self.circuit_id,
                    passive['name'],
                    passive.get('kind', 'passive'),
                    passive.get('kind', 'passive'),
                    json.dumps(
                        encode_json({
                            key: value
                            for key, value in passive.items()
                            if key not in {'name', 'kind'}
                        }), allow_nan=False,
                        default=str,
                    ),
                ),
            )
            device_count += 1

        net_count = 0
        for net in structure.get('nets', []):
            # Any key beyond name/type (e.g. parse_spice_netlist's
            # 'inferred': True on a name-classified net, TASK-97) is
            # provenance, not identity — thread it through add_net's
            # **properties the same way the device/passive loops above
            # thread their extra keys, instead of dropping it (TASK-112).
            extra = {k: v for k, v in net.items() if k not in ('name', 'type')}
            self.db.add_net(
                circuit_version_id=self.circuit_id,
                name=net['name'],
                net_type=net['type'],
                **extra,
            )
            net_count += 1

        # Store modules using new schema (independent modules table + junction table)
        module_count = 0
        if self.debug:
            print(f"\n  📦 Storing modules...")
            print(f"     Modules to store: {len(structure.get('modules', {}))}")

        for position, (module_name, module_data) in enumerate(structure.get('modules', {}).items()):
            if isinstance(module_data, dict):
                module_hash = module_data.get('hash', '')
                device_count_in_module = module_data.get('device_count', 0)

                if self.debug:
                    print(f"     Processing module: {module_name} (hash: {module_hash[:16]}...)")

                # Prepare module structure as JSON
                module_structure = {
                    'devices': module_data.get('devices', []),
                    'nets': module_data.get('nets', []),
                    'submodules': module_data.get('submodules', {}),
                    'passives': module_data.get('passives', []),
                }
                if module_data.get('sources'):
                    module_structure['sources'] = module_data['sources']

                module_id = self.db.insert_or_get_module(
                    module_name=module_name,
                    module_path=module_name,
                    module_hash=module_hash,
                    device_count=device_count_in_module,
                    structure=module_structure,
                    module_hash_scheme=STRUCTURE_FINGERPRINT_SCHEME,
                )
                self.db.link_module_to_circuit(
                    circuit_version_id=self.circuit_id,
                    module_id=module_id,
                    instance_name=module_name,
                    position=position,
                )
                module_count += 1

        self._last_structure_fingerprint = Fingerprint(
            STRUCTURE_FINGERPRINT_SCHEME, structure_hash
        )
        self._last_structure_hash = structure_hash
        self._last_module_fingerprints = {
            name: Fingerprint(STRUCTURE_FINGERPRINT_SCHEME, digest)
            for name, digest in module_hashes.items()
        }
        self._last_module_hashes = module_hashes
        self._current_run_count = 1
        self._update_run_count()

        if self.debug:
            print(f"  ✓ Stored {device_count} devices")
            print(f"  ✓ Stored {net_count} new nets")
            if module_count > 0:
                print(f"  ✓ Stored {module_count} modules in database")
            if module_hashes:
                print(f"  ✓ Tracked {len(module_hashes)} module hashes in metadata")

        self.db._maybe_commit()

    def increment_run_count(self):
        """Increment run count for unchanged structure"""
        self.db._require_unlocked(self.circuit_id)
        self._current_run_count += 1
        self._update_run_count()

        if self.debug:
            print(f"  ✓ Incremented run count to {self._current_run_count}")

    def _update_run_count(self):
        """Update run count in metadata"""
        self.db._require_unlocked(self.circuit_id)
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT metadata FROM circuit_versions
            WHERE version_id = ?
        """, (self.circuit_id,))

        row = cursor.fetchone()
        metadata = {}

        if row and row[0]:
            try:
                metadata = decode_json(json.loads(row[0]))
            except (json.JSONDecodeError, TypeError):
                pass

        metadata['run_count'] = self._current_run_count
        metadata['last_run'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            UPDATE circuit_versions
            SET metadata = ?
            WHERE version_id = ?
        """, (json.dumps(encode_json(metadata), allow_nan=False), self.circuit_id))

        self.db._maybe_commit()

    def get_run_count(self) -> int:
        """Get current run count"""
        return self._current_run_count

    def get_structure_hash(self) -> Optional[str]:
        """Get current structure hash"""
        return self._last_structure_hash

    def get_module_changes(self) -> Dict[str, bool]:
        """Get dictionary of module changes from last run"""
        return self._last_module_hashes


# Example usage
if __name__ == "__main__":
    print("""
Circuit Structure Tracker - Usage Example

# In Module, after building circuit but before simulation:

from silk.db.circuit_db import Database
from silk.db.structure_tracker import CircuitStructureTracker

# Assuming you have a Module instance with tracking enabled
netlist = Module(..., circuit_name="ldo", db_path="designs.db")
netlist.create_circuit_version("LDO v1.0")

# Build your circuit
netlist.add_element(...)
netlist.add_source(...)

# Track structure automatically
tracker = CircuitStructureTracker(netlist.db, netlist.circuit_id, debug=True)
changed, hash_val, module_changes = tracker.track_and_store_if_changed(netlist)

if changed:
    print("New structure detected and stored!")
    if module_changes:
        changed_modules = [m for m, c in module_changes.items() if c]
        print(f"  {len(changed_modules)} module(s) changed: {changed_modules}")
else:
    print(f"Structure unchanged, run #{tracker.get_run_count()}")

# Continue with simulation...
netlist.run_simulator()
    """)
