"""Generate runnable ``Module`` Python code from a circuit description dict.

:class:`Generator` turns a circuit description dict
or a :class:`CircuitSpec` from any tool into Python source. The dict
format is the one specified in ``docs/codegen-contract.md``.

Circuit description format
--------------------------
The caller passes a dict (or JSON file) with this structure::

    {
      # Required
      "circuit_name": "diff_pair",          # used for file name and comments
      "pdk": "sky130",                      # any registered PDK name (sky130, gf180mcu, sg13g2)

      # Optional — defaults shown
      "output_dir": "./sim",                # where to write the netlist file
      "corner": "tt",                       # tt | ss | ff | sf | fs

      # Optional — voltage domains, referenced by name from ports.
      # `nominal` is required. `tolerance` / `min` / `max` are optional and
      # drive the stress checks (check_voltage_domains, junction_overrange).
      "voltage_domains": {
        "AVDD": {"nominal": 1.8, "tolerance": 0.1},
        "DVDD": {"nominal": 0.9, "tolerance": 0.1},
      },

      # Optional — ports are dicts. `dir` is required; everything else
      # optional. When non-empty, the generator emits a subcircuit
      # (Module with define_module_ports) instead of a top-level netlist.
      "ports": {
        "clk":   {"dir": "input"},                                  # minimal
        "vh":    {"dir": "output", "role": "analog", "domain": "AVDD"},
        "code":  {"dir": "input",  "role": "digital", "bits": 4},
        "lanes": {"dir": "input",  "role": "analog",  "bits": [8, 12]},
        "dbg":   {"dir": "output", "role": "digital", "bits": 4,
                  "naming": "underscore"},
      },

      # Optional — internal bus net declarations (small, optional list).
      # Required when an internal bus net is needed; not needed for ports
      # (bus ports auto-declare). bits int -> 0..N-1; bits [a,b] -> range.
      "bus_nets": [
        {"name": "count", "bits": 4},
        {"name": "addr",  "bits": [4, 12], "naming": "underscore"},
      ],

      # Optional — module variables shared by parameters.
      # Declaration order is emission order. Any numeric device, element or
      # source parameter may hold {"variable": name} or
      # {"variable": name, "scale": number} instead of a number.
      "variables": [
        {"name": "w_in", "value": 2.0},
      ],

      # Devices from the PDK (pdk.nmos1p8, pdk.pmos1p8, etc.)
      "devices": [
        {
          "type":     "nmos1p8",
          "instance": "mn1",
          "nets":     {"d": "vout", "g": "vin", "s": "gnd", "b": "gnd"},
          "params":   {"w": 2.0, "l": 0.15, "m": 1}
        }
      ],

      # Ideal passive elements (elements.resistor, etc.).
      "ideal_elements": [
        {
          "type":     "resistor",
          "instance": "rload",
          "nets":     {"p": "vout", "n": "gnd"},
          "params":   {"value": 1000}
        }
      ],

      # Sources with transient waveforms (only used in standalone mode).
      "sources": [
        {"type": "VoltageSource", "instance": "vin",
         "positive": "vin", "negative": "gnd", "dc": 0,
         "waveform": {"type": "Pulse", "v1": 0, "v2": 1.8,
                      "tr": 1e-9, "tf": 1e-9,
                      "pw": 10e-9, "per": 20e-9}}
      ],

      # Dependent (controlled) sources — VCVS/VCCS/CCVS/CCCS.
      # 'type' accepts either the full class name or the short alias
      # (VCVS/VCCS/CCVS/CCCS). No dc/ac/waveform — these are structural, a
      # linear function of a control node-pair voltage (VCVS/VCCS) or of a
      # controlling source's branch current (CCVS/CCCS). ctrl_vname names
      # the controlling source's INSTANCE (present elsewhere in this same
      # sources[] list) — see silk/sources/schema.py's `references` field.
      # "sources": [
      #   {"type": "VCVS", "instance": "e1", "out_pos": "vout", "out_neg": "gnd",
      #    "ctrl_pos": "vin", "ctrl_neg": "gnd", "gain": 10},
      #   {"type": "VCCS", "instance": "g1", "out_pos": "iout", "out_neg": "gnd",
      #    "ctrl_pos": "vin", "ctrl_neg": "gnd", "gm": 0.01},
      #   {"type": "VoltageSource", "instance": "vmeas",
      #    "positive": "vmeas_p", "negative": "vmeas_n", "dc": 0},
      #   {"type": "CCVS", "instance": "h1", "out_pos": "vout", "out_neg": "gnd",
      #    "transresistance": 1000, "ctrl_vname": "vmeas"},
      #   {"type": "CCCS", "instance": "f1", "out_pos": "iout", "out_neg": "gnd",
      #    "gain": 2, "ctrl_vname": "vmeas"},
      # ],

      # Behavioral sources — BehavioralVoltage/BehavioralCurrent.
      # 'type' is the full class name (no short alias). Fields are
      # {type, instance, positive, negative, expression} (+ optional array/
      # naming) — no dc/ac/waveform/pwl. 'expression' is a SPICE B-element expression string
      # (node voltages v(n), branch currents i(vname), math functions).
      # "sources": [
      #   {"type": "BehavioralVoltage", "instance": "bdiff",
      #    "positive": "vd", "negative": "gnd", "expression": "v(inp)-v(inn)"},
      #   {"type": "BehavioralCurrent", "instance": "bgm",
      #    "positive": "io", "negative": "gnd", "expression": "0.01*v(vin,gnd)"},
      # ],

      # Simulation (only used in standalone mode).
      "simulation": {
        "type":  "Transient", "step": 1e-9, "stop": 100e-9
      },

      # Hierarchical child instances. Consumed by both output modes.
      # Each entry instantiates a child Module via add_element. See
      # docs/codegen-contract.md#submodules for the full contract.
      "submodules": [
        {
          "instance_name": "inv1",
          "module_ref": {"import_path": ".inverter", "module_var": "mod"},
          "nets":     {"vin": "vin", "vout": "vmid"},
          "parameters": {"drive_strength": 4}     # optional
        }
      ]
    }

Output modes
------------
The generator emits two modes, selected by the ``ports`` key:

* **Standalone simulation script** (no ``ports`` key) — emits a top-level
  ``Module`` plus simulation/results stubs. Every element — device, ideal
  element, or source — is emitted via ``ckt.add(...)`` (``ckt.add_source(...)`` remains a valid public alias), plus
  ``ckt.simulate(...)``.

* **Subcircuit / reusable module** (``ports`` present, incl. {}) — emits a
  ``Module`` with ``define_module_ports({...})`` and any
  ``voltage_domains`` / ``bus_nets`` declarations, but no ``simulate``
  call. Port literals use the shortest legal form: bare ``PortDir``
  when only ``dir`` is set, ``(PortDir, PortRole)`` tuple when only
  ``dir`` and ``role`` are set, or full ``Port(...)`` when any of
  domain / bits / non-default naming is set.
"""

from __future__ import annotations

import functools
import json
from silk.utils import require_python_identifier

import keyword
import math
import re
import textwrap
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from silk import sources
from silk.circuit.circuit_elements import Port, PortDir, PortRole
from silk.circuit.voltage_domain import VoltageDomain
from silk.simulation import analysis
from silk.utils._utils import _parse_si


# ---------------------------------------------------------------------------
# Data classes — typed representation of the circuit description
# ---------------------------------------------------------------------------

@dataclass
class DeviceSpec:
    """One placed device instance (a PDK device or an ideal element).

    Attributes
    ----------
    type : str
        Device-factory name on the PDK (``devices``) or on the ideal element
        library (``ideal_elements``), for example ``'nmos1p8'``.
    instance : str
        Instance name, for example ``'mn1'``; it must be a valid Python
        identifier.
    nets : dict of str to str or list of dict
        Maps each pin name to the net it connects to. On an arrayed entry a
        net may contain ``$i`` / ``$(i+N)`` index templates. Alternatively,
        a list supplies one plain-net dict per copy in copy order.
    params : dict
        Maps parameter name to its value: a number, or a
        ``{"variable": name}`` / ``{"variable": name, "scale": number}``
        reference to a declared variable. Units are the circuit's (PDK-native
        micrometres for W and L on the shipped PDKs).
    array : int or tuple of int, optional
        ``None`` for a single instance. An int ``N`` means indices
        ``range(N)``; a pair ``(a, b)`` means ``range(a, b)``.
    naming : str
        Array instance naming, ``'bracketed'`` (``U[0]``) or ``'underscore'``
        (``U_0``). Ignored when ``array`` is ``None``.

    Examples
    --------
    >>> from silk.codegen import DeviceSpec
    >>> dev = DeviceSpec('nmos1p8', 'mn1',
    ...                  {'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
    ...                  {'w': 2.0, 'l': 0.15})
    >>> dev.naming
    'bracketed'
    """
    type: str                          # e.g. 'nmos1p8'
    instance: str                      # e.g. 'mn1'
    nets: dict[str, str] | list[dict[str, str]]  # per-pin or per-copy wiring
    params: dict[str, Any]             # {param_name: value}
    array: 'int | tuple[int, int] | None' = None
    naming: str = 'bracketed'


@dataclass
class SourceSpec:
    """One independent source instance (VoltageSource or CurrentSource).

    ``waveform`` is an optional dict carrying the waveform's class name under
    the reserved ``type`` key.  Its constructor keyword arguments go in a
    nested ``params`` dict — the same ``{"type": ..., "params": {...}}``
    shape ``devices[]`` and ``ideal_elements[]`` already use (see the module
    docstring above).  Nesting under ``params`` is what lets a waveform
    parameter be named ``type`` without colliding with the reserved class
    key — :class:`~silk.sources.waveforms.RandomSource` (the ``trrandom``
    kind) has a required constructor parameter literally named ``type``
    (1=Uniform, 2=Gaussian, 3=Exponential, 4=Poisson), and it lives in
    ``params`` right alongside it.

    A legacy flat form (parameters as sibling keys of ``type``, no nested
    ``params``) is still accepted unchanged for every kind whose own
    parameters do not include one named ``type`` — i.e. every kind except
    ``trrandom``. A dict is rejected if it mixes both forms (``params`` plus
    a flat keyword sitting beside ``type``).

    ``pwl`` is a flat list of time-value pairs and is handled separately
    from ``waveform``.

    Example (DC bias)::

        {"type": "VoltageSource", "instance": "vbias",
         "positive": "vb", "negative": "gnd", "dc": 1.2}

    Example (pulse excitation, params form)::

        {"type": "VoltageSource", "instance": "vin",
         "positive": "vin", "negative": "gnd", "dc": 0,
         "waveform": {"type": "Pulse",
                      "params": {"v1": 0, "v2": 1.8, "td": 0,
                                 "tr": 1e-9, "tf": 1e-9,
                                 "pw": 10e-9, "per": 20e-9}}}

    Example (pulse excitation, legacy flat form — still accepted)::

        {"type": "VoltageSource", "instance": "vin",
         "positive": "vin", "negative": "gnd", "dc": 0,
         "waveform": {"type": "Pulse",
                      "v1": 0, "v2": 1.8, "td": 0,
                      "tr": 1e-9, "tf": 1e-9, "pw": 10e-9, "per": 20e-9}}

    Example (trrandom — only expressible via ``params``, since the waveform's
    own ``type`` parameter would otherwise collide with the reserved class
    key)::

        {"type": "VoltageSource", "instance": "vnoise",
         "positive": "vn", "negative": "gnd",
         "waveform": {"type": "RandomSource",
                      "params": {"type": 1, "ts": 1e-9}}}

    Example (PWL)::

        {"type": "VoltageSource", "instance": "vpwl",
         "positive": "vin", "negative": "gnd",
         "pwl": [0, 0, 1e-9, 1.8, 2e-9, 0]}

    Dependent (controlled) sources — VCVS/VCCS/CCVS/CCCS — use
    this same dataclass but a disjoint set of fields: ``out_pos``/
    ``out_neg`` instead of ``positive``/``negative``, ``ctrl_pos``/
    ``ctrl_neg`` for VCVS/VCCS, ``gain``/``gm``/``transresistance`` instead
    of ``dc``/``ac``, and (CCVS/CCCS only) ``ctrl_vname`` — the INSTANCE
    NAME of the controlling source, validated at
    :meth:`CircuitSpec.from_dict <silk.codegen.CircuitSpec.from_dict>` time against the other entries in the same
    ``sources[]`` list. Arrays use the independent-source naming and pin
    templates; arrayed current controls bind by matching index. No waveform/pwl. See
    ``_controlled_source_call`` and ``docs/codegen-contract.md``.

    Example (VCVS)::

        {"type": "VCVS", "instance": "e1", "out_pos": "vout", "out_neg": "gnd",
         "ctrl_pos": "vin", "ctrl_neg": "gnd", "gain": 10}

    Example (CCCS, referencing a VoltageSource ammeter elsewhere in sources[])::

        {"type": "CCCS", "instance": "f1", "out_pos": "icopy", "out_neg": "gnd",
         "gain": 2, "ctrl_vname": "vmeas"}

    Behavioral sources — BehavioralVoltage/BehavioralCurrent —
    also use this dataclass but a third disjoint field set: ``positive``/
    ``negative`` (like VoltageSource/CurrentSource) plus ``expression`` (a
    SPICE B-element expression string), and none of dc/ac/waveform/pwl/
    ctrl_vname. ``array``/``naming`` follow the independent-source rules; the ``$i``/``$(i±N)`` templates are rewritten in
    ``positive``, ``negative`` and ``expression`` and nothing else — silk
    does not parse expressions. ``i()`` in an expression takes the EMITTED,
    V-prefixed source name (``emitted_instance_name``), which silk does not
    rewrite; each element sees only its own index arithmetic, and an
    out-of-range reference fails loudly at simulation. No declarative alias surface (not in
    source_metadata.yaml's ``sources:`` map — see
    :func:`export_source_schema <silk.sources.export_source_schema>`'s ``unschematized_sources``), so like the
    controlled sources they are emitted via a direct ``sources.<ClassName>``
    call. See ``_behavioral_source_call``.

    Example::

        {"type": "BehavioralVoltage", "instance": "bdiff",
         "positive": "vd", "negative": "gnd", "expression": "v(inp)-v(inn)"}

    Attributes
    ----------
    type : str
        ``'VoltageSource'``, ``'CurrentSource'``, a controlled-source class
        name or short alias (``'VCVS'``, ``'VCCS'``, ``'CCVS'``, ``'CCCS'``),
        or ``'BehavioralVoltage'`` / ``'BehavioralCurrent'``.
    instance : str
        Source instance name.
    positive : str or None
        Positive terminal net of an independent or behavioral source;
        ``None`` for a controlled source.
    negative : str or None
        Negative terminal net of an independent or behavioral source;
        ``None`` for a controlled source.
    dc : float
        DC value (volts or amperes, by source kind). Defaults to 0.
    ac : float
        AC magnitude (volts or amperes). Defaults to 0.
    waveform : dict or None
        Transient waveform: the class name under ``'type'`` and its
        keyword arguments under ``'params'`` (or, in the legacy flat form,
        beside ``'type'``).
    pwl : list or None
        Flat piecewise-linear list ``[t0, v0, t1, v1, ...]``, with times in
        seconds.
    array : int or tuple of int, optional
        ``None`` for a single source; an int ``N`` means ``range(N)`` and a
        pair ``(a, b)`` means ``range(a, b)``.
    naming : str
        Array instance naming, ``'bracketed'`` or ``'underscore'``.
    out_pos : str or None
        Positive output net of a controlled source.
    out_neg : str or None
        Negative output net of a controlled source.
    ctrl_pos : str or None
        Positive control net of a VCVS or VCCS.
    ctrl_neg : str or None
        Negative control net of a VCVS or VCCS.
    gain : float or None
        Gain of a VCVS or CCCS (dimensionless).
    gm : float or None
        Transconductance of a VCCS, in siemens.
    transresistance : float or None
        Transresistance of a CCVS, in ohms.
    ctrl_vname : str or None
        Instance name of the controlling source of a CCVS or CCCS; it must
        be another entry in the same ``sources`` list.
    expression : str or None
        SPICE B-element expression of a behavioral source; silk does not
        parse it.
    """
    type: str                          # 'VoltageSource' | 'CurrentSource' | controlled-source type/alias
    instance: str
    # Required (no default) for VoltageSource/CurrentSource — testbench_generator's
    # _source_add_line relies on this to raise when a testbench source dict omits
    # 'positive'/'negative' (TestbenchSpecError). Controlled-source construction
    # passes positive=None, negative=None explicitly (see
    # _controlled_source_spec_from_dict) rather than relying on a default.
    positive: 'str | None'
    negative: 'str | None'
    dc: float = 0
    ac: float = 0
    waveform: 'dict | None' = None   # {type: 'Pulse', v1: 0, ...}
    pwl: 'list | None' = None        # flat [t0,v0, t1,v1, ...]
    array: 'int | tuple[int, int] | None' = None
    naming: str = 'bracketed'
    # Dependent (controlled) sources only (TASK-253) — see class docstring.
    out_pos: 'str | None' = None
    out_neg: 'str | None' = None
    ctrl_pos: 'str | None' = None
    ctrl_neg: 'str | None' = None
    gain: 'float | None' = None
    gm: 'float | None' = None
    transresistance: 'float | None' = None
    ctrl_vname: 'str | None' = None
    # Behavioral sources only (TASK-282.2) — see class docstring.
    expression: 'str | None' = None


@dataclass
class BusNetSpec:
    """Declaration of one internal bus net.

    Maps 1:1 to a ``Module.add_net(name, bits=..., naming=...)`` call in
    the generated code. Width comes from ``bits`` (int width or
    ``range``); per-bit names are derived from ``naming``.

    Attributes
    ----------
    name : str
        Bus net name.
    bits : int or range
        Width as an int (bits ``0`` to ``N - 1``), or an explicit ``range``
        of bit indices.
    naming : str
        Per-bit naming, ``'bracketed'`` or ``'underscore'``.

    Examples
    --------
    >>> from silk.codegen import BusNetSpec
    >>> bus = BusNetSpec.from_dict({'name': 'addr', 'bits': [4, 12]})
    >>> bus.bits
    range(4, 12)
    """
    name: str
    bits: 'int | range'
    naming: str = 'bracketed'

    @classmethod
    def from_dict(cls, d: dict) -> 'BusNetSpec':
        """Build a bus-net declaration from its ``bus_nets[]`` JSON entry.

        Parameters
        ----------
        d : dict
            Entry with required ``'name'`` and ``'bits'`` (an int width, or a
            two-element ``[start, stop]`` list meaning ``range(start,
            stop)``) and optional ``'naming'``.

        Returns
        -------
        BusNetSpec
            The parsed declaration.

        Raises
        ------
        KeyError
            ``'name'`` or ``'bits'`` is missing.
        ValueError
            ``'bits'`` is neither an int nor a two-element list.
        """
        if 'name' not in d:
            raise KeyError(f"bus_nets entry missing required 'name': {d!r}")
        if 'bits' not in d:
            raise KeyError(
                f"bus_nets entry {d['name']!r} missing required 'bits'"
            )
        raw_bits = d['bits']
        if isinstance(raw_bits, int):
            bits = raw_bits
        elif isinstance(raw_bits, list) and len(raw_bits) == 2:
            bits = range(raw_bits[0], raw_bits[1])
        else:
            if isinstance(raw_bits, list):
                raise ValueError(
                    f"bus_nets entry {d['name']!r}: bits must be int "
                    f"or a 2-element [start, stop] list, got "
                    f"{raw_bits!r} ({len(raw_bits)} elements)"
                )
            raise ValueError(
                f"bus_nets entry {d['name']!r}: bits must be int "
                f"or [start, stop] list, got {raw_bits!r}"
            )
        kwargs = {'name': d['name'], 'bits': bits}
        if d.get('naming') is not None:
            kwargs['naming'] = d['naming']
        return cls(**kwargs)


@dataclass
class SubmoduleSpec:
    """One submodule instance declaration.

    Maps 1:1 to a ``Module.add_element(child_mod, instance, nets=...,
    parameters=...)`` call in the generated code. ``import_path`` /
    ``module_var`` describe how to import the child Module in the
    emitted file; the import-alias name is computed silk-side at
    emission time (not stored here).

    Attributes
    ----------
    instance_name : str
        Instance name; a valid Python identifier.
    import_path : str
        Python import path of the module that defines the child, for example
        ``'.inverter'``.
    module_var : str
        Name of the child ``Module`` variable inside that module.
    nets : dict of str to str or list of dict
        Maps the child's port names to nets of the parent. An array may
        supply one plain-net dict per copy in copy order.
    parameters : dict or None
        Optional parameter overrides; values are ``str``, ``int``, ``float``,
        ``bool`` or ``None``.
    array : int or tuple of int, optional
        ``None`` for a single instance; an int ``N`` means ``range(N)`` and a
        pair ``(a, b)`` means ``range(a, b)``.
    naming : str
        Array instance naming, ``'bracketed'`` or ``'underscore'``.

    Examples
    --------
    >>> from silk.codegen import SubmoduleSpec
    >>> sub = SubmoduleSpec.from_dict({
    ...     'instance_name': 'inv1',
    ...     'module_ref': {'import_path': '.inverter', 'module_var': 'mod'},
    ...     'nets': {'vin': 'vin', 'vout': 'vmid'}})
    >>> sub.import_path, sub.module_var
    ('.inverter', 'mod')
    """
    instance_name: str
    import_path: str
    module_var: str
    nets: dict[str, str] | list[dict[str, str]]
    parameters: 'dict | None' = None
    array: 'int | tuple[int, int] | None' = None
    naming: str = 'bracketed'

    @classmethod
    def from_dict(cls, d: dict) -> 'SubmoduleSpec':
        """Build a submodule declaration from its ``submodules[]`` JSON entry.

        Parameters
        ----------
        d : dict
            Entry with required ``'instance_name'``, ``'module_ref'`` (a dict
            with ``'import_path'`` and ``'module_var'``) and ``'nets'``, and
            optional ``'parameters'``, ``'array'`` and ``'naming'``.

        Returns
        -------
        SubmoduleSpec
            The parsed declaration.

        Raises
        ------
        KeyError
            A required key is missing.
        TypeError
            ``'nets'`` is not a dict of strings or a list of such dicts,
            or ``'parameters'`` is not a
            dict of simple values.
        ValueError
            ``'array'`` or ``'naming'`` is invalid, or per-copy nets have the
            wrong count, mismatched pins, or invalid plain net names.
        """
        if 'instance_name' not in d:
            raise KeyError(
                f"submodules entry missing required 'instance_name': {d!r}"
            )
        if 'module_ref' not in d:
            raise KeyError(
                f"submodules entry {d['instance_name']!r} missing required "
                f"'module_ref'"
            )
        ref = d['module_ref']
        if 'import_path' not in ref:
            raise KeyError(
                f"submodules entry {d['instance_name']!r}: module_ref "
                f"missing required 'import_path'"
            )
        if 'module_var' not in ref:
            raise KeyError(
                f"submodules entry {d['instance_name']!r}: module_ref "
                f"missing required 'module_var'"
            )
        if 'nets' not in d:
            raise KeyError(
                f"submodules entry {d['instance_name']!r} missing required "
                f"'nets'"
            )
        nets_raw = d['nets']
        array, naming = _parse_array_and_naming(
            d, f"submodules entry {d['instance_name']!r}"
        )
        _validate_per_copy_nets(nets_raw, array, f"submodules entry {d['instance_name']!r}",
                                allow_bit_pins=True)
        if not isinstance(nets_raw, (dict, list)):
            raise TypeError(
                f"submodules entry {d['instance_name']!r}: 'nets' must "
                f"be a dict, got {type(nets_raw).__name__}"
            )
        for k, v in (nets_raw.items() if isinstance(nets_raw, dict) else ()):
            if not isinstance(v, str):
                raise TypeError(
                    f"submodules entry {d['instance_name']!r}: "
                    f"nets[{k!r}] must be a string net name, got "
                    f"{type(v).__name__} ({v!r})"
                )
        params_raw = d.get('parameters')
        if params_raw:
            if not isinstance(params_raw, dict):
                raise TypeError(
                    f"submodules entry {d['instance_name']!r}: "
                    f"'parameters' must be a dict, got "
                    f"{type(params_raw).__name__}"
                )
            allowed = (str, int, float, bool, type(None))
            for k, v in params_raw.items():
                if not isinstance(v, allowed):
                    raise TypeError(
                        f"submodules entry {d['instance_name']!r}: "
                        f"parameters[{k!r}] must be str/int/float/bool/"
                        f"None, got {type(v).__name__} ({v!r})"
                    )
            parameters = dict(params_raw)
        else:
            parameters = None
        return cls(
            instance_name=d['instance_name'],
            import_path=ref['import_path'],
            module_var=ref['module_var'],
            nets=([dict(n) for n in nets_raw] if isinstance(nets_raw, list) else dict(nets_raw)),
            parameters=parameters,
            array=array,
            naming=naming,
        )


def _validate_per_copy_nets(nets, array, label, *, allow_bit_pins=False):
    """Validate explicit per-copy wiring without changing broadcast dicts."""
    if not isinstance(nets, list):
        return
    if array is None:
        raise TypeError(f"{label}: 'nets' must be a dict when 'array' is absent")
    lo, hi = (0, array) if isinstance(array, int) else array
    if len(nets) != hi - lo:
        raise ValueError(f"{label}: nets list must contain {hi - lo} dicts, got {len(nets)}")
    from silk.elements.base import validate_net_token
    pins = None
    for connections in nets:
        if not isinstance(connections, dict):
            raise TypeError(f"{label}: each nets list item must be a dict")
        if pins is not None and set(connections) != pins:
            raise ValueError(f"{label}: nets list dicts must have the same pin set")
        pins = set(connections)
        for pin, value in connections.items():
            if allow_bit_pins:
                if not isinstance(pin, str):
                    raise ValueError(
                        f"{label}: pin {pin!r} must be a port name or one bit "
                        "of it (name[k], k a non-negative decimal integer)"
                    )
                validate_net_token(pin, who=f"{label} pin", kind='port')
            else:
                _require_port_name(pin, f"{label} pin")
            if not isinstance(value, str):
                raise TypeError(f"{label}: nets[{pin!r}] must be a string net name")
            validate_net_token(value, who=f"{label} nets[{pin!r}]")


def _per_copy_nets_bindings(spec):
    """Names bound by explicit per-copy wiring lists."""
    return {f"__silk_{entry.instance_name if isinstance(entry, SubmoduleSpec) else entry.instance}_nets"
            for entry in [*spec.devices, *spec.ideal_elements, *spec.submodules]
            if isinstance(entry.nets, list)}


def _emit_per_copy_nets(entry, lines, formatter):
    """Emit the wiring list and return its copy-indexed expression."""
    instance = entry.instance_name if isinstance(entry, SubmoduleSpec) else entry.instance
    name = f'__silk_{instance}_nets'
    lines.append(f"{name} = [" + ', '.join(formatter(n) for n in entry.nets) + ']')
    lo = 0 if isinstance(entry.array, int) else entry.array[0]
    index = 'i' if lo == 0 else f'i - ({lo})'
    return f'{name}[{index}]'


def _parse_array_and_naming(d: dict, label: str) -> 'tuple[int | tuple[int, int] | None, str]':
    """Shared array/naming parser. Used by device, ideal_element, source entries.

    Returns ``(array, naming)``. ``label`` is the entry's identifier for
    error messages (e.g. ``f"devices entry {dev['instance']!r}"``).
    """
    array_raw = d.get('array')
    if array_raw is None:
        array = None
    elif isinstance(array_raw, int) and not isinstance(array_raw, bool):
        if array_raw <= 0:
            raise ValueError(f"{label}: 'array' must be > 0, got {array_raw}")
        array = array_raw
    elif isinstance(array_raw, (list, tuple)) and len(array_raw) == 2:
        a, b = array_raw
        if not (isinstance(a, int) and isinstance(b, int)) or b <= a:
            raise ValueError(
                f"{label}: 'array' list must be [a, b] with int a < b, "
                f"got {array_raw!r}"
            )
        array = (a, b)
    else:
        raise ValueError(
            f"{label}: 'array' must be int > 0 or [a, b] with a < b, "
            f"got {array_raw!r}"
        )
    naming = d.get('naming', 'bracketed')
    if naming not in ('bracketed', 'underscore'):
        raise ValueError(
            f"{label}: 'naming' must be 'bracketed' or 'underscore', "
            f"got {naming!r}"
        )
    return array, naming


# ---------------------------------------------------------------------------
# Dependent (controlled) sources — VCVS/VCCS/CCVS/CCCS (TASK-253).
#
# These have no declarative alias surface (silk/sources/library.py's
# build_source_surfaces() skips any source_metadata.yaml entry with no
# quantity/alias — they take no dc/ac/waveform), so unlike
# VoltageSource/CurrentSource they are emitted via a direct
# ``sources.<ClassName>(...)`` call — see _controlled_source_call().
# 'type' accepts either the full class name or the short alias.
# ---------------------------------------------------------------------------
_CONTROLLED_SOURCE_ALIASES = {
    'VCVS': 'VoltageControlledVoltageSource',
    'VCCS': 'VoltageControlledCurrentSource',
    'CCVS': 'CurrentControlledVoltageSource',
    'CCCS': 'CurrentControlledCurrentSource',
}
# Pins, in constructor-argument order, per canonical (full) class name.
_CONTROLLED_SOURCE_PINS = {
    'VoltageControlledVoltageSource': ('out_pos', 'out_neg', 'ctrl_pos', 'ctrl_neg'),
    'VoltageControlledCurrentSource': ('out_pos', 'out_neg', 'ctrl_pos', 'ctrl_neg'),
    'CurrentControlledVoltageSource': ('out_pos', 'out_neg'),
    'CurrentControlledCurrentSource': ('out_pos', 'out_neg'),
}
# The one scalar (gain/gm/transresistance) each class's __init__ takes.
_CONTROLLED_SOURCE_SCALAR = {
    'VoltageControlledVoltageSource': 'gain',
    'VoltageControlledCurrentSource': 'gm',
    'CurrentControlledVoltageSource': 'transresistance',
    'CurrentControlledCurrentSource': 'gain',
}
# CCVS/CCCS additionally take ctrl_vname (a source-instance-name reference,
# not a pin or a scalar — see source_metadata.yaml's `references:` key).
_CONTROLLED_SOURCE_HAS_CTRL_VNAME = {
    'CurrentControlledVoltageSource', 'CurrentControlledCurrentSource',
}

# ---------------------------------------------------------------------------
# Behavioral sources — BehavioralVoltage/BehavioralCurrent (TASK-282.2).
#
# export_source_schema() advertises these (via _meta.unschematized_sources —
# they are not in source_metadata.yaml's `sources:` map, so they get no
# declarative dc/ac/waveform alias surface either), but like the controlled
# sources above they are emitted via a direct ``sources.<ClassName>(...)``
# call — see _behavioral_source_call(). 'type' is the full class name only
# (no short alias exists for these).
# ---------------------------------------------------------------------------
_BEHAVIORAL_SOURCE_TYPES = {'BehavioralVoltage', 'BehavioralCurrent'}
_BEHAVIORAL_SOURCE_FIELDS = ('positive', 'negative', 'expression')


def _behavioral_source_spec_from_dict(src: dict, canonical_type: str,
                                       array, naming) -> 'SourceSpec':
    """Build the ``SourceSpec`` for one behavioral-source ``sources[]``
    entry: ``{type, instance, positive, negative, expression}``. A missing
    field raises ``KeyError`` naming the field and the instance, mirroring
    :func:`_controlled_source_spec_from_dict`.
    """
    instance = src.get('instance', '?')
    label = f"sources entry {instance!r} ({canonical_type})"
    kwargs = {'type': canonical_type, 'instance': src['instance'],
              'array': array, 'naming': naming}
    for field_name in _BEHAVIORAL_SOURCE_FIELDS:
        if field_name not in src:
            raise KeyError(f"{label} missing required {field_name!r}")
        kwargs[field_name] = src[field_name]
    return SourceSpec(**kwargs)


def _canonical_controlled_source_type(raw_type: str) -> 'str | None':
    """Return the canonical (full) class name if *raw_type* names a
    controlled source (full name or short alias), else ``None``."""
    canonical = _CONTROLLED_SOURCE_ALIASES.get(raw_type, raw_type)
    return canonical if canonical in _CONTROLLED_SOURCE_PINS else None


def _controlled_source_spec_from_dict(src: dict, canonical_type: str,
                                       array, naming) -> 'SourceSpec':
    """Build the ``SourceSpec`` for one controlled-source ``sources[]``
    entry, checking the required keys for *canonical_type* up front (a
    missing pin/scalar/reference raises ``KeyError`` here rather than
    surfacing as an opaque ``TypeError`` from the emitted constructor call).
    """
    instance = src.get('instance', '?')
    label = f"sources entry {instance!r} ({canonical_type})"
    kwargs = {'type': canonical_type, 'instance': src['instance'],
              'positive': None, 'negative': None,
              'array': array, 'naming': naming}
    for pin in _CONTROLLED_SOURCE_PINS[canonical_type]:
        if pin not in src:
            raise KeyError(f"{label} missing required {pin!r}")
        kwargs[pin] = src[pin]
    scalar = _CONTROLLED_SOURCE_SCALAR[canonical_type]
    if scalar not in src:
        raise KeyError(f"{label} missing required {scalar!r}")
    kwargs[scalar] = src[scalar]
    if canonical_type in _CONTROLLED_SOURCE_HAS_CTRL_VNAME:
        if 'ctrl_vname' not in src:
            raise KeyError(f"{label} missing required 'ctrl_vname'")
        kwargs['ctrl_vname'] = src['ctrl_vname']
    return SourceSpec(**kwargs)


def _source_index_range(src: 'SourceSpec') -> 'tuple[int, int] | None':
    """Normalize a source's half-open index range; None denotes a scalar."""
    return (0, src.array) if isinstance(src.array, int) else src.array


def _source_range_label(src: 'SourceSpec') -> str:
    bounds = _source_index_range(src)
    return 'scalar' if bounds is None else f'[{bounds[0]}, {bounds[1]})'


def _validate_ctrl_vname_references(srcs: 'list[SourceSpec]', *,
                                    array_controls_only: bool = False) -> None:
    """Require a voltage-source branch current in the same ``sources[]`` list.

    Testbench generation preserves its legacy scalar reference handling;
    there, validate only references involving an array entry.
    """
    by_instance = {s.instance: s for s in srcs}
    # SPICE V, E, H and B V= sources have a branch equation. Use the source
    # classes so aliases and subclasses retain their electrical meaning.
    voltage_types = (sources.VoltageSource,
                     sources.VoltageControlledVoltageSource,
                     sources.CurrentControlledVoltageSource,
                     sources.BehavioralVoltage)
    for s in srcs:
        if s.ctrl_vname is None:
            continue
        target = by_instance.get(s.ctrl_vname)
        if array_controls_only and s.array is None and (target is None or target.array is None):
            continue
        if s.ctrl_vname not in by_instance:
            raise ValueError(
                f"sources entry {s.instance!r} ({s.type}): "
                f"ctrl_vname={s.ctrl_vname!r} does not name any source "
                f"instance in this spec's sources[] (have: "
                f"{sorted(by_instance)})"
            )
        target = by_instance[s.ctrl_vname]
        ranges = (f"entry range {_source_range_label(s)}, "
                  f"controller range {_source_range_label(target)}")
        range_detail = f' ({ranges})' if s.array is not None or target.array is not None else ''
        target_class = getattr(sources, target.type, None)
        if not (isinstance(target_class, type)
                and issubclass(target_class, voltage_types)):
            raise ValueError(
                f"sources entry {s.instance!r} ({s.type}): "
                f"ctrl_vname={s.ctrl_vname!r} targets source "
                f"{target.instance!r} ({target.type}), which has no "
                f"voltage-source branch current to sense. Choose a voltage "
                f"source (V, E, H or B V=), or insert a zero-volt ammeter "
                f"source in the current path{range_detail}."
            )

        if target.array is not None and _source_index_range(s) != _source_index_range(target):
            raise ValueError(
                f"sources entry {s.instance!r} ({s.type}): "
                f"ctrl_vname={s.ctrl_vname!r} has mismatched ranges "
                f"({ranges}); use matching arrays or a scalar controller."
            )


def _validate_entry_type(entry: dict, collection: str) -> None:
    """Reject unknown factories before parsing or emitting an entry."""
    if collection == 'ideal_elements':
        from silk.elements import elements
        accepted = set(elements.device_names)
    else:
        accepted = {name for name, info in _source_meta()['sources'].items()
                    if info.get('alias')}
        accepted.update(_CONTROLLED_SOURCE_PINS)
        accepted.update(_CONTROLLED_SOURCE_ALIASES)
        accepted.update(_BEHAVIORAL_SOURCE_TYPES)
    raw_type = entry.get('type')
    if collection == 'ideal_elements' and raw_type in ('vsource', 'isource'):
        raise ValueError(
            f"ideal_elements entry {entry.get('instance', '?')!r} ({raw_type!r}): "
            "sources belong in sources[]; use VoltageSource or CurrentSource.")
    if not isinstance(raw_type, str) or raw_type not in accepted:
        raise ValueError(
            f"{collection} entry {entry.get('instance', '?')!r}: "
            f"unknown type {raw_type!r}; accepted types: {', '.join(sorted(accepted))}"
        )


def _source_entry_to_spec(src: dict, label: str) -> 'SourceSpec':
    """Build one ``SourceSpec`` from a ``sources[]``-shaped dict."""
    _validate_entry_type(src, 'sources')
    array, naming = _parse_array_and_naming(src, label)
    canonical = _canonical_controlled_source_type(src.get('type', ''))
    if canonical is not None:
        return _controlled_source_spec_from_dict(src, canonical, array, naming)
    if src.get('type') in _BEHAVIORAL_SOURCE_TYPES:
        return _behavioral_source_spec_from_dict(src, src['type'], array, naming)
    return SourceSpec(
        type=src['type'],
        instance=src['instance'],
        positive=src['positive'],
        negative=src['negative'],
        dc=src.get('dc', 0),
        ac=src.get('ac', 0),
        waveform=src.get('waveform'),
        pwl=src.get('pwl'),
        array=array,
        naming=naming,
    )


@dataclass
class VariableSpec:
    """One module variable declaration.

    Maps 1:1 to a ``<name> = Variable(<name>, <value>)`` line in the
    generated code. A parameter refers to it with ``{"variable": name}`` or
    ``{"variable": name, "scale": number}``; see ``_var_ref_expr``.

    Attributes
    ----------
    name : str
        Variable name; it must be a Python identifier that the generated code
        does not otherwise use.
    value : float
        The variable's initial value.

    Examples
    --------
    >>> from silk.codegen import CircuitSpec
    >>> spec = CircuitSpec.from_dict({'circuit_name': 'demo',
    ...                               'variables': [{'name': 'w_in', 'value': 2.0}]})
    >>> spec.variables[0].name, spec.variables[0].value
    ('w_in', 2.0)
    """
    name: str
    value: Any


_VARIABLE_KEYS = ('name', 'value')
_VAR_REF_KEYS = ('variable', 'scale')


def _variables_from_list(raw: Any) -> 'list[VariableSpec]':
    """Parse the ``variables`` key: a list of ``{name, value}`` objects.

    Only the entry shape is checked here; the name and value rules run at
    generate time (:func:`_validate_variables`) so a directly constructed
    :class:`CircuitSpec` is held to the same contract.
    """
    if not isinstance(raw, list):
        raise ValueError(
            f"variables must be a list of {{'name', 'value'}} objects, got "
            f"{type(raw).__name__}"
        )
    out = []
    for n, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"variables[{n}] must be an object with 'name' and 'value', "
                f"got {entry!r}"
            )
        unknown = sorted(k for k in entry if k not in _VARIABLE_KEYS)
        missing = [k for k in _VARIABLE_KEYS if k not in entry]
        if unknown or missing:
            raise ValueError(
                f"variables[{n}] ({entry.get('name', '?')!r}) must have "
                f"exactly the keys 'name' and 'value'"
                + (f"; unknown key(s) {unknown}" if unknown else '')
                + (f"; missing {missing}" if missing else '')
            )
        out.append(VariableSpec(name=entry['name'], value=entry['value']))
    return out


@dataclass
class CircuitSpec:
    """The full circuit description: the generator's input from any tool.

    Usually built with :meth:`from_dict`, :meth:`from_json` or
    :meth:`from_json_file`, which parse and check the JSON payload. The
    payload format is documented in ``docs/codegen-contract.md``.

    Attributes
    ----------
    circuit_name : str
        Module name and output file stem; it must be a Python identifier
        (checked when code is generated).
    pdk : str
        Registered PDK name. Defaults to ``'sky130'``.
    output_dir : str
        Directory the generated ``Module`` writes its netlist to. Defaults to
        ``'./sim'``.
    corner : str
        Process-corner name used by the standalone simulate call; only the
        syntax is checked at generation. Defaults to ``'tt'``.
    voltage_domains : dict of str to silk.circuit.VoltageDomain
        Named supply rails, keyed by name.
    ports : dict of str to silk.circuit.Port, optional
        ``None`` means no ``ports`` key (a standalone simulation script);
        ``{}`` an explicit zero-pin subcircuit; a non-empty dict a subcircuit
        with those pins.
    bus_nets : list of BusNetSpec
        Internal bus-net declarations.
    devices : list of DeviceSpec
        PDK devices.
    ideal_elements : list of DeviceSpec
        Ideal passive elements; sources belong in ``sources[]``.
    sources : list of SourceSpec
        Independent, controlled and behavioral sources.
    simulation : dict
        Analysis to register, for example ``{'type': 'Transient',
        'step': 1e-9, 'stop': 100e-9}`` (seconds); used in standalone mode
        only.
    submodules : list of SubmoduleSpec
        Child module instances.
    variables : list of VariableSpec
        Module variables, in declaration order.

    Examples
    --------
    >>> from silk.codegen import CircuitSpec
    >>> spec = CircuitSpec.from_dict({'circuit_name': 'demo'})
    >>> spec.pdk, spec.corner, spec.ports is None
    ('sky130', 'tt', True)
    """
    circuit_name: str
    pdk: str = 'sky130'
    output_dir: str = './sim'
    corner: str = 'tt'
    voltage_domains: 'dict[str, VoltageDomain]' = field(default_factory=dict)
    # Parsed at from_dict time from {port: {dir, role?, domain?, bits?, naming?}}.
    # None = no 'ports' key (standalone script); {} = explicit zero-pin
    # subcircuit; non-empty = subcircuit with those pins.
    ports: 'dict[str, Port] | None' = None
    bus_nets: list[BusNetSpec] = field(default_factory=list)
    devices: list[DeviceSpec] = field(default_factory=list)
    ideal_elements: list[DeviceSpec] = field(default_factory=list)
    sources: list[SourceSpec] = field(default_factory=list)
    simulation: dict[str, Any] = field(default_factory=dict)
    submodules: list[SubmoduleSpec] = field(default_factory=list)
    # Module variables, in declaration order (TASK-376).
    variables: list[VariableSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> 'CircuitSpec':
        """Build a spec from a caller's dict payload.

        Parameters
        ----------
        d : dict
            Circuit description in the format of
            ``docs/codegen-contract.md``. Missing optional keys take
            their defaults (``circuit_name`` defaults to ``'my_circuit'``).

        Returns
        -------
        CircuitSpec
            The parsed spec.

        Raises
        ------
        KeyError
            A required key of an entry is missing (for example a device's
            ``type``, a port's ``dir`` or a voltage domain's ``nominal``).
        ValueError
            A value is invalid: a bad ``array`` or ``naming``, an invalid
            domain or submodule name, a port naming an undefined domain, or
            a malformed ``variables`` list.
        TypeError
            An entry has the wrong type, for example a port that is not a
            dict.

        """
        devices = []
        for dev in d.get('devices', []):
            array, naming = _parse_array_and_naming(
                dev, f"devices entry {dev.get('instance', '?')!r}"
            )
            devices.append(DeviceSpec(
                type=dev['type'],
                instance=dev['instance'],
                nets=dev.get('nets', {}),
                params=dev.get('params', {}),
                array=array,
                naming=naming,
            ))
        ideal = []
        srcs = []
        for elem in d.get('ideal_elements', []):
            _validate_entry_type(elem, 'ideal_elements')
            array, naming = _parse_array_and_naming(
                elem, f"ideal_elements entry {elem.get('instance', '?')!r}"
            )
            ideal.append(DeviceSpec(
                type=elem['type'],
                instance=elem['instance'],
                nets=elem.get('nets', {}),
                params=elem.get('params', {}),
                array=array,
                naming=naming,
            ))
        for src in d.get('sources', []):
            srcs.append(_source_entry_to_spec(
                src, f"sources entry {src.get('instance', '?')!r}"))
        _validate_ctrl_vname_references(srcs)
        voltage_domains = {}
        for dom_name, dom_entry in d.get('voltage_domains', {}).items():
            if not dom_name.isidentifier() or keyword.iskeyword(dom_name):
                raise ValueError(
                    f"voltage_domains: domain name {dom_name!r} is not a "
                    f"valid Python identifier or is a Python keyword"
                )
            if 'nominal' not in dom_entry:
                raise KeyError(
                    f"voltage_domains entry {dom_name!r} missing required "
                    f"'nominal'"
                )
            opt_kwargs = {
                k: dom_entry[k]
                for k in ('min', 'max', 'tolerance', 'switchable')
                if k in dom_entry
            }
            voltage_domains[dom_name] = VoltageDomain(
                name=dom_name,
                nominal=dom_entry['nominal'],
                **opt_kwargs,
            )

        ports_raw = d.get('ports')
        ports = None if ports_raw is None else {}
        ports_raw = ports_raw or {}
        for name, entry in ports_raw.items():
            _require_port_name(name)
            if not isinstance(entry, dict):
                raise TypeError(
                    f"port {name!r}: expected dict, got "
                    f"{type(entry).__name__} ({entry!r}). String form is no "
                    f"longer accepted — use {{'dir': '<input|output|...>'}}."
                )
            if 'dir' not in entry:
                raise KeyError(
                    f"port {name!r}: required key 'dir' missing"
                )

            dir_ = PortDir(entry['dir'])
            role = PortRole(entry['role']) if entry.get('role') else None

            domain_name = entry.get('domain')
            if domain_name is None:
                domain = None
            elif domain_name not in voltage_domains:
                raise ValueError(
                    f"port {name!r}: domain {domain_name!r} not defined in "
                    f"voltage_domains (known: "
                    f"{sorted(voltage_domains.keys()) or '[]'})"
                )
            else:
                domain = voltage_domains[domain_name]

            raw_bits = entry.get('bits')
            if raw_bits is None:
                bits = None
            elif isinstance(raw_bits, int):
                bits = raw_bits
            elif isinstance(raw_bits, list) and len(raw_bits) == 2:
                bits = range(raw_bits[0], raw_bits[1])
            else:
                raise ValueError(
                    f"port {name!r}: bits must be int or [start, stop] list, "
                    f"got {raw_bits!r}"
                )

            port_kwargs = {'dir': dir_, 'role': role, 'domain': domain, 'bits': bits}
            if entry.get('naming') is not None:
                port_kwargs['naming'] = entry['naming']
            ports[name] = Port(**port_kwargs)

        bus_nets = [BusNetSpec.from_dict(b) for b in d.get('bus_nets', [])]
        submodules = [
            SubmoduleSpec.from_dict(s) for s in d.get('submodules', [])
        ]
        seen_inst_names = set()
        for sm in submodules:
            if (not sm.instance_name.isidentifier()
                    or keyword.iskeyword(sm.instance_name)):
                raise ValueError(
                    f"submodules entry {sm.instance_name!r}: "
                    f"instance_name is not a valid Python identifier "
                    f"or is a Python keyword"
                )
            if sm.instance_name in seen_inst_names:
                raise ValueError(
                    f"submodules: duplicate instance_name "
                    f"{sm.instance_name!r}"
                )
            seen_inst_names.add(sm.instance_name)
            if (not sm.module_var.isidentifier()
                    or keyword.iskeyword(sm.module_var)):
                raise ValueError(
                    f"submodules entry {sm.instance_name!r}: "
                    f"module_var {sm.module_var!r} is not a valid "
                    f"Python identifier"
                )
            if not sm.import_path:
                raise ValueError(
                    f"submodules entry {sm.instance_name!r}: "
                    f"import_path must be non-empty"
                )

        return cls(
            circuit_name=d.get('circuit_name', 'my_circuit'),
            pdk=d.get('pdk', 'sky130'),
            output_dir=d.get('output_dir', './sim'),
            corner=d.get('corner', 'tt'),
            voltage_domains=voltage_domains,
            ports=ports,
            bus_nets=bus_nets,
            devices=devices,
            ideal_elements=ideal,
            sources=srcs,
            simulation=d.get('simulation', {}),
            submodules=submodules,
            variables=_variables_from_list(d.get('variables', [])),
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'CircuitSpec':
        """Build a spec from a JSON string.

        Parameters
        ----------
        json_str : str
            JSON text of the circuit description.

        Returns
        -------
        CircuitSpec
            The parsed spec.

        Raises
        ------
        ValueError
            The text is not valid JSON, or the description is invalid (see
            :meth:`from_dict`).
        """
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_json_file(cls, path: str) -> 'CircuitSpec':
        """Build a spec from a JSON file.

        Parameters
        ----------
        path : str
            Path of a JSON file holding the circuit description.

        Returns
        -------
        CircuitSpec
            The parsed spec.

        Raises
        ------
        OSError
            The file cannot be read.
        ValueError
            The file is not valid JSON, or the description is invalid (see
            :meth:`from_dict`).
        """
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Code generator
# ---------------------------------------------------------------------------

def _sanitize_filename(name: str) -> str:
    """Sanitize a circuit name for use as a filename.

    Keeps alphanumeric, underscore, and hyphen. Replaces spaces with
    underscores. Strips everything else.
    """
    name = name.replace(' ', '_')
    return re.sub(r'[^\w\-]', '', name)


def _pdk_conf(name: str) -> dict:
    """Look up a PDK's manifest via the registry.

    Raises ``silk.pdk.registry.PDKRegistryError`` (a ``ValueError``
    subclass) for an unknown name — no silent placeholder. Kept as a thin
    wrapper so callers don't need to know the registry import path.
    """
    from silk.pdk import registry
    return registry.manifest(name)


def _emit_pdk_setup(spec: 'CircuitSpec', lines: list) -> None:
    """Emit the shared ``# --- PDK setup ---`` preamble (standalone + module).

    Emits ``from silk import load_pdk`` + ``pdk = load_pdk('<name>')``.
    ``load_pdk`` resolves the volare library path (``$PDK_ROOT`` + the
    manifest's ``library.lib_path``) and the package-relative device-details
    yaml internally (see silk/pdk/_loader.py), so the generated file works
    regardless of the process cwd — not only from silk's repo root, and not
    only with a specific hardcoded raw skywater-pdk layout. (A cwd-relative
    literal broke ``python -m <proj>.<mod>`` for standalone consumers.) Validates ``spec.pdk`` against
    the registry so an unknown PDK name fails at generation time, not with a
    silent placeholder in the emitted code.
    """
    _pdk_conf(spec.pdk)  # raises PDKRegistryError for an unknown pdk name
    lines.append('# --- PDK setup ---')
    lines.append('from silk import load_pdk')
    lines.append(f'pdk = load_pdk({spec.pdk!r})')
    lines.append('')


def _fmt_value(v: Any) -> str:
    """Format a parameter value for Python source code.

    A variable reference object emits the variable expression; it
    has already been checked by :func:`_validate_variables`.
    """
    if isinstance(v, dict) and 'variable' in v:
        return _var_ref_expr(v)
    if isinstance(v, float):
        # Use scientific notation for very small/large values
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
            return f'{v:g}'
        return repr(v)
    return repr(v)


# Corner names are emitted into ``ckt.simulate(corner=...)`` and the generated
# header. The contract (TASK-13) is syntactic only: letters, digits,
# underscore, hyphen. Membership in the selected PDK's corner set is NOT
# checked here — emission must work without a resolvable PDK environment,
# and ``Module.simulate`` already rejects unknown corners against the
# manifest at run time (the single authoritative check).
_CORNER_RE = re.compile(r'[A-Za-z0-9_-]+')

# Analysis defaults applied when the codegen contract omits a field (AUDIT-02:
# named with their unit; values unchanged).
_DEFAULT_TRAN_STEP_S = 1e-9
_DEFAULT_TRAN_STOP_S = 100e-9
_DEFAULT_AC_FSTOP_HZ = 1e9


def _require_corner(value: Any, context: str = 'corner') -> str:
    """Return *value* or fail if it is not a syntactically valid corner name."""
    if not isinstance(value, str) or _CORNER_RE.fullmatch(value) is None:
        raise ValueError(
            f"{context} {value!r} must be a non-empty string of letters, "
            f"digits, underscores, or hyphens ([A-Za-z0-9_-]+)"
        )
    return value


# Port names become SPICE subcircuit pins and Python dict keys in the
# emitted ``define_module_ports({...})`` call (TASK-324). They are emitted
# via repr(), so — unlike circuit_name/instance names, which are bound as
# bare Python locals — a Python keyword such as 'in' stays legal (analog
# designers name ports in/out routinely; tests/test_codegen_emission_safety
# depends on it). The contract is syntactic only: an identifier-shaped
# string. Bracketed bus forms like 'd[3:0]' are not a port-name form —
# width is carried separately via Port.bits / bus_nets.
_PORT_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _require_port_name(value: Any, context: str = 'port') -> str:
    """Return *value* or fail if it is not an identifier-shaped port name."""
    if not isinstance(value, str) or _PORT_NAME_RE.fullmatch(value) is None:
        raise ValueError(
            f"{context} name {value!r} must be an identifier-shaped string "
            f"matching ^[A-Za-z_][A-Za-z0-9_]*$"
        )
    return value


def _require_points(value: Any, context: str = 'points') -> str:
    """Return an integer-literal string for an AC/Noise ``points`` count.

    Accepts a plain ``int`` or an SI string that :func:`_parse_si` resolves
    to a positive integer (``'1k'`` -> ``1000``) within finite float range.
    No smaller sweep-count cap is imposed. Anything else — a
    non-integer number, an unparseable string, or any other type — is
    refused. Silk owns the emitted call, so this is the only path 'points'
    takes into generated code; nothing from *value* is interpolated bare.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = _parse_si(value)
        except ValueError:
            raise ValueError(
                f"{context} {value!r} must be an int or an SI string "
                f"parsing to a positive integer"
            ) from None
    else:
        raise ValueError(
            f"{context} {value!r} must be an int or an SI string "
            f"parsing to a positive integer"
        )

    try:
        finite = math.isfinite(parsed)
    except OverflowError:
        # Avoid repr(value): huge ints can exceed Python's digit limit too.
        raise ValueError(
            f"{context} <integer outside float range> must be an int or an SI string "
            f"parsing to a positive integer"
        ) from None

    if not finite or parsed <= 0 or int(parsed) != parsed:
        raise ValueError(
            f"{context} {value!r} must be an int or an SI string "
            f"parsing to a positive integer"
        )
    return str(int(parsed))


# ---------------------------------------------------------------------------
# Module variables (TASK-376)
# ---------------------------------------------------------------------------

# Names a module variable may not take: every local, import or loop index the
# generated file binds or reads as a bare name. ``i`` is the index of the
# arrayed-component comprehensions, and ``range``/``slice`` are read by them
# and by the port literals.
_RESERVED_VARIABLE_NAMES = frozenset({
    'pdk', 'mod', 'ckt', 'os', 'analysis', 'sources', 'elements',
    '_pdk_root', '_lib_path',
    'Module', 'Variable', 'Port', 'PortDir', 'PortRole',
    'VoltageDomain', 'Net', 'load_pdk',
    'i', 'range', 'slice',
})


def _is_number(v: Any) -> bool:
    """A finite int or float, never a bool."""
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v))


def _var_ref_expr(ref: dict) -> str:
    """Python expression for a checked variable reference.

    ``{"variable": "w_in"}`` emits ``w_in``; ``{"variable": "w_in",
    "scale": 2}`` emits ``2.0 * w_in`` — scale first, as a float literal,
    never simplified (a scale of 1 still emits ``1.0 * w_in``).
    """
    if 'scale' in ref:
        return f"{float(ref['scale'])!r} * {ref['variable']}"
    return ref['variable']


def _validate_source_waveforms(src):
    """Preserve omission and reject malformed authored source waveforms."""
    label = f'Source {src.instance!r}'
    if src.waveform is not None and src.pwl is not None:
        raise ValueError(f'{label}: conflicting waveform and pwl fields; supply one.')
    if src.waveform is not None:
        wave = src.waveform
        if not isinstance(wave, dict) or not isinstance(wave.get('type'), str) or not wave['type']:
            raise ValueError(f'{label}: waveform must be a nonempty dict with a type.')
        if 'params' in wave and not isinstance(wave['params'], dict):
            raise ValueError(f'{label}: waveform params must be a dict.')
        if 'params' in wave and set(wave) - {'type', 'params'}:
            raise ValueError(f"{label}: waveform dict mixes params and flat keywords; use one form.")
        params = wave.get('params', {k: v for k, v in wave.items() if k != 'type'})
        try:
            kind = require_python_identifier(_waveform_kind(wave['type']), 'source waveform kind')
            for key in params:
                require_python_identifier(key, 'source waveform parameter key')
            registry = _source_meta()['waveforms']
            if kind not in registry:
                raise ValueError(f"unknown waveform type {wave['type']!r}")
            from silk.sources import waveforms
            import inspect
            signature = inspect.signature(getattr(waveforms, registry[kind]))
            signature.bind(**params)
            missing = [name for name, param in signature.parameters.items()
                       if param.default is waveforms.REQUIRED and name not in params]
            if missing:
                raise ValueError(f"missing required waveform parameter(s): {', '.join(missing)}")
        except (TypeError, ValueError) as error:
            raise ValueError(f'{label}: waveform: {error}') from error
    if src.pwl is not None:
        if not isinstance(src.pwl, (list, tuple)) or len(src.pwl) < 2 or len(src.pwl) % 2:
            raise ValueError(f'{label}: pwl must be a nonempty sequence of time/value pairs.')


def _source_value_slots(src: 'SourceSpec'):
    """Yield ``(param, value)`` for every numeric slot of one source."""
    _validate_source_waveforms(src)
    for name in ('dc', 'ac', 'gain', 'gm', 'transresistance'):
        value = getattr(src, name)
        if value is not None:
            yield name, value
    if isinstance(src.waveform, dict):
        params = src.waveform.get('params')
        items = (params.items() if isinstance(params, dict) else
                 ((k, v) for k, v in src.waveform.items() if k != 'type'))
        for k, v in items:
            yield f"waveform {k}", v
    for j, v in enumerate(src.pwl or ()):
        yield f"pwl[{j}]", v


def _value_slots(spec: 'CircuitSpec'):
    """Yield ``(entry_label, param, value)`` for every numeric slot that may
    hold a variable reference."""
    for dev in spec.devices:
        for k, v in dev.params.items():
            yield f"device {dev.instance!r}", k, v
    for elem in spec.ideal_elements:
        for k, v in elem.params.items():
            yield f"ideal element {elem.instance!r}", k, v
    for src in spec.sources:
        for k, v in _source_value_slots(src):
            yield f"source {src.instance!r}", k, v


def _check_var_ref(ref: dict, where: str, declared: 'set[str]') -> str:
    """Check one reference object; return the variable name it names."""
    unknown = sorted(str(k) for k in ref if k not in _VAR_REF_KEYS)
    if unknown or 'variable' not in ref:
        raise ValueError(
            f"{where} must be a number or a variable reference "
            f"{{'variable': <name>}} / {{'variable': <name>, 'scale': "
            f"<number>}}, got {ref!r}"
        )
    name = ref['variable']
    if not isinstance(name, str) or name not in declared:
        raise ValueError(
            f"{where} references undefined variable {name!r} (declared: "
            f"{sorted(declared) or '[]'})"
        )
    if 'scale' in ref and not _is_number(ref['scale']):
        raise ValueError(
            f"{where} scales variable {name!r} by {ref['scale']!r}; the "
            f"scale must be a finite int or float"
        )
    return name


def _validate_variables(spec: 'CircuitSpec') -> None:
    """Enforce the module-variable contract before emission.

    Declarations: identifier, not a keyword, unique, not a name the
    generated file binds (:data:`_RESERVED_VARIABLE_NAMES`, voltage domain
    names, submodule import aliases), finite numeric value. References: a
    well-formed object naming a declared variable, finite numeric scale.
    A declared variable that no parameter references is an error: the line
    would be dead code that ``mod.variables()`` and the optimiser never see.
    """
    variables = spec.variables
    slots = _value_slots(spec)
    if not variables:
        for label, param, value in slots:
            if isinstance(value, dict):
                _check_var_ref(value, f"{label} param {param!r}", set())
        return
    aliases = set(_compute_submodule_aliases(
        spec.submodules,
        additional_reserved=_emitted_voltage_domain_names(spec) | _per_copy_nets_bindings(spec),
    ).values())
    declared: set = set()
    for var in variables:
        name = var.name
        if (not isinstance(name, str) or not name.isidentifier()
                or keyword.iskeyword(name)):
            raise ValueError(
                f"variable {name!r}: name must be a valid Python identifier "
                f"and not a Python keyword"
            )
        if name in declared:
            raise ValueError(f"variable {name!r}: declared more than once")
        if name in _RESERVED_VARIABLE_NAMES | _per_copy_nets_bindings(spec):
            raise ValueError(
                f"variable {name!r}: name clashes with a name the generated "
                f"code binds (reserved: {sorted(_RESERVED_VARIABLE_NAMES)})"
            )
        if (name in spec.voltage_domains
                or name in _emitted_voltage_domain_names(spec)):
            raise ValueError(
                f"variable {name!r}: name clashes with voltage domain {name!r}"
            )
        if name in aliases:
            raise ValueError(
                f"variable {name!r}: name clashes with a generated submodule "
                f"import alias"
            )
        if not _is_number(var.value):
            raise ValueError(
                f"variable {name!r}: value {var.value!r} must be a finite "
                f"int or float"
            )
        declared.add(name)
    used: set = set()
    for label, param, value in slots:
        if isinstance(value, dict):
            used.add(_check_var_ref(value, f"{label} param {param!r}", declared))
    unused = [v.name for v in variables if v.name not in used]
    if unused:
        raise ValueError(
            f"variable(s) {unused} declared but not referenced by any device, "
            f"element or source parameter; reference or remove them"
        )


def _emit_variables(spec: 'CircuitSpec', lines: list) -> None:
    """Append the ``# --- module variables ---`` block, declaration order."""
    if not spec.variables:
        return
    lines.append('# --- module variables ---')
    for var in spec.variables:
        lines.append(f"{var.name} = Variable({var.name!r}, {var.value!r})")
    lines.append('')


def _validate_identity_fields(spec: 'CircuitSpec') -> None:
    """Enforce circuit, corner, port and emitted voltage-domain name contracts.

    ``circuit_name`` is an identity (SPICE ``.subckt`` namespace, tracking
    DB structure identity, ``module_name=`` literal, output filename), so it
    must be a valid Python identifier and not a keyword — the same rule
    ``testbench_generator`` applies before emitting
    ``build_<circuit_name>_with_tests``. ``corner`` must satisfy
    :func:`_require_corner`. Port names must satisfy :func:`_require_port_name` — they become SPICE subcircuit pins and Python dict keys.
    Runs at the emission boundary so dict input (already checked once in
    ``CircuitSpec.from_dict``) and directly constructed specs are held to
    the same contract, and fails before any output is assembled.
    """
    require_python_identifier(spec.circuit_name, 'circuit_name')
    _require_corner(spec.corner, 'corner')
    if spec.ports is not None:
        for domain in spec.voltage_domains.values():
            require_python_identifier(domain.name, 'voltage_domains: domain name')
    for port_name in (spec.ports or {}):
        _require_port_name(port_name, 'port')


def _docstring_safe(value: Any) -> str:
    """Render *value* so it cannot terminate a generated triple-quoted header.

    Escapes backslashes, double quotes, and CR/LF. Byte-identical for any
    value that passes the emission contract (identifiers and corner names
    contain none of those characters); this is defense in depth behind
    :func:`_validate_identity_fields`, not the primary guard.
    """
    return (str(value)
            .replace('\\', '\\\\')
            .replace('"', '\\"')
            .replace('\r', '\\r')
            .replace('\n', '\\n'))


def _voltage_domain_line(domain, bindings) -> str:
    """Render one VoltageDomain(...) constructor line.

    Emits `nominal=` (required) plus any of `tolerance`, `min`, `max`
    that are set on the instance and `switchable=True` when the rail is
    declared switchable. Skips unset optional fields.
    """
    kwargs = [f"nominal={_fmt_value(domain.nominal)}"]
    if domain.tolerance is not None:
        kwargs.append(f"tolerance={_fmt_value(domain.tolerance)}")
    if domain.min is not None:
        kwargs.append(f"min={_fmt_value(domain.min)}")
    if domain.max is not None:
        kwargs.append(f"max={_fmt_value(domain.max)}")
    if getattr(domain, 'switchable', False):
        kwargs.append("switchable=True")
    return (
        f"{bindings[domain.name]} = VoltageDomain({domain.name!r}, "
        f"{', '.join(kwargs)})"
    )


def _require_source(sim: dict, kind: str) -> str:
    """The source a dc or noise analysis names; never defaulted."""
    src = sim.get('source')
    if not str(src or '').strip():
        raise ValueError(f"{kind} analysis: names no source")
    return src


def _sim_code(sim: dict, receiver: str = 'ckt') -> str:
    """Generate the add_sim_type line from a simulation dict.

    Supported types are defined by silk.simulation.analysis classes.
    Type comparison is case-insensitive (e.g., 'transient' or 'Transient').
    The simulation kind is read from either the `type` key (circuit path)
    or the `kind` key (testbench analyses); `type` wins if both are present.
    `receiver` names the variable the emitted line calls add_sim_type on
    (`'ckt'` for the circuit path, `'tb'` for the testbench generator).
    Raises ValueError if the simulation type is not supported.
    """
    sim_type = sim.get('type') or sim.get('kind', 'Transient')
    # Normalize to title case for comparison
    sim_type_normalized = sim_type.capitalize() if isinstance(sim_type, str) else sim_type

    if sim_type_normalized == 'Transient':
        step = _fmt_value(sim.get('step', _DEFAULT_TRAN_STEP_S))
        stop = _fmt_value(sim.get('stop', _DEFAULT_TRAN_STOP_S))
        return f"{receiver}.add_sim_type(analysis.Transient(step={step}, stop={stop}))"

    if sim_type_normalized == 'Ac':
        variation = repr(sim.get('variation', 'dec'))
        points    = _require_points(sim.get('points', 10))
        fstart    = _fmt_value(sim.get('fstart', 1))
        fstop     = _fmt_value(sim.get('fstop', _DEFAULT_AC_FSTOP_HZ))
        return f"{receiver}.add_sim_type(analysis.AC({variation}, {points}, {fstart}, {fstop}))"

    if sim_type_normalized == 'Dc':
        src   = repr(_require_source(sim, 'dc'))
        start = _fmt_value(sim.get('start', 0))
        stop  = _fmt_value(sim.get('stop', 1.8))
        step  = _fmt_value(sim.get('step', 0.01))
        return (f"{receiver}.add_sim_type(analysis.DC("
                f"source={src}, start={start}, stop={stop}, step={step}))")

    if sim_type_normalized == 'Noise':
        out       = repr(sim.get('output', 'vout'))
        ref       = repr(sim.get('ref', '0'))
        source    = repr(_require_source(sim, 'noise'))
        variation = repr(sim.get('variation', 'dec'))
        points    = _require_points(sim.get('points', 10))
        fstart    = _fmt_value(sim.get('fstart', 1))
        fstop     = _fmt_value(sim.get('fstop', _DEFAULT_AC_FSTOP_HZ))
        return (f"{receiver}.add_sim_type(analysis.Noise("
                f"{out}, {ref}, {source}, {variation}, {points}, {fstart}, {fstop}))")

    if sim_type_normalized == 'Op':
        # No parameters (TASK-330.2): a bare operating-point request.
        return f"{receiver}.add_sim_type(analysis.Op())"

    # Get supported types from analysis module classes
    supported = {name for name in dir(analysis)
                 if not name.startswith('_') and
                 isinstance(getattr(analysis, name), type) and
                 issubclass(getattr(analysis, name), analysis.AnalysisBase) and
                 getattr(analysis, name) is not analysis.AnalysisBase}

    raise ValueError(
        f"Unknown simulation type: {sim_type!r}. "
        f"Supported types: {', '.join(sorted(supported))}"
    )


_RESERVED_SUBMODULE_ALIASES = frozenset({
    'mod', 'ckt', 'pdk', 'os', 'analysis',
    'sources', 'elements',
    'Module', 'Port', 'PortDir', 'PortRole', 'VoltageDomain', 'Net', 'load_pdk',
})


def _sanitize_import_path(import_path: str) -> str:
    """Sanitize an import path into the suffix used in the alias name.

    Strips leading dots, replaces ``.``, ``/``, ``-`` with ``_``, and
    drops any other non-identifier character. May return an empty
    string for inputs that consist only of leading dots.
    """
    stripped = import_path.lstrip('.')
    normalized = re.sub(r'[./\-]', '_', stripped)
    return re.sub(r'[^A-Za-z0-9_]', '', normalized)


def _compute_submodule_aliases(
    submodules: 'list[SubmoduleSpec]',
    *,
    additional_reserved=(),
) -> 'dict[tuple[str, str], str]':
    """Compute a deterministic alias for each unique (import_path, module_var).

    Returns a dict keyed by ``(import_path, module_var)`` so multiple
    instances sharing the same child collapse to a single import line.
    Aliases avoid the silk-generator banlist, caller-supplied live
    bindings, and earlier aliases, disambiguating with a numeric counter in
    spec-list order.
    """
    aliases: dict[tuple[str, str], str] = {}
    used: set[str] = set(additional_reserved)
    for sm in submodules:
        key = (sm.import_path, sm.module_var)
        if key in aliases:
            continue
        sanitized = _sanitize_import_path(sm.import_path)
        base = f'{sm.module_var}_{sanitized}' if sanitized else f'{sm.module_var}_'
        candidate = base
        suffix = 1
        while (candidate in _RESERVED_SUBMODULE_ALIASES
               or candidate in used
               or not candidate.isidentifier()
               or keyword.iskeyword(candidate)):
            candidate = f'{base}_{suffix}'
            suffix += 1
            if suffix > 1000:
                raise RuntimeError(
                    f"submodule alias computation failed to converge for "
                    f"import_path={sm.import_path!r} module_var={sm.module_var!r}"
                )
        aliases[key] = candidate
        used.add(candidate)
    return aliases


def _emitted_voltage_domain_names(spec: 'CircuitSpec') -> 'set[str]':
    """Names bound by VoltageDomain constructors in module-mode output."""
    if spec.ports is None:
        return set()
    return {domain.name for domain in spec.voltage_domains.values()}


def _compute_voltage_domain_bindings(spec: 'CircuitSpec') -> 'dict[str, str]':
    """Allocate safe Python bindings while preserving authored rail names.

    Reserve all authored names up front so an alias cannot displace an
    ordinary domain. Submodule aliases keep their existing allocation order;
    domains avoid those aliases, variables and per-copy wiring locals.
    Standalone output emits no domain bindings.
    """
    if spec.ports is None:
        return {}
    names = _emitted_voltage_domain_names(spec)
    wiring = _per_copy_nets_bindings(spec)
    aliases = _compute_submodule_aliases(
        spec.submodules, additional_reserved=names | wiring)
    reserved = (_RESERVED_VARIABLE_NAMES | wiring | set(aliases.values())
                | {var.name for var in spec.variables})
    # Python normalizes identifiers before binding them in a namespace.
    reserved = {unicodedata.normalize('NFKC', name) for name in reserved}
    used = {unicodedata.normalize('NFKC', name) for name in names} | reserved
    allocated = set()
    bindings = {}
    for domain in spec.voltage_domains.values():
        name = domain.name
        if name in bindings:
            continue
        candidate = name
        normalized = unicodedata.normalize('NFKC', candidate)
        if normalized in reserved or normalized in allocated:
            base = f'domain_{name}'
            candidate = base
            suffix = 1
            while unicodedata.normalize('NFKC', candidate) in used:
                candidate = f'{base}_{suffix}'
                suffix += 1
        bindings[name] = candidate
        normalized = unicodedata.normalize('NFKC', candidate)
        allocated.add(normalized)
        used.add(normalized)
    return bindings


def _validate_sweep_source(spec: 'CircuitSpec') -> None:
    """Refuse a circuit-level dc or noise analysis whose source is not one
    source element of the circuit. ``vin[1]`` is an element of an
    arrayed source ``vin``; the bare ``vin`` is not a source."""
    sim = spec.simulation or {}
    kind = str(sim.get('type') or sim.get('kind', 'Transient')).lower()
    if spec.ports is not None or kind not in ('dc', 'noise'):
        return   # a subcircuit emits no analysis
    from silk.circuit.sweep_sources import (
        missing_source_problem, sweep_source_problem)
    names, arrays = [], {}
    for src in spec.sources:
        if src.array is None:
            names.append(src.instance)
            continue
        lo, hi = (0, src.array) if isinstance(src.array, int) else src.array
        fmt = "{}[{}]" if src.naming == 'bracketed' else "{}_{}"
        arrays[src.instance] = [fmt.format(src.instance, i) for i in range(lo, hi)]
        names += arrays[src.instance]
    allow = ('temp',) if kind == 'dc' else ()
    if not str(sim.get('source') or '').strip():
        raise ValueError(
            f"{kind} analysis: {missing_source_problem(names, allow)}")
    problem = sweep_source_problem(sim['source'], names, arrays, allow=allow)
    if problem:
        raise ValueError(f"{kind} analysis: source {problem}")


def _validate_instance_bindings(spec: 'CircuitSpec') -> None:
    """Require every emitted instance name to be a valid Python identifier.

    Instance names are only ever string arguments of the factory calls
    (``mod.add(pdk.nmos1p8('pdk', ...))``), never Python variables, so a
    name equal to a generator local (``pdk``, ``mod``, ``ckt``, ...) is legal.
    """
    groups = (
        ("device instance name", (dev.instance for dev in spec.devices)),
        (
            "ideal element instance name",
            (elem.instance for elem in spec.ideal_elements),
        ),
        ("source instance name", (src.instance for src in spec.sources)),
        (
            "submodule instance name",
            (
                sm.instance_name
                for sm in spec.submodules
                if sm.array is not None
            ),
        ),
    )
    for context, instances in groups:
        for instance in instances:
            require_python_identifier(instance, context)


# ══════════════════════════════════════════════════════════════════════════════
#  Index template rewrite — $i, $(i), $(i±N) → f-string {i}, {i±N}
# ══════════════════════════════════════════════════════════════════════════════
_TEMPLATE_RE = re.compile(
    r'\$(?:'
    r'\(\s*i\s*([+\-])\s*(\d+)\s*\)'   # $(i+N) or $(i-N)
    r'|\(\s*i\s*\)'                    # $(i)
    r'|i'                              # $i (bare)
    r')'
)


def _rewrite_template(s: str) -> 'tuple[str, bool]':
    """Rewrite index placeholders in ``s`` into Python f-string fragments.

    Tokens recognized (where N is a positive integer):

    - ``$i``        →  ``{i}``
    - ``$(i)``      →  ``{i}``
    - ``$(i+N)``    →  ``{i+N}``
    - ``$(i-N)``    →  ``{i-N}``

    Returns ``(rewritten, has_placeholder)``. If ``has_placeholder`` is
    False the caller can emit ``s`` as a plain string literal; otherwise
    the caller must emit it as an f-string.
    """
    if not _TEMPLATE_RE.search(s):
        return s, False

    escaped = s.replace('{', '{{').replace('}', '}}')

    def _sub(m):
        sign, n = m.group(1), m.group(2)
        if sign is None:
            return '{i}'
        return '{i' + sign + n + '}'
    return _TEMPLATE_RE.sub(_sub, escaped), True


def _bus_typed_names(spec: 'CircuitSpec') -> 'set[str]':
    """Names in the parent spec that refer to multi-bit nets.

    Union of bus_nets entries and ports whose Port.bits is set.
    Used by the submodule emitter to decide between Net-handle and
    plain-string emission (spec §4.3).
    """
    names = {bn.name for bn in spec.bus_nets}
    for port_name, port in (spec.ports or {}).items():
        if port.bits is not None:
            names.add(port_name)
    return names


def _format_submodule_nets_dict(
    nets: 'dict[str, str]',
    spec: 'CircuitSpec',
    *,
    module_var: str = 'mod',
) -> str:
    """Format the ``nets={...}`` literal for a submodule add_element call.

    Values that match a parent bus-typed name are emitted as
    ``<module_var>.nets.<name>`` (Net handle, required by silk runtime
    to take the bus-expansion branch). Python keywords and non-identifiers
    use bracket lookup with a repr key instead. Other values are emitted as
    plain strings. Mixed types coexist in a single dict.

    ``module_var`` is the Python variable name the emitted code uses
    for the parent Module (``'mod'`` in subcircuit mode, ``'ckt'`` in
    standalone mode).
    """
    bus_names = _bus_typed_names(spec)
    if not nets:
        return '{}'
    parts = []
    for port, value in nets.items():
        if value in bus_names:
            if value.isidentifier() and not keyword.iskeyword(value):
                parts.append(f"{port!r}: {module_var}.nets.{value}")
            else:
                # Python keywords such as in and and, and non-identifiers,
                # cannot follow a dot even when they are legal circuit names.
                parts.append(f"{port!r}: {module_var}.nets[{value!r}]")
        else:
            parts.append(f"{port!r}: {value!r}")
    return '{' + ', '.join(parts) + '}'


def _format_array_nets_dict(
    nets: 'dict[str, str]',
    spec: 'CircuitSpec',
    *,
    module_var: str = 'mod',
) -> str:
    """Format the ``nets={...}`` literal for an ARRAYED submodule call.

    Identical to ``_format_submodule_nets_dict`` except that string values
    containing ``$i`` / ``$(i±N)`` placeholders are rewritten to f-string
    fragments and emitted as ``f'...'`` rather than plain ``'...'``.

    Bus-typed parent net values emit as ``<module_var>.nets.<name>``
    (Net handle) for broadcast cases (no placeholder), with repr-key bracket
    lookup only for Python keywords and non-identifiers. Values with
    placeholders always emit as f-strings regardless of bus-typed-name
    membership, without assuming positional indexing — bus
    expansion happens once per loop iteration through the silk runtime
    path.
    """
    bus_names = _bus_typed_names(spec)
    if not nets:
        return '{}'
    parts = []
    for port, value in nets.items():
        rewritten, has_placeholder = _rewrite_template(value)
        if has_placeholder:
            parts.append(f"{port!r}: f{rewritten!r}")
        elif value in bus_names:
            if value.isidentifier() and not keyword.iskeyword(value):
                parts.append(f"{port!r}: {module_var}.nets.{value}")
            else:
                # Python keywords such as in and and, and non-identifiers,
                # cannot follow a dot even when they are legal circuit names.
                parts.append(f"{port!r}: {module_var}.nets[{value!r}]")
        else:
            parts.append(f"{port!r}: {value!r}")
    return '{' + ', '.join(parts) + '}'


def _emit_submodule_imports(
    spec: 'CircuitSpec',
    lines: list,
) -> 'dict[tuple[str, str], str]':
    """Append `from … import … as …` lines for spec.submodules.

    Mutates ``lines``. Returns the alias map so callers can re-use it
    when emitting the corresponding ``add_element`` calls.

    Emits nothing when spec.submodules is empty.
    """
    domain_bindings = _emitted_voltage_domain_names(spec) | _per_copy_nets_bindings(spec)
    aliases = _compute_submodule_aliases(
        spec.submodules,
        additional_reserved=domain_bindings,
    )
    if not aliases:
        return aliases
    seen_keys: list = []
    for sm in spec.submodules:
        key = (sm.import_path, sm.module_var)
        if key in seen_keys:
            continue
        seen_keys.append(key)
        alias = aliases[key]
        lines.append(
            f'from {sm.import_path} import {sm.module_var} as {alias}'
        )
    return aliases


def _emit_component_sections(
    spec: 'CircuitSpec',
    lines: list,
    module_var: str,
) -> None:
    """Append the PDK-device, ideal-element and source sections.

    Every component is emitted inline as ``<module_var>.add(<factory call>)``
    (arrays: ``<module_var>.add(*[<factory call> for i in range(N)])``), so an
    instance name is only ever a string argument and never a Python variable
    — a device named ``pdk`` or ``mod`` cannot shadow the generator's own
    names.
    """
    def add(call: str, array: bool = False) -> None:
        lines.append(f"{module_var}.add({'*' if array else ''}{call})")

    if spec.devices:
        lines.append('# --- PDK devices ---')
        for dev in spec.devices:
            if isinstance(dev.nets, list):
                _emit_per_copy_device(dev, lines, 'pdk', module_var)
            elif dev.array is None:
                add(_device_call(dev, 'pdk'))
            else:
                add(_device_array_call(dev, 'pdk'), array=True)
        lines.append('')
    if spec.ideal_elements:
        lines.append('# --- ideal elements ---')
        for elem in spec.ideal_elements:
            if isinstance(elem.nets, list):
                _emit_per_copy_device(elem, lines, 'elements', module_var)
            elif elem.array is None:
                add(_device_call(elem, 'elements'))
            else:
                add(_device_array_call(elem, 'elements'), array=True)
        lines.append('')
    if spec.sources:
        lines.append('# --- sources ---')
        for src in spec.sources:
            if src.type in _CONTROLLED_SOURCE_PINS:      # TASK-253
                add(_controlled_source_call(src, spec.sources), array=src.array is not None)
            elif src.type in _BEHAVIORAL_SOURCE_TYPES:   # TASK-282.2
                add(_behavioral_source_call(src), array=src.array is not None)
            elif src.array is None:
                add(_source_call(src))
            else:
                add(_source_array_call(src), array=True)
        lines.append('')


def _emit_submodule_instances(
    spec: 'CircuitSpec',
    lines: list,
    aliases: 'dict[tuple[str, str], str]',
    module_var: str,
) -> None:
    """Append `# --- submodule instances ---` section and placement calls.

    Mutates ``lines``. Emits nothing when spec.submodules is empty.
    For entries with ``sm.array`` set, emits a one-line ``for i in range(...)``
    that registers all N instances under bracketed or underscore-suffixed
    names. Net values with ``$i`` placeholders become f-string fragments
    inside the comprehension.

    Emits ``add_module(...)``. An entry with ``sm.parameters``
    still emits ``add_element(..., parameters=...)`` — ``add_module`` has
    no ``parameters`` argument, and this form already raises at run time
    for a Module child; see docs/known_issues.md for this limitation.

    ``module_var`` is the Python variable name for the parent Module
    (``'mod'`` in subcircuit mode, ``'ckt'`` in standalone mode).
    """
    if not spec.submodules:
        return
    lines.append('# --- submodule instances ---')
    for sm in spec.submodules:
        instance = require_python_identifier(
            sm.instance_name, "submodule instance name"
        )
        alias = aliases[(sm.import_path, sm.module_var)]
        method = 'add_element' if sm.parameters else 'add_module'
        if sm.array is None:
            # Existing flat emission
            nets_literal = _format_submodule_nets_dict(
                sm.nets, spec, module_var=module_var
            )
            call = (
                f"{module_var}.{method}({alias}, {sm.instance_name!r}, "
                f"nets={nets_literal}"
            )
            if sm.parameters:
                call += f", parameters={sm.parameters!r}"
            call += ')'
            lines.append(call)
        else:
            # Arrayed emission — one-line for loop (no binding)
            range_lit = (f'range({sm.array})' if isinstance(sm.array, int)
                         else f'range({sm.array[0]}, {sm.array[1]})')
            inst_template = (
                repr(f"{instance}[{{i}}]") if sm.naming == 'bracketed'
                else repr(f"{instance}_{{i}}")
            )
            inst_fstring = 'f' + inst_template
            if isinstance(sm.nets, list):
                nets_literal = _emit_per_copy_nets(
                    sm, lines, lambda n: _format_submodule_nets_dict(
                        n, spec, module_var=module_var))
            else:
                nets_literal = _format_array_nets_dict(
                    sm.nets, spec, module_var=module_var
                )
            call = (
                f'for i in {range_lit}: '
                f'{module_var}.{method}({alias}, {inst_fstring}, '
                f'nets={nets_literal}'
            )
            if sm.parameters:
                call += f", parameters={sm.parameters!r}"
            call += ')'
            lines.append(call)
    lines.append('')


_PORT_NAMING_DEFAULT = 'bracketed'

_PORTDIR_LITERAL = {
    'input':  'PortDir.IN',
    'output': 'PortDir.OUT',
    'inout':  'PortDir.INOUT',
    'power':  'PortDir.POWER',
    'ground': 'PortDir.GROUND',
}

_PORTROLE_LITERAL = {
    'analog':  'PortRole.ANALOG',
    'digital': 'PortRole.DIGITAL',
    'clock':   'PortRole.CLOCK',
    'power':   'PortRole.POWER',
    'ground':  'PortRole.GROUND',
}


def _bits_literal(bits) -> str:
    """Render an int, range, or slice as Python source."""
    if isinstance(bits, int):
        return str(bits)
    if isinstance(bits, range):
        if bits.step != 1:
            raise ValueError(
                f"range bits must have step=1, got step={bits.step}"
            )
        return f"range({bits.start}, {bits.stop})"
    if isinstance(bits, slice):
        step = bits.step if bits.step is not None else 1
        if step != 1:
            raise ValueError(
                f"slice bits must have step=1 or None, got step={bits.step}"
            )
        if bits.stop is None:
            raise ValueError("slice bits must have explicit stop")
        start = bits.start if bits.start is not None else 0
        return f"slice({start}, {bits.stop})"
    raise TypeError(
        f"Port.bits must be int, range, or slice, got {type(bits).__name__}"
    )


def _bus_net_line(bn) -> str:
    """Render one `mod.add_net(...)` line for an internal bus net."""
    parts = [f"{bn.name!r}", f"bits={_bits_literal(bn.bits)}"]
    if bn.naming != _PORT_NAMING_DEFAULT:
        parts.append(f"naming={bn.naming!r}")
    return f"mod.add_net({', '.join(parts)})"


def _port_to_python(port, bindings) -> str:
    """Emit the shortest legal Python literal for one Port.

    - Bare PortDir literal when only `dir` is set.
    - (PortDir, PortRole) 2-tuple when `dir` and `role` are set and
      nothing else.
    - Full Port(...) constructor when any of `domain`, `bits`, or
      non-default `naming` is set.
    """
    dir_lit = _PORTDIR_LITERAL[port.dir.value]
    has_role = port.role is not None
    has_domain = port.domain is not None
    has_bits = port.bits is not None
    has_naming = port.naming != _PORT_NAMING_DEFAULT

    if not (has_role or has_domain or has_bits or has_naming):
        return dir_lit

    role_lit = _PORTROLE_LITERAL[port.role.value] if has_role else None

    if has_role and not (has_domain or has_bits or has_naming):
        return f"({dir_lit}, {role_lit})"

    parts = [f"dir={dir_lit}"]
    if has_role:
        parts.append(f"role={role_lit}")
    if has_domain:
        parts.append(f"domain={bindings[port.domain.name]}")
    if has_bits:
        parts.append(f"bits={_bits_literal(port.bits)}")
    if has_naming:
        parts.append(f"naming={port.naming!r}")
    return f"Port({', '.join(parts)})"


def _emit_per_copy_device(dev, lines, source, module_var):
    """Emit one wiring list and loop for a device or ideal element."""
    instance = require_python_identifier(dev.instance, "device instance name")
    dev_type = require_python_identifier(dev.type, "device type")
    for pin in dev.nets[0]:
        require_python_identifier(pin, f"entry {instance!r} net key")
    nets = _emit_per_copy_nets(dev, lines, repr)
    range_lit = (f'range({dev.array})' if isinstance(dev.array, int)
                 else f'range({dev.array[0]}, {dev.array[1]})')
    name = f'{instance}[{{i}}]' if dev.naming == 'bracketed' else f'{instance}_{{i}}'
    factory = f'{source}.{dev_type}'
    params = ', '.join(f"{require_python_identifier(k, 'device parameter key')}={_fmt_value(v)}"
                       for k, v in dev.params.items())
    args = ', '.join(filter(None, [params, f'**{nets}']))
    lines.append(f'for i in {range_lit}: {module_var}.add({factory}(f{name!r}, {args}))')


def _device_call(dev: DeviceSpec, source: str) -> str:
    """The factory call expression for one device."""
    instance = require_python_identifier(dev.instance, "device instance name")
    dev_type = require_python_identifier(dev.type, "device type")
    net_args = ', '.join(
        f"{require_python_identifier(k, 'device net key')}={v!r}"
        for k, v in dev.nets.items()
    )
    param_args = ', '.join(
        f"{require_python_identifier(k, 'device parameter key')}={_fmt_value(v)}"
        for k, v in dev.params.items()
    )
    all_args = ', '.join(filter(None, [net_args, param_args]))
    return f"{source}.{dev_type}({instance!r}, {all_args})"


def _device_array_call(dev: DeviceSpec, source: str) -> str:
    """Emit a list-comprehension for an arrayed device entry.

    ``source`` is the factory namespace ``'pdk'`` or ``'elements'``.
    Returns the bare list-comprehension expression, e.g.::

        [pdk.nmos1p8(f'M[{i}]', d=f'vd[{i}]', g='vin', ...) for i in range(8)]
    """
    instance = require_python_identifier(dev.instance, "device instance name")
    dev_type = require_python_identifier(dev.type, "device type")
    range_lit = (f'range({dev.array})' if isinstance(dev.array, int)
                 else f'range({dev.array[0]}, {dev.array[1]})')
    inst_template = (
        repr(f"{instance}[{{i}}]") if dev.naming == 'bracketed'
        else repr(f"{instance}_{{i}}")
    )
    inst_fstring = 'f' + inst_template

    # Net args — rewrite placeholders, choose f-string vs plain.
    net_args = []
    for k, v in dev.nets.items():
        key = require_python_identifier(k, "device net key")
        rewritten, has = _rewrite_template(v)
        if has:
            net_args.append(f"{key}=f{rewritten!r}")
        else:
            net_args.append(f"{key}={v!r}")

    # Params are broadcast; no template support today.
    param_args = ', '.join(
        f"{require_python_identifier(k, 'device parameter key')}={_fmt_value(v)}"
        for k, v in dev.params.items()
    )

    all_args = ', '.join(filter(None, [', '.join(net_args), param_args]))
    return (
        f'[{source}.{dev_type}({inst_fstring}, {all_args}) '
        f'for i in {range_lit}]'
    )


@functools.lru_cache(maxsize=1)
def _source_meta():
    """Load source_metadata.yaml once (cached)."""
    from silk.sources.schema import _load_metadata
    return _load_metadata()


def _waveform_kind(class_name: str) -> str:
    """Waveform class name -> kwarg kind, reflected from source_metadata.yaml
    (e.g. 'Pulse' -> 'pulse'). Falls back to lower-case for unregistered kinds."""
    inv = {v: k for k, v in _source_meta()['waveforms'].items()}
    return inv.get(class_name, class_name.lower())


def _source_alias(type_name: str) -> str:
    """'VoltageSource' -> 'v', 'CurrentSource' -> 'i' (from metadata alias)."""
    return _source_meta()['sources'][type_name]['alias']


def _source_call_body(alias: str, term: str, src: 'SourceSpec') -> tuple[str, str]:
    """Compute the shared ``(head, body)`` for a source factory call.

    ``term`` is the already-formatted ``positive=..., negative=...``
    fragment — its literal-vs-f-string form differs between the scalar and
    arrayed emitters, so callers compute it themselves and pass it in.
    ``head`` is the factory attribute (``sources.<alias>{dc|kind|pwl}``);
    ``body`` is the comma-joined argument list that follows the
    instance-name token (``term`` plus waveform/pwl/scalar args). Shared by
    :func:`_source_call` and :func:`_source_array_call` so the dc/ac/kind
    branch logic — the exact place two divergence bugs (``value=``/``dc=``
    mismatch, double-``dc=``) crept in — exists in one place.
    """
    _validate_source_waveforms(src)
    alias = require_python_identifier(alias, "source alias")
    scalars = []
    if src.dc != 0:
        scalars.append(f'dc={_fmt_value(src.dc)}')
    if src.ac != 0:
        scalars.append(f'ac={_fmt_value(src.ac)}')
    if src.waveform is not None:
        wf_type = src.waveform.get('type', '')
        if not isinstance(wf_type, str):
            raise ValueError(
                f"source waveform type {wf_type!r} must be a string"
            )
        kind = require_python_identifier(
            _waveform_kind(wf_type), "source waveform kind"
        )
        if 'params' in src.waveform:
            flat_keys = sorted(k for k in src.waveform if k not in ('type', 'params'))
            if flat_keys:
                raise ValueError(
                    "source waveform dict mixes the {'type', 'params': {...}} "
                    f"form with flat keyword(s) {flat_keys!r} beside 'type' — "
                    "use either {'type': ..., 'params': {...}} or the legacy "
                    "{'type': ..., <kwargs...>} form, not both"
                )
            params = src.waveform['params']
            if not isinstance(params, dict):
                raise ValueError(
                    f"source waveform 'params' must be a dict, got "
                    f"{type(params).__name__}"
                )
            wf_items = params.items()
        else:
            wf_items = ((k, v) for k, v in src.waveform.items() if k != 'type')
        wf_args = ', '.join(
            f"{require_python_identifier(k, 'source waveform parameter key')}="
            f"{_fmt_value(v)}"
            for k, v in wf_items
        )
        head = f'sources.{alias}{kind}'
        body = ', '.join(filter(None, [term, wf_args, ', '.join(scalars)]))
        return head, body
    if src.pwl is not None:
        pairs = ', '.join(_fmt_value(v) for v in src.pwl)
        head = f'sources.{alias}pwl'
        body = ', '.join(filter(None, [term, f'points=[{pairs}]', ', '.join(scalars)]))
        return head, body
    # pure dc / ac (no waveform): dc becomes the positional `value=`; keep ac
    # if set — including `dc=` from scalars too would double up the same
    # value under two kwargs and the scalar ctor rejects it.
    head = f'sources.{alias}dc'
    ac_only = [s for s in scalars if s.startswith('ac=')]
    body = ', '.join([term, f'value={_fmt_value(src.dc)}'] + ac_only)
    return head, body


def _source_call(src: 'SourceSpec') -> str:
    """The declarative factory call expression for a placed source."""
    instance = require_python_identifier(src.instance, "source instance name")
    alias = _source_alias(src.type)
    term = f"positive={src.positive!r}, negative={src.negative!r}"
    head, body = _source_call_body(alias, term, src)
    return f"{head}({instance!r}, {body})"


def _source_array_call(src: 'SourceSpec') -> str:
    """The list-comprehension expression for an arrayed source entry;
    shared with testbench codegen."""
    instance = require_python_identifier(src.instance, "source instance name")
    alias = _source_alias(src.type)
    range_lit = (f'range({src.array})' if isinstance(src.array, int)
                 else f'range({src.array[0]}, {src.array[1]})')
    inst_template = (repr(f"{instance}[{{i}}]") if src.naming == 'bracketed'
                     else repr(f"{instance}_{{i}}"))
    inst_fstring = 'f' + inst_template
    pos_re, pos_has = _rewrite_template(src.positive)
    neg_re, neg_has = _rewrite_template(src.negative)
    term = (('positive=' + (f'f{pos_re!r}' if pos_has else repr(src.positive))) +
            ', negative=' + (f'f{neg_re!r}' if neg_has else repr(src.negative)))
    head, body = _source_call_body(alias, term, src)
    return (
        f'[{head}({inst_fstring}, {body}) '
        f'for i in {range_lit}]'
    )


def _controlled_source_call(src: 'SourceSpec', source_specs=()) -> str:
    """Compute the ``sources.<ClassName>(...)`` call expression for one
    dependent (controlled) source — VCVS/VCCS/CCVS/CCCS.

    Unlike :func:`_source_call_body`, this does not go through the
    declarative alias surface: controlled sources have no dc/ac/waveform
    kinds, so ``silk.sources.library.build_source_surfaces`` never builds
    one for them (no ``quantity``/``alias`` in ``source_metadata.yaml``).
    All arguments are passed by keyword — the exact form shown in the
    classes' own docstrings (``silk/sources/classes.py``) — so constructor
    argument order does not matter here.

    Returns the bare call expression (no assignment) so both
    the circuit emitters (``<ckt>.add(<call>)``) and
    ``testbench_generator._source_add_line`` (``tb.add(<call>)``) share the
    one place that knows each class's pins/scalar/ctrl_vname —
    ``_CONTROLLED_SOURCE_PINS`` et al. stay defined only here.
    """
    instance = require_python_identifier(src.instance, "source instance name")
    arrayed = src.array is not None
    name = (f"{instance}[{{i}}]" if src.naming == 'bracketed'
            else f"{instance}_{{i}}")
    args = ['f' + repr(name) if arrayed else repr(instance)]
    for pin in _CONTROLLED_SOURCE_PINS[src.type]:
        value = getattr(src, pin)
        rewritten, has = _rewrite_template(value) if arrayed else (value, False)
        literal = 'f' + repr(rewritten) if has else repr(value)
        args.append(f'{pin}={literal}')
    if src.type in _CONTROLLED_SOURCE_HAS_CTRL_VNAME:
        target = next((s for s in source_specs if s.instance == src.ctrl_vname), None)
        control = repr(src.ctrl_vname)
        if arrayed and target is not None and target.array is not None:
            name = (f"{target.instance}[{{i}}]" if target.naming == 'bracketed'
                    else f"{target.instance}_{{i}}")
            control = 'f' + repr(name)
        args.append(f'ctrl_vname={control}')
    scalar = _CONTROLLED_SOURCE_SCALAR[src.type]
    args.append(f'{scalar}={_fmt_value(getattr(src, scalar))}')
    call = f"sources.{src.type}({', '.join(args)})"
    if arrayed:
        lo, hi = _source_index_range(src)
        return f'[{call} for i in range({lo}, {hi})]'
    return call


def _behavioral_source_call(src: 'SourceSpec') -> str:
    """Compute the ``sources.<ClassName>(...)`` call expression for one
    behavioral source — BehavioralVoltage/BehavioralCurrent.

    Like :func:`_controlled_source_call`, this does not go through the
    declarative alias surface: behavioral sources have no dc/ac/waveform
    kinds and no entry in ``source_metadata.yaml``'s ``sources:`` map.
    Returns the bare call expression; see :func:`_controlled_source_call`
    for why. An arrayed source returns a list comprehension over
    the index range; ``positive``, ``negative`` and ``expression`` have
    their index templates rewritten per element (braces such as
    ``{param}`` survive; an expression with no template is unchanged).
    """
    instance = require_python_identifier(src.instance, "source instance name")
    arrayed = src.array is not None
    name = (f"{instance}[{{i}}]" if src.naming == 'bracketed'
            else f"{instance}_{{i}}")
    args = ['f' + repr(name) if arrayed else repr(instance)]
    for field_name in _BEHAVIORAL_SOURCE_FIELDS:
        value = getattr(src, field_name)
        rewritten, has = _rewrite_template(value) if arrayed else (value, False)
        literal = 'f' + repr(rewritten) if has else repr(value)
        args.append(f'{field_name}={literal}')
    call = f"sources.{src.type}({', '.join(args)})"
    if arrayed:
        lo, hi = _source_index_range(src)
        return f'[{call} for i in range({lo}, {hi})]'
    return call


class Generator:
    """Generate a runnable ``Module`` Python script from a circuit description.

    The spec may be a :class:`CircuitSpec` or a plain circuit-description dict.
    Without a ``ports`` key the output is a standalone simulation script;
    with ``ports`` (even ``{}``) it is a reusable ``Module`` definition. The
    format is described in ``docs/codegen-contract.md``.

    Parameters
    ----------
    indent : int, default 4
        Stored as ``indent``; the current emitter does not use it.

    Attributes
    ----------
    indent : int
        The indentation width given at construction.

    Examples
    --------
    >>> from silk import Generator
    >>> code = Generator().generate({'circuit_name': 'demo', 'pdk': 'sky130'})
    >>> compile(code, '<generated>', 'exec') is not None
    True
    """

    def __init__(self, indent: int = 4):
        self.indent = indent

    def generate(self, spec: 'dict | CircuitSpec') -> str:
        """Generate Python source code from a circuit spec.

        When ``spec.ports`` is not ``None`` (a non-empty map, or an explicit
        ``{}`` for a zero-pin block), emits a ``Module`` (subcircuit)
        definition. When the spec has no ``ports`` key, emits a standalone
        ``Module`` simulation script. Nothing is simulated and no PDK model
        library is needed, but the PDK name must be registered.

        Parameters
        ----------
        spec : dict or CircuitSpec
            Circuit description; a dict is parsed with
            :meth:`CircuitSpec.from_dict`.

        Returns
        -------
        str
            Python source code.

        Raises
        ------
        KeyError
            A required key is missing from the description.
        TypeError
            An entry has the wrong type.
        ValueError
            The description violates the contract: an invalid ``circuit_name``,
            ``corner``, port or instance name, a bad variable reference or
            simulation, or an unregistered PDK name (a ``ValueError``
            subclass).
        """
        if isinstance(spec, dict):
            spec = CircuitSpec.from_dict(spec)
        _validate_identity_fields(spec)
        for entry in spec.ideal_elements:
            _validate_entry_type({'type': entry.type, 'instance': entry.instance},
                                 'ideal_elements')
        for entry in [*spec.devices, *spec.ideal_elements, *spec.submodules]:
            name = entry.instance_name if isinstance(entry, SubmoduleSpec) else entry.instance
            _validate_per_copy_nets(
                entry.nets, entry.array, f"entry {name!r}",
                allow_bit_pins=isinstance(entry, SubmoduleSpec),
            )
        _validate_instance_bindings(spec)
        _validate_sweep_source(spec)
        _validate_variables(spec)
        if spec.ports is not None:
            return self._generate_module(spec)
        return self._generate_netlist(spec)

    def _generate_netlist(self, spec: 'CircuitSpec') -> str:
        """Generate a standalone Module simulation script."""

        lines = []

        # ------------------------------------------------------------------
        # Imports
        # ------------------------------------------------------------------
        # Header fields pass through _docstring_safe so no input can close
        # the triple-quoted docstring; byte-identical for contract-valid
        # values (see _validate_identity_fields, which runs first).
        lines.append(f'"""Generated by silk CircuitCodeGenerator')
        lines.append(f'Circuit: {_docstring_safe(spec.circuit_name)}')
        lines.append(f'PDK:     {_docstring_safe(spec.pdk)}')
        lines.append(f'Corner:  {_docstring_safe(spec.corner)}')
        lines.append('"""')
        lines.append('')
        lines.append('from silk.circuit.modules import Module')
        lines.append('from silk.simulation import analysis')

        if spec.ideal_elements:
            lines.append('from silk.elements import elements')
        if spec.sources:
            lines.append('from silk import sources')
        if spec.variables:
            lines.append('from silk import Variable')
        submodule_aliases = _emit_submodule_imports(spec, lines)

        lines.append('')

        # ------------------------------------------------------------------
        # PDK setup
        # ------------------------------------------------------------------
        _emit_pdk_setup(spec, lines)
        _emit_variables(spec, lines)

        # ------------------------------------------------------------------
        # Circuit object
        # ------------------------------------------------------------------
        fname = _sanitize_filename(spec.circuit_name) + '.cir'
        lines.append('# --- circuit ---')
        # spec.pdk is registry-validated by _emit_pdk_setup above; the repr
        # is still applied so a hostile name can never break out of the
        # literal. fname is [\w-]-only by _sanitize_filename; repr of both
        # is byte-identical to plain single-quoting for well-formed specs.
        lines.append(
            f"ckt = Module({('PDK_' + spec.pdk)!r}, {spec.output_dir!r}, "
            f"{fname!r}, pdk_details=pdk)"
        )
        lines.append('')

        if spec.bus_nets:
            lines.append('# --- internal bus nets ---')
            for bn in spec.bus_nets:
                # _bus_net_line emits `mod.add_net(...)`. In standalone
                # the Module variable is `ckt`.
                lines.append(_bus_net_line(bn).replace('mod.add_net', 'ckt.add_net', 1))
            lines.append('')

        _emit_component_sections(spec, lines, module_var='ckt')

        _emit_submodule_instances(spec, lines, submodule_aliases, module_var='ckt')

        # ------------------------------------------------------------------
        # Simulation
        # ------------------------------------------------------------------
        lines.append('# --- simulation ---')
        if spec.simulation:
            lines.append(_sim_code(spec.simulation))
        # corner is contract-validated ([A-Za-z0-9_-]+) by
        # _validate_identity_fields; repr is belt-and-braces and renders
        # byte-identically to the former hand-quoting for valid names.
        lines.append(f"ckt.simulate(corner={spec.corner!r})")
        lines.append('')

        # ------------------------------------------------------------------
        # Results loading stub
        # ------------------------------------------------------------------
        lines.append('# --- load results ---')
        lines.append('# ckt.get_signal_names()            # list available signals')
        lines.append("# sig = ckt.load_signal('v(out)')  # load a waveform")
        lines.append("# sig.plot()                      # plot it")
        lines.append('')

        return '\n'.join(lines)

    def _generate_module(self, spec: 'CircuitSpec') -> str:
        """Generate a Module (subcircuit) definition when spec.ports is not None."""
        lines = []
        lines.append(
            f'"""Generated subcircuit: {_docstring_safe(spec.circuit_name)}"""'
        )
        lines.append('')
        lines.append('from silk import Module')
        port_imports = ['PortDir']
        any_role = any(p.role is not None for p in spec.ports.values())
        any_full = any(
            (p.domain is not None or p.bits is not None
             or p.naming != _PORT_NAMING_DEFAULT)
            for p in spec.ports.values()
        )
        if any_role:
            port_imports.append('PortRole')
        if any_full:
            port_imports.append('Port')
        lines.append(
            'from silk.circuit.circuit_elements import '
            + ', '.join(port_imports)
        )
        if spec.voltage_domains:
            lines.append(
                'from silk.circuit.voltage_domain import VoltageDomain'
            )
        if spec.ideal_elements:
            lines.append('from silk.elements import elements')
        if spec.sources:
            lines.append('from silk import sources')
        if spec.variables:
            lines.append('from silk import Variable')
        submodule_aliases = _emit_submodule_imports(spec, lines)
        domain_bindings = _compute_voltage_domain_bindings(spec)
        lines.append('')
        _emit_pdk_setup(spec, lines)
        if spec.voltage_domains:
            lines.append('# --- voltage domains ---')
            for dom in spec.voltage_domains.values():
                lines.append(_voltage_domain_line(dom, domain_bindings))
            lines.append('')
        _emit_variables(spec, lines)
        fname = _sanitize_filename(spec.circuit_name) + '.cir'
        lines.append('# --- module definition ---')
        # Same escaping rationale as the standalone branch: spec.pdk is
        # registry-validated by _emit_pdk_setup, fname is [\w-]-only, and
        # repr keeps well-formed output byte-identical.
        lines.append(
            f"mod = Module({('PDK_' + spec.pdk)!r}, {spec.output_dir!r}, "
            f"{fname!r}, pdk_details=pdk,"
        )
        # circuit_name is identifier-validated by _validate_identity_fields;
        # repr keeps it a literal regardless and is byte-identical for
        # identifiers.
        lines.append(f"             module_name={spec.circuit_name!r})")
        # Port names are identifier-validated by _validate_identity_fields
        # (called earlier in generate(), covering dict input and direct
        # CircuitSpec construction alike); repr() is belt-and-braces here
        # and renders byte-identically for identifier-shaped names (TASK-324).
        port_items = ', '.join(
            f"{name!r}: {_port_to_python(port, domain_bindings)}" for name, port in spec.ports.items()
        )
        lines.append(f"mod.define_module_ports({{{port_items}}})")
        lines.append('')
        if spec.bus_nets:
            lines.append('# --- internal bus nets ---')
            for bn in spec.bus_nets:
                lines.append(_bus_net_line(bn))
            lines.append('')
        _emit_component_sections(spec, lines, module_var='mod')
        _emit_submodule_instances(spec, lines, submodule_aliases, module_var='mod')
        return '\n'.join(lines)

    def generate_to_file(self, spec: dict | CircuitSpec, path: str) -> None:
        """Write generated code to a file.

        Parameters
        ----------
        spec : dict or CircuitSpec
            Circuit description.
        path : str
            Output file path (e.g. ``'my_circuit.py'``). Overwritten if it
            exists; it is written only after generation succeeds.

        Raises
        ------
        KeyError, TypeError, ValueError
            As for :meth:`generate`.
        OSError
            The file cannot be written.
        """
        code = self.generate(spec)
        with open(path, 'w') as f:
            f.write(code)

    def generate_testbenches(self, spec) -> str:
        """Generate testbench Python code from a TestbenchSpec.

        Parameters
        ----------
        spec : dict or TestbenchSpec
            Testbench specification.

        Returns
        -------
        str
            Generated testbench Python code.

        Raises
        ------
        ValueError
            The spec is structurally invalid. The exception raised is
            ``TestbenchSpecError``, a ``ValueError`` subclass defined in
            ``silk.codegen.testbench_generator``. The generator does not
            emit uncompilable code.
        """
        from silk.codegen.testbench_generator import generate_testbenches
        return generate_testbenches(spec)

    def generate_testbenches_to_file(self, spec, path: str) -> None:
        """Write generated testbench code to a file.

        Parameters
        ----------
        spec : dict or TestbenchSpec
            Testbench specification.
        path : str
            Output file path (e.g. ``'my_testbenches.py'``). Overwritten if
            it exists; it is written only after generation succeeds.

        Raises
        ------
        ValueError
            As for :meth:`generate_testbenches`.
        OSError
            The file cannot be written.
        """
        code = self.generate_testbenches(spec)
        with open(path, 'w') as f:
            f.write(code)


# ---------------------------------------------------------------------------
# CLI:  python -m silk.codegen.generator  circuit_spec.json
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Generate silk Python code from a circuit spec JSON file.')
    parser.add_argument('spec', help='Path to circuit spec JSON file')
    parser.add_argument('-o', '--output', default='-',
                        help='Output .py file (default: stdout)')
    args = parser.parse_args()

    spec = CircuitSpec.from_json_file(args.spec)
    gen  = Generator()
    code = gen.generate(spec)

    if args.output == '-':
        print(code)
    else:
        gen.generate_to_file(spec, args.output)
        print(f'Code written to {args.output}', file=sys.stderr)
