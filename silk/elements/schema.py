"""Device schema export — machine-readable device specs for tools that inspect devices or generate silk code.

Usage
-----
From a PDK::

    from silk.pdk import load_pdk
    from silk import export_schema
    import json
    from silk._json_codec import encode_json

    pdk = load_pdk('sky130')
    schema = export_schema(pdk)
    print(json.dumps(encode_json(schema), allow_nan=False, indent=2))

From the ideal element library::

    from silk.elements import elements
    from silk import export_schema
    import json
    from silk._json_codec import encode_json

    schema = export_schema(elements)
    print(json.dumps(encode_json(schema), allow_nan=False, indent=2))

From the command line (writes to stdout)::

    python -m silk.elements.schema --pdk sky130
    python -m silk.elements.schema --pdk path/to/device_details.yaml
    python -m silk.elements.schema --elements

Schema format
-------------
The top-level dict maps device names to device spec dicts::

    {
      "nmos1p8": {
        "name": "nmos1p8",
        "is_pdk_device": true,
        "pdk_model": "sky130_fd_pr__nfet_01v8",
        "spice_prefix": "XM",
        "category": "transistor",
        "pins": [
          {"name": "d", "description": "drain"},
          ...
        ],
        "parameters": [
          {"name": "w", "default": "0.42", "unit": "u", "min": "0.42", "max": "-1"},
          ...
        ]
      },
      ...
    }

The ``category`` field is set to ``"transistor"`` / ``"resistor"`` / etc.
based on the YAML ``category`` field (ideal elements) or inferred from the
device name prefix for PDK devices.

Code-generation hint
-------------------------
Given a device spec, a code generator can produce::

    mn1 = pdk.nmos1p8('mn1', d='vout', g='vin', s='gnd', b='gnd',
                       w=2.0, l=0.15, m=1)
    ckt.add(mn1)
"""

import json
from typing import Union

from silk._json_codec import encode_json
from silk.elements.base import AnalogElement


# ---------------------------------------------------------------------------
# Per-class schema
# ---------------------------------------------------------------------------

def _source_catalog_fields(source_class: str, pins) -> dict:
    """Fold the rich source schema into device-catalog shape for a source element.

    Returns dc+ac params (device shape), a waveforms block with DECLARATIVE
    per-kind code_templates, waveform_exclusive, and a declarative scalar
    code_template. Reflected from source_schema — a new waveform kind flows in
    with no edit here.
    """
    from silk.sources.schema import source_schema, _load_metadata
    meta = _load_metadata()
    alias = meta['sources'][source_class]['alias']          # 'v' | 'i'
    ss = source_schema(source_class, meta)

    pin_names = [p['name'] for p in pins]                    # ['p', 'n']
    pin_args = ', '.join(f'{n}={{net_{n}}}' for n in pin_names)

    # dc/ac in device param shape {name, default, unit, min, max}.
    parameters = [
        {'name': p['name'], 'default': p['default'], 'unit': p['unit'],
         'min': None, 'max': None}
        for p in ss['parameters']
    ]
    # scalar ctor (sources.vdc/idc): the primary scalar (dc) fills the ctor's
    # `value=` slot; the rest (ac) keep their own name. (The ctor also accepts
    # `dc=` as an alias for `value=`, but the template uses `value=` uniformly.)
    scalar_arg_parts = [
        (f"value={{param_{p['name']}}}" if i == 0 else f"{p['name']}={{param_{p['name']}}}")
        for i, p in enumerate(ss['parameters'])
    ]
    scalar_args = ', '.join(scalar_arg_parts)
    scalar_ct = (f"{{instance_name}} = sources.{alias}dc('{{instance_name}}', "
                 f"{', '.join(filter(None, [pin_args, scalar_args]))})")

    # Declarative per-kind templates: sources.v<kind>(inst, pins, subparams, dc=, ac=)
    # — the waveform ctor accepts dc=/ac= as optional keyword offsets (unlike
    # the scalar ctor above), so keep those names here.
    waveform_scalar_args = ', '.join(f"{p['name']}={{param_{p['name']}}}" for p in ss['parameters'])
    waveforms = {}
    for kind, spec in ss['waveforms'].items():
        sub_args = ', '.join(f"{c['name']}={{param_{c['name']}}}" for c in spec['parameters'])
        ct = (f"{{instance_name}} = sources.{alias}{kind}('{{instance_name}}', "
              f"{', '.join(filter(None, [pin_args, sub_args, waveform_scalar_args]))})")
        waveforms[kind] = {**spec, 'code_template': ct}

    return {
        'symbol': ss.get('symbol'),
        'parameters': parameters,
        'waveforms': waveforms,
        'waveform_exclusive': ss.get('waveform_exclusive', True),
        'code_template': scalar_ct,
    }


def device_schema(cls) -> dict:
    """Return a JSON-serialisable schema dict for a single device class.

    Parameters
    ----------
    cls : type
        An :class:`AnalogElement` subclass (e.g. ``pdk.nmos1p8``).

    Returns
    -------
    dict
        Keys: ``name``, ``is_pdk_device``, ``pdk_model``, ``spice_prefix``,
        ``category``, ``pins``, ``parameters``, ``code_template``,
        ``source_class``. Source entries (those with a truthy
        ``source_class``) additionally carry ``symbol``, ``waveforms``, and
        ``waveform_exclusive``; for these, the base ``parameters`` and
        ``code_template`` are overridden — ``parameters`` becomes the
        ``dc``/``ac`` list (from ``source_metadata.yaml``) and
        ``code_template`` becomes the declarative scalar-ctor form.
    """
    if not (isinstance(cls, type) and issubclass(cls, AnalogElement)):
        raise TypeError(f"Expected an AnalogElement subclass, got {cls!r}")

    pins = []
    for pin_name in cls._pin_names:
        desc = cls._pin_descriptions.get(pin_name, pin_name) \
               if hasattr(cls, '_pin_descriptions') else pin_name
        pins.append({"name": pin_name, "description": desc})

    parameters = []
    for param_name in cls._param_names:
        constraint = cls._param_constraints.get(param_name)
        entry = {"name": param_name}
        if constraint:
            entry["default"] = constraint.default
            entry["unit"]    = constraint.pdk_unit
            entry["min"]     = constraint.minval
            entry["max"]     = constraint.maxval
        else:
            entry["default"] = None
            entry["unit"]    = None
            entry["min"]     = None
            entry["max"]     = None
        parameters.append(entry)

    category = getattr(cls, '_category', None) or _infer_category(cls)

    # Code generation template — shows consumers the exact Python line to emit.
    # Placeholders use {{name}} syntax (double-braces = literal { } in f-string).
    def _pin_placeholder(p):
        n = p["name"]
        return f'{n}={{net_{n}}}'

    def _param_placeholder(p):
        n = p["name"]
        return f'{n}={{param_{n}}}'

    pin_args   = ', '.join(_pin_placeholder(p) for p in pins)
    param_args = ', '.join(_param_placeholder(p) for p in parameters)
    all_args   = ', '.join(filter(None, [pin_args, param_args]))
    source     = 'pdk' if cls._is_pdk_device else 'elements'
    code_template = (
        "{instance_name} = " + source + "." + cls._device_type_name +
        "('{instance_name}', " + all_args + ")"
    )

    result = {
        "name":           cls._device_type_name,
        "is_pdk_device":  cls._is_pdk_device,
        "pdk_model":      cls._pdk_device or None,
        "spice_prefix":   cls._spice_prefix,
        "category":       category,
        "pins":           pins,
        "parameters":     parameters,
        "code_template":  code_template,
        "source_class":   getattr(cls, '_source_class', None),
    }
    if result["source_class"]:
        result.update(_source_catalog_fields(result["source_class"], pins))
    return result


def _infer_category(cls) -> str:
    """Guess a category string from the device name when not set in YAML."""
    name = cls._device_type_name.lower()
    if 'nmos' in name or 'pmos' in name or 'fet' in name or 'bjt' in name:
        return 'transistor'
    if 'res' in name or name.startswith('r'):
        return 'passive'
    if 'cap' in name or name.startswith('c'):
        return 'passive'
    if 'ind' in name or name.startswith('l'):
        return 'passive'
    if 'vsource' in name or 'isource' in name:
        return 'source'
    if 'diode' in name:
        return 'diode'
    return 'other'


# ---------------------------------------------------------------------------
# Module schema
# ---------------------------------------------------------------------------

def module_schema(module) -> dict:
    """Return a JSON-serialisable schema dict for a Module (subcircuit).

    Parameters
    ----------
    module : Module
        A Module instance with module_name set and define_module_ports() called.

    Returns
    -------
    dict
        Keys: ``name``, ``type``, ``is_pdk_device``, ``pdk_model``,
        ``spice_prefix``, ``category``, ``pins``, ``devices``,
        ``code_template``.

    Examples
    --------
    ::

        import json
        from silk._json_codec import encode_json
        from silk.circuit.circuit_elements import PortDir
        ldo = Module(..., module_name='ldo')
        ldo.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT, 'gnd': None})
        schema = module_schema(ldo)
        print(json.dumps(encode_json(schema), allow_nan=False, indent=2))
    """
    name = module.module_name or 'unnamed_module'

    pins = []
    if module.pin_list:
        for pin_name, pin in module.pin_list.items():
            direction = getattr(pin, 'direction', None)
            pins.append({"name": pin_name, "direction": direction})

    devices = list(module.elem_dict.keys())

    # Emit a runnable, non-deprecated snippet: define_module_ports with typed
    # PortDir values. (repr() of the PortDir str-enum produces invalid
    # `<PortDir.IN: 'input'>` source, and define_module_nets is deprecated and
    # drops directions — both fixed here.)
    from silk.circuit.circuit_elements import PortDir

    def _portdir_member(direction):
        if direction is None:
            return 'INOUT'
        return PortDir(direction).name

    ports_items = ', '.join(
        f"{p['name']!r}: PortDir.{_portdir_member(p['direction'])}" for p in pins
    )
    code_template = (
        "from silk.circuit.circuit_elements import PortDir\n"
        f"mod = Module(..., module_name='{name}')\n"
        f"mod.define_module_ports({{{ports_items}}})"
    )

    return {
        "name":          name,
        "type":          "module",
        "is_pdk_device": False,
        "pdk_model":     None,
        "spice_prefix":  "X",
        "category":      "module",
        "pins":          pins,
        "devices":       devices,
        "code_template": code_template,
    }


# ---------------------------------------------------------------------------
# Library-level schema export
# ---------------------------------------------------------------------------

def export_schema(source, include_metadata: bool = True) -> dict:
    """Export a JSON-serialisable schema for all devices in a PDK or library.

    Parameters
    ----------
    source : BasePDK or ElementLibrary
        A loaded PDK object (``BasePDK``) or ``ElementLibrary``.
        Any object that exposes AnalogElement subclasses as attributes works.
    include_metadata : bool
        If ``True`` (default), include a ``_meta`` key with source info.

    Returns
    -------
    dict
        Maps device names to their schema dicts. Top-level ``_meta`` key
        contains source information when ``include_metadata=True``.

    Examples
    --------
    >>> from silk import export_schema, load_pdk
    >>> schema = export_schema(load_pdk('sky130'), include_metadata=False)
    >>> 'nmos1p8' in schema
    True
    """
    # Check if source is a Module instance
    try:
        from silk.circuit.modules import Module as _Module
        if isinstance(source, _Module):
            name = source.module_name or 'unnamed_module'
            result = {name: module_schema(source)}
            if include_metadata:
                result['_meta'] = {
                    'source_type': 'Module',
                    'device_count': 1,
                    'format_version': '1.0',
                }
            return result
    except ImportError:
        pass

    result = {}

    # Include schema-only source entries from an ideal-element catalog.
    from silk.elements.library import ElementLibrary
    catalog = source._device_classes if isinstance(source, ElementLibrary) else None
    device_names = catalog if catalog is not None else (
        getattr(source, 'device_names', None) or getattr(source, 'model_names', []))

    for name in device_names:
        cls = catalog[name] if catalog is not None else getattr(source, name, None)
        if cls is None or not (isinstance(cls, type) and issubclass(cls, AnalogElement)):
            continue
        result[name] = device_schema(cls)

    if include_metadata and getattr(source, 'corners', None):
        result['_library'] = {
            'pdk_name': getattr(source, 'pdk_name', None),
            'default_corner': getattr(source, 'default_corner', None),
            'corners': source.corners,
        }

    if include_metadata:
        result['_meta'] = {
            'source_type': type(source).__name__,
            'device_count': sum(1 for k in result if not k.startswith('_')),
            'format_version': '1.0',
        }

    return result


def export_schema_json(source, indent: int = 2, **kwargs) -> str:
    """Return the schema as a formatted JSON string.

    Parameters
    ----------
    source : BasePDK or ElementLibrary
        See :func:`export_schema`.
    indent : int
        JSON indentation level (default 2).
    """
    return json.dumps(encode_json(export_schema(source, **kwargs)), allow_nan=False, indent=indent)


# ---------------------------------------------------------------------------
# CLI entry point:  python -m silk.elements.schema [options]
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import sys
    import os

    parser = argparse.ArgumentParser(
        description='Export silk device schema as JSON.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--pdk',
        metavar='NAME_OR_YAML_PATH',
        help='A registered PDK name (e.g. sky130) or a device-details YAML file path',
    )
    group.add_argument(
        '--elements',
        action='store_true',
        help='Export the ideal element library (analog_devices.yaml)',
    )
    parser.add_argument(
        '--output', '-o',
        metavar='FILE',
        default='-',
        help='Output file path (default: stdout)',
    )
    parser.add_argument(
        '--indent',
        type=int,
        default=2,
        help='JSON indentation (default: 2)',
    )
    args = parser.parse_args()

    if args.pdk:
        from silk.pdk.basepdk import BasePDK
        from silk.pdk import load_pdk
        if (os.path.isfile(args.pdk) or os.path.dirname(args.pdk)
                or args.pdk.lower().endswith(('.yaml', '.yml'))):
            source = BasePDK(args.pdk, library_location='')
        else:
            source = load_pdk(args.pdk)
    else:
        from silk.elements import elements as source  # noqa: F811

    schema_json = export_schema_json(source, indent=args.indent)

    if args.output == '-':
        print(schema_json)
    else:
        with open(args.output, 'w') as f:
            f.write(schema_json)
        print(f'Schema written to {args.output}', file=sys.stderr)
