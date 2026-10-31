"""Public PDK loader — thin front door over silk.pdk.registry."""
import importlib
import os

from silk.pdk.basepdk import (  # noqa: F401
    UNIT_PDK, UNIT_SI, _SI_TO_PDK_SCALE)
from silk import settings as _settings
from silk.pdk import registry as _registry
from silk.pdk.statistical import parse_statistical

# 'pdk' = the author writes the PDK's own unit; 'si' = SI base units; any
# prefix token ('u', 'n', 'p', 'm', 'base') = the author writes THAT unit,
# whatever the PDK wants. See BasePDK._param_scales.
_UNIT = {'pdk': UNIT_PDK, 'si': UNIT_SI}
_UNIT.update({k: k for k in _SI_TO_PDK_SCALE})


def _resolve(path, root):
    return path if os.path.isabs(path) else os.path.join(root, path)


def load_pdk(name, *, pdk_root=None, library_location=None, corner=None,
             user_units=None):
    """Load a named PDK from its manifest.

    Reads the manifest registered under *name* (or one of its aliases) from
    ``silk/pdk/data/*.pdk.yaml``, resolves the model-library paths against
    the PDK root, and returns an instance of the manifest's PDK class with its
    device catalog loaded. The device classes are then attributes of the
    result (``pdk.nmos1p8``); ``pdk.model_names`` lists them.

    Nothing is checked on disk at load time: the catalog, units, corners and
    manifest ship with silk, so a PDK whose models are not installed still
    loads, generates netlists and supports behavioural simulation. A
    simulation that needs the models raises ``SimulationError`` naming the
    missing library path.

    Parameters
    ----------
    name : str
        A canonical PDK name or alias, such as ``'sky130'`` or
        ``'skywater130'``; see :func:`~silk.pdk.available`.
    pdk_root : str, optional
        Directory the manifest's relative library paths resolve against.
        Defaults to the ``PDK_ROOT`` setting (environment variable, then the
        user's settings file, then ``~/pdk``). Absolute manifest paths are
        used as they are.
    library_location : str, optional
        Replaces the resolved path of the main model library only; the extra
        includes and corner include files still resolve against the PDK root.
    corner : str, optional
        Default corner. Defaults to the manifest's ``default_corner``. It is
        not checked against the PDK's corners at load time.
    user_units : str, optional
        The unit the circuit author writes in: ``'pdk'`` (the PDK's own unit),
        ``'si'``, or a prefix token (``'base'``, ``'u'``, ``'n'``, ``'p'``,
        ``'m'``). Defaults to the manifest's ``user_units``, which every shipped
        manifest sets to ``'u'`` (micrometres for W/L); a manifest that sets
        none means ``'pdk'``, and the resulting PDK prints a notice to
        standard error.

    Returns
    -------
    BasePDK
        An instance of the manifest's PDK class (``BasePDK`` for all
        shipped PDKs).

    Raises
    ------
    silk.pdk.PDKRegistryError
        *name* is not a registered PDK, or a manifest is malformed or
        duplicates a name. It is a ``ValueError``.
    ValueError
        *user_units* is not ``'pdk'``, ``'si'`` or a known prefix.
    silk.pdk.basepdk.ValidationError
        The device catalog does not parse or lacks a required field.
    silk.exceptions.PDKError
        The catalog defines a device twice or uses an invalid Python name.

    See Also
    --------
    available : The registered PDK names.
    manifest : The parsed manifest of one PDK.

    Examples
    --------
    >>> from silk.pdk import load_pdk
    >>> pdk = load_pdk('sky130')  # catalog access needs no installed models
    >>> device = pdk.nmos1p8('m1', d='out', g='in', s='gnd', b='gnd', w=4.0, l=0.5)
    >>> device.name
    'm1'
    """
    if corner is not None:
        # It becomes default_corner: the .lib section and the corner folder
        # of every run that names no corner (TASK-593).
        from silk.utils.deck_text import validate_corner_name
        validate_corner_name(corner, who='load_pdk(corner=)')
    m = _registry.manifest(name)
    lib_conf = m['library']
    root = pdk_root or _settings.get('PDK_ROOT')

    # The lib path is RESOLVED here, never checked here. Everything load_pdk
    # returns except that one path — the device catalog, the units, the
    # corners, the manifest — ships inside silk, so a PDK that is not
    # installed still loads usefully: behavioral simulation, schema export,
    # and (since TASK-71) SPICE decks that instantiate no PDK device at all.
    # Stat-ing at load time made all three impossible on a machine with no
    # PDK download, which is the whole point of the pip + ngspice on-ramp.
    #
    # The check moved to the one place it is actually true — running a deck
    # that references the models (silk.circuit.modules_sim._require_pdk_library)
    # — matching the contract silk already has for OSDI plugins: emission
    # never requires the files, the run does, and fails loudly with the fix.
    lib = library_location or _resolve(lib_conf['lib_path'], root)

    yaml_path = os.path.join(m['_manifest_dir'], m['device_details'])
    extra = [_resolve(p, root) for p in lib_conf.get('extra_includes', [])]

    # Xyce-native twins of lib_path/extra_includes (optional, generic — any
    # PDK may declare them). `None` when the manifest omits them, so
    # XyceNetlister's fallback to the ngspice-oriented fields above is
    # unambiguous (an empty list would mean "no extra includes on Xyce",
    # which is a different statement).
    xyce_lib = (_resolve(lib_conf['xyce_lib_path'], root)
                if 'xyce_lib_path' in lib_conf else None)
    xyce_extra = ([_resolve(p, root) for p in lib_conf['xyce_extra_includes']]
                  if 'xyce_extra_includes' in lib_conf else None)

    # Per-corner include lists (optional). A corner that declares `includes:`
    # is emitted as one `.lib "<file>" <section>` line PER ENTRY instead of the
    # single `.lib "<lib_path>" <corner>` line. Paths resolve against the PDK
    # root here, so the netlister never has to know where the root is.
    #
    # `xyce_includes` is the same shape, one sibling key, used by
    # XyceNetlister in place of `includes` when present (see
    # silk/circuit/xyce_netlister.py).
    corners = {}
    for cname, meta in lib_conf['corners'].items():
        meta = dict(meta or {})
        if meta.get('includes'):
            meta['includes'] = [
                {**inc, 'file': _resolve(inc['file'], root)}
                for inc in meta['includes']
            ]
        if meta.get('xyce_includes'):
            meta['xyce_includes'] = [
                {**inc, 'file': _resolve(inc['file'], root)}
                for inc in meta['xyce_includes']
            ]
        corners[cname] = meta
    if user_units:
        try:
            umode = _UNIT[user_units]
        except KeyError:
            raise ValueError(
                f"unknown user_units {user_units!r}; use 'pdk', 'si', or a "
                f"prefix the author writes in: {sorted(_SI_TO_PDK_SCALE)}") from None
    else:
        umode = _UNIT.get(m.get('user_units', 'pdk'))

    mod_name, cls_name = m.get('class', 'silk.pdk.basepdk:BasePDK').split(':')
    cls = getattr(importlib.import_module(mod_name), cls_name)

    # Compiled Verilog-A models (optional). Declaring them here is stating a
    # FACT about the PDK — its device models are OSDI plugins — not asking
    # silk to build anything. Compilation (openvaf + the PDK's install.py) is
    # environment setup, owned by the user or pdksetup, exactly like ngspice.
    osdi = [_resolve(p, root) for p in lib_conf.get('osdi', [])]

    # Monte Carlo facts (optional); None when the manifest declares none.
    statistical = (
        parse_statistical(lib_conf['statistical'], root, m['name'])
        if lib_conf.get('statistical') else None)

    return cls(yaml_path, lib, user_units=umode,
               corners=corners,
               default_corner=corner or lib_conf['default_corner'],
               extra_includes=extra, pdk_name=m['name'], osdi_files=osdi,
               xyce_library_location=xyce_lib, xyce_extra_includes=xyce_extra,
               statistical=statistical)
