"""Discovery + parsing of per-PDK manifests (``*.pdk.yaml``)."""
import glob
import os

import yaml


class PDKRegistryError(ValueError):
    """Raised for an unknown, duplicate or malformed PDK manifest.

    Raised by :func:`~silk.pdk.manifest` and :func:`~silk.pdk.load_pdk` for an
    unregistered name (the message lists the available names), and, on the first
    use of the registry, for a manifest that cannot be parsed, has no ``name``,
    has a non-list ``aliases``, or reuses a name or alias of another manifest.

    It subclasses ``ValueError``, so ``except ValueError`` still catches it, as
    the historical ``load_pdk`` contract for an unknown name requires.
    """


_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
_CACHE = None  # {name_or_alias: parsed manifest dict}


def _scan():
    reg = {}
    origin = {}   # key -> the manifest path it was first registered from
    for path in sorted(glob.glob(os.path.join(_DATA_DIR, '*.pdk.yaml'))):
        try:
            with open(path) as f:
                m = yaml.safe_load(f)
        except (OSError, yaml.YAMLError) as e:
            raise PDKRegistryError(f'malformed manifest {path}: {e}')
        if not isinstance(m, dict) or 'name' not in m:
            raise PDKRegistryError(f'manifest {path} missing required "name"')
        m['_manifest_dir'] = os.path.dirname(os.path.abspath(path))
        aliases = m.get('aliases', [])
        if not isinstance(aliases, list):
            raise PDKRegistryError(
                f'manifest {path}: "aliases" must be a list, got {type(aliases).__name__}')
        for key in [m['name'], *aliases]:
            if key in reg and origin[key] != path:
                raise PDKRegistryError(
                    f'duplicate PDK name/alias {key!r} ({path} vs {origin[key]})')
            reg[key] = m
            origin[key] = path
    return reg


def _registry():
    global _CACHE
    if _CACHE is None:
        _CACHE = _scan()
    return _CACHE


def available():
    """Return the canonical PDK names, sorted.

    Returns
    -------
    list of str
        One name per registered manifest in ``silk/pdk/data``, without
        aliases. A name listed here has a shipped catalog; it does not mean
        its real-model simulation is release-validated.

    Raises
    ------
    PDKRegistryError
        A manifest is malformed or duplicates a name (the registry is scanned
        on first use).

    Examples
    --------
    >>> from silk.pdk import available
    >>> available()
    ['gf180mcu', 'sg13g2', 'sky130']
    """
    return sorted({m['name'] for m in _registry().values()})


def manifest(name):
    """Return the parsed manifest of a registered PDK.

    Parameters
    ----------
    name : str
        A canonical PDK name or one of its aliases.

    Returns
    -------
    dict
        The manifest as parsed from the ``*.pdk.yaml`` file, plus the key
        ``_manifest_dir`` holding the directory it was read from. This is the
        registry's own cached dict, not a copy: do not modify it.

    Raises
    ------
    PDKRegistryError
        *name* is not registered (the message lists the available names), or a
        manifest is malformed.

    Examples
    --------
    >>> from silk.pdk import manifest
    >>> manifest('skywater130')['name']
    'sky130'
    """
    reg = _registry()
    if name not in reg:
        raise PDKRegistryError(
            f'unknown PDK {name!r}; available: {available()}')
    return reg[name]
