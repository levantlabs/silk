"""Process design kits: manifest registry, loader and device catalogs.

A PDK is registered by a manifest (``silk/pdk/data/<name>.pdk.yaml``) that
names its model library, corners, units and device-details catalog.

- :func:`load_pdk` loads a registered PDK by name; :func:`available` lists the
  registered names and :func:`manifest` returns one parsed manifest.
- :class:`BasePDK` holds the loaded catalog; each
  catalog device is a generated class reachable as an attribute.
- :class:`PDKDevice`, :class:`Pin` and :class:`ParameterConstraint` are the
  catalog records the legacy ``add_element`` path uses.
- :func:`device_geometry_um` reads a device's width and length in micrometres.
- :py:data:`UNIT_PDK` and :py:data:`UNIT_SI` are the ``user_units`` modes.
- :class:`PDKRegistryError` reports an unknown or malformed manifest.

Importing the package needs neither installed models nor a simulator.
"""
from silk.pdk.basepdk import (BasePDK, PDKDevice, Pin, ParameterConstraint, UNIT_PDK,
                              UNIT_SI, device_geometry_um)
from silk.pdk._loader import load_pdk
from silk.pdk.registry import available, manifest, PDKRegistryError

__all__ = [
    "BasePDK",
    "PDKDevice",
    "Pin",
    "ParameterConstraint",
    "device_geometry_um",
    "UNIT_PDK",
    "UNIT_SI",
    "load_pdk",
    "available",
    "manifest",
    "PDKRegistryError",
]
