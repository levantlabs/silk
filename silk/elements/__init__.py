"""Ideal analog elements and the machinery that generates device classes.

- :class:`ElementLibrary` builds one class per entry of a device catalog YAML;
  the bundled catalog holds the process-independent ``resistor``,
  ``capacitor`` and ``inductor`` factories, plus source metadata for schema export.
- :data:`elements` is the ready-made library built from the bundled catalog.
- :class:`AnalogElement` is the base class of every generated device class,
  including the PDK device classes returned by :func:`silk.pdk.load_pdk`.
- :mod:`~silk.elements.passives` holds the deprecated ``Resistor``,
  ``Capacitor`` and ``Inductor`` constructors that take ``pos``, ``neg`` and a
  value positionally.

The generated classes are described on the factory page of the reference.
"""
from silk.elements.library import ElementLibrary
from silk.elements.base import AnalogElement

# Built before importing passives (TASK-299): passives.py's Resistor/
# Capacitor/Inductor now subclass elements.resistor/capacitor/inductor, so
# ``elements`` must already exist as an attribute of this package by the time
# ``from silk.elements import elements`` runs inside passives.py.
#: The :class:`ElementLibrary` built from the bundled ``analog_devices.yaml``:
#: ``elements.resistor``, ``elements.capacitor`` and ``elements.inductor``.
#: Source catalog entries are available through schema export; construct sources
#: with :mod:`silk.sources`. Created at import time.
elements = ElementLibrary()

from silk.elements import passives
from silk import sources

__all__ = ["ElementLibrary", "AnalogElement", "elements", "sources", "passives"]
