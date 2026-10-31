"""ElementLibrary — loads ideal analog elements from analog_devices.yaml."""

import os
import yaml

from silk.elements.factory import make_device_class
from silk.elements._catalog import _validate_device_names
from silk.exceptions import CircuitError

_DEFAULT_YAML = os.path.join(os.path.dirname(__file__), 'analog_devices.yaml')


class ElementLibrary:
    """Loads ideal analog element classes from a YAML file.

    After construction, each non-source catalog entry is an attribute holding
    a generated subclass of :class:`~silk.elements.AnalogElement`::

        from silk.elements import elements
        r1 = elements.resistor('r1', p='vout', n='gnd', value=1e3)

    Every generated class takes the instance name as its only positional
    argument; pins and parameters are keyword-only. A missing pin or
    parameter without a default raises ``TypeError``, as does an unexpected
    keyword. The bundled catalog's elements also accept alternate spellings
    (``pos``/``neg`` for ``p``/``n``, and ``resistance``/``res``,
    ``capacitance``/``cap``, ``inductance``/``ind`` for ``value``); passing a
    name and its alias together raises ``ValueError``. Instantiating inside a
    ``with module:`` block adds the element to that module.

    Source catalog entries are retained for :func:`silk.circuit.export_schema`;
    construct sources with ``silk.sources`` (``sources.vdc`` / ``sources.idc``).

    Parameters
    ----------
    yaml_path : str, optional
        Path to the device YAML file.  Defaults to the bundled
        ``analog_devices.yaml``.

    Attributes
    ----------
    device_names : list of str
        Catalog entry names, in file order; each is also an attribute holding
        the generated class.

    Raises
    ------
    OSError
        The YAML file cannot be opened.
    CircuitError
        An entry name, pin name or parameter name is not a valid Python
        identifier.

    Examples
    --------
    >>> from silk.elements import ElementLibrary
    >>> library = ElementLibrary()
    >>> r = library.resistor('r1', p='out', n='gnd', value=1000)
    >>> r.name
    'r1'
    """

    def __init__(self, yaml_path=None):
        if yaml_path is None:
            yaml_path = _DEFAULT_YAML

        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        self._device_classes = {}
        self.device_names = []

        for device_name, spec in config.items():
            _validate_device_names(device_name, spec, yaml_path, CircuitError)
            cls = make_device_class(device_name, spec, is_pdk=False)
            self._device_classes[device_name] = cls
            if not spec.get('source_class'):
                setattr(self, device_name, cls)
                self.device_names.append(device_name)

    def __repr__(self):
        return f'ElementLibrary({self.device_names})'
