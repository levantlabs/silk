"""Run-local voltage publication and per-pin current bookkeeping.

Numeric port assignments use the evaluation grid and per-sample write masks.
Reads return copies, never writable aliases of the stored quantities.
"""
from dataclasses import dataclass
from math import isfinite
from sys import float_info, maxsize

import numpy as np

from silk.exceptions import ConflictError, KCLError, RewriteError

# Below this bound, even the largest possible pin list cannot overflow a
# float64 sum. Exceptional nets keep the original whole-grid diagnostics.
_SCALAR_KCL_LIMIT = float_info.max / (2 * maxsize)


def _label(path):
    return str(path[-1] if isinstance(path, tuple) else path)


def _register_writer(writers, path):
    """Return the stable writer number, registering its label on first use."""
    ids, names = writers
    if path not in ids:
        number = len(ids) + 1
        ids[path] = number
        names[number] = _label(path)
    return ids[path]


@dataclass
class _Quantity:
    value: object
    assigned: np.ndarray
    writers: np.ndarray
    latest: int = -1


class SharedState:
    """Electrical values for one transient grid or one DC point.

    ``pins`` maps ``(placement_path, port_name)`` to the resolved net. Only
    modeled pins and independent current-source terminals participate in KCL;
    voltage equations and wire adapters publish by net without adding currents.
    ``signals`` retains the engine's existing result-net inventory.
    """

    def __init__(self, size, pins, signals=None, *, writers=None):
        self.size = size
        self.pins = dict(pins)
        self.signals = signals if signals is not None else {}
        self.initial_values = {}
        self.read_events = set()
        self._voltages = {}
        self._currents = {}
        self._net_pins = {}
        self._filled = {}
        self._kcl_currents = {}
        self._general_kcl = set()
        # Identities are circuit metadata; ownership masks remain state-local.
        self._writer_ids, self._writer_names = ({}, {}) if writers is None else writers
        for pin, net in self.pins.items():
            self._currents[pin] = self._quantity()
            self._net_pins.setdefault(net, []).append(pin)

    def _quantity(self):
        return _Quantity(np.zeros(self.size), np.zeros(self.size, dtype=bool),
                         np.zeros(self.size, dtype=np.intp))

    def _voltage(self, net):
        if net not in self._voltages:
            quantity = self._quantity()
            if net in self.signals:
                quantity.value = np.array(np.broadcast_to(
                    np.asarray(self.signals[net], dtype=float), (self.size,)), copy=True)
            self._voltages[net] = quantity
        return self._voltages[net]

    def _writer(self, path):
        return _register_writer((self._writer_ids, self._writer_names), path)

    def _check(self, quantity, writer, index, label, *, voltage=False):
        assigned = quantity.assigned[index]
        owners = quantity.writers[index]
        if voltage:
            conflicts = np.asarray(owners)[assigned & (owners != writer)]
            if conflicts.size:
                first = self._writer_names[int(conflicts.flat[0])]
                second = self._writer_names[writer]
                raise ConflictError(
                    f"Voltage on net {label!r} is written by both models "
                    f"{first!r} and {second!r}.")
        if np.any(assigned):
            raise RewriteError(
                f"Model {self._writer_names[writer]!r} assigned {label!r} "
                "more than once at the same sample. Compute in a local value "
                "and assign it to the port once.")

    def publish_voltage(self, net, path, value):
        """Publish an internal source/wire equation using the voltage write rules."""
        quantity = self._voltage(net)
        self._write(quantity, path, value, slice(None), net, voltage=True)
        self.signals[net] = quantity.value

    def read_voltage(self, path, port, index=slice(None)):
        net = self.pins[path, port]
        quantity = self._voltage(net)
        return np.array(np.broadcast_to(quantity.value, (self.size,))[index], copy=True)

    def write_voltage(self, path, port, value, index=slice(None)):
        net = self.pins[path, port]
        quantity = self._voltage(net)
        self._write(quantity, path, value, index, net, voltage=True)
        self.signals[net] = quantity.value

    def write_current(self, path, port, value, index=slice(None)):
        quantity = self._currents[path, port]
        self._write(quantity, path, value, index,
                    f'{path}.{port} current')
        published = quantity.value[index]
        if np.ndim(published) == 0:
            scalar = float(published)
            exceptional = not isfinite(scalar) or abs(scalar) > _SCALAR_KCL_LIMIT
        else:
            exceptional = (np.any(~np.isfinite(published))
                           or np.any(np.abs(published) > _SCALAR_KCL_LIMIT))
        if exceptional:
            self._general_kcl.add(self.pins[path, port])

    def _write(self, quantity, path, value, index, label, *, voltage=False):
        writer = self._writer(path)
        if isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool_)):
            self._write_sample(quantity, writer, value, index, label, voltage=voltage)
            return
        self._check(quantity, writer, index, label, voltage=voltage)
        # Validate and copy before changing any sample or its publication flag.
        values = np.array(np.broadcast_to(np.asarray(value, dtype=float),
                                         quantity.assigned[index].shape), copy=True)
        indices = np.arange(self.size)[index]
        if np.size(indices):
            quantity.latest = max(quantity.latest, int(np.max(indices)))
        quantity.value[index] = values
        quantity.assigned[index] = True
        quantity.writers[index] = writer

    def _write_sample(self, quantity, writer, value, index, label, *, voltage=False):
        if quantity.assigned[index]:
            if voltage and quantity.writers[index] != writer:
                first = self._writer_names[int(quantity.writers[index])]
                raise ConflictError(
                    f"Voltage on net {label!r} is written by both models "
                    f"{first!r} and {self._writer_names[writer]!r}.")
            raise RewriteError(
                f"Model {self._writer_names[writer]!r} assigned {label!r} "
                "more than once at the same sample. Compute in a local value "
                "and assign it to the port once.")
        # The uncommon conversion path retains NumPy's scalar shape errors,
        # including refusal of a length-one array. Validate before publication.
        if type(value) in (float, int, np.float64):
            values = float(value)
        else:
            values = np.array(np.broadcast_to(np.asarray(value, dtype=float), ()), copy=True)
        quantity.latest = max(quantity.latest, int(index) % self.size)
        quantity.value[index] = values
        quantity.assigned[index] = True
        quantity.writers[index] = writer

    def read_current(self, path, port, index=slice(None)):
        """Fill a requested unset pin with KCL, at most one pin per net/sample."""
        pin = (path, port)
        net = self.pins[pin]
        quantity = self._currents[pin]
        if (isinstance(index, (int, np.integer))
                and not isinstance(index, (bool, np.bool_)) and net not in self._general_kcl):
            if not quantity.assigned[index]:
                filled = self._filled.get(net)
                if filled is None:
                    filled = self._filled[net] = np.zeros(self.size, dtype=np.intp)
                if filled[index]:
                    first = self._writer_names[int(filled[index])]
                    raise KCLError(
                        f"Net {net!r} needs a second unknown pin current: "
                        f"{first!r} and {path + '.' + port!r}. "
                        "Assign the other pin currents before reading; KCL can "
                        "fill only one unknown current per net and sample.")
                # A scalar-shaped array keeps the array ufunc diagnostics
                # (including warnings promoted to errors) of vector KCL.
                total = np.zeros(())
                for other in self._net_pins[net]:
                    if other != pin:
                        current = self._currents[other]
                        total += current.value[index] if current.assigned[index] else 0
                self.write_current(path, port, np.negative(total)[()], index=index)
                derived = self._kcl_currents.get(pin)
                if derived is None:
                    derived = self._kcl_currents[pin] = np.zeros(self.size, dtype=bool)
                derived[index] = True
                filled[index] = self._writer(('pin', path + '.' + port))
            return np.array(quantity.value[index], copy=True)
        requested = np.zeros(self.size, dtype=bool)
        requested[index] = True
        missing = requested & ~quantity.assigned
        if np.any(missing):
            filled = self._filled.setdefault(net, np.zeros(self.size, dtype=np.intp))
            if np.any(filled[missing]):
                first = self._writer_names[int(filled[missing][filled[missing] != 0][0])]
                raise KCLError(
                    f"Net {net!r} needs a second unknown pin current: "
                    f"{first!r} and {path + '.' + port!r}. "
                    "Assign the other pin currents before reading; KCL can "
                    "fill only one unknown current per net and sample.")
            total = np.zeros(self.size)
            for other in self._net_pins[net]:
                if other != pin:
                    current = self._currents[other]
                    total += np.where(current.assigned, current.value, 0)
            self.write_current(path, port, -total[missing], index=missing)
            derived = self._kcl_currents.setdefault(pin, np.zeros(self.size, dtype=bool))
            derived[missing] = True
            filled[missing] = self._writer(('pin', path + '.' + port))
        return np.array(quantity.value[index], copy=True)
