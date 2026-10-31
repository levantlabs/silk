"""Electrical port adapters and live parameter access for behavioral models."""

from numbers import Integral, Real

import numpy as np

from silk.simulation._analysis_context import analysis
from silk.exceptions import SimulationError


def _validate_initial(value, path, port):
    """Validate real numeric inputs without coercing the stored initial."""
    # NumPy durations inherit from integer but are not real electrical values.
    if (isinstance(value, (Real, np.bool_))
            and not isinstance(value, np.timedelta64)):
        return
    try:
        if np.asarray(value).dtype.kind in 'biuf':
            return
    except Exception:
        pass
    raise TypeError(
        f"Model {path!r}, port {port!r}: initial_value must be real numeric, got {value!r}")


class ParamsView:
    """Read live parameter entries by attribute or bracket, without dict methods."""

    __slots__ = ('_params',)

    def __init__(self, params):
        object.__setattr__(self, '_params', {} if params is None else params)

    def __getitem__(self, name):
        return object.__getattribute__(self, '_params')[name]

    def __iter__(self):
        return iter(object.__getattribute__(self, '_params'))

    def __len__(self):
        return len(object.__getattribute__(self, '_params'))

    def __getattribute__(self, name):
        if name.startswith('__'):
            return object.__getattribute__(self, name)
        params = object.__getattribute__(self, '_params')
        try:
            return params[name]
        except KeyError:
            raise AttributeError(
                f"params has no entry {name!r}; entries: {sorted(map(str, params))}. "
                f"Read an arbitrary name with params[{name!r}].") from None

    def __repr__(self):
        return f"ParamsView({object.__getattribute__(self, '_params')!r})"


def _prepare_port_metadata(module):
    """Cache declared pins and real bus specs with expanded names for one placement."""
    return (frozenset(module.pin_list or {}),
            {name: (spec, tuple(spec.expanded_names()))
             for name, spec in getattr(module, '_buses', {}).items()})


def bus_bit(spec, names, k):
    """Return the pin name of declared bit number *k* (``code[8]`` for ``range(8, 11)``)."""
    from silk.circuit.declared_indices import select_positions
    if isinstance(k, bool) or not isinstance(k, Integral):
        raise TypeError(
            f"bus {spec.name!r} bit subscript must be a declared bit number, got {k!r}")
    (position,) = select_positions(spec.indices, k)
    return names[position]


def stack_bus(spec, names, values, *, execution):
    """Stack a bus's bits in ``expanded_names()`` order into one read-only array.

    *values* maps each bit's pin name to its array (``execution='vector'``,
    giving ``(width, T)``) or its scalar (``'sample'``, giving ``(width,)``).
    Each bit is fetched only here, when the bus is actually read. A width-one
    bus keeps its bus axis: ``(1, T)`` or ``(1,)``.
    """
    if execution == 'vector':
        stacked = np.stack([np.asarray(values[name], dtype=float) for name in names])
    else:
        stacked = np.array([float(values[name]) for name in names])
    stacked.setflags(write=False)
    return stacked


def split_bus(spec, names, value, *, execution, n):
    """Split an assigned bus array into one value per bit pin, checking its shape.

    ``execution='vector'`` needs shape ``(width, n)``; ``'sample'`` needs
    ``(width,)``. Anything else raises :class:`ValueError` naming the bus.
    """
    array = np.asarray(value, dtype=float)
    expected = (spec.width, n) if execution == 'vector' else (spec.width,)
    if array.shape != expected:
        raise ValueError(
            f"bus {spec.name!r} has {spec.width} bit(s): assign an array of shape "
            f"{expected} (one row per bit, in declared order), got shape {array.shape}")
    return {name: array[k].copy() if execution == 'vector' else float(array[k])
            for k, name in enumerate(names)}


class _ValueView(np.lib.mixins.NDArrayOperatorsMixin):
    """Array/scalar operations read copies; indexed assignment publishes."""

    def __array__(self, dtype=None, copy=None):
        return np.array(self._read(), dtype=dtype, copy=True)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = tuple(np.asarray(x) if isinstance(x, _ValueView) else x for x in inputs)
        outputs = kwargs.pop('out', None)
        if outputs is not None and any(isinstance(x, _ValueView) for x in outputs):
            if method != '__call__':
                return NotImplemented
            values = getattr(ufunc, method)(*inputs, **kwargs)
            values = values if isinstance(values, tuple) else (values,)
            for target, value in zip(outputs, values):
                if isinstance(target, _ValueView):
                    target._write(value)
                elif target is not None:
                    target[...] = value
            return outputs[0] if len(outputs) == 1 else outputs
        if outputs is not None:
            kwargs['out'] = outputs
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __len__(self):
        return len(self._read())

    def __iter__(self):
        return iter(self._read())

    def __getitem__(self, index):
        return self._read(index)

    def __setitem__(self, index, value):
        self._write(value, index)

    def __float__(self):
        return float(self._read())

    def __bool__(self):
        return bool(self._read())

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def copy(self):
        return np.array(self)

    def __repr__(self):
        return repr(self._read())


class _PinValue(_ValueView):
    def __init__(self, state, path, port, quantity, index):
        self._state, self._path, self._port = state, path, port
        self._quantity, self._index = quantity, index

    @property
    def shape(self):
        return (self._state.size,) if isinstance(self._index, slice) else ()

    @property
    def initial_value(self):
        return self._state.initial_values.get((self._path, self._port, self._quantity), 0)

    @initial_value.setter
    def initial_value(self, value):
        _validate_initial(value, self._path, self._port)
        self._state.initial_values[self._path, self._port, self._quantity] = value

    @property
    def previous_value(self):
        return _DelayedValue(self, 1)

    def _read(self, index=slice(None)):
        return self._read_delayed(index, 0)

    def _read_delayed(self, index, delay):
        state = self._state
        initial = 0 if analysis() == 'dc' else self.initial_value
        if self._quantity == 'v':
            state.read_events.add((self._path, self._port, self._quantity, delay))
        vector = isinstance(self._index, slice)
        if not vector:
            target = self._index - delay
            if target < 0:
                value = initial
            elif self._quantity == 'i':
                pin = (self._path, self._port)
                derived = state._kcl_currents.get(pin)
                if (not state._currents[pin].assigned[target]
                        or (derived is not None and derived[target])):
                    state.read_events.add((self._path, self._port, 'i', delay))
                value = state.read_current(self._path, self._port, target)
            else:
                net = state.pins[self._path, self._port]
                quantity = state._voltage(net)
                if not delay and not quantity.assigned[target]:
                    if quantity.latest <= target:
                        target = quantity.latest
                    else:
                        available = np.flatnonzero(quantity.assigned[:target + 1])
                        target = int(available[-1]) if available.size else -1
                if delay or (target >= 0 and quantity.assigned[target]):
                    value = quantity.value[target]
                else:
                    value = 0 if net in ('gnd', '0') else initial
            # Read/KCL side effects precede scalar subscript validation.
            if not (isinstance(index, slice) and index == slice(None)):
                value = np.asarray([value]).reshape(())[index]
                return np.float64(value) if np.ndim(value) == 0 else value
            if np.isscalar(value):
                return np.float64(value)
            return np.float64(np.asarray([value]).reshape(())[()])
        selected = np.arange(state.size)[index]
        selected = np.asarray(selected)
        if self._quantity == 'v':
            quantity = state._voltage(state.pins[self._path, self._port])
            targets = selected - delay
            clipped = np.clip(targets, 0, state.size - 1)
            values = np.broadcast_to(quantity.value, (state.size,))[clipped]
            grounded = state.pins[self._path, self._port] in ('gnd', '0')
            available = (targets >= 0) & (quantity.assigned[clipped] | grounded | (delay > 0))
            return np.array(np.where(available, values, initial), copy=True)
        if self._quantity == 'i':
            targets = selected - delay
            valid = targets >= 0
            values = np.full(selected.shape, initial, dtype=float)
            pin = (self._path, self._port)
            if np.any(valid):
                derived = state._kcl_currents.get(pin)
                if (np.any(~state._currents[pin].assigned[targets[valid]])
                        or (derived is not None and np.any(derived[targets[valid]]))):
                    state.read_events.add((self._path, self._port, 'i', delay))
                values[valid] = state.read_current(self._path, self._port, targets[valid])
            return values[()] if values.ndim == 0 else values

    def _write(self, value, index=slice(None)):
        write = (self._state.write_voltage if self._quantity == 'v'
                 else self._state.write_current)
        if not isinstance(self._index, slice):
            if not (isinstance(index, slice) and index == slice(None)):
                raise IndexError('a sample-mode port value is a scalar')
            index = self._index
        write(self._path, self._port, value, index)


class _DelayedValue(_ValueView):
    """A reader-local view shifted by a whole number of transient points."""

    def __init__(self, value, delay):
        self._value, self._delay = value, delay
        self._check_analysis()

    def _check_analysis(self):
        value = self._value
        if analysis() == 'dc':
            port = value._port if isinstance(value, _PinValue) else value._spec.name
            raise SimulationError(
                f"Model {value._path!r}, port {port!r}: previous_value is unavailable "
                "at a DC point, which has no previous time point. Branch on "
                "silk.analysis() and use the delay only in transient.")

    @property
    def shape(self):
        return self._value.shape

    @property
    def initial_value(self):
        return self._value.initial_value

    @initial_value.setter
    def initial_value(self, value):
        self._value.initial_value = value

    @property
    def previous_value(self):
        return _DelayedValue(self._value, self._delay + 1)

    def _read(self, index=slice(None)):
        self._check_analysis()
        return self._value._read_delayed(index, self._delay)


class _PinView:
    def __init__(self, state, path, port, index):
        self._v = _PinValue(state, path, port, 'v', index)
        self._i = _PinValue(state, path, port, 'i', index)

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        self._v._write(value)

    @property
    def i(self):
        return self._i

    @i.setter
    def i(self, value):
        self._i._write(value)


class _BusValue(_ValueView):
    def __init__(self, state, path, spec, names, quantity, index):
        self._state, self._path, self._spec = state, path, spec
        self._names = names
        self._quantity, self._index = quantity, index

    @property
    def shape(self):
        return ((self._spec.width, self._state.size) if isinstance(self._index, slice)
                else (self._spec.width,))

    def _pin(self, name):
        return _PinValue(self._state, self._path, name, self._quantity, self._index)

    @property
    def initial_value(self):
        return np.array([self._pin(name).initial_value
                         for name in self._names])

    @initial_value.setter
    def initial_value(self, value):
        _validate_initial(value, self._path, self._spec.name)
        values = np.broadcast_to(value, (self._spec.width,))
        for name, initial in zip(self._names, values):
            self._pin(name).initial_value = initial

    @property
    def previous_value(self):
        return _DelayedValue(self, 1)

    def _read_delayed(self, index, delay):
        coords = np.indices(self.shape)
        rows = np.asarray(coords[0][index])
        samples = (np.asarray(coords[1][index])
                   if isinstance(self._index, slice) else None)
        result = np.empty(rows.shape, dtype=float)
        for row, name in enumerate(self._names):
            mask = rows == row
            if np.any(mask):
                selected = samples[mask] if samples is not None else slice(None)
                result[mask] = self._pin(name)._read_delayed(selected, delay)
        return result[()]

    def _read(self, index=slice(None)):
        # Select coordinates before reading: unselected pins must neither
        # record voltage dependencies nor publish current KCL fills.
        coords = np.indices(self.shape)
        rows = np.asarray(coords[0][index])
        samples = (np.asarray(coords[1][index])
                   if isinstance(self._index, slice) else None)
        parts = []
        for row, name in enumerate(self._names):
            mask = rows == row
            if np.any(mask):
                selected = samples[mask] if samples is not None else slice(None)
                parts.append((mask, self._pin(name)._read(selected)))
        dtype = (np.result_type(*(np.asarray(value).dtype for _, value in parts))
                 if parts else float)
        result = np.empty(rows.shape, dtype=dtype)
        for mask, value in parts:
            result[mask] = value
        return result[()]

    def _write(self, value, index=slice(None)):
        execution = 'vector' if isinstance(self._index, slice) else 'sample'
        if isinstance(index, slice) and index == slice(None):
            values = split_bus(self._spec, self._names, value, execution=execution, n=self._state.size)
            for name, row in values.items():
                self._pin(name)._write(row)
            return
        # Map selected bus coordinates to the same scalar-pin accessors;
        # selection never reads currents or creates electrical storage.
        coords = np.indices(self.shape)
        rows = np.asarray(coords[0][index])
        values = np.broadcast_to(np.asarray(value), rows.shape)
        for row, name in enumerate(self._names):
            mask = rows == row
            if np.any(mask):
                if execution == 'vector':
                    samples = np.asarray(coords[1][index])[mask]
                    self._pin(name)._write(values[mask], samples)
                else:
                    self._pin(name)._write(values[mask].item())


class _BusView:
    def __init__(self, state, path, spec, names, index):
        self._state, self._path, self._spec, self._index = state, path, spec, index
        self._names = names
        self._v = _BusValue(state, path, spec, names, 'v', index)
        self._i = _BusValue(state, path, spec, names, 'i', index)

    def __getitem__(self, bit):
        return _PinView(self._state, self._path, bus_bit(self._spec, self._names, bit), self._index)

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        self._v._write(value)

    @property
    def i(self):
        return self._i

    @i.setter
    def i(self, value):
        self._i._write(value)


def _reject_dictionary_access(path, key):
    spelling = (f'ports.{key}.v' if isinstance(key, str) and key.isidentifier()
                else 'ports.VIN.v')
    raise SimulationError(
        f"Model {path!r}: ports are electrical pins, not a dictionary. "
        f"Read {spelling}; publish by assigning ports.VOUT.v (or ports.VOUT.i) "
        "and return None instead of a dictionary.")


class PortsView:
    """Named pins and buses backed solely by a run's SharedState.

    Run state lives in slots in the reserved ``__silk_`` namespace, leaving
    valid port names free for ordinary attribute lookup. Missing attributes
    resolve declared pins and buses.
    """
    # The trailing '__' keeps Python from mangling these reserved names.
    __slots__ = ('__silk_state__', '__silk_path__', '__silk_index__', '__silk_module__')

    def __init__(self, state, path, module, index=slice(None), *, metadata=None):
        self.__silk_state__, self.__silk_path__, self.__silk_index__ = state, path, index
        self.__silk_module__ = (
            _prepare_port_metadata(module) if metadata is None else metadata, {})

    def __getitem__(self, key):
        _reject_dictionary_access(self.__silk_path__, key)

    def __setitem__(self, key, value):
        _reject_dictionary_access(self.__silk_path__, key)

    def __getattr__(self, name):
        (pins, buses), views = self.__silk_module__
        if name in views:
            return views[name]
        if name in buses:
            view = _BusView(self.__silk_state__, self.__silk_path__, *buses[name],
                            self.__silk_index__)
        elif name in pins:
            view = _PinView(self.__silk_state__, self.__silk_path__, name,
                            self.__silk_index__)
        else:
            raise AttributeError(
                f"Model {self.__silk_path__!r} has no declared port {name!r}")
        views[name] = view
        return view
