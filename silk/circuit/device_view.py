"""Read-only device inspection anchored to one actual placement chain."""
from collections.abc import Mapping

from silk.exceptions import CircuitError


class _ReadOnlyDeviceMapping(Mapping):
    """A live mapping whose reads also check the held placement's lifetime."""

    def __init__(self, view, name):
        self._view = view
        self._name = name

    def _mapping(self):
        return getattr(self._view._validated_device(), self._name)

    def __getitem__(self, key):
        return self._mapping()[key]

    def __iter__(self):
        return iter(self._mapping())

    def __len__(self):
        return len(self._mapping())


def _new_device_view():
    # Allocate before restoring graph state, including cycles back to this view.
    return object.__new__(PlacementDeviceView)


class PlacementDeviceView:
    """A device read through a placement, distinct from the registered device.

    Parameters
    ----------
    owner : SubcircuitInstance
        Existing placement handle supplying ancestry and lifetime validation.
    key : str
        Authored primitive device key on the owner's definition.
    entry : dict
        Original registered leaf entry whose device identity is retained.

    Notes
    -----
    Public attribute writes/deletes and edits to ``params``, ``nets`` and
    ``layout`` mappings refuse. Values inside these mappings, including live
    Variables, remain shared. Reads follow legitimate edits to the original
    device but refuse if any placement or the leaf entry/device was replaced.
    """

    __slots__ = ('_owner', '_key', '_entry', '_device')

    def __init__(self, owner, key, entry):
        self._owner = owner
        self._key = key
        self._entry = entry
        self._device = entry['device']

    def _validated_device(self):
        self._owner._placement_chain()
        if (self._owner._def.elem_dict.get(self._key) is not self._entry or
                self._entry['device'] is not self._device):
            raise CircuitError(f'Device {self._key!r} was removed or replaced; '
                               'obtain a new device view.')
        return self._device

    @property
    def _silk_fault_target_path(self):
        self._validated_device()
        return '.'.join((*('x' + key for _, key, _ in self._owner._placement_chain()), self._key))

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        device = self._validated_device()
        value = getattr(device, name)
        if name in ('params', 'nets', 'layout') and isinstance(value, Mapping):
            return _ReadOnlyDeviceMapping(self, name)
        return value

    def _refuse_write(self, name):
        self._validated_device()
        module = self._owner._def.module_name or '<top-level>'
        raise AttributeError(
            f'Device views are read-only: cannot change {name!r}; '
            f'edit device {self._key!r} on module {module!r} instead.')

    def __setattr__(self, name, value):
        if not name.startswith('_'):
            self._refuse_write(name)
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if not name.startswith('_'):
            self._refuse_write(name)
        object.__delattr__(self, name)

    def __dir__(self):
        return sorted(set(object.__dir__(self)) |
                      {name for name in dir(self._validated_device()) if not name.startswith('_')})

    def __repr__(self):
        device = self._validated_device()
        return f'PlacementDeviceView({self._silk_fault_target_path!r}, {device!r})'

    def __copy__(self):
        result = _new_device_view()
        result.__setstate__(self.__getstate__())
        return result

    def __deepcopy__(self, memo):
        from copy import deepcopy
        result = _new_device_view()
        memo[id(self)] = result
        result.__setstate__(deepcopy(self.__getstate__(), memo))
        return result

    def __getstate__(self):
        return tuple(object.__getattribute__(self, name) for name in self.__slots__)

    def __setstate__(self, state):
        for name, value in zip(self.__slots__, state, strict=True):
            object.__setattr__(self, name, value)

    def __reduce_ex__(self, protocol):
        return (_new_device_view, (), self.__getstate__())


def _inspection_device(value):
    """Unwrap only for supported readers; never use for registry insertion."""
    return value._validated_device() if isinstance(value, PlacementDeviceView) else value


def _refuse_device_view(value, api):
    """Keep inspection views out of all supported device registries."""
    if isinstance(value, PlacementDeviceView):
        raise TypeError(f'{api} cannot register a device view; use the original '
                        'device on its holding module instead.')
