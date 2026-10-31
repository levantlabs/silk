"""Leaf, checked address context and inventory comparison for view emission."""
from dataclasses import dataclass
import os
import warnings

from silk.signals.rawfile import RawFile


class _FrozenFacts(dict):
    """Private immutable mappings; tuples hold arrays, strings are shared."""
    def _readonly(self, *args, **kwargs):
        raise TypeError('checked node facts are read-only')

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _readonly


def _freeze(value):
    if isinstance(value, dict):
        return _FrozenFacts((key, _freeze(child)) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(child) for child in value)
    return value


def _inventory(raw):
    return tuple((p.plotname, p.real, p.npoints,
                  tuple((v.name, v.type) for v in p.vectors)) for p in raw.plots)


@dataclass(frozen=True, init=False)
class ViewNodeContext:
    """Opaque checked snapshot returned by ``view_node_context(owner)``.

    ``source`` is borrowed read-only: do not mutate plots, vectors or samples.
    Obtain a fresh context for each emission. This is not a live file watcher.
    """
    source: RawFile
    _fingerprint: str
    _source_path: str
    _inventory: tuple
    _metadata: tuple
    _proofs: tuple
    _source_currents: tuple

    @classmethod
    def _checked(cls, source, metadata, proofs, source_currents=()):
        obj = object.__new__(cls)
        values = dict(source=source, _fingerprint=source._content_sha256,
                      _source_path=os.path.abspath(source.source_path),
                      _inventory=_inventory(source), _metadata=_freeze(metadata),
                      _proofs=tuple(tuple(p) for p in proofs),
                      _source_currents=_freeze(source_currents))
        for key, value in values.items():
            object.__setattr__(obj, key, value)
        return obj

    def _for_source(self, raw):
        if (getattr(raw, '_content_sha256', None) != self._fingerprint
                or not raw.source_path
                or os.path.abspath(raw.source_path) != self._source_path
                or _inventory(raw) != self._inventory):
            warnings.warn('Node context source/fingerprint or emission inventory changed; '
                          'omitting graph metadata and using legacy labels.', UserWarning,
                          stacklevel=3)
            return None
        return self._metadata
