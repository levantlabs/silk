"""Private root/owner selection for runtime measurements."""
from contextlib import contextmanager
from contextvars import ContextVar
import weakref

_current = ContextVar('silk_spec_owner', default=None)
# A shared definition's reduced annotations are never evidence for another root.
_annotated_definitions = weakref.WeakKeyDictionary()


@contextmanager
def owner_scope(index, owner):
    token = _current.set((index, owner))
    try:
        yield
    finally:
        _current.reset(token)


def bind_measurement(spec, measurement, index, owner):
    """Capture the destination now, including for delayed DB publication."""
    index.scoped_specs[id(spec)] = weakref.ref(spec)
    references = spec.__dict__.setdefault('_scope_references', {})
    reference = references.get(id(index))
    if reference is None:
        spec_ref = weakref.ref(spec)
        identity = id(index)
        def expired(dead):
            live = spec_ref()
            if live is None:
                return
            for key, value in list(live.measurements.all_items() if isinstance(live.measurements, MeasurementSlots) else live.measurements.items()):
                scope = getattr(value, '_owner_scope', None)
                if scope is not None and scope[0] is dead:
                    del live.measurements[key]
            live.__dict__['_attempts'] = [(scope, a) for scope, a in live.__dict__.get('_attempts', ())
                                             if scope is None or scope[0] is not dead]
            live.__dict__.get('_scope_references', {}).pop(identity, None)
        reference = weakref.ref(index, expired)
        references[id(index)] = reference
    measurement._owner_scope = (reference, owner)
    return measurement


def bind_current(spec, measurement):
    # Explicitly staged scopes take precedence over an incidental caller scope.
    if not hasattr(measurement, '_owner_scope') and _current.get() is not None:
        bind_measurement(spec, measurement, *_current.get())


def selected_measurements(spec):
    scope = _current.get()
    values = list(spec.measurements.all_values() if isinstance(spec.measurements, MeasurementSlots) else spec.measurements.values())
    if scope is None:
        return values
    index, owner = scope
    scoped = [m for m in values if hasattr(m, '_owner_scope')]
    if not scoped:
        return values  # Existing unscoped/manual declarations keep their contract.
    return [m for m in scoped if m._owner_scope[0]() is index
            and m._owner_scope[1] == owner]


def transient_free(state):
    state = dict(state)
    state.pop('_scope_references', None)
    if '_attempts' in state:
        state['_attempts'] = [(scope, a) for scope, a in state['_attempts'] if scope is None]
    slots = state['measurements']
    items = list(slots.all_items() if isinstance(slots, MeasurementSlots) else slots.items())
    if any(hasattr(value, '_owner_scope') for _, value in items):
        state['measurements'] = MeasurementSlots({key: value for key, value in items
                                                  if not hasattr(value, '_owner_scope')})
    return state


def stage_for_root(root, spec, measurement, owner):
    """Preserve legacy local DB slots until this root uses scoped evidence."""
    index = root.__dict__.get('_test_evidence')
    # A projection-only index is bookkeeping, not scoped simulation evidence.
    scoped = index is not None and (index.records or index.selectors or index.annotations or index.scoped_specs)
    if scoped or owner != '$' or spec.__dict__.get('_scope_references'):
        from silk.circuit.test_placement import evidence_index
        bind_measurement(spec, measurement, evidence_index(root), owner)
    return measurement


def annotation_reader(index, owner, module):
    """Return an explicit owner-aware dotted-metric reader, never a Net override."""
    from silk.specs.resolver import ResolveResult
    def read(metric):
        key = (owner, metric)
        if key in index.annotations:
            return ResolveResult(True, index.annotations[key])
        if (metric in _annotated_definitions.get(module, ())
                or any(name == metric for _, name in index.annotations)):
            return ResolveResult(False, reason='no_owner_annotation')
        return None
    return read


def retain_annotations(module, report):
    """Copy only declared numeric annotation evidence out of instance snapshots."""
    from silk.circuit.test_placement import evidence_index, _active
    from silk.specs.resolver import resolve_metric
    selection = next((s for s in reversed(_active.get())
                      if s.entry is not None
                      and (s.test.netlist is module or s.reader is module)), None)
    if selection is not None:
        root = selection.entry._root_ref()
        definition = selection.entry.module
        owner_base = selection.owner_path
        from silk.circuit.modules_results import DUT_ELEMENT_NAME
        from silk.circuit.emitted_names import emitted_submodule_name
        prefix = emitted_submodule_name(DUT_ELEMENT_NAME)
    else:
        metadata = module.__dict__.get('_evidence_owner')
        if metadata is not None:
            root = metadata[0]()
            if root is None:
                return
            definition, owner_base, prefix = metadata[1:]
        else:
            root, definition, owner_base, prefix = module, module, '$', ''
    index = evidence_index(root)
    for instance in definition.iter_instances(include_self=True):
        path = '.'.join(part for part in (prefix, instance.spice_path) if part)
        snapshot = report.instance_modules.get(path)
        if snapshot is None:
            continue
        owner = owner_base + instance.owner_path[1:]
        for spec in instance.module._specs:
            if spec.metric.startswith(('net.', 'device.')):
                result = resolve_metric(snapshot, spec.metric)
                if result.resolved:
                    index.annotations[(owner, spec.metric)] = result.value
                    identity = getattr(report, '_observation_id', None)
                    if identity is not None:
                        from silk.specs.spec import Measurement
                        measurement = Measurement('user', result.value,
                            timestamp=getattr(report, '_observation_time', 0.))
                        measurement._attempt_id = identity + ':' + spec.key
                        bind_measurement(spec, measurement, index, owner)
                        spec.record_measurement(measurement)
                    _annotated_definitions.setdefault(instance.module, set()).add(spec.metric)


class MeasurementSlots(dict):
    """Private scoped storage with the existing public source/profile view.

    Public reads never expose runtime owner keys. Multiple owners' values are
    selected by the active scope, or latest per legacy slot when unqualified.
    """
    def all_items(self):
        return dict.items(self)

    def all_values(self):
        return dict.values(self)

    def _view(self):
        from silk.specs.spec import _legacy_measurement_slot_key
        values = list(dict.values(self))
        scope = _current.get()
        if scope is not None and any(hasattr(m, '_owner_scope') for m in values):
            index, owner = scope
            values = [m for m in values if getattr(m, '_owner_scope', (lambda: None, None))[0]() is index
                      and m._owner_scope[1] == owner]
        view = {}
        for measurement in sorted(values, key=lambda m: m.timestamp):
            view[_legacy_measurement_slot_key(measurement)] = measurement
        return view

    def __iter__(self):
        return iter(self._view())

    def __len__(self):
        return len(self._view())

    def __getitem__(self, key):
        if isinstance(key, tuple) and key and key[0] == 'owner':
            return dict.__getitem__(self, key)
        return self._view()[key]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return self._view().keys()

    def values(self):
        return self._view().values()

    def items(self):
        return self._view().items()

    def __contains__(self, key):
        return key in self._view()

    def __eq__(self, other):
        return self._view() == dict(other)

    def __repr__(self):
        return repr(self._view())

    def copy(self):
        return self._view().copy()

    def __delitem__(self, key):
        if isinstance(key, tuple) and key and key[0] == 'owner':
            return dict.__delitem__(self, key)
        value = self._view()[key]
        for internal, candidate in list(dict.items(self)):
            if candidate is value:
                dict.__delitem__(self, internal)

    def pop(self, key, *default):
        try:
            value = self[key]
        except KeyError:
            if default:
                return default[0]
            raise
        del self[key]
        return value

    def popitem(self):
        view = self._view()
        if not view:
            raise KeyError('popitem(): dictionary is empty')
        key = next(reversed(view))
        return key, self.pop(key)
