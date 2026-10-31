"""Structural placement targets, bound explicitly through the existing probe route."""
import contextvars
import re
import weakref
from numbers import Integral

from silk.exceptions import CircuitError, UnboundSignalError
from silk.circuit.declared_indices import select_positions

_modules = weakref.WeakSet()

#: Set by a selected test's probe wrapper once it has checked that every bench
#: reference it was given was made by that same selected test; a reference
#: owned by a selected test binds on the authored bench only inside that call.
selected_probe_call = contextvars.ContextVar('silk_selected_probe_call', default=False)


def register_module(module):
    """Track live graphs weakly so implicit reads can discover unrun contexts."""
    _modules.add(module)


class LiteralVoltageName(str):
    """A literal node address carried through the shared acquisition route."""
    _literal_voltage = True

    def __new__(cls, value, *, hierarchy=()):
        obj = super().__new__(cls, value)
        obj._hierarchy = tuple(hierarchy) or getattr(value, '_hierarchy', ())
        return obj

    def voltage_forms(self):
        forms = {str(self)}
        if self._hierarchy:
            prefix = '.'.join(self._hierarchy) + '.'
            forms.add(':'.join(self._hierarchy) + ':' + str(self)[len(prefix):])
        return forms


class _ModuleReader:
    def __init__(self, root):
        self._root = weakref.ref(root)
        self.test_name = '<direct Module simulation>'

    @property
    def netlist(self):
        return self._root()


class PlacementArray:
    """Actual placement keys grouped by their declared integer suffix.

    Obtained from :meth:`silk.circuit.modules.SubcircuitInstance.child` or
    :meth:`silk.circuit.modules.SubcircuitInstance.device` for a name such as ``'x'`` whose
    placements are ``x[0]``, ``x[1]``. Indexing uses the declared number, not
    the position; slices keep the declared numbers. Iterating yields the
    member handles in declared-number order.

    Parameters
    ----------
    name : str
        The array's base name.
    owner : SubcircuitInstance
        The handle the members were looked up on.
    kind : ElementType
        ``MODULE`` or ``DEVICE``.
    members : iterable of tuple
        ``(number, key, entry)`` or ``(number, key, entry, definition)`` for
        each member.
    """
    def __init__(self, name, owner, kind, members):
        self._name = name
        self._owner = owner
        self._kind = kind
        self._members = tuple(item if len(item) == 4 else (*item, item[2]['device']) for item in members)
        self._indices = tuple(item[0] for item in self._members)

    @property
    def name(self):
        return self._name

    def __setattr__(self, name, value):
        if not name.startswith('_'):
            raise AttributeError(
                f"Placement arrays are read-only: cannot assign {name!r}. "
                "To rewire a port, call connect_module on the parent module.")
        object.__setattr__(self, name, value)

    def _member(self, position):
        _, key, entry, definition = self._members[position]
        self._owner._placement_chain()
        if self._owner._def.elem_dict.get(key) is not entry or entry['device'] is not definition:
            raise CircuitError(f'Placement {key!r} was removed or replaced; obtain a new array handle.')
        return self._owner._lookup_element(key, self._kind)

    def __iter__(self):
        return (self._member(p) for p in range(len(self)))

    def __len__(self):
        return len(self._members)

    def __getitem__(self, key):
        positions = select_positions(self._indices, key)
        if isinstance(key, Integral):
            return self._member(positions[0])
        return type(self)(self.name, self._owner, self._kind, tuple(self._members[p] for p in positions))


def array_entries(module, name, kind):
    """Recognize whole actual keys; never decode simulator or underscore aliases."""
    pattern = re.compile(re.escape(name) + r'\[([+-]?\d+)\]\Z')
    found = {}
    for key, entry in module.elem_dict.items():
        match = pattern.fullmatch(key)
        if match is None or entry.get('type') != kind:
            continue
        number = int(match[1])
        if number in found:
            raise CircuitError(f'Array {name!r} has duplicate declared index {number}: '
                               f'{found[number][0]!r} and {key!r}.')
        found[number] = (key, entry)
    return tuple((number, *item) for number, item in sorted(found.items()))


def _module_paths(root, target, active=()):
    """Find actual entry chains embedding a definition, preserving repeated placements."""
    if root is target:
        yield ()
    if id(root) in active:
        return
    from silk.circuit.circuit_elements import ElementType
    for key, entry in root.elem_dict.items():
        if entry.get('type') == ElementType.MODULE:
            for tail in _module_paths(entry['device'], target, (*active, id(root))):
                yield ((root, key, entry), *tail)


def _resolve_path(chain, name):
    from silk.circuit.emitted_names import emitted_submodule_name
    # Walk ports upward through LIVE placement-local wiring.
    for position in range(len(chain) - 1, -1, -1):
        _, key, entry = chain[position]
        pin = (entry.get('pin_list') or {}).get(name)
        if pin is None:
            hierarchy = tuple(emitted_submodule_name(link[1]) for link in chain[:position + 1])
            return LiteralVoltageName('.'.join((*hierarchy, name)), hierarchy=hierarchy)
        name = pin.net
        if name is None:
            raise CircuitError(f'Port on placement {key!r} is not connected.')
    return name


class PlacementReference:
    """A net/port/bus target in an actual placement, with no acquired samples.

    Use ``tb.probe(stage.vbias)`` to bind it explicitly. A bare read or Signal
    operation is permitted only when exactly one live result context is eligible.
    Bus subscripts name declared numbers and preserve them through slicing.

    Obtained from :meth:`silk.circuit.modules.SubcircuitInstance.net` or
    :meth:`silk.circuit.modules.SubcircuitInstance.bus`, or by attribute access on a handle.

    Parameters
    ----------
    root : Module
        The top-level design the reference is anchored to.
    chain : iterable of tuple
        ``(parent, key, entry)`` for each placement from the root down.
    name : str
        The net, port or bus name in the innermost placement.
    children : tuple of PlacementReference, optional
        The per-bit references of a bus; ``None`` for a scalar.
    indices : tuple of int, optional
        The declared bit numbers of a bus; ``None`` for a scalar.
    """
    def __init__(self, root, chain, name, *, children=None, indices=None):
        self._root = root
        self._chain = tuple(chain)
        self._definitions = tuple(entry['device'] for _, _, entry in self._chain)
        self.name = name
        self._children = children
        self._indices = indices

    def _validate(self):
        for (parent, key, entry), definition in zip(self._chain, self._definitions):
            if parent.elem_dict.get(key) is not entry or entry['device'] is not definition:
                raise CircuitError(f'Placement {key!r} was removed or replaced; obtain a new placement handle.')
        definition = self._chain[-1][2]['device']
        if self._children is None and self.name not in definition.net_dict and self.name not in definition.pin_list:
            raise CircuitError(f'Net {self.name!r} was removed from its placement.')

    def _chain_for(self, bench):
        self._validate()
        netlist = bench.netlist
        if netlist is None:
            raise CircuitError('This bench has no placement netlist; add the module first.')
        paths = list(_module_paths(netlist, self._root))
        if len(paths) == 1:
            return (*paths[0], *self._chain)
        # A reference taken through a root design can be bound to that owner's
        # selected characterization bench by trimming the owner's embedding.
        from silk.circuit.test_placement import current_placement, authored_test
        authored = authored_test(bench)
        selection = current_placement(authored)
        owner = selection.entry if selection is not None else None
        if owner is not None and owner._root_ref() is self._root:
            for prefix in _module_paths(self._root, authored.module):
                from silk.circuit.modules_structure import _child_spec_owner_path
                # Compare owner paths using the same encoded design keys.
                path = '$'
                for _, key, _ in prefix:
                    path = _child_spec_owner_path(path, key)
                if path == owner.owner_path and all(a[2] is b[2] for a, b in zip(self._chain[:len(prefix)], prefix)) and len(self._chain) >= len(prefix):
                    suffix = self._chain[len(prefix):]
                    embeddings = list(_module_paths(netlist, authored.module))
                    if len(embeddings) == 1:
                        return (*embeddings[0], *suffix)
        raise CircuitError('Reference placement is absent or ambiguous in this bench; '
                           'use tb.probe(stage.net(name)) with this bench\'s placement handle.')

    def _bind(self, bench, *, type='V', analysis=None):
        if type is not None and str(type).upper() != 'V':
            raise CircuitError('A net/port reference is a voltage, not a current. '
                               "Use tb.probe('i(source)') for source current or a terminal-current/ammeter probe.")
        chain = self._chain_for(bench)
        from silk.signals.base import Signal
        if self._children is not None:
            return Signal._make_bus_handle(self.name,
                tuple(child._bind(bench, analysis=analysis) for child in self._children), indices=self._indices)
        return bench.probe(LiteralVoltageName(_resolve_path(chain, self.name)), analysis=analysis)

    def _contexts(self):
        self._validate()
        modules = list(_modules)
        # Test decks are execution graphs, not authored design roots.
        decks = {id(test.netlist) for mod in modules for test in mod._tests.values() if test.netlist is not None}
        designs = [mod for mod in modules if id(mod) not in decks]
        children = {id(entry['device']) for mod in designs for entry in mod.elem_dict.values()
                    if entry.get('type') == 'm'}
        roots = [mod for mod in designs if id(mod) not in children]
        contexts = []
        seen = set()
        for root in roots:
            for entry in root.list_test_entries():
                from silk.circuit.test_placement import authored_test, execution_scope
                test = authored_test(entry.test)
                key = (id(root), entry.owner_path, id(test))
                if key in seen:
                    continue
                try:
                    with execution_scope(entry):
                        self._chain_for(test)
                except CircuitError:
                    continue
                seen.add(key)
                contexts.append(entry.test)
        # A bench netlist reference must also discover authored owners whose
        # DUT is itself embedded; those are already covered by root entries.
        if not contexts and self._root.__dict__.get('_probe_execution_deck'):
            raise CircuitError('Reference placement belongs to an expired execution context; obtain a new handle.')
        if (id(self._root) not in decks and not self._root.__dict__.get('_probe_execution_deck') and (not contexts
                or self._root.sim_results_loaded)):
            reader = self._root.__dict__.get('_placement_reader')
            if reader is None:
                reader = _ModuleReader(self._root)
                self._root.__dict__['_placement_reader'] = reader
            contexts.append(reader)
        return contexts

    def _as_signal(self):
        contexts = self._contexts()
        if len(contexts) != 1:
            raise UnboundSignalError('Reference has several or no eligible result contexts; '
                                     'use tb.probe(reference) to select the bench/owner explicitly.')
        context = contexts[0]
        if isinstance(context, _ModuleReader):
            from silk.signals.base import Signal
            if self._children is not None:
                return Signal._make_bus_handle(self.name, tuple(child._as_signal() for child in self._children),
                                               indices=self._indices)
            return Signal._unbound(LiteralVoltageName(_resolve_path(self._chain, self.name)), bench=context)
        return context.probe(self)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        value = getattr(self._as_signal(), name)
        if callable(value):
            def call(*args, **kwargs):
                return value(*(normalize_reference(arg) for arg in args),
                             **{key: normalize_reference(arg) for key, arg in kwargs.items()})
            return call
        return value

    def __len__(self):
        return 1 if self._children is None else len(self._children)

    def __iter__(self):
        if self._children is None:
            raise TypeError('Scalar placement reference cannot be iterated.')
        return iter(self._children)

    def __getitem__(self, key):
        self._validate()
        if self._children is None:
            raise TypeError('Scalar placement reference cannot be indexed.')
        positions = select_positions(self._indices, key)
        if isinstance(key, Integral):
            return self._children[positions[0]]
        return type(self)(self._root, self._chain, self.name,
                          children=tuple(self._children[p] for p in positions),
                          indices=tuple(self._indices[p] for p in positions))

    def __repr__(self):
        return f'PlacementReference({self.name!r})'

    def __add__(self, other):
        return self._as_signal() + normalize_reference(other)

    def __radd__(self, other):
        return normalize_reference(other) + self._as_signal()

    def __sub__(self, other):
        return self._as_signal() - normalize_reference(other)

    def __rsub__(self, other):
        return normalize_reference(other) - self._as_signal()

    def __mul__(self, other):
        return self._as_signal() * normalize_reference(other)

    def __rmul__(self, other):
        return normalize_reference(other) * self._as_signal()

    def __truediv__(self, other):
        return self._as_signal() / normalize_reference(other)

    def __rtruediv__(self, other):
        return normalize_reference(other) / self._as_signal()

    def __pow__(self, other):
        return self._as_signal() ** normalize_reference(other)

    def __rpow__(self, other):
        return normalize_reference(other) ** self._as_signal()

    def __neg__(self):
        return -self._as_signal()


class BenchBitReference(PlacementReference):
    """One scalar net of the test bench itself, with no acquired samples.

    Obtained from a :class:`BenchBusReference` (``tb.bus('d')[4]`` or by
    iteration). It keeps the bench identity of its bus: the bench it was
    made on, and the selected-owner view (``top.get_test(...)``) when it came
    from one, so binding it on another bench or owner refuses.
    """
    def __init__(self, bench, name, owner=None):
        self._authored = weakref.ref(bench)
        self._owner = weakref.ref(owner if owner is not None else bench)
        self.name = name
        self._children = None
        self._indices = None

    def _validate(self):
        if self._authored() is None or self._owner() is None:
            raise CircuitError(f'The bench of {self.name!r} no longer exists.')

    def _set_owner(self, owner):
        self._owner = weakref.ref(owner)

    def _bind(self, bench, *, type='V', analysis=None):
        if type is not None and str(type).upper() != 'V':
            raise CircuitError('A net reference is a voltage, not a current.')
        self._validate()
        if self._authored() is not bench:
            raise CircuitError(f'{self.name!r} belongs to another test bench; '
                               'use that bench\'s probe, or call tb.bus() on this one.')
        if self._owner() is not bench and not selected_probe_call.get():
            raise CircuitError(f'{self.name!r} was made by a selected test; probe it '
                               'through that selected test.')
        return bench.probe(LiteralVoltageName(self.name), analysis=analysis)

    def _as_signal(self):
        self._validate()
        return self._owner().probe(self)

    def __repr__(self):
        return f'BenchBitReference({self.name!r})'


class BenchBusReference(PlacementReference):
    """A bus net of the test bench itself, with no acquired samples.

    ``tb.bus('d')`` returns one. It follows the declared-number rules of a
    placement bus reference (``ref[7]``, ``ref[5:7]``, ``len``, iteration)
    and binds through the same door: ``tb.probe(tb.bus('d'))`` gives a bus
    Signal over this bench. A bit subscript or iteration yields a
    :class:`BenchBitReference`. The reference remembers its bench, and the
    selected owner (``top.get_test(...)``) when it came from one, through
    slicing, arithmetic and probing.
    """
    def __init__(self, bench, name, bits, indices, owner=None):
        self._authored = weakref.ref(bench)
        self._owner = weakref.ref(owner if owner is not None else bench)
        self.name = name
        self._children = tuple(bits)
        self._indices = tuple(indices)

    @classmethod
    def _from_names(cls, bench, name, bit_names, indices):
        return cls(bench, name, tuple(BenchBitReference(bench, bit) for bit in bit_names),
                   indices)

    def _validate(self):
        if self._authored() is None or self._owner() is None:
            raise CircuitError(f'The bench of bus {self.name!r} no longer exists.')

    def _set_owner(self, owner):
        self._owner = weakref.ref(owner)
        for child in self._children:
            child._set_owner(owner)

    def _bind(self, bench, *, type='V', analysis=None):
        if type is not None and str(type).upper() != 'V':
            raise CircuitError('A bus reference is a voltage, not a current.')
        self._validate()
        if self._authored() is not bench:
            raise CircuitError(f'Bus {self.name!r} belongs to another test bench; '
                               'use that bench\'s probe, or call tb.bus() on this one.')
        if self._owner() is not bench and not selected_probe_call.get():
            raise CircuitError(f'{self.name!r} was made by a selected test; probe it '
                               'through that selected test.')
        from silk.signals.base import Signal
        return Signal._make_bus_handle(
            self.name, tuple(child._bind(bench, analysis=analysis) for child in self._children),
            indices=self._indices)

    def _as_signal(self):
        self._validate()
        return self._owner().probe(self)

    def __getitem__(self, key):
        self._validate()
        positions = select_positions(self._indices, key)
        if isinstance(key, Integral):
            return self._children[positions[0]]
        sliced = type(self)(self._authored(), self.name,
                            tuple(self._children[p] for p in positions),
                            tuple(self._indices[p] for p in positions),
                            owner=self._owner())
        return sliced

    def __repr__(self):
        return f'BenchBusReference({self.name!r})'


def foreign_bench_reference(value, owner):
    """True when *value* is a bench reference made by an owner other than *owner*."""
    return isinstance(value, (BenchBusReference, BenchBitReference)) and value._owner() is not owner


def normalize_reference(value):
    return value._as_signal() if isinstance(value, PlacementReference) else value
