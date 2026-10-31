"""Value-independent physical layout of a logical independent source."""
from dataclasses import dataclass

from silk.exceptions import CircuitError

WAVEFORM_SLOTS = ('pwl', 'pulse', 'sine', 'exponential', 'sffm', 'am',
                  'trnoise', 'trrandom', 'waveform')


def selected_waveform(name, state):
    """Validate exclusivity before any caller chooses or discards a slot."""
    slots = [slot for slot in WAVEFORM_SLOTS if state.get(slot) is not None]
    if len(slots) > 1:
        raise CircuitError(f'Source {name!r} has conflicting waveform slots: '
                           f'{", ".join(slots)}. Supply exactly one waveform.')
    return (slots[0], state[slots[0]]) if slots else (None, None)


def _escape_component(name):
    escape = {'_': '_e', '.': '_d', ':': '_c'}
    return ''.join(escape.get(ch, ch) for ch in name)


@dataclass(frozen=True)
class SourceLayout:
    names: tuple
    dc_name: str
    internal_node: str | None
    current_terms: tuple


def source_layout(source, state=None):
    state = vars(source) if state is None else state
    return source_card_layout(state['name'], source.instance_name, state)


def source_card_layout(name, prefix, state):
    """Shared string-level layout; ``prefix`` is the logical instance token."""
    slot, _ = selected_waveform(name, state)
    if slot is None:
        return SourceLayout((prefix,), prefix, None, ((prefix, 1),))
    wave_name = prefix + ('__silk_pwl' if slot == 'pwl' else '__silk_wave')
    dc_name = prefix + '__silk_dc'
    voltage = prefix.startswith('V')
    node = '__silk_src_V' + _escape_component(name) if voltage else None
    terms = ((wave_name, 1),) if voltage else ((wave_name, 1), (dc_name, 1))
    return SourceLayout((wave_name, dc_name), dc_name, node, terms)


def source_instances(sources, elements=None, logical_prefix='', physical_prefix=''):
    """Yield authored paths, source objects and emitted hierarchy prefixes."""
    from silk.circuit.circuit_elements import ElementType
    from silk.circuit.emitted_names import emitted_submodule_name
    for name, source in (sources or {}).items():
        yield logical_prefix + str(name), source, physical_prefix
    for name, entry in (elements or {}).items():
        if entry['type'] != ElementType.MODULE:
            continue
        child = entry['device']
        yield from source_instances(
            child.sig_sources_dict, child.elem_dict,
            logical_prefix + 'x' + str(name) + '.',
            physical_prefix + emitted_submodule_name(str(name)) + '.')


def _current_request_key(request):
    from silk.spice_names import to_spice_name
    token = str(request)
    if token.lower().startswith('i(') and token.endswith(')'):
        token = token[2:-1]
    if token.startswith('@') and token.lower().endswith('[current]'):
        token = token[1:-len('[current]')]
    return to_spice_name(token, kind='branch').replace(':', '.').lower()


def physical_source_card(request, cards):
    """Canonical inventory card for designer-array or native current spelling."""
    token = _current_request_key(request)
    for card in cards:
        key = _current_request_key(card)
        if token == key or ('.' in card and token == 'i.' + key):
            return card
    return None


def current_source_card(sources, request, elements=None):
    """Match a logical or physical current source; keep the matched card."""
    from silk.sources.classes import CurrentSource
    token = _current_request_key(request)
    for name, source, prefix in source_instances(sources, elements):
        if not isinstance(source, CurrentSource):
            continue
        cards = [prefix + n for n in source.physical_instance_names()]
        card = physical_source_card(request, cards)
        if card is not None:
            return source, card
        if token in (_current_request_key(name),
                     _current_request_key(prefix + source.instance_name)):
            return source, None
    return None, None


def current_source_for_request(sources, request, elements=None):
    return current_source_card(sources, request, elements)[0]


def refuse_ac_current(source):
    from silk.exceptions import NotSupportedError
    raise NotSupportedError(
        f'CurrentSource {source.name!r}: an ideal current source\'s AC current is '
        'its ac= value; to measure it, place a 0 V VoltageSource in series '
        "(silk's ammeter pattern). AC current-source readback is unsupported.")
