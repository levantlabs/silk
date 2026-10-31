"""Backend waveform cards and operating points for independent sources."""
from silk.sources.classes import VoltageSource, CurrentSource
from silk.sources.waveforms import AM
from silk.simulation._ngspice_compat import NgspiceCompatibility
from silk.utils._utils import _parse_si


def _am_spice(wave):
    """Map silk's AM envelope to the newer native AM card, resolving live values."""
    va, vo = wave._tokens('va', 'vo')
    try:
        offset = str(_parse_si(va) * _parse_si(vo))
    except (TypeError, ValueError):
        # Native card parameters may remain simulator expressions. Remove only
        # their outer delimiters before combining them into one parameter.
        def operand(token):
            token = token.strip()
            if ((token.startswith('{') and token.endswith('}'))
                    or (token.startswith("'") and token.endswith("'"))):
                token = token[1:-1]
            return f'({token})'
        offset = '{' + operand(va) + '*' + operand(vo) + '}'
    parts = ['0', offset, va, *wave._tokens('mf', 'fc', 'td', 'phases'), '0']
    return 'AM(' + ' '.join(parts) + ')'


def source_spice(source, backend):
    text = source.to_spice()
    if not isinstance(source, (VoltageSource, CurrentSource)):
        return text
    wave = source.selected_waveform()
    if (backend == 'ngspice' and isinstance(wave, AM)
            and NgspiceCompatibility().has('am_new_order')):
        # Custom emitters own their syntax; only translate the built-in card.
        if (getattr(source.to_spice, '__func__', None) not in
                (VoltageSource.to_spice, CurrentSource.to_spice)
                or getattr(wave.to_spice, '__func__', None) is not AM.to_spice):
            return text
        first, separator, rest = text.partition('\n')
        name, positive, negative, _ = first.split(maxsplit=3)
        return f'{name} {positive} {negative} {_am_spice(wave)}' + separator + rest
    if backend != 'xyce' or wave is None:
        return text
    first, separator, rest = text.partition('\n')
    name, positive, negative, definition = first.split(maxsplit=3)
    value = source.waveform_initial_value()
    return f'{name} {positive} {negative} DC {value} {definition}' + separator + rest
