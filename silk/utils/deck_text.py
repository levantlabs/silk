"""The one check that a caller's string fits on one SPICE card.

Every value silk writes into a card it builds itself (device and source
parameters, waveform fields, controlled-source gains and expressions,
simulator options, analysis fields, library and output paths, corner names)
passes through :func:`refuse_line_breaks`. A line break would end the card
early and make the rest of the string deck text of its own; an injected
``.control`` block runs under ``ngspice -b``, and ngspice's ``shell`` command
runs a program. Corner names also pass :func:`validate_corner_name`, since a
corner is a ``.lib`` section and an output folder name as well.

Explicit opaque content (``add_custom_line``, a custom ``to_spice()``) is the
caller's own deck text and is written verbatim; it never comes here.
"""
import os
import unicodedata

from silk.exceptions import DeckTextError

#: Every character ``str.splitlines()`` splits on. A test derives the same set
#: by scanning all code points, so this cannot drift from Python's definition.
LINE_BREAKS = frozenset('\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029')


def refuse_line_breaks(value, *, who):
    """Return *value*, or raise :class:`DeckTextError` if it is a string
    holding a line break.

    The whole raw string is checked. silk's SPICE number parser strips its
    input before matching, so ``'1n\\n'`` passed the parser while the
    unstripped string was what reached the deck.

    Parameters
    ----------
    value : object
        Anything. Only a ``str`` is inspected; numbers, ``None`` and
        Variables pass through (a Variable's value is checked when it is
        resolved for rendering).
    who : str
        Where the value is written, e.g. ``"nmos1p8 'mn' parameter 'w'"``.
    """
    if isinstance(value, str):
        for ch in value:
            if ch in LINE_BREAKS:
                raise DeckTextError(
                    f'{who}: {value!r} contains a line break ({ch!r}). silk '
                    f'writes this value into one SPICE card, where a line '
                    f'break would start a new deck line. Remove the line '
                    f'break; to add deck text of your own, use '
                    f'add_custom_line().')
    return value


def card_text(value, *, who):
    """Return the text silk writes for *value* into a card, refused if that
    text holds a line break.

    The check is on the FINAL text, after conversion: ``os.fspath`` for a
    path-like value (a :class:`pathlib.Path`), ``str()`` for anything else.
    Checking only the authored object missed a value whose string form holds
    a break. Emitters call this at the point they write.
    """
    text = os.fspath(value) if isinstance(value, os.PathLike) else str(value)
    if isinstance(text, bytes):
        text = os.fsdecode(text)
    return refuse_line_breaks(text, who=who)


def validate_corner_name(corner, *, who='corner'):
    """Return ``str(corner)``, or raise :class:`DeckTextError`.

    A corner name is written as the ``.lib`` section of the PDK library line
    and names the corner's output folder. Refused: any line break or other
    control character, and the exact names ``'.'`` and ``'..'``, which as a
    folder name would put the run outside its own folder. Nothing stricter:
    a name the PDK does not declare may still be a valid library section,
    and the folder name is percent-encoded, so separators and spaces cannot
    leave it.
    """
    name = str(corner)
    for ch in name:
        if ch in LINE_BREAKS or unicodedata.category(ch) == 'Cc':
            raise DeckTextError(
                f'{who} {name!r} contains the control character {ch!r}. A '
                f'corner name is written into the PDK .lib line and names an '
                f'output folder; it cannot hold a line break or a control '
                f'character.')
    if name in ('.', '..'):
        raise DeckTextError(
            f'{who} {name!r} is not a usable corner name: it also names the '
            f"corner's output folder, and {name!r} would put the run outside "
            f'its own folder.')
    return name
