"""silk.simulation.corner — process corner enumeration.

Corners are str-subclass enum members, so existing code that passes bare
string corner names (e.g. ``'tt'``) keeps working unchanged::

    # Both are valid and equivalent:
    tb.run(corner='tt')
    tb.run(corner=Corner.TT)
    tb.run_corners(corners=Corner.standard())
"""
from enum import Enum


class Corner(str, Enum):
    """Standard process corners for SkyWater 130nm (and most CMOS PDKs).

    The ``silk.pdk`` manifest (``silk/pdk/data/*.pdk.yaml``, read through
    ``silk.pdk.registry``) is the authoritative source of a PDK's corners —
    it lists every named section (``tt``, ``ff``, ``ss_hh_mm``, ``mc``, ...)
    along with its ``kind`` (``process``/``rc``/``composed``/``statistical``).
    ``Corner`` is typed sugar over the five standard *process* corners for
    convenience and IDE completion; it does not compete with or replace the
    manifest. See ``tests/test_corner.py::test_corner_enum_values_are_valid_sky130_corners``
    for the invariant that every ``Corner`` member is a real sky130 corner.

    Because ``Corner`` inherits from ``str``, instances compare equal to
    plain strings — ``Corner.TT == 'tt'`` is ``True``.

    Attributes
    ----------
    TT : Corner
        Typical-Typical (nominal).
    FF : Corner
        Fast-Fast (best-case speed, worst-case power).
    SS : Corner
        Slow-Slow (worst-case speed).
    SF : Corner
        Slow NMOS, Fast PMOS.
    FS : Corner
        Fast NMOS, Slow PMOS.

    Examples
    --------
    >>> from silk import Corner
    >>> Corner.TT == 'tt'
    True
    >>> Corner.FF.value
    'ff'
    """
    TT = 'tt'
    FF = 'ff'
    SS = 'ss'
    SF = 'sf'
    FS = 'fs'

    def __str__(self) -> str:
        return self.value

    def __format__(self, format_spec: str) -> str:
        return self.value.__format__(format_spec)

    @classmethod
    def standard(cls) -> tuple:
        """Return the five standard corners in canonical order.

        Returns
        -------
        tuple of Corner
            ``(TT, FF, SS, SF, FS)``
        """
        return (cls.TT, cls.FF, cls.SS, cls.SF, cls.FS)
