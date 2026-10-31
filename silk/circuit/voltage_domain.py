"""silk.circuit.voltage_domain — VoltageDomain descriptor for supply-rail annotation."""
from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Optional


@dataclass
class VoltageDomain:
    """A named voltage supply rail with tolerance specification.

    Define once and share across modules by passing the same object.
    Identity comparison (``is``) is the fast-path; ``is_compatible``
    handles separately-constructed objects with the same name.

    Parameters
    ----------
    name : str
        Logical rail name, e.g. ``'avdd'``, ``'dvdd'``, ``'iovdd'``.
    nominal : float
        Nominal supply voltage in volts (required).
    min : float, optional
        Explicit lower voltage bound.
    max : float, optional
        Explicit upper voltage bound.
    tolerance : float, optional
        Symmetric fractional tolerance, e.g. ``0.1`` for ±10 %.
        Ignored when explicit ``min``/``max`` are provided.
    switchable : bool, default False
        ``True`` when this rail can be powered down (power-gated, shut off
        by a sequencer) while other rails stay live. The lint rule DOM-003
        SUPPLY_DISABLED_STRESS only considers switchable domains as the
        disabled side of a crossing; an always-on rail never triggers it.

    Examples
    --------
    >>> from silk import VoltageDomain
    >>> avdd = VoltageDomain('avdd', nominal=1.8, tolerance=0.1)
    >>> round(avdd.v_min(), 2), round(avdd.v_max(), 2)
    (1.62, 1.98)
    >>> avdd.would_stress(1.9)
    True
    """

    name: str
    nominal: float
    min: Optional[float] = None
    max: Optional[float] = None
    tolerance: Optional[float] = None
    switchable: bool = False

    def v_min(self) -> float:
        """Lower voltage bound.

        Uses ``min`` if given, else ``nominal * (1 - tolerance)``, else
        ``nominal``.

        Returns
        -------
        float
            Lowest voltage (V) the rail may take.
        """
        if self.min is not None:
            return self.min
        if self.tolerance is not None:
            return self.nominal * (1.0 - self.tolerance)
        return self.nominal

    def v_max(self) -> float:
        """Upper voltage bound.

        Uses ``max`` if given, else ``nominal * (1 + tolerance)``, else
        ``nominal``.

        Returns
        -------
        float
            Highest voltage (V) the rail may take.
        """
        if self.max is not None:
            return self.max
        if self.tolerance is not None:
            return self.nominal * (1.0 + self.tolerance)
        return self.nominal

    def v_low(self) -> float:
        """Disabled (power-off) voltage — always 0.0 V.

        Only meaningful for a ``switchable`` rail; an always-on rail never
        reaches this state.

        Returns
        -------
        float
            Always ``0.0`` (V).
        """
        return 0.0

    def would_stress(self, rating: float) -> bool:
        """Whether the rail's upper bound exceeds a device rating.

        Parameters
        ----------
        rating : float
            Voltage rating (V) of the device the rail would drive.

        Returns
        -------
        bool
            ``True`` if :meth:`v_max` strictly exceeds ``rating``; values equal within a relative tolerance of 1e-9 do not count.
        """
        # Use isclose to absorb IEEE 754 rounding (e.g. 1.8 * 1.1 = 1.9800000000000002)
        return self.v_max() > rating and not isclose(self.v_max(), rating, rel_tol=1e-9)

    def contains(self, voltage: float) -> bool:
        """Whether a voltage lies inside the rail's operating range.

        Parameters
        ----------
        voltage : float
            Voltage (V) to test.

        Returns
        -------
        bool
            ``True`` if ``v_min() <= voltage <= v_max()``.
        """
        return self.v_min() <= voltage <= self.v_max()

    def is_compatible(self, other: 'VoltageDomain') -> bool:
        """True if this domain and *other* represent the same physical rail.

        Compatibility requires matching names AND overlapping voltage ranges.
        Same object identity is always compatible.
        Different names are always incompatible (domain crossing, not checked here).

        Parameters
        ----------
        other : VoltageDomain
            The domain to compare against.

        Returns
        -------
        bool
            ``True`` for the same object, or for equal names with overlapping voltage ranges.
        """
        if self is other:
            return True
        if self.name != other.name:
            return False
        # Same name — check ranges overlap
        return self.v_max() >= other.v_min() and other.v_max() >= self.v_min()
