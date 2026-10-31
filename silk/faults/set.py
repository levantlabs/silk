"""FaultSet: a declarative collection of faults with lazy sprinkle resolution.

Two storage lists back it:

* ``_faults``: explicit faults added through ``add()`` or the constructor.
* ``_sprinkles``: ``_SprinkleSpec`` entries from ``sprinkle()`` calls, resolved
  at ``realize()`` time against the live module.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

from silk.faults.types import Fault
from silk.faults.sugar import FaultFactory


def _validate_numeric_params(fault_type, params, context):
    """Check the numeric declarations for the supported parametric faults."""
    names = {'w_variation': ('sigma',), 'l_variation': ('sigma',),
             'vth_shift': ('delta', 'sigma')}.get(fault_type, ())
    for name in names:
        value = params.get(name, 0.0)
        if not math.isfinite(value) or (name == 'sigma' and value < 0):
            expected = 'finite and >= 0' if name == 'sigma' else 'finite'
            raise ValueError(
                f'{fault_type} {context}: {name}={value!r}; expected {expected}')


@dataclass
class _SprinkleSpec:
    """A lazy sprinkle. ``fault_factory`` is a FaultFactory callable that
    takes a device path and returns a Fault."""
    scope: str                     # regex
    fault_factory: FaultFactory
    rate: float
    seed: Optional[int] = None     # optional per-sprinkle sub-seed

    def to_json(self) -> dict:
        return {
            'scope': self.scope,
            'rate': self.rate,
            'seed': self.seed,
            'fault_type': self.fault_factory.fault_type,
            'fault_params': dict(self.fault_factory.params),
        }

    @classmethod
    def from_json(cls, d: dict) -> '_SprinkleSpec':
        factory = FaultFactory(d['fault_type'], d['fault_params'])
        return cls(scope=d['scope'], fault_factory=factory,
                   rate=d['rate'], seed=d.get('seed'))


class FaultSet:
    """A declarative collection of faults for a campaign.

    A set holds explicit :class:`silk.faults.Fault` objects and *sprinkles*:
    templates that are expanded to concrete faults only when
    :meth:`realize` is called against a module. Iterating a set yields its
    explicit faults only, not sprinkles.

    Parameters
    ----------
    faults : iterable of Fault, optional
        Explicit faults to start with. Defaults to none.

    Examples
    --------
    >>> from silk.faults import FaultSet, stuck_open, w_variation
    >>> faults = FaultSet([stuck_open('m1')])
    >>> faults.sprinkle(r'.*', w_variation(sigma=0.02), rate=0.5)
    >>> sorted(faults.to_json())
    ['faults', 'sprinkles']
    >>> [f.fault_type for f in faults]
    ['stuck_open']
    """

    def __init__(self, faults: Iterable[Fault] = ()):
        self._faults: List[Fault] = list(faults)
        self._sprinkles: List[_SprinkleSpec] = []
        self._last_realized: Optional[List[Fault]] = None

    def add(self, fault: Fault) -> None:
        """Append an explicit fault.

        Parameters
        ----------
        fault : Fault
            The fault to add. It is appended without validation; an unknown
            target or fault type fails later, when the set is realized and
            applied.
        """
        self._faults.append(fault)

    def sprinkle(
        self,
        scope: str,
        fault: FaultFactory,
        rate: float,
        seed: Optional[int] = None,
    ) -> None:
        """Add a stochastic fault template, resolved at :meth:`realize` time.

        Parameters
        ----------
        scope : str
            Regular expression. A device path is matched when the whole
            path matches (``re.fullmatch``).
        fault : FaultFactory
            Template from a fault helper called without a device, e.g.
            ``stuck_open()`` or ``w_variation(sigma=0.02)``.
        rate : float
            Probability, per matched device path, that the fault is applied.
            A path is faulted when its uniform draw is below ``rate``.
        seed : int, optional
            ``seed=N`` makes this sprinkle's device selection reproducible
            regardless of the campaign rng (and of other sprinkles); a
            campaign seed then no longer varies it per run. ``None`` draws
            from the generator given to :meth:`realize`, so
            ``FaultCampaign(seed=...)`` governs it. Written by :meth:`to_json`.

        Raises
        ------
        TypeError
            ``fault`` is not a :class:`silk.faults.FaultFactory`.
        """
        if not isinstance(fault, FaultFactory):
            raise TypeError(
                'sprinkle() requires a FaultFactory — call a sugar function '
                'with zero positional args to get one, e.g. '
                'stuck_open() or w_variation(sigma=0.02).'
            )
        self._sprinkles.append(
            _SprinkleSpec(scope=scope, fault_factory=fault,
                          rate=float(rate), seed=seed))

    def realize(self, module, rng: np.random.Generator) -> List[Fault]:
        """Expand the sprinkles into the concrete fault list for one run.

        The result is the explicit faults followed by the sprinkled ones. For
        each sprinkle, in the order added, one uniform draw is taken for
        every device path that fully matches its scope, from ``rng`` or, for a
        sprinkle with a ``seed``, from that sprinkle's own generator (it does
        not draw from ``rng``); an
        invalid ``scope`` regular expression raises ``re.error`` here. The
        result is remembered for :meth:`faults_targeting`.

        Parameters
        ----------
        module : silk.circuit.Module
            Module whose device paths (``x<instance>.<device>`` hierarchy)
            the sprinkle scopes are matched against.
        rng : numpy.random.Generator
            Source of the draws; the same generator state gives the same
            faults. A sprinkle with a ``seed`` draws from its own generator
            instead and ignores this one.

        Returns
        -------
        list of Fault
            Explicit faults followed by the sprinkled faults.

        Raises
        ------
        AttributeError
            ``module`` does not provide the internal device-path enumeration
            that ``Module`` supplies.
        ValueError
            A declared sigma is not finite and nonnegative, a threshold
            delta is not finite, or a sprinkle rate is not finite and in
            [0, 1]. All declarations are checked before selection draws,
            including templates with no matches or a zero rate.
        """
        get_paths = getattr(module, '_silk_fault_all_device_paths', None)
        if get_paths is None:
            raise AttributeError(
                'FaultSet.realize() requires module to expose '
                '_silk_fault_all_device_paths(); silk Module provides '
                'this method.')
        all_paths = list(get_paths())

        for fault in self._faults:
            _validate_numeric_params(fault.fault_type, fault.params,
                                     f'target {fault.target!r}')
        for sprinkle in self._sprinkles:
            factory = sprinkle.fault_factory
            context = f'sprinkle scope {sprinkle.scope!r}'
            _validate_numeric_params(factory.fault_type, factory.params, context)
            rate = sprinkle.rate
            if not math.isfinite(rate) or not 0 <= rate <= 1:
                raise ValueError(
                    f'{factory.fault_type} {context}: rate={rate!r}; '
                    'expected finite and in [0, 1]')

        realized: List[Fault] = list(self._faults)
        for sprinkle in self._sprinkles:
            pattern = re.compile(sprinkle.scope)
            matches = [p for p in all_paths if pattern.fullmatch(p)]
            # A per-sprinkle seed draws from its own generator, so the
            # selection ignores the caller's rng state; seed=None draws
            # from the campaign rng.
            source = (np.random.default_rng(sprinkle.seed)
                      if sprinkle.seed is not None else rng)
            draws = source.random(len(matches))
            for path, draw in zip(matches, draws):
                if draw < sprinkle.rate:
                    realized.append(sprinkle.fault_factory(path))

        self._last_realized = realized
        return realized

    def faults_targeting(self, device_path: str) -> List[Fault]:
        """Return the faults aimed at one device.

        Uses the result of the last :meth:`realize` call. Before any
        realization, only the explicit faults are searched.

        Parameters
        ----------
        device_path : str
            Device path, compared with each fault's ``target`` for equality.

        Returns
        -------
        list of Fault
            The matching faults, possibly empty.
        """
        if self._last_realized is None:
            return [f for f in self._faults if f.target == device_path]
        return [f for f in self._last_realized if f.target == device_path]

    def __iter__(self):
        return iter(self._faults)

    def to_json(self) -> dict:
        """Return the set as a JSON-serializable dict.

        Returns
        -------
        dict
            ``{'faults': [...], 'sprinkles': [...]}``: each fault as
            ``target``, ``fault_type`` and ``params``; each sprinkle as
            ``scope``, ``rate``, ``seed``, ``fault_type`` and
            ``fault_params``. Realized faults are not included.
        """
        return {
            'faults': [
                {'target': f.target, 'fault_type': f.fault_type,
                 'params': dict(f.params)}
                for f in self._faults
            ],
            'sprinkles': [s.to_json() for s in self._sprinkles],
        }

    @classmethod
    def from_json(cls, d: dict) -> 'FaultSet':
        """Rebuild a set from the dict :meth:`to_json` returns.

        Parameters
        ----------
        d : dict
            Dict with optional ``'faults'`` and ``'sprinkles'`` lists; a
            missing key means none.

        Returns
        -------
        FaultSet
            The reconstructed set.

        Raises
        ------
        TypeError
            A fault entry has missing or unexpected fields.
        KeyError
            A sprinkle entry lacks a required key (``scope``, ``rate``,
            ``fault_type`` or ``fault_params``).
        """
        fs = cls([Fault(**ff) for ff in d.get('faults', [])])
        for s in d.get('sprinkles', []):
            fs._sprinkles.append(_SprinkleSpec.from_json(s))
        return fs
