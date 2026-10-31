"""Data classes for silk's fault-injection framework.

Three types live here:

* ``Fault``: declarative, "apply ``fault_type`` to ``target`` with ``params``".
* ``AppliedFault``: realized, what was actually emitted after random draws.
* ``FaultManifest``: per-run record of the realized faults and the input spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional
from silk.utils._utils import _actual_values


@dataclass(frozen=True)
class Fault:
    """Declarative fault description: pure data, nothing is executed.

    Build one with the helper functions (:func:`silk.faults.stuck_open`,
    :func:`silk.faults.w_variation`, ...) rather than by hand.

    Attributes
    ----------
    target : str
        Hierarchical device path, e.g. ``'xcomp1.mn_tail'``.
    fault_type : str
        One of the fault-type names: ``'stuck_open'``, ``'short_ds'``,
        ``'short_gs'``, ``'short_gd'``, ``'w_variation'``,
        ``'l_variation'``, ``'vth_shift'``. An unknown name is accepted here
        and raises :class:`ValueError` when the fault is realized.
    params : dict
        Fault-specific parameters, e.g. ``{'sigma': 0.02}`` for
        ``w_variation`` (fractional) or ``{'delta': 0.03}`` (volts) for
        ``vth_shift``. The ``stuck_open`` and short helpers store
        ``{'per_finger': True}`` by default. Defaults to an empty dict.

    Examples
    --------
    >>> from silk.faults import Fault
    >>> Fault('xcomp.mn1', 'vth_shift', {'delta': 0.03}).fingerprint()
    'vth_shift@xcomp.mn1:delta=0.03'
    """
    target: str
    fault_type: str
    params: dict = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Return a stable identity string for this fault.

        Returns
        -------
        str
            ``'<fault_type>@<target>:<k>=<v>,...'`` with the parameters sorted
            by key, so two equal faults give the same string regardless of
            dict insertion order.
        """
        items = sorted(self.params.items())
        params_str = ','.join(f'{k}={v}' for k, v in items)
        return f'{self.fault_type}@{self.target}:{params_str}'


@dataclass(frozen=True)
class AppliedFault:
    """A fault after random expansion: what was actually emitted.

    ``declared_params`` preserves the request as the caller wrote it;
    ``realized_params`` holds the concrete values the run drew (for a
    ``w_variation``, keys such as ``w_multiplier``, ``w_nominal`` and
    ``w_new``). The keys depend on the fault type.

    Attributes
    ----------
    target : str
        Hierarchical device path the fault was applied to.
    fault_type : str
        Fault-type name, the same vocabulary as :attr:`silk.faults.Fault.fault_type`.
    declared_params : dict
        The parameters the caller declared on the :class:`silk.faults.Fault`.
    realized_params : dict
        The concrete values the run used, after any random draw. Voltage
        shifts are in volts. ``w_multiplier`` and ``l_multiplier`` are
        dimensionless; only the ``_nominal`` and ``_new`` width and length
        values are lengths, in the units the circuit was written in
        (micrometres under the shipped author-unit defaults).
    netlist_comment : str
        The ``*** FAULT: ...`` comment line written next to the device in the
        generated netlist.

    Examples
    --------
    >>> from silk.faults import AppliedFault
    >>> applied = AppliedFault('m1', 'stuck_open', {'per_finger': True},
    ...                        {'m_before': 1, 'm_after': 0, 'deleted': True},
    ...                        '*** FAULT: stuck_open on m1 (was m=1, deleted)')
    >>> applied.realized_params['deleted']
    True
    """
    target: str
    fault_type: str
    declared_params: dict
    realized_params: dict
    netlist_comment: str


@dataclass
class FaultManifest:
    """Per-run record of what was faulted.

    ``input_spec`` is the declarative :class:`silk.faults.FaultSet` with its
    sprinkles still unresolved. It is stored as the JSON-serializable dict
    from :meth:`silk.faults.FaultSet.to_json` rather than a ``FaultSet``, so the manifest
    does not depend on ``FaultSet``.

    Attributes
    ----------
    seed : int
        Seed the run's random generator was created from.
    input_spec : dict
        The fault set that was requested, as returned by
        :meth:`silk.faults.FaultSet.to_json`.
    applied : list of AppliedFault
        What the netlister actually emitted, in emission order. Starts empty
        and is filled while the netlist is generated. Faults that
        resolve to a device already faulted in the same run are not recorded
        (see :meth:`silk.faults.FaultSet.realize`).

    Examples
    --------
    >>> from silk.faults import FaultManifest
    >>> manifest = FaultManifest(seed=7, input_spec={'faults': [], 'sprinkles': []})
    >>> FaultManifest.from_json(manifest.to_json()).seed
    7
    >>> print(manifest.summary())
    Seed: 7  Applied faults: 0
    """
    seed: int
    input_spec: dict
    applied: List[AppliedFault] = field(default_factory=list)

    def summary(self) -> str:
        """Return a short text summary of the run.

        Returns
        -------
        str
            A ``Seed: ... Applied faults: ...`` header line followed by one
            line per applied fault giving its type and target.
        """
        lines = [f'Seed: {self.seed}  Applied faults: {len(self.applied)}']
        for af in self.applied:
            lines.append(f'  {af.fault_type:<14} {af.target}')
        return '\n'.join(lines)

    def to_json(self) -> dict:
        """Return the manifest as a JSON-serializable dict.

        Returns
        -------
        dict
            Keys ``'seed'``, ``'input_spec'`` and ``'applied'``; each applied
            fault becomes a dict with its five fields. Any
            :class:`silk.circuit.Variable` among the recorded parameters is
            replaced by its number.
        """
        # Recorded parameters are numbers, never a Variable (TASK-402).
        return _actual_values({
            'seed': self.seed,
            'input_spec': self.input_spec,
            'applied': [
                {'target': af.target, 'fault_type': af.fault_type,
                 'declared_params': af.declared_params,
                 'realized_params': af.realized_params,
                 'netlist_comment': af.netlist_comment}
                for af in self.applied
            ],
        })

    @classmethod
    def from_json(cls, d: dict) -> 'FaultManifest':
        """Rebuild a manifest from the dict :meth:`to_json` returns.

        Parameters
        ----------
        d : dict
            A dict with keys ``'seed'``, ``'input_spec'`` and ``'applied'``
            (a list of dicts holding the five :class:`silk.faults.AppliedFault` fields).

        Returns
        -------
        FaultManifest
            The reconstructed manifest.

        Raises
        ------
        KeyError
            A required key is missing.
        TypeError
            An applied entry has missing or unexpected fields.
        """
        return cls(
            seed=d['seed'],
            input_spec=d['input_spec'],
            applied=[AppliedFault(**a) for a in d['applied']],
        )
