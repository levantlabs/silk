"""Device-level STR-001 coverage, independent of electrical failure."""
from dataclasses import dataclass, field


@dataclass
class DeviceStressCoverage:
    """One device's measured pairs; failure can overlap partial coverage.

    Attributes
    ----------
    instance_path : str
        Hierarchical path of the device.
    type_name : str
        Device type name used to look up its ratings.
    failed : bool
        ``True`` if any rated pair violated its limit.
    coverage : str
        ``'full'`` (every pair checked), ``'partial'`` (some checked) or
        ``'none'`` (no rated pair checked).
    reasons : tuple of str
        Reason codes for pairs or devices that could not be checked, for
        example ``'no_waveform'`` or ``'small_signal'``. Empty when nothing
        was skipped.
    pairs : dict
        Maps each terminal-pair name to its status: ``'pass'`` or ``'fail'``
        for a checked pair, ``'unavailable'`` for a rated pair that could
        not be checked, and ``'unrated'`` for a pair with no rating.
    """
    instance_path: str
    type_name: str
    failed: bool
    coverage: str
    reasons: tuple[str, ...]
    pairs: dict[str, str]

    def to_dict(self) -> dict:
        """Return a JSON-ready dict of the fields.

        Returns
        -------
        dict
            Keys ``instance_path``, ``type_name``, ``failed``, ``coverage``,
            ``reasons`` (as a list) and ``pairs`` (a copy).
        """
        return dict(instance_path=self.instance_path, type_name=self.type_name,
                    failed=self.failed, coverage=self.coverage,
                    reasons=list(self.reasons), pairs=dict(self.pairs))


@dataclass
class StressCoverage:
    """Coverage partition (full + partial + none = total), plus failed.

    ``failed`` overlaps full/partial: it counts devices with any failing rated
    pair, not another coverage category. ``none`` means no rated pair checked.
    The counts are read-only properties computed from ``devices``.

    Attributes
    ----------
    devices : list of DeviceStressCoverage
        One entry per device the junction-stress rule (STR-001) considered.
        Empty by default.
    """
    devices: list[DeviceStressCoverage] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Number of devices considered.

        Returns
        -------
        int
            ``len(devices)``.
        """
        return len(self.devices)

    @property
    def full(self) -> int:
        """Devices whose every rated pair was checked.

        Returns
        -------
        int
            Count of devices whose ``coverage`` is 'full'.
        """
        return sum(d.coverage == 'full' for d in self.devices)

    @property
    def partial(self) -> int:
        """Devices with some, but not all, pairs checked.

        Returns
        -------
        int
            Count of devices whose ``coverage`` is 'partial'.
        """
        return sum(d.coverage == 'partial' for d in self.devices)

    @property
    def none(self) -> int:
        """Devices with no rated pair checked.

        Returns
        -------
        int
            Count of devices whose ``coverage`` is 'none'.
        """
        return sum(d.coverage == 'none' for d in self.devices)

    @property
    def failed(self) -> int:
        """Devices with at least one failing rated pair.

        Returns
        -------
        int
            Count of devices whose ``failed`` flag is true. Overlaps the
            ``full`` and ``partial`` counts.
        """
        return sum(d.failed for d in self.devices)

    def summary(self) -> dict:
        """Return the counts as a dict.

        Returns
        -------
        dict
            Maps ``'total'``, ``'full'``, ``'partial'``, ``'none'`` and
            ``'failed'`` to their integer counts.
        """
        return {key: getattr(self, key) for key in ('total', 'full', 'partial', 'none', 'failed')}

    def to_dict(self) -> dict:
        """Return a JSON-ready dict of the summary and the devices.

        Returns
        -------
        dict
            ``{'summary': <see summary>, 'devices': [<device dicts>]}``; the
            shape ``LintReport`` and ``CoverageReport`` store as
            ``stress_coverage``.
        """
        return {'summary': self.summary(), 'devices': [d.to_dict() for d in self.devices]}


def stress_coverage_text(data: dict) -> str:
    """Shared terminal/HTML/Markdown wording for serialized coverage."""
    if not data:
        return ''
    counts = data['summary']
    status = ('FAIL' if counts['failed'] else
              'INCOMPLETE' if counts['partial'] or counts['none'] else
              'NO DEVICES' if not counts['total'] else 'PASS')
    text = (f"STR-001 {status}: {counts['full']}/{counts['total']} devices fully checked; "
            f"{counts['partial']} partial, {counts['none']} unchecked, "
            f"{counts['failed']} failed")
    details = [f"{d['instance_path']}: {', '.join(d['reasons'])}"
               for d in data['devices'] if d['reasons']]
    return text + ('; ' + '; '.join(details) if details else '')
