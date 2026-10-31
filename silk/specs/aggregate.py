"""Immutable attribution for a selected Monte Carlo statistic.

One codec serves database rows and CircuitReport JSON. This is attribution,
not an archive of raw samples or a reconstruction of a simulation.
"""
from dataclasses import asdict, dataclass, fields

SCHEME = 'silk.measurement.aggregate/v1'


@dataclass(frozen=True)
class AggregateMeasurement:
    """Which Monte Carlo samples a statistic was computed from.

    Carried on :attr:`Measurement.aggregate <silk.specs.Measurement.aggregate>`
    when the measurement's ``value`` is a statistic over a campaign (as
    returned by a Monte Carlo report's ``measurement()``), so that the value
    can be traced back to the campaign, corner and samples. It is attribution
    only: it holds no sample values. The instance is frozen and is validated
    at construction.

    The counts must describe one inventory of selected observations:
    ``samples_total == len(sample_seeds)``, ``samples_used == len(used_indices)``
    and ``samples_total == samples_used + samples_failed + samples_unavailable``.

    Parameters
    ----------
    metric : str
        Name of the aggregated metric; non-empty.
    statistic : {'mean', 'rms', 'std', 'min', 'max'}
        The statistic computed over the usable samples.
    variation : {'mismatch', 'process', 'both'}
        Which variation the campaign drew.
    selected_corner : str or None
        Corner the statistic was taken at, or ``None`` when the campaign
        recorded none.
    samples_total : int
        Number of selected observations.
    samples_used : int
        Observations that contributed to the statistic.
    samples_failed : int
        Selected observations whose run failed.
    samples_unavailable : int
        Observations that ran but gave no usable value for the metric.
    used_indices : tuple of int
        Sample indices (each in ``[0, requested_samples)``) that contributed;
        a subset of the indices in ``sample_seeds``. Any iterable of ints is
        accepted and stored as a tuple.
    sample_seeds : tuple of tuple of int
        ``(sample_index, seed)`` pair for each selected observation; indices
        are unique. Stored as a tuple of tuples.
    campaign_id : str
        Identifier of the campaign; non-empty. When the aggregate is attached
        to a :class:`~silk.specs.Measurement`, that measurement's
        ``sim_run_id`` must equal it.
    base_seed : int
        Seed the campaign's per-sample seeds derive from.
    requested_samples : int
        Number of samples the campaign requested; at least 1.
    structure_hash : str, optional
        Structure hash of the circuit the campaign ran, when known.
    draw_scheme : int, optional
        Version of the random-draw scheme (positive) used by a behavioural
        campaign; ``None`` when not applicable.

    Raises
    ------
    ValueError
        A field is outside the ranges above, the counts do not agree with the
        index and seed inventories, or sample indices repeat.
    TypeError
        *selected_corner* or *structure_hash* is neither a string nor ``None``.
    """
    #: Name of the aggregated metric.
    metric: str
    #: Statistic computed, one of ``'mean'``, ``'rms'``, ``'std'``, ``'min'`` or ``'max'``.
    statistic: str
    #: Variation drawn, one of ``'mismatch'``, ``'process'`` or ``'both'``.
    variation: str
    #: Corner the statistic was taken at, or ``None``.
    selected_corner: str | None
    #: Number of selected observations.
    samples_total: int
    #: Observations that contributed to the statistic.
    samples_used: int
    #: Selected observations whose run failed.
    samples_failed: int
    #: Observations without a usable value for the metric.
    samples_unavailable: int
    #: Indices of the contributing samples.
    used_indices: tuple[int, ...]
    #: ``(sample_index, seed)`` pairs of the selected observations.
    sample_seeds: tuple[tuple[int, int], ...]
    #: Identifier of the campaign.
    campaign_id: str
    #: Seed the per-sample seeds derive from.
    base_seed: int
    #: Number of samples the campaign requested.
    requested_samples: int
    #: Structure hash of the simulated circuit, when known.
    structure_hash: str | None = None
    #: Version of the behavioural random-draw scheme, or ``None``.
    draw_scheme: int | None = None

    def __post_init__(self):
        for name in ('metric', 'campaign_id'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f'aggregate {name} must be a nonempty string')
        for name in ('selected_corner', 'structure_hash'):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f'aggregate {name} must be a string or None')
        if self.statistic not in ('mean', 'rms', 'std', 'min', 'max'):
            raise ValueError('aggregate statistic must be mean/rms/std/min/max')
        if self.variation not in ('mismatch', 'process', 'both'):
            raise ValueError('aggregate variation must be mismatch/process/both')
        for name in ('samples_total', 'samples_used', 'samples_failed',
                     'samples_unavailable', 'requested_samples'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f'aggregate {name} must be a nonnegative integer')
        if self.requested_samples < 1:
            raise ValueError('aggregate requested_samples must be positive')
        if type(self.base_seed) is not int:
            raise ValueError('aggregate base_seed must be an integer')
        if self.draw_scheme is not None and (type(self.draw_scheme) is not int or self.draw_scheme < 1):
            raise ValueError('aggregate draw_scheme must be a positive integer or None')
        used = tuple(self.used_indices)
        seeds = tuple(tuple(pair) for pair in self.sample_seeds)
        if any(type(i) is not int or not 0 <= i < self.requested_samples for i in used):
            raise ValueError('aggregate used_indices must contain valid sample indices')
        if any(len(pair) != 2 or type(pair[0]) is not int or
               not 0 <= pair[0] < self.requested_samples or type(pair[1]) is not int
               for pair in seeds):
            raise ValueError('aggregate sample_seeds must contain index/seed pairs')
        indices = {pair[0] for pair in seeds}
        if len(indices) != len(seeds) or len(set(used)) != len(used):
            raise ValueError('aggregate sample indices must be unique')
        if not set(used) <= indices:
            raise ValueError('aggregate used_indices must be a subset of selected observations')
        if self.samples_used != len(used) or self.samples_total != len(seeds):
            raise ValueError('aggregate counts must match the selected observation inventory')
        if self.samples_total != self.samples_used + self.samples_failed + self.samples_unavailable:
            raise ValueError('aggregate counts must partition selected observations')
        object.__setattr__(self, 'used_indices', used)
        object.__setattr__(self, 'sample_seeds', seeds)


def encode_aggregate(value):
    """Return the shared JSON object, or None for an ordinary observation."""
    if value is None:
        return None
    if not isinstance(value, AggregateMeasurement):
        raise TypeError('aggregate must be an AggregateMeasurement or None')
    # Validate again at the storage boundary (including unpickled evidence).
    record = AggregateMeasurement(**asdict(value))
    result = asdict(record)
    result['used_indices'] = list(record.used_indices)
    result['sample_seeds'] = [list(pair) for pair in record.sample_seeds]
    return {'scheme': SCHEME, **result}


def decode_aggregate(payload):
    """Validate a stored aggregate; never silently discard malformed evidence."""
    if payload is None:
        return None
    if not isinstance(payload, dict) or payload.get('scheme') != SCHEME:
        raise ValueError(f'aggregate scheme must be {SCHEME!r}')
    names = {f.name for f in fields(AggregateMeasurement)}
    if set(payload) != names | {'scheme'}:
        raise ValueError('aggregate payload has missing or unknown fields')
    if not isinstance(payload['used_indices'], list) or not isinstance(payload['sample_seeds'], list):
        raise ValueError('aggregate used_indices and sample_seeds must be JSON arrays')
    if any(not isinstance(pair, list) for pair in payload['sample_seeds']):
        raise ValueError('aggregate sample_seeds entries must be JSON arrays')
    try:
        return AggregateMeasurement(**{name: payload[name] for name in names})
    except (TypeError, ValueError) as exc:
        raise ValueError(f'malformed aggregate: {exc}') from exc


def validate_attribution(aggregate, metric, sim_run_id):
    """An explicit stored declaration and run must own this aggregate."""
    encode_aggregate(aggregate)
    if aggregate is not None:
        if aggregate.metric != metric:
            raise ValueError(f'aggregate metric {aggregate.metric!r} does not match {metric!r}')
        if aggregate.campaign_id != sim_run_id:
            raise ValueError('aggregate campaign_id must match measurement sim_run_id')
