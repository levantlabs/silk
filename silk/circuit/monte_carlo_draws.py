"""Drawn-parameter records and sensitivity ranking for Monte Carlo runs.

Pure functions, no run wiring. Two halves:

* **Reading the draws.** ngspice prints each instance's drawn model
  parameters through ``showmod`` (:func:`showmod_lines` builds the control
  lines, :func:`parse_showmod` reads the log); Xyce writes one ``.res`` file
  per sampling run with each device's *normalised* draw per parameter
  (:func:`parse_xyce_res`). Both return values keyed by
  ``(designer device name, parameter)``.
* **Ranking a metric against them.** :func:`rank_sensitivity` standardises
  every drawn column across the samples, so ngspice's actual values and
  Xyce's normalised draws rank alike, and fits the metric by least squares
  on the standardised draws .

Which parameters are recorded per device type comes from the PDK manifest,
never from this module; nothing here names a PDK.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import NamedTuple, Optional

import numpy as np

from silk.exceptions import SimulationError
from silk.metrics.diagnostics import emit


# ---------------------------------------------------------------------------
# ngspice: showmod
# ---------------------------------------------------------------------------

def showmod_lines(devices: list[tuple[str, str, tuple[str, ...]]]) -> list[str]:
    """Build one ngspice ``showmod`` control line per device.

    ``devices`` holds ``(emitted instance path, model name, params)``.
    Single-quote the complete selector so angle brackets remain literal:
    ``showmod 'm.<path>.m<model>' : <params>``. Unlike ``save``, ``showmod``
    retains double quotes in its lookup. Refuse single quotes in a selector
    rather than producing an ambiguous command. The returned order is the
    order :func:`parse_showmod` matches blocks in.
    """
    lines = []
    for path, model, params in devices:
        if not params:
            raise ValueError(
                f'showmod_lines: device {path!r} has no parameters to record')
        from silk.utils.deck_text import card_text
        # Instance path, internal model name and record names are written
        # into a control-block command (TASK-593).
        selector = card_text(f'm.{path}.m{model}', who='showmod selector')
        params = [card_text(p, who=f'showmod record parameter of {path!r}')
                  for p in params]
        if "'" in selector:
            raise ValueError(
                f'showmod_lines: selector {selector!r} contains a single quote; '
                'rename the instance or internal model without single quotes '
                'before recording draws')
        lines.append(f"showmod '{selector}' : {' '.join(params)}")
    return lines


_SHOWMOD_MODEL = re.compile(r'^\s*model\s+\S+\s*$')
_SHOWMOD_VALUE = re.compile(r'^\s*(\S+)\s+(\S+)\s*$')


def parse_showmod(log_text: str,
                  requested: list[tuple[str, tuple[str, ...]]],
                  ) -> dict[tuple[str, str], float]:
    """Read ``showmod`` blocks from an ngspice log.

    ``requested`` holds ``(designer name, params)`` in the order
    :func:`showmod_lines` emitted the lines. ngspice truncates the model
    name it prints (``model xm1:sky130_fd_pr__nfe``), so blocks are matched
    to requests by **order**, never by the printed name.

    Raises :class:`~silk.exceptions.SimulationError` when the log holds a
    different number of blocks than requested, or a block lacks a requested
    parameter or prints a value that is not a number.
    """
    blocks: list[dict[str, str]] = []
    in_block = False
    for line in log_text.splitlines():
        if not line.strip():
            # ngspice's -o log puts a blank line after each 'model' header.
            continue
        if _SHOWMOD_MODEL.match(line):
            blocks.append({})
            in_block = True
            continue
        m = _SHOWMOD_VALUE.match(line) if in_block else None
        if m is None:
            # Any other line (a model-family header, a later print) ends it.
            in_block = False
            continue
        blocks[-1][m.group(1).lower()] = m.group(2)
    if len(blocks) != len(requested):
        raise SimulationError(
            f'parse_showmod: the log holds {len(blocks)} showmod block(s) but '
            f'{len(requested)} were requested; the control block did not run '
            f'every showmod line, or the log is from a different deck')
    out: dict[tuple[str, str], float] = {}
    for block, (name, params) in zip(blocks, requested):
        for param in params:
            text = block.get(param.lower())
            if text is None:
                raise SimulationError(
                    f'parse_showmod: device {name!r} has no {param!r} in its '
                    f'showmod block (printed: {sorted(block)})')
            try:
                out[(name, param)] = float(text)
            except ValueError:
                raise SimulationError(
                    f'parse_showmod: device {name!r} parameter {param!r} '
                    f'printed {text!r}, not a number') from None
    return out


# ---------------------------------------------------------------------------
# Xyce: .res
# ---------------------------------------------------------------------------

def parse_xyce_res(path: str, name_map: dict[str, str],
                   ) -> dict[tuple[str, str], float]:
    """Read one sample's normalised draws from a Xyce ``.res`` file.

    Columns look like ``XOFF:XM1:SKY130_FD_PR__NFET_01V8__MODEL.12:VTH0``:
    an instance prefix, the model, the parameter. ``name_map`` maps the
    instance prefix (``'XOFF:XM1'``, case-insensitive) to the designer name;
    the result is keyed ``(designer name, lower-case parameter)``.

    Columns whose prefix is not in ``name_map`` are left out (the caller
    decides which devices it records). Raises
    :class:`~silk.exceptions.SimulationError` when the file does not hold
    exactly one sample row, or a mapped device has no column.
    """
    with open(path) as fh:
        lines = [ln.split() for ln in fh if ln.strip()]
    if not lines or lines[0][0].upper() != 'STEP':
        raise SimulationError(
            f'parse_xyce_res: {path} has no STEP header line')
    header = lines[0][1:]
    rows = [ln for ln in lines[1:] if ln[0].lstrip('+-').isdigit()]
    if len(rows) != 1:
        raise SimulationError(
            f'parse_xyce_res: {path} holds {len(rows)} sample row(s); each '
            f'Monte Carlo sample deck must run exactly one sample')
    row = rows[0][1:]
    if len(row) != len(header):
        raise SimulationError(
            f'parse_xyce_res: {path} row has {len(row)} values for '
            f'{len(header)} columns')
    upper_map = {k.upper(): v for k, v in name_map.items()}
    out: dict[tuple[str, str], float] = {}
    seen: set[str] = set()
    for column, text in zip(header, row):
        parts = column.split(':')
        if len(parts) < 3:
            continue
        prefix = ':'.join(parts[:-2]).upper()
        if prefix not in upper_map:
            continue
        seen.add(prefix)
        out[(upper_map[prefix], parts[-1].lower())] = float(text)
    missing = sorted(set(upper_map) - seen)
    if missing:
        raise SimulationError(
            f'parse_xyce_res: {path} has no column for {missing}; '
            f'columns present: {header}')
    return out


# ---------------------------------------------------------------------------
# Tables and ranking
# ---------------------------------------------------------------------------

@dataclass
class DrawTable:
    """Each sample's drawn parameters, aligned to ``columns``.

    ``columns`` are ``(designer device name, param)``; ``values`` maps a
    sample key to an array aligned to ``columns``. ``raw`` is True for
    actual parameter values (ngspice), False for normalised draws (Xyce).
    ``skipped_types`` names device types that had no manifest ``record``
    entry and so were not recorded.
    """
    columns: tuple[tuple[str, str], ...]
    values: dict
    raw: bool
    skipped_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.columns = tuple(tuple(c) for c in self.columns)
        self.values = {k: np.asarray(v, dtype=float)
                       for k, v in self.values.items()}
        for key, arr in self.values.items():
            if arr.shape != (len(self.columns),):
                raise ValueError(
                    f'DrawTable: sample {key!r} has shape {arr.shape}, '
                    f'expected ({len(self.columns)},) to match columns')


class SensitivityRow(NamedTuple):
    """One ranked parameter (``'st1.MN:vth0'``) or device (``'st1.MN'``)."""
    name: str
    effect_per_sigma: Optional[float]
    share: Optional[float]
    correlation: float


@dataclass
class Sensitivity:
    """A metric's ranking against the drawn parameters.

    ``method`` is ``'least_squares'`` or ``'correlation'`` (the fallback
    when parameters are not fewer than samples). ``rows`` are sorted by
    ``share``, or by ``|correlation|`` in the fallback. ``correlated`` holds
    the named operating-point quantities, ranked by correlation only.
    ``n`` is the number of samples and ``p`` the number of drawn columns
    that vary across them (the ones the fit uses).
    """
    metric: str
    method: str
    n: int
    p: int
    r_squared: Optional[float]
    rows: list[SensitivityRow]
    correlated: list[SensitivityRow]

    def summary(self, top: int = 3, *, scale: float = 1.0,
                unit: str = '') -> str:
        """The first *top* rows as text, laid out for the ranking method.

        ``'least_squares'`` shows each row's effect per sigma and share, then
        the fit's R squared; ``'correlation'`` (the fallback) shows each
        row's correlation and says why there are no shares. *scale* and
        *unit* express the effect in other units (``scale=1e3, unit=' mV'``
        for a metric in volts); the metric's own units by default.

        Raises:
            ValueError: *top* is not a positive int.
        """
        if isinstance(top, bool) or not isinstance(top, int) or top < 1:
            raise ValueError(
                f'Sensitivity.summary: top must be a positive int, got {top!r}')
        lines = []
        if self.method == 'least_squares':
            for row in self.rows[:top]:
                effect = ('n/a' if row.effect_per_sigma is None
                          else f'{row.effect_per_sigma * scale:+.3f}{unit}')
                share = ('n/a' if row.share is None
                         else f'{row.share:.1%}')
                lines.append(f'  {row.name:<14} effect/sigma {effect}'
                             f'  share {share}')
            r2 = ('n/a' if self.r_squared is None
                  else f'{self.r_squared:.4f}')
            lines.append(f'  (least-squares fit R^2 = {r2})')
        else:
            for row in self.rows[:top]:
                lines.append(f'  {row.name:<14} correlation '
                             f'{row.correlation:+.3f}')
            lines.append('  (shares need more samples than recorded '
                         'parameters; ranked by correlation only)')
        return '\n'.join(lines)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float('nan')
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return float('nan')
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _abs_key(value: float) -> float:
    return -abs(value) if np.isfinite(value) else float('inf')


def _finite_vector(name: str, data: dict, keys: list) -> np.ndarray:
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(
            f'rank_sensitivity: {name} has no value for sample(s) {missing[:5]}')
    vec = np.array([float(data[k]) for k in keys])
    bad = [k for k, v in zip(keys, vec) if not np.isfinite(v)]
    if bad:
        raise ValueError(
            f'rank_sensitivity: {name} is not finite for sample(s) {bad[:5]}; '
            f'exclude failed and nan samples before ranking')
    return vec


def rank_sensitivity(metric: str, values: dict, draws: DrawTable, *,
                     by: str = 'parameter',
                     correlated: Optional[dict] = None) -> Sensitivity:
    """Rank ``metric`` against the drawn parameters.

    ``values`` maps sample key to metric value, with failed and nan samples
    already excluded by the caller; every key must be in ``draws.values``.
    Each drawn column is standardised across the samples and the metric is
    fitted with an intercept by least squares. Per column the row carries
    the effect per σ (the fitted coefficient, in the metric's units), its
    share of the explained variance (``β²/Σβ²``, the draws being
    independent) and its plain correlation with the metric; rows are sorted
    by share. ``by='device'`` sums each device's shares, reports the
    root-sum-square of its effects, and the correlation of the metric with
    the device's fitted contribution.

    A column that does not vary across the samples (a device whose mismatch
    is off) cannot be standardised: it stays out of the fit and reports
    effect ``None``, share 0 and correlation nan. A metric that is the same
    in every sample has nothing to share out: effects 0, shares and
    ``r_squared`` nan, and one ``MetricWarning``.

    When the varying parameters are not fewer than the samples the fit is
    underdetermined (the intercept is one more unknown), so the ranking
    falls back to correlation only, with every share and effect ``None``,
    ``r_squared`` ``None``, and one ``MetricWarning`` stating n and p.

    ``correlated`` maps a named operating-point quantity to
    ``{sample key: value}``; each is ranked by correlation only, since it is
    an effect of the draws, not an independent input.

    Raises ``ValueError`` for caller errors: an unknown ``by``, no samples,
    a sample with no draws, or a non-finite metric or quantity value.
    """
    if by not in ('parameter', 'device'):
        raise ValueError(
            f"rank_sensitivity: by must be 'parameter' or 'device', got {by!r}")
    keys = list(values)
    if not keys:
        raise ValueError(f'rank_sensitivity: no samples for metric {metric!r}')
    missing = [k for k in keys if k not in draws.values]
    if missing:
        raise ValueError(
            f'rank_sensitivity: sample(s) {missing[:5]} have no recorded draws')
    y = _finite_vector(f'metric {metric!r}', values, keys)
    x = np.vstack([draws.values[k] for k in keys]) if draws.columns else \
        np.zeros((len(keys), 0))
    if not np.all(np.isfinite(x)):
        raise ValueError('rank_sensitivity: the draw table holds non-finite values')
    n, p = x.shape
    sd = x.std(axis=0)
    varying = np.ptp(x, axis=0) > 0   # exact: std of a constant is not 0.0
    names = [f'{d}:{q}' for d, q in draws.columns]
    devices = [d for d, _ in draws.columns]
    p_fit = int(varying.sum())

    corr = np.full(p, np.nan)
    z = np.zeros_like(x)
    z[:, varying] = (x[:, varying] - x[:, varying].mean(axis=0)) / sd[varying]
    for j in np.flatnonzero(varying):
        corr[j] = _corr(z[:, j], y)

    effect: list[Optional[float]] = [None] * p
    share: list[Optional[float]] = [None] * p
    r_squared: Optional[float] = None
    if p_fit >= n:
        method = 'correlation'
        emit(metric='rank_sensitivity', code='more_parameters_than_samples',
             message=(f'{metric}: n={n} samples for p={p_fit} varying '
                      f'parameters; ranking by correlation only, no shares'),
             details={'n': n, 'p': p_fit},
             suggestion='Run more samples than recorded parameters for '
                        'effects and shares.')
    elif np.ptp(y) == 0:
        # A constant metric has no variance to share out; a fit would only
        # rank round-off. Effects are 0, shares and R² are undefined.
        method = 'least_squares'
        r_squared = float('nan')
        emit(metric='rank_sensitivity', code='constant_metric',
             message=f'{metric}: the metric is the same in all {n} samples; '
                     f'no variance to rank',
             details={'n': n, 'value': float(y[0])})
        for j in range(p):
            effect[j] = 0.0 if varying[j] else None
            share[j] = float('nan') if varying[j] else 0.0
    else:
        method = 'least_squares'
        a = np.column_stack([z[:, varying], np.ones(n)])
        coef, *_ = np.linalg.lstsq(a, y, rcond=None)
        beta = coef[:-1]
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        ss_res = float(np.sum((y - a @ coef) ** 2))
        r_squared = 1.0 - ss_res / ss_tot
        total = float(np.sum(beta ** 2))
        for j, b in zip(np.flatnonzero(varying), beta):
            effect[j] = float(b)
            share[j] = float(b ** 2 / total) if total > 0 else float('nan')
        for j in np.flatnonzero(~varying):
            share[j] = 0.0

    if by == 'parameter':
        rows = [SensitivityRow(names[j], effect[j], share[j], float(corr[j]))
                for j in range(p)]
    else:
        rows = []
        for dev in dict.fromkeys(devices):
            idx = [j for j in range(p) if devices[j] == dev]
            if method == 'least_squares':
                fitted = [j for j in idx if effect[j] is not None]
                eff = (float(np.sqrt(sum(effect[j] ** 2 for j in fitted)))
                       if fitted else None)
                contrib = sum((z[:, j] * effect[j] for j in fitted),
                              np.zeros(n))
                rows.append(SensitivityRow(
                    dev, eff, float(sum(share[j] for j in idx)),
                    _corr(contrib, y)))
            else:
                best = min(idx, key=lambda j: _abs_key(corr[j]))
                rows.append(SensitivityRow(dev, None, None, float(corr[best])))

    if method == 'least_squares':
        rows.sort(key=lambda r: -r.share if np.isfinite(r.share) else float('inf'))
    else:
        rows.sort(key=lambda r: _abs_key(r.correlation))

    corr_rows = []
    for qname, qvalues in (correlated or {}).items():
        q = _finite_vector(f'quantity {qname!r}', qvalues, keys)
        corr_rows.append(SensitivityRow(qname, None, None, _corr(q, y)))
    corr_rows.sort(key=lambda r: _abs_key(r.correlation))

    return Sensitivity(metric=metric, method=method, n=n, p=p_fit,
                       r_squared=r_squared, rows=rows, correlated=corr_rows)
