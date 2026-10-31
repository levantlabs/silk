"""Report section for a Monte Carlo run.

:func:`build_section` turns a :class:`~silk.circuit.monte_carlo.MonteCarloResults`
into a :class:`MonteCarloSection`: plain numbers (what ``save_json`` writes)
plus rendered PNG images (what ``save_html`` embeds). Nothing here simulates;
the sensitivity table appears only when the results carry a recorded
:class:`~silk.circuit.monte_carlo_draws.DrawTable` (``mc.draws``).
Grid reports use :func:`build_grid_section` and :func:`render_grid_section`
to present recorded spec statistics, per-point yields and planned samples.
"""
from __future__ import annotations

import base64
import html
import io
import math
import warnings
from dataclasses import dataclass, field

import numpy as np

from silk.circuit.metric_handle import _metric_key
from silk.circuit.monte_carlo import _usable, wilson_interval
from silk.circuit.monte_carlo_draws import rank_sensitivity
from silk.exceptions import MonteCarloError


@dataclass
class MonteCarloSection:
    """Numbers (``data``) and images (``images``) of one Monte Carlo run.

    ``images`` is a list of ``{'kind', 'title', 'png_b64'}``; it is not part
    of :meth:`to_dict`.
    """
    data: dict
    images: list = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return self.data

    @classmethod
    def from_dict(cls, d: dict) -> 'MonteCarloSection':
        return cls(data=d)


def _png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _spec_metrics(mc) -> list:
    return sorted({s.metric for s in mc._spec_declarations})


def _limits(mc, metric) -> list:
    return [{'op': op, 'target': float(target)}
            for op, target in mc._spec_limits(metric)]


def build_section(mc, *, metrics=None, top: int = 3) -> MonteCarloSection:
    """Summarise and plot ``mc``; see :meth:`CircuitReport.add_monte_carlo`."""
    from matplotlib.figure import Figure
    if not isinstance(top, int) or isinstance(top, bool) or top < 1:
        raise ValueError(f'add_monte_carlo: top must be a positive int, '
                         f'got {top!r}')
    names = (_spec_metrics(mc) if metrics is None
             else [_metric_key(m) for m in metrics])
    keys = mc._keys()
    corners = sorted({k[0] for k in keys})
    s = mc.settings

    data = {
        'settings': {'n': s.n, 'variation': s.variation, 'seed': s.seed,
                     'record_draws': s.record_draws,
                     'min_yield': s.min_yield},
        'passed': bool(mc.passed),
        'corners': [], 'yields': [], 'metrics': [],
        'sensitivity': [], 'failed_samples': [],
    }
    images = []

    for c in corners:
        run = sum(1 for k in keys if k[0] == c)
        bad = sum(1 for k in mc.failed if k[0] == c)
        data['corners'].append({'corner': c, 'run': run, 'failed': bad,
                                'used': run - bad})
        passes = mc.corner(c)._spec_pass_counts()
        for spec, p in passes.items():
            lo, hi = wilson_interval(p, run)
            data['yields'].append({'spec': spec, 'corner': c, 'passed': p,
                                   'run': run, 'failed': bad,
                                   'used': run - bad,
                                   'fraction': p / run, 'lo': lo, 'hi': hi})

    with warnings.catch_warnings():
        # the run/failed/used counts are tabulated; skip warnings would
        # only repeat them once per rendered plot
        warnings.simplefilter('ignore')
        for m in names:
            try:
                vals = mc._metric_values(m)
            except ValueError as e:
                raise MonteCarloError(
                    f'add_monte_carlo: {e}') from None
            used = {k: float(v) for k, v in vals.items() if _usable(v)}
            data['metrics'].append({'metric': m, 'used': len(used),
                                    'limits': _limits(mc, m)})
            if used:
                fig = Figure(figsize=(5, 3.2))
                mc.plot_histogram(m, ax=fig.subplots())
                images.append({'kind': 'histogram', 'title': m,
                               'png_b64': _png_b64(fig)})
            draws = getattr(mc, 'draws', None)
            if draws is None:
                continue
            col = {f'{d}:{q}': j for j, (d, q) in enumerate(draws.columns)}
            for c in corners:
                cv = {k: v for k, v in used.items()
                      if k[0] == c and k in draws.values}
                if not cv:
                    continue
                sens = rank_sensitivity(m, cv, draws)
                data['sensitivity'].append({
                    'metric': m, 'corner': c, 'method': sens.method,
                    'n': sens.n, 'p': sens.p, 'r_squared': sens.r_squared,
                    'rows': [r._asdict() for r in sens.rows]})
                ranked = [r for r in sens.rows
                          if r.name in col and not (r.share == 0)
                          and r.correlation == r.correlation][:top]
                for r in ranked:
                    fig = Figure(figsize=(4, 3))
                    ax = fig.subplots()
                    ks = sorted(cv)
                    ax.scatter([draws.values[k][col[r.name]] for k in ks],
                               [cv[k] for k in ks], s=8)
                    ax.set_xlabel(r.name + ('' if draws.raw
                                            else ' (normalised draw)'))
                    ax.set_ylabel(m)
                    ax.set_title(f'{c}: r={r.correlation:.2f}')
                    images.append({'kind': 'scatter',
                                   'title': f'{m} vs {r.name} ({c})',
                                   'png_b64': _png_b64(fig)})

    for k in sorted(mc.failed):
        f = mc.failed[k]
        data['failed_samples'].append({'corner': k[0], 'sample': k[1],
                                       'seed': f.seed, 'error': f.error})
    return MonteCarloSection(data, images)


def _num(x, spec='.3f') -> str:
    return 'n/a' if x is None or (isinstance(x, float) and math.isnan(x)) \
        else format(x, spec)


def render_html(section: MonteCarloSection) -> str:
    """The section as an HTML fragment (images embedded as data URIs)."""
    e = lambda s: html.escape(str(s))
    d = section.data
    s = d['settings']
    out = ['<section class="monte-carlo"><h2>Monte Carlo</h2>']
    out.append(
        f'<p><span class="badge {"pass" if d["passed"] else "fail"}">'
        f'{"PASS" if d["passed"] else "FAIL"}</span> '
        f'n={s["n"]}, variation={e(s["variation"])}, seed={s["seed"]}'
        + (f', min yield {s["min_yield"]:g}' if s['min_yield'] else '')
        + '</p>')
    out.append('<h3>Yield</h3><table><thead><tr><th>Spec</th><th>Corner</th>'
               '<th>Yield</th><th>95% interval</th><th>Run</th>'
               '<th>Failed</th><th>Used</th></tr></thead><tbody>')
    for y in d['yields']:
        out.append(
            f'<tr><td>{e(y["spec"])}</td><td>{e(y["corner"])}</td>'
            f'<td>{_num(y["fraction"])}</td>'
            f'<td>[{_num(y["lo"])}, {_num(y["hi"])}]</td>'
            f'<td>{y["run"]}</td><td>{y["failed"]}</td>'
            f'<td>{y["used"]}</td></tr>')
    out.append('</tbody></table>')
    hist = [i for i in section.images if i['kind'] == 'histogram']
    if hist:
        out.append('<h3>Distributions</h3>')
        for i in hist:
            out.append(f'<p class="fig-title">{e(i["title"])}</p>'
                       f'<img src="data:image/png;base64,{i["png_b64"]}" '
                       f'alt="{e(i["title"])}">')
    if d['sensitivity']:
        out.append('<h3>Sensitivity</h3>')
        for sn in d['sensitivity']:
            out.append(
                f'<p><strong>{e(sn["metric"])}</strong> at {e(sn["corner"])}: '
                f'{e(sn["method"])}, n={sn["n"]}, p={sn["p"]}, '
                f'R&sup2;={_num(sn["r_squared"])}</p>'
                '<table><thead><tr><th>Parameter</th><th>Effect per &sigma;'
                '</th><th>Share</th><th>Correlation</th></tr></thead><tbody>')
            for r in sn['rows']:
                out.append(
                    f'<tr><td>{e(r["name"])}</td>'
                    f'<td>{_num(r["effect_per_sigma"], ".4g")}</td>'
                    f'<td>{_num(r["share"])}</td>'
                    f'<td>{_num(r["correlation"], ".2f")}</td></tr>')
            out.append('</tbody></table>')
        for i in section.images:
            if i['kind'] == 'scatter':
                out.append(f'<p class="fig-title">{e(i["title"])}</p>'
                           f'<img src="data:image/png;base64,{i["png_b64"]}" '
                           f'alt="{e(i["title"])}">')
    if d['failed_samples']:
        out.append('<h3>Failed samples</h3><table><thead><tr><th>Corner</th>'
                   '<th>Sample</th><th>Seed</th><th>Error</th></tr></thead>'
                   '<tbody>')
        for f in d['failed_samples']:
            out.append(f'<tr><td>{e(f["corner"])}</td><td>{f["sample"]}</td>'
                       f'<td>{f["seed"]}</td><td>{e(f["error"])}</td></tr>')
        out.append('</tbody></table>')
    out.append('</section>')
    return '\n'.join(out)


def build_grid_section(mc) -> MonteCarloSection:
    """Tabulate recorded grid statistics, sample evidence and collection yields."""
    from copy import copy
    from silk.circuit.report import _pvt_number, _pvt_point

    rows = mc.table()
    variables = mc._variable_names()
    variable_columns = ["run." + name for name in variables]
    points = mc.yield_by_point()
    pooled = mc.yield_fraction()
    minimum = mc.settings.min_yield
    floor = 1. if minimum is None else minimum
    requirement = ('every sample must pass' if minimum is None else
                   f'required yield {minimum:g} per point')
    sizes = [len(mc.point(point)._keys()) for point in points]
    per_point = (f'{sizes[0]} sample{"s" if sizes[0] != 1 else ""} per point'
                 if sizes and len(set(sizes)) == 1
                 else 'samples per point: ' + ', '.join(map(str, sizes)))
    failed = sum(row['run.status'] == 'failed' for row in rows)
    summary = (f'Summary: {len(rows)} sample{"s" if len(rows) != 1 else ""} at '
               f'{len(points)} operating point{"s" if len(points) != 1 else ""}, '
               f'{per_point}; {requirement}; {failed} failed and produced no report.')
    if mc._plan_summary():
        summary += ' ' + mc._plan_summary()
    def failure_counts(sample_rows, name):
        keys = [d['key'] for d in mc._check_declarations if d['name'] == name]
        missed = sum(row['run.status'] != 'failed' and
                     any(row.get('check.' + key) != 'PASS' for key in keys)
                     for row in sample_rows)
        absent = sum(row['run.status'] == 'failed' for row in sample_rows)
        return missed, absent

    yield_rows = []
    for point, yields in points.items():
        point_rows = mc.point(point).table()
        for name, (fraction, interval) in yields.items():
            missed, absent = failure_counts(point_rows, name)
            yield_rows.append([point[0], _pvt_number(point[1]), _pvt_number(point[2]),
                *(repr(dict(point[3])[label]) for label in variables), name,
                len(point_rows), missed, absent, _num(fraction, '#.3g'),
                '[' + ', '.join(_num(v, '#.3g') for v in interval) + ']',
                'PASS' if fraction >= floor else 'FAIL'])
    for name, (fraction, interval) in pooled.items():
        missed, absent = failure_counts(rows, name)
        yield_rows.append(['All points (informational)', '', '', *('' for _ in variables), name, len(rows), missed, absent,
            _num(fraction, '#.3g'), '[' + ', '.join(_num(v, '#.3g') for v in interval) + ']', '—'])

    metrics = mc._table_metric_names
    extrema = {}
    for metric in metrics:
        try:
            extrema[metric] = [mc.stat(metric, how) for how in ('min', 'max')]
        except (TypeError, ValueError):
            extrema[metric] = [None, None]
    spec_rows = []
    checked = set()
    for declaration in mc._check_declarations:
        metric, key = declaration['metric'], declaration['key']
        checked.add(metric)
        # Restrict worst() to this check, preserving its stable identity and direction.
        view = copy(mc)
        view.reports = {}
        for run, report in mc.reports.items():
            record = copy(report)
            record._results = [result for result in report.results
                if (result.spec_key or result.name) == key and
                (declaration.get('owner_path') is None or
                 result.owner_path == declaration['owner_path'])]
            view.reports[run] = record
        try:
            worst = view.worst(metric)
        except (TypeError, ValueError):
            worst = None
        value = worst.value if worst is not None and worst.corner is not None else None
        fractions = [yields[declaration['name']][0] for yields in points.values()
                     if declaration['name'] in yields]
        status = ('PASS' if fractions and all(f >= floor for f in fractions) else 'FAIL')
        low, high = extrema.get(metric, (None, None))
        spec_rows.append([status, declaration['name'], declaration['op'],
            _pvt_number(declaration['target']), _pvt_number(value),
            _pvt_point(worst.corner) if worst is not None else '—',
            _pvt_number(abs(value - declaration['target']) if value is not None else None),
            _pvt_number(low), _pvt_number(high)])
    for metric in metrics:
        if metric not in checked:
            spec_rows.append(['—', metric, '—', '—', '—', '—', '—',
                              *(_pvt_number(v) for v in extrema[metric])])
    run_rows = [[row['run.corner'], _pvt_number(row['run.temperature']),
                 _pvt_number(row['run.supply']),
                 *(repr(row['run.' + label]) for label in variables), row['run.sample'],
                 'NO REPORT' if row['run.status'] == 'failed' else row['run.status'],
                 *(_pvt_number(row['metric.' + metric], missing='') for metric in metrics)]
                for row in rows]
    errors = [_pvt_point((row['run.corner'], row['run.temperature'], row['run.supply'])
                         + ((tuple((label, row['run.' + label]) for label in variables),) if variables else ()))
              + f", sample {row['run.sample']}: "
              + (str(row['run.error'] or '').splitlines() or [''])[0]
              for row in rows if row['run.status'] == 'failed']
    return MonteCarloSection(data=dict(
        summary=summary, passed=bool(mc.passed), samples=len(rows),
        test_name=next((r.test_name for r in mc.reports.values()), 'unnamed'),
        spec_columns=['Status', 'Spec', 'Op', 'Target', 'Worst', 'Worst at', 'Margin', 'Min', 'Max'],
        spec_rows=spec_rows,
        yield_columns=['Corner', 'Temp (°C)', 'Supply (V)', *variable_columns, 'Spec', 'Samples', 'FAIL', 'No report',
                       'Yield', '95% interval', 'Status'], yield_rows=yield_rows,
        run_columns=['Corner', 'Temp (°C)', 'Supply (V)', *variable_columns, 'Sample', 'Status', *metrics],
        run_rows=run_rows, errors=errors))


def render_grid_section(section: MonteCarloSection, *, name='', mode='html') -> str:
    """Render a grid's spec, yield and sample tables without rerunning analysis."""
    from silk.circuit.report import _pvt_table, _pvt_text
    d = section.data
    heading = f"PVT Grid: {name or d['test_name']} — {'PASS' if d['passed'] else 'FAIL'}"
    if mode == 'html':
        out = ['<section class="pvt-grid"><h2>' + html.escape(heading) + '</h2>',
               '<p>' + html.escape(d['summary']) + '</p>',
               _pvt_table(d['spec_columns'], d['spec_rows'], mode, specs=True),
               '<h3>Yield at each point</h3>',
               _pvt_table(d['yield_columns'], d['yield_rows'], mode),
               f"<details><summary>All {d['samples']} samples</summary>",
               _pvt_table(d['run_columns'], d['run_rows'], mode), '</details>']
        out.extend('<p>' + html.escape(error) + '</p>' for error in d['errors'])
        out.append('</section>')
    else:
        out = [('## ' if mode == 'markdown' else '\n') + _pvt_text(heading, mode), '',
               _pvt_text(d['summary'], mode), '',
               _pvt_table(d['spec_columns'], d['spec_rows'], mode), '',
               ('### ' if mode == 'markdown' else '') + 'Yield at each point', '',
               _pvt_table(d['yield_columns'], d['yield_rows'], mode), '',
               ('### ' if mode == 'markdown' else '') + f"All {d['samples']} samples", '',
               _pvt_table(d['run_columns'], d['run_rows'], mode), '']
        out.extend(_pvt_text(error, mode) for error in d['errors'])
    return '\n'.join(out)
