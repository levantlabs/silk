"""CampaignReport: aggregate reporting for a campaign result.

Reports include run-status counts and per-spec measurement statistics.
"""

from __future__ import annotations

import json
import math
from io import StringIO

from silk._json_codec import encode_json


class CampaignReport:
    """Render a :class:`silk.faults.CampaignResult` as text, markdown, HTML or JSON.

    Usually obtained from :meth:`silk.faults.CampaignResult.generate_report`.

    Parameters
    ----------
    result : silk.faults.CampaignResult
        The campaign result to report on.

    Attributes
    ----------
    result : silk.faults.CampaignResult
        The result being reported on.

    Examples
    --------
    >>> from silk.faults import CampaignResult, FaultSet
    >>> result = CampaignResult('campaign_0', 'top', 0, FaultSet(), 0, [], 'out')
    >>> print(result.generate_report().summary())
    Campaign: campaign_0
      Module: top
      Runs:   0  (passed: 0, failed: 0, unverified: 0, simulated: 0, errored: 0)
    <BLANKLINE>
    """

    def __init__(self, result):
        self.result = result

    def summary(self) -> str:
        """Return a plain-text summary for the terminal.

        Returns
        -------
        str
            Campaign id, module name, run counts by status, and one line per
            spec that has measurements: pass percentage and the mean,
            standard deviation, minimum and maximum of the measured values
            (in the spec's units).
        """
        r = self.result
        status_counts = self._status_counts()
        buf = StringIO()
        print(f'Campaign: {r.campaign_id}', file=buf)
        print(f'  Module: {r.module_name}', file=buf)
        print(f'  Runs:   {r.n_runs}  ({self._counts_text(status_counts)})',
              file=buf)
        for label, identity in self._spec_labels():
            dist = r._measurement_distribution(identity)
            usable = len(r._usable_rows(identity))
            if len(dist) == 0 and usable == 0:
                continue
            frac = r._pass_fraction(identity)
            pct = 'pass    n/a' if math.isnan(frac) else f'pass {frac*100:5.1f}%'
            stats = ('' if len(dist) == 0 else
                     f'  mu={dist.mean():.3f}  sigma={dist.std():.3f}  '
                     f'min={dist.min():.3f}  max={dist.max():.3f}')
            print(f'  {label:<20} {pct} ({usable}/{len(r.runs)} usable){stats}',
                  file=buf)
        scope = self._scope_note()
        if scope:
            print(f'  {scope}', file=buf)
        for owner, name, key, reason in self._excluded_rows():
            print(f'  not checked: {owner} / {name} [{key}]: {reason}', file=buf)
        return buf.getvalue()

    _STATUSES = ('passed', 'failed', 'unverified', 'simulated', 'errored')

    def _status_counts(self) -> dict:
        counts = {k: 0 for k in self._STATUSES}
        for run in self.result.runs:
            counts[run.status] = counts.get(run.status, 0) + 1
        return counts

    @staticmethod
    def _counts_text(counts) -> str:
        return ', '.join(f'{k}: {v}' for k, v in counts.items())

    def _excluded_rows(self) -> list:
        """Distinct not-checked specs ``(owner_path, spec_name, spec_key,
        reason)`` over all runs, in first-seen order."""
        seen: dict = {}
        for run in self.result.runs:
            for e in run.excluded:
                seen.setdefault((e['owner_path'], e['spec_name'],
                                 e['spec_key'], e['reason']), None)
        return list(seen)

    def _scope_note(self) -> str:
        """One line naming what a pass covers, or why nothing was verified."""
        runs = self.result.runs
        tests = sorted({r.test_name for r in runs if r.test_name})
        excluded = sum(len(r.excluded) for r in runs)
        if any(r.evaluation == 'execution-only' for r in runs):
            return 'Execution only: specs were not scored (status simulated).'
        if tests:
            note = f'A pass covers test {tests[0]!r} at its DUT placement only'
            if excluded:
                note += f'; {excluded} spec row(s) were not checked'
            return note + '.'
        if any(r.evaluation == 'fresh-v1' for r in runs):
            return 'No campaign testbench: no spec could be verified.'
        return ''

    def _spec_labels(self) -> list:
        """``(display label, identity)`` per spec; the label is display only."""
        ids = self.result._spec_identities()
        by_name: dict = {}
        for owner, key, name in ids:
            by_name.setdefault(name, []).append((owner, key))
        out = []
        for owner, key, name in ids:
            same = by_name[name]
            if len(same) == 1:
                label = name
            elif len({o for o, _ in same}) == len(same):
                label = f'{owner} / {name}'
            else:
                label = f'{owner} / {name} [{key}]'
            out.append((str(label), (owner, key)))
        return sorted(out)

    def save(self, path: str) -> None:
        """Write the report as markdown.

        Parameters
        ----------
        path : str
            Destination file; overwritten if it exists. The content has the
            campaign header, run counts by status, a per-spec statistics
            table, and for each spec up to ten fault correlations, largest
            first.

        Raises
        ------
        OSError
            The file cannot be written.
        """
        with open(path, 'w') as f:
            f.write(self._render_md())

    def _render_md(self) -> str:
        r = self.result
        buf = StringIO()
        print(f'# Campaign {r.campaign_id}', file=buf)
        print('', file=buf)
        print(f'- **Module:** `{r.module_name}`', file=buf)
        print(f'- **Runs:** {r.n_runs}', file=buf)
        print(f'- **Seed:** {r.seed}', file=buf)
        print('', file=buf)

        status_counts = self._status_counts()
        print('## Campaign health', file=buf)
        print('', file=buf)
        for status in self._STATUSES:
            print(f'- {status.capitalize()}: {status_counts[status]}', file=buf)
        scope = self._scope_note()
        if scope:
            print('', file=buf)
            print(scope, file=buf)
        excluded = self._excluded_rows()
        if excluded:
            print('', file=buf)
            print('Not checked (excluded from every statistic):', file=buf)
            print('', file=buf)
            for owner, name, key, reason in excluded:
                print(f'- `{owner}` / `{name}` [{key}]: {reason}', file=buf)
        print('', file=buf)

        print('## Per-spec statistics', file=buf)
        print('', file=buf)
        print('| Spec | Pass % | Usable runs | mu | sigma | Min | Max |', file=buf)
        print('|------|--------|-------------|----|-------|-----|-----|', file=buf)
        for sn, identity in self._spec_labels():
            dist = r._measurement_distribution(identity)
            usable = f'{len(r._usable_rows(identity))}/{len(r.runs)}'
            frac = r._pass_fraction(identity)
            pct = '--' if math.isnan(frac) else f'{frac*100:.1f}%'
            if len(dist) == 0:
                print(f'| `{sn}` | {pct} | {usable} | -- | -- | -- | -- |', file=buf)
                continue
            print(f'| `{sn}` | {pct} | {usable} | '
                  f'{dist.mean():.3f} | {dist.std():.3f} | '
                  f'{dist.min():.3f} | {dist.max():.3f} |', file=buf)
        print('', file=buf)

        for sn, identity in self._spec_labels():
            corr = r._correlate(identity)
            if not corr:
                continue
            top = sorted(corr.items(), key=lambda kv: -kv[1])[:10]
            if not top:
                continue
            print(f'## Fault correlations for `{sn}`', file=buf)
            print('', file=buf)
            print('| Fault | P(fail | active) - P(fail | inactive) |', file=buf)
            print('|-------|---------------------------------------|', file=buf)
            for key, val in top:
                print(f'| `{key}` | {val:+.3f} |', file=buf)
            print('', file=buf)
        return buf.getvalue()

    def save_html(self, path: str) -> None:
        """Write the report as a standalone HTML page.

        Parameters
        ----------
        path : str
            Destination file; overwritten if it exists. The page is the
            :meth:`save` markdown converted by a minimal built-in renderer.

        Raises
        ------
        OSError
            The file cannot be written.
        """
        md = self._render_md()
        html = (
            '<html><head><meta charset="utf-8">'
            '<style>body{font-family:sans-serif;max-width:1000px;'
            'margin:2em auto;}'
            'table{border-collapse:collapse;}'
            'th,td{border:1px solid #ccc;padding:4px 8px;}'
            'code{background:#f4f4f4;padding:0 4px;}</style>'
            '</head><body>\n' + _markdown_to_html(md) + '\n</body></html>')
        with open(path, 'w') as f:
            f.write(html)

    def save_json(self, path: str) -> None:
        """Write :meth:`to_json` to a JSON file.

        Parameters
        ----------
        path : str
            Destination file; overwritten if it exists. NaN and infinite
            numbers are written as tagged objects (``status`` ``'unavailable'``
            or ``'infinite'``) and values JSON cannot represent are written
            as strings.

        Raises
        ------
        OSError
            The file cannot be written.
        """
        with open(path, 'w') as f:
            json.dump(encode_json(self.to_json()), f, allow_nan=False, indent=2, default=str)

    def to_json(self) -> dict:
        """Return the campaign as a JSON-serializable dict.

        Returns
        -------
        dict
            ``campaign_id``, ``module_name``, ``n_runs``, ``seed`` and a
            ``runs`` list with each run's id, seed, status, evaluation marker,
            checked test, backend, requested corner, excluded (not-checked)
            specs, manifest (as
            :meth:`silk.faults.FaultManifest.to_json`), rawfile path,
            wall time in seconds and error text.
        """
        r = self.result
        return {
            'campaign_id': r.campaign_id,
            'module_name': r.module_name,
            'n_runs': r.n_runs,
            'seed': r.seed,
            'runs': [
                {'run_id': run.run_id, 'seed': run.seed,
                 'status': run.status,
                 'evaluation': run.evaluation,
                 'test_name': run.test_name,
                 'backend': run.backend,
                 'requested_corner': run.requested_corner,
                 'excluded': run.excluded,
                 'manifest': run.manifest.to_json(),
                 'rawfile_path': run.rawfile_path,
                 'wall_time_s': run.wall_time_s,
                 'error': run.error}
                for run in r.runs
            ],
        }

def _markdown_to_html(md: str) -> str:
    """Minimal ad-hoc markdown -> HTML for CampaignReport. Handles H1/H2,
    unordered lists, code spans, and pipe-tables. Not a general renderer
    — just enough for our own output."""
    out_lines = []
    in_table = False
    in_list = False
    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith('# '):
            if in_table:
                out_lines.append('</table>')
                in_table = False
            if in_list:
                out_lines.append('</ul>')
                in_list = False
            out_lines.append(f'<h1>{_inline(line[2:])}</h1>')
            continue
        if line.startswith('## '):
            if in_table:
                out_lines.append('</table>')
                in_table = False
            if in_list:
                out_lines.append('</ul>')
                in_list = False
            out_lines.append(f'<h2>{_inline(line[3:])}</h2>')
            continue
        if line.startswith('|'):
            if not in_table:
                out_lines.append('<table>')
                in_table = True
            if set(line.replace('|', '').strip()) <= {'-', ' ', ':'}:
                continue
            cells = [c.strip() for c in line.strip('|').split('|')]
            row_html = ''.join(f'<td>{_inline(c)}</td>' for c in cells)
            out_lines.append(f'<tr>{row_html}</tr>')
            continue
        if line.startswith('- '):
            if not in_list:
                out_lines.append('<ul>')
                in_list = True
            out_lines.append(f'<li>{_inline(line[2:])}</li>')
            continue
        if in_list:
            out_lines.append('</ul>')
            in_list = False
        if in_table:
            out_lines.append('</table>')
            in_table = False
        if line.strip() == '':
            out_lines.append('')
            continue
        out_lines.append(f'<p>{_inline(line)}</p>')
    if in_table:
        out_lines.append('</table>')
    if in_list:
        out_lines.append('</ul>')
    return '\n'.join(out_lines)


def _inline(text: str) -> str:
    out = []
    in_code = False
    buf: list = []
    for ch in text:
        if ch == '`':
            out.append(''.join(buf))
            buf = []
            out.append('<code>' if not in_code else '</code>')
            in_code = not in_code
        else:
            buf.append(ch)
    out.append(''.join(buf))
    return ''.join(out)
