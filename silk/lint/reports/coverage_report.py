"""CoverageReport — hierarchical per-rule coverage with v2 functional reserved."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape
from silk._json_codec import decode_json, encode_json
from silk.lint.reports.metadata import ReportMetadata
from silk.lint.registry import VisitEvent, VisitOutcome
from silk.lint.stress_coverage import stress_coverage_text


@dataclass
class CoverageDiff:
    """Per-rule coverage delta between two CoverageReports.

    For each rule, holds a tuple of (delta_checked, delta_excluded,
    delta_unchecked) measured as new minus old.

    Attributes
    ----------
    deltas : dict
        Maps each ``rule_id`` present in both reports to the tuple
        ``(delta_checked, delta_excluded, delta_unchecked)``.
    added_rules : list of str
        Sorted rule ids present only in the left report.
    removed_rules : list of str
        Sorted rule ids present only in the other report.
    """
    deltas: dict = field(default_factory=dict)
    added_rules: list = field(default_factory=list)
    removed_rules: list = field(default_factory=list)


@dataclass
class RuleCoverageNode:
    """Hierarchical per-module coverage tree node.

    local[rule_id] -> (checked, excluded, unchecked) tuple.

    "checked" = PASS + VIOLATION
    "excluded" = EXCLUDED (scope-excluded subtree)
    "unchecked" = SKIPPED_NO_SIM + INCOMPLETE

    STR-001 with device context uses full as checked, partial + none as
    unchecked. Its unit is devices; other rules retain scope-visit counts.

    Attributes
    ----------
    module_name : str
        Name of the module this node describes.
    local : dict
        Maps ``rule_id`` to the tuple ``(checked, excluded, unchecked)`` for
        this module alone. Empty by default.
    children : list of RuleCoverageNode
        Nodes for submodules. Empty by default.
    """
    module_name: str
    local: dict = field(default_factory=dict)
    children: list = field(default_factory=list)

    @property
    def aggregate(self) -> dict:
        """Per-rule counts summed over this node and all descendants.

        Returns
        -------
        dict
            Maps ``rule_id`` to ``(checked, excluded, unchecked)``. A new
            dict on every access.
        """
        agg: dict = dict(self.local)
        for child in self.children:
            child_agg = child.aggregate
            for rule_id, (c, e, u) in child_agg.items():
                existing = agg.get(rule_id, (0, 0, 0))
                agg[rule_id] = (
                    existing[0] + c,
                    existing[1] + e,
                    existing[2] + u,
                )
        return agg

    def percentage(self, rule_id: str) -> float:
        """Coverage percentage of one rule in this node alone.

        Computed as ``100 * checked / (checked + unchecked)`` from ``local``;
        excluded visits are not in the denominator. Children are not
        included (use ``aggregate`` for a subtree).

        Parameters
        ----------
        rule_id : str
            Rule identifier, for example ``'STRUCT-001'``.

        Returns
        -------
        float
            Percentage in ``[0, 100]``. ``0.0`` when the rule is absent from
            ``local``; ``100.0`` when it is present but nothing was checked
            or unchecked.
        """
        if rule_id not in self.local:
            return 0.0
        checked, excluded, unchecked = self.local[rule_id]
        denom = checked + unchecked
        if denom == 0:
            return 100.0
        return 100.0 * checked / denom

    def to_dict(self) -> dict:
        """Return a JSON-ready dict of the node and its children.

        Returns
        -------
        dict
            Keys ``module_name``, ``local`` (tuples written as lists) and
            ``children`` (recursively).
        """
        return {
            "module_name": self.module_name,
            "local": {k: list(v) for k, v in self.local.items()},
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RuleCoverageNode":
        """Rebuild a node from the dict :meth:`to_dict` produces.

        Parameters
        ----------
        d : dict
            Must contain ``module_name``; ``local`` and ``children`` are
            optional.

        Returns
        -------
        RuleCoverageNode
            The restored node, with ``local`` values as tuples.

        Raises
        ------
        KeyError
            If ``module_name`` is missing.
        """
        return cls(
            module_name=d["module_name"],
            local={k: tuple(v) for k, v in d.get("local", {}).items()},
            children=[cls.from_dict(c) for c in d.get("children", [])],
        )


@dataclass
class RuleCoverageReport:
    """The per-rule coverage tree of a :class:`CoverageReport`.

    Attributes
    ----------
    root : RuleCoverageNode
        Root of the module tree.
    """
    root: RuleCoverageNode

    def to_dict(self) -> dict:
        """Return a JSON-ready dict of the tree.

        Returns
        -------
        dict
            ``{'root': <root node dict>}``.
        """
        return {"root": self.root.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "RuleCoverageReport":
        """Rebuild a report from the dict :meth:`to_dict` produces.

        Parameters
        ----------
        d : dict
            Must contain ``root``.

        Returns
        -------
        RuleCoverageReport
            The restored report.

        Raises
        ------
        KeyError
            If ``root`` (or a node's ``module_name``) is missing.
        """
        return cls(root=RuleCoverageNode.from_dict(d["root"]))


@dataclass
class CoverageReport:
    """Hierarchical per-rule coverage of a lint run, with JSON and HTML output.

    Attributes
    ----------
    metadata : ReportMetadata
        Provenance and the context hash used to guard :meth:`diff`.
    rule_coverage : RuleCoverageReport
        Per-rule ``(checked, excluded, unchecked)`` counts per module.
    functional_coverage : dict, optional
        Reserved for a later release; always ``None`` in reports built by
        :meth:`from_visit_log`.
    stress_coverage : dict
        Junction-stress device coverage summed over the visit log, in the
        ``StressCoverage.to_dict`` shape; empty when no visit carried it.
    """
    metadata: ReportMetadata
    rule_coverage: RuleCoverageReport
    functional_coverage: Optional[dict] = None  # reserved for v2 Slice B

    stress_coverage: dict = field(default_factory=dict)

    @classmethod
    def from_visit_log(
        cls,
        metadata: ReportMetadata,
        visit_log: list,
        module_name: str = "",
    ) -> "CoverageReport":
        """Build a coverage report from a registry visit log.

        Each event adds to its rule's ``(checked, excluded, unchecked)``
        counts: ``PASS`` and ``VIOLATION`` count as checked, ``EXCLUDED`` as
        excluded, ``SKIPPED_NO_SIM`` and ``INCOMPLETE`` as unchecked. An event
        whose ``context`` carries ``stress_coverage`` is counted in devices
        instead: fully checked devices as checked, partially checked and
        unchecked devices as unchecked.

        Parameters
        ----------
        metadata : ReportMetadata
            Metadata to attach.
        visit_log : list of VisitEvent
            Events from :attr:`RegistryRunResult.visit_log`. It should come
            from an unfiltered run; see :meth:`RuleRegistry.run`.
        module_name : str, optional
            Name for the root node. Defaults to ``metadata.module_name``.

        Returns
        -------
        CoverageReport
            A report with a single root node and no children.
        """
        root = RuleCoverageNode(module_name=module_name or metadata.module_name)
        counts: dict = {}
        stress_coverage = {}
        for event in visit_log:
            c = counts.setdefault(event.rule_id, [0, 0, 0])
            stress = event.context.get('stress_coverage')
            if stress:
                if not stress_coverage:
                    stress_coverage = {'summary': dict.fromkeys(stress['summary'], 0), 'devices': []}
                for key, value in stress['summary'].items():
                    stress_coverage['summary'][key] += value
                stress_coverage['devices'].extend(stress['devices'])
                c[0] += stress['summary']['full']
                c[2] += stress['summary']['partial'] + stress['summary']['none']
            elif event.outcome in (VisitOutcome.PASS, VisitOutcome.VIOLATION):
                c[0] += 1
            elif event.outcome == VisitOutcome.EXCLUDED:
                c[1] += 1
            elif event.outcome in (VisitOutcome.SKIPPED_NO_SIM, VisitOutcome.INCOMPLETE):
                c[2] += 1
        root.local = {k: tuple(v) for k, v in counts.items()}
        return cls(
            metadata=metadata,
            rule_coverage=RuleCoverageReport(root=root),
            functional_coverage=None,
            stress_coverage=stress_coverage,
        )

    def save_json(self, path) -> None:
        """Write the report as strict JSON.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file; overwritten if it exists. The parent directory
            must exist.

        Raises
        ------
        OSError
            If the file cannot be written.
        """
        Path(path).write_text(json.dumps(encode_json(self.to_dict()), allow_nan=False, indent=2))

    def to_dict(self) -> dict:
        """Return the report as a dict of plain Python values.

        Returns
        -------
        dict
            Keys ``metadata``, ``rule_coverage`` and ``functional_coverage``,
            plus ``stress_coverage`` only when it is non-empty.
        """
        return {
            "metadata": self.metadata.to_dict(),
            "rule_coverage": self.rule_coverage.to_dict(),
            "functional_coverage": self.functional_coverage,
            **({"stress_coverage": self.stress_coverage} if self.stress_coverage else {}),
        }

    @classmethod
    def load_json(cls, path) -> "CoverageReport":
        """Read a report written by :meth:`save_json`.

        Parameters
        ----------
        path : str or os.PathLike
            The JSON file.

        Returns
        -------
        CoverageReport
            The restored report.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file is not valid JSON.
        KeyError
            If ``metadata`` or ``rule_coverage`` is missing.
        """
        data = decode_json(json.loads(Path(path).read_text()))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "CoverageReport":
        """Rebuild a report from the dict :meth:`to_dict` produces.

        Parameters
        ----------
        data : dict
            Must contain ``metadata`` and ``rule_coverage``;
            ``functional_coverage`` and ``stress_coverage`` are optional.

        Returns
        -------
        CoverageReport
            The restored report.

        Raises
        ------
        KeyError
            If a required key is missing.
        """
        return cls(
            metadata=ReportMetadata.from_dict(data["metadata"]),
            rule_coverage=RuleCoverageReport.from_dict(data["rule_coverage"]),
            functional_coverage=data.get("functional_coverage"),
            stress_coverage=data.get("stress_coverage", {}),
        )

    def save_html(self, path) -> None:
        """Render a styled coverage scorecard as a self-contained HTML file.

        The table has one row per rule, from the aggregate over the whole
        tree, showing the checked, excluded and unchecked counts and the
        coverage ``checked / (checked + unchecked)``. A rule with neither
        checked nor unchecked visits shows 100.0%.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file; overwritten if it exists.

        Raises
        ------
        OSError
            If the file cannot be written.
        """
        agg = self.rule_coverage.root.aggregate
        rows = []
        for rule_id in sorted(agg.keys()):
            checked, excluded, unchecked = agg[rule_id]
            denom = checked + unchecked
            pct = 100.0 if denom == 0 else (100.0 * checked / denom)
            css = "ok" if pct >= 90 else ("warn" if pct >= 60 else "bad")
            rows.append(
                f'<tr class="{css}">'
                f"<td>{escape(rule_id)}</td>"
                f"<td>{checked}</td>"
                f"<td>{excluded}</td>"
                f"<td>{unchecked}</td>"
                f'<td>{pct:.1f}%</td>'
                f"</tr>"
            )
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Silk Coverage Scorecard — {escape(self.metadata.module_name)}</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2em; }}
h1 {{ color: #333; }}
.summary {{ padding: 1em; background: #f5f5f5; border-radius: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1em; }}
th, td {{ border: 1px solid #ddd; padding: 0.5em; text-align: left; }}
th {{ background: #eaeaea; }}
.ok td:last-child {{ color: #060; font-weight: 600; }}
.warn td:last-child {{ color: #a60; font-weight: 600; }}
.bad td:last-child {{ color: #c00; font-weight: 600; }}
</style>
</head>
<body>
<h1>Silk Coverage Scorecard — {escape(self.metadata.module_name)}</h1>
<div class="summary">
<p>Generated: {escape(self.metadata.generated_at)}</p>
<p>PDK: {escape(self.metadata.pdk_name)} {escape(self.metadata.pdk_version)}</p>
<p>Silk version: {escape(self.metadata.silk_version)}</p>
</div>
<p>{escape(stress_coverage_text(self.stress_coverage))}</p>
<h2>Per-rule coverage (aggregate)</h2>
<table>
<thead><tr><th>Rule</th><th>Checked</th><th>Excluded</th><th>Unchecked</th><th>Coverage</th></tr></thead>
<tbody>
{''.join(rows) if rows else '<tr><td colspan="5">No rule activity recorded</td></tr>'}
</tbody>
</table>
<p><em>Coverage = checked / (checked + unchecked). STR-001 counts devices: checked means fully checked, unchecked includes partial. Other rules count scopes. Excluded subtrees are not in the denominator.</em></p>
</body>
</html>
"""
        Path(path).write_text(html)

    def diff(self, other: "CoverageReport") -> "CoverageDiff":
        """Per-rule coverage delta vs. another report (self minus other).

        Refuses to diff reports with mismatched ``context_hash``, matching
        :meth:`LintReport.diff` semantics.

        Parameters
        ----------
        other : CoverageReport
            The report to subtract, typically the older one.

        Returns
        -------
        CoverageDiff
            Deltas for the rules in both reports (computed on the aggregate
            over each tree), and the sorted rule ids found in only one.

        Raises
        ------
        ValueError
            If the two reports' ``metadata.context_hash`` values differ.
        """
        if self.metadata.context_hash != other.metadata.context_hash:
            raise ValueError(
                f"Cannot diff coverage reports with mismatched context_hash: "
                f"{self.metadata.context_hash} vs {other.metadata.context_hash}"
            )
        self_agg = self.rule_coverage.root.aggregate
        other_agg = other.rule_coverage.root.aggregate
        deltas = {}
        for rule_id in set(self_agg) & set(other_agg):
            sc, se, su = self_agg[rule_id]
            oc, oe, ou = other_agg[rule_id]
            deltas[rule_id] = (sc - oc, se - oe, su - ou)
        added = sorted(set(self_agg) - set(other_agg))
        removed = sorted(set(other_agg) - set(self_agg))
        return CoverageDiff(deltas=deltas, added_rules=added, removed_rules=removed)
