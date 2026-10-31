"""LintReport — top-level container with violations + waivers + JSON I/O."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET
from silk._json_codec import decode_json, encode_json
from silk.lint.reports.metadata import ReportMetadata
from silk.lint.violations import Violation, InterfaceViolation, JunctionViolation
from silk.lint.severity import Severity
from silk.lint.stress_coverage import stress_coverage_text
from silk.lint.locations import (
    ViolationLocation, NetLocation, PortLocation,
    InstanceLocation, TerminalLocation, ModuleLocation,
)
from silk.lint.waivers import Waiver, WaiverMatcher


@dataclass
class LintDiff:
    """Difference between two lint reports, from :meth:`LintReport.diff`.

    Attributes
    ----------
    added : list of Violation
        Violations present in the reporting (left) report but not in the
        other one.
    removed : list of Violation
        Violations present in the other report but not in the left one.
    """
    added: list = field(default_factory=list)
    removed: list = field(default_factory=list)


def _violation_key(v) -> tuple:
    """Stable identity for diff: rule_id + location fingerprint + message."""
    return (v.rule_id, v.location.describe(), v.message)


LOCATION_TYPE_MAP = {
    "net": NetLocation,
    "port": PortLocation,
    "instance": InstanceLocation,
    "terminal": TerminalLocation,
    "module": ModuleLocation,
}


@dataclass
class LintReport:
    """Top-level lint result: findings, waivers, markers and JSON/HTML/JUnit I/O.

    Attributes
    ----------
    metadata : ReportMetadata
        Provenance and the context hash used to guard :meth:`diff`.
    violations : list of Violation
        Findings that no waiver or exclusion suppressed. Empty by default.
    waivers_applied : list
        The waived entries; ``summary`` and ``compliance_matrix`` count them
        by ``rule_id``. ``generate_report`` fills it with the suppressed
        :class:`Violation` objects; a report built by hand or loaded from
        older JSON may hold :class:`Waiver` objects instead. Empty by default.
    waivers_matched : list
        The :class:`Waiver` objects that matched at least one violation.
        ``to_dict`` writes them, together with any :class:`Waiver` in
        ``waivers_applied``, under ``waivers_applied``, and writes suppressed
        violations in ``waivers_applied`` under ``waived_violations``;
        ``from_dict`` restores both lists. Empty by default.
    no_connects : list of tuple
        ``(instance_name, net_name, NoConnect)`` entries for nets the
        designer marked intentionally unconnected. Empty by default.
    no_connect_ports : list of NoConnectPort
        Per-pin ``NoConnectPort`` markers, kept apart from ``no_connects`` so
        unpacking of the tuples keeps working. Both kinds are listed under one
        "Intentionally Unconnected" heading in JSON and HTML output.
    stale_markers : list of StaleMarker
        Markers whose target got wired after all, so they suppress nothing.
    stress_coverage : dict
        Serialised junction-stress coverage (the output of
        ``silk.lint.stress_coverage.StressCoverage.to_dict``), or an empty dict when there is none.
    """
    metadata: ReportMetadata
    violations: list = field(default_factory=list)
    # generate_report stores the suppressed Violation objects here (the
    # consumers read .rule_id/.message/.location); LintReport built by hand
    # or loaded from older JSON may hold Waiver objects instead.
    waivers_applied: list = field(default_factory=list)
    no_connects: list = field(default_factory=list)
    # NoConnectPort markers from Module.add_no_connect_port — the per-pin
    # suppressions. Kept apart from the (inst, net, NoConnect) tuples so
    # existing unpacking of no_connects keeps working; JSON and HTML list
    # both kinds under one 'Intentionally Unconnected' heading.
    no_connect_ports: list = field(default_factory=list)
    # StaleMarker rows from silk.lint_markers.find_stale_markers — markers
    # whose target got wired after all, so they suppress nothing.
    stale_markers: list = field(default_factory=list)
    stress_coverage: dict = field(default_factory=dict)
    # The Waiver objects that matched at least one violation (TASK-601).
    waivers_matched: list = field(default_factory=list)

    def summary(self) -> dict:
        """Count findings by severity.

        Returns
        -------
        dict
            ``{'errors', 'warnings', 'infos', 'waived', 'stale_markers'}``
            mapped to ints. Any severity other than ``ERROR`` or ``WARNING``
            counts as an info; ``waived`` is ``len(waivers_applied)``.
        """
        counts = {"errors": 0, "warnings": 0, "infos": 0}
        for v in self.violations:
            if v.severity == Severity.ERROR:
                counts["errors"] += 1
            elif v.severity == Severity.WARNING:
                counts["warnings"] += 1
            else:
                counts["infos"] += 1
        counts["waived"] = len(self.waivers_applied)
        counts["stale_markers"] = len(self.stale_markers)
        return counts

    def save_json(self, path) -> None:
        """Write the report as JSON.

        Non-finite floats are written as tagged objects (see
        ``silk._json_codec``), so the file is strict JSON that
        :meth:`load_json` reads back.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file; overwritten if it exists. The parent directory
            must exist.

        Raises
        ------
        TypeError
            If a violation's ``context`` holds a value JSON cannot encode,
            or its location has an unsupported type.
            Nothing is written in that case.
        OSError
            If the file cannot be written.
        """
        Path(path).write_text(json.dumps(encode_json(self.to_dict()), allow_nan=False, indent=2))

    def to_dict(self) -> dict:
        """Return the report as a dict of plain Python values.

        Returns
        -------
        dict
            Keys ``metadata``, ``violations``, ``waivers_applied`` (waiver
            dictionaries for the :class:`~silk.lint.Waiver` objects in
            ``waivers_applied`` and ``waivers_matched``), ``no_connects`` (net and port entries,
            distinguished by a
            ``kind`` key) and ``stale_markers``; plus ``stress_coverage``
            only when it is non-empty, and ``waived_violations`` only when
            ``waivers_applied`` holds suppressed violations rather than
            waivers. Violations carry a ``__class`` key
            and tagged locations so :meth:`from_dict` can rebuild them.
            Non-finite floats are left as floats here; :meth:`save_json`
            encodes them.
        """
        waiver_objs = [w for w in self.waivers_applied if isinstance(w, Waiver)]
        waived_violations = [v for v in self.waivers_applied if not isinstance(v, Waiver)]
        return {
            "metadata": self.metadata.to_dict(),
            **({"stress_coverage": self.stress_coverage} if self.stress_coverage else {}),
            "violations": [_violation_to_dict(v) for v in self.violations],
            "waivers_applied": [
                _waiver_to_dict(w) for w in waiver_objs + list(self.waivers_matched)],
            **({"waived_violations": [_violation_to_dict(v) for v in waived_violations]}
               if waived_violations else {}),
            "no_connects": [
                {"kind": "net", "instance_name": inst, "net_name": net,
                 "reason": nc.reason}
                for (inst, net, nc) in self.no_connects
            ] + [
                {"kind": "port", "instance_name": m.instance, "port": m.port,
                 "reason": m.reason}
                for m in self.no_connect_ports
            ],
            "stale_markers": [m.to_dict() for m in self.stale_markers],
        }

    @classmethod
    def load_json(cls, path) -> "LintReport":
        """Read a report written by :meth:`save_json`.

        Parameters
        ----------
        path : str or os.PathLike
            The JSON file.

        Returns
        -------
        LintReport
            The restored report.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file is not valid JSON, or a violation has an unknown
            location type or severity name.
        KeyError
            If a required key is missing.
        """
        data = decode_json(json.loads(Path(path).read_text()))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "LintReport":
        """Rebuild a report from the dict :meth:`to_dict` produces.

        Parameters
        ----------
        data : dict
            Must contain ``metadata``; the other sections are optional and
            default to empty. A ``no_connects`` entry without a ``kind`` is
            read as a net entry (the layout before port markers existed).

        Returns
        -------
        LintReport
            The restored report.

        Raises
        ------
        KeyError
            If ``metadata`` or a required key inside a section is missing.
        ValueError
            If a violation has an unknown location type or severity name.
        """
        from silk.lint_markers import NoConnect, NoConnectPort, StaleMarker
        metadata = ReportMetadata.from_dict(data["metadata"])
        violations = [_violation_from_dict(v) for v in data.get("violations", [])]
        waivers = [_waiver_from_dict(w) for w in data.get("waivers_applied", [])]
        waived_violations = [_violation_from_dict(v)
                             for v in data.get("waived_violations", [])]
        waivers_matched = []
        if waived_violations:
            # generate_report shape: waivers_applied holds suppressed violations.
            waivers_matched, waivers = waivers, waived_violations
        no_connects = []
        no_connect_ports = []
        for entry in data.get("no_connects", []):
            # Pre-TASK-51 files carry no 'kind'; they only ever held nets.
            if entry.get("kind", "net") == "port":
                no_connect_ports.append(NoConnectPort(
                    instance=entry["instance_name"], port=entry["port"],
                    reason=entry["reason"]))
            else:
                no_connects.append((entry["instance_name"], entry["net_name"],
                                    NoConnect(reason=entry["reason"])))
        stale = [StaleMarker.from_dict(d) for d in data.get("stale_markers", [])]
        return cls(
            metadata=metadata,
            stress_coverage=data.get("stress_coverage", {}),
            violations=violations,
            waivers_applied=waivers,
            waivers_matched=waivers_matched,
            no_connects=no_connects,
            no_connect_ports=no_connect_ports,
            stale_markers=stale,
        )

    def save_html(self, path) -> None:
        """Write a self-contained HTML summary of the report.

        The page lists the counts from :meth:`summary`, the violations table,
        the intentionally unconnected nets and ports, and the stale markers.
        Text is HTML-escaped.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file; overwritten if it exists.

        Raises
        ------
        OSError
            If the file cannot be written.
        """
        rows = []
        for v in self.violations:
            rows.append(
                f"<tr>"
                f"<td>{escape(v.rule_id)}</td>"
                f"<td>{escape(v.severity.to_string())}</td>"
                f"<td>{escape(v.location.describe())}</td>"
                f"<td>{escape(v.message)}</td>"
                f"</tr>"
            )
        summary = self.summary()
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Silk Lint Report — {escape(self.metadata.module_name)}</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2em; }}
h1 {{ color: #333; }}
.summary {{ padding: 1em; background: #f5f5f5; border-radius: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1em; }}
th, td {{ border: 1px solid #ddd; padding: 0.5em; text-align: left; }}
th {{ background: #eaeaea; }}
.error {{ color: #c00; }}
.warning {{ color: #c80; }}
.info {{ color: #068; }}
</style>
</head>
<body>
<h1>Silk Lint Report — {escape(self.metadata.module_name)}</h1>
<div class="summary">
<p>Generated: {escape(self.metadata.generated_at)}</p>
<p>PDK: {escape(self.metadata.pdk_name)} {escape(self.metadata.pdk_version)}</p>
<p>Silk version: {escape(self.metadata.silk_version)}</p>
<p><strong>{summary['errors']}</strong> errors,
   <strong>{summary['warnings']}</strong> warnings,
   <strong>{summary['infos']}</strong> infos,
   {summary['waived']} waived,
   <span class="{'warning' if summary['stale_markers'] else ''}">{summary['stale_markers']} stale marker(s)</span></p>
</div>
<p>{escape(stress_coverage_text(self.stress_coverage))}</p>
<h2>Violations</h2>
<table>
<thead><tr><th>Rule</th><th>Severity</th><th>Location</th><th>Message</th></tr></thead>
<tbody>
{''.join(rows) if rows else '<tr><td colspan="4">No violations</td></tr>'}
</tbody>
</table>
{self._render_no_connects_html()}
{self._render_stale_markers_html()}
</body>
</html>
"""
        Path(path).write_text(html)

    def _render_no_connects_html(self) -> str:
        if not self.no_connects and not self.no_connect_ports:
            return ""
        rows = "".join(
            f"<tr><td>{escape(inst)}</td><td>{escape(net)}</td><td></td>"
            f"<td>{escape(nc.reason)}</td></tr>"
            for (inst, net, nc) in self.no_connects
        ) + "".join(
            f"<tr><td>{escape(m.instance)}</td><td></td><td>{escape(m.port)}</td>"
            f"<td>{escape(m.reason)}</td></tr>"
            for m in self.no_connect_ports
        )
        return (
            "<h2>Intentionally Unconnected</h2>\n"
            "<table>\n"
            "<thead><tr><th>Instance</th><th>Net</th><th>Port</th><th>Reason</th></tr></thead>\n"
            f"<tbody>{rows}</tbody>\n"
            "</table>\n"
        )

    def _render_stale_markers_html(self) -> str:
        """WARNING-styled rows: each marker whose target is wired after all."""
        if not self.stale_markers:
            return ""
        rows = "".join(
            f"<tr class=\"warning\"><td>{escape(m.target())}</td>"
            f"<td>{escape(', '.join(m.wired_to))}</td>"
            f"<td>{escape(m.reason)}</td></tr>"
            for m in self.stale_markers
        )
        return (
            "<h2>Stale Markers</h2>\n"
            "<p class=\"warning\">These NoConnect / NoConnectPort markers sit on "
            "a target that is now wired, so they suppress nothing. Remove "
            "the marker or re-justify it.</p>\n"
            "<table>\n"
            "<thead><tr><th>Marker</th><th>Wired to</th><th>Reason</th></tr></thead>\n"
            f"<tbody>{rows}</tbody>\n"
            "</table>\n"
        )

    def save_junit_xml(self, path) -> None:
        """Write a JUnit-style XML file, one ``testcase`` per active rule.

        A rule is active if it has a violation (STR-001 is also included
        whenever ``stress_coverage`` is set). Each ``ERROR`` violation becomes
        a ``failure`` element; each ``WARNING`` becomes ``system-out`` text.
        ``INFO`` violations are not written. For STR-001 the coverage text is
        added as ``system-out``, and the case is marked ``skipped`` when the
        coverage counts include partially checked or unchecked devices and no
        ``ERROR`` violation was raised for the rule.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file; overwritten if it exists.

        Raises
        ------
        OSError
            If the file cannot be written.
        """
        rules = self._rules_with_activity()
        root = ET.Element("testsuites", attrib={
            "name": f"silk_lint_{self.metadata.module_name}",
        })
        suite = ET.SubElement(root, "testsuite", attrib={
            "name": "silk_lint",
            "tests": str(len(rules)),
            "failures": str(sum(1 for v in self.violations if v.severity == Severity.ERROR)),
        })
        for rule_id in rules:
            case = ET.SubElement(suite, "testcase", attrib={
                "classname": "silk_lint",
                "name": rule_id,
            })
            if rule_id == 'STR-001' and self.stress_coverage:
                text = stress_coverage_text(self.stress_coverage)
                ET.SubElement(case, 'system-out').text = text
                counts = self.stress_coverage['summary']
                if (counts['partial'] or counts['none']) and not any(
                        v.rule_id == rule_id and v.severity == Severity.ERROR for v in self.violations):
                    ET.SubElement(case, 'skipped', attrib={'message': text})
            for v in self.violations:
                if v.rule_id != rule_id:
                    continue
                if v.severity == Severity.ERROR:
                    failure = ET.SubElement(case, "failure", attrib={
                        "message": v.message,
                        "type": v.location.describe(),
                    })
                    failure.text = f"{v.location.describe()}: {v.message}"
                elif v.severity == Severity.WARNING:
                    sys_out = ET.SubElement(case, "system-out")
                    sys_out.text = f"WARNING {v.location.describe()}: {v.message}"
        tree = ET.ElementTree(root)
        tree.write(str(path), encoding="utf-8", xml_declaration=True)

    def _rules_with_activity(self) -> list:
        seen = ["STR-001"] if self.stress_coverage else []
        for v in self.violations:
            if v.rule_id not in seen:
                seen.append(v.rule_id)
        return seen

    def diff(self, other: "LintReport") -> "LintDiff":
        """Compare this report's violations with another report's.

        Violations are identified by ``(rule_id, location.describe(),
        message)``; severity and context do not take part.

        Parameters
        ----------
        other : LintReport
            The report to compare against, typically the older one.

        Returns
        -------
        LintDiff
            ``added`` holds violations only in ``self``; ``removed`` holds
            violations only in *other*.

        Raises
        ------
        ValueError
            If the two reports' ``metadata.context_hash`` values differ.
        """
        if self.metadata.context_hash != other.metadata.context_hash:
            raise ValueError(
                f"Cannot diff reports with mismatched context_hash: "
                f"{self.metadata.context_hash} vs {other.metadata.context_hash}"
            )
        self_keys = {_violation_key(v): v for v in self.violations}
        other_keys = {_violation_key(v): v for v in other.violations}
        added = [self_keys[k] for k in self_keys if k not in other_keys]
        removed = [other_keys[k] for k in other_keys if k not in self_keys]
        return LintDiff(added=added, removed=removed)

    def compliance_matrix(self) -> dict:
        """Count violations and waivers per rule.

        Returns
        -------
        dict
            Maps each ``rule_id`` that appears in ``violations`` or
            ``waivers_applied`` to ``{'violations': int, 'waivers': int}``,
            where ``waivers`` is the number of entries in
            ``waivers_applied`` for that rule (waived findings are not in
            ``violations``).
        """
        matrix: dict = {}
        for v in self.violations:
            bucket = matrix.setdefault(v.rule_id, {"violations": 0, "waivers": 0})
            bucket["violations"] += 1
        for w in self.waivers_applied:
            bucket = matrix.setdefault(w.rule_id, {"violations": 0, "waivers": 0})
            bucket["waivers"] += 1
        return matrix


# --- Serialization helpers ---

def _location_to_dict(loc: ViolationLocation) -> dict:
    for type_name, cls in LOCATION_TYPE_MAP.items():
        if type(loc) is cls:
            d = {"__type": type_name}
            d.update(loc.__dict__)
            return d
    raise TypeError(f"Unsupported location type {type(loc).__name__}")


def _location_from_dict(d: dict) -> ViolationLocation:
    d = dict(d)  # copy so we don't mutate caller
    type_name = d.pop("__type", "")
    cls = LOCATION_TYPE_MAP.get(type_name)
    if cls is None:
        raise ValueError(f"Unknown location type {type_name}")
    if "terminal_pair" in d and isinstance(d["terminal_pair"], list):
        d["terminal_pair"] = tuple(d["terminal_pair"])
    return cls(**d)


def _violation_to_dict(v: Violation) -> dict:
    d = {
        "__class": type(v).__name__,
        "rule_id": v.rule_id,
        "severity": v.severity.to_string(),
        "message": v.message,
        "location": _location_to_dict(v.location),
        "context": dict(v.context),
    }
    if isinstance(v, JunctionViolation):
        d["device_type"] = v.device_type
        d["measured_v"] = v.measured_v
        d["rated_max"] = v.rated_max
        d["worst_time"] = v.worst_time
    return d


def _violation_from_dict(d: dict) -> Violation:
    cls_name = d.get("__class", "Violation")
    location = _location_from_dict(d["location"])
    kwargs = {
        "rule_id": d["rule_id"],
        "severity": Severity.from_string(d["severity"]),
        "message": d["message"],
        "location": location,
        "context": dict(d.get("context", {})),
    }
    if cls_name == "InterfaceViolation":
        return InterfaceViolation(**kwargs)
    if cls_name == "JunctionViolation":
        return JunctionViolation(
            **kwargs,
            device_type=d.get("device_type", ""),
            measured_v=d.get("measured_v", 0.0),
            rated_max=d.get("rated_max", 0.0),
            worst_time=d.get("worst_time", 0.0),
        )
    return Violation(**kwargs)


def _waiver_to_dict(w: Waiver) -> dict:
    return {
        "rule_id": w.rule_id,
        "matcher": {
            "location_type": w.matcher.location_type,
            "fields": dict(w.matcher.fields),
        },
        "reason": w.reason,
        "author": w.author,
        "expires": w.expires.isoformat() if w.expires else None,
    }


def _waiver_from_dict(d: dict) -> Waiver:
    from datetime import date
    matcher = WaiverMatcher(
        location_type=d["matcher"]["location_type"],
        fields=d["matcher"].get("fields", {}),
    )
    expires = d.get("expires")
    if expires:
        expires = date.fromisoformat(expires)
    else:
        expires = None
    return Waiver(
        rule_id=d["rule_id"],
        matcher=matcher,
        reason=d["reason"],
        author=d.get("author", ""),
        expires=expires,
    )
