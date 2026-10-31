"""Report metadata: schema versioning, provenance, context hash."""
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone


#: Version string of the report JSON layout, ``"1.0"``. New
#: :class:`ReportMetadata` objects carry it as ``schema_version``.
SCHEMA_VERSION = "1.0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


@dataclass
class ReportMetadata:
    """Versioning and provenance metadata attached to every report.

    Reports with mismatched context_hash refuse to diff — two reports
    generated against different PDKs are not comparable.

    Attributes
    ----------
    silk_version : str
        Version of silk that produced the report.
    pdk_name : str
        Name of the PDK in use.
    pdk_version : str
        Version string of that PDK.
    module_name : str
        Name of the linted module.
    generated_at : str
        UTC creation time as an ISO-8601 string with a ``Z`` suffix and
        second resolution (for example ``'2026-01-31T12:00:00Z'``). Defaults
        to the time of construction.
    schema_version : str
        Report layout version; defaults to ``SCHEMA_VERSION``.
    context_hash : str
        16-hex-digit SHA-256 prefix over ``silk_version``, ``pdk_name``,
        ``pdk_version`` and ``module_name``. Computed in construction when
        left empty (the default); a non-empty value is kept as given.
        ``generated_at`` is not part of the hash.
    """
    silk_version: str
    pdk_name: str
    pdk_version: str
    module_name: str
    generated_at: str = field(default_factory=_now_iso)
    schema_version: str = SCHEMA_VERSION
    context_hash: str = ""

    def __post_init__(self):
        if not self.context_hash:
            self.context_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.silk_version.encode())
        h.update(b"|")
        h.update(self.pdk_name.encode())
        h.update(b"|")
        h.update(self.pdk_version.encode())
        h.update(b"|")
        h.update(self.module_name.encode())
        return h.hexdigest()[:16]

    def to_dict(self) -> dict:
        """Return a JSON-ready dict of every field.

        Returns
        -------
        dict
            Keys ``schema_version``, ``silk_version``, ``pdk_name``,
            ``pdk_version``, ``module_name``, ``generated_at`` and
            ``context_hash``.
        """
        return {
            "schema_version": self.schema_version,
            "silk_version": self.silk_version,
            "pdk_name": self.pdk_name,
            "pdk_version": self.pdk_version,
            "module_name": self.module_name,
            "generated_at": self.generated_at,
            "context_hash": self.context_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReportMetadata":
        """Rebuild metadata from the dict :meth:`to_dict` produces.

        Parameters
        ----------
        d : dict
            Must contain ``silk_version``, ``pdk_name``, ``pdk_version`` and
            ``module_name``. ``generated_at`` (default: now),
            ``schema_version`` (default ``SCHEMA_VERSION``) and
            ``context_hash`` (default: recomputed) are optional.

        Returns
        -------
        ReportMetadata
            The restored metadata.

        Raises
        ------
        KeyError
            If a required key is missing.
        """
        return cls(
            silk_version=d["silk_version"],
            pdk_name=d["pdk_name"],
            pdk_version=d["pdk_version"],
            module_name=d["module_name"],
            generated_at=d.get("generated_at", _now_iso()),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            context_hash=d.get("context_hash", ""),
        )
