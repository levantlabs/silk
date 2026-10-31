"""Scope exclusions for silk lint.

Distinct from waivers: waivers suppress specific violations on specific
elements; exclusions drop findings on a subtree after the rules have run
(the rules themselves still execute on it). Used for
vendor IP black boxes the designer doesn't own.
"""
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class ScopeExclusion:
    """A declared region of the design whose findings are filtered out.

    Findings are removed after rule dispatch, when the report is generated;
    rules are still dispatched over the region.

    Attributes
    ----------
    target_type : str
        ``'instance'`` (matches by instance_path with subtree) or
        ``'module'`` (matches by module name; not used in v1).
    target_path : str
        For example ``'ldo.sram_macro'``.
    reason : str
        Required, non-empty justification.
    owner : str, optional
        Vendor or module identifier. Empty by default.

    Raises
    ------
    ValueError
        If ``reason`` is empty or whitespace-only.
    """
    target_type: str
    target_path: str
    reason: str
    owner: str = ""

    def __post_init__(self):
        if not self.reason or not self.reason.strip():
            raise ValueError("ScopeExclusion requires a non-empty reason")

    def matches_instance_path(self, path: str) -> bool:
        """Test whether *path* lies in the excluded subtree.

        Parameters
        ----------
        path : str
            Dot-separated hierarchical instance path.

        Returns
        -------
        bool
            ``True`` for an ``'instance'`` exclusion when *path* equals
            ``target_path`` or starts with ``target_path + '.'``. Always
            ``False`` for other ``target_type`` values.
        """
        if self.target_type != "instance":
            return False
        if path == self.target_path:
            return True
        return path.startswith(self.target_path + ".")


@dataclass
class ExclusionFile:
    """A set of scope exclusions loaded from a YAML file.

    Attributes
    ----------
    exclusions : list of ScopeExclusion
        The exclusions, in file order. Empty by default.
    """
    exclusions: list = field(default_factory=list)

    @classmethod
    def load(cls, path) -> "ExclusionFile":
        """Read scope exclusions from a YAML file.

        The file is a mapping with an ``exclusions`` list. Each entry has a
        ``target`` mapping (``type``, default ``'instance'``, and ``path``),
        a required ``reason`` and an optional ``owner``. An empty file, or
        one without ``exclusions``, loads as no exclusions.

        Parameters
        ----------
        path : str or os.PathLike
            The YAML file.

        Returns
        -------
        ExclusionFile
            The parsed exclusions.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If an entry's ``reason`` is missing or empty.
        """
        path = Path(path)
        data = yaml.safe_load(path.read_text()) or {}
        entries = data.get("exclusions", []) or []
        exclusions = []
        for entry in entries:
            target = entry.get("target", {})
            exclusions.append(ScopeExclusion(
                target_type=target.get("type", "instance"),
                target_path=target.get("path", ""),
                reason=entry.get("reason", ""),
                owner=entry.get("owner", ""),
            ))
        return cls(exclusions=exclusions)
