from enum import IntEnum


class Severity(IntEnum):
    """Severity of a lint finding, ordered so that larger means more serious.

    An :class:`enum.IntEnum`, so severities compare and sort as integers.
    The members are ``INFO`` (10), ``WARNING`` (20) and ``ERROR`` (30). By
    convention ``ERROR`` fails a CI gate while ``WARNING`` and ``INFO`` do
    not; the reports count each level separately and JUnit output marks only
    ``ERROR`` findings as failures.
    """
    INFO = 10
    WARNING = 20
    ERROR = 30

    @classmethod
    def from_string(cls, s: str) -> "Severity":
        """Look a severity up by name, ignoring case.

        Parameters
        ----------
        s : str
            ``'info'``, ``'warning'`` or ``'error'`` in any letter case.

        Returns
        -------
        Severity
            The matching member.

        Raises
        ------
        ValueError
            If *s* is not the name of a member.
        """
        s = s.upper()
        if s not in cls.__members__:
            raise ValueError(f"Unknown severity: {s}")
        return cls[s]

    def to_string(self) -> str:
        """Return the lower-case member name, for example ``'warning'``.

        Returns
        -------
        str
            The name used in reports; :meth:`from_string` reverses it.
        """
        return self.name.lower()
