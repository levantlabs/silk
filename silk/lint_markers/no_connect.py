"""NoConnect marker — declares a net as intentionally unconnected.

The schematic "no-connection" flag, as a marker. Registering a NoConnect on a
net suppresses both STRUCT-003 FLOATING_NET (zero connections) and
STRUCT-011 SINGLE_CONNECTION_NET (exactly one) for that net, and surfaces
the marker in LintReport's 'Intentionally Unconnected' section.
"""


class NoConnect:
    """Lint marker element declaring a net as intentionally unconnected.

    Add to a module via ``Module.add_no_connect(net_name, reason=...)``.
    The reason string is required and must be non-empty.
    """
    is_lint_marker = True

    def __init__(self, reason: str):
        if not isinstance(reason, str):
            raise TypeError("NoConnect requires a 'reason' string argument")
        if not reason.strip():
            raise ValueError("NoConnect reason cannot be empty or blank")
        self.reason = reason

    def __repr__(self) -> str:
        return f"NoConnect(reason={self.reason!r})"
