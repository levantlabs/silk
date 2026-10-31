"""NoConnectPort marker — declares a submodule port as intentionally unwired.

Counterpart to :class:`silk.lint_markers.NoConnect`, but targets an
individual submodule-instance port rather than a net. Registering a
NoConnectPort on ``(instance, port)`` suppresses STRUCT-001
DANGLING_PORT for that specific pin and surfaces the marker in
LintReport's 'Intentionally Unconnected' section.

Use when the upstream schematic legitimately leaves a port floating —
e.g., an optional ``test_enable`` pin that's tied off at a higher
assembly level, an unused auxiliary output, or a debug tap that's only
wired in a separate testbench. Prefer this over adding a glob to
``DanglingPortRule.ignore_patterns``: the per-instance marker forces
documentation (``reason`` is required) and survives renaming.
"""


class NoConnectPort:
    """Lint marker declaring a specific submodule-instance port as
    intentionally unwired.

    Add via
    ``Module.add_no_connect_port(instance, port, reason='...')``.
    The reason string is required and must be non-empty.

    Attributes
    ----------
    instance : str
        Submodule instance name inside the parent module
        (e.g. ``'comp1'``, ``'th1'``).
    port : str
        Port name on that instance (e.g. ``'test_enable'``).
    reason : str
        Why this port is deliberately floating. Required, non-blank.
    """
    is_lint_marker = True

    def __init__(self, instance: str, port: str, reason: str):
        if not isinstance(instance, str) or not instance:
            raise TypeError(
                "NoConnectPort requires a non-empty 'instance' string")
        if not isinstance(port, str) or not port:
            raise TypeError(
                "NoConnectPort requires a non-empty 'port' string")
        if not isinstance(reason, str):
            raise TypeError("NoConnectPort requires a 'reason' string argument")
        if not reason.strip():
            raise ValueError("NoConnectPort reason cannot be empty or blank")
        self.instance = instance
        self.port = port
        self.reason = reason

    def __repr__(self) -> str:
        return (f"NoConnectPort(instance={self.instance!r}, "
                f"port={self.port!r}, reason={self.reason!r})")
