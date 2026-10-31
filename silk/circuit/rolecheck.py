"""CLI entry point for the port-role ruleset checker.

Run::

    python -m silk.circuit.rolecheck [--file rules.yaml] [--names vdd clk ...]

The logic lives in :func:`silk.circuit.port_roles.main`. This thin module
exists only so the runnable entry is NOT the same module the ``silk.circuit``
package imports for its exports — running an already-imported module with
``python -m`` triggers a spurious ``runpy`` double-import warning, which this
avoids (mirrors ``silk/pdk/validate.py``'s warning-free invocation).

See ``docs/authoring-port-role-rules.md``.
"""
from silk.circuit.port_roles import main

if __name__ == '__main__':  # pragma: no cover
    import sys
    sys.exit(main())
