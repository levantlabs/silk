"""Type-dispatched constraint checkers — stubs only in this session.

A *checker* is a callable ``(constraint, layout) -> Any`` that decides
whether a placed/routed layout satisfies one constraint. Real checker
logic lands in a later session — for example, ``MaxIRDrop`` will be
served by ``silk.layout.ir`` once that module exists. Right now every
shipped type has a NotImplementedError stub, and the dispatch table
itself is fully working so layout consumers can
register their own checkers without modifying silk.

Design notes
------------
* The constraint object never carries a ``.check()`` method (per
  prompt). Dispatch lives here.
* The table is open: ``register(constraint_cls, fn)`` and
  ``unregister(constraint_cls, token)`` let layout consumers
  wire concrete logic later.
* ``check(constraint, layout)`` is the single entry point;
  ``check_all(mod, layout)`` is convenience for "run every constraint
  on a module".
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Type

from silk.constraints.envelope import ConstraintEnvelope


CheckerFn = Callable[[ConstraintEnvelope, Any], Any]


# Registered (active) checker per constraint type. Inserts replace any
# stub. ``register`` returns a token for ``unregister`` so a test can
# install a temporary checker without losing the prior stub.
_REGISTRY: Dict[Type[ConstraintEnvelope], List[CheckerFn]] = {}


def _stub(name: str) -> CheckerFn:
    def _inner(constraint, layout):
        raise NotImplementedError(
            f"checker for {name} is not implemented yet — "
            f"register a checker with silk.constraints.checkers.register()"
        )

    _inner.__name__ = f"_stub_{name}"
    return _inner


def register(constraint_cls: Type[ConstraintEnvelope], fn: CheckerFn) -> int:
    """Push a checker onto the stack for ``constraint_cls``.

    Returns a token; pass it back to :func:`unregister` to pop.
    Stacking lets tests install a temporary checker that automatically
    falls back to the prior one (or stub) on unregister.
    """
    stack = _REGISTRY.setdefault(constraint_cls, [])
    stack.append(fn)
    return len(stack) - 1


def unregister(constraint_cls: Type[ConstraintEnvelope], token: int) -> None:
    """Remove a previously-registered checker by its token (LIFO)."""
    stack = _REGISTRY.get(constraint_cls)
    if not stack or len(stack) <= token:
        return
    # Truncate to the size before this register() call.
    del stack[token:]


def check(constraint: ConstraintEnvelope, layout: Any) -> Any:
    """Dispatch one constraint to its registered checker.

    Raises ``LookupError`` if no checker is registered for the type.
    Raises ``NotImplementedError`` if the registered checker is a
    not-yet-implemented stub.
    """
    cls = type(constraint)
    stack = _REGISTRY.get(cls)
    if not stack:
        raise LookupError(
            f"no checker registered for constraint type {cls.__name__}"
        )
    return stack[-1](constraint, layout)


def check_all(mod, layout: Any) -> List[Tuple[ConstraintEnvelope, Any]]:
    """Run every constraint on a module against ``layout``.

    Returns a list of ``(constraint, result)`` tuples in attach order.
    Does not catch checker exceptions — if a checker raises, the
    iteration aborts. Callers that want best-effort behaviour should
    wrap individual ``check()`` calls themselves.
    """
    out: List[Tuple[ConstraintEnvelope, Any]] = []
    for c in mod.list_constraints():
        out.append((c, check(c, layout)))
    return out


# ---------------------------------------------------------------------------
# Register one stub per shipped concrete type.
# ---------------------------------------------------------------------------

def _install_stubs() -> None:
    from silk.constraints.types import (
        GroupBlocks, ProximityConstraint, SameTemplate, SymmetricBlocks,
    )

    for cls in (
        SymmetricBlocks, SameTemplate, GroupBlocks, ProximityConstraint,
    ):
        # Bottom of the stack — never popped by unregister, since the
        # token returned to test code is always >= 1.
        _REGISTRY.setdefault(cls, []).append(_stub(cls.__name__))


_install_stubs()


__all__ = ["check", "check_all", "register", "unregister", "CheckerFn"]
