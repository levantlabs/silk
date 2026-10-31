"""Stale-marker detection — a suppression whose target got wired after all.

A ``NoConnect``/``NoConnectPort`` marker is placed while a net or pin is
deliberately left open. Nothing stops the design from wiring that target
later (a tap that becomes a real node, an enable that gets connected at
integration), and when it does the marker keeps sitting on it: the
structural rules simply find nothing to suppress, so the report stays
green and the stale "intentionally unconnected" claim is never
questioned. Waivers carry ``expires`` for exactly this reason; markers
have no clock, so their staleness is decided from the connectivity they
were meant to excuse.

``find_stale_markers`` is the single pass that decides this — one row
per marker, independent of how many rules honour that marker (STRUCT-003
and STRUCT-011 both honour ``NoConnect``; reporting from inside each rule
would list the same marker twice).

Staleness, per marker kind:

* ``NoConnect`` on a net — stale when the net has **two or more** real
  (non-marker) bindings. Zero real bindings is the STRUCT-003 case and
  exactly one is the STRUCT-011 case; both are the marker's documented
  job (a probe point, a test tap, a monitor-only output). At two the net
  is an ordinary node neither rule would report, so the marker suppresses
  nothing.
* ``NoConnectPort`` on ``instance.port`` — stale as soon as the pin
  carries a net: STRUCT-001 only ever reports unbound pins.

A marker whose target no longer exists (net deleted, instance removed)
is *orphaned*, not stale, and is not reported here: registration
validates the target, and nothing in the public API removes one.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StaleMarker:
    """One marker whose target is wired — the suppression is a no-op.

    Attributes
    ----------
    kind : str
        ``'net'`` for a ``NoConnect``, ``'port'`` for a ``NoConnectPort``.
    reason : str
        The reason recorded on the marker, verbatim.
    net_name : str or None
        ``'net'`` kind: the marked net. ``None`` for ``'port'`` markers.
    instance : str or None
        ``'port'`` kind: the marked instance. ``None`` for ``'net'`` markers.
    port : str or None
        ``'port'`` kind: the marked pin. ``None`` for ``'net'`` markers.
    wired_to : tuple of str
        What made the marker stale — ``'instance.port'`` for every real
        binding on a marked net; the net name the marked pin now carries.
    """
    kind: str
    reason: str
    net_name: 'str | None' = None
    instance: 'str | None' = None
    port: 'str | None' = None
    wired_to: tuple = field(default_factory=tuple)

    def target(self) -> str:
        """``'net probe_a'`` or ``'port u1.en'`` — the row label."""
        if self.kind == 'net':
            return f"net {self.net_name}"
        return f"port {self.instance}.{self.port}"

    def describe(self) -> str:
        wired = ', '.join(self.wired_to)
        return f"stale NoConnect{'Port' if self.kind == 'port' else ''} on " \
               f"{self.target()} ({self.reason!r}): wired to {wired}"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "reason": self.reason,
            "net_name": self.net_name,
            "instance": self.instance,
            "port": self.port,
            "wired_to": list(self.wired_to),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StaleMarker":
        return cls(
            kind=d["kind"],
            reason=d["reason"],
            net_name=d.get("net_name"),
            instance=d.get("instance"),
            port=d.get("port"),
            wired_to=tuple(d.get("wired_to", ())),
        )


# STRUCT-003 fires at 0 real bindings, STRUCT-011 at 1; from here on the
# net is a node no honouring rule would report.
_NET_MARKER_STALE_AT = 2


def find_stale_markers(module) -> list:
    """Return one :class:`StaleMarker` per marker on ``module`` whose
    target is wired, in registration order (nets first, then ports).

    Reads ``Module._no_connects`` and ``Module._no_connect_ports`` — the
    same registries the structural rules consult — so a marker that
    suppresses nothing here is exactly one the rules would also skip.
    """
    stale = []
    net_dict = getattr(module, 'net_dict', None) or {}
    for _inst, net_name, nc in getattr(module, '_no_connects', []):
        net = net_dict.get(net_name)
        if net is None:
            continue                        # orphaned, not stale (see module doc)
        real = [f"{b.instance}.{b.port}" for b in net.pins
                if getattr(b, 'kind', None) != 'lint_marker']
        if len(real) >= _NET_MARKER_STALE_AT:
            stale.append(StaleMarker(kind='net', reason=nc.reason,
                                     net_name=net_name, wired_to=tuple(real)))

    elem_dict = getattr(module, 'elem_dict', None) or {}
    for marker in getattr(module, '_no_connect_ports', []):
        entry = elem_dict.get(marker.instance)
        device = entry.get('device') if entry else None
        pin = (getattr(device, 'pin_list', None) or {}).get(marker.port)
        net_name = getattr(pin, 'net', None) if pin is not None else None
        if net_name:
            stale.append(StaleMarker(kind='port', reason=marker.reason,
                                     instance=marker.instance, port=marker.port,
                                     wired_to=(str(net_name),)))
    return stale
