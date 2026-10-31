"""BusSpec — value-typed metadata sidecar for bus ports.

Stored on Module._buses. Carries only str | int | tuple | enum fields so
it survives the existing Module.__deepcopy__ swallow-on-error path
(modules.py:241) without losing bus identity.
"""

from dataclasses import dataclass
from typing import Optional, Union


def indices_from_bits(bits: Union[int, range, slice]) -> tuple:
    """Normalize the user's `bits` arg into a tuple of ints.

    Accepts:
      - int N   →  (0, 1, ..., N-1)
      - range(start, stop) with step==1
      - slice(start, stop) with step==1 or None
    """
    if isinstance(bits, int):
        if bits <= 0:
            raise ValueError(
                f"bits={bits!r}: zero-width or negative buses are not allowed"
            )
        return tuple(range(bits))
    if isinstance(bits, range):
        if bits.step != 1:
            raise ValueError(
                f"bits={bits!r}: step must be 1 (got step={bits.step})"
            )
        if len(bits) <= 0:
            raise ValueError(f"bits={bits!r}: empty range")
        return tuple(bits)
    if isinstance(bits, slice):
        step = bits.step if bits.step is not None else 1
        if step != 1:
            raise ValueError(
                f"bits={bits!r}: slice step must be 1 or None (got step={step})"
            )
        start = bits.start if bits.start is not None else 0
        stop = bits.stop
        if stop is None:
            raise ValueError(f"bits={bits!r}: slice must have explicit stop")
        if stop <= start:
            raise ValueError(f"bits={bits!r}: empty slice")
        return tuple(range(start, stop))
    raise TypeError(
        f"bits must be int, range, or slice; got {type(bits).__name__}"
    )


def _format_pin_name(name: str, idx: int, naming: str) -> str:
    if naming == 'bracketed':
        return f'{name}[{idx}]'
    if naming == 'underscore':
        return f'{name}_{idx}'
    raise ValueError(
        f"naming={naming!r}: must be 'bracketed' or 'underscore'"
    )


def expand_bits(name: str, bits, *, naming: str = 'bracketed') -> tuple:
    """Return the tuple of expanded scalar pin names for a bus declaration.

    >>> expand_bits('d', 4)
    ('d[0]', 'd[1]', 'd[2]', 'd[3]')
    >>> expand_bits('addr', range(8, 12))
    ('addr[8]', 'addr[9]', 'addr[10]', 'addr[11]')
    >>> expand_bits('dbg', 4, naming='underscore')
    ('dbg_0', 'dbg_1', 'dbg_2', 'dbg_3')
    """
    indices = indices_from_bits(bits)
    return tuple(_format_pin_name(name, i, naming) for i in indices)


@dataclass(frozen=True)
class BusSpec:
    """Value-typed bus metadata. ALL fields must be str | int | tuple | enum.

    No live references to Net / Module / VoltageDomain. The domain is stored
    as the domain's ``name`` string and re-resolved at read time.
    """
    name: str
    indices: tuple        # tuple[int, ...]
    direction: 'PortDir'  # enum
    role: Optional['PortRole']
    domain: Optional[str]  # VoltageDomain.name, NOT the live object
    naming: str            # 'bracketed' | 'underscore'

    @property
    def width(self) -> int:
        return len(self.indices)

    def expanded_names(self) -> tuple:
        return tuple(
            _format_pin_name(self.name, i, self.naming) for i in self.indices
        )


def _bus_claims(buses):
    """Case-folded alias/member ownership, derived for one declaration batch."""
    return {name.casefold(): (spec, name)
            for spec in buses.values()
            for name in (spec.name, *spec.expanded_names())}


def _bus_collision(module, spec, declaration, member):
    from silk.exceptions import DuplicateNameError

    case_note = (f" {declaration!r} differs only by case from {member!r}."
                 if declaration != member and declaration.casefold() == member.casefold()
                 else '')
    return DuplicateNameError(
        f"module {module.module_name or '<unnamed>'!r}: bus {spec.name!r} "
        f"owns alias or bit {member!r}; conflicting declaration {declaration!r} "
        f"cannot claim it.{case_note} Bus bits: {', '.join(spec.expanded_names())}. "
        "Select a bit or rename the conflicting declaration."
    )


def _guard_scalar_alias(module, name, *, connection=False):
    """Electrical bit nets are legal; a bus's logical alias is not a net."""
    spec = (getattr(module, '_buses', {}) or {}).get(name)
    if spec is None:
        spec = next((bus for key, bus in
                     (getattr(module, '_buses', {}) or {}).items()
                     if key.casefold() == name.casefold()), None)
    if spec is not None:
        if connection:
            from silk.exceptions import DuplicateNameError
            case_note = (f" {name!r} differs only by case from {spec.name!r}."
                         if name != spec.name else '')
            raise DuplicateNameError(
                f"module {module.module_name or '<unnamed>'!r}: bus {spec.name!r} "
                f"cannot be wired as a scalar net {name!r}.{case_note} "
                f"Its bits are {', '.join(spec.expanded_names())}. Select a bit, "
                f"or pass module.nets[{spec.name!r}] to a matching child bus port."
            )
        raise _bus_collision(module, spec, name, spec.name)


def _formal_bus(device, name):
    """Return a bus only when all its members are actual formal ports."""
    spec = (getattr(device, '_buses', {}) or {}).get(name)
    pins = getattr(device, 'pin_list', None) or {}
    if spec is not None and all(bit in pins for bit in spec.expanded_names()):
        return spec
    return None


def _port_targets(device, name):
    """Expand a formal bus target; scalar/unknown targets retain their name."""
    spec = _formal_bus(device, name)
    return spec.expanded_names() if spec is not None else (name,)
