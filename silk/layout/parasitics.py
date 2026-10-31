"""silk.layout.parasitics — lumped RC and shunt-C loading injection.

Public API:
    inject_parasitics(module, estimate, mode='rc', *, return_net='0') -> None

Modes
-----
'c_only'
    One shunt capacitor per net (``__par_C_<net>`` from net to ``return_net``).
    Original terminals stay on the net; no resistor or midpoint is added.
    Suitable for first-order capacitive-loading estimates, not signoff.

'rc'
    Lumped-π loading branch per net:
    - ``__par_R_<net>``   : resistor from net to ``__par_<net>_mid``
    - ``__par_C1_<net>``  : shunt capacitor from net to ``return_net``  (C/2)
    - ``__par_C2_<net>``  : shunt capacitor from mid to ``return_net``  (C/2)
    The midpoint net ``__par_<net>_mid`` is added to ``module.net_dict``.
    A trailing index stays last: ``data[3]`` gets ``__par_data_mid[3]``.
    Every original terminal stays on the original net. Only the added R
    and C2 connect to the midpoint, so R is in series with C2, not between
    original terminals. This models added loading, not DC wire drop,
    branch-specific resistance, or routed inter-terminal delay.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from silk.circuit.modules import Module
    from silk.layout.estimator import LayoutEstimate

_VALID_MODES = ('rc', 'c_only')


def _midpoint_name(net_name: str) -> str:
    """Insert the midpoint suffix before an authored net's optional index."""
    stem, bracket, index = net_name.partition('[')
    return f'__par_{stem}_mid{bracket}{index}'


def _remove_previous_parasitics(module: 'Module') -> None:
    """Remove only recorded injected objects, including their pin bindings."""
    passives, nets = getattr(module, '_parasitic_ownership', (set(), set()))
    for name in passives:
        passive = module.passives_dict.get(name)
        if passive is not None:
            module._element_mgr._retire_instance('passives_dict', name, passive)
            module.passives_dict.pop(name, None)
    for net in module.net_dict.values():
        net.pins[:] = [pin for pin in net.pins
                       if not (pin.kind == 'passive' and pin.instance in passives)]
    graph = module._element_mgr
    for name in passives:
        for port in ('pos', 'neg'):
            graph._pin_to_net.pop((name, port), None)
    for name in nets:
        if name in module.net_dict:
            module.remove_net(name)


def inject_parasitics(
    module: 'Module',
    estimate: 'LayoutEstimate',
    mode: str = 'rc',
    *,
    return_net: str = '0',
) -> None:
    """Inject lumped parasitic elements into *module* from a layout estimate.

    Replaces passives and nets recorded by the previous injection. Unrecorded
    objects are never removed: a stray ``gnd`` from older silk remains until
    the module is rebuilt. An estimated net equal to the return gets no
    branch in either mode.

    Placed children are live references: injection before or after placement
    reaches the parent's next deck and a testbench's next run. A return port
    follows each placement's binding; the default ``'0'`` is global ground.

    Original terminal connections are unchanged in both modes. In ``'rc'``,
    R connects the original net to a new midpoint whose only other terminal
    belongs to C2; C1 connects the original net to the chosen return. Each
    capacitor receives half the estimated C. This is a loading approximation, not
    inter-terminal series resistance or an extracted-layout model.

    Parameters
    ----------
    module : Module
        The module to inject into. ``passives_dict`` and ``net_dict`` are
        modified in place.
    estimate : LayoutEstimate
        The source of per-net resistance (``r``) and capacitance (``c``)
        scalars (ohms and fF respectively; capacitance is converted to F).
        Only nets present in ``estimate.nets`` are processed.
    mode : str
        ``'rc'`` (default) for a lumped-π loading branch, ``'c_only'`` for
        one lumped shunt capacitor with the full estimated capacitance.
    return_net : str
        Capacitor return, default ``'0'`` (global ground). Any other return
        must already be declared in the module, including a port such as
        ``'vss'``. Names follow the module's case-insensitive net convention.

    Raises
    ------
    ValueError
        If *mode* is not one of ``'rc'``, ``'c_only'``.
    UndeclaredNetError
        If the return (other than ``'0'``) or an estimated net is undeclared.
        Mode and all net names are validated before removing old parasitics.
    DuplicateNameError
        If a generated name has a destination-slot or emitted SPICE card
        conflict with a retained placement in any registry or another
        candidate in the proposed batch, or a generated midpoint collides
        with a declared net.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"inject_parasitics: mode must be one of {_VALID_MODES!r}, got {mode!r}"
        )

    from silk.elements import elements
    from silk.exceptions import DuplicateNameError, UndeclaredNetError

    old_passives, old_nets = getattr(module, '_parasitic_ownership', (set(), set()))
    # Validate against the circuit that will remain after cleanup, without
    # changing the graph's canonical-name cache or creating implicit nets.
    declared = {name.casefold(): name for name in module.net_dict
                if name not in old_nets}

    def resolve(name, *, allow_ground=False):
        if isinstance(name, str):
            if allow_ground and name == '0':
                return name
            if name.casefold() in declared:
                return declared[name.casefold()]
        raise UndeclaredNetError(
            f"inject_parasitics: net {name!r} is not declared in the module; "
            "declare it before injection (or use return_net='0')."
        )

    for name in module.pin_list or ():
        declared.setdefault(name.casefold(), name)

    return_net = resolve(return_net, allow_ground=True)
    branches = []
    for name, layout in estimate.nets.items():
        net_name = resolve(name, allow_ground=name == '0')
        if net_name != return_net:
            branches.append((net_name, layout, _midpoint_name(net_name)))

    # Preflight generated names as well, so collisions cannot leave a partial
    # replacement after the previous injection has been removed.
    for _, _, mid in branches:
        if mode == 'rc' and mid.casefold() in declared:
            raise DuplicateNameError(f"inject_parasitics: midpoint net {mid!r} already exists")

    passives = []
    required_nets = set()
    # Constructors normally auto-register in ``with module:``. Stage without
    # that side effect, even when the active module is not our target.
    from silk.circuit._module_base import _push_active, _pop_active
    _push_active(None)
    try:
        for net_name, nl, mid_name in branches:
            c_total_f = nl.c_farads     # fF → F, via the sole accessor (TASK-25)
            r_ohm     = nl.r            # already in Ω

            if c_total_f < 0:
                raise ValueError(f'capacitance must be >= 0, got {c_total_f}')
            if mode == 'rc' and r_ohm < 0:
                raise ValueError(f'resistance must be >= 0, got {r_ohm}')

            if mode == 'c_only':
                passives.append(
                    elements.capacitor(
                        name=f'__par_C_{net_name}',
                        p=net_name,
                        n=return_net,
                        value=c_total_f,
                    )
                )

            else:  # 'rc' — lumped-π
                required_nets.add(mid_name)

                passives.append(
                    elements.resistor(
                        name=f'__par_R_{net_name}',
                        p=net_name,
                        n=mid_name,
                        value=r_ohm,
                    )
                )
                c_half_f = c_total_f / 2.0
                passives.append(
                    elements.capacitor(
                        name=f'__par_C1_{net_name}',
                        p=net_name,
                        n=return_net,
                        value=c_half_f,
                    )
                )
                passives.append(
                    elements.capacitor(
                        name=f'__par_C2_{net_name}',
                        p=mid_name,
                        n=return_net,
                        value=c_half_f,
                    )
                )

            required_nets.update((net_name, return_net))
    finally:
        _pop_active()

    graph = module._element_mgr
    retiring = {}
    for name in old_passives:
        old = module.passives_dict.get(name)
        if old is not None:
            for token in graph._placement_tokens.get(('passives_dict', name), ()):
                retiring[token.lower()] = ('passives_dict', name, old)
    staged = {}
    staged_slots = set()
    for passive in passives:
        if passive.name in staged_slots:
            raise DuplicateNameError(
                f"inject_parasitics: exact destination-slot clash for {passive.name!r}.")
        tokens = graph._check_instance_name(
            'passives_dict', passive.name, passive, staged=staged, retiring=retiring)
        staged_slots.add(passive.name)
        for token in tokens:
            staged[token.lower()] = ('passives_dict', passive.name, passive)

    _remove_previous_parasitics(module)
    added_nets = required_nets - set(module.net_dict)
    for name in sorted(added_nets):
        module.add_net(name)
    for passive in passives:
        module.add_passive(passive)
    # Cleanup consumes the same helper-generated names used by preflight and construction.
    module._parasitic_ownership = ({p.name for p in passives}, added_nets)
