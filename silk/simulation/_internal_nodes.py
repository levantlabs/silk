"""Admission for the explicitly requested ngspice internal-node filter."""
from silk.exceptions import NotSupportedError
from silk.simulation._ngspice_compat import NgspiceCompatibility


def _refuse(reason):
    raise NotSupportedError(
        f'omit_device_internal_nodes=True: {reason}. Keep '
        'omit_device_internal_nodes=False, use set_save_list() with internals '
        'kept, or use a compatible ngspice and verified model content.')


def require_filterable(module, corner):
    """Refuse unsafe requests before any output deck is emitted."""
    if module.simulator != 'ngspice':
        _refuse('filtering requires ngspice')
    policy = NgspiceCompatibility()
    if not policy.has('save_nointernals'):
        _refuse(f'ngspice {policy.version or "unknown"} lacks save_nointernals; '
                f'ngspice {policy.since("save_nointernals")} or newer is required')
    if getattr(module, '_save_list', None) is not None:
        _refuse('filtering cannot combine with set_save_list()')
    from silk.simulation.completion import _opaque_emission_reason
    reason = _opaque_emission_reason(module)
    if reason:
        _refuse(f'unverified custom content: {reason}')
    _verify_cards(module)
    from silk.circuit.spice_netlister import module_uses_pdk_devices
    if module_uses_pdk_devices(module):
        return _verify_models(module, corner)
    return 'built-in-ideal'


def _certificate():
    from importlib.resources import files
    import yaml
    return yaml.safe_load(files('silk.pdk').joinpath(
        'data/sky130-internal-node-certificate.yaml').read_text())


def _verify_models(module, corner):
    """Match a finite audited manifest; never parse arbitrary SPICE includes.

    Re-hash on every emission (including the launch freshness check). No stat
    cache can conceal edits made between prepare and simulate.
    """
    from hashlib import sha256
    from pathlib import Path
    from silk import settings
    from silk.circuit.circuit_elements import ElementType

    cert = _certificate()
    root = (Path(settings.get('PDK_ROOT')) / cert['root']).resolve()
    expected_library = root / cert['library']
    expected_extras = [str(root / name) for name in cert['extra_includes']]
    if getattr(module, '_mc_sample', None) is not None:
        _refuse('unverified Monte Carlo model entry point')
    if corner not in cert['sections']:
        _refuse(f'unverified model section {corner!r}')
    seen = set()

    def visit(mod):
        if id(mod) in seen:
            return
        seen.add(id(mod))
        pdk = mod.pdk_details
        from silk.circuit.spice_netlister import module_uses_pdk_devices
        if module_uses_pdk_devices(mod):
            if (Path(pdk.library_location).resolve() != expected_library
                    or [str(Path(p).resolve()) for p in pdk.extra_includes] != expected_extras
                    or (getattr(pdk, 'corners', {}).get(corner) or {}).get('includes')
                    or getattr(pdk, 'osdi_files', None)):
                _refuse('unverified model bundle or include entry point')
        for entry in mod.elem_dict.values():
            device = entry.get('device')
            if entry.get('type') == ElementType.MODULE:
                visit(device)
            elif getattr(device, '_is_pdk_device', True):
                name = getattr(device, 'device_name', None)
                model = getattr(device, '_pdk_device', getattr(device, 'pdk_device', None))
                if cert['factories'].get(name) != model or model is None:
                    _refuse(f'unverified model factory {name!r}: {model!r}')
        for entry in mod.module_dict.values():
            visit(entry['device'])

    visit(module)
    for relative, record in cert['files'].items():
        path = root / relative
        try:
            # A changed dependency symlink is a resolution change even when
            # its target currently has the same bytes.
            if path.resolve() != path:
                _refuse(f'unverified model dependency resolution: {relative}')
            if sha256(path.read_bytes()).hexdigest() != record['sha256']:
                _refuse(f'unverified model content: {relative}')
            for include in record['includes']:
                if (path.parent / include['token']).resolve() != root / include['target']:
                    _refuse(f'unverified model include resolution: {relative}')
        except OSError as exc:
            _refuse(f'unverified model dependency {relative}: {exc}')
    return cert['id']


def _verify_cards(module):
    """Built-in renderers can still carry opaque string-valued parameters.

    Each device must emit just its declared physical cards. A filter request
    refuses multiline literals and hash tokens even when the renderer itself
    is built in. This is deliberately stricter than ordinary SPICE emission.
    """
    from silk.circuit.circuit_elements import ElementType
    from silk.elements.base import AnalogElement
    from silk.elements import elements
    from silk.circuit.emitted_names import emitted_component
    seen = set()

    def check(text, names, who):
        lines = text.strip().splitlines()
        # Built-in finite bit waveforms wrap PWL points with SPICE '+'
        # continuations. They extend the preceding card, never introduce
        # another directive or physical device.
        actual = [line.split()[0] if line.split() else ''
                  for i, line in enumerate(lines)
                  if i == 0 or not line.lstrip().startswith('+')]
        if '#' in text or actual != list(names):
            _refuse(f'opaque card or multiline literal in {who}')

    def visit(mod):
        if id(mod) in seen:
            return
        seen.add(id(mod))
        for instance, entry in mod.elem_dict.items():
            obj = entry['device']
            if entry.get('type') == ElementType.MODULE:
                visit(obj)
                continue
            if isinstance(obj, AnalogElement):
                if not obj._is_pdk_device:
                    known = getattr(elements, obj.device_name, None)
                    fields = ('_spice_prefix', '_pdk_device', '_pin_names', '_param_names')
                    if known is None or any(getattr(obj, f) != getattr(known, f) for f in fields):
                        _refuse(f'unverified ideal factory {obj.device_name!r}')
                names = [obj._spice_prefix + emitted_component(obj.name)]
                check(obj.to_spice(), names, obj.name)
            else:
                names = [obj.spice_prefix + emitted_component(instance)]
                check(mod.pdk_details.generate_netlist_entry(instance, entry), names, instance)
        for obj in (*mod.sig_sources_dict.values(), *mod.passives_dict.values()):
            if isinstance(obj, AnalogElement):
                names = [obj._spice_prefix + emitted_component(obj.name)]
            else:
                names = obj.physical_instance_names()
            check(obj.to_spice(), names, obj.name)
        for analysis in mod.sim_analysis_dict.values():
            # Built-in analyses emit one directive. An opaque numeric literal
            # must not insert control commands or new include entry points.
            if len(analysis.to_spice().strip().splitlines()) != 1:
                _refuse('opaque multiline analysis literal')
        for value in mod.simulator_options.values():
            if any(c in str(value) for c in ('\n', '\r', '#')):
                _refuse('opaque simulator option literal')
        for entry in mod.module_dict.values():
            visit(entry['device'])

    visit(module)
