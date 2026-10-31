"""SPICE netlist generation from circuit modules and simulation settings."""

# Python based SPICE netlist generator
# Maintained by the silk contributors
# Generates ngspice-format netlists from circuit data
# This separates netlist formatting from circuit data management

import copy
import os
import warnings

from silk.simulation.run_files import RunFiles
from silk.circuit.circuit_elements import ElementType
from silk.elements.base import AnalogElement
from silk.exceptions import CircuitError, DuplicateNameError, UndeclaredNetError, SimulationError, PDKError
from silk.simulation.analysis import AC, DC, Noise, Op, Transient


def _source_maps(sig_sources_dict):
    """Build the shared physical source-name maps for DC, Noise and notices."""
    _src_names = {name: getattr(src, 'instance_name', name)
                  for name, src in (sig_sources_dict or {}).items()}
    _targets = {name: (src.physical_layout().dc_name
                      if hasattr(src, 'physical_layout') else _src_names[name])
                for name, src in (sig_sources_dict or {}).items()}
    _emitted_targets = {str(_src_names[n]).lower(): target for n, target in _targets.items()}
    _by_lower = {}
    for name in _src_names:
        _by_lower.setdefault(str(name).lower(), _targets[name])
    _emitted_lower = {str(e).lower() for e in _src_names.values()}

    return _src_names, _targets, _emitted_targets, _by_lower, _emitted_lower


def _resolve_dc(sig_sources_dict, n):
    """Resolve DC using emission's precedence and physical source layout."""
    (_src_names, _targets, _emitted_targets,
     _by_lower, _emitted_lower) = _source_maps(sig_sources_dict)

    # Emitted source name (case-insensitive) first: the historic
    # pass-through, so existing decks stay byte-identical. Then a
    # designer name (exact, then case-insensitive, the mapping
    # probe('i(vs)') applies) -> its emitted name.
    if str(n).lower() in _emitted_lower:
        target = _emitted_targets[str(n).lower()]
        return n if target.lower() == str(n).lower() else target
    if n in _src_names:
        return _targets[n]
    return _by_lower.get(str(n).lower(), n)


#: ngspice's own name for the plot each analysis leaves behind, used to
#: qualify the ``.control`` ``write`` when a deck declares more than one
#: analysis. nutmeg numbers plots per analysis TYPE (``tran1``,
#: ``ac1``, …) and ``sim_analysis_dict`` is keyed by analysis class, so a
#: silk deck can hold at most one of each and the ``1`` suffix is exact.
#: MEASURED on ngspice-42 with a four-analysis RC deck: ``write <path>
#: tran1.all ac1.all dc1.all noise1.all`` produces a four-plot rawfile whose
#: plots appear in the ARGUMENT order, named ``Transient Analysis`` /
#: ``AC Analysis`` / ``DC transfer characteristic`` / ``Noise Spectral
#: Density Curves``. ``.noise`` also leaves an integrated-totals plot
#: (``noise2``); ``noise1`` is the spectral-density one, the same plot the
#: single-analysis noise path selects with ``setplot noise1``.
#: ``Op: 'op1'`` is measured the same way: an explicit ``op``
#: command (or a deck's lone ``.op`` card) leaves a plot ngspice 42/47 both
#: name ``Operating Point``, retained under ``op1`` — see's
#: ``op_probe_ac.sp`` fixture, whose ``setplot`` listing shows ``op1``
#: alongside ``ac1``/``ac2`` after an explicit ``op``.
NGSPICE_PLOT_NAMES = {
    Transient: 'tran1',
    AC: 'ac1',
    DC: 'dc1',
    Noise: 'noise1',
    Op: 'op1',
}


def _ngspice_plot_names(sim_analysis_dict, save_device_params=False):
    """The nutmeg plot names for a deck's declared analyses, in deck order.

    Deck order is ``sim_analysis_dict`` order, i.e. ``add_sim_type`` order —
    NOT the order ngspice runs them in (measured: ngspice runs ``.ac`` before
    ``.tran`` whatever the deck says). The write-argument order is what sets
    the order of the plots in the rawfile, so the file reads the way the user
    declared the test.

    An analysis class with no entry in :data:`NGSPICE_PLOT_NAMES` is skipped
    rather than guessed at: a name nutmeg does not know would make ``write``
    fail for the whole deck, losing the plots that WERE nameable.

    save_device_params: an ``Op`` entry is skipped here too, the
    same way :meth:`PySpiceNetlister.generate_analysis_commands` skips its
    netlist card — ``op1`` is appended exactly once by
    :meth:`generate_output_block` in this mode (the interactive post-run
    sequence produces it, not a declared analysis's own netlist card), so
    listing it here as well would duplicate it in the ``write`` line.
    """
    names = []
    for analysis in (sim_analysis_dict or {}).values():
        if save_device_params and isinstance(analysis, Op):
            continue
        name = NGSPICE_PLOT_NAMES.get(type(analysis))
        if name is not None:
            names.append(name)
    return names


def _quote_save_argument(vector: str) -> str:
    """Quote a complete generated nutmeg argument, never its raw identity.

    Quoting cannot hold a line break, so one is refused."""
    from silk.utils.deck_text import card_text
    vector = card_text(vector, who='saved vector name')
    return '"' + vector.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _quote_if_needed(name: str) -> str:
    """Wrap a node name in double quotes if it contains characters that
    conflict with nutmeg's vector-slice syntax in .control/.print/.save
    contexts. Brackets [ and ] trigger quoting; other characters do not.

    Bare bracketed names work in the netlist proper, but
    nutmeg parses ``v(d[3])`` as a vector slice unless the name is quoted.
    Backslash escaping does NOT work in ngspice; double-quote is the only
    workaround.
    """
    if '[' in name or ']' in name:
        return f'"{name}"'
    return name


def deck_uses_pdk_devices(elem_dict, module_dict=None, custom_lines=None,
                          _seen=None):
    """Does this deck need the PDK model library, walking the whole hierarchy?

    A netlist of nothing but ideal elements, passives and sources is valid
    SPICE on its own — it needs no ``.lib``/``.include`` at all, and so runs
    under a bare simulator with no PDK on disk (silk's cheapest on-ramp rung:
    ``pip install`` + ``apt install ngspice``). This walk decides whether the
    include block may be omitted.

    It answers **"is the answer provably no?"**, never "is it probably no":
    a false negative would drop the model library out of a deck that needs
    it, and the failure would land in the simulator as an unknown-model parse
    error far from the cause. So everything the walk does not positively
    recognise as ideal counts as a PDK device:

    * an :class:`~silk.elements.base.AnalogElement` says for itself, via
      ``_is_pdk_device`` — ``False`` only for the bundled ideal catalog
      (:class:`~silk.elements.library.ElementLibrary`), ``True`` for anything
      built from a PDK's device table;
    * a legacy ``PDKDevice`` entry (emitted through
      ``pdk_details.generate_netlist_entry``) is a PDK device by definition;
    * an ``elem_dict`` entry of an unrecognised shape counts as one;
    * ANY ``add_custom_line`` text counts as one. Custom lines are opaque
      SPICE handed straight to the simulator — a hand-registered subckt, a
      stdcell instantiation, an ``.include`` of the user's own — so their
      mere presence keeps the library. That is deliberately blunt: a
      ``.param``-only testbench keeps the includes it does not strictly
      need, which costs nothing, while parsing the text to decide would be
      exactly the heuristic that can be wrong.

    Sources and native passives (``R``/``C``/``L`` with a value) are always
    ideal and never consulted.

    Parameters
    ----------
    elem_dict, module_dict, custom_lines
        The same three collections ``generate_complete_netlist`` emits from.
    _seen : set, optional
        Internal cycle guard (module ``id()``s already being answered).

    Returns
    -------
    bool
        ``True`` when the deck must keep its PDK include block.
    """
    if custom_lines:
        return True
    if _seen is None:
        _seen = set()

    for entry in (elem_dict or {}).values():
        if not isinstance(entry, dict):
            return True
        device = entry.get('device')
        if entry.get('type') == ElementType.MODULE:
            if _submodule_uses_pdk_devices(device, _seen):
                return True
            continue
        if isinstance(device, AnalogElement):
            # Default True: a subclass that somehow lost the attribute is not
            # provably ideal.
            if getattr(device, '_is_pdk_device', True):
                return True
            continue
        return True

    # Definition copies, which carry the bodies actually emitted as .subckt.
    for entry in (module_dict or {}).values():
        sub = entry.get('device') if isinstance(entry, dict) else None
        if sub is None or _submodule_uses_pdk_devices(sub, _seen):
            return True

    return False


def _submodule_uses_pdk_devices(module, seen):
    """:func:`deck_uses_pdk_devices` for a child ``Module``, cycle-guarded.

    A module already in *seen* is either being answered further up this same
    walk or was answered ``False`` — a ``True`` short-circuits the whole walk
    before it could be revisited — so returning ``False`` here can only ever
    repeat an answer, never invent one.
    """
    if module is None:
        return True
    key = id(module)
    if key in seen:
        return False
    seen.add(key)
    return deck_uses_pdk_devices(
        getattr(module, 'elem_dict', None),
        getattr(module, 'module_dict', None),
        getattr(module, 'custom_lines', None),
        _seen=seen,
    )


def module_uses_pdk_devices(module):
    """:func:`deck_uses_pdk_devices` for a whole :class:`~silk.circuit.modules.Module`.

    The simulate-time gates (``_require_pdk_library``,
    ``_require_osdi_compiled``) ask this so they demand exactly what the deck
    that was emitted actually needs.
    """
    return deck_uses_pdk_devices(
        getattr(module, 'elem_dict', None),
        getattr(module, 'module_dict', None),
        getattr(module, 'custom_lines', None),
    )


class PySpiceNetlister:
    """Generates ngspice-format netlists from circuit data"""

    #: Whether the last-emitted deck kept its PDK include block. Set per
    #: emission by ``generate_complete_netlist``; ``True`` by default so any
    #: other entry point (a bare netlister built via ``__new__``, a subclass
    #: calling ``generate_control_block`` directly) behaves exactly as before.
    _needs_pdk_library = True
    _source_backend = 'ngspice'

    def __init__(self, pdk_details):
        """Initialize netlister with PDK information

        Args:
            pdk_details: PDK object with device models and netlist generation methods
        """
        self.pdk_details = pdk_details

    def generate_header(self, circuit_name='circuit'):
        """Generate a circuit title followed by provenance comments.

        Args:
            circuit_name: Name of the circuit. Line breaks are flattened to
                spaces; a blank name uses 'circuit'. Module callers normally
                supply the netlist filename when no explicit name is set.

        Returns:
            String with header comments
        """
        # SPICE uses line one as the rawfile title. Keep names on that single
        # comment line so embedded newlines cannot become deck directives.
        title = ' '.join(str(circuit_name or '').splitlines()).strip() or 'circuit'
        header = f'*** Circuit: {title}\n'
        header += '*** Automated netlist generator\n'
        header += '*** Generated by silk\n'
        return header

    def generate_pdk_omitted_note(self):
        """The comment that stands in for the PDK include block when a deck
        provably instantiates no PDK device.

        Emitting nothing at all would leave the reader of a deck guessing
        whether silk forgot the library or decided it was unnecessary, so the
        decision is written into the deck itself — the same provenance the
        ``*** Including PDK library`` header gives the other branch.
        """
        # ASCII only: this text goes into the deck, which is read by every
        # simulator silk emits for, and a comment is no place to find out
        # which of them is byte-oriented.
        return ('*** No PDK library included: this deck instantiates no PDK\n'
                '*** device (ideal elements, passives and sources only), so it\n'
                '*** runs with no PDK installed. One PDK device anywhere in\n'
                '*** the hierarchy, or any add_custom_line() text, restores it.\n')

    #: True when a sample's seed is a deck option (ngspice). Xyce takes it on
    #: the command line instead (``XyceBackend.sample_args``).
    _MC_SEED_AS_OPTION = True

    def _check_mc_blocks(self, module_dict, mc_sample):
        """Hook for a netlister that cannot honour ``set_mismatch(False)`` on
        a block of a Monte Carlo sample deck; *module_dict* is the deck's
        top-level one. The default honours it."""

    def _check_mc_sample_supported(self, corner, mc_sample):
        """Refuse a Monte Carlo sample this netlister cannot honour.

        A sample names ONE library section on ``library_location``; a corner
        that declares per-corner ``includes`` (multi-file PDKs) has no such
        single section, and silently emitting one would drop the other files.
        """
        meta = (getattr(self.pdk_details, 'corners', {}) or {}).get(corner) or {}
        if self._corner_includes(meta):
            from silk.exceptions import NotSupportedError
            name = (getattr(self.pdk_details, 'name', None)
                    or type(self.pdk_details).__name__)
            raise NotSupportedError(
                f'Monte Carlo is not supported for PDK {name!r} at corner '
                f'{corner!r}: the corner declares per-corner library includes '
                f'(several files), and a sample selects one statistical '
                f'section of a single library.')

    def generate_pdk_include(self, corner='tt', sample=None):
        """Generate PDK library include statement

        Args:
            corner: Process corner ('tt', 'ff', 'ss', etc.)
            sample: optional Monte Carlo ``_SampleSpec``; its section (or the
                both-wrapper's) replaces the corner's library section.

        Returns:
            String with .lib statement
        """
        from silk.utils.deck_text import card_text, validate_corner_name
        # The last defence for every deck (main, parameter preflight, Xyce):
        # the corner is the .lib section and each path sits inside one card,
        # so a line break in either would start a new deck line (TASK-593).
        validate_corner_name(corner)

        # The text written (os.fspath for a Path), checked as written.
        def path(p, what):
            return card_text(p, who=f'PDK {what} path')

        def section(s):
            return card_text(s, who='PDK library section')

        header = '*** Including PDK library\n'

        # A corner may declare its own list of (file, section) pairs. PDKs that
        # split device families across separate corner files need this: IHP
        # SG13G2 keeps MOS, resistors, caps, diodes and HBTs in six different
        # .lib files, each with its own section names (mos_tt, res_typ,
        # cap_typ, ...), so ONE corner is six `.lib` lines. The section name is
        # not derivable from the corner name — hence explicit pairs.
        #
        # Without `includes`, the historical behaviour: one line, with the
        # corner name used directly as the section of the single lib_path file.
        # That is what sky130 and GF180MCU use.
        meta = (getattr(self.pdk_details, 'corners', {}) or {}).get(corner) or {}
        includes = self._corner_includes(meta)
        if sample is not None:
            if sample.wrapper_path:
                header += (f'.lib "{path(sample.wrapper_path, "Monte Carlo wrapper")}" '
                           f'{type(sample).WRAPPER_SECTION}\n')
            else:
                header += ('.lib "' + path(self._pdk_library_location(), 'library')
                           + '" ' + section(sample.section) + '\n')
        elif includes:
            for inc in includes:
                sec = inc.get('section')
                header += ('.lib "' + path(inc['file'], 'library') + '" '
                           + section(sec) + '\n'
                           if sec else
                           '.include "' + path(inc['file'], 'include') + '"\n')
        else:
            header += ('.lib "' + path(self._pdk_library_location(), 'library')
                       + '" ' + corner + '\n')
        seen = set()
        for inc in self._pdk_extra_includes():
            if inc not in seen:
                seen.add(inc)
                header += f'.include "{path(inc, "extra include")}"\n'
        return header

    # The three hooks below read the ngspice-oriented manifest fields
    # (`includes`, `library_location`, `extra_includes`). XyceNetlister
    # overrides them to prefer the `xyce_*` siblings a manifest may declare
    # (silk/pdk/_loader.py), falling back to these same fields when a PDK
    # declares none — so a PDK with no Xyce-specific models is unaffected.
    def _corner_includes(self, meta):
        return meta.get('includes')

    def _pdk_library_location(self):
        return self.pdk_details.library_location

    def _pdk_extra_includes(self):
        return getattr(self.pdk_details, 'extra_includes', []) or []

    def generate_module_definitions(self, module_dict, terminal_current_method=None,
                                    ammeter_wrappers=None, emitted=None,
                                    mc_sample=None):
        """Generate subcircuit definitions for modules

        Args:
            module_dict: Dictionary of modules from PyCircuitElements
            terminal_current_method: propagated to submodules so in-module
                devices get ammeter-wrapped consistently (or None = default).
            ammeter_wrappers: shared dict the whole hierarchy registers wrapper
                defs into (deduped by model), so they can be hoisted to top.
            emitted: shared set of subckt names already written anywhere in
                this deck. ``uniqueModules`` only dedupes within ONE level,
                but each recursive ``generate_netlist(subckt=True)`` re-emits
                its own children — so a module instantiated from two parents
                (``div24`` holds ``div8`` and ``div3``, both holding
                ``nand2``) got its definition written once per parent. The
                copies were byte-identical, so ngspice's pick-one behaviour
                was harmless, but redefinition has no meaning in SPICE and a
                stricter parser refuses the deck outright.

        Returns:
            String with all .subckt definitions
        """
        str_module = ''
        if emitted is None:
            emitted = set()
        uniqueModules = list(module_dict.keys())

        if self.pdk_details.debug if hasattr(self.pdk_details, 'debug') else False:
            print(f"Generating subcircuits for modules: {uniqueModules}")

        from silk.circuit.module_placement import resolved_name, _fault_node, _fault_body
        node = _fault_node.get()
        definitions = [(resolved_name(mod), module_dict[mod]['device'], None, mod)
                       for mod in uniqueModules]
        if node:
            definitions.extend((child['name'], child['module'], child, None)
                               for child in node['children'].values())
        for canonical, sub, child_node, mod in definitions:
            if canonical.casefold() in emitted:
                continue
            emitted.add(canonical.casefold())
            # These settings belong to this deck only. Preserve references and
            # absence so a later direct child export starts in its own context.
            unset = object()
            saved = {name: getattr(sub.netlister, name, unset) for name in (
                '_terminal_current_method', '_ammeter_wrappers',
                '_emitted_subckts', '_mc_sample', '_subckt_name',
                '_source_backend')}
            str_netlist = '\n*** subckt definition for {}\n'.format(canonical)
            try:
                sub.netlister._terminal_current_method = (
                    terminal_current_method or 'savecurrents')
                # Keep the active deck's accumulators shared during recursion.
                sub.netlister._ammeter_wrappers = (
                    ammeter_wrappers if ammeter_wrappers is not None else {})
                sub.netlister._emitted_subckts = emitted
                sub.netlister._mc_sample = mc_sample
                sub.netlister._subckt_name = canonical
                sub.netlister._source_backend = self._source_backend
                with _fault_body(child_node):
                    str_netlist += sub.generate_netlist(subckt=True)
            finally:
                for name, value in saved.items():
                    if value is unset:
                        delattr(sub.netlister, name)
                    else:
                        setattr(sub.netlister, name, value)
            if mod is not None:
                module_dict[mod]['defined'] = True
            str_module += str_netlist

        return str_module

    def generate_device_instances(self, elem_dict):
        """Generate device instantiation lines

        Args:
            elem_dict: Dictionary of elements from PyCircuitElements

        Returns:
            String with all device instantiations
        """
        str_devices = ''

        from silk.circuit.module_placement import _fault_node
        node = _fault_node.get()
        for idx, k in enumerate(elem_dict.keys()):
            entry = elem_dict[k]
            device = node['devices'].get(k, entry['device']) if node else entry['device']

            # AnalogElement instances carry their own to_spice() method;
            # delegate entirely to them. Legacy PDKDevice entries fall
            # through to the pdk_details.generate_netlist_entry() path below.
            if isinstance(device, AnalogElement):
                str_devices += f'*** {k}.  Connections = {list(device._pin_names)}.\n'
                pending = getattr(device, '_silk_pending_fault', None)
                if pending is not None:
                    str_devices += _emit_faulted_device(
                        self, device, k)
                else:
                    str_devices += self._emit_analog_element(device)
                continue

            if getattr(device, 'pin_list', 0) is None:
                from silk.circuit.modules_netlist import require_declared_ports
                require_declared_ports(device, f"Cannot instantiate {k!r}")
            pin_list = entry.get('pin_list', device.pin_list)

            # Add comment with connections
            str_comment = '*** {}.  Connections = {}.\n'.format(k, list(pin_list.keys()))
            str_devices += str_comment

            # Generate netlist entry based on type
            if entry['type'] == ElementType.DEVICE:  # Device
                # Check all pins are connected
                for p in pin_list.keys():
                    if pin_list[p].net is None:
                        raise CircuitError(
                            'ERROR: pin {} in element {} is still undefined.'.format(p, k))

                if getattr(device, '_silk_pending_fault', None) is not None:
                    # Same fault handling as a factory-built device (TASK-416):
                    # a marked device never falls back to its clean line.
                    str_netlist = _emit_faulted_device(
                        self, device, k)
                else:
                    str_netlist = self._emit_pdk_device(k, entry)

            elif entry['type'] == ElementType.MODULE:  # Module (subcircuit instance)
                # Generate subcircuit instantiation
                from silk.circuit.module_placement import instance_line
                str_netlist = instance_line(k, entry)

            else:
                str_netlist = '\n'

            str_devices += str_netlist

        return str_devices

    def _emit_pdk_device(self, name, entry):
        """The one emission path for a stored ``PDKDevice``
        (``add_element(pdk.cls, ...)``): the PDK's own
        ``generate_netlist_entry``, wrapped in terminal ammeters in ammeter
        mode. A faulted device's instances go through here too."""
        line = self.pdk_details.generate_netlist_entry(name, entry)
        if getattr(self, '_terminal_current_method', 'savecurrents') == 'ammeter_wrapper':
            # Wrap the device in a subckt with 0 V terminal ammeters, so
            # terminal currents are readable as i(v.x<inst>.vamm_*) —
            # simulator-independent, no drain-only/polarity limitation.
            return self._ammeter_wrap_line(
                list(entry['device'].pin_list.keys()), line)
        return line

    def _emit_analog_element(self, device):
        """The one emission path for an :class:`AnalogElement`: its own
        ``to_spice()``, wrapped in terminal ammeters in ammeter mode unless
        it is an ideal passive. Faulted devices and the elements a fault
        adds go through here too."""
        from silk.circuit.modules_results import _is_ideal_passive

        if (getattr(self, '_terminal_current_method', 'savecurrents') == 'ammeter_wrapper'
                and not _is_ideal_passive(device)):
            return self._ammeter_wrap_line(
                list(device._pin_names), device.to_spice())
        return device.to_spice()

    def _ammeter_wrap_line(self, terms, raw_line):
        """Wrap an already-emitted device line ``XM<name> <nets> <model>
        <params>`` in a 0 V-ammeter subckt and return the instance line.
        Registers the wrapper def (once per model) in ``self._ammeter_wrappers``.
        Terminal current of instance ``XM<name>`` is then the branch current
        ``i(v.xm<name>.vamm_<term>)`` (uniform sign: into-net = −i(vamm)).

        Works for both device representations — ``PDKDevice`` (via
        ``generate_netlist_entry``) and ``AnalogElement`` (via ``to_spice()``) —
        because both emit the same line format; the caller supplies the terminal
        list and the raw line, so the device physics stays byte-identical.
        """
        nterms = len(terms)
        toks = raw_line.strip().split()
        inst_name = toks[0]                              # 'XM'+name
        nets = toks[1:1 + nterms]
        model = toks[1 + nterms]
        param_toks = toks[2 + nterms:]                   # ['m=1','w=1.0',...]
        key = 'amm_' + model

        if key not in self._ammeter_wrappers:
            self._ammeter_wrappers[key] = self._build_ammeter_wrapper(
                key, terms, model, param_toks)

        return (inst_name + ' ' + ' '.join(nets) + ' ' + key
                + (' ' + ' '.join(param_toks) if param_toks else '') + '\n')

    @staticmethod
    def _build_ammeter_wrapper(key, terms, model, param_toks):
        """Build the ``.subckt`` def: 0 V ammeter per terminal + the device on
        internal nodes. Params are threaded through so one wrapper serves every
        instance of the model."""
        internal = [t + '_i' for t in terms]
        param_names = [pt.split('=', 1)[0] for pt in param_toks]
        lines = ['.subckt ' + key + ' ' + ' '.join(terms)
                 + (' ' + ' '.join(param_toks) if param_toks else '')]
        for t, i in zip(terms, internal):
            lines.append(f'Vamm_{t} {t} {i} 0')          # 0 V ammeter, term→internal
        inner_params = ' '.join(f'{n}={{{n}}}' for n in param_names)
        lines.append('XMdev ' + ' '.join(internal) + ' ' + model
                     + (' ' + inner_params if inner_params else ''))
        lines.append('.ends')
        return '\n'.join(lines) + '\n'

    def generate_sources(self, sig_sources_dict, sig_source_details=None):
        """Generate voltage and current source lines

        Args:
            sig_sources_dict: Dictionary mapping source name → source object
            sig_source_details: Unused; kept for signature compatibility

        Returns:
            String with all source definitions
        """
        str_sources = ''

        from silk.sources._timing import validate_source_timing
        from silk.sources._operating_point import source_spice

        for name, source in sig_sources_dict.items():
            validate_source_timing(source, self._source_backend)
            str_sources += f'*** {type(source).__name__}: {name}\n'
            str_sources += source_spice(source, self._source_backend)

        return str_sources

    def generate_custom_lines(self, custom_lines):
        """Generate arbitrary custom SPICE lines (add_custom_line entries)

        Args:
            custom_lines: List of custom SPICE lines

        Returns:
            String with all custom lines
        """
        str_custom = ''

        if custom_lines:
            str_custom += '\n*** Custom circuit elements\n'
            for line in custom_lines:
                import re
                from silk.exceptions import CircuitError
                if (not line.lstrip().lower().startswith(('*', '.include', '.lib'))
                        and re.search(r'(?i)(?:^|[^\w])__silk_', line)):
                    raise CircuitError('Custom netlist content uses the reserved __silk_ '
                                       'net namespace; rename the authored net. External '
                                       'include contents must honor this reservation too.')
                str_custom += line + '\n'

        return str_custom

    def generate_analysis_commands(self, sim_analysis_dict, sig_sources_dict=None,
                                   simulator='ngspice', save_device_params=False,
                                   emitted_deck=''):
        """Generate simulation analysis commands

        Args:
            sim_analysis_dict: Mapping of name -> AnalysisBase object
                (AC/Transient/DC/Noise/Op); each is rendered by the backend.
            sig_sources_dict: Sources, used to resolve analysis source names to their
                netlist instance names via the backend renderer.
            simulator: Simulator backend name (default 'ngspice').
            emitted_deck: Already emitted body, for private analysis-name allocation.
            save_device_params: On ngspice, when True, an
                ``Op`` entry is skipped here — never rendered as its own
                netlist-level ``.op`` card. A pre-existing netlist ``.op``
                card claims ngspice's ``op1`` plot slot during the main
                ``run``, before the ``'@'`` device-param saves exist; a
                LATER interactive ``op`` (what ``generate_control_block``
                adds in this mode) then lands in a distinct, phantom
                ``op2`` instead of refreshing ``op1`` — measured directly
                (``tran_plus_op_fixed.sp``/``.raw``).
                Dropping the netlist card here makes the interactive
                ``op`` the sole producer of the operating-point plot, so
                there is only ever one, whether or not the test itself
                declared ``Op()`` (alone or with another analysis) —
                keeping ``select_plot(raw, 'op')`` unambiguous by
                construction. Xyce has no interactive post-run mechanism
                at all: its ``Op()`` rendering IS the deck's
                one-point ``.dc`` analysis card, the same thing that makes
                the run happen in the first place — skipping it there
                leaves the deck with "No analysis specified" (measured
                directly), so this skip is ngspice-only (``simulator ==
                'ngspice'``), never applied to Xyce.

        Returns:
            String with analysis commands (.tran, .ac, .dc, etc.)
        """
        from silk.simulation.backend import SimulatorBackend, EmitContext
        backend = SimulatorBackend.get(simulator)
        str_analysis = ''
        # Build a logical-name -> instance-name resolver from the sources.
        (_src_names, _targets, _emitted_targets,
         _by_lower, _emitted_lower) = _source_maps(sig_sources_dict)

        def _resolve_noise(n):
            # Exact designer name first (the historic Noise mapping), then
            # an emitted name passed through; the case-insensitive designer
            # match applies only when neither matched.
            if n in _src_names:
                return _targets[n]
            if str(n).lower() in _emitted_lower:
                target = _emitted_targets[str(n).lower()]
                return n if target.lower() == str(n).lower() else target
            return _by_lower.get(str(n).lower(), n)

        def _warn_collision(v):
            # The name is one source's emitted name AND a different
            # source's designer name (sources 'g' -> 'Vg' and 'vg' ->
            # 'Vvg': 'vg' is both). Each analysis keeps its historic
            # precedence: DC takes the emitted match, Noise the exact
            # designer match.
            low = str(v.source).lower()
            as_emitted = [n for n, e in _src_names.items()
                          if str(e).lower() == low]
            if isinstance(v, Noise):
                as_designer = [n for n in _src_names
                               if n == v.source and n not in as_emitted]
            else:
                as_designer = [n for n in _src_names
                               if str(n).lower() == low
                               and n not in as_emitted]
            if not (as_emitted and as_designer):
                return
            e, d = as_emitted[0], as_designer[0]
            if isinstance(v, Noise):
                chosen, other = d, e
                why = 'designer names win for Noise'
                # An exact designer match wins, so the other source is
                # selected by its designer name.
                pick = f"its designer name {e!r}"
            else:
                chosen, other = e, d
                why = 'emitted names win for DC'
                pick = f"its emitted spelling {_src_names[d]!r}"
            warnings.warn(
                f"{type(v).__name__}(source={v.source!r}) is ambiguous: "
                f"it is the emitted name of source {e!r} and the "
                f"designer name of source {d!r}. Resolved to source "
                f"{chosen!r} ({why}); to select source {other!r} "
                f"instead, use {pick}.",
                UserWarning, stacklevel=2)

        for v in sim_analysis_dict.values():
            if isinstance(v, (DC, Noise)) and sig_sources_dict is not None:
                _warn_collision(v)

        if sig_sources_dict is not None:
            for v in sim_analysis_dict.values():
                if not isinstance(v, DC):
                    continue
                name = str(v.source)
                if (name.lower() == 'temp' or name.lower() in _by_lower
                        or name.lower() in _emitted_lower):
                    continue
                listing = ', '.join(
                    f"{n!r} (emitted {e!r})" for n, e in _src_names.items()
                ) or 'none'
                raise ValueError(
                    f"DC(source={v.source!r}): the testbench has no source "
                    f"named {v.source!r}. Sources on this bench: {listing}. "
                    f"(DC('temp', ...) sweeps the circuit temperature.)")

        ctxs = {
            DC: EmitContext(resolve_source=lambda n: _resolve_dc(sig_sources_dict, n),
                            sources=sig_sources_dict or {},
                            emitted_deck=emitted_deck),
            Noise: EmitContext(resolve_source=_resolve_noise,
                               sources=sig_sources_dict or {},
                               emitted_deck=emitted_deck),
        }
        default_ctx = ctxs[DC]
        for v in sim_analysis_dict.values():
            if save_device_params and isinstance(v, Op) and simulator == 'ngspice':
                continue
            # All analyses are AnalysisBase objects (AC/Transient/DC/Noise);
            # the backend renders each to its dialect. The legacy string/dict
            # form and its hardcoded ngspice rendering were removed in C3.
            str_analysis += f'*** {type(v).__name__} analysis\n'
            str_analysis += backend.render_analysis(
                v, ctxs.get(type(v), default_ctx))

        return str_analysis

    def generate_passives(self, passives_dict):
        """Generate passive element lines (R, C, L)

        Args:
            passives_dict: Dictionary mapping passive name → PassiveBase instance

        Returns:
            String with all passive element definitions
        """
        if not passives_dict:
            return ''
        out = '\n*** Passive elements\n'
        for passive in passives_dict.values():
            out += passive.to_spice()
        return out

    def _osdi_lines(self):
        """``pre_osdi`` lines for a PDK whose models are compiled Verilog-A.

        A PDK manifest may declare ``library.osdi:`` — the ``.osdi`` plugins
        its device models need (IHP SG13G2's MOSFETs are PSP Verilog-A; without
        the plugins ngspice's parse fails with "unknown model"). ``pre_osdi``
        is the in-deck command that loads a plugin BEFORE the circuit is
        parsed — verified on ngspice 42 and 47 — so the whole mechanism is
        ordinary deck emission: no ``.spiceinit``, no global state.

        silk does not COMPILE the models — that is environment setup, owned
        by the user or pdksetup, exactly like installing ngspice. And like
        ngspice, emission does not require it: a deck written here is valid on
        any machine whose plugins ARE compiled, so generating a netlist never
        checks the files exist. The existence check lives at SIMULATE time
        (:func:`silk.circuit.modules_sim._require_osdi_compiled`) — the run is
        what cannot work, not the deck.
        """
        # getattr on self too: several tests build a bare netlister via
        # __new__ with no pdk_details at all, and a control block for no PDK
        # is legitimately OSDI-free.
        # A deck that instantiates no PDK device references no PDK model, so
        # there is nothing for the plugins to serve — and loading them would
        # reintroduce the very file dependency the omitted `.lib` just removed
        # (`pre_osdi` on a missing path is an ngspice error). Gated on the same
        # answer as the include block, so the two can never disagree.
        if not getattr(self, '_needs_pdk_library', True):
            return ''
        osdi = getattr(getattr(self, 'pdk_details', None),
                       'osdi_files', None) or []
        from silk.utils.deck_text import card_text
        return ''.join(
            f"pre_osdi {card_text(p, who='PDK osdi plugin path')}\n"
            for p in osdi)

    def generate_control_block(self, dir_folder, noise=False, save_list=None,
                               run_files=None, plots=None,
                               save_device_params=False, op_save_lines=None,
                               probe_save_lines=None, draw_lines=None):
        """Generate ngspice control block for simulation execution

        Args:
            dir_folder: Directory where output files should be written
            noise: When True, inserts ``setplot noise1`` immediately before the
                ``write`` line so the spectral-density plot (not the
                integrated-totals plot that ngspice leaves current) is captured
                in the rawfile.
            save_list: Resolved node-voltage paths (``Module.set_save_list``,
                already expanded/validated) — emitted as explicit ``save``
                commands instead of ``save all``, which restricts the rawfile
                (verified on ngspice-42: an explicit save list in the
                ``.control`` block limits ``write`` to exactly those vectors
                plus the scale). ``None`` keeps the default ``save all``.
            plots: nutmeg plot names to write, in deck order — the
                :data:`NGSPICE_PLOT_NAMES` entry for each declared analysis. More than one qualifies the ``write`` line as
                ``write <path> tran1.all ac1.all``, which is what puts EVERY
                declared analysis in the rawfile; a bare ``write`` saves the
                *current* plot only, i.e. whichever analysis ngspice happened
                to run last, and silently drops the rest. ``None`` or a single
                name keeps the bare ``write <path>`` — so every single-
                analysis deck silk has ever emitted is byte-identical.
            save_device_params: (``Module.save_device_params``).
                When True, appends ``op_save_lines`` (the ``'@'`` device-
                parameter vector names) and an interactive ``op`` AFTER
                ``run`` — never before: a ``.control`` ``save`` applies to
                EVERY analysis ``run`` executes, so an earlier ``'@'`` save
                would contaminate the main analysis' own plot with per-point
                device vectors. No
                netlist-level ``.op`` card ever coexists with this mode
                (``generate_analysis_commands`` / ``_ngspice_plot_names``
                both suppress it), so this interactive ``op`` is the SOLE
                producer of ``op1`` — one op plot, by construction, in every
                case, including an ``Op()``-only deck (measured directly,
                ``op_only_via_sequence.sp``). ``op1`` is always added to the
                ``write`` plot list in this mode, even for an otherwise
                single-analysis deck (which would normally keep the bare
                ``write <path>`` form) — the rawfile now genuinely holds two
                plots and both must be named.
            op_save_lines: The ``'@m....[param]'`` vector names to ``save``
                when ``save_device_params`` is True
                (:func:`~silk.circuit.modules_results.device_op_save_lines`);
                ignored otherwise.
            probe_save_lines: (``tb.probe('<name>:<param>')``).
                ``'@m....[param]'`` vector names to ``save`` BEFORE ``run`` —
                unlike ``op_save_lines`` (post-run, a dedicated ``op1``
                snapshot), these join whatever the deck already saves so
                they land in the MAIN analysis's own plot at every point.
                An additional ``save`` line, never a replacement: ``save all``/an explicit ``save_list`` still
                names everything it always did, and multiple ``save``
                commands accumulate (verified on ngspice-42) — this list is
                always additive, on top of whichever save mode is active.
            draw_lines: The ``showmod`` lines of a Monte Carlo
                sample with ``record_draws=True``, emitted after the
                analysis has run and its rawfile is written, so they never
                change what the rawfile holds; their output goes to the log.

        Returns:
            String with .control block
        """
        files = run_files or RunFiles.resolve('', dir_folder)
        str_control = ''
        if save_list is not None:
            # Provenance in the artifact itself: a deck (and the rawfile made
            # from it) that saves a subset must not read as a complete design.
            str_control += (f'*** save list active: rawfile holds '
                            f'{len(save_list)} node voltages (partial)\n')
        str_control += '.control\n'
        str_control += self._osdi_lines()
        if save_list is None:
            str_control += 'save all\n'   # include subckt-internal nodes in rawfile
        else:
            # Multiple `save` commands accumulate (verified on ngspice-42);
            # chunked so no single line grows unboundedly with design size.
            for i in range(0, len(save_list), 8):
                chunk = save_list[i:i + 8]
                str_control += 'save ' + ' '.join(_quote_save_argument(f'v({p})') for p in chunk) + '\n'
        for i in range(0, len(probe_save_lines or []), 8):
            chunk = (probe_save_lines or [])[i:i + 8]
            str_control += 'save ' + ' '.join(_quote_save_argument(v) for v in chunk) + '\n'
        str_control += 'run\n'
        all_plots = list(plots) if plots else []
        if save_device_params:
            for i in range(0, len(op_save_lines or []), 8):
                chunk = (op_save_lines or [])[i:i + 8]
                str_control += 'save ' + ' '.join(_quote_save_argument(v) for v in chunk) + '\n'
            str_control += 'op\n'
            all_plots.append('op1')
        str_control += 'set wr_singlescale\n'
        str_control += 'set wr_vecnames\n'
        str_control += 'option numdgt=3\n'
        if noise and len(all_plots) <= 1:
            # ngspice keeps the integrated-noise plot current after a .noise
            # run; select the spectral-density plot ('noise1') before writing
            # so the rawfile holds onoise_spectrum/inoise_spectrum vs frequency.
            # Unnecessary when the write names its plots (noise1 is one of
            # them, possibly alongside op1), so a multi-plot deck does not
            # carry a redundant setplot whose only effect would be to reorder
            # nothing.
            str_control += 'setplot noise1\n'
        from silk.utils.deck_text import card_text
        str_control += 'write ' + card_text(
            files.rawfile_path, who='output rawfile path')
        if len(all_plots) > 1:
            str_control += ''.join(f' {name}.all' for name in all_plots)
        str_control += '\n'
        for line in draw_lines or ():
            str_control += line + '\n'
        str_control += 'quit 0\n'
        str_control += '.endc\n'
        str_control += '.end\n'
        return str_control

    def generate_output_block(self, dir_folder, sim_analysis_dict, output_vars=None,
                              save_device_currents=False, save_list=None,
                              run_files=None, save_device_params=False,
                              elem_dict=None, device_probe_requests=None,
                              corner='tt', current_probe_names=None,
                              source_instance_names=None):
        """Emit the simulator-specific wrapper tail (run/output directives).

        This is the one seam where SPICE-family backends diverge: ngspice uses
        a nutmeg ``.control`` block, Xyce uses ``.print ... format=raw``. The
        default (ngspice) keeps the control block and ignores ``output_vars``;
        :class:`~silk.circuit.xyce_netlister.XyceNetlister` overrides it.

        When ``save_device_currents`` is set, ngspice switches to **batch-write
        mode**: a deck-level ``.save all`` and **no
        ``.control`` block**, so the rawfile is written by the runner's existing
        ``ngspice -b -r`` invocation — which tolerates the zero-length
        ``@m…[ib]`` vectors that ``savecurrents`` registers and that the
        ``.control`` ``write`` aborts on. Requires exactly one Transient
        analysis (the noise flow needs the control block's ``setplot``; multi-
        plot batch rawfiles are undefined).

        Args:
            dir_folder: Output directory for the rawfile.
            sim_analysis_dict: Analyses (used to detect noise / pick the
                Xyce ``.print`` analysis keyword).
            output_vars: Net names to capture. Ignored by ngspice (``write``
                already saves all vectors); required by Xyce, which otherwise
                prints only the x-axis.
            save_device_currents: Batch-write mode (see above).
            save_list: Resolved node-voltage save list (``Module.set_save_list``),
                or None for the default ``save all``. Contradicts
                ``save_device_currents`` (batch-write mode is built on
                ``.save all``) — that combination raises. A deck with a Noise
                analysis ignores the list (the noise output path writes its
                own spectral vectors, and noise sweeps are small anyway).
            save_device_params: (``Module.save_device_params``).
                Structurally incompatible with ``save_device_currents`` (any
                ``terminal_current_method``) — that combination raises, in
                the same shape as the ``save_list`` contradiction above.
                Reason (confirmed against ``generate_output_block``'s own
                docstring): ``save_device_currents`` unconditionally forces
                batch-write mode, which emits NO ``.control`` block at all —
                there is no place left for the interactive ``op`` this mode
                needs, and batch mode is refused for anything but exactly one
                Transient besides.
            elem_dict: Device instances, needed only to build the ``'@'``
                save lines when ``save_device_params`` is True (ignored
                otherwise). ``None`` is only safe when the switch is off.
            device_probe_requests: ``[(designer_name, param), ...]``
                — every device-parameter probe currently registered on this
                bench (``ModuleTestbench._pending_device_probes``).
                Resolved and validated HERE, before anything is
                emitted, so an unaddressable device is refused before the
                run rather than read back as a confusing missing vector.
        """
        self._device_param_requests = ()
        probe_save_lines = self._resolve_device_probe_lines(
            elem_dict, device_probe_requests, sim_analysis_dict,
            simulator='ngspice', corner=corner)
        device_probe_lines = list(probe_save_lines)
        # Current-source vectors are device parameters, not voltage branches.
        from silk.sources._layout import physical_source_card
        for name in dict.fromkeys(current_probe_names or ()):
            name = physical_source_card(name, source_instance_names or ()) or name
            name = name.replace(':', '.')
            leaf = name.rsplit('.', 1)[-1]
            if save_list is None and not leaf.lower().startswith('i'):
                continue  # Default save-all already includes voltage branches.
            device_name = 'i.' + name if '.' in name else name
            vector = f'@{device_name}[current]' if leaf.lower().startswith('i') else f'i({name})'
            if vector not in probe_save_lines:
                probe_save_lines.append(vector)
        if save_list is not None and save_device_currents:
            raise SimulationError(
                'save_device_currents and set_save_list contradict each '
                'other: batch-write mode (spec 2026-07-03 §4.1) emits a '
                'deck-level `.save all`, which a save list is meant to '
                'replace. Clear one of them (set_save_list(None), or '
                'save_device_currents = False).'
            )
        if save_device_params and save_device_currents:
            raise SimulationError(
                'save_device_params and save_device_currents contradict '
                'each other: save_device_currents unconditionally forces '
                'batch-write mode (spec 2026-07-03 §4.1) — a deck-level '
                '`.save all` and NO `.control` block — and save_device_params '
                'needs the `.control` block\'s interactive `op` after `run` '
                'to capture device operating-point parameters. Clear one of '
                'them (save_device_params = False, or save_device_currents = '
                'False).'
            )
        sample = getattr(self, '_mc_sample', None)
        record_draws = sample is not None and sample.record_draws
        if record_draws and save_device_currents:
            from silk.exceptions import MonteCarloError
            raise MonteCarloError(
                'record_draws=True and save_device_currents contradict each '
                'other: save_device_currents forces batch-write mode, which '
                'emits no `.control` block, and the drawn parameters are '
                'read by `showmod` lines in that block. Clear one of them '
                '(record_draws=False, or save_device_currents = False).'
            )
        if save_device_currents:
            if getattr(self.pdk_details, 'osdi_files', None):
                raise SimulationError(
                    'save_device_currents is not supported on a PDK with OSDI '
                    'models: batch-write mode emits no .control block, which '
                    'is where the pre_osdi load lines live. Run without '
                    'save_device_currents.'
                )
            analyses = list(sim_analysis_dict.values())
            if len(analyses) != 1 or not isinstance(analyses[0], Transient):
                raise SimulationError(
                    "save_device_currents requires exactly one Transient "
                    f"analysis; got {[type(a).__name__ for a in analyses]}. "
                    "Current annotation reads transient device currents in "
                    "batch-write mode (spec §4.1) — AC/DC/Noise and multi-"
                    "analysis decks are unsupported."
                )
            # Batch-write mode: deck-level `.save all` (keeps subckt-internal
            # node voltages in the rawfile), no `.control` block. Device-
            # parameter probe lines join the SAME deck-level directive
            # (MISSING 1 — must not narrow what `.save all` already keeps).
            line = '.save all'
            if probe_save_lines:
                from silk.utils.deck_text import card_text
                line += ' ' + ' '.join(
                    card_text(v, who='saved vector name')
                    for v in probe_save_lines)
            from silk.simulation._device_requests import snapshot_requests
            self._device_param_requests = snapshot_requests(
                elem_dict or {}, (), probe_save_lines, sim_analysis_dict)
            return line + '\n.end\n'

        _has_noise = any(isinstance(v, Noise) for v in sim_analysis_dict.values())
        if _has_noise and save_list is not None:
            # Explicit no-op, not an error: the noise rawfile holds the
            # spectral-density vectors the analysis itself creates, and a
            # frequency sweep is small — the rawfile-size motivation for a
            # save list does not apply.
            import sys
            print('note: set_save_list is ignored for a deck with a Noise '
                  'analysis — the noise output path saves its own vectors.',
                  file=sys.stderr)
            save_list = None
        op_save_lines = None
        if save_device_params:
            from silk.circuit.modules_results import device_op_save_lines
            op_save_lines = device_op_save_lines(elem_dict or {})
            if not op_save_lines:
                # Ruling (5), TASK-330.3 gate finding: no device in this
                # design has a PDK-declared op-param set (or there is no MOS
                # device at all) -- warn once, naming the PDK, and let the
                # main analysis run exactly as it would with the switch off.
                # Never raise: an undeclared device must not cost a caller
                # their rawfile.
                import warnings
                pdk_name = getattr(getattr(self, 'pdk_details', None),
                                   'pdk_name', None) or 'this PDK'
                warnings.warn(
                    f"save_device_params is on, but no device in this "
                    f"design has a declared op-point parameter set for "
                    f"{pdk_name!r}; no device operating-point parameters "
                    f"will be captured (the main analysis is unaffected).",
                    stacklevel=2,
                )
        from silk.simulation._device_requests import snapshot_requests
        self._device_param_requests = snapshot_requests(
            elem_dict or {}, op_save_lines, device_probe_lines, sim_analysis_dict)
        draw_lines = None
        if record_draws:
            from silk.circuit.monte_carlo import draw_devices, showmod_requests
            devices, _ = draw_devices(elem_dict or {},
                                      self.pdk_details.statistical)
            draw_lines, _ = showmod_requests(devices)
        return self.generate_control_block(dir_folder, noise=_has_noise,
                                           save_list=save_list,
                                           run_files=run_files,
                                           plots=_ngspice_plot_names(
                                               sim_analysis_dict,
                                               save_device_params=save_device_params),
                                           save_device_params=save_device_params,
                                           op_save_lines=op_save_lines,
                                           probe_save_lines=probe_save_lines,
                                           draw_lines=draw_lines)

    def _resolve_device_probe_lines(self, elem_dict, device_probe_requests,
                                    sim_analysis_dict, *, simulator, corner='tt'):
        """validate + address every requested device-parameter
        probe, returning the lines/terms to emit — shared by
        :class:`PySpiceNetlister` (ngspice ``'@'`` saves) and
        :class:`~silk.circuit.xyce_netlister.XyceNetlister` (``N()``/``ID()``
        print terms), which passes ``simulator='xyce'``.

        On ngspice, a param OUTSIDE the device's own declared ``op_params``
        (the set ``save_device_params``/``op_table()`` already trust) is
        checked by a PREFLIGHT before it is ever added to the main save
        list. A
        guarded re-run cannot catch ngspice silently
        returning a populated zero for such a param with NO warning at
        all, which real repros on this exact deck shape showed happens
        for 10-or-more concurrently-saved ``'@'`` vectors. See
        :meth:`_ngspice_param_preflight` for the mechanism. An unavailable
        param's save line is dropped from the main run entirely — so the
        main run never aborts and never returns a filler value for it —
        and the dropped (designer_name, param) pairs are recorded on
        ``self._preflight_unavailable_params`` for
        :meth:`~silk.circuit.modules_results.Module._load_device_probe`
        to raise on at read time.

        Refuses before the run when every declared analysis is Noise: a
        probe's pre-run save never reaches Noise's own separate spectral-
        vector output path, so a request that would otherwise be silently
        dropped is refused instead (never a silent miss).

        Also refuses ammeter-wrapper (``save_device_currents``) mode
        entirely — discovered empirically during this build: ngspice's
        batch-write launch (``-b -r``, no ``.control`` block) never runs
        the ``checkvalid`` gate an interactive ``write`` does, so a
        missing/unsupported ``'@'`` name comes back as a REAL, populated,
        all-zero vector instead of zero-length — indistinguishable from a
        genuine reading of 0. A missing parameter must never be reported as a silent 0;
        since silk cannot tell the two apart from the rawfile alone here,
        the only honest answer is to refuse the combination, the same
        posture ``save_device_params``+``save_device_currents`` already
        takes for a different reason.
        """
        if not device_probe_requests:
            return []
        from silk.circuit.modules_results import (
            dut_module_from_elem_dict, resolve_probe_device_lines)
        if (getattr(self, '_terminal_current_method', 'savecurrents')
                == 'ammeter_wrapper'):
            raise SimulationError(
                "device-parameter probes are not supported together with "
                "terminal_current_method='ammeter_wrapper' "
                "(save_device_currents=True): ngspice's batch-write launch "
                "used in this mode never runs the checkvalid gate an "
                "interactive write does, so a missing/unsupported "
                "device-parameter vector would come back as a real, "
                "populated, all-zero reading instead of raising — silk "
                "never returns a silent 0 for a device-parameter probe. "
                "Run the device-parameter probe in a separate test without "
                "save_device_currents, or read terminal currents via the "
                "':' probe spelling in this test instead."
            )
        analyses = list((sim_analysis_dict or {}).values())
        if analyses and all(isinstance(a, Noise) for a in analyses):
            raise SimulationError(
                "device-parameter probes are not supported in a Noise-only "
                "test: Noise's own output path writes spectral vectors "
                "directly and never reaches the pre-run device-parameter "
                "save/print list."
            )
        dut_module = dut_module_from_elem_dict(elem_dict)
        if simulator == 'ngspice':
            device_probe_requests = self._ngspice_preflight_filter(
                dut_module, device_probe_requests, corner)
        # ammeter_wrapped is always False by construction here: the refusal
        # above already raises for every deck whose
        # terminal_current_method is 'ammeter_wrapper' (the only mode that
        # nests MOS devices under AMMETER_WRAPPER_INTERNAL_NAME), before
        # this line is reached. Passed explicitly, not omitted, so this
        # call site keeps the same shape as Module._load_device_probe's
        # (read-time) call, which DOES need the real value -- read time has
        # no such refusal, because a probe registered before the
        # ammeter-wrapper switch was ever turned on must still be readable
        # by its ORIGINAL (non-nested) address once the switch is off again.
        return resolve_probe_device_lines(dut_module, device_probe_requests,
                                          simulator=simulator,
                                          ammeter_wrapped=False)

    def _ngspice_preflight_filter(self, dut_module, device_probe_requests, corner):
        """Partition *device_probe_requests* into declared (trusted, no
        check needed) and undeclared (checked by a preflight). Returns only the AVAILABLE requests, so the
        caller's save list never names a param this ngspice binary does
        not actually provide; the unavailable ones are recorded on
        ``self._preflight_unavailable_params`` (cleared and rebuilt on
        every call, so a stale entry from an earlier deck never lingers)
        for read time to raise on directly, without ever touching the
        main run's rawfile for that param."""
        from silk.circuit.modules_results import (
            _device_model_name, _device_op_params, resolve_device_probe_address)
        self._preflight_unavailable_params = {}
        available = []
        for designer_name, param in device_probe_requests:
            address = resolve_device_probe_address(
                dut_module, designer_name, param, ammeter_wrapped=False)
            declared = _device_op_params(address.device) or ()
            if param in declared:
                available.append((designer_name, param))
                continue
            model = address.model or _device_model_name(address.device)
            result = self._ngspice_param_preflight(corner, model, param,
                                                    address.polarity,
                                                    address.internal)
            if result.available:
                available.append((designer_name, param))
            else:
                self._preflight_unavailable_params[(designer_name, param)] = result
        return available

    def _ngspice_param_preflight(self, corner, model, param, polarity, internal):
        """Run (or reuse a cached) minimal single-device ngspice check for
        whether *param* is available on THIS binary/PDK/model: decided by what the binary actually does,
        never by a hardcoded version table. One instance of *model* at a
        trivial bias, ``op``, then the SOLE ``'@'`` vector for *param*
        read back through the post-run interactive ``print`` (not a
        ``save``+``write`` — this never touches a rawfile at all, and a
        real repro on this exact deck shape confirmed ngspice's
        ``checkvalid`` warning prints RELIABLY for a vector queried alone
        this way, unlike the main run's save list, which can silently
        return a populated zero for it depending on what else is saved).

        Cached in-process by (resolved binary path, this binary's own
        version string, model, param) — one tiny extra ngspice launch per
        distinct combination per session, never per probe, never per run,
        and never persisted to disk.
        """
        from silk.simulation.runner import ngspice_executable
        binary = ngspice_executable()
        cache = PySpiceNetlister._PREFLIGHT_CACHE
        # Version-less first lookup key: most processes only ever resolve
        # one version per binary path, and re-running the preflight to
        # LEARN the version before it can even be used as a cache key
        # would defeat the cache. Keyed on (binary, model, param) first;
        # the result itself carries the version this binary reported.
        pre_key = (binary, model, param)
        if pre_key in cache:
            return cache[pre_key]
        result = self._ngspice_param_preflight_uncached(binary, corner, model,
                                                         param, polarity,
                                                         internal)
        cache[pre_key] = result
        return result

    #: Process-lifetime cache —
    #: no on-disk persistence. Shared across every netlister instance
    #: (class attribute, not per-instance): the answer depends only on
    #: the resolved binary and the model/param, never on which bench
    #: asked.
    _PREFLIGHT_CACHE = {}

    def _ngspice_param_preflight_uncached(self, binary, corner, model, param,
                                          polarity, internal):
        import shutil
        import subprocess
        import tempfile
        from silk.circuit.modules_results import PreflightResult
        from silk.simulation.conditions import read_ngspice_version
        from silk.simulation.executor import linux_preexec_fn
        bias = -1.0 if polarity == 'p' else 1.0
        # Catalog and probe tokens written into this auxiliary deck, checked
        # before any folder or file exists and before ngspice runs (TASK-593).
        from silk.utils.deck_text import card_text
        model = card_text(model, who='preflight device model')
        param = card_text(param, who='preflight device parameter')
        internal = card_text(internal, who='preflight internal device address')
        deck = ('*** TASK-330.8 preflight: is @...[%s] available for %s '
               'on this ngspice binary?\n' % (param, model))
        deck += self.generate_pdk_include(corner)
        deck += self._osdi_lines()
        deck += '\n'
        deck += f'XM1 d g 0 0 {model} m=1 w=1 l=0.15\n'
        deck += f'Vd d 0 DC {bias}\n'
        deck += f'Vg g 0 DC {bias}\n'
        deck += '\n.control\nop\n'
        deck += f'print @m.xm1.{internal}[{param}]\n'
        deck += 'quit 0\n.endc\n.end\n'
        tmpdir = tempfile.mkdtemp(prefix='silk_preflight_')
        deck_path = os.path.join(tmpdir, 'preflight.cir')
        log_path = os.path.join(tmpdir, 'preflight.log')
        with open(deck_path, 'w') as fh:
            fh.write(deck)
        try:
            proc = subprocess.run(
                [binary, '-b', '-o', log_path, deck_path],
                capture_output=True, timeout=60,
                preexec_fn=linux_preexec_fn(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return PreflightResult(available=False, version=None,
                                   log_path=log_path,
                                   reason=f'preflight could not be launched '
                                          f'({exc}); see {log_path}')
        try:
            with open(log_path) as fh:
                log_text = fh.read()
        except OSError:
            log_text = ''
        version = read_ngspice_version(log_path)
        vector_tail = f'[{param}]'
        available = False
        for line in log_text.splitlines():
            stripped = line.strip()
            if (stripped.startswith(f'@m.xm1.{internal}{vector_tail}')
                    and '=' in stripped):
                available = True
                break
        if proc.returncode != 0 and not available:
            return PreflightResult(
                available=False, version=version, log_path=log_path,
                reason=f'preflight exited {proc.returncode}; see {log_path}')
        if available:
            # Nothing names the log on success: remove the directory.
            shutil.rmtree(tmpdir, ignore_errors=True)
            return PreflightResult(available=True, version=version,
                                   log_path=None, reason='')
        return PreflightResult(available=False, version=version,
                               log_path=log_path,
                               reason=(f'ngspice does not provide {param!r} '
                                       f'for {model!r} (checked before the '
                                       f'run; preflight log: {log_path})'))

    def generate_complete_netlist(self, circuit_name, corner, module_dict, elem_dict,
                                  sig_sources_dict, sig_source_details, sim_analysis_dict,
                                  custom_lines, dir_folder, passives_dict=None,
                                  simulator='ngspice', simulator_options=None,
                                  output_vars=None, save_device_currents=False,
                                  save_list=None,
                                  terminal_current_method='savecurrents',
                                  run_files=None, save_device_params=False,
                                  device_probe_requests=None,
                                  current_probe_names=None, mc_sample=None,
                                  current_probe_analyses=None):
        """Generate complete ngspice netlist

        Args:
            circuit_name: Name of the circuit
            corner: Process corner
            module_dict: Module definitions
            elem_dict: Device instances
            sig_sources_dict: Voltage/current sources
            sig_source_details: Source generation object
            sim_analysis_dict: Simulation commands
            custom_lines: Custom SPICE lines (arbitrary add_custom_line entries)
            dir_folder: Output directory
            passives_dict: Native passive elements (Resistor, Capacitor, Inductor)
            save_device_params: (``Module.save_device_params``).
                Suppresses any ``Op`` entry's own netlist ``.op`` card
                (``generate_analysis_commands``) and threads through to
                ``generate_output_block``, which raises if combined with
                ``save_device_currents`` and otherwise emits the post-run
                ``'@'`` saves + interactive ``op`` + extended ``write``.
            device_probe_requests: Threaded straight to
                ``generate_output_block`` — see its own docstring.
            mc_sample: Monte Carlo ``_SampleSpec`` (``module._mc_sample``);
                ``None`` leaves every deck byte-identical.
            current_probe_names: Emitted source-instance names
                behind every ``tb.probe('i(<source>)')``. Both backends
                acquire the physical operands of logical source currents.
            current_probe_analyses: Analysis selection for each pending
                current probe; omitted selections must also be safe for AC.

        Returns:
            Complete netlist as string
        """
        from silk.sources._layout import source_instances
        netlist = ''
        if any(isinstance(a, AC) for a in sim_analysis_dict.values()):
            from silk.sources._layout import current_source_for_request, refuse_ac_current
            for index, request in enumerate(current_probe_names or ()):
                selection = (current_probe_analyses[index]
                             if current_probe_analyses is not None else None)
                if selection is not None and selection.lower() != 'ac':
                    continue
                source = current_source_for_request(sig_sources_dict, request, elem_dict)
                if source is not None:
                    refuse_ac_current(source)
        if mc_sample is not None:
            self._check_mc_sample_supported(corner, mc_sample)
            self._check_mc_blocks(module_dict, mc_sample)

        # Terminal-current method (ammeter wrappers are collected while emitting
        # device instances, then emitted once at top level — deduped by model,
        # so one .subckt amm_<model> serves every instance design-wide; per-
        # instance uniqueness comes from the hierarchical path v.x<path>.vamm_*).
        self._terminal_current_method = (
            terminal_current_method if save_device_currents else 'savecurrents')
        self._ammeter_wrappers = {}      # shared accumulator, deduped by model

        # Emit devices + module bodies FIRST so every ammeter wrapper (top-level
        # and inside module subckts, at any depth) is collected into the shared
        # dict before we assemble the top-level wrapper defs.
        device_str = self.generate_device_instances(elem_dict)
        # Fresh per deck: one .subckt definition per module name, no matter how
        # many parents instantiate it.
        self._emitted_subckts = set()
        self._mc_sample = mc_sample
        module_str = self.generate_module_definitions(
            module_dict, self._terminal_current_method, self._ammeter_wrappers,
            self._emitted_subckts, mc_sample=mc_sample)

        # Header
        netlist += self.generate_header(circuit_name)
        netlist += '\n'

        # PDK library — omitted entirely for a deck that provably instantiates
        # no PDK device (ideal elements, passives and sources only). Such a
        # deck is self-contained SPICE and runs with no PDK installed; the
        # unconditional `.lib` was the only thing that made it need one.
        # `deck_uses_pdk_devices` is deliberately conservative: anything it
        # cannot prove ideal — including any add_custom_line text — keeps the
        # includes, so a PDK deck's emission is untouched, byte for byte.
        self._needs_pdk_library = deck_uses_pdk_devices(
            elem_dict, module_dict, custom_lines)
        if self._needs_pdk_library:
            netlist += self.generate_pdk_include(corner, sample=mc_sample)
        else:
            netlist += self.generate_pdk_omitted_note()
        netlist += '\n'

        # Device terminal-current ammeter wrapper subckts (top-level defs)
        if self._ammeter_wrappers:
            netlist += '*** Device terminal-current ammeter wrappers\n'
            for k in self._ammeter_wrappers:
                netlist += self._ammeter_wrappers[k]
            netlist += '\n'

        netlist += '*** Defining netlist\n\n'

        # Module definitions (subcircuits)
        netlist += module_str
        netlist += '\n'

        # Device instances
        netlist += device_str
        netlist += '\n'

        # Sources
        netlist += self.generate_sources(sig_sources_dict, sig_source_details)
        netlist += '\n'

        # Passive elements
        netlist += self.generate_passives(passives_dict or {})

        # Custom lines
        netlist += self.generate_custom_lines(custom_lines)
        netlist += '\n'

        # Simulator options (tolerances, method, …) — rendered per the
        # active simulator's YAML config.
        if mc_sample is not None and self._MC_SEED_AS_OPTION:
            simulator_options = {**(simulator_options or {}),
                                 'seed': mc_sample.seed}
        if simulator_options:
            from silk.simulators import load_simulator_config
            cfg = load_simulator_config(simulator)
            netlist += cfg.emit_options_line(simulator_options) + '\n'

        # Device-current save directive (opt-in via Module.save_device_currents).
        # Emitted only if the active simulator's YAML defines it; otherwise warn
        # and omit — never a malformed deck.
        if save_device_currents:
            from silk.simulators import load_simulator_config
            cfg = load_simulator_config(simulator)
            directive = cfg.device_current_save_directive
            if directive:
                netlist += directive + '\n'
            else:
                import warnings
                warnings.warn(
                    f"save_device_currents=True but simulator {simulator!r} "
                    f"defines no device_current_save_directive; no current-save "
                    f"directive emitted.",
                    stacklevel=2,
                )

        # Analysis commands
        netlist += self.generate_analysis_commands(
            sim_analysis_dict, sig_sources_dict=sig_sources_dict, simulator=simulator,
            save_device_params=save_device_params, emitted_deck=netlist)
        netlist += '\n'

        # Output/run block — simulator-specific wrapper tail (nutmeg control
        # block for ngspice; .print format=raw for Xyce). ngspice switches to
        # batch-write mode when save_device_currents is set.
        netlist += self.generate_output_block(
            dir_folder, sim_analysis_dict, output_vars, save_list=save_list,
            save_device_currents=save_device_currents, run_files=run_files,
            save_device_params=save_device_params, elem_dict=elem_dict,
            device_probe_requests=device_probe_requests, corner=corner,
            current_probe_names=current_probe_names,
            source_instance_names=[prefix + name
                                   for _, src, prefix in source_instances(sig_sources_dict, elem_dict)
                                   for name in src.physical_instance_names()])

        return netlist


# ----------------------------------------------------------------------
# Fault-aware AnalogElement emission (see silk/faults/application.py).
# ----------------------------------------------------------------------


#: The private attributes the fault markers put on a device; never copied
#: onto a derived instance.
_FAULT_MARKERS = ('_silk_pending_fault', '_silk_fault_realization')


def _derive_faulted_instance(device, instance):
    """A copy of the :class:`AnalogElement` *device* with one realized fault
    instance applied: the optional name, the parameter overrides (already in
    the PDK's own unit, so they leave the caller-supplied set and are not
    scaled again) and the pin rewires. The copy is detached from the module,
    so building it never touches the module's nets, and *device* itself is
    never changed."""
    derived = object.__new__(type(device))
    state = {k: v for k, v in device.__dict__.items()
             if k not in (*_FAULT_MARKERS, '_wiring_holders')}
    derived.__dict__.update(state)
    object.__setattr__(derived, '_module', None)
    generated_name = getattr(instance, 'name', None)
    object.__setattr__(derived, 'name', generated_name if generated_name is not None
                       else device.name)
    object.__setattr__(derived, 'nets', {**device.nets, **instance.nets})
    object.__setattr__(derived, 'params', {**device.params, **instance.params})
    object.__setattr__(derived, '_caller_supplied',
                       set(device._caller_supplied) - set(instance.params))
    return derived


class _PDKDeviceView:
    """A stored ``PDKDevice`` (``add_element(pdk.cls, ...)``) read in the
    shape :func:`silk.faults.application._realize_fault` reads an
    :class:`AnalogElement`: a caller-set constraint ``value`` is in the
    user's units and scales by the factor ``BasePDK.generate_netlist_entry``
    applies to it (:meth:`BasePDK._scale_factor`); an unset one emits its
    catalog ``default``, already in the PDK's own unit."""

    def __init__(self, device, name, pdk):
        self.name = name
        self.nets = {p: pin.net for p, pin in device.pin_list.items()}
        self.params = {p: (c.value if c.value is not None else c.default)
                       for p, c in device.param_list.items()}
        self._caller_supplied = {p for p, c in device.param_list.items()
                                 if c.value is not None}
        self._param_scales = {
            p: pdk._scale_factor(c.pdk_unit, c.quantity)
            for p, c in device.param_list.items()}
        self._device_type_name = getattr(device, 'device_name', '')


def _derive_faulted_pdk_device(device, instance):
    """A copy of the stored ``PDKDevice`` *device* with one realized fault
    instance applied, for ``generate_netlist_entry``. An override is in the
    PDK's own unit, so it is set as the constraint's ``default`` with no
    caller ``value`` — the form ``generate_netlist_entry`` writes unscaled.
    Pins and constraints are copied; *device* itself is never changed."""
    derived = copy.copy(device)
    for marker in _FAULT_MARKERS:
        derived.__dict__.pop(marker, None)
    pins = {}
    for p, pin in device.pin_list.items():
        pins[p] = copy.copy(pin)
        if p in instance.nets:
            pins[p].net = instance.nets[p]
    derived.pin_list = pins
    params = {}
    for p, constraint in device.param_list.items():
        params[p] = copy.copy(constraint)
        if p in instance.params:
            params[p].value = None
            params[p].default = instance.params[p]
    derived.param_list = params
    return derived


def _build_added_element(spec):
    """The silk object for one element a fault adds (see
    :class:`silk.faults.application._AddedElement`). Built with no module
    active, so it is never registered on the user's module."""
    from silk.circuit._module_base import _pop_active, _push_active
    from silk.elements import elements
    from silk.sources.classes import VCVS, VoltageSource

    _push_active(None)
    try:
        if spec.kind == 'resistor':
            return elements.resistor(spec.name, value=spec.value, **spec.nets)
        if spec.kind == 'vcvs':
            return VCVS(spec.name, gain=spec.value, **spec.nets)
        if spec.kind == 'vdc':
            return VoltageSource(spec.name, dc=spec.value, **spec.nets)
    finally:
        _pop_active()
    raise ValueError(f'Unknown added-element kind from a fault: {spec.kind!r}')


def _emit_faulted_device(netlister, device, name) -> str:
    """Render the prepared realization through each element's own writer.

    Preparation resolves the owning emission context and stores the realized
    changes on a temporary device copy. Repeated writes consume neither random
    draws nor manifest entries.
    """
    from silk.circuit.emitted_names import _generated_token_scope

    is_element = isinstance(device, AnalogElement)
    realization = device._silk_fault_realization

    out = ''.join(comment + '\n' for comment in realization.comments)
    for spec in realization.added:
        # Only the shift helper's output terminals are newly generated nets;
        # its control terminals and the short's terminals are circuit nodes.
        nets = ((spec.nets['out_pos'], spec.nets['out_neg'])
                if spec.kind == 'vcvs' else
                (spec.nets['positive'],) if spec.kind == 'vdc' else ())
        with _generated_token_scope(
                instances=(spec.name,), nets=nets):
            added = _build_added_element(spec)
            out += (netlister._emit_analog_element(added)
                    if isinstance(added, AnalogElement) else added.to_spice())
    for instance in realization.instances:
        with _generated_token_scope(
                instances=() if instance.name is None else (instance.name,),
                nets=instance.nets.values()):
            if is_element:
                out += netlister._emit_analog_element(
                    _derive_faulted_instance(device, instance))
            else:
                out += netlister._emit_pdk_device(
                    instance.name if instance.name is not None else name,
                    {'type': ElementType.DEVICE,
                     'device': _derive_faulted_pdk_device(device, instance)})
    return out
