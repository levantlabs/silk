"""XyceNetlister — Xyce-dialect netlist emission.

Xyce shares silk's entire SPICE *body* with ngspice (header, ``.lib`` include,
module/device/passive emission, and the ``.tran/.ac/.dc/.noise``
analysis lines via :class:`~silk.simulation.backend.SpiceBackend`). It diverges
in waveform voltage-source DC defaults and in the wrapper *tail*:

* waveform voltage elements without their own DC clause receive the waveform
  initial value, matching ngspice operating points (including PWL series pairs);
* no nutmeg ``.control`` block — Xyce reads the deck and runs it directly;
* ``.options parser scale=<PDK length unit in metres>`` so a micron-native
  catalog's bare W/L resolve against the metre-valued model bins (without it,
  sky130's ``w=4`` reads as 4 metres). The factor is the PDK's, read from
  :attr:`~silk.pdk.basepdk.BasePDK.length_unit_in_metres` — ``1e-6`` for
  sky130, and no line at all for the metre-native catalogs (GF180MCU,
  SG13G2), whose W/L already ARE metres and must not be rescaled twice;
* ``.print <analysis> format=raw file=.../rawfile.raw v(net) ...`` to write a
  silk-readable rawfile. Xyce prints only the x-axis without an explicit var
  list, so the output nets must be enumerated;
* a second ``.print <analysis> format=csv file=.../xyce_temperature.csv
  {TEMP}`` — the ambient-temperature probe — *only* when this run configures
  TEMP. Xyce writes no log for silk to scrape, so this is how an applied
  temperature is confirmed at runtime rather than assumed. A temperature-free
  deck emits nothing extra and is byte-identical to before.

See ``docs/known_issues.md`` for current Xyce support and limitations.
"""
import os

from silk.simulation.run_files import RunFiles
from silk.circuit.spice_netlister import PySpiceNetlister
from silk.exceptions import SimulationError, NotSupportedError
from silk.simulation.analysis import Transient, AC, DC, Noise, Op


# Nets that are never meaningful .print targets.
_GROUND_NETS = {'0', 'gnd'}


def _fmt_scale(factor):
    """Render a scale factor the way the sky130 deck always spelled it.

    A power of ten is written ``1e-6`` (not ``1e-06``, not ``1u``) so the
    sky130 emission stays byte-identical to the original hardcoded line;
    anything else falls back to ``repr``.
    """
    mantissa, exponent = f'{factor:.12e}'.split('e')
    if float(mantissa) == 1.0:
        return f'1e{int(exponent)}'
    return repr(factor)


class XyceNetlister(PySpiceNetlister):
    """:class:`PySpiceNetlister` specialized for the Xyce dialect."""

    _MC_SEED_AS_OPTION = False
    _source_backend = 'xyce'

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
        from silk.circuit.xyce_ground import check_inventory, check_emitted_nodes
        from silk.circuit.module_placement import _fault_definition_paths
        paths = check_inventory(output_vars, elem_dict, module_dict)
        paths.update(_fault_definition_paths())
        deck = super().generate_complete_netlist(
            circuit_name, corner, module_dict, elem_dict, sig_sources_dict,
            sig_source_details, sim_analysis_dict, custom_lines, dir_folder,
            passives_dict=passives_dict, simulator=simulator,
            simulator_options=simulator_options, output_vars=output_vars,
            save_device_currents=save_device_currents, save_list=save_list,
            terminal_current_method=terminal_current_method, run_files=run_files,
            save_device_params=save_device_params,
            device_probe_requests=device_probe_requests,
            current_probe_names=current_probe_names, mc_sample=mc_sample,
            current_probe_analyses=current_probe_analyses)
        check_emitted_nodes(deck, paths)
        return deck

    def _check_mc_sample_supported(self, corner, mc_sample):
        # Xyce draws only under .SAMPLING useExpr=true (without it every draw
        # is the mean, TASK-300); generate_output_block emits that, and the
        # seed travels as -randseed (XyceBackend.sample_args).
        super()._check_mc_sample_supported(corner, mc_sample)

    def _check_mc_blocks(self, module_dict, mc_sample):
        # set_mismatch(False) is a subcircuit-local .param; Xyce ignores that
        # scope (verified on 7.10.0), so the block would still draw mismatch.
        if (mc_sample.variation not in ('mismatch', 'both')
                or getattr(self.pdk_details, 'statistical', None) is None):
            return
        from silk.circuit.ground_port import block_name
        from silk.exceptions import NotSupportedError
        seen = set()

        def walk(modules):
            for entry in (modules or {}).values():
                block = entry.get('device')
                if block is None or id(block) in seen:
                    continue
                seen.add(id(block))
                if not block.mismatch_enabled:
                    raise NotSupportedError(
                        f'set_mismatch(False) on block {block_name(block)!r} '
                        'is not supported on Xyce: Xyce ignores a '
                        'subcircuit-local .param, so the block would still '
                        "draw mismatch. Use simulator='ngspice', or leave "
                        'mismatch on for every block.')
                walk(getattr(block, 'module_dict', None))

        walk(module_dict)

    def generate_pdk_include(self, corner='tt', sample=None):
        # Xyce has no OSDI loader — the compiled-Verilog-A mechanism is
        # ngspice's. A PDK that declares `library.osdi:` cannot run here at
        # all, so refuse at netlist time with the reason instead of letting
        # Xyce fail on every model card.
        #
        # The refusal is scoped to decks that actually reference PDK models by
        # virtue of WHERE it lives: `generate_complete_netlist` calls this
        # method only when `deck_uses_pdk_devices` says the include block is
        # needed. An ideal R/C/V bench under a loaded SG13G2 therefore emits
        # and runs on Xyce instead of being refused for models it never names.
        if getattr(self.pdk_details, 'osdi_files', None):
            from silk.exceptions import NotSupportedError
            raise NotSupportedError(
                'This PDK\'s device models are compiled Verilog-A (OSDI), '
                'which Xyce cannot load — OSDI is an ngspice mechanism. '
                "Use simulator='ngspice' (with the plugins compiled: see "
                'docs/installing-the-pdk.md).'
            )
        return super().generate_pdk_include(corner, sample=sample)

    # Prefer a manifest's `xyce_*` fields (xyce_includes per corner,
    # xyce_library_location, xyce_extra_includes) when it declares them —
    # e.g. GF180MCU ships a model set re-authored for Xyce's stricter
    # `.LIB`/`.ENDL` grammar (TASK-339: the ngspice-oriented file relies on a
    # bare `.ENDL` and a mismatched `.ENDL` name that ngspice tolerates and
    # Xyce does not). Falls back to the ngspice-oriented fields otherwise, so
    # a PDK that declares no Xyce-specific models (sky130, and SG13G2 short
    # of its OSDI refusal above) emits exactly what it always has.
    def _corner_includes(self, meta):
        return meta.get('xyce_includes') or meta.get('includes')

    def _pdk_library_location(self):
        return (getattr(self.pdk_details, 'xyce_library_location', None)
                or self.pdk_details.library_location)

    def _pdk_extra_includes(self):
        xyce_extra = getattr(self.pdk_details, 'xyce_extra_includes', None)
        if xyce_extra is not None:
            return xyce_extra
        return getattr(self.pdk_details, 'extra_includes', []) or []

    # Map a silk analysis to the Xyce .print analysis keyword.
    # Op renders as a one-point .dc sweep on Xyce (XyceBackend.render_analysis).
    _PRINT_KEYWORD = {Transient: 'tran', AC: 'ac', DC: 'dc', Noise: 'noise',
                      Op: 'dc'}

    def _print_keyword(self, sim_analysis_dict):
        """Return the Xyce ``.print`` analysis keyword for the (single) analysis.

        ``generate_output_block`` refuses multiple declared analyses: Xyce
        cannot run them all, even with separate ``.print`` lines. See
        ``docs/known_issues.md``.
        """
        for analysis in sim_analysis_dict.values():
            kw = self._PRINT_KEYWORD.get(type(analysis))
            if kw is not None:
                return kw
        return 'tran'

    def _scale_option(self):
        """The ``.options parser scale=`` line for this PDK, or ``''``.

        Xyce reads every W/L in the deck as metres. A catalog whose models take
        another unit needs the deck-wide ``scale`` to say so; the PDK, not the
        netlister, owns that number (architecture.md I11), so it is read from
        ``BasePDK.length_unit_in_metres`` and never interpreted here. A factor
        of 1 (metre-native PDK) or None (no lengths in the catalog, or no PDK
        at all — an R/C/V deck) means Xyce's default is already right, and no
        line is emitted rather than a no-op ``scale=1``.

        A PDK object that does not expose the factor is an error, not a
        micron: guessing 1e-6 is exactly the bug this replaced.
        """
        # getattr on self too: some tests build a bare netlister via __new__
        # with no pdk_details at all (same allowance as generate_osdi_block).
        pdk = getattr(self, 'pdk_details', None)
        if pdk is None:
            return ''
        try:
            factor = pdk.length_unit_in_metres
        except AttributeError:
            raise TypeError(
                f'{type(pdk).__name__} does not expose length_unit_in_metres; '
                f'the Xyce netlister needs the PDK to say what unit its W/L '
                f'are in (see BasePDK.length_unit_in_metres).') from None
        if factor is None or factor == 1.0:
            return ''
        return f'.options parser scale={_fmt_scale(factor)}\n'

    @staticmethod
    def _source_current_terms(current_probe_names, source_instance_names):
        """``I(<SOURCE>)`` print terms for ``probe('i(<source>)')``.

        The source-owned physical inventory includes nested cards. Xyce
        spells hierarchy with colons in I(...) print expressions. Unknown
        cards still refuse before a native undefined-symbol abort.
        """
        names = [n for n in (current_probe_names or ()) if n]
        if not names:
            return []
        known = {str(n).lower(): str(n) for n in (source_instance_names or ())
                 if n}
        from silk.sources._layout import physical_source_card
        terms = []
        for name in names:
            emitted = physical_source_card(name, known.values())
            if emitted is None:
                raise NotSupportedError(
                    f"probe('i({name})') on Xyce: {name!r} is not a "
                    f"source card of this testbench (sources here: "
                    f"{sorted(known.values())}). A device current outside "
                    f"the source inventory is not supported on Xyce by "
                    f"probe('i(...)'). Use simulator='ngspice' for those.")
            path = emitted.replace('.', ':').upper()
            term = f'I({path})'
            if term not in terms:          # one column per source
                terms.append(term)
        return terms

    def generate_output_block(self, dir_folder, sim_analysis_dict, output_vars=None,
                              save_device_currents=False, save_list=None,
                              run_files=None, save_device_params=False,
                              elem_dict=None, device_probe_requests=None,
                              corner='tt', current_probe_names=None,
                              source_instance_names=None):
        # save_device_currents is ngspice-only (batch-write mode); Xyce ignores
        # it here — Step-1 already warned+omitted the save directive.
        analyses = list(sim_analysis_dict.values())
        # TASK-330.8: resolved (and, for Xyce, capability-checked) BEFORE
        # anything below — one bad .print symbol aborts the WHOLE Xyce run,
        # so this must raise here, not read back as a missing vector.
        # corner is unused on this path: the ngspice-only preflight
        # (reviewer ruling 2026-09-28) never runs for simulator='xyce'.
        probe_terms = self._resolve_device_probe_lines(
            elem_dict, device_probe_requests, sim_analysis_dict,
            simulator='xyce', corner=corner)
        current_terms = self._source_current_terms(
            current_probe_names, source_instance_names)
        if save_device_params:
            # TASK-330.5, reviewer ruling (A): owner rule #1 ("device
            # parameters are captured on an operating-point run only, never
            # per timestep/sweep-point") applies on Xyce too. Appending
            # N()/ID() to a '.print tran' or a multi-point '.print dc' would
            # record them at EVERY solve point — forbidden regardless of
            # backend. Op() is the only analysis that renders as a
            # single-point '.dc' (XyceBackend.render_analysis), so it is the
            # only shape this switch accepts here.
            if len(analyses) != 1 or not isinstance(analyses[0], Op):
                requested = (', '.join(type(a).__name__ for a in analyses)
                            or 'none')
                raise NotSupportedError(
                    'save_device_params on Xyce requires Op() as the '
                    "deck's one analysis (TASK-330.5): any other analysis "
                    "(Transient, a swept DC, AC, Noise) would append "
                    "N()/ID() device-parameter columns to a print line "
                    'that records every solve point, not one operating '
                    f'point. Requested: {requested}. Run save_device_params '
                    'in a separate Op()-only test, or set '
                    'Module.save_device_params = False.'
                )
        elif len(analyses) > 1:
            requested = ', '.join(type(a).__name__ for a in analyses)
            raise SimulationError(
                'Xyce supports only one analysis per netlist; '
                f'requested {len(analyses)}: {requested}. '
                'Split the analyses into separate tests or use '
                "simulator='ngspice' to run them together.")
        noise = [a for a in analyses if isinstance(a, Noise)]
        others = [a for a in analyses if not isinstance(a, Noise)]

        files = run_files or RunFiles.resolve(
            '', dir_folder, raw=bool(others), noise=bool(noise))
        # Each .print below writes an absolute output path: check that final
        # string where it is written, not the directory it came from
        # (TASK-593). A supplied RunFiles may carry any path, and abspath
        # adds the working directory to a relative one.
        from silk.utils.deck_text import card_text

        def out_path(p, what):
            return card_text(os.path.abspath(p), who=f'Xyce {what} output path')

        block = self._scale_option()

        # Node analyses (AC/Transient/DC) -> a rawfile print.
        if others:
            keyword = self._print_keyword({i: a for i, a in enumerate(others)})
            # A resolved save list (Module.set_save_list) replaces the default
            # top-level nets plus declared node probes. Paths arrive in
            # silk's dot-separated spelling; Xyce's hierarchy separator is
            # ':' (v(X1:MID), verified on Xyce 7.10). The explicit list can
            # add unprobed internal nodes or exclude declared probes.
            nets = output_vars if save_list is None else save_list
            nets = [n for n in (nets or []) if n.lower() not in _GROUND_NETS]
            terms = [f"v({n.replace('.', ':')})" for n in nets]
            if save_device_params:
                # TASK-330.5: N()/ID() device-parameter columns, ADDITIVE to
                # the node-voltage prints above. Built from a fixed
                # known-good set only (never "ask for everything, drop what
                # errors") — Xyce aborts the WHOLE run on one unresolvable
                # .print symbol (TASK-330.1's spike).
                from silk.circuit.modules_results import xyce_op_print_terms
                op_terms = xyce_op_print_terms(elem_dict or {})
                if not op_terms:
                    import warnings
                    pdk_name = getattr(getattr(self, 'pdk_details', None),
                                       'pdk_name', None) or 'this PDK'
                    warnings.warn(
                        f"save_device_params is on, but no device in this "
                        f"design is declared for the 'xyce' backend on "
                        f"{pdk_name!r}; no device operating-point "
                        f"parameters will be captured (the main analysis "
                        f"is unaffected).",
                        stacklevel=2,
                    )
                terms.extend(op_terms)
            # TASK-330.8: device-parameter probe terms, additive on top of
            # both the node-voltage prints and any save_device_params terms
            # above — unlike save_device_params, NOT restricted to Op(): the
            # AC #1 spike verified N()/ID() work per-point in .dc and .tran
            # on Xyce 7.10.
            terms.extend(probe_terms)
            # TASK-346: branch currents of the sources a probe('i(...)')
            # names. ngspice saves every branch current by default; Xyce
            # prints only what is listed. Present ONLY when such a probe
            # exists, so every probe-free deck is unchanged.
            terms.extend(current_terms)
            # A device-parameter probe naming the SAME (device, param)
            # save_device_params already declared would otherwise print the
            # identical N()/ID() term twice -- Xyce then writes it as two
            # rawfile columns sharing one name, which reads back "ambiguous"
            # (build-time finding: 'M1:gm' probed alongside
            # save_device_params=True on a device whose op_params include
            # 'gm'). De-duped in place, order preserved, so an accidental
            # collision costs nothing rather than breaking the read.
            seen_terms = set()
            terms = [t for t in terms
                    if not (t.upper() in seen_terms or seen_terms.add(t.upper()))]
            var_list = ' '.join(card_text(t, who='Xyce .print term') for t in terms)
            line = f'.print {keyword} format=raw file={out_path(files.rawfile_path, "rawfile")}'
            if var_list:
                line += ' ' + var_list
            block += line + '\n'

        # Noise -> a CSV print (Xyce refuses rawfile/PROBE/Touchstone for noise).
        # ONOISE/INOISE are V²/Hz; the loader sqrt-converts to V/√Hz.
        if noise:
            block += (f'.print noise format=csv file={out_path(files.noise_path, "noise")} '
                      f'ONOISE INOISE\n')

        # Ambient-temperature probe. Present only for a run that configures
        # TEMP (silk.simulation.run_files resolves the path only then), so
        # every existing temperature-free deck is unchanged. `{TEMP}` is Xyce's
        # circuit-temperature expression: this reads back what the DEVICE
        # package actually applied, independent of what silk asked for. Xyce
        # accepts several .print lines for the same analysis when files differ
        # (verified on Xyce 7.10.0); the rawfile keeps its previous vectors.
        if files.temperature_path is not None:
            keyword = 'noise' if noise and not others else self._print_keyword(
                {i: a for i, a in enumerate(others)})
            block += (f'.print {keyword} format=csv '
                      f'file={out_path(files.temperature_path, "temperature")} {{TEMP}}\n')

        if getattr(self, '_mc_sample', None) is not None:
            # One sample per deck; its seed is the -randseed argument.
            block += '.SAMPLING useExpr=true\n'
            block += '.options SAMPLES numsamples=1 sample_type=MC\n'

        block += '.end\n'
        return block
