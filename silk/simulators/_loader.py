"""Loader for simulator-specific YAML configs (ngspice.yaml, xyce.yaml, …).

Kept deliberately thin — the core netlister asks for an ``emit_options_line``
string and a ``resolve_preset`` dict, and doesn't need to know anything else
about the target simulator.
"""

import os
import warnings

import yaml


_CACHE: dict = {}


class SimulatorConfig:
    """Parsed simulator YAML — see ``silk/simulators/ngspice.yaml`` for the
    schema.

    Attributes
    ----------
    name : str
    presets : dict[str, dict]
        Preset name → option dict.
    known_options : frozenset[str]
        Allowlist for option-key warnings.
    """

    def __init__(self, data: dict):
        self.name = data['name']
        self.version_policy = data.get('version_policy', {})
        self._emit = data.get('emit', {})
        # Option keys that belong on a line of their own because the simulator
        # files them under a different options package (see xyce.yaml: TEMP is
        # a DEVICE option, while silk's portable tolerances are TIMEINT ones).
        # Absent -> every option shares the single `line_format` line, which is
        # what every simulator without packages (ngspice) wants.
        self._option_line_formats = {
            str(k).lower(): v
            for k, v in (self._emit.get('option_line_formats') or {}).items()
        }
        self.presets = data.get('presets', {}) or {}
        # False on a backend with no options at all (behavioral.yaml): every
        # key is then refused by reject_unknown() instead of being stored.
        self.accepts_options = bool(data.get('accepts_options', True))
        self.known_options = frozenset(data.get('known_options', []) or [])
        # Literal line that enables per-device current saving, or None if this
        # simulator has no equivalent (spec 2026-07-03 §4).
        self.device_current_save_directive = data.get(
            'device_current_save_directive')
        # How terminal currents are obtained: 'savecurrents' (default) or
        # 'ammeter_wrapper' (see ngspice.yaml). Module.terminal_current_method
        # overrides this per run.
        self.terminal_current_method = data.get(
            'terminal_current_method', 'savecurrents')

    def resolve_preset(self, name: str) -> dict:
        """Return a copy of the options dict for ``name``.

        Raises
        ------
        ValueError
            If ``name`` is not a known preset for this simulator.
        """
        if name not in self.presets:
            raise ValueError(
                f"Unknown preset {name!r} for simulator {self.name!r}. "
                f"Valid: {sorted(self.presets)}."
            )
        return dict(self.presets[name])

    def emit_options_line(self, options: dict) -> str:
        """Render an options dict as netlist line(s), each ending in ``\\n``.

        Normally one line. Keys listed in ``emit.option_line_formats`` are
        routed onto their own line instead, because the simulator files them
        under a different options package — Xyce's TEMP is a DEVICE option and
        is *silently discarded* if it arrives on the TIMEINT line (verified on
        Xyce 7.10.0). Routed keys keep their relative order and are emitted
        after the default line, which keeps carrying every other option.

        Returns empty string when ``options`` is falsy.
        """
        if not options:
            return ''
        kv_format = self._emit.get('kv_format', '{key}={value}')
        separator = self._emit.get('separator', ' ')
        line_format = self._emit.get('line_format', '.options {options}')

        # Group in first-seen order: the default line first, then one line per
        # distinct routed format. dict preserves insertion order, so emission
        # is deterministic for a given options dict.
        _refuse_option_line_breaks(options)
        groups: dict = {}
        for key, value in options.items():
            fmt = self._option_line_formats.get(str(key).lower(), line_format)
            # The text written, after formatting (TASK-593).
            from silk.utils.deck_text import card_text
            groups.setdefault(fmt, []).append(card_text(
                kv_format.format(key=key, value=value),
                who=f'simulator option {key!r}'))
        ordered = ([line_format] if line_format in groups else []) + [
            fmt for fmt in groups if fmt != line_format]
        return ''.join(fmt.format(options=separator.join(groups[fmt])) + '\n'
                       for fmt in ordered)

    def reject_unknown(self, options: dict) -> None:
        """Raise for any option on a simulator that takes none.

        A no-op unless the YAML sets ``accepts_options: false``.

        Raises
        ------
        NotSupportedError
            Naming the refused option keys and the simulator.
        """
        if self.accepts_options or not options:
            return
        from silk.exceptions import NotSupportedError
        keys = ', '.join(repr(k) for k in options)
        raise NotSupportedError(
            f"Unknown option(s) {keys} for simulator {self.name!r}: it takes "
            f"no simulator options. Set simulator to a SPICE backend to use "
            f"them."
        )

    def warn_unknown(self, options: dict) -> None:
        """Warn on option keys outside ``known_options``. Non-fatal."""
        if not self.known_options:
            return
        for k in options:
            if k.lower() not in self.known_options:
                warnings.warn(
                    f"Option {k!r} is not in {self.name}'s known_options list; "
                    f"netlist will pass it through but the simulator may reject it.",
                    UserWarning,
                    stacklevel=3,
                )


def load_simulator_config(name: str) -> SimulatorConfig:
    """Load and cache the YAML config for a simulator by name.

    Looks for ``silk/simulators/<name>.yaml`` relative to this file.
    Unknown simulators fall back to a minimal config with no presets and
    a permissive ``.options`` emit line — keeps silk working even if a
    YAML is missing.
    """
    if name in _CACHE:
        return _CACHE[name]

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, f'{name}.yaml')
    if not os.path.isfile(path):
        warnings.warn(
            f"No simulator config found at {path}. Using minimal defaults; "
            f"presets will be unavailable.",
            UserWarning,
            stacklevel=2,
        )
        data = {'name': name}
    else:
        with open(path) as f:
            data = yaml.safe_load(f) or {}

    cfg = SimulatorConfig(data)
    _CACHE[name] = cfg
    return cfg


def _refuse_option_line_breaks(options: dict) -> None:
    """Refuse an option key or value holding a line break: every
    option is written into an options card, where a break would start a new
    deck line."""
    from silk.utils.deck_text import refuse_line_breaks
    for key, value in options.items():
        refuse_line_breaks(key, who='simulator option name')
        refuse_line_breaks(value, who=f'simulator option {key!r}')
