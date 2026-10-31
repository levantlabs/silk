"""Tests for the simulator-options YAML framework.

Covers:
  * ``silk.simulators.load_simulator_config`` — caching, presets, missing
    YAML fallback, emit-syntax rendering.
  * ``Module.set_simulator_options`` — accumulate semantics, preset
    resolution, raw-override, deprecation alias.
  * End-to-end: options render into the generated netlist.
"""

import glob
import os
import warnings

import pytest

from silk.circuit.modules import Module
from silk.simulators import (
    SimulatorConfig,
    load_simulator_config,
)
from silk.simulators import _loader as _simloader

# Every simulator YAML that ships, discovered rather than listed, so a new one
# is held to the same contract the day it lands.
SIMULATOR_NAMES = sorted(
    os.path.splitext(os.path.basename(p))[0]
    for p in glob.glob(os.path.join(os.path.dirname(_simloader.__file__), '*.yaml'))
)


# ---------------------------------------------------------------------------
# SimulatorConfig / load_simulator_config
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_cache():
    """Every test starts with an empty simulator-config cache so a warning
    on the missing-YAML path fires reliably."""
    _simloader._CACHE.clear()
    yield
    _simloader._CACHE.clear()


class TestLoadSimulatorConfig:

    def test_loads_ngspice_yaml(self):
        cfg = load_simulator_config('ngspice')
        assert cfg.name == 'ngspice'
        assert 'default' in cfg.presets
        assert 'tight' in cfg.presets

    def test_is_cached(self):
        """Second load returns the same object."""
        a = load_simulator_config('ngspice')
        b = load_simulator_config('ngspice')
        assert a is b

    def test_missing_yaml_warns_and_falls_back(self):
        """Unknown simulator name → UserWarning + minimal config."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            cfg = load_simulator_config('definitely_not_a_simulator')
        assert any('No simulator config found' in str(x.message) for x in w)
        assert cfg.name == 'definitely_not_a_simulator'
        assert cfg.presets == {}
        assert cfg.known_options == frozenset()


class TestEverySimulatorYaml:
    """The contract every ``silk/simulators/<name>.yaml`` must satisfy.

    Parametrized over whatever ships, so adding a simulator = adding a YAML
    (the claim ngspice.yaml's header makes) is enforced rather than asserted.
    """

    def test_at_least_the_known_simulators_ship(self):
        assert {'ngspice', 'xyce', 'behavioral'} <= set(SIMULATOR_NAMES)

    @pytest.mark.parametrize('name', SIMULATOR_NAMES)
    def test_name_field_matches_filename(self, name):
        assert load_simulator_config(name).name == name

    @pytest.mark.parametrize('name', SIMULATOR_NAMES)
    def test_carries_the_portable_presets(self, name):
        """``preset='tight'`` has to mean something on every simulator, or the
        preset vocabulary is not portable."""
        cfg = load_simulator_config(name)
        assert {'default', 'tight', 'fast'} <= set(cfg.presets)

    @pytest.mark.parametrize('name', SIMULATOR_NAMES)
    def test_default_preset_is_empty(self, name):
        assert load_simulator_config(name).resolve_preset('default') == {}

    @pytest.mark.parametrize('name', SIMULATOR_NAMES)
    def test_preset_keys_are_in_known_options(self, name):
        """A preset that emits a key its own allowlist rejects would warn on
        every use — the YAML would be arguing with itself."""
        cfg = load_simulator_config(name)
        for preset in cfg.presets:
            for key in cfg.resolve_preset(preset):
                assert key.lower() in cfg.known_options, (
                    f'{name}.yaml preset {preset!r} sets {key!r}, which is not '
                    f'in its known_options')

    @pytest.mark.parametrize('name', SIMULATOR_NAMES)
    def test_emits_an_options_line(self, name):
        cfg = load_simulator_config(name)
        line = cfg.emit_options_line({'reltol': '1e-5'})
        assert line.startswith('.options')
        assert line.endswith('\n')

    @pytest.mark.parametrize('name', SIMULATOR_NAMES)
    def test_has_a_registered_backend(self, name):
        """A simulator YAML with no backend behind it is unselectable."""
        from silk.simulation.backend import SimulatorBackend
        assert SimulatorBackend.get(name) is not None


class TestPresetResolution:

    def test_tight_preset_has_expected_keys(self):
        cfg = load_simulator_config('ngspice')
        d = cfg.resolve_preset('tight')
        # A StrongArm-safe preset must include method and tolerance keys.
        assert 'method' in d
        assert 'reltol' in d
        assert 'trtol' in d

    def test_default_preset_is_empty(self):
        cfg = load_simulator_config('ngspice')
        assert cfg.resolve_preset('default') == {}

    def test_unknown_preset_raises(self):
        cfg = load_simulator_config('ngspice')
        with pytest.raises(ValueError, match='Unknown preset'):
            cfg.resolve_preset('not_a_preset')

    def test_resolve_returns_copy(self):
        """Mutating the returned dict must not affect the cached config."""
        cfg = load_simulator_config('ngspice')
        d = cfg.resolve_preset('tight')
        d['reltol'] = 999
        fresh = cfg.resolve_preset('tight')
        assert fresh['reltol'] != 999


class TestEmitOptionsLine:

    def test_empty_options_emits_empty_string(self):
        cfg = load_simulator_config('ngspice')
        assert cfg.emit_options_line({}) == ''

    def test_ngspice_emits_dotoptions_line(self):
        cfg = load_simulator_config('ngspice')
        out = cfg.emit_options_line({'reltol': '1e-5', 'method': 'gear'})
        assert out.startswith('.options')
        assert 'reltol=1e-5' in out
        assert 'method=gear' in out
        assert out.endswith('\n')

    def test_fallback_simulator_uses_default_emit(self):
        """Missing-YAML fallback still produces a usable .options line."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cfg = load_simulator_config('nonexistent_sim')
        assert cfg.emit_options_line({'foo': 'bar'}) == '.options foo=bar\n'


class TestKnownOptionsWarn:

    def test_unknown_option_emits_warning(self):
        cfg = load_simulator_config('ngspice')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            cfg.warn_unknown({'totally_made_up_option': 42})
        assert any('not in' in str(x.message)
                   and 'known_options' in str(x.message) for x in w)

    def test_known_option_no_warning(self):
        cfg = load_simulator_config('ngspice')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            cfg.warn_unknown({'reltol': 1e-6})
        assert not any('known_options' in str(x.message) for x in w)

    def test_no_allowlist_no_warning(self):
        """A config with empty known_options should not warn on anything."""
        cfg = SimulatorConfig({'name': 'anon'})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            cfg.warn_unknown({'anything': 1})
        assert not any('known_options' in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# Module.set_simulator_options
# ---------------------------------------------------------------------------

class TestSetSimulatorOptions:

    def test_preset_populates_options(self, pdk, tmp_dir):
        m = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        m.set_simulator_options(preset='tight')
        assert 'reltol' in m.simulator_options
        assert 'method' in m.simulator_options

    def test_raw_options_without_preset(self, pdk, tmp_dir):
        m = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # reltol is known, won't warn
            m.set_simulator_options(reltol=1e-6)
        assert m.simulator_options == {'reltol': 1e-6}

    def test_preset_plus_override(self, pdk, tmp_dir):
        m = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        m.set_simulator_options(preset='tight', method='trap')
        # preset set method=gear; kwargs must win
        assert m.simulator_options['method'] == 'trap'
        # other preset keys still present
        assert m.simulator_options['trtol'] == 1

    def test_calls_accumulate(self, pdk, tmp_dir):
        m = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        m.set_simulator_options(preset='default')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m.set_simulator_options(reltol=1e-7)
            m.set_simulator_options(abstol=1e-13)
        assert m.simulator_options == {'reltol': 1e-7, 'abstol': 1e-13}

    def test_unknown_preset_raises(self, pdk, tmp_dir):
        m = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        with pytest.raises(ValueError, match='Unknown preset'):
            m.set_simulator_options(preset='bogus_preset')

    def test_deprecated_alias_still_works(self, pdk, tmp_dir):
        """set_spice_options forwards to set_simulator_options with a
        DeprecationWarning."""
        m = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            m.set_spice_options(preset='tight')
        assert any(issubclass(x.category, DeprecationWarning) for x in w)
        assert 'reltol' in m.simulator_options


# ---------------------------------------------------------------------------
# End-to-end: options render into the netlist
# ---------------------------------------------------------------------------

class TestNetlistEmission:

    def test_options_line_present_in_generated_netlist(self, pdk, tmp_dir):
        """A module with set_simulator_options(...) produces an .options
        line in the generated netlist, before any .tran / .ac."""
        from silk.simulation import Transient
        from silk.sources.classes import VoltageSource

        m = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        m.add_source(VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8))
        m.set_simulator_options(preset='tight')
        m.add_sim_type(Transient(step=1e-9, stop=10e-9))
        m_deck_path = m.generate_netlist(corner='tt')
        netlist_text = open(m_deck_path).read()
        assert '.options' in netlist_text
        assert 'reltol=' in netlist_text

    def test_no_options_line_when_not_set(self, pdk, tmp_dir):
        """Modules that never call set_simulator_options don't emit the line."""
        from silk.simulation import Transient
        from silk.sources.classes import VoltageSource

        m = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        m.add_source(VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8))
        m.add_sim_type(Transient(step=1e-9, stop=10e-9))
        m_deck_path = m.generate_netlist(corner='tt')
        netlist_text = open(m_deck_path).read()
        assert '.options' not in netlist_text
