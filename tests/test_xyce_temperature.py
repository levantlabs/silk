"""Xyce applies the temperature silk configures, or silk says it does not know.

Xyce files ambient TEMP under its DEVICE options package. Verified on Xyce
7.10.0, ``.options timeint temp=85`` — the line silk used to emit — is accepted
with only a netlist warning and the circuit then solves at the default 27 C.
So "the deck parsed" is not evidence, and neither is a silenced known-option
warning: these tests check the package silk emits into, and the temperature the
simulator reports having actually solved at.

The fast tests need neither Xyce nor an installed PDK (ideal sources and
passives emit no ``.lib``). The real-Xyce proof is at the bottom, behind
``slow``/``requires_xyce``.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests._mock_rawfile import write_complete_rawfile

from silk import Module, load_pdk, sources
from silk.elements import elements
from silk.exceptions import (
    CharacterizationContextError, NotSupportedError, SimulationError,
)
from silk.simulation import AC, DC, Noise, Transient
from silk.simulation.conditions import read_xyce_temperatures
from silk.simulation.xyce_runner import _xyce_executable
from silk.simulators import load_simulator_config
from tests._env import needs_pdk

_FIXTURE_RAW = Path(__file__).parent / 'fixtures/rawfiles/e_ng_tran.raw'
# Resolve Xyce exactly the way silk does ($SILK_XYCE_BIN, then PATH), falling
# back to the owner's install location that AGENTS.md documents.
_XYCE_BIN = (_xyce_executable() if (os.environ.get('SILK_XYCE_BIN')
                                    or shutil.which('Xyce'))
             else os.path.expanduser('~/xyce/Xyce-7.10.0/bin/Xyce'))
_have_xyce = os.path.isfile(_XYCE_BIN) and os.access(_XYCE_BIN, os.X_OK)


def _module(tmp_path, simulator='xyce', analyses=None):
    """A PDK-free top-level Module: ideal source + resistor, one analysis."""
    mod = Module('sky130', str(tmp_path), 'tb.cir', load_pdk('sky130'))
    for net in ('out', '0'):
        mod.add_net(net)
    mod.add_source(sources.VoltageSource('vin', positive='out', negative='0',
                                         dc=1.8, ac=1.0))
    mod.add(elements.resistor('rload', p='out', n='0', value=1000))
    for analysis in analyses or [Transient(step=1e-6, stop=2e-6)]:
        mod.add_sim_type(analysis)
    mod.simulator = simulator
    return mod


def _deck(mod, corner='tt'):
    mod_deck_path = mod.generate_netlist(corner=corner)
    return Path(mod_deck_path).read_text()


# ---------------------------------------------------------------------------
# Emission: the right options package, not merely an accepted deck
# ---------------------------------------------------------------------------

def test_temperature_is_emitted_into_the_device_package(tmp_path):
    """TEMP on the TIMEINT line is the bug; being in known_options is not a fix."""
    mod = _module(tmp_path)
    mod.set_simulator_options(preset='tight')
    mod.set_temperature(85)
    deck = _deck(mod)

    assert '.options device temp=85.0\n' in deck
    # The precise regression: Xyce accepts this line and ignores the value.
    assert not re.search(r'^\.options\s+timeint\b.*\btemp=', deck, re.M | re.I)
    assert deck.count('temp=') == 1

    # Suppressing the unknown-option warning alone cannot satisfy this: temp
    # is in the allowlist, and the allowlist is not what puts it in DEVICE.
    cfg = load_simulator_config('xyce')
    assert 'temp' in cfg.known_options
    assert cfg.emit_options_line({'temp': 85}) == '.options device temp=85\n'


def test_timeint_tolerances_survive_the_temperature_route(tmp_path):
    """Moving TEMP must not move — or drop — the options sharing its old line."""
    mod = _module(tmp_path)
    mod.set_simulator_options(preset='tight')
    mod.set_temperature(85)
    deck = _deck(mod)

    timeint = [ln for ln in deck.splitlines() if ln.startswith('.options timeint')]
    assert len(timeint) == 1
    for key, value in load_simulator_config('xyce').resolve_preset('tight').items():
        assert f'{key}={value}' in timeint[0]
    assert 'temp' not in timeint[0]


def test_temp_and_tnom_are_not_conflated(tmp_path):
    """TNOM is a model reference, not a way to ask for a hot run.

    Only TEMP is routed. TNOM keeps exactly the treatment it had before this
    change (unlisted, unrouted, passed through with the allowlist warning);
    exposing further DEVICE options is deliberately out of scope.
    """
    mod = _module(tmp_path)
    with pytest.warns(UserWarning, match='known_options'):
        mod.set_simulator_options(tnom=20)
    mod.set_temperature(85)
    deck = _deck(mod)
    assert '.options device temp=85.0' in deck
    assert 'tnom' not in [ln for ln in deck.splitlines()
                          if ln.startswith('.options device')][0]
    assert re.search(r'^\.options\s+timeint\b.*\btnom=20', deck, re.M)


def test_ngspice_emission_is_untouched_by_the_routing_change(tmp_path):
    """ngspice has no packages; its options stay on exactly one .options line."""
    cfg = load_simulator_config('ngspice')
    assert (cfg.emit_options_line({'temp': 85, 'reltol': 1e-5})
            == '.options temp=85 reltol=1e-05\n')

    mod = _module(tmp_path, simulator='ngspice')
    mod.set_simulator_options(preset='tight')
    mod.set_temperature(85)
    deck = _deck(mod)
    options = [ln for ln in deck.splitlines() if ln.startswith('.options')]
    assert len(options) == 1 and 'temp=85.0' in options[0]


def test_routed_line_order_is_deterministic_and_default_line_comes_first():
    cfg = load_simulator_config('xyce')
    for options in ({'temp': 85, 'reltol': 1e-5}, {'reltol': 1e-5, 'temp': 85}):
        assert (cfg.emit_options_line(options)
                == '.options timeint reltol=1e-05\n.options device temp=85\n')
    # All-routed: no empty TIMEINT line.
    assert cfg.emit_options_line({'temp': 85}) == '.options device temp=85\n'
    assert cfg.emit_options_line({}) == ''


# ---------------------------------------------------------------------------
# The runtime probe
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('analyses,keyword', [
    ([Transient(step=1e-6, stop=2e-6)], 'tran'),
    ([AC('dec', 2, 1, 10)], 'ac'),
    ([Noise('out', '0', 'vin', 'dec', 2, 1, 10)], 'noise'),
])
def test_probe_is_emitted_only_for_a_temperature_configured_run(
        tmp_path, analyses, keyword):
    mod = _module(tmp_path, analyses=analyses)
    assert 'xyce_temperature.csv' not in _deck(mod)
    assert mod._run_files.temperature_path is None

    mod.set_temperature(85)
    deck = _deck(mod)
    probe = mod._run_files.temperature_path
    assert probe.endswith('/xyce_temperature.csv')
    assert f'.print {keyword} format=csv file={probe} {{TEMP}}\n' in deck
    # The analysis output keeps its own file and its own vectors.
    assert deck.count('.print') == 2


def test_probe_file_is_removed_before_a_new_attempt(tmp_path):
    mod = _module(tmp_path)
    mod.set_temperature(85)
    _deck(mod)
    probe = Path(mod._run_files.temperature_path)
    probe.write_text('TIME,{TEMP}\n0.0,85.0\n')
    mod._run_files.clear_previous_outputs()
    assert not probe.exists()


@pytest.mark.parametrize('text,expected', [
    ('TIME,{TEMP}\n0.0,8.5e+01\n1.0,8.5e+01\n', (85.0, 85.0)),
    ('{TEMP}\n-4.0e+01\n', (-40.0,)),                      # dc: no x column
    ('"FREQ","{TEMP}"\n1.0,8.5e+01\n', (85.0,)),           # noise: quoted
    ('FREQ,{temp}\n1.0,2.7e+01\n', (27.0,)),               # case-insensitive
    ('TIME,V(out)\n0.0,1.8\n', ()),                        # no TEMP column
    ('TIME,{TEMP}\n0.0,not-a-number\n', ()),               # unparseable
    ('TIME,{TEMP}\n0.0,-400\n', ()),                       # below absolute zero
    ('', ()),
])
def test_probe_reader_returns_evidence_or_nothing(tmp_path, text, expected):
    path = tmp_path / 'xyce_temperature.csv'
    path.write_text(text)
    assert read_xyce_temperatures(path) == expected


def test_probe_reader_has_no_evidence_without_a_file(tmp_path):
    assert read_xyce_temperatures(None) == ()
    assert read_xyce_temperatures(tmp_path / 'absent.csv') == ()


# ---------------------------------------------------------------------------
# Confirm-or-fail, with a stubbed Xyce (no simulator needed)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_xyce(monkeypatch):
    """Stand in for Xyce, writing whatever temperature the test dictates."""
    state = {'applied': None, 'write_probe': True, 'calls': 0}

    def run(argv, **kwargs):
        state['calls'] += 1
        deck = Path(argv[-1]).read_text()
        raw = re.search(r'^\.print \w+ format=raw file=(\S+)', deck, re.M)
        write_complete_rawfile(raw.group(1), deck, simulator='xyce', template=_FIXTURE_RAW)
        probe = re.search(r'^\.print \w+ format=csv file=(\S+) \{TEMP\}', deck, re.M)
        if probe and state['write_probe']:
            # Xyce reports what it solved at, which is the DEVICE setting —
            # a TIMEINT one leaves it at the 27 C default.
            device = re.search(r'^\.options device\b.*\btemp=(\S+)', deck, re.M | re.I)
            applied = state['applied']
            if applied is None:
                applied = float(device.group(1)) if device else 27.0
            Path(probe.group(1)).write_text(
                f'TIME,{{TEMP}}\n0.0,{applied:.8e}\n1.0e-06,{applied:.8e}\n')
        return subprocess.CompletedProcess(argv, 0, stdout='End of Xyce', stderr='')

    monkeypatch.setattr(subprocess, 'run', run)
    return state


def test_confirmed_temperature_is_published_with_an_observed_basis(tmp_path, fake_xyce):
    mod = _module(tmp_path)
    mod.set_temperature(85)
    mod.simulate(corner='tt')
    conditions = mod.operating_conditions
    assert conditions.backend == 'xyce'
    assert conditions.temperature_kind == 'fixed'
    assert conditions.temperature_c == 85
    assert conditions.temperature_basis == 'configured-and-observed'


def test_native_option_route_shares_the_same_confirmed_value(tmp_path, fake_xyce):
    mod = _module(tmp_path)
    mod.set_simulator_options(temp='85')
    mod.simulate(corner='tt')
    assert mod.operating_conditions.temperature_c == 85
    assert mod.operating_conditions.temperature_basis == 'native-and-observed'


def test_repeated_setting_replaces_and_is_confirmed_at_the_new_value(tmp_path, fake_xyce):
    mod = _module(tmp_path)
    mod.set_output_options(mode='overwrite')
    mod.set_temperature(85)
    mod.set_temperature(-40)
    mod.simulate(corner='tt')
    assert mod.operating_conditions.temperature_c == -40
    deck = Path(os.path.join(mod.dir_folder, mod.netlistname)).read_text()
    assert deck.count('temp=') == 1 and '.options device temp=-40.0' in deck

    mod.set_temperature(None)
    mod.simulate(corner='tt')
    assert mod.operating_conditions.temperature_c is None
    assert mod.operating_conditions.temperature_kind == 'unknown'


def test_a_simulator_that_ignored_the_setting_fails_instead_of_reporting_it(
        tmp_path, fake_xyce):
    """The TASK-140 bug class: deck accepted, temperature silently discarded."""
    fake_xyce['applied'] = 27.0
    mod = _module(tmp_path)
    mod.set_temperature(85)
    with pytest.raises(SimulationError, match='Configured TEMP=85'):
        mod.simulate(corner='tt')
    assert mod.sim_results_loaded is False
    # A failed run publishes no conditions at all, never a hopeful 85.
    assert mod.operating_conditions is None


def test_missing_runtime_confirmation_is_never_treated_as_agreement(
        tmp_path, fake_xyce):
    fake_xyce['write_probe'] = False
    mod = _module(tmp_path)
    mod.set_temperature(85)
    with pytest.raises(SimulationError, match='no ambient TEMP confirmation'):
        mod.simulate(corner='tt')
    assert mod.operating_conditions is None


def test_a_stale_probe_cannot_confirm_a_later_run(tmp_path, fake_xyce):
    """Clearing outputs before launch keeps yesterday's 85 C out of today's run."""
    mod = _module(tmp_path)
    mod.set_temperature(85)
    mod.simulate(corner='tt')
    assert mod.operating_conditions.temperature_c == 85

    fake_xyce['write_probe'] = False
    mod.set_temperature(-40)
    with pytest.raises(SimulationError, match='no ambient TEMP confirmation'):
        mod.simulate(corner='tt')


def test_deprecated_assertion_checks_xyce_setup_and_refuses_a_mismatch(
        tmp_path, fake_xyce):
    """`run(temperature=)` asserts; it never configures, on Xyce either."""
    dut = Module('sky130', str(tmp_path), 'dut.cir', load_pdk('sky130'),
                 module_name='load')
    dut.define_module_ports({'out': 'out', 'vss': 'ground'})
    dut.add(elements.resistor('rload', p='out', n='vss', value=1000))
    tb = dut.create_test('level')
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_source(sources.VoltageSource('supply', positive='out', negative='0',
                                        dc=1.8))
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    tb.add_metric('supply', lambda c: float(c.load_signal('out').y[-1]))
    tb.add_spec('supply present', 'supply', '>', 0.5, key='supply')
    tb.netlist.simulator = 'xyce'
    try:
        # No setup at all: the assertion cannot be satisfied by the label.
        with pytest.raises(CharacterizationContextError):
            with pytest.warns(DeprecationWarning):
                tb.run(corner='tt', temperature=85, voltage=1.8)
        assert fake_xyce['calls'] == 0

        tb.set_temperature(85)
        with pytest.warns(DeprecationWarning):
            report = tb.run(corner='tt', temperature=85, voltage=1.8)
        assert report.operating_conditions.temperature_c == 85
        assert report.characterization_context.temperature == 85

        with pytest.raises(CharacterizationContextError):
            with pytest.warns(DeprecationWarning):
                tb.run(corner='tt', temperature=35, voltage=1.8)
    finally:
        dut.close()


def test_new_output_mode_relocates_the_probe_with_the_run(tmp_path, fake_xyce):
    mod = _module(tmp_path)
    mod.set_output_options(mode='new')
    mod.set_temperature(85)
    mod.simulate(corner='tt')
    probe = Path(mod._run_files.temperature_path)
    assert probe.parent == Path(mod._run_files.directory) != tmp_path
    assert probe.is_file()
    assert mod.operating_conditions.temperature_c == 85


def test_unconfigured_run_reports_unknown_rather_than_a_default(tmp_path, fake_xyce):
    mod = _module(tmp_path)
    mod.simulate(corner='tt')
    conditions = mod.operating_conditions
    assert conditions.temperature_c is None
    assert conditions.temperature_kind == 'unknown'
    assert conditions.temperature_basis == 'unknown'


def test_custom_deck_text_that_may_set_a_temperature_is_refused(tmp_path):
    """Opaque custom SPICE plus managed TEMP fails before any launch.

    TASK-157's duplicate-declaration check now runs first and names BOTH inputs,
    which is the more useful refusal, so this asserts that message rather than
    the generic custom-SPICE one. Either way nothing launches. A custom line
    that could set a temperature but does not declare a second explicit value
    still takes the custom-SPICE path, covered below.
    """
    mod = _module(tmp_path)
    mod.add_custom_line('.temp 35')
    mod.set_temperature(85)
    with pytest.raises(SimulationError, match='Multiple explicit temperature'):
        mod.generate_netlist(corner='tt')


def test_opaque_custom_text_with_managed_temp_is_refused(tmp_path):
    """A line silk cannot read, plus managed TEMP, still fails before launch."""
    mod = _module(tmp_path)
    mod.add_custom_line('.include "unknown.lib"')
    mod.set_temperature(85)
    with pytest.raises(NotSupportedError, match='custom SPICE'):
        mod.generate_netlist(corner='tt')


def test_temperature_sweep_is_reported_as_a_sweep_not_a_fixed_point(
        tmp_path, fake_xyce):
    """A swept run has no single temperature, and asks for no probe."""
    mod = _module(tmp_path, analyses=[DC(source='temp', start=-40, stop=125,
                                         step=5)])
    mod.set_output_options(mode='overwrite')
    mod.simulate(corner='tt')
    conditions = mod.operating_conditions
    assert conditions.temperature_kind == 'sweep'
    assert conditions.temperature_c is None
    assert conditions.temperature_sweep.start == -40
    assert conditions.temperature_sweep.stop == 125
    assert 'xyce_temperature.csv' not in Path(os.path.join(mod.dir_folder, mod.netlistname)).read_text()


# ---------------------------------------------------------------------------
# Real Xyce
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.skipif(not _have_xyce, reason='Xyce unavailable; set $SILK_XYCE_BIN')
class TestRealXyce:
    """Executed evidence on Xyce 7.10.0 — emission proves nothing on its own."""

    @pytest.fixture(autouse=True)
    def _bin(self, monkeypatch):
        monkeypatch.setenv('SILK_XYCE_BIN', _XYCE_BIN)

    def test_xyce_applies_and_reports_the_configured_temperature(self, tmp_path):
        mod = _module(tmp_path)
        mod.set_simulator_options(preset='tight')
        mod.set_temperature(85)
        mod.simulate(corner='tt')

        conditions = mod.operating_conditions
        assert conditions.temperature_c == 85
        assert conditions.temperature_basis == 'configured-and-observed'
        # Independent of silk's bookkeeping: Xyce's own {TEMP} record.
        observed = read_xyce_temperatures(mod._run_files.temperature_path)
        assert observed and set(observed) == {85.0}

    @pytest.mark.parametrize('preset', ['default', 'tight', 'fast'])
    def test_xyce_accepts_the_deck_without_ignoring_the_temperature(
            self, tmp_path, preset):
        """Run silk's own deck directly and read Xyce's stdout.

        `.options timeint temp=` earns 'No options parameter temp found,
        parameter will be ignored'. The DEVICE route must not, and every named
        preset's TIMEINT knobs must still be accepted alongside it.
        """
        mod = _module(tmp_path)
        mod.set_simulator_options(preset=preset)
        mod.set_temperature(85)
        mod.set_output_options(mode='overwrite')
        deck = _deck(mod)
        assert '.options device temp=85.0' in deck

        out = subprocess.run([_XYCE_BIN, os.path.join(mod.dir_folder, mod.netlistname)], cwd=str(tmp_path),
                             capture_output=True, text=True, timeout=300)
        combined = (out.stdout or '') + (out.stderr or '')
        assert 'End of Xyce' in combined
        assert 'will be ignored' not in combined
        assert 'temp' not in combined.lower().split('device count summary')[0]

    def test_the_old_timeint_line_is_silently_ignored_by_xyce(self, tmp_path):
        """Why routing is the fix: Xyce takes the wrong-package line and drops it."""
        deck = tmp_path / 'timeint.cir'
        probe = tmp_path / 'probe.csv'
        deck.write_text(
            '* wrong package\n'
            'V1 1 0 DC 1\nR1 1 0 1k\n'
            '.options timeint temp=85 reltol=1e-5\n'
            '.tran 1n 2n\n'
            f'.print tran format=raw file={tmp_path}/t.raw V(1)\n'
            f'.print tran format=csv file={probe} {{TEMP}}\n.end\n')
        out = subprocess.run([_XYCE_BIN, str(deck)], cwd=str(tmp_path),
                             capture_output=True, text=True, timeout=300)
        assert 'will be ignored' in (out.stdout or '') + (out.stderr or '')
        assert set(read_xyce_temperatures(probe)) == {27.0}

    def test_ngspice_temp_card_is_not_portable_to_xyce(self, tmp_path):
        """`.temp 85` is an ngspice card; Xyce does not recognize it."""
        deck = tmp_path / 'dotcard.cir'
        probe = tmp_path / 'probe.csv'
        deck.write_text(
            '* ngspice card\n'
            'V1 1 0 DC 1\nR1 1 0 1k\n.temp 85\n.tran 1n 2n\n'
            f'.print tran format=raw file={tmp_path}/t.raw V(1)\n'
            f'.print tran format=csv file={probe} {{TEMP}}\n.end\n')
        out = subprocess.run([_XYCE_BIN, str(deck)], cwd=str(tmp_path),
                             capture_output=True, text=True, timeout=300)
        assert 'Unrecognized dot line' in (out.stdout or '') + (out.stderr or '')
        assert set(read_xyce_temperatures(probe)) == {27.0}

    def test_report_and_context_carry_the_confirmed_xyce_temperature(self, tmp_path):
        """The whole run()/report path, not just Module.simulate()."""
        dut = Module('sky130', str(tmp_path), 'dut.cir', load_pdk('sky130'),
                     module_name='load')
        dut.define_module_ports({'out': 'out', 'vss': 'ground'})
        dut.add(elements.resistor('rload', p='out', n='vss',
                                 value=1000))
        tb = dut.create_test('level')
        tb.add_module_under_test({'out': 'out', 'vss': '0'})
        tb.add_source(sources.VoltageSource('supply', positive='out',
                                            negative='0', dc=1.8))
        tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
        tb.add_metric('supply', lambda c: float(c.load_signal('out').y[-1]))
        tb.add_spec('supply present', 'supply', '>', 0.5, key='supply')
        tb.netlist.simulator = 'xyce'
        try:
            assert tb.set_temperature(85) is tb
            report = tb.run(corner='tt', voltage=1.8)
            conditions = report.operating_conditions
            assert conditions.backend == 'xyce'
            assert conditions.temperature_c == 85
            assert conditions.temperature_basis == 'configured-and-observed'
            assert report.characterization_context.temperature == 85

            # The deprecated assertion agrees, and a wrong one refuses.
            with pytest.warns(DeprecationWarning):
                assert tb.run(corner='tt', temperature=85,
                              voltage=1.8).operating_conditions.temperature_c == 85
            with pytest.raises(CharacterizationContextError):
                with pytest.warns(DeprecationWarning):
                    tb.run(corner='tt', temperature=35, voltage=1.8)
        finally:
            dut.close()

    def test_xyce_actually_runs_a_temperature_sweep(self, tmp_path):
        """`.dc temp` is not ngspice-only: Xyce 7.10.0 sweeps it for real."""
        mod = _module(tmp_path, analyses=[DC(source='temp', start=-40,
                                             stop=125, step=5)])
        mod.set_output_options(mode='overwrite')
        mod.simulate(corner='tt')
        assert mod.operating_conditions.temperature_kind == 'sweep'

        # Confirm the axis independently, with a probe added by hand.
        deck = Path(os.path.join(mod.dir_folder, mod.netlistname)).read_text().replace(
            '.end\n', f'.print dc format=csv file={tmp_path}/probe.csv '
                      '{TEMP}\n.end\n')
        swept = tmp_path / 'swept.cir'
        swept.write_text(deck)
        subprocess.run([_XYCE_BIN, str(swept)], cwd=str(tmp_path),
                       capture_output=True, text=True, timeout=300)
        observed = read_xyce_temperatures(tmp_path / 'probe.csv')
        assert observed[0] == -40 and observed[-1] == 125

    @needs_pdk
    def test_a_real_device_responds_to_the_applied_temperature(self, tmp_path):
        """Physics, not bookkeeping: same deck, two temperatures, two currents."""
        from silk.pdk.basepdk import BasePDK

        def current(temperature_c):
            pdk = load_pdk('sky130')
            assert isinstance(pdk, BasePDK)
            mod = Module('sky130', str(tmp_path / f't{temperature_c}'),
                         'hot.cir', pdk)
            for net in ('vout', 'vdd', 'g', '0'):
                mod.add_net(net)
            mod.add(pdk.nmos1p8('M1', d='vout', g='g', s='0', b='0', w=4.0, l=0.5, m=1))
            mod.add_source(sources.VoltageSource('vdd', positive='vdd',
                                                 negative='0', dc=1.8))
            mod.add(elements.resistor('rd', p='vdd', n='vout',
                                     value=10000))
            mod.add_source(sources.VoltageSource('vg', positive='g',
                                                 negative='0', dc=0.6))
            mod.add_sim_type(Transient(step=1e-8, stop=1e-7))
            mod.simulator = 'xyce'
            mod.set_temperature(temperature_c)
            mod.simulate(corner='tt')
            assert mod.operating_conditions.temperature_c == temperature_c
            return float(mod.load_signal('vout').y[-1])

        cold, hot = current(27), current(85)
        assert abs(hot - cold) > 1e-3, (
            f'vout identical at 27 C and 85 C ({cold} vs {hot}): the '
            f'temperature reached the report but not the devices')


def test_xyce_supply_sweep_is_not_a_nominal_fixed_voltage(tmp_path, fake_xyce):
    mod = _module(tmp_path, analyses=[DC('supply', 0, 2, 0.5)])
    name = next(iter(mod.sig_sources_dict))
    mod.set_supply_voltage(1.8, source=name)
    mod.add_sim_type(DC(name, 0, 2, 0.5))
    mod.simulate()
    c = mod.operating_conditions
    assert c.supply_kind == 'sweep' and c.supply_voltage_v is None
    assert c.supply_source == name
    assert (c.supply_sweep.start, c.supply_sweep.stop, c.supply_sweep.step) == (0, 2, 0.5)


@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.skipif(not _have_xyce, reason='Xyce unavailable; set $SILK_XYCE_BIN')
def test_real_xyce_supply_sweep(tmp_path):
    import numpy as np
    mod = _module(tmp_path, analyses=[DC('vin', 0, 2, 0.5)])
    name = next(iter(mod.sig_sources_dict))
    mod.set_supply_voltage(1.8, source=name)
    mod.simulate()
    assert mod.operating_conditions.supply_kind == 'sweep'
    np.testing.assert_allclose(mod.load_signal('out').y, [0, 0.5, 1, 1.5, 2])
