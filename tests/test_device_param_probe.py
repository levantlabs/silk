"""``tb.probe('st1.MN:gm')`` — device-parameter probes (TASK-330.8).

One probe spelling for a MOS device's raw operating-point parameter,
usable in every analysis (``Op``, ``DC`` sweep, ``Transient``, ``AC``),
alongside the existing node / ``i(...)`` / ``':'`` terminal-current probe
spellings ``ModuleTestbench.probe`` already supports.

**Spelling history**: TASK-330.8 originally shipped ``'st1.MN[gm]'`` (a
trailing bracket). A regression on the merged commit
(a492303d) found this collided with a net literally named the same way
(a bus bit designed ``'bus[en]'`` was wrongly refused as a bad device
parameter). OWNER DECISION (2026-09-28) dropped the bracket spelling
ENTIRELY in favour of ``'st1.MN:gm'`` — ``':'`` was already reserved for
the pre-330.8 terminal-current spelling (``'xdut.m1:d'``). TASK-553 now
refuses separators and symbolic indices in authored names; numeric indices
remain legal, and probe parsing is unchanged.
``probe()`` dispatches on the suffix after the LAST ``':'``
(case-insensitive): one of the 4 terminals (``d``/``g``/``s``/``b``) is
the unchanged terminal-current probe; one of the 17
:data:`~silk.simulation.backend.DEVICE_OP_PARAMS` names is the
device-parameter probe, with the prefix as the DESIGNER name; anything
else is refused, listing both closed sets. ``'[param]'`` has NO special
meaning at all any more — a bracket-shaped name is always a plain net.

``TestEmissionNoSimulator`` covers CHANGE 1 (address-only pre-run refusal),
CHANGE 2 (derived ``op_table()`` columns are not probe parameters), the
ammeter-wrapper refusal, and AC #4's byte-identical deck.
``TestRealNgspice``/``TestRealXyce`` need a real simulator + the sky130
PDK and cover AC #3 (op-vs-``op_table()`` agreement, DC-sweep tracking,
``gm``/``gds`` arithmetic + ``Spec`` pass/fail, both polarities),
MISSING 1 (joint node + ``i()`` + device probe, default save and
``set_save_list``, byte-identical other vectors), MISSING 2 (post-run
probe-creation lifecycle raise), MISSING 3 (AC shape — real values, no
complex, same ``xlabel``), and AC #6 (42-vs-47 diagnostic).

``TestPreflightMechanism`` covers the reviewer's 2026-09-28 ruling on the
fix (B) proposal: a rejected guarded re-run (a silent, unwarned all-zero
vector at a ~10-saved-vector threshold cannot be caught by re-running) was
replaced by a mandated ngspice-only pre-run availability check, scoped to
params outside a device's declared ``op_params``, cached per (binary,
model, param) for the process.

``TestColonDispatch`` covers the owner's 2026-09-28 ``':'``-spelling
decision itself: the bracket spelling is dead (any bracket-shaped name,
including one that collides with a real device's designer name, is
always a plain net — no fixture-byte-identical drift from pre-330.8);
the two colon-suffix sets are disjoint; a ``'v(...)'`` wrapper always
bypasses the device branch; and an unrecognised colon suffix is refused
listing both sets.
"""
import json
import math
import os

import pytest

from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.exceptions import SimulationError
from silk import sources
from silk.elements import elements
from silk.simulation import analysis
from silk.simulation.backend import DEVICE_OP_PARAMS, XYCE_OP_CAPABILITY

from tests._env import needs_pdk, needs_ngspice
from tests._env import scratch_dir

_NGSPICE_47 = os.path.expanduser('~/ngspice/ngspice-47/bin/ngspice')


def _ngspice_bins():
    bins = [None]
    if os.path.isfile(_NGSPICE_47) and os.access(_NGSPICE_47, os.X_OK):
        bins.append(_NGSPICE_47)
    return bins


_XYCE_BIN = os.path.expanduser('~/xyce/Xyce-7.10.0/bin/Xyce')
_SKY130_LIB = os.path.expanduser(
    '~/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice')
_have_xyce = os.path.exists(_XYCE_BIN) and os.path.exists(_SKY130_LIB)

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures',
                            'device_param_probe')


def _sky130():
    from silk.pdk import load_pdk
    return load_pdk('sky130')


def _assert_probe_message(signal, designer_name, param, analysis_word):
    """Reviewer gate finding 2026-09-28 on c0fa35b9 (bracket spelling;
    still the correct shape for the current ':' spelling): an AC #6
    diagnostic must render the probe as the caller typed it
    ("probe('<name>:<param>')"), and must name the analysis it actually
    read (*analysis_word*), never the literal 'analysis=None' a caller
    who didn't pass analysis= would otherwise see."""
    try:
        signal.y
    except SimulationError as exc:
        message = str(exc)
        assert f"probe('{designer_name}:{param}')" in message
        assert f"analysis={analysis_word!r}" in message
        assert 'analysis=None' not in message
        return
    raise AssertionError('expected SimulationError on read, got none')


def _prepared_deck(t, corner='tt'):
    """Force deck emission (validating + resolving any registered device
    probe, CHANGE 1) without driving a simulator -- the same
    ``_prepare_run()`` -> ``generate_netlist()`` sequence ``run()``/
    ``run_test()`` take, stopped one step short of launching."""
    spec = t._prepare_run(corner=corner)
    spec_netlist_deck_path = spec.netlist.generate_netlist(corner=spec.corner_str)
    with open(spec_netlist_deck_path) as fh:
        return fh.read()


def _single_bench(pdk=None, devname='nmos1p8', vg=1.0, vd=1.0, m1_name='M1',
                  extra_ports=None, extra_nets=None):
    """One MOS device, relative bias (source/bulk grounded) -- vg/vd
    negative gives the PMOS bias the same way ``test_op_table.py`` does.
    *extra_ports*/*extra_nets* wire additional DUT ports straight through
    to like-named top-level nets, for the bracket-shaped bus-bit guard
    tests (TASK-330.8 owner decision 2026-09-28)."""
    pdk = pdk or _sky130()
    tmp = scratch_dir()
    dut = Module('sky130', tmp, 'dut.cir', pdk_details=pdk, module_name='dut')
    ports = {'d': PortDir.OUT, 'g': PortDir.IN,
            's': PortDir.GROUND, 'b': PortDir.GROUND}
    ports.update(extra_ports or {})
    dut.define_module_ports(ports)
    dut.add(getattr(pdk, devname)(m1_name, d='d', g='g', s='s', b='b', w=1.0, l=0.15, m=1))
    t = dut.create_test('probechk')
    nets = {'d': 'vd', 'g': 'vg', 's': '0', 'b': '0'}
    nets.update(extra_nets or {})
    t.add_module_under_test(nets)
    t.add_source(sources.VoltageSource('vd', positive='vd', negative='0', dc=vd))
    t.add_source(sources.VoltageSource('vg', positive='vg', negative='0', dc=vg))
    return dut, t


# ---------------------------------------------------------------------------
# Emission-only -- no simulator needed.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _remove_preflight_logs():
    """A failed/unavailable preflight keeps its log directory (the error
    message names it); remove those after each test so the run leaves
    nothing behind (TASK-453)."""
    yield
    import shutil
    from silk.circuit.spice_netlister import PySpiceNetlister
    for result in list(PySpiceNetlister._PREFLIGHT_CACHE.values()):
        if result.log_path:
            shutil.rmtree(os.path.dirname(result.log_path),
                          ignore_errors=True)

class TestEmissionNoSimulator:
    def test_no_device_probes_deck_is_byte_identical(self):
        """AC #4: the feature is fully opt-in -- a bench that never
        registers a device-parameter probe emits exactly the deck it always
        did (two independently-built, otherwise-identical benches produce
        byte-identical decks; no '@m....[..]' artifact appears at all)."""
        import re

        def _deck():
            dut, t = _single_bench()
            t.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
            t.probe('vd')                      # ordinary node probe only
            return re.sub(r'/tmp/\S+', '<tmpdir>', _prepared_deck(t))
        deck_a, deck_b = _deck(), _deck()
        assert deck_a == deck_b
        assert '@m.' not in deck_a.lower()

    def test_derived_column_raises_before_run_with_arithmetic_hint(self):
        dut, t = _single_bench()
        t.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
        with pytest.raises(ValueError, match=r"gm_id.*DERIVED"):
            t.probe('M1:gm_id')
        try:
            t.probe('M1:gm_id')
        except ValueError as exc:
            assert "tb.probe('M1' + ':gm') / tb.probe('M1' + ':id')" in str(exc)

    def test_categorical_derived_column_raises(self):
        dut, t = _single_bench()
        with pytest.raises(ValueError, match='region'):
            t.probe('M1:region')

    @pytest.mark.parametrize('param', ['region', 'saturation_weak'])
    @pytest.mark.parametrize('helper', [False, True])
    def test_categorical_guidance_uses_dut_row(self, param, helper):
        from silk.circuit.modules_results import (
            capture_op_device_records, derived_column_probe_error)
        from silk.circuit.op_table import build_op_table
        from silk.simulation.backend import SimulatorBackend
        stage, _ = _single_bench()
        dut = Module('sky130', scratch_dir(), 'outer.cir', pdk_details=_sky130(),
                     module_name='outer')
        dut.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN,
                                 's': PortDir.GROUND, 'b': PortDir.GROUND})
        mapping = {'d': 'd', 'g': 'g', 's': 's', 'b': 'b'}
        dut.add_module(stage, 'bias1', nets=mapping)
        tb = dut.create_test('opchk')
        tb.add_module_under_test(mapping)
        if helper:
            tb.add_module(stage, 'bias1', nets=mapping)
        with pytest.raises(ValueError) as exc:
            tb.probe('bias1.M1:' + param)
        key = 'dut.bias1.M1'
        assert f"report.op_table()[{key!r}].{param}" in str(exc.value)
        assert "tb.probe('bias1.M1' + ':vgs')" in str(exc.value)
        backend = SimulatorBackend.get('ngspice')
        backend.device_op_vectors = lambda *args, **kwargs: {}
        rows = capture_op_device_records(tb.netlist, backend, None, pdk=_sky130())
        table = build_op_table(rows, None, None)
        assert table[key].path == 'xdut.xbias1.XMM1'
        if helper:
            assert table['bias1.M1'].path == 'xbias1.XMM1'
        # The same helper formats refusals for post-run probe reads.
        message = str(derived_column_probe_error('bias1.MN:' + param, 'bias1.MN', param))
        assert "report.op_table()['dut.bias1.MN']." + param in message

    def test_unknown_device_name_raises_with_close_match(self):
        dut, t = _single_bench()
        t.add_sim_type(analysis.Op())
        t.probe('N1:gm')                      # registration never validates
        with pytest.raises(ValueError, match='M1') as excinfo:
            _prepared_deck(t)
        assert "probe('N1:gm')" in str(excinfo.value)

    def test_non_mos_device_raises_not_mos_shaped(self):
        dut = Module('sky130', scratch_dir(), 'dut.cir',
                     pdk_details=_sky130(), module_name='dut')
        dut.define_module_ports({'a': PortDir.INOUT, 'b': PortDir.INOUT})
        with dut:
            elements.resistor('r1', p='a', n='b', value=1000)
        t = dut.create_test('probechk')
        t.add_module_under_test({'a': 'va', 'b': 'vb'})
        t.add_source(sources.VoltageSource('va', positive='va', negative='0', dc=1.0))
        t.add_sim_type(analysis.Op())
        t.probe('r1:gm')
        with pytest.raises(ValueError, match='not a MOS-shaped device') as excinfo:
            _prepared_deck(t)
        assert "probe('r1:gm')" in str(excinfo.value)

    def test_unrecognized_suffix_refused_at_probe_time_not_deferred(self):
        """Reviewer gate finding on 85cd1657 (2026-09-28), fix (A),
        carried over into the ':' spelling (owner decision 2026-09-28):
        a colon suffix that is neither a terminal nor a recognised
        device parameter is refused immediately in probe() itself,
        before any deck is even built (not deferred to a read-time
        raise), naming both closed sets with a difflib hint. Also
        confirms a RECOGNISED-but-undeclared param (a real
        DEVICE_OP_PARAMS name the device model just doesn't happen to
        declare, unlike a typo) still reaches emission unhindered --
        that check stays at read time, only the closed-set membership
        check moved earlier."""
        dut, t = _single_bench()
        t.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
        with pytest.raises(ValueError, match='gm') as excinfo:
            t.probe('M1:totally_bogus_suffix')
        assert "probe('M1:totally_bogus_suffix')" in str(excinfo.value)
        assert 'terminal' in str(excinfo.value)
        deck = _prepared_deck(t)                # no probe registered -- fine
        assert '@m.' not in deck.lower()

    def test_recognized_param_reaches_emission_unaffected_by_fix_a(self):
        """'gm' -- declared in sky130's op_params, so it is trusted and
        needs no preflight (reviewer ruling 2026-09-28, fix B): this
        stays a pure no-simulator emission test. 'cgso', by contrast, is
        a recognised DEVICE_OP_PARAMS name but NOT declared for sky130,
        so reaching emission for it now genuinely launches a preflight
        ngspice process -- see TestPreflightMechanism for that path."""
        dut, t = _single_bench()
        t.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
        t.probe('M1:gm')                       # a REAL, DECLARED param
        deck = _prepared_deck(t)
        assert '[gm]' in deck.lower()          # ngspice's OWN '@' addressing

    def test_xyce_refuses_unsupported_param_before_run(self):
        """CHANGE 1: Xyce is restrictive pre-run -- a param outside
        XYCE_OP_CAPABILITY (e.g. 'cgg', a capacitance) is refused at deck
        generation, before any run, since one bad Xyce .print symbol
        aborts the whole run."""
        dut, t = _single_bench()
        dut.simulator = 'xyce'
        t.add_sim_type(analysis.Op())
        t.probe('M1:cgg')
        assert 'cgg' not in XYCE_OP_CAPABILITY
        with pytest.raises(ValueError, match='XYCE_OP_CAPABILITY|N\\(\\)/ID\\(\\)') as excinfo:
            _prepared_deck(t)
        assert "probe('M1:cgg')" in str(excinfo.value)

    def test_xyce_accepts_supported_param_before_run(self):
        dut, t = _single_bench()
        dut.simulator = 'xyce'
        t.add_sim_type(analysis.Op())
        t.probe('M1:gm')
        deck = _prepared_deck(t)               # must not raise
        assert 'N(' in deck.upper() or 'ID(' in deck.upper()

    def test_ammeter_wrapper_combination_refused(self):
        """A device probe together with save_device_currents=True /
        terminal_current_method='ammeter_wrapper' is refused unconditionally
        (batch-write mode never runs ngspice's checkvalid gate, so a
        missing/unsupported vector would come back as a real, populated,
        all-zero reading -- AC #6 forbids a silent 0)."""
        dut, t = _single_bench()
        dut.save_device_currents = True
        dut.terminal_current_method = 'ammeter_wrapper'
        t.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
        t.probe('M1:gm')
        with pytest.raises(SimulationError, match='ammeter_wrapper'):
            _prepared_deck(t)

    def test_noise_only_analysis_refused(self):
        dut, t = _single_bench()
        t.add_sim_type(analysis.Noise(output='vd', ref=None, source='vg',
                                      variation='dec', points=10,
                                      fstart=1e3, fstop=1e6))
        t.probe('M1:gm')
        with pytest.raises(SimulationError, match='Noise'):
            _prepared_deck(t)


# ---------------------------------------------------------------------------
# ':' dispatch itself -- owner decision 2026-09-28, no simulator needed.
# ---------------------------------------------------------------------------
class TestColonDispatch:
    """Covers the ':' spelling decision directly: brackets are dead (a
    bus-bit net collision cannot recur, by construction -- there is no
    bracket handling left to collide), the two colon-suffix sets are
    disjoint, 'v(...)' always bypasses the device branch, and an
    unrecognised suffix lists both sets."""

    def test_terminal_and_device_param_suffix_sets_are_disjoint(self):
        from silk.circuit.module_test import _TERMINAL_PORTS
        assert not (set(_TERMINAL_PORTS) & set(DEVICE_OP_PARAMS))

    def test_bracket_shaped_names_are_always_plain_nets(self):
        """Numeric brackets remain net spelling (TASK-553 owner ruling)."""
        import re
        dut, t = _single_bench(
            extra_ports={'data3': PortDir.INOUT, 'busen': PortDir.INOUT},
            extra_nets={'data3': 'data[3]', 'busen': 'bus[1]'})
        t.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
        d3 = t.probe('data[3]')
        be = t.probe('bus[1]')
        assert d3.name == 'data[3]'
        assert be.name == 'bus[1]'
        # Normalize library roots before /tmp output paths: a PDK root may
        # itself live under /tmp. Keep the complete PDK-relative paths strict.
        deck = re.sub(r'(?m)^(\.(?:lib|include) ")[^"\n]+(?=/sky130A/)',
                      r'\1<pdk_root>', _prepared_deck(t))
        deck = re.sub(r'/tmp/\S+', '<tmpdir>', deck)
        fixture_path = os.path.join(_FIXTURE_DIR,
                                    'pre_330_8_bracket_net_names.cir')
        with open(fixture_path) as fh:
            fixture = fh.read()
        assert deck == fixture

    def test_bracket_named_net_colliding_with_a_real_device_name(self):
        """Numeric net indices do not dispatch to a device parameter."""
        dut, t = _single_bench(
            extra_ports={'weird': PortDir.INOUT},
            extra_nets={'weird': 'M1[3]'})
        t.add_sim_type(analysis.Op())
        sig = t.probe('M1[3]')
        assert sig.name == 'M1[3]'
        deck = _prepared_deck(t)                # must not raise
        assert '@m.' not in deck.lower()

    def test_v_wrapper_always_bypasses_the_device_branch(self):
        """Owner decision 2026-09-28, point (3): 'v(M1:gm)' always means
        the net -- checked BEFORE the colon dispatch, so a name that
        would otherwise parse as a device-parameter probe is never
        reinterpreted once wrapped in 'v(...)'."""
        dut, t = _single_bench()
        t.add_sim_type(analysis.Op())
        sig = t.probe('v(M1:gm)')
        assert sig.name == 'M1:gm'
        assert sig.type == 'voltage'

    def test_unrecognized_colon_suffix_lists_both_sets(self):
        dut, t = _single_bench()
        with pytest.raises(ValueError) as excinfo:
            t.probe('M1:zzz')
        message = str(excinfo.value)
        assert "probe('M1:zzz')" in message
        for terminal in ('d', 'g', 's', 'b'):
            assert repr(terminal) in message
        assert 'gm' in message

    def test_terminal_suffix_dispatch_unchanged(self):
        """':d'/':g'/':s'/':b' still dispatch to the terminal-current
        probe -- unchanged by the device-parameter suffix set being
        added alongside it (owner decision 2026-09-28, point 4: the
        terminal-current spelling itself, its emitted-path prefix, is
        untouched; a designer-name terminal spelling is a filed
        follow-up, not part of this change)."""
        dut, t = _single_bench()
        dut.save_device_currents = True
        dut.terminal_current_method = 'ammeter_wrapper'
        t.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
        sig = t.probe('xdut.m1:d')
        assert sig.name.startswith('v.')
        assert sig.name.endswith('.vamm_d')

    def test_case_insensitive_suffix(self):
        dut, t = _single_bench()
        t.add_sim_type(analysis.Op())
        upper = t.probe('M1:GM')
        lower = t.probe('M1:gm')
        assert upper.name == lower.name == 'M1:gm'


# ---------------------------------------------------------------------------
# Real ngspice.
# ---------------------------------------------------------------------------
@pytest.mark.slow
@needs_pdk
@needs_ngspice
@pytest.mark.parametrize('ngspice_bin', _ngspice_bins(),
                         ids=lambda b: 'ngspice47' if b else 'ngspice42')
class TestRealNgspice:
    def _tb(self, monkeypatch, ngspice_bin, **kw):
        if ngspice_bin:
            monkeypatch.setenv('SILK_NGSPICE_BIN', ngspice_bin)
        else:
            monkeypatch.delenv('SILK_NGSPICE_BIN', raising=False)
        return _single_bench(**kw)

    def test_op_probe_matches_op_table_cell(self, monkeypatch, ngspice_bin):
        """AC #3: an op probe equals its op_table() cell exactly -- both
        read the same saved vector by construction."""
        dut, t = self._tb(monkeypatch, ngspice_bin)
        t.add_sim_type(analysis.Op())
        gm = t.probe('M1:gm')
        dut.save_device_params = True
        report = dut.run_test('probechk', corner='tt')
        row = report.op_table()[0]
        assert float(gm.y[0]) == pytest.approx(row.gm, rel=1e-9)

    def test_dc_sweep_gm_matches_op_value_at_same_bias(self, monkeypatch, ngspice_bin):
        """AC #3: a DC-sweep gm probe matches the op value at the same
        bias point."""
        op_dut, op_t = self._tb(monkeypatch, ngspice_bin)
        op_t.add_sim_type(analysis.Op())
        op_gm = op_t.probe('M1:gm')
        op_dut.run_test('probechk', corner='tt')
        op_value = float(op_gm.y[0])

        dut, t = self._tb(monkeypatch, ngspice_bin)
        t.add_sim_type(analysis.DC(source='Vvg', start=0.0, stop=1.8, step=0.05))
        gm = t.probe('M1:gm')
        vg_axis = t.probe('vg')
        dut.run_test('probechk', corner='tt')
        idx = min(range(len(vg_axis.y)), key=lambda i: abs(vg_axis.y[i] - 1.0))
        assert float(gm.y[idx]) == pytest.approx(op_value, rel=2e-3)

    def test_gm_gds_arithmetic_and_spec_pass_fail(self, monkeypatch, ngspice_bin):
        """AC #3: gm/gds arithmetic and a Spec on a device probe pass and
        fail correctly, both polarities."""
        for devname, vg, vd in (('nmos1p8', 1.0, 1.0), ('pmos1p8', -1.0, -1.0)):
            dut, t = self._tb(monkeypatch, ngspice_bin, devname=devname, vg=vg, vd=vd)
            t.add_sim_type(analysis.Op())
            gm = t.probe('M1:gm')
            gds = t.probe('M1:gds')
            gain = gm / gds

            def _gain(ckt, gain=gain):
                return float(gain.y[0])

            t.add_metric('intrinsic_gain', _gain)
            t.add_spec('gain passes', 'intrinsic_gain', '>', 1.0)
            t.add_spec('gain fails', 'intrinsic_gain', '>', 1e12)
            report = t.run('tt')
            assert report.metrics['intrinsic_gain'] > 1.0
            by_name = {r.name: r for r in report.results}
            assert by_name['gain passes'].passed
            assert not by_name['gain fails'].passed

    def test_ac_shape_real_values_same_xlabel_as_other_probes(self, monkeypatch, ngspice_bin):
        """MISSING 3: ngspice writes an '@' AC vector as complex with zero
        imaginary part -- the Signal returns real values, constant across
        frequency, with the same xlabel as an ordinary AC probe."""
        dut, t = self._tb(monkeypatch, ngspice_bin)
        t.add_sim_type(analysis.AC(variation='dec', points=10, fstart=1e3, fstop=1e9))
        gm = t.probe('M1:gm')
        vd = t.probe('vd')
        dut.run_test('probechk', corner='tt')
        assert not any(isinstance(v, complex) for v in gm.y)
        assert all(math.isclose(float(v), float(gm.y[0])) for v in gm.y)
        assert gm.xlabel == vd.xlabel

    def test_joint_node_i_and_device_probe_one_run(self, monkeypatch, ngspice_bin):
        """MISSING 1: a node probe + an i() probe + a device probe together
        in ONE run -- under default save mode AND set_save_list -- and the
        main analysis's other vectors stay byte-identical with/without the
        device probe."""
        for use_save_list in (False, True):
            baseline_dut, baseline_t = self._tb(monkeypatch, ngspice_bin)
            baseline_t.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
            baseline_vd = baseline_t.probe('vd')
            baseline_ivd = baseline_t.probe('i(vd)')
            if use_save_list:
                baseline_dut.set_save_list(['d'])
            baseline_dut.run_test('probechk', corner='tt')

            dut, t = self._tb(monkeypatch, ngspice_bin)
            t.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
            vd = t.probe('vd')
            ivd = t.probe('i(vd)')
            gm = t.probe('M1:gm')
            if use_save_list:
                dut.set_save_list(['d'])
            dut.run_test('probechk', corner='tt')

            assert list(vd.y) == pytest.approx(list(baseline_vd.y))
            assert list(ivd.y) == pytest.approx(list(baseline_ivd.y))
            assert len(gm.y) == len(vd.y)
            assert not any(math.isnan(v) for v in gm.y)

    def test_probe_created_after_run_raises_on_read(self, monkeypatch, ngspice_bin):
        """MISSING 2: a probe created AFTER a run raises on read, saying to
        re-run -- never a silent miss."""
        dut, t = self._tb(monkeypatch, ngspice_bin)
        t.add_sim_type(analysis.Op())
        t.probe('vd')
        dut.run_test('probechk', corner='tt')
        late = t.probe('M1:gm')
        with pytest.raises(SimulationError, match='re-run') as excinfo:
            late.y
        assert "probe('M1:gm')" in str(excinfo.value)

    def test_probes_without_device_probes_deck_unaffected_by_real_run(self, monkeypatch, ngspice_bin):
        """AC #4, exercised against a real run: existing probe spellings
        (node, i()) are unaffected when no device probe is registered."""
        dut, t = self._tb(monkeypatch, ngspice_bin)
        t.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
        vd = t.probe('vd')
        ivd = t.probe('i(vd)')
        dut.run_test('probechk', corner='tt')
        assert len(vd.y) > 0
        assert len(ivd.y) > 0

    def test_undeclared_suffix_refused_before_any_simulator_is_invoked(
            self, monkeypatch, ngspice_bin):
        """Reviewer gate finding on 85cd1657 (2026-09-28), fix (A),
        carried over into the ':' spelling: probe() itself refuses an
        unrecognised suffix immediately, with a difflib hint, before any
        deck is generated or any simulator invoked. Build-time finding
        that motivated the override: an unrecognised '@' name saved
        alongside a genuinely readable one can come back a real,
        populated, all-zero vector with NO checkvalid warning at all on
        some ngspice-42/47 runs, and aborts the whole write on others,
        depending on what else is saved -- neither failure mode is
        safely deferrable to read time."""
        dut, t = self._tb(monkeypatch, ngspice_bin)
        t.add_sim_type(analysis.Op())
        t.probe('M1:gm')                       # unaffected by the check
        with pytest.raises(ValueError, match='gm') as excinfo:
            t.probe('M1:gmm')                  # typo of 'gm' -- refused NOW
        assert "probe('M1:gmm')" in str(excinfo.value)
        # No deck was ever generated, let alone a simulator invoked --
        # confirmed by never having called run_test()/dut.simulate() here.

    def test_terminal_current_probe_still_works(self, monkeypatch, ngspice_bin):
        """Owner decision 2026-09-28, point (4): the ':' spelling's
        terminal branch is bit-for-bit the pre-330.8 mechanism -- a real
        run confirms it still reads a genuine current, not just that
        dispatch reaches the right code path (TestColonDispatch covers
        dispatch alone, without a simulator)."""
        dut, t = self._tb(monkeypatch, ngspice_bin)
        dut.save_device_currents = True
        dut.terminal_current_method = 'ammeter_wrapper'
        t.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
        id_current = t.probe('xdut.m1:d')
        dut.run_test('probechk', corner='tt')
        assert len(id_current.y) > 0
        assert not any(math.isnan(v) for v in id_current.y)


@pytest.mark.slow
@needs_pdk
@needs_ngspice
def test_cgso_ngspice42_raises_ngspice47_reads_a_value(monkeypatch):
    """AC #6 + reviewer ruling 2026-09-28 on the fix (B) preflight
    mechanism: the SAME probe on ngspice 42 and 47. cgso is a recognised
    DEVICE_OP_PARAMS name that sky130 does NOT declare in its op_params
    (module docstring's 'undeclared -> preflight' set), so it goes
    through the ngspice-only pre-run availability check (fix B)
    alongside gm, which IS declared and needs no check. On 42 the
    preflight determines cgso is unavailable BEFORE the run: the main
    run completes cleanly -- no abort, no silent zero, exactly the
    failure mode fix (B) exists to prevent -- gm reads normally, and
    cgso raises a diagnostic SimulationError at read time citing the
    preflight log. On 47 the preflight determines cgso IS available,
    its save line goes through unmodified, and both read real values.
    Skips outright if ngspice-47 is not installed."""
    if not (os.path.isfile(_NGSPICE_47) and os.access(_NGSPICE_47, os.X_OK)):
        pytest.skip('needs ngspice-47 at ~/ngspice/ngspice-47/bin/ngspice')

    monkeypatch.delenv('SILK_NGSPICE_BIN', raising=False)
    dut42, t42 = _single_bench()
    t42.add_sim_type(analysis.Op())
    gm42 = t42.probe('M1:gm')
    cgso42 = t42.probe('M1:cgso')
    dut42.run_test('probechk', corner='tt')            # completes cleanly
    assert float(gm42.y[0]) > 0.0
    with pytest.raises(SimulationError, match='cgso'):
        cgso42.y
    _assert_probe_message(cgso42, 'M1', 'cgso', 'op')

    monkeypatch.setenv('SILK_NGSPICE_BIN', _NGSPICE_47)
    dut47, t47 = _single_bench()
    t47.add_sim_type(analysis.Op())
    gm47 = t47.probe('M1:gm')
    cgso47 = t47.probe('M1:cgso')
    dut47.run_test('probechk', corner='tt')
    assert float(gm47.y[0]) > 0.0
    assert float(cgso47.y[0]) > 0.0
    monkeypatch.delenv('SILK_NGSPICE_BIN', raising=False)


@pytest.mark.slow
@needs_pdk
@needs_ngspice
class TestPreflightMechanism:
    """Reviewer ruling 2026-09-28 on the fix (B) proposal: a guarded
    re-run was REJECTED -- the reviewer's own reproduction found ngspice-42
    reading a silent, unwarned all-zero vector for cgso at the 10-/11-others
    save-list-size threshold, which a re-run cannot catch because there is
    no abort to retry. This mandated a pre-run availability check instead,
    scoped to params NOT in the device's declared op_params (the declared
    ones are trusted; see :func:`_device_op_params`) -- see
    ``PySpiceNetlister._ngspice_param_preflight`` and its cache."""

    def setup_method(self):
        from silk.circuit.spice_netlister import PySpiceNetlister
        PySpiceNetlister._PREFLIGHT_CACHE.clear()

    def test_many_others_never_returns_silent_zero(self, monkeypatch):
        """The reviewer's own reproduction: on ngspice-42, cgso alongside
        10 or 11 OTHER declared params used to come back [0. 0.] with NO
        checkvalid warning at all -- a save-list-size threshold, not a
        principled emission difference. The preflight decides cgso's
        availability from a query run ALONE, before it can ever be added
        to any save list regardless of what else is requested -- so it
        must raise cleanly here too, and the other 11 probes must read
        real, non-NaN values undisturbed."""
        monkeypatch.delenv('SILK_NGSPICE_BIN', raising=False)
        dut, t = _single_bench()
        t.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
        others = ['capbd', 'capbs', 'cgd', 'cgg', 'cgs', 'gds', 'gm',
                  'gmbs', 'id', 'vbs', 'vds']
        probes = {name: t.probe(f'M1:{name}') for name in others}
        cgso = t.probe('M1:cgso')
        dut.run_test('probechk', corner='tt')
        for name, p in probes.items():
            assert len(p.y) > 0
            assert not any(math.isnan(v) for v in p.y)
        with pytest.raises(SimulationError, match='cgso'):
            cgso.y
        _assert_probe_message(cgso, 'M1', 'cgso', 'tran')

    def test_preflight_runs_once_per_combination(self, monkeypatch):
        """Reviewer requirement: 'count the launches' -- the preflight is
        cached in-process by (binary, model, param). Three separate
        benches probing the SAME combination cost exactly ONE extra
        ngspice launch total, never per-probe or per-run."""
        import subprocess
        monkeypatch.delenv('SILK_NGSPICE_BIN', raising=False)
        launches = {'n': 0}
        orig_run = subprocess.run

        def counting_run(*a, **kw):
            if a and isinstance(a[0], list) and any(
                    'preflight' in str(x) for x in a[0]):
                launches['n'] += 1
            return orig_run(*a, **kw)

        monkeypatch.setattr(subprocess, 'run', counting_run)
        for _ in range(3):
            dut, t = _single_bench()
            t.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
            cgso = t.probe('M1:cgso')
            dut.run_test('probechk', corner='tt')
            with pytest.raises(SimulationError, match='cgso'):
                cgso.y
        assert launches['n'] == 1

    def test_declared_only_bench_launches_no_preflight(self, monkeypatch):
        """Reviewer requirement: a bench with only declared params (the
        module docstring's 'trusted, no preflight needed' set, e.g. gm
        and vth) launches NO preflight at all -- the emission path is
        exactly what it was before this mechanism existed (AC #4's
        byte-identical-deck guarantee), now confirmed against a real
        run rather than emission alone."""
        import subprocess
        monkeypatch.delenv('SILK_NGSPICE_BIN', raising=False)
        launches = {'n': 0}
        orig_run = subprocess.run

        def counting_run(*a, **kw):
            if a and isinstance(a[0], list) and any(
                    'preflight' in str(x) for x in a[0]):
                launches['n'] += 1
            return orig_run(*a, **kw)

        monkeypatch.setattr(subprocess, 'run', counting_run)
        dut, t = _single_bench()
        t.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
        gm = t.probe('M1:gm')
        vth = t.probe('M1:vth')
        dut.run_test('probechk', corner='tt')
        assert len(gm.y) > 0
        assert len(vth.y) > 0
        assert launches['n'] == 0

    def test_pmos_preflight_polarity(self, monkeypatch):
        """The preflight deck must bias the probed device at ITS OWN
        polarity (PMOS gets a negative bias) -- confirmed against a real
        PMOS run, not just the NMOS case the other tests exercise."""
        monkeypatch.delenv('SILK_NGSPICE_BIN', raising=False)
        dut, t = _single_bench(devname='pmos1p8', vg=-1.0, vd=-1.0)
        t.add_sim_type(analysis.Op())
        gm = t.probe('M1:gm')
        cgso = t.probe('M1:cgso')
        dut.run_test('probechk', corner='tt')
        assert float(gm.y[0]) > 0.0
        with pytest.raises(SimulationError, match='cgso'):
            cgso.y
        _assert_probe_message(cgso, 'M1', 'cgso', 'op')


# ---------------------------------------------------------------------------
# Real Xyce.
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.skipif(not _have_xyce, reason='Xyce binary or sky130 PDK not present')
class TestRealXyce:
    def _tb(self, monkeypatch, **kw):
        monkeypatch.setenv('SILK_XYCE_BIN', _XYCE_BIN)
        dut, t = _single_bench(**kw)
        dut.simulator = 'xyce'
        return dut, t

    def test_op_probe_matches_op_table_cell(self, monkeypatch):
        dut, t = self._tb(monkeypatch)
        t.add_sim_type(analysis.Op())
        gm = t.probe('M1:gm')
        dut.save_device_params = True
        report = dut.run_test('probechk', corner='tt')
        row = report.op_table()[0]
        assert float(gm.y[0]) == pytest.approx(row.gm, rel=1e-6)

    def test_dc_sweep_gm_matches_op_value(self, monkeypatch):
        op_dut, op_t = self._tb(monkeypatch)
        op_t.add_sim_type(analysis.Op())
        op_gm = op_t.probe('M1:gm')
        op_dut.run_test('probechk', corner='tt')
        op_value = float(op_gm.y[0])

        dut, t = self._tb(monkeypatch)
        t.add_sim_type(analysis.DC(source='Vvg', start=0.0, stop=1.8, step=0.05))
        gm = t.probe('M1:gm')
        vg_axis = t.probe('vg')
        dut.run_test('probechk', corner='tt')
        idx = min(range(len(vg_axis.y)), key=lambda i: abs(vg_axis.y[i] - 1.0))
        assert float(gm.y[idx]) == pytest.approx(op_value, rel=2e-3)

    def test_pmos_gm_id_arithmetic_and_spec(self, monkeypatch):
        dut, t = self._tb(monkeypatch, devname='pmos1p8', vg=-1.0, vd=-1.0)
        t.add_sim_type(analysis.Op())
        gm = t.probe('M1:gm')
        idd = t.probe('M1:id')
        gm_id = gm / idd

        def _gm_id(ckt, gm_id=gm_id):
            return float(gm_id.y[0])

        t.add_metric('gm_id', _gm_id)
        t.add_spec('gm_id positive', 'gm_id', '>', 0.0)
        report = t.run('tt')
        assert report.metrics['gm_id'] > 0.0
        assert next(r for r in report.results if r.name == 'gm_id positive').passed

    def test_pre_run_refusal_unsupported_param(self, monkeypatch):
        """AC #6: covers the Xyce pre-run refusal explicitly."""
        dut, t = self._tb(monkeypatch)
        t.add_sim_type(analysis.Op())
        t.probe('M1:gmbs')                     # not in XYCE_OP_CAPABILITY
        with pytest.raises(ValueError, match='XYCE_OP_CAPABILITY|N\\(\\)/ID\\(\\)'):
            _prepared_deck(t)


@pytest.mark.slow
@needs_ngspice
@needs_pdk
@pytest.mark.parametrize('save_params', [False, True])
def test_retained_corner_uses_full_device_reader_and_original_declarations(save_params):
    import numpy as np
    dut, tb = _single_bench()
    tb.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
    dut.save_device_params = save_params
    probe = tb.probe('M1:gm')
    reports = tb.run_corners(['tt', 'ss'])
    expected = {}
    for key, report in reports.items():
        expected[key] = report.sim_results.load_signal('M1:gm').y.copy()
        np.testing.assert_array_equal(probe.corner(key).y, expected[key])
        assert report.sim_results.save_device_params == save_params
    assert not np.array_equal(expected['tt'], expected['ss'])
    tb.netlist.sim_analysis_dict.clear()
    tb.netlist.sim_analysis_dict['ac'] = analysis.AC('dec', 2, 1, 10)
    tb.netlist.save_device_params = not save_params
    tb.netlist._pending_device_probes = set()
    for key, report in reports.items():
        np.testing.assert_array_equal(report.sim_results.load_signal('M1:gm').y, expected[key])
        np.testing.assert_array_equal(probe.corner(key).y, expected[key])


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_retained_preflight_reason_survives_live_declaration_changes(monkeypatch):
    from silk.circuit.modules_results import PreflightResult
    from silk.circuit.spice_netlister import PySpiceNetlister
    refusal = PreflightResult(False, 'probe-control', 'frozen-preflight.log', 'unavailable')
    monkeypatch.setattr(PySpiceNetlister, '_ngspice_param_preflight',
                        lambda *args, **kwargs: refusal)
    dut, tb = _single_bench()
    tb.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
    tb.probe('M1:cgso')
    tb.add_metric('unavailable', lambda module: module.load_signal('M1:cgso').y[-1])
    reports = tb.run_corners(['tt', 'ss'], n_workers=1)
    tb.netlist.netlister._preflight_unavailable_params = {}
    tb.netlist._pending_device_probes = set()
    for key, report in reports.items():
        assert report.metric_errors['unavailable']
        assert report.sim_results._raw._preflight_unavailable_params
        with pytest.raises(SimulationError, match='cgso') as error:
            report.sim_results.load_signal('M1:cgso')
        assert 'AFTER' not in str(error.value)
        assert 'preflight log: frozen-preflight.log' in str(error.value)
        with pytest.raises(SimulationError, match='cgso'):
            tb.probe('M1:cgso').corner(key)


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_retained_reader_methods_use_own_source_currents():
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    from silk.circuit.modules_results import _trapezoid_window
    from silk.simulation.generation import read_retained
    dut, tb = _single_bench()
    dut.save_device_currents = True
    tb.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
    reports = tb.run_corners(['tt', 'ss'])
    currents = []
    for report in reports.values():
        reader = report.sim_results
        expected_raw = tb.netlist.backend.load_raw(report.rawfile_path)
        expected = tb.netlist.backend.load_signal(expected_raw, 'vvd', type='I')
        actual = reader.load_signal('vvd', 'I')
        np.testing.assert_array_equal(actual.y, expected.y)
        currents.append(actual.y[-1])
        assert reader.get_signal_names(raw=True) == [v.name for v in expected_raw.native.vectors]
        np.testing.assert_array_equal(
            reader.backend.load_signal(type(expected_raw)(read_retained(reader), report.rawfile_path),
                                       'vvd', type='I').y, expected.y)
        assert Path(reader.view(open=False, inline=True)).is_file()
        figure = reader.plot('vvd:I', show=False)
        np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(),
                                   reader._auto_scale(expected.y, 'A')[0])
        plt.close(figure)
        annotations = reader.annotate_currents(force=True)
        assert annotations.results_file == report.rawfile_path
        pin = next(pin for pin in reader.net_dict['vd'].pins if pin.instance == 'vd')
        assert abs(pin.avg_current) == pytest.approx(abs(_trapezoid_window(expected.x, expected.y, None)[0]))
        assert all(v._data is None for p in reader.simResultsAll.plots for v in p.vectors)
    assert currents[0] != currents[1]


# ---------------------------------------------------------------------------
# evaluate(rawfile=...) in a process that never ran or prepared the test.
# ---------------------------------------------------------------------------

def _external_raw_for(tb, corner='tt'):
    """Prepare *tb*, simulate its deck outside silk, return the raw path."""
    import subprocess
    from pathlib import Path
    from silk.simulation.runner import ngspice_executable
    deck = Path(tb.prepare(corner=corner))
    files = tb.netlist._run_files
    completed = subprocess.run(
        [ngspice_executable(), '-b', '-r', files.rawfile_path,
         '-o', files.logfile_path, str(deck)],
        cwd=files.directory, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return files.rawfile_path


def _gm_final(signal):
    return float(signal.y[-1])


@pytest.mark.slow
@needs_ngspice
@needs_pdk
class TestEvaluateRawfileDeviceProbes:
    VECTOR = '@m.xdut.xmm1.msky130_fd_pr__nfet_01v8[gm]'

    def _bench(self, with_metric=True):
        dut, tb = _single_bench()
        tb.add_sim_type(analysis.Op())
        gm = tb.probe('M1:gm')
        if with_metric:
            tb.add_metric('gm_metric', lambda sig: float(sig.y[-1]), sig=gm)
        return dut, tb, gm

    def _external(self):
        _, producer, _ = self._bench()
        raw = _external_raw_for(producer)
        return raw

    def test_fresh_bench_reads_declared_probe_and_metric(self):
        raw = self._external()
        _, tb, gm = self._bench()
        report = tb.evaluate(rawfile=raw, corner='tt')
        expected = float(tb.netlist.load_signal(self.VECTOR.lower()).y[-1])
        assert expected > 0
        assert _gm_final(gm) == pytest.approx(expected, rel=1e-12)
        assert report.metrics['gm_metric'] == pytest.approx(expected, rel=1e-12)
        assert not report.metric_errors

    def test_probe_created_after_evaluate_still_refused(self):
        raw = self._external()
        _, tb, _ = self._bench()
        tb.evaluate(rawfile=raw, corner='tt')
        late = tb.probe('M1:gds')
        with pytest.raises(SimulationError, match='AFTER'):
            late.y
        tb.evaluate()
        with pytest.raises(SimulationError, match='AFTER'):
            late.y

    def test_explicit_reevaluate_admits_probe_present_in_file(self):
        # Producer declares both parameters; the consumer declares gm first.
        _, producer, _ = self._bench()
        producer.probe('M1:gds')
        raw = _external_raw_for(producer)
        _, tb, _ = self._bench()
        tb.evaluate(rawfile=raw, corner='tt')
        late = tb.probe('M1:gds')
        with pytest.raises(SimulationError, match='AFTER'):
            late.y
        tb.evaluate(rawfile=raw, corner='tt')
        assert float(late.y[-1]) > 0

    def test_declared_parameter_missing_from_file_is_availability_error(self):
        raw = self._external()
        _, tb, _ = self._bench(with_metric=False)
        missing = tb.probe('M1:gds')
        tb.evaluate(rawfile=raw, corner='tt')
        with pytest.raises(SimulationError) as excinfo:
            missing.y
        assert 'AFTER' not in str(excinfo.value)

    def test_invalid_declaration_keeps_previous_snapshot(self):
        raw = self._external()
        _, tb, gm = self._bench()
        tb.evaluate(rawfile=raw, corner='tt')
        before = list(tb.netlist._pending_device_probes)
        tb.probe('M1:gds')
        with pytest.raises(Exception):
            tb.evaluate(rawfile=raw, corner='nonexistent-corner-xyz',
                        temperature='not-a-temperature')
        assert list(tb.netlist._pending_device_probes) == before
        assert _gm_final(gm) > 0


# ---------------------------------------------------------------------------
# Simulator-free evaluate(rawfile=...): raw data written with write_rawfile.
# ---------------------------------------------------------------------------

def _op_raw(path, gm, vector):
    import numpy as np
    from silk.signals.rawfile import RawPlot, RawVector, write_rawfile
    header = {'title': 't', 'date': 'd', 'plotname': 'Operating Point',
              'flags': 'real'}
    write_rawfile(path, RawPlot(
        header,
        [RawVector('v(d)', 'voltage', (), np.array([1.0])),
         RawVector(vector, 'conductance', (), np.array([gm]))],
        1, True, None))


_NG_GM = '@m.xdut.xmm1.msky130_fd_pr__nfet_01v8[gm]'
_XY_GM = 'N(XDUT:XMM1:gm)'


@needs_pdk
@pytest.mark.parametrize('simulator,vector', [
    ('ngspice', _NG_GM), ('xyce', _XY_GM)], ids=['ngspice', 'xyce'])
class TestEvaluateRawfileSimulatorFree:
    def _bench(self, simulator, metric=True):
        dut, tb = _single_bench()
        dut.simulator = simulator
        tb.netlist.simulator = simulator
        tb.add_sim_type(analysis.Op())
        gm = tb.probe('M1:gm')
        if metric:
            tb.add_metric('gm_metric', lambda sig: float(sig.y[-1]), sig=gm)
        return tb, gm

    def test_declared_probe_and_metric_read_native_vector(
            self, tmp_path, simulator, vector):
        raw = str(tmp_path / 'ext.raw')
        _op_raw(raw, 2.5e-4, vector)
        tb, gm = self._bench(simulator)
        report = tb.evaluate(rawfile=raw, corner='tt')
        assert float(gm.y[-1]) == pytest.approx(2.5e-4)
        assert report.metrics['gm_metric'] == pytest.approx(2.5e-4)
        assert not report.metric_errors

    def test_missing_vector_is_availability_not_lifecycle(
            self, tmp_path, simulator, vector):
        raw = str(tmp_path / 'ext.raw')
        _op_raw(raw, 2.5e-4, vector.replace('gm', 'gds'))
        tb, gm = self._bench(simulator, metric=False)
        tb.evaluate(rawfile=raw, corner='tt')
        with pytest.raises(SimulationError) as excinfo:
            gm.y
        assert 'AFTER' not in str(excinfo.value)

    def test_late_probe_refused_until_explicit_reevaluate(
            self, tmp_path, simulator, vector):
        raw = str(tmp_path / 'ext.raw')
        _op_raw(raw, 2.5e-4, vector.replace('gm', 'gds'))
        tb, gm = self._bench(simulator, metric=False)
        tb.evaluate(rawfile=raw, corner='tt')
        late = tb.probe('M1:gds')
        with pytest.raises(SimulationError, match='AFTER'):
            late.y
        tb.evaluate(rawfile=raw, corner='tt')
        assert float(late.y[-1]) == pytest.approx(2.5e-4)


@needs_pdk
def test_selected_placements_read_their_own_external_files(tmp_path):
    dut, tb = _single_bench()
    tb.add_sim_type(analysis.Op())
    top = Module(None, str(tmp_path / 'top'), 'top.cir', None)
    top.add_module(dut, 'one', d='a', g='ga', s='0', b='0')
    top.add_module(dut, 'two', d='b', g='gb', s='0', b='0')
    one = top.get_test('probechk', instance='one')
    two = top.get_test('probechk', instance='two')
    gm_one = one.probe('M1:gm')
    gm_two = two.probe('M1:gm')
    raw_one, raw_two = str(tmp_path / 'one.raw'), str(tmp_path / 'two.raw')
    vec = lambda inst: f'@m.x{inst}.xdut.xmm1.msky130_fd_pr__nfet_01v8[gm]'
    _op_raw(raw_one, 1.0e-4, vec('one'))
    _op_raw(raw_two, 3.0e-4, vec('two'))
    one.evaluate(rawfile=raw_one, corner='tt')
    two.evaluate(rawfile=raw_two, corner='tt')
    assert float(gm_one.y[-1]) == pytest.approx(1.0e-4)
    assert float(gm_two.y[-1]) == pytest.approx(3.0e-4)


# ---------------------------------------------------------------------------
# TASK-820: tb.device_probe_address('M1:gm') -- the exact raw vector address
# and designer label, with no simulation. The prepared deck is the oracle.
# ---------------------------------------------------------------------------
import re
import shutil
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _deck_device_addresses(deck, backend):
    """Every device-parameter vector the deck asks the simulator to save."""
    if backend == 'xyce':
        printed = [ln for ln in deck.splitlines()
                   if ln.lower().startswith('.print')]
        return re.findall(r'(?:\bN|\bID)\([^)]*\)', ' '.join(printed))
    return re.findall(r'"(@m[^"]*)"', '\n'.join(
        ln for ln in deck.splitlines() if ln.lstrip().startswith('save ')))


def _addr_bench(backend='ngspice', analysis_obj=None):
    dut, t = _single_bench()
    dut.simulator = backend
    t.add_sim_type(analysis_obj or analysis.Op())
    return dut, t


class TestDeviceProbeAddress:
    @pytest.mark.parametrize('backend', ['ngspice', 'xyce'])
    @pytest.mark.parametrize('corner', ['tt', 'ss'])
    def test_address_equals_deck_save_argument(self, backend, corner):
        dut, t = _addr_bench(backend)
        before = t.device_probe_address('M1:gm')
        assert before['label'] == 'dut.M1:gm'
        assert set(before) == {'address', 'label'}
        t.probe('M1:gm')
        deck_path = t.prepare(corner=corner)
        deck = Path(deck_path).read_text()
        assert _deck_device_addresses(deck, backend) == [before['address']]
        assert t.device_probe_address('M1:gm') == before
        if backend == 'ngspice':
            assert before['address'] == (
                '@m.xdut.XMM1.msky130_fd_pr__nfet_01v8[gm]')
        else:
            assert before['address'] == 'N(XDUT:XMM1:MSKY130_FD_PR__NFET_01V8:GM)'

    @pytest.mark.parametrize('backend', ['ngspice', 'xyce'])
    def test_id_and_parameter_normalisation(self, backend):
        dut, t = _addr_bench(backend)
        got = t.device_probe_address('M1:GM')
        assert got['label'] == 'dut.M1:gm'
        t.probe('M1:gm')
        t.probe('M1:id')
        deck = _prepared_deck(t)
        ident = t.device_probe_address('M1:id')
        assert ident['label'] == 'dut.M1:id'
        # sky130 declares gm and id: both are saved, and the deck is the oracle
        assert sorted(_deck_device_addresses(deck, backend)) == sorted(
            [got['address'], ident['address']])

    def test_nested_designer_path_keeps_case_and_label(self):
        stage, _ = _single_bench()
        dut = Module('sky130', scratch_dir(), 'outer.cir', pdk_details=_sky130(),
                     module_name='outer')
        dut.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN,
                                 's': PortDir.GROUND, 'b': PortDir.GROUND})
        mapping = {'d': 'd', 'g': 'g', 's': 's', 'b': 'b'}
        dut.add_module(stage, 'Core', nets=mapping)
        t = dut.create_test('nested')
        t.add_module_under_test(mapping)
        t.add_sim_type(analysis.Op())
        got = t.device_probe_address('Core.M1:vth')
        assert got['label'] == 'dut.Core.M1:vth'
        t.probe('Core.M1:vth')
        deck = _prepared_deck(t)
        assert _deck_device_addresses(deck, 'ngspice') == [got['address']]

    def test_contract_wording_in_docstring_and_reference(self):
        from silk.circuit.module_test import ModuleTestbench
        sentence = ('Noise-only tests and ammeter-wrapper mode are refused by '
                    'prepare()/run(). prepare() also refuses parameters '
                    'needing auxiliary ngspice preflight; run() performs '
                    'that preflight.')
        doc = ' '.join(ModuleTestbench.device_probe_address.__doc__.split())
        ref = ' '.join((_ROOT / 'docs/api-reference.md').read_text().split())
        ref = ref.replace('`prepare()`', 'prepare()').replace('`run()`', 'run()')
        assert sentence in ref
        assert sentence in doc
        assert 'neither validates capture support' in doc

    def test_lookup_before_any_probe_and_corner_independent(self):
        dut, t = _addr_bench()
        a = t.device_probe_address('M1:gm')
        assert t.device_probe_address('M1:gm') == a

    @pytest.mark.parametrize('path, exc, words', [
        ('N1:gm', ValueError, ("N1", "gm")),
        ('M1:foo', ValueError, ("foo",)),
        ('M1:gm_id', ValueError, ("gm_id",)),
        ('M1:d', ValueError, ("M1:d",)),
        ('M1', ValueError, ("M1",)),
        ('v(M1:gm)', ValueError, ("v(M1:gm)",)),
        ('', ValueError, ()),
        (5, TypeError, ()),
        (None, TypeError, ()),
    ])
    def test_bad_input_and_resolver_errors(self, path, exc, words):
        dut, t = _addr_bench()
        with pytest.raises(exc) as info:
            t.device_probe_address(path)
        for word in words:
            assert word in str(info.value)

    def test_error_matches_prepare_error_for_unknown_device(self):
        dut, t = _addr_bench()
        t.probe('N1:gm')
        with pytest.raises(ValueError) as from_prepare:
            _prepared_deck(t)
        with pytest.raises(ValueError) as from_lookup:
            t.device_probe_address('N1:gm')
        assert str(from_lookup.value) == str(from_prepare.value)

    def test_non_mos_device_refused(self):
        dut = Module('sky130', scratch_dir(), 'dut.cir',
                     pdk_details=_sky130(), module_name='dut')
        dut.define_module_ports({'a': PortDir.INOUT, 'b': PortDir.INOUT})
        with dut:
            elements.resistor('r1', p='a', n='b', value=1000)
        t = dut.create_test('probechk')
        t.add_module_under_test({'a': 'va', 'b': 'vb'})
        with pytest.raises(ValueError, match='not a MOS-shaped device') as info:
            t.device_probe_address('r1:gm')
        assert 'r1' in str(info.value) and 'gm' in str(info.value)

    def test_device_outside_the_dut_refused(self):
        stage, _ = _single_bench()
        dut = Module('sky130', scratch_dir(), 'outer.cir', pdk_details=_sky130(),
                     module_name='outer')
        dut.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN,
                                 's': PortDir.GROUND, 'b': PortDir.GROUND})
        mapping = {'d': 'd', 'g': 'g', 's': 's', 'b': 'b'}
        t = dut.create_test('outside')
        t.add_module_under_test(mapping)
        t.add_module(stage, 'helper', nets=mapping)      # a bench helper, not the DUT
        with pytest.raises(ValueError, match='helper.M1'):
            t.device_probe_address('helper.M1:gm')

    def test_lookup_has_no_side_effects(self, monkeypatch):
        dut, t = _addr_bench()
        t.probe('M1:vth')
        registry = list(t._probe_registry)
        pending = getattr(t.netlist, '_pending_device_probes', None)

        def boom(*a, **k):
            raise AssertionError('the lookup must not launch or prepare anything')
        monkeypatch.setattr(subprocess, 'Popen', boom)
        monkeypatch.setattr(subprocess, 'run', boom)
        monkeypatch.setattr(type(t), 'prepare', boom)
        monkeypatch.setattr(type(t), 'run', boom)
        monkeypatch.setattr(type(t), 'evaluate', boom)
        t.device_probe_address('M1:gm')
        assert t._probe_registry == registry
        assert len(t._probe_registry) == 1
        assert getattr(t.netlist, '_pending_device_probes', None) == pending
        assert t.results_file is None

    def test_prepare_still_refuses_what_the_lookup_does_not_check(self):
        # Controls: the lookup is not a capture-support check.
        dut, t = _addr_bench(analysis_obj=analysis.Noise(
            output='vd', ref=None, source='vg', variation='dec', points=10,
            fstart=1e3, fstop=1e6))
        t.probe('M1:gm')
        assert t.device_probe_address('M1:gm')['label'] == 'dut.M1:gm'
        with pytest.raises(SimulationError, match='Noise'):
            t.prepare(corner='tt')
        dut, t = _single_bench()
        dut.save_device_currents = True
        dut.terminal_current_method = 'ammeter_wrapper'
        t.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
        t.probe('M1:gm')
        assert t.device_probe_address('M1:gm')['label'] == 'dut.M1:gm'
        with pytest.raises(SimulationError, match='ammeter_wrapper'):
            t.prepare(corner='tt')
        from silk.exceptions import NotSupportedError
        dut, t = _addr_bench()
        t.probe('M1:cgso')
        assert t.device_probe_address('M1:cgso')['label'] == 'dut.M1:cgso'
        with pytest.raises(NotSupportedError, match='cgso'):
            t.prepare(corner='tt')


# --- selector guarantee on real raw files ----------------------------------

def _raw_plot_names(raw_path):
    from silk.signals.rawfile import read_rawfile
    raw = read_rawfile(str(raw_path))
    return [[v.name for v in plot.vectors[1:]] for plot in raw.plots]


def _js_matches(names, selector):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for the JavaScript selector check')
    script = """
import {matchesSelector} from './silk/signals/view/assets/selectors.mjs';
const {names, selector} = JSON.parse(process.argv[1]);
console.log(JSON.stringify(names.map(n => matchesSelector(n, selector))));
"""
    out = subprocess.run(
        [node, '--input-type=module', '-e', script,
         json.dumps({'names': names, 'selector': selector})],
        cwd=_ROOT, check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def _strip_wrapper(name):
    m = re.fullmatch(r'(?:v|i)\((.*)\)', name, flags=re.IGNORECASE)
    return m.group(1) if m else name


def _assert_selects_the_row(raw_path, address):
    """The address selects exactly one native row in EVERY plot, in every case
    variant, with silk's Python matches_selector, select_for_payload and the
    JavaScript matchesSelector."""
    from silk.signals.view.select import matches_selector, select_for_payload
    plots = _raw_plot_names(raw_path)
    assert plots
    for names in plots:
        want = [i for i, n in enumerate(names)
                if _strip_wrapper(n).lower() == address.lower()]
        assert len(want) == 1, (address, names)
        for selector in (address, address.lower(), address.upper()):
            got = [i for i, n in enumerate(names) if matches_selector(n, selector)]
            assert got == want, (selector, names)
            js = [i for i, hit in enumerate(_js_matches(names, selector)) if hit]
            assert js == want, ('js', selector, names)
            mask = select_for_payload([names], signals=[selector])[0]
            assert [i for i, m in enumerate(mask) if m] == want, selector


@needs_ngspice
@needs_pdk
class TestAddressOnRealRawFiles:
    @pytest.mark.parametrize('kind', ['op', 'tran'])
    @pytest.mark.parametrize('param', ['gm', 'id'])
    def test_ngspice(self, kind, param):
        sim = (analysis.Op() if kind == 'op'
               else analysis.Transient(step=1e-9, stop=3e-9))
        dut, t = _addr_bench('ngspice', sim)
        sig = t.probe(f'M1:{param}')
        before = t.device_probe_address(f'M1:{param}')
        report = dut.run_test('probechk', corner='tt')
        sig.y                                             # the vector is readable
        assert t.device_probe_address(f'M1:{param}') == before
        _assert_selects_the_row(report.rawfile_path, before['address'])
        t.evaluate(rawfile=report.rawfile_path)
        assert t.device_probe_address(f'M1:{param}') == before


@needs_pdk
@pytest.mark.requires_xyce
@pytest.mark.skipif(not _have_xyce, reason='Xyce binary or sky130 PDK not present')
class TestAddressOnRealXyceRawFiles:
    @pytest.mark.parametrize('param', ['gm', 'id'])
    def test_xyce(self, monkeypatch, param):
        monkeypatch.setenv('SILK_XYCE_BIN', _XYCE_BIN)
        dut, t = _addr_bench('xyce')
        sig = t.probe(f'M1:{param}')
        before = t.device_probe_address(f'M1:{param}')
        report = dut.run_test('probechk', corner='tt')
        sig.y
        assert t.device_probe_address(f'M1:{param}') == before
        _assert_selects_the_row(report.rawfile_path, before['address'])
        t.evaluate(rawfile=report.rawfile_path)
        assert t.device_probe_address(f'M1:{param}') == before
