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
