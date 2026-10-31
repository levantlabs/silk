"""``Module.save_device_params`` (TASK-330.3) — MOS device operating-point
parameter capture on a dedicated op-point solve, mapped back through
``NgspiceBackend.device_op_vectors``.

Netlist-string tests (no simulator) live in ``TestEmissionNoSimulator``.
Real-ngspice coverage (``TestRealNgspice``, both ngspice 42 and 47 —
parametrized via ``$SILK_NGSPICE_BIN``) proves the reviewer's four stop-point
conditions (ruling 2): the fixed ``tran_plus_op_fixed.sp`` shape, no
ergonomics cost, ``op1`` content, and the ammeter-wrapper matcher. See
``tests/fixtures/op_point/README.md`` for the hand-written-deck probe this
implementation is built from.
"""
import os

import pytest

from silk.circuit.modules import Module
from silk.pdk.basepdk import BasePDK
from silk import sources
from silk.simulation import analysis
from silk.simulation.backend import SimulatorBackend, DEVICE_OP_PARAMS, RawResult
from silk.exceptions import SimulationError
from silk.signals.rawfile import read_rawfile

from tests._env import needs_pdk, needs_pdk_named, needs_ngspice, needs_osdi_runnable
from tests._env import scratch_dir

_PDK_YAML = os.path.join(os.path.dirname(__file__), '..', 'silk', 'pdk', 'data',
                         'skywater_130_device_details.yaml')
_FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures', 'op_point')


def _pdk(lib='/fake/sky130.lib.spice'):
    return BasePDK(_PDK_YAML, library_location=lib)


def _single_nmos_tb(lib='/fake/sky130.lib.spice'):
    tb = Module('PDK_skywater130', scratch_dir(), 'op_params_tb.cir',
                pdk_details=_pdk(lib), module_name=None)
    tb.add(_pdk(lib).nmos1p8('M1', d='vd', g='vg', s='0', b='0', w=1.0, l=0.15, m=1))
    tb.add_source(sources.VoltageSource('vd', positive='vd', negative='0', dc=1.0))
    tb.add_source(sources.VoltageSource('vg', positive='vg', negative='0', dc=1.0))
    return tb


def _gen(tb):
    deck_path = tb.generate_netlist(corner='tt')
    with open(deck_path) as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Netlist-string tests — no simulator.
# ---------------------------------------------------------------------------
class TestEmissionNoSimulator:
    def test_flag_off_is_unchanged(self):
        """Reviewer condition 4: switch off, no '@'/interactive-op emission
        at all -- the existing golden/emission suites already pin the exact
        byte content; this is a direct, local confirmation."""
        tb = _single_nmos_tb()
        tb.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
        deck = _gen(tb).lower()
        assert '@m.' not in deck
        assert '\nop\n' not in deck
        assert 'op1' not in deck

    def test_flag_on_adds_at_saves_and_interactive_op(self):
        tb = _single_nmos_tb()
        tb.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
        tb.save_device_params = True
        deck = _gen(tb).lower()
        assert '@m.xmm1.msky130_fd_pr__nfet_01v8[id]' in deck
        # cgso/cgdo/cgbo are DELIBERATELY undeclared on sky130 (TASK-330.4
        # gate finding: BSIM4 only fills these overlap caps after a .tran
        # has run in the same ngspice session -- Op/AC/DC report a silent
        # 0, which the arc forbids), so only 14 of the 17 recognised names
        # ever get a 'save @...' line here -- never all 17.
        undeclared = {'cgso', 'cgdo', 'cgbo'}
        for param in set(DEVICE_OP_PARAMS) - undeclared:
            assert f'[{param}]' in deck
        for param in undeclared:
            assert f'[{param}]' not in deck
        assert '\nop\n' in deck
        assert 'write ' in deck and 'tran1.all op1.all' in deck

    def test_flag_on_with_op_declared_suppresses_the_op_card(self):
        """No netlist-level '.op' card ever coexists with save_device_params
        -- not even when the user declares Op() themselves (TASK-330.3 probe,
        tran_plus_op_fixed.sp: a pre-existing '.op' card claims 'op1' during
        the first run and strands the '@' params in a phantom 'op2')."""
        tb = _single_nmos_tb()
        tb.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
        tb.add_sim_type(analysis.Op())
        tb.save_device_params = True
        deck = _gen(tb)
        lines = [l.strip().lower() for l in deck.splitlines()]
        assert '.op' not in lines            # no netlist-level card
        assert lines.count('op') == 1        # exactly one interactive op

    def test_flag_on_op_only_suppresses_the_op_card_too(self):
        tb = _single_nmos_tb()
        tb.add_sim_type(analysis.Op())
        tb.save_device_params = True
        deck = _gen(tb)
        lines = [l.strip().lower() for l in deck.splitlines()]
        assert '.op' not in lines
        assert lines.count('op') == 1
        # Op()-only stays a genuinely single-plot deck (op1 is the only
        # plot either way), so the bare `write <path>` form is kept -- no
        # plot names on the write line.
        write_lines = [l for l in deck.splitlines() if l.strip().lower().startswith('write ')]
        assert len(write_lines) == 1
        assert '.all' not in write_lines[0]

    def test_incompatible_with_save_device_currents_raises(self):
        tb = _single_nmos_tb()
        tb.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
        tb.save_device_params = True
        tb.save_device_currents = True
        with pytest.raises(SimulationError, match='save_device_params.*save_device_currents'):
            _gen(tb)

    def test_incompatible_with_ammeter_wrapper_method_too(self):
        """Any terminal_current_method -- savecurrents is only the default."""
        tb = _single_nmos_tb()
        tb.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
        tb.save_device_params = True
        tb.save_device_currents = True
        tb.terminal_current_method = 'ammeter_wrapper'
        with pytest.raises(SimulationError, match='save_device_params.*save_device_currents'):
            _gen(tb)

    def test_xyce_rejects_the_switch_on_a_non_op_analysis(self):
        """TASK-330.5, reviewer ruling (A): Xyce now supports
        save_device_params, but ONLY when Op() is the deck's one analysis --
        appending N()/ID() to a '.print tran' or a multi-point '.print dc'
        would record device columns at every solve point, which is
        forbidden regardless of backend. NotSupportedError, not the old
        blanket 'ngspice-only' SimulationError."""
        from silk.exceptions import NotSupportedError
        tb = _single_nmos_tb()
        tb.simulator = 'xyce'
        tb.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
        tb.save_device_params = True
        with pytest.raises(NotSupportedError, match='save_device_params on Xyce requires Op'):
            _gen(tb)

    def test_xyce_op_only_adds_n_id_print_terms(self):
        """TASK-330.5: Op()-only + save_device_params on Xyce emits
        additive N()/ID() '.print dc' terms for the 8-name capability set
        (XYCE_OP_CAPABILITY), never gmbs/the capacitances -- built from
        elem_dict via xyce_op_print_terms, same discipline as ngspice's
        device_op_save_lines (guess nothing; Xyce aborts the WHOLE run on
        one unresolvable .print symbol)."""
        tb = _single_nmos_tb()
        tb.simulator = 'xyce'
        tb.add_sim_type(analysis.Op())
        tb.save_device_params = True
        deck = _gen(tb).upper()
        qualified = 'XMM1:MSKY130_FD_PR__NFET_01V8'
        for param in ('GM', 'VTH', 'VDSAT', 'GDS', 'VGS', 'VDS', 'VBS'):
            assert f'N({qualified}:{param})' in deck
        assert f'ID({qualified})' in deck
        # Never asked for on Xyce (unavailable, confirmed by TASK-330.1's spike).
        for param in ('GMBS', 'CGG', 'CGS', 'CGD', 'CGSO', 'CGDO', 'CGBO'):
            assert f':{param})' not in deck
        assert '.print dc format=raw' in deck.lower()


# ---------------------------------------------------------------------------
# Ammeter matcher -- device_op_vectors reads through a nested ammeter-wrapper
# nesting ('.xmdev' segment) fine on its own; the SILK-LEVEL combination
# (save_device_params + save_device_currents/ammeter_wrapper) is refused
# above, so this proves the READ side against a hand-written fixture that
# never went through silk's emission at all (reviewer's amendment to AC#2).
# ---------------------------------------------------------------------------
class TestAmmeterMatcher:
    @pytest.mark.parametrize('engine', ['ngspice42', 'ngspice47'])
    def test_device_op_vectors_reads_through_the_xmdev_nesting(self, engine):
        path = os.path.join(_FIXTURES, f'{engine}_tran_ammeter_noSC_withparams.raw')
        raw = RawResult(native=read_rawfile(path), source_path=path)
        backend = SimulatorBackend.get('ngspice')
        vecs = backend.device_op_vectors(raw, 'xdut', 'xmmm1')
        assert set(vecs) == set(DEVICE_OP_PARAMS)
        # Only id/gm/vth were saved in this fixture (see its .sp); the rest
        # are absent vectors, i.e. nan -- never a missing key, never 0.
        assert vecs['id'] == pytest.approx(6.942150413922116e-05, rel=1e-6)
        assert vecs['gm'] == pytest.approx(0.00038250867934364453, rel=1e-6)
        assert vecs['vth'] == pytest.approx(0.7693912887284248, rel=1e-6)
        import math
        for param in set(DEVICE_OP_PARAMS) - {'id', 'gm', 'vth'}:
            assert math.isnan(vecs[param]), f'{param} should be nan (not saved)'

    def test_no_op_plot_returns_empty_dict(self):
        path = os.path.join(_FIXTURES, 'ngspice42_tran_baseline_off.raw')
        raw = RawResult(native=read_rawfile(path), source_path=path)
        backend = SimulatorBackend.get('ngspice')
        assert backend.device_op_vectors(raw, 'xdut', 'xmmmn') == {}


# ---------------------------------------------------------------------------
# Real ngspice, both versions -- the reviewer's explicit GO scope: tran, ac,
# dc, Op-only, Op+tran, plus (above) the ammeter matcher.
# ---------------------------------------------------------------------------
_NGSPICE_47 = os.path.expanduser('~/ngspice/ngspice-47/bin/ngspice')


def _ngspice_bins():
    bins = [None]                                    # None = system default (42)
    if os.path.isfile(_NGSPICE_47) and os.access(_NGSPICE_47, os.X_OK):
        bins.append(_NGSPICE_47)
    return bins


pytestmark_real = [pytest.mark.slow, needs_pdk, needs_ngspice]


@pytest.mark.slow
@needs_pdk
@needs_ngspice
@pytest.mark.parametrize('ngspice_bin', _ngspice_bins(),
                         ids=lambda b: 'ngspice47' if b else 'ngspice42')
class TestRealNgspice:
    def _tb(self, monkeypatch, ngspice_bin):
        if ngspice_bin:
            monkeypatch.setenv('SILK_NGSPICE_BIN', ngspice_bin)
        else:
            monkeypatch.delenv('SILK_NGSPICE_BIN', raising=False)
        return _single_nmos_tb(os.path.expanduser(
            '~/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice'))

    # -- condition 2 (no ergonomics cost) + condition 3 (op1 content) -------
    def test_tran_ergonomics_and_op1_content(self, monkeypatch, ngspice_bin):
        off = self._tb(monkeypatch, ngspice_bin)
        off.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
        off.simulate(corner='tt')
        off_sig = off.load_signal('vg')
        off_names = set(off.get_signal_names())

        on = self._tb(monkeypatch, ngspice_bin)
        on.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
        on.save_device_params = True
        on.simulate(corner='tt')

        # Condition 2: no analysis= needed, identical data to switch-off.
        on_sig = on.load_signal('vg')
        assert list(on_sig.y) == pytest.approx(list(off_sig.y))
        assert len(on_sig.y) == len(off_sig.y)

        # Condition 3: op1 holds every node/branch vector the switch-off
        # tran1 had (same names, 'all' still in force) PLUS the '@' params.
        backend = SimulatorBackend.get('ngspice')
        vecs = backend.device_op_vectors(on._raw, '', 'XMM1')
        assert set(vecs) == set(DEVICE_OP_PARAMS)
        import math
        # cgso/cgdo/cgbo are DELIBERATELY undeclared on sky130 nmos1p8/pmos1p8
        # (TASK-330.4 gate finding: BSIM4 only fills these overlap caps after
        # a transient has run in the SAME ngspice session -- reporting a
        # SILENT 0, not nan, outside one; declaring them would put a
        # not-really-measured number on the table). 14 of 17 populate here.
        unavailable = {'cgso', 'cgdo', 'cgbo'}
        nans = {k for k, v in vecs.items() if math.isnan(v)}
        assert nans == unavailable, f"expected exactly {unavailable} nan"
        for k in set(DEVICE_OP_PARAMS) - unavailable:
            assert not math.isnan(vecs[k]), f"{k} should be readable"

    def test_ac_ergonomics(self, monkeypatch, ngspice_bin):
        on = self._tb(monkeypatch, ngspice_bin)
        on.add_sim_type(analysis.AC(variation='dec', points=10, fstart=1e3, fstop=1e9))
        on.save_device_params = True
        on.simulate(corner='tt')
        sig = on.load_signal('vg')             # no analysis= -- must not raise
        assert len(sig.y) > 0
        backend = SimulatorBackend.get('ngspice')
        vecs = backend.device_op_vectors(on._raw, '', 'XMM1')
        assert vecs['id'] == pytest.approx(6.942150413922116e-05, rel=1e-3)

    def test_dc_ergonomics(self, monkeypatch, ngspice_bin):
        on = self._tb(monkeypatch, ngspice_bin)
        on.add_sim_type(analysis.DC(source='Vvg', start=0.0, stop=1.8, step=0.1))
        on.save_device_params = True
        on.simulate(corner='tt')
        sig = on.load_signal('vd')
        assert len(sig.y) > 0
        backend = SimulatorBackend.get('ngspice')
        vecs = backend.device_op_vectors(on._raw, '', 'XMM1')
        assert vecs['vth'] == pytest.approx(0.7693912887284248, rel=1e-3)

    def test_op_only(self, monkeypatch, ngspice_bin):
        """TASK-330.4 gate finding regression: an Op()-only bench (no
        preceding .tran in the same ngspice session) is EXACTLY the case
        that exposed BSIM4 reporting cgso/cgdo/cgbo as a SILENT 0 rather
        than a zero-length/nan vector -- every other declared param stayed
        identical to a .tran bench's own values. Undeclaring the three on
        sky130 (same set already excluded on gf180mcu, for a different,
        zero-length reason) makes them nan/absent here regardless of which
        analysis ran, never a real-looking-but-wrong 0."""
        on = self._tb(monkeypatch, ngspice_bin)
        on.add_sim_type(analysis.Op())
        on.save_device_params = True
        on.simulate(corner='tt')
        sig = on.load_signal('vd')             # single plot either way
        assert len(sig.y) == 1
        backend = SimulatorBackend.get('ngspice')
        vecs = backend.device_op_vectors(on._raw, '', 'XMM1')
        import math
        unavailable = {'cgso', 'cgdo', 'cgbo'}
        nans = {k for k, v in vecs.items() if math.isnan(v)}
        assert nans == unavailable, f"expected exactly {unavailable} nan"
        for k in set(DEVICE_OP_PARAMS) - unavailable:
            assert not math.isnan(vecs[k]), f"{k} should be readable"

    def test_run_test_flow_propagates_the_switch(self, monkeypatch, ngspice_bin):
        """Module.run_test()'s testbench flow (docs 'Flow B') copies
        save_device_currents/terminal_current_method onto the live netlist
        at every run (module_test.py's _build_run_spec); save_device_params
        needs the identical propagation line or setting the switch on the
        OWNER Module before run_test() silently does nothing, since run_test
        drives a separate ModuleTestbench-owned netlist Module instance."""
        if ngspice_bin:
            monkeypatch.setenv('SILK_NGSPICE_BIN', ngspice_bin)
        else:
            monkeypatch.delenv('SILK_NGSPICE_BIN', raising=False)
        from silk.circuit.circuit_elements import PortDir
        pdk = _pdk(os.path.expanduser(
            '~/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice'))
        tmp = scratch_dir()
        # create_test() tests the module it is called ON (which must be a
        # subcircuit -- add_module_under_test instantiates self.module as
        # 'dut'), not a separate module passed in.
        mod = Module('PDK_skywater130', tmp, 'opdut.cir', pdk_details=pdk,
                     module_name='opdut')
        mod.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN,
                                 's': PortDir.GROUND, 'b': PortDir.GROUND})
        mod.add(pdk.nmos1p8('M1', d='d', g='g', s='s', b='b', w=1.0, l=0.15, m=1))

        test = mod.create_test('opparams')
        test.add_module_under_test({'d': 'vd', 'g': 'vg', 's': '0', 'b': '0'})
        test.add_source(sources.VoltageSource('vd', positive='vd', negative='0', dc=1.0))
        test.add_source(sources.VoltageSource('vg', positive='vg', negative='0', dc=1.0))
        test.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))

        mod.save_device_params = True
        report = mod.run_test('opparams', corner='tt')
        tb_netlist = report.sim_results
        assert tb_netlist.save_device_params is True   # propagated, not default False

        backend = SimulatorBackend.get('ngspice')
        vecs = backend.device_op_vectors(tb_netlist._raw, 'xdut', 'XMM1')
        import math
        unavailable = {'cgso', 'cgdo', 'cgbo'}          # undeclared on sky130, TASK-330.4 gate finding
        nans = {k for k, v in vecs.items() if math.isnan(v)}
        assert nans == unavailable, (
            "save_device_params on the OWNER module must reach the testbench "
            f"netlist run_test() actually emits/simulates; expected exactly "
            f"{unavailable} nan")

    def test_undeclared_device_still_yields_a_rawfile(self, monkeypatch, ngspice_bin):
        """Ruling (1), TASK-330.3 gate finding: a device whose model declares
        no op_params gets no '@' saves and must NEVER cost the caller their
        rawfile. Explicitly remove the fixture's declaration so this safety
        check survives additions to the shipped catalog."""
        if ngspice_bin:
            monkeypatch.setenv('SILK_NGSPICE_BIN', ngspice_bin)
        else:
            monkeypatch.delenv('SILK_NGSPICE_BIN', raising=False)
        pdk = _pdk(os.path.expanduser(
            '~/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice'))
        tb = Module('PDK_skywater130', scratch_dir(), 'undeclared_tb.cir',
                    pdk_details=pdk, module_name=None)
        device = pdk.nmos1p8_lvt('M1', d='vd', g='vg', s='0', b='0', w=1.0, l=0.15, m=1)
        device._op_params = None
        device._op_internal_name = None
        device._op_backends = None
        tb.add(device)
        tb.add_source(sources.VoltageSource('vd', positive='vd', negative='0', dc=1.0))
        tb.add_source(sources.VoltageSource('vg', positive='vg', negative='0', dc=1.0))
        tb.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
        tb.save_device_params = True
        with pytest.warns(UserWarning, match='save_device_params.*no device'):
            tb.simulate(corner='tt')
        sig = tb.load_signal('vd')          # main data intact, no analysis= needed
        assert len(sig.y) > 1
        backend = SimulatorBackend.get('ngspice')
        vecs = backend.device_op_vectors(tb._raw, '', 'XMM1')
        import math
        assert all(math.isnan(v) for v in vecs.values()), (
            "undeclared device: every param should be unreadable, not a "
            "guessed/wrong value")

    def test_op_plus_tran(self, monkeypatch, ngspice_bin):
        """Op() declared explicitly alongside Transient() -- the netlist-
        level '.op' card is suppressed either way, so this must behave
        exactly like the tran-only case, not create a second op1/op2 split
        (TASK-330.3 probe, tran_plus_op_fixed.sp)."""
        on = self._tb(monkeypatch, ngspice_bin)
        on.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
        on.add_sim_type(analysis.Op())
        on.save_device_params = True
        on.simulate(corner='tt')
        # Two declared analyses (tran, op) -- analysis= is required, exactly
        # as it already is without the switch (not a new restriction).
        sig = on.load_signal('vg', analysis='tran')
        assert len(sig.y) > 1
        backend = SimulatorBackend.get('ngspice')
        vecs = backend.device_op_vectors(on._raw, '', 'XMM1')
        import math
        unavailable = {'cgso', 'cgdo', 'cgbo'}          # undeclared on sky130, TASK-330.4 gate finding
        nans = {k for k, v in vecs.items() if math.isnan(v)}
        assert nans == unavailable, f"expected exactly {unavailable} nan"


# ---------------------------------------------------------------------------
# TASK-330.3 gate finding: reproduces the reviewer's own nested two-level
# check (dut -> st1/st2 -> MN/MP) across all three PDKs, committed as a slow
# test rather than left as a scratchpad script. sky130 was already passing
# before the fix (single-level benches only); this is the first nested-
# hierarchy coverage for any PDK, and the first real-run coverage at all for
# gf180mcu/sg13g2 under save_device_params.
# ---------------------------------------------------------------------------
def _nested_two_level_bench(pdkname, nmos, pmos, w, l, vdd):
    """dut(amp) -> st1/st2(stage) -> MN/MP -- the reviewer's own r3303/check.py
    shape, kept identical so this test reproduces their finding exactly."""
    from silk.pdk import load_pdk
    from silk.circuit.circuit_elements import PortDir
    pdk = load_pdk(pdkname)
    tmp = scratch_dir()
    stage = Module(pdkname, tmp, 'stage.cir', pdk_details=pdk, module_name='stage')
    stage.define_module_ports({'vdd': PortDir.INOUT, 'in': PortDir.IN,
                               'out': PortDir.OUT, 'vss': PortDir.GROUND})
    stage.add(getattr(pdk, nmos)('MN', d='out', g='in', s='vss', b='vss', w=w, l=l))
    stage.add(getattr(pdk, pmos)('MP', d='out', g='in', s='vdd', b='vdd', w=w, l=l))
    amp = Module(pdkname, tmp, 'amp.cir', pdk_details=pdk, module_name='amp')
    amp.define_module_ports({'vdd': PortDir.INOUT, 'in': PortDir.IN,
                             'out': PortDir.OUT, 'vss': PortDir.GROUND})
    amp.add_module(stage, 'st1', nets={'vdd': 'vdd', 'in': 'in', 'out': 'mid', 'vss': 'vss'})
    amp.add_module(stage, 'st2', nets={'vdd': 'vdd', 'in': 'mid', 'out': 'out', 'vss': 'vss'})
    t = amp.create_test('opchk')
    t.add_module_under_test({'vdd': 'vdd', 'in': 'in', 'out': 'out', 'vss': '0'})
    t.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=vdd))
    t.add_source(sources.VoltageSource('vin', positive='in', negative='0', dc=vdd / 2))
    t.add_sim_type(analysis.Transient(step=1e-9, stop=5e-9))
    return amp, t


@pytest.mark.slow
@needs_pdk
@needs_ngspice
class TestGateFindingNestedHierarchy:
    """TASK-330.3 gate finding reproduction (reviewer's r3303/check.py)."""

    def test_sky130_nested_hierarchy_14_of_17(self):
        """TASK-330.4 gate finding: cgso/cgdo/cgbo are DELIBERATELY
        undeclared on sky130 nmos1p8/pmos1p8 too (not just gf180mcu) --
        BSIM4 only fills these overlap caps after a transient has run in
        the same ngspice session; an Op()-only/AC-only/one-point-DC bench
        reports them as a SILENT 0 (not a zero-length vector), which the
        arc forbids regardless of the underlying reason. 14/17 here,
        same shape as gf180mcu's own (different underlying cause: gf180's
        are genuinely zero-length everywhere, sky130's are session-order
        dependent)."""
        amp, t = _nested_two_level_bench('sky130', 'nmos1p8', 'pmos1p8',
                                         w=1.0, l=0.15, vdd=1.8)
        amp.save_device_params = True
        rep = amp.run_test('opchk', corner='tt')
        nl = rep.sim_results
        sig = nl.load_signal('vdd')             # main data intact, no analysis=
        assert len(sig.y) > 1
        from silk.circuit.modules_results import _iter_mos_device_paths
        from silk.circuit.emitted_names import emitted_instance_name
        backend = SimulatorBackend.get('ngspice')
        paths = list(_iter_mos_device_paths(nl.elem_dict))
        assert len(paths) == 4      # xst1.MN, xst1.MP, xst2.MN, xst2.MP
        import math
        unavailable = {'cgso', 'cgdo', 'cgbo'}
        for path, _designer_path, name, device in paths:
            vecs = backend.device_op_vectors(nl._raw, path, emitted_instance_name(device, name))
            assert set(vecs) == set(DEVICE_OP_PARAMS)
            nans = {k for k, v in vecs.items() if math.isnan(v)}
            assert nans == unavailable, f"{path}.{name}: expected exactly {unavailable} nan"
            for k in set(DEVICE_OP_PARAMS) - unavailable:
                assert not math.isnan(vecs[k]), f"{path}.{name}.{k} should be readable"

    @pytest.mark.requires_gf180
    @needs_pdk_named('gf180mcu')
    def test_gf180mcu_nested_hierarchy_14_of_17(self):
        """The gate finding's own failing case: DIED with no rawfile before
        the per-device-declared fix; now yields 14/17 (cgso/cgdo/cgbo nan,
        measured unavailable on this model -- tests/fixtures/op_point/gf180mcu/)."""
        amp, t = _nested_two_level_bench('gf180mcu', 'nmos3p3', 'pmos3p3',
                                         w=1.0, l=0.28, vdd=3.3)  # microns (gf180mcu user_units)
        amp.save_device_params = True
        rep = amp.run_test('opchk', corner='typical')     # must not raise/abort
        nl = rep.sim_results
        sig = nl.load_signal('vdd')
        assert len(sig.y) > 1
        from silk.circuit.modules_results import _iter_mos_device_paths
        from silk.circuit.emitted_names import emitted_instance_name
        backend = SimulatorBackend.get('ngspice')
        paths = list(_iter_mos_device_paths(nl.elem_dict))
        assert len(paths) == 4
        import math
        unavailable = {'cgso', 'cgdo', 'cgbo'}
        for path, _designer_path, name, device in paths:
            vecs = backend.device_op_vectors(nl._raw, path, emitted_instance_name(device, name))
            assert set(vecs) == set(DEVICE_OP_PARAMS)
            nans = {k for k, v in vecs.items() if isinstance(v, float) and math.isnan(v)}
            assert nans == unavailable, f"{path}.{name}: expected exactly {unavailable} nan"
            for k in set(DEVICE_OP_PARAMS) - unavailable:
                assert not math.isnan(vecs[k]), f"{path}.{name}.{k} should be readable"

    @pytest.mark.requires_sg13g2
    @needs_osdi_runnable('sg13g2')
    def test_sg13g2_nested_hierarchy_undeclared_warns_and_keeps_rawfile(self):
        """Ruling (4): sg13g2 is undeclared (PSP/OSDI, follow-up). Ruling
        (5): the switch on with nothing declared warns once naming the PDK
        and the main analysis (and rawfile) is unaffected -- never a crash,
        unlike gf180mcu before its fix."""
        amp, t = _nested_two_level_bench('sg13g2', 'nmos_lv', 'pmos_lv',
                                         w=1.0, l=0.13, vdd=1.2)
        amp.save_device_params = True
        with pytest.warns(UserWarning, match="save_device_params.*'sg13g2'"):
            rep = amp.run_test('opchk', corner='typical')
        nl = rep.sim_results
        sig = nl.load_signal('vdd')
        assert len(sig.y) > 1
        from silk.circuit.modules_results import _iter_mos_device_paths
        from silk.circuit.emitted_names import emitted_instance_name
        backend = SimulatorBackend.get('ngspice')
        paths = list(_iter_mos_device_paths(nl.elem_dict))
        assert len(paths) == 4
        import math
        for path, _designer_path, name, device in paths:
            vecs = backend.device_op_vectors(nl._raw, path, emitted_instance_name(device, name))
            assert set(vecs) == set(DEVICE_OP_PARAMS)
            assert all(math.isnan(v) for v in vecs.values()), (
                f"{path}.{name}: sg13g2 is undeclared, every param unreadable")
