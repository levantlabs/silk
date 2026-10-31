"""TASK-330.2: analysis.Op() emits a real '.op' and is a first-class,
selectable analysis whose node voltages and source currents are ordinary
scalar Signals a plain-def metric (or a declarative spec) can read.

Companion coverage:
- tests/test_multi_analysis_plots.py — plot-name/selection tables agree,
  and real ngspice/Xyce fixture plots classify as 'op'/'dc' respectively.
- tests/test_signal_kinds.py — 'op' is deliberately absent from
  SignalKind.produced_by / SILK_ANALYSES (the published metric-schema
  vocabulary); see the REVIEWER BINDING comments there.

This module is the one real-ngspice, end-to-end run: emission -> a live
simulation -> a probed node voltage -> a plain-def metric and a declarative
spec reading it, plus the documented testbench-codegen form.
"""
import pytest

from silk.codegen.testbench_generator import generate_testbenches
from silk.simulation.analysis import Op
from tests._env import HAVE_NGSPICE, HAVE_XYCE, needs_pdk


def test_op_to_spice_emits_a_bare_dot_op():
    assert Op().to_spice() == '.op\n'


def test_op_takes_no_parameters():
    # A bare request for the operating point every other analysis already
    # solves first — nothing to configure (TASK-330.1/330.2 design note).
    with pytest.raises(TypeError):
        Op(step=1)


# ---------------------------------------------------------------------------
# Codegen: {'kind': 'op'} -> tb.add_sim_type(analysis.Op())
# ---------------------------------------------------------------------------

OP_DIVIDER_SPEC = {
    "circuit_name": "op_divider",
    "circuit_builder": "_build_op_divider",
    "tests": [
        {"name": "bias",
         "dut_portmap": {"vin": "vin", "vout": "vout", "vss": "0"},
         "sources": [
             {"type": "VoltageSource", "instance": "vin",
              "positive": "vin", "negative": "0", "dc": 2.0},
         ],
         "analyses": [{"kind": "op"}],
         "metrics": [],
         "specs": []}
    ]
}


def test_op_kind_generates_add_sim_type_op_call():
    code = generate_testbenches(OP_DIVIDER_SPEC)
    assert "tb.add_sim_type(analysis.Op())" in code


@pytest.mark.slow
def test_slow_op_run_via_codegen_on_real_ngspice_reads_the_divider_midpoint(
        tmp_path):
    """Generate -> exec -> build -> simulate -> read back a real Op() node
    voltage, entirely PDK-free (TASK-71's rung, no installed PDK needed)."""
    if not HAVE_NGSPICE:
        pytest.skip('needs the ngspice binary')

    from silk.circuit.circuit_elements import PortDir
    from silk.circuit.modules import Module
    from silk.elements import elements

    code = generate_testbenches(OP_DIVIDER_SPEC)
    assert "tb.add_sim_type(analysis.Op())" in code

    def _build_op_divider(pdk):
        mod = Module('op_divider', str(tmp_path), 'op_divider.cir',
                     pdk_details=None, module_name='op_divider')
        mod.define_module_ports({
            'vin': PortDir.INOUT, 'vout': PortDir.INOUT,
            'vss': PortDir.GROUND,
        })
        with mod:
            # 1k/1k divider: Vout = Vin / 2 exactly, at DC.
            elements.resistor('r1', p='vin', n='vout', value=1000)
            elements.resistor('r2', p='vout', n='vss', value=1000)
        return mod

    ns = {'_build_op_divider': _build_op_divider}
    exec(code, ns)
    mod = ns['build_op_divider_with_tests'](None)
    test = mod._tests['bias']

    report = test.run('tt')
    assert report.passed

    # A real .op plot, selectable by name, with the exact divider value —
    # measured against real ngspice, not derived by hand.
    vout = test.netlist.load_signal('v(vout)')
    assert vout.y == pytest.approx([1.0])


@pytest.mark.slow
def test_slow_plain_def_metric_and_spec_read_an_op_probe(tmp_path):
    """AC#2: a plain-def metric (undecorated -- no @metric schema entry)
    and a declarative spec both read an Op()-run Signal reached through
    tb.probe(analysis='op')."""
    if not HAVE_NGSPICE:
        pytest.skip('needs the ngspice binary')

    from silk import sources
    from silk.circuit.circuit_elements import PortDir
    from silk.circuit.modules import Module
    from silk.elements import elements
    from silk.simulation import analysis

    dut = Module('op_divider2', str(tmp_path), 'op_divider2.cir',
                 pdk_details=None, module_name='op_divider2')
    dut.define_module_ports({
        'vin': PortDir.INOUT, 'vout': PortDir.INOUT, 'vss': PortDir.GROUND,
    })
    with dut:
        elements.resistor('r1', p='vin', n='vout', value=1000)
        elements.resistor('r2', p='vout', n='vss', value=1000)

    test = dut.create_test('bias2')
    test.add_module_under_test({'vin': 'vin', 'vout': 'vout', 'vss': '0'})
    test.add_source(sources.VoltageSource('vin', positive='vin',
                                          negative='0', dc=3.0))
    test.add_sim_type(analysis.Op())

    vout = test.probe('vout', analysis='op')

    # Undecorated -- no @metric marker -- so add_metric treats this as the
    # plain callable(ckt) form, not the sig= slot-binding form; it closes
    # over the probe rather than reading ckt directly, which is exactly
    # what "a plain-def metric... reads it" means: no schema entry, no
    # produced_by involvement, just Python reading a Signal.
    def _vmid(ckt):
        return float(vout.y[0])

    test.add_metric('vmid', _vmid)
    test.add_spec('midpoint_ok', 'vmid', '>', 1.0)

    report = test.run('tt')
    assert report.passed
    assert report.metrics['vmid'] == pytest.approx(1.5)  # 3.0 * 1k/(1k+1k)


# --- Xyce rendering (reviewer, TASK-330.2 gate) -----------------------------
# Xyce has no '.print op' form. A private zero-volt source supplies the
# one-point sweep without changing any user source's operating-point value.

def _xyce_render(sources_by_name):
    from silk.simulation.backend import EmitContext, SimulatorBackend
    ctx = EmitContext(sources=sources_by_name)
    return SimulatorBackend.get('xyce').render_analysis(Op(), ctx)


def test_xyce_renders_op_as_a_single_point_dc_sweep():
    from silk import sources
    vin = sources.VoltageSource('vin', positive='vin', negative='0', dc=3.0)
    line = _xyce_render({'vin': vin})
    card, sweep = line.splitlines()
    name, node, ground, dc, value = card.split()
    assert name != vin.instance_name
    assert ground == '0' and dc == 'DC' and value == '0'
    assert sweep == f'.dc {name} {value} {value} 1'


def test_xyce_op_on_a_zero_volt_source_uses_a_nonzero_step():
    from silk import sources
    amm = sources.VoltageSource('amm', positive='a', negative='b', dc=0)
    assert _xyce_render({'amm': amm}).split()[-1] == '1'


def test_xyce_op_without_a_voltage_source_uses_a_private_source():
    assert _xyce_render({}).splitlines()[-1].endswith(' 0 0 1')


def test_ngspice_op_ignores_the_sources_it_is_given():
    from silk import sources
    from silk.simulation.backend import EmitContext, SimulatorBackend
    vin = sources.VoltageSource('vin', positive='vin', negative='0', dc=3.0)
    ctx = EmitContext(sources={'vin': vin})
    assert SimulatorBackend.get('ngspice').render_analysis(Op(), ctx) == '.op\n'


@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.skipif(not HAVE_XYCE, reason="needs configured Xyce")
def test_slow_op_on_real_xyce_reads_the_divider_midpoint(tmp_path):
    """Op() on a real Xyce run: the synthesized one-point '.dc' sweep gives
    the operating point. On Xyce the plot classifies as 'dc', not 'op'
    (see analysis_kind._KIND_ANALYSIS), so the probe reads it as 'dc'."""
    from silk import sources
    from silk.circuit.circuit_elements import PortDir
    from silk.circuit.modules import Module
    from silk.elements import elements
    from silk.simulation import analysis

    dut = Module('op_divider_x', str(tmp_path), 'op_divider_x.cir',
                 pdk_details=None, module_name='op_divider_x')
    dut.define_module_ports({
        'vin': PortDir.INOUT, 'vout': PortDir.INOUT, 'vss': PortDir.GROUND,
    })
    with dut:
        elements.resistor('r1', p='vin', n='vout', value=1000)
        elements.resistor('r2', p='vout', n='vss', value=1000)

    dut.simulator = 'xyce'
    test = dut.create_test('bias_x')
    test.add_module_under_test({'vin': 'vin', 'vout': 'vout', 'vss': '0'})
    test.add_source(sources.VoltageSource('vin', positive='vin',
                                          negative='0', dc=3.0))
    test.add_sim_type(analysis.Op())
    vout = test.probe('vout')

    def _vmid(ckt):
        return float(vout.y[0])

    test.add_metric('vmid', _vmid)
    report = test.run('tt')
    assert report.metrics['vmid'] == pytest.approx(1.5)


@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.skipif(not HAVE_XYCE, reason='needs configured Xyce')
@pytest.mark.skipif(not HAVE_NGSPICE, reason='needs ngspice reference')
@pytest.mark.parametrize('case, expected', [
    ('sine', 0.5), ('phase', 0.6), ('delay_phase', 0.6),
    ('pulse', 0.4), ('pwl_offset', 0.7), ('pwl_zero', 0.5),
    ('pwl_delayed', 0.8), ('other_bias', 0.5), ('dc', 0.9),
])
def test_xyce_op_preserves_waveform_bias(tmp_path, case, expected):
    """The Op sweep must neither override waveform bias nor name a missing card."""
    from silk import sources
    from silk.circuit.modules import Module
    from silk.elements import elements

    values = {}
    for simulator in ('ngspice', 'xyce'):
        mod = Module('bias', str(tmp_path / simulator), 'bias.cir',
                     pdk_details=None)
        mod.simulator = simulator
        kwargs = {}
        if case in ('sine', 'phase', 'delay_phase', 'other_bias'):
            kwargs['sine'] = sources.Sine(
                0.5, 0.1, 1e9, phase=90 if 'phase' in case else 0,
                td=1e-9 if case == 'delay_phase' else 0)
        elif case == 'pulse':
            kwargs['pulse'] = sources.Pulse(0.4, 0.8, td=1e-9, tr=1e-10,
                                           tf=1e-10, pw=2e-9, per=4e-9)
        elif case.startswith('pwl'):
            kwargs['pwl'] = ([(1e-9, 0.8), (2e-9, 0.9)]
                             if case == 'pwl_delayed'
                             else [(0, 0.5), (2e-9, 0.9)])
            kwargs['dc'] = 0.2 if case == 'pwl_offset' else 0
        else:
            kwargs['dc'] = 0.9
        mod.add_source(sources.VoltageSource('a', positive='out', negative='0',
                                              **kwargs))
        with mod:
            elements.resistor('load', p='out', n='0', value=1000)
        if case == 'other_bias':
            mod.add_source(sources.VoltageSource('bias', positive='bias',
                                                  negative='0', dc=1.2))
            with mod:
                elements.resistor('biasload', p='bias', n='0', value=1000)
        mod.add_sim_type(Op())
        mod.simulate()
        values[simulator] = mod.load_signal('v(out)').y
        assert values[simulator] == pytest.approx([expected])
        if case == 'other_bias':
            assert mod.load_signal('v(bias)').y == pytest.approx([1.2])
    assert values['xyce'] == pytest.approx(values['ngspice'])


@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.skipif(not HAVE_XYCE, reason='needs configured Xyce')
def test_xyce_private_op_source_avoids_user_names_and_results(tmp_path):
    from silk import sources
    from silk.circuit.modules import Module
    from silk.elements import elements

    mod = Module('private_op', str(tmp_path), 'private_op.cir', pdk_details=None)
    mod.simulator = 'xyce'
    # Authored sources cannot occupy the reserved private helper namespace.
    mod.add_source(sources.VoltageSource('stim', positive='out',
                                          negative='0', dc=0.7))
    with mod:
        elements.resistor('load', p='out', n='0', value=1000)
    mod.add_sim_type(Op())
    mod.simulate()
    assert mod.load_signal('v(out)').y == pytest.approx([0.7])
    names = mod.get_signal_names()
    assert not any('__silk_op_1' in name.lower() for name in names)


@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.skipif(not HAVE_XYCE, reason='needs configured Xyce')
def test_xyce_private_op_source_supports_current_only_deck(tmp_path):
    from silk import sources
    from silk.circuit.modules import Module
    from silk.elements import elements

    mod = Module('current_op', str(tmp_path), 'current_op.cir', pdk_details=None)
    mod.simulator = 'xyce'
    mod.add_source(sources.CurrentSource('drive', positive='0', negative='out',
                                         dc=1e-3))
    with mod:
        elements.resistor('load', p='out', n='0', value=1000)
    mod.add_sim_type(Op())
    mod.simulate()
    assert mod.load_signal('v(out)').y == pytest.approx([1.0])
    assert not any('__silk_op' in name.lower() for name in mod.get_signal_names())


def test_xyce_op_name_allocation_does_not_resolve_source_values():
    from silk.circuit.variable import Variable
    from silk.sources import VoltageSource

    source = VoltageSource('stim', positive='out', negative='0',
                           dc=Variable('v', value=None))
    assert _xyce_render({'stim': source}) == (
        'V__silk_op __silk_op 0 DC 0\n.dc V__silk_op 0 0 1\n')


@pytest.mark.parametrize('pwl', [False, True])
def test_xyce_op_name_allocation_uses_metadata_without_rendering(monkeypatch, pwl):
    from unittest.mock import Mock
    from silk.sources import VoltageSource, PWL, Sine

    wave = PWL([(0, 0.5), (1e-9, 0.7)]) if pwl else Sine(0.5, 0.1, 1e9)
    source = VoltageSource('stim', positive='out', negative='0',
                           **({'pwl': wave} if pwl else {'sine': wave}))
    source_spy = Mock(wraps=source.to_spice)
    waveform_spy = Mock(wraps=wave.to_spice)
    monkeypatch.setattr(source, 'to_spice', source_spy)
    monkeypatch.setattr(wave, 'to_spice', waveform_spy)
    assert _xyce_render({'stim': source}) == (
        'V__silk_op __silk_op 0 DC 0\n.dc V__silk_op 0 0 1\n')
    source_spy.assert_not_called()
    waveform_spy.assert_not_called()


@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.skipif(not HAVE_XYCE, reason='needs configured Xyce')
def test_xyce_op_save_all_and_source_current_exclude_private_branch(tmp_path):
    from silk import Module, Port, sources
    from silk.elements import elements

    dut = Module('saved_op', str(tmp_path), 'saved_op.cir',
                 pdk_details=None, module_name='load')
    dut.simulator = 'xyce'
    dut.define_module_ports({'out': Port.analog_input(), 'vss': Port.ground()})
    with dut:
        elements.resistor('load', p='out', n='vss', value=1000)
    tb = dut.create_test('bias')
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_source(sources.VoltageSource('stim', positive='out', negative='0',
                                         dc=0.5))
    tb.add_sim_type(Op())
    tb.netlist.set_save_list('*')
    current = tb.probe('i(stim)')
    voltage = tb.probe('out')
    tb.run('tt')
    assert voltage.y == pytest.approx([0.5])
    assert current.y == pytest.approx([-0.5e-3])
    names = tb.netlist.get_signal_names()
    assert not any('__silk_op' in name.lower() for name in names)
    # Check the rawfile inventory too, not only designer-name presentation.
    assert not any('__silk_op' in name.lower()
                   for name in tb.netlist.get_signal_names(raw=True))


@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.skipif(not HAVE_XYCE, reason='needs configured Xyce')
@needs_pdk
def test_xyce_op_monte_carlo_sample_replays(tmp_path):
    from silk import Module, Port, load_pdk, sources

    pdk = load_pdk('sky130')
    dut = Module('mc_op', str(tmp_path), 'mc_op.cir',
                 pdk_details=pdk, module_name='diode')
    dut.simulator = 'xyce'
    dut.define_module_ports({'gate': Port.analog_output(), 'vss': Port.ground()})
    with dut:
        pdk.nmos1p8('m1', d='gate', g='gate', s='vss', b='vss', w=5.0, l=2.0)
    tb = dut.create_test('bias')
    tb.add_module_under_test({'gate': 'gate', 'vss': '0'})
    tb.add_source(sources.CurrentSource('bias', positive='0', negative='gate',
                                         dc=20e-6))
    tb.add_sim_type(Op())
    tb.add_metric('voltage', lambda ckt: float(ckt.load_signal('gate', 'V').y[0]))
    samples = tb.run_monte_carlo(2, seed=7, n_workers=1)
    assert not samples.failed
    voltages = [report.metrics['voltage'] for report in samples.reports.values()]
    assert len(voltages) == 2 and voltages[0] != voltages[1]
    assert all(0.2 < value < 2.0 for value in voltages)
    assert samples.rerun(('tt', 0)).metrics['voltage'] == voltages[0]


def test_private_op_reserves_authored_names_and_keeps_opaque_body_collision_loop():
    from silk import sources
    from silk.simulation.backend import EmitContext, SimulatorBackend
    from silk.exceptions import CircuitError
    for name in ('__SILK_OP', '__silk_op_1'):
        with pytest.raises(CircuitError, match='reserved __silk_ prefix'):
            sources.VoltageSource(name, 'out', '0', dc=0)
    expected = 'V__silk_op_2 __silk_op_2 0 DC 0\n.dc V__silk_op_2 0 0 1\n'
    # This is the backend's already-emitted body, not authored graph net names.
    ctx = EmitContext(emitted_deck='V__SILK_OP out 0 0\nR1 __silk_op_1 0 1k\n')
    assert SimulatorBackend.get('xyce').render_analysis(Op(), ctx) == expected
