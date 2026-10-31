"""Folded-cascode OTA with linked device sizes.

Variable arithmetic links matched pairs and mirror dimensions. Ideal
voltage sources bias the cascode gates. The top-level transient bench
connects output feedback and applies a step to the other input. It measures
the final bias voltages and settling within 1% of the output step swing.
add_module() returns a placement handle retaining the live definition;
its attributes provide access to simulated internal nets.

Run ``MPLBACKEND=Agg .venv/bin/python -m silk.examples.folded_cascode``.
It prints bias values, settling time and a circuit report, then plots
groups of internal-node voltages. Needs ngspice and sky130 models.
"""


import matplotlib.pyplot as plt


from silk import Port, VoltageDomain, generate_report
from silk import load_pdk
from silk.circuit.modules import Module
from silk.circuit.variable import Variable
from silk import sources
from silk.metrics import final_value, settling_time
from silk.simulation import Transient
from silk.specs.spec import Measurement
from silk.examples import example_output_dir


VIN         = 0.9
VSTEP       = 0.05
T_STEP_EDGE = 500e-9


AVDD_1V8 = VoltageDomain('avdd_1v8', nominal=1.8, tolerance=0.1)


def _pdk():
    return load_pdk('sky130')


def _module(name, pdk, out_dir):
    return Module('PDK_skywater130', out_dir, f'{name}.cir',
                  pdk_details=pdk, module_name=name)


def make_fcamp(pdk, out_dir, domain: VoltageDomain) -> Module:

    fcamp = _module('fc_amp', pdk, out_dir)

    w_bias        = Variable('w_bias',        value=1.0)
    m_tail        = Variable('m_tail',        value=8)
    m_bias_ndiode = Variable('m_bias_ndiode', value=4)
    m_bias_nmirr  = 2 * m_bias_ndiode
    m_bias_pdiode = Variable('m_bias_pdiode', value=8)

    i_bias = Variable('i_bias', value=1e-3)

    l_min  = Variable('l_min',  value=0.18)

    w_diff = Variable('w_diff', value=1.0)
    m_diff = Variable('m_diff', value=16)

    w_load = w_bias
    m_load = m_bias_pdiode

    w_casc = Variable('w_casc', value=1.0)
    m_casc = Variable('m_casc', value=1)
    w_mirr = w_bias
    m_mirr = m_tail/2

    with fcamp:

        pdk.nmos1p8('mn_diff_p', d='vdp', g='vinp', s='vtail', b='vss',
                    w=w_diff, l=l_min, m=m_diff)
        pdk.nmos1p8('mn_diff_n', d='vdn', g='vout', s='vtail', b='vss',
                    w=w_diff, l=l_min, m=m_diff)
        pdk.nmos1p8('mn_diff_tail', d='vtail', g='vbiasn', s='vss', b='vss',
                    w=w_bias, l=l_min, m=m_tail)
        pdk.pmos1p8('mp_load_p', d='vdp', g='vbiasp', s='avdd_1p8', b='avdd_1p8',
                    w=w_load, l=l_min, m=m_load)
        pdk.pmos1p8('mp_load_n', d='vdn', g='vbiasp', s='avdd_1p8', b='avdd_1p8',
                    w=w_load, l=l_min, m=m_load)

        pdk.pmos1p8('mp_casc_p', d='vint', g='vcascp', s='vdp', b='avdd_1p8',
                    w=w_casc, l=l_min, m=m_casc)
        pdk.pmos1p8('mp_casc_n', d='vout', g='vcascp', s='vdn', b='avdd_1p8',
                    w=w_casc, l=l_min, m=m_casc)

        pdk.nmos1p8('mn_casc_p', d='vint', g='vcascn', s='vdcp', b='vss',
                    w=w_casc, l=l_min, m=m_casc)
        pdk.nmos1p8('mn_casc_n', d='vout', g='vcascn', s='vdcn', b='vss',
                    w=w_casc, l=l_min, m=m_casc)

        pdk.nmos1p8('mn_mirr_p', d='vdcp', g='vint', s='vss', b='vss',
                    w=w_mirr, l=l_min, m=m_mirr)
        pdk.nmos1p8('mn_mirr_n', d='vdcn', g='vint', s='vss', b='vss',
                    w=w_mirr, l=l_min, m=m_mirr)

        sources.CurrentSource('ibias', positive='avdd_1p8', negative='vbiasn', dc=i_bias)
        pdk.nmos1p8('mn_diode', d='vbiasn', g='vbiasn', s='vss', b='vss',
                    w=w_bias, l=l_min, m=m_bias_ndiode)
        pdk.nmos1p8('mn_mirror_1', d='vbiasp', g='vbiasn', s='vss', b='vss',
                    w=w_bias, l=l_min, m=m_bias_nmirr)
        pdk.pmos1p8('mp_diode', d='vbiasp', g='vbiasp', s='avdd_1p8', b='avdd_1p8',
                    w=w_bias, l=l_min, m=m_bias_pdiode)

        # Ideal sources supply the cascode gate biases.
        sources.VoltageSource('vcascn_src', positive='vcascn', negative='0', dc=1)
        sources.VoltageSource('vcascp_src', positive='vcascp', negative='0', dc=0.8)

    fcamp.define_module_ports({
        'vinp':     Port.analog_input(domain),
        'vout':     Port.analog_output(domain),
        'avdd_1p8': Port.supply(domain),
        'vss':      Port.ground(),
    })

    return fcamp


FCAMP_NODE_GROUPS = [
    ('I/O', [
        ('vinp', 'Vin+ (input)'),
        ('vout', 'Vout (output)'),
    ]),
    ('Differential pair drains & tail', [
        ('vdp',   'Vdp  (diff-p drain / fold point)'),
        ('vdn',   'Vdn  (diff-n drain / fold point)'),
        ('vtail', 'Vtail (tail current source)'),
    ]),
    ('Folded cascode mid-node & mirror sources', [
        ('vint', 'Vint (cascode mid / mirror gate ref)'),
        ('vdcp', 'Vdcp (NMOS cascode-p source)'),
        ('vdcn', 'Vdcn (NMOS cascode-n source)'),
    ]),
    ('Bias nodes', [
        ('vbiasn', 'Vbiasn (NMOS bias)'),
        ('vbiasp', 'Vbiasp (PMOS bias)'),
        ('vcascn', 'Vcascn (NMOS cascode gate)'),
        ('vcascp', 'Vcascp (PMOS cascode gate)'),
    ]),
]


def plot_fcamp_nodes(fcamp_inst) -> None:
    """Plot all fc_amp internal nodes in one figure, one panel per group.

    Reads signals via attribute access on the SubcircuitInstance handle,
    which hides the hierarchical SPICE path convention.
    """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_panels = len(FCAMP_NODE_GROUPS)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels), sharex=True)
    fig.suptitle('Folded Cascode OTA — Internal Node Voltages',
                 fontsize=13, fontweight='bold')

    for ax, (title, nets) in zip(axes, FCAMP_NODE_GROUPS):
        for i, (net, label) in enumerate(nets):
            try:
                sig = getattr(fcamp_inst, net)
            except AttributeError as e:
                print(f"  [warn] skipping {net!r}: {e}")
                continue
            t_us = sig.x * 1e6
            ax.plot(t_us, sig.y, label=label, color=colors[i % len(colors)])

        ax.set_ylabel('Voltage (V)')
        ax.set_title(title, fontsize=10, loc='left')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.5)

    axes[-1].set_xlabel('Time (µs)')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    pdk = _pdk()
    out_dir = example_output_dir()

    print("=" * 62)
    print("  FCAMP Folded Cascode — Testbench")
    print("=" * 62)

    fcamp = make_fcamp(pdk, out_dir, AVDD_1V8)

    tb = Module('PDK_skywater130', out_dir, 'fcamp_tb.cir', pdk_details=pdk)

    avdd = tb.add_net('avdd')
    vinp = tb.add_net('vinp')
    vout = tb.add_net('vout')
    tb.add_net('0')
    # Invalid port keywords fail at this call; the handle retains the live definition.
    fcamp_inst = tb.add_module(fcamp, 'fcamp',
                                vinp=vinp, vout=vout,
                                avdd_1p8=avdd, vss=tb.nets['0'])

    tb.add_source(sources.VoltageSource('vavdd', positive='avdd', negative='0', dc=1.8))
    tb.add_source(sources.VoltageSource(
        'vvinp', positive='vinp', negative='0',
        pulse=sources.Pulse(v1=VIN, v2=VIN + VSTEP,
                            td=T_STEP_EDGE, tr=1e-9, tf=1e-9,
                            pw=10e-6, per=20e-6),
    ))

    spec_t_settle = tb.declare_spec(
        name='t_settle_1pct', metric='t_settle_1pct',
        op='<=', target=500e-9, units='s',
        description='Time for vout to settle within 1% of the output step swing '
                    'after a 50 mV input step.')

    tb.add_sim_type(Transient(step=2e-9, stop=3e-6))

    tb.simulate(corner='tt')
    print("Simulation complete.")
    print(f"  vbiasn final value {final_value(fcamp_inst.vbiasn):.3f} V")
    print(f"  vbiasp final value {final_value(fcamp_inst.vbiasp):.3f} V")
    print(f"  vout   final value {final_value(fcamp_inst.vout):.3f} V")

    vout  = fcamp_inst.vout
    v_pre = float(vout.y[vout.x < T_STEP_EDGE][-1])
    t_settle = settling_time(vout, initial=v_pre, tol=0.01,
                             start=T_STEP_EDGE)
    spec_t_settle.record_measurement(
        Measurement(source='spice', value=t_settle, timestamp=0.0))
    print(f"  settling time (1%) : {t_settle*1e9:.1f} ns")

    print()
    generate_report(tb).print_summary()

    plot_fcamp_nodes(fcamp_inst)
