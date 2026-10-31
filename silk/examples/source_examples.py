"""
Source Examples

Demonstrates independent, controlled and behavioral SPICE sources in one
transient simulation. Each source drives a simple resistive (or RC) load;
output node voltages are captured and plotted.

Circuit topology
----------------
Independent V/I sources  → 1 kΩ resistor → GND
Controlled sources        share a common 1 MHz sine control input
Behavioral sources        operate on independent sine inputs
"""

# Let matplotlib use MPLBACKEND or matplotlibrc.
import matplotlib.pyplot as plt

from silk.circuit.modules import Module
from silk import load_pdk
from silk.examples import example_output_dir
from silk import sources
from silk.elements import elements
from silk.simulation import analysis

# ---------------------------------------------------------------------------
# PDK setup
# ---------------------------------------------------------------------------


def main():
    pdk130 = load_pdk('sky130')

    dir_folder = example_output_dir('source_examples')

    # ---------------------------------------------------------------------------
    # Build netlist
    # ---------------------------------------------------------------------------
    nt = Module('PDK_skywater130', dir_folder, 'source_examples.cir',
                pdk_details=pdk130)

    # ── 1. Pulse voltage source ─────────────────────────────────────────────────
    nt.add_source(sources.VoltageSource('pulse',
        positive='n_pulse', negative='0',
        pulse=sources.Pulse(v1=0, v2=1.8,
                            td=100e-9, tr=10e-9, tf=10e-9,
                            pw=200e-9, per=500e-9)))
    nt.add(elements.resistor('pulse', p='n_pulse', n='0', value=1e3))

    # ── 2. Sine voltage source ───────────────────────────────────────────────────
    nt.add_source(sources.VoltageSource('sine',
        positive='n_sine', negative='0',
        sine=sources.Sine(voffset=0, vampl=1.0, freq=1e6)))
    nt.add(elements.resistor('sine', p='n_sine', n='0', value=1e3))

    # ── 3. Exponential voltage source ────────────────────────────────────────────
    #   Rise: starts at 100 ns, τ = 50 ns
    #   Fall: starts at 400 ns, τ = 100 ns
    nt.add_source(sources.VoltageSource('exp',
        positive='n_exp', negative='0',
        exponential=sources.Exponential(v1=0, v2=1.8,
                                        td1=100e-9, tau1=50e-9,
                                        td2=400e-9, tau2=100e-9)))
    nt.add(elements.resistor('exp', p='n_exp', n='0', value=1e3))

    # ── 4. PWL voltage source ────────────────────────────────────────────────────
    pwl_wave = (sources.PWL()
                .add(0, 0)
                .add(100e-9, 0)
                .add(200e-9, 1.8)
                .add(600e-9, 1.8)
                .add(700e-9, 0)
                .add(1e-6, 0))
    nt.add_source(sources.VoltageSource('pwl',
        positive='n_pwl', negative='0', pwl=pwl_wave))
    nt.add(elements.resistor('pwl', p='n_pwl', n='0', value=1e3))

    # ── 5. SFFM — Single-Frequency FM ───────────────────────────────────────────
    #   Carrier 1 MHz, modulating signal 100 kHz, modulation index = 2
    nt.add_source(sources.VoltageSource('sffm',
        positive='n_sffm', negative='0',
        sffm=sources.SFFM(vo=0, va=1.0, fc=1e6, mdi=2, fs=100e3)))
    nt.add(elements.resistor('sffm', p='n_sffm', n='0', value=1e3))

    # ── 6. AM — Amplitude Modulation ─────────────────────────────────────────────
    #   Carrier 1 MHz, modulating signal 100 kHz
    nt.add_source(sources.VoltageSource('am',
        positive='n_am', negative='0',
        am=sources.AM(va=1.0, vo=0.5, mf=100e3, fc=1e6)))
    nt.add(elements.resistor('am', p='n_am', n='0', value=1e3))

    # ── 7. TransientNoise ────────────────────────────────────────────────────────
    #   White noise 10 mV rms, 1 ns correlation time.
    #   RC filter (R=100 Ω, C=1 nF) smooths the waveform for visibility.
    nt.add_source(sources.VoltageSource('noise',
        positive='n_noise_src', negative='0',
        trnoise=sources.TransientNoise(na='10m', nt='1n')))
    nt.add(elements.resistor('noise_r', p='n_noise_src', n='n_noise', value=100))
    nt.add(elements.capacitor('noise_c', p='n_noise', n='0', value=1e-9))

    # ── 8. RandomSource — Gaussian ───────────────────────────────────────────────
    #   Type 2 = Gaussian, new value every 10 ns, std = 1 V
    nt.add_source(sources.VoltageSource('rand',
        positive='n_rand', negative='0',
        trrandom=sources.RandomSource(type=2, ts='10n', param1=1.0)))
    nt.add(elements.resistor('rand', p='n_rand', n='0', value=1e3))

    # ── 9. DC current source ─────────────────────────────────────────────────────
    #   1 mA drawn through 1 kΩ → -1 V at n_ibias
    nt.add_source(sources.CurrentSource('ibias',
        positive='n_ibias', negative='0', dc=1e-3))
    nt.add(elements.resistor('ibias', p='n_ibias', n='0', value=1e3))

    # ── 10. Pulse current source ─────────────────────────────────────────────────
    nt.add_source(sources.CurrentSource('ipulse',
        positive='n_ipulse', negative='0',
        pulse=sources.Pulse(v1=0, v2=1e-3,
                            td=100e-9, tr=10e-9, tf=10e-9,
                            pw=200e-9, per=500e-9)))
    nt.add(elements.resistor('ipulse', p='n_ipulse', n='0', value=1e3))

    # ── 11. VCVS — gain = 2 ──────────────────────────────────────────────────────
    #   Common control input: sine 0.5 V amplitude, 0.5 V offset
    nt.add_source(sources.VoltageSource('ctrl',
        positive='n_ctrl', negative='0',
        sine=sources.Sine(voffset=0.5, vampl=0.5, freq=1e6)))
    nt.add(elements.resistor('ctrl', p='n_ctrl', n='0', value=10e3))

    nt.add_source(sources.VCVS('eamp',
        out_pos='n_vcvs', out_neg='0',
        ctrl_pos='n_ctrl', ctrl_neg='0',
        gain=2))

    # ── 12. VCCS — gm = 1 mA/V, R_load = 1 kΩ ───────────────────────────────────
    #   Output voltage = -gm × R_load × v(n_ctrl) = -1 × v(n_ctrl)
    nt.add_source(sources.VCCS('gm1',
        out_pos='n_vccs', out_neg='0',
        ctrl_pos='n_ctrl', ctrl_neg='0',
        gm=1e-3))
    nt.add(elements.resistor('vccs', p='n_vccs', n='0', value=1e3))

    # ── 13. CCVS — transresistance = 1 kΩ ───────────────────────────────────────
    #   Input sine through 1 kΩ → 0 V ammeter → measures current.
    #   V_out = R_trans × I_in = 1k × (V_in / 1k) = V_in  (unity gain)
    nt.add_source(sources.VoltageSource('ctrl_ccvs',
        positive='n_ctrl_ccvs', negative='0',
        sine=sources.Sine(voffset=0.5, vampl=0.5, freq=1e6)))
    nt.add(elements.resistor('ccvs_in', p='n_ctrl_ccvs', n='n_ccvs_amm', value=1e3))
    nt.add_source(sources.VoltageSource('ammeter_ccvs',
        positive='n_ccvs_amm', negative='0', dc=0))  # 0 V ammeter
    nt.add_source(sources.CCVS('htrans',
        out_pos='n_ccvs', out_neg='0',
        transresistance=1000, ctrl_vname='Vammeter_ccvs'))

    # ── 14. CCCS — gain = 2 ──────────────────────────────────────────────────────
    #   I_out = 2 × I_in;  V_out = -I_out × R_load = -2 × V_ctrl / 1k × 1k = -2 × V_ctrl
    nt.add_source(sources.VoltageSource('ctrl_cccs',
        positive='n_ctrl_cccs', negative='0',
        sine=sources.Sine(voffset=0.5, vampl=0.5, freq=1e6)))
    nt.add(elements.resistor('cccs_in', p='n_ctrl_cccs', n='n_cccs_amm', value=1e3))
    nt.add_source(sources.VoltageSource('ammeter_cccs',
        positive='n_cccs_amm', negative='0', dc=0))  # 0 V ammeter
    nt.add_source(sources.CCCS('fmir',
        out_pos='n_cccs', out_neg='0',
        gain=2, ctrl_vname='Vammeter_cccs'))
    nt.add(elements.resistor('cccs_out', p='n_cccs', n='0', value=1e3))

    # ── 15. BehavioralVoltage — differential ─────────────────────────────────────
    #   V_out = V(n_ba) − V(n_bb)
    nt.add_source(sources.VoltageSource('bsrc_a',
        positive='n_ba', negative='0',
        sine=sources.Sine(voffset=0, vampl=1.0, freq=1e6)))
    nt.add_source(sources.VoltageSource('bsrc_b',
        positive='n_bb', negative='0',
        sine=sources.Sine(voffset=0, vampl=0.7, freq=1e6, phase=45)))
    nt.add(elements.resistor('ba', p='n_ba', n='0', value=10e3))
    nt.add(elements.resistor('bb', p='n_bb', n='0', value=10e3))
    nt.add_source(sources.BehavioralVoltage('bdiff',
        positive='n_bdiff', negative='0',
        expression='v(n_ba)-v(n_bb)'))

    # ── 16. BehavioralCurrent — transconductance gm = 1 mA/V ─────────────────────
    #   I_out = 1 mA/V × V_in;  V_out = -I_out × 1 kΩ  →  V_out = -V_in  (gain -1)
    nt.add_source(sources.VoltageSource('bI_in',
        positive='n_bi_in', negative='0',
        sine=sources.Sine(voffset=0, vampl=1.0, freq=1e6)))
    nt.add(elements.resistor('bi_in', p='n_bi_in', n='0', value=10e3))
    nt.add_source(sources.BehavioralCurrent('bgm',
        positive='n_bgm', negative='0',
        expression='1e-3*v(n_bi_in)'))
    nt.add(elements.resistor('bgm', p='n_bgm', n='0', value=1e3))

    # ---------------------------------------------------------------------------
    # Simulate: 1 ns step, 5 µs total
    # ---------------------------------------------------------------------------
    nt.add_sim_type(analysis.Transient(step=1e-9, stop=5e-6))

    # simulate() emits the deck, runs the simulator and loads the results;
    # failures raise SimulationError.
    nt.simulate(corner='tt')

    # ---------------------------------------------------------------------------
    # Load signals
    # ---------------------------------------------------------------------------
    s_pulse      = nt.load_signal('n_pulse')
    s_sine       = nt.load_signal('n_sine')
    s_exp        = nt.load_signal('n_exp')
    s_pwl        = nt.load_signal('n_pwl')
    s_sffm       = nt.load_signal('n_sffm')
    s_am         = nt.load_signal('n_am')
    s_noise      = nt.load_signal('n_noise')
    s_rand       = nt.load_signal('n_rand')
    s_ibias      = nt.load_signal('n_ibias')
    s_ipulse     = nt.load_signal('n_ipulse')
    s_ctrl       = nt.load_signal('n_ctrl')
    s_vcvs       = nt.load_signal('n_vcvs')
    s_vccs       = nt.load_signal('n_vccs')
    s_ctrl_ccvs  = nt.load_signal('n_ctrl_ccvs')
    s_ccvs       = nt.load_signal('n_ccvs')
    s_ctrl_cccs  = nt.load_signal('n_ctrl_cccs')
    s_cccs       = nt.load_signal('n_cccs')
    s_ba         = nt.load_signal('n_ba')
    s_bb         = nt.load_signal('n_bb')
    s_bdiff      = nt.load_signal('n_bdiff')
    s_bi_in      = nt.load_signal('n_bi_in')
    s_bgm        = nt.load_signal('n_bgm')

    # ---------------------------------------------------------------------------
    # Plot — 4×4 grid, one panel per source type
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle('silk Source Types — Transient Response', fontsize=14)


    def us(x):
        """Seconds → microseconds."""
        return [v * 1e6 for v in x]


    def panel(ax, title, x, traces, ylim=None):
        """traces: list of (label, y_array) tuples."""
        for label, y in traces:
            ax.plot(us(x), y, label=label or None, linewidth=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Time (µs)', fontsize=7)
        ax.set_ylabel('V', fontsize=7)
        ax.tick_params(labelsize=7)
        if any(t[0] for t in traces):
            ax.legend(fontsize=6)
        if ylim:
            ax.set_ylim(ylim)


    # Row 0 — basic waveform types
    panel(axes[0, 0], 'Pulse',       s_pulse.x,  [('', s_pulse.y)],  ylim=[-0.2, 2.0])
    panel(axes[0, 1], 'Sine',        s_sine.x,   [('', s_sine.y)],   ylim=[-1.2, 1.2])
    panel(axes[0, 2], 'Exponential', s_exp.x,    [('', s_exp.y)],    ylim=[-0.2, 2.0])
    panel(axes[0, 3], 'PWL',         s_pwl.x,    [('', s_pwl.y)],    ylim=[-0.2, 2.0])

    # Row 1 — modulated and stochastic
    panel(axes[1, 0], 'SFFM (FM)', s_sffm.x,  [('', s_sffm.y)],  ylim=[-1.2, 1.2])
    panel(axes[1, 1], 'AM',        s_am.x,    [('', s_am.y)])
    panel(axes[1, 2], 'Noise (filtered)', s_noise.x, [('', s_noise.y)])
    panel(axes[1, 3], 'Random (Gaussian)', s_rand.x, [('', s_rand.y)])

    # Row 2 — current sources and voltage-controlled sources
    panel(axes[2, 0], 'DC CurrentSource',    s_ibias.x,  [('', s_ibias.y)])
    panel(axes[2, 1], 'Pulse CurrentSource', s_ipulse.x, [('', s_ipulse.y)])
    panel(axes[2, 2], 'VCVS (gain=2)',       s_ctrl.x,
          [('ctrl', s_ctrl.y), ('out ×2', s_vcvs.y)])
    panel(axes[2, 3], 'VCCS (gm=1mA/V)',     s_ctrl.x,
          [('ctrl', s_ctrl.y), ('out', s_vccs.y)])

    # Row 3 — current-controlled and behavioral sources
    panel(axes[3, 0], 'CCVS (Rtr=1kΩ)', s_ctrl_ccvs.x,
          [('ctrl', s_ctrl_ccvs.y), ('out', s_ccvs.y)])
    panel(axes[3, 1], 'CCCS (gain=2)',   s_ctrl_cccs.x,
          [('ctrl', s_ctrl_cccs.y), ('out', s_cccs.y)])
    panel(axes[3, 2], 'BehavioralV (A−B)', s_ba.x,
          [('A', s_ba.y), ('B', s_bb.y), ('A−B', s_bdiff.y)])
    panel(axes[3, 3], 'BehavioralI (gm=1mA/V)', s_bi_in.x,
          [('in', s_bi_in.y), ('out', s_bgm.y)])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
