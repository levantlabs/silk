"""Bus declaration, whole-bus wiring, behavioral simulation and readback.

A 4-bit counter model feeds a 4-bit DAC model through a bus net. The counter
increments once per time step, rather than on clock edges. Read the bus as
bits and as integers, then check the count sequence and DAC voltage.
``tb.add_module(dac, 'dac1', code=count_net)`` connects a whole bus.

Print the bus declarations and assertion results, and save three panels as
``silk-output/bus_ports_demo.png`` (or under ``SILK_EXAMPLES_OUT``). All panels
use sample indices. No simulator or installed PDK models are required.

Run: ``python -m silk.examples.bus_ports_example``.
"""

from __future__ import annotations

import os
import numpy as np

from silk.circuit.modules import Module
from silk.circuit.circuit_elements import Port
from silk.sources.classes import VoltageSource
from silk.sources.waveforms import Pulse
from silk.examples import example_output_dir


# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────
N_BITS    = 4
VREF      = 1.8        # DAC reference voltage (volts)
VDD       = 1.8        # logic-1 voltage
T_STEP    = 1e-9       # 1 ns
T_STOP    = 32e-9      # 32 intervals, including both endpoints
CLK_PER   = 2e-9       # 2 ns clock; the model increments per time step
CLK_TR    = CLK_PER / 20   # 100 ps clock edge; must be nonzero (silk.sources.Pulse)
THRESHOLD = 0.5 * VDD


# ──────────────────────────────────────────────────────────────────────────
# Behavioral models
# ──────────────────────────────────────────────────────────────────────────
def make_counter():
    """4-bit counter — outputs an N_BITS-wide bus that counts 0,1,2,…
    advancing on each simulation step (one count per t-vector entry).
    """
    counter = Module(pdk_name='sky130', dir_folder=example_output_dir('bus_ports'),
                     netlistname='counter4.cir',
                     pdk_details=None, module_name='counter4')
    counter.define_module_ports({
        'clk':   Port.clock_input(),
        'count': Port.digital_output(bits=N_BITS),
    })

    def model(ports, t, params):
        # The model increments per time step without detecting clock edges.
        T = len(t)
        counts = (np.arange(T) % (1 << N_BITS)).astype(int)
        # Bit b at time-i = (counts[i] >> b) & 1, scaled to logic levels.
        bits = np.stack(
            [((counts >> b) & 1).astype(float) * VDD for b in range(N_BITS)]
        )                                    # shape (N_BITS, T)
        ports.count.v = bits               # silk auto-expands to per-bit
    counter.set_behavioral_model(model, execution='vector')
    return counter


def make_dac():
    """4-bit DAC — receives an N_BITS-wide bus, outputs analog voltage."""
    dac = Module(pdk_name='sky130', dir_folder=example_output_dir('bus_ports'),
                 netlistname='dac4.cir',
                 pdk_details=None, module_name='dac4')
    dac.define_module_ports({
        'code': Port.digital_input(bits=N_BITS),
        'vout': Port.analog_output(),
    })

    def model(ports, t, params):
        # Bus input arrives as 2-D (N_BITS, T). LSB-first decode.
        code_bits = ports.code.v                        # shape (N_BITS, T)
        weights = np.array(
            [1 << b for b in range(N_BITS)], dtype=float
        ).reshape(-1, 1)                                  # shape (N_BITS, 1)
        code_int = ((code_bits > THRESHOLD).astype(float) * weights).sum(axis=0)
        full_scale = float((1 << N_BITS) - 1)
        vout = (code_int / full_scale) * VREF
        ports.vout.v = vout                             # scalar — 1-D (T,)
    dac.set_behavioral_model(model, execution='vector')
    return dac


# ──────────────────────────────────────────────────────────────────────────
# Top-level testbench
# ──────────────────────────────────────────────────────────────────────────
def build_testbench():
    counter = make_counter()
    dac     = make_dac()

    tb = Module(pdk_name='sky130', dir_folder=example_output_dir('bus_ports'),
                netlistname='bus_demo_tb.cir',
                pdk_details=None, module_name='bus_demo_tb')

    # Clock source — required by simulate_behavioral's source-graph stage.
    tb.add_source(VoltageSource(
        'clk_src', positive='clk', negative='gnd',
        pulse=Pulse(v1=0, v2=VDD, td=0, tr=CLK_TR, tf=CLK_TR,
                    pw=CLK_PER / 2, per=CLK_PER),
    ))

    # 4-bit bus net connecting counter.count → dac.code
    count_net = tb.add_net('count', bits=N_BITS)
    print(f"  count_net.width = {count_net.width}")
    print(f"  count_net[2].name = {count_net[2].name!r}    "
          f"# subscript returns underlying scalar Net")

    # Place the counter and dac, wiring the whole bus through.
    counter_inst = tb.add_module(counter, 'cnt1', clk='clk', count=count_net)
    dac_inst     = tb.add_module(dac,     'dac1',
                                  code=count_net,           # whole-bus wire
                                  vout='vout')

    return tb, counter_inst, dac_inst


# ──────────────────────────────────────────────────────────────────────────
# Verification + plotting
# ──────────────────────────────────────────────────────────────────────────
def run_and_verify(tb, counter_inst):
    print("\n[2/3] Running simulate_behavioral (exercises bus packing path)...")
    tb.simulate_behavioral(t_step=T_STEP, t_stop=T_STOP)
    t = tb.load_signal('vout').x
    T = len(t)

    # ── Read-side: bus handle on the counter sub-instance ───────────────────
    print("\n[3/3] Verifying via the read-side bus Signal handle...")
    count_sig = counter_inst.count          # bus Signal (width=4)
    print(f"  counter_inst.count.width = {count_sig.width}")
    print(f"  counter_inst.count[3].name = {count_sig[3].name!r}    "
          f"# scalar bit")

    # as_int() decodes the bit-vector at each timepoint into an integer.
    count_int = count_sig.as_int(threshold=THRESHOLD)        # scalar Signal

    # ── Verification ────────────────────────────────────────────────────────
    expected_counts = (np.arange(T) % (1 << N_BITS)).astype(int)
    np.testing.assert_array_equal(
        count_int.y.astype(int), expected_counts,
        err_msg="counter integer trace did not match 0,1,2,...,15 sequence",
    )
    print(f"  ✓ counter integer trace matches expected: 0..{(1<<N_BITS)-1} "
          f"wrapping")

    vout_actual = tb.net_dict['vout'].voltage
    vout_expected = (expected_counts / float((1 << N_BITS) - 1)) * VREF
    np.testing.assert_allclose(vout_actual, vout_expected, atol=1e-9)
    print(f"  ✓ DAC vout matches code/15 × VREF "
          f"(range: 0…{vout_expected.max():.3f} V)")

    return t, count_sig, count_int


def render_plots(tb, t):
    """Save a 3-panel PNG demonstrating bus rendering modes."""
    import matplotlib
    matplotlib.use('Agg')                # headless safety
    import matplotlib.pyplot as plt
    from silk.circuit.report import render_bus_trace

    out_dir = example_output_dir()
    out_path = os.path.join(out_dir, 'bus_ports_demo.png')

    # Behavioral results dict for render_bus_trace (per-bit names → arrays)
    behavioral_results = {
        f'count[{b}]': tb.net_dict[f'count[{b}]'].voltage for b in range(N_BITS)
    }
    behavioral_results['vout'] = tb.net_dict['vout'].voltage

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True)

    # Panel A — counter as integer trace (default mode)
    render_bus_trace(ax1, tb, 'count',
                     behavioral_results=behavioral_results,
                     expand_bits=False, threshold=THRESHOLD)
    ax1.set_ylabel('count (integer)')
    ax1.set_title(f'{N_BITS}-bit counter — render_bus_trace integer mode')
    ax1.grid(True, alpha=0.3)

    # Panel B — DAC analog output (scalar trace)
    ax2.plot(np.arange(len(t)), behavioral_results['vout'])
    ax2.set_ylabel('vout (V)')
    ax2.set_title('DAC analog output — vout = code / 15 × VREF')
    ax2.grid(True, alpha=0.3)

    # Panel C — counter bits expanded (one trace per bit, vertically stacked)
    render_bus_trace(ax3, tb, 'count',
                     behavioral_results=behavioral_results,
                     expand_bits=True, threshold=THRESHOLD)
    ax3.set_ylabel('counter bits')
    ax3.set_xlabel('sample index')
    ax3.set_title('render_bus_trace expand_bits=True — per-bit stack')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(np.arange(0, len(t), 4))

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\n  ✓ wrote {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────
def main():
    print("[1/3] Building testbench (counter4 → dac4 over a 4-bit bus net)...")
    tb, counter_inst, dac_inst = build_testbench()

    # Show what bus declarations did to the child modules' pin_lists.
    counter = tb.elem_dict['cnt1']['device']
    dac     = tb.elem_dict['dac1']['device']
    print(f"  counter.pin_list = {list(counter.pin_list.keys())}")
    print(f"  counter._buses   = {list(counter._buses.keys())}    "
          f"# bus port → BusSpec")
    print(f"  dac.pin_list     = {list(dac.pin_list.keys())}")

    t, count_sig, count_int = run_and_verify(tb, counter_inst)

    # Report a plot-write failure separately from the completed simulation checks.
    try:
        out = render_plots(tb, t)
    except OSError as exc:
        print("\nThe simulation completed "
              "and every assertion above passed.")
        print(f"  the plot could not be written: {exc}")
        print("  set $SILK_EXAMPLES_OUT to a writable directory to keep it.")
        return 1

    print("\nBus and DAC checks passed. Plot:", out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
