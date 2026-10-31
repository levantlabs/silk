"""
PWL Waveform Example
====================

Demonstrates the different ways to create and use piecewise-linear (PWL)
waveforms with silk source elements.

Methods shown:
  1. Incremental building with .add()
  2. Fluent chaining
  3. From a list of (time, value) tuples
  4. From a flat list
  5. With repeat (r) and delay (td) parameters
  6. PWL on a current source
  7. Loading PWL data from a file via Python
"""

import os
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

    dir_folder = example_output_dir('pwl')

    # ---------------------------------------------------------------------------
    # 1. Incremental building with .add()
    # ---------------------------------------------------------------------------

    print("=" * 60)
    print("1. Build PWL incrementally with .add()")
    print("=" * 60)

    pwl_step = sources.PWL()
    pwl_step.add(0, 0)
    pwl_step.add(1e-9, 0)       # hold low
    pwl_step.add(2e-9, 1.8)     # ramp up
    pwl_step.add(100e-9, 1.8)   # hold high

    print(f"  Points: {len(pwl_step)}")
    for t, v in pwl_step:
        print(f"    t={t:10.2e}  v={v}")
    print(f"  SPICE: {pwl_step.to_spice()}")

    # ---------------------------------------------------------------------------
    # 2. Fluent chaining
    # ---------------------------------------------------------------------------

    print()
    print("=" * 60)
    print("2. Fluent chaining")
    print("=" * 60)

    pwl_pulse = (sources.PWL()
                 .add(0, 0)
                 .add(10e-9, 0)
                 .add(11e-9, 1.8)
                 .add(50e-9, 1.8)
                 .add(51e-9, 0)
                 .add(100e-9, 0))

    print(f"  Points: {len(pwl_pulse)}")
    print(f"  SPICE: {pwl_pulse.to_spice()}")

    # ---------------------------------------------------------------------------
    # 3. From a list of tuples
    # ---------------------------------------------------------------------------

    print()
    print("=" * 60)
    print("3. From list of (time, value) tuples")
    print("=" * 60)

    pwl_ramp = sources.PWL([
        (0, 0),
        (50e-9, 1.8),
        (100e-9, 0),
    ])

    print(f"  Points: {len(pwl_ramp)}")
    print(f"  SPICE: {pwl_ramp.to_spice()}")

    # ---------------------------------------------------------------------------
    # 4. From a flat list
    # ---------------------------------------------------------------------------

    print()
    print("=" * 60)
    print("4. From flat list [t0, v0, t1, v1, ...]")
    print("=" * 60)

    pwl_flat = sources.PWL([0, 0, 25e-9, 0.9, 50e-9, 1.8, 100e-9, 1.8])
    print(f"  Points: {len(pwl_flat)}")
    print(f"  SPICE: {pwl_flat.to_spice()}")

    # ---------------------------------------------------------------------------
    # 5. With repeat and delay parameters
    # ---------------------------------------------------------------------------

    print()
    print("=" * 60)
    print("5. PWL with repeat (r) and delay (td)")
    print("=" * 60)

    pwl_repeat = sources.PWL(
        [(0, 0), (5e-9, 1.8), (10e-9, 0)],
        r=0,         # repeat from t=0
        td=2e-9,     # delay before first edge
    )
    print(f"  SPICE: {pwl_repeat.to_spice()}")

    # ---------------------------------------------------------------------------
    # 6. Build a circuit using PWL sources
    # ---------------------------------------------------------------------------

    print()
    print("=" * 60)
    print("6. Circuit with PWL sources")
    print("=" * 60)

    ckt = Module('PDK_skywater130', dir_folder, 'pwl_demo.cir', pdk_details=pdk130)

    # Step input via PWL
    ckt.add_source(sources.VoltageSource(
        'vstep', positive='vin_step', negative='0', pwl=pwl_step))

    # Pulse via PWL
    ckt.add_source(sources.VoltageSource(
        'vpulse', positive='vin_pulse', negative='0', pwl=pwl_pulse))

    # Ramp via PWL
    ckt.add_source(sources.VoltageSource(
        'vramp', positive='vin_ramp', negative='0', pwl=pwl_ramp))

    # PWL current source — load step
    pwl_iload = sources.PWL().add(0, 0).add(20e-9, 0).add(21e-9, 1e-3).add(100e-9, 1e-3)
    ckt.add_source(sources.CurrentSource(
        'iload', positive='0', negative='vout', pwl=pwl_iload))

    # Load resistors
    ckt.add(elements.resistor('r_step',  p='vin_step',  n='0', value=1e3))
    ckt.add(elements.resistor('r_pulse', p='vin_pulse', n='0', value=1e3))
    ckt.add(elements.resistor('r_ramp',  p='vin_ramp',  n='0', value=1e3))
    ckt.add(elements.resistor('r_out',   p='vout',      n='0', value=1e3))

    ckt.add_sim_type(analysis.Transient(step=0.1e-9, stop=100e-9))

    netlist_path = ckt.generate_netlist(corner='tt')

    print("  Netlist written to:", netlist_path)
    print("  Sources:")
    for name in ckt.sig_sources_dict:
        print(f"    {name}")

    # ---------------------------------------------------------------------------
    # 7. Loading PWL data from a file via Python
    # ---------------------------------------------------------------------------

    print()
    print("=" * 60)
    print("7. Loading PWL data from a file")
    print("=" * 60)

    # Write sample data file
    data_file = os.path.join(dir_folder, 'pwl_data.txt')
    with open(data_file, 'w') as f:
        f.write("0     0.0\n")
        f.write("10e-9 0.0\n")
        f.write("11e-9 1.8\n")
        f.write("90e-9 1.8\n")
        f.write("91e-9 0.0\n")
        f.write("100e-9 0.0\n")

    # Read two-column time/value data
    data = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                t, v = line.split()
                data.append((float(t), float(v)))

    pwl_from_file = sources.PWL(data)
    print(f"  Loaded {len(pwl_from_file)} points from {data_file}")
    print(f"  SPICE: {pwl_from_file.to_spice()}")

    print()
    print("Done.")


if __name__ == '__main__':
    main()
