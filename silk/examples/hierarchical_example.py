"""
Hierarchical Circuit Example
=============================

Demonstrates multi-level subcircuit hierarchy using silk's Module class.

Hierarchy:
  inverter      — CMOS inverter (1 NMOS + 1 PMOS)
  buffer        — two inverters in series
  top-level     — uses buffer subcircuit with sources and simulation

This shows how to:
  1. Define a subcircuit with module_name and define_module_ports()
  2. Instantiate a subcircuit inside another subcircuit (nesting)
  3. Instantiate subcircuits in the top-level netlist via add_module()
  4. Generate and inspect the hierarchical SPICE netlist
"""

from silk.circuit.modules import Module
from silk.circuit.circuit_elements import PortDir
from silk import load_pdk
from silk.examples import example_output_dir
from silk import sources
from silk.elements import elements
from silk.simulation import analysis

# ---------------------------------------------------------------------------
# PDK setup
# ---------------------------------------------------------------------------
pdk130 = load_pdk('sky130')

dir_folder = example_output_dir('hierarchical', create=False)
fname = 'hierarchical_demo.cir'


# ===========================================================================
# Level 1: CMOS Inverter subcircuit
# ===========================================================================

def build_inverter(pdk_inst, dir_folder, fname, w_n=1.0, w_p=2.0, l=0.15):
    """Build a CMOS inverter subcircuit.

    Ports: vin, vout, vdd, vss
    """
    inv = Module('PDK_skywater130', dir_folder, fname,
                 pdk_details=pdk_inst, module_name='inverter')
    inv.define_module_ports({
        'vin':  PortDir.IN,
        'vout': PortDir.OUT,
        'vdd':  PortDir.POWER,
        'vss':  PortDir.GROUND,
    })

    mn = pdk_inst.nmos1p8('mn', d='vout', g='vin', s='vss', b='vss',
                           w=w_n, l=l)
    mp = pdk_inst.pmos1p8('mp', d='vout', g='vin', s='vdd', b='vdd',
                           w=w_p, l=l)
    inv.add(mn, mp)

    return inv


# ===========================================================================
# Level 2: Buffer subcircuit (two inverters)
# ===========================================================================

def build_buffer(pdk_inst, dir_folder, fname, w_n=1.0, w_p=2.0, l=0.15):
    """Build a buffer (two cascaded inverters).

    Ports: vin, vout, vdd, vss
    Internal net: vmid (output of first inverter)
    """
    buf = Module('PDK_skywater130', dir_folder, fname,
                 pdk_details=pdk_inst, module_name='buffer')
    buf.define_module_ports({
        'vin':  PortDir.IN,
        'vout': PortDir.OUT,
        'vdd':  PortDir.POWER,
        'vss':  PortDir.GROUND,
    })

    # Build the inverter subcircuit (reused for both instances)
    inv_subckt = build_inverter(pdk_inst, dir_folder, fname, w_n, w_p, l)

    # First inverter: vin → vmid
    buf.add_module(inv_subckt, 'inv1', nets={
        'vin':  'vin',
        'vout': 'vmid',
        'vdd':  'vdd',
        'vss':  'vss',
    })

    # Second inverter: vmid → vout
    buf.add_module(inv_subckt, 'inv2', nets={
        'vin':  'vmid',
        'vout': 'vout',
        'vdd':  'vdd',
        'vss':  'vss',
    })

    return buf


# ===========================================================================
# Top-level: testbench with buffer
# ===========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Hierarchical Circuit Example")
    print("  inverter → buffer → top-level testbench")
    print("=" * 60)

    # Build subcircuits
    inverter = build_inverter(pdk130, dir_folder, fname)
    buffer   = build_buffer(pdk130, dir_folder, fname)

    print(f"\n  Inverter subcircuit: '{inverter.module_name}'")
    print(f"    Devices: {list(inverter.elem_dict.keys())}")

    print(f"\n  Buffer subcircuit: '{buffer.module_name}'")
    print(f"    Submodules: {list(buffer.module_dict.keys())}")

    # Top-level netlist (no module_name → generates full simulation netlist)
    top = Module('PDK_skywater130', dir_folder, fname, pdk_details=pdk130)

    # Instantiate buffer
    top.add_module(buffer, 'buf1', nets={
        'vin':  'vin',
        'vout': 'vout',
        'vdd':  'vdd',
        'vss':  '0',
    })

    # Add a second buffer in series to demonstrate reuse
    top.add_module(buffer, 'buf2', nets={
        'vin':  'vout',
        'vout': 'vout2',
        'vdd':  'vdd',
        'vss':  '0',
    })

    # Sources
    top.add_source(sources.VoltageSource(
        'supply', positive='vdd', negative='0', dc=1.8))

    # Input: PWL step waveform
    pwl_input = (sources.PWL()
                 .add(0, 0)
                 .add(1e-9, 0)
                 .add(2e-9, 1.8)
                 .add(50e-9, 1.8)
                 .add(51e-9, 0)
                 .add(100e-9, 0))
    top.add_source(sources.VoltageSource(
        'vin_src', positive='vin', negative='0', pwl=pwl_input))

    # Load capacitors
    top.add(elements.capacitor('cload1', p='vout',  n='0', value=10e-15))
    top.add(elements.capacitor('cload2', p='vout2', n='0', value=10e-15))

    # Simulation
    top.add_sim_type(analysis.Transient(step=10e-12, stop=100e-9))

    # Print structure
    print("\n  Top-level elements:")
    top.print_elements()
    print("\n  Top-level nets:")
    top.print_nets()

    # Generate netlist
    netlist_path = top.generate_netlist(corner='tt')
    print(f"\n  Netlist written to: {netlist_path}")

    # Show the generated netlist
    print("\n" + "=" * 60)
    print("  Generated SPICE netlist:")
    print("=" * 60)
    with open(netlist_path) as f:
        print(f.read())

    # Run simulation (requires ngspice and installed sky130 models)
    print("=" * 60)
    print("  Running simulation...")
    print("=" * 60)
    top.simulate(corner='tt')

    # One-line plot — no matplotlib boilerplate needed
    top.plot('vin', 'vout', 'vout2')

    print("\nDone.")
