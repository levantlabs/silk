"""
Construction-only Factory Example
=================================

Demonstrates device construction, parameter edits and ckt.add().

  - pdk.<device>('name', pin=net, ..., param=value)  — direct instantiation
  - ckt.add(...)                                      — unified add method
  - mn.w = value                                      — direct attribute assignment

Assembles two ground-referenced NMOS devices and a PMOS mirror load.
The input gates are undriven; this example only constructs and inspects
the structure and does not simulate it.
"""

from silk.circuit.modules import Module

from silk import load_pdk
from silk.examples import example_output_dir
from silk import sources

# ---------------------------------------------------------------------------
# PDK setup
# ---------------------------------------------------------------------------


def main():
    pdk130 = load_pdk('sky130')

    fname = 'example.cir'
    dir_folder = example_output_dir('diff_pair_construction')

    # ---------------------------------------------------------------------------
    # 1. Instantiate devices directly using the factory API
    # ---------------------------------------------------------------------------

    # Ground-referenced NMOS devices with undriven gates vinp/vinn
    mn_input_p = pdk130.nmos1p8('nmos_input_p',
                                 d='voutn', g='vinp', s='0', b='0',
                                 w=1.0, l=0.35, m=5)

    mn_input_n = pdk130.nmos1p8('nmos_input_n',
                                 d='voutp', g='vinn', s='0', b='0',
                                 w=1.0, l=0.35, m=5)

    # PMOS current-mirror load
    mp_load_p = pdk130.pmos1p8('pmos_load_p',
                                d='voutn', g='voutp', s='vdd', b='vdd',
                                w=1.0, l=0.35)

    mp_load_n = pdk130.pmos1p8('pmos_load_n',
                                d='voutp', g='voutp', s='vdd', b='vdd',
                                w=1.0, l=0.35)

    # ---------------------------------------------------------------------------
    # 2. Build circuit with ckt.add()
    # ---------------------------------------------------------------------------

    main_netlist = Module('PDK_skywater130', dir_folder, fname, pdk_details=pdk130)

    main_netlist.add(mn_input_p, mn_input_n, mp_load_p, mp_load_n)

    # ---------------------------------------------------------------------------
    # 3. Direct attribute assignment — adjust a parameter without reconstruction
    # ---------------------------------------------------------------------------

    # Example: widen NMOS pair after initial instantiation
    # mn_input_p.w = 2.0
    # mn_input_n.w = 2.0

    # ---------------------------------------------------------------------------
    # 4. Supply and structure inspection
    # ---------------------------------------------------------------------------

    main_netlist.add_source(sources.VoltageSource('supply', positive='vdd', negative='0', dc=1.8))

    main_netlist.print_elements()
    main_netlist.print_nets()


if __name__ == '__main__':
    main()
