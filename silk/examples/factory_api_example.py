"""
Example: Factory API — Direct Device Instantiation
=======================================================

Demonstrates direct device construction, attribute updates and circuit
assembly with ckt.add().

Key features shown:
  - pdk.nmos1p8('mn1', d=..., g=..., ...)  — direct device instantiation
  - sources.vdc('vdd', 'vdd', 'gnd', 1.8)   — declarative source library
  - sources.vpulse('clk', ..., v1=0, ...)   — waveform-first source
  - ckt.add(mn1, ..., vdd, ...)             — unified add() (devices AND sources)
  - elements.resistor('r1', ...)             — ideal passive elements
  - mn1.w = 2.0                              — direct attribute assignment
  - mn1.to_spice()                           — inspect SPICE line directly
"""

from silk.circuit.modules import Module
from silk import load_pdk
from silk.examples import example_output_dir
from silk.elements import elements
from silk import sources
from silk.simulation import analysis

# ---------------------------------------------------------------------------
# PDK setup
# ---------------------------------------------------------------------------


def main():
    pdk = load_pdk('sky130')

    # ---------------------------------------------------------------------------
    # 1. Direct device instantiation
    # ---------------------------------------------------------------------------

    # Each PDK device is a Python class with named constructor arguments.
    # Pins and parameters are keyword arguments — IDE autocomplete works on them.
    mn1 = pdk.nmos1p8('mn1', d='voutn', g='vinp', s='tail', b='gnd', w=4.0, l=0.15, m=2)
    mn2 = pdk.nmos1p8('mn2', d='voutp', g='vinn', s='tail', b='gnd', w=4.0, l=0.15, m=2)
    mp1 = pdk.pmos1p8('mp1', d='voutp', g='voutp', s='vdd', b='vdd', w=4.0, l=0.15, m=1)
    mp2 = pdk.pmos1p8('mp2', d='voutn', g='voutp', s='vdd', b='vdd', w=4.0, l=0.15, m=1)
    mn_tail = pdk.nmos1p8('mn_tail', d='tail', g='vbias', s='gnd', b='gnd', w=8.0, l=0.5, m=4)

    # Inspect the SPICE line for any device before adding to circuit
    print("mn1 SPICE:", mn1.to_spice().strip())

    # ---------------------------------------------------------------------------
    # 2. Direct attribute assignment — update a parameter without reconstructing
    # ---------------------------------------------------------------------------

    # After instantiation you can update parameters or nets directly:
    mn1.w = 6.0          # width changed from 4.0 to 6.0
    mn1.d = 'voutn_new'  # reconnect drain to a different net

    print("After update  mn1.params['w']:", mn1.params['w'])
    print("After update  mn1.nets['d']  :", mn1.nets['d'])

    # Reset to original for the circuit below
    mn1.w = 4.0
    mn1.d = 'voutn'

    # ---------------------------------------------------------------------------
    # 3. Ideal passive elements from ElementLibrary
    # ---------------------------------------------------------------------------

    # elements.<device>(...) — process-independent ideal passives
    r_load = elements.resistor('r_load', p='voutp', n='vdd', value=10e3)
    c_load = elements.capacitor('c_load', p='voutp', n='gnd', value=100e-15)

    # sources.<kind>(...) — the declarative source library (YAML-backed, like
    # pdk.<device>): waveform is the primary noun. Two equivalent surfaces —
    #   flat:        sources.vdc(name, p, n, value)     sources.vpulse(name, p, n, ...)
    #   namespaced:  sources.voltage.dc(...)            sources.voltage.pulse(...)
    # Terminals accept p/n or positive/negative; dc=/ac= coexist with a waveform.
    vdd    = sources.vdc('vdd',   'vdd',   'gnd', 1.8)
    vinn   = sources.vdc('vinn',  'vinn',  'gnd', 0.9)
    vinp   = sources.vdc('vinp',  'vinp',  'gnd', 0.9)
    vbias  = sources.vdc('vbias', 'vbias', 'gnd', 0.6)

    print("Resistor SPICE:", r_load.to_spice().strip())
    print("VSource  SPICE:", vdd.to_spice().strip())
    # a waveform source, waveform-first (not built into the circuit — just to show):
    _clk = sources.vpulse('clk', 'clk', 'gnd', v1=0, v2=1.8,
                          td='1n', tr='1n', tf='1n', pw='5n', per='10n')
    print("Pulse    SPICE:", _clk.to_spice().strip())

    # ---------------------------------------------------------------------------
    # 4. Build the circuit with ckt.add()
    # ---------------------------------------------------------------------------

    dir_folder = example_output_dir('factory_api')
    fname      = 'diffpair_factory.cir'

    ckt = Module('PDK_skywater130', dir_folder, fname, pdk_details=pdk)

    # Add all devices in one call; each instance carries its name
    ckt.add(mn1, mn2, mp1, mp2, mn_tail,
            r_load, c_load,
            vdd, vinn, vinp, vbias)

    print("\nElements in circuit:", list(ckt.elem_dict.keys()))
    print("Nets auto-registered:", sorted(ckt.net_dict.keys()))

    # ---------------------------------------------------------------------------
    # 5. Add simulation and generate netlist
    # ---------------------------------------------------------------------------

    ckt.add_sim_type(analysis.Transient(step=1e-9, stop=50e-9))

    ckt.generate_netlist(corner='tt')

    print("\nNetlist written.  Run ckt.simulate() to invoke ngspice.")
    print("(Requires ngspice + SkyWater PDK installed at PDK_ROOT)")


if __name__ == '__main__':
    main()
