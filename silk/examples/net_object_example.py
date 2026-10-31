"""Inspect Net metadata, pin connectivity and automatic rewiring.

The example builds a CMOS inverter, rewires its NMOS drain, then assigns
illustrative voltage arrays to Net.voltage. Assigning a device pin updates
its connection and removes the old entry from Net.pins. The arrays are not
simulation results.

Run: python -m silk.examples.net_object_example
The script prints net metadata, connections and stored voltage values.
It uses the packaged sky130 catalog and needs no simulator or installed models.
"""

from __future__ import annotations


import numpy as np

from silk.circuit.modules import Module
from silk.circuit.net import Net, PortBinding
from silk import load_pdk
from silk import sources
from silk.examples import example_output_dir

# ---------------------------------------------------------------------------
# PDK setup
# ---------------------------------------------------------------------------


def main():
    out_dir = example_output_dir()

    pdk = load_pdk('sky130')

    # ---------------------------------------------------------------------------
    # Build a simple CMOS inverter so we have nets with real connectivity
    # ---------------------------------------------------------------------------

    ckt = Module('PDK_skywater130', out_dir, 'net_object_example.cir',
                 pdk_details=pdk)

    mn1 = pdk.nmos1p8('mn1', d='vout', g='vin', s='gnd', b='gnd', w=1.0, l=0.15)
    mp1 = pdk.pmos1p8('mp1', d='vout', g='vin', s='vdd', b='vdd', w=2.0, l=0.15)
    ckt.add(mn1, mp1)

    ckt.add_source(sources.VoltageSource('vdd_supply',
                                        positive='vdd', negative='gnd', dc=1.8))
    ckt.add_source(sources.VoltageSource('vin_src',
                                        positive='vin', negative='gnd', dc=0.9))

    print("=" * 64)
    print("1. Inspect Net objects in net_dict")
    print("=" * 64)
    vout_net = ckt.net_dict['vout']
    print(f"  type(ckt.net_dict['vout'])  = {type(vout_net).__name__}")
    print(f"  isinstance(.., Net)          = {isinstance(vout_net, Net)}")
    print(f"  vout_net.name                = {vout_net.name!r}")
    print(f"  vout_net.module is ckt       = {vout_net.module is ckt}")

    # ---------------------------------------------------------------------------
    # 2. net.pins reflects current connectivity — filled automatically
    # ---------------------------------------------------------------------------

    print()
    print("=" * 64)
    print("2. net.pins lists every pin bound to the net")
    print("=" * 64)
    for net_name in ('vout', 'vin', 'vdd', 'gnd'):
        net = ckt.net_dict[net_name]
        print(f"  {net_name:>5s}  fanout={net.fanout}")
        for b in net.pins:
            kind_tag = f"[{b.kind}]"
            print(f"         {kind_tag:<10s} {b.instance}.{b.port}")

    # ---------------------------------------------------------------------------
    # 3. Rewire a device pin and inspect Net.pins
    # ---------------------------------------------------------------------------

    print()
    print("=" * 64)
    print("3. Rewire the NMOS drain with `mn1.d = 'vout_new'`")
    print("=" * 64)

    print("  BEFORE rewire:")
    print(f"    vout.pins     = {[(b.instance, b.port) for b in ckt.net_dict['vout'].pins]}")

    # Assigning a device pin updates its connection and removes the old
    # entry from Net.pins.
    mn1.d = 'vout_new'

    vout_after     = ckt.net_dict['vout']
    vout_new_after = ckt.net_dict['vout_new']

    print("  AFTER rewire:")
    print(f"    vout.pins     = {[(b.instance, b.port) for b in vout_after.pins]}")
    print(f"    vout_new.pins = {[(b.instance, b.port) for b in vout_new_after.pins]}")
    print()
    print("  The stale (mn1,d) binding was removed from 'vout' automatically and")
    print("  a fresh one was appended to 'vout_new'. The device's own `mn1.nets['d']`")
    print("  was also updated.")
    print(f"    mn1.nets['d']  = {mn1.nets['d']!r}")

    # Restore the inverter connection before assigning illustrative voltages.
    mn1.d = 'vout'

    # Rewire factory devices with pin assignment, as shown above.
    # Rewire a placed module's ports with ckt.connect_module(name, port=net).

    # ---------------------------------------------------------------------------
    # 4. Store illustrative voltage arrays on the nets
    # ---------------------------------------------------------------------------

    print()
    print("=" * 64)
    print("4. net.voltage stored-field")
    print("=" * 64)

    # An unassigned voltage field returns None.
    print(f"  Before assignment:  ckt.net_dict['vout'].voltage = "
          f"{ckt.net_dict['vout'].voltage}")

    # Store illustrative arrays without running a simulation.
    t = np.linspace(0, 10e-9, 11)
    _fake_signals = [
        ('vout', np.sin(2 * np.pi * 1e8 * t) * 0.9 + 0.9),
        ('vin',  np.ones_like(t) * 0.9),
    ]
    for _name, _arr in _fake_signals:
        ckt.net_dict[_name].voltage = _arr

    vout_sig = ckt.net_dict['vout'].voltage
    print(f"  After assigning illustrative voltage arrays:")
    print(f"    ckt.net_dict['vout'].voltage[:3] = "
          f"{None if vout_sig is None else vout_sig[:3]}")
    print(f"    ckt.net_dict['vin'].voltage[:3]  = "
          f"{ckt.net_dict['vin'].voltage[:3]}")
    print(f"    ckt.net_dict['gnd'].voltage       = "
          f"{ckt.net_dict['gnd'].voltage}  (never assigned)")

    # ---------------------------------------------------------------------------
    # 5. Building a PortBinding by hand (for tools that annotate connectivity)
    # ---------------------------------------------------------------------------

    print()
    print("=" * 64)
    print("5. PortBinding is the dataclass stored in Net.pins")
    print("=" * 64)

    b = PortBinding(
        net='vout', instance='mn1', port='d',
        direction=None, role=None, kind='device',
    )
    print(f"  example binding: {b}")
    print()
    print("  The 'kind' field is used by checkers to filter submodule-only")
    print("  bindings from device / passive / source bindings on the same net.")

    print()
    print("Done.")


if __name__ == '__main__':
    main()
