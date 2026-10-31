"""Run the sampling-buffer design loop with the preview GF180MCU integration.

Use 3.3 V devices and the shared ``design_loop_walkthrough`` runner. The
transient tests measure settling under nominal and hot conditions. AC tests
measure open-loop gain and bandwidth before and after estimated parasitics.
Store evidence, rebuild the circuit, restore currents and size routing wires.
Print the measurements, spec checks and artifact paths.

The author writes dimensions in microns; the models use metres. Step 0 checks
that emitted netlist dimensions, the geometry accessor and layout estimates
agree for both construction styles.

Needs ngspice and GF180MCU models under ``PDK_ROOT``; see
``docs/installing-the-pdk.md``. Run::

    python -m silk.examples.design_loop_walkthrough_gf180mcu [OUT_DIR]

Artifacts default to ``silk-output/design_loop_gf180mcu/``, subject to
``SILK_EXAMPLES_OUT``.
"""
import os
import re
import sys

from silk.circuit.modules import Module
from silk.circuit.ports import Port
from silk.examples import design_loop_walkthrough as dl
from silk.layout import estimate_layout
from silk.utils import parse_spice_value
from silk.examples import example_output_dir


#: Device names and corner spelling from the catalog; assume a 5 nm width grid.
GF180MCU = dl.Technology(pdk='gf180mcu', label='PDK_gf180mcu', nmos='nmos3p3',
                         pmos='pmos3p3', corner='typical', width_grid_um=0.005)

#: Supply, gain and bandwidth targets for the sampling buffer.
INTENT = dict(
    dl.INTENT,
    vdd_v=3.3,
    hot_vdd_v=2.97,              # -10 % supply
    gain_db_min=30.0,            # open-loop DC gain, light load
    bw_hz_min=40e6,              # open-loop -3 dB bandwidth, light load
)

_NETLIST_W = re.compile(r'\bw=([0-9.eE+-]+)([a-zA-Z]*)')


def build_add_element_twin(knobs, *, work_dir):
    """Build the six-device buffer using explicit Module.add calls; dimensions are microns."""
    pdk = dl.get_pdk(GF180MCU)
    nmos, pmos = getattr(pdk, GF180MCU.nmos), getattr(pdk, GF180MCU.pmos)
    mod = Module(GF180MCU.label, work_dir, 'sampling_buffer_twin.cir',
                 pdk_details=pdk, module_name='sampling_buffer')
    mod.define_module_ports({
        'VDD': Port.supply(), 'VSS': Port.ground(),
        'VINP': Port.analog_input(), 'VINM': Port.analog_input(),
        'VOUT': Port.analog_output(), 'IBIAS': Port.analog_input(),
    })
    for net in ('vint', 'tail'):
        mod.add_net(net)
    devices = (
        ('M1', nmos, dict(d='vint', g='VINP', s='tail', b='VSS'),
         dict(w=knobs['w_in_um'], l=dl.L_INPUT_UM)),
        ('M2', nmos, dict(d='VOUT', g='VINM', s='tail', b='VSS'),
         dict(w=knobs['w_in_um'], l=dl.L_INPUT_UM)),
        ('M3', pmos, dict(d='vint', g='vint', s='VDD', b='VDD'),
         dict(w=knobs['w_load_um'], l=dl.L_LOAD_UM)),
        ('M4', pmos, dict(d='VOUT', g='vint', s='VDD', b='VDD'),
         dict(w=knobs['w_load_um'], l=dl.L_LOAD_UM)),
        ('M5', nmos, dict(d='tail', g='IBIAS', s='VSS', b='VSS'),
         dict(w=knobs['w_tail_um'], l=dl.L_TAIL_UM)),
        ('M6', nmos, dict(d='IBIAS', g='IBIAS', s='VSS', b='VSS'),
         dict(w=knobs['w_ref_um'], l=dl.L_TAIL_UM)),
    )
    for name, cls, nets, params in devices:
        mod.add(cls(name, **nets, **params))
    return mod


def prove_geometry_units(knobs, work_dir):
    """Compare emitted dimensions, geometry accessors and layout areas; return per-device readings."""
    pdk = dl.get_pdk(GF180MCU)
    factory, _tests = dl.build_sampling_buffer(
        knobs, tech=GF180MCU, intent=INTENT,
        work_dir=os.path.join(work_dir, 'factory'))
    twin = build_add_element_twin(knobs, work_dir=os.path.join(work_dir, 'twin'))
    est_f = estimate_layout(factory, dl.MET1)
    est_t = estimate_layout(twin, dl.MET1)
    packing = est_f.packing_factor

    # The netlist is the one reading that is in the PDK's unit: GF180MCU's
    # models want metres, and the emitted `w=15.35u` says so in its suffix.
    lines = {line.split()[0]: line for line in factory.generate_netlist().splitlines()
             if line.startswith('XMM')}
    expected_w = {'M1': 'w_in_um', 'M2': 'w_in_um', 'M3': 'w_load_um',
                  'M4': 'w_load_um', 'M5': 'w_tail_um', 'M6': 'w_ref_um'}
    expected_l = {'M1': dl.L_INPUT_UM, 'M2': dl.L_INPUT_UM, 'M3': dl.L_LOAD_UM,
                  'M4': dl.L_LOAD_UM, 'M5': dl.L_TAIL_UM, 'M6': dl.L_TAIL_UM}

    print(f'PDK {pdk.pdk_name}: author writes µm (manifest user_units), models '
          f'take metres (catalog pdk_unit: base); packing factor {packing}')
    print(f'{"dev":<4} {"netlist w=":>12} {"= metres":>10} | {"accessor W×L×M (µm)":>22} '
          f'| {"est. area":>9} {"hand W·L·M·pack":>15} | Module.add twin')
    out = {}
    for name in sorted(expected_w):
        number, suffix = _NETLIST_W.search(lines['XM' + name]).groups()
        w_m = parse_spice_value(number + suffix)
        dev_f = factory.elem_dict[name]['device']
        dev_t = twin.elem_dict[name]['device']
        geom_f = pdk.device_geometry_um(dev_f)
        geom_t = pdk.device_geometry_um(dev_t)
        W_um, L_um, M = geom_f
        hand = knobs[expected_w[name]] * expected_l[name] * 1 * packing
        area_f, area_t = est_f.devices[name].area, est_t.devices[name].area
        agree = (abs(area_f - area_t) < 1e-9 and geom_f == geom_t)
        out[name] = dict(netlist_w_m=w_m, accessor_um=geom_f, twin_um=geom_t,
                         area_um2=area_f, twin_area_um2=area_t, hand_um2=hand)
        print(f'{name:<4} {number + suffix:>12} {w_m:>10.3e} | '
              f'{W_um:>7.3f}×{L_um:<5.3f}×{M:<4.0f}     | {area_f:>9.3f} '
              f'{hand:>15.3f} | {"agrees" if agree else "DISAGREES"} ({area_t:.3f})')
        # Check each printed geometry against the sizing inputs.
        assert abs(w_m - W_um * 1e-6) < 1e-15, (name, w_m, W_um)
        assert abs(W_um - knobs[expected_w[name]]) < 1e-9, (name, geom_f)
        assert abs(L_um - expected_l[name]) < 1e-9, (name, geom_f)
        assert abs(area_f - hand) < 1e-9 * max(1.0, hand), (name, area_f, hand)
        assert agree, (name, geom_f, geom_t, area_f, area_t)
    assert est_f.bounding_box == est_t.bounding_box, (est_f.bounding_box,
                                                      est_t.bounding_box)
    print(f'module bounding box {est_f.bounding_box[0]:.2f} × '
          f'{est_f.bounding_box[1]:.2f} µm, both styles')
    print('(a sky130 device with the same W/L in µm has the same footprint: the '
          'estimator reads canonical microns from the accessor, not the '
          "PDK's native unit)")
    return out


def main(out_dir=None):
    out_dir = out_dir or example_output_dir('design_loop_gf180mcu')
    os.makedirs(out_dir, exist_ok=True)

    dl._h('0. Units: metre-native models, micron author, one geometry accessor')
    knobs = dl.size_from_intent(INTENT, GF180MCU)
    geometry = prove_geometry_units(knobs, os.path.join(out_dir, 'work', 'units'))

    # The shared runner applies INTENT temperatures to the simulator as well
    # as declaring them in report metadata; supplies use the same intent.
    summary = dl.main(out_dir, tech=GF180MCU, intent=INTENT)
    summary['geometry'] = geometry
    return summary


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
