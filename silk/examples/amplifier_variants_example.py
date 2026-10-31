"""Place three common-source amplifiers at two different sizings.

Each stage has an NMOS input, a PMOS load with an external gate bias,
and a load capacitor. build_cs_amp() accepts dimensions and capacitance
as keyword arguments. With variants=True, distinct definitions receive
hash-suffixed subcircuit names; identical builds share one definition.
The suffix also appears for a single placement.

Run ``.venv/bin/python -m silk.examples.amplifier_variants_example``.
It prints definitions and instances, then the CircuitError raised when
different definitions share a name without variants=True. Netlists go under
example_output_dir(). Only the packaged sky130 catalog is needed; no
simulation runs.
"""
import re

from silk import load_pdk
from silk import sources
from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk.elements import elements
from silk.examples import example_output_dir
from silk.exceptions import CircuitError


_W_IN, _L_IN = 4.0, 0.5
_W_LOAD, _L_LOAD = 8.0, 0.5
_C_LOAD = 0.1e-12

_VDD = 1.8
_VBP = 1.0
_VIN_DC = 0.8


SETTING_A = dict(w_in=4.0, c_load=0.1e-12)
SETTING_B = dict(w_in=8.0, c_load=0.3e-12)

_MODULE_NAME = 'cs_amp'
_TOP_NETLIST = 'amp_variants.cir'
_CELL_NETLIST = 'cs_amp.cir'


def build_cs_amp(pdk, *, w_in=_W_IN, l_in=_L_IN, w_load=_W_LOAD, l_load=_L_LOAD,
                 c_load=_C_LOAD, variants=True) -> Module:
    """Build one common-source amplifier with declared ports.

    Widths and lengths are in microns; c_load is in farads.
    variants=True permits different definitions with the same module name.
    """
    mod = Module('PDK_skywater130', example_output_dir('amplifier_variants'),
                 _CELL_NETLIST, pdk_details=pdk, module_name=_MODULE_NAME,
                 variants=variants)
    with mod:
        pdk.nmos1p8('MN', d='VOUT', g='VIN', s='VSS', b='VSS',
                    w=w_in, l=l_in, m=1)
        pdk.pmos1p8('MP', d='VOUT', g='VBP', s='VDD', b='VDD',
                    w=w_load, l=l_load, m=1)
        elements.capacitor('CL', p='VOUT', n='VSS', value=c_load)
    mod.define_module_ports({
        'VDD': Port.supply(), 'VSS': Port.ground(),
        'VIN': Port.analog_input(), 'VBP': Port.analog_input(),
        'VOUT': Port.analog_output(),
    })
    return mod


def build_top(pdk, placements, *, variants=True) -> Module:
    """Top-level circuit placing one amplifier per ``(name, knobs)`` entry.

    Each placement is a fresh :func:`build_cs_amp` call, so two entries with
    the same knobs are two independent builds with identical bodies.
    """
    top = Module('PDK_skywater130', example_output_dir('amplifier_variants'),
                 _TOP_NETLIST, pdk_details=pdk)
    top.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=_VDD))
    top.add_source(sources.VoltageSource('vbp', positive='vbp', negative='0', dc=_VBP))
    for i, (name, knobs) in enumerate(placements):
        top.add_module(build_cs_amp(pdk, variants=variants, **knobs), name,
                       VDD='vdd', VSS='0', VIN=f'in{i}', VBP='vbp',
                       VOUT=f'out{i}')
        top.add_source(sources.VoltageSource(f'vin{i}', positive=f'in{i}',
                                             negative='0', dc=_VIN_DC))
    return top


def emit_netlist(top) -> str:
    """Write the top-level deck and return its text."""
    deck_path = top.generate_netlist()
    with open(deck_path, encoding='utf-8') as f:
        return f.read()


def subckt_blocks(text):
    """``(name, body_lines)`` for each ``.subckt`` definition in the deck."""
    blocks = []
    for m in re.finditer(r'^\.subckt\s+(\S+).*?\n(.*?)^\.ends', text,
                         re.IGNORECASE | re.MULTILINE | re.DOTALL):
        body = [ln for ln in m.group(2).splitlines()
                if ln.strip() and not ln.startswith('*')]
        blocks.append((m.group(1), body))
    return blocks


def top_instances(text):
    """The top-level X lines: after the last ``.ends``, so transistor cards
    inside a definition (also spelled ``X...`` for sky130) are skipped.
    """
    top = text[text.lower().rfind('.ends'):]
    return [line for line in top.splitlines() if line.startswith('x')]


def main():
    pdk = load_pdk('sky130')

    print('=' * 60)
    print('Step 1 — Three placements, two settings, one repeat')
    print('=' * 60)
    placements = [('xa', SETTING_A), ('xb', SETTING_B), ('xc', SETTING_A)]
    text = emit_netlist(build_top(pdk, placements))
    print('definitions (device lines show which values differ):')
    for name, body in subckt_blocks(text):
        print(f'  .subckt {name}')
        for line in body:
            print(f'    {line}')
    print('instances:')
    for line in top_instances(text):
        print(f'  {line}')
    print()
    print('xa and xc are two builds with identical bodies, so they share one')
    print('definition and one suffix. xb has different W and C, so it gets its')
    print('own. Its emitted name is '
          f'{_MODULE_NAME}_<8 hex of the definition hash>.')
    print()

    print('=' * 60)
    print('Step 2 — A single placement is suffixed too')
    print('=' * 60)
    text = emit_netlist(build_top(pdk, [('x1', SETTING_A)]))
    for name, _ in subckt_blocks(text):
        print(f'  .subckt {name}')
    print()

    print('=' * 60)
    print('Step 3 — The same placements without variants=True')
    print('=' * 60)
    try:
        emit_netlist(build_top(pdk, placements, variants=False))
    except CircuitError as exc:
        print(f'  CircuitError: {exc}')
    else:
        raise RuntimeError('variants=False placements should have clashed')


if __name__ == '__main__':
    main()
