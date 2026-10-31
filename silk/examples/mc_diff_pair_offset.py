"""Monte Carlo mismatch in a matched NMOS pair.

Two diode-connected NMOS devices each carry the same ideal current.
Their gate-voltage difference measures input-referred offset for a pair
with an ideal matched load. The offset bench runs mismatch samples over
the declared corners and checks the absolute offset against its limit.

Run ``MPLBACKEND=Agg .venv/bin/python -m silk.examples.mc_diff_pair_offset``.
It prints yield, offset spread, the worst sample and, when draws are
recorded, a sensitivity ranking for the first corner. It saves an HTML report
under example_output_dir(); rawfiles remain in each sample's output folder.
The default campaign has three corners and 50 samples per corner. Pass
smaller n or fewer corners to run() for a shorter campaign. Needs ngspice
and sky130 models with mismatch sections.
"""
import os

from silk import load_pdk, sources
from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.circuit.report import generate_report
from silk.examples import example_output_dir
from silk.exceptions import SimulationError
from silk.metrics import final_value
from silk.simulation import Op

CORNERS = ['tt', 'ss', 'ff']
N_SAMPLES = 50
SEED = 0
OFFSET_LIMIT_V = 4e-3
BRANCH_CURRENT = 20e-6
PAIR_W_UM, PAIR_L_UM = 5.0, 2.0


def build_diff_pair(out_dir=None, *, n=N_SAMPLES, corners=CORNERS, seed=SEED,
                    record_draws=True, limit=OFFSET_LIMIT_V):
    """Build the pair and declare a mismatch campaign over the requested corners."""
    pdk = load_pdk('sky130')
    out_dir = out_dir or example_output_dir('mc_diff_pair_offset')
    mod = Module('PDK_skywater130', out_dir, 'pair.cir',
                 pdk_details=pdk, module_name='pair')
    mod.define_module_ports({'g1': PortDir.INOUT, 'g2': PortDir.INOUT,
                             'vss': PortDir.GROUND})
    with mod:
        pdk.nmos1p8('m1', d='g1', g='g1', s='vss', b='vss',
                    w=PAIR_W_UM, l=PAIR_L_UM)
        pdk.nmos1p8('m2', d='g2', g='g2', s='vss', b='vss',
                    w=PAIR_W_UM, l=PAIR_L_UM)

    tb = mod.create_test('offset')
    tb.add_module_under_test({'g1': 'g1', 'g2': 'g2', 'vss': '0'})
    tb.add(sources.idc('i1', positive='0', negative='g1', value=BRANCH_CURRENT))
    tb.add(sources.idc('i2', positive='0', negative='g2', value=BRANCH_CURRENT))
    tb.add_sim_type(Op())

    offset = tb.probe('g1') - tb.probe('g2')
    offset_v = tb.add_metric('offset_V', final_value, sig=offset)
    tb.add_metric('abs_offset_V', abs(offset_v))
    tb.add_spec('|offset| < limit', 'abs_offset_V', '<', limit)

    tb.set_corners(list(corners))
    tb.set_monte_carlo(n, variation='mismatch', seed=seed,
                       record_draws=record_draws)
    return mod


def run(out_dir=None, *, n_workers=4, **build_kwargs):
    """Run the campaign, print the summary, save the HTML report.

    Returns ``(mc, html_path)``.
    """
    mod = build_diff_pair(out_dir, **build_kwargs)
    results = mod.run_all_tests(n_workers=n_workers)
    mc = results['pair.offset']
    if isinstance(mc, dict):

        raise SimulationError(f"pair.offset failed: {mc['error']}")

    print(mc.summary())

    print('\noffset sigma per corner:')
    for corner in sorted(mc.corners):
        sigma = mc.corner(corner).stat('offset_V', 'std')
        print(f'  {corner}: {sigma * 1e3:.3f} mV')
    worst = mc.worst('abs_offset_V')
    print(f'worst sample: {worst.value * 1e3:.3f} mV at corner '
          f'{worst.corner}, sample {worst.sample}  '
          f'(mc.rerun(({worst.corner!r}, {worst.sample})) re-simulates it)')

    # Rank within one corner to avoid correlating draws through corner shifts.
    if mc.draws is not None:
        rank_corner = mc.corners[0]
        print(f'\ntop sensitivity rows for offset_V at {rank_corner}:')
        ranking = mc.corner(rank_corner).sensitivity('offset_V')
        print(ranking.summary(top=3, scale=1e3, unit=' mV'))

    report = generate_report(mod, auto_discover_config=False)
    report.add_monte_carlo(mc)
    html_path = os.path.join(mod.dir_folder, 'pair_monte_carlo.html')
    report.save_html(html_path)
    print(f'HTML report: {html_path}')
    return mc, html_path


if __name__ == '__main__':
    run()
