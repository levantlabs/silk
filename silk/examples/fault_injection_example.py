"""Fault-injection example: flash ADC under transistor faults.

The campaign applies two targeted faults, independent 2% width variation on
every comparator tail transistor, and sparse stuck-open faults. A fixed seed
reproduces the applied fault list. Each run records fault application and
simulator completion; this example does not score ADC behavior. Waveforms
are discarded. Netlists, manifests and summary reports are retained.

Run:
    python -m silk.examples.fault_injection_example --n-runs 1 --n-workers 1
    python -m silk.examples.fault_injection_example --dry-run

The default campaign uses 20 runs and four workers. SPICE runs need ngspice
and sky130 models. Dry-run prints one realized fault list without simulation.
Reports are written under example_output_dir('fault_campaign') unless
--out-dir is supplied. --no-spice exits with an explanatory message.
"""

from __future__ import annotations

import argparse
import os
import sys
from silk.examples import example_output_dir



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--n-runs', type=int, default=20,
                        help='Number of Monte Carlo runs (default: 20).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Campaign seed (default: 42).')
    parser.add_argument('--n-workers', type=int, default=4,
                        help='Parallel workers (default: 4).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Build the fault set and print the realized '
                             'fault list for one seed; do not run ngspice.')
    parser.add_argument('--no-spice', action='store_true',
                        help='(Not supported.) Fault injection is SPICE-only '
                             '— behavioral models do not carry '
                             'transistor-level fault semantics.')
    # Resolve the campaign output path without creating it here.
    parser.add_argument('--out-dir',
                        default=example_output_dir('fault_campaign',
                                                   create=False),
                        help='Output directory for the campaign.')
    args = parser.parse_args()
    if args.n_runs <= 0:
        parser.error("--n-runs must be positive")
    if args.n_workers <= 0:
        parser.error("--n-workers must be positive")

    if args.no_spice:
        print('Fault injection is SPICE-only. The flash ADC\'s '
              'behavioral models do not carry transistor-level fault '
              'semantics. Install ngspice and run without --no-spice.',
              file=sys.stderr)
        return 2

    import numpy as np

    from silk.examples.flash_adc_example import make_flash_adc, T_STEP, CLK_PER
    from silk.faults import (
        FaultCampaign,
        FaultSet,
        stuck_open,
        vth_shift,
        w_variation,
    )
    from silk.simulation.analysis import Transient

    print('Building flash ADC ...', flush=True)
    adc, _ = make_flash_adc()
    # Register four clock cycles per fault run.
    adc.add_sim_type(Transient(step=T_STEP, stop=4 * CLK_PER))

    fs = FaultSet([
        stuck_open('xcomp1.mn1'),
        vth_shift('xcomp5.mn2', delta=0.05),
    ])
    fs.sprinkle(
        scope=r'xcomp\d+\.mn5',
        fault=w_variation(sigma=0.02),
        rate=1.0,
    )
    fs.sprinkle(
        scope=r'xcomp\d+\.(mn|mp)\w+',
        fault=stuck_open(),
        rate=0.01,
    )

    if args.dry_run:
        rng = np.random.default_rng(args.seed)
        realized = fs.realize(adc, rng)
        print(f'\nDry run at seed={args.seed} — {len(realized)} realized faults:')
        for f in realized:
            kv = ', '.join(f'{k}={v}' for k, v in sorted(f.params.items()))
            print(f'  {f.fault_type:<14} {f.target:<28} {{{kv}}}')
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'\nLaunching campaign: n_runs={args.n_runs}  '
          f'n_workers={args.n_workers}  seed={args.seed}', flush=True)
    print(f'Artefacts: {args.out_dir}/\n', flush=True)

    campaign = FaultCampaign(
        adc, fs,
        n_runs=args.n_runs,
        seed=args.seed,
        out_dir=args.out_dir,
        n_workers=args.n_workers,
        archive_waveforms=False,
        reconcile_specs=False,
        progress=True,
    )
    result = campaign.run()

    report = result.generate_report()
    print('\n' + report.summary())

    md_path = os.path.join(args.out_dir, 'summary.md')
    html_path = os.path.join(args.out_dir, 'summary.html')
    json_path = os.path.join(args.out_dir, 'summary.json')
    report.save(md_path)
    report.save_html(html_path)
    report.save_json(json_path)
    print(f'Wrote: {md_path}')
    print(f'Wrote: {html_path}')
    print(f'Wrote: {json_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
