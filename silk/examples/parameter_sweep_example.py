"""
Parameter Sweep and Design-of-Experiments Example
==================================================

Demonstrates how to use Variable and VariableSpace to explore a
circuit's design space — without requiring ngspice to be installed.

Variable objects can be passed directly as parameter values at
instantiation time.  When space.apply(point) updates a variable's .value,
the next generate_netlist() call picks up the new value automatically —
no re-wiring needed.

What this example shows:
  1. Passing Variable objects directly to the factory constructor
  2. Creating a VariableSpace from those variables
  3. Comparing four sampling strategies: random, grid, LHS, Halton
  4. Sweeping through sampled points (apply → generate_netlist)
  5. Displaying device cards with their current parameter values
  6. Visualising sampling coverage (requires matplotlib)

The circuit is a simple CMOS inverter. The design variables are:
  w_n : NMOS width  [0.42 µm – 20 µm]
  w_p : PMOS width  [0.42 µm – 20 µm]
  l   : shared channel length [0.15 µm – 1 µm]
"""

from silk.examples import example_output_dir

from silk import load_pdk
from silk.circuit.modules import Module
from silk.optimize import VariableSpace
from silk import sources
from silk.simulation import analysis

# ---------------------------------------------------------------------------
# PDK setup
# ---------------------------------------------------------------------------


def main():
    pdk = load_pdk('sky130')

    # ---------------------------------------------------------------------------
    # Step 1 — Define design variables with bounds
    # ---------------------------------------------------------------------------

    print('=' * 60)
    print('Step 1 — Define design variables')
    print('=' * 60)

    w_n = pdk.nmos1p8.w.variable(name='w_n', value=1.0,  bounds=(0.42, 20.0))
    w_p = pdk.pmos1p8.w.variable(name='w_p', value=2.0,  bounds=(0.42, 20.0))
    l   = pdk.nmos1p8.l.variable(name='l',   value=0.15, bounds=(0.15,  1.0))

    print(f'  w_n : {w_n.bounds[0]} – {w_n.bounds[1]} µm  (NMOS width)')
    print(f'  w_p : {w_p.bounds[0]} – {w_p.bounds[1]} µm  (PMOS width)')
    print(f'  l   : {l.bounds[0]}  – {l.bounds[1]}  µm  (shared length)')

    # ---------------------------------------------------------------------------
    # Step 2 — Build circuit with Variable objects passed directly
    # ---------------------------------------------------------------------------

    print()
    print('=' * 60)
    print('Step 2 — Build circuit and link variables')
    print('=' * 60)

    # Variable objects are passed directly as parameter values.
    # to_spice() reads .value at netlist-generation time, so space.apply(point)
    # updates the variable and the next generate_netlist() picks it up automatically.
    tmp = example_output_dir('parameter_sweep')
    ckt = Module('PDK_skywater130', tmp, 'inverter.cir', pdk_details=pdk)

    mn = pdk.nmos1p8('mn', d='out', g='in', s='gnd', b='gnd', w=w_n, l=l)
    mp = pdk.pmos1p8('mp', d='out', g='in', s='vdd', b='vdd', w=w_p, l=l)

    ckt.add(mn, mp)
    ckt.add_source(sources.VoltageSource('vdd_sup', positive='vdd', negative='gnd', dc=1.8))
    ckt.add_source(sources.VoltageSource('vin',     positive='in',  negative='gnd',
                                        pulse=sources.Pulse(v1=0, v2=1.8, td=0,
                                                            tr=100e-12, tf=100e-12,
                                                            pw=5e-9, per=10e-9)))
    ckt.add_sim_type(analysis.Transient(step=10e-12, stop=30e-9))

    print('  mn.w → w_n,  mn.l → l')
    print('  mp.w → w_p,  mp.l → l  (l is shared)')

    # ---------------------------------------------------------------------------
    # Step 3 — Create VariableSpace
    # ---------------------------------------------------------------------------

    print()
    print('=' * 60)
    print('Step 3 — Create VariableSpace')
    print('=' * 60)

    space = VariableSpace([w_n, w_p, l])
    print(f'  Variables : {space.names}')
    print(f'  Bounds    : {space.to_bounds()}')

    # ---------------------------------------------------------------------------
    # Step 4 — Compare sampling strategies
    # ---------------------------------------------------------------------------

    print()
    print('=' * 60)
    print('Step 4 — Compare sampling strategies (12 random/LHS/Halton points; 27 grid points)')
    print('=' * 60)

    n = 12
    methods = ['random', 'lhs', 'halton', 'grid']
    samples = {}

    for method in methods:
        if method == 'grid':
            # grid means n points per variable → n^3 total; use n=3 to keep it small
            pts = space.sample(3, method='grid')
            label = f'grid (3^3={len(pts)} total)'
        else:
            pts = space.sample(n, method=method)
            label = f'{method} ({len(pts)} points)'
        samples[method] = pts
        print(f'  {label:<30}  first point: { {k: f"{v:.3f}" for k,v in list(pts[0].items())} }')

    # ---------------------------------------------------------------------------
    # Step 5 — Parameter sweep: generate a netlist for each LHS point
    # ---------------------------------------------------------------------------

    print()
    print('=' * 60)
    print('Step 5 — Parameter sweep using LHS (generate netlist per point)')
    print('=' * 60)

    print(f'  {"#":<4}  {"w_n (µm)":<12}  {"w_p (µm)":<12}  {"l (µm)":<10}  netlist fragment')
    print(f'  {"-"*4}  {"-"*12}  {"-"*12}  {"-"*10}  {"-"*30}')

    lhs_points = samples['lhs']
    for i, pt in enumerate(lhs_points):
        space.apply(pt)
        ckt.generate_netlist(corner='tt')
        # Render the device directly with its current parameter values.
        mn_line = mn.to_spice().strip()
        print(f'  {i+1:<4}  {pt["w_n"]:<12.3f}  {pt["w_p"]:<12.3f}  {pt["l"]:<10.3f}  {mn_line[:50]}')

    # ---------------------------------------------------------------------------
    # Step 6 — Verify shared variable propagates to both devices
    # ---------------------------------------------------------------------------

    print()
    print('=' * 60)
    print('Step 6 — Shared variable: l controls both mn and mp')
    print('=' * 60)

    space.apply({'w_n': 2.0, 'w_p': 4.0, 'l': 0.5})
    ckt.generate_netlist(corner='tt')
    mn_line = mn.to_spice().strip()
    mp_line = mp.to_spice().strip()

    print(f'  Applied  : w_n=2.0, w_p=4.0, l=0.5')
    print(f'  mn line  : {mn_line}')
    print(f'  mp line  : {mp_line}')
    assert 'l=' in mn_line and ('l=0.5' in mn_line or 'l=500m' in mn_line), \
        f"l=0.5 not found in mn line: {mn_line}"
    assert 'l=' in mp_line and ('l=0.5' in mp_line or 'l=500m' in mp_line), \
        f"l=0.5 not found in mp line: {mp_line}"
    print('  Both devices see the shared l=0.5')

    # ---------------------------------------------------------------------------
    # Step 7 — Visualise coverage (optional, requires matplotlib)
    # ---------------------------------------------------------------------------

    print()
    print('=' * 60)
    print('Step 7 — Coverage visualisation')
    print('=' * 60)

    try:
        # Let matplotlib pick its backend from MPLBACKEND / matplotlibrc.
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle('Sampling coverage: w_n vs w_p (12 random/LHS/Halton; 27 grid, l suppressed)',
                     fontsize=13)

        colors = {'random': 'steelblue', 'lhs': 'darkorange',
                  'halton': 'green',  'grid': 'crimson'}

        for ax, method in zip(axes, methods):
            pts = samples[method]
            xs = [p['w_n'] for p in pts]
            ys = [p['w_p'] for p in pts]
            ax.scatter(xs, ys, c=colors[method], s=40, alpha=0.8)
            ax.set_xlim(w_n.bounds)
            ax.set_ylim(w_p.bounds)
            ax.set_xlabel('w_n (µm)')
            ax.set_ylabel('w_p (µm)')
            ax.set_title(f'{method.upper()} ({len(pts)} pts)')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        print('  Showing coverage plot — close the window to exit.')
        plt.show()

    except ImportError:
        print('  matplotlib not available — skipping plot.')

    print()
    print('Done.')


if __name__ == '__main__':
    main()
