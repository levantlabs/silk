"""Search for a source follower with minimum reference current.

A Latin-hypercube sweep varies the reference-current knob i_ref and follower
width. The AC testbench supplies gain and bandwidth. Margins on its specs
require bandwidth above 100 MHz and gain above 0.8 V/V. opt.best selects
the feasible sampled point with the lowest reference current.

Reference current is a power proxy, not a measured branch current or total
power. The mirrored current also depends on operating point and sizing.

Run ``MPLBACKEND=Agg .venv/bin/python -m silk.examples.source_follower_optimize``.
It prints the sweep summary and best feasible sample, if one exists.
Needs ngspice and sky130 models. Set the corner on opt.sweep().
"""


from silk import Variable
from silk.optimize import Optimizer, VariableSpace

from silk.examples.source_follower_example import build_source_follower, GAIN_METRIC, BW_METRIC


def build_optimizer():
    """Build a follower optimizer over reference current and follower width."""
    i_ref = Variable("i_ref", value=20e-6, bounds=(2e-6, 50e-6))
    w_follower = Variable("w_follower", value=10.0, bounds=(4.0, 40.0))

    mod = build_source_follower(i_ref=i_ref, w_follower=w_follower)
    ac = mod.get_test("ac")
    opt = Optimizer(ac, VariableSpace([i_ref, w_follower]))
    opt.add_objective(i_ref, minimize=True)

    # Margins raise the testbench limits to 100 MHz and 0.8 V/V.
    opt.add_margin("BW > 10 MHz", 90e6)
    opt.add_margin("gain > 0.7 V/V", 0.1)
    return opt


if __name__ == "__main__":
    opt = build_optimizer()
    print("Sweeping (LHS, n=20): minimize reference current s.t. "
          "BW > 100 MHz, gain > 0.8 V/V (spec + margin)\n")
    opt.sweep(n=20, method="lhs", corner="tt")

    print(opt.summary())
    best = opt.best
    if best is None:
        print("\nNo feasible point found in the sample.")
    else:
        p, m = best["params"], best["metrics"]
        print("\nBest (minimum-reference-current feasible) design:")
        print(f"  i_ref      = {p['i_ref']*1e6:.2f} µA")
        print(f"  w_follower = {p['w_follower']:.2f} µm")
        print(f"  -> reference current {p['i_ref']*1e6:.2f} µA, "
              f"BW {m[BW_METRIC]/1e6:.1f} MHz, gain {m[GAIN_METRIC]:.3f} V/V")
