"""Measure ring frequency, duty cycle and extracted delay across stage counts.

Each ring has an embedded oscillation test that starts it with an enable edge.
The sweep runs this test for 3, 5, 7, 9 and 11 stages. It computes average per-stage delay as
``t_d = 1 / (2 * N * f_osc)`` and reports its spread across the sweep.

Run ``MPLBACKEND=Agg .venv/bin/python -m silk.examples.ring_oscillator_stage_sweep``
from the repository root with ngspice and sky130 models installed. It prints
frequency, duty, delay and spec results for each ring, then the delay spread.
It saves ``ring_oscillator_stages.png`` in the configured examples output
directory, showing frequency against 1/N and extracted delay against N.
"""
import os

import numpy as np

from silk.examples.ring_oscillator_example import (build_ring_oscillator, run_characterization,
                                                   F_METRIC, DUTY_METRIC, STAGES)
from silk.examples import example_output_dir


# Odd only — an even ring is a latch, and build_ring_oscillator rejects it.
STAGE_COUNTS = (3, 5, 7, 9, 11)


def sweep(stage_counts=STAGE_COUNTS):
    """Build the ring at each stage count and run its embedded tests.

    Returns a list of ``(stages, f_mhz, duty_pct, t_d_ps, all_specs_passed)``.
    """
    rows = []
    for n in stage_counts:
        mod = build_ring_oscillator(stages=n)      # testbenches already attached
        results, recon = run_characterization(mod)
        m = results[f"{mod.module_name}.osc"].metrics
        f_mhz = m[F_METRIC]
        t_d_ps = 1.0 / (2 * n * f_mhz * 1e6) * 1e12     # f = 1/(2*N*t_d)
        rows.append((n, f_mhz, m[DUTY_METRIC], t_d_ps, recon.fail_count == 0))
    return rows


def save_plot(rows, path):
    """Plot frequency against 1/N with a linear fit, and extracted delay against N."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = np.array([r[0] for r in rows], dtype=float)
    f = np.array([r[1] for r in rows], dtype=float)
    td = np.array([r[3] for r in rows], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Fit frequency against inverse stage count.
    inv_n = 1.0 / n
    slope, intercept = np.polyfit(inv_n, f, 1)
    xs = np.linspace(0, inv_n.max() * 1.1, 50)
    ax1.plot(xs, slope * xs + intercept, "--", color="#5a5a6e", lw=1,
             label=f"linear fit: f = {slope:.0f}/N {intercept:+.0f} MHz")
    ax1.plot(inv_n, f, "o", color="#0072B2", ms=7, label="measured")
    for ni, xi, fi in zip(n, inv_n, f):
        ax1.annotate(f"N={ni:.0f}", (xi, fi), textcoords="offset points",
                     xytext=(6, -3), fontsize=8, color="#5a5a6e")
    ax1.set_xlabel("1 / N")
    ax1.set_ylabel("f_osc (MHz)")
    ax1.set_title("Frequency against inverse stage count")
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax1.legend(frameon=False, fontsize=8)

    # Compare extracted delay across stage counts.
    ax2.plot(n, td, "o-", color="#D55E00", lw=1.5, ms=7)
    ax2.axhline(td.mean(), color="#5a5a6e", ls="--", lw=1,
                label=f"mean {td.mean():.1f} ps")
    ax2.set_xlabel("stages N")
    ax2.set_ylabel("extracted per-stage delay t_d (ps)")
    ax2.set_title("Extracted average per-stage delay")
    ax2.set_ylim(0, td.max() * 1.35)
    ax2.legend(frameon=False, fontsize=8)

    for ax in (ax1, ax2):
        ax.grid(True, color="#d5d5dd", lw=0.6, alpha=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    OUT = example_output_dir()

    print("=" * 66)
    print("Ring oscillator — stage-count sweep")
    print("  the DUT arrives with its testbenches already embedded;")
    print("  this file only varies `stages` and reads the metrics off")
    print("=" * 66)

    rows = sweep()
    print(f"\n  {'N':>3} {'f_osc (MHz)':>12} {'duty (%)':>9} {'t_d (ps)':>9}   specs")
    for n, f, duty, td, ok in rows:
        print(f"  {n:>3} {f:12.1f} {duty:9.1f} {td:9.1f}   {'PASS' if ok else 'FAIL'}")

    td = np.array([r[3] for r in rows])
    f = np.array([r[1] for r in rows])
    n = np.array([r[0] for r in rows], dtype=float)
    print(f"\n  f_osc falls {f[0]:.0f} -> {f[-1]:.0f} MHz as N goes {n[0]:.0f} -> {n[-1]:.0f}"
          f"  ({f[0]/f[-1]:.1f}x, vs {n[-1]/n[0]:.1f}x more stages)")
    print(f"  extracted t_d: {td.min():.1f} .. {td.max():.1f} ps "
          f"(mean {td.mean():.1f}, spread {np.ptp(td)/td.mean()*100:.0f}%)")
    print(f"\n  (the default build is {STAGES} stages — see ring_oscillator_example.py)")
    print(f"\n  saved {save_plot(rows, os.path.join(OUT, 'ring_oscillator_stages.png'))}")
