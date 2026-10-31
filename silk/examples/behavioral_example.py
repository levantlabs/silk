"""
Behavioral Simulation Example
==============================

Demonstrates silk's Python-native behavioral simulation without any PDK,
ngspice, or external tools. The circuit models:

  1. A comparator (stateless function) — outputs 1.8 V when vin > vref
  2. An RC low-pass filter (stateful class) — integrates the comparator output

Circuit topology:
  vin (pulse) ─┬─► comparator ─► vcomp ─► rc_filter ─► vfilt
  vref (DC)   ─┘

No PDK_ROOT, ngspice, or PDK YAML required.
"""

from silk import load_pdk
from silk.examples import example_output_dir
from silk.metrics import minimum, maximum
import numpy as np


# ---------------------------------------------------------------------------
# Minimal Module factory — no real PDK files needed
# ---------------------------------------------------------------------------

def _make_bare_module(name='top'):
    """Create a Module backed by the built-in YAML device table."""
    from silk.circuit.modules import Module

    pdk = load_pdk('sky130')
    work = example_output_dir('behavioral')
    m = Module('PDK_skywater130', work, f'{name}.cir', pdk_details=pdk)
    m.module_name = name
    return m


# ---------------------------------------------------------------------------
# 1. Comparator subcircuit — stateless behavioral model
# ---------------------------------------------------------------------------

def build_comparator():
    """Return a Module modelling a rail-to-rail comparator."""
    from silk import PortDir

    comp = _make_bare_module('comparator')
    comp.define_module_ports({
        'vin':  PortDir.IN,
        'vref': PortDir.IN,
        'vout': PortDir.OUT,
    })

    def comparator_model(ports, t, params):
        """Output 1.8 V when vin > vref, else 0 V."""
        ports.vout.v = np.where(ports.vin.v > ports.vref.v, 1.8, 0.0)

    comp.set_behavioral_model(comparator_model, execution='vector')
    return comp


# ---------------------------------------------------------------------------
# 2. RC low-pass filter — stateful class-based behavioral model
# ---------------------------------------------------------------------------

def build_rc_filter(R=1e3, C=100e-12):
    """Return a Module modelling a first-order RC low-pass filter."""
    from silk import PortDir

    rc = _make_bare_module('rc_filter')
    rc.define_module_ports({
        'vin':  PortDir.IN,
        'vout': PortDir.OUT,
    })

    class RCFilterModel:
        """Discrete-time RC filter using the backward Euler method."""

        def __init__(self, R, C):
            self.R = R
            self.C = C

        def __call__(self, ports, t, params):
            dt = t[1] - t[0] if len(t) > 1 else (t[0] if len(t) == 1 else 1e-9)
            alpha = dt / (self.R * self.C + dt)   # first-order IIR coefficient
            vin = ports.vin.v
            vout = np.zeros_like(vin)
            v = 0.0
            for i, vi in enumerate(vin):
                v = v + alpha * (vi - v)
                vout[i] = v
            ports.vout.v = vout

    rc.set_behavioral_model(RCFilterModel(R, C), execution='vector')
    return rc


# ---------------------------------------------------------------------------
# 3. Top-level circuit — wire comparator → RC filter with pulse + DC sources
# ---------------------------------------------------------------------------

def build_top():
    from silk import sources

    # Sources
    vin_src  = sources.VoltageSource(
        'vin_src', positive='vin', negative='gnd',
        pulse=sources.Pulse(v1=0, v2=2.0, td=5e-9, tr=1e-9, tf=1e-9,
                            pw=20e-9, per=40e-9))
    vref_src = sources.VoltageSource(
        'vref_src', positive='vref', negative='gnd', dc=0.9)

    comp = build_comparator()
    rc   = build_rc_filter(R=1e3, C=100e-12)

    top = _make_bare_module('top')
    top.add_source(vin_src)
    top.add_source(vref_src)

    # Instantiate comparator: maps top-level nets → subcircuit ports
    top.add_module(comp, 'comp1',
                   nets={'vin': 'vin', 'vref': 'vref', 'vout': 'vcomp'})

    # Instantiate RC filter: comparator output feeds filter input
    top.add_module(rc, 'rc1',
                   nets={'vin': 'vcomp', 'vout': 'vfilt'})

    return top


# ---------------------------------------------------------------------------
# 4. Run behavioral simulation and display results
# ---------------------------------------------------------------------------

def main():
    top = build_top()

    # Run the Python-native simulation — no ngspice needed
    top.simulate_behavioral(t_step=1e-9, t_stop=120e-9)

    # Inspect available signals
    names = top.get_signal_names()
    print(f"Available signals: {sorted(names)}")

    # Load key waveforms
    s_vin   = top.load_signal('vin')
    s_vcomp = top.load_signal('vcomp')
    s_vfilt = top.load_signal('vfilt')

    # Print basic stats for each signal
    for label, sig in [('vin', s_vin), ('vcomp', s_vcomp), ('vfilt', s_vfilt)]:
        print(f"  {label:6s}  min={minimum(sig):.3f} V  "
              f"max={maximum(sig):.3f} V  "
              f"mean={sig.y.mean():.3f} V  "
              f"({len(sig.y)} samples)")

    # Sanity checks
    assert maximum(s_vcomp) == 1.8, "Comparator high level should be 1.8 V"
    assert minimum(s_vcomp) == 0.0, "Comparator low level should be 0.0 V"
    assert maximum(s_vfilt) < 1.8,  "RC filter should attenuate the square wave"
    assert maximum(s_vfilt) > 0.0,  "RC filter output should be non-zero"

    print("Behavioral simulation completed successfully.")


if __name__ == '__main__':
    main()
