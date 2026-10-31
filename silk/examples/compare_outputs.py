"""Check a simulated supply waveform against the declared source value.

Build a differential-pair circuit and compare only its ideal source's fixed
5 V ``vdd`` waveform with that source value. This does not check the amplifier
outputs or device operating limits. Print the sample count, mean voltage and
maximum error. Simulation errors and mismatches fail the script.

Needs ngspice and installed sky130 models.
Run: ``python -m silk.examples.compare_outputs``.
"""

import tempfile
import numpy as np
from silk.metrics import average, maximum

# Build and simulate the circuit once.
def run_silk(tmp_dir):
    from silk.circuit.modules import Module
    from silk import load_pdk
    from silk import sources
    from silk.simulation import analysis

    pdk130 = load_pdk('sky130')

    ckt = Module('PDK_skywater130', tmp_dir, 'cmp_silk.cir', pdk_details=pdk130)

    mn_p = pdk130.nmos1p8('nmos_input_p', d='voutn', g='vinp', s='0', b='0', w=1.0, l=0.35, m=5)
    mn_n = pdk130.nmos1p8('nmos_input_n', d='voutp', g='vinn', s='0', b='0', w=1.0, l=0.35, m=5)
    mp_p = pdk130.pmos1p8('pmos_load_p',  d='voutn', g='voutp', s='vdd', b='vdd', w=1.0, l=0.35)
    mp_n = pdk130.pmos1p8('pmos_load_n',  d='voutp', g='voutp', s='vdd', b='vdd', w=1.0, l=0.35)
    ckt.add(mn_p, mn_n, mp_p, mp_n)
    ckt.add_source(sources.VoltageSource('supply', positive='vdd', negative='0', dc=5, ac=1))
    ckt.add_sim_type(analysis.Transient(step=1e-9, stop=100e-9))

    ckt.simulate(corner='tt')
    return ckt.load_signal('vdd')


def main():
    """Check the simulated supply waveform against its declared 5 V value."""
    print('Running one simulation and checking the supply waveform...')
    with tempfile.TemporaryDirectory() as tmp_dir:
        sig = run_silk(tmp_dir)
        print(f'vdd signal: {len(sig.y)} points, mean={average(sig):.4f} V')
        max_diff = maximum((sig - 5.0).mag())
        print(f'Max absolute difference from 5 V: {max_diff:.2e} V')
        np.testing.assert_allclose(sig.y, 5.0, rtol=1e-6, atol=0)
        print('PASS — supply waveform matches the declared 5 V source')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
