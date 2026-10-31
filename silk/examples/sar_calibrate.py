"""Characterize the SAR capacitor DAC and compare SPICE codes with the twin.

The DAC sweep prints fitted gain and peak error from the ideal transfer,
including gain and offset. The fit is reported without changing model constants.
The loop comparison uses nominal capacitors and zero comparator offset and noise.
Use VIN_CM=VREF/2 to keep the conversion residue near VDD/2; a wrong common
mode is not comparator offset. Mismatch and noise need separate characterization.

Run .venv/bin/python -m silk.examples.sar_calibrate from the repository root.
Requires ngspice and sky130 models. Prints DAC measurements and a code table.
"""

import numpy as np

from silk import load_pdk
from silk.examples import example_output_dir
from silk.examples import sar_adc_example as sar
from silk.examples import sar_adc_spice as spice
def measure(n=8, inputs=None):
    """Measure DAC gain, ideal-transfer error and loop code differences."""
    lsb = 2 * spice.VREFP / 2 ** n

    print("  measuring cap-DAC transfer (charge redistribution on real caps)...")
    codes, vsp, vid = spice.dac_transfer_spice(n)
    gain = float(np.polyfit(vid, vsp, 1)[0])            # SPICE vs ideal slope
    peak_nl = float(np.abs((vsp - vid) / lsb).max())

    if inputs is None:
        inputs = np.linspace(-0.7 * spice.VREFP, 0.7 * spice.VREFP, 5)
    inputs = np.asarray(inputs)
    print(f"  comparing the SAR loop (real StrongARM, {len(inputs)} inputs)...")
    pdk = load_pdk('sky130')
    od = example_output_dir()
    w, wtot = sar.unit_cap_array(n, sigma=0.0)
    codes_sa = np.array([spice.convert_spice(n, float(v), pdk, od) for v in inputs])
    codes_tw = np.asarray(sar.sar_codes(inputs, w, wtot, vref=spice.VREFP,
                                        comp_sigma=0.0, comp_offset=0.0))
    return {"gain": gain, "peak_nl_lsb": peak_nl, "inputs": inputs,
            "codes_sa": codes_sa, "codes_tw": codes_tw, "lsb": lsb}


def main(n=8):
    print("=" * 72)
    print(f"SAR DAC characterization and uncalibrated twin comparison ({n}-bit)")
    print("=" * 72)
    m = measure(n)

    print(f"\n  cap-DAC effective gain = {m['gain']:.4f} (ideal 1.000), "
          f"peak ideal-transfer error = {m['peak_nl_lsb']:.2f} LSB")

    print("\n  The fitted DAC gain is reported without updating the twin.")
    print("  Comparing nominal SPICE loop codes with the uncalibrated twin...")
    resid = m["codes_sa"] - m["codes_tw"]
    print(f"    {'vin(mV)':>9}{'SPICE':>8}{'twin':>7}{'Δ':>5}")
    for i, v in enumerate(m["inputs"]):
        print(f"    {v*1e3:>9.0f}{m['codes_sa'][i]:>8d}{m['codes_tw'][i]:>7d}{resid[i]:>5d}")
    print(f"    Maximum code difference: {int(np.abs(resid).max())} LSB.")
    return m


if __name__ == "__main__":
    main()
