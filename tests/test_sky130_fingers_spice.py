"""Real-model controls for catalog-derived mismatch multiplicity."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from silk import Module, load_pdk, sources
from silk.circuit.monte_carlo import _SampleSpec
from silk.circuit.module_test import _read_sample_draws
from silk.elements.factory import make_device_class
from silk.simulation import Op
from tests._env import needs_ngspice, needs_pdk

pytestmark = [pytest.mark.slow, needs_ngspice, needs_pdk]


def test_derived_multiplicity_halves_paired_threshold_mismatch(tmp_path):
    """Hold seed, nf/W/L/m and order fixed; only mismatch normalization changes."""
    sigmas = []
    draws = []
    for derived in (False, True):
        pdk = load_pdk('sky130')
        factory = pdk.nmos1p8
        if not derived:
            # Scratch catalog control reproduces the old mult-omitted emission.
            path = Path(__file__).parents[1] / 'silk/pdk/data/skywater_130_device_details.yaml'
            spec = yaml.safe_load(path.read_text())['nmos1p8']
            spec.pop('derivedParams')
            factory = make_device_class('nmos1p8', spec, is_pdk=True)
        mod = Module('sky130', str(tmp_path / str(derived)), 'mismatch', pdk)
        for i in range(32):
            mod.add(factory(f'm{i}', d=f'd{i}', g='g', s='0', b='0',
                            w=1, l=.15, nf=1, m=4))
            mod.add(sources.vdc(f'vd{i}', positive=f'd{i}', negative='0', value=1))
        mod.add(sources.vdc('vg', positive='g', negative='0', value=.8))
        mod.add_sim_type(Op())
        mod._mc_sample = _SampleSpec(index=0, seed=574, variation='mismatch',
                                    section='tt_mm', wrapper_path=None, record_draws=True)
        mod.simulate(corner='tt')
        values = _read_sample_draws(mod)
        vth = np.array([values[(f'm{i}', 'vth0')] for i in range(32)])
        assert np.all(np.isfinite(vth))
        draws.append(vth)
        sigmas.append(np.std(vth, ddof=1))
        text = Path(mod._run_files.netlist_path).read_text()
        assert (' mult=4' in text) == derived
    assert sigmas[0] > .001
    assert sigmas[1] / sigmas[0] == pytest.approx(.5, abs=.001)
    np.testing.assert_allclose(draws[1] - np.mean(draws[1]),
                               (draws[0] - np.mean(draws[0])) / 2, atol=2e-6)
    print(f'paired vth0 sigma: old={sigmas[0]:.9g} V derived={sigmas[1]:.9g} V')
