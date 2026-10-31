import os
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from silk.faults.pdk_metadata import (
    load_device_wl_schema,
    perturb_l,
    perturb_w,
)
from silk.pdk.basepdk import BasePDK, FaultDimSchema, FaultParams
from silk.pdk.basepdk import BasePDK

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
_SKY130_YAML = os.path.join(_REPO_ROOT, 'silk', 'pdk', 'data',
                            'skywater_130_device_details.yaml')
_GF180MCU_YAML = os.path.join(_REPO_ROOT, 'silk', 'pdk', 'data',
                              'gf180mcu_device_details.yaml')


@pytest.fixture
def sky130():
    """Real BasePDK, parsed from the bundled catalog (fake lib
    path — fault_params resolution never opens the .lib file)."""
    return BasePDK(_SKY130_YAML, library_location='/fake/sky130.lib')


@pytest.fixture
def gf180mcu():
    """A second, distinct PDK whose catalog declares no fault_params: at
    all — proves resolution is per-PDK, never a fallback to sky130."""
    return BasePDK(_GF180MCU_YAML, library_location='/fake/gf180mcu.lib')


# --- resolution through a loaded PDK (no hardcoded catalog path) ---

def test_schema_loaded_for_sky130_nmos1p8_through_pdk(sky130):
    schema = load_device_wl_schema(sky130, 'nmos1p8')
    assert schema is not None
    assert schema.w.kind == 'continuous'
    assert schema.w.min == 0.42
    assert schema.w.max == 100.0
    assert schema.l.kind == 'continuous'


def test_schema_missing_device_on_pdk_returns_none(sky130):
    schema = load_device_wl_schema(sky130, 'not_a_real_device_name')
    assert schema is None


def test_schema_device_with_no_fault_params_block_returns_none(sky130):
    # res_hpo is a real sky130 device whose catalog entry has no
    # fault_params: block (only the MOS devices declare one).
    schema = load_device_wl_schema(sky130, 'res_hpo')
    assert schema is None


def test_schema_none_pdk_returns_none():
    # No PDK given at all — never guesses, never falls back to sky130.
    schema = load_device_wl_schema(None, 'nmos1p8')
    assert schema is None


def test_schema_pdk_with_no_fault_params_anywhere_returns_none(gf180mcu):
    # A real, distinct PDK (GF180MCU) whose catalog declares fault_params
    # for no device at all.
    assert 'nmos3p3' in gf180mcu._devices
    schema = load_device_wl_schema(gf180mcu, 'nmos3p3')
    assert schema is None


def test_same_named_device_on_a_different_pdk_never_leaks_bounds():
    """AUDIT-02 finding: the old name-keyed global cache let a
    same-named device on a different PDK silently receive sky130's
    fault bounds. Resolution must be scoped to the PDK object passed in.
    """
    sky130_bounds = FaultParams(
        w=FaultDimSchema(kind='continuous', min=0.42, max=100.0),
        l=FaultDimSchema(kind='continuous', min=0.15, max=100.0),
    )
    pdk_a = SimpleNamespace(_devices={
        'nmos1p8': SimpleNamespace(fault_params=sky130_bounds),
    })
    # A different PDK happens to reuse the name 'nmos1p8' but declares no
    # fault_params for it.
    pdk_b = SimpleNamespace(_devices={
        'nmos1p8': SimpleNamespace(fault_params=None),
    })
    assert load_device_wl_schema(pdk_a, 'nmos1p8') is sky130_bounds
    assert load_device_wl_schema(pdk_b, 'nmos1p8') is None


# --- Continuous perturbation ---

def test_perturb_w_continuous_centered_on_nominal(sky130):
    schema = load_device_wl_schema(sky130, 'nmos1p8')
    rng = np.random.default_rng(42)
    samples = [perturb_w(schema, nominal=4.0, sigma=0.02, rng=rng)
               for _ in range(1000)]
    w_values = [s['w_new'] for s in samples]
    assert 3.8 < np.mean(w_values) < 4.2
    assert 0.05 < np.std(w_values) < 0.12


def test_perturb_w_continuous_clamped_to_min(sky130):
    schema = load_device_wl_schema(sky130, 'nmos1p8')
    rng = np.random.default_rng(42)
    # sigma large enough that the draw is essentially guaranteed outside
    # [w.min, w.max] = [0.42, 100.0]; the first standard_normal for seed=42
    # is +0.3047, giving raw = 0.42 * (1 + 10000 * 0.3047) >> 100.
    result = perturb_w(schema, nominal=0.42, sigma=10000.0, rng=rng)
    assert result['w_new'] >= schema.w.min
    assert result['w_new'] <= schema.w.max
    assert result['clamped'] is True


def test_perturb_w_returns_full_realized_params_dict(sky130):
    schema = load_device_wl_schema(sky130, 'nmos1p8')
    rng = np.random.default_rng(42)
    result = perturb_w(schema, nominal=4.0, sigma=0.02, rng=rng)
    assert 'w_nominal' in result
    assert 'w_multiplier' in result
    assert 'w_new' in result
    assert 'clamped' in result
    assert result['w_nominal'] == 4.0


# --- Quantized perturbation (future-FinFET hypothetical) ---

def test_perturb_w_quantized_rounds_to_step():
    schema = FaultParams(
        w=FaultDimSchema(kind='quantized', step=0.1, min_count=1, max_count=100),
        l=FaultDimSchema(kind='continuous', min=0.15, max=100.0),
    )
    rng = np.random.default_rng(42)
    result = perturb_w(schema, nominal=1.0, sigma=0.1, rng=rng)
    assert abs(result['w_new'] % 0.1) < 1e-9 or abs(result['w_new'] % 0.1 - 0.1) < 1e-9


# --- Discrete perturbation ---

def test_perturb_l_discrete_snaps_to_allowed():
    schema = FaultParams(
        w=FaultDimSchema(kind='continuous', min=0.1, max=10.0),
        l=FaultDimSchema(kind='discrete', allowed=[0.018, 0.036, 0.054]),
    )
    rng = np.random.default_rng(42)
    result = perturb_l(schema, nominal=0.036, sigma=0.3, rng=rng)
    assert result['l_new'] in (0.018, 0.036, 0.054)


# --- Missing schema fallback ---

def test_perturb_w_missing_schema_warns_and_continues():
    rng = np.random.default_rng(42)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = perturb_w(None, nominal=4.0, sigma=0.02, rng=rng)
    assert result['w_new'] > 0
    assert any('no w/l schema' in str(x.message).lower() for x in w)
