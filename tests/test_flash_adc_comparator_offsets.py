"""flash_adc_example's comparator_offsets path, exercised without a simulator (TASK-524).

``make_flash_adc(comparator_offsets=...)`` marks each comparator ``variants=True``
and calibrates its ``offset_voltage`` to the cancelling value ``-offset``. The
path is pure construction plus netlist generation, so these tests need neither
ngspice nor a PDK model library. Output goes to ``tmp_path``.
"""
import re
from pathlib import Path

import pytest

from silk.examples import flash_adc_example as example

# Three distinct offsets over the first four comparators (2 mV is shared by
# comp2 and comp3); the remaining comparators sit at 0 V like comp1.
OFFSETS = (0.0, 2e-3, 2e-3, -3e-3) + (0.0,) * (example.N_COMP - 4)


@pytest.fixture
def adc(tmp_path, monkeypatch):
    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    dut, _ = example.make_flash_adc(with_specs=False, comparator_offsets=OFFSETS)
    return dut


@pytest.fixture
def deck(adc):
    return Path(adc.generate_netlist()).read_text()


def _comparator_placements(adc):
    """comp<k> -> resolved definition name, read through the public iterator."""
    placements = {}
    for record in adc.iter_instances():
        if record.path.startswith('comp') and record.module_name.startswith('comparator_'):
            placements[record.path] = record.module_name
    return placements


def _comparator_subckts(deck):
    return re.findall(r'^\.subckt (comparator_\w+)', deck, flags=re.MULTILINE)


def _top_level_comparator_targets(deck):
    """xcomp<k> instance lines in the top deck -> the definition each one names."""
    targets = {}
    for line in deck.splitlines():
        match = re.match(r'^xcomp(\d+) ', line)
        if match:
            targets[f'comp{match.group(1)}'] = line.split()[-1]
    return targets


def test_one_definition_per_distinct_offset(deck):
    subckts = _comparator_subckts(deck)
    assert len(subckts) == len(set(subckts))  # no definition emitted twice
    assert len(subckts) == len(set(OFFSETS))


def test_equal_offsets_share_a_definition_and_distinct_offsets_do_not(adc, deck):
    targets = _top_level_comparator_targets(deck)
    assert len(targets) == example.N_COMP
    assert set(targets.values()) <= set(_comparator_subckts(deck))
    # Same offset -> same target; different offset -> different target.
    assert targets['comp2'] == targets['comp3']
    assert targets['comp1'] != targets['comp4']
    assert targets['comp1'] == targets['comp5']  # 0 V comparators share one definition
    # The placement-level view agrees with the deck.
    placements = _comparator_placements(adc)
    assert placements == targets


def test_each_comparator_carries_its_requested_offset(adc, deck):
    records = {record.path: record for record in adc.iter_instances()
               if record.path.startswith('comp')}
    for k, offset in enumerate(OFFSETS, 1):
        # The placement's definition is the live module; calibrate_behavioral
        # stores the cancelling value, the negated offset.
        module = records[f'comp{k}'].module
        assert module._behavioral_params['offset_voltage'] == pytest.approx(-offset)


def test_ladder_model_retains_constructed_resolution(tmp_path, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from silk import load_pdk

    monkeypatch.setattr(example, 'N_COMP', 3)
    monkeypatch.setattr(example, 'N_BITS', 2)
    ladder = example.make_ref_ladder(load_pdk('sky130'), str(tmp_path))
    monkeypatch.setattr(example, 'N_COMP', 15)
    monkeypatch.setattr(example, 'N_BITS', 4)
    ports = SimpleNamespace(vref=SimpleNamespace(v=np.array([1.4])),
                            vrefl=SimpleNamespace(v=np.array([0.8])),
                            **{f'vref{k}': SimpleNamespace(v=None) for k in range(1, 4)})
    ladder._behavioral_model(ports, np.array([0.]), {})
    assert ports.vref3.v[0] == pytest.approx(1.25)


def test_censored_offsets_cannot_calibrate_chip_variation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from silk.exceptions import SimulationError

    reports = {i: SimpleNamespace(metrics={'offset_v': value, 'offset_cap_side': side})
               for i, (value, side) in enumerate([(-.1, -1), (.1, 1)])}
    bench = SimpleNamespace(run_monte_carlo=lambda *args, **kw:
                            SimpleNamespace(reports=reports, failed={}))
    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    monkeypatch.setattr(example, 'make_comparator_offset_tb', lambda *args, **kw: bench)
    with pytest.raises(SimulationError, match='uncensored'):
        example.characterize_offset(samples=2)


def test_characterization_retains_built_sine_settings(tmp_path, monkeypatch):
    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    dut, _ = example.make_flash_adc(with_specs=False, sine=True, n_bits=2)
    monkeypatch.setattr(example, 'T_STOP', 1e-9)
    monkeypatch.setattr(example, 'N_BITS', 8)
    monkeypatch.setattr(example, 'N_COMP', 255)
    monkeypatch.setattr(example, 'USE_SINE', False)
    test = example._make_characterization(dut)
    assert dut.adc_t_stop == example.SINE_T_STOP
    assert test.netlist.sim_analysis_dict['transient'].stop == example.SINE_T_STOP
    assert any(outcome.spec.name == 'adc_sndr'
               for outcome in dut.reconcile_specs().outcomes)
