"""Behavioral contract for the repository's pytest execution tiers."""

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

PACKAGING = 'tests/test_packaging.py::test_wheel_is_complete_and_importable'
CODEGEN_ARRAY = (
    'tests/test_codegen_arrays.py::TestArrayedSubmoduleEndToEnd::'
    'test_arrayed_submodule_execs_and_registers'
)
XYCE = {
    'tests/test_noise_canonical.py::test_ngspice_xyce_noise_agree',
    ('tests/test_xyce_netlister.py::TestXyceEndToEnd::'
     'test_xyce_module_runs_through_silk'),
}
GF180 = {
    ('tests/test_gf180mcu_passives.py::'
     'test_a_diffusion_resistor_matches_its_sheet_resistance'),
    ('tests/test_gf180mcu_passives.py::'
     'test_the_moscap_pin_called_gate_really_is_the_gate'),
    ('tests/test_design_loop_walkthrough_gf180mcu.py::'
     'test_gf180mcu_walkthrough_end_to_end'),
}


def _collect(marker_expression, nodeids):
    """Return node IDs selected by pytest's real marker evaluator."""
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-o', 'addopts=', '--collect-only',
         '-q', '-m', marker_expression, *nodeids],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if line.startswith('tests/') and '::' in line
    }


def test_core_slow_profile_excludes_optional_capabilities():
    candidates = [PACKAGING, sorted(XYCE)[0],
                  sorted(GF180)[0]]
    selected = _collect(
        'slow and not requires_xyce '
        'and not requires_gf180',
        candidates,
    )
    assert selected == {PACKAGING}


def test_arrayed_codegen_exec_runs_in_fast_profile():
    assert _collect('not slow', [CODEGEN_ARRAY]) == {CODEGEN_ARRAY}


@pytest.mark.parametrize(('marker', 'expected'), [
    ('requires_xyce', XYCE),
    ('requires_gf180', GF180),
])
def test_optional_slow_profile_selects_its_contract(marker, expected):
    assert _collect(f'slow and {marker}', sorted(expected)) == expected
