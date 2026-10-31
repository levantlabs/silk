"""Guards for the flagship walkthrough ``silk/examples/design_loop_walkthrough.py``
(release task TASK-11.3: one system-to-layout design loop, end to end).

Fast tests pin the structural claims (intent → sizing is pure, the factory
builds typed ports with roles and four uniquely named tests with distinct
declared parameters, the netlist comes from the design object). The two slow
tests execute the documented command: once in-process with assertions on
every step's outcome, and once from a freshly built wheel in a foreign
working directory, which is the only way to prove the shipped artifact runs.
"""
import os
import subprocess
import sys
import zipfile

import pytest

from tests._env import needs_ngspice, needs_pdk
from tests._example_conditions import assert_walkthrough_temperatures

# Importing the example must not touch a PDK or a simulator.
from silk.examples import design_loop_walkthrough as dl

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# fast
# --------------------------------------------------------------------------- #
def test_intent_to_sizing_is_pure_and_responds_to_the_spec():
    knobs = dl.size_from_intent()
    assert knobs['i_tail_a'] > 0
    assert knobs['i_ref_a'] == pytest.approx(knobs['i_tail_a'] / dl.MIRROR_RATIO)
    assert knobs['w_tail_um'] > knobs['w_ref_um'] > 0
    # Every width sits on the sky130 manufacturing grid.
    for key in ('w_in_um', 'w_load_um', 'w_tail_um', 'w_ref_um'):
        steps = knobs[key] / dl.WIDTH_GRID_UM
        assert abs(steps - round(steps)) < 1e-6, (key, knobs[key])
    # Halving the allowed settling time doubles the current budget.
    tighter = dict(dl.INTENT, settle_heavy_s_max=dl.INTENT['settle_heavy_s_max'] / 2)
    assert dl.size_from_intent(tighter)['i_tail_a'] == pytest.approx(
        2 * knobs['i_tail_a'])
    assert dl.size_from_intent() == knobs                     # deterministic


def test_example_has_no_import_time_pdk_or_simulator_dependency():
    # In a fresh interpreter: another test module may already have built a
    # design (and so populated the cache) before this one runs.
    probe = subprocess.run(
        [sys.executable, '-c',
         'from silk.examples import design_loop_walkthrough as dl; '
         'assert not dl._PDK_CACHE, "import must not load the PDK"'],
        capture_output=True, text=True)
    assert probe.returncode == 0, probe.stderr


@needs_pdk
class TestDesignObject:

    def test_ports_carry_directions_and_roles(self, tmp_path):
        from silk.circuit.ports import PortDir, PortRole
        mod, _tests = dl.build_sampling_buffer(dl.size_from_intent(),
                                               work_dir=str(tmp_path))
        pins = mod.pin_list
        assert pins['VDD'].role is PortRole.POWER
        assert pins['VSS'].role is PortRole.GROUND
        assert pins['VOUT'].direction is PortDir.OUT
        assert pins['VOUT'].role is PortRole.ANALOG
        for name in ('VINP', 'VINM', 'IBIAS'):
            assert pins[name].direction is PortDir.IN
            assert pins[name].role is PortRole.ANALOG

    def test_four_uniquely_named_tests_with_distinct_profiles(self, tmp_path):
        mod, tests = dl.build_sampling_buffer(dl.size_from_intent(),
                                              work_dir=str(tmp_path))
        names = mod.list_tests()
        assert names == [dl.T_STEP_LIGHT, dl.T_STEP_HEAVY, dl.T_AC_PRE, dl.T_AC_POST]
        assert len(set(names)) == 4
        # Distinct declared parameters → distinct characterization profiles
        # even before any PVT point is chosen.
        assert tests[dl.T_STEP_LIGHT].parameters != tests[dl.T_STEP_HEAVY].parameters
        assert tests[dl.T_STEP_LIGHT].parameters['cload_f'] == dl.INTENT['light_load_f']
        assert tests[dl.T_STEP_HEAVY].parameters['cload_f'] == dl.INTENT['heavy_load_f']
        # Every test declared metrics and a first-class spec on the module.
        keys = [s.key for s in mod._specs]
        assert keys == ['settle_light', 'settle_heavy', 'gain', 'bw_pre_layout',
                        'bw_post_layout']
        assert len(set(keys)) == len(keys)
        # Nothing has been simulated or restored.
        assert mod.net_dict['tail'].peak_current is None
        assert all(not s.measurements for s in mod._specs)

    def test_duplicate_test_name_is_refused(self, tmp_path):
        from silk.exceptions import DuplicateNameError
        mod, _tests = dl.build_sampling_buffer(dl.size_from_intent(),
                                               work_dir=str(tmp_path))
        with pytest.raises(DuplicateNameError):
            mod.create_test(dl.T_STEP_LIGHT)

    def test_netlist_is_generated_from_the_design_object(self, tmp_path):
        knobs = dl.size_from_intent()
        mod, _tests = dl.build_sampling_buffer(knobs, work_dir=str(tmp_path))
        text = mod.generate_netlist()
        assert text.startswith('.subckt sampling_buffer ')
        for inst in ('XMM1', 'XMM2', 'XMM3', 'XMM4', 'XMM5', 'XMM6'):
            assert any(line.startswith(inst + ' ') for line in text.splitlines()), inst
        assert f"w={knobs['w_tail_um']}" in text

    def test_layout_estimate_uses_micron_geometry(self, tmp_path):
        """The factory-built devices must estimate a µm-scale footprint (the
        area-model unit bug this walkthrough exposed made it 1e12 too big)."""
        from silk.layout import estimate_layout
        mod, _tests = dl.build_sampling_buffer(dl.size_from_intent(),
                                               work_dir=str(tmp_path))
        est = estimate_layout(mod, dl.MET1)
        assert 1.0 < est.nets['VOUT'].length < 100.0
        assert est.nets['VOUT'].c < 10.0                      # fF


# --------------------------------------------------------------------------- #
# slow — the documented command, in process
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@needs_pdk
@needs_ngspice
def test_walkthrough_end_to_end(tmp_path):
    summary = dl.main(str(tmp_path))
    assert_walkthrough_temperatures(summary)

    # Two distinct profiles at one corner, both with selected current sets.
    prov = summary['provenance']
    assert prov[dl.T_STEP_LIGHT]['profile_id'] != prov[dl.T_STEP_HEAVY]['profile_id']
    for name in (dl.T_STEP_LIGHT, dl.T_STEP_HEAVY):
        assert prov[name]['origin_version_id'] == summary['version_id']
        assert prov[name]['rows'] > 0
        assert prov[name]['tail_peak_a'] > 100e-6

    # The fresh rebuild restored one exact context with full provenance.
    restored = summary['restored']
    assert restored.profile_id == prov[dl.T_STEP_LIGHT]['profile_id']
    assert restored.origin_version_id == summary['version_id']
    assert restored.target_version_id == summary['version_id']
    assert restored.current_set_id
    assert restored.current_count > 0 and restored.measurement_count == 1

    # Restored currents drove EM sizing above min width on the two internal
    # nets; port nets stayed at min width by design.
    em = summary['em']
    for net in ('tail', 'vint'):
        assert em[net]['peak_a'] is not None
        assert em[net]['width_um'] == pytest.approx(
            min(dl.MET1.max_width, em[net]['peak_a'] / dl.MET1.j_max))
        assert em[net]['width_um'] > dl.MET1.min_width
    for net in ('VOUT', 'VDD', 'VSS', 'VINP', 'VINM', 'IBIAS'):
        assert em[net]['peak_a'] is None
        assert em[net]['width_um'] == pytest.approx(dl.MET1.min_width)

    # Injected parasitics changed a measured result in the physical direction
    # and by more than the sweep's 1.2 % grid (measured ~7.7 %).
    bw_pre, bw_post = summary['bw_pre_hz'], summary['bw_post_hz']
    assert bw_pre > dl.INTENT['bw_hz_min']
    drop = (bw_pre - bw_post) / bw_pre
    assert 0.03 < drop < 0.30, f'bandwidth drop {drop:.1%} outside the expected band'
    assert summary['post_version_id'] != summary['version_id']

    # Both mismatches raised the documented typed error (the example asserts
    # itself that nothing was mutated).
    assert summary['mismatch'] == {'edited': 'EvidenceMismatchError',
                                   'post_layout': 'EvidenceMismatchError'}

    # Before/after reports and the netlist exist and carry the effect.
    out = summary['out_dir']
    for name in ('report_pre_layout.json', 'report_pre_layout.html',
                 'report_post_layout.json', 'report_post_layout.html',
                 'pre_vs_post_layout.md', 'design_loop.db'):
        path = os.path.join(out, name)
        assert os.path.getsize(path) > 0, name
    recon = summary['report_post'].spec_reconciliation
    statuses = {o.spec.key: o.status for o in recon.outcomes}
    assert statuses == {'settle_light': 'PASS', 'settle_heavy': 'PASS',
                        'gain': 'PASS', 'bw_pre_layout': 'PASS',
                        'bw_post_layout': 'PASS'}
    netlist = open(summary['netlist_path']).read()
    assert netlist.startswith('.subckt sampling_buffer ')
    assert 'R__par_R_VOUT' in netlist and 'XMM1 ' in netlist


# --------------------------------------------------------------------------- #
# slow — the documented command, from an installed wheel, foreign cwd
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@needs_pdk
@needs_ngspice
def test_walkthrough_runs_from_installed_wheel(tmp_path):
    pytest.importorskip('build', reason="the 'build' package (dev extra) is "
                                        "required to build the wheel")
    dist = tmp_path / 'dist'
    subprocess.run([sys.executable, '-m', 'build', '--wheel', '-o', str(dist),
                    REPO_ROOT], check=True, capture_output=True, text=True)
    (wheel,) = dist.glob('*.whl')
    site = tmp_path / 'site'
    zipfile.ZipFile(wheel).extractall(site)
    assert (site / 'silk' / 'examples' / 'design_loop_walkthrough.py').is_file()
    # The GF180MCU sibling ships in the same wheel (its own slow test runs it).
    assert (site / 'silk' / 'examples' / 'design_loop_walkthrough_gf180mcu.py').is_file()

    env = dict(os.environ, PYTHONPATH=str(site), MPLBACKEND='Agg')
    cwd = tmp_path / 'elsewhere'
    cwd.mkdir()
    probe = subprocess.run(
        [sys.executable, '-c', 'import silk; print(silk.__file__)'],
        cwd=str(cwd), env=env, capture_output=True, text=True, check=True)
    assert probe.stdout.strip().startswith(str(site)), probe.stdout

    out = tmp_path / 'out'
    run = subprocess.run(
        [sys.executable, '-m', 'silk.examples.design_loop_walkthrough', str(out)],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    assert 'EM: above min width' in run.stdout
    assert 'EvidenceMismatchError' in run.stdout
    assert run.stdout.rstrip().splitlines()[-1].startswith('Done.')
    assert (out / 'sampling_buffer_post_layout.cir').stat().st_size > 0
    assert (out / 'report_post_layout.html').stat().st_size > 0
