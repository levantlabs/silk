"""Separate-process calibration controls, with synthetic launch data and real SPICE."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from tests._env import needs_ngspice, needs_pdk

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

# Each child imports the worktree, never the main checkout's editable install.
# The guard blocks every external process and model read during restoration.
GUARD = '''
import builtins
import os
import matplotlib.pyplot
import subprocess
from pathlib import Path
from silk.simulation.backend import NgspiceBackend, XyceBackend

def forbidden(*args, **kwargs):
    raise AssertionError("restore attempted an external launch")
NgspiceBackend.run_module = forbidden
XyceBackend.run_module = forbidden
subprocess.Popen = forbidden
original_open = builtins.open
original_path_open = Path.open

def no_models(path):
    if str(path).endswith((".spice", ".lib", ".osdi")):
        raise FileNotFoundError("model libraries unavailable in restore control")

def guarded_open(path, *args, **kwargs):
    no_models(path)
    return original_open(path, *args, **kwargs)

def guarded_path_open(path, *args, **kwargs):
    no_models(path)
    return original_path_open(path, *args, **kwargs)
builtins.open = guarded_open
Path.open = guarded_path_open
original_isfile = os.path.isfile
original_path_isfile = Path.is_file

def model_path(path):
    return str(path).endswith((".spice", ".lib", ".osdi"))

os.path.isfile = lambda path: False if model_path(path) else original_isfile(path)
Path.is_file = lambda path: False if model_path(path) else original_path_isfile(path)
from silk.examples.source_follower_example import pdk
assert not Path(pdk.library_location).is_file()
'''

SYNTHETIC = '''
import matplotlib.pyplot
import subprocess
import silk.circuit.modules_sim as sim_methods

# The synthetic producer has no physical model dependency. Replace only the
# external-model preflight and launch boundary; keep emission/evidence real.
sim_methods._require_pdk_library = lambda pdk: None

def forbidden_process(*args, **kwargs):
    raise AssertionError("synthetic control attempted an external launch")
subprocess.Popen = forbidden_process
from silk.simulation.backend import NgspiceBackend, RawResult
from silk.signals.rawfile import read_rawfile
from pathlib import Path

def synthetic(self, files, **kwargs):
    deck = Path(files.netlist_path).read_text()
    ac = ".ac " in deck.lower()
    if ac:
        title, flags, axis = "AC Analysis", "complex", "frequency"
        # Complete 20 points/decade grid through 1 GHz, gain .79, pole 170 MHz.
        import numpy as np
        x = np.logspace(0, 9, 181)
        y = .79 / (1 + 1j*x/170e6)
    else:
        title, flags, axis = "Transient Analysis", "real", "time"
        import numpy as np
        x = np.linspace(0, 120e-9, 601)
        y = np.full_like(x, .31)
    path = Path(files.rawfile_path)
    header = (f"Title: synthetic control\\nDate: today\\nPlotname: {title}\\n"
              f"Flags: {flags}\\nNo. Variables: 2\\nNo. Points: {len(x)}\\nVariables:\\n"
              f"0\\t{axis}\\t{axis}\\n1\\tv(vout)\\tvoltage\\nValues:\\n")
    rows = []
    for i, (t, v) in enumerate(zip(x, y)):
        if ac:
            rows.append(f"{i}\\t{t},0\\n\\t{v.real},{v.imag}\\n")
        else:
            rows.append(f"{i}\\t{t}\\n\\t{v}\\n")
    path.write_text(header + "".join(rows))
    log = path.with_suffix(".log")
    log.write_text("Doing analysis at TEMP = 27.000000 and TNOM = 27.000000\\n")
    return RawResult(read_rawfile(str(path)), str(path), files), str(log), None
NgspiceBackend._launch_prepared = synthetic
'''


def child(tmp_path, code, *, synthetic=False, guard=False, real_dependencies=False):
    env = dict(os.environ, PYTHONPATH=str(ROOT), MPLBACKEND='Agg',
               MPLCONFIGDIR=str(tmp_path / 'mpl'), SILK_EXAMPLES_OUT=str(tmp_path / 'out'))
    if not real_dependencies:
        # Isolated child overrides exercise the bare-checkout contract; the
        # real integration inherits the owner’s configured settings.
        env['PDK_ROOT'] = str(tmp_path / 'no-installed-pdk')
        env['SILK_NGSPICE_BIN'] = str(tmp_path / 'no-ngspice')
        env['SILK_XYCE_BIN'] = str(tmp_path / 'no-Xyce')
    script = ('import silk; print(silk.__file__)\n'
              + (GUARD if guard else '') + (SYNTHETIC if synthetic else '')
              + textwrap.dedent(code))
    result = subprocess.run([PYTHON, '-c', script], cwd=ROOT, env=env,
                            text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert str(ROOT / 'silk/__init__.py') in result.stdout
    return result


def phase_code(tmp_path, phase):
    return f'''
    import sys
    sys.argv = ['source_follower_calibrate', {phase!r}, '--database',
                {str(tmp_path / 'evidence.db')!r}, '--validation-output',
                {str(tmp_path / (phase + '.json'))!r}]
    import runpy
    runpy.run_module('silk.examples.source_follower_calibrate', run_name='__main__')
    '''


def assert_restart(tmp_path, *, synthetic):
    child(tmp_path, phase_code(tmp_path, 'characterize'), synthetic=synthetic,
          real_dependencies=not synthetic)
    # Remove comparison evidence before restore: it cannot supply calibration.
    reference_path = tmp_path / 'characterize.json'
    reference = json.loads(reference_path.read_text())
    reference_path.unlink()
    result = child(tmp_path, phase_code(tmp_path, 'restore'), guard=True,
                   real_dependencies=not synthetic)
    restored = json.loads((tmp_path / 'restore.json').read_text())
    for param, nominal in dict(gain=.85, bw_hz=1e8, offset=.55).items():
        assert reference['params'][param] != pytest.approx(nominal, rel=1e-3)
        assert restored['params'][param] == pytest.approx(reference['params'][param], rel=1e-12)
    np.testing.assert_allclose(restored['time_s'], reference['time_s'], rtol=0, atol=1e-20)
    np.testing.assert_allclose(restored['vout_V'], reference['vout_V'], rtol=1e-12, atol=1e-12)
    assert 'deck=checked' in result.stdout
    return reference


def test_synthetic_restart(tmp_path):
    """No simulator launches: synthetic evidence still takes public storage paths."""
    assert_restart(tmp_path, synthetic=True)


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_real_sky130_restart(tmp_path):
    assert_restart(tmp_path, synthetic=False)


@pytest.fixture
def stored(tmp_path):
    child(tmp_path, phase_code(tmp_path, 'characterize'), synthetic=True)
    return str(tmp_path / 'evidence.db')


@pytest.mark.parametrize('change', ['design', 'ac_bench', 'step_bench', 'missing_test', 'corner'])
def test_refuses_incompatible_or_missing_context(tmp_path, stored, change):
    child(tmp_path, f'''
    from silk.examples import source_follower_calibrate as example
    from silk.exceptions import EvidenceMismatchError, EvidenceNotFoundError
    original = example._persistent_recipe
    def changed(path, **kwargs):
        mod = original(path, **kwargs)
        if {change!r} == 'design':
            from silk.elements import elements
            mod.add(elements.capacitor('extra', p='VOUT', n='VSS', value=1e-12))
        if {change!r} in ('ac_bench', 'step_bench'):
            from silk import sources
            name = 'ac' if {change!r} == 'ac_bench' else 'beh_check'
            mod.get_test(name).add_source(sources.VoltageSource(
                'extra', positive='extra', negative='0', dc=.1))
        if {change!r} == 'missing_test':
            del mod._tests['ac']
        return mod
    example._persistent_recipe = changed
    try:
        example.restore({stored!r}, corner='ff' if {change!r} == 'corner' else 'tt')
    except (EvidenceMismatchError, EvidenceNotFoundError):
        pass
    else:
        raise AssertionError('incompatible evidence accepted')
    ''', guard=True)


def test_missing_database(tmp_path):
    child(tmp_path, f'''
    from silk.examples.source_follower_calibrate import restore
    try:
        restore({str(tmp_path / 'absent.db')!r})
    except FileNotFoundError as error:
        assert 'characterize first' in str(error)
    else:
        raise AssertionError('missing database accepted')
    ''', guard=True)


def test_each_parameter_reaches_executed_instance(tmp_path, stored):
    # An independent assertion on the executing copy prevents the two restart
    # phases from sharing a bug that leaves the same nominal parameter behind.
    child(tmp_path, f'''
    from silk.examples import source_follower_calibrate as example
    from silk.circuit.module_test import ModuleTestbench
    run = ModuleTestbench.run
    seen = []
    def checked(self, *args, **kwargs):
        if self.netlist.simulator == 'behavioral':
            params = self.netlist.elem_dict['dut']['device']._behavioral_params
            assert params['gain'] == .79
            assert abs(params['offset'] - .48) < 1e-12
            assert abs(params['bw_hz'] - 1e8) > 1e7
            seen.append(True)
        return run(self, *args, **kwargs)
    ModuleTestbench.run = checked
    example.restore({stored!r})
    assert seen == [True]
    ''', guard=True)


@pytest.mark.parametrize('kind', ['empty', 'no_transient', 'ambiguous'])
def test_refuses_incomplete_or_ambiguous_evidence(tmp_path, kind):
    db = str(tmp_path / 'evidence.db')
    child(tmp_path, f'''
    from silk.examples.source_follower_calibrate import _persistent_recipe, GAIN_METRIC, BW_METRIC
    mod = _persistent_recipe({db!r})
    mod.create_circuit_version()
    if {kind!r} != 'empty':
        mod.run_test('ac', corner='tt', store=True, store_options={{
            'sim_type': 'ac', 'units': {{GAIN_METRIC: 'V/V', BW_METRIC: 'Hz'}}}})
    if {kind!r} == 'ambiguous':
        mod.run_test('ac', corner='tt', store=True, store_options={{
            'sim_type': 'ac', 'tool_name': 'alternative-label',
            'units': {{GAIN_METRIC: 'V/V', BW_METRIC: 'Hz'}}}})
        mod.run_test('beh_check', corner='tt', store=True,
                     store_options={{'sim_type': 'tran', 'units': {{'settled_vout_V': 'V'}}}})
    mod.close()
    ''', synthetic=True)
    child(tmp_path, f'''
    from silk.examples.source_follower_calibrate import restore
    from silk.exceptions import EvidenceNotFoundError, EvidenceAmbiguityError
    try:
        restore({db!r})
    except (EvidenceNotFoundError, EvidenceAmbiguityError):
        pass
    else:
        raise AssertionError('incomplete or ambiguous evidence accepted')
    ''', guard=True)


def test_uncheckable_evidence_keeps_warning(tmp_path):
    # Opaque decks retain the public API's usable-but-not-checked policy.
    setup = '''
from silk.examples import source_follower_calibrate as example
original = example._persistent_recipe
def opaque(path, **kwargs):
    mod = original(path, **kwargs)
    for name in ('ac',):
        mod.get_test(name).netlist.add_custom_line('* opaque control')
    return mod
example._persistent_recipe = opaque
'''
    db = str(tmp_path / 'opaque.db')
    child(tmp_path, setup + f'example.characterize({db!r})\n', synthetic=True)
    result = child(tmp_path, setup + f'''
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    example.restore({db!r})
assert any('Stored evidence not checked' in str(w.message) for w in caught)
''', guard=True)
    assert 'deck=not checked' in result.stdout


@pytest.mark.parametrize('option,value', [('--corner', 'ff'), ('--database', 'unused.db'),
                                         ('--validation-output', 'unused.json')])
def test_phase_options_require_a_phase_before_any_launch(tmp_path, option, value):
    child(tmp_path, f'''
    import runpy
    import sys
    sys.argv = ['source_follower_calibrate', {option!r}, {value!r}]
    try:
        runpy.run_module('silk.examples.source_follower_calibrate', run_name='__main__')
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError('phase option silently ignored')
    ''', guard=True)
