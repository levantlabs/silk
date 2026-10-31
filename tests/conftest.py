"""
Shared pytest fixtures for Module test suite.

Fixtures provide:
- Temporary directories for netlist output
- Real PDK instances loaded from YAML (with fake library path)
- Mock for ngspice invocation (os.system)
- Pre-configured Module instances with devices added
- In-memory Database for DB tests
"""

import sys
import os
import pytest

# Add project root to path so imports like `import Module` work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'circuit_db'))

from silk import sources
from silk.simulation.analysis import Transient, AC, DC, Noise


def _settings_file_into_environment():
    """Resolve the silk settings file ONCE, export what it sets into the
    environment, then disable the file for the rest of the session.

    The suite was written against environment variables: probes such as
    ``tests/_env.py`` and module-level skip conditions read them at
    collection, and many tests ``monkeypatch.delenv`` a setting to exercise
    silk's built-in default. Exporting the file's values keeps the first
    group agreeing with silk (environment > file, so resolution is
    unchanged), and disabling the file (empty ``SILK_SETTINGS``) keeps a
    ``delenv`` meaning "unset" instead of falling through to the user's
    file. Subprocesses inherit both. Settings tests that need a file point
    ``SILK_SETTINGS``/``HOME``/``XDG_CONFIG_HOME`` at ``tmp_path``.
    """
    from silk import settings
    for st in settings.all_settings():
        if st.source == 'file':
            os.environ[st.name] = st.value
    os.environ[settings.SETTINGS_ENV] = ''


_settings_file_into_environment()


# ---------------------------------------------------------------------------
# Directory / path fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session', autouse=True)
def _bind_scratch_dir(tmp_path_factory):
    """Give ``tests._env.scratch_dir`` the session's tmp_path_factory."""
    from tests import _env
    _env._scratch_factory = tmp_path_factory
    yield
    _env._scratch_factory = None


@pytest.fixture
def tmp_dir(tmp_path):
    """Temporary directory for netlist files and simulation output."""
    return str(tmp_path)


@pytest.fixture
def project_root():
    """Absolute path to the Module project root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def pdk_yaml_path(project_root):
    """Path to the real PDK YAML device details file."""
    return os.path.join(project_root, 'silk', 'pdk', 'data', 'skywater_130_device_details.yaml')


# ---------------------------------------------------------------------------
# PDK fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def stub_lib_path(tmp_path_factory):
    """A stand-in ``.lib`` file that actually exists on disk.

    Unit tests mock the *simulator*, not the *PDK install* — and since TASK-71
    a run whose deck names PDK devices checks that the model library is really
    there (``modules_sim._require_pdk_library``), because ``load_pdk`` no
    longer does it at load time. A path like ``/fake/path/sky130.lib.spice``
    would now read as "the PDK is not installed" and refuse the mocked run.
    The file's *contents* are still never read by anything under test.
    """
    path = tmp_path_factory.mktemp('stub-pdk') / 'sky130.lib.spice'
    path.write_text('* stub sky130 lib for unit tests: never parsed\n')
    return str(path)


@pytest.fixture
def pdk(pdk_yaml_path, stub_lib_path):
    """
    Real BasePDK instance loaded from the actual YAML file.

    Uses a stub library path since we never run ngspice in unit tests.
    The PDK object will have all device attributes (nmos1p8, pmos1p8, etc.)
    and a populated model_names list.
    """
    from silk.pdk.basepdk import BasePDK

    return BasePDK(pdk_yaml_path, library_location=stub_lib_path)


# ---------------------------------------------------------------------------
# ngspice mock fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ngspice(monkeypatch, mock_ngspice_version):
    """
    Patch subprocess.run (used by Module.run_simulator) to simulate a
    successful ngspice run without actually invoking the simulator.

    The mock:
    - Captures the command string for assertion
    - Returns 0 (success)
    - Creates complete fake plots for the analyses in the launched deck

    Usage in tests:
        def test_run(mock_ngspice, basic_netlist):
            basic_netlist.run_simulator()
            assert mock_ngspice['called']
            assert 'ngspice' in mock_ngspice['command']
    """
    import subprocess
    from unittest.mock import MagicMock

    state = {'called': False, 'command': [], 'call_count': 0}

    def fake_subprocess_run(cmd, **kwargs):
        state['called'] = True
        state['command'] = cmd  # list of args
        state['call_count'] += 1

        # Extract the raw file path: ngspice -b -r <rawfile> -o <logfile> <netlist>
        if cmd and 'ngspice' in cmd[0]:
            try:
                r_idx = cmd.index('-r')
                raw_path = cmd[r_idx + 1]
                os.makedirs(os.path.dirname(raw_path), exist_ok=True)
                from pathlib import Path
                from tests._mock_rawfile import write_complete_rawfile
                write_complete_rawfile(raw_path, Path(cmd[-1]).read_text())
            except (ValueError, IndexError):
                pass

            try:
                o_idx = cmd.index('-o')
                log_path = cmd[o_idx + 1]
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, 'w') as f:
                    f.write('ngspice mock: simulation complete\n')
            except (ValueError, IndexError):
                pass

        mock_result = MagicMock()
        mock_result.returncode = 0
        return mock_result

    monkeypatch.setattr(subprocess, 'run', fake_subprocess_run)
    return state


# ---------------------------------------------------------------------------
# Module fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_netlist(pdk, tmp_dir):
    """
    A bare Module instance with real PDK and temp directory.
    No elements, sources, or sim commands added yet.
    """
    from silk.circuit.modules import Module

    return Module(
        'PDK_skywater130',
        tmp_dir,
        'test_circuit.cir',
        pdk_details=pdk,
    )


@pytest.fixture
def basic_netlist(pdk, tmp_dir):
    """
    A Module instance with an nmos and pmos already added and connected,
    ready for generate_netlist() to be called.

    Devices:
    - 'mn1': nmos1p8 (W=1, L=0.15) connected d=out, g=in, s=gnd, b=gnd
    - 'mp1': pmos1p8 (W=2, L=0.15) connected d=out, g=in, s=vdd, b=vdd

    Nets added: vdd, gnd, in, out
    """
    from silk.circuit.modules import Module

    ckt = Module(
        'PDK_skywater130',
        tmp_dir,
        'basic_circuit.cir',
        pdk_details=pdk,
    )

    # Add nets
    ckt.add_net('vdd')
    ckt.add_net('gnd')
    ckt.add_net('in')
    ckt.add_net('out')

    # Keep legacy devices for pin_list/param_list delegation compatibility tests.
    # Add NMOS
    ckt.add_element(
        pdk.nmos1p8, 'mn1',
        nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
        parameters={'w': 1, 'l': 0.15},
    )

    # Add PMOS
    ckt.add_element(
        pdk.pmos1p8, 'mp1',
        nets={'d': 'out', 'g': 'in', 's': 'vdd', 'b': 'vdd'},
        parameters={'w': 2, 'l': 0.15},
    )

    return ckt


@pytest.fixture
def netlist_with_sources(basic_netlist):
    """
    Extends basic_netlist with voltage sources and a transient sim command.

    Sources:
    - 'vdd_supply': DC 1.8V on vdd/gnd
    - 'vin': DC 0.9V on in/gnd

    Simulation: transient 1n step, 100n stop
    """
    ckt = basic_netlist

    ckt.add_source(sources.VoltageSource('vdd_supply', positive='vdd', negative='gnd', dc=1.8))

    ckt.add_source(sources.VoltageSource('vin', positive='in', negative='gnd', dc=0.9))

    ckt.add_sim_type(Transient(step=1e-9, stop=1e-7))

    return ckt


# ---------------------------------------------------------------------------
# Signal analysis fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_signal():
    """
    A NgspiceSignal instance with simple synthetic data for testing
    arithmetic operators and plotting.
    """
    import numpy as np
    from silk.signals import NgspiceSignal

    x = np.linspace(0, 1e-6, 100)
    y = np.sin(2 * np.pi * 1e6 * x)

    return NgspiceSignal(
        name='v(out)',
        type='voltage',
        xlabel='time',
        ylabel='voltage',
        xvalues=x,
        yvalues=y,
    )


@pytest.fixture
def sample_signal_pair():
    """
    Two compatible NgspiceSignal instances (same x-axis) for testing
    arithmetic operations (add, subtract).
    """
    import numpy as np
    from silk.signals import NgspiceSignal

    x = np.linspace(0, 1e-6, 100)
    y1 = np.sin(2 * np.pi * 1e6 * x)
    y2 = 0.5 * np.cos(2 * np.pi * 1e6 * x)

    sig1 = NgspiceSignal('v(out)', 'voltage', 'time', 'voltage', x, y1)
    sig2 = NgspiceSignal('v(ref)', 'voltage', 'time', 'voltage', x, y2)

    return sig1, sig2


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def in_memory_db():
    """
    Database instance backed by an in-memory SQLite database.
    Tables are created automatically. No cleanup needed.
    """
    from silk.db.circuit_db import Database

    db = Database(':memory:')
    yield db
    db.close()


@pytest.fixture
def db_with_circuit(in_memory_db):
    """
    In-memory database with one circuit version already created.

    Returns (db, circuit_id) tuple.
    Circuit: name='test_inverter', version='1.0', stage=DEV
    """
    from tests.db_snapshot_helpers import create_sealed_snapshot

    circuit_id = create_sealed_snapshot(
        in_memory_db,
        name='test_inverter',
        version='1.0',
        description='Test inverter for unit tests',
    )

    return in_memory_db, circuit_id


@pytest.fixture
def db_with_building_circuit(in_memory_db):
    """In-memory database with one mutable building snapshot."""
    from tests.db_snapshot_helpers import create_building_snapshot

    circuit_id = create_building_snapshot(
        in_memory_db,
        name='test_inverter',
        version='1.0',
        description='Test inverter for unit tests',
    )
    return in_memory_db, circuit_id


# ---------------------------------------------------------------------------
# Module fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_module(pdk, tmp_dir):
    """
    A Module instance with module name and pins defined,
    and one nmos device added. Ready for netlist generation as subcircuit.

    The ground port is 'vss', not 'gnd': this is a ``.subckt`` interface, and
    since TASK-36 a port named 'gnd' is rejected at emission (ngspice folds
    the port name onto global node 0 and silently discards the parent's
    wiring). The top-level ``simple_circuit`` fixture above keeps its 'gnd'
    *net* — that case is not a port and is not affected.
    """
    from silk.circuit.modules import Module

    mod = Module(
        'PDK_skywater130',
        tmp_dir,
        'test_module.cir',
        pdk_details=pdk,
    )

    mod.define_module_name('inv_cell')
    mod.define_module_nets({'in': None, 'out': None, 'vdd': None, 'vss': None})

    # Keep legacy devices for pin_list/param_list delegation compatibility tests.
    mod.add_element(
        pdk.nmos1p8, 'mn',
        nets={'d': 'out', 'g': 'in', 's': 'vss', 'b': 'vss'},
        parameters={'w': 1, 'l': 0.15},
    )

    mod.add_element(
        pdk.pmos1p8, 'mp',
        nets={'d': 'out', 'g': 'in', 's': 'vdd', 'b': 'vdd'},
        parameters={'w': 2, 'l': 0.15},
    )

    return mod


# ---------------------------------------------------------------------------
# Source fixtures
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# SimulatorBackend registry isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _restore_simulator_backend_registry():
    from silk.simulation.backend import SimulatorBackend
    saved = dict(SimulatorBackend._registry)
    yield
    SimulatorBackend._registry.clear()
    SimulatorBackend._registry.update(saved)


# ---------------------------------------------------------------------------
# pyplot figure isolation (TASK-561)
# ---------------------------------------------------------------------------
#
# ``Signal.plot()`` with no ``ax`` draws on ``plt.gca()``, which is process
# global. A test that leaves a figure open would hand its axes to any later
# test that plots without an axis of its own. Close every figure after each
# test. Only touch pyplot when something already imported it, so tests that
# never plot do not pay for the import.

@pytest.fixture(autouse=True)
def _close_pyplot_figures():
    yield
    import sys
    plt = sys.modules.get('matplotlib.pyplot')
    if plt is not None:
        plt.close('all')


# ---------------------------------------------------------------------------
# Netlister fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def netlister(pdk):
    """A PySpiceNetlister instance with real PDK."""
    from silk.circuit.spice_netlister import PySpiceNetlister

    return PySpiceNetlister(pdk)


# ---------------------------------------------------------------------------
# Guard: no fast-suite skip may key on an artifact a previous run left behind
# ---------------------------------------------------------------------------
#
# TASK-103: a fresh worktree reported 2 more skips than the main checkout
# because two tests skipped on files only the main tree happened to have —
# testSpice/ldo_example.cir (an example's emitted deck) and
# tests/browser/node_modules (npm install, a legitimate build-step gate).
# The first kind is banned outright: a fast test must generate or commit its
# own input, never rely on a path an earlier run happened to leave behind.
# This hook records every skip reason seen during the session and fails the
# run if one names such a path; `_skip_reason_names_generated_artifact` is
# the pure predicate, unit-tested directly in test_skip_reason_guard.py.

# 'silk-output/' is where the examples write since TASK-243 (the old
# 'examples/output/' stays listed so an older leftover still trips it).
_FORBIDDEN_SKIP_PATH_MARKERS = ('testSpice/', 'examples/output/',
                                'silk-output/')
_collected_skip_reasons = []  # [(nodeid, reason), ...], filled during the run


def _skip_reason_names_generated_artifact(reason):
    """True if a skip `reason` names a path under testSpice/,
    silk-output/ or examples/output/ — i.e. the test is gated on an artifact a
    previous run left behind rather than input it generates or ships itself."""
    return any(marker in reason for marker in _FORBIDDEN_SKIP_PATH_MARKERS)


def _skip_reason_from_report(report):
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr)


def pytest_runtest_logreport(report):
    # An xfail is reported with ``skipped=True`` too, but it is not a skip:
    # its longrepr is the failure traceback, which may quote a path under
    # testSpice/ for reasons that have nothing to do with gating on a leftover
    # artifact (TASK-36's TASK-83 pins did exactly that and made the whole
    # run exit 1 with 0 failed). Only a real skip is subject to the guard.
    if report.skipped and not hasattr(report, 'wasxfail'):
        _collected_skip_reasons.append(
            (report.nodeid, _skip_reason_from_report(report)))


def pytest_sessionfinish(session, exitstatus):
    violations = [
        (nodeid, reason) for nodeid, reason in _collected_skip_reasons
        if _skip_reason_names_generated_artifact(reason)
    ]
    if violations:
        detail = '\n'.join(f'  {nodeid}: {reason}' for nodeid, reason in violations)
        print(
            '\n\nGUARD FAILURE (TASK-103): skip reason names a path under '
            'testSpice/, silk-output/ or examples/output/ — generate the '
            'input instead of '
            f'gating on a leftover artifact:\n{detail}\n'
        )
        session.exitstatus = 1


@pytest.fixture
def mock_ngspice_version(monkeypatch):
    """Simulation-only doubles need a separate version identity.

    Keep the real probe/cache out of fake subprocesses and avoid polluting it
    for real simulator tests later in the same session. Version-policy tests
    provide their own versions; these doubles model a current release.
    """
    from silk.simulation import _ngspice_version, _ngspice_compat
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: '47')
    monkeypatch.setattr(_ngspice_compat, '_advised_binaries', set())
