"""OSDI (compiled Verilog-A) support — the silk side only.

The division of labour, decided 2026-08-22: **compiling** the models is
environment setup (openvaf + the PDK's own install.py, or pdksetup), exactly
like installing ngspice — silk never runs a compiler. silk's three jobs:

1. The manifest **declares the fact** (`library.osdi:` — paths, like lib_path).
2. Emission **loads them**: `pre_osdi <path>` at the top of the .control
   block, which ngspice executes BEFORE parsing (verified on 42 and 47 — the
   "Error opening osdi lib" for a bogus path prints before the `Circuit:`
   banner). Pure deck emission: no .spiceinit, no global state.
3. **Fail loudly** where it cannot work — and only there. Emitting a deck
   never requires the plugins (a deck written on this box is valid on any box
   whose plugins ARE compiled — the same contract as ngspice itself, which
   emission also does not require); the RUN is what cannot work, so
   declared-but-missing fails at simulate time with instructions. Structural
   impossibilities do fail at netlist time: Xyce (no OSDI loader) and
   batch-write mode (emits no .control block, so the load lines have no home).

`python -m silk.doctor` reports declared-but-uncompiled per PDK, closing the
"presence is not usable" gap where SG13G2 read `ok` on a machine where no
SG13G2 run could succeed.
"""
import os
from pathlib import Path

import pytest

from silk.exceptions import NotSupportedError, SimulationError
from silk.pdk import load_pdk

SG_NG = 'ihp_sg13g2/ihp-sg13g2/libs.tech/ngspice'


def _stub_sg13g2_root(root, *, compiled):
    """A stub SG13G2 install; `compiled=True` adds the four .osdi stubs."""
    models = root / SG_NG / 'models'
    models.mkdir(parents=True)
    (models / 'cornerMOSlv.lib').write_text('* stub cornerMOSlv.lib\n')
    if compiled:
        osdi = root / SG_NG / 'osdi'
        osdi.mkdir()
        for name in ('psp103', 'psp103_nqs', 'r3_cmc', 'mosvar'):
            (osdi / f'{name}.osdi').write_bytes(b'\x7fELF stub')
    return root


def _netlist(pdk, tmp_path, **kwargs):
    from silk import sources
    from silk.circuit.modules import Module
    m = Module('sg13g2', str(tmp_path / 'out'), 'x.cir', pdk)
    m.add(pdk.nmos_lv('M1', d='d', g='g', s='0', b='0', w=1.0, l=0.13))
    m.add_source(sources.VoltageSource('D', positive='d', negative='0', dc=1.2))
    for k, v in kwargs.items():
        setattr(m, k, v)
    from pathlib import Path
    return Path(m.generate_netlist()).read_text()


def test_manifest_osdi_paths_resolve_against_the_pdk_root(tmp_path):
    root = _stub_sg13g2_root(tmp_path, compiled=True)
    pdk = load_pdk('sg13g2', pdk_root=str(root))
    assert len(pdk.osdi_files) == 4
    assert all(os.path.isabs(p) and str(root) in p for p in pdk.osdi_files)


def test_compiled_models_are_loaded_via_pre_osdi_before_run(tmp_path):
    root = _stub_sg13g2_root(tmp_path, compiled=True)
    deck = _netlist(load_pdk('sg13g2', pdk_root=str(root)), tmp_path)
    lines = deck.splitlines()
    pre = [i for i, l in enumerate(lines) if l.startswith('pre_osdi ')]
    assert len(pre) == 4, deck
    # Inside .control, before run — pre_osdi is executed before parsing, but
    # putting it first also keeps the deck readable in the order it acts.
    assert lines.index('.control') < pre[0] < lines.index('run')
    assert any(l.endswith('psp103.osdi') for l in lines)


def test_uncompiled_models_still_emit_a_valid_deck(tmp_path):
    # Emission is unconditional: the deck is valid wherever the plugins ARE
    # compiled. Only the RUN requires the files (next test).
    root = _stub_sg13g2_root(tmp_path, compiled=False)
    deck = _netlist(load_pdk('sg13g2', pdk_root=str(root)), tmp_path)
    assert deck.count('pre_osdi ') == 4


def test_declared_but_uncompiled_fails_at_simulate_time_with_instructions(tmp_path):
    from silk.circuit.modules_sim import _require_osdi_compiled
    root = _stub_sg13g2_root(tmp_path, compiled=False)
    with pytest.raises(SimulationError) as err:
        _require_osdi_compiled(load_pdk('sg13g2', pdk_root=str(root)))
    msg = str(err.value)
    assert 'openvaf' in msg and 'install.py' in msg and 'psp103.osdi' in msg
    # ...and compiled means no complaint.
    _require_osdi_compiled(load_pdk(
        'sg13g2', pdk_root=str(_stub_sg13g2_root(tmp_path / 'c', compiled=True))))


def test_a_pdk_without_osdi_emits_no_pre_osdi_line(tmp_path):
    # sky130 declares none; its decks must be byte-identical to before the
    # feature existed. (The goldens enforce byte-identity; this states why.)
    lib = tmp_path / 'sky130A' / 'libs.tech' / 'ngspice'
    lib.mkdir(parents=True)
    (lib / 'sky130.lib.spice').write_text('* stub\n')
    pdk = load_pdk('sky130', pdk_root=str(tmp_path))
    assert pdk.osdi_files == []
    from silk.circuit.spice_netlister import PySpiceNetlister
    assert PySpiceNetlister(pdk)._osdi_lines() == ''


def test_xyce_refuses_an_osdi_pdk_with_the_reason(tmp_path):
    root = _stub_sg13g2_root(tmp_path, compiled=True)
    pdk = load_pdk('sg13g2', pdk_root=str(root))
    from silk.circuit.xyce_netlister import XyceNetlister
    with pytest.raises(NotSupportedError, match='ngspice'):
        XyceNetlister(pdk).generate_pdk_include('typical')


def test_batch_write_mode_refuses_an_osdi_pdk(tmp_path):
    # save_device_currents emits no .control block — the pre_osdi lines would
    # have no home, and the deck would die in ngspice's parser instead.
    root = _stub_sg13g2_root(tmp_path, compiled=True)
    pdk = load_pdk('sg13g2', pdk_root=str(root))
    from silk.circuit.spice_netlister import PySpiceNetlister
    with pytest.raises(SimulationError, match='pre_osdi|OSDI'):
        PySpiceNetlister(pdk).generate_output_block(
            'out', {}, save_device_currents=True)


def test_doctor_reports_uncompiled_osdi_as_not_runnable(tmp_path, monkeypatch):
    from silk.doctor import check_pdks
    monkeypatch.setenv('PDK_ROOT', str(_stub_sg13g2_root(tmp_path, compiled=False)))
    sg = next(c for c in check_pdks() if c.name == 'pdk sg13g2')
    assert sg.status != 'ok' and 'OSDI' in sg.detail


def test_doctor_reports_compiled_osdi_as_ok(tmp_path, monkeypatch):
    from silk import doctor
    monkeypatch.setenv('PDK_ROOT', str(_stub_sg13g2_root(tmp_path, compiled=True)))
    # The stubs are not real plugins, so stand in for the load probe: this
    # test is about the file check, the probe has its own tests below.
    monkeypatch.setattr(doctor, '_osdi_load_problem', lambda osdi: None)
    sg = next(c for c in doctor.check_pdks() if c.name == 'pdk sg13g2')
    assert sg.status == 'ok'
    assert '4 OSDI plugins load' in sg.detail


# ---------------------------------------------------------------------------
# Compiled is not loadable: the OSDI *version* the ngspice binary speaks
# (bring-up 2026-09-02: openvaf-r emits OSDI v0.4; Ubuntu's ngspice-42 stops
# at v0.3 and then blames every PSP model — "could not find a valid
# modelname" — without ever saying OSDI. ngspice-47 loads them.)
# ---------------------------------------------------------------------------

# ngspice-42's output, verbatim: all four complaints on ONE line, no
# separator, printed before the `Circuit:` banner; then the generic errors.
NGSPICE_42_TEXT = (
    'Error: Library /pdk/osdi/psp103.osdi couldn\'t be loaded!\n'
    'Error: Library /pdk/osdi/psp103_nqs.osdi couldn\'t be loaded!\n'
    '\nNote: No compatibility mode selected!\n\n'
    'NGSPICE only supports OSDI v0.3 but "/pdk/osdi/psp103.osdi" targets '
    'v0.4!NGSPICE only supports OSDI v0.3 but "/pdk/osdi/psp103_nqs.osdi" '
    'targets v0.4!\n'
    'Circuit: * x\n'
    'Error on line:\n  n.xm1.nsg13_lv_nmos d g 0 0 xm1:sg13g2_lv_nmos_psp w=1e-06\n'
    'could not find a valid modelname\n'
)


def test_osdi_parser_names_the_version_mismatch_first():
    from silk.simulation.runner import osdi_problem_from_text
    got = osdi_problem_from_text(NGSPICE_42_TEXT)
    # One clause, the first plugin's — not the whole glued-together line and
    # not the generic "couldn't be loaded" that precedes it in the text.
    assert got == ('NGSPICE only supports OSDI v0.3 but '
                   '"/pdk/osdi/psp103.osdi" targets v0.4!')


def test_osdi_parser_falls_back_to_the_load_error_and_to_none():
    from silk.simulation.runner import osdi_problem_from_text
    assert osdi_problem_from_text(
        'Error opening osdi lib "/x.osdi": No such file or directory!\n'
        'Error: Library /x.osdi couldn\'t be loaded!\n'
    ) == 'Error opening osdi lib "/x.osdi": No such file or directory!'
    assert osdi_problem_from_text('Circuit: * fine\nngspice-47 done\n') is None
    assert osdi_problem_from_text('') is None
    assert osdi_problem_from_text(None) is None


def test_runner_failure_names_the_osdi_cause_and_the_fix(tmp_path, monkeypatch, mock_ngspice_version):
    """The generic failure path lists 'Error' lines from the log — none of
    which mention OSDI. With the version complaint in the log, the exception
    must lead with it and say what to do (SILK_NGSPICE_BIN)."""
    import subprocess
    from silk.simulation import runner as r
    results = tmp_path / 'res'
    results.mkdir()
    (tmp_path / 'x.cir').write_text('OSDI diagnostic fixture\n.end\n')
    def fake_run(argv, **kw):
        Path(argv[argv.index('-o') + 1]).write_text(NGSPICE_42_TEXT)
        return subprocess.CompletedProcess(argv, 1, stdout='', stderr='')
    monkeypatch.setattr(r.subprocess, 'run', fake_run)
    with pytest.raises(SimulationError) as ei:
        r.NgspiceRunner(str(tmp_path / 'x.cir'), str(results)).run()
    msg = str(ei.value)
    assert 'OSDI plugin load failed: NGSPICE only supports OSDI v0.3' in msg
    assert 'SILK_NGSPICE_BIN' in msg and 'ngspice needs release 44' in msg
    assert "couldn't be loaded" in msg            # the raw errors still follow


def test_runner_failure_without_osdi_is_unchanged(tmp_path, monkeypatch, mock_ngspice_version):
    import subprocess
    from silk.simulation import runner as r
    results = tmp_path / 'res'
    results.mkdir()
    (tmp_path / 'x.cir').write_text('OSDI diagnostic fixture\n.end\n')
    def fake_run(argv, **kw):
        Path(argv[argv.index('-o') + 1]).write_text('Error on line 3: singular matrix\n')
        return subprocess.CompletedProcess(argv, 1, stdout='', stderr='')
    monkeypatch.setattr(r.subprocess, 'run', fake_run)
    with pytest.raises(SimulationError) as ei:
        r.NgspiceRunner(str(tmp_path / 'x.cir'), str(results)).run()
    assert 'OSDI' not in str(ei.value)
    assert 'singular matrix' in str(ei.value)


def test_osdi_probe_is_none_for_no_plugins_or_no_ngspice(tmp_path):
    from silk.simulation.runner import osdi_load_problem
    assert osdi_load_problem([]) is None
    assert osdi_load_problem(['/x.osdi'],
                             ngspice=str(tmp_path / 'no-such-ngspice')) is None


def test_doctor_warns_when_ngspice_cannot_load_the_compiled_plugins(
        tmp_path, monkeypatch):
    """Present + compiled used to read `ok` on a box whose ngspice refuses the
    plugins' OSDI version — every SG13G2 run there fails. The probe closes it."""
    from silk import doctor
    monkeypatch.setenv('PDK_ROOT', str(_stub_sg13g2_root(tmp_path, compiled=True)))
    monkeypatch.setattr(
        doctor, '_osdi_load_problem',
        lambda osdi: 'NGSPICE only supports OSDI v0.3 but "x" targets v0.4!')
    monkeypatch.setattr(doctor, '_ngspice_label', lambda: 'ngspice-42')
    sg = next(c for c in doctor.check_pdks() if c.name == 'pdk sg13g2')
    assert sg.status == doctor.WARN                       # non-gating, honest
    assert 'ngspice-42 cannot load its OSDI plugins' in sg.detail
    assert 'targets v0.4' in sg.detail
    assert 'SILK_NGSPICE_BIN' in sg.fix and 'ngspice 44' in sg.fix


def test_doctor_probe_asks_the_ngspice_the_runner_would_use(monkeypatch, tmp_path):
    """The probe must follow $SILK_NGSPICE_BIN, or doctor would vouch for a
    binary the run never touches."""
    from silk import doctor
    from silk.simulation import runner as r
    seen = {}
    fake = tmp_path / 'ngspice'
    fake.write_text('#!/bin/sh\nexit 0\n')
    fake.chmod(0o755)
    monkeypatch.setenv('SILK_NGSPICE_BIN', str(fake))
    monkeypatch.setattr(r, 'osdi_load_problem',
                        lambda paths, ngspice=None, timeout=30:
                        seen.setdefault('ngspice', ngspice))
    doctor._osdi_load_problem(['/a.osdi'])
    assert seen['ngspice'] == str(fake)
