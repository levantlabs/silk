"""TASK-571: silk's control block must end the batch session itself.

``exit`` exists in ngspice only as ``alias exit quit`` in the startup file
(spinit). Without that file the command is unknown, the batch session carries
on and, with ``-r``, ngspice simulates a second time into the same rawfile.
These decks use ideal devices only and an empty ``SPICE_SCRIPTS`` so the
startup file cannot be found.
"""
import os
import subprocess

import numpy as np
import pytest

from silk.circuit.spice_netlister import PySpiceNetlister
from silk.signals.rawfile import read_rawfile
from silk.simulation.runner import ngspice_executable
from tests._env import needs_ngspice

pytestmark = needs_ngspice

RANDOM = 'V1 a 0 dc 0 trrandom(1 1e-7 0 0.1 0.5)\nR1 a 0 1\n'
TRAN = '.tran 1e-8 1e-6\n'


def _netlister():
    return PySpiceNetlister.__new__(PySpiceNetlister)


def _run(tmp_path, deck, rawfile, stdin=None):
    """Run ngspice as silk does (-b -r -o) with no startup file reachable."""
    scripts = tmp_path / 'empty_scripts'
    scripts.mkdir(exist_ok=True)
    cir, log = tmp_path / 'deck.cir', tmp_path / 'out.log'
    cir.write_text(deck)
    env = dict(os.environ, SPICE_SCRIPTS=str(scripts))
    result = subprocess.run(
        [ngspice_executable(), '-n', '-b', '-r', str(rawfile), '-o', str(log), str(cir)],
        env=env, input=stdin, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return log.read_text()


def _runs(log_text):
    return log_text.count('Initial Transient Solution')


def _first(path, plot=0, name='v(a)'):
    raw = read_rawfile(str(path))
    vector = next(v for v in raw.get_plots()[plot].vectors if v.name == name)
    return float(np.asarray(vector.data)[0])


def _deck(tmp_path, control_dir, analyses=TRAN, **kwargs):
    control = _netlister().generate_control_block(str(control_dir), **kwargs)
    return f'* quit terminator\n{RANDOM}{analyses}{control}'


def test_control_block_runs_once_and_leaves_no_fallback_file(tmp_path):
    written = tmp_path / 'ctl'
    written.mkdir()
    fallback = tmp_path / 'fallback.raw'
    log = _run(tmp_path, _deck(tmp_path, written), fallback)
    assert _runs(log) == 1
    assert not fallback.exists()
    assert _first(written / 'rawfile.raw') == pytest.approx(0.5)


def test_production_same_path_form_keeps_the_authored_start(tmp_path):
    rawfile = tmp_path / 'rawfile.raw'
    log = _run(tmp_path, _deck(tmp_path, tmp_path), rawfile)
    assert _runs(log) == 1
    assert _first(rawfile) == pytest.approx(0.5)


def test_multi_plot_write_survives(tmp_path):
    rawfile = tmp_path / 'rawfile.raw'
    deck = _deck(tmp_path, tmp_path, analyses=TRAN + '.op\n', plots=['tran1', 'op1'])
    log = _run(tmp_path, deck, rawfile)
    assert _runs(log) == 1
    names = [p.plotname for p in read_rawfile(str(rawfile)).get_plots()]
    assert names == ['Transient Analysis', 'Operating Point']


def test_askquit_with_an_unsaved_plot_ends_once(tmp_path):
    rawfile = tmp_path / 'rawfile.raw'
    deck = _deck(tmp_path, tmp_path, analyses=TRAN + '.op\n', plots=['tran1'])
    deck = deck.replace('.control\n', '.control\nset askquit\n', 1)
    log = _run(tmp_path, deck, rawfile, stdin='n\n')
    assert _runs(log) == 1


def test_batch_write_deck_without_control_block_still_writes_once(tmp_path):
    rawfile = tmp_path / 'rawfile.raw'
    deck = f'* no control\n{RANDOM}{TRAN}.save all\n.end\n'
    log = _run(tmp_path, deck, rawfile)
    assert _runs(log) == 1
    assert rawfile.exists()
