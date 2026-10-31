"""Independent deck/log checks for the operating points claimed by examples."""
from pathlib import Path
import re
from urllib.parse import unquote

import pytest


def assert_simulator_temperature(deck_path, temperature):
    deck_path = Path(deck_path)
    deck = deck_path.read_text()
    options = re.findall(r'^\s*\.options\s+(.+)$', deck, re.I | re.M)
    temps = re.findall(r'\btemp\s*=\s*([^\s]+)', ' '.join(options), re.I)
    assert [float(t) for t in temps] == [temperature]
    assert not re.search(r'^\s*\.temp\b', deck, re.I | re.M)
    assert not re.search(r'\btnom\s*=', ' '.join(options), re.I)
    log = deck_path.with_name('output.log').read_text()
    points = re.findall(r'Doing analysis at TEMP = (\S+) and TNOM = (\S+)', log)
    assert points, log[-2000:]
    assert all(float(temp) == pytest.approx(temperature) for temp, _ in points)
    return float(points[-1][0]), float(points[-1][1])


def assert_walkthrough_temperatures(summary):
    """Check all four executed decks, including the post-layout AC run."""
    from silk.examples import design_loop_walkthrough as dl

    expected = {
        dl.T_STEP_LIGHT: summary['intent']['nominal_temperature_c'],
        dl.T_STEP_HEAVY: summary['intent']['hot_temperature_c'],
        dl.T_AC_PRE: summary['intent']['nominal_temperature_c'],
        dl.T_AC_POST: summary['intent']['nominal_temperature_c'],
    }
    for name, temperature in expected.items():
        decks = [p for p in Path(summary['out_dir']).rglob('*.cir')
                 if unquote(p.stem) == f'test_sampling_buffer~{name}']
        assert len(decks) == 1, decks
        assert_simulator_temperature(decks[0], temperature)
