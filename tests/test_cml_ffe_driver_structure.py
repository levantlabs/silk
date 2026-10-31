"""build_cml_ffe_driver is built from tap and load modules.

Guards the split: the flattened devices match the pre-split recipe (sign
crossing, per-tap multiplier, forwarded ``rl`` / ``pd_rl``), and the composed
driver is clean under default lint (the taps sum current on OUTP/OUTN; that is
not a driver conflict)."""
import pytest
import numpy as np

from silk.examples import cml_cells as cml
from silk.lint import RuleRegistry


@pytest.fixture(autouse=True)
def _out_dir(tmp_path, monkeypatch):
    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))


def _parse(text):
    subs, cur = {}, None
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith('*'):
            continue
        t = ln.split()
        if t[0] == '.subckt':
            cur = t[1]
            subs[cur] = (t[2:], [])
        elif t[0] == '.ends':
            cur = None
        elif cur:
            subs[cur][1].append(t)
    return subs


def _flatten(subs, name, pre, netmap, out):
    def net(x):
        return netmap.get(x, x if x == '0' else pre + x)
    for t in subs[name][1]:
        if t[-1] in subs:
            m = dict(zip(subs[t[-1]][0], (net(a) for a in t[1:-1])))
            _flatten(subs, t[-1], pre + t[0] + '.', m, out)
        elif t[0][0] == 'R':
            out.append(('R', t[3], tuple(net(x) for x in t[1:3])))
        else:   # XM<name> d g s b <model> k=v ...
            params = tuple(sorted(tuple(kv.split('=')) for kv in t[6:]))
            out.append(('M', t[5], params, tuple(net(x) for x in t[1:5])))


def _devices(driver):
    subs = _parse(driver.generate_netlist())
    out = []
    _flatten(subs, 'cml_ffe_driver', '', {p: p for p in subs['cml_ffe_driver'][0]}, out)
    return out


TAPS = (-0.5, 1.0, 0.25, -0.1)
MODEL = 'sky130_fd_pr__nfet_01v8'


def _nm(model, m, w, l):
    return ('M', model, tuple(sorted([('l', l), ('m', str(m)), ('w', w)])))


def _expected(taps, main_mul, rl, pd_rl):
    """The pre-split recipe, written out device by device with the old net names
    (``gp_<tap>``, ``tail_<tap>``, ``pd_<tap>``) mapped to the new hierarchy."""
    exp = [('R', str(rl), ('VDD', 'OUTP')), ('R', str(rl), ('VDD', 'OUTN'))]
    for nm, c in zip(cml._FFE_NAMES, taps):
        mm = max(1, round(abs(c) * main_mul))
        da, db = ('OUTN', 'OUTP') if c > 0 else ('OUTP', 'OUTN')
        t, pd = f'xtap_{nm}.', f'xtap_{nm}.xpd.'
        for kind, mod_, m_, w_, l_, nets in (
                ('M', MODEL, mm, '4', '0.15', (da, t + 'gp', t + 'tail', 'VSS')),
                ('M', MODEL, mm, '4', '0.15', (db, t + 'gn', t + 'tail', 'VSS')),
                ('M', MODEL, mm, '5', '0.5', (t + 'tail', 'IBIAS', 'VSS', 'VSS')),
                # predriver (imul=16): tail w=5*16, pair w=4*round(sqrt(16))
                ('M', MODEL, 1, '80', '0.5', (pd + 'tail', 'IBIAS', 'VSS', 'VSS')),
                ('M', MODEL, 1, '16', '0.15', (t + 'gn', f'{nm.upper()}P', pd + 'tail', 'VSS')),
                ('M', MODEL, 1, '16', '0.15', (t + 'gp', f'{nm.upper()}N', pd + 'tail', 'VSS'))):
            exp.append((kind, mod_, _nm(mod_, m_, w_, l_)[2], nets))
        exp.append(('R', str(pd_rl), ('VDD', t + 'gp')))
        exp.append(('R', str(pd_rl), ('VDD', t + 'gn')))
    return exp


def test_flattened_devices_match_the_pre_split_recipe():
    drv = cml.build_cml_ffe_driver(taps=TAPS, main_mul=20, rl=33.0, pd_rl=150.0)
    got = _devices(drv)
    want = _expected(TAPS, 20, 33, 150)
    assert len(got) == len(want) == 4 * 8 + 2
    assert sorted(got, key=repr) == sorted(want, key=repr)


def test_default_lint_has_no_driver_conflict():
    result = RuleRegistry.default().run(cml.build_cml_ffe_driver())
    assert [v for v in result.violations if v.rule_id in ('INT-001', 'INT-004')] == []


@pytest.mark.parametrize('taps, main_mul, rl, expected', [
    ((-0.385, 1, -0.610, 0.106), 40, 50, (1.41, 1.36)),
    ((-0.5, 1, 0.25, -0.1), 20, 100, (1.56, 1.30)),
])
@pytest.mark.parametrize('polarity', [1, -1, 0])
def test_behavioral_taps_sum_into_load(taps, main_mul, rl, expected, polarity):
    """Catch missing current models, reversed KCL signs and lost weight settings."""
    from silk.sources import VoltageSource
    from silk.simulation.behavioral import BehavioralEngine

    top = cml.Module('sky130', cml.example_output_dir(), 'sum.cir', pdk_details=cml.pdk)
    nets = {'VDD': 'vdd', 'VSS': '0', 'IBIAS': 'ibias', 'OUTP': 'outp', 'OUTN': 'outn'}
    for name in cml._FFE_NAMES:
        for suffix, voltage in [('P', 1.6 + polarity * 0.2), ('N', 1.6 - polarity * 0.2)]:
            pin = name.upper() + suffix
            nets[pin] = pin.lower()
            top.add_source(VoltageSource(pin, positive=nets[pin], negative='0', dc=voltage))
    top.add_source(VoltageSource('supply', positive='vdd', negative='0', dc=1.8))
    top.add_module(cml.build_cml_ffe_driver(taps=taps, main_mul=main_mul, rl=rl),
                   'drv', nets=nets)
    with pytest.warns(UserWarning, match='INOUT'):
        result, _ = BehavioralEngine().run(top, 1, 2)
    if polarity == -1:
        expected = expected[::-1]
    elif polarity == 0:
        expected = (sum(expected) / 2,) * 2
    np.testing.assert_allclose(result['outp'], expected[0])
    np.testing.assert_allclose(result['outn'], expected[1])


def test_behavioral_driver_preserves_independent_tap_delays():
    """One low bit isolates each delayed tap and its crossed-output sign."""
    bits = [1] * 20 + [0] + [1] * 20
    with pytest.warns(UserWarning, match='INOUT'):
        t, launch = cml._run_ffe_driver(bits, simulator='behavioral')
    # All-high launch is 50 mV. Flipping each tap changes it by twice
    # its signed contribution: +300, -800, +480 and -80 mV.
    sample_times = cml._FFE_T0 + np.array([18.4, 19.4, 20.4, 21.4, 22.4, 23.4]) * cml._FFE_TBIT
    np.testing.assert_allclose(np.interp(sample_times, t, launch),
                               [0.05, 0.35, -0.75, 0.53, -0.03, 0.05], atol=1e-12)
