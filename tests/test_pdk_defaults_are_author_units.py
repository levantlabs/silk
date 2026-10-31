"""A catalog `default:` is in the PDK's OWN unit and is emitted **unscaled**.

Only caller-supplied values are unit-converted. That asymmetry is deliberate:
`minval`/`maxval` sit beside `default` and are already in the PDK's unit, and a
default is a fixed physical geometry — 0.28 µm is 0.28 µm whether the author
chose to write microns or metres.

Getting this wrong is silent. Before the fix, defaults were scaled like caller
values, and YAML scalar typing then decided per line whether that happened:

    default: 2.8e-07     # has a decimal point -> YAML float -> scaled -> 2.8e-13
    default: "1e-05"     # quoted / no point   -> YAML str   -> passed through

So one GF180MCU MOSFET default emitted 10 µm and the next 0.00000028 µm, from
the same file. `l=280f` is not a length any process has.

A second tempting test — "no default is a YAML string" — is not here either.
YAML 1.1 types `1e-05` as a STRING (no decimal point in the mantissa) and
`2.8e-07` as a float, and that split is exactly what made the original bug
selective. But now that defaults are emitted unscaled, the two differ only in
FORMATTING: a string is written through verbatim (`1e-05`), a float goes via
`_fmt` (`10u`). Same value, same physics. Asserting on it would churn emitted
decks for no safety.

A third tempting test — "the catalog default equals the PDK's own .subckt
default" — is NOT here, and deliberately. sky130's subckts are not internally
consistent about units: `sky130_fd_pr__nfet_20v0` declares `w=6e-05` (SI metres)
while `sky130_fd_pr__cap_mim_m3_1` declares `w=1` (µm). There is no invariant to
assert across that, and a test that pretends otherwise would fail for reasons
unrelated to silk.
"""
import pytest

from tests._env import needs_pdk, pdk_available


def _registered():
    from silk.pdk import available
    return sorted({n for n in available() if pdk_available(n)})


@needs_pdk
@pytest.mark.parametrize('pdk_name', _registered())
def test_an_omitted_parameter_emits_the_same_whatever_unit_the_author_writes(pdk_name):
    """The load-bearing invariant. A default is a fixed geometry, so switching
    `user_units` must not move it — only values the CALLER passes convert."""
    from silk import load_pdk

    per_mode, checked = {}, 0
    for mode in ('u', 'si', 'pdk'):
        pdk = load_pdk(pdk_name, user_units=mode)
        lines = {}
        for dev_name, cls in sorted(pdk._device_classes.items()):
            if not any(getattr(cls, '_param_defaults', {}).get(p) is not None
                       for p in cls._param_names):
                continue
            try:
                lines[dev_name] = cls('X1', **{p: '0' for p in cls._pin_names}).to_spice()
            except TypeError:
                continue          # a parameter with no default — not our case
        per_mode[mode] = lines
        checked = max(checked, len(lines))

    assert checked, f'{pdk_name}: no device has defaults — test is vacuous'
    for dev in per_mode['u']:
        variants = {m: per_mode[m].get(dev) for m in per_mode}
        assert len(set(variants.values())) == 1, (
            f'{pdk_name}.{dev}: the default moved with user_units — a default is '
            f'a fixed geometry, not something the author restates:\n  '
            + '\n  '.join(f'{m}: {v.strip()}' for m, v in variants.items()))
