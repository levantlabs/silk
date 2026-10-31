"""silk has ONE SPICE-number parser: ``silk.utils._utils._parse_si``.

Three independent suffix tables previously agreed on ``M``/``Meg``
and disagreed on everything at the edges: the foreign-netlist importer
returned ``None`` for ``2.56us`` (a trailing unit letter), the others parsed
it; nobody accepted ngspice's ``a`` (atto) or ``mil``. This file is the
contract those tables were consolidated under (architecture.md invariant
I16), in five parts:

1. the parse contract itself — suffix table, trailing letters, the rejected
   Unicode micro sign, and the one failure mode (``ValueError``);
2. every entry point that reads a SPICE number computes the same number;
3. a mechanical guard: no Python file outside ``silk/utils/_utils.py`` binds
   the mega token to ``1e6`` (that is how a suffix table gives itself away);
4. the display side: axis labels and explorer readouts write SI for humans
   (``M`` = mega), which the SPICE parser reads as milli. The explorer's own
   ``parseSI`` must therefore refuse an uppercase ``M``/``P`` so a copied
   readout never silently comes back 1e9 too small. Verified by running the
   explorer's ``Fmt`` block under node when node is on PATH.
5. the public entry point: ``silk.utils.parse_spice_value`` delegates to this
   same ``_parse_si`` rather than adding
   a second table, and round-trips with ``silk.utils.format_spice_value``.
"""
import json
import re
import shutil
import subprocess
import warnings
from pathlib import Path

import pytest

from tests._silk_sources import silk_py_files

from silk.utils._utils import _fmt, _parse_si
from silk import utils as silk_utils

_REPO = Path(__file__).resolve().parents[1]
_SILK = _REPO / 'silk'
_CANONICAL = _SILK / 'utils' / '_utils.py'
_VIEW_JS = _SILK / 'signals' / 'view' / 'assets' / 'explorer.mjs'


# ---------------------------------------------------------------------------
# 1. The contract
# ---------------------------------------------------------------------------

class TestParseContract:

    @pytest.mark.parametrize('text,expected', [
        # ngspice's table, T G Meg K mil m u n p f a, any case
        ('1T', 1e12), ('1G', 1e9), ('1Meg', 1e6), ('1MEG', 1e6), ('1meg', 1e6),
        ('1k', 1e3), ('1K', 1e3), ('1mil', 25.4e-6), ('1MIL', 25.4e-6),
        ('1m', 1e-3), ('1M', 1e-3), ('1u', 1e-6), ('1n', 1e-9), ('1p', 1e-12),
        ('1f', 1e-15), ('1a', 1e-18), ('1A', 1e-18),
        # plain numbers
        ('0', 0.0), ('2.5', 2.5), ('-3', -3.0), ('1.5e-6', 1.5e-6), ('.5n', 5e-10),
        # letters after the suffix are ignored (ngspice: '10Volts' == '10')
        ('2.56us', 2.56e-6), ('1ns', 1e-9), ('10kHz', 1e4), ('150nm', 150e-9),
        ('1MA', 1e-3), ('1MegHz', 1e6),
        # a trailing letter that is not a suffix is ignored, not rescaled
        ('1s', 1.0), ('5V', 5.0), ('10Hz', 10.0),
        # whitespace
        ('  4  ', 4.0), ('1 n', 1e-9),
    ])
    def test_string_parses_to_what_the_simulator_computes(self, text, expected):
        assert _parse_si(text) == pytest.approx(expected)

    @pytest.mark.parametrize('text,mantissa,scale', [
        ('10u', 10.0, 1e-6), ('7n', 7.0, 1e-9), ('1mil', 1.0, 25.4e-6),
        ('3.3k', 3.3, 1e3), ('2.2Meg', 2.2, 1e6),
    ])
    def test_value_is_the_simulator_product_not_the_shortest_repr(
            self, text, mantissa, scale):
        """TASK-296: the guarantee is agreement with the simulator, which
        forms mantissa * scale in double. Exact equality (no approx), so a
        switch to shortest-repr construction fails here."""
        assert _parse_si(text) == mantissa * scale

    def test_ulp_artifacts_are_kept_and_display_still_round_trips(self):
        """'10u' and '7n' are one ULP off their shortest repr by design
        (float('10e-6') would give 1e-05 and disagree with the simulator);
        formatting the parsed value still returns what was typed."""
        assert _parse_si('10u') == 9.999999999999999e-06
        assert _parse_si('10u') != float('10e-6')
        assert _parse_si('7n') == 7.000000000000001e-09
        assert _parse_si('7n') != float('7e-9')
        assert _fmt(_parse_si('10u')) == '10u'
        assert _fmt(_parse_si('7n')) == '7n'

    def test_longest_suffix_wins(self):
        """'MEG' and 'MIL' must beat 'M', or both read as milli."""
        assert _parse_si('1Meg') == pytest.approx(1e6)
        assert _parse_si('1mil') == pytest.approx(25.4e-6)
        assert _parse_si('1m') == pytest.approx(1e-3)

    def test_unicode_micro_sign_is_rejected(self):
        """Strings are emitted into the deck verbatim and the simulator
        would not read 'µ'; accepting it here would split twin from netlist
        by 1e6. Spell micro 'u'."""
        with pytest.raises(ValueError):
            _parse_si('1µ')
        with pytest.raises(ValueError):
            _parse_si('1μs')       # Greek mu, the other code point

    @pytest.mark.parametrize('bad', [
        '', '   ', 'abc', 'n', 'Meg', None, [1], object(), '{2*wn}', "'wn*2'",
        '1..5', '--1',
    ])
    def test_the_one_failure_mode_is_value_error(self, bad):
        with pytest.raises(ValueError):
            _parse_si(bad)

    @pytest.mark.parametrize('value', [
        1e-15, 100e-12, 55e-9, 2.56e-6, 1e-3, 0.15, 0.5, 1.8, 4.0, 1e3,
        10e3, 1e6, 2.5e9, 1e12, -3e-3, -1e6,
    ])
    def test_inverse_of_fmt(self, value):
        assert _parse_si(_fmt(value)) == pytest.approx(value, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Every entry point computes the same number
# ---------------------------------------------------------------------------

_CORPUS = [
    ('2.56us', 2.56e-6), ('150n', 150e-9), ('1Meg', 1e6), ('150M', 0.15),
    ('10k', 1e4), ('1.5e-6', 1.5e-6), ('4', 4.0), ('1a', 1e-18),
    ('1mil', 25.4e-6), ('100p', 1e-10),
]


class TestEntryPointsAgree:


    def test_circuit_import_has_no_table_of_its_own(self):
        from silk.db import circuit_import
        assert circuit_import._parse_si is _parse_si
        assert not hasattr(circuit_import, '_SPICE_SUFFIXES')

    @pytest.mark.parametrize('text,expected', _CORPUS)
    def test_importer_agrees_and_does_not_warn_on_valid_input(self, text, expected):
        from silk.db.circuit_import import _parse_spice_value
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert _parse_spice_value(text) == pytest.approx(expected)

    @pytest.mark.parametrize('text,expected', _CORPUS)
    def test_analysis_fields_agree(self, text, expected):
        from silk.simulation.analysis import _finite_number
        assert _finite_number('Transient', 'tstep', text) == pytest.approx(expected)

    def test_importer_records_unknown_as_none_but_says_so(self):
        """The importer is the ONE lenient consumer: foreign netlists carry
        expressions silk cannot evaluate. None is allowed, silence is not."""
        from silk.db.circuit_import import _parse_spice_value
        with pytest.warns(UserWarning, match=r"\{2\*wn\}"):
            assert _parse_spice_value('{2*wn}') is None

    def test_importer_end_to_end_trailing_unit_letter(self, tmp_path):
        """Before TASK-26 this MOSFET imported with width=None, length=None."""
        from silk.db.circuit_db import Database
        from silk.db.circuit_import import CircuitImporter
        netlist = tmp_path / 'foreign.cir'
        netlist.write_text(
            "* foreign netlist with unit letters\n"
            ".subckt amp vin vout vdd gnd\n"
            "M1 vout vin gnd gnd nmos_model w=2.56us l=150nm\n"
            "R1 vout vdd 10kOhm\n"
            "V1 vdd gnd DC 1.8V\n"
            ".ends\n"
        )
        db = Database(str(tmp_path / 'x.db'))
        try:
            result = CircuitImporter(db).parse_spice_netlist(str(netlist))
        finally:
            db.close()
        m1 = result['transistors'][0]
        assert m1['width'] == pytest.approx(2.56e-6)
        assert m1['length'] == pytest.approx(150e-9)
        assert result['passives'][0]['value'] == pytest.approx(1e4)
        assert result['sources'][0]['dc'] == pytest.approx(1.8)


# ---------------------------------------------------------------------------
# 3. Mechanical guard: no second suffix table
# ---------------------------------------------------------------------------

# A SPICE suffix table gives itself away by binding the mega token to 1e6.
_MEGA_BINDING = re.compile(r"""['"]MEG['"]\s*[,:]\s*1e\+?0?6\b""", re.IGNORECASE)


def _silk_py_files():
    files = [p for p in silk_py_files() if p != _CANONICAL]
    assert files, f'no python files found under {_SILK} — glob is broken'
    return files


@pytest.mark.parametrize('path', _silk_py_files(),
                         ids=lambda p: str(p.relative_to(_REPO)))
def test_no_spice_suffix_table_outside_silk_utils(path):
    if _MEGA_BINDING.search(path.read_text()):
        pytest.fail(
            f'{path.relative_to(_REPO)} binds a mega suffix to 1e6: that is a '
            f'SPICE-number parse table. Import silk.utils._utils._parse_si '
            f'instead (architecture.md I16).')


def test_canonical_table_is_where_the_docs_say():
    text = _CANONICAL.read_text()
    assert _MEGA_BINDING.search(text)
    assert 'MIL' in text and "('A', 1e-18)" in text


# ---------------------------------------------------------------------------
# 4. Display side: readouts are SI for humans and never round-trip wrong
# ---------------------------------------------------------------------------

def test_axis_label_prefix_table_is_marked_display_only():
    """modules_results._PREFIXES writes 'M' for mega (SI, for plot labels).
    That is the opposite of what the parser reads; the comment beside the
    table is the documentation AC #3 asks for."""
    src = (_SILK / 'circuit' / 'modules_results.py').read_text()
    table = src[src.index('_PREFIXES = ['):src.index('_UNIT_LABELS')]
    assert "(1e6,  'M')" in table
    before = src[:src.index('_PREFIXES = [')]
    assert 'Display-only' in before[-600:]
    assert '_parse_si' in before[-600:]


def _explorer_fmt_block():
    src = _VIEW_JS.read_text(encoding='utf-8')
    start = src.index('var Fmt = (function () {')
    end = src.index('})();', start) + len('})();')
    return src[start:end]


def test_explorer_parser_guard_is_present_in_source():
    block = _explorer_fmt_block()
    assert 'DISPLAY_ONLY_UPPER' in block
    assert "'a': 1e-18" in block
    # No uppercase key: the parser is case-insensitive by lowercasing, and
    # the M/P refusal happens before that.
    keys = re.findall(r"'([^']+)':\s*1e", block[block.index('var SI_SUFFIX'):])
    assert keys and all(k == k.lower() for k in keys), keys


_NODE_HARNESS = r'''
function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }
function isFiniteNum(v) { return v === v && v !== Infinity && v !== -Infinity; }
%(fmt)s
var out = { parse: {}, roundtrip: [] };
var inputs = %(inputs)s;
for (var i = 0; i < inputs.length; i++) out.parse[inputs[i]] = Fmt.parseSI(inputs[i]);
var values = %(values)s;
for (var j = 0; j < values.length; j++) {
  var shown = Fmt.num(values[j], '');
  out.roundtrip.push([values[j], shown, Fmt.parseSI(shown)]);
}
process.stdout.write(JSON.stringify(out));
'''

_EXPLORER_INPUTS = {
    '2.5 MHz': None,        # readout form: uppercase M refused, not milli
    '2.5M': None,
    '100P': None,           # readout peta, SPICE pico: refused
    '2.5 mHz': None,        # explorer is strict: an unknown trailing letter is refused
    '2.5 m': 2.5e-3,
    '2.5meg': 2.5e6,
    '2.5MEG': 2.5e6,
    '2.5Meg': 2.5e6,
    '1µ': 1e-6,
    '1u': 1e-6,
    '1µs': None,            # strict again: 'µs' is not a suffix the explorer knows
    '1a': 1e-18,
    '100p': 1e-10,
    '5n': 5e-9,
    '1k': 1e3,
    '1e-9': 1e-9,
    '1x': None,
    '': None,
}

_EXPLORER_VALUES = [1e-18, 1e-12, 5e-9, 2.5e-6, 1e-3, 0.5, 1.0, 1e3, 2.5e6, 1e9, 1e12, 1e15]


@pytest.mark.skipif(shutil.which('node') is None, reason='node not on PATH')
def test_explorer_readout_never_round_trips_to_a_wrong_number(tmp_path):
    """Run the explorer's own Fmt block: parseSI honours the SPICE table,
    refuses the two uppercase letters the readout uses differently, and a
    readout (number + prefix, as the cursor table shows it) typed back in is
    either the same number or refused — never a silently rescaled one.

    Unlike ``_parse_si``, the explorer refuses unknown trailing letters
    outright ('2.5 mHz' is a red flash, not 2.5e-3): it parses typed input,
    where a typo should not become a cursor position."""
    script = tmp_path / 'fmt_check.js'
    script.write_text(_NODE_HARNESS % {
        'fmt': _explorer_fmt_block(),
        'inputs': json.dumps(list(_EXPLORER_INPUTS)),
        'values': json.dumps(_EXPLORER_VALUES),
    }, encoding='utf-8')
    proc = subprocess.run(['node', str(script)], capture_output=True, text=True,
                          check=True)
    out = json.loads(proc.stdout)
    for text, expected in _EXPLORER_INPUTS.items():
        got = out['parse'][text]
        if expected is None:
            assert got is None, (text, got)
        else:
            assert got == pytest.approx(expected), (text, got)
    for value, shown, back in out['roundtrip']:
        assert back is None or back == pytest.approx(value, rel=1e-3), (value, shown, back)
    # ... and the guard bites where it must: mega and peta readouts are refused,
    # everything the two conventions agree on comes back as itself.
    refused = {shown for value, shown, back in out['roundtrip'] if back is None}
    assert refused == {'2.500 M', '1.000 P'}, refused


# ---------------------------------------------------------------------------
# 5. Public entry point (TASK-251): parse_spice_value delegates, round-trips
# ---------------------------------------------------------------------------

class TestPublicEntryPoint:

    def test_parse_spice_value_is_public(self):
        assert 'parse_spice_value' in silk_utils.__all__
        assert 'format_spice_value' in silk_utils.__all__

    def test_parse_spice_value_delegates_to_the_one_table(self):
        """No second table: the public name is the canonical function, not
        a wrapper that copies its logic."""
        assert silk_utils.parse_spice_value is _parse_si

    @pytest.mark.parametrize('formatted,value', [
        ('1T', 1e12), ('1G', 1e9), ('1Meg', 1e6), ('1k', 1e3),
        ('1mil', 25.4e-6), ('1m', 1e-3), ('1u', 1e-6), ('1n', 1e-9),
        ('1p', 1e-12), ('1f', 1e-15), ('1a', 1e-18),
    ])
    def test_scale_factors_easy_to_get_wrong(self, formatted, value):
        assert silk_utils.parse_spice_value(formatted) == pytest.approx(value)

    def test_meg_and_mil_beat_m(self):
        """The ordering bug a naive reimplementation would hit: 'Meg' and
        'mil' must be matched before the single-letter 'M'/'m', or 1Meg
        reads as 1 milli and 1mil reads as 1 milli."""
        assert silk_utils.parse_spice_value('1Meg') == pytest.approx(1e6)
        assert silk_utils.parse_spice_value('1mil') == pytest.approx(25.4e-6)
        assert silk_utils.parse_spice_value('1m') == pytest.approx(1e-3)
        assert silk_utils.parse_spice_value('1M') == pytest.approx(1e-3)

    @pytest.mark.parametrize('value', [
        1e12, 2.5e9, 1e6, 10e3, 1.8, 0.5, 1e-3, 2.56e-6, 55e-9, 100e-12,
        1e-15, -3e-3, -1e6,
    ])
    def test_round_trips_with_format_spice_value(self, value):
        assert silk_utils.parse_spice_value(
            silk_utils.format_spice_value(value)
        ) == pytest.approx(value, rel=1e-12)
