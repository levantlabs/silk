"""TASK-527: the one SPICE-name codec (instance parts [k] <-> <k>)."""
import ast
from pathlib import Path

import pytest

from silk import spice_names as sn
from silk.spice_names import (designer_component, spice_component,
                              to_designer_name, to_spice_name)


def test_component_roundtrip_preserves_case():
    assert spice_component('Xx1[0]') == 'Xx1<0>'
    assert designer_component('Xx1<0>') == 'Xx1[0]'
    assert designer_component(spice_component('aB[12]c')) == 'aB[12]c'
    assert spice_component('plain') == 'plain'


def test_ngspice_path_converts_instance_positions_only():
    assert to_spice_name('xx1[0].xy[1].g[0]') == 'xx1<0>.xy<1>.g[0]'
    assert to_designer_name('xx1<0>.xy<1>.g[0]') == 'xx1[0].xy[1].g[0]'


def test_node_leaf_unchanged_and_literal_angle_net_distinct():
    assert to_designer_name('xx1<0>.g<0>') == 'xx1[0].g<0>'
    assert to_spice_name('xx1[0].g[0]') != to_spice_name('xx1[0].g<0>')
    # g<0> and g[0] are node leaves: neither converts to the other
    assert to_designer_name('g<0>') == 'g<0>'
    assert to_spice_name('g[0]') == 'g[0]'


def test_wrappers_set_role_and_nest():
    assert to_designer_name('v(xdut.xx1<0>.mid)') == 'v(xdut.xx1[0].mid)'
    assert to_spice_name('V(vvs[0])') == 'V(vvs[0])'          # V( is a node
    assert to_spice_name('I(vvs[0])') == 'I(vvs<0>)'          # I( is a branch
    assert to_spice_name('i(v.xdut.xx1[0].vint)') == 'i(v.xdut.xx1<0>.vint)'
    assert to_designer_name('i(vvs<0>)') == 'i(vvs[0])'


def test_kind_parameter_decides_bare_names_and_defaults_to_node():
    assert to_spice_name('vvs[0]') == 'vvs[0]'
    assert to_spice_name('vvs[0]', kind='node') == 'vvs[0]'
    assert to_spice_name('vvs[0]', kind='branch') == 'vvs<0>'
    assert to_designer_name('vvs<0>', kind='branch') == 'vvs[0]'
    with pytest.raises(ValueError):
        to_spice_name('a', kind='net')


def test_wrapper_overrides_kind():
    assert to_spice_name('i(vvs[0])', kind='node') == 'i(vvs<0>)'
    assert to_spice_name('v(vvs[0])', kind='branch') == 'v(vvs[0])'


def test_never_infers_role_from_leading_letter_or_case():
    assert to_spice_name('Vvs[0]') == 'Vvs[0]'
    assert to_spice_name('xvs[0]') == 'xvs[0]'


def test_at_address_keeps_param_and_converts_instances():
    assert to_designer_name('@m.xdut.xx1<0>.xmm1.m0[gm]') == '@m.xdut.xx1[0].xmm1.m0[gm]'
    assert to_spice_name('@m.xdut.xx1[0].xmm1.m0[gm]') == '@m.xdut.xx1<0>.xmm1.m0[gm]'
    assert to_spice_name('i(@m.xdut.xx1[0].xmm1.m0[id])') == 'i(@m.xdut.xx1<0>.xmm1.m0[id])'


def test_xyce_colon_upper_case_and_raw_vector_grammar():
    assert to_designer_name('V(XDUT:XX1<0>:MID)') == 'V(XDUT:XX1[0]:MID)'
    assert to_spice_name('I(XDUT:XX1[0]:VINT)') == 'I(XDUT:XX1<0>:VINT)'
    assert to_designer_name('N(XDUT:XX1<0>:XMM1:GM)') == 'N(XDUT:XX1[0]:XMM1:GM)'
    assert to_designer_name('ID(XDUT:XX1<0>:XMM1)') == 'ID(XDUT:XX1[0]:XMM1)'


def test_style_auto_default_and_explicit_style_restricts_grammar():
    assert to_designer_name('v(a<0>.b)', style='auto') == 'v(a[0].b)'
    assert to_designer_name('V(A<0>:B)', style='ngspice') == 'V(A<0>:B)'  # not ngspice grammar
    assert to_designer_name('V(A<0>:B)', style='xyce') == 'V(A[0]:B)'
    assert to_designer_name('a<0>.b', style='behavioral') == 'a[0].b'
    with pytest.raises(ValueError):
        to_designer_name('a', style='bogus')


def test_behavioral_names_keep_case():
    assert to_designer_name('Xdut.XX1<0>.Mid') == 'Xdut.XX1[0].Mid'


def test_foreign_expressions_and_unknown_forms_unchanged():
    for name in ('v(a<0>)-v(b<1>)', 'a[0] + b[1]', 'onoise_spectrum', '',
                 'sqrt(x1<0>.y)', 'v(a<0>) v(b<0>)'):
        assert to_designer_name(name) == name
        assert to_spice_name(name) == name


def test_idempotent():
    for name in ('xx1[0].xy[1].g[0]', 'i(vvs[0])', '@m.x[0].m0[gm]'):
        once = to_spice_name(name)
        assert to_spice_name(once) == once
    assert to_designer_name(to_designer_name('xx1<0>.g')) == 'xx1[0].g'


def test_module_is_stdlib_only_leaf():
    tree = ast.parse(Path(sn.__file__).read_text())
    mods = [n.module or '' if isinstance(n, ast.ImportFrom) else a.name
            for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in (n.names if isinstance(n, ast.Import) else [None])]
    assert all(m.split('.')[0] in {'re', '__future__'} for m in mods), mods


def test_emitted_names_uses_the_codec():
    from silk.circuit.emitted_names import emitted_component
    assert emitted_component('X1[0]') == spice_component('X1[0]') == 'X1<0>'


# --- code review P2-2: recognise the grammar before converting --------------

@pytest.mark.parametrize('name', [
    'I(XA<0>:VB<1>)', 'V(XA<0>:MID)', 'ID(XA<0>:XM1)', 'N(XA<0>:XM1:GM)',
])
def test_colon_forms_unchanged_under_ngspice_style(name):
    assert to_designer_name(name, style='ngspice') == name
    assert to_spice_name(name.replace('<0>', '[0]').replace('<1>', '[1]'), style='ngspice') \
        == name.replace('<0>', '[0]').replace('<1>', '[1]')


@pytest.mark.parametrize('name', [
    'v(xa<0>.mid)', 'i(v.xa<0>.vint)', '@m.xa<0>.xm1.m0[gm]', 'xa<0>.mid', 'i(@m.xa<0>.m0[id])',
])
def test_dotted_and_at_forms_unchanged_under_xyce_style(name):
    assert to_designer_name(name, style='xyce') == name
    assert to_spice_name(name.replace('<0>', '[0]'), style='xyce') == name.replace('<0>', '[0]')


def test_behavioral_style_has_no_wrappers_or_at_addresses():
    for name in ('v(xa<0>.mid)', '@m.xa<0>.m0[gm]', 'I(XA<0>:B)'):
        assert to_designer_name(name, style='behavioral') == name
    assert to_designer_name('xa<0>.mid', style='behavioral') == 'xa[0].mid'


def test_explicit_style_still_converts_its_own_grammar():
    assert to_designer_name('I(XA<0>:VB<1>)', style='xyce') == 'I(XA[0]:VB[1])'
    assert to_designer_name('ID(XA<0>:XM1)', style='xyce') == 'ID(XA[0]:XM1)'
    assert to_designer_name('i(@m.xa<0>.m0[id])', style='ngspice') == 'i(@m.xa[0].m0[id])'
    assert to_designer_name('i(v.xa<0>.vint)', style='ngspice') == 'i(v.xa[0].vint)'


@pytest.mark.parametrize('name', [
    'xx1<0>.mid:xx2<0>.a', 'xx1<0>.a:b', 'XA<0>:XB<1>.c', 'I(XA<0>:VB<1>.c)',
    'N(XA<0>.XM1.GM)', 'ID(XA<0>.XM1)', '@m.xa<0>:xm1.m0[gm]',
])
def test_mixed_separators_unchanged(name):
    assert to_designer_name(name) == name


@pytest.mark.parametrize('name', [
    'xx1<0>.mid?xx2<0>.a:b', 'xx1<0>.m1#body', 'v(xdut.xx1<0>.m1#body)',
    'i(v.xdut.xx1<0>.vint#branch)', 'xdut.xx1<0>.m1#drain', 'a<0>.b|c', 'a<0>.b;c',
])
def test_unrecognised_forms_and_real_internal_names_unchanged(name):
    assert to_designer_name(name) == name
    assert to_spice_name(name.replace('<0>', '[0]')) == name.replace('<0>', '[0]')


def test_valid_auto_paths_convert_with_either_separator():
    assert to_designer_name('xdut.xx1<0>.mid') == 'xdut.xx1[0].mid'
    assert to_designer_name('XDUT:XX1<0>:MID') == 'XDUT:XX1[0]:MID'
    assert to_spice_name('xdut.xx1[0].xy[1].g') == 'xdut.xx1<0>.xy<1>.g'


def test_guard_adds_no_designer_name_restriction():
    # negative indices, ordinary punctuation and hyphenated sweeps still convert/pass
    assert to_spice_name('x[-1].a') == 'x<-1>.a'
    assert to_designer_name('x<-1>.a') == 'x[-1].a'
    assert to_spice_name('a$b.c_d[2].e') == 'a$b.c_d<2>.e'
    assert to_designer_name('temp-sweep') == 'temp-sweep'
    assert to_spice_name('x[0]_pwl', kind='branch') == 'x<0>_pwl'


@pytest.mark.parametrize('suffix', ['é', '中', '²', 'Ω'])
def test_unicode_names_convert_like_ascii(suffix):
    assert to_designer_name(f'x{suffix}<0>.mid') == f'x{suffix}[0].mid'
    assert to_spice_name(f'x{suffix}[0].mid') == f'x{suffix}<0>.mid'
    assert to_spice_name(f'v{suffix}[0]', kind='branch') == f'v{suffix}<0>'


def test_unicode_guard_still_refuses_non_name_characters():
    assert to_designer_name('x\u00a0<0>.mid') == 'x\u00a0<0>.mid'
    assert to_designer_name('x<0>.a b') == 'x<0>.a b'


@pytest.mark.parametrize('name', [
    'xx1<0>.mid&xx2<0>.mid', 'xx1<0>.mid%2', 'xx1<0>.mid^2', '!xx1<0>.mid',
    'xx1<0>.mid`', 'xx1<0>.mid@x', 'xx1<0>.mid~', 'xx1<0>.m"id', "xx1<0>.m'id",
])
def test_ascii_marks_outside_the_original_set_stay_unrecognised(name):
    assert to_designer_name(name) == name
    spice = name.replace('<0>', '[0]')
    assert to_spice_name(spice) == spice


@pytest.mark.parametrize('code', [0x80, 0x85, 0x8d, 0x9f])
def test_c1_controls_are_all_excluded(code):
    name = 'xx1<0>.mid' + chr(code)
    assert to_designer_name(name) == name
    assert to_designer_name('xx1<0>.m' + chr(code) + 'id') == 'xx1<0>.m' + chr(code) + 'id'


def test_original_ascii_set_still_converts():
    assert to_designer_name('a$b-c_d<0>.e9') == 'a$b-c_d[0].e9'


_ASCII_SET = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_$-[]<>')
_LISTED = {'\u00a0', '\u1680', *map(chr, range(0x2000, 0x200b)), '\u2028', '\u2029',
           '\u202f', '\u205f', '\u3000', '\ufeff'}


def _original_ascii_guard(name):
    """The 0ea26491 recognition of a plain dotted path: every segment is a
    non-empty run of the ASCII set, and there is no ':' mixed in."""
    return ':' not in name and all(seg and set(seg) <= _ASCII_SET for seg in name.split('.'))


@pytest.mark.parametrize('code', range(0x80))
def test_ascii_boundary_equals_the_original_guard(code):
    c = chr(code)
    for name in (f'xx1<0>{c}.mid', f'{c}xx1<0>.mid', f'xx1<0>.mid{c}'):
        if name.startswith('@'):
            continue                      # an '@' prefix is the address grammar, not a segment
        assert (to_designer_name(name) != name) == _original_ascii_guard(name), repr(name)


@pytest.mark.parametrize('c', [chr(x) for x in range(0x80, 0x100)] + sorted(_LISTED))
def test_non_ascii_boundary_rule(c):
    ok = not (0x80 <= ord(c) <= 0x9f or c in _LISTED)
    for name in (f'xx1<0>{c}.mid', f'{c}xx1<0>.mid', f'xx1<0>.mid{c}'):
        got = to_designer_name(name)
        assert (got == name.replace('<0>', '[0]')) if ok else (got == name), repr(name)


@pytest.mark.parametrize('ending', ['\n', '\r', '\r\n', '\u2003'])
@pytest.mark.parametrize('name', ['xx1<0>.mid', 'v(xx1<0>.mid)', 'i(vvs<0>)', '@m.xx1<0>.m0[gm]',
                                  'i(@m.xx1<0>.m0[id])', 'I(XA<0>:VB<1>)', 'XDUT:XX1<0>:MID'])
def test_trailing_line_ends_and_spaces_leave_the_name_byte_for_byte(name, ending):
    spelled = name + ending
    assert to_designer_name(spelled) == spelled
    assert to_spice_name(spelled.replace('<', '[').replace('>', ']')) == \
        spelled.replace('<', '[').replace('>', ']')


def test_positive_controls_still_convert():
    for name, want in [('xx1<-1>.mid', 'xx1[-1].mid'), ('a$b<0>.c', 'a$b[0].c'),
                       ('a_b<0>.c', 'a_b[0].c'), ('xé<0>.mid', 'xé[0].mid'),
                       ('x中<0>.mid', 'x中[0].mid'), ('x²<0>.mid', 'x²[0].mid'),
                       ('xe\u0301<0>.mid', 'xe\u0301[0].mid')]:
        assert to_designer_name(name) == want
        assert to_spice_name(want) == name


def test_allow_glob_widens_recognition_only_for_selectors():
    from silk.spice_names import to_spice_name
    assert to_spice_name('xx1[*].g[0]', allow_glob=True) == 'xx1<*>.g[0]'
    assert to_spice_name('xx1[*].g[0]') == 'xx1[*].g[0]'            # default off: unchanged
    assert to_spice_name('d[*]', allow_glob=True) == 'd[*]'          # literal leaf
    assert to_spice_name('v(xx1[?].g[*])', allow_glob=True) == 'v(xx1<?>.g[*])'
    assert to_spice_name('vbias[*]', kind='branch', allow_glob=True) == 'vbias<*>'
    # every other 527 refusal holds with the flag on
    for s in ('xx1[0]&foo', 'xx1[0]\nfoo', 'xx1[0].g[0]\n', 'xx1[0] .g'):
        assert to_spice_name(s, allow_glob=True) == to_spice_name(s) == s
