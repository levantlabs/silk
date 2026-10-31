"""Tests for silk.pdk.validate — PDK device vetting."""
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from silk.simulation.runner import ngspice_executable
from tests._env import needs_pdk, needs_ngspice

def test_scan_subckt_and_model(tmp_path):
    from silk.pdk.validate import scan_lib_definitions
    (tmp_path / 'a.spice').write_text(
        ".subckt  sky130_fd_pr__res_high_po r0 r1 b\n.ends\n"
        ".model sky130_fd_pr__diode_pw2nd_05v5 d\n"
        ".subckt cap_var c0 c1 b w=5 l=0.5 vm=1\n.ends\n")
    defs = scan_lib_definitions(str(tmp_path))
    assert defs['sky130_fd_pr__res_high_po'] == {('subckt', 3)}
    assert defs['sky130_fd_pr__diode_pw2nd_05v5'] == {('model', 'd')}
    assert defs['cap_var'] == {('subckt', 3)}          # params after terminals ignored

def test_scan_stops_terminals_at_params_keyword(tmp_path):
    """`.subckt X a b params: w=1` has TWO terminals. The `params:` keyword
    ends the terminal list: it carries no '=' itself, so counting every
    token without one made the arity one too high, and the spaced-out
    `params: w = 1` form would have inflated it further."""
    from silk.pdk.validate import scan_lib_definitions
    (tmp_path / 'p.spice').write_text(
        ".subckt with_params a b params: w=1\n.ends\n"
        ".subckt spaced_params a b PARAMS: w = 1 l = 2\n.ends\n"
        ".subckt glued_params a b params:w=1\n.ends\n")
    defs = scan_lib_definitions(str(tmp_path))
    assert defs['with_params'] == {('subckt', 2)}
    assert defs['spaced_params'] == {('subckt', 2)}
    assert defs['glued_params'] == {('subckt', 2)}


def test_scan_word_boundary_not_variant(tmp_path):
    from silk.pdk.validate import scan_lib_definitions
    (tmp_path / 'b.spice').write_text(".subckt sky130_fd_pr__res_xhigh_po_2p85 r0 r1 b\n.ends\n")
    defs = scan_lib_definitions(str(tmp_path))
    assert 'sky130_fd_pr__res_xhigh_po_2p85' in defs
    assert 'sky130_fd_pr__res_xhigh_po' not in defs  # exact names only

def test_closure_collects_all_arities_no_last_wins(tmp_path):
    """A name defined multiple times WITHIN ONE PDK's include closure
    (corner/variant/parasitic files) with different arities must retain ALL
    definitions, not just the last one scanned — this is what lets
    validate_pdk avoid false arity_mismatch positives (Bug 2).

    The permissiveness is deliberate but scoped: it holds across the files
    one lib actually pulls in, NOT across whatever else happens to sit under
    $PDK_ROOT — see test_unrelated_tree_cannot_launder_arity_mismatch."""
    from silk.pdk.validate import resolve_lib_closure, scan_lib_files
    (tmp_path / 'a.spice').write_text(
        ".subckt sky130_fd_pr__pnp_05v5_W3p40L3p40 c b e s\n.ends\n")
    (tmp_path / 'z_parasitic.spice').write_text(
        ".subckt sky130_fd_pr__pnp_05v5_W3p40L3p40 c b e\n.ends\n")
    (tmp_path / 'top.lib.spice').write_text(
        '.lib tt\n.include "a.spice"\n.include "z_parasitic.spice"\n.endl tt\n')
    defs = scan_lib_files(resolve_lib_closure(str(tmp_path / 'top.lib.spice')))
    assert ('subckt', 4) in defs['sky130_fd_pr__pnp_05v5_W3p40L3p40']
    assert ('subckt', 3) in defs['sky130_fd_pr__pnp_05v5_W3p40L3p40']


# ---------------------------------------------------------------------------
# Include-closure resolution (scoping the check to the PDK actually netlisted)
# ---------------------------------------------------------------------------

def test_closure_follows_nested_relative_includes(tmp_path):
    """`.include` paths resolve against the INCLUDING file's directory (as
    SPICE resolves them), recursively, so a lib reaches cells several
    directories away."""
    from silk.pdk.validate import resolve_lib_closure
    (tmp_path / 'corners').mkdir()
    (tmp_path / 'cells').mkdir()
    (tmp_path / 'sky.lib.spice').write_text('.lib tt\n.include "corners/tt.spice"\n.endl tt\n')
    (tmp_path / 'corners' / 'tt.spice').write_text('.include "../cells/nfet.spice"\n')
    (tmp_path / 'cells' / 'nfet.spice').write_text('.subckt nfet d g s b\n.ends\n')
    files = resolve_lib_closure(str(tmp_path / 'sky.lib.spice'))
    assert str(tmp_path / 'cells' / 'nfet.spice') in files
    assert len(files) == 3


def test_closure_handles_unquoted_and_absolute_includes(tmp_path):
    from silk.pdk.validate import resolve_lib_closure
    (tmp_path / 'bare.spice').write_text('.subckt bare a b\n.ends\n')
    (tmp_path / 'abs.spice').write_text('.subckt abs a b\n.ends\n')
    (tmp_path / 'sky.lib.spice').write_text(
        f'.inc bare.spice\n.include {tmp_path / "abs.spice"}\n')
    files = resolve_lib_closure(str(tmp_path / 'sky.lib.spice'))
    assert str(tmp_path / 'bare.spice') in files
    assert str(tmp_path / 'abs.spice') in files


def test_closure_tolerates_sections_cycles_and_missing_files(tmp_path):
    """`.lib <section>` names a section, not a path; a self/mutual include
    must not loop forever; a dangling `.include` is skipped, not raised on
    (PDK libs routinely reference optional files)."""
    from silk.pdk.validate import resolve_lib_closure
    (tmp_path / 'a.spice').write_text(
        '.lib tt\n.include "b.spice"\n.include "nope_missing.spice"\n.endl tt\n')
    (tmp_path / 'b.spice').write_text('.include "a.spice"\n.subckt b x y\n.ends\n')
    files = resolve_lib_closure(str(tmp_path / 'a.spice'))
    assert sorted(files) == sorted([str(tmp_path / 'a.spice'), str(tmp_path / 'b.spice')])


def test_closure_ignores_section_headers_named_like_a_file(tmp_path):
    """`.lib tt` is the header that OPENS a section, not an include. Treating
    its argument as a path pulls in any file that happens to be named after a
    section ('tt' beside the lib), letting definitions the simulator never
    binds into the scope. A one-token `.lib` is only followed when it looks
    like a path."""
    from silk.pdk.validate import resolve_lib_closure
    (tmp_path / 'tt').write_text('.subckt smuggled a b c\n.ends\n')
    (tmp_path / 'mylib.spice').write_text('.subckt real a b\n.ends\n')
    (tmp_path / 'sky.lib.spice').write_text(
        '.lib tt\n.lib mylib.spice\n.endl tt\n')
    files = resolve_lib_closure(str(tmp_path / 'sky.lib.spice'))
    assert str(tmp_path / 'tt') not in files
    assert str(tmp_path / 'mylib.spice') in files      # path-shaped: still followed


def test_closure_follows_the_two_token_lib_form(tmp_path):
    """`.lib <file> <section>` names a real file — unchanged by the one-token
    section rule, even when the file name has no extension."""
    from silk.pdk.validate import resolve_lib_closure
    (tmp_path / 'corner_defs').write_text('.subckt real a b\n.ends\n')
    (tmp_path / 'sky.lib.spice').write_text('.lib corner_defs tt\n')
    files = resolve_lib_closure(str(tmp_path / 'sky.lib.spice'))
    assert str(tmp_path / 'corner_defs') in files


def test_closure_seeds_extra_includes(tmp_path):
    """Manifest `library.extra_includes` cells live outside the lib's own
    include chain but ARE emitted by the netlister, so they must be in
    scope — otherwise sky130's cap_vpp_11p5x11p7 regresses to
    missing_definition."""
    from silk.pdk.validate import resolve_lib_closure, scan_lib_files
    (tmp_path / 'sky.lib.spice').write_text('.lib tt\n.endl tt\n')
    (tmp_path / 'cap_vpp.spice').write_text('.subckt cap_vpp c0 c1 b\n.ends\n')
    defs = scan_lib_files(resolve_lib_closure(
        str(tmp_path / 'sky.lib.spice'), [str(tmp_path / 'cap_vpp.spice')]))
    assert defs['cap_vpp'] == {('subckt', 3)}


class _FakeDev:
    def __init__(self, pdk_device, pins, spice_prefix='XM'):
        self.pdk_device = pdk_device
        self.pin_list = {p: p for p in pins}
        # PDKDevice.spice_prefix — what the validator reads for its
        # wrong_emission verdict (TASK-23). _FakePDK overwrites it from the
        # per-device tuple so the fake class and device always agree.
        self.spice_prefix = spice_prefix


class _FakeCls:
    def __init__(self, prefix):
        self._spice_prefix = prefix


class _FakePDK:
    library_location = None

    def __init__(self, devs):
        self._devices = {n: d for n, (d, _p) in devs.items()}
        self._p = {n: p for n, (_d, p) in devs.items()}
        for n, (d, p) in devs.items():
            d.spice_prefix = p

    def __getattr__(self, n):  # getattr(pdk, name)._spice_prefix
        if n in self.__dict__.get('_p', {}):
            return _FakeCls(self._p[n])
        raise AttributeError(n)


def test_validate_flags_arity_and_emission(tmp_path, monkeypatch):
    from silk.pdk import validate as V
    monkeypatch.setattr(V, 'scan_lib_definitions', lambda root: {
        'sub3': {('subckt', 3)}, 'mod_d': {('model', 'd')}})
    pdk = _FakePDK({
        'res_bad':  (_FakeDev('sub3', ['a', 'b']), 'XM'),      # 2 pins vs 3 -> arity_mismatch
        'res_ok':   (_FakeDev('sub3', ['a', 'b', 'c']), 'XM'), # ok
        'diode_bad':(_FakeDev('mod_d', ['a', 'k']), 'XM'),     # model but XM -> wrong_emission
        'diode_ok': (_FakeDev('mod_d', ['a', 'k']), 'D'),      # ok
        'ghost':    (_FakeDev('nope', ['a']), 'XM'),           # missing_definition
    })
    issues = V.validate_pdk(pdk, lib_root=str(tmp_path))
    by = {(i.device, i.kind) for i in issues}
    assert ('res_bad', 'arity_mismatch') in by
    assert ('diode_bad', 'wrong_emission') in by
    assert ('ghost', 'missing_definition') in by
    assert not any(i.device in ('res_ok', 'diode_ok') for i in issues)


def test_validate_clean_pdk_has_no_issues(tmp_path, monkeypatch):
    """Clean/well-formed devices produce zero issues."""
    from silk.pdk import validate as V
    monkeypatch.setattr(V, 'scan_lib_definitions', lambda root: {
        'sub3': {('subckt', 3)}, 'mod_d': {('model', 'd')}})
    pdk = _FakePDK({
        'res_ok':   (_FakeDev('sub3', ['a', 'b', 'c']), 'XM'),
        'diode_ok': (_FakeDev('mod_d', ['a', 'k']), 'D'),
    })
    assert V.validate_pdk(pdk, lib_root=str(tmp_path)) == []


def test_validate_no_false_positive_when_name_has_multiple_arities(tmp_path, monkeypatch):
    """Bug 2 regression: a subckt name defined with multiple different
    arities across the tree (corner/variant/parasitic files) must not be
    flagged as arity_mismatch as long as ONE of those arities matches the
    device's pinList length."""
    from silk.pdk import validate as V
    monkeypatch.setattr(V, 'scan_lib_definitions', lambda root: {
        'pnp_multi': {('subckt', 3), ('subckt', 4)}})
    pdk = _FakePDK({
        'pnp_4pin': (_FakeDev('pnp_multi', ['c', 'b', 'e', 's']), 'XM'),  # matches the 4-arity def
    })
    assert V.validate_pdk(pdk, lib_root=str(tmp_path)) == []


def test_validate_arity_mismatch_when_no_arity_in_set_matches(tmp_path, monkeypatch):
    from silk.pdk import validate as V
    monkeypatch.setattr(V, 'scan_lib_definitions', lambda root: {
        'pnp_multi': {('subckt', 3), ('subckt', 4)}})
    pdk = _FakePDK({
        'pnp_2pin': (_FakeDev('pnp_multi', ['a', 'b']), 'XM'),  # matches neither 3 nor 4
    })
    issues = V.validate_pdk(pdk, lib_root=str(tmp_path))
    assert len(issues) == 1
    assert issues[0].device == 'pnp_2pin'
    assert issues[0].kind == 'arity_mismatch'
    assert '[3, 4]' in issues[0].detail


def test_validate_mixed_subckt_and_model_defs_uses_subckt_branch(tmp_path, monkeypatch):
    """If a name has BOTH a subckt def (some file) and a model def (another
    file), the subckt branch takes precedence for arity/emission checks."""
    from silk.pdk import validate as V
    monkeypatch.setattr(V, 'scan_lib_definitions', lambda root: {
        'mixed': {('subckt', 3), ('model', 'd')}})
    pdk = _FakePDK({
        'dev_ok': (_FakeDev('mixed', ['a', 'b', 'c']), 'XM'),
    })
    assert V.validate_pdk(pdk, lib_root=str(tmp_path)) == []


# ---------------------------------------------------------------------------
# Definition scope: which files the cross-check is allowed to believe
# ---------------------------------------------------------------------------

def _two_tree_pdk_root(tmp_path):
    """$PDK_ROOT holding two unrelated PDK trees that define `pnp_05v5` at
    DIFFERENT arities: the installed lib silk netlists against (3-terminal)
    and a vendor source checkout that is never netlisted (4-terminal)."""
    installed = tmp_path / 'sky130A'
    installed.mkdir()
    (installed / 'pnp.spice').write_text('.subckt pnp_05v5 c b e\n.ends\n')
    (installed / 'sky130.lib.spice').write_text(
        '.lib tt\n.include "pnp.spice"\n.endl tt\n')
    vendor = tmp_path / 'skywater-pdk'
    vendor.mkdir()
    (vendor / 'pnp.spice').write_text('.subckt pnp_05v5 c b e s\n.ends\n')
    return str(installed / 'sky130.lib.spice')


def test_unrelated_tree_cannot_launder_arity_mismatch(tmp_path, monkeypatch):
    """The scope bug this closes. $PDK_ROOT routinely holds several unrelated
    PDK trees; a definition from a tree that is never netlisted must NOT
    vouch for a device whose in-chain subckt disagrees. That laundering is
    exactly how silk's wrong 4-pin sky130 pnp_05v5 pinList passed validation
    while ngspice-47 rejected the 4-node instance line."""
    from silk.pdk import validate as V
    lib_file = _two_tree_pdk_root(tmp_path)
    monkeypatch.setenv('PDK_ROOT', str(tmp_path))

    # The unrelated 4-terminal definition really is under $PDK_ROOT: the
    # unscoped walk still sees it, which is what used to make this pass.
    assert ('subckt', 4) in V.scan_lib_definitions(str(tmp_path))['pnp_05v5']

    pdk = _FakePDK({'pnp': (_FakeDev('pnp_05v5', ['c', 'b', 'e', 's']), 'XM')})
    pdk.library_location = lib_file
    issues = V.validate_pdk(pdk)
    assert [(i.device, i.kind) for i in issues] == [('pnp', 'arity_mismatch')]
    assert '[3]' in issues[0].detail


def test_in_chain_definition_still_passes_a_matching_pinlist(tmp_path, monkeypatch):
    """The other side of the same scoping: a device matching the in-chain
    arity is clean, even though the unrelated tree disagrees."""
    from silk.pdk import validate as V
    lib_file = _two_tree_pdk_root(tmp_path)
    monkeypatch.setenv('PDK_ROOT', str(tmp_path))
    pdk = _FakePDK({'pnp': (_FakeDev('pnp_05v5', ['c', 'b', 'e']), 'XM')})
    pdk.library_location = lib_file
    assert V.validate_pdk(pdk) == []


def test_collect_definitions_falls_back_to_walk_without_a_lib_file(tmp_path, monkeypatch):
    """A PDK still under construction has the /dev/null library_location
    placeholder and so no closure root; the unscoped $PDK_ROOT walk is the
    only thing left, and must still find definitions (rather than reporting
    everything as missing_definition).

    The placeholder is the ONLY path that still degrades this way — it
    declares "there is no lib", where a path that names a missing file is a
    typo (test_nonexistent_lib_file_refuses_to_fall_back)."""
    from silk.pdk import validate as V
    _two_tree_pdk_root(tmp_path)
    monkeypatch.setenv('PDK_ROOT', str(tmp_path))
    pdk = _FakePDK({'pnp': (_FakeDev('pnp_05v5', ['c', 'b', 'e', 's']), 'XM')})
    pdk.library_location = os.devnull
    assert V.validate_pdk(pdk) == []   # laundered by the vendor tree — see docstring


def test_nonexistent_lib_file_refuses_to_fall_back(tmp_path, monkeypatch):
    """The laundering hole this closes: a library_location naming a file that
    isn't there (one wrong character in --lib) used to degrade to the unscoped
    $PDK_ROOT walk and report a clean bill — re-opening, through a typo,
    exactly the laundering the closure scoping was introduced to remove.

    Same fixture as test_unrelated_tree_cannot_launder_arity_mismatch, so the
    device really would be vouched for by the vendor tree if we fell back."""
    from silk.pdk import validate as V
    _two_tree_pdk_root(tmp_path)
    monkeypatch.setenv('PDK_ROOT', str(tmp_path))
    pdk = _FakePDK({'pnp': (_FakeDev('pnp_05v5', ['c', 'b', 'e', 's']), 'XM')})
    pdk.library_location = str(tmp_path / 'sky130A' / 'sky130.lib.spce')  # typo'd ext
    with pytest.raises(V.PDKValidationError) as exc:
        V.validate_pdk(pdk)
    assert 'sky130.lib.spce' in str(exc.value)


def test_directory_as_library_location_is_rejected(tmp_path, monkeypatch):
    """A lib *tree* passed where the lib *file* belongs can't root a closure;
    say so instead of walking something. (`lib_root=` is the flag that takes a
    tree — see test_explicit_lib_root_overrides_the_lib_closure.)"""
    from silk.pdk import validate as V
    _two_tree_pdk_root(tmp_path)
    monkeypatch.setenv('PDK_ROOT', str(tmp_path))
    pdk = _FakePDK({'pnp': (_FakeDev('pnp_05v5', ['c', 'b', 'e']), 'XM')})
    pdk.library_location = str(tmp_path / 'sky130A')
    with pytest.raises(V.PDKValidationError, match='directory'):
        V.validate_pdk(pdk)


def test_devnull_placeholder_without_pdk_root_raises(monkeypatch):
    """Preserved behaviour: the placeholder leaves the fallback walk as the
    only scope, and with no $PDK_ROOT there is no fallback either."""
    from silk.pdk import validate as V
    monkeypatch.delenv('PDK_ROOT', raising=False)
    pdk = _FakePDK({'pnp': (_FakeDev('pnp_05v5', ['c', 'b', 'e']), 'XM')})
    pdk.library_location = os.devnull
    with pytest.raises(V.PDKValidationError, match='no PDK lib root'):
        V.validate_pdk(pdk)


def test_explicit_lib_root_overrides_the_lib_closure(tmp_path, monkeypatch):
    """`lib_root=` keeps meaning "check against exactly this tree" and wins
    over the PDK's own lib file."""
    from silk.pdk import validate as V
    lib_file = _two_tree_pdk_root(tmp_path)
    monkeypatch.setenv('PDK_ROOT', str(tmp_path))
    pdk = _FakePDK({'pnp': (_FakeDev('pnp_05v5', ['c', 'b', 'e', 's']), 'XM')})
    pdk.library_location = lib_file
    assert V.validate_pdk(pdk, lib_root=str(tmp_path / 'skywater-pdk')) == []
    assert len(V.validate_pdk(pdk, lib_root=str(tmp_path / 'sky130A'))) == 1


def test_lib_root_prefers_explicit_arg(tmp_path, monkeypatch):
    from silk.pdk.validate import _lib_root
    monkeypatch.setenv('PDK_ROOT', str(tmp_path))
    assert _lib_root(None, '/explicit/path') == '/explicit/path'


def test_lib_root_falls_back_to_library_location(tmp_path, monkeypatch):
    """library_location is used (dirname taken) only when it's a real file
    and no PDK_ROOT is set."""
    from silk.pdk.validate import _lib_root
    monkeypatch.delenv('PDK_ROOT', raising=False)
    lib_file = tmp_path / 'sky130.lib.spice'
    lib_file.write_text('* lib\n')

    class P:
        library_location = str(lib_file)
    assert _lib_root(P(), None) == str(tmp_path)


def test_lib_root_falls_back_to_env(tmp_path, monkeypatch):
    from silk.pdk.validate import _lib_root
    monkeypatch.setenv('PDK_ROOT', str(tmp_path))

    class P:
        library_location = None
    assert _lib_root(P(), None) == str(tmp_path)


def test_lib_root_prefers_env_over_library_location(tmp_path, monkeypatch):
    """Within the _lib_root FALLBACK, PDK_ROOT (whole tree) wins over
    library_location (one lib file's dir) even when library_location is a
    legitimate file — one lib file's directory is far too narrow to find all
    cell subckts by walking. Note this is only the fallback ordering:
    validate_pdk prefers the include closure rooted at library_location over
    _lib_root entirely (see test_unrelated_tree_cannot_launder_arity_mismatch)."""
    from silk.pdk.validate import _lib_root
    pdk_root = tmp_path / 'pdk_root'
    pdk_root.mkdir()
    monkeypatch.setenv('PDK_ROOT', str(pdk_root))
    other_dir = tmp_path / 'other'
    other_dir.mkdir()
    lib_file = other_dir / 'sky130.lib.spice'
    lib_file.write_text('* lib\n')

    class P:
        library_location = str(lib_file)
    assert _lib_root(P(), None) == str(pdk_root)


def test_lib_root_skips_dev_null_placeholder(monkeypatch):
    """Bug 1 regression: loaded PDKs commonly set library_location='/dev/null'
    as a placeholder. /dev/null exists but is not a regular file, so it must
    be skipped rather than treated as a real lib (which used to make
    _lib_root return '/dev' and silently miss every device definition)."""
    from silk.pdk.validate import _lib_root, PDKValidationError
    monkeypatch.delenv('PDK_ROOT', raising=False)

    class P:
        library_location = '/dev/null'
    try:
        _lib_root(P(), None)
        assert False, 'expected PDKValidationError'
    except PDKValidationError:
        pass


def test_lib_root_raises_when_nothing_available(monkeypatch):
    from silk.pdk.validate import _lib_root, PDKValidationError
    monkeypatch.delenv('PDK_ROOT', raising=False)

    class P:
        library_location = None
    try:
        _lib_root(P(), None)
        assert False, 'expected PDKValidationError'
    except PDKValidationError:
        pass


def test_validate_pdk_simulate_stub_returns_no_extra_issues(tmp_path, monkeypatch):
    """simulate=True calls the (stubbed) dynamic pass; stub returns []."""
    from silk.pdk import validate as V
    monkeypatch.setattr(V, 'scan_lib_definitions', lambda root: {'sub3': {('subckt', 3)}})
    monkeypatch.setattr(V, '_simulate_devices', lambda pdk: [])
    pdk = _FakePDK({'res_ok': (_FakeDev('sub3', ['a', 'b', 'c']), 'XM')})
    assert V.validate_pdk(pdk, lib_root=str(tmp_path), simulate=True) == []


@pytest.mark.skipif(not os.environ.get('PDK_ROOT'), reason='needs installed PDK')
@needs_pdk
def test_real_pdk_has_no_validation_issues():
    """The permanent guard: every shipped sky130 device passes the static
    cross-check against the lib silk actually netlists against. This went RED
    before the PV3 yaml fixes (6 real issues: 5 arity_mismatch + diode_nw
    missing_definition) and again for the 4-pin pnp_05v5 pinLists once the
    check was scoped to the in-chain lib; it is now GREEN. It guards against
    any future device-details entry whose pinList arity or PDKDevice name
    diverges from the real sky130 subckt/model.

    Loaded via load_pdk (not the bare BasePDK constructor, which
    takes the /dev/null library_location placeholder) so the manifest's
    lib_path + extra_includes resolve and validate_pdk checks the include
    closure. Under the placeholder there is no closure root and the check
    degrades to the unscoped $PDK_ROOT walk, which cannot tell this PDK's
    definitions from a neighbouring tree's — i.e. it would not be a guard."""
    from silk import load_pdk
    from silk.pdk.validate import validate_pdk
    pdk = load_pdk('sky130')
    issues = validate_pdk(pdk)
    assert issues == [], sorted(f'{i.device}:{i.kind}:{i.detail}' for i in issues)


# ---------------------------------------------------------------------------
# Dynamic pass (Task 4): _simulate_devices / _scan_sim_log
# ---------------------------------------------------------------------------

def test_scan_sim_log_clean_returns_none():
    from silk.pdk.validate import _scan_sim_log
    assert _scan_sim_log('') is None
    assert _scan_sim_log(None) is None
    assert _scan_sim_log(
        'Circuit: vet_res_hpo\n'
        'Doing analysis at TEMP = 27.000000 and TNOM = 27.000000\n'
        'No. of Data Rows : 3\n'
    ) is None


def test_scan_sim_log_flags_unknown_subckt():
    from silk.pdk.validate import _scan_sim_log
    hit = _scan_sim_log('Circuit: vet_res_hpo\nunknown subckt: sky130_fd_pr__res_high_po\n')
    assert hit is not None
    assert 'unknown subckt' in hit.lower()


def test_scan_sim_log_flags_singular_matrix_case_insensitive():
    from silk.pdk.validate import _scan_sim_log
    hit = _scan_sim_log('...\nError: Singular matrix\n')
    assert hit is not None


def test_scan_sim_log_flags_fatal_keyword():
    from silk.pdk.validate import _scan_sim_log
    hit = _scan_sim_log('run started\nfatal error in netlist parse\ndone\n')
    assert hit is not None
    assert 'fatal' in hit


def test_device_parameters_uses_minval():
    """_device_parameters builds params from each param's minval — a valid
    value for every shipped device (Task 4 brief: maxval == -1 just means
    unbounded, minval is always usable)."""
    from silk.pdk import validate as V

    class _Constraint:
        def __init__(self, minval):
            self.minval = minval

    class _Dev:
        param_list = {'w': _Constraint('0.42'), 'm': _Constraint(1)}

    params = V._device_parameters(_Dev())
    assert params == {'w': 0.42, 'm': 1.0}


def test_device_parameters_no_param_list_returns_empty():
    from silk.pdk import validate as V

    class _Dev:
        pass  # no param_list attribute at all

    assert V._device_parameters(_Dev()) == {}


class _FakeSimModule:
    """Stand-in for silk.circuit.modules.Module used to exercise
    _simulate_devices' exception-catch / log-scan wiring without ngspice or
    a real PDK. Records the (name, nets, parameters) it was built with so
    tests can assert on the topology _simulate_devices constructs.
    """
    behavior = 'clean'   # class-level default; per-test instances override

    def __init__(self, pdk_name, dir_folder, netlistname, pdk_details=None, module_name=None):
        self.pdk_name = pdk_name
        type(self).last_pdk_name = pdk_name
        self.dir_folder = dir_folder
        self.nets_added = []
        self.sources_added = []
        self.element = None

    def add_net(self, netname, *a, **kw):
        self.nets_added.append(netname)

    def add_source(self, source):
        self.sources_added.append(source)

    def add_element(self, device_cls, inst_name, nets=None, parameters=None):
        self.element = (device_cls, inst_name, nets, parameters)

    def add_sim_type(self, sim_type, details=None):
        pass

    def simulate(self, corner='tt'):
        if self.behavior == 'raises':
            raise RuntimeError('ngspice simulation failed: unknown subckt frobnicator')
        from silk.simulation.run_files import RunFiles
        run_folder = os.path.join(self.dir_folder, 'run-fake')
        os.makedirs(run_folder, exist_ok=True)
        self._run_files = RunFiles.resolve(
            'vet.cir', run_folder, log=True)
        # A stale root log must never mask the current run's diagnostics.
        Path(self.dir_folder, 'output.log').write_text('No. of Data Rows : 1\n')
        if self.behavior == 'missing_log':
            return
        if self.behavior == 'no_log_record':
            self._run_files = None
            return
        if self.behavior == 'unreadable_log':
            Path(self._run_files.logfile_path).mkdir()
            return
        text = ('Circuit: vet\nunknown subckt frobnicator\n'
                if self.behavior == 'silent_fail'
                else 'Circuit: vet\nNo. of Data Rows : 1\n')
        Path(self._run_files.logfile_path).write_text(text)


def test_simulate_devices_labels_module_with_the_loaded_pdk_name(monkeypatch):
    """The vetting Module's label used to be a fixed 'PDK_skywater130' for
    every --pdk (TASK-23). It now names the PDK actually loaded — pdk_name
    from the manifest, or the PDK class when constructed bare."""
    from silk.pdk import validate as V

    class _LabelModule(_FakeSimModule):
        behavior = 'clean'

    monkeypatch.setattr(V, 'Module', _LabelModule)
    pdk = _FakePDK({'dev': (_FakeDev('sub3', ['a', 'b', 'c']), 'XM')})
    pdk.pdk_name = 'gf180mcu'
    assert V._simulate_devices(pdk) == []
    assert _LabelModule.last_pdk_name == 'gf180mcu'

    bare = _FakePDK({'dev': (_FakeDev('sub3', ['a', 'b', 'c']), 'XM')})
    assert V._pdk_label(bare) == '_FakePDK'
    assert V._simulate_devices(bare) == []
    assert _LabelModule.last_pdk_name == '_FakePDK'
    assert 'skywater' not in V._pdk_label(bare)


def test_validate_reads_the_prefix_from_the_device_not_the_class(tmp_path, monkeypatch):
    """wrong_emission is judged on PDKDevice.spice_prefix — the value the
    netlister writes — so a class/device disagreement cannot hide a
    mis-emission. Here the fake class says 'D' but the device says 'XM'."""
    from silk.pdk import validate as V
    monkeypatch.setattr(V, 'scan_lib_definitions', lambda root: {
        'mod_d': {('model', 'd')}})
    pdk = _FakePDK({'diode': (_FakeDev('mod_d', ['a', 'k']), 'D')})
    pdk._devices['diode'].spice_prefix = 'XM'   # device disagrees with class
    issues = V.validate_pdk(pdk, lib_root=str(tmp_path))
    assert [(i.device, i.kind) for i in issues] == [('diode', 'wrong_emission')]


def test_simulate_devices_flags_sim_failed_on_simulate_exception(monkeypatch):
    """A device whose .simulate() raises is a real "cannot be simulated"
    signal -> sim_failed, with the exception message as detail."""
    from silk.pdk import validate as V

    class _RaisingModule(_FakeSimModule):
        behavior = 'raises'

    monkeypatch.setattr(V, 'Module', _RaisingModule)
    pdk = _FakePDK({'bad_dev': (_FakeDev('sub3', ['a', 'b', 'c']), 'XM')})
    issues = V._simulate_devices(pdk)
    assert len(issues) == 1
    assert issues[0].device == 'bad_dev'
    assert issues[0].kind == 'sim_failed'
    assert 'unknown subckt' in issues[0].detail


def test_simulate_devices_flags_sim_failed_on_silent_log_error(monkeypatch):
    """ngspice can exit 0 while the log shows a singular matrix / unknown
    subckt line (the A1 silent mis-parse failure mode) — this must also be
    caught even though .simulate() itself didn't raise."""
    from silk.pdk import validate as V

    class _SilentFailModule(_FakeSimModule):
        behavior = 'silent_fail'

    monkeypatch.setattr(V, 'Module', _SilentFailModule)
    pdk = _FakePDK({'sneaky_dev': (_FakeDev('sub3', ['a', 'b', 'c']), 'XM')})
    issues = V._simulate_devices(pdk)
    assert len(issues) == 1
    assert issues[0].device == 'sneaky_dev'
    assert issues[0].kind == 'sim_failed'
    assert 'unknown subckt' in issues[0].detail.lower()


@pytest.mark.parametrize('behavior', ['missing_log', 'unreadable_log', 'no_log_record'])
def test_simulate_devices_flags_unreadable_run_log(monkeypatch, behavior):
    from silk.pdk import validate as V

    class _MissingLogModule(_FakeSimModule):
        pass

    _MissingLogModule.behavior = behavior
    monkeypatch.setattr(V, 'Module', _MissingLogModule)
    pdk = _FakePDK({'dev': (_FakeDev('sub3', ['a', 'b', 'c']), 'XM')})
    issues = V._simulate_devices(pdk)
    assert [(i.device, i.kind) for i in issues] == [('dev', 'sim_failed')]
    assert 'log' in issues[0].detail.lower()


def test_simulate_devices_clean_returns_no_issues(monkeypatch):
    from silk.pdk import validate as V

    class _CleanModule(_FakeSimModule):
        behavior = 'clean'

    monkeypatch.setattr(V, 'Module', _CleanModule)
    pdk = _FakePDK({'good_dev': (_FakeDev('sub3', ['a', 'b', 'c']), 'XM')})
    assert V._simulate_devices(pdk) == []


def test_simulate_devices_wires_every_pin_to_its_own_node(monkeypatch):
    """Topology check (Task 4 override): every pin gets its own node and its
    own independent source to ground — no two pins share a node."""
    from silk.pdk import validate as V

    built = {}

    class _RecordingModule(_FakeSimModule):
        def simulate(self, corner='tt'):
            built['nets'] = dict(self.element[2])
            built['n_sources'] = len(self.sources_added)
            built['n_nets'] = len(self.nets_added)
            super().simulate(corner)

    monkeypatch.setattr(V, 'Module', _RecordingModule)
    pdk = _FakePDK({'three_pin': (_FakeDev('sub3', ['a', 'b', 'c']), 'XM')})
    assert V._simulate_devices(pdk) == []
    assert len(set(built['nets'].values())) == 3   # 3 distinct nodes, one per pin
    assert built['n_sources'] == 3
    assert built['n_nets'] == 3


def test_simulate_devices_build_failure_is_also_sim_failed(monkeypatch):
    """A device that raises while being BUILT (e.g. add_element rejects a
    bad param) is also a "cannot be simulated" signal, not just a raise
    from .simulate()."""
    from silk.pdk import validate as V

    class _BuildFailsModule(_FakeSimModule):
        def add_element(self, device_cls, inst_name, nets=None, parameters=None):
            raise ValueError('missing required parameter w')

    monkeypatch.setattr(V, 'Module', _BuildFailsModule)
    pdk = _FakePDK({'unbuildable': (_FakeDev('sub3', ['a', 'b', 'c']), 'XM')})
    issues = V._simulate_devices(pdk)
    assert len(issues) == 1
    assert issues[0].device == 'unbuildable'
    assert issues[0].kind == 'sim_failed'
    assert 'missing required parameter' in issues[0].detail


# cap_vpp_11p5x11p7 used to be a KNOWN, backlogged limitation: its subckt lives
# outside the tt corner lib's own `.include` chain, so silk's single-`.lib`
# netlist couldn't bind it without emitting an extra `.include` for the cell
# file. Fixed by the "PDK library config in the yaml" work — the manifest now
# carries `extra_includes` and the netlister emits them (silk/pdk/data/
# sky130.pdk.yaml, silk/circuit/spice_netlister.py:generate_pdk_include). See
# docs/adding-a-pdk.md for extra cell include configuration.
KNOWN_SIM_UNSUPPORTED = set()   # cap_vpp_11p5x11p7 fixed via manifest extra_includes


@pytest.mark.slow
@pytest.mark.skipif(not (os.environ.get('PDK_ROOT')
                         and shutil.which(ngspice_executable())),
                    reason='needs PDK + ngspice')
def test_all_devices_simulate():
    """The permanent dynamic guard: every shipped sky130 device binds and
    simulates cleanly against the real installed PDK, EXCEPT the documented
    KNOWN_SIM_UNSUPPORTED set. Any new sim failure (or an unexpected recovery of
    a known one) trips this test. The ESD FETs (nmos_esd/pmos_esd) are expected
    to PASS here thanks to the corrected w/l dimensions. Loaded via load_pdk so
    the manifest's extra_includes (needed to bind cap_vpp_11p5x11p7) apply —
    the direct BasePDK(...) constructor path has no extra_includes."""
    from silk import load_pdk
    from silk.pdk.validate import validate_pdk
    pdk = load_pdk('sky130')            # manifest lib + extra_includes
    failed = {i.device for i in validate_pdk(pdk, simulate=True)
              if i.kind == 'sim_failed'}
    assert failed == KNOWN_SIM_UNSUPPORTED, (
        f'unexpected sim outcome: newly failing={sorted(failed - KNOWN_SIM_UNSUPPORTED)}, '
        f'newly passing (drop from KNOWN_SIM_UNSUPPORTED)={sorted(KNOWN_SIM_UNSUPPORTED - failed)}')


# ---------------------------------------------------------------------------
# CLI (Task 5): python -m silk.pdk.validate
# ---------------------------------------------------------------------------

def test_cli_returns_1_and_prints_issues(monkeypatch, capsys):
    """main() exits 1 and prints each issue when validation is non-clean —
    no PDK/ngspice needed. Uses the ad-hoc `--yaml` path (generic BasePDK,
    lazy lib) so it doesn't require $PDK_ROOT; the real validation is stubbed."""
    from silk.pdk import validate as V
    monkeypatch.setattr(V, '_validate_with_scope', lambda pdk, simulate=False: (
        [V.DeviceIssue('res_bad', 'arity_mismatch', 'x')], 'closure of fake.lib, 3 files'))
    rc = V.main(['--yaml', V._default_yaml()])
    out = capsys.readouterr().out
    assert rc == 1
    assert 'res_bad' in out and 'arity_mismatch' in out
    assert 'scope: closure of fake.lib, 3 files' in out   # issues disclose it too


def test_cli_reports_ok_when_clean(monkeypatch, capsys):
    from silk.pdk import validate as V
    monkeypatch.setattr(V, '_validate_with_scope',
                        lambda pdk, simulate=False: ([], 'closure of fake.lib, 3 files'))
    rc = V.main(['--yaml', V._default_yaml()])
    out = capsys.readouterr().out
    assert rc == 0
    assert 'no issues' in out


def test_cli_ok_line_discloses_the_scope(monkeypatch, capsys):
    """The point of the disclosure: 'no issues' from the lib closure and 'no
    issues' from the unscoped $PDK_ROOT walk used to be byte-identical, so a
    verdict laundered by an unrelated PDK tree read exactly like a real one."""
    from silk.pdk import validate as V
    monkeypatch.setattr(V, '_validate_with_scope', lambda pdk, simulate=False: (
        [], 'unscoped walk of $PDK_ROOT (/pdk) — no lib file'))
    rc = V.main(['--yaml', V._default_yaml()])
    out = capsys.readouterr().out
    assert rc == 0
    assert 'no issues (scope: unscoped walk of $PDK_ROOT (/pdk) — no lib file).' in out


def test_scope_names_the_closure_and_its_file_count(tmp_path):
    """The scope string comes from the same call that produced the issues, so
    it reports the files actually scanned rather than a re-derived guess."""
    from silk.pdk import validate as V
    (tmp_path / 'cell.spice').write_text('.subckt sub3 a b c\n.ends\n')
    lib = tmp_path / 'sky.lib.spice'
    lib.write_text('.lib tt\n.include "cell.spice"\n.endl tt\n')
    pdk = _FakePDK({'ok3': (_FakeDev('sub3', ['a', 'b', 'c']), 'XM')})
    pdk.library_location = str(lib)
    issues, scope = V._validate_with_scope(pdk)
    assert issues == []
    assert scope == f'closure of {lib}, 2 files'


def test_scope_marks_the_unscoped_fallback_as_unscoped(tmp_path, monkeypatch):
    """No lib file -> the $PDK_ROOT walk, and the verdict must say so."""
    from silk.pdk import validate as V
    monkeypatch.setenv('PDK_ROOT', str(tmp_path))
    (tmp_path / 'a.spice').write_text('.subckt sub3 a b c\n.ends\n')
    pdk = _FakePDK({'ok3': (_FakeDev('sub3', ['a', 'b', 'c']), 'XM')})
    pdk.library_location = os.devnull        # documented "no lib" placeholder
    issues, scope = V._validate_with_scope(pdk)
    assert issues == []
    assert scope == f'unscoped walk of $PDK_ROOT ({tmp_path}) — no lib file'


def test_cli_unknown_pdk_exits_2(capsys):
    """An unregistered --pdk name fails clearly (exit 2) before touching any
    lib — so this needs neither $PDK_ROOT nor ngspice."""
    from silk.pdk.validate import main
    rc = main(['--pdk', 'definitely-not-a-pdk'])
    out = capsys.readouterr().out
    assert rc == 2
    assert 'definitely-not-a-pdk' in out


def test_cli_typoed_lib_exits_2_without_scanning(capsys):
    """CLI face of the laundering hole: `--lib sky130.lib.spce` used to spend
    seconds on the unscoped walk and print 'OK: sky130: 33 devices, no
    issues.' with exit 0. It must now fail fast, on one line, naming the path.

    Needs no $PDK_ROOT: an explicit library_location skips load_pdk's own
    existence check, which is what made the typo CLI-reachable."""
    from silk.pdk.validate import main
    rc = main(['--pdk', 'sky130', '--lib', '/nonexistent/sky130.lib.spce'])
    out = capsys.readouterr().out
    assert rc == 2, out
    assert 'no issues' not in out
    errors = [ln for ln in out.splitlines() if ln.startswith('validate:')]
    assert len(errors) == 1, out
    assert '/nonexistent/sky130.lib.spce' in errors[0]


@pytest.mark.skipif(not os.environ.get('PDK_ROOT'), reason='needs installed PDK')
@needs_pdk
def test_cli_pdk_flag_uses_registry(capsys):
    """--pdk <name> loads via the registry (manifest lib + corners), not a
    hardcoded PDK class; the shipped sky130 catalog validates clean."""
    from silk.pdk.validate import main
    rc = main(['--pdk', 'sky130'])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert 'no issues' in out


@pytest.mark.skipif(not os.environ.get('PDK_ROOT'), reason='needs installed PDK')
@needs_pdk
def test_cli_default_is_sky130_registry(capsys):
    """No args → default --pdk sky130 via the registry (exit 0 clean)."""
    from silk.pdk.validate import main
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert 'no issues' in out


# ---------------------------------------------------------------------------
# Junction-ratings provenance (TASK-48): check_junction_provenance
# ---------------------------------------------------------------------------

def _rated(ratings, prov):
    """A fake rated device: `prov` is None or (source, heuristic-dict)."""
    from silk.pdk.basepdk import JunctionRatingsProvenance
    d = _FakeDev('sub4', ['d', 'g', 's', 'b'])
    d.junction_ratings = dict(ratings)
    d.junction_ratings_provenance = (
        None if prov is None else JunctionRatingsProvenance(prov[0], dict(prov[1])))
    return d


def _prov_issues(devs):
    from silk.pdk.validate import check_junction_provenance
    pdk = _FakePDK({n: (d, 'XM') for n, d in devs.items()})
    return [(i.device, i.kind) for i in check_junction_provenance(pdk)]


_MOS = {'Vgs_max': 1.95, 'Vds_max': 1.95, 'Vgd_max': 1.95, 'Vbs_max': 1.95, 'Vbd_max': 1.95}


def test_provenance_unsourced_when_block_is_missing():
    """Pre-TASK-48 every shipped rating looked exactly like this — numbers
    with no provenance block — and the validator said 'no issues'. A rated
    device with no `junction_ratings_provenance` is now `unsourced_rating`."""
    assert _prov_issues({'m': _rated(_MOS, None)}) == [('m', 'unsourced_rating')]


def test_provenance_source_covers_every_key_not_listed_as_heuristic():
    src = 'skywater docs device-details/nfet_01v8: VDS/VGS 0-1.95 V, VBS -1.95 V'
    assert _prov_issues({'m': _rated(_MOS, (src, {
        'Vgd_max': '= Vds_max; gate at source potential',
        'Vbd_max': '= Vds_max; body at source potential'}))}) == []
    # a source alone covers everything
    assert _prov_issues({'m': _rated(_MOS, (src, {}))}) == []


def test_provenance_all_heuristic_needs_no_source():
    """A catalog with no foundry number at all may derive every key, as long
    as each rule is stated in a checkable form."""
    r = {'Vgs_max': 6.6, 'Vds_max': 6.6}
    assert _prov_issues({'m': _rated(r, (None, {
        'Vgs_max': '1.1 x nominal 6.0 V; supply overshoot tolerance',
        'Vds_max': '= Vgs_max'}))}) == []


def test_provenance_uncovered_key_without_source_is_unsourced():
    r = {'Vgs_max': 6.6, 'Vds_max': 6.6}
    issues = _prov_issues({'m': _rated(r, (None, {'Vgs_max': '1.1 x nominal 6.0 V'}))})
    assert issues == [('m', 'unsourced_rating')]


def test_provenance_equal_rule_is_checked_against_the_numbers():
    """'= Vds_max' on a key whose value is NOT Vds_max is a lie the validator
    must catch — the rule text is a claim about the numbers, not a comment."""
    r = dict(_MOS, Vgd_max=2.5)
    issues = _prov_issues({'m': _rated(r, ('doc', {'Vgd_max': '= Vds_max'}))})
    assert issues == [('m', 'bad_provenance')]


def test_provenance_factor_rule_is_checked_against_the_numbers():
    r = {'Vgs_max': 6.0}
    assert _prov_issues({'m': _rated(r, (None, {'Vgs_max': '1.1 x nominal 6.0 V'}))}) \
        == [('m', 'bad_provenance')]
    assert _prov_issues({'m': _rated({'Vgs_max': 6.6},
                                     (None, {'Vgs_max': '1.1 x nominal 6.0 V'}))}) == []


def test_provenance_rejects_unrecognised_rule_self_reference_and_unknown_key():
    from silk.pdk.validate import check_junction_provenance
    pdk = _FakePDK({
        'free_text': (_rated(_MOS, ('doc', {'Vgd_max': 'looks about right'})), 'XM'),
        'self_ref':  (_rated(_MOS, ('doc', {'Vgd_max': '= Vgd_max'})), 'XM'),
        'other_dev': (_rated(_MOS, ('doc', {'Vgd_max': '= Vce_max'})), 'XM'),
        'not_rated': (_rated(_MOS, ('doc', {'Vak_max': '= Vds_max'})), 'XM'),
        'empty_src': (_rated(_MOS, ('   ', {})), 'XM'),
    })
    kinds = {}
    for i in check_junction_provenance(pdk):
        kinds.setdefault(i.device, []).append(i.kind)
    assert kinds['free_text'] == ['bad_provenance']
    assert kinds['self_ref'] == ['bad_provenance']
    assert kinds['other_dev'] == ['bad_provenance']
    assert kinds['not_rated'] == ['bad_provenance']
    # a blank source is malformed AND leaves every key uncovered
    assert sorted(kinds['empty_src']) == ['bad_provenance', 'unsourced_rating']


def test_provenance_rating_convention_rejects_non_positive_values():
    """Ratings are magnitudes (STR-001 compares |V| > rating): a 0, negative,
    nan or inf value can never mean anything and is `rating_convention`."""
    for bad in (0.0, -1.95, float('nan'), float('inf')):
        issues = _prov_issues({'m': _rated(dict(_MOS, Vgs_max=bad), ('doc', {}))})
        assert issues == [('m', 'rating_convention')], bad


# --- signed-range ratings (TASK-91) -----------------------------------------

_MOS_SIGNED = dict(_MOS, Vbs_max={'min': -1.95, 'max': 0.3})
_SRC = 'skywater device-details/nfet_01v8: VDS/VGS 0-1.95 V, VBS +0.3 to -1.95 V'
_HEUR = {'Vgd_max': '= Vds_max; gate at source potential',
         'Vbd_max': '= Vds_max; body at source potential'}


def test_provenance_accepts_a_signed_range_beside_magnitudes():
    """AC #1. Pre-TASK-91 a mapping value was not a rating at all
    (``float({'min': ..})`` raised in the loader); a device mixing the two
    forms now validates clean, and the source still covers the signed key."""
    assert _prov_issues({'m': _rated(_MOS_SIGNED, (_SRC, _HEUR))}) == []
    # a signed range with no source is as unsourced as a magnitude
    assert _prov_issues({'m': _rated(_MOS_SIGNED, None)}) == [('m', 'unsourced_rating')]


def test_provenance_signed_range_rejects_min_not_below_max():
    """AC #1. ``{min: 0.3, max: -1.95}`` — the SkyWater bounds written in the
    order the page lists them ("+0.3 to -1.95 V") — is an inverted range
    that would make STR-001 flag every sample; ``rating_convention``, with
    the numbers in the message."""
    from silk.pdk.validate import check_junction_provenance
    for bad in ({'min': 0.3, 'max': -1.95}, {'min': 0.3, 'max': 0.3}):
        d = _rated(dict(_MOS, Vbs_max=bad), (_SRC, {}))
        issues = check_junction_provenance(_FakePDK({'m': (d, 'XM')}))
        assert [(i.device, i.kind) for i in issues] == [('m', 'rating_convention')], bad
        assert 'min >= max' in issues[0].detail


def test_provenance_signed_range_rejects_non_finite_ends():
    for bad in ({'min': float('-inf'), 'max': 0.3}, {'min': -1.95, 'max': float('nan')}):
        assert _prov_issues({'m': _rated(dict(_MOS, Vbs_max=bad), (_SRC, {}))}) \
            == [('m', 'rating_convention')], bad


def test_provenance_equal_rule_compares_both_ends_of_a_signed_range():
    """``= Vbs_max`` claimed by a magnitude key when Vbs_max is the signed
    range (-1.95, +0.3): the magnitudes agree (1.95 == 1.95) but the ranges
    do not, so the claim is a lie the re-check must catch."""
    r = dict(_MOS_SIGNED)
    assert _prov_issues({'m': _rated(r, (_SRC, {'Vbd_max': '= Vbs_max'}))}) \
        == [('m', 'bad_provenance')]
    # two signed keys with the same range DO satisfy the rule
    r2 = dict(_MOS_SIGNED, Vbd_max={'min': -1.95, 'max': 0.3})
    assert _prov_issues({'m': _rated(r2, (_SRC, {'Vbd_max': '= Vbs_max'}))}) == []


def test_provenance_factor_rule_cannot_derive_a_signed_range():
    """``1.1 x nominal`` yields one magnitude; claiming it for an asymmetric
    range is ``bad_provenance`` (the heuristic re-check still runs on the
    new form rather than skipping it)."""
    r = {'Vbs_max': {'min': -1.95, 'max': 0.3}}
    assert _prov_issues({'m': _rated(r, (None, {'Vbs_max': '1.1 x nominal 1.8 V'}))}) \
        == [('m', 'bad_provenance')]


def test_provenance_loaded_pdkdevice_is_judged_on_its_junction_ranges():
    """A real ``PDKDevice`` (loader-built ``junction_ranges``) goes through
    the same checks as the fakes above, which carry ``junction_ratings``
    only and are expanded through ``parse_junction_rating``."""
    from silk.pdk.basepdk import PDKDevice
    from silk.pdk.validate import check_junction_provenance
    dev = PDKDevice.from_dict('m', {
        'PDKDevice': 'x', 'paramList': {}, 'pinList': {'d': 'd', 'g': 'g', 's': 's', 'b': 'b'},
        'junction_ratings': {'Vgs_max': 1.95, 'Vbs_max': {'min': 0.3, 'max': -1.95}},
        'junction_ratings_provenance': {'source': _SRC},
    })
    dev.spice_prefix = 'XM'
    issues = check_junction_provenance(_FakePDK({'m': (dev, 'XM')}))
    assert [(i.device, i.kind) for i in issues] == [('m', 'rating_convention')]
    assert 'Vbs_max=[+0.3, -1.95]' in issues[0].detail


def test_provenance_unrated_device_is_fine_but_stale_block_is_not():
    """Passives carry no ratings and no block — silent. A block with nothing
    to describe can only be stale, so it is flagged."""
    assert _prov_issues({'r': _rated({}, None)}) == []
    assert _prov_issues({'r': _rated({}, ('doc', {}))}) == [('r', 'bad_provenance')]


def test_provenance_check_runs_inside_the_static_pass(tmp_path, monkeypatch):
    """validate_pdk (and so the CLI and test_real_pdk_has_no_validation_issues)
    reports an unsourced rating without --simulate and without a lib scan."""
    from silk.pdk import validate as V
    monkeypatch.setattr(V, 'scan_lib_definitions', lambda root: {'sub4': {('subckt', 4)}})
    pdk = _FakePDK({'m': (_rated(_MOS, None), 'XM')})
    issues = V.validate_pdk(pdk, lib_root=str(tmp_path))
    assert [(i.device, i.kind) for i in issues] == [('m', 'unsourced_rating')]


@pytest.mark.slow
@needs_pdk
@needs_ngspice
def test_simulate_devices_reads_run_log_after_returning_data(monkeypatch):
    from silk import load_pdk
    from silk.pdk import validate as V

    pdk = load_pdk('sky130')
    pdk._devices = {'res_generic_nd': pdk._devices['res_generic_nd']}

    class _DiagnosticModule(V.Module):
        def simulate(self, *args, **kwargs):
            result = super().simulate(*args, **kwargs)
            assert Path(self.results_file).is_file()
            assert Path(self._run_files.logfile_path).parent != Path(self.dir_folder)
            with open(self._run_files.logfile_path, 'a') as log:
                log.write('\nunknown subckt frobnicator\n')
            return result

    monkeypatch.setattr(V, 'Module', _DiagnosticModule)
    issues = V._simulate_devices(pdk)
    assert [(i.device, i.kind) for i in issues] == [('res_generic_nd', 'sim_failed')]
    assert 'unknown subckt frobnicator' in issues[0].detail
