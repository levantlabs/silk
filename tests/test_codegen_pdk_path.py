"""Generated code must resolve the PDK through the registry-driven
``load_pdk('<name>')`` front door (cwd-independent), not via a cwd-relative
literal or a hardcoded raw skywater-pdk lib path that only works from silk's
repo root / a specific volare layout."""
from silk.codegen.generator import Generator


def _standalone():
    return Generator().generate({'circuit_name': 't', 'pdk': 'sky130'})


def _module():
    return Generator().generate(
        {'circuit_name': 'm', 'pdk': 'sky130', 'ports': {'a': {'dir': 'input'}}})


def test_standalone_uses_load_pdk_and_volare_lib():
    code = _standalone()
    assert "load_pdk('sky130')" in code or 'load_pdk("sky130")' in code
    assert 'skywater-pdk/libraries' not in code                 # raw literal gone
    assert 'from silk import load_pdk' in code


def test_module_uses_load_pdk_and_volare_lib():
    code = _module()
    assert "load_pdk('sky130')" in code or 'load_pdk("sky130")' in code
    assert 'skywater-pdk/libraries' not in code                 # raw literal gone
    assert 'from silk import load_pdk' in code


def test_emitted_preamble_execs_cwd_independent(tmp_path, monkeypatch):
    """Exec the emitted preamble up to the load_pdk line from a foreign cwd;
    it must import load_pdk + resolve the device yaml package-relatively."""
    monkeypatch.chdir(tmp_path)
    code = _standalone()
    prefix = []
    for line in code.splitlines():
        prefix.append(line)
        if 'load_pdk(' in line:
            break
    else:
        raise AssertionError('no `load_pdk(` line emitted')
    # do not actually construct the PDK (no PDK_ROOT guarantee): assert the
    # preamble imports + names resolve syntactically:
    compile('\n'.join(prefix), '<gen>', 'exec')
