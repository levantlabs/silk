import glob
import os

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
_EXAMPLES_GLOB = os.path.join(_REPO_ROOT, 'silk', 'examples', '*.py')


def _example_files():
    files = glob.glob(_EXAMPLES_GLOB)
    assert files, f'no example files matched {_EXAMPLES_GLOB!r} — glob is broken'
    return files


def test_no_example_hardcodes_the_raw_lib():
    hits = [f for f in _example_files()
            if 'skywater-pdk/libraries/sky130_fd_pr' in open(f).read()]
    assert hits == [], hits


def test_no_example_hardcodes_the_raw_lib_fragment():
    """Catch a raw-lib literal even when split/line-wrapped across two
    adjacent string literals (e.g. an f-string concatenated with a plain
    string on the next line), which defeats a contiguous-substring search
    for the full path. The fragment 'skywater-pdk/libraries' always stays
    on one line because it's the start of the path, before any wrap point.
    """
    hits = [f for f in _example_files()
            if 'skywater-pdk/libraries' in open(f).read()]
    assert hits == [], hits


def test_no_example_hardcodes_the_volare_lib_path():
    """Examples must not hardcode the volare lib path (`libs.tech/ngspice/
    sky130.lib.spice`) either — that's the registry's job via `load_pdk`.
    (Build-only examples that deliberately use a `/fake/…` placeholder lib
    with the direct constructor are fine — they don't hit this fragment.)
    """
    hits = [f for f in _example_files()
            if 'libs.tech/ngspice' in open(f).read()]
    assert hits == [], hits
