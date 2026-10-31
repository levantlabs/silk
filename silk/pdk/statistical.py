"""Statistical (Monte Carlo) facts a PDK manifest declares.

silk hardcodes none of these: which library section gives mismatch at each
corner, the process section and the corners it is valid at, the switch
names, the include files the both-variation wrapper pulls in, and which model
parameters to record per device all come from the manifest's
``library.statistical:`` block.
"""
import os
import re
import threading
from dataclasses import dataclass
from typing import Mapping

from silk.exceptions import MissingLibrarySectionError, PDKError


@dataclass(frozen=True)
class StatisticalModels:
    mismatch_sections: Mapping[str, str]    # corner -> library section
    mismatch_switch: str
    record: Mapping[str, tuple]             # silk device name -> model params
    process_section: str
    process_corners: tuple
    process_switch: str
    process_includes: tuple                 # absolute paths
    min_open_pdks: str = None               # message hint only

    def required_sections(self):
        """Every library section the block relies on, in a stable order."""
        out = list(dict.fromkeys(self.mismatch_sections.values()))
        if self.process_section not in out:
            out.append(self.process_section)
        return out


def parse_statistical(block, root, pdk_name='?'):
    """Build StatisticalModels from a manifest ``statistical:`` mapping.

    Relative include paths resolve against *root*. Raises PDKError naming the
    missing key when the block is malformed.
    """
    try:
        mm, pr = block['mismatch'], block['process']
        return StatisticalModels(
            mismatch_sections=dict(mm['sections']),
            mismatch_switch=mm['switch'],
            record={k: tuple(v) for k, v in (mm.get('record') or {}).items()},
            process_section=pr['section'],
            process_corners=tuple(pr['corners']),
            process_switch=pr['switch'],
            process_includes=tuple(
                p if os.path.isabs(p) else os.path.join(root, p)
                for p in pr.get('includes', [])),
            min_open_pdks=block.get('min_open_pdks'))
    except (KeyError, TypeError) as exc:
        raise PDKError(
            f'PDK {pdk_name!r}: malformed library.statistical block '
            f'({type(exc).__name__}: {exc})') from None


_LIB_LINE = re.compile(r'^\s*\.lib\s+(\S+)\s*$', re.IGNORECASE)
_cache = {}
_cache_lock = threading.Lock()


def library_sections(path):
    """Section names declared by ``.lib <name>`` lines, lower-cased.

    A ``.lib "<file>" <section>`` include has a second token and is not a
    declaration. Cached by (path, mtime).
    """
    key = (path, os.stat(path).st_mtime_ns)
    with _cache_lock:
        hit = _cache.get(key)
    if hit is not None:
        return hit
    names = set()
    with open(path, errors='replace') as f:
        for line in f:
            m = _LIB_LINE.match(line)
            if m:
                names.add(m.group(1).lower())
    result = frozenset(names)
    with _cache_lock:
        for k in [k for k in _cache if k[0] == path]:
            del _cache[k]
        _cache[key] = result
    return result


def missing_sections(path, sections):
    """The *sections* absent from the library at *path*, in order."""
    have = library_sections(path)
    return [s for s in sections if s.lower() not in have]


def require_library_sections(pdk, sections, *, simulator='ngspice'):
    """Raise MissingLibrarySectionError naming each missing section."""
    path = (pdk.xyce_library_location
            if simulator == 'xyce' and pdk.xyce_library_location
            else pdk.library_location)
    if not os.path.isfile(path):
        raise MissingLibrarySectionError(
            f'cannot check library sections {list(sections)}: model library '
            f'{path!r} does not exist (see docs/installing-the-pdk.md)')
    missing = missing_sections(path, sections)
    if not missing:
        return
    st = getattr(pdk, 'statistical', None)
    hint = (f' (needs open_pdks {st.min_open_pdks} or newer)'
            if st is not None and st.min_open_pdks else '')
    raise MissingLibrarySectionError(
        f'model library {path!r} has no {", ".join(repr(s) for s in missing)} '
        f'section{"s" if len(missing) > 1 else ""}; the installed open_pdks '
        f'predates statistical sections{hint}. Update the PDK install '
        f'(docs/installing-the-pdk.md) or pass library_location= to load_pdk().')
