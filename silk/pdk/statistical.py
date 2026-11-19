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
    record: Mapping[str, tuple | Mapping]   # device -> MOS params or internal targets
    process_section: str
    process_corners: tuple
    process_switch: str
    process_includes: tuple                 # absolute paths
    min_open_pdks: str = None               # message hint only
    process_warnings: tuple = ()

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
        if not isinstance(pr, Mapping):
            raise PDKError(f'PDK {pdk_name!r}: library.statistical.process must be a mapping')
        entries = pr.get('warnings', [])
        field = 'library.statistical.process.warnings'
        if not isinstance(entries, list):
            raise PDKError(f'PDK {pdk_name!r}: {field} must be a list')
        process_warnings = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping) or set(entry) != {'devices', 'message'}:
                raise PDKError(
                    f'PDK {pdk_name!r}: {field}[{index}] requires devices and message')
            devices, message = entry['devices'], entry['message']
            if (not isinstance(devices, list) or not devices
                    or any(not isinstance(d, str) or not d.strip() for d in devices)
                    or len(set(devices)) != len(devices)
                    or not isinstance(message, str) or not message.strip()):
                raise PDKError(
                    f'PDK {pdk_name!r}: {field}[{index}] requires distinct '
                    'nonempty device names and a nonblank message')
            process_warnings.append((tuple(devices), message))
        record = {}
        from silk.spice_names import validate_authored_name
        for device, value in (mm.get('record') or {}).items():
            field = f'library.statistical.mismatch.record.{device}'
            if not isinstance(value, Mapping):
                record[device] = tuple(value)
                continue
            if not value:
                raise PDKError(f'{field}: internal-target mapping must not be empty')
            targets, addresses = {}, set()
            for target, entry in value.items():
                location = f'{field}.{target}'
                if not isinstance(entry, Mapping) or set(entry) != {'params', 'xyce_model'}:
                    raise PDKError(f'{location}: requires exactly params and xyce_model')
                params, model = entry['params'], entry['xyce_model']
                if not isinstance(params, (list, tuple)) or not params:
                    raise PDKError(f'{location}.params: requires nonempty distinct parameter tokens')
                for label, token in [('target', target), ('xyce_model', model),
                                     *[('params', param) for param in params]]:
                    try:
                        validate_authored_name(token, kind='instance', who=label)
                        if '[' in token:
                            raise ValueError('literal single-component token required')
                    except ValueError as exc:
                        raise PDKError(f'{location}.{label}: {exc}') from None
                if target.lower() in {t.lower() for t in targets}:
                    raise PDKError(f'{location}: duplicate target')
                if len({p.lower() for p in params}) != len(params):
                    raise PDKError(f'{location}.params: duplicate parameter')
                for param in params:
                    address = (model.lower(), param.lower())
                    if address in addresses:
                        raise PDKError(f'{location}.xyce_model: duplicate model/parameter address {address}')
                    addresses.add(address)
                targets[target] = {'params': tuple(params), 'xyce_model': model}
            record[device] = targets
        return StatisticalModels(
            mismatch_sections=dict(mm['sections']),
            mismatch_switch=mm['switch'],
            record=record,
            process_section=pr['section'],
            process_corners=tuple(pr['corners']),
            process_switch=pr['switch'],
            process_includes=tuple(
                p if os.path.isabs(p) else os.path.join(root, p)
                for p in pr.get('includes', [])),
            min_open_pdks=block.get('min_open_pdks'),
            process_warnings=tuple(process_warnings))
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
