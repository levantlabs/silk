"""``BasePDK`` is the sole reader and interpreter of the device catalog and of units.

A second unit interpreter in ``silk/layout/area_model.py`` previously caused
a 1e12 area error in the flagship walkthrough. These checks run on every
push so a third
interpreter cannot appear silently (architecture.md invariant I11):

* the unit-interpretation identifiers (``_SI_TO_PDK_SCALE``,
  ``_QUANTITY_EXPONENT``, ``_pdk_unit_to_um``, ``_CANONICAL_LENGTH_PREFIX``)
  are referenced by no Python file outside ``silk/pdk/``;
* the device-catalog file (``*_device_details.yaml``) is opened by no Python
  file outside ``silk/pdk/`` except the recorded, justified consumers listed in
  the audit. Examples that merely *pass* the catalog path to ``BasePDK``
  are on the API and are excluded by directory.

Extending either allowlist is a same-PR change to the audit document and to
``docs/architecture.md`` §3.
"""
import re
from pathlib import Path

import pytest

from tests._silk_sources import silk_py_files

_REPO = Path(__file__).resolve().parents[1]
_SILK = _REPO / 'silk'
_PDK_DIR = _SILK / 'pdk'

_UNIT_INTERPRETATION_IDENTIFIERS = (
    '_SI_TO_PDK_SCALE',
    '_QUANTITY_EXPONENT',
    '_pdk_unit_to_um',
    '_CANONICAL_LENGTH_PREFIX',
)

# The last exception was schema.py's CLI help path; TASK-550 replaced it
# with registered-name lookup and a generic custom YAML path example.
_CATALOG_READ_ALLOWLIST = set()

_CATALOG_FILENAME = re.compile(r'_device_details\.yaml')


def _silk_py_files_outside_pdk():
    files = [p for p in silk_py_files()
         if _PDK_DIR not in p.parents and 'examples' not in p.parts]
    assert files, f'no python files found under {_SILK} — glob is broken'
    return files


@pytest.mark.parametrize('path', _silk_py_files_outside_pdk(),
                         ids=lambda p: str(p.relative_to(_REPO)))
def test_no_unit_interpretation_outside_silk_pdk(path):
    text = path.read_text()
    offenders = [ident for ident in _UNIT_INTERPRETATION_IDENTIFIERS if ident in text]
    assert not offenders, (
        f'{path.relative_to(_REPO)} references {offenders}: unit interpretation '
        f'belongs in silk/pdk/basepdk.py only. Call device_geometry_um or read '
        f'the pdk_unit tag BasePDK already resolved (docs/architecture.md I11).')


def test_catalog_file_reads_outside_silk_pdk_are_the_recorded_ones():
    readers = sorted(
        str(p.relative_to(_REPO)) for p in _silk_py_files_outside_pdk()
        if _CATALOG_FILENAME.search(p.read_text()))
    unrecorded = [r for r in readers if r not in _CATALOG_READ_ALLOWLIST]
    assert not unrecorded, (
        f'device catalog referenced outside silk/pdk/ by {unrecorded}. Either read '
        f'through BasePDK / load_pdk, or record the consumer with a justification '
        f'alongside the allowlist here and add it with that justification.')
    stale = sorted(_CATALOG_READ_ALLOWLIST - set(readers))
    assert not stale, (
        f'allowlist entries no longer read the catalog: {stale} — remove them so '
        f'the allowlist stays honest.')
