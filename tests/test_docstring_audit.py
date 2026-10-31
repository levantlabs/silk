"""Deterministic checks of ``scripts/docstring_audit.py`` (TASK-305).

The audit's counts are only worth reading if its definitions hold, so each
check is pinned here against a small fixture package written to ``tmp_path``
and registered as ``silk._audit_fixture`` (the audit treats only ``silk.*``
objects as silk's). The last tests pin the real manifest's surface: it must
reach the APIs the first version missed, and must not shrink.

Fast suite: no PDK models, no simulator.
"""
import importlib.util
import subprocess
import sys
import textwrap

import pytest

from scripts import docstring_audit as audit

FIXTURE = '''
"""Fixture package for the docstring audit tests.

>>> 1 + 1
2
"""
import dataclasses
from typing import Literal

__all__ = ['helper_raises', 'returns_none', 'returns_value', 'zero_arg',
           'other_params', 'sphinx_params', 'bad_examples', 'console_sessions',
           'stale_imports', 'Public', 'Alias',
           'Data', 'UNIT', 'Kind', 'LIB']

Kind = Literal['a', 'b']

#: Length unit of the fixture.
UNIT = 'u'


def helper_raises(x):
    """Call a helper with x."""
    def inner():
        raise ValueError('only the nested helper raises')
        return 3
    inner()


def returns_none(x):
    """Do something with x."""
    if x:
        return None
    return


def returns_value(x):
    """Do something with x."""
    return x + 1


def zero_arg():
    """Do nothing.

    Parameters
    ----------
    ghost : int
        Not a parameter.
    """


def other_params(a):
    """Use a.

    Other Parameters
    ----------------
    b : int
        Not a parameter either.
    """


def sphinx_params(a):
    """Use a.

    :param c: not a parameter
    """


def bad_examples():
    """Hold broken examples.

    .. code-block:: python

        return 1

    ```python
    await thing()
    ```

    The integral::

        y = ∫ x dt

    A Python literal::

        z = 1
    """


def console_sessions():
    """Hold console sessions, which are doctests, not programs.

    .. code-block:: pycon

        >>> 1+1 / 2
        1.5

    ```pycon
    >>> 1+1 / 2
    1.5
    ```
    """


def stale_imports():
    """Import a missing symbol from a plain module.

    .. code-block:: python

        from silk.signals.base import AuditMissingSymbol
        from silk.signals import base
        from silk import metrics
    """


class _Mixin:
    def mixed(self):
        """Mixin method."""
        return 1


class Public(_Mixin):
    """Public class."""

    def __init__(self):
        self.y = 1  #: the y value in volts

    @property
    def p(self):
        """The p."""
        return 2


Alias = Public


@dataclasses.dataclass
class Data:
    """Data holder."""

    documented: int = 0
    """The documented field."""
    #: Documented by a comment.
    hashed: int = 0
    bare: int = 0


class _Library:
    def __init__(self):
        self.made = type('made', (), {'__module__': __name__})


LIB = _Library()
'''

DOCTESTS = '''
"""Module docstring.

>>> 2 * 3
6
"""


def mixed():
    """One passing statement and one skipped.

    >>> 1 + 1
    2
    >>> 1 / 0  # doctest: +SKIP
    """


def skip_only():
    """Skipped.

    >>> 1 / 0  # doctest: +SKIP
    """


def broken():
    """Wrong.

    >>> 1 + 1
    3
    """


def good():
    """Right.

    >>> 1 + 1
    2
    """
'''

NAME = 'silk._audit_fixture'


@pytest.fixture
def rows(tmp_path):
    path = tmp_path / 'fixture.py'
    path.write_text(textwrap.dedent(FIXTURE), encoding='utf-8')
    spec = importlib.util.spec_from_file_location(NAME, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = mod
    try:
        spec.loader.exec_module(mod)
        manifest = {
            'packages': [NAME],
            'attributes': {'Public': ['y']},
            'factories': [{'group': 'fac', 'label': 'lib', 'object': f'{NAME}.LIB'}],
        }
        yield {r.name: r for r in map(audit.score, audit.collect(manifest))}
    finally:
        sys.modules.pop(NAME, None)


def test_collection_covers_aliases_mixins_fields_attributes_factories(rows):
    kinds = {name: r.kind for name, r in rows.items()}
    assert kinds[NAME] == 'module'
    assert kinds[f'{NAME}.Alias'] == 'alias'
    assert 'alias of' in rows[f'{NAME}.Alias'].notes[0]
    assert kinds['Public.mixed'] == 'method'       # defined on a private mixin
    assert kinds['Public.p'] == 'property'
    assert kinds['Public.y'] == 'attribute'
    assert {kinds[f'Data.{f}'] for f in ('documented', 'hashed', 'bare')} == {'field'}
    assert kinds['lib.made'] == 'factory'
    assert kinds[f'{NAME}.UNIT'] == kinds[f'{NAME}.Kind'] == 'constant'
    assert kinds[f'{NAME}.LIB'] == 'constant'


def test_nested_helper_raise_and_return_are_not_the_functions_own(rows):
    r = rows[f'{NAME}.helper_raises'].results
    assert r['failure'] is None and r['returns'] is None


def test_explicit_return_none_is_not_a_return_value(rows):
    assert rows[f'{NAME}.returns_none'].results['returns'] is None
    assert rows[f'{NAME}.returns_value'].results['returns'] is False


@pytest.mark.parametrize('func, ghost', [('zero_arg', 'ghost'),
                                         ('other_params', 'b'),
                                         ('sphinx_params', 'c')])
def test_ghost_parameters_are_stale_in_every_style(rows, func, ghost):
    row = rows[f'{NAME}.{func}']
    assert row.results['stale'] is False
    assert f'param {ghost} not in signature' in row.notes


def test_constants_need_their_own_documentation(rows):
    assert rows[f'{NAME}.UNIT'].results['present'] is True     # #: comment
    assert rows[f'{NAME}.Kind'].results['present'] is False    # not str.__doc__


def test_fields_and_attributes_documented_where_autodoc_reads(rows):
    assert rows['Data.documented'].results['present'] is True
    assert rows['Data.hashed'].results['present'] is True
    assert rows['Data.bare'].results['present'] is False
    assert rows['Public.y'].results['present'] is True


def test_property_returns_is_unknown_not_not_applicable(rows):
    assert rows['Public.p'].results['returns'] == audit.UNKNOWN


def test_examples_are_classified_by_language_and_compiled(rows):
    blocks = [(b['kind'], b['lang'], b['status'].split(':')[0])
              for b in rows[f'{NAME}.bad_examples'].examples]
    assert ('code-block', 'python', 'compile') in blocks     # top-level return
    assert ('fence', 'python', 'compile') in blocks          # top-level await
    assert ('literal', 'none', 'not python') in blocks       # a formula
    assert ('literal', 'none', 'compiles as python') in blocks


@pytest.mark.parametrize('res, status', [
    # doctest's attempted excludes skipped examples.
    ({'attempted': 0, 'failed': 0, 'skipped': 1}, 'skipped'),
    ({'attempted': 1, 'failed': 0, 'skipped': 1}, 'passed'),
    ({'attempted': 2, 'failed': 1, 'skipped': 0}, 'failed 1/2'),
    ({'attempted': 0, 'failed': 0, 'skipped': 0}, 'not run'),
])
def test_doctest_outcomes_are_not_conflated(res, status):
    assert audit.classify_run(res) == status


def test_run_examples_executes_in_a_fresh_process(tmp_path, monkeypatch):
    (tmp_path / 'audit_doctest_fixture.py').write_text(DOCTESTS, encoding='utf-8')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    doctest_block = [{'kind': 'doctest', 'lang': 'python', 'status': 'ok'}]
    rows = [audit.Row('g', n, 'function', examples=doctest_block,
                      target=('audit_doctest_fixture', n))
            for n in ('mixed', 'skip_only', 'broken', 'good')]
    rows.append(audit.Row('g', 'audit_doctest_fixture', 'module', examples=doctest_block,
                          target=('audit_doctest_fixture', '')))
    results = audit.run_examples(rows)
    out = {name: (status, text) for name, status, text in results}
    assert [n for n, _, _ in results].count('audit_doctest_fixture') == 1
    assert out['audit_doctest_fixture'][0] == 'passed'
    assert out['mixed'][0] == 'passed'
    assert out['skip_only'][0] == 'skipped'
    assert out['good'][0] == 'passed'
    assert out['broken'][0] == 'failed 1/1' and 'Expected' in out['broken'][1]
    assert 'audit_doctest_fixture' not in sys.modules      # never imported here


def test_module_docstrings_are_execution_targets(rows):
    module_rows = [r for r in rows.values() if r.kind == 'module']
    assert [r.target for r in module_rows] == [(NAME, '')]
    assert [b['kind'] for b in module_rows[0].examples] == ['doctest']


def test_nonzero_worker_exit_is_not_run_even_with_passing_json(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 7, stdout='{"attempted": 1, "failed": 0, "skipped": 0, "text": ""}\n',
            stderr='child crash')
    monkeypatch.setattr(audit.subprocess, 'run', fake_run)
    row = audit.Row('g', 'x', 'function',
                    examples=[{'kind': 'doctest', 'lang': 'python', 'status': 'ok'}],
                    target=('somewhere', 'x'))
    [(name, status, text)] = audit.run_examples([row])
    assert status == 'not run'
    assert 'exited 7' in text and 'child crash' in text


def test_console_sessions_are_doctests_only(rows):
    blocks = rows[f'{NAME}.console_sessions'].examples
    assert [(b['kind'], b['status']) for b in blocks] == [('doctest', 'ok')]


def test_stale_import_from_a_plain_module_is_reported_not_raised(rows):
    [block] = rows[f'{NAME}.stale_imports'].examples
    assert block['status'] == 'import: silk.signals.base.AuditMissingSymbol'


def test_real_manifest_reaches_the_documented_surface():
    names = {e.name: e.kind for e in audit.collect(audit.load_manifest())}
    for name in ('silk.sources.VoltageSource', 'silk.sources.CurrentSource',
                 'Module.add_constraint', 'Spec.target',
                 'silk.constraints.SymmetricBlocks'):
        assert name in names, name
    # Instance attributes the reference promises (TASK-305 re-review).
    for name in ('Module.results_file', 'Signal.x', 'Signal.y', 'Signal.name',
                 'Signal.type', 'Signal.xlabel', 'Signal.ylabel'):
        assert names.get(name) == 'attribute', name
    assert names['Spec.target'] == 'field'
    assert any(n.startswith('Measurement.') and k == 'field' for n, k in names.items())
    assert any(k == 'factory' for k in names.values())
    # A floor against shrinking back to the first audit's 718 rows. It does not
    # prove coverage of the existing reference; that is Phase 2 A1's gate.
    assert len(names) > 1400


def test_removed_names_prefers_development_history_with_public_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'REPO', tmp_path)
    public = tmp_path / 'CHANGELOG.md'
    public.write_text('- **BREAKING:** `old_public_name` is removed.\n')
    assert audit.removed_names() == ['old_public_name']
    development = tmp_path / 'CHANGELOG-dev.md'
    development.write_text('- **BREAKING:** `old_development_name` is removed.\n')
    assert audit.removed_names() == ['old_development_name']
    assert audit.removed_names(public) == ['old_public_name']
