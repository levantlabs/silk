"""Published documentation describes behaviour without internal process history."""
import ast
import io
import re
import subprocess
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INTERNAL_REFERENCE = re.compile(r'(?i:superpowers/|\bdocs/superpowers\b(?![\w-]))')
RECORDS = ('docs/archive/', 'docs/superpowers/', 'docs/design/', 'docs/backlog/')
PROCESS = re.compile(
    r"(?i:\btask[- ]\d+(?:\.\d+)*\b|\bdoc-1\b|\breviewers?\b|"
    r"\b(?:internal|code|codex|owner|design)[\s-]+reviews?\b|"
    r"\breviews?\s+(?:found|approved|requested|identified|ruled)\b|"
    r"\brulings?\b|\bdecision-\d+\b|\*\*owner:?\*\*|\bAC\s*#\d+\b|"
    r"\bT-[A-H]\b|\bowner[\s-]+(?:approved|asked|ruled|decided|wants)\b|"
    r"\bowner[\s'’_-]*(?:s[\s_-]*)?decisions?\b|"
    r"\bowner[\s-]+Q\d+\b|\bQ\d+\s+(?:ruling|decision|owner\s+question)\b|"
    r"\bpre-\d{3}\.\d+\b|\b\d{3}\.\d+(?:'s|\s+(?:capture\s+path|verified))\b|"
    r"\bin\s+sweep\s+order\s+\(D\d+\)|"
    r"\b(?:decision|design|spec)\s+(?:§[\d.]+\s*)?\(D\d+\))"
)



def python_documentation(source):
    """Yield source line numbers and text for docstrings and #: comments."""
    lines = source.splitlines()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        if not node.body or not isinstance(node.body[0], ast.Expr):
            continue
        value = node.body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            for number in range(value.lineno, value.end_lineno + 1):
                line = lines[number - 1].encode('utf-8')
                start = value.col_offset if number == value.lineno else 0
                end = value.end_col_offset if number == value.end_lineno else len(line)
                yield number, line[start:end].decode('utf-8')
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT and token.string.startswith('#:'):
            yield token.start[0], token.string


def violations(path, source):
    """Name every offending file and line, without importing circuit code."""
    lines = list(python_documentation(source) if path.suffix == '.py'
                 else enumerate(source.splitlines(), 1))
    text = '\n'.join(line for _, line in lines)
    indices = {text.count('\n', 0, match.start())
               for match in PROCESS.finditer(text)} if in_scope(path) else set()
    if internal_reference_scope(path):
        indices.update(text.count('\n', 0, match.start())
                       for match in INTERNAL_REFERENCE.finditer(text))
    return [f'{path}:{lines[i][0]}: {lines[i][1].strip()}'
            for i in sorted(indices)]


def in_scope(path):
    name = path.as_posix()
    return (name in ('CHANGELOG.md', 'CHANGELOG-dev.md') or name.startswith('silk/') and path.suffix == '.py'
            or name.startswith('docs/') and not name.startswith(RECORDS)
            and name != 'docs/release-gating.md')


def internal_reference_scope(path):
    name = path.as_posix()
    return (name in ('README.md', 'CONTRIBUTING.md', 'CHANGELOG.md', 'CHANGELOG-dev.md')
            or name.startswith('silk/') and path.suffix == '.py'
            or name.startswith('docs/') and path.suffix == '.md'
            and not name.startswith(('docs/archive/', 'docs/superpowers/',
                                     'docs/backlog/')))


def test_published_documentation_has_no_process_history():
    tracked = subprocess.check_output(
        ['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    found = []
    for name in tracked:
        path = Path(name)
        if not name or not (in_scope(path) or internal_reference_scope(path)):
            continue
        try:
            source = (ROOT / path).read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue
        found.extend(violations(path, source))
    assert not found, 'Internal process references:\n' + '\n'.join(found)


@pytest.mark.parametrize('text', [
    'TASK-123', 'task-123.4', 'Task 16 wrapper migration', 'doc-1 Q34',
    'owner Q34', 'Q34 ruling', 'internal review', 'code review', 'reviewer',
    'review found a defect', 'pre-330.8 capture', 'in sweep order (D7)', 'ruling', 'rulings', 'owner decision',
    "owner's decision", 'owner’s decision', 'OWNER DECISION', 'owner approved', 'owner asked',
    'owner ruled', 'owner-approved', '**Owner:** task assignee', 'decision-1', 'AC #6', 'T-D',
])
def test_process_references_are_reported_with_file_and_line(text):
    assert violations(Path('docs/example.md'), '\n' + text) == [
        f'docs/example.md:2: {text}']


def test_python_scanner_reads_only_documentation():
    source = ('# TASK-1 ordinary implementation comment\n'
              'value = "TASK-2"\n'
              '#: TASK-3 published attribute documentation\n'
              'async def f():\n'
              '    """TASK-4 published function documentation."""\n'
              '    return "TASK-5"\n')
    assert set(violations(Path('silk/example.py'), source)) == {
        'silk/example.py:3: #: TASK-3 published attribute documentation',
        'silk/example.py:5: """TASK-4 published function documentation."""',
    }


@pytest.mark.parametrize('name', [
    'docs/archive/record.md', 'docs/superpowers/specs/record.md',
    'docs/design/record.md', 'docs/release-gating.md', 'docs/backlog/note.md',
])
def test_only_named_historical_records_are_excluded(name):
    assert not in_scope(Path(name))


@pytest.mark.parametrize('name', [
    'docs/reference/conf.py', 'docs/example.txt',
    'silk/example.py', 'CHANGELOG.md', 'CHANGELOG-dev.md',
])
def test_other_documentation_remains_in_scope(name):
    assert in_scope(Path(name))


def test_current_contract_path_is_not_process_prose():
    assert not violations(Path('docs/example.md'),
                          'The protocol contract is in docs/reviews/agreement.md.')
    assert violations(Path('docs/example.md'),
                      'The reviewer approved docs/reviews/agreement.md.')


def test_owner_decision_wrapped_across_lines_is_reported():
    assert violations(Path('docs/example.md'), "The owner's\ndecision was recorded.") == [
        "docs/example.md:1: The owner's"]


@pytest.mark.parametrize('text', [
    'Review the convergence log.', 'The Q factor is 3.16.',
    'The quality factors Q1 and Q2 differ.', 'Diode-connected PNPs Q1 and Q2.',
    'The D1 and D7 signal labels are legal.',
])
def test_technical_prose_is_accepted(text):
    assert not violations(Path('docs/example.md'), text)


@pytest.mark.parametrize('name', [
    'README.md', 'CONTRIBUTING.md', 'CHANGELOG.md', 'CHANGELOG-dev.md', 'docs/example.md',
    'docs/README.md', 'docs/release-gating.md', 'docs/design/guide.md',
])
@pytest.mark.parametrize('text', [
    '[design](superpowers/specs/example.md)',
    '`docs/superpowers/plans/example.md`', 'See ../superpowers/specs/example.md.',
    'SUPERPOWERS/specs/example.md',
    '[internal](docs/superpowers)', '`docs/superpowers`',
])
def test_internal_references_in_public_markdown_are_rejected(name, text):
    assert violations(Path(name), '\n' + text) == [f'{name}:2: {text}']


def test_internal_reference_scanner_reads_only_python_documentation():
    source = ('"""See docs/superpowers/specs/example.md."""\n'
              '#: See superpowers/plans/example.md\n'
              '# See docs/superpowers/specs/example.md\n'
              'value = "docs/superpowers/specs/example.md"\n'
              'def f():\n'
              '    """See superpowers/specs/example.md."""\n')
    assert set(violations(Path('silk/example.py'), source)) == {
        'silk/example.py:1: """See docs/superpowers/specs/example.md."""',
        'silk/example.py:2: #: See superpowers/plans/example.md',
        'silk/example.py:6: """See superpowers/specs/example.md."""',
    }


@pytest.mark.parametrize('source', [
    '"""Public behavior."""  # docs/superpowers/specs/example.md',
    '"""Public behavior."""; value = "docs/superpowers/specs/example.md"',
])
def test_same_line_implementation_references_are_excluded(source):
    assert not violations(Path('silk/example.py'), source)


@pytest.mark.parametrize('source', [
    '"""See docs/superpowers."""', '#: See docs/superpowers',
])
def test_bare_internal_directory_in_python_documentation_is_rejected(source):
    assert violations(Path('silk/example.py'), source)


@pytest.mark.parametrize('name', [
    'docs/archive/record.md', 'docs/superpowers/specs/record.md',
    'docs/backlog/note.md', 'backlog/tasks/record.md',
    'silk/pdk/data/catalog.yaml', 'docs/example.txt',
])
def test_internal_reference_guard_preserves_excluded_records(name):
    assert not violations(Path(name), 'See docs/superpowers/specs/example.md.')


@pytest.mark.parametrize('text', [
    'See docs/architecture.md.', 'The viewer encodes typed arrays.',
    'Public documentation describes behavior.',
    'See docs/superpowers_extra.', 'See docs/superpowers-extra.',
    'See docs/superpowersful.',
])
def test_public_references_and_behavior_are_accepted(text):
    assert not violations(Path('docs/example.md'), text)
