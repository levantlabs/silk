"""Planted-failure checks for local public release tooling."""

import json

import os

from pathlib import Path

import subprocess

import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]

def tool(name, *args, env=None):
    return subprocess.run([sys.executable, str(ROOT / 'scripts' / name), *map(str, args)],
                          env=env, capture_output=True, text=True)

def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()

@pytest.fixture
def repo(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    git(root, 'init', '-q')
    git(root, 'config', 'user.name', 'Example Maintainer')
    git(root, 'config', 'user.email', 'maintainer@example.invalid')
    (root / 'public.txt').write_text('committed content\n')
    (root / 'private.txt').write_text('withheld content\n')
    (root / 'MANIFEST.public').write_text(json.dumps({
        'version': 1, 'include': ['public.txt'],
        'exclude': ['private.txt', 'MANIFEST.public']}))
    git(root, 'add', 'public.txt', 'private.txt', 'MANIFEST.public')
    git(root, 'commit', '-qm', 'Fixture')
    return root

def scan(root, names='syntheticmarker', *args):
    env = dict(os.environ, RELEASE_PRIVATE_NAMES=names,
               RELEASE_AUTHOR_NAME='Example Maintainer', RELEASE_AUTHOR_EMAIL='maintainer@example.invalid')
    return tool('scan_release.py', root, *args, env=env)

def test_scan_unarmed_and_explicit_empty(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != 'RELEASE_PRIVATE_NAMES'}
    # Disable ambient config discovery using an isolated script copy.
    script = tmp_path / 'scripts' / 'scan_release.py'
    script.parent.mkdir()
    original = ROOT / 'scripts' / 'scan_release.py'
    assert original.exists(), 'scanner has not been implemented'
    script.write_bytes(original.read_bytes())
    empty = tmp_path / 'empty'
    empty.mkdir()
    result = subprocess.run([sys.executable, str(script), str(empty)], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 3
    assert 'UNARMED' in result.stdout
    assert scan(empty, '').returncode == 0

@pytest.mark.parametrize('where', ['fixture', 'filename', 'changelog', 'development-changelog', 'binary', 'png'])
def test_scan_finds_case_insensitive_marker_everywhere(tmp_path, where):
    text = 'SyNtHeTiCmArKeR'
    if where == 'filename':
        (tmp_path / (text + '.bin')).write_bytes(b'\0')
    elif where == 'png':
        from PIL import Image, PngImagePlugin
        meta = PngImagePlugin.PngInfo()
        meta.add_text('Description', text, zip=True)
        Image.new('RGB', (1, 1)).save(tmp_path / 'image.png', pnginfo=meta)
    elif where == 'binary':
        (tmp_path / 'data.bin').write_bytes(b'\0' + text.encode())
    else:
        path = tmp_path / ({'changelog': 'CHANGELOG.md', 'development-changelog': 'CHANGELOG-dev.md'}.get(where, 'tests/fixtures/data.txt'))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    result = scan(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'private-name' in result.stdout

def test_scan_word_boundaries_and_credentials(tmp_path):
    p = tmp_path / 'data.txt'
    p.write_text('presyntheticmarkerpost\n')
    assert scan(tmp_path).returncode == 0
    p.write_text('Bearer ' + 'a' * 24)
    assert scan(tmp_path).returncode == 1


def test_release_identity_does_not_make_provider_urls_private(repo):
    git(repo, 'config', 'user.name', 'Release Custodian')
    email = 'release-account' + '@users.noreply.github.com'
    git(repo, 'config', 'user.email', email)
    git(repo, 'commit', '--amend', '--no-edit', '--reset-author')
    (repo / 'public.txt').write_text('https://github.com/example/project\nhttps://noreply.example.invalid/\n')
    git(repo, 'add', 'public.txt')
    git(repo, 'commit', '-qm', 'Provider URL fixture')
    env = dict(os.environ, RELEASE_PRIVATE_NAMES='',
               RELEASE_AUTHOR_NAME='Release Custodian', RELEASE_AUTHOR_EMAIL=email)
    result = tool('scan_release.py', repo, '--history', env=env)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('name', ['Example Custodian', 'Example-Custodian'])
def test_scan_still_flags_git_name_parts(repo, name):
    git(repo, 'config', 'user.name', name)
    (repo / 'public.txt').write_text('A cUsToDiAn authored this.\n')
    result = scan(repo, '')
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'derived-identity' in result.stdout


def test_scan_still_flags_full_git_email(repo):
    # Reserved domains bypass generic email shapes, but not derived identity.
    email = 'syntheticaccount' + '@example.invalid'
    git(repo, 'config', 'user.email', email)
    (repo / 'public.txt').write_text('syntheticaccount')
    assert scan(repo, '').returncode == 0
    (repo / 'public.txt').write_text(email.upper())
    result = scan(repo, '')
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'derived-identity' in result.stdout


def test_release_identity_still_refuses_mismatched_history_author(repo):
    git(repo, 'config', 'user.name', 'Release Custodian')
    email = 'release-account' + '@users.noreply.github.com'
    git(repo, 'config', 'user.email', email)
    env = dict(os.environ, RELEASE_PRIVATE_NAMES='',
               RELEASE_AUTHOR_NAME='Release Custodian', RELEASE_AUTHOR_EMAIL=email)
    result = tool('scan_release.py', repo, '--history', env=env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'author/committer outside allowlist' in result.stdout

def test_scan_ignored_config_discovered_in_main_worktree(repo, tmp_path):
    (repo / '.gitignore').write_text('.release-remote\n')
    git(repo, 'add', '.gitignore')
    git(repo, 'commit', '-qm', 'Ignore local config')
    (repo / '.release-remote').write_text('RELEASE_PRIVATE_NAMES="syntheticmarker"\n')
    work = tmp_path / 'linked'
    git(repo, 'worktree', 'add', '--detach', str(work))
    (work / 'public.txt').write_text('syntheticmarker')
    env = {k: v for k, v in os.environ.items() if k != 'RELEASE_PRIVATE_NAMES'}
    result = tool('scan_release.py', work, env=env)
    assert result.returncode == 1
    assert 'UNARMED' not in result.stdout

def test_scan_history_and_author_allowlist(repo):
    (repo / 'public.txt').write_text('syntheticmarker')
    git(repo, 'add', 'public.txt')
    git(repo, 'commit', '-qm', 'Planted history')
    (repo / 'public.txt').write_text('clean')
    git(repo, 'add', 'public.txt')
    git(repo, 'commit', '-qm', 'Clean head')
    result = scan(repo, 'syntheticmarker', '--history')
    assert result.returncode == 1
    env = dict(os.environ, RELEASE_PRIVATE_NAMES='', RELEASE_AUTHOR_NAME='Other Maintainer',
               RELEASE_AUTHOR_EMAIL='other@example.invalid')
    result = tool('scan_release.py', repo, '--history', env=env)
    assert result.returncode == 1
    assert 'author' in result.stdout

def test_link_check_angle_brackets_references_images_and_docstrings(tmp_path):
    (tmp_path / 'README.md').write_text('[bad](<docs/missing file.md>)\n![image](missing.png)\n[ref][target]\n[target]: <docs/other missing.md>\n')
    (tmp_path / 'code.py').write_text('"""See docs/absent.md."""')
    result = tool('check_public_links.py', tmp_path)
    assert result.returncode == 1
    assert all(x in result.stdout for x in ['missing file.md', 'missing.png', 'other missing.md', 'absent.md'])
    (tmp_path / 'docs').mkdir()
    for x in ['missing file.md', 'other missing.md', 'absent.md']:
        (tmp_path / 'docs' / x).write_text('present')
    (tmp_path / 'missing.png').write_bytes(b'image')
    assert tool('check_public_links.py', tmp_path).returncode == 0

def test_allowlist_requires_reason_and_no_file_exemptions():
    import runpy
    scanner = runpy.run_path(str(ROOT / 'scripts/scan_release.py'))
    assert all(reason.strip() for _, _, reason in scanner['ALLOWLIST'])
    # Removing the reason must prevent an exception from applying.
    scan_text = scanner['check_text']
    scan_text.__globals__['ALLOWLIST'] = [('custom', 'safe', '')]
    import re
    assert scan_text('fixture', 'safe', [('custom', re.compile('safe'))])

def test_exif_unicode_identity_is_scanned(tmp_path):
    from PIL import Image
    exif = Image.Exif()
    exif[37510] = b'UNICODE\0' + 'SyNtHeTiCmArKeR'.encode('utf-16-le')
    Image.new('RGB', (1, 1)).save(tmp_path / 'photo.jpg', exif=exif)
    assert scan(tmp_path).returncode == 1


def test_reasoned_reserved_addresses_and_generic_identity_shapes(tmp_path):
    p = tmp_path / 'sample.txt'
    p.write_text('maintainer@example.invalid')
    assert scan(tmp_path, '').returncode == 0
    p.write_text('/' + 'home/' + 'syntheticaccount' + '\n' + '192.' + '168.1.5')
    result = scan(tmp_path, '')
    assert result.returncode == 1
    assert 'home-path' in result.stdout and 'private-ipv4' in result.stdout


def test_scanner_rejects_symlink_directories(tmp_path):
    (tmp_path / 'linked').symlink_to(ROOT / 'scripts', target_is_directory=True)
    assert scan(tmp_path, '').returncode == 2


def test_history_validates_annotated_tag_identity(repo):
    git(repo, 'tag', '-a', 'v-test', '-m', 'Release fixture')
    env = dict(os.environ, RELEASE_PRIVATE_NAMES='', RELEASE_AUTHOR_NAME='Example Maintainer',
               RELEASE_AUTHOR_EMAIL='maintainer@example.invalid')
    # Public author headers are allowed, while tag messages are scanned.
    result = tool('scan_release.py', repo, '--history', env=env)
    assert result.returncode == 0, result.stdout
    git(repo, 'tag', '-d', 'v-test')
    git(repo, '-c', 'user.name=Unapproved Custodian', 'tag', '-a', 'v-test', '-m', 'Release fixture')
    assert tool('scan_release.py', repo, '--history', env=env).returncode == 1


def test_link_check_ignores_illustrative_code_links_but_checks_bare_doc_paths(tmp_path):
    p = tmp_path / 'README.md'
    p.write_text('Use `[label](path)` to describe a link.\nSee `docs/' + 'absent.md`.\n')
    result = tool('check_public_links.py', tmp_path)
    assert result.returncode == 1
    assert 'destination path' not in result.stdout
    assert 'absent.md' in result.stdout


def test_scoped_exceptions_preserve_generic_detection():
    import runpy
    scanner = runpy.run_path(str(ROOT / 'scripts/scan_release.py'))
    patterns = [(n, __import__('re').compile(p, __import__('re').I)) for n, p in scanner['PATTERNS']]
    check = scanner['check_text']
    key = 'spice' + '@ConditionState.NONE'
    assert not check('CHANGELOG.md', key, patterns)
    assert not check('CHANGELOG-dev.md', key, patterns)
    assert check('unrelated.txt', key, patterns)
    account = '/' + 'home/runner'
    assert not check('tests/test_ci_workflow_runner.py', account, patterns)
    assert check('unrelated.txt', account, patterns)
    placeholder = '/' + 'home/userX'
    assert not check('tests/test_provenance.py', placeholder, patterns)
    assert check('unrelated.txt', placeholder, patterns)


def test_link_check_importer_pointer_after_code_spans_and_fences(tmp_path):
    text = """An unmatched `delimiter in a preceding paragraph.

```python
pass
```

The `load_pdk('<name>')` / `Module('<name>', …)` it emits from that PDK;
nothing device-specific is built in. See [known issues](known_issues.md)
and the [SPICE→silk importer rework](backlog/2026-07-21-spice-importer-needs-rework.md).

---

## Exceptions

```python
pass
```
"""
    (tmp_path / 'api-reference.md').write_text(text)
    (tmp_path / 'known_issues.md').write_text('Known limitations')
    result = tool('check_public_links.py', tmp_path)
    assert result.returncode == 1
    assert 'backlog/2026-07-21-spice-importer-needs-rework.md' in result.stdout
    (tmp_path / 'backlog').mkdir()
    (tmp_path / 'backlog/2026-07-21-spice-importer-needs-rework.md').write_text('present')
    assert tool('check_public_links.py', tmp_path).returncode == 0


@pytest.mark.parametrize('prefix', ['An unmatched `delimiter.\n\n', '```python\n"`"\n```\n\n', 'A ``code ` span``.\n\n'])
def test_link_check_preserves_links_after_different_backtick_runs(tmp_path, prefix):
    (tmp_path / 'README.md').write_text(prefix + 'See [missing](<missing file.md>).\n\n```python\npass\n```\n')
    result = tool('check_public_links.py', tmp_path)
    assert result.returncode == 1
    assert 'missing file.md' in result.stdout


def test_link_check_manifest_treats_existing_excluded_target_as_missing(tmp_path):
    (tmp_path / 'README.md').write_text('[link](withheld.md)')
    (tmp_path / 'withheld.md').write_text('not shipping')
    manifest = tmp_path / 'policy.json'
    manifest.write_text(json.dumps({'include': ['README.md'], 'exclude': ['withheld.md']}))
    result = tool('check_public_links.py', tmp_path, '--manifest', manifest)
    assert result.returncode == 1
    assert 'withheld.md' in result.stdout


@pytest.mark.parametrize('text', ['```inline```\n\n[missing](absent.md)\n', r'Escaped \`literal [missing](absent.md) and closing `.' + '\n'])
def test_link_check_does_not_hide_links_after_invalid_or_escaped_markers(tmp_path, text):
    (tmp_path / 'README.md').write_text(text)
    result = tool('check_public_links.py', tmp_path)
    assert result.returncode == 1
    assert 'absent.md' in result.stdout


def test_link_check_manifest_refuses_missing_shipping_file(tmp_path):
    manifest = tmp_path / 'policy.json'
    manifest.write_text(json.dumps({'include': ['absent.md']}))
    result = tool('check_public_links.py', tmp_path, '--manifest', manifest)
    assert result.returncode == 1
    assert 'absent.md' in result.stdout


@pytest.mark.parametrize('ignored', [False, True])
def test_nested_candidate_is_scanned_despite_ancestor_repository(repo, ignored):
    if ignored:
        (repo / '.gitignore').write_text('candidate/\n')
        git(repo, 'add', '.gitignore')
        git(repo, 'commit', '-qm', 'Ignored candidate fixture')
    candidate = repo / 'candidate'
    candidate.mkdir()
    (candidate / 'payload.txt').write_text('syntheticmarker')
    result = scan(candidate)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'payload.txt' in result.stdout and 'private-name' in result.stdout


@pytest.mark.parametrize('changelog', ['CHANGELOG.md', 'CHANGELOG-dev.md'])
def test_history_exception_is_not_shared_by_identical_blob_at_another_path(repo, changelog):
    allowed = repo / changelog
    allowed.write_text('spice' + '@ConditionState.NONE\n')
    unrelated = repo / 'unrelated.txt'
    unrelated.write_bytes(allowed.read_bytes())
    git(repo, 'add', changelog, 'unrelated.txt')
    git(repo, 'commit', '-qm', 'Duplicate blob fixture')
    unrelated.unlink()
    git(repo, 'add', 'unrelated.txt')
    git(repo, 'commit', '-qm', 'Clean current fixture')
    assert scan(repo, '').returncode == 0
    result = scan(repo, '', '--history')
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'unrelated.txt' in result.stdout and 'email' in result.stdout
