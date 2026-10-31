#!/usr/bin/env python3
"""Scan every shipping file, filename, image metadata and optional Git history.

Exit 0: clean; 1: findings; 2: invalid configuration; 3: UNARMED.
RELEASE_PRIVATE_NAMES is a comma/whitespace separated runtime list. An explicitly
empty value arms an empty list. Otherwise read the assignment from a gitignored
.release-remote in the target, its main worktree, or this script's worktrees.
The config is parsed as data, never executed. No file or rule family is exempt.

Credential/identity rule shapes and configuration discovery adapted from an
MIT-licensed release scanner, copyright Levant Labs. No private policy is bundled.
"""
import argparse
import getpass
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import zlib

CONFIG_VAR = 'RELEASE_PRIVATE_NAMES'
CONFIG_BASENAME = '.release-remote'
GENERIC = {'admin', 'build', 'builder', 'ci', 'dev', 'docker', 'git', 'guest',
           'home', 'me', 'node', 'root', 'runner', 'test', 'ubuntu', 'user',
           'users', 'vagrant', 'you', 'localhost'}
# Each exception is restricted to matched text and carries an auditable reason.
ALLOWLIST = [
    ('email', r'@example\.(?:com|net|org)$', 'Reserved documentation domains.'),
    ('email', r'@(?:[A-Za-z0-9-]+\.)*(?:invalid|test|example|localhost)$',
     'Reserved test domains, not deliverable mailboxes.'),
    ('email', r'^git@github\.com$', 'Generic SSH clone syntax, not a person.'),
    ('home-path', r'^/(?:home|Users)/user$', 'Generic documented account placeholder.'),
]

SCOPED_ALLOWLIST = [
    ('email', r'^spice@ConditionState\.NONE$',
     'Serialized source/corner key, not a mailbox.',
     {'CHANGELOG.md', 'CHANGELOG-dev.md', 'silk/specs/spec.py'}),
    # silk/provenance.py is the module's earlier path; history scans still read it there.
    ('home-path', r'^/home/(?:u|you|userX|users)$',
     'Synthetic accounts used to explain and test home-path boundary handling.',
     {'docs/reports.md', 'docs/settings.md', 'silk/provenance.py', 'silk/simulation/provenance.py',
      'tests/test_provenance.py'}),
    ('home-path', r'^/' r'home/runner$',
     'Regression assertion forbidding assumptions about the hosted runner account.',
     {'tests/test_ci_workflow_runner.py'}),
]

PATTERNS = [
    ('private-key', r'-----BEGIN(?: [A-Z0-9]+)* PRIVATE KEY-----'),
    ('api-key', r'\bsk-[A-Za-z0-9_-]{20,}'),
    ('access-token', r'\bgh[pousr]_[A-Za-z0-9]{20,}'),
    ('fine-grained-token', r'\bgithub_pat_[A-Za-z0-9_]{20,}'),
    ('cloud-key', r'\bAKIA[0-9A-Z]{16}\b'),
    ('service-token', r'\bxox[baprs]-[A-Za-z0-9-]{10,}'),
    ('bearer-token', r'\bBearer\s+[A-Za-z0-9._~+/=-]{16,}'),
    ('secret-assignment', r'''\b(?:api[-_]?key|secret|passwd|password|access[-_]?key|auth[-_]?token|client[-_]?secret|private[-_]?key)\b\s*["']?\s*[:=]\s*["'][^"'\s]{8,}["']'''),
    ('home-path', r'/(?:home|Users)/[A-Za-z0-9._][A-Za-z0-9._-]*'),
    ('email', r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
    ('private-ipv4', r'(?<![\d.])(?:10\.(?:25[0-5]|2[0-4]\d|1?\d?\d)|172\.(?:1[6-9]|2\d|3[01])|192\.168)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){2}(?!\.?\d)'),
]


def git(root, *args):
    proc = subprocess.run(['git', '-C', str(root), *args], capture_output=True)
    if proc.returncode:
        raise ValueError('Git operation failed: ' + args[0])
    return proc.stdout


def config(root, key):
    try:
        return git(root, 'config', '--get', key).decode().strip()
    except ValueError:
        return ''


def main_worktree(root):
    try:
        raw = git(root, 'rev-parse', '--git-common-dir').decode().strip()
        return (root / raw).resolve().parent
    except ValueError:
        return None


def load_private_names(root):
    if CONFIG_VAR in os.environ:
        return re.split(r'[\s,]+', os.environ[CONFIG_VAR].strip()), True
    seen = set()
    for base in (root, Path(__file__).resolve().parents[1]):
        for directory in (base, main_worktree(base)):
            if directory is None or directory in seen:
                continue
            seen.add(directory)
            path = directory / CONFIG_BASENAME
            if not path.exists():
                continue
            if path.is_symlink():
                raise ValueError('private config must not be a symlink')
            # Reject config that can enter an archive, even if listed in an ignore file.
            tracked = git(directory, 'ls-files', '-z', '--', CONFIG_BASENAME)
            ignored = subprocess.run(['git', '-C', str(directory), 'check-ignore', '-q',
                                      CONFIG_BASENAME]).returncode == 0
            if tracked or not ignored:
                raise ValueError('private config must be untracked and gitignored')
            text = path.read_text(encoding='utf-8')
            values = re.findall(r'^\s*(?:export\s+)?' + CONFIG_VAR + r'\s*=\s*(.*?)\s*$', text, re.M)
            if values:
                value = values[-1]
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                else:
                    value = value.split('#', 1)[0].strip()
                return re.split(r'[\s,]+', value.strip()), True
    return [], False


def literal(term):
    return re.compile(r'(?<![A-Za-z0-9])' + re.escape(term) + r'(?![A-Za-z0-9])', re.I)


def rules(root, names):
    result = [(name, re.compile(pattern, re.I)) for name, pattern in PATTERNS]
    candidates = [getpass.getuser(), os.environ.get('USER', ''), Path.home().name]
    full_name = config(root, 'user.name')
    candidates.extend([full_name, config(root, 'user.email')])
    candidates.extend(p for p in re.split(r'[\s-]+', full_name)
                      if len(p) >= 5 and re.fullmatch(r"[A-Za-z][A-Za-z'-]+", p))
    host = socket.gethostname()
    candidates.extend([host, host.split('.')[0]])
    result.extend(('derived-identity', literal(p)) for p in sorted(set(candidates))
                  if len(p) >= 5 and p.casefold() not in GENERIC)
    result.extend(('private-name', literal(n)) for n in names if n)
    return result


def check_text(label, text, patterns):
    out = []
    path = re.sub(r'^[0-9a-f]{12}:', '', label).split(' [', 1)[0]
    for name, pattern in patterns:
        for match in pattern.finditer(text):
            if any(rule == name and reason.strip() and re.search(allowed, match.group(), re.I)
                   for rule, allowed, reason in ALLOWLIST) or any(
                    rule == name and path in paths and reason.strip() and re.fullmatch(allowed, match.group(), re.I)
                    for rule, allowed, reason, paths in SCOPED_ALLOWLIST):
                continue
            # Never print a credential or confidential matched value.
            line = text.count('\n', 0, match.start()) + 1
            out.append(f'{label}:{line}: {name}')
    return out


def image_metadata(data):
    """Decode compressed PNG text and EXIF strings as well as raw image bytes."""
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        offset = 8
        while offset < len(data):
            if offset + 12 > len(data):
                raise ValueError('truncated PNG metadata')
            length = struct.unpack('>I', data[offset:offset + 4])[0]
            kind = data[offset + 4:offset + 8]
            payload = data[offset + 8:offset + 8 + length]
            if offset + length + 12 > len(data):
                raise ValueError('truncated PNG chunk')
            if kind == b'zTXt':
                key, rest = payload.split(b'\0', 1)
                if rest[0] != 0:
                    raise ValueError('unsupported PNG compression')
                yield key + b'\0' + zlib.decompress(rest[1:])
            elif kind == b'iTXt':
                key, rest = payload.split(b'\0', 1)
                flag, method = rest[:2]
                language, translated, content = rest[2:].split(b'\0', 2)
                if method != 0 or flag not in (0, 1):
                    raise ValueError('unsupported PNG text encoding')
                yield key + b'\0' + language + b'\0' + translated + b'\0' + (
                    zlib.decompress(content) if flag else content)
            elif kind in (b'tEXt', b'eXIf'):
                yield payload
            offset += length + 12
    if data.startswith(b'\xff\xd8') or b'Exif\0\0' in data:
        # EXIF strings may be UTF-16. Decode both byte orders without guessing.
        for offset in (0, 1):
            for encoding in ('utf-16-le', 'utf-16-be'):
                yield data[offset:].decode(encoding, errors='replace').encode('utf-8')


def check_data(label, data, patterns):
    findings = check_text(label + ' [filename]', label, patterns)
    # Binary fixtures are included: their printable strings are still scanned.
    findings.extend(check_text(label, data.decode('utf-8', errors='replace'), patterns))
    for meta in image_metadata(data):
        findings.extend(check_text(label + ' [metadata]', meta.decode('utf-8', errors='replace'), patterns))
        findings.extend(check_text(label + ' [metadata]', meta.decode('latin-1'), patterns))
    return findings


def scan(root, patterns):
    try:
        top = Path(git(root, 'rev-parse', '--show-toplevel').decode().strip()).resolve()
        if top != root.resolve():
            raise ValueError('scan directory is inside an ancestor repository')
        paths = [p for p in git(root, 'ls-files', '-z').decode().split('\0') if p]
    except ValueError:
        walked = [p for p in root.rglob('*') if '.git' not in p.relative_to(root).parts]
        if any(p.is_symlink() for p in walked):
            raise ValueError('nonregular shipping entry')
        paths = [p.relative_to(root).as_posix() for p in walked if p.is_file()]
    out = []
    for rel in sorted(paths):
        path = root / rel
        if path.is_symlink() or not path.is_file():
            raise ValueError('missing or nonregular shipping file')
        out.extend(check_data(rel, path.read_bytes(), patterns))
    return out


def split_identities(raw, fields):
    """Return the identity header lines of a commit or tag object, and the text to scan.

    Identities are read only from the header, which ends at the first blank line;
    every message line is scanned as text, whatever word it starts with.
    """
    header, separator, message = raw.partition('\n\n')
    identities, clean = [], []
    for line in header.splitlines():
        if line.startswith(fields):
            identities.append(line.split(' ', 1)[1].rsplit(' ', 2)[0])
        else:
            clean.append(line)
    return identities, clean + ([''] if separator else []) + message.splitlines()


def scan_history(root, patterns):
    name = os.environ.get('RELEASE_AUTHOR_NAME')
    email = os.environ.get('RELEASE_AUTHOR_EMAIL')
    if not name or not email:
        raise ValueError('history checks require RELEASE_AUTHOR_NAME and RELEASE_AUTHOR_EMAIL')
    expected = f'{name} <{email}>'
    findings, seen = [], set()
    for commit in git(root, 'rev-list', '--all').decode().splitlines():
        raw = git(root, 'cat-file', 'commit', commit).decode(errors='replace')
        identities, clean = split_identities(raw, ('author ', 'committer '))
        for identity in identities:
            if identity != expected:
                findings.append(f'{commit[:12]}: author/committer outside allowlist')
        findings.extend(check_text(commit[:12] + ' [commit]', '\n'.join(clean), patterns))
        for entry in git(root, 'ls-tree', '-rz', commit).split(b'\0'):
            if not entry:
                continue
            header, path = entry.split(b'\t', 1)
            mode, kind, oid = header.decode().split()
            rel = path.decode()
            findings.extend(check_text(commit[:12] + ' [filename]', rel, patterns))
            if kind != 'blob' or mode not in ('100644', '100755'):
                raise ValueError('nonregular history entry')
            if (oid, rel) not in seen:
                seen.add((oid, rel))
                findings.extend(check_data(commit[:12] + ':' + rel, git(root, 'cat-file', 'blob', oid), patterns))
    # Annotated tags carry independent messages and identities.
    for ref in git(root, 'for-each-ref', '--format=%(objecttype) %(objectname) %(refname)').decode().splitlines():
        kind, oid, refname = ref.split()
        findings.extend(check_text('[ref]', refname, patterns))
        if kind == 'tag':
            text = git(root, 'cat-file', 'tag', oid).decode(errors='replace')
            identities, clean = split_identities(text, ('tagger ',))
            for identity in identities:
                if identity != expected:
                    findings.append('[tag]: author outside allowlist')
            findings.extend(check_text('[tag]', '\n'.join(clean), patterns))
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, nargs='?', default=Path.cwd())
    parser.add_argument('--history', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error('root must be a directory')
    try:
        names, armed = load_private_names(root)
        if not armed:
            print('UNARMED: no private-name configuration found')
            return 3
        patterns = rules(root, names)
        findings = scan(root, patterns)
        if args.history:
            findings.extend(scan_history(root, patterns))
        for finding in sorted(set(findings)):
            print(finding)
        print(f'Release scan: {len(set(findings))} findings; private-name configuration armed')
        return 1 if findings else 0
    except (OSError, ValueError, zlib.error, IndexError) as exc:
        print(f'Scan refused: {exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
