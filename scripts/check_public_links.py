#!/usr/bin/env python3
"""Check local public Markdown destinations and bare documentation references.

Checks files in the supplied shipping directory, including tests and docstrings.
Remote URLs are not fetched. Fragment validation remains with the documentation
suite; this gate verifies that each local destination survives the export.
"""
import argparse
import ast
import io
import json
import tokenize
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

INLINE = re.compile(r'!?\[[^\]\n]*\]\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+["\'][^\n]*?["\'])?\s*\)')
REFERENCE = re.compile(r'^\s{0,3}\[[^\]\n]+\]:\s*(?:<([^>]+)>|(\S+))', re.M)
CODE = re.compile(r'(?<![\\`])(`+)(?!`)((?:(?!\n[ \t]*\n).)*?)(?<!`)\1(?!`)', re.S)
FENCE = re.compile(r'^ {0,3}(`{3,}|~{3,})(.*)$')
BARE = re.compile(r'(?<![\w/])docs/[\w./-]+\.md\b')


def prose_links(text):
    """Mask fenced code and exact-delimiter inline code, retaining positions.

    A code span cannot consume part of a longer backtick run, cross a fenced
    block, or bridge separate paragraphs. Ambiguous unmatched markers remain
    visible so they cannot hide a destination from the checker.
    """
    lines = []
    fence = None
    blank = lambda line: ''.join('\n' if c == '\n' else ' ' for c in line)
    for line in text.splitlines(keepends=True):
        marker = FENCE.match(line)
        if fence:
            lines.append(blank(line))
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= fence[1] and not marker[2].strip():
                fence = None
        elif marker and not (marker[1][0] == '`' and '`' in marker[2]):
            fence = (marker[1][0], len(marker[1]))
            lines.append(blank(line))
        else:
            lines.append(line)
    return CODE.sub(lambda m: blank(m.group()), ''.join(lines))


def check(root, include=None):
    root = root.resolve()
    findings = []
    files = root.rglob('*') if include is None else (root / p for p in include)
    for file in sorted(files):
        if '.git' in file.relative_to(root).parts:
            continue
        if not file.is_file():
            if include is not None:
                findings.append(f'{file.relative_to(root)}: missing shipping file')
            continue
        if file.suffix not in ('.md', '.py', '.rst', '.txt'):
            continue
        text = file.read_text(encoding='utf-8', errors='replace')
        if file.suffix == '.py':
            try:
                tree = ast.parse(text)
                docs = [ast.get_docstring(n) for n in ast.walk(tree)
                        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
                comments = [t.string for t in tokenize.generate_tokens(io.StringIO(text).readline)
                            if t.type == tokenize.COMMENT]
                text = '\n\n'.join(t for t in docs + comments if t)
            except (SyntaxError, tokenize.TokenError):
                findings.append(f'{file.relative_to(root)}: unable to parse documentation')
                continue
        targets = []
        markup = prose_links(text)
        if file.suffix in ('.md', '.py', '.rst', '.txt'):
            for pattern in (INLINE, REFERENCE):
                for match in pattern.finditer(markup):
                    targets.append((match.group(1) or match.group(2), file.parent))
        targets.extend((m.group(), root) for m in BARE.finditer(text))
        for value, base in targets:
            parsed = urlsplit(value)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            path = (base / unquote(parsed.path)).resolve()
            shipping = True
            if include is not None and path.is_relative_to(root):
                rel = path.relative_to(root).as_posix()
                shipping = rel in include or any(p.startswith(rel.rstrip('/') + '/') for p in include)
            if not path.is_relative_to(root) or not path.exists() or not shipping:
                findings.append(f'{file.relative_to(root)}: missing local destination {value}')
    return sorted(set(findings))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--manifest', type=Path, help='check only literal include paths and their shipping destinations')
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error('root must be a directory')
    include = None
    if args.manifest:
        try:
            include = set(json.loads(args.manifest.read_text())['include'])
            if any(not isinstance(p, str) or Path(p).is_absolute() or '..' in Path(p).parts for p in include):
                parser.error('manifest includes must be safe relative paths')
        except (OSError, ValueError, KeyError, TypeError):
            parser.error('unable to read manifest include paths')
    findings = check(args.root, include)
    for finding in findings:
        print(finding)
    print(f'Local link check: {len(findings)} findings')
    return bool(findings)


if __name__ == '__main__':
    raise SystemExit(main())
