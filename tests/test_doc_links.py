"""Repo-wide doc-to-doc link guard (TASK-189).

``tests/test_architecture_doc.py`` pins every cited path in one document. This
generalises the same idea to every tracked markdown file that is *living*:

* each markdown link target (``[text](path)``, resolved relative to the file
  that contains it; URLs skipped), and
* each backticked ``silk/``, ``tests/``, ``scripts/``, ``docs/`` or
  ``examples/`` path ending in ``.md``, resolved against the repo root,

must exist. A ``#fragment`` on a link to a living markdown file (or alone,
for the file itself) must name a heading or explicit anchor in that file, with
headings slugged the way the docs render them (``_slugs``). A dead pointer costs a reader a search before they learn the
target is gone.

Frozen historical records are **excluded, not rewritten** (``_HISTORICAL``):
correspondence, reviews, brainstorms, specs and dated notes describe the repo
as it was on the day they were written, and repairing a path inside one would
falsify the record. New material belongs outside those directories, where this
guard applies.

Deliberate cross-repo pointers, which name another repository's file in a path
spelled like a local one, are listed in ``_CROSS_REPO`` with the sentence that
makes the owning repo explicit. Fast suite, no PDK or simulator.
"""
import re
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# Frozen records: not checked, not rewritten.
_HISTORICAL = ('docs/archive/', 'docs/superpowers/', 'docs/backlog/', 'backlog/')

# (file, cited path) -> why it is allowed to dangle. Each names its repo in the
# surrounding sentence, so a reader is not misled.
_CROSS_REPO = {}

_LINK = re.compile(r'(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)')
_TICKED = re.compile(r'`((?:silk|tests|scripts|docs|examples)/[^`\s]*?\.md)(?::\d+)?`')


def _living_docs():
    out = subprocess.run(['git', 'ls-files', '*.md'], cwd=_REPO, check=True,
                         capture_output=True, text=True).stdout.split()
    return [p for p in out if not p.startswith(_HISTORICAL)
            and (_REPO / p).is_file()]


# Fence recognition is deliberately not a Markdown parser. An opener may sit
# behind blockquote markers and/or one list marker; the scanner remembers the
# opener's quote depth and indent and holds the fence open only while lines
# keep that depth and indent.
_QUOTE = re.compile(r'^ {0,3}>[ \t]?')
_OPEN = re.compile(r'^([ \t]*)((?:[-*+]|\d{1,9}[.)])[ \t]+)?(`{3,}|~{3,})(.*)$')
_CLOSE = re.compile(r'^ {0,3}(`{3,}|~{3,})[ \t]*$')
_HEADING = re.compile(r'^ {0,3}#{1,6}[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$')
_EXPLICIT = re.compile(r'<a\s[^>]*?\b(?:id|name)\s*=\s*["\']([^"\']+)["\']', re.I)
_CODE_SPAN = re.compile(r'(`+)(.+?)\1')


def _unquote(line, depth=None):
    """``(markers_stripped, rest)``: strip blockquote markers, all of them or
    at most ``depth``."""
    n = 0
    while depth is None or n < depth:
        m = _QUOTE.match(line)
        if not m:
            break
        line, n = line[m.end():], n + 1
    return n, line


def _outside_fences(text):
    """Lines of ``text`` that are not inside a fenced code block.

    A fence opens behind any blockquote markers and one optional list marker,
    and closes on the same character, at least as long as the opener, at the
    opener's quote depth and indent, with no list marker. A non-blank line
    with less quote depth or less indent leaves the container and so ends the
    fence (that line is then read normally). A fence still open at the end of
    the file simply runs to the end.

    Limits (not a general parser): nested containers beyond quote-then-one-list
    marker are not tracked, a fence indented one space under no list is read as
    unindented, and lazy continuation lines are not modelled. Where the scanner
    is unsure it ends the fence, so a line is checked rather than hidden.
    """
    out, fence = [], None            # fence = (char, length, depth, indent)
    for line in text.splitlines():
        if fence is not None:
            char, length, depth, indent = fence
            n, body = _unquote(line, depth)
            blank = not body.strip()
            if n < depth or (not blank
                             and len(body) - len(body.lstrip(' \t')) < indent):
                fence = None                       # left the container: reread
            else:
                m = _CLOSE.match(body[indent:])
                if m and m.group(1)[0] == char and len(m.group(1)) >= length:
                    fence = None
                continue
        depth, body = _unquote(line)
        m = _OPEN.match(body)
        if m and not (m.group(3)[0] == '`' and '`' in m.group(4)):
            lead = len(m.group(1).expandtabs())
            indent = len(m.group(1)) + len(m.group(2)) if m.group(2) \
                else (lead if lead >= 2 else 0)
            fence = (m.group(3)[0], len(m.group(3)), depth, indent)
            continue
        out.append(line)
    return out


def _prose(text):
    """``text`` as a renderer shows it outside code: fenced blocks and inline
    code spans removed. Used to find real links and real HTML anchors."""
    return '\n'.join(_CODE_SPAN.sub('', ln) for ln in _outside_fences(text))


def _heading_slug(heading):
    """GitHub-style slug of a heading's source text.

    Formatting contributes its text only: links keep their label, ``*``/``_``
    emphasis and HTML tags vanish. Code-span contents stay literal (so
    ``foo__bar`` keeps both underscores). The text is lowercased, everything
    but letters, digits, ``_``, ``-`` and spaces is dropped, and each space
    becomes ``-`` (consecutive spaces are not merged).
    """
    spans = []

    def keep(m):
        spans.append(m.group(2))
        return f'\x00{len(spans) - 1}\x00'

    h = _CODE_SPAN.sub(keep, heading)
    h = re.sub(r'!?\[([^\]]*)\]\([^)]*\)', r'\1', h)
    h = re.sub(r'<[^>]+>', '', h)
    h = h.replace('*', '')
    prev = None
    while prev != h:            # nested emphasis: __a _b_ c__ needs two passes
        prev = h
        h = re.sub(r'(?<![\w])(_+)(?=\S)(.+?)(?<=\S)\1(?![\w])', r'\2', h)
    h = re.sub(r'\x00(\d+)\x00', lambda m: spans[int(m.group(1))], h)
    h = re.sub(r'[^\w\- ]', '', h.strip().lower())
    return h.replace(' ', '-')


def _slugs(text):
    """Every fragment a reader can link to in ``text``: automatic heading IDs
    plus explicit ``<a id=...>`` / ``<a name=...>`` anchors.

    Automatic IDs are unique: a repeated base gets the first free ``-1``,
    ``-2``, ... suffix, even when a heading is literally spelled ``Notes-1``.
    Explicit anchors take no part in that numbering. Headings and anchors
    inside code fences or code spans are examples, not targets."""
    allocated, counts, explicit = set(), {}, set()
    for line in _outside_fences(text):
        m = _HEADING.match(line)
        if m:
            base = _heading_slug(m.group(1))
            ident, n = base, counts.get(base, 0)
            while ident in allocated:
                n += 1
                ident = f'{base}-{n}'
            counts[base] = n
            allocated.add(ident)
        explicit.update(_EXPLICIT.findall(_CODE_SPAN.sub('', line)))
    return allocated | explicit


def _links(text):
    """Every markdown link target in the rendered prose of ``text``."""
    return [t for t in _LINK.findall(_prose(text))]


def _fragment_links(text):
    """``[(target, fragment), ...]`` for each link with a ``#fragment``, ignoring
    fenced blocks and code spans (intentional Markdown examples)."""
    out = []
    for target in _links(text):
        if re.match(r'[a-z][a-z0-9+.-]*:', target) or '#' not in target:
            continue
        path, frag = target.split('#', 1)
        if frag:
            out.append((path, frag))
    return out


def dead_fragments():
    """``[(file, target, fragment), ...]`` whose fragment names no heading or
    anchor in the (living, markdown) file the link points at."""
    bad, cache = [], {}
    for rel in _living_docs():
        path = _REPO / rel
        for target, frag in _fragment_links(path.read_text(errors='replace')):
            dest = (path.parent / target) if target else path
            if dest.suffix != '.md' or not dest.is_file():
                continue          # missing files are dangling()'s business
            if dest not in cache:
                cache[dest] = _slugs(dest.read_text(errors='replace'))
            if frag not in cache[dest] and (rel, target + '#' + frag) not in _CROSS_REPO:
                bad.append((rel, target, frag))
    return sorted(set(bad))


def dangling():
    """``[(file, target), ...]`` for every unresolved, non-allowlisted reference."""
    bad = []
    for rel in _living_docs():
        path = _REPO / rel
        text = path.read_text(errors='replace')
        refs = []
        for target in _links(text):
            if re.match(r'[a-z][a-z0-9+.-]*:', target) or target.startswith('#'):
                continue
            target = target.split('#', 1)[0]
            if target:
                refs.append((target, (path.parent / target)))
        for target in _TICKED.findall('\n'.join(_outside_fences(text))):
            refs.append((target, _REPO / target))
        for target, resolved in refs:
            if not resolved.exists() and (rel, target) not in _CROSS_REPO:
                bad.append((rel, target))
    return sorted(set(bad))


def test_living_docs_are_found():
    assert len(_living_docs()) > 20


def test_no_dangling_doc_references():
    bad = dangling()
    assert not bad, ('dangling doc references (repair, or allowlist a deliberate '
                     'cross-repo pointer in _CROSS_REPO):\n'
                     + '\n'.join(f'  {f} -> {t}' for f, t in bad))


def test_cross_repo_allowlist_has_no_stale_entries():
    """An allowlist entry must still be cited (else delete it) and still dangle."""
    for (rel, target), why in _CROSS_REPO.items():
        assert why
        assert (_REPO / rel).is_file(), f'{rel} no longer exists; drop its allowlist entry'
        assert target in (_REPO / rel).read_text(), \
            f'{rel} no longer cites {target}; drop its allowlist entry'
        assert not (_REPO / target).exists(), \
            f'{target} now exists locally; drop its allowlist entry'


def test_no_dead_fragment_links():
    bad = dead_fragments()
    assert not bad, ('links whose #fragment matches no heading or anchor '
                     '(fix the link, or add the heading/anchor):\n'
                     + '\n'.join(f'  {f} -> {t}#{a}' for f, t, a in bad))


# --- controls for the slug and fragment rules ------------------------------
def test_slug_plain_and_punctuation():
    assert _heading_slug('Choosing a simulator — ngspice (default) or Xyce') == \
        'choosing-a-simulator--ngspice-default-or-xyce'
    assert _heading_slug('1. What silk is + the build') == '1-what-silk-is--the-build'
    assert _heading_slug('snake_case stays') == 'snake_case-stays'


def test_slug_inline_markup_keeps_text_only():
    assert _heading_slug('Two `add_constraint` methods') == 'two-add_constraint-methods'
    assert _heading_slug('**Bold** and *em* [label](x.md)') == 'bold-and-em-label'
    assert _heading_slug('Closing hashes ##'.rsplit(' ##', 1)[0]) == 'closing-hashes'


def test_slugs_duplicate_headings_get_numeric_suffixes():
    got = _slugs('# Notes\n\n## Notes\n\n### Notes\n')
    assert got == {'notes', 'notes-1', 'notes-2'}


def test_slugs_explicit_anchors():
    got = _slugs('<a id="legacy-name"></a>\n## Real\n<a name=\'other\'>x</a>\n')
    assert {'legacy-name', 'other', 'real'} <= got


def test_slugs_ignore_headings_and_anchors_inside_fences():
    text = '## Real\n```md\n## Fake\n<a id="ghost"></a>\n```\n~~~\n## Tilde\n~~~\n## After\n'
    assert _slugs(text) == {'real', 'after'}


def test_fence_needs_a_matching_closer():
    # A shorter or different-character line does not close the fence.
    assert _slugs('````\n```\n## Inner\n~~~~\n## Still\n````\n## Out\n') == {'out'}


def test_fragment_links_skip_fences_and_code_spans():
    text = ('[ok](#real)\n```md\n[example](#fenced)\n```\n'
            'prose `[span](#spanned)` and [two](other.md#sec).\n'
            '[web](https://x.org/p#frag) [bare](other.md)\n')
    assert _fragment_links(text) == [('', 'real'), ('other.md', 'sec')]


def test_negative_control_planted_dead_anchor_is_caught(tmp_path, monkeypatch):
    """The checker must flag a dead anchor and pass a live one."""
    (tmp_path / 'a.md').write_text('# A\n\nsee [b](b.md#live) and [c](b.md#dead) '
                                   'and [d](#a) and [e](#nope)\n')
    (tmp_path / 'b.md').write_text('# B\n\n## Live\n')
    monkeypatch.setattr('tests.test_doc_links._REPO', tmp_path)
    monkeypatch.setattr('tests.test_doc_links._living_docs', lambda: ['a.md', 'b.md'])
    assert dead_fragments() == [('a.md', '', 'nope'), ('a.md', 'b.md', 'dead')]


# --- controls for review ruling 612a (P2-1 .. P2-3) -------------------------
def _heading_ids(markdown):
    """Slugs a document offers, through the real heading recogniser."""
    return _slugs(markdown)


def test_slug_emphasis_is_stripped_but_code_spans_stay_literal():
    assert _heading_ids('# _Italic_\n') == {'italic'}
    assert _heading_ids('# __Bold__ and *em*\n') == {'bold-and-em'}
    assert _heading_ids('# `foo__bar`\n') == {'foo__bar'}          # positive
    assert 'foobar' not in _heading_ids('# `foo__bar`\n')            # negative
    assert _heading_ids('# snake_case_name\n') == {'snake_case_name'}  # not emphasis
    assert _heading_ids('# `a` _b_ `c_d`\n') == {'a-b-c_d'}


def test_heading_recogniser_drops_closing_hashes():
    assert _heading_ids('## Closing hashes ##\n') == {'closing-hashes'}


def test_generated_ids_are_unique_in_both_collision_orders():
    # The second 'Notes' takes 'notes-1', so a literal 'Notes-1' must get 'notes-1-1'.
    assert _heading_ids('# Notes\n# Notes\n# Notes-1\n') == {'notes', 'notes-1', 'notes-1-1'}
    # Reversed: the literal heading holds 'notes-1', so the repeat skips to 'notes-2'.
    assert _heading_ids('# Notes\n# Notes-1\n# Notes\n') == {'notes', 'notes-1', 'notes-2'}


def test_explicit_anchors_do_not_take_part_in_numbering():
    got = _heading_ids('<a id="notes"></a>\n# Notes\n# Notes\n')
    assert got == {'notes', 'notes-1'}      # first heading still gets plain 'notes'


def test_anchors_in_inline_code_are_not_targets():
    assert _heading_ids('Write `<a id="ghost"></a>` to add one.\n') == set()
    assert _heading_ids('<a id="real"></a> and `<a id="ghost"></a>`\n') == {'real'}


def test_fences_inside_blockquotes_and_lists_are_code():
    text = ('> ```md\n> [example](#ghost)\n> ## Quoted\n> ```\n'
            '- item\n\n  ```md\n  [e2](missing.md#x)\n  ## Listed\n  ```\n'
            '1. ```\n   ## Numbered\n   ```\n## Real\n')
    assert _slugs(text) == {'real'}
    assert _fragment_links(text) == []


def _fake_repo(monkeypatch, tmp_path, files):
    for name, body in files.items():
        (tmp_path / name).write_text(body)
    monkeypatch.setattr('tests.test_doc_links._REPO', tmp_path)
    monkeypatch.setattr('tests.test_doc_links._living_docs', lambda: sorted(files))


def test_real_gates_ignore_fenced_and_quoted_example_links(tmp_path, monkeypatch):
    _fake_repo(monkeypatch, tmp_path, {
        'a.md': ('# A\n\n```md\n[example](missing.md#ghost)\n`docs/nope.md`\n```\n\n'
                 '> ```md\n> [quoted](gone.md)\n> ```\n\n`[span](also-gone.md)`\n'),
    })
    assert dangling() == []
    assert dead_fragments() == []


def test_real_gates_still_catch_real_dead_links(tmp_path, monkeypatch):
    _fake_repo(monkeypatch, tmp_path, {
        'a.md': '# A\n\n[x](missing.md#ghost) and [y](#ghost) and `docs/nope.md`\n',
    })
    assert ('a.md', 'missing.md') in dangling()
    assert ('a.md', 'docs/nope.md') in dangling()
    assert dead_fragments() == [('a.md', '', 'ghost')]


def test_real_fragment_gate_uses_corrected_slugs(tmp_path, monkeypatch):
    _fake_repo(monkeypatch, tmp_path, {
        'a.md': ('[i](b.md#italic) [c](b.md#foo__bar) [n](b.md#notes-1-1) '
                 '[bad1](b.md#_italic_) [bad2](b.md#foobar) [bad3](b.md#notes-2) '
                 '[bad4](b.md#ghost)\n'),
        'b.md': ('# _Italic_\n# `foo__bar`\n# Notes\n# Notes\n# Notes-1\n'
                 'Inline `<a id="ghost"></a>` is not an anchor.\n'),
    })
    assert dead_fragments() == [('a.md', 'b.md', '_italic_'), ('a.md', 'b.md', 'foobar'),
                                ('a.md', 'b.md', 'ghost'), ('a.md', 'b.md', 'notes-2')]


# --- controls for review ruling 612b ------------------------------------------
def test_nested_emphasis_slug_through_the_real_fragment_gate(tmp_path, monkeypatch):
    _fake_repo(monkeypatch, tmp_path, {
        'a.md': '[ok](b.md#foo-bar-baz) [bad](b.md#foo-_bar_-baz)\n',
        'b.md': '# __foo _bar_ baz__\n',
    })
    assert dead_fragments() == [('a.md', 'b.md', 'foo-_bar_-baz')]


_CODEX_EXAMPLE_1 = (
    '```md\n> ```\n[file](missing.md) [fragment](#ghost)\n```\n')
_CODEX_EXAMPLE_2 = (
    '> ```md\n> example\n\n[real broken link](missing.md) [frag](#ghost2)\n')
_LIST_FENCE = (
    '- item\n\n  ```md\n  [e](gone.md#x)\n  ```\n\n[after](lost.md)\n')
_LIST_MARKER_NOT_CLOSER = (
    '- ```md\n  - ```\n  [e](gone.md)\n  ```\n[after](lost2.md)\n')


def test_container_fences_through_the_real_gates(tmp_path, monkeypatch):
    _fake_repo(monkeypatch, tmp_path, {
        'one.md': '# One\n\n' + _CODEX_EXAMPLE_1,                 # all example: no hits
        'two.md': '# Two\n\n' + _CODEX_EXAMPLE_2,                 # link after the quote is real
        'three.md': '# Three\n\n' + _LIST_FENCE,                  # only [after] is real
        'four.md': '# Four\n\n' + _LIST_MARKER_NOT_CLOSER,        # marker line is content
        'five.md': ('# Five\n\n```md\n# Fake\n```\n\n> ```\n> [q](qgone.md)\n> ```\n'
                    '[x](#fake) [y](#five)\n'),
        'six.md': '# Six\n\n# Notes\n# Notes-1\n# Notes\n\n[ok](#notes-2) [bad](#notes-3)\n',
    })
    assert dangling() == [('four.md', 'lost2.md'), ('three.md', 'lost.md'),
                          ('two.md', 'missing.md')]
    assert dead_fragments() == [('five.md', '', 'fake'), ('six.md', '', 'notes-3'),
                                ('two.md', '', 'ghost2')]


def test_fence_scanner_container_edges():
    assert _outside_fences('> ```\n> a\n\nreal\n') == ['', 'real']
    assert _outside_fences('```\n- ```\nx\n```\ny\n') == ['y']       # marker line is content
    assert _outside_fences('```\n> ```\nx\n```\ny\n') == ['y']       # quote line is content
    assert _outside_fences('> ```\n> a\n> ```\nout\n') == ['out']
    assert _outside_fences('- ```\n  a\n  ```\nout\n') == ['out']
    assert _outside_fences('- ```\n  a\nless indented\n') == ['less indented']
    assert _outside_fences('```inline``` text\n') == ['```inline``` text']
