"""``python -m silk.signals.view`` — CLI for the waveform explorer.

See spec §5.2. Registered in ``docs/cli.md``'s verified-working-tools table.
"""
import argparse
import json
import os
import sys
import webbrowser

from silk.signals.rawfile import RawfileError
from silk.signals.view.emit import ViewError, emit_view
from silk.signals.view.select import block_label, emit_views_by_block, list_blocks
from silk.signals.view.serve import run_server

__all__ = ['main', 'build_parser']


def build_parser():
    parser = argparse.ArgumentParser(
        prog='python -m silk.signals.view',
        description='Turn a nutmeg .raw file into a self-contained, '
                    'browser-openable waveform explorer.',
    )
    parser.add_argument('rawfile', help='path to a nutmeg .raw file')
    parser.add_argument('-o', '--output', dest='output', default=None,
                        help='output HTML path (default: <rawfile dir>/view.html)')
    parser.add_argument('--title', default=None, help='page title')
    parser.add_argument('--inline', action='store_true',
                        help='single self-contained file, no sidecars, no auto-reload')
    parser.add_argument('--serve', action='store_true',
                        help='serve the output dir over http://127.0.0.1 and '
                             're-emit on rawfile mtime change (re-emits keep '
                             '--signals/--exclude/--max-depth); blocks until Ctrl-C')
    parser.add_argument('--port', type=int, default=0,
                        help='port for --serve (default 0 = OS-assigned, printed on stdout)')
    parser.add_argument('--open', dest='open_browser', action='store_true',
                        help='open the result in the default browser')
    parser.add_argument('--poll-ms', dest='poll_ms', type=int, default=2000,
                        help='sidecar poll interval baked into the page (default 2000, 0 = off)')
    view_state_group = parser.add_mutually_exclusive_group()
    view_state_group.add_argument('--view-state', dest='view_state', default=None,
                        help='embed this view-state file (error if missing)')
    view_state_group.add_argument('--no-view-state', dest='no_view_state', action='store_true',
                        help='do not auto-load an adjacent <stem>.silkview.json')
    parser.add_argument('--commands', dest='commands_json', default=None,
                        help='a JSON array of commands to replay on load (design spec §4)')
    parser.add_argument('--context-title', dest='context_title', default=None,
                        metavar='TEXT',
                        help='run provenance: page title shown in the info popover '
                             '(view-context-and-scopes spec §2.1); takes precedence '
                             'over --title')
    parser.add_argument('--context-subtitle', dest='context_subtitle', default=None,
                        metavar='TEXT', help='run provenance: subtitle text')
    parser.add_argument('--context-field', dest='context_field', action='append',
                        default=None, metavar='LABEL=VALUE',
                        help='run provenance field (repeatable), e.g. '
                             '--context-field corner=tt --context-field PDK=sky130A')
    parser.add_argument('--signals', dest='signals', action='append', default=None,
                        metavar='PAT',
                        help='include only signals matching PAT (repeatable; a '
                             'vector is kept if ANY --signals selector matches — '
                             'exact name, glob, or hierarchy contiguous-run, all '
                             'case-insensitive; an array index may be [k] or <k>, '
                             'and [ ] is never a glob class; see docs/cli.md)')
    parser.add_argument('--exclude', dest='exclude', action='append', default=None,
                        metavar='PAT',
                        help='drop signals matching PAT (repeatable, applied '
                             'after --signals; same selector forms; exclusion wins)')
    parser.add_argument('--max-depth', dest='max_depth', type=int, default=None,
                        metavar='N',
                        help='keep only signals with hierarchy depth <= N '
                             '(0 = bare/top-level names only)')
    parser.add_argument('--list-blocks', dest='list_blocks', nargs='?', type=int,
                        const=1, default=None, metavar='DEPTH',
                        help='print the hierarchy block table at DEPTH (default 1) '
                             'and exit; emits nothing')
    parser.add_argument('--split-by-block', dest='split_by_block', nargs='?', type=int,
                        const=1, default=None, metavar='DEPTH',
                        help='emit one page per hierarchy block at DEPTH (default 1) '
                             'plus top.html, into the output directory; each page '
                             'derives its own selection from its block path, so this '
                             'is not combinable with --signals/--exclude/--max-depth')
    return parser


# Flags that are meaningless (or actively contradictory) with --list-blocks,
# which prints a table and writes nothing.
_LIST_BLOCKS_CONFLICTS = (
    ('output', '-o/--output'),
    ('title', '--title'),
    ('inline', '--inline'),
    ('serve', '--serve'),
    ('view_state', '--view-state'),
    ('no_view_state', '--no-view-state'),
    ('commands_json', '--commands'),
    ('context_title', '--context-title'),
    ('context_subtitle', '--context-subtitle'),
    ('context_field', '--context-field'),
    ('signals', '--signals'),
    ('exclude', '--exclude'),
    ('max_depth', '--max-depth'),
    ('split_by_block', '--split-by-block'),
)


# Flags --split-by-block cannot honour: every page it emits already carries a
# selection derived from its own block path (signals=[path], or max_depth=0 for
# top.html), which a user selection can neither be ORed nor ANDed into without
# silently changing what "this block's page" means. Rejected up front rather
# than accepted-and-ignored — a typo'd selector must fail loud here exactly as
# it does on every other path.
_SPLIT_BY_BLOCK_CONFLICTS = (
    ('signals', '--signals'),
    ('exclude', '--exclude'),
    ('max_depth', '--max-depth'),
)


def _conflicting_flags(args, spec):
    names = []
    for attr, flag in spec:
        value = getattr(args, attr)
        if isinstance(value, bool):
            if value:
                names.append(flag)
        elif value is not None:
            names.append(flag)
    return names


def _format_success_line(result):
    if result.n_signals != result.n_signals_total:
        signals_part = f'{result.n_signals} of {result.n_signals_total} signals'
    else:
        signals_part = f'{result.n_signals} signals'
    suffix = ' [+view-state]' if result.view_state_path else ''
    return (
        f'wrote {os.path.basename(result.html_path)} '
        f'({result.n_plots} plots, {signals_part}, '
        f'{result.n_points} pts){suffix} -> {result.html_path}'
    )


def _print_blocks_table(depth, blocks, n_top_level):
    print(f'hierarchy blocks at depth {depth}:')
    if blocks:
        blocks = [(block_label(path), n) for path, n in blocks]
        width = max(len(path) for path, _n in blocks)
        width = max(width, len('block'))
        print(f'{"block":<{width}}  n_signals')
        for path, n in blocks:
            print(f'{path:<{width}}  {n}')
    else:
        print('(no blocks at this depth)')
    print(f'top-level (no hierarchy): {n_top_level} signals')
    if blocks:
        # Counts are what each block's page will hold, so they overlap: a
        # device internal like m.xdut.xmm1.… is inside both `m` and `xdut`.
        total = sum(n for _p, n in blocks) + n_top_level
        print(f'(blocks overlap — a signal can appear in several; '
              f'counts sum to {total}, not the signal total)')


def main(argv=None):
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    if args.inline and args.serve:
        parser.print_usage(sys.stderr)
        print(f'{parser.prog}: error: argument --inline: not allowed with '
             f'argument --serve (nothing to watch in inline mode)', file=sys.stderr)
        return 2

    if args.split_by_block is not None and args.serve:
        parser.print_usage(sys.stderr)
        print(f'{parser.prog}: error: argument --split-by-block: not allowed '
             f'with argument --serve (serve watches a single output file)',
             file=sys.stderr)
        return 2

    if args.split_by_block is not None:
        conflicts = _conflicting_flags(args, _SPLIT_BY_BLOCK_CONFLICTS)
        if conflicts:
            parser.print_usage(sys.stderr)
            print(f'{parser.prog}: error: argument --split-by-block: not allowed '
                 f'with {", ".join(conflicts)} (each page already selects its own '
                 f'block; use --split-by-block DEPTH to change the split, or drop '
                 f'--split-by-block to emit one filtered page)', file=sys.stderr)
            return 2

    if args.list_blocks is not None:
        conflicts = _conflicting_flags(args, _LIST_BLOCKS_CONFLICTS)
        if conflicts:
            parser.print_usage(sys.stderr)
            print(f'{parser.prog}: error: argument --list-blocks: not allowed '
                 f'with {", ".join(conflicts)} (it prints a table and writes '
                 f'nothing)', file=sys.stderr)
            return 2
        try:
            blocks, n_top = list_blocks(args.rawfile, depth=args.list_blocks)
        except (ViewError, RawfileError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        _print_blocks_table(args.list_blocks, blocks, n_top)
        return 0

    out_path = args.output
    if out_path is None:
        rawfile_dir = os.path.dirname(os.path.abspath(args.rawfile)) or '.'
        out_path = os.path.join(rawfile_dir, 'view.html')

    commands = None
    if args.commands_json is not None:
        try:
            commands = json.loads(args.commands_json)
        except json.JSONDecodeError as exc:
            print(f'{parser.prog}: error: --commands is not valid JSON: {exc}',
                 file=sys.stderr)
            return 1

    context = None
    if args.context_title or args.context_subtitle or args.context_field:
        context = {}
        if args.context_title:
            context['title'] = args.context_title
        if args.context_subtitle:
            context['subtitle'] = args.context_subtitle
        if args.context_field:
            fields = []
            for item in args.context_field:
                if '=' not in item:
                    parser.print_usage(sys.stderr)
                    print(f'{parser.prog}: error: argument --context-field: '
                         f'expected LABEL=VALUE, got {item!r}', file=sys.stderr)
                    return 2
                label, _, value = item.partition('=')
                fields.append({'label': label, 'value': value})
            context['fields'] = fields

    view_kwargs = dict(
        view_state=args.view_state,
        autoload_view_state=not args.no_view_state,
        commands=commands,
        context=context,
        labels='designer',
    )
    # Kept out of view_kwargs: emit_views_by_block derives signals=/max_depth=
    # per page and would collide with them (that combination is rejected
    # above). Everything else — the single-page emit and the --serve watcher's
    # re-emits — must carry them, or a served page reverts to every signal on
    # the first rawfile change.
    selection_kwargs = dict(
        signals=args.signals, exclude=args.exclude, max_depth=args.max_depth,
    )

    if args.split_by_block is not None:
        out_dir = os.path.dirname(os.path.abspath(out_path)) or '.'
        try:
            results = emit_views_by_block(
                args.rawfile, out_dir, depth=args.split_by_block,
                title_prefix=args.title, inline=args.inline,
                poll_ms=args.poll_ms, **view_kwargs,
            )
        except (ViewError, RawfileError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        for result in results:
            print(_format_success_line(result))
        if not results:
            # Otherwise a silent no-op: exit 0, nothing on stdout or stderr,
            # in an otherwise fail-loud package (the single-page path always
            # honestly emits a 0-signal page instead). This is also reached
            # with a rawfile that DOES have data vectors, just none that are
            # hierarchical or bare at this depth (e.g. --split-by-block 0 on
            # an all-hierarchical file) — so the message names the actual
            # cause, not "no data vectors" specifically.
            print(f'{parser.prog}: error: --split-by-block wrote no pages: '
                 f'no hierarchy blocks at this depth and no top-level '
                 f'signals', file=sys.stderr)
            return 1
        if args.open_browser:
            # The set is scope-linked (every page carries the SILK_VIEW_SCOPES
            # manifest), so any page reaches the rest; top.html is the natural
            # entry point when the rawfile has bare/top-level signals. results
            # is non-empty here (the empty case already returned above).
            target = next(
                (r for r in results
                 if os.path.basename(r.html_path) == 'top.html'),
                results[0],
            )
            webbrowser.open('file://' + target.html_path)
        return 0

    try:
        result = emit_view(
            args.rawfile, out_path, title=args.title, inline=args.inline,
            poll_ms=args.poll_ms,
            **selection_kwargs,
            **view_kwargs,
        )
    except (ViewError, RawfileError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(_format_success_line(result))

    if args.serve:
        run_server(
            out_path, args.rawfile, port=args.port,
            open_browser=args.open_browser,
            emit_kwargs=dict(title=args.title, poll_ms=args.poll_ms,
                             **selection_kwargs, **view_kwargs),
        )
    elif args.open_browser:
        webbrowser.open('file://' + result.html_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
