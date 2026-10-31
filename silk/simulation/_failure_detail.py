"""Bounded diagnostic excerpts for nonzero simulator exits, never verdicts."""

import re


# Match diagnostic syntax, not a word boundary: wrapped filenames such as
# error-cases and fatal-cases are still part of a Xyce file location.
_DIAGNOSTIC_HEADER = re.compile(
    r'^\s*(?:(?:(?:netlist|user)\s+)?(?:error|fatal)(?:\s*:|\s*$)|'
    r'error\s+on\s+line(?=\s|:|$)|'
    r'(?:netlist|user)\s+error\s+in\s+file(?=\s|$)|'
    r'fatal\s+error\s*:|simulation (?:aborted|interrupted)\b)', re.IGNORECASE)
_XYCE_LOCATION_END = re.compile(r'\b(?:at or near|near)\s+line\s+\d+\s*$', re.IGNORECASE)


def failure_detail(text, *, lead=(), noise=None):
    """Keep ordered, bounded diagnostic blocks, stopping at the next header.

    Most diagnostics have two following nonblank context lines. Xyce file
    locations instead end with "near line N", possibly across several lines;
    retain the whole location and its following reason. Incomplete blocks
    remain incomplete. Never consume the next diagnostic as missing context.
    Cap lines at 500 characters and the formatted excerpt at 4000, omitting
    whole blocks that cannot fit. Full logs/streams remain on the runner.

    When no diagnostic block is found, ``lead`` (already-matched cause lines,
    such as convergence lines) is the detail. Without it the fallback is the
    last 4000 characters of the text, after dropping lines for which the
    ``noise`` callable removes from the nonblank lines (a simulator's banner
    and statistics).
    """
    limit = 4000
    marker = ' ... [truncated]'
    lines = []
    size = 0
    source = [line.rstrip() for line in text.splitlines() if line.strip()]
    index = 0
    while index < len(source):
        line = source[index]
        index += 1
        lower = line.lower()
        if 'error' not in lower and 'fatal' not in lower:
            continue
        block = [line]
        xyce_location = lower.lstrip().startswith('netlist error in file')
        location_complete = bool(_XYCE_LOCATION_END.search(line))
        remaining = 1 if xyce_location and location_complete else 2
        while index < len(source):
            context = source[index]
            if _DIAGNOSTIC_HEADER.match(context):
                break
            block.append(context)
            index += 1
            if xyce_location and not location_complete:
                location_complete = bool(_XYCE_LOCATION_END.search(' '.join(block)))
                remaining = 1
            else:
                remaining -= 1
                if not remaining:
                    break
        clipped = []
        for entry in block:
            entry = entry.rstrip()
            if len(entry) > 500:
                entry = entry[:500 - len(marker)] + marker
            clipped.append(entry)
        addition = '\n  '.join(clipped)
        added_size = len(addition) + (3 if lines else 0)
        # Reserve the marker before adding any block: never slice through
        # an already accepted card/reason pair to make room for truncation.
        if size + added_size > limit - len(marker):
            return '\n  '.join(lines) + marker
        lines.append(addition)
        size += added_size
    detail = '\n  '.join(lines)
    if not detail:
        if lead:
            return _bounded(lead, limit, marker)
        if noise is None:
            # Unrecognized diagnostics still carry evidence, bounded as a tail.
            detail = text.strip() or '(no error lines found)'
        else:
            # The text without the simulator's noise lines; never restore them.
            kept = noise(source)
            detail = '\n'.join(kept).strip() or (
                '(no diagnostic lines found beyond the simulator banner and statistics)')
        if len(detail) > limit:
            return marker.strip() + '\n' + detail[-(limit - len(marker)):]
    return detail


def _bounded(lines, limit, marker):
    """Join cause lines in order within ``limit``, marking any clipping."""
    out = []
    size = 0
    for line in lines:
        line = line.rstrip()
        if len(line) > 500:
            line = line[:500 - len(marker)] + marker
        added = len(line) + (3 if out else 0)
        if size + added > limit - len(marker):
            return '\n  '.join(out) + marker
        out.append(line)
        size += added
    return '\n  '.join(out)
