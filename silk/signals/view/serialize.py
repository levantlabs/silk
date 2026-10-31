"""silk.signals.view.serialize — payload dict -> ``data.js`` / ``stamp.js`` text.

See design spec §4.1 for the JSONP-style handshake this module writes, and the
rationale for a separate ``stamp.js`` (cheap change detector so the reload
loop does not re-parse a multi-megabyte ``data.js`` on every poll).
"""
import json
import re

from silk._json_codec import encode_json

__all__ = ['to_json', 'build_data_js', 'build_stamp_js']

# A Title: line from a rawfile is attacker-adjacent text opened with file://
# privileges (§7.7). json.dumps does not escape '/', so a title containing
# "</script>" would, verbatim, prematurely close an inline <script> block (in
# --inline mode) or (defensively) any consumer that greps script bodies.
# json.dumps() leaves round-trip-decodable "\/" pairs intact, so this is a
# semantics-preserving, JSON-safe defusal, not a hack.
_SCRIPT_CLOSE_RE = re.compile(r'(?i)</script')


def to_json(payload):
    """Serialize *payload* to JSON text.

    ``allow_nan=False``: every numeric value that reaches this function is
    either an int, a validated-finite annotation value, or a plain string —
    raw signal data always travels as base64 inside a string, never as a bare
    JSON float. A NaN/Inf surfacing here would mean a bug upstream, and
    ``json.dumps`` raising is the fail-loud behaviour we want instead of
    emitting the non-standard (and uPlot-hostile) ``NaN`` / ``Infinity``
    tokens.
    """
    return json.dumps(encode_json(payload, finite_only=True), allow_nan=False)


def _escape_script_close(text):
    return _SCRIPT_CLOSE_RE.sub('<\\/script', text)


def build_data_js(payload):
    """Render the ``data.js`` (or inlined ``<script>`` body) JSONP handshake.

    On first page load ``__silkViewData`` does not exist yet, so the payload
    lands in ``window.SILK_VIEW_DATA``; on a reload injection the app has
    installed the function already, so it is called directly (spec §4.1).
    """
    json_text = _escape_script_close(to_json(payload))
    return (
        '(function(){var p=' + json_text + ';\n'
        "if(typeof window.__silkViewData==='function'){window.__silkViewData(p);}\n"
        'else{window.SILK_VIEW_DATA=p;}})();'
    )


def build_stamp_js(run_id, mtime):
    """Render ``stamp.js``: ~80 bytes carrying the run id and mtime.

    Polled every ``poll_ms`` by the reload loop; re-injecting the much larger
    ``data.js`` only when the run id here differs from the loaded payload's.
    """
    return (
        'window.__silkViewStamp&&window.__silkViewStamp('
        f'{json.dumps(encode_json(run_id), allow_nan=False)},{json.dumps(encode_json(float(mtime)), allow_nan=False)});'
    )
