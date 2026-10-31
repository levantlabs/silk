"""Recognise a simulator's convergence-failure messages.

Detection only. When a run has *already failed* (non-zero exit, no rawfile, or
a rawfile with no data rows) the failing run's log text is matched against the
per-simulator pattern table below; a hit turns the generic failure into a
:class:`~silk.exceptions.ConvergenceError`. Nothing here decides *whether* a
run failed, diagnoses the cause, retries, or touches tolerances.

The tables are data. Every entry was observed in a real log captured from
ngspice 42, ngspice 47 or Xyce 7.10 on a small deliberately non-converging deck
(``tests/fixtures/convergence_logs/``) unless its comment says otherwise.
"""
import os
import re

#: simulator name -> ((regex, note), ...). Matched per line, case-insensitively.
CONVERGENCE_PATTERNS = {
    'ngspice': (
        (r'timestep too small', 'transient / operating-point step collapsed'),
        (r'singular matrix', 'matrix could not be factored'),
        (r'gmin stepping failed', 'dynamic or true gmin stepping gave up'),
        (r'source stepping failed', 'source stepping gave up'),
        (r'no convergence in (?:dc|transient) ', 'iteration limit reached'),
        (r'operating point could not be simulated', 'no starting point'),
    ),
    'xyce': (
        (r'time step too small', 'transient step collapsed'),
        (r'dc operating point failed', 'no starting point'),
        (r'step size reached minimum step size bound', 'step collapsed'),
        (r'numerically singular matrix', 'matrix could not be factored'),
        (r'number nonlinear convergence failures:\s*[1-9]',
         'solution summary reports a failed nonlinear solve'),
        (r'nonlinear solver failed', 'listed by TASK-316; not seen in captured logs'),
    ),
}

_COMPILED = {sim: tuple(re.compile(p, re.IGNORECASE) for p, _ in table)
             for sim, table in CONVERGENCE_PATTERNS.items()}

#: Repeated warnings (a singular matrix is printed dozens of times) are
#: reported once each, and no more than this many distinct lines.
MAX_MATCHED_LINES = 10


def convergence_lines(simulator, text):
    """Return the distinct log lines of *text* that match *simulator*'s
    convergence patterns, in order (``[]`` for no match or an unknown
    simulator)."""
    patterns = _COMPILED.get(simulator)
    if not patterns or not text:
        return []
    found = []
    for line in text.splitlines():
        line = line.strip()
        if line and line not in found and any(p.search(line) for p in patterns):
            found.append(line)
            if len(found) == MAX_MATCHED_LINES:
                break
    return found


def read_log_text(log_path):
    """The text of *log_path*, or ``''`` when it cannot be read."""
    if not log_path or not os.path.isfile(log_path):
        return ''
    try:
        with open(log_path, errors='replace') as f:
            return f.read()
    except OSError:
        return ''


def convergence_error(simulator, text, message, *, reason=None,
                      data_rows=None, log_path=None):
    """A :class:`~silk.exceptions.ConvergenceError` when *text* matches a
    convergence pattern for *simulator*, else ``None`` (the caller then raises
    exactly what it raised before)."""
    lines = convergence_lines(simulator, text)
    if not lines:
        return None
    from silk.exceptions import ConvergenceError
    return ConvergenceError(message, simulator=simulator, matched_lines=lines,
                            reason=reason, data_rows=data_rows,
                            log_path=log_path)
