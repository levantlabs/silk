"""What one simulation attempt really launched.

The attempt that wants to know (``ModuleTestbench``'s outcome wrapper) opens a
tracker with :func:`track_launch`; the code at the real launch boundary calls
:func:`note_launch`. Nothing is inferred from a broader stage: a refusal before
the boundary (deck unreadable, output path occupied, executable missing)
leaves the tracker saying "not launched".
"""
from contextlib import contextmanager
from contextvars import ContextVar

_tracker = ContextVar('silk_launch_fact', default=None)


@contextmanager
def track_launch():
    state = {'launched': False, 'files': None}
    token = _tracker.set(state)
    try:
        yield state
    finally:
        _tracker.reset(token)


def note_launch(launched, files=None):
    """Record that the current attempt did (or, on a failed start, did not) launch.

    *files* are the concrete :class:`~silk.simulation.run_files.RunFiles` of an
    in-process backend that owns them (the behavioral engine); a SPICE launch
    passes none and its public files are read from the module afterwards.
    """
    state = _tracker.get()
    if state is not None:
        state['launched'] = bool(launched)
        if files is not None:
            state['files'] = files
