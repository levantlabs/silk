"""Classify what kind of analysis a loaded rawfile plot holds.

One classifier, shared by every consumer that has to know whether the
"voltages" in a plot are node voltages over time, a DC transfer, a
small-signal transfer ratio or a noise density: STR-001
(``silk.lint.rules.stress``) declines the small-signal kinds, and the
report layer (``silk.circuit.report``) summarises them by magnitude under
frequency-axis field names. Keeping the rule here, lint-free and
numpy-only, is what lets both import it without a cycle and what stops
them disagreeing about the same rawfile.

The classification is keyed on the plot the reader actually returns —
``RawFile.current_plot``, the last plot in the file, which is the one
``NgspiceSignal.from_rawfile`` takes ``vectors[0]`` from — and NOT on the
testbench's ``sim_analysis_dict``: a testbench may declare several analyses
while the rawfile holds the results of one, so only the loaded data can
say what a consumer is about to read.

A rawfile can legitimately hold SEVERAL analyses' plots, so
this module also answers the selection question — :func:`plot_silk_analysis`,
"which declared analysis produced THIS plot", in silk's own ``'tran'/'ac'/
'dc'/'noise'`` vocabulary. It takes one plot and never a whole rawfile,
precisely so that no caller can reach a plot by guessing.
"""
from typing import Optional

import numpy as np

__all__ = [
    'ANALYSIS_TRANSIENT', 'ANALYSIS_AC', 'ANALYSIS_NOISE', 'ANALYSIS_DC',
    'ANALYSIS_OP', 'SMALL_SIGNAL_KINDS', 'SELECTABLE_ANALYSES',
    'rawfile_analysis_kind', 'plot_silk_analysis',
    '_spice_results_are_small_signal',
]

ANALYSIS_TRANSIENT = 'transient'
ANALYSIS_AC = 'ac'
ANALYSIS_NOISE = 'noise'
ANALYSIS_DC = 'dc'
ANALYSIS_OP = 'op'

# Kinds whose vectors are not node voltages: an AC "V" is a transfer ratio
# and a noise "V" is a density in V/sqrt(Hz).
SMALL_SIGNAL_KINDS = frozenset({ANALYSIS_AC, ANALYSIS_NOISE})

# HEURISTIC (axis-based): a rawfile plot whose independent variable is named
# for frequency, or whose data is complex, holds small-signal or spectral
# results (.ac, .noise) rather than node voltages over time. Verified against
# real ngspice: transient plots have a real 'time' axis, .ac a complex
# 'frequency' axis, .noise a real 'frequency' axis, .dc a real 'v(v-sweep)'
# axis (TASK-81); Xyce names the DC axis 'sweep' (TASK-74 fixtures).
_FREQ_AXIS_TOKEN = 'freq'
_TIME_AXIS_NAME = 'time'

# An operating point has NO independent variable, so the vector-0 rule
# cannot see it: measured on the TASK-102 fixtures (resistive divider, .op),
# ngspice-42 writes plotname 'Operating Point' with the FIRST NODE ('v(in)')
# as vector 0, and Xyce 7.10 — which has no '.print op' — writes the .op
# result through '.print dc format=raw' under plotname 'DC transfer
# characteristic' with a placeholder 'sweep' vector 0 holding 0.0 (not the
# source value). Both hold exactly one point. So the plotname is the key
# where the simulator says it (ngspice), and the point count where it does
# not (Xyce): a one-point real plot is an operating point whatever its
# vector 0 is called. That makes a one-point '.dc V1 1.5 1.5 1' classify
# 'op' too — nothing was swept, and Xyce's file for it is byte-for-byte the
# .op shape with 1.5 in the placeholder — while any swept DC plot keeps 'dc'.
_OP_PLOTNAME = 'operating point'


def _plot_point_count(plot, vectors):
    """The plot's point count: the reader's ``npoints`` when the plot
    carries one, else the length of vector 0's data, else ``None`` (a stub
    with names only — those classify by axis name alone)."""
    n = getattr(plot, 'npoints', None)
    if n is not None:
        return n
    data = getattr(vectors[0], 'data', None)
    return None if data is None else len(data)


def _plot_is_complex(plot, vectors) -> bool:
    """A complex plot is small-signal by construction (nutmeg stores .ac that
    way); the header flag is authoritative and ``real`` is True for ordinary
    transient/dc plots. The dtype walk backs it up for a plot object that
    carries no flag (hand-built result stubs), so a complex vector can never
    be summarised as a real one on the strength of a missing attribute.
    Unread rawfile columns expose their format without loading samples;
    materialized or caller-replaced arrays still report their actual dtype."""
    if getattr(plot, 'real', True) is False:
        return True
    for vec in vectors:
        complex_data = getattr(vec, '_is_complex', None)
        if complex_data is None:
            data = getattr(vec, 'data', None)
            complex_data = data is not None and np.iscomplexobj(data)
        if complex_data:
            return True
    return False


def rawfile_analysis_kind(raw) -> Optional[str]:
    """Return which analysis produced the current plot of *raw*.

    Parameters
    ----------
    raw : silk.signals.rawfile.RawFile or a compatible stub
        Read through ``current_plot`` when present, else *raw* itself is
        taken as the plot (``vectors`` / ``real``).

    Returns
    -------
    str or None
        ``'ac'`` for a complex plot, ``'op'`` for a real plot whose
        plotname is ``Operating Point`` or that holds exactly one point
        (an operating point has no axis: vector 0 is a node on ngspice and
        a placeholder ``sweep`` on Xyce, see :data:`_OP_PLOTNAME`),
        ``'noise'`` for a real plot on a ``freq``-named axis,
        ``'transient'`` for a real plot on a ``time`` axis, ``'dc'`` for a
        real plot on any other axis (``v(v-sweep)``, ``sweep``,
        ``temp-sweep`` — the vector-0 rule, never a name whitelist), or
        ``None`` when there is nothing to classify (no rawfile, no plot, an
        empty plot). A caller that reads only certain kinds must treat
        ``None`` as "unclassifiable", not as any kind.

    Axis names are compared case-folded, the same fold applies to
    ``Signal.xlabel`` for Xyce's upper-cased ``TIME`` / ``FREQUENCY``.
    """
    if raw is None:
        return None
    plot = getattr(raw, 'current_plot', None) or raw
    vectors = getattr(plot, 'vectors', None) or []
    if not len(vectors):
        return None
    if _plot_is_complex(plot, vectors):
        return ANALYSIS_AC
    plotname = (getattr(plot, 'plotname', '') or '').lower()
    if plotname == _OP_PLOTNAME or _plot_point_count(plot, vectors) == 1:
        return ANALYSIS_OP
    axis_name = (getattr(vectors[0], 'name', '') or '').lower()
    if _FREQ_AXIS_TOKEN in axis_name:
        return ANALYSIS_NOISE
    if axis_name == _TIME_AXIS_NAME:
        return ANALYSIS_TRANSIENT
    return ANALYSIS_DC


def _spice_results_are_small_signal(module) -> bool:
    """True when this result carries ONLY AC/spectral plots — i.e. there is no
    node-voltage-over-time data in it for a consumer to read.

    ``rawfile_analysis_kind`` applied to ``module.simResultsAll``.
    Unclassifiable results (no rawfile object, an empty plot) return
    ``False``: STR-001 then behaves exactly as it did before this guard
    existed rather than silently declining to check.

    A rawfile may hold several plots, and the plot
    ``load_signal`` reads is then CHOSEN rather than fixed — so the question
    is asked of every plot, not only of the current one. A tran+ac run holds
    real node voltages whichever analysis ngspice ran last, and a consumer
    that declines it would decline on the strength of plot ORDER. A
    single-plot file takes the current-plot path unchanged.
    """
    raw = getattr(module, 'simResultsAll', None)
    plots = getattr(raw, 'plots', None)
    if isinstance(plots, (list, tuple)) and len(plots) > 1:
        return all(rawfile_analysis_kind(p) in SMALL_SIGNAL_KINDS
                   for p in plots)
    return rawfile_analysis_kind(raw) in SMALL_SIGNAL_KINDS


# ---------------------------------------------------------------------------
# Which DECLARED analysis a plot belongs to (TASK-212)
# ---------------------------------------------------------------------------

#: What each simulator calls the plot an analysis leaves behind, in silk's own
#: analysis vocabulary (``SILK_ANALYSES`` in silk/signals/kinds.py — ``'tran'``, not
#: ``'transient'``). MEASURED on ngspice-42 (a four-analysis RC deck written
#: with ``write <path> tran1.all ac1.all dc1.all noise1.all``) and on Xyce 7.10
#: (``.print <kind> format=raw`` fixtures): both write these names,
#: and Xyce reuses ``DC transfer characteristic`` for an operating point.
#: Compared case-folded, and a prefix match on the noise family because
#: ngspice writes ``Noise Spectral Density Curves`` for ``noise1`` and
#: ``Integrated Noise`` for ``noise2``.
_PLOTNAME_ANALYSIS = {
    'transient analysis': 'tran',
    'ac analysis': 'ac',
    'dc transfer characteristic': 'dc',
}

#: Translation from :func:`rawfile_analysis_kind`'s vocabulary, which answers
#: a different question with different strings, into silk's analysis words.
#: ``ANALYSIS_OP: 'op'`` is deliberately here on ngspice's own
#: terms only: an explicit ``op`` leaves a plot NAMED ``Operating Point``,
#: which :func:`plot_silk_analysis` recognises before it ever reaches
#: :func:`rawfile_analysis_kind`'s point-count fallback. Xyce's Op()-
#: synthesized one-point ``.dc`` sweep is plotname-indistinguishable from a
#: genuine one-point ``DC()`` (both are ``'DC transfer characteristic'``,
#: see :data:`_PLOTNAME_ANALYSIS`) — :func:`rawfile_analysis_kind`'s own
#: docstring already rules that the plotname wins there, so a Xyce Op()
#: plot classifies as ``'dc'``, not ``'op'``, by design. A test that needs
#: ``analysis='op'`` selection on Xyce specifically is a concern.
_KIND_ANALYSIS = {
    ANALYSIS_TRANSIENT: 'tran',
    ANALYSIS_AC: 'ac',
    ANALYSIS_DC: 'dc',
    ANALYSIS_NOISE: 'noise',
    ANALYSIS_OP: 'op',
}

#: The analysis words a caller may pass as ``analysis=``. DERIVED from the
#: translation above rather than authored again: it is, by construction, the
#: set of words :func:`plot_silk_analysis` can return. It equals
#: ``SILK_ANALYSES`` (silk/signals/kinds.py) PLUS ``'op'``: ``'op'`` is a
#: real, selectable silk analysis, but is deliberately kept OUT of
#: ``SILK_ANALYSES`` (the published metric-schema vocabulary) so it can never
#: become a legal ``SignalKind.produced_by`` / ``@metric(domain=...)`` value
#: by construction; widening the schema vocabulary to include it is a
#: schema compatibility change. ``SILK_ANALYSES`` lives in
#: that module — a deliberate LEAF with a pinned importer list, and reading
#: it from here to validate a keyword is not worth widening that list
#: (``tests/test_multi_analysis_plots.py`` asserts
#: ``SELECTABLE_ANALYSES == SILK_ANALYSES + ('op',)``).
SELECTABLE_ANALYSES = tuple(_KIND_ANALYSIS.values())


def plot_silk_analysis(plot) -> Optional[str]:
    """Which DECLARED analysis produced *plot*, in silk's analysis vocabulary.

    Parameters
    ----------
    plot : silk.signals.rawfile.RawPlot or a compatible stub
        ONE plot — not a :class:`~silk.signals.rawfile.RawFile`. (A RawFile
        would answer for its current plot only, which is the very guess explicit plot selection avoids.)

    Returns
    -------
    str or None
        ``'tran'``, ``'ac'``, ``'dc'``, ``'noise'`` or ``'op'`` (an
        ngspice ``Operating Point`` plot, a Xyce Op()
        deck's one-point ``.dc`` sweep still reads as ``'dc'``, see
        :data:`_KIND_ANALYSIS`), else ``None`` for the synthetic
        ``constants`` plot an aborted run leaves behind, or a plot there
        is nothing to classify in.

    The simulator's own plotname is asked FIRST, because it is a statement
    rather than an inference and it is exact for every deck silk emits.
    Xyce Step Analysis prefixes are resolved by their trailing analysis name.
    The axis/dtype classifier :func:`rawfile_analysis_kind` is the fallback, which
    is what keeps a hand-built stub or an unfamiliar spelling readable. The
    two only disagree where the classifier cannot win on observables alone —
    a one-point ``.dc`` sweep reads as ``'op'`` by axis and as ``'dc'`` by
    plotname, and the plotname is right.
    """
    if plot is None:
        return None
    plotname = (getattr(plot, 'plotname', '') or '').strip().lower()
    if plotname.startswith('step analysis:'):
        # Xyce samples prefix the actual analysis name with step parameters.
        # Only the trailing name identifies the analysis, not parameter text.
        for name in _PLOTNAME_ANALYSIS:
            if plotname.endswith(' ' + name):
                plotname = name
                break
    named = _PLOTNAME_ANALYSIS.get(plotname)
    if named is not None:
        return named
    if 'noise' in plotname:
        return 'noise'
    return _KIND_ANALYSIS.get(rawfile_analysis_kind(plot))
