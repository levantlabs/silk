"""NgspiceSignal — ngspice-specific waveform signal."""

import numpy as np

from silk.signals.base import Signal
from silk.exceptions import SimulationError


# Canonical x-axis labels the rest of silk is written against (every axis
# check in silk.metrics compares ``xlabel`` with exact lowercase equality).
# The x-axis is NOT detected by name — see from_rawfile: in a nutmeg rawfile
# (ngspice AND Xyce) the independent variable is always vector 0, whatever it
# is called. A `.dc` sweep names its axis 'temp-sweep' / 'v(v-sweep)'
# (ngspice) or 'sweep' (Xyce), none of which a name whitelist would catch —
# hence the vector-0 rule. Xyce's ``.print ... format=raw`` writes the axis
# name upper-cased ('TIME' / 'FREQUENCY'); from_rawfile case-folds the LABEL it
# assigns so both simulators present these canonical spellings (TASK-87).
_XAXIS_NAMES = {'time', 'frequency'}


class NgspiceSignal(Signal):
    """Waveform signal loaded from an ngspice simulation.

    The SPICE-family signal type. ``SpiceBackend.load_signal`` calls
    :meth:`from_rawfile` directly (for both ngspice and Xyce rawfiles), so no
    simulator-specific dispatch lives in the netlist/signal layer.

    Parameters
    ----------
    name : str
        Signal name, e.g. ``'v(out)'``.
    type : str
        Physical type string from the rawfile, e.g. ``'voltage'``.
    xlabel : str
        X-axis label (``'time'`` for transient, ``'frequency'`` for AC).
        :meth:`from_rawfile` case-folds vector 0's name, so the label is
        the same lowercase spelling from ngspice and Xyce alike.
    ylabel : str
        Y-axis label.
    xvalues : array-like
        Independent-axis values (time or frequency array).
    yvalues : array-like
        Dependent-axis values.
    """

    def __init__(self, name, type, xlabel, ylabel, xvalues, yvalues):  # noqa: A002
        super().__init__(name, type, xlabel, ylabel, xvalues, yvalues)

    @classmethod
    def from_rawfile(cls, rawdata, signal_name: str, signal_type: str = 'V',
                     plot=None):
        """Extract a single signal from an ngspice rawfile result object.

        Parameters
        ----------
        rawdata : silk.signals.rawfile.RawFile
            The parsed rawfile returned by
            ``silk.signals.rawfile.read_rawfile``.
        signal_name : str
            The bare signal name to find, e.g. ``'out'`` (without the
            ``v(...)`` wrapper).
        signal_type : str
            ``'V'`` for voltage, ``'I'`` for current.  Case-insensitive.
        plot : silk.signals.rawfile.RawPlot, optional
            Read this plot of a multi-analysis rawfile instead of *rawdata*'s
            own default (``current_plot``, the LAST plot in the file).
            ``None`` — the default — is the path every caller took previously and is the only path a single-plot file ever takes; the
            selection itself is made once, in
            ``silk.simulation.backend.select_plot``, and never here.
            ``RawFile`` proxies ``nvars``/``vectors`` to its current plot and
            a ``RawPlot`` exposes the same two attributes, so the body below
            reads one object either way.

        Returns
        -------
        NgspiceSignal
            ``x`` is vector 0 of the plot (the independent axis) and ``y``
            the matched vector. For an operating-point plot (``.op`` — one
            point and no axis; ``silk.signals.analysis_kind.rawfile_analysis_kind``
            says ``'op'``) the Signal is ONE sample
            with ``x = [0.0]`` and ``xlabel = 'op'``, the same shape from
            either simulator, and the match runs over every vector: on
            ngspice vector 0 is the first node itself, so the vector-0 rule
            used to raise "not found" for that node and hand every other
            node ``x = v(<first node>)`` as its axis. Axis-checking
            metrics reject the ``'op'`` label, as they should.

        Raises
        ------
        SimulationError
            If the requested signal is not found in the rawfile.
        """
        from silk.signals.analysis_kind import ANALYSIS_OP, rawfile_analysis_kind

        # The one substitution: everything below reads `rawdata` through this
        # name, so a selected plot and a whole rawfile take the identical path.
        rawdata = rawdata if plot is None else plot

        xlabel = None
        xvals  = None
        ylabel = None
        yvals  = None
        ysig_type = None

        # X-axis: vector 0 is the independent variable in a nutmeg rawfile,
        # whatever it is named. This is what makes DC sweeps (axis 'temp-sweep'
        # / 'v(v-sweep)' / 'sweep') readable, and it works for ngspice and Xyce
        # alike — neither a name whitelist nor a suffix rule covers all cases.
        #
        # The LABEL is case-folded: ngspice already lower-cases every name, but
        # Xyce writes 'TIME' / 'FREQUENCY', and every axis check in silk.metrics
        # compares xlabel with exact lowercase equality — so a verbatim label
        # made every transient/AC metric reject Xyce output (TASK-87). Only the
        # Signal's label is folded; RawVector.name stays verbatim (the reader
        # contract — the rawfile viewer shows names as the simulator wrote them).
        is_op = rawfile_analysis_kind(rawdata) == ANALYSIS_OP
        # Y-axis: ONE collector gathers every candidate row of the selected
        # plot before anything is chosen (TASK-527).
        first = 0 if is_op else 1
        candidates = collect_vector_candidates(rawdata, signal_name,
                                               signal_type, first=first)

        # Distinguish a missing x-axis from a missing y-signal: reporting the
        # former as the latter is what historically mis-diagnosed the DC-sweep
        # bug as a "constants plot" problem.
        if not is_op and rawdata.nvars == 0:
            names = [rawdata.vectors[i].name for i in range(rawdata.nvars)]
            raise SimulationError(
                f'No x-axis vector found: the rawfile has no vectors '
                f'(vector 0 is the independent axis). Vectors present: {names}'
            )
        if not candidates:
            raise SimulationError(
                f'Signal {signal_name!r} (type={signal_type!r}) not found '
                f'in rawfile.'
            )
        if len(candidates) > 1:
            from silk.exceptions import AmbiguousSignalError
            # A plot= call hands a RawPlot; a default call hands the RawFile,
            # whose current plot is the one read (selection is unchanged).
            named = getattr(rawdata, 'current_plot', rawdata)
            plot_name = getattr(named, 'plotname', None) or '(unnamed plot)'
            names = [rawdata.vectors[i].name for i in candidates]
            raise AmbiguousSignalError(
                f'Signal {signal_name!r} (type={signal_type!r}) is ambiguous '
                f'in plot {plot_name!r}: it matches {len(names)} distinct '
                f'vectors {names}. Write the exact vector name of the one you '
                f'mean, in the simulator spelling (instance parts as <k>), '
                f'or rename the net.'
            )
        # Only a unique, valid request acquires samples. Failed lookups must
        # leave lazy readers untouched, including their independent axis.
        if is_op:
            xlabel = ANALYSIS_OP
            xvals = np.zeros(1)
        else:
            xlabel = rawdata.vectors[0].name.lower()
            xvals = rawdata.vectors[0].data
        if xvals is None:
            names = [rawdata.vectors[i].name for i in range(rawdata.nvars)]
            raise SimulationError(
                f'No x-axis vector found: the rawfile has no vectors '
                f'(vector 0 is the independent axis). Vectors present: {names}'
            )
        vec = rawdata.vectors[candidates[0]]
        # A bare name (noise spectra) labels itself; a wrapped one by its head.
        ylabel = (vec.name if '(' not in vec.name else vec.name.split('(')[0])
        yvals = vec.data
        ysig_type = vec.type

        return cls(signal_name, ysig_type, xlabel, ylabel, xvals, yvals)


# collect_vector_candidates lives in silk.vector_match (the shared matcher).
from silk.vector_match import collect_vector_candidates  # noqa: E402,F401
