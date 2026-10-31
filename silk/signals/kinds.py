"""silk.signals.kinds — what KIND of Signal an input must be.

The vocabulary a metric declares its inputs against, and the honest
predicate that says whether a given :class:`~silk.signals.base.Signal`
can be one. Metric signatures declare these kinds; runtime guards inspect
the available signal observables and metric bindings use named slots.
See ``docs/schema-export.md`` for the signal-input kind catalog.

Why kinds exist
---------------
``signal_inputs`` in schema v1 publishes a bare parameter NAME, which
tells a consumer how many probes to bind and what to call them, but not
whether any probe it can offer will satisfy the slot. The measured
consequence (design §3.4) is silently wrong numbers: an ``.ac`` transfer
function handed to ``integrated_noise`` used to return a confident
number of volts (measured 1.25e+04 on a 100 V/V single pole) that was
never an RMS noise voltage. A kind says which probe fits which slot —
and, through :attr:`SignalKind.produced_by`, which slots NOTHING silk
can simulate will ever fit, so a consumer can refuse to offer the metric
instead of wiring it wrong.

A LEAF module on purpose
------------------------
stdlib + numpy, and **nothing from silk** — the same rule, for the same
reason, as :mod:`silk.metrics._guards`: ``silk.signals`` and
``silk.metrics`` import into each other, so a module either of them may
need mid-cycle must not reach back into either. The ``Signal``
annotations stay unevaluated under ``from __future__ import
annotations`` and every predicate reads the Signal by attribute, so a
caller may hand in any object with ``xlabel`` and ``y``. A future task
that wants the ``Annotated[Signal, KIND]`` aliases of design §4.2 should
either put them in :mod:`silk.signals.schema` (which already imports
``Signal``) or confirm that importing ``silk.signals.base`` from here
does not re-enter that cycle.

The three-valued answer, and why it is not a bool
-------------------------------------------------
A Signal does **not** carry a stamped kind. Whatever
produced it is gone by the time a metric sees it, so a predicate can
only read what is observable — the x-axis label, the dtype of ``y``.
For most kinds that is enough. For the ``xlabel='frequency'`` family it
is provably not: a noise density and a real-valued frequency response
(``psrr_db``, ``group_delay``) are the same two observables (design
§11) — a real ``y`` on the frequency axis, full stop — so nothing
observable can tell them apart.

So :meth:`SignalKind.check` returns a :class:`KindMatch`, not a bool:

``CONFLICTS``
    something observable RULES THE KIND OUT. A guard may refuse.
``MATCHES``
    the observables fit this kind and NO OTHER kind in this vocabulary
    shares them, so the identification is certain.
``UNDECIDABLE``
    the observables fit this kind, and they fit at least one other kind
    just as well. Nothing on the Signal separates them. A guard must
    ACCEPT — refusing here would reject legitimate input, and claiming
    ``MATCHES`` would be a lie.

:data:`NOISE_DENSITY` and :data:`FREQUENCY_RESPONSE` therefore both
answer ``UNDECIDABLE`` for any real ``y`` on the frequency axis. That is
the design's §11 finding expressed in code rather than a predicate that
looks like it works.

What the predicates read, stated once
-------------------------------------
**The x-axis label is the discriminator.** It is compared exactly, in
lower case, consistently with the metric input guards.
This is safe because ``from_rawfile``
lower-cases the label it assigns (for Xyce's ``TIME`` /
``FREQUENCY``). **The dtype of ``y`` is read only inside the frequency
family**, where the axis alone cannot separate ``ac`` from ``noise``:
complex samples are produced by ``.ac`` and by nothing else in this
vocabulary. No predicate reads length: an empty Signal is
``_check_nonempty``'s business, not a question of kind.

Adding a kind
-------------
The vocabulary is OPEN. A new kind is three lines and no redesign: write
its ``_match_*`` predicate, build the :class:`SignalKind`, and add it to
the :data:`SIGNAL_KINDS` tuple. Nothing branches on how many kinds there
are, and the tests are driven off the table rather than off a count, so
the only deliberate edits are the two that pin the table against the
approved design (``test_the_vocabulary_is_the_design_table`` and
``test_produced_by_is_the_design_table``) — one line each.

A kind may also be a REFINEMENT of another: nothing here assumes the
kinds partition the observables, and two kinds are free to accept the
same Signal. What a refinement costs is decidability — a Signal that
satisfies both a kind and its sub-kind is ``UNDECIDABLE`` for neither
only while some observable separates them.

**The third observable, deliberately unused.** ``Signal.type`` carries
the SIMULATOR's own word for the vector — MEASURED ``'voltage'`` and
``'current'`` on ngspice AND Xyce through the real load path (it is
``RawVector.type``, not the ``type=`` selector passed to
``load_signal``) — so a voltage/current split IS checkable by predicate,
cheaply, by the same ``getattr`` shape as :func:`_axis`.

**No kind ever splits on it.**
``average_power(sig_v, sig_i)`` genuinely has two slots that take
different things, but that distinction belongs to the metric's own
runtime guard — ``Signal.type`` is right there for ``_check_v_i`` to
read — not to a vocabulary a consumer wires probes against. Both slots
are plain ``transient``; the runtime guard checks their physical types.

Two traps are recorded here for whatever future kind DOES want to read
``type`` — not as a plan for one. A hand-built Signal carries whatever
its author passed (silk normalises nothing — ``'V'``, ``'voltage'``,
``'I'``, ``'signal'`` all appear in silk's own sources), so an
unrecognised type would have to be ``UNDECIDABLE``, not ``CONFLICTS``;
and a derived Signal composes its type (``v * i`` is
``'voltage*current'``).

produced_by is the one AUTHORED field in v2
-------------------------------------------
:attr:`SignalKind.produced_by` names the silk analyses that yield a
Signal of the kind. It is authored — design §1 — because the fact that
matters most is an ABSENCE: no silk analysis produces a power sweep or
a code-transition table, and an absence cannot be reflected from code
that does not exist. It is kept to ONE
table in THIS module, and :class:`tests.test_signal_kinds` asserts that
every kind with a non-empty ``produced_by`` really is reachable from the
analysis it names, loaded through the real path from a real simulator's
rawfile.

The entries are **silk's own analysis vocabulary** —
:data:`SILK_ANALYSES`, the same four strings ``@metric(domain=...)``
already exports. No mapping to any consumer's names is published here;
a consumer maps on its own end.

    An empty ``produced_by`` means no silk ANALYSIS hands one back. It
    does not always mean silk cannot make one: ``code_transitions`` is
    exactly what :func:`silk.metrics.converter.extract_code_transitions`
    derives from two ``tran`` Signals. The refusal rule still holds for
    a consumer that can only author analyses and bind their outputs —
    it cannot reach that Signal — which is why the table records the
    analysis, not the derivation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Tuple

import numpy as np

__all__ = [
    'KindMatch', 'SignalKind', 'SILK_ANALYSES',
    'ANY', 'TRANSIENT', 'FREQUENCY_RESPONSE', 'NOISE_DENSITY',
    'CODE_TRANSITIONS', 'POWER_SWEEP',
    'REFERENCE_WAVEFORM',
    'SIGNAL_KINDS', 'signal_kinds_meta', '_check_kind', 'SignalKindError',
]


# ---------------------------------------------------------------------------
# The analysis vocabulary
# ---------------------------------------------------------------------------

#: The PUBLISHED metric-schema analysis vocabulary — the same four strings
#: ``@metric(domain=...)`` declares and ``export_metric_schema()`` already
#: publishes, and what :class:`SignalKind`'s
#: ``produced_by`` validates against (line ~261 below). Deliberately NOT
#: :mod:`silk.signals.analysis_kind`'s vocabulary, which answers a different
#: question (which analysis produced a loaded rawfile plot) with a different
#: spelling for one entry: ``'transient'`` rather than ``'tran'``.
#:
#: ``'op'`` (:class:`~silk.simulation.analysis.Op`) is deliberately
#: NOT a member of this tuple, even though it is a real, selectable silk
#: analysis. Putting it here would make ``'op'`` a legal
#: ``SignalKind.produced_by`` / ``@metric(domain=...)`` value by
#: construction (the validator below checks against this exact tuple),
#: so the metric analysis vocabulary is exactly ``tran|ac|dc|noise``.
#: Admitting 'op' here would move the
#: digest-pinned exported metric schema. 'op' is selectable
#: (:mod:`silk.signals.analysis_kind`'s ``_KIND_ANALYSIS``/
#: ``SELECTABLE_ANALYSES``/``select_plot``, and the plot-name tables)
#: entirely outside this vocabulary; adding it here too is a
#: schema compatibility change.
SILK_ANALYSES: Tuple[str, ...] = ('tran', 'ac', 'dc', 'noise')


class KindMatch(Enum):
    """What a :class:`SignalKind` predicate could establish about a Signal.

    Three-valued because a Signal carries no stamped kind (see the limitations in docs/known_issues.md): see the module docstring. ``UNDECIDABLE`` is not a failure of
    the predicate, it is the truthful answer wherever two kinds share
    every observable.
    """

    MATCHES = 'matches'
    CONFLICTS = 'conflicts'
    UNDECIDABLE = 'undecidable'


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------

_TIME_AXIS = 'time'
_FREQ_AXIS = 'frequency'
_CODE_AXIS = 'code'
_POWER_AXIS = 'power'


def _axis(sig) -> str:
    """The Signal's x-axis label, verbatim (``''`` when it has none)."""
    return getattr(sig, 'xlabel', None) or ''


def _is_complex(sig) -> bool:
    """True when ``y`` holds complex samples.

    ``.ac`` is the only analysis in this vocabulary that produces them
    (MEASURED on the gen_e fixtures: ngspice and Xyce both load
    ``complex128`` for ``.ac`` and ``float64`` for ``tran`` / ``dc`` /
    ``noise``).
    """
    y = getattr(sig, 'y', None)
    if y is None:
        return False
    return bool(np.iscomplexobj(y))


def _on_axis(sig, axis: str) -> bool:
    return _axis(sig) == axis


# ---------------------------------------------------------------------------
# The kind object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalKind:
    """One kind of Signal a metric input can require.

    Parameters
    ----------
    name : str
        The kind's name as it is published in the schema.
    produced_by : tuple of str
        The silk analyses that yield a Signal of this kind, drawn from
        :data:`SILK_ANALYSES`. **Empty means nothing silk can simulate
        hands one back** — that is the fact a consumer refuses on, and
        the reason this is the one authored field in v2 (design §1).
    expectation : str
        What a Signal of this kind looks like, in the words a refusal
        message uses (``"xlabel='time'"``).
    predicate : callable
        ``Signal -> KindMatch``. Reads only what is observable on the
        Signal itself; see the module docstring.

    Raises
    ------
    ValueError
        If ``produced_by`` names something outside
        :data:`SILK_ANALYSES` — a typo there would silently make a kind
        unwireable (or wireable) for every consumer.
    """

    name: str
    produced_by: Tuple[str, ...]
    expectation: str
    predicate: Callable[[object], KindMatch] = field(repr=False, compare=False)

    def __post_init__(self):
        unknown = [a for a in self.produced_by if a not in SILK_ANALYSES]
        if unknown:
            raise ValueError(
                f"SignalKind({self.name!r}): produced_by names "
                f"{unknown!r}, which is not silk analysis vocabulary; "
                f"expected a subset of {list(SILK_ANALYSES)}"
            )

    def check(self, sig) -> KindMatch:
        """Say what the observables on *sig* establish about this kind.

        Returns a :class:`KindMatch` — never a bool, because for part of
        the vocabulary the honest answer is ``UNDECIDABLE``.
        """
        return self.predicate(sig)

    def __str__(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# The kinds (seven today — see "Adding a kind" in the module docstring)
# ---------------------------------------------------------------------------

def _match_any(sig) -> KindMatch:
    """Every Signal is of kind ``any``.

    Not a lie of convenience: ~15 primitives (``rms``, ``average``,
    ``value_at``, ``crossing``, …) are documented as deliberately
    axis-agnostic (``silk.metrics.waveform`` and ``silk.metrics.statistics``).
    A vocabulary that could not say "any 1-D Signal" would force one of the other six
    kinds onto them.
    """
    return KindMatch.MATCHES


def _match_transient(sig) -> KindMatch:
    """A trace over time. Decidable: no other kind uses the time axis.

    The dtype is deliberately not read. The axis is what makes a trace a
    transient, and this keeps the predicate exactly as strict as
    ``_check_transient`` in :mod:`silk.metrics.waveform`.
    """
    return KindMatch.MATCHES if _on_axis(sig, _TIME_AXIS) else KindMatch.CONFLICTS


def _match_frequency_response(sig) -> KindMatch:
    """A small-signal transfer over frequency.

    Complex samples settle it — ``.ac`` is the only producer of complex
    ``y``. Real samples on a frequency axis are UNDECIDABLE: ``psrr_db``
    and ``group_delay`` return real-valued frequency responses, and a
    ``.noise`` density is real on the same axis, so nothing observable
    separates them (design §11, left open there deliberately).
    """
    if not _on_axis(sig, _FREQ_AXIS):
        return KindMatch.CONFLICTS
    return KindMatch.MATCHES if _is_complex(sig) else KindMatch.UNDECIDABLE


def _match_noise_density(sig) -> KindMatch:
    """A noise density over frequency (V/√Hz, A/√Hz).

    Complex samples rule it out — a density is a real magnitude, and
    ``load_noise`` loads real (MEASURED, both backends). Real samples on
    a frequency axis are UNDECIDABLE: a real-valued frequency response
    looks identical.
    """
    if not _on_axis(sig, _FREQ_AXIS) or _is_complex(sig):
        return KindMatch.CONFLICTS
    # A density is never in dB: a Signal that SAYS it is (silk's own
    # noise_figure_vs_freq result, type 'noise-figure') is ruled out.
    if str(getattr(sig, 'ylabel', '')).strip().lower() == 'db' \
            or getattr(sig, 'type', None) == 'noise-figure':
        return KindMatch.CONFLICTS
    return KindMatch.UNDECIDABLE


def _match_code_transitions(sig) -> KindMatch:
    """An ADC transition-level table: x = code, y = transition voltage.

    Decidable, and grounded in a real producer even though no ANALYSIS
    yields one: :func:`silk.metrics.converter.extract_code_transitions` builds
    exactly this Signal from two ``tran`` captures, and ``inl`` / ``dnl``
    / ``is_monotonic`` / ``missing_codes`` already refuse anything else
    on the same axis test (``silk/metrics/converter.py``).
    """
    return KindMatch.MATCHES if _on_axis(sig, _CODE_AXIS) else KindMatch.CONFLICTS


def _match_power_sweep(sig) -> KindMatch:
    """x = input power in dBm, y = output power in dBm.

    Decidable on the axis, but note what the predicate is grounded in:
    nothing in silk produces a power sweep, so this reflects ``p1db``'s
    own documented input contract (``silk/metrics/rf.py``) rather
    than an observed producer. A consumer refuses ``p1db`` on the empty
    :attr:`~SignalKind.produced_by`, not on this.
    """
    return KindMatch.MATCHES if _on_axis(sig, _POWER_AXIS) else KindMatch.CONFLICTS



def _match_reference_waveform(sig) -> KindMatch:
    """A target waveform the USER authored, to compare a measurement against.

    ``error_norm(sig, goal)``'s second slot. It is the same
    structural case as :data:`POWER_SWEEP` — **no silk analysis hands
    one back**, because a
    goal is something a designer writes down, not something a probe
    yields — and that empty :attr:`~SignalKind.produced_by` is the whole
    point: a consumer that can only author analyses and bind their
    outputs must NOT be offered ``error_norm`` with two probes wired into
    it, which would compute the difference between two measurements and
    label it an error against a target.

    Never returns ``MATCHES``. A reference waveform lives on whatever
    axis the measurement lives on — ``error_norm`` is documented
    axis-agnostic and interpolates ``goal`` onto ``sig.x`` — so no
    observable can establish that a given Signal *is* a goal.

    One observable rules it OUT, and it is read from ``error_norm``'s own
    body rather than invented here: the metric does ``np.asarray(goal.y,
    dtype=float)``, and numpy does not raise on that — it DISCARDS the
    imaginary part (``ComplexWarning``) and returns a confident norm
    computed from half the data. A complex ``y`` (an ``.ac`` response)
    therefore CONFLICTS: it is precisely the input that would produce a
    silently wrong number.
    """
    return KindMatch.CONFLICTS if _is_complex(sig) else KindMatch.UNDECIDABLE


ANY = SignalKind(
    name='any',
    produced_by=('tran', 'ac', 'dc', 'noise'),
    # Axis-agnostic vocabulary: each metric defines its sample contract.
    # Most generic scalar statistics require real projections (TASK-490);
    # RMS, complex-preserving transforms and error norms accept complex data.
    # REVIEWER BINDING (2026-09-27, TASK-330.2, commit 178250ed): 'op' is
    # deliberately NOT added here even though a single-point Op() signal
    # is a 1-D Signal like any other. This tuple is a literal design
    # vocabulary on purpose -- the schema validates a metric's analysis/domain
    # as exactly tran|ac|dc|noise, and widening produced_by would widen
    # the exported metric schema (kinds -> produced_by feeds
    # export_metric_schema()) underneath it. 'op' is selectable
    # (analysis_kind.py's SELECTABLE_ANALYSES, plot-name tables; NOT
    # SILK_ANALYSES) without being a
    # metric-schema-visible analysis domain.
    expectation='any 1-D Signal; most scalar metrics require real samples '
                '(choose .real(), .mag(), or .imag()); complex-preserving '
                'transforms, RMS and error norms accept complex samples',
    predicate=_match_any,
)

TRANSIENT = SignalKind(
    name='transient',
    produced_by=('tran',),
    expectation="xlabel='time'",
    predicate=_match_transient,
)

FREQUENCY_RESPONSE = SignalKind(
    name='frequency_response',
    produced_by=('ac',),
    expectation="xlabel='frequency', complex y",
    predicate=_match_frequency_response,
)

NOISE_DENSITY = SignalKind(
    name='noise_density',
    produced_by=('noise',),
    expectation="xlabel='frequency', real y in V/sqrt(Hz) or A/sqrt(Hz)",
    predicate=_match_noise_density,
)

CODE_TRANSITIONS = SignalKind(
    name='code_transitions',
    produced_by=(),
    expectation="xlabel='code', x = code index, y = transition voltage",
    predicate=_match_code_transitions,
)

POWER_SWEEP = SignalKind(
    name='power_sweep',
    produced_by=(),
    expectation="xlabel='power', x = p_in in dBm, y = p_out in dBm",
    predicate=_match_power_sweep,
)

REFERENCE_WAVEFORM = SignalKind(
    name='reference_waveform',
    produced_by=(),
    expectation="a user-authored target waveform on the measurement's own "
                "axis, with real y; no silk analysis produces one",
    predicate=_match_reference_waveform,
)


#: The whole vocabulary, name → kind. The AUTHORED table of design §1: one
#: module, nothing else authored anywhere. Seven entries TODAY — the set is
#: open, and nothing (here or in the tests) is written against its size.
SIGNAL_KINDS: Dict[str, SignalKind] = {
    k.name: k for k in (
        ANY, TRANSIENT, FREQUENCY_RESPONSE, NOISE_DENSITY,
        CODE_TRANSITIONS, POWER_SWEEP,
        REFERENCE_WAVEFORM,
    )
}


def signal_kinds_meta() -> dict:
    """The table in the shape a schema publishes it — ``{name:
    {'produced_by': [...]}}``, JSON-safe.

    Published under ``_meta.signal_kinds`` so a consumer can evaluate the one-line
    refusal rule — offer a metric iff EVERY one of its
    ``signal_inputs[].kind`` has a ``produced_by`` that intersects the
    analyses the consumer can author.
    """
    return {name: {'produced_by': list(kind.produced_by)}
            for name, kind in SIGNAL_KINDS.items()}


class SignalKindError(ValueError):
    """A Signal handed to a metric CONFLICTS with the kind its slot
    requires — the observables rule it out, e.g. a transient trace where a
    frequency response is required.

    A ``ValueError`` subclass, the same convention as
    :class:`~silk.signals.schema.DuplicateMetricError`: existing ``except
    ValueError`` handlers around :func:`_check_kind` keep catching it
    unchanged, and its message is unchanged too. ``silk.circuit``'s
    evaluation loop re-raises this as
    :class:`~silk.exceptions.MetricBindingError` so the failure raises from
    ``run()`` instead of landing as a report string.
    """


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------

def _check_kind(sig, kind: SignalKind, metric: str, slot: str = 'sig') -> KindMatch:
    """Refuse *sig* if its observables rule out *kind*; report what was
    established either way.

    Parameters
    ----------
    sig : Signal
        The Signal a metric was handed.
    kind : SignalKind
        The kind that metric's slot requires.
    metric : str
        The metric's name, for the message — the function the CALLER
        invoked, following the convention of
        :func:`silk.metrics._guards._check_nonempty`.
    slot : str, optional
        The parameter name to blame, for a metric with more than one
        Signal input. Defaults to ``'sig'``, which is what most of them
        call it.

    Returns
    -------
    KindMatch
        ``MATCHES`` when the kind is established, ``UNDECIDABLE`` when
        the input is CONSISTENT with the kind but nothing observable
        separates it from another kind. **The return value is the point
        of the function**: an ``UNDECIDABLE`` pass is not a verified
        input, and a caller that reports otherwise is overclaiming.

    Raises
    ------
    SignalKindError
        A ``ValueError`` subclass. Raised only on ``CONFLICTS`` —
        something observable rules the kind out. Never on
        ``UNDECIDABLE``: refusing there would reject legitimate input on
        a guess.
    """
    result = kind.check(sig)
    if result is KindMatch.CONFLICTS:
        detail = f'xlabel={_axis(sig)!r}'
        if _is_complex(sig):
            detail += ', complex y'
        raise SignalKindError(
            f"{metric}: {slot} must be a {kind.name} Signal "
            f"({kind.expectation}), got {detail}"
        )
    return result
