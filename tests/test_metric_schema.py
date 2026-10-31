"""Tests for signal-metric schema export (silk.signals.schema)."""
import inspect
import json
import subprocess
import sys
import typing

import numpy as np
import pytest

import silk.metrics as M
from silk.signals.base import Signal
from silk.signals import schema as S
from silk.signals.kinds import SIGNAL_KINDS, SignalKind, TRANSIENT
from silk.signals.schema import (
    AnySignal, FrequencyResponse, NoiseDensity,
    ReferenceWaveform, Transient, UnsupportedSignalKindError,
    export_metric_schema, export_metric_schema_json, metric_schema, get_metric,
)


@pytest.fixture(scope='module')
def schema():
    return export_metric_schema()


def test_catalog_registered(schema):
    for name in ('dc_gain_dB', 'peak_gain_dB', 'gain_at_freq', 'bandwidth_3dB',
                 'unity_gain_freq', 'phase_at_freq', 'phase_margin',
                 'gain_margin', 'cmrr_dB', 'integrated_noise'):
        assert name in schema, name
    assert schema['_meta']['metric_count'] == 80   # TASK-414: noise_figure_at_freq, band_noise_figure; TASK-417: period_jitter_pp, cycle_to_cycle_jitter_pp; TASK-412: three removed; TASK-379: variance; TASK-397: oip3_dbm, iip3_dbm; TASK-386: relative_error_norm; TASK-399 offset/gain/full-scale wrappers; TASK-233: 61 + eight wrappers; TASK-323.3: final_value; TASK-369: evm_percent; TASK-363: comparator_prop_delay removed; TASK-366: period_zero_cross -> average_period + average_frequency; TASK-226.20: crossing_jitter_rms


def test_meta_block(schema):
    m = schema['_meta']
    # '2.0' — signal_inputs[] changed shape, from names to {name, kind}
    # objects (TASK-196, design §8.1). Everything else v2 added is additive.
    assert m['format_version'] == '2.0'
    assert m['source_type'] == 'metrics'
    # the one AUTHORED table of v2, published once per export
    assert set(m['signal_kinds']) == set(SIGNAL_KINDS)


def test_domain_and_unit(schema):
    assert schema['dc_gain_dB']['domain'] == 'ac'
    assert schema['dc_gain_dB']['unit'] == 'dB'
    assert schema['bandwidth_3dB']['unit'] == 'Hz'
    assert schema['phase_margin']['unit'] == 'deg'
    assert schema['integrated_noise']['domain'] == 'noise'


def test_gain_at_freq_reflects_actual_unit_not_assumption(schema):
    """silk returns linear magnitude — schema says V/V, not dB. This is the
    value of reflecting silk's truth rather than consumers guessing."""
    assert schema['gain_at_freq']['unit'] == 'V/V'


def test_scalar_params_separated_from_signals(schema):
    gaf = schema['gain_at_freq']
    assert gaf['signal_inputs'] == [{'name': 'sig',
                                     'kind': 'frequency_response'}]
    assert [p['name'] for p in gaf['parameters']] == ['freq', 'interpolate']
    freq = gaf['parameters'][0]
    assert freq['unit'] == 'Hz'
    assert freq['required'] is True          # no default → mandatory


@pytest.mark.parametrize('name', ['eye_height', 'eye_width'])
def test_eye_wrappers_use_required_scalar_threshold_without_schema_extension(schema, name):
    params = {p['name']: p for p in schema[name]['parameters']}
    assert params['threshold'] == {
        'name': 'threshold', 'type': 'float', 'unit': '[y]',
        'default': None, 'required': True, 'optional': False, 'choices': None,
    }
    assert params['auto_center']['default'] is True
    assert schema['_meta']['format_version'] == '2.0'
    assert schema['_meta']['metric_count'] == 80
    assert set(schema['_meta']['signal_kinds']) == set(SIGNAL_KINDS)


def test_multi_signal_metric(schema):
    """cmrr_dB takes two Signals plus the optional measurement frequency
    (2026-08-23 — explicit instead of the accidental sample-0)."""
    cm = schema['cmrr_dB']
    assert cm['signal_inputs'] == [
        {'name': 'vdiff', 'kind': 'frequency_response'},
        {'name': 'vcm', 'kind': 'frequency_response'},
    ]
    assert [p['name'] for p in cm['parameters']] == ['freq']
    assert cm['parameters'][0]['required'] is False   # has a default


def test_two_scalar_params(schema):
    inb = schema['integrated_noise']
    assert inb['signal_inputs'] == [{'name': 'sig', 'kind': 'noise_density'}]
    assert [p['name'] for p in inb['parameters']] == ['f1', 'f2']
    assert all(p['unit'] == 'Hz' for p in inb['parameters'])


def test_noise_density_at_freq_unit_is_the_inputs_own_unit(schema):
    """The result is exactly one ``sig.y`` sample (V/sqrt(Hz) for a
    voltage-noise probe, A/sqrt(Hz) for a current-noise probe), so a
    hard-coded 'V/sqrt(Hz)' would be wrong on a current-noise Signal — the
    same defect TASK-179 found and fixed for ``error_norm`` ("the unit
    '[y]' ... is NOT volts: it is the input's y unit, so a current input
    gives amps"). ``noise_density_at_freq`` declares ``'[y]'`` for the same
    reason, even though its ``sig`` input is kind='noise_density', not
    kind='any' — the ``ANY_BY_DESIGN`` allow-list below governs which
    SIGNAL KIND a slot may take, not which metrics may declare a
    result-in-the-input's-unit."""
    ndf = schema['noise_density_at_freq']
    assert ndf['unit'] == '[y]'
    assert ndf['signal_inputs'] == [{'name': 'sig', 'kind': 'noise_density'}]


def test_integrated_noise_unit_is_relative_to_the_inputs_own_unit(schema):
    """TASK-288: the result is ``sqrt(integral of y^2 df)`` over ``sig``'s
    density (V/sqrt(Hz) for a voltage-noise probe, A/sqrt(Hz) for a
    current-noise probe), so it comes out in the input's own y unit times
    sqrt(Hz) — V for the former, A for the latter. The old hard-coded
    ``'V'`` mislabelled a current-noise probe's result as volts, the same
    family of defect TASK-179 fixed for ``error_norm`` and TASK-170 avoided
    for ``noise_density_at_freq``."""
    inb = schema['integrated_noise']
    assert inb['unit'] == '[y]*sqrt([x])'
    assert inb['signal_inputs'] == [{'name': 'sig', 'kind': 'noise_density'}]


class TestReflectionContract:
    def test_signal_inputs_match_annotations(self, schema):
        # RESOLVED annotations, not the raw ``p.annotation`` text: the metrics
        # live in silk.metrics, whose modules carry ``from __future__ import
        # annotations``, so the raw annotation is a STRING — 'Transient',
        # 'FrequencyResponse', … since TASK-196. This test asks the same
        # question the exporter asks instead of assuming a spelling.
        #
        # The v1 form of this test asserted membership with
        # ``issubclass(hints[n], Signal)``. ``Annotated[...]`` is NOT a class,
        # so that mechanism could not survive the change (design §4.2 names
        # this cost). The v2 form reads the SAME two facts out of the
        # annotation that the exporter does — the underlying type and the
        # SignalKind carried in ``__metadata__`` — and compares the whole
        # {name, kind} entry, so the kind is pinned to the annotation too and
        # not merely to itself.
        #
        # Resolve each decorated function from the complete metric registry.
        from silk.signals.schema import _REGISTRY
        for name, spec in schema.items():
            if name == '_meta':
                continue
            fn = _REGISTRY[name]
            hints = typing.get_type_hints(fn, include_extras=True)
            expected = []
            for n in inspect.signature(fn).parameters:
                hint = hints.get(n)
                metadata = getattr(hint, '__metadata__', ())
                if not metadata:
                    continue
                base = typing.get_args(hint)[0]
                if not (inspect.isclass(base) and issubclass(base, Signal)):
                    continue
                kinds = [m for m in metadata if isinstance(m, SignalKind)]
                assert len(kinds) == 1, (name, n)
                expected.append({'name': n, 'kind': kinds[0].name})
            assert spec['signal_inputs'] == expected, name

    def test_no_signal_input_exports_without_a_kind(self, schema):
        """AC#1 — the whole point of v2's shape. A slot with no kind is a
        slot a consumer has to guess at."""
        seen = 0
        for name, spec in schema.items():
            if name == '_meta':
                continue
            for entry in spec['signal_inputs']:
                assert set(entry) == {'name', 'kind'}, name
                assert entry['kind'] in SIGNAL_KINDS, (name, entry)
                seen += 1
        # The catalog measured at T-D: 71 signal inputs across 61 metrics;
        # 72 across 62 since TASK-200 re-decorated p1db; 73 across 63 since
        # TASK-170 added noise_density_at_freq; 72 across 62 since TASK-287
        # removed sh_settling_error (one signal input, ``sig``); 71 across 61
        # since TASK-210 removed phase_noise_at_offset (one signal input,
        # ``L_sig``). Pinned so a
        # metric added without a kind cannot hide behind a per-entry loop
        # that simply never runs.
        assert seen == 92  # TASK-414: noise_figure_at_freq adds one, band_noise_figure two; TASK-417: period_jitter_pp, cycle_to_cycle_jitter_pp add one each; TASK-412: removals take four; TASK-379: variance adds one; TASK-397: oip3_dbm adds one, iip3_dbm two; TASK-386: relative_error_norm adds two; TASK-399: three wrappers add six; TASK-233: eight wrappers; TASK-323.3: final_value; TASK-369: evm_percent adds two; TASK-363: comparator_prop_delay's two removed; TASK-366: average_frequency adds one; TASK-226.20: crossing_jitter_rms adds one

    def test_a_bare_signal_input_is_refused_by_name(self):
        """AC#4 — and the new failure mode design §4.2 warns about.

        v1 matched the annotation TEXT, so an unresolvable or unexpected
        annotation degraded QUIETLY into a scalar param. v2 resolves hints,
        and the mirror failure — a metric exporting with fewer signal inputs
        than it has, in the limit zero — is refused at decoration time
        instead."""
        from silk.signals.schema import metric, _REGISTRY

        with pytest.raises(UnsupportedSignalKindError) as exc:
            @metric(domain='tran', unit='V')
            def _kindless_probe(sig: Signal, thr: float = 0.0) -> float:
                return 0.0

        msg = str(exc.value)
        assert '_kindless_probe' in msg          # names the metric
        assert "'sig'" in msg                    # names the parameter
        assert 'Transient' in msg                # names what to write instead
        assert '_kindless_probe' not in _REGISTRY

    def test_the_refusal_is_a_type_error(self):
        assert issubclass(UnsupportedSignalKindError, TypeError)

    def test_decorator_does_not_wrap(self):
        """@metric returns the function unchanged — no behavior/signature change."""
        assert M.dc_gain_dB.__name__ == 'dc_gain_dB'
        assert not hasattr(M.dc_gain_dB, '__wrapped__')
        assert 'sig' in inspect.signature(M.dc_gain_dB).parameters

    def test_signal_subclass_detected(self):
        """A metric annotated with a Signal *subclass* is a signal input, not a
        scalar param (review finding on _is_signal_param)."""
        from silk.signals.schema import metric
        from silk.signals.spice import NgspiceSignal

        @metric(domain='tran', unit='V')
        def _probe(sig: typing.Annotated[NgspiceSignal, TRANSIENT],
                   thr: float = 0.0) -> float:
            return 0.0

        try:
            spec = metric_schema(_probe)
            assert spec['signal_inputs'] == [{'name': 'sig',
                                              'kind': 'transient'}]
            assert [p['name'] for p in spec['parameters']] == ['thr']
        finally:
            # Do not leak a fixture metric into the process-wide registry:
            # every later ``metric_count == 72`` assertion would read 73.
            from silk.signals.schema import _REGISTRY
            _REGISTRY.pop('_probe', None)


class TestSerializationAndCLI:
    def test_strict_json_serializable(self, schema):
        json.dumps(schema)
        text = export_metric_schema_json()
        assert json.loads(text)['dc_gain_dB']['domain'] == 'ac'

    def test_cli_emits_valid_json(self):
        out = subprocess.check_output(
            [sys.executable, '-m', 'silk.signals.schema'], text=True)
        assert json.loads(out)['_meta']['metric_count'] == 80


def test_single_metric_schema_matches_bulk(schema):
    assert metric_schema(M.phase_margin) == schema['phase_margin']


def test_get_metric_resolves_and_raises():
    """get_metric(name) returns the registered function, raises KeyError on miss."""
    fn = get_metric('dc_gain_dB')
    assert callable(fn)
    assert fn is M.dc_gain_dB

    with pytest.raises(KeyError, match='unknown metric'):
        get_metric('not_a_metric')


class TestDuplicateNameGuard:
    """TASK-172: ``_REGISTRY[fn.__name__] = fn`` used to overwrite silently.

    Two metrics sharing a name meant whichever module imported last won,
    the other simply disappeared from the exported schema, and
    ``_meta.metric_count`` dropped by one with nothing to show for it —
    the precise failure mode of moving metrics between packages.
    """

    @staticmethod
    def _unregister(*names):
        from silk.signals.schema import _REGISTRY
        for n in names:
            _REGISTRY.pop(n, None)

    def test_second_registration_under_an_existing_name_raises(self):
        from silk.signals.schema import metric, DuplicateMetricError, _REGISTRY

        @metric(domain='ac', unit='dB')
        def _dup_probe(sig: Transient) -> float:
            return 1.0

        first = _REGISTRY['_dup_probe']
        try:
            with pytest.raises(DuplicateMetricError) as exc:
                @metric(domain='tran', unit='V')
                def _dup_probe(sig: Transient) -> float:   # noqa: F811 — the point
                    return 2.0

            msg = str(exc.value)
            assert '_dup_probe' in msg                       # names the metric
            assert msg.count(__name__.split('.')[-1]) >= 2   # names both sides
            # the winner is unchanged: the second registration did NOT land
            assert _REGISTRY['_dup_probe'] is first
            assert _REGISTRY['_dup_probe']._silk_metric['unit'] == 'dB'
        finally:
            self._unregister('_dup_probe')

    def test_duplicate_metric_error_is_a_value_error(self):
        from silk.signals.schema import DuplicateMetricError
        assert issubclass(DuplicateMetricError, ValueError)

    def test_re_registering_the_same_function_object_is_allowed(self):
        """``python -m silk.signals.schema`` loads this module twice under two
        names; re-decorating the identical object is not a collision."""
        from silk.signals.schema import metric, _REGISTRY

        @metric(domain='ac', unit='dB')
        def _idempotent_probe(sig: Transient) -> float:
            return 1.0

        try:
            same = metric(domain='ac', unit='dB')(_idempotent_probe)
            assert same is _REGISTRY['_idempotent_probe']
        finally:
            self._unregister('_idempotent_probe')

    def test_the_real_catalog_registers_exactly_once(self):
        """Importing every metric-owning module again is a no-op, not a raise."""
        import importlib
        for name in ('silk.metrics', 'silk.metrics.ac', 'silk.metrics.noise',
                     'silk.signals'):
            importlib.import_module(name)
        assert export_metric_schema()['_meta']['metric_count'] == 80


class TestParameterTypes:
    """TASK-193 — ``type`` and ``optional``, reflected from the annotation.

    Before this, 75 of the 103 exported params published ``"default": null``
    and nothing else, so a consumer could not tell a required float from an
    optional one — callers previously needed
    ``typing.get_type_hints`` on silk's live function objects to recover what
    the schema would not say (design §3.3). These assert the schema says it.
    """

    def test_every_parameter_carries_type_and_optional(self, schema):
        """AC#1 — no parameter may export without both."""
        missing = [f"{name}.{p['name']}"
                   for name, spec in schema.items() if name != '_meta'
                   for p in spec['parameters']
                   if 'type' not in p or 'optional' not in p]
        assert missing == []
        total = sum(len(spec['parameters'])
                    for name, spec in schema.items() if name != '_meta')
        assert total == 280  # TASK-226.22: + 2 scalar eye thresholds; TASK-438: + 4 hysteresis n_rising/n_falling/start/end; TASK-414: + 10 NF params (spot freq/rs/temp_sim_k/temp_ref_k/interpolate, band f1/f2/rs/temp_sim_k/temp_ref_k); TASK-417: jitter pair 2 -> 6 params each + two 6-param _pp forms; TASK-418: + 7 long_term_jitter units/reference/phase/threshold/edge/start/end; TASK-413: + 3 interpolate (gain/phase/noise_density_at_freq); TASK-379: + variance's 3; TASK-380: + 2 refine; TASK-378: + 10 interpolate; TASK-396: + 30 spectral options; TASK-397: + 12 IP3; TASK-386: + relative_error_norm kind/floor/start/end; TASK-399: + 9 static-error wrapper params; TASK-385: + error_norm start/end; TASK-233: 106 + 14 wrapper parameters; TASK-364: + crossing start/end; TASK-369: + 4 evm_percent; TASK-365: + 16 rise/fall/slew edge params; TASK-363: propagation_delay 3 -> 8, comparator_prop_delay's 2 removed; TASK-366: + 7 period/frequency params; TASK-226.20: + 5 crossing_jitter_rms; TASK-367: + 11 duty/pulse/on/off params

    def test_type_vocabulary_is_closed(self, schema):
        """The point of normalising: a consumer switches on a known string.

        Catches an annotation that slipped through as something a JS consumer
        cannot read, the same way ``test_unit_vocabulary_is_closed`` catches a
        typo'd unit.
        """
        used = {p['type'] for name, spec in schema.items() if name != '_meta'
                for p in spec['parameters']}
        assert used <= {'float', 'int', 'str', 'bool', 'complex_points'}, sorted(used)

    def test_optional_and_required_are_different_questions(self, schema):
        """``required`` = no default; ``optional`` = ``None`` is accepted.

        ``spectral_power.window`` is the case that separates them: it is
        ``Optional[str] = 'hann'`` — optional, not required, and its default is
        not ``None`` (``None`` means "no window", spectral.py ``_make_window``).
        """
        window = next(p for p in schema['spectral_power']['parameters']
                      if p['name'] == 'window')
        assert (window['type'], window['optional'],
                window['required'], window['default']) == ('str', True, False, 'hann')

        target = next(p for p in schema['settling_time']['parameters']
                      if p['name'] == 'target')
        # TASK-323.3: target defaults to the trace's own final value.
        assert (target['type'], target['optional'],
                target['required']) == ('float', True, False)

    def test_cmrr_freq_is_an_optional_float(self, schema):
        """AC#3 — it was annotated ``float`` while defaulting to ``None``
        (design §3.7), which would have published the self-contradictory
        ``type: float, optional: false, default: null``."""
        freq = next(p for p in schema['cmrr_dB']['parameters']
                    if p['name'] == 'freq')
        assert freq['type'] == 'float'
        assert freq['optional'] is True
        assert freq['required'] is False
        assert freq['default'] is None

    def test_no_param_defaults_to_None_without_being_optional(self, schema):
        """The §3.7 defect class, now checkable from the schema alone: a param
        whose default is ``None`` must accept ``None``, i.e. be ``Optional``."""
        bad = [f"{name}.{p['name']}"
               for name, spec in schema.items() if name != '_meta'
               for p in spec['parameters']
               if not p['required'] and p['default'] is None and not p['optional']]
        assert bad == []

    def test_reflected_types_match_the_live_annotations(self, schema):
        """Reflection, not authoring: recompute every type from
        ``get_type_hints`` here and require the schema to agree.

        A ``Literal[...]`` (TASK-194) contributes the type of its values, so
        the recomputation unwraps it the same way the exporter does — see
        :class:`TestParamChoices` for the choices half.
        """
        from silk.signals.schema import _REGISTRY

        for name, spec in schema.items():
            if name == '_meta':
                continue
            hints = typing.get_type_hints(_REGISTRY[name], include_extras=True)
            for p in spec['parameters']:
                ann = hints[p['name']]
                args = typing.get_args(ann)
                optional = type(None) in args
                base = next(a for a in args if a is not type(None)) if optional else ann
                if typing.get_origin(base) is typing.Literal:
                    base = type(typing.get_args(base)[0])
                expected = ('complex_points' if base == S.ComplexPoints else base.__name__)
                assert (p['type'], p['optional']) == (expected, optional), \
                    f"{name}.{p['name']}"

    def test_types_are_json_safe(self, schema):
        """``optional`` must be a real bool, not numpy's or a string."""
        for name, spec in schema.items():
            if name == '_meta':
                continue
            for p in spec['parameters']:
                assert type(p['optional']) is bool, f"{name}.{p['name']}"
                assert isinstance(p['type'], str)
        json.dumps(schema)


class TestParamTypeRefusal:
    """AC#2 — an annotation the schema cannot normalise is refused at
    DECORATION time, naming the metric, the param and the annotation.

    Same principle as the return-annotation guard: the decorator will not
    guess, and the check lives at the definition rather than in a skip list.
    """

    @staticmethod
    def _refuse(fn):
        from silk.signals.schema import UnsupportedParamTypeError, metric
        with pytest.raises(UnsupportedParamTypeError) as exc:
            metric(domain='tran', unit='V')(fn)
        return str(exc.value)

    def test_an_unknown_annotation_is_refused(self):
        def _bytes_probe(sig: Transient, mask: bytes = b'') -> float:
            return 0.0

        msg = self._refuse(_bytes_probe)
        assert '_bytes_probe' in msg          # names the metric
        assert "'mask'" in msg                # names the parameter
        assert 'bytes' in msg                 # names the annotation

    def test_an_unannotated_param_is_refused(self):
        def _bare_probe(sig: Transient, thr=0.0) -> float:
            return 0.0

        msg = self._refuse(_bare_probe)
        assert '_bare_probe' in msg and "'thr'" in msg
        assert 'no annotation' in msg

    @pytest.mark.parametrize('annotation,optional', [(bool, False),
                                                      (typing.Optional[bool], True)])
    def test_bool_is_distinct_from_int(self, annotation, optional):
        def _bool_probe(sig: Transient, invert=False) -> float:
            return 0.0

        _bool_probe.__annotations__['invert'] = annotation
        try:
            S.metric(domain='tran', unit='')(_bool_probe)
            param = S.metric_schema(_bool_probe)['parameters'][0]
            assert (param['type'], param['optional']) == ('bool', optional)
        finally:
            S._REGISTRY.pop('_bool_probe', None)

    def test_a_union_of_two_real_types_is_refused(self):
        """``Union[int, float]`` has no single type name to publish; only
        ``Optional[X]`` — a union with ``None`` — normalises."""
        def _union_probe(sig: Transient, n: typing.Union[int, float] = 0) -> float:
            return 0.0

        msg = self._refuse(_union_probe)
        assert "'n'" in msg and 'Union' in msg

    def test_optional_of_an_unknown_type_is_refused(self):
        def _optional_bytes_probe(sig: Transient, mask: typing.Optional[bytes] = None) -> float:
            return 0.0

        msg = self._refuse(_optional_bytes_probe)
        assert "'mask'" in msg and 'bytes' in msg

    def test_a_refused_metric_is_not_registered_and_gains_no_metadata(self):
        from silk.signals.schema import _REGISTRY

        def _rejected_probe(sig: Transient, mask: bytes = b'') -> float:
            return 0.0

        self._refuse(_rejected_probe)
        assert '_rejected_probe' not in _REGISTRY
        assert not hasattr(_rejected_probe, '_silk_metric')

    def test_the_error_is_a_type_error(self):
        """Same family as NonScalarMetricError: decoration-time API misuse."""
        from silk.signals.schema import UnsupportedParamTypeError
        assert issubclass(UnsupportedParamTypeError, TypeError)

    def test_the_scalar_return_guard_still_runs_first(self):
        """A function that fails both checks reports the return, not the param:
        an unexportable return type is the more fundamental refusal."""
        from silk.signals.schema import NonScalarMetricError, metric

        def _both_wrong_probe(sig: Transient, mask=b''):
            return 0.0

        with pytest.raises(NonScalarMetricError):
            metric(domain='tran', unit='V')(_both_wrong_probe)

    def test_every_real_metric_normalises(self):
        """The refusal is not theoretical cover for a catalog that dodges it:
        re-reflect all 61 live metrics and require every one to pass."""
        from silk.signals.schema import _REGISTRY, _reflect_param_types

        total = 0
        for fn in _REGISTRY.values():
            if fn._silk_metric.get('outputs'):
                continue
            total += len(_reflect_param_types(fn))
        assert total == 280  # TASK-226.22: + 2 scalar eye thresholds; TASK-438: + 4 hysteresis n_rising/n_falling/start/end; TASK-414: + 10 NF params (spot freq/rs/temp_sim_k/temp_ref_k/interpolate, band f1/f2/rs/temp_sim_k/temp_ref_k); TASK-417: jitter pair 2 -> 6 params each + two 6-param _pp forms; TASK-418: + 7 long_term_jitter units/reference/phase/threshold/edge/start/end; TASK-413: + 3 interpolate (gain/phase/noise_density_at_freq); TASK-379: + variance's 3; TASK-380: + 2 refine; TASK-378: + 10 interpolate; TASK-396: + 30 spectral options; TASK-397: + 12 IP3; TASK-386: + relative_error_norm kind/floor/start/end; TASK-399: + 9 static-error wrapper params; TASK-385: + error_norm start/end; TASK-365: + 16 rise/fall/slew edge params; TASK-363: propagation_delay 3 -> 8, comparator_prop_delay's 2 removed; TASK-226.20: + 5 crossing_jitter_rms; TASK-367: + 11 duty/pulse/on/off params


# ---------------------------------------------------------------------------
# TASK-194 — parameters[].choices
# ---------------------------------------------------------------------------

_T = 1e-9                      # clock period for the conformance fixtures


def _clock(delay: float = 0.0) -> Signal:
    t = np.linspace(0, 64 * _T, 64 * 50 + 1)
    y = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * (t - delay) / _T + 1e-12))
    return Signal('clk', 'voltage', 'time', 'V', t, y)


def _ramp() -> Signal:
    t = np.linspace(0, 64 * _T, 64 * 50 + 1)
    return Signal('ramp', 'voltage', 'time', 'V', t, np.linspace(0.0, 1.0, len(t)))


def _sine() -> Signal:
    dt = _T / 8
    t = np.arange(256) * dt
    return Signal('sine', 'voltage', 'time', 'V', t, np.sin(2 * np.pi * t / (8 * dt)))


def _freq_ac() -> Signal:
    f = np.logspace(1, 7, 121)
    return Signal('h', 'voltage', 'frequency', 'V', f, 100.0 / (1 + 1j * f / 1e4))


#: ``(metric, param) -> callable(value)`` — one real call per string param,
#: with every OTHER argument valid, so the only thing under test is the value
#: of the choice param. Driven by the schema, never by a hand-written copy of
#: the permitted set: the test reads ``choices`` out of the export and feeds
#: each one back through the live function.
_CHOICE_INVOKERS = {
    ('propagation_delay', 'in_edge'):
        lambda v: M.propagation_delay(_clock(), _clock(0.1 * _T), in_edge=v),
    ('propagation_delay', 'out_edge'):
        lambda v: M.propagation_delay(_clock(), _clock(0.1 * _T), out_edge=v),
    ('period_jitter', 'edge'):
        lambda v: M.period_jitter(_clock(), edge=v),
    ('cycle_to_cycle_jitter', 'edge'):
        lambda v: M.cycle_to_cycle_jitter(_clock(), edge=v),
    ('period_jitter_pp', 'edge'):
        lambda v: M.period_jitter_pp(_clock(), edge=v),
    ('cycle_to_cycle_jitter_pp', 'edge'):
        lambda v: M.cycle_to_cycle_jitter_pp(_clock(), edge=v),
    ('period_jitter', 'reference'):
        lambda v: M.period_jitter(_clock(), reference=v,
                                  cycles=3 if v == 'moving_average' else None),
    ('period_jitter_pp', 'reference'):
        lambda v: M.period_jitter_pp(_clock(), reference=v,
                                     cycles=3 if v == 'moving_average' else None),
    ('cycle_to_cycle_jitter', 'reference'):
        lambda v: M.cycle_to_cycle_jitter(
            _clock(), reference=v, cycles=3 if v == 'moving_average' else None),
    ('cycle_to_cycle_jitter_pp', 'reference'):
        lambda v: M.cycle_to_cycle_jitter_pp(
            _clock(), reference=v, cycles=3 if v == 'moving_average' else None),
    ('crossing_jitter_rms', 'edge'):
        lambda v: M.crossing_jitter_rms(_clock(), period=_T, edge=v),
    ('crossing', 'edge'):
        lambda v: M.crossing(_ramp(), 0.5, edge=v),
    ('average_period', 'edge'):
        lambda v: M.average_period(_clock(), edge=v),
    ('average_frequency', 'edge'):
        lambda v: M.average_frequency(_clock(), edge=v),
    ('average_duty', 'edge'):
        lambda v: M.average_duty(_clock(), edge=v),
    ('pulse_width', 'polarity'):
        lambda v: M.pulse_width(_clock(), polarity=v),
    ('error_norm', 'kind'):
        lambda v: M.error_norm(_ramp(), _ramp(), kind=v),
    ('relative_error_norm', 'kind'):
        lambda v: M.relative_error_norm(_ramp(), _ramp(), kind=v, floor=0.0),
    ('spectral_power', 'window'):
        lambda v: M.spectral_power(_sine(), 0.0, 4e9, tclk=_T / 8, window=v),
    ('enob', 'window'):
        lambda v: M.enob(_sine(), tclk=_T / 8, window=v),
    ('noise_floor_dbfs', 'window'):
        lambda v: M.noise_floor_dbfs(_sine(), tclk=_T / 8, window=v),
    ('sfdr_db', 'window'):
        lambda v: M.sfdr_db(_sine(), tclk=_T / 8, window=v),
    ('sinad', 'window'):
        lambda v: M.sinad(_sine(), tclk=_T / 8, window=v),
    ('sndr', 'window'):
        lambda v: M.sndr(_sine(), tclk=_T / 8, window=v),
    ('snr', 'window'):
        lambda v: M.snr(_sine(), tclk=_T / 8, window=v),
    ('thd', 'window'):
        lambda v: M.thd(_sine(), tclk=_T / 8, window=v),
    ('thd_percent', 'window'):
        lambda v: M.thd_percent(_sine(), tclk=_T / 8, window=v),
    ('inl_max_lsb', 'method'):
        lambda v: M.inl_max_lsb(Signal('codes', 'voltage', 'code', 'V',
                                     np.arange(4), np.arange(4)), method=v),
    ('bandwidth_3dB', 'reference'):
        lambda v: M.bandwidth_3dB(_freq_ac(), reference=v),
    ('evm_percent', 'modulation'):
        lambda v: M.evm_percent(_sine(), _sine(), symbol_period=_T,
                                modulation=v),
    ('slew_rate', 'edge'):
        lambda v: M.slew_rate(_clock(), edge=v),
    ('long_term_jitter', 'units'):  # TASK-418
        lambda v: M.long_term_jitter(_clock(), units=v),
    ('long_term_jitter', 'reference'):  # TASK-418
        lambda v: M.long_term_jitter(_clock(), reference=v),
    ('long_term_jitter', 'edge'):  # TASK-418
        lambda v: M.long_term_jitter(_clock(), edge=v),
}

#: Values no metric accepts — a near-miss typo, a case variant, a plausible
#: numpy-ism, an empty string and a plain nonsense word. Filtered against each
#: param's own ``choices`` at use, so a value that IS legal somewhere is never
#: asserted to raise there.
_BOGUS = ('', 'rsing', 'Rising', 'hanning', 'nope')


def _string_params(schema):
    """``[(metric, param_dict)]`` for every ``type: 'str'`` param exported."""
    return [(name, p)
            for name, spec in schema.items() if name != '_meta'
            for p in spec['parameters'] if p['type'] == 'str']


class TestParamChoices:
    """TASK-194 (design §5.2, owner decision D2a) — the permitted values of a
    string param are REFLECTED from its ``Literal[...]`` annotation.

    Before this, a consumer reading the schema saw ``type: "str"`` and had no
    way to know that ``edge`` takes exactly three words — it could only offer
    a free-text box and discover ``edge='rsing'`` after simulating. The set
    was in the Python all along, as a tuple inside the function body, so it is
    reflected rather than authored: there is no ``param_choices=`` on the
    decorator, which would be a second copy of a fact the body already states.
    """

    def test_the_nine_string_params_carry_their_choices(self, schema):
        """AC#1 — the census, spelled out, so a change to any of the ten is
        visible in the diff of this test rather than only in behaviour."""
        got = {(name, p['name']): tuple(p['choices'])
               for name, p in _string_params(schema)}
        assert got == {
            ('propagation_delay', 'in_edge'): ('rising', 'falling', 'any'),
            ('propagation_delay', 'out_edge'): ('rising', 'falling', 'any'),
            ('period_jitter', 'edge'): ('rising', 'falling', 'any'),
            ('cycle_to_cycle_jitter', 'edge'): ('rising', 'falling', 'any'),
            ('period_jitter_pp', 'edge'): ('rising', 'falling', 'any'),
            ('cycle_to_cycle_jitter_pp', 'edge'): ('rising', 'falling', 'any'),
            ('period_jitter', 'reference'): ('mean', 'moving_average'),
            ('period_jitter_pp', 'reference'): ('mean', 'moving_average'),
            ('cycle_to_cycle_jitter', 'reference'): ('mean', 'moving_average'),
            ('cycle_to_cycle_jitter_pp', 'reference'):
                ('mean', 'moving_average'),
            ('crossing_jitter_rms', 'edge'): ('rising', 'falling', 'any'),
            ('crossing', 'edge'): ('rising', 'falling', 'any'),
            ('slew_rate', 'edge'): ('rising', 'falling', 'any'),
            ('average_period', 'edge'): ('rising', 'falling', 'any'),
            ('average_frequency', 'edge'): ('rising', 'falling', 'any'),
            ('average_duty', 'edge'): ('rising', 'falling', 'any'),
            ('pulse_width', 'polarity'): ('positive', 'negative'),
            ('error_norm', 'kind'): ('max', 'rms', 'mean_abs'),
            ('relative_error_norm', 'kind'): ('max', 'rms', 'mean_abs'),
            ('spectral_power', 'window'):
                ('hann', 'hamming', 'blackman', 'rectangular'),
            ('enob', 'window'):  # TASK-396
                ('hann', 'hamming', 'blackman', 'rectangular'),
            ('noise_floor_dbfs', 'window'):  # TASK-396
                ('hann', 'hamming', 'blackman', 'rectangular'),
            ('sfdr_db', 'window'):  # TASK-396
                ('hann', 'hamming', 'blackman', 'rectangular'),
            ('sinad', 'window'):  # TASK-396
                ('hann', 'hamming', 'blackman', 'rectangular'),
            ('sndr', 'window'):  # TASK-396
                ('hann', 'hamming', 'blackman', 'rectangular'),
            ('snr', 'window'):  # TASK-396
                ('hann', 'hamming', 'blackman', 'rectangular'),
            ('thd', 'window'):  # TASK-396
                ('hann', 'hamming', 'blackman', 'rectangular'),
            ('thd_percent', 'window'):  # TASK-396
                ('hann', 'hamming', 'blackman', 'rectangular'),
            ('bandwidth_3dB', 'reference'): ('dc', 'peak'),
            ('inl_max_lsb', 'method'): ('endpoint', 'best_fit'),
            ('evm_percent', 'modulation'):
                ('bpsk', 'qpsk', '8psk', '16qam', '64qam', '256qam',
                 'pam4', 'pam8'),
            ('long_term_jitter', 'units'): ('s', 'ui', 'rad', 'deg'),  # TASK-418
            ('long_term_jitter', 'reference'): ('fit', 'endpoints'),  # TASK-418
            ('long_term_jitter', 'edge'): ('rising', 'falling', 'any'),  # TASK-418
        }

    def test_no_string_param_exports_without_choices(self, schema):
        """AC#3 — TASK-185's ask, as a standing rule rather than a one-off
        check of today's seven: a ``str`` param with no permitted set published
        is one a consumer can only offer as free text."""
        bare = [f"{name}.{p['name']}"
                for name, p in _string_params(schema) if not p['choices']]
        assert bare == []

    def test_every_parameter_carries_the_key(self, schema):
        """``choices`` is present on every param, ``None`` where there is no
        permitted set — a consumer tests the value, never ``'choices' in p``."""
        missing = [f"{name}.{p['name']}"
                   for name, spec in schema.items() if name != '_meta'
                   for p in spec['parameters'] if 'choices' not in p]
        assert missing == []

    def test_choices_is_null_for_every_non_string_param(self, schema):
        """The float/int params are open ranges, not menus; publishing an
        empty list there would read as "nothing is permitted"."""
        wrong = [f"{name}.{p['name']}"
                 for name, spec in schema.items() if name != '_meta'
                 for p in spec['parameters']
                 if p['type'] != 'str' and p['choices'] is not None]
        assert wrong == []

    def test_all_four_edge_params_declare_the_same_vocabulary(self, schema):
        """``edge`` is annotated at four call sites. They are four copies of
        one vocabulary, and a value added to one but not the others would be
        an inconsistency no single-metric test could see."""
        vocab = {tuple(p['choices'])
                 for name, p in _string_params(schema)
                 if p['name'] in ('edge', 'in_edge')}
        assert len(vocab) == 1, vocab

    def test_choices_are_reflected_not_authored(self, schema):
        """Recompute every permitted set from ``get_type_hints`` and require
        the schema to agree — the same guarantee ``type`` gets. If someone
        adds a ``param_choices=`` to the decorator this still passes only
        while it agrees with the annotation, and the annotation is the thing
        the function body is tested against."""
        from silk.signals.schema import _REGISTRY

        for name, spec in schema.items():
            if name == '_meta':
                continue
            hints = typing.get_type_hints(_REGISTRY[name], include_extras=True)
            for p in spec['parameters']:
                ann = hints[p['name']]
                args = typing.get_args(ann)
                if type(None) in args:
                    ann = next(a for a in args if a is not type(None))
                expected = (list(typing.get_args(ann))
                            if typing.get_origin(ann) is typing.Literal
                            else None)
                assert p['choices'] == expected, f"{name}.{p['name']}"

    def test_the_decorator_has_no_param_choices_argument(self):
        """The design's whole reason for preferring ``Literal`` over an
        authored ``param_choices={...}``: the set must live in exactly one
        place. An authored back door would let the two drift."""
        from silk.signals.schema import metric
        assert 'param_choices' not in inspect.signature(metric).parameters

    def test_choices_are_json_safe(self, schema):
        """Plain ``str`` values in a plain ``list`` — a tuple would serialise
        as a JSON array too, but would not survive a round-trip unchanged."""
        for name, p in _string_params(schema):
            assert type(p['choices']) is list, f"{name}.{p['name']}"
            for v in p['choices']:
                assert type(v) is str, f"{name}.{p['name']}"
        json.dumps(schema)

    def test_every_default_is_one_of_the_choices(self, schema):
        """A default outside the permitted set would make the metric raise on
        a call that supplied nothing — the schema can catch that here."""
        for name, p in _string_params(schema):
            if p['required']:
                continue           # evm_percent.modulation (TASK-369)
            if p['default'] is None:
                assert p['optional'], f"{name}.{p['name']}"
                continue
            assert p['default'] in p['choices'], f"{name}.{p['name']}"

    def test_the_addition_is_additive(self, schema):
        """``choices`` is new information alongside the old, not a reshape.

        ``parameters[]`` is still purely additive — every 1.0 key is present
        and unchanged, and T-B added one more. The ``format_version`` bump to
        '2.0' is NOT about this: exactly one thing reshaped in v2, and it is
        ``signal_inputs[]`` (design §8.1, TASK-196)."""
        assert schema['_meta']['format_version'] == '2.0'
        for name, spec in schema.items():
            if name == '_meta':
                continue
            for p in spec['parameters']:
                assert set(p) == {'name', 'unit', 'default', 'required',
                                  'type', 'optional', 'choices'}


class TestChoicesConformance:
    """AC#2 — the declared set and the enforcing guard agree, for all eight.

    An annotation does NOT enforce anything at runtime; the ``if x not in
    (...)`` inside each function body still does, and it is a separate copy of
    the same fact. This class is the only thing that binds them: it reads each
    permitted set out of the exported schema and pushes every value through
    the live metric. A value added to a ``Literal`` but not to the guard fails
    :meth:`test_every_declared_choice_is_accepted`; a value dropped from a
    ``Literal`` but still accepted by the guard is invisible to the schema and
    is why :meth:`test_a_value_outside_the_set_raises` exists.
    """

    def test_the_invoker_table_covers_every_string_param(self, schema):
        """A new string param with no invoker would otherwise be silently
        unconformed — the schema would publish a set nothing tested."""
        assert (set(_CHOICE_INVOKERS)
                == {(name, p['name']) for name, p in _string_params(schema)})

    @pytest.mark.parametrize('key', sorted(_CHOICE_INVOKERS))
    def test_every_declared_choice_is_accepted(self, schema, key):
        """Every value the schema offers must actually work. A consumer that
        believes the schema and sends ``edge='any'`` must not get a
        ``ValueError`` back."""
        metric_name, param_name = key
        p = next(p for p in schema[metric_name]['parameters']
                 if p['name'] == param_name)
        for value in p['choices']:
            _CHOICE_INVOKERS[key](value)          # must not raise

    @pytest.mark.parametrize('key', sorted(_CHOICE_INVOKERS))
    def test_a_value_outside_the_set_raises(self, schema, key):
        """The refusal half. ``propagation_delay(in_edge='rsing')`` used to
        return a finite number indistinguishable from a measurement — that is
        TASK-192, fixed before this landed, and this keeps it fixed for all
        seven rather than for the one that was noticed."""
        metric_name, param_name = key
        p = next(p for p in schema[metric_name]['parameters']
                 if p['name'] == param_name)
        for value in _BOGUS:
            if value in p['choices']:
                continue
            with pytest.raises(ValueError):
                _CHOICE_INVOKERS[key](value)

    @pytest.mark.parametrize('key', sorted(_CHOICE_INVOKERS))
    def test_None_is_accepted_only_where_optional_says_so(self, schema, key):
        """``optional`` is the other half of the contract and is checked the
        same way: ``spectral_power.window=None`` means "no window" and must
        work; ``edge=None`` must not."""
        metric_name, param_name = key
        p = next(p for p in schema[metric_name]['parameters']
                 if p['name'] == param_name)
        if p['optional']:
            _CHOICE_INVOKERS[key](None)           # must not raise
        else:
            with pytest.raises(ValueError):
                _CHOICE_INVOKERS[key](None)

    def test_window_none_is_refused_like_any_other_bogus_value(self, schema):
        """The legacy string ``'none'`` alias of ``'rectangular'`` is GONE
        (owner ruling 2026-09-13): the declared ``choices`` and the body guard
        agree exactly, so nothing outside ``choices`` is tolerated on input.
        ``None`` (the object) remains accepted because ``optional`` says so.
        """
        window = next(p for p in schema['spectral_power']['parameters']
                      if p['name'] == 'window')
        assert 'none' not in window['choices']
        with pytest.raises(ValueError):
            _CHOICE_INVOKERS[('spectral_power', 'window')]('none')

class TestChoicesRefusal:
    """A ``Literal`` the schema cannot publish honestly is refused at
    DECORATION time, like any other un-normalisable annotation."""

    @staticmethod
    def _refuse(fn):
        from silk.signals.schema import UnsupportedParamTypeError, metric
        with pytest.raises(UnsupportedParamTypeError) as exc:
            metric(domain='tran', unit='V')(fn)
        return str(exc.value)

    def test_a_mixed_type_literal_is_refused(self):
        """``Literal['a', 1]`` has no single ``type`` to publish, and guessing
        one would put a value in ``choices`` that does not match ``type``."""
        def _mixed_probe(sig: Transient,
                         mode: typing.Literal['a', 1] = 'a') -> float:
            return 0.0

        msg = self._refuse(_mixed_probe)
        assert "'mode'" in msg and 'Literal' in msg
        assert '_mixed_probe' in msg               # names the metric
        assert "'mode'" in msg                     # names the parameter
        assert "Literal['a', 1]" in msg            # and the annotation

    def test_a_literal_of_an_unsupported_type_is_refused(self):
        """``Literal[b'x']`` normalises to ``bytes``, which is outside the
        vocabulary — the Literal path is not a back door into it."""
        def _bytes_literal_probe(sig: Transient,
                                 mask: typing.Literal[b'x'] = b'x') -> float:
            return 0.0

        msg = self._refuse(_bytes_literal_probe)
        assert "'mask'" in msg
        assert '_bytes_literal_probe' in msg
        assert "'mask'" in msg
        assert "Literal[b'x']" in msg

    def test_a_bool_literal_is_distinct_from_int(self):
        def _bool_literal_probe(sig: Transient,
                                flag: typing.Literal[True, False] = False) -> float:
            return 0.0

        try:
            S.metric(domain='tran', unit='')(_bool_literal_probe)
            param = S.metric_schema(_bool_literal_probe)['parameters'][0]
            assert param['type'] == 'bool'
            assert param['choices'] == [True, False]
        finally:
            S._REGISTRY.pop('_bool_literal_probe', None)

    def test_the_lookup_is_by_identity_not_by_type_name(self):
        """The ``Literal`` branch resolves the member type and hands it to the
        SAME identity-keyed ``_PARAM_TYPES`` lookup ``type`` already used — it
        does not compare type names. A class merely *called* ``str`` is not
        ``str`` and must be refused, on the plain path and through a Literal
        alike; a name comparison would admit both and publish ``type: "str"``
        for something no consumer could parse as one.
        """
        class _NotReallyStr:
            pass
        _NotReallyStr.__name__ = 'str'
        imposter = _NotReallyStr()

        def _plain_probe(sig: Transient, mode: _NotReallyStr = imposter) -> float:
            return 0.0

        def _literal_probe(sig: Transient,
                           mode: typing.Literal[imposter] = imposter) -> float:
            return 0.0

        assert "'mode'" in self._refuse(_plain_probe)
        assert "'mode'" in self._refuse(_literal_probe)

    def test_an_int_literal_normalises(self):
        """Nothing in the catalog uses one today, but the rule is about the
        VALUES' type, not about strings: an int Literal publishes
        ``type: 'int'`` with its choices rather than being refused."""
        from silk.signals.schema import _REGISTRY, metric, metric_schema

        def _int_literal_probe(sig: Transient,
                               n: typing.Literal[1, 2, 4] = 1) -> float:
            return 0.0

        try:
            metric(domain='tran', unit='V')(_int_literal_probe)
            p = metric_schema(_int_literal_probe)['parameters'][0]
            assert (p['type'], p['choices']) == ('int', [1, 2, 4])
        finally:
            _REGISTRY.pop('_int_literal_probe', None)


class TestSignalKinds:
    """v2's ``signal_inputs[].kind`` and ``_meta.signal_kinds`` (TASK-196).

    The schema half of TASK-182 and TASK-183: a consumer can tell which probe
    fits which slot, and can refuse a metric no probe will ever satisfy.
    """

    #: THE ALLOW-LIST (reviewer finding 1 on TASK-196). ``any`` has a
    #: ``produced_by`` naming every analysis, so a slot annotated ``any`` is
    #: refused by nobody — it is the escape hatch with no guard, and the lazy
    #: pick whenever a kind is unclear. It is CORRECT for exactly one family:
    #: the primitives ``silk.metrics.waveform`` documents as deliberately
    #: axis-agnostic, the ones whose declared unit is ``'[y]'`` or ``'[x]'``
    #: because it is the INPUT Signal's unit and no concrete one can be named.
    #: Adding an entry here has to be a deliberate edit, the same way
    #: ``UNEXPORTED_BY_DESIGN`` works in tests/test_metric_scalar_return_guard.
    ANY_BY_DESIGN = {
        ('average', 'sig'),
        ('rms', 'sig'),
        ('stddev', 'sig'),
        ('variance', 'sig'),
        ('peak', 'sig'),
        ('peak_to_peak', 'sig'),
        ('minimum', 'sig'),
        ('maximum', 'sig'),
        ('argmin_x', 'sig'),
        ('argmax_x', 'sig'),
        ('integral', 'sig'),
        ('value_at', 'sig'),
        ('final_value', 'sig'),
        ('crossing', 'sig'),
        # error_norm's MEASURED input. Its docstring is explicit that goal is
        # interpolated onto sig's grid and takes "the same axis kind as sig",
        # and the metric validates no axis at all, so 'transient' would be a
        # narrowing the code does not make. It costs nothing in offerability:
        # the metric is already unofferable through its OTHER slot, below.
        ('error_norm', 'sig'),
        ('relative_error_norm', 'sig'),
    }

    #: The slots NOTHING silk can simulate will ever fill — an empty
    #: ``produced_by``. This is what a consumer refuses on (design §2), so the
    #: set is pinned: a kind quietly gaining a producer, or a metric quietly
    #: becoming offerable, is a deliberate edit here.
    UNPRODUCIBLE_BY_DESIGN = {
        ('inl_max_lsb', 'sig'),
        ('dnl_max_lsb', 'sig'),
        ('missing_code_count', 'sig'),
        ('non_monotonic_steps', 'sig'),
        ('offset_error_lsb', 'sig'),
        ('gain_error_lsb', 'sig'),
        ('full_scale_error_lsb', 'sig'),
        # reviewer finding 2 on TASK-196: a goal is a waveform the USER
        # writes down. Offering error_norm wired to two probes would compute
        # the difference between two measurements and call it an error
        # against a target.
        ('error_norm', 'goal'),
        ('relative_error_norm', 'goal'),
        # TASK-200 (design §9, owner decision D5a): a power sweep is
        # assembled from a parameter sweep across N runs, not produced by any
        # single analysis. This entry is why p1db can be IN the catalog --
        # re-decorated, discoverable, callable -- and still never offered.
        ('p1db', 'sig'),
    }

    @staticmethod
    def _slots(schema):
        for name, spec in schema.items():
            if name == '_meta':
                continue
            for entry in spec['signal_inputs']:
                yield name, entry['name'], entry['kind']

    def test_the_any_allow_list_is_exact(self, schema):
        actual = {(m, slot) for m, slot, kind in self._slots(schema)
                  if kind == 'any'}
        assert actual == self.ANY_BY_DESIGN

    def test_every_any_input_declares_an_input_relative_unit(self, schema):
        """The allow-list is not arbitrary: every metric on it declares its
        result in the INPUT's units, which is what makes it axis-agnostic.

        ``'[y]'``, ``'[x]'`` or a product of them (``integral`` is
        ``'[y]*[x]'``) — never a concrete unit. A metric that can name a
        real unit knows what its input is, and has no business taking
        ``any``."""
        for name, _slot in self.ANY_BY_DESIGN:
            unit = schema[name]['unit']
            if unit == '':
                # A dimensionless result is equally axis-agnostic
                # (TASK-386 relative_error_norm).
                assert name == 'relative_error_norm', (name, unit)
                continue
            residue = unit.replace('[y]', '').replace('[x]', '')
            assert '[' in unit and not any(c.isalpha() for c in residue), \
                (name, unit)

    def test_the_unproducible_slots_are_exact(self, schema):
        actual = {(m, slot) for m, slot, kind in self._slots(schema)
                  if not SIGNAL_KINDS[kind].produced_by}
        assert actual == self.UNPRODUCIBLE_BY_DESIGN

    def test_the_refusal_rule_can_be_evaluated_from_the_export_alone(self, schema):
        """The one-line rule v2 exists to enable: offer a metric iff EVERY
        signal_inputs[].kind has a produced_by that intersects the analyses
        the consumer can author. Evaluated here off the exported dict only —
        no silk import — because that is all a consumer has."""
        kinds = schema['_meta']['signal_kinds']
        can_author = {'tran', 'ac', 'dc', 'noise'}

        def offerable(spec):
            return all(set(kinds[e['kind']]['produced_by']) & can_author
                       for e in spec['signal_inputs'])

        refused = {n for n, spec in schema.items()
                   if n != '_meta' and not offerable(spec)}
        assert refused == {m for m, _ in self.UNPRODUCIBLE_BY_DESIGN}

    def test_the_rule_refuses_p1db_by_kind(self, schema):
        """TASK-200 AC#2, spelled out rather than left to the set above.

        ``p1db`` is IN the catalog (TASK-200 reversed TASK-184's
        undecorating) and must never be offered: its only slot is a
        ``power_sweep``, a curve assembled from a parameter sweep across N
        runs, so no analysis a consumer can author fills it. The refusal is
        evaluated off the exported dict alone — no silk import — because
        that is all a consumer has."""
        kinds = schema['_meta']['signal_kinds']
        can_author = {'tran', 'ac', 'dc', 'noise'}

        assert schema['p1db']['signal_inputs'] == [
            {'name': 'sig', 'kind': 'power_sweep'}]
        assert kinds['power_sweep']['produced_by'] == []
        assert not all(set(kinds[e['kind']]['produced_by']) & can_author
                       for e in schema['p1db']['signal_inputs'])
        # and it is refused for the SAME reason as the other two, not by a
        # special case: the domain it declares is one a consumer can author.
        assert schema['p1db']['domain'] in can_author

    def test_a_kind_that_names_producers_includes_the_metrics_own_domain(
            self, schema):
        """Drift guard. An ``ac``-domain metric whose slot says 'transient'
        would be offered a probe its own analysis cannot produce, and nothing
        else in the schema would notice."""
        kinds = schema['_meta']['signal_kinds']
        for name, slot, kind in self._slots(schema):
            produced_by = kinds[kind]['produced_by']
            if not produced_by:
                continue
            assert schema[name]['domain'] in produced_by, (name, slot, kind)

    def test_meta_signal_kinds_is_the_kinds_module_verbatim(self, schema):
        from silk.signals.kinds import signal_kinds_meta
        assert schema['_meta']['signal_kinds'] == signal_kinds_meta()

    def test_every_kind_in_use_is_in_the_published_table(self, schema):
        published = schema['_meta']['signal_kinds']
        for name, slot, kind in self._slots(schema):
            assert kind in published, (name, slot, kind)

    def test_the_kinds_survive_the_json_round_trip(self, schema):
        """AC#3 — the shape a consumer actually receives."""
        round_tripped = json.loads(export_metric_schema_json())
        assert round_tripped == json.loads(json.dumps(schema))
        assert round_tripped['error_norm']['signal_inputs'] == [
            {'name': 'sig', 'kind': 'any'},
            {'name': 'goal', 'kind': 'reference_waveform'},
        ]
        assert round_tripped['_meta']['format_version'] == '2.0'

    def test_the_kind_is_the_object_the_runtime_guard_would_use(self):
        """Reflected, not authored twice. The schema's string and the object
        a guard checks against (T-E) are the SAME SignalKind — that is the
        whole argument for the annotation form over a ``signal_kinds={}``
        dict beside ``param_units`` (design §4.2)."""
        hints = typing.get_type_hints(get_metric('integrated_noise'),
                                      include_extras=True)
        kind, = [m for m in hints['sig'].__metadata__
                 if isinstance(m, SignalKind)]
        assert kind is NoiseDensity.__metadata__[0]
        assert kind.produced_by == ('noise',)
        assert (metric_schema(get_metric('integrated_noise'))
                ['signal_inputs'][0]['kind'] == kind.name)

    @pytest.mark.parametrize('alias,expected', [
        (AnySignal, 'any'),
        (Transient, 'transient'),
        (FrequencyResponse, 'frequency_response'),
        (NoiseDensity, 'noise_density'),
        (ReferenceWaveform, 'reference_waveform'),
    ])
    def test_an_alias_is_an_erased_signal_annotation(self, alias, expected):
        """``Annotated`` is erased at runtime: the Python API is unchanged
        and a caller still passes a plain Signal."""
        base, kind = typing.get_args(alias)
        assert base is Signal
        assert kind.name == expected
