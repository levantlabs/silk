"""The @metric return-annotation guard and the scalar catalog (TASK-173/TASK-233).

TASK-173 decorated the 51 Signal-in, float-out metrics in :mod:`silk.metrics`
(the exported count went 10 -> 61) and made :func:`silk.signals.schema.metric`
refuse anything whose return annotation is not a scalar. TASK-180 added the
62nd, ``dc.p1db``, by giving it the power-sweep Signal and the ``float``
return it should always have had; TASK-184 held it back out of the schema and
TASK-200 put it back under ``kind='power_sweep'``, so the catalog was 62;
TASK-170 added the 63rd, ``noise.noise_density_at_freq``. TASK-287 removed
``transient.sh_settling_error``, taking the catalog back to 62.

Two families of test live here:

* the GUARD — every one of the 32 Signal-taking functions in ``silk.metrics``
  that does NOT return a scalar is refused without outputs=, with a message
  that distinguishes "returns X, which is not a scalar" from "has no return
  annotation". The refusal is a return-annotation check at the definition, NOT
  a skip list, so the test decorates the LIVE functions rather than stubs.
* the CATALOG — the inventory is rediscovered by reflection on every run and
  compared against the registry, so a metric added, renamed or re-typed later
  cannot silently fall in or out of the exported schema.
"""
import importlib
import inspect
import typing

import pytest

from silk.signals.base import Signal
from silk.signals.schema import (
    NonScalarMetricError,
    Transient,
    _REGISTRY,
    _is_signal_param,
    export_metric_schema,
    metric,
)

METRIC_MODULES = (
    'waveform', 'statistics', 'timing', 'jitter', 'eye', 'power',
    'spectral', 'rf', 'converter', 'fom', 'ac', 'noise',
)

#: The 32 Signal-taking functions in silk.metrics that do NOT return a scalar,
#: as re-measured on the TASK-172 merge. ``ac.gain_margins`` is the odd one
#: out: it carries no return annotation at all rather than a non-scalar one.
#: ``test_non_scalar_inventory_is_exactly_this_list`` re-derives the set live,
#: so this list cannot rot into a stale exclusion.
#:
#: ``('eye', 'eye_diagram')`` is back (TASK-226.12): the replacement
#: returns the same ``EyeResult``, so the decorator refuses it for the same
#: reason the withdrawn one was refused when no outputs are declared.
NON_SCALAR = [
    ('ac', 'group_delay'), ('ac', 'psrr_db'), ('ac', 'gain_margins'),
    ('converter', 'extract_code_transitions'), ('converter', 'inl'), ('converter', 'dnl'), ('converter', 'static_errors'),
    ('converter', 'is_monotonic'), ('converter', 'missing_codes'),
    ('noise', 'noise_figure_vs_freq'),  # TASK-414: NF per sweep point, a Signal
    ('spectral', 'fft'), ('spectral', 'sfdr'),
    ('spectral', 'harmonic_levels_db'), ('spectral', 'psd'),
    ('spectral', 'harmonic'),
    ('rf', 'mix_product'), ('rf', 'evm_constellation'),
    ('spectral', 'harmonic_freqs'), ('spectral', 'welch_psd'),
    ('spectral', 'coherence'),
    ('waveform', 'crossings'), ('timing', 'propagation_delays'),
    ('waveform', 'running_integral'), ('waveform', 'moving_average'),
    ('waveform', 'derivative'), ('timing', 'cycle_period'),
    ('timing', 'cycle_frequency'), ('timing', 'cycle_duty'),
    ('timing', 'pulse_widths'),
    ('jitter', 'period_deviation'), ('jitter', 'cycle_to_cycle_difference'),
    ('timing', 'rise_times'), ('timing', 'fall_times'),
    ('timing', 'slew_rates'),
    ('eye', 'eye_diagram'), ('statistics', 'histogram'),
    ('waveform', 'convolve'),
    ('waveform', 'lshift'), ('waveform', 'rshift'), ('waveform', 'clip'), ('waveform', 'window'), ('waveform', 'flip'),
    ('waveform', 'sample'), ('jitter', 'tie'),
    ('jitter', 'allan_deviation'), ('waveform', 'cross_correlation'),
]

#: The one member of NON_SCALAR that is refused for the OTHER reason.
UNANNOTATED = ('ac', 'gain_margins')


def _takes_a_signal(fn) -> bool:
    """Does *fn* take a waveform input? Asked the way the exporter asks.

    Rewritten for schema v2 (TASK-196). The v1 form compared the annotation
    TEXT against ``'Signal'``, which now misses the whole exported catalog:
    every decorated input is annotated ``Transient`` /
    ``FrequencyResponse`` / … , and under PEP 563 the raw annotation is that
    alias's NAME. Resolving the hints and calling
    :func:`silk.signals.schema._is_signal_param` — the ONE place the
    signal/scalar split is decided — is what keeps this inventory a
    re-measurement of the live package rather than a parallel rule that can
    disagree with the exporter. Undecorated helpers still spell their inputs
    with a bare ``Signal`` and are still counted.
    """
    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}
    return any(_is_signal_param(hints.get(n))
               for n in inspect.signature(fn).parameters)


def _public_functions(mod_name):
    """Yield ``(name, fn)`` for functions DEFINED in ``silk.metrics.<mod>``."""
    mod = importlib.import_module(f'silk.metrics.{mod_name}')
    for name in dir(mod):
        if name.startswith('_'):
            continue
        fn = getattr(mod, name)
        if inspect.isfunction(fn) and fn.__module__ == mod.__name__:
            yield name, fn


def _inventory():
    """Re-measure silk.metrics: (scalar_in_signal, non_scalar, scalar_only)."""
    scalar, non_scalar, scalar_only = [], [], []
    for mod_name in METRIC_MODULES:
        for name, fn in _public_functions(mod_name):
            sig = inspect.signature(fn)
            takes_signal = _takes_a_signal(fn)
            ret = sig.return_annotation
            ret_name = (ret if isinstance(ret, str)
                        else getattr(ret, '__name__', str(ret)))
            if not takes_signal:
                scalar_only.append((mod_name, name))
            elif ret is inspect.Signature.empty or ret_name not in ('float', 'float | complex'):
                non_scalar.append((mod_name, name))
            else:
                scalar.append((mod_name, name))
    return scalar, non_scalar, scalar_only


class TestScalarReturnGuard:
    """AC#2 / AC#3 — the decorator refuses every non-scalar return."""

    @pytest.mark.parametrize('mod_name,fn_name', NON_SCALAR,
                             ids=[f'{m}.{f}' for m, f in NON_SCALAR])
    def test_non_scalar_metric_is_refused(self, mod_name, fn_name):
        mod = importlib.import_module(f'silk.metrics.{mod_name}')
        fn = getattr(mod, fn_name)
        with pytest.raises(NonScalarMetricError) as exc:
            metric(domain='tran', unit='V')(fn)
        msg = str(exc.value)
        assert fn_name in msg                      # names the function
        # and it did not sneak into the registry or gain metadata
        if fn_name in {'sfdr', 'eye_diagram', 'inl', 'dnl', 'static_errors', 'evm_constellation'}:
            assert fn._silk_metric['outputs']
            assert fn_name not in export_metric_schema()
        else:
            assert fn_name not in _REGISTRY
            assert not hasattr(fn, '_silk_metric')

    def test_the_two_refusals_are_distinguishable(self):
        """'returns X, which is not a scalar' vs 'has no return annotation'."""
        from silk.metrics.ac import gain_margins, group_delay

        with pytest.raises(NonScalarMetricError) as unannotated:
            metric(domain='ac', unit='dB')(gain_margins)
        with pytest.raises(NonScalarMetricError) as non_scalar:
            metric(domain='ac', unit='s')(group_delay)

        assert 'no return annotation' in str(unannotated.value)
        assert 'not a scalar' not in str(unannotated.value)
        assert "returns 'Signal', which is not a scalar" in str(non_scalar.value)
        assert 'no return annotation' not in str(non_scalar.value)

    def test_error_names_the_offending_return_type(self):
        """AC#2: the message names the function AND its return type."""
        from silk.metrics.converter import is_monotonic
        from silk.metrics.spectral import harmonic

        with pytest.raises(NonScalarMetricError, match=r"returns 'bool'"):
            metric(domain='dc', unit='')(is_monotonic)
        with pytest.raises(NonScalarMetricError, match=r"returns 'complex'"):
            metric(domain='tran', unit='V')(harmonic)

    def test_bool_is_refused_even_though_it_subclasses_int(self):
        """The check compares type NAMES, never ``issubclass`` — ``bool`` is an
        ``int`` subclass in Python and would otherwise slip through."""
        def _bool_probe(sig: Transient) -> bool:
            return True

        with pytest.raises(NonScalarMetricError, match=r"returns 'bool'"):
            metric(domain='tran', unit='')(_bool_probe)
        assert '_bool_probe' not in _REGISTRY

    def test_qualified_non_scalar_annotation_cannot_sneak_past(self):
        """``np.ndarray`` is compared on its trailing bare name."""
        def _array_probe(sig: Transient) -> 'np.ndarray':
            return None

        with pytest.raises(NonScalarMetricError, match=r"np\.ndarray"):
            metric(domain='tran', unit='V')(_array_probe)

    def test_int_return_is_accepted(self):
        """A count-returning metric is still a scalar with one unit."""
        def _int_probe(sig: Transient) -> int:
            return 0

        try:
            metric(domain='tran', unit='')(_int_probe)
            assert _REGISTRY['_int_probe'] is _int_probe
        finally:
            _REGISTRY.pop('_int_probe', None)

    def test_non_scalar_metric_error_is_a_type_error(self):
        assert issubclass(NonScalarMetricError, TypeError)


class TestInventoryCannotRot:
    """The lists above are re-derived from the live package on every run."""

    def test_non_scalar_inventory_is_exactly_this_list(self):
        _, non_scalar, _ = _inventory()
        assert sorted(non_scalar) == sorted(NON_SCALAR)
        # 32 at TASK-172; back to 32 with eye_diagram's replacement
        # (TASK-226.12), which returns EyeResult as the old one did; 33
        # since TASK-363 added propagation_delays (every delay, an array);
        # 34 at TASK-366: inst_frequency -> cycle_frequency, + cycle_period;
        # 36 at TASK-367: + cycle_duty (Signal), + pulse_widths (array).
        # 39 at TASK-372: + rise_times, fall_times, slew_rates (arrays).
        # 40 at TASK-381: + moving_average (Signal).
        # 41 at TASK-383: + rshift (Signal).
        # 42 at TASK-384: + window (the x-range cut; clip now limits y).
        # 43 at TASK-382: + histogram (HistogramResult).
        # 44 at TASK-399: + static_errors (a result object).
        # 46 at TASK-417: + period_deviation, cycle_to_cycle_difference.
        # 47 at TASK-414: + noise_figure_vs_freq (Signal).
        # 46 at TASK-334: - psd_baseband (removed).
        assert len(NON_SCALAR) == 46

    #: Signal-in, float-out metrics deliberately kept OUT of the schema.
    #: EMPTY since TASK-200: ``p1db`` was the only entry, held out by
    #: TASK-184 because its ``xlabel='power'`` input is one no silk analysis
    #: produces, and schema v2 gave that fact a place to live — the slot
    #: declares ``kind='power_sweep'``, whose ``produced_by`` is empty, so a
    #: consumer refuses to OFFER it while a Python caller can still find it.
    #: The mechanism stays as the deliberate-exception pattern: a metric that
    #: stops being exported fails this suite until someone names it here on
    #: purpose.
    UNEXPORTED_BY_DESIGN = set()

    def test_every_signal_in_float_out_metric_is_exported(self):
        """AC#1 — no Signal-in, float-out metric is left out of the schema,
        except any the owner ruled out by name (none today)."""
        scalar, _, _ = _inventory()
        missing = {n for m, n in scalar if n not in _REGISTRY}
        assert missing == self.UNEXPORTED_BY_DESIGN
        assert len(scalar) == 80          # TASK-414 noise_figure_at_freq, band_noise_figure; TASK-417 period_jitter_pp, cycle_to_cycle_jitter_pp; TASK-412 removed input_referred_offset, kickback, hold_droop; TASK-379 variance; TASK-397 oip3_dbm, iip3_dbm; TASK-386 relative_error_norm; TASK-399 offset_error_lsb/gain_error_lsb/full_scale_error_lsb; TASK-233 eight wrappers; TASK-323.3 final_value; TASK-369 evm_percent; TASK-363 comparator_prop_delay removed; TASK-366 average_period/average_frequency replace period_zero_cross; TASK-226.20 crossing_jitter_rms

    def test_scalar_only_helpers_stay_undecorated(self):
        """AC#5 — the closed-form helpers take no Signal, so they are not
        waveform metrics and are deliberately out of the schema. 14 at
        TASK-173; 13 since TASK-180 gave ``dc.p1db`` the power-sweep Signal
        it should always have taken; 12 since TASK-210 removed
        ``rms_jitter_from_phase_noise``; 13 again since TASK-369 added
        ``spectral.modulation_grid`` (a name in, a constellation out)."""
        _, _, scalar_only = _inventory()
        assert len(scalar_only) == 13
        assert 'p1db' not in {n for _, n in scalar_only}
        leaked = [f'{m}.{n}' for m, n in scalar_only if n in _REGISTRY]
        assert leaked == []
        # a representative sample, so a rename is visible in the diff
        names = {n for _, n in scalar_only}
        assert {'noise_figure', 'kT_over_C', 'sum_uncorrelated',
                'walden_fom', 'cascade_noise_figure'} <= names


class TestEnlargedCatalog:
    """AC#1 / AC#4 — the arrivals, their domains and their units."""

    @pytest.fixture(scope='class')
    def schema(self):
        return export_metric_schema()

    def test_count_and_domain_split(self, schema):
        assert schema['_meta']['metric_count'] == 80
        rows = {k: v for k, v in schema.items() if not k.startswith('_')}
        assert len(rows) == 80
        domains = {}
        for spec in rows.values():
            domains[spec['domain']] = domains.get(spec['domain'], 0) + 1
        # 36 transient + 9 spectral arrivals join nothing (tran had 0);
        # ac 9 already-decorated + 4; noise 1 + 2, +1 since TASK-170
        # (noise_density_at_freq). The single 'dc' entry is p1db,
        # re-decorated at TASK-200 under kind='power_sweep' — in the
        # catalog and mechanically non-offerable, rather than invisible.
        # TASK-287 removed transient.sh_settling_error: tran 45 -> 44.
        # TASK-210 removed noise.phase_noise_at_offset: noise 4 -> 3.
        assert domains == {'tran': 54, 'ac': 13, 'noise': 5, 'dc': 8}  # TASK-414 two noise figures; TASK-417 two _pp; TASK-412 three removed; TASK-379 variance; TASK-397 oip3_dbm, iip3_dbm; TASK-386 relative_error_norm; TASK-323.3 final_value; TASK-369 evm_percent; TASK-363 comparator_prop_delay removed; TASK-366 average_frequency; TASK-226.20 crossing_jitter_rms

    def test_the_ten_pre_existing_metrics_are_untouched(self, schema):
        """TASK-172's ten keep their names and domains. ``integrated_noise``'s
        unit is the one deliberate exception: TASK-288 corrected it from a
        hard-coded ``'V'`` (wrong for a current-noise probe) to
        ``'[y]*sqrt([x])'``, the input-relative form — see
        ``test_metric_schema.test_integrated_noise_unit_is_relative_to_the_inputs_own_unit``."""
        before = {
            'dc_gain_dB': ('ac', 'dB'), 'peak_gain_dB': ('ac', 'dB'),
            'gain_at_freq': ('ac', 'V/V'), 'bandwidth_3dB': ('ac', 'Hz'),
            'unity_gain_freq': ('ac', 'Hz'), 'phase_at_freq': ('ac', 'deg'),
            'phase_margin': ('ac', 'deg'), 'gain_margin': ('ac', 'dB'),
            'cmrr_dB': ('ac', 'dB'),
            'integrated_noise': ('noise', '[y]*sqrt([x])'),
        }
        for name, (domain, unit) in before.items():
            assert (schema[name]['domain'], schema[name]['unit']) == (domain, unit), name

    @pytest.mark.parametrize('name,unit', [
        # units a reviewer is most likely to want to see pinned
        ('rise_time', 's'), ('fall_time', 's'), ('settling_time', 's'),
        ('propagation_delay', 's'), ('slew_rate', '[y]/s'),
        ('dynamic_energy_per_cycle', 'J'), ('average_power', 'W'),
        ('enob', 'bits'), ('sndr', 'dB'), ('sinad', 'dB'), ('thd', 'dB'),
        ('thd_percent', '%'), ('noise_floor_dbfs', 'dBFS'),
        ('one_over_f_corner', 'Hz'),
        ('noise_density_at_freq', '[y]'),
        ('overshoot', ''), ('undershoot', ''), ('average_duty', ''),
        ('q_from_peak', ''), ('peak_to_peak', '[y]'), ('crossing', '[x]'),
    ])
    def test_authored_units(self, schema, name, unit):
        assert schema[name]['unit'] == unit

    def test_thd_and_thd_percent_do_not_share_a_unit(self, schema):
        """The trap the unit authoring exists to avoid: same measurement, two
        scales. ``thd`` is a dB ratio, ``thd_percent`` is a percentage."""
        assert schema['thd']['unit'] == 'dB'
        assert schema['thd_percent']['unit'] == '%'

    def test_unit_vocabulary_is_closed(self, schema):
        """Catches a typo'd unit (``'Hz '``, ``'dBc/hz'``) that would otherwise
        reach schema consumers and a Spec unnoticed."""
        allowed = {
            '', 's', 'V', 'W', 'J', 'Hz', 'dB', 'dBc', 'LSB', 'deg', 'bits', '%',
            'V/V', 'dBFS', 'dBm', 'dBc/Hz', 'Hz*V/V',
            '[x]', '[y]', '[y]*[x]', '[y]^2', '[y]/s', '[y]*sqrt([x])',
        }
        used = {v['unit'] for k, v in schema.items() if not k.startswith('_')}
        assert used <= allowed, sorted(used - allowed)

    def test_param_unit_vocabulary_is_closed(self, schema):
        # 'K' since TASK-414: the NF metrics' temp_sim_k / temp_ref_k.
        allowed = {'s', 'Hz', 'V', '[x]', '[y]', 'ohm', 'K'}
        used = set()
        for k, v in schema.items():
            if k.startswith('_'):
                continue
            used |= {p['unit'] for p in v['parameters'] if p['unit'] is not None}
        assert used <= allowed, sorted(used - allowed)

    def test_declared_param_units_name_real_params(self, schema):
        """A ``param_units`` key that matches no parameter reflects as nothing
        and would silently do nothing."""
        for name, spec in schema.items():
            if name.startswith('_'):
                continue
            fn = _REGISTRY[name]
            declared = set(fn._silk_metric['param_units'])
            actual = set(inspect.signature(fn).parameters)
            assert declared <= actual, (name, sorted(declared - actual))
            # a Signal input must never carry a scalar param unit.
            # schema v2: signal_inputs entries are {name, kind} objects.
            assert declared.isdisjoint(
                {e['name'] for e in spec['signal_inputs']}), name


def test_consumer_reads_the_enlarged_schema_unchanged():
    """The schema can be cached and read through its existing entry point:
    ``from silk.signals.schema import export_metric_schema`` then cache the
    live result, excluding keys that start with ``_`` from ``metric_names()``.
    """
    import json

    from silk.signals.schema import export_metric_schema as schema_entrypoint

    cached = schema_entrypoint()
    names = sorted(k for k in cached if not k.startswith('_'))
    assert len(names) == 80
    assert cached['_meta']['metric_count'] == 80
    # domain maps to a published analysis kind
    assert {cached[n]['domain'] for n in names} <= {'tran', 'ac', 'noise', 'dc'}
    # the cache is a plain dict consumers may serialise
    json.dumps(cached)


@pytest.mark.parametrize('annotation', ['float | complex', float | complex])
def test_real_or_complex_return_is_registered(annotation):
    def _complex_probe(sig: Transient):
        return 1+2j
    _complex_probe.__annotations__['return'] = annotation
    try:
        metric(domain='tran', unit='V')(_complex_probe)
        assert _REGISTRY['_complex_probe'] is _complex_probe
        assert export_metric_schema()['_complex_probe']['unit'] == 'V'
    finally:
        _REGISTRY.pop('_complex_probe', None)


def test_public_metric_definitions_match_purpose_modules():
    """Public metric names resolve to one definition in their purpose module."""
    import silk.metrics as metrics
    expected = {
        'waveform': (
            'crossing', 'crossings', 'value_at', 'final_value', 'argmin_x',
            'argmax_x', 'peak', 'peak_to_peak', 'minimum', 'maximum', 'derivative',
            'integral', 'running_integral', 'moving_average', 'clip', 'window',
            'flip', 'lshift', 'rshift', 'sample', 'convolve', 'cross_correlation',
        ),
        'statistics': (
            'average', 'rms', 'stddev', 'variance', 'histogram', 'HistogramResult',
            'error_norm', 'relative_error_norm',
        ),
        'timing': (
            'rise_time', 'fall_time', 'slew_rate', 'rise_times', 'fall_times',
            'slew_rates', 'settling_time', 'propagation_delay', 'propagation_delays',
            'overshoot', 'undershoot', 'on_time', 'off_time', 'pulse_width',
            'pulse_widths', 'average_duty', 'cycle_duty', 'average_period',
            'average_frequency', 'cycle_period', 'cycle_frequency', 'hysteresis',
        ),
        'jitter': (
            'period_jitter', 'period_jitter_pp', 'cycle_to_cycle_jitter',
            'cycle_to_cycle_jitter_pp', 'period_deviation',
            'cycle_to_cycle_difference', 'crossing_jitter_rms', 'tie',
            'long_term_jitter', 'allan_deviation',
        ),
        'eye': (
            'eye_diagram', 'EyeResult', 'eye_height', 'eye_width',
        ),
        'power': (
            'average_power', 'dynamic_energy_per_cycle',
        ),
        'spectral': (
            'fft', 'psd', 'welch_psd', 'spectral_power', 'coherence', 'harmonic',
            'harmonic_freqs', 'harmonic_level_dbc', 'harmonic_levels_db', 'thd',
            'thd_percent', 'noise_floor_dbfs', 'snr', 'sndr', 'sinad', 'sfdr',
            'sfdr_db', 'SFDR',
        ),
        'rf': (
            'p1db', 'iip3_dbm', 'oip3_dbm', 'imd3_db', 'mix_product',
            'evm_constellation', 'evm_percent', 'EVMResult', 'modulation_grid',
        ),
        'converter': (
            'extract_code_transitions', 'inl', 'INL', 'inl_max_lsb', 'dnl', 'DNL',
            'dnl_max_lsb', 'missing_codes', 'missing_code_count', 'is_monotonic',
            'non_monotonic_steps', 'static_errors', 'StaticErrors',
            'offset_error_lsb', 'gain_error_lsb', 'full_scale_error_lsb', 'enob',
            'aperture_jitter_snr_db',
        ),
        'fom': (
            'walden_fom', 'schreier_fom', 'iip3_extrapolation', 'pae',
            'dc_to_rf_efficiency',
        ),
        'ac': (
            'dc_gain_dB', 'peak_gain_dB', 'gain_at_freq', 'phase_at_freq',
            'bandwidth_3dB', 'unity_gain_freq', 'gain_bw_product', 'phase_margin',
            'gain_margin', 'gain_margins', 'group_delay', 'passband_ripple_db',
            'stopband_attenuation_db', 'q_from_peak', 'psrr_db', 'cmrr_dB',
        ),
        'noise': (
            'noise_figure', 'noise_figure_from_factor', 'noise_figure_at_freq',
            'noise_figure_vs_freq', 'band_noise_figure', 'cascade_noise_figure',
            'input_referred_noise', 'integrated_noise', 'noise_density_at_freq',
            'one_over_f_corner', 'kT_over_C', 'sum_uncorrelated',
        ),
    }
    names = {name for group in expected.values() for name in group}
    assert len(names) == 146
    assert set(metrics.__all__) == names | set(expected)
    assert len(metrics.__all__) == len(names) + len(expected)
    for module, group in expected.items():
        defining_module = importlib.import_module('silk.metrics.' + module)
        for name in group:
            obj = getattr(metrics, name)
            assert obj is getattr(defining_module, name)
            assert obj.__module__ == defining_module.__name__
            if name in _REGISTRY:
                assert _REGISTRY[name] is obj
