"""Reject source timing syntax that the selected SPICE backend cannot honor."""
from silk.exceptions import NotSupportedError
from silk.simulation._ngspice_compat import NgspiceCompatibility
from silk.sources.classes import CurrentSource, VoltageSource
from silk.sources.waveforms import PWL, Pulse, SFFM, AM, TransientNoise, RandomSource


def emitted_waveform(source):
    """Return the one authored waveform, refusing conflicting slots."""
    if isinstance(source, (VoltageSource, CurrentSource)):
        return source.selected_waveform()


def validate_source_timing(source, backend):
    """Reject timing syntax the selected backend cannot honor."""
    wave = emitted_waveform(source)
    if backend == 'ngspice' and isinstance(wave, (AM, SFFM)):
        policy = NgspiceCompatibility()
        version = policy.version
        verified = policy.verified_waveforms_label
        if isinstance(wave, SFFM) and not policy.waveforms_verified:
            raise NotSupportedError(
                f'ngspice {version or "(version unknown)"} cannot emit '
                f'{type(wave).__name__} for {type(source).__name__} '
                f'{source.name!r}; only ngspice {verified} is verified for these '
                'waveforms. Newer releases start SFFM at 0 and clamp the modulation depth. '
                f'Select ngspice {verified} with SILK_NGSPICE_BIN or use '
                'a supported waveform such as explicit PWL points.')
        if isinstance(wave, AM) and (policy.release is None or policy.below_minimum):
            raise NotSupportedError(
                f'ngspice {version or "(version unknown)"} cannot emit AM for '
                f'{type(source).__name__} {source.name!r}; the version table '
                f'requires an identifiable ngspice release >= {policy.minimum}. '
                'Select a supported binary with SILK_NGSPICE_BIN or use '
                'explicit PWL points.')
        if isinstance(wave, AM) and not policy.has('am_new_order'):
            from silk.utils._utils import _parse_si
            from silk.exceptions import DeckTextError
            try:
                phase = _parse_si(wave._resolve('phases'))
            except DeckTextError:
                raise
            except (TypeError, ValueError):
                phase = None
            if phase != 0:
                raise NotSupportedError(
                    f'ngspice {version} source {source.name!r}: AM phases must '
                    'resolve to zero; this version ignores the phase token. '
                    'Use phases=0 or explicit PWL points.')
    if backend == 'xyce' and isinstance(wave, (AM, TransientNoise, RandomSource)):
        raise NotSupportedError(
            f'Xyce does not support {type(wave).__name__} for '
            f'{type(source).__name__} {source.name!r}; a DC clause can make '
            "Xyce silently ignore the waveform. Use simulator='ngspice'.")
    if backend == 'xyce' and isinstance(wave, SFFM):
        from silk.utils._utils import _parse_si
        for field in ('phasec', 'phases'):
            from silk.exceptions import DeckTextError
            try:
                value = _parse_si(wave._resolve(field))
            except DeckTextError:
                raise
            except (TypeError, ValueError):
                value = None
            if value != 0:
                raise NotSupportedError(
                    f'Xyce source {source.name!r}: SFFM {field} must resolve to zero; '
                    'nonzero or unresolved phases are not supported by this backend.')
    if isinstance(wave, PWL) and (wave.r is not None or wave.td is not None):
        if backend == 'ngspice' and isinstance(source, CurrentSource):
            policy = NgspiceCompatibility()
            version = policy.version
            release = policy.since('current_pwl_timing')
            if not policy.has('current_pwl_timing'):
                raise NotSupportedError(
                    f'ngspice {version or "(version unknown)"} cannot emit PWL '
                    f'r/td for CurrentSource {source.name!r}; ngspice >= {release} '
                    'is required. Point SILK_NGSPICE_BIN at a supported binary '
                    'or use explicit delayed/repeated points without r or td.')
        if backend == 'xyce' and wave.td is not None:
            raise NotSupportedError(
                f'{backend} does not support the emitted PWL r/td syntax for '
                f'{type(source).__name__} {source.name!r}. Use explicit delayed/'
                'repeated points without r or td.')
    if backend == 'xyce' and isinstance(wave, PWL) and wave.r is not None:
        from silk.utils._utils import _parse_si
        repeat = _parse_si(wave._resolve('r'))
        # Xyce retains the last value at *every* exact join; the NumPy model
        # follows ngspice's repeat-start value after the first traversal.
        # Continuous joins agree on both backends. Evaluation also validates r.
        start_value = wave.to_numpy([repeat])[0]
        if start_value != float(wave.points[-1][1]):
            raise NotSupportedError(
                f'Xyce PWL repeat join for source {source.name!r} is discontinuous '
                'and differs from the NumPy model at exact joins. Use equal '
                'repeat-start/final values or explicit repeated points.')
    if backend == 'xyce' and isinstance(wave, Pulse) and wave.np is not None:
        raise NotSupportedError(
            f'Xyce does not support Pulse np for source {source.name!r}; '
            'use np=None for unlimited pulses or an explicit finite PWL.')
