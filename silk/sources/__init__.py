"""Analog source elements and waveform shapes.

Independent sources (:class:`~silk.sources.VoltageSource`,
:class:`~silk.sources.CurrentSource`) carry a ``dc`` value, an ``ac`` magnitude
and at most one waveform: :class:`~silk.sources.PWL`,
:class:`~silk.sources.Pulse`, :class:`~silk.sources.Sine`,
:class:`~silk.sources.Exponential`, :class:`~silk.sources.SFFM`,
:class:`~silk.sources.AM`, :class:`~silk.sources.TransientNoise`,
:class:`~silk.sources.RandomSource`, or the finite bit waveforms
:class:`~silk.sources.BitPattern` and :class:`~silk.sources.PRBS`. Linear
controlled sources (:class:`~silk.sources.VoltageControlledVoltageSource`,
:class:`~silk.sources.VoltageControlledCurrentSource`,
:class:`~silk.sources.CurrentControlledVoltageSource`,
:class:`~silk.sources.CurrentControlledCurrentSource`; the short aliases
``VCVS``, ``VCCS``, ``CCVS`` and ``CCCS`` name the same classes) and behavioral
sources (:class:`~silk.sources.BehavioralVoltage`,
:class:`~silk.sources.BehavioralCurrent`) complete the set. Each source class
owns its SPICE emission (``to_spice``); the independent sources also evaluate
themselves for the behavioral engine (``to_numpy``) for the waveforms that
have a numpy model.

Two construction surfaces produce the same sources. The classes take explicit
keywords; the factories take the instance name and two terminals first and
come in two spellings that are the same function objects:
``sources.voltage.pulse(...)`` and ``sources.vpulse(...)`` (likewise
``sources.current.dc(...)`` and ``sources.idc(...)``). Values are in volts
for voltage sources, amps for current sources, seconds for times and hertz
for frequencies; a value may be a number, a SPICE value string such as
``'100p'`` or a :class:`~silk.circuit.Variable`.

Examples
--------
>>> from silk import sources
>>> print(sources.VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8).to_spice(), end='')
Vvdd vdd gnd DC 1.8
>>> sources.vdc('vdd', 'vdd', 'gnd', 1.8).to_spice() == sources.voltage.dc('vdd', 'vdd', 'gnd', 1.8).to_spice()
True
>>> sources.vdc is sources.voltage.dc
True
"""
from silk.sources._bits import prbs_bits
from silk.sources.waveforms import (
    PWL, Pulse, Sine, Exponential, SFFM, AM, AmplitudeModulated,
    TransientNoise, RandomSource, RandomType,
)
from silk.sources.classes import (
    SourceBase,
    VoltageSource, CurrentSource,
    VoltageControlledVoltageSource, VoltageControlledCurrentSource,
    CurrentControlledVoltageSource, CurrentControlledCurrentSource,
    BehavioralVoltage, BehavioralCurrent,
)

SingleFrequencyFM = SFFM
VCVS = VoltageControlledVoltageSource
VCCS = VoltageControlledCurrentSource
CCVS = CurrentControlledVoltageSource
CCCS = CurrentControlledCurrentSource

__all__ = [
    'PWL', 'Pulse', 'Sine', 'Exponential', 'SFFM', 'AM', 'TransientNoise', 'RandomSource',
    'RandomType', 'prbs_bits',
    'SourceBase', 'VoltageSource', 'CurrentSource',
    'VoltageControlledVoltageSource', 'VoltageControlledCurrentSource',
    'CurrentControlledVoltageSource', 'CurrentControlledCurrentSource',
    'BehavioralVoltage', 'BehavioralCurrent',
    'VCVS', 'VCCS', 'CCVS', 'CCCS',
    'source_schema', 'export_source_schema',
]

# Schema export (parallel to silk.elements.schema.export_schema for devices).
# Imported last: schema.py binds ``silk.sources`` and resolves classes at call
# time, so the package is fully constructed by now — no import cycle.
from silk.sources.schema import source_schema, export_source_schema  # noqa: E402

# Auto-export the waveform classes named in source_metadata.yaml so adding a
# kind is a one-place change (a class + one YAML line — no edit here).
from silk.sources.schema import _load_metadata as _load_src_meta  # noqa: E402
import silk.sources.waveforms as _wf_mod  # noqa: E402
_WAVEFORMS_AUTO_EXPORTED = True
for _wf_name in _load_src_meta()['waveforms'].values():
    globals()[_wf_name] = getattr(_wf_mod, _wf_name)
    if _wf_name not in __all__:
        __all__.append(_wf_name)

# Declarative source library (spec 2026-07-07): sources.voltage.pulse(...) (B)
# and the flat sources.vpulse(...) alias (A). Built from source_metadata.yaml.
from silk.sources.library import build_source_surfaces as _build_surfaces  # noqa: E402
_source_namespaces, _source_flat = _build_surfaces()
#: Namespace of voltage-source factories (``sources.voltage.dc``,
#: ``sources.voltage.pulse``, ...), the same function objects as the flat
#: ``v``-prefixed names (``sources.vdc``, ``sources.vpulse``, ...). Each builds
#: a :class:`~silk.sources.VoltageSource`.
voltage = _source_namespaces.get('voltage')
#: Namespace of current-source factories (``sources.current.dc``,
#: ``sources.current.pulse``, ...), the same function objects as the flat
#: ``i``-prefixed names (``sources.idc``, ``sources.ipulse``, ...). Each builds
#: a :class:`~silk.sources.CurrentSource`.
current = _source_namespaces.get('current')
globals().update(_source_flat)                 # vdc, vpulse, idc, … (same objects)
__all__ += ['voltage', 'current', *sorted(_source_flat)]
