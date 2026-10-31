"""Source YAML-factory + generic waveform slot (spec 2026-07-07).

Step 1 here: the additive generic ``waveform=`` slot on VoltageSource/
CurrentSource (spec §5.3, B1) — lets a NEW waveform kind emit without editing
the classes, with the same additive DC layout as dedicated slots.
"""


class _DummyWave:
    """A stand-in for a new waveform kind: anything with a to_spice()."""
    def to_spice(self):
        return 'DUMMY(1 2 3)'


class TestGenericWaveformSlot:
    def test_voltage_generic_slot_emits(self):
        from silk.sources import VoltageSource
        v = VoltageSource('vx', positive='a', negative='0', waveform=_DummyWave())
        assert v.to_spice() == 'Vvx__silk_wave a __silk_src_Vvx DUMMY(1 2 3)\nVvx__silk_dc __silk_src_Vvx 0 DC 0\n'

    def test_voltage_generic_slot_with_dc_offset(self):
        from silk.sources import VoltageSource
        v = VoltageSource('vx', positive='a', negative='0', dc=0.9, waveform=_DummyWave())
        assert v.to_spice() == 'Vvx__silk_wave a __silk_src_Vvx DUMMY(1 2 3)\nVvx__silk_dc __silk_src_Vvx 0 DC 0.9\n'

    def test_current_generic_slot_emits(self):
        from silk.sources import CurrentSource
        i = CurrentSource('ix', positive='a', negative='0', waveform=_DummyWave())
        assert i.to_spice() == 'Iix__silk_wave a 0 DUMMY(1 2 3)\nIix__silk_dc a 0 DC 0\n'

    def test_existing_per_kind_pulse_unchanged(self):
        """The dedicated pulse path still emits its waveform."""
        from silk.sources import VoltageSource, Pulse
        v = VoltageSource('vx', positive='a', negative='0',
                          pulse=Pulse(v1=0, v2=1.8, tr='1n', tf='1n', pw='5n', per='10n'))
        assert 'PULSE' in v.to_spice().upper()

    def test_per_kind_and_generic_conflict(self):
        import pytest
        from silk.exceptions import CircuitError
        from silk.sources import VoltageSource, Pulse
        with pytest.raises(CircuitError, match='vx.*pulse.*waveform'):
            VoltageSource('vx', positive='a', negative='0',
                          pulse=Pulse(v1=0, v2=1.8, tr='1n', tf='1n', pw='5n', per='10n'),
                          waveform=_DummyWave())


class TestSchemaCompositionContract:
    """The source schema must describe how to compose a waveform
    (kwarg + constructor) and that waveforms are mutually exclusive, instead of
    hardcoding silk's convention."""

    def test_waveform_kind_has_kwarg_and_code_template(self):
        from silk.sources.schema import export_source_schema
        s = export_source_schema()
        p = s['VoltageSource']['waveforms']['pulse']
        assert p['kwarg'] == 'pulse'                      # the source-ctor slot
        assert p['code_template'].startswith('sources.Pulse(')   # how to build it
        assert p['class'] == 'Pulse'

    def test_waveforms_marked_mutually_exclusive(self):
        from silk.sources.schema import export_source_schema
        s = export_source_schema()
        assert s['VoltageSource']['waveform_exclusive'] is True
        assert s['CurrentSource']['waveform_exclusive'] is True

    def test_exported_contract_still_has_symbol_and_reflected_params(self):
        from silk.sources.schema import export_source_schema
        s = export_source_schema()
        assert s['VoltageSource']['symbol'] == 'VSource'
        assert {p['name'] for p in s['VoltageSource']['parameters']} == {'dc', 'ac'}


class TestFactorySurface:
    def test_namespaced_scalar_dc(self):
        import silk.sources as sources
        v = sources.voltage.dc('vdd', 'vdd', '0', 1.8)
        assert v.to_spice() == 'Vvdd vdd 0 DC 1.8\n'

    def test_flat_alias_is_same_object_as_namespaced(self):
        import silk.sources as sources
        assert sources.vdc is sources.voltage.dc
        assert sources.vpulse is sources.voltage.pulse
        assert sources.idc is sources.current.dc

    def test_flat_scalar_and_current(self):
        import silk.sources as sources
        assert sources.vdc('vdd', 'vdd', '0', 1.8).to_spice() == 'Vvdd vdd 0 DC 1.8\n'
        assert sources.idc('ib', 'a', '0', 1e-3).to_spice() == 'Iib a 0 DC 0.001\n'

    def test_waveform_byte_identical_to_class(self):
        import silk.sources as sources
        from silk.sources import VoltageSource, Pulse
        fac = sources.vpulse('clk', 'clk', '0', v1=0, v2=1.8,
                             td='1n', tr='1n', tf='1n', pw='5n', per='10n')
        hand = VoltageSource('clk', positive='clk', negative='0',
                             pulse=Pulse(v1=0, v2=1.8, td='1n', tr='1n', tf='1n', pw='5n', per='10n'))
        assert fac.to_spice() == hand.to_spice()

    def test_both_pn_and_positive_negative(self):
        import silk.sources as sources
        a = sources.vdc('vdd', 'vdd', '0', 1.8)
        b = sources.vdc('vdd', positive='vdd', negative='0', value=1.8)
        assert a.to_spice() == b.to_spice() == 'Vvdd vdd 0 DC 1.8\n'

    def test_dc_offset_coexists_with_waveform(self):
        import silk.sources as sources
        v = sources.vpulse('clk', 'clk', '0', v1=0, v2=1.8, td='1n', tr='1n',
                           tf='1n', pw='5n', per='10n', dc=0.9)
        assert v.to_spice() == ('Vclk__silk_wave clk __silk_src_Vclk '
                                'PULSE(0 1.8 1n 1n 1n 5n 10n)\n'
                                'Vclk__silk_dc __silk_src_Vclk 0 DC 0.9\n')

    def test_scalar_requires_value_no_silent_zero(self):
        import silk.sources as sources
        import pytest
        with pytest.raises(TypeError):
            sources.vdc('vz', 'a', '0')                 # no value -> must raise
        with pytest.raises(TypeError):
            sources.vdc('vz', 'a', '0', 1.8, value=2.0)  # positional + keyword conflict

    def test_builds_the_right_class(self):
        import silk.sources as sources
        from silk.sources import VoltageSource, CurrentSource
        assert isinstance(sources.vsine('v', 'a', '0', voffset=0, vampl=1, freq='1meg'), VoltageSource)
        assert isinstance(sources.idc('i', 'a', '0', 1e-3), CurrentSource)


class TestScalarCtorAcceptsQuantityKwarg:
    def test_vdc_accepts_dc_kwarg(self):
        from silk import sources
        s = sources.vdc('v1', p='a', n='b', dc=1.8)
        assert s.to_spice().strip().startswith('Vv1')

    def test_vdc_value_still_works(self):
        from silk import sources
        assert sources.vdc('v1', p='a', n='b', value=1.8).to_spice().strip().startswith('Vv1')

    def test_idc_accepts_dc_kwarg(self):
        from silk import sources
        assert sources.idc('i1', p='a', n='b', dc=1e-3).to_spice().strip().startswith('Ii1')

    def test_vdc_dc_and_value_conflict_raises(self):
        from silk import sources
        import pytest
        with pytest.raises(TypeError):
            sources.vdc('v1', p='a', n='b', value=1.0, dc=2.0)


class TestAddDispatch:
    def _tb(self):
        from silk.circuit.modules import Module
        from tests._env import scratch_dir
        return Module('tb', scratch_dir(), 'tb.cir', pdk_details=None, module_name=None)

    def test_ckt_add_routes_source_to_add_source(self):
        import silk.sources as sources
        tb = self._tb()
        v = sources.vdc('vdd', 'vdd', '0', 1.8)
        tb.add(v)                                   # today: AttributeError
        assert 'vdd' in tb.sig_sources_dict

    def test_add_still_idempotent_for_source(self):
        import silk.sources as sources
        tb = self._tb()
        v = sources.vpulse('clk', 'clk', '0', v1=0, v2=1.8, td='1n', tr='1n',
                           tf='1n', pw='5n', per='10n')
        tb.add(v)
        tb.add(v)                                   # re-add is harmless
        assert list(tb.sig_sources_dict).count('clk') == 1

    def test_add_source_of_factory_source_still_works(self):
        import silk.sources as sources
        tb = self._tb()
        tb.add_source(sources.idc('ib', 'a', '0', 1e-3))   # explicit path unchanged
        assert 'ib' in tb.sig_sources_dict

    def test_source_autoregisters_inside_module_context(self):
        """The metaclass registers a constructed source into the active module;
        the factory (which builds real VoltageSource/CurrentSource) inherits it."""
        import silk.sources as sources
        tb = self._tb()
        with tb:
            sources.vdc('vdd', 'vdd', '0', 1.8)
        assert 'vdd' in tb.sig_sources_dict


class TestExtensibility:
    def test_new_waveform_kind_needs_no_class_edit(self, monkeypatch):
        """G3 (the whole point): a brand-new waveform kind + a trivial emitter,
        declared only in metadata, auto-produces sources.v<kind> and emits via
        the generic waveform= slot — with NO edit to VoltageSource."""
        import silk.sources.library as lib
        from silk.sources import waveforms as wf

        class _Ramp:
            def __init__(self, slope):
                self.slope = slope

            def to_spice(self):
                return f'RAMP({self.slope})'

        monkeypatch.setattr(wf, 'Ramp', _Ramp, raising=False)
        base = lib._load_metadata()
        fake = dict(base)
        fake['waveforms'] = dict(base['waveforms'], ramp='Ramp')
        monkeypatch.setattr(lib, '_load_metadata', lambda: fake)

        _ns, flat = lib.build_source_surfaces()
        assert 'vramp' in flat and 'iramp' in flat
        v = flat['vramp']('vr', 'a', '0', slope=2)     # generic slot; no class edit
        assert v.to_spice() == 'Vvr__silk_wave a __silk_src_Vvr RAMP(2)\nVvr__silk_dc __silk_src_Vvr 0 DC 0\n'


class TestFourSurfaceContract:
    """Adding a waveform kind = 1 emitter class + 1 metadata line reaches the
    Python API, schema, catalog, and codegen with no other edits.

    Surface 1 (API/factory) is covered by
    ``TestExtensibility.test_new_waveform_kind_needs_no_class_edit`` above (a
    new kind reaches ``sources.v<kind>``/``sources.i<kind>``). This test covers
    the other three surfaces for the same kind of change: the reflected
    schema, the device catalog (``export_schema``), and codegen's kind map.

    Note this test monkeypatches a *different* target than
    ``TestExtensibility`` above: it patches ``silk.sources.schema._load_metadata``
    (and clears codegen's ``_source_meta`` lru_cache) because it exercises the
    schema/catalog/codegen surfaces, which read metadata through
    ``silk.sources.schema``; ``TestExtensibility`` patches the factory/library
    path (``silk.sources.library._load_metadata``) because it only exercises
    the Python-API surface built by ``silk.sources.library``. This is not
    accidental duplication — each test patches the loader its own surface
    actually reads from.
    """

    def test_new_kind_reaches_all_surfaces(self, monkeypatch, request):
        from silk.sources import schema as schema_mod
        from silk.sources import waveforms as wf
        from silk.sources.schema import source_schema
        from silk.codegen import generator as g

        class _Ramp:
            def __init__(self, slope=1):
                self.slope = slope

            def to_spice(self):
                return f'RAMP({self.slope})'

        # Register the class under its own __name__ (not an alias) so that
        # codegen's name -> kind inversion (which keys off the real class
        # name, e.g. type(obj).__name__) round-trips correctly.
        monkeypatch.setattr(wf, '_Ramp', _Ramp, raising=False)
        base = schema_mod._load_metadata()
        fake = dict(base)
        fake['waveforms'] = dict(base['waveforms'], ramp='_Ramp')
        monkeypatch.setattr(schema_mod, '_load_metadata', lambda: fake)

        # lru_cache gotcha (Task 5): codegen reads metadata through a cached
        # loader, so it won't see the patch above until the cache is
        # cleared — and must be cleared again at teardown so the fake
        # 'ramp' kind doesn't leak into later tests.
        g._source_meta.cache_clear()
        request.addfinalizer(g._source_meta.cache_clear)

        # 2. schema reflects it
        assert 'ramp' in source_schema('VoltageSource')['waveforms']
        # 4. codegen kind map resolves it (schema-driven, no hardcoded entry)
        assert g._waveform_kind('_Ramp') == 'ramp'
        # 1 + 3 (API + catalog): the API/factory surface is asserted by
        # TestExtensibility above; the catalog surface
        # (export_schema) is asserted here.
        import warnings
        from silk.elements.schema import export_schema
        from silk.elements import elements
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cat = export_schema(elements, include_metadata=False)
        assert 'ramp' in cat['vsource']['waveforms']


class TestNewSourceTypeAxis:
    def test_new_sourcebase_subclass_surfaces_in_source_schema(self):
        from silk.sources.classes import SourceBase
        from silk.sources.schema import export_source_schema
        # Hold the classes through export: __subclasses__ is weakly held and
        # a class from another test may otherwise disappear between snapshots.
        classes = SourceBase.__subclasses__()
        names = {c.__name__ for c in classes}
        sch = export_source_schema()
        # every SourceBase subclass is either schematized or listed under _meta
        surfaced = set(sch) | set(sch['_meta']['unschematized_sources'])
        assert names <= surfaced


class TestWaveformAutoExport:
    def test_all_metadata_waveforms_importable(self):
        import silk.sources as S
        from silk.sources.schema import _load_metadata
        for cls_name in _load_metadata()['waveforms'].values():
            assert hasattr(S, cls_name), cls_name
            assert cls_name in S.__all__

    def test_known_names_still_importable(self):
        from silk.sources import Pulse, PWL, Sine  # noqa: F401

    def test_export_set_is_metadata_derived(self):
        import silk.sources as S
        from silk.sources.schema import _load_metadata
        meta_names = set(_load_metadata()['waveforms'].values())
        # every metadata waveform class is auto-present in the package namespace
        # via the same mechanism (not a coincidence of the hand list)
        assert meta_names <= set(S.__all__)
        assert getattr(S, '_WAVEFORMS_AUTO_EXPORTED', False) is True
