"""Tests for export_schema() Module support and define_module_nets direction."""
import json
import os
import subprocess
import sys

import pytest
import silk.pdk
from silk.circuit.modules import Module, ModulePin
from silk.elements.schema import module_schema, export_schema

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDK_YAML = os.path.join(REPO_ROOT, 'silk/pdk/data/skywater_130_device_details.yaml')


# ---- Fixtures ----

@pytest.fixture
def pdk(tmp_path):
    pdk_yaml = os.path.join(os.path.dirname(silk.pdk.__file__), 'data', 'skywater_130_device_details.yaml')
    try:
        from silk.pdk.basepdk import BasePDK
        return BasePDK(pdk_yaml, library_location='/dev/null')
    except Exception:
        pytest.skip('PDK not available')

@pytest.fixture
def simple_module(pdk, tmp_path):
    from silk.circuit.circuit_elements import PortDir
    mod = Module('PDK_skywater130', str(tmp_path), 'ldo.cir', pdk_details=pdk)
    mod.define_module_name('ldo')
    mod.define_module_ports({
        'vin':  PortDir.IN,
        'vout': PortDir.OUT,
        'vdd':  PortDir.INOUT,
        'gnd':  PortDir.GROUND,
    })
    return mod


# ---- Gap 3: define_module_ports stores direction ----

class TestModulePinDirection:
    def test_direction_stored(self, simple_module):
        from silk.circuit.circuit_elements import PortDir
        assert simple_module.pin_list['vin'].direction == PortDir.IN
        assert simple_module.pin_list['vout'].direction == PortDir.OUT
        assert simple_module.pin_list['vdd'].direction == PortDir.INOUT
        assert simple_module.pin_list['gnd'].direction == PortDir.GROUND

    def test_none_value_compat(self, pdk, tmp_path):
        """define_module_nets with None values defaults to INOUT."""
        from silk.circuit.circuit_elements import PortDir
        mod = Module('PDK_skywater130', str(tmp_path), 'x.cir', pdk_details=pdk)
        mod.define_module_name('buf')
        mod.define_module_nets({'a': None, 'z': None})
        assert mod.pin_list['a'].direction == PortDir.INOUT

    def test_modulepin_has_direction_attribute(self):
        pin = ModulePin(name='x')
        assert hasattr(pin, 'direction')
        assert pin.direction is None


# ---- Gap 4: module_schema() ----

class TestModuleSchema:
    def test_schema_name(self, simple_module):
        s = module_schema(simple_module)
        assert s['name'] == 'ldo'

    def test_schema_type_is_module(self, simple_module):
        s = module_schema(simple_module)
        assert s['type'] == 'module'

    def test_schema_not_pdk_device(self, simple_module):
        s = module_schema(simple_module)
        assert s['is_pdk_device'] is False

    def test_schema_category(self, simple_module):
        s = module_schema(simple_module)
        assert s['category'] == 'module'

    def test_schema_pins_count(self, simple_module):
        s = module_schema(simple_module)
        assert len(s['pins']) == 4

    def test_schema_pin_names(self, simple_module):
        s = module_schema(simple_module)
        names = [p['name'] for p in s['pins']]
        assert 'vin' in names
        assert 'vout' in names

    def test_schema_pin_directions(self, simple_module):
        from silk.circuit.circuit_elements import PortDir
        s = module_schema(simple_module)
        by_name = {p['name']: p['direction'] for p in s['pins']}
        assert by_name['vin'] == PortDir.IN
        assert by_name['vout'] == PortDir.OUT
        assert by_name['vdd'] == PortDir.INOUT
        assert by_name['gnd'] == PortDir.GROUND

    def test_schema_devices_empty(self, simple_module):
        s = module_schema(simple_module)
        assert s['devices'] == []

    def test_schema_has_code_template(self, simple_module):
        s = module_schema(simple_module)
        assert 'code_template' in s
        assert 'ldo' in s['code_template']
        assert "module_name='ldo'" in s['code_template']

    def test_code_template_uses_current_api_and_is_valid_python(self, simple_module):
        """The emitted snippet must use define_module_ports (not the deprecated
        define_module_nets), reference typed PortDir values, and parse as valid
        Python — it was previously emitting `<PortDir.IN: 'input'>` repr garbage
        through the deprecated API (poisoned the codegen contract)."""
        import ast
        ct = module_schema(simple_module)['code_template']
        assert 'define_module_nets' not in ct          # deprecated API gone
        assert 'define_module_ports' in ct
        assert 'PortDir.' in ct and '<PortDir' not in ct   # typed, not repr garbage
        ast.parse(ct)                                   # raises if not valid Python

    def test_schema_json_serialisable(self, simple_module):
        s = module_schema(simple_module)
        json_str = json.dumps(s)
        assert json_str  # no exception


# ---- export_schema() accepts Module ----

class TestExportSchemaModule:
    def test_export_schema_with_module(self, simple_module):
        result = export_schema(simple_module)
        assert 'ldo' in result

    def test_export_schema_module_entry(self, simple_module):
        result = export_schema(simple_module)
        assert result['ldo']['type'] == 'module'

    def test_export_schema_module_meta(self, simple_module):
        result = export_schema(simple_module)
        assert '_meta' in result
        assert result['_meta']['source_type'] == 'Module'

    def test_export_schema_module_no_meta(self, simple_module):
        result = export_schema(simple_module, include_metadata=False)
        assert '_meta' not in result


# ---- Module.export_schema() instance method ----

class TestModuleExportSchemaMethod:
    def test_instance_method_exists(self, simple_module):
        assert hasattr(simple_module, 'export_schema')
        assert callable(simple_module.export_schema)

    def test_instance_method_returns_schema(self, simple_module):
        s = simple_module.export_schema()
        assert s['name'] == 'ldo'
        assert s['type'] == 'module'

    def test_instance_method_json_serialisable(self, simple_module):
        s = simple_module.export_schema()
        json.dumps(s)  # no exception


# ---- silk-level exports ----

class TestSilkExports:
    def test_export_schema_importable_from_silk(self):
        from silk import export_schema
        assert callable(export_schema)

    def test_module_schema_importable_from_silk(self):
        from silk import module_schema
        assert callable(module_schema)


# ---- Source catalog enrichment (§B) ----

class TestSourceCatalogEnrichment:
    @pytest.fixture(scope='class')
    def sch(self):
        import warnings
        from silk.elements.schema import export_schema
        from silk.elements import elements
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return export_schema(elements, include_metadata=False)

    def test_vsource_has_ac_param(self, sch):
        names = {p['name'] for p in sch['vsource']['parameters']}
        assert names == {'dc', 'ac'}

    def test_param_shape_is_device_shape(self, sch):
        dc = next(p for p in sch['vsource']['parameters'] if p['name'] == 'dc')
        assert set(dc) == {'name', 'default', 'unit', 'min', 'max'}
        assert dc['unit'] == 'V' and dc['default'] == 0

    def test_waveforms_block_present(self, sch):
        wf = sch['vsource']['waveforms']
        for kind in ('pulse', 'pwl', 'sine'):
            assert kind in wf
        assert sch['vsource']['waveform_exclusive'] is True

    def test_source_class_mapping(self, sch):
        assert sch['vsource']['source_class'] == 'VoltageSource'
        assert sch['isource']['source_class'] == 'CurrentSource'

    def test_scalar_code_template_is_declarative(self, sch):
        ct = sch['vsource']['code_template']
        assert 'sources.vdc(' in ct
        assert 'elements.vsource' not in ct

    def test_scalar_code_template_uses_value_kwarg(self, sch):
        """sources.vdc()/idc() take the primary scalar (dc) as `value=`
        positionally-or-keyword and reject a stray `dc=` kwarg (it's already
        consumed as the ctor's implicit slot) — see silk/sources/library.py
        _make_scalar_ctor. The scalar code_template must reflect that, or a
        caller filling it emits a line that raises TypeError."""
        ct = sch['vsource']['code_template']
        assert 'value={param_dc}' in ct
        assert 'dc={param_dc}' not in ct

    def test_scalar_code_template_executes(self, sch):
        """Exec-based guard (mirrors tests/test_codegen_sources.py's
        TestSourceCtorCallsActuallyExecute): fill the vsource/isource scalar
        code_templates with real values and confirm they actually construct a
        working source, not just look correct as a string."""
        from silk import sources

        ct = sch['vsource']['code_template']
        filled = ct.format(instance_name='v1', net_p=repr('a'), net_n=repr('b'),
                            param_dc=1.8, param_ac=0)
        ns = {'sources': sources}
        exec(filled, ns)
        assert "Vv1" in ns['v1'].to_spice()

        ct_i = sch['isource']['code_template']
        filled_i = ct_i.format(instance_name='i1', net_p=repr('a'), net_n=repr('b'),
                                param_dc=0.001, param_ac=0)
        ns2 = {'sources': sources}
        exec(filled_i, ns2)
        assert "Ii1" in ns2['i1'].to_spice()

    def test_per_kind_code_template_is_declarative(self, sch):
        pulse = sch['vsource']['waveforms']['pulse']['code_template']
        assert 'sources.vpulse(' in pulse
        assert 'v1={param_v1}' in pulse
        assert 'dc={param_dc}' in pulse
        assert 'ac={param_ac}' in pulse

    def test_pulse_edges_are_required_in_the_catalog(self, sch):
        """Pulse rejects a zero tr/tf, so the catalog a consumer builds from
        must not advertise 0 as a fillable default for them."""
        for src in ('vsource', 'isource'):
            by = {p['name']: p
                  for p in sch[src]['waveforms']['pulse']['parameters']}
            for edge in ('tr', 'tf'):
                assert by[edge]['required'] is True, (src, edge)
                assert by[edge]['default'] is None, (src, edge)

    def test_passive_entry_has_no_symbol(self, sch):
        assert 'symbol' not in sch['resistor']

    def test_current_alias_is_i(self, sch):
        assert 'sources.idc(' in sch['isource']['code_template']
        assert 'sources.ipulse(' in sch['isource']['waveforms']['pulse']['code_template']

    def test_passives_unchanged(self, sch):
        assert 'waveforms' not in sch['resistor']
        assert sch['resistor']['source_class'] is None

    def test_source_entry_carries_symbol(self, sch):
        assert sch['vsource']['symbol'] == 'VSource'
        assert sch['isource']['symbol'] == 'ISource'


# ---- PDK library metadata (_library block, Task 7) ----

class TestExportSchemaLibraryBlock:
    def test_export_schema_library_block_and_device_count(self):
        from silk.elements.schema import export_schema
        from silk.pdk.basepdk import BasePDK
        pdk = BasePDK('silk/pdk/data/skywater_130_device_details.yaml',
                              library_location='/dev/null',
                              corners={'tt': {'kind': 'process', 'desc': 'typical'}},
                              default_corner='tt', pdk_name='sky130')
        sch = export_schema(pdk)
        assert sch['_library']['pdk_name'] == 'sky130'
        assert sch['_library']['default_corner'] == 'tt'
        assert sch['_library']['corners']['tt']['kind'] == 'process'
        # device_count counts devices only, not _library/_meta
        assert sch['_meta']['device_count'] == sum(
            1 for k in sch if not k.startswith('_'))

    def test_export_schema_legacy_pdk_has_no_library_block(self):
        from silk.elements.schema import export_schema
        from silk.pdk.basepdk import BasePDK
        pdk = BasePDK('silk/pdk/data/skywater_130_device_details.yaml',
                              library_location='/dev/null')
        sch = export_schema(pdk)
        assert '_library' not in sch
        assert sch['_meta']['device_count'] == sum(
            1 for k in sch if not k.startswith('_'))


class TestSchemaCLIStdoutIsCleanJSON:
    """Regression: `python -m silk.elements.schema --pdk <yaml>` must emit
    JSON-only on stdout (diagnostic banners belong on stderr) so the output
    can be piped straight into a JSON parser."""

    def _run(self):
        env = dict(os.environ)
        env.setdefault('PDK_ROOT', os.path.expanduser('~/pdk'))
        return subprocess.run(
            [sys.executable, '-m', 'silk.elements.schema', '--pdk', PDK_YAML],
            capture_output=True, text=True, env=env, cwd=REPO_ROOT,
        )

    def test_stdout_parses_as_json(self):
        result = self._run()
        assert result.returncode == 0, result.stderr
        # The whole of stdout must be valid JSON — no leading banner line.
        schema = json.loads(result.stdout)
        assert isinstance(schema, dict)

    def test_unit_banner_goes_to_stderr_not_stdout(self):
        result = self._run()
        assert 'User units' not in result.stdout
        assert 'User units' in result.stderr


# ---------------------------------------------------------------------------
# TASK-330.4 ruling (F): 'polarity' (that task), 'op_params'/
# 'op_internal_name' (TASK-330.3), and 'op_backends' (TASK-330.5) are
# per-device PDK catalog declarations not exported as device
# parameters -- device_schema() builds its result dict from a fixed,
# explicit attribute allowlist (name/is_pdk_device/pdk_model/spice_prefix/
# category/pins/parameters/code_template/source_class), so a new private
# class attribute cannot leak into it by construction. These tests pin
# that down as a real regression check, for both include_metadata
# settings, on every device these four keys were actually added to.
# ---------------------------------------------------------------------------
_LEAKED_KEYS = ('polarity', 'op_params', 'op_internal_name', 'op_backends')


def _gf180_pdk():
    from silk.pdk.basepdk import BasePDK
    yaml_path = os.path.join(REPO_ROOT, 'silk/pdk/data/gf180mcu_device_details.yaml')
    return BasePDK(yaml_path, library_location='/dev/null', pdk_name='gf180mcu')


class TestOpTableDeclarationsExcludedFromExportSchema:
    @pytest.mark.parametrize('include_metadata', [False, True])
    @pytest.mark.parametrize('device_name', ['nmos1p8', 'pmos1p8'])
    def test_sky130_declared_devices_unaffected(self, pdk, include_metadata, device_name):
        schema = export_schema(pdk, include_metadata=include_metadata)
        entry = schema[device_name]
        for key in _LEAKED_KEYS:
            assert key not in entry, (
                f"{device_name}: {key!r} leaked into export_schema output")
        # The device DOES carry the declaration internally (sanity: this
        # isn't passing because the YAML change silently failed to apply).
        cls = getattr(pdk, device_name)
        assert getattr(cls, '_polarity', None) in ('n', 'p')
        assert getattr(cls, '_op_params', None)
        assert getattr(cls, '_op_backends', None)

    @pytest.mark.parametrize('include_metadata', [False, True])
    @pytest.mark.parametrize('device_name', ['nmos3p3', 'pmos3p3'])
    def test_gf180mcu_declared_devices_unaffected(self, include_metadata, device_name):
        pdk = _gf180_pdk()
        schema = export_schema(pdk, include_metadata=include_metadata)
        entry = schema[device_name]
        for key in _LEAKED_KEYS:
            assert key not in entry, (
                f"{device_name}: {key!r} leaked into export_schema output")
        cls = getattr(pdk, device_name)
        assert getattr(cls, '_polarity', None) in ('n', 'p')
        assert getattr(cls, '_op_params', None)
        assert getattr(cls, '_op_backends', None)

    def test_sky130_schema_keys_match_fixed_allowlist(self, pdk):
        """device_schema()'s own documented key set (schema.py's docstring)
        -- exhaustive, not just 'the three don't appear': proves nothing
        ELSE unexpected leaked in either, now or from a future declaration."""
        schema = export_schema(pdk, include_metadata=False)
        expected_keys = {
            'name', 'is_pdk_device', 'pdk_model', 'spice_prefix', 'category',
            'pins', 'parameters', 'code_template', 'source_class',
        }
        assert set(schema['nmos1p8']) == expected_keys | {'derivedParams'}
        assert set(schema['pmos1p8']) == expected_keys | {'derivedParams'}
