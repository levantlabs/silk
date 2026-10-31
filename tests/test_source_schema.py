"""Tests for testbench-source schema export (silk.sources.schema)."""
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from silk import sources
from silk.sources.schema import (
    export_source_schema, export_source_schema_json, source_schema, _load_metadata,
)
from silk.sources.waveforms import RandomSource, RandomType

_REPO = Path(__file__).resolve().parents[1]
_CODEGEN_CONTRACT = _REPO / 'docs' / 'codegen-contract.md'


@pytest.fixture(scope='module')
def schema():
    return export_source_schema()


def test_exports_stimulus_sources(schema):
    assert 'VoltageSource' in schema
    assert 'CurrentSource' in schema


def test_meta_block(schema):
    m = schema['_meta']
    assert m['format_version'] == '1.0'
    assert m['source_type'] == 'sources'
    assert m['source_count'] >= 2


def test_exact_parameter_kinds(schema):
    """Name-keyed hints must not silently spread to other source parameters."""
    tagged = {}
    for source, spec in schema.items():
        if source == '_meta':
            continue
        groups = [(None, spec.get('parameters', []))]
        groups.extend((waveform, wave['parameters'])
                      for waveform, wave in spec.get('waveforms', {}).items())
        for waveform, parameters in groups:
            for param in parameters:
                if 'kind' in param:
                    tagged.setdefault(param['kind'], set()).add(
                        (source, waveform, param['name']))
    assert tagged == {
        'bits': {
            ('VoltageSource', 'bitpattern', 'bits'),
            ('CurrentSource', 'bitpattern', 'bits'),
        },
        'integer': {
            ('VoltageSource', 'prbs', 'order'),
            ('VoltageSource', 'prbs', 'length'),
            ('VoltageSource', 'prbs', 'seed'),
            ('VoltageSource', 'bitpattern', 'repeat'),
            ('CurrentSource', 'prbs', 'order'),
            ('CurrentSource', 'prbs', 'length'),
            ('CurrentSource', 'prbs', 'seed'),
            ('CurrentSource', 'bitpattern', 'repeat'),
        },
    }


class TestVoltageSource:
    def test_symbol_unit_category(self, schema):
        vs = schema['VoltageSource']
        assert vs['symbol'] == 'VSource'
        assert vs['unit'] == 'V'
        assert vs['category'] == 'source'

    def test_pins(self, schema):
        vs = schema['VoltageSource']
        assert [p['name'] for p in vs['pins']] == ['positive', 'negative']

    def test_ac_param_surfaced(self, schema):
        """The whole point of the request: `ac` must be visible."""
        vs = schema['VoltageSource']
        names = {p['name'] for p in vs['parameters']}
        assert 'ac' in names and 'dc' in names
        ac = next(p for p in vs['parameters'] if p['name'] == 'ac')
        assert ac['unit'] == 'V'
        assert ac['default'] == 0
        assert ac['required'] is False

    def test_waveform_kinds_present(self, schema):
        vs = schema['VoltageSource']
        for kind in ('pulse', 'pwl', 'sine', 'exponential', 'sffm', 'am',
                     'trnoise', 'trrandom'):
            assert kind in vs['waveforms'], kind

    def test_pulse_subparams_reflected_with_units(self, schema):
        pulse = schema['VoltageSource']['waveforms']['pulse']['parameters']
        by = {p['name']: p for p in pulse}
        assert by['v1']['unit'] == 'V'          # amplitude → source unit
        assert by['v1']['required'] is True     # no default
        assert by['td']['unit'] == 's'          # time
        assert by['td']['default'] == 0
        assert by['td']['required'] is False

    def test_pulse_edges_export_as_required(self, schema):
        """tr/tf are mandatory on Pulse (a zero edge is simulator-dependent),
        so the exported contract must tell consumers they have no default and
        must be supplied — not offer 0 as a fillable default."""
        pulse = schema['VoltageSource']['waveforms']['pulse']['parameters']
        by = {p['name']: p for p in pulse}
        for edge in ('tr', 'tf'):
            assert by[edge]['required'] is True, edge
            assert by[edge]['default'] is None, edge
            assert by[edge]['unit'] == 's', edge


class TestCurrentSource:
    def test_amplitudes_resolve_to_amps(self, schema):
        cs = schema['CurrentSource']
        assert cs['unit'] == 'A'
        dc = next(p for p in cs['parameters'] if p['name'] == 'dc')
        assert dc['unit'] == 'A'
        v1 = next(p for p in cs['waveforms']['pulse']['parameters']
                  if p['name'] == 'v1')
        assert v1['unit'] == 'A'          # sentinel 'source' → A
        freq = next(p for p in cs['waveforms']['sine']['parameters']
                    if p['name'] == 'freq')
        assert freq['unit'] == 'Hz'       # frequency stays Hz


class TestUnitCoverage:
    """Every amplitude/time/frequency waveform param must carry a unit; only
    genuinely dimensionless/structural params may be null (review finding)."""
    ALLOWED_NULL = {'bits', 'np', 'points', 'nalpha', 'type', 'param1', 'param2'}

    def test_no_known_unit_falls_through_to_null(self, schema):
        offenders = []
        for src in ('VoltageSource', 'CurrentSource'):
            for kind, spec in schema[src]['waveforms'].items():
                for p in spec['parameters']:
                    if p['unit'] is None and p['name'] not in self.ALLOWED_NULL:
                        offenders.append(f"{src}.{kind}.{p['name']}")
        assert not offenders, offenders


class TestReflectionAndDrift:
    def test_params_are_reflected_not_hardcoded(self, schema):
        from silk.sources import waveforms
        for kind, spec in schema['VoltageSource']['waveforms'].items():
            cls = getattr(waveforms, spec['class'])
            real = set(inspect.signature(cls.__init__).parameters) - {'self'}
            reflected = {p['name'] for p in spec['parameters']}
            assert reflected == real, (kind, reflected ^ real)

    def test_defaults_match_signature(self, schema):
        sig = inspect.signature(sources.VoltageSource.__init__)
        for c in schema['VoltageSource']['parameters']:
            assert c['default'] == sig.parameters[c['name']].default

    def test_param_classification_is_complete(self):
        """Drift guard: every schematized source class's __init__ param must
        be classified as pin / scalar_param / reference (TASK-253) /
        expression (TASK-282.5) / waveform-kwarg (or the known PWL
        convenience kwargs). A new, unclassified param fails here loudly
        rather than silently vanishing from the schema."""
        meta = _load_metadata()
        wf_kwargs = set(meta['waveforms'])
        known_extra = {'name', 'pwl_r', 'pwl_td', 'waveform'}   # generic slot (§5.3)
        for name, src_meta in meta['sources'].items():
            cls = getattr(sources, name)
            params = set(inspect.signature(cls.__init__).parameters) - {'self'}
            classified = (set(src_meta.get('pins', [])) | set(src_meta.get('scalar_params', []))
                          | set(src_meta.get('references', []))
                          | set(src_meta.get('expressions', []))
                          | wf_kwargs | known_extra)
            unclassified = params - classified
            assert not unclassified, (name, unclassified)

    def test_unschematized_sources_surfaced(self):
        """New/other source classes appear under _meta, never silently dropped."""
        # A fresh interpreter contains only shipped classes. Other tests may
        # legitimately leave user classes alive in this interpreter, and their
        # lifetime must not decide whether the shipped catalog is complete.
        out = subprocess.check_output([sys.executable, '-c', '''
import json
from silk.sources.classes import SourceBase
from silk.sources.schema import export_source_schema
before = export_source_schema()['_meta']['unschematized_sources']
class CustomSource(SourceBase):
    pass
after = export_source_schema()['_meta']['unschematized_sources']
print(json.dumps([before, after]))
'''], text=True)
        assert json.loads(out) == [[], ['CustomSource']]


class TestSerializationAndCLI:
    def test_strict_json_serializable(self, schema):
        # No default= fallback — must be strictly serializable.
        json.dumps(schema)
        text = export_source_schema_json()
        assert json.loads(text)['VoltageSource']['symbol'] == 'VSource'

    def test_cli_emits_valid_json(self):
        out = subprocess.check_output(
            [sys.executable, '-m', 'silk.sources.schema'], text=True)
        assert json.loads(out)['CurrentSource']['unit'] == 'A'


def test_single_source_schema_matches_bulk(schema):
    assert source_schema('VoltageSource') == schema['VoltageSource']


class TestControlledSources:
    """TASK-253: VCVS/VCCS/CCVS/CCCS schema entries."""

    _EXPECTED = {
        'VoltageControlledVoltageSource': {
            'symbol': 'VCVS', 'unit': 'V/V',
            'pins': ['out_pos', 'out_neg', 'ctrl_pos', 'ctrl_neg'],
            'scalar': 'gain',
        },
        'VoltageControlledCurrentSource': {
            'symbol': 'VCCS', 'unit': 'A/V',
            'pins': ['out_pos', 'out_neg', 'ctrl_pos', 'ctrl_neg'],
            'scalar': 'gm',
        },
        'CurrentControlledVoltageSource': {
            'symbol': 'CCVS', 'unit': 'V/A',
            'pins': ['out_pos', 'out_neg'],
            'scalar': 'transresistance',
        },
        'CurrentControlledCurrentSource': {
            'symbol': 'CCCS', 'unit': 'A/A',
            'pins': ['out_pos', 'out_neg'],
            'scalar': 'gain',
        },
    }

    def test_four_classes_schematized(self, schema):
        for name in self._EXPECTED:
            assert name in schema, name

    def test_symbol_unit_pins_scalar(self, schema):
        for name, exp in self._EXPECTED.items():
            entry = schema[name]
            assert entry['symbol'] == exp['symbol'], name
            assert entry['unit'] == exp['unit'], name
            assert entry['category'] == 'source', name
            assert [p['name'] for p in entry['pins']] == exp['pins'], name
            params = {p['name']: p for p in entry['parameters']}
            assert set(params) == {exp['scalar']}, name
            scalar = params[exp['scalar']]
            assert scalar['unit'] == exp['unit'], name
            assert scalar['required'] is True, name
            assert scalar['default'] is None, name

    def test_no_waveforms(self, schema):
        """No dc/ac/waveform kinds on these — structural sources, not stimuli."""
        for name in self._EXPECTED:
            assert schema[name]['waveforms'] == {}, name

    def test_ccvs_cccs_carry_ctrl_vname_reference(self, schema):
        for name in ('CurrentControlledVoltageSource', 'CurrentControlledCurrentSource'):
            refs = schema[name]['references']
            assert refs == [{'name': 'ctrl_vname', 'kind': 'source',
                              'required': True, 'default': None}], name

    def test_vcvs_vccs_carry_no_references_key(self, schema):
        """references is present only when the class declares one (mirrors
        the 'choices' absent-not-null convention, TASK-252)."""
        for name in ('VoltageControlledVoltageSource', 'VoltageControlledCurrentSource'):
            assert 'references' not in schema[name], name

    def test_gain_resolves_per_class_unit(self, schema):
        """The same param name ('gain') resolves to a different unit on
        VCVS (V/V) vs CCCS (A/A) via the 'source' sentinel + each class's
        own base unit — not a global gain->unit mapping."""
        vcvs_gain = next(p for p in schema['VoltageControlledVoltageSource']['parameters']
                          if p['name'] == 'gain')
        cccs_gain = next(p for p in schema['CurrentControlledCurrentSource']['parameters']
                          if p['name'] == 'gain')
        assert vcvs_gain['unit'] == 'V/V'
        assert cccs_gain['unit'] == 'A/A'

    def test_existing_vsource_isource_schema_unchanged(self, schema):
        """AC #3: adding the four controlled-source entries does not add a
        'references' key (or anything else) to VoltageSource/CurrentSource."""
        for name in ('VoltageSource', 'CurrentSource'):
            assert 'references' not in schema[name], name


_BASELINE_SIX_PATH = _REPO / 'tests' / 'fixtures' / 'source_schema_baseline_six.json'


def test_six_pre_existing_entries_byte_identical(schema):
    """TASK-282.5 AC #3: VoltageSource/CurrentSource and the four controlled
    sources must not change at all when the two behavioral entries are
    added. ``tests/fixtures/source_schema_baseline_six.json`` was captured
    from ``export_source_schema()`` before this task edited
    source_metadata.yaml/schema.py (worktree HEAD == main at capture time)."""
    with open(_BASELINE_SIX_PATH) as fh:
        baseline = json.load(fh)
    # TASK-535 deliberately defaults only sine voffset; other historical fields stay frozen.
    for name in ('VoltageSource', 'CurrentSource'):
        param = next(p for p in baseline[name]['waveforms']['sine']['parameters']
                     if p['name'] == 'voffset')
        param.update(default=0, required=False)
    import copy
    six = {name: copy.deepcopy(schema[name]) for name in baseline}
    for name in ('VoltageSource', 'CurrentSource'):
        for kind in ('bitpattern', 'prbs'):
            six[name]['waveforms'].pop(kind)
    assert json.dumps(six, sort_keys=True) == json.dumps(baseline, sort_keys=True)


class TestBehavioralSources:
    """TASK-282.5: BehavioralVoltage/BehavioralCurrent schema entries."""

    _EXPECTED = {
        'BehavioralVoltage': {'symbol': 'BV', 'unit': 'V'},
        'BehavioralCurrent': {'symbol': 'BI', 'unit': 'A'},
    }

    def test_classes_schematized(self, schema):
        for name in self._EXPECTED:
            assert name in schema, name

    def test_symbol_unit_pins(self, schema):
        for name, exp in self._EXPECTED.items():
            entry = schema[name]
            assert entry['symbol'] == exp['symbol'], name
            assert entry['unit'] == exp['unit'], name
            assert entry['category'] == 'source', name
            assert [p['name'] for p in entry['pins']] == ['positive', 'negative'], name

    def test_no_waveforms_no_parameters_no_references(self, schema):
        """Structural, like the controlled sources: no dc/ac/waveform kinds,
        no scalar params, and `expression` is not a numeric parameter or a
        source-instance reference — it must not leak into either key."""
        for name in self._EXPECTED:
            entry = schema[name]
            assert entry['waveforms'] == {}, name
            assert entry['parameters'] == [], name
            assert 'references' not in entry, name

    def test_expression_is_its_own_field_shape(self, schema):
        """AC #2: a consumer can tell `expression` is a free-form SPICE
        expression string, not a number or a source-instance reference —
        it lives under its own `expressions` key with `kind: 'expression'`,
        never under `parameters` (no unit to give it) or `references`
        (`kind: 'source'`, resolved against other sources[] entries)."""
        for name in self._EXPECTED:
            exprs = schema[name]['expressions']
            assert exprs == [{'name': 'expression', 'kind': 'expression',
                               'required': True, 'default': None}], name

    def test_expressions_key_absent_elsewhere(self, schema):
        """Absent-not-null, mirroring 'references'/'choices' (TASK-252/253)."""
        for name, entry in schema.items():
            if name in self._EXPECTED or name == '_meta':
                continue
            assert 'expressions' not in entry, name

    def test_code_template_matches_generator_emission(self, schema):
        """AC #1: filling code_template from a sample entry produces the
        exact Python the generator emits for that sources[] entry."""
        from silk.codegen.generator import Generator
        cases = {
            'BehavioralVoltage': dict(
                instance='bv1', positive='vout', negative='gnd',
                expression='v(vin) * 2'),
            'BehavioralCurrent': dict(
                instance='bi1', positive='io', negative='gnd',
                expression='0.01*v(vin,gnd)'),
        }
        for type_name, case in cases.items():
            entry = schema[type_name]
            filled = entry['code_template'].format(
                instance_name=case['instance'],
                net_positive=repr(case['positive']),
                net_negative=repr(case['negative']),
                expr_expression=repr(case['expression']),
            )
            code = Generator().generate({
                "circuit_name": "tb", "pdk": "sky130",
                "sources": [{"type": type_name, "instance": case['instance'],
                             "positive": case['positive'],
                             "negative": case['negative'],
                             "expression": case['expression']}],
            })
            assert f"ckt.add({filled})" in code, (type_name, filled)


class TestRandomTypeChoices:
    """TASK-252: ``RandomSource.type`` is annotated with the ``RandomType``
    IntEnum, and the schema reflects it into a ``choices`` field — the one
    place a parameter editor needs to render a dropdown instead of
    asking for a bare integer code."""

    _EXPECTED = [
        {'value': 1, 'label': 'uniform'},
        {'value': 2, 'label': 'gaussian'},
        {'value': 3, 'label': 'exponential'},
        {'value': 4, 'label': 'poisson'},
    ]

    def _type_param(self, schema, src='VoltageSource'):
        params = schema[src]['waveforms']['trrandom']['parameters']
        return next(p for p in params if p['name'] == 'type')

    def test_trrandom_type_carries_the_four_choices(self, schema):
        for src in ('VoltageSource', 'CurrentSource'):
            p = self._type_param(schema, src)
            assert p['choices'] == self._EXPECTED, src
            # unit/default/required are unchanged by adding choices
            assert p['unit'] is None
            assert p['default'] is None
            assert p['required'] is True

    def test_choices_equal_the_enum(self, schema):
        p = self._type_param(schema)
        assert p['choices'] == [{'value': int(m.value), 'label': m.name}
                                 for m in RandomType]

    def test_no_other_parameter_carries_choices(self, schema):
        """AC #1: the ``choices`` key appears on RandomSource.type and
        nowhere else — not as an always-present/null field (that convention
        belongs to the metric schema, TASK-194), absent entirely."""
        offenders = []
        for src_name in ('VoltageSource', 'CurrentSource'):
            src = schema[src_name]
            for p in src['parameters']:
                if 'choices' in p:
                    offenders.append(f"{src_name}.{p['name']}")
            for wf_name, wf in src['waveforms'].items():
                for p in wf['parameters']:
                    if 'choices' in p and not (wf_name == 'trrandom' and p['name'] == 'type'):
                        offenders.append(f"{src_name}.{wf_name}.{p['name']}")
        assert not offenders, offenders

    def test_docstring_references_the_enum(self):
        """RandomSource's docstring does not hand-author the code/label
        mapping (a second copy that could drift) — it points at
        ``RandomType``, which is the one place the mapping actually lives
        (the enum's own members)."""
        assert 'RandomType' in RandomSource.__doc__

    def test_docs_and_schema_agree_with_the_enum(self, schema):
        """AC #2: the schema's choices and docs/codegen-contract.md's
        RandomSource subsection both name every RandomType member, so if the
        enum ever changes, both of these must be updated to keep passing —
        they cannot silently drift apart from the enum or each other."""
        contract_text = _CODEGEN_CONTRACT.read_text()
        schema_choices = self._type_param(schema)['choices']
        for member in RandomType:
            assert member.name in contract_text, member.name
            assert {'value': int(member.value), 'label': member.name} in schema_choices

    def test_random_source_accepts_int_or_enum_member(self):
        by_int = RandomSource(type=2, ts=1e-9)
        by_enum = RandomSource(type=RandomType.gaussian, ts=1e-9)
        assert by_int.to_spice() == by_enum.to_spice() == 'TRRANDOM(2 1e-09 0 1 0)'
        # stored/emitted as the bare int either way, never the enum repr
        assert by_enum.type == 2 and type(by_enum.type) is int

    def test_random_source_rejects_out_of_range_type(self):
        with pytest.raises(ValueError, match='uniform.*gaussian.*exponential.*poisson'):
            RandomSource(type=7, ts=1e-9)


@pytest.mark.parametrize('source_class,unit', [('VoltageSource', 'V'), ('CurrentSource', 'A')])
@pytest.mark.parametrize('kind,structural', [
    ('bitpattern', {'bits': '10'}), ('prbs', {'order': 7, 'length': 10}),
])
def test_finite_bit_schema_composes_source(source_class, unit, kind, structural):
    entry = source_schema(source_class)['waveforms'][kind]
    assert entry['kwarg'] == 'waveform'
    params = {p['name']: p for p in entry['parameters']}
    for name in ('v1', 'v2'):
        assert params[name]['unit'] == unit and params[name]['required']
    for name in ('ui', 'tr', 'tf'):
        assert params[name]['unit'] == 's' and params[name]['required']
    assert params['td']['unit'] == 's' and params['td']['default'] == 0
    if kind == 'bitpattern':
        assert params['bits']['kind'] == 'bits'
        assert params['repeat']['default'] == 1
    else:
        assert params['length']['required'] and params['order']['required']
        assert params['seed']['default'] == 0
    wave = getattr(sources, entry['class'])(**structural, v1=0, v2=1,
                                           ui=1, tr=.1, tf=.2)
    src = getattr(sources, source_class)('data', 'out', '0', **{entry['kwarg']: wave})
    assert 'PWL(' in src.to_spice()
    assert src.to_numpy([0])[0] == wave.to_numpy([0])[0]

@pytest.mark.parametrize('class_name, catalog_name', [
    ('VoltageSource', 'vsource'), ('CurrentSource', 'isource'),
])
def test_generic_schema_composition_and_authored_kinds(tmp_path, monkeypatch,
                                                       class_name, catalog_name):
    """New kinds must compose via the real slot and carry authored input kinds."""
    import yaml
    from silk import export_schema
    from silk.elements import elements
    from silk.sources import schema as schema_mod, waveforms

    class TestPattern(waveforms._Waveform):
        def __init__(self, bits, level=1.25):
            self.bits, self.level = bits, level

        def to_spice(self):
            return f'PWL(0 0 1 {self.level})'

    baseline = export_source_schema()[class_name]
    meta = _load_metadata()
    meta['waveforms']['testpattern'] = 'TestPattern'
    meta.setdefault('param_kinds', {})['bits'] = 'bits'
    path = tmp_path / 'metadata.yaml'
    path.write_text(yaml.safe_dump(meta))
    monkeypatch.setattr(schema_mod, '_META_PATH', str(path))
    monkeypatch.setattr(waveforms, 'TestPattern', TestPattern, raising=False)
    monkeypatch.setattr(sources, 'TestPattern', TestPattern, raising=False)
    spec = export_source_schema()[class_name]
    wave = spec['waveforms']['testpattern']
    assert wave['kwarg'] == 'waveform'
    assert wave['parameters'] == [
        {'name': 'bits', 'unit': None, 'default': None, 'required': True, 'kind': 'bits'},
        {'name': 'level', 'unit': None, 'default': 1.25, 'required': False},
    ]
    expression = wave['code_template'].format(param_bits=repr('101'), param_level=1.25)
    emitter = eval(expression, {'sources': sources})
    src = getattr(sources, class_name)('stim', positive='a', negative='0',
                                       **{wave['kwarg']: emitter})
    assert 'PWL(0 0 1 1.25)' in src.to_spice()
    catalog = export_schema(elements, include_metadata=False)[catalog_name]
    assert catalog['waveforms']['testpattern']['parameters'] == wave['parameters']
    assert catalog['waveforms']['testpattern']['kwarg'] == 'waveform'
    del spec['waveforms']['testpattern']
    assert spec == baseline
