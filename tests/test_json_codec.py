"""Strict JSON is a package-wide writing contract (TASK-274)."""

import ast
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess

import pytest

from silk._json_codec import decode_json, encode_json

ROOT = Path(__file__).resolve().parents[1]


def _reject_constant(token):
    raise AssertionError(f'Non-standard JSON constant: {token}')


def _round_trip(value):
    return decode_json(json.loads(
        json.dumps(encode_json(value), allow_nan=False),
        parse_constant=_reject_constant,
    ))


@pytest.mark.parametrize('value, expected', [
    (math.nan, {'value': None, 'status': 'unavailable'}),
    (math.inf, {'value': None, 'status': 'infinite', 'sign': '+'}),
    (-math.inf, {'value': None, 'status': 'infinite', 'sign': '-'}),
])
def test_nonfinite_wire_and_numeric_round_trip(value, expected):
    assert encode_json(value) == expected
    restored = _round_trip(value)
    assert isinstance(restored, float)
    assert math.isnan(restored) if math.isnan(value) else restored == value


def test_nested_values_keep_zero_error_none_and_input_unchanged():
    original = {'metrics': (math.nan, 0.0, -0.0, 'Error: missing signal', None),
                'limits': [math.inf, -math.inf], 'finite': 1.25, 'boolean': False}
    restored = _round_trip(original)
    assert math.isnan(restored['metrics'][0])
    assert restored['metrics'][1:] == [0.0, -0.0, 'Error: missing signal', None]
    assert math.copysign(1, restored['metrics'][2]) == -1
    assert restored['limits'] == [math.inf, -math.inf]
    assert restored['finite'] == 1.25
    assert restored['boolean'] is False
    assert isinstance(original['metrics'], tuple)
    assert math.isnan(original['metrics'][0])


@pytest.mark.parametrize('literal', [
    {'value': None, 'status': 'unavailable'},
    {'value': None, 'status': 'infinite', 'sign': '+'},
    {'value': None, 'status': 'infinite', 'sign': '-'},
    {'value': {'value': None, 'status': 'unavailable'}, 'status': 'literal'},
    {'value': None, 'status': 'unavailable', 'note': 'user data'},
    {'value': None, 'status': 'infinite', 'sign': 'invalid'},
    {'value': 0.0, 'status': 'unavailable'},
    {'value': None, 'status': 'literal'},
])
def test_literal_tag_like_mappings_round_trip(literal):
    assert _round_trip(literal) == literal


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_finite_only_validation_and_nonfinite_keys_stay_errors(value):
    with pytest.raises(ValueError, match='JSON compliant'):
        encode_json({'input': [value]}, finite_only=True)
    with pytest.raises(ValueError, match='mapping keys'):
        encode_json({value: 'a key cannot contain an object tag'})


def test_cycles_fail_but_shared_containers_are_valid():
    value = []
    value.append(value)
    with pytest.raises(ValueError, match='Circular reference'):
        encode_json(value)
    shared = [1.0]
    assert _round_trip([shared, shared]) == [[1.0], [1.0]]


def _dump_sites():
    # Grep the whole package on every collection, including aliases and direct
    # imports. The AST distinguishes executable calls from comments/docstrings.
    # There is deliberately no hand-maintained allowlist of files or sites.
    found = subprocess.run(
        ['grep', '-rlE', r'\bjson\b|\b(dump|dumps)[[:space:]]*\(',
         '--include=*.py', 'silk'],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    sites = []
    for name in sorted(found.stdout.splitlines()):
        tree = ast.parse((ROOT / name).read_text())
        modules = set()
        functions = set()
        codec_imported = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(a.asname or a.name for a in node.names if a.name == 'json')
            if isinstance(node, ast.ImportFrom):
                if node.module == 'json':
                    functions.update(a.asname or a.name for a in node.names
                                     if a.name in ('dump', 'dumps'))
                if node.module in ('silk._json_codec', '_json_codec'):
                    codec_imported |= any(a.name == 'encode_json' for a in node.names)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if ((isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                 and f.value.id in modules and f.attr in ('dump', 'dumps'))
                    or (isinstance(f, ast.Name) and f.id in functions)):
                sites.append(pytest.param(node, codec_imported, id=f'{name}:{node.lineno}'))
    assert sites, 'The JSON writer inventory must not silently be empty'
    return sites


@pytest.mark.parametrize('site, codec_imported', _dump_sites())
def test_every_json_dump_site_uses_shared_encoder_and_strict_guard(site, codec_imported):
    assert codec_imported
    options = {kw.arg: kw.value for kw in site.keywords}
    assert isinstance(options.get('allow_nan'), ast.Constant)
    assert options['allow_nan'].value is False
    assert None not in options, 'Do not hide JSON guard overrides in **kwargs'
    encoded = site.args[0]
    assert isinstance(encoded, ast.Call)
    assert isinstance(encoded.func, ast.Name) and encoded.func.id == 'encode_json'
    # Exercise the encoder policy declared at each site on non-finite values.
    policy = {kw.arg: ast.literal_eval(kw.value) for kw in encoded.keywords}
    payload = {'result': [math.nan, math.inf, -math.inf, 0.0, 'Error: missing']}
    if policy.get('finite_only'):
        with pytest.raises(ValueError):
            encode_json(payload, **policy)
    else:
        output = io.StringIO()
        json.dump(encode_json(payload, **policy), output, allow_nan=False)
        json.loads(output.getvalue(), parse_constant=_reject_constant)
        assert not re.search(r'(?<![\w"])(?:NaN|-?Infinity)(?![\w"])', output.getvalue())


def test_metric_schema_canonical_bytes_unchanged():
    from silk.signals.schema import export_metric_schema, export_metric_schema_json

    schema = export_metric_schema()
    canonical = json.dumps(schema, sort_keys=True, separators=(',', ':'), allow_nan=False)
    # TASK-223 deliberately added propagation_delay.out_edge; digest bumped.
    # TASK-288 deliberately corrected integrated_noise's unit from 'V' to
    # '[y]*sqrt([x])'; digest bumped again.
    # TASK-326 deliberately added bandwidth_3dB's reference={'dc','peak'}
    # param (default 'dc', BREAKING); digest bumped again.
    # TASK-364 replaced x_when with crossing (threshold, start/end); bumped again.
    assert hashlib.sha256(canonical.encode()).hexdigest() == (
        # TASK-226.22: required scalar eye threshold; centering default True.
        'f47ca45cb4a0666deb6a955dfe495e17888bfa63308be96cb15e27966e65c040'  # TASK-417 period/c2c jitter params + _pp; TASK-418 long_term_jitter units/reference/phase/threshold/edge/start/end; TASK-412 removed input_referred_offset, kickback, hold_droop; TASK-379 variance; TASK-380 argmin_x/argmax_x refine; TASK-386 relative_error_norm; TASK-378 interpolate; TASK-399 static_errors wrappers; TASK-385 error_norm mean_abs + start/end; TASK-377 minimum/maximum + start/end; TASK-233 eight wrappers; TASK-323.3 final_value; TASK-364 crossing; TASK-369 evm_percent; TASK-365 rise/fall/slew edge params; TASK-363 propagation_delay; TASK-366 period/frequency; TASK-226.20 crossing_jitter_rms; TASK-367 average_duty, pulse_width, on/off_time; TASK-438 hysteresis n_rising/n_falling/start/end
    )
    assert export_metric_schema_json() == json.dumps(schema, indent=2, allow_nan=False)


@pytest.mark.parametrize('literal', [
    {'value': None, 'status': 'unavailable'},
    {'value': None, 'status': 'infinite', 'sign': '+'},
    {'value': None, 'status': 'infinite', 'sign': '-'},
    {'value': {'value': None, 'status': 'unavailable'}, 'status': 'literal'},
])
def test_finite_only_view_payload_preserves_literal_mapping_bytes(literal):
    from silk.signals.view.serialize import to_json

    payload = {'commands': [{'cmd': 'custom', 'args': literal}]}
    expected = json.dumps(payload, allow_nan=False)
    assert to_json(payload) == expected
    assert json.dumps(encode_json(payload, finite_only=True), allow_nan=False) == expected


class TestComplexOptIn:
    def test_nested_complex_nonfinite_and_literal_roundtrip(self):
        value = {'z': complex(-0.0, float('inf')), 'literal': {
            'status': 'complex', 're': 1.0, 'im': float('nan')}}
        wire = encode_json(value, complex_values=True)
        assert wire['z'] == {'status': 'complex', 're': -0.0,
                             'im': {'value': None, 'status': 'infinite', 'sign': '+'}}
        assert wire['literal']['status'] == 'literal'
        json.dumps(wire, allow_nan=False)
        restored = decode_json(wire, complex_values=True)
        assert restored['z'] == value['z']
        assert math.copysign(1, restored['z'].real) == -1
        assert restored['literal']['status'] == 'complex'
        assert math.isnan(restored['literal']['im'])
        assert encode_json(restored, complex_values=True) == wire

    @pytest.mark.parametrize('value', [
        {'status': 'complex', 're': 'bad', 'im': 0},
        {'status': 'complex', 're': True, 'im': 0},
        {'status': 'complex', 're': 1, 'im': None},
    ])
    def test_malformed_complex_reports_path(self, value):
        with pytest.raises(ValueError, match='metrics.z'):
            decode_json({'metrics': {'z': value}}, complex_values=True)

    def test_default_mode_unchanged(self):
        literal = {'status': 'complex', 're': 1, 'im': 2}
        assert encode_json(literal) == literal
        assert decode_json(literal) == literal
        assert encode_json(1 + 2j) == 1 + 2j


def test_complex_shaped_mapping_with_extra_fields_is_literal_data():
    literal = {'status': 'complex', 're': 1, 'im': 2, 'note': 'user data'}
    wire = encode_json(literal, complex_values=True)
    assert wire == literal
    assert decode_json(wire, complex_values=True) == literal
