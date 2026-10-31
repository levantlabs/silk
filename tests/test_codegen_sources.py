"""Tests for declarative source emission in silk.codegen.generator (2026-07-07 source-catalog plan)."""

import re
import subprocess

import pytest

from silk.codegen import generator as g
from silk.codegen.generator import Generator
from silk.signals.rawfile import read_rawfile
from silk.simulation.runner import ngspice_executable
from tests._env import needs_pdk, needs_ngspice


def _call_for(code, instance):
    """The factory-call expression emitted for ``instance``: the argument of
    its ``ckt.add(...)``/``mod.add(...)`` line (a leading ``*`` of an array
    splat removed). Instance names are never Python variables in the emitted
    code (TASK-351), so the call is evaluated, not exec'd."""
    for line in code.splitlines():
        m = re.match(r"(?:ckt|mod)\.add\(\*?(.*)\)$", line)
        if m and (f"('{instance}'" in m.group(1) or f"(f'{instance}" in m.group(1)):
            return m.group(1)
    raise AssertionError(f"no add() line for instance {instance!r}")


class TestSchemaDrivenMaps:
    def test_waveform_kind_reflected(self):
        assert g._waveform_kind('Pulse') == 'pulse'
        assert g._waveform_kind('RandomSource') == 'trrandom'

    def test_source_alias(self):
        assert g._source_alias('VoltageSource') == 'v'
        assert g._source_alias('CurrentSource') == 'i'


    def test_waveform_kind_fallback_for_unregistered(self):
        assert g._waveform_kind('TotallyNewThing') == 'totallynewthing'


def _call_for(code, instance):
    """The factory-call expression emitted for ``instance``: the argument of
    its single ``ckt.add(...)``/``mod.add(...)`` line (a leading ``*`` of an
    array splat removed). Instance names are never Python variables in the
    emitted code (TASK-351), so the call is evaluated, not exec'd."""
    for line in code.splitlines():
        m = re.match(r"(?:ckt|mod)\.add\(\*?(.*)\)$", line)
        if m and (f"('{instance}'" in m.group(1) or f"(f'{instance}" in m.group(1)):
            return m.group(1)
    raise AssertionError(f"no add() line for instance {instance!r}")


class TestSourcesArrayEmission:
    def _spec(self, src):
        # ports must be a dict ({}), not a list — CircuitSpec.from_dict calls .items()
        return {"circuit_name": "tb", "pdk": "sky130", "sources": [src]}

    def test_dc_source_declarative(self):
        code = Generator().generate(self._spec(
            {"type": "VoltageSource", "instance": "vdd",
             "positive": "vdd", "negative": "gnd", "dc": 1.8}))
        assert "ckt.add(sources.vdc('vdd'" in code
        assert "positive='vdd'" in code and "negative='gnd'" in code
        assert "sources.VoltageSource(" not in code

    def test_pulse_source_declarative(self):
        code = Generator().generate(self._spec(
            {"type": "VoltageSource", "instance": "clk",
             "positive": "clk", "negative": "gnd", "dc": 0,
             "waveform": {"type": "Pulse", "v1": 0, "v2": 1.8,
                          "td": 1e-9, "tr": 1e-9, "tf": 1e-9, "pw": 5e-9, "per": 10e-9}}))
        assert "ckt.add(sources.vpulse('clk'" in code
        assert "v1=0" in code and "per=1e-08" in code


class TestSourceCtorCallsActuallyExecute:
    """Real-behavior guard (not just string matching): the declarative call
    each line emits must actually construct against silk.sources.

    Caught during Task 5's rewrite: the pure-dc/ac branch of
    ``_source_array_call`` was emitting both ``value=<dc>`` and ``dc=<dc>`` —
    the scalar ctor (``_make_scalar_ctor``) takes ``dc`` positionally as
    ``value`` and rejects any leftover ``dc=`` kwarg, so the generated line
    raised ``TypeError: unexpected argument(s) ['dc']`` the moment it was
    executed, even though it looked correct as a string.
    """

    def test_source_call_dc_and_ac_constructs(self):
        from silk.codegen.generator import SourceSpec, _source_call
        from silk import sources
        src = SourceSpec(type='VoltageSource', instance='vdd',
                          positive='vdd', negative='gnd', dc=1.8, ac=0.1)
        ns = {'sources': sources}
        result = eval(_source_call(src), ns)
        assert result is not None

    def test_source_array_call_dc_and_ac_constructs(self):
        from silk.codegen.generator import SourceSpec, _source_array_call
        from silk import sources
        src = SourceSpec(type='VoltageSource', instance='VS', array=2,
                          positive='vref[$i]', negative='gnd', dc=1.0, ac=0.5)
        ns = {'sources': sources}
        result = eval(_source_array_call(src), ns)
        assert len(result) == 2

    def test_source_array_call_pulse_constructs(self):
        from silk.codegen.generator import SourceSpec, _source_array_call
        from silk import sources
        src = SourceSpec(type='VoltageSource', instance='clk', array=2,
                          positive='clk[$i]', negative='gnd', dc=0,
                          waveform={'type': 'Pulse', 'v1': 0, 'v2': 1.8,
                                    'td': 1e-9, 'tr': 1e-9, 'tf': 1e-9,
                                    'pw': 5e-9, 'per': 10e-9})
        ns = {'sources': sources}
        result = eval(_source_array_call(src), ns)
        assert len(result) == 2

    def test_source_call_pwl_constructs(self):
        from silk.codegen.generator import SourceSpec, _source_call
        from silk import sources
        src = SourceSpec(type='VoltageSource', instance='vpwl',
                          positive='vin', negative='gnd',
                          pwl=[0, 0, 1e-9, 1.8, 2e-9, 0])
        ns = {'sources': sources}
        result = eval(_source_call(src), ns)
        assert result is not None


class TestWaveformParamsShape:
    """TASK-250: the waveform dict takes the same ``{"type": <kind>,
    "params": {...}}`` shape ``devices[]``/``ideal_elements[]`` already use
    (generator.py module docstring), so a waveform constructor parameter
    literally named ``type`` — RandomSource / ``trrandom`` — can be
    expressed without colliding with the reserved class-name key. The
    legacy flat form (parameters as siblings of ``type``) stays accepted
    unchanged for every OTHER kind.

    Fixed defect: before this, ``_source_call_body`` read the class name
    from ``waveform['type']`` and dropped 'type' from the kwargs
    unconditionally, so TRRANDOM — whose own required ctor param is named
    'type' — could never be expressed through sources[]; only the direct
    call (``sources.vtrrandom(...)``) worked.
    """

    # Per-parameter-name values used to build a minimal, valid instance of
    # any waveform kind. Keyed by NAME, not by kind, because the same name
    # means the same thing across classes (v1/v2 in Pulse and Exponential;
    # vo/va/fc shared by SFFM and AM; ...).
    _PARAM_VALUES = {
        'v1': 0, 'v2': 1.8, 'tr': 1e-9, 'tf': 1e-9,
        'voffset': 0.9, 'vampl': 0.1, 'freq': 1e6,
        'vo': 0.9, 'va': 0.1, 'fc': 1e6, 'mdi': 5, 'fs': 1e3, 'mf': 1e3,
        'na': 20e-9, 'nt': 0.5e-9,
        'type': 1, 'ts': 1e-9,
        'points': [0, 0, 1e-9, 1.8],
        'bits': '101', 'order': 7, 'length': 10, 'ui': 10e-9,
    }

    def _spec(self, waveform):
        return {
            "circuit_name": "tb", "pdk": "sky130",
            "sources": [
                {"type": "VoltageSource", "instance": "vwf",
                 "positive": "vwf", "negative": "gnd",
                 "waveform": waveform},
            ],
        }

    def test_every_export_schema_waveform_kind_is_expressible(self):
        """AC #1/#2: enumerate the waveform kinds from
        ``silk.sources.schema.export_source_schema`` at RUNTIME — the exported
        schema (source_metadata.yaml is its authored half; a new
        kind needs no edit here) — never a hand-written list. For each kind,
        build a sources[] spec in the params form, generate, exec the
        generated module, and assert the emitted SPICE card carries that
        waveform's own keyword (derived from the class's own to_spice(),
        not a hand kind -> keyword map), so a kind added later cannot arrive
        unreachable with a green suite."""
        from silk.sources.schema import export_source_schema
        from silk.sources import waveforms as wf_mod
        from silk import sources

        schema = export_source_schema(include_metadata=False)
        kinds = schema['VoltageSource']['waveforms']
        assert kinds, "no waveform kinds advertised — test would pass vacuously"
        assert 'trrandom' in kinds  # the reported defect kind must be covered

        for kind, wf_schema in kinds.items():
            class_name = wf_schema['class']
            params = {
                p['name']: self._PARAM_VALUES[p['name']]
                for p in wf_schema['parameters'] if p['required']
            }
            if kind == 'pwl':
                # PWL's only substantive param ('points') has no default that
                # produces a valid card — without it to_spice() raises.
                params['points'] = self._PARAM_VALUES['points']

            spec = self._spec({"type": class_name, "params": params})
            code = Generator().generate(spec)

            # Expected SPICE keyword, derived from the real class's own
            # to_spice() — not a hand-written kind -> keyword map.
            wf_cls = getattr(wf_mod, class_name)
            expected_keyword = wf_cls(**params).to_spice().split('(')[0]

            card = eval(_call_for(code, 'vwf'), {'sources': sources}).to_spice()
            assert f'{expected_keyword}(' in card, (
                f"kind {kind!r} ({class_name}): expected {expected_keyword!r} "
                f"card, got: {card!r}"
            )

    def test_legacy_flat_form_still_generates_byte_identical(self):
        """The flat form (parameters as siblings of 'type', no nested
        'params') is untouched by TASK-250: the else-branch that builds
        wf_args from it is the same generator expression the pre-fix code
        used. Pin the exact generated lines for Pulse and Sine so a future
        change to this branch is caught."""
        pulse_flat = {"type": "Pulse", "v1": 0, "v2": 1.8, "td": 1e-9,
                      "tr": 1e-9, "tf": 1e-9, "pw": 5e-9, "per": 10e-9}
        code = Generator().generate(self._spec(pulse_flat))
        assert ("ckt.add(sources.vpulse('vwf', positive='vwf', negative='gnd', "
                "v1=0, v2=1.8, td=1e-09, tr=1e-09, tf=1e-09, pw=5e-09, "
                "per=1e-08))") in code

        sine_flat = {"type": "Sine", "voffset": 0.9, "vampl": 0.1, "freq": 1e6}
        code = Generator().generate(self._spec(sine_flat))
        assert ("ckt.add(sources.vsine('vwf', positive='vwf', negative='gnd', "
                "voffset=0.9, vampl=0.1, freq=1e+06))") in code

    def test_params_and_flat_together_raises(self):
        """A dict mixing the nested 'params' form with a flat keyword beside
        'type' is rejected with a clear error, not silently merged."""
        import pytest
        bad = {"type": "Pulse", "params": {"v1": 0, "v2": 1.8,
                                            "tr": 1e-9, "tf": 1e-9},
               "pw": 5e-9}
        with pytest.raises(ValueError, match="mixes"):
            Generator().generate(self._spec(bad))

    def test_trrandom_via_params_matches_direct_call(self):
        """The reported defect, closed: TRRANDOM's own 'type' parameter,
        expressed through 'params', produces the same card as the direct
        (non-declarative) call."""
        from silk import sources
        spec = self._spec({"type": "RandomSource",
                            "params": {"type": 1, "ts": 1e-9}})
        code = Generator().generate(spec)
        generated_card = eval(_call_for(code, 'vwf'), {'sources': sources}).to_spice()

        direct = sources.vtrrandom('vwf', positive='vwf', negative='gnd',
                                    type=1, ts=1e-9)
        assert generated_card == direct.to_spice()

    def test_trrandom_with_enum_choice_matches_direct_call(self):
        """TASK-252 AC #3: adding ``choices`` to the schema (RandomType) is
        purely additive to the exported metadata — a spec carrying
        ``{"type": "trrandom", "params": {"type": 2, ...}}`` (as a caller would
        send it, 2 = gaussian) still emits the same TRRANDOM card as before."""
        from silk import sources
        spec = self._spec({"type": "RandomSource",
                            "params": {"type": 2, "ts": 1e-9}})
        code = Generator().generate(spec)
        generated_card = eval(_call_for(code, 'vwf'), {'sources': sources}).to_spice()

        direct = sources.vtrrandom('vwf', positive='vwf', negative='gnd',
                                    type=2, ts=1e-9)
        assert generated_card == direct.to_spice()
        assert 'TRRANDOM(2 1e-09 0 1 0)' in generated_card


class TestControlledSourceCodegen:
    """TASK-253: VCVS/VCCS/CCVS/CCCS round-trip through spec['sources'] ->
    generate -> exec -> to_spice(), per AC #2."""

    def _spec(self, entries):
        return {"circuit_name": "tb", "pdk": "sky130",
                "sources": entries}

    def _generate_and_exec(self, entries):
        """Generate, then eval only each entry's own factory call (same
        pattern as TestWaveformParamsShape/TestSourceCtorCallsActuallyExecute
        above) — never the whole script, which would call load_pdk/simulate."""
        from silk import sources
        code = Generator().generate(self._spec(entries))
        ns = {'sources': sources}
        for entry in entries:
            instance = entry['instance']
            ns[instance] = eval(_call_for(code, instance), ns)
        return ns, code

    def test_vcvs_round_trip(self):
        entries = [{"type": "VCVS", "instance": "e1",
                    "out_pos": "vout", "out_neg": "gnd",
                    "ctrl_pos": "vin", "ctrl_neg": "gnd", "gain": 10}]
        ns, code = self._generate_and_exec(entries)
        assert "ckt.add(sources.VoltageControlledVoltageSource('e1'" in code
        card = ns['e1'].to_spice()
        assert card == 'Ee1 vout gnd vin gnd 10\n'

    def test_vccs_round_trip(self):
        entries = [{"type": "VoltageControlledCurrentSource", "instance": "g1",
                    "out_pos": "iout", "out_neg": "gnd",
                    "ctrl_pos": "vin", "ctrl_neg": "gnd", "gm": 0.01}]
        ns, code = self._generate_and_exec(entries)
        assert "ckt.add(sources.VoltageControlledCurrentSource('g1'" in code
        card = ns['g1'].to_spice()
        assert card == 'Gg1 iout gnd vin gnd 0.01\n'

    def test_ccvs_round_trip(self):
        entries = [
            {"type": "VoltageSource", "instance": "vmeas",
             "positive": "vmeas_p", "negative": "vmeas_n", "dc": 0},
            {"type": "CCVS", "instance": "h1",
             "out_pos": "vout", "out_neg": "gnd",
             "transresistance": 1000, "ctrl_vname": "vmeas"},
        ]
        ns, code = self._generate_and_exec(entries)
        assert "ckt.add(sources.CurrentControlledVoltageSource('h1'" in code
        card = ns['h1'].to_spice()
        assert card == 'Hh1 vout gnd vmeas 1000\n'

    def test_cccs_round_trip(self):
        entries = [
            {"type": "VoltageSource", "instance": "vmeas",
             "positive": "vmeas_p", "negative": "vmeas_n", "dc": 0},
            {"type": "CCCS", "instance": "f1",
             "out_pos": "icopy", "out_neg": "gnd",
             "gain": 2, "ctrl_vname": "vmeas"},
        ]
        ns, code = self._generate_and_exec(entries)
        assert "ckt.add(sources.CurrentControlledCurrentSource('f1'" in code
        card = ns['f1'].to_spice()
        assert card == 'Ff1 icopy gnd vmeas 2\n'

    def test_all_four_coexist_with_add_source(self):
        """Wired into the same ckt.add(...) loop as VoltageSource, with an
        independent source (the ammeter) alongside them (TASK-282.1: every
        source is emitted via ckt.add; add_source stays a valid alias)."""
        entries = [
            {"type": "VoltageSource", "instance": "vmeas",
             "positive": "vmeas_p", "negative": "vmeas_n", "dc": 0},
            {"type": "VCVS", "instance": "e1", "out_pos": "vout", "out_neg": "gnd",
             "ctrl_pos": "vin", "ctrl_neg": "gnd", "gain": 10},
            {"type": "VCCS", "instance": "g1", "out_pos": "iout", "out_neg": "gnd",
             "ctrl_pos": "vin", "ctrl_neg": "gnd", "gm": 0.01},
            {"type": "CCVS", "instance": "h1", "out_pos": "vout2", "out_neg": "gnd",
             "transresistance": 1000, "ctrl_vname": "vmeas"},
            {"type": "CCCS", "instance": "f1", "out_pos": "icopy", "out_neg": "gnd",
             "gain": 2, "ctrl_vname": "vmeas"},
        ]
        code = Generator().generate(self._spec(entries))
        for inst in ('vmeas', 'e1', 'g1', 'h1', 'f1'):
            assert f"('{inst}'" in code and f"{inst} = " not in code

    def test_ctrl_vname_referencing_unknown_source_raises(self):
        entries = [{"type": "CCVS", "instance": "h1",
                    "out_pos": "vout", "out_neg": "gnd",
                    "transresistance": 1000, "ctrl_vname": "nope"}]
        with pytest.raises(ValueError, match="ctrl_vname"):
            Generator().generate(self._spec(entries))

    def test_missing_required_pin_raises_key_error(self):
        entries = [{"type": "VCVS", "instance": "e1",
                    "out_pos": "vout", "out_neg": "gnd",
                    "ctrl_pos": "vin", "gain": 10}]   # ctrl_neg missing
        with pytest.raises(KeyError, match="ctrl_neg"):
            Generator().generate(self._spec(entries))

    def test_missing_scalar_param_raises_key_error(self):
        entries = [{"type": "CCCS", "instance": "f1",
                    "out_pos": "icopy", "out_neg": "gnd",
                    "ctrl_vname": "vmeas"}]   # gain missing
        with pytest.raises(KeyError, match="gain"):
            Generator().generate(self._spec(entries))

    def test_missing_ctrl_vname_raises_key_error(self):
        entries = [{"type": "CCCS", "instance": "f1",
                    "out_pos": "icopy", "out_neg": "gnd", "gain": 2}]
        with pytest.raises(KeyError, match="ctrl_vname"):
            Generator().generate(self._spec(entries))

    def test_short_alias_and_full_class_name_both_accepted(self):
        """'type' accepts VCVS or VoltageControlledVoltageSource — same
        emitted class."""
        via_alias = self._generate_and_exec(
            [{"type": "VCVS", "instance": "e1", "out_pos": "vout", "out_neg": "gnd",
              "ctrl_pos": "vin", "ctrl_neg": "gnd", "gain": 10}])[0]
        via_full = self._generate_and_exec(
            [{"type": "VoltageControlledVoltageSource", "instance": "e1",
              "out_pos": "vout", "out_neg": "gnd",
              "ctrl_pos": "vin", "ctrl_neg": "gnd", "gain": 10}])[0]
        assert via_alias['e1'].to_spice() == via_full['e1'].to_spice()

    def test_module_path_emits_controlled_source(self):
        """Same branch, module (non-standalone) output mode."""
        spec = {"circuit_name": "m", "pdk": "sky130",
                "ports": {"a": {"dir": "input"}},
                "sources": [{"type": "VCVS", "instance": "e1",
                             "out_pos": "vout", "out_neg": "gnd",
                             "ctrl_pos": "a", "ctrl_neg": "gnd", "gain": 10}]}
        code = Generator().generate(spec)
        assert "mod.add(sources.VoltageControlledVoltageSource('e1'" in code
        assert "e1 = " not in code


class TestBehavioralSourceCodegen:
    """TASK-282.2: BehavioralVoltage/BehavioralCurrent round-trip through
    spec['sources'] -> generate -> exec -> to_spice(), same pattern as
    TestControlledSourceCodegen above."""

    def _spec(self, entries, ports=None):
        return {"circuit_name": "tb", "pdk": "sky130",
                **({"ports": ports} if ports else {}), "sources": entries}

    def _generate_and_exec(self, entries):
        from silk import sources
        code = Generator().generate(self._spec(entries))
        ns = {'sources': sources}
        for entry in entries:
            instance = entry['instance']
            ns[instance] = eval(_call_for(code, instance), ns)
        return ns, code

    def test_behavioral_voltage_round_trip(self):
        entries = [{"type": "BehavioralVoltage", "instance": "bv1",
                    "positive": "vd", "negative": "gnd",
                    "expression": "v(inp)-v(inn)"}]
        ns, code = self._generate_and_exec(entries)
        assert "ckt.add(sources.BehavioralVoltage('bv1'" in code
        assert "bv1 = " not in code
        card = ns['bv1'].to_spice()
        assert card == 'Bbv1 vd gnd V={v(inp)-v(inn)}\n'

    def test_behavioral_current_round_trip(self):
        entries = [{"type": "BehavioralCurrent", "instance": "bi1",
                    "positive": "io", "negative": "gnd",
                    "expression": "0.01*v(vin,gnd)"}]
        ns, code = self._generate_and_exec(entries)
        assert "ckt.add(sources.BehavioralCurrent('bi1'" in code
        assert "bi1 = " not in code
        card = ns['bi1'].to_spice()
        assert card == 'Bbi1 io gnd I={0.01*v(vin,gnd)}\n'

    def test_all_sources_coexist_with_add(self):
        entries = [
            {"type": "VoltageSource", "instance": "vmeas",
             "positive": "vmeas_p", "negative": "vmeas_n", "dc": 0},
            {"type": "BehavioralVoltage", "instance": "bv1",
             "positive": "vd", "negative": "gnd", "expression": "v(inp)-v(inn)"},
            {"type": "BehavioralCurrent", "instance": "bi1",
             "positive": "io", "negative": "gnd", "expression": "0.01*v(vin,gnd)"},
            {"type": "CCVS", "instance": "h1", "out_pos": "vout2", "out_neg": "gnd",
             "transresistance": 1000, "ctrl_vname": "vmeas"},
        ]
        code = Generator().generate(self._spec(entries))
        for inst in ('vmeas', 'bv1', 'bi1', 'h1'):
            assert f"('{inst}'" in code and f"{inst} = " not in code

    def test_missing_expression_raises_key_error(self):
        entries = [{"type": "BehavioralVoltage", "instance": "bv1",
                    "positive": "vd", "negative": "gnd"}]
        with pytest.raises(KeyError, match="bv1"):
            Generator().generate(self._spec(entries))
        with pytest.raises(KeyError, match="expression"):
            Generator().generate(self._spec(entries))

    def test_missing_positive_raises_key_error(self):
        entries = [{"type": "BehavioralCurrent", "instance": "bi1",
                    "negative": "gnd", "expression": "0"}]
        with pytest.raises(KeyError, match="positive"):
            Generator().generate(self._spec(entries))

    def test_missing_negative_raises_key_error(self):
        entries = [{"type": "BehavioralVoltage", "instance": "bv1",
                    "positive": "vd", "expression": "0"}]
        with pytest.raises(KeyError, match="negative"):
            Generator().generate(self._spec(entries))

    def test_module_path_emits_behavioral_source(self):
        spec = self._spec(
            [{"type": "BehavioralVoltage", "instance": "bv1",
              "positive": "a", "negative": "gnd", "expression": "v(a)"}],
            ports={"a": {"dir": "input"}},
        )
        code = Generator().generate(spec)
        assert "mod.add(sources.BehavioralVoltage('bv1'" in code
        assert "bv1 = " not in code

    @pytest.mark.slow
    @needs_ngspice
    @pytest.mark.parametrize('type_name,expression,expected_v', [
        ('BehavioralVoltage', '1.0', 1.0),
        # B I=expression sinks current INTO its positive terminal from the
        # external circuit (opposite convention from B V=), so 1 mA through
        # Rload=1k gives v(out) = -1.0, not +1.0 — measured directly here
        # rather than assumed.
        ('BehavioralCurrent', '0.001', -1.0),
    ])
    def test_generated_source_simulates_expected_value_in_ngspice(
            self, tmp_path, type_name, expression, expected_v):
        """AC #1: the generated ``sources.<Type>(...)`` call constructs the
        real class, and the resulting card simulates the expected value —
        not just that to_spice() has the right text."""
        entries = [{"type": type_name, "instance": "b1",
                    "positive": "out", "negative": "0",
                    "expression": expression}]
        ns, code = self._generate_and_exec(entries)
        deck = tmp_path / 'behavioral.cir'
        deck.write_text(
            '* Behavioral source codegen regression\n'
            + ns['b1'].to_spice()
            + 'Rload out 0 1k\n.op\n.end\n'
        )
        raw_path = tmp_path / 'behavioral.raw'
        subprocess.run([ngspice_executable(), '-b', '-r', str(raw_path), str(deck)],
                       check=True, capture_output=True, timeout=30)
        vectors = {v.name.lower(): v.data for v in read_rawfile(raw_path).vectors}
        assert vectors['v(out)'] == pytest.approx([expected_v], abs=1e-9)


class TestGeneratorAcceptsEveryExportedSourceSchemaType:
    """TASK-282.2 AC #4: every type export_source_schema() advertises (its
    full-schema `sources:` map plus `_meta.unschematized_sources`) is
    accepted by the generator — a future schema addition without generator
    support fails this test rather than surfacing as a bare KeyError."""

    # Minimal valid sources[] kwargs per class, keyed by class name so a new
    # entry in the schema requires an explicit addition here.
    _KWARGS = {
        'VoltageSource': dict(positive='o', negative='gnd', dc=1),
        'CurrentSource': dict(positive='o', negative='gnd', dc=0.001),
        'VoltageControlledVoltageSource': dict(
            out_pos='o', out_neg='gnd', ctrl_pos='i', ctrl_neg='gnd', gain=1),
        'VoltageControlledCurrentSource': dict(
            out_pos='o', out_neg='gnd', ctrl_pos='i', ctrl_neg='gnd', gm=0.01),
        'CurrentControlledVoltageSource': dict(
            out_pos='o', out_neg='gnd', transresistance=1000, ctrl_vname='seed'),
        'CurrentControlledCurrentSource': dict(
            out_pos='o', out_neg='gnd', gain=2, ctrl_vname='seed'),
        'BehavioralVoltage': dict(positive='o', negative='gnd', expression='1'),
        'BehavioralCurrent': dict(positive='o', negative='gnd', expression='0.001'),
    }

    def test_every_schema_type_is_accepted(self):
        from silk.sources.schema import export_source_schema
        schema = export_source_schema()
        advertised = (set(schema) - {'_meta'}) | set(schema['_meta']['unschematized_sources'])
        assert advertised == set(self._KWARGS), (
            "a source type was added to export_source_schema() without a "
            "matching entry in this test's _KWARGS table"
        )
        for type_name, kwargs in self._KWARGS.items():
            entries = [{"type": "VoltageSource", "instance": "seed",
                        "positive": "seed", "negative": "gnd", "dc": 0},
                       {"type": type_name, "instance": "under_test", **kwargs}]
            code = Generator().generate(
                {"circuit_name": "tb", "pdk": "sky130",
                 "sources": entries})
            assert "under_test = " not in code and "('under_test'" in code


class TestEveryStepUsesCktAdd:
    """TASK-282.1 AC #3: every emitted source uses ckt.add (not
    ckt.add_source); the SPICE deck it produces is byte-identical either
    way, since Module.add() already routes a SourceBase to add_source()
    internally."""

    @needs_pdk
    def test_ckt_add_and_add_source_produce_byte_identical_deck(self, tmp_path):
        import os

        def _spec(output_dir):
            return {
                "circuit_name": "rep", "pdk": "sky130",
                "output_dir": str(output_dir),
                "ideal_elements": [
                    {"type": "resistor", "instance": "r1",
                     "nets": {"p": "vout", "n": "gnd"}, "params": {"value": 1000.0}},
                ],
                "sources": [
                    {"type": "VoltageSource", "instance": "vdd",
                     "positive": "vdd", "negative": "gnd", "dc": 1.8},
                    {"type": "CurrentSource", "instance": "iin",
                     "positive": "gnd", "negative": "vout", "dc": 1e-3},
                ],
            }

        def _generate_deck(output_dir, use_add_source):
            code = Generator().generate(_spec(output_dir))
            assert "ckt.add(sources.vdc('vdd'" in code
            assert "ckt.add(sources.idc('iin'" in code
            if use_add_source:
                code = (code.replace("ckt.add(sources.vdc('vdd'",
                                     "ckt.add_source(sources.vdc('vdd'")
                            .replace("ckt.add(sources.idc('iin'",
                                     "ckt.add_source(sources.idc('iin'"))
            code = code.replace("ckt.simulate(corner='tt')",
                                 "ckt.set_output_options(mode='overwrite')\ndeck_path = ckt.generate_netlist(corner='tt')")
            ns: dict = {}
            exec(code, ns)
            ckt = ns['ckt']
            with open(ns['deck_path']) as fh:
                return fh.read()

        dir_add = tmp_path / 'via_add'
        dir_add_source = tmp_path / 'via_add_source'
        dir_add.mkdir()
        dir_add_source.mkdir()

        deck_via_add = _generate_deck(dir_add, use_add_source=False)
        deck_via_add_source = _generate_deck(dir_add_source, use_add_source=True)
        # Normalise the two decks' own output directories out of the text —
        # they necessarily differ (each needs its own dir to avoid clobbering
        # the other), which is unrelated to the add()-vs-add_source() claim.
        assert (deck_via_add.replace(str(dir_add), '<dir>')
                == deck_via_add_source.replace(str(dir_add_source), '<dir>'))


@pytest.mark.parametrize('array', [None, 4])
@pytest.mark.parametrize('collection,bad_type,accepted', [
    ('ideal_elements', 'VCVS',
     ['resistor', 'capacitor', 'inductor']),
    ('ideal_elements', 'Resistor',
     ['resistor', 'capacitor', 'inductor']),
    ('sources', 'vcvs',
     ['VoltageSource', 'CurrentSource', 'VCVS', 'VCCS', 'CCVS', 'CCCS',
      'VoltageControlledVoltageSource', 'VoltageControlledCurrentSource',
      'CurrentControlledVoltageSource', 'CurrentControlledCurrentSource',
      'BehavioralVoltage', 'BehavioralCurrent']),
    ('sources', 'NotASource', ['VoltageSource', 'CurrentSource']),
])
@pytest.mark.parametrize('context', ['circuit', 'rigs', 'tests'])
def test_unknown_entry_type_refused_at_spec_time(collection, bad_type, accepted,
                                                array, context):
    """Unknown factories must fail before emission, including nested benches."""
    from silk.codegen.testbench_generator import TestbenchSpec

    entry = {'type': bad_type, 'instance': 'bad_entry',
             'nets': {'p': 'out', 'n': '0'}, 'params': {},
             'positive': 'out', 'negative': '0'}
    if array is not None:
        entry['array'] = array
    group = {collection: [entry]}
    if context == 'circuit':
        parse = g.CircuitSpec.from_dict
        spec = {'circuit_name': 'c', **group}
    else:
        parse = TestbenchSpec.from_dict
        spec = {'circuit_name': 'c', context: [{'name': 'bench', **group}]}
    with pytest.raises(ValueError) as exc:
        parse(spec)
    message = str(exc.value)
    assert collection in message
    assert 'bad_entry' in message
    assert bad_type in message
    assert 'accepted' in message.lower()
    for name in accepted:
        assert name in message


@pytest.mark.parametrize('entries', ['invalid', [None], ['invalid']])
def test_testbench_malformed_ideal_entries_keep_structural_error(entries):
    from silk.codegen.testbench_generator import (
        TestbenchSpecError, generate_testbenches,
    )

    spec = {'circuit_name': 'c', 'tests': [
        {'name': 'bench', 'dut_portmap': {}, 'ideal_elements': entries},
    ]}
    with pytest.raises(TestbenchSpecError, match='ideal_elements.*must be'):
        generate_testbenches(spec)

@pytest.mark.parametrize('kind', ['vsource', 'isource'])
@pytest.mark.parametrize('nets', [{'p': 'out', 'n': 'gnd'}, [{'p': 'out', 'n': 'gnd'}]])
def test_ideal_source_spec_is_refused(kind, nets):
    with pytest.raises(ValueError, match=r"ideal_elements entry 'bias'.*sources\[\]"):
        g.CircuitSpec.from_dict({'ideal_elements': [
            dict(type=kind, instance='bias', nets=nets, params={'dc': 1},
                 **({'array': 1} if isinstance(nets, list) else {}))]})


def test_elements_schema_export_stays_byte_identical():
    # Repinned for TASK-475: additive integer source parameter kinds.
    # Pre-existing source entries are separately checked byte-for-byte.
    import hashlib
    import json
    import warnings

    from silk.elements import elements
    from silk.elements.schema import export_schema

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cat = export_schema(elements, include_metadata=False)
    digest = hashlib.sha256(
        json.dumps(cat, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()
    # TASK-535: sine voffset now defaults to 0; other schema fields are frozen.
    assert digest == (
        '6a6a72d8560db820e0cdec002895c2338fdf8d8c0c6802a6211dc0b3997adbcb'
    )


@pytest.mark.parametrize('ports', [None, {'a': {'dir': 'input'}}])
def test_passive_only_spec_imports_elements(ports):
    spec = {'circuit_name': 'tb', 'pdk': 'sky130', 'ideal_elements': [
        {'type': 'resistor', 'instance': 'r1',
         'nets': {'p': 'a', 'n': 'gnd'}, 'params': {'value': 1000}}]}
    if ports is not None:
        spec['ports'] = ports
    code = Generator().generate(spec)
    assert 'from silk.elements import elements' in code
    assert 'from silk import sources' not in code


@pytest.mark.parametrize('kind', ['vsource', 'isource'])
@pytest.mark.parametrize('wiring', ['scalar', 'broadcast', 'per_copy'])
@pytest.mark.parametrize('ports', [None, {}])
def test_direct_ideal_source_spec_is_refused(kind, wiring, ports):
    nets = {'p': 'out', 'n': 'gnd'}
    entry = g.DeviceSpec(kind, 'bias', [nets] if wiring == 'per_copy' else nets,
                         {'dc': 1}, array=None if wiring == 'scalar' else 1)
    spec = g.CircuitSpec('demo', ports=ports, ideal_elements=[entry])
    with pytest.raises(ValueError) as direct:
        Generator().generate(spec)
    with pytest.raises(ValueError) as parsed:
        g.CircuitSpec.from_dict({'ideal_elements': [vars(entry)]})
    assert str(direct.value) == str(parsed.value)
    assert "ideal_elements entry 'bias'" in str(direct.value)
    assert 'sources[]' in str(direct.value)


@pytest.mark.parametrize('kind', ['vsource', 'isource'])
@pytest.mark.parametrize('wiring', ['scalar', 'broadcast', 'per_copy'])
@pytest.mark.parametrize('existing', [False, True])
def test_direct_ideal_source_spec_does_not_publish_file(tmp_path, kind, wiring, existing):
    nets = {'p': 'out', 'n': 'gnd'}
    entry = g.DeviceSpec(kind, 'bias', [nets] if wiring == 'per_copy' else nets,
                         {'dc': 1}, array=None if wiring == 'scalar' else 1)
    spec = g.CircuitSpec('demo', ideal_elements=[entry])
    path = tmp_path / 'generated.py'
    if existing:
        path.write_text('sentinel\n')
    with pytest.raises(ValueError, match=r"ideal_elements entry 'bias'.*sources\[\]"):
        Generator().generate_to_file(spec, path)
    if existing:
        assert path.read_text() == 'sentinel\n'
    else:
        assert not path.exists()
