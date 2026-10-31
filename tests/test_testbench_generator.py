"""Test silk.codegen.testbench_generator."""
import pytest

from silk.codegen.testbench_generator import TestbenchSpec, TestbenchSpecError, generate_testbenches


def test_from_dict_roundtrips_minimal():
    d = {"circuit_name": "ac_gain", "circuit_builder": "_build_ac_gain",
         "rigs": [], "tests": [{"name": "ac", "dut_portmap": {"vout": "vout"},
                                "applied_rigs": [], "sources": [], "analyses": [],
                                "metrics": [], "specs": []}]}
    spec = TestbenchSpec.from_dict(d)
    assert spec.circuit_name == 'ac_gain'
    assert spec.circuit_builder == '_build_ac_gain'
    assert spec.tests[0]['name'] == 'ac'


def test_testbench_spec_error_is_valueerror():
    assert issubclass(TestbenchSpecError, ValueError)


def test_from_dict_raises_on_missing_circuit_name():
    d = {"circuit_builder": "_build_ac_gain", "rigs": [], "tests": []}
    try:
        TestbenchSpec.from_dict(d)
        assert False, "Should have raised TestbenchSpecError"
    except TestbenchSpecError:
        pass


def test_circuit_builder_defaults_to_none():
    d = {"circuit_name": "ac_gain", "rigs": [], "tests": []}
    spec = TestbenchSpec.from_dict(d)
    assert spec.circuit_builder is None


def test_rigs_and_tests_default_to_empty_lists():
    d = {"circuit_name": "ac_gain"}
    spec = TestbenchSpec.from_dict(d)
    assert spec.rigs == []
    assert spec.tests == []


# --- _validate: §2.1 fail-loud validation pass -----------------------------
#
# generate_testbenches (Task 8) runs _validate as its first step, so these
# tests now go through the full public entry point rather than calling
# _validate directly — this also exercises Task 8's assembly for every
# valid spec (test_valid_spec_does_not_raise) and its KeyError backstop.

def _base(**test_over):
    t = {"name": "t", "dut_portmap": {"vout": "vout"}, "applied_rigs": [],
         "sources": [], "analyses": [], "metrics": [], "specs": []}
    t.update(test_over)
    return {"circuit_name": "c", "circuit_builder": "_build_c", "rigs": [], "tests": [t]}


def _validate_dict(d):
    generate_testbenches(d)


def test_spec_references_absent_metric():
    d = _base(specs=[{"name": "s", "metric": "ghost", "op": ">", "target": 1}])
    with pytest.raises(TestbenchSpecError, match="ghost"):
        _validate_dict(d)


def test_applied_rig_undefined():
    d = _base(applied_rigs=["nope"])
    with pytest.raises(TestbenchSpecError, match="nope"):
        _validate_dict(d)


def test_duplicate_test_names_rejected():
    d = _base()
    d["tests"].append(dict(d["tests"][0]))
    with pytest.raises(TestbenchSpecError, match="duplicate test name 't'"):
        _validate_dict(d)


def test_duplicate_rig_names_rejected():
    d = _base()
    d["rigs"] = [
        {"name": "supply", "sources": []},
        {"name": "supply", "sources": []},
    ]
    with pytest.raises(TestbenchSpecError, match="duplicate rig name 'supply'"):
        _validate_dict(d)


def test_metric_missing_signal():
    d = _base(metrics=[{"name": "g", "type": "dc_gain_dB", "params": {}}])
    with pytest.raises(TestbenchSpecError, match="signal"):
        _validate_dict(d)


def test_unknown_metric_type():
    d = _base(metrics=[{"name": "g", "type": "bogus", "signal": "xdut.vout"}])
    with pytest.raises(TestbenchSpecError, match="bogus"):
        _validate_dict(d)


def test_missing_required_param():
    d = _base(metrics=[{"name": "g", "type": "gain_at_freq", "signal": "xdut.vout", "params": {}}])
    with pytest.raises(TestbenchSpecError, match="freq"):
        _validate_dict(d)


def test_unknown_analysis_kind():
    d = _base(analyses=[{"kind": "montecarlo"}])
    with pytest.raises(TestbenchSpecError, match="montecarlo"):
        _validate_dict(d)


def test_valid_spec_does_not_raise():
    d = _base(
        metrics=[{"name": "g", "type": "dc_gain_dB", "signal": "xdut.vout"}],
        specs=[{"name": "s", "metric": "g", "op": ">", "target": 1}],
        analyses=[{"kind": "AC", "variation": "dec", "points": 10,
                   "fstart": 1, "fstop": 1e9}],
    )
    _validate_dict(d)  # must not raise


# --- v2 validator: satisfiability, not input count (TASK-199) --------------
#
# _validate used to refuse a metric
# for TAKING more than one signal — the "v1 boundary", which made all eight
# multi-signal metrics undeclarable (TASK-183) — and it is retired here, along
# with its test (test_multi_signal_metric_rejected). What replaces it asks
# whether the spec can be SATISFIED: every slot bound by name and none
# invented, no slot needing a kind nothing silk simulates produces, every
# param inside its declared choices, and an analysis in the same test that
# matches the metric's domain (D7). Every message names what it refuses.
#
# The §6.2 messages are asserted verbatim: they are the whole point of the
# refusal — "needs 2 signals" told a user nothing about which one they forgot.

_AC = [{"kind": "ac", "variation": "dec", "points": 10, "fstart": 1, "fstop": 1e9}]
_TRAN = [{"kind": "transient", "step": 1e-9, "stop": 1e-6}]
_NOISE = [{"kind": "noise", "output": "vout", "ref": "0", "source": "vin"}]


def _metric_lines(d):
    return [ln.strip() for ln in generate_testbenches(d).splitlines()
            if 'add_metric' in ln]


def test_two_signal_metric_is_bound_by_name_and_emitted():
    """The headline: cmrr_dB — refused outright until now — generates."""
    d = _base(analyses=_AC,
              metrics=[{"name": "cmrr", "type": "cmrr_dB",
                        "signals": {"vdiff": "voutd", "vcm": "voutc"},
                        "params": {"freq": 1e3}}])
    assert _metric_lines(d) == [
        "tb.add_metric('cmrr', type='cmrr_dB', "
        "signals={'vdiff': 'voutd', 'vcm': 'voutc'}, freq=1000.0)"]


def test_four_signal_metric_generates():
    d = _base(analyses=_TRAN,
              metrics=[{"name": "h", "type": "hysteresis",
                        "signals": {"sig_in_rising": "a", "sig_out_rising": "b",
                                    "sig_in_falling": "c", "sig_out_falling": "d"}}])
    assert _metric_lines(d) == [
        "tb.add_metric('h', type='hysteresis', signals={'sig_in_rising': 'a', "
        "'sig_out_rising': 'b', 'sig_in_falling': 'c', 'sig_out_falling': 'd'})"]


def test_signals_on_a_one_input_metric_is_accepted():
    """§6.2 case four: the explicit spelling of signal=, not an error."""
    d = _base(analyses=_AC,
              metrics=[{"name": "g", "type": "dc_gain_dB", "signals": {"sig": "vout"}}])
    assert _metric_lines(d) == [
        "tb.add_metric('g', type='dc_gain_dB', signals={'sig': 'vout'})"]


def test_one_input_signal_form_emits_byte_identically():
    """A v1 spec must generate exactly the line it always did."""
    d = _base(analyses=_AC,
              metrics=[{"name": "g", "type": "dc_gain_dB", "signal": "xdut.vout"}])
    assert _metric_lines(d) == [
        "tb.add_metric('g', type='dc_gain_dB', signal='xdut.vout')"]


def _refusal(d):
    with pytest.raises(TestbenchSpecError) as exc:
        _validate_dict(d)
    return str(exc.value)


def test_signal_on_a_multi_input_metric_is_refused_by_name():
    d = _base(analyses=_AC,
              metrics=[{"name": "cmrr", "type": "cmrr_dB", "signal": "vout"}])
    assert _refusal(d) == (
        "test 't' metric 'cmrr' (type 'cmrr_dB'): 'signal' binds a single "
        "input, but this metric takes 2: 'vdiff', 'vcm'. Use \"signals\": "
        "{\"vdiff\": ..., \"vcm\": ...}.")


def test_unbound_slot_is_refused_by_name():
    d = _base(analyses=_TRAN,
              metrics=[{"name": "tpd", "type": "propagation_delay",
                        "signals": {"sig_in": "vin"}, "params": {"in_edge": "rising"}}])
    assert _refusal(d) == (
        "test 't' metric 'tpd' (type 'propagation_delay'): 'signals' is "
        "missing 'sig_out'. This metric takes 2 signal inputs: 'sig_in' (kind "
        "'transient'), 'sig_out' (kind 'transient'); bound: 'sig_in'.")


def test_unknown_slot_is_refused_by_name():
    d = _base(analyses=_TRAN,
              metrics=[{"name": "tpd", "type": "propagation_delay",
                        "signals": {"sig_in": "vin", "sig_output": "vout"}}])
    assert _refusal(d) == (
        "test 't' metric 'tpd' (type 'propagation_delay'): 'signals' has "
        "unknown key 'sig_output'; this metric's signal inputs are 'sig_in', "
        "'sig_out'.")


def test_no_binding_at_all_is_refused_with_the_form_to_use():
    d = _base(analyses=_TRAN,
              metrics=[{"name": "tpd", "type": "propagation_delay"}])
    assert _refusal(d) == (
        "test 't' metric 'tpd' (type 'propagation_delay'): missing 'signals'. "
        "This metric takes 2 signal inputs: 'sig_in' (kind 'transient'), "
        "'sig_out' (kind 'transient'); bind them with \"signals\": "
        "{\"sig_in\": ..., \"sig_out\": ...}.")


def test_one_input_metric_with_no_binding_is_refused():
    d = _base(analyses=_AC, metrics=[{"name": "g", "type": "dc_gain_dB"}])
    assert _refusal(d) == (
        "test 't' metric 'g' (type 'dc_gain_dB'): missing 'signal'. This "
        "metric takes 1 signal input: 'sig' (kind 'frequency_response'); bind "
        "it with \"signal\": \"<rawfile signal name>\".")


def test_both_signal_and_signals_is_refused():
    d = _base(analyses=_TRAN,
              metrics=[{"name": "tpd", "type": "propagation_delay", "signal": "vin",
                        "signals": {"sig_in": "vin", "sig_out": "vout"}}])
    assert _refusal(d).startswith(
        "test 't' metric 'tpd' (type 'propagation_delay'): has both 'signal' "
        "and 'signals'; pass one.")


def test_signals_must_be_an_object():
    d = _base(analyses=_TRAN,
              metrics=[{"name": "tpd", "type": "propagation_delay",
                        "signals": ["vin", "vout"]}])
    assert _refusal(d) == (
        "test 't' metric 'tpd' (type 'propagation_delay'): 'signals' must be "
        "an object keyed by this metric's signal-input names ('sig_in', "
        "'sig_out'); got list.")


# --- Unproducible kinds (§7.3) ---------------------------------------------
#
# TASK-210 retired `phase_noise_at_offset` and the `phase_noise_sideband`
# kind it was this section's worked example for (no silk analysis produces
# one, so no end-to-end route ever existed). `error_norm` and `p1db` remain
# as the live examples of the rule below.

def test_error_norm_is_refused_for_its_goal_slot():
    """The second slot is a target the DESIGNER authors; two probes wired into
    error_norm would compute the difference between two measurements and label
    it an error against a goal."""
    d = _base(analyses=_TRAN,
              metrics=[{"name": "e", "type": "error_norm",
                        "signals": {"sig": "vout", "goal": "vin"},
                        "params": {"kind": "rms"}}])
    assert _refusal(d) == (
        "test 't' metric 'e' (type 'error_norm'): input 'goal' needs a Signal "
        "of kind 'reference_waveform', which no silk analysis produces — it "
        "must be constructed and passed through add_metric's callable form.")


def test_exactly_the_unproducible_metrics_are_refused():
    """Driven off the live schema, not a hardcoded list: the set refused for
    an unproducible kind must be exactly the set whose signal_inputs name a
    kind with an empty produced_by. A new metric on such a kind (p1db's
    'power_sweep', TASK-184/T-H) joins both sides at once."""
    from silk.signals.schema import export_metric_schema

    schema = export_metric_schema()
    kinds = schema.pop('_meta')['signal_kinds']
    expected = {name for name, e in schema.items()
                if any(not kinds[i['kind']]['produced_by']
                       for i in e['signal_inputs'])}
    assert 'p1db' in expected and 'error_norm' in expected

    refused = set()
    for name, entry in schema.items():
        # Bind every slot and leave the params out: the kind refusal precedes
        # the param and domain checks, so a metric that reaches one of those
        # messages instead is one this rule did not refuse.
        d = _base(metrics=[{"name": "m", "type": name,
                            "signals": {i['name']: 'x' for i in entry['signal_inputs']}}])
        if 'which no silk analysis produces' in _refusal(d):
            refused.add(name)
    assert refused == expected


# --- choices (§7.4) --------------------------------------------------------

def test_param_outside_its_choices_is_refused_by_name():
    d = _base(analyses=_TRAN,
              metrics=[{"name": "tpd", "type": "propagation_delay",
                        "signals": {"sig_in": "vin", "sig_out": "vout"},
                        "params": {"in_edge": "rsing"}}])
    assert _refusal(d) == (
        "test 't' metric 'tpd' (type 'propagation_delay'): param 'in_edge' is "
        "'rsing', which is not one of ['rising', 'falling', 'any']")


def test_every_declared_choice_is_accepted():
    for edge in ('rising', 'falling', 'any'):
        d = _base(analyses=_TRAN,
                  metrics=[{"name": "tpd", "type": "propagation_delay",
                            "signals": {"sig_in": "vin", "sig_out": "vout"},
                            "params": {"in_edge": edge}}])
        _validate_dict(d)   # must not raise


def test_an_optional_choice_param_still_accepts_none():
    """spectral_power's window is Optional[Literal[...]]; None is the
    documented 'no window' value and is not one of the choices."""
    d = _base(analyses=_TRAN,
              metrics=[{"name": "p", "type": "spectral_power", "signal": "vout",
                        "params": {"f1_hz": 1e6, "f2_hz": 2e6, "tclk": 1e-9,
                                   "window": None}}])
    _validate_dict(d)   # must not raise


def test_out_edge_outside_its_choices_is_refused_by_name():
    """TASK-223 AC#2: out_edge is a Literal choice like edge and is refused
    at generate time the same way."""
    d = _base(analyses=_TRAN,
              metrics=[{"name": "tpd", "type": "propagation_delay",
                        "signals": {"sig_in": "vin", "sig_out": "vout"},
                        "params": {"out_edge": "rsing"}}])
    assert _refusal(d) == (
        "test 't' metric 'tpd' (type 'propagation_delay'): param 'out_edge' "
        "is 'rsing', which is not one of ['rising', 'falling', 'any']")


def test_out_edge_none_is_accepted_at_generate_time():
    """out_edge is Optional[Literal[...]]; None ('same as edge') is not one
    of the choices but is still accepted (TASK-223 AC#3)."""
    d = _base(analyses=_TRAN,
              metrics=[{"name": "tpd", "type": "propagation_delay",
                        "signals": {"sig_in": "vin", "sig_out": "vout"},
                        "params": {"in_edge": "rising", "out_edge": None}}])
    _validate_dict(d)   # must not raise


# --- D7: the domain needs a matching analysis ------------------------------

def test_metric_domain_without_a_matching_analysis_is_refused():
    d = _base(analyses=_TRAN,
              metrics=[{"name": "g", "type": "dc_gain_dB", "signal": "vout"}])
    assert _refusal(d) == (
        "test 't' metric 'g' (type 'dc_gain_dB'): domain 'ac' has no matching "
        "analysis in this test; declared: 'transient'. Add an 'ac' analysis, "
        "or measure this metric in a test that has one.")


def test_a_test_with_no_analyses_at_all_is_refused():
    d = _base(metrics=[{"name": "g", "type": "dc_gain_dB", "signal": "vout"}])
    assert "declared: none" in _refusal(d)


def test_the_tran_domain_is_satisfied_by_a_transient_analysis():
    """The one word the two vocabularies disagree on: @metric(domain='tran')
    versus an analysis of kind 'transient'."""
    d = _base(analyses=_TRAN, metrics=[{"name": "r", "type": "rms", "signal": "vout"}])
    _validate_dict(d)   # must not raise
    d = _base(analyses=[{"kind": "tran", "step": 1e-9, "stop": 1e-6}],
              metrics=[{"name": "r", "type": "rms", "signal": "vout"}])
    with pytest.raises(TestbenchSpecError, match="unknown analysis kind 'tran'"):
        _validate_dict(d)   # 'tran' is not analysis vocabulary — named as itself


def test_a_noise_metric_needs_a_noise_analysis():
    d = _base(analyses=_AC,
              metrics=[{"name": "n", "type": "integrated_noise", "signal": "vout",
                        "params": {"f1": 1.0, "f2": 1e6}}])
    assert "domain 'noise' has no matching analysis" in _refusal(d)


def test_one_of_several_analyses_is_enough():
    """D7 asks whether SOME declared analysis can feed the metric. It is
    satisfied here for both metrics; ``rms`` additionally has to say WHICH of
    the two it reads, which is TASK-212's question and not D7's."""
    d = _base(analyses=_TRAN + _AC,
              metrics=[{"name": "g", "type": "dc_gain_dB", "signal": "vout"},
                       {"name": "r", "type": "rms", "signal": "vout",
                        "analysis": "tran"}])
    _validate_dict(d)   # must not raise


# --- D7 v2, kinds-based (TASK-208) ------------------------------------------
#
# The schema publishes signal_kinds['any'].produced_by = (tran, ac, dc,
# noise): a consumer that wires by kind offers the twelve kind='any' metrics
# (average, rms, ... crossing) in an ac-only, dc-only or noise-only test, but
# the OLD domain-only D7 refused all three (domain='tran' names one analysis
# word). The fix reads the same signal_kinds table the schema exports.

def test_an_any_metric_is_accepted_in_a_dc_only_test():
    d = _base(analyses=[{"kind": "dc", "source": "vin"}],
              sources=[{"type": "VoltageSource", "instance": "vin", "positive": "vin", "negative": "gnd", "dc": 0.5}],
              metrics=[{"name": "a", "type": "average", "signal": "vout"}])
    _validate_dict(d)   # must not raise (AC #2)


def test_an_any_metric_is_accepted_in_a_noise_only_test():
    d = _base(analyses=[{"kind": "noise", "source": "vin"}],
              sources=[{"type": "VoltageSource", "instance": "vin", "positive": "vin", "negative": "gnd", "dc": 0.5}],
              metrics=[{"name": "a", "type": "average", "signal": "vout"}])
    _validate_dict(d)   # must not raise (AC #2)


def test_an_any_metric_is_accepted_in_an_ac_only_test():
    d = _base(analyses=_AC,
              metrics=[{"name": "a", "type": "average", "signal": "vout"}])
    _validate_dict(d)   # must not raise -- this was refused before TASK-208


def test_ac_domain_metric_refusal_keeps_the_diagnostic_substring():
    """The diagnostic substring is preserved. The kinds-based check
    still refuses a 'frequency_response'-kind metric in a transient-only
    test, and the message is unchanged even though the criterion moved from
    domain to the kinds table -- for a non-'any' metric the two facts are
    the same."""
    d = _base(analyses=_TRAN,
              metrics=[{"name": "g", "type": "dc_gain_dB", "signal": "vout"}])
    assert "domain 'ac' has no matching analysis in this test" in _refusal(d)


def test_d7_agrees_with_the_exported_schema_for_every_registered_metric():
    """AC #1: the validator cannot disagree with the ``signal_kinds`` table
    the schema publishes under ``_meta.signal_kinds`` -- for EVERY metric
    silk has registered, across every single-analysis scenario (and the
    no-analysis scenario), whether ``_refuse_kind_analysis_mismatch`` raises
    is exactly what an independent read of that published table predicts.
    """
    from silk.codegen.testbench_generator import (
        _refuse_kind_analysis_mismatch, _DOMAIN_ANALYSES,
    )
    from silk.circuit.module_test import _metric_signal_slots
    from silk.signals.schema import export_metric_schema, get_metric

    schema = export_metric_schema()
    kinds_meta = schema['_meta']['signal_kinds']
    scenarios = [set(), {'transient'}, {'ac'}, {'dc'}, {'noise'}]

    def produced_by_analyses(produced_by):
        out = set()
        for pb in produced_by:
            out |= _DOMAIN_ANALYSES.get(pb, {pb})
        return out

    checked = 0
    for name, entry in schema.items():
        if name == '_meta':
            continue
        fn = get_metric(name)
        slots = _metric_signal_slots(fn)
        domain = entry['domain']
        for declared in scenarios:
            expect_refuse = False
            for _slot, kind in slots:
                kmeta = kinds_meta.get(kind)
                if kmeta is None or not kmeta['produced_by']:
                    continue   # not a statement of disjointness (design)
                if not (produced_by_analyses(kmeta['produced_by']) & declared):
                    expect_refuse = True
                    break
            if expect_refuse:
                with pytest.raises(TestbenchSpecError):
                    _refuse_kind_analysis_mismatch('w', domain, slots, declared)
            else:
                _refuse_kind_analysis_mismatch('w', domain, slots, declared)
            checked += 1
    # every registered metric, across all 5 declared-analyses scenarios
    assert checked == (len(schema) - 1) * len(scenarios)


# --- Required-key hardening (TB8 review folded from TB5/TB7 minors) --------
#
# A malformed test/spec/rig/metric dict missing a required key must surface
# as TestbenchSpecError, never a bare KeyError escaping generate_testbenches.

def test_rig_missing_name_raises_testbench_spec_error():
    d = _base()
    d["rigs"] = [{"sources": []}]  # rig dict with no 'name'
    with pytest.raises(TestbenchSpecError):
        generate_testbenches(d)


def test_test_missing_name_raises_testbench_spec_error():
    d = _base()
    del d["tests"][0]["name"]
    with pytest.raises(TestbenchSpecError):
        generate_testbenches(d)


def test_test_missing_dut_portmap_raises_testbench_spec_error():
    d = _base()
    del d["tests"][0]["dut_portmap"]
    with pytest.raises(TestbenchSpecError):
        generate_testbenches(d)


def test_metric_missing_name_raises_testbench_spec_error():
    d = _base(metrics=[{"type": "dc_gain_dB", "signal": "xdut.vout"}])  # no 'name'
    with pytest.raises(TestbenchSpecError):
        generate_testbenches(d)


def test_spec_missing_name_raises_testbench_spec_error():
    d = _base(
        metrics=[{"name": "g", "type": "dc_gain_dB", "signal": "xdut.vout"}],
        specs=[{"metric": "g", "op": ">", "target": 1}],  # no 'name'
    )
    with pytest.raises(TestbenchSpecError):
        generate_testbenches(d)


# --- generator._sim_code: receiver var + `kind` key (Task 6) ---------------

def test_sim_code_receiver_and_kind():
    from silk.codegen.generator import _sim_code
    assert _sim_code({'kind': 'transient', 'step': 1e-9, 'stop': 5e-6}, receiver='tb') \
        .startswith('tb.add_sim_type(analysis.Transient(')
    # circuit path unchanged:
    assert _sim_code({'type': 'Transient', 'step': 1e-9, 'stop': 100e-9}) \
        .startswith('ckt.add_sim_type(')


# --- TASK-321: AC/Noise 'points' is validated + rendered, never bare -------
#
# points used to be interpolated bare (`sim.get('points', 10)` straight into
# the f-string): an SI string like '1k' produced a SyntaxError in the
# generated module, and any string was pasted verbatim into executed code.

from silk.codegen.generator import _sim_code  # noqa: E402


def test_sim_code_points_int_is_byte_identical():
    # AC #3: existing (int) behavior is unchanged, byte for byte.
    assert _sim_code({'type': 'Ac', 'points': 10}) == \
        "ckt.add_sim_type(analysis.AC('dec', 10, 1, 1e+09))"
    assert _sim_code({'type': 'Noise', 'source': 'vin', 'points': 10}) == \
        "ckt.add_sim_type(analysis.Noise('vout', '0', 'vin', 'dec', 10, 1, 1e+09))"


def test_sim_code_points_accepts_si_string():
    # AC #1: an SI string parsing to a positive integer renders as an int
    # literal, for both the Ac and Noise branches.
    assert _sim_code({'type': 'Ac', 'points': '1k'}) == \
        "ckt.add_sim_type(analysis.AC('dec', 1000, 1, 1e+09))"
    assert _sim_code({'type': 'Noise', 'source': 'vin', 'points': '1k'}) == \
        "ckt.add_sim_type(analysis.Noise('vout', '0', 'vin', 'dec', 1000, 1, 1e+09))"


@pytest.mark.parametrize("sim_type", ["Ac", "Noise"])
@pytest.mark.parametrize("bad_points", [
    "10.5",              # non-integer numeric string
    "x); import os #",   # injection attempt
    "",                  # unparseable
    "abc",               # unparseable
    0,                   # not positive
    -3,                  # not positive
    2.5,                 # non-integer number
    "inf",               # non-finite (OverflowError on bare int())
    "1e400",             # non-finite (overflows to inf; same OverflowError)
    "nan",               # non-finite (bare ValueError on int(), no field/value)
])
def test_sim_code_points_refuses_bad_value(sim_type, bad_points):
    # AC #2: refused with an error naming the field; nothing is
    # interpolated into generated code (the call raises before returning
    # any string at all).
    with pytest.raises(ValueError, match=r"points .* must be an int or an SI string"):
        _sim_code({'type': sim_type, 'source': 'vin', 'points': bad_points})


# --- Fail-loud identifier + metric-param validation (final review fixes) ---
#
# generate_testbenches must NEVER emit uncompilable Python. Test names, rig
# names, and circuit_name (when circuit_builder is set) are interpolated
# into def/call identifiers (_attach_<name>, _apply_<rig>_rig,
# build_<circuit_name>_with_tests) — a non-identifier name would emit a
# SyntaxError, which the `except KeyError` backstop does NOT catch. Metric
# params are interpolated as kwargs (`tb.add_metric(..., <key>=<value>)`) —
# an unknown/misspelled param key compiles but raises TypeError at runtime,
# a deferred failure the generate-time contract must prevent.

def test_non_identifier_test_name_raises():
    d = _base()
    d["tests"][0]["name"] = "my test"
    with pytest.raises(TestbenchSpecError, match="my test"):
        generate_testbenches(d)


def test_non_identifier_rig_name_raises():
    d = _base(applied_rigs=["sup-ply"])
    d["rigs"] = [{"name": "sup-ply", "sources": []}]
    with pytest.raises(TestbenchSpecError, match="sup-ply"):
        generate_testbenches(d)


def test_non_identifier_circuit_name_with_builder_raises():
    d = _base()
    d["circuit_name"] = "ac gain"
    assert d["circuit_builder"]  # precondition: builder def would be emitted
    with pytest.raises(TestbenchSpecError, match="ac gain"):
        generate_testbenches(d)


def test_unknown_metric_param_raises():
    d = _base(metrics=[{"name": "g", "type": "dc_gain_dB", "signal": "xdut.vout",
                        "params": {"bogus": 1}}])
    with pytest.raises(TestbenchSpecError, match="bogus"):
        generate_testbenches(d)


def test_non_dict_spec_raises():
    with pytest.raises(TestbenchSpecError):
        generate_testbenches(None)


def test_valid_identifiers_and_params_still_generate():
    """Guard against over-rejection: a spec with valid identifier names and
    only known metric params must still generate cleanly."""
    code = generate_testbenches(AC_EXAMPLE)
    assert "def _attach_ac(mod):" in code
    assert "def build_ac_gain_with_tests(pdk):" in code


# --- Task 7: per-line emit helpers ------------------------------------------

def test_source_add_line():
    from silk.codegen.testbench_generator import _source_add_line
    line = _source_add_line({"type": "VoltageSource", "instance": "vin",
                             "positive": "vin", "negative": "gnd", "dc": 0.5, "ac": 1.0})
    assert line == "tb.add(sources.vdc('vin', positive='vin', negative='gnd', value=0.5, ac=1.0))"


def test_metric_line_with_and_without_params():
    from silk.codegen.testbench_generator import _metric_line
    assert _metric_line({"name": "gain", "type": "dc_gain_dB", "signal": "xdut.vout", "params": {}}) \
        == "tb.add_metric('gain', type='dc_gain_dB', signal='xdut.vout')"
    assert _metric_line({"name": "g", "type": "gain_at_freq", "signal": "xdut.vout",
                         "params": {"freq": 1e6}}) \
        == "tb.add_metric('g', type='gain_at_freq', signal='xdut.vout', freq=1000000.0)"


def test_spec_line():
    from silk.codegen.testbench_generator import _spec_line
    assert _spec_line({"name": "gain > 0.7 V/V", "metric": "gain", "op": ">", "target": 0.7}) \
        == "tb.add_spec('gain > 0.7 V/V', 'gain', '>', 0.7)"


def test_analysis_line():
    from silk.codegen.testbench_generator import _analysis_line
    assert _analysis_line({"kind": "transient", "step": 1e-9, "stop": 5e-6}) \
        .startswith("tb.add_sim_type(analysis.Transient(")


def test_rig_helper():
    from silk.codegen.testbench_generator import _rig_helper
    out = _rig_helper({"name": "supply", "sources": [
        {"type": "VoltageSource", "instance": "vdd", "positive": "vdd", "negative": "gnd", "dc": 1.8}]})
    assert out.startswith("def _apply_supply_rig(tb):")
    assert "tb.add(sources.vdc('vdd'" in out


def test_rig_helper_no_sources_emits_pass():
    from silk.codegen.testbench_generator import _rig_helper
    out = _rig_helper({"name": "empty", "sources": []})
    assert out == "def _apply_empty_rig(tb):\n    pass"


def test_source_add_line_missing_field_raises_testbench_spec_error():
    from silk.codegen.testbench_generator import _source_add_line, TestbenchSpecError
    # missing 'positive'
    with pytest.raises(TestbenchSpecError, match="Source 'vin': missing required field 'positive'"):
        _source_add_line({"type": "VoltageSource", "instance": "vin",
                          "negative": "gnd", "dc": 0.5})


def test_source_add_line_unknown_type_raises_testbench_spec_error():
    from silk.codegen.testbench_generator import _source_add_line, TestbenchSpecError
    with pytest.raises(TestbenchSpecError, match="Source 'vin': .*NotASource"):
        _source_add_line({"type": "NotASource", "instance": "vin",
                          "positive": "vin", "negative": "gnd", "dc": 0.5})


def test_testbench_accepts_every_dependent_and_behavioral_source_class_the_circuit_side_does():
    """No second copy of the controlled/behavioral source-class lists
    (TASK-329 AC#2). Every minimal entry here is built straight from
    generator.py's own tables (_CONTROLLED_SOURCE_PINS/_SCALAR/
    _HAS_CTRL_VNAME, _BEHAVIORAL_SOURCE_TYPES) — the same ones the circuit
    generator dispatches on — so a class the circuit side accepts can never
    silently be refused here without this test failing."""
    from silk.codegen import generator as gen
    from silk.codegen.testbench_generator import _source_add_line

    for canonical, pins in gen._CONTROLLED_SOURCE_PINS.items():
        entry = {"type": canonical, "instance": "x1"}
        for pin in pins:
            entry[pin] = "n1"
        entry[gen._CONTROLLED_SOURCE_SCALAR[canonical]] = 1
        if canonical in gen._CONTROLLED_SOURCE_HAS_CTRL_VNAME:
            entry["ctrl_vname"] = "vseed"
        line = _source_add_line(entry)  # must not raise
        assert line.startswith(f"tb.add(sources.{canonical}(")

    for behavioral in gen._BEHAVIORAL_SOURCE_TYPES:
        entry = {"type": behavioral, "instance": "b1",
                 "positive": "p", "negative": "n", "expression": "0"}
        line = _source_add_line(entry)  # must not raise
        assert line.startswith(f"tb.add(sources.{behavioral}(")


def test_emitted_lines_are_real_executable_silk_code():
    """Sanity-check per the brief: eval the emitted lines against real silk
    (from silk import sources) with a fake tb, not just string equality."""
    from silk import sources
    from silk.codegen.testbench_generator import _source_add_line, _rig_helper

    class FakeTB:
        def __init__(self):
            self.added = []

        def add(self, x):
            self.added.append(x)
            return x

    tb = FakeTB()
    line = _source_add_line({"type": "VoltageSource", "instance": "vin",
                             "positive": "vin", "negative": "gnd", "dc": 0.5, "ac": 1.0})
    eval(line, {"sources": sources, "tb": tb})
    assert len(tb.added) == 1
    assert isinstance(tb.added[0], sources.VoltageSource)

    # the rig helper is a full function definition — exec it, then call it.
    rig_src = _rig_helper({"name": "supply", "sources": [
        {"type": "VoltageSource", "instance": "vdd", "positive": "vdd",
         "negative": "gnd", "dc": 1.8}]})
    ns = {"sources": sources}
    exec(rig_src, ns)
    tb2 = FakeTB()
    ns["_apply_supply_rig"](tb2)
    assert len(tb2.added) == 1
    assert isinstance(tb2.added[0], sources.VoltageSource)


# --- Task 8: generate_testbenches — whole-function assembly -----------------
#
# The two worked examples from the design spec (§4 AC-gain test) and the
# bandgap startup example (bandgap startup, two specs on one
# metric). NOTE: the request doc's bandgap example uses a metric type
# "final_value" that does not exist in silk's registered metric catalog
# (silk.metrics / get_metric — only dc_gain_dB, gain_at_freq,
# bandwidth_3dB, unity_gain_freq, phase_at_freq, phase_margin, gain_margin,
# cmrr_dB, integrated_noise are registered). generate_testbenches correctly
# raises TestbenchSpecError on an unregistered type (§2.1) — that's exactly
# the fail-loud contract working as designed — so this fixture substitutes
# a registered single-signal metric for 'final_value', keeping every other
# structural feature (two specs on one metric, a Pulse waveform source, a
# transient analysis, the builder) intact. The catalog gap itself is a
# separate, real finding (flagged in the task report) — a
# 'final_value'/settled-value metric may be worth adding as a fast-follow.
#
# The substitute used to be 'dc_gain_dB', whose domain is 'ac', in a test
# whose only analysis is transient. TASK-199's D7 refuses exactly that, so
# the metric's semantic fit is no longer irrelevant to a codegen test: the
# substitute is now 'average', a tran-domain metric that is also what a
# settled bandgap reference would actually be measured with.

AC_EXAMPLE = {
    "circuit_name": "ac_gain",
    "circuit_builder": "_build_ac_gain",
    "rigs": [
        {"name": "supply",
         "sources": [{"type": "VoltageSource", "instance": "vdd",
                      "positive": "vdd", "negative": "gnd", "dc": 1.8}]}
    ],
    "tests": [
        {"name": "ac",
         "dut_portmap": {"vout": "vout", "vin": "vin"},
         "applied_rigs": ["supply"],
         "sources": [{"type": "VoltageSource", "instance": "vin",
                      "positive": "vin", "negative": "gnd", "dc": 0.5, "ac": 1.0}],
         "analyses": [{"kind": "ac", "variation": "dec", "points": 10,
                       "fstart": 1, "fstop": 1e9}],
         "metrics": [{"name": "gain", "type": "dc_gain_dB", "signal": "xdut.vout", "params": {}}],
         "specs": [{"name": "gain > 0.7 V/V", "metric": "gain", "op": ">", "target": 0.7}]}
    ]
}

BANDGAP_EXAMPLE = {
    "circuit_name": "bandgap", "circuit_builder": "_build_bandgap",
    "tests": [
        {"name": "startup",
         "dut_portmap": {"vref": "vref", "vdd": "vdd"},
         "sources": [{"type": "VoltageSource", "instance": "vdd",
                      "positive": "vdd", "negative": "gnd",
                      "waveform": {"type": "Pulse", "v1": 0, "v2": 1.8,
                                   "td": 0, "tr": 1e-6, "tf": 0, "pw": 1, "per": 2}}],
         "analyses": [{"kind": "transient", "step": 1e-9, "stop": 5e-6}],
         # 'average' substituted for the request doc's unregistered
         # 'final_value' (see the module comment above).
         "metrics": [{"name": "vref_final", "type": "average", "signal": "xdut.vref", "params": {}}],
         "specs": [
             {"name": "vref > 1.15", "metric": "vref_final", "op": ">", "target": 1.15},
             {"name": "vref < 1.25", "metric": "vref_final", "op": "<", "target": 1.25}
         ]}
    ]
}


def test_generate_ac_example_structure():
    code = generate_testbenches(AC_EXAMPLE)
    assert "def _apply_supply_rig(tb):" in code
    assert "def _attach_ac(mod):" in code
    assert "tb = mod.create_test('ac')" in code
    assert "tb.add_module_under_test({'vout': 'vout', 'vin': 'vin'})" in code
    assert "_apply_supply_rig(tb)" in code
    assert "tb.add(sources.vdc('vin', positive='vin', negative='gnd', value=0.5, ac=1.0))" in code
    assert "tb.add_sim_type(analysis.AC(" in code
    assert "tb.add_metric('gain', type='dc_gain_dB', signal='xdut.vout')" in code
    assert "tb.add_spec('gain > 0.7 V/V', 'gain', '>', 0.7)" in code
    assert "def build_ac_gain_with_tests(pdk):" in code
    assert "from silk.metrics" not in code      # no metric import (§6.3)


def test_generated_code_execs_against_real_silk(pdk, tmp_path):
    """The ownership-seam guard: exec() the generated block against a
    namespace with the real silk API + a stub sidecar builder, then call
    build_ac_gain_with_tests(pdk) and confirm it runs end-to-end against
    real silk (Module/ModuleTestbench/sources/analysis), not just that the
    strings match."""
    code = generate_testbenches(AC_EXAMPLE)
    ns = {}

    # Provide the sidecar builder the spec references. A bare Module with
    # no declared ports/nets can't be wired as a DUT (add_element requires
    # the child's port list to resolve the port_mapping keys), so this
    # stub declares the two ports the AC_EXAMPLE portmap needs — the
    # minimal real Module a generated circuit_builder would return.
    def _build_ac_gain(pdk):
        from silk.circuit.modules import Module
        mod = Module('ac_gain', str(tmp_path), 'ac_gain.cir', pdk_details=pdk)
        mod.define_module_name('ac_gain')
        mod.define_module_nets({'vout': None, 'vin': None})
        return mod

    ns['_build_ac_gain'] = _build_ac_gain
    exec(code, ns)                                  # generated code must import/parse cleanly
    mod = ns['build_ac_gain_with_tests'](pdk)        # and RUN against real silk without error
    assert mod is not None                           # core guard: no exception on build/attach

    # The Module exposes attached tests via _tests / list_tests(); assert
    # the 'ac' test + its metric/spec actually landed on the real Module.
    assert mod.list_tests() == ['ac']
    tb = mod._tests['ac']
    assert tb._module_added is True
    assert [m['name'] for m in tb._metrics] == ['gain']
    assert callable(tb._metrics[0]['extractor'])
    assert [s['name'] for s in tb._specs] == ['gain > 0.7 V/V']
    assert tb._specs[0]['metric'] == 'gain'
    assert tb._specs[0]['op'] == '>'
    assert tb._specs[0]['target'] == 0.7


def test_shipped_ac_gain_example_assembles_and_prepares(tmp_path):
    """The shipped spec wires a legal DUT through to a standalone deck."""
    from pathlib import Path

    from silk.circuit.circuit_elements import PortDir
    from silk.circuit.modules import Module
    from silk.elements import elements
    from silk.examples.testbench_codegen_example import AC_GAIN_SPEC

    def _build_ac_gain(pdk):
        mod = Module('ac_gain', str(tmp_path), 'ac_gain.cir',
                     pdk_details=pdk, module_name='ac_gain')
        mod.define_module_ports({'vout': PortDir.OUT, 'vin': PortDir.IN,
                                 'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
        with mod:
            elements.resistor('rload', p='vout', n='vss', value=1000)
            elements.resistor('rin', p='vin', n='vss', value=1000)
            elements.resistor('rsupply', p='vdd', n='vss', value=1000)
        return mod

    ns = {'_build_ac_gain': _build_ac_gain}
    exec(generate_testbenches(AC_GAIN_SPEC), ns)
    mod = ns['build_ac_gain_with_tests'](None)
    assert mod.list_tests() == ['ac']
    tb = mod._tests['ac']
    tb.netlist.simulator = 'ngspice'
    deck = Path(tb.prepare(ngspice_version='42')).read_text()
    assert '.subckt ac_gain vout vin vdd vss' in deck
    assert 'xdut vout vin vdd gnd ac_gain' in deck
    assert 'vvdd vdd gnd dc 1.8' in deck.lower()
    assert 'vvin vin gnd dc 0.9 ac 1' in deck.lower()
    assert 'ac dec 20 1 1G' in deck
    assert '.lib ' not in deck.lower()
    assert '.include ' not in deck.lower()
    assert [m['name'] for m in tb._metrics] == ['gain_dB']
    assert callable(tb._metrics[0]['extractor'])
    assert [s['name'] for s in tb._specs] == ['DC gain > 20 dB']
    assert tb._specs[0]['metric'] == 'gain_dB'
    assert tb._specs[0]['op'] == '>'
    assert tb._specs[0]['target'] == 20.0


def test_bandgap_two_specs_one_metric():
    code = generate_testbenches(BANDGAP_EXAMPLE)
    assert code.count("tb.add_spec(") == 2
    assert "tb.add(sources.vpulse('vdd'" in code
    assert "def build_bandgap_with_tests(pdk):" in code


# --- Task 8 review: builder omission regression coverage --------------------
#
# TB8 correctly omits the build_<circuit>_with_tests assembler when
# circuit_builder is absent, but every fixture set circuit_builder, leaving
# the omission branch uncovered. These two tests ensure the branch stays
# working: one verifies the builder IS emitted when set, the other verifies
# it is NOT when absent.

def test_builder_omitted_when_no_circuit_builder():
    # circuit_builder absent → no build_<circuit>_with_tests emitted
    spec = {"circuit_name": "c", "tests": [
        {"name": "t", "dut_portmap": {"vout": "vout"}, "applied_rigs": [],
         "sources": [], "analyses": [], "metrics": [], "specs": []}]}
    code = generate_testbenches(spec)
    assert "def build_" not in code
    assert "def _attach_t(mod):" in code          # attach fn still emitted


def test_builder_present_when_circuit_builder_set():
    spec = {"circuit_name": "c", "circuit_builder": "_build_c", "tests": [
        {"name": "t", "dut_portmap": {"vout": "vout"}, "applied_rigs": [],
         "sources": [], "analyses": [], "metrics": [], "specs": []}]}
    code = generate_testbenches(spec)
    assert "def build_c_with_tests(pdk):" in code
    assert "mod = _build_c(pdk)" in code


# --- Task 9: Generator.generate_testbenches delegation ----------------------

def test_generator_delegates():
    from silk.codegen.generator import Generator
    code = Generator().generate_testbenches(AC_EXAMPLE)
    assert "def build_ac_gain_with_tests(pdk):" in code


@pytest.mark.parametrize("existing", [False, True])
def test_generate_testbenches_to_file_preserves_destination_on_rejection(tmp_path, existing):
    from copy import deepcopy
    from silk.codegen.generator import Generator

    generator = Generator()
    path = tmp_path / "testbenches.py"
    if existing:
        generator.generate_testbenches_to_file(AC_EXAMPLE, str(path))
        original = path.read_bytes()
        assert original
    invalid = deepcopy(AC_EXAMPLE)
    invalid["tests"][0]["metrics"][0]["type"] = "dc_gian_dB"
    with pytest.raises(TestbenchSpecError) as direct:
        generator.generate_testbenches(invalid)
    with pytest.raises(TestbenchSpecError) as written:
        generator.generate_testbenches_to_file(invalid, str(path))
    assert type(written.value) is type(direct.value)
    assert str(written.value) == str(direct.value)
    assert "dc_gian_dB" in str(written.value)
    if existing:
        assert path.read_bytes() == original
    else:
        assert not path.exists()


@pytest.mark.parametrize("existing", [False, True])
def test_generate_testbenches_to_file_writes_complete_source(tmp_path, existing):
    from silk.codegen.generator import Generator

    generator = Generator()
    path = tmp_path / "testbenches.py"
    if existing:
        path.write_text("old generated source\n")
    expected = generator.generate_testbenches(AC_EXAMPLE)
    result = generator.generate_testbenches_to_file(AC_EXAMPLE, str(path))
    assert result is None
    assert path.read_text() == expected
    assert path.read_bytes() == expected.encode()


# --- Task 199: a two-signal spec generates AND RUNS -------------------------
#
# Everything above this line
# stops at the generated STRING; this one execs it against real silk, builds
# the testbench and simulates it on real ngspice against sky130, then reads
# back the number the two-signal metric measured. A spec that generates but
# cannot run would satisfy every other test in this file.

INVERTER_TPD_SPEC = {
    "circuit_name": "inverter",
    "circuit_builder": "_build_inverter",
    "rigs": [
        {"name": "supply",
         "sources": [{"type": "VoltageSource", "instance": "vsup",
                      "positive": "vdd", "negative": "0", "dc": 1.8}]}
    ],
    "tests": [
        {"name": "tran",
         "dut_portmap": {"vin": "vin", "vout": "vout", "vdd": "vdd", "vss": "0"},
         "applied_rigs": ["supply"],
         "sources": [{"type": "VoltageSource", "instance": "vin",
                      "positive": "vin", "negative": "0",
                      "waveform": {"type": "Pulse", "v1": 0.0, "v2": 1.8,
                                   "td": 1e-9, "tr": 100e-12, "tf": 100e-12,
                                   "pw": 4e-9, "per": 10e-9}}],
         "analyses": [{"kind": "transient", "step": 10e-12, "stop": 40e-9}],
         # The two-signal metric, bound by slot name — the whole point.
         "metrics": [{"name": "tpd", "type": "propagation_delay",
                      "signals": {"sig_in": "vin", "sig_out": "vout"},
                      "params": {"in_edge": "any"}}],
         "specs": [{"name": "tpd < 100 ps", "metric": "tpd", "op": "<",
                    "target": 100e-12}]}
    ]
}


@pytest.mark.slow
def test_slow_two_signal_spec_generates_and_runs_on_ngspice(tmp_path):
    """Generate → exec → build → simulate → read the measured delay."""
    from tests._env import HAVE_NGSPICE, HAVE_PDK
    if not (HAVE_PDK and HAVE_NGSPICE):
        pytest.skip('needs an installed sky130 PDK and the ngspice binary')

    from silk import load_pdk
    from silk.circuit.circuit_elements import PortDir
    from silk.circuit.modules import Module

    code = generate_testbenches(INVERTER_TPD_SPEC)
    assert ("tb.add_metric('tpd', type='propagation_delay', "
            "signals={'sig_in': 'vin', 'sig_out': 'vout'}, in_edge='any')") in code

    def _build_inverter(pdk):
        mod = Module('PDK_skywater130', str(tmp_path), 'inverter.cir',
                     pdk_details=pdk, module_name='inverter')
        mod.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT,
                                 'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
        with mod:
            pdk.nmos1p8('mn', d='vout', g='vin', s='vss', b='vss', w=1.0, l=0.15)
            pdk.pmos1p8('mp', d='vout', g='vin', s='vdd', b='vdd', w=2.0, l=0.15)
        return mod

    ns = {'_build_inverter': _build_inverter}
    exec(code, ns)
    mod = ns['build_inverter_with_tests'](load_pdk('sky130'))
    report = mod._tests['tran'].run('tt')

    tpd = report.metrics['tpd']
    # A 1x/2x sky130 inverter driving its own drain capacitance: tens of ps,
    # and positive — the same measurement tests/test_add_metric_signals.py
    # makes through add_metric(signals=) directly, here through codegen.
    assert 0 < tpd < 100e-12
    # The spec declared in the same spec dict rode through with it.
    assert [r.name for r in report.results] == ['tpd < 100 ps']
    assert report.passed


# --- TASK-329: dependent (VCVS/VCCS/CCVS/CCCS) and behavioral sources in ----
# rigs[]/tests[] sources[] -----------------------------------------------
#
# Before TASK-329, _source_add_line always did SourceSpec(**src), which
# requires positive/negative — every dependent/behavioral source dict raised
# a TypeError at generation. This spec puts a VCVS in a rig and a CCVS (with
# ctrl_vname), a VCCS and a CCCS in a test that applies that rig, generates
# through the real Generator, execs, and runs on real ngspice — not just
# string-level emission. PDK-free (pdk_details=None, TASK-71's rung), so it
# only needs ngspice, not an installed PDK.
BRANCH_CURRENT_SPEC = {
    "circuit_name": "branch_current",
    "circuit_builder": "_build_branch_current",
    "rigs": [
        {"name": "ctrl",
         "sources": [
             {"type": "VoltageSource", "instance": "vseed",
              "positive": "seed", "negative": "0", "dc": 1.0},
             {"type": "VCVS", "instance": "e1", "out_pos": "mirror",
              "out_neg": "0", "ctrl_pos": "seed", "ctrl_neg": "0", "gain": 2},
         ]}
    ],
    "tests": [
        {"name": "sense",
         "dut_portmap": {"seed": "seed", "mirror": "mirror",
                          "vread": "vread", "voutg": "voutg",
                          "voutf": "voutf", "vss": "0"},
         "applied_rigs": ["ctrl"],
         # CCVS senses vseed's own branch current (a real voltage-source
         # branch, TASK-281); VCCS and CCCS are driven off the same seed.
         "sources": [
             {"type": "CCVS", "instance": "h1", "out_pos": "vread",
              "out_neg": "0", "ctrl_vname": "vseed", "transresistance": 1000},
             {"type": "VCCS", "instance": "g1", "out_pos": "0",
              "out_neg": "voutg", "ctrl_pos": "seed", "ctrl_neg": "0",
              "gm": 1e-3},
             {"type": "CCCS", "instance": "f1", "out_pos": "voutf",
              "out_neg": "0", "ctrl_vname": "vseed", "gain": 1},
         ],
         "analyses": [{"kind": "dc", "source": "vvseed",
                        "start": 1.0, "stop": 1.0, "step": 1.0}],
         "metrics": [],
         "specs": []}
    ]
}


@pytest.mark.slow
def test_slow_dependent_and_behavioral_sources_in_rig_and_test_run_on_ngspice(tmp_path):
    """Generate -> exec -> build -> simulate -> read back real node voltages
    for a VCVS (rig), and a CCVS/VCCS/CCCS (test) applying that rig."""
    from tests._env import HAVE_NGSPICE
    if not HAVE_NGSPICE:
        pytest.skip('needs the ngspice binary')

    from silk.circuit.circuit_elements import PortDir
    from silk.circuit.modules import Module
    from silk.elements import elements

    code = generate_testbenches(BRANCH_CURRENT_SPEC)
    # Every dependent/behavioral kind this task adds emits a direct
    # sources.<ClassName>(...) call inside tb.add(...) — see the AC#1 lines
    # recorded in the task notes; independent vseed is unchanged (still
    # sources.vdc(...)).
    assert ("tb.add(sources.VoltageControlledVoltageSource("
            "'e1', out_pos='mirror', out_neg='0', ctrl_pos='seed', "
            "ctrl_neg='0', gain=2))") in code
    assert ("tb.add(sources.CurrentControlledVoltageSource("
            "'h1', out_pos='vread', out_neg='0', ctrl_vname='vseed', "
            "transresistance=1000))") in code
    assert ("tb.add(sources.VoltageControlledCurrentSource("
            "'g1', out_pos='0', out_neg='voutg', ctrl_pos='seed', "
            "ctrl_neg='0', gm=0.001))") in code
    assert ("tb.add(sources.CurrentControlledCurrentSource("
            "'f1', out_pos='voutf', out_neg='0', ctrl_vname='vseed', "
            "gain=1))") in code
    assert "tb.add(sources.vdc('vseed'" in code

    def _build_branch_current(pdk):
        mod = Module('branch_current', str(tmp_path), 'branch_current.cir',
                     pdk_details=None, module_name='branch_current')
        mod.define_module_ports({
            'seed': PortDir.INOUT, 'mirror': PortDir.INOUT,
            'vread': PortDir.INOUT, 'voutg': PortDir.INOUT,
            'voutf': PortDir.INOUT, 'vss': PortDir.GROUND,
        })
        with mod:
            # Load every dependent source's output node so none floats;
            # rseed also gives vseed's branch a real current for ctrl_vname
            # to sense (a bare control pin draws none).
            elements.resistor('rseed', p='seed', n='vss', value=1000)
            elements.resistor('rmirror', p='mirror', n='vss', value=1000)
            elements.resistor('rread', p='vread', n='vss', value=1000)
            elements.resistor('rg', p='voutg', n='vss', value=1000)
            elements.resistor('rf', p='voutf', n='vss', value=1000)
        return mod

    ns = {'_build_branch_current': _build_branch_current}
    exec(code, ns)
    mod = ns['build_branch_current_with_tests'](None)
    report = mod._tests['sense'].run('tt')
    assert report.passed

    test = mod._tests['sense']
    # 1 kOhm * 1 mA everywhere; signs are SPICE's own branch-current
    # convention (H senses vseed's current one way, F the other), not a bug
    # — measured against real ngspice, not derived by hand.
    assert test.netlist.load_signal('v(seed)').y == pytest.approx([1.0])
    assert test.netlist.load_signal('v(mirror)').y == pytest.approx([2.0])   # VCVS gain=2
    assert test.netlist.load_signal('v(vread)').y == pytest.approx([-1.0])  # CCVS
    assert test.netlist.load_signal('v(voutg)').y == pytest.approx([1.0])   # VCCS
    assert test.netlist.load_signal('v(voutf)').y == pytest.approx([1.0])   # CCCS


# --- TASK-212: which analysis a metric reads ---------------------------------
#
# Layer 1 of three that share one wording. The rule lives in
# ``module_test._resolve_metric_analysis``; these pin what a SPEC sees.

def test_an_unambiguous_metric_needs_no_analysis_field():
    """Only .ac produces a frequency_response, so a tran+ac test settles
    dc_gain_dB by itself — asking the author would be noise."""
    d = _base(analyses=_TRAN + _AC,
              metrics=[{"name": "g", "type": "dc_gain_dB", "signal": "vout"}])
    _validate_dict(d)   # must not raise


def test_an_ambiguous_metric_is_refused_naming_the_candidates():
    d = _base(analyses=_TRAN + _AC,
              metrics=[{"name": "vavg", "type": "average", "signal": "vout"}])
    msg = _refusal(d)
    assert "'vout' could come from more than one analysis in this test" in msg
    assert "'ac', 'tran'" in msg
    assert "Add analysis='ac' or analysis='tran' to the metric." in msg


def test_an_analysis_field_settles_it():
    d = _base(analyses=_TRAN + _AC,
              metrics=[{"name": "vavg", "type": "average", "signal": "vout",
                        "analysis": "tran"}])
    _validate_dict(d)   # must not raise


def test_an_undeclared_analysis_is_refused_in_deck_order():
    """The analyses are named in the order the spec wrote them, which is the
    order the run-time layer reads off sim_analysis_dict — one wording, from
    whichever layer catches it first."""
    d = _base(analyses=_TRAN + _AC,
              metrics=[{"name": "vavg", "type": "average", "signal": "vout",
                        "analysis": "noise"}])
    assert "this test declares 'tran', 'ac'" in _refusal(d)


def test_an_analysis_the_kinds_cannot_come_from_is_refused():
    d = _base(analyses=_TRAN + _AC,
              metrics=[{"name": "g", "type": "dc_gain_dB", "signal": "vout",
                        "analysis": "tran"}])
    assert "cannot feed this metric's input" in _refusal(d)


def test_an_undeclared_analysis_is_refused_in_a_one_analysis_test_too():
    """A spec that names an analysis its test does not declare is a spec bug
    whether or not the choice would have mattered."""
    d = _base(analyses=_TRAN,
              metrics=[{"name": "vavg", "type": "average", "signal": "vout",
                        "analysis": "ac"}])
    assert "but this test declares 'tran'" in _refusal(d)


def test_a_one_analysis_test_emits_the_line_it_always_did():
    """No analysis field, no emitted keyword — the 0.1.x line, byte for byte."""
    d = _base(analyses=_TRAN,
              metrics=[{"name": "vavg", "type": "average", "signal": "vout"}])
    code = generate_testbenches(TestbenchSpec.from_dict(d))
    assert "tb.add_metric('vavg', type='average', signal='vout')" in code


def test_the_analysis_field_emits_as_its_own_keyword_before_the_params():
    d = _base(analyses=_TRAN + _AC,
              metrics=[{"name": "v", "type": "value_at", "signal": "vout",
                        "analysis": "tran", "params": {"x": 1e-6}}])
    code = generate_testbenches(TestbenchSpec.from_dict(d))
    assert ("tb.add_metric('v', type='value_at', signal='vout', "
            "analysis='tran', x=1e-06)") in code


@pytest.mark.parametrize("owner", ["test", "rig"])
@pytest.mark.parametrize("change, identity, reason", [
    ({"waveform": {"type": "pulse", "params": {"v1": 0}}},
     "Source 'V1'", "waveform: missing a required argument: 'v2'"),
    ({"negative": None}, "Source 'V1'", "missing required field 'negative'"),
    ({"instance": None}, "source entry 2", "missing required field 'instance'"),
    ({"waveform": {"type": 3}}, "Source 'V1'",
     "waveform must be a nonempty dict with a type."),
    ({"waveform": {"type": "pulse", "params": {"v1": 0}, "v2": 1}},
     "Source 'V1'", "waveform dict mixes params and flat keywords; use one form."),
])
def test_bad_source_names_owner_and_entry(owner, change, identity, reason):
    source = {"type": "VoltageSource", "instance": "V1",
              "positive": "vd", "negative": "0"}
    for key, value in change.items():
        if value is None:
            del source[key]
        else:
            source[key] = value
    entries = [{"type": "VoltageSource", "instance": "good",
                "positive": "vdd", "negative": "0"}, source]
    d = _base()
    if owner == "rig":
        d["rigs"] = [{"name": "supply", "sources": entries}]
        d["tests"][0]["applied_rigs"] = ["supply"]
        name = "supply"
    else:
        d["rigs"] = [{"name": "supply", "sources": [
            {"type": "VoltageSource", "instance": "supply", "positive": "vs", "negative": "0"}]}]
        d["tests"][0]["applied_rigs"] = ["supply"]
        d["tests"][0]["sources"] = entries
        name = "t"
    with pytest.raises(TestbenchSpecError) as exc:
        generate_testbenches(d)
    assert str(exc.value) == f"{owner} {name!r}: {identity}: {reason}"
    assert "{" not in str(exc.value)
    assert "invalid testbench source" not in str(exc.value)
