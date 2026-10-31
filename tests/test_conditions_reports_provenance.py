"""TASK-410: reports and provenance show none and unknown."""
import json
from types import SimpleNamespace

import pytest

from silk.circuit.report import (
    _build_auto_context_fields, _measurement_from_dict, _provenance_from_list)
from silk.circuit.op_table import RegionThresholds
from silk.circuit.spec_report import SpecReport
from silk.simulation.provenance import FIELDS, RunProvenance, SCHEME, build
from silk.simulation.conditions import ConditionState, OperatingConditions
from silk.specs import Measurement
from silk.specs.report import SpecOutcome
from silk.specs import Spec


def _summary(**kw):
    kw.setdefault('corner', None)
    c = OperatingConditions('silk', 'ngspice', **kw)
    return SpecReport('tb', 'm', {}, operating_conditions=c).summary()


def test_summary_spells_out_none_with_reason_and_unknown():
    text = _summary(corner_kind='none', corner_basis='proven-no-pdk-device',
                    supply_kind='none', temperature_c=27.0,
                    temperature_kind='fixed', temperature_basis='configured')
    assert 'primary supply declared' not in text
    assert 'corner none (proven-no-pdk-device)' in text
    assert 'supply none' in text
    assert 'TEMP 27 C (configured)' in text
    unknown = _summary()
    assert 'corner unknown' in unknown and 'supply unknown' in unknown
    assert 'None V' not in unknown


def test_summary_prints_fixed_values():
    text = _summary(corner_kind='fixed', corner='tt', supply_voltage_v=1.8,
                    supply_kind='fixed')
    assert 'corner tt' in text and 'supply 1.8 V' in text


def _outcome(**m):
    spec = Spec(name='s', module=None, metric='g', op='>', target=1.0)
    meas = Measurement(source='spice', value=1.0, **m)
    return SpecOutcome(spec=spec, measurement=meas, status='pass')


@pytest.mark.parametrize('kw, wire', [
    (dict(corner=ConditionState.NONE, temperature=ConditionState.NONE,
          voltage=ConditionState.NONE), {'state': 'none'}),
    (dict(temperature=ConditionState.SWEEP), None),
])
def test_report_json_round_trips_tagged_states(kw, wire):
    d = json.loads(json.dumps(_outcome(**kw).as_dict()))['measurement']
    back = _measurement_from_dict(d)
    assert back.corner == kw.get('corner')
    assert back.temperature == kw.get('temperature')
    assert back.voltage == kw.get('voltage')
    if 'corner' in kw:
        assert d['corner'] == {'state': 'none'}
    else:
        assert d['temperature'] == {'state': 'sweep'}
        assert d['corner'] is None


def test_report_json_values_null_and_old_files():
    d = _outcome(corner='tt', temperature=27.0, voltage=1.8).as_dict()['measurement']
    assert (d['corner'], d['temperature'], d['voltage']) == ('tt', 27.0, 1.8)
    back = _measurement_from_dict(d)
    assert (back.corner, back.temperature, back.voltage) == ('tt', 27.0, 1.8)
    unknown = _measurement_from_dict(_outcome().as_dict()['measurement'])
    assert unknown.corner is None and unknown.temperature is None
    old = {'source': 'spice', 'value': 1.0, 'corner': 'ff'}
    assert _measurement_from_dict(old).corner == 'ff'


def test_provenance_v3_carries_kinds_and_v1_stays_readable():
    c = OperatingConditions('silk', 'ngspice', None, corner_kind='none',
                            supply_kind='none')
    rec = build(run_origin='silk-launch', launch=None, prepared=None,
                pdk_name=None, conditions=c)
    d = rec.to_dict()
    assert SCHEME == 'silk.provenance/v3' == d['scheme']
    assert d['corner_kind'] == 'none' and d['supply_kind'] == 'none'
    assert 'corner_kind' in FIELDS and 'supply_kind' in FIELDS
    v1 = {k: v for k, v in d.items() if k not in ('corner_kind', 'supply_kind')}
    v1['scheme'] = 'silk.provenance/v1'
    old = _provenance_from_list([v1])[0]
    assert old.scheme == 'silk.provenance/v1'
    assert old.corner_kind is None and old.supply_kind is None
    assert old.digest() != rec.digest()


def test_waveform_context_prints_none_and_omits_unknown():
    def fields(corner):
        src = SimpleNamespace(_last_sim_corner=corner)
        return {f['label']: f['value'] for f in _build_auto_context_fields(src, 'm')}
    assert fields(ConditionState.NONE)['corner'] == 'none'
    assert 'corner' not in fields(None)
    assert fields('tt')['corner'] == 'tt'


def test_op_table_refuses_none_temperature():
    c = OperatingConditions('behavioral', None, None, temperature_kind='none')
    with pytest.raises(ValueError, match='region_thresholds='):
        RegionThresholds.at_temperature(c)
    RegionThresholds.at_temperature(None)
    RegionThresholds.at_temperature(OperatingConditions('silk', 'ngspice', None))


def test_spec_report_op_table_none_temperature_raises_and_override_bypasses():
    c = OperatingConditions('behavioral', None, None, temperature_kind='none')
    report = SpecReport('tb', 'm', {}, operating_conditions=c, op_devices=[])
    with pytest.raises(ValueError, match='region_thresholds='):
        report.op_table()
    assert report.op_table(region_thresholds=RegionThresholds()) is not None


def test_supply_sweep_summary_and_optional_codec_field_preserve_old_bytes():
    from silk.simulation import ConditionSweep
    from silk.circuit._spec_report_json import _conditions
    old = OperatingConditions('silk', 'ngspice', None).to_dict()
    assert 'supply_sweep' not in old
    assert _conditions(old).to_dict() == old
    c = OperatingConditions('silk', 'ngspice', None, supply_kind='sweep',
                            supply_basis='configured', supply_source='rail',
                            supply_sweep=ConditionSweep(0, 2, 0.5))
    assert c.to_dict()['supply_sweep'] == {'start': 0, 'stop': 2, 'step': 0.5}
    assert _conditions(c.to_dict()) == c
    assert 'supply sweep' in SpecReport('tb', 'm', {}, operating_conditions=c).summary()
