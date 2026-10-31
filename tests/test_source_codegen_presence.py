"""Omitted source waveforms stay distinct from malformed authored fields."""
import pytest
from silk import sources
from silk.codegen import generator as g
from silk.codegen.testbench_generator import _source_add_line

BASE = dict(type='VoltageSource', instance='stim', positive='out', negative='0')
SINE = {'type': 'Sine', 'vampl': .3, 'freq': 1000}


@pytest.mark.parametrize('path', ['direct', 'json', 'array', 'testbench'])
@pytest.mark.parametrize('fields,kind', [({}, 'constant'), ({'waveform': SINE}, 'wave'),
                                      ({'pwl': [0, .5, 1, 1]}, 'wave')])
def test_omission_and_each_single_waveform_generate(path, fields, kind):
    spec = g.SourceSpec(**BASE, **fields)
    if path == 'json':
        spec = g._source_entry_to_spec(dict(BASE, **fields), 'source')
    if path == 'testbench':
        text = _source_add_line(dict(BASE, **fields))
        assert 'tb.add(' in text
        return
    if path == 'array':
        spec.array = 2
        objects = eval(g._source_array_call(spec), {'sources': sources})
    else:
        objects = [eval(g._source_call(spec), {'sources': sources})]
    assert all(len(obj.to_spice().splitlines()) == (1 if kind == 'constant' else 2)
               for obj in objects)
    if not fields:
        assert spec.waveform is None and spec.pwl is None


@pytest.mark.parametrize('path', ['direct', 'json', 'array', 'testbench'])
@pytest.mark.parametrize('fields,field', [
    ({'waveform': {}, 'pwl': [0, 1, 1, 2]}, 'waveform'),
    ({'waveform': SINE, 'pwl': []}, 'pwl'),
    ({'waveform': {}}, 'waveform'), ({'waveform': []}, 'waveform'),
    ({'waveform': {'type': 'Sine', 'params': []}}, 'waveform'),
    ({'waveform': {'type': 'Sine', 'params': {}, 'vampl': 1}}, 'waveform'),
    ({'waveform': {'type': 'Sine', 'vampl': 1}}, 'waveform'),
    ({'waveform': {'type': 'UnknownWave'}}, 'waveform'),
    ({'pwl': []}, 'pwl'), ({'pwl': 'bad'}, 'pwl'), ({'pwl': [0, 1, 2]}, 'pwl'),
])
def test_supplied_conflict_or_malformed_field_refuses_before_generation(path, fields, field):
    with pytest.raises(ValueError, match=f'stim.*{field}'):
        if path == 'testbench':
            _source_add_line(dict(BASE, **fields))
        else:
            spec = (g._source_entry_to_spec(dict(BASE, **fields), 'source')
                    if path == 'json' else g.SourceSpec(**BASE, **fields))
            if path == 'array':
                spec.array = 2
                g._source_array_call(spec)
            else:
                g._source_call(spec)


@pytest.mark.parametrize('naming', ['bracketed', 'underscored'])
def test_generated_source_arrays_use_shared_instance_reservation(naming):
    from silk.exceptions import CircuitError
    spec = g.SourceSpec(**dict(BASE, instance='stim__SILK__silk_dc'), array=2, naming=naming)
    with pytest.raises(CircuitError, match='reserved __silk_ prefix'):
        eval(g._source_array_call(spec), {'sources': sources})


@pytest.mark.parametrize('naming,want', [('bracketed', 'Vstim<0>__silk_wave'),
                                      ('underscored', 'Vstim_0__silk_wave')])
def test_generated_source_arrays_use_reserved_card_layout(naming, want):
    spec = g.SourceSpec(**BASE, array=2, naming=naming, waveform=SINE)
    objects = eval(g._source_array_call(spec), {'sources': sources})
    assert objects[0].to_spice().split()[0] == want
