"""Module.declare_spec, reconcile_specs, and calibrate_behavioral."""
import os
import pytest

from silk.circuit.modules import Module
from silk.exceptions import DuplicateNameError
from silk.pdk.basepdk import BasePDK
from silk.specs import Spec, Measurement


_PDK_YAML = os.path.join(
    os.path.dirname(__file__), '..', '..', 'silk', 'pdk', 'data',
    'skywater_130_device_details.yaml'
)


def _pdk():
    return BasePDK(_PDK_YAML, library_location='/fake/lib')


def _module(tmp_path, name='m'):
    return Module('PDK_skywater130', str(tmp_path), f'{name}.cir',
                  pdk_details=_pdk(), module_name=name)


def test_declare_spec_returns_spec_object(tmp_path):
    top = _module(tmp_path, 'top')
    s = top.declare_spec(
        name='system_NF',
        metric='noise_figure',
        op='<=',
        target=5.0,
        units='dB',
    )
    assert isinstance(s, Spec)
    assert s.name == 'system_NF'
    assert s.module is top
    assert s.metric == 'noise_figure'
    assert s.target == 5.0


def test_declare_spec_added_to_module_specs_list(tmp_path):
    top = _module(tmp_path, 'top')
    assert top._specs == []
    s = top.declare_spec(name='system_NF', metric='noise_figure',
                         op='<=', target=5.0)
    assert top._specs == [s]


def test_declare_spec_accepts_explicit_stable_key(tmp_path):
    top = _module(tmp_path, 'top')

    s = top.declare_spec(
        name='Readable gain target', metric='gain', op='>=', target=60.0,
        key='dc_gain',
    )

    assert s.name == 'Readable gain target'
    assert s.key == 'dc_gain'


def test_declare_spec_rejects_duplicate_local_key_before_appending(tmp_path):
    top = _module(tmp_path, 'top')
    first = top.declare_spec(
        name='First display name', metric='gain', op='>=', target=60.0,
        key='dc_gain',
    )

    with pytest.raises(DuplicateNameError, match='dc_gain'):
        top.declare_spec(
            name='Renamed display', metric='gain', op='>=', target=60.0,
            key='dc_gain',
        )

    assert top._specs == [first]


def test_declare_spec_allows_same_display_name_with_distinct_keys(tmp_path):
    top = _module(tmp_path, 'top')

    first = top.declare_spec(
        name='Gain', metric='gain', op='>=', target=60.0, key='dc_gain',
    )
    second = top.declare_spec(
        name='Gain', metric='gain', op='>=', target=55.0, key='ac_gain',
    )

    assert [spec.key for spec in top._specs] == ['dc_gain', 'ac_gain']
    assert top._specs == [first, second]


def test_declare_spec_with_explicit_parent(tmp_path):
    top = _module(tmp_path, 'top')
    lna = _module(tmp_path, 'lna')
    parent = top.declare_spec(name='system_NF', metric='noise_figure',
                              op='<=', target=5.0)
    child = lna.declare_spec(name='NF', metric='noise_figure',
                             op='<=', target=2.0, parent=parent)
    assert child.parent is parent
    assert child in parent.children
    assert lna._specs == [child]
    assert top._specs == [parent]


def test_declare_spec_captures_declaration_site(tmp_path):
    top = _module(tmp_path, 'top')
    s = top.declare_spec(name='system_NF', metric='noise_figure',
                         op='<=', target=5.0)
    assert 'test_module_declare_spec.py' in s.declared_at


from silk.specs import SpecReconciliationReport, CalibrationBinding


def test_reconcile_specs_returns_empty_report_when_no_specs(tmp_path):
    top = _module(tmp_path, 'top')
    r = top.reconcile_specs()
    assert isinstance(r, SpecReconciliationReport)
    assert r.outcomes == []
    assert r.module is top


def test_reconcile_specs_evaluates_local_specs(tmp_path):
    top = _module(tmp_path, 'top')
    s = top.declare_spec(name='system_NF', metric='noise_figure',
                         op='<=', target=5.0, units='dB')
    s.record_measurement(Measurement(source='spice', value=4.8, timestamp=100.0))
    r = top.reconcile_specs()
    assert len(r.outcomes) == 1
    assert r.outcomes[0].status == 'PASS'
    assert r.outcomes[0].measurement.value == 4.8


def test_reconcile_specs_walks_submodule_specs(tmp_path):
    """Specs declared on submodules are included when reconciling the top."""
    top = _module(tmp_path, 'top')
    lna = _module(tmp_path, 'lna')
    top.add_module(lna, 'lna1', nets={})
    # add_element deepcopies — declare specs on the wired instance, not the original.
    # elem_dict is keyed by instance name (e.g. 'lna1'), so multiple instances
    # of the same type remain distinct.
    wired_lna = top._element_mgr.elem_dict['lna1']['device']
    sys_nf = top.declare_spec(name='system_NF', metric='noise_figure',
                              op='<=', target=5.0)
    lna_nf = wired_lna.declare_spec(name='NF', metric='noise_figure',
                                    op='<=', target=2.0, parent=sys_nf)
    sys_nf.record_measurement(Measurement(source='behavioral', value=4.9, timestamp=100.0))
    lna_nf.record_measurement(Measurement(source='spice', value=2.3, timestamp=200.0))
    r = top.reconcile_specs()
    names = [o.spec.name for o in r.outcomes]
    assert 'system_NF' in names
    assert 'NF' in names
    # Outcome order is pre-order from the root module: top first, then submodules.
    assert names[0] == 'system_NF'


def test_reconcile_specs_walks_multiple_instances(tmp_path):
    """Multiple instances of the same submodule type each carry their own specs.

    Regression test for an earlier bug where _iter_all_specs walked
    module_dict (keyed by type) instead of elem_dict (keyed by
    instance), causing all but one instance's specs to be lost.
    """
    top = _module(tmp_path, 'top')
    block = _module(tmp_path, 'blk')
    top.add_module(block, 'blk1', nets={})
    top.add_module(_module(tmp_path, 'blk'), 'blk2', nets={})
    top.add_module(_module(tmp_path, 'blk'), 'blk3', nets={})

    # Declare distinct specs on each wired instance
    for k in ('blk1', 'blk2', 'blk3'):
        wired = top._element_mgr.elem_dict[k]['device']
        wired.declare_spec(name=f'{k}_gain', metric='gain', op='>',
                           target=10.0)

    r = top.reconcile_specs()
    names = sorted(o.spec.name for o in r.outcomes)
    assert names == ['blk1_gain', 'blk2_gain', 'blk3_gain'], f'got {names}'


def test_reconcile_specs_prefer_source_propagates(tmp_path):
    top = _module(tmp_path, 'top')
    s = top.declare_spec(name='system_NF', metric='noise_figure',
                         op='<=', target=5.0)
    s.record_measurement(Measurement(source='behavioral', value=4.8, timestamp=100.0))
    s.record_measurement(Measurement(source='spice', value=5.3, timestamp=200.0))
    r = top.reconcile_specs(prefer_source='behavioral')
    assert r.outcomes[0].source_used == 'behavioral'
    assert r.outcomes[0].status == 'PASS'


def test_calibrate_behavioral_writes_to_params(tmp_path):
    m = _module(tmp_path, 'lna')
    def model_1(ports, t, p):
        return
    m.set_behavioral_model(model_1, params={'NF': 2.0, 'gain': 15.0}, execution='vector')
    m.calibrate_behavioral('NF', 2.3)
    assert m._behavioral_params['NF'] == 2.3
    assert m._behavioral_params['gain'] == 15.0  # unchanged


def test_calibrate_behavioral_raises_when_params_dict_missing(tmp_path):
    m = _module(tmp_path, 'lna')
    # set_behavioral_model not called, so _behavioral_params is None
    with pytest.raises(RuntimeError, match='not parameterized'):
        m.calibrate_behavioral('NF', 2.3)


def test_calibrate_behavioral_raises_when_param_name_missing(tmp_path):
    m = _module(tmp_path, 'lna')
    def model_2(ports, t, p):
        return
    m.set_behavioral_model(model_2, params={'gain': 15.0}, execution='vector')
    with pytest.raises(KeyError, match='NF'):
        m.calibrate_behavioral('NF', 2.3)
