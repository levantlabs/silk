"""TASK-487: readable test output folders (test_<module>~<test>), no aliasing."""
import hashlib
import os

import pytest

from silk import Module, load_pdk
from silk.circuit.module_test import ModuleTestbench
from silk.exceptions import NotSupportedError


def _tb(root, module_name, test_name):
    mod = Module('sky130', str(root), 'dut.cir', load_pdk('sky130'),
                 module_name=module_name)
    return ModuleTestbench(mod, test_name)


def _dir(root, module_name, test_name, **kw):
    tb = _tb(root, module_name, test_name)
    tb_dir, deck = tb._output_paths(**kw)
    return os.path.relpath(tb_dir, str(root)), deck


def test_underscores_are_literal(tmp_path):
    folder, deck = _dir(tmp_path, 'comparator_sa', 'decision_time_od0005mV')
    assert folder == 'test_comparator_sa~decision_time_od0005mV'
    assert deck == folder + '.cir'
    assert '%5F' not in folder


def test_a_b_c_pair_does_not_alias(tmp_path):
    assert _dir(tmp_path, 'a_b', 'c')[0] != _dir(tmp_path, 'a', 'b_c')[0]


def test_corpus_is_injective(tmp_path):
    names = ['a', 'a_b', 'b', 'b_c', 'c', 'a~b', 'a%7Eb', 'a%5Fb', 'x~owner~y',
             'p~run~q', 'None', '%N', 'a/b', 'a%2Fb', 'é', 'z' * 300]
    seen = {}
    for m in names + [None]:
        for t in names:
            try:
                folder = _dir(tmp_path, m, t)[0]
            except ValueError:      # module_name refused at construction
                break
            assert folder not in seen, (m, t, seen[folder])
            seen[folder] = (m, t)
    assert _dir(tmp_path, None, 'x')[0] == 'test_%N~x'
    assert _dir(tmp_path, 'None', 'x')[0] == 'test_None~x'


def test_run_label_suffix_and_no_alias(tmp_path):
    base = _dir(tmp_path, 'a', 'b')[0]
    labelled = _dir(tmp_path, 'a', 'b', run_label='c')[0]
    assert labelled == base + '~run~c'
    assert _dir(tmp_path, 'a', 'b~run~c')[0] not in (base, labelled)


def test_long_component_digest_fallback(tmp_path):
    folder, deck = _dir(tmp_path, 'a', 'z' * 300)
    full = 'test_a~' + 'z' * 300
    assert folder == 'test~' + hashlib.sha256(full.encode()).hexdigest()
    assert deck.startswith('test~')


def test_corner_subdir_keeps_depth(tmp_path):
    folder, _ = _dir(tmp_path, 'a', 'b_c', corner_subdir=('ss', 'mc_0001'))
    assert folder == os.path.join('test_a~b_c', 'ss', 'mc_0001')


def test_layout_parameter(tmp_path):
    mod = _tb(tmp_path, 'a', 'b').module
    mod.set_output_options(mode='new')
    with pytest.raises(NotSupportedError, match='results_file'):
        mod.set_output_options(mode='overwrite', layout='nested')
    with pytest.raises(ValueError, match="flat"):
        mod.set_output_options(mode='overwrite', layout='bogus')
    with pytest.raises(ValueError):
        mod.set_output_options(mode='bogus', layout='flat')
    assert mod._output_mode == 'new'
    mod.set_output_options(mode='overwrite', layout='flat')
    assert mod._output_mode == 'overwrite'
