"""Build coverage and real, independently reread output-lifecycle evidence."""

from pathlib import Path

import numpy as np
import pytest

from silk import Module, load_pdk
from silk.examples import output_modes_example as example
from tests._env import needs_ngspice


def test_load_embeds_characterization_without_an_installed_pdk(tmp_path, monkeypatch):
    monkeypatch.setenv('PDK_ROOT', str(tmp_path / 'missing_pdk'))
    mod = example.build_load(output_root=tmp_path / 'build', resistance=2200,
                             input_voltage=1.25)
    assert mod.list_tests() == ['level']
    assert set(mod.pin_list) == {'out', 'vss'}
    assert len(mod.elem_dict) == 1
    subckt = mod.generate_netlist()
    assert '.subckt load out vss' in subckt
    assert '2.2k' in subckt
    assert '.ends' in subckt
    recon = mod.reconcile_specs()
    assert len(recon.outcomes) == 1
    assert recon.outcomes[0].spec.metric == 'voltage'
    assert recon.outcomes[0].spec.op == '>'
    assert recon.outcomes[0].spec.target == 0.0
    assert recon.outcomes[0].status == 'UNMEASURED'
    # TASK-243: the default output root is resolved by
    # silk.examples.example_output_dir() — under the working directory, or
    # $SILK_EXAMPLES_OUT — never beside the example's own source, which for a
    # pip install is inside site-packages.
    assert example.DEFAULT_OUTPUT_ROOT.name == 'output_modes'
    package_dir = Path(example.__file__).resolve().parent
    assert package_dir not in example.DEFAULT_OUTPUT_ROOT.parents
    assert example.DEFAULT_OUTPUT_ROOT != package_dir


def _read_fresh(path, reader_root):
    # A different public loading route from the example, on a never-run Module.
    reader = Module('sky130', str(reader_root), 'reader.cir',
                    pdk_details=load_pdk('sky130'))
    reader.load_file(str(path))
    reader.read_raw_file()
    return reader.load_signal('out')


@pytest.mark.slow
@needs_ngspice
def test_real_example_retains_readable_runs_across_repeats(tmp_path, monkeypatch,
                                                         capsys):
    monkeypatch.setenv('PDK_ROOT', str(tmp_path / 'missing_pdk'))
    output_root = tmp_path / 'outputs'
    output_root.mkdir()
    unrelated = output_root / 'notes.txt'
    unrelated.write_text('Keep this user file.\n')

    first = example.run_example(output_root=output_root)
    output = capsys.readouterr().out
    assert 'Fresh disk read through the first path: 2 V' in output
    assert 'First scalar report: 1 V; its live Module: 2 V' in output
    assert 'fresh disk reads: 3 V and 4 V' in output
    assert 'original path now reads 5 V' in output
    saved = {first[key]: first[key].read_bytes()
             for key in ('new_first', 'new_second')}

    # Running the whole teaching flow again must keep both previous children.
    second = example.run_example(output_root=output_root)
    assert first['restored'] == second['restored']
    retained_paths = {run[key] for run in (first, second)
                      for key in ('new_first', 'new_second')}
    assert len(retained_paths) == 4
    for path, before in saved.items():
        assert path.read_bytes() == before
    assert unrelated.read_text() == 'Keep this user file.\n'

    for run in (first, second):
        base = run['restored'].parent
        assert run['overwrite_first'] == run['overwrite_second'] == run['restored']
        assert run['new_first'].parent != run['new_second'].parent
        for key, expected in (('restored', 5), ('new_first', 3), ('new_second', 4)):
            path = run[key]
            assert path.is_relative_to(output_root)
            if key != 'restored':
                assert path.parent.parent == base
            signal = _read_fresh(path, tmp_path / 'reader')
            assert signal.xlabel == 'time'
            assert len(signal.y) > 1
            np.testing.assert_allclose(signal.y, expected, rtol=0, atol=1e-9)
            artifacts = list(path.parent.iterdir())
            # Managed generations retain private producing decks alongside
            # the single public prepared deck checked by this example.
            decks = [p for p in artifacts
                     if p.suffix == '.cir' and not p.name.startswith('.silk-')]
            assert len(decks) == 1
            deck = decks[0].read_text()
            assert f'write {path}' in deck
            assert 'No PDK library included' in deck
            assert not any(line.startswith(('.lib ', '.include '))
                           for line in deck.splitlines())
            logs = [p for p in artifacts if p.suffix == '.log']
            assert logs and all(p.stat().st_size > 0 for p in logs)
