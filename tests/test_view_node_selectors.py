"""Shared JS grammar and full-inventory graph-selector behavior."""
import json
from pathlib import Path
import subprocess

from silk.signals.view.node_select import select_node_paths
from silk.signals.view import ViewError
from tests.test_view_js_differential import _require_node

ROOT = Path(__file__).resolve().parents[1]


def test_exported_js_shared_fixtures():
    script = """
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {hierParse, matchesSelector, selectNodePaths} from './silk/signals/view/assets/selectors.mjs';
for (const c of JSON.parse(fs.readFileSync('tests/fixtures/hierparse_cases.json'))) {
  assert.deepEqual(hierParse(c), {wrapper:c.wrapper, segments:c.segments, leaf:c.leaf});
}
const f = JSON.parse(fs.readFileSync('tests/fixtures/selector_cases.json'));
for (const c of f.cases) {
  if (c.match) assert.equal(matchesSelector(c.match.name,c.match.selector),c.match.expected,c.id);
  const names = c.names || f.names;
  const plots = [{vectors:names.map(name=>({name,selectable:true}))}];
  if (c.expect.refuses) assert.throws(()=>selectNodePaths(plots,c));
  else assert.deepEqual(names.filter((_,i)=>selectNodePaths(plots,c).masks[0][i]),c.expect,c.id);
  if (c.exclude_expect) assert.deepEqual(names.filter((_,i)=>selectNodePaths(plots,{exclude:c.signals}).masks[0][i]),c.exclude_expect,c.id);
}
"""
    subprocess.run([_require_node(), '--input-type=module', '-e', script], cwd=ROOT, check=True)


def test_shared_node_selectors():
    cases = json.loads((ROOT/'tests/fixtures/node_path_selector_cases.json').read_text())
    script = """
import fs from 'node:fs';
import {selectNodePaths} from './silk/signals/view/assets/selectors.mjs';
const cases=JSON.parse(fs.readFileSync('tests/fixtures/node_path_selector_cases.json'));
console.log(JSON.stringify(cases.map(c=>{try{return selectNodePaths(c.plots,c.options)}catch(e){return {error:true}}})));
"""
    result = subprocess.run([_require_node(),'--input-type=module','-e',script],cwd=ROOT,check=True,capture_output=True,text=True)
    js = json.loads(result.stdout)
    for case, actual in zip(cases,js):
        try:
            py = select_node_paths(case['plots'],**case['options'])
        except ViewError:
            py = {'error': True}
        assert py == actual == case['expected'],case['name']


def test_shared_unicode_selector_python_masks():
    from silk.signals.view.select import matches_selector
    fixture = json.loads((ROOT/'tests/fixtures/selector_cases.json').read_text())
    for case in fixture['cases']:
        if 'match' not in case:
            continue
        check = case['match']
        assert matches_selector(check['name'], check['selector']) is check['expected']
        plots = [{'vectors': [{'name': name, 'selectable': True} for name in case['names']]}]
        for options, expected in [({'signals': case['signals']}, case['expect']),
                                  ({'exclude': case['signals']}, case['exclude_expect'])]:
            mask = select_node_paths(plots, **options)['masks'][0]
            assert [name for name, keep in zip(case['names'], mask) if keep] == expected


def test_physical_and_depth_selection_do_not_enumerate_current_aliases():
    from unittest.mock import MagicMock
    plots = [{'vectors': [{'name': 'i(split)', 'selectable': True}]}]
    aliases = MagicMock()
    aliases.__iter__.side_effect = AssertionError('physical match enumerated source aliases')
    for options in ({'signals': ['i(split)']}, {'max_depth': 2}):
        result = select_node_paths(plots, current_aliases=aliases, **options)
        assert result['masks'] == [[True]]
        assert result['current_aliases'] == {}


def test_current_alias_fallback_selects_existing_row_and_keeps_emitted_precedence():
    plots = [{'vectors': [{'name': 'i(vsense)', 'selectable': True},
                          {'name': 'i(vother)', 'selectable': True}]}]
    aliases = ((('i(dut.core.sense)', ((0, 1),), (), True),
                ('i(vsense)', ((1, 1),), (), True)),)
    selected = select_node_paths(plots, signals=['i(dut.core.sense)'], current_aliases=aliases)
    assert selected['masks'] == [[True, False]]
    assert selected['current_aliases'] == {(0, 0): 'i(dut.core.sense)'}
    assert select_node_paths(plots, signals=['i(vsense)'], current_aliases=aliases)['masks'] == [[True, False]]
    assert select_node_paths(plots, exclude=['i(dut.core.sense)'], current_aliases=aliases)['masks'] == [[False, True]]


def test_current_alias_depth_uses_selected_spelling_and_preserves_exclusion():
    import pytest
    from silk.signals.view.node_select import EmptySelectionError
    plots = [{'vectors': [{'name': 'i(vsense)', 'selectable': True},
                          {'name': 'i(vother)', 'selectable': True}]}]
    aliases = ((('i(dut.core.sense)', ((0, 1),), (), True),),)
    assert select_node_paths(plots, signals=['i(dut.core.sense)'], max_depth=2,
                             current_aliases=aliases)['masks'] == [[True, False]]
    with pytest.raises(EmptySelectionError):
        select_node_paths(plots, signals=['i(dut.core.sense)'], max_depth=1,
                          current_aliases=aliases)
    assert select_node_paths(plots, signals=['i(dut.core.sense)', 'i(vother)'],
                             exclude=['i(vsense)'], max_depth=2,
                             current_aliases=aliases)['masks'] == [[False, True]]
