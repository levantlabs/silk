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
