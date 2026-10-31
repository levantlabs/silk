"""``docs/examples.md`` must stay complete, and its **Needs** labels honest.

The examples guide is the human-facing map of ``silk/examples/``; it ships in
the repo next to the code and is what a newcomer reads to pick a script and
learn what it needs. Two things rot silently: a new example lands without an
entry (or an entry outlives its file), and a label says "PDK-free" or "PDK
only" about a script that quietly grew a PDK load or a simulator call. Before
this guard the labels were hand-verified once and date-stamped; every push
since could have drifted.

Follows the README-fence guard pattern (``tests/test_readme_lead_example.py``):
the *published* text is the fixture. Runs in the fast suite, so every CI push
— and therefore every release gate — re-checks it.

What "actual requirements" means here, per label:

* **PDK-free** — the documented ``python -m silk.examples.<name>`` invocation
  runs to completion with **no PDK on disk and no simulator reachable**
  (``$PDK_ROOT`` pointing nowhere, ``$SILK_NGSPICE_BIN`` and ``ngspice`` on
  ``PATH`` both a shim that records the call and fails). Exactly the bare CI
  leg's environment.
* **PDK only** — runs to completion with sky130 present and the same
  simulator shim; the shim must never be invoked.
* **ngspice only** — needs the simulator but no installed PDK. The dedicated
  slow test in ``tests/test_output_modes_example.py`` runs that example
  against a missing PDK root and skips explicitly if ngspice is unavailable.
* **PDK + ngspice** — needs both. Running all of those is the hour-long
  ``scripts/run_examples.py`` sweep, not a unit test; here they are checked
  statically to at least reference a PDK (a PDK-free script cannot honestly
  carry the heavier label). A label may carry a qualifier after the canonical
  prefix (``PDK only; ngspice optional for the final step``); the prefix names
  the default invocation and is what gets exercised.

There is no "slow" label in the guide: the run-based checks above take a few
seconds in total, which is why they live in the fast suite rather than behind
``-m slow``.
"""
import glob
import os
import re
import subprocess
import sys

import pytest

from tests._env import needs_pdk

_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
_EXAMPLES = os.path.join(_REPO, 'silk', 'examples')
_DOC = os.path.join(_REPO, 'docs', 'examples.md')

# The dependency labels the guide defines under "How to run". A label may be
# followed by a qualifier, but must START with one of these.
# 'GF180MCU + ngspice' (TASK-16) names WHICH PDK a non-sky130 example needs;
# any '<PDK> + ngspice' variant is part of the vocabulary. 'SG13G2 + ngspice'
# (TASK-17) added the same way, in the same commit as its docs/examples.md
# row, to avoid the TASK-16/TASK-20 merge collision where a new label-vs-
# vocabulary guard landed in parallel with a row using an unlisted label.
LABELS = ('PDK-free', 'PDK only', 'PDK + ngspice', 'GF180MCU + ngspice',
          'SG13G2 + ngspice', 'ngspice only')

_PER_EXAMPLE_TIMEOUT_S = 300


def _stem_to_module():
    """``stem -> dotted module path``, over the flat examples AND the topic
    subpackages (``silk/examples/metrics/``, the metric-coverage suite). Only
    real packages count: ``silk/examples/output/`` is the gitignored run-output
    directory and holds generated scripts, not examples.

    The guide keys every entry on the bare filename, so a stem must be unique
    across the whole tree — asserted here rather than left to produce a
    confusing mismatch in one of the tests below.
    """
    files = sorted(glob.glob(os.path.join(_EXAMPLES, '*.py'))
                   + [f for f in glob.glob(os.path.join(_EXAMPLES, '*', '*.py'))
                      if os.path.exists(os.path.join(os.path.dirname(f),
                                                     '__init__.py'))])
    out = {}
    for f in files:
        stem = os.path.basename(f)[:-3]
        if stem == '__init__':
            continue
        rel = os.path.relpath(f, _REPO)[:-3].replace(os.sep, '.')
        assert stem not in out, (
            f'two examples share the basename {stem}.py ({out[stem]} and '
            f'{rel}); the guide keys entries on the filename alone')
        out[stem] = rel
    assert out, f'no examples under {_EXAMPLES!r} — glob is broken'
    return out


def _example_stems():
    return set(_stem_to_module())


def _doc_text():
    return open(_DOC, encoding='utf-8').read()


def _mentioned_stems(text):
    """Every ``name.py`` the guide mentions in backticks, table or prose."""
    return {m.group(1) for m in re.finditer(r'`(\w+)\.py`', text)}


def _labelled(text):
    """stem -> Needs label, from the by-topic tables and the learning path."""
    out = {}
    # Table rows: | `x.py` | ... | <label> |
    for m in re.finditer(r'^\| `(\w+)\.py` \|.*\| ([^|]+) \|\s*$', text, re.M):
        out[m.group(1)] = ' '.join(m.group(2).split())
    # Learning path bullets: **[`x.py`](../docs/architecture.md)** — **Label.**
    # (the label may wrap across a line break inside the bold span)
    for m in re.finditer(r'\*\*\[`(\w+)\.py`\]\([^)]*\)\*\*\s*—\s*\*\*([^*]+)\*\*', text):
        out[m.group(1)] = ' '.join(m.group(2).split()).rstrip('.')
    return out


def _with_prefix(labels, prefix):
    return sorted(stem for stem, label in labels.items() if label.startswith(prefix))


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------
def test_every_example_has_an_entry():
    """A new script under silk/examples/ must be mentioned in the guide —
    in a topic table, the learning path, or the utility-files paragraph."""
    missing = sorted(_example_stems() - _mentioned_stems(_doc_text()))
    assert missing == [], (
        'silk/examples/ files with no entry in docs/examples.md: '
        + ', '.join(f'{s}.py' for s in missing))


def test_every_entry_references_an_existing_file():
    """A renamed or deleted example must not leave a dangling entry."""
    text = _doc_text()
    stale = sorted(_mentioned_stems(text) - _example_stems())
    assert stale == [], (
        'docs/examples.md mentions examples that do not exist: '
        + ', '.join(f'{s}.py' for s in stale))
    # Learning-path links carry an explicit relative path — check it resolves.
    for m in re.finditer(r'\]\(\.\./(silk/examples/[\w/]+\.py)\)', text):
        assert os.path.exists(os.path.join(_REPO, m.group(1))), m.group(0)


def test_shipped_script_count_matches_the_guide():
    """The intro's 'ships N runnable scripts' must be the real count."""
    m = re.search(r'ships (\d+) runnable scripts', _doc_text())
    assert m, 'docs/examples.md no longer states how many scripts ship'
    assert int(m.group(1)) == len(_example_stems())


def test_every_curated_example_has_a_label_from_the_vocabulary():
    """Every table/learning-path entry carries a known dependency label; the
    only unlabelled mentions allowed are the two utility files, which the
    guide describes in prose."""
    text = _doc_text()
    labels = _labelled(text)
    for stem, label in sorted(labels.items()):
        assert label.startswith(LABELS), f'{stem}.py: unknown Needs label {label!r}'
    unlabelled = sorted(_mentioned_stems(text) - set(labels))
    utility = {'diff_pair_construction_example', 'compare_outputs'}
    assert set(unlabelled) <= utility, (
        f'examples mentioned without a Needs label (only the utility files '
        f'{sorted(utility)} may be): {unlabelled}')


# ---------------------------------------------------------------------------
# Labels vs. reality
# ---------------------------------------------------------------------------
def _sim_shim(tmp_path):
    """A fake ngspice/Xyce first on PATH (and as $SILK_NGSPICE_BIN) that logs
    every invocation and fails, so a label claiming no simulator is proven."""
    marker = tmp_path / 'simulator-invoked.txt'
    shim = tmp_path / 'bin'
    shim.mkdir()
    for name in ('ngspice', 'Xyce'):
        exe = shim / name
        exe.write_text(f'#!/bin/sh\necho {name} "$@" >> {marker}\nexit 1\n')
        exe.chmod(0o755)
    return shim, marker


def _run(stem, env):
    cp = subprocess.run(
        [sys.executable, '-m', _stem_to_module()[stem]],
        cwd=_REPO, env=env, capture_output=True, text=True,
        timeout=_PER_EXAMPLE_TIMEOUT_S)
    return cp


def _check(stem, label, cp, marker):
    expected = 0
    assert cp.returncode == expected, (
        f'{stem}.py exited {cp.returncode} (expected {expected}) under the '
        f'environment its Needs label {label!r} promises:\n--- stderr ---\n{cp.stderr[-3000:]}')
    # A bare label promises the simulator is never touched. A qualified label
    # ("PDK only; ngspice optional for the final step") may legitimately
    # *probe* for one and degrade when it fails — the exit code above is the
    # contract there.
    if label in LABELS:
        assert not marker.exists(), (
            f'{stem}.py invoked a simulator, contradicting its Needs label {label!r}: '
            + marker.read_text().strip().replace('\n', ', '))


def _base_env(tmp_path):
    shim, marker = _sim_shim(tmp_path)
    env = dict(os.environ,
               PATH=f'{shim}{os.pathsep}' + os.environ.get('PATH', ''),
               SILK_NGSPICE_BIN=str(shim / 'ngspice'),
               SILK_XYCE_BIN=str(shim / 'Xyce'),
               # These run with cwd=_REPO; without this they would each leave
               # a silk-output/ in the checkout (TASK-243).
               SILK_EXAMPLES_OUT=str(tmp_path / 'examples-out'),
               MPLBACKEND='Agg')
    return env, marker


@pytest.mark.parametrize('stem', _with_prefix(_labelled(_doc_text()), 'PDK-free'))
def test_pdk_free_examples_run_with_no_pdk_and_no_simulator(stem, tmp_path):
    env, marker = _base_env(tmp_path)
    env['PDK_ROOT'] = str(tmp_path / 'no-pdk-here')       # the bare CI leg's shape
    env['HOME'] = str(tmp_path / 'home'); os.makedirs(env['HOME'])  # no ~/pdk fallback
    _check(stem, _labelled(_doc_text())[stem], _run(stem, env), marker)


@needs_pdk
@pytest.mark.parametrize('stem', _with_prefix(_labelled(_doc_text()), 'PDK only'))
def test_pdk_only_examples_run_without_a_simulator(stem, tmp_path):
    env, marker = _base_env(tmp_path)                     # PDK_ROOT inherited
    _check(stem, _labelled(_doc_text())[stem], _run(stem, env), marker)


def _source_mentions_pdk(stem, seen=None):
    """Does the example — or a sibling it imports — load a PDK?"""
    seen = seen if seen is not None else set()
    if stem in seen:
        return False
    seen.add(stem)
    rel = _stem_to_module()[stem].replace('.', os.sep) + '.py'
    src = open(os.path.join(_REPO, rel), encoding='utf-8').read()
    if re.search(r'load_pdk\(|BasePDK\(|pdk_details=', src):
        return True
    siblings = set(re.findall(r'from silk\.examples import (\w+)', src))
    siblings |= set(re.findall(r'from silk\.examples\.(\w+) import', src))
    siblings |= set(re.findall(r'from \.(\w+) import', src))
    return any(_source_mentions_pdk(s, seen) for s in siblings if s in _example_stems())


@pytest.mark.parametrize('stem', _with_prefix(_labelled(_doc_text()), 'PDK + ngspice'))
def test_pdk_plus_ngspice_examples_do_reference_a_pdk(stem):
    """The heavy label on a script that never touches a PDK would be a
    mislabel in the other direction. Static — the full ngspice sweep is
    scripts/run_examples.py."""
    assert _source_mentions_pdk(stem), (
        f'{stem}.py is labelled "PDK + ngspice" but neither it nor any sibling '
        f'it imports loads a PDK')


# ---------------------------------------------------------------------------
# The agent guide's curated routes (docs/agent-guide.md §7)
# ---------------------------------------------------------------------------
_AGENT_GUIDE = os.path.join(_REPO, 'docs', 'agent-guide.md')

# Examples deliberately absent from the agent guide's curated index: they are
# listed in docs/examples.md (checked above) but are not a pattern an agent
# building a circuit is routed to. The two indexes are not forced to match;
# this list is the explicit difference, and it must stay exact.
_NOT_ROUTED = {
    # silk/examples/metrics/: the metric-coverage suite, one script per metric
    # family, reached through the metrics docs rather than as build patterns.
    'comparator_and_sampler', 'cs_amp_ac_noise', 'cs_amp_distortion',
    'cs_amp_power_sweep', 'cs_amp_two_tone', 'diff_pair_rejection',
    'eye_centering', 'flash_adc_codes', 'inverter_timing', 'prbs_eye',
    'ring_osc_jitter', 'rlc_filter_ac', 'rlc_step',
    # Top-level examples reached through examples.md only.
    'pll_spec_tree_example', 'source_examples',
}


def _agent_guide_section7(text=None):
    text = text if text is not None else open(_AGENT_GUIDE, encoding='utf-8').read()
    m = re.search(r'^## 7\. Curated example index\n(.*?)(?=^## )', text, re.M | re.S)
    assert m, 'docs/agent-guide.md no longer has a "## 7. Curated example index"'
    return m.group(1)


def _routed(section):
    """``(stems, paths)``: every example the section names, by bare
    backticked filename or by ``silk/examples/...py`` path."""
    paths = set(re.findall(r'silk/examples/([\w/]+\.py)', section))
    stems = {m for m in re.findall(r'`(\w+)\.py`', section)}
    stems |= {p.rsplit('/', 1)[-1][:-3] for p in paths}
    return stems, paths


def test_agent_guide_routes_resolve_to_real_examples():
    """Every example path or filename §7 names exists in the inventory, and a
    path names the file's real location (not just its basename)."""
    stems, paths = _routed(_agent_guide_section7())
    modules = _stem_to_module()
    unknown = sorted(stems - set(modules))
    assert unknown == [], f'agent-guide §7 names examples that do not exist: {unknown}'
    for p in sorted(paths):
        assert os.path.isfile(os.path.join(_EXAMPLES, p)), \
            f'agent-guide §7 cites silk/examples/{p}, which does not exist'


def test_agent_guide_routes_cover_the_inventory_except_the_listed_exclusions():
    stems, _ = _routed(_agent_guide_section7())
    unrouted = _example_stems() - stems
    assert unrouted == _NOT_ROUTED, (
        'agent-guide §7 vs the example inventory differ from _NOT_ROUTED.\n'
        f'  unrouted but not excluded (route it in §7 or exclude it here): '
        f'{sorted(unrouted - _NOT_ROUTED)}\n'
        f'  excluded but now routed or deleted (drop the exclusion): '
        f'{sorted(_NOT_ROUTED - unrouted)}')


def test_route_checker_controls():
    """Positive and negative controls for ``_routed``."""
    sec = ('| a | `silk/examples/foo.py` | x |\n| b | `bar.py` | y |\n'
           '[l](../silk/examples/sub/baz.py)\n')
    stems, paths = _routed(sec)
    assert stems == {'foo', 'bar', 'baz'} and paths == {'foo.py', 'sub/baz.py'}
    assert _routed('no examples here')[0] == set()
    with pytest.raises(AssertionError):
        _agent_guide_section7('## 6. Other\n\n## 8. Gotchas\n')
