"""The Tests workflow takes its runner from a repo variable (TASK-11.10.1).

The private-first repo runs CI on a self-hosted runner; after the repo is
public the variable is cleared and GitHub-hosted runners take over. A hardcoded
runner or a hosted-runner home path would silently break one of the two phases.
"""
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / '.github' / 'workflows' / 'tests.yml'
RUNNER_EXPR = "${{ vars.SILK_RUNNER || 'ubuntu-24.04' }}"


def test_every_job_runs_on_the_silk_runner_variable():
    jobs = yaml.safe_load(WORKFLOW.read_text())['jobs']
    assert jobs, 'tests.yml defines no jobs'
    for name, job in jobs.items():
        assert job.get('runs-on') == RUNNER_EXPR, (
            f'job {name!r} runs-on {job.get("runs-on")!r}; '
            f'expected {RUNNER_EXPR!r}')


def test_no_hosted_runner_home_path():
    assert '/home/runner' not in WORKFLOW.read_text()


JOBS = yaml.safe_load(WORKFLOW.read_text())['jobs']

# Jobs exempt from the Node gate. The docs job builds the Sphinx API reference
# (TASK-305.1): it runs Sphinx, not pytest, so it has nothing that needs Node.
# test_node_gate_exemptions_run_no_pytest keeps this list from hiding a pytest job.
NODE_GATE_EXEMPT = {'docs'}


def _runs_pytest(job):
    return any('pytest' in step.get('run', '') for step in job['steps'])


def test_node_gate_exemptions_run_no_pytest():
    for name in NODE_GATE_EXEMPT:
        assert name in JOBS, f'exempt job {name!r} is not in tests.yml'
        assert not _runs_pytest(JOBS[name]), (
            f'{name!r} runs pytest, so it cannot be exempt from the Node gate')


@pytest.mark.parametrize('name', [n for n in JOBS if n not in NODE_GATE_EXEMPT])
def test_every_job_provisions_node_before_pytest(name):
    steps = yaml.safe_load(WORKFLOW.read_text())['jobs'][name]['steps']
    node_indices = [i for i, step in enumerate(steps)
                    if step.get('uses') == 'actions/setup-node@v4']
    assert len(node_indices) == 1, f'{name}: expected one Node setup step'
    node_index = node_indices[0]
    node = steps[node_index]
    assert node['with'] == {'node-version': '22'}
    assert 'if' not in node
    assert 'continue-on-error' not in node
    checkout_indices = [i for i, step in enumerate(steps)
                        if step.get('uses') == 'actions/checkout@v4']
    assert checkout_indices and max(checkout_indices) < node_index
    pytest_indices = [i for i, step in enumerate(steps)
                      if 'pytest' in step.get('run', '')]
    assert pytest_indices, f'{name}: expected pytest steps'
    assert all(node_index < i for i in pytest_indices)


def test_workflow_does_not_opt_out_of_node_gate():
    assert 'SILK_ALLOW_NO_NODE' not in WORKFLOW.read_text()
