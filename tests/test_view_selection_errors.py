"""Typed waveform selection errors (TASK-615, doc-1 Q35 a).

Each of the three selector failures raises its own subclass of
``SelectionError``: ``SelectorNotFoundError`` and ``AmbiguousSelectorError``
carry ``.selector`` and ``.kwarg``; ``EmptySelectionError`` covers an empty
final selection. Every subclass is still a ``ViewError`` and a ``ValueError``,
and the messages are the ones the untyped errors always printed.

Ambiguity only arises from designer-path node metadata, so it is tested on
``select_node_paths`` alone; ``select_for_payload`` never sees that metadata.

Fast suite: no PDK models, no simulator. The browser half runs under Node and
is skipped only when Node is unavailable (see ``_require_node``).
"""
import subprocess
from pathlib import Path

import pytest

import silk.signals.view as view
from silk.signals.view import (
    AmbiguousSelectorError, EmptySelectionError, SelectionError,
    SelectorNotFoundError, ViewError,
)
from silk.signals.view.node_select import select_node_paths
from silk.signals.view.select import select_for_payload
from tests.test_view_js_differential import _require_node

ROOT = Path(__file__).resolve().parents[1]

VECTORS = [
    {'name': 'time', 'selectable': False},
    {'name': 'v(xdut.mid)', 'selectable': True},
    {'name': 'v(other)', 'selectable': True},
]


def _node_plots(mid_targets):
    """One plot whose node inventory maps ``dut.mid`` to ``mid_targets``."""
    return [{
        'vectors': VECTORS,
        'node_paths': {
            'version': 1,
            'vectors': VECTORS,
            'paths': [{'name': 'dut.mid', 'targets': mid_targets}],
        },
    }]


ONE_NODE = [{'id': 0, 'rows': [1]}]
TWO_NODES = [{'id': 0, 'rows': [1]}, {'id': 2, 'rows': []}]


class TestHierarchy:
    @pytest.mark.parametrize('cls', [
        SelectorNotFoundError, AmbiguousSelectorError, EmptySelectionError,
    ])
    def test_every_case_is_a_selection_error_view_error_and_value_error(self, cls):
        assert issubclass(cls, SelectionError)
        assert issubclass(cls, ViewError)
        assert issubclass(cls, ValueError)

    def test_selection_error_itself_is_the_base(self):
        assert issubclass(SelectionError, ViewError)
        assert issubclass(SelectionError, ValueError)

    def test_cases_are_distinct_from_each_other(self):
        assert not issubclass(SelectorNotFoundError, AmbiguousSelectorError)
        assert not issubclass(AmbiguousSelectorError, SelectorNotFoundError)
        assert not issubclass(EmptySelectionError, SelectorNotFoundError)

    def test_all_four_names_are_exported_from_the_view_package(self):
        for name in ('SelectionError', 'SelectorNotFoundError',
                     'AmbiguousSelectorError', 'EmptySelectionError'):
            assert name in view.__all__
            assert getattr(view, name) is globals()[name]


class TestSelectForPayload:
    """The emit-time filter (silk/signals/view/select.py)."""

    def test_unmatched_signals_selector_raises_not_found_with_attributes(self):
        with pytest.raises(SelectorNotFoundError) as info:
            select_for_payload([['v(xdut.out)']], signals=['nope'])
        assert info.value.selector == 'nope'
        assert info.value.kwarg == 'signals'
        assert str(info.value) == "signals= selector 'nope' matched no signal in this rawfile."

    def test_unmatched_exclude_selector_raises_not_found_with_attributes(self):
        with pytest.raises(SelectorNotFoundError) as info:
            select_for_payload([['v(xdut.out)']], exclude=['nope'])
        assert info.value.selector == 'nope'
        assert info.value.kwarg == 'exclude'
        assert str(info.value) == "exclude= selector 'nope' matched no signal in this rawfile."

    def test_near_miss_suggestion_stays_in_the_message(self):
        with pytest.raises(SelectorNotFoundError, match=r'xdut') as info:
            select_for_payload([['v(xdut.out)']], signals=['xdu'])
        assert info.value.selector == 'xdu'

    def test_empty_final_selection_raises_empty(self):
        with pytest.raises(EmptySelectionError) as info:
            select_for_payload([['v(vdd)', 'v(vp)']], max_depth=0,
                               exclude=['v(vdd)', 'v(vp)'])
        assert str(info.value) == (
            'signal selection (signals=/exclude=/max_depth=) matched zero '
            'signals in this rawfile — nothing would be emitted'
        )

    def test_empty_is_not_a_not_found_error(self):
        with pytest.raises(EmptySelectionError) as info:
            select_for_payload([['v(out)', 'v(in)']], exclude=['*', 'v(out)'])
        assert not isinstance(info.value, SelectorNotFoundError)

    def test_except_view_error_still_catches_all_three(self):
        cases = [
            lambda: select_for_payload([['v(out)']], signals=['nope']),
            lambda: select_for_payload([['v(out)']], exclude=['nope']),
            lambda: select_for_payload([['v(out)']], max_depth=0, exclude=['v(out)']),
        ]
        for case in cases:
            with pytest.raises(ViewError):
                case()


class TestSelectNodePaths:
    """The designer-path selector (silk/signals/view/node_select.py)."""

    def test_unmatched_signals_selector_raises_not_found_with_attributes(self):
        with pytest.raises(SelectorNotFoundError) as info:
            select_node_paths(_node_plots(ONE_NODE), signals=['dut.missing'])
        assert info.value.selector == 'dut.missing'
        assert info.value.kwarg == 'signals'
        assert str(info.value) == (
            "signals= selector 'dut.missing' matched no signal in this rawfile."
        )

    def test_unmatched_exclude_selector_raises_not_found_with_attributes(self):
        with pytest.raises(SelectorNotFoundError) as info:
            select_node_paths(_node_plots(ONE_NODE), exclude=['dut.missing'])
        assert info.value.selector == 'dut.missing'
        assert info.value.kwarg == 'exclude'

    def test_two_physical_nodes_raise_ambiguous_with_attributes(self):
        with pytest.raises(AmbiguousSelectorError) as info:
            select_node_paths(_node_plots(TWO_NODES), signals=['dut.mid'])
        assert info.value.selector == 'dut.mid'
        assert info.value.kwarg == 'signals'
        assert str(info.value) == (
            "signals= selector 'dut.mid' is ambiguous: distinct physical nodes"
        )

    def test_ambiguous_is_not_a_not_found_error(self):
        with pytest.raises(AmbiguousSelectorError) as info:
            select_node_paths(_node_plots(TWO_NODES), signals=['dut.mid'])
        assert not isinstance(info.value, SelectorNotFoundError)

    def test_empty_final_selection_raises_empty(self):
        with pytest.raises(EmptySelectionError) as info:
            select_node_paths(_node_plots(ONE_NODE),
                              signals=['v(other)'], exclude=['v(other)'])
        assert str(info.value) == (
            'signal selection (signals=/exclude=/max_depth=) matched zero '
            'signals in this rawfile — nothing would be emitted'
        )

    def test_except_view_error_still_catches_all_three(self):
        cases = [
            lambda: select_node_paths(_node_plots(ONE_NODE), signals=['dut.missing']),
            lambda: select_node_paths(_node_plots(TWO_NODES), signals=['dut.mid']),
            lambda: select_node_paths(_node_plots(ONE_NODE),
                                      signals=['v(other)'], exclude=['v(other)']),
        ]
        for case in cases:
            with pytest.raises(ViewError):
                case()


def test_browser_selection_errors_under_node():
    """The browser mirror raises the same class names for the same cases."""
    node = _require_node()
    result = subprocess.run(
        [node, '--test', str(ROOT / 'tests/js/selection_errors.test.mjs')],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
