// SPDX-License-Identifier: MIT
// Browser mirror of tests/test_view_selection_errors.py (TASK-615): the same three
// selector failures raise the same class names, with .selector and .kwarg set.
import assert from 'node:assert/strict';
import test from 'node:test';
import {
    AmbiguousSelectorError, EmptySelectionError, SelectionError, SelectorNotFoundError,
    selectNodePaths,
} from '../../silk/signals/view/assets/selectors.mjs';

// ViewError is not exported; SelectionError's prototype is the ViewError class.
const ViewError = Object.getPrototypeOf(SelectionError);

const VECTORS = [
    {name: 'time', selectable: false},
    {name: 'v(xdut.mid)', selectable: true},
    {name: 'v(other)', selectable: true},
];
const nodePlots = targets => [{
    vectors: VECTORS,
    node_paths: {version: 1, vectors: VECTORS, paths: [{name: 'dut.mid', targets}]},
}];
const ONE_NODE = [{id: 0, rows: [1]}];
const TWO_NODES = [{id: 0, rows: [1]}, {id: 2, rows: []}];

test('hierarchy: every case is a SelectionError, ViewError and Error', () => {
    for (const cls of [SelectorNotFoundError, AmbiguousSelectorError, EmptySelectionError]) {
        const e = new cls('x', 'sel', 'signals');
        assert.ok(e instanceof SelectionError);
        assert.ok(e instanceof ViewError);
        assert.ok(e instanceof Error);
    }
    assert.ok(!(new EmptySelectionError('x') instanceof SelectorNotFoundError));
    assert.ok(!(new SelectorNotFoundError('x', 'a', 'signals') instanceof AmbiguousSelectorError));
});

test('not found via signals= carries selector and kwarg', () => {
    assert.throws(() => selectNodePaths(nodePlots(ONE_NODE), {signals: ['dut.missing']}), err => {
        assert.ok(err instanceof SelectorNotFoundError);
        assert.equal(err.name, 'SelectorNotFoundError');
        assert.equal(err.selector, 'dut.missing');
        assert.equal(err.kwarg, 'signals');
        assert.equal(err.message, 'signals selector dut.missing matched no signal');
        return true;
    });
});

test('not found via exclude= carries kwarg exclude', () => {
    assert.throws(() => selectNodePaths(nodePlots(ONE_NODE), {exclude: ['dut.missing']}), err => {
        assert.ok(err instanceof SelectorNotFoundError);
        assert.equal(err.selector, 'dut.missing');
        assert.equal(err.kwarg, 'exclude');
        return true;
    });
});

test('two physical nodes raise AmbiguousSelectorError with selector and kwarg', () => {
    assert.throws(() => selectNodePaths(nodePlots(TWO_NODES), {signals: ['dut.mid']}), err => {
        assert.ok(err instanceof AmbiguousSelectorError);
        assert.equal(err.name, 'AmbiguousSelectorError');
        assert.equal(err.selector, 'dut.mid');
        assert.equal(err.kwarg, 'signals');
        assert.equal(err.message, 'signals selector dut.mid is ambiguous: distinct physical nodes');
        return true;
    });
});

test('empty final selection raises EmptySelectionError', () => {
    assert.throws(
        () => selectNodePaths(nodePlots(ONE_NODE), {signals: ['v(other)'], exclude: ['v(other)']}),
        err => {
            assert.ok(err instanceof EmptySelectionError);
            assert.ok(!(err instanceof SelectorNotFoundError));
            assert.equal(err.message, 'signal selection matched zero signals — nothing would be emitted');
            return true;
        });
});
