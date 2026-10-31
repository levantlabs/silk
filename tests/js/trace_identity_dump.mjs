// SPDX-License-Identifier: MIT
// TASK-375 helper for tests/test_view_trace_identity.py: read {payload,
// derived} JSON, append the derived signals to their plots in the shape the
// explorer's derive_signal stores, and print every trace's identity record
// and its "Copy as silk code" result.
import {readFile} from 'node:fs/promises';

import {silkCode, traceIdentityOf} from '../../silk/signals/view/assets/explorer.mjs';

const input = JSON.parse(await readFile(process.argv[2], 'utf8'));
const payload = input.payload;
for (const def of input.derived || []) {
    payload.plots[def.plot].signals.push({name: def.name, unit: '', type: 'derived', part: null, group: null,
        derived: {operation: def.operation, operands: def.operands, params: def.params || null}});
}
const out = [];
payload.plots.forEach((plot, plotIdx) => {
    for (const signal of plot.signals) {
        const identity = traceIdentityOf(payload, plotIdx, signal.name);
        out.push({plot: plotIdx, name: signal.name, identity, code: silkCode(identity)});
    }
});
process.stdout.write(JSON.stringify(out));
