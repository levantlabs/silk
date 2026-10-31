// SPDX-License-Identifier: MIT
// TASK-375: trace identity -> tb.probe spelling and "Copy as silk code".
// The Python side (tests/test_view_trace_identity.py) proves each mapping
// reads the same rawfile vector; these pin the spellings and the rendering.
import assert from 'node:assert/strict';
import test from 'node:test';

import {TRACE_MENU_ITEMS, plotAnalysis, probeFor, silkCode} from '../../silk/signals/view/assets/explorer.mjs';

const raw = (name, probe, extra = {}) => ({
    identity: 1, kind: 'raw', name, unit: 'V', source: {}, vector: name, part: null,
    probe: probe && {transform: null, analysis: null, ...probe}, probe_reason: probe ? null : 'no equivalent', ...extra,
});
const derived = (operation, operands, parameters = {}) => ({
    identity: 1, kind: 'derived', name: operation, unit: '', source: {}, operation, parameters, operands,
});
const a = raw('v(a)', {name: 'a', type: 'V'});
const b = raw('v(xdut.b)', {name: 'xdut.b', type: 'V'});
const c = raw('i(vvs)', {name: 'vvs', type: 'I'});

test('built-in item ids', () => {
    assert.deepEqual([...TRACE_MENU_ITEMS], ['copy-silk-code']);
    assert.ok(Object.isFrozen(TRACE_MENU_ITEMS));
});

test('vector names map to the tb.probe spelling that reads them', () => {
    for (const [vector, part, want] of [
        ['v(out)', null, {name: 'out', type: 'V', transform: null}],
        ['V(OUT)', null, {name: 'out', type: 'V', transform: null}],
        ['v(xdut.mid)', null, {name: 'xdut.mid', type: 'V', transform: null}],
        ['V(XDUT:VMID)', null, {name: 'xdut.vmid', type: 'V', transform: null}],
        ['v(bus[0])', null, {name: 'bus[0]', type: 'V', transform: null}],
        ['i(vvs)', null, {name: 'vvs', type: 'I', transform: null}],
        ['I(V1)', null, {name: 'v1', type: 'I', transform: null}],
        ['i(v.xdut.xmm1.vamm_d)', null, {name: 'v.xdut.xmm1.vamm_d', type: 'I', transform: null}],
        ['OUT', null, {name: 'out', type: 'V', transform: null}],
        ['V1#branch', null, {name: 'v1#branch', type: 'I', transform: null}],
        ['v(out)', 'mag', {name: 'out', type: 'V', transform: 'mag'}],
        ['v(out)', 'phase', {name: 'out', type: 'V', transform: 'phase_deg_unwrapped'}],
    ]) {
        const result = probeFor(vector, part);
        assert.deepEqual(result.probe, want, vector);
    }
});

test('bus-bit and array-copy vectors map in every spelling a producer writes (TASK-608)', () => {
    for (const [vector, want] of [
        ['v(d[0])', 'd[0]'],                         // ngspice-42
        ['V(D[0])', 'd[0]'],                         // Xyce 7.10 (upper-cased)
        ['v(d<0>)', 'd<0>'],
        ['v(xdut.xamp1<0>.mid)', 'xdut.xamp1<0>.mid'],
        ['V(XDUT:XAMP1<0>:MID)', 'xdut.xamp1<0>.mid'],
    ]) {
        assert.equal(probeFor(vector, null).probe.name, want, vector);
    }
    // A bare bracketed name is no spelling any supported producer writes; it
    // stays refused with the simulator-internal reason, as @device[param] does.
    for (const vector of ['d[0]', 'amp1[0]', '@m1[gm]']) {
        const result = probeFor(vector, null);
        assert.equal(result.probe, null, vector);
        assert.match(result.reason, /simulator-internal vector/, vector);
    }
});

test('vectors with no probe equivalent say so instead of guessing', () => {
    for (const vector of [
        'onoise_spectrum', 'inoise_spectrum', 'onoise_r1_thermal',
        '@m.xdut.xm1.msky130_fd_pr__nfet_01v8[gm]', 'v(@m1[vgs])',
        'xdut.xm1.msky130#dbody', 'v(a,b)', 'a:b', 'v()', '', 'v(it\'s)',
    ]) {
        const result = probeFor(vector, null);
        assert.equal(result.probe, null, vector);
        assert.ok(result.reason && result.reason.length > 10, vector);
    }
    assert.equal(probeFor('v(out)', 'imag').probe, null);
});

test('plot analysis mirrors plot_silk_analysis', () => {
    const plot = (name, extra = {}) => ({name, complex: false, npoints: 10, x: {name: 'time'}, ...extra});
    assert.equal(plotAnalysis(plot('Transient Analysis')), 'tran');
    assert.equal(plotAnalysis(plot('AC Analysis', {complex: true})), 'ac');
    assert.equal(plotAnalysis(plot('DC transfer characteristic', {npoints: 1})), 'dc');
    assert.equal(plotAnalysis(plot('Operating Point', {npoints: 1})), 'op');
    assert.equal(plotAnalysis(plot('Noise Spectral Density Curves')), 'noise');
    assert.equal(plotAnalysis(plot('Integrated Noise')), 'noise');
    assert.equal(plotAnalysis(plot('constants', {kind: 'constants'})), null);
    assert.equal(plotAnalysis(plot('mystery', {x: {name: 'frequency'}})), 'noise');
    assert.equal(plotAnalysis(plot('mystery', {complex: true})), 'ac');
    assert.equal(plotAnalysis(plot('mystery', {x: {name: 'v(v-sweep)'}})), 'dc');
    assert.equal(plotAnalysis(plot('mystery', {npoints: 1})), 'op');
    assert.equal(plotAnalysis(null), null);
});

test('raw traces render tb.probe with selector and analysis only when needed', () => {
    assert.deepEqual(silkCode(a), {ok: true, code: "tb.probe('a')"});
    assert.deepEqual(silkCode(b), {ok: true, code: "tb.probe('xdut.b')"});
    assert.deepEqual(silkCode(c), {ok: true, code: "tb.probe('vvs', 'I')"});
    const ac = raw('v(out) (mag)', {name: 'out', type: 'V', transform: 'mag', analysis: 'ac'});
    assert.deepEqual(silkCode(ac), {ok: true, code: "tb.probe('out', analysis='ac').mag()"});
    const quoted = raw("v(o'k)", {name: "o'k\\x", type: 'V'});
    assert.equal(silkCode(quoted).code, "tb.probe('o\\'k\\\\x')");
});

test('derived traces render the pinned operation table with correct precedence', () => {
    for (const [identity, code] of [
        [derived('sum', [a, b]), "tb.probe('a') + tb.probe('xdut.b')"],
        [derived('difference', [a, b]), "tb.probe('a') - tb.probe('xdut.b')"],
        [derived('product', [a, c]), "tb.probe('a') * tb.probe('vvs', 'I')"],
        [derived('quotient', [a, c]), "tb.probe('a') / tb.probe('vvs', 'I')"],
        [derived('mean', [a, b, a]), "(tb.probe('a') + tb.probe('xdut.b') + tb.probe('a')) / 3"],
        [derived('abs', [a]), "tb.probe('a').mag()"],
        [derived('derivative', [a]), "tb.probe('a').derivative()"],
        [derived('integral', [a]), "tb.probe('a').integral()"],
        [derived('scale', [a], {gain: 2.5, offset: -0.1}), "2.5 * tb.probe('a') - 0.1"],
        [derived('scale', [a], {gain: 1, offset: 0.5}), "tb.probe('a') + 0.5"],
        [derived('scale', [a], {gain: -3, offset: 0}), "-3 * tb.probe('a')"],
        [derived('scale', [a], {}), "tb.probe('a')"],
        [derived('abs', [derived('difference', [a, b])]), "(tb.probe('a') - tb.probe('xdut.b')).mag()"],
        [derived('difference', [a, derived('difference', [b, a])]), "tb.probe('a') - (tb.probe('xdut.b') - tb.probe('a'))"],
        [derived('quotient', [derived('product', [a, b]), derived('product', [b, a])]),
            "tb.probe('a') * tb.probe('xdut.b') / (tb.probe('xdut.b') * tb.probe('a'))"],
        [derived('product', [derived('sum', [a, b]), a]), "(tb.probe('a') + tb.probe('xdut.b')) * tb.probe('a')"],
        [derived('scale', [derived('sum', [a, b])], {gain: 2, offset: 1}), "2 * (tb.probe('a') + tb.probe('xdut.b')) + 1"],
    ]) {
        assert.deepEqual(silkCode(identity), {ok: true, code}, code);
    }
});

test('a derived trace with an unmappable operand reports it instead of guessing', () => {
    const noise = raw('onoise_spectrum', null, {probe_reason: 'onoise_spectrum is a noise vector'});
    const result = silkCode(derived('sum', [a, noise]));
    assert.equal(result.ok, false);
    assert.match(result.reason, /noise vector.*operand "onoise_spectrum"/);
    assert.equal(silkCode(derived('mystery', [a])).ok, false);
    assert.equal(silkCode(derived('scale', [a], {gain: Infinity})).ok, false);
    assert.equal(silkCode(null).ok, false);
});

// TASK-530 (owner D2): Copy-as-code spells the [k] label when that alias
// resolves to exactly one vector of the plot, the raw spelling otherwise.
import {traceIdentityOf} from '../../silk/signals/view/assets/explorer.mjs';

const sig = (name, extra = {}) => ({name, unit: 'V', type: 'voltage', ...extra});
const payloadOf = signals => ({run: {id: 'r'}, plots: [{name: 'Transient Analysis', npoints: 4, x: {name: 'time'}, signals}]});

// TASK-530 round 3: Copy-as-code reads the proven `address`, never a label.
test('copy-as-code uses signal.address through the probe formatter; labels are display only', () => {
    const p = payloadOf([
        sig('v(xx1<0>.mid)', {label: 'anything', address: 'v(xx1[0].mid)'}),
        sig('i(vbias<0>)', {type: 'current', address: 'i(vbias[0])'}),
        sig('v(out)', {label: 'in'}),                      // arbitrary label, no address
        sig('v(a<0>)', {label: 'v(a[0])'}),                // codec-looking label, no address
    ]);
    assert.equal(silkCode(traceIdentityOf(p, 0, 'v(xx1<0>.mid)')).code, "tb.probe('xx1[0].mid')");
    assert.equal(silkCode(traceIdentityOf(p, 0, 'i(vbias<0>)')).code, "tb.probe('vbias[0]', 'I')");
    assert.equal(silkCode(traceIdentityOf(p, 0, 'v(out)')).code, "tb.probe('out')");
    assert.equal(silkCode(traceIdentityOf(p, 0, 'v(a<0>)')).code, "tb.probe('a<0>')");
});

test('op display shape does not override the selectable header analysis', () => {
    for (const [name, want] of [['Operating Point','op'], ['DC transfer characteristic','dc'],
        ['Transient Analysis','tran'], ['AC Analysis','ac']]) {
        const p = {run: {}, plots: [{name: 'AC Analysis', signals: []},
            {name, kind: 'op', npoints: 1, x: {name: 'sample'}, signals: [sig('v(in)')]}]};
        assert.equal(plotAnalysis(p.plots[1]), want);
        assert.equal(silkCode(traceIdentityOf(p, 1, 'v(in)')).code, `tb.probe('in', analysis='${want}')`);
    }
});

test('coordinate roles refuse code while genuine time/sweep voltages remain probeable', () => {
    const p = payloadOf([sig('time', {role:'coordinate'}), sig('sweep', {role:'coordinate'}),
        sig('v(in)'), sig('v(time)'), sig('v(sweep)')]);
    for (const name of ['time','sweep']) {
        const identity = traceIdentityOf(p, 0, name);
        assert.equal(identity.probe, null);
        assert.match(silkCode(identity).reason, /coordinate/);
    }
    for (const name of ['in','time','sweep']) assert.equal(silkCode(traceIdentityOf(p,0,`v(${name})`)).code, `tb.probe('${name}')`);
    assert.throws(() => traceIdentityOf(p, 0, 'sample'), /unknown signal/);
});

test('checked node probes preserve case and explicit refusal bypasses legacy address', async () => {
    const {traceIdentityOf} = await import('../../silk/signals/view/assets/explorer.mjs');
    const sig={name:'V(XDUT:MID)',type:'voltage',node_probe:{name:'v(dut.Stage.Mid)',type:'V'}};
    const payload={run:{id:'checked'},plots:[{name:'Transient Analysis',kind:'tran',signals:[sig]}]};
    assert.equal(silkCode(traceIdentityOf(payload,0,sig.name)).code,"tb.probe('v(dut.Stage.Mid)')");
    sig.node_probe=null;sig.address='v(xdut.mid)';
    assert.equal(traceIdentityOf(payload,0,sig.name).probe,null);
});

test('checked source-card probes carry current type and ignore display labels', async () => {
    const {traceIdentityOf} = await import('../../silk/signals/view/assets/explorer.mjs');
    const node_probe = {name:'i(xdut.Ibias[0]__silk_dc)', type:'I'};
    const sig = {name:'i(@i.xdut.ibias<0>__silk_dc[current])', type:'current',
        label:'i(unrelated)', node_probe};
    const payload = {run:{id:'source-card'}, plots:[
        {name:'Transient Analysis', kind:'tran', signals:[sig]},
    ]};
    const identity = traceIdentityOf(payload, 0, sig.name);
    assert.equal(identity.vector, sig.name);
    assert.deepEqual(identity.probe, {...node_probe, transform:null, analysis:null});
    assert.deepEqual(silkCode(identity), {ok:true,
        code:"tb.probe('i(xdut.Ibias[0]__silk_dc)', 'I')"});
    sig.node_probe = {name:'v(out)', type:'I'};
    assert.equal(traceIdentityOf(payload, 0, sig.name).probe, null);
});

test('Python code escapes control characters in exact node arguments', async () => {
    const {traceIdentityOf} = await import('../../silk/signals/view/assets/explorer.mjs');
    const sig={name:'v(raw)',node_probe:{name:'v(dut.a\nb\t\x00)',type:'V'}};
    const identity=traceIdentityOf({plots:[{name:'Transient Analysis',signals:[sig]}]},0,sig.name);
    const result=silkCode(identity);
    assert.ok(!result.code.includes('\n'));
    assert.ok(result.code.includes('\\n'));
    assert.ok(result.code.includes('\\x00'));
});

test('checked copy uniqueness is local to the selected plot, not analyses or parts', async () => {
    const {traceIdentityOf} = await import('../../silk/signals/view/assets/explorer.mjs');
    const node_probe = {name:'v(dut.Mid)', type:'V'};
    const transient = {name:'v(xdut.mid)', node_probe};
    const magnitude = {name:'v(xdut.mid) (mag)', group:'v(xdut.mid)', part:'mag', node_probe};
    const phase = {name:'v(xdut.mid) (phase)', group:'v(xdut.mid)', part:'phase', node_probe};
    const payload = {plots:[
        {name:'Transient Analysis', signals:[transient]},
        {name:'AC Analysis', complex:true, signals:[magnitude, phase]},
    ]};
    assert.equal(silkCode(traceIdentityOf(payload,0,transient.name)).code,
        "tb.probe('v(dut.Mid)', analysis='tran')");
    assert.equal(silkCode(traceIdentityOf(payload,1,magnitude.name)).code,
        "tb.probe('v(dut.Mid)', analysis='ac').mag()");
    assert.equal(silkCode(traceIdentityOf(payload,1,phase.name)).code,
        "tb.probe('v(dut.Mid)', analysis='ac').phase_deg_unwrapped()");
    payload.plots[0].signals.push({name:'v(xdut.mid)#2', renamed:true, node_probe});
    assert.equal(traceIdentityOf(payload,0,transient.name).probe,null);
    assert.ok(traceIdentityOf(payload,1,magnitude.name).probe);
});
