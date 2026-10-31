// TASK-591 (replaces TASK-533's table contract): one-point analyses are ordinary
// charts with a visible point; only a true operating point draws strips.
import {createExplorer} from './explorer.mjs';
import * as core from './waveform-core.mjs';
const errors=[];
window.addEventListener('error',e=>errors.push(e.message));
const check=(v,m)=>{if(!v) throw Error(m);};
const same=(a,b,m)=>check(JSON.stringify(a)===JSON.stringify(b),m+': '+JSON.stringify(a)+' != '+JSON.stringify(b));
const wait=ms=>new Promise(r=>setTimeout(r,ms));
const root=document.createElement('div'); root.style.cssText='width:1200px;height:800px'; document.body.append(root);
const copied=[];
Object.defineProperty(navigator,'clipboard',{value:{writeText:async text=>{copied.push(text);}}});
const charts=[];
class Capture extends uPlot { constructor(o,d,t){ super(o,d,t); charts.push(this); } }
let explorer;
const q=(sel)=>root.querySelector(sel), qa=(sel)=>[...root.querySelectorAll(sel)];
const command=async(cmd,args={})=>{const r=await explorer.run({cmd,args}); check(r.ok,JSON.stringify(r)); return r;};
const show=async (payload,method='load')=>{await explorer[method](payload); await wait(100); check(!errors.length,errors.join('\n'));};
const pickerNames=()=>qa('[data-silk="picker-list"] .picker-row[data-name]').map(r=>r.dataset.name);
const pickerRow=n=>q('[data-silk="picker-list"] .picker-row[data-name="'+n+'"]');
const table=()=>q('[data-silk="value-table"]');
const stripRows=()=>qa('[data-silk="panels"] .op-row').map(r=>r.dataset.signal);
function copy(name,enabled=true) {
    const row=pickerRow(name); check(row,'missing picker row '+name);
    row.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true}));
    const item=q('[data-silk="derive-menu"] [data-item="copy-silk-code"]');
    check(item && item.disabled===!enabled,'copy enablement for '+name);
    if(enabled) item.click();
}
// A one-point analysis: the real axis, a chart, and a point that is drawn.
async function onePoint(payload,axis,signals,plotIdx=0) {
    const plot=payload.plots[plotIdx];
    check(plot.kind!=='op','one-point analysis is not an operating point: '+plot.kind);
    same(plot.x.name,axis,'axis name'); same(plot.signals.map(s=>s.name),signals,'signals (axis is not one)');
    await show(payload);
    if(plotIdx!==payload.default_plot) await command('setPlot',{index:plotIdx});
    check(!table() && !q('[data-silk="op-view"]') && !qa('[data-silk="panels"] .panel-op').length,'no table, no strips');
    check(!pickerNames().includes(axis) || signals.includes(axis),'the axis is not a signal');
    await command('addSignal',{signals:[signals[0]]}); await wait(100);
    const canvas=q('[data-silk="panels"] canvas'); check(canvas,'one-point analysis draws a chart');
    const u=charts.at(-1);
    const pts=u.series[1].points.show;
    check((typeof pts==='function' ? pts(u,1,0,0) : pts)===true,'the single sample is drawn as a point (samples off, default style)');
    check(u.scales.x.min<u.scales.x.max,'x range is padded: '+u.scales.x.min+'..'+u.scales.x.max);
    const seen=new Set(); const d=u.data[1];
    check(d.length===1 && d[0]!==null,'one sample');
}
try {
    const cases=await(await fetch('cases.json')).json();
    for(const typed of [false,true]) {
        const payloads={};
        for(const name of Object.keys(cases)) payloads[name]=typed
            ? await core.decimate(core.plan(core.read(await(await fetch(name+'.raw')).arrayBuffer(),{file_name:name+'.raw'})))
            : cases[name];
        explorer=createExplorer(root,{uPlot:Capture,waveformCore:core,groupSignalsByKind:false});
        // True operating points: strips, every vector kept, vector zero is a signal.
        await show(payloads.op);
        same(stripRows().sort(),['i(v1)','v(in)','v(out)'],'op strips');
        await show(payloads['ac-op']);
        copy('v(in)'); same(copied.at(-1),"tb.probe('in', analysis='op')",'Op copy');
        same(payloads['ac-op'].default_plot,1,'operating point is the default of a combined file');
        same(stripRows().sort(),['i(v1)','v(in)','v(out)'],'ac-op strips');
        await command('setPlot',{index:0});
        check(!qa('[data-silk="panels"] .panel-op').length,'AC plot has no strips');
        await command('addSignal',{signals:['v(out) (mag)']}); await wait(100);
        check(q('[data-silk="panels"] canvas'),'AC chart');
        await command('setPlot',{index:1}); await wait(100);
        same(stripRows().sort(),['i(v1)','v(in)','v(out)'],'op strips after switching back');
        // Picker and panel commands work on strips (no zero-width data in uPlot).
        await command('addPanel',{signals:['v(out)']}); await wait(80);
        check(qa('[data-silk="panels"] .panel-op').length===3,'panel added on an operating point');
        // One-point DC, transient and AC: axis kept, ordinary chart, visible point.
        await onePoint(payloads.dc,'sweep',['v(out)','v(sweep)']);
        copy('v(sweep)'); same(copied.at(-1),"tb.probe('sweep')",'genuine voltage named like the sweep');
        await onePoint(payloads['dc-wrapped'],'v(v-sweep)',['v(out)','v(sweep)']);
        await onePoint(payloads.tran,'time',['v(out)','v(time)']);
        copy('v(time)'); same(copied.at(-1),"tb.probe('time')",'genuine voltage named like time');
        await onePoint(payloads['ac-one'],'frequency',['v(out) (mag)','v(out) (phase)']);
        // The axis value is the real one, not a synthetic sample index.
        await show(payloads.tran);
        check(payloads.tran.plots[0].x.levels[0].n===1,'one x sample');
        // Equal-x and tiny-span waveforms are charts too (padded domain), as before.
        await show(payloads.equal); await command('addSignal',{signals:['v(out)']}); await wait(80);
        check(q('[data-silk="panels"] canvas') && !table(),'equal-x chart');
        for(const name of ['loop','tiny']) {
            await show(payloads[name]); await command('addSignal',{signals:['v(out)']}); await wait(80);
            check(q('[data-silk="panels"] canvas'),name+' missing chart');
        }
        await show(payloads.many); await command('addSignal',{signals:['v(out)']}); await wait(80);
        check(q('[data-silk="panels"] canvas') && !table() && !q('[data-silk="constants-view"] button'),'many equal-x samples: chart, no paging');
        // Derived signals on an operating point reach the picker and the strips.
        explorer.destroy();
        explorer=createExplorer(root,{uPlot:Capture,waveformCore:core,stateKey:'op-contract-'+typed,groupSignalsByKind:false});
        await show(payloads.op);
        await wait(1000);
        pickerRow('i(v1)')?.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true}));
        if(!pickerRow('i(v1)')) { qa('.picker-row.group-row').find(r=>r.textContent.includes('Currents')).click(); await wait(60);
            pickerRow('i(v1)').dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true})); }
        [...qa('[data-silk="derive-menu"] button')].find(b=>b.textContent.startsWith('Abs:')).click();
        await wait(150);
        const derived=(await command('deriveSignal',{operation:'scale',operands:['v(in)'],params:{gain:3,offset:0}})).results[0];
        await wait(100);
        check(pickerNames().includes(derived.name),'derived signal in the picker');
        await command('addSignal',{panel:0,signals:[derived.name]}); await wait(100);
        check(stripRows().includes(derived.name),'derived signal drawn as a strip row');
        await wait(1000);
        check(JSON.parse(localStorage.getItem('silk.view.op-contract-'+typed))?.plots['0']?.derived?.length===2,'derived signals persisted');
        explorer.destroy();
        // Saved layouts: invalid ranges never reach uPlot.
        explorer=createExplorer(root,{uPlot:Capture,waveformCore:core,viewState:{activePlot:1,plots:{'1':{
            panels:[{signals:['v(in)']}],xRange:[0,0]}}}});
        await show(payloads['ac-op']); same(stripRows(),['v(in)'],'restored op layout');
        explorer.destroy();
        explorer=createExplorer(root,{uPlot:Capture,waveformCore:core,viewState:{activePlot:0,plots:{'0':{
            panels:[{signals:['v(out) (mag)']}],xRange:[10,10]}}}});
        await show(payloads['ac-op']);
        check(q('[data-silk="panels"] canvas'),'restored AC range');
        await command('setXRange',{min:10,max:10});
        const state=(await command('getState')).results[0];
        check(state.plots['0'].xRange[0]<state.plots['0'].xRange[1],'equal user range persisted');
        explorer.destroy();
    }
    check(!errors.length,errors.join('\n'));
    await fetch('/result',{method:'POST',body:'PASS'});
} catch(e) { explorer?.destroy(); await fetch('/result',{method:'POST',body:String(e)+'\n'+(e.stack||'')}); }
