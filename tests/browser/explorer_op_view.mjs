// TASK-591: an operating point is drawn as dot-plot strips inside ordinary panels.
import {createExplorer} from './explorer.mjs';
import * as core from './waveform-core.mjs';
const errors=[];
window.addEventListener('error',e=>errors.push(e.message));
const check=(v,m)=>{if(!v) throw Error(m);};
const same=(a,b,m)=>check(JSON.stringify(a)===JSON.stringify(b),m+': '+JSON.stringify(a)+' != '+JSON.stringify(b));
const wait=ms=>new Promise(r=>setTimeout(r,ms));
const root=document.createElement('div'); root.style.cssText='width:1000px;height:800px'; document.body.append(root);
let explorer;
const q=(sel,from=root)=>from.querySelector(sel), qa=(sel,from=root)=>[...from.querySelectorAll(sel)];
const panelEls=()=>qa('[data-silk="panels"] .panel');
const rows=(i)=>qa('.op-row',panelEls()[i]).map(r=>r.dataset.signal);
const allRows=()=>qa('[data-silk="panels"] .op-row').map(r=>r.dataset.signal);
const ticks=(i)=>qa('.op-svg text.op-tick,.op-svg text.op-reftext',panelEls()[i]).map(t=>t.textContent);
const pickerRow=n=>q('[data-silk="picker-list"] .picker-row[data-name="'+n+'"]');
const click=async el=>{el.dispatchEvent(new MouseEvent('click',{bubbles:true,detail:1})); await wait(60);};
const dbl=async el=>{el.dispatchEvent(new MouseEvent('dblclick',{bubbles:true,detail:2})); await wait(80);};
const dot=(i,n)=>q('.op-row[data-signal="'+n+'"] circle',panelEls()[i]);
const command=async(cmd,args={})=>{const r=await explorer.run({cmd,args}); check(r.ok,JSON.stringify(r)); return r;};
const state=async()=>(await command('getState')).results[0];
const cases=await(await fetch('cases.json')).json();
const trace=n=>getComputedStyle(root.firstElementChild).getPropertyValue('--trace-'+n).trim().toLowerCase();
const fill=(i,n)=>dot(i,n).getAttribute('fill').toLowerCase();
try {
    const load=async(name,opts={})=>{
        explorer?.destroy();
        explorer=createExplorer(root,{uPlot,waveformCore:core,groupSignalsByKind:false,...opts});
        await explorer.load(cases[name]); await wait(150);
    };
    // Fresh OP: one pre-filled panel per unit, strips instead of charts.
    await load('op');
    check(!q('[data-silk="value-table"]') && !q('[data-silk="op-view"]'),'old table or parallel view present');
    same(panelEls().length,2,'one panel per unit');
    same(rows(0),['v(vdd)','v(mid)','v(gnd2)','v(neg)'],'voltage panel, sorted by value');
    same(rows(1),['i(v2)','i(v1)'],'current panel (0 above -30 uA)');
    check(panelEls().every(p=>p.classList.contains('panel-op')) && !q('[data-silk="panels"] canvas'),'op panels draw no uPlot');
    check(panelEls().every(p=>q('select.panel-sort',p) && !q('.panel-yauto,select[title*="Y"]',p)),'sort select, and no Y controls');
    const vrow=q('.op-row[data-signal="v(vdd)"]');
    check(vrow.textContent.includes('1.800 V'),'value with unit: '+vrow.textContent);
    const tip=q('title',vrow).textContent;
    check(tip.includes('v(vdd)') && tip.includes(String(Math.fround(1.8))),'tooltip is the stored value: '+tip);
    check(q('.op-stem',vrow),'stem');
    check(ticks(0).includes('0 V') && ticks(0).some(t=>t.startsWith('vdd 1.800 V')),'reference labels: '+ticks(0));
    check(q('.op-heading',panelEls()[1]).textContent.includes('positive flows into'),'current sign convention');
    const zeroX=Number(q('.op-ref',panelEls()[0]).getAttribute('x1'));
    check(Number(dot(0,'v(neg)').getAttribute('cx'))<zeroX,'negative voltage left of 0');
    check(Number(dot(0,'v(gnd2)').getAttribute('cx'))===zeroX,'zero voltage on the 0 line');
    check(q('.op-row[data-signal="i(v2)"]',panelEls()[1]).textContent.includes('0 A'),'zero current');
    // Samples and X are disabled with a reason; the s key does nothing.
    check(q('[data-silk="samples-toggle"]').disabled && q('[data-silk="samples-toggle"]').title.includes('no samples'),'samples disabled');
    check(q('[data-silk="xlog-toggle"]').disabled,'x scale disabled');
    q('[data-silk="picker-list"]').focus();
    root.dispatchEvent(new KeyboardEvent('keydown',{key:'s',bubbles:true})); await wait(30);
    same(q('[data-silk="samples-toggle"]').getAttribute('aria-pressed'),'false','s key');
    // Axes never rescale when a panel signal is removed or added.
    const before=ticks(0).filter(t=>!t.startsWith('vdd'));
    await command('removeSignal',{panel:0,signal:'v(vdd)'});
    check(!rows(0).includes('v(vdd)'),'removed row still drawn');
    same(ticks(0).filter(t=>!t.startsWith('vdd')),before,'axis rescaled');
    check(ticks(0).some(t=>t.startsWith('vdd 1.800 V')),'reference to highest net lost');
    // Picker: click selects (no plotting); double-click plots into the focused panel.
    await click(pickerRow('v(vdd)'));
    check(!rows(0).includes('v(vdd)') && pickerRow('v(vdd)').classList.contains('selected'),'click only selects');
    panelEls()[0].dispatchEvent(new MouseEvent('mousedown',{bubbles:true}));
    await dbl(pickerRow('v(vdd)'));
    check(rows(0).includes('v(vdd)'),'double-click plots');
    same(qa('.op-added',panelEls()[0]).length,1,'added tag');
    check(Number(getComputedStyle(q('.op-added')).opacity)>0.9,'added tag visible at first');
    await wait(2800);
    check(!q('.op-added') && !q('.op-fresh'),'added highlight released');
    const band=n=>Number(getComputedStyle(q('.op-row[data-signal="'+n+'"] .op-band',panelEls()[0])).opacity);
    check(band('v(vdd)')===0,'band idle');
    q('.op-row[data-signal="v(vdd)"]').dispatchEvent(new MouseEvent('mouseenter'));
    check(band('v(vdd)')>0.1,'hover highlights after the fade: '+band('v(vdd)'));
    check(getComputedStyle(pickerRow('v(vdd)')).outlineStyle!=='none','picker outline on hover');
    q('.op-row[data-signal="v(vdd)"]').dispatchEvent(new MouseEvent('mouseleave'));
    pickerRow('v(neg)').dispatchEvent(new MouseEvent('mouseenter'));
    check(q('.op-row[data-signal="v(neg)"]').classList.contains('op-hover'),'picker->strip hover');
    pickerRow('v(neg)').dispatchEvent(new MouseEvent('mouseleave'));
    // Focused Clear panel and Plot visible.
    await click(q('[data-silk="clear-panel-btn"]'));
    check([...panelEls()].some(p=>!qa('.op-row',p).length && q('.op-empty',p)),'focused panel cleared');
    same(panelEls().length,2,'clear keeps the panel');
    await click(q('[data-silk="plot-visible-btn"]'));
    check(allRows().length>=6,'plot visible adds the visible signals');
    // + panel adds an empty panel; 2 cols lays out.
    q('[data-silk="add-panel-btn"]').click(); await wait(80);
    same(panelEls().length,3,'+ panel');
    q('[data-silk="layout-toggle"]').click(); await wait(80);
    check(q('[data-silk="panels"]').classList.contains('cols-2'),'two columns');
    q('[data-silk="layout-toggle"]').click(); await wait(80);
    // Drag signals into the new panel by the real route: dragstart on the picker row, then dragover and drop
    // on the panel's chart (no command fallback). Two signals dropped together are both tagged as added.
    const drag=async(names,target)=>{
        const dt=new DataTransfer();
        if(names.length>1) { // a multi-selection drag: select the rows first
            for(const [k,n] of names.entries()) pickerRow(n).dispatchEvent(new MouseEvent('click',{bubbles:true,detail:1,ctrlKey:k>0}));
            await wait(60);
        }
        pickerRow(names[0]).dispatchEvent(new DragEvent('dragstart',{bubbles:true,dataTransfer:dt}));
        const chart=q('.panel-chart',target);
        chart.dispatchEvent(new DragEvent('dragover',{bubbles:true,cancelable:true,dataTransfer:dt}));
        chart.dispatchEvent(new DragEvent('drop',{bubbles:true,cancelable:true,dataTransfer:dt}));
        pickerRow(names[0]).dispatchEvent(new DragEvent('dragend',{bubbles:true,dataTransfer:dt}));
        await wait(80);
    };
    await drag(['i(v1)'],panelEls()[2]);
    check(rows(2).includes('i(v1)'),'dragged signal landed in the new panel');
    await drag(['v(mid)','v(neg)'],panelEls()[2]);
    check(rows(2).includes('v(mid)') && rows(2).includes('v(neg)'),'both dropped signals landed');
    same(qa('.op-added',panelEls()[2]).length,2,'every dropped row carries the added tag');
    same(qa('.op-fresh',panelEls()[2]).length,2,'and the highlight');
    // Sort is per panel and saved per panel.
    const sel=q('select.panel-sort',panelEls()[0]);
    sel.value='name'; sel.dispatchEvent(new Event('change',{bubbles:true})); await wait(60);
    const names0=qa('.op-strip[data-unit="V"] .op-row',panelEls()[0]).map(r=>r.dataset.signal.replace(/^[vi]\((.*)\)$/,"$1"));
    same(names0,[...names0].sort(),'by name');
    check(q('select.panel-sort',panelEls()[1]).value==='value','other panel keeps its sort');
    check(qa('.op-row',panelEls()[0]).some(g=>g.getAnimations().length>0),'rows slide to new places');
    sel.value='added'; sel.dispatchEvent(new Event('change',{bubbles:true})); await wait(60);
    await wait(900);
    const st=await state();
    same(st.plots['0'].panels.map(p=>p.sort),['added','value','value'],'saved sort per panel');
    // Collapsed picker counts follow the panels (default grouped picker).
    await load('op',{groupSignalsByKind:true});
    const count=()=>qa('.picker-row.group-row').find(r=>r.textContent.includes('Currents')).textContent;
    check(count().includes('2 plotted'),'count at start: '+count());
    await command('clearPanel',{panel:1});
    check(count().includes('0 plotted'),'count after clearing the panel: '+count());
    await command('addSignal',{panel:1,signals:['i(v1)']}); await wait(60);
    check(count().includes('1 plotted'),'count after adding one: '+count());
    // Colours: colorFor on the real panel, like traces and chips.
    await load('op');
    check(fill(0,'v(vdd)')===trace(1) && fill(0,'v(mid)')===trace(2),'palette slots by position in the panel');
    check(fill(1,'i(v1)')===trace(1),'second panel restarts at slot 1');
    await load('op',{viewState:{activePlot:0,plots:{'0':{panels:[
        {signals:['v(mid)','v(vdd)']},
        {signals:['v(vdd)'],styles:{'v(vdd)':{color:'#abcdef'}}},
        {signals:['v(vdd)','v(mid)'],styles:{'v(vdd)':{color:'#fedcba'}}}]}}}});
    check(fill(0,'v(mid)')===trace(1) && fill(0,'v(vdd)')===trace(2),'reordered panel uses its own order');
    check(fill(1,'v(vdd)')==='#abcdef' && fill(2,'v(vdd)')==='#fedcba','each panel keeps its own override');
    // The style popover offers colour only on a strip; picking one recolours the dot.
    await load('op');
    const sw=qa('.chip .swatch',panelEls()[0])[1];
    sw.dispatchEvent(new MouseEvent('click',{bubbles:true})); await wait(60);
    const pop=q('.style-popover:not([hidden])'); check(pop,'style popover opens on an op chip');
    check(!pop.textContent.includes('Line width') && !pop.textContent.includes('Points') && !pop.textContent.includes('Interpolation'),'colour only: '+pop.textContent);
    const opt=qa('.style-swatch-opt',pop)[5]; opt.click(); await wait(80);
    check(fill(0,'v(mid)')===opt.title.toLowerCase(),'dot takes the chosen colour: '+fill(0,'v(mid)'));
    // A stale saved name does not shift palette slots (same as a restored waveform panel).
    await load('op',{viewState:{activePlot:0,plots:{'0':{panels:[{signals:['v(gone)','v(mid)']}]}}}});
    check(fill(0,'v(mid)')===trace(1),'stale name does not shift colour: '+fill(0,'v(mid)'));
    // A deliberately empty layout survives; a fresh one is pre-filled.
    await load('op',{viewState:{activePlot:0,plots:{'0':{panels:[]}}}});
    same(panelEls().length,0,'empty saved layout kept');
    await load('op',{viewState:{activePlot:0,plots:{'0':{panels:[{signals:[]}]}}}});
    same(panelEls().length,1,'one empty saved panel kept'); same(allRows(),[],'and stays empty');
    // Tooltip precision: stored float32 and float64 values are not rounded.
    await load('prec');
    const tips=qa('.op-row title').map(t=>t.textContent);
    check(tips.some(t=>t.includes(String(Math.fround(1.234567890123)))),'float32 stored value: '+tips);
    check(tips.some(t=>t.includes('1.234567890123e-40')),'float64 stored value: '+tips);
    // Switching plots restores each view; strip width follows the visible host.
    await load('ac-op');
    const idx=cases['ac-op'].plots.findIndex(p=>p.kind==='op');
    check(idx>=0,'fixture has an op plot');
    await command('setPlot',{index:idx}); await wait(120);
    const shown=allRows();
    await command('setPlot',{index:0}); await wait(150);
    check(q('[data-silk="panels"] canvas'),'waveform view restored');
    await command('setPlot',{index:idx}); await wait(120);
    same(allRows(),shown,'op view restored');
    const widthOk=()=>{const svg=q('.op-svg'),host=svg.closest('.panel-chart'),cs=getComputedStyle(host);
        const w=host.clientWidth-parseFloat(cs.paddingLeft)-parseFloat(cs.paddingRight);
        return Math.abs(Number(svg.getAttribute('width'))-Math.max(520,w))<2;};
    check(widthOk(),'svg width after plot switch');
    await explorer.reload(cases['ac-op']); await wait(150);
    check(widthOk(),'svg width after reload');
    // 60 nets stay readable.
    await load('many');
    same(allRows().length,60,'60 rows');
    check(q('.op-svg').getAttribute('height')>=60*20,'tall enough for 60 rows');
    check(!errors.length,errors.join('\n'));
    explorer.destroy();
    await fetch('/result',{method:'POST',body:'PASS'});
} catch(e) { explorer?.destroy(); await fetch('/result',{method:'POST',body:String(e)+'\n'+(e.stack||'')}); }
