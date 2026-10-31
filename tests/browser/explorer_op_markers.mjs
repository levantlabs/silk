// TASK-591: reference-level markers and export on an operating point.
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
const strip=(p,unit)=>q('.op-strip[data-unit="'+unit+'"]',panelEls()[p]);
const command=async(cmd,args={})=>{const r=await explorer.run({cmd,args}); return r;};
const ok=async(cmd,args={})=>{const r=await command(cmd,args); check(r.ok,cmd+' '+JSON.stringify(r)); await wait(40); return r;};
const state=async()=>(await ok('getState')).results[0];
const markers=async()=>(await state()).plots[String(explorer.plot ?? 0)]?.markers;
const cases=await(await fetch('cases.json')).json();
const hover=async(el,frac)=>{const svg=q('svg',el),r=svg.getBoundingClientRect();
    svg.dispatchEvent(new MouseEvent('mousemove',{bubbles:true,clientX:Math.round(r.left+r.width*frac),clientY:r.top+40})); await wait(20);};
const tabs=(p,unit)=>qa('.op-marker-badge',strip(p,unit)).map(b=>b.textContent);
const table=()=>qa('[data-silk="markers-table"] tr').map(tr=>[...tr.children].map(c=>c.textContent.trim()));
const grabbed=[]; const origCreate=URL.createObjectURL;
URL.createObjectURL=b=>{grabbed.push(b); return origCreate.call(URL,b);};
HTMLAnchorElement.prototype.click=function(){};
try {
    const load=async(name,opts={})=>{
        explorer?.destroy();
        explorer=createExplorer(root,{uPlot,waveformCore:core,groupSignalsByKind:false,...opts});
        await explorer.load(cases[name]); await wait(150);
    };
    await load('op');
    // + marker without a hovered strip: a toast, no marker.
    q('[data-silk="add-marker-btn"]').click(); await wait(40);
    same((await state()).plots['0'].markers,[],'no marker without a hovered strip');
    // + marker uses the strip last hovered (kept after the toolbar click).
    // Expected level from the strip geometry (the 0 V line and the highest-net line give the scale),
    // at a pointer position that is neither a signal value nor a tick, so snapping would show.
    const geo=(frac)=>{ const w=strip(0,'V'),svg=q('svg',w),r=svg.getBoundingClientRect();
        const refs=qa('.op-ref',w).map(l=>Number(l.getAttribute('x1')));
        const hi=Number(qa('.op-reftext',w).map(t=>t.textContent).find(t=>/ [\d.]+ V$/.test(t)&&!t.startsWith('\u0394')).match(/([\d.]+) V$/)[1]);
        const clientX=Math.round(r.left+r.width*frac); // mouse events carry integer coordinates
        return {value:hi*(clientX-r.left-refs[0])/(refs[1]-refs[0]),
            special:[0,hi,0.9,0.4,...qa('.op-tick',w).map(t=>parseFloat(t.textContent))].filter(Number.isFinite)}; };
    const g=geo(0.47);
    check(g.special.every(v=>Math.abs(v-g.value)>0.01),'fixture point is not a signal or tick value: '+g.value+' vs '+g.special);
    await hover(strip(0,'V'),0.47);
    q('[data-silk="add-marker-btn"]').click(); await wait(80);
    const m1=(await state()).plots['0'].markers[0];
    same([m1.name,m1.unit,m1.ref],['M1','V',true],'first marker is a V reference');
    check(Math.abs(m1.x-g.value)<1e-6,'level is the unsnapped value under the pointer: wanted '+g.value+', got '+m1.x);
    same(tabs(0,'V').length,1,'tab in the V strip');
    same(tabs(1,'A').length,0,'no tab in the A strip');
    check(q('.op-marker',strip(0,'V')) && !q('.op-marker',strip(1,'A')),'line only in V strips');
    check(qa('.op-delta',strip(0,'V')).length===4,'row differences in the V strip');
    check(!q('.op-delta',strip(1,'A')),'no differences in the A strip');
    // The m key on the A strip adds an A marker.
    await hover(strip(1,'A'),0.25);
    q('[data-silk="picker-list"]').focus();
    root.dispatchEvent(new KeyboardEvent('keydown',{key:'m',bubbles:true})); await wait(80);
    same(tabs(1,'A').length,1,'A marker via m');
    same(tabs(0,'V').length,1,'V strips ignore the A marker');
    // Newest marker of a unit becomes its reference.
    await hover(strip(0,'V'),0.8);
    q('[data-silk="add-marker-btn"]').click(); await wait(80);
    let ms=(await state()).plots['0'].markers;
    same(ms.map(m=>[m.name,m.unit,!!m.ref]),[['M1','V',false],['M2','A',true],['M3','V',true]],'insertion order and refs');
    check(q('.op-reftext',strip(0,'V')) && qa('.op-reftext',strip(0,'V')).some(t=>t.textContent==='Δ vs M3'),'strip names its reference');
    // Typed edit / drag make a marker the reference.
    await ok('setMarkerX',{name:'M1',x:0.5,unit:'V'});
    ms=(await state()).plots['0'].markers;
    same(ms.map(m=>[m.name,m.x,!!m.ref]).filter(r=>r[0]!=='M2'),[['M1',0.5,true],['M3',ms[2].x,false]],'edit makes M1 the reference');
    // Drag across redraws: each move redraws the strips (the marker becomes the
    // reference, which narrows the plot area), and the next move must still read
    // the live strip geometry. Several moves separated by redraws, each checked
    // against the value under the pointer.
    const markerX=async(name)=>(await state()).plots['0'].markers.find(m=>m.name===name).x;
    const px=(v)=>{ // pixel of a value from the live strip (0 V line and the highest net line)
        const w=strip(0,'V'),r=q('svg',w).getBoundingClientRect(),refs=qa('.op-ref',w).map(l=>Number(l.getAttribute('x1')));
        const hi=Number(qa('.op-reftext',w).map(t=>t.textContent).find(t=>/ [\d.]+ V$/.test(t)&&!t.startsWith('\u0394')).match(/([\d.]+) V$/)[1]);
        return r.left+refs[0]+(refs[1]-refs[0])*v/hi;};
    let badge=qa('.op-marker-badge',strip(0,'V')).find(b=>b.textContent.startsWith('M3'));
    let bx=badge.getBoundingClientRect();
    badge.dispatchEvent(new MouseEvent('mousedown',{bubbles:true,button:0,detail:1,clientX:bx.left+bx.width/2,clientY:bx.top+3}));
    for(const target of [0.2,1.5,0.7,1.1]) {
        document.dispatchEvent(new MouseEvent('mousemove',{bubbles:true,clientX:px(target),clientY:bx.top})); await wait(120);
        const got=await markerX('M3');
        check(Math.abs(got-target)<0.02,'drag follows the pointer across redraws: wanted '+target+', got '+got);
    }
    document.dispatchEvent(new MouseEvent('mouseup',{bubbles:true})); await wait(80);
    ms=(await state()).plots['0'].markers;
    check(ms[2].ref && !ms[0].ref,'the dragged marker is the reference');
    // After the drop further pointer moves do nothing.
    document.dispatchEvent(new MouseEvent('mousemove',{bubbles:true,clientX:px(0.3),clientY:bx.top})); await wait(80);
    check(Math.abs((await markerX('M3'))-1.1)<0.02,'no drag after mouseup');
    // Errors: conflicting unit, missing/forbidden unit, non-OP reference.
    check(!(await command('setMarkerX',{name:'M1',x:1,unit:'A'})).ok,'conflicting unit refused');
    check(!(await command('addMarker',{x:1})).ok,'unit required on an operating point');
    const named=await ok('addMarker',{x:'1.2',unit:'V',name:'vcm'});
    same(named.results[0].name,'vcm','named OP marker');
    await ok('setMarkerReference',{name:'M1'});
    same((await state()).plots['0'].markers.filter(m=>m.unit==='V'&&m.ref).map(m=>m.name),['M1'],'explicit reference');
    // Right-click menu: Use as reference.
    const tab=qa('.op-marker-badge',strip(0,'V')).find(b=>b.textContent.startsWith('vcm'));
    tab.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true})); await wait(40);
    const use=qa('.dropdown-menu button').find(b=>b.textContent==='Use as reference'&&b.offsetParent);
    check(use,'menu offers Use as reference'); use.click(); await wait(60);
    same((await state()).plots['0'].markers.filter(m=>m.unit==='V'&&m.ref).map(m=>m.name),['vcm'],'reference from the tab menu');
    // Table: columns across every plotted signal, deltas only for matching units.
    q('[data-silk="markers-toggle"]').click(); await wait(60);
    const t=table();
    same(t[0].slice(0,3),['name','unit','level'],'table head');
    check(t[0].includes('v(vdd)') && t[0].includes('i(v1)'),'columns from every panel: '+t[0]);
    const row=t.find(r=>r[0]==='M2'), c=t[0].indexOf('v(vdd)');
    same(row[c],'—','unlike unit is a dash');
    check(/^[+-]/.test(t.find(r=>r[0]==='M1')[c]),'matching unit shows a signed difference: '+t.find(r=>r[0]==='M1')[c]);
    // The open markers table follows every panel mutation (columns are all plotted signals).
    const heads=()=>table()[0].slice(3,-2);
    await ok('removeSignal',{panel:0,signal:'v(neg)'});
    const before=heads();
    check(!before.includes('v(neg)'),'table follows removeSignal while open: '+before);
    await ok('addPanel',{signals:['v(neg)']});
    check(heads().includes('v(neg)') && !before.includes('v(neg)'),'table column after addPanel: '+heads());
    await ok('removeSignal',{panel:2,signal:'v(neg)'});
    check(!heads().includes('v(neg)'),'table column gone after removeSignal');
    await ok('addSignal',{panel:2,signals:['v(neg)']});
    check(heads().includes('v(neg)'),'table column back after addSignal');
    await ok('clearPanel',{panel:2});
    check(!heads().includes('v(neg)'),'table column gone after clearPanel');
    await ok('addSignal',{panel:2,signals:['v(neg)']});
    await ok('removePanel',{panel:2});
    check(!heads().includes('v(neg)'),'table column gone after removePanel');
    // Deleting the reference makes the last remaining marker of that unit the reference.
    await ok('clearMarkers');
    for(const [n,x] of [['A',3],['B',1],['C',2]]) await ok('addMarker',{x,unit:'V',name:n});
    await ok('removeMarker',{name:'C'});
    ms=(await state()).plots['0'].markers;
    same(ms.map(m=>[m.name,!!m.ref]),[['A',false],['B',true]],'B (last added of the rest) is the reference');
    // Save, reopen: order, units and the reference survive; a third marker added out of value order.
    await ok('addMarker',{x:2,unit:'V',name:'C'});
    await ok('setMarkerX',{name:'A',x:2.5});
    await ok('renameMarker',{name:'B',to:'bias'});
    await wait(900);
    const saved=await state();
    same(saved.plots['0'].markers.map(m=>m.name),['A','bias','C'],'insertion order survives edit and rename');
    explorer.destroy();
    explorer=createExplorer(root,{uPlot,waveformCore:core,groupSignalsByKind:false,viewState:saved});
    await explorer.load(cases.op); await wait(150);
    same((await state()).plots['0'].markers.map(m=>[m.name,m.unit,!!m.ref]),[['A','V',true],['bias','V',false],['C','V',false]],'reopened (A was edited last, so it is the reference)');
    await ok('removeMarker',{name:'A'});
    same((await state()).plots['0'].markers.filter(m=>m.ref).map(m=>m.name),['C'],'delete after reopen: C is the last added of the rest');
    // Zero or several saved refs: the last marker of the unit in saved order.
    const mk=(extra)=>({activePlot:0,plots:{'0':{panels:[{signals:['v(vdd)']}],markers:extra}}});
    for(const extra of [[{name:'P',x:1,unit:'V'},{name:'Q',x:2,unit:'V'}],
        [{name:'P',x:1,unit:'V',ref:true},{name:'Q',x:2,unit:'V',ref:true}]]) {
        explorer.destroy();
        explorer=createExplorer(root,{uPlot,waveformCore:core,viewState:mk(extra)});
        await explorer.load(cases.op); await wait(120);
        same((await state()).plots['0'].markers.filter(m=>m.ref).map(m=>m.name),['Q'],'fallback reference');
    }
    // Unit "" is a real (dimensionless) unit: it round-trips and draws in no V or A strip.
    explorer.destroy();
    explorer=createExplorer(root,{uPlot,waveformCore:core,viewState:mk([{name:'dimless',x:2,unit:''}])});
    await explorer.load(cases.op); await wait(120);
    same(qa('.op-marker-badge').length,0,'a dimensionless marker draws in no V/A strip');
    same((await state()).plots['0'].markers.map(m=>[m.name,m.unit]),[['dimless','']],'empty unit round-trips');
    // Waveform markers: still sorted by x, no unit.
    await load('ac-op'); await ok('setPlot',{index:0});
    await ok('addMarker',{x:100}); await ok('addMarker',{x:10});
    same((await state()).plots['0'].markers.map(m=>m.x),[10,100],'waveform markers sorted by x');
    check(!(await command('addMarker',{x:5,unit:'V'})).ok,'unit refused on a waveform');
    // ---- export ----
    await load('csv');
    await ok('addMarker',{x:0.75,unit:'V',name:'M1'}); await ok('addMarker',{x:0.0625,unit:'A',name:'M2'});
    await ok('addSignal',{panel:1,signals:['v(vout)']}); // vout is now in two panels
    const text=async()=>{const b=grabbed.at(-1); return b.text();};
    await ok('exportCsv',{scope:'visible'}); await wait(50);
    same(await text(),[`"signal","value","unit","delta:M1 (V)","delta:M2 (A)"`,
        `"v(vdd)",1.5,"V",0.75,`,`"v(vout)",1.25,"V",0.5,`,
        `"i(ibias)",0.125,"A",,0.0625`,`"i(idd)",-0.25,"A",,-0.3125`,''].join('\n'),'all-panel CSV: one row per distinct signal');
    await ok('exportCsv',{scope:'full'}); await wait(50);
    same((await text()).split('\n').length,6,'full equals visible');
    await ok('exportCsv',{scope:'visible',panel:0}); await wait(50);
    same(await text(),[`"signal","value","unit","delta:M1 (V)","delta:M2 (A)"`,
        `"v(vdd)",1.5,"V",0.75,`,`"v(vout)",1.25,"V",0.5,`,''].join('\n'),'panel CSV');
    const refused=await command('exportCsv',{scope:'cursor'});
    check(!refused.ok && JSON.stringify(refused).includes('needs an x axis'),'cursor export refused: '+JSON.stringify(refused));
    // PNG: panel and all panels compose the strips.
    for(const args of [{scope:'panel',panel:0},{scope:'all'}]) {
        const n=grabbed.length;
        await ok('exportPng',args); await wait(400);
        check(grabbed.length===n+1 && grabbed.at(-1).type==='image/png' && grabbed.at(-1).size>2000,'PNG '+JSON.stringify(args));
        const bmp=await createImageBitmap(grabbed.at(-1)), cv=document.createElement('canvas');
        cv.width=bmp.width; cv.height=bmp.height; const cx=cv.getContext('2d'); cx.drawImage(bmp,0,0);
        const px=cx.getImageData(0,0,cv.width,cv.height).data; let coloured=0;
        for(let i=0;i<px.length;i+=4) if(Math.abs(px[i]-px[i+2])>60) coloured++;
        check(coloured>50,'PNG has dots and levels drawn: '+coloured);
    }
    // A panel narrower than the 520 px strip clamp: the export is sized from the strip model,
    // so the rightmost content (values, difference column, reference tab) is not cropped.
    root.style.width='420px'; await wait(400);
    const svgW=Number(q('.op-svg').getAttribute('width')), hostW=q('.op-svg').closest('.panel-chart').clientWidth;
    check(svgW===520 && hostW<520,'fixture is a narrow panel: svg '+svgW+', host '+hostW);
    const dpr=window.devicePixelRatio||1;
    for(const args of [{scope:'panel',panel:0},{scope:'all'}]) {
        const n=grabbed.length;
        await ok('exportPng',args); await wait(400);
        check(grabbed.length===n+1,'narrow PNG '+JSON.stringify(args));
        const bmp=await createImageBitmap(grabbed.at(-1)), cv=document.createElement('canvas');
        cv.width=bmp.width; cv.height=bmp.height; const cx=cv.getContext('2d'); cx.drawImage(bmp,0,0);
        check(bmp.width>=(520+16)*dpr-1,'PNG holds the whole clamped strip: width '+bmp.width);
        const d=cx.getImageData(0,0,cv.width,cv.height).data, bg=[d[0],d[1],d[2]];
        let ink=0;
        for(let y=0;y<cv.height;y++) for(let x=Math.floor(cv.width-110*dpr);x<cv.width-4*dpr;x++) {
            const i=(y*cv.width+x)*4;
            if(Math.abs(d[i]-bg[0])+Math.abs(d[i+1]-bg[1])+Math.abs(d[i+2]-bg[2])>60) ink++;
        }
        check(ink>30,'rightmost content (difference column) is in the PNG '+JSON.stringify(args)+': '+ink);
    }
    check(!errors.length,errors.join('\n'));
    explorer.destroy();
    await fetch('/result',{method:'POST',body:'PASS'});
} catch(e) { explorer?.destroy(); await fetch('/result',{method:'POST',body:String(e)+'\n'+(e.stack||'')}); }
