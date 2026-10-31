// TASK-591: operating-point panels persist through the ordinary saved layout.
import {createExplorer} from './explorer.mjs';
import * as core from './waveform-core.mjs';
const root=document.createElement('div');
root.style.cssText='width:1200px;height:800px'; document.body.append(root);
const errors=[], failures=[];
window.addEventListener('error',e=>errors.push(e.message));
const check=(value,message)=>{if(!value) throw Error(message);};
const same=(a,b,message)=>check(JSON.stringify(a)===JSON.stringify(b),message+': '+JSON.stringify(a)+' != '+JSON.stringify(b));
const wait=ms=>new Promise(r=>setTimeout(r,ms));
let explorer;
const state=async()=>{
    const result=await explorer.run({cmd:'getState',args:{}});
    check(result.ok,JSON.stringify(result)); return result.results[0];
};
const settle=()=>wait(1000); // both save debounce stages
const raw=(plotname,last)=>new TextEncoder().encode(`Title: preserved state
Plotname: ${plotname}
Flags: real
No. Variables: 3
No. Points: ${plotname==='Operating Point'?1:2}
Variables:
0 sweep voltage
1 v(out) voltage
2 v(xamp.mid) voltage
Values:
${plotname==='Operating Point'?'0 1\n2\n4\n':'0 1\n2\n4\n1 '+last+'\n3\n5\n'}`);
const payload=async(plotname,last)=>core.decimate(core.plan(core.read(raw(plotname,last),{file_name:'rerun.raw'})));
const styled={color:'#123456'};
const savedPlot=panels=>({panels,xRange:[1,2],xLog:false,cursors:[1],markers:[],
    expanded:['kind:voltage/xamp'],sections:[]});
const dots=()=>[...root.querySelectorAll('[data-silk="panels"] .op-row circle')];
async function scenario(name,run) {
    try { await run(); } catch(error) { failures.push(name+': '+error.message+'\n'+error.stack); }
    finally { explorer?.destroy(); explorer=null; }
}
try {
    const op=await payload('Operating Point'), wave=await payload('DC transfer characteristic',2);
    await scenario('annotations, styles and sort survive boot autosave and a new instance',async()=>{
        const stateKey='state-op';
        const panels=[{signals:['v(out)','v(xamp.mid)'],styles:{'v(out)':styled},sort:'name'},{signals:[]}];
        const saved={...savedPlot(panels),markers:[{name:'ref',x:0.5,color:'#00ff00',unit:'V',ref:true}]};
        explorer=createExplorer(root,{uPlot,waveformCore:core,stateKey,viewState:{activePlot:0,plots:{'0':saved}}});
        await explorer.load(op); await wait(150);
        same(dots().length,2,'saved panel drawn');
        check(dots()[0].getAttribute('fill')===styled.color || dots()[1].getAttribute('fill')===styled.color,'style override applied');
        await settle();
        const persisted=JSON.parse(localStorage.getItem('silk.view.'+stateKey)).plots['0'];
        same(persisted.panels.map(p=>[p.signals,p.sort]),[[['v(out)','v(xamp.mid)'],'name'],[[],'value']],'panels and sort persisted (empty panel kept)');
        same(persisted.cursors,[1],'cursors'); same(persisted.expanded,['kind:voltage/xamp'],'expanded'); same(persisted.sections,[],'sections');
        same(persisted.markers.map(m=>[m.name,m.x,m.unit,m.ref]),[['ref',0.5,'V',true]],'unit-bearing marker persisted');
        const before=await state();
        explorer.destroy();
        explorer=createExplorer(root,{uPlot,waveformCore:core,stateKey});
        await explorer.load(op); await wait(150);
        const after=await state();
        same(after.plots['0'].panels,before.plots['0'].panels,'panels/styles/sort lost in a fresh instance');
        same(after.plots['0'].markers,before.plots['0'].markers,'markers lost in a fresh instance');
        same(dots().length,2,'fresh instance draws the saved panel');
    });
    await scenario('a cleared layout is not repopulated',async()=>{
        const stateKey='state-cleared';
        explorer=createExplorer(root,{uPlot,waveformCore:core,stateKey});
        await explorer.load(op); await wait(150);
        check(dots().length===3,'fresh op is pre-filled');
        await explorer.run({cmd:'clearPanel',args:{panel:0}}); await explorer.run({cmd:'clearPanel',args:{panel:1}});
        await settle();
        explorer.destroy();
        explorer=createExplorer(root,{uPlot,waveformCore:core,stateKey});
        await explorer.load(op); await wait(150);
        same(dots().length,0,'cleared layout stays cleared');
        await explorer.run({cmd:'removePanel',args:{panel:0}}); await explorer.run({cmd:'removePanel',args:{panel:0}});
        await settle();
        explorer.destroy();
        explorer=createExplorer(root,{uPlot,waveformCore:core,stateKey});
        await explorer.load(op); await wait(150);
        same(root.querySelectorAll('[data-silk="panels"] .panel').length,0,'empty panel list stays empty');
    });
    await scenario('waveform, operating point, waveform reload keeps the layout',async()=>{
        const panels=[{signals:['v(out)'],styles:{'v(out)':styled}}];
        explorer=createExplorer(root,{uPlot,waveformCore:core,stateKey:'state-rerun',
            viewState:{activePlot:0,plots:{'0':savedPlot(panels)}}});
        await explorer.load(wave); await wait(150);
        const before=(await state()).plots['0'];
        same(before.panels.map(p=>p.signals),[['v(out)']],'control selection');
        await explorer.reload(op); await wait(150);
        same(dots().length,1,'operating point draws the saved selection');
        check(dots()[0].getAttribute('fill')===styled.color,'style survives the switch');
        const during=(await state()).plots['0'];
        same(during.panels.map(p=>[p.signals,p.styles]),before.panels.map(p=>[p.signals,p.styles]),'op erased waveform panels/styles');
        same(during.cursors,before.cursors,'cursors'); same(during.expanded,before.expanded,'expanded');
        await explorer.reload(wave); await wait(150);
        check(root.querySelector('[data-silk="panels"] canvas'),'waveform chart not restored');
        const after=(await state()).plots['0'];
        same(after.panels.map(p=>[p.signals,p.styles]),before.panels.map(p=>[p.signals,p.styles]),'reload erased waveform panels/styles');
    });
    check(!errors.length,errors.join('\n'));
    check(!failures.length,failures.join('\n'));
    await fetch('/result',{method:'POST',body:'PASS'});
} catch(error) { explorer?.destroy(); await fetch('/result',{method:'POST',body:String(error)+'\n'+error.stack}); }
