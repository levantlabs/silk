import puppeteer from 'puppeteer-core';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import fs from 'node:fs';
const [html,payloadFile,raw,chrome]=process.argv.slice(2);
const browser=await puppeteer.launch({executablePath:chrome,headless:true,args:['--no-sandbox','--allow-file-access-from-files']});
try {
 const page=await browser.newPage(),errors=[];
 await page.setViewport({width:1400,height:1000});
 page.on('pageerror',e=>errors.push(String(e)));
 await page.evaluateOnNewDocument(()=>{
   window.copied=[];
   Object.defineProperty(navigator,'clipboard',{value:{writeText:async text=>copied.push(text)}});
 });
 await page.goto(pathToFileURL(html).href);await page.waitForSelector('.u-over');
 const command=(cmd,args={})=>page.evaluate(async (cmd,args)=>{
   const r=await silkView.run([{cmd,args}]);if(!r.ok) throw new Error(JSON.stringify(r));return r.results[0];
 },cmd,args);
 await command('setTreeMode',{on:false});
 const row='[data-silk="picker-list"] .picker-row[data-name="v(xdut.mid)"]';
 // An unselected designer-labelled row must select by its raw identity.
 await page.click(row,{button:'right'});
 await page.waitForSelector('[data-silk="derive-menu"]:not([hidden])');
 await page.click('[data-silk="derive-menu"] button[data-item="copy-silk-code"]');
 await page.waitForFunction(()=>copied.length===1);
 assert.equal(await page.evaluate(()=>copied[0]),"tb.probe('v(dut.Mid)')");
 await page.focus(row);await page.keyboard.down('Shift');await page.keyboard.press('F10');await page.keyboard.up('Shift');
 await page.waitForSelector('[data-silk="derive-menu"]:not([hidden])');
 await page.click('[data-silk="derive-menu"] button[data-item="copy-silk-code"]');
 await page.waitForFunction(()=>copied.length===2);
 assert.equal(await page.evaluate(()=>copied[1]),"tb.probe('v(dut.Mid)')");
 await command('addPanel',{signals:['v(xdut.mid)']});
 await command('addMarker',{name:'kept',x:1.5});
 await command('deriveSignal',{operation:'sum',operands:['v(xdut.mid)','v(vout)'],name:'sum'});
 const before=await command('getState');
 const old=await page.evaluate(()=>({id:SILK_VIEW_DATA.run.id,key:SILK_VIEW_DATA.run.state_key}));
 const changed=JSON.parse(fs.readFileSync(payloadFile,'utf8'));
 changed.run.id='label-only-refresh';changed.run.state_key=old.key;
 changed.plots[0].signals[1].label='Readable label';
 fs.writeFileSync(html.replace(/\.html$/,'.data.js'),'window.__silkViewData('+JSON.stringify(changed)+');');
 fs.writeFileSync(html.replace(/\.html$/,'.stamp.js'),'window.__silkViewStamp("label-only-refresh",1);');
 await page.waitForFunction(()=>document.querySelector('[data-name="v(xdut.mid)"]').textContent.includes('Readable label'));
 const after=await command('getState');
 for(const state of [before,after]) for(const p of Object.values(state.plots)) p.panels=p.panels.filter(p=>p.signals.length);
 assert.deepEqual(after,before,'raw-key state survives node metadata refresh (empty panels prune)');
 // Opening a local rawfile must clear the old checked metadata and copy route.
 const chooserPromise=page.waitForFileChooser();await page.click('#open-raw-btn');
 await (await chooserPromise).accept([raw]);
 await page.waitForFunction(()=>!document.getElementById('open-raw-btn').disabled);
 assert.equal(await page.evaluate(()=>SILK_VIEW_DATA.plots.some(p=>'node_paths' in p || p.signals.some(s=>'node_probe' in s || 'source_row' in s))),false);
 assert.deepEqual(errors,[]);console.log('PASS checked copy, state refresh and local-file metadata removal');
} finally {await browser.close();}
