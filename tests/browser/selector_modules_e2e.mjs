import puppeteer from 'puppeteer-core';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import fs from 'node:fs';
const [html,chrome]=process.argv.slice(2);
const browser=await puppeteer.launch({executablePath:chrome,headless:true,args:['--no-sandbox','--allow-file-access-from-files']});
try {
  const page=await browser.newPage(), errors=[], external=[];
  page.on('pageerror',e=>errors.push(String(e)));
  page.on('request',r=>{if(/^https?:/.test(r.url())) external.push(r.url());});
  await page.goto(pathToFileURL(html).href);
  await page.waitForSelector('.u-over',{timeout:15000});
  // Import exactly the selector URL linked by the emitted explorer module.
  const text=fs.readFileSync(html,'utf8');
  const match=/import \{hierParse, validateNodePaths\} from '(data:[^']+)'/.exec(text);
  assert.ok(match,'emitted explorer must link the selector module');
  const result=await page.evaluate(async url=>{
    const {hierParse,matchesSelector,selectNodePaths}=await import(url);
    return [hierParse({name:'V(XDUT:XX1<0>:MID)'}),matchesSelector('V(XDUT:XX1<0>:MID)','xdut.xx1[0].mid'),
      selectNodePaths([{vectors:[{name:'V(XDUT:MID)',selectable:true}]}],{signals:['xdut.mid']}).masks];
  },match[1]);
  assert.deepEqual(result,[{wrapper:'V',segments:['XDUT','XX1<0>'],leaf:'MID'},true,[[true]]]);
  assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
  console.log('PASS mounted shared selectors with no external fetch');
} finally {await browser.close();}
