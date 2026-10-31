// Exercise the actual emitted standalone document, including its startup path.
const errors = [];
window.addEventListener('message', e => { if(e.data.error) errors.push(e.data.error); });
const frame = document.createElement('iframe');
frame.style.cssText = 'width:1200px;height:800px';
document.body.append(frame);
try {
    frame.src = 'view.html';
    const deadline = Date.now()+12000;
    while (!frame.contentDocument?.querySelector('[data-silk="panels"] .panel-op .op-row')) {
        if(errors.length) throw Error(errors.join('\n'));
        if(Date.now()>deadline) throw Error('no operating-point strips');
        await new Promise(r=>setTimeout(r,50));
    }
    const doc = frame.contentDocument;
    for(const [name,value] of [['v(in)','1.000 V'],['v(out)','2.000 V'],['i(v1)','-3.000 mA']]) {
        const row = [...doc.querySelectorAll('[data-silk="panels"] .op-row')].find(r=>r.dataset.signal===name);
        if(!row || !row.textContent.includes(value)) throw Error(name+': '+row?.textContent);
    }
    if(doc.querySelector('[data-silk="value-table"]')) throw Error('value table present');
    if(doc.querySelector('[data-silk="panels"] canvas')) throw Error('scalar canvas');
    if(errors.length) throw Error(errors.join('\n'));
    await fetch('/result', {method:'POST', body:'PASS'});
} catch(e) { await fetch('/result', {method:'POST', body:String(e)+'\n'+(e.stack||'')}); }
