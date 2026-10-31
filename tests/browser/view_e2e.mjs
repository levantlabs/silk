// End-to-end interaction test for the waveform explorer.
//
// Loads a REAL emitted view.html in headless Chrome and drives it with real
// mouse/keyboard events, asserting on screenshot diffs. uPlot draws its axes
// to <canvas>, so there is no DOM text to read a scale off -- the only honest
// way to check "did the zoom change the picture / did `f` put it back" is to
// look at the pixels. This is what catches interaction bugs (async state,
// event wiring) that reading the source cannot.
//
// Invoked by tests/test_view_e2e.py, which emits the page and resolves Chrome:
//   node view_e2e.mjs <view.html> <chrome-executable>
// Exit 0 = all checks passed, 1 = a check failed (details on stdout).

import puppeteer from 'puppeteer-core';
import { PNG } from 'pngjs';
import pixelmatch from 'pixelmatch';
import fs from 'fs';
import path from 'path';

const HTML = process.argv[2];
const CHROME = process.argv[3];
if (!HTML || !CHROME) {
  console.error('usage: node view_e2e.mjs <view.html> <chrome-executable>');
  process.exit(2);
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function diffPct(a, b) {
  const pa = PNG.sync.read(a), pb = PNG.sync.read(b);
  if (pa.width !== pb.width || pa.height !== pb.height) return 100;
  const n = pixelmatch(pa.data, pb.data, null, pa.width, pa.height, { threshold: 0.1 });
  return (100 * n) / (pa.width * pa.height);
}

const results = [];
function check(name, cond, detail) {
  results.push({ name, pass: !!cond });
  console.log((cond ? 'PASS' : 'FAIL') + ' :: ' + name + (detail ? '  (' + detail + ')' : ''));
}

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: 'new',
  args: ['--no-sandbox', '--disable-gpu', '--window-size=1400,900'],
});
// Created after the launch so a failed launch leaves no directory behind;
// removed in the finally below.
const OUTDIR = fs.mkdtempSync(path.join(process.env.TMPDIR || '/tmp', 'view-e2e-'));
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });
  const pageErrors = [];
  page.on('pageerror', (e) => pageErrors.push(e.message));

  await page.goto('file://' + path.resolve(HTML), { waitUntil: 'load' });
  await page.waitForSelector('.u-over', { timeout: 15000 });
  await sleep(700); // let uPlot settle

  const box = await page.$eval('.u-over', (el) => {
    const r = el.getBoundingClientRect();
    return { x: r.x, y: r.y, w: r.width, h: r.height };
  });
  const clip = { x: Math.round(box.x), y: Math.round(box.y), width: Math.round(box.w), height: Math.round(box.h) };
  const cx = box.x + box.w / 2, cy = box.y + box.h / 2;

  let shotN = 0;
  async function shot() {
    const p = path.join(OUTDIR, 'shot_' + (shotN++) + '.png');
    await page.screenshot({ path: p, clip });
    return fs.readFileSync(p);
  }
  async function dragTo(x1, y1, x2, y2) {
    await page.mouse.move(x1, y1);
    await page.mouse.down();
    await page.mouse.move((x1 + x2) / 2, (y1 + y2) / 2);
    await page.mouse.move(x2, y2);
    await page.mouse.up();
    await sleep(400);
  }

  // TASK-238: the default layout opens on an empty panel now, so plot real
  // signals first -- an empty chart has nothing for the drag-zoom pixel
  // diffs below to detect.
  await page.evaluate(async () => {
    const plot = window.SILK_VIEW_DATA.plots[window.SILK_VIEW_DATA.default_plot];
    const signals = plot.signals.slice(0, 8).map((s) => s.name);
    const result = await window.silkView.run([{ cmd: 'addSignal', args: { signals } }]);
    if (!result.ok) throw new Error(JSON.stringify(result));
  });
  await sleep(300);

  const A = await shot();

  // Horizontal drag -> X zoom; f -> fit back to full.
  await dragTo(box.x + box.w * 0.30, cy, box.x + box.w * 0.60, cy + 4);
  const B = await shot();
  check('horizontal drag zooms X', diffPct(A, B) > 0.5, diffPct(A, B).toFixed(2) + '% vs initial');
  await page.keyboard.press('f');
  await sleep(450);
  const C = await shot();
  check('f restores full view after X zoom', diffPct(A, C) < 0.3, diffPct(A, C).toFixed(2) + '% vs initial');

  // Vertical drag -> Y zoom; f -> fit back.
  await dragTo(cx, box.y + box.h * 0.25, cx + 4, box.y + box.h * 0.65);
  const D = await shot();
  check('vertical drag zooms Y', diffPct(A, D) > 0.5, diffPct(A, D).toFixed(2) + '% vs initial');
  await page.keyboard.press('f');
  await sleep(450);
  const E = await shot();
  check('f restores full view after Y zoom', diffPct(A, E) < 0.3, diffPct(A, E).toFixed(2) + '% vs initial');

  // Help dialog: opens, closes on backdrop click and on the X button.
  await page.click('#shortcuts-btn');
  await sleep(150);
  check('help opens on ? button', await page.$eval('#shortcuts-help', (el) => !el.hidden));
  await page.mouse.click(clip.x + 5, 8); // the dim backdrop, above .help-inner however many shortcut rows it lists
  await sleep(150);
  check('help closes on click-outside', await page.$eval('#shortcuts-help', (el) => el.hidden));
  await page.click('#shortcuts-btn'); await sleep(120);
  await page.click('#shortcuts-close'); await sleep(150);
  check('help closes on X button', await page.$eval('#shortcuts-help', (el) => el.hidden));

  check('no uncaught page errors', pageErrors.length === 0, pageErrors.join(' | '));
} finally {
  await browser.close();
  fs.rmSync(OUTDIR, { recursive: true, force: true });
}

const failed = results.filter((r) => !r.pass).length;
console.log('\n' + (results.length - failed) + '/' + results.length + ' checks passed');
process.exit(failed ? 1 : 0);
