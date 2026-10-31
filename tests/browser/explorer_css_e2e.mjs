// Browser regression for the waveform explorer's CSS isolation contract.
//
// Usage:
//   node explorer_css_e2e.mjs <view.css> <chrome-executable>

// The test snapshots computed styles on ordinary host-page elements before
// and after loading the real stylesheet. It also mounts two explorers with
// different sizes and custom properties to exercise container responsiveness,
// independent theming, and root-contained overlays.

import fs from 'fs';
import path from 'path';
import puppeteer from 'puppeteer-core';

const CSS_PATH = process.argv[2];
const CHROME = process.argv[3];
if (!CSS_PATH || !CHROME) {
  console.error('usage: node explorer_css_e2e.mjs <view.css> <chrome-executable>');
  process.exit(2);
}

const css = fs.readFileSync(path.resolve(CSS_PATH), 'utf8');
const results = [];

function check(name, condition, detail = '') {
  const pass = Boolean(condition);
  results.push({ name, pass });
  console.log(`${pass ? 'PASS' : 'FAIL'} :: ${name}${detail ? `  (${detail})` : ''}`);
}

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: 'new',
  args: ['--no-sandbox', '--disable-gpu', '--window-size=1400,900'],
});

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });
  await page.setContent(`
    <!doctype html>
    <style>
      html { background: rgb(248, 249, 250); }
      body { margin: 19px; background: rgb(245, 240, 235); color: rgb(31, 41, 55); font: 17px/1.7 Georgia, serif; }
      button { color: rgb(95, 20, 135); background: rgb(255, 238, 170); border: 4px dotted rgb(10, 90, 120); border-radius: 11px; padding: 9px 13px; font: 21px/1.2 Georgia, serif; cursor: crosshair; }
      select { color: rgb(20, 95, 35); background: rgb(220, 255, 225); border: 3px solid rgb(30, 80, 40); padding: 8px; font: 18px/1.3 Georgia, serif; }
      input { color: rgb(135, 35, 30); background: rgb(255, 225, 220); border: 2px dashed rgb(110, 20, 15); padding: 7px; font: 16px/1.4 Georgia, serif; }
      .uplot { color: rgb(80, 25, 140); width: 333px; font-family: Georgia, serif; line-height: 2; }
      .narrow-mount { width: 780px; height: 320px; margin: 12px 0; }
      .wide-mount { width: 1100px; height: 320px; margin: 12px 0; }
    </style>
    <main id="foreign">
      <h1>Host application</h1>
      <button id="foreign-button">Host button</button>
      <select id="foreign-select"><option>Host option</option></select>
      <input id="foreign-input" value="Host input">
      <div id="foreign-uplot" class="uplot">Host chart class</div>
    </main>
    <section class="narrow-mount silk-explorer" id="explorer-a" style="--accent: rgb(255, 0, 102)">
        <div data-silk="app">
          <header data-silk="toolbar">
            <button data-silk="sidebar-toggle">Signals</button>
            <span data-silk="toolbar-center"><span data-silk="app-title">Explorer A</span></span>
          </header>
          <div data-silk="body"><aside data-silk="sidebar"></aside><main data-silk="main"></main></div>
        </div>
        <div data-silk="toast-container"></div>
        <div data-silk="fatal-banner" hidden></div>
        <div data-silk="shortcuts-help" hidden></div>
    </section>
    <section class="wide-mount silk-explorer" id="explorer-b" style="--accent: rgb(0, 204, 136)">
        <div data-silk="app">
          <header data-silk="toolbar">
            <button data-silk="sidebar-toggle">Signals</button>
            <span data-silk="toolbar-center"><span data-silk="app-title">Explorer B</span></span>
          </header>
          <div data-silk="body"><aside data-silk="sidebar"></aside><main data-silk="main"></main></div>
        </div>
    </section>
  `);

  const snapshotForeign = () => page.evaluate(() => {
    const properties = [
      'margin', 'backgroundColor', 'color', 'fontFamily', 'fontSize',
      'lineHeight', 'border', 'borderRadius', 'padding', 'cursor', 'width',
      'boxSizing',
    ];
    const selectors = ['html', 'body', '#foreign', '#foreign-button', '#foreign-select', '#foreign-input', '#foreign-uplot'];
    return Object.fromEntries(selectors.map((selector) => {
      const style = getComputedStyle(document.querySelector(selector));
      return [selector, Object.fromEntries(properties.map((property) => [property, style[property]]))];
    }));
  });

  const before = await snapshotForeign();
  const dimensionsBefore = await page.evaluate(() => Object.fromEntries(
    ['#explorer-a', '#explorer-b'].map((selector) => {
      const rect = document.querySelector(selector).getBoundingClientRect();
      return [selector, { width: rect.width, height: rect.height }];
    }),
  ));
  await page.addStyleTag({ content: css });
  const after = await snapshotForeign();
  check(
    'loading explorer CSS leaves foreign computed styles unchanged',
    JSON.stringify(after) === JSON.stringify(before),
    JSON.stringify({ before, after }),
  );

  const explorerState = await page.evaluate(() => {
    function state(rootSelector) {
      const root = document.querySelector(rootSelector);
      const rootRect = root.getBoundingClientRect();
      const appRect = root.querySelector('[data-silk="app"]').getBoundingClientRect();
      const title = root.querySelector('[data-silk="app-title"]');
      const toggle = root.querySelector('[data-silk="sidebar-toggle"]');
      const center = root.querySelector('[data-silk="toolbar-center"]');
      return {
        accent: getComputedStyle(root).getPropertyValue('--accent').trim(),
        titleColor: getComputedStyle(title).color,
        rootPosition: getComputedStyle(root).position,
        rootWidth: rootRect.width,
        rootHeight: rootRect.height,
        appHeight: appRect.height,
        toggleDisplay: getComputedStyle(toggle).display,
        centerDisplay: getComputedStyle(center).display,
      };
    }
    return { narrow: state('#explorer-a'), wide: state('#explorer-b') };
  });

  check('explorer roots inherit theme overrides independently',
    explorerState.narrow.accent === 'rgb(255, 0, 102)' &&
    explorerState.wide.accent === 'rgb(0, 204, 136)' &&
    explorerState.narrow.titleColor === 'rgb(255, 0, 102)' &&
    explorerState.wide.titleColor === 'rgb(0, 204, 136)',
    JSON.stringify(explorerState));
  check('explorer preserves host-defined root dimensions and app fills them',
    explorerState.narrow.rootWidth === dimensionsBefore['#explorer-a'].width &&
    explorerState.narrow.rootHeight === dimensionsBefore['#explorer-a'].height &&
    explorerState.narrow.appHeight === explorerState.narrow.rootHeight &&
    explorerState.wide.rootWidth === dimensionsBefore['#explorer-b'].width &&
    explorerState.wide.rootHeight === dimensionsBefore['#explorer-b'].height &&
    explorerState.wide.appHeight === explorerState.wide.rootHeight,
    JSON.stringify({ dimensionsBefore, explorerState }));
  check('responsive layout follows each explorer container width',
    explorerState.narrow.toggleDisplay === 'flex' &&
    explorerState.narrow.centerDisplay === 'none' &&
    explorerState.wide.toggleDisplay === 'none' &&
    explorerState.wide.centerDisplay !== 'none',
    JSON.stringify(explorerState));

  const overlayState = await page.evaluate(() => {
    const root = document.querySelector('#explorer-a');
    root.classList.add('view-drop-active', 'sidebar-open');
    return {
      rootPosition: getComputedStyle(root).position,
      dropPosition: getComputedStyle(root, '::after').position,
      toastPosition: getComputedStyle(root.querySelector('[data-silk="toast-container"]')).position,
      fatalPosition: getComputedStyle(root.querySelector('[data-silk="fatal-banner"]')).position,
      helpPosition: getComputedStyle(root.querySelector('[data-silk="shortcuts-help"]')).position,
      sidebarPosition: getComputedStyle(root.querySelector('[data-silk="sidebar"]')).position,
    };
  });
  check('overlays are positioned within the explorer root',
    overlayState.rootPosition === 'relative' &&
    overlayState.dropPosition === 'absolute' &&
    overlayState.toastPosition === 'absolute' &&
    overlayState.fatalPosition === 'absolute' &&
    overlayState.helpPosition === 'absolute' &&
    overlayState.sidebarPosition === 'absolute',
    JSON.stringify(overlayState));
} finally {
  await browser.close();
}

const failed = results.filter(({ pass }) => !pass).length;
console.log(`\n${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
