import {hierParse, validateNodePaths} from './selectors.mjs';
/* silk waveform explorer — framework-independent mountable renderer.
 * Requires the scoped view.css and a uPlot constructor (MIT, 1.6.32).
 * Payload strings are untrusted; only the static markup below uses innerHTML.
 */
const EXPLORER_HTML = `<div data-silk="app">
  <div data-silk="toolbar">
    <div data-silk="toolbar-left">
      <button type="button" data-silk="sidebar-toggle" title="Toggle signal picker">&#9776;</button>
      <span data-silk="app-title">silk waveform explorer</span>
      <select data-silk="plot-select" title="Select plot"></select>
      <button type="button" data-silk="scope-btn" title="Switch to a different scope (block) of this design" hidden>Scope &#9662;</button>
      <button type="button" data-silk="xlog-toggle" title="X axis scale (linear/log) -- shared by every panel, since panels are X-synced">X: lin (all)</button>
      <button type="button" data-silk="add-panel-btn" title="Add empty panel">+ panel</button>
      <button type="button" data-silk="layout-toggle" title="Toggle 1/2-column panel layout">1 col</button>
      <select data-silk="font-scale-select" title="Text size (applies everywhere, including axis/canvas text)">
        <option value="compact">Compact</option>
        <option value="normal" selected>Normal</option>
        <option value="large">Large</option>
      </select>
      <button type="button" data-silk="samples-toggle" aria-pressed="false" title="Show or hide a dot at every simulated sample (s). Dots are hidden while they would be denser than one per 4 px; zoom in to see them.">Samples</button>
      <button type="button" data-silk="markers-toggle" title="Toggle the markers table">Markers</button>
      <button type="button" data-silk="add-marker-btn" title="Add a named marker at the mouse X (m)">+ marker</button>
    </div>
    <div data-silk="toolbar-center">
      <span data-silk="run-title-group">
        <span data-silk="run-title"></span>
        <span data-silk="run-subtitle" hidden></span>
        <button type="button" data-silk="context-info-btn" title="Run details (i)" hidden>i</button>
      </span>
      <span data-silk="cursor-readout"></span>
    </div>
    <div data-silk="toolbar-right">
      <button type="button" data-silk="open-raw-btn" title="Open a local .raw file (or drop it onto this page)">Open .raw…</button>
      <input data-silk="raw-file-input" type="file" accept=".raw" hidden>
      <span data-silk="reload-badge" class="badge" hidden></span>
      <div data-silk="view-menu-wrap" class="dropdown-wrap">
        <button type="button" data-silk="view-btn" title="Save, load, copy or reset the view state">View &#9662;</button>
        <div data-silk="view-menu" class="dropdown-menu" hidden></div>
      </div>
      <div data-silk="export-menu-wrap" class="dropdown-wrap">
        <button type="button" data-silk="export-btn" title="Export data or an image">Export &#9662;</button>
        <div data-silk="export-menu" class="dropdown-menu" hidden></div>
      </div>
      <button type="button" data-silk="shortcuts-btn" title="Keyboard shortcuts">?</button>
      <input data-silk="view-file-input" type="file" accept=".json,application/json" hidden>
    </div>
  </div>
  <div data-silk="markers-panel" hidden>
    <div data-silk="markers-panel-header">
      <span data-silk="markers-panel-title">Markers</span>
      <button type="button" data-silk="markers-clear-all">Clear all</button>
      <button type="button" data-silk="markers-panel-close" title="Close">&times;</button>
    </div>
    <div data-silk="markers-table-wrap"><table data-silk="markers-table"></table></div>
  </div>
  <div data-silk="body">
    <aside data-silk="sidebar">
      <div data-silk="picker-controls">
        <div data-silk="picker-filter-wrap">
          <div data-silk="picker-filter-input-wrap">
            <input data-silk="picker-filter" type="text" placeholder="filter signals ( / )" autocomplete="off">
            <button data-silk="picker-filter-clear" type="button" title="Clear filter" hidden>&times;</button>
          </div>
          <span data-silk="picker-count-inline"></span>
          <div data-silk="picker-more-wrap" class="dropdown-wrap">
            <button type="button" data-silk="picker-more-btn" title="Signal list actions" aria-haspopup="true">&#8942;</button>
            <div data-silk="picker-more-menu" class="dropdown-menu" hidden></div>
          </div>
        </div>
        <div data-silk="picker-actions">
          <button type="button" data-silk="plot-visible-btn">Plot visible</button>
          <button type="button" data-silk="clear-panel-btn">Clear panel</button>
          <button type="button" data-silk="add-selected-btn" title="Add every selected signal to the focused panel" hidden>Add</button>
        </div>
        <div data-silk="picker-tree-actions">
          <button type="button" data-silk="tree-mode-toggle" title="Toggle between hierarchical tree and flat list">Tree</button>
          <button type="button" data-silk="tree-expand-all" title="Expand all groups">&#8862;</button>
          <button type="button" data-silk="tree-collapse-all" title="Collapse all groups">&#8863;</button>
        </div>
      </div>
      <div data-silk="picker-list" tabindex="0"></div>
      <div data-silk="picker-count"></div>
    </aside>
    <main data-silk="main">
      <div data-silk="constants-view" hidden></div>
      <div data-silk="zero-point-hint" hidden>No data points in this plot.</div>
      <div data-silk="panels"></div>
      <div data-silk="empty-hint" hidden>No panels yet &mdash; drag a signal from the picker, or click &ldquo;Plot visible.&rdquo;</div>
    </main>
  </div>
</div>
<div data-silk="toast-container"></div>
<div data-silk="scope-menu" class="dropdown-menu" hidden></div>
<div data-silk="stamp-slot" style="display:none"></div>
<div data-silk="shortcuts-help" hidden>
  <div class="help-inner">
    <button data-silk="shortcuts-close" type="button" title="Close (Esc)" aria-label="Close">&times;</button>
    <h3>Keyboard shortcuts</h3>
    <table>
      <tr><td>/</td><td>focus the signal filter</td></tr>
      <tr><td>Esc</td><td>clear the filter / close this help</td></tr>
      <tr><td>+ / -</td><td>zoom X in/out about the cursor</td></tr>
      <tr><td>drag</td><td>box zoom &mdash; horizontal drag zooms X (all panels), vertical zooms that panel's Y, diagonal zooms both</td></tr>
      <tr><td>0</td><td>reset X zoom to the full range</td></tr>
      <tr><td>f</td><td>fit: reset X to full and autoscale the Y of the panel under the mouse</td></tr>
      <tr><td>a / b</td><td>place cursor A / B at the mouse X</td></tr>
      <tr><td>shift-click</td><td>place cursor A on a panel</td></tr>
      <tr><td>alt-click</td><td>place cursor B on a panel</td></tr>
      <tr><td>shift-drag / ctrl-drag</td><td>force a Y-only zoom on that panel (does not affect other panels)</td></tr>
      <tr><td>shift-wheel</td><td>zoom that panel's Y axis about the pointer</td></tr>
      <tr><td>shift-0</td><td>reset the Y axis of the panel under the mouse to autoscale</td></tr>
      <tr><td>click a chip's color swatch</td><td>open the series style popover (color/width/dash/points/interp)</td></tr>
      <tr><td>dbl-click row</td><td>add signal(s) to the focused panel</td></tr>
      <tr><td>ctrl/cmd-click row</td><td>toggle row in picker selection</td></tr>
      <tr><td>click / ctrl-click / shift-click row</td><td>select one signal, toggle one, extend a range (escape deselects all)</td></tr>
      <tr><td>right-click selection / Shift+F10</td><td>derive a signal from two or more selected signals</td></tr>
      <tr><td>s</td><td>toggle a dot at every simulated sample (hidden while denser than 1 per 4 px)</td></tr>
      <tr><td>m</td><td>drop a named marker at the mouse X</td></tr>
      <tr><td>i</td><td>toggle the run-details popover (title/subtitle/fields), if this run has any</td></tr>
      <tr><td>Scope &#9662;</td><td>switch to a sibling scope (block) of this design, shown only when the page is part of a set; this does not navigate, so the address bar keeps naming the current page</td></tr>
      <tr><td>click A/B/marker value</td><td>type an exact position (SI suffixes accepted)</td></tr>
      <tr><td>dbl-click marker badge/name</td><td>rename the marker</td></tr>
      <tr><td>&darr; / &uarr; (picker focused)</td><td>move the picker focus cursor</td></tr>
      <tr><td>&rarr; / &larr; (picker focused)</td><td>expand/first-child &nbsp;/&nbsp; collapse/parent</td></tr>
      <tr><td>Enter (picker focused)</td><td>add the focused row/subtree to the focused panel</td></tr>
      <tr><td>Space (picker focused)</td><td>toggle the focused row in the selection</td></tr>
      <tr><td>Home / End (picker focused)</td><td>first / last visible picker row</td></tr>
    </table>
  </div>
</div>
<div data-silk="fatal-banner" hidden></div>
`;
const mountedRoots = new WeakSet();
let nextInstance = 0;

// ======================================================================
// Trace identity and "Copy as silk code" (TASK-375). DOM-free and exported
// so Node tests exercise the same functions the context menu calls.
// ======================================================================

/** Ids of the explorer's own trace context-menu items; a host may hide them. */
export const TRACE_MENU_ITEMS = Object.freeze(['copy-silk-code']);

// Plotnames as both simulators write them, in silk's analysis words. Mirrors
// silk.signals.analysis_kind.plot_silk_analysis: the plotname first, then the
// axis/dtype fallback of rawfile_analysis_kind.
const PLOTNAME_ANALYSIS = {
  'transient analysis': 'tran', 'ac analysis': 'ac',
  'dc transfer characteristic': 'dc', 'operating point': 'op',
};

/** silk analysis word ('tran', 'ac', 'dc', 'noise', 'op') for a payload plot, or null. */
export function plotAnalysis(plot) {
  if (!plot) return null;
  var plotname = String(plot.name || '').trim().toLowerCase();
  if (Object.prototype.hasOwnProperty.call(PLOTNAME_ANALYSIS, plotname)) return PLOTNAME_ANALYSIS[plotname];
  if (plotname.indexOf('noise') !== -1) return 'noise';
  if (plotname === 'constants' || plot.kind === 'constants') return null;
  if (plot.complex) return 'ac';
  if (plot.npoints === 1) return 'op';
  var axis = String((plot.x && plot.x.name) || '').toLowerCase();
  if (!axis) return null;
  if (axis.indexOf('freq') !== -1) return 'noise';
  if (axis === 'time') return 'tran';
  return 'dc';
}

/**
 * The tb.probe spelling that reads simulator vector `vector`, or a reason.
 *
 * Mirrors NgspiceSignal.from_rawfile's matching (case-insensitive, Xyce ':'
 * hierarchy read as '.') and ModuleTestbench.probe's parsing, so the probe
 * resolves to this exact vector. A branch current uses the raw-vector form
 * probe('<branch>', 'I'): probe('i(<name>)') maps a source NAME to its
 * emitted instance and could read a different branch.
 */
export function probeFor(vector, part) {
  var name = typeof vector === 'string' ? vector.trim() : '';
  var transform = part === 'mag' ? 'mag' : part === 'phase' ? 'phase_deg_unwrapped' : null;
  if (part != null && !transform) return { probe: null, reason: 'unknown complex part ' + JSON.stringify(part) };
  if (!name) return { probe: null, reason: 'the trace has no simulator vector name' };
  var lower = name.toLowerCase();
  var wrapped = /^([vi])\((.*)\)$/.exec(lower);
  if (wrapped) {
    var inner = wrapped[2].trim().replace(/:/g, '.');
    if (!inner || /[(),@#\s'"\\]/.test(inner)) {
      return { probe: null, reason: 'tb.probe has no spelling for the vector ' + name };
    }
    return { probe: { name: inner, type: wrapped[1] === 'i' ? 'I' : 'V', transform: transform } };
  }
  if (/^[io]noise/.test(lower)) {
    return { probe: null, reason: name + ' is a noise vector; tb.probe names nets and branches, not noise spectra' };
  }
  var branch = /^([^#]+)#branch$/.exec(lower);
  if (branch && !/[():@\s'"\\]/.test(branch[1])) {
    return { probe: { name: lower, type: 'I', transform: transform } };
  }
  if (/[():@#\[\]\s'"\\,]/.test(lower)) {
    return { probe: null, reason: name + ' is a simulator-internal vector with no tb.probe net path' };
  }
  return { probe: { name: lower, type: 'V', transform: transform } };
}

function vectorOf(sig) {
  if (sig.group) return sig.group;
  return sig.renamed ? sig.name.replace(/#\d+$/, '') : sig.name;
}

/**
 * Trace identity record for signal `name` of plot `plotIdx` in `payload`
 * (a format-1 payload, or {run, plots} whose plots also hold the explorer's
 * derived signals). JSON-safe and freshly built on every call; the fields
 * are documented in docs/explorer-integration.md.
 */
export function traceIdentityOf(payload, plotIdx, name, seen) {
  var plots = payload && payload.plots, p = plots && plots[plotIdx];
  if (!Number.isInteger(plotIdx) || !p) throw new Error('plot ' + plotIdx + ' does not exist');
  var sig = null;
  for (var i = 0; i < p.signals.length; i++) if (p.signals[i].name === name) { sig = p.signals[i]; break; }
  var wholeVector = false;
  if (!sig && p.complex) {
    sig = p.signals.find(function (s) { return s.group === name && s.part === 'mag'; });
    wholeVector = !!sig;
  }
  if (!sig) throw new Error('unknown signal "' + name + '" in plot ' + plotIdx);
  var run = payload.run || {};
  var analysis = plotAnalysis(p);
  var record = {
    identity: 1, kind: sig.derived ? 'derived' : 'raw', name: sig.name, unit: sig.unit || '',
    source: {
      run_id: run.id == null ? null : run.id,
      run_source: run.source == null ? null : run.source,
      run_title: run.title == null ? null : run.title,
      plot: plotIdx, plot_name: p.name || '', analysis: analysis,
    },
  };
  if (sig.derived) {
    seen = seen || new Set();
    if (seen.has(sig.name)) throw new Error('derived signal "' + sig.name + '" refers to itself');
    seen.add(sig.name);
    if (sig.group) record.part = wholeVector ? null : sig.part;
    if (sig.derived.given_name !== undefined) record.given_name = sig.derived.given_name;
    record.operation = sig.derived.operation;
    record.parameters = sig.derived.params ? Object.assign({}, sig.derived.params) : {};
    record.operands = sig.derived.operands.map(function (operand) {
      return traceIdentityOf(payload, plotIdx, operand, seen);
    });
    seen.delete(sig.name);
    return record;
  }
  var vector = vectorOf(sig);
  record.vector = vector;
  record.part = wholeVector ? null : sig.part || null;
  var mapped = sig.role === 'coordinate'
    ? { probe: null, reason: 'this source row is an analysis coordinate, not a voltage net or branch' }
    : probeFor(vector, record.part);
  // TASK-530: Copy-as-code never reads a label. A signal's proven `address`
  // (designer-mode emission only: the bracket spelling the reader resolves to
  // exactly this source row) goes through the same probeFor formatter, under
  // the same renamed/duplicate guards below; without one the raw path stands.
  if (mapped.probe && sig.address && !sig.renamed) {
    var addressed = probeFor(sig.address, record.part);
    if (addressed.probe) mapped = addressed;
  }
  if (sig.role !== 'coordinate' && Object.prototype.hasOwnProperty.call(sig, 'node_probe')) {
    var node = sig.node_probe;
    if (node === null) mapped = {probe:null, reason:'no unique checked node probe for this source row'};
    else if (!node || typeof node.name !== 'string' || !['V', 'I'].includes(node.type) ||
             !node.name.startsWith(node.type.toLowerCase() + '(') || !node.name.endsWith(')')) {
      mapped = {probe:null, reason:'invalid checked node probe'};
    } else if (record.part == null || record.part === 'mag' || record.part === 'phase') {
      mapped = {probe:{name:node.name, type:node.type, transform:record.part === 'mag' ? 'mag' :
        record.part === 'phase' ? 'phase_deg_unwrapped' : null}};
    }
  }
  var duplicate = p.signals.some(function (other) {
    return other !== sig && !other.derived && other.renamed && vectorOf(other) === vector;
  });
  if (mapped.probe && (sig.renamed || duplicate)) {
    mapped = { probe: null, reason: 'the run holds more than one vector named ' + vector +
      ', so a tb.probe of that name cannot single this one out' };
  }
  if (mapped.probe && analysis === null && (p.kind === 'constants' || String(p.name || '').toLowerCase() === 'constants')) {
    mapped = { probe: null, reason: 'the constants plot holds simulator constants, not simulation output' };
  }
  // select_plot refuses a multi-plot rawfile without analysis=.
  if (mapped.probe) {
    if (plots.length > 1 && !analysis) {
      mapped = { probe: null, reason: 'plot "' + (p.name || plotIdx) + '" is not a silk analysis, so analysis= cannot select it' };
    } else {
      mapped.probe.analysis = plots.length > 1 ? analysis : null;
    }
  }
  record.probe = mapped.probe;
  record.probe_reason = mapped.probe ? null : mapped.reason;
  return record;
}

function pyString(text) {
  return "'" + String(text).replace(/\\/g, '\\\\').replace(/'/g, "\\'")
    .replace(/[\x00-\x1f\x7f]/g, c => c === '\n' ? '\\n' : c === '\r' ? '\\r' :
      c === '\t' ? '\\t' : '\\x' + c.charCodeAt(0).toString(16).padStart(2, '0')) + "'";
}

function pyNumber(value) {
  if (typeof value !== 'number' || !isFinite(value)) throw new Error('parameter ' + value + ' is not a finite number');
  return String(value);
}

// Precedence: 3 atom/call, 2 product/quotient, 1 sum/difference.
function renderIdentity(identity, path) {
  if (!identity || typeof identity !== 'object') throw new Error('not a trace identity record');
  if (identity.kind === 'raw') {
    if (!identity.probe) {
      throw new Error((identity.probe_reason || 'no tb.probe equivalent') +
        (path.length ? ' (operand ' + JSON.stringify(identity.name) + ')' : ''));
    }
    var p = identity.probe, args = [pyString(p.name)];
    if (p.type === 'I') args.push("'I'");
    if (p.analysis) args.push('analysis=' + pyString(p.analysis));
    return { code: 'tb.probe(' + args.join(', ') + ')' + (p.transform ? '.' + p.transform + '()' : ''), level: 3 };
  }
  if (identity.kind === 'derived' && identity.part) {
    var projection = identity.part === 'mag' ? 'mag' : 'phase_deg_unwrapped';
    var term = renderIdentity(Object.assign({}, identity, { part: null }), path);
    return { code: '(' + term.code + ').' + projection + '()', level: 3 };
  }
  if (identity.kind !== 'derived') throw new Error('unknown trace kind ' + JSON.stringify(identity.kind));
  var operands = (identity.operands || []).map(function (operand) {
    return renderIdentity(operand, path.concat([identity.name]));
  });
  function wrap(term, min) { return term.level >= min ? term.code : '(' + term.code + ')'; }
  var op = identity.operation, params = identity.parameters || {};
  if (op === 'sum' || op === 'mean') {
    var total = operands.map(function (t) { return wrap(t, 1); }).join(' + ');
    return op === 'sum' ? { code: total, level: 1 } : { code: '(' + total + ') / ' + operands.length, level: 2 };
  }
  if (op === 'difference') return { code: wrap(operands[0], 1) + ' - ' + wrap(operands[1], 2), level: 1 };
  if (op === 'product') return { code: operands.map(function (t) { return wrap(t, 2); }).join(' * '), level: 2 };
  if (op === 'quotient') return { code: wrap(operands[0], 2) + ' / ' + wrap(operands[1], 3), level: 2 };
  if (op === 'abs') return { code: wrap(operands[0], 3) + '.mag()', level: 3 };
  if (op === 'derivative' || op === 'integral') return { code: wrap(operands[0], 3) + '.' + op + '()', level: 3 };
  if (op === 'scale') {
    var gain = params.gain == null ? 1 : params.gain, offset = params.offset == null ? 0 : params.offset;
    var term = operands[0];
    if (gain !== 1) term = { code: pyNumber(gain) + ' * ' + wrap(term, 3), level: 2 };
    if (offset !== 0) {
      term = { code: wrap(term, 1) + (offset < 0 ? ' - ' + pyNumber(-offset) : ' + ' + pyNumber(offset)), level: 1 };
    }
    return term;
  }
  throw new Error('operation ' + JSON.stringify(op) + ' has no silk Python form');
}

/**
 * Runnable silk Python for a trace identity record: `{ok: true, code}` or
 * `{ok: false, reason}` when the trace (or one of its operands) has no
 * tb.probe equivalent. Never guesses a net.
 */
export function silkCode(identity) {
  try {
    return { ok: true, code: renderIdentity(identity, []).code };
  } catch (error) {
    return { ok: false, reason: error.message };
  }
}

function validateContextMenu(option) {
  if (option == null) return { items: [], hide: [] };
  if (typeof option !== 'object' || Array.isArray(option)) {
    throw new TypeError('contextMenu must be an object {items?, hide?}');
  }
  Object.keys(option).forEach(function (key) {
    if (key !== 'items' && key !== 'hide') throw new TypeError('contextMenu has no option "' + key + '"; use items and hide');
  });
  var hide = option.hide == null ? [] : option.hide;
  if (!Array.isArray(hide)) throw new TypeError('contextMenu.hide must be an array of built-in item ids');
  hide.forEach(function (id) {
    if (TRACE_MENU_ITEMS.indexOf(id) === -1) {
      throw new TypeError('contextMenu.hide: unknown built-in item ' + JSON.stringify(id) +
        '; built-in ids are ' + TRACE_MENU_ITEMS.join(', '));
    }
  });
  var items = option.items == null ? [] : option.items;
  if (!Array.isArray(items)) throw new TypeError('contextMenu.items must be an array of {label, onSelect}');
  var ids = new Set();
  items = items.map(function (item, i) {
    if (!item || typeof item !== 'object') throw new TypeError('contextMenu.items[' + i + '] must be an object');
    Object.keys(item).forEach(function (key) {
      if (['id', 'label', 'onSelect'].indexOf(key) === -1) {
        throw new TypeError('contextMenu.items[' + i + '] has no field "' + key + '"; use id, label and onSelect');
      }
    });
    if (typeof item.label !== 'string' || !item.label.trim()) {
      throw new TypeError('contextMenu.items[' + i + '].label must be a nonempty string');
    }
    if (typeof item.onSelect !== 'function') throw new TypeError('contextMenu.items[' + i + '].onSelect must be a function');
    var id = item.id == null ? 'host-' + i : item.id;
    if (typeof id !== 'string' || !id) throw new TypeError('contextMenu.items[' + i + '].id must be a nonempty string');
    if (TRACE_MENU_ITEMS.indexOf(id) !== -1) {
      throw new TypeError('contextMenu.items[' + i + '].id ' + JSON.stringify(id) +
        ' is a built-in item; built-in items cannot be replaced or renamed, only hidden');
    }
    if (ids.has(id)) throw new TypeError('contextMenu.items: duplicate id ' + JSON.stringify(id));
    ids.add(id);
    return { id: id, label: item.label, onSelect: item.onSelect };
  });
  return { items: items, hide: hide.slice() };
}

/** Mount an explorer. See docs/explorer-integration.md for options and lifecycle. */
export function createExplorer(root, options = {}) {
  if (!root || root.nodeType !== 1 || !root.ownerDocument.defaultView) {
    throw new TypeError('createExplorer requires a container element in a browser document');
  }
  if (mountedRoots.has(root)) throw new Error('An explorer is already mounted in this container');
  if (typeof options.uPlot !== 'function') throw new TypeError('Pass the uPlot constructor as options.uPlot');
  if (options.stateKey != null && (typeof options.stateKey !== 'string' || !options.stateKey)) {
    throw new TypeError('stateKey must be a nonempty host-supplied string');
  }
  var contextMenu = validateContextMenu(options.contextMenu);
  var document = root.ownerDocument, window = document.defaultView;
  var uPlot = options.uPlot;
  var transport = options.transport || null;
  var bridge = transport || { data: function () {}, stamp: function () {} };
  var initialized = false, destroyed = false;
  var syncKey = 'silk-explorer-' + (++nextInstance) + '-' + Math.random().toString(36).slice(2) + '-';
  var signalDragType = 'application/x-' + syncKey;
  var timers = new Set(), frames = new Set(), dragListeners = new Set();
  var pendingLoads = new Set(), readers = new Set(), objectURLs = new Set();
  var events = new window.AbortController();
  var previousChildren = Array.from(root.childNodes);
  var previousClass = root.getAttribute('class'), previousTabindex = root.getAttribute('tabindex');
  var previousFontSize = root.style.getPropertyValue('--fs-base');
  var previousFontPriority = root.style.getPropertyPriority('--fs-base');
  root.classList.add('silk-explorer');
  if (!root.hasAttribute('tabindex')) root.tabIndex = 0;
  root.innerHTML = EXPLORER_HTML;
  mountedRoots.add(root);

  function theme(name, fallback) {
    return window.getComputedStyle(root).getPropertyValue(name).trim() || fallback;
  }
  function stateKeyFor(payload) {
    return options.stateKey != null ? options.stateKey : (options.standaloneStateKey ? options.standaloneStateKey(payload) : null);
  }
  function listen(target, type, fn, capture) {
    target.addEventListener(type, fn, { capture: !!capture, signal: events.signal });
  }
  function listenDrag(type, fn) {
    var handler = function (event) { if (!destroyed) fn(event); };
    dragListeners.add({ type: type, fn: fn, handler: handler });
    window.addEventListener(type, handler);
  }
  function removeDragListener(type, fn) {
    dragListeners.forEach(function (entry) {
      if (entry.type === type && entry.fn === fn) {
        window.removeEventListener(type, entry.handler);
        dragListeners.delete(entry);
      }
    });
  }
  function cancelDrag() {
    dragListeners.forEach(function (entry) { window.removeEventListener(entry.type, entry.handler); });
    dragListeners.clear();
  }
  function setTimeout(fn, delay) {
    var id = window.setTimeout(function () { timers.delete(id); if (!destroyed) fn(); }, delay);
    timers.add(id);
    return id;
  }
  function clearTimeout(id) { window.clearTimeout(id); timers.delete(id); }
  function requestFrame(fn) {
    var id = window.requestAnimationFrame(function () { frames.delete(id); if (!destroyed) fn(); });
    frames.add(id);
    return id;
  }
  function makeObjectURL(blob) {
    var url = window.URL.createObjectURL(blob); objectURLs.add(url); return url;
  }
  function revokeObjectURL(url) { window.URL.revokeObjectURL(url); objectURLs.delete(url); }
  function validatePayload(payload) {
    if (!payload || (payload.format !== 1 && payload.format !== 2) || !payload.run || !Array.isArray(payload.plots) || !payload.plots.length) {
      throw new TypeError('Explorer requires a format 1 or 2 waveform payload with a run and at least one plot');
    }
    if (!payload.plots[payload.default_plot]) throw new TypeError('Waveform default_plot must name an existing plot');
    payload.plots.forEach(function (plot, index) {
      if (!plot || !Array.isArray(plot.signals)) {
        throw new TypeError('Waveform plot ' + index + ' signals must be an array');
      }
      if (Object.prototype.hasOwnProperty.call(plot, 'node_paths')) {
        const meta = validateNodePaths(plot.node_paths);
        for (const sig of plot.signals) {
          const row = sig.source_row;
          const terms = sig.node_probe?.terms;
          if (terms !== undefined) {
            if (row !== undefined || sig.node_probe.type !== 'I' || vectorOf(sig) !== sig.node_probe.name || !Array.isArray(terms) || !terms.length ||
                !terms.every(term => Number.isSafeInteger(term.row) && term.row >= 0 &&
                  term.row < meta.vectors.length && meta.vectors[term.row].selectable &&
                  (term.sign === 1 || term.sign === -1))) {
              throw new TypeError('invalid source-current terms');
            }
          } else if (!Number.isSafeInteger(row) || row < 0 || row >= meta.vectors.length ||
              vectorOf(sig) !== meta.vectors[row].name) throw new TypeError('node_paths source_row mismatch');
          if (Object.prototype.hasOwnProperty.call(sig, 'node_probe') && sig.node_probe !== null) {
            const probe = sig.node_probe;
            if ((!terms && !meta.vectors[row].selectable) || !probe || !['V', 'I'].includes(probe.type) ||
                typeof probe.name !== 'string' || !probe.name.startsWith(probe.type.toLowerCase() + '(') || !probe.name.endsWith(')')) {
              throw new TypeError('invalid node_probe');
            }
          }
        }
      } else if (plot.signals.some(sig => Object.prototype.hasOwnProperty.call(sig, 'node_probe') ||
                                  Object.prototype.hasOwnProperty.call(sig, 'source_row'))) {
        throw new TypeError('node_probe/source_row requires node_paths');
      }
    });
  }
  listen(root, 'mousedown', function (event) {
    var focusable = event.target.closest('input,textarea,select,button,a,[contenteditable="true"],[tabindex]');
    if (!focusable || focusable === root) {
      root.focus({ preventScroll: true });
    }
  }, true);
  listen(root, 'focusout', function () {
    requestFrame(function () { if (!root.contains(document.activeElement)) closeAllDropdowns(); });
  });


  // ======================================================================
  // Small helpers
  // ======================================================================

  function el(id) { return root.querySelector('[data-silk="' + id + '"]'); }

  function ce(tag, cls) {
    var e = document.createElement(tag);
    if (tag === 'button') e.type = 'button';
    if (cls) e.className = cls;
    return e;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function setText(node, text) {
    clear(node);
    node.appendChild(document.createTextNode(text == null ? '' : String(text)));
  }

  function debounce(ms, fn) {
    var t = null;
    return function () {
      var args = arguments, ctx = this;
      if (t) clearTimeout(t);
      t = setTimeout(function () { t = null; fn.apply(ctx, args); }, ms);
    };
  }

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  function bsearchLeft(arr, val) {
    // first index i such that arr[i] >= val
    var lo = 0, hi = arr.length;
    while (lo < hi) {
      var mid = (lo + hi) >>> 1;
      if (arr[mid] < val) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  function bsearchRight(arr, val) {
    // first index i such that arr[i] > val
    var lo = 0, hi = arr.length;
    while (lo < hi) {
      var mid = (lo + hi) >>> 1;
      if (arr[mid] <= val) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  function bsearchNearest(arr, val) {
    if (arr.length === 0) return -1;
    var i = bsearchLeft(arr, val);
    if (i <= 0) return 0;
    if (i >= arr.length) return arr.length - 1;
    return (val - arr[i - 1] <= arr[i] - val) ? i - 1 : i;
  }

  function isFiniteNum(v) {
    return v === v && v !== Infinity && v !== -Infinity;
  }

  // Small Levenshtein distance, used only for near-miss suggestions (command
  // names, signal names) -- registries and per-plot signal lists are small
  // (tens to a few hundred entries), so an O(n*m) DP per candidate is cheap.
  function levenshtein(a, b) {
    var al = a.length, bl = b.length;
    if (al === 0) return bl;
    if (bl === 0) return al;
    var prev = new Array(bl + 1);
    var cur = new Array(bl + 1);
    for (var j = 0; j <= bl; j++) prev[j] = j;
    for (var i = 1; i <= al; i++) {
      cur[0] = i;
      var ca = a.charCodeAt(i - 1);
      for (j = 1; j <= bl; j++) {
        var cost = ca === b.charCodeAt(j - 1) ? 0 : 1;
        cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
      }
      var tmp = prev; prev = cur; cur = tmp;
    }
    return prev[bl];
  }

  // Up to `n` closest candidates to `name` by Levenshtein distance,
  // case-insensitive, ties broken by original order. Used for unknown
  // command / unknown signal error messages (design doc section 4.2, 4.3).
  function nearMisses(name, candidates, n) {
    var q = String(name || '').toLowerCase();
    var scored = candidates.map(function (c, i) {
      return { c: c, d: levenshtein(q, String(c).toLowerCase()), i: i };
    });
    scored.sort(function (x, y) { return x.d - y.d || x.i - y.i; });
    return scored.slice(0, n == null ? 3 : n).map(function (s) { return s.c; });
  }

  // djb2, 8 lowercase hex chars -- the sidecar-precedence hash (design doc
  // section 3.4). Not cryptographic; a cheap, deterministic change-detector
  // over an embedded view-state's JSON text.
  function djb2Hash8(text) {
    var h = 5381;
    for (var i = 0; i < text.length; i++) {
      h = ((h * 33) ^ text.charCodeAt(i)) >>> 0;
    }
    var hex = (h >>> 0).toString(16);
    while (hex.length < 8) hex = '0' + hex;
    return hex.slice(-8);
  }

  // ======================================================================
  // hierParse -- section 2.2 of the v1.3 design doc, normative. Splits a
  // (possibly wrapped) signal name into {segments, leaf, wrapper}. Does not
  // look at the tree at all; pure per-signal string parsing.
  // ======================================================================

  // TASK-530 (owner D1): the caller's [k] label is the PRIMARY text wherever
  // one exists; the raw name stays the identity (state, annotations,
  // measurements, tooltips) and is shown as secondary text.
  function labelParse(sig) {
    return sig.label ? hierParse({ name: sig.label, part: sig.part }) : null;
  }
  function designerSegment(text, on) {
    return on ? String(text).replace(/</g, '[').replace(/>/g, ']') : text;
  }
  // alias (lower case) -> Set of distinct raw segments (lower case) of a plot
  function segmentAliases(plot) {
    var n = plot.signals.length;
    if (plot.__segAliases && plot.__segAliases.n === n) return plot.__segAliases.map;
    var map = {};
    plot.signals.forEach(function (sg) {
      if (sg.derived || (sg.node_probe && sg.node_probe.terms)) return;
      hierParse(sg).segments.forEach(function (seg) {
        var key = designerSegment(seg, true).toLowerCase();
        (map[key] = map[key] || {})[seg.toLowerCase()] = true;
      });
    });
    plot.__segAliases = { n: n, map: map };
    return map;
  }
  // A group's own segment is shown in [k] only if one of its leaves carries a
  // label AND its alias names no other segment of the plot (its own collision
  // status, not a plot-wide switch); otherwise the raw segment stays.
  function groupText(row, plot) {
    if (!plot || !row.node || !String(row.label).match(/[<>]/)) return row.label;
    var labelled = leafDescendantsOf(row.node).some(function (sg) { return sg.label; });
    if (!labelled) return row.label;
    var aliases = segmentAliases(plot);
    return String(row.label).split(' / ').map(function (seg) {
      var alias = designerSegment(seg, true);
      var owners = aliases[alias.toLowerCase()] || {};
      var distinct = Object.keys(owners);
      return (distinct.length > 1 || (distinct.length === 1 && distinct[0] !== seg.toLowerCase())) ? seg : alias;
    }).join(' / ');
  }
  function leafDescendantsOf(node, out) {
    out = out || [];
    if (node.sig) out.push(node.sig);
    for (var i = 0; i < node.children.length; i++) {
      var ch = node.children[i];
      if (ch.type === 'leaf') out.push(ch.sig); else leafDescendantsOf(ch, out);
    }
    return out;
  }

  // ======================================================================
  // Fmt -- engineering-prefix number formatting, 4 significant digits.
  // ======================================================================

  var Fmt = (function () {
    var PREFIXES = [
      [1e-24, 'y'], [1e-21, 'z'], [1e-18, 'a'], [1e-15, 'f'], [1e-12, 'p'],
      [1e-9, 'n'], [1e-6, 'µ'], [1e-3, 'm'], [1, ''],
      [1e3, 'k'], [1e6, 'M'], [1e9, 'G'], [1e12, 'T'], [1e15, 'P'],
      [1e18, 'E'], [1e21, 'Z'], [1e24, 'Y']
    ];

    function pickPrefix(av) {
      // av > 0, finite. Default to the SMALLEST prefix (yocto), not '' --
      // a value below 1e-24 (e.g. an intentionally-extreme test signal)
      // must still clamp into the table rather than silently falling back
      // to an unscaled x1 display.
      var idx = 0;
      for (var i = PREFIXES.length - 1; i >= 0; i--) {
        if (av >= PREFIXES[i][0]) { idx = i; break; }
      }
      return idx;
    }

    function sig4(x) {
      if (x === 0) return '0';
      var av = Math.abs(x);
      var exp = Math.floor(Math.log10(av));
      var decimals = clamp(3 - exp, 0, 10);
      var s = x.toFixed(decimals);
      return s;
    }

    function num(v, unit) {
      if (v == null) return '—'; // em dash: no data / gap
      if (!isFiniteNum(v)) {
        if (v !== v) return 'NaN';
        return v > 0 ? '+Inf' : '-Inf';
      }
      unit = unit || '';
      if (v === 0) return '0' + (unit ? ' ' + unit : '');
      var neg = v < 0;
      var av = Math.abs(v);
      var pidx = pickPrefix(av);
      var scale = PREFIXES[pidx][0];
      var prefix = PREFIXES[pidx][1];
      var scaled = av / scale;
      var s = sig4(scaled);
      return (neg ? '-' : '') + s + ' ' + prefix + unit;
    }

    function plain(v, decimals) {
      if (v == null) return '—';
      if (!isFiniteNum(v)) return v !== v ? 'NaN' : (v > 0 ? '+Inf' : '-Inf');
      return v.toFixed(decimals == null ? 3 : decimals);
    }

    // SI-suffix parser for typed cursor/marker placement (v1.2 item 4).
    // Case-insensitive. SPICE convention, deliberately NOT the naive
    // "M=mega" reading some non-SPICE tools use: a single letter m/M always
    // means milli; mega is spelled out as the three-letter suffix "meg"
    // (any case). g/G=giga and t/T=tera are unambiguous either way. The
    // table mirrors silk.utils._utils._parse_si (the Python-side contract)
    // with two typed-input differences: 'µ' is accepted because Fmt.num
    // writes it (this is display input, never netlist text), and an unknown
    // trailing letter is refused rather than ignored, so a typo does not
    // become a cursor position.
    var SI_SUFFIX = {
      'a': 1e-18, 'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'µ': 1e-6,
      'm': 1e-3, 'k': 1e3, 'meg': 1e6, 'g': 1e9, 't': 1e12
    };
    var SI_RE = /^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Zµ]*)\s*$/;
    // Round-trip guard (TASK-26). Fmt.num above writes SI for humans: an
    // uppercase 'M' is mega and 'P' is peta in a readout, but SPICE reads
    // them as milli and pico -- so a readout copied back in ("2.500 MHz")
    // would silently land 1e9 too small. Refuse exactly those two uppercase
    // letters (the red flash + tooltip say to type 'meg' or 'm'); every
    // other suffix means the same thing in both conventions.
    var DISPLAY_ONLY_UPPER = /^[MP](?![Ee][Gg])/;

    function parseSI(text) {
      if (text == null) return null;
      var m = SI_RE.exec(String(text));
      if (!m) return null;
      var num = parseFloat(m[1]);
      if (!isFiniteNum(num)) return null;
      if (DISPLAY_ONLY_UPPER.test(m[2])) return null;
      var suffix = m[2].toLowerCase();
      if (suffix === '') return num;
      if (!Object.prototype.hasOwnProperty.call(SI_SUFFIX, suffix)) return null;
      return num * SI_SUFFIX[suffix];
    }

    return { num: num, plain: plain, parseSI: parseSI };
  })();

  // ======================================================================
  // FontScale -- item 6 (UX round addendum). ONE base text size
  // (compact/normal/large, mapped to a px value) that every CSS font-size
  // in view.css derives from via `calc(var(--fs-base) +/- Npx)` (a 4-tier
  // ladder: base / -1 / -2 / -3px), applied by writing an inline
  // `--fs-base` override onto <html> -- the cheapest way to make the WHOLE
  // stylesheet re-derive at once, no per-rule JS needed.
  //
  // Canvas text (axis labels, cursor/marker tags, PNG export titles) does
  // NOT live in the stylesheet at all, so it cannot pick this up for free;
  // `mono()`/`css()` below mirror the same tiering for `ctx.font` strings,
  // and callers must re-read them at draw time (never cache a font string
  // across a scale change). `PanelManager.rebuildAllForFontChange()` forces
  // that re-read for the live charts by rebuilding every open panel.
  // ======================================================================

  var FontScale = (function () {
    var SCALES = { compact: 13, normal: 14, large: 17 }; // px, matches --fs-base's own default (14)
    var MONO = '"SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace';
    var current = 'normal';

    function basePx() { return SCALES[current] || SCALES.normal; }

    // An explicit pixel size (caller has already applied any DPR scaling
    // for a PNG export) -> a full canvas font string, floored so nothing
    // ever collapses to an unreadable point size.
    function css(px, bold) {
      var size = Math.max(9, Math.round(px));
      return (bold ? 'bold ' : '') + size + 'px ' + MONO;
    }

    // An offset from the CURRENT base (mirrors the CSS ladder's tiers,
    // e.g. -2 matches the stylesheet's "chips/tables" tier) -> a font string.
    function mono(offset, bold) { return css(basePx() + (offset || 0), bold); }

    function apply() {
      root.style.setProperty('--fs-base', basePx() + 'px');
    }

    function set(key) {
      if (!Object.prototype.hasOwnProperty.call(SCALES, key)) key = 'normal';
      if (key === current) return;
      current = key;
      apply();
      PanelManager.rebuildAllForFontChange(); // no-op if no panels are mounted yet (boot)
    }

    function get() { return current; }
    function isValid(key) { return Object.prototype.hasOwnProperty.call(SCALES, key); }

    return { set: set, get: get, apply: apply, mono: mono, css: css, basePx: basePx, isValid: isValid };
  })();

  // Shared by the toolbar select's change handler and the setFontScale
  // command (§4.3 pattern: every new user action gets a command, and here
  // the command and the UI control should do the exact same three things).
  function applyFontScale(key) {
    FontScale.set(key);
    var sel = el('font-scale-select');
    if (sel) sel.value = FontScale.get();
    var st = StateStore.get();
    if (st) { st.fontScale = FontScale.get(); StateStore.save(); }
  }

  // ======================================================================
  // ValueEditor -- click-to-type inline editor for A/B cursor and marker
  // readout values, with SI-suffix parsing (Fmt.parseSI) and a red flash on
  // invalid input that leaves the previous value untouched (v1.2 item 4).
  // ======================================================================

  var ValueEditor = (function () {
    var SI_TITLE = 'Click to type an exact value. SI suffixes: a f p n u/µ m k meg g t ' +
      '(SPICE convention: m = milli, meg = mega; an uppercase M or P is refused ' +
      'because readouts use them for mega/peta). Snaps to the nearest sample.';

    // opts: {getValue(): number|null, setValue(number), formatFn(number|null): string}
    function attach(span, opts) {
      span.classList.add('editable-value');
      span.title = SI_TITLE;

      function render() {
        clear(span);
        span.appendChild(document.createTextNode(opts.formatFn(opts.getValue())));
      }

      function open() {
        if (span.querySelector('input')) return;
        var current = opts.getValue();
        var input = ce('input', 'value-edit-input');
        input.type = 'text';
        input.value = current == null ? '' : String(current);
        input.title = SI_TITLE;
        clear(span);
        span.appendChild(input);
        input.focus();
        input.select();

        function commit() {
          var parsed = Fmt.parseSI(input.value);
          if (parsed == null || !isFiniteNum(parsed)) {
            input.classList.add('invalid');
            setTimeout(function () { if (input.classList) input.classList.remove('invalid'); }, 500);
            return;
          }
          opts.setValue(parsed);
          // The caller's setValue normally triggers a readout/table rebuild
          // that replaces this span's contents; render() here is a no-op
          // safety net for callers that don't rebuild.
          render();
        }

        input.addEventListener('keydown', function (ev) {
          if (ev.key === 'Enter') { ev.preventDefault(); commit(); }
          else if (ev.key === 'Escape') { ev.preventDefault(); render(); }
        });
        input.addEventListener('blur', function () {
          if (span.contains(input)) render();
        });
        input.addEventListener('click', function (e) { e.stopPropagation(); });
      }

      span.addEventListener('click', function (e) {
        e.stopPropagation();
        open();
      });
      render();
      return span;
    }

    return { attach: attach };
  })();

  // ======================================================================
  // inlineTextEdit -- shared rename-in-place helper (marker names).
  // ======================================================================

  function inlineTextEdit(container, initialText, onCommit) {
    var input = ce('input', 'inline-edit-input');
    input.type = 'text';
    input.value = initialText;
    clear(container);
    container.appendChild(input);
    input.focus();
    input.select();
    function commit() { onCommit(input.value.trim()); }
    function cancel() { onCommit(null); }
    input.addEventListener('keydown', function (ev) {
      if (ev.key === 'Enter') { ev.preventDefault(); commit(); }
      else if (ev.key === 'Escape') { ev.preventDefault(); cancel(); }
    });
    input.addEventListener('blur', commit);
    input.addEventListener('click', function (e) { e.stopPropagation(); });
  }

  // ======================================================================
  // Dropdown menus -- shared open/close for the toolbar Export menu and each
  // panel's per-panel export menu (v1.2 items 1-2).
  // ======================================================================

  function closeAllDropdowns(except) {
    var menus = root.querySelectorAll('.dropdown-menu');
    for (var i = 0; i < menus.length; i++) {
      if (menus[i] !== except) menus[i].hidden = true;
    }
  }

  // `anchorBtn` is optional: when given, the menu is treated as a
  // root-appended `.anchored-menu` (position:absolute) and repositioned under
  // that button before it opens -- see the `.anchored-menu` comment in
  // view.css for why the toolbar's own menus need this (§toolbar
  // overflow-y:hidden clips an in-place `.dropdown-menu` invisibly). Menus
  // that are still nested in-place (the per-panel export menu, which fits
  // inside its always-tall-enough `.panel`) omit it and keep the
  // stylesheet's default `top:calc(100% + 4px); right:0` positioning.
  // `align` is forwarded to positionAnchoredMenu ('right', the default, or
  // 'left'). Content is built and the menu unhidden BEFORE positioning so
  // `offsetWidth` reflects the real rendered width for the overflow clamp
  // -- there is no visible flash, since nothing repaints mid-synchronous-
  // JS.
  function toggleDropdown(menu, buildFn, anchorBtn, align) {
    var willOpen = menu.hidden;
    closeAllDropdowns(menu);
    if (willOpen) {
      clear(menu);
      buildFn(menu);
      menu.hidden = false;
      if (anchorBtn) positionAnchoredMenu(menu, anchorBtn, align);
    } else {
      menu.hidden = true;
    }
  }

  // Root-relative coordinates keep overlays inside their instance, including
  // when the host is offset, scrolled, or smaller than the viewport.
  function positionAnchoredMenu(menu, btn, align) {
    var r = btn.getBoundingClientRect(), bounds = root.getBoundingClientRect();
    var width = menu.offsetWidth || 200;
    var left = (align === 'left' ? r.left : r.right - width) - bounds.left - root.clientLeft;
    menu.style.left = Math.max(4, Math.min(left, root.clientWidth - width - 4)) + 'px';
    menu.style.right = 'auto';
    menu.style.top = Math.max(4, Math.min(r.bottom - bounds.top - root.clientTop + 4,
      root.clientHeight - menu.offsetHeight - 4)) + 'px';
  }

  function menuButton(menu, text, title, onClick, disabled) {
    var b = ce('button');
    setText(b, text);
    if (title) b.title = title;
    if (disabled) b.disabled = true;
    b.addEventListener('click', function (e) {
      e.stopPropagation();
      menu.hidden = true;
      if (!disabled) onClick();
    });
    menu.appendChild(b);
    return b;
  }

  function menuHeading(menu, text) {
    var h = ce('div', 'dropdown-heading');
    setText(h, text);
    menu.appendChild(h);
  }

  function menuSep(menu) {
    menu.appendChild(ce('div', 'dropdown-sep'));
  }

  // ======================================================================
  // Base64 / typed-array decode
  // ======================================================================

  function b64ToBuffer(b64) {
    if (!b64) return new ArrayBuffer(0);
    var bin = atob(b64);
    var len = bin.length;
    var buf = new ArrayBuffer(len);
    var view = new Uint8Array(buf);
    for (var i = 0; i < len; i++) view[i] = bin.charCodeAt(i);
    return buf;
  }

  function decodeF64(b64) {
    if (ArrayBuffer.isView(b64)) return b64;
    return new Float64Array(b64ToBuffer(b64));
  }
  function decodeY(b64, dt) {
    if (ArrayBuffer.isView(b64)) return b64;
    var buf = b64ToBuffer(b64);
    return dt === 'f64' ? new Float64Array(buf) : new Float32Array(buf);
  }

  function nullifyNonFinite(typedArr) {
    var out = new Array(typedArr.length);
    for (var i = 0; i < typedArr.length; i++) {
      var v = typedArr[i];
      out[i] = isFiniteNum(v) ? v : null;
    }
    return out;
  }

  function checkLittleEndian() {
    var buf = new ArrayBuffer(2);
    new DataView(buf).setInt16(0, 256, true);
    return new Int16Array(buf)[0] === 256;
  }

  // ======================================================================
  // Store -- decoded payload + lazy per-(plot,signal,stride) decode cache.
  // ======================================================================

  var Store = (function () {
    var plots = [];
    var definitions = new Map();
    var complexValues = new WeakMap();
    var cache = new Map(); // key -> {x: Float64Array, y: TypedArray|Array, n}

    function init(p) {
      // Runtime name ownership cannot be imported in a waveform payload.
      // Saved definitions must replay the derive_signal name argument instead.
      if (p) p.plots.forEach(function (plot) {
        plot.signals.forEach(function (sig) {
          if (sig.derived && sig.derived.given_name !== undefined) {
            throw new Error('Waveform payload signal ' + JSON.stringify(sig.name) +
              ' must not supply given_name; use deriveSignal or restore saved view definitions');
          }
        });
      });
      // View objects belong to this instance. Never append browser arithmetic
      // to a host-owned payload (which may also be loaded by another instance).
      plots = p ? p.plots.map(function (plot) {
        return Object.assign({}, plot, { signals: plot.signals.slice() });
      }) : [];
      definitions.clear();
      complexValues = new WeakMap();
      cache.clear();
    }

    function plot(idx) { return plots[idx]; }

    function xLevel0(plotIdx) {
      var key = 'X0|' + plotIdx;
      var hit = cache.get(key);
      if (hit) return hit;
      var p = plot(plotIdx);
      var arr = decodeF64(p.x.levels[0].x);
      cache.set(key, arr);
      return arr;
    }

    function rawYLevel0(plotIdx, sig) {
      var key = 'Y0|' + plotIdx + '|' + sig.name;
      var hit = cache.get(key);
      if (hit) return hit;
      var lvl = sig.levels[0];
      var arr = decodeY(lvl.y, lvl.dt);
      cache.set(key, arr);
      return arr;
    }

    // Returns {x: Float64Array, y: (TypedArray|Array), n} for uPlot use,
    // where y has been nullified for uPlot gap semantics iff the signal
    // carries any nonfinite values.
    function levelFor(plotIdx, sig, stride) {
      var key = 'L|' + plotIdx + '|' + sig.name + '|' + stride;
      var hit = cache.get(key);
      if (hit) return hit;
      var lvl = null;
      for (var i = 0; i < sig.levels.length; i++) {
        if (sig.levels[i].stride === stride) { lvl = sig.levels[i]; break; }
      }
      if (!lvl) lvl = sig.levels[0];
      var x = (lvl.stride === 1) ? xLevel0(plotIdx) : decodeF64(lvl.x);
      var yRaw = decodeY(lvl.y, lvl.dt);
      var y = sig.nonfinite > 0 ? nullifyNonFinite(yRaw) : yRaw;
      var rec = { x: x, y: y, yRaw: yRaw, n: lvl.n };
      cache.set(key, rec);
      return rec;
    }

    function strides(sig) {
      var s = [];
      for (var i = 0; i < sig.levels.length; i++) s.push(sig.levels[i].stride);
      return s;
    }

    // Exact stats over the visible viewport [x0, x1], from level 0 only.
    function stats(plotIdx, sig, x0, x1) {
      var xArr = xLevel0(plotIdx);
      if (xArr.length === 0) return null;
      var yArr = rawYLevel0(plotIdx, sig);
      var i0 = bsearchLeft(xArr, Math.min(x0, x1));
      var i1 = bsearchRight(xArr, Math.max(x0, x1));
      var min = Infinity, max = -Infinity, sum = 0, sumsq = 0, count = 0;
      for (var i = i0; i < i1; i++) {
        var v = yArr[i];
        if (!isFiniteNum(v)) continue;
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
        sumsq += v * v;
        count++;
      }
      if (!count) return null;
      var mean = sum / count;
      return { min: min, max: max, pp: max - min, mean: mean, rms: Math.sqrt(sumsq / count) };
    }

    function sampleAt(plotIdx, sig, xval) {
      var xArr = xLevel0(plotIdx);
      if (xArr.length === 0) return null;
      var i = bsearchNearest(xArr, xval);
      var yArr = rawYLevel0(plotIdx, sig);
      var v = yArr[i];
      return isFiniteNum(v) ? v : null;
    }

    function findSignal(plotIdx, name) {
      var p = plot(plotIdx);
      for (var i = 0; i < p.signals.length; i++) {
        if (p.signals[i].name === name) return p.signals[i];
      }
      return null;
    }

    // A display part names its entire vector when used as an operand.
    // Keep findSignal exact: drawing, stats and cursor reads still use parts.
    function operand_signal(plotIdx, name) {
      var p = plot(plotIdx), sig = findSignal(plotIdx, name);
      var group = sig && sig.group ? sig.group : name;
      if (p.complex) {
        var parts = p.signals.filter(function (s) { return s.group === group; });
        if (parts.length) {
          var mags = parts.filter(function (s) { return s.part === 'mag'; });
          var phases = parts.filter(function (s) { return s.part === 'phase'; });
          if (mags.length !== 1 || phases.length !== 1) {
            throw new Error('complex vector "' + group + '" needs one magnitude and one phase part');
          }
          return mags[0];
        }
      }
      return sig;
    }

    function complex_samples(plotIdx, sig) {
      // Keep derived rectangular samples: unwrapped display phase can contain
      // a NaN tail after a singularity even when later complex values are valid.
      if (complexValues.has(sig)) return complexValues.get(sig);
      var re = rawYLevel0(plotIdx, sig), im = new Float64Array(re.length);
      if (!sig.group) return { re: re, im: im };
      var phase = plot(plotIdx).signals.find(function (s) { return s.group === sig.group && s.part === 'phase'; });
      var angles = rawYLevel0(plotIdx, phase), mag = re;
      if (angles.length !== mag.length) throw new Error('complex parts must match the shared x array length');
      re = new Float64Array(mag.length);
      for (var i = 0; i < re.length; i++) {
        var angle = angles[i] * (Math.PI / 180);
        re[i] = mag[i] * Math.cos(angle);
        im[i] = mag[i] * Math.sin(angle);
      }
      return { re: re, im: im };
    }

    function complex_trace(operation, x, arrays, params) {
      var first = arrays[0], n = x.length;
      if (operation === 'abs') return { re: Float64Array.from(first.re, function (r, i) {
        return Math.hypot(r, first.im[i]);
      }), im: new Float64Array(n) };
      if (SINGLE_TRACE_OPERATIONS.indexOf(operation) !== -1) {
        return { re: single_trace(operation, x, first.re, params),
          im: single_trace(operation, x, first.im,
            operation === 'scale' ? { gain: params.gain, offset: 0 } : params) };
      }
      var re = new Float64Array(n), im = new Float64Array(n);
      for (var i = 0; i < n; i++) {
        var r = first.re[i], v = first.im[i];
        for (var j = 1; j < arrays.length; j++) {
          var a = arrays[j].re[i], b = arrays[j].im[i], next;
          if (operation === 'product') {
            next = r * a - v * b; v = r * b + v * a; r = next;
          } else if (operation === 'quotient') {
            // Ratio form avoids squaring very large/small denominator values.
            var ratio, denominator;
            if (a === 0 && b === 0) {
              next = r / 0; v = v / 0;
            } else if (Math.abs(a) >= Math.abs(b)) {
              ratio = b / a; denominator = a + b * ratio;
              next = (r + v * ratio) / denominator;
              v = (v - r * ratio) / denominator;
            } else {
              ratio = a / b; denominator = b + a * ratio;
              next = (r * ratio + v) / denominator;
              v = (v * ratio - r) / denominator;
            }
            r = next;
          } else {
            var sign = operation === 'difference' ? -1 : 1;
            r += sign * a; v += sign * b;
          }
        }
        re[i] = operation === 'mean' ? r / arrays.length : r;
        im[i] = operation === 'mean' ? v / arrays.length : v;
      }
      return { re: re, im: im };
    }

    function complex_parts(values) {
      var mag = new Float64Array(values.re.length), phase = new Float64Array(mag.length);
      var previous, correction = 0;
      for (var i = 0; i < mag.length; i++) {
        mag[i] = Math.hypot(values.re[i], values.im[i]);
        var angle = Math.atan2(values.im[i], values.re[i]);
        // numpy.unwrap propagates a nonfinite phase through the remaining tail.
        if (previous !== undefined) {
          var delta = angle - previous;
          var remainder = (delta + Math.PI) % (2 * Math.PI);
          if (remainder < 0) remainder += 2 * Math.PI;
          var wrapped = remainder - Math.PI;
          if (wrapped === -Math.PI && delta > 0) wrapped = Math.PI;
          correction += Math.abs(delta) < Math.PI ? 0 : wrapped - delta;
        }
        phase[i] = (previous === undefined ? angle : angle + correction) * (180 / Math.PI);
        previous = angle;
      }
      return { mag: mag, phase: phase };
    }

    // Exponents in voltage/current/time/angle/temperature. Keep dimensions on
    // derived view objects even when their display unit is outside the payload
    // vocabulary: (V * V) / V must recover V. An ordinary empty unit is unknown,
    // whereas an internally derived dimensionless ratio has all-zero exponents.
    var unit_dimensions = new Map([
      ['V', [1, 0, 0, 0, 0]], ['A', [0, 1, 0, 0, 0]], ['W', [1, 1, 0, 0, 0]],
      ['C', [0, 1, 1, 0, 0]], ['F', [-1, 1, 1, 0, 0]], ['s', [0, 0, 1, 0, 0]],
      ['Hz', [0, 0, -1, 0, 0]], ['Ω', [1, -1, 0, 0, 0]], ['S', [-1, 1, 0, 0, 0]],
      ['deg', [0, 0, 0, 1, 0]], ['°C', [0, 0, 0, 0, 1]],
    ]);

    function named_unit(dimension) {
      var result = '';
      unit_dimensions.forEach(function (d, unit) {
        if (d.every(function (value, index) { return value === dimension[index]; })) result = unit;
      });
      return result;
    }

    function derived_unit(operation, signals, xUnit) {
      var units = signals.map(function (sig) { return sig.unit || ''; });
      var dimensions = signals.map(function (sig) {
        return sig.derived ? sig.derived.dimensions : unit_dimensions.get(sig.unit) || null;
      });
      // gain is a plain number, as in silk's gain * a + offset, so scale and
      // abs keep the operand's quantity.
      if (operation === 'scale' || operation === 'abs') {
        return { unit: units[0], dimensions: dimensions[0] ? dimensions[0].slice() : null };
      }
      if (operation === 'derivative' || operation === 'integral') {
        var x_dimension = unit_dimensions.get(xUnit) || null;
        if (!dimensions[0] || !x_dimension) return { unit: '', dimensions: null };
        var step = operation === 'derivative' ? -1 : 1;
        var per_x = dimensions[0].map(function (value, index) { return value + step * x_dimension[index]; });
        var shown = named_unit(per_x);
        if (!shown && units[0]) shown = units[0] + (operation === 'derivative' ? '/' : '·') + xUnit;
        return { unit: shown, dimensions: per_x };
      }
      if (['sum', 'difference', 'mean'].indexOf(operation) !== -1) {
        if (!units.every(function (unit) { return unit === units[0]; })) {
          throw new Error(operation + ' requires operands with the same unit; received ' +
            signals.map(function (sig, i) { return sig.name + ' [' + (units[i] || 'no unit') + ']'; }).join(', '));
        }
        // Acceptance keys on literal units, including ''. If equal empty units
        // hide different dimensions, accept the arithmetic but claim no dimension.
        var same = dimensions[0] && dimensions.every(function (d) {
          return d && d.every(function (value, i) { return value === dimensions[0][i]; });
        });
        return { unit: units[0], dimensions: same ? dimensions[0].slice() : null };
      }
      var dimension = [0, 0, 0, 0, 0];
      for (var i = 0; i < dimensions.length; i++) {
        var d = dimensions[i];
        if (!d) return { unit: '', dimensions: null };
        var sign = operation === 'quotient' && i === 1 ? -1 : 1;
        for (var j = 0; j < dimension.length; j++) dimension[j] += sign * d[j];
      }
      return { unit: named_unit(dimension), dimensions: dimension };
    }

    // silk's Signal operation list (TASK-373): the same names, operand counts,
    // parameters and maths as silk/signals/base.py, so a viewer trace equals
    // what a test recomputes.
    var MULTI_TRACE_OPERATIONS = ['sum', 'difference', 'product', 'quotient', 'mean'];
    var SINGLE_TRACE_OPERATIONS = ['scale', 'abs', 'derivative', 'integral'];
    var SCALE_DEFAULTS = { gain: 1.0, offset: 0.0 };

    function derive_params(operation, params) {
      if (params === undefined || params === null) params = {};
      if (typeof params !== 'object' || Array.isArray(params)) throw new Error('params must be an object');
      var keys = Object.keys(params);
      if (operation !== 'scale') {
        if (keys.length) throw new Error(operation + ' takes no parameters; received ' + keys.join(', '));
        return null;
      }
      var result = Object.assign({}, SCALE_DEFAULTS);
      keys.forEach(function (key) {
        if (!Object.prototype.hasOwnProperty.call(SCALE_DEFAULTS, key)) {
          throw new Error('scale accepts gain and offset; unknown parameter "' + key + '"');
        }
        if (typeof params[key] !== 'number' || !isFinite(params[key])) {
          throw new Error('scale ' + key + ' must be a finite number; received ' + JSON.stringify(params[key]));
        }
        result[key] = params[key];
      });
      return result;
    }

    // numpy.gradient(y, x) with first-order edges: the uniform-spacing branch
    // when every step is equal, the second-order non-uniform stencil otherwise.
    function gradient(x, y) {
      var n = y.length, out = new Float64Array(n), i;
      var dx = new Float64Array(n - 1);
      for (i = 0; i < n - 1; i++) dx[i] = x[i + 1] - x[i];
      var uniform = true;
      for (i = 1; i < n - 1; i++) if (dx[i] !== dx[0]) { uniform = false; break; }
      for (i = 1; i < n - 1; i++) {
        if (uniform) {
          out[i] = (y[i + 1] - y[i - 1]) / (2 * dx[0]);
        } else {
          var h1 = dx[i - 1], h2 = dx[i];
          var a = -h2 / (h1 * (h1 + h2)), b = (h2 - h1) / (h1 * h2), c = h1 / (h2 * (h1 + h2));
          out[i] = a * y[i - 1] + b * y[i] + c * y[i + 1];
        }
      }
      out[0] = (y[1] - y[0]) / dx[0];
      out[n - 1] = (y[n - 1] - y[n - 2]) / dx[n - 2];
      return out;
    }

    function single_trace(operation, x, values, params) {
      var n = values.length, y, i;
      if (operation === 'derivative') return gradient(x, values);
      y = new Float64Array(n);
      if (operation === 'scale') {
        for (i = 0; i < n; i++) y[i] = params.gain * values[i] + params.offset;
      } else if (operation === 'abs') {
        for (i = 0; i < n; i++) y[i] = Math.abs(values[i]);
      } else {
        // Cumulative trapezoid with y[0] = 0, as Signal.integral.
        var total = 0;
        for (i = 1; i < n; i++) {
          total += 0.5 * (values[i] + values[i - 1]) * (x[i] - x[i - 1]);
          y[i] = total;
        }
      }
      return y;
    }

    function format_param(value) { return String(value); }

    function depends_on(plotIdx, sig, definition, seen) {
      if (!sig || !sig.derived) return false;
      if (sig.derived === definition) return true;
      if (seen.has(sig)) return false;
      seen.add(sig);
      return sig.derived.operands.some(function (name) {
        return depends_on(plotIdx, operand_signal(plotIdx, name), definition, seen);
      });
    }

    // refresh is private: replay an existing descendant's unchanged definition
    // and preserve its display name, including legacy unnamed collision suffixes.
    function derive_signal(plotIdx, operation, operands, params, givenName, replace, refresh) {
      if (givenName !== undefined && (typeof givenName !== 'string' || !givenName.trim())) {
        throw new Error('deriveSignal name must be a non-empty string');
      }
      if (replace !== undefined && typeof replace !== 'boolean') {
        throw new Error('deriveSignal replace must be a boolean');
      }
      var p = plot(plotIdx);
      if (!Number.isInteger(plotIdx) || !p) throw new Error('plot ' + plotIdx + ' does not exist');
      if (p.kind === 'constants') throw new Error('deriveSignal requires a waveform plot with a shared x array');
      var single = SINGLE_TRACE_OPERATIONS.indexOf(operation) !== -1;
      if (!single && MULTI_TRACE_OPERATIONS.indexOf(operation) === -1) {
        throw new Error('operation must be sum, difference, product, quotient, mean, scale, abs, derivative or integral');
      }
      if (!Array.isArray(operands) || !operands.every(function (name) { return typeof name === 'string'; })) {
        throw new Error('operands must be an array of existing signal names (strings) in one plot');
      }
      var ordered = operation === 'difference' || operation === 'quotient';
      if (single ? operands.length !== 1 : ordered ? operands.length !== 2 : operands.length < 2) {
        throw new Error(operation + ' requires ' + (single ? 'exactly one' : ordered ? 'exactly two' : 'at least two') + ' operand' +
          (single ? '' : 's') + '; received ' + operands.length);
      }
      params = derive_params(operation, params);
      // Canonical ordering makes identity and floating-point evaluation stable
      // for commutative operations, preserving repeated operands in the list.
      var names = operands.map(function (name) {
        var sig = operand_signal(plotIdx, name);
        if (!sig) throw new Error('unknown signal "' + name + '" in plot ' + plotIdx + '; choose operands from this plot');
        return p.complex && sig.group ? sig.group : sig.name;
      });
      if (!ordered) names.sort();
      var signals = names.map(function (name) {
        var sig = operand_signal(plotIdx, name);
        if (!sig) throw new Error('unknown signal "' + name + '" in plot ' + plotIdx + '; choose operands from this plot');
        return sig;
      });
      var quantity = derived_unit(operation, signals, (p.x && p.x.unit) || ''), unit = quantity.unit;
      var key = [plotIdx, operation, names, params];
      var previous = refresh || null;
      if (givenName !== undefined) {
        var collisions = p.signals.filter(function (sig) {
          return sig.name === givenName || sig.group === givenName ||
            sig.name === givenName + ' (mag)' || sig.name === givenName + ' (phase)' ||
            sig.group === givenName + ' (mag)' || sig.group === givenName + ' (phase)' ||
            (sig.derived && sig.derived.given_name !== undefined &&
              (givenName === sig.derived.given_name + ' (mag)' || givenName === sig.derived.given_name + ' (phase)'));
        });
        previous = collisions.find(function (sig) {
          return sig.derived && sig.derived.given_name === givenName;
        });
        var other = collisions.find(function (sig) { return !previous || sig.derived !== previous.derived; });
        function clash(sig) {
          throw new Error('deriveSignal name ' + JSON.stringify(givenName) +
            ' conflicts with existing signal or trace ' + JSON.stringify(sig.name) + ' in plot ' + plotIdx);
        }
        if (other) clash(other);
        if (previous) {
          var old = previous.derived;
          if (!refresh && JSON.stringify(key) === JSON.stringify([plotIdx, old.operation, old.operands, old.params])) {
            return { name: givenName, unit: previous.unit, created: false };
          }
          // Replace safety invariant: runtime definition.given_name enters only
          // through this function's name argument; restore replays it here.
          // A new naming path (UI rename, Copy-as-code, an import) must revisit
          // replace safety first. Never infer ownership from a display name.
          if (!replace) clash(previous);
          if (signals.some(function (sig) { return depends_on(plotIdx, sig, old, new Set()); })) {
            throw new Error('deriveSignal replacement ' + JSON.stringify(givenName) + ' would create cyclic operands');
          }
        }
        key.push(givenName);
      }
      var identity = JSON.stringify(key);
      var existing = definitions.get(identity);
      if (existing && !refresh) return { name: existing.name, unit: existing.unit, created: false };
      var core = options.waveformCore;
      if (!core || typeof core.build_pyramid !== 'function') {
        throw new Error('deriveSignal requires a matching waveformCore with build_pyramid; update the explorer and core assets together');
      }
      var x = xLevel0(plotIdx);
      // This is the only operand data access. Decimated extrema from different
      // signals occur at different instants and must NEVER be combined.
      var arrays = signals.map(function (sig) { return rawYLevel0(plotIdx, sig); });
      if (arrays.some(function (values) { return values.length !== x.length; })) {
        throw new Error('deriveSignal operands must match the shared x array length in plot ' + plotIdx);
      }
      if (single) {
        // Refuse rather than draw an empty trace: Signal.derivative raises
        // below two samples, and nothing is a quick look at zero samples.
        var minimum = operation === 'derivative' ? 2 : 1;
        if (x.length < minimum) {
          throw new Error(operation + ': signal "' + names[0] + '" has ' + x.length + ' sample' +
            (x.length === 1 ? '' : 's') + '; ' + operation + ' needs at least ' + minimum);
        }
      }
      var isComplex = p.complex && signals.some(function (sig) { return !!sig.group; });
      var complex = isComplex ? complex_trace(operation, x, signals.map(function (sig) {
        return complex_samples(plotIdx, sig);
      }), params) : null;
      var y = complex ? complex.re : single ? single_trace(operation, x, arrays[0], params) : new Float64Array(x.length);
      for (var i = 0; !complex && !single && i < y.length; i++) {
        var value = arrays[0][i];
        for (var j = 1; j < arrays.length; j++) {
          if (operation === 'product') value *= arrays[j][i];
          else if (operation === 'quotient') value /= arrays[j][i];
          else if (operation === 'difference') value -= arrays[j][i];
          else value += arrays[j][i];
        }
        y[i] = operation === 'mean' ? value / arrays.length : value;
      }
      var base = operation + '(' + names.join(', ') + (params ? Object.keys(params).map(function (key) {
        return ', ' + key + '=' + format_param(params[key]);
      }).join('') : '') + ')';
      var name = refresh ? (refresh.group || refresh.name) : givenName === undefined ? base : givenName, suffix = 2;
      if (givenName === undefined && !refresh) {
        while (operand_signal(plotIdx, name) || findSignal(plotIdx, name + ' (mag)') ||
            findSignal(plotIdx, name + ' (phase)')) name = base + ' [' + (suffix++) + ']';
      }
      var definition = { operation: operation, operands: names, params: params, dimensions: quantity.dimensions };
      if (givenName !== undefined) {
        definition.given_name = givenName;
        definition.expression = base;
      }
      var createdSignals = [];
      if (complex && operation !== 'abs') {
        var parts = complex_parts(complex);
        ['mag', 'phase'].forEach(function (part) {
          var derivedPart = Object.assign({ name: name + ' (' + part + ')',
            unit: part === 'mag' ? unit : 'deg', type: 'derived', part: part, group: name,
            derived: definition }, core.build_pyramid(x, parts[part]));
          if (part === 'mag') complexValues.set(derivedPart, complex);
          createdSignals.push(derivedPart);
        });
      } else {
        createdSignals.push(Object.assign({ name: name, unit: unit, type: 'derived', part: null, group: null,
          derived: definition }, core.build_pyramid(x, y)));
      }
      if (refresh) {
        var occupied = p.signals.find(function (sig) {
          return sig.derived !== previous.derived && (sig.name === name || sig.group === name ||
            createdSignals.some(function (next) { return next.name === sig.name || next.name === sig.group; }));
        });
        if (occupied) throw new Error('trace ' + JSON.stringify(name) +
          ' conflicts with existing signal or trace ' + JSON.stringify(occupied.name));
      }
      var transaction = null, descendants = [];
      if (previous && !refresh) {
        var affected = new Map();
        p.signals.forEach(function (sig) {
          if (sig.derived && sig.derived !== previous.derived && sig.part !== 'phase' &&
              depends_on(plotIdx, sig, previous.derived, new Set())) affected.set(sig.derived, sig);
        });
        var visited = new Set();
        function visit(sig) {
          if (visited.has(sig.derived)) return;
          visited.add(sig.derived);
          sig.derived.operands.forEach(function (operand) {
            var dependency = operand_signal(plotIdx, operand);
            if (dependency && affected.has(dependency.derived)) visit(affected.get(dependency.derived));
          });
          descendants.push(sig);
        }
        affected.forEach(visit);
        // No existing signal/definition objects are mutated. Keep the old maps
        // and signal array until every descendant has evaluated successfully.
        transaction = { signals: p.signals, definitions: definitions, cache: cache };
        definitions = new Map(definitions);
        cache = new Map(cache);
      }
      try {
        if (previous) {
          var at = p.signals.indexOf(previous);
          p.signals = p.signals.filter(function (sig) { return sig.derived !== previous.derived; });
          p.signals.splice.apply(p.signals, [at, 0].concat(createdSignals));
          definitions.forEach(function (entry, k) {
            if (JSON.parse(k)[0] === plotIdx && entry.name === (previous.group || previous.name)) definitions.delete(k);
          });
          cache.clear(); // decoded levels are keyed by name, which replacement preserves
        } else {
          p.signals.push.apply(p.signals, createdSignals);
        }
        definitions.set(identity, { name: name, unit: unit });
        descendants.forEach(function (sig) {
          var def = sig.derived;
          try {
            derive_signal(plotIdx, def.operation, def.operands, def.params, def.given_name, true, sig);
          } catch (error) {
            throw new Error('deriveSignal replacement ' + JSON.stringify(name) + ' failed for descendant ' +
              JSON.stringify(sig.group || sig.name) + ': ' + error.message);
          }
        });
      } catch (error) {
        if (transaction) {
          p.signals = transaction.signals;
          definitions = transaction.definitions;
          cache = transaction.cache;
        }
        throw error;
      }
      return { name: name, unit: unit, created: true };
    }

    function derived_for_persist(plotIdx) {
      return plot(plotIdx).signals.filter(function (sig) { return sig.derived && sig.part !== 'phase'; }).map(function (sig) {
        var def = { name: sig.group || sig.name, operation: sig.derived.operation, operands: sig.derived.operands.slice() };
        if (sig.derived.given_name !== undefined) def.given_name = sig.derived.given_name;
        if (sig.derived.params) def.params = Object.assign({}, sig.derived.params);
        return def;
      });
    }

    function restore_derived(savedPlots) {
      Object.keys(savedPlots).forEach(function (key) {
        var saved = savedPlots[key];
        if (!saved || saved.derived === undefined) return; // additive: older views have no definitions
        var plotIdx = Number(key), pending = [], seen = new Set(), failed = new Set();
        function dropped(name, reason) {
          failed.add(name);
          showToast('Saved derived signal ' + name + ' in plot ' + key + ' dropped: ' + reason);
        }
        function available(name) { return !failed.has(name) && operand_signal(plotIdx, name); }
        if (!Array.isArray(saved.derived)) {
          dropped('(definitions)', 'expected an array');
        } else if (!Number.isInteger(plotIdx) || !plot(plotIdx)) {
          if (saved.derived.length) dropped('(definitions)', 'plot is not in this run');
        } else {
          saved.derived.forEach(function (def) {
            if (!def || typeof def.name !== 'string' || !def.name ||
                (def.given_name !== undefined && def.given_name !== def.name) ||
                !Array.isArray(def.operands) || !def.operands.every(function (n) { return typeof n === 'string'; })) {
              dropped(def && typeof def.name === 'string' ? def.name : '(unnamed)', 'invalid definition');
            } else if (seen.has(def.name) || operand_signal(plotIdx, def.name)) {
              dropped(def.name, 'name already exists');
            } else {
              seen.add(def.name);
              pending.push(def);
            }
          });
          // Repeated passes give an explicit dependency order even for shuffled
          // view files. No progress means missing operands or a dependency cycle.
          while (pending.length) {
            var waiting = [];
            pending.forEach(function (def) {
              if (failed.has(def.name)) return;
              if (!def.operands.every(available)) {
                waiting.push(def);
                return;
              }
              try {
                var result = derive_signal(plotIdx, def.operation, def.operands, def.params, def.given_name);
                if (!result.created && result.name !== def.name) {
                  dropped(def.name, 'duplicate definition of ' + result.name);
                  return;
                }
                // Saved names are references in panels and later definitions.
                // A changed payload can change the engine's collision suffix.
                var restored = operand_signal(plotIdx, result.name);
                if (restored.group) {
                  plot(plotIdx).signals.forEach(function (sig) {
                    if (sig.group === result.name) {
                      sig.group = def.name;
                      sig.name = def.name + ' (' + sig.part + ')';
                    }
                  });
                } else restored.name = def.name;
                definitions.forEach(function (entry, identity) {
                  if (JSON.parse(identity)[0] === plotIdx && entry.name === result.name) entry.name = def.name;
                });
              } catch (error) { dropped(def.name, error.message); }
            });
            if (waiting.length === pending.length) {
              waiting.forEach(function (def) {
                dropped(def.name, 'missing or cyclic operands: ' + def.operands.filter(function (n) {
                  return !available(n);
                }).join(', '));
              });
              break;
            }
            pending = waiting;
          }
        }
        saved.derived = plot(plotIdx) ? derived_for_persist(plotIdx) : [];
      });
    }

    // Nearest level-0 sample x-value to `xval` -- the snap-to-sample rule
    // (v1.2 item 4): cursor/marker placement always lands exactly on a
    // sample, so the position and the readout agree exactly.
    function snap(plotIdx, xval) {
      var xArr = xLevel0(plotIdx);
      if (!xArr.length || !isFiniteNum(xval)) return xval;
      return xArr[bsearchNearest(xArr, xval)];
    }

    return {
      init: init, plot: plot, xLevel0: xLevel0, levelFor: levelFor,
      strides: strides, stats: stats, sampleAt: sampleAt, findSignal: findSignal,
      rawYLevel0: rawYLevel0, operand_signal: operand_signal, snap: snap, derive_signal: derive_signal,
      derived_for_persist: derived_for_persist, restore_derived: restore_derived
    };
  })();

  // ======================================================================
  // Application state (not persisted directly; StateStore owns persistence)
  // ======================================================================

  var PAYLOAD = null;
  var CONFIG = { poll_ms: 2000, inline: false, data_src: 'data.js', stamp_src: 'stamp.js' };
  var activePlotIdx = 0;
  var xLogEnabled = false;

  // Trace identity of a signal as this instance holds it, derived signals
  // included (Store plots are per-instance copies of the payload plots).
  function traceIdentity(name, plotIdx) {
    if (!PAYLOAD) throw new Error('no waveform payload is loaded');
    var plots = PAYLOAD.plots.map(function (_, i) { return Store.plot(i); });
    return traceIdentityOf({ run: PAYLOAD.run, plots: plots }, plotIdx, name);
  }
  // TASK-530: the primary text of a trace is its [k] label when one exists.
  function shownName(plotIdx, name) {
    var sg = Store.findSignal(plotIdx, name);
    return sg && sg.label ? sg.label : name;
  }
  function traceTitle(sig) {
    return sig && sig.derived && sig.derived.given_name !== undefined ? sig.derived.expression :
      sig ? sig.name : '';
  }
  var lastHoverPanel = null; // panel object the mouse most recently hovered

  // TASK-319: one collapsible section per signal kind, keyed on the
  // payload's sig.type, table-driven so a later kind is one entry here --
  // e.g. adding power would be one more line:
  //   { type: 'power', label: 'Powers', key: 'kind:power', defaultOpen: false },
  // Order is the on-screen section order (spec: Voltages, Currents, then
  // any OTHER kind -- which has no entry here, so it stays in the
  // ordinary, unsectioned tree exactly as before this feature -- then
  // Derived signals last, unconditionally). `key` doubles as the
  // expandedByPlot/toggleExpand pathKey, so a section's open/closed state
  // rides the SAME Set and toggle/expand/collapse code paths as an
  // ordinary tree group at runtime (simplest). Persistence is a separate
  // story (reviewer decision (c)): a section's open/closed state is saved
  // and restored through its OWN 'sections' layout field, never through
  // 'expanded' -- see restoreSections/getSectionsForPersist in Picker, and
  // defaultLayoutFor, which deliberately does NOT seed sections; a missing
  // 'sections' field (a brand-new plot, or a pre-TASK-319 saved layout) is
  // what makes restoreSections apply these defaultOpen values. Declared
  // here (outer createExplorer scope, not inside Picker) only because
  // groupSignalsByKind and rebuild()'s section-partitioning logic below
  // also need it and live in this same outer scope.
  var KIND_SECTIONS = [
    { type: 'voltage', label: 'Voltages', key: 'kind:voltage', defaultOpen: true },
    { type: 'current', label: 'Currents', key: 'kind:current', defaultOpen: false },
  ];
  // The escape hatch (AC#3): off restores today's single tree exactly, in
  // both the embedded API (a plain createExplorer option) and the
  // standalone page (see view.js, which forwards window.SILK_VIEW_CONFIG's
  // group_by_kind -- itself baked in by emit_view(group_signals_by_kind=)
  // via the same config channel poll_ms/inline already use).
  var groupSignalsByKind = options.groupSignalsByKind !== false;

  // view-context-and-scopes design §6 R2: a monotonically increasing
  // generation counter shared by every path that assigns
  // `window.__silkViewData` (the periodic Reloader AND ScopeManager both
  // do, since they are the same JSONP handshake against different sidecar
  // files). Each such attempt captures `++dataGen` before installing its
  // handler and re-checks it when the handler actually fires; a response
  // that lands after a *later* attempt has already won is silently
  // dropped instead of clobbering fresher state -- the guard against a
  // scope switch racing an in-flight reload (or a second scope switch
  // racing the first).
  var dataGen = 0;

  var PALETTE = [
    '#4fd1c5', '#e0a640', '#5b9bd5', '#e08a3c', '#9b7fe0',
    '#7fc97f', '#e0575a', '#c9c15a', '#5ad1a0', '#c98fd6'
  ].map(function (fallback, i) { return theme('--trace-' + (i + 1), fallback); });

  // ======================================================================
  // StateStore -- localStorage persistence, design doc 7.6.
  // ======================================================================

  var StateStore = (function () {
    var key = null;
    var state = null;

    // activePlot starts as null (not 0) so App.init can tell "no saved
    // state yet" apart from "the user explicitly left plot 0 active" --
    // 0 is falsy-adjacent but a valid index, so the distinction matters:
    // a fresh run must fall back to the payload's own default_plot.
    // treeMode: global, additive field (design doc v1.3 section 2.6),
    // default true. appliedViewState: additive bookkeeping field (section
    // 3.4) -- deliberately NOT given a default value here (absent means
    // "no sidecar has ever been applied to this state"); it is never part
    // of the canonical view shape ViewState.snapshot()/getState() return
    // (see ViewState module comment for why).
    function defaults() { return { v: 1, plots: {}, activePlot: null, filter: '', cols: 1, treeMode: true, fontScale: 'normal' }; }

    function load(stateKey) {
      key = stateKey == null ? null : 'silk.view.' + stateKey;
      if (!key) { state = defaults(); return state; }
      try {
        var raw = localStorage.getItem(key);
        if (!raw) { state = defaults(); return state; }
        var parsed = JSON.parse(raw);
        if (!parsed || parsed.v !== 1 || typeof parsed.plots !== 'object') {
          state = defaults();
          return state;
        }
        if (!parsed.plots) parsed.plots = {};
        if (typeof parsed.filter !== 'string') parsed.filter = '';
        if (parsed.cols !== 1 && parsed.cols !== 2) parsed.cols = 1; // additive field (§7.6); old saved state has none
        if (typeof parsed.treeMode !== 'boolean') parsed.treeMode = true; // additive field (v1.3); old saved state has none
        if (!FontScale.isValid(parsed.fontScale)) parsed.fontScale = 'normal'; // additive field (item 6); old saved state has none
        state = parsed;
        return state;
      } catch (e) {
        try { localStorage.removeItem(key); } catch (e2) { /* ignore */ }
        state = defaults();
        return state;
      }
    }

    function replace(newState) {
      state = newState;
      return state;
    }

    var saveNow = function () {
      if (!key || !state) return;
      try {
        localStorage.setItem(key, JSON.stringify(state));
      } catch (e) {
        // Quota or serialization error: never brick the page over persistence.
      }
    };
    var saveDebounced = debounce(400, saveNow);

    function get() { return state; }
    function getKey() { return key; }

    return {
      load: load, save: saveDebounced, saveNow: saveNow, get: get,
      defaults: defaults, replace: replace, getKey: getKey,
    };
  })();

  // ======================================================================
  // Toast + fatal banner
  // ======================================================================

  // Clipboard text with the textarea fallback for pages where the async
  // clipboard API is unavailable (file:// and other insecure contexts).
  function copyText(text, done) {
    function fallback() {
      if (destroyed) return;
      var ok = false;
      try {
        var ta = ce('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.left = '-9999px';
        root.appendChild(ta);
        ta.focus(); ta.select();
        ok = document.execCommand('copy');
        root.removeChild(ta);
      } catch (e) { ok = false; }
      showToast(ok ? done : 'clipboard unavailable here; the code is: ' + text);
    }
    var clipboard = window.navigator.clipboard;
    if (clipboard && clipboard.writeText) {
      clipboard.writeText(text).then(function () { showToast(done); }, fallback);
    } else {
      fallback();
    }
  }

  function showToast(text) {
    if (destroyed) return;
    var container = el('toast-container');
    var t = ce('div', 'toast');
    setText(t, text);
    container.appendChild(t);
    setTimeout(function () {
      if (t.parentNode) t.parentNode.removeChild(t);
    }, 4500);
  }

  // Toast with a clickable action link (the sidecar-precedence "undo",
  // design doc v1.3 section 3.4). Lives longer than a plain toast (9s vs
  // 4.5s) since it carries a decision, not just an FYI.
  function showToastAction(text, actionLabel, onAction) {
    if (destroyed) return;
    var container = el('toast-container');
    var t = ce('div', 'toast');
    t.appendChild(document.createTextNode(text + '  ·  '));
    var link = ce('span', 'toast-action');
    setText(link, actionLabel);
    link.addEventListener('click', function () {
      if (t.parentNode) t.parentNode.removeChild(t);
      onAction();
    });
    t.appendChild(link);
    container.appendChild(t);
    setTimeout(function () {
      if (t.parentNode) t.parentNode.removeChild(t);
    }, 9000);
  }

  function fatal(message) {
    if (destroyed) return;
    var b = el('fatal-banner');
    setText(b, message);
    b.hidden = false;
    el('app').style.display = 'none';
  }

  // ======================================================================
  // Annotations -- markers/bands/notes. Rendered against the active plot's
  // currently-open panels; x-axis markers render on every open panel of the
  // active plot (there is only ever one plot's panels mounted at a time).
  // ======================================================================

  var Annotations = (function () {
    function ann() { return (PAYLOAD && PAYLOAD.annotations) || {}; }

    function markersForPanel(panel) {
      var markers = ann().markers || [];
      var out = [];
      for (var i = 0; i < markers.length; i++) {
        var m = markers[i];
        if (m.axis === 'x') { out.push(m); continue; }
        if (m.signal == null || panel.signals.indexOf(m.signal) !== -1) out.push(m);
      }
      return out;
    }

    function bandsForPanel(panel) {
      var bands = ann().bands || [];
      var out = [];
      for (var i = 0; i < bands.length; i++) {
        var bnd = bands[i];
        if (bnd.signal == null || panel.signals.indexOf(bnd.signal) !== -1) out.push(bnd);
      }
      return out;
    }

    function notesForSignal(name) {
      var notes = ann().notes || [];
      var out = [];
      for (var i = 0; i < notes.length; i++) {
        if (notes[i].signal === name) out.push(notes[i]);
      }
      return out;
    }

    function severityColor(sev) {
      if (sev === 'error') return theme('--sev-error', '#e0575a');
      if (sev === 'warn') return theme('--sev-warn', '#e0a640');
      return theme('--sev-info', '#5b9bd5');
    }

    function drawBands(u, panel) {
      var bands = bandsForPanel(panel);
      if (!bands.length) return;
      var ctx = u.ctx;
      ctx.save();
      for (var i = 0; i < bands.length; i++) {
        var b = bands[i];
        if (b.axis !== 'y') continue;
        var scale = AxisModes.for_signal(panel, b.signal).key;
        var yLo = u.valToPos(b.lo, scale, true);
        var yHi = u.valToPos(b.hi, scale, true);
        var top = Math.min(yLo, yHi);
        var h = Math.abs(yHi - yLo);
        ctx.fillStyle = theme('--annotation-band', 'rgba(79, 209, 197, 0.10)');
        ctx.fillRect(u.bbox.left, top, u.bbox.width, h);
      }
      ctx.restore();
    }

    function drawMarkers(u, panel) {
      var markers = markersForPanel(panel);
      if (!markers.length) return;
      var ctx = u.ctx;
      ctx.save();
      ctx.font = FontScale.mono(-3); // item 6: badge-tier canvas text, was a fixed 10px
      for (var i = 0; i < markers.length; i++) {
        var m = markers[i];
        var color = m.color || theme('--annotation-marker', '#e0a640');
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.setLineDash([5, 3]);
        ctx.lineWidth = 1;
        if (m.axis === 'x') {
          var xpix = u.valToPos(m.value, 'x', true);
          if (xpix < u.bbox.left - 1 || xpix > u.bbox.left + u.bbox.width + 1) continue;
          ctx.beginPath();
          ctx.moveTo(xpix, u.bbox.top);
          ctx.lineTo(xpix, u.bbox.top + u.bbox.height);
          ctx.stroke();
          if (m.label) ctx.fillText(m.label, xpix + 3, u.bbox.top + 11);
        } else {
          var ypix = u.valToPos(m.value, AxisModes.for_signal(panel, m.signal).key, true);
          if (ypix < u.bbox.top - 1 || ypix > u.bbox.top + u.bbox.height + 1) continue;
          ctx.beginPath();
          ctx.moveTo(u.bbox.left, ypix);
          ctx.lineTo(u.bbox.left + u.bbox.width, ypix);
          ctx.stroke();
          if (m.label) ctx.fillText(m.label, u.bbox.left + 3, ypix - 3);
        }
      }
      ctx.setLineDash([]);
      ctx.restore();
    }

    return {
      markersForPanel: markersForPanel, bandsForPanel: bandsForPanel,
      notesForSignal: notesForSignal, severityColor: severityColor,
      drawBands: drawBands, drawMarkers: drawMarkers
    };
  })();

  // ======================================================================
  // CursorEngine -- A/B cursors, per plot.
  // ======================================================================

  var CursorEngine = (function () {
    var cursors = {}; // plotIdx -> {A: xval|null, B: xval|null}

    function get(plotIdx) {
      if (!cursors[plotIdx]) cursors[plotIdx] = { A: null, B: null };
      return cursors[plotIdx];
    }

    function set(which, plotIdx, xval) {
      // Snap-to-sample (v1.2 item 4): every placement path -- click, drag,
      // badge-drag, or typed entry -- lands on an actual level-0 sample, so
      // the drawn position and the sampled readout never disagree.
      get(plotIdx)[which] = Store.snap(plotIdx, xval);
      PanelManager.redrawPanels(plotIdx);
      updateReadouts(plotIdx);
      PanelManager.scheduleStateSave();
    }

    // Grab-handle badges are DOM elements (not canvas text) so they can be
    // dragged directly and read clearly against any trace color; the line
    // itself stays a canvas draw, matching every other overlay in this file.
    function draw(u, panel) {
      var c = get(panel.plotIdx);
      ['A', 'B'].forEach(function (which) {
        var xv = c[which];
        var badge = which === 'A' ? panel.badgeA : panel.badgeB;
        if (xv == null) {
          if (badge) badge.hidden = true;
          return;
        }
        var xpix = u.valToPos(xv, 'x', true);
        if (xpix >= u.bbox.left - 1 && xpix <= u.bbox.left + u.bbox.width + 1) {
          var ctx = u.ctx;
          ctx.save();
          ctx.strokeStyle = which === 'A' ? theme('--cursor-a', '#5b9bd5') : theme('--cursor-b', '#e08a3c');
          ctx.setLineDash([3, 3]);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(xpix, u.bbox.top);
          ctx.lineTo(xpix, u.bbox.top + u.bbox.height);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.restore();
        }
        if (badge && panel.overEl) {
          var xCss = u.valToPos(xv, 'x');
          var w = panel.overEl.clientWidth;
          if (xCss >= -14 && xCss <= w + 14) {
            badge.hidden = false;
            badge.style.left = xCss + 'px';
          } else {
            badge.hidden = true;
          }
        }
      });
    }

    function buildChip(which, plotIdx) {
      var chip = ce('span', 'cursor-chip cursor-chip-' + which.toLowerCase());
      var label = ce('span', 'cursor-chip-label');
      setText(label, which);
      chip.appendChild(label);
      var value = ce('span', 'cursor-chip-value');
      // click-to-type SI-suffixed exact placement (v1.2 item 4); the colored
      // label already says which cursor this is.
      ValueEditor.attach(value, {
        getValue: function () { return get(plotIdx)[which]; },
        setValue: function (v) { set(which, plotIdx, v); },
        formatFn: function (v) { return Fmt.num(v, xUnit()); }
      });
      chip.appendChild(value);
      return chip;
    }

    function updateReadouts(plotIdx) {
      var c = get(plotIdx);
      var out = el('cursor-readout');
      clear(out);
      if (c.A == null && c.B == null) return;
      if (c.A != null) out.appendChild(buildChip('A', plotIdx));
      if (c.B != null) out.appendChild(buildChip('B', plotIdx));
      if (c.A != null && c.B != null) {
        var dx = c.B - c.A;
        var extraText = 'Δx=' + Fmt.num(dx, xUnit());
        if (dx !== 0) extraText += '   1/Δx=' + Fmt.num(1 / dx, xUnit() ? '/' + xUnit() : '');
        var extra = ce('span', 'cursor-readout-extra');
        setText(extra, extraText);
        out.appendChild(extra);
      }
      PanelManager.updateAllMonitorTables(plotIdx);
    }

    function xUnit() {
      var p = Store.plot(activePlotIdx);
      return (p && p.x && p.x.unit) || '';
    }

    function clearAll(plotIdx) {
      cursors[plotIdx] = { A: null, B: null };
      updateReadouts(plotIdx);
    }

    function forPersist(plotIdx) {
      var c = get(plotIdx);
      var arr = [];
      if (c.A != null) arr.push(c.A);
      if (c.B != null) arr.push(c.B);
      return arr;
    }

    function restore(plotIdx, arr) {
      var c = get(plotIdx);
      c.A = (arr && arr.length > 0) ? arr[0] : null;
      c.B = (arr && arr.length > 1) ? arr[1] : null;
    }

    return {
      get: get, set: set, draw: draw, updateReadouts: updateReadouts,
      clearAll: clearAll, forPersist: forPersist, restore: restore
    };
  })();

  // ======================================================================
  // Monitors -- exact min/max/pp/mean/rms over the visible viewport, from
  // level 0 only. Debounced 120 ms after a scale change.
  // ======================================================================

  var Monitors = (function () {
    function computeForPanel(panel) {
      var out = {};
      var u = panel.uplot;
      if (!u) return out;
      var x0 = u.scales.x.min, x1 = u.scales.x.max;
      for (var i = 0; i < panel.signals.length; i++) {
        var name = panel.signals[i];
        var sig = Store.findSignal(panel.plotIdx, name);
        if (!sig) continue;
        out[name] = Store.stats(panel.plotIdx, sig, x0, x1);
      }
      return out;
    }

    return { computeForPanel: computeForPanel };
  })();

  // ======================================================================
  // Picker -- left sidebar: hierarchical tree (v1.3, design doc section 2)
  // over the instance paths already encoded in vector names, or a flat
  // list (the permanent "Flat" escape hatch, R4). Both render through the
  // same per-row builders, which is what makes the two modes byte-for-byte
  // identical on a plot with no dotted names (§7.2, R4).
  // ======================================================================

  var Picker = (function () {
    var ROW_H = 24;
    var filterText = '';
    var treeMode = true;
    var listEl = null;
    var spacer = null;
    var rowsHost = null;
    var selected = new Set(); // signal names currently multi-selected
    var selectionAnchor = null; // index into `visibleRows` for shift-range extend
    var lastPlotIdx = null;
    var derive_menu = null;

    var visibleRows = [];         // flattened rows currently rendered for the active plot
    var currentTreeKeys = null;   // Set of valid (post-compression) group path keys
    var expandedByPlot = {};      // plotIdx -> Set(pathKey), default-collapsed (§2.6)
    var focusIdxByPlot = {};      // plotIdx -> index into visibleRows (keyboard focus cursor)

    // KIND_SECTIONS / groupSignalsByKind (TASK-319) live in the outer
    // createExplorer scope, not here, since rebuild()'s section-partitioning
    // logic and view.js/emit_view's config channel need them too; see the
    // comment there. Their persisted open/closed state, however, is a
    // Picker-local concern -- restoreSections/getSectionsForPersist below.

    // ---- tree construction (design doc section 2.3) ----------------------

    // Item 5: a hierarchy node can be BOTH a signal and the parent of other
    // signals (a device like `...xmm1.msky130_fd_pr__nfet_01v8` is a signal
    // in its own right AND the parent of `.id`, `.rd`, `.1overf`, ...). The
    // naive walk below would create TWO sibling entries with the identical
    // label -- a leaf (no twisty) and a group (twisty) -- so a first click
    // on the leaf twin looked like expansion was broken. Whichever of the
    // two is inserted second is merged into the first instead of added as a
    // sibling: a 'leaf' entry gets converted in place into a 'group' entry
    // that also carries `sig`/`hp` (if the group arrives first), or an
    // existing 'group' entry just gets `sig`/`hp` attached (if the leaf
    // arrives second). Either way there is ONE node, type 'group', with an
    // optional `sig` marking it as also being a signal.
    // *signalsList*, when given, builds the tree over exactly those signals
    // instead of the whole plot (TASK-319: one kind-section's own subtree,
    // or the leftover "any other kind" signals once sectioned kinds are
    // pulled out). Omitted, it is plot.signals -- unchanged behavior for
    // every pre-existing call site.
    //
    // *keyPrefix*, when given, is prepended (as ``prefix + '/' + path``) to
    // every group's fullKey -- otherwise identical hierarchy under two
    // DIFFERENT sections (TASK-319: the same device name, e.g. "xdut", can
    // legitimately appear under both Voltages and Currents) would compute
    // the exact same bare fullKey ("xdut") for both and collide in the one
    // shared expandedByPlot Set, so expanding one section's group would
    // also silently expand the other's same-named group. Omitted (the
    // pre-existing, unsectioned call site), fullKey is unprefixed, unchanged.
    function buildTree(plot, signalsList, keyPrefix) {
      var root = { children: [] };
      if (!plot || plot.kind === 'constants') return root;
      var signals = signalsList || plot.signals;
      function fullKeyFor(path) { return (keyPrefix ? keyPrefix + '/' : '') + path.join('.'); }
      for (var i = 0; i < signals.length; i++) {
        var sig = signals[i];
        if (sig.derived) continue;
        var hp = hierParse(sig);
        if (sig.node_probe && sig.node_probe.terms) {
          var firstTerm = sig.node_probe.terms[0].row;
          hp = hierParse({name: plot.node_paths.vectors[firstTerm].name, type: 'current'});
          hp.leaf = hp.leaf.replace(/^@/, '').replace(/\[current\]$/i, '')
            .replace(/__silk_(?:wave|pwl|dc)$/i, '');
          if (/^(v|@i)$/i.test(hp.segments[0] || '')) hp.segments.shift();
        }
        // Checked physical source cards share one source parent. The raw
        // card name remains the selectable leaf and trace identity.
        if (sig.node_probe && sig.node_probe.type === 'I') {
          var cardLeaf = hp.leaf.replace(/^@/, '').replace(/\[current\]$/i, '');
          var sourceCard = /^(.*)(__silk_(?:wave|pwl|dc))$/i.exec(cardLeaf);
          if (sourceCard) {
            // Keep the raw instance keys shared with voltage rows. Native
            // source-current prefixes are not placement hierarchy.
            if (/^(v|@i)$/i.test(hp.segments[0] || '')) hp.segments.shift();
            hp.segments.push(sourceCard[1]);
          }
        }
        var node = root;
        var path = [];
        for (var s = 0; s < hp.segments.length; s++) {
          var seg = hp.segments[s];
          path.push(seg);
          var found = null;
          for (var c = 0; c < node.children.length; c++) {
            var ch = node.children[c];
            if (ch.type === 'group' && ch.segKey === seg) { found = ch; break; }
          }
          if (!found) {
            var leafIdx = -1;
            for (var lc = 0; lc < node.children.length; lc++) {
              if (node.children[lc].type === 'leaf' && node.children[lc].hp.leaf === seg) { leafIdx = lc; break; }
            }
            if (leafIdx !== -1) {
              var prior = node.children[leafIdx];
              found = { type: 'group', segKey: seg, fullKey: fullKeyFor(path), children: [], sig: prior.sig, hp: prior.hp };
              node.children[leafIdx] = found;
            } else {
              found = { type: 'group', segKey: seg, fullKey: fullKeyFor(path), children: [] };
              node.children.push(found);
            }
          }
          node = found;
        }
        var existingGroupIdx = -1;
        for (var g = 0; g < node.children.length; g++) {
          if (node.children[g].type === 'group' && node.children[g].segKey === hp.leaf) { existingGroupIdx = g; break; }
        }
        if (existingGroupIdx !== -1) {
          node.children[existingGroupIdx].sig = sig;
          node.children[existingGroupIdx].hp = hp;
        } else {
          node.children.push({ type: 'leaf', sig: sig, hp: hp });
        }
      }
      return root;
    }

    // Counts a node's own signal (if it is a hybrid node) plus every
    // descendant leaf -- the "N signals in this subtree" number shown next
    // to a group row and used to size the "N / total" picker count.
    function countLeaves(node) {
      var n = node.sig ? 1 : 0;
      for (var i = 0; i < node.children.length; i++) {
        n += node.children[i].type === 'leaf' ? 1 : countLeaves(node.children[i]);
      }
      return n;
    }

    function leafDescendants(node, out) {
      out = out || [];
      if (node.sig) out.push(node.sig);
      for (var i = 0; i < node.children.length; i++) {
        var ch = node.children[i];
        if (ch.type === 'leaf') out.push(ch.sig);
        else leafDescendants(ch, out);
      }
      return out;
    }

    function leafDescendantNames(node) {
      return leafDescendants(node).map(function (s) { return s.name; });
    }

    // Walk a lone single-group-child chain to its terminal node (path
    // compression, §2.3.3): every node along the chain shares that
    // terminal node's identity and expand state. A hybrid node (one that
    // is itself a signal, `.sig` set) always stops the chain right there,
    // even if it has exactly one group child -- it needs its OWN visible
    // row for its leaf affordances, so it can never be compressed into an
    // ancestor's label the way a pure organizational node can.
    function deepestOf(entry) {
      var chain = [entry.segKey];
      var deepest = entry;
      while (!deepest.sig && deepest.children.length === 1 && deepest.children[0].type === 'group') {
        deepest = deepest.children[0];
        chain.push(deepest.segKey);
      }
      return { deepest: deepest, chain: chain };
    }

    // Every group's addressable key exists regardless of expand state --
    // used for expand()/collapse() validation and expandAll().
    function collectAllGroupKeys(node, out) {
      for (var i = 0; i < node.children.length; i++) {
        var entry = node.children[i];
        if (entry.type === 'leaf') continue;
        var d = deepestOf(entry);
        out.add(d.deepest.fullKey);
        collectAllGroupKeys(d.deepest, out);
      }
    }

    function flattenTree(node, expSet, out, depth, parentIdx) {
      for (var i = 0; i < node.children.length; i++) {
        var entry = node.children[i];
        if (entry.type === 'leaf') {
          out.push({
            type: 'leaf', sig: entry.sig, hp: entry.hp, depth: depth,
            showFullName: false, parentIdx: parentIdx,
          });
          continue;
        }
        var d = deepestOf(entry);
        var key = d.deepest.fullKey;
        var rowIdx = out.length;
        var expanded = expSet.has(key);
        out.push({
          type: 'group', key: key, label: d.chain.join(' / '), depth: depth,
          expanded: expanded, count: countLeaves(d.deepest), node: d.deepest,
          parentIdx: parentIdx,
          sig: d.deepest.sig || null, hp: d.deepest.hp || null, // item 5: hybrid marker
        });
        if (expanded) flattenTree(d.deepest, expSet, out, depth + 1, rowIdx);
      }
    }

    function expandedSet(plotIdx) {
      if (!expandedByPlot[plotIdx]) expandedByPlot[plotIdx] = new Set();
      return expandedByPlot[plotIdx];
    }

    // TASK-319, reviewer decision (c): a kind section's open/closed state
    // rides the SAME Set and the SAME toggleExpand/expand/collapse code
    // paths as an ordinary tree group at RUNTIME (simplest), but is
    // persisted and restored through its OWN 'sections' field -- never
    // through 'expanded' -- so an old saved layout's ordinary group state
    // restores byte-for-byte unchanged, and a missing 'sections' field
    // means "this layout predates sections" (apply the KIND_SECTIONS
    // defaults), never "the user closed both" (which an EMPTY 'sections'
    // array means, honoured exactly). No heuristic: the field's mere
    // presence, not its contents, is what decides.
    // TASK-360: the ONE place the kind-section partition and its key set are
    // computed. rebuild() renders from it; expandAll() opens exactly its keys.
    function buildSectionedTrees(plot) {
      var usedSet = new Set();
      var byKey = {};
      KIND_SECTIONS.forEach(function (section) {
        var members = plot.signals.filter(function (sig) {
          if (sig.derived || sig.type !== section.type) return false;
          usedSet.add(sig);
          return true;
        });
        byKey[section.key] = { section: section, members: members, root: buildTree(plot, members, section.key) };
      });
      var restSignals = plot.signals.filter(function (sig) { return !sig.derived && !usedSet.has(sig); });
      return { byKey: byKey, restTree: buildTree(plot, restSignals) };
    }

    function collectSectionedKeys(sectioned) {
      var keys = new Set();
      KIND_SECTIONS.forEach(function (section) {
        collectAllGroupKeys(sectioned.byKey[section.key].root, keys);
        keys.add(section.key);
      });
      collectAllGroupKeys(sectioned.restTree, keys);
      return keys;
    }

    // Every expandable key for *plot* under the current grouping mode.
    function allExpandableKeys(plot) {
      if (treeMode && groupSignalsByKind) return collectSectionedKeys(buildSectionedTrees(plot));
      var keys = new Set();
      collectAllGroupKeys(buildTree(plot), keys);
      return keys;
    }

    function isKindSectionKey(key) {
      for (var i = 0; i < KIND_SECTIONS.length; i++) if (KIND_SECTIONS[i].key === key) return true;
      return false;
    }

    function restoreExpanded(plotIdx, arr) {
      expandedByPlot[plotIdx] = new Set(arr || []);
    }

    // *sectionsArr* is the persisted list of open kind TYPES (e.g.
    // ['voltage']), or `undefined` when the saved layout has no 'sections'
    // field at all. Must run after restoreExpanded (which replaces the
    // whole Set) since this only adds/removes the 'kind:*' keys within it.
    function restoreSections(plotIdx, sectionsArr) {
      var set = expandedByPlot[plotIdx];
      if (!set) set = expandedByPlot[plotIdx] = new Set();
      KIND_SECTIONS.forEach(function (section) { set.delete(section.key); });
      var opened = sectionsArr !== undefined ? sectionsArr
        : KIND_SECTIONS.filter(function (section) { return section.defaultOpen; })
          .map(function (section) { return section.type; });
      opened.forEach(function (type) {
        for (var i = 0; i < KIND_SECTIONS.length; i++) {
          if (KIND_SECTIONS[i].type === type) { set.add(KIND_SECTIONS[i].key); break; }
        }
      });
    }

    function getExpandedForPersist(plotIdx) {
      var set = expandedByPlot[plotIdx];
      if (!set) return [];
      // Ordinary tree-group keys only -- kind-section keys live in their
      // own 'sections' field (getSectionsForPersist), never here.
      return Array.from(set).filter(function (k) { return !isKindSectionKey(k); }).sort();
    }

    function getSectionsForPersist(plotIdx) {
      var set = expandedByPlot[plotIdx];
      if (!set) return [];
      return KIND_SECTIONS.filter(function (section) { return set.has(section.key); })
        .map(function (section) { return section.type; });
    }

    // ---- init / filter / mode -------------------------------------------

    function init() {
      listEl = el('picker-list');
      spacer = ce('div');
      spacer.style.position = 'relative';
      rowsHost = ce('div');
      rowsHost.style.position = 'absolute';
      rowsHost.style.left = '0';
      rowsHost.style.right = '0';
      rowsHost.style.top = '0';
      spacer.appendChild(rowsHost);
      listEl.appendChild(spacer);
      listEl.addEventListener('scroll', onScroll);
      // A mouse press moves DOM focus into the list. Re-rendering inside that
      // 'focus' event would detach the pressed element before mouseup, and the
      // browser then never fires its click. So while a press is in flight the
      // focus render is deferred until just after the click has been delivered.
      var pointerDown = false;
      listEl.addEventListener('mousedown', function () { pointerDown = true; }, true);
      listenDrag('mouseup', function () {
        if (!pointerDown) return;
        setTimeout(function () {
          if (!pointerDown) return;
          pointerDown = false;
          render();
        }, 0);
      });
      listEl.addEventListener('focus', function () { if (!pointerDown) render(); });
      listEl.addEventListener('blur', render);
      listEl.addEventListener('keydown', onKeyDown);
      listen(root, 'contextmenu', on_context_menu);

      var filterInput = el('picker-filter');
      filterInput.addEventListener('input', function (e) {
        setFilter(e.target.value);
      });
      el('picker-filter-clear').addEventListener('click', function () {
        setFilter('');
        filterInput.focus();
      });
      el('plot-visible-btn').addEventListener('click', function () {
        var names = [];
        for (var i = 0; i < visibleRows.length; i++) {
          var r = visibleRows[i];
          if (r.type === 'leaf') names.push(r.sig.name);
        }
        PanelManager.plotVisible(names);
      });
      el('clear-panel-btn').addEventListener('click', function () {
        PanelManager.clearFocusedPanel();
      });
      el('add-selected-btn').addEventListener('click', function () {
        if (!selected.size) return;
        PanelManager.plotVisible(Array.from(selected));
        clearSelection();
      });
      el('tree-mode-toggle').addEventListener('click', function () {
        setTreeMode(!treeMode);
      });
      el('tree-expand-all').addEventListener('click', expandAll);
      el('tree-collapse-all').addEventListener('click', collapseAll);
      updateTreeModeButton();
      installPickerMoreMenu();
    }

    // TASK-341: below the compact-mode container-query breakpoint,
    // picker-actions/picker-tree-actions are display:none (view.css) and
    // their controls are reachable instead through this single anchored
    // menu, built fresh on each open by proxy-clicking the real (hidden but
    // still present and still the sole source of truth) buttons -- so every
    // behavior (disabled/dynamic-label states, click handlers) stays defined
    // in exactly one place. Root-append + `.anchored-menu`, same idiom as
    // installViewMenu(): the menu must escape the sidebar's own
    // `overflow: hidden` (view.css), the same way toolbar menus already
    // escape the toolbar's overflow.
    function fillPickerMoreMenu(menu) {
      menuButton(menu, 'Plot visible', 'Plot every signal currently visible in the list',
        function () { el('plot-visible-btn').click(); });
      menuButton(menu, 'Clear panel', 'Remove every signal from the focused panel',
        function () { el('clear-panel-btn').click(); });
      if (selected.size) {
        menuButton(menu, 'Add selected (' + selected.size + ')', 'Add every selected signal to the focused panel',
          function () { el('add-selected-btn').click(); });
      }
      menuSep(menu);
      menuButton(menu, treeMode ? 'Switch to flat list' : 'Switch to tree',
        'Toggle between hierarchical tree and flat list', function () { el('tree-mode-toggle').click(); });
      menuButton(menu, 'Expand all', 'Expand all groups', function () { el('tree-expand-all').click(); });
      menuButton(menu, 'Collapse all', 'Collapse all groups', function () { el('tree-collapse-all').click(); });
    }

    function installPickerMoreMenu() {
      var menu = el('picker-more-menu');
      menu.classList.add('anchored-menu');
      root.appendChild(menu);
      el('picker-more-btn').addEventListener('click', function (e) {
        e.stopPropagation();
        toggleDropdown(menu, fillPickerMoreMenu, el('picker-more-btn'));
      });
    }

    function close_picker_more_menu() {
      var menu = el('picker-more-menu');
      if (menu.hidden) return;
      var restore_focus = menu.contains(document.activeElement);
      menu.hidden = true;
      if (restore_focus) listEl.focus();
    }

    function updateTreeModeButton() {
      var btn = el('tree-mode-toggle');
      setText(btn, treeMode ? 'Tree' : 'Flat');
      btn.classList.toggle('flat-active', !treeMode);
      btn.title = treeMode ? 'Switch to a flat signal list' : 'Switch to the hierarchical tree';
    }

    // Sets the mode without persisting/rebuilding -- used once at boot,
    // before the first rebuild(), which happens via PanelManager.loadPlot.
    function setTreeModeQuiet(on) {
      treeMode = !!on;
      updateTreeModeButton();
    }

    function setTreeMode(on) {
      setTreeModeQuiet(on);
      var st = StateStore.get();
      if (st) { st.treeMode = treeMode; StateStore.save(); }
      rebuild();
    }

    function getTreeMode() { return treeMode; }

    function setFilter(text) {
      filterText = text;
      el('picker-filter').value = text;
      el('picker-filter-clear').hidden = text.length === 0;
      var st = StateStore.get();
      if (st) { st.filter = text; StateStore.save(); }
      rebuild();
    }

    // ---- rebuild / render --------------------------------------------

    function firstFocusableIndex() {
      for (var i = 0; i < visibleRows.length; i++) if (visibleRows[i].type !== 'header') return i;
      return -1;
    }

    function clampFocus() {
      var i = focusIdxByPlot[activePlotIdx];
      if (i == null || i < 0 || i >= visibleRows.length || visibleRows[i].type === 'header') {
        i = firstFocusableIndex();
      }
      focusIdxByPlot[activePlotIdx] = i;
    }

    function getFocusIndex() {
      clampFocus();
      return focusIdxByPlot[activePlotIdx];
    }

    function setFocus(index) { focusIdxByPlot[activePlotIdx] = index; }

    function rebuild() {
      if (lastPlotIdx !== activePlotIdx) {
        lastPlotIdx = activePlotIdx;
        selected.clear();
        selectionAnchor = null;
      }
      var plot = Store.plot(activePlotIdx);
      visibleRows = [];
      currentTreeKeys = new Set();
      var totalSignals = (plot && plot.kind !== 'constants') ? plot.signals.length : 0;
      var q = filterText.trim().toLowerCase();

      var derived = plot && plot.kind !== 'constants' ? plot.signals.filter(function (sig) {
        return sig.derived && (!q || sig.name.toLowerCase().indexOf(q) !== -1 ||
          (sig.label || '').toLowerCase().indexOf(q) !== -1 ||
          sig.type.toLowerCase().indexOf(q) !== -1);
      }) : [];

      if (!plot || plot.kind === 'constants') {
        // nothing to show
      } else if (q) {
        var matches = [];
        for (var i = 0; i < plot.signals.length; i++) {
          var s = plot.signals[i];
          if (s.derived) continue;
          if (s.name.toLowerCase().indexOf(q) === -1 &&
              (s.label || '').toLowerCase().indexOf(q) === -1 &&
              (s.type || '').toLowerCase().indexOf(q) === -1) continue;
          matches.push(s);
        }
        // Filter behaviour, decided (§2.5): tree mode renders flat results,
        // full names, with a header row; the persisted expanded-set is
        // untouched, so clearing the filter restores exactly the tree left.
        if (treeMode) {
          visibleRows.push({
            type: 'header',
            text: 'filter "' + filterText.trim() + '" — ' + (matches.length + derived.length) + ' of ' + totalSignals + ', flat',
          });
        }
        for (i = 0; i < matches.length; i++) {
          visibleRows.push({
            type: 'leaf', sig: matches[i], hp: hierParse(matches[i]), depth: 0,
            showFullName: true, parentIdx: -1,
          });
        }
      } else if (treeMode && groupSignalsByKind) {
        // TASK-319: one collapsible section per KIND_SECTIONS entry, in
        // table order, each built as its own subtree (today's tree layout,
        // restricted to that kind's signals) -- then the leftover "any
        // other kind" signals in the ordinary, unsectioned tree, exactly as
        // when this option is off.
        var sectioned = buildSectionedTrees(plot);
        var byKey = sectioned.byKey;
        var restTree = sectioned.restTree;
        currentTreeKeys = collectSectionedKeys(sectioned);

        var curSetK = expandedSet(activePlotIdx);
        var staleK = [];
        curSetK.forEach(function (k) { if (!currentTreeKeys.has(k)) staleK.push(k); });
        staleK.forEach(function (k) { curSetK.delete(k); });

        // Plotted count for a closed header's "(N · M plotted)" (computed
        // once, lazily, and reused across sections): the union of every
        // panel's own signals for the active plot -- collapsing a section
        // never touches this, it only reads it.
        var plottedNames = null;
        function plottedCountFor(names) {
          if (!plottedNames) {
            plottedNames = new Set();
            PanelManager.panelsForPlot(activePlotIdx).forEach(function (p) {
              p.signals.forEach(function (n) { plottedNames.add(n); });
            });
          }
          var n = 0;
          for (var i = 0; i < names.length; i++) if (plottedNames.has(names[i])) n++;
          return n;
        }

        KIND_SECTIONS.forEach(function (section) {
          var entry = byKey[section.key];
          var expanded = curSetK.has(section.key);
          var rowIdx = visibleRows.length;
          visibleRows.push({
            type: 'group', key: section.key, label: section.label, depth: 0,
            expanded: expanded, count: countLeaves(entry.root), node: entry.root,
            parentIdx: -1, sig: null, hp: null, isKindSection: true,
            plottedCount: plottedCountFor(entry.members.map(function (s) { return s.name; })),
          });
          if (expanded) flattenTree(entry.root, curSetK, visibleRows, 1, rowIdx);
        });

        flattenTree(restTree, curSetK, visibleRows, 0, -1);
      } else if (treeMode) {
        var tree = buildTree(plot);
        collectAllGroupKeys(tree, currentTreeKeys);
        // Prune stale expanded keys against the just-built tree, silently
        // (§2.6: "it is a display preference, not data").
        var curSet = expandedSet(activePlotIdx);
        var stale = [];
        curSet.forEach(function (k) { if (!currentTreeKeys.has(k)) stale.push(k); });
        stale.forEach(function (k) { curSet.delete(k); });
        flattenTree(tree, curSet, visibleRows, 0, -1);
      } else {
        // Flat mode (§2.4 "permanent escape hatch"): every signal, file
        // order, no grouping -- the same leaf-row shape tree mode uses for
        // an ungrouped signal, which is what makes this mode pixel-for-pixel
        // identical to tree mode on a plot with no dotted names (R4).
        for (i = 0; i < plot.signals.length; i++) {
          if (plot.signals[i].derived) continue;
          visibleRows.push({
            type: 'leaf', sig: plot.signals[i], hp: hierParse(plot.signals[i]), depth: 0,
            showFullName: false, parentIdx: -1,
          });
        }
      }

      // Derived names can contain dotted/hierarchical operand names. Keep
      // them literal in their own group in every mode, never hierParse them.
      if (derived.length) {
        visibleRows.push({ type: 'header', text: 'Derived signals' });
        var derivedGroups = new Set();
        derived.forEach(function (sig) {
          if (sig.group && !derivedGroups.has(sig.group)) {
            derivedGroups.add(sig.group);
            visibleRows.push({ type: 'header', text: sig.group });
          }
          visibleRows.push({ type: 'leaf', sig: sig,
            hp: { segments: [], leaf: sig.name, wrapper: null }, depth: sig.group ? 1 : 0,
            showFullName: true, parentIdx: -1 });
        });
      }

      spacer.style.height = (visibleRows.length * ROW_H) + 'px';
      var leafCount = 0;
      // A hybrid row (item 5, type 'group' with .sig set) counts as one
      // addressable signal too -- it merged what used to be a separate
      // leaf row, and the "N / total" count must not silently drop it.
      for (i = 0; i < visibleRows.length; i++) {
        var vr = visibleRows[i];
        if (vr.type === 'leaf' || (vr.type === 'group' && vr.sig)) leafCount++;
      }
      setText(el('picker-count'), leafCount + ' / ' + totalSignals + ' signals');
      setText(el('picker-count-inline'), leafCount + '/' + totalSignals);
      clampFocus();
      updateSelectionUI();
      render();
    }

    function updateSelectionUI() {
      close_derive_menu();
      close_picker_more_menu();
      var btn = el('add-selected-btn');
      var n = selected.size;
      if (n > 0) {
        setText(btn, 'Add (' + n + ')');
        btn.hidden = false;
      } else {
        btn.hidden = true;
      }
    }

    function clearSelection() {
      selected.clear();
      selectionAnchor = null;
      updateSelectionUI();
      render();
    }

    function selectRangeVisible(anchorIdx, targetIdx) {
      var lo = Math.min(anchorIdx, targetIdx), hi = Math.max(anchorIdx, targetIdx);
      selected.clear();
      for (var i = lo; i <= hi; i++) {
        var r = visibleRows[i];
        if (!r) continue;
        if (r.type === 'leaf') selected.add(r.sig.name);
        else if (r.type === 'group') leafDescendantNames(r.node).forEach(function (n) { selected.add(n); });
      }
    }

    function toggleGroupSelection(node) {
      var names = leafDescendantNames(node);
      var allIn = names.length > 0 && names.every(function (n) { return selected.has(n); });
      names.forEach(function (n) { if (allIn) selected.delete(n); else selected.add(n); });
    }

    function toggleExpand(key) {
      var set = expandedSet(activePlotIdx);
      if (set.has(key)) set.delete(key); else set.add(key);
      PanelManager.scheduleStateSave();
      rebuild();
    }

    function findRowIndexByKey(key) {
      for (var i = 0; i < visibleRows.length; i++) {
        if (visibleRows[i].type === 'group' && visibleRows[i].key === key) return i;
      }
      return -1;
    }

    // Item 5: a hybrid row (row.type 'group' with row.sig set) gets full
    // leaf affordances for click/dblclick/space -- it merged what used to
    // be a separate leaf row, so it behaves like one everywhere except the
    // twisty (which has its own handler in buildGroupRow, stopping
    // propagation so it never reaches this function at all). A PURE group
    // row (row.sig == null) keeps today's behavior: any plain click
    // toggles expand.
    function handleRowClick(e, row, index) {
      if (e.detail >= 2) return; // second click of a dblclick: leave selection alone
      setFocus(index);
      if (row.type === 'group' && !row.sig && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
        toggleExpand(row.key);
        var newIdx = findRowIndexByKey(row.key);
        if (newIdx >= 0) setFocus(newIdx);
        render();
        return;
      }
      var isLeafLike = row.type === 'leaf' || (row.type === 'group' && row.sig);
      if (e.shiftKey && selectionAnchor != null) {
        selectRangeVisible(selectionAnchor, index);
      } else if (e.ctrlKey || e.metaKey) {
        if (isLeafLike) {
          if (selected.has(row.sig.name)) selected.delete(row.sig.name); else selected.add(row.sig.name);
        } else {
          toggleGroupSelection(row.node);
        }
        selectionAnchor = index;
      } else {
        // A plain click SELECTS the clicked row, replacing the selection --
        // clearing alone broke the universal click-then-ctrl-click idiom: the
        // first click added nothing, so two clicks left one signal selected.
        // Deselect-all moved to Escape in the picker (see onKeyDown).
        // A group row without a signal already returned above (expand/collapse),
        // so anything reaching here with no modifier is leaf-like.
        selected.clear();
        if (isLeafLike) selected.add(row.sig.name);
        selectionAnchor = index;
      }
      updateSelectionUI();
      render();
    }

    function handleRowDblClick(row) {
      if (!row) return;
      if (row.type === 'leaf' || (row.type === 'group' && row.sig)) {
        var names = (selected.size > 1 && selected.has(row.sig.name)) ? Array.from(selected) : [row.sig.name];
        PanelManager.plotVisible(names);
      } else if (row.type === 'group') {
        PanelManager.plotVisible(leafDescendantNames(row.node));
      }
    }

    function close_derive_menu() {
      if (!derive_menu || derive_menu.hidden) return;
      var restore_focus = derive_menu.contains(document.activeElement);
      derive_menu.hidden = true;
      if (restore_focus) listEl.focus();
    }

    function open_derive_menu(anchor, row) {
      if (!selected.size || !row) return false;
      var row_names = row.sig ? [row.sig.name] :
        row.type === 'group' ? leafDescendantNames(row.node) : [];
      if (!row_names.length || !row_names.every(function (name) { return selected.has(name); })) return false;
      var names = Array.from(selected);
      var signals = names.map(function (name) { return Store.operand_signal(activePlotIdx, name); });
      if (signals.some(function (sig) { return !sig; })) return false;
      var single = names.length === 1;
      var same_unit = signals.every(function (sig) { return (sig.unit || '') === (signals[0].unit || ''); });
      var reason = same_unit ? '' : 'Sum, difference and mean require the same unit: ' +
        signals.map(function (sig) { return sig.name + ' [' + (sig.unit || 'no unit') + ']'; }).join(', ');
      if (!derive_menu) {
        derive_menu = ce('div', 'dropdown-menu anchored-menu derive-menu');
        derive_menu.dataset.silk = 'derive-menu';
        derive_menu.setAttribute('role', 'menu');
        derive_menu.setAttribute('aria-label', 'Derive signal from selection');
        derive_menu.addEventListener('keydown', function (e) {
          if (e.key === 'Escape') {
            e.preventDefault(); e.stopPropagation(); close_derive_menu();
          } else if (e.key === 'Tab') {
            if (!e.target.closest('form')) close_derive_menu();
          } else if (e.target.tagName !== 'INPUT' && ['ArrowDown', 'ArrowUp', 'Home', 'End'].indexOf(e.key) !== -1) {
            e.preventDefault(); e.stopPropagation();
            var buttons = Array.from(derive_menu.querySelectorAll('button:not(:disabled)'));
            var index = buttons.indexOf(document.activeElement);
            if (e.key === 'Home') index = 0;
            else if (e.key === 'End') index = buttons.length - 1;
            else index = (index + (e.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length;
            buttons[index].focus();
          }
        });
        root.appendChild(derive_menu);
      }
      closeAllDropdowns();
      clear(derive_menu);
      if (reason) {
        var note = ce('div', 'derive-reason');
        setText(note, reason);
        derive_menu.appendChild(note);
      }
      function create(operation, operands, params) {
        try {
          var result = Store.derive_signal(activePlotIdx, operation, operands, params);
          if (result.created) { rebuild(); PanelManager.scheduleStateSave(); }
          var traces = Store.plot(activePlotIdx).signals.filter(function (sig) {
            return sig.name === result.name || sig.group === result.name;
          }).map(function (sig) { return sig.name; });
          PanelManager.plotVisible(traces);
          clearSelection();
        } catch (error) {
          showToast('Cannot derive signal: ' + error.message);
        }
        listEl.focus();
      }
      function entry(operation, label, operands, needs_same_unit) {
        var disabled = needs_same_unit && !same_unit;
        var button = menuButton(derive_menu, label, disabled ? reason : '', function () {
          create(operation, operands);
        }, disabled);
        button.setAttribute('role', 'menuitem');
      }
      // Scale takes parameters, so its entry swaps the menu for a small form.
      function open_scale_form(name) {
        clear(derive_menu);
        var form = ce('form', 'derive-scale');
        form.dataset.silk = 'derive-scale';
        var title = ce('div', 'derive-scale-title');
        setText(title, 'Scale: gain × ' + name + ' + offset');
        form.appendChild(title);
        function field(key, value) {
          var label = ce('label'), text = ce('span'), input = ce('input');
          setText(text, key);
          input.type = 'text';
          input.inputMode = 'decimal';
          input.spellcheck = false;
          input.value = value;
          input.dataset.silk = 'derive-scale-' + key;
          label.appendChild(text);
          label.appendChild(input);
          form.appendChild(label);
          return input;
        }
        var inputs = { gain: field('gain', '1'), offset: field('offset', '0') };
        var submit = ce('button');
        submit.type = 'submit';
        setText(submit, 'Create');
        form.appendChild(submit);
        form.addEventListener('click', function (e) { e.stopPropagation(); });
        form.addEventListener('submit', function (e) {
          e.preventDefault();
          var params = {};
          for (var key in inputs) {
            var text = inputs[key].value.trim(), value = Number(text);
            if (!text || !isFinite(value)) {
              showToast('Scale ' + key + ' must be a finite number; received "' + text + '"');
              inputs[key].focus();
              return;
            }
            params[key] = value;
          }
          derive_menu.hidden = true;
          create('scale', [name], params);
        });
        derive_menu.appendChild(form);
        inputs.gain.focus();
        inputs.gain.select();
      }
      if (single) {
        var x_name = (Store.plot(activePlotIdx).x || {}).name || 'x';
        entry('abs', 'Abs: |' + names[0] + '|', names, false);
        entry('derivative', 'Derivative: d(' + names[0] + ')/d(' + x_name + ')', names, false);
        entry('integral', 'Integral: ∫ ' + names[0] + ' d(' + x_name + ')', names, false);
        var scale = ce('button');
        setText(scale, 'Scale: gain × ' + names[0] + ' + offset…');
        scale.setAttribute('role', 'menuitem');
        scale.addEventListener('click', function (e) {
          e.stopPropagation();
          open_scale_form(names[0]);
        });
        derive_menu.appendChild(scale);
      }
      if (!single) entry('sum', 'Sum: ' + names.join(' + '), names, true);
      if (names.length === 2) {
        entry('difference', 'Difference: ' + names[0] + ' − ' + names[1], names, true);
        entry('difference', 'Difference: ' + names[1] + ' − ' + names[0], names.slice().reverse(), true);
      }
      if (!single) entry('product', 'Product: ' + names.join(' × '), names, false);
      if (names.length === 2) {
        entry('quotient', 'Quotient: ' + names[0] + ' / ' + names[1], names, false);
        entry('quotient', 'Quotient: ' + names[1] + ' / ' + names[0], names.slice().reverse(), false);
      }
      if (!single) entry('mean', 'Mean: (' + names.join(' + ') + ') / ' + names.length, names, true);
      derive_menu.style.maxWidth = Math.max(0, root.clientWidth - 8) + 'px';
      derive_menu.style.maxHeight = Math.max(0, root.clientHeight - 8) + 'px';
      derive_menu.hidden = false;
      positionAnchoredMenu(derive_menu, anchor, 'left');
      return true;
    }

    // ---- trace context menu (TASK-375): built-in items plus host items ----

    function copy_silk_code(name) {
      var result;
      try {
        result = silkCode(traceIdentity(name, activePlotIdx));
      } catch (error) {
        result = { ok: false, reason: error.message };
      }
      if (!result.ok) {
        showToast('No silk code for ' + name + ': ' + result.reason);
        return;
      }
      copyText(result.code, 'copied silk code: ' + result.code);
    }

    function append_trace_items(menu, name) {
      var items = [];
      if (contextMenu.hide.indexOf('copy-silk-code') === -1) {
        items.push({ id: 'copy-silk-code', label: 'Copy as silk code', title: 'Copy the tb.probe Python that reproduces this trace',
          run: function () { copy_silk_code(name); } });
      }
      contextMenu.items.forEach(function (item) {
        items.push({ id: item.id, label: item.label, title: '', run: function () {
          var identity;
          try { identity = traceIdentity(name, activePlotIdx); } catch (error) {
            showToast('Cannot identify ' + name + ': ' + error.message);
            return;
          }
          try { item.onSelect(identity); } catch (error) {
            showToast(item.label + ' failed: ' + (error && error.message ? error.message : error));
          }
        } });
      });
      if (!items.length) return 0;
      if (menu.childNodes.length) menuSep(menu);
      items.forEach(function (item) {
        var button = menuButton(menu, item.label, item.title, function () { item.run(); listEl.focus(); });
        button.setAttribute('role', 'menuitem');
        button.dataset.item = item.id;
      });
      if (Store.findSignal(activePlotIdx, name).role === 'coordinate') {
        var copy = menu.querySelector('[data-item="copy-silk-code"]');
        if (copy) { copy.disabled = true; copy.title = 'Analysis coordinates have no net or branch probe'; }
      }
      return items.length;
    }

    function open_value_menu(anchor, name) {
      selected.clear(); selected.add(name);
      return open_row_menu(anchor, {sig: Store.findSignal(activePlotIdx, name)});
    }

    // The clicked trace's items follow the derive entries for one or more
    // selected traces. An unselected signal first gets plain-click selection,
    // shared by right-click and the keyboard menu gestures.
    function open_row_menu(anchor, row) {
      if (row && row.sig && !selected.has(row.sig.name)) {
        var index = Number(anchor.dataset.index);
        handleRowClick({}, row, index);
        // Plain-click selection renders the list, replacing its row elements.
        anchor = rowsHost.querySelector('[data-index="' + index + '"]');
      }
      if (!open_derive_menu(anchor, row)) return false;
      if (row && row.sig && append_trace_items(derive_menu, row.sig.name)) {
        positionAnchoredMenu(derive_menu, anchor, 'left');
      }
      // Focus blurs the picker and renders its rows again; finish positioning first.
      derive_menu.querySelector('button:not(:disabled)').focus();
      return true;
    }

    function on_context_menu(e) {
      var anchor = e.target.closest('.picker-row');
      if (!anchor || !listEl.contains(anchor)) return;
      var row = visibleRows[Number(anchor.dataset.index)];
      if (open_row_menu(anchor, row)) { e.preventDefault(); e.stopPropagation(); }
    }

    // ---- keyboard nav (design doc section 2.4) --------------------------

    function focusable(i) { return visibleRows[i] && visibleRows[i].type !== 'header'; }

    function scrollToFocus(index) {
      var top = index * ROW_H;
      var viewTop = listEl.scrollTop, viewH = listEl.clientHeight;
      if (top < viewTop) listEl.scrollTop = top;
      else if (top + ROW_H > viewTop + viewH) listEl.scrollTop = top + ROW_H - viewH;
    }

    function onKeyDown(e) {
      if (!visibleRows.length) return;
      var idx = getFocusIndex();
      if (idx < 0) return;
      var row = visibleRows[idx];
      if (e.key === 'Escape') {
        // A plain click now selects rather than clears (TASK-159), so this is
        // the deselect-all gesture. The derive menu handles Escape itself and
        // stops propagation, so it never reaches here while open.
        if (!selected.size) return;
        e.preventDefault();
        clearSelection();
      } else if (e.key === 'ContextMenu' || (e.shiftKey && e.key === 'F10')) {
        scrollToFocus(idx); render();
        var anchor = rowsHost.querySelector('[data-index="' + idx + '"]');
        if (anchor && open_row_menu(anchor, row)) { e.preventDefault(); e.stopPropagation(); }
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        var n = idx; do { n++; } while (n < visibleRows.length && !focusable(n));
        if (n < visibleRows.length) { setFocus(n); scrollToFocus(n); render(); }
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        var p = idx; do { p--; } while (p >= 0 && !focusable(p));
        if (p >= 0) { setFocus(p); scrollToFocus(p); render(); }
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        if (row && row.type === 'group') {
          if (!row.expanded) {
            toggleExpand(row.key);
            var ni = findRowIndexByKey(row.key);
            if (ni >= 0) { setFocus(ni); scrollToFocus(ni); render(); }
          } else if (visibleRows[idx + 1] && visibleRows[idx + 1].parentIdx === idx) {
            setFocus(idx + 1); scrollToFocus(idx + 1); render();
          }
        }
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        if (row && row.type === 'group' && row.expanded) {
          toggleExpand(row.key);
          var ni2 = findRowIndexByKey(row.key);
          if (ni2 >= 0) { setFocus(ni2); scrollToFocus(ni2); render(); }
        } else if (row && row.parentIdx != null && row.parentIdx >= 0) {
          setFocus(row.parentIdx); scrollToFocus(row.parentIdx); render();
        }
      } else if (e.key === 'Enter') {
        e.preventDefault();
        handleRowDblClick(row);
      } else if (e.key === ' ') {
        e.preventDefault();
        if (row) {
          if (row.type === 'leaf' || (row.type === 'group' && row.sig)) {
            if (selected.has(row.sig.name)) selected.delete(row.sig.name); else selected.add(row.sig.name);
          } else if (row.type === 'group') {
            toggleGroupSelection(row.node);
          }
          selectionAnchor = idx;
          updateSelectionUI(); render();
        }
      } else if (e.key === 'Home') {
        e.preventDefault();
        var f0 = firstFocusableIndex();
        if (f0 >= 0) { setFocus(f0); scrollToFocus(f0); render(); }
      } else if (e.key === 'End') {
        e.preventDefault();
        var fEnd = visibleRows.length - 1;
        while (fEnd >= 0 && !focusable(fEnd)) fEnd--;
        if (fEnd >= 0) { setFocus(fEnd); scrollToFocus(fEnd); render(); }
      }
    }

    // ---- row builders -----------------------------------------------

    function noteBadgeFor(name) {
      var notes = Annotations.notesForSignal(name);
      if (!notes.length) return null;
      var worst = notes.reduce(function (a, b) {
        var rank = { info: 0, warn: 1, error: 2 };
        return (rank[b.severity] || 0) > (rank[a.severity] || 0) ? b : a;
      });
      var badge = ce('span', 'pbadge sev-' + (worst.severity || 'info'));
      badge.title = notes.map(function (n) { return n.text; }).join(' | ');
      return badge;
    }

    function buildLeafRow(row, index, listHasFocus) {
      var sig = row.sig, hp = row.hp;
      var isSel = selected.has(sig.name);
      var isFocus = listHasFocus && index === getFocusIndex();
      var cls = 'picker-row' + (isSel ? ' selected' : '') + (isFocus ? ' focused-row' : '');
      var rowEl = ce('div', cls);
      rowEl.style.top = (index * ROW_H) + 'px';
      rowEl.dataset.index = index;
      rowEl.draggable = true;
      rowEl.dataset.name = sig.name;
      rowEl.title = sig.derived && sig.derived.given_name !== undefined ? traceTitle(sig) :
        sig.label ? sig.label + ' \u2014 ' + sig.name : sig.name;  // primary \u2014 raw

      if (row.depth) {
        var indent = ce('span', 'pindent');
        indent.style.width = (12 * row.depth) + 'px';
        rowEl.appendChild(indent);
      }
      var handle = ce('span', 'handle');
      setText(handle, '⋮');
      rowEl.appendChild(handle);

      // Currents show wrapper(leaf) (e.g. "i(vvclk)" at top level, or
      // "i(vsup)" for a nested "i(xdut.vsup)") instead of the bare leaf
      // ("vvclk"/"vsup"), which on its own reads like an odd voltage name --
      // gated on sig.type, not on hp.wrapper, since voltages are wrapped the
      // same way ("v(...)") and must keep today's leaf-only display.
      // Rebuilding from hp.wrapper/hp.leaf rather than using sig.name
      // verbatim keeps a nested current's group path from repeating under
      // its own group row (sig.name would spell out "i(xdut.vsup)" again,
      // right under the "xdut" group). hp.wrapper keeps its original case
      // (Xyce's "I", ngspice's "i"). Falls back to sig.name if hierParse
      // found no wrapper/leaf to rebuild from. showFullName rows (flat search results, derived signals) always
      // keep sig.name, current or not -- that escape hatch is unrelated to
      // this fix.
      var isCurrent = sig.type === 'current';
      var rawText = function (h, full) {
        var cur = (h.wrapper != null && h.leaf) ? (h.wrapper + '(' + h.leaf + ')') : full;
        return row.showFullName ? full : (isCurrent ? cur : (h.leaf || full));
      };
      var name = ce('span', 'pname');
      // TASK-530: the label is the primary text, the raw spelling the secondary.
      var lhp = labelParse(sig);
      setText(name, lhp ? rawText(lhp, sig.label) : rawText(hp, sig.name));
      rowEl.appendChild(name);
      if (lhp) {
        var labelEl = ce('span', 'plabel');
        setText(labelEl, rawText(hp, sig.name));
        labelEl.title = traceTitle(sig);
        rowEl.appendChild(labelEl);
      }

      var type = ce('span', 'ptype');
      setText(type, sig.unit || sig.type || '');
      rowEl.appendChild(type);

      if (sig.nonfinite > 0) {
        var nf = ce('span', 'pnonfinite');
        setText(nf, '⚠');
        nf.title = sig.nonfinite + ' non-finite sample(s)';
        rowEl.appendChild(nf);
      }
      var badge = noteBadgeFor(sig.name);
      if (badge) rowEl.appendChild(badge);

      rowEl.addEventListener('dragstart', function (e) {
        var names = (selected.size > 1 && selected.has(sig.name)) ? Array.from(selected) : [sig.name];
        rowEl.classList.add('dragging');
        e.dataTransfer.setData('text/plain', names.join('\n'));
        e.dataTransfer.setData(signalDragType, '1');
        e.dataTransfer.effectAllowed = 'copy';
      });
      rowEl.addEventListener('dragend', function () { rowEl.classList.remove('dragging'); });
      // Set the focus cursor on mousedown, before the browser moves DOM
      // focus to the list and fires its 'focus' -> render(): otherwise that
      // render paints the focus outline on the STALE (default 0) row, so the
      // first click after the list gains focus highlights the wrong row until
      // a second click catches getFocusIndex() up.
      rowEl.addEventListener('mousedown', function () { setFocus(index); });
      rowEl.addEventListener('click', function (e) { handleRowClick(e, row, index); });
      rowEl.addEventListener('dblclick', function () { handleRowDblClick(row); });
      if (row.sig) {
        rowEl.addEventListener('mouseenter', function () { PanelManager.opHover(row.sig.name); });
        rowEl.addEventListener('mouseleave', function () { PanelManager.opHover(null); });
      }

      return rowEl;
    }

    // Item 5: a group row for a HYBRID node (row.sig set -- the node is
    // both a signal and a parent) additionally gets the leaf decorations
    // (unit/type, non-finite warning, note badge) and full
    // leaf-affordance click/dblclick/drag (wired below, keyed off row.sig
    // rather than row.type so the SAME code path as buildLeafRow's is
    // reused via handleRowClick/handleRowDblClick). The twisty gets its
    // own click handler so it always toggles expand regardless of hybrid
    // status, independent of the row's own click semantics.
    function buildGroupRow(row, index, listHasFocus) {
      var isHybrid = !!row.sig;
      var isSel = isHybrid && selected.has(row.sig.name);
      var isFocus = listHasFocus && index === getFocusIndex();
      var cls = 'picker-row group-row' +
        (isHybrid ? ' hybrid-row' : '') +
        (isSel ? ' selected' : '') + (isFocus ? ' focused-row' : '');
      var rowEl = ce('div', cls);
      rowEl.style.top = (index * ROW_H) + 'px';
      rowEl.dataset.index = index;
      rowEl.draggable = true;
      if (isHybrid) rowEl.dataset.name = row.sig.name;
      rowEl.title = isHybrid ? row.sig.name : row.key;

      if (row.depth) {
        var indent = ce('span', 'pindent');
        indent.style.width = (12 * row.depth) + 'px';
        rowEl.appendChild(indent);
      }
      var twisty = ce('span', 'ptwisty');
      setText(twisty, row.expanded ? '▾' : '▸');
      twisty.addEventListener('click', function (e) {
        e.stopPropagation();
        toggleExpand(row.key);
        var newIdx = findRowIndexByKey(row.key);
        if (newIdx >= 0) setFocus(newIdx);
        render();
      });
      rowEl.appendChild(twisty);

      // Same wrapper(leaf) treatment as buildLeafRow, for the rare case
      // where a current vector's own hierarchy segment is also a parent (a
      // hybrid node). row.hp is only non-null when isHybrid is true, so its
      // fields are read only inside that guard.
      var isHybridCurrent = isHybrid && row.sig.type === 'current';
      var hybridCurrentLabel = isHybridCurrent
        ? ((row.hp.wrapper != null && row.hp.leaf) ? (row.hp.wrapper + '(' + row.hp.leaf + ')') : row.sig.name)
        : null;
      var name = ce('span', 'pname');
      // TASK-530: a group row shows its own segment in [k]; a hybrid row with
      // a label shows that label's segment, the raw text beside it.
      var hybridLhp = isHybrid ? labelParse(row.sig) : null;
      var groupShown = groupText(row, Store.plot(activePlotIdx));
      if (hybridLhp) {
        groupShown = isHybridCurrent && hybridLhp.wrapper != null && hybridLhp.leaf
          ? hybridLhp.wrapper + '(' + hybridLhp.leaf + ')' : (hybridLhp.leaf || groupShown);
      }
      setText(name, isHybridCurrent && !hybridLhp ? hybridCurrentLabel : groupShown);
      rowEl.appendChild(name);

      if (isHybrid) {
        if (hybridLhp) {
          var hybridLabel = ce('span', 'plabel');
          setText(hybridLabel, isHybridCurrent ? hybridCurrentLabel : row.label);
          hybridLabel.title = row.sig.name;
          rowEl.appendChild(hybridLabel);
        }
        var type = ce('span', 'ptype');
        setText(type, row.sig.unit || row.sig.type || '');
        rowEl.appendChild(type);

        if (row.sig.nonfinite > 0) {
          var nf = ce('span', 'pnonfinite');
          setText(nf, '⚠');
          nf.title = row.sig.nonfinite + ' non-finite sample(s)';
          rowEl.appendChild(nf);
        }
        var badge = noteBadgeFor(row.sig.name);
        if (badge) rowEl.appendChild(badge);
      }

      // TASK-319: a kind-section header additionally shows how many of its
      // signals are plotted, but only while collapsed (spec: "a closed
      // header shows its count and how many of its signals are plotted",
      // e.g. "Currents (3 · 1 plotted)"); expanded, the rows below already
      // show that, so it is just the count, like any other group.
      var count = ce('span', 'pcount');
      var countText = row.isKindSection
        ? ('(' + row.count + (row.expanded ? '' : ' · ' + row.plottedCount + ' plotted') + ')')
        : String(row.count);
      setText(count, countText);
      rowEl.appendChild(count);

      rowEl.addEventListener('dragstart', function (e) {
        var names = isHybrid
          ? ((selected.size > 1 && selected.has(row.sig.name)) ? Array.from(selected) : [row.sig.name])
          : leafDescendantNames(row.node);
        rowEl.classList.add('dragging');
        e.dataTransfer.setData('text/plain', names.join('\n'));
        e.dataTransfer.setData(signalDragType, '1');
        e.dataTransfer.effectAllowed = 'copy';
      });
      rowEl.addEventListener('dragend', function () { rowEl.classList.remove('dragging'); });
      // Set the focus cursor on mousedown, before the browser moves DOM
      // focus to the list and fires its 'focus' -> render(): otherwise that
      // render paints the focus outline on the STALE (default 0) row, so the
      // first click after the list gains focus highlights the wrong row until
      // a second click catches getFocusIndex() up.
      rowEl.addEventListener('mousedown', function () { setFocus(index); });
      rowEl.addEventListener('click', function (e) { handleRowClick(e, row, index); });
      rowEl.addEventListener('dblclick', function () { handleRowDblClick(row); });
      if (row.sig) {
        rowEl.addEventListener('mouseenter', function () { PanelManager.opHover(row.sig.name); });
        rowEl.addEventListener('mouseleave', function () { PanelManager.opHover(null); });
      }

      return rowEl;
    }

    function buildHeaderRow(row, index) {
      var rowEl = ce('div', 'picker-row flat-header');
      rowEl.style.top = (index * ROW_H) + 'px';
      setText(rowEl, row.text);
      return rowEl;
    }

    function render() {
      var scrollTop = listEl.scrollTop;
      var viewH = listEl.clientHeight || 400;
      var first = Math.max(0, Math.floor(scrollTop / ROW_H) - 4);
      var last = Math.min(visibleRows.length, Math.ceil((scrollTop + viewH) / ROW_H) + 4);
      clear(rowsHost);
      var listHasFocus = document.activeElement === listEl;
      for (var i = first; i < last; i++) {
        var row = visibleRows[i];
        if (row.type === 'leaf') rowsHost.appendChild(buildLeafRow(row, i, listHasFocus));
        else if (row.type === 'group') rowsHost.appendChild(buildGroupRow(row, i, listHasFocus));
        else rowsHost.appendChild(buildHeaderRow(row, i));
      }
    }

    function onScroll() {
      requestFrame(render);
    }

    // ---- command-bus surface (design doc section 4.3) -------------------

    function expand(path) {
      if (!currentTreeKeys || !currentTreeKeys.has(path)) {
        var sug = nearMisses(path, currentTreeKeys ? Array.from(currentTreeKeys) : [], 3);
        return { ok: false, error: 'unknown group path "' + path + '"' + (sug.length ? ' -- did you mean: ' + sug.join(', ') + '?' : '') };
      }
      expandedSet(activePlotIdx).add(path);
      PanelManager.scheduleStateSave();
      rebuild();
      return { ok: true };
    }

    function collapse(path) {
      if (!currentTreeKeys || !currentTreeKeys.has(path)) {
        var sug = nearMisses(path, currentTreeKeys ? Array.from(currentTreeKeys) : [], 3);
        return { ok: false, error: 'unknown group path "' + path + '"' + (sug.length ? ' -- did you mean: ' + sug.join(', ') + '?' : '') };
      }
      expandedSet(activePlotIdx).delete(path);
      PanelManager.scheduleStateSave();
      rebuild();
      return { ok: true };
    }

    function expandAll() {
      var plot = Store.plot(activePlotIdx);
      expandedByPlot[activePlotIdx] = (plot && plot.kind !== 'constants') ? allExpandableKeys(plot) : new Set();
      PanelManager.scheduleStateSave();
      rebuild();
    }

    function collapseAll() {
      expandedByPlot[activePlotIdx] = new Set();
      PanelManager.scheduleStateSave();
      rebuild();
    }

    return {
      init: init, setFilter: setFilter, rebuild: rebuild, render: render,
      openValueMenu: open_value_menu,
      clearSelection: clearSelection, setTreeMode: setTreeMode,
      setTreeModeQuiet: setTreeModeQuiet, getTreeMode: getTreeMode,
      expand: expand, collapse: collapse, expandAll: expandAll, collapseAll: collapseAll,
      restoreExpanded: restoreExpanded, getExpandedForPersist: getExpandedForPersist,
      restoreSections: restoreSections, getSectionsForPersist: getSectionsForPersist,
    };
  })();

  // ======================================================================
  // AxisModes -- one Y scale per literal unit, shared X transform.
  // ======================================================================

  var AxisModes = (function () {
    function valid_range(range) {
      return Array.isArray(range) && range.length === 2 &&
        isFiniteNum(range[0]) && isFiniteNum(range[1]) && range[1] > range[0];
    }

    function sync_axes(panel) {
      var previous = panel.yAxes;
      var groups = new Map();
      panel.signals.forEach(function (name) {
        var sig = Store.findSignal(panel.plotIdx, name);
        var unit = sig ? sig.unit || '' : '';
        if (!groups.has(unit)) {
          var saved = previous && previous.find(function (axis) { return axis.unit === unit; });
          var mode = saved ? saved.yMode : previous ? 'linear' : panel.yMode;
          var range = saved ? saved.yRange : previous || groups.size ? null : panel.yRange;
          groups.set(unit, { unit: unit, signals: [], plotIdx: panel.plotIdx,
            yMode: mode || 'linear', yRange: valid_range(range) ? range.slice() : null });
        }
        groups.get(unit).signals.push(name);
      });
      // Keep the existing empty-panel controls and uPlot default Y scale.
      if (!groups.size) groups.set('', { unit: '', signals: [], plotIdx: panel.plotIdx,
        yMode: 'linear', yRange: null });
      panel.yAxes = Array.from(groups.values());
      panel.yAxes.forEach(function (axis, index) {
        axis.key = index === 0 ? 'y' : 'y' + (index + 1);
        axis.side = panel.yAxes.length === 2 && index === 1 ? 1 : 3;
        if (['linear', 'log', 'dB'].indexOf(axis.yMode) === -1 ||
            (axis.yMode === 'dB' && !allMag(axis))) axis.yMode = 'linear';
        if (axis.yMode !== 'linear' && axis.yRange && axis.yRange[0] <= 0) axis.yRange = null;
      });
    }

    function for_signal(panel, name) {
      return panel.yAxes.find(function (axis) { return axis.signals.indexOf(name) !== -1; }) || panel.yAxes[0];
    }

    function axis_label(panel, axis) {
      var side = panel.yAxes.length === 2 ? (axis.side === 1 ? 'Right' : 'Left') :
        'Y' + (panel.yAxes.indexOf(axis) + 1);
      return side + ' (' + (axis.unit || 'no unit') + ')';
    }

    function allMag(panel) {
      for (var i = 0; i < panel.signals.length; i++) {
        var sig = Store.findSignal(panel.plotIdx, panel.signals[i]);
        if (!sig || sig.part !== 'mag') return false;
      }
      return panel.signals.length > 0;
    }

    function yScaleOpts(mode) {
      if (mode === 'log' || mode === 'dB') return { distr: 3, log: 10 };
      return { distr: 1 };
    }

    function yAxisValues(axis, precise_db) {
      return function (u, splits) {
        return splits.map(function (v) {
          if (v == null) return '';
          if (axis.yMode === 'dB') {
            var scale = u.scales[axis.key];
            var span = 20 * (Math.log10(scale.max) - Math.log10(scale.min));
            var digits = precise_db && span > 0 ? clamp(1 - Math.floor(Math.log10(span / 4)), 1, 10) : 1;
            return Fmt.plain(20 * Math.log10(v), digits) + ' dB';
          }
          return Fmt.num(v, axis.unit);
        });
      };
    }

    // uPlot's built-in axis-width auto-sizing falls back to a fixed guess
    // (50 CSS px) that is too narrow for our labels ("-40.0 dB", "-200.0
    // deg", engineering-prefixed values with units): the text silently
    // clips inside the axis gutter, and because clipping keeps the
    // right-anchored *end* of the string, visually-different ticks that
    // happen to share a suffix (many of ours do -- "0.0 dB", "0.0 dB", ...)
    // become indistinguishable. Measure the actual label strings ourselves
    // and size the gutter to fit.
    var measureCtx = null;
    function measureCtx2d() {
      if (!measureCtx) measureCtx = document.createElement('canvas').getContext('2d');
      return measureCtx;
    }
    // item 6: the axis font is no longer a fixed string -- it tracks
    // FontScale (the "chips/tables" tier, -2, which reproduces the
    // pre-item-6 hardcoded 12px at the default "normal" scale). Read fresh
    // on every call, never cached, so a font-scale change takes effect the
    // next time a panel is (re)built.
    function axisFont() { return FontScale.mono(-2); }
    function ySizeFn() {
      return function (self, values, axisIdx) {
        // uPlot runs multiple layout passes and calls this before it has
        // computed label strings on the first one (values is null/undefined
        // then); fall back to a sane guess, which a later pass corrects
        // once real labels exist.
        if (!values || !values.length) return 50;
        var ctx = measureCtx2d();
        ctx.font = axisFont();
        var max = 0;
        for (var i = 0; i < values.length; i++) {
          if (!values[i]) continue;
          var w = ctx.measureText(values[i]).width;
          if (w > max) max = w;
        }
        return Math.ceil(max) + 18; // tick mark + gap + a little breathing room
      };
    }

    // Same clipping problem, sideways: uPlot's default minimum tick spacing
    // on the X axis doesn't know our labels carry a unit suffix ("10.00
    // ns"), so at typical panel widths it packs ticks closer than the text
    // is wide and they visually run together. Measure a representative
    // label and use it as the minimum space between ticks.
    function measureWidth(text) {
      var ctx = measureCtx2d();
      ctx.font = axisFont();
      return ctx.measureText(text).width;
    }

    // Item 1 fix: uPlot's own default `axis.stroke` (label text color) is a
    // hardcoded black (`#000` in uplot.iife.min.js), and its default
    // `grid`/`ticks` strokes are near-black too (`rgba(0,0,0,0.07)`) --
    // meant for a light page background. Against this app's dark panel
    // background (--bg-1, ~#14181d) that is black-on-near-black: exactly
    // the "labels are almost like the background" bug. Read the colors off
    // view.css's own custom properties (not hardcoded hex) so the two files
    // cannot drift apart; cached once since these tokens never change at
    // runtime (no theme toggle in this app).
    var _axisColors = null;
    function axisColors() {
      if (_axisColors) return _axisColors;
      var cs = getComputedStyle(root);
      function v(name, fallback) {
        var s = cs.getPropertyValue(name).trim();
        return s || fallback;
      }
      _axisColors = {
        text: v('--text-1', '#a7b0bd'),   // tick label glyphs
        ticks: v('--text-2', '#6d7686'),  // small tick marks
        grid: v('--border', '#2b323d'),   // gridlines -- the subtlest of the three
      };
      return _axisColors;
    }

    return {
      allMag: allMag, yScaleOpts: yScaleOpts, yAxisValues: yAxisValues,
      sync_axes: sync_axes, for_signal: for_signal, axis_label: axis_label,
      ySizeFn: ySizeFn, axisFont: axisFont,
      measureWidth: measureWidth, axisColors: axisColors,
    };
  })();

  // ======================================================================
  // PanelManager
  // ======================================================================

  var PanelManager = (function () {
    var panels = []; // ordered panels; yAxes holds ordered unit groups and their mode/range
    var nextId = 1;
    var focusedId = null;
    var applyingRemoteScale = false;
    var colsCount = 1;

    function panelsEl() { return el('panels'); }

    function panelsForPlot(plotIdx) {
      return panels.filter(function (p) { return p.plotIdx === plotIdx; });
    }

    function teardown() {
      cancelDrag();
      lastHoverPanel = null;
      for (var i = 0; i < panels.length; i++) {
        if (panels[i].uplot) panels[i].uplot.destroy();
      }
      panels = [];
      focusedId = null;
      clear(panelsEl());
      updateEmptyHint();
    }

    function updateEmptyHint() {
      el('empty-hint').hidden = panels.length > 0;
    }

    // Item 3: a per (panel, signal) style override, if the user has set one
    // via the chip's style popover, otherwise the palette-by-index default
    // (unchanged). colorFor is the single choke point every color-reading
    // caller (chip swatch, monitor-table swatch, series stroke) already
    // goes through, so the override "just works" everywhere for free.
    function colorFor(panel, name) {
      var st = panel.styles && panel.styles[name];
      if (st && st.color) return st.color;
      var idx = panel.signals.indexOf(name);
      return PALETTE[idx % PALETTE.length];
    }

    var extentCache = new WeakMap();
    function extentFor(plotIdx) {
      var plot = Store.plot(plotIdx);
      if (extentCache.has(plot)) return extentCache.get(plot);
      var x = Store.xLevel0(plotIdx), min = Infinity, max = -Infinity;
      for (var i = 0; i < x.length; i++) if (isFiniteNum(x[i])) {
        min = Math.min(min, x[i]); max = Math.max(max, x[i]);
      }
      var extent = min <= max ? [min, max] : [0, 1];
      extentCache.set(plot, extent);
      return extent;
    }

    function domainFor(plotIdx) {
      var extent = extentFor(plotIdx), min = extent[0], max = extent[1];
      if (min < max) return validXRange(extent, [0, 1]);
      var pad = min === 0 ? 1 : Math.max(Math.abs(min) * 0.01, Number.MIN_VALUE);
      return [isFiniteNum(min - pad) ? min - pad : min,
        isFiniteNum(max + pad) ? max + pad : max];
    }

    function validXRange(range, fallback) {
      if (!range || !isFiniteNum(range[0]) || !isFiniteNum(range[1]) || range[0] >= range[1]) return fallback;
      // uPlot's tick increments must be representable at this magnitude.
      // This pads chart coordinates only; exact zero-span detection and data
      // remain untouched (a tiny nonzero sweep is still a waveform).
      var pad = Math.max(Math.abs(range[0]), Math.abs(range[1])) * 1e-9;
      if (range[1] - range[0] < pad) return [
        isFiniteNum(range[0] - pad) ? range[0] - pad : range[0],
        isFiniteNum(range[1] + pad) ? range[1] + pad : range[1]];
      return range;
    }

    function xLogTickFilter(u, splits) {
      // uPlot's default log filter labels only exact decades (1e n); on a
      // sub-decade zoom that leaves a single labeled tick. Densify: decades
      // when >=3 are visible, else 1-2-5 multiples, else every split.
      if (!u.scales.x || u.scales.x.distr !== 3) return splits;
      function mant(v) {
        var m = v / Math.pow(10, Math.floor(Math.log10(Math.abs(v)) + 1e-9));
        return Math.round(m * 100) / 100;
      }
      function keep(pred) {
        var kept = splits.map(function (v) {
          return (v != null && v > 0 && pred(mant(v))) ? v : null;
        });
        var n = kept.filter(function (v) { return v != null; }).length;
        return n >= 3 ? kept : null;
      }
      return keep(function (m) { return m === 1; })
          || keep(function (m) { return m === 1 || m === 2 || m === 5; })
          || splits;
    }

    function xTickSpace(plot, domain) {
      // A fixed minimum pixel gap between x ticks, sized to the widest
      // label we're likely to draw (unit-suffixed, e.g. "10.00 ns"),
      // otherwise uPlot's default spacing packs numeric-only-width ticks
      // and our unit suffix makes adjacent labels visually run together.
      var unit = (plot.x || {}).unit || '';
      var sample = Fmt.num(Math.max(Math.abs(domain[0]), Math.abs(domain[1])), unit);
      return Math.ceil(AxisModes.measureWidth(sample)) + 28;
    }

    function chooseStride(sig, N, domain, view, widthPx) {
      var span = domain[1] - domain[0];
      var frac = span > 0 ? (view[1] - view[0]) / span : 1;
      if (!isFinite(frac) || frac <= 0) frac = 1;
      var visible = N * frac;
      var strides = Store.strides(sig);
      var chosen = 1;
      for (var i = 0; i < strides.length; i++) {
        var s = strides[i];
        if (visible / s >= 2 * Math.max(1, widthPx)) chosen = s;
      }
      return chosen;
    }

    // k-way merge of ascending x-arrays into their sorted union. Every array
    // advances by at most one element per output slot, so a repeated x keeps
    // its multiplicity: the union carries a value as many times as whichever
    // signal repeats it most. minmax_level does emit the same sample twice
    // for a bucket whose min and max are the same sample (a flat rail, say),
    // and placeOnUnion walks its own x with the identical one-per-slot rule,
    // so the two stay in step.
    //
    // Matching on exact `===` is sound rather than a float-tolerance trap:
    // every decimated x is a bit-for-bit copy of one of the level-0 samples
    // (minmax_level indexes the original x array, it never computes a bucket
    // centre), so two signals that picked the same sample carry the same bits.
    function unionX(xs) {
      var k = xs.length, total = 0, i;
      for (i = 0; i < k; i++) total += xs[i].length;
      var out = new Float64Array(total);
      var pos = new Int32Array(k);
      var n = 0;
      for (;;) {
        var best = 0, found = false;
        for (i = 0; i < k; i++) {
          if (pos[i] >= xs[i].length) continue;
          var v = xs[i][pos[i]];
          if (!found || v < best) { best = v; found = true; }
        }
        if (!found) break;
        var advanced = 0;
        for (i = 0; i < k; i++) {
          if (pos[i] < xs[i].length && xs[i][pos[i]] === best) { pos[i]++; advanced++; }
        }
        // Unreachable for well-formed data, but a `best` that matches nothing
        // (a NaN in an x array -- every comparison against it is false) would
        // otherwise spin here forever and hang the tab. Stopping early just
        // truncates the merge, which the caller renders as a shorter chart.
        if (!advanced) break;
        out[n] = best;
        n++;
      }
      return n === total ? out : out.subarray(0, n);
    }

    // Linear value at `xv` between this signal's own samples p-1 and p; null
    // outside its range or next to a null, so a genuine gap stays a gap.
    // The `== null` test is load-bearing and must come first: isFiniteNum is
    // a NaN/+-Inf test written for numbers, and `null` passes it (null === null,
    // and null is neither infinity), after which null would coerce to 0 and
    // interpolate a line straight down across the missing data.
    function interpAt(sx, sy, p, xv) {
      if (p === 0 || p >= sx.length) return null;
      var y0 = sy[p - 1], y1 = sy[p];
      if (y0 == null || y1 == null || !isFiniteNum(y0) || !isFiniteNum(y1)) return null;
      var x0 = sx[p - 1], x1 = sx[p];
      if (x1 === x0) return y0;
      return y0 + (y1 - y0) * ((xv - x0) / (x1 - x0));
    }

    // Lay one signal's decimated y out along the shared union x, holding a
    // hole wherever the signal owns no sample at that union position.
    function placeOnUnion(ux, sx, sy, interpolate) {
      // A signal only ever matches slots, never creates ones it does not
      // fill, so |sx| === |ux| means it matched every slot: it already IS the
      // shared array and its y can be handed back untouched (and still typed).
      if (sx.length === ux.length) return sy;
      var out = new Array(ux.length);
      var m = sx.length, p = 0;
      for (var i = 0; i < ux.length; i++) {
        if (p < m && sx[p] === ux[i]) { out[i] = sy[p]; p++; continue; }
        out[i] = interpolate ? interpAt(sx, sy, p, ux[i]) : null;
      }
      return out;
    }

    // uPlot draws every series of a chart against ONE shared x-array, but the
    // decimation pyramid (encode.py minmax_level) picks each signal's
    // per-bucket argmin/argmax INDEPENDENTLY -- at the same stride two
    // signals genuinely land on different sample x-positions (measured: 999
    // of 3126 positions differ at stride 64 on a two-signal 1e6-point run,
    // worst case half a bucket out). Reusing the first signal's x for all the
    // others therefore drew their y values at the wrong times, corrupting
    // exactly the delay/skew reading an overlay exists for.
    //
    // So: merge the panel's decimated x-arrays into their sorted union and
    // place each signal's y only at the union positions it actually owns.
    // The holes are filled two ways, by whether the signal has real gaps:
    //
    //   nonfinite === 0 -> null, spanned by the series' `spanGaps` (set in
    //     makeSeries). uPlot punches gaps by CLIPPING rather than by breaking
    //     the path, so spanning simply drops the clip and the polyline runs
    //     straight from the signal's own sample to its own next one -- what
    //     plotting it alone draws, under linear, stepped and spline alike.
    //
    //   nonfinite > 0 -> linear interpolation between the signal's own
    //     neighbours, null next to a null. `spanGaps` is deliberately NOT set
    //     for these series because it would also swallow their genuine
    //     NaN/Inf gaps and draw a solid line across missing data. Fills on
    //     the straight line between two samples redraw the same path under
    //     the default linear paths and leave the real gaps intact.
    //
    // stride 1 stays on the original single-array path: every signal's
    // level-0 x IS the shared plot x (Store.levelFor hands back the one
    // cached xLevel0 object), so there is nothing to merge or to fill.
    function buildData(panel, stride) {
      var plotIdx = panel.plotIdx;
      var data = [null];
      var recs = [], sigs = [], present = 0, i;
      for (i = 0; i < panel.signals.length; i++) {
        var sig = Store.findSignal(plotIdx, panel.signals[i]);
        sigs.push(sig);
        recs.push(sig ? Store.levelFor(plotIdx, sig, stride) : null);
        if (sig) present++;
      }

      if (stride === 1 || present < 2) {
        var sharedX = null;
        for (i = 0; i < recs.length; i++) {
          if (recs[i] && !sharedX) sharedX = recs[i].x;
        }
        data[0] = sharedX || Store.xLevel0(plotIdx);
        for (i = 0; i < recs.length; i++) data.push(recs[i] ? recs[i].y : []);
        return data;
      }

      var xs = [];
      for (i = 0; i < recs.length; i++) if (recs[i]) xs.push(recs[i].x);
      var ux = unionX(xs);
      data[0] = ux;
      for (i = 0; i < recs.length; i++) {
        if (!recs[i]) { data.push([]); continue; }
        data.push(placeOnUnion(ux, recs[i].x, recs[i].y, sigs[i].nonfinite > 0));
      }
      return data;
    }

    // Item 3: display-only line style. `dash`/`paths` change how uPlot
    // DRAWS between samples, never the samples themselves -- SPICE data is
    // linearly interpolated regardless of what's picked here (the style
    // popover's own copy says so too).
    var DASH_PATTERNS = { dashed: [6, 4], dotted: [1, 3] }; // 'solid'/unset -> no dash entry (uPlot's own solid default)

    function interpPathsFn(interp) {
      if (interp === 'stepped') return uPlot.paths.stepped({ align: 1 });
      if (interp === 'spline') return uPlot.paths.spline();
      return null; // linear/unset -> uPlot's own default path builder
    }

    function makeSeries(panel) {
      var series = [{ label: (Store.plot(panel.plotIdx).x || {}).name || 'x' }];
      for (var i = 0; i < panel.signals.length; i++) {
        var name = panel.signals[i];
        var st = (panel.styles && panel.styles[name]) || {};
        var shownSig = Store.findSignal(panel.plotIdx, name);   // TASK-530: display label, raw key kept
        var s = {
          label: shownSig && shownSig.label ? shownSig.label : name,
          scale: AxisModes.for_signal(panel, name).key,
          stroke: colorFor(panel, name),
          width: st.width || 1.5,
          // A single sample must stay visible whatever the style says.
          points: { show: !!st.points || Store.plot(panel.plotIdx).npoints === 1 },
        };
        // buildData leaves a null wherever this signal owns no sample at a
        // shared union x (decimated multi-signal panels only); spanGaps draws
        // straight through those, so the line matches the signal plotted
        // alone. Safe to set statically even though the stride changes under
        // a live chart: a nonfinite === 0 signal's y carries no OTHER nulls
        // (Store.levelFor only nullifies nonfinite values), so at stride 1 --
        // or in a single-signal panel, which never merges -- it has nothing
        // to span and is a no-op. Signals that do carry genuine NaN/Inf gaps
        // are excluded here and get interpolated fills instead, so spanGaps
        // can never paper over missing data.
        var sig = Store.findSignal(panel.plotIdx, name);
        if (panel.signals.length > 1 && sig && sig.nonfinite === 0) s.spanGaps = true;
        if (DASH_PATTERNS[st.dash]) s.dash = DASH_PATTERNS[st.dash];
        var pathsFn = interpPathsFn(st.interp);
        if (pathsFn) s.paths = pathsFn;
        series.push(s);
      }
      return series;
    }

    // Sample dots: a display-only toggle that marks every level-0 (true
    // simulated) sample of each plotted trace inside the visible X window, so
    // timestep density and large steps show. Shared by every panel. Dots are
    // drawn only while the samples are at least SAMPLE_MARK_MIN_PX css pixels
    // apart; denser than that they are hidden and a note on the plot says so
    // (canvas text: no DOM, so toggling never shifts the layout).
    var SAMPLE_MARK_MIN_PX = 4;
    var sampleMarksOn = false;

    function drawSampleMarks(u, panel) {
      var host = panel.chartEl;
      if (!sampleMarksOn) { if (host) host.removeAttribute('data-sample-marks'); return; }
      var xs = Store.xLevel0(panel.plotIdx);
      var lo = u.scales.x.min, hi = u.scales.x.max;
      if (!xs.length || !isFiniteNum(lo) || !isFiniteNum(hi)) return;
      var i0 = bsearchLeft(xs, Math.min(lo, hi)), i1 = bsearchRight(xs, Math.max(lo, hi));
      var count = i1 - i0;
      var ratio = u.pxRatio || 1;
      var cssWidth = u.bbox.width / ratio;
      var ctx = u.ctx;
      var dense = count > 1 && cssWidth / (count - 1) < SAMPLE_MARK_MIN_PX;
      if (host) host.setAttribute('data-sample-marks', dense ? 'hidden' : 'dots');
      ctx.save();
      ctx.beginPath();
      ctx.rect(u.bbox.left, u.bbox.top, u.bbox.width, u.bbox.height);
      ctx.clip();
      if (dense) {
        ctx.font = FontScale.mono(-3);
        ctx.textAlign = 'right';
        ctx.fillStyle = theme('--annotation-marker', '#e0a640');
        ctx.fillText('sample dots hidden: ' + count + ' samples in view; zoom in',
          u.bbox.left + u.bbox.width - 6 * ratio, u.bbox.top + 14 * ratio);
        ctx.restore();
        return;
      }
      var r = 2.5 * ratio;
      for (var k = 0; k < panel.signals.length; k++) {
        var name = panel.signals[k];
        var sig = Store.findSignal(panel.plotIdx, name);
        if (!sig) continue;
        var ys = Store.rawYLevel0(panel.plotIdx, sig);
        var scaleKey = AxisModes.for_signal(panel, name).key;
        ctx.fillStyle = colorFor(panel, name);
        ctx.beginPath();
        for (var i = i0; i < i1; i++) {
          var v = ys[i];
          if (!isFiniteNum(v)) continue;
          var px = u.valToPos(xs[i], 'x', true), py = u.valToPos(v, scaleKey, true);
          if (!isFiniteNum(px) || !isFiniteNum(py)) continue;
          ctx.moveTo(px + r, py);
          ctx.arc(px, py, r, 0, 2 * Math.PI);
        }
        ctx.fill();
      }
      ctx.restore();
    }

    function setSampleMarks(on) {
      if (opPlot(activePlotIdx)) return; // disabled on an operating point
      sampleMarksOn = !!on;
      setSampleMarksButton();
      for (var i = 0; i < panels.length; i++) if (panels[i].uplot) panels[i].uplot.redraw();
    }

    function setSampleMarksButton() {
      var b = el('samples-toggle');
      if (!b) return;
      b.setAttribute('aria-pressed', sampleMarksOn ? 'true' : 'false');
      b.classList.toggle('active', sampleMarksOn);
    }

    function makeHooks(panel) {
      // uPlot can finish a queued draw after destroy(). The retired chart
      // must not use the replacement panel's new signal/scale assignments.
      var retired = false;
      return {
        destroy: [function () { retired = true; }],
        drawClear: [function (u) { if (!retired) Annotations.drawBands(u, panel); }],
        draw: [function (u) {
          if (retired) return;
          Annotations.drawMarkers(u, panel);
          CursorEngine.draw(u, panel);
          NamedMarkers.draw(u, panel);
          drawSampleMarks(u, panel);
        }],
        // Cache the last x the mouse was actually over. A toolbar button
        // click (add-panel, "+ marker", …) moves the pointer away from the
        // chart *before* the click handler runs, which is exactly when the
        // browser fires a real mouseleave on `.u-over` -- uPlot responds by
        // setting `u.cursor.left` to a sentinel negative value. Reading
        // `u.cursor.left` live at that point would always report "no
        // position", even though the mouse was genuinely just there.
        // `placeCursorAtMouse`/`NamedMarkers.addAtMouse` read this cache
        // instead of the live (and by-then-stale) cursor state.
        setCursor: [function (u) {
          if (retired) return;
          if (u.cursor.left != null && u.cursor.left >= 0) {
            panel._lastCursorX = u.posToVal(u.cursor.left, 'x');
          }
        }],
        setScale: [function (u, key) {
          if (retired) return;
          if (key !== 'x') {
            if (panel.uplot === u) updateYRangeRow(panel);
            return;
          }
          // Track the live X window so any rebuild reseeds from it (fixes the
          // Y-autoscale button and other rebuilds silently resetting X). This
          // runs for a remote-sync change too -- the panel's X genuinely
          // moved -- so it must precede the applyingRemoteScale early-out.
          panel._savedXRange = [u.scales.x.min, u.scales.x.max];
          if (applyingRemoteScale) return;
          broadcastXScale(panel, u.scales.x.min, u.scales.x.max);
          scheduleLevelReresolve(panel);
          scheduleStateSave();
        }],
        // No setSelect hook: uPlot's native drag-zoom is disabled (drag
        // x:false,y:false below). All drag zoom -- X, Y, and box -- is owned
        // by the custom startBoxDrag handler (wireYZoom), which gives the
        // dead-zoned direction latch uPlot's `uni` could not.
      };
    }

    function db_range(u, min, max) {
      if (!(min > 0 && max > min)) return uPlot.rangeLog(min, max, 10);
      // Decade-rounded log bounds flatten small AC changes near unity.
      // Range in dB, keeping the samples and scale coordinates in raw units.
      return uPlot.rangeNum(20 * Math.log10(min), 20 * Math.log10(max), 0.1, false)
        .map(function (value) { return Math.pow(10, value / 20); });
    }

    function db_splits(u, axis_idx, min, max) {
      var lo = 20 * Math.log10(min), hi = 20 * Math.log10(max);
      if (!isFiniteNum(lo) || !isFiniteNum(hi) || !(hi > lo)) return [];
      var step = (hi - lo) / 4;
      var power = Math.pow(10, Math.floor(Math.log10(step)));
      step = [1, 2, 5, 10].find(function (n) { return n * power >= step; }) * power;
      var first = Math.ceil(lo / step), last = Math.floor(hi / step);
      var splits = [];
      for (var i = first; i <= last; i++) splits.push(Math.pow(10, i * step / 20));
      return splits;
    }

    function y_scale_opts(axis, precise_db) {
      var opts = AxisModes.yScaleOpts(axis.yMode);
      if (precise_db && !axis.yRange) opts.range = db_range;
      if (axis.yRange) {
        opts.min = axis.yRange[0];
        opts.max = axis.yRange[1];
        // uPlot needs auto:false at construction, not in setScale().
        opts.auto = false;
      }
      return opts;
    }

    function chart_width(panel) {
      var width = panel.chartEl.clientWidth;
      if (panel.yAxes.length > 1) {
        // clientWidth includes padding; a right axis needs its entire
        // canvas inside the content box, or the panel clips its tick text.
        var css = getComputedStyle(panel.chartEl);
        width -= (parseFloat(css.paddingLeft) || 0) + (parseFloat(css.paddingRight) || 0);
      }
      return width;
    }

    function buildUplot(panel) {
      // An operating point has no x axis: its panels draw strips, not uPlot.
      if (panelIsOp(panel)) { renderStrips(panel); return; }
      var plot = Store.plot(panel.plotIdx);
      var domain = domainFor(panel.plotIdx);
      var widthPx = Math.max(200, chart_width(panel) || 600);
      var stride = 1;
      var N = plot.npoints;
      // _savedXRange is the single source of truth for the panel's X window
      // and is kept current by the setScale('x') hook on every zoom/pan/sync,
      // so a rebuild (Y-range change, Y-autoscale button, yMode/font/distr
      // toggle, first manual-Y transition) preserves X without snapping back
      // to full-domain -- and state restore, which sets _savedXRange to the
      // saved zoom before rebuilding, is honored rather than overridden.
      var view = validXRange(panel._savedXRange, domain);
      if (N > 0) stride = maxStrideAcrossSignals(panel, N, domain, view, widthPx);
      panel._currentStride = stride;
      var data = buildData(panel, stride);
      var axisFont = AxisModes.axisFont();
      var axisColors = AxisModes.axisColors();

      var opts = {
        width: widthPx,
        height: 260 - 2, // minus chart padding fudge
        title: '',
        series: makeSeries(panel),
        scales: {
          // Without a `range` function, uPlot re-derives a non-time x scale's
          // range from the full data extent on every internal reconciliation,
          // silently discarding whatever `setScale('x', ...)` just requested
          // -- verified while fixing item 1 (drag-select zoom) that this
          // broke *every* zoom path (drag-select, +/- keyboard, 0 reset,
          // cross-panel X sync), not just this one; pre-existing, unrelated
          // to the CSS fix here. `range` as a passthrough function (uPlot's
          // own documented recipe for interactive x-zoom) fixes it; `min`/
          // `max` still seed the initial view.
          x: {
            time: false, distr: xLogEnabled ? 3 : 1, log: 10,
            min: view[0], max: view[1],
            range: function (u, dmin, dmax) { return validXRange([dmin, dmax], domain); },
          },
        },
        axes: [
          // Item 1: explicit stroke/grid/ticks colors -- uPlot's own
          // defaults are near-black (`#000` text, `rgba(0,0,0,0.07)`
          // grid/ticks), invisible against this app's dark background.
          { scale: 'x', stroke: axisColors.text, font: axisFont, space: xTickSpace(plot, domain),
            filter: xLogTickFilter,
            grid: { stroke: axisColors.grid, width: 1 },
            ticks: { stroke: axisColors.ticks, width: 1 },
            values: function (u, splits) {
              return splits.map(function (v) { return v == null ? '' : Fmt.num(v, (plot.x || {}).unit || ''); });
            } },
        ],
        cursor: {
          sync: { key: syncKey + panel.plotIdx },
          points: { show: false },
          // Native drag config is left at uPlot's proven default shape, but
          // it never actually fires: wireYZoom's capture-phase mousedown
          // stopPropagation()s every drag on the chart before uPlot's own
          // .u-over listener sees it, so startBoxDrag is the sole drag-zoom
          // path. (Setting drag:{x:false,y:false} instead risked a uPlot
          // construction throw, which broke panel creation entirely.) The
          // custom handler gives the dead-zoned direction latch uPlot's `uni`
          // band-preview could not.
          drag: { x: true, y: false, uni: 30 },
        },
        legend: { show: false },
        hooks: makeHooks(panel),
      };

      panel.yAxes.forEach(function (axis, index) {
        var precise_db = panel.yAxes.length > 1 && axis.yMode === 'dB';
        opts.scales[axis.key] = y_scale_opts(axis, precise_db);
        var axis_opts = { scale: axis.key, side: axis.side, stroke: axisColors.text, font: axisFont,
          values: AxisModes.yAxisValues(axis, precise_db), size: AxisModes.ySizeFn(),
          grid: { show: index === 0, stroke: axisColors.grid, width: 1 },
          ticks: { stroke: axisColors.ticks, width: 1 } };
        if (precise_db) {
          axis_opts.splits = db_splits;
          axis_opts.filter = function (u, splits) { return splits; };
        }
        opts.axes.push(axis_opts);
      });

      if (panel.uplot) panel.uplot.destroy();
      panel.uplot = new uPlot(opts, data, panel.chartEl);
      panel.overEl = panel.chartEl.querySelector('.u-over');
      // uPlot.destroy() tore down the previous .u-over (and any marker
      // badges living inside it); the dict must not keep stale references
      // into a detached subtree, or the next NamedMarkers.draw() would think
      // badges already exist and skip re-creating them in the new .u-over.
      panel.markerBadges = {};
      wireCursorClicks(panel);
      wireCursorBadges(panel);
      // The constructor above already ran one internal draw pass with
      // `panel.overEl` still unset (it doesn't exist until the line above
      // this comment runs), so cursor/marker DOM badges were skipped that
      // first time -- same reason CursorEngine.draw is re-invoked manually
      // here rather than relying solely on the hook.
      CursorEngine.draw(panel.uplot, panel);
      NamedMarkers.draw(panel.uplot, panel);
      // Belt-and-braces for the same poison window: if anything disturbed
      // the constructor's initial ranging, pin the seed range explicitly.
      if (panel.uplot.scales.x.min == null) {
        panel.uplot.setScale('x', { min: view[0], max: view[1] });
      }
    }

    function maxStrideAcrossSignals(panel, N, domain, view, widthPx) {
      var chosen = 1;
      for (var i = 0; i < panel.signals.length; i++) {
        var sig = Store.findSignal(panel.plotIdx, panel.signals[i]);
        if (!sig) continue;
        var s = chooseStride(sig, N, domain, view, widthPx);
        if (s > chosen) chosen = s;
      }
      return chosen;
    }

    // NOTE: these two schedulers must accumulate panels, not close over a
    // single argument — a shared trailing-edge debounce would let the last
    // panel's call cancel the others' pending updates (the AC two-panel
    // default hit exactly that: the mag panel's monitors never refreshed).
    var _reresolvePending = [];
    var _reresolveFlush = debounce(60, function () {
      var ps = _reresolvePending.splice(0);
      for (var i = 0; i < ps.length; i++) reresolveLevel(ps[i]);
    });
    function scheduleLevelReresolve(panel) {
      if (_reresolvePending.indexOf(panel) === -1) _reresolvePending.push(panel);
      _reresolveFlush();
    }

    function reresolveLevel(panel) {
      var u = panel.uplot;
      if (!u || !panel.signals.length) return;
      var plot = Store.plot(panel.plotIdx);
      var domain = domainFor(panel.plotIdx);
      var view = [u.scales.x.min, u.scales.x.max];
      var widthPx = u.bbox ? u.bbox.width : (panel.chartEl.clientWidth || 600);
      var stride = maxStrideAcrossSignals(panel, plot.npoints, domain, view, widthPx);
      if (stride !== panel._currentStride) {
        panel._currentStride = stride;
        var data = buildData(panel, stride);
        u.setData(data, false);
      }
      Monitors.computeForPanel; // referenced lazily below
      scheduleMonitorUpdate(panel);
    }

    var _monitorPending = [];
    var _monitorFlush = debounce(120, function () {
      var ps = _monitorPending.splice(0);
      for (var i = 0; i < ps.length; i++) updateMonitorTable(ps[i]);
    });
    function scheduleMonitorUpdate(panel) {
      if (_monitorPending.indexOf(panel) === -1) _monitorPending.push(panel);
      _monitorFlush();
    }

    function broadcastXScale(sourcePanel, min, max) {
      applyingRemoteScale = true;
      try {
        for (var i = 0; i < panels.length; i++) {
          var p = panels[i];
          if (p === sourcePanel || p.plotIdx !== sourcePanel.plotIdx || !p.uplot) continue;
          p.uplot.setScale('x', { min: min, max: max });
        }
      } finally {
        applyingRemoteScale = false;
      }
    }

    // RAF-coalesced: a synchronous redraw() in the SAME task as a panel's
    // `new uPlot(...)` (e.g. a command batch running addPanel + placeCursor
    // back-to-back) lands inside the constructor's deferred size-convergence
    // window and leaves that panel's x scale [null, null] permanently.
    // Both callers (cursor + marker draws) are visual paths, so a one-frame
    // deferral is imperceptible and removes the poison window entirely.
    var _redrawPending = {};
    function redrawPanels(plotIdx) {
      if (_redrawPending[plotIdx]) return;
      _redrawPending[plotIdx] = true;
      requestFrame(function () {
        _redrawPending[plotIdx] = false;
        for (var i = 0; i < panels.length; i++) {
          if (panels[i].plotIdx !== plotIdx) continue;
          if (panels[i].uplot) panels[i].uplot.redraw();
          else if (panelIsOp(panels[i])) renderStrips(panels[i]);
        }
      });
    }

    // Alt-click places cursor B (drag continues repositioning B while
    // held). Shift-click used to place cursor A the same way; item 2 claims
    // shift-drag for Y-zoom (see wireYZoom below), so that gesture moved --
    // a plain shift-click (no real movement) still places A, just via
    // wireYZoom's own mouseup fallback, which makes this simpler than it
    // was (one modifier/one cursor, not two interleaved in one handler).
    function wireCursorClicks(panel) {
      var overEl = panel.chartEl.querySelector('.u-over');
      if (!overEl) return;
      overEl.addEventListener('mousedown', function (e) {
        if (!e.altKey || e.shiftKey) return;
        var u = panel.uplot;
        var rect = overEl.getBoundingClientRect();
        var left = e.clientX - rect.left;
        CursorEngine.set('B', panel.plotIdx, u.posToVal(left, 'x'));
        function onMove(ev) {
          var l2 = ev.clientX - rect.left;
          CursorEngine.set('B', panel.plotIdx, u.posToVal(l2, 'x'));
        }
        function onUp() {
          removeDragListener('mousemove', onMove);
          removeDragListener('mouseup', onUp);
        }
        listenDrag('mousemove', onMove);
        listenDrag('mouseup', onUp);
        e.preventDefault();
      });
    }

    // ---- item 2: per-panel Y zoom (shift/ctrl-drag box zoom, shift-wheel,
    // typed range, autoscale reset) -----------------------------------

    var Y_ZOOM_CLICK_PX = 4; // movement below this counts as a click, not a drag

    // uPlot's own mousedown handling for x-drag-select is bound directly on
    // `.u-over` (bubble phase); for THAT element as the event target, DOM
    // listener order is registration order regardless of the `capture`
    // flag, so a same-target listener added later (as wireCursorClicks is,
    // after `new uPlot()` has already run) cannot reliably run first. This
    // listener is bound on `panel.chartEl` -- an ANCESTOR of `.u-over` --
    // with `capture:true`, so it sees shift/ctrl mousedowns during the
    // capture walk *before* they ever reach `.u-over`'s own listeners;
    // stopPropagation() there reliably pre-empts uPlot's native x-select
    // for exactly the modifier combo this claims.
    //
    // Wired ONCE per panel (in addPanel, on the stable chartEl -- unlike
    // `.u-over`, which uPlot tears down and rebuilds on every buildUplot()
    // call), so `panel.uplot`/`panel.overEl` are read fresh at drag-start,
    // not captured at wire time.
    function wireYZoom(panel) {
      // Capture-phase on chartEl (an ANCESTOR of .u-over) so we pre-empt the
      // native drag handlers before the mousedown reaches them; the same
      // reason startYZoomDrag documents. Left button only; alt is reserved
      // for cursor B. shift/ctrl/meta force an explicit Y-only drag (kept);
      // a plain drag is the smart box (X / Y / both with a direction latch).
      panel.chartEl.addEventListener('mousedown', function (e) {
        if (e.button !== 0 || e.altKey) return;
        var u = panel.uplot, overEl = panel.overEl;
        if (!u || !overEl) return;
        e.stopPropagation();
        e.preventDefault();
        if (e.shiftKey || e.ctrlKey || e.metaKey) startYZoomDrag(panel, e, u, overEl);
        else startBoxDrag(panel, e, u, overEl);
      }, true);
    }

    // Smart box zoom for a plain drag. Tracks the drag rectangle and latches
    // a mode -- 'x', 'y', or 'box' -- with a dead band so a horizontal drag
    // with small vertical wobble stays X (and vice versa) instead of flipping
    // to box. The rubber-band is drawn to MATCH the latched mode (full-height
    // band for X, full-width for Y, a real rect for box) so the preview shows
    // exactly what release will do. Applies X via setScale (fires the setScale
    // hook -> cross-panel sync) and Y via applyPanelYRange (the auto:false
    // path); box does both, and buildUplot's live-X read preserves the X zoom
    // across the Y rebuild.
    function startBoxDrag(panel, e, u, overEl) {
      var rect = overEl.getBoundingClientRect();
      var startX = e.clientX - rect.left, startY = e.clientY - rect.top;
      var fullW = overEl.clientWidth, fullH = overEl.clientHeight;
      var ENTER = 0.4;  // minor/major below this -> single axis
      var EXIT = 0.7;   // minor/major above this -> box. ENTER<EXIT = dead band
      var mode = null;  // 'x' | 'y' | 'box', latched once past the click threshold
      var box = null, curX = startX, curY = startY;

      function ensureBox() {
        if (box) return;
        box = ce('div', 'y-zoom-box'); // reuse the styled selection div
        overEl.appendChild(box);
      }
      function decide(dx, dy) {
        if (mode == null) {
          if (dx >= dy) mode = (dy < ENTER * dx) ? 'x' : 'box';
          else mode = (dx < ENTER * dy) ? 'y' : 'box';
        } else if (mode === 'x') {
          if (dy > EXIT * dx) mode = 'box';
        } else if (mode === 'y') {
          if (dx > EXIT * dy) mode = 'box';
        } else { // box: collapse back only when the minor axis is clearly small
          if (dy < ENTER * dx) mode = 'x';
          else if (dx < ENTER * dy) mode = 'y';
        }
      }
      function draw(dx, dy) {
        ensureBox();
        var left = Math.min(startX, curX), top = Math.min(startY, curY);
        // Emphasize the edges the drag actually constrains: X -> left/right,
        // Y -> top/bottom, box -> all four. (The base .y-zoom-box class sets
        // top/bottom borders; override per mode so an X drag reads as a
        // vertical pair, not the horizontal pair a Y drag gets.)
        var on = '1px solid var(--accent)';
        if (mode === 'x') {
          box.style.left = left + 'px'; box.style.width = dx + 'px'; box.style.top = '0px'; box.style.height = fullH + 'px';
          box.style.borderTop = 'none'; box.style.borderBottom = 'none'; box.style.borderLeft = on; box.style.borderRight = on;
        } else if (mode === 'y') {
          box.style.left = '0px'; box.style.width = fullW + 'px'; box.style.top = top + 'px'; box.style.height = dy + 'px';
          box.style.borderTop = on; box.style.borderBottom = on; box.style.borderLeft = 'none'; box.style.borderRight = 'none';
        } else {
          box.style.left = left + 'px'; box.style.width = dx + 'px'; box.style.top = top + 'px'; box.style.height = dy + 'px';
          box.style.borderTop = on; box.style.borderBottom = on; box.style.borderLeft = on; box.style.borderRight = on;
        }
      }
      function onMove(ev) {
        curX = ev.clientX - rect.left; curY = ev.clientY - rect.top;
        var dx = Math.abs(curX - startX), dy = Math.abs(curY - startY);
        if (mode == null && Math.max(dx, dy) < Y_ZOOM_CLICK_PX) return; // not yet a drag
        decide(dx, dy);
        draw(dx, dy);
      }
      function onUp(ev) {
        removeDragListener('mousemove', onMove);
        removeDragListener('mouseup', onUp);
        if (box && box.parentNode) box.parentNode.removeChild(box);
        curX = ev.clientX - rect.left; curY = ev.clientY - rect.top;
        var dx = Math.abs(curX - startX), dy = Math.abs(curY - startY);
        if (mode == null || Math.max(dx, dy) < Y_ZOOM_CLICK_PX) return; // a click, not a zoom
        if (mode === 'x' || mode === 'box') {
          var xa = u.posToVal(Math.min(startX, curX), 'x');
          var xb = u.posToVal(Math.max(startX, curX), 'x');
          if (isFiniteNum(xa) && isFiniteNum(xb) && xb > xa) {
            u.setScale('x', { min: xa, max: xb });
            // Keep _savedXRange authoritative synchronously (the setScale hook
            // updates it too, but async), so a later Y rebuild preserves this
            // X zoom instead of reading a stale value.
            panel._savedXRange = [xa, xb];
          }
        }
        if (mode === 'y' || mode === 'box') {
          zoom_y_pixels(panel, u, startY, curY);
        }
      }
      listenDrag('mousemove', onMove);
      listenDrag('mouseup', onUp);
    }

    function startYZoomDrag(panel, e, u, overEl) {
      var rect = overEl.getBoundingClientRect();
      var startY = e.clientY - rect.top;
      var startX = e.clientX - rect.left;
      var startShiftOnly = e.shiftKey && !e.ctrlKey && !e.metaKey;
      var moved = false;
      var box = null;

      function ensureBox() {
        if (box) return;
        box = ce('div', 'y-zoom-box');
        box.style.left = '0';
        box.style.width = overEl.clientWidth + 'px';
        overEl.appendChild(box);
      }
      function onMove(ev) {
        var curY = ev.clientY - rect.top;
        if (!moved && Math.abs(curY - startY) >= Y_ZOOM_CLICK_PX) moved = true;
        if (moved) {
          ensureBox();
          box.style.top = Math.min(startY, curY) + 'px';
          box.style.height = Math.abs(curY - startY) + 'px';
        }
      }
      function onUp(ev) {
        removeDragListener('mousemove', onMove);
        removeDragListener('mouseup', onUp);
        if (box && box.parentNode) box.parentNode.removeChild(box);
        if (moved) {
          var curY = ev.clientY - rect.top;
          zoom_y_pixels(panel, u, startY, curY);
        } else if (startShiftOnly) {
          // preserves the pre-existing "shift-click places cursor A" gesture
          CursorEngine.set('A', panel.plotIdx, u.posToVal(startX, 'x'));
        }
      }
      listenDrag('mousemove', onMove);
      listenDrag('mouseup', onUp);
    }

    // Wired once per panel alongside wireYZoom; wheel bubbles from `.u-over`
    // up through the stable `chartEl`, so this survives `.u-over` being
    // torn down and rebuilt too. Plain wheel is untouched (falls through to
    // the browser's normal page scroll, unchanged from before item 2).
    function wireWheelYZoom(panel) {
      panel.chartEl.addEventListener('wheel', function (e) {
        if (!e.shiftKey) return;
        var u = panel.uplot, overEl = panel.overEl;
        if (!u || !overEl) return;
        e.preventDefault();
        var rect = overEl.getBoundingClientRect();
        var factor = e.deltaY < 0 ? 0.85 : (1 / 0.85);
        var ranges = panel.yAxes.map(function (axis) {
          var pivot = u.posToVal(e.clientY - rect.top, axis.key);
          var cur = currentYRange(panel, axis);
          if (!isFiniteNum(pivot) || cur[0] == null || cur[1] == null) return null;
          return [pivot - (pivot - cur[0]) * factor, pivot + (cur[1] - pivot) * factor];
        });
        apply_y_ranges(panel, ranges);
      }, { passive: false });
    }

    function zoom_y_pixels(panel, u, start, end) {
      // Calculate every range before a rebuild destroys the source chart.
      var ranges = panel.yAxes.map(function (axis) {
        var a = u.posToVal(Math.min(start, end), axis.key);
        var b = u.posToVal(Math.max(start, end), axis.key);
        return [Math.min(a, b), Math.max(a, b)];
      });
      apply_y_ranges(panel, ranges);
    }

    function currentYRange(panel, axis) {
      axis = axis || panel.yAxes[0];
      if (axis.yRange) return axis.yRange;
      var scale = panel.uplot && panel.uplot.scales[axis.key];
      return scale && isFiniteNum(scale.min) && isFiniteNum(scale.max) ? [scale.min, scale.max] : [null, null];
    }

    function apply_y_ranges(panel, ranges) {
      var rebuild = false;
      panel.yAxes.forEach(function (axis, index) {
        var range = ranges[index];
        if (!range || !isFiniteNum(range[0]) || !isFiniteNum(range[1]) || !(range[1] > range[0])) return;
        if (axis.yMode !== 'linear' && range[0] <= 0) return;
        if (!axis.yRange) rebuild = true;
        axis.yRange = range;
      });
      if (panel.uplot) {
        // auto:false must be supplied when constructing the scale. Later
        // manual updates can use setScale without rebuilding the canvas.
        if (rebuild) buildUplot(panel);
        else panel.yAxes.forEach(function (axis) {
          if (axis.yRange) panel.uplot.setScale(axis.key, { min: axis.yRange[0], max: axis.yRange[1] });
        });
      }
      updateMonitorTable(panel);
      scheduleStateSave();
    }

    function applyPanelYRange(panel, lo, hi, axis) {
      axis = axis || panel.yAxes[0];
      apply_y_ranges(panel, panel.yAxes.map(function (group) { return group === axis ? [lo, hi] : null; }));
    }

    function setPanelYRangeManual(panel, lo, hi, axis) {
      axis = axis || panel.yAxes[0];
      if (!isFiniteNum(lo) || !isFiniteNum(hi) || !(hi > lo)) {
        showToast('Y range needs a numeric min < max');
        return;
      }
      if (axis.yMode !== 'linear' && lo <= 0) {
        showToast('A log/dB axis needs a positive Y range');
        return;
      }
      applyPanelYRange(panel, lo, hi, axis);
    }

    function autoscaleYPanel(panel, axis) {
      if (!panel) return;
      (axis ? [axis] : panel.yAxes).forEach(function (group) { group.yRange = null; });
      if (panel.signals.length) { buildUplot(panel); updateMonitorTable(panel); }
      scheduleStateSave();
    }

    function autoscaleYFocused() {
      var p = (lastHoverPanel && lastHoverPanel.plotIdx === activePlotIdx) ? lastHoverPanel : getFocused();
      autoscaleYPanel(p);
    }

    // "Fit everything": reset the shared X to the full domain AND autoscale
    // the Y of EVERY panel on this plot -- regardless of the current zoom or
    // which panel the mouse is over. This is what the 'f' key wants; the
    // single-panel autoscaleYFocused above is for shift-0.
    //
    // Self-contained on purpose: it sets _savedXRange = full domain and
    // yRange = null SYNCHRONOUSLY before rebuilding each panel, rather than
    // leaning on resetZoom() + the setScale hook. That hook updates
    // _savedXRange asynchronously, so a rebuild fired in the same tick would
    // read the stale (still-zoomed) X and snap the view back -- exactly the
    // bug the puppeteer screenshot test caught here.
    function fitAll() {
      var domain = domainFor(activePlotIdx);
      for (var i = 0; i < panels.length; i++) {
        var p = panels[i];
        if (p.plotIdx !== activePlotIdx) continue;
        p._savedXRange = [domain[0], domain[1]];
        p.yAxes.forEach(function (axis) { axis.yRange = null; });
        if (p.uplot && p.signals.length) { buildUplot(p); updateMonitorTable(p); }
      }
      scheduleStateSave();
    }

    function updateYRangeRow(panel) {
      var host = panel.cursorRowEl;
      if (!host) return;
      clear(host);
      var multiple = panel.yAxes.length > 1;
      host.classList.toggle('panel-multi-yrange', multiple);
      if (!panel.signals.length || !panel.uplot) return;
      panel.yAxes.forEach(function (axis) {
        var row = multiple ? ce('div', 'panel-axis-range') : host;
        row.dataset.unit = axis.unit;
        if (multiple) host.appendChild(row);
        var label = ce('span', 'panel-yrange-label');
        setText(label, (multiple ? AxisModes.axis_label(panel, axis) : 'Y') +
          (axis.yRange ? ' (manual):' : ' (auto):'));
        label.title = 'Type a value to set this axis range in its original unit. Chart Y zoom gestures affect all axes.';
        row.appendChild(label);
        var loSpan = ce('span');
        var hiSpan = ce('span');
        ValueEditor.attach(loSpan, {
          getValue: function () { return currentYRange(panel, axis)[0]; },
          setValue: function (v) { setPanelYRangeManual(panel, v, currentYRange(panel, axis)[1], axis); },
          formatFn: function (v) { return Fmt.num(v, axis.unit); }
        });
        ValueEditor.attach(hiSpan, {
          getValue: function () { return currentYRange(panel, axis)[1]; },
          setValue: function (v) { setPanelYRangeManual(panel, currentYRange(panel, axis)[0], v, axis); },
          formatFn: function (v) { return Fmt.num(v, axis.unit); }
        });
        row.appendChild(loSpan);
        var dash = ce('span', 'panel-yrange-dash');
        setText(dash, '–');
        row.appendChild(dash);
        row.appendChild(hiSpan);
        if (multiple) {
          row.appendChild(buildYModeSelect(panel, axis));
          row.appendChild(buildYAutoButton(panel, axis));
        }
      });
    }

    // Grab-handle badges: small "A"/"B" tabs pinned to the top of each
    // cursor line, draggable directly (no modifier key needed), living
    // inside uPlot's own `.u-over` element so their coordinate frame
    // matches `posToVal`/`valToPos` without any manual offset math.
    function wireCursorBadges(panel) {
      if (!panel.overEl) return;
      ['A', 'B'].forEach(function (which) {
        var badge = ce('div', 'cursor-badge cursor-badge-' + which.toLowerCase());
        setText(badge, which);
        badge.hidden = true;
        badge.addEventListener('mousedown', function (e) {
          e.preventDefault();
          e.stopPropagation();
          var u = panel.uplot;
          var rect = panel.overEl.getBoundingClientRect();
          function onMove(ev) {
            var l = ev.clientX - rect.left;
            CursorEngine.set(which, panel.plotIdx, u.posToVal(l, 'x'));
          }
          function onUp() {
            removeDragListener('mousemove', onMove);
            removeDragListener('mouseup', onUp);
          }
          listenDrag('mousemove', onMove);
          listenDrag('mouseup', onUp);
        });
        panel.overEl.appendChild(badge);
        if (which === 'A') panel.badgeA = badge; else panel.badgeB = badge;
      });
    }

    function buildMonitorTable(panel) {
      var host = ce('div', 'panel-monitor-row');
      var table = ce('table', 'panel-table');
      host.appendChild(table);
      panel.monitorTableEl = table;
      return host;
    }

    function updateMonitorTable(panel) {
      if (panelIsOp(panel)) return; // strips carry their values; no trace statistics
      updateYRangeRow(panel); // same refresh cadence as the monitor table (item 2)
      if (!panel.monitorTableEl) return;
      var table = panel.monitorTableEl;
      clear(table);
      var thead = ce('tr');
      ['signal', 'min', 'max', 'p-p', 'mean', 'rms', 'yA', 'yB', 'Δy'].forEach(function (h) {
        var th = ce('th');
        setText(th, h);
        thead.appendChild(th);
      });
      table.appendChild(thead);
      var stats = Monitors.computeForPanel(panel);
      var c = CursorEngine.get(panel.plotIdx);
      for (var i = 0; i < panel.signals.length; i++) {
        var name = panel.signals[i];
        var sig = Store.findSignal(panel.plotIdx, name);
        var unit = sig ? sig.unit : '';
        var st = stats[name];
        var tr = ce('tr');
        var tdName = ce('td', 'sig-name');
        var sw = ce('span', 'swatch-inline');
        sw.style.background = colorFor(panel, name);
        tdName.appendChild(sw);
        tdName.appendChild(document.createTextNode(shownName(panel.plotIdx, name)));
        tdName.title = traceTitle(sig);
        tr.appendChild(tdName);
        [st ? st.min : null, st ? st.max : null, st ? st.pp : null, st ? st.mean : null, st ? st.rms : null]
          .forEach(function (v) {
            var td = ce('td');
            setText(td, Fmt.num(v, unit));
            tr.appendChild(td);
          });
        var yA = (sig && c.A != null) ? Store.sampleAt(panel.plotIdx, sig, c.A) : null;
        var yB = (sig && c.B != null) ? Store.sampleAt(panel.plotIdx, sig, c.B) : null;
        var dy = (yA != null && yB != null) ? (yB - yA) : null;
        [yA, yB, dy].forEach(function (v) {
          var td = ce('td');
          setText(td, Fmt.num(v, unit));
          tr.appendChild(td);
        });
        table.appendChild(tr);
      }
    }

    function updateAllMonitorTables(plotIdx) {
      for (var i = 0; i < panels.length; i++) {
        if (panels[i].plotIdx === plotIdx) updateMonitorTable(panels[i]);
      }
    }

    function buildChip(panel, name) {
      var chip = ce('span', 'chip');
      var sw = ce('span', 'swatch');
      sw.style.background = colorFor(panel, name);
      sw.title = 'Click to style this series (color / width / dash / points)';
      sw.addEventListener('click', function (e) {
        e.stopPropagation();
        openStylePopover(panel, name, sw);
      });
      chip.appendChild(sw);
      var notes = Annotations.notesForSignal(name);
      if (notes.length) {
        var worst = notes.reduce(function (a, b) {
          var rank = { info: 0, warn: 1, error: 2 };
          return (rank[b.severity] || 0) > (rank[a.severity] || 0) ? b : a;
        });
        var dot = ce('span', 'cnote');
        dot.style.background = Annotations.severityColor(worst.severity);
        dot.title = notes.map(function (n) { return n.text; }).join(' | ');
        chip.appendChild(dot);
      }
      var cname = ce('span', 'cname');
      setText(cname, shownName(panel.plotIdx, name));
      cname.title = traceTitle(Store.findSignal(panel.plotIdx, name));
      chip.appendChild(cname);
      var rm = ce('span', 'cremove');
      setText(rm, '×');
      rm.title = 'Remove ' + name;
      rm.addEventListener('click', function () { removeSignal(panel, name); });
      chip.appendChild(rm);
      return chip;
    }

    function buildYModeSelect(panel, axis) {
      axis = axis || panel.yAxes[0];
      var sel = ce('select', 'panel-ymode');
      sel.title = "Y-axis scale: " + AxisModes.axis_label(panel, axis);
      sel.setAttribute('aria-label', sel.title);
      sel.dataset.unit = axis.unit;
      var opts = ['linear', 'log'];
      if (AxisModes.allMag(axis)) opts.push('dB');
      opts.forEach(function (o) {
        var opt = ce('option');
        opt.value = o;
        setText(opt, o);
        if (o === axis.yMode) opt.selected = true;
        sel.appendChild(opt);
      });
      sel.addEventListener('change', function () {
        setPanelYMode(panel, sel.value, axis);
      });
      return sel;
    }

    function buildYAutoButton(panel, axis) {
      var b = ce('button', 'panel-yauto-btn');
      setText(b, 'Y↺'); // Y + anticlockwise-open-circle-arrow ("reset")
      b.title = axis ? 'Autoscale ' + AxisModes.axis_label(panel, axis) :
        "Reset this panel's Y axes to autoscale (shift-0 with the mouse over the panel)";
      b.addEventListener('click', function () { autoscaleYPanel(panel, axis); });
      return b;
    }

    function renderPanelHeader(panel) {
      AxisModes.sync_axes(panel);
      clear(panel.headerEl);
      panel.headerEl.appendChild(buildDragHandle(panel));
      var chips = ce('div', 'panel-chips');
      panel.headerEl.appendChild(chips);
      panel.chipsEl = chips;
      renderChips(panel);
      if (panelIsOp(panel)) {
        panel.headerEl.appendChild(buildSortSelect(panel)); // strips have fixed axes: no Y-mode or autoscale
      } else {
        if (panel.yAxes.length === 1) panel.headerEl.appendChild(buildYModeSelect(panel));
        panel.headerEl.appendChild(buildYAutoButton(panel));
      }
      panel.headerEl.appendChild(buildPanelExportControl(panel));
      var closeBtn = ce('button', 'panel-close');
      setText(closeBtn, '×');
      closeBtn.title = 'Close panel';
      closeBtn.addEventListener('click', function () { removePanel(panel); });
      panel.headerEl.appendChild(closeBtn);
    }

    function buildSortSelect(panel) {
      var select = ce('select', 'panel-sort');
      select.title = 'Order of the rows in this panel';
      [['value', 'by value'], ['name', 'by name'], ['added', 'in the order added']].forEach(function (o) {
        var opt = ce('option'); opt.value = o[0]; setText(opt, o[1]);
        if (o[0] === (panel.sort || 'value')) opt.selected = true;
        select.appendChild(opt);
      });
      select.addEventListener('change', function () {
        panel.sort = select.value;
        renderStrips(panel);
        scheduleStateSave();
      });
      return select;
    }

    // Per-panel CSV/PNG export menu (v1.2 items 1-2). Built fresh on every
    // open so the "cursor window" CSV option and its enabled/disabled state
    // always reflect the *current* A/B cursor placement, not whatever it was
    // when the panel was constructed.
    function buildPanelExportControl(panel) {
      var wrap = ce('span', 'panel-export-wrap');
      var btn = ce('button', 'panel-export-btn');
      setText(btn, '⤓'); // downwards-arrow-to-bar glyph
      btn.title = 'Export this panel (CSV / PNG)';
      var menu = ce('div', 'dropdown-menu');
      menu.hidden = true;
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        toggleDropdown(menu, function (m) { fillPanelExportMenu(m, panel); });
      });
      wrap.appendChild(btn);
      wrap.appendChild(menu);
      return wrap;
    }

    function fillPanelExportMenu(menu, panel) {
      var c = CursorEngine.get(panel.plotIdx);
      var haveCursors = c.A != null && c.B != null;
      menuHeading(menu, 'CSV (level-0 data)');
      menuButton(menu, 'Visible range', 'Export the current zoom window', function () {
        Exporter.exportPanelCsv(panel, 'visible');
      });
      menuButton(menu, 'Cursor window (A–B)',
        PanelManager.panelIsOp(panel) ? 'An operating point has no x axis, so no cursors' : haveCursors ? 'Export between cursors A and B' : 'Place both A and B cursors first',
        function () { Exporter.exportPanelCsv(panel, 'cursor'); }, !haveCursors || PanelManager.panelIsOp(panel));
      menuButton(menu, 'Full range', 'Export every level-0 sample', function () {
        Exporter.exportPanelCsv(panel, 'full');
      });
      menuSep(menu);
      menuHeading(menu, 'PNG image');
      menuButton(menu, 'Download PNG', 'Save this panel as a PNG file', function () {
        Exporter.exportPanelPng(panel, 'download');
      });
      menuButton(menu, 'Copy image', 'Copy this panel to the clipboard (falls back to download)', function () {
        Exporter.exportPanelPng(panel, 'copy');
      });
    }

    // ---- item 3: per-series style popover ---------------------------

    // The currently-EFFECTIVE style for (panel, signal) -- overrides layered
    // over the palette/1.5px/solid/off/linear defaults, exactly what the
    // chart is drawing right now.
    function currentStyle(panel, name) {
      var st = (panel.styles && panel.styles[name]) || {};
      return {
        color: colorFor(panel, name),
        width: st.width || 1.5,
        dash: st.dash || 'solid',
        points: !!st.points,
        interp: st.interp || 'linear',
      };
    }

    // Applies a partial patch (only the keys present are changed), rebuilds
    // the chart + monitor table, and persists. This is the core the command
    // bus and the popover both funnel through.
    function applySeriesStylePatch(panel, name, patch) {
      if (!panel.styles) panel.styles = {};
      var cur = panel.styles[name] || {};
      var next = {};
      ['color', 'width', 'dash', 'points', 'interp'].forEach(function (k) {
        next[k] = (patch[k] !== undefined) ? patch[k] : cur[k];
      });
      panel.styles[name] = next;
      buildUplot(panel);
      updateMonitorTable(panel);
      scheduleStateSave();
    }

    function resetSeriesStyle(panel, name) {
      if (panel.styles) delete panel.styles[name];
      buildUplot(panel);
      updateMonitorTable(panel);
      scheduleStateSave();
    }

    // UI wrapper: also live-updates the chip's swatch color and refreshes
    // an ALREADY-OPEN popover in place (clear+rebuild its own contents,
    // never closing it) so the user can change color, width, dash and
    // points in one open/close cycle instead of one click each.
    function setSeriesStyleFromPopover(panel, name, patch, swatchEl, menuEl) {
      applySeriesStylePatch(panel, name, patch);
      if (swatchEl) swatchEl.style.background = colorFor(panel, name);
      if (menuEl) { clear(menuEl); fillStylePopover(menuEl, panel, name, swatchEl); }
    }

    function fillStylePopover(menu, panel, name, swatchEl) {
      var cur = currentStyle(panel, name);

      menuHeading(menu, 'Series style — ' + name);

      var swatches = ce('div', 'style-swatches');
      PALETTE.forEach(function (c) {
        var dot = ce('span', 'style-swatch-opt' + (c === cur.color ? ' active' : ''));
        dot.style.background = c;
        dot.title = c;
        dot.addEventListener('click', function (e) {
          e.stopPropagation();
          setSeriesStyleFromPopover(panel, name, { color: c }, swatchEl, menu);
        });
        swatches.appendChild(dot);
      });
      menu.appendChild(swatches);

      if (panelIsOp(panel)) {
        // A strip dot has a colour only: width, dash, points and interpolation do not apply.
        menuSep(menu);
        menuButton(menu, 'Reset to default', 'Clear the colour override for this signal', function () {
          resetSeriesStyle(panel, name);
          if (swatchEl) swatchEl.style.background = colorFor(panel, name);
        });
        return;
      }

      menuSep(menu);
      menuHeading(menu, 'Line width');
      var widthRow = ce('div', 'style-btn-row');
      [1, 1.5, 2, 3].forEach(function (w) {
        var b = ce('button', 'style-opt-btn' + (w === cur.width ? ' active' : ''));
        setText(b, String(w));
        b.addEventListener('click', function (e) {
          e.stopPropagation();
          setSeriesStyleFromPopover(panel, name, { width: w }, swatchEl, menu);
        });
        widthRow.appendChild(b);
      });
      menu.appendChild(widthRow);

      menuSep(menu);
      menuHeading(menu, 'Line style');
      var dashRow = ce('div', 'style-btn-row');
      [['solid', 'solid'], ['dashed', 'dashed'], ['dotted', 'dotted']].forEach(function (pair) {
        var b = ce('button', 'style-opt-btn' + (pair[1] === cur.dash ? ' active' : ''));
        setText(b, pair[0]);
        b.addEventListener('click', function (e) {
          e.stopPropagation();
          setSeriesStyleFromPopover(panel, name, { dash: pair[1] }, swatchEl, menu);
        });
        dashRow.appendChild(b);
      });
      menu.appendChild(dashRow);

      menuSep(menu);
      var ptsBtn = ce('button', 'style-opt-btn' + (cur.points ? ' active' : ''));
      setText(ptsBtn, cur.points ? 'Points: on' : 'Points: off');
      ptsBtn.title = 'Show a marker at each sample -- useful for sparse sweeps';
      ptsBtn.addEventListener('click', function (e) {
        e.stopPropagation();
        setSeriesStyleFromPopover(panel, name, { points: !cur.points }, swatchEl, menu);
      });
      menu.appendChild(ptsBtn);

      menuSep(menu);
      menuHeading(menu, 'Interpolation (display only)');
      var interpRow = ce('div', 'style-btn-row');
      [['lin', 'linear'], ['step', 'stepped'], ['spline', 'spline']].forEach(function (pair) {
        var b = ce('button', 'style-opt-btn' + (pair[1] === cur.interp ? ' active' : ''));
        setText(b, pair[0]);
        b.title = 'Draw ' + pair[1] + ' between samples -- SPICE data is linearly interpolated; this only changes how it is drawn, not the data.';
        b.addEventListener('click', function (e) {
          e.stopPropagation();
          setSeriesStyleFromPopover(panel, name, { interp: pair[1] }, swatchEl, menu);
        });
        interpRow.appendChild(b);
      });
      menu.appendChild(interpRow);

      menuSep(menu);
      menuButton(menu, 'Reset to default', 'Clear all overrides for this series', function () {
        resetSeriesStyle(panel, name);
        if (swatchEl) swatchEl.style.background = colorFor(panel, name);
      });
    }

    // Singleton, root-appended popover (not nested under `.panel`, which is
    // `overflow:hidden` for its rounded corners -- a nested popover risked
    // silent clipping). Reused/repositioned on every open rather than
    // created-and-destroyed per signal, so it participates in the SAME
    // `.dropdown-menu` close-on-outside-click / Escape machinery every
    // other dropdown in this file already uses, with no changes to either.
    var stylePopoverEl = null;
    var stylePopoverFor = null; // {panel, name} currently shown, or null

    function ensureStylePopoverEl() {
      if (!stylePopoverEl) {
        stylePopoverEl = ce('div', 'dropdown-menu style-popover');
        stylePopoverEl.hidden = true;
        root.appendChild(stylePopoverEl);
      }
      return stylePopoverEl;
    }

    function openStylePopover(panel, name, swatchEl) {
      var m = ensureStylePopoverEl();
      var wasOpenForThis = !m.hidden && stylePopoverFor &&
        stylePopoverFor.panel === panel && stylePopoverFor.name === name;
      closeAllDropdowns();
      if (wasOpenForThis) { stylePopoverFor = null; return; } // second click = close

      clear(m);
      fillStylePopover(m, panel, name, swatchEl);
      m.hidden = false;
      positionAnchoredMenu(m, swatchEl, 'left');
      stylePopoverFor = { panel: panel, name: name };
    }

    function renderChips(panel) {
      clear(panel.chipsEl);
      for (var i = 0; i < panel.signals.length; i++) {
        panel.chipsEl.appendChild(buildChip(panel, panel.signals[i]));
      }
    }

    function focusPanel(panel) {
      focusedId = panel.id;
      for (var i = 0; i < panels.length; i++) {
        panels[i].el.classList.toggle('focused', panels[i].id === focusedId);
      }
    }

    function getFocused() {
      for (var i = 0; i < panels.length; i++) {
        if (panels[i].id === focusedId) return panels[i];
      }
      return panels.length ? panels[0] : null;
    }

    function ownsSignalDrag(event) {
      return event.dataTransfer && Array.from(event.dataTransfer.types).indexOf(signalDragType) !== -1;
    }

    function wireDrop(panel) {
      panel.chartEl.addEventListener('dragover', function (e) {
        if (reorderDragId == null && !ownsSignalDrag(e)) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = reorderDragId != null ? 'move' : 'copy';
        panel.el.classList.add('panel-drop-target');
      });
      panel.chartEl.addEventListener('dragleave', function () {
        panel.el.classList.remove('panel-drop-target');
      });
      panel.chartEl.addEventListener('drop', function (e) {
        if (reorderDragId == null && !ownsSignalDrag(e)) return;
        e.preventDefault();
        panel.el.classList.remove('panel-drop-target');
        if (reorderDragId != null) { reorderPanel(reorderDragId, panel.id); return; }
        var raw = e.dataTransfer.getData('text/plain');
        if (!raw) return;
        var names = raw.split('\n');
        if (panelIsOp(panel)) { opAddSignals(panel, names); return; }
        for (var i = 0; i < names.length; i++) if (names[i]) addSignal(panel, names[i]);
      });
    }

    // ---- panel reorder (drag the header handle; grid flows row-major, so
    // reordering the DOM order alone is sufficient for both intra- and
    // cross-column moves -- see design doc addendum, punch-list item 6). ----
    var reorderDragId = null;

    function panelIndexById(id) {
      for (var i = 0; i < panels.length; i++) if (panels[i].id === id) return i;
      return -1;
    }

    function reorderPanel(dragId, targetId) {
      var dragIdx = panelIndexById(dragId);
      if (dragIdx === -1 || dragId === targetId) return;
      var moved = panels.splice(dragIdx, 1)[0];
      var targetIdx = panelIndexById(targetId);
      if (targetIdx === -1) targetIdx = panels.length;
      panels.splice(targetIdx, 0, moved);
      var host = panelsEl();
      for (var i = 0; i < panels.length; i++) host.appendChild(panels[i].el);
      scheduleStateSave();
    }

    function wireReorder(panel) {
      panel.el.addEventListener('dragover', function (e) {
        if (reorderDragId == null || reorderDragId === panel.id) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        panel.el.classList.add('panel-reorder-target');
      });
      panel.el.addEventListener('dragleave', function () {
        panel.el.classList.remove('panel-reorder-target');
      });
      panel.el.addEventListener('drop', function (e) {
        if (reorderDragId == null || reorderDragId === panel.id) return;
        e.preventDefault();
        panel.el.classList.remove('panel-reorder-target');
        reorderPanel(reorderDragId, panel.id);
      });
    }

    function buildDragHandle(panel) {
      var handle = ce('span', 'panel-drag-handle');
      setText(handle, '⠷'); // braille-8-dot, reads as a grip glyph
      handle.title = 'Drag to reorder panel';
      handle.draggable = true;
      handle.addEventListener('dragstart', function (e) {
        reorderDragId = panel.id;
        e.dataTransfer.effectAllowed = 'move';
        try { e.dataTransfer.setData('text/plain', ''); } catch (err) { /* ignore */ }
        panel.el.classList.add('panel-dragging');
      });
      handle.addEventListener('dragend', function () {
        reorderDragId = null;
        panel.el.classList.remove('panel-dragging');
        for (var i = 0; i < panels.length; i++) panels[i].el.classList.remove('panel-reorder-target');
      });
      return handle;
    }

    function addPanel(initialSignals, yMode, yRange, styles, yAxes, sort) {
      var panel = {
        id: nextId++, plotIdx: activePlotIdx, signals: (initialSignals || []).slice(),
        yMode: yMode || 'linear', uplot: null, _currentStride: 1, _savedXRange: null, _lastCursorX: null,
        yRange: yRange || null, // legacy layout seed
        yAxes: Array.isArray(yAxes) ? yAxes.filter(function (axis) { return axis && typeof axis.unit === 'string'; }) : null,
        styles: styles || {}, // additive, item 3: per-signal appearance overrides
        sort: sort === 'name' || sort === 'added' ? sort : 'value', // TASK-591: strip row order (operating point only)
      };
      panel.el = ce('div', 'panel' + (opPlot(activePlotIdx) ? ' panel-op' : ''));
      panel.headerEl = ce('div', 'panel-header');
      panel.el.appendChild(panel.headerEl);
      panel.chartEl = ce('div', 'panel-chart');
      panel.el.appendChild(panel.chartEl);
      panel.cursorRowEl = ce('div', 'panel-cursor-row panel-yrange-row');
      panel.el.appendChild(panel.cursorRowEl);
      panel.el.appendChild(buildMonitorTable(panel));
      panel.el.addEventListener('mousedown', function () { focusPanel(panel); }, true);
      panelsEl().appendChild(panel.el);
      panels.push(panel);
      renderPanelHeader(panel);
      wireDrop(panel);
      wireReorder(panel);
      // Wired ONCE here, on the stable chartEl -- see wireYZoom's comment on
      // why (`.u-over` gets torn down/rebuilt by every buildUplot() call,
      // chartEl does not). Same reasoning for the mouseenter listener below:
      // it used to live inside buildUplot() itself, which stacked another
      // identical listener on chartEl every rebuild (font-scale change,
      // addSignal/removeSignal, yMode/yRange edits, setXLog, ...) -- an
      // unbounded leak over a session, since chartEl is never recreated.
      wireYZoom(panel);
      wireWheelYZoom(panel);
      panel.chartEl.addEventListener('mouseenter', function () { lastHoverPanel = panel; });
      buildUplot(panel);
      updateMonitorTable(panel);
      focusPanel(panel);
      updateEmptyHint();
      opMutated(panel.plotIdx);
      scheduleStateSave();
      return panel;
    }

    function removePanel(panel) {
      var idx = panels.indexOf(panel);
      if (idx === -1) return;
      if (panel.uplot) panel.uplot.destroy();
      panel.el.parentNode.removeChild(panel.el);
      panels.splice(idx, 1);
      if (focusedId === panel.id) focusedId = panels.length ? panels[0].id : null;
      updateEmptyHint();
      opMutated(panel.plotIdx);
      scheduleStateSave();
    }

    function addSignal(panel, name) {
      if (panel.signals.indexOf(name) !== -1) return;
      panel.signals.push(name);
      renderPanelHeader(panel);
      if (panelIsOp(panel)) { renderStrips(panel, [name]); opMutated(panel.plotIdx); } else buildUplot(panel);
      updateMonitorTable(panel);
      scheduleStateSave();
    }

    function removeSignal(panel, name) {
      var idx = panel.signals.indexOf(name);
      if (idx === -1) return;
      panel.signals.splice(idx, 1);
      renderPanelHeader(panel);
      if (panel.signals.length || panelIsOp(panel)) {
        buildUplot(panel);
        updateMonitorTable(panel);
      } else {
        if (panel.uplot) { panel.uplot.destroy(); panel.uplot = null; }
        clear(panel.chartEl);
        clear(panel.cursorRowEl); // stop showing a stale Y range for an emptied panel
      }
      opMutated(panel.plotIdx);
      scheduleStateSave();
    }

    function plotVisible(names) {
      var target = getFocused();
      if (!target) target = addPanel([]);
      if (panelIsOp(target)) { opAddSignals(target, names); return; }
      for (var i = 0; i < names.length; i++) addSignal(target, names[i]);
    }

    function clearPanel(panel) {
      if (!panel) return;
      panel.signals = [];
      renderPanelHeader(panel);
      if (panel.uplot) { panel.uplot.destroy(); panel.uplot = null; }
      clear(panel.chartEl);
      clear(panel.cursorRowEl);
      if (panelIsOp(panel)) renderStrips(panel);
      opMutated(panel.plotIdx);
      scheduleStateSave();
    }

    function clearFocusedPanel() { clearPanel(getFocused()); }

    // Command-bus surface (design doc v1.3 section 4.3): change yMode on an
    // arbitrary panel (not necessarily the focused one), refreshing the
    // header select and the chart the same way the UI-driven change does.
    function setPanelYMode(panel, mode, axis) {
      (axis || panel.yAxes[0]).yMode = mode;
      renderPanelHeader(panel);
      buildUplot(panel);
      updateMonitorTable(panel);
      scheduleStateSave();
    }

    function setXRangeAll(min, max) {
      var range = validXRange([min, max], domainFor(activePlotIdx));
      min = range[0]; max = range[1];
      for (var i = 0; i < panels.length; i++) {
        var p = panels[i];
        if (p.plotIdx === activePlotIdx && p.uplot) p.uplot.setScale('x', { min: min, max: max });
      }
    }

    // Command-bus surface: reorder by index within the active plot's panel
    // list -- the same splice `reorderPanel` (drag handle) performs, keyed
    // by position instead of a drag-tracked panel id.
    function movePanelByIndex(fromIdx, toIdx) {
      if (fromIdx < 0 || fromIdx >= panels.length) return false;
      var moved = panels.splice(fromIdx, 1)[0];
      var clampedTo = clamp(toIdx, 0, panels.length);
      panels.splice(clampedTo, 0, moved);
      var host = panelsEl();
      for (var i = 0; i < panels.length; i++) host.appendChild(panels[i].el);
      scheduleStateSave();
      return true;
    }

    function setXLog(on) {
      xLogEnabled = on;
      for (var i = 0; i < panels.length; i++) {
        var p = panels[i];
        if (p.plotIdx !== activePlotIdx || !p.uplot) continue;
        p.uplot.setScale('x', { min: p.uplot.scales.x.min, max: p.uplot.scales.x.max });
        // distr changes require a fresh axis config; simplest correct path
        // is rebuild-in-place, preserving the current signal set.
      }
      // Rebuild panels of the active plot so the new distr takes effect.
      var idxToRebuild = [];
      for (i = 0; i < panels.length; i++) if (panels[i].plotIdx === activePlotIdx) idxToRebuild.push(panels[i]);
      idxToRebuild.forEach(function (p) {
        p._savedXRange = p.uplot ? [p.uplot.scales.x.min, p.uplot.scales.x.max] : null;
        buildUplot(p);
        updateMonitorTable(p);
      });
      scheduleStateSave();
    }

    function resetZoom() {
      var domain = domainFor(activePlotIdx);
      for (var i = 0; i < panels.length; i++) {
        var p = panels[i];
        if (p.plotIdx === activePlotIdx && p.uplot) p.uplot.setScale('x', { min: domain[0], max: domain[1] });
      }
    }

    function zoomAbout(factor) {
      var ref = (lastHoverPanel && lastHoverPanel.plotIdx === activePlotIdx) ? lastHoverPanel : panels[0];
      if (!ref || !ref.uplot) return;
      var u = ref.uplot;
      var center;
      if (u.cursor && u.cursor.left != null && u.cursor.left >= 0) {
        center = u.posToVal(u.cursor.left, 'x');
      } else {
        center = (u.scales.x.min + u.scales.x.max) / 2;
      }
      var half = (u.scales.x.max - u.scales.x.min) * factor / 2;
      var domain = domainFor(activePlotIdx);
      var min = clamp(center - half, domain[0], domain[1]);
      var max = clamp(center + half, domain[0], domain[1]);
      if (min >= max) return;
      for (var i = 0; i < panels.length; i++) {
        var p = panels[i];
        if (p.plotIdx === activePlotIdx && p.uplot) p.uplot.setScale('x', { min: min, max: max });
      }
    }

    function placeCursorAtMouse(which) {
      var ref = (lastHoverPanel && lastHoverPanel.plotIdx === activePlotIdx) ? lastHoverPanel : panels[0];
      if (!ref || !ref.uplot || ref._lastCursorX == null) return;
      CursorEngine.set(which, activePlotIdx, ref._lastCursorX);
    }

    var scheduleStateSave = debounce(400, function () { persistCurrentLayout(); });

    function persistCurrentLayout() {
      var st = StateStore.get();
      if (!st) return;
      var thisPlotPanels = panels.filter(function (p) { return p.plotIdx === activePlotIdx; });
      var domain = domainFor(activePlotIdx);
      var xRange = domain;
      if (thisPlotPanels.length && thisPlotPanels[0].uplot) {
        xRange = [thisPlotPanels[0].uplot.scales.x.min, thisPlotPanels[0].uplot.scales.x.max];
      }
      st.plots[String(activePlotIdx)] = {
        panels: thisPlotPanels.map(function (p) {
          return {
            signals: p.signals.slice(), yMode: p.yAxes[0].yMode,
            yRange: p.yAxes[0].yRange || null,
            yAxes: p.yAxes.map(function (axis) {
              return { unit: axis.unit, signals: axis.signals.slice(), yMode: axis.yMode, yRange: axis.yRange };
            }),
            styles: p.styles || {}, // additive field, item 3
            sort: p.sort || 'value', // additive field, TASK-591 (strip row order)
          };
        }),
        xRange: xRange,
        xLog: xLogEnabled,
        cursors: CursorEngine.forPersist(activePlotIdx),
        markers: NamedMarkers.forPersist(activePlotIdx), // additive field, §7.6
        expanded: Picker.getExpandedForPersist(activePlotIdx), // additive field, v1.3 §2.6
        sections: Picker.getSectionsForPersist(activePlotIdx), // additive field, TASK-319
      };
      st.activePlot = activePlotIdx;
      st.cols = colsCount;
      st.treeMode = Picker.getTreeMode(); // additive field, v1.3 §2.6 (global, not per-plot)
      st.fontScale = FontScale.get(); // additive field, item 6 (global, not per-plot)
      // Include definitions created on plots that have never been opened, and
      // picker-only definitions that are not referenced by any panel yet.
      if (PAYLOAD) PAYLOAD.plots.forEach(function (_, idx) {
        var key = String(idx), derived = Store.derived_for_persist(idx);
        if (derived.length || (st.plots[key] && st.plots[key].derived)) {
          if (!st.plots[key]) st.plots[key] = {};
          st.plots[key].derived = derived;
        }
      });
      StateStore.save();
    }

    // Undebounced variant for callers that need the persisted state to be
    // exactly current *now* (ViewState.snapshot(), the sidecar/getState()
    // serializer, design doc v1.3 §3.2/§9 R7) rather than after the 400ms
    // debounce settles.
    function persistCurrentLayoutNow() { persistCurrentLayout(); }

    function setCols(n) {
      colsCount = (n === 2) ? 2 : 1;
      panelsEl().classList.toggle('cols-2', colsCount === 2);
      setText(el('layout-toggle'), colsCount + ' col' + (colsCount > 1 ? 's' : ''));
      scheduleStateSave();
      // Column count changes the CSS grid track width under each panel's
      // chart; reuse the resize machinery so uPlot re-measures, same as a
      // container resize does (a frame lets the grid reflow land first).
      requestFrame(function () { onResize(); });
    }

    function getCols() { return colsCount; }

    // With no saved layout, silk does not know which signals matter to the
    // user, so the default is one empty panel rather than a guess (owner
    // ruling, TASK-238): every signal is exhaustively plotted otherwise,
    // which is unreadable on a real circuit. AC still seeds its
    // log-frequency x default; the panel itself starts with no signals.
    function defaultLayoutFor(plot) {
      // No 'sections' key here (TASK-319): a brand-new plot has no saved
      // layout at all, which Picker.restoreSections treats exactly like an
      // old saved layout's ABSENT 'sections' field -- both apply the
      // KIND_SECTIONS defaults (Voltages open, Currents closed). 'expanded'
      // (ordinary tree groups) is unrelated and unchanged: still empty.
      // TASK-591: an operating point has one value per signal, so nothing is
      // unreadable: a fresh one opens with one panel per unit, each holding
      // every signal of that unit in payload order (a saved layout is kept as saved).
      if (plot.kind === 'op') {
        var units = [], byUnit = {};
        plot.signals.forEach(function (sig) {
          var u = sig.unit || '';
          if (!byUnit[u]) { byUnit[u] = []; units.push(u); }
          byUnit[u].push(sig.name);
        });
        var rank = function (u) { return u === 'V' ? 0 : u === 'A' ? 1 : 2; };
        units = units.map(function (u, i) { return [u, i]; })
          .sort(function (a, b) { return rank(a[0]) - rank(b[0]) || a[1] - b[1]; })
          .map(function (pair) { return pair[0]; });
        return {
          panels: units.map(function (u) { return { signals: byUnit[u], yMode: 'linear' }; }),
          xRange: null, xLog: false, cursors: [], markers: [], expanded: []
        };
      }
      return {
        panels: [{ signals: [], yMode: 'linear' }],
        xRange: null, xLog: plot.x_log_default, cursors: [], markers: [], expanded: []
      };
    }

    function pruneSavedLayout(plot, saved) {
      // TASK-591: for an operating point a saved panel list is authoritative,
      // even an empty one (a cleared layout must not repopulate); only absent
      // state gets the fresh default.
      var isOp = plot.kind === 'op';
      if (isOp && !Array.isArray(saved.panels)) return null;
      var validNames = {};
      for (var i = 0; i < plot.signals.length; i++) validNames[plot.signals[i].name] = true;
      var dropped = 0;
      var keptPanels = [];
      for (i = 0; i < (saved.panels || []).length; i++) {
        var p = saved.panels[i];
        var keep = (p.signals || []).filter(function (n) {
          var ok = !!validNames[n];
          if (!ok) dropped++;
          return ok;
        });
        if (keep.length || isOp) {
          // Drop styles for signals that got pruned above, same as the
          // signal list itself (item 3's "additive... drop unknown signals
          // on load exactly as the existing pruning does").
          var savedStyles = p.styles || {};
          var keptStyles = {};
          keep.forEach(function (n) { if (savedStyles[n]) keptStyles[n] = savedStyles[n]; });
          var yr = (Array.isArray(p.yRange) && p.yRange.length === 2 &&
            isFiniteNum(p.yRange[0]) && isFiniteNum(p.yRange[1]) && p.yRange[1] > p.yRange[0])
            ? p.yRange : null;
          keptPanels.push({ signals: keep, yMode: p.yMode || 'linear', yRange: yr, styles: keptStyles, yAxes: p.yAxes, sort: p.sort });
        }
      }
      if (dropped > 0) {
        showToast(dropped + ' saved signal' + (dropped === 1 ? '' : 's') + ' not in this run');
      }
      if (!keptPanels.length && !isOp) return null;
      return {
        panels: keptPanels,
        xRange: saved.xRange || null,
        xLog: !!saved.xLog,
        cursors: saved.cursors || [],
        markers: saved.markers || [], // additive field (§7.6); absent on pre-v1.2 saved state
        expanded: saved.expanded || [], // additive field (v1.3 §2.6); absent on pre-v1.3 saved state
        // additive field (TASK-319); passed through AS-IS, not `|| []` --
        // Picker.restoreSections must see `undefined` (absent: apply
        // defaults) as distinct from `[]` (present: both closed, honoured
        // exactly), so this must not collapse that distinction.
        sections: saved.sections,
      };
    }

    function loadPlot(plotIdx) {
      activePlotIdx = plotIdx;
      teardown();
      var plot = Store.plot(plotIdx);
      el('constants-view').hidden = true;
      el('zero-point-hint').hidden = true;
      panelsEl().hidden = false;

      if (plot.kind === 'constants') {
        renderConstants(plot);
        panelsEl().hidden = true;
        el('empty-hint').hidden = true;
        el('constants-view').hidden = false;
        el('xlog-toggle').disabled = true;
        Picker.rebuild();
        NamedMarkers.renderTable(plotIdx); // else it shows the previous plot's stale rows/headers
        return;
      }
      setOpToolbar(plot.kind === 'op');

      if (plot.npoints === 0) {
        el('zero-point-hint').hidden = false;
        panelsEl().hidden = true;
        el('empty-hint').hidden = true;
        xLogEnabled = false;
        setXLogButtonLabel();
        Picker.rebuild();
        NamedMarkers.renderTable(plotIdx); // else it shows the previous plot's stale rows/headers
        return;
      }

      var st = StateStore.get();
      var saved = st && st.plots ? st.plots[String(plotIdx)] : null;
      var layout = saved ? pruneSavedLayout(plot, saved) : null;
      if (!layout) layout = defaultLayoutFor(plot);

      xLogEnabled = !!layout.xLog;
      setXLogButtonLabel();
      CursorEngine.restore(plotIdx, layout.cursors);
      NamedMarkers.restore(plotIdx, layout.markers || []);
      Picker.restoreExpanded(plotIdx, layout.expanded || []);
      Picker.restoreSections(plotIdx, layout.sections); // TASK-319; undefined means "apply defaults"

      for (var i = 0; i < layout.panels.length; i++) {
        var pl = layout.panels[i];
        var panel = addPanel(pl.signals, pl.yMode, pl.yRange, pl.styles, pl.yAxes, pl.sort);
        if (layout.xRange && !panelIsOp(panel)) {
          panel._savedXRange = layout.xRange;
          buildUplot(panel);
          updateMonitorTable(panel);
        }
      }
      updateEmptyHint();
      CursorEngine.updateReadouts(plotIdx);
      NamedMarkers.renderTable(plotIdx);
      Picker.rebuild();
    }

    // Item 4: "(all)" makes explicit that this control is shared by every
    // panel (they're X-synced by design), unlike the per-panel Y-mode
    // select next to it -- see also the toolbar button's static title and
    // buildYModeSelect's title.
    // An operating point has no x axis and one sample: X scale and Samples
    // do nothing there, so they are disabled with a tooltip saying why.
    var TOOLBAR_TITLES = {};
    function setOpToolbar(isOp) {
      ['xlog-toggle', 'samples-toggle'].forEach(function (id) {
        var b = el(id);
        if (!b) return;
        if (!(id in TOOLBAR_TITLES)) TOOLBAR_TITLES[id] = b.title;
        b.disabled = isOp;
        b.title = isOp ? (id === 'xlog-toggle' ? 'An operating point has no x axis' : 'An operating point is one point: there are no samples to show') : TOOLBAR_TITLES[id];
      });
    }

    function setXLogButtonLabel() {
      setText(el('xlog-toggle'), 'X: ' + (xLogEnabled ? 'log' : 'lin') + ' (all)');
    }

    // Item 6: rebuild every mounted panel so the new font size takes effect
    // in the live charts (axis text/gutter sizing is baked into uPlot's
    // opts at construction time) and its cached width measurements. A
    // no-op if called before any panel exists (FontScale.set() at boot,
    // before PanelManager.loadPlot() has run).
    function rebuildAllForFontChange() {
      for (var i = 0; i < panels.length; i++) {
        if (panels[i].uplot) { buildUplot(panels[i]); updateMonitorTable(panels[i]); }
        else if (panelIsOp(panels[i])) renderStrips(panels[i]);
      }
    }

    function replaceDerived(plotIdx, changes) {
      function remap(panel) {
        var changed = false;
        changes.forEach(function (change) {
          var before = change.before, after = change.after;
          if (!panel.signals || !panel.signals.some(function (name) { return before.indexOf(name) !== -1; })) return;
          changed = true;
          if (before.length === after.length) return;
          var names = [];
          panel.signals.forEach(function (name) {
            var replacements = before.indexOf(name) === -1 ? [name] : after;
            replacements.forEach(function (next) { if (names.indexOf(next) === -1) names.push(next); });
          });
          panel.signals = names;
          if (panel.styles) {
            var oldStyle = before.map(function (name) { return panel.styles[name]; }).find(Boolean);
            before.forEach(function (name) { delete panel.styles[name]; });
            if (oldStyle) after.forEach(function (name) { panel.styles[name] = Object.assign({}, oldStyle); });
          }
        });
        return changed;
      }
      // Apply every descendant mapping before regrouping: intermediate missing
      // part names must not discard an otherwise unchanged axis mode/range.
      panelsForPlot(plotIdx).forEach(function (panel) {
        if (!remap(panel)) return;
        renderPanelHeader(panel); // regroup axes by the replacement's units
        buildUplot(panel);
        updateMonitorTable(panel);
      });
      // Inactive plots and value-table layouts have no mounted panels.
      var saved = StateStore.get().plots[String(plotIdx)];
      if (saved && saved.panels) saved.panels.forEach(function (panel) {
        if (!remap(panel)) return;
        var grouping = Object.assign({}, panel, {plotIdx: plotIdx});
        AxisModes.sync_axes(grouping);
        panel.yAxes = grouping.yAxes.map(function (axis) {
          return {unit: axis.unit, signals: axis.signals, yMode: axis.yMode, yRange: axis.yRange};
        });
      });
    }

    // ---- TASK-591: an operating point drawn as dot-plot strips ------------
    // An operating point (kind 'op', no x axis) uses the ordinary panels. A
    // panel on such a plot has no uPlot: its body is one strip per unit.
    // Strip axes come from every signal of the plot with that unit, so adding
    // or removing a panel signal never rescales them. Colour is colorFor on
    // the real panel, like every trace and chip. Reference markers (see
    // NamedMarkers) draw inside the strips of their unit.
    var SVG_NS = 'http://www.w3.org/2000/svg';
    var OP_ROW_H = 26, OP_TOP = 34, OP_LEFT = 130, OP_RIGHT = 110, OP_DELTA = 96;
    var lastOpStrip = null; // {plotIdx, unit, value}: the strip last hovered, for "+ marker"

    function opPlot(plotIdx) {
      var plot = Store.plot(plotIdx);
      return !!plot && plot.kind === 'op';
    }
    function panelIsOp(panel) { return opPlot(panel.plotIdx); }

    // After any panel mutation on an operating point: the open markers table
    // (columns are the signals of every panel) and the picker's plotted counts.
    function opMutated(plotIdx) {
      if (!opPlot(plotIdx)) return;
      NamedMarkers.renderTable(plotIdx);
      Picker.rebuild();
    }

    // Add several signals to a strip panel at once; every new row gets the added tag.
    function opAddSignals(panel, names) {
      var added = [];
      names.forEach(function (n) {
        if (n && panel.signals.indexOf(n) === -1) { panel.signals.push(n); added.push(n); }
      });
      if (!added.length) return;
      renderPanelHeader(panel); renderStrips(panel, added); opMutated(panel.plotIdx); scheduleStateSave();
    }

    function opHover(name) {
      opHoverName = name;
      root.querySelectorAll('.op-row').forEach(function (g) {
        g.classList.toggle('op-hover', !!name && g.dataset.signal === name);
      });
      root.querySelectorAll('[data-silk="picker-list"] .picker-row').forEach(function (r) {
        r.classList.toggle('op-hover', !!name && r.dataset.name === name);
      });
    }
    var opHoverName = null;

    function svgEl(tag, attrs, parent, cls) {
      var e = document.createElementNS(SVG_NS, tag);
      if (cls) e.setAttribute('class', cls);
      for (var k in attrs) e.setAttribute(k, attrs[k]);
      if (parent) parent.appendChild(e);
      return e;
    }
    function svgText(parent, x, y, s, cls, anchor) {
      var t = svgEl('text', { x: x, y: y }, parent, cls);
      if (anchor) t.setAttribute('text-anchor', anchor);
      t.textContent = s; return t;
    }

    function niceTicks(lo, hi) {
      var raw = (hi - lo) / 8, mag = Math.pow(10, Math.floor(Math.log10(raw)));
      var step = [1, 2, 5, 10].map(function (m) { return m * mag; }).find(function (c) { return c >= raw; });
      var out = [];
      for (var v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
      return out;
    }

    function opFmt(unit) {
      if (unit === 'V') return function (v) { return (v === 0 || Math.abs(v) >= 1e-3) ? v.toFixed(3) + ' V' : Fmt.num(v, 'V'); };
      return function (v) { return Fmt.num(v, unit); };
    }

    function opLabel(sig) { return (sig.label || sig.name).replace(/^[vi]\((.*)\)$/, '$1'); }

    // Stored value, unrounded (a JS number, so String() round-trips it).
    function opValue(plotIdx, sig) { return Store.levelFor(plotIdx, sig, 1).yRaw[0]; }

    // Units in a panel: V, then A, then any other unit in first-seen order.
    function opUnits(panel) {
      var seen = [];
      panel.signals.forEach(function (name) {
        var sig = Store.findSignal(panel.plotIdx, name), u = sig ? (sig.unit || '') : null;
        if (u !== null && seen.indexOf(u) === -1) seen.push(u);
      });
      var rank = function (u) { return u === 'V' ? 0 : u === 'A' ? 1 : 2; };
      return seen.map(function (u, i) { return [u, i]; })
        .sort(function (a, b) { return rank(a[0]) - rank(b[0]) || a[1] - b[1]; })
        .map(function (p) { return p[0]; });
    }

    function opTitle(unit) {
      if (unit === 'V') return 'Node voltages (V)';
      if (unit === 'A') return 'Branch currents (A): positive flows into the source’s + terminal';
      return unit ? 'Signals (' + unit + ')' : 'Signals (no unit)';
    }

    // Fixed per-unit axis from every signal of the plot with that unit.
    function opAxis(plotIdx, unit) {
      var plot = Store.plot(plotIdx), fmt = opFmt(unit);
      var sigs = plot.signals.filter(function (s) { return (s.unit || '') === unit; });
      var vals = sigs.map(function (s) { return opValue(plotIdx, s); }).filter(isFiniteNum);
      var lo = Math.min.apply(null, vals.concat(0)), hi = Math.max.apply(null, vals.concat(0));
      if (unit === 'A') {
        var a = Math.max.apply(null, vals.map(Math.abs).concat(0)) || 1;
        return { lo: -a, hi: a, ticks: [-a, -a / 2, 0, a / 2, a], fmt: fmt, refs: [{ text: '0', v: 0 }] };
      }
      if (hi === lo) hi = lo + 1;
      var refs = [{ text: unit === 'V' ? '0 V' : '0', v: 0 }];
      if (unit === 'V') {
        var top = sigs.filter(function (s) { return isFiniteNum(opValue(plotIdx, s)); })
          .reduce(function (b, s) { return !b || opValue(plotIdx, s) > opValue(plotIdx, b) ? s : b; }, null);
        if (top && opValue(plotIdx, top) !== 0) refs.push({ text: opLabel(top) + ' ' + fmt(opValue(plotIdx, top)), v: opValue(plotIdx, top) });
      }
      return { lo: lo, hi: hi, ticks: unit === 'V' ? niceTicks(lo, hi) : [lo, (lo + hi) / 2, hi], fmt: fmt, refs: refs };
    }

    function opSorted(panel, sigs, plotIdx) {
      var by = panel.sort || 'value';
      if (by === 'name') return sigs.sort(function (a, b) { var x = opLabel(a), y = opLabel(b); return x < y ? -1 : x > y ? 1 : 0; });
      if (by === 'added') return sigs.sort(function (a, b) { return panel.signals.indexOf(a.name) - panel.signals.indexOf(b.name); });
      return sigs.sort(function (a, b) { return opValue(plotIdx, b) - opValue(plotIdx, a); });
    }

    // The geometry and text of a panel's strips, shared by the SVG drawing,
    // the PNG export and the CSV.
    function opModel(panel, width) {
      var plotIdx = panel.plotIdx, W = Math.max(520, width || 868);
      var strips = opUnits(panel).map(function (unit) {
        var axis = opAxis(plotIdx, unit), ref = NamedMarkers.reference(plotIdx, unit);
        var sigs = opSorted(panel, panel.signals.map(function (n) { return Store.findSignal(plotIdx, n); })
          .filter(function (s) { return s && (s.unit || '') === unit; }), plotIdx);
        var right = OP_RIGHT + (ref ? OP_DELTA : 0), pw = W - OP_LEFT - right;
        var X = function (v) { return OP_LEFT + pw * (v - axis.lo) / (axis.hi - axis.lo); };
        var rows = sigs.map(function (sig, i) {
          var v = opValue(plotIdx, sig), d = ref && isFiniteNum(v) ? v - ref.x : null;
          return {
            sig: sig, v: v, y: OP_TOP + OP_ROW_H * i + OP_ROW_H / 2, x: X(isFiniteNum(v) ? v : 0),
            text: axis.fmt(v), color: colorFor(panel, sig.name),
            delta: d === null ? null : (d > 0 ? '+' : '') + axis.fmt(d)
          };
        });
        return { unit: unit, title: opTitle(unit), axis: axis, rows: rows, ref: ref, X: X, W: W, pw: pw,
          H: OP_TOP + OP_ROW_H * Math.max(rows.length, 1) + 30 };
      });
      return { W: W, strips: strips };
    }

    function renderStrips(panel, fresh) {
      if (!panelIsOp(panel)) return;
      var host = panel.chartEl, prev = {};
      host.querySelectorAll('.op-row').forEach(function (g) { prev[g.dataset.signal] = Number(g.dataset.y); });
      clear(host);
      host.classList.add('op-strips');
      if (!panel.signals.length) {
        var empty = ce('div', 'op-empty'); setText(empty, 'No signals in this panel'); host.appendChild(empty);
        return;
      }
      var cs = getComputedStyle(host);
      var width = host.clientWidth - (parseFloat(cs.paddingLeft) || 0) - (parseFloat(cs.paddingRight) || 0);
      panel._opWidth = width;
      opModel(panel, width).strips.forEach(function (strip) { buildStrip(panel, host, strip, prev, fresh); });
    }

    function buildStrip(panel, host, strip, prev, fresh) {
      var plotIdx = panel.plotIdx, axis = strip.axis, W = strip.W, H = strip.H;
      var h = ce('div', 'op-heading'); setText(h, strip.title); host.appendChild(h);
      var wrap = ce('div', 'op-strip'); wrap.dataset.silk = 'op-strip'; wrap.dataset.unit = strip.unit; host.appendChild(wrap);
      var svg = svgEl('svg', { width: W, height: H }, wrap, 'op-svg');
      axis.ticks.forEach(function (t) {
        svgEl('line', { x1: strip.X(t), x2: strip.X(t), y1: OP_TOP - 6, y2: H - 24 }, svg, 'op-grid');
        svgText(svg, strip.X(t), H - 8, axis.fmt(t), 'op-tick', 'middle');
      });
      axis.refs.forEach(function (r) {
        svgEl('line', { x1: strip.X(r.v), x2: strip.X(r.v), y1: OP_TOP - 14, y2: H - 24 }, svg, 'op-ref');
        svgText(svg, strip.X(r.v), OP_TOP - 18, r.text, 'op-reftext', 'middle');
      });
      if (strip.ref) svgText(svg, W - 8, OP_TOP - 18, 'Δ vs ' + strip.ref.name, 'op-reftext', 'end');
      strip.rows.forEach(function (row) {
        var sig = row.sig, y = row.y;
        var g = svgEl('g', { 'data-signal': sig.name, 'data-y': y }, svg, 'op-row' + (opHoverName === sig.name ? ' op-hover' : ''));
        var band = svgEl('rect', { x: 4, y: y - OP_ROW_H / 2 + 1, width: W - 8, height: OP_ROW_H - 2, rx: 3 }, g, 'op-band');
        svgText(g, OP_LEFT - 10, y + 4, opLabel(sig), 'op-name', 'end');
        svgEl('line', { x1: Math.min(strip.X(0), row.x), x2: Math.max(strip.X(0), row.x), y1: y, y2: y }, g, 'op-stem');
        var dot = svgEl('circle', { cx: row.x, cy: y, r: 5.5, fill: row.color }, g, 'op-dot');
        svgEl('title', {}, dot).textContent = traceTitle(sig) + ' = ' + String(row.v) + (strip.unit ? ' ' + strip.unit : '');
        svgText(g, row.x + 10, y + 4, row.text, 'op-value');
        if (row.delta !== null) svgText(g, W - 8, y + 4, row.delta, 'op-delta', 'end');
        g.addEventListener('mouseenter', function () { opHover(sig.name); });
        g.addEventListener('mouseleave', function () { opHover(null); });
        var isFresh = !!fresh && fresh.indexOf(sig.name) !== -1;
        if (isFresh) {
          g.classList.add('op-fresh');
          var tag = svgText(g, W - OP_RIGHT + 40, y + 4, 'added', 'op-added');
          var anims = [];
          if (band.animate) anims.push(band.animate([{ opacity: 0.35 }, { opacity: 0.35, offset: 0.6 }, { opacity: 0 }], { duration: 2500, fill: 'forwards' }));
          if (tag.animate) anims.push(tag.animate([{ opacity: 1 }, { opacity: 1, offset: 0.6 }, { opacity: 0 }], { duration: 2500, fill: 'forwards' }));
          // Release the finished highlight so ordinary hover styling rules again.
          setTimeout(function () {
            anims.forEach(function (a) { a.cancel(); });
            g.classList.remove('op-fresh');
            if (tag.parentNode) tag.parentNode.removeChild(tag);
          }, 2500);
        }
        var old = prev[sig.name];
        if (g.animate && old !== undefined && old !== y) g.animate([{ transform: 'translateY(' + (old - y) + 'px)' }, { transform: 'none' }], { duration: 350, easing: 'ease-out' });
        else if (g.animate && old === undefined && isFresh) g.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 350 });
      });
      // The scale lives on the live strip: a redraw (a drag makes the marker the
      // reference, which adds the difference column and narrows the plot area)
      // replaces this element, so pointer values are always read from the current one.
      wrap._opScale = { lo: axis.lo, hi: axis.hi, pw: strip.pw };
      // Pointer value for "+ marker", cached so the toolbar click keeps its target.
      svg.addEventListener('mousemove', function (e) {
        lastOpStrip = { plotIdx: plotIdx, unit: strip.unit, value: opValueAt(panel, strip.unit, e.clientX) };
      });
      NamedMarkers.drawStrip({ panel: panel, wrap: wrap, svg: svg, unit: strip.unit, X: strip.X, axis: axis, top: OP_TOP - 14, bottom: H - 24 });
    }

    // Axis value under a pointer, from the strip of that unit that is on screen now.
    function opValueAt(panel, unit, clientX) {
      var wraps = panel.chartEl.querySelectorAll('.op-strip');
      for (var i = 0; i < wraps.length; i++) {
        if (wraps[i].dataset.unit !== unit || !wraps[i]._opScale) continue;
        var sc = wraps[i]._opScale, rect = wraps[i].querySelector('svg').getBoundingClientRect();
        return sc.lo + (clientX - rect.left - OP_LEFT) / sc.pw * (sc.hi - sc.lo);
      }
      return null;
    }

    // Paint a panel's strips on a canvas (PNG export); returns the height used.
    function paintStrips(ctx, panel, ox, oy, dpr) {
      var model = opModel(panel, panel._opWidth || 868), y0 = oy;
      var font = function (px, bold) { return FontScale.css(px * dpr, bold); };
      model.strips.forEach(function (st) {
        ctx.font = font(FontScale.basePx() - 1); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
        ctx.fillStyle = theme('--text-1', '#a7b0bd'); ctx.fillText(st.title, ox, y0 + 14 * dpr);
        var sy = y0 + 20 * dpr, X = function (v) { return ox + st.X(v) * dpr; };
        var ink = function (name, fb) { return theme(name, fb); };
        ctx.lineWidth = 1;
        st.axis.ticks.forEach(function (t) {
          ctx.strokeStyle = ink('--border-soft', '#232933'); ctx.beginPath();
          ctx.moveTo(X(t), sy + (OP_TOP - 6) * dpr); ctx.lineTo(X(t), sy + (st.H - 24) * dpr); ctx.stroke();
          ctx.fillStyle = ink('--text-2', '#6d7686'); ctx.font = font(FontScale.basePx() - 3); ctx.textAlign = 'center';
          ctx.fillText(st.axis.fmt(t), X(t), sy + (st.H - 8) * dpr);
        });
        ctx.setLineDash([4 * dpr, 3 * dpr]);
        st.axis.refs.forEach(function (r) {
          ctx.strokeStyle = ink('--amber', '#e0a640'); ctx.beginPath();
          ctx.moveTo(X(r.v), sy + (OP_TOP - 14) * dpr); ctx.lineTo(X(r.v), sy + (st.H - 24) * dpr); ctx.stroke();
          ctx.fillStyle = ink('--amber', '#e0a640'); ctx.textAlign = 'center';
          ctx.fillText(r.text, X(r.v), sy + (OP_TOP - 18) * dpr);
        });
        NamedMarkers.list(panel.plotIdx).forEach(function (m) {
          if (m.unit !== st.unit) return;
          ctx.strokeStyle = m.color; ctx.beginPath();
          ctx.moveTo(X(m.x), sy + (OP_TOP - 14) * dpr); ctx.lineTo(X(m.x), sy + (st.H - 24) * dpr); ctx.stroke();
        });
        ctx.setLineDash([]);
        st.rows.forEach(function (row) {
          var y = sy + row.y * dpr;
          ctx.font = font(FontScale.basePx() - 2); ctx.fillStyle = ink('--text-0', '#dfe4ea');
          ctx.textAlign = 'right'; ctx.fillText(opLabel(row.sig), ox + (OP_LEFT - 10) * dpr, y + 4 * dpr);
          ctx.strokeStyle = ink('--border', '#2b323d'); ctx.beginPath();
          ctx.moveTo(X(0), y); ctx.lineTo(ox + row.x * dpr, y); ctx.stroke();
          ctx.fillStyle = row.color; ctx.beginPath(); ctx.arc(ox + row.x * dpr, y, 5.5 * dpr, 0, 2 * Math.PI); ctx.fill();
          ctx.fillStyle = ink('--text-0', '#dfe4ea'); ctx.textAlign = 'left';
          ctx.fillText(row.text, ox + (row.x + 10) * dpr, y + 4 * dpr);
          if (row.delta !== null) { ctx.textAlign = 'right'; ctx.fillStyle = ink('--text-1', '#a7b0bd'); ctx.fillText(row.delta, ox + (st.W - 8) * dpr, y + 4 * dpr); }
        });
        NamedMarkers.list(panel.plotIdx).forEach(function (m) {
          if (m.unit !== st.unit) return;
          Exporter.tag(ctx, m.name + (m.ref ? ' (ref)' : ''), m.color, X(m.x), sy + (st.H - 24) * dpr, dpr, 'below');
        });
        y0 += (20 + st.H + 4) * dpr;
      });
      return y0 - oy;
    }

    // Width the export must allow: the strip model's own (clamped) width.
    function opModelWidth(panel) { return opModel(panel, panel._opWidth || 868).W; }

    function refreshOpPanels(plotIdx) {
      for (var i = 0; i < panels.length; i++) {
        if (panels[i].plotIdx === plotIdx && panelIsOp(panels[i])) renderStrips(panels[i]);
      }
    }

    function refreshValues() {
      if (opPlot(activePlotIdx)) refreshOpPanels(activePlotIdx);
    }

    function renderConstants(plot) {
      var host = el('constants-view');
      clear(host);
      var table = ce('table');
      var thead = ce('tr');
      ['signal', 'value'].forEach(function (h) {
        var th = ce('th');
        setText(th, h);
        thead.appendChild(th);
      });
      table.appendChild(thead);
      for (var i = 0; i < plot.signals.length; i++) {
        var sig = plot.signals[i];
        var rec = Store.levelFor(plot === Store.plot(activePlotIdx) ? activePlotIdx : activePlotIdx, sig, 1);
        var v = rec.yRaw.length ? rec.yRaw[0] : null;
        var tr = ce('tr');
        var tdName = ce('td');
        setText(tdName, sig.label || sig.name);
        tdName.title = traceTitle(sig);
        tr.appendChild(tdName);
        var tdVal = ce('td', 'val');
        setText(tdVal, Fmt.num(isFiniteNum(v) ? v : null, sig.unit || ''));
        tr.appendChild(tdVal);
        table.appendChild(tr);
      }
      host.appendChild(table);
    }

    function onResize() {
      for (var i = 0; i < panels.length; i++) {
        var p = panels[i];
        if (panelIsOp(p)) {
          var cs = getComputedStyle(p.chartEl);
          var w0 = p.chartEl.clientWidth - (parseFloat(cs.paddingLeft) || 0) - (parseFloat(cs.paddingRight) || 0);
          if (w0 > 0 && w0 !== p._opWidth) renderStrips(p);
          continue;
        }
        if (!p.uplot) continue;
        var w = chart_width(p);
        if (w > 0) p.uplot.setSize({ width: w, height: 258 });
      }
    }

    return {
      refreshValues: refreshValues, replaceDerived: replaceDerived,
      opActive: function () { return opPlot(activePlotIdx); }, opHover: opHover,
      lastOpStrip: function () { return lastOpStrip; }, paintStrips: paintStrips,
      refreshOpPanels: refreshOpPanels, panelIsOp: panelIsOp, opValueAt: opValueAt, opModelWidth: opModelWidth,
      loadPlot: loadPlot, addPanel: addPanel, removePanel: removePanel,
      addSignal: addSignal, removeSignal: removeSignal, plotVisible: plotVisible,
      clearFocusedPanel: clearFocusedPanel, clearPanel: clearPanel, setXLog: setXLog, resetZoom: resetZoom, fitAll: fitAll,
      zoomAbout: zoomAbout, placeCursorAtMouse: placeCursorAtMouse,
      redrawPanels: redrawPanels, updateAllMonitorTables: updateAllMonitorTables,
      scheduleStateSave: scheduleStateSave, persistCurrentLayout: persistCurrentLayoutNow,
      teardown: teardown,
      onResize: onResize, isXLog: function () { return xLogEnabled; },
      setSampleMarks: setSampleMarks, sampleMarksOn: function () { return sampleMarksOn; },
      setCols: setCols, getCols: getCols, panelsForPlot: panelsForPlot,
      getFocused: getFocused, setPanelYMode: setPanelYMode, setXRange: setXRangeAll,
      movePanel: movePanelByIndex,
      setYRange: applyPanelYRange, autoscaleY: autoscaleYPanel, autoscaleYFocused: autoscaleYFocused,
      setSeriesStyle: applySeriesStylePatch,
      rebuildAllForFontChange: rebuildAllForFontChange,
    };
  })();

  // ======================================================================
  // NamedMarkers -- persistent, named, per-plot x markers beyond A/B
  // (v1.2 item 3). Reuses the A/B badge machinery's shape (a draggable DOM
  // tab living inside uPlot's `.u-over`, plus a dashed canvas line drawn in
  // the same `draw` hook) but stacks its badges at the BOTTOM of the panel
  // so they never collide with the A/B badges pinned to the top.
  // ======================================================================

  var NamedMarkers = (function () {
    var byPlot = {};   // plotIdx -> [{id, name, x, color}], kept sorted by x
    var nextNum = {};  // plotIdx -> highest "Mn" number issued so far
    var nextId = 1;
    var COLORS = ['#e0575a', '#5ad1a0', '#c98fd6', '#e0a640', '#9b7fe0', '#7fc97f', '#4fd1c5', '#c9c15a'].map(function (fallback, i) { return theme('--marker-' + (i + 1), fallback); });

    function list(plotIdx) {
      if (!byPlot[plotIdx]) byPlot[plotIdx] = [];
      return byPlot[plotIdx];
    }

    // TASK-591: on an operating point a marker is a named reference level on
    // the strip axis of one unit (`unit` beside `x`, which holds the level).
    // Its array stays in insertion order (saved and restored as such), so
    // "most recently added" survives rename, edit and reopen; waveform
    // markers stay sorted by x.
    function isOp(plotIdx) {
      var plot = Store.plot(plotIdx);
      return !!plot && plot.kind === 'op';
    }

    function sortByX(plotIdx) {
      if (isOp(plotIdx)) return;
      list(plotIdx).sort(function (a, b) { return a.x - b.x; });
    }

    // Exactly one marker per unit is the reference (ref: true).
    function makeRef(plotIdx, m) {
      list(plotIdx).forEach(function (o) { if (o.unit === m.unit) o.ref = (o === m); });
    }

    function lastOfUnit(plotIdx, unit) {
      var arr = list(plotIdx);
      for (var i = arr.length - 1; i >= 0; i--) if (arr[i].unit === unit) return arr[i];
      return null;
    }

    // Per unit: keep the saved single ref; otherwise (none or several) the
    // last marker of that unit in saved order.
    function normalizeRefs(plotIdx) {
      var arr = list(plotIdx), seen = {};
      arr.forEach(function (m) {
        if (typeof m.unit !== 'string' || seen[m.unit]) return;
        seen[m.unit] = true;
        var refs = arr.filter(function (o) { return o.unit === m.unit && o.ref; });
        makeRef(plotIdx, refs.length === 1 ? refs[0] : lastOfUnit(plotIdx, m.unit));
      });
    }

    function reference(plotIdx, unit) {
      var arr = list(plotIdx);
      for (var i = 0; i < arr.length; i++) if (arr[i].unit === unit && arr[i].ref) return arr[i];
      return lastOfUnit(plotIdx, unit);
    }

    function setReference(plotIdx, id) {
      var arr = list(plotIdx);
      for (var i = 0; i < arr.length; i++) if (arr[i].id === id) { makeRef(plotIdx, arr[i]); break; }
      afterChange(plotIdx);
    }

    function colorForIndex(i) { return COLORS[i % COLORS.length]; }

    function nextName(plotIdx) {
      var n = (nextNum[plotIdx] || 0) + 1;
      nextNum[plotIdx] = n;
      return 'M' + n;
    }

    function bumpNextNumFromName(plotIdx, name) {
      var m = /^M(\d+)$/.exec(name || '');
      if (m) nextNum[plotIdx] = Math.max(nextNum[plotIdx] || 0, parseInt(m[1], 10));
    }

    function add(plotIdx, xval, unit) {
      var arr = list(plotIdx), op = isOp(plotIdx);
      var marker = {
        id: nextId++, name: nextName(plotIdx),
        x: op ? xval : Store.snap(plotIdx, xval), color: colorForIndex(arr.length)
      };
      if (op) marker.unit = unit;
      arr.push(marker);
      sortByX(plotIdx);
      if (op) makeRef(plotIdx, marker); // the newest marker of a unit is its reference
      afterChange(plotIdx);
      return marker;
    }

    function addAtMouse() {
      if (isOp(activePlotIdx)) {
        var strip = PanelManager.lastOpStrip();
        if (!strip || strip.plotIdx !== activePlotIdx) { showToast('hover a strip, then press m'); return; }
        add(activePlotIdx, strip.value, strip.unit);
        return;
      }
      var ref = (lastHoverPanel && lastHoverPanel.plotIdx === activePlotIdx)
        ? lastHoverPanel : PanelManager.panelsForPlot(activePlotIdx)[0];
      if (!ref || !ref.uplot) { showToast('hover a panel, then press m'); return; }
      // Uses the panel's cached last-known cursor x, not the live
      // `u.cursor.left` -- clicking the "+ marker" toolbar button moves the
      // mouse off the chart before this handler runs, which would otherwise
      // always read back "no position" (see the `setCursor` hook comment).
      if (ref._lastCursorX == null) { showToast('move the mouse over a panel first'); return; }
      add(activePlotIdx, ref._lastCursorX);
    }

    function setX(plotIdx, id, xval) {
      var arr = list(plotIdx);
      for (var i = 0; i < arr.length; i++) {
        if (arr[i].id === id) {
          if (isOp(plotIdx)) { arr[i].x = xval; makeRef(plotIdx, arr[i]); } // dragged or typed: now the reference
          else arr[i].x = Store.snap(plotIdx, xval);
          break;
        }
      }
      sortByX(plotIdx);
      afterChange(plotIdx);
    }

    function rename(plotIdx, id, name) {
      if (!name) { renderTable(plotIdx); return; }
      var arr = list(plotIdx);
      for (var i = 0; i < arr.length; i++) {
        if (arr[i].id === id) { arr[i].name = name; break; }
      }
      afterChange(plotIdx);
    }

    function remove(plotIdx, id) {
      var arr = list(plotIdx);
      for (var i = 0; i < arr.length; i++) {
        if (arr[i].id === id) {
          var gone = arr.splice(i, 1)[0];
          if (gone.ref) { var next = lastOfUnit(plotIdx, gone.unit); if (next) makeRef(plotIdx, next); }
          break;
        }
      }
      afterChange(plotIdx);
    }

    function clearAll(plotIdx) {
      byPlot[plotIdx] = [];
      afterChange(plotIdx);
    }

    function afterChange(plotIdx) {
      PanelManager.redrawPanels(plotIdx);
      renderTable(plotIdx);
      PanelManager.scheduleStateSave();
    }

    function forPersist(plotIdx) {
      return list(plotIdx).map(function (m) {
        var out = { name: m.name, x: m.x, color: m.color };
        if (typeof m.unit === 'string') { out.unit = m.unit; if (m.ref) out.ref = true; }
        return out;
      });
    }

    function restore(plotIdx, arr) {
      var out = (arr || []).map(function (m) {
        var o = { id: nextId++, name: m.name, x: m.x, color: m.color || colorForIndex(0) };
        if (typeof m.unit === 'string') { o.unit = m.unit; if (m.ref) o.ref = true; }
        return o;
      });
      byPlot[plotIdx] = out;
      if (isOp(plotIdx)) normalizeRefs(plotIdx);
      nextNum[plotIdx] = 0;
      out.forEach(function (m) { bumpNextNumFromName(plotIdx, m.name); });
    }

    // ---- canvas line + DOM badge, drawn from the panel's uPlot draw hook ----

    function draw(u, panel) {
      var arr = list(panel.plotIdx);
      if (arr.length) {
        var ctx = u.ctx;
        ctx.save();
        ctx.setLineDash([2, 2]);
        ctx.lineWidth = 1;
        for (var i = 0; i < arr.length; i++) {
          var m = arr[i];
          var xpix = u.valToPos(m.x, 'x', true);
          if (xpix < u.bbox.left - 1 || xpix > u.bbox.left + u.bbox.width + 1) continue;
          ctx.strokeStyle = m.color;
          ctx.beginPath();
          ctx.moveTo(xpix, u.bbox.top);
          ctx.lineTo(xpix, u.bbox.top + u.bbox.height);
          ctx.stroke();
        }
        ctx.setLineDash([]);
        ctx.restore();
      }
      // DOM badges need `.u-over` to exist; on the very first internal draw
      // (inside `new uPlot(...)`) it does not yet -- same reason
      // CursorEngine.draw guards on `panel.overEl` before touching it.
      if (panel.overEl) updateBadges(u, panel, arr);
    }

    function updateBadges(u, panel, arr) {
      if (!panel.markerBadges) panel.markerBadges = {};
      var seen = {};
      for (var i = 0; i < arr.length; i++) {
        var m = arr[i];
        seen[m.id] = true;
        var badge = panel.markerBadges[m.id];
        if (!badge) {
          badge = buildBadge(panel, m);
          panel.markerBadges[m.id] = badge;
        }
        badge.el.style.background = m.color;
        if (!badge.editing) setText(badge.labelEl, m.name);
        var xCss = u.valToPos(m.x, 'x');
        var w = panel.overEl.clientWidth;
        if (xCss >= -20 && xCss <= w + 20) {
          badge.el.hidden = false;
          badge.el.style.left = xCss + 'px';
        } else {
          badge.el.hidden = true;
        }
      }
      for (var id in panel.markerBadges) {
        if (!seen[id]) {
          var stale = panel.markerBadges[id];
          if (stale.el.parentNode) stale.el.parentNode.removeChild(stale.el);
          delete panel.markerBadges[id];
        }
      }
    }

    function buildBadge(panel, m) {
      var badgeEl = ce('div', 'marker-badge');
      var labelEl = ce('span');
      badgeEl.appendChild(labelEl);
      badgeEl.title = 'Drag to move; double-click to rename';
      var rec = { el: badgeEl, labelEl: labelEl, editing: false };
      badgeEl.addEventListener('mousedown', function (e) {
        if (e.detail >= 2) return; // let dblclick handle renaming, not a drag
        e.preventDefault();
        e.stopPropagation();
        var u = panel.uplot;
        var rect = panel.overEl.getBoundingClientRect();
        function onMove(ev) { setX(panel.plotIdx, m.id, u.posToVal(ev.clientX - rect.left, 'x')); }
        function onUp() {
          removeDragListener('mousemove', onMove);
          removeDragListener('mouseup', onUp);
        }
        listenDrag('mousemove', onMove);
        listenDrag('mouseup', onUp);
      });
      badgeEl.addEventListener('dblclick', function (e) {
        e.stopPropagation();
        rec.editing = true;
        var current = m.name;
        inlineTextEdit(labelEl, current, function (newName) {
          rec.editing = false;
          if (newName) rename(panel.plotIdx, m.id, newName);
          else setText(labelEl, current);
        });
      });
      if (panel.overEl) panel.overEl.appendChild(badgeEl);
      return rec;
    }

    // ---- operating-point strips: a reference level in the strips of its unit ----

    var markerMenu = null;
    function openMarkerMenu(anchor, plotIdx, m) {
      if (!markerMenu) {
        markerMenu = ce('div', 'dropdown-menu anchored-menu');
        markerMenu.hidden = true;
        root.appendChild(markerMenu);
      }
      toggleDropdown(markerMenu, function (menu) {
        menuHeading(menu, 'Marker ' + m.name);
        menuButton(menu, 'Use as reference', 'Row differences in strips of ' + (m.unit || 'this unit') + ' use this marker',
          function () { setReference(plotIdx, m.id); }, !!m.ref);
        menuButton(menu, 'Delete', 'Remove marker ' + m.name, function () { remove(plotIdx, m.id); });
      }, anchor, 'left');
    }

    // c: {panel, wrap, svg, unit, X, axis, top, bottom} from PanelManager (pointer values come from PanelManager.opValueAt).
    function drawStrip(c) {
      var plotIdx = c.panel.plotIdx;
      list(plotIdx).forEach(function (m) {
        if (m.unit !== c.unit) return;
        var x = c.X(m.x);
        var line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('class', 'op-marker');
        line.setAttribute('x1', x); line.setAttribute('x2', x);
        line.setAttribute('y1', c.top); line.setAttribute('y2', c.bottom);
        line.style.stroke = m.color;
        c.svg.appendChild(line);
        var badge = ce('div', 'marker-badge op-marker-badge' + (m.ref ? ' is-ref' : ''));
        badge.style.left = x + 'px';
        badge.style.borderColor = m.color;
        if (m.ref) badge.style.background = m.color; else badge.style.color = m.color;
        var label = ce('span'); setText(label, m.name + (m.ref ? ' \u00b7 ref' : ''));
        badge.appendChild(label);
        badge.title = 'Drag to move (it becomes the reference); double-click to rename; right-click for more';
        badge.addEventListener('mousedown', function (e) {
          if (e.detail >= 2 || e.button !== 0) return;
          e.preventDefault(); e.stopPropagation();
          function onMove(ev) { var v = PanelManager.opValueAt(c.panel, c.unit, ev.clientX); if (v !== null) setX(plotIdx, m.id, v); }
          function onUp() { removeDragListener('mousemove', onMove); removeDragListener('mouseup', onUp); }
          listenDrag('mousemove', onMove); listenDrag('mouseup', onUp);
        });
        badge.addEventListener('dblclick', function (e) {
          e.stopPropagation();
          var current = m.name;
          inlineTextEdit(label, current, function (newName) {
            if (newName) rename(plotIdx, m.id, newName); else setText(label, current);
          });
        });
        badge.addEventListener('contextmenu', function (e) {
          e.preventDefault(); e.stopPropagation();
          openMarkerMenu(badge, plotIdx, m);
        });
        c.wrap.appendChild(badge);
      });
    }

    // ---- the collapsible markers table ----

    function visibleSignalNames(plotIdx) {
      var out = [];
      var ps = PanelManager.panelsForPlot(plotIdx);
      for (var i = 0; i < ps.length; i++) {
        for (var j = 0; j < ps[i].signals.length; j++) {
          if (out.indexOf(ps[i].signals[j]) === -1) out.push(ps[i].signals[j]);
        }
      }
      return out;
    }

    // Operating point: rows are markers (insertion order); one column per
    // signal plotted in ANY panel of the plot, holding signal - level when
    // the units match and a dash when they differ.
    function renderOpTable(plotIdx, table) {
      var arr = list(plotIdx);
      var sigNames = visibleSignalNames(plotIdx);
      var thead = ce('tr');
      ['name', 'unit', 'level'].concat(sigNames, ['reference', '']).forEach(function (h) {
        var th = ce('th'); setText(th, h); thead.appendChild(th);
      });
      table.appendChild(thead);
      arr.forEach(function (m) {
        var tr = ce('tr');
        var tdName = ce('td', 'sig-name');
        var sw = ce('span', 'swatch-inline'); sw.style.background = m.color; tdName.appendChild(sw);
        var nameText = ce('span'); setText(nameText, m.name); tdName.appendChild(nameText);
        tdName.title = 'Double-click to rename';
        tdName.addEventListener('dblclick', function (e) {
          e.stopPropagation();
          var current = m.name;
          inlineTextEdit(nameText, current, function (newName) {
            if (newName) rename(plotIdx, m.id, newName); else setText(nameText, current);
          });
        });
        tr.appendChild(tdName);
        var tdUnit = ce('td'); setText(tdUnit, typeof m.unit === 'string' ? (m.unit || '(none)') : '\u2014'); tr.appendChild(tdUnit);
        var tdLevel = ce('td'), span = ce('span');
        ValueEditor.attach(span, {
          getValue: function () { return m.x; },
          setValue: function (v) { setX(plotIdx, m.id, v); },
          formatFn: function (v) { return Fmt.num(v, m.unit || ''); }
        });
        tdLevel.appendChild(span); tr.appendChild(tdLevel);
        sigNames.forEach(function (n) {
          var sig = Store.findSignal(plotIdx, n), td = ce('td');
          var same = sig && typeof m.unit === 'string' && (sig.unit || '') === m.unit;
          var v = same ? Store.levelFor(plotIdx, sig, 1).yRaw[0] : null;
          setText(td, same && isFiniteNum(v) ? ((v - m.x) > 0 ? '+' : '') + Fmt.num(v - m.x, sig.unit || '') : '\u2014');
          tr.appendChild(td);
        });
        var tdRef = ce('td');
        if (m.ref) setText(tdRef, 'ref');
        else if (typeof m.unit === 'string') {
          var useBtn = ce('button'); setText(useBtn, 'Use as reference');
          useBtn.addEventListener('click', function () { setReference(plotIdx, m.id); });
          tdRef.appendChild(useBtn);
        }
        tr.appendChild(tdRef);
        var tdDel = ce('td'), delBtn = ce('button', 'marker-del');
        setText(delBtn, '\u00d7'); delBtn.title = 'Delete marker ' + m.name;
        delBtn.addEventListener('click', function () { remove(plotIdx, m.id); });
        tdDel.appendChild(delBtn); tr.appendChild(tdDel);
        table.appendChild(tr);
      });
      setText(el('markers-panel-title'), 'Markers (' + arr.length + ')');
    }

    function renderTable(plotIdx) {
      if (plotIdx !== activePlotIdx) return;
      var table = el('markers-table');
      if (!table) return;
      clear(table);
      if (isOp(plotIdx)) { renderOpTable(plotIdx, table); return; }
      var arr = list(plotIdx);
      var plot = Store.plot(plotIdx);
      var xUnit = (plot && plot.x && plot.x.unit) || '';
      var sigNames = visibleSignalNames(plotIdx);
      var thead = ce('tr');
      ['name', 'x'].concat(sigNames, ['Δx vs prev', '']).forEach(function (h) {
        var th = ce('th');
        setText(th, h);
        thead.appendChild(th);
      });
      table.appendChild(thead);
      var prevX = null;
      for (var i = 0; i < arr.length; i++) {
        var m = arr[i];
        var tr = ce('tr');

        var tdName = ce('td', 'sig-name');
        var sw = ce('span', 'swatch-inline');
        sw.style.background = m.color;
        tdName.appendChild(sw);
        var nameText = ce('span');
        setText(nameText, m.name);
        tdName.appendChild(nameText);
        tdName.title = 'Double-click to rename';
        tdName.addEventListener('dblclick', function (mm, nameSpan) {
          return function (e) {
            e.stopPropagation();
            var current = mm.name;
            inlineTextEdit(nameSpan, current, function (newName) {
              if (newName) rename(plotIdx, mm.id, newName);
              else setText(nameSpan, current);
            });
          };
        }(m, nameText));
        tr.appendChild(tdName);

        var tdX = ce('td');
        var xSpan = ce('span');
        ValueEditor.attach(xSpan, {
          getValue: function (mm) { return function () { return mm.x; }; }(m),
          setValue: function (mm) { return function (v) { setX(plotIdx, mm.id, v); }; }(m),
          formatFn: function (v) { return Fmt.num(v, xUnit); }
        });
        tdX.appendChild(xSpan);
        tr.appendChild(tdX);

        for (var s = 0; s < sigNames.length; s++) {
          var sig = Store.findSignal(plotIdx, sigNames[s]);
          var v = sig ? Store.sampleAt(plotIdx, sig, m.x) : null;
          var td = ce('td');
          setText(td, Fmt.num(v, sig ? sig.unit : ''));
          tr.appendChild(td);
        }

        var tdDelta = ce('td');
        setText(tdDelta, prevX == null ? '—' : Fmt.num(m.x - prevX, xUnit));
        tr.appendChild(tdDelta);
        prevX = m.x;

        var tdDel = ce('td');
        var delBtn = ce('button', 'marker-del');
        setText(delBtn, '×');
        delBtn.title = 'Delete marker ' + m.name;
        delBtn.addEventListener('click', function (mm) {
          return function () { remove(plotIdx, mm.id); };
        }(m));
        tdDel.appendChild(delBtn);
        tr.appendChild(tdDel);

        table.appendChild(tr);
      }
      setText(el('markers-panel-title'), 'Markers (' + arr.length + ')');
    }

    return {
      list: list, add: add, addAtMouse: addAtMouse, setX: setX, rename: rename,
      remove: remove, clearAll: clearAll, forPersist: forPersist, restore: restore,
      draw: draw, renderTable: renderTable, drawStrip: drawStrip,
      reference: reference, setReference: setReference, lastOfUnit: lastOfUnit
    };
  })();

  // ======================================================================
  // Exporter -- CSV (v1.2 item 1) and PNG (v1.2 item 2) export, per-panel and
  // "all plotted signals" / "all panels". CSV always reads level-0 data
  // (Store.rawYLevel0), never a decimated level. PNG composites uPlot's own
  // canvas (which already contains traces + axes + our cursor/marker/
  // annotation overlays, all drawn in the same `draw` hook) onto an offscreen
  // canvas with a page-matching background and a title bar.
  // ======================================================================

  var Exporter = (function () {
    var CSV_WARN_BYTES = 50 * 1000 * 1000;

    function slugify(text) {
      var s = String(text == null ? '' : text).replace(/[^A-Za-z0-9_-]+/g, '_');
      s = s.replace(/_+/g, '_').replace(/^_+|_+$/g, '');
      return s || 'silk_view';
    }

    function runSlug() {
      var run = PAYLOAD && PAYLOAD.run;
      return slugify(run && (run.title || run.source));
    }

    function downloadBlob(blob, filename) {
      if (destroyed) return;
      var url = makeObjectURL(blob);
      var a = ce('a');
      a.href = url;
      a.download = filename;
      root.appendChild(a);
      a.click();
      root.removeChild(a);
      setTimeout(function () { revokeObjectURL(url); }, 4000);
    }

    // ---------------------------------------------------------------- CSV

    function numStr(v) {
      if (v == null) return '';
      if (v !== v) return 'NaN';         // Python/numpy's float() parses
      if (v === Infinity) return 'Infinity'; // "NaN"/"Infinity"/"-Infinity"
      if (v === -Infinity) return '-Infinity'; // case-insensitively.
      return String(v); // JS Number#toString: shortest round-tripping repr.
    }

    // Text cells only (signal names, the x-axis name) -- never called on
    // numeric data, which goes through `numStr` straight into the row with
    // no quoting (see buildCsvBlob), so a legitimate leading '-' on a
    // negative number is never at risk of being treated as formula text
    // here. Signal/run-title strings are attacker-adjacent (rawfile-
    // sourced, see the file header note); a cell whose first character is
    // one of `= + - @` or a tab/CR is opened by common spreadsheet apps as
    // a formula the moment the CSV is opened, so such cells get a leading
    // single quote inside the quoted field per OWASP's CSV-injection
    // guidance -- this defuses the formula while leaving the visible text
    // (after the leading quote character) intact.
    var CSV_FORMULA_LEAD = /^[=+\-@\t\r]/;
    function quoteCsv(s) {
      var str = String(s);
      if (CSV_FORMULA_LEAD.test(str)) str = "'" + str;
      return '"' + str.replace(/"/g, '""') + '"';
    }

    function fullDomain(plotIdx) {
      var xArr = Store.xLevel0(plotIdx);
      return xArr.length ? [xArr[0], xArr[xArr.length - 1]] : [0, 0];
    }

    function scopeRange(plotIdx, refPanel, scope) {
      if (scope === 'cursor') {
        var c = CursorEngine.get(plotIdx);
        if (c.A == null || c.B == null) return fullDomain(plotIdx);
        return [Math.min(c.A, c.B), Math.max(c.A, c.B)];
      }
      if (scope === 'visible' && refPanel && refPanel.uplot) {
        return [refPanel.uplot.scales.x.min, refPanel.uplot.scales.x.max];
      }
      return fullDomain(plotIdx);
    }

    function collectColumns(plotIdx, names) {
      var cols = [];
      for (var i = 0; i < names.length; i++) {
        var sig = Store.findSignal(plotIdx, names[i]);
        if (!sig) continue;
        cols.push({ name: sig.name, arr: Store.rawYLevel0(plotIdx, sig) });
      }
      return cols;
    }

    // Builds the Blob from chunked string parts (never one giant
    // concatenated string) so a large export does the same O(n) work either
    // way but never holds one multi-hundred-MB JS string mid-build.
    function buildCsvBlob(xArr, xName, cols, i0, i1) {
      var CHUNK = 20000;
      var parts = [quoteCsv(xName) + ',' + cols.map(function (c) { return quoteCsv(c.name); }).join(',') + '\n'];
      var buf = [];
      for (var i = i0; i < i1; i++) {
        var row = numStr(xArr[i]);
        for (var c = 0; c < cols.length; c++) row += ',' + numStr(cols[c].arr[i]);
        buf.push(row);
        if (buf.length >= CHUNK) { parts.push(buf.join('\n') + '\n'); buf = []; }
      }
      if (buf.length) parts.push(buf.join('\n') + '\n');
      return new Blob(parts, { type: 'text/csv' });
    }

    function confirmBigExport(rows, ncols) {
      // Average field+comma width, measured against sample_tran_big's real
      // export (float64 level-0 values print up to ~17 significant digits
      // via JS Number#toString, not the ~13 a leaner guess assumed --
      // verified empirically: a 1e6x2 export came out to ~21 bytes/field).
      var bytes = rows * (ncols + 1) * 20;
      if (bytes <= CSV_WARN_BYTES) return true;
      var mb = (bytes / 1e6).toFixed(0);
      return window.confirm(
        'This export is about ' + rows + ' rows × ' + (ncols + 1) +
        ' columns (~' + mb + ' MB estimated) and may take a while. Continue?'
      );
    }

    function doCsvExport(plotIdx, names, xName, scope, refPanel, filename) {
      var xArr = Store.xLevel0(plotIdx);
      if (!xArr.length) { showToast('no data points to export'); return; }
      var range = scopeRange(plotIdx, refPanel, scope);
      var i0 = bsearchLeft(xArr, Math.min(range[0], range[1]));
      var i1 = bsearchRight(xArr, Math.max(range[0], range[1]));
      if (i1 <= i0) { showToast('export scope has no samples'); return; }
      var cols = collectColumns(plotIdx, names);
      if (!cols.length) { showToast('no signals to export'); return; }
      if (!confirmBigExport(i1 - i0, cols.length)) { showToast('export cancelled'); return; }
      var blob = buildCsvBlob(xArr, xName, cols, i0, i1);
      downloadBlob(blob, filename);
      showToast('exported ' + (i1 - i0) + ' rows × ' + (cols.length + 1) + ' cols to ' + filename);
    }

    // Operating point: one row per signal (value, unit), then one fixed
    // `delta:<marker> (<unit>)` column per marker in insertion order holding
    // signal - level when the units match and empty otherwise. Text cells are
    // quoted, numbers use numStr exactly as the waveform export does.
    function opCsv(plotIdx, names, filename, scope) {
      if (scope === 'cursor') { showToast('cursor export needs an x axis; use visible or full'); return; }
      var markers = NamedMarkers.list(plotIdx).filter(function (m) { return typeof m.unit === 'string'; });
      var rows = [['signal', 'value', 'unit'].concat(markers.map(function (m) { return 'delta:' + m.name + ' (' + m.unit + ')'; }))
        .map(quoteCsv).join(',')];
      names.forEach(function (name) {
        var sig = Store.findSignal(plotIdx, name);
        if (!sig) return;
        var v = Store.levelFor(plotIdx, sig, 1).yRaw[0], unit = sig.unit || '';
        var cells = [quoteCsv(sig.name), numStr(v), quoteCsv(unit)];
        markers.forEach(function (m) { cells.push(m.unit === unit ? numStr(v - m.x) : ''); });
        rows.push(cells.join(','));
      });
      if (rows.length === 1) { showToast('no signals to export'); return; }
      downloadBlob(new Blob([rows.join('\n') + '\n'], { type: 'text/csv' }), filename);
      showToast('exported ' + (rows.length - 1) + ' signals to ' + filename);
    }

    function exportPanelCsv(panel, scope) {
      var plotIdx = panel.plotIdx;
      var idx = PanelManager.panelsForPlot(plotIdx).indexOf(panel);
      if (PanelManager.panelIsOp(panel)) { opCsv(plotIdx, panel.signals, runSlug() + '_panel' + (idx + 1) + '.csv', scope); return; }
      var xName = (Store.plot(plotIdx).x || {}).name || 'x';
      doCsvExport(plotIdx, panel.signals, xName, scope, panel, runSlug() + '_panel' + (idx + 1) + '.csv');
    }

    function exportAllCsv(plotIdx, scope) {
      var panelsOfPlot = PanelManager.panelsForPlot(plotIdx);
      var names = [];
      for (var i = 0; i < panelsOfPlot.length; i++) {
        for (var j = 0; j < panelsOfPlot[i].signals.length; j++) {
          if (names.indexOf(panelsOfPlot[i].signals[j]) === -1) names.push(panelsOfPlot[i].signals[j]);
        }
      }
      if (!names.length) { showToast('no signals plotted'); return; }
      if (opPlotNow(plotIdx)) { opCsv(plotIdx, names, runSlug() + '_all.csv', scope); return; }
      var xName = (Store.plot(plotIdx).x || {}).name || 'x';
      doCsvExport(plotIdx, names, xName, scope, panelsOfPlot[0], runSlug() + '_all.csv');
    }

    // ---------------------------------------------------------------- PNG

    function bgColor() {
      return getComputedStyle(root).getPropertyValue('--bg-1').trim() || '#14181d';
    }
    function textColor() {
      return getComputedStyle(root).getPropertyValue('--text-0').trim() || '#dfe4ea';
    }

    function panelTitleText(panel) {
      var plot = Store.plot(panel.plotIdx);
      return (plot ? plot.name : '') + (panel.signals.length ? ': ' + panel.signals.join(', ') : ' (empty)');
    }

    // A page-matching background, a title bar (panel name + signal list),
    // uPlot's own canvas (traces + axes + our cursor/marker/annotation
    // overlays are already baked into it, drawn in the same `draw` hook),
    // and a redraw of the A/B/marker *labels* -- the one piece that is DOM
    // (the badge tabs), not canvas, and so would otherwise be missing.
    function opPlotNow(plotIdx) { var pl = Store.plot(plotIdx); return !!pl && pl.kind === 'op'; }

    // Operating point: strips, reference levels and labels painted straight
    // from the strip model (there is no uPlot canvas to copy).
    function compositeOpCanvas(panel) {
      var dpr = window.devicePixelRatio || 1, pad = Math.round(8 * dpr), titleH = Math.round(22 * dpr);
      var width = Math.round((PanelManager.opModelWidth(panel) + 16) * dpr); // the model clamps narrow panels to 520 px
      var scratch = document.createElement('canvas').getContext('2d');
      var body = PanelManager.paintStrips(scratch, panel, 0, 0, dpr);
      var out = document.createElement('canvas');
      out.width = width; out.height = body + pad * 2 + titleH;
      var ctx = out.getContext('2d');
      ctx.fillStyle = bgColor(); ctx.fillRect(0, 0, out.width, out.height);
      ctx.fillStyle = textColor();
      ctx.font = FontScale.css((FontScale.basePx() - 1) * dpr, true);
      ctx.textBaseline = 'top';
      ctx.fillText(panelTitleText(panel), pad, Math.round(pad * 0.5));
      PanelManager.paintStrips(ctx, panel, pad, pad + titleH, dpr);
      return out;
    }

    function compositePanelCanvas(panel) {
      if (PanelManager.panelIsOp(panel)) return panel.signals.length ? compositeOpCanvas(panel) : null;
      var u = panel.uplot;
      if (!u) return null;
      var src = u.ctx.canvas;
      var dpr = window.devicePixelRatio || 1;
      var pad = Math.round(8 * dpr);
      var titleH = Math.round(22 * dpr);
      var w = src.width + pad * 2;
      var h = src.height + pad * 2 + titleH;
      var out = document.createElement('canvas');
      out.width = w; out.height = h;
      var ctx = out.getContext('2d');
      ctx.fillStyle = bgColor();
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = textColor();
      // item 6: was `'bold ' + Npx + 'px ' + AxisModes.AXIS_FONT`, but
      // AxisModes.AXIS_FONT (now .axisFont()) is itself a FULL font string
      // that already carries its own size ("12px ...") -- concatenating a
      // second size in front produced an invalid `ctx.font` value (canvas
      // silently ignores it and keeps the context's prior/default font), so
      // this title was almost certainly never actually rendering at the
      // intended size/family. FontScale.css() takes an explicit px number.
      ctx.font = FontScale.css((FontScale.basePx() - 1) * dpr, true);
      ctx.textBaseline = 'top';
      ctx.fillText(panelTitleText(panel), pad, Math.round(pad * 0.5));
      ctx.drawImage(src, pad, pad + titleH);
      drawCursorAndMarkerLabels(ctx, panel, pad, pad + titleH, dpr);
      return out;
    }

    function drawCursorAndMarkerLabels(ctx, panel, ox, oy, dpr) {
      var u = panel.uplot;
      var c = CursorEngine.get(panel.plotIdx);
      ['A', 'B'].forEach(function (which) {
        var xv = c[which];
        if (xv == null) return;
        var xpix = u.valToPos(xv, 'x', true);
        if (xpix < u.bbox.left - 1 || xpix > u.bbox.left + u.bbox.width + 1) return;
        drawTag(ctx, which, which === 'A' ? theme('--cursor-a', '#5b9bd5') : theme('--cursor-b', '#e08a3c'),
          ox + xpix, oy + u.bbox.top, dpr, 'above');
      });
      NamedMarkers.list(panel.plotIdx).forEach(function (m) {
        var xpix = u.valToPos(m.x, 'x', true);
        if (xpix < u.bbox.left - 1 || xpix > u.bbox.left + u.bbox.width + 1) return;
        drawTag(ctx, m.name, m.color, ox + xpix, oy + u.bbox.top + u.bbox.height, dpr, 'below');
      });
    }

    function drawTag(ctx, text, color, cx, cy, dpr, side) {
      ctx.save();
      ctx.font = FontScale.css((FontScale.basePx() - 3) * dpr, true); // same double-size bug as above, fixed
      var tw = ctx.measureText(text).width;
      var boxW = tw + 6 * dpr, boxH = 12 * dpr;
      var top = side === 'above' ? cy - boxH : cy;
      ctx.fillStyle = color;
      ctx.fillRect(cx - boxW / 2, top, boxW, boxH);
      ctx.fillStyle = theme('--tag-text', '#08151a');
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, cx, top + boxH / 2);
      ctx.restore();
    }

    // Stacked composite respecting the current column layout: same
    // row-major order as the on-screen CSS grid (#panels / #panels.cols-2).
    function compositeAllCanvas(plotIdx) {
      var panelsOfPlot = PanelManager.panelsForPlot(plotIdx);
      var cells = panelsOfPlot.map(compositePanelCanvas).filter(Boolean);
      if (!cells.length) return null;
      var dpr = window.devicePixelRatio || 1;
      var cols = PanelManager.getCols();
      var cw = 0, ch = 0;
      cells.forEach(function (c) { cw = Math.max(cw, c.width); ch = Math.max(ch, c.height); });
      var rows = Math.ceil(cells.length / cols);
      var gap = Math.round(8 * dpr);
      var headerH = Math.round(28 * dpr);
      var totalW = cols * cw + (cols + 1) * gap;
      var totalH = headerH + rows * ch + (rows + 1) * gap;
      var out = document.createElement('canvas');
      out.width = totalW; out.height = totalH;
      var ctx = out.getContext('2d');
      ctx.fillStyle = bgColor();
      ctx.fillRect(0, 0, totalW, totalH);
      ctx.fillStyle = textColor();
      ctx.font = FontScale.css((FontScale.basePx() + 2) * dpr, true); // same double-size bug as above, fixed
      ctx.textBaseline = 'top';
      ctx.fillText((PAYLOAD.run.title || 'silk waveform explorer'), gap, Math.round(gap * 0.4));
      for (var i = 0; i < cells.length; i++) {
        var r = Math.floor(i / cols), c = i % cols;
        ctx.drawImage(cells[i], gap + c * (cw + gap), headerH + gap + r * (ch + gap));
      }
      return out;
    }

    // Feature-detected clipboard-image copy (recommendation #3's honest
    // half): `navigator.clipboard.write` with a ClipboardItem needs a secure
    // context and, in practice, is blocked or absent under `file://`. On any
    // failure -- missing API or a rejected promise -- fall back to a normal
    // download and tell the user why, rather than silently doing nothing.
    function tryClipboard(canvas, filename) {
      if (!(navigator.clipboard && window.ClipboardItem)) {
        canvas.toBlob(function (blob) {
          downloadBlob(blob, filename);
          showToast('clipboard API unavailable here (needs a secure context) — downloaded ' + filename + ' instead');
        }, 'image/png');
        return;
      }
      canvas.toBlob(function (blob) {
        if (destroyed) return;
        navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]).then(function () {
          showToast('image copied to clipboard');
        }).catch(function (err) {
          downloadBlob(blob, filename);
          showToast('clipboard blocked (' + (err && err.message ? err.message : 'denied') +
            ') — downloaded ' + filename + ' instead');
        });
      }, 'image/png');
    }

    function exportPanelPng(panel, mode) {
      var canvas = compositePanelCanvas(panel);
      if (!canvas) { showToast('nothing to export'); return; }
      var idx = PanelManager.panelsForPlot(panel.plotIdx).indexOf(panel);
      var filename = runSlug() + '_panel' + (idx + 1) + '.png';
      if (mode === 'copy') { tryClipboard(canvas, filename); return; }
      canvas.toBlob(function (blob) { downloadBlob(blob, filename); showToast('exported ' + filename); }, 'image/png');
    }

    function exportAllPng(plotIdx, mode) {
      var canvas = compositeAllCanvas(plotIdx);
      if (!canvas) { showToast('no panels to export'); return; }
      var filename = runSlug() + '_all.png';
      if (mode === 'copy') { tryClipboard(canvas, filename); return; }
      canvas.toBlob(function (blob) { downloadBlob(blob, filename); showToast('exported ' + filename); }, 'image/png');
    }

    return {
      exportPanelCsv: exportPanelCsv, exportAllCsv: exportAllCsv,
      exportPanelPng: exportPanelPng, exportAllPng: exportAllPng,
      downloadBlob: downloadBlob, tag: drawTag,
    };
  })();

  // ======================================================================
  // Context -- run provenance block (view-context-and-scopes design §2).
  // `payload.context = {title, subtitle, fields:[{label,value,severity}]}`
  // is an OPTIONAL top-level payload key, absent entirely (not null, not
  // an empty object) when the emitter had none -- `current()` below must
  // not assume the key exists. `title` itself is not rendered here: Python
  // already folds `context.title` into `payload.run.title` with the
  // documented precedence (design §2.1), and `#run-title` is set from
  // `payload.run.title` the same way it always has been (App.init/
  // applyReload). This module only adds what `run.title` cannot carry:
  // the dimmer subtitle beside it, and the "i" popover of fields.
  //
  // Security (§6 R1): context strings come from a rawfile title or user
  // code and this page runs with file:// privileges -- every string here
  // goes through `setText`/`textContent`, never `innerHTML`.
  // ======================================================================

  var Context = (function () {
    var popoverEl = null;

    function current() { return (PAYLOAD && PAYLOAD.context) || null; }

    function ensurePopoverEl() {
      if (!popoverEl) {
        // Root-appended + `.anchored-menu` (position:absolute, JS-positioned
        // at open time): `#toolbar` is `overflow-y:hidden`, so an in-place
        // `.dropdown-menu` here would be invisible -- the same trap the
        // per-series style popover already routes around for `.panel`.
        popoverEl = ce('div', 'dropdown-menu anchored-menu context-popover');
        popoverEl.hidden = true;
        root.appendChild(popoverEl);
      }
      return popoverEl;
    }

    function fillPopover(menu, ctx) {
      if (!ctx.fields.length) {
        var empty = ce('div', 'dropdown-heading');
        setText(empty, 'no fields');
        menu.appendChild(empty);
        return;
      }
      var table = ce('table');
      for (var i = 0; i < ctx.fields.length; i++) {
        var f = ctx.fields[i];
        var tr = ce('tr');
        var tdLabel = ce('td', 'context-label');
        setText(tdLabel, f.label);
        var tdVal = ce('td', 'context-value sev-' + (f.severity || 'info'));
        setText(tdVal, f.value);
        tr.appendChild(tdLabel);
        tr.appendChild(tdVal);
        table.appendChild(tr);
      }
      menu.appendChild(table);
    }

    function toggle() {
      var ctx = current();
      if (!ctx) return; // no context -> the button is hidden and this is unreachable via click, but the `i` key has no such guard
      var m = ensurePopoverEl();
      var wasOpen = !m.hidden;
      closeAllDropdowns();
      if (wasOpen) return; // second toggle = close (closeAllDropdowns already hid it)
      clear(m);
      fillPopover(m, ctx);
      m.hidden = false;
      // 'left': hangs from the button's left edge, clamped back inside the
      // viewport by its own rendered width -- see positionAnchoredMenu's
      // comment (found while verifying at 1280px with a long title).
      positionAnchoredMenu(m, el('context-info-btn'), 'left');
    }

    // Reflects the current payload's context into the toolbar: subtitle
    // text, and the "i" button's very presence (§2.3: no context -> no
    // button at all). Called at boot and after every reload/scope switch,
    // since each scope's data.js carries its own (possibly absent) copy.
    function render() {
      var ctx = current();
      var sub = el('run-subtitle');
      var btn = el('context-info-btn');
      if (ctx && ctx.subtitle) {
        setText(sub, ctx.subtitle);
        sub.hidden = false;
      } else {
        setText(sub, '');
        sub.hidden = true;
      }
      if (ctx) {
        btn.hidden = false;
      } else {
        btn.hidden = true;
        if (popoverEl) popoverEl.hidden = true;
      }
    }

    function install() {
      el('context-info-btn').addEventListener('click', function (e) {
        e.stopPropagation();
        toggle();
      });
    }

    return { render: render, install: install, toggle: toggle, getContext: current };
  })();

  // ======================================================================
  // ViewState -- the v1.3 view-state sidecar, browser half (design doc
  // section 3.4-3.5). One canonical "view" shape feeds getState(), the
  // sidecar file, and StateStore's own persisted object (§9 R7) -- see
  // snapshot()/mutateStateOnly() below, which are the two directions of
  // that one shape.
  //
  // `appliedViewState` (the sidecar-precedence hash bookkeeping field, §3.4)
  // is deliberately NOT part of the canonical view shape returned by
  // snapshot()/getState(): it records "which embedded sidecar content has
  // already been applied this session", which is boot bookkeeping about
  // *how* the state arrived, not view content. Folding it into the shared
  // shape would make the §7.2 round-trip test self-referential (the hash of
  // an object that contains its own prior hash) for no benefit -- a
  // deliberate, narrow deviation from a fully literal reading of "view is
  // byte-for-byte the object stored in localStorage" (§3.2); it still lives
  // on the *same* StateStore object, just outside the shape every consumer
  // treats as "the view".
  // ======================================================================

  var ViewState = (function () {
    function deriveSidecarFilename() {
      return options.viewFilename || 'view.silkview.json';
    }

    function sortKeysDeep(x) {
      if (Array.isArray(x)) return x.map(sortKeysDeep);
      if (x && typeof x === 'object') {
        var out = {};
        Object.keys(x).sort().forEach(function (k) { out[k] = sortKeysDeep(x[k]); });
        return out;
      }
      return x;
    }

    function prettyJson(obj) { return JSON.stringify(sortKeysDeep(obj), null, 2); }

    // The one serializer (§5 item 4, §9 R7): the canonical view object,
    // reflecting the *live* UI -- flushes the active plot's panel layout
    // synchronously first, since the normal path is a 400ms-debounced save.
    function snapshot() {
      PanelManager.persistCurrentLayout();
      var st = StateStore.get() || StateStore.defaults();
      var view = {
        v: 1,
        activePlot: st.activePlot,
        filter: st.filter || '',
        cols: st.cols || 1,
        treeMode: st.treeMode !== false,
        fontScale: FontScale.isValid(st.fontScale) ? st.fontScale : 'normal', // additive, item 6
        plots: st.plots || {},
      };
      return JSON.parse(JSON.stringify(view)); // plain-JSON deep clone
    }

    function buildEnvelope() {
      var view = snapshot();
      var run = (PAYLOAD && PAYLOAD.run) || {};
      return {
        silkview: 1,
        generator: 'silk.signals.view',
        created: new Date().toISOString(),
        run: { state_key: run.state_key || '', source: run.source || '', title: run.title || '' },
        view: view,
      };
    }

    // ---- the two ways an incoming view object is turned into `state` ----

    function mutateStateOnly(block, hash) {
      var merged = StateStore.defaults();
      merged.activePlot = (block.activePlot != null) ? block.activePlot : null;
      merged.filter = typeof block.filter === 'string' ? block.filter : '';
      merged.cols = (block.cols === 2) ? 2 : 1;
      merged.treeMode = block.treeMode !== false;
      merged.fontScale = FontScale.isValid(block.fontScale) ? block.fontScale : 'normal'; // additive, item 6
      // Restoration prunes definitions; keep caller-owned view state intact.
      merged.plots = (block.plots && typeof block.plots === 'object') ? JSON.parse(JSON.stringify(block.plots)) : {};
      if (hash != null) merged.appliedViewState = hash;
      StateStore.replace(merged);
      StateStore.saveNow();
    }

    // Pushes the current StateStore state into the live DOM -- used after
    // any post-boot state replacement (undo link, Load view/drag-drop,
    // loadViewState command, Reset view). Not used at boot: at boot, the
    // normal App.init sequence performs this same push once, after Picker
    // and PanelManager exist.
    function rerenderFromState() {
      var st = StateStore.get();
      if (!PAYLOAD || !st) return;
      PanelManager.teardown();
      Store.init(PAYLOAD);
      Store.restore_derived(st.plots);
      var idx = (st.activePlot != null && PAYLOAD.plots[st.activePlot]) ? st.activePlot : PAYLOAD.default_plot;
      el('plot-select').value = String(idx);
      Picker.setTreeModeQuiet(st.treeMode !== false);
      FontScale.set(FontScale.isValid(st.fontScale) ? st.fontScale : 'normal');
      var fsSel = el('font-scale-select');
      if (fsSel) fsSel.value = FontScale.get();
      PanelManager.setCols(st.cols || 1);
      PanelManager.loadPlot(idx);
      Picker.setFilter(st.filter || '');
    }

    // Boot-time sidecar precedence (§3.4), normative algorithm. `emb` is
    // options.viewState -- already the bare "view" object, not an
    // envelope. Mutates StateStore only; the caller renders once, normally,
    // right after this returns (no DOM exists to push into yet at the point
    // App.init calls this).
    function applyEmbeddedAtBoot(emb) {
      if (!emb) return;
      var st = StateStore.get() || StateStore.load(StateStore.getKey());
      var h = djb2Hash8(JSON.stringify(emb));
      if (st.appliedViewState === h) return; // already applied this exact content this session
      var priorSnapshot = JSON.parse(JSON.stringify(st));
      mutateStateOnly(emb, h);
      var fname = deriveSidecarFilename();
      showToastAction('view loaded from ' + fname, 'use my saved view', function () {
        StateStore.replace(priorSnapshot);
        StateStore.get().appliedViewState = h; // keep the hash so it does not re-apply again this session
        StateStore.saveNow();
        rerenderFromState();
      });
    }

    // Accepts either the full §3.2 envelope or a bare `view` block (same
    // detection rule as the Python side, §3.6: presence of a `silkview`
    // key), for both the file-picker/drag-drop path and the loadViewState
    // command -- one validator, not two drifting copies (§9 R7's spirit
    // extended to this boundary too).
    function extractViewBlock(obj) {
      if (obj == null || typeof obj !== 'object' || Array.isArray(obj)) {
        throw new Error('view state is not an object');
      }
      if (Object.prototype.hasOwnProperty.call(obj, 'silkview')) {
        if (obj.silkview !== 1) throw new Error('unknown silkview version ' + obj.silkview + ' (understood: 1)');
        if (!obj.view || typeof obj.view !== 'object') throw new Error('"view" is missing or not an object');
        if (obj.view.v != null && obj.view.v !== 1) throw new Error('unknown view schema v=' + obj.view.v + ' (understood: 1)');
        return { block: obj.view, hashSrc: JSON.stringify(obj.view) };
      }
      if (obj.v != null && obj.v !== 1) throw new Error('unknown view schema v=' + obj.v + ' (understood: 1)');
      return { block: obj, hashSrc: JSON.stringify(obj) };
    }

    function applyExternal(obj) {
      var extracted;
      try { extracted = extractViewBlock(obj); }
      catch (e) { return { ok: false, error: e.message }; }
      var h = djb2Hash8(extracted.hashSrc);
      mutateStateOnly(extracted.block, h);
      rerenderFromState();
      return { ok: true };
    }

    function readAndApplyFile(file) {
      var reader = new window.FileReader();
      readers.add(reader);
      reader.onloadend = function () { readers.delete(reader); };
      reader.onload = function () {
        if (destroyed) return;
        var obj;
        try { obj = JSON.parse(reader.result); }
        catch (e) { showToast('invalid view file: not valid JSON'); return; }
        var r = applyExternal(obj);
        if (!r.ok) showToast('invalid view file: ' + r.error);
        else showToast('view loaded from ' + (file.name || 'file'));
      };
      reader.onerror = function () { showToast('could not read file'); };
      reader.readAsText(file);
    }

    // ---- Save / Copy / Reset ----

    function triggerDownload() {
      var text = prettyJson(buildEnvelope());
      var blob = new Blob([text], { type: 'application/json' });
      var fname = deriveSidecarFilename();
      Exporter.downloadBlob(blob, fname);
      showToast('saved ' + fname + ' — move it next to view.html to auto-load on next emit, or use Load view…');
    }

    function execFallbackCopy(text) {
      if (destroyed) return;
      try {
        var ta = ce('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.left = '-9999px';
        root.appendChild(ta);
        ta.focus(); ta.select();
        var ok = document.execCommand('copy');
        root.removeChild(ta);
        showToast(ok ? 'view JSON copied to clipboard' : 'clipboard unavailable — use Save view instead');
      } catch (e) {
        showToast('clipboard unavailable — use Save view instead');
      }
    }

    function copyJson() {
      var text = prettyJson(buildEnvelope());
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () {
          showToast('view JSON copied to clipboard');
        }).catch(function () { execFallbackCopy(text); });
      } else {
        execFallbackCopy(text);
      }
    }

    function resetToDefaults() {
      var key = StateStore.getKey();
      try { if (key) localStorage.removeItem(key); } catch (e) { /* ignore */ }
      StateStore.replace(StateStore.defaults());
      StateStore.saveNow();
      rerenderFromState();
      showToast('view reset to defaults');
    }

    // ---- toolbar / drag-drop wiring ----

    function fillViewMenu(menu) {
      menuButton(menu, 'Save view (.silkview.json)', 'Download the current view state', triggerDownload);
      menuButton(menu, 'Copy view JSON', 'Copy the view-state JSON to the clipboard', copyJson);
      menuButton(menu, 'Load view…', 'Load a .silkview.json file', function () { el('view-file-input').click(); });
      menuSep(menu);
      menuButton(menu, 'Reset view to defaults', 'Clear saved state for this run and rebuild the default layout', resetToDefaults);
    }

    function installViewMenu() {
      // Root-append + `.anchored-menu`: see the view.css comment on that
      // class (`#toolbar` is overflow-y:hidden, which clipped this menu
      // fully invisible in place -- found while adding the Scope menu
      // right next to it).
      var menu = el('view-menu');
      menu.classList.add('anchored-menu');
      root.appendChild(menu);
      el('view-btn').addEventListener('click', function (e) {
        e.stopPropagation();
        toggleDropdown(menu, fillViewMenu, el('view-btn'));
      });
      el('view-file-input').addEventListener('change', function (e) {
        var f = e.target.files && e.target.files[0];
        e.target.value = '';
        if (f) readAndApplyFile(f);
      });
    }

    // §3.5: "the existing signal drag-drop onto panels uses text/plain,
    // files use dataTransfer.files -- no conflict, but the drop handler
    // must check files.length first." This listener sits on `document`
    // (bubble phase), so panel-specific drop handlers (deeper in the tree,
    // added directly to each panel's chart element) always run first; this
    // one only acts when the drop actually carries files, and is a no-op
    // (returns without calling preventDefault) for a plain internal
    // text/plain signal drag.
    function installDropZone() {
      listen(root, 'dragover', function (e) {
        var types = (e.dataTransfer && e.dataTransfer.types) || [];
        if (Array.prototype.indexOf.call(types, 'Files') !== -1) {
          e.preventDefault();
          root.classList.add('view-drop-active');
        }
      });
      listen(root, 'dragleave', function (e) {
        if (!e.relatedTarget || !root.contains(e.relatedTarget)) {
          root.classList.remove('view-drop-active');
        }
      });
      listen(root, 'drop', function (e) {
        root.classList.remove('view-drop-active');
        var files = e.dataTransfer && e.dataTransfer.files;
        if (!files || !files.length) return; // not a file drop; let panel handlers run
        e.preventDefault();
        if (/\.raw$/i.test(files[0].name)) LocalRawFile.open(files[0]);
        else readAndApplyFile(files[0]);
      });
    }

    return {
      snapshot: snapshot, buildEnvelope: buildEnvelope, prettyJson: prettyJson,
      triggerDownload: triggerDownload, copyJson: copyJson, resetToDefaults: resetToDefaults,
      applyExternal: applyExternal, applyEmbeddedAtBoot: applyEmbeddedAtBoot,
      rerenderFromState: rerenderFromState, installViewMenu: installViewMenu,
      installDropZone: installDropZone, deriveSidecarFilename: deriveSidecarFilename,
    };
  })();

  // ======================================================================
  // Commands -- the v1.3 command bus (design doc section 4). A registry of
  // named, JSON-argument actions over the methods the other modules already
  // expose (§4.1: "additive... nothing existing has to move"), reachable
  // from window.silkView.run(), ?commands=, and an emit-time list.
  // ======================================================================

  var Commands = (function () {
    function CmdError(message) { this.message = message; }
    CmdError.prototype = Object.create(Error.prototype);

    function requireArgs(a, names) {
      for (var i = 0; i < names.length; i++) {
        if (a[names[i]] === undefined) throw new CmdError('missing required arg "' + names[i] + '"');
      }
    }

    function parseNumOrSI(v) {
      if (typeof v === 'number') return isFiniteNum(v) ? v : null;
      if (typeof v === 'string') return Fmt.parseSI(v);
      return null;
    }

    function activePanelsArr() { return PanelManager.panelsForPlot(activePlotIdx); }

    function resolvePanel(argPanel) {
      var arr = activePanelsArr();
      var idx = argPanel;
      if (idx == null) {
        var focused = PanelManager.getFocused();
        idx = focused ? arr.indexOf(focused) : (arr.length ? 0 : -1);
      }
      if (idx == null || idx < 0 || idx >= arr.length) {
        throw new CmdError('panel ' + (argPanel == null ? '(none focused)' : argPanel) +
          ' does not exist (0..' + Math.max(arr.length - 1, 0) + ' open)');
      }
      return arr[idx];
    }

    function resolveSignalNames(sArg) {
      if (Array.isArray(sArg)) return sArg;
      if (typeof sArg === 'string') return [sArg];
      throw new CmdError('signals must be a string or an array of strings');
    }

    function allSignalNames() {
      var plot = Store.plot(activePlotIdx);
      return (plot && plot.signals) ? plot.signals.map(function (s) { return s.name; }) : [];
    }

    function checkSignalExists(name) {
      if (Store.findSignal(activePlotIdx, name)) return;
      var sug = nearMisses(name, allSignalNames(), 3);
      throw new CmdError('unknown signal "' + name + '"' + (sug.length ? ' -- did you mean: ' + sug.join(', ') + '?' : ''));
    }

    function findMarkerByName(name) {
      var arr = NamedMarkers.list(activePlotIdx);
      for (var i = 0; i < arr.length; i++) if (arr[i].name === name) return arr[i];
      var names = arr.map(function (m) { return m.name; });
      var sug = nearMisses(name, names, 3);
      throw new CmdError('unknown marker "' + name + '"' + (sug.length ? ' -- did you mean: ' + sug.join(', ') + '?' : ''));
    }

    function resolve_y_axis(panel, unit) {
      if (unit === undefined) return panel.yAxes[0];
      var axis = panel.yAxes.find(function (group) { return group.unit === unit; });
      if (!axis) throw new CmdError('no Y axis for unit "' + unit + '" in this panel');
      return axis;
    }

    // ---- the v1.3 command set (design doc section 4.3, normative) -------

    var REGISTRY = [
      { name: 'setPlot', args: { index: 'int' }, help: 'switch to plot <index>, rebuilding its panels', run: function (a) {
        requireArgs(a, ['index']);
        if (!PAYLOAD || !PAYLOAD.plots[a.index]) {
          throw new CmdError('plot ' + a.index + ' does not exist (0..' + ((PAYLOAD ? PAYLOAD.plots.length : 1) - 1) + ')');
        }
        el('plot-select').value = String(a.index);
        PanelManager.loadPlot(a.index);
      } },
      { name: 'setFilter', args: { text: 'string' }, help: 'set the picker filter text ("" clears)', run: function (a) {
        requireArgs(a, ['text']);
        Picker.setFilter(String(a.text));
      } },
      { name: 'setTreeMode', args: { on: 'bool' }, help: 'switch the picker between tree and flat', run: function (a) {
        requireArgs(a, ['on']);
        Picker.setTreeMode(!!a.on);
      } },
      { name: 'expand', args: { path: 'string' }, help: 'expand a group by its dot-joined path key', run: function (a) {
        requireArgs(a, ['path']);
        var r = Picker.expand(a.path);
        if (!r.ok) throw new CmdError(r.error);
      } },
      { name: 'collapse', args: { path: 'string' }, help: 'collapse a group by its dot-joined path key', run: function (a) {
        requireArgs(a, ['path']);
        var r = Picker.collapse(a.path);
        if (!r.ok) throw new CmdError(r.error);
      } },
      { name: 'expandAll', args: {}, help: 'expand every group in the active plot', run: function () { Picker.expandAll(); } },
      { name: 'collapseAll', args: {}, help: 'collapse every group in the active plot', run: function () { Picker.collapseAll(); } },
      { name: 'deriveSignal', args: { operation: 'string', operands: 'array', params: 'object', plot: 'int', name: 'string', replace: 'boolean' },
        help: 'create a derived trace saved with the view in one plot (default: active): sum|difference|product|quotient|mean ' +
          'of 2+ traces, or scale (params {gain, offset})|abs|derivative|integral of one; AC vector or part operands use complex values; ' +
          'optional name labels the result; replace:true updates an existing explicitly named definition; ' +
          'returns {name, unit, created}, with name (mag) and name (phase) traces for complex results',
        run: function (a) {
          requireArgs(a, ['operation', 'operands']);
          var plotIdx = a.plot === undefined ? activePlotIdx : a.plot;
          var before = a.name !== undefined && Store.plot(plotIdx) ? Store.plot(plotIdx).signals.filter(function (sig) {
            return sig.derived && sig.part !== 'phase';
          }) : [];
          var result = Store.derive_signal(plotIdx, a.operation, a.operands, a.params, a.name, a.replace);
          if (result.created) {
            var changes = [];
            before.forEach(function (sig) {
              var next = Store.operand_signal(plotIdx, sig.group || sig.name);
              if (next && next !== sig) changes.push({
                before: sig.group ? [sig.group + ' (mag)', sig.group + ' (phase)'] : [sig.name],
                after: next.group ? [next.group + ' (mag)', next.group + ' (phase)'] : [next.name]
              });
            });
            if (changes.length) PanelManager.replaceDerived(plotIdx, changes);
            PanelManager.scheduleStateSave();
          }
          if (plotIdx === activePlotIdx && result.created) {
            Picker.rebuild(); PanelManager.refreshValues();
          }
          return result;
        }
      },
      { name: 'addPanel', args: { signals: 'array', yMode: 'string' }, help: 'add a new panel; returns {panel: index}', run: function (a) {
        var sigs = a.signals ? resolveSignalNames(a.signals) : [];
        sigs.forEach(checkSignalExists);
        var panel = PanelManager.addPanel(sigs, a.yMode);
        return { panel: activePanelsArr().indexOf(panel) };
      } },
      { name: 'removePanel', args: { panel: 'int' }, help: 'remove a panel (default: focused)', run: function (a) {
        PanelManager.removePanel(resolvePanel(a.panel));
      } },
      { name: 'clearPanel', args: { panel: 'int' }, help: 'remove every signal from a panel (default: focused)', run: function (a) {
        PanelManager.clearPanel(resolvePanel(a.panel));
      } },
      { name: 'addSignal', args: { signals: 'string|array', panel: 'int' }, help: 'add signal(s) to a panel (default: focused)', run: function (a) {
        requireArgs(a, ['signals']);
        var names = resolveSignalNames(a.signals);
        names.forEach(checkSignalExists);
        var p = resolvePanel(a.panel);
        names.forEach(function (n) { PanelManager.addSignal(p, n); });
      } },
      { name: 'removeSignal', args: { signal: 'string', panel: 'int' }, help: 'remove a signal from a panel (default: focused)', run: function (a) {
        requireArgs(a, ['signal']);
        PanelManager.removeSignal(resolvePanel(a.panel), a.signal);
      } },
      { name: 'setYMode', args: { mode: 'string', panel: 'int', unit: 'string' }, help: 'linear|log|dB for a unit group (default: first axis); dB requires every signal on that axis to be part=="mag"', run: function (a) {
        requireArgs(a, ['mode']);
        if (['linear', 'log', 'dB'].indexOf(a.mode) === -1) throw new CmdError('mode must be linear|log|dB, got "' + a.mode + '"');
        var p = resolvePanel(a.panel);
        var axis = resolve_y_axis(p, a.unit);
        if (a.mode === 'dB' && !AxisModes.allMag(axis)) throw new CmdError('dB is only available when every signal on the axis has part=="mag"');
        PanelManager.setPanelYMode(p, a.mode, axis);
      } },
      { name: 'setXLog', args: { on: 'bool' }, help: 'toggle the shared X axis linear/log', run: function (a) {
        requireArgs(a, ['on']);
        PanelManager.setXLog(!!a.on);
      } },
      { name: 'setXRange', args: { min: 'number|string', max: 'number|string' }, help: 'numbers or SI-suffixed strings', run: function (a) {
        requireArgs(a, ['min', 'max']);
        var lo = parseNumOrSI(a.min), hi = parseNumOrSI(a.max);
        if (lo == null || hi == null) throw new CmdError('min/max must be numbers or SI-suffixed strings');
        PanelManager.setXRange(lo, hi);
      } },
      { name: 'resetZoom', args: {}, help: 'reset X zoom to the full domain', run: function () { PanelManager.resetZoom(); } },
      { name: 'setCols', args: { n: 'int' }, help: '1 or 2 panel columns', run: function (a) {
        requireArgs(a, ['n']);
        if (a.n !== 1 && a.n !== 2) throw new CmdError('n must be 1 or 2, got ' + a.n);
        PanelManager.setCols(a.n);
      } },
      // item 2
      { name: 'setYRange', args: { panel: 'int', min: 'number|string', max: 'number|string', unit: 'string' }, help: 'set a fixed Y range in original units (default: first axis); reset via autoscaleY', run: function (a) {
        requireArgs(a, ['min', 'max']);
        var lo = parseNumOrSI(a.min), hi = parseNumOrSI(a.max);
        if (lo == null || hi == null) throw new CmdError('min/max must be numbers or SI-suffixed strings');
        if (!(hi > lo)) throw new CmdError('min must be less than max');
        var p = resolvePanel(a.panel), axis = resolve_y_axis(p, a.unit);
        if (axis.yMode !== 'linear' && lo <= 0) throw new CmdError('a log/dB axis needs a positive Y range');
        PanelManager.setYRange(p, lo, hi, axis);
      } },
      { name: 'autoscaleY', args: { panel: 'int', unit: 'string' }, help: 'autoscale a unit group (default: all axes in the panel)', run: function (a) {
        var p = resolvePanel(a.panel);
        PanelManager.autoscaleY(p, a.unit === undefined ? null : resolve_y_axis(p, a.unit));
      } },
      // item 3
      { name: 'setSeriesStyle', args: { panel: 'int', signal: 'string', color: 'string', width: 'number', dash: 'string', points: 'bool', interp: 'string' }, help: 'style overrides for one series; omitted args leave that property unchanged', run: function (a) {
        requireArgs(a, ['signal']);
        var p = resolvePanel(a.panel);
        if (p.signals.indexOf(a.signal) === -1) {
          throw new CmdError('signal "' + a.signal + '" is not plotted in panel ' + activePanelsArr().indexOf(p));
        }
        var patch = {};
        if (a.color !== undefined) patch.color = String(a.color);
        if (a.width !== undefined) {
          if (typeof a.width !== 'number' || !isFiniteNum(a.width) || a.width <= 0) throw new CmdError('width must be a positive number');
          patch.width = a.width;
        }
        if (a.dash !== undefined) {
          if (['solid', 'dashed', 'dotted'].indexOf(a.dash) === -1) throw new CmdError('dash must be solid|dashed|dotted, got "' + a.dash + '"');
          patch.dash = a.dash;
        }
        if (a.points !== undefined) patch.points = !!a.points;
        if (a.interp !== undefined) {
          if (['linear', 'stepped', 'spline'].indexOf(a.interp) === -1) throw new CmdError('interp must be linear|stepped|spline, got "' + a.interp + '"');
          patch.interp = a.interp;
        }
        PanelManager.setSeriesStyle(p, a.signal, patch);
      } },
      // item 6
      { name: 'setFontScale', args: { scale: 'string' }, help: 'compact|normal|large -- text size for the whole page, including axis/canvas text', run: function (a) {
        requireArgs(a, ['scale']);
        if (!FontScale.isValid(a.scale)) throw new CmdError('scale must be compact|normal|large, got "' + a.scale + '"');
        applyFontScale(a.scale);
      } },
      { name: 'movePanel', args: { from: 'int', to: 'int' }, help: 'the reorder that drag performs', run: function (a) {
        requireArgs(a, ['from', 'to']);
        var arr = activePanelsArr();
        if (a.from < 0 || a.from >= arr.length) {
          throw new CmdError('panel ' + a.from + ' does not exist (0..' + Math.max(arr.length - 1, 0) + ' open)');
        }
        PanelManager.movePanel(a.from, a.to);
      } },
      { name: 'placeCursor', args: { which: 'string', x: 'number|string' }, help: 'A|B, snaps to sample', run: function (a) {
        requireArgs(a, ['which', 'x']);
        if (a.which !== 'A' && a.which !== 'B') throw new CmdError('which must be "A" or "B", got "' + a.which + '"');
        var xv = parseNumOrSI(a.x);
        if (xv == null) throw new CmdError('x must be a number or SI-suffixed string');
        CursorEngine.set(a.which, activePlotIdx, xv);
      } },
      { name: 'clearCursors', args: {}, help: 'clear both A and B cursors', run: function () {
        CursorEngine.clearAll(activePlotIdx);
        PanelManager.scheduleStateSave();
      } },
      { name: 'addMarker', args: { x: 'number|string', name: 'string', unit: 'string' }, help: 'returns {name}; on an operating point x is the level and unit (the strip unit) is required', run: function (a) {
        requireArgs(a, ['x']);
        var xv = parseNumOrSI(a.x);
        if (xv == null) throw new CmdError('x must be a number or SI-suffixed string');
        var opNow = PanelManager.opActive();
        if (opNow && typeof a.unit !== 'string') throw new CmdError('unit is required on an operating point (the unit of the strip, e.g. "V")');
        if (!opNow && a.unit !== undefined) throw new CmdError('unit applies only to an operating point');
        var m = NamedMarkers.add(activePlotIdx, xv, opNow ? a.unit : undefined);
        if (a.name) NamedMarkers.rename(activePlotIdx, m.id, String(a.name));
        return { name: a.name ? String(a.name) : m.name };
      } },
      { name: 'setMarkerReference', args: { name: 'string' }, help: 'operating point: make the named marker the reference of its unit', run: function (a) {
        requireArgs(a, ['name']);
        var m = findMarkerByName(a.name);
        if (!PanelManager.opActive() || typeof m.unit !== 'string') throw new CmdError('reference markers exist only on an operating point');
        NamedMarkers.setReference(activePlotIdx, m.id);
      } },
      { name: 'setMarkerX', args: { name: 'string', x: 'number|string', unit: 'string' }, help: 'on an operating point x is the level; unit, if given, must match the marker', run: function (a) {
        requireArgs(a, ['name', 'x']);
        var m = findMarkerByName(a.name);
        if (a.unit !== undefined && a.unit !== m.unit) throw new CmdError('unit "' + a.unit + '" does not match marker ' + a.name + (typeof m.unit === 'string' ? ' (unit "' + m.unit + '")' : ' (no unit)'));
        var xv = parseNumOrSI(a.x);
        if (xv == null) throw new CmdError('x must be a number or SI-suffixed string');
        NamedMarkers.setX(activePlotIdx, m.id, xv);
      } },
      { name: 'renameMarker', args: { name: 'string', to: 'string' }, help: '', run: function (a) {
        requireArgs(a, ['name', 'to']);
        var m = findMarkerByName(a.name);
        NamedMarkers.rename(activePlotIdx, m.id, String(a.to));
      } },
      { name: 'removeMarker', args: { name: 'string' }, help: '', run: function (a) {
        requireArgs(a, ['name']);
        var m = findMarkerByName(a.name);
        NamedMarkers.remove(activePlotIdx, m.id);
      } },
      { name: 'clearMarkers', args: {}, help: 'remove every named marker on the active plot', run: function () {
        NamedMarkers.clearAll(activePlotIdx);
      } },
      { name: 'exportCsv', args: { scope: 'string', panel: 'int' }, help: 'visible|cursor|full; panel omitted = all plotted signals', run: function (a) {
        requireArgs(a, ['scope']);
        if (['visible', 'cursor', 'full'].indexOf(a.scope) === -1) throw new CmdError('scope must be visible|cursor|full, got "' + a.scope + '"');
        if (a.scope === 'cursor' && PanelManager.opActive()) throw new CmdError('cursor export needs an x axis; use visible or full');
        if (a.panel == null) Exporter.exportAllCsv(activePlotIdx, a.scope);
        else Exporter.exportPanelCsv(resolvePanel(a.panel), a.scope);
      } },
      { name: 'exportPng', args: { scope: 'string', panel: 'int', mode: 'string' }, help: 'scope panel|all; mode download|copy', run: function (a) {
        requireArgs(a, ['scope']);
        var mode = a.mode === 'copy' ? 'copy' : 'download';
        if (a.scope === 'all') Exporter.exportAllPng(activePlotIdx, mode);
        else if (a.scope === 'panel') Exporter.exportPanelPng(resolvePanel(a.panel), mode);
        else throw new CmdError('scope must be panel|all, got "' + a.scope + '"');
      } },
      { name: 'saveViewState', args: {}, help: 'triggers the .silkview.json download', run: function () { ViewState.triggerDownload(); } },
      { name: 'loadViewState', args: { state: 'object' }, help: 'applies a §3.2 envelope or bare view block; no file picker', run: function (a) {
        requireArgs(a, ['state']);
        var r = ViewState.applyExternal(a.state);
        if (!r.ok) throw new CmdError(r.error);
      } },
      { name: 'getState', args: {}, help: 'returns the current view-state object (same shape as the sidecar\'s "view" block)', run: function () {
        return ViewState.snapshot();
      } },
      // view-context-and-scopes design §2.3/§3.3
      { name: 'getContext', args: {}, help: 'returns the current run-provenance context block ({title,subtitle,fields}), or null when this run has none', run: function () {
        return Context.getContext();
      } },
      { name: 'listScopes', args: {}, help: 'returns {current, scopes:[{key,label,html,data_src,stamp_src,n_signals}]}; scopes is [] when this page is not part of a set', run: function () {
        return { current: ScopeManager.currentKey(), scopes: ScopeManager.list() };
      } },
      { name: 'setScope', args: { key: 'string' }, help: 'switch to a sibling scope by key (async: resolves once the new payload is applied, rejects loudly if the sibling cannot be loaded)', run: function (a) {
        requireArgs(a, ['key']);
        return ScopeManager.switchTo(a.key).then(
          function (r) { return r; },
          function (e) { throw new CmdError((e && e.message) ? e.message : String(e)); }
        );
      } },
    ];

    var REGISTRY_BY_NAME = {};
    REGISTRY.forEach(function (r) { REGISTRY_BY_NAME[r.name] = r; });

    function commandsList() {
      return REGISTRY.map(function (r) { return { name: r.name, args: r.args, help: r.help }; });
    }

    function normalizeOne(c) {
      if (typeof c === 'string') return { cmd: c, args: {} };
      return c;
    }

    // §4.2, normative: never throws; always resolves; a batch stops at the
    // first failure (commands before it have already applied); a failure
    // also raises a toast.
    function run(cmdOrList) {
      return new Promise(function (resolve) {
        var list;
        try {
          list = (Array.isArray(cmdOrList) ? cmdOrList : [cmdOrList]).map(normalizeOne);
        } catch (e) {
          resolve({ ok: false, index: 0, cmd: null, error: 'malformed command list: ' + (e && e.message) });
          return;
        }
        var results = [];
        var i = 0;
        function step() {
          if (destroyed) { resolve({ ok: false, index: i, cmd: null, error: 'Explorer has been destroyed' }); return; }
          if (i >= list.length) { resolve({ ok: true, results: results }); return; }
          var entry = list[i];
          var cmdName = entry && entry.cmd;
          var args = (entry && entry.args) || {};
          var errMsg = null, value;
          if (!entry || typeof entry !== 'object' || typeof cmdName !== 'string') {
            errMsg = 'malformed command entry at index ' + i;
          } else {
            var def = REGISTRY_BY_NAME[cmdName];
            if (!def) {
              var sug = nearMisses(cmdName, Object.keys(REGISTRY_BY_NAME), 3);
              errMsg = 'unknown command "' + cmdName + '"' + (sug.length ? ' -- did you mean: ' + sug.join(', ') + '?' : '');
            } else {
              try { value = def.run(args || {}); }
              catch (e) { errMsg = (e && e.message) ? e.message : String(e); }
            }
          }
          if (errMsg != null) {
            showToast('command failed: ' + (cmdName || '?') + ' -- ' + errMsg);
            resolve({ ok: false, index: i, cmd: cmdName || null, error: errMsg });
            return;
          }
          // v1.4: a command's run() may return a thenable (currently only
          // setScope, design §3.3 -- "async: resolve after the new
          // payload is applied, fail loud if it cannot load"). Await it
          // in place so the batch still runs sequentially, one command at
          // a time, exactly like every synchronous command already does;
          // a rejection is reported the same way a synchronous throw is.
          if (value && typeof value.then === 'function') {
            value.then(function (resolved) {
              results.push(resolved === undefined ? null : resolved);
              i++;
              step();
            }, function (e) {
              var msg = (e && e.message) ? e.message : String(e);
              showToast('command failed: ' + cmdName + ' -- ' + msg);
              resolve({ ok: false, index: i, cmd: cmdName, error: msg });
            });
            return;
          }
          results.push(value === undefined ? null : value);
          i++;
          step();
        }
        step();
      });
    }

    return { run: run, commands: commandsList };
  })();



  // ======================================================================
  // Reloader -- design doc 7.2
  // ======================================================================

  var Reloader = (function () {
    var pollMs = 2000;
    var failCount = 0;
    var timer = null;
    var currentRunId = null;
    var stopped = false;

    function stop() {
      stopped = true;
      clearTimeout(timer);
      timer = null;
      clear(el('stamp-slot'));
    }

    // Re-entrant by design (view-context-and-scopes design §3.3): a scope
    // switch calls this again for the new scope's payload once it has
    // applied, so `stopped`/`failCount` reset here rather than only at
    // module init -- otherwise a scope that had tripped the 2-failure
    // "auto-reload unavailable" badge would leave every LATER scope
    // switched to unable to ever poll again.
    function start(payload) {
      if (!transport) { hideBadge(); return; }
      currentRunId = payload.run.id;
      pollMs = CONFIG.poll_ms || 0;
      stopped = false;
      failCount = 0;
      hideBadge();
      if (CONFIG.inline) {
        showBadge('inline file — auto-reload unavailable', false);
        return;
      }
      if (!pollMs) {
        showBadge('auto-reload disabled (poll_ms=0)', false);
        return;
      }
      scheduleNext();
    }

    function scheduleNext() {
      if (stopped) return;
      if (timer) clearTimeout(timer);
      timer = setTimeout(poll, pollMs);
    }

    function poll() {
      if (stopped) return;
      var settled = false;
      var to = setTimeout(function () {
        if (settled || stopped) return;
        settled = true;
        onMissed();
      }, pollMs + 3000);

      bridge.stamp = function (runId) {
        if (settled || stopped) return;
        settled = true;
        clearTimeout(to);
        failCount = 0;
        hideBadge();
        if (runId !== currentRunId) {
          loadNewData();
        } else {
          scheduleNext();
        }
      };

      transport.request(CONFIG.stamp_src, el('stamp-slot'), function () {
        if (settled || stopped) return;
        settled = true;
        clearTimeout(to);
        onMissed();
      });
    }

    function onMissed() {
      failCount++;
      if (failCount >= 2) {
        stopped = true;
        showBadge('auto-reload unavailable — press F5 after a re-run', true, true);
        return;
      }
      scheduleNext();
    }

    function loadNewData() {
      // dataGen guard (§6 R2): if a scope switch starts (and wins the
      // handshake) while this periodic reload's data.js is still
      // in-flight, this stale response must not clobber the newly
      // switched scope's payload.
      var myGen = ++dataGen;
      bridge.data = function (newPayload) {
        if (myGen !== dataGen) return; // superseded by a scope switch
        currentRunId = newPayload.run.id;
        App.applyReload(newPayload);
        showToast('reloaded — run ' + newPayload.run.id);
        scheduleNext();
      };
      transport.request(CONFIG.data_src, el('stamp-slot'), function () {
        if (myGen !== dataGen) return;
        onMissed();
      });
    }

    function showBadge(text, isWarn, withRetry) {
      var b = el('reload-badge');
      clear(b);
      b.appendChild(document.createTextNode(text));
      b.className = 'badge' + (isWarn ? ' badge-warn' : '');
      if (withRetry) {
        var retry = ce('span', 'badge-retry');
        setText(retry, 'retry');
        retry.addEventListener('click', function () {
          stopped = false;
          failCount = 0;
          hideBadge();
          scheduleNext();
        });
        b.appendChild(retry);
      }
      b.hidden = false;
    }

    function hideBadge() {
      if (!CONFIG.inline && pollMs) el('reload-badge').hidden = true;
    }

    return { start: start, stop: stop, showBadge: showBadge };
  })();

  // ======================================================================
  // ScopeManager -- scope switching (view-context-and-scopes design §3).
  // `options.scopes` is ALWAYS present, shape
  // `{current: key|null, scopes: [{key,label,html,data_src,stamp_src,
  // n_signals}, ...]}`; `scopes` is `[]` when this page is not part of a
  // set. A set of exactly one page still lists that one entry -- the
  // manifest is read once here and never mutated; only the local
  // `currentKey` tracking variable changes as the user switches.
  //
  // Selecting a scope injects a fresh <script> element whose `src` is
  // the sibling scope's data_src (cache-busted with a `?_=<ts>` query) --
  // the SAME JSONP handshake (`window.__silkViewData`) the Reloader's own
  // periodic reload uses against the page's OWN data_src -- so the
  // rebuild is just `App.applyReload()`. That reuse is also what gives
  // "one layout across scopes, prune-and-keep" for free (design §3.3):
  // A scope reload keeps the StateStore key (only boot and local file opens
  // choose a new key), so the *same* saved layout is simply
  // re-pruned against the new payload's signal set by the existing v1
  // §7.6 path already inside `PanelManager.loadPlot()` -- no scope-aware
  // pruning code is needed here at all.
  // ======================================================================

  var ScopeManager = (function () {
    var RAW = (options.scopes && Array.isArray(options.scopes.scopes))
      ? options.scopes : { current: null, scopes: [] };
    var currentKey = RAW.current;

    function list() { return RAW.scopes.slice(); }

    function findByKey(key) {
      for (var i = 0; i < RAW.scopes.length; i++) {
        if (RAW.scopes[i].key === key) return RAW.scopes[i];
      }
      return null;
    }

    // Honest failure (design §3.3): the sibling's data.js failing to load
    // (a page emailed on its own, siblings absent) rejects with exactly
    // this message -- callers toast it and the control reverts to the
    // previous selection simply because `currentKey` is never mutated on
    // a failure path.
    function notNextToThisPage(key) {
      return new Error('scope "' + key + '" is not next to this page');
    }

    function switchTo(key) {
      return new Promise(function (resolve, reject) {
        var scope = findByKey(key);
        if (!scope) {
          reject(new Error('unknown scope "' + key + '" (known: ' + RAW.scopes.map(function (s) { return s.key; }).join(', ') + ')'));
          return;
        }
        var myGen = ++dataGen;
        pendingLoads.forEach(function (cancel) { cancel(); });
        Reloader.stop();
        if (key === currentKey) {
          Reloader.start(PAYLOAD);
          resolve({ key: key, changed: false });
          return;
        }
        var settled = false;
        var controller = new window.AbortController();
        function finish(error, payload) {
          if (settled) return;
          settled = true;
          clearTimeout(timeout);
          pendingLoads.delete(cancel);
          controller.abort();
          if (destroyed || myGen !== dataGen) {
            reject(new Error('scope switch to "' + key + '" superseded by a newer selection'));
          } else if (error) {
            Reloader.start(PAYLOAD);
            reject(error);
          } else {
            var previousPayload = PAYLOAD;
            try {
              // Reject malformed input before installing any of it, including
              // responses from the standalone JSONP transport.
              validatePayload(payload);
              App.applyReload(payload);
            } catch (applicationError) {
              // A host callback or chart peer can throw after Store changed.
              // Restore the prior dataset where possible and always settle.
              if (PAYLOAD !== previousPayload && previousPayload) {
                try { App.applyReload(previousPayload); }
                catch (restoreError) {
                  applicationError = new Error(applicationError.message +
                    '; restoring the previous waveform failed: ' + restoreError.message);
                }
              }
              Reloader.start(PAYLOAD);
              reject(applicationError);
              return;
            }
            currentKey = key;
            CONFIG.data_src = scope.data_src;
            CONFIG.stamp_src = scope.stamp_src;
            Reloader.start(payload);
            render();
            showToast('scope: ' + scope.label);
            resolve({ key: key, changed: true, n_signals: scope.n_signals });
          }
        }
        function cancel() { finish(new Error('scope load cancelled')); }
        pendingLoads.add(cancel);
        var timeout = setTimeout(function () { finish(notNextToThisPage(key)); }, Math.max(8000, (CONFIG.poll_ms || 0) + 5000));
        if (transport) {
          bridge.data = function (payload) { finish(null, payload); };
          transport.request(scope.data_src, el('stamp-slot'), function () { finish(notNextToThisPage(key)); });
        } else if (options.loadScope) {
          Promise.resolve().then(function () {
            return options.loadScope(scope, { signal: controller.signal });
          }).then(function (payload) {
            finish(null, payload);
          }).catch(function (error) { finish(error); });
        } else {
          finish(new Error('Provide loadScope(scope, {signal}) to load this scope in a host page.'));
        }
      });
    }

    // ---- toolbar wiring ----

    function currentScope() { return findByKey(currentKey); }

    function detach() {
      RAW = { current: null, scopes: [] };
      currentKey = null;
      el('scope-menu').hidden = true;
      render();
    }

    function fillMenu(menu) {
      menuHeading(menu, 'Scope');
      RAW.scopes.forEach(function (s) {
        var isCurrent = s.key === currentKey;
        var label = (isCurrent ? '✓ ' : '') + s.label + '  —  ' + s.n_signals + ' sig';
        menuButton(menu, label, isCurrent ? 'current scope' : ('switch to ' + s.label), function () {
          if (isCurrent) return;
          switchTo(s.key).catch(function (e) {
            // switchTo() itself never toasts on failure (so a setScope
            // command failure toasts exactly once, via Commands.run's own
            // generic "command failed: ..." wrapper) -- this direct
            // toolbar-click path owns its own single toast instead.
            showToast((e && e.message) ? e.message : String(e));
          });
        });
      });
    }

    // Button visibility/label. Design §3.3: the control is hidden only
    // when `scopes.length < 2` -- a set of exactly one page still lists
    // that one entry and must NOT be treated as "no set".
    function render() {
      var btn = el('scope-btn');
      if (!btn) return;
      if (RAW.scopes.length < 2) { btn.hidden = true; return; }
      btn.hidden = false;
      var cur = currentScope();
      setText(btn, 'Scope: ' + (cur ? cur.label : '?') + ' ▾');
    }

    function install() {
      var btn = el('scope-btn');
      if (!btn) return;
      var menu = el('scope-menu');
      menu.classList.add('anchored-menu');
      root.appendChild(menu);
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        toggleDropdown(menu, fillMenu, btn, 'left');
      });
    }

    return {
      list: list, currentKey: function () { return currentKey; },
      switchTo: switchTo, render: render, install: install, detach: detach,
    };
  })();

  var LocalRawFile = (function () {
    async function open(file) {
      if (destroyed) return;
      var my_gen = ++dataGen;
      pendingLoads.forEach(function (cancel) { cancel(); });
      Reloader.stop();
      ScopeManager.detach();
      // A sidecar already in flight can still invoke the global callback.
      bridge.data = function () {};
      var button = el('open-raw-btn');
      button.disabled = true;
      setText(button, 'Opening…');
      var retired = false;
      function retire() {
        if (retired) return;
        retired = true;
        pendingLoads.delete(retire);
        // File.arrayBuffer() cannot be aborted, but its control ownership can
        // be released immediately without waiting for obsolete bytes.
        button.disabled = false;
        setText(button, 'Open .raw…');
        if (my_gen === dataGen) bridge.data = function () {};
      }
      pendingLoads.add(retire);
      try {
        var core = options.waveformCore;
        if (!core) throw new Error('The waveform core is unavailable. Re-emit this page with a matching silk version.');
        var bytes = await file.arrayBuffer();
        if (my_gen !== dataGen) return;
        var planned = core.plan(core.read(bytes, { file_name: file.name }));
        // TASK-527/530: the explicit designer mode of the core labels each vector
        // with its bracket alias and attaches a proven address; the raw name
        // stays every identity. (A plain labels dict never yields an address.)
        var payload = await core.decimate(planned, core.to_designer_name ? { labels: 'designer' } : {});
        if (my_gen !== dataGen) return;
        bridge.data = function (incoming) {
          if (my_gen !== dataGen || incoming !== payload) return;
          el('fatal-banner').hidden = true;
          el('app').style.display = '';
          Reloader.showBadge('local file — reopen .raw to refresh', false);
          if (PAYLOAD && PAYLOAD.run.id === incoming.run.id && PAYLOAD.run.state_key === incoming.run.state_key) {
            showToast('unchanged — ' + file.name);
            return;
          }
          if (initialized) App.applyReload(incoming, true);
          else App.init(incoming);
          if (options.onPayload) options.onPayload(incoming);
          showToast('opened — ' + file.name);
        };
        bridge.data(payload);
      } catch (error) {
        if (my_gen === dataGen) fatal(error.name + ': ' + error.message);
      } finally {
        // Idempotence prevents stale completion from resetting a newer open.
        retire();
      }
    }

    function install() {
      el('open-raw-btn').addEventListener('click', function () { el('raw-file-input').click(); });
      el('raw-file-input').addEventListener('change', function (e) {
        var file = e.target.files && e.target.files[0];
        e.target.value = '';
        if (file) open(file);
      });
    }

    return { open: open, install: install };
  })();

  // ======================================================================
  // Keyboard shortcuts -- design doc 7.7
  // ======================================================================

  function installKeyboardShortcuts() {
    listen(root, 'keydown', function (e) {
      if (!root.contains(document.activeElement)) return;
      var tag = (e.target && e.target.tagName) || '';
      var typing = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (e.target && e.target.isContentEditable);
      if (e.key === '/' && !typing) {
        e.preventDefault();
        el('picker-filter').focus();
        return;
      }
      if (e.key === 'Escape') {
        if (typing && tag === 'INPUT') { Picker.setFilter(''); el('picker-filter').blur(); }
        el('shortcuts-help').hidden = true;
        closeAllDropdowns();
        return;
      }
      if (typing) return;
      if (e.key === '+' || e.key === '=') { PanelManager.zoomAbout(0.5); return; }
      if (e.key === '-' || e.key === '_') { PanelManager.zoomAbout(2); return; }
      // item 2: shift-0 resets Y autoscale on the hovered panel. Checked
      // BEFORE the plain '0' case and matched on `)` too -- on a US
      // keyboard layout, shift+0 reports e.key === ')', not '0' with
      // shiftKey true, so relying on `e.shiftKey && e.key === '0'` alone
      // would silently never fire there.
      if (e.key === ')' || (e.shiftKey && e.key === '0')) { PanelManager.autoscaleYFocused(); return; }
      if (e.key === '0') { PanelManager.resetZoom(); return; }
      // Fit everything: reset X to the full domain AND autoscale every
      // panel's Y, regardless of zoom or which panel the mouse is over. '0'
      // resets X only and shift-0 one panel's Y only; 'f' is the single-key
      // "fit the whole window" that neither did before.
      if (e.key === 'f') { PanelManager.fitAll(); return; }
      if (e.key === 'a') { PanelManager.placeCursorAtMouse('A'); return; }
      if (e.key === 'b') { PanelManager.placeCursorAtMouse('B'); return; }
      if (e.key === 's') { PanelManager.setSampleMarks(!PanelManager.sampleMarksOn()); return; }
      if (e.key === 'm') { NamedMarkers.addAtMouse(); return; }
      if (e.key === 'i') { Context.toggle(); return; }
    });
    listen(root, 'click', function () { closeAllDropdowns(); });
  }

  // ======================================================================
  // App -- top-level wiring
  // ======================================================================

  var App = (function () {
    function buildPlotSelect(payload) {
      var sel = el('plot-select');
      clear(sel);
      for (var i = 0; i < payload.plots.length; i++) {
        var p = payload.plots[i];
        var opt = ce('option');
        opt.value = String(i);
        setText(opt, i + ': ' + p.name + ' — ' + p.npoints + ' pts, ' + p.signals.length + ' sig');
        sel.appendChild(opt);
      }
      sel.addEventListener('change', function () {
        var idx = parseInt(sel.value, 10);
        PanelManager.loadPlot(idx);
        Picker.rebuild();
      });
    }

    function wireToolbar() {
      el('xlog-toggle').addEventListener('click', function () {
        PanelManager.setXLog(!PanelManager.isXLog());
        // "(all)" (item 4): this button scales every panel's shared X axis,
        // unlike each panel's own Y-mode select -- see also both controls'
        // `title` attributes.
        setText(el('xlog-toggle'), 'X: ' + (PanelManager.isXLog() ? 'log' : 'lin') + ' (all)');
      });
      el('samples-toggle').addEventListener('click', function () {
        PanelManager.setSampleMarks(!PanelManager.sampleMarksOn());
      });
      el('add-panel-btn').addEventListener('click', function () { PanelManager.addPanel([]); });
      el('layout-toggle').addEventListener('click', function () {
        PanelManager.setCols(PanelManager.getCols() === 2 ? 1 : 2);
      });
      el('font-scale-select').addEventListener('change', function () {
        applyFontScale(el('font-scale-select').value);
      });
      el('shortcuts-btn').addEventListener('click', function () {
        el('shortcuts-help').hidden = !el('shortcuts-help').hidden;
      });
      el('shortcuts-close').addEventListener('click', function () {
        el('shortcuts-help').hidden = true;
      });
      // Click the dimmed backdrop (the overlay itself, not the dialog inside
      // it) to dismiss -- Esc already worked, but clicking outside did not.
      el('shortcuts-help').addEventListener('click', function (e) {
        if (e.target === el('shortcuts-help')) el('shortcuts-help').hidden = true;
      });
      el('sidebar-toggle').addEventListener('click', function () {
        root.classList.toggle('sidebar-open');
      });


      el('markers-toggle').addEventListener('click', function (e) {
        e.stopPropagation();
        var panelEl = el('markers-panel');
        panelEl.hidden = !panelEl.hidden;
        if (!panelEl.hidden) NamedMarkers.renderTable(activePlotIdx);
      });
      el('add-marker-btn').addEventListener('click', function (e) {
        e.stopPropagation();
        NamedMarkers.addAtMouse();
        el('markers-panel').hidden = false;
      });
      el('markers-clear-all').addEventListener('click', function () {
        NamedMarkers.clearAll(activePlotIdx);
      });
      el('markers-panel-close').addEventListener('click', function () {
        el('markers-panel').hidden = true;
      });

      // Root-append + `.anchored-menu`: see the view.css comment on that
      // class -- an in-place `.dropdown-menu` here is invisible under
      // `#toolbar`'s `overflow-y: hidden`.
      var exportMenu = el('export-menu');
      exportMenu.classList.add('anchored-menu');
      root.appendChild(exportMenu);
      el('export-btn').addEventListener('click', function (e) {
        e.stopPropagation();
        toggleDropdown(exportMenu, fillAllExportMenu, el('export-btn'));
      });
    }

    function fillAllExportMenu(menu) {
      var plotIdx = activePlotIdx;
      var c = CursorEngine.get(plotIdx);
      var haveCursors = c.A != null && c.B != null;
      menuHeading(menu, 'CSV — all plotted signals');
      menuButton(menu, 'Visible range', 'Export the current zoom window', function () {
        Exporter.exportAllCsv(plotIdx, 'visible');
      });
      menuButton(menu, 'Cursor window (A–B)',
        PanelManager.opActive() ? 'An operating point has no x axis, so no cursors' : haveCursors ? 'Export between cursors A and B' : 'Place both A and B cursors first',
        function () { Exporter.exportAllCsv(plotIdx, 'cursor'); }, !haveCursors || PanelManager.opActive());
      menuButton(menu, 'Full range', 'Export every level-0 sample', function () {
        Exporter.exportAllCsv(plotIdx, 'full');
      });
      menuSep(menu);
      menuHeading(menu, 'PNG — all panels');
      menuButton(menu, 'Download PNG', 'Save a stacked composite of every open panel', function () {
        Exporter.exportAllPng(plotIdx, 'download');
      });
      menuButton(menu, 'Copy image', 'Copy the composite to the clipboard (falls back to download)', function () {
        Exporter.exportAllPng(plotIdx, 'copy');
      });
    }

    function init(payload) {
      if (!checkLittleEndian()) {
        fatal('This browser reports a big-endian platform; the waveform ' +
          'data is little-endian and cannot be decoded safely here.');
        return;
      }
      if (payload.format !== 1 && payload.format !== 2) {
        fatal('Unrecognized data format ' + payload.format + ' (this viewer ' +
          'understands formats 1 and 2). Re-emit with a matching silk version.');
        return;
      }
      PAYLOAD = payload;
      Store.init(payload);
      if (options.onTitle) options.onTitle(payload.run.title || 'silk waveform explorer');
      setText(el('run-title'), payload.run.title || '');
      Context.render();

      buildPlotSelect(payload);
      if (!initialized) {
      Picker.init();
      wireToolbar();
      ViewState.installViewMenu();
      ViewState.installDropZone();
      LocalRawFile.install();
      Context.install();
      ScopeManager.install();
      ScopeManager.render();
      installKeyboardShortcuts();
      initialized = true;
      }

      // Boot order (design doc v1.3 §3.4/§4.4), normative:
      //   defaults -> localStorage -> embedded view state -> SILK_VIEW_COMMANDS -> ?commands=
      var st = StateStore.load(stateKeyFor(payload));
      ViewState.applyEmbeddedAtBoot(options.viewState || null); // mutates StateStore only
      st = StateStore.get();
      Store.restore_derived(st.plots);

      var startPlot = (st.activePlot != null && payload.plots[st.activePlot]) ? st.activePlot : payload.default_plot;
      el('plot-select').value = String(startPlot);
      Picker.setTreeModeQuiet(st.treeMode !== false);
      // item 6: apply before any panel exists, so the very first buildUplot()
      // already measures/draws at the right size (no flash of the wrong size,
      // no wasted rebuild -- rebuildAllForFontChange() is a no-op here since
      // `panels` is still empty).
      FontScale.set(st.fontScale);
      el('font-scale-select').value = FontScale.get();
      PanelManager.setCols(st.cols || 1);
      PanelManager.loadPlot(startPlot);
      Picker.setFilter(st.filter || '');
      Reloader.start(payload);

      // First render is now complete. Commands run after it, sequentially,
      // each list stopping at its own first failure (§4.4).
      return Commands.run(options.commands || []).then(function () {
        return Commands.run(options.queryCommands || []);
      });
    }

    function applyReload(newPayload, local_file) {
      validatePayload(newPayload);
      var activeBefore = activePlotIdx;
      PanelManager.persistCurrentLayout();
      if (local_file) {
        StateStore.saveNow();
        PanelManager.teardown();
      }
      Store.init(newPayload);
      PAYLOAD = newPayload;
      if (options.onTitle) options.onTitle(newPayload.run.title || 'silk waveform explorer');
      setText(el('run-title'), newPayload.run.title || '');
      // Every scope's data.js -- and every periodic reload of the SAME
      // scope -- carries its own (possibly absent) `context` copy, so a
      // switch/reload can never leave a stale header on screen (design
      // §2.1). Shared by the periodic Reloader and ScopeManager, since
      // both funnel through this one function.
      Context.render();
      buildPlotSelect(newPayload);
      if (local_file) {
        StateStore.load(stateKeyFor(newPayload));
        ViewState.rerenderFromState();
        return;
      }
      Store.restore_derived(StateStore.get().plots);
      var idx = newPayload.plots[activeBefore] ? activeBefore : newPayload.default_plot;
      el('plot-select').value = String(idx);
      PanelManager.loadPlot(idx);
      Picker.rebuild();
    }

    return { init: init, applyReload: applyReload };
  })();

  // ======================================================================
  // Host lifecycle
  // ======================================================================

  CONFIG = Object.assign(CONFIG, options.config || {});
  var resizeObserver = new window.ResizeObserver(function () {
    if (!destroyed && initialized) PanelManager.onResize();
  });
  resizeObserver.observe(root);

  function load(payload) {
    if (destroyed) throw new Error('Explorer has been destroyed');
    validatePayload(payload);
    ++dataGen;
    pendingLoads.forEach(function (cancel) { cancel(); });
    Reloader.stop();
    if (initialized) {
      PanelManager.persistCurrentLayout();
      StateStore.saveNow();
      PanelManager.teardown();
    }
    el('fatal-banner').hidden = true;
    el('app').style.display = '';
    return App.init(payload);
  }
  function reload(payload) {
    if (destroyed) throw new Error('Explorer has been destroyed');
    validatePayload(payload);
    if (!initialized) return load(payload);
    ++dataGen;
    pendingLoads.forEach(function (cancel) { cancel(); });
    App.applyReload(payload);
    Reloader.start(payload);
    return Promise.resolve();
  }
  function destroy() {
    if (destroyed) return;
    if (initialized) { PanelManager.persistCurrentLayout(); StateStore.saveNow(); }
    destroyed = true;
    ++dataGen;
    pendingLoads.forEach(function (cancel) { cancel(); });
    Reloader.stop();
    bridge.data = function () {};
    bridge.stamp = function () {};
    readers.forEach(function (reader) { reader.abort(); });
    readers.clear();
    events.abort();
    resizeObserver.disconnect();
    PanelManager.teardown();
    timers.forEach(function (id) { window.clearTimeout(id); }); timers.clear();
    frames.forEach(function (id) { window.cancelAnimationFrame(id); }); frames.clear();
    objectURLs.forEach(revokeObjectURL);
    Store.init(null);
    PAYLOAD = null;
    root.replaceChildren(...previousChildren);
    previousChildren = [];
    var originalClasses = (previousClass || '').split(/\s+/);
    ['silk-explorer', 'sidebar-open', 'view-drop-active'].forEach(function (name) {
      root.classList.toggle(name, originalClasses.indexOf(name) !== -1);
    });
    if (previousClass == null && !root.className) root.removeAttribute('class');
    if (previousTabindex == null && root.getAttribute('tabindex') === '0') root.removeAttribute('tabindex');
    if (previousFontSize) root.style.setProperty('--fs-base', previousFontSize, previousFontPriority);
    else root.style.removeProperty('--fs-base');
    mountedRoots.delete(root);
  }
  function publicTraceIdentity(name, plotIdx) {
    if (!initialized || destroyed) throw new Error('traceIdentity needs a loaded explorer');
    if (typeof name !== 'string') throw new TypeError('traceIdentity(name, plot?) needs a signal name string');
    if (plotIdx != null && !Number.isInteger(plotIdx)) throw new TypeError('traceIdentity plot must be an integer index');
    return traceIdentity(name, plotIdx == null ? activePlotIdx : plotIdx);
  }
  return { load: load, reload: reload, run: Commands.run, destroy: destroy,
    commands: Commands.commands, version: '1.4',
    traceIdentity: publicTraceIdentity, silkCode: silkCode };
}
