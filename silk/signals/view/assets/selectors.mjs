/** Shared textual selectors; Python produces all graph evidence. */
import {to_spice_name} from './waveform-core.mjs';
class ViewError extends Error { constructor(message) { super(message); this.name = 'ViewError'; } }
// Mirrors the Python selection errors in silk/signals/view/template.py.
export class SelectionError extends ViewError { constructor(message) { super(message); this.name = 'SelectionError'; } }
export class SelectorNotFoundError extends SelectionError {
  constructor(message, selector, kwarg) { super(message); this.name = 'SelectorNotFoundError'; this.selector = selector; this.kwarg = kwarg; }
}
export class AmbiguousSelectorError extends SelectionError {
  constructor(message, selector, kwarg) { super(message); this.name = 'AmbiguousSelectorError'; this.selector = selector; this.kwarg = kwarg; }
}
export class EmptySelectionError extends SelectionError { constructor(message) { super(message); this.name = 'EmptySelectionError'; } }
  var HIER_WRAP_RE = /^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$/;

  export function hierParse(sig) {
    var base = sig.name;
    if (sig.part === 'mag' && base.slice(-6) === ' (mag)') base = base.slice(0, -6);
    else if (sig.part === 'phase' && base.slice(-8) === ' (phase)') base = base.slice(0, -8);

    var wrapper = null, inner = base;
    var m = HIER_WRAP_RE.exec(base);
    if (m) { wrapper = m[1]; inner = m[2]; }

    var segments = [];
    var cur = '';
    var depth = 0;
    for (var i = 0; i < inner.length; i++) {
      var c = inner[i];
      if (c === '[') depth++;
      else if (c === ']') depth = Math.max(0, depth - 1);
      // ngspice uses dots, Xyce colons; bracket contents stay literal.
      if ((c === '.' || c === ':') && depth === 0) {
        segments.push(cur);
        cur = '';
      } else {
        cur += c;
      }
    }
    segments.push(cur);
    segments = segments.filter(function (s) { return s.length > 0; });

    var leaf = segments.length ? segments[segments.length - 1] : '';
    var groupSegs = segments.slice(0, segments.length - 1);
    return { segments: groupSegs, leaf: leaf, wrapper: wrapper };
  }


const wrapped = text => /^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$/.exec(text);
const glob = (pattern, text) => {
  const rx = Array.from(pattern, c => c === '*' ? '[\\s\\S]*' : c === '?' ? '[\\s\\S]' : c.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('');
  return new RegExp('^(?:' + rx + ')$(?![\\s\\S])', 'u').test(text);
};

/** Same three selector forms and literal bracket rules as Python select.py. */
export function matchesSelector(name, selector) {
  const lower = name.toLowerCase(), sel = selector.toLowerCase();
  const head = wrapped(name), role = head && ['i','id'].includes(head[1].toLowerCase()) ? 'branch' : 'node';
  for (const candidate of [sel, to_spice_name(sel, {kind:role,allow_glob:true})]) {
    if (lower === candidate || glob(candidate, lower)) return true;
  }
  const sh = wrapped(selector);
  if (sh && ['v','i','n','id'].includes(sh[1].toLowerCase())) return false;
  const original = sel.split('.');
  const converted = to_spice_name(sel, {kind:'branch',allow_glob:true});
  let readings = converted.split('.');
  if (converted === sel || readings.length !== original.length || readings.some(s=>!s)) readings = original.map(()=>null);
  const pairs = original.map((s,i)=>[s,readings[i]]).filter(([s])=>s);
  if (!pairs.length) return false;
  const parsed = hierParse({name}), comps = parsed.segments.map(s=>s.toLowerCase());
  const flags = comps.map(()=>true);
  if (parsed.leaf) {
    comps.push(parsed.leaf.toLowerCase());
    flags.push((parsed.wrapper || '').toLowerCase()==='i' || parsed.leaf.startsWith('@') || (parsed.wrapper===null && name.startsWith('@')));
  }
  for (let i=0;i<=comps.length-pairs.length;i++) {
    if (pairs.every(([lit,inst],k)=>glob(lit,comps[i+k]) || (flags[i+k] && inst!==null && glob(inst,comps[i+k])))) return true;
  }
  return false;
}

export function validateNodePaths(meta) {
  const bad = () => {throw new ViewError('invalid node_paths metadata');};
  if (!meta || meta.version!==1 || !Array.isArray(meta.vectors) || !Array.isArray(meta.paths)) bad();
  for (const v of meta.vectors) if (!v || typeof v.name!=='string' || typeof v.selectable!=='boolean') bad();
  const names=new Set(), targets=new Map();
  for (const p of meta.paths) {
    if (!p || typeof p.name!=='string' || !p.name || names.has(p.name) || !Array.isArray(p.targets) || !p.targets.length) bad();
    names.add(p.name);
    const seen=new Set();
    for (const t of p.targets) {
      if (!t || !Number.isSafeInteger(t.id) || t.id<0 || seen.has(t.id) || !Array.isArray(t.rows)) bad();
      if (t.rows.some((r,i)=>!Number.isSafeInteger(r) || r<0 || r>=meta.vectors.length || (i>0 && r<=t.rows[i-1]))) bad();
      const rows=JSON.stringify(t.rows);
      if (targets.has(t.id) && targets.get(t.id)!==rows) bad();
      seen.add(t.id); targets.set(t.id,rows);
    }
  }
  return meta;
}

// Unicode 15.0.0 case-fold exceptions to lowercase, generated with Python str.casefold.
const CASE_FOLD = {"\u00b5":"\u03bc","\u00df":"ss","\u0149":"\u02bcn","\u017f":"s","\u01f0":"j\u030c","\u0345":"\u03b9","\u0390":"\u03b9\u0308\u0301","\u03b0":"\u03c5\u0308\u0301","\u03c2":"\u03c3","\u03d0":"\u03b2","\u03d1":"\u03b8","\u03d5":"\u03c6","\u03d6":"\u03c0","\u03f0":"\u03ba","\u03f1":"\u03c1","\u03f5":"\u03b5","\u0587":"\u0565\u0582","\u13a0":"\u13a0","\u13a1":"\u13a1","\u13a2":"\u13a2","\u13a3":"\u13a3","\u13a4":"\u13a4","\u13a5":"\u13a5","\u13a6":"\u13a6","\u13a7":"\u13a7","\u13a8":"\u13a8","\u13a9":"\u13a9","\u13aa":"\u13aa","\u13ab":"\u13ab","\u13ac":"\u13ac","\u13ad":"\u13ad","\u13ae":"\u13ae","\u13af":"\u13af","\u13b0":"\u13b0","\u13b1":"\u13b1","\u13b2":"\u13b2","\u13b3":"\u13b3","\u13b4":"\u13b4","\u13b5":"\u13b5","\u13b6":"\u13b6","\u13b7":"\u13b7","\u13b8":"\u13b8","\u13b9":"\u13b9","\u13ba":"\u13ba","\u13bb":"\u13bb","\u13bc":"\u13bc","\u13bd":"\u13bd","\u13be":"\u13be","\u13bf":"\u13bf","\u13c0":"\u13c0","\u13c1":"\u13c1","\u13c2":"\u13c2","\u13c3":"\u13c3","\u13c4":"\u13c4","\u13c5":"\u13c5","\u13c6":"\u13c6","\u13c7":"\u13c7","\u13c8":"\u13c8","\u13c9":"\u13c9","\u13ca":"\u13ca","\u13cb":"\u13cb","\u13cc":"\u13cc","\u13cd":"\u13cd","\u13ce":"\u13ce","\u13cf":"\u13cf","\u13d0":"\u13d0","\u13d1":"\u13d1","\u13d2":"\u13d2","\u13d3":"\u13d3","\u13d4":"\u13d4","\u13d5":"\u13d5","\u13d6":"\u13d6","\u13d7":"\u13d7","\u13d8":"\u13d8","\u13d9":"\u13d9","\u13da":"\u13da","\u13db":"\u13db","\u13dc":"\u13dc","\u13dd":"\u13dd","\u13de":"\u13de","\u13df":"\u13df","\u13e0":"\u13e0","\u13e1":"\u13e1","\u13e2":"\u13e2","\u13e3":"\u13e3","\u13e4":"\u13e4","\u13e5":"\u13e5","\u13e6":"\u13e6","\u13e7":"\u13e7","\u13e8":"\u13e8","\u13e9":"\u13e9","\u13ea":"\u13ea","\u13eb":"\u13eb","\u13ec":"\u13ec","\u13ed":"\u13ed","\u13ee":"\u13ee","\u13ef":"\u13ef","\u13f0":"\u13f0","\u13f1":"\u13f1","\u13f2":"\u13f2","\u13f3":"\u13f3","\u13f4":"\u13f4","\u13f5":"\u13f5","\u13f8":"\u13f0","\u13f9":"\u13f1","\u13fa":"\u13f2","\u13fb":"\u13f3","\u13fc":"\u13f4","\u13fd":"\u13f5","\u1c80":"\u0432","\u1c81":"\u0434","\u1c82":"\u043e","\u1c83":"\u0441","\u1c84":"\u0442","\u1c85":"\u0442","\u1c86":"\u044a","\u1c87":"\u0463","\u1c88":"\ua64b","\u1e96":"h\u0331","\u1e97":"t\u0308","\u1e98":"w\u030a","\u1e99":"y\u030a","\u1e9a":"a\u02be","\u1e9b":"\u1e61","\u1e9e":"ss","\u1f50":"\u03c5\u0313","\u1f52":"\u03c5\u0313\u0300","\u1f54":"\u03c5\u0313\u0301","\u1f56":"\u03c5\u0313\u0342","\u1f80":"\u1f00\u03b9","\u1f81":"\u1f01\u03b9","\u1f82":"\u1f02\u03b9","\u1f83":"\u1f03\u03b9","\u1f84":"\u1f04\u03b9","\u1f85":"\u1f05\u03b9","\u1f86":"\u1f06\u03b9","\u1f87":"\u1f07\u03b9","\u1f88":"\u1f00\u03b9","\u1f89":"\u1f01\u03b9","\u1f8a":"\u1f02\u03b9","\u1f8b":"\u1f03\u03b9","\u1f8c":"\u1f04\u03b9","\u1f8d":"\u1f05\u03b9","\u1f8e":"\u1f06\u03b9","\u1f8f":"\u1f07\u03b9","\u1f90":"\u1f20\u03b9","\u1f91":"\u1f21\u03b9","\u1f92":"\u1f22\u03b9","\u1f93":"\u1f23\u03b9","\u1f94":"\u1f24\u03b9","\u1f95":"\u1f25\u03b9","\u1f96":"\u1f26\u03b9","\u1f97":"\u1f27\u03b9","\u1f98":"\u1f20\u03b9","\u1f99":"\u1f21\u03b9","\u1f9a":"\u1f22\u03b9","\u1f9b":"\u1f23\u03b9","\u1f9c":"\u1f24\u03b9","\u1f9d":"\u1f25\u03b9","\u1f9e":"\u1f26\u03b9","\u1f9f":"\u1f27\u03b9","\u1fa0":"\u1f60\u03b9","\u1fa1":"\u1f61\u03b9","\u1fa2":"\u1f62\u03b9","\u1fa3":"\u1f63\u03b9","\u1fa4":"\u1f64\u03b9","\u1fa5":"\u1f65\u03b9","\u1fa6":"\u1f66\u03b9","\u1fa7":"\u1f67\u03b9","\u1fa8":"\u1f60\u03b9","\u1fa9":"\u1f61\u03b9","\u1faa":"\u1f62\u03b9","\u1fab":"\u1f63\u03b9","\u1fac":"\u1f64\u03b9","\u1fad":"\u1f65\u03b9","\u1fae":"\u1f66\u03b9","\u1faf":"\u1f67\u03b9","\u1fb2":"\u1f70\u03b9","\u1fb3":"\u03b1\u03b9","\u1fb4":"\u03ac\u03b9","\u1fb6":"\u03b1\u0342","\u1fb7":"\u03b1\u0342\u03b9","\u1fbc":"\u03b1\u03b9","\u1fbe":"\u03b9","\u1fc2":"\u1f74\u03b9","\u1fc3":"\u03b7\u03b9","\u1fc4":"\u03ae\u03b9","\u1fc6":"\u03b7\u0342","\u1fc7":"\u03b7\u0342\u03b9","\u1fcc":"\u03b7\u03b9","\u1fd2":"\u03b9\u0308\u0300","\u1fd3":"\u03b9\u0308\u0301","\u1fd6":"\u03b9\u0342","\u1fd7":"\u03b9\u0308\u0342","\u1fe2":"\u03c5\u0308\u0300","\u1fe3":"\u03c5\u0308\u0301","\u1fe4":"\u03c1\u0313","\u1fe6":"\u03c5\u0342","\u1fe7":"\u03c5\u0308\u0342","\u1ff2":"\u1f7c\u03b9","\u1ff3":"\u03c9\u03b9","\u1ff4":"\u03ce\u03b9","\u1ff6":"\u03c9\u0342","\u1ff7":"\u03c9\u0342\u03b9","\u1ffc":"\u03c9\u03b9","\uab70":"\u13a0","\uab71":"\u13a1","\uab72":"\u13a2","\uab73":"\u13a3","\uab74":"\u13a4","\uab75":"\u13a5","\uab76":"\u13a6","\uab77":"\u13a7","\uab78":"\u13a8","\uab79":"\u13a9","\uab7a":"\u13aa","\uab7b":"\u13ab","\uab7c":"\u13ac","\uab7d":"\u13ad","\uab7e":"\u13ae","\uab7f":"\u13af","\uab80":"\u13b0","\uab81":"\u13b1","\uab82":"\u13b2","\uab83":"\u13b3","\uab84":"\u13b4","\uab85":"\u13b5","\uab86":"\u13b6","\uab87":"\u13b7","\uab88":"\u13b8","\uab89":"\u13b9","\uab8a":"\u13ba","\uab8b":"\u13bb","\uab8c":"\u13bc","\uab8d":"\u13bd","\uab8e":"\u13be","\uab8f":"\u13bf","\uab90":"\u13c0","\uab91":"\u13c1","\uab92":"\u13c2","\uab93":"\u13c3","\uab94":"\u13c4","\uab95":"\u13c5","\uab96":"\u13c6","\uab97":"\u13c7","\uab98":"\u13c8","\uab99":"\u13c9","\uab9a":"\u13ca","\uab9b":"\u13cb","\uab9c":"\u13cc","\uab9d":"\u13cd","\uab9e":"\u13ce","\uab9f":"\u13cf","\uaba0":"\u13d0","\uaba1":"\u13d1","\uaba2":"\u13d2","\uaba3":"\u13d3","\uaba4":"\u13d4","\uaba5":"\u13d5","\uaba6":"\u13d6","\uaba7":"\u13d7","\uaba8":"\u13d8","\uaba9":"\u13d9","\uabaa":"\u13da","\uabab":"\u13db","\uabac":"\u13dc","\uabad":"\u13dd","\uabae":"\u13de","\uabaf":"\u13df","\uabb0":"\u13e0","\uabb1":"\u13e1","\uabb2":"\u13e2","\uabb3":"\u13e3","\uabb4":"\u13e4","\uabb5":"\u13e5","\uabb6":"\u13e6","\uabb7":"\u13e7","\uabb8":"\u13e8","\uabb9":"\u13e9","\uabba":"\u13ea","\uabbb":"\u13eb","\uabbc":"\u13ec","\uabbd":"\u13ed","\uabbe":"\u13ee","\uabbf":"\u13ef","\ufb00":"ff","\ufb01":"fi","\ufb02":"fl","\ufb03":"ffi","\ufb04":"ffl","\ufb05":"st","\ufb06":"st","\ufb13":"\u0574\u0576","\ufb14":"\u0574\u0565","\ufb15":"\u0574\u056b","\ufb16":"\u057e\u0576","\ufb17":"\u0574\u056d"};
const caseFold = text => Array.from(text, c => CASE_FOLD[c] ?? c.toLowerCase()).join('');

/** Full source-row masks; no graph is required for raw-name selection. */
export function selectNodePaths(plots, {signals=null, exclude=null, max_depth=null}={}) {
  const diagnostics=[];
  for (const p of plots) {
    validateNodePaths(p.node_paths || {version:1,vectors:p.vectors,paths:[]});
    if (p.node_paths && (p.vectors.length!==p.node_paths.vectors.length || p.vectors.some((v,i)=>v.name!==p.node_paths.vectors[i].name || v.selectable!==p.node_paths.vectors[i].selectable))) throw new ViewError('node_paths inventory does not match plot vectors');
  }
  const key=(pi,row)=>pi+','+row;
  const expand=(selectors,keyword)=>{
    const chosen=new Set();
    for (const selector of selectors) {
      const emitted=new Set(), designer=new Set(), exact=new Set();
      plots.forEach((p,pi)=>{
        p.vectors.forEach((v,row)=>{if(v.selectable && matchesSelector(v.name,selector)) emitted.add(key(pi,row));});
        for (const path of p.node_paths?.paths || []) {
          const full='v('+path.name+')';
          const isExact=[caseFold(path.name),caseFold(full)].includes(caseFold(selector));
          if (isExact) for (const t of path.targets) exact.add(t.id);
          if (matchesSelector(full,selector)) {
            for (const t of path.targets) {
              for (const row of t.rows) if (p.vectors[row].selectable) designer.add(key(pi,row));
            }
          }
        }
      });
      let found;
      if (emitted.size) {
        if (designer.size && (emitted.size!==designer.size || [...emitted].some(k=>!designer.has(k)))) diagnostics.push({kind:'emitted-precedence',selector,keyword});
        found=emitted;
      } else {
        if (exact.size>1) throw new AmbiguousSelectorError(keyword+' selector '+selector+' is ambiguous: distinct physical nodes', selector, keyword);
        found=designer;
      }
      if (!found.size) throw new SelectorNotFoundError(keyword+' selector '+selector+' matched no signal', selector, keyword);
      for(const k of found) chosen.add(k);
    }
    return chosen;
  };
  let selected;
  if(signals===null) {
    selected=new Set();
    plots.forEach((p,pi)=>p.vectors.forEach((v,row)=>{if(v.selectable) selected.add(key(pi,row));}));
  } else selected=expand(signals,'signals');
  if(exclude) for(const k of expand(exclude,'exclude')) selected.delete(k);
  const masks=plots.map((p,pi)=>p.vectors.map((v,row)=>selected.has(key(pi,row)) && (max_depth===null || hierParse({name:v.name}).segments.length<=max_depth)));
  if(!masks.some(m=>m.some(Boolean))) throw new EmptySelectionError('signal selection matched zero signals — nothing would be emitted');
  return {masks,diagnostics};
}
