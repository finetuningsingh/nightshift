/* Nightshift dashboard. Vanilla JS, no dependencies, no network except GET /api/state on this origin.
 *
 * Trust model: every string in the state JSON came from a language model or from experiment output, so it is
 * untrusted. Dynamic text only ever reaches the DOM through textContent / text nodes, and dynamic attributes only
 * through setAttribute on a fixed allow-list. There is no markup parsing anywhere in this file, and SVG is built with
 * createElementNS. The DOM is patched in place by key, so polling never flickers and never loses selection or scroll.
 */
'use strict';
(function () {
  // ===========================================================================================
  // constants
  // ===========================================================================================
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const POLL_ACTIVE_MS = 1000;
  const POLL_DONE_MS = 5000;
  const BACKOFF_MAX_MS = 15000;
  const FETCH_TIMEOUT_MS = 8000;
  const MAX_BODY_CHARS = 12 * 1024 * 1024;
  const TIMELINE_RENDER_MAX = 1500;
  const TEXT_MAX = 2400;
  const SCRIPTED_BANNER = 'SCRIPTED BACKEND: no language model is running. This is a fault-injection harness proving the ' +
    'audit machinery, not an AI result.';
  const SIMULATED_BANNER = 'SIMULATED MODEL: the language model endpoint is a local test server. This run exercises the live ' +
    'code path (prompts, tool calls, metering, audit) but is not Meta model output.';
  // Shown, never dismissible, whenever the operator planted scripted attacker lanes into a meta run (run.red_team is non-empty).
  const RED_TEAM_NOTICE = 'RED-TEAM LANES: {n} lane(s) in this run are scripted attackers injected by the operator. ' +
    'The supervisor, designer, reviewer and auditor are real model calls.';
  // What a SIMULATED run (a local test server standing in for the model) says wherever a figure or a name could pass for a Meta model
  // result. Copies of the strings in nightshift/dashboard/state_api.py and the report; tests/test_frontend.py pins them equal.
  const SIM_COST_LABEL = 'simulated list-price estimate (nothing was billed)';
  const SIM_MODELS_LABEL = 'Models the simulator reported answering (a stand-in, not Meta)';
  const SIM_JUDGE_LABEL = 'checked by a simulated judge';
  const SIM_SUPERVISOR_LABEL = 'written by the simulated supervisor (a scripted stand-in), not a model';
  const SIM_RED_TEAM_NOTE = 'In this simulated run those model roles are answered by the local simulator (a scripted stand-in), not by a model.';
  const STAGES = [['design', 'Design'], ['implement', 'Implement'], ['run', 'Run'], ['grade', 'Grade'],
    ['review', 'Review'], ['record', 'Record']];
  const STATUS_ACTIVE = { proposed: 0, workspace: 1, implemented: 2, running: 2, ran: 3, graded: 4, reviewed: 5 };
  const STAGE_INDEX = { design: 0, implement: 1, run: 2, grade: 3, review: 4, repair: 4, record: 5 };
  // own-property lookup: a state value such as "constructor" must not resolve to Object.prototype members
  const own = (map, key) => (Object.prototype.hasOwnProperty.call(map, key) ? map[key] : undefined);
  const THEME_ORDER = ['auto', 'light', 'dark'];
  const THEME_LABEL = { auto: 'Auto', light: 'Light', dark: 'Dark' };
  const TL_GROUPS = ['run', 'supervisor', 'claim', 'experiment', 'incumbent', 'memory', 'budget', 'audit', 'watchdog'];
  const SAFE_ATTR = /^(?:class|id|role|tabindex|type|title|for|name|value|placeholder|hidden|disabled|scope|lang|dir|aria-[a-z]+|data-[a-z-]+)$/;

  const ICONS = {
    check: 'M5 12.5l4.5 4.5L19 7.5',
    x: 'M6 6l12 12M18 6L6 18',
    alert: 'M12 3.5 2.5 20h19L12 3.5zM12 10v4.5M12 17.4v.4',
    dash: 'M6 12h12',
    help: 'M9.2 9.2a2.9 2.9 0 1 1 4.4 2.5c-1 .6-1.6 1.2-1.6 2.3M12 17.3v.4',
    clock: 'M12 3.5a8.5 8.5 0 1 0 0 17 8.5 8.5 0 0 0 0-17zM12 7.5V12l3 2',
    up: 'M12 19V5M6 11l6-6 6 6',
    down: 'M12 5v14M6 13l6 6 6-6',
    dot: 'M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7z',
    half: 'M12 3.5a8.5 8.5 0 1 0 0 17zM12 3.5v17',
    ban: 'M12 3.5a8.5 8.5 0 1 0 0 17 8.5 8.5 0 0 0 0-17zM6 6l12 12',
    repeat: 'M17 3l4 4-4 4M3 11V9a3 3 0 0 1 3-3h15M7 21l-4-4 4-4M21 13v2a3 3 0 0 1-3 3H3',
    flag: 'M5 21V4M5 4h11l-2.5 4L16 12H5',
    flask: 'M9 3h6M10 3v6.2L4.8 18.4A1.6 1.6 0 0 0 6.2 21h11.6a1.6 1.6 0 0 0 1.4-2.6L14 9.2V3',
    bolt: 'M13 3 5 13.2h6L10 21l8-10.2h-6L13 3z',
    file: 'M14 3H7.5A1.5 1.5 0 0 0 6 4.5v15A1.5 1.5 0 0 0 7.5 21h9a1.5 1.5 0 0 0 1.5-1.5V7l-4-4zM14 3v4h4',
    chat: 'M4 5h16v11H9.5L4 20.5V5z',
    chain: 'M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3A4 4 0 0 0 11 18.7l1-1',
    box: 'M4 8l8-4 8 4v8l-8 4-8-4V8zM4 8l8 4 8-4M12 12v8',
    arrow: 'M5 12h14M13 6l6 6-6 6',
    doc: 'M7 4h10M7 9h10M7 14h6',
    coin: 'M12 3.5a8.5 8.5 0 1 0 0 17 8.5 8.5 0 0 0 0-17zM14.6 9.3c-.5-.8-1.4-1.2-2.6-1.2-1.5 0-2.5.8-2.5 1.9 0 2.8 5.2 1.3 5.2 4 0 1.2-1.1 1.9-2.7 1.9-1.2 0-2.2-.4-2.7-1.3M12 6.3v1.8M12 16v1.8',
  };

  // ===========================================================================================
  // small utilities (all total functions: they never throw on odd input)
  // ===========================================================================================
  const $ = (sel, root) => (root || document).querySelector(sel);
  const isObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
  const arr = (v) => (Array.isArray(v) ? v : []);
  const objs = (v) => arr(v).filter(isObj);
  const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);

  function str(v, max) {
    let s;
    if (typeof v === 'string') s = v;
    else if (v === null || v === undefined) s = '';
    else if (typeof v === 'number' || typeof v === 'boolean') s = String(v);
    else {
      try { s = JSON.stringify(v); } catch (e) { s = ''; }
      if (typeof s !== 'string') s = '';
    }
    const cap = max || TEXT_MAX;
    // eslint-disable-next-line no-control-regex
    s = s.replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f‪-‮⁦-⁩]/g, '');
    return s.length > cap ? s.slice(0, cap - 1) + '…' : s;
  }

  const clip = (s, n) => (s.length > n ? s.slice(0, n - 1).trimEnd() + '…' : s);
  const cls = (...names) => names.filter(Boolean).join(' ');
  const pad2 = (n) => (n < 10 ? '0' + n : String(n));

  function fmtNum(v, digits) {
    const n = num(v);
    if (n === null) return '—';
    if (digits !== undefined) return n.toFixed(digits);
    const a = Math.abs(n);
    if (a >= 1000) return n.toFixed(0);
    if (a >= 1) return n.toFixed(3);
    return n.toFixed(4);
  }
  function fmtSigned(v) {
    const n = num(v);
    if (n === null) return '—';
    const s = fmtNum(Math.abs(n));
    // a value that rounds to zero carries no sign: "-0.0000" would read as a small loss
    if (/^0\.?0*$/.test(s)) return s;
    return n > 0 ? '+' + s : n < 0 ? '−' + s : s;
  }
  // Money: cents for the usual range, three decimals below ten cents (a single model call costs fractions of a cent),
  // and never "$0.00" for a non-zero amount.
  function fmtUsd(v) {
    const n = num(v);
    if (n === null) return '—';
    const a = Math.abs(n);
    if (a === 0) return '$0.00';
    if (a >= 100) return '$' + n.toFixed(0);
    if (a >= 0.1) return '$' + n.toFixed(2);
    if (a >= 0.001) return '$' + n.toFixed(3);
    return (n < 0 ? '−' : '') + '<$0.001';
  }
  function fmtInt(v) {
    const n = num(v);
    return n === null ? '—' : Math.round(n).toLocaleString('en-US');
  }
  function fmtDur(v) {
    const n = num(v);
    if (n === null) return '—';
    const s = Math.max(0, Math.round(n));
    if (s < 60) return s + ' s';
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
    return h > 0 ? h + ':' + pad2(m) + ':' + pad2(r) : m + ':' + pad2(r);
  }
  function fmtRel(ts, start) {
    const t = num(ts), s = num(start);
    if (t === null) return '';
    if (s === null || s <= 0) return clockOf(new Date(t * 1000));
    const d = Math.max(0, Math.round(t - s));
    const h = Math.floor(d / 3600), m = Math.floor((d % 3600) / 60), r = d % 60;
    return '+' + (h > 0 ? h + ':' + pad2(m) : pad2(m)) + ':' + pad2(r);
  }
  function clockOf(d) { return pad2(d.getHours()) + ':' + pad2(d.getMinutes()) + ':' + pad2(d.getSeconds()); }

  function modelLabel(id) {
    const s = str(id, 80);
    const m = /^muse-(spark|wonder)-(.+)$/.exec(s);
    if (m) return 'Muse ' + (m[1] === 'spark' ? 'Spark' : 'Wonder') + ' ' + m[2];
    return s;
  }
  function words(s) { return str(s, 60).replace(/_/g, ' '); }

  // ---- DOM construction (text only) --------------------------------------------------------
  function appendKids(node, kids) {
    for (const k of kids) {
      if (k === null || k === undefined || k === false) continue;
      if (Array.isArray(k)) appendKids(node, k);
      else if (k.nodeType) node.append(k);
      else node.append(document.createTextNode(String(k)));
    }
  }
  function h(tag, props, ...kids) {
    const node = document.createElement(tag);
    if (props) {
      for (const k of Object.keys(props)) {
        const v = props[k];
        if (v === null || v === undefined || v === false) continue;
        if (k === 'class') node.className = v;
        else if (k === 'text') node.textContent = v;
        else if (SAFE_ATTR.test(k)) node.setAttribute(k, v === true ? '' : String(v));
      }
    }
    appendKids(node, kids);
    return node;
  }
  function s(tag, attrs, ...kids) {
    const node = document.createElementNS(SVG_NS, tag);
    if (attrs) for (const k of Object.keys(attrs)) { if (attrs[k] !== null && attrs[k] !== undefined) node.setAttribute(k, String(attrs[k])); }
    appendKids(node, kids);
    return node;
  }
  function icon(name, extra) {
    const svg = s('svg', { viewBox: '0 0 24 24', class: cls('icon', extra), 'aria-hidden': 'true', focusable: 'false' });
    svg.append(s('path', { d: ICONS[name] || ICONS.dot }));
    return svg;
  }

  // ---- in-place update helpers --------------------------------------------------------------
  function setText(node, text) {
    const t = String(text);
    if (node.textContent !== t) node.textContent = t;
  }
  function setClass(node, name) { if (node.className !== name) node.className = name; }
  function setAttr(node, name, value) {
    if (value === null || value === undefined || value === false) { if (node.hasAttribute(name)) node.removeAttribute(name); return; }
    const v = value === true ? '' : String(value);
    if (node.getAttribute(name) !== v) node.setAttribute(name, v);
  }
  function setHidden(node, hidden) { if (node.hidden !== !!hidden) node.hidden = !!hidden; }
  function setIcon(slot, name) {
    if (slot._icon === name) return;
    slot._icon = name;
    slot.replaceChildren(name ? icon(name) : '');
  }

  // Keyed reconciliation: nodes are created once per key and then only patched or moved.
  function syncList(container, items, keyOf, make, patch) {
    const existing = new Map();
    for (const c of container.children) { if (c.dataset.key !== undefined) existing.set(c.dataset.key, c); }
    const used = new Set();
    let i = 0;
    for (const item of items) {
      let key = String(keyOf(item, i));
      if (used.has(key)) key += '#' + i;
      used.add(key);
      let node = existing.get(key);
      if (!node) { node = make(item, key); node.dataset.key = key; }
      patch(node, item, i);
      const at = container.children[i];
      if (at !== node) container.insertBefore(node, at || null);
      i += 1;
    }
    for (const [k, node] of existing) { if (!used.has(k)) node.remove(); }
  }

  // A tag is a chip or pill: optional icon, text, tone, optional action (then it is a button).
  function makeTag(spec, base) {
    const node = h(spec.act ? 'button' : 'span', { class: base || 'chip', type: spec.act ? 'button' : null });
    node._ic = h('span', { class: 'tag-ic' });
    node._tx = h('span', { class: 'tag-tx' });
    node.append(node._ic, node._tx);
    node._base = base || 'chip';
    return node;
  }
  function patchTag(node, spec) {
    setClass(node, cls(node._base, spec.tone ? 'tone-' + spec.tone : '', spec.mono ? 'mono' : '', spec.off ? 'is-off' : ''));
    setIcon(node._ic, spec.icon || '');
    setHidden(node._ic, !spec.icon);
    setText(node._tx, spec.text);
    setAttr(node, 'title', spec.title || null);
    if (spec.act) {
      setAttr(node, 'data-act', spec.act);
      setAttr(node, 'data-arg', spec.arg === undefined ? null : spec.arg);
      setAttr(node, 'aria-disabled', spec.off ? 'true' : null);
    }
    node._exps = spec.exps || null;
    if (spec.exps) node.classList.add('js-exp');
  }
  function tagList(container, specs, base) {
    syncList(container, specs, (sp) => sp.key, (sp) => makeTag(sp, base), patchTag);
  }

  function setEmpty(el, title, text, show) {
    if (!show) { setHidden(el, true); return; }
    if (!el._init) {
      el._t = h('p', { class: 'empty-t' });
      el._d = h('p', { class: 'empty-d' });
      el.append(icon('doc', 'empty-ic'), el._t, el._d);
      el._init = true;
    }
    setText(el._t, title);
    setText(el._d, text);
    setHidden(el, false);
  }

  // ===========================================================================================
  // UI state (selection survives every poll; nothing here is derived from the network)
  // ===========================================================================================
  const ui = {
    S: null, raw: '', runId: null,
    selExp: null, selInv: null,
    newestFirst: true,
    expanded: new Set(),
    tlHidden: new Set(), tlQuery: '',
    theme: 'auto',
    chartW: 0, chartSig: '', chartModel: null,
    flashTimer: null, flashRow: null,
    follow: { turns: true, tl: true },
    conn: { ok: null, lastOk: 0, failures: 0, retryAt: 0 },
    findingsById: new Map(), seqSet: new Set(),
    sim: false,                       // the run being shown was produced against the local simulator (set by render)
  };

  // ===========================================================================================
  // normalisation: one place that makes the payload safe to index
  // ===========================================================================================
  function normalize(raw) {
    if (!isObj(raw) || !isObj(raw.run)) throw new Error('state has no run object');
    const b = isObj(raw.budget) ? raw.budget : {};
    const fin = isObj(raw.finish) ? raw.finish : {};
    const fi = isObj(raw.fault_injection) ? raw.fault_injection : {};
    return {
      run: raw.run,
      banner: typeof raw.banner === 'string' && raw.banner ? raw.banner : null,
      budget: { limits: isObj(b.limits) ? b.limits : {}, used: isObj(b.used) ? b.used : {}, remaining: isObj(b.remaining) ? b.remaining : {} },
      baseline: isObj(raw.baseline) ? raw.baseline : {},
      incumbent: isObj(raw.incumbent) ? raw.incumbent : {},
      experiments: objs(raw.experiments),
      series: objs(raw.series),
      claims: objs(raw.claims),
      finish: { requests: objs(fin.requests), refusals: objs(fin.refusals), accepted: !!fin.accepted,
        acceptedClaims: Array.isArray(fin.accepted_claims) ? fin.accepted_claims.map((x) => str(x, 60)) : null },
      lessons: arr(raw.lessons).filter((x) => isObj(x) || typeof x === 'string'),
      pivots: arr(raw.pivots).filter((x) => typeof x === 'string' || isObj(x)),
      findings: objs(raw.findings),
      invariants: objs(raw.invariants),
      models_usage: objs(raw.models_usage),
      turns: objs(raw.turns),
      timeline: objs(raw.timeline),
      parallel: isObj(raw.parallel) ? raw.parallel : {},
      fault_injection: { planted: arr(fi.planted), detected: arr(fi.detected), missed: arr(fi.missed), not_evaluated: arr(fi.not_evaluated),
        not_run: arr(fi.not_run), red_team: isObj(fi.red_team) ? fi.red_team : null,
        false_positives: arr(fi.false_positives), explained: arr(fi.explained), recall: isObj(fi.recall) ? fi.recall : null },
      headline: isObj(raw.headline) ? raw.headline : {},
      ledger: isObj(raw.ledger) ? raw.ledger : {},
      replay: isObj(raw.replay) ? raw.replay : null,
    };
  }

  // ===========================================================================================
  // domain helpers shared by several panels
  // ===========================================================================================
  const expId = (e) => str(e.exp_id, 40);
  // Model spend the ledger tied to one experiment, or null when it did not (scripted runs, older ledgers, the baseline).
  function expCost(e) {
    const usd = num(e.cost_usd);
    if (usd === null) return null;
    const roles = isObj(e.cost_by_role) ? Object.keys(e.cost_by_role).sort().filter((k) => num(e.cost_by_role[k]) !== null).slice(0, 8) : [];
    const tokens = num(e.tokens), calls = num(e.llm_calls);
    const bits = roles.map((k) => str(k, 20) + ' ' + fmtUsd(e.cost_by_role[k]));
    const tail = [tokens !== null ? fmtInt(tokens) + ' tokens' : '', calls !== null ? fmtInt(calls) + (calls === 1 ? ' call' : ' calls') : ''].filter(Boolean);
    return { usd, detail: bits.concat(tail).join(' · '),
      title: 'Model calls charged to this experiment (design, implement, review and repair). Supervisor, auditor and reflector calls are not tied to one experiment.' +
        (ui.sim ? ' This figure is a ' + SIM_COST_LABEL + '.' : '') };
  }
  function outcomeInfo(e) {
    const status = str(e.status, 20), outcome = str(e.outcome, 24);
    if (status === 'blocked' && !outcome) return { tone: 'warn', icon: 'ban', text: 'blocked' };
    switch (outcome) {
      case 'baseline': return { tone: 'neutral', icon: 'box', text: 'baseline' };
      case 'improved': return { tone: 'ok', icon: 'up', text: 'improved' };
      case 'no_gain': return { tone: 'neutral', icon: 'dash', text: 'no gain' };
      case 'worse': return { tone: 'bad', icon: 'down', text: 'worse' };
      case 'failed': return { tone: 'bad', icon: 'x', text: 'failed' };
      case 'invalid': return { tone: 'bad', icon: 'x', text: 'invalid' };
      case 'infra_error': return { tone: 'warn', icon: 'alert', text: 'infra error' };
      case 'unresolved': return { tone: 'warn', icon: 'help', text: 'unresolved' };
      default: break;
    }
    if (status === 'blocked') return { tone: 'warn', icon: 'ban', text: 'blocked' };
    if (status === 'violation') return { tone: 'bad', icon: 'x', text: 'protected violation' };
    const ai = own(STATUS_ACTIVE, status) !== undefined ? own(STATUS_ACTIVE, status) : own(STAGE_INDEX, str(e.stage, 20));
    return { tone: 'live', icon: 'clock', text: 'running: ' + (ai !== undefined ? STAGES[ai][0] : 'starting') };
  }
  function replicationInfo(e) {
    if (e.awaiting_replication) return { tone: 'warn', icon: 'clock', text: 'awaiting replication', title: 'Improved on dev; not yet checked on the sealed split.' };
    if (e.replicated) return { tone: 'ok', icon: 'repeat', text: 'replicated', title: 'Confirmed on the sealed split the agents never see.' };
    if (str(e.dev_outcome) === 'improved' && num(e.sealed_delta) !== null) {
      return { tone: 'bad', icon: 'x', text: 'failed replication', title: 'Improved on dev but not on the sealed split.' };
    }
    return null;
  }
  function overfitSuspect(e) {
    if (e.overfit_suspect === true) return true;
    const id = expId(e);
    for (const f of ui.findingsById.values()) {
      if (str(f.invariant_id) === 'I02' && str(f.exp_id) === id && str(f.verdict) === 'violated') return true;
    }
    return false;
  }
  function seriesByExp(S) {
    const rank = { promoted: 3, baseline: 2, candidate: 1 };
    const out = new Map();
    for (const p of S.series) {
      const id = str(p.exp_id, 40);
      if (!id) continue;
      const prev = out.get(id);
      const kind = own(rank, str(p.kind)) ? str(p.kind) : 'candidate';
      if (prev && rank[prev.kind] > rank[kind]) { prev.score = num(p.score) === null ? prev.score : num(p.score); continue; }
      out.set(id, { id, kind, score: num(p.score), train: num(p.train), std: num(p.std), replicated: !!p.replicated, ts: num(p.ts), seq: num(p.seq) });
    }
    return out;
  }
  function expIdsIn(text) { return text.match(/\be\d{3,}\b/g) || []; }

  // ===========================================================================================
  // header, badges, budget, banner, replay
  // ===========================================================================================
  // What produced the run's words, ignoring replay: scripted | simulated | live | unknown | '' (an older state that says nothing).
  function originOf(run) {
    const o = str(run.origin, 20);
    if (o) return o;
    const m = str(run.mode, 20);
    return m === 'replay' ? '' : m;
  }
  // The model that actually answered a role (from the charges), else the configured one. Labels only; never both.
  function answeredModels(run, role) {
    const a = isObj(run.models_answered) ? run.models_answered[role] : null;
    return arr(a).map((x) => str(x, 80)).filter(Boolean);
  }

  function badge(el, spec) {
    if (!el._init) {
      el._ic = h('span', { class: 'badge-ic' });
      el._main = h('span', { class: 'badge-main' });
      el._sub = h('span', { class: 'badge-sub' });
      el.append(el._ic, el._main, el._sub);
      el._init = true;
    }
    setClass(el, cls('badge', 'tone-' + spec.tone));
    setIcon(el._ic, spec.icon);
    setText(el._main, spec.main);
    setText(el._sub, spec.sub || '');
    setHidden(el._sub, !spec.sub);
    setAttr(el, 'title', spec.title || null);
  }

  function renderHeader(S) {
    const run = S.run;
    setText($('#run-id'), str(run.run_id, 60) || '—');
    setText($('#run-task'), str(run.task, 80) || '—');
    const obj = str(run.objective, 20), metric = str(run.primary_metric, 60);
    setText($('#run-objective'), (obj || '?') + (metric ? ' ' + metric : ''));
    const taskName = str(run.task, 40);
    document.title = 'Nightshift' + (taskName ? ' · ' + taskName : '');

    const replay = S.replay, backend = str(run.backend, 20), mode = str(run.mode, 20), origin = str(run.origin, 20);
    const ep = isObj(run.endpoint) ? run.endpoint : null;
    const hosts = ep ? arr(ep.hosts).map((x) => str(x, 80)).filter(Boolean) : [];
    const led = S.ledger, chainOk = led.chain_ok === true;
    let b;
    if (replay || mode === 'replay') {
      const src = str(replay && replay.source_run_id, 40) || str(run.run_id, 40);
      b = { tone: 'replay', icon: 'repeat', main: 'REPLAY', sub: replayOf(run, replay) + (chainOk ? ' · chain verified' : ''),
        title: 'This page is replaying a finished ledger; no experiment is running.' + (src && origin ? ' The recorded run was ' + origin + '.' : '') };
    } else if (backend === 'scripted') {
      b = { tone: 'scripted', icon: 'flask', main: 'SCRIPTED', sub: 'no language model', title: 'Scripted actors drive this run. It is a fault-injection harness, not an AI result.' };
    } else if (backend === 'meta' && mode === 'simulated') {
      b = { tone: 'simulated', icon: 'flask', main: 'SIMULATED', sub: 'local test server', title: 'A local test server stands in for the model endpoint. The live code path ran; none of the text is Meta model output.' };
    } else if (backend === 'meta' && (mode === 'unknown' || origin === 'unknown')) {
      // the state refuses to call this run live (its endpoint record contradicts the backend); the page must not either
      b = { tone: 'neutral', icon: 'help', main: 'UNVERIFIED', sub: 'endpoint record contradicts backend',
        title: 'The ledger says the meta backend ran but records a scripted endpoint. Where the text came from is not established.' };
    } else if (backend === 'meta') {
      const configured = isObj(run.models) ? run.models.supervisor : '';
      const answered = answeredModels(run, 'supervisor');
      const sup = answered.length ? answered[0] : configured;
      const custom = ep && str(ep.kind, 20) === 'custom';
      b = { tone: 'live', icon: 'bolt', main: 'LIVE', sub: modelLabel(sup || 'Muse Spark') + (custom && hosts.length ? ' @ ' + hosts[0] : ''),
        title: 'Supervisor model' + (answered.length ? ' that answered: ' : ': ') + str(sup, 60) + (custom ? '. Custom endpoint host: ' + (hosts.join(', ') || 'unknown') + ' (not the Meta API host).' : '') };
    } else {
      b = { tone: 'neutral', icon: 'help', main: 'BACKEND', sub: backend || 'unknown' };
    }
    badge($('#badge-backend'), b);
    renderModelsLine(run, hosts, ep);
    const impl = str(run.implementer, 30);
    setHidden($('#run-implementer-wrap'), !impl);
    setText($('#run-implementer'), impl);
    setAttr($('#run-implementer'), 'title', implementerNote(impl));

    // run status
    const fin = run.finished === true;
    const elapsed = num(run.now) !== null && num(run.started_ts) !== null ? fmtDur(run.now - run.started_ts) : '';
    let st;
    if (!fin) {
      st = { tone: 'live', icon: 'dot', main: 'RUNNING', sub: elapsed ? 'elapsed ' + elapsed : '' };
    } else {
      const vc = str(run.verdict_class, 40);
      const vi = verdictInfo(vc);
      st = { tone: vi.tone, icon: vi.icon, main: 'FINISHED', sub: (vc ? words(vc) : words(run.reason || 'ended')) + (elapsed ? ' · ' + elapsed : ''),
        title: 'Reason: ' + words(run.reason || 'unknown') };
    }
    badge($('#badge-status'), st);
    $('#badge-status').classList.toggle('is-running', !fin);

    // ledger
    let lb;
    const anchor = str(led.head_anchor, 48);
    if (led.chain_ok === true) {
      lb = { tone: 'ok', icon: 'chain', main: 'chain consistent', sub: anchor ? 'head ' + anchor : '',
        title: 'The hash chain verifies: no edit, deletion or reordering in the middle. Truncation is detectable only against a head anchor recorded elsewhere. ' + fmtInt(led.events) + ' events.' };
    } else if (led.chain_ok === false) {
      lb = { tone: 'bad', icon: 'x', main: 'chain broken at seq ' + str(led.bad_seq, 12), sub: anchor ? 'head ' + anchor : '',
        title: 'The ledger hash chain does not verify. Do not trust this run.' };
    } else {
      lb = { tone: 'neutral', icon: 'dash', main: 'chain not verified', sub: '', title: 'This view was built without the raw events, so the hash chain was not checked.' };
    }
    badge($('#badge-ledger'), lb);

    // isolation
    const iso = str(run.isolation, 30);
    let ib;
    if (!iso) ib = { tone: 'neutral', icon: 'help', main: 'isolation unknown', sub: '', title: 'The run did not record which isolation mode was applied.' };
    else if (iso === 'none') ib = { tone: 'warn', icon: 'alert', main: 'isolation: none', sub: 'hash checks only', title: 'No namespace isolation was available. Protection rests on hash checks and detection.' };
    else if (iso === 'unshare-net') ib = { tone: 'warn', icon: 'alert', main: 'isolation: unshare-net', sub: 'labels readable',
      title: 'Network connections are blocked, but experiment code can still read the labels and the ledger by path, and a read is not detected. Run as root it can also read the API key unless the pid-namespace variant was in use. What is detected: a change to the trusted directory after the run. Not a security boundary.' };
    else ib = { tone: 'neutral', icon: 'box', main: 'isolation: ' + iso, sub: 'best effort', title: 'Best-effort containment of experiment code. Not a security boundary; tampering is detected by hashes.' };
    badge($('#badge-isolation'), ib);

    const resumes = objs(run.resumes);
    const rb = $('#badge-resumes');
    setHidden(rb, resumes.length === 0);
    if (resumes.length) badge(rb, { tone: 'neutral', icon: 'repeat', main: 'resumed ×' + resumes.length, sub: '', title: 'The run was resumed after an interruption; in-flight experiments were reconciled first.' });

    renderBudget(S);
  }

  // "REPLAY of r-123 (scripted)": the id and the origin of the recorded run, so a replay never hides what it replays.
  function replayOf(run, replay) {
    const src = str(replay && replay.source_run_id, 40) || str(run.run_id, 40);
    const origin = originOf(run);
    return 'of ' + (src ? src : 'a recorded run') + (origin ? ' (' + origin + ')' : '');
  }
  function implementerNote(impl) {
    if (impl === 'scripted') return 'The scripted suite wrote every patch; no model edited a file.';
    if (impl === 'patch') return 'A model edits files through the confined patch tools.';
    if (impl === 'muse_code') return 'The Muse Code CLI edits files; its sandbox forces one experiment at a time.';
    return impl ? 'Implementer recorded by the run: ' + impl : '';
  }

  // One plain line saying which models (and which endpoint) produced the run, so a custom host is never hidden.
  function renderModelsLine(run, hosts, ep) {
    const el = $('#models-line');
    const mode = str(run.mode, 20), origin = str(run.origin, 20) || mode;
    const models = isObj(run.models) ? run.models : {};
    // the configured model per role; when the charges show a different model answered (a fallback), say so
    const roles = Object.keys(models).sort().map((r) => {
      const ans = answeredModels(run, r);
      const differs = ans.length > 0 && (ans.length !== 1 || ans[0] !== str(models[r], 80));
      return str(r, 24) + ' ' + modelLabel(models[r]) + (differs ? (origin === 'simulated' ? ' (the simulator reported ' : ' (answered by ') + ans.map(modelLabel).join(', ') + ')' : '');
    });
    let text;
    if (origin === 'scripted') text = 'Models: none. Every role is scripted.';
    else if (origin === 'unknown') text = 'Models: ' + (roles.join(', ') || 'none recorded') + ' (labels only). The endpoint record contradicts the backend, so the source of this text is not established.';
    else if (origin === 'simulated') text = 'Models: ' + (roles.join(', ') || 'none recorded') + ' (labels only). Endpoint: local test server' + (hosts.length ? ' ' + hosts.join(', ') : '') + ', not Meta.';
    else {
      const kind = ep ? str(ep.kind, 20) : '';
      const where = kind === 'custom' ? ' Endpoint: custom host ' + (hosts.join(', ') || 'unknown') + '.'
        : kind === 'meta-api' ? ' Endpoint: ' + (hosts.join(', ') || 'Meta API') + '.' : '';
      text = 'Models: ' + (roles.join(', ') || 'none recorded') + '.' + where;
    }
    setText(el, text);
    setClass(el, cls('models-line', origin === 'simulated' ? 'is-simulated' : ''));
  }

  function renderBudget(S) {
    const host = $('#budget');
    const L = S.budget.limits, U = S.budget.used;
    const defs = [
      { key: 'experiments', label: 'Experiments', used: num(U.experiments), limit: num(L.max_experiments), f: fmtInt },
      { key: 'replications', label: 'Replications', used: num(U.replications), limit: num(L.max_replications), f: fmtInt },
      // a scripted run makes no model call: it has no spend to show, and "$0.00" would read as a measured zero
      { key: 'usd', label: originOf(S.run) === 'simulated' ? 'LLM spend (simulated list-price estimate)' : 'LLM spend', used: num(U.llm_usd !== undefined ? U.llm_usd : U.usd), limit: num(L.max_llm_usd), f: fmtUsd,
        none: originOf(S.run) === 'scripted' ? 'none · scripted' : '' },
      { key: 'wall', label: 'Wall time', used: num(U.wall_sec), limit: num(L.max_wall_sec), f: fmtDur },
    ];
    syncList(host, defs, (d) => d.key, (d) => {
      const n = h('div', { class: 'meter' });
      n._l = h('span', { class: 'meter-l' });
      n._v = h('span', { class: 'meter-v num' });
      n._bar = h('div', { class: 'meter-bar', role: 'progressbar', 'aria-valuemin': '0', 'aria-valuemax': '100' });
      n._fill = h('div', { class: 'meter-fill' });
      n._bar.append(n._fill);
      n.append(h('div', { class: 'meter-top' }, n._l, n._v), n._bar);
      return n;
    }, (n, d) => {
      const used = d.used === null ? 0 : d.used;
      const ratio = d.limit && d.limit > 0 ? used / d.limit : 0;
      const pct = Math.max(0, Math.min(100, Math.round(ratio * 100)));
      const tone = ratio >= 1 ? 'bad' : ratio >= 0.75 ? 'warn' : 'ok';
      setClass(n, cls('meter', 'tone-' + tone));
      setText(n._l, d.label);
      setText(n._v, d.none ? d.none : d.f(d.used === null ? 0 : d.used) + (d.limit !== null ? ' / ' + d.f(d.limit) : '') + ' · ' + pct + '%' + (ratio >= 1 ? ' (limit reached)' : ''));
      setAttr(n._bar, 'aria-valuenow', pct);
      setAttr(n._bar, 'aria-label', d.label + ' budget used');
      setAttr(n, 'title', d.key === 'usd' && originOf(S.run) === 'simulated' ? 'A ' + SIM_COST_LABEL + '. The simulator charged nobody.' : null);
      const w = pct + '%';
      if (n._fill.style.width !== w) n._fill.style.width = w;
    });
  }

  function renderBanner(S) {
    const el = $('#banner');
    const origin = str(S.run.origin, 20) || str(S.run.mode, 20);
    // A scripted or simulated run can never be shown without its banner, even if the producer forgot to send it.
    const fallback = origin === 'simulated' ? SIMULATED_BANNER : (str(S.run.backend) === 'scripted' || origin === 'scripted' ? SCRIPTED_BANNER : '');
    const text = S.banner || fallback;
    setHidden(el, !text);
    setText($('#banner-text'), text);
    el.classList.toggle('is-simulated', origin === 'simulated');
    el.classList.toggle('is-scripted', !!text && origin !== 'simulated');
  }

  // The red-team notice sits under the SCRIPTED / SIMULATED banner (the two can both apply). index.html has no slot for it, so the
  // element is built once, on first need, and from then on only shown or hidden; it has no dismiss control.
  function renderRedTeam(S) {
    const lanes = arr(S.run.red_team).filter((r) => isObj(r) && str(r.behavior, 40) !== '');
    let el = $('#red-team-notice');
    if (!el) {
      const anchor = $('#banner');
      if (!lanes.length || !anchor || !anchor.parentNode) return;
      el = h('div', { id: 'red-team-notice', class: 'banner is-redteam', role: 'note' }, icon('flag'),
        h('div', { class: 'banner-body' }, h('p', { id: 'red-team-text' }), h('p', { id: 'red-team-ran' }), h('p', { id: 'red-team-list', class: 'banner-sub' })));
      anchor.parentNode.insertBefore(el, anchor.nextSibling);
    }
    setHidden(el, !lanes.length);
    setText($('#red-team-text'), RED_TEAM_NOTICE.replace('{n}', String(lanes.length)));
    // how many of the planted lanes really ran: a lane the duplicate gate or the budget stopped attacked nothing
    const rt = S.fault_injection.red_team;
    const ranText = rt && num(rt.planted) > 0 ? str(rt.text, 400) : '';
    setText($('#red-team-ran'), ranText);
    setHidden($('#red-team-ran'), !ranText);
    setClass($('#red-team-ran'), cls('red-team-ran', rt && num(rt.not_run) > 0 ? 'is-short' : ''));
    const names = lanes.map((r) => str(r.behavior, 40) + (str(r.scripted_id, 40) ? ' (' + str(r.scripted_id, 40) + ')' : ''));
    setText($('#red-team-list'), names.length ? 'Planted: ' + names.join(', ') + '. These lanes are tagged PLANTED FAULT (scripted actor) below.' +
      (originOf(S.run) === 'simulated' ? ' ' + SIM_RED_TEAM_NOTE : '') : '');
  }

  function renderReplay(S) {
    const bar = $('#replay-bar');
    const r = S.replay;
    setHidden(bar, !r);
    if (!r) return;
    const pos = num(r.position_seq) || 0, tot = Math.max(num(r.total_seq) || 0, pos, 1);
    const speed = num(r.speed);
    setText($('#replay-text'), replayOf(S.run, r) + ' · event ' + fmtInt(pos) + ' / ' + fmtInt(tot) + (speed ? ' · ' + speed + '× speed' : ''));
    const pct = Math.round((pos / tot) * 100);
    const track = bar.querySelector('.replay-track');
    setAttr(track, 'aria-valuenow', pct);
    const fill = bar.querySelector('.replay-fill');
    const w = pct + '%';
    if (fill.style.width !== w) fill.style.width = w;
  }

  // ===========================================================================================
  // hero chart
  // ===========================================================================================
  function niceTicks(lo, hi, count) {
    const span = hi - lo;
    const raw = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    const ticks = [];
    // bounded: when step is below the float resolution at this magnitude, v += step would never advance
    const first = Math.ceil(lo / step);
    for (let k = 0; k <= 40 && (first + k) * step <= hi + step * 1e-9; k += 1) ticks.push((first + k) * step);
    return { ticks, step };
  }

  // Bounds beyond which a value is "off scale". Needs at least five values; below that nothing is clipped.
  function robustBounds(values) {
    const v = values.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
    if (v.length < 5) return { lo: -Infinity, hi: Infinity };
    const q = (f) => v[Math.min(v.length - 1, Math.max(0, Math.round(f * (v.length - 1))))];
    const q1 = q(0.25), q3 = q(0.75), med = q(0.5);
    const fence = 3 * (q3 - q1);
    let lo = q1 - fence, hi = q3 + fence;
    const core = v.filter((x) => x >= lo && x <= hi);
    const span = Math.max(core[core.length - 1] - core[0], 0.25 * Math.abs(med), 1e-9);
    // only a value several core-spans away counts as off scale
    lo = Math.min(lo, core[0] - 3 * span);
    hi = Math.max(hi, core[core.length - 1] + 3 * span);
    return { lo, hi };
  }

  function chartModel(S) {
    const byId = new Map(S.experiments.map((e) => [expId(e), e]));
    const pts = seriesByExp(S);
    const order = [];
    const seen = new Set();
    for (const e of S.experiments) { const id = expId(e); if (id && !seen.has(id)) { seen.add(id); order.push(id); } }
    for (const id of pts.keys()) { if (!seen.has(id)) { seen.add(id); order.push(id); } }
    const slots = [];
    let inc = null;
    for (const id of order) {
      const e = byId.get(id) || { exp_id: id };
      const p = pts.get(id) || null;
      const outcome = str(e.outcome, 24), status = str(e.status, 20);
      let kind;
      if (p) kind = p.kind;
      else if (status === 'blocked' && !outcome) kind = 'blocked';
      else if (['failed', 'invalid', 'infra_error', 'unresolved'].includes(outcome) || status === 'violation') kind = 'failed';
      else if (status === 'blocked') kind = 'blocked';
      else if (['recorded'].includes(status)) kind = 'failed';
      else kind = 'pending';
      const hollow = !p || ['failed', 'invalid', 'infra_error'].includes(outcome);
      if (p && (p.kind === 'baseline' || p.kind === 'promoted') && p.score !== null) inc = p.score;
      slots.push({ id, e, p, kind, hollow, inc, overfit: overfitSuspect(e), outcome });
    }
    const bs = slots.find((x) => x.p && x.p.kind === 'baseline');
    return { slots, objective: str(S.run.objective, 12) || 'minimize', metric: str(S.run.primary_metric, 40),
      baseline: num(S.baseline.score) !== null ? num(S.baseline.score) : (bs ? bs.p.score : null) };
  }

  function shapeNodes(kind, hollow, overfit, off) {
    const out = [];
    if (off) out.push(s('path', { class: 'sh sh-off', d: off === 'lo' ? 'M0 7 L7.5 -5.5 L-7.5 -5.5 Z' : 'M0 -7 L7.5 5.5 L-7.5 5.5 Z' }));
    else if (kind === 'baseline') out.push(s('rect', { class: 'sh sh-baseline', x: -5, y: -5, width: 10, height: 10, rx: 1.5 }));
    else if (kind === 'promoted' && !hollow) out.push(s('path', { class: 'sh sh-promoted', d: 'M0 -8.5 L8.5 0 L0 8.5 L-8.5 0 Z' }));
    else if (kind === 'blocked') {
      out.push(s('circle', { class: 'sh sh-blocked', r: 5.5 }), s('path', { class: 'sh-mark', d: 'M-3 0H3' }));
    } else if (kind === 'pending') {
      out.push(s('circle', { class: 'sh sh-pending', r: 5.5 }));
    } else if (hollow) {
      out.push(s('circle', { class: 'sh sh-failed', r: 5.5 }), s('path', { class: 'sh-mark sh-mark-x', d: 'M-2.8 -2.8L2.8 2.8M2.8 -2.8L-2.8 2.8' }));
    } else {
      out.push(s('circle', { class: 'sh sh-candidate', r: 5.5 }));
    }
    if (overfit) out.push(s('path', { class: 'sh-warn', d: 'M0 -23 L6.5 -11.5 L-6.5 -11.5 Z' }), s('path', { class: 'sh-warn-mark', d: 'M0 -20v4.2M0 -13.3v.4' }));
    return out;
  }

  function slotLabel(sl) {
    const parts = [sl.id];
    const o = outcomeInfo(sl.e);
    parts.push(o.text);
    if (sl.p && sl.p.score !== null) parts.push('dev score ' + fmtNum(sl.p.score));
    else parts.push('no score');
    const rep = replicationInfo(sl.e);
    if (rep) parts.push(rep.text);
    if (sl.off) parts.push('off the chart scale, drawn on the edge');
    if (sl.overfit) parts.push('overfit suspect: train improved but held-out did not');
    return parts.join(', ');
  }

  function chartBounds(model) {
    const all = [];
    for (const sl of model.slots) {
      if (!sl.p) continue;
      if (sl.p.score !== null) all.push(sl.p.score);
      if (sl.p.train !== null) all.push(sl.p.train);
    }
    if (num(model.baseline) !== null) all.push(model.baseline);
    return robustBounds(all);
  }

  function buildChart(model, W, H) {
    const n = model.slots.length;
    const small = W < 520;
    // Scale to the bulk of the data: one runaway score (an overfit run can be 300x the baseline) must not flatten the rest.
    // A value far outside the interquartile fence is drawn on the plot edge and labelled with its true number.
    const bounds = chartBounds(model);
    const inCore = (v) => v >= bounds.lo && v <= bounds.hi;
    const anyHi = model.slots.some((x) => x.p && x.p.score !== null && x.p.score > bounds.hi);
    const m = { l: small ? 46 : 58, r: 16, t: anyHi ? 56 : 32, b: 46 };
    const hasRail = model.slots.some((x) => !x.p);
    const railH = hasRail ? 26 : 0;
    const x0 = m.l, x1 = W - m.r;
    const yTop = m.t, yBot = H - m.b - (hasRail ? railH + 8 : 0);
    const slotW = (x1 - x0) / n;
    const X = (i) => x0 + slotW * (i + 0.5);

    const vals = [];
    for (const sl of model.slots) {
      if (!sl.p) continue;
      const sd = sl.p.std || 0;
      sl.off = sl.p.score === null || inCore(sl.p.score) ? null : (sl.p.score > bounds.hi ? 'hi' : 'lo');
      if (sl.p.score !== null && !sl.off) vals.push(sl.p.score - sd, sl.p.score + sd);
      if (sl.p.train !== null && inCore(sl.p.train)) vals.push(sl.p.train);
    }
    if (num(model.baseline) !== null) vals.push(model.baseline);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) { lo = 0; hi = 1; }
    if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
    const padv = (hi - lo) * 0.08;
    lo -= padv; hi += padv;
    const Y = (v) => yBot - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (yBot - yTop);
    const { ticks, step } = niceTicks(lo, hi, small ? 4 : 5);
    const dec = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));

    const svg = s('svg', { class: 'chart', viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'group',
      'aria-label': 'Chart of held-out score per experiment. ' + n + ' experiments, ' + (model.objective === 'maximize' ? 'higher' : 'lower') + ' is better.' });

    // grid + y ticks
    const g0 = s('g', { class: 'chart-grid' });
    for (const t of ticks) {
      g0.append(s('line', { class: 'grid-line', x1: x0, x2: x1, y1: Y(t), y2: Y(t) }));
      const lab = s('text', { class: 'tick', x: x0 - 8, y: Y(t) + 4, 'text-anchor': 'end' });
      lab.textContent = t.toFixed(dec);
      g0.append(lab);
    }
    g0.append(s('line', { class: 'axis-line', x1: x0, x2: x1, y1: yBot, y2: yBot }));
    svg.append(g0);

    const mt = s('text', { class: 'axis-title', x: x0, y: 14 });
    mt.textContent = (model.metric || 'score') + ' (held-out, dev split)';
    svg.append(mt);

    // rail for experiments without a score
    if (hasRail) {
      const ry = yBot + 8;
      svg.append(s('rect', { class: 'noscore-rail', x: x0, y: ry, width: x1 - x0, height: railH, rx: 4 }));
      const rl = s('text', { class: 'rail-label', x: x0 + 6, y: ry + 4 + railH / 2 - 2 });
      rl.textContent = 'no score';
      svg.append(rl);
    }

    // x labels
    const labelEvery = slotW >= 34 ? 1 : slotW >= 18 ? 2 : 4;
    const xy = H - 12;
    model.slots.forEach((sl, i) => {
      if (i % labelEvery !== 0) return;
      const t = s('text', { class: 'tick tick-x', x: X(i), y: xy, 'text-anchor': 'middle' });
      t.textContent = slotW >= 34 ? sl.id : (sl.id.replace(/^e0*/, '') || '0');
      svg.append(t);
    });

    // baseline reference
    const base = num(model.baseline);
    if (base !== null) {
      svg.append(s('line', { class: 'baseline-line', x1: x0, x2: x1, y1: Y(base), y2: Y(base) }));
      const bl = s('text', { class: 'ref-label', x: x1 - 4, y: Y(base) - 5, 'text-anchor': 'end' });
      bl.textContent = 'baseline ' + fmtNum(base);
      svg.append(bl);
    }

    // train (dashed)
    let d = '';
    model.slots.forEach((sl, i) => {
      if (sl.p && sl.p.train !== null) d += (d ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(sl.p.train).toFixed(1);
    });
    if (d) svg.append(s('path', { class: 'train-line', d }));

    // incumbent step line
    let incD = '';
    let prevInc = null;
    model.slots.forEach((sl, i) => {
      if (sl.inc === null) return;
      if (prevInc === null) incD = 'M' + x0.toFixed(1) + ' ' + Y(sl.inc).toFixed(1);
      else if (sl.inc !== prevInc) incD += 'H' + X(i).toFixed(1) + 'V' + Y(sl.inc).toFixed(1);
      prevInc = sl.inc;
    });
    if (incD) svg.append(s('path', { class: 'inc-line', d: incD + 'H' + x1.toFixed(1) }));

    // error bars
    model.slots.forEach((sl, i) => {
      if (!sl.p || sl.p.score === null || !sl.p.std || sl.p.std <= 0 || sl.off) return;
      const yA = Y(sl.p.score - sl.p.std), yB = Y(sl.p.score + sl.p.std), x = X(i);
      svg.append(s('path', { class: 'err', d: 'M' + x.toFixed(1) + ' ' + yA.toFixed(1) + 'V' + yB.toFixed(1) +
        'M' + (x - 4).toFixed(1) + ' ' + yA.toFixed(1) + 'h8M' + (x - 4).toFixed(1) + ' ' + yB.toFixed(1) + 'h8' }));
    });

    // markers
    const railY = yBot + 8 + railH / 2;
    const markers = s('g', { class: 'markers' });
    model.slots.forEach((sl, i) => {
      const x = X(i);
      const y = sl.p && sl.p.score !== null ? Y(sl.p.score) : railY;
      const g = s('g', { class: 'pt js-exp', transform: 'translate(' + x.toFixed(1) + ' ' + y.toFixed(1) + ')', tabindex: '0',
        role: 'button', 'aria-pressed': 'false', 'aria-label': slotLabel(sl), 'data-act': 'sel-exp', 'data-arg': sl.id });
      g._exps = [sl.id];
      g._slot = sl; g._x = x; g._y = y;
      g.append(s('circle', { class: 'hit', r: 15 }), s('circle', { class: 'sel-ring', r: 12 }), s('circle', { class: 'focus-ring', r: 14 }));
      for (const nd of shapeNodes(sl.kind, sl.hollow, sl.overfit, sl.off)) g.append(nd);
      if (sl.off) {
        const lab = s('text', { class: 'off-label', x: x > x1 - 110 ? -10 : 10, y: sl.off === 'hi' ? -4 : 16, 'text-anchor': x > x1 - 110 ? 'end' : 'start' });
        lab.textContent = fmtNum(sl.p.score) + ' (off scale)';
        g.append(lab);
      }
      markers.append(g);
    });
    svg.append(markers);
    return svg;
  }

  function tipLines(sl) {
    const e = sl.e, lines = [];
    const o = outcomeInfo(e);
    lines.push({ cls: 'tip-title', text: sl.id + (str(e.lane) ? ' · lane ' + str(e.lane, 20) : '') + ' · ' + o.text });
    const hyp = str(e.hypothesis, 400);
    if (hyp) lines.push({ cls: 'tip-hyp', text: clip(hyp, 150) });
    if (sl.p && sl.p.score !== null) {
      lines.push({ cls: 'tip-row', text: 'dev score ' + fmtNum(sl.p.score) + (sl.p.std ? ' ± ' + fmtNum(sl.p.std) : '') });
      if (sl.p.train !== null) lines.push({ cls: 'tip-row', text: 'train ' + fmtNum(sl.p.train) });
    } else lines.push({ cls: 'tip-row', text: 'no valid score' });
    if (num(e.delta) !== null) lines.push({ cls: 'tip-row', text: 'gain vs base ' + fmtSigned(e.delta) });
    if (num(e.cost_usd) !== null) lines.push({ cls: 'tip-row', text: (ui.sim ? 'simulated model cost ' : 'model cost ') + fmtUsd(e.cost_usd) });
    const rep = replicationInfo(e);
    if (rep) lines.push({ cls: 'tip-row', text: rep.text + (num(e.sealed_delta) !== null ? ' (sealed gain ' + fmtSigned(e.sealed_delta) + ')' : '') });
    if (sl.overfit) lines.push({ cls: 'tip-row tip-warn', text: 'overfit suspect: train improved, held-out did not' });
    if (str(e.planted_fault)) lines.push({ cls: 'tip-row', text: 'PLANTED FAULT (scripted actor): ' + str(e.planted_fault, 30) });
    return lines;
  }

  function showTip(g) {
    const tip = $('#chart-tip'), wrap = $('#chart-wrap');
    const lines = tipLines(g._slot);
    tip.replaceChildren(...lines.map((l) => h('div', { class: l.cls, text: l.text })));
    setHidden(tip, false);
    const w = wrap.clientWidth, tw = tip.offsetWidth, th = tip.offsetHeight;
    let left = g._x - tw / 2;
    left = Math.max(4, Math.min(left, w - tw - 4));
    let top = g._y - th - 22;
    if (top < 4) top = g._y + 20;
    tip.style.left = Math.round(left) + 'px';
    tip.style.top = Math.round(top) + 'px';
  }
  function hideTip() { setHidden($('#chart-tip'), true); }

  function renderChart(S) {
    const wrap = $('#chart-wrap');
    const model = chartModel(S);
    ui.chartModel = model;
    const scored = model.slots.filter((x) => x.p && x.p.score !== null).length;
    const dir = model.objective === 'maximize' ? 'higher is better' : 'lower is better';
    const dirEl = $('#chart-dir');
    if (!dirEl._init) { dirEl._ic = h('span', { class: 'dir-ic' }); dirEl._tx = h('span'); dirEl.append(dirEl._ic, dirEl._tx); dirEl._init = true; }
    setIcon(dirEl._ic, model.objective === 'maximize' ? 'up' : 'down');
    setText(dirEl._tx, dir);
    setText($('#chart-sub'), model.slots.length + ' experiments in proposal order, ' + scored + ' with a valid dev score.');

    const empty = $('#chart-empty');
    if (!scored) {
      setEmpty(empty, 'No graded experiments yet', 'The baseline appears here as soon as the trusted grader has scored it. Hollow markers show experiments that never produced a score.', true);
      const host = wrap.querySelector('.chart-host');
      if (host) host.remove();
      ui.chartSig = '';
      renderLegend(model, false);
      renderDetail();
      return;
    }
    setEmpty(empty, '', '', false);

    const W = Math.max(280, Math.floor(wrap.clientWidth || ui.chartW || 640));
    const H = W < 480 ? 270 : W < 760 ? 320 : W < 1000 ? 380 : 450;
    ui.chartW = W;
    const sig = JSON.stringify([W, H, model.objective, model.metric, model.baseline, model.slots.map((x) => [x.id, x.kind, x.hollow, x.overfit, x.inc, x.p && [x.p.score, x.p.train, x.p.std], x.outcome, x.e.status, x.e.replicated, x.e.awaiting_replication, x.e.dev_outcome, x.e.sealed_delta, x.e.delta, str(x.e.hypothesis, 160), x.e.planted_fault])]);
    if (sig !== ui.chartSig) {
      ui.chartSig = sig;
      const active = document.activeElement;
      const refocus = active && active.classList && active.classList.contains('pt') && wrap.contains(active) ? active.getAttribute('data-arg') : null;
      let host = wrap.querySelector('.chart-host');
      if (!host) { host = h('div', { class: 'chart-host' }); wrap.insertBefore(host, wrap.firstChild); }
      const svg = buildChart(model, W, H);
      host.replaceChildren(svg);
      if (refocus) {
        for (const g of svg.querySelectorAll('.pt')) { if (g.getAttribute('data-arg') === refocus) { g.focus({ preventScroll: true }); break; } }
      }
      hideTip();
    }
    renderLegend(model, true);
    renderDetail();
  }

  function renderLegend(model, show) {
    const ul = $('#chart-legend');
    setHidden(ul, !show);
    if (!show) return;
    const bnd = chartBounds(model);
    const anyOff = model.slots.some((x) => x.p && x.p.score !== null && (x.p.score > bnd.hi || x.p.score < bnd.lo));
    const items = [
      { key: 'baseline', label: 'Baseline', draw: () => shapeNodes('baseline', false, false) },
      { key: 'candidate', label: 'Candidate', draw: () => shapeNodes('candidate', false, false) },
      { key: 'promoted', label: 'Promoted (replicated)', draw: () => shapeNodes('promoted', false, false) },
      { key: 'failed', label: 'Failed, invalid or blocked (hollow)', draw: () => shapeNodes('failed', true, false) },
      { key: 'pending', label: 'In progress', draw: () => shapeNodes('pending', true, false) },
      { key: 'off', label: 'Off the scale (true value labelled)', draw: () => shapeNodes('candidate', false, false, 'hi') },
      { key: 'overfit', label: 'Overfit suspect', tall: true, draw: () => shapeNodes('candidate', false, true) },
      { key: 'inc', label: 'Incumbent', line: 'inc-line' },
      { key: 'base', label: 'Baseline score', line: 'baseline-line' },
      { key: 'train', label: 'Train score', line: 'train-line' },
      { key: 'err', label: '±1 std over seeds', draw: () => [s('path', { class: 'err', d: 'M0 -8V8M-4 -8h8M-4 8h8' })] },
    ].filter((i) => i.key !== 'off' || anyOff);
    syncList(ul, items, (i) => i.key, (i) => {
      const li = h('li', { class: 'legend-i' });
      const svg = i.tall
        ? s('svg', { class: 'legend-sw', viewBox: '-14 -26 28 40', width: 22, height: 31, 'aria-hidden': 'true', focusable: 'false' })
        : s('svg', { class: 'legend-sw', viewBox: '-14 -14 28 28', width: 22, height: 22, 'aria-hidden': 'true', focusable: 'false' });
      if (i.line) svg.append(s('line', { class: i.line, x1: -13, x2: 13, y1: 0, y2: 0 }));
      else for (const nd of i.draw()) svg.append(nd);
      li.append(svg, h('span', { text: i.label }));
      return li;
    }, () => {});
  }

  function renderDetail() {
    const host = $('#chart-detail');
    if (!host._init) {
      host._hint = h('p', { class: 'cd-hint' }, icon('arrow'), h('span', { text: 'Select a marker or an experiment card to inspect it. Selection is highlighted in every panel.' }));
      host._id = h('span', { class: 'mono cd-id' });
      host._pill = makeTag({}, 'pill');
      host._lane = h('span', { class: 'chip' });
      host._rep = makeTag({}, 'pill');
      host._clear = h('button', { class: 'btn btn-sm', type: 'button', 'data-act': 'clear-sel', text: 'Clear selection' });
      host._goto = h('button', { class: 'btn btn-sm', type: 'button', 'data-act': 'goto-card', text: 'Show experiment card' });
      host._hyp = h('p', { class: 'cd-hyp' });
      host._m = h('dl', { class: 'cd-metrics' });
      host._mv = {};
      host._mw = {};
      host._md = {};
      for (const k of ['Dev score', 'Train score', 'Gain (dev)', 'Gain (sealed)', 'Model cost']) {
        const dd = h('dd', { class: 'num' });
        host._mv[k] = dd;
        host._md[k] = h('dt', { text: k });
        host._mw[k] = h('div', {}, host._md[k], dd);
        host._m.append(host._mw[k]);
      }
      host._body = h('div', { class: 'cd-body' }, h('div', { class: 'cd-head' }, host._id, host._pill, host._rep, host._lane, h('span', { class: 'cd-spacer' }), host._goto, host._clear), host._hyp, host._m);
      host.append(host._hint, host._body);
      host._init = true;
    }
    const sl = ui.selExp && ui.chartModel ? ui.chartModel.slots.find((x) => x.id === ui.selExp) : null;
    setHidden(host._hint, !!sl);
    setHidden(host._body, !sl);
    if (!sl) return;
    const e = sl.e, o = outcomeInfo(e), rep = replicationInfo(e);
    setText(host._id, sl.id);
    patchTag(host._pill, { tone: o.tone, icon: o.icon, text: o.text });
    patchTag(host._rep, rep ? { tone: rep.tone, icon: rep.icon, text: rep.text, title: rep.title } : { tone: 'neutral', text: '' });
    setHidden(host._rep, !rep);
    setText(host._lane, 'lane ' + (str(e.lane, 20) || '-'));
    setText(host._hyp, str(e.hypothesis) || 'No hypothesis recorded.');
    const p = sl.p;
    setText(host._mv['Dev score'], p && p.score !== null ? fmtNum(p.score) + (p.std ? ' ± ' + fmtNum(p.std) : '') : '—');
    setText(host._mv['Train score'], p ? fmtNum(p.train) : '—');
    setText(host._mv['Gain (dev)'], fmtSigned(e.delta));
    setText(host._mv['Gain (sealed)'], fmtSigned(e.sealed_delta));
    const cf = expCost(e);
    setHidden(host._mw['Model cost'], cf === null);
    setText(host._md['Model cost'], ui.sim ? 'Model cost (simulated)' : 'Model cost');
    if (cf !== null) setText(host._mv['Model cost'], fmtUsd(cf.usd));
  }

  // ===========================================================================================
  // headline + verdict
  // ===========================================================================================
  function verdictInfo(vc) {
    switch (vc) {
      case 'success': return { tone: 'ok', icon: 'check', title: 'Accepted improvement', text: 'A replicated improvement passed the finish gate on the sealed split.' };
      case 'inconclusive': return { tone: 'neutral', icon: 'dash', title: 'Honestly inconclusive', text: 'No improvement survived replication. Reporting that is a result, not a failure.' };
      case 'unresolved_defects': return { tone: 'bad', icon: 'x', title: 'Unresolved defects', text: 'The run ended with open blocking findings or an unresolved experiment.' };
      case 'missing_authority': return { tone: 'warn', icon: 'alert', title: 'Stopped without authority to continue', text: 'The budget or a HALT ended the run before a finish was accepted.' };
      default: return { tone: 'neutral', icon: 'help', title: vc ? words(vc) : 'No verdict', text: 'The run did not report a verdict class.' };
    }
  }

  // Cost facts of the run, tolerant of states written before the cost keys existed. A scripted run has none: known is false.
  function costFacts(S) {
    const H = S.headline, U = S.budget.used;
    const calls = num(H.llm_calls) !== null ? num(H.llm_calls) : num(U.llm_calls);
    const total = num(H.cost_usd);
    const known = typeof H.cost_known === 'boolean' ? H.cost_known : ((calls || 0) > 0 || (total || 0) > 0);
    return { known, calls, total, attributed: num(H.cost_attributed_usd), unattributed: num(H.cost_unattributed_usd),
      promoted: num(H.promoted), perImprovement: known ? num(H.cost_per_improvement_usd) : null };
  }

  function renderHeadline(S) {
    const H = S.headline, par = S.parallel;
    const wall = num(par.wall_sec), serial = num(par.serial_sec), sp = num(H.speedup) !== null ? num(H.speedup) : num(par.speedup);
    const cost = costFacts(S);
    const sim = originOf(S.run) === 'simulated';
    const lanes = num(par.max_parallel);
    const reason = str(par.reason, 300);
    const speedSub = [];
    if (wall !== null && serial !== null) speedSub.push('wall ' + fmtDur(wall) + ' vs serial ' + fmtDur(serial));
    if (lanes !== null) speedSub.push('up to ' + fmtInt(lanes) + (lanes === 1 ? ' lane' : ' lanes') + ' at once');
    const costSub = cost.calls !== null ? [fmtInt(cost.calls) + (cost.calls === 1 ? ' model call' : ' model calls')] : [];
    if (cost.attributed !== null && cost.unattributed !== null) {
      costSub.push(fmtUsd(cost.attributed) + ' tied to experiments, ' + fmtUsd(cost.unattributed) + ' to the supervisor, auditor and others');
    }
    if (sim) costSub.unshift(SIM_COST_LABEL);
    const items = [
      { key: 'made', label: 'Claims made', value: fmtInt(H.claims_made), tone: '' },
      { key: 'acc', label: 'Accepted', value: fmtInt(H.claims_accepted), tone: 'ok', icon: 'check' },
      { key: 'ref', label: 'Refused', value: fmtInt(H.claims_refused), tone: num(H.claims_refused) > 0 ? 'bad' : '', icon: num(H.claims_refused) > 0 ? 'x' : '' },
      { key: 'unrep', label: 'Unreplicated improvements blocked', value: fmtInt(H.unreplicated_blocked), tone: num(H.unreplicated_blocked) > 0 ? 'warn' : '', wide: true, icon: num(H.unreplicated_blocked) > 0 ? 'ban' : '',
        sub: 'Improved on dev, never confirmed on the sealed split, so never reported as a result.' },
      { key: 'speed', label: 'Parallel speedup', value: sp === null ? '—' : sp.toFixed(2) + '×', tone: '', md: cost.known, wide: !cost.known,
        sub: speedSub.join(' · '), note: reason, noteTitle: 'Why parallelism was limited' },
    ];
    // no cost card at all unless the ledger recorded a model call: a scripted run shows nothing, never $0
    if (cost.known) {
      items.push({ key: 'cost', label: sim ? 'Simulated LLM cost' : 'LLM cost', value: fmtUsd(cost.total), tone: '', md: true, icon: 'coin', sub: costSub.join(' · ') });
      if (cost.perImprovement !== null) {
        items.push({ key: 'cpi', label: (sim ? 'Simulated cost' : 'Cost') + ' per promoted improvement', value: fmtUsd(cost.perImprovement), tone: '', wide: true, icon: 'coin',
          sub: (sim ? SIM_COST_LABEL + '. ' : '') + 'Whole-run spend ' + fmtUsd(cost.total) + ' divided by ' + fmtInt(cost.promoted) + ' ' + (cost.promoted === 1 ? 'improvement' : 'improvements') + ' that passed the sealed replication.' });
      }
    }
    syncList($('#headline'), items, (i) => i.key, (i) => {
      const n = h('div', { class: 'stat' });
      n._v = h('div', { class: 'stat-v num' });
      n._ic = h('span', { class: 'stat-ic' });
      n._l = h('div', { class: 'stat-l' });
      n._sub = h('div', { class: 'stat-sub' });
      n._note = h('div', { class: 'stat-note' });
      n.append(h('div', { class: 'stat-top' }, n._ic, n._v), n._l, n._sub, n._note);
      return n;
    }, (n, i) => {
      setClass(n, cls('stat', i.wide ? 'stat-wide' : '', i.md ? 'stat-md' : '', i.tone ? 'tone-' + i.tone : ''));
      setText(n._v, i.value);
      setText(n._l, i.label);
      setIcon(n._ic, i.icon || '');
      setHidden(n._ic, !i.icon);
      setText(n._sub, i.sub || '');
      setHidden(n._sub, !i.sub);
      setText(n._note, i.note || '');
      setHidden(n._note, !i.note);
      setAttr(n._note, 'title', i.note ? i.noteTitle : null);
    });
  }

  function renderVerdict(S) {
    const host = $('#verdict');
    if (!host._init) {
      host._empty = h('div', { class: 'empty empty-sm' });
      host._pill = makeTag({}, 'pill pill-lg');
      host._reason = h('p', { class: 'v-reason' });
      host._by = h('p', { class: 'v-summary-by' });
      host._sum = h('p', { class: 'v-summary' });
      host._h = h('h3', { class: 'sub-h', text: 'Claims the finish gate accepted' });
      host._claims = h('ul', { class: 'v-claims' });
      host._body = h('div', { class: 'v-body' }, host._pill, host._reason, host._by, host._sum, host._h, host._claims);
      host.append(host._empty, host._body);
      host._init = true;
    }
    const run = S.run;
    const fin = run.finished === true;
    setEmpty(host._empty, 'No verdict yet', 'The verdict appears when the supervisor requests to finish and the finish gate accepts it. A refused finish does not end the run.', !fin);
    setHidden(host._body, !fin);
    if (!fin) return;
    const vc = str(run.verdict_class, 40), vi = verdictInfo(vc);
    patchTag(host._pill, { tone: vi.tone, icon: vi.icon, text: vi.title });
    setText(host._reason, vi.text + ' Ended by: ' + words(run.reason || 'unknown') + '.');
    let summary = str(run.summary, 1200);
    if (!summary) {
      const reqs = S.finish.requests;
      if (reqs.length) summary = str(reqs[reqs.length - 1].summary, 1200);
    }
    setText(host._sum, summary);
    setHidden(host._sum, !summary);
    // a summary written against the simulator is the scripted stand-in's, whatever the verdict beside it says
    setText(host._by, ui.sim ? 'Final summary, ' + SIM_SUPERVISOR_LABEL + '.' : '');
    setHidden(host._by, !(ui.sim && summary));
    // the claims the finish gate accepted; an older state without that list falls back to the supported ones
    const acc = S.finish.acceptedClaims;
    const supported = acc ? S.claims.filter((c) => acc.indexOf(str(c.claim_id, 60)) >= 0) : S.claims.filter((c) => str(c.status) === 'supported');
    setHidden(host._h, supported.length === 0);
    syncList(host._claims, supported, (c) => str(c.claim_id), () => h('li', { class: 'v-claim' }, h('span', { class: 'mono v-cid' }), h('span', { class: 'v-ctext' })), (li, c) => {
      setText(li.children[0], str(c.claim_id, 40));
      setText(li.children[1], str(c.text, 400));
    });
  }

  // ===========================================================================================
  // experiment lanes
  // ===========================================================================================
  function stepStates(e) {
    const status = str(e.status, 20), outcome = str(e.outcome, 24);
    const hasScore = num(e.score) !== null, hasReview = !!str(e.review);
    const protectedHit = arr(e.protected).length > 0;
    const st = ['pending', 'pending', 'pending', 'pending', 'pending', 'pending'];
    if (status === 'recorded' || status === 'blocked' || status === 'violation') {
      st[0] = 'done';
      if (status === 'blocked' && !outcome) {
        // stopped by the engine before it ran (for example the duplicate gate): not a failure of the experiment's own work
        st[1] = st[2] = st[3] = st[4] = st[5] = 'skipped';
        return st;
      }
      if (protectedHit || status === 'violation') {
        st[1] = 'failed'; st[2] = st[3] = st[4] = 'skipped'; st[5] = status === 'recorded' ? 'done' : 'skipped';
        return st;
      }
      st[1] = 'done';
      if (outcome === 'failed' && !hasScore) { st[2] = 'failed'; st[3] = st[4] = 'skipped'; }
      else if (outcome === 'infra_error') { st[2] = 'failed'; st[3] = st[4] = 'skipped'; }
      else if (outcome === 'invalid' && !hasScore) { st[2] = 'done'; st[3] = 'failed'; st[4] = 'skipped'; }
      else {
        st[2] = 'done';
        st[3] = hasScore ? 'done' : 'skipped';
        st[4] = hasReview ? 'done' : 'skipped';
      }
      st[5] = status === 'recorded' ? 'done' : 'skipped';
      return st;
    }
    let active = own(STATUS_ACTIVE, status);
    if (active === undefined) active = own(STAGE_INDEX, str(e.stage, 20));
    if (active === undefined) active = 0;
    for (let i = 0; i < 6; i += 1) st[i] = i < active ? 'done' : i === active ? 'active' : 'pending';
    return st;
  }
  const STEP_ICON = { done: 'check', active: 'clock', failed: 'x', skipped: 'dash', pending: '' };
  const STEP_TEXT = { done: 'done', active: 'in progress', failed: 'failed', skipped: 'skipped', pending: 'pending' };

  function makeCard() {
    const n = h('article', { class: 'lane-card js-exp' });
    const r = n._r = {};
    r.id = h('span', { class: 'lc-id mono' });
    r.lane = h('span', { class: 'lc-lane' });
    r.sel = h('button', { class: 'lc-select', type: 'button', 'data-act': 'sel-exp', 'aria-pressed': 'false' }, r.id, r.lane);
    r.outcome = makeTag({}, 'pill');
    r.hyp = h('p', { class: 'lc-hyp' });
    r.more = h('button', { class: 'link-btn', type: 'button', 'data-act': 'toggle-more' });
    r.sub = h('p', { class: 'lc-sub' });
    r.steps = STAGES.map(([, label]) => {
      const ic = h('span', { class: 'step-ic' });
      const st = h('span', { class: 'visually-hidden' });
      const li = h('li', { class: 'step' }, ic, h('span', { class: 'step-name', text: label }), st);
      li._ic = ic; li._st = st; li._label = label;
      return li;
    });
    r.pipe = h('ol', { class: 'pipeline', 'aria-label': 'Stage pipeline' }, r.steps);
    r.metrics = {};
    r.mgrid = h('dl', { class: 'lc-metrics' });
    for (const [k, t] of [['dev', 'Dev score'], ['train', 'Train'], ['gain', 'Gain (dev)'], ['sealed', 'Gain (sealed)']]) {
      r.metrics[k] = h('dd', { class: 'num' });
      const dt = h('dt', { text: t });
      if (k === 'gain' || k === 'sealed') dt.setAttribute('title', 'Positive means better than the incumbent, whichever way the objective points.');
      r.mgrid.append(h('div', {}, dt, r.metrics[k]));
    }
    r.cost = h('p', { class: 'lc-cost' });
    r.costL = h('span', { class: 'lc-cost-l', text: 'Cost' });     // patchCard says "Cost (simulated)" for a simulated run
    r.costV = h('span', { class: 'lc-cost-v num' });
    r.costD = h('span', { class: 'lc-cost-d' });
    r.cost.append(r.costL, r.costV, r.costD);
    r.tags = h('div', { class: 'lc-tags' });
    r.finds = h('div', { class: 'lc-finds' });
    n.append(h('div', { class: 'lc-head' }, r.sel, r.outcome), h('div', { class: 'lc-hyp-wrap' }, r.hyp, r.more), r.sub, r.pipe, r.mgrid, r.cost, r.tags, r.finds);
    return n;
  }

  function patchCard(n, e, _i, ctx) {
    const r = n._r, id = expId(e);
    n._exps = [id];
    setText(r.id, id);
    setText(r.lane, 'lane ' + (str(e.lane, 20) || '-'));
    setAttr(r.sel, 'data-arg', id);
    setAttr(r.sel, 'aria-label', 'Select experiment ' + id);
    const o = outcomeInfo(e);
    patchTag(r.outcome, o);
    const hyp = str(e.hypothesis) || 'No hypothesis recorded.';
    setText(r.hyp, hyp);
    const long = hyp.length > 170;
    const open = ui.expanded.has(id);
    setClass(r.hyp, cls('lc-hyp', long && !open ? 'is-clamped' : ''));
    setHidden(r.more, !long);
    setText(r.more, open ? 'Show less' : 'Show more');
    setAttr(r.more, 'data-arg', id);
    setAttr(r.more, 'aria-expanded', open ? 'true' : 'false');
    const sub = [];
    if (str(e.diff_summary)) sub.push(str(e.diff_summary, 160));
    else if (str(e.parent_exp_id)) sub.push('builds on ' + str(e.parent_exp_id, 20));
    setText(r.sub, sub.join(' · '));
    setHidden(r.sub, sub.length === 0);

    const states = stepStates(e);
    r.steps.forEach((li, i) => {
      setClass(li, 'step is-' + states[i]);
      setIcon(li._ic, STEP_ICON[states[i]]);
      setText(li._st, ': ' + STEP_TEXT[states[i]]);
    });

    const p = ctx.series.get(id);
    setText(r.metrics.dev, num(e.score) !== null ? fmtNum(e.score) + (p && p.std ? ' ± ' + fmtNum(p.std) : '') : '—');
    setText(r.metrics.train, fmtNum(e.train));
    setText(r.metrics.gain, fmtSigned(e.delta));
    setText(r.metrics.sealed, fmtSigned(e.sealed_delta));

    // cost only where the ledger tied model calls to this experiment; unknown is hidden, never $0
    const cf = expCost(e);
    setHidden(r.cost, cf === null);
    if (cf !== null) {
      setText(r.costL, ui.sim ? 'Cost (simulated)' : 'Cost');
      setText(r.costV, fmtUsd(cf.usd));
      setText(r.costD, cf.detail);
      setAttr(r.cost, 'title', cf.title);
    }

    const tags = [];
    const rep = replicationInfo(e);
    if (rep) tags.push({ key: 'rep', tone: rep.tone, icon: rep.icon, text: rep.text, title: rep.title });
    if (str(e.planted_fault)) tags.push({ key: 'planted', tone: 'scripted', icon: 'flag', text: 'PLANTED FAULT (scripted actor)', title: 'Fault kind: ' + str(e.planted_fault, 40) });
    if (str(e.planted_fault)) tags.push({ key: 'fault', tone: 'scripted', text: str(e.planted_fault, 40), mono: true, title: 'Planted fault kind' });
    if (str(e.blocked_reason)) tags.push({ key: 'blocked', tone: 'warn', icon: 'ban', text: 'blocked: ' + words(e.blocked_reason), title: 'The engine stopped this experiment before it ran.' });
    const prot = arr(e.protected).map((x) => str(x, 120)).filter(Boolean);
    prot.slice(0, 3).forEach((pth, k) => tags.push({ key: 'prot' + k, tone: 'bad', icon: 'file', text: pth, mono: true, title: 'Protected path: ' + pth }));
    if (prot.length > 3) tags.push({ key: 'protmore', tone: 'bad', text: '+' + (prot.length - 3) + ' protected', title: prot.slice(3).join(', ') });
    if (overfitSuspect(e)) tags.push({ key: 'overfit', tone: 'warn', icon: 'alert', text: 'overfit suspect', title: 'Train score improved but the held-out score did not.' });
    const nc = arr(e.claim_ids).length;
    if (nc) tags.push({ key: 'claims', tone: 'neutral', icon: 'chat', text: nc + (nc === 1 ? ' claim' : ' claims'), title: arr(e.claim_ids).map((x) => str(x, 40)).join(', ') });
    const rv = str(e.review, 20);
    if (rv) tags.push({ key: 'review', tone: rv === 'accept' ? 'neutral' : 'warn', icon: rv === 'accept' ? 'check' : 'help', text: 'review: ' + rv, title: 'Reviewer: ' + (str(e.reviewer, 80) || 'unknown') });
    tagList(r.tags, tags);

    const fs = arr(e.finding_ids).map((x) => str(x, 80)).filter(Boolean).map((fid) => {
      const f = ui.findingsById.get(fid);
      const invId = fid.split('@')[0];
      const closed = f && str(f.status) !== 'open';
      return { key: fid, act: 'sel-inv', arg: invId, tone: closed ? 'neutral' : (f && str(f.severity) === 'block' ? 'bad' : 'warn'), icon: closed ? 'check' : 'alert',
        text: invId + (f ? ' ' + str(f.status, 12) : ''), mono: true, title: 'Finding ' + fid + ': open the invariants panel' };
    });
    tagList(r.finds, fs);
    setHidden(r.finds, fs.length === 0);
  }

  function renderLanes(S) {
    const grid = $('#lane-grid');
    const ctx = { series: seriesByExp(S) };
    const list = S.experiments.filter((e) => expId(e));
    if (ui.newestFirst) list.reverse();
    syncList(grid, list, (e) => expId(e), makeCard, (n, e, i) => patchCard(n, e, i, ctx));
    const total = S.experiments.length;
    setEmpty($('#lane-empty'), 'No experiments yet', 'Lanes appear here when the supervisor launches the first experiment. Each card shows its stage pipeline, its outcome and what the auditor found.', total === 0);
    setHidden(grid, total === 0);
    const c = { run: 0, imp: 0, rep: 0, aw: 0, blocked: 0, planted: 0 };
    for (const e of S.experiments) {
      const st = str(e.status);
      if (!['recorded', 'blocked', 'violation'].includes(st)) c.run += 1;
      if (str(e.dev_outcome) === 'improved') c.imp += 1;
      if (e.replicated) c.rep += 1;
      if (e.awaiting_replication) c.aw += 1;
      if (st === 'blocked') c.blocked += 1;
      if (str(e.planted_fault)) c.planted += 1;
    }
    const parts = [total + (total === 1 ? ' experiment' : ' experiments')];
    if (c.run) parts.push(c.run + ' running');
    parts.push(c.imp + ' improved on dev', c.rep + ' replicated');
    if (c.aw) parts.push(c.aw + ' awaiting replication');
    if (c.blocked) parts.push(c.blocked + ' blocked');
    if (c.planted) parts.push(c.planted + ' with a planted fault');
    setText($('#lane-summary'), parts.join(' · '));
    const ob = $('#lane-order');
    setText(ob, ui.newestFirst ? 'Newest first' : 'Oldest first');
    setAttr(ob, 'aria-pressed', ui.newestFirst ? 'true' : 'false');
  }

  // ===========================================================================================
  // invariants + findings
  // ===========================================================================================
  const INV_STATE = {
    ok: { tone: 'ok', icon: 'check', text: 'ok' },
    violated: { tone: 'bad', icon: 'x', text: 'violated' },
    unchecked: { tone: 'neutral', icon: 'dash', text: 'unchecked' },
    insufficient: { tone: 'warn', icon: 'help', text: 'insufficient evidence' },
    llm_not_run: { tone: 'dashed', icon: 'half', text: 'LLM not run' },
    // against the simulator the LLM half of the invariant was read by a scripted stand-in: checked, but never by a model
    ok_simulated: { tone: 'simulated', icon: 'flask', text: SIM_JUDGE_LABEL },
  };
  const INV_KIND = { deterministic: 'deterministic', llm: 'LLM judge', both: 'deterministic + LLM' };

  function renderInvariants(S) {
    const grid = $('#inv-grid');
    const list = S.invariants.filter((v) => str(v.id));
    syncList(grid, list, (v) => str(v.id), () => {
      const t = h('button', { class: 'inv-tile', type: 'button', 'data-act': 'sel-inv' });
      t._id = h('span', { class: 'inv-id mono' });
      t._st = makeTag({}, 'inv-state');
      t._title = h('span', { class: 'inv-title' });
      t._kind = h('span', { class: 'inv-kind' });
      t._open = h('span', { class: 'inv-open num' });
      t.append(h('span', { class: 'inv-top' }, t._id, t._st), t._title, h('span', { class: 'inv-foot' }, t._kind, t._open));
      return t;
    }, (t, v) => {
      const id = str(v.id, 12);
      const key = own(INV_STATE, str(v.state)) ? str(v.state) : 'unchecked';
      const si = INV_STATE[key];
      setText(t._id, id);
      patchTag(t._st, si);
      setText(t._title, str(v.title, 80));
      setText(t._kind, own(INV_KIND, str(v.kind)) || str(v.kind, 20));
      const open = num(v.open) || 0;
      setText(t._open, open ? open + ' open' : 'none open');
      setClass(t, cls('inv-tile', 'tone-' + si.tone, ui.selInv === id ? 'is-selected' : '', open ? 'has-open' : ''));
      setAttr(t, 'data-arg', id);
      setAttr(t, 'aria-pressed', ui.selInv === id ? 'true' : 'false');
      setAttr(t, 'aria-label', id + ' ' + str(v.title, 80) + ': ' + si.text + ', ' + (open ? open + ' open' : 'none open'));
    });
    const counts = {};
    for (const v of list) { const k = own(INV_STATE, str(v.state)) ? str(v.state) : 'unchecked'; counts[k] = (counts[k] || 0) + 1; }
    const bits = [];
    for (const k of ['violated', 'ok', 'ok_simulated', 'llm_not_run', 'insufficient', 'unchecked']) { if (counts[k]) bits.push(counts[k] + ' ' + INV_STATE[k].text); }
    setText($('#inv-summary'), list.length ? list.length + ' invariants · ' + bits.join(' · ') + '. Missing evidence is never shown as compliance.' : '');
    setEmpty($('#inv-empty'), 'No invariants reported', 'The auditor registry is not available to this view, so nothing is shown as checked.', list.length === 0);
    setHidden(grid, list.length === 0);
  }

  function evidenceChips(f) {
    return arr(f.evidence_seqs).filter((x) => num(x) !== null).map((seq) => {
      const there = ui.seqSet.has(seq);
      return { key: 'ev' + seq, act: 'goto-ev', arg: seq, tone: 'neutral', icon: 'arrow', text: 'seq ' + seq, mono: true, off: !there,
        title: there ? 'Scroll to event ' + seq + ' in the timeline' : 'Event ' + seq + ' is outside the timeline window the server sent' };
    });
  }

  function renderFindings(S) {
    const list = ui.selInv ? S.findings.filter((f) => str(f.invariant_id) === ui.selInv) : S.findings.slice();
    const rank = { open: 0, addressed: 1, dismissed: 2 };
    list.sort((a, b) => (own(rank, str(a.status)) ?? 3) - (own(rank, str(b.status)) ?? 3));
    const inv = S.invariants.find((v) => str(v.id) === ui.selInv);
    setText($('#findings-title'), ui.selInv ? 'Findings for ' + ui.selInv + (inv ? ' · ' + str(inv.title, 60) : '') + ' (' + list.length + ')' : 'All findings (' + list.length + ')');
    const ul = $('#findings-list');
    syncList(ul, list, (f) => str(f.finding_id), () => {
      const li = h('li', { class: 'finding js-exp' });
      li._sev = makeTag({}, 'pill'); li._status = makeTag({}, 'pill'); li._src = makeTag({}, 'chip');
      li._subj = h('span', { class: 'finding-subj mono' });
      li._exp = makeTag({ act: 'sel-exp' }, 'chip');
      li._txt = h('p', { class: 'finding-text' });
      li._ev = h('div', { class: 'finding-ev' });
      li.append(h('div', { class: 'finding-top' }, li._sev, li._status, li._src, li._subj, li._exp), li._txt, li._ev);
      return li;
    }, (li, f) => {
      const sev = str(f.severity, 12), status = str(f.status, 14), verdict = str(f.verdict, 30);
      const sevTone = sev === 'block' ? 'bad' : sev === 'warn' ? 'warn' : 'neutral';
      patchTag(li._sev, { tone: sevTone, icon: sev === 'block' ? 'ban' : sev === 'warn' ? 'alert' : 'help', text: sev || 'info', title: 'Severity. Block findings stop a finish until closed.' });
      const stTone = status === 'open' ? (verdict === 'violated' ? 'bad' : 'warn') : 'ok';
      patchTag(li._status, { tone: stTone, icon: status === 'open' ? 'clock' : 'check', text: (verdict && verdict !== 'violated' ? words(verdict) + ', ' : '') + (status || 'open') });
      patchTag(li._src, { tone: 'neutral', text: str(f.source, 20) || 'unknown', title: 'Deterministic checks are code; LLM findings are a model judgment.' });
      setText(li._subj, str(f.finding_id, 80));
      const ex = str(f.exp_id, 40);
      setHidden(li._exp, !ex);
      patchTag(li._exp, { act: 'sel-exp', arg: ex, tone: 'neutral', text: ex, mono: true, title: 'Select ' + ex });
      setText(li._txt, str(f.rationale, 900) || 'No rationale recorded.');
      li._exps = ex ? [ex] : null;
      const chips = evidenceChips(f);
      tagList(li._ev, chips);
      setHidden(li._ev, chips.length === 0);
    });
    const none = list.length === 0;
    setEmpty($('#findings-empty'), ui.selInv ? 'No findings for ' + ui.selInv : 'No findings',
      ui.selInv ? 'This invariant has produced no finding. For an LLM-judged invariant that is only meaningful if the judge actually ran.' : 'The auditor has not raised a finding. That says nothing about invariants shown as LLM not run.', none);
  }

  // ===========================================================================================
  // supervisor
  // ===========================================================================================
  function renderTurns(S) {
    const ul = $('#turns-list');
    const turns = S.turns;
    syncList(ul, turns, (t, i) => (num(t.seq) !== null ? t.seq : 'i' + i), () => {
      const li = h('li', { class: 'turn' });
      li._n = h('span', { class: 'turn-n num' });
      li._src = makeTag({}, 'chip');
      li._eff = h('span', { class: 'chip chip-plain' });
      li._lat = h('span', { class: 'chip chip-plain num' });
      li._seq = makeTag({ act: 'goto-ev' }, 'chip');
      li._say = h('p', { class: 'turn-say' });
      li._text = h('p', { class: 'turn-text' });
      li._tools = h('div', { class: 'turn-tools' });
      li.append(h('div', { class: 'turn-head' }, li._n, li._src, li._eff, li._lat, h('span', { class: 'cd-spacer' }), li._seq), li._say, li._text, li._tools);
      return li;
    }, (li, t, i) => {
      const scripted = str(t.source) === 'script';
      setText(li._n, 'Turn ' + (i + 1));
      patchTag(li._src, scripted
        ? { tone: 'scripted', icon: 'flask', text: 'script narration', title: 'Written by a script, not generated by a model.' }
        : ui.sim
          ? { tone: 'simulated', icon: 'flask', text: 'simulated supervisor', title: SIM_SUPERVISOR_LABEL + '. The model name the simulator reported: ' + (str(t.model, 60) || 'none') }
          : { tone: 'live', icon: 'bolt', text: modelLabel(t.model) || 'model', title: 'Model: ' + str(t.model, 60) });
      const eff = str(t.reasoning_effort, 16);
      setText(li._eff, 'effort ' + eff);
      setHidden(li._eff, !eff);
      const lat = num(t.latency_ms);
      setText(li._lat, lat === null ? '' : (lat >= 1000 ? (lat / 1000).toFixed(1) + ' s' : Math.round(lat) + ' ms'));
      setHidden(li._lat, lat === null);
      const seq = num(t.seq);
      patchTag(li._seq, { act: 'goto-ev', arg: seq === null ? '' : seq, tone: 'neutral', text: seq === null ? '' : 'seq ' + seq, mono: true, off: seq === null || !ui.seqSet.has(seq), title: 'Scroll to this turn in the timeline' });
      setHidden(li._seq, seq === null);
      const say = str(t.say), text = str(t.text);
      setText(li._say, say);
      setHidden(li._say, !say);
      setClass(li._say, cls('turn-say', scripted ? 'is-script' : ''));
      setText(li._text, text);
      setHidden(li._text, !text);
      const calls = arr(t.tool_calls).map((c, k) => {
        const name = isObj(c) ? str(c.name, 40) : str(c, 60);
        const args = isObj(c) ? str(c.args_summary, 200) : '';
        return { key: 'c' + k, name, args };
      });
      syncList(li._tools, calls, (c) => c.key, () => {
        const c = h('span', { class: 'tool' });
        c._name = h('span', { class: 'tool-name mono' });
        c._args = h('span', { class: 'tool-args mono' });
        c.append(icon('arrow', 'tool-ic'), c._name, c._args);
        return c;
      }, (c, v) => {
        setText(c._name, v.name || 'tool');
        setText(c._args, v.args);
        setHidden(c._args, !v.args);
        setAttr(c, 'title', v.name + (v.args ? '(' + v.args + ')' : ''));
      });
      setHidden(li._tools, calls.length === 0);
    });
    const real = turns.filter((t) => str(t.source) !== 'script').length;
    setText($('#turns-sub'), turns.length ? turns.length + ' turns' + (real ? '' : ' of script narration') + '. Labels say who wrote each one.' +
      (ui.sim && real ? ' Every turn that is not script narration is ' + SIM_SUPERVISOR_LABEL + '.' : '') : '');
    setEmpty($('#turns-empty'), 'No supervisor turns yet', 'The supervisor composes each round of the workflow; its turns, tool calls and reasoning effort appear here in order.', turns.length === 0);
    setHidden(ul, turns.length === 0);
    followTail(ul, 'turns');
  }

  // ===========================================================================================
  // claims + finish
  // ===========================================================================================
  const CLAIM_STATUS = {
    supported: { tone: 'ok', icon: 'check', text: 'supported' },
    unreplicated: { tone: 'warn', icon: 'clock', text: 'unreplicated' },
    contradicted: { tone: 'bad', icon: 'x', text: 'contradicted' },
    refused: { tone: 'bad', icon: 'ban', text: 'refused' },
  };

  function renderClaims(S) {
    const H = S.headline;
    const score = [
      { key: 'made', label: 'made', value: fmtInt(H.claims_made), tone: '' },
      { key: 'acc', label: 'accepted', value: fmtInt(H.claims_accepted), tone: 'ok' },
      { key: 'ref', label: 'refused', value: fmtInt(H.claims_refused), tone: num(H.claims_refused) > 0 ? 'bad' : '' },
    ];
    syncList($('#claims-score'), score, (i) => i.key, () => {
      const n = h('div', { class: 'score' });
      n._v = h('span', { class: 'score-v num' }); n._l = h('span', { class: 'score-l' });
      n.append(n._v, n._l);
      return n;
    }, (n, i) => { setClass(n, cls('score', i.tone ? 'tone-' + i.tone : '')); setText(n._v, i.value); setText(n._l, i.label); });

    const ul = $('#claims-list');
    const claims = S.claims.filter((c) => str(c.claim_id));
    syncList(ul, claims, (c) => str(c.claim_id), () => {
      const li = h('li', { class: 'claim js-exp' });
      li._st = makeTag({}, 'pill'); li._id = h('span', { class: 'claim-id mono' }); li._type = h('span', { class: 'chip chip-plain' });
      li._exp = makeTag({ act: 'sel-exp' }, 'chip'); li._seq = makeTag({ act: 'goto-ev' }, 'chip');
      li._rf = makeTag({}, 'chip'); li._acc = makeTag({}, 'chip');
      li._txt = h('p', { class: 'claim-text' }); li._actor = h('span', { class: 'claim-actor' });
      li.append(h('div', { class: 'claim-top' }, li._st, li._acc, li._rf, li._id, li._type, li._exp, h('span', { class: 'cd-spacer' }), li._seq), li._txt, li._actor);
      return li;
    }, (li, c) => {
      let si = own(CLAIM_STATUS, str(c.status)) || { tone: 'neutral', icon: 'help', text: str(c.status, 20) || 'unknown' };
      // only an "improved" or "no_effect" claim is something the grader can check; for the rest "unreplicated" would mislead
      const checkable = ['improved', 'no_effect'].indexOf(str(c.type, 20)) >= 0;
      if (!checkable && str(c.status) === 'unreplicated') si = { tone: 'neutral', icon: 'help', text: 'not checkable', title: 'This kind of claim names no result the trusted grader could confirm or contradict.' };
      patchTag(li._st, si);
      patchTag(li._rf, { tone: 'bad', icon: 'ban', text: 'finish refused', title: 'A finish request citing this claim was refused by the finish gate.' });
      setHidden(li._rf, c.refused_finish !== true);
      patchTag(li._acc, { tone: 'ok', icon: 'check', text: 'finish accepted', title: 'The finish gate accepted this claim in the final summary.' });
      setHidden(li._acc, c.accepted !== true);
      setText(li._id, str(c.claim_id, 60));
      setText(li._type, str(c.type, 20) || 'claim');
      const ex = str(c.exp_id, 40);
      setHidden(li._exp, !ex);
      patchTag(li._exp, { act: 'sel-exp', arg: ex, tone: 'neutral', text: ex, mono: true, title: 'Select ' + ex });
      const seq = num(c.seq);
      setHidden(li._seq, seq === null);
      patchTag(li._seq, { act: 'goto-ev', arg: seq === null ? '' : seq, tone: 'neutral', text: seq === null ? '' : 'seq ' + seq, mono: true, off: seq === null || !ui.seqSet.has(seq), title: 'Scroll to this claim in the timeline' });
      setText(li._txt, str(c.text, 600));
      setText(li._actor, 'asserted by ' + (str(c.actor, 60) || 'unknown') + '. An assertion is not evidence.');
      li._exps = ex ? [ex] : null;
    });
    setEmpty($('#claims-empty'), 'No claims yet', 'Anything an agent asserts is recorded as a claim and checked against the grader. Nothing has been asserted so far.', claims.length === 0);
    setHidden(ul, claims.length === 0);

    // finish attempts, in ledger order
    const items = [];
    for (const r of S.finish.requests) items.push({ kind: 'request', seq: num(r.seq) === null ? 0 : num(r.seq), r });
    for (const r of S.finish.refusals) items.push({ kind: 'refusal', seq: num(r.seq) === null ? 0 : num(r.seq), r });
    items.sort((a, b) => a.seq - b.seq);
    if (S.finish.accepted) items.push({ kind: 'accepted', seq: Number.MAX_SAFE_INTEGER, r: {} });
    const fl = $('#finish-list');
    syncList(fl, items, (i) => i.kind + '-' + i.seq, () => {
      const li = h('li', { class: 'finish' });
      li._tag = makeTag({}, 'pill'); li._seq = makeTag({ act: 'goto-ev' }, 'chip');
      li._txt = h('p', { class: 'finish-text' });
      li._reasons = h('ul', { class: 'reasons' });
      li._cl = h('div', { class: 'finish-claims' });
      li.append(h('div', { class: 'finish-top' }, li._tag, li._seq), li._txt, li._reasons, li._cl);
      return li;
    }, (li, it) => {
      const r = it.r;
      const t = it.kind === 'request' ? { tone: 'neutral', icon: 'flag', text: 'finish requested' }
        : it.kind === 'refusal' ? { tone: 'bad', icon: 'ban', text: 'finish refused' } : { tone: 'ok', icon: 'check', text: 'finish accepted' };
      patchTag(li._tag, t);
      const real = it.seq !== Number.MAX_SAFE_INTEGER;
      setHidden(li._seq, !real);
      if (real) patchTag(li._seq, { act: 'goto-ev', arg: it.seq, tone: 'neutral', text: 'seq ' + it.seq, mono: true, off: !ui.seqSet.has(it.seq), title: 'Scroll to this event in the timeline' });
      const txt = it.kind === 'request' ? str(r.summary, 700) : it.kind === 'accepted' ? 'The finish gate accepted the final claims; the run was sealed.' : '';
      setText(li._txt, txt);
      setHidden(li._txt, !txt);
      const reasons = it.kind === 'refusal' ? arr(r.reasons).map((x, k) => ({ k, text: str(x, 400) })) : [];
      syncList(li._reasons, reasons, (x) => x.k, () => h('li'), (n, x) => setText(n, x.text));
      setHidden(li._reasons, reasons.length === 0);
      const cids = it.kind === 'request' ? arr(r.claim_ids).map((x) => str(x, 60)).filter(Boolean) : [];
      tagList(li._cl, cids.map((c) => ({ key: c, tone: 'neutral', icon: 'chat', text: c, mono: true, title: 'Claim cited by this finish request' })));
      setHidden(li._cl, cids.length === 0);
    });
    setEmpty($('#finish-empty'), 'No finish request yet', 'When the supervisor asks to finish, the gate checks replication and open findings. Refusals and their reasons are listed here.', items.length === 0);
    setHidden(fl, items.length === 0);
  }

  // ===========================================================================================
  // memory
  // ===========================================================================================
  function renderMemory(S) {
    const pv = S.pivots.map((p) => (typeof p === 'string' ? p : str(p.text || p.pivot)));
    const pl = $('#pivots-list');
    syncList(pl, pv.map((t, i) => ({ i, t })), (x) => x.i, () => h('li', { class: 'pivot' }, icon('arrow', 'pivot-ic'), h('span', {})), (li, x) => setText(li.lastChild, str(x.t, 500)));
    setEmpty($('#pivots-empty'), 'No pivots yet', 'A pivot is written when several experiments in a row fail to improve and the reflector changes course.', pv.length === 0);
    setHidden(pl, pv.length === 0);

    const ll = $('#lessons-list');
    const lessons = S.lessons.map((l, i) => (isObj(l) ? l : { text: String(l), lesson_id: 'L' + (i + 1) }));
    syncList(ll, lessons, (l, i) => str(l.lesson_id) || 'i' + i, () => {
      const li = h('li', { class: 'lesson js-exp' });
      li._id = h('span', { class: 'mono lesson-id' }); li._kind = makeTag({}, 'chip'); li._cls = makeTag({}, 'chip');
      li._txt = h('p', { class: 'lesson-text' }); li._src = h('div', { class: 'lesson-src' });
      li.append(h('div', { class: 'lesson-top' }, li._id, li._kind, li._cls), li._txt, li._src);
      return li;
    }, (li, l) => {
      setText(li._id, str(l.lesson_id, 20));
      patchTag(li._kind, { tone: 'neutral', text: str(l.kind, 20) || 'lesson' });
      const c = str(l.classification, 16);
      patchTag(li._cls, { tone: c === 'infra' ? 'warn' : 'neutral', icon: c === 'infra' ? 'alert' : '', text: c === 'model' ? 'about the hypothesis' : c === 'infra' ? 'infrastructure (not about the hypothesis)' : c || 'unclassified', title: 'Infrastructure lessons never claim a hypothesis failed.' });
      setText(li._txt, str(l.text, 700));
      const src = arr(l.source_exps).map((x) => str(x, 40)).filter(Boolean);
      tagList(li._src, src.map((x) => ({ key: x, act: 'sel-exp', arg: x, tone: 'neutral', text: x, mono: true, title: 'Select ' + x })));
      setHidden(li._src, src.length === 0);
      li._exps = src;
    });
    setEmpty($('#lessons-empty'), 'No lessons yet', 'Lessons are written after experiments that do not improve, so the next lane does not repeat them.', lessons.length === 0);
    setHidden(ll, lessons.length === 0);
  }

  // ===========================================================================================
  // models usage, fault injection
  // ===========================================================================================
  function renderUsage(S) {
    const rows = S.models_usage;
    const body = $('#usage-body');
    syncList(body, rows, (r) => str(r.role) + '|' + str(r.model), () => {
      const tr = h('tr');
      tr._c = ['role', 'model', 'calls', 'prompt', 'completion', 'cache', 'usd', 'effort', 'schema', 'retries'].map((k) => {
        const td = h('td', { 'data-label': ({ role: 'Role', model: 'Model', calls: 'Calls', prompt: 'Prompt tok', completion: 'Completion tok', cache: 'Cache', usd: 'USD', effort: 'Effort', schema: 'Strict schema', retries: 'Retries' })[k], class: ['calls', 'prompt', 'completion', 'cache', 'usd', 'schema', 'retries'].includes(k) ? 'r num' : '' });
        tr.append(td);
        return td;
      });
      return tr;
    }, (tr, r) => {
      const c = tr._c;
      setText(c[0], str(r.role, 30));
      setText(c[1], modelLabel(r.model));
      setText(c[2], fmtInt(r.calls));
      setText(c[3], fmtInt(r.prompt_tokens));
      setText(c[4], fmtInt(r.completion_tokens));
      const known = str(r.cache) === 'known' && num(r.prompt_tokens) > 0;
      setText(c[5], known ? Math.round((100 * (num(r.cached_tokens) || 0)) / num(r.prompt_tokens)) + '%' : 'n/a');
      setAttr(c[5], 'title', known ? fmtInt(r.cached_tokens) + ' of ' + fmtInt(r.prompt_tokens) + ' prompt tokens were served from cache' : 'The API did not report cached tokens for this role.');
      setText(c[6], num(r.usd) === null ? '—' : '$' + num(r.usd).toFixed(3));
      setAttr(c[6], 'data-label', ui.sim ? 'USD (simulated)' : 'USD');
      setAttr(c[6], 'title', ui.sim ? 'A ' + SIM_COST_LABEL : null);
      setAttr(c[1], 'title', ui.sim ? SIM_MODELS_LABEL + ': ' + str(r.model, 80) : str(r.model, 80));
      setText(c[7], str(r.effort, 16) || '—');
      setText(c[8], num(r.strict_schema_calls) > 0 ? fmtInt(r.schema_ok) + '/' + fmtInt(r.strict_schema_calls) : '—');
      setText(c[9], fmtInt(r.retries));
    });
    const total = rows.reduce((a, r) => a + (num(r.calls) || 0), 0);
    const usd = rows.reduce((a, r) => a + (num(r.usd) || 0), 0);
    const scripted = str(S.run.backend) === 'scripted';
    const simUsage = originOf(S.run) === 'simulated';
    setText($('#usage-sub'), rows.length ? fmtInt(total) + ' calls, ' + fmtUsd(usd) + ' in total' + (simUsage ? ' (' + SIM_COST_LABEL + ')' : '') + '. ' +
      (simUsage ? SIM_MODELS_LABEL + '.' : 'Models are the ones that answered.') + ' Cache shows n/a unless the API reported cached tokens.' : '');
    setText($('#usage-usd-h'), simUsage ? 'USD (simulated list-price estimate, nothing was billed)' : 'USD');
    setEmpty($('#usage-empty'), scripted ? 'No model calls were made' : 'No model calls recorded yet',
      scripted ? 'The scripted backend runs no language model, so there is nothing to meter. Live runs list every role, model, token count, cost and retry here.'
        : 'Each supervisor, designer, implementer, reviewer and judge call is charged to the ledger and listed here by role and model.', rows.length === 0);
    setHidden($('#usage-wrap'), rows.length === 0);
  }

  // One scoreboard row: a chip naming the planted fault plus a plain sentence saying what should have caught it and what did.
  function faultRow(x, group) {
    if (typeof x === 'string') return { key: group + '|' + str(x, 60), text: str(x, 60), why: '' };
    if (!isObj(x)) return { key: group + '|' + str(x, 60), text: str(x, 60), why: '' };
    const fault = str(x.fault, 40);
    const finish = fault === 'premature_finish';
    const seq = num(x.finish_seq);
    const id = str(x.finding_id || x.exp_id || x.id, 60);
    const label = finish ? 'premature finish' + (seq !== null ? ' @' + seq : '') : (id + (fault ? ' ' + fault : '') || str(x.invariant_id, 20));
    const exp = arr(x.expected).map((e) => str(e, 40)).filter(Boolean);
    const mustNot = arr(x.must_not_fire).map((e) => str(e, 20)).filter(Boolean);
    const parts = [];
    if (exp.length) parts.push('expected: ' + exp.join(', '));
    else if (mustNot.length) parts.push('must not fire: ' + mustNot.join(', '));
    if (str(x.reason, 200)) parts.push(str(x.reason, 200));
    if (!finish && str(x.by, 30) === 'premature_finish' && seq !== null) parts.push('finish request at seq ' + seq);
    const act = finish ? (seq !== null ? 'goto-ev' : null) : (str(x.exp_id, 40) ? 'sel-exp' : null);
    const arg = finish ? seq : (str(x.exp_id, 40) || null);
    return { key: group + '|' + (id || label) + '|' + (seq === null ? '' : seq) + '|' + fault, text: label, why: parts.join(' · '), act, arg };
  }

  function recallOf(f) {
    const r = f.recall;
    if (r && num(r.scored) !== null && num(r.caught) !== null) return { caught: num(r.caught), scored: num(r.scored) };
    const sc = f.detected.filter((d) => isObj(d) && !d.negative).length;
    const miss = f.missed.filter((d) => isObj(d) && !d.negative).length;
    return { caught: sc, scored: sc + miss };
  }

  function renderFaults(S) {
    const f = S.fault_injection;
    const sec = $('#sec-faults');
    const show = f.planted.length > 0;
    setHidden(sec, !show);
    if (!show) return;
    const rc = recallOf(f);
    const rec = rc.scored ? Math.round((100 * rc.caught) / rc.scored) : null;
    // While the run is going the auditor has not yet had its windows: an uncaught fault is "not caught yet", never a final miss.
    const running = S.run.finished !== true;
    const items = [
      { key: 'pl', label: 'Planted', value: f.planted.length, tone: '' },
      { key: 'de', label: 'Detected', value: f.detected.length, tone: 'ok', icon: 'check' },
      { key: 'mi', label: running ? 'Not caught yet' : 'Missed', value: f.missed.length, tone: f.missed.length ? (running ? 'neutral' : 'warn') : '', icon: f.missed.length ? (running ? 'clock' : 'alert') : '' },
      { key: 'ne', label: 'Not evaluated', value: f.not_evaluated.length, tone: f.not_evaluated.length ? 'neutral' : '', icon: f.not_evaluated.length ? 'half' : '' },
      { key: 'nr', label: 'Never ran', value: f.not_run.length, tone: 'warn', icon: 'ban' },
      { key: 'fp', label: 'False positives', value: f.false_positives.length, tone: f.false_positives.length ? 'bad' : '', icon: f.false_positives.length ? 'x' : '' },
    ].filter((i) => i.key !== 'nr' || f.not_run.length > 0);       // a run without a stopped lane keeps the board it always had
    syncList($('#faults-score'), items, (i) => i.key, () => {
      const n = h('div', { class: 'stat' });
      n._v = h('div', { class: 'stat-v num' }); n._ic = h('span', { class: 'stat-ic' }); n._l = h('div', { class: 'stat-l' });
      n.append(h('div', { class: 'stat-top' }, n._ic, n._v), n._l);
      return n;
    }, (n, i) => {
      setClass(n, cls('stat', i.tone ? 'tone-' + i.tone : ''));
      setText(n._v, String(i.value)); setText(n._l, i.label);
      setIcon(n._ic, i.icon || ''); setHidden(n._ic, !i.icon);
    });
    setText($('#faults-note'), rec === null ? 'Recall: nothing was scored.' + (f.not_run.length ? ' ' + f.not_run.length + ' planted ' + (f.not_run.length === 1 ? 'fault' : 'faults') + ' never ran.' : '') :
      (running ? 'Provisional recall ' : 'Recall ') + rec + '% (' + rc.caught + ' of ' + rc.scored + ' scored). Not-evaluated faults and negative controls are not in the recall figure.' +
      (f.not_run.length ? ' ' + f.not_run.length + (f.not_run.length === 1 ? ' planted fault never ran' : ' planted faults never ran') + ' (a gate stopped ' + (f.not_run.length === 1 ? 'it' : 'them') + ' before ' + (f.not_run.length === 1 ? 'it' : 'they') + ' ran), so ' + (f.not_run.length === 1 ? 'it is' : 'they are') + ' not in the recall figure either.' : '') +
      (running ? ' The run has not finished, so a fault not caught yet may still be caught by a later audit window.' : ''));
    const host = $('#faults-lists');
    const groups = [
      { key: 'detected', title: 'Detected', list: f.detected, tone: 'ok', icon: 'check', none: 'None.' },
      { key: 'missed', title: running ? 'Not caught yet' : 'Missed', list: f.missed, tone: running ? 'neutral' : 'warn', icon: running ? 'clock' : 'alert', none: 'None.',
        note: running ? 'The run is still going. Faults appear here until the auditor or reviewer has judged them; a later audit window may still catch them.' : '' },
      { key: 'ne', title: 'Not evaluated', list: f.not_evaluated, tone: 'neutral', icon: 'half', none: 'None.',
        note: 'Needs an LLM judge and no audit window ran one. This is not a miss and not a pass.' },
      { key: 'nr', title: 'Never ran', list: f.not_run, tone: 'warn', icon: 'ban', none: 'None.',
        note: 'A gate stopped these planted lanes before they ran (blocked as a duplicate, blocked by the budget, or never launched). They are neither caught nor missed and are not in the recall figure.' },
      { key: 'fp', title: 'False positives', list: f.false_positives, tone: 'bad', icon: 'x', none: 'None.' },
      { key: 'ex', title: 'Explained by a planted premature finish', list: f.explained, tone: 'neutral', icon: 'flag', none: 'None.',
        note: 'Findings the refused finish request is responsible for. They are the gate working, not detector noise.' },
    ].filter((g) => g.key !== 'ex' || g.list.length > 0).filter((g) => g.key !== 'ne' || g.list.length > 0).filter((g) => g.key !== 'nr' || g.list.length > 0);
    syncList(host, groups, (g) => g.key, () => {
      const d = h('div', { class: 'fault-group' });
      d._h = h('h3', { class: 'sub-h' });
      d._note = h('p', { class: 'fault-note' });
      d._c = h('ul', { class: 'fault-list' });
      d._none = h('p', { class: 'fault-none', text: 'None.' });
      d.append(d._h, d._note, d._c, d._none);
      return d;
    }, (d, g) => {
      setText(d._h, g.title + ' (' + g.list.length + ')');
      setText(d._note, g.note || '');
      setHidden(d._note, !g.note);
      const rows = g.list.map((x) => faultRow(x, g.key)).filter((x) => x.text);
      syncList(d._c, rows, (r) => r.key, (r) => {
        const li = h('li', { class: 'fault-row' });
        li._tag = makeTag({ act: r.act }, 'chip');
        li._why = h('span', { class: 'fault-why' });
        li.append(li._tag, li._why);
        return li;
      }, (li, r) => {
        patchTag(li._tag, { act: r.act, arg: r.arg === null ? null : r.arg, tone: g.tone, icon: g.icon, text: r.text, mono: true });
        setText(li._why, r.why || '');
        setHidden(li._why, !r.why);
      });
      setHidden(d._none, rows.length > 0);
    });
  }

  // ===========================================================================================
  // timeline
  // ===========================================================================================
  function kindGroup(kind) {
    const k = str(kind, 60);
    const head = k.split('.')[0];
    if (head === 'agent') return 'claim';
    if (head === 'supervisor' && k.indexOf('finish') >= 0) return 'supervisor';
    return TL_GROUPS.indexOf(head) >= 0 ? head : 'run';
  }

  function renderTimeline(S) {
    const rows = S.timeline.filter((r) => num(r.seq) !== null).slice(-TIMELINE_RENDER_MAX);
    ui.seqSet = new Set(rows.map((r) => r.seq));
    const ul = $('#tl-list');
    const start = num(S.run.started_ts);
    syncList(ul, rows, (r) => r.seq, (r) => {
      const li = h('li', { class: 'tl-row', id: 'ev-' + Math.trunc(r.seq), tabindex: '-1' });
      const kind = str(r.kind, 60), group = kindGroup(kind), text = str(r.text, 600);
      li._group = group;
      li._lc = (kind + ' ' + str(r.actor, 60) + ' ' + text).toLowerCase();
      li._exps = expIdsIn(text);
      if (li._exps.length) li.classList.add('js-exp');
      li.append(
        h('span', { class: 'tl-seq mono', text: '#' + Math.trunc(r.seq) }),
        h('span', { class: 'tl-time num', text: fmtRel(r.ts, start) }),
        h('span', { class: 'chip kind-' + group, text: kind }),
        h('span', { class: 'tl-actor', text: str(r.actor, 60) }),
        h('span', { class: 'tl-text', text }));
      return li;
    }, () => {});
    // filter chips: counts per group, only groups that occur
    const counts = {};
    for (const li of ul.children) counts[li._group] = (counts[li._group] || 0) + 1;
    const groups = TL_GROUPS.filter((g) => counts[g]);
    const fh = $('#tl-filters');
    syncList(fh, groups, (g) => g, (g) => h('button', { class: 'chip chip-toggle kind-' + g, type: 'button', 'data-act': 'tl-kind', 'data-arg': g }), (b, g) => {
      const on = !ui.tlHidden.has(g);
      setText(b, g + ' ' + counts[g]);
      setAttr(b, 'aria-pressed', on ? 'true' : 'false');
      setClass(b, cls('chip chip-toggle', 'kind-' + g, on ? 'is-on' : ''));
    });
    const shown = applyTimelineFilter();
    setText($('#tl-sub'), rows.length ? rows.length + ' events' + (S.ledger.events && S.ledger.events > rows.length ? ' of ' + fmtInt(S.ledger.events) + ' (latest shown)' : '') + '. Heartbeats are omitted.' : '');
    updateTlEmpty(rows.length, shown);
    setHidden(ul, rows.length === 0);
    followTail(ul, 'tl');
  }

  function updateTlEmpty(total, shown) {
    if (total === 0) setEmpty($('#tl-empty'), 'No events yet', 'Every action lands in the append-only ledger first; this list is that ledger, in order.', true);
    else setEmpty($('#tl-empty'), 'No matching events', 'No event matches the current filter. Clear the search or re-enable a kind.', shown === 0);
  }

  function applyTimelineFilter() {
    const q = ui.tlQuery;
    let shown = 0;
    for (const li of $('#tl-list').children) {
      const hide = ui.tlHidden.has(li._group) || (q !== '' && li._lc.indexOf(q) < 0);
      if (li.hidden !== hide) li.hidden = hide;
      if (!hide) shown += 1;
    }
    return shown;
  }

  // keep a scroll container pinned to its tail only while the reader is already at the tail
  function followTail(el, key) {
    if (!el._followInit) {
      el._followInit = true;
      el.addEventListener('scroll', () => { ui.follow[key] = el.scrollHeight - el.scrollTop - el.clientHeight < 32; }, { passive: true });
    }
    if (ui.follow[key]) el.scrollTop = el.scrollHeight;
  }

  // ===========================================================================================
  // selection + navigation (event delegation: handlers survive in-place patching)
  // ===========================================================================================
  function applySelection() {
    const sel = ui.selExp;
    for (const n of document.querySelectorAll('.js-exp')) {
      const ids = n._exps;
      const on = !!sel && !!ids && ids.indexOf(sel) >= 0;
      if (n.classList.contains('is-selected') !== on) n.classList.toggle('is-selected', on);
      if (n.classList.contains('pt') || n.classList.contains('lane-card')) {
        const b = n.classList.contains('pt') ? n : n._r.sel;
        setAttr(b, 'aria-pressed', on ? 'true' : 'false');
      }
    }
    for (const t of document.querySelectorAll('.inv-tile')) {
      const on = t.getAttribute('data-arg') === ui.selInv;
      t.classList.toggle('is-selected', on);
      t.setAttribute('aria-pressed', on ? 'true' : 'false');
    }
  }

  function selectExp(id) {
    ui.selExp = ui.selExp === id ? null : id;
    renderDetail();
    applySelection();
  }
  function selectInv(id, scroll) {
    ui.selInv = ui.selInv === id && !scroll ? null : id;
    if (ui.S) { renderFindings(ui.S); renderInvariants(ui.S); }
    applySelection();
    if (scroll) {
      const t = $('#findings-title');
      t.scrollIntoView({ block: 'center', behavior: motion() });
    }
  }
  function motion() { return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'; }

  function jumpToEvent(seq) {
    const n = num(Number(seq));
    if (n === null) return;
    let row = document.getElementById('ev-' + Math.trunc(n));
    if (!row) return;
    if (row.hidden) {
      ui.tlHidden.clear(); ui.tlQuery = '';
      $('#tl-search').value = '';
      if (ui.S) renderTimeline(ui.S);
      row = document.getElementById('ev-' + Math.trunc(n));
    }
    ui.follow.tl = false;
    if (ui.flashRow) ui.flashRow.classList.remove('flash');
    row.classList.add('flash');
    ui.flashRow = row;
    row.scrollIntoView({ block: 'center', behavior: motion() });
    clearTimeout(ui.flashTimer);
    ui.flashTimer = setTimeout(() => { row.classList.remove('flash'); }, 3200);
  }

  function onClick(ev) {
    const t = ev.target;
    if (!(t instanceof Element)) return;
    const a = t.closest('[data-act]');
    if (a) {
      const act = a.getAttribute('data-act'), arg = a.getAttribute('data-arg');
      if (a.getAttribute('aria-disabled') === 'true') return;
      switch (act) {
        case 'sel-exp': if (arg) selectExp(arg); break;
        case 'sel-inv': if (arg) selectInv(arg, a.closest('.lane-card') !== null); break;
        case 'goto-ev': jumpToEvent(arg); break;
        case 'clear-sel': ui.selExp = null; renderDetail(); applySelection(); break;
        case 'goto-card': {
          const grid = $('#lane-grid');
          const card = Array.from(grid.children).find((c) => c.dataset.key === ui.selExp);
          if (card) card.scrollIntoView({ block: 'center', behavior: motion() });
          break;
        }
        case 'toggle-more':
          if (ui.expanded.has(arg)) ui.expanded.delete(arg); else ui.expanded.add(arg);
          if (ui.S) renderLanes(ui.S);
          break;
        case 'tl-kind':
          if (ui.tlHidden.has(arg)) ui.tlHidden.delete(arg); else ui.tlHidden.add(arg);
          if (ui.S) renderTimeline(ui.S);
          break;
        default: break;
      }
      return;
    }
    const card = t.closest('.lane-card');
    if (card && card._exps && !t.closest('button, a, input')) selectExp(card._exps[0]);
  }

  function onKeydown(ev) {
    const t = ev.target;
    if (!(t instanceof Element) || !t.classList.contains('pt')) return;
    if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); selectExp(t.getAttribute('data-arg')); return; }
    if (ev.key === 'ArrowRight' || ev.key === 'ArrowLeft') {
      const all = Array.from(t.parentNode.children).filter((c) => c.classList.contains('pt'));
      const i = all.indexOf(t) + (ev.key === 'ArrowRight' ? 1 : -1);
      if (all[i]) { ev.preventDefault(); all[i].focus(); }
    }
    if (ev.key === 'Escape') hideTip();
  }

  function wireChartHover() {
    const wrap = $('#chart-wrap');
    const find = (t) => (t instanceof Element ? t.closest('.pt') : null);
    wrap.addEventListener('pointerover', (ev) => { const g = find(ev.target); if (g) showTip(g); });
    wrap.addEventListener('pointerout', (ev) => { const g = find(ev.target); if (g && !g.contains(ev.relatedTarget)) { if (document.activeElement !== g) hideTip(); } });
    wrap.addEventListener('focusin', (ev) => { const g = find(ev.target); if (g) showTip(g); });
    wrap.addEventListener('focusout', (ev) => { if (find(ev.target)) hideTip(); });
  }

  // ===========================================================================================
  // theme
  // ===========================================================================================
  function applyTheme(theme) {
    ui.theme = THEME_ORDER.indexOf(theme) >= 0 ? theme : 'auto';
    const root = document.documentElement;
    if (ui.theme === 'auto') root.removeAttribute('data-theme'); else root.setAttribute('data-theme', ui.theme);
    const btn = $('#theme-toggle');
    setText($('#theme-label'), THEME_LABEL[ui.theme]);
    btn.setAttribute('aria-label', 'Theme: ' + THEME_LABEL[ui.theme].toLowerCase() + '. Activate to change.');
  }
  function initTheme() {
    let saved = 'auto';
    try { saved = window.localStorage.getItem('nightshift.theme') || 'auto'; } catch (e) { saved = 'auto'; }
    applyTheme(saved);
    $('#theme-toggle').addEventListener('click', () => {
      const next = THEME_ORDER[(THEME_ORDER.indexOf(ui.theme) + 1) % THEME_ORDER.length];
      applyTheme(next);
      try { window.localStorage.setItem('nightshift.theme', next); } catch (e) { /* storage may be blocked */ }
      ui.chartSig = '';
      if (ui.S) renderChart(ui.S);
    });
  }

  // ===========================================================================================
  // connection + polling
  // ===========================================================================================
  function renderConn() {
    const el = $('#conn');
    if (!el._init) {
      el._ic = h('span', { class: 'conn-ic' });
      el._tx = h('span', { class: 'conn-tx' });
      el._btn = h('button', { class: 'link-btn', type: 'button', 'data-act': 'retry-now', text: 'Retry now' });
      el.append(el._ic, el._tx, el._btn);
      el._btn.addEventListener('click', () => { poll.now(); });
      el._init = true;
    }
    const c = ui.conn;
    if (c.ok === null) {
      setClass(el, 'conn tone-neutral'); setIcon(el._ic, 'clock'); setText(el._tx, 'Connecting…'); setHidden(el._btn, true);
    } else if (c.ok) {
      setClass(el, 'conn tone-ok'); setIcon(el._ic, 'dot');
      setText(el._tx, 'Connected · updated ' + clockOf(new Date(c.lastOk)));
      setHidden(el._btn, true);
    } else {
      const wait = Math.max(0, Math.ceil((c.retryAt - Date.now()) / 1000));
      setClass(el, 'conn tone-bad'); setIcon(el._ic, 'x');
      setText(el._tx, 'Disconnected' + (c.lastOk ? ' · showing data from ' + clockOf(new Date(c.lastOk)) : '') + ' · retrying in ' + wait + ' s');
      setHidden(el._btn, false);
    }
    document.body.classList.toggle('is-disconnected', c.ok === false);
  }

  const poll = (function () {
    let timer = null, busy = false;
    async function fetchState() {
      const ctl = new AbortController();
      const to = setTimeout(() => ctl.abort(), FETCH_TIMEOUT_MS);
      try {
        const res = await fetch('/api/state', { cache: 'no-store', credentials: 'same-origin', headers: { Accept: 'application/json' }, signal: ctl.signal });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const text = await res.text();
        if (text.length > MAX_BODY_CHARS) throw new Error('state too large');
        return text;
      } finally { clearTimeout(to); }
    }
    function schedule(ms) { clearTimeout(timer); timer = setTimeout(tick, ms); }
    async function tick() {
      if (document.hidden || busy) return;
      busy = true;
      try {
        const text = await fetchState();
        let S = ui.S;
        if (text !== ui.raw) {
          S = normalize(JSON.parse(text));
          ui.raw = text;
          render(S);
        }
        ui.conn.ok = true; ui.conn.lastOk = Date.now(); ui.conn.failures = 0;
        renderConn();
        schedule(S && S.run.finished === true ? POLL_DONE_MS : POLL_ACTIVE_MS);
      } catch (err) {
        ui.conn.ok = false;
        ui.conn.failures += 1;
        const wait = Math.min(BACKOFF_MAX_MS, 1000 * Math.pow(2, Math.min(ui.conn.failures - 1, 4)));
        ui.conn.retryAt = Date.now() + wait;
        renderConn();
        schedule(wait);
      } finally { busy = false; }
    }
    return {
      start() { schedule(0); },
      now() { schedule(0); },
      stop() { clearTimeout(timer); },
    };
  }());

  // ===========================================================================================
  // render orchestration: every panel is isolated so one bad payload cannot blank the page
  // ===========================================================================================
  function resetForNewRun() {
    // a different run is now served at this URL: nothing derived from the old one may linger (timeline rows are keyed by seq)
    ui.selExp = null; ui.selInv = null; ui.expanded.clear(); ui.tlHidden.clear(); ui.tlQuery = '';
    const q = $('#tl-search'); if (q) q.value = '';
    ui.follow.turns = true; ui.follow.tl = true; ui.chartSig = '';
    if (ui.flashRow) { ui.flashRow.classList.remove('flash'); ui.flashRow = null; }
    for (const id of ['#tl-list', '#turns-list', '#lane-grid', '#claims-list', '#findings-list', '#finish-list']) {
      const el = $(id); if (el) el.replaceChildren();
    }
    renderDetail();
  }

  function render(S) {
    const runId = str(S.run.run_id, 80);
    if (ui.runId !== null && ui.runId !== runId) { try { resetForNewRun(); } catch (err) { console.error('reset failed', err); } }
    ui.runId = runId;
    ui.S = S;
    ui.sim = originOf(S.run) === 'simulated';
    ui.findingsById = new Map(S.findings.map((f) => [str(f.finding_id), f]));
    const steps = [renderTimeline, renderHeader, renderBanner, renderRedTeam, renderReplay, renderChart, renderHeadline, renderVerdict,
      renderLanes, renderInvariants, renderFindings, renderTurns, renderClaims, renderMemory, renderUsage, renderFaults];
    for (const fn of steps) {
      try { fn(S); } catch (err) { console.error('panel failed: ' + fn.name, err); }
    }
    applySelection();
  }

  function init() {
    initTheme();
    document.addEventListener('click', onClick);
    document.addEventListener('keydown', onKeydown);
    wireChartHover();
    $('#lane-order').addEventListener('click', () => {
      ui.newestFirst = !ui.newestFirst;
      if (ui.S) renderLanes(ui.S);
    });
    $('#tl-search').addEventListener('input', (ev) => {
      ui.tlQuery = ev.target.value.trim().toLowerCase();
      updateTlEmpty(ui.seqSet.size, applyTimelineFilter());
    });
    const wrap = $('#chart-wrap');
    if (typeof ResizeObserver === 'function') {
      let raf = 0;
      new ResizeObserver(() => {
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(() => {
          const w = Math.floor(wrap.clientWidth);
          if (ui.S && w > 0 && w !== ui.chartW) { try { renderChart(ui.S); } catch (err) { console.error('chart resize failed', err); } }
        });
      }).observe(wrap);
    }
    document.addEventListener('visibilitychange', () => { if (document.hidden) poll.stop(); else poll.now(); });
    window.addEventListener('online', () => poll.now());
    renderConn();
    setInterval(() => { if (ui.conn.ok === false) renderConn(); }, 1000);
    poll.start();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
}());
