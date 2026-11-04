'use strict';
/* Browser verification of the Nightshift dashboard.
 *
 * Serves nightshift/dashboard/{index.html,app.js,app.css} and a mutable /api/state from a tiny node http server that
 * sets the exact CSP the real server sets, then drives chromium (preinstalled Playwright) through the contract.
 *
 *   node tests/frontend/check.js            exit 0 = every check passed, 1 = a check failed, 77 = playwright unavailable
 *   NIGHTSHIFT_SHOTS_DIR=/some/dir          where screenshots are written (default: <tmp>/nightshift-shots)
 */
const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');

const DASH_DIR = path.join(__dirname, '..', '..', 'nightshift', 'dashboard');
const FIX_DIR = path.join(__dirname, '..', 'fixtures');
const SHOTS = process.env.NIGHTSHIFT_SHOTS_DIR || path.join(os.tmpdir(), 'nightshift-shots');
const CSP = "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; " +
  "base-uri 'none'; form-action 'none'; frame-ancestors 'none'";
const PWN = '<img src=x onerror=window.__pwned=1>';
const STATIC = {
  '/': ['index.html', 'text/html; charset=utf-8'],
  '/app.js': ['app.js', 'text/javascript; charset=utf-8'],
  '/app.css': ['app.css', 'text/css; charset=utf-8'],
};

// ---------------------------------------------------------------------------------------------
function startServer() {
  const ctl = { body: '{}', status: 200, hits: [], mode: 'json' };
  const server = http.createServer((req, res) => {
    const headers = { 'Content-Security-Policy': CSP, 'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer', 'Cache-Control': 'no-store' };
    const url = req.url.split('?')[0];
    if (req.method !== 'GET' && req.method !== 'HEAD') { res.writeHead(405, headers); return res.end(); }
    if (url === '/api/state') {
      ctl.hits.push(Date.now());
      res.writeHead(ctl.status, { ...headers, 'Content-Type': 'application/json' });
      return res.end(ctl.body);
    }
    const f = STATIC[url];
    if (!f) { res.writeHead(404, headers); return res.end('not found'); }
    res.writeHead(200, { ...headers, 'Content-Type': f[1] });
    return res.end(fs.readFileSync(path.join(DASH_DIR, f[0])));
  });
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => resolve({ server, ctl, base: 'http://127.0.0.1:' + server.address().port }));
  });
}

const fixture = (name) => JSON.parse(fs.readFileSync(path.join(FIX_DIR, name), 'utf8'));
const clone = (o) => JSON.parse(JSON.stringify(o));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------------------------
const results = { passed: 0, failed: [] };
function check(cond, msg) {
  if (cond) results.passed += 1;
  else { results.failed.push(msg); console.log('  FAIL: ' + msg); }
  return !!cond;
}
async function scenario(name, fn) {
  const t0 = Date.now();
  const before = results.failed.length;
  try { await fn(); } catch (err) { results.failed.push(name + ': threw ' + (err && err.stack || err)); console.log('  THREW: ' + (err && err.stack || err)); }
  console.log((results.failed.length === before ? 'ok   ' : 'FAIL ') + name + ' (' + (Date.now() - t0) + ' ms)');
}

async function openPage(browser, srv, state, opts) {
  const o = Object.assign({ width: 1440, height: 900, scheme: 'light', wait: '#lane-grid .lane-card' }, opts || {});
  srv.ctl.body = typeof state === 'string' ? state : JSON.stringify(state);
  srv.ctl.status = 200;
  const ctx = await browser.newContext({ viewport: { width: o.width, height: o.height }, colorScheme: o.scheme });
  const page = await ctx.newPage();
  const log = { console: [], pageerrors: [], requests: [] };
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') log.console.push(m.type() + ': ' + m.text()); });
  page.on('pageerror', (e) => log.pageerrors.push(e.message));
  page.on('request', (r) => log.requests.push(r.url()));
  await page.addInitScript(() => {
    window.__csp = [];
    document.addEventListener('securitypolicyviolation', (e) => window.__csp.push(e.violatedDirective + ' ' + e.blockedURI));
  });
  await page.goto(srv.base + '/');
  if (o.wait) await page.waitForSelector(o.wait, { timeout: 8000 });
  return { ctx, page, log };
}

function cleanLog(name, log, page) {
  return page.evaluate(() => window.__csp).then((csp) => {
    check(log.console.length === 0, name + ': no console errors or warnings, got ' + JSON.stringify(log.console));
    check(log.pageerrors.length === 0, name + ': no uncaught page errors, got ' + JSON.stringify(log.pageerrors));
    check(csp.length === 0, name + ': no CSP violations, got ' + JSON.stringify(csp));
  });
}

const text = (page, sel) => page.evaluate((s) => { const n = document.querySelector(s); return n ? n.textContent : null; }, sel);
const count = (page, sel) => page.evaluate((s) => document.querySelectorAll(s).length, sel);
const visible = (page, sel) => page.evaluate((s) => { const n = document.querySelector(s); return !!n && !n.hidden && n.getClientRects().length > 0; }, sel);
const noHScroll = (page) => page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth);

// ---------------------------------------------------------------------------------------------
async function main() {
  let chromium;
  try { ({ chromium } = require('playwright')); } catch (e) { console.log('playwright is not available: ' + e.message); process.exit(77); }
  fs.mkdirSync(SHOTS, { recursive: true });
  const srv = await startServer();
  let browser;
  try { browser = await chromium.launch(); } catch (e) { console.log('chromium could not launch: ' + e.message); srv.server.close(); process.exit(77); }

  const scripted = fixture('state_scripted_sample.json');
  const live = fixture('state_live_sample.json');
  const simulated = fixture('state_simulated_sample.json');
  const real = fixture('state_scripted_real.json');
  const simReal = fixture('state_simulated_real.json');
  const redteam = fixture('state_redteam_sim.json');
  const partial = fixture('state_redteam_partial.json');
  // the wording a simulated run must use (copies of nightshift/dashboard/state_api.py; tests/test_frontend.py pins app.js to the same)
  const SIM_COST = 'simulated list-price estimate (nothing was billed)';
  const SIM_MODELS = 'Models the simulator reported answering (a stand-in, not Meta)';
  const SIM_JUDGE = 'checked by a simulated judge';
  const SIM_SUP = 'written by the simulated supervisor (a scripted stand-in), not a model';
  const SIM_RT_NOTE = 'In this simulated run those model roles are answered by the local simulator (a scripted stand-in), not by a model.';
  const RT_NOTICE = (n) => 'RED-TEAM LANES: ' + n + ' lane(s) in this run are scripted attackers injected by the operator. ' +
    'The supervisor, designer, reviewer and auditor are real model calls.';
  const SIM_BANNER = 'SIMULATED MODEL: the language model endpoint is a local test server. This run exercises the live code path ' +
    '(prompts, tool calls, metering, audit) but is not Meta model output.';

  // ---- headers + contract of the static page -------------------------------------------------
  await scenario('csp header and static page', async () => {
    const { ctx, page, log } = await openPage(browser, srv, scripted);
    const resp = await page.request.get(srv.base + '/');
    check(resp.headers()['content-security-policy'] === CSP, 'the page is served with the exact contract CSP');
    check((await page.evaluate(() => document.querySelectorAll('script:not([src])').length)) === 0, 'no inline script element');
    check((await page.evaluate(() => document.querySelectorAll('style').length)) === 0, 'no inline style element');
    const external = log.requests.filter((u) => !u.startsWith(srv.base) && !u.startsWith('data:'));
    check(external.length === 0, 'every request is same-origin or data:, got ' + JSON.stringify(external));
    await cleanLog('csp', log, page);
    await ctx.close();
  });

  // ---- scripted sample, desktop light: every section ------------------------------------------
  await scenario('scripted sample renders every section', async () => {
    const { ctx, page, log } = await openPage(browser, srv, scripted);
    await sleep(400);
    check((await text(page, '#badge-backend')).includes('SCRIPTED'), 'backend badge says SCRIPTED');
    check((await text(page, '#badge-ledger')).includes('chain consistent'), 'ledger badge says chain consistent');
    check((await text(page, '#badge-ledger')).includes(scripted.ledger.head_anchor), 'ledger badge shows the head anchor');
    check((await text(page, '#badge-isolation')).includes('userns'), 'isolation mode shown');
    check((await text(page, '#badge-status')).includes('FINISHED'), 'status badge shows finished');
    check((await count(page, '#budget .meter')) === 4, 'four budget bars');
    check((await text(page, '#budget')).includes('Replications') && (await text(page, '#budget')).includes('LLM spend') && (await text(page, '#budget')).includes('Wall time'), 'budget bars are labelled');
    check((await page.getAttribute('#budget .meter-bar', 'aria-valuenow')) === '81', 'experiments bar is 13 of 16');
    check((await text(page, '#run-id')) === 'r-7f3a9c21' && (await text(page, '#run-task')) === 'law-discovery', 'run id and task shown');

    check((await visible(page, '#banner')) && (await text(page, '#banner-text')).startsWith('SCRIPTED BACKEND: no language model is running'), 'scripted banner is visible');
    check((await count(page, '#banner button, #banner a, #banner input, #banner [role=button]')) === 0, 'banner has no control that could dismiss it');
    await page.click('#banner-text');
    await page.keyboard.press('Escape');
    await sleep(1300);
    check(await visible(page, '#banner'), 'banner survives clicks, Escape and a poll');

    check((await count(page, '#chart-wrap svg.chart')) === 1, 'hero chart svg exists');
    check((await count(page, '#chart-wrap .pt')) === 14, 'one chart marker per experiment');
    check((await count(page, '#chart-wrap .inc-line')) === 1 && (await count(page, '#chart-wrap .train-line')) === 1 && (await count(page, '#chart-wrap .baseline-line')) === 1, 'incumbent, train and baseline lines drawn');
    check((await count(page, '#chart-wrap .err')) >= 8, 'std error bars drawn');
    check((await count(page, '#chart-wrap .sh-promoted')) === 1 && (await count(page, '#chart-wrap .sh-baseline')) === 1, 'promoted and baseline markers');
    check((await count(page, '#chart-wrap .sh-failed')) >= 3 && (await count(page, '#chart-wrap .sh-blocked')) === 1, 'failed, invalid and blocked experiments are hollow markers');
    check((await count(page, '#chart-wrap .sh-warn')) >= 1, 'overfit suspect warning mark');
    check((await text(page, '#chart-dir')).includes('lower is better'), 'objective direction labelled');

    check((await count(page, '#lane-grid .lane-card')) === 14, 'one lane card per experiment');
    check((await count(page, '#lane-grid .lane-card:first-child .step')) === 6, 'six pipeline steps per card');
    const stepNames = await page.evaluate(() => [...document.querySelectorAll('#lane-grid .lane-card:first-child .step-name')].map((n) => n.textContent));
    check(JSON.stringify(stepNames) === JSON.stringify(['Design', 'Implement', 'Run', 'Grade', 'Review', 'Record']), 'pipeline stages in order');
    const lanes = await text(page, '#lane-grid');
    check(lanes.includes('PLANTED FAULT (scripted actor)'), 'planted fault tag');
    check(lanes.includes('awaiting replication') && lanes.includes('replicated') && lanes.includes('failed replication'), 'all three replication states');
    check(lanes.includes('blocked: duplicate'), 'blocked reason shown');
    check(lanes.includes('probe:model.py'), 'protected paths shown');
    check(lanes.includes('1 claim'), 'claim counts shown');
    check((await count(page, '#lane-grid .lc-finds button')) >= 4, 'finding links on cards');

    check((await count(page, '#inv-grid .inv-tile')) === 14, 'fourteen invariant tiles');
    const inv = await text(page, '#inv-grid');
    check(inv.includes('LLM not run') && inv.includes('violated') && inv.includes('ok'), 'invariant states shown as text');
    check((await count(page, '#findings-list .finding')) === 5, 'all findings listed with no tile selected');

    check((await count(page, '#turns-list .turn')) === 10, 'ten supervisor turns');
    check((await text(page, '#turns-list')).includes('script narration'), 'turns are labelled script narration');
    check((await text(page, '#turns-list')).includes('launch_experiments'), 'tool call chips');

    check((await count(page, '#claims-list .claim')) === 5, 'five claims');
    const cl = await text(page, '#claims-list');
    check(cl.includes('supported') && cl.includes('contradicted') && cl.includes('refused'), 'claim status chips');
    const fin = await text(page, '#finish-list');
    check(fin.includes('finish refused') && fin.includes('has no sealed replication') && fin.includes('finish accepted'), 'finish requests and refusal reasons');
    const hl = await text(page, '#headline');
    check(hl.includes('Claims made') && hl.includes('Unreplicated improvements blocked') && hl.includes('2.47'), 'headline card');
    check((await text(page, '#verdict')).includes('Accepted improvement'), 'final verdict card when finished');
    check((await count(page, '#lessons-list .lesson')) === 5 && (await count(page, '#pivots-list .pivot')) === 2, 'lessons and pivots');
    check((await visible(page, '#usage-empty')) && (await text(page, '#usage-empty')).includes('No model calls were made'), 'scripted run shows an intentional models-usage empty state');
    check(await visible(page, '#sec-faults'), 'fault-injection scoreboard shown when faults were planted');
    const fi = await text(page, '#sec-faults');
    check(fi.includes('Planted') && fi.includes('Detected') && fi.includes('Missed') && fi.includes('False positives') && fi.includes('e010'), 'scoreboard figures and the not-evaluated fault');
    check(fi.includes('e007 defect') && fi.includes('reviewer found the planted bug; repaired and re-reviewed'), 'a planted defect the reviewer caught is detected with its reason');
    check(fi.includes('premature finish @42') && fi.includes('refused by the finish gate'), 'the premature finish is a planted fault detected by the finish gate');
    check(fi.includes('Recall 100% (7 of 7 scored)'), 'recall counts scored faults only, got ' + fi.slice(0, 400));
    check((await visible(page, '#faults-lists')) && (await text(page, '#faults-lists')).includes('Not evaluated (1)') && (await text(page, '#faults-lists')).includes('needs an LLM judge; not run offline'), 'the not-evaluated group names its reason');
    check((await text(page, '#faults-lists')).includes('Missed (0)'), 'the injection is not counted as missed');
    check((await text(page, '#claims-list')).includes('finish refused') && (await text(page, '#claims-list')).includes('finish accepted'), 'claims show when the finish gate refused or accepted them');
    check((await count(page, '#verdict .v-claim')) === 2, 'the verdict card lists only the claims the finish gate accepted');
    check(await visible(page, '#verdict .v-summary') && (await text(page, '#verdict .v-summary')).includes('damped sinusoid'), 'the run summary is on the verdict card');
    await (await page.$('#sec-faults')).screenshot({ path: path.join(SHOTS, 'fault-scoreboard.png') });
    check((await count(page, '#tl-list .tl-row')) === 173, 'timeline rows');
    check((await count(page, '#tl-filters button')) >= 8, 'timeline kind filters');
    check(!/\bsecure\b/i.test(await page.evaluate(() => document.body.innerText)), 'the word secure never appears');
    await page.screenshot({ path: path.join(SHOTS, 'desktop-light.png'), fullPage: true });
    await cleanLog('scripted', log, page);
    await ctx.close();
  });

  // ---- selection highlights everywhere -----------------------------------------------------
  await scenario('selecting an experiment highlights it everywhere', async () => {
    const { ctx, page, log } = await openPage(browser, srv, scripted);
    await page.click('#chart-wrap .pt[data-arg="e005"]');
    check((await count(page, '#chart-wrap .pt.is-selected')) === 1, 'marker selected');
    check((await page.getAttribute('#chart-wrap .pt[data-arg="e005"]', 'aria-pressed')) === 'true', 'marker aria-pressed');
    check((await count(page, '#lane-grid .lane-card.is-selected')) === 1 && (await text(page, '#lane-grid .lane-card.is-selected .lc-id')) === 'e005', 'lane card selected');
    check((await text(page, '#chart-detail')).includes('e005') && (await text(page, '#chart-detail')).includes('damped sinusoid'), 'detail strip shows the experiment');
    check((await count(page, '#claims-list .claim.is-selected')) === 1, 'its claim is highlighted');
    check((await count(page, '#tl-list .tl-row.is-selected')) > 5, 'its timeline rows are highlighted');
    await page.click('#lane-grid .lane-card[data-key="e008"] .lc-select');
    check((await text(page, '#lane-grid .lane-card.is-selected .lc-id')) === 'e008' && (await count(page, '#chart-wrap .pt.is-selected')) === 1, 'clicking a card moves the selection');
    check((await page.evaluate(() => document.querySelector('#chart-wrap .pt.is-selected').getAttribute('data-arg'))) === 'e008', 'chart follows the card');
    await page.click('#chart-detail [data-act="clear-sel"]');
    check((await count(page, '.is-selected:not(.inv-tile)')) === 0, 'clear removes the highlight everywhere');
    // keyboard
    await page.focus('#chart-wrap .pt[data-arg="e001"]');
    await page.keyboard.press('Enter');
    check((await text(page, '#lane-grid .lane-card.is-selected .lc-id')) === 'e001', 'Enter on a focused marker selects it');
    await page.keyboard.press('ArrowRight');
    check((await page.evaluate(() => document.activeElement.getAttribute('data-arg'))) === 'e002', 'arrow keys move between markers');
    await cleanLog('selection', log, page);
    await ctx.close();
  });

  await scenario('chart tooltip on hover and focus', async () => {
    const { ctx, page, log } = await openPage(browser, srv, scripted);
    await page.hover('#chart-wrap .pt[data-arg="e005"]');
    check(await visible(page, '#chart-tip'), 'tooltip shows on hover');
    check((await text(page, '#chart-tip')).includes('e005') && (await text(page, '#chart-tip')).includes('replicated'), 'tooltip content');
    await page.mouse.move(5, 5);
    check(!(await visible(page, '#chart-tip')), 'tooltip hides again');
    await page.focus('#chart-wrap .pt[data-arg="e008"]');
    check((await visible(page, '#chart-tip')) && (await text(page, '#chart-tip')).includes('failed replication'), 'tooltip shows on keyboard focus');
    await ctx.close();
  });

  // ---- invariants view + evidence chips ------------------------------------------------------
  await scenario('invariants: select a tile, jump to evidence', async () => {
    const { ctx, page, log } = await openPage(browser, srv, scripted);
    await page.click('#inv-grid .inv-tile[data-arg="I04"]');
    check((await text(page, '#findings-title')).includes('I04'), 'findings title names the invariant');
    check((await count(page, '#findings-list .finding')) === 1, 'only that invariant findings are listed');
    const f = await text(page, '#findings-list .finding');
    check(f.includes('warn') && f.includes('open') && f.includes('e011'), 'severity, status and subject shown');
    check((await page.getAttribute('#inv-grid .inv-tile[data-arg="I04"]', 'aria-pressed')) === 'true', 'tile pressed');
    await page.click('#findings-list .finding [data-act="goto-ev"]');
    await sleep(500);
    const hit = await page.evaluate(() => {
      const r = document.getElementById('ev-128');
      const b = r.getBoundingClientRect();
      return { flash: r.classList.contains('flash'), inView: b.top >= 0 && b.bottom <= window.innerHeight };
    });
    check(hit.flash, 'the matching timeline row is highlighted');
    check(hit.inView, 'the matching timeline row is scrolled into view');
    await page.click('#inv-grid .inv-tile[data-arg="I04"]');
    check((await count(page, '#findings-list .finding')) === 5, 'second click clears the tile filter');
    await page.click('#inv-grid .inv-tile[data-arg="I13"]');
    check(await visible(page, '#findings-empty'), 'an invariant with no findings gets an intentional empty state');
    await page.click('#inv-grid .inv-tile[data-arg="I03"]');
    await page.evaluate(() => document.querySelector('#sec-invariants').scrollIntoView());
    await sleep(150);
    await (await page.$('#sec-invariants')).screenshot({ path: path.join(SHOTS, 'invariants.png') });
    await cleanLog('invariants', log, page);
    await ctx.close();
  });

  // ---- timeline filter ------------------------------------------------------------------------
  await scenario('timeline filter and search', async () => {
    const { ctx, page, log } = await openPage(browser, srv, scripted);
    const shown = () => page.evaluate(() => [...document.querySelectorAll('#tl-list .tl-row')].filter((r) => !r.hidden).length);
    check((await shown()) === 173, 'all rows shown');
    await page.click('#tl-filters [data-arg="experiment"]');
    check((await shown()) === 173 - 113, 'toggling a kind hides it');
    await page.click('#tl-filters [data-arg="experiment"]');
    await page.fill('#tl-search', 'watchdog');
    check((await shown()) >= 1 && (await shown()) < 5, 'search narrows the rows');
    await page.fill('#tl-search', 'zzzz-no-such-event');
    check((await shown()) === 0 && (await visible(page, '#tl-empty')), 'no match gives an empty state');
    await ctx.close();
  });

  // ---- themes ---------------------------------------------------------------------------------
  await scenario('light and dark themes both render', async () => {
    const dark = await openPage(browser, srv, scripted, { scheme: 'dark' });
    const bgDark = await dark.page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    check(bgDark === 'rgb(13, 17, 23)', 'prefers-color-scheme dark gives the dark palette, got ' + bgDark);
    await sleep(300);
    await dark.page.screenshot({ path: path.join(SHOTS, 'desktop-dark.png'), fullPage: true });
    await cleanLog('dark', dark.log, dark.page);
    await dark.ctx.close();

    const light = await openPage(browser, srv, scripted, { scheme: 'light' });
    const bgLight = await light.page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    check(bgLight === 'rgb(243, 244, 246)', 'light palette, got ' + bgLight);
    await light.page.click('#theme-toggle');
    check((await light.page.evaluate(() => document.documentElement.getAttribute('data-theme'))) === 'light', 'first toggle forces light');
    await light.page.click('#theme-toggle');
    check((await light.page.evaluate(() => document.documentElement.getAttribute('data-theme'))) === 'dark', 'second toggle forces dark');
    check((await light.page.evaluate(() => getComputedStyle(document.body).backgroundColor)) === 'rgb(13, 17, 23)', 'data-theme override beats the media query');
    check((await text(light.page, '#theme-label')) === 'Dark', 'toggle label follows');
    await light.page.click('#theme-toggle');
    check((await light.page.evaluate(() => document.documentElement.hasAttribute('data-theme'))) === false, 'third toggle returns to automatic');
    await cleanLog('light', light.log, light.page);
    await light.ctx.close();
  });

  // ---- mobile ---------------------------------------------------------------------------------
  await scenario('mobile 390x844: no horizontal scroll', async () => {
    for (const [name, st] of [['scripted', scripted], ['live', live], ['simulated', simulated], ['real', real], ['simulated-real', simReal]]) {
      const { ctx, page, log } = await openPage(browser, srv, st, { width: 390, height: 844 });
      await sleep(300);
      check(await noHScroll(page), name + ': page has no horizontal scroll at 390px');
      const sw = await page.evaluate(() => [document.documentElement.scrollWidth, document.documentElement.clientWidth]);
      check(sw[0] <= 390, name + ': scrollWidth ' + sw[0] + ' fits the viewport');
      const gutter = await page.evaluate(() => Math.round(document.querySelector('#sec-chart').getBoundingClientRect().left));
      check(gutter === 16, name + ': 16px side gutter, got ' + gutter);
      const wide = await page.evaluate(() => [...document.querySelectorAll('.panel')].filter((p) => p.getBoundingClientRect().right > 391).length);
      check(wide === 0, name + ': no panel wider than the viewport');
      if (name === 'scripted') await page.screenshot({ path: path.join(SHOTS, 'mobile.png'), fullPage: true });
      if (name === 'simulated') await page.screenshot({ path: path.join(SHOTS, 'mobile-simulated.png'), fullPage: true });
      if (name === 'real') await page.screenshot({ path: path.join(SHOTS, 'mobile-real.png'), fullPage: true });
      if (name === 'simulated-real') await page.screenshot({ path: path.join(SHOTS, 'mobile-simulated-real.png'), fullPage: true });
      if (name === 'live') await page.screenshot({ path: path.join(SHOTS, 'mobile-costs.png'), fullPage: true });
      await cleanLog('mobile ' + name, log, page);
      await ctx.close();
    }
  });

  await scenario('very long and hostile strings do not break the layout', async () => {
    const s = clone(scripted);
    s.experiments[2].hypothesis = 'A'.repeat(5000);
    s.experiments[5].hypothesis = 'word '.repeat(600);
    s.claims[0].text = 'B'.repeat(4000);
    s.turns[0].say = 'C'.repeat(4000);
    s.timeline[3].text = 'D'.repeat(4000);
    s.findings[0].rationale = 'E'.repeat(4000);
    s.run.task = 'T'.repeat(300);
    const { ctx, page, log } = await openPage(browser, srv, s, { width: 390, height: 844 });
    await sleep(300);
    check(await noHScroll(page), 'no horizontal scroll with unbroken 5000 character strings on a phone');
    await page.setViewportSize({ width: 1440, height: 900 });
    await sleep(300);
    check(await noHScroll(page), 'no horizontal scroll on desktop either');
    await cleanLog('long strings', log, page);
    await ctx.close();
  });

  // ---- injection ------------------------------------------------------------------------------
  await scenario('model-written markup never executes or becomes an element', async () => {
    const s = clone(scripted);
    s.experiments[2].hypothesis = PWN + ' quadratic';
    s.experiments[2].diff_summary = PWN;
    s.experiments[2].protected = [PWN];
    s.experiments[2].planted_fault = PWN;
    s.claims[0].text = PWN;
    s.claims[0].actor = PWN;
    s.findings[0].rationale = PWN;
    s.findings[0].title = PWN;
    s.lessons[0].text = PWN;
    s.pivots[0] = PWN;
    s.turns[0].say = PWN;
    s.turns[0].tool_calls = [{ name: PWN, args_summary: PWN }];
    s.timeline[5].text = PWN;
    s.timeline[5].kind = PWN;
    s.finish.refusals[0].reasons = [PWN];
    s.invariants[0].title = PWN;
    s.run.task = PWN;
    s.run.backend = PWN;
    s.banner = PWN;
    s.models_usage = [{ role: PWN, model: PWN, calls: 1, prompt_tokens: 1, completion_tokens: 1, cached_tokens: 0, cache: 'n/a', usd: 0, effort: PWN, strict_schema_calls: 0, schema_ok: 0, retries: 0 }];
    s.fault_injection.detected[0] = PWN;
    const { ctx, page, log } = await openPage(browser, srv, s);
    await sleep(500);
    check((await page.evaluate(() => window.__pwned)) === undefined, 'onerror handler never ran');
    check((await count(page, 'img')) === 0, 'no img element exists');
    check((await count(page, '[onerror]')) === 0, 'no element carries an onerror attribute');
    check((await count(page, 'script:not([src])')) === 0, 'no injected script');
    const where = await page.evaluate((p) => ({
      card: [...document.querySelectorAll('.lane-card')].some((c) => c.textContent.includes(p)),
      claim: [...document.querySelectorAll('.claim')].some((c) => c.textContent.includes(p)),
      timeline: [...document.querySelectorAll('.tl-row')].some((c) => c.textContent.includes(p)),
      turn: [...document.querySelectorAll('.turn')].some((c) => c.textContent.includes(p)),
      banner: document.getElementById('banner-text').textContent.includes(p),
    }), PWN);
    check(where.card && where.claim && where.timeline && where.turn && where.banner, 'the markup is shown as literal text: ' + JSON.stringify(where));
    await page.click('#chart-wrap .pt[data-arg="e002"]');
    await page.hover('#chart-wrap .pt[data-arg="e002"]');
    check((await count(page, 'img')) === 0 && (await page.evaluate(() => window.__pwned)) === undefined, 'still inert after selecting and hovering');
    await cleanLog('injection', log, page);
    await ctx.close();
  });


  // ---- honest mode labelling: simulated, live, custom host, scripted ------------------------------
  await scenario('simulated run: own badge colour, non-dismissible banner, no LIVE label', async () => {
    // the colours of the other two modes, read first: every page polls the one shared /api/state body
    const live0 = await openPage(browser, srv, live);
    const bgLive = await live0.page.evaluate(() => getComputedStyle(document.querySelector('#badge-backend')).backgroundColor);
    await live0.ctx.close();
    const scr0 = await openPage(browser, srv, scripted);
    const bgScr = await scr0.page.evaluate(() => getComputedStyle(document.querySelector('#badge-backend')).backgroundColor);
    await scr0.ctx.close();
    const { ctx, page, log } = await openPage(browser, srv, simulated);
    const badge = await text(page, '#badge-backend');
    check(badge.includes('SIMULATED') && !badge.includes('LIVE') && !badge.includes('SCRIPTED'), 'badge says SIMULATED only, got ' + badge);
    const tones = await page.evaluate(() => ({ b: document.querySelector('#badge-backend').className, bg: getComputedStyle(document.querySelector('#badge-backend')).backgroundColor }));
    check(tones.b.includes('tone-simulated') && !tones.b.includes('tone-live') && !tones.b.includes('tone-scripted'), 'its own tone class, got ' + tones.b);
    check(tones.bg !== bgLive && tones.bg !== bgScr && bgLive !== bgScr, 'SIMULATED, LIVE and SCRIPTED have three different badge colours: ' + [tones.bg, bgLive, bgScr].join(' | '));
    check((await visible(page, '#banner')) && (await text(page, '#banner-text')) === SIM_BANNER, 'the exact simulated banner is visible');
    check((await count(page, '#banner button, #banner a, #banner input, #banner [role=button]')) === 0, 'the banner has nothing that could dismiss it');
    await page.click('#banner-text'); await page.keyboard.press('Escape'); await sleep(1300);
    check(await visible(page, '#banner'), 'the banner survives clicks, Escape and a poll');
    check(await page.evaluate(() => document.querySelector('#banner').classList.contains('is-simulated') && !document.querySelector('#banner').classList.contains('is-scripted')), 'the banner uses the simulated styling');
    const ml = await text(page, '#models-line');
    check(ml.includes('local test server') && ml.includes('not Meta') && ml.includes('127.0.0.1:41873'), 'the models line names the local test server, got ' + ml);
    check(!(await visible(page, '#sec-faults')), 'no fault scoreboard on a simulated run');
    check(!(await text(page, '#turns-list')).includes('script narration'), 'turns are not labelled script narration');
    await (await page.$('.site-header')).screenshot({ path: path.join(SHOTS, 'simulated-header.png') });
    await page.screenshot({ path: path.join(SHOTS, 'simulated-desktop.png'), clip: { x: 0, y: 0, width: 1440, height: 420 } });
    await cleanLog('simulated', log, page);
    await ctx.close();
  });

  await scenario('simulated banner is derived from the mode even if the producer sends none', async () => {
    const s = clone(simulated);
    s.banner = null;
    const { ctx, page } = await openPage(browser, srv, s);
    check((await visible(page, '#banner')) && (await text(page, '#banner-text')) === SIM_BANNER, 'the page supplies the simulated banner itself');
    const r = clone(simulated);
    r.run.mode = 'replay'; r.run.origin = 'simulated'; r.replay = { speed: 2, position_seq: 10, total_seq: 90, source_run_id: 'r-5c20a8d3' };
    const rp = await openPage(browser, srv, r);
    check((await text(rp.page, '#badge-backend')).includes('REPLAY') && (await text(rp.page, '#badge-backend')).includes('simulated'), 'a replay of a simulated run says so');
    check(await visible(rp.page, '#banner'), 'and keeps the simulated banner');
    await rp.ctx.close();
    await ctx.close();
  });

  // ---- red-team lanes (nightshift run --plant): scripted attackers inside a run with real model roles -----------------
  await scenario('red-team run: persistent notice under the simulated banner, attackers tagged, scoreboard computed', async () => {
    const n = redteam.run.red_team.length;
    check(n === 3 && redteam.run.origin === 'simulated' && redteam.run.backend === 'meta', 'the fixture is a simulated meta run with three attackers');
    const { ctx, page, log } = await openPage(browser, srv, redteam);
    await sleep(400);
    check((await visible(page, '#red-team-notice')) && (await text(page, '#red-team-text')) === RT_NOTICE(n), 'the exact red-team notice is visible, got ' + (await text(page, '#red-team-text')));
    check((await visible(page, '#banner')) && (await text(page, '#banner-text')) === SIM_BANNER, 'the simulated banner still applies on top');
    check(!(await text(page, '#banner-text')).includes('SCRIPTED BACKEND') && !(await text(page, '#banner-text')).includes('fault-injection harness'), 'no scripted-backend banner on a meta run');
    const order = await page.evaluate(() => { const a = document.querySelector('#banner'), b = document.querySelector('#red-team-notice'); return !!(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING); });
    check(order, 'the red-team notice sits under the simulated banner');
    check((await count(page, '#red-team-notice button, #red-team-notice a, #red-team-notice input, #red-team-notice [role=button]')) === 0, 'the notice has nothing that could dismiss it');
    await page.click('#red-team-text'); await page.keyboard.press('Escape'); await sleep(1300);
    check(await visible(page, '#red-team-notice'), 'the notice survives clicks, Escape and a poll');
    check((await count(page, '#red-team-notice')) === 1, 'polling never duplicates the notice');
    const list = await text(page, '#red-team-list');
    check(redteam.run.red_team.every((r) => list.includes(r.behavior + ' (' + r.scripted_id + ')')), 'the planted behaviors and bank ids are listed, got ' + list);
    check(list.includes(SIM_RT_NOTE), 'a simulated run says its model roles are the simulator, got ' + list);
    const badge = await text(page, '#badge-backend');
    check(badge.includes('SIMULATED') && !badge.includes('SCRIPTED') && !badge.includes('LIVE'), 'the badge still says SIMULATED, got ' + badge);
    const tagged = await page.evaluate(() => [...document.querySelectorAll('#lane-grid .lane-card')].filter((c) => c.textContent.includes('PLANTED FAULT (scripted actor)')).map((c) => c.textContent));
    check(tagged.length === 3, 'exactly the three attacker lanes are tagged PLANTED FAULT (scripted actor), got ' + tagged.length);
    check(tagged.every((t) => /tamper|overfit|claim_without_run/.test(t)), 'each tag names its fault kind');
    check(await visible(page, '#sec-faults'), 'the fault scoreboard is shown');
    const fi = await text(page, '#sec-faults');
    check(fi.includes('Recall 100% (3 of 3 scored)') && fi.includes('tamper') && fi.includes('overfit') && fi.includes('claim_without_run'), 'scoreboard 3 of 3, got ' + fi.slice(0, 300));
    check(!fi.includes('premature finish'), 'no scripted-supervisor rows on a meta run');
    await (await page.$('.site-header')).screenshot({ path: path.join(SHOTS, 'redteam-header.png') });
    await page.screenshot({ path: path.join(SHOTS, 'redteam-desktop.png'), clip: { x: 0, y: 0, width: 1440, height: 520 } });
    await cleanLog('red team', log, page);
    await ctx.close();
  });

  await scenario('red-team notice: live meta run, mobile width, absent without attackers, hostile values inert', async () => {
    const liveRt = clone(live);
    liveRt.run.red_team = [{ behavior: 'tamper', scripted_id: 'tamper' }, { behavior: 'overfit', scripted_id: 'poly9' }];
    const a = await openPage(browser, srv, liveRt);
    check((await visible(a.page, '#red-team-notice')) && (await text(a.page, '#red-team-text')) === RT_NOTICE(2), 'a live run with attackers shows the notice for 2 lanes');
    check(!(await text(a.page, '#red-team-list')).includes('simulat'), 'a live run with attackers does not mention a simulator');
    check(!(await visible(a.page, '#banner')), 'and no scripted or simulated banner (it is live)');
    check((await text(a.page, '#badge-backend')).includes('LIVE'), 'the badge stays LIVE: the model roles are real');
    await cleanLog('live red team', a.log, a.page);
    await a.ctx.close();
    const m = await openPage(browser, srv, redteam, { width: 390, height: 844 });
    check(await noHScroll(m.page), 'no horizontal scroll at 390 px with both banners');
    await m.page.screenshot({ path: path.join(SHOTS, 'redteam-mobile.png') });
    // the stacked models table: a figure under the long label "USD (simulated)" must stay on one line, "$0.025" never splits
    const usdLines = await m.page.evaluate(() => [...document.querySelectorAll('#usage-body td[data-label^="USD"]')].map((td) => {
      const r = document.createRange(); r.selectNodeContents([...td.childNodes].find((n) => n.nodeType === 3) || td);
      return new Set([...r.getClientRects()].map((q) => Math.round(q.top))).size;
    }));
    check(usdLines.length > 0 && usdLines.every((n) => n === 1), 'each simulated USD figure sits on one line at 390 px, got ' + JSON.stringify(usdLines));
    await m.ctx.close();
    for (const [label, s] of [['live', live], ['scripted', scripted], ['simulated', simulated]]) {
      const p = await openPage(browser, srv, s);
      check(!(await visible(p.page, '#red-team-notice')), 'no red-team notice on the ' + label + ' sample');
      await p.ctx.close();
    }
    const hostile = clone(liveRt);
    hostile.run.red_team = [{ behavior: PWN, scripted_id: '<script>window.__pwned=1</script>' }, 'junk', null, 7, { behavior: 'ok' }, { behavior: '' }, { scripted_id: 'x' }];
    const h = await openPage(browser, srv, hostile);
    check((await count(h.page, '#red-team-notice img, #red-team-notice script')) === 0, 'hostile attacker names never become elements');
    check((await h.page.evaluate(() => window.__pwned)) === undefined, 'nothing executed');
    check((await text(h.page, '#red-team-text')) === RT_NOTICE(2), 'only well-formed records are counted');
    await cleanLog('hostile red team', h.log, h.page);
    await h.ctx.close();
    const gone = clone(liveRt);
    gone.run.red_team = [];
    const g = await openPage(browser, srv, liveRt);
    srv.ctl.body = JSON.stringify(gone);
    await sleep(1800);
    check(!(await visible(g.page, '#red-team-notice')), 'the notice is hidden again when the state no longer lists attackers');
    await g.ctx.close();
  });

  // ---- planted lanes a gate stopped before they ran; a simulated run never reads like a model run ------------------
  await scenario('red-team run where two attackers never ran: the page says so and recall does not count them', async () => {
    const fi = partial.fault_injection;
    check(partial.run.origin === 'simulated' && fi.not_run.length === 2 && fi.red_team.planted === 3 && fi.red_team.ran === 1, 'the fixture has 3 planted lanes, 1 ran, 2 never did');
    const { ctx, page, log } = await openPage(browser, srv, partial);
    await sleep(400);
    const notice = await text(page, '#red-team-notice');
    check((await text(page, '#red-team-text')) === RT_NOTICE(3), 'the exact notice is unchanged');
    check((await visible(page, '#red-team-ran')) && (await text(page, '#red-team-ran')) === fi.red_team.text, 'the notice says how many lanes ran, got ' + (await text(page, '#red-team-ran')));
    check(notice.includes('1 ran') && notice.includes('2 were blocked as duplicates before they ran'), 'it names the duplicate blocks, got ' + notice);
    check((await text(page, '#red-team-ran')).length > 0 && (await count(page, '#red-team-notice button, #red-team-notice a, #red-team-notice input')) === 0, 'still nothing that could dismiss it');
    const sec = await text(page, '#sec-faults');
    check(sec.includes('Never ran') && sec.includes('Never ran (2)'), 'a Never ran card and group, got ' + sec.slice(0, 400));
    check(sec.includes('Recall 100% (1 of 1 scored)') && sec.includes('2 planted faults never ran') && sec.includes('not in the recall figure either'), 'recall counts only the lane that ran, got ' + sec.slice(0, 600));
    check(sec.includes('RT-overfit') && sec.includes('RT-claim_without') && sec.includes('blocked as a duplicate before it ran'), 'the blocked lanes are listed with their reason');
    const cards = await page.evaluate(() => [...document.querySelectorAll('#faults-score .stat')].map((c) => c.textContent));
    check(cards.some((t) => t.includes('Never ran') && t.includes('2')) && cards.some((t) => t.includes('Planted') && t.includes('3')), 'the scoreboard cards show 3 planted and 2 never ran, got ' + JSON.stringify(cards));
    await page.screenshot({ path: path.join(SHOTS, 'redteam-partial.png'), clip: { x: 0, y: 0, width: 1440, height: 700 } });
    await cleanLog('partial red team', log, page);
    await ctx.close();
  });

  await scenario('red-team run where every planted lane ran says so; hostile text in the new fields stays inert', async () => {
    const all = clone(redteam);
    all.fault_injection.red_team = { planted: 3, ran: 3, not_run: 0, text: 'All 3 planted attacker lane(s) ran.' };
    const a = await openPage(browser, srv, all);
    check((await visible(a.page, '#red-team-ran')) && (await text(a.page, '#red-team-ran')) === 'All 3 planted attacker lane(s) ran.', 'a full run says all lanes ran');
    check(!(await text(a.page, '#sec-faults')).includes('Never ran'), 'and shows no Never ran group');
    await a.ctx.close();
    const old = await openPage(browser, srv, redteam);          // an older state without fault_injection.red_team / not_run
    check(!(await visible(old.page, '#red-team-ran')) && !(await text(old.page, '#sec-faults')).includes('Never ran'), 'an older state without the new keys still renders, nothing invented');
    await old.ctx.close();
    const bad = clone(partial);
    bad.fault_injection.red_team.text = PWN;
    bad.fault_injection.not_run[0].reason = PWN;
    bad.fault_injection.not_run[1] = 'junk';
    bad.fault_injection.not_run.push(null, 7, { lane: PWN, fault: PWN, id: PWN, reason: { x: 1 } });
    const h = await openPage(browser, srv, bad);
    check((await count(h.page, '#red-team-notice img, #sec-faults img, #sec-faults script')) === 0, 'hostile values never become elements');
    check((await h.page.evaluate(() => window.__pwned)) === undefined, 'nothing executed');
    await cleanLog('hostile not-run', h.log, h.page);
    await h.ctx.close();
    const garbage = clone(partial);
    garbage.fault_injection.not_run = 'nope';
    garbage.fault_injection.red_team = [1, 2];
    const g = await openPage(browser, srv, garbage);
    check(await visible(g.page, '#sec-faults'), 'garbage types in the new keys do not blank the scoreboard');
    await cleanLog('garbage not-run', g.log, g.page);
    await g.ctx.close();
  });

  await scenario('simulated run: costs, models, judge and supervisor are all labelled as stand-ins', async () => {
    const { ctx, page, log } = await openPage(browser, srv, partial);
    await sleep(300);
    const hl = await text(page, '#headline');
    check(hl.includes(SIM_COST) && hl.includes('Simulated LLM cost'), 'the headline cost card carries the simulated-cost label, got ' + hl);
    check((await text(page, '#budget')).includes('LLM spend (simulated list-price estimate)'), 'the spend meter does too');
    check((await text(page, '#usage-usd-h')).includes('simulated list-price estimate') && (await text(page, '#usage-sub')).includes(SIM_COST), 'the models table header and subtitle label the USD column');
    check((await text(page, '#usage-sub')).includes(SIM_MODELS), 'the models tab says the simulator reported the names, got ' + (await text(page, '#usage-sub')));
    check(!(await text(page, '#usage-sub')).includes('Models are the ones that answered'), 'and never that the models answered');
    const inv = await text(page, '#inv-grid');
    for (const id of ['I07', 'I08', 'I13', 'I14']) {
      const t = await page.evaluate((i) => { const n = document.querySelector('#inv-grid .inv-tile[data-arg="' + i + '"]'); return n ? n.textContent : ''; }, id);
      check(t.includes(SIM_JUDGE), id + ' is shown as checked by a simulated judge, got ' + t);
    }
    check(inv.includes(SIM_JUDGE) && !/\bok\b/.test(await page.evaluate(() => [...document.querySelectorAll('#inv-grid .inv-tile')].filter((n) => ['I07', 'I08', 'I13', 'I14'].includes(n.dataset.arg)).map((n) => n.querySelector('.inv-state').textContent).join(' '))), 'none of the LLM-judged tiles reads plain ok');
    const summ = await text(page, '#inv-summary');
    check(summ.includes('4 ' + SIM_JUDGE), 'the invariant summary counts them, got ' + summ);
    check((await text(page, '#turns-sub')).includes(SIM_SUP) && (await count(page, '#turns-list .turn .chip')) > 0, 'the supervisor panel says its turns are the stand-in');
    const chips = await page.evaluate(() => [...document.querySelectorAll('#turns-list .turn .turn-head .chip')].map((n) => n.textContent + '|' + (n.title || '')));
    check(chips.some((c) => c.startsWith('simulated supervisor') && c.includes(SIM_SUP)), 'each turn is tagged simulated supervisor, got ' + JSON.stringify(chips.slice(0, 3)));
    check((await text(page, '#verdict')).includes('Final summary, ' + SIM_SUP + '.'), 'the final summary is attributed to the simulated supervisor');
    const cost = await page.evaluate(() => [...document.querySelectorAll('#lane-grid .lane-card .lc-cost:not([hidden]) .lc-cost-l')].map((n) => n.textContent));
    check(cost.length > 0 && cost.every((t) => t === 'Cost (simulated)'), 'per-experiment costs are labelled simulated, got ' + JSON.stringify(cost));
    const all = (await text(page, 'body'));
    check(!all.includes('Models that answered'), 'the page never says models answered in a simulated run');
    await page.screenshot({ path: path.join(SHOTS, 'simulated-labels.png'), fullPage: true });
    await cleanLog('simulated labels', log, page);
    await ctx.close();
    const m = await openPage(browser, srv, partial, { width: 390, height: 844 });
    check(await noHScroll(m.page), 'no horizontal scroll at 390 px with the longer simulated labels');
    await m.page.screenshot({ path: path.join(SHOTS, 'simulated-labels-mobile.png'), fullPage: true });
    await m.ctx.close();
  });

  await scenario('live and scripted runs keep their wording: no simulated label appears', async () => {
    for (const [label, st] of [['live', live], ['scripted', scripted]]) {
      const { ctx, page, log } = await openPage(browser, srv, st);
      const body = await text(page, 'body');
      check(!body.includes(SIM_COST) && !body.includes(SIM_JUDGE) && !body.includes(SIM_SUP) && !body.includes(SIM_MODELS), label + ': none of the simulated-run labels');
      check(!(await visible(page, '.v-summary-by')), label + ': no simulated attribution on the final summary');
      check((await text(page, '#usage-usd-h')) === 'USD', label + ': the USD column keeps its plain header');
      check(!(await text(page, '#sec-faults')).includes('Never ran'), label + ': no Never ran card');
      await cleanLog(label + ' wording', log, page);
      await ctx.close();
    }
  });

  await scenario('isolation badge: unshare-net warns like none, userns stays neutral', async () => {
    for (const [iso, tone] of [['unshare-net', 'tone-warn'], ['none', 'tone-warn'], ['userns', 'tone-neutral']]) {
      const s = clone(live);
      s.run.isolation = iso;
      const p = await openPage(browser, srv, s);
      const cls = await p.page.evaluate(() => document.querySelector('#badge-isolation').className);
      check(cls.includes(tone), iso + ' badge has ' + tone + ', got ' + cls);
      if (iso === 'unshare-net') {
        const t = await p.page.evaluate(() => document.querySelector('#badge-isolation').title);
        check((await text(p.page, '#badge-isolation')).includes('isolation: unshare-net') && /labels/.test(await text(p.page, '#badge-isolation')), 'it says labels readable');
        check(t.includes('read the labels') && t.includes('not detected') && t.includes('trusted directory') && !/sealed-safe/i.test(t), 'the title states what it does not stop and what it detects, got ' + t);
      }
      await p.ctx.close();
    }
  });

  await scenario('live badge: meta-api host, custom host, and older ledgers without an endpoint', async () => {
    let p = await openPage(browser, srv, live);
    check((await text(p.page, '#badge-backend')).includes('LIVE') && !(await visible(p.page, '#banner')), 'meta-api is LIVE with no banner');
    check((await text(p.page, '#models-line')).includes('api.meta.ai'), 'the models line shows the Meta API host');
    await p.ctx.close();
    const c = clone(live);
    c.run.endpoint = { kind: 'custom', hosts: ['models.partner.example'] };
    p = await openPage(browser, srv, c);
    check((await text(p.page, '#badge-backend')).includes('LIVE') && (await text(p.page, '#badge-backend')).includes('models.partner.example'), 'a custom host is LIVE but the badge shows the host');
    check((await text(p.page, '#models-line')).includes('custom host models.partner.example'), 'and the models line says it is a custom host');
    check(!(await visible(p.page, '#banner')), 'a custom host is not a simulation, so no banner');
    await p.ctx.close();
    const o = clone(live);
    delete o.run.endpoint; delete o.run.origin;
    p = await openPage(browser, srv, o);
    check((await text(p.page, '#badge-backend')).includes('LIVE') && !(await visible(p.page, '#banner')), 'an older ledger with no endpoint record stays LIVE');
    await p.ctx.close();
  });

  // ---- a real scripted run (regenerated with the CLI) -----------------------------------------------
  await scenario('real scripted run renders sensibly', async () => {
    const { ctx, page, log } = await openPage(browser, srv, real);
    await sleep(400);
    check((await count(page, '#lane-grid .lane-card')) === real.experiments.length && real.experiments.length >= 12, 'one card per experiment (12 in the CLI run)');
    check((await count(page, '#findings-list .finding')) === real.findings.length && real.findings.length >= 8, 'every finding listed');
    check((await count(page, '#claims-list .claim')) === real.claims.length && real.claims.length >= 13, 'every claim listed');
    check((await count(page, '#turns-list .turn')) === real.turns.length && real.turns.length >= 10 && (await text(page, '#turns-list')).includes('script narration'), 'every turn, labelled script narration');
    check((await count(page, '#tl-list .tl-row')) === real.timeline.length, 'the whole timeline is rendered');
    check((await text(page, '#verdict')).includes('Accepted improvement') && (await text(page, '#verdict .v-summary')).includes('damped sinusoid'), 'verdict card with the run summary');
    check((await count(page, '#verdict .v-claim')) === 1, 'the verdict card lists the one accepted claim');
    const tiles = await page.evaluate(() => [...document.querySelectorAll('#inv-grid .inv-tile')].map((t) => t.getAttribute('data-arg') + ':' + t.className.match(/tone-(\w+)/)[1] + ':' + t.textContent.includes('LLM not run')));
    const llm = tiles.filter((t) => t.endsWith(':true')).map((t) => t.split(':')[0]).sort();
    check(JSON.stringify(llm) === JSON.stringify(['I07', 'I08', 'I13', 'I14']), 'I07, I08, I13, I14 read LLM not run, got ' + JSON.stringify(llm));
    check(tiles.filter((t) => t.includes(':ok:')).length === 10 && tiles.every((t) => !(t.includes(':ok:') && t.endsWith(':true'))), 'the other ten read ok and none of the LLM ones is green');
    const chart = await text(page, '#chart-wrap');
    check(chart.includes('532.322 (off scale)'), 'the runaway overfit score is drawn on the edge with its true value');
    check((await count(page, '#chart-wrap .sh-off')) === 1, 'one off-scale marker');
    const yTicks = await page.evaluate(() => [...document.querySelectorAll('#chart-wrap .tick:not(.tick-x)')].map((n) => parseFloat(n.textContent)));
    check(Math.max(...yTicks) < 5, 'the axis follows the bulk of the data, got ticks ' + JSON.stringify(yTicks));
    const fi = await text(page, '#sec-faults');
    const rc = real.fault_injection.recall;
    const pf = real.fault_injection.planted.find((p) => p.fault === 'premature_finish');
    const defect = real.fault_injection.planted.find((p) => p.fault === 'defect');
    const inj = real.fault_injection.planted.find((p) => p.fault === 'injection');
    check(fi.includes('Recall ' + Math.round(100 * rc.value) + '% (' + rc.caught + ' of ' + rc.scored + ' scored)'), 'recall line from the scored faults, got ' + fi.slice(0, 300));
    check(fi.includes(defect.exp_id + ' defect') && defect.result === 'detected' && fi.includes('reviewer found the planted bug; repaired and re-reviewed'), 'the planted defect is detected by the reviewer');
    check(fi.includes('premature finish @' + pf.finish_seq) && pf.result === 'detected', 'the premature finish is a detected planted fault');
    check(fi.includes(inj.exp_id + ' injection') && inj.result === 'not_evaluated', 'the injection is listed');
    check((await text(page, '#faults-lists')).includes('Not evaluated (1)') && (await text(page, '#faults-lists')).includes('Missed (0)'), 'the injection is not evaluated, not missed');
    check((await text(page, '#faults-lists')).includes('Explained by a planted premature finish (' + real.fault_injection.explained.length + ')') && real.fault_injection.explained.length === 2, 'two findings explained by the premature finish');
    check((await text(page, '#faults-lists')).includes('False positives (' + real.fault_injection.false_positives.length + ')'), 'false positives counted');
    const lanes = await text(page, '#lane-grid');
    check(lanes.includes('blocked: duplicate') && lanes.includes('replicated') && lanes.includes('failed replication'), 'blocked, replicated and failed-replication lanes');
    check(!/[−-]0\.0000/.test(await text(page, '#lane-grid')), 'no negative zero gain on a card');
    const dup = real.experiments.find((e) => e.blocked_reason === 'duplicate');
    check(!!dup && (await page.evaluate((id) => document.querySelectorAll('#lane-grid .lane-card[data-key="' + id + '"] .step.is-failed').length, dup.exp_id)) === 0, 'a duplicate stopped by the gate is not drawn as a failed stage');
    await page.screenshot({ path: path.join(SHOTS, 'real-desktop-light.png'), fullPage: true });
    await (await page.$('#sec-chart')).screenshot({ path: path.join(SHOTS, 'real-chart.png') });
    await (await page.$('#sec-faults')).screenshot({ path: path.join(SHOTS, 'real-fault-scoreboard.png') });
    await cleanLog('real run', log, page);
    await ctx.close();
  });

  await scenario('a missed fault is listed as missed and lowers recall', async () => {
    const s = clone(scripted);
    const fi = s.fault_injection;
    const m = fi.detected.find((d) => d.fault === 'tamper');
    m.result = 'missed'; m.reason = 'no matching finding';
    fi.detected = fi.detected.filter((d) => d !== m); fi.missed = [m];
    fi.recall = { caught: 6, scored: 7, value: 6 / 7, not_evaluated: 1, negatives_ok: 1, negatives_total: 1 };
    const { ctx, page, log } = await openPage(browser, srv, s);
    await sleep(300);
    const t = await text(page, '#sec-faults');
    check(t.includes('Missed (1)') && t.includes('e003 tamper') && t.includes('Recall 86% (6 of 7 scored)'), 'missed group and recall, got ' + t.slice(0, 300));
    await cleanLog('missed', log, page);
    await ctx.close();
  });

  await scenario('while the run is going an uncaught fault is "not caught yet", never a final miss', async () => {
    const s = clone(scripted);
    const fi = s.fault_injection;
    const m = fi.detected.find((d) => d.fault === 'tamper');
    m.result = 'missed'; m.reason = 'no matching finding';
    fi.detected = fi.detected.filter((d) => d !== m); fi.missed = [m];
    fi.recall = { caught: 6, scored: 7, value: 6 / 7, not_evaluated: 1, negatives_ok: 1, negatives_total: 1 };
    s.run.finished = false; s.run.reason = null; s.run.verdict_class = null;
    const { ctx, page, log } = await openPage(browser, srv, s);
    await sleep(300);
    const t = await text(page, '#sec-faults');
    check(t.includes('Not caught yet (1)') && !/Missed \(/.test(t) && !t.includes('Missed'), 'running run says not caught yet, got ' + t.slice(0, 400));
    check(t.includes('Provisional recall 86%') && t.includes('may still be caught by a later audit window'), 'recall is marked provisional');
    await cleanLog('running misses', log, page);
    await ctx.close();
  });

  await scenario('a meta backend whose endpoint record says scripted is never badged LIVE', async () => {
    const s = clone(live);
    s.run.mode = 'unknown'; s.run.origin = 'unknown'; s.run.endpoint = { kind: 'scripted', hosts: [] }; s.banner = null;
    const { ctx, page, log } = await openPage(browser, srv, s);
    await sleep(300);
    const b = await text(page, '#badge-backend');
    check(b.includes('UNVERIFIED') && !b.includes('LIVE'), 'badge is UNVERIFIED, got ' + b);
    check((await text(page, '#models-line')).includes('source of this text is not established'), 'models line says the source is not established');
    await cleanLog('unknown origin', log, page);
    await ctx.close();
  });

  await scenario('fault scoreboard hostile text stays inert', async () => {
    const s = clone(real);
    s.fault_injection.not_evaluated[0].reason = PWN;
    s.fault_injection.explained[0].reason = PWN;
    s.fault_injection.false_positives.push({ finding_id: 'I11@run', invariant_id: 'I11', exp_id: null, reason: PWN });
    s.fault_injection.planted[0].reason = PWN;
    s.run.endpoint = { kind: 'custom', hosts: [PWN] };
    s.run.origin = 'live'; s.run.mode = 'live'; s.run.backend = 'meta'; s.banner = null;
    const { ctx, page, log } = await openPage(browser, srv, s);
    await sleep(400);
    check((await count(page, 'img')) === 0 && (await page.evaluate(() => window.__pwned)) === undefined, 'no element or handler was created');
    check((await text(page, '#faults-lists')).includes(PWN) && (await text(page, '#models-line')).includes(PWN), 'the markup is literal text');
    await cleanLog('fault hostile', log, page);
    await ctx.close();
  });


  // ---- cost per experiment, run cost, implementer, parallel reason -----------------------------------
  await scenario('costs: lane cards, detail strip, tooltip and headline; nothing for a scripted run', async () => {
    const { ctx, page, log } = await openPage(browser, srv, live);
    const cards = await page.evaluate(() => [...document.querySelectorAll('#lane-grid .lane-card')].map((c) => ({ id: c.dataset.key, hidden: c.querySelector('.lc-cost').hidden, text: c.querySelector('.lc-cost').textContent })));
    const by = Object.fromEntries(cards.map((c) => [c.id, c]));
    check(cards.length === 7 && cards.filter((c) => !c.hidden).length === 6 && by.e000.hidden, 'six experiments show a cost, the baseline (no model calls) shows none');
    const e1 = live.experiments.find((e) => e.exp_id === 'e001');
    check(by.e001.text.includes('$0.061') && by.e001.text.includes('designer $0.009') && by.e001.text.includes('implementer $0.034') && by.e001.text.includes('reviewer $0.018') && by.e001.text.includes('3 calls') && by.e001.text.includes(String(e1.tokens).replace(/(\d)(\d{3})$/, '$1,$2')), 'card cost shows the total, the split by role, tokens and calls, got ' + by.e001.text);
    check(/\$0\.009/.test(by.e006.text) && !by.e006.text.includes('implementer'), 'an experiment with only a design call shows only that');
    const hl = await text(page, '#headline');
    check(hl.includes('LLM cost') && !hl.includes('Simulated') && hl.includes('$0.41') && hl.includes('21 model calls') && hl.includes('$0.29 tied to experiments, $0.12 to the supervisor, auditor and others'), 'headline cost, calls and attribution, got ' + hl);
    check(hl.includes('Cost per promoted improvement') && hl.includes('divided by 1 improvement that passed the sealed replication'), 'headline cost per promoted improvement');
    await page.click('#chart-wrap .pt[data-arg="e001"]');
    check((await text(page, '#chart-detail')).includes('Model cost') && (await text(page, '#chart-detail .cd-metrics')).includes('$0.061'), 'the detail strip shows the model cost');
    await page.hover('#chart-wrap .pt[data-arg="e002"]');
    check((await text(page, '#chart-tip')).includes('model cost $0.061'), 'the tooltip shows the model cost');
    await page.click('#chart-wrap .pt[data-arg="e000"]');
    check(await page.evaluate(() => [...document.querySelectorAll('#chart-detail .cd-metrics > div')].filter((d) => d.textContent.includes('Model cost')).every((d) => d.hidden)), 'the baseline detail strip hides the cost');
    check((await text(page, '#usage-body')).includes('$0.051'), 'the models table keeps its USD column');
    await page.evaluate(() => document.querySelector('#sec-lanes').scrollIntoView());
    await sleep(150);
    await (await page.$('#sec-lanes')).screenshot({ path: path.join(SHOTS, 'cost-lanes.png') });
    await (await page.$('#sec-headline')).screenshot({ path: path.join(SHOTS, 'cost-headline.png') });
    await cleanLog('costs', log, page);
    await ctx.close();

    const sc = await openPage(browser, srv, scripted);
    check((await count(sc.page, '#lane-grid .lane-card .lc-cost:not([hidden])')) === 0, 'a scripted run shows no cost on any card');
    const sh = await text(sc.page, '#headline');
    check(!/cost/i.test(sh) && !sh.includes('$'), 'a scripted run has no cost in the headline (not $0), got ' + sh);
    const bud = await text(sc.page, '#budget');
    check(bud.includes('none · scripted') && !/\$0/.test(bud.replace(/\$5\.00/, '')), 'the scripted LLM spend meter says none, not $0.00, got ' + bud);
    const rl = await openPage(browser, srv, real);
    check(!/cost/i.test(await text(rl.page, '#headline')) && (await count(rl.page, '#lane-grid .lc-cost:not([hidden])')) === 0, 'the regenerated scripted CLI run shows no cost either');
    await sc.ctx.close(); await rl.ctx.close();
  });

  await scenario('costs: a recorded simulated run attributes spend and is labelled simulated', async () => {
    const { ctx, page, log } = await openPage(browser, srv, simReal);
    await sleep(300);
    const hl = await text(page, '#headline');
    check((await text(page, '#budget')).includes('LLM spend (simulated list-price estimate)'), 'the spend meter of a simulated run says it is a simulated list-price estimate');
    check(hl.includes(SIM_COST) && !(await text(page, '#budget')).includes('LLM spend (simulated)'), 'the cost card carries the exact simulated-cost label, got ' + hl);
    check(hl.includes('Simulated LLM cost') && hl.includes('nothing was billed') && hl.includes('Simulated cost per promoted improvement'), 'simulated cost is labelled simulated, got ' + hl);
    const shown = await count(page, '#lane-grid .lane-card .lc-cost:not([hidden])');
    check(shown === simReal.experiments.filter((e) => e.cost_usd !== null).length && shown >= 4, 'a cost on every experiment the ledger attributed a charge to, got ' + shown);
    check((await text(page, '#usage-sub')).includes('simulated') && (await text(page, '#usage-sub')).includes('nothing was billed'), 'the models table says the cost is simulated');
    check(!(await text(page, '#badge-backend')).includes('LIVE'), 'still not labelled LIVE');
    await page.screenshot({ path: path.join(SHOTS, 'sim-real-desktop-light.png'), fullPage: true });
    await cleanLog('simulated real', log, page);
    await ctx.close();
  });

  await scenario('header: effective implementer and the parallel reason', async () => {
    const sc = await openPage(browser, srv, scripted);
    check(await visible(sc.page, '#run-implementer-wrap') && (await text(sc.page, '#run-implementer')) === 'scripted', 'a scripted run says implementer scripted');
    await sc.ctx.close();
    const m = clone(live);
    m.run.implementer = 'muse_code'; m.headline.speedup = 1.0;
    m.parallel = { wall_sec: 100, serial_sec: 100, speedup: 1.0, max_parallel: 1, reason: 'the implementer requires serial execution: max_parallel capped from 3 to 1' };
    const p = await openPage(browser, srv, m);
    check((await text(p.page, '#run-implementer')) === 'muse_code', 'muse_code shown');
    const sp = await text(p.page, '#headline');
    check(sp.includes('1.00×') && sp.includes('up to 1 lane at once') && sp.includes('the implementer requires serial execution: max_parallel capped from 3 to 1'), 'the parallel reason sits with the speedup figure, got ' + sp);
    await cleanLog('muse_code', p.log, p.page);
    await p.ctx.close();
    const pt = await openPage(browser, srv, live);
    check((await text(pt.page, '#run-implementer')) === 'patch' && !(await text(pt.page, '#headline')).includes('requires serial'), 'patch shown and no reason when none was recorded');
    await pt.ctx.close();
  });

  await scenario('models line names the model that answered when it differs from the configured one', async () => {
    const f = clone(live);
    f.run.models_answered = { reviewer_hard: ['muse-spark-1.3'], supervisor: ['muse-spark-1.3'], designer: ['muse-spark-1.3', 'muse-wonder-1.0'] };
    const p = await openPage(browser, srv, f);
    const ml = await text(p.page, '#models-line');
    check(ml.includes('reviewer_hard Muse Wonder 1.0 (answered by Muse Spark 1.3)'), 'a fallback answer is named, got ' + ml);
    check(ml.includes('designer Muse Spark 1.3 (answered by Muse Spark 1.3, Muse Wonder 1.0)') && !ml.includes('supervisor Muse Spark 1.3 (answered'), 'several answers listed, an equal one not repeated');
    const g = clone(live);
    g.run.models_answered = { supervisor: ['muse-spark-1.2'] };
    const q = await openPage(browser, srv, g);
    check((await text(q.page, '#badge-backend')).includes('Muse Spark 1.2'), 'the LIVE badge names the supervisor model that answered, got ' + (await text(q.page, '#badge-backend')));
    await p.ctx.close(); await q.ctx.close();
  });

  await scenario('replay honesty: a replay says what it replays, keeps its banner, never says live for a scripted run', async () => {
    for (const [name, st, origin, banner] of [['scripted', scripted, 'scripted', true], ['simulated', simulated, 'simulated', true], ['live', live, 'live', false]]) {
      const r = clone(st);
      r.run.mode = 'replay'; r.run.finished = false;
      r.replay = { speed: 10, position_seq: 20, total_seq: 90, source_run_id: 'r-replayed1' };
      const p = await openPage(browser, srv, r);
      const b = await text(p.page, '#badge-backend');
      check(b.includes('REPLAY') && b.includes('of r-replayed1 (' + origin + ')'), name + ': badge reads REPLAY of <run> (' + origin + '), got ' + b);
      check(!b.includes('LIVE') || origin === 'live', name + ': a replay of a ' + origin + ' run never says LIVE');
      check((await visible(p.page, '#banner')) === banner, name + ': banner ' + (banner ? 'kept' : 'absent'));
      check((await text(p.page, '#replay-text')).startsWith('of r-replayed1 (' + origin + ')') && (await text(p.page, '#replay-text')).includes('10× speed'), name + ': replay bar names the run and origin');
      if (name === 'scripted') await p.page.screenshot({ path: path.join(SHOTS, 'replay-scripted.png'), clip: { x: 0, y: 0, width: 1440, height: 330 } });
      await cleanLog('replay ' + name, p.log, p.page);
      await p.ctx.close();
    }
    const o = clone(live);                       // an older state: replay with no origin key at all
    o.run.mode = 'replay'; delete o.run.origin; o.replay = { speed: 2, position_seq: 5, total_seq: 50, source_run_id: 'r-old' };
    const q = await openPage(browser, srv, o);
    check((await text(q.page, '#badge-backend')).includes('of r-old') && !(await text(q.page, '#badge-backend')).includes('()'), 'no origin key: no empty parentheses');
    await q.ctx.close();
  });

  await scenario('older states without any of the new keys render quietly', async () => {
    for (const [name, st] of [['scripted', scripted], ['live', live], ['simulated-real', simReal]]) {
      const o = clone(st);
      delete o.run.implementer; delete o.run.models_answered; delete o.parallel.reason; delete o.parallel.max_parallel;
      for (const k of ['cost_known', 'llm_calls', 'cost_attributed_usd', 'cost_unattributed_usd', 'promoted', 'cost_per_improvement_usd']) delete o.headline[k];
      for (const e of o.experiments) { delete e.cost_usd; delete e.tokens; delete e.llm_calls; delete e.cost_by_role; }
      const p = await openPage(browser, srv, o);
      await sleep(300);
      check((await count(p.page, '#lane-grid .lane-card .lc-cost:not([hidden])')) === 0, name + ': no card cost without the keys');
      check(!(await visible(p.page, '#run-implementer-wrap')), name + ': no implementer without the key');
      const hl = await text(p.page, '#headline');
      check(name === 'scripted' ? !hl.includes('$') : (hl.includes('$') && !hl.includes('per promoted improvement')), name + ': headline falls back to the run total only, got ' + hl);
      await cleanLog('old state ' + name, p.log, p.page);
      await p.ctx.close();
    }
    const bare = { run: { run_id: 'r-bare', task: 't', backend: 'meta', mode: 'live', models: {}, started_ts: 1, now: 2, finished: false, objective: 'minimize', primary_metric: 'm', isolation: '', resumes: [] }, headline: {}, parallel: {}, experiments: [{ exp_id: 'e001', status: 'proposed' }] };
    const p = await openPage(browser, srv, bare);
    check((await count(p.page, '#lane-grid .lane-card')) === 1 && (await count(p.page, '#lane-grid .lc-cost:not([hidden])')) === 0, 'a minimal state with one bare experiment renders');
    await cleanLog('bare state', p.log, p.page);
    await p.ctx.close();
  });

  await scenario('garbage and hostile values in the cost and config fields stay inert', async () => {
    const g = clone(live);
    const SHORT = '<img src=x onerror=1>';
    g.run.implementer = SHORT; g.run.models_answered = { supervisor: [PWN, 5, null], [PWN]: [PWN], designer: 'x' };
    g.parallel = { speedup: 1, wall_sec: 'x', serial_sec: null, max_parallel: 'many', reason: PWN };
    g.headline.cost_usd = 'NaN'; g.headline.cost_known = 'yes'; g.headline.llm_calls = {}; g.headline.cost_attributed_usd = [1]; g.headline.promoted = 'x'; g.headline.cost_per_improvement_usd = 'cheap';
    g.experiments[1].cost_usd = 0.5; g.experiments[1].tokens = 'lots'; g.experiments[1].llm_calls = null; g.experiments[1].cost_by_role = { [PWN]: 0.25, b: 'x', c: null };
    g.experiments[2].cost_usd = 'x'; g.experiments[2].cost_by_role = 7;
    g.experiments[3].cost_usd = -0.0004;
    g.experiments[4].cost_usd = 1e9;
    const p = await openPage(browser, srv, g);
    await sleep(400);
    check((await count(p.page, 'img')) === 0 && (await p.page.evaluate(() => window.__pwned)) === undefined, 'no element or handler created');
    check((await text(p.page, '#run-implementer')) === SHORT && (await text(p.page, '#headline')).includes(PWN), 'markup in the new fields is literal text');
    check((await text(p.page, '#lane-grid')).includes('$0.500') || (await text(p.page, '#lane-grid')).includes('$0.50'), 'a numeric cost still renders next to garbage siblings');
    check(await noHScroll(p.page), 'no horizontal scroll');
    await cleanLog('cost garbage', p.log, p.page);
    await p.ctx.close();
  });

  await scenario('a flood of cost roles cannot blow up a lane card', async () => {
    const g = clone(live);
    const roles = {};
    for (let i = 0; i < 300; i++) roles['role' + String(i).padStart(3, '0') + 'x'.repeat(100)] = 0.01;
    g.experiments[1].cost_usd = 3; g.experiments[1].cost_by_role = roles;
    const p = await openPage(browser, srv, g);
    await sleep(300);
    const detail = await p.page.evaluate(() => Math.max(...Array.from(document.querySelectorAll('.lc-cost-d')).map((n) => n.textContent.length)));
    check(detail > 0 && detail < 600, 'the per-role cost line stays short (' + detail + ' chars)');
    check(await noHScroll(p.page), 'no horizontal scroll with 300 cost roles');
    await cleanLog('cost role flood', p.log, p.page);
    await p.ctx.close();
  });

  // ---- live sample ----------------------------------------------------------------------------
  await scenario('live sample: badges, in-flight lanes, no scripted banner', async () => {
    const { ctx, page, log } = await openPage(browser, srv, live);
    check((await text(page, '#badge-backend')).includes('LIVE') && (await text(page, '#badge-backend')).includes('Muse Spark'), 'LIVE Muse Spark badge');
    check((await text(page, '#badge-status')).includes('RUNNING'), 'running badge');
    check(!(await visible(page, '#banner')), 'no scripted banner on a live run');
    check(!(await visible(page, '#sec-faults')), 'fault scoreboard hidden when nothing was planted');
    check((await text(page, '#lane-grid')).includes('running: run'), 'in-flight lane shows its stage');
    check((await count(page, '#lane-grid .step.is-active')) === 2, 'two lanes have an active step');
    check((await count(page, '#usage-body tr')) === 5 && (await text(page, '#usage-body')).includes('69%'), 'models usage table with cache hit rate when known');
    check((await text(page, '#turns-list')).includes('Muse Spark 1.3') && !(await text(page, '#turns-list')).includes('script narration'), 'turns are attributed to the model');
    check((await text(page, '#turns-list')).includes('effort medium') && (await text(page, '#turns-list')).includes('2.1 s'), 'reasoning effort and latency');
    check((await text(page, '#verdict')).includes('No verdict yet'), 'verdict card has an intentional empty state while running');
    const inv = await text(page, '#inv-grid');
    check(inv.includes('insufficient evidence') && inv.includes('LLM not run'), 'insufficient and LLM-not-run tiles');
    check((await text(page, '#claims-list')).includes('unreplicated'), 'unreplicated claim chip');
    await page.screenshot({ path: path.join(SHOTS, 'live-desktop.png'), fullPage: true });
    await cleanLog('live', log, page);
    await ctx.close();
  });

  // ---- polling, in-place updates, selection and scroll kept ------------------------------------
  await scenario('polling cadence and in-place updates', async () => {
    const { ctx, page, log } = await openPage(browser, srv, live);
    const t0 = srv.ctl.hits.length;
    await sleep(3300);
    const running = srv.ctl.hits.length - t0;
    check(running >= 2 && running <= 5, 'polls about once a second while running, got ' + running + ' in 3.3 s');
    await ctx.close();

    const fin = await openPage(browser, srv, scripted);
    const f0 = srv.ctl.hits.length;
    await sleep(3300);
    check(srv.ctl.hits.length - f0 <= 1, 'polls slowly once the run finished, got ' + (srv.ctl.hits.length - f0));
    await fin.ctx.close();

    const o = await openPage(browser, srv, live);
    await o.page.click('#lane-grid .lane-card[data-key="e001"] .lc-select');
    await o.page.evaluate(() => { document.querySelector('#lane-grid .lane-card[data-key="e001"]').__same = true; document.querySelector('#tl-list').scrollTop = 0; });
    await sleep(200);
    const next = clone(live);
    next.headline.cost_usd = 9.99;
    next.timeline.push({ seq: 999, ts: next.timeline[next.timeline.length - 1].ts + 5, kind: 'run.resumed', actor: 'engine', text: 'a late event' });
    srv.ctl.body = JSON.stringify(next);
    await o.page.waitForFunction(() => document.querySelector('#headline').textContent.includes('$9.99'), null, { timeout: 4000 });
    check(await o.page.evaluate(() => document.querySelector('#lane-grid .lane-card[data-key="e001"]').__same === true), 'the lane card is patched in place, not rebuilt');
    check((await text(o.page, '#lane-grid .lane-card.is-selected .lc-id')) === 'e001', 'selection survives an update');
    check((await o.page.evaluate(() => document.querySelector('#tl-list').scrollTop)) === 0, 'timeline scroll position survives an update');
    check((await count(o.page, '#ev-999')) === 1, 'the new event was appended');
    await cleanLog('in place', o.log, o.page);
    await o.ctx.close();
  });

  await scenario('pauses while the tab is hidden', async () => {
    const { ctx, page } = await openPage(browser, srv, live);
    await page.evaluate(() => { Object.defineProperty(document, 'hidden', { configurable: true, get: () => true }); document.dispatchEvent(new Event('visibilitychange')); });
    await sleep(1300);
    const h0 = srv.ctl.hits.length;
    await sleep(2500);
    check(srv.ctl.hits.length === h0, 'no polling while hidden, got ' + (srv.ctl.hits.length - h0) + ' requests');
    await page.evaluate(() => { Object.defineProperty(document, 'hidden', { configurable: true, get: () => false }); document.dispatchEvent(new Event('visibilitychange')); });
    await sleep(600);
    check(srv.ctl.hits.length > h0, 'polling resumes when the tab is visible again');
    await ctx.close();
  });

  await scenario('disconnected state with retry and recovery', async () => {
    const { ctx, page, log } = await openPage(browser, srv, live);
    srv.ctl.status = 500;
    await page.waitForFunction(() => document.querySelector('#conn').textContent.includes('Disconnected'), null, { timeout: 5000 });
    check((await text(page, '#conn')).includes('retrying in'), 'shows the retry countdown');
    check((await count(page, '#lane-grid .lane-card')) === 7, 'last good data stays on screen');
    check(await page.evaluate(() => document.body.classList.contains('is-disconnected')), 'body marked disconnected');
    await page.screenshot({ path: path.join(SHOTS, 'disconnected.png') });
    srv.ctl.status = 200;
    await page.click('#conn button');
    await page.waitForFunction(() => document.querySelector('#conn').textContent.includes('Connected'), null, { timeout: 5000 });
    check(!(await page.evaluate(() => document.body.classList.contains('is-disconnected'))), 'recovers without a reload');
    check(log.pageerrors.length === 0, 'no uncaught errors while disconnected');
    const unexpected = log.console.filter((m) => !/Failed to load resource/.test(m));
    check(unexpected.length === 0, 'no unexpected console output while disconnected, got ' + JSON.stringify(unexpected));
    await ctx.close();

    // a 200 with a body that is not a state object is also a disconnect, never a crash
    const bad = await openPage(browser, srv, '{"hello": "world"}', { wait: '#conn' });
    await bad.page.waitForFunction(() => document.querySelector('#conn').textContent.includes('Disconnected'), null, { timeout: 5000 });
    check(bad.log.pageerrors.length === 0 && bad.log.console.length === 0, 'a malformed state is handled quietly');
    await bad.ctx.close();
  });

  // ---- empty and garbage states -----------------------------------------------------------------
  await scenario('empty run: every panel has an intentional empty state', async () => {
    const empty = { run: { run_id: 'r-empty', task: 'law-discovery', backend: 'meta', mode: 'live', models: {}, started_ts: 1, now: 2, finished: false, reason: null, verdict_class: null, objective: 'maximize', primary_metric: 'acc', isolation: '', resumes: [] } };
    const { ctx, page, log } = await openPage(browser, srv, empty, { wait: '#lane-empty:not([hidden])' });
    for (const id of ['#chart-empty', '#lane-empty', '#inv-empty', '#turns-empty', '#claims-empty', '#finish-empty', '#pivots-empty', '#lessons-empty', '#usage-empty', '#tl-empty']) {
      check(await visible(page, id), 'empty state visible: ' + id);
    }
    check((await text(page, '#chart-dir')).includes('higher is better'), 'maximize objective labelled higher is better');
    check((await text(page, '#badge-ledger')).includes('chain not verified'), 'null chain reads chain not verified');
    check(!(await visible(page, '#sec-faults')) && !(await visible(page, '#banner')), 'no faults panel and no banner');
    check(await noHScroll(page), 'empty page has no horizontal scroll');
    await cleanLog('empty', log, page);
    await ctx.close();
  });

  await scenario('garbage field types do not crash any panel', async () => {
    const g = clone(scripted);
    g.experiments = [null, 5, 'x', { exp_id: 7 }, { exp_id: 'e1', hypothesis: { a: 1 }, protected: 'nope', claim_ids: 4, finding_ids: [null, 1], status: 9, score: 'NaN' }, ...g.experiments.slice(0, 3)];
    g.series = [null, { exp_id: 'e1', score: 'x' }, { exp_id: 'e000', score: 0.4, std: -1, kind: 'weird' }];
    g.timeline = [{ seq: 'a' }, { seq: 1, kind: 5, text: null }, { seq: 1, kind: 'run.started', text: 'dup seq' }];
    g.turns = [{ tool_calls: 'x' }, { tool_calls: [null, 4, { name: {}, args_summary: [] }] }];
    g.lessons = [3, null, { text: 5, source_exps: 'e1' }];
    g.findings = [{ finding_id: 'x', evidence_seqs: ['a', 1e99, null] }];
    g.budget = { limits: { max_experiments: 0 }, used: { experiments: 'many' } };
    g.ledger = { chain_ok: false, bad_seq: 12 };
    g.headline = { claims_made: 'x', speedup: null };
    const { ctx, page, log } = await openPage(browser, srv, g, { wait: '#lane-grid .lane-card' });
    await sleep(400);
    check((await text(page, '#badge-ledger')).includes('chain broken at seq 12'), 'broken chain shown in red words');
    check(await page.evaluate(() => document.querySelector('#badge-ledger').classList.contains('tone-bad')), 'broken chain badge is the bad tone');
    check((await count(page, '#lane-grid .lane-card')) >= 3, 'valid experiments still render');
    check(log.pageerrors.length === 0, 'no uncaught errors, got ' + JSON.stringify(log.pageerrors));
    check(log.console.length === 0, 'no console errors (every panel survived), got ' + JSON.stringify(log.console));
    await ctx.close();
  });

  await scenario('replay mode', async () => {
    const r = clone(scripted);
    r.run.mode = 'replay';
    r.run.finished = false;
    r.replay = { speed: 4, position_seq: 128, total_seq: 342, source_run_id: 'r-7f3a9c21' };
    const { ctx, page, log } = await openPage(browser, srv, r);
    check((await text(page, '#badge-backend')).includes('REPLAY') && (await text(page, '#badge-backend')).includes('chain verified'), 'REPLAY badge says chain verified');
    check(await visible(page, '#replay-bar') && (await text(page, '#replay-text')).includes('128') && (await text(page, '#replay-text')).includes('342'), 'position and total shown');
    check((await page.getAttribute('#replay-bar .replay-track', 'aria-valuenow')) === '37', 'progress bar value');
    check(await visible(page, '#banner'), 'a replay of a scripted run keeps the scripted banner');
    check((await text(page, '#badge-backend')).includes('of r-7f3a9c21 (scripted)'), 'the replay badge names the replayed run and its origin, got ' + (await text(page, '#badge-backend')));
    check((await text(page, '#replay-text')).startsWith('of r-7f3a9c21 (scripted)'), 'so does the replay bar, got ' + (await text(page, '#replay-text')));
    await cleanLog('replay', log, page);
    await ctx.close();
  });

  await scenario('ledger states: broken, unverified', async () => {
    const b = clone(live);
    b.ledger.chain_ok = false; b.ledger.bad_seq = 41;
    let p = await openPage(browser, srv, b);
    check((await text(p.page, '#badge-ledger')).includes('chain broken at seq 41'), 'chain broken at seq N');
    check(!(await text(p.page, '#badge-ledger')).includes('consistent'), 'a broken chain never reads consistent');
    await p.ctx.close();
    const u = clone(live);
    u.ledger.chain_ok = null; u.ledger.head_anchor = '';
    p = await openPage(browser, srv, u);
    check((await text(p.page, '#badge-ledger')).includes('chain not verified'), 'null reads chain not verified');
    await p.ctx.close();
    const n = clone(live);
    n.run.isolation = 'none';
    p = await openPage(browser, srv, n);
    check((await text(p.page, '#badge-isolation')).includes('isolation: none'), 'isolation none is shown, not hidden');
    await p.ctx.close();
  });

  // ---- regressions found by the adversarial pass ---------------------------------------------------
  await scenario('a different run served at the same URL replaces the stale timeline, selection and cards', async () => {
    const { ctx, page, log } = await openPage(browser, srv, scripted);
    await page.click('#chart-wrap .pt[data-arg="e005"]');
    const other = clone(scripted);
    other.run.run_id = 'r-another-run';
    other.timeline.forEach((t) => { t.text = 'OTHERRUN event ' + t.seq; });
    other.experiments.forEach((e) => { e.hypothesis = 'OTHERRUN hypothesis ' + e.exp_id; });
    other.turns.forEach((t) => { t.say = 'OTHERRUN narration'; });
    srv.ctl.body = JSON.stringify(other);
    await page.waitForFunction(() => document.querySelector('#run-id').textContent === 'r-another-run', null, { timeout: 9000 });
    const stale = await page.evaluate(() => ({
      rows: [...document.querySelectorAll('#tl-list .tl-row')].filter((r) => !r.textContent.includes('OTHERRUN')).length,
      cards: [...document.querySelectorAll('#lane-grid .lane-card')].filter((r) => !r.textContent.includes('OTHERRUN')).length,
      turns: [...document.querySelectorAll('#turns-list .turn')].filter((r) => !r.textContent.includes('OTHERRUN')).length,
      selected: document.querySelectorAll('.is-selected:not(.inv-tile)').length,
    }));
    check(stale.rows === 0 && stale.cards === 0 && stale.turns === 0, 'no row, card or turn of the previous run survives: ' + JSON.stringify(stale));
    check(stale.selected === 0, 'the selection of the previous run is cleared');
    check((await count(page, '#tl-list .tl-row')) === 173, 'the new timeline is complete');
    await cleanLog('run switch', log, page);
    await ctx.close();
  });

  await scenario('state values that name Object.prototype members stay inert', async () => {
    const s = clone(scripted);
    s.invariants[0].state = 'constructor'; s.invariants[1].state = '__proto__'; s.invariants[1].kind = 'toString';
    s.claims[0].status = 'toString'; s.claims[1].status = 'hasOwnProperty';
    s.experiments[1].status = 'constructor'; s.experiments[1].outcome = null; s.experiments[1].stage = '__proto__';
    s.series[1].kind = 'constructor';
    const { ctx, page, log } = await openPage(browser, srv, s);
    await sleep(300);
    check(!/undefined|\[native|function/.test(await text(page, '#inv-grid')), 'invariant tiles never print undefined or a function');
    check(!/undefined|\[native|function/.test(await text(page, '#claims-list')), 'claim chips never print undefined or a function');
    check((await count(page, '#lane-grid .lane-card')) === 14, 'every lane card still renders');
    await cleanLog('proto keys', log, page);
    await ctx.close();
  });

  await scenario('scores below float resolution cannot hang the chart axis', async () => {
    const s = clone(scripted);
    s.series.forEach((p, i) => { p.score = 1e15 + (i % 2) * 0.125; p.train = null; p.std = null; });
    s.baseline.score = 1e15;
    const { ctx, page } = await openPage(browser, srv, s, { wait: '#lane-grid .lane-card' });
    const alive = await Promise.race([page.evaluate(() => 'alive'), sleep(6000).then(() => 'hung')]);
    check(alive === 'alive', 'the page stays responsive when tick spacing is below float resolution, got ' + alive);
    if (alive === 'alive') check((await count(page, '#chart-wrap .pt')) === 14, 'the chart still draws its markers');
    await ctx.close().catch(() => {});
  });

  await scenario('keyboard and accessibility basics', async () => {
    const { ctx, page, log } = await openPage(browser, srv, scripted);
    await page.keyboard.press('Tab');
    check(await page.evaluate(() => document.activeElement.classList.contains('skip-link')), 'first Tab stop is the skip link');
    const unnamed = await page.evaluate(() => [...document.querySelectorAll('button')].filter((b) => b.getClientRects().length > 0 && !(b.textContent.trim() || b.getAttribute('aria-label'))).length);
    check(unnamed === 0, 'every rendered button has an accessible name, ' + unnamed + ' do not');
    const noLabel = await page.evaluate(() => [...document.querySelectorAll('.pt')].filter((g) => !g.getAttribute('aria-label')).length);
    check(noLabel === 0, 'every chart marker has an aria-label');
    check((await page.evaluate(() => document.documentElement.lang)) === 'en', 'html lang set');
    check((await page.evaluate(() => [...document.querySelectorAll('.panel')].every((p) => p.getAttribute('aria-labelledby')))), 'every panel is labelled by its heading');
    // status is never colour alone: every invariant tile and pill carries text
    const bare = await page.evaluate(() => [...document.querySelectorAll('.inv-state, .pill')].filter((n) => n.getClientRects().length > 0 && !n.textContent.trim()).length);
    check(bare === 0, 'state chips always carry a text label');
    await cleanLog('a11y', log, page);
    await ctx.close();
  });

  await browser.close();
  srv.server.close();
  console.log('\n' + results.passed + ' checks passed, ' + results.failed.length + ' failed. Screenshots in ' + SHOTS);
  if (results.failed.length) { console.log(results.failed.map((f) => ' - ' + f).join('\n')); process.exit(1); }
  process.exit(0);
}

main().catch((e) => { console.error(e); process.exit(1); });
