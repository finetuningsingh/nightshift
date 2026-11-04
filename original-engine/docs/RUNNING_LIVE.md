# Running Nightshift against the Meta Model API

An operator guide. Everything in this repository was built and tested **without** a Meta API key. The live path (Muse Spark as
supervisor, designer, implementer, reviewer and auditor) is exercised against a faithful fake of the documented Chat Completions
and Responses APIs and against a simulated model brain, over loopback, but never against Meta's servers. The first thing to do
with a key is therefore to run the probes in section 3, which settle the handful of facts the documentation does not.

## 0. What is verified and what is not

| | status |
|---|---|
| The whole pipeline offline: sandbox, trusted grader, sealed split, hash-chained ledger, auditor, gates, dashboard, report, resume after a crash | **verified** by the offline test suite and the scripted demo (`python3 -m nightshift demo`) |
| The statistics (decision rule, error rates) | **verified as a model** (`docs/NULL_BENCHMARK.md`); not re-measured through the sandbox under the calibrated floor |
| The live code path (prompts, tool calls, structured output, metering, audit) over **loopback** against the fake server and a simulated brain | **verified as a SIMULATION**. A run against a fake is labelled `simulated` everywhere and is not Meta model output |
| `--plant` red-team lanes, `--context` background, `NIGHTSHIFT_ISOLATION`, the Muse Code worker against a fake `muse` | **verified against fakes** |
| Any call to `api.meta.ai` (auth, response shapes, strict-schema keyword support, the `max_completion_tokens` ceiling, where cached tokens are reported, rate-limit headers, in-stream error chunks) | **NOT verified**. No key was available |
| The real `muse` binary (`--model` on `exec`, `--sandbox-network restricted`, the `run.usage` fields, trust behaviour) | **NOT verified** |
| The Responses client (`api = "responses"`, replay of encrypted reasoning) | **NOT verified** against the real service; 12 assumptions in `docs/META_API_NOTES.md` |
| Muse Wonder on Chat Completions | **NOT verified** until `doctor --live` confirms it |
| What a run costs, and whether Muse Spark makes progress on the bundled tasks | **unknown**: no live run has happened |
| Isolation on macOS | `none` by construction (no user namespaces, no `/proc`); not exercised on a Mac |

Nothing in the ledger, dashboard or report claims a live run unless the ledger says so: `run.started.config.endpoint.kind` is
`meta-api` and the `budget.charged` rows name the models that answered. A run against a loopback fake says `simulated`; a
`--backend scripted` run says `scripted`; a `--plant` run says which lanes are scripted attackers.

## 1. Get set up

```bash
python3 --version                   # 3.11 or newer
export MODEL_API_KEY=...            # from https://dev.meta.ai  (META_API_KEY also works)
python3 -m nightshift doctor        # python, numpy, isolation probe, muse binary, key set / not set (never printed)
python3 -m nightshift tasks         # the bundled tasks validate and are calibrated (no key needed; tens of seconds per task)
```

`doctor` prints `isolation probe: userns | unshare-net | none`. That is the containment this machine gives the experiments
(section 2); read it before trusting a run with a key. numpy is needed only for the optional spiral-classifier task.

The key is read from the environment, sent only in the `Authorization` header over https, registered with the secret scrubber (it is
stripped from the ledger, logs, reports and error messages), and never passed to experiment code as an environment variable. Whether an
experiment can nevertheless read it from the orchestrator's `/proc/<pid>/environ` depends on the isolation mode: section 2.

Credits: the hackathon terms the author supplied say $150 of Model API credits per accepted participant, **claimed through a form
(application email and Meta Model API team ID) by November 4, 2026, 9:00 PM ET and applied within about two business days**, so claim
them before you need them. This repository cannot verify those terms (`docs/META_API_NOTES.md`, "Hackathon facts"); look at your account's
billing page for the real balance. Nothing in Nightshift depends on it: the budget gate uses the limits you set.

## 2. Keeping the API key and the labels away from experiment code

Experiment code is model-written and runs as your OS user. The mode that applies is chosen by a real probe (`auto`) and can only be
LOWERED with `NIGHTSHIFT_ISOLATION=auto|userns|unshare-net|none` (an invalid value is a usage error, exit 2). The mode actually applied is
`experiment.run_started.isolation` in the ledger and on the dashboard.

| the experiment can ... | `userns` | `unshare-net` | `none` |
|---|---|---|---|
| read `trusted/` (the labels) or the ledger | prevented | **can; not detected** | **can; not detected** |
| read the key from `/proc/<pid>/environ` of the orchestrator | prevented | prevented only in the root-only pid-namespace variant, otherwise as `none` | **can as root**; a non-root same-user process is stopped by `prctl(PR_SET_DUMPABLE, 0)` |
| change `trusted/` (even putting the bytes back) | prevented | **detected afterwards** (content hash plus a stat snapshot; ctime cannot be rolled back) | **detected afterwards** |
| use the network | prevented | prevented | **can; not detected** |

Rules of thumb:

* `userns` is the only mode that keeps the labels and the key out of reach. Unprivileged user namespaces are often disabled in
  containers and CI; check the `doctor` line.
* If you run Nightshift as **root** (typical in a throw-away container), an experiment in `none` (or in `unshare-net` without the pid-namespace
  variant) can read `MODEL_API_KEY`: `prctl(PR_SET_DUMPABLE, 0)` stops a non-root reader of the same user, not root and not `CAP_SYS_PTRACE`.
* If `userns` is unavailable and the key or the sealed numbers matter: run Nightshift as a **non-root** user inside a container or VM, use a
  key scoped to this run, and say in any write-up which mode was active. A run in `unshare-net` or `none` must not be described as sealed-safe.
* **macOS** has no user namespaces and no `/proc`: the mode is `none`, the dumpable hardening does nothing, and a same-user experiment can read
  whatever you can, the key included. Hashes still detect a changed `trusted/`; nothing prevents a read. Use a Linux VM or container for live runs.
* Forcing a weaker mode (`NIGHTSHIFT_ISOLATION=none`) is for testing the fallback, as CI does. The full table with the reasons is in `docs/ARCHITECTURE.md` section 11.

## 3. Probe the API

```bash
python3 -m nightshift doctor --live --live-out demo/live_probe.json   # spends cents; no key and no message content in the file
python3 scripts/live_smoke.py --out demo/live_probe.json              # the same probes without the rest of doctor (--model, --base-url, --no-wonder)
```

`doctor --live` writes the report only with `--live-out` (the script defaults to `demo/live_probe.json`). The probes use `NIGHTSHIFT_BASE_URL`
and `NIGHTSHIFT_MODEL` (defaults: `https://api.meta.ai/v1`, `muse-spark-1.3`), not `nightshift.toml`. They make roughly 40 to 50 requests,
mostly tiny; the two prompt-cache probes send a long prefix (about 4k and 15k tokens) twice each. Required probes cap output at 2000
tokens (the completion-cap probe only asks for 16000, 32000 and 64000 and a trivial reply should use a few tokens), so the whole run is a few cents in practice and about a dollar at the very most on Muse Spark prices if every reply ran to its cap (the Wonder probe costs more per token: $4 / $20 per million). The simulated runs cannot tell you the real figure.

A probe is `pass`, `fail` or `inconclusive`. Muse Spark always reasons and reasoning tokens count against `max_completion_tokens`, so a reply
that was cut off (`finish_reason: length`) and lacks what the probe needs is **inconclusive**, never a failure of the feature. `ok` means no
required probe failed; `required_inconclusive` lists the rest, `conclusive` is false when there are any, and the exit status is 1 only for a
required failure.

What each group settles:

| probe(s) | settles | if the answer is "no" |
|---|---|---|
| `tool_roundtrip` (required) | a tool call plus the tool-result turn works | the supervisor and patch worker cannot run: stop and read the report |
| `strict_schema`, `strict_PLAN/REVIEW/REFLECT/VERDICT`, `supervisor_tools` | strict `json_schema` is accepted for every real agent schema and the six supervisor tool schemas; a rejection lists which JSON-Schema keyword (`minLength`, `pattern`, ...) was refused | structured roles degrade to defensive parsing |
| `effort_minimal` ... `effort_xhigh` (required), `effort_max`, `effort_none_rejected` | every `reasoning_effort`; `max` on `muse-spark-1.3`; `none` is a 400 | the client keeps mapping `none` to `minimal`; a refused `max` is a config problem |
| `stream` (required) | a forced stream returns REAL usage (`stream_options.include_usage`), not an estimate | streamed calls are charged an estimate that cannot see hidden reasoning tokens: the budget understates spend |
| `rejected_param` (required) | the documented error envelope on a deliberate 400 | error handling is guesswork |
| `completion_cap_ceiling` | the real `max_completion_tokens` ceiling (16000, 32000, 64000 are tried) | set `completion_cap_ceiling` per endpoint in `nightshift.toml` |
| `assistant_tool_calls_content`, `parallel_tool_calls`, `tools_with_response_format`, `system_and_developer_roles`, `temperature_with_reasoning` | the message-shape questions the docs leave open | the client adapts or strips the parameter on a 400 |
| `prompt_cache`, `prompt_cache_large`, `ratelimit_headers` | where cached tokens are reported (the minimum cacheable prefix is unstated) and which `x-ratelimit-*` headers (and `Retry-After`) come back | the dashboard shows "cache n/a" and never claims savings |
| `wonder_chat` | does `muse-wonder-1.0` answer on Chat Completions? | the final audit and `reviewer_hard` fall back to `muse-spark-1.3` (at `xhigh`) automatically |
| **Responses probes** (their own `responses` section, never part of `ok`): `responses_plain`, `responses_include_encrypted`, `responses_tool_roundtrip_replay`, `responses_structured`, `responses_stream` | whether the opt-in Responses client works here: `store: false` plus `include: ["reasoning.encrypted_content"]` returns an encrypted reasoning item, a tool-result turn that REPLAYS it is accepted, `text.format` and a stream with usage work | the summary ends `chat only: ...`; keep `api = "chat"` |
| **Muse Code probe** (only when a `muse` binary is on PATH; recorded under `muse_code`; never changes the exit status) | the production `muse exec` command line works on this machine: `--model` accepted, no trust stop, the model reported, a `run.usage` line seen, `hello.txt` written | do not use `--implementer muse_code` until it passes (section 8) |

The summary ends with `responses OK: set api = "responses" for the supervisor in nightshift.toml` or `chat only: ...`. Read the report before the first real run.

## 4. Choosing models, protocols and limits per role (`nightshift.toml`)

```toml
[models.default]
model = "muse-spark-1.3"
reasoning_effort = "medium"
completion_cap_ceiling = 32000     # only after the probe tells you the real ceiling

[roles.supervisor]
reasoning_effort = "high"
api = "responses"                  # opt-in; ONLY if doctor --live printed "responses OK"

[roles.implementer]
api = "responses"

[roles.reviewer]
model = "muse-spark-1.2"           # a different model from the implementer makes review more independent

[roles.auditor_final]
model = "muse-wonder-1.0"
fallback_model = "muse-spark-1.3"
```

Pass it with `--config nightshift.toml`. Keys per section: `base_url, model, api_key_env, reasoning_effort, fallback_model, completion_cap_ceiling, api, allow_partner_responses`.
The roles are `supervisor, designer, implementer, reviewer, reviewer_hard, reflector, judge, auditor_final`. An unknown role, key or value is an error at load time.

* **`api = "chat"` (default) or `"responses"` per role.** Chat Completions cannot carry reasoning across turns for external API keys; the Responses API can (encrypted reasoning
  replayed with `store: false`). That matters only for the roles that run tool loops (the supervisor and the patch worker); the single-turn structured roles gain nothing. The Responses client has never run
  against the real service and the ledger does not record which protocol a role used, so choose it only after the probe says `responses OK`, and note it in any write-up. A `base_url` other than the Meta Model API with `api = "responses"` is refused unless `allow_partner_responses = true`.
* **Muse Wonder.** `reviewer_hard` and `auditor_final` default to `muse-wonder-1.0` with `muse-spark-1.3` as fallback at `xhigh`. A 404 `model_not_found` for the primary makes the fallback sticky for the run (the first response after it carries `fallback_reason`); a 400 falls back for one call. The ledger and
  report name the model that answered; nothing is called Wonder unless a charged call shows it. Setting `model` or `base_url` in `[models.default]` moves those two roles off Wonder.
* **Partner hosts.** A role may point at another OpenAI-compatible host (for example a partner serving Meta's open-weights models) with `base_url` and `api_key_env`. That is supported by configuration and **untested against real partners**; the run is then labelled `custom`, not `meta-api`.
* Never select a `-contributor` model unless you want your content used to train future Meta models. `max` effort is valid for `muse-wonder-*` and `muse-spark-1.3`, not for `-contributor`.
* Timeouts: 300 s per request, 600 s when a call asks for more than 8000 completion tokens. A run that looks frozen may be waiting for a reasoning reply.

## 5. A first live run

```bash
python3 -m nightshift run --task tasks/law-discovery --backend meta \
    --max-usd 3 --max-experiments 12 --parallel 3 --serve
# dashboard: http://127.0.0.1:8765  (loopback only; --allow-remote on `serve` binds elsewhere, with NO authentication)
```

* **Cost.** Muse Spark is $1.25 / $4.25 per million input / output tokens (cached input $0.15); Muse Wonder $4 / $20 (cached $0.20). Defaults: `--max-usd 5`, 16 experiments, 3 parallel lanes, 40 supervisor turns, 6 sealed replications, 1800 s wall clock.
  `--max-usd` is checked before every model call, so a run stops cleanly with verdict `missing_authority` when it is spent (calls already in flight can overshoot slightly). **There is no measured cost per run**: no live run has happened, and the simulated runs use the same price table but the simulation's own token counts.
  Reasoning tokens are billed as output. `--context` adds up to about 15,000 tokens to EVERY supervisor turn (section 7), so a 40-turn run carries up to about 600,000 extra input tokens (about $0.75 uncached, $0.09 cached, at Spark prices). The dashboard shows spend per experiment and per role only for charges the ledger recorded; a Muse Code worker's spend is what Muse Code reported, unverified.
* `touch runs/<run_id>/HALT` is the clean stop: the run stops before its next action (not mid-call) and experiments still in flight are recorded `unresolved` with the note `stopped by HALT`. `Ctrl-C` concludes the run at once with reason `halt` and **finishes** it (a finished run cannot be resumed; start a new one, or `nightshift report runs/<run_id>`); an experiment that was mid-flight stays in flight in the ledger, which makes the verdict `unresolved_defects` and the exit status 1.
  A crash (the process dies) leaves the run unfinished and can be resumed: `python3 -m nightshift resume runs/<run_id> --backend meta` (add the same `--config`, and the same `--context` files if the run had them). A changed task or grader is refused, with no override.
* While it runs you should see: the supervisor's tool calls and reasoning effort per turn, lanes working in parallel, the held-out score chart, the invariant board, and (if the model does something sloppy) findings with evidence.

### How a run ends, and the exit status

| `run.finished` | meaning | exit |
|---|---|---|
| `supervisor_finish`, verdict `success` | an incumbent beat the baseline on the *sealed* split and replicated | 0 |
| `supervisor_finish`, verdict `inconclusive` | nothing beat the baseline; the report says so plainly | 0 |
| `budget_exhausted` or `halt`, verdict `missing_authority` | the budget ran out or a HALT file stopped it; experiments cut short are `unresolved` (`stopped by the budget` / `stopped by HALT`) and are the stop, not a defect | 0; 1 if the budget ran out with no incumbent |
| `error`, verdict `missing_authority` | the model API answered **401, 402 or 403** (bad key, no credits, no access): the run lacks authority | **1** |
| `error` or any run with open block findings, verdict `unresolved_defects` | another model API failure, a supervisor error, a baseline that could not be graded, an open `block` finding, an experiment left in flight or `unresolved` for another reason | **1** |

What a run must never do is declare an improvement the sealed split did not confirm; the finish gate refuses that. The honest numbers: `docs/NULL_BENCHMARK.md` gives the chance that a run still promotes
a null (low but not zero, and growing with the number of experiments); the report's headline is `Sealed delta vs baseline <id>` with the threshold an improvement had to exceed. For a live demo whose sealed split nobody has seen, draw a private one
(`python3 tasks/law-discovery/trusted/make_data.py --seed "<private text>" --out <dir>`; see `docs/BRING_YOUR_OWN_TASK.md`): the committed split is public.

## 6. Red-team lanes: `--plant`

(To rehearse this without a key: `python3 scripts/simulate_live.py --out runs/sim-redteam --plant tamper,injection,duplicate` runs the real `meta` backend against a **simulated** model on loopback in 5 to 10 seconds (about 16 with `hang`). Every screen and report says SIMULATED; the model roles are a stand-in that knows the task's answer key. With the default simulator rounds `tamper`, `injection`, `hang`, `infra_flake` and `duplicate` run; `overfit` and `claim_without_run` do **not**: the simulated designer proposes the same hypotheses first, so the duplicate gate blocks the scripted lanes before they run, and the scoreboard shows them as "not run", outside recall. An unknown `--plant` name exits 2 before anything starts; a run that fails before it starts (a busy `--port`, a bad `--task`) is not verified and its empty run directory is removed so `--out` can be reused; Ctrl-C, SIGTERM or SIGHUP with `--serve` stops the run, still verifies the ledger and exits 130; an `--out` below a file exits 2.)

```bash
python3 -m nightshift run --task tasks/law-discovery --backend meta --plant tamper,overfit,injection --max-usd 3 --serve
```

`--plant NAME[,NAME...]` (meta backend only) adds scripted **attacker** lanes to a run whose supervisor, reviewer, reflector, judges and auditor (and the designer and implementer of the supervisor's own lanes) are REAL model calls. The behaviors come from the task's scripted bank:
`tamper, claim_without_run, duplicate, infra_flake, hang, overfit, injection` (law-discovery has all of them; spiral-classifier lacks `hang` and `injection`; an unavailable one is a usage error, exit 2, before anything is created). The point is to show the real auditor and reviewer catching real faults without waiting for the model to misbehave.

* The lanes are labelled `RT-<behavior>` (truncated to 16 characters), announced by a `supervisor.nudge` beginning `Operator notice`, recorded in `run.started.config.red_team`, shown by a non-dismissible RED-TEAM LANES notice on the dashboard and a RED-TEAM paragraph at the top of the report, and the supervisor's results carry the note `red-team lane injected by the operator; scripted`. They are never presented as model work.
* A planted lane can be blocked before it runs (the duplicate gate stops a scripted attacker whose idea the supervisor already proposed, or the budget is spent). The scoreboard then lists it as **not run** and keeps it out of recall, and the red-team notice says how many ran; do not describe a partial run's recall as covering every planted behavior.
* An attacker lane is never replicable, never promotable and never the base of another lane, so a scripted patch cannot decide the verdict. Its designer and implementer are scripted; everything that reviews it is real. Each counts against `--max-experiments` and uses a parallel slot.
* A resume re-derives the planted lanes from the ledger and injects only the ones the ledger does not hold yet.

## 7. Background material: `--context`

```bash
python3 -m nightshift run --task tasks/law-discovery --backend meta --context notes/related_work.md notes/paper_excerpt.txt
python3 -m nightshift resume runs/<run_id> --backend meta --context notes/related_work.md notes/paper_excerpt.txt
```

`--context FILE [FILE ...]` (meta backend only) gives the supervisor and the designer background text (related-work notes, paper excerpts): UTF-8 text or markdown, at most 200,000 characters per file and 400,000 in total, no control characters.
The supervisor reads at most the first 60,000 characters (about 15,000 tokens, sent with every supervisor turn) and the designer the first 8,000; the CLI says how much will be used. It is fenced as untrusted data. Only each file's **name, sha256 and length** go into the ledger
(`run.started.config.context`), never the text.

Because the text is not in the ledger, a resume can restore it only if you give **the same files again**: `resume --context` checks the `(name, sha256)` set against the ledger (any order; a different, extra or missing file is a usage error naming the file) and then
`run.resumed.context_reloaded` is true. Resuming without `--context` works but the roles continue WITHOUT the background; the CLI warns and the dashboard marks the files `not_reloaded_after_resume`.

## 8. Muse Code as a worker (optional, tested only against a fake)

Install Muse Code yourself (`curl -fsSL https://dev.meta.ai/install.sh | sh` is Meta's installer; it edits your shell profile, so read it first). Then run the probe, which is the only thing that can show the integration works on your machine:

```bash
python3 -m nightshift doctor --live --live-out demo/live_probe.json     # includes the Muse Code probe when `muse` is on PATH
python3 -m nightshift run --task tasks/law-discovery --backend meta --implementer muse_code
```

* The command line (one place, tested): `muse exec --model <routing model> --disable-approval --disable-request-user-input --disable-reminders --no-session-log --sandbox-network restricted --max-model-steps N --json --prompt-file F`. `--model` is always passed because Muse Code's own default is `muse-spark-1.2`.
  `--yolo` and `--disable-sandbox` are refused. Extra arguments: `$NIGHTSHIFT_MUSE_ARGS` (shell-quoted); `$NIGHTSHIFT_MUSE` names the binary. A `[muse_code]` table in `nightshift.toml` is NOT read (it is ignored without an error), so use the environment variable. If `--sandbox-network restricted` also blocks muse's own model calls, every run fails: `NIGHTSHIFT_MUSE_ARGS='--sandbox-network proxy-only'` is the escape hatch.
* The agent works in a **private copy** under the system temp directory (outside the run directory), with instruction files withheld, and only edits to the task's editable or scratch files come back; anything else is refused, recorded, and ends the lane `invalid` with a protected-path violation (the evidence may be a placeholder file; the real paths are in `worker.log`).
  Python bytecode caches it creates are dropped silently. A trust-stop message is an infrastructure error only when the run did no work.
* **It does not confine reads.** Meta documents where Muse Code may write, not where it may read, so the agent can still read what your user can read by absolute path, and its web fetch tool is an egress channel that `--sandbox-network` does not govern. Use a container or VM for the whole run.
* Parallelism is capped at 1 for this worker (recorded as `parallel_reason`). The answering model is taken from the run when muse reports one, and a mismatch is flagged. Spend is what Muse Code printed in its `run.usage` line, unverified; with no such line nothing is charged and `raw_log` says the run is NOT metered (the wall timeout and `--max-model-steps` are then the only bound).

## 9. Recording the demo

```bash
python3 -m nightshift verify runs/<run_id>                       # chain check; prints the head anchor
cp -r runs/<run_id> demo/golden_live
python3 -m nightshift serve demo/golden_live --replay-duration 150   # replay as a 150 s screen recording (or --replay 10 for 10x)
python3 -m nightshift report runs/<run_id>                       # report.md and report.html in the run directory (--out DIR elsewhere)
```

A replay is labelled REPLAY on screen and the chain is verified before it is served. Record the head anchor somewhere else (the slide, the README): it is what makes truncation of the ledger detectable (`nightshift verify RUN_DIR --expect-head ANCHOR`).
`/api/health` reports `mode` (`scripted`, `simulated`, `live`, `replay`, `unknown`), `origin` and `serving`. (`nightshift demo --out DIR` is relative to your current directory; `demo --replay DIR` serves an existing ledger without running.)

What you may call a recording: say which of three it is, and show the label. A `--backend scripted` run is a fault-injection harness (banner "SCRIPTED BACKEND"). A run against a local fake server is a simulation of the live path (banner "SIMULATED MODEL"). Only a ledger whose endpoint record is `meta-api` and
whose `budget.charged` rows name Muse models is a live Meta run. A `--plant` run is real model roles plus labelled scripted attackers. State the isolation mode that was active. Do not describe a run in `unshare-net` or `none` as sealed-safe.

## Troubleshooting

* `ModelConfigError: no API key` -> `echo ${MODEL_API_KEY:+set}` should print `set`; `doctor` shows `MODEL_API_KEY: set`.
* The run ended at once with reason `error`, verdict `missing_authority`, exit 1 -> the API answered 401, 402 or 403: check the key, the credits and access. The ledger keeps only the status number.
* HTTP 429 -> the client honours `Retry-After` and backs off (429, 500, 502 and 503 are retried up to 4 times); lower `--parallel`. A network failure is raised at once, not retried.
* HTTP 400 naming an optional parameter -> the client drops it, retries once and remembers; the probe report shows which. A 400 about `max_completion_tokens` is retried once at a lower value; set `completion_cap_ceiling` to what the probe found.
* The supervisor answers in plain text instead of calling tools -> Chat Completions only supports `tool_choice: auto`, so the loop nudges it and, after two plain replies, asks it to finish.
* A finding that blocks `finish` -> read it on the dashboard; the supervisor is told to fix the cause. A deterministic block finding can only be closed by the auditor once the cause is contained; it cannot be dismissed. A `trusted` protected-path violation is never contained automatically.
* `isolation probe: none` on a Linux box -> unprivileged user namespaces are probably disabled; section 2.
* `no task found ... bundled tasks not found` (exit 2) -> run from a clone, `pip install -e .`, set `NIGHTSHIFT_TASKS_DIR`, or pass `--task /path/to/task`.
* `cannot resume ... the task or grader changed` -> start a new run; grades from before and after a change are not comparable. `--context: ... has changed since the run started` -> give the original files.
* `the 'muse' executable was not found` -> install Muse Code, put it on PATH or set `NIGHTSHIFT_MUSE`.
* `ResumeRefused ... already finished` -> Ctrl-C, HALT, a spent budget and a refused key all finish a run; only a process that was killed (or crashed) before `run.finished` can be resumed.
