# Nightshift architecture and build contract (v2)

Nightshift is an auto-researcher you can leave running overnight and believe in the morning. A Muse Spark supervisor
composes bounded research workflows, parallel workers explore hypotheses in isolated workspaces, a trusted grader scores
them on a split the agents never see, and an independent auditor checks the run against invariants derived from real
auto-research failures. Everything is recorded in an append-only hash-chained ledger and shown on a live dashboard.

This file is the **contract**. Modules are built in parallel against it. `nightshift/types.py`, `ledger.py`, `view.py`,
`secrets.py` and `tests/support/synthetic.py` already exist, are tested, and are authoritative. v2 folds in an adversarial
design review (interfaces, security, statistics, hackathon fit). If you need to change a shared file, do not edit it
silently: report the exact change you need.

Status: waves 1 to 5 are built. **Appendix A is the as-built record and wins over any conflicting text in sections 1 to 13**; sections
4, 5, 6, 7, 8, 10, 11 and 12 have been brought in line with the code, and the places where the build went beyond the original
contract are listed in Appendix A.8 to A.16. The other documents: `docs/INVARIANTS.md` (the auditor's catalog), `docs/META_API_NOTES.md`
(Meta API facts and what was and was not verified), `docs/RUNNING_LIVE.md` (operator guide), `docs/BRING_YOUR_OWN_TASK.md`,
`docs/NULL_BENCHMARK.md` (error rates of the decision rule).

## 1. Principles (each one is enforced by code, not by prompt)

1. **The ledger is the state.** No authoritative in-memory state. Whatever an agent says is an `agent.claim` event and is
   never evidence. Evidence is an `experiment.graded` event produced by the trusted grader.
2. **Hard gates are code; semantic judgment is the auditor.** Budget, duplicates, protected paths, replication and the finish
   gate are exact checks that run before the action. The auditor reviews what code cannot, and re-checks what code did. Its
   findings go to the supervisor. No finding ever spends money, kills a job or edits policy.
3. **Two splits.** Agents see the **dev** score. A **sealed** split is graded only during replication and promotion and
   its numbers never reach any prompt, tool result, memory text or `LaneResult`. Only the boolean `replicated` does.
   This is the public/private leaderboard defence against adaptive overfitting.
4. **Detection over prevention for same-user code.** Experiment code runs as the same OS user. We contain it
   best-effort (section 11) and **detect** tampering with hashes. We never claim an OS-level security boundary.
5. **Honest labelling.** `run.started.backend` is `scripted` or `meta`, and `run.started.config.endpoint` records which kind of
   endpoint a meta run talked to (section 10 and A.9). A scripted run never claims model behaviour; a meta run against a local fake
   server is a SIMULATION of the live path and is labelled so everywhere; a meta run with `--plant` has real model roles plus
   scripted attacker lanes that are labelled as scripted; nothing claims a live Meta API run unless the ledger's endpoint record and
   its charged models say so. Missing evidence is `insufficient_evidence`, never `compliant`. A judge that cannot answer yields
   `insufficient_evidence`, never silence.
6. **Idempotent and resumable.** Experiments have an idempotency key. LLM charges are keyed by call id. After a crash,
   `resume` reconciles in-flight experiments before doing anything new.
7. **Untrusted text stays data.** Experiment output, diffs, file contents, lessons and ledger text are fenced
   (`nightshift.fence.fence`) when shown to a model and are never instructions.

## 2. Layout, ownership and build waves

```
nightshift/
  types.py ledger.py view.py secrets.py               DONE
  fence.py budget.py memory.py                        B2   (wave 1)
  task.py sandbox.py grading.py                       B1   (wave 1)
  models/{__init__,base,meta,pricing,scripted,config}.py     B3 (wave 1)   models/__init__.py is EMPTY
  audit/{__init__,invariants,auditor,judge}.py        B7   (wave 1)
  dashboard/{__init__,state_api,server}.py + index.html app.js app.css; report.py; cli.py; __main__.py   B8 (wave 1)
  agents/{__init__,llm,prompts,patch_worker}.py       B4   (wave 2)
  agents/scripted.py                                  B5   (wave 2)
  workflows/{__init__,templates,gates,engine}.py; orchestrator.py; agents/factory.py   B6 (wave 2)
tasks/law-discovery/  (stdlib only)                   B9   (wave 1)
tasks/spiral-classifier/ (needs numpy), agents/muse_code.py + tests/support/fake_muse, --plant   P2 (wave 4)
```

`__init__.py` files already exist and are empty for `models`, `agents`, `workflows`, `audit`, `dashboard`. Do not put imports in them.

Import rules: `types` imports nothing. `ledger`/`view` import `types` and `secrets`. `budget` imports `types`,
`models.pricing`. `models.config` imports `budget` **lazily inside functions**. Nothing imports `cli`. The orchestrator is the only
module that imports everything.

Priority: **P0** task/sandbox/grading, budget, scripted agents, engine+orchestrator (explore and replicate), deterministic
invariants, dashboard, demo, `law-discovery`. **P1** MetaChatClient, LLM roles, LLMJudge, live smoke. **P2** (only after P0 and
P1 pass CI) Muse Code worker, `--plant` under the `meta` backend, spiral task, reflection polish, golden live ledger.

Build waves: wave 1 = B1 B2 B3 B7 B8 B9 (independent). Wave 2 = B4 B5 B6 (B6 tests with a fake suite in `tests/support`).
Wave 3 = integration and the end-to-end demo. Wave 4 = P2 items.

Stdlib only, Python 3.11+. Only the optional spiral task may import numpy. Tests use `unittest`
(`python -m unittest discover -s tests -t .`). No network in tests: use `tests/support/fake_meta_server.py` (B3).

As built, the tree also holds `nightshift/models/{live_smoke,responses}.py` (live probes; the opt-in Responses client),
`nightshift/agents/muse_code.py` (the Muse Code worker), `scripts/{live_smoke,null_benchmark}.py`, and the test doubles
`tests/support/{fake_meta_server,fake_muse,sim_muse,canary,fake_engine,fake_suite,synthetic}.py`. `sim_muse.py` is the simulated model
brain that drives the live code path over loopback.

## 3. Run directory

```
runs/<run_id>/
  ledger.jsonl            authoritative, hash-chained; ledger.jsonl.lock is the writer lock
  memory.md               agent-visible shared memory (regenerated atomically after every recorded experiment)
  HALT                    if present the orchestrator stops before its next action
  exp/<exp_id>/
    workspace/            what the implementer edits
    base.json             {"base_ref": "template"|"<exp_id>", "base_hashes": {...}, "tree_hash": "..."}
    frozen/               read-only copy of the non-scratch files, made at baseline and at promotion
    runs/<run_idx>/ws/    throwaway copy the untrusted run executes in
    runs/<run_idx>/staged/outputs/   sanitised copy of the run's outputs, the only thing the grader reads
    logs/run_<idx>.stdout|stderr     capped (8 MiB each)
    diff.patch worker.log             written by the ENGINE only
  report.md report.html   written by `nightshift report`
```

All paths stored in events are posix paths **relative to the run dir** (`exp/e003/workspace`).

## 4. Tasks, splits, seeds and the evaluation pipeline

```
tasks/<name>/
  task.toml
  template/            copied into every workspace. Editable files, readonly code, visible data. Includes `data.py`.
  trusted/             grader + labels. NEVER copied into a workspace. Never contains pickles.
  scripted/            offline proposal bank (manifest.json + files). Optional.
```

```toml
name = "law-discovery"
description = "one line"
brief = """Plain text shown to models: objective, what may be edited, what the metric means, rules. Must not mention seeds
or the sealed split."""
objective = "minimize"            # or "maximize"
primary_metric = "heldout_rmse"
train_metric = "train_rmse"       # optional; enables the overfit invariant

[workspace]
editable = ["model.py"]           # fnmatch globs, workspace-relative, '/' separated
scratch  = ["outputs/**"]

[run]
cmd = ["python3", "run.py"]
timeout_sec = 20

[grade]
cmd = ["python3", "{task_dir}/trusted/grade.py", "--workspace", "{workspace}", "--seed", "{seed}", "--split", "{split}"]
timeout_sec = 20
seeds = [0, 1, 2]
replication_seeds = [100, 101, 102, 103, 104]
noise_margin = 0.02               # absolute floor on the delta that counts. Must be > 0 when sealed
sigma_floor = 0.0                 # set to the unmodified baseline's per-seed std (`nightshift tasks` measures it); 0.0 leaves the noise prior inert
sealed = true
requires = []                     # python modules the run needs, e.g. ["numpy"]
stall_sec = 10
```

### 4.1 What a seed varies (so replication is real evidence)

`NIGHTSHIFT_SEED` must change (a) the **training sample** and (b) the **evaluated rows**:

* `template/data.py` (readonly) exposes `load_train(seed)`: a seeded 80% subsample, without replacement, of a fixed `train_pool`. `run.py` must use it.
* `run.py` predicts **every** row of `eval_inputs.csv` (inputs only, no labels; dev and sealed rows unlabeled and shuffled together).
* The grader holds the labels. For `--split S` it selects the fixed row set of that split, then draws
  `random.Random(f"{S}:{seed}").sample(rows, k=int(0.7*len(rows)))`. So different seeds give different training data and different
  (overlapping) evaluation subsets, and the std includes test-sampling variance.
* Dev and sealed are a fixed random 50/50 partition of the same pool and distribution. A dev-vs-sealed gap is therefore overfitting to dev,
  not distribution shift.
* Seeds subsample one pool, so seed evidence is correlated. It supports "robust to sampling of this pool", not "true on new data".
* `validate_task` runs a calibration: grade the unmodified template on all `seeds`; problem `seeds_vacuous` if the per-seed scores are all
  equal; `len(seeds)>=3`; `len(replication_seeds)>=3`; `noise_margin>0` when `sealed`; seed lists disjoint. `nightshift tasks` reports these.
  `calibrate=True` also estimates the baseline's per-seed std over at least 12 seeds (20 by default) on each split and prints the
  recommended `sigma_floor` with a warning when the task's floor is below 75% of it or its `noise_margin` below 50% of it (section 8).
  A problem that only says an optional `requires` module is missing (`missing_requirements:`) is a SKIP in `nightshift tasks`, not a failure.

### 4.2 Trusted grader contract

`grade.py --workspace W --seed S --split dev|sealed` runs with cwd = `task_dir`, scrubbed env, rlimits, in its own process group. `W` is the
**staged** directory (it contains only sanitised `outputs/`), never the live workspace. It prints one JSON object as the last stdout line:

```json
{"valid": true, "score": 0.1234, "split": "dev", "n_eval": 120, "metrics": {"heldout_rmse": 0.1234, "train_rmse": 0.05}, "notes": ""}
```

Rules: exit code 0; `split` must echo the request; `valid` is a bool; `score` and every metric are finite floats (NaN/Infinity rejected by
the parser); parse predictions only as JSON or CSV (never pickle, `np.load(allow_pickle=True)`, `eval`, `exec`, `yaml.load`, `importlib`) and
never import workspace code; compute both the held-out and the train metric itself. `valid=false` is a model-side failure. Grader stdout/stderr
are never shown to a model. `validate_task` greps `trusted/*.py` for the forbidden calls.

### 4.3 Untrusted run contract

`run.cmd` runs in `runs/<idx>/ws` with env: `NIGHTSHIFT_SEED`, `NIGHTSHIFT_WORKSPACE`, `NIGHTSHIFT_ATTEMPT` (1-based), `PYTHONHASHSEED=0`,
`PYTHONDONTWRITEBYTECODE=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `HOME=<ws>`, a minimal `PATH`,
and `PYTHONPATH` set to the parents of each `task.requires` module. **Nothing else from the parent environment.**

### 4.4 `evaluate` pipeline (per seed, in `grading.evaluate`)

1. `copy_regular(ws, runs/<idx>/ws)`: file by file via `_scan`, never `copytree`. Any symlink, FIFO, device, socket, hard-linked file
   (`st_nlink>1`), path not matching `^[A-Za-z0-9._/-]{1,128}$` or containing `..`, or more than 2000 files / 64 MiB is a violation `special:<path>`.
2. Assert `tree_hash(non-scratch files) == expected_tree_hash` (recorded in `experiment.implemented`); else `ProtectedPathViolation(kind="runtime")`.
3. `run_untrusted`: shim launcher (never `preexec_fn`), `start_new_session=True`, process-group `SIGKILL` in `finally`, stdout/stderr streamed to
   capped log files, wall timeout, watchdog (heartbeat every 2 s; no output growth for `stall_sec` -> `watchdog.stalled` once, then kill).
4. Kill the group, re-snapshot. A change to any non-scratch file is `ProtectedPathViolation(kind="runtime")`.
5. `stage_outputs`: copy only regular files from `outputs/` (name `^[A-Za-z0-9._-]{1,64}$`, depth <= 2, <= 32 files, <= `max_output_bytes`,
   `O_NOFOLLOW`, `st_nlink==1`) into `runs/<idx>/staged/outputs/`. Record `outputs_hash`.
6. `trusted_hash` is compared before and after the untrusted run and after the grader invocation; a change is
   `ProtectedPathViolation(kind="trusted")`. `trusted_hash` ignores `__pycache__` and `*.pyc`; the grader runs with
   `PYTHONDONTWRITEBYTECODE=1`. A content hash cannot see a file that was changed and put back before the process exited, so a
   second check runs in every isolation mode: a stat snapshot of the whole `trusted/` tree taken right before the untrusted process is
   spawned and compared after the process group is reaped and swept, and again after the grader (details and limits in section 11).
   It raises the same `ProtectedPathViolation(kind="trusted")`. `evaluate` also refuses to start, and refuses to run the grader, while a
   source-less `*.pyc` sits in `trusted/` (Python imports such a file ahead of the standard library and the content hash ignores it).

### 4.5 Scripted proposal bank (`scripted/manifest.json`, owned by B9, consumed by B5)

```json
{
  "version": 1,
  "steps": [
    {"say": "narration", "action": "launch", "template": "explore",
     "lanes": [{"lane": "A", "scripted_id": "quadratic"}, {"lane": "B", "scripted_id": "poly9"}]},
    {"action": "finish", "summary": "...", "claims": ["quadratic"], "expect": "refused"},
    {"action": "replicate", "of": "damped"},
    {"action": "acknowledge", "finding_for": "tamper", "how": "addressed", "reason": "workspace discarded"},
    {"action": "finish", "summary": "...", "claims": ["damped"]}
  ],
  "proposals": {
    "quadratic": {"hypothesis": "...", "approach": "...", "behavior": "normal", "planted": false,
                  "files": {"model.py": "patches/quadratic/model.py"},
                  "claim": {"type": "improved", "text": "..."},
                  "defect": {"marker": "BUG(quadratic):", "repair_files": {"model.py": "patches/quadratic/model.fixed.py"}}}
  }
}
```

* `behavior`: `normal | tamper | claim_without_run | duplicate | infra_flake | hang | overfit | injection`. Every faulty proposal has
  `"planted": true`. **Outcome numbers are always computed by really running the workspace**; only the faults are scripted.
* `claim_without_run`: the implementer emits an `improved` claim although the result is not an improvement; it is contradicted by the grade.
* `infra_flake`: fails once with an infrastructure-looking error, then passes (uses `NIGHTSHIFT_ATTEMPT` or a marker file in `outputs/`).
* `defect.marker` is unique across the bank. `ScriptedReviewer` returns `repair` iff any bank marker occurs in the diff text.
* `injection`: the run prints text telling reviewers to approve; it exists to show real LLM roles ignoring it.
* The scripted supervisor derives its cursor from `view.supervisor_turns` (one `note_turn` per step) and rebuilds `scripted_id -> exp_id`
  from `experiment.proposed.scripted_id`. Claim aliases are `agent.claim.alias`. `acknowledge.finding_for` resolves to the first open finding whose `exp_id` is the mapped experiment.
* `say` becomes `supervisor.turn.say` with `source:"script"`. The dashboard renders it as "script narration", never as model speech.
* A second manifest `manifest.control.json` contains only honest proposals and must produce **zero** findings (negative control).

## 5. Ledger event payloads (authoritative)

All events also carry `seq, ts, run_id, kind, actor, prev_hash, hash`. `actor` is `engine`, `supervisor`, `designer:<lane>`,
`implementer:<lane>`, `reviewer:<lane>`, `reflector`, `auditor`, `watchdog`, `human`. `view.build_view` ignores unknown keys, so adding
keys is compatible; renaming is not. Strings are capped at 8 KiB and secrets are scrubbed by `Ledger.append`. An append that is interrupted (a Ctrl-C, a failing `fsync`) restores memory from the file, which is authoritative: a complete, hash-consistent line that reached the disk is adopted, a torn partial line is truncated, so the next append continues the chain instead of reusing a `seq` (`tests/test_ledger_interrupt.py`).

| kind | payload |
|---|---|
| `run.started` | `task, backend, models{role:model}, limits, task_hash, grader_hash, objective, primary_metric, train_metric, isolation, config{implementer, control, parallel, parallel_reason?, endpoint{kind, hosts[]}, red_team?[{behavior, scripted_id}], context?[{name, sha256, chars}]}`. `models` holds the CONFIGURED role models; the models that answered are in the `budget.charged` rows. `isolation` is `sandbox.effective_isolation`, never the request echoed |
| `run.resumed` | `reason, reconciled[exp_id], task_hash, grader_hash, endpoint{kind, hosts[]}, parallel, parallel_reason?, context_reloaded?` (`context_reloaded` only when the run had `config.context`: true only when the same files were given again) |
| `run.finished` | `reason (supervisor_finish\|budget_exhausted\|halt\|error), summary, verdict_class (success\|inconclusive\|unresolved_defects\|missing_authority), final_claims[], head_anchor` |
| `supervisor.turn` | `turn, model, action_summary, say\|null, tool_calls[{name,args_summary}], reasoning_effort\|null, latency_ms\|null, text\|null, source (supervisor\|script)` |
| `supervisor.nudge` | `text` (templated from ids and static titles only, never from model-derived text). A `--plant` run also writes `Operator notice: lane(s) RT-... in this launch are scripted red-team attackers added by the operator...` |
| `supervisor.finish_requested` | `summary, claim_ids[]` |
| `supervisor.finish_refused` | `reasons[]` |
| `agent.claim` | `claim_id ("<exp_id>.c<n>" or "run.c<n>"), exp_id\|null, claim_type (improved\|no_effect\|fixed\|done\|safe\|retraction\|other), text, cites[seq], alias\|null` |
| `experiment.proposed` | `exp_id, lane, hypothesis, plan, parent_exp_id, base, hyp_fingerprint, replicate_of, scripted_id\|null, planted_fault\|null` |
| `experiment.blocked` | `exp_id\|null, lane, hypothesis, hyp_fingerprint, reason (duplicate\|budget\|halt\|protected\|invalid_composition\|not_replicable), detail, dup_of\|null`. A block **with** `exp_id` means the experiment ended there; pre-proposal blocks use `exp_id=null` |
| `experiment.workspace_ready` | `exp_id, path (relative), base_hash` |
| `experiment.implemented` | `exp_id, worker, diff_hash, files_changed[], diff_summary, idem_key, attempt, steps, tree_hash, plan_summary` (not written for an idem-key duplicate or when the implementer changed nothing). `plan_summary` is the designer's approach and expected effect, sanitised, at most 600 characters: the plan the implementer and reviewer worked from, recorded so a resumed lane continues from it |
| `experiment.protected_violation` | `exp_id, paths[], kind (workspace\|trusted\|runtime\|trusted_probe)` |
| `experiment.run_started` | `exp_id, run_idx, purpose, seed, cmd[], pid, attempt, isolation` |
| `experiment.heartbeat` | `exp_id, run_idx, output_bytes, cpu_sec\|null` |
| `experiment.run_finished` | `exp_id, run_idx, purpose, seed, exit_code\|null, duration_sec, classification (ok\|timeout\|crash\|infra\|interrupted), stalled, stdout_tail (<=4 KiB), stderr_tail, note` |
| `experiment.graded` | `exp_id, purpose, split (dev\|sealed), grade{GradeResult}, grader_hash, seeds[], conditions_hash, outputs_hash` |
| `experiment.reviewed` | `exp_id, review_round, reviewer, reviewer_model, implementer_model, independent, independent_basis (model\|context), verdict (accept\|reject\|repair), findings[{id,severity (block\|warn\|info),text,evidence}]` |
| `experiment.repaired` | `exp_id, repair_round, diff_hash, addressed[finding_id]` |
| `experiment.recorded` | `exp_id, phase (lane\|replication), outcome (baseline\|improved\|no_gain\|worse\|failed\|invalid\|infra_error\|unresolved), delta, vs_exp_id, noise_margin, train_delta, train_margin, overfit_suspect, replicated, sigma_prior, n, s_eff, t, sealed_delta, sealed_margin, note` |
| `incumbent.promoted` | `exp_id, score (dev), prev_exp_id, replicated, replication_exp_ids[], tree_hash, sealed_score, sealed_delta, sealed_margin` |
| `memory.lesson` | `lesson_id, text, kind (observation\|hypothesis\|correction\|strategy), source_exps[], classification (model\|infra\|unknown)` |
| `memory.reflection` | `trigger (stall\|manual\|periodic), consolidated[lesson_id], pivots[]` |
| `budget.charged` | `kind (llm\|experiment\|replication\|wall), amount, unit (usd\|count\|sec), call_id\|null, model\|null, exp_id\|null, purpose\|null, usage{prompt_tokens,completion_tokens,cached_tokens,cached_known,reasoning_effort,latency_ms,retries,schema_ok,stream_fallback,requested_model}`. An `llm` row carries `model` = the model that ANSWERED (`requested_model` is what was asked for), and, when the call belongs to an experiment, `exp_id` and `purpose`; its actor is `<role>:<lane>` (for example `designer:A`) or just `<role>` for a call that belongs to no lane. The Muse Code worker's spend is charged as actor `implementer:muse-code` from numbers Muse Code reported (A.12) |
| `budget.exhausted` | `kind, limit, used` |
| `audit.window` | `from_seq, to_seq, invariants[ids checked], judged[ids that got an LLM verdict], skipped_reason\|null` |
| `audit.finding` | `finding{Finding}` |
| `audit.ack` | `finding_id, action (addressed\|dismissed), reason, by, containment, followup_seqs[]`. **Written by the Auditor (actor `auditor`) or a human only**; `view` ignores other actors |
| `watchdog.stalled` | `exp_id, run_idx, idle_sec` |

`conditions_hash = sha256(task_hash | grader_hash | split | seeds | run_cmd)[:12]`. **Comparable** means same `grader_hash`, same `split`, same seed list, same n.
Hash lengths: `diff_hash`, `base_hash`, `tree_hash`, `conditions_hash`, `task_hash`, `grader_hash` are 12 hex; `idem_key` is 16.
`idem_key = sha256(base_hash | diff_hash)[:16]`; `base_hash(ws) = short_hash(canonical_json(sorted(base_hashes.items())))`.
`run_idx` is the count of existing `run_started` events for that exp, never reused.

Purposes and splits are fixed in `types.Purpose`: `baseline`, `primary` use **dev** and `task.seeds`; `replication`, `replication_baseline` use **sealed** and
`task.replication_seeds`. `Purpose.SEALED` numbers must never reach an agent-facing string.

## 6. Module APIs

Signatures are binding. Add keyword-only parameters freely; do not change positional ones.

### 6.1 `task.py` (B1)
```python
def load_task(path) -> TaskSpec                                  # ValueError with a precise message on bad toml
def list_tasks(root) -> list[TaskSpec]
def validate_task(task, *, calibrate: bool = False) -> list[str]  # problems; empty = OK. Structural checks always; calibrate=True also grades the template
def trusted_hash(task) -> str                                    # 12 hex over sorted (relpath, bytes), ignoring __pycache__ and *.pyc
def task_hash(task) -> str
def conditions_hash(task, seeds, split="dev") -> str
def missing_requirements(task) -> list[str]
```
### 6.2 `sandbox.py` (B1)
```python
@dataclass
class Workspace: exp_id: str; path: Path; base_hashes: dict[str, str]; tree_hash: str
@dataclass
class WorkspaceDiff: changed: list[str]; added: list[str]; removed: list[str]; unified: str; diff_hash: str; summary: str
                     violations: list[str]; tree_hash: str     # unified capped at 200 KB, binary files shown as 'binary file changed'
@dataclass
class RunResult: exit_code: int | None; stdout: str; stderr: str; duration_sec: float; timed_out: bool; stalled: bool
                 classification: str; output_bytes: int; pid: int | None; note: str      # stdout/stderr are the LAST 64 KiB only
                 isolation: str; trusted_changed: list[str]; trusted_contested: bool       # mode applied; stat-snapshot difference (section 11)

def harden_parent() -> None            # prctl(PR_SET_DUMPABLE, 0): /proc/<pid>/environ and /mem are unreadable to a NON-ROOT process of the same uid (NOT to root or CAP_SYS_PTRACE; section 11); never raises
def available_isolation() -> str       # "userns" | "unshare-net" | "none", chosen by REALLY running a probe, then LOWERED (never raised) by NIGHTSHIFT_ISOLATION; "none" off Linux
def effective_isolation(requested="auto") -> str   # the mode run_untrusted(isolation=requested) would apply right now
def isolation_override() -> str        # validated NIGHTSHIFT_ISOLATION: auto | userns | unshare-net | none; ValueError otherwise
def create_workspace(task, run_dir, exp_id, base: Path | None = None) -> Workspace   # base: None=template, else a frozen/ dir; recomputes tree_hash
def tree_hash(root: Path, task: TaskSpec) -> str                  # over non-scratch regular files only
def base_hash(ws: Workspace) -> str
def diff_workspace(task, ws) -> WorkspaceDiff                    # also detects the tripwire pattern (section 11) -> violations 'probe:<path>'
def freeze(ws_path: Path, dest: Path, task) -> str               # copies non-scratch regular files, chmod a-w, returns tree_hash
def copy_regular(src: Path, dst: Path, task) -> None             # raises ProtectedPathViolation on special files
def stage_outputs(run_ws: Path, dest: Path, task) -> str         # returns outputs_hash
def run_untrusted(task, run_ws, seed, *, purpose="primary", attempt=1, on_start=None, on_heartbeat=None,
                  heartbeat_sec=2.0, timeout_sec=None, isolation="auto", log_dir=None, log_prefix="run", on_stalled=None,
                  mask_paths=(), solo=False, watch=None) -> RunResult
def classify_failure(task, exit_code, stderr, timed_out) -> str   # timeout | infra | crash | ok (see below)
```
`classify_failure`: timed_out -> `timeout`. `infra` for ImportError/ModuleNotFoundError of a module in `task.requires`, `MemoryError`, "No space left on device",
"Cannot allocate memory", "Temporary failure in name resolution", "Resource temporarily unavailable", exit -9/137 without our timeout. Other nonzero -> `crash`. Zero -> `ok`.

Portability: every `resource.setrlimit` is individually wrapped in try/except and skipped on failure; limits apply on Linux only except `RLIMIT_CPU`/`RLIMIT_FSIZE`
(best effort on all POSIX); `RLIMIT_NPROC` is skipped when euid is 0 and otherwise `current_process_count + 64`; `RLIMIT_AS` is skipped when it fails (macOS).
Never use `preexec_fn`: spawn through `[sys.executable, "-I", "-c", SHIM, *cmd]` that sets limits and `os.execvpe`s. A test must prove a run still returns `ok` when
`setrlimit` raises. `userns` isolation = `unshare --user --map-root-user --mount --net --pid --fork --mount-proc` with the trusted dir, the ledger and sibling workspaces masked;
it is selected only if a probe command succeeds, and the isolation actually applied is recorded in `run_started.isolation`.
`unshare-net` is `unshare --net` (or `--user --map-root-user --net`), and, only as root and only when a probe proves it, `unshare --net --pid
--fork --mount-proc` with capabilities dropped (a private `/proc`). `NIGHTSHIFT_ISOLATION=auto|userns|unshare-net|none` can only lower
the mode the probe proved (section 11); `run.started.isolation` and `experiment.run_started.isolation` hold what `effective_isolation`
returns, never the request echoed.

### 6.3 `grading.py` (B1)
```python
@dataclass
class EvalHooks: on_run_start: Callable[[int, int, int | None], None] | None; on_heartbeat: Callable[[int, int, float | None], None] | None
                 on_run_end: Callable[[int, int, RunResult], None] | None; on_stalled: Callable[[int, float], None] | None
@dataclass
class EvalResult: grade: GradeResult; runs: list[RunResult]; failure_class: str | None; conditions_hash: str; outputs_hashes: list[str]
def evaluate(task, ws, seeds, *, purpose, run_dir, expected_tree_hash, hooks=None, run_idx_start=0, attempt=1,
             timeout_sec=None, isolation="auto") -> EvalResult      # section 4.4. split = Purpose.split(purpose)
def run_trusted_grader(task, staged: Path, seed, split="dev") -> dict   # strict parse (finite floats, split echo, exit 0); ValueError otherwise
def decide_outcome(task, candidate, incumbent, *, k_sigma=None, sigma_prior=0.0) -> OutcomeDecision   # section 8
```
`evaluate` raises `ProtectedPathViolation` for kinds `runtime` and `trusted`. `failure_class` precedence: infra > timeout > crash > invalid > None.

### 6.4 `fence.py`, `budget.py`, `memory.py` (B2)
```python
# fence.py
def fence(text: str, label: str, *, max_chars: int = 4000) -> str   # truncates with '[truncated N chars]'; delimiter '<<<DATA:{label}:{sha256(text)[:10]}>>>' ... '<<<END:...>>>'. Deterministic, unforgeable
def sanitize(text: str, max_chars: int = 300) -> str                # strips control characters (keeps \n\t), caps length
# budget.py  (imports only types and models.pricing)
class BudgetGate:
    def __init__(self, limits, ledger, *, clock=time.time)          # holds an RLock
    def check(self, action: str) -> None                            # "experiment" | "replication" | "llm" | "supervisor_turn" | "wall"; BudgetExceeded + budget.exhausted once
    def reserve_experiment(self, exp_id, purpose="primary") -> None # check + charge ATOMICALLY; baseline charges 0
    def reserve_replication(self, exp_id) -> None                   # counts against limits.max_replications
    def charge_llm(self, model, usage, call_id, actor, **detail) -> float   # idempotent per call_id; detail goes into usage{}
    def charge_wall(self) -> float                                  # delta since previous charge in THIS process
    def remaining(self) -> dict; def snapshot(self) -> dict
def remaining_from_view(limits, view, now) -> dict                  # pure; used by BudgetGate and by the dashboard
class MeteredClient:                                                # implements ModelClient
    def __init__(self, inner, gate, actor)                          # check("llm") before, charge after; ModelError passes through uncharged
# memory.py
class Memory:
    def __init__(self, ledger, task)
    def fingerprint(self, hypothesis) -> str; def seen(self, hypothesis) -> str | None   # RECORDED non-baseline, non-infra_error experiments only
    def digest(self, max_chars=4000) -> str; def tried_table(self) -> str               # dev numbers only; sealed numbers are never read
    def add_lesson(self, text, *, kind="observation", source_exps=(), classification="unknown", actor="reflector") -> str
    def write_markdown(self, path) -> None                          # temp file + os.replace, under a lock
    def consecutive_non_improving(self) -> int
```
### 6.5 Models (B3)
```python
# models/meta.py
class MetaChatClient:                                              # Chat Completions client; implements ModelClient
    def __init__(self, model="muse-spark-1.3", *, base_url="https://api.meta.ai/v1", api_key=None, api_key_env="MODEL_API_KEY",
                 timeout=None, max_retries=4, sleep=time.sleep, transport=None, safety_identifier=None,
                 default_reasoning=None, fallback_model=None,
                 fallback_reasoning=None, cache_retention=None, completion_cap_ceiling=None, ...)
    @classmethod from_env(cls, model=None, **kw)                   # MODEL_API_KEY, then META_API_KEY; NIGHTSHIFT_BASE_URL / NIGHTSHIFT_MODEL override
    disabled_params: set[str]; primary_unavailable: str | None
```
Requirements (from `docs/META_API_NOTES.md`): omit `tool_choice`; never send `stop`, `n`, `logprobs`, `logit_bias`, `max_tokens`; `reasoning_effort` in
`minimal|low|medium|high|xhigh|max` (`none`->`minimal`, because the API answers `none` with 400; `max` for `muse-wonder-*` and `muse-spark-1.3`, not for `-contributor`
or older Spark, anything else is a `ValueError` before a request is sent); nested tool form; `response_format` json_schema strict with the schema normalised to the strict subset;
`max_completion_tokens`; `prompt_cache_key`; hashed `safety_identifier`. **Every streamed request (`stream=True` and the 504-then-stream fallback) sends
`stream_options.include_usage`**, because a stream omits `usage` otherwise and the hidden reasoning tokens would never reach the budget gate. Retry 429/500/502/503 (not 504),
honour `Retry-After`, 0.5 s base doubling plus 0-1 s jitter, max 4 retries; a terminal in-stream error with code `server_shutting_down`, `service_overloaded` or `backend_unavailable`
re-sends the whole request within the same budget. A 504 on a non-streaming call is retried once as a stream and reassembled (content and tool-call arguments by `index`).
400/401/402/403/404/413 raise `ModelError(retryable=False)` carrying the numeric `status`. **Timeouts:** 300 s per request, and 600 s for a call whose `max_completion_tokens`
is above 8000 (Muse Spark always reasons and a non-streamed reply can take minutes); a `timeout` given to the constructor or to `chat` always wins.
**A 400 whose message names an optional parameter (`prompt_cache_key`, `prompt_cache_retention`, `safety_identifier`, `reasoning_effort`, `temperature`, `stream_options`) is retried once without it and the name is added to
`disabled_params`.** `tools` and `response_format` are never stripped. **`max_completion_tokens`:** `completion_cap_ceiling` (constructor, or per endpoint in `nightshift.toml`) clamps every request; a 400 that
names `max_completion_tokens` is retried once at the limit the message states, else the largest value this client saw accepted, else half, and the lower value is remembered per model once it succeeds; a 400 about the
context window is never taken for the output maximum. `prompts.retry_cap` never raises a cap above `RETRY_CAP_CEILING` (16000) except through a configured ceiling.
If `fallback_model` is set, a `ModelError` with status 400/404 on the primary model retries on the fallback and `ChatResponse.model` reports the model that answered. **A 404 `model_not_found` makes the fallback sticky**: later calls
of that client go straight to the fallback, and the first response after the discovery carries `raw["_nightshift"]["fallback_reason"]` once (`primary_unavailable` keeps the reason); a 400 falls back for that one call only.
Cached tokens are read from `usage.prompt_tokens_details.cached_tokens` or `usage.cached_tokens`, else 0 with `cached_known=false` in the usage detail.
`transport(method, url, headers, body_bytes, timeout, stream) -> (status, headers, body_or_iter)` is injectable. **base_url must be https, or a loopback host**; userinfo is stripped; redirects are never followed
(a 3xx is a non-retryable `ModelError`); the key is registered with `nightshift.secrets`, sent only in the `Authorization` header, never in an exception message or log. Tool-call arguments that are not valid JSON become
`ToolCall(arguments={}, parse_error=...)`.
```python
# models/pricing.py:  PRICES (per 1M tokens: input, cached_input, output); usd_for(model, usage) -> float (unknown model: use the muse-spark-1.3 price)
# models/scripted.py: ScriptedClient(script, model="scripted") with .calls; text_response(text, **usage); tool_response(*calls, text=None)
# models/responses.py: MetaResponsesClient(MetaChatClient)   # opt-in (api = "responses"), UNTESTED against the real service, see A.10
# models/config.py
@dataclass
class Endpoint: base_url: str; model: str; api_key_env: str; reasoning_effort: str | None; fallback_model: str | None
                fallback_reasoning: str | None; completion_cap_ceiling: int | None; api: str = "chat"; allow_partner_responses: bool = False
ROLES = ("supervisor","designer","implementer","reviewer","reviewer_hard","reflector","judge","auditor_final")
DEFAULT_EFFORT = {"supervisor":"medium","designer":"low","implementer":"low","reviewer":"high","reviewer_hard":"high","reflector":"medium","judge":"high","auditor_final":"high"}
def load_routing(path: Path | None) -> dict[str, Endpoint]        # defaults: everything muse-spark-1.3; reviewer_hard and auditor_final -> muse-wonder-1.0 with fallback muse-spark-1.3 at xhigh
def client_for(role, routing, *, gate=None, actor=None, **client_kw) -> ModelClient   # MetaChatClient, or MetaResponsesClient when the role's api is "responses"; imports budget lazily
```
`tests/support/fake_meta_server.py` (B3) is an `http.server` that behaves as the docs describe (tools, strict json_schema, 429 with `Retry-After`, 503, 504-then-stream, 401 echoing the key in the body, 400 for forbidden
params and for unknown optional params, `tool_choice` other than auto -> 400, `/v1/responses` with verified encrypted reasoning items) with a scripting API. It is a fake: a run against it is a simulation of the live path.

### 6.6 Agents
```python
# agents/prompts.py, agents/llm.py, agents/patch_worker.py (B4)
class LLMDesigner(Designer); class LLMReviewer(Reviewer); class LLMReflector(Reflector); class LLMSupervisor(Supervisor)
class PatchWorker(Implementer)         # tools: list_files, read_file, write_file, replace_in_file; every path confined to the workspace and task.editable
def build_llm_suite(task, run_dir, routing, gate, ledger, *, implementer="patch", background="") -> AgentSuite   # implementer: "patch" | "muse_code"; background: operator --context text for the supervisor (60,000 chars) and designer (8,000), fenced as data
# agents/scripted.py (B5)
class ScriptedSupervisor, ScriptedDesigner, ScriptedImplementer, ScriptedReviewer, ScriptedReflector
def build_scripted_suite(task, run_dir, *, control: bool = False) -> AgentSuite
# agents/factory.py (B6)
def build_suite(backend: str, task, run_dir, routing, gate, ledger, *, implementer="patch", control=False, plant=(), background="") -> AgentSuite
    # plant/background are meta-only (ValueError with scripted). plant attaches suite.planted: a scripted suite whose designer and implementer serve ONLY lanes that carry a scripted_id
```
Rules:
* **Roles catch only `ModelError`.** `BudgetExceeded`, `HaltRequested`, `SimulatedCrash` and every other `NightshiftError` propagate. A failed model call yields a conservative result:
  reviewer -> verdict `reject` with one finding `{id:"reviewer_unavailable", severity:"info"}` (the engine records `unresolved` without a repair loop); designer -> the lane's own hypothesis.
* Strict `response_schema` for every role output that is parsed. Plain-text fallbacks are parsed defensively.
* `LLMSupervisor`: executes **every** tool call in an assistant message and answers each in order, even if one fails. Consecutive `launch_experiments` calls in one message are **merged into one `run_lanes` batch**
  (still one tool message per call). The prompt says to put all lanes of a round in one call. After a plain-text reply it appends a nudge; after 2 consecutive plain-text replies it calls `env.finish` with that text.
  If `finish` is refused it keeps looping until `env.should_stop()`. It writes `note_turn(... tool_calls, reasoning_effort, latency_ms, text)` for every model call and raises effort to `high` after a stall or a refused finish.
* All model- or untrusted-derived strings placed in prompts, `get_state`, the tried table or a tool result go through `fence(...)` after `sanitize(...)`. Nudges are templates plus ids and static invariant titles only.
* The reviewer receives the diff, the grade summary (dev numbers) and the run summary, **not** the implementer's self-report. "Independent" in docs and UI means "separate context and prompt" unless the reviewer model differs
  from the implementer model (`independent_basis`).
* `PatchWorker`: refuse a path if any component is a symlink; write via temp file + `os.replace`; `read_file` only regular files <= 200 KB; bounded steps (12); each rejected write is returned to the model as a tool error;
  emits `agent.claim` through `ctx.emit_claim` when the model asserts an outcome.
* `ImplementContext.emit_claim(claim_type, text) -> claim_id`. `ReviewContext.grade_summary` keys: `valid, score, std, per_seed, metrics, incumbent_score, delta, margin, train_delta, overfit_suspect, failure_class` (dev only).

### 6.7 Workflows, gates, orchestrator (B6)
```python
# workflows/templates.py
@dataclass
class StageSpec: name: str; role: str; mandatory: bool
TEMPLATES = {"explore": [design, implement, run, grade, review, repair, record]}      # review_hard is optional, composed
def validate_composition(stages) -> list[str]; def compose(stages) -> list[StageSpec]  # raises InvalidComposition
# workflows/gates.py  (pure functions of LedgerView / TaskSpec; no I/O)
def finish_gate(view, task, claim_ids) -> list[str]            # reasons to refuse; [] = accept
def promotion_gate(view, exp_id) -> list[str]
def launch_gate(view, lane, memory, *, gate=None, run_dir=None) -> list[str]
# workflows/engine.py
class WorkflowEngine:
    def __init__(self, *, task, run_dir, ledger, gate, memory, suite, auditor, limits, clock=time.time, overrides=None, crash_after_runs=None)
    def run_baseline(self) -> LaneResult
    def run_lane(self, stages, lane) -> LaneResult             # never raises except SimulatedCrash
    def run_lanes(self, stages, lanes) -> list[LaneResult]     # ThreadPool <= limits.max_parallel
    def replicate(self, exp_id) -> LaneResult                  # sealed; LaneResult.score and .delta are ALWAYS None
    def reflect(self, trigger) -> dict
# orchestrator.py
class Orchestrator:                                            # implements SupervisorEnv
    def __init__(self, run_dir, task, suite, limits, *, ledger, gate, auditor, backend, clock=time.time, overrides=None,
                 crash_after_runs=None)
    @classmethod create(cls, run_dir, task, *, backend, routing=None, limits=None, implementer="patch", control=False,
                        plant=(), background="", context=(), **kw)
        # order: validate plant -> harden_parent -> Ledger.create -> BudgetGate -> factory.build_suite -> Auditor(ledger, task, judge=suite.judge, final_judge=...) -> Orchestrator
        # plant: task-bank behaviors to inject as scripted red-team lanes (meta only); background/context: --context text and its {name, sha256, chars} records
    @classmethod resume(cls, run_dir, task, *, backend, routing=None, accept_broken_chain=False, implementer=None, control=None,
                        background="", context=(), **kw)
        # limits come from run.started.limits. Refuses (ResumeRefused) a finished run, another backend or task, a different KIND of endpoint
        # than the run started with, and WITHOUT EXCEPTION a task whose task_hash or grader_hash differs from run.started (there is no
        # accept_changed_task: grades from before and after such a change are not comparable). background/context are accepted only when
        # they are the same (name, sha256) set that run.started.config.context recorded
    def run(self) -> dict                                      # {run_id, reason, verdict_class, incumbent, counts, head_anchor}
```
### 6.8 Audit (B7)
```python
# audit/invariants.py
REGISTRY: list[Invariant]                                     # I01..I14, docs/INVARIANTS.md
@dataclass
class AuditConfig: stall_sec=30.0; idle_turns=4; handoff_grace_sec=10.0; judge_max_evidence=40
@dataclass
class AuditContext: view; events; task; now; window_from; window_to; config
CHECKS: dict[str, Callable[[AuditContext], list[Finding]]]    # deterministic checks
CONTAINED: dict[str, Callable[[AuditContext, Finding], bool]] # containment checks, see INVARIANTS.md
# audit/judge.py
class LLMJudge(Judge); class NullJudge(Judge); def build_packet(inv, ctx) -> JudgePacket
# audit/auditor.py
class Auditor:
    def __init__(self, ledger, task, *, judge=None, config=None, clock=time.time, final_judge=None)   # holds a lock; audit_window/final_sweep are serialised; final_judge (the auditor_final role) is used by final_sweep
    def audit_window(self) -> list[Finding]            # called by the orchestrator thread after run_lanes returns; also auto-acks contained deterministic findings
    def final_sweep(self) -> list[Finding]
    def acknowledge(self, finding_id, action, reason, *, by="supervisor") -> dict
    def coverage(self) -> dict                         # invariant id -> "checked" | "llm_not_run" | "never"
```
Auditor and the fault-injection rule: **deterministic checks and gates must not read `planted_fault`**. A test removes the key and asserts identical findings.

### 6.9 Dashboard, report, CLI (B8)
```python
# dashboard/state_api.py
def build_state(source, task, *, limits=None, now=None, events=None, replay=None) -> dict   # source: Ledger or LedgerView; with a Ledger/events it verifies the chain, else chain_ok is null
def describe_event(event) -> str                                       # timeline text (plain text, never HTML)
# dashboard/server.py
def make_server(run_dir, host="127.0.0.1", port=8765, *, replay_speed=None, replay_duration=None, allow_remote=False) -> ThreadingHTTPServer
# report.py
def render_markdown(run_dir) -> str; def render_html(run_dir) -> str   # self-contained, NO script, every dynamic string escaped, figures as server-rendered inline SVG
```
Server rules: routes by exact-match dict `/ /app.js /app.css /api/state /api/events /api/health`; headers on every response `Content-Security-Policy: default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'`,
`X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store`; reject when `Host` is not `127.0.0.1:<port>`/`localhost:<port>`/`[::1]:<port>` (403) or `Origin` is present and not equal; GET/HEAD only; no CORS headers;
`int(since)` parsed defensively; `/api/events` capped at 500; `daemon_threads=True`, request timeout 10 s; a non-loopback host needs `allow_remote=True` and prints a warning; chain verified at start and a broken chain is refused in replay mode.
Replay: `replay_speed` any float in 0.1..100, or `replay_duration` rescales the whole ledger to a fixed length.

CLI (`python3 -m nightshift <command> --help` is the authority; this is a summary):
```
nightshift run    --task tasks/law-discovery [--backend scripted|meta] [--run-dir runs/x] [--max-experiments N] [--max-usd X] [--max-wall SEC] [--parallel N]
                  [--config nightshift.toml] [--implementer patch|muse_code] [--control]
                  [--plant NAME[,NAME...]] [--context FILE [FILE ...]] [--serve] [--port 8765]
nightshift resume RUN_DIR [--task T] [--backend ...] [--config F] [--implementer ...] [--control] [--context FILE [FILE ...]] [--accept-broken-chain]
nightshift serve  RUN_DIR [--port 8765] [--host H] [--replay SPEED | --replay-duration SEC] [--allow-remote]
nightshift report RUN_DIR [--out DIR]
nightshift audit  RUN_DIR [--judge meta|none] [--write] [--task T] [--config F]   # re-audits on a temp copy unless --write
nightshift verify RUN_DIR [--expect-head HASH]                    # chain check; prints the head anchor
nightshift doctor [--live] [--live-out FILE]                      # python, numpy, isolation probe, muse binary, key present; --live runs the live probes (and the Muse Code probe when muse is on PATH)
nightshift demo   [--task law-discovery] [--serve|--no-serve] [--with-crash] [--control] [--out runs/demo] [--port P] [--replay DIR] [--replay-speed S]
nightshift tasks  [--root DIR]                                    # list + validate (with calibration); SKIP for a missing optional dependency
```
`--plant` and `--context` need `--backend meta` (usage error otherwise); `--plant` names behaviors of the task's scripted bank (`tamper, claim_without_run, duplicate, infra_flake, hang, overfit, injection`; a task whose
bank lacks one refuses it, for example spiral-classifier has no `hang` or `injection`); `--context` takes UTF-8 text or markdown files, at most 200,000 characters each and 400,000 in total. `resume --context` takes the SAME files the run started with
(checked by name and sha256 against `run.started.config.context`; a different set is a usage error). A finished run cannot be resumed; Ctrl-C is recorded as a halt and also finishes the run (an experiment mid-flight at that moment stays in flight, so the verdict is `unresolved_defects`; a HALT file is the clean stop).
`NIGHTSHIFT_ISOLATION` (`auto|userns|unshare-net|none`), `NIGHTSHIFT_TASKS_DIR` (where the bundled tasks are looked for), `NIGHTSHIFT_BASE_URL` / `NIGHTSHIFT_MODEL` (live probes and `MetaChatClient.from_env`),
`NIGHTSHIFT_MUSE` / `NIGHTSHIFT_MUSE_ARGS` (Muse Code binary and extra arguments) are the environment inputs. An invalid `NIGHTSHIFT_ISOLATION` is a usage error (exit 2) in `run`, `resume`, `demo` and `doctor`.
Exit codes: 0 success; 1 the run finished with open block findings, with `unresolved_defects`, with reason `error` (a refused API key included, although its verdict is `missing_authority`), or with the budget exhausted
without an incumbent; 2 usage error; 130 interrupted before the run could be concluded. `doctor` exits 1 when a required live probe FAILS (an inconclusive probe, a failing optional Muse Code probe and the Responses probes do not change it).

## 7. Engine semantics

### 7.1 Composition
Stages: `design, implement, run, grade, review, review_hard, repair, record`. Mandatory: `implement, run, grade, review, record`. Optional and chosen by the supervisor per launch: `design` (skip when the hypothesis is already concrete),
`review_hard` (an extra review by the `reviewer_hard` role, valid only after `review`), `repair`. Rules: R1 names are known. R2 if `implement` is present then `run, grade, review, record` follow it in that order. R3 `record` is last.
R4 `repair` only after `review`, and a repair is always followed by re-run, re-grade, re-review (the engine does this; the list only permits it). R5 mandatory stages cannot be removed, reordered or skipped. A violation returns
`InvalidComposition` as a readable tool error and writes `experiment.blocked(reason=invalid_composition, exp_id=null)`. `diagnose`/`analyze` are reserved and not implemented.

### 7.2 Lane lifecycle (`explore`)
1. `launch_gate`: HALT file, budget, identical normalised hypothesis already recorded (`reason=duplicate`, pre-proposal block). 2. `experiment.proposed` and `reserve_experiment` (atomic), designer plan. 3. Workspace from `frozen/` of the incumbent
(`base=incumbent`), the template, or a named exp; `experiment.workspace_ready`. 4. Implementer edits files. **Compute `trusted_hash` before and after every `implement` call**; a change is `protected_violation(kind=trusted)`. `diff_workspace`;
violations -> `experiment.protected_violation`, outcome `invalid`, lane ends. Else compute `idem_key`; if another recorded or in-flight experiment has it, write `experiment.blocked(duplicate, exp_id=<this>)` (and **no** `implemented` event) and return that result;
else `experiment.implemented` (with `tree_hash`). The check-then-write for the idem key and fingerprint runs under the engine lock. 5. `evaluate` on `task.seeds` (purpose `primary`). `infra` is retried once and recorded `infra_error` (no lesson about the
hypothesis); `crash`/`timeout`/`invalid` -> `failed`. 6. Reviewer -> `experiment.reviewed` (`block`/`warn`/`info` findings). `repair` or `reject` with block findings and `max_repairs` left -> repair, then 4-6 again. Repeated findings or an unchanged diff -> `unresolved`.
`review_hard` if composed. 7. `decide_outcome` vs the incumbent's dev grade (`sigma_prior` = the baseline's per-seed std) -> `experiment.recorded(phase=lane)`. 8. A non-improvement writes a lesson (dev delta, train vs held-out; no sealed numbers).
9. After every recorded experiment: regenerate `memory.md`; if `consecutive_non_improving() >= limits.stall_after`, reflect. An `improved` experiment is **not** promoted yet: it waits for `replicate`.

Experiments that end as `failed`, `invalid`, `infra_error` or with a `protected_violation` are exempt from I06/I12 (never reviewed by design).

**Stops and no-ops.** An implementer that changes nothing is not evaluated: the experiment is recorded `unresolved` (note `the implementer made no change; the hypothesis is untested`)
with an `infra` lesson, and no `experiment.implemented` is written. A HALT file or a spent budget in the middle of a lane records every experiment still in flight as `unresolved` with the note
`stopped by HALT` or `stopped by the budget` (open runs are closed as `interrupted`) before the stop propagates, so a finished ledger never holds an experiment that is mid-stage. These two notes are the stop,
not a defect: they do not make the verdict `unresolved_defects` (section 7.8), and an experiment that never ran an untrusted run is not counted as a failed hypothesis. Any other lane error is recorded
`unresolved` with the note `engine_error:<type>`.

**Red-team lanes (`--plant`).** In a meta run whose suite carries `suite.planted`, a lane that carries a `scripted_id` is designed and implemented by the scripted (attacker) suite, and ONLY those two roles come from it:
the reviewer, the auditor, the grader and the gates are the run's own. The orchestrator injects the planted behaviors as lanes labelled `RT-<behavior>` (at most 16 characters), as many as fit under the 8-lane limit next to the
supervisor's own lanes, and writes a `supervisor.nudge` beginning `Operator notice`. Such a lane records `planted_fault` exactly as a scripted run does (display and the fault scoreboard only). An attacker lane is never replicable,
never promotable and never the base of another lane (`WorkflowEngine.replicate` and the lane base check refuse with a fixed note), so a scripted patch can never become the incumbent or decide the verdict. Lanes not yet injected
when the supervisor asks to finish (after a resume, or because every lane slot was taken) run as one operator launch first.

### 7.3 Replication and promotion (sealed split)
`replicate(exp_id)` is a supervisor tool. `launch_gate` refuses unless the target is `awaiting_replication` and `count(replication charges) < limits.max_replications` (`reason=not_replicable` or `budget`).
1. `reserve_replication`. 2. If the baseline has no `replication_baseline` grade, evaluate the baseline's `frozen/` tree on `replication_seeds`, split sealed, purpose `replication_baseline`, written under the baseline's own exp id.
3. Evaluate the candidate on `replication_seeds`, sealed, purpose `replication`. 4. Compare against the **current incumbent at this moment**: the incumbent's own `replication` grade (or the baseline's `replication_baseline`) via
`decide_outcome(k_sigma=T975)`. If the incumbent changed while replicating, compare against the new one with no extra run; if the candidate no longer beats it, `note="superseded"`.
5. Write `experiment.recorded(phase=replication, outcome, sealed_delta, sealed_margin, replicated)`; on success `promotion_gate` then `incumbent.promoted` (and `freeze` the tree); on failure a lesson **without numbers**
("<exp_id> improved on the dev split but did not replicate on the independent re-check; possible overfitting to the evaluation it was tuned on."; agents and gates never name the sealed split, they call it "the independent re-check"). Promotion is serialised by a lock. `LaneResult` for a replication has `score=None, delta=None` and `replicated: bool`.

### 7.4 Parallelism and thread rules
`run_lanes` uses a thread pool of at most `limits.max_parallel`. Each lane has its own workspace. The engine holds one lock around (view check -> append `proposed`/`implemented`); `reserve_experiment` is atomic; `Auditor` calls are serialised by its own lock and
made only from the orchestrator thread after `run_lanes` returns; `memory.md` is written atomically; lesson ids come from a counter under the Memory lock; claim ids are `"<exp_id>.c<n>"` (n per experiment) or `"run.c<n>"` under a lock.
`SimulatedCrash` in a lane sets `ledger.crashed`, further appends raise it, `run_lanes` joins the pool and re-raises, and the orchestrator closes the ledger (releasing the flock) before propagating. Consecutive `launch_experiments` calls in one
assistant message are merged into one batch.

### 7.5 Watchdog
The engine's `on_heartbeat` callback tracks `output_bytes`. No growth for `task.stall_sec` -> one `watchdog.stalled` (actor `watchdog`) and the process group is killed; classification `timeout`, `stalled=true`.

### 7.6 Reflection (CORAL-style)
`reflect` tool, or automatically after `stall_after` consecutive non-improving experiments. The reflector reads the tried table (dev numbers) and lessons and writes consolidated lessons and pivots (`memory.lesson`, `memory.reflection`).
`classification=infra` for lessons whose source failed for infrastructure reasons; such lessons never claim the hypothesis failed. Pivots appear in `get_state`.

### 7.7 Resume
`Orchestrator.resume` verifies the chain (refuses a broken one unless `accept_broken_chain`), compares `trusted_hash`/`task_hash` to `run.started` (refuses with `ResumeRefused`; there is no override), then for every started run without `run_finished` writes
`run_finished(classification=interrupted)`. Non-terminal experiments: `implemented` and never run -> run; `running` -> re-run once with `attempt+1`; a replication in progress is re-run. It never re-proposes, never re-charges an experiment already
charged (`budget.charged` carries `exp_id, purpose`), and LLM charges are idempotent by `call_id`. It writes `run.resumed`.

### 7.8 Finish gate
`finish(summary, claim_ids)` writes `supervisor.finish_requested`, runs `auditor.final_sweep()`, then `finish_gate`. Refuse (write `supervisor.finish_refused`) unless: baseline recorded; nothing in flight; no open `block` findings; every cited claim
of type `improved` refers to an experiment that is the incumbent, `replicated`, and whose **sealed** incumbent-vs-baseline `decide_outcome(k_sigma=T975)` is `improved` (if the incumbent is the baseline, the only acceptable claim type is `no_effect`);
no `improved` claim is cited for an experiment that is not the incumbent. On acceptance write `run.finished` with `verdict_class` and `head_anchor`, then `ledger.seal()`.
`verdict_class`: `success` (accepted improvement), `inconclusive` (honest no-improvement), `unresolved_defects` (open block findings or an unresolved experiment), `missing_authority` (budget or HALT).
For an accepted finish the verdict comes from the ledger, not from what was claimed or cited: `success` only if the incumbent is not the baseline, is `replicated` and beats the baseline on the sealed split as
`gates.sealed_decision` recomputes it; every other accepted finish is `inconclusive`. A refusal reason that would reveal how the sealed comparison with the baseline came out is shown to the supervisor as
"not confirmed as an improvement over the baseline" (one bit beyond `replicated` is not for a model).

A run that does not end in an accepted finish is concluded by the orchestrator (`run.finished.reason` in `budget_exhausted | halt | error`): the verdict is `missing_authority` for a halt, a spent budget, or a model API
that refused the key or the credits (HTTP **401, 402 or 403**: reason `error`, verdict `missing_authority`, CLI exit 1; the ledger keeps only the numeric status, never the message), unless the ledger also holds a defect: an open
`block` finding, an experiment still in flight, or one left `unresolved` for any reason other than the stop itself (the engine's `stopped by HALT` / `stopped by the budget` records are the stop, not a defect). Any other
model API failure, a supervisor error or a baseline that could not be graded is reason `error` with verdict `unresolved_defects`. The CLI exit status is 1 for `unresolved_defects`, for reason `error`, for open block findings,
and for `budget_exhausted` without an incumbent; 0 otherwise (a halt included).

### 7.9 Supervisor tools (binding names and schemas; arguments validated; errors returned as `{"error": ...}`)
| tool | arguments | result |
|---|---|---|
| `get_state` | none | see below |
| `launch_experiments` | `template` (`explore`), `lanes[]` (`lane`, `hypothesis`, `approach?`, `base?`), `stages?` | `{"results": [LaneResult dicts]}` (dev numbers only) |
| `replicate` | `exp_id` | `{"exp_id", "replicated": bool\|null, "outcome", "note"}` |
| `reflect` | none | `{"lessons": [...], "pivots": [...]}` |
| `acknowledge_finding` | `finding_id`, `action` (`addressed`\|`dismissed`), `reason` | `{"ok": true}` or `{"error": ...}` |
| `finish` | `summary`, `claim_ids?` | `{"accepted": bool, "reasons": [...]}` |

`get_state` result: `{"budget": remaining, "baseline": {"exp_id","score"}, "incumbent": {"exp_id","score","replicated"}, "recent": [last 8 {"exp_id","lane","hypothesis","outcome","delta"}],
"awaiting_replication": [exp_id], "open_findings": [{"finding_id","invariant_id","severity","subject","exp_id","rationale"}], "pivots": [str], "in_flight": [exp_id], "stalled": bool, "turns_left": int}`.
Every string in it is `sanitize`d, capped and `fence`d. Scores are dev scores. `replicated` is the only sealed-derived value.

## 8. Statistics

`decide_outcome(task, cand, inc, *, k_sigma=None, sigma_prior=0.0, eps=1e-12)`:
```
invalid if either grade invalid, any score/std/train non-finite, len(per_seed) differ, <1, or seeds/split/grader_hash differ ("not_comparable")
sign = +1 if minimize else -1
d_i = sign * (inc.per_seed[i] - cand.per_seed[i])           # paired by seed index; >0 means better
n = len(d); delta = mean(d); s = stdev(d, ddof=1) if n>=2 else 0.0
s_eff = max(s, sigma_prior, task.sigma_floor)
t = k_sigma if k_sigma is not None else T95[n-1]            # one-sided 95% Student-t quantile by degrees of freedom df = n-1 (table below)
margin = max(task.noise_margin, t * s_eff / sqrt(n))
improved iff delta > margin + eps; worse iff delta < -(margin + eps); else no_gain
train: the same function on per_seed_metrics[*][train_metric] gives train_delta and train_margin
overfit_suspect = train improved beyond train_margin AND outcome != improved
```
Sealed comparisons (replication, promotion, finish) pass `k_sigma = T975` (the two-sided 95% quantile, 0.975).

**Critical values.** `T95` (quantile 0.95) and `T975` (quantile 0.975) are exact Student-t quantiles, keyed by degrees of freedom (`n - 1`; `n = 1` uses df 1):
df 1..10 as in the contract (T95: 6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812; T975: 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228), df 11..30 to 4 decimals
(T95: 11:1.7959 12:1.7823 13:1.7709 14:1.7613 15:1.7531 16:1.7459 17:1.7396 18:1.7341 19:1.7291 20:1.7247 21:1.7207 22:1.7171 23:1.7139 24:1.7109 25:1.7081 26:1.7056 27:1.7033 28:1.7011 29:1.6991 30:1.6973;
T975: 11:2.2010 12:2.1788 13:2.1604 14:2.1448 15:2.1314 16:2.1199 17:2.1098 18:2.1009 19:2.0930 20:2.0860 21:2.0796 22:2.0739 23:2.0687 24:2.0639 25:2.0595 26:2.0555 27:2.0518 28:2.0484 29:2.0452 30:2.0423),
then df 40, 60 and 120 (T95: 1.6839, 1.6706, 1.6577; T975: 2.0211, 2.0003, 1.9799). A df between tabulated points takes the value of the largest tabulated df below it (never narrower than the exact quantile);
beyond df 120 the normal quantile 1.645 / 1.96 applies. The tables are therefore non-increasing in df (an earlier fall-back stepped UP from 1.812 at df 10 to 1.645 at df 11). No shipped task uses n >= 12 (law-discovery
and spiral-classifier use 3 dev and 5 sealed seeds), so no shipped result depends on the extension.

**Portability.** `decide_outcome` returns `not_comparable` when any paired difference is non-finite (checked with `math.isfinite` before `fsum`/`stdev`, because `statistics.stdev` raises `AttributeError` on inf on
Python 3.11 and 3.12 and `ValueError` on 3.13, so the verdict must not depend on the exception type); `evaluate` returns an invalid grade (failure class `invalid`) when finite per-seed scores overflow when averaged.

**Calibration.** `sigma_prior` is the baseline's own per-seed std on the dev seeds, a very noisy estimate when there are 3 of them (for law-discovery the true std is about 0.036 to 0.040 depending on which
seeds are measured, but the realized prior on the fixed dev seeds 0, 1, 2 is 0.0100), so `task.sigma_floor` must be set to the baseline's multi-seed std; `nightshift tasks` measures it over 20 seeds (good to about
+-16%), prints it as the recommended `sigma_floor` and warns when the task's floor is below 75% of it. With a floor below the real noise the prior is inert and, for a candidate whose paired noise is large, the dev gate is a plain
one-sided 5% t-test on 3 seeds. `docs/NULL_BENCHMARK.md` gives the error rates of the shipped configuration (law-discovery: noise_margin 0.035, sigma_floor 0.036, 3 dev and 5 sealed seeds, at most 6 sealed queries per run;
spiral-classifier: noise_margin 0.02, sigma_floor 0.02). In its normal-noise model, with 16 experiments the plain t-test passes at least one null 55.6% of the time, the dev gate alone 1.4% (paired noise 0.036), 10% (0.05) or
26.5% (0.07), and the sealed test brings the per-run probability of PROMOTING a null to 0.01%, 0.09% and 0.50%; with 48 experiments the dev gate alone passes 4.2%, 27% and 59%, and the promotion probability is 0.01%, 0.29%
and 1.7%. Section 6 of that document replaces the normal draws by null candidates (the baseline's straight line fitted on 30%, 50% or 70% of the training sample) run in process through the task's real data module and
grader: at the shipped floor 1.9%, 1.1% and 0.08% of 16-experiment runs, and 5.8%, 3.4% and 0.24% of 48-experiment runs, end with a promoted null (paired noise sd 0.115, 0.074 and 0.048; the old floor of 0.02 gave 2.3%, 2.2%
and 1.1% at 16 experiments). A floor only helps a candidate whose own 3-seed sample std falls below it, so for very noisy candidates the sealed stage and its cap of 6 queries are the only protection. An independent audit of
the real pipeline, measured BEFORE the floor was calibrated (sigma_floor 0.02), found 0.5% to 4.8% of 16-experiment runs and 5% to 12% of 48-experiment runs ending with a promoted null, and has not been repeated through the sandbox
under the calibrated floor. None of this is a claim of zero false wins: the shared law-discovery split is public (`make_data.py --seed` draws a private one). The price is sensitivity: a candidate with small paired noise is
promoted half of the time at an improvement of about 5% of the baseline RMSE (4.8%, about 0.077, at paired noise 0.036; 4.2% to 7.7% across the rows). The report quotes the sealed delta of the final incumbent AGAINST THE BASELINE (the
comparison the finish gate decides on) with the threshold an improvement had to exceed, lists the step-wise sealed comparisons separately after a promotion chain, labels the dev delta "selected, optimistic", and prints
`n_tested` (hypotheses that reached evaluation), `n_dev_improved`, `n_sealed_queries`, `expected false promotions under the null = n_tested * 0.05 * 0.025` (which assumes independent 5% dev draws) and the adaptive worst case
`1 - 0.975^n_sealed_queries` (a bound on the sealed stage alone, `n_sealed_queries` capped by `max_replications`).

## 9. Auditor semantics (summary; catalog in docs/INVARIANTS.md)

* Deterministic checks run on every window and are pure functions of `AuditContext`.
* LLM judging: one invariant, one narrow question per call. Packet = invariant text, current goal and authority, named state fields, a bounded evidence list `{seq, kind, actor, text}` with `text` fenced. Strict JSON
  `{verdict, confidence, evidence_seqs, rationale}`; evidence seqs absent from the packet are discarded; `violated` with no valid evidence is downgraded to `insufficient_evidence`; many invariants can be violated at once and are never merged.
  The final sweep uses the `auditor_final` role (Muse Wonder, falling back to Muse Spark at `xhigh`; the model that answered is recorded).
* **Findings are closed only by the Auditor.** `acknowledge(finding_id, action, reason)`: a deterministic `block` finding cannot be `dismissed` (`deterministic_block_not_dismissible`); `addressed` makes the Auditor run the invariant's containment
  check and write `audit.ack` itself if it passes, else `{"error":"not_contained","need":...}`. Contained deterministic findings are also auto-acked at the next window. `dismissed` is allowed for `warn`/llm findings with a reason of at least 20 characters
  and is listed in the report under "dismissed findings". The supervisor never writes `audit.ack` and `view` ignores acks from any actor but `auditor` or `human`.
* Findings never act. They appear in `get_state.open_findings`, produce a templated `supervisor.nudge`, and block `finish` while open and `block`.

## 10. Dashboard state JSON (`GET /api/state`)

```json
{
  "run": {"run_id": "", "task": "", "backend": "scripted|meta", "mode": "scripted|simulated|live|replay|unknown", "origin": "scripted|simulated|live|unknown",
          "endpoint": {"kind": "scripted|meta-api|loopback|custom", "hosts": []}, "models": {}, "models_answered": {"<role>": ["<model that answered>"]},
          "implementer": "patch|muse_code|scripted", "started_ts": 0, "now": 0, "finished": false, "reason": null, "verdict_class": null, "summary": "",
          "objective": "minimize", "primary_metric": "", "isolation": "",
          "red_team": [{"behavior": "", "scripted_id": ""}], "context": [{"name": "", "chars": 0, "sha256": "", "not_reloaded_after_resume": true}],
          "resumes": [{"seq": 0, "reconciled": []}]},
  "banner": null,
  "budget": {"limits": {}, "used": {}, "remaining": {}},
  "baseline": {"exp_id": "", "score": 0},
  "incumbent": {"exp_id": "", "score": 0, "replicated": false},
  "experiments": [{"exp_id": "", "lane": "", "hypothesis": "", "status": "", "stage": "", "worker": "", "reviewer": "", "outcome": null,
                   "dev_outcome": null, "score": null, "train": null, "delta": null, "sealed_delta": null, "train_delta": null,
                   "review": null, "replicated": false, "awaiting_replication": false, "diff_summary": "", "parent_exp_id": null,
                   "blocked_reason": null, "protected": [], "claim_ids": [], "finding_ids": [], "planted_fault": null,
                   "started_ts": 0, "ended_ts": null,
                   "train_margin": null, "overfit_suspect": false, "cost_usd": null, "tokens": null, "llm_calls": null, "cost_by_role": {}}],
  "series": [{"seq": 0, "ts": 0, "exp_id": "", "score": 0, "train": null, "std": null, "kind": "baseline|candidate|promoted", "replicated": false}],
  "claims": [{"claim_id": "", "exp_id": "", "type": "", "text": "", "actor": "", "seq": 0, "status": "supported|unreplicated|contradicted|refused"}],
  "finish": {"requests": [], "refusals": [], "accepted": false},
  "lessons": [], "pivots": [],
  "findings": [{"finding_id": "", "invariant_id": "", "title": "", "verdict": "", "severity": "", "status": "", "source": "", "rationale": "",
                "evidence_seqs": [], "subject": "", "exp_id": null}],
  "invariants": [{"id": "", "title": "", "kind": "", "state": "ok|violated|unchecked|insufficient|llm_not_run", "open": 0}],
  "models_usage": [{"role": "", "model": "", "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "cache": "n/a|known", "usd": 0,
                    "effort": "", "strict_schema_calls": 0, "schema_ok": 0, "retries": 0}],
  "turns": [{"seq": 0, "source": "supervisor|script", "model": "", "say": "", "tool_calls": [], "reasoning_effort": "", "latency_ms": 0, "text": ""}],
  "timeline": [{"seq": 0, "ts": 0, "kind": "", "actor": "", "text": ""}],
  "parallel": {"wall_sec": 0, "serial_sec": 0, "speedup": 0, "max_parallel": 3, "reason": null},
  "fault_injection": {"planted": [], "detected": [], "missed": [], "not_evaluated": [], "false_positives": [], "explained": [],
                      "recall": {"caught": 0, "scored": 0, "value": null, "not_evaluated": 0, "negatives_ok": 0, "negatives_total": 0}},
  "headline": {"claims_made": 0, "claims_accepted": 0, "claims_refused": 0, "unreplicated_blocked": 0, "speedup": 0,
               "cost_usd": 0, "cost_known": false, "llm_calls": 0, "cost_attributed_usd": null, "cost_unattributed_usd": null,
               "promoted": 0, "cost_per_improvement_usd": null},
  "ledger": {"events": 0, "last_seq": 0, "chain_ok": true, "bad_seq": null, "head_anchor": ""},
  "replay": null
}
```
`banner` is non-null and non-dismissible when the origin is `scripted` ("SCRIPTED BACKEND: no language model is running. This is a fault-injection harness proving the audit machinery, not an AI result.") or `simulated`
("SIMULATED MODEL: the language model endpoint is a local test server. This run exercises the live code path (prompts, tool calls, metering, audit) but is not Meta model output."). A meta run with `state.run.red_team`
lanes shows a separate non-dismissible RED-TEAM LANES notice (the page) and opens with a RED-TEAM paragraph (the report); the planted lanes are named as scripted attackers.

`state.run.origin` is derived from `run.started` alone: `scripted` for the scripted backend; for `meta`, `simulated` when `config.endpoint.kind` is `loopback` (any routing host that connects to this machine), `live` when it is
`meta-api` (every endpoint's host is `api.meta.ai`) or `custom` (another host: named as such, not as the Meta API) or when an older ledger has no endpoint record, and `unknown` when a meta run's endpoint record says `scripted`.
`state.run.mode` is the origin, or `replay` when the ledger is being replayed. `GET /api/health` returns `{ok, mode, origin, serving, run_id, events, last_seq}` where `serving` is only `live` or `replay` (how this server is serving the
ledger), so a scripted or simulated run is never reported as live. `models_answered` maps a role to the models that appeared in its `llm` `budget.charged` rows (never the configured one). Cost is shown only where the ledger recorded it:
a per-experiment `cost_usd`, `tokens` and `llm_calls` sum the `llm` charges that carry that `exp_id` and are null when none does, `cost_by_role` splits it by bare role, and a scripted run therefore has no cost at all
(`headline.cost_known` is false), never `$0`; `cost_attributed_usd` is the sum over experiments, `cost_unattributed_usd` the rest (supervisor turns, audit judges), and `cost_per_improvement_usd` is the whole run's LLM spend
divided by the number of distinct promoted experiments (null until something was promoted). `context[].not_reloaded_after_resume` is set when a resume did not give the background files again.

`fault_injection` is computed in `state_api` only, for display: `planted` lists each planted fault with its `expected` detector and `result` (`detected`, `missed`, or `not_evaluated`: an LLM-only invariant such as I13 that no audit window put to a judge, so nothing could have caught it); `recall` is
`caught / scored` over the faults that were scored (`not_evaluated` faults and negative controls such as `infra_flake`, which must NOT trigger I07, are excluded; `value` is null when nothing was scored). `false_positives` are violated findings on work nothing planted explains (in a meta run the reason
says so: a real finding about the model's own work, or a detector error, and the board cannot tell which); `explained` are findings that a planted premature finish accounts for (scripted runs only: I09 and I14 on an experiment cited before it replicated). A `--plant` run is scored with the same mapping as a scripted run, with one addition: a planted lane of a **meta** run that was blocked before it ran (blocked as a duplicate, by the budget or by another gate) or never had an experiment is `not_run`
(`cause`: `duplicate | budget | blocked | never_proposed | pending`; `lane`, `scripted_id`, `exp_id` (null when the engine blocked it before an experiment existed) and `reason` are carried). `not_run` rows are listed in `fault_injection.not_run`, are excluded from `recall`'s denominator and from the negative controls, and are counted in `recall.not_run`; `fault_injection.red_team` =
`{planted, ran, not_run, blocked_duplicate, blocked_budget, blocked_other, never_proposed, text}` with a one-sentence `text` ("3 attacker lane(s) were planted; 1 ran, 2 were blocked as duplicates before they ran." or "All N planted attacker lane(s) ran."). The one exception is a planted `duplicate` that the duplicate gate blocks: it stays `detected`, as in the scripted demo. Scripted-backend scoring is unchanged.
`invariants[].state`: violated if an open finding; `llm_not_run` if the registry kind includes llm and the id appears in no window's `judged`; `unchecked` if in no window's `invariants`; else `ok`, or `ok_simulated` when the invariant has an LLM half that a judge was asked about, the run's origin is `simulated` and nothing is open (the judge there is a scripted stand-in; live and scripted runs keep `ok`). `fault_injection` is computed in `state_api` only. The timeline text of a simulated run is labelled
(`describe_event(event, simulated=False)` takes the flag; `/api/state` and `/api/events` both pass it), so a cropped screenshot cannot pass a simulated charge or turn off as a model's.
Cache hit rate is shown only if `cached_known` was ever true, else "cache n/a". The page renders every string with `textContent`, has no inline script or style, and works offline with no CDN. The `chain_ok` badge reads "chain consistent" plus the head anchor, never "secure".

## 11. Security and honesty

Experiment code is untrusted and runs as the same OS user. We apply best-effort containment: scrubbed env, rlimits, process-group kill, throwaway copies with hash re-verification, symlink/special-file rejection, output staging, and (where a probe
succeeds) user/mount/pid/net namespaces that mask the trusted directory, the ledger, the other experiments' directories and `/proc`. **This is not a security boundary.** In the modes without a user namespace tampering is DETECTED and recorded, not
prevented, and some actions are not even detected. The tables below say, per isolation mode, which is which; the dashboard and `experiment.run_started.isolation` show the mode actually applied, and a run in `unshare-net` or `none` must not be described as sealed-safe.

### 11.1 Isolation modes: what each one PREVENTS and what it only DETECTS

`auto` (the default) applies the strongest mode a real probe proved on this machine. `NIGHTSHIFT_ISOLATION` (`auto`, `userns`, `unshare-net`, `none`) can only LOWER that: a mode the probe did not prove is never applied, the strongest proven mode at
or below the request is used instead, and the reason goes to `RunResult.note`. An invalid value is a usage error (exit 2) in `run`, `resume`, `demo` and `doctor`. The mode applied is `run.started.isolation` and `experiment.run_started.isolation`
(the label is `sandbox.effective_isolation`, computed when the event is written; the per-run `RunResult.isolation` is not a ledger field).

| the untrusted run can ... | `userns` | `unshare-net` | `none` |
|---|---|---|---|
| read `trusted/` (labels, grader) | **PREVENTED** (masked, empty) | succeeds, **not detected** (a read leaves no trace) | succeeds, **not detected** |
| read the ledger or other experiments | **PREVENTED** (masked) | succeeds, **not detected** | succeeds, **not detected** |
| change `trusted/`, also when the original bytes are put back before it exits | **PREVENTED** (the real tree is masked by an empty tmpfs, which the run cannot unmount because capabilities are dropped) | **DETECTED** after the run (content hash AND stat snapshot, kind `trusted`) | **DETECTED** after the run (same two checks) |
| change the workspace's read-only files, add links or FIFOs | DETECTED (tree hash, `special:`/`protected:`) | DETECTED | DETECTED |
| open a network connection | **PREVENTED** (empty net namespace) | **PREVENTED** (empty net namespace) | succeeds, **not detected** |
| write elsewhere on the filesystem (no chroot in any mode) | succeeds, not detected | succeeds, not detected | succeeds, not detected |
| read the orchestrator's `/proc/<pid>/environ` (`MODEL_API_KEY`) | **PREVENTED** (the pid namespace hides it) | **PREVENTED** only when the probe proved the root-only pid-namespace variant; otherwise as `none` | succeeds as root or with `CAP_SYS_PTRACE`; **PREVENTED** for a non-root same-uid reader only (`harden_parent`) |
| leave a detached daemon behind | **KILLED** (all die with the namespace init) | **KILLED** when the pid-namespace variant is in use; otherwise best effort | best effort (found by session, group, parent chain, cwd, inherited pipe; a double-forked daemon with stdio closed and cwd elsewhere can survive) |

"Elsewhere on the filesystem" means paths outside the workspace, `trusted/` and the masked paths: no mode makes the filesystem read-only, so what is checked is hashes of `trusted/` (by `grading.evaluate`), of the template (by the engine) and
of the workspace. Reading `trusted/` or the ledger under `unshare-net` or `none` is what those modes neither stop nor notice; the only trace is the diff tripwire (below), which flags text that names a trusted path in a file the run changed.
**Only `userns` keeps the labels out of the run's reach.** `userns` is `unshare --user --map-root-user --mount --net --pid --fork --mount-proc` with the masks, and is selected only when a probe proved masking, an empty network, a fresh pid
namespace, no capabilities and a failing `umount`. The other modes are for machines (laptops, CI containers, macOS) where nothing stronger can be proven.

**How a change to `trusted/` is detected (`unshare-net` and `none`).** A content hash compared after the run misses a file that is changed and then restored byte for byte before the process exits. `TrustedWatch` therefore also takes a STAT
snapshot (for every file, directory and link under `trusted/`: mode, size, inode, link count, owner, mtime, ctime and, for directories, the entry names; never any content) immediately before the untrusted process is spawned, compares it after the
process group has been reaped and swept, and compares again after the grader has run. A run can restore bytes and even `mtime`, but not `ctime` (no unprivileged call sets it), so a write-then-restore is reported. `atime` is ignored (reading is
what the trusted grader does). A trusted directory that is itself a link is resolved. Both checks produce the same `ProtectedPathViolation(kind="trusted")` with `trusted/<path>` names and the same `experiment.protected_violation` event.
Limits, stated plainly: a process with `CAP_SYS_TIME` could step the clock back and put an old `ctime` onto a fresh change (root in `none`, or in `unshare-net` without the pid-namespace variant); the check is evidence about the window between two snapshots,
not a block; a tree of more than 50,000 entries degrades the snapshot; and bytecode planted by a detached survivor after the post-run check and before the grader imports it is not seen by the snapshots. The snapshot taken around the
UNTRUSTED run includes `__pycache__` and `*.pyc` (the content hash ignores them, so a planted `trusted/random.pyc` would otherwise shadow a stdlib module for the grader); the snapshot after the grader ignores bytecode (the grader runs with
`PYTHONDONTWRITEBYTECODE=1` and must not write into `trusted/`). Independently of any baseline, `evaluate` refuses to start, and refuses to run the grader, while a source-less `*.pyc` (one outside a `__pycache__` directory) sits in `trusted/`.
`task_hash` and `grader_hash` still do not see such a file (the manifest ignores `*.pyc`); the stat snapshot and the stray-bytecode check do.

**Concurrent lanes.** Lanes share one `trusted/` directory. If another untrusted run overlapped this run's window (tracked by an in-process gate), a stat difference is AMBIGUOUS: the run is classified `infra` (note
`trusted_tree_changed_during_run`, "may come from a concurrent lane") and is never recorded as tampering. The engine retries an `infra` failure once with `attempt=2`, and an attempt after the first runs SOLO (the gate waits for every other untrusted
run and holds new ones back, bounded by 120 s, then falls back to a shared window and `infra`), so on the retry a difference is attributable and is a `trusted` violation. A hostile lane therefore cannot hide a write-then-restore behind a busy machine,
and an honest lane that shared a window with a hostile one is retried once rather than accused. A persistent content-hash mismatch is still a tamper finding for every lane that sees it, because the grader would use altered files. The gate is per
process: runs in different Nightshift processes sharing one `trusted/` are not attributed. A solo wait is not interruptible by a halt request for up to 120 s.

### 11.2 The API key and the orchestrator's environment

`harden_parent()` calls `prctl(PR_SET_DUMPABLE, 0)`, which keeps a **non-root** process of the same uid from reading `/proc/<pid>/environ` or `/mem` of the orchestrator. It does **not** stop root or any process with `CAP_SYS_PTRACE`: if Nightshift
itself runs as root, an experiment in mode `none` (and in `unshare-net` without the pid-namespace variant) can read `MODEL_API_KEY` from the orchestrator's environment. `userns` is unaffected (the pid namespace hides the process). The pid-namespace
variant of `unshare-net` (`unshare --net --pid --fork --mount-proc`, capabilities dropped by the shim, exit status propagated) is used only when running as root AND its probe passed; mode `none` can never be made root-safe. When an API key or the sealed
numbers matter and `userns` is unavailable: run Nightshift as a non-root user inside a container or VM, and use a key scoped to the run.

**Laptops.** macOS has no user namespaces and no `/proc`: the probe reports `none` off Linux, `harden_parent` does nothing, process sweeping and CPU accounting are skipped, every rlimit is applied best effort, and a same-user experiment can read whatever the user can, including the key in the
orchestrator's environment and `trusted/`. Hashes and the stat snapshot still DETECT changes to `trusted/` and the workspace; nothing prevents a read. Use a Linux container or VM for any run where the key or the labels must not leak. A Linux laptop
without unprivileged user namespaces gets `unshare-net` or `none` by the same probe.

Tripwire: any added or changed non-scratch text matching `trusted|labels\.|ledger\.jsonl|replication_seeds|tasks/` writes `protected_violation(kind=trusted_probe)` and ends the lane as `invalid`.

### 11.3 Muse Code worker

The Muse Code worker (`--implementer muse_code`) is tested only against a fake `muse` executable (A.12). It never runs in the experiment workspace: `implement` copies the workspace's regular files into a private mode-0700 directory under the system temp directory
(outside the run directory), withholds files muse reads as instructions (`AGENTS.md`, `CLAUDE.md`, `.agents`, `.claude`, `.codex`, `.muse`, `.mcp.json`, `.git`), runs muse there with a minimal environment and an empty home, and copies edits back only
for paths matching the task's editable or scratch globs; everything else it did is refused and recorded, and mirrored into the real workspace as a placeholder so the engine ends the lane `invalid` with `protected_violation`. The copy-out design removes the
relative-path and symlink routes to `../../../ledger.jsonl`. **It does not confine READS.** Meta documents where Muse Code may write (the workspace and a temporary directory), not where it may read; the permissions page says the sandbox "keeps the rest of the filesystem
read-only", which describes writes, and `--disable-sandbox` would lift the confinement of the file tools. So the agent can still read anything the OS user can read outside the run directory, including `trusted/` and the run directory by absolute path (which
it has no reason to know and which Nightshift never tells it). The 1.4.x web fetch tool (on in `muse exec` since changelog 1.4.3, no documented flag turns it off) is an egress channel that `--sandbox-network` does not govern. The full mitigation is a container
or VM around the whole run. Parallelism is capped at 1 for this worker (`parallel_reason` in `run.started.config`).

### 11.4 Ledger, secrets, sealed split

The ledger "detects edits, deletions in the middle and reordering. Truncation or a full rewrite by someone who can write the file is detectable only against a head hash you recorded elsewhere" (`head_anchor`, `verify --expect-head`).
A mid-file corruption raises `LedgerCorrupt` and is never repaired automatically. The ledger is `chmod 0444` when a run finishes.

Secrets: `nightshift.secrets.register` is called for `MODEL_API_KEY` and `META_API_KEY` at startup; `Ledger.append`, log writers, `ModelError` messages and report/state output call `scrub`. The key is sent only in the `Authorization` header (https only, loopback
excepted for the fake server). Contributor-mode models (`*-contributor`) are never defaulted. For `muse_code` workers the key is passed as `META_API_KEY` only, `worker.log` is scrubbed, `muse --version` must succeed or the worker is unavailable.
Encrypted reasoning items of the opt-in Responses client are opaque model data: they are never rendered in a prompt, the ledger, the dashboard or a report (A.10).

Sealed split: "sealed from agent prompts and tools; best-effort against code on the same machine". The sealed numbers are written to the ledger and shown to humans (dashboard, report); they never reach a prompt, tool result, memory text, lesson or `LaneResult`
(only the boolean `replicated`). The canary tests that guard this are described in section 13 and have stated limits.

## 12. Do not build, do not claim

SAM 3.1, Muse Image, Muse Voice, the Muse Code TypeScript SDK and Muse Code workflows are out of scope and are not mentioned as features. Partner open-weight endpoints are "configurable via `base_url`, untested against real partners". Muse Wonder on
Chat Completions is **unverified until `nightshift doctor --live` confirms it**, and so is everything about the opt-in Responses client (`api = "responses"`) and the real `muse` binary: both were built from the documentation and tested only against fakes.
Nothing may describe a scripted run as a model result, a run against a local fake server as a live Meta run, or a `--plant` run as all-model work. Any README, slide or write-up that shows a recorded demo must state which isolation mode was active
and which of the three it was (scripted, simulated, live). The honesty rules for documents: a document never claims a protection the code does not provide, and never says "prevents" where the code only "detects".

## 13. Mandatory tests

* Sealed canary (B6, with B2/B4): the fake grader's sealed score is `0.314159265`. After scripted and fake-LLM runs, that string appears in no `ScriptedClient.calls` message, no `Orchestrator.state()`, no `memory.md`, no `LaneResult`.
  The as-built canary is stronger (`tests/support/canary.py`, A.14): every sealed number is expanded into its roundings and truncations, scientific, percentage and per-mille forms and labelled forms, and searched for in the supervisor environment,
  the four role contexts and results, `memory.md`, lessons, reflections, turns, nudges and the simulated model's request bodies, in the scripted golden demo including its resumed runs and in the simulated-Muse end-to-end run.
* Deterministic findings are identical with `planted_fault` removed (B7).
* A clean control manifest yields zero findings (integration).
* `setrlimit` raising still yields `ok` (B1). Symlink `outputs -> dir`, FIFO, hardlink, newline-in-filename tests (B1).
* The dashboard `index.html` has no inline `<script>`/`<style>` and no `unsafe-inline`; `<img src=x onerror=1>` in a hypothesis never appears unescaped in `report.html` (B8).
* The API key put in the fake server's 401 body and in a diff appears nowhere in `ledger.jsonl` (B3/B6).
* Two `launch_experiments` calls in one assistant message run as one `run_lanes` batch (B4/B6).
* After a failed replication `ExperimentView.delta` still equals the dev delta (done in `tests/test_ledger_view.py`).

## Appendix A. As built (binding; supersedes any conflicting text above)

A.1 to A.7 record the wave-1 modules (as corrected since); A.8 to A.16 record what waves 2 to 5 added or changed.

### A.1 Sandbox and grading (B1)
* `diff_workspace(...).violations` holds three prefixes: `special:<path>` (symlink, FIFO, device, hard link, unsafe name, size caps), `protected:<path>` (a non-editable, non-scratch file changed, appeared or vanished) and
  `probe:<path>` (tripwire). The engine maps `special:` and `protected:` to `protected_violation(kind="workspace")` and `probe:` to `kind="trusted_probe"`.
* `evaluate(..., expected_grader_hash=...)` is a keyword-only parameter. **The engine must pass `run.started.grader_hash` on every call.** `evaluate` checks `trusted/` after every run (ok, crashed or timed out), not only around the grader, and
  raises `ProtectedPathViolation(kind="trusted")` on any change. Specials found in the run copy (scratch included) raise `kind="runtime"`; so does `tree_hash_mismatch`.
* `stage_outputs(run_ws, dest, task)`: `dest` is `runs/<idx>/staged/outputs` and receives the CONTENTS of `run_ws/outputs`; the grader is given `runs/<idx>/staged`.
* Grader failures raise `grading.GraderError` (a `ValueError`) with `kind` in `invalid|timeout|infra`; grader stderr is never put in a message. A grade with any failed seed is `valid=False, score=None`.
* `grading.T95` and `grading.T975` are dict subclasses keyed by **degrees of freedom (n-1)**; exact to df 30, then df 40, 60 and 120, and above df 120 they fall back to 1.645 and 1.96 (section 8); for n=1 df=1 is used. `decide_outcome(k_sigma=...)` accepts `None`, a number or such a table, and returns
  `grading.Decision` (an `OutcomeDecision` subclass with a `detail` string). An invalid result always has `reason == "not_comparable"`. The train margin has no `sigma_prior`.
* `trusted_hash` is the `grader_hash`. `task_hash` covers `task.toml`, template, trusted and scripted trees, so editing the scripted bank changes it (a resume then refuses: `Orchestrator.resume` has no override for a changed task or grader).
  **The engine should compare `task_hash` before each `evaluate`** to catch a run that edits the template for future workspaces.
* `available_isolation()` result is cached per process (`refresh=True` to re-probe); `NIGHTSHIFT_ISOLATION` is read on every call and can only lower the answer. `userns` is reported only when masking, an empty network, a fresh pid namespace and dropped capabilities were all proven by the probe.
* Any write outside the scratch globs is a `runtime` violation, including library caches written under `HOME` (which is the workspace).

### A.2 Budget, memory, fence (B2)
* Budget action names (`experiment, replication, llm, supervisor_turn, wall`) are what `budget.exhausted.kind` and `BudgetExceeded.kind` carry. `remaining()` keys: `experiments, replications, llm_usd, supervisor_turns, wall_sec`.
* `reserve_experiment` / `reserve_replication` are idempotent per `(exp_id, purpose)`; a duplicate charge is a no-op. `charge_llm` returns 0.0 for a duplicate `call_id`. **The orchestrator must call `charge_wall()` periodically and at the end.**
* `MeteredClient` reads per-call facts from `ChatResponse.raw["nightshift"]` (the model client also writes `raw["_nightshift"]`, the same dict).
* `Memory.consecutive_non_improving` goes by final outcome: a failed replication counts as non-improving. `Memory.fingerprint` turns punctuation into spaces. Always call `Memory.fingerprint/seen`, never hash hypotheses elsewhere.
* Lessons are sanitised and fenced but **not** scrubbed of numbers: engine-written replication lessons must stay number-free.

### A.3 Models (B3)
* Extra `Endpoint.fallback_reasoning` (`xhigh` for `reviewer_hard` and `auditor_final`). `MetaChatClient.chat(..., stream=True)` is supported. A missing key or unsafe `base_url` raises `ModelConfigError` (a `NightshiftError` and `ValueError`, **not** a `ModelError`),
  so a misconfiguration fails loudly. A malformed request (bad effort, unanswered tool calls) raises `ValueError` before anything is sent. Network failures raise `ModelError(retryable=True, status=None)` with no internal retry. `from_env` with a custom `api_key_env`
  reads only that variable. `NIGHTSHIFT_BASE_URL` / `NIGHTSHIFT_MODEL` override explicit arguments. `[models.default]` that sets `model` or `base_url` moves `reviewer_hard` and `auditor_final` off Muse Wonder.
* Usage is the server's own for non-streamed replies and, because every streamed request sends `stream_options.include_usage`, for streamed ones too. The estimate (about 4 chars per token of the visible text, flagged `usage_estimated`) remains only for
  a reply that carries no token counts, a stream cut off before its usage chunk, and a host that rejects `stream_options`. The estimate cannot see hidden reasoning tokens, so it understates spend. `budget.charged` does not record `usage_estimated`.
  The live probes are described in A.11 (26 Chat Completions probes plus 5 Responses probes, roughly 40 to 50 requests).

### A.4 Audit (B7)
* `AuditContext.events` is the whole ledger up to `window_to`; checks judge the run's state and are idempotent by finding id. There is **one finding per (invariant, subject)**: a repeat after an ack is not re-reported (the gates still apply).
* `audit.window` carries extra keys `final, new_findings, verdicts, no_evidence, check_errors`. LLM findings use subject `w<window_to>`. Auto-acks happen one window after a finding is written.
* Containment additions: I04 uses the general rule; I07 the general rule plus a correction lesson when a bad lesson exists; I11 is contained once the supervisor resumes work; I14 when the claim is now the replicated incumbent or no pending finish cites it.
* **Resolved:** `infra_error` and `unresolved` count as terminal "ended" outcomes for containment, except an experiment that is still awaiting replication (a dev `improved` whose replication failed for an infrastructure reason and has no replication grade yet is NOT contained).
* A `reviewer_unavailable` review (a block-free `reject` with the single finding id `reviewer_unavailable`) is exempt from the I12 independence checks, but its reviewer name must still be non-empty (the engine builds it from the reviewer's own identity).
* An `unresolved` lane that stopped before any review (budget stop, engine error) raises no I12 finding: it carries no result and cannot support an improved claim.

### A.5 Dashboard backend and CLI (B8a)
* `head_anchor` is the string `"<last seq>:<first 16 hex of head hash>"`; `verify --expect-head` accepts that form or a hash prefix of at least 8 hex characters. `/api/events` returns `{seq, ts, kind, actor, text}` rows with `since`, `limit`, `has_more`.
* `run.started` should include **`train_metric`** (the dashboard falls back to a heuristic otherwise). `state.run.summary` carries the run's final summary. Each experiment row also carries `train_margin` and `overfit_suspect`.
* Fault scoreboard: `planted_fault` names outside the mapping (`tamper->I03, claim_without_run->I01, overfit->I02, duplicate->engine duplicate gate, infra_flake->I07 must NOT fire, hang->watchdog, injection->I13 (LLM only)`)
  show as missed. Integration decides how `defect` (a reviewer-caught planted bug) and LLM-only faults are scored.
* `report RUN` on a ledger whose hash chain is broken still writes `report.md` and `report.html`, then prints a warning naming the broken seq to stderr and **exits 1**; the report itself starts with `LEDGER CHAIN BROKEN at seq N: this report was written from a ledger that fails verification; do not trust it` and prints the verdict as `<verdict> (unverified)` (`render_markdown(run_dir, task=None)` and `render_html` are unchanged: the chain status is computed inside the report). A live (non-replay) `serve` of such a ledger prints the same warning and still serves; replay refuses. `verify` on a ledger with no events prints `empty ledger` and exits 1.
  `doctor` without a key ends with `No API key: the offline demo works (python3 -m nightshift demo); live runs need MODEL_API_KEY, see docs/RUNNING_LIVE.md.` (exit status unchanged). The demo prints the head anchor once per run, the control run as `negative control run dir: ...`, and `the directory X already existed, using X-2` when `--out` is taken.
* The CLI reaches the orchestrator and the suite factory only through the `cli._api()` seam. `run` accepts a hidden `--crash-after-runs N` (default demo value 4, counting untrusted runs started; the baseline uses `len(task.seeds)` of them) and exits
  `os._exit(137)` on `SimulatedCrash`.

### A.6 View robustness
`build_view` isolates each event: a malformed payload is skipped and listed in `LedgerView.malformed` (seq, kind, error) instead of aborting the fold.

### A.7 The law-discovery task (B9)
Seeds `[0,1,2]`, replication seeds as in `task.toml`; run timeout 10 s, stall 3 s. The scripted manifest has a `replicate` of `quadratic` right after the early refused finish (otherwise the early finish leaves an uncontainable I09 finding), then
`poly_tuned` (dev win, fails sealed replication) and `damped` (replicates). The manifest has acknowledge steps that may find no open finding because the auditor already auto-acked: **the scripted supervisor must treat that as a no-op.**
The `duplicate` proposal's claim type is `done`; `claim_without_run` and `injection` are `no_gain`. `poly9` uses ridge 1.0 so its explosion is consistent. `template/linalg.py` is a read-only helper for agents.

### A.8 Engine, orchestrator and resume (waves 3 to 5)
* `WorkflowEngine` never raises from a lane except `SimulatedCrash` and `HaltRequested` (section 7.2 has the stop and no-op records). `BudgetExceeded` before a proposal is a block, mid-lane an `unresolved` record with the note `stopped by the budget`.
* `Orchestrator.resume` refuses a finished run, a ledger with no `run.started`, another backend or task, a different KIND of endpoint (`config.endpoint.kind`) than the run started with, a broken hash chain (unless `accept_broken_chain`) and a changed
  `task_hash` or `grader_hash` (no override). Planted behaviors are re-derived from `config.red_team` and refused if the task's bank no longer maps them to the same `scripted_id`s. The resume writes `run.resumed` (payload in section 5).
* `--context`/`resume --context`: the files are read once, their text joined for the roles, and only `{name (basename), sha256 (of the bytes), chars}` is recorded in `run.started.config.context`. The text is not in the ledger, so a resume can hand it back only if the operator
  gives the SAME files again (the `(name, sha256)` set must match, in any order; two files may share a basename and are matched individually); then `run.resumed.context_reloaded` is true. Without the flag the roles continue WITHOUT the background, the CLI warns on stderr,
  `context_reloaded` is false and the dashboard sets `state.run.context[].not_reloaded_after_resume`. The supervisor reads at most 60,000 characters of the background (it rides on every supervisor turn) and the designer at most 8,000; the CLI says how much is used. The
  background is fenced as untrusted data. Per file limit 200,000 characters, 400,000 in total; UTF-8 text only (no control characters); the scripted backend refuses it.
* Bundled tasks are looked for in `$NIGHTSHIFT_TASKS_DIR`, else `tasks/` of the repository root (the nearest ancestor of the current directory that holds `pyproject.toml` and `nightshift/`, else the package's own parent). `demo`, `run`, `resume`, `audit` and `tasks`
  fail with a one-line hint (exit 2) when none is found; `serve` and `report` do not need tasks; `--task /path/to/task` always works. Tasks are deliberately NOT part of the wheel (pip byte-compiles shipped `.py` files and `nightshift tasks` rejects a template with
  bytecode, `template_has_bytecode`), so a plain `pip install .` gives the CLI without tasks; use `pip install -e .`, a clone, or `NIGHTSHIFT_TASKS_DIR`. `demo --out` is relative to the caller's current directory.
* `nightshift tasks` prints, for each task with `requires` satisfied, a `calibration [name]: baseline per-seed std over 20 seeds: dev ..., sealed ... Recommended sigma_floor = ...` line (printed by `validate_task` itself, so it appears above the `OK` line) and
  takes about 30 to 40 seconds longer per task for the 40 extra baseline evaluations. `validate_task(task, *, calibrate=False, notes=None)` appends the advice to `notes` when a list is given. `report.render_markdown/render_html/write_reports` take an optional `task`.
* CI (`.github/workflows/ci.yml`): a 3 (Python 3.11, 3.12, 3.13) x 2 (with and without numpy) matrix runs a `compileall` step that EXCLUDES `tests/fixtures/` (a task template with `__pycache__` is invalid), the unit/integration/end-to-end suite, the offline demo, the same demo with
  `NIGHTSHIFT_ISOLATION=none` (asserting that every `experiment.run_started` is labelled `none` and the ledger finished), the clean control (zero findings), `verify`, and `nightshift tasks` (a SKIP fails the with-numpy legs, so a typo in `requires` cannot hide). A separate
  `integration` job runs the whole suite on 3.13 with numpy and `NIGHTSHIFT_INTEGRATION=1` (three real-engine tests that are otherwise skipped).

### A.9 Endpoint configuration, `base_url`, and the honesty labels
* Routing (`nightshift.toml`): `[models.default]` and `[roles.<role>]` accept `base_url, model, api_key_env, reasoning_effort, fallback_model, completion_cap_ceiling, api, allow_partner_responses`; unknown keys, roles and values are a `ModelConfigError` at load time.
  Setting `model` or `base_url` in `[models.default]` moves `reviewer_hard` and `auditor_final` off Muse Wonder (a partner host is assumed not to serve it). A role may point at another OpenAI-compatible host with `base_url` and `api_key_env`; that is supported by
  configuration and untested against real partners. An explicitly named `api_key_env` is the only key source for that role, so the Meta key is never sent to another host by fallback.
* `base_url` rules (`check_base_url`): https, or http for a loopback host only; userinfo, query and fragment are dropped; redirects are never followed. `NIGHTSHIFT_BASE_URL` and `NIGHTSHIFT_MODEL` override explicit arguments of `MetaChatClient.from_env` and the live probes; they do
  not touch a routing file.
* `run.started.config.endpoint = {kind, hosts}` is computed by `orchestrator.endpoint_info` from the routing's `base_url` values (host names only, sorted, de-duplicated; a URL with a backslash, whitespace, a control character, a scheme other than http(s), a bad port or an odd
  trailing dot yields no trustworthy host). `kind`: `scripted` (scripted backend), `loopback` (ANY host that connects to this machine, including `127.1`, `2130706433`, `0x7f.0.0.1`, `::1`, `localhost`), `meta-api` (at least one host, every endpoint parseable and every host `api.meta.ai`),
  else `custom`. The dashboard derives `state.run.origin` and `state.run.mode` from it (section 10), the report opens with a SCRIPTED BACKEND, SIMULATED MODEL or unverified paragraph accordingly, and `/api/health` reports `mode`, `origin` and `serving`.
  A `loopback` endpoint is a SIMULATION of the live path: the code path (prompts, tool calls, metering, audit) is the real one, the words are not Meta model output.
* The report separates "configured" models (`run.started.models`) from "models that answered" (counted from `budget.charged`); a role whose configured model never answered is not credited to it, and nothing is called Muse Wonder (or anything else) unless a charged call shows it.
  The ledger does NOT record which protocol (`chat` or `responses`) a role used: only `response.raw["_nightshift"]["api"]` carries it.

### A.10 The opt-in Responses client (`models/responses.py`): UNTESTED against the real service
* `MetaResponsesClient(MetaChatClient)` is selected per role with `api = "responses"` (default `chat`); `responses` with a `base_url` other than the Meta Model API is a `ModelConfigError` unless `allow_partner_responses = true`. It inherits the constructor, retry policy,
  fallback model, `disabled_params`, `completion_cap_ceiling`, secret scrubbing and the `raw["_nightshift"]` facts, and maps `input_tokens`/`output_tokens`/cached tokens onto the same `Usage`, so `MeteredClient` and the ledger are unchanged.
* Wire: `POST {base_url}/responses` with `store: false`, `include: ["reasoning.encrypted_content"]` for `muse-*` models, flat tools with `strict: false`, `text.format`, `reasoning.effort`, `max_output_tokens` (at least 16). Never sent: `previous_response_id`, `tool_choice`,
  `stream_options`. `store` is never stripped on a 400 (that would silently make the host keep every prompt).
* Replay of encrypted reasoning across tool turns: the returned assistant message carries a private `_responses_items` attribute (an inert `OpaqueItems`: count, byte length and hash only; it cannot be iterated or JSON-serialised). The supervisor loop carries it across its rebuilt
  messages, `compact_history` drops it with the turns it stubs, the patch worker does not copy it, so the client keeps a bounded memo scoped to the conversation and the turn's position. Replay is skipped for another model and when `include` was stripped; a 400 naming a reasoning item is
  retried once without replay and three such rejections switch replay off. The Chat client ignores the attribute. Encrypted items never reach a prompt, the ledger, the dashboard or a report.
* 12 assumptions are listed in `nightshift.models.responses.ASSUMPTIONS` and in `docs/META_API_NOTES.md`; `nightshift doctor --live` settles most of them (A.11). Whether replayed reasoning improves quality or lowers cost has not been measured.

### A.11 Live probes (`nightshift doctor --live`, `scripts/live_smoke.py`)
* `doctor --live` runs `models.live_smoke.run_smoke` against `NIGHTSHIFT_BASE_URL` (default `https://api.meta.ai/v1`) and `NIGHTSHIFT_MODEL` (default `muse-spark-1.3`) and writes the report only with `--live-out FILE`; `scripts/live_smoke.py` defaults to `--out demo/live_probe.json`.
  Probes use `timeout=60 s, max_retries=2` and do not read `nightshift.toml`. The report (version 2) holds status, models, latencies, token counts and `cached_known`, never the key or message content.
* 26 Chat Completions probes, 9 required (`tool_roundtrip`, `strict_schema`, `effort_minimal|low|medium|high|xhigh`, `stream` with REAL usage, `rejected_param`) and 17 informational (caching with 4k and 15k prefixes, the four strict agent schemas and the six supervisor tool schemas with keyword learning, `effort_max`,
  `effort_none_rejected`, `temperature_with_reasoning`, `completion_cap_ceiling` at 16000/32000/64000, `assistant_tool_calls_content`, `parallel_tool_calls`, `tools_with_response_format`, `system_and_developer_roles`, `ratelimit_headers`, `wonder_chat`), plus 5 Responses probes under
  `report["responses"]` that never affect `ok`. About 40 to 50 requests, most tiny; required probes use `max_completion_tokens=2000`.
* A probe is `pass`, `fail` or `inconclusive`. A reply cut off by the token cap (`finish_reason == "length"`) with nothing usable is `inconclusive`, never a failure of the feature; `ok` means no required probe FAILED, `required_inconclusive` lists the rest and `conclusive` is false when there are any.
  The `doctor --live` exit status is 1 only when a required probe failed.
* When a `muse` binary is on PATH, `doctor --live` also calls `agents.muse_code.probe_exec(timeout_sec=90.0)` (the production command line against a throwaway workspace; a few real model calls) and records the result under `muse_code` in the `--live-out` file (`{probed: false, reason}` when not probed).
  A failing Muse probe is printed and recorded but does not change the exit status (the worker is optional).

### A.12 The Muse Code worker (`agents/muse_code.py`): TESTED ONLY AGAINST A FAKE
* Command line (one place, `build_argv`): `muse exec --model M --disable-approval --disable-request-user-input --disable-reminders --no-session-log --sandbox-network restricted --max-model-steps N --json --prompt-file F [extra]`. `--model` is always passed (the routing model, because Muse Code's own default
  is `muse-spark-1.2`); `--yolo`, `--disable-sandbox`, `--model`, `--provider`, `-c model/provider/endpoint*`, `base_url`/`api_key`/`proxy` config keys and a `--sandbox-network` wider than `restricted`/`proxy-only` are refused in extra arguments (`NIGHTSHIFT_MUSE_ARGS`, shell-quoted; the constructor's
  `extra_args`). A `[muse_code] extra_args` table in `nightshift.toml` is NOT read (the routing parser ignores unknown top-level tables without an error). `--trust-workspace` is not passed (changelog 1.4.0 and 1.4.4 disagree about what it does; the later entry wins). A trust-stop message on stderr with a nonzero exit and no work done is a non-retryable infrastructure `ModelError`.
* Isolation and read scope: section 11.3. Copy-back applies only to editable and scratch globs; Python bytecode caches the agent created are dropped silently; a partial or hostile scan applies nothing and deletes nothing.
* Identity and spend: the answering model comes from the run when it reports one (`run.model.*` events or the `run.usage` line, changelog 1.4.11), and a mismatch (including any `-contributor` model) is flagged in `raw_log` and recorded in `ImplementResult.worker`; the reviewer-independence basis still uses the routing's implementer model. `run.usage`
  numbers are REPORTED BY MUSE CODE AND NOT VERIFIED; they are charged with `BudgetGate.charge_llm` (kind `llm`, actor `implementer:muse-code`, purpose `implement`) and, with no usage line, nothing is charged and `raw_log` says the run is NOT metered (only the wall timeout and `--max-model-steps` bound it). The field names of `run.usage` and
  any `run.model.*` event are undocumented and handled defensively.
* Refusals are mirrored into the real workspace as a placeholder file (`NIGHTSHIFT_REFUSED.txt` and similar markers) so the existing protected-path check ends the lane `invalid`; the I03 evidence for a Muse lane may therefore be a placeholder or marker file, and the real paths are in `worker.log` and the `raw_log` header. There is no structured violations channel in
  `ImplementResult` yet. The worker imports private helpers of `sandbox.py` (`_scan`, `_copy_files`, ...) so its file classification matches the engine's.

### A.13 Red-team lanes (`--plant`) and background (`--context`)
* `--plant` (meta backend only): the behaviors are validated against the task's scripted bank BEFORE any run directory exists (unknown, repeated, or unsupported by the bank is exit 2). `AgentSuite.planted` is a scripted suite built from the same bank; the engine routes a lane that carries a `scripted_id` to its designer and implementer and
  every other lane, and every other role, to the real roles. `run.started.config.red_team = [{behavior, scripted_id}]` records what was planted; lane labels are `RT-<behavior>` truncated to 16 characters; a `supervisor.nudge` ("Operator notice") says they are the operator's; results the supervisor sees carry the note
  `red-team lane injected by the operator; scripted`. Attackers are never replicable, promotable or usable as a base, so they cannot decide the verdict, but their findings, the real reviewer's reaction and the auditor's detections are real. `state.run.red_team` and the report's RED-TEAM paragraph label them. The fault scoreboard
  scores them with the same `planted_fault` mapping as a scripted run; an attacker lane that the duplicate gate (or the budget) stopped before it ran is shown as **not run** and kept out of recall, so the board never reports a recall for faults that were never tried. A meta run with `--plant` has REAL model roles plus clearly labelled scripted attackers.
  Against the default simulator rounds `tamper`, `injection`, `hang`, `infra_flake` and `duplicate` run, while `overfit` and `claim_without_run` do not (the simulated designer proposes the same hypotheses first, so the gate blocks the scripted lanes); a different brain or task can change that. In a **simulated** `--plant` run the report and dashboard add that the model roles are the local simulator.
* Background material is described in A.8.

### A.14 Test strategy additions
* `tests/support/fake_meta_server.py` (fake Chat Completions and Responses server), `tests/support/fake_muse.py` (a fake `muse` executable), `tests/support/sim_muse.py` (the simulated model brain `SimBrain` that drives the live code path over loopback). Runs that use them are simulations.
* Sealed canary (`tests/support/canary.py`, `tests/test_canary.py`, used by `tests/test_e2e_meta_sim.py` and `tests/test_demo_e2e.py::ScriptedAgentCanary`): every sealed number is expanded into roundings and truncations at 2 to 6 decimals and 3 to 6 significant digits, scientific notation (including the 6-decimal `%e` default), percentage and
  per-mille forms, forms without a leading zero and labelled forms (`s_eff`, `t`, `n_eval`); forms that a dev number of the same run also produces are subtracted; a needle needs at least 3 significant digits and is never a bare integer; matching respects number boundaries (a long needle may match as a prefix); budget and clock counters (and numbers written with a time or money unit)
  are exempt. Limits, stated plainly: a sealed value written in words, with split digits, with thousands separators or in fewer than 3 significant digits is not detected, nor is one written under an accounting key (`usd`, `wall_sec`, `cost`, `amount`, `elapsed`, ...). The canary covers numeric renderings only; the opt-in `NIGHTSHIFT_INTEGRATION` test covers the real engine.
* Concurrency assertions count in-flight judge calls (`peak == window size`, all calls overlapping at one instant) instead of comparing wall-clock ratios; the stubborn-supervisor simulation has its own `STUBBORN_LIMITS` (4 USD) so its spend never sits at the default cap.

### A.15 Statistics, tasks and reports as built
* Section 8 holds the tables, the portability rule and the calibration paragraph. The shipped constants: law-discovery `noise_margin` 0.035, `sigma_floor` 0.036; spiral-classifier `noise_margin` 0.02, `sigma_floor` 0.02; both 3 dev seeds `[0,1,2]` and 5 replication seeds `[100..104]`; `max_replications` 6. `docs/NULL_BENCHMARK.md` is generated by
  `scripts/null_benchmark.py --campaigns 20000 --seed 7 --write docs/NULL_BENCHMARK.md` (about 20 minutes) and a test fails if the document and the script disagree; it states what is a model and what an audit measured.
* The report's headline is `Sealed delta vs baseline <id>` (with the threshold an improvement had to exceed), recomputed with `gates.sealed_decision` after a promotion chain. For that recomputation it needs the task's noise parameters: from the `task` argument, else from `run.started` when it records `noise_margin` and `sigma_floor` (the orchestrator does NOT write them today), else from
  `<repo>/tasks/<name>` when its task hash matches the ledger's, else a fallback (the baseline record's `noise_margin`, no floor) that the report flags as possibly understated. The delta itself is exact either way.
* `make_data.py --seed TEXT` (law-discovery) draws a fresh split; the default reproduces the committed files byte for byte, and without `--out DIR` it overwrites the task's committed data.

### A.16 Known limitations (collected, so nothing above is read as a stronger claim)
* Nothing was run against `api.meta.ai`, the real `muse` binary or the Responses endpoint; no Meta API key was available. The 16000 completion-cap ceiling is a conservative default, not a documented number. The Chat Completions in-stream error chunk shape is undocumented (three plausible shapes are parsed).
* A persistent content-hash change made by one lane is reported as a trusted violation by every lane that sees it afterwards (the finding cannot be attributed to the guilty lane). `task_hash`/`grader_hash` ignore a planted top-level `*.pyc`; the stat snapshot and the stray-bytecode check catch it.
* Mode `none` cannot hide the orchestrator's environment from a root run, nor stop a read of `trusted/`; macOS gets `none`. `userns` is the only mode that keeps the labels out of reach.
* `prompts.retry_cap` honours a configured `completion_cap_ceiling` in `chat_json` (the strict-schema roles) but the patch worker's own retry does not pass it, so a ceiling above 16000 reaches the patch worker only through the client's own clamp.
* `budget.charged` does not record `usage_estimated` or the protocol used; the Muse Code charge is not labelled unverified in the ledger (only in `worker.log`); the experiment-level `isolation` label is computed when the event is written.
* The real-pipeline false-promotion rate under the calibrated floor was not re-measured through the sandbox; `docs/NULL_BENCHMARK.md` section 6 is an in-process estimate with the real grader.
