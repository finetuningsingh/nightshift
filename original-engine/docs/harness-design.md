---
type: design
title: Harness design for the Glimmer three-arm experiment
description: The module layout, interfaces, on-disk and host layout, trial sequence, budget and statistics design, real-versus-unverified split, the six disjoint implementer assignments, and the resolution of every critic finding for the harness specified in harness-spec.md.
timestamp: 2026-11-01T18:00:00-05:00
---

# Harness design

This turns the [harness specification](harness-spec.md) into modules, files and work packages. Scope and goals are in the [plan](goal-and-critical-path.md). Nothing below has been run on a GPU, against CORAL with Muse, or with a Meta API key: there is no Meta key in this environment. The shared contracts already exist and are tested: `nightshift/lab/contracts.py` and `tests/test_lab_contracts.py` (83 tests, run with `PYTHONDONTWRITEBYTECODE=1 python3.13 -m unittest tests.test_lab_contracts`). This revision applies two independent critiques; each blocker and major finding is fixed or rejected with a reason in [Critic findings and resolutions](#critic-findings-and-resolutions).

Labels used throughout. **NOTE** means a fact taken from a reader's notes, which were verified by that reader as the notes say. **UNVERIFIED** means no one has checked it. **DECISION** means this design chooses it and the reason is given. The six notes are the sources for NOTE claims: Nightshift reuse, CORAL, IFBench, Glimmer training, GPU lifecycle and outer-loop reuse (in the session scratchpad `notes/`, which is temporary; the first thing each implementer does is copy the scripts named in [Work breakdown](#work-breakdown) into the repository).

## Contents

1. [Decisions at a glance](#decisions-at-a-glance)
2. [Contracts](#contracts)
3. [Host and directory layout](#host-and-directory-layout)
4. [Module layout and per-module design](#module-layout-and-per-module-design)
5. [One trial, end to end](#one-trial-end-to-end)
6. [Budget enforcement](#budget-enforcement)
7. [Ledger decision](#ledger-decision)
8. [Statistics](#statistics)
9. [Real and tested offline, versus GPU or Meta key only](#real-and-tested-offline-versus-gpu-or-meta-key-only)
10. [CORAL integration plan](#coral-integration-plan)
11. [IFBench scorer plan](#ifbench-scorer-plan)
12. [Glimmer trainer and serving plan](#glimmer-trainer-and-serving-plan)
13. [GPU executor and watchdog](#gpu-executor-and-watchdog)
14. [Outer-loop controller and redaction rules](#outer-loop-controller-and-redaction-rules)
15. [Interface appendix](#interface-appendix)
16. [File formats](#file-formats)
17. [Ids, clocks, hashes and imports](#ids-clocks-hashes-and-imports)
18. [Closing the leaks](#closing-the-leaks)
19. [Work breakdown](#work-breakdown)
20. [Open decisions](#open-decisions)
21. [Critic findings and resolutions](#critic-findings-and-resolutions)

## Decisions at a glance

| # | DECISION | Reason (source) |
|---|---|---|
| D1 | Reuse `nightshift.ledger.Ledger` unchanged: one master lab ledger plus one small ledger per trial, lab-owned kinds `lab.*`. No second ledger implementation. The GPU rental manager keeps its own flock JSON file because several processes (watchdogs, reaper) must write it. | Any string kind is accepted; chain, crash recovery and `nightshift verify` already work (reuse notes section 4). |
| D2 | All tooling runs on one always-on Linux **control host** (proposed: ax43). The Mac only sends commands. Researchers run on a **researcher host** (ax41); each trial runs as its own throwaway unix user (fallback without root: trials run one at a time in one user and the report labels the isolation level). The researcher host holds no Thunder or Meta secret: the `~/.thunder` token that the GPU notes found on ax41 is removed or made unreadable to the lab users, and `verify-isolation` tests that. | Mac sleep kills watchdogs; CORAL gives no OS boundary by default; one shared unix user lets trials read each other's token and notes (CORAL notes section 5; GPU notes R5, section 4.5; critic B11). |
| D3 | Selection and test data, ground truth and verifier code run only on the control host. A rented GPU receives prompt text only and returns responses; verification happens on the control host. Dev scores generated on the training rental are reached through an ssh-forwarded control-host loopback port. | Verifiers are CPU-only; this keeps ground truth off every rented machine (GPU notes R13; IFBench notes section 6). |
| D4 | The Meta key lives only on the control host. A metering proxy there forwards the researchers' LLM calls, counts tokens and dollars, and refuses past the cap. The proxy is reached from the researcher host through an SSH reverse tunnel, so the control host opens no inbound port. | CORAL has no budget and the key is readable by researchers otherwise (CORAL notes sections 5.3, 7; reuse notes 2.1 point 4). |
| D5 | The keep rule is **screen, then confirm**: each round screens up to five candidates by delta with no threshold, then confirms only the best one against a one-sided t margin with fresh, interleaved incumbent trials and a pooled, frozen-sigma-anchored spread. Per-prompt paired methods are diagnostics only. | Prompt resampling measures scorer noise only (spec; reuse notes 2.2); a sequential per-candidate test has a family false-accept rate of 0.13 to 0.25 (critic B3). See [Statistics](#statistics). |
| D6 | GPU minutes are charged to the trial budget by a one-minute tick while a rental is live; a budget crossing deletes the rental from the executor. The rental's armed deadline and a detached watchdog are the backstop. | The cap is the budget; the hard stop sits outside the researcher (spec; GPU notes section 4.1). |
| D7 | Dev data is public to researchers. Dev scores come from the same scorer code path, run on the training rental right after a job. | One code path for all arms (spec); avoids a second 60 GB model load per dev score. |
| D8 | IFEvalG scores dev and selection; unchanged upstream IFBench scores the test split only. | The 29 IFBench-Train constraint ids exist only in IFEvalG (IFBench notes section 3). |
| D9 | CORAL provider config, gateway, grader and tools are runtime-owned and applied at launch. The pack holds only text levers. | The outer loop must not be able to edit the metering route, the grader or the budget (spec access rules). |
| D10 | Do not vendor third-party verifier code into the repository. Pin and fetch it at setup, verify hashes at run time. Test fixtures are synthetic rows written for the test, never rows of `IFBench_test.jsonl` or `sample_output.jsonl`. | Plan: reference reused code as labelled, pinned dependencies; the repository is pushed to GitHub every session, so a copied test row would publish the test set (critic A17). |
| D11 | The CLI is `cli.main(argv, *, world=None)`. A `LabWorld` (contracts.py) is the dependency container; production builds it in `cli.build_world`, tests build one of fakes in `tests/support/lab_world.py`. There is no `--fake` flag and no fake in production code. | Fakes live in `tests/`, which is not packaged and which the production CLI must not import (critic A4). |
| D12 | The split salt is a random 256-bit secret under `private/`; only its hash is recorded. Selection and held-out items are built as fresh constraint instantiations, not copied from the public IF-RLVR rows. Every training dataset a job produces is scanned on the control host against selection, held-out and test prompts before any selection score. | A location-only boundary fails when a researcher finds the public dataset and a deterministic split (critic B1). See [Closing the leaks](#closing-the-leaks). |
| D13 | The final phase is gated by ledger events, not by a caller-supplied phase: `lab.search_closed`, `lab.final_analysis_frozen`, a `FinalRequest`, and a planned trial id. Per-trial held-out and test scores stay hidden until `lab.final_revealed`. | A per-pack-commit guard that anything can satisfy by committing a whitespace change is not a guard (critic B2, A1). |

## Contracts

`nightshift/lab/contracts.py` is the only module every implementer may import without depending on a sibling. It is frozen: changes go through the lead, are additive, and come with a test change. It is standard library only and imports nothing else from `nightshift`. `SCHEMA_VERSION` stays 1 because nothing has been persisted yet.

**Rules every record follows.**

- Frozen dataclasses with explicit `to_dict` / `from_dict`. A record that is persisted as its own file or ledger payload has a `schema_version` field. Small value objects embedded in a record (`PackRef`, `FileRef`, `SeedAggregate`, ...) are versioned by their container.
- `from_dict` rejects unknown fields, missing fields (an optional field must be present as `null`), a wrong `schema_version`, wrong types, non-finite numbers and negative budgets. Construction validates the same way, so a bad record cannot be built in memory either.
- Free-text fields are limited to 4000 bytes. Reason: `Ledger.append` silently truncates any string over 8192 bytes and writes `NaN` as a bare token (reuse notes 1.1). `assert_ledger_safe` checks a payload before append. Per-prompt results, traces, diffs and prompts are files, referenced by `FileRef` (path, sha256, size).
- Identifiers match `[A-Za-z0-9][A-Za-z0-9_.-]{0,63}` because they become directory names. Persisted paths are lab-root-relative (`PackRef.repo_path`, `AdapterRef.path`, `FileRef.path`): a record stays valid when the lab root moves, and no control-host absolute path reaches a ledger or a Wonder prompt.
- Every enum is a `_StrEnum`: `str(x)` and `f"{x}"` are the value on Python 3.11, 3.12 and later (3.12 changed mixed-in enum formatting).
- Records with a `dict` field are unhashable (`hash(TrialSpec)` raises `TypeError`): key collections by id, never put records in a set. Exchange types that are never persisted (`Quote`, `GpuInstance`, `CreateReceipt`, `ServeSpec`, `ServeEndpoint`, `GenPrompt`, `GenResult`, `TrialContext`, `SubmittedAttempt`, `RuntimeResult`, `ProposalContext`, `ProposalResult`, `DecisionNumbers`, `LintResult`, `LabWorld`) have no `to_dict`; `canonical_json` raises `TypeError` on them. `TrialContext.llm_token` and `control_endpoint` are excluded from `repr`.
- Helpers every module uses: `write_record` (atomic, returns the sha256 of the bytes), `loads_strict` (refuses `NaN`, `Infinity`, duplicate keys), `file_ref` and `read_regular_bytes` (refuse symbolic links, hard links and non-regular files; the second reads without following links and with a size cap), `utc_iso(epoch)` (the only way to make a record time), `load_frozen` (see [Ledger decision](#ledger-decision)).

**Inventory.**

| Record | Persisted as | Ledger kind | Writer |
|---|---|---|---|
| `LabConfig` | `lab.json` | none | operator; only `cli` reads it and hands each component its slice |
| `LabInit` | event only | `lab.initialized` | cli |
| `BudgetSpec` | `frozen/budget.json` | `lab.budget_frozen` (as `FrozenFile`) | cli |
| `AssignmentManifest` | `frozen/assignments.json` | `lab.assignments_frozen` | cli |
| `SpreadPlan` | event only | `lab.spread_planned` | cli, before the first spread trial |
| `SpreadMeasured` (+ `SpreadEstimate`) | event only | `lab.spread_measured` | cli |
| `Rule` | `frozen/rule.json` | `lab.rule_frozen` | cli |
| `GateRecord` | event only | `lab.gate_recorded` | cli |
| `OperatorChange` | event only | `lab.operator_change` | cli, for any operator change after the first candidate |
| `PackCommitted` | event only | `lab.pack_committed` | packs / outer loop, through `LabLedger` |
| `TrialSpec` | `trials/<id>/spec.json` | `lab.trial_started` | trial runner |
| `TrialRecord` | `trials/<id>/trial.json` | `lab.trial_finished` | trial runner, or `cli abort-trial` |
| `ScoreRequest` | `scores/<id>/request.json` | not appended | `Scorer.score` writes it |
| `ScoreRecord` | `scores/<id>/score.json` | `lab.score_recorded` | `Scorer.score` writes the file and returns the record; the caller (trial runner, `cli final`) appends the event |
| `ScoreCost` | event only | `lab.score_cost` | the caller of `score_session` |
| `ScoreFailure` | event only | `lab.score_failed` | the caller of `Scorer.score` when it returns no record |
| `ContaminationReport` | event only | `lab.contamination_scanned` | trial runner |
| `Proposal` | `candidates/<id>/proposal.json` | not appended (referenced by `CandidateRecord`) | outer loop |
| `CandidateRecord` | event only | `lab.candidate_proposed` | outer loop |
| `Decision` | event only | `lab.candidate_decided` | outer loop |
| `SearchClosed` | event only | `lab.search_closed` | cli `close-search` |
| `FinalAnalysis` | `frozen/final_analysis.json` | `lab.final_analysis_frozen` | cli `freeze-final-analysis` |
| `FinalRequest` | event only | `lab.final_requested` | cli `final` |
| `FinalReveal` | event only | `lab.final_revealed` | cli `reveal` |
| `FinalComparison` | `reports/final_comparison.json` | not appended | cli `reveal`, report |
| `AdapterRef` | `artifacts/adapters/<id>/adapter.json` | embedded in `TrialRecord`, `TrainResult` | trainer |
| `BudgetCharge`, `BudgetExhaustedEvent` | trial ledger | `lab.budget_charged`, `lab.budget_exhausted`, `lab.budget_reserved` (a `BudgetCharge` holding one call's worst-case tokens, with `detail.usd` its worst-case dollars) | budget meter |
| `TrainJob`, `TrainResult` | trial ledger | `lab.gpu_job_started`, `lab.gpu_job_finished` | trial runner (the one writer of its trial ledger entries for jobs) |

`LabKind` lists the strings; `PAYLOAD_TYPES` maps each to its class; `payload_from_dict` parses a payload; `LabKind.FREEZE_ONCE` lists the kinds that may appear once (a second one is malformed and fails `verify`). The Wonder calls use the reused `BudgetGate` and write `budget.charged` / `budget.exhausted` into `outer/ledger.jsonl` (`LabKind.REUSED`). The kind `experiment.proposed` is never used: `Ledger._load` allocates experiment ids from it.

**Enums.** `Arm` (stock, v0, evolved; arm 1 has no trial and no pack), `Split` (dev, selection, test), `Phase` (search, final), `AssignmentRole`, `Verifier`, `Resource` (with `AXIS_OF`, the only place the resource-to-axis map is written), `TrialStatus` (adds `contaminated`), `ExitReason` (adds `stopped_contaminated`), `JobStatus`, `Outcome`, `DecisionStage` (screen, confirm), `FailedTrialPolicy` (invalidates_candidate, scored_zero), `ComparisonMethod` (blocked_t, welch), `MixedRowPolicy` (dropped only), `ItemMode` (fresh_instantiation, salted_draw), `Lever`, `EditOp`, `Gate`, `GateOutcome`. `Confidence` and `IncumbentPolicy` were removed: the rule carries a numeric `alpha`, and confirmation always uses fresh incumbent trials.

**Constants** (use them, never literals): `TEST_ASSIGNMENT_ID = "ifbench_test"`; the six `METRIC_*` names and `METRICS_BY_SPLIT`; `TIMING_KEYS`; `DATA_REVISION_KEYS` and `SOURCE_REVISION_KEYS`; `CONFIG_PIN_KEYS`, `CONFIG_SECRET_NAMES`; `MIN_POOLED_DF = 8`, `MAX_SCREEN_BATCH = 5`, `MIN_FINAL_TRIALS_PER_PACK = 5`, `CONTAMINATION_NGRAM = 13`; `ADAPTER_REL_ROOT`.

**Invariants the records enforce** (each is tested):

- A final-phase trial has arm v0 or evolved, runs on a held-out assignment and carries `final_request_id`; a search-phase trial has no arm, runs on an evolution assignment and has no `final_request_id` (`check_trial_against_frozen`).
- A `ScoreRequest` / `ScoreRecord` has `final_request_id` if and only if its phase is final. The test split can only be requested in the final phase, and the test split, and only the test split, uses `TEST_ASSIGNMENT_ID`. `ScoreRecord.score_id == request_id`. A record's metric names belong to its split (`METRICS_BY_SPLIT`).
- A test score is verified with upstream IFBench; dev and selection scores with IFEvalG.
- `FinalRequest.arm` is stock if and only if `pack` and `pack_version` are null; dev is never a final split; a repeat needs a reason.
- `FinalAnalysis`: weights sum to 1; both arms plan the same number of trials in every held-out assignment; at least 5 trials per pack; blocked t with two or more held-out assignments, Welch with one; every planned trial id, arm, assignment and seed is fixed.
- `Rule`: evaluated on the selection split; `alpha` in (0, 0.5]; at most 5 candidates per screening round; at least 2 confirmation trials per side; pooled degrees of freedom (`spread.df + (n_candidate - 1) + (n_incumbent - 1)`) at least 8; the minimum detectable gain does not exceed the assumed gain; sigma 0 with `min_gain` 0 needs a recorded override reason. `SpreadEstimate.df == len(trial_ids) - n_assignments`. `SpreadMeasured` accounts for every planned trial (completed plus failed).
- A decision's outcome agrees with its numbers: a confirmation is accepted if and only if `delta > margin`; a screen can only reject; an invalid candidate carries no numbers. `candidate_version == candidate_id == CandidateRecord.version_label`.
- `AdapterRef.path` is `artifacts/adapters/<adapter_id>`; its sha256 equals the hash of the canonical file listing; its size equals the sum of the file sizes. `TrialRecord.submission.adapter_id == adapter.adapter_id`, and `TrainResult.adapter_id == adapter.adapter_id`.
- A scored trial has a submission, an adapter, a selection score id and an `initial_context_sha256`; an unscored trial has a failure text; `initial_context_sha256` is null only when the workspace was never checked out.
- `TrainJob.dev_request` is a template: dev split, `arm` null, `adapter` null, the trial's phase and `final_request_id`, score seeds and decoding. `TrainResult.timings` keys are a subset of `TIMING_KEYS`; an OK result has `train_data`.
- `ContaminationReport.passed` is true exactly when there are no exact and no 13-gram hits.

**Cross-record checks** (functions, tested): `comparability_key(spec, metric)`, `validate_rule_against_manifest` (rule assignments are evolution assignments), `validate_analysis_against_manifest` (final assignments are held-out), `check_trial_against_frozen(spec, manifest=, budget=, rule=, analysis=)` (roles, data revisions, exact frozen budget, scorer version, score seeds, decoding, planned arm/assignment/seed), `load_frozen`.

**Protocols** (all `runtime_checkable`): `LedgerWriter`, `BudgetMeter`, `GpuProvider`, `ModelServer` / `ServeSession`, `Generator`, `Scorer`, `Trainer`, `ResearcherRuntime` / `RuntimeHandle`, `ProposalSource`, `TrialExecutor`, `PackStore`, `DecisionRule`, `StateReader`; callable aliases `Clock`, `PortAllocator`, `ScoreGuard`, `GeneratorFactory`, `MeterFactory`, `TrialLedgerFactory`; and the `LabWorld` container. Each protocol is the seam between two implementers, so each side can be built and tested against a fake of the other. The exact constructors that bind them are in [Module layout](#module-layout-and-per-module-design).

**Exceptions**: `LabError` base; `ContractError`, `SchemaVersionError`, `BudgetExhausted`, `AccessDenied`, `FinalComparisonRefused`, `GateNotRecorded`, `LedgerError`, `GpuError` (with `ProviderUnavailable`, `CreateRejected`, `CreateAmbiguous`, `OwnershipError`), `ScorerError`, `TrainerError`, `RuntimeFailure`, `ProposalInvalid`. A module may define a local subclass (for example `RentalRefused(GpuError)`).

**Layout.** `LabLayout(root)` is pure path arithmetic for the control host (next section). Every implementer uses it instead of building paths by hand; it now has every path this document names, including `lab_config_path`, `final_analysis_path`, `anchors_path`, `third_party_dir`, `split_salt_path`, the trial subdirectories, the score and candidate files, `gpu_session_dir` and `adapter_json_path`. The control API listens on a loopback TCP port; there is no trial socket path.

## Host and directory layout

```text
Mac (operator, Claude coordinator)         never holds a watchdog, key or private data
  |  ssh (commands only)
  v
CONTROL HOST  (proposed ax43, always-on Linux, unix user `lab`, home mode 0700)
  coordinator processes: trial runner, scorer, outer loop, report, cli
  metering proxy + control API (one loopback listener per trial)
  GPU rental manager + detached watchdogs, rental ledger, Thunder token, Meta key
  lab root  <lab>/  (below), including private/ (selection, test, ground truth)
  |  ssh -R: loopback port forwarded to the researcher host; ssh to rented GPUs
  v
RESEARCHER HOST  (ax41, one throwaway unix user per trial: no ssh key to the control host, no Thunder or Meta secret)
  CORAL manager + LiteLLM gateway + opencode + worktrees
  holds only: pack checkout, dev and train data, a per-trial token, the researcher tool
  |
RENTED GPU (Thunder A100 80GB or H100)  no Thunder token, no Meta key, no ground truth
  training jobs (dev and train data, recipe) and model serving (prompt text in, responses out)
```

**Control-host lab root** (`LabLayout`):

```text
<lab>/
  lab.json                                 LabConfig: hosts, pins, names of secret environment variables
  ledger.jsonl  ledger.jsonl.lock          master ledger (one process: the coordinator; one thread-safe LabLedger object)
  anchors/master.anchors                   head anchors, committed and pushed to git (see Ledger decision)
  frozen/        rule.json  budget.json  assignments.json  final_analysis.json
  packs/pack.git                           the pack repository (bare); versions are commits, candidates are tags
  outer/ledger.jsonl                       Wonder calls via the reused BudgetGate (budget.charged events)
  gpu/rentals.json  gpu/sessions/<s>/      rental manager state, ssh config, keys (0600)
  data/assignments/<aid>/{train,dev}.jsonl public to researchers; copied to the researcher host per trial
  private/assignments/<aid>/selection.jsonl   control host only, directory mode 0700
  private/test/IFBench_test.jsonl             control host only
  private/split_salt                          the secret split salt (random 256 bits); only its hash is recorded
  third_party/IFBench  third_party/open-instruct   pinned checkouts (not vendored)
  trials/<tid>/  spec.json trial.json ledger.jsonl pack/ run/ trace/ recipe/ train_data/<job>.jsonl
  scores/<sid>/  request.json score.json per_prompt.jsonl gen_seed<k>.jsonl      (responses are private)
  candidates/<cid>/ proposal.json prompt.txt response.txt diff.patch lint.json
  artifacts/adapters/<aid>/  adapter_config.json adapter_model.safetensors adapter.json
  reports/
```

**Researcher host, per trial** (`/srv/lab/trials/<tid>/`, owned by that trial's throwaway user): `pack/` (clean checkout), `data/{train,dev}.jsonl`, `xdg/` (isolated opencode config, data, cache, state), `run/` (CORAL run directory), `tools/`, `.glimmer_token`.

**Why this keeps selection and test out of reach.** The researcher host has no selection, test or ground-truth file for those splits, no path to the control host except a forwarded loopback port that answers only token-scoped calls, and no secret. CORAL's `.coral/private/` is not a boundary: in the probe every path to a planted canary worked, and the environment of `coral start` and the grader source were readable and writable by the agent (CORAL notes section 5). Nothing here depends on it. The per-trial unix user is defence in depth, not the boundary. Acceptance check: `python -m nightshift.lab verify-isolation`, run as a trial user on the researcher host, tries `ssh` to the control host, reads of the lab root, reads of `~/.thunder` and of another trial's token and notes, leftover processes of the previous trial, a symlink planted under a harvest path that points at a canary under `private/` (it must never reach a prompt, a ledger or a scoring rental), a comparison of workspace and caches against the initial-context listing, and a search of the process environment for planted canary values, and must find none. **UNVERIFIED** until run: whether trial users can be created (the ax41 user is in the sudo group; passwordless sudo is not checked), and whether ax41 can reach the control host over SSH at all.

## Module layout and per-module design

One file per responsibility, flat under `nightshift/lab/`, names grouped by prefix. The owner number refers to [Work breakdown](#work-breakdown). Imports are absolute (`from nightshift.lab.contracts import ...`) and lab code is run with `python -m`, never as `python nightshift/lab/x.py`: `nightshift/lab/ledger.py` and `budget.py` share a basename with `nightshift/ledger.py` and `budget.py`.

| File | Responsibility | Owner |
|---|---|---|
| `__init__.py`, `contracts.py` | shared contracts | lead (done) |
| `hostkit/recipes/reference_rft/` | the reference method (generate, filter, SFT, reload) that seeds v0 | 2 |
| `ledger.py` | write side of the lab ledgers | 6 |
| `fold.py` | read model: ledger events to `LabState`, query helpers | 6 |
| `budget.py` | per-trial hard-stop meter | 6 |
| `cli.py`, `__main__.py` | commands and the composition root | 6 |
| `verifiers.py` | upstream IFBench and IFEvalG adapters, per-prompt flags | 1 |
| `ifdata.py` | JSONL loading, salted deterministic splits, fresh selection and held-out items, assignment manifest | 1 |
| `contamination.py` | scan of a job's training data against selection, held-out and test prompts | 1 |
| `rewards.py` | training-time reward for researcher recipes | 1 |
| `generator.py` | `Generator` over an OpenAI-compatible endpoint | 1 |
| `scorer.py` | `Scorer`: access policy, seeds, aggregation, files | 1 |
| `score_session.py` | serve, score, tear down, record cost | 1 |
| `gpu_rentals.py` | rental manager: money rule, state machine, ownership | 2 |
| `gpu_watchdog.py` | detached watchdog and reaper entry points | 2 |
| `gpu_thunder.py` | `GpuProvider` for Thunder (REST) | 2 |
| `gpu_remote.py` | ssh/rsync helpers with sha256 verification | 2 |
| `model_server.py` | `ModelServer` on a rented GPU | 2 |
| `trainer.py` | `Trainer`: one job per lease, dev score, pull, delete | 2 |
| `hostkit/` | scripts and configs that run on the rented GPU | 2 |
| `stats.py` | spread, rule decision, final comparison, diagnostics | 3 |
| `report.py`, `chart.py` | tables, search-trace chart | 3 |
| `packs.py` | pack git operations, strict apply, lint, hashes | 4 |
| `outer_context.py` | redaction: facts to the Wonder prompt | 4 |
| `proposer.py` | `ProposalSource` over Wonder | 4 |
| `outer_loop.py` | the controller | 4 |
| `assets/pack_v0/`, `assets/lint_pack.py` | v0 pack template and the lint script | 4 |
| `runtime_coral.py` | `ResearcherRuntime` over CORAL | 5 |
| `metering_proxy.py` | LLM forwarding proxy with the hard stop | 5 |
| `control_api.py` | train and status endpoints for the researcher tool | 5 |
| `researcher_tool.py` | the `glimmer` command on the researcher host | 5 |
| `trial.py` | `TrialExecutor` | 5 |
| `assets/coral_grader/` (with `ENTRYPOINT.json`), `assets/shims/opencode` | dev grader package; PWD shim | 5 |

### 1. Ledger side (owner 6)

**Concurrency model** (DECISION). The coordinator is ONE process. `TrialRunner.run` executes in a thread of that process, and trials may run concurrently. Every master append goes through one shared, thread-safe `LabLedger`; each trial ledger is created and closed inside `TrialRunner.run` and is also a thread-safe `LabLedger`, because the proxy, the control API, the GPU tick thread and the trainer thread all charge through the trial's meter at the same time. Nightshift's `Ledger` raises `LedgerLocked` for a second writer process, so nothing outside the coordinator process ever opens a lab ledger for writing. `ledger.py` carries a test in which many threads append concurrently and `verify()` still passes.

**`ledger.py`**

- API: `LabLedger.create(dir: Path, *, clock: Clock, fsync: bool = True) -> LabLedger`; `LabLedger.open(dir: Path, *, clock: Clock) -> LabLedger`; `LabLedger.read_only(dir: Path) -> LabLedger`. It implements `LedgerWriter`: `append(kind: str, actor: str, record) -> Event`, `anchor() -> LedgerAnchor`, `seal()`, `close()`; plus `events(since_seq: int = 0) -> list[Event]` and `verify() -> tuple[bool, Optional[int]]`. A module function `register_secrets(*env_names: str)` wraps `secrets.register_env`.
- `append` accepts only kinds in `LabKind.all()` and a payload of the class in `PAYLOAD_TYPES[kind]`; it calls `assert_ledger_safe(record.to_dict())` and then `Ledger.append`. Actors: `coordinator`, `scorer`, `outer`, `trial:<id>`. A `FREEZE_ONCE` kind appended a second time raises `LedgerError`.
- Failure modes: a second writer process raises `LedgerLocked` (convert to `LedgerError`); an unknown kind, wrong payload class or over-long string raises before any write; `LedgerCorrupt` propagates; a torn last line is recovered by the reused class. Credentials must be registered before the first append (reuse notes 1.1).

**`fold.py`**

- API: `fold_events(events: Iterable[Event]) -> LabState`; `fold_ledger(dir: Path) -> LabState`; `FoldReader(layout: LabLayout)` implementing `StateReader`: `state()`, `incumbent_version(state)`, `selection_trial_scores(state, rule, pack_version) -> list[TrialScore]`, `cost_breakdown(state, trial_ids, candidate_id) -> CostBreakdown`, `next_id(state, prefix) -> str`, `has_gate(state, gate)`, `final_requests(state, pack)`, `load_frozen_all(state) -> (Rule, AssignmentManifest, BudgetSpec)` which re-hashes each file against its `FrozenFile` and raises `ContractError` on a mismatch.
- `selection_trial_scores` takes only search-phase trials of `pack_version` on `rule.assignments`; reads `rule.metric` with `ScoreRecord.metric_by_seed(rule.metric)`; skips (and lists as malformed) a score whose metric, scorer version, decoding, seeds or verifier differ from the rule's; counts a `BUDGET_EXHAUSTED` trial with a scored adapter normally; treats every other non-scored trial by `rule.failed_trial_policy` (`INVALIDATES_CANDIDATE` makes the whole set unusable, `SCORED_ZERO` adds a `TrialScore` of 0.0 with `score_id` None). Each `TrialScore.comparability_sha256` is `comparability_key(spec, rule.metric)`.
- Malformed, never fatal, listed with the sequence number: a non-lab or unknown kind, a bad payload, a duplicate trial or score id, a second event of a `FREEZE_ONCE` kind (**the first wins**), a version label that maps to two commits, a duplicate `(phase, pack.commit, assignment_id, seed)` trial, a second selection score for the same `(adapter_sha256, assignment_id, phase)` (the first completed record is the one used), a score whose trial has no `lab.trial_started`, a decision before the rule or without its candidate.

**`budget.py`**

- API: `TrialBudgetMeter(spec: BudgetSpec, ledger: LedgerWriter, trial_id: str, *, clock: Clock, lab_cap: Optional[LabCap] = None)` implements `BudgetMeter`; `LabCap(cap_usd: float, spent_usd: float, *, clock)` is one shared, thread-safe object for the whole coordinator (see [Budget enforcement](#budget-enforcement)); the meter factory is `lambda spec, ledger: TrialBudgetMeter(spec, ledger, spec.trial_id, clock=clock, lab_cap=lab_cap)`.
- Behaviour: see [Budget enforcement](#budget-enforcement). Idempotent per `call_id`; rebuilds state from the trial ledger on reopen; thread-safe.
- Failure modes: if a charge cannot be recorded, the meter treats the axis as exhausted and raises (fail closed).

**`cli.py`** (`python -m nightshift.lab <lab_root> <command>`; `cli.main(argv: Sequence[str], *, world: Optional[LabWorld] = None) -> int`)

- Commands: `init`, `freeze-manifest`, `freeze-budget`, `record-gate`, `commit-pack`, `plan-spread`, `trial`, `spread`, `freeze-rule`, `outer`, `close-search`, `freeze-final-analysis`, `final`, `reveal`, `score`, `abort-trial`, `operator-change`, `report`, `status`, `verify`, `verify-isolation`, `rentals`, `anchor`.
- The composition root: `cli.build_world(layout: LabLayout, config: LabConfig) -> LabWorld` reads `lab.json` and builds the real objects. It calls these factories by name: `HttpGenerator` (through a `GeneratorFactory`), `ScoreService`, `GlimmerTrainer`, `RentalModelServer`, `CoralRuntime`, `TrialRunner`, `PackRepo`, `OuterController`, `WonderProposalSource`, `LabLedger`, `FoldReader`, `TrialBudgetMeter`, `stats` (as a `DecisionRule`). When `world` is given, `main` uses it and builds nothing; tests pass a world of fakes.
- Order rules it enforces: `plan-spread` precedes the first spread trial; `freeze-rule` refuses if a `lab.candidate_proposed` event exists, if the spread has fewer than one degree of freedom or `n_failed > 0` or does not match the plan, if any rule assignment is not an evolution assignment, if `rule.planned_search_trials()` exceeds half of `LabConfig.affordable_trials`, or if the minimum detectable gain exceeds the assumed gain; `trial` refuses a spec that fails `check_trial_against_frozen`; `outer` refuses without a rule, without a terms-gate record (`GateNotRecorded`) and without the lab-wide dollar room for its worst case; `close-search` refuses while a search trial has no `lab.trial_finished`; `freeze-final-analysis` refuses before `close-search`, if planned search plus final trials exceed `affordable_trials`, or if any assignment is not held-out; `final` refuses before both `lab.search_closed` and `lab.final_analysis_frozen` exist and raises `FinalComparisonRefused` for a second request on the same pack commit without `--repeat-of --reason`; `reveal` refuses until every planned trial has its held-out and test score, then writes `FinalReveal` and `final_comparison.json`.
- `score` is a diagnostic: it goes through the same `ScoreGuard` as every other score, so it cannot re-roll a selection score or touch held-out or test data.
- `abort-trial <trial_id>` records `ABORTED` for a trial that has `lab.trial_started` but no `lab.trial_finished`: it rebuilds `budget_used` by replaying that trial's ledger, and uses the context hash from the trial directory if it exists, else null.
- `verify` calls `Ledger.verify_chain` on the master, every trial and the outer ledger; recomputes every `Decision` (delta, margin, outcome) from the ledger's scores and the rule loaded with `load_frozen_all` and fails on any mismatch; fails on any malformed event; prints each `head_anchor()`. `anchor` appends the master head to `anchors/master.anchors` at rule freeze, at every decision, at `final` and at `reveal`, and commits and pushes that file. `python -m nightshift verify <dir> --expect-head` also works on a lab directory (reuse notes section 3).
- `operator-change` records any change after the first candidate (pack lint ranges, rule inputs, proposer prompt, held-out families, v0 notes).
- During the search, `status` and `report` show the outer loop's view only (accept, reject, invalid bits and costs); the full view is `report --full`, which refuses before `lab.search_closed`. Neither ever prints per-prompt selection rows or ground truth to the terminal.

### 2. Scorer side (owner 1)

**`verifiers.py`**

- API: `IfbenchUpstream(repo_path)` and `IfEvalG(open_instruct_path)`, both with `score(rows, responses, *, mode) -> list[PromptResult]` and `pins() -> dict[str, str]`; `aggregate(results) -> dict[str, float]`; `preflight()`.
- Input: prompt rows (key, prompt, constraint ids, kwargs or `ground_truth`) and response texts, in the same order. Output: per-prompt, per-constraint flags and the metrics below.
- On disk: reads `third_party/IFBench` (commit `1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d`) and `third_party/open-instruct` (`b781e782a8fc530577f235aefe282dd046cb8ad2`, three `IFEvalG` files whose sha256 are checked against the pins).
- Failure modes: missing NLTK data is a `ScorerError` raised by `preflight()` (NOTE: otherwise a swallowed `LookupError` reads as reward 0); a missing response is a `ScorerError`; a checker exception is counted and surfaced as `n_errors`, never scored as an unexplained 0; `HOME` and `NLTK_DATA` are set explicitly so no import tries the network.

**`ifdata.py`**

- API: `load_rows(path: Path) -> list[dict]`, `build_splits(rows_jsonl: Path, ifbench_test_files: Sequence[Path], out_dir: Path, *, salt: bytes, evolution_families: Sequence[str], heldout_families: Sequence[str], sizes: Mapping[str, int]) -> Path`, `build_assignment_manifest(...) -> AssignmentManifest`. The salt is read from `layout.split_salt_path` (random 256 bits generated at `init`); only its sha256 is recorded as `split_salt_sha256`. Input is JSONL; converting the IF-RLVR parquet is the opt-in script `scripts/lab_convert_parquet.py`, the only place `pyarrow` is imported.
- Port of the reader's `build_splits.py` (deterministic for a given salt: sort by `sha256(salt|row_id)`, base-prompt decontamination against both test files and of every selection and held-out base prompt against all train and dev pools, drop unsatisfiable `nth_paragraph` rows and prompts over 12,000 characters). Extension: per assignment, `train`, `dev` and `selection` files are drawn from that assignment's pool. The evolution pool holds rows whose constraints are all in evolution families; the held-out pool holds rows whose constraints are all in held-out families; mixed rows are dropped (`MixedRowPolicy.DROPPED`) and recorded.
- Writes `data/assignments/<aid>/{train,dev}.jsonl` and `private/assignments/<aid>/selection.jsonl`. Selection files carry `ground_truth`, so they never leave the control host. Selection and held-out items are built as fresh constraint instantiations when the IFEvalG generators allow it (see [Closing the leaks](#closing-the-leaks)); `AssignmentManifest.item_mode` (`fresh_instantiation` or `salted_draw`) records which mode was used.
- Failure modes: the result differs between two runs (a test asserts byte identity); a held-out id appears in an evolution file (asserted); too few rows for a requested size.

**`rewards.py`**: port of the reader's `if_rlvr_reward.py` (`reward(response, ground_truth, mode)`, `verify_constraints`, `reward_batch`, `error_counts`, `preflight`) for researcher recipes. It is staged to the GPU by `hostkit`. Shares `IfEvalG` loading with `verifiers.py`.

**`generator.py`**: `HttpGenerator(base_url, api_key=None, workers=8, timeout, retries)` implements `Generator`. It posts to `/v1/chat/completions` with an explicit system message, `temperature`, `top_p`, `top_k`, `max_tokens`, `seed` and `chat_template_kwargs: {reasoning_strength}`, reads only `message.content`, records `finish_reason` and `usage.completion_tokens`, and keeps input order. A per-prompt transport failure is retried, then reported through `GenResult.error`; a dead endpoint raises `ScorerError`. **UNVERIFIED** against vLLM 0.32.0: `top_k` as an extra body field, and `chat_template_kwargs`.

**`scorer.py`**

- API: `ScoreService(layout: LabLayout, manifest: AssignmentManifest, verifiers: Mapping[Verifier, VerifierBackend], *, guard: ScoreGuard, clock: Clock)` implements `Scorer`, including `scorer_version` (`scorer_version_of(pins, metric names, SCORER_MODULE_VERSION)`). `VerifierBackend` is the module-local protocol `score(rows, responses, *, mode) -> list[PromptResult]` and `pins() -> dict[str, str]`.
- **Access policy** (a table in `scorer.py`; the phase is derived from the ledger by the guard, never trusted from the request):

| Phase | Assignment | Split | Rule |
|---|---|---|---|
| search | evolution | dev | allowed |
| search | evolution | selection | allowed only if `guard` passes: exactly one selection `ScoreRecord` per `(adapter_sha256, assignment, phase)`; seeds, decoding and scorer version equal the rule's; a second request only after a `lab.score_failed` with `result_revealed` False |
| search | held-out | dev, selection | `AccessDenied` (held-out data is unreachable in the search phase, dev included) |
| search | test id | test | `AccessDenied` |
| final | held-out | dev, selection | allowed only if `final_guard` passes |
| final | evolution | any | `AccessDenied` |
| final | test id | test | allowed only if `final_guard` passes |

- **`final_guard(request)`** (built by the cli from the ledger, injected as `guard`; called for EVERY final-phase score, including those the trial runner makes, so the trial runner cannot bypass it): the `FinalRequest` named by `request.final_request_id` exists, its arm equals `request.arm` and its pack commit equals the commit of the scored trial's `TrialSpec` (null for arm 1); `lab.search_closed` and `lab.final_analysis_frozen` exist, the request's `analysis_sha256` equals the frozen analysis (loaded with `load_frozen`), and no search trial is pending; `request.trial_id` is one of the analysis's planned trials with the same arm and assignment (arm 1 has none); the request's split is in the `FinalRequest.splits` (dev needs only a valid request id); seeds and decoding equal the analysis's; at most one held-out or test `ScoreRecord` exists per `(adapter_sha256 or "stock", split, assignment_id)`, and a further request is allowed only for each earlier `lab.score_failed` with `result_revealed` False; no second `FinalRequest` exists for the same `(arm, pack commit)` without `repeat_of` (the cli refuses it at `final`, the guard re-checks).
- Steps: check access; load rows from `LabLayout`; for each seed call the generator; refuse if any generation errored (`ScorerError`); verify with the verifier the split requires; write `gen_seed<k>.jsonl` (responses, private) and `per_prompt.jsonl` (flags only); write `scores/<id>/request.json` and `score.json` with `write_record` and return the `ScoreRecord` (`score_id == request_id`) with `verifier_pins`, `n_prompts`, the per-seed aggregates and the `FileRef`. The caller appends `lab.score_recorded` (reading the record back with `read_record` when it only has the id, as the trial runner does for the dev score of a training job).
- Metrics (constants in contracts): dev and selection record `METRIC_PROMPT_ALL` (every constraint passes; headline) and `METRIC_PROMPT_FRACTION` (open-instruct's mean reward); test records `METRIC_PROMPT_STRICT` (headline), `METRIC_PROMPT_LOOSE`, `METRIC_INSTRUCTION_STRICT`, `METRIC_INSTRUCTION_LOOSE`. The 'dev score' of a trial is `headline_mean()` of its dev `ScoreRecord`, unrounded. Truncated generations (`finish_reason == length`) count as failures in every arm and are counted in `n_truncated`.
- Failure modes: `AccessDenied` for a violation; `ScorerError` for missing data, hash mismatch of an input file, or an endpoint failure; the record is never built from missing responses.

**`score_session.py`**: `score_adapter(request: ScoreRequest, model_server: ModelServer, scorer: Scorer, generator_factory: GeneratorFactory, *, meter: Optional[BudgetMeter] = None) -> tuple[ScoreRecord, GpuUsage]` builds a `ServeSpec` for the base and the adapter, serves, scores with `generator_factory(endpoint)`, and always closes the session. The caller appends `lab.score_recorded` and `lab.score_cost`. One server serves the stock model and the adapter by LoRA name, so arm 1 and arms 2 and 3 share one process when scored together (Glimmer notes 4.3).

### 3. GPU side (owner 2)

Detailed in [GPU executor and watchdog](#gpu-executor-and-watchdog) and [Glimmer trainer and serving plan](#glimmer-trainer-and-serving-plan). Public API summary:

- `gpu_rentals.RentalManager(rental_ledger_path: Path, provider: GpuProvider, *, clock: Clock, poll_seconds: float = 5.0)`: `init(hard_stop_usd, machine, why)`, `create(session, shape, cap_hours, trial_id, item) -> Rental`, `wait_ready`, `endpoint(rental)`, `delete(rental, reason)`, `rearm`, `recover`, `usage_for(trial_id) -> GpuUsage`, `cost()`. Local `RentalRefused(GpuError)` when the money rule says no.
- `gpu_watchdog`: `python -m nightshift.lab.gpu_watchdog watchdog <session> --at-epoch E` and `reap` (idempotent, run from cron or a timer on a second host).
- `gpu_thunder.ThunderProvider(api_token_env: str, base_url: str, *, timeout: float = 30.0)` implements `GpuProvider` over the REST API; the ownership tag is `owner_tag_of_key(ssh_public_key)`, derived by the provider, so there is no separate tag parameter.
- `gpu_remote`: `SshTarget`, `upload(target, local, remote)`, `fetch(target, remote, local, *, sha256, max_bytes)`, `run_detached(target, argv, log)`, `forward_port(target, local_port, remote_port)`; every transfer verifies sha256, refuses symbolic links, hard links, devices and any path that escapes the destination (`rsync --no-links` plus an `lstat` walk, or extraction from a tar that contains none), and enforces a per-file and total size cap.
- `model_server.RentalModelServer(manager: RentalManager, shape: Shape, hostkit_dir: Path)` implements `ModelServer`; its function `serve_on_lease(lease, spec: ServeSpec) -> ServeEndpoint` is shared with the trainer. `ServeEndpoint.base_url` is a control-host loopback port forwarded with `gpu_remote.forward_port`; the control host never talks to a public port of a rental.
- `trainer.GlimmerTrainer(manager: RentalManager, shape: Shape, scorer: Scorer, generator_factory: GeneratorFactory, layout: LabLayout, hostkit_dir: Path, base: BaseModelRef, *, clock: Clock)` implements `Trainer`. It fills `job.dev_request` with `with_adapter`, calls `scorer.score` (which writes the dev score files) and returns `TrainResult.dev_score_id`; it also fetches the processed training dataset (`train_data`) and accepts from the rental only `adapter_config.json` and `adapter_model.safetensors`, by exact name, as regular files, with `adapter_config.json` keys validated.

### 4. Report and statistics (owner 3)

**`stats.py`**: all functions pure and deterministic; implements `DecisionRule`. See [Statistics](#statistics) for the formulas.

- `trial_spread(scores: Sequence[TrialScore]) -> SpreadEstimate`: cells are assignments with at least two trials; a cell with one trial is excluded from `trial_ids`; `df = sum(n_a - 1)`; `sigma^2` is the pooled within-cell variance. Raises `ContractError("too few trials to estimate spread")` when no assignment has two trials.
- `t_quantile(alpha: float, df: float) -> float`: one-sided Student-t quantile for any alpha and df (about 20 lines, no table), tested against the reused `nightshift.grading` `T95` and `T975` at their tabulated df. Nothing is imported from `nightshift.grading` by production code; see [Critic findings and resolutions](#critic-findings-and-resolutions) (A28).
- `screen(rule, incumbent, candidates) -> dict[str, Optional[float]]`: the delta of each candidate against the incumbent, or None when its trials are not comparable.
- `decide(rule, incumbent, candidate) -> DecisionNumbers(outcome, delta, margin, df, sigma_used, reason)`: `INVALID` when an assignment lacks trials on either side, any `comparability_sha256` differs within an assignment, the counts differ from the rule's, or the candidate's pooled sample sd exceeds `variance_guard_ratio` times the frozen sigma; never raises for data problems.
- `minimum_detectable_gain(sigma: float, df: int, n_candidate: int, n_incumbent: int, n_assignments: int, alpha: float, power: float = 0.8) -> float`, used by `freeze-rule` (the gain the confirmation detects with that power).
- `compare_final(analysis: FinalAnalysis, v0: Sequence[TrialScore], evolved: Sequence[TrialScore], split: Split) -> FinalComparison` (blocked difference with a t interval over assignments when there are at least two, Welch over trials when there is one); `compare_to_stock(arm_scores: Sequence[TrialScore], stock: ScoreRecord)`.
- Diagnostics: `seed_spread(score_record)`, `assignment_spread(...)`, `paired_bootstrap_prompts(per_prompt_a, per_prompt_b, *, resamples, seed)`, `mcnemar(per_prompt_a, per_prompt_b)`. These describe two checkpoints on identical prompts; they never decide a keep.
- Failure modes: scores computed on different scorer versions, metrics, prompt sets, budgets or GPU shapes are `INVALID` (as `decide_outcome` treats `grader_hash`, split and seed mismatch).

**`report.py`, `chart.py`**

- API: `build_report(state: LabState, layout: LabLayout, *, full: bool = False) -> Report` (`Report` is module-local; the cli passes it straight to the renderers), `render_markdown(report) -> str`, `render_html(report) -> str`, `chart.search_trace_svg(points, title) -> str` (standard-library SVG, no dependency).
- Contents: the three-arm table; paired per-prompt comparison between arms (labelled checkpoint-level); the three spreads (generation seed, trial, assignment); the search-trace chart titled "adaptive search trace" with accepted, rejected and invalid candidates marked; the cost table with researcher inference, GPU, outer proposals and scoring as separate rows; the held-out comparison with its uncertainty over assignments and trials.
- Hidden until `lab.final_revealed`: per-trial held-out and test scores. Before `lab.search_closed`, `full=True` is refused and the default view is bits and costs only.
- Money: budget enforcement uses the meter's ticks; money reporting uses the rental ledger (`GpuUsage`, rounded-up minutes). The cost table prints both whenever they differ.
- Labels it must print: the verifier and prompt count behind every score (partial or custom scoring labelled as such); "arm 3 versus arm 2 was not tested" when no held-out comparison exists; "components tested, system not yet demonstrated" when no joined trial exists; "spread not estimated" when the rule is missing; a warning for every `usage_estimated` charge; "dev scores are researcher-influenced; decisions use control-host numbers only"; the isolation level of each trial; the residual contamination risk; "the bits and the operator's choices steer the search"; the minimum detectable gain next to every final interval.
- Failure modes: a referenced file whose sha256 does not match is listed as a warning and its numbers are not used; it reads the ledger only through `fold`.

### 5. Outer loop and packs (owner 4)

Detailed in [Outer-loop controller and redaction rules](#outer-loop-controller-and-redaction-rules).

- `packs.PackRepo(layout: LabLayout, *, editable_paths: Sequence[str], lint_ranges_path: Path)` implements `PackStore`: `init_v0(template_dir) -> PackRef`, `checkout(pack, dest)`, `apply(proposal, parent, version_label) -> tuple[PackRef, str]`, `lint(pack) -> LintResult`, `accept(pack, version_label)`, `tree_sha256(pack) -> str`. `lint` is two steps: parse `task.yaml` inside CORAL's virtual environment and dump it as JSON (`assets/lint_pack.py --dump`), then compare keys in pure Python over that dump (`packs.compare_task_keys(parent_json, candidate_json, lint_ranges) -> list[str]`), which is tested offline on JSON fixtures.
- `outer_context.build_context(facts: ProposalFacts, *, fence) -> ProposalContext` and `assert_no_forbidden(prompt: str, forbidden: frozenset[str])`; `outer_context.forbidden_values(state: LabState, layout: LabLayout, hosts: Sequence[str]) -> frozenset[str]` (selection scores, sigma, margin, delta, held-out ids and families, control-host paths and hostnames, tokens), called by the controller every time and asserted non-empty once the rule exists. `ProposalFacts` is module-local (produced and consumed inside owner 4).
- `proposer.WonderProposalSource(client: MeteredClient, *, schema: Mapping[str, Any])` implements `ProposalSource`.
- `outer_loop.OuterController(layout: LabLayout, ledger: LedgerWriter, state_reader: StateReader, packs: PackStore, proposer: ProposalSource, executor: TrialExecutor, decisions: DecisionRule, *, clock: Clock, lab_cap: LabCap)`: `run_round() -> list[Decision]` (one screening round plus at most one confirmation) and `run(max_rounds: int) -> list[Decision]`.

### 6. Runtime and trial runner (owner 5)

Detailed in [CORAL integration plan](#coral-integration-plan) and [One trial, end to end](#one-trial-end-to-end).

- `runtime_coral.CoralRuntime(layout: LabLayout, researcher_target: SshTarget, coral_home: str, pins: Mapping[str, str], port_allocator: PortAllocator, *, clock: Clock)` implements `ResearcherRuntime`. `RuntimeHandle.stop(reason: ExitReason)` takes the enum, not free text. `collect()` returns every attempt that left a `submission.json` as a `SubmittedAttempt(submission, committed_utc)`, taken from CORAL's attempt JSON; it refuses symbolic links, hard links and devices on the way.
- `metering_proxy.MeteringProxy(meter: BudgetMeter, upstream_base_url: str, api_key: str, token: str, *, clock: Clock)`; `control_api.ControlApi(trainer: Trainer, meter: BudgetMeter, layout: LabLayout, spec: TrialSpec, token: str, ledger: LedgerWriter, *, clock: Clock)`; both are mounted on one `ThreadingHTTPServer` (loopback) by the trial runner.
- `researcher_tool` (`glimmer budget | submit | status | wait | dev-score | pointer`), standard library only, run on the researcher host; interface in the [appendix](#interface-appendix).
- `trial.TrialRunner(layout: LabLayout, ledger: LedgerWriter, new_trial_ledger: TrialLedgerFactory, meter_factory: MeterFactory, runtime: ResearcherRuntime, trainer: Trainer, scorer: Scorer, model_server: ModelServer, generator_factory: GeneratorFactory, packs: PackStore, state_reader: StateReader, *, clock: Clock, port_allocator: PortAllocator)` implements `TrialExecutor`. It keeps no module-level state, so concurrent `run` calls are independent.

## One trial, end to end

Preconditions recorded in the master ledger: `lab.assignments_frozen`, `lab.budget_frozen`, the pack commit (`lab.pack_committed`). A search trial of a candidate also needs `lab.rule_frozen` and `lab.gate_recorded`. A final trial also needs `lab.search_closed`, `lab.final_analysis_frozen` and a `FinalRequest`. Ids (`trial_id`, score ids, job ids) are allocated by the coordinator with `next_id` before the spec is built.

1. **Start.** The runner receives a `TrialSpec` and calls `check_trial_against_frozen(spec, manifest=, budget=, rule=, analysis=)`; a mismatch refuses the trial before anything is spent. It creates `trials/<tid>/`, writes `spec.json`, appends `lab.trial_started` (master), opens the trial ledger with `new_trial_ledger(trial_id)`, builds the meter with `meter_factory(spec, trial_ledger)`, and registers the trial token with `secrets.register`. Both ledgers are thread-safe `LabLedger` objects inside the coordinator process; this method runs in a thread of it.
2. **Clean workspace.** `PackRepo.checkout(spec.pack, layout.trial_pack_dir(tid))`; `tree_sha256` of the commit is recorded. The runtime copies the checkout, the assignment's `train.jsonl` and `dev.jsonl`, the researcher tool and the runtime-owned config to the researcher host, into a throwaway trial user's home. It computes `initial_context_sha256` over the sorted listing `sha256 size path` of the whole initial worktree, notes included.
3. **Services.** The runner starts one loopback HTTP server holding the metering proxy (`/v1/...`) and the control API (`/control/...`), and opens an SSH session to the researcher host with `-R`, so the researcher host reaches them on its own loopback.
4. **Launch.** `runtime.start(context)` runs `coral start` detached with the overrides in the [CORAL plan](#coral-integration-plan). The runner then polls `handle.poll()` and the meter every few seconds.
5. **Research loop (on the researcher host).** Spark researchers edit code and call `glimmer submit` with a recipe directory. The control API checks the token, stores the recipe tarball by sha256 (rules in [File formats](#file-formats)) and starts one job: `Trainer.run_job(TrainJob, meter)`. The `TrainJob` has `shape = spec.shape` and a dev template with arm and adapter null, `phase = spec.phase`, `final_request_id = spec.final_request_id`, `seeds = spec.score_seeds`, `params = spec.score_params`, and a request id allocated now (`<score id>`); the trainer fills the adapter with `with_adapter(job.dev_request.request_id, adapter)`.
6. **Job.** The trainer caps the lease from the meter, rents a GPU, bootstraps it, uploads and runs the recipe, charges one GPU minute per tick, fetches the adapter and the processed training dataset (symlinks and extra files refused) and verifies them, serves the adapter on the same rental, scores dev through `Scorer.score` on the control host (which writes `scores/<id>/request.json` and `score.json`), fetches timings, deletes the rental (confirmed absent), and returns a `TrainResult`. The runner then (a) reads the dev record with `read_record(layout.score_record_path(result.dev_score_id), ScoreRecord)` and appends `lab.score_recorded`, (b) appends `lab.gpu_job_started` and `lab.gpu_job_finished` to the trial ledger, (c) scans `result.train_data` and the recipe tarball against the selection, held-out and test prompts and appends `lab.contamination_scanned`. A hit stops the runtime at once (`ExitReason.STOPPED_CONTAMINATED`) and marks the trial CONTAMINATED; its adapter is never scored on selection. The researcher sees the dev aggregate through `glimmer status`, and the CORAL grader returns the same number to `coral eval`.
7. **Stop.** The run ends when the researchers finish, `run.stop.max_real_attempts` is reached, or the runner stops it because a budget axis is exhausted or wall time is over (`handle.stop(ExitReason.STOPPED_BUDGET | STOPPED_WALL)`). The runner never waits on the researcher to stop itself.
8. **Harvest.** `handle.collect()` returns the attempts, the traces (opencode event logs, concatenated by restart number), the notes, the environment pins and every `SubmittedAttempt`. DECISION: the runner picks the submission, so the researcher has no discretion over which checkpoint is scored: among attempts whose `adapter_id` appears in THIS trial's own `lab.gpu_job_finished` events (an adapter id from another trial is ignored), the one with the highest control-host dev score (`headline_mean()` of its dev record), ties to the earliest `committed_utc`. A `dev_score` written by the researcher into `submission.json` is discarded; the runner writes the control-host value into `Submission.dev_score`. Dev numbers are labelled researcher-influenced (the recipe runs as root on the rental that produced them); decisions use selection numbers from a separate, fresh scoring rental. The runner records `git diff` of the pack checkout (agents can edit the grader source on the default path).
9. **Adapter.** The adapter already sits in `artifacts/adapters/<aid>/` from step 6; its `AdapterRef` is re-verified with `verify_adapter_dir(ref, root=layout.root)`.
10. **Selection score** (skipped for a contaminated trial). `score_session.score_adapter(...)` on a scorer-side rental that boots from a fresh, immutable image with the base weights verified by sha256 and shares no writable disk with any rental a researcher recipe ran on: `ScoreRequest(request_id=<score id>, phase=spec.phase, assignment_id=spec.assignment_id, split=selection, trial_id, arm=spec.arm, adapter, seeds=spec.score_seeds, params=spec.score_params, scorer_version=spec.scorer_version, final_request_id=spec.final_request_id)`. The `ScoreGuard` runs inside `ScoreService`; for final-phase trials that is `final_guard`, so the runner cannot score held-out or test data outside the plan. The runner appends `lab.score_recorded` and `lab.score_cost`. If the scorer fails without producing a number it appends `lab.score_failed` (`result_revealed` False); the guard then allows exactly one retry with the same seeds. After that failure the trial is `FAILED` with the adapter kept. A final-phase trial also scores the test split under the same guard (`assignment_id = TEST_ASSIGNMENT_ID`).
11. **Close.** The runner seals the trial ledger, reads its `head_anchor()`, writes `trial.json` and appends `lab.trial_finished` with the `TrialRecord` (budget used from the meter's ticks, adapter, score ids, trace, `trial_ledger_anchor`). Tokens and ports are released, the trial user's processes are killed and its home is deleted. The status follows this table.

| Runtime exit / outcome | Trainer, scan, scorer outcome | `TrialStatus` |
|---|---|---|
| any exit | a submission from this trial's own jobs was selected and scored, a meter axis (tokens, GPU minutes, dollars, lab dollars or wall seconds) ended the run | `BUDGET_EXHAUSTED` |
| `FINISHED` or `CRASHED` | selected and scored | `COMPLETED` (a crash is noted in `environment["runtime_exit"]`) |
| `FINISHED`, `CRASHED`, either stop | no attempt with an adapter from this trial's own jobs | `NO_SUBMISSION` (`FAILED` if the exit was `CRASHED` and nothing was submitted) |
| `STOPPED_CONTAMINATED` | contamination scan hit | `CONTAMINATED` |
| any | selection scoring failed twice | `FAILED`, adapter kept |
| coordinator died before step 2 | none | `ABORTED` by `cli abort-trial`, `initial_context_sha256` null |
| coordinator died after step 2 | none | `ABORTED` by `cli abort-trial`, context hash read from the trial directory |

Crash handling: the master ledger shows `lab.trial_started` without `lab.trial_finished`; `cli status` lists such trials; the rental manager's own ledger and watchdogs have already limited the money. A trial is never resumed; `cli abort-trial` records it `ABORTED`, rebuilding `budget_used` by replaying the trial ledger, and a new `trial_id` is used.

## Budget enforcement

- `budget.json` (`BudgetSpec`) is equal for every arm-2 and arm-3 trial and frozen before candidates run; `check_trial_against_frozen` refuses a trial whose `spec.budget` differs, and `comparability_key` includes the budget and the GPU shape so a trial run under another budget can never be compared. Four axes: researcher tokens (prompt plus completion tokens), GPU minutes, dollars (researcher inference plus GPU together), wall seconds. DECISION: the dollar axis is one cap over both, so spending on either shrinks room for the other and a single number is auditable. `AXIS_OF` in contracts maps `RESEARCHER_USD` and `GPU_USD` to `dollars`.
- **Where it is enforced, all outside the researcher:**
  - The metering proxy (control host) is **reserve-then-settle**. Before forwarding it calls `meter.reserve_llm(model, prompt_tokens_estimate, max_tokens, call_id)`: the estimate is the request's UTF-8 bytes divided by 3 (conservative for English; UNVERIFIED for the Muse tokenizer; the first trial reconciles it against the provider's `usage`), `max_tokens` is the request's value or `limit.output` 16000 if absent, and the worst case is priced with the reused `usd_for`. If booking it would cross a cap the proxy answers HTTP 429 and the runner stops the runtime. On the response `usage` it calls `meter.charge_llm(model, LlmUsage, call_id)`, which replaces the reservation with the actual usage. `charge_llm` books two events with call ids `<call_id>:tokens` and `<call_id>:usd`, both before it raises `BudgetExhausted`, so a cap crossed by the first half never leaves the second unbooked. A stream that ends without a `usage` chunk, or a client that disconnects, leaves the worst-case reservation booked, so a missing chunk can only over-count. The reservation is also appended to the trial ledger as `lab.budget_reserved` before the request is forwarded (if that append fails the call is refused and both axes close); a reopened meter and `usage_from_events` treat a reservation with no `<call_id>:tokens` settlement as still booked, so a coordinator crash cannot erase spend the provider may have billed. The proxy forces `stream_options.include_usage` on every request.
  - The GPU executor charges one minute per tick while a rental is live, with call id `gpu:<rental_id>:<n>` (`GPU_MINUTES` and `GPU_USD` at the quoted rate; billing is per minute while running and starts at the create attempt, which is conservative; GPU notes R9). A `BudgetExhausted` from any tick makes the trainer delete the rental at once.
  - The rental's `cap_epoch` (armed in the rental ledger before the provider is called) and the detached watchdog stop money even if the coordinator dies (GPU notes section 4.1).
  - Wall time is derived from the injected clock since trial start; the runner's own timer calls `check(WALL_SECONDS)`. Nobody calls `charge(WALL_SECONDS)`.
- **Overshoot bound** (DECISION, stated rather than guaranteed to be zero). LLM spend: the cap cannot be crossed by a call whose prompt estimate was right, because the worst case is reserved before the call. The residual error is the prompt-estimate error, so the bound is `(concurrent researchers) x (actual prompt tokens - estimated prompt tokens)` per axis, and the proxy rejects a request whose estimated prompt alone would cross the cap. The earlier claim that `limit.output` 16000 bounds a call was wrong: it bounds completion tokens only, CORAL's context limit is 1,048,576 (CORAL notes 1.3) and prompt tokens count toward tokens and dollars. GPU spend: one tick (one minute). The report prints which bound applied and the measured estimate error.
- **Lab-wide cap.** `LabConfig.lab_budget_usd` caps researcher inference plus Wonder across the whole lab (the Meta credit). One shared `LabCap` (thread-safe, seeded from the ledger on start) is passed to every trial meter; `reserve_llm` also raises `BudgetExhausted("lab_dollars", ...)` when the lab-wide remainder (cap minus ledger-derived spend minus every trial's outstanding reservations) would be crossed. At `outer` start the controller computes the worst case of the coming round (candidates times trials times the per-trial dollar cap, plus Wonder calls times their per-call cap) and refuses if it exceeds the recorded balance. Candidates are counted from the ledger (`lab.candidate_proposed` events, invalid ones included), so repeated invocations of `outer` cannot exceed `rule.max_candidates`.
- **Researcher cannot raise it:** the budget file, meter, proxy token, control API, rental manager, Thunder token and ledger are on the control host. The researcher holds a token that can only submit jobs inside the trial.
- **Two cost numbers.** Enforcement uses the meter's one-minute ticks (quoted rate, charged from the create attempt) and is what `BudgetUsage` in `TrialRecord` holds. Money reporting uses the rental ledger's rounded-up `GpuUsage` (`TrainResult.gpu`, `ScoreCost.gpu`). They differ; the report prints both. `StateReader.cost_breakdown(state, trial_ids, candidate_id)` sums a candidate's trials, its scoring and its proposal into a `CostBreakdown`; the outer controller calls it when it builds a `Decision`.
- **Outer-loop cost is separate** (not against any trial): Wonder calls go through the reused `MeteredClient` and `BudgetGate` with a dollar cap in `outer/ledger.jsonl`. Candidate trials still run under the same per-trial cap, and their cost is reported as candidate-trial cost. `CostBreakdown` keeps four centres apart: outer proposals, candidate researcher inference, candidate GPU, scoring.
- **Provider billing.** It is UNVERIFIED whether `usage` includes redacted reasoning tokens. The first trial's token and dollar totals are reconciled against the provider's billing before the token axis is trusted; the report warns for every charge whose `detail` says `usage_estimated`.
- Pre-flight: `price_for(model)` is asserted at startup because an unknown model id would be charged at the Spark rate (reuse notes 5.2).

## Ledger decision

**Reuse `nightshift.ledger.Ledger` unchanged**, in separate directories, with lab-owned kinds. Reasons from the reuse notes:

- Any string kind is accepted; verified by appending `lab.trial_started` and checking the chain and `view().malformed == []` (section 4). No edit to `types.py`.
- Hash chaining, the advisory writer lock, torn-line recovery and Ctrl-C handling are already tested (`tests/test_ledger_interrupt.py`, `tests/test_ledger_view.py`). A new ledger would lose that for a four-day build, and tamper evidence is a submission claim.
- Per-trial ledgers give per-trial budget isolation for free: `BudgetGate` and `view()` are per ledger, and `view()` is rebuilt per append (3000 charges took 4.9 s), which argues for small ledgers.
- `python -m nightshift verify <dir> --expect-head <seq>:<hash16>` already works on a lab directory.

**Costs we accept and how each is handled.**

| Limit | Handling |
|---|---|
| Single writer process per file | The coordinator is one process. `TrialRunner.run` runs in a thread of it, trials may run concurrently, and `LabLedger` is thread-safe: it is the only object that touches the master ledger, and each trial ledger (created and closed inside `run`) is a `LabLedger` too, shared by the proxy, control API, GPU tick and trainer threads. No other process opens a lab ledger for writing. The scorer writes files; the caller appends the event with the file hash. Rental processes write the rental ledger, not a `Ledger`. |
| Strings over 8192 bytes are truncated | Payloads hold references; contracts limit text to 4000 bytes. |
| `NaN` is written as a bare token | Contracts reject non-finite numbers. |
| Credentials are scrubbed only if registered | `register_secrets` for Thunder, Meta and trial tokens before the first append. |
| The ledger proves order and non-editing, not truth, and the hash chain has no secret | The report states this. Master head anchors are appended to `anchors/master.anchors` at rule freeze, at every decision, at `final` and at `reveal`, and that file is committed and pushed to git (the remote is an open decision), so a rewritten whole ledger no longer matches. `head_anchor()` of each trial is copied into `lab.trial_finished` as `TrialRecord.trial_ledger_anchor` (the reuse notes call it `data.ledger_anchor`; that name is stale). |
| Several ledgers to keep in sync | One master record per trial carries the child anchor. |
| Nothing makes a frozen file freeze-once | `LabKind.FREEZE_ONCE`: `LabLedger.append` refuses a second event of those kinds; the fold keeps the first and lists later ones as malformed, which fails `verify`. `frozen/` files are made read-only (`chmod`, or `chattr +i` when root). |
| Two different hashes | `FrozenFile.file.sha256` is the hash of the bytes written (`write_record` returns it). `Decision.rule_sha256` is `Rule.sha256()`, the hash of the canonical JSON of the parsed record. They are never compared with each other. `FoldReader.load_frozen_all` and `load_frozen` check the file against the ledger's `FileRef` first; the controller and `verify` use it every time; a `Decision` is checked against `Rule.sha256()` of the loaded rule. |
| A self-consistent forged `Decision` passes every contract check | `verify` recomputes every decision's delta, margin and outcome from the ledger's scores and the rule and fails on a mismatch. |

The reused Nightshift read model `LedgerView` ignores lab kinds, so `fold.py` is the read model. Reuse for `Event`, `canonical_json`, `sha256_hex` comes from `nightshift.types`.

## Statistics

**What is measured, at three levels, reported separately.**

1. **Generation-seed spread**: the standard deviation, within one `ScoreRecord`, of the headline metric over generation seeds. Glimmer is not reproducible at temperature 1.0, so several seeds are required (Glimmer notes 4.1, 8.13). This is scorer noise.
2. **Trial spread**: the standard deviation of per-trial scores `x` among independent trials of the same pack on the same assignment. A trial's `x` is `ScoreRecord.metric_by_seed(rule.metric)` averaged over its generation seeds on the selection split. This contains researcher and training variance and is the quantity the rule needs. `trial_spread` uses cells (assignments) with at least two trials, `df = sum(n_a - 1)`, `sigma^2` = pooled within-cell variance.
3. **Assignment spread**: the standard deviation of the per-assignment means. Reported when there is more than one assignment.

**Which trials are measured and which count.** The spread plan (`lab.spread_planned`) fixes the number of v0 trials before the first one runs. Sigma is computed from ALL completed trials of that plan, never from a subset; `freeze-rule` refuses unless `n_failed == 0` and the measured trials equal the plan (otherwise a new plan is declared and the old attempt stays in the ledger). For search trials, every scheduled trial counts: a `BUDGET_EXHAUSTED` trial with a scored adapter counts normally; any other non-scored trial follows `Rule.failed_trial_policy` (`INVALIDATES_CANDIDATE`, recommended, or `SCORED_ZERO`; an "exclude" policy does not exist because it judges a flaky pack on its lucky survivors). Any candidate that reached trial start counts toward `max_candidates`.

**Comparability.** `TrialScore.comparability_sha256` is `comparability_key(spec, metric)`: scorer version, metric, decoding, score seeds, the selection file hash, the budget, the GPU shape and the base checkpoint. Within one assignment every incumbent and candidate trial must carry the same key; otherwise the decision is `INVALID`. Incumbent trials are interleaved with candidate trials (alternating order and start time) so that time drift, such as an A100 versus H100 availability change, is balanced.

**The keep rule (`Rule`, frozen before any candidate): screen, then confirm.**

- **Screen.** Each round proposes up to `screen_batch_size` (at most 5) candidates. Each runs `screen_trials_per_candidate` (recommended 2) trials per assignment. `delta_k = (1/A) * sum_a (m_k,a - m_i,a)` against the incumbent's existing trials. There is no threshold: the candidate with the largest `delta_k > 0` advances (ties to the smaller id); every other candidate gets a `SCREEN` decision, `REJECTED` with its `screen_delta` (or `INVALID`).
- **Confirm.** Only the advanced candidate is tested: `confirm_trials_per_candidate` FRESH trials of it and `confirm_trials_per_incumbent` FRESH, interleaved trials of the incumbent, per assignment. Screen trials are not reused, because the candidate was chosen for a good screen (winner's curse).
- `delta = (1/A) * sum_a (m_c,a - m_i,a)`; `se_factor = (1/A) * sqrt( sum_a (1/n_c,a + 1/n_i,a) )`.
- **Pooled sigma.** `sigma_used^2` pools the frozen v0 variance (`df0 = spread.df`) with the candidate's and the incumbent's own within-cell variances from the confirmation, `df = df0 + sum_a (n_c,a - 1) + sum_a (n_i,a - 1)`. The rule requires `df >= 8` (`MIN_POOLED_DF`, the value of the critics' simulation), so one underestimated sigma from three trials cannot decide a round.
- `margin = t_quantile(alpha, df) * sigma_used * se_factor + min_gain`, one-sided, `alpha` in the rule (0.05 recommended). **Accept iff `delta > margin`.**
- **Variance guard.** The candidate is `INVALID` if its pooled sample sd exceeds `variance_guard_ratio` (1.5 recommended) times the frozen sigma. This replaces a Welch term for the candidate's own variance (see B3 in [Critic findings and resolutions](#critic-findings-and-resolutions)).
- An assignment without trials on either side, or any comparability mismatch, makes the decision `INVALID`, which counts as rejected and is not retried.
- **Not a proof.** The rule is a heuristic filter, not evidence. The claim rests on the held-out comparison. Across `R = ceil(max_candidates / screen_batch_size)` rounds the chance of at least one false accept under a true null is at most `1 - (1 - alpha)^R` (about 0.10 for two rounds at alpha 0.05); the report prints it.

**Evidence for this choice (reported by the critics, simulations kept in their scratchpad as `far_sim.py` and `far2.py`; not rerun in this design).** One-sided 95 % t test, sigma frozen from the v0 trials, accepted candidates becoming the incumbent, all candidates truly equal: P(at least one false accept) was 0.13 (K = 5), 0.19 (K = 10), 0.25 (K = 20) candidates for 4 v0 trials, 3 candidate trials, 3 incumbent trials, and 0.15 / 0.21 / 0.29 for 2 candidate trials; more v0 trials did not help (K = 10: 0.14 to 0.22 for 2 to 8 v0 trials). Per-test false-accept rate was 0.9 % at the true sigma but 12 % when the estimate was half the truth (22 % of the time at df 2), and a candidate with twice the trial sd gave 0.50 at K = 10. Screen-then-confirm with a pooled df of at least 8 gave about 0.05 for any K, P(accept) 0.67 at a +2 sigma gain, and a cost of `2K + 9` trials. **Acceptance test** (so the numbers are not taken on trust): `tests/test_lab_stats.py` simulates the controller with a fixed seed and asserts the false-accept rate stays below 0.08 for K = 5 and 10 candidates all equal to the incumbent, and also with a candidate sd of twice the incumbent's (the variance guard must hold it); if either fails the owner stops and reports instead of loosening the threshold. The 0.08 bound is the critic-reported 0.05 plus Monte-Carlo slack, not an arbitrary number.

**Power, honestly.** At affordable sizes the confirmation is demanding. For the sequential variant with 3 candidate and 3 incumbent trials the critic reports P(accept) of about 0.25, 0.58 and 0.87 at true gains of 1, 2 and 3 trial standard deviations, and a minimum detectable effect (80 % power, one assignment, Welch) of about 3.0 sigma at 3 trials per pack and about 2.0 sigma at 5. Therefore: `stats.minimum_detectable_gain` is computed from the measured sigma and the confirmation sizes, stored in `Rule.minimum_detectable_gain` next to `Rule.assumed_gain` (the gain the operator believes a good candidate has), and `freeze-rule` refuses when the first exceeds the second. With sigma 0.02 and a minimum detectable gain of three sigma, the operator must believe in a six-point gain or the lab cannot honestly run a search. The design's earlier example margin (0.053 at df 2) is withdrawn.

**Allocation.** `Rule.planned_search_trials()` is the worst-case count of search trials (v0 spread trials, every candidate screened, every round confirmed). `freeze-rule` refuses when it exceeds half of `LabConfig.affordable_trials`, and `freeze-final-analysis` refuses when search plus final trials exceed all of it: at least half of the affordable trials are reserved for the confirmatory comparison, and the final plan needs at least 5 trials per pack.

**Why this method and not the per-prompt paired methods.**

- The unit of independent evidence for a researcher is the trial. Resampling prompts (paired bootstrap, McNemar, or `decide_outcome` on per-prompt vectors) treats the two adapters as fixed and measures only prompt-sampling noise; with a few hundred prompts it yields a very small margin and would keep noise (spec; reuse notes 2.2 and outer-loop notes 3.4). `decide_outcome` paired by seed index is not a true pairing across different packs either.
- The comparison is paired on the factor that really repeats, the assignment. With one evolution assignment the blocking is trivial and the test reduces to a two-sample comparison of trial means.
- Sigma is anchored on the v0 trials as the spec says ("based on the measured spread of v0 trials") and pooled with the confirmation's own variances. It does not carry over to the held-out comparison, which re-estimates spread from its own trials.
- If no assignment has two v0 trials, the spread has no degrees of freedom: `freeze-rule` refuses and the report prints "spread not estimated" instead of applying a rule without evidence (spec).

**Final comparison (confirmatory, `FinalAnalysis`).** `frozen/final_analysis.json` is frozen before any held-out or test score exists and holds: the held-out metric and the test metric, the held-out assignments with weights, the planned trial ids (arm, assignment, seed), the two-sided level, the method, score seeds and decoding, the failed-trial policy, the retry policy and the budget hash. Held-out assignments `h` (at least one) get repeated budget-matched trials of frozen v0 and the frozen evolved pack from stock Glimmer:

- `Delta = sum_h w_h * (m_evolved,h - m_v0,h)`.
- With `H >= 2`: a two-sided t interval over the per-assignment differences (`df = H - 1`, `ComparisonMethod.BLOCKED_T`).
- With `H = 1`: a Welch interval over trials (both sides need at least two trials; Welch-Satterthwaite degrees of freedom, floored). The report labels this "within one assignment; does not generalise across assignments".
- Arm 1 versus arm 2 on the test split: arm 2's trial scores against arm 1's mean over generation seeds, plus a checkpoint-level paired bootstrap and McNemar over the same prompts. The bootstrap uses a fixed seed; the number of resamples (default 2000) is a reporting resolution, not a decision threshold, and is printed.
- Arm 3 versus arm 2 is reported with the number of assignments, the number of trials per pack, the interval, the minimum detectable gain, and a plain statement when the interval includes zero.
- **Hidden until reveal.** Per-trial held-out and test scores do not appear in `status` or `report` until `lab.final_revealed`; `reveal` refuses until every planned trial has its scores, so trial counts cannot be adapted to results. Exactly one held-out or test `ScoreRecord` exists per planned trial and split; a retry is allowed only after a recorded infrastructure failure that revealed no number.

## Real and tested offline, versus GPU or Meta key only

No GPU and no Meta key are available where the tests run. "Real" means the production code runs in the test; the thing it talks to may be a fake.

**Real code, tested offline in `unittest` (no GPU, no key):**

| Area | What the test uses |
|---|---|
| Contracts (done) | pure |
| Ledger wrapper, fold, budget meter | the real `Ledger` with `FakeClock`, `fsync=False` |
| Statistics, report, chart | `LabState` built by hand; numeric checks against hand-computed values and `T95`/`T975`; a seeded Monte-Carlo test of the screen-then-confirm false-accept rate |
| Pack git operations, strict apply, lint key comparison, tree hash | real `git` on temporary repositories; the lint comparison runs over already-parsed JSON dumps of `task.yaml` (fixtures), because Python has no standard-library YAML parser and CORAL's virtual environment exists only on the researcher host; the YAML-to-JSON dump step is a lead-run check on that host |
| Redaction | canary values placed in facts, final prompt searched |
| Proposer parsing and retry | `FakeMetaServer` with the real `MetaChatClient`; `ScriptedClient` for logic |
| Verifiers | opt-in: set `LAB_THIRD_PARTY` to the directory holding the pinned IFBench and open-instruct checkouts (`scripts/lab_setup_verifiers.sh`) and have `nltk`, `langdetect`, `immutabledict` and the NLTK data; without it the tests skip with the message `set LAB_THIRD_PARTY to run the verifier parity tests`. The parity test (a mock `/v1/chat/completions` reproducing the unmodified `run_eval.py` flags, as the reader's smoke test did) is a lead-run acceptance check, not a CI gate; CI covers the adapter logic with fake verifiers |
| Split builder | reads JSONL rows; the reader's determinism test on a tiny synthetic fixture (rows written for the test, never rows of the IFBench test file); the 72 MB real parquet only through an opt-in converter script that imports `pyarrow` lazily |
| Generator, scorer, access policy, score session | a loopback fake endpoint and `FakeGenerator`; `FakeModelServer` |
| Metering proxy | a loopback fake upstream with streaming and non-streaming `usage`; budget crossing returns 429 |
| Control API and researcher tool | loopback server, real tar, real sha256 |
| Rental manager, watchdog, money rule, ownership | `FakeProvider` from the GPU notes with an injected clock and the failure injectors listed there |
| Thunder provider parsing | recorded JSON from the public GET endpoints as fixtures; create and delete not exercised |
| Remote helpers | a fake `ssh`/`rsync` script on `PATH` backed by a directory |
| CORAL runtime: config rendering, command lines, harvest parsers | fixtures copied from the CORAL probe (attempt JSON, opencode event logs); no CORAL process |
| Trial runner | all of the above fakes, whole sequence, including budget stops and a crashed runtime |
| Outer loop | scripted proposer, fake trial executor, real stats and packs |
| End to end | one fake trial, one candidate round, one decision, one final comparison, one report, through `cli.main(argv, world=build_fake_world(tmp))`; `tests/support/lab_world.py` builds the world from the fakes of owners 1 to 5 |

**Only on a GPU** (UNVERIFIED until the first run; the first-run checklist is at the end of [GPU executor and watchdog](#gpu-executor-and-watchdog)):

- The Axolotl 0.19.0 QLoRA recipe on one 80 GB GPU: whether the text-only config trains without `processor_type`, label masking with `eot_tokens`, peak memory, step time, adapter dtype and size on disk, flash-attention and cut-cross-entropy builds.
- vLLM 0.32.0 serving Glimmer with `--enable-lora --max-lora-rank 32`, adapter name mapping, dynamic loading, parity with Transformers plus PEFT, truncation at `--max-model-len 8300`, throughput, start time.
- The measured load, generation, training, export and scoring times and therefore the cost of a trial.
- Adapter reload through PEFT (`dtype=torch.bfloat16`, `bnb_4bit_quant_storage=torch.bfloat16`).
- NF4-trained adapter on a bf16 base: size of the mismatch.
- Thunder REST create, delete, ownership tag in `sshPublicKeys`, whether delete accepts the uuid, and a real kill of the coordinator after create.
- Stock Glimmer headroom against Meta's 77.0, the noise floor and the real GPU cost (the first scorer run).

**Only with the Meta key:**

- Muse Spark inside opencode through CORAL's LiteLLM gateway and the metering proxy: tool calls with `tool_choice: auto`, the nine opencode tools, replay of assistant tool-call messages, `stream_options`, `max_tokens`, restart behaviour, token use of the default heartbeats.
- The proxy forwarding to `https://api.meta.ai/v1` and the `usage` chunk of real streams.
- Muse Wonder over Chat Completions, strict `json_schema` keyword support, the real `max_completion_tokens` ceiling, and whether the fallback never triggers (the endpoint uses `fallback_model=None`).
- Rate limits with several concurrent researchers (3,000 RPM and 4,000,000 TPM per team), and a re-check of prices (last checked 2026-10-26).

**Only with the hosts:** CORAL install on the researcher host, the opencode PWD shim, XDG isolation, the per-trial users, the SSH reverse tunnel, `verify-isolation`, the lint YAML-to-JSON dump, the verifier parity check.

**Only with a human:** the terms gate outcome, GPU funding and the dollar hard stop, the rule's form and numbers after the v0 spread, the choice of held-out families.

## CORAL integration plan

Facts below are NOTE from the CORAL reader (revision `0123dfb939b35228cf2c1fde224cd0561e727408`, probed on ax41 against a scripted stub, never against Meta).

**What the pack is.** A git directory with CORAL's `task.yaml` (the text levers), `seed/` (files copied into the repository root), optional `skills/`, `roles/`, and `notes/seed.md` (seeded notes; part of the treatment and hashed). The outer loop edits only these paths (allowlist in `packs.py`). It does not contain the provider config, the gateway, the grader or the tools.

**What the runtime owns and applies at launch** (decision D9). It writes these into the trial directory on the researcher host and overrides them on the command line, so no pack edit can change them:

| Item | How |
|---|---|
| Provider | `opencode.json` with provider id `muse` (not `meta`, which collides with a built-in), npm `@ai-sdk/openai-compatible`, model `muse-spark-1.3`, `limit.output` 16000, `baseURL` `http://127.0.0.1:<gateway port>/v1`; mounted through `agents.runtime_options.mounts` so it overrides any seed copy |
| Upstream route | `litellm_config.yaml`: model `openai/muse-spark-1.3`, `api_base` = the loopback metering-proxy port (forwarded by the SSH tunnel), `api_key` = the trial token, `extra_body.reasoning_effort` (top-level `reasoning_effort` is dropped by `drop_params`) |
| Runtime and model | dotlist overrides `agents.runtime=opencode`, `agents.model=muse/muse-spark-1.3` |
| Session and paths | `run.session=local`, absolute `workspace.results_dir`, `workspace.repo_path` (nonexistent, so a fresh repo is made), `workspace.run_dir` (predictable) |
| Ports | `agents.gateway.port=<unique per trial>` and the matching `baseURL`; reuse of a port fails for about 60 s after a run (TIME_WAIT) |
| Grader | `grader.entrypoint`, `grader.setup`, `grader.timeout` pointing at the runtime-owned dev grader package under `assets/coral_grader/` |
| Stop | `run.stop.max_real_attempts=<N>` from the trial parameters |
| Environment | the PWD shim (`assets/shims/opencode`) first on `PATH` (opencode 1.19.3 trusts a stale `$PWD`; without it the first trial died with `Model not found`); `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME`, `XDG_STATE_HOME` inside the trial directory so the user's global opencode config does not leak in |
| Tool | `researcher_tool.py` as `glimmer`, mounted outside the git-tracked tree |

**The researcher loop.** Researchers edit code and call `glimmer submit` to run a job on a rented GPU. `coral eval` is the dev-score reader: the dev grader reads `submission.json` from the committed tree, asks the control API for that adapter's dev aggregate, and returns it as the score with a one-line explanation. It never contains selection or test numbers (everything a grader returns is written to the public attempt JSON and shown to the agent). Evals serialise per run (`max_workers: 1`, `max_pending_per_agent: 1`).

**Trace capture** (NOTE sources, best first): `logs/<agent>.<n>.log` (opencode event stream with per-step tokens), `gateway/requests.jsonl` (full request bodies; large; tool-call deltas are missing for streamed replies; per-agent attribution is lost for two or more agents because the static key is shared, so per-agent tokens come from the opencode logs), `attempts/*.json`, notes and skills. The runner's meter, not these logs, is the budget authority. Reasoning content is redacted for external callers, so traces carry no chain of thought.

**Isolation facts that shape the design.** The agent can read `.coral/private/` by three routes, read the environment of `coral start`, and edit the grader source (CORAL notes section 5). The researcher host therefore holds nothing worth stealing: no secret, no selection or test data. `agents.isolate_user` / Docker / `agents.sandbox` are not relied on and not planned (they need root, Docker image work or packages that are absent on ax41).

**Gateway exposure.** The embedded gateway binds `0.0.0.0` and is unauthenticated (CODE, not tested from another host). It forwards only to the loopback tunnel with a trial token whose budget is capped, so the worst case is a stranger spending the remainder of one trial's budget. Mitigate with a host firewall rule or a one-line patch to bind `127.0.0.1`.

**Heartbeats.** The default `reflect` heartbeat fires after every eval and interrupts the agent; defaults are tuned for Claude. v0 sets large `every` values for all but one heartbeat. Whether Spark behaves well in the restart loop (exit without a tool call, restart with `--continue`, burst breaker) is UNVERIFIED; one live smoke call with a trivial task is the first thing to run when the key is available.

**Pack CI.** Two steps, so the check is testable offline. (1) `assets/lint_pack.py --dump` runs inside CORAL's own virtual environment on the researcher host: `CoralConfig.from_yaml` loads the candidate `task.yaml` and the script prints it as JSON. (2) `packs.compare_task_keys(parent_json, candidate_json, lint_ranges)` compares every non-editable key with the parent pack in pure Python; its tests use JSON fixtures. `coral validate` (3.5 s in the probe) runs in the trial runner's pre-flight, before any GPU is rented. The statement 'the v0 template lints clean' is proven offline by the JSON fixture of v0 and on the host by the lead-run dump.

## IFBench scorer plan

- **Unchanged upstream verifiers for the test split.** `verifiers.IfbenchUpstream` puts the pinned IFBench checkout on `sys.path`, builds `InputExample` with `None` kwargs filtered, and calls `evaluation_lib.test_instruction_following_strict` and `_loose` per prompt with a one-entry `{prompt: response}` dictionary (this removes prompt-text collisions and leaves upstream code untouched). Strict must run first or loose crashes on `None` kwargs; the adapter filters them itself. The smoke test reproduces the unmodified `run_eval.py`: strict 0.28 and loose 0.3133 on the sample responses, identical flags on 300 of 300 rows.
- **Which file.** The current 300-row file at the pinned commit is the primary test set. The legacy 294-row file had a corrupted row 268 and lacked 269 to 274; a legacy-comparable subset (current file minus keys 268 to 274, 293 rows) is reported alongside. Which file Meta used for 77.0 is UNVERIFIED. All arms use identical rows, so the choice does not affect the comparison between arms.
- **Headline.** Prompt-level strict, with loose and instruction-level also recorded. The IFBench README generally reports loose and the paper's main tables strict (the reader could not settle this; UNVERIFIED which Meta used). Strict cannot be raised by the line-dropping heuristics.
- **Dev and selection** use IFEvalG, the code Ai2 trains with, for all 54 IF-RLVR ids; IFBench's registry has only 25 of them. The 58 test ids are disjoint from the 54 training ids by id, but the five `copy` ids (including `new:copy_span_idx`) share a template with the test `repeat:*` ids, and `keywords:keyword_specific_position`, `count:count_unique` and `keywords:palindrome` have test twins, so "out of distribution" holds by id, not by concept. The reader's default held-out family set is `copy` and `firstlast`, removed from the evolution assignments; a stricter set also holds out `keywords_new` and `counting`. The choice is made before any candidate result and recorded in the manifest.
- **Selection is in-distribution.** Dev and selection measure IFEvalG on training-style constraints; only the test split measures IFBench's 58 out-of-distribution constraints. Do not build selection by attaching OOD constraints to IF-RLVR prompts (it would leak the test constraint types into the outer-loop signal).
- **Floor for headroom.** Constant answers such as `Yes.` score a mean fraction reward of about 0.30 and about 11 % all-constraints-pass on the reader's dev split. `rewards.py` reports `error_counts()` and the floor so researchers and the report can see degenerate gains.
- **Truncation and reasoning.** A generation with `finish_reason == length` counts as a failure in every arm and is counted. The scorer reads only `message.content` (the final answer); vLLM's reasoning parser keeps reasoning separate. An empty content with `finish_reason: stop` after a tight `max_tokens` is a known Glimmer failure; `max_tokens` is set generously and recorded in `GenerationParams`.
- **Reproducibility pins** go in `ScoreRecord.verifier_pins`: IFBench commit, the three IFEvalG file hashes, the prompt file hash, `strip_thinking` flag, and NLTK data versions. `langdetect` is non-deterministic unless seeded; the scorer leaves upstream behaviour untouched and the reward module seeds it.
- **Open data questions** (UNVERIFIED): whether Glimmer was trained on IF-RLVR or IFBench-like data, and the provenance of the IF-RLVR user prompts beyond the dataset card's sentence on third-party model outputs. Do not claim Glimmer never saw the IFBench questions.
- **Fixtures are synthetic.** `tests/fixtures/lab/ifbench_mini/` and every other committed fixture holds rows written for the test, never rows copied from `IFBench_test.jsonl` or `sample_output.jsonl`; this repository is pushed to GitHub every session. A CI test (`tests/test_lab_fixtures_clean.py`, owner 1) walks the repository and fails if any file's sha256 equals the pinned hash of either file (hashes live in `scripts/lab_pins.json`).
- **Split secrecy and fresh items.** See [Closing the leaks](#closing-the-leaks): the split salt is a secret under `private/`, selection and held-out items are fresh constraint instantiations, and every dataset a job trains on is scanned before any selection score.

## Glimmer trainer and serving plan

All NOTE from the Glimmer reader, none run on a GPU.

**Training.** Axolotl 0.19.0, QLoRA, one 80 GB GPU, in its own virtual environment (Python 3.12, torch 2.12.1, `axolotl==0.19.0`, flash-attn 2.8.3, cut-cross-entropy at the pinned revision). Base `meta-models/Muse-Glimmer-30B` at revision `a4e59da52a7bc87ae7251dd5545c0dd437c44b68` (identical files to HEAD). The harness config `hostkit/glimmer-qlora-text.yaml` is the reader's derived config: the upstream example plus `eot_tokens: ["<|eot|>", "<|eom|>"]` (the 0.19.0 example omits them, so the turn terminator would not be trained), `train_on_inputs: false`, `roles_to_train: [assistant]`, `train_on_eos: turn`, W&B off, an explicit system message in every example (the template otherwise injects the current date). Fallback if the text-only path misbehaves: restore `processor_type: AutoProcessor`, `skip_prepare_dataset: true`, `remove_unused_columns: false`, the path Meta actually measured. The training method is the researcher's: the trainer provides the GPU and this stack, plus helper commands for the recipe (`glimmer-axolotl-train <config>`, `glimmer-vllm-up` for base-model generation in a rejection-sampling recipe), and does not decide the method.

**Job contract on the rented GPU** (`hostkit/run_job.sh`): the recipe tarball is unpacked at `/work/recipe`; training data and dev data are at `/work/data` (public data only); the recipe's entrypoint runs with `GLIMMER_BASE`, the config template path and the helper commands on `PATH`; it must leave the final adapter in `/work/out/adapter` (`adapter_config.json` and `adapter_model.safetensors`; other files such as `checkpoint-*` are ignored). The script writes `timings.json` (load, generation, training, export) and a `trial.done` marker. A cap shorter than the job ends it by deleting the rental, not by asking the recipe to stop.

**Adapter.** About 767 MB (r=32, fp32 PEFT safetensors; 191,692,800 trainable parameters) if Axolotl saves fp32; check the real dtype and size after the first run. `AdapterRef.sha256` is the hash of the sorted listing of `adapter_config.json` and `adapter_model.safetensors` (hash, size, name). Before the job is called successful the host reloads the adapter through Transformers plus PEFT with `dtype=torch.bfloat16` and `bnb_4bit_quant_storage=torch.bfloat16`. The adapter is fetched to the control host and re-verified after the copy.

**Serving for the scorer and for dev scores.** vLLM 0.32.0 in a second virtual environment (torch 2.13.0): `vllm serve meta-models/Muse-Glimmer-30B --revision a4e59da5... --served-model-name muse-glimmer --gpu-memory-utilization 0.9 --max-model-len 8300 --enforce-eager --tool-call-parser muse_glimmer --reasoning-parser muse_glimmer --generation-config auto --enable-lora --max-loras 1 --max-lora-rank 32 --lora-modules <sha-derived-name>=/adapters/<id>`. bf16 base for all three arms; one server serves the base and the adapter by model name (DECISION: removes server-start variance between arms). Training (about 40 to 42 GiB estimated on one GPU) and serving (about 72 GB) run in sequence, never together. The adapter name sent in requests comes from `ServeEndpoint.model_for(adapter)`.

- **Dev scoring on the training rental** needs a tunnel: the trainer forwards the rental's vLLM port to a control-host loopback port (`gpu_remote.forward_port`), `ServeEndpoint.base_url` is that loopback URL, and `Generator` and the verifiers run on the control host. `rewards.py` (the researchers' training-time reward) runs on the rental and needs IFEvalG there: `hostkit/stage.sh` uploads exactly the three pinned IFEvalG files from the public train side to `/opt/lab/ifevalg` (`IFEVALG_PATH`) and `rewards.py` to `/opt/lab/rewards.py`, verifies their sha256 pins, and never uploads selection, held-out or test data.
- **Scoring-host trust.** A rental on which a researcher recipe ran is never reused for selection scoring and never shares a writable image or volume with the scoring rental: selection scoring boots a fresh, immutable image and verifies the base weights by sha256 (the Glimmer notes' pre-baked persistent weights volume must be read-only for scoring). Dev numbers produced on a recipe rental are researcher-influenced and labelled so.
- vLLM v0.29.0 or later is required for the LoRA startup fix (PR 53513); v0.28.0 and the `muse-glimmer` Docker tag predate it. Nobody has published a Glimmer plus LoRA vLLM run: **UNVERIFIED end to end**.
- **Fallback:** merge the adapter into the bf16 base (`PeftModel.merge_and_unload`, about 59.6 GB written) and serve the merged directory; it costs a fresh server start per trial, and the saved parameter names must start with `model.language_model.`.
- Do not use NVFP4 or any quantised checkpoint with LoRA, speculative decoding, the SGLang runtime for adapters (no evidence of support), or the K-Quant GGUF variants for scoring.
- Recommended sampling: temperature 1.0, top_p 0.95, top_k 64 (`GenerationParams.recommended`); greedy is not reproducible, so repeated seeds are mandatory.

**Measurements the first run must record** (they set `budget.json`): weights download, NF4 load, steps per second and peak memory at the real token lengths, adapter save time and size, adapter reload, vLLM start time, generation throughput, truncation rate at the chosen `max_tokens`, dev-score time. The reader's estimate for one train-and-score trial is 1.5 to 4 GPU-hours on one 80 GB card, which is **INFERRED**, not measured.

**Which GPU.** The cheapest supported option by the GPU reader is one Thunder A100 80GB (about $1.31 per hour with 12 vCPU and 300 GB disk). H100 is unavailable on Thunder at the time of reading, and the reader found no H100 number for Glimmer. Meta's "verified on one H100" is a memory fact, not an H100-only kernel dependency, but flash-attn 2.8.3 for sm_80 with torch 2.12 and CUDA 13 is UNVERIFIED. The GPU shape is a parameter (`Shape`), not hard-coded.

## GPU executor and watchdog

Adapt the evolved K004 `lifecycle.py` and `vast.py` from the kge-train project (GPU notes section 3) rather than the Qwen experiment; split it into the modules above. Nothing is copied wholesale: the hard-coded cap, the Qwen setup and the `/proc` scans are replaced as listed in the GPU notes section 3.2. Reused code from other projects is labelled in the provenance inventory.

**Provider.** `ThunderProvider` uses the REST API (`POST /instances/create` with our own `public_key`, `GET /instances/list`, `POST /instances/{id}/delete`), because the public key comes back in `sshPublicKeys` and works as an unforgeable ownership tag, and the CLI cannot name or tag instances. `tnr` stays as a fallback implementing the same protocol. A `VastProvider` is not planned: third-party hosts are a leak risk for scoring data and the time is better spent elsewhere (DECISION).

**Rental manager rules** (tested against the fake provider):

- State machine `creating -> live -> deleted`, plus `not_created` and `uuid_unknown`. The rental row is written before `provider.create`. A second create is blocked while any rental is unsettled. A rental is `deleted` only after `exists()` returns `False`.
- Money rule before every create or rearm: billed minutes of all rentals (rounded up) plus what other live rentals can still bill up to their caps plus the new cap plus a reserve must not exceed `hard_stop_usd`, read from the rental ledger. There is no default dollar value in code; the owner sets it with `init --hard-stop-usd N --why ...`. The refusal is `RentalRefused` and creates nothing.
- Ownership: every rental carries `trial_id`, `item` and an owner tag; delete re-resolves the provider's current row for the handle, checks the tag, and only then sends the call. `OwnershipError` for a foreign row. Never touch an instance that is not in our ledger: the Thunder account is shared with the Mac, ax41 and other projects (a foreign A100 was running when the GPU reader looked).
- Index race: Thunder delete addresses an instance by list position. After each delete the manager re-lists and asserts that exactly our handle vanished, and logs any other disappearance. Whether delete accepts the uuid is UNVERIFIED; it may be tested only on our own instance, immediately before a planned delete.
- `list_instances` raises `ProviderUnavailable` on any error; it never returns an empty list on failure, because an empty list reads as "absent".
- Create is never retried automatically. `CreateAmbiguous` leaves `uuid_unknown`; recovery matches the owner tag and shape in a unique new row; a human confirms "none" only after reading the provider console.

**Watchdog** (detached, `start_new_session=True`, one per rental): spawned before the provider is called (the GPU notes' case C), sleeps to `cap_epoch`, deletes in a loop with 60 s sleeps until absence is confirmed, refuses an epoch later than the ledger's cap, and writes `watchdog.json` with its pid within 15 s or the create fails loudly. A `reap` command, run every five minutes from a second always-on host with that host's own provider token, deletes any rental past its cap by more than ten minutes whose tag matches (case D). **`reap` is required before the first paid rental and is not on the cut list**: Thunder has no TTL and shutdown is not a billing stop, so one orphaned A100 (about $1.31 per hour) over a weekend is about $60, plausibly most of the pilot budget. If no second host or second token exists, the substitute is a documented operator procedure with an alert plus the shortest workable lease caps (2 hours at the first run); the choice is an open decision. Liveness of watchdogs is checked by an advisory lock the watchdog holds, not by scanning `/proc` (Linux only).

**Hard stop at the budget** is the tick in [Budget enforcement](#budget-enforcement); the watchdog is the backstop when the coordinator is dead.

**Pull before delete.** The trainer fetches the adapter and verifies sha256 before deleting, with a pull deadline of cap minus a margin; if the pull cannot be verified before the cap, the watchdog still deletes at the cap and the data is lost, not the money.

**Credentials.** Thunder token on the control host only (`~/.thunder` mode 0700 owned by `lab`), the rented GPU never receives it. The GPU notes found that ax41 and the Mac share one Thunder account and that ax41 holds a token under the main user: that file is removed or made unreadable to every lab user on the researcher host, and `verify-isolation` tries to read it. `THUNDER_API_KEY`, the Meta key and each trial token are registered with `secrets.register_env` before the first ledger append. A Hugging Face token, if the download needs one (UNVERIFIED), goes in over ssh stdin or a one-shot environment variable, never written into a session directory. Per-rental ssh keys are generated per session and deleted at teardown.

**First-run checklist** (from the GPU notes; costs counted by the rental ledger): fake-provider tests with no money; `init --hard-stop-usd`; check availability and cost; create a 2-hour-cap rental and confirm the watchdog pid, ssh config and quoted rate; record `nvidia-smi`, disk and RAM; install the Axolotl and vLLM environments and time the download; time and record memory for vLLM load, 100-prompt generation, 20 QLoRA steps, adapter save and reload, and scoring of the reloaded adapter; delete and confirm; kill the coordinator right after a 15-minute-cap create (the whole process tree of the control host) to check that the watchdog survives and deletes, and check from the second host that `reap` sees and deletes an expired rental.

## Outer-loop controller and redaction rules

**Reuse and gaps.** WeCo published a method and numbers, not code; the proposal-generating optimizer is hosted and is not used (outer-loop notes section 0). Reused as-is: `MetaChatClient`, `MeteredClient` and `BudgetGate` for Wonder with `fallback_model=None` and `resp.model` plus `fallback_used` recorded in every candidate (a silent fallback to Spark would mislabel a Spark proposal as Wonder), `chat_json`, `fence`, `secrets`. `parse_json_object` (about 30 lines) is copied into `proposer.py` with a test that its output equals `nightshift.agents.prompts.parse_json_object` on a fixture set: the reuse notes recommend copying because importing `agents` couples the lab to modules `contracts.py` must not depend on, and the earlier 'import cost 0.06 s' figure had no source (re-measured by a critic at about 0.04 s for `agents.prompts` and about 0.2 s for `nightshift.grading`, so the coupling argument, not the cost, decides). Taken as prompt text, MIT: the "single actionable, atomic improvement" wording from aideml. Not taken: aideml's code, `weco-cli`, OpenEvolve's diff parser (the strict apply is about 30 lines and must reject unmatched blocks, which OpenEvolve silently skips). To write: the controller, the redaction layer, pack versioning, the rule evaluation (in `stats`), trace summaries, cost centres.

**Controller (`outer_loop.py`).**

```text
refuse unless: rule frozen (load_frozen_all), terms gate recorded, v0 spread trials exist, lab-wide dollar room for the worst case of this round
incumbent = incumbent_version(state)
repeat while candidates_in_ledger < rule.max_candidates (and stop_after_rejections, outer budget, wall cutoff):
    batch = []
    for k in 1..rule.screen_batch_size:                          # never more than 5 per round
        facts   = collect_facts(state, incumbent)                # typed, allowlisted
        ctx     = build_context(facts)                           # redaction, one place; assert_no_forbidden(prompt, forbidden_values(...))
        prop    = proposer.propose(ctx)                          # one Wonder call; ProposalInvalid -> record INVALID, continue
        cand    = packs.apply(prop, incumbent, version_label)    # strict; allowlist; no GPU spent on failure
        lint    = packs.lint(cand)                               # key comparison against the parent
        append  lab.candidate_proposed (CandidateRecord, diff, reason, lint)   # counts toward max_candidates from here on
        if lint failed: decide SCREEN/INVALID; continue
        batch.append(cand)
    run screen_trials_per_candidate trials per assignment for each, interleaved with incumbent trials; same runner, same budget.json
    deltas = decisions.screen(rule, incumbent_scores, candidate_scores)
    best   = argmax delta where delta > 0 (ties: smaller id); every other candidate gets a SCREEN Decision (REJECTED with screen_delta, or INVALID)
    if best: run confirm_trials_per_candidate FRESH trials of best and confirm_trials_per_incumbent FRESH incumbent trials, interleaved
             nums = decisions.decide(rule, incumbent_confirm, best_confirm)      # the controller decides, never the proposer
             append lab.candidate_decided (CONFIRM Decision with sigma_used, df, cost_breakdown)
             if accepted: packs.accept(best); incumbent = best
    anchor the master ledger head (cli anchor) after each round
```

The controller counts candidates from the ledger (invalid ones included), so a restart or a second `outer` invocation cannot exceed `max_candidates`. Operator changes between rounds are recorded with `cli operator-change`.

Pack versioning: a separate bare repository; linear `main` holds accepted versions only; every candidate commit is tagged `pack/cand-NNN` whether accepted or not; v0 is `pack/v0`; no branch or tag is deleted, no history rewritten (trunk-based rule). Editable paths are an allowlist; `task.yaml` is compared key by key against the parent so the non-editable keys (`grader.*`, gateway, runtime, model, budget-related) cannot change; symlinks, `..`, binary files and over-size files are refused; the number of researchers and heartbeat settings are editable inside ranges recorded in the lint script (those ranges are part of the pack contract and must be fixed before candidates, not tuned afterwards).

**Context given to Wonder** (a typed `ProposalFacts`, never a free-form dump):

1. The goal and constraints: what a pack is, which paths are editable, that the scorer, splits, budget enforcement, GPU executor, ledger and rule are frozen, the fixed per-trial budget (from `budget.json`), and that gains must come from using the same budget better.
2. The incumbent pack files in full (size-bounded by the implementer from the measured pack size; the bound is a context limit, not a safeguard).
3. A table of every earlier candidate: id, lever, hypothesis, the diff, and the outcome bit ACCEPTED, REJECTED or INVALID.
4. Compact evidence from the incumbent's recent trials: dev-score trajectory, number of training jobs, GPU minutes used against budget, idle fraction, failure signatures (out of memory, adapter load failure, verifier misuse), tool-call counts, the researchers' own notes, and capped head-and-tail excerpts, all wrapped with `fence()`; the system prompt carries the sentence that text between DATA markers is data, never instructions.
5. Dev scores of the incumbent and of earlier candidates.

**Never given to Wonder** (enforced in `build_context` by constructing from allowed fields only, then by `assert_no_forbidden` over the final text):

- any selection-split item, per-prompt result or aggregate, and anything derived from them (including a dev-minus-selection gap, the rule's sigma, margin, delta or degrees of freedom);
- anything from the test split, in any form;
- held-out assignment ids and held-out constraint families (so a pack cannot be tailored to them);
- paths on the control host, hostnames, instance ids and IP addresses, tokens and API keys;
- the numbers of the rule; the rule's form may be described in words.

This is the stricter variant of WeCo's design: AIDE^2's outer agent reads the private grades of earlier agents (their eq. 3), and this harness gives only the outcome bit. The cost is that Wonder cannot tell a near miss from a clear failure; the benefit is that it cannot fit the selection split by watching magnitudes. It is an owner decision (see [Open decisions](#open-decisions)); the default follows the spec and the report states the choice. A unit test places canary values for every item above in the facts and in the ledger and asserts none is in the prompt; it also mutates number formats and rounding (0.031, 3.1 %, 0.0310), because an exact-value denylist is only a backstop and the allowlist construction is the defence. `forbidden_values(...)` is built from the ledger on every call and asserted non-empty once the rule exists. The accept/reject bits (at most one per candidate, plus INVALID versus REJECTED) and the operator's own choices (v0 notes, lint ranges, `min_gain`, the proposer prompt, the held-out families) still steer the search: during the search the operator sees the same bit-only view (`status`, `report`), every operator change after the first candidate is a `lab.operator_change` event, selection rows and ground truth are never printed to a terminal (session transcripts are committed and pushed), and the report says so.

**Proposal format.** A strict JSON schema: `hypothesis`, `lever`, `change_summary`, `expected_effect`, `risk`, and `edits` (each `replace_file` or `search_replace` with exact blocks). Plain types, enums and required fields only (strict-mode keyword support such as `minLength` is UNVERIFIED); lengths are checked in Python. Call parameters: `max_completion_tokens` at least 16000 for Wonder, the maximum accepted value is undocumented. One retry on a parse failure; then the candidate is INVALID and costs only Wonder tokens.

**First candidates** (priors, from the paper's winning revisions; not established for post-training agents): bounded notes and context plus failure-signature memory; a plateau-triggered pivot or fork-best strategy in the heartbeat prompts; strategy roles across researchers; a robustness instruction plus an internal held-out check before declaring a checkpoint final. Do not seed the changes the paper found harmful (islands, tournaments, ensembles at submission, MCTS-style propagation). With a handful of affordable candidates, each must be one atomic, high-prior lever.

## Interface appendix

Owners 2, 4 and 5 build against these interfaces; owner 4 cannot write a working v0 pack without them. Anything not listed is the owner's choice and stays inside the owner's files. Where the CORAL grader signature is not in this document it is **UNVERIFIED** here: owner 5 reads `coral-probe-evidence/` and publishes the three grader values in `nightshift/lab/assets/coral_grader/ENTRYPOINT.json` (keys `entrypoint`, `setup`, `timeout`) as the first deliverable; owner 4 copies exactly those values into the v0 `task.yaml`, and the pack lint asserts equality.

**`glimmer` (researcher tool, owner 5; standard library only; runs on the researcher host).** It reads the control URL from `GLIMMER_CONTROL` (`http://127.0.0.1:<port>/control`) and the token from the file `./.glimmer_token`. Every subcommand prints one JSON object on stdout. Exit codes: 0 ok, 1 refused (budget or validation), 2 usage error, 3 control API unreachable, 4 `wait` timed out.

| Command | Output (stdout JSON) |
|---|---|
| `glimmer budget` | `{"researcher_tokens_left": int, "gpu_minutes_left": float, "dollars_left": float, "wall_seconds_left": float}` |
| `glimmer submit <recipe_dir> [--entrypoint "bash run.sh"] [--max-gpu-minutes N]` | `{"job_id": str, "status": "queued" or "refused", "reason": str (when refused)}`; packs the directory into a tar.gz under the rules in [File formats](#file-formats) |
| `glimmer status [<job_id>]` | `{"job_id", "state": "queued", "running", "ok", "failed", "timeout" or "budget_exhausted", "adapter_id" (when ok), "dev_score" (when ok), "gpu_minutes_used", "failure" (when not ok)}` |
| `glimmer wait <job_id> [--timeout S]` | the `status` object once terminal; exit 0 when the state is `ok`, 1 otherwise, 4 on timeout |
| `glimmer dev-score <adapter_id>` | `{"adapter_id", "dev_score": float, "n_prompts": int, "metric": "prompt_all"}`: the control-host dev score, the only score a researcher ever sees |
| `glimmer pointer <adapter_id>` | writes `submission.json` into the current directory and prints it; the file must be committed, because the grader reads the committed tree |

**`submission.json`** is exactly a `Submission`: `{"adapter_id": str, "recipe_sha256": 64 hex, "commit": 40 hex (the commit that records the recipe), "dev_score": number or null}`. The `dev_score` is ignored and overwritten by the runner.

**Job contract on the rental (`hostkit/run_job.sh`, owner 2).** Paths: `/work/recipe` (the unpacked tarball), `/work/data/train.jsonl` and `/work/data/dev.jsonl` (public data only), `/work/out/adapter/` (must end with `adapter_config.json` and `adapter_model.safetensors`; other files are ignored), `/work/out/train_data.jsonl` (the processed training dataset, written by the helper commands), `/work/out/timings.json`, `/work/out/trial.done`. Environment: `GLIMMER_BASE` (base model directory), `GLIMMER_CONFIG_TEMPLATE`, `GLIMMER_OUT=/work/out`, `IFEVALG_PATH=/opt/lab/ifevalg`, `REWARDS_PY=/opt/lab/rewards.py`, and `/opt/lab/bin` on `PATH`. Helper commands, the ONLY training entry points (they write the processed dataset into `/work/out/train_data.jsonl`; a job that never calls one has no `train_data` and fails): `glimmer-axolotl-train <config.yaml> --data <jsonl>` and `glimmer-vllm-up [--port P]` (base-model generation for rejection sampling). The entrypoint exits 0 and leaves an adapter for the job to be `ok`.

**Grader entrypoint (owner 5, `assets/coral_grader/`).** The runtime overrides `grader.entrypoint`, `grader.setup` and `grader.timeout` on the command line with the values in `ENTRYPOINT.json`. The grader reads `submission.json` from the committed tree, calls `GET /control/dev-score?adapter_id=<id>` on the control API, and returns the dev score with a one-line explanation. It never contains selection or test numbers (everything a grader returns is written to the public attempt JSON).

**Reference method and v0 seed recipe: assigned.** The reference method (generate with the base model, filter with `rewards.py`, SFT, reload) is **owner 2's** `nightshift/lab/hostkit/recipes/reference_rft/` (`run.sh` plus config) because it is a recipe for the rental; **owner 4** integrates it as `seed/recipe/` in `assets/pack_v0/` and writes `task.yaml` and `notes/seed.md` against the commands above. Due: owner 2's recipe at the end of wave A; owner 4's integration before the spread plan is declared. Fallback if it is late: v0 ships the plain Axolotl SFT on the public train split (no generation), recorded as an operator decision. The lead decides whether the measured reference method or the fallback is v0, before `plan-spread`.

## File formats

Formats that cross owners. Anything else a file contains is its owner's choice.

- **Data rows** (`train.jsonl`, `dev.jsonl`, `selection.jsonl`): one JSON object per line with the required keys `key` (stable row id), `prompt`, `ground_truth` (the JSON string open-instruct's verifier reads), `instruction_ids` (list of constraint ids) and `family`; other keys pass through. `train` and `dev` rows are public; `selection` rows are private. `IFBench_test.jsonl` keeps the upstream format (`key`, `prompt`, `instruction_id_list`, `kwargs`).
- **`per_prompt.jsonl`** (flags only, no response text, no ground truth), one row per (prompt, seed). Dev and selection: `{"prompt_key", "seed", "passed": bool, "fraction": float, "constraints": [{"id", "passed"}], "truncated": bool, "error": bool}`. Test: `{"prompt_key", "seed", "strict_prompt": bool, "loose_prompt": bool, "strict_instructions": [bool], "loose_instructions": [bool], "truncated": bool, "error": bool}`. `prompt_key` is the row `key`, which is what stats uses for the paired bootstrap, McNemar and the legacy-293 subset.
- **`gen_seed<k>.jsonl`** (private): `{"prompt_key", "text", "finish_reason", "completion_tokens", "error"}`.
- **Recipe tarball**, made by `glimmer submit`, re-validated by the control API, re-checked by `run_job.sh` before extraction: a gzip tar; entries are regular files and directories only; relative paths with no `..`, no leading `/`, no `.git`; no symbolic links, hard links or devices; at most 5,000 entries, 50 MB compressed and 200 MB unpacked (UNVERIFIED sizing; owner 5 may lower these, raising needs the lead); the entrypoint is a path inside it; stored as `trials/<tid>/recipe/<sha256>.tar.gz`, with the sha256 of the bytes in `TrainJob.recipe`.
- **`timings.json`**: an object whose keys are a subset of `download_s, load_s, generation_s, train_s, export_s, eval_s` with numbers of at least 0; the trainer maps any recipe-side names to these before building `TrainResult.timings`.
- **Metric names**: the `METRIC_*` constants; scorer, stats and report use no literals.
- **Hostkit staging** (owner 2, `hostkit/stage.sh`): `/opt/lab/rewards.py` (owner 1 owns the content and exposes `rewards.REQUIRED_FILES`), `/opt/lab/ifevalg/{instructions.py, instructions_registry.py, instructions_util.py}` pinned by sha256 and taken from the public train side only, `/opt/lab/bin/glimmer-*`. The script verifies the pins before the job starts.
- **`lab.json`** is a `LabConfig` record (hosts, the researcher user, base model, GPU shape, pins for CORAL, opencode, IFBench and open-instruct, the names of the secret environment variables, the upstream base URL, `lab_budget_usd`, `affordable_trials`).

## Ids, clocks, hashes and imports

- **Id formats and allocation.** `t-0001` (trials, including final trials `t-...` named in the final plan), `cand-001` (candidates; the candidate id IS the pack version label and the git tag `pack/cand-001`, and `CandidateRecord.version_label == Decision.candidate_version == candidate_id` is enforced), `ad-0001` (adapters), `j-0001` (jobs), `sc-0001` (score requests and records), `f-0001` (final requests), `v0` (the first pack version). Operators name assignments freely (`a-evo-1`, `a-held-1`), never `ifbench_test`. The coordinator allocates every id with `StateReader.next_id(state, prefix)`; it is one process, so there is no race, and after a restart the next id is recomputed from the ledger. Directory creation uses `mkdir(exist_ok=False)` and `Ledger.create` raises on an existing file, which is the only collision guard.
- **Clock.** Every component takes `clock: Clock` (epoch seconds, `time.time` in production, a `FakeClock` in tests) and writes record times with `utc_iso(epoch)`: always `+00:00`, seconds precision. Nobody formats a time by hand.
- **Hashes.** `FrozenFile.file.sha256` is the sha256 of the bytes written (`write_record` returns it); `Rule.sha256()`, `FinalAnalysis.sha256()` and `Decision.rule_sha256` are `record_sha256` of the canonical JSON of the parsed record. The two kinds are never compared with each other. `AdapterRef.sha256` is the hash of the canonical file listing. `comparability_key` is a canonical hash of the comparable settings. `ScoreRecord.adapter_sha256` equals `AdapterRef.sha256`.
- **`scorer_version`** is `scorer_version_of(verifier pins, metric names, SCORER_MODULE_VERSION)`, a string like `sv1-<16 hex>`; `Scorer.scorer_version` exposes it, and `freeze-rule`, `plan-spread` and `trial` read it from the scorer before any score exists.
- **`owner_tag`** is `owner_tag_of_key(public key)`. The provider derives it; `GpuInstance.owner_tag` of our rentals equals it.
- **The dev score** of a trial is `headline_mean()` of its dev `ScoreRecord` (unrounded), and CORAL's attempt JSON carries the `committed_utc` the runner uses to break ties.
- **Imports.** Absolute only (`from nightshift.lab.contracts import ...`); lab code is run as `python -m nightshift.lab ...` or through `unittest`, never as a script path. Every owner runs tests with `PYTHONDONTWRITEBYTECODE=1 python3.13 -m unittest tests.test_lab_<topic>`.

## Closing the leaks

The first critique found that selection and held-out data were hidden only by their location. These are the defences, in order of strength; the report states the residual risk.

1. **Secret split.** The split salt is a random 256-bit value generated at `init` and stored only at `private/split_salt`; the manifest records its sha256 (`split_salt_sha256`) and nothing else. No real seed appears in code or documents (the reader's default seed 1234 is a test fixture only). Selection and held-out base prompts are excluded from every train and dev pool, so a 13-gram hit in training data is a real hit and not a shared base prompt.
2. **Fresh items.** `ifdata` builds selection and held-out items as fresh constraint instantiations: base prompts (which are public) plus constraints sampled by the IFEvalG generators from a salt-derived generator, with ground truth from the generator, so the verbatim strings are not in the public parquet. **UNVERIFIED:** that the IFEvalG generators expose a sampling interface for all 54 constraint ids. Fallback: draw selection and held-out rows from the public dataset with the secret salt, accept the residual risk that a researcher who retrieves the dataset can train on those rows, and rely on items 3 to 5. `AssignmentManifest.item_mode` records which was used and the report prints it.
3. **Contamination scan.** The glimmer helper commands are the only training entry points; they write the processed dataset to `train_data.jsonl` in the job output. The control host scans it, and the recipe tarball, before any selection score: exact prompt match and 13-gram match (`CONTAMINATION_NGRAM`) against the selection, held-out and test prompts (`contamination.py`, owner 1). A hit appends `lab.contamination_scanned` with `passed` False, stops the runtime, marks the trial `CONTAMINATED` (a failed trial under the rule's failed-trial policy) and the adapter is never scored on selection. A recipe that trains on data it did not route through a helper has no `train_data` and its job fails.
4. **Egress.** The researcher host runs behind an HTTP proxy or allowlist that denies `huggingface.co/datasets` and the IFBench GitHub path, and the model weights are pre-staged from the control host so rentals need no Hugging Face access. UNVERIFIED whether rental egress can be restricted; recipes run as root on rentals, so an in-instance firewall is not a boundary, and item 3 is the real defence.
5. **Harvest safety.** Everything a researcher or a rental can write is copied to the control host and read by privileged code only through `read_regular_bytes` and `file_ref` (no symbolic links, hard links or devices; size caps; O_NOFOLLOW under a realpath check). `gpu_remote.fetch` and the runtime's `collect` apply the same rules to rsync, scp and tar. The rental returns only `adapter_config.json` and `adapter_model.safetensors`, by exact name, as regular files. A test plants a symlink to a canary under `private/` at every harvest path and asserts it never reaches a prompt, a ledger or a scoring rental.
6. **Trial isolation.** Each trial runs as its own throwaway unix user (PrivateTmp, a fresh home, CPU, memory and process limits); at the trial's end every process of that user is killed and the home is deleted; no operator Claude session runs as a lab user. `verify-isolation` reads another trial's token, the lab user's files and `~/.thunder`, looks for leftover processes of the previous trial, and compares the workspace and caches with the initial-context listing. UNVERIFIED whether passwordless sudo exists on ax41. Fallback: trials run one at a time in one user, the `TrialRecord.environment` isolation level says so, and the report labels it.
7. **Adapter lineage.** The runner resolves a submission only from adapter ids in this trial's own `lab.gpu_job_finished` events; a `submission.json` that names another trial's adapter is ignored.
8. **Orphan GPUs.** `reap` or its documented substitute is required before the first paid rental (see [GPU executor and watchdog](#gpu-executor-and-watchdog)).
9. **Residual risk** printed in the report: a determined researcher who routes data around the helpers and never trips the scan; the operator and the accept/reject bits as an adaptive channel; dev numbers produced on a recipe rental.

## Work breakdown

Six implementers with disjoint file ownership. Every implementer: Python 3.11-compatible code only, standard library only (heavy third-party code is imported lazily and the test skips if it is absent), `unittest`, tests named `tests/test_lab_<topic>.py`, run with `PYTHONDONTWRITEBYTECODE=1 python3.13 -m unittest tests.test_lab_<topic>` (this Mac's default `python3` is 3.14, outside the CI matrix; 3.11 is not installed here and runs in CI only), absolute imports only. Do not edit `contracts.py` or its test; ask the lead (changes are additive). Do not edit any file another implementer owns, and do not change the existing Nightshift modules. Each implementer copies the scratchpad artefacts named in their row into the repository first (the scratchpad is temporary). Owners 1 to 5 import only `contracts` and their own test support; only owner 6 imports other owners' test support, in `tests/support/lab_world.py`.

| # | Assignment | Owns (create these, and nothing else) | Depends on | Scratchpad inputs to copy first | Done when |
|---|---|---|---|---|---|
| 1 | **Scorer and data** | `nightshift/lab/{verifiers,ifdata,rewards,generator,scorer,score_session,contamination}.py`; `tests/test_lab_{verifiers,ifdata,generator,scorer,score_session,contamination,fixtures_clean}.py`; `tests/support/lab_fake_generator.py`; `tests/fixtures/lab/ifbench_mini/` (synthetic rows only); `scripts/lab_setup_verifiers.sh`, `scripts/lab_pins.json`, `scripts/lab_convert_parquet.py` (opt-in `pyarrow` converter) | contracts only | `scripts/ifbench_adapter/{score_ifbench.py,if_rlvr_reward.py,build_splits.py,smoke_test.py}` | An arm-1-shaped test request scores through `ScoreService` with the fake generator and fake verifier backends and writes `request.json` and `score.json`; the access-policy table is tested row by row (held-out denied in the search phase, dev included; a request whose phase disagrees with the ledger is refused; every final-phase score calls the guard); the contamination scan finds planted exact and 13-gram hits and no hit when base prompts are disjoint; the split builder reads JSONL, is byte-deterministic for a given salt, never contains the test set, and records fresh-instantiation or salted-draw mode; `test_lab_fixtures_clean` passes. The real-verifier smoke parity is a **lead-run acceptance check** (needs `LAB_THIRD_PARTY`), not a CI gate |
| 2 | **Trainer and GPU executor** | `nightshift/lab/{gpu_rentals,gpu_watchdog,gpu_thunder,gpu_remote,model_server,trainer}.py`; `nightshift/lab/hostkit/**` (including `stage.sh`, `run_job.sh`, `recipes/reference_rft/`); `tests/test_lab_{gpu_rentals,gpu_watchdog,gpu_thunder,gpu_remote,trainer,hostkit}.py`; `tests/support/lab_fake_gpu.py`; `tests/fixtures/lab/thunder/` | contracts; `Scorer` and `Generator` protocols by injection | `proto/gpu_provider.py`; K004 `lifecycle.py`, `vast.py`, `finish.sh`, `test_lifecycle.py` under `vendor/kge-004-remote/`; the Thunder OpenAPI JSON | Every hazard test in the GPU notes section 2.4 passes against the fake; `reap` has tests and a runnable entry point; a fake-ssh trainer run returns a verified `TrainResult` with `train_data`, accepts only the two adapter files and tears down on every failure path; `gpu_remote` refuses a planted symlink, hard link, device, escaping path and over-size file; every hostkit script passes `bash -n` and a Python test asserts that every path a script references exists in the hostkit tree (shellcheck is not required); **a real two-hour rental, create to confirmed delete, is done by the lead with the owner's hard stop** |
| 3 | **Report and statistics** | `nightshift/lab/{stats,report,chart}.py`; `tests/test_lab_{stats,report}.py`; `tests/support/lab_state_builder.py` | contracts only (builds `LabState` by hand) | none | `t_quantile` matches `T95` and `T975` at their tabulated df; `decide` matches hand-computed margins for pooled df 8 and larger; INVALID cases (comparability mismatch, counts, variance guard); `screen`; `minimum_detectable_gain`; `compare_final` for `H = 1` and `H >= 2`; the seeded false-accept simulation stays below 0.08 for K = 5 and 10 and for a candidate with twice the sd; a table and an SVG chart render from a synthetic ledger-shaped `LabState`; every label in the Report section prints under its condition; no per-trial held-out or test score prints before `lab.final_revealed` |
| 4 | **Outer loop and packs** | `nightshift/lab/{packs,outer_context,proposer,outer_loop}.py`; `nightshift/lab/assets/pack_v0/**`; `nightshift/lab/assets/lint_pack.py`; `tests/test_lab_{packs,outer_context,proposer,outer_loop}.py`; `tests/support/lab_fake_proposer.py`; `tests/fixtures/lab/pack_json/` | contracts; `TrialExecutor`, `ProposalSource`, `DecisionRule`, `StateReader` protocols (injected fakes in tests); owner 5's `ENTRYPOINT.json`; owner 2's reference recipe (fallback in the Interface appendix) | none (reads outer-loop notes) | Strict apply rejects unmatched blocks and forbidden paths; the lint key comparison over JSON dumps passes for the v0 fixture and fails for a changed non-editable key; a canary-redaction test (including number-format mutations) passes and `forbidden_values` is non-empty once a rule exists; a scripted round runs through the controller with a fake executor to a screen loss, a confirmed accept and a confirmed reject; candidates are counted from the ledger; the real `MetaChatClient` parses a structured reply from `FakeMetaServer` |
| 5 | **Runtime and trial runner** | `nightshift/lab/{runtime_coral,metering_proxy,control_api,researcher_tool,trial}.py`; `nightshift/lab/assets/{coral_grader,shims}/**` (first deliverable: `coral_grader/ENTRYPOINT.json`); `tests/test_lab_{runtime_coral,metering_proxy,control_api,researcher_tool,trial}.py`; `tests/support/lab_fake_runtime.py` (also holds its own minimal trainer, scorer, model-server and ledger doubles); `tests/fixtures/lab/coral/` | contracts; `BudgetMeter`, `Trainer`, `Scorer`, `ModelServer`, `LedgerWriter`, `PackStore`, `StateReader` protocols by injection | `probe-scripts/`, `probe-pack/`, `coral-probe-evidence/` | The whole trial sequence runs with fakes including a budget stop, a crashed runtime, a contamination hit, a failed scorer with one retry, a submission naming another trial's adapter (ignored) and the status mapping table row by row; the proxy returns 429 at the cap, reserves before forwarding, settles on stream usage and keeps the reservation when there is none; harvest parsers read the probe fixtures and refuse planted links; the control API refuses a wrong token; two trials run concurrently against one shared thread-safe fake ledger |
| 6 | **Ledger, CLI and integration** | `nightshift/lab/{ledger,fold,budget,cli,__main__}.py`; `tests/test_lab_{ledger,fold,budget,cli,integration}.py`; `tests/support/lab_world.py`; one line in `pyproject.toml` `[tool.setuptools.package-data]`: `"nightshift.lab" = ["hostkit/**/*", "assets/**/*"]`; append-only edits to `docs/index.md` and `docs/log.md` | contracts for waves A and B; every other module and test support for integration | none | Kinds and payloads are enforced; many threads append concurrently and `verify()` passes; freeze-once is enforced at append and in the fold (first wins); the fold tolerates malformed events and lists every case in the fold section; `load_frozen_all` rejects a rewritten file; `verify` recomputes decisions; the meter reserves, settles, hard-stops and is idempotent across reopen; the CLI order rules refuse out-of-order commands (every rule in the cli section); `abort-trial` rebuilds usage; `reveal` is gated; one fake trial, one candidate round, one final comparison and one report run end to end through `cli.main(argv, world=build_fake_world(tmp))`; `tests/test_packaging` still passes; the existing core tests (`test_budget`, `test_ledger_interrupt`, `test_grading`, `test_models`, `test_fence`, `test_ledger_view`) give the same results as before (two pre-existing macOS failures in `test_grading`) |

**Fakes: who contributes what.** There is no `--fake` flag and no fake in production code. The CLI is `cli.main(argv, *, world=None)`; tests build the world.

| Owner | File | Classes the world builder expects |
|---|---|---|
| 1 | `tests/support/lab_fake_generator.py` | `FakeGenerator`, `FakeVerifierBackend`, `FakeModelServer`, `FakeServeSession` |
| 2 | `tests/support/lab_fake_gpu.py` | `FakeProvider`, `FakeSsh`, `FakeTrainer` (returns `TrainResult` and writes a real dev score through the real layout) |
| 3 | `tests/support/lab_state_builder.py` | `LabStateBuilder` |
| 4 | `tests/support/lab_fake_proposer.py` | `ScriptedProposalSource`, `InMemoryPackStore` |
| 5 | `tests/support/lab_fake_runtime.py` | `FakeRuntime`, `FakeRuntimeHandle` |
| 6 | `tests/support/lab_world.py` | `FakeClock`, `build_fake_world(tmp) -> LabWorld` (imports the five files above) |

Production factories the CLI calls, by name: `HttpGenerator` (through a `GeneratorFactory`), `ScoreService`, `GlimmerTrainer`, `RentalModelServer`, `CoralRuntime`, `TrialRunner`, `PackRepo`, `OuterController`, `WonderProposalSource`, plus `LabLedger`, `FoldReader`, `TrialBudgetMeter`, `LabCap`.

**Launch checklist for the lead (before the owners start).** Commit or stash the pending documentation edits (`docs/index.md`, `docs/log.md`, `docs/goal-and-critical-path.md` are modified and this design file is untracked), so owner 6's appends are clean diffs; add the `pyproject.toml` package-data line above; run `tests.test_lab_contracts` under Python 3.11 in CI.

**Order.**

- **Wave A, starts now, all in parallel, depends only on contracts:** 6 writes `ledger.py`, `fold.py`, `budget.py` first (everyone else will want them within hours); 1 writes `verifiers.py`, `generator.py`, `scorer.py`; 2 writes `gpu_rentals.py`, `gpu_thunder.py`, `gpu_remote.py`, the hostkit scripts and the reference recipe; 3 writes `stats.py`; 4 writes `packs.py` and the v0 template; 5 writes `ENTRYPOINT.json`, `metering_proxy.py`, `control_api.py`, `researcher_tool.py`.
- **Wave B, as the wave-A interfaces land:** 1 `ifdata.py`, `rewards.py`, `contamination.py`, `score_session.py`; 2 `model_server.py`, `trainer.py`, `gpu_watchdog.py`; 3 `report.py`, `chart.py`; 4 `outer_context.py`, `proposer.py`, `outer_loop.py`; 5 `runtime_coral.py`, `trial.py`; 6 `cli.py`.
- **Wave C, integration:** 6 assembles the fake world and the end-to-end tests; the others fix their modules against it.
- **Critical path to the first GPU run (build-order step 1):** assignments 1 and 2 are the only ones on it. Their hostkit scripts and the scorer must be ready for the lead to run the scorer on arm 1 on a real rental (headroom, noise, real cost), then join runtime and trainer into one trial. The remaining assignments are on the critical path to Monday's rule freeze and outer loop. Today is November 1 and the submission closes on November 4 at 9 PM Eastern; if wave B slips, cut in this order: `chart.py` polish, the held-out-family pool beyond one assignment, the second candidate lever, fresh constraint instantiation (fall back to the salted draw, stating the residual risk). **Do not cut:** the metering proxy and its reservation, the hard stop, `reap` or its documented substitute, the redaction test, the contamination scan, freeze-once and `load_frozen`, the final guard and the final-comparison refusal, `verify-isolation`'s link and token checks.
- **Hardware steps owned by the lead, not by an implementer:** copy the repository to the control host, create the trial users, install CORAL and opencode, run the first real rental, the first live Spark smoke call (when a key exists), the first Wonder call, the verifier parity acceptance check, and record the terms-gate outcome.

## Open decisions

1. **Terms gate** (owner Ankit): record an outcome with `record-gate` before the first Wonder-directed revision. The outer loop refuses to run without it.
2. **GPU funding and the dollar hard stop**: no source is confirmed; there is no default in code. Set with `rentals init --hard-stop-usd`. `LabConfig.lab_budget_usd` (the Meta credit) is the lab-wide cap for researcher inference plus Wonder.
3. **GPU shape**: Thunder A100 80GB (cheapest, supported by evidence) or an H100 route (unavailable on Thunder at reading time, Modal $3.95 per hour unconfirmed credit). Decide after the first measured trial; the shape is pinned in `TrialSpec.shape` and in the comparability key.
4. **Lease per job or per trial**: the design rents one GPU per job (idle thinking time is not billed; the model download and load are repeated). Revisit after the pilot measures the fixed overhead.
5. **Held-out family set and assignment count**: default `copy` and `firstlast`; stricter adds `keywords_new` and `counting`. Number of assignments, and the sizes of train, dev and selection, are sized from measured trial cost and the pilot noise.
6. **Test-set file and headline metric**: current 300-row file with the 293-row legacy-comparable subset alongside; prompt-level strict as headline. UNVERIFIED which file and mode Meta used for 77.0.
7. **Outer loop sees the accept/reject bit only** (spec) or also selection magnitudes (WeCo). Default: bit only, and the operator sees the same bit-only view during the search.
8. **Rule parameters**: `alpha`, `min_gain`, screen batch size and trials, confirm trials, the variance guard ratio, the failed-trial policy, `assumed_gain`, `max_candidates`: set after the v0 spread is measured and before any candidate result. `freeze-rule` computes the minimum detectable gain and refuses when it exceeds `assumed_gain`; if the lab cannot afford a rule that can detect the gain the operator believes in, the honest options are fewer candidates, a bigger effect target or no search claim.
9. **Control host and researcher host**: ax43 and ax41 are proposals; the user separation, SSH reachability and the second host for `reap` are UNVERIFIED. Per-trial throwaway users need root (passwordless sudo on ax41 is unchecked); the serial shared-user fallback is labelled in the report.
10. **System name**: Nightshift versus a rename (open item in the specification). This design keeps the `nightshift.lab` package name only because the package path is already shipped by `find`.
11. **Third-party verifier code**: fetched at setup and pinned (D10) versus vendored; ask the organisers or avoid vendoring.
12. **Dollar axis semantics**: one cap over researcher inference plus GPU (this design) versus separate caps.
13. **Documentation index and log**: done in this revision (`docs/index.md` and `docs/log.md` list this document); owner 6 only appends.
14. **Second provider token and host for `reap`**: required before the first paid rental, or the documented substitute (operator procedure, alert, 2-hour caps) is accepted in writing.
15. **Where master anchors are pushed**: the git remote that receives `anchors/master.anchors`.
16. **Fresh constraint instantiation**: confirm that IFEvalG exposes a sampling interface for every constraint id, else the salted-draw fallback applies.
17. **Rule false-accept simulation**: the critics' numbers (0.05 for screen-then-confirm) are reported, not rerun here; owner 3's acceptance test must reproduce them or the rule is not frozen.

## Critic findings and resolutions

Two independent critiques reviewed the first version of this design and `contracts.py`. IDs: **A** is the first critique (contracts, interfaces, ownership), **B** the second (leaks, statistics, safeguards). Blockers and majors are listed in full; minors are listed when they changed something. "Fixed" names the place. "Rejected" and "Partly" give the reason.

| ID | Sev. | Finding | Resolution |
|---|---|---|---|
| A1 | blocker | Nothing linked a score to a `FinalRequest`; the guard could not count requests versus scorings; held-out data reachable in the search phase | Fixed. `final_request_id` on `ScoreRequest`, `ScoreRecord`, `TrialSpec`, present iff the phase is final (contracts, tested). Access-policy table and `final_guard` in [Scorer side](#2-scorer-side-owner-1): held-out denied in the search phase for every split, dev included; every final-phase score, including the trial runner's, calls the guard. |
| A2 | blocker | Stock arm has no pack; test set has no assignment id | Fixed. `FinalRequest.pack` and `pack_version` are optional with arm stock iff no pack; `TEST_ASSIGNMENT_ID` is required for the test split, forbidden elsewhere, and reserved from manifests (tested). |
| A3 | blocker | `final_analysis.json`, `FinalComparison`, `DecisionNumbers`, `LintResult`, `ProposalFacts`, `Report` undefined | Fixed except two. `FinalAnalysis` (+ `AssignmentWeight`, `PlannedTrial`), `lab.final_analysis_frozen`, `LabLayout.final_analysis_path`, `freeze-final-analysis`, `DecisionNumbers`, `FinalComparison`, `LintResult` and an extended `TrialScore` are in contracts. Rejected for `ProposalFacts` and `Report`: each is produced and consumed inside one owner's modules (4 and 3), and the cli treats `Report` as an opaque value passed to `render_*`; a shared type would only freeze an internal shape. |
| A4 | blocker | `--fake` cannot import `tests/support`; no owner for fakes | Fixed (option a). `cli.main(argv, *, world=None)` with the `LabWorld` container in contracts; `tests/support/lab_world.py` builds the fake world; fake classes per owner and production factory names are listed in [Work breakdown](#work-breakdown). |
| A5 | blocker | Elided constructors and missing protocols (`LedgerWriter`, `PackStore`, `DecisionRule`, `StateReader`, `MeterFactory`, `GeneratorFactory`) | Fixed. All added to contracts (tested as protocols); every `...` in this document replaced by a full signature in [Module layout](#module-layout-and-per-module-design). |
| A6 | major | `charge_llm` missing from `BudgetMeter`; no usage type; resource-to-axis map; wall semantics | Fixed. `LlmUsage`, `reserve_llm`/`charge_llm` on `BudgetMeter`, `AXIS_OF`; two-event booking with `<id>:tokens` and `<id>:usd`, GPU call id `gpu:<rental>:<n>`, wall derived from the clock ([Budget enforcement](#budget-enforcement)). |
| A7 | major | `TrainJob.dev_request` template cannot be built with arm v0 or evolved | Fixed. The template has arm and adapter null, the trial's phase, `final_request_id`, seeds and decoding; enforced and tested (a final-phase dev template builds). |
| A8 | major | Who writes the dev `ScoreRecord`; `score_id` versus `request_id` | Fixed (option a). `Scorer.score` writes `request.json` and `score.json` and returns the record; the runner reads it back and appends the event; `score_id == request_id` enforced. |
| A9 | major | Single-writer ledger model contradicted the concurrent writers | Fixed. One coordinator process, `TrialRunner.run` in a thread, one thread-safe `LabLedger` per ledger, trials may run concurrently; thread-safety test required of `ledger.py`. |
| A10 | major | `LabState` untyped; frozen-file hash domains | Fixed. Typed fields keyed as documented; `load_frozen` and `FoldReader.load_frozen_all` re-hash the bytes; both hashes stated ([Ledger decision](#ledger-decision)). |
| A11 | major | Unit of statistics under-defined; failed trials; duplicate trials | Fixed. `selection_trial_scores(state, rule, pack_version)` rules, `Rule.failed_trial_policy`, fold malformed cases ([Statistics](#statistics), [Ledger side](#1-ledger-side-owner-6)). Partly rejected: the proposed `EXCLUDED` policy, because dropping failures judges a flaky pack on its survivors (B14). |
| A12 | major | ABORTED trials cannot carry a context hash; status mapping; `stop(reason: str)` | Fixed. `initial_context_sha256` optional for unscored statuses; status table; `abort-trial` replays the trial ledger; `stop(reason: ExitReason)`. |
| A13 | major | `HELDOUT_ONLY` contradicts the manifest invariant; revision keys unspecified | Fixed. `HELDOUT_ONLY` deleted; exact key sets for `data_revisions` and `source_revisions`; `check_trial_against_frozen` compares them. |
| A14 | major | File formats crossing owners unspecified | Fixed. [File formats](#file-formats) and metric/timing constants. |
| A15 | major | v0 pack cannot be written without the `glimmer` CLI, job contract and grader values; reference method unassigned | Fixed. [Interface appendix](#interface-appendix); reference recipe assigned to owner 2, integration to owner 4, fallback and due time stated. |
| A16 | major | Offline tests need YAML, pyarrow, nltk | Fixed. Lint split into dump and pure comparison; JSONL split builder with opt-in parquet converter; `LAB_THIRD_PARTY` skip; parity is lead-run. |
| A17 | major | `ifbench_mini` fixture could contain test rows | Fixed. Synthetic fixtures only; `test_lab_fixtures_clean` hash check (D10, [IFBench scorer plan](#ifbench-scorer-plan)). |
| A18 | major | `AdapterRef.path` absolute; adapter identity split | Fixed. Lab-relative path, `adapter_id`, `resolve(root)`, submission/adapter/result id agreement, `adapter_json_path`. |
| A19 | major | `LabLayout` missing paths; dead socket path | Fixed. All design paths added and tested; `trial_control_socket` removed; `LabConfig` record for `lab.json`. |
| A20 | major | Secrets in `repr`; unhashable records; unpersistable exchange types | Fixed. `repr=False`; contracts docstring and tests state unhashability and the exchange-type list. |
| A21 | major | Overshoot bound unsupported by the notes | Fixed. Bound restated; reserve-then-settle and a prompt-estimate rejection rule; report prints which applied. |
| A22 | major | Rule can degenerate (sigma 0, min_gain 0); spread definition | Fixed. Override reason required; pooled-df floor; `trial_spread` defined; `SpreadEstimate.df` must equal trials minus cells. |
| A23 | major | Two GPU cost sources; who computes `Decision.cost` | Fixed. Ticks enforce, rental ledger reports, both printed; `cost_breakdown` added to `StateReader`. |
| A24 | major | Ownership overlaps: docs files, package-data, import clash | Fixed or lead action. Index and log done here and owner 6 is append-only; one package-data glob line; absolute imports and `python -m`. Committing or stashing the pending documentation edits is a lead action in the launch checklist (this task may not commit). |
| A25 | major | No clock-to-ISO conversion | Fixed. `utc_iso`; every component takes `clock`. |
| A26 | major | `scorer_version` undefined | Fixed. `scorer_version_of`, `Scorer.scorer_version`. |
| A27 | major | Dev scoring on the rental needs a tunnel and IFEvalG staging | Fixed. Loopback forward and `hostkit/stage.sh` pins ([Glimmer trainer plan](#glimmer-trainer-and-serving-plan)). |
| A28 | minor | Unsourced import-cost figures | Fixed. Figures withdrawn; `parse_json_object` copied with an equality test; `t_quantile` implemented in `stats.py`. |
| A29 | minor | Enum formatting differs by Python version | Fixed. `_StrEnum`; tested on 3.12, 3.13 and 3.14 here (3.11 is not installed locally; CI runs it). |
| A30 to A36 | minor | Dev score definition; id allocation; `forbidden_values`; `owner_tag`; shellcheck; anchor field name; verification note | Fixed. [Ids, clocks, hashes and imports](#ids-clocks-hashes-and-imports); `forbidden_values` in `outer_context`; `owner_tag_of_key`; `bash -n`; the field is `trial_ledger_anchor`; the contract suite now has 83 tests. |
| B1 | blocker | Selection hidden only by location; deterministic public split; researchers can retrieve it | Fixed in design, partly UNVERIFIED. Secret salt, fresh instantiations (with a stated fallback), contamination scan with `CONTAMINATED`, egress allowlist, residual risk in the report ([Closing the leaks](#closing-the-leaks)). The feasibility of IFEvalG sampling and of restricting rental egress is not verified. |
| B2 | blocker | `FinalRequest` guard per pack commit is unsound; no search-closed or analysis precondition; scores visible as they arrive | Fixed. `SearchClosed`, `FinalAnalysis`, planned trial ids, one `FinalRequest` per (arm, pack commit) covering the plan, `final_guard` checks, one held-out/test score per planned trial and split, retry only after `ScoreFailure` with no revealed number, `FinalReveal` gating; final trials use held-out assignments only. |
| B3 | major | Sequential rule has a family false-accept rate of 0.13 to 0.25 and is fragile to sigma and variance | Fixed with one deviation. Screen-then-confirm, pooled df of at least 8, arbitrary-alpha t quantile, an acceptance simulation. Partly rejected: the critic's Welch term for the candidate's variance is replaced by a variance guard (INVALID above 1.5 times the frozen sigma), because Welch with three candidate trials gives about 2 degrees of freedom and would remove most power; the simulation test decides whether the guard suffices. The critic's numbers are reported, not rerun here. |
| B4 | major | At affordable n the rule is almost unwinnable; no trial allocation | Fixed. `Rule.minimum_detectable_gain` versus `assumed_gain` (freeze refused), at least half of the affordable trials reserved for the final plan, at least 5 trials per pack, the search rule labelled a heuristic filter. |
| B5 | major | `decide` cannot check comparability; budget equality unchecked | Fixed. `TrialScore` carries the key; `comparability_key`; `check_trial_against_frozen` refuses a non-frozen budget; interleaved incumbent trials; `TrialSpec.shape`. |
| B6 | major | Selection score can be re-rolled | Fixed. One selection score per (adapter, assignment, phase), seeds from the rule (`Rule.score_seeds`), retry only after a no-number `ScoreFailure`; the fold uses the first. |
| B7 | major | Freeze-once not enforced; hash domains; forged decisions; anchor | Fixed. `FREEZE_ONCE` at append and fold, `load_frozen` order, `verify` recomputes decisions, read-only `frozen/`, anchors committed and pushed (remote is an open decision). |
| B8 | major | Harvested files copied then read by privileged code; symlinks | Fixed. `file_ref` and `read_regular_bytes` refuse links (tested with a planted canary); `gpu_remote`, `collect` and the adapter fetch apply the same rules. |
| B9 | major | Budget gaps: missing usage, no lab-wide cap, concurrent bound, reasoning tokens | Fixed. Reserve-then-settle; `LabCap` and `lab_budget_usd`; bound restated; billing reconciliation on the first trial; `lab_dollars` axis. |
| B10 | major | Orphan GPU if the control host dies; `reap` cuttable; Thunder token on ax41 | Fixed in design. `reap` or a written substitute is required before the first paid rental and is off the cut list; the researcher-host token is removed and tested. The second host/token is an open decision. |
| B11 | major | One shared unix user for all trials | Fixed with a fallback. Throwaway user per trial and extended `verify-isolation`; the serial shared-user fallback is labelled when root is unavailable (UNVERIFIED). |
| B12 | major | Adapter lineage and scoring-host trust | Fixed. Own-job lineage, discarded researcher `dev_score`, fresh immutable scoring image, dev labelled researcher-influenced. |
| B13 | minor | Phase and role not tied | Fixed. `check_trial_against_frozen`, `validate_rule_against_manifest`, `validate_analysis_against_manifest`; the guard derives the phase from the ledger. |
| B14 | major | Attrition and spread selection undefined for the search | Fixed. Failed-trial policy, `SpreadPlan`, `SpreadMeasured` accounting, candidates counted from the ledger. |
| B15 | minor | Remaining adaptive channels | Fixed. Canary mutation test, bit-only operator view, `OperatorChange`, no terminal printing of selection rows. |
