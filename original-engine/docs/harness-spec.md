---
type: spec
title: Harness specification for the Glimmer three-arm experiment
description: What to build for the November 4 submission: the inner and outer loops, the trainer, scorer and report tooling, their interfaces, access rules, build order and acceptance checks.
timestamp: 2026-11-01T18:00:00-05:00
---

# Harness specification

This specifies what to build for the [Glimmer three-arm experiment](goal-and-critical-path.md). It is a build spec, not a status report: as of the [October 30 inspection](implementation-critical-path-2026-10-30.md) none of it existed in this repository, and I have not re-checked since. The existing Nightshift modules named under [Reuse](#reuse) were not re-read when writing this; verify each before depending on it.

## What the submission is

The submission is **one system with two loops**, plus the measurement tooling that makes its claims believable.

- **System (the product):** the inner loop and the outer loop.
- **Tooling (the proof):** trainer, scorer and report, which both loops and the final comparison use.

The system's name is undecided (see [Open items](#open-items)). This spec calls it "the system".

## Architecture

```text
                       outer loop (Muse Wonder)
        reads incumbent pack + traces + dev scores, proposes a revision
                              |
                              v
   researcher pack vN  ----> TRIAL RUNNER ----> adapter + trial record
   (the thing evolved)         |   ^
                               v   |
          inner loop: CORAL team of Muse Spark researchers
          edit code, call the TRAINER on a rented H100, read DEV scores
                               |
                               v
                  SCORER (separate host, holds private data)
                  selection split -> keep/reject rule (search only)
                  held-out assignments + test split -> final comparison
                               |
                               v
                     LEDGER ---> REPORT (table, chart, costs)
```

## Terms

- **Researcher pack:** the directory, versioned in git, that defines a researcher team: CORAL config, instructions, tools, coordination and budget-use settings, and any seeded notes or memory. v0 is the first commit. An evolved researcher is a later commit. The outer loop edits only this. The whole pack, seeded notes included, is the treatment.
- **Assignment:** one complete post-training task instance (starting checkpoint, training inputs, dev data, private selection data) with an ID. Assignments come from a frozen **assignment manifest** that partitions constraint families into evolution families and held-out families.
- **Trial:** one run of one pack version on one assignment, one seed, one budget, starting from the same Glimmer checkpoint with a clean workspace. A trial returns an adapter and a trial record.
- **Arm:** arm 1 is stock Glimmer. Arms 2 and 3 are, for the final claim, new trials of the frozen v0 pack and the frozen evolved pack on held-out assignments. Trials run during evolution belong to the search and are not the arms.

## Components

### 1. Researcher runtime (inner loop)

- Runs a CORAL team on a remote worker. Muse Spark is the model, through CORAL's custom-model route (OpenCode through LiteLLM). Not yet exercised against Muse.
- Each researcher can read and edit its worktree, run commands, call the trainer, and read **dev** scores. It cannot read the selection or test data, the scorer, or the budget enforcement.
- Its environment is provisioned from approved snapshots only: a pinned training verifier built from Open-Instruct's `IFEvalG` checkers (see [Test-set isolation](#test-set-isolation)), and the training data. The IFBench package is not provisioned to it. Installing from the public index or cloning the upstream IFBench repository is not available to it.
- Output: a final adapter path and its own notes. The trial runner, not the researcher, requests the selection score.
- **Done when** a Spark researcher edits and runs code through CORAL and its tool trace is captured.
- A Muse model directing Glimmer training does not start until the [terms gate](goal-and-critical-path.md#terms-gate-an-early-prerequisite) is closed.

### 2. Outer loop controller

- Input: the incumbent pack version and, for recent trials, the **outer-visible record** defined in the [ledger section](#6-ledger-and-manifest) (permitted traces, dev results, costs and the decision bit), never the full ledger record. It never receives selection or test scores, per-seed or per-prompt selection results, decision margins or private artifact locations; the one thing it learns from selection is the accept/reject bit. That limits what it can read off the selection data, but it does not make selection confirmatory: the bits still steer the search, and many candidates are compared on the same data. Treat the selection results as an adaptive search trace.
- Wonder-directed revision does not start until the [terms gate](goal-and-critical-path.md#terms-gate-an-early-prerequisite) is closed. Owner acceptance of risk does not close it.
- Muse Wonder proposes **one** versioned revision of the pack with a written reason, committed as a new pack version with a diff.
- The candidate runs through the same trial runner at the same budget, on the same evolution assignments. Held-out assignments are never used here.
- A rule written to `rule.json` **before any candidate result** decides keep or reject. Every candidate, accepted or rejected, is recorded with its diff, reason, scores, cost and decision.
- Editable: the pack. Not editable: scorer, splits, budget enforcement, GPU executor, ledger, the rule.
- **Done when** one Wonder-proposed revision has been run and decided.

### 3. Trainer

- Wraps the supported Glimmer recipe: Axolotl QLoRA SFT, which Meta's post-training guide lists as verified on one H100 (checked November 1). The guide's results table and its measured TitanRL GRPO recipe come from eight-H100 runs (GRPO on a single node of eight 95 GB H100s, six for the trainer and two for the generator); it does not say whether fewer GPUs work. Axolotl's published Glimmer configuration reads chat-format response demonstrations, not prompts plus checker metadata. So the verified single-GPU path covers SFT on demonstrations only; nothing verified covers reward, rollouts or GRPO on one H100.
- Manages the rented GPU: start, upload, run, fetch the adapter, delete the instance, with a watchdog that deletes it on timeout or budget exhaustion. Adapt the lifecycle code from grpo-finetuning rather than writing it again; do not copy its Qwen trainer or its hard-coded cap.
- Training signal: IF-RLVR gives prompts and verifier metadata but no demonstrations. The trainer provides GPU and the supported stack; it does not decide the method, and the pack may try others. But the **reference method is a prerequisite to freezing v0**: local Glimmer generation on the rented GPU, verifier filtering with the pinned `IFEvalG` snapshot, SFT of the kept responses through the supported recipe, adapter reload, and a private score. Its samples come from Glimmer itself, so it needs no hosted-API-generated training data. Run it by hand, without a Muse service, and record generation yield (fraction of samples that pass), peak memory and total cost. Until it has run, v0 cannot be frozen. Any GRPO or other online-RL route on one H100 is a separate unverified integration, listed as such until a run exercises it; the report does not describe it as supported.
- Meters GPU minutes and dollars per trial, and reports them to the ledger.
- **Done when** the reference method has produced one adapter that is saved, reloaded and scored, with measured load, generation, training, export and evaluation times, plus generation yield and peak memory.

### 4. Scorer

- Same code path for all three arms. Input: base revision, optional adapter, split, decoding settings, number of generation seeds. Researcher-requested **dev** scoring goes through this scorer too, so it is charged to the trial's account (see [Budget](#budget)); selection and final scoring are not.
- Serves the model and runs the verifier that matches each split's constraint source: the pinned `IFEvalG` checkers for IF-RLVR-derived dev, selection and held-out data, and IFBench's own constraint verifiers for the final IFBench test. Only the scorer host holds the IFBench package and the test JSONL; the IFBench README says the installable package bundles the test file. Both verifier revisions are recorded in the ledger. The serving stack for Glimmer plus an adapter is **unverified**; confirm it in the first run.
- Output: per-prompt pass/fail and the aggregate, for each seed.
- Runs on a host the researchers cannot reach, and holds the selection and test data. Researcher workers never receive those files.
- The **test** split and the held-out assignments' private data are scored only by an explicit final-comparison command. The command refuses to run twice for the same pack version without a recorded reason.
- Arm 1 is the scorer with no adapter.
- **Done when** stock Glimmer is scored on dev and selection, for evolution families only, several times. This first run gives headroom against Meta's 77.0, the noise floor and the real GPU cost.

### 5. Report

- Reads the ledger. Produces the three-arm table, a paired per-prompt comparison between arms, trial-to-trial spread, and the chart of selection score against pack version with accepted and rejected revisions marked. The chart is titled as an adaptive search trace. Improvement is reported from the held-out-assignment comparison, with uncertainty over independent trials and assignments.
- Reports researcher inference, GPU and outer-loop costs separately.
- Labels any partial or custom IFBench scoring as such.
- **Done when** a table and chart render from real ledger entries.

### 6. Ledger and manifest

Every trial and every score appends one record. Required fields:

| Field | Content |
|---|---|
| ids | trial id, assignment id, pack version (git commit), seed, arm or phase (search or final) |
| inputs | starting checkpoint revision and hash, data revisions, scorer version, decoding settings, hash of the initial workspace, notes, memory and artifacts |
| budget | allowed and used: researcher tokens, GPU minutes, dollars |
| outputs | adapter hash, artifact location, trace location |
| result | split, per-seed aggregate, reference to per-prompt results |
| decision | for outer-loop candidates: rule, margin, accepted or rejected, reason |

This is the **full record**, for the evaluator and report only. The outer loop gets a separate **outer-visible record**, built from it by a fixed projection that is not under the pack's control:

| Field | Content |
|---|---|
| ids | trial id, pack version, seed, phase |
| budget | allowed and used |
| trace | the researcher trace and notes, and the diff and reason of the candidate |
| dev | dev-split aggregates and per-prompt results (researchers already see these) |
| decision | accepted or rejected only |

It omits selection and test aggregates, per-seed selection results, per-prompt selection references, decision margins, the rule's thresholds as applied to a specific candidate, and private artifact locations. Whether a trace can itself contain selection-derived text is unverified; the projection takes traces from the researcher workspace, which never held selection data.

## Access rules

| Resource | Researcher | Outer loop | Scorer | Report |
|---|---|---|---|---|
| Pack | edits its worktree | writes new versions | no | reads |
| Dev data and scores | yes | yes | yes | yes |
| Selection data | no | accept/reject bit only, via the outer-visible record | yes | aggregates |
| Test data | no | no | final command only | final aggregates |
| Budget enforcement, executor, ledger | no | no | no | reads |

Verify CORAL's actual user and permission configuration rather than assuming a directory named `private` protects itself.

### Test-set isolation

Separating the scorer host is not enough: the IFBench README (Installation) says the test JSONL ships inside the installable wheel (`ifbench.data_path()`), and the Evaluation section points to the repository's `data` folder. The public set is also retrievable over the internet. So:

- Researcher workers and the outer loop use a pinned **training verifier** from Open-Instruct's `IFEvalG`, not the IFBench package minus its test file. IFBench's `instructions_registry.py` has the classic IFEval checkers and the IFBench out-of-domain constraints; the IF-RLVR training constraints are in the `IFEvalG` registry. Checked November 1: the first IF-RLVR train row uses `detectable_format:sentence_hyphens` and `last_word:last_word_answer`, present in `IFEvalG` and absent from IFBench's registry, so an IFBench-derived verifier cannot reward those rows. Whether `IFEvalG` can be installed or copied without IFBench test data is unverified; the scan below covers it.
- **Coverage check before renting training hardware:** for every row kept for training, dev or selection, build each instruction ID's checker from the pinned snapshot with the row's arguments and run it on a placeholder response. Unresolved or raising rows are dropped or repaired, with counts recorded. This reads metadata only, so it precedes the partition freeze. Record the `IFEvalG` snapshot revision and the IFBench scorer-package revision.
- Training reward, dev feedback and selection scores on IF-RLVR-derived rows come from `IFEvalG`; only the final IFBench test uses IFBench's checkers. The report names the verifier behind each number, and headroom on IF-RLVR-derived rows is not comparable to Meta's 77.0 (an IFBench figure).
- Their dependency installs come from approved snapshots. Whether outbound network access can be restricted to the Muse endpoints, the GPU host and those snapshots is unverified. If it cannot be, the report states that researchers could in principle retrieve the public test set.
- Before evolution starts, scan every provisioned researcher environment and workspace for the test file (content hash plus a sample of test prompts) and record the result in the ledger.

## Assignment manifest and context

- The constraint-family partition and row-exclusion rule are frozen **before any pilot scoring, headroom check, v0 development or candidate run**, from constraint metadata alone and never from a score. Stock scoring, headroom, v0 debugging and all candidates use evolution families only. Held-out assignments' contents, scores and trial behaviour are unavailable to researchers, the outer loop and anyone tuning v0 until the frozen final comparison. If a pilot forces a change of task or partition, re-freeze and record it; a partition changed after any held-out score or behaviour was seen is not held out, and the report says so.
- The full manifest is frozen before any candidate runs. It lists assignment IDs, their data revisions and the constraint-family partition. Rows whose constraints span evolution and held-out families are never in an evolution assignment; whether they are dropped or used only in held-out assignments is recorded in the manifest. Whether IF-RLVR verifier metadata supports a clean family partition is unverified.
- Every independent trial starts from a clean workspace. Shared notes and attempt history live inside a trial only. Cached trained weights, and solutions or notes derived from held-out assignments, are never available to a pack.
- The initial-context hash is recorded for every trial, so a carry-over between trials shows up in the ledger.

## Budget

- **One trial account**, set in `budget.json` and equal for every arm-2 and arm-3 trial: researcher tokens and GPU minutes and dollars. It is debited by everything the trial causes on any host: training, rollouts and generation, and every dev evaluation a researcher requests from the scorer. The scorer authenticates each dev request with the trial's id and the executor debits the account; a request that would take the account past its allowance is rejected before it runs, not billed afterwards. A pack that makes more dev-evaluation calls therefore has less left for training, rather than drawing on a separate pool.
- **Centrally funded scoring is separate and identical for every trial:** the selection score the trial runner requests and the final held-out and test scoring use one fixed protocol (same decoding, seeds and sample counts) and are paid from their own line of the allocation, not from the trial account. Pack-specific choices cannot change how much of it a trial gets.
- Outer-loop search cost is counted separately and not against a trial.
- Hard stop in the executor, not in the researcher.
- Pricing is in the [plan](goal-and-critical-path.md#budget-gpu-hours-not-tokens). Per-trial cost is unmeasured; the first scorer run and first trial set it.
- GPU funding is not confirmed (see the plan's [GPU funding gate](goal-and-critical-path.md#gpu-funding-gate)). Before the first paid rental the owner records the credit source and balance, or an owner-paid amount he sets for the pilot; the executor's hard stop for the pilot uses that recorded amount. Neither is known now. If none is recorded, no rented-GPU step runs and the status is components tested, system not demonstrated. After the first trial, write the full allocation with a funding source per line: v0 repetitions, candidate trials, held-out comparison trials of both packs, generation and final scoring. The held-out comparison and final scoring are reserved before evolution spends, and the whole allocation must fit the remaining recorded balance; if it does not fit even the minimum confirmatory design, narrow the claim in advance.

## Keep-or-reject rule

- Fixed before any candidate result and saved to `rule.json`, together with the comparison statistic, the replication policy for incumbent and candidates, the candidate or stopping limit, and how repeated decisions are handled. Repeated comparisons on one selection set raise the chance of keeping noise, and the bits steer the search.
- Based on the measured spread of v0 trials. The exact form, for example a margin over the v0 spread, is chosen after the v0 variance is measured, not guessed now. Resampling prompts inside one trial measures scorer noise only; researcher reliability needs independent trials, with assignments as a further level when there are several. The v0 spread does not carry over to later packs, so the final incumbent is re-run.
- If there are too few trials to estimate spread, say so in the report instead of applying a rule without evidence.
- The rule governs the search. The improvement claim rests on the held-out comparison in the build order, not on the rule's decisions.

## Final-analysis manifest

The held-out comparison has its own contract, written to `final_analysis.json` and frozen **before any held-out or test score is revealed** to anyone who can still change the plan. The `rule.json` above does not cover it. It fixes:

- the primary metric and contrast (arm 3 minus arm 2 on held-out assignments); the IFBench README says the paper generally reports prompt-level loose accuracy, and the manifest names the variant used, with others reported as secondary;
- the assignments and how they are weighted in the aggregate;
- the number of trials per pack per assignment, or a sequential stopping procedure with its rules written in advance; no adding trials after seeing results;
- the uncertainty calculation, with independent trials and assignments as the sampling levels;
- the failure and retry policy: what counts as a failed trial, whether and when a retry is allowed, and how failures enter the aggregate.

The report lists every scheduled trial, failed ones included. Cost this design from the pilot's measured trial cost and variability before allocating evolution spending. If it cannot be funded, narrow the claim in advance instead of shrinking the design afterwards.

## Build order

0. Metadata-only data pass: the verifier coverage check and the family-partition check. Then freeze the family partition and row-exclusion rule. No model scores before this. Confirm the GPU funding gate (below) is recorded before any paid rental.
1. Scorer on arm 1, evolution families only (needs one rented GPU). Gives headroom, noise and cost. This uses no Muse service, so it is not held by the terms gate.
2. Researcher runtime and trainer, in parallel. Run the trainer's reference method by hand first (generate, filter, SFT, reload, private score) and record its yield, memory and cost; v0 is not frozen before this. Then join runtime and trainer into one trial that yields an adapter and a record.
3. Ledger fields, the outer-visible projection and a minimal report over real entries.
4. Freeze the full assignment manifest (built on the step 0 partition) and context treatment. Freeze v0; measure its spread; write `rule.json`. Draft `final_analysis.json` and cost it.
5. Outer loop controller; run candidates on evolution assignments.
6. Freeze the evolved pack and `final_analysis.json`. **Held-out comparison (completion gate):** new, budget-matched trials of frozen v0 and the frozen evolved pack on held-out assignments, from stock Glimmer, as scheduled in `final_analysis.json`, then private scoring.
7. Final comparison of arms 1, 2 and 3 on the held-out private data and the test split; render the table and chart.

The harness is not complete until step 6 has run. Scoring adapters left over from the search is a test of selected checkpoints, not of the evolved researcher. If the joined trial does not work by the end of November 1, the status is components tested, system not yet demonstrated. If the held-out comparison was not run, report that arm 3 versus arm 2 was not tested and show the search trace as such. If arm 3 is not ready, present arm 1 against arm 2 and say the outer loop was not demonstrated.

## Reuse

Candidates from the existing Nightshift code, to verify before use: `nightshift/ledger.py` (hash-chained ledger), `nightshift/budget.py` (metering; add GPU hours), `nightshift/grading.py` (split handling and statistics), `nightshift/secrets.py` and `nightshift/fence.py` (secret scrubbing and untrusted-text fencing), and the Meta client under `nightshift/models/`. Reused code must be labelled as pre-existing in the submission's provenance inventory (see the plan's November 4 section). The sandbox, which limits untrusted address space to 4 GiB and output files to 8 MiB, does not fit GPU training unchanged. The existing supervisor, workflow and auditor roles are not required.

## Out of scope

- PostTrainBench, Qwen and any leaderboard track.
- Dashboard work beyond the one evolution chart.
- Hosted WeCo services.

## Open items

- **System name.** Nightshift is the repository's public name and what [SUBMISSION.md](SUBMISSION.md) and the README currently describe, but that text describes an auditing pipeline, not this system. Decide whether to keep the name and rewrite the pitch, or rename.
- Whether IFBench headroom and the IF-RLVR training signal are usable. Decided by the first measurements.
- The serving stack for scoring Glimmer with an adapter.
- Where the scorer and private data live, so researcher workers cannot reach them.
- Whether the `IFEvalG` training verifier can be provisioned without IFBench test data and network access restricted so the IFBench test set stays off researcher environments.
- The contents of `final_analysis.json` (metric variant, weights, trial counts), which depend on the pilot's measured cost and variability.
- Meta API Terms §10.1(ix) and organiser rulings on Muse-directed fine-tuning of Glimmer and on Wonder revising the researchers. This is a gate with an owner in the [plan](goal-and-critical-path.md#terms-gate-an-early-prerequisite): no Muse-directed Glimmer training and no Wonder-directed revision until it is closed by written authorization or a documented determination. Accepting the risk does not close it. The plan sets a decision point and the fallback status if it is still open; no alternative route has been verified.
- The GPU funding source, balance and the owner's pilot spending limit. None is recorded.
- Whether any online-RL route (GRPO) runs on one H100. Unverified; only the SFT reference method is planned.
- Whether IF-RLVR metadata supports partitioning by constraint family.
- Whether every retained IF-RLVR row's `IFEvalG` checker resolves and runs, and the snapshot revision to pin.
