# How to reproduce the work

Status of this document: authored. Steps (a) and parts of (b) and (c) rest on work that has run. Step (d) is planned and has not run. Where a command is not known from the verified facts, the step is written in words and marked `command: pending`. Facts verified 2026-11-03 21:00 ET, updated 2026-11-03 21:30 ET for the Muse-authored starter, the new `v0` pack, the data-origin rule and authoring provenance; this revision claims no new run.

Read the safety cautions at the end before you rent any GPU or handle any scoring data.

## Before you start

1. Get the submitted standalone repository. Public repository: pending (none exists yet). `command: pending`
2. Install the Python package `nightshift` and its pinned dependencies. Verified test environments: Python 3.12 (ax41) and macOS Python 3.13/3.14 for the lab modules. `command: pending`
3. Leave `MUSE_API_KEY` and `BASETEN_API_KEY` unset. The verified proof uses the Muse Code subscription route (Muse Code 1.4.14) with API keys unset, and no API purchases. The Muse API and Baseten routes are planned only and have not been tested.
4. Confirm you can tell the parts apart: pre-existing Nightshift work, work built during the event by an AI coding agent under human direction, work authored by Muse Code, and public upstream dependencies. The inventory is in `AUTHORSHIP_AND_LICENSES.md`. Two distinctions inside that inventory matter below: the Muse-authored starter files are in `nightshift/lab/assets/pack_muse_v0/` (recipe, README, tests, notes/seed.md), while `task.yaml` and `seed/.gitignore` in that directory are harness-owned template copies (authored by Claude Code, not Muse-authored); and the data-origin check is harness code authored by Claude Code (commit 3823ab6), not Muse-authored work. Muse session records are in `PROVENANCE.md`, with prompts and event logs stored in `submission-pack/provenance/starter-author/` and `submission-pack/provenance/pack-writer/` and outputs listed with sha256 in `muse_authored_files.sha256` in each folder.
5. Student checkpoint for every live step below: `meta-models/Muse-Glimmer-30B` revision `a4e59da52a7bc87ae7251dd5545c0dd437c44b68`. Weights access steps: pending. Weights licence: to verify before submission.

## (a) Run the unit tests

Status: checked. The lab test suite passes on Python 3.12 (ax41) and on macOS Python 3.13/3.14 for the lab modules. The earlier review process (four Codex rounds) did not reach a clean pass. Tests use fakes. Passing tests are not experimental results.

1. From a clean machine, install the package and pinned dependencies (step 2 above).
2. Run the lab test suite in full, without narrowing it. `command: pending`
3. Expected result: the suite passes in the environments above. If a test fails, record the failing test name, the Python version, the code commit and the dirty flag. Do not skip or rewrite a test to make the suite pass.
4. Note the narrow checks already on record, which you can repeat but cannot upgrade by wording: the outer reviser route (Muse Code 1.4.14 with `muse-wonder-full-1.0`) is verified with thirty unit tests plus three live calls; the Muse-Code runtime for CORAL is built and unit-tested and ran only with a scripted stand-in inside a container. The real Muse probe awaits a login on the researcher host (the Muse login on the researcher host is waiting for the owner).
5. Separately, the Muse-authored starter has its own narrow checks on record, and they belong to the starter, not to the lab test suite above: its own 10 unit tests pass in a no-network container, and the full `run.sh` smoke path ran end to end against the real harness origin check (the data-origin check in `glimmer-axolotl-train`, Claude Code commit 3823ab6) with a fake server and a stand-in verifier, no GPU, no real verifier. Status of the starter: checked in exactly those ways; not run on a GPU. To repeat them: run the starter's tests in a no-network container, then the starter's `run.sh` smoke path against a fake server and a stand-in verifier. `command: pending` A pass repeats a wiring check. It does not become a training run, a GPU result or evidence about any trial.

## (b) Score a model through the Inspect-based path

Status: checked in part, planned in part. Read the split carefully.

What has been checked:

- The scoring path now generates through an Inspect AI task. Each ledger score record carries the Inspect log file name and sha256 per seed, the task name and version (`glimmer_generate@1`), inspect_ai version, code commit and dirty flag. Pinned Inspect AI version: inspect_ai 0.3.287 (UK AISI).
- A dev-only Inspect task (`glimmer_ifevalg`) with the pinned dataset hash re-scored the stored generations of both reference scores: 300 of 300 prompts agree with the ledger on pass-all, constraint fraction and truncation.
- The Inspect path has been rehearsed against a local fake OpenAI-compatible endpoint only.

What has not run: the Inspect path has NOT yet run against a real vLLM server (vLLM 0.32.0). Any live vLLM scoring is `PENDING: needs real run`.

Steps to repeat the checked part:

1. Obtain the stored generations and the ledger export for the two reference scores. Paths and hashes: pending (see `EVIDENCE_MANIFEST.md`; hash fields are `null` in the template until the coordinator fills them from records).
2. Run the dev-only Inspect task `glimmer_ifevalg` over the stored generations, using the pinned dataset hash. The pinned dataset hash value: pending in this pack. Do not substitute another dataset revision. `command: pending`
3. Compare the re-score with the ledger, prompt by prompt, on pass-all, constraint fraction and truncation. The recorded agreement to match is 300 of 300 prompts. A smaller number is a mismatch to report, not to round up.
4. For a new score through `glimmer_generate@1`, confirm the ledger record carries all of: Inspect log file name and sha256 per seed, task name and version (`glimmer_generate@1`), inspect_ai version, code commit, dirty flag. If any field is missing, the score is not reproducible as recorded. `command: pending`
5. Do not run selection or test scoring on a researcher machine. The private scorer (pinned IFEvalG / IFBench verifiers) and its data stay behind the ledger guard. See the safety cautions.

## (c) The hand-run reference experiment

Status: run. This is the only completed comparison. It is hand-run. It is not a result of the two-loop system, not a CORAL result and not a WeCo result. No researcher agent took part.

Method, as verified: rejection-sampling fine-tuning. Glimmer samples its own answers to 100 public training prompts, the pinned checkers keep answers satisfying every constraint, one QLoRA epoch of 28 steps, rank 32, lr 2e-4, 465 s.

Steps to repeat it:

1. Rent one A100 80GB (Thunder at $1.12/h was used for the verified runs). Apply the rental safety cautions below before the rental starts, including the independent off-host stop.
2. Generate Glimmer samples for the 100 public training prompts, from the stock checkpoint above. `command: pending`
3. Filter with the pinned checkers, keeping only answers that satisfy every constraint (open-instruct IFEvalG checkers, commit b781e782a8fc530577f235aefe282dd046cb8ad2; IFBench, commit 1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d). `command: pending`
4. Train one QLoRA epoch of 28 steps, rank 32, lr 2e-4 (recorded training time 465 s). Training stack on record: Axolotl 0.19.0. `command: pending`
5. Score stock and adapter through the ledger, guard and pinned verifiers on 150 public E1 dev prompts, one sample each, max_tokens 6000. `command: pending`
6. Record cost and wall time. The verified paired reference run cost $1.10 and took 59 min. The earlier first smoke run cost $1.44 and took 77 min. Spend so far on Thunder: $2.54 total ($1.44 + $1.10).
7. Compare with the recorded result, quoted exactly. Stock 95/150 = 63.3% pass-all, adapter 91/150 = 60.7%; difference -2.7 points, 95% paired bootstrap interval -6.7 to +1.3, exact McNemar p = 0.34 (adapter-only passes 3, stock-only passes 7); truncated at 6000 tokens: 14 vs 25. Mean constraint fraction 75.6% vs 70.7%.
8. Interpretation to carry into any report: no improvement, indistinguishable from zero at this size; not an IFBench score; E1 dev is public; resolution about ±8 points. Meta's published IFBench 77.0 is not comparable.
9. Delete the rental by the exact-owned-instance procedure below, and confirm deletion from the off-host reaper's record. In the verified runs, the instance was deleted by an independent off-host reaper running on another machine with its own login.

A repeat that gets different numbers does not replace the recorded result. Report both, with commits, dataset revision and seeds. Seed values: pending in this pack (coordinator fills from ledger records). Do not guess them.

## (d) The two-loop experiment, stages 0 to 3 ("Tier B", exploratory)

Status: planned. None of this has run. Every result cell in `REPORT.md` for these stages is `PENDING`. The design below is approved but not yet run. The smallness of the comparison is exploratory by design, and every public mention must say so.

### Budget, hard stops and rules (fixed before candidate results)

- Budget: at most $25 additional Thunder spend in aggregate (cumulative at most $27.54 of the original $40 ceiling).
- Each trial: one rental per researcher job, a trial GPU account of 72 GPU-minutes, a hard stop of 105 minutes per trial including setup and a selection score of up to 30 minutes.
- At most 3 rentals at once.
- Exact-owned-instance cleanup; ledger hard stop plus detached watchdog plus an independent off-host reaper.
- Keep rule: keep a candidate only if its two-trial mean selection score exceeds the v0 mean by more than the v0 spread and by at least 2 points.
- A trial with no valid adapter counts as stock (no gain) and is listed.
- No trial added or dropped after seeing a score.
- Test file scored once, never used for decisions.
- Pilot-failure stop: if the pilot cannot return a privately scored adapter after its attempt and the one infrastructure retry, later paid stages are cancelled and the failure is reported.
- The student model is reset to the stock checkpoint for every trial.
- Researchers see public dev feedback only. Selection and test data stay behind the ledger guard and never reach researcher machines.
- Data-origin rule (enforced in the harness; authored by Claude Code, commit 3823ab6): the training command `glimmer-axolotl-train` refuses any data in which an assistant target is not an unmodified, verifier-passing, non-truncated sample of the local Glimmer model (`muse-glimmer`) recorded for an approved public training prompt, and it writes a hash record (train file, samples file, data file, verifier module) for what it trains on. Owner rule: no answers from Claude, Muse Spark, Muse Wonder or other agents may be training targets; code authorship and training targets are different things. Limit, so nobody over-reads a pass: the check cannot detect a recipe that forges its own samples file; it catches hand-written, fixture and other-model targets and non-approved prompts, and gives an audit trail.

### Data for these stages

IF-RLVR rows partitioned by constraint family from metadata only, before any scoring: evolution assignment E1 (families firstlast, keywords_new, format_new, ifeval25) and held-out assignment H1 (families copy, counting). 348 rows exposed to a model in an early smoke run were excluded before the split was frozen. Training prompts come from the IF-RLVR training prompts (allenai/IF_multi_constraints_upto5). Dataset revision values: pending in this pack. Do not re-split. Do not move rows between E1 and H1 after any score exists. Whatever a recipe samples for its training data, the data-origin rule above decides what may be trained on: assistant targets must be unmodified, verifier-passing, non-truncated samples of the local Glimmer model (`muse-glimmer`) recorded for an approved public training prompt. Where the approved-prompt list is recorded: pending in the facts on record; do not substitute a list of your own and still call the result a repeat.

### Stage 0, pilot

1. Confirm the Muse-Code runtime for CORAL runs with a real Muse login on the researcher host. Today it has run only with a scripted stand-in inside a container, and the real Muse probe awaits a login on the researcher host. This step is `PENDING: needs real run`. `command: pending`
2. Take the pack version `v0` as committed and do not edit it for the trial: pack version `v0`, fresh lab root `glimmer-lab-2loop`, pack git commit `b3ebdcfcd82f087c90ba1b8e8aa9662ac534ef30`. No trial has used it yet. Record that commit in the evidence manifest with the trial, and record a pack sha256 only if the coordinator fills one from the pack record: pack sha256: `PENDING`, and the git commit is never entered as the sha256. In the starter directory on record beside this work (`nightshift/lab/assets/pack_muse_v0/`), keep the authorship line straight: recipe, README, tests and notes/seed.md are Muse-authored; `task.yaml` and `seed/.gitignore` are harness-owned template copies, not Muse-authored. The earlier Claude-authored starter pack (`pack_v0`) is history, was never used in any trial, and is not the pack for this stage. `command: pending`
3. Run one joined v0 trial on assignment E1: the researcher team (Muse Code 1.4.14 with `muse-spark-1.3`) writes and runs QLoRA recipes on one rental through the restricted `glimmer` tool, reads public dev feedback, and hands in an adapter within the trial accounts above (72 GPU-minutes; hard stop 105 minutes including setup and a selection score of up to 30 minutes). Training runs through `glimmer-axolotl-train`, so the data-origin rule above applies to every recipe in the trial: data that fails it is refused, and data that passes leaves the hash record (train file, samples file, data file, verifier module) for the manifest. `command: pending`
4. Score the handed-in adapter privately through the ledger, guard and pinned verifiers. `command: pending`
5. One infrastructure retry is allowed for this pilot. If the pilot cannot return a privately scored adapter after its attempt and the one infrastructure retry, cancel later paid stages and report the failure. Do not convert an infrastructure retry into an extra scored trial.
6. What stage 0 would show: the inner path operates. It would not show that researchers improve the student.

### Stage 1, second v0 trial (spread)

1. Run a second v0 trial on assignment E1 under the same frozen v0 pack and the same trial accounts. `command: pending`
2. Score it privately as in stage 0. `command: pending`
3. Record the v0 mean and the v0 spread from the two v0 trials. Both values: `PENDING` (fill from the ledger only).
4. Rule reminder: no trial added or dropped after seeing a score. A trial with no valid adapter counts as stock (no gain) and is listed.

### Stage 2, stock score and exactly one candidate pack

1. Score stock Glimmer on the E1 selection set through the private scorer. `command: pending`
2. The outer reviser (Muse Code 1.4.14 with `muse-wonder-full-1.0`) revises the pack once, producing exactly one candidate pack. Record the pack hash and the full revision trace. Hash: `PENDING`. `command: pending`
3. Run 2 trials of the candidate pack on assignment E1, at an equal budget to the v0 trials (same trial accounts, same hard stops). `command: pending`
4. Score both candidate trials privately. `command: pending`
5. Apply the keep rule fixed before candidate results: keep the candidate only if its two-trial mean selection score exceeds the v0 mean by more than the v0 spread and by at least 2 points. Accepted or rejected: `PENDING`. No accepted revision exists today.
6. What stage 2 would show: the outer path operates (adaptive search trace). Because selection scores drove the revision, stage 2 alone is not a held-out result.

### Stage 3, fresh held-out comparison on assignment H1

1. Freeze both packs: the frozen v0 pack and the frozen best pack coming out of stage 2. Hashes: `PENDING`. `command: pending`
2. Run 2 trials of each frozen pack on assignment H1 (families copy, counting), at equal budgets and under the same hard stops. That is 4 trials. `command: pending`
3. Score stock Glimmer on H1. `command: pending`
4. Score each H1 trial privately. `command: pending`
5. Run a single final scoring of three arms on the IFBench test file: stock, v0-researcher-trained, evolved-researcher-trained. The test file is scored once and never used for decisions. Which handed-in adapter represents each trained arm: owner to confirm from the frozen stage 3 plan before any H1 score exists, because the verified facts do not name the picking rule. `command: pending`
6. What stage 3 would support: an exploratory held-out comparison with two trials per pack. It is not a confirmation of researcher improvement. Say so in every summary.

### Cost accounting for stages 0 to 3

1. Before each trial, record the current cumulative Thunder spend. Verified spend so far: $2.54 total ($1.44 + $1.10). The Tier B ceiling is at most $25 additional in aggregate, cumulative at most $27.54 of the original $40 ceiling. If the next trial could cross the ceiling, stop and report. Do not start it.
2. Per trial, record GPU-minutes used against the 72 GPU-minute account, wall time against the 105-minute hard stop, and the selection score time against its 30-minute allowance.
3. Researcher token usage under the CLI route is measured from Muse usage events, not metered by a proxy. Report it under that label. Do not present it as proxy-metered usage.
4. Muse Code subscription allowance is used; no API purchases. Record usage from the session records (see `PROVENANCE.md`). Values: `PENDING`.

## (e) Produce the report from the ledger

1. Export the ledger records for the trials in scope. Export path and hash: pending (see `EVIDENCE_MANIFEST.md`). `command: pending`
2. Confirm each score record carries the Inspect log file name and sha256 per seed, the task name and version (`glimmer_generate@1`), inspect_ai version, code commit and dirty flag. Records missing a field are listed as incomplete, not repaired by hand.
3. Fill the Results section of `REPORT.md` from the ledger export only. Every cell that the ledger cannot fill stays `PENDING`. Never copy a number from a chat log, a video script or this file into a result cell. For each training row, attach the data-origin hash record written by `glimmer-axolotl-train` (train file, samples file, data file, verifier module) or record the refusal; a row with neither is incomplete as evidence that its targets were the student's own samples, because the check cannot detect a recipe that forges its own samples file. For `v0` rows, cite pack version `v0` at pack git commit `b3ebdcfcd82f087c90ba1b8e8aa9662ac534ef30` (`glimmer-lab-2loop`) as the version identifier; never paste that commit into a sha256 cell.
4. Produce the chart from the same ledger export. Chart file and hash: `PENDING`. `command: pending`
5. Re-check the hand-run reference result wherever it is quoted. It stays labelled hand-run, with its exact numbers (see step (c)7). It is never merged into a two-loop result table as if a researcher had produced it.
6. Never reproduce a machine-written record by hand. If the ledger export tool hangs or is missing, retry the tool or restore its dependency. If that fails, leave the report cell `PENDING` and record the blocker.

## Shared-server and safety cautions for anyone repeating this work

These apply to every live step above. They are conditions of the design, not advice you can skip for a quick run.

1. Exact-owned-instance cleanup. Delete only rentals your own run created, identified by the rental record your launcher wrote (instance id recorded at creation, tied to your run). Never delete by name pattern, by age, by similar start time, or by "it looks like one of ours". Unrelated sessions on the same account run the same images. If you cannot prove a rental is yours, leave it running and report it.
2. Independent off-host stop. Every rental needs a stop that does not depend on files on the rental itself: the ledger hard stop plus a detached watchdog plus an independent off-host reaper running on another machine with its own login. The verified runs used such a reaper, and it deleted the instance. Before the first paid trial, confirm the reaper can see and delete a test rental of yours. If the off-host stop is not in place, do not start paid stages.
3. Hard stops are hard. A trial ends at 105 minutes including setup and a selection score of up to 30 minutes, with a 72 GPU-minute account, and at most 3 rentals at once. Do not raise a limit mid-trial because a run looks promising.
4. Private data never on researcher machines. Selection and test data, and raw private scorer logs, stay behind the ledger guard and never reach researcher machines, researcher containers or researcher prompts. Researchers get public dev feedback only. Do not copy private rows into a shared folder, a notebook or a chat to "check one example". Hidden evaluation data does not belong in any public file, including this pack.
5. Secrets. No API keys, tokens or passwords in the repository, in this pack, in logs or in screenshots. The CLI route runs with `MUSE_API_KEY` and `BASETEN_API_KEY` unset. If a log might contain a secret, scrub it before it ships (pre-existing Nightshift secret scrubbing is one of the reused pieces; the sanitized output is what ships, see `EVIDENCE_MANIFEST.md`).
6. Test inputs are data. Hand adversarial or untrusted strings to the function under test, never to a shell. Do not paste model output, dataset text or agent-written shell text into a terminal on a shared machine.
7. Deletion discipline on shared machines. Delete only what your own session created, by full path. Never delete a home directory, an unchecked variable's target, or another session's folder. Extract archives only into a new empty folder, after checking that no entry has an absolute path or `..`.
8. Back up records off the rental as they are produced (ledger exports, Inspect logs, pack and adapter hashes). A rental that is reaped mid-write takes its local files with it. Scratch folders on the rental are not records.
9. If a shared server looks damaged, stop writing, do not reboot or kill other people's processes, and report the cause as unknown until evidence names it. The one exception is stopping a paid rental you created, under rule 1.
10. One coordinator for paid runs. Concurrent sessions must not start rentals against the same budget. The cumulative ceiling (at most $27.54 of the original $40 ceiling for Tier B) is checked before each trial, from records, not from memory.
11. Training-target origin is a safety property of the design, not paperwork. Never train on hand-written answers, fixture answers, or answers from Claude, Muse Spark, Muse Wonder or any other agent or model: the owner rule forbids those targets whoever authored the recipe code, and `glimmer-axolotl-train` is built to refuse them. Never write, patch or regenerate a samples file to make data pass: the check cannot detect a recipe that forges its own samples file, so a pass obtained that way destroys the audit trail the hash record (train file, samples file, data file, verifier module) exists to give. If training data is refused, record the refusal and fix the data's origin; do not route around the command.
