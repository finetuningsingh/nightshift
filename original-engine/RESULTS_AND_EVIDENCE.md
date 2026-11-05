# Results and evidence manifest

## In this extract

| Evidence | Path | What it shows |
|---|---|---|
| Sanitized summary of the real study | `evidence/recorded-study/lab2p-summary.json` | All 56 main-ledger events in order, each with seq, hash and prev_hash, and whitelisted fields only. It covers the frozen plans, the spread, the rule, the candidate, scores (aggregates, Inspect log hashes, per-prompt file hashes), scans, operator-change targets, the frozen final analysis and trial outcomes. It also holds each trial's GPU jobs, failures, GPU minutes and ledger head hash, and every rental's minutes and rate. Free-text decision records are omitted. |
| Replay | `scripts/replay_recorded_study.py` | Checks the summary's hash-linked sequence and prints the trial, score and decision tables. With `--private-ledger`, it also verifies against the private ledgers |
| Muse authoring provenance | `submission-pack/provenance/` | Command lines, versions, times, exit codes and sha256 of every Muse authoring step and authored file |

## Results (aggregates, from `lab.score_recorded`)

| Score | Assignment, split | Trial (pack) | Pass-all | Mean constraint fraction | n | Truncated |
|---|---|---|---|---|---|---|
| `t-0006-sel` | E1 selection | `t-0006` (v0, fresh) | 55.0% | 76.4% | 300 | 0 |
| `t-0003-sel` | E1 selection | `t-0003` (`cand-001`, screen) | 52.0% | 75.4% | 300 | 0 |
| `t-0004-sel` | E1 selection | `t-0004` (`cand-001`, fresh) | 47.0% | 70.1% | 300 | 0 |
| `sc-0001` | E1 selection | stock Glimmer, context only | 63.0% | 75.4% | 300 | 29 |
| Dev scores | E1 dev (150 prompts), H1 dev (`t-0009`, 100 prompts) | various | see replay output | | | |

No H1 selection score and no IFBench test-set score exists. Reading and limits: [README.md](README.md#what-actually-happened-exploratory-no-improvement-claim) and [CLAIM_AUDIT.md](CLAIM_AUDIT.md).

## Kept private (not in this extract)

| Kept private | Why | How it is referenced |
|---|---|---|
| Data splits (train, dev, selection, held-out, test), split salt, template n-gram list | Evaluation integrity; test-set licence | sha256 in `frozen/assignments.json` and in contamination scan events |
| Per-prompt results and generations of every score | Contain evaluation prompts and answers | `per_prompt` sha256 in each `lab.score_recorded` event |
| Inspect `.eval` logs | Contain evaluation prompts and answers | `inspect_log_sha256_seed<N>` in each score's `verifier_pins` |
| Adapters (about 767 MB each) | Weights are not published without owner release | sha256 in `lab.trial_finished` and score events |
| Job logs, researcher traces, CORAL notes | Contain prompts and training rows | Trial ledgers record job ids and outcomes |
| Raw ledgers (free-text decision records), run record, experiment plan, coordination history, session transcripts, Muse authoring prompts and event logs | Coordination material, not project material | Muse step hashes in `submission-pack/provenance/*/steps.jsonl` |
| Ledgers of the other lab roots (lab2l, lab2m, lab2n, lab2o) | Earlier attempts (the pilot and roots replaced before lab2p), described in the private run record | Can be added on request |

All private material is backed up off the machine (ax41), and copies are kept with the owner.

## Test run of this extract

Run on 2026-11-04 on ax41, inside a docker container with no network and a read-only root filesystem (image `python:3.12-slim` plus `git` and `procps`), one test module at a time.

- 73 modules, about 4,770 tests.
- All modules pass except 7 tests in 3 modules:
  - `test_sandbox`: 2 failures and 3 errors.
  - `test_grading`: 1 failure.
  - `test_isolation_modes`: 1 failure.
- Triage of the 7 failures, reproduced in the same container:
  - **5 come from the container's user setup.** The untrusted-run sandbox (`nightshift/sandbox.py`) sets the per-user process limit to "this user's threads plus 64". It counts the threads from `/proc`, which shows only the container's own processes (about 2). The kernel counts all of host uid 1000's threads (about 600) against that limit, so every `fork` in the untrusted script fails.
    - The 5 tests: the three daemon tests in `test_sandbox`, `test_a_detached_process_cannot_alter_the_staged_outputs` and `ModeNoneTests.test_hostile_run_prevents_and_detects_exactly_what_the_mode_promises`.
    - Run as an unused uid (4242), all three modules pass in full: 167, 114 and 23 tests.
  - **2 are timing flaws in the tests themselves.** On Linux 6.8, file timestamps advance once per kernel tick of about 1 ms, so a file created and deleted within one tick of the snapshot looks unchanged.
    - The 2 tests: `test_added_removed_replaced_renamed_and_chmodded_entries_are_all_named` and `test_a_file_that_comes_and_goes_is_still_seen_after_the_grader_when_no_bytecode_is_involved`.
    - The sandbox check itself is not weakened in real runs, because the trusted tree was last changed long before a run starts.
    - A third test has the same flaw, `test_bytecode_next_to_a_transient_file_is_excused_only_for_the_bytecode`.
    - The fix is a 20 ms pause in these tests, in `post-study-changes/`. This folder's copy still has the flaky tests.
  - A possible hardening, not made: after setting the process limit, try one fork; if it fails, relax the limit and record a note.
- Inspect AI tests are skipped here: the container has no `inspect_ai`.

Separately, a later development run of the outer-loop changes (`outer_challenge.py`, served-model check), also in the ax41 container, passed 398 tests. Those changes are newer than this extract and are not part of it.
