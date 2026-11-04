# Nightshift original engine: Glimmer Lab two-loop post-training

This folder holds the **original implementation** that produced every recorded result of the Nightshift hackathon entry (Meta Muse hackathon, October 26 to November 4, 2026), together with the evidence of the one real study it ran. It is a standalone extract: the code, tests, selected design documents, the Muse-authored submission pack and the recorded ledgers, without the private repository's history, coordination notes, data splits, prompts, answers or per-prompt results.

It is separate from the newer standalone controller in this repository's root, which is a later rewrite and has not run any real experiment (see the repository root README). Results below belong to this original engine only.

## What it does

Nightshift asks whether a stronger agent can improve the *researchers* that post-train a model, measured by what the trained model scores under a fixed budget, not by whether code ran.

- **Inner loop.** A team of researcher agents (Muse Code 1.4.14 running `muse-spark-1.3`, inside CORAL, in a docker container on a separate host) receives a researcher pack: written instructions, a working recipe and a `glimmer` tool. They post-train **Muse Glimmer 30B** (`meta-models/Muse-Glimmer-30B` at revision `a4e59da52a7bc87ae7251dd5545c0dd437c44b68`): Glimmer samples answers to approved training prompts, verifiers keep only answers that pass every constraint, and QLoRA SFT trains an adapter on those kept answers. Each training or evaluation job gets its own GPU rental. A trial has a 72 GPU-minute account.
- **Outer loop.** Muse Code running `muse-wonder-full-1.0` reads permitted trial evidence and proposes one revision of the researcher pack (its instructions, recipe or tools). The same runner then trains with the revised pack at the same budget.
- **Scoring.** A private `ScoreService` scores adapters on frozen splits through an Inspect AI task (`glimmer_generate@1`). Researchers never see selection, held-out or test data. Every score records its Inspect log hash, scorer version and code commit in a hash-chained ledger.
- **Guards.** The guards cover the following:
  - Every event lives in a hash-chained ledger, and plans and rules are frozen before use.
  - Failed or contaminated trials stay in the record.
  - A training-data origin check runs inside the trainer: every target must be an unmodified, verifier-passing Glimmer sample on an approved prompt, never an answer from Muse Spark, Muse Wonder or any coding assistant.
  - A contamination scan checks training data against evaluation text.
  - Rentals get hard stops, plus an off-host reaper.

Every model the project calls at runtime is a Meta model: Muse Spark 1.3 (researchers), Muse Wonder Full 1.0 (outer reviser) and Muse Glimmer 30B (the model being trained). See [AUTHORSHIP.md](AUTHORSHIP.md) for which tools wrote which code.

## What actually happened (exploratory; no improvement claim)

All numbers come from the recorded ledger of lab root lab2p, published here as a sanitized summary, `evidence/recorded-study/lab2p-summary.json`. Replay it with `python scripts/replay_recorded_study.py`. The scores are on the frozen private E1 selection split: 300 prompts, one seed, max 6000 tokens, the same Glimmer checkpoint and the same 72 GPU-minute trial budget.

| Arm | Trial | Pass-all | Mean constraint fraction |
|---|---|---|---|
| Original SFT (v0 researcher pack) | `t-0006` (fresh) | 55.0% | 76.4% |
| Nightshift-revised SFT (`cand-001` pack) | `t-0003` (the screening trial that selected it) | 52.0% | 75.4% |
| Nightshift-revised SFT (`cand-001` pack) | `t-0004` (fresh) | 47.0% | 70.1% |
| Stock Glimmer, context only (not SFT) | `sc-0001` | 63.0% | 75.4% (29 of 300 answers truncated) |

- **Both loops ran for real.** Spark researchers chose and ran Glimmer training. Wonder Full proposed `cand-001` (lever `budget_use`: generation time box 2700 s to 720 s), which passed lint, was committed and was trained at the same budget.
- **The revision did not improve quality on this evidence.** It may have improved reliability:
  - Both candidate trials that ran produced a selection-scored adapter. `t-0007` was opened and aborted with zero use.
  - Of the four original-pack trials in the same lab, one did:
    - `t-0001` submitted nothing.
    - `t-0002` trained an adapter, but its selection scoring failed twice on SSH, which is an infrastructure fault.
    - `t-0005`'s researcher runtime crashed.
  - One scored original trial against two candidate trials, chosen adaptively, cannot establish a quality difference either way.
- **The held-out H1 comparison was attempted and produced no valid pair:**
  - `t-0008`: a harness guard bug refused every job's dev evaluation. It was fixed afterwards, and this extract includes the fix.
  - `t-0009`: the contamination scan flagged its training data. Held-out instruction templates are not recognised as boilerplate, which is a design gap, not a researcher fault.
  - `t-0010`: stopped by the operator.
  - `t-0011`: never run.
  - No H1 selection score and no IFBench test-set score exists.
- **Failed trials** are listed as failures, not as 0% scores.
- **The E1 confirmation** run by the keep rule (5 candidate and 4 original trials needed) stopped incomplete. No keep or reject decision exists.
- **Cost.** About 15.73 USD of GPU rental across all lab roots, against a 40 USD ceiling (ledger estimates). Muse model use ran on a Muse Code subscription.

Full account: [submission-pack/REPORT.md](submission-pack/REPORT.md) (Muse-authored). The coordinator's chronological run record and the experiment plan stay private because they contain coordination history.

## Layout

| Path | What |
|---|---|
| `nightshift/` | Core: standard-library-only hash-chained ledger, orchestrator, CLI, dashboard |
| `nightshift/lab/` | Glimmer Lab: scoring service and guard, Inspect generator, trainer and GPU rentals (Thunder), CORAL + Muse Code researcher runtime, Muse CLI outer proposer, contamination scan, statistics, reports |
| `nightshift/lab/hostkit/` | What runs on a rented GPU: bootstrap, vLLM serving, axolotl QLoRA, data-origin check |
| `nightshift/lab/assets/pack_muse_v0/` | The researcher starter pack v0 (Muse Wonder Full-authored) |
| `tests/` | Unit and integration tests, with fake GPUs and fake models (no network) |
| `scripts/` | Live runner, spend report, off-host reaper, the earlier hand-run reference recipe, recorded-study replay |
| `docs/` | Architecture, invariants, harness spec and design, Inspect discipline, reference experiment |
| `submission-pack/` | Muse Wonder Full-authored submission documents and their provenance hashes |
| `evidence/recorded-study/lab2p-summary.json` | Sanitized summary of the real study's hash-chained ledgers (event hashes kept), frozen final plan and rental records |

Some design documents link to private planning notes that are not published here.

Setup, tests and replay: [SETUP_AND_REPLAY.md](SETUP_AND_REPLAY.md). Claim audit: [CLAIM_AUDIT.md](CLAIM_AUDIT.md). Evidence manifest: [RESULTS_AND_EVIDENCE.md](RESULTS_AND_EVIDENCE.md). Authorship and tools: [AUTHORSHIP.md](AUTHORSHIP.md).
