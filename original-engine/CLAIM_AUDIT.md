# Claim audit (2026-11-04, Claude Code, from the recorded ledgers)

Each claim the submission makes, with its evidence and verdict. "Ledger seq N" means event N of the recorded lab2p main ledger, listed in `evidence/recorded-study/lab2p-summary.json`; `scripts/replay_recorded_study.py` prints them.

## Claims that hold

| Claim | Evidence | Verdict |
|---|---|---|
| The inner loop ran for real: Muse Spark researchers chose and ran Glimmer training | `lab.trial_finished` environment for lab2p `t-0001`..`t-0006` and `t-0009`: runtime `coral+muse_code+docker`, model `muse-spark-1.3`, Muse Code `1.4.14-R7309.1`, CORAL `0123dfb9`; trial ledgers record GPU jobs, adapters and dev evaluations | Supported |
| The outer loop ran for real: Muse Wonder Full revised the researcher pack and the same runner evaluated it at an equal budget | Ledger seq 17 `lab.candidate_proposed`: `cand-001`, lever `budget_use`, lint passed, proposer `muse-wonder-full-1.0`, no fallback; pack commit `3694c668`; trials `t-0003`, `t-0004` with the same 72 GPU-minute / 1.50 USD budget | Supported |
| E1 numbers: original `t-0006` 55.0% / 76.4%; candidate `t-0003` 52.0% / 75.4%, `t-0004` 47.0% / 70.1%; stock 63.0% / 75.4% (29 truncated) | `lab.score_recorded` `t-0006-sel`, `t-0003-sel`, `t-0004-sel`, `sc-0001`: aggregates, n 300, one seed, scorer `sv1-7880d387c32e86d0` | Supported, exact |
| Scoring ran through Inspect AI with recorded provenance | All 9 lab2p scores carry `generator: inspect`, `inspect_ai 0.3.287`, task `glimmer_generate@1`, and the Inspect log sha256 for each seed | Supported. The logs themselves are private: they contain evaluation prompts |
| The scoring code did not change across the scores compared | Every score's `code_commit` falls in `e6daab2`..`82a974b`; over that range `scorer.py`, `verifiers.py`, `rewards.py`, `inspect_generator.py`, `score_session.py` and `generator.py` are unchanged (private-repository diff) | Supported. Not checkable from this extract alone, which has no history |
| The revised pack did not produce better adapters on this evidence | 52.0% and 47.0% against 55.0% | Supported as a description. One original trial against two candidate trials, chosen adaptively: no difference is established either way |
| The held-out H1 comparison was attempted and produced no valid pair | Ledger seq 47 to 56 (below); no H1 selection score exists | Supported |
| Training targets were only verifier-passing samples of the local Glimmer model on approved prompts | `hostkit/lib/data_origin.py` runs inside `glimmer-axolotl-train` before training. The job logs record an "origin check passed" line for each training job | Supported by the private job logs, which are not published |
| GPU spend about 15.73 USD of a 40 USD ceiling | Rental ledgers of all lab roots: lab2p 9.17 (in this extract), lab2m 1.71, lab2n 0.56, lab2o 1.75, plus 2.54 before | Supported as ledger estimates (rental minutes × hourly rate). There is no provider invoice. Only lab2p's rentals are in this extract |
| Every model called at runtime is a Meta model | See [AUTHORSHIP.md](AUTHORSHIP.md); no non-Meta model endpoint appears in `nightshift/` or `scripts/` | Supported |

## Claims that need care or correction

| Claim | Problem | Correction |
|---|---|---|
| "More reliable: 1 of 3 v0 trials in the same lab produced a scored adapter" (submission pack, update 4) | Wrong count. lab2p has four v0 search trials:<br>- `t-0001` no submission<br>- `t-0002` adapter trained, but its selection scoring failed twice on SSH (an infrastructure fault)<br>- `t-0005` runtime crash<br>- `t-0006` scored | "1 of 4". Corrected by Muse in the pack (step 5, exit 0, hashes in `submission-pack/provenance/pack-writer/muse_authored_files.step5.sha256`) and in the run record. The reliability reading is also weaker than it sounds, for two reasons:<br>- `t-0002`'s loss was an infrastructure fault.<br>- The SSH fix was in place before the candidate trials ran. |
| H1 failure causes | `t-0008`: harness guard bug (job errors "AccessDenied: the adapter is not the output of trial 't-0008'"), aborted at seq 49. `t-0009`: contamination scan seq 52 failed, 22 n-gram hits. `t-0010`: operator stop at seq 56 after 3.2 minutes. `t-0011`: never started. | Report each cause separately. None is a 0% score. Under the frozen plan's failure policy they count as 0.0 in its analysis; that is an accounting rule, not a measured quality. |
| `t-0010` and `t-0011` were not run | An operator judgement, not a frozen-rule stop. The frozen contamination scanner flags 51 of the 300 approved H1 training prompts, so further H1 trials would very likely end contaminated. The owner's later pivot (no new experiments or GPU leases) closed the question. | State it as an operator decision with that reason. |
| IFBench test-set score | Never run. The approved stage-3 test scoring was narrowed away by operator change seq 45, recorded before any H1 trial, because the confirmatory final could not finish in time. | Say "not run". The test file is unused. |
| Submission-pack `AUTHORSHIP_AND_LICENSES.md` says the new v0 pack "has not been used by any trial yet"; the pack README names the private repository `meta-hackathon` and says "licence to be chosen" | Stale or private details | v0 ran in every recorded trial. The public repository is `finetuningsingh/nightshift`, and both trees carry an MIT licence. The owner decides whether Muse revises these or a coordinator note stands. |
| `t-0009` H1 dev evaluation (62.0% / 64.5%, n 100) is in the ledger | It is a dev-split number for one trial, not a comparison | Do not quote it as an H1 result. |
