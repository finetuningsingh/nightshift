---
type: results
title: Paired reference experiment — results
description: Measured outcome of the one bounded rental that compared stock Muse Glimmer with the reference rejection-sampling LoRA on 150 E1 dev prompts through the harness scorer and ledger, with cost, cleanup and what the result does and does not show.
timestamp: 2026-11-03T00:15:00-05:00
---

# Paired reference experiment — results

Plan and authority: the costed plan is kept privately (it records coordination decisions). This was the **non-Muse** checkpoint comparison; it is not the Spark/Wonder two-loop product and says nothing about researcher improvement.

## Result

Exploratory paired comparison on the public E1 dev split (150 prompts, one sample each, `max_tokens` 6000, reasoning high, same server, same decoding), recorded as ledger scores `sc-0001` (stock) and `sc-0002` (adapter `ref-rft-1`):

| Arm | Pass all constraints | Mean constraint fraction | Truncated at 6000 tokens |
|---|---|---|---|
| Stock Glimmer | 63.3% (95/150) | 75.6% | 14 |
| Reference-method LoRA | 60.7% (91/150) | 70.7% | 25 |

Difference (adapter minus stock): **−2.7 points**, 95% paired bootstrap interval −6.7 to +1.3, exact McNemar p = 0.34 (adapter-only passes 3, stock-only passes 7). The adapter did **not** improve on stock here; the difference is not distinguishable from zero at this sample size, and it truncated more often. This is a negative-to-null result and is reportable as such.

What it does not show: that the method cannot work. This was one run with one small training set and one hyper-parameter setting chosen in advance (see below). It is also not an IFBench score (Meta's 77.0 is not comparable), not a held-out confirmation (E1 dev is public to researchers by design) and has roughly ±8 point resolution at n = 150.

## What ran

- **Generation:** Glimmer sampled 2 answers to each of 100 E1 train prompts (first 100 rows in file order), `max_tokens` 6000, temperature 1.0, top-p 0.95, top-k 64, reasoning high, 1,348 s, 0 errors.
- **Filter (pinned IFEvalG checkers on the control host):** 113 of 200 samples passed every constraint (yield 56.5%); 31 were truncated and dropped; 112 distinct answers from 60 prompts were kept (two per prompt where both passed), each with its reasoning. No hosted-model output was used.
- **Training:** Axolotl 0.19.0 QLoRA, 1 epoch, 28 steps, lr 2e-4, rank 32, sequence length 7168, 465 s, train loss 0.4155, peak 26 GiB. Hyper-parameters are the recipe's, fixed before the run; nothing was tuned on the result.
- **Reload and serve:** listing hash `04db7ce7…` matched on the rental and the control host; PEFT reload check passed (40 s); vLLM served base and adapter together.
- **Scoring:** through the real lab (`score` / `score --stock`, pinned verifiers, ledger guard). `lab verify` passes (ledger head `8:ab6542da…`). The stock score ran concurrently with generation on the same server; the adapter score ran alone. Rental cost is booked in the rental ledger; the score-cost events record the scoring window at the rental's rate.
- **Exposure:** only E1 train (100 rows) and E1 dev (150 rows) were used; selection, H1 and the IFBench test file were not scored or consulted. The 348 smoke-run rows were excluded from the split source before it was frozen.

## Cost, timing and cleanup

| | |
|---|---|
| Rental | one A100 80GB, handle `eys090qi`, created 04:03:45Z, deleted 05:01:42Z (58 min) |
| Billed | 59 minutes, **$1.10** (limit $3.00 / two hours; planned $2.0–2.1) |
| Cumulative Thunder spend | $1.44 (first run) + $1.10 = **$2.54** of the $40 ceiling |
| Setup | 3 min (vLLM env 58 s, weights 56 GB in 89 s, Axolotl env in parallel) — far faster than the 16 min planned |
| Throughput | 1,500–1,650 generated tokens/s at 256 concurrent sequences, versus about 600 measured at 64 concurrent in the first run |

Cleanup: the instance was deleted by the **off-host stop on ax41** (`gpu_watchdog reap` with ax41's own Thunder login, against the replicated ledger, exact handle and ownership tag), which proves that path on a real instance. `tnr status` then showed no instances; the ledger records the rental deleted; the control-host watchdog exited on its own. The account held no other instances during the run and none was touched. Results were copied to the control host before deletion: the adapter, the samples, the SFT file, timings, training and setup logs, and (in the lab root) both score records.

## Findings worth keeping

- The harness path works end to end on a real GPU for scoring: `AttachedModelServer` and `score --stock` are new, tested, and rehearsed against a fake endpoint before use.
- Scoring 150 prompts per arm with 128 generator workers took about 12–20 minutes including the long-answer tail (the generator's default is 8 workers, which would be far slower; the production `build_world` still uses it).
- `reasoning_content` in SFT rows is rendered by Glimmer's chat template as the `to=self` message, and trained without error.
- Stock Glimmer on the E1 evolution families passes 63% on dev (9% truncated at 6000 tokens): headroom exists on this distribution, and the first run's random mix, which included the copy and counting families, passed only 47.7%.
- The adapter's higher truncation rate (25 vs 14) is consistent with SFT on long reasoning traces teaching longer reasoning; a next attempt would filter for shorter successful traces or train on answers only. That is a hypothesis, not tested.
- No Muse model, CORAL or opencode ran; the terms waiver remains `open_waived`.
