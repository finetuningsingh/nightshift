---
type: policy
title: Experiments run through Inspect AI
description: The owner's rule that model experiments run through Inspect AI with a pinned, versioned dataset and recorded provenance, what is implemented, how it was checked against the ledger scorer, and what is still outside it.
timestamp: 2026-11-03T20:10:00-05:00
---

# Experiments run through Inspect AI

**Rule (owner, 2026-11-03):** model experiments run through Inspect AI (`inspect_ai`), with discipline about datasets, hygiene and versioning for everything. The first live runs (the 2026-11-02 smoke run and the paired reference experiment) did **not** use Inspect: they scored through the lab's own `ScoreService`. This document records the correction.

## What is implemented

`nightshift/lab/inspect_eval.py` (optional dependency, `pip install -e .[lab-eval]`) and `scripts/inspect_run.py`:

| Concern | Rule |
|---|---|
| Dataset | Read only from the lab root; the file's sha256 must equal the revision frozen in the assignment manifest, or the run refuses ("changed after the freeze"). Same strict row checks as the scorer. Dataset name carries the hash; sample ids are the ledger row keys. |
| Splits | DEV only. Selection, held-out and test data stay behind the ledger guard of the `score` command; this module never opens them. |
| Scoring | The pinned `IfEvalG` backend that `score` uses; per-sample `prompt_all`, `fraction`, `truncated`. A truncated generation is scored as empty text, the harness rule. |
| Decoding | `GenerationParams.recommended`, one explicit system message, seed 0, and the request body checked against the harness generator's (including `top_k` and `chat_template_kwargs.reasoning_strength`; an OpenAI-style provider silently drops `top_k` unless it is sent in `extra_body`, which the rehearsal caught). |
| Task version | `TASK_VERSION` is bumped whenever the prompt, decoding, cleaning or scoring rule changes. |
| Provenance | Every log records git commit and dirty flag, lab id, assignment, dataset sha256, `scorer_version`, verifier pins, base revision, adapter listing hash, rental id, decoding, and library versions. A run on a dirty tree is refused unless `--allow-dirty`, which is then visible in the log. |
| Logs | `LAB_ROOT/inspect_logs/*.eval`; the runner prints each log's sha256 so it can be cited. |

## The ledger `score` command now generates through Inspect (2026-11-03)

`nightshift/lab/inspect_generator.py`: `InspectGenerator` is the `Generator` that `score` uses by default (`cli.build_world`; `scripts/lab_attached_score.py` uses it too). `ScoreService` is unchanged in what it owns: access table and guard, frozen data revisions, no-re-roll, the pinned verifiers, the per-prompt files and the `ScoreRecord`. Only the model calls moved into an Inspect task.

| Concern | Rule |
|---|---|
| Task | `glimmer_generate`, `TASK_VERSION` 1 (bump when the system message, request fields or the output-to-`GenResult` mapping change). One Inspect evaluation and one `.eval` log per seed. |
| Dataset | The exact prompts the service loaded after its own hash check against the manifest; the dataset name carries the sha256 of the ordered ids and texts, also stored in the log metadata. |
| Request | Same body as `HttpGenerator` (one explicit system message, temperature, top_p, top_k, max_tokens, seed, `reasoning_strength`), checked in the tests from the log's recorded config and on the wire against the fake vLLM. |
| Failure | A failed sample becomes a generation error (the service then refuses to score from missing answers); an evaluation that ends in error raises; 16 failed samples stop a dead endpoint, as `HttpGenerator`'s rule does. |
| Evidence in the record | `ScoreRecord.verifier_pins` gains `generator`, `inspect_ai`, `inspect_task`, `inspect_log_seed<N>` (file name), `inspect_log_sha256_seed<N>`, `code_commit`, `code_dirty`. The plain HTTP loop is a named choice (`NIGHTSHIFT_LAB_GENERATOR=http`) and records `generator: http`; a missing `inspect_ai` is an error, not a silent fallback. |
| Where logs go | `LAB_ROOT/private/inspect_logs/`. Logs of guarded splits contain private prompts and answers, so they live in the private directory, which is never copied to a researcher host. |

Checked: unit tests (mapping, ordering, truncation, hashes, request config, record pins, the generator-without-evidence case), and the real `score` command against a local fake vLLM on a copy of the rehearsal lab: the server received the expected body, the record carried three log hashes, and `verify` passed. The two ledger scores of the reference experiment (`sc-0001`, `sc-0002`) predate this and have no `generator` pin; they were produced by the plain HTTP loop.

## Check against the ledger scorer

`scripts/inspect_run.py replay` re-scores the stored generations of a ledger score with the Inspect scorer and compares every prompt with `per_prompt.jsonl`. For the paired reference experiment, `sc-0001` (stock) and `sc-0002` (adapter), **300 of 300 prompts agree** on pass-all, constraint fraction and truncation (the first attempt found one mismatch, the truncation rule, which was fixed). The live path (`run`) was rehearsed against a local fake OpenAI-compatible endpoint only; it has not met a real vLLM yet.

## Not yet inside Inspect

- Verification still runs in `ScoreService` on the generations the Inspect task produced (the guard needs the private ground truth to stay in the service); the dev-only `scripts/inspect_run.py run` also scores inside Inspect. A shared scoring rule, checked by the 300/300 replay, keeps the two consistent.
- The live path has not met a real vLLM. The first live Inspect log must be read to see how the provider returns a reasoning answer that vLLM splits into `reasoning_content` (only the final answer is used) and whether the server honours `top_k` and `chat_template_kwargs`.
- Generation for the reference method (`rft_pipeline.py`) and the training run are not Inspect tasks; their inputs and outputs are versioned by file hash in the results documents.
- IFBench test scoring uses the upstream IFBench verifiers and is not wrapped yet.
