# BUILD_NOTES — starter training recipe (Assignment 1)

## Files

| File | Purpose |
|---|---|
| `starter/recipe/run.sh` | Entry point: strict bash, 8 timestamped stages, env knobs, SMOKE mode, `trap` stops vLLM on every exit |
| `starter/recipe/generate.py` | Stdlib-only sampler: urllib + thread pool, seeded shuffle, sample-major order, bounded concurrency, time box + grace, per-sample flush, retries, both reasoning field names |
| `starter/recipe/build_sft.py` | Verifier preflight + filtering (truncated/failed/vacuous/empty/duplicate/too-long drops), per-prompt cap (shortest first), round-robin overall cap, SFT JSONL + yield stats, zero/low-yield hard failures; `--preflight-only` for stage 0 |
| `starter/recipe/summarize.py` | Merges gen/filter stats + stage durations into `timings.json` |
| `starter/recipe/config.yaml` | Axolotl QLoRA config based on `refs/axolotl_muse_glimmer_qlora.yaml`; every deliberate change marked `# CHANGED:` and tabulated in the README; SMOKE `max_steps` anchor line |
| `starter/recipe/README.md` | Stages, all knobs/defaults, reasoning-trace decision, decoding rationale, outputs, cost estimates, smoke, limitations, top experiments |
| `starter/recipe/origin_check.py` | Local training-data origin check (harness rules 2–5): `check_sft_origin()` + CLI (added in Assignment 2) |
| `starter/recipe/tests/test_pipeline.py` | 15 stdlib `unittest` tests with a fake OpenAI-compatible HTTP server and a fake verifier |
| `starter/notes/seed.md` | Team notes seed: method in 5 lines, known/unknown, experiment template |

## How it was tested (all local, no GPU)

- `python3 -m unittest discover -s starter/recipe/tests` → **OK (10 tests, 3 skipped here)**.
  Passing here: filtering incl. truncated-never-kept, multi-constraint all-must-pass,
  vacuous empty-check rejection, dedup + shortest-first cap, empty/short drops,
  harness SFT message format (exactly `{role, content}` by default,
  `reasoning_content` only under the flag), zero-yield and below-minimum
  failure messages, verifier preflight/missing-module failures, prompt-file
  validation, config invariants (one dataset entry, no 1% `split`, no active
  `wandb_*`, smoke anchor), and unreachable-server failure.
  Skipped here (3): generation, time-box, and end-to-end tests that bind the
  fake HTTP server — this sandbox forbids binding *any* socket
  (`PermissionError`, loopback and unix alike). The tests skip **only** on
  that bind error, with the reason printed, and run normally elsewhere.
- Same generation/time-box scenarios re-run through a stubbed transport
  (scratch script, not shipped): both `reasoning_content` and `reasoning`
  fields extracted, request payload asserted (temperature/top_p/seed/
  `chat_template_kwargs`), time box returned partial results in 1.0 s with
  3 of 4 requests abandoned and `timed_out: true`.
- `run.sh` exercised end-to-end against stub `glimmer-vllm-up` /
  `glimmer-axolotl-train` on PATH (SMOKE=1, sample train file, fake verifier,
  dead server URL): preflight passed, stage banners printed, generation
  failed with the intended `FATAL` message, the trap invoked
  `glimmer-vllm-up --stop`, training was never called, exit code 1.
- `bash -n run.sh`, `python3 -m py_compile` on all scripts, and the SMOKE
  `sed` substitution (`max_steps: 1` exactly once) verified.

## What could not be tested without a GPU / the real harness

- Actual `glimmer-vllm-up` output shape (the recipe takes its last stdout
  line as the base URL and validates it is `http...`), real vLLM response
  fields and throughput, and whether `glimmer-vllm-up --stop` fully frees
  GPU memory before training.
- Real verifier behaviour on real `ground_truth` strings (only the documented
  interface was coded against) and whether `error_counts` stays empty.
- Axolotl accepting `config.yaml` as overridden by the harness, 4096-token
  memory/step-time on the A100, and where exactly the adapter lands
  (stage 6 searches `$GLIMMER_OUT` and copies if needed).
- Any model-quality effect: pass rates, yields, dev accuracy. All duration
  figures in the README are estimates from the provided facts.
- `config.yaml` was not parsed by a YAML library locally (none installed;
  recipe code itself needs only stdlib). Structure mirrors the reference
  config and is checked by text invariants in the tests.

## Assumptions made (flagged in the files where they bite)

- Decoding temperature 1.0 / top_p 0.95 taken from the guide's evaluation
  snippet — the excerpt contains no other sampling recommendation.
- `reasoning_strength: low` (smoke only) and `reasoning_content` as an
  optional message field (`KEEP_REASONING=1`) are inferred counterparts of
  documented features; defaults avoid both.
- Training targets drop the reasoning trace (rationale: README). The
  character-based `MAX_SFT_CHARS` guard approximates tokens.
- The real `train.jsonl` size is unknown; defaults (250 prompts × 4 samples,
  45-min generation time box) target ≈1 GPU-hour and degrade gracefully
  (shuffled, sample-major) if the file or answers are larger than assumed.

## Open risks

- Final-answer-only SFT may suppress chain-of-thought and hurt
  planning-heavy constraints (guide's warning) — first experiment to run.
- If the base pass rate is very low, defaults may hit the zero/low-yield
  gate by design; the messages say which knob to raise, but the first real
  run should be SMOKE, then a small (`PROMPT_LIMIT`/`GEN_TIMEBOX_SECONDS`)
  probe reading `sft_stats.json` before a full run.
- If the harness's chat template *requires* reasoning tokens to separate
  `content` correctly at evaluation time, adapter behaviour could differ
  from generation-time behaviour here; only a real dev evaluation detects it.
- Shortest-first selection could bias toward degenerate minimal answers that
  technically pass; kept samples must be eyeballed (`generations.jsonl`).

## Assignment 2 — training-data origin contract

The harness now checks training data inside `glimmer-axolotl-train` (before
registering/training): each SFT row's user content must be byte-identical to
an approved `/work/data/train.jsonl` prompt, the assistant content must
exactly equal a recorded `muse-glimmer` sample (`finish_reason != length`)
in `$GLIMMER_OUT/glimmer_samples.jsonl`, any reasoning field must equal that
sample's reasoning, and the verifier is re-run. Recipe changes:

| File | Change |
|---|---|
| `recipe/generate.py` | Records `glimmer_samples.jsonl` (`{key, model, answer, reasoning, finish_reason}`) as each successful response completes; `extract()` returns server values unmodified (no `""` coercion, absent reasoning = `null`); errored requests are never recorded; new `--samples-out` (defaults next to `--out`) |
| `recipe/build_sft.py` | Exact text throughout (`.strip()` removed from answer path; length guard drops whole, dedup hashes exact bytes); new `--train-file`/`--samples` make the approved train file + samples record the authority (prompt/ground_truth/reasoning from them; unrecorded answers dropped as `not_a_recorded_sample`, unknown keys as `key_not_in_train`) |
| `recipe/origin_check.py` (new) | `check_sft_origin()` applying harness rules 2–5 exactly + a CLI (`--sft/--train/--samples`, exit 3 on violation) |
| `recipe/run.sh` | Passes `--samples-out $OUT/glimmer_samples.jsonl` to generation, `--train-file/--samples` to the builder, and runs the origin check at the end of stage 4 (run stops before training on failure) |
| `recipe/README.md`, `notes/seed.md` | Origin rule documented (what may change vs what may not) |
| `recipe/tests/test_pipeline.py` | +5 tests (15 total): origin accept/refuse incl. negative hand-written-answer and changed-prompt cases, trimmed-answer refusal, samples-file recording via stubbed transport, builder authority incl. tampered-prompt repair to the approved prompt |

Tested: `python3 -m unittest discover -s starter/recipe/tests` → OK
(15 tests, same 3 loopback-socket skips as before); `bash -n run.sh`;
`py_compile` on all scripts; `origin_check.py` CLI exit 0/3 on good/bad
fixtures. Not testable here: the harness's own in-training check and
`origin_report.json` (no harness/GPU on this machine). One operational note:
the harness reads `/work/data/train.jsonl` itself, so a real run must not
point `TRAIN_FILE` at a different copy (README says so too).
