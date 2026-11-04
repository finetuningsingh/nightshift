# Starter recipe: verifier-filtered self-distillation (RFT) for Muse Glimmer

Rejection-sampling fine-tuning: sample answers to public training prompts from
the **base model itself**, keep only samples that pass **every** constraint of
their prompt (checked by the provided verifier), and QLoRA-SFT the base model
on those kept answers. No other model's output, no other data, no downloads.
The dev file is never read by this recipe (measurement only, run elsewhere).

Everything runs with the system `python3` standard library; the only ways this
recipe serves or trains are the harness helpers `glimmer-vllm-up` and
`glimmer-axolotl-train`.

## Training-data origin rule (harness-enforced — do not break)

Training targets must be answers the **local Glimmer model itself** produced
for the approved public training prompts, filtered by the pinned verifier.
Text from any other model (coding agents, hosted models, fixture generators)
or typed by hand must never become a training target. `glimmer-axolotl-train`
enforces this before registering data or training: every SFT row must be
`(system)* user assistant`, the user content byte-for-byte an approved
`train.jsonl` prompt (checked against `/work/data/train.jsonl` — do not point
`TRAIN_FILE` at a different copy for a real run), and the assistant content
exactly a recorded `muse-glimmer` sample (`finish_reason != length`) for that
prompt's key in `$GLIMMER_OUT/glimmer_samples.jsonl`; a `reasoning_content`
field, if present, must equal that sample's recorded reasoning. The harness
then re-runs the verifier and writes `$GLIMMER_OUT/origin_report.json`.
Answers and reasoning are therefore stored and used **unmodified** end to
end — never strip, trim, normalise, re-wrap or truncate a sample; drop it
whole instead. `generate.py` records `glimmer_samples.jsonl` as samples
complete, `build_sft.py` builds only from that record + the approved train
file, and `origin_check.py` applies the same rules locally in stage 4 so a
violation stops the run before training. Changing prompt choice, sample
counts, decoding, filtering, hyper-parameters or adding rounds is fine; the
rule above is not negotiable.

## Stages (`bash run.sh`)

| # | Stage | What happens | Fails loudly when |
|---|---|---|---|
| 0 | Preflight | Checks env/paths/helpers, runs verifier `preflight()` — before any GPU spend | missing `GLIMMER_BASE`/`REWARDS_PY`/train file/helpers; verifier preflight raises |
| 1 | Serve | `glimmer-vllm-up` starts vLLM; printed base URL is validated as `http...` | helper error or no URL printed |
| 2 | Sample | `generate.py`: concurrent chat-completions, partial results appended per sample, wall-clock time box | server `/v1/models` unreachable; zero completions; every request errored |
| 3 | Stop server | `glimmer-vllm-up --stop` (also on every exit path via `trap`) | — warning only |
| 4 | Filter | `build_sft.py`: verifier-filter → `sft.jsonl` + yield stats, then `origin_check.py` applies the harness origin rules locally | zero yield, fewer than `MIN_EXAMPLES`, or any origin violation |
| 5 | Train | `glimmer-axolotl-train config.yaml --data sft.jsonl` | helper exits non-zero |
| 6 | Check adapter | verifies `$GLIMMER_OUT/adapter/{adapter_config.json,adapter_model.safetensors}` (copies them there if training left them elsewhere under `$GLIMMER_OUT`) | adapter pair nowhere to be found |
| 7 | Summary | `summarize.py` → `$GLIMMER_OUT/timings.json` | warning only (adapter stays valid) |

Generation details that matter:

- Prompts are **shuffled (seeded)** before the limit is applied, so a limit
  never means "only the first family in the file".
- Requests are submitted **sample-major** (sample 0 for all prompts, then
  sample 1, ...), so a time-box cut leaves broad prompt coverage.
- Every finished sample is flushed to `generations.jsonl` immediately; if the
  machine is deleted at the GPU-minute allowance, completed work is on disk.
  Each successful response is likewise appended to `glimmer_samples.jsonl`
  (`{key, model, answer, reasoning, finish_reason}`, answer/reasoning exactly
  as returned) — the origin authority for filtering and for the harness
  check. Errored requests are never recorded there.
- Per-request retries: transient errors (5xx, connection) retry twice; 4xx
  does not. Errors are recorded per sample, never silently dropped.
- The reasoning trace is read from `reasoning_content` or `reasoning`
  (whichever the server sends) and stored in `generations.jsonl` for analysis.

Filter keep-rule (all must hold): no request error, `finish_reason != length`
(truncated samples are **never** kept, even if the verifier passes them),
non-empty final answer, `verify_constraints(content, ground_truth)` returns a
non-empty list of all `True` (an empty list is *not* a pass), prompt+answer
fits `MAX_SFT_CHARS`. Exact duplicates per prompt are dropped; per prompt the
**shortest** passing answers are preferred (cheapest to train, least
rambling) up to `MAX_KEEP_PER_PROMPT`. `error_counts()` from the verifier is
reported in the stats and on stderr if non-empty.

## Reasoning trace in training targets: dropped, by default

Meta's guide says to include the reasoning trace (via `reasoning_content`)
if you want the model to keep reasoning, and warns that training only on
final answers may suppress chain-of-thought. The starter nevertheless trains
on **final answers only** (`KEEP_REASONING=0`), because:

1. The harness's training format is messages of `{"role", "content"}`. A third
   key's handling by the chat template is not in the contract — it might be
   ignored silently, giving final-answer training under a false label, or
   rejected. Putting the trace *inside* `content` would be worse: it teaches
   the model to emit unconstrained reasoning in the scored channel, where
   constraints like "no commas", word counts and all-lowercase apply to the
   whole `content` the verifier sees.
2. Traces here are long (typical answers run 1,500–6,000 tokens at reasoning
   "high"). At the reference `sequence_len: 4096`, keeping traces would
   truncate many examples before the final answer and its end-of-turn token,
   and would multiply training time per example.
3. The verifier — and presumably scoring — judges the final answer only, so
   final-answer targets teach exactly the judged behaviour.

The known risk (suppressed chain-of-thought could hurt constraints that need
planning, e.g. counting) is real and unmeasured; see "Most promising changes".
`KEEP_REASONING=1` adds `reasoning_content` to the assistant message for that
experiment — raise `sequence_len` (e.g. 8192) and `MAX_SFT_CHARS` with it, and
first confirm the template actually consumes that field.

## Decoding settings

`temperature=1.0`, `top_p=0.95` — the sampling values in Meta's post-training
guide (its post-fine-tune evaluation snippet uses exactly these); the guide
excerpt provided here states no other decoding recommendation for sampling.
`top_k` is not sent unless `TOP_K>0`. `reasoning_strength="high"` is sent via
`chat_template_kwargs`, matching how the environment's measured answer
lengths were obtained. A per-sample `seed` (seed + sample index + prompt
hash) makes runs reproducible and samples distinct. `max_tokens=6000` matches
the observed "some hit a 6,000-token limit" scale; truncated samples are
discard, never trained on.

## Knobs (environment variables; `run.sh` passes them through)

| Variable | Default | Meaning |
|---|---|---|
| `SMOKE` | `0` | `1` = smoke mode (below) |
| `PROMPT_LIMIT` | `250` | prompts sampled (after seeded shuffle); `0` in `generate.py` CLI = all |
| `SAMPLES_PER_PROMPT` | `4` | samples per prompt in generation order sample-major |
| `CONCURRENCY` | `128` | simultaneous generation requests (measured: ~600 tok/s @64, ~1,500 @256) |
| `MAX_TOKENS` | `6000` | per-sample generation cap; `finish_reason=length` is always dropped |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | `1.0` / `0.95` / `0` | decoding; `TOP_K=0` means not sent |
| `REASONING_STRENGTH` | `high` | `chat_template_kwargs`; empty string omits it |
| `GEN_TIMEBOX_SECONDS` | `2700` | generation wall-clock budget; keeps what finished |
| `GEN_GRACE_SECONDS` | `180` | after the time box, wait this long for in-flight requests, then abandon |
| `REQUEST_TIMEOUT_SECONDS` | `1200` | per-request HTTP timeout |
| `VLLM_PORT` / `MAX_MODEL_LEN` / `VLLM_WAIT_SECONDS` | `8000` / `8300` / `900` | passed to `glimmer-vllm-up` |
| `MODEL_NAME` | `muse-glimmer` | served model name |
| `SEED` | `42` | shuffle, sampling seeds, SFT shuffle; training seed is in `config.yaml` |
| `RESUME` | `0` | `1` = skip `(key, sample_idx)` already in `generations.jsonl` instead of truncating |
| `MAX_KEEP_PER_PROMPT` | `2` | kept verified samples per prompt (shortest first, exact-deduped) |
| `MAX_TRAIN_EXAMPLES` | `800` | overall SFT cap, applied round-robin across prompts |
| `MIN_EXAMPLES` | `20` | below this many verified examples, refuse to train |
| `MAX_SFT_CHARS` | `14000` | prompt+answer char guard ≈ `sequence_len 4096` tokens |
| `KEEP_REASONING` | `0` | `1` = add `reasoning_content` to SFT targets (see above) |
| `TRAIN_FILE` / `GLIMMER_DATA_DIR` | `/work/data/train.jsonl` | public training prompts (dev is never used) |
| `GLIMMER_OUT` | `/work/out` | outputs (set by the harness) |

Harness-set (not knobs): `GLIMMER_BASE`, `REWARDS_PY`, `IFEVALG_PATH`,
`GLIMMER_CONFIG_TEMPLATE` (ignored; the recipe carries its own config).

Smoke overrides (each overridable): `SMOKE_PROMPT_LIMIT=4`,
`SMOKE_MAX_TOKENS=1024`, `SMOKE_REASONING_STRENGTH=low` (so a final answer
fits inside the small token limit), `SMOKE_GEN_TIMEBOX_SECONDS=600`,
plus 1 sample/prompt, `MIN_EXAMPLES=1`, and a generated
`$GLIMMER_OUT/config.smoke.yaml` with `max_steps: 1`.

## Smoke mode

```bash
SMOKE=1 bash run.sh
```

Runs every stage on at most 4 prompts × 1 sample, 1,024-token generation
limit, one training step. Purpose: find broken code/paths/config for a few
GPU minutes. Caveat: with 4 single samples the verified yield can
legitimately be zero — that now-familiar `FATAL: zero yield` message means
stages 0–4 worked; raise `SMOKE_MAX_TOKENS` or rerun to also exercise
training. The smoke adapter is meaningless as a model.

## Training config (`config.yaml`)

Based on the public Axolotl Muse Glimmer QLoRA example
(`refs/axolotl_muse_glimmer_qlora.yaml`); every deliberate change is also a
`# CHANGED:` comment in the file:

| Change | Why |
|---|---|
| dataset → one local `chat_template` entry; dropped `split: train[:1%]` and the vision-mix path | the split would silently discard 99% of our data; harness requires exactly one entry |
| `roles_to_train: ["assistant"]` stated explicitly | harness default; train on answers, not prompts |
| `eot_tokens: ["<|eot|>", "<|eom|>"]` (reference: only `<|eot|>`) | matches the harness default; explicit beats implicit |
| all `wandb_*` keys removed | harness forces W&B off; no credential/network on the GPU box |
| `num_epochs: 1 → 2` | a few hundred short examples, not a large vision mix; the guide's LoRA/QLoRA range is 2–5 |
| `seed: 42` added | the guide's controlled runs use seed 42; reproducibility |

Kept as-is: 4-bit QLoRA, `lora_r=32` / `lora_alpha=16` / dropout 0.05 (the
guide's *generic* TRL text suggests alpha = 2×rank; the Glimmer-specific
reference uses 16, so 16 stays — it is an experiment knob), language-only
target modules (vision tower frozen), `sequence_len: 4096`, packing off,
`skip_prepare_dataset: true`, micro-batch 1 × accumulation 4,
`adamw_bnb_8bit`, lr 2e-4 cosine (inside the guide's 1e-4–3e-4 range),
bf16/tf32, gradient checkpointing, flash-attention-2, warmup 0.1.

## Files produced (in `$GLIMMER_OUT`)

- `adapter/adapter_config.json`, `adapter/adapter_model.safetensors` — the deliverable
- `timings.json` — stage seconds, generation counts, yield (overall + per family), verifier `error_counts`
- `generations.jsonl` — every completed sample incl. reasoning, finish reason, usage, errors
- `glimmer_samples.jsonl` — the origin record: exact server answers/reasoning per sample (harness-checked authority)
- `origin_report.json` — written by the harness on a successful origin check (sha256 of prompts/samples/data/verifier)
- `gen_stats.json`, `sft.jsonl` (the exact training file), `sft_stats.json` (drop reasons, per-family pass rates)
- `vllm.log` (written by the helper), `config.smoke.yaml` (smoke only)

## Expected duration / GPU cost (estimates, not measurements)

Using only the provided environment facts (throughput ~600 tok/s @64 and
~1,500 @256 concurrent requests; answers 1,500–6,000 tokens at reasoning
high; ~8 min/epoch per 100 long examples + ~3 min load):

- **Generation**: 250 prompts × 4 samples = 1,000 requests. At an assumed
  ~2,500 tokens average ≈ 2.5M tokens; at an interpolated ~1,000 tok/s for
  concurrency 128 ≈ **~40 min**, hard-capped by the 45-min time box.
- **Filtering**: CPU-only, seconds to a couple of minutes (verifier-bound).
- **Training**: kept examples are final answers only — much shorter than the
  "100 long examples" calibration — so a few hundred examples × 2 epochs
  should be well under that calibration's per-100 rate; budget **~10–20 min**
  including model load. This is the least certain number here.
- **Total ≈ 1 GPU-hour per full run** on one A100, dominated by generation.
  Smoke mode: a few GPU minutes, dominated by server/model loads.

Actual pass rates, yields and timings are unmeasured until the first real
run; `timings.json` and `sft_stats.json` exist to replace these estimates.

## Known limitations

- No token-level length knowledge in the stdlib filter: `MAX_SFT_CHARS` is a
  characters-per-token approximation of `sequence_len`. Some examples may
  still be truncated by the trainer, and some usable long ones are dropped.
- Pass/fail is only as good as the provided verifier; swallowed checker
  exceptions are visible via `error_counts` but their affected samples are
  not identified.
- Shortest-first selection biases toward terse answers; capped per prompt at
  2, so easy prompts do not flood the set, but hard prompts with one lucky
  pass contribute equally with easy ones.
- One epoch-equivalent of evidence: nothing here measures dev accuracy.
  Evaluate the adapter on `dev.jsonl` outside this recipe before believing
  any improvement, and watch a general-capability signal for regressions
  (per the guide).
- `reasoning_strength: low` in smoke mode and the `reasoning_content`
  message field under `KEEP_REASONING=1` are inferred counterparts of the
  documented `"high"` / guide usage, not verified against the real template.

## Most promising changes to try (in rough priority)

1. **Reasoning traces on/off** (`KEEP_REASONING=1`, `sequence_len` 8192,
   matched `MAX_SFT_CHARS`) — the guide's warning about suppressing
   chain-of-thought is the biggest unmeasured risk in the starter.
2. **Scale generation where yield is low**: read `sft_stats.json` per-family
   pass rates; raise `SAMPLES_PER_PROMPT`/`PROMPT_LIMIT`/`CONCURRENCY=256`
   (measured best throughput) rather than training longer on few prompts.
3. **Epochs / LR / LoRA alpha**: 1 vs 2 vs 3 epochs, lr 1e-4–3e-4, and
   `lora_alpha` 16 (reference) vs 64 (2×rank, generic guidance) — small-data
   overfitting vs undertraining is unmeasured.
4. **Selection policy**: `MAX_KEEP_PER_PROMPT` 1 vs 2 vs all, and
   shortest-first vs first-passing — diversity/balance effects on constraint
   families, measurable from `generations.jsonl` without new GPU generation.

## Tests (no GPU)

```bash
python3 -m unittest discover -s tests -v   # from starter/recipe/
```

Fake OpenAI-compatible HTTP server + fake verifier cover generation (both
reasoning field names), time-boxing, filtering (truncated, failed, vacuous,
duplicate, caps), SFT format, zero/low-yield and unreachable-server failure
messages, summarize output, and config invariants. Origin coverage: exact
`glimmer_samples.jsonl` recording (stubbed transport, untrimmed text),
builder use of the train/samples files as authority, and `origin_check`
refusing a hand-written answer, a changed prompt, a trimmed answer, a
truncated or other-model sample, mismatched reasoning, and extra turns. In
a sandbox that forbids binding sockets, only the loopback-server tests skip
(with the reason).
