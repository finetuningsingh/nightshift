# Shared notes — Muse Glimmer instruction-following post-training

## Method (five lines)

1. Sample several answers per public training prompt from the base model itself (vLLM, temperature 1.0 / top_p 0.95, reasoning high).
2. Keep only samples that pass every verifiable constraint of their prompt; truncated (`length`) samples are never kept.
3. Build chat-format SFT data from the kept final answers (reasoning trace dropped by default — rationale in the recipe README).
4. Train a QLoRA adapter (Axolotl, based on the public Glimmer example config) on one 80 GB A100.
5. Measure on dev (never train on it), record the experiment below, then change one thing at a time.

## What is known (reference material and environment facts; no new measurements yet)

- Muse Glimmer 30B is a reasoning model; Meta's guide: include reasoning
  traces in SFT targets to preserve chain-of-thought, final-answer-only
  training may suppress it. The starter drops traces for format/length
  reasons (recipe README) — this trade-off is **unresolved**.
- Guide ranges for LoRA/QLoRA SFT: lr 1e-4–3e-4, rank 8–32, alpha 2×rank
  (generic TRL text) — but the Glimmer-specific Axolotl reference config uses
  rank 32 / alpha 16, lr 2e-4, sequence_len 4096, packing off, vision tower
  frozen, gradient checkpointing on. Starter follows the specific reference.
- Guide: Adapter QLoRA at 4096 tokens peaked ~27–29 GB/GPU on H100 in its
  runs; Adam state lands at step 2, so size memory for the second step.
  SFT first, RL (e.g. GRPO with verifiable reward) only from an SFT
  checkpoint with a measured baseline. This starter is the SFT/RFT step.
- Measured environment: one A100 80 GB; vLLM ~1,500 generated tok/s total at
  ~256 concurrent requests, ~600/s at 64; reasoning-high answers are
  typically 1,500–6,000 tokens, some hit a 6,000 cap; ~100 long examples ≈
  8 min/epoch + 3 min load; jobs are hard-killed at the GPU-minute
  allowance, so write partial results early and respect the time box.
- Contracts: verifier judges final-answer `content` only (never reasoning);
  a sample passes only if *every* check is true; serving and training cannot
  share the GPU; only base-model samples from the public training file may
  become training data.
- **Training-data origin rule (harness-enforced, non-negotiable):** every
  training target must be a recorded, *unmodified*, verifier-passing sample
  from the local Glimmer model for an approved prompt — never another
  model's or hand-written text, and never an edited copy (no trim/strip/
  re-wrap/truncate; drop a sample whole instead). `glimmer-axolotl-train`
  checks each SFT row against `/work/data/train.jsonl` +
  `$GLIMMER_OUT/glimmer_samples.jsonl` before training and refuses
  violations; `origin_check.py` runs the same rules locally first. Prompt
  choice, sample counts, decoding, filtering, hyper-parameters and extra
  rounds may all change — this rule may not (full text: recipe README).

## What is unknown and worth measuring (prioritised)

- Base-model pass rate overall and per constraint family at reasoning high —
  and what fraction is lost to truncation at 6,000 tokens (worth a cap sweep).
- Yield curve: verified examples per GPU-minute as samples/prompt and
  concurrency (128 vs measured-best 256) change; where the time box binds.
- Dev accuracy: base vs starter adapter, strict all-constraints rate, plus
  one general-capability signal to catch overfitting regressions.
- Reasoning kept vs dropped (needs sequence_len 8192 + confirmed template
  support for `reasoning_content`): dev effect and training-time cost.
- Epochs (1/2/3), lr within 1e-4–3e-4, LoRA alpha 16 vs 64, keep-per-prompt
  1/2/all, shortest-first vs first-pass selection.
- Whether verifier `error_counts` stay empty on real data; inspect kept
  samples by hand each run for reward-hacking-shaped answers (technically
  passing, substantively empty).

## Experiment template (copy this block per run)

```markdown
### YYYY-MM-DD — <short name> (run/commit: <id>)
- Idea: <hypothesis and why it might help>
- Change: <exactly one change vs named baseline; env/config values used>
- Evidence: <dev metric definition, n, generation settings; yield + timings.json numbers; samples eyeballed?>
- Result: <metric vs baseline, cost in GPU minutes, anything that broke>
- Next step: <keep / revert / follow-up experiment>
```

Rules of the house: never train on dev; never train on another model's
text or on edited copies of samples (origin rule above — the harness checks);
a claim needs a number and a run id; record failures too — a zero-yield
run with its drop-reason breakdown is useful evidence.
