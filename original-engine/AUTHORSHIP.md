# Authorship, tools and dependencies

## Who wrote what

| Part | Written by |
|---|---|
| The original engine: `nightshift/`, `nightshift/lab/`, hostkit, tests, scripts, the design documents in `docs/` | Ankit Singh, building with the AI coding assistant **Claude Code** (Anthropic), from 2026-10-26. Commits marked `Claude <noreply@anthropic.com>` were made by Claude Code sessions. The organisers allowed this: "Build with whatever tools you want, but every model your project actually calls at runtime has to be a Meta model!" (Kai, Cerebral Valley, hackathon Discord, 2026-10-26). No Claude model is called at runtime. |
| Researcher starter pack v0, `nightshift/lab/assets/pack_muse_v0/` (except `task.yaml`, which the harness owns) | **Muse Code 1.4.14, `muse-wonder-full-1.0`**, in a fresh session that never saw earlier starter code; prompts, step records and file hashes in `submission-pack/provenance/starter-author/` |
| Submission documents in `submission-pack/` | **Muse Code 1.4.14, `muse-wonder-full-1.0`**, steps 1 to 4, with one mechanical coordinator correction recorded in `submission-pack/provenance/pack-writer/COORDINATOR_EDITS.md`; hashes in `submission-pack/provenance/` |
| The `cand-001` researcher-pack revision | **`muse-wonder-full-1.0`** acting as the outer-loop reviser at run time (ledger event seq 17) |
| Training decisions inside each trial | **`muse-spark-1.3`** researcher agents at run time |
| `README.md`, `AUTHORSHIP.md`, `SETUP_AND_REPLAY.md`, `CLAIM_AUDIT.md`, `RESULTS_AND_EVIDENCE.md`, `scripts/replay_recorded_study.py` and the sanitized evidence summary in this folder | Claude Code, on 2026-11-04, from the recorded ledgers |

## Models the project calls at runtime (all Meta)

| Model | Role | How it is reached |
|---|---|---|
| `muse-spark-1.3` | Inner-loop researchers | Muse Code 1.4.14 CLI (subscription login; API keys stripped from the environment) |
| `muse-wonder-full-1.0` | Outer-loop reviser | Muse Code 1.4.14 CLI |
| Muse Glimmer 30B, `meta-models/Muse-Glimmer-30B@a4e59da52a7bc87ae7251dd5545c0dd437c44b68` | The model being post-trained and scored | vLLM on rented A100 80GB GPUs |

The older `runtime_coral.py` path (opencode plus a metering gateway) is configured only for a `muse` provider; the recorded study used `coral+muse_code+docker` with `muse-spark-1.3`, as each trial's ledger record shows. The alias `CoralCliRuntime` in `runtime_coral_cli.py` is a leftover from an earlier plan in which Claude Code would be the researcher. Before the first trial, the owner changed the researchers to `muse-spark-1.3`. Every lab root was run by `scripts/two_loop/run_lab.py`, which sets the Muse Code runtime (`coral_muse`) and the Muse CLI proposer.

## Prior work this builds on (pinned, not authored here)

| Dependency | Pin | Use |
|---|---|---|
| CORAL (Human-Agent-Society/CORAL) | commit `0123dfb939b35228cf2c1fde224cd0561e727408` | Researcher workspaces and shared notes; installed on the researcher host, not vendored |
| WeCo AIDE² write-up | method inspiration only | The outer-loop idea; no WeCo code or product |
| Inspect AI | `inspect_ai` 0.3.287 (recorded in every score) | Score generation task `glimmer_generate@1` |
| IFEval-G verifiers (allenai/open-instruct) | commit `b781e782a8fc530577f235aefe282dd046cb8ad2`; file hashes in each score's `verifier_pins` | Constraint checking; fetched by `scripts/lab_setup_verifiers.sh`, not vendored |
| IFBench | file hashes in `scripts/lab_pins.json` | Test file (never scored in this study) |
| vLLM, axolotl, torch | `vllm==0.32.0`, `axolotl==0.19.0`, `torch==2.12.1` (hostkit) | Serving and QLoRA on rented GPUs |
| Thunder Compute | provider API | GPU rentals |
