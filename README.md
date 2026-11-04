# Nightshift

*Research agents that post-train a small open model overnight, and an outer
agent that improves the researchers.*

Nightshift is a two-level post-training system with a trusted,
deterministic controller at its centre.

- **Inner loop (trial).** A small team of researcher agents collaborates on
  one post-training task inside one trial. Each researcher has an isolated
  private workspace; the team shares a notes area that exists only inside
  that trial, following the CORAL collaboration pattern. Researchers plan,
  request sample generation from a local open model on approved training
  prompts, filter samples with training-only verifiers, request a QLoRA
  fine-tune, submit a checkpoint, and receive public development-set
  feedback, all within a fixed per-trial budget.
- **Outer loop (round).** An outer reviser agent reads permitted public
  traces and feedback from finished trials and edits the *researcher pack*
  (instructions, tools, recipe template, collaboration settings, budget
  habits, initial notes). The controller freezes the edited pack as a new
  version, runs fresh trials with it under identical constraints, and
  screens it against the incumbent on a private selection split. This
  follows the published AIDE² recipe of improving the researcher rather
  than the artefact directly.
- **Final comparison.** After the rounds, the frozen original pack and the
  frozen final pack are compared in fresh matched trials on an untouched
  private test split. The question is whether Nightshift's improved
  supervised fine-tuning (SFT) procedure beats the original SFT procedure
  from the same base checkpoint under matched resources — not whether
  either beats the untrained base model.

The controller (plain deterministic code, not a model) owns everything
that must not be influenced by agents: GPU provisioning and cleanup,
budgets, private data, scoring, and the event ledger.

## Status

The CPU demo and test suite run end to end with labelled deterministic
fakes and real Inspect AI scoring against a mock model. No live GPU
training result is claimed by this repository: real provider, generation,
training, and agent adapters are marked *not yet validated live* in their
docstrings until a recorded live run exercises them. Every record from a
fake run carries `"fake": true`, and its report states "no evidence of
improvement" by construction.

## Quick start (CPU demo, no secrets, no GPU)

```sh
pip install -e ".[test]"
python -m pytest -q
nightshift demo --root ./demo-out
nightshift status ./demo-out/run
nightshift ledger verify ./demo-out/run
nightshift report ./demo-out/run
```

The demo runs at least two outer rounds with a frozen pack per round,
writes Inspect logs and an evidence-linked `report.md`, and verifies the
hash-chained ledger. See `docs/reproducibility.md` for the offline and
Docker details and `docs/demo-script.md` for a short narrated walkthrough.

## How a run is controlled

One TOML file describes a run: models, provider, budgets, data manifests,
verifier set, outer-loop limits, comparison design, and seeds. Budget
numbers have no defaults; a missing budget field is a configuration error.

```sh
nightshift init --demo            # write the CPU demo config
nightshift run nightshift.toml    # run a config
nightshift stop --leases <run>/leases.json
```

Safety properties the controller enforces in plain code:

- An aggregate dollar ceiling shared across run roots through one budget
  file; every GPU request is reserved (compute + setup + transfer +
  evaluation seconds at a declared price) *before* any provider create
  call, and rejected if it cannot fit the ceiling, the per-trial seconds,
  the rental lifetime, or the concurrency limit.
- Leases are written before provider create returns control to any agent,
  rentals carry a hard expiry, and cleanup destroys exactly the owned
  rental ids recorded — never a name pattern. `nightshift stop` can run
  from a different machine with only the lease file.
- Data arrives as frozen manifests (source, revision, row count, SHA-256).
  Researchers see training prompts and development feedback only; the
  selection and test splits, their paths, and per-item results never reach
  an agent. The test split is claimed exactly once per final comparison.
- Training targets come only from the local base model's own generations
  on approved training prompts, filtered by pinned verifiers. Hosted-model
  output is reasoning and code only and never becomes an SFT target.
- Submitted checkpoints must be safetensors weights plus JSON/YAML config;
  pickle formats, symlinks, path escapes, and malformed headers are
  rejected, and accepted checkpoints are stored by content hash.
- All scoring runs through Inspect AI. A failed evaluation records
  `status: failed` with no score — never an imputed zero — and failed
  trials stay in the comparison denominator.

## Agent backends

Researchers and the reviser can run through three kinds of backend, chosen
in the run config and recorded in every run record:

| Backend | Roles | Notes |
| --- | --- | --- |
| Deterministic fakes | researcher, reviser | CPU demo and tests; every artefact labelled fake |
| Muse | researcher `muse-spark-1.3`, reviser `muse-wonder-full-1.0` | Subscription login; not yet validated live in this repo |
| OpenAI-compatible API | researcher and/or reviser | Operator-configured base URL and model (for example a Muse Model API reviser `muse-wonder-1.0` with Baseten Muse Spark Open researchers `meta-models/muse-spark-open-1026`); fails closed without credentials; not yet validated live |

The backend is held constant inside a matched comparison, and requested
and served model ids are both recorded. See `docs/configuration.md` for
the `[agents]` table.

## Repository layout

- `nightshift/` — controller package: config, ledger, budget and leases,
  providers, data and verifiers, workspaces, packs, checkpoints,
  generation, training, Inspect evaluation, agents, loops, reporting, CLI.
- `nightshift/packs/default/` — the bundled researcher pack (instructions,
  Axolotl QLoRA recipe template, collaboration settings, tools, notes).
- `nightshift/demo_data/` — a tiny synthetic toy dataset written for the
  demo. No benchmark test questions are bundled.
- `tests/` — the CPU test suite covering the acceptance behaviours.
- `docker/Dockerfile.cpu` — the offline-capable CPU image (see
  `AGENTS.md` for the sandbox rules it runs under).
- `docs/` — reproducibility, configuration, demo script, product summary.

## Upstream work and licenses

Nightshift builds on published work without vendoring it: the CORAL
collaboration pattern (Human-Agent-Society/CORAL, revision
`0123dfb939b35228cf2c1fde224cd0561e727408`), WeCo's AIDE² outer method,
Inspect AI for evaluation, and Axolotl for real-mode QLoRA training.
See `THIRD_PARTY_NOTICES.md` for the pinned inventory. This repository is
released under the MIT License (`LICENSE`).
